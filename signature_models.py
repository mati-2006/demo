"""Signature detection (YOLO11s) + izamscientist ConvNeXt-Tiny verification."""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps
from ultralytics import YOLO

DETECTOR_REPO = "Mels22/Signature-Detection-Verification"
DETECTOR_FILE = "detector_yolo_1cls.pt"  # YOLO11s, 1 class: signature

# Verification: izamscientist ConvNeXt-Tiny similarity model
VERIFIER_REPO = "izamscientist/signature-similarity-convnext-tiny"

DEFAULT_DETECT_CONF = 0.25
DEFAULT_DETECT_IOU = 0.45
DEFAULT_DETECT_IMGSZ = 768
# Match when cosine-like similarity (0–1) >= threshold (model threshold_main ≈ 0.7785)
DEFAULT_VERIFY_THRESHOLD = 0.7785
SIGNATURE_CLASS_NAMES = {"signature"}
FALLBACK_IMGSZ = (640, 1024, 1280, 1536)
FALLBACK_CONF = 0.08
# Relative (xmin, ymin, xmax, ymax) ROIs for passport holder-signature bands.
# Tuned first for Republic of Iraq biodata page (thin ink above MRZ); also covers
# similar passport layouts when full-page YOLO misses.
# NOTE: keep ymax well above the MRZ (~y>0.74 on Iraq passports) or ink-fallback
# will latch onto OCR-B characters instead of the scribble.
PASSPORT_SIGNATURE_ROIS: tuple[tuple[float, float, float, float], ...] = (
    # Iraq: scribble to the right of "Holder's Signature" label
    (0.52, 0.58, 0.97, 0.66),
    # Iraq: wider band (label + ink) for YOLO context
    (0.34, 0.56, 0.98, 0.68),
    # Generic passport mid-lower band (above typical MRZ)
    (0.40, 0.54, 0.98, 0.70),
)
PASSPORT_ROI_DETECT_CONF = 0.05
PASSPORT_ROI_UPSCALE = 2.0
PASSPORT_ROI_CONTRAST = 1.8
# Dense printed text / MRZ rows exceed this dark-pixel fraction; signatures stay lower.
PASSPORT_ROI_MAX_INK_FRAC = 0.16


def _load_izam_inference_module():
    """Load inference.py shipped with izamscientist/signature-similarity-convnext-tiny."""
    root = snapshot_download(
        VERIFIER_REPO,
        allow_patterns=["inference.py", "model/*"],
    )
    module_path = f"{root}/inference.py"
    spec = importlib.util.spec_from_file_location("izam_signature_inference", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load inference module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pil_to_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    """EXIF-correct PIL/ndarray → RGB uint8 (transparent → white)."""
    if isinstance(image, Image.Image):
        image = ImageOps.exif_transpose(image)
        if image.mode in ("RGBA", "LA", "P"):
            rgba = image.convert("RGBA")
            paper = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            paper.alpha_composite(rgba)
            image = paper.convert("RGB")
        else:
            image = image.convert("RGB")
        return np.asarray(image, dtype=np.uint8)
    arr = np.asarray(image)
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1).astype(np.uint8)
    if arr.shape[2] == 4:
        rgb = arr[:, :, :3].astype(np.float32)
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        out = rgb * alpha + 255.0 * (1.0 - alpha)
        return np.clip(out, 0, 255).astype(np.uint8)
    return arr[:, :, :3].astype(np.uint8)


def _looks_like_clean_pad(rgb: np.ndarray) -> bool:
    """Draw-pad / scanned pad: mostly white paper, little texture."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(np.mean(gray > 245)) > 0.50 and float(np.mean(gray)) > 200


def _document_to_gray(rgb: np.ndarray) -> np.ndarray:
    """
    Grayscale for document crops.

    Black/blue pen stays dark; printed blue labels (e.g. PAN
    'हस्ताक्षर / Signature') are wiped toward white so they do not
    dominate the embedding.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.int16)
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    # Chromatic blue print (not dark black ink).
    blue_print = (b > r + 20) & (b > g + 12) & (b > 70) & (gray > 60)
    gray = gray.copy()
    gray[blue_print] = 255
    # Slight boost for dark chromatic ink (blue pen).
    spread = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    dark = gray < 180
    gray[dark] = np.clip(gray[dark] - spread[dark] // 3, 0, 255)
    return gray.astype(np.uint8)


def _pad_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Grayscale for draw-pad / clean specimen (ink already dark on white)."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _soft_contrast(gray: np.ndarray) -> np.ndarray:
    """Gentle percentile stretch — keeps anti-aliased stroke edges."""
    lo, hi = np.percentile(gray.astype(np.float32), (2, 98))
    if hi - lo < 8:
        return gray
    out = (gray.astype(np.float32) - lo) / (hi - lo) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _ink_mask_soft(gray: np.ndarray) -> np.ndarray:
    """Otsu ink mask for bbox work only — not used as final pixels."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def _trim_bottom_label(gray: np.ndarray) -> np.ndarray:
    """
    Drop the bottom caption band on ID crops ('Signature' under the ink).

    Handwriting almost always sits in the upper portion of the YOLO box.
    """
    h = gray.shape[0]
    if h < 40:
        return gray
    # Keep upper ~72%; label text is consistently in the bottom strip.
    return gray[: max(20, int(h * 0.72)), :]


def _drop_label_components(mask: np.ndarray) -> np.ndarray:
    """Drop tiny speckles and short wide label strips; keep stroke fragments."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    h, w = mask.shape[:2]
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return mask
    largest = int(areas.max())
    min_area = max(12, int(0.008 * largest))
    keep = np.zeros_like(mask)
    for i, area in enumerate(areas, start=1):
        area_i = int(area)
        if area_i < min_area:
            continue
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        cy = stats[i, cv2.CC_STAT_TOP] + bh / 2
        # Printed captions under the signature.
        if bh < max(8, 0.12 * h) and bw > 0.50 * w and area_i < 0.35 * largest:
            continue
        # Anything hugging the bottom edge that is short.
        if cy > 0.82 * h and bh < 0.20 * h and area_i < 0.40 * largest:
            continue
        keep[labels == i] = 255
    return keep if keep.any() else mask


def _inpaint_thin_box_lines(gray: np.ndarray) -> np.ndarray:
    """
    Remove only thin full-span box/rule lines.

    Does NOT use short horizontal kernels — those destroy signature ovals
    (long curved strokes were being inpainted as 'ruled lines').
    """
    h, w = gray.shape[:2]
    # Tiny ID crops: skipping is safer than shredding handwriting.
    if h < 90 or w < 120:
        return gray

    inv = _ink_mask_soft(gray)
    # Require nearly full-width / full-height runs so oval arcs are ignored.
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(60, int(0.75 * w)), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(50, int(0.75 * h))))
    h_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, hk)
    v_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, vk)
    line_mask = cv2.bitwise_or(h_lines, v_lines)
    if not line_mask.any():
        return gray

    # Keep only geometrically thin line pixels (ruled ink), not thick strokes.
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    line_mask = np.where((line_mask > 0) & (dist <= 1.6), 255, 0).astype(np.uint8)
    if not line_mask.any():
        return gray
    line_mask = cv2.dilate(line_mask, np.ones((2, 2), np.uint8), iterations=1)
    return cv2.inpaint(gray, line_mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)


def _crop_gray_to_ink(gray: np.ndarray, *, pad_frac: float = 0.06) -> np.ndarray:
    """Crop to ink bbox but keep original gray values (no hard 0/255 canvas)."""
    mask = _drop_label_components(_ink_mask_soft(gray))
    if int(mask.sum()) < 30 * 255:
        raise ValueError("No signature ink found after cleanup.")
    coords = cv2.findNonZero(mask)
    if coords is None:
        raise ValueError("No signature ink found after cleanup.")
    x, y, bw, bh = cv2.boundingRect(coords)
    if bw < 5 or bh < 3:
        raise ValueError("Detected signature is too small after cleanup.")
    pad = max(4, int(pad_frac * max(bw, bh)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(gray.shape[1], x + bw + pad), min(gray.shape[0], y + bh + pad)
    crop = gray[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]
    crop_mask = cv2.dilate(crop_mask, np.ones((3, 3), np.uint8), iterations=1)
    paper = np.full_like(crop, 255)
    paper[crop_mask > 0] = crop[crop_mask > 0]
    paper = np.where(paper > 230, 255, paper).astype(np.uint8)
    return paper


def _thicken_if_thin(gray: np.ndarray) -> np.ndarray:
    """Mild 1px thicken for sparse draw-pad strokes only."""
    ink = gray < 200
    if not ink.any():
        return gray
    if float(ink.mean()) >= 0.045:
        return gray
    return cv2.erode(gray, np.ones((2, 2), np.uint8), iterations=1)


def _pad_to_photo_domain(gray: np.ndarray) -> np.ndarray:
    """
    Soften pure digital pad ink so it is closer to photographed pen strokes.

    Without this, draw-vs-document pairs often land in a false mid-band
    (~0.55–0.70) even for different writers.
    """
    soft = cv2.GaussianBlur(gray, (3, 3), 0.9)
    # Stable micro-texture (deterministic from image moments — no RNG flicker).
    yy, xx = np.mgrid[0 : soft.shape[0], 0 : soft.shape[1]]
    texture = ((xx * 17 + yy * 29) % 7).astype(np.float32) - 3.0
    out = soft.astype(np.float32) + texture * 0.8
    return np.clip(out, 0, 255).astype(np.uint8)


def clean_signature_for_model(
    image: Image.Image | np.ndarray,
    *,
    aggressive_lines: bool = True,
    source: str = "auto",
) -> Image.Image:
    """
    Prepare a signature for izamscientist preprocess.

    Preserves stroke geometry (no skeletonize). Document path suppresses
    blue printed labels and trims the bottom caption band; pad path
    soft-normalizes and mild photo-domain adaptation.
    """
    rgb = _pil_to_rgb_array(image)

    if source == "pad":
        use_pad = True
    elif source == "document":
        use_pad = False
    else:
        use_pad = _looks_like_clean_pad(rgb)
    _ = aggressive_lines

    if use_pad:
        gray = _pad_to_gray(rgb)
        gray = _soft_contrast(gray)
        gray = _thicken_if_thin(gray)
        gray = _pad_to_photo_domain(gray)
    else:
        gray = _document_to_gray(rgb)
        gray = _trim_bottom_label(gray)
        gray = _inpaint_thin_box_lines(gray)
        gray = _soft_contrast(gray)

    cropped = _crop_gray_to_ink(gray)
    return Image.fromarray(
        np.stack([cropped, cropped, cropped], axis=-1), mode="RGB"
    )


def refine_detection_crop(crop: Image.Image) -> Image.Image:
    """Tighten a YOLO crop: drop bottom 'Signature' caption, then ink bbox."""
    rgb = _pil_to_rgb_array(crop)
    h, w = rgb.shape[:2]
    # First cut the printed caption strip under handwriting.
    trimmed = rgb[: max(20, int(h * 0.72)), :, :]
    gray = _document_to_gray(trimmed)
    mask = _drop_label_components(_ink_mask_soft(gray))
    coords = cv2.findNonZero(mask)
    if coords is None:
        return Image.fromarray(trimmed, mode="RGB")
    x, y, bw, bh = cv2.boundingRect(coords)
    if bw < 5 or bh < 3:
        return Image.fromarray(trimmed, mode="RGB")
    pad = max(4, int(0.08 * max(bw, bh)))
    th, tw = trimmed.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(tw, x + bw + pad), min(th, y + bh + pad)
    return Image.fromarray(trimmed[y0:y1, x0:x1], mode="RGB")


def refine_passport_roi_crop(crop: Image.Image) -> Image.Image:
    """Tighten a passport signature-band ROI; prefer stroke ink over printed labels."""
    rgb = _pil_to_rgb_array(crop)
    h, w = rgb.shape[:2]
    gray = _document_to_gray(rgb)
    gray = _inpaint_thin_box_lines(gray)
    mask = _drop_label_components(_ink_mask_soft(gray))
    # Prefer right-side stroke clusters (Iraq: label is left, scribble is right).
    if w >= 80:
        left_cut = int(0.28 * w)
        right_mask = mask.copy()
        right_mask[:, :left_cut] = 0
        if int(right_mask.sum()) >= 40 * 255:
            mask = right_mask
    coords = cv2.findNonZero(mask)
    if coords is None:
        return Image.fromarray(rgb, mode="RGB")
    x, y, bw, bh = cv2.boundingRect(coords)
    if bw < 5 or bh < 3:
        return Image.fromarray(rgb, mode="RGB")
    pad = max(4, int(0.08 * max(bw, bh)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    return Image.fromarray(rgb[y0:y1, x0:x1], mode="RGB")


def _rel_roi_to_pixels(
    image: Image.Image, roi: tuple[float, float, float, float]
) -> tuple[int, int, int, int]:
    xmin_r, ymin_r, xmax_r, ymax_r = roi
    w, h = image.size
    return (
        max(0, int(xmin_r * w)),
        max(0, int(ymin_r * h)),
        min(w, int(xmax_r * w)),
        min(h, int(ymax_r * h)),
    )


def _enhance_roi_for_detection(crop: Image.Image) -> Image.Image:
    """Boost contrast so thin passport ink stands out from guilloche."""
    enhanced = ImageOps.autocontrast(crop.convert("RGB"))
    return ImageEnhance.Contrast(enhanced).enhance(PASSPORT_ROI_CONTRAST)


def _roi_dark_ink_fraction(crop: Image.Image) -> float:
    gray = _document_to_gray(_pil_to_rgb_array(crop))
    return float((gray < 100).mean())


def _ink_is_mostly_black_stroke(crop: Image.Image) -> bool:
    """Reject teal/green printed labels; keep near-black pen strokes."""
    rgb = _pil_to_rgb_array(crop)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ink = gray < 130
    if int(ink.sum()) < 40:
        return False
    pixels = rgb[ink].astype(np.int16)
    chroma = pixels.max(axis=1) - pixels.min(axis=1)
    # Printed passport labels are often chromatic (green/teal); pen ink is neutral.
    return float(np.mean(chroma < 45)) >= 0.55


def _roi_has_signature_ink(crop: Image.Image) -> bool:
    """True when crop has sparse stroke-like ink (not dense MRZ / printed lines)."""
    gray = _document_to_gray(_pil_to_rgb_array(crop))
    mask = _ink_mask_soft(gray)
    if int(mask.sum()) < 80 * 255:
        return False
    # MRZ / OCR-B rows are much denser than a thin handwritten scribble.
    if _roi_dark_ink_fraction(crop) > PASSPORT_ROI_MAX_INK_FRAC:
        return False
    if not _ink_is_mostly_black_stroke(crop):
        return False
    return True


def _to_gray_on_white(image: Image.Image) -> Image.Image:
    """Transparent → white paper, then grayscale (matches izamscientist)."""
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA", "P"):
        rgba = image.convert("RGBA")
        paper = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        paper.alpha_composite(rgba)
        image = paper
    return image.convert("L")


def _zero_thin_fullspan_rules(
    ink: np.ndarray,
    *,
    span_frac: float = 0.75,
    max_thickness: int = 3,
) -> np.ndarray:
    """
    Remove only thin ruled box lines (near full-span AND ≤ max_thickness).

    Unlike izamscientist's row wipe (ink.sum/w > 0.4 → wipe entire row), this
    keeps thick handwritten strokes that span much of the width/height.
    """
    out = ink.copy()
    h, w = out.shape

    def _clear_thin_runs(wide: np.ndarray, axis: int) -> None:
        n = wide.shape[0]
        i = 0
        while i < n:
            if not wide[i]:
                i += 1
                continue
            j = i
            while j < n and wide[j]:
                j += 1
            if (j - i) <= max_thickness:
                if axis == 0:
                    out[i:j, :] = False
                else:
                    out[:, i:j] = False
            i = j

    row_frac = out.sum(axis=1) / max(1, w)
    col_frac = out.sum(axis=0) / max(1, h)
    _clear_thin_runs(row_frac >= span_frac, axis=0)
    _clear_thin_runs(col_frac >= span_frac, axis=1)
    return out


def _upscale_min_side(image: Image.Image, min_side: int) -> Image.Image:
    """Upscale small crops before 224 resize so LANCZOS does not dissolve strokes."""
    w, h = image.size
    m = min(w, h)
    if m <= 0 or m >= min_side:
        return image
    scale = min_side / float(m)
    return image.resize(
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        Image.Resampling.LANCZOS,
    )


def preprocess_for_verifier(
    image: Image.Image,
    *,
    image_size: int = 224,
    remove_ruled_lines: bool = True,
    min_side_before_resize: int = 160,
) -> Image.Image:
    """
    Verifier input pipeline (gray → contrast → ink crop → square pad → resize).

    Drop-in replacement for izamscientist ``preprocess`` that does **not** wipe
    thick signature strokes. Square padding avoids aspect-ratio distortion so
    resize preserves geometry for scoring.
    """
    gray = _to_gray_on_white(image)
    gray = _upscale_min_side(gray, min_side_before_resize)
    a = np.asarray(gray, dtype=np.float32)
    if float(np.median(a)) < 127.0:
        a = 255.0 - a
    lo, hi = np.percentile(a, 1), np.percentile(a, 90)
    if hi - lo < 10:
        raise ValueError("picture is blank")
    a = np.clip((a - lo) / (hi - lo) * 255.0, 0, 255)
    ink = a < 160
    if remove_ruled_lines:
        ink = _zero_thin_fullspan_rules(ink)
    if int(ink.sum()) < 30:
        raise ValueError("picture is blank")
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    pad = max(2, int(0.04 * max(int(rows[-1] - rows[0]), int(cols[-1] - cols[0]))))
    r0, r1 = max(int(rows[0]) - pad, 0), min(int(rows[-1]) + pad + 1, a.shape[0])
    c0, c1 = max(int(cols[0]) - pad, 0), min(int(cols[-1]) + pad + 1, a.shape[1])
    crop = np.where(ink[r0:r1, c0:c1], a[r0:r1, c0:c1], 255.0)
    ch, cw = crop.shape
    side = max(ch, cw)
    canvas = np.full((side, side), 255.0, dtype=np.float32)
    y0, x0 = (side - ch) // 2, (side - cw) // 2
    canvas[y0 : y0 + ch, x0 : x0 + cw] = crop
    return Image.fromarray(canvas.astype(np.uint8), mode="L").resize(
        (int(image_size), int(image_size)), Image.Resampling.LANCZOS
    )


def _preview_from_model_preprocess(
    image: Image.Image,
    *,
    image_size: int = 224,
    remove_ruled_lines: bool = True,
) -> Image.Image:
    """Run verifier preprocess and return an RGB preview for Gradio."""
    try:
        processed = preprocess_for_verifier(
            image,
            image_size=image_size,
            remove_ruled_lines=remove_ruled_lines,
        )
    except ValueError:
        gray = _to_gray_on_white(image)
        arr = np.asarray(ImageOps.autocontrast(gray), dtype=np.uint8)
        return Image.fromarray(np.stack([arr, arr, arr], axis=-1), mode="RGB")

    arr = np.asarray(processed, dtype=np.uint8)
    return Image.fromarray(np.stack([arr, arr, arr], axis=-1), mode="RGB")


def _box_iou(a: dict[str, int], b: dict[str, int]) -> float:
    ax1, ay1, ax2, ay2 = a["xmin"], a["ymin"], a["xmax"], a["ymax"]
    bx1, by1, bx2, by2 = b["xmin"], b["ymin"], b["xmax"], b["ymax"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _is_plausible_signature_box(
    box: dict[str, int], image_w: int, image_h: int
) -> bool:
    w = box["xmax"] - box["xmin"]
    h = box["ymax"] - box["ymin"]
    if w < 24 or h < 12:
        return False
    aspect = w / max(1, h)
    area_frac = (w * h) / max(1, image_w * image_h)
    cy = (box["ymin"] + box["ymax"]) / 2 / max(1, image_h)
    if aspect < 1.25 and area_frac < 0.025 and cy < 0.45:
        return False
    if 0.75 <= aspect <= 1.25 and area_frac > 0.02:
        return False
    if area_frac > 0.25:
        return False
    return True


def _nms_detections(
    detections: list[dict[str, Any]], iou_thresh: float = 0.35
) -> list[dict[str, Any]]:
    ordered = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept: list[dict[str, Any]] = []
    for det in ordered:
        if any(_box_iou(det["box"], other["box"]) > iou_thresh for other in kept):
            continue
        kept.append(det)
    return kept


class SignaturePipeline:
    """YOLO11s detection + izamscientist/signature-similarity-convnext-tiny verification."""

    def __init__(self) -> None:
        self._detector: YOLO | None = None
        self._verifier = None
        self._inference = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def detector(self) -> YOLO:
        if self._detector is None:
            model_path = hf_hub_download(
                repo_id=DETECTOR_REPO,
                filename=DETECTOR_FILE,
            )
            self._detector = YOLO(model_path)
        return self._detector

    @property
    def inference(self):
        if self._inference is None:
            self._inference = _load_izam_inference_module()
        return self._inference

    @property
    def verifier(self):
        if self._verifier is None:
            model = self.inference.load_model()
            model.to(self.device)
            model.eval()
            self._verifier = model
        return self._verifier

    @property
    def verify_threshold_main(self) -> float:
        return float(self.inference.CFG.get("threshold_main", DEFAULT_VERIFY_THRESHOLD))

    @property
    def verifier_image_size(self) -> int:
        return int(self.inference.CFG.get("image_size", 224))

    def preprocess_document(self, image: Image.Image | np.ndarray) -> Image.Image:
        """Clean document crop, then verifier preprocess → preview."""
        cleaned = clean_signature_for_model(
            image, aggressive_lines=True, source="document"
        )
        return _preview_from_model_preprocess(
            cleaned,
            image_size=self.verifier_image_size,
            remove_ruled_lines=True,
        )

    def preprocess_pad(self, image: Image.Image | np.ndarray) -> Image.Image:
        """Clean specimen / pad image, then verifier preprocess → preview."""
        cleaned = clean_signature_for_model(
            image, aggressive_lines=False, source="pad"
        )
        return _preview_from_model_preprocess(
            cleaned,
            image_size=self.verifier_image_size,
            remove_ruled_lines=False,
        )

    def _compare_similarity(
        self,
        image_a: Image.Image,
        image_b: Image.Image,
        *,
        a_remove_ruled_lines: bool = True,
        b_remove_ruled_lines: bool = True,
    ) -> float:
        """izamscientist score on [0, 1], with tensors on the model device."""
        inf = self.inference
        model = self.verifier
        size = self.verifier_image_size
        arr_a = np.asarray(
            preprocess_for_verifier(
                image_a,
                image_size=size,
                remove_ruled_lines=a_remove_ruled_lines,
            ),
            dtype=np.float32,
        )
        arr_b = np.asarray(
            preprocess_for_verifier(
                image_b,
                image_size=size,
                remove_ruled_lines=b_remove_ruled_lines,
            ),
            dtype=np.float32,
        )
        x = torch.from_numpy(np.stack([arr_a, arr_b]) / 255.0)
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)
        mean = torch.tensor(inf.CFG["normalisation_mean"]).view(1, 3, 1, 1)
        std = torch.tensor(inf.CFG["normalisation_std"]).view(1, 3, 1, 1)
        x = (x - mean) / std
        x = x.to(self.device)
        ea, eb = model.embed(x[0:1]), model.embed(x[1:2])
        return float(((ea * eb).sum(1) + 1).item() / 2)

    def detect_signatures(
        self,
        image: Image.Image,
        conf: float = DEFAULT_DETECT_CONF,
        iou: float = DEFAULT_DETECT_IOU,
        imgsz: int = DEFAULT_DETECT_IMGSZ,
        classes: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        image = ImageOps.exif_transpose(image).convert("RGB")
        detections = self._detect_once(
            image, conf=conf, iou=iou, imgsz=imgsz, classes=classes
        )
        if detections:
            return detections
        detections = self._detect_multiscale_fallback(
            image, conf=min(conf, FALLBACK_CONF), iou=iou, classes=classes
        )
        if detections:
            return detections
        # Iraqi / similar passports: thin ink on guilloche often missed full-page.
        return self._detect_passport_roi_fallback(
            image, conf=min(conf, PASSPORT_ROI_DETECT_CONF), iou=iou, classes=classes
        )

    def _detect_passport_roi_fallback(
        self,
        image: Image.Image,
        conf: float = PASSPORT_ROI_DETECT_CONF,
        iou: float = DEFAULT_DETECT_IOU,
        classes: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Crop passport signature bands, enhance, re-detect; else ink-ROI crop."""
        candidates: list[dict[str, Any]] = []
        synthetic: list[dict[str, Any]] = []

        for roi_rel in PASSPORT_SIGNATURE_ROIS:
            abs_box = _rel_roi_to_pixels(image, roi_rel)
            x0, y0, x1, y1 = abs_box
            if x1 - x0 < 40 or y1 - y0 < 16:
                continue

            region = image.crop(abs_box)
            if not _roi_has_signature_ink(region):
                continue

            enhanced = _enhance_roi_for_detection(region)
            scale = PASSPORT_ROI_UPSCALE
            up_w = max(1, int(enhanced.width * scale))
            up_h = max(1, int(enhanced.height * scale))
            upscaled = enhanced.resize((up_w, up_h), Image.Resampling.LANCZOS)

            for size in (768, 1024, 1280):
                local = self._detect_once(
                    upscaled,
                    conf=conf,
                    iou=iou,
                    imgsz=size,
                    classes=classes,
                )
                for det in local:
                    box = det["box"]
                    # Map from upscaled-ROI coords → full-image coords.
                    fx0 = x0 + int(box["xmin"] / scale)
                    fy0 = y0 + int(box["ymin"] / scale)
                    fx1 = x0 + int(box["xmax"] / scale)
                    fy1 = y0 + int(box["ymax"] / scale)
                    fx0, fy0 = max(0, fx0), max(0, fy0)
                    fx1, fy1 = min(image.width, fx1), min(image.height, fy1)
                    if fx1 - fx0 < 12 or fy1 - fy0 < 6:
                        continue
                    raw_crop = image.crop((fx0, fy0, fx1, fy1))
                    if not _roi_has_signature_ink(raw_crop):
                        continue
                    candidates.append(
                        {
                            "score": float(det["score"]),
                            "class_id": int(det["class_id"]),
                            "label": det["label"],
                            "box": {
                                "xmin": fx0,
                                "ymin": fy0,
                                "xmax": fx1,
                                "ymax": fy1,
                            },
                            "crop": refine_passport_roi_crop(raw_crop),
                            "source": "passport_roi_yolo",
                        }
                    )

            # Last resort for this ROI: use refined ink crop of the band itself.
            refined = refine_passport_roi_crop(region)
            if _roi_has_signature_ink(refined) and _roi_has_signature_ink(region):
                synthetic.append(
                    {
                        "score": 0.01,
                        "class_id": 0,
                        "label": "signature",
                        "box": {
                            "xmin": x0,
                            "ymin": y0,
                            "xmax": x1,
                            "ymax": y1,
                        },
                        "crop": refined,
                        "source": "passport_roi_ink",
                    }
                )

        if candidates:
            plausible = [
                d
                for d in candidates
                if _is_plausible_signature_box(d["box"], image.width, image.height)
            ]
            kept = _nms_detections(plausible or candidates, iou_thresh=0.35)
            if kept:
                return kept

        return synthetic[:1]

    def _detect_once(
        self,
        image: Image.Image,
        conf: float,
        iou: float,
        imgsz: int,
        classes: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        results = self.detector(
            image,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            classes=classes,
            verbose=False,
            device=str(self.device),
        )
        detections: list[dict[str, Any]] = []
        if not results:
            return detections

        result = results[0]
        names = result.names or self.detector.names or {}
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        xyxy = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, score, cls_id in zip(xyxy, scores, cls_ids):
            label = str(names.get(int(cls_id), cls_id)).lower()
            if label not in SIGNATURE_CLASS_NAMES and int(cls_id) != 0:
                continue

            xmin, ymin, xmax, ymax = (int(v) for v in box)
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(image.width, xmax), min(image.height, ymax)
            if xmax <= xmin or ymax <= ymin:
                continue

            pad = max(2, min(8, (xmax - xmin) // 20, (ymax - ymin) // 10))
            crop_box = (
                max(0, xmin - pad),
                max(0, ymin - pad),
                min(image.width, xmax + pad),
                min(image.height, ymax + pad),
            )
            raw_crop = image.crop(crop_box)
            detections.append(
                {
                    "score": float(score),
                    "class_id": int(cls_id),
                    "label": label,
                    "box": {
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                    },
                    # Store a tightened crop (drops PAN 'Signature' caption).
                    "crop": refine_detection_crop(raw_crop),
                }
            )

        detections.sort(key=lambda d: d["score"], reverse=True)
        return detections

    def _detect_multiscale_fallback(
        self,
        image: Image.Image,
        conf: float,
        iou: float,
        classes: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for size in FALLBACK_IMGSZ:
            candidates.extend(
                self._detect_once(
                    image, conf=conf, iou=iou, imgsz=size, classes=classes
                )
            )
        plausible = [
            d
            for d in candidates
            if _is_plausible_signature_box(d["box"], image.width, image.height)
        ]
        return _nms_detections(plausible, iou_thresh=0.35)

    def draw_detections(
        self,
        image: Image.Image,
        detections: list[dict[str, Any]],
    ) -> Image.Image:
        annotated = image.convert("RGB").copy()
        draw = ImageDraw.Draw(annotated)
        try:
            font = ImageFont.truetype("Arial.ttf", 16)
        except OSError:
            font = ImageFont.load_default()

        for i, det in enumerate(detections, start=1):
            box = det["box"]
            xy = (box["xmin"], box["ymin"], box["xmax"], box["ymax"])
            draw.rectangle(xy, outline="#0b6e4f", width=3)
            label = f"#{i}  {det['score']:.2f}"
            text_y = max(0, box["ymin"] - 20)
            draw.rectangle(
                (box["xmin"], text_y, box["xmin"] + 90, text_y + 18),
                fill="#0b6e4f",
            )
            draw.text((box["xmin"] + 4, text_y + 2), label, fill="white", font=font)
        return annotated

    @torch.no_grad()
    def verify(
        self,
        document_signature: Image.Image,
        reference_signature: Image.Image,
        threshold: float | None = None,
        reference_source: str = "auto",
    ) -> dict[str, Any]:
        """Clean both crops, then compare with izamscientist ConvNeXt-Tiny.

        reference_source: "pad" for Draw, "document" for photo crops, or "auto".
        """
        if threshold is None:
            threshold = self.verify_threshold_main

        try:
            cleaned_document = clean_signature_for_model(
                document_signature, aggressive_lines=True, source="document"
            )
            cleaned_reference = clean_signature_for_model(
                reference_signature, source=reference_source
            )
        except ValueError as exc:
            # Do not pretend similarity is 0 — cleanup failed before scoring.
            fallback_doc = document_signature
            fallback_ref = reference_signature
            if isinstance(fallback_doc, Image.Image):
                fallback_doc = ImageOps.exif_transpose(fallback_doc).convert("RGB")
            if isinstance(fallback_ref, Image.Image):
                fallback_ref = ImageOps.exif_transpose(fallback_ref).convert("RGB")
            return {
                "distance": float("nan"),
                "cosine_similarity": float("nan"),
                "threshold": float(threshold),
                "is_match": False,
                "decision": f"ERROR: {exc}",
                "label": "COULD NOT SCORE",
                "error": str(exc),
                "processed_document": fallback_doc,
                "processed_reference": fallback_ref,
            }

        ref_is_pad = reference_source == "pad" or (
            reference_source == "auto"
            and _looks_like_clean_pad(_pil_to_rgb_array(reference_signature))
        )
        processed_document = _preview_from_model_preprocess(
            cleaned_document,
            image_size=self.verifier_image_size,
            remove_ruled_lines=True,
        )
        processed_reference = _preview_from_model_preprocess(
            cleaned_reference,
            image_size=self.verifier_image_size,
            remove_ruled_lines=not ref_is_pad,
        )

        try:
            score = self._compare_similarity(
                cleaned_document,
                cleaned_reference,
                a_remove_ruled_lines=True,
                b_remove_ruled_lines=not ref_is_pad,
            )
        except ValueError as exc:
            return {
                "distance": float("nan"),
                "cosine_similarity": float("nan"),
                "threshold": float(threshold),
                "is_match": False,
                "decision": f"ERROR: {exc}",
                "label": "COULD NOT SCORE",
                "error": str(exc),
                "processed_document": processed_document,
                "processed_reference": processed_reference,
            }

        score = max(0.0, min(1.0, score))
        cosine = 2.0 * score - 1.0
        distance = float((2.0 - 2.0 * cosine) ** 0.5)

        # Use the caller/UI threshold as-is (no pad-specific floor).
        effective_threshold = float(threshold)
        is_match = score >= effective_threshold
        decision = self.inference.decide(score)

        return {
            "distance": distance,
            "cosine_similarity": score,
            "threshold": effective_threshold,
            "is_match": is_match,
            "decision": decision,
            "label": "GENUINE / MATCH" if is_match else "FORGED / NO MATCH",
            "processed_document": processed_document,
            "processed_reference": processed_reference,
        }
