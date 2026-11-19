"""Signature detection (YOLO11s) + izamscientist ConvNeXt-Tiny verification."""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image, ImageDraw, ImageFont, ImageOps
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


def _ink_mask_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """Dark / chromatic ink on paper → binary mask (255 = ink)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Colour ink (e.g. blue pen) is often mid-grey after RGB→L; boost via channel spread.
    spread = (
        np.max(rgb.astype(np.int16), axis=2) - np.min(rgb.astype(np.int16), axis=2)
    ).astype(np.uint8)
    inkish = cv2.max(cv2.bitwise_not(gray), spread)
    blur = cv2.GaussianBlur(inkish, (3, 3), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # If Otsu picked the paper (rare bright-ink case), invert.
    if np.mean(mask) > 127:
        mask = cv2.bitwise_not(mask)
    return mask


def _strip_ruled_lines(mask: np.ndarray) -> np.ndarray:
    """Remove long horizontal / vertical document lines from an ink mask."""
    h, w = mask.shape[:2]
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(20, w // 4), 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(16, h // 3))
    )
    horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel)
    cleaned = cv2.subtract(mask, horizontal)
    cleaned = cv2.subtract(cleaned, vertical)
    # Drop near-full-width row bands (print rules that survived morphology).
    row_frac = cleaned.sum(axis=1) / (255.0 * max(1, w))
    cleaned[row_frac > 0.45, :] = 0
    col_frac = cleaned.sum(axis=0) / (255.0 * max(1, h))
    cleaned[:, col_frac > 0.45] = 0
    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)


def _keep_signature_components(mask: np.ndarray) -> np.ndarray:
    """Keep the main handwriting blob(s); drop tiny label / noise components."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    h, w = mask.shape[:2]
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return mask
    order = np.argsort(areas)[::-1]
    keep = np.zeros_like(mask)
    largest = int(areas[order[0]])
    min_area = max(40, int(0.02 * largest))
    kept_area = 0
    for idx in order:
        area = int(areas[idx])
        if area < min_area:
            break
        i = int(idx) + 1
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        # Reject short wide text strips under the signature (e.g. "Signatur").
        if bh < max(10, 0.12 * h) and bw > 0.55 * w and area < 0.35 * largest:
            continue
        keep[labels == i] = 255
        kept_area += area
        # Signature strokes are usually one dominant blob plus a few accents.
        if kept_area >= 0.85 * largest and len(np.unique(keep)) > 1:
            # keep accumulating mid-sized stroke fragments
            pass
        if area < 0.08 * largest and kept_area > largest:
            break
    if keep.any():
        return keep
    # Fallback: largest component only
    keep[labels == int(order[0]) + 1] = 255
    return keep


def clean_signature_for_model(
    image: Image.Image | np.ndarray,
    *,
    aggressive_lines: bool = True,
) -> Image.Image:
    """
    Isolate handwriting → black ink on white paper (RGB).

    Runs before izamscientist preprocess so ruled lines / label text do not
    blank the crop or dominate the embedding.
    """
    rgb = _pil_to_rgb_array(image)
    mask = _ink_mask_from_rgb(rgb)
    if aggressive_lines:
        mask = _strip_ruled_lines(mask)
    mask = _keep_signature_components(mask)
    if int(mask.sum()) < 30 * 255:
        raise ValueError("No signature ink found after cleanup.")
    coords = cv2.findNonZero(mask)
    if coords is None:
        raise ValueError("No signature ink found after cleanup.")
    x, y, bw, bh = cv2.boundingRect(coords)
    if bw < 5 or bh < 3:
        raise ValueError("Detected signature is too small after cleanup.")
    pad = max(4, int(0.06 * max(bw, bh)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(rgb.shape[1], x + bw + pad), min(rgb.shape[0], y + bh + pad)
    crop_mask = mask[y0:y1, x0:x1]
    canvas = np.full(crop_mask.shape, 255, dtype=np.uint8)
    canvas[crop_mask > 0] = 0
    return Image.fromarray(np.stack([canvas, canvas, canvas], axis=-1), mode="RGB")


def _preview_from_model_preprocess(inference, image: Image.Image) -> Image.Image:
    """Run the model's own preprocess and return an RGB preview for Gradio."""
    image = ImageOps.exif_transpose(image)
    try:
        processed = inference.preprocess(image)  # PIL mode L
    except ValueError:
        # Fallback display if model preprocess rejects the crop
        gray = image.convert("L")
        if image.mode in ("RGBA", "LA", "P"):
            rgba = image.convert("RGBA")
            paper = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            paper.alpha_composite(rgba)
            gray = paper.convert("L")
        arr = np.asarray(ImageOps.autocontrast(gray), dtype=np.uint8)
        return Image.fromarray(np.stack([arr, arr, arr], axis=-1), mode="RGB")

    arr = np.asarray(processed.convert("L"), dtype=np.uint8)
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

    def preprocess_document(self, image: Image.Image | np.ndarray) -> Image.Image:
        """Clean document crop, then izamscientist preprocess → preview."""
        cleaned = clean_signature_for_model(image, aggressive_lines=True)
        return _preview_from_model_preprocess(self.inference, cleaned)

    def preprocess_pad(self, image: Image.Image | np.ndarray) -> Image.Image:
        """Clean specimen / pad image, then izamscientist preprocess → preview."""
        cleaned = clean_signature_for_model(image, aggressive_lines=True)
        return _preview_from_model_preprocess(self.inference, cleaned)

    def _compare_similarity(
        self, image_a: Image.Image, image_b: Image.Image
    ) -> float:
        """izamscientist score on [0, 1], with tensors on the model device."""
        inf = self.inference
        model = self.verifier
        arr_a = np.asarray(inf.preprocess(image_a), dtype=np.float32)
        arr_b = np.asarray(inf.preprocess(image_b), dtype=np.float32)
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
        return self._detect_multiscale_fallback(
            image, conf=min(conf, FALLBACK_CONF), iou=iou, classes=classes
        )

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
                    "crop": image.crop(crop_box),
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
    ) -> dict[str, Any]:
        """Clean both crops, then compare with izamscientist ConvNeXt-Tiny."""
        if threshold is None:
            threshold = self.verify_threshold_main

        try:
            cleaned_document = clean_signature_for_model(
                document_signature, aggressive_lines=True
            )
            cleaned_reference = clean_signature_for_model(
                reference_signature, aggressive_lines=True
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

        processed_document = _preview_from_model_preprocess(
            self.inference, cleaned_document
        )
        processed_reference = _preview_from_model_preprocess(
            self.inference, cleaned_reference
        )

        try:
            score = self._compare_similarity(cleaned_document, cleaned_reference)
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
        is_match = score >= float(threshold)
        decision = self.inference.decide(score)

        return {
            "distance": distance,
            "cosine_similarity": score,
            "threshold": float(threshold),
            "is_match": is_match,
            "decision": decision,
            "label": "GENUINE / MATCH" if is_match else "FORGED / NO MATCH",
            "processed_document": processed_document,
            "processed_reference": processed_reference,
        }
