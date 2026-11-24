"""Gradio demo: detect signatures on a document, then verify against a reference."""

from __future__ import annotations

import gradio as gr
from PIL import Image

from signature_models import (
    DEFAULT_DETECT_CONF,
    DEFAULT_DETECT_IMGSZ,
    DEFAULT_DETECT_IOU,
    DEFAULT_VERIFY_THRESHOLD,
    SignaturePipeline,
)

pipeline = SignaturePipeline()
# Document toolbar: file · webcam · clipboard
IMAGE_SOURCES = ["upload", "clipboard", "webcam"]
IMAGE_PLACEHOLDER = (
    "# Add an image\n"
    "Drop a file, paste from clipboard, or use the webcam"
)
# Specimen sources (Verify tab): Draw pad · Upload · Clipboard
SPECIMEN_MODES = ["Draw", "Upload", "Clipboard"]
# Common eKYC-style signature pad canvas (width × height)
SPECIMEN_PAD_SIZE = (600, 200)
SPECIMEN_UPLOAD_PLACEHOLDER = (
    "# Specimen image\n"
    "Drop a file or paste from clipboard"
)


def detect(
    document: Image.Image | None,
    conf: float,
    iou: float,
    imgsz: int,
):
    if document is None:
        raise gr.Error("Upload or paste a document image first.")

    detections = pipeline.detect_signatures(
        document,
        conf=conf,
        iou=iou,
        imgsz=int(imgsz),
    )
    annotated = pipeline.draw_detections(document, detections)

    if not detections:
        return (
            annotated,
            [],
            [],
            _status_banner(
                "No signatures found",
                "Try lowering confidence, raising image size, or use a sharper scan.",
                tone="warn",
            ),
            gr.update(choices=[], value=None),
            None,
            None,
            _idle_result(),
        )

    gallery = [
        (d["crop"], f"#{i} · {d['score']:.0%}") for i, d in enumerate(detections, 1)
    ]
    choices = [f"Signature #{i} ({d['score']:.0%})" for i, d in enumerate(detections, 1)]
    summary = _status_banner(
        f"{len(detections)} signature(s) detected",
        "Open Verify, choose a crop, add a specimen, then run comparison.",
        tone="ok",
    )
    return (
        annotated,
        detections,
        gallery,
        summary,
        gr.update(choices=choices, value=choices[0]),
        detections[0]["crop"],
        None,
        _idle_result(),
    )


def _image_has_ink(image: Image.Image) -> bool:
    """True if the image has non-empty ink (not a blank pad)."""
    if image.mode in ("RGBA", "LA"):
        alpha = image.getchannel("A")
        if alpha.getextrema()[1] == 0:
            return False
        # Ink = non-transparent pixels that are not near-white
        rgba = image.convert("RGBA")
        pixels = list(rgba.getdata())
        ink = sum(
            1
            for r, g, b, a in pixels
            if a > 10 and (r < 245 or g < 245 or b < 245)
        )
        return ink >= 30
    gray = image.convert("L")
    # Count dark-ish pixels
    hist = gray.histogram()
    return sum(hist[:200]) >= 30


def resolve_specimen(
    mode: str | None,
    draw_value,
    file_value: Image.Image | None,
) -> Image.Image | None:
    """Pick the active specimen from Draw pad or Upload/Clipboard image."""
    if mode == "Draw":
        if draw_value is None:
            return None
        if isinstance(draw_value, dict):
            image = draw_value.get("composite")
        else:
            image = draw_value
        if image is None:
            return None
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if not _image_has_ink(image):
            return None
        return image
    return file_value


def switch_specimen_mode(mode: str):
    """Show Draw pad or Upload/Clipboard image; sync Image sources to the mode."""
    if mode == "Draw":
        return (
            gr.update(visible=True),
            gr.update(visible=False),
        )
    sources = ["upload"] if mode == "Upload" else ["clipboard"]
    placeholder = (
        "# Upload specimen\nDrop or browse a signature image"
        if mode == "Upload"
        else "# Paste specimen\nClick Paste, then Ctrl/Cmd+V"
    )
    return (
        gr.update(visible=False),
        gr.update(visible=True, sources=sources, placeholder=placeholder),
    )


def verify(
    detections: list | None,
    selected: str | None,
    specimen_mode: str | None,
    draw_value,
    file_value: Image.Image | None,
    verify_threshold: float,
):
    if not detections:
        raise gr.Error("Run detection first so there is a document signature to compare.")
    reference = resolve_specimen(specimen_mode, draw_value, file_value)
    if reference is None:
        raise gr.Error("Add a specimen via Draw, Upload, or Clipboard.")
    if not selected:
        raise gr.Error("Select which detected signature to verify.")

    try:
        index = int(selected.split("#")[1].split()[0]) - 1
    except (IndexError, ValueError) as exc:
        raise gr.Error("Could not parse the selected signature.") from exc

    if index < 0 or index >= len(detections):
        raise gr.Error("Selected signature is out of range. Re-run detection.")

    document_crop = detections[index]["crop"]
    # Draw pad must use the pad cleanup path (not document line-inpaint).
    ref_source = "pad" if specimen_mode == "Draw" else "auto"
    try:
        result = pipeline.verify(
            document_crop,
            reference,
            threshold=verify_threshold,
            reference_source=ref_source,
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    return (
        result["processed_document"],
        result["processed_reference"],
        _result_card(result),
    )


def preview_selected(detections: list | None, selected: str | None):
    if not detections or not selected:
        return None
    try:
        index = int(selected.split("#")[1].split()[0]) - 1
    except (IndexError, ValueError):
        return None
    if index < 0 or index >= len(detections):
        return None
    try:
        return pipeline.preprocess_document(detections[index]["crop"])
    except ValueError:
        return detections[index]["crop"]


def preview_specimen(
    mode: str | None,
    draw_value,
    file_value: Image.Image | None,
):
    """Live specimen preview from the active Draw / Upload / Clipboard source."""
    image = resolve_specimen(mode, draw_value, file_value)
    if image is None:
        return None
    try:
        return pipeline.preprocess_pad(image)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


def _status_banner(title: str, detail: str, tone: str = "neutral") -> str:
    styles = {
        "ok": ("#ecfdf5", "#065f46", "#6ee7b7"),
        "warn": ("#fff7ed", "#9a3412", "#fdba74"),
        "neutral": ("#f1f5f9", "#334155", "#cbd5e1"),
    }
    bg, text, border = styles.get(tone, styles["neutral"])
    return f"""
    <div class="status-banner" style="background:{bg};color:{text};border-color:{border};">
      <strong>{title}</strong>
      <span>{detail}</span>
    </div>
    """


def _idle_result() -> str:
    return """
    <div class="result-idle">
      <strong>Ready to verify</strong>
      <span>Select a detected signature, add a specimen (draw · upload · clipboard), then verify.</span>
    </div>
    """


def _result_card(result: dict) -> str:
    error = result.get("error")
    matched = result["is_match"]
    if error:
        accent, bg, border, text = "#b45309", "#fffbeb", "#fcd34d", "#78350f"
        badge, verdict = "ERROR", str(result.get("decision") or error)
    elif matched:
        accent, bg, border, text = "#059669", "#ecfdf5", "#6ee7b7", "#064e3b"
        badge, verdict = "MATCH", result.get("decision") or "Genuine signature"
    else:
        accent, bg, border, text = "#dc2626", "#fef2f2", "#fca5a5", "#7f1d1d"
        badge, verdict = "NO MATCH", result.get("decision") or "Likely forged / different"

    distance = result["distance"]
    threshold = result["threshold"]
    cosine = result["cosine_similarity"]

    def _fmt(val: float) -> str:
        try:
            if val != val:  # NaN
                return "—"
            return f"{float(val):.4f}"
        except (TypeError, ValueError):
            return "—"

    # izamscientist similarity is already on [0, 1]
    try:
        bar_pct = min(max(float(cosine) * 100.0, 0.0), 100.0) if cosine == cosine else 0.0
    except (TypeError, ValueError):
        bar_pct = 0.0
    thresh_pct = min(max(threshold * 100.0, 0.0), 100.0)

    return f"""
    <div class="result-card" style="background:{bg};border-color:{border};color:{text};">
      <div class="result-top">
        <span class="result-badge" style="background:{accent};">{badge}</span>
        <span class="result-verdict">{verdict}</span>
      </div>
      <div class="metric-grid">
        <div class="metric"><div class="metric-label">Similarity</div><div class="metric-value">{_fmt(cosine)}</div></div>
        <div class="metric"><div class="metric-label">Distance</div><div class="metric-value">{_fmt(distance)}</div></div>
        <div class="metric"><div class="metric-label">Threshold</div><div class="metric-value">≥ {threshold:.2f}</div></div>
      </div>
      <div class="bar-wrap">
        <div class="bar-labels"><span>Less similar</span><span>More similar</span></div>
        <div class="bar-track">
          <div class="bar-fill" style="width:{bar_pct:.1f}%;background:{accent};"></div>
          <div class="bar-threshold" style="left:{thresh_pct:.1f}%;"></div>
        </div>
        <div class="bar-hint">Cleanup → izamscientist ConvNeXt-Tiny. Match when similarity ≥ threshold.</div>
      </div>
    </div>
    """


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,650&family=Manrope:wght@400;500;600;700&display=swap');

:root {
  --ink: #102a43;
  --ink-soft: #334e68;
  --muted: #627d98;
  --line: #d9e2ec;
  --panel: #ffffff;
  --panel-2: #f5f7fa;
  --canvas: #e8eef5;
  --accent: #0b6e4f;
  --accent-hover: #085c42;
  --accent-soft: #d8f3e7;
  --focus: #243b53;
}

html, body, .gradio-container, .main, .app,
.dark, .dark .main, .dark .app {
  background:
    radial-gradient(900px 420px at 12% -8%, #d7e7f7 0%, transparent 55%),
    radial-gradient(800px 380px at 100% 0%, #d9f0e6 0%, transparent 50%),
    linear-gradient(180deg, #eef3f8 0%, #e4ebf3 100%) !important;
  color: var(--ink) !important;
  font-family: "Manrope", sans-serif !important;
}

.gradio-container {
  max-width: 1040px !important;
  margin: 0 auto !important;
  padding-top: 12px !important;
  padding-bottom: 28px !important;
}

/* Force light surfaces even if OS prefers dark */
.block, .form, .panel, .tabitem, .tabs,
.svelte-1edwuoz, .gr-group, .gr-box {
  background: transparent !important;
}
.block {
  border: none !important;
  box-shadow: none !important;
}

/* Image / gallery cards */
.image-container, .upload-container, .gallery,
div[data-testid="image"], .image-frame {
  background: var(--panel) !important;
  border: 1px solid var(--line) !important;
  border-radius: 14px !important;
  box-shadow: 0 1px 2px rgba(16, 42, 67, 0.04) !important;
}

/* Labels */
label, .label-wrap, span.label-text, .block > label span {
  color: var(--ink-soft) !important;
  background: transparent !important;
  font-family: "Manrope", sans-serif !important;
  font-weight: 650 !important;
  font-size: 0.86rem !important;
  letter-spacing: 0.01em;
}

/* Inputs */
input, textarea, select, .wrap, .secondary-wrap {
  border-radius: 10px !important;
}
input, textarea, .scroll-hide {
  background: var(--panel) !important;
  color: var(--ink) !important;
  border-color: var(--line) !important;
}

/* Tabs */
.tabs {
  margin-top: 8px !important;
}
.tab-nav {
  border-bottom: 1px solid var(--line) !important;
  gap: 4px !important;
  justify-content: flex-start !important;
  background: transparent !important;
}
.tab-nav button {
  font-family: "Manrope", sans-serif !important;
  font-size: 0.95rem !important;
  font-weight: 650 !important;
  color: var(--muted) !important;
  padding: 12px 18px !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  background: transparent !important;
}
.tab-nav button.selected {
  color: var(--accent) !important;
  border-bottom-color: var(--accent) !important;
  background: transparent !important;
}

/* Primary button */
button.primary, .primary {
  background: var(--accent) !important;
  border: 1px solid var(--accent) !important;
  color: #fff !important;
  font-family: "Manrope", sans-serif !important;
  font-weight: 700 !important;
  border-radius: 10px !important;
  box-shadow: none !important;
  min-height: 46px !important;
}
button.primary:hover {
  background: var(--accent-hover) !important;
  border-color: var(--accent-hover) !important;
}

/* Image upload source bar: Upload · Webcam · Clipboard inside the image card */
#doc-input .image-container,
#specimen-input .image-container {
  overflow: hidden !important;
  display: flex !important;
  flex-direction: column !important;
  height: auto !important;
  min-height: 0 !important;
}
#specimen-input .image-container {
  min-height: 300px !important;
}
#doc-input .upload-container,
#specimen-input .upload-container {
  flex: 1 1 auto !important;
  min-height: 260px !important;
}
#doc-input .source-selection,
#specimen-input .source-selection {
  display: flex !important;
  flex-wrap: wrap !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 6px !important;
  width: 100% !important;
  height: auto !important;
  min-height: 48px !important;
  padding: 10px 8px 12px !important;
  margin-top: 0 !important;
  flex: 0 0 auto !important;
  border-top: 1px solid var(--line) !important;
  background: var(--panel-2) !important;
  border-radius: 0 0 14px 14px !important;
  box-sizing: border-box !important;
  position: relative !important;
  z-index: 2 !important;
}
#doc-input .source-selection button.icon,
#specimen-input .source-selection button.icon {
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 5px !important;
  width: auto !important;
  min-width: 0 !important;
  height: 34px !important;
  padding: 0 10px !important;
  border-radius: 9px !important;
  border: 1.5px solid var(--line) !important;
  background: #ffffff !important;
  color: var(--ink-soft) !important;
  font-family: "Manrope", sans-serif !important;
  font-size: 0.74rem !important;
  font-weight: 700 !important;
  box-shadow: none !important;
  flex: 0 1 auto !important;
}
#doc-input .source-selection button.icon svg,
#specimen-input .source-selection button.icon svg {
  width: 15px !important;
  height: 15px !important;
  flex-shrink: 0 !important;
}
#doc-input .source-selection button.icon.selected,
#specimen-input .source-selection button.icon.selected {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  background: var(--accent-soft) !important;
}
/* Emphasize clipboard as a clear paste action while uploading */
#doc-input .source-selection button[aria-label="Paste from clipboard"],
#specimen-input .source-selection button[aria-label="Paste from clipboard"] {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  background: #ffffff !important;
}
#doc-input .source-selection button[aria-label="Upload file"]::after,
#specimen-input .source-selection button[aria-label="Upload file"]::after {
  content: "Upload";
}
#doc-input .source-selection button[aria-label="Capture from camera"]::after,
#specimen-input .source-selection button[aria-label="Capture from camera"]::after {
  content: "Webcam";
}
#doc-input .source-selection button[aria-label="Paste from clipboard"]::after,
#specimen-input .source-selection button[aria-label="Paste from clipboard"]::after {
  content: "Paste";
}
/* Specimen upload / clipboard source bar */
#specimen-input .source-selection {
  flex-wrap: nowrap !important;
  gap: 4px !important;
  padding: 6px 4px 8px !important;
  min-height: 40px !important;
}
#specimen-input .source-selection button.icon {
  padding: 0 6px !important;
  font-size: 0.66rem !important;
  height: 30px !important;
  flex: 1 1 0 !important;
}
#specimen-input .source-selection button.icon svg {
  width: 12px !important;
  height: 12px !important;
}

/* Signature pad (Draw) */
#specimen-pad {
  min-height: 220px !important;
}
#specimen-pad .image-container,
#specimen-pad [data-testid="image"] {
  background: #ffffff !important;
  border: 1px dashed var(--line) !important;
  border-radius: 12px !important;
}
#specimen-mode label,
#specimen-mode span {
  font-family: "Manrope", sans-serif !important;
  font-weight: 650 !important;
}

/* Sliders */
input[type="range"] {
  accent-color: var(--accent) !important;
}

/* Hero */
#hero {
  text-align: left;
  padding: 18px 4px 8px;
  margin-bottom: 4px;
}
#hero .brand {
  font-family: "Fraunces", Georgia, serif;
  font-size: 0.78rem;
  font-weight: 650;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--accent);
  margin: 0 0 8px;
}
#hero h1 {
  font-family: "Fraunces", Georgia, serif !important;
  font-size: 2.35rem !important;
  font-weight: 650 !important;
  letter-spacing: -0.03em;
  color: var(--ink) !important;
  margin: 0 0 8px !important;
  line-height: 1.15 !important;
}
#hero p {
  margin: 0 !important;
  max-width: 560px;
  color: var(--muted) !important;
  font-size: 1rem !important;
  line-height: 1.55 !important;
}

#detect-row, #verify-row {
  align-items: stretch !important;
  gap: 14px !important;
}
#detect-row > div,
#verify-row > div {
  flex: 1 1 0 !important;
  min-width: 0 !important;
}

/* Detect/Verify panels: source toolbar stays inside the upload card */
#detect-row > div,
#verify-row > div {
  display: flex !important;
  flex-direction: column !important;
}
#detect-row .image-container,
#verify-row .image-container {
  flex: 1 1 auto !important;
}
#verify-row > div:first-child {
  flex: 1.55 1 0 !important;
  min-width: 300px !important;
}

/* Dropdown / select surfaces */
.wrap.svelte-1hfxr2k,
.secondary-wrap,
div.svelte-1gfkn6j,
.container.svelte-1p9xokt {
  background: #ffffff !important;
  border-color: var(--line) !important;
  color: var(--ink) !important;
}
[role="listbox"], [role="option"], ul.options {
  background: #ffffff !important;
  color: var(--ink) !important;
}

.status-banner, .result-idle {
  display: flex;
  flex-direction: column;
  gap: 4px;
  border: 1px solid;
  border-radius: 12px;
  padding: 14px 16px;
  margin-top: 4px;
  font-family: "Manrope", sans-serif;
}
.status-banner strong, .result-idle strong {
  font-size: 0.98rem;
}
.status-banner span, .result-idle span {
  font-size: 0.9rem;
  line-height: 1.45;
  opacity: 0.92;
}
.result-idle {
  border-style: dashed;
  border-color: var(--line);
  background: var(--panel);
  color: var(--ink-soft);
  text-align: center;
  padding: 24px 16px;
}

.result-card {
  border: 1px solid;
  border-radius: 14px;
  padding: 18px;
  margin-top: 4px;
  font-family: "Manrope", sans-serif;
}
.result-top {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 14px;
  flex-wrap: wrap;
}
.result-badge {
  color: #fff;
  font-weight: 800;
  font-size: 0.75rem;
  letter-spacing: 0.08em;
  padding: 6px 12px;
  border-radius: 6px;
}
.result-verdict {
  font-size: 1.1rem;
  font-weight: 700;
}
.metric-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 14px;
}
.metric {
  background: rgba(255,255,255,0.7);
  border: 1px solid rgba(16,42,67,0.08);
  border-radius: 10px;
  padding: 12px;
}
.metric-label {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  opacity: 0.7;
  margin-bottom: 4px;
}
.metric-value {
  font-size: 1.15rem;
  font-weight: 750;
  font-variant-numeric: tabular-nums;
}
.bar-labels {
  display: flex;
  justify-content: space-between;
  font-size: 0.72rem;
  opacity: 0.7;
  margin-bottom: 6px;
}
.bar-track {
  position: relative;
  height: 8px;
  border-radius: 999px;
  background: rgba(16,42,67,0.1);
  overflow: hidden;
}
.bar-fill { height: 100%; border-radius: 999px; }
.bar-threshold {
  position: absolute;
  top: -3px;
  bottom: -3px;
  width: 2px;
  background: var(--ink);
}
.bar-hint {
  margin-top: 8px;
  font-size: 0.84rem;
  opacity: 0.8;
}

footer {
  color: var(--muted) !important;
  opacity: 0.7;
}

@media (max-width: 800px) {
  #hero h1 { font-size: 1.75rem !important; }
  .metric-grid { grid-template-columns: 1fr; }
}
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#e8f8f1",
        c100="#d8f3e7",
        c200="#a8e4c8",
        c300="#6fd0a5",
        c400="#3bb884",
        c500="#0b6e4f",
        c600="#085c42",
        c700="#064c37",
        c800="#053c2c",
        c900="#042c20",
        c950="#021a13",
    ),
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Manrope"),
    font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
).set(
    body_background_fill="#e8eef5",
    body_background_fill_dark="#e8eef5",
    body_text_color="#102a43",
    body_text_color_dark="#102a43",
    background_fill_primary="#ffffff",
    background_fill_primary_dark="#ffffff",
    background_fill_secondary="#f5f7fa",
    background_fill_secondary_dark="#f5f7fa",
    border_color_primary="#d9e2ec",
    border_color_primary_dark="#d9e2ec",
    block_background_fill="#ffffff",
    block_background_fill_dark="#ffffff",
    block_border_color="#d9e2ec",
    block_border_color_dark="#d9e2ec",
    block_label_text_color="#334e68",
    block_label_text_color_dark="#334e68",
    block_title_text_color="#102a43",
    block_title_text_color_dark="#102a43",
    button_primary_background_fill="#0b6e4f",
    button_primary_background_fill_hover="#085c42",
    button_primary_background_fill_dark="#0b6e4f",
    button_primary_background_fill_hover_dark="#085c42",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_primary_border_color="#0b6e4f",
    button_primary_border_color_dark="#0b6e4f",
)

with gr.Blocks() as demo:
    gr.HTML(
        """
        <div id="hero">
          <div class="brand">InkCheck</div>
          <h1>Signature Detect &amp; Verify</h1>
          <p>Locate signatures on a document, then compare a specimen via draw pad, upload, or clipboard.</p>
        </div>
        """
    )

    detections_state = gr.State([])

    with gr.Tabs():
        with gr.Tab("Detect"):
            with gr.Row(equal_height=True, elem_id="detect-row"):
                document_input = gr.Image(
                    label="Document",
                    type="pil",
                    height=360,
                    sources=IMAGE_SOURCES,
                    placeholder=IMAGE_PLACEHOLDER,
                    elem_id="doc-input",
                    scale=1,
                )
                annotated_output = gr.Image(
                    label="Annotated result",
                    type="pil",
                    height=360,
                    interactive=False,
                    scale=1,
                )

            with gr.Row(equal_height=True):
                detect_conf = gr.Slider(
                    0.05,
                    0.95,
                    value=DEFAULT_DETECT_CONF,
                    step=0.05,
                    label="Confidence (conf)",
                    info="Ultralytics conf — higher = fewer boxes",
                    scale=1,
                )
                detect_iou = gr.Slider(
                    0.1,
                    0.95,
                    value=DEFAULT_DETECT_IOU,
                    step=0.05,
                    label="IoU (NMS)",
                    info="Ultralytics iou — overlap suppression",
                    scale=1,
                )
                detect_imgsz = gr.Slider(
                    320,
                    1280,
                    value=DEFAULT_DETECT_IMGSZ,
                    step=32,
                    label="Image size",
                    info="Ultralytics imgsz (trained at 768)",
                    scale=1,
                )
                detect_btn = gr.Button("Detect signatures", variant="primary", size="lg", scale=1)

            detect_summary = gr.HTML(
                value=_status_banner(
                    "Ready",
                    "Add a document via upload, clipboard, or webcam, then detect.",
                    tone="neutral",
                )
            )
            crops_gallery = gr.Gallery(
                label="Cropped signatures",
                columns=4,
                height=150,
                object_fit="contain",
                preview=True,
            )

        with gr.Tab("Verify"):
            with gr.Row(equal_height=True):
                signature_choice = gr.Dropdown(
                    label="Detected signature",
                    choices=[],
                    interactive=True,
                    info="From Detect",
                    scale=2,
                )
                verify_threshold = gr.Slider(
                    0.0,
                    1.0,
                    value=DEFAULT_VERIFY_THRESHOLD,
                    step=0.01,
                    label="Match threshold",
                    info="Match when similarity ≥ this value (from this slider).",
                    scale=2,
                )

            specimen_mode = gr.Radio(
                SPECIMEN_MODES,
                value="Draw",
                label="Specimen source",
                info="Draw on the pad, upload a file, or paste from clipboard",
                elem_id="specimen-mode",
            )

            with gr.Row(equal_height=True, elem_id="verify-row"):
                with gr.Column(scale=2, min_width=300):
                    specimen_draw = gr.Sketchpad(
                        label="Draw specimen",
                        type="pil",
                        image_mode="RGBA",
                        sources=(),
                        canvas_size=SPECIMEN_PAD_SIZE,
                        fixed_canvas=True,
                        layers=False,
                        brush=gr.Brush(
                            default_size=5,
                            colors=["#111111"],
                            color_mode="fixed",
                            default_color="#111111",
                        ),
                        eraser=gr.Eraser(default_size=12),
                        height=240,
                        elem_id="specimen-pad",
                        visible=True,
                    )
                    specimen_file = gr.Image(
                        label="Specimen",
                        type="pil",
                        height=300,
                        sources=["upload"],
                        placeholder=SPECIMEN_UPLOAD_PLACEHOLDER,
                        elem_id="specimen-input",
                        visible=False,
                    )
                doc_sig_preview = gr.Image(
                    label="Document (cleaned → model)",
                    type="pil",
                    height=300,
                    image_mode="RGB",
                    interactive=False,
                    buttons=[],
                    scale=1,
                    min_width=160,
                )
                specimen_preview = gr.Image(
                    label="Specimen (cleaned → model)",
                    type="pil",
                    height=300,
                    image_mode="RGB",
                    interactive=False,
                    buttons=[],
                    scale=1,
                    min_width=160,
                )

            verify_btn = gr.Button("Verify signature", variant="primary", size="lg")
            verify_result = gr.HTML(value=_idle_result())

    signature_choice.change(
        fn=preview_selected,
        inputs=[detections_state, signature_choice],
        outputs=[doc_sig_preview],
    )
    specimen_mode.change(
        fn=switch_specimen_mode,
        inputs=[specimen_mode],
        outputs=[specimen_draw, specimen_file],
    ).then(
        fn=preview_specimen,
        inputs=[specimen_mode, specimen_draw, specimen_file],
        outputs=[specimen_preview],
    )
    specimen_draw.change(
        fn=preview_specimen,
        inputs=[specimen_mode, specimen_draw, specimen_file],
        outputs=[specimen_preview],
    )
    specimen_file.change(
        fn=preview_specimen,
        inputs=[specimen_mode, specimen_draw, specimen_file],
        outputs=[specimen_preview],
    )

    detect_btn.click(
        fn=detect,
        inputs=[document_input, detect_conf, detect_iou, detect_imgsz],
        outputs=[
            annotated_output,
            detections_state,
            crops_gallery,
            detect_summary,
            signature_choice,
            doc_sig_preview,
            specimen_preview,
            verify_result,
        ],
    )

    verify_btn.click(
        fn=verify,
        inputs=[
            detections_state,
            signature_choice,
            specimen_mode,
            specimen_draw,
            specimen_file,
            verify_threshold,
        ],
        outputs=[doc_sig_preview, specimen_preview, verify_result],
    )

if __name__ == "__main__":
    demo.queue().launch(css=CUSTOM_CSS, theme=THEME, server_port=7860)
