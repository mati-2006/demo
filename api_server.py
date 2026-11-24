"""HTTP API for signature detect + verify (called from digipay-java-ekyc-v3 / local tests)."""

from __future__ import annotations

import base64
import io
import logging
import os
import re
from pathlib import Path
from typing import Any

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field, model_validator

from signature_models import (
    DEFAULT_DETECT_CONF,
    DEFAULT_DETECT_IMGSZ,
    DEFAULT_DETECT_IOU,
    DEFAULT_VERIFY_THRESHOLD,
    SignaturePipeline,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("signature_api")

PROPERTIES_PATH = Path(__file__).resolve().parent / "application.properties"


def _load_application_properties(path: Path = PROPERTIES_PATH) -> dict[str, str]:
    props: dict[str, str] = {}
    if not path.is_file():
        log.warning("application.properties not found at %s — using code defaults", path)
        return props
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        props[key.strip()] = value.strip()
    return props


def _prop_float(props: dict[str, str], key: str, default: float) -> float:
    raw = props.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("Invalid float for %s=%r — using default %s", key, raw, default)
        return default


def _prop_int(props: dict[str, str], key: str, default: int) -> int:
    raw = props.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(float(raw))
    except ValueError:
        log.warning("Invalid int for %s=%r — using default %s", key, raw, default)
        return default


_PROPS = _load_application_properties()
APP_DETECT_CONF = _prop_float(_PROPS, "signature.detect.conf", DEFAULT_DETECT_CONF)
APP_DETECT_IOU = _prop_float(_PROPS, "signature.detect.iou", DEFAULT_DETECT_IOU)
APP_DETECT_IMGSZ = _prop_int(_PROPS, "signature.detect.imgsz", DEFAULT_DETECT_IMGSZ)
APP_VERIFY_THRESHOLD = _prop_float(
    _PROPS, "signature.verify.threshold", DEFAULT_VERIFY_THRESHOLD
)
APP_SERVER_PORT = _prop_int(_PROPS, "server.port", 8001)

app = FastAPI(
    title="Signature Detect & Verify API",
    version="1.1.0",
    description=(
        "Detect a signature on a document image, then verify it against a "
        "signature-pad specimen. Thresholds default from application.properties."
    ),
)
pipeline = SignaturePipeline()


class VerifySignatureRequest(BaseModel):
    """Provide each image as a URL and/or base64 (data-URL prefix allowed)."""

    documentImageUrl: str | None = Field(
        None, description="URL of the ID / document image"
    )
    specimenImageUrl: str | None = Field(
        None, description="URL of the signature-pad / specimen image"
    )
    documentImageBase64: str | None = Field(
        None, description="Base64 (or data:image/...;base64,...) of the document"
    )
    specimenImageBase64: str | None = Field(
        None, description="Base64 (or data:image/...;base64,...) of the pad specimen"
    )

    # Verification threshold (aliases supported for clients)
    threshold: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Verify similarity threshold; default from application.properties",
    )
    verifyThreshold: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Alias of threshold",
    )

    # Detection thresholds
    detectConf: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="YOLO confidence; default from application.properties",
    )
    detectIou: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="YOLO NMS IoU; default from application.properties",
    )
    detectImgsz: int | None = Field(
        None,
        ge=320,
        le=2048,
        description="YOLO imgsz; default from application.properties",
    )

    @model_validator(mode="after")
    def _require_both_images(self) -> VerifySignatureRequest:
        has_doc = bool(self.documentImageUrl) or bool(self.documentImageBase64)
        has_spec = bool(self.specimenImageUrl) or bool(self.specimenImageBase64)
        if not has_doc:
            raise ValueError(
                "Provide documentImageUrl or documentImageBase64"
            )
        if not has_spec:
            raise ValueError(
                "Provide specimenImageUrl or specimenImageBase64"
            )
        return self


class DetectionDetail(BaseModel):
    index: int
    score: float
    source: str
    box: dict[str, int]
    confidence: float | None = None
    matched: bool | None = None
    decision: str | None = None


class VerifySignatureResponse(BaseModel):
    matched: bool
    confidence: float
    threshold: float
    detectConf: float
    detectIou: float
    detectImgsz: int
    detectionCount: int
    bestDetectionScore: float | None = None
    bestDetectionSource: str | None = None
    message: str
    detections: list[DetectionDetail] = Field(default_factory=list)


class DefaultsResponse(BaseModel):
    detectConf: float
    detectIou: float
    detectImgsz: int
    verifyThreshold: float
    propertiesFile: str


_DATA_URL_RE = re.compile(
    r"^data:image/[a-zA-Z0-9.+-]+;base64,(.+)$", re.IGNORECASE | re.DOTALL
)


def _load_image_from_url(url: str) -> Image.Image:
    log.info("Downloading image url=%s", url)
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to download image: {exc}"
        ) from exc

    try:
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400, detail=f"Invalid image data from URL: {exc}"
        ) from exc


def _load_image_from_base64(payload: str) -> Image.Image:
    raw = payload.strip()
    match = _DATA_URL_RE.match(raw)
    if match:
        raw = match.group(1)
    try:
        data = base64.b64decode(raw, validate=False)
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400, detail=f"Invalid base64 image: {exc}"
        ) from exc


def _resolve_image(*, url: str | None, b64: str | None, label: str) -> Image.Image:
    if b64:
        log.info("Loading %s from base64 (%d chars)", label, len(b64))
        return _load_image_from_base64(b64)
    if url:
        return _load_image_from_url(url)
    raise HTTPException(status_code=400, detail=f"Missing {label} image")


def _resolved_thresholds(body: VerifySignatureRequest) -> tuple[float, float, float, int]:
    verify_thr = (
        body.verifyThreshold
        if body.verifyThreshold is not None
        else body.threshold
        if body.threshold is not None
        else APP_VERIFY_THRESHOLD
    )
    detect_conf = (
        body.detectConf if body.detectConf is not None else APP_DETECT_CONF
    )
    detect_iou = body.detectIou if body.detectIou is not None else APP_DETECT_IOU
    detect_imgsz = (
        body.detectImgsz if body.detectImgsz is not None else APP_DETECT_IMGSZ
    )
    return verify_thr, detect_conf, detect_iou, detect_imgsz


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/defaults", response_model=DefaultsResponse)
def defaults() -> DefaultsResponse:
    """Current defaults loaded from application.properties (with code fallbacks)."""
    return DefaultsResponse(
        detectConf=APP_DETECT_CONF,
        detectIou=APP_DETECT_IOU,
        detectImgsz=APP_DETECT_IMGSZ,
        verifyThreshold=APP_VERIFY_THRESHOLD,
        propertiesFile=str(PROPERTIES_PATH),
    )


@app.post("/verify-signature", response_model=VerifySignatureResponse)
def verify_signature(body: VerifySignatureRequest) -> VerifySignatureResponse:
    """
    Detect signature(s) on the document, then verify each crop against the
    signature-pad specimen. Returns the best match.
    """
    verify_thr, detect_conf, detect_iou, detect_imgsz = _resolved_thresholds(body)

    log.info(
        "verify-signature start detectConf=%s detectIou=%s detectImgsz=%s verifyThreshold=%s",
        detect_conf,
        detect_iou,
        detect_imgsz,
        verify_thr,
    )

    document = _resolve_image(
        url=body.documentImageUrl,
        b64=body.documentImageBase64,
        label="document",
    )
    specimen = _resolve_image(
        url=body.specimenImageUrl,
        b64=body.specimenImageBase64,
        label="specimen",
    )

    detections = pipeline.detect_signatures(
        document,
        conf=detect_conf,
        iou=detect_iou,
        imgsz=detect_imgsz,
    )
    log.info("Detection finished count=%d", len(detections))

    if not detections:
        return VerifySignatureResponse(
            matched=False,
            confidence=0.0,
            threshold=verify_thr,
            detectConf=detect_conf,
            detectIou=detect_iou,
            detectImgsz=detect_imgsz,
            detectionCount=0,
            bestDetectionScore=None,
            bestDetectionSource=None,
            message="NO_SIGNATURE_DETECTED_ON_DOCUMENT",
            detections=[],
        )

    details: list[DetectionDetail] = []
    best: dict[str, Any] | None = None

    for index, det in enumerate(detections, start=1):
        result = pipeline.verify(
            det["crop"],
            specimen,
            threshold=verify_thr,
            reference_source="pad",
        )
        conf_val = (
            None
            if result.get("error")
            else float(result["cosine_similarity"])
        )
        matched_val = None if result.get("error") else bool(result["is_match"])
        details.append(
            DetectionDetail(
                index=index,
                score=float(det["score"]),
                source=str(det.get("source", "yolo")),
                box=det["box"],
                confidence=conf_val,
                matched=matched_val,
                decision=str(result.get("decision") or result.get("error") or ""),
            )
        )
        if result.get("error") or conf_val is None:
            log.warning(
                "Verify skipped/failed for detection #%d source=%s error=%s",
                index,
                det.get("source", "yolo"),
                result.get("error") or result.get("decision"),
            )
            continue

        candidate = {
            "confidence": conf_val,
            "matched": matched_val,
            "detectionScore": float(det["score"]),
            "source": str(det.get("source", "yolo")),
        }
        if best is None or candidate["confidence"] > best["confidence"]:
            best = candidate

    if best is None:
        log.warning("No scorable detection after verify")
        return VerifySignatureResponse(
            matched=False,
            confidence=0.0,
            threshold=verify_thr,
            detectConf=detect_conf,
            detectIou=detect_iou,
            detectImgsz=detect_imgsz,
            detectionCount=len(detections),
            bestDetectionScore=None,
            bestDetectionSource=None,
            message="NO_SIGNATURE_DETECTED_ON_DOCUMENT",
            detections=details,
        )

    matched = bool(best["matched"]) and best["confidence"] >= verify_thr
    message = "SIGNATURE_MATCHED" if matched else "SIGNATURE_NOT_MATCHED"
    log.info(
        "verify-signature done matched=%s confidence=%.4f threshold=%.4f source=%s message=%s",
        matched,
        best["confidence"],
        verify_thr,
        best["source"],
        message,
    )
    return VerifySignatureResponse(
        matched=matched,
        confidence=best["confidence"],
        threshold=verify_thr,
        detectConf=detect_conf,
        detectIou=detect_iou,
        detectImgsz=detect_imgsz,
        detectionCount=len(detections),
        bestDetectionScore=best["detectionScore"],
        bestDetectionSource=best["source"],
        message=message,
        detections=details,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", APP_SERVER_PORT))
    log.info(
        "Starting signature API on 0.0.0.0:%s (detectConf=%s verifyThreshold=%s)",
        port,
        APP_DETECT_CONF,
        APP_VERIFY_THRESHOLD,
    )
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=False)
