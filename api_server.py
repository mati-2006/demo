"""HTTP API for signature detect + verify (called from digipay-java-ekyc-v3)."""

from __future__ import annotations

import io
from typing import Any

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from signature_models import DEFAULT_VERIFY_THRESHOLD, SignaturePipeline

app = FastAPI(title="Signature Detect & Verify API", version="1.0.0")
pipeline = SignaturePipeline()


class VerifySignatureRequest(BaseModel):
    documentImageUrl: str = Field(..., description="URL of the ID / document image")
    specimenImageUrl: str = Field(
        ..., description="URL of the user-provided signature specimen"
    )
    threshold: float = Field(
        DEFAULT_VERIFY_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Cosine similarity match threshold",
    )


class VerifySignatureResponse(BaseModel):
    matched: bool
    confidence: float
    threshold: float
    detectionCount: int
    bestDetectionScore: float | None = None
    message: str


def _load_image_from_url(url: str) -> Image.Image:
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to download image: {exc}"
        ) from exc

    try:
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as exc:  # noqa: BLE001 — surface bad image payloads
        raise HTTPException(
            status_code=400, detail=f"Invalid image data from URL: {exc}"
        ) from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/verify-signature", response_model=VerifySignatureResponse)
def verify_signature(body: VerifySignatureRequest) -> VerifySignatureResponse:
    document = _load_image_from_url(body.documentImageUrl)
    specimen = _load_image_from_url(body.specimenImageUrl)

    detections = pipeline.detect_signatures(document)
    if not detections:
        return VerifySignatureResponse(
            matched=False,
            confidence=0.0,
            threshold=body.threshold,
            detectionCount=0,
            bestDetectionScore=None,
            message="NO_SIGNATURE_DETECTED_ON_DOCUMENT",
        )

    best: dict[str, Any] | None = None
    for det in detections:
        result = pipeline.verify(det["crop"], specimen, threshold=body.threshold)
        candidate = {
            "confidence": float(result["cosine_similarity"]),
            "matched": bool(result["is_match"]),
            "detectionScore": float(det["score"]),
        }
        if best is None or candidate["confidence"] > best["confidence"]:
            best = candidate

    assert best is not None
    matched = best["confidence"] >= body.threshold
    return VerifySignatureResponse(
        matched=matched,
        confidence=best["confidence"],
        threshold=body.threshold,
        detectionCount=len(detections),
        bestDetectionScore=best["detectionScore"],
        message="SIGNATURE_MATCHED" if matched else "SIGNATURE_NOT_MATCHED",
    )


if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8001, reload=False)
