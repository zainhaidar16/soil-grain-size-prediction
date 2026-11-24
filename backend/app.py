"""FastAPI production inference service for SoilVision AI."""
from __future__ import annotations

import logging
import os
import re
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from backend.inference import DEFAULT_ARTIFACT_PATH, DEFAULT_METADATA_PATH, DISCLAIMER_TEXT, SoilVisionEngine

logger = logging.getLogger("soilvision.api")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

MAX_FILES_PER_REQUEST = int(os.environ.get("SOILVISION_MAX_FILES", "12"))
MAX_FILE_BYTES = int(os.environ.get("SOILVISION_MAX_FILE_MB", "15")) * 1024 * 1024
RATE_LIMIT_PER_MIN = int(os.environ.get("SOILVISION_RATE_LIMIT_PER_MIN", "30"))
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
Image.MAX_IMAGE_PIXELS = 45_000_000  # Protect against decompression bombs

_engine: SoilVisionEngine | None = None
_startup_error: str | None = None
_rate_buckets: dict[str, deque[float]] = defaultdict(deque)


def _get_engine() -> SoilVisionEngine:
    if _engine is None:
        detail = _startup_error or "Inference model is not initialized."
        raise HTTPException(status_code=503, detail=detail)
    return _engine


def _verify_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    expected = os.environ.get("SOILVISION_API_KEY", "").strip()
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def _enforce_rate_limit(request: Request) -> None:
    if RATE_LIMIT_PER_MIN <= 0:
        return
    forwarded = request.headers.get("x-forwarded-for", "")
    client_ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")
    now = time.monotonic()
    bucket = _rate_buckets[client_ip]
    while bucket and now - bucket[0] > 60.0:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please retry in a minute.")
    bucket.append(now)


def _sanitize_sample_id(raw: str | None) -> str:
    if not raw or not raw.strip():
        return "Soil_Sample"
    cleaned = re.sub(r"[^A-Za-z0-9._\-\s]", "", raw.strip())
    return cleaned[:80] or "Soil_Sample"


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _engine, _startup_error
    artifact_path = Path(os.environ.get("SOILVISION_ARTIFACT_PATH", str(DEFAULT_ARTIFACT_PATH)))
    metadata_path = Path(os.environ.get("SOILVISION_METADATA_PATH", str(DEFAULT_METADATA_PATH)))
    eager_backbone = os.environ.get("SOILVISION_EAGER_BACKBONE", "0") == "1"
    try:
        _engine = SoilVisionEngine(
            artifact_path=artifact_path,
            metadata_path=metadata_path,
            load_backbone=eager_backbone,
        )
        _startup_error = None
        logger.info("Loaded SoilVision artifact %s (version=%s)", artifact_path.name, _engine.bundle["model_version"])
    except Exception as exc:
        _engine = None
        _startup_error = f"Model artifact unavailable: {exc}"
        logger.error("Failed to initialize SoilVisionEngine: %s", exc)
    yield


app = FastAPI(
    title="SoilVision AI Inference API",
    version="1.0.0",
    description="Physically calibrated computer-vision soil grain-size distribution inference service.",
    lifespan=lifespan,
)

cors_origins = [
    o.strip()
    for o in os.environ.get("SOILVISION_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


@app.get("/health")
def health() -> dict[str, object]:
    ready = _engine is not None
    return {
        "status": "ok" if ready else "degraded",
        "model_ready": ready,
        "model_version": _engine.bundle["model_version"] if ready else None,
        "error": None if ready else "Model artifact is not loaded.",
    }


@app.get("/model-info")
def model_info(engine: SoilVisionEngine = Depends(_get_engine)) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "model_version": engine.bundle["model_version"],
        "metadata": engine.metadata,
        "camera_presets": engine.bundle["camera_presets"],
        "limits": {
            "max_files": MAX_FILES_PER_REQUEST,
            "max_file_mb": MAX_FILE_BYTES // (1024 * 1024),
            "allowed_formats": ["JPG", "PNG", "WebP"],
            "min_dimension_px": 224,
        },
        "limitations": [
            "Trained on 24 laboratory-sieved Austrian soil samples (127 photographs); unseen geology or lighting increases uncertainty.",
            "Silt and clay particles (<0.063 mm) are below optical pixel resolution at 4.5 px/mm and cannot be mineralogically separated.",
            "Physical scale calibration (px/mm or known field-of-view width) is required for accurate grain-size estimates.",
            "Large-stone correction is applied only when calibrated surface stone area >= 20 mm reaches at least 10%.",
        ],
        "disclaimer": DISCLAIMER_TEXT,
    }


@app.post("/predict", dependencies=[Depends(_verify_api_key), Depends(_enforce_rate_limit)])
async def predict(
    files: Annotated[list[UploadFile], File(description="1 to 12 soil photographs (JPG, PNG, WebP)")],
    sample_id: Annotated[str | None, Form()] = None,
    calibration_mode: Annotated[str, Form()] = "uncalibrated",
    effective_ppm: Annotated[float | None, Form()] = None,
    fov_width_mm: Annotated[float | None, Form()] = None,
    camera_preset: Annotated[str | None, Form()] = None,
    apply_stone_correction: Annotated[bool, Form()] = True,
    engine: SoilVisionEngine = Depends(_get_engine),
) -> dict[str, object]:
    if not files or len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Upload between 1 and {MAX_FILES_PER_REQUEST} soil photographs.",
        )

    payloads: list[tuple[str, bytes]] = []
    for f in files:
        fname = f.filename or "upload.jpg"
        ext = Path(fname).suffix.lower()
        ctype = (f.content_type or "").lower()
        if ext not in ALLOWED_EXTENSIONS and ctype not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file format for '{fname}'. Allowed formats: JPG, PNG, WebP.",
            )
        raw = await f.read()
        if not raw:
            raise HTTPException(status_code=400, detail=f"Uploaded file '{fname}' is empty.")
        if len(raw) > MAX_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File '{fname}' exceeds the {MAX_FILE_BYTES // (1024 * 1024)} MB limit.",
            )
        payloads.append((fname, raw))

    try:
        return engine.predict_images(
            image_payloads=payloads,
            sample_id=_sanitize_sample_id(sample_id),
            calibration_mode=calibration_mode,
            effective_ppm=effective_ppm,
            fov_width_mm=fov_width_mm,
            camera_preset=camera_preset,
            apply_stone_correction=apply_stone_correction,
        )
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="One or more uploaded files could not be decoded as valid images.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Inference failure during /predict")
        raise HTTPException(status_code=500, detail="Soil analysis failed due to an internal inference error.") from exc
