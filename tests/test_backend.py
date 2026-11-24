"""Automated tests for SoilVision AI FastAPI backend and geotechnical calculations."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.inference import (
    classify_soil_indicative,
    compute_engineering_parameters,
    compute_soil_composition,
    interpolate_percentile_diameter,
    resolve_calibration_ppm,
)
from src.metric import validate_curves


def _make_jpeg_bytes(width: int = 320, height: int = 320, seed: int = 7) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(50, 210, size=(height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def test_engineering_parameters_and_missing_percentiles():
    # Coarse gravel curve where F(0.002) = 1.0% -> all D10..D60 available
    gravel = np.array([1.0, 2.0, 4.0, 8.0, 14.0, 24.0, 35.0, 50.0, 75.0, 98.0, 100.0])
    eng = compute_engineering_parameters(gravel)
    assert eng["d50_mm"] == pytest.approx(6.3, rel=1e-2)
    assert eng["d10_mm"] is not None and eng["d60_mm"] is not None
    assert eng["cu"] is not None and eng["cu"] > 1.0
    assert eng["cc"] is not None and eng["cc"] > 0.0

    # Ultra-fine clay/silt curve where F(0.002) = 25% -> D10 must be None (Not available)
    fine = np.array([25.0, 40.0, 60.0, 85.0, 95.0, 99.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    assert interpolate_percentile_diameter(fine, 10.0) is None
    eng_fine = compute_engineering_parameters(fine)
    assert eng_fine["d10_mm"] is None
    assert eng_fine["cu"] is None
    assert eng_fine["cc"] is None
    assert eng_fine["d50_mm"] is not None


def test_soil_composition_and_indicative_classification():
    sand = np.array([0.5, 1.0, 2.0, 3.0, 25.0, 75.0, 98.0, 100.0, 100.0, 100.0, 100.0])
    comp = compute_soil_composition(sand)
    assert comp["fines_pct"] == pytest.approx(3.0)
    assert comp["sand_pct"] == pytest.approx(95.0)
    assert comp["gravel_and_coarser_pct"] == pytest.approx(2.0)

    eng = compute_engineering_parameters(sand)
    cls = classify_soil_indicative(sand, comp, eng)
    assert cls["is_definitive"] is False
    assert "S" in cls["symbol"]
    assert len(cls["missing_laboratory_inputs"]) >= 2


def test_calibration_resolution_modes():
    presets = {"iphone_16": {"label": "iPhone 16", "ref_width": 5712, "ref_height": 4284, "native_ppm": 19.525}}
    ppm, meta, warns = resolve_calibration_ppm(1000, 1000, "fov_mm", None, 200.0, None, presets)
    assert ppm == pytest.approx(5.0)
    assert meta["status"] == "calibrated" and not warns

    ppm_u, meta_u, warns_u = resolve_calibration_ppm(1000, 1000, "uncalibrated", None, None, None, presets)
    assert meta_u["status"] == "uncalibrated" and len(warns_u) >= 1
    assert ppm_u > 0

    with pytest.raises(ValueError, match="effective_ppm"):
        resolve_calibration_ppm(1000, 1000, "ppm", -2.0, None, None, presets)


def test_fastapi_endpoints_validation_and_missing_artifact(monkeypatch, tmp_path: Path):
    import backend.app as api

    # 1. Missing artifact returns graceful 503 on /model-info and degraded /health
    monkeypatch.setenv("SOILVISION_ARTIFACT_PATH", str(tmp_path / "does_not_exist.joblib"))
    with TestClient(api.app) as client:
        r_h = client.get("/health")
        assert r_h.status_code == 200
        assert r_h.json()["model_ready"] is False
        assert client.get("/model-info").status_code == 503

    # 2. Loaded artifact serves /health, /model-info, rejects invalid uploads, and runs real /predict
    monkeypatch.delenv("SOILVISION_ARTIFACT_PATH", raising=False)
    monkeypatch.setenv("SOILVISION_EAGER_BACKBONE", "0")
    with TestClient(api.app) as client:
        r_h = client.get("/health")
        assert r_h.status_code == 200 and r_h.json()["model_ready"] is True

        r_info = client.get("/model-info")
        assert r_info.status_code == 200
        assert r_info.json()["model_version"] == "soilvision-v1.0.0"

        # Invalid file format rejected with 415
        r_bad = client.post(
            "/predict",
            files={"files": ("notes.txt", b"not an image", "text/plain")},
            data={"calibration_mode": "uncalibrated"},
        )
        assert r_bad.status_code == 415

        # Corrupt image bytes rejected with 400
        r_corrupt = client.post(
            "/predict",
            files={"files": ("broken.jpg", b"not-a-valid-jpeg", "image/jpeg")},
            data={"calibration_mode": "uncalibrated"},
        )
        assert r_corrupt.status_code == 400

        # Real inference on a valid calibrated JPEG
        jpg_bytes = _make_jpeg_bytes(320, 320)
        r_ok = client.post(
            "/predict",
            files={"files": ("sample_01.jpg", jpg_bytes, "image/jpeg")},
            data={"sample_id": "Test_Sample_A", "calibration_mode": "ppm", "effective_ppm": "4.6"},
        )
        assert r_ok.status_code == 200, r_ok.text
        body = r_ok.json()
        assert body["schema_version"] == "1.0"
        assert body["sample_id"] == "Test_Sample_A"
        assert body["num_images"] == 1
        assert body["calibration"]["status"] == "calibrated"
        assert len(body["cumulative_passing_pct"]) == 11
        validate_curves(np.array(body["cumulative_passing_pct"], dtype=float))
