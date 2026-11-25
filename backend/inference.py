"""Production inference engine and geotechnical post-processing for SoilVision AI."""
from __future__ import annotations

import gc
import io
import json
import os
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# On Vercel Serverless / Fluid Functions, only /tmp is writable for weight caching.
if os.environ.get("VERCEL"):
    os.environ.setdefault("HF_HOME", "/tmp/soilvision_hf_cache")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/soilvision_cache")

import joblib
import numpy as np
from PIL import Image, ImageOps
from safetensors.numpy import load_file as load_safetensors
from scipy.special import erf

from src.dataset import compute_effective_ppm
from src.features import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    estimate_coarse_stone_area,
    extract_dog_band5_features,
    extract_dog_lab_granulometry_features,
    l2_normalize,
    load_train_like_image,
    preprocess_soil_image,
)
from src.metric import (
    DEFAULT_QUANTILE_PROBS,
    DIAMETERS,
    LOG_DIAMETERS,
    TARGET_COLUMNS,
    log_quantiles_to_curve,
    repair_isotonic,
    validate_curves,
    weibull_lnb_c_to_curve,
)

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
DEFAULT_ARTIFACT_PATH = ARTIFACT_DIR / "soilvision_v1.joblib"
DEFAULT_METADATA_PATH = ARTIFACT_DIR / "metadata.json"

DISCLAIMER_TEXT = (
    "Results are AI estimates derived from soil photographs and are not a replacement "
    "for laboratory sieve analysis or certified geotechnical testing."
)


def interpolate_percentile_diameter(curve: np.ndarray, percentile: float) -> float | None:
    """Log-linearly interpolate diameter D_p (mm) at cumulative passing `percentile` (%).

    Returns None when `percentile` falls outside the reliably supported range
    `[curve[0], curve[-1]]` or inside an uninformative boundary plateau.
    """
    c = validate_curves(curve)[0]
    p = float(percentile)
    if p < c[0] - 1e-6 or p > c[-1] + 1e-6:
        return None
    if p <= c[0] + 1e-6:
        # Cannot extrapolate below 0.002 mm reliably
        return None if c[0] > p + 1e-6 else float(DIAMETERS[0])
    idx = int(np.searchsorted(c, p, side="left"))
    if idx == 0:
        return float(DIAMETERS[0])
    if idx >= len(c):
        return None
    c0, c1 = float(c[idx - 1]), float(c[idx])
    if c1 <= c0 + 1e-9:
        return float(DIAMETERS[idx])
    t = (p - c0) / (c1 - c0)
    log_d = float(LOG_DIAMETERS[idx - 1] + t * (LOG_DIAMETERS[idx] - LOG_DIAMETERS[idx - 1]))
    return float(10.0 ** log_d)


def compute_engineering_parameters(curve: np.ndarray) -> dict[str, float | None]:
    """Compute D10, D30, D50, D60, Cu, and Cc via log-diameter interpolation."""
    d10 = interpolate_percentile_diameter(curve, 10.0)
    d30 = interpolate_percentile_diameter(curve, 30.0)
    d50 = interpolate_percentile_diameter(curve, 50.0)
    d60 = interpolate_percentile_diameter(curve, 60.0)

    cu: float | None = None
    cc: float | None = None
    if d10 is not None and d60 is not None and d10 > 0.0:
        cu = float(d60 / d10)
        if d30 is not None and d60 > 0.0:
            cc = float((d30 ** 2) / (d10 * d60))

    def _round(v: float | None, digits: int = 4) -> float | None:
        return None if v is None else round(float(v), digits)

    return {
        "d10_mm": _round(d10, 4),
        "d30_mm": _round(d30, 4),
        "d50_mm": _round(d50, 4),
        "d60_mm": _round(d60, 4),
        "cu": _round(cu, 2),
        "cc": _round(cc, 2),
    }


def compute_soil_composition(curve: np.ndarray) -> dict[str, Any]:
    """Estimate particle-size fractions using ISO 14688-1 (0.063 mm and 2 mm boundaries)."""
    c = validate_curves(curve)[0]
    f063 = float(c[TARGET_COLUMNS.index("0.063")])
    f2 = float(c[TARGET_COLUMNS.index("2")])
    fines = max(0.0, min(100.0, f063))
    sand = max(0.0, min(100.0, f2 - f063))
    gravel_and_coarser = max(0.0, min(100.0, 100.0 - f2))
    return {
        "standard": "ISO 14688-1 (0.063 mm and 2.0 mm boundaries)",
        "fines_pct": round(fines, 2),
        "sand_pct": round(sand, 2),
        "gravel_and_coarser_pct": round(gravel_and_coarser, 2),
        "note": (
            "Estimated particle-size mass fractions derived from the predicted cumulative curve. "
            "The fine fraction (<0.063 mm) combines silt- and clay-sized particles and does not "
            "distinguish mineralogy or plasticity."
        ),
    }


def classify_soil_indicative(
    curve: np.ndarray,
    composition: dict[str, Any],
    eng: dict[str, float | None],
) -> dict[str, Any]:
    """Provide an indicative grain-size-only classification (ISO 14688-1 / USCS-aligned).

    Explicitly lists unavailable laboratory inputs (Atterberg limits, organic content)
    and never claims a definitive geotechnical classification.
    """
    fines = float(composition["fines_pct"])
    sand = float(composition["sand_pct"])
    gravel = float(composition["gravel_and_coarser_pct"])
    cu = eng.get("cu")
    cc = eng.get("cc")

    missing_lab = [
        "Liquid Limit (LL) and Plasticity Index (PI / Atterberg limits)",
        "Organic matter content",
        "Particle mineralogy and angularity",
    ]

    if fines >= 50.0:
        code = "F (Fine-grained soil — indicative)"
        label = "Fine-grained soil (silt/clay-sized dominant)"
        rationale = (
            f"Estimated fine fraction (<0.063 mm) is {fines:.1f}% (>= 50%). "
            "Distinguishing silt (M) vs. clay (C) and low vs. high plasticity requires Atterberg limits."
        )
    else:
        coarse_dominant = "Gravel" if gravel >= sand else "Sand"
        prefix = "G" if coarse_dominant == "Gravel" else "S"
        if fines < 5.0:
            well_graded = False
            if cu is not None and cc is not None:
                cu_thresh = 4.0 if prefix == "G" else 6.0
                well_graded = (cu >= cu_thresh) and (1.0 <= cc <= 3.0)
            if cu is None or cc is None:
                code = f"{prefix} (Clean {coarse_dominant.lower()} — gradation incomplete)"
                label = f"Clean {coarse_dominant.lower()} (grading coefficients unavailable)"
            elif well_graded:
                code = f"{prefix}W (Indicative)"
                label = f"Well-graded {coarse_dominant.lower()}"
            else:
                code = f"{prefix}P (Indicative)"
                label = f"Poorly graded {coarse_dominant.lower()}"
            rationale = (
                f"Coarse-grained ({100.0 - fines:.1f}% > 0.063 mm), {coarse_dominant.lower()}-dominant "
                f"({gravel:.1f}% gravel vs. {sand:.1f}% sand), with low estimated fines ({fines:.1f}% < 5%)."
            )
        elif fines <= 12.0:
            code = f"{prefix}W-{prefix}M/{prefix}C or {prefix}P-{prefix}M/{prefix}C (Indicative)"
            label = f"{coarse_dominant} with fines (borderline 5–12%)"
            rationale = (
                f"Coarse-grained, {coarse_dominant.lower()}-dominant with borderline fines ({fines:.1f}%). "
                "Dual-symbol assignment requires plasticity testing of the fine fraction."
            )
        else:
            code = f"{prefix}M / {prefix}C (Indicative)"
            label = f"Silty or clayey {coarse_dominant.lower()}"
            rationale = (
                f"Coarse-grained, {coarse_dominant.lower()}-dominant with significant estimated fines "
                f"({fines:.1f}% > 12%). Separating silty ({prefix}M) from clayey ({prefix}C) requires Atterberg limits."
            )

    return {
        "standard": "Indicative grain-size-only classification (ISO 14688-1 / USCS boundaries)",
        "symbol": code,
        "category": label,
        "rationale": rationale,
        "missing_laboratory_inputs": missing_lab,
        "is_definitive": False,
    }


def resolve_calibration_ppm(
    width: int,
    height: int,
    calibration_mode: str,
    effective_ppm: float | None,
    fov_width_mm: float | None,
    camera_preset: str | None,
    presets: dict[str, dict[str, Any]],
    fallback_ppm: float = 4.6,
) -> tuple[float, dict[str, Any], list[str]]:
    """Resolve effective px/mm from explicit user calibration without silently assuming scale."""
    warnings: list[str] = []
    mode = (calibration_mode or "uncalibrated").strip().lower()

    if mode == "ppm":
        if effective_ppm is None or effective_ppm <= 0.5 or effective_ppm > 100.0:
            raise ValueError("effective_ppm must be between 0.5 and 100.0 px/mm when calibration_mode='ppm'")
        return float(effective_ppm), {
            "mode": "ppm",
            "status": "calibrated",
            "effective_ppm": round(float(effective_ppm), 4),
            "source": "User-provided effective pixels per millimeter",
        }, warnings

    if mode == "fov_mm":
        if fov_width_mm is None or fov_width_mm < 20.0 or fov_width_mm > 2000.0:
            raise ValueError("fov_width_mm must be between 20 and 2000 mm when calibration_mode='fov_mm'")
        ppm = float(width) / float(fov_width_mm)
        if ppm <= 0.5 or ppm > 100.0:
            raise ValueError(f"Derived scale ({ppm:.2f} px/mm) is outside supported range [0.5, 100.0] px/mm")
        return ppm, {
            "mode": "fov_mm",
            "status": "calibrated",
            "effective_ppm": round(ppm, 4),
            "fov_width_mm": round(float(fov_width_mm), 2),
            "source": "Computed from image width and known physical field-of-view width (mm)",
        }, warnings

    if mode == "camera_preset":
        if not camera_preset or camera_preset not in presets:
            raise ValueError(f"Unknown camera_preset '{camera_preset}'. Supported: {sorted(presets)}")
        cam = presets[camera_preset]
        ppm = compute_effective_ppm(
            width=width,
            height=height,
            ref_width=int(cam["ref_width"]),
            ref_height=int(cam["ref_height"]),
            native_ppm=float(cam["native_ppm"]),
        )
        warnings.append(
            f"Camera preset '{cam['label']}' assumes standardized ~210 mm capture distance. "
            "Deviations in stand-off height will proportionally shift predicted grain sizes."
        )
        return ppm, {
            "mode": "camera_preset",
            "status": "preset_calibrated",
            "camera_preset": camera_preset,
            "camera_label": str(cam["label"]),
            "effective_ppm": round(ppm, 4),
            "source": f"Reference calibration preset ({cam['label']})",
        }, warnings

    if mode == "uncalibrated":
        warnings.append(
            "Uncalibrated upload: no physical scale (px/mm or field-of-view width) was provided. "
            f"Using fallback scale ({fallback_ppm:.2f} px/mm) for inference; grain-size predictions "
            "may be substantially shifted unless the photograph matches the reference capture geometry."
        )
        return float(fallback_ppm), {
            "mode": "uncalibrated",
            "status": "uncalibrated",
            "effective_ppm": round(float(fallback_ppm), 4),
            "source": "Fallback training median scale (uncalibrated — reduced reliability)",
        }, warnings

    raise ValueError(f"Unsupported calibration_mode '{calibration_mode}'")


def _weighted_median_matrix(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute column-wise weighted median (L1-optimal point estimate in quantile space)."""
    w = np.asarray(weights, dtype=float)
    w = w / (w.sum() + 1e-12)
    out = np.zeros(values.shape[1], dtype=float)
    for col in range(values.shape[1]):
        order = np.argsort(values[:, col])
        cw = np.cumsum(w[order])
        idx = int(np.searchsorted(cw, 0.5))
        out[col] = values[order[min(idx, len(order) - 1)], col]
    return out


class NumPyDINOv2Extractor:
    """Pure-NumPy + safetensors ViT-Small/14 DINOv2 encoder matching PyTorch timm to ~3e-7."""

    HF_WEIGHTS_URL_TEMPLATE = "https://huggingface.co/timm/{model_name}/resolve/main/model.safetensors"

    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        image_size: int = 224,
        crop_mms: Sequence[float | None] = (90.0, 150.0),
    ) -> None:
        self.model_name = model_name
        self.image_size = int(image_size)
        self.crop_mms = tuple(crop_mms)
        self._grid_n = self.image_size // 14
        self._mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 1, 3)
        self._std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 1, 3)

        weights_path = self._resolve_weights_path()
        raw = load_safetensors(str(weights_path))
        self.cls_token = raw["cls_token"].astype(np.float32)
        self.pos_embed = self._resample_pos_embed(raw["pos_embed"].astype(np.float32), self._grid_n, self._grid_n)
        self.patch_w = raw["patch_embed.proj.weight"].astype(np.float32).reshape(384, -1).T
        self.patch_b = raw["patch_embed.proj.bias"].astype(np.float32)
        self.norm_w = raw["norm.weight"].astype(np.float32)
        self.norm_b = raw["norm.bias"].astype(np.float32)

        self.blocks: list[dict[str, np.ndarray]] = []
        for i in range(12):
            p = f"blocks.{i}."
            self.blocks.append(
                {
                    "n1_w": raw[p + "norm1.weight"].astype(np.float32),
                    "n1_b": raw[p + "norm1.bias"].astype(np.float32),
                    "qkv_w": raw[p + "attn.qkv.weight"].astype(np.float32).T,
                    "qkv_b": raw[p + "attn.qkv.bias"].astype(np.float32),
                    "proj_w": raw[p + "attn.proj.weight"].astype(np.float32).T,
                    "proj_b": raw[p + "attn.proj.bias"].astype(np.float32),
                    "ls1": raw[p + "ls1.gamma"].astype(np.float32),
                    "n2_w": raw[p + "norm2.weight"].astype(np.float32),
                    "n2_b": raw[p + "norm2.bias"].astype(np.float32),
                    "fc1_w": raw[p + "mlp.fc1.weight"].astype(np.float32).T,
                    "fc1_b": raw[p + "mlp.fc1.bias"].astype(np.float32),
                    "fc2_w": raw[p + "mlp.fc2.weight"].astype(np.float32).T,
                    "fc2_b": raw[p + "mlp.fc2.bias"].astype(np.float32),
                    "ls2": raw[p + "ls2.gamma"].astype(np.float32),
                }
            )
        del raw

    def _resolve_weights_path(self) -> Path:
        env_path = os.environ.get("SOILVISION_DINOV2_WEIGHTS")
        if env_path and Path(env_path).is_file():
            return Path(env_path)

        # Reuse existing HuggingFace Hub cache when available (e.g., local dev / CI)
        hf_roots = [
            Path(os.environ.get("HF_HOME", "")) / "hub" if os.environ.get("HF_HOME") else None,
            Path.home() / ".cache" / "huggingface" / "hub",
        ]
        repo_dir_name = f"models--timm--{self.model_name}"
        for root in hf_roots:
            if root and root.is_dir():
                for candidate in sorted((root / repo_dir_name / "snapshots").glob("*/model.safetensors")):
                    if candidate.is_file():
                        return candidate

        cache_dir = Path("/tmp/soilvision_hf_cache") if os.environ.get("VERCEL") else Path.home() / ".cache" / "soilvision"
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / f"{self.model_name}.safetensors"
        if target.is_file() and target.stat().st_size > 1_000_000:
            return target

        url = self.HF_WEIGHTS_URL_TEMPLATE.format(model_name=self.model_name)
        tmp_target = target.with_suffix(".safetensors.tmp")
        req = urllib.request.Request(url, headers={"User-Agent": "SoilVisionAI/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp_target, "wb") as out_f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out_f.write(chunk)
        os.replace(tmp_target, target)
        return target

    @staticmethod
    def _resample_pos_embed(pos_embed: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
        cls_pos = pos_embed[:, :1, :]
        grid_pos = pos_embed[:, 1:, :]
        old_n = int(round(grid_pos.shape[1] ** 0.5))
        if old_n == new_h and old_n == new_w:
            return pos_embed
        grid = grid_pos.reshape(old_n, old_n, -1)
        c_dim = grid.shape[-1]
        out = np.empty((new_h, new_w, c_dim), dtype=np.float32)
        for c in range(c_dim):
            im = Image.fromarray(grid[:, :, c], mode="F")
            out[:, :, c] = np.asarray(im.resize((new_w, new_h), resample=Image.Resampling.BICUBIC), dtype=np.float32)
        return np.concatenate([cls_pos, out.reshape(1, new_h * new_w, c_dim)], axis=1)

    @staticmethod
    def _layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        mu = x.mean(axis=-1, keepdims=True)
        var = np.mean((x - mu) ** 2, axis=-1, keepdims=True)
        return (x - mu) / np.sqrt(var + eps) * w + b

    def encode_batch(self, pil_images: Sequence[Image.Image]) -> np.ndarray:
        arr = np.stack([np.asarray(im, dtype=np.float32) / 255.0 for im in pil_images], axis=0)
        arr = (arr - self._mean) / self._std
        bsz = arr.shape[0]
        gn = self._grid_n
        n_patches = gn * gn
        patches = (
            arr.transpose(0, 3, 1, 2)
            .reshape(bsz, 3, gn, 14, gn, 14)
            .transpose(0, 2, 4, 1, 3, 5)
            .reshape(bsz, n_patches, -1)
        )
        x = patches @ self.patch_w + self.patch_b
        cls = np.broadcast_to(self.cls_token, (bsz, 1, 384))
        x = np.concatenate([cls, x], axis=1) + self.pos_embed

        scale = np.float32(64.0 ** -0.5)
        inv_sqrt2 = np.float32(0.7071067811865476)
        n_tok = n_patches + 1
        for blk in self.blocks:
            h = self._layer_norm(x, blk["n1_w"], blk["n1_b"])
            qkv = (h @ blk["qkv_w"] + blk["qkv_b"]).reshape(bsz, n_tok, 3, 6, 64).transpose(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
            attn = attn - attn.max(axis=-1, keepdims=True)
            attn = np.exp(attn)
            attn = attn / attn.sum(axis=-1, keepdims=True)
            out = np.matmul(attn, v).transpose(0, 2, 1, 3).reshape(bsz, n_tok, 384)
            out = out @ blk["proj_w"] + blk["proj_b"]
            x = x + out * blk["ls1"]

            h2 = self._layer_norm(x, blk["n2_w"], blk["n2_b"])
            fc1 = h2 @ blk["fc1_w"] + blk["fc1_b"]
            fc1 = (0.5 * fc1 * (1.0 + erf(fc1 * inv_sqrt2))).astype(np.float32)
            fc2 = fc1 @ blk["fc2_w"] + blk["fc2_b"]
            x = x + fc2 * blk["ls2"]

        x = self._layer_norm(x, self.norm_w, self.norm_b)
        feats = np.concatenate([x[:, 0, :], x[:, 1:, :].mean(axis=1)], axis=-1)
        return l2_normalize(feats)


class SoilVisionEngine:
    """Thread-safe production inference wrapper around the fitted 37.32208 artifact."""

    def __init__(
        self,
        artifact_path: Path = DEFAULT_ARTIFACT_PATH,
        metadata_path: Path = DEFAULT_METADATA_PATH,
        load_backbone: bool = True,
    ) -> None:
        self.artifact_path = Path(artifact_path)
        self.metadata_path = Path(metadata_path)
        if not self.artifact_path.exists():
            raise FileNotFoundError(
                f"Fitted model artifact not found at {self.artifact_path}. "
                "Run `python -m backend.export_model` to generate it."
            )
        self.bundle: dict[str, Any] = joblib.load(self.artifact_path)
        self.metadata: dict[str, Any] = (
            json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if self.metadata_path.exists()
            else {"model_version": self.bundle["model_version"]}
        )
        self._ext_rgb: NumPyDINOv2Extractor | None = None
        if load_backbone:
            self._ensure_backbones()

    def _ensure_backbones(self) -> None:
        if self._ext_rgb is None:
            self._ext_rgb = NumPyDINOv2Extractor(
                model_name=str(self.bundle["dinov2_model_name"]),
            )

    def _extract_dinov2_pair(self, image_path: Path, effective_ppm: float) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_backbones()
        assert self._ext_rgb is not None
        ext = self._ext_rgb
        rgb_views: list[Image.Image] = []
        gn_views: list[Image.Image] = []
        for crop_mm in ext.crop_mms:
            im = preprocess_soil_image(
                image_path,
                effective_ppm=effective_ppm,
                crop_mm=crop_mm,
                image_size=ext.image_size,
            )
            rgb_views.append(im)
            g = np.asarray(im.convert("L"), dtype=np.float32)
            g = np.clip((g - g.mean()) / (g.std() + 1e-6) * 45.0 + 128.0, 0.0, 255.0).astype(np.uint8)
            gn_views.append(Image.fromarray(np.stack([g, g, g], axis=-1), mode="RGB"))

        emb = ext.encode_batch(rgb_views + gn_views)
        n_s = len(ext.crop_mms)
        rgb_concat = np.concatenate([emb[i] for i in range(n_s)], axis=0)[None, :]
        gn_concat = np.concatenate([emb[n_s + i] for i in range(n_s)], axis=0)[None, :]
        return l2_normalize(rgb_concat)[0].astype(np.float32), l2_normalize(gn_concat)[0].astype(np.float32)

    def predict_from_features(
        self,
        x_tl: np.ndarray,
        x_b5: np.ndarray,
        x_dino_rgb: np.ndarray,
        x_dino_gn: np.ndarray,
        stone_area: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Run the exact 37.32208 SoftRoute(Option 1 -> Scale14+DINO-GN) + Cobble step."""
        b = self.bundle

        # 1. Logit-PCA Ridge (per photo -> mean)
        z_ev = b["pca_logit"].inverse_transform(b["reg_logit"].predict(b["sc_logit"].transform(x_tl)))
        p_logit_ph = 100.0 / (1.0 + np.exp(-np.clip(z_ev, -20.0, 20.0)))
        p_logit_ph[:, -1] = 100.0
        p_logit = repair_isotonic(p_logit_ph.mean(axis=0))

        # 2. Weibull Ridge (per photo -> mean)
        w_ev = b["reg_weib"].predict(b["sc_weib"].transform(x_tl))
        p_weib_ph = np.vstack([weibull_lnb_c_to_curve(float(a), float(c)) for a, c in w_ev])
        p_weib = repair_isotonic(p_weib_ph.mean(axis=0))

        # 3. DINOv2-RGB LKM (sample-mean embedding -> PLS -> weighted median in LogQ11)
        xd_mean = x_dino_rgb.mean(axis=0, keepdims=True)
        z_ev_lkm = np.clip((xd_mean - b["lkm_mu"]) / b["lkm_sd"], -b["lkm_clip_z"], b["lkm_clip_z"])
        u_ev = b["pls_lkm"].transform(z_ev_lkm) / b["u_sd"]
        d2 = np.sum((b["u_tr_norm"] - u_ev) ** 2, axis=-1)
        weights = np.exp(-0.5 * d2 / (float(b["lkm_bandwidth"]) ** 2))
        q_pred = _weighted_median_matrix(b["y_tr_q"], weights)
        p_dino = repair_isotonic(log_quantiles_to_curve(q_pred, DEFAULT_QUANTILE_PROBS))

        w1, w2, w3 = b["opt1_weights"]
        opt1_curve = repair_isotonic(w1 * p_logit + w2 * p_weib + w3 * p_dino)

        # 4. Scale14 + DINO-GN Weibull Sand Specialist
        x_scale = np.hstack([x_b5, x_tl[:, b["dogfrac_idx"]]])
        c_ev = b["pca_dgn"].transform(b["sc_dgn"].transform(x_dino_gn))
        x_sand = np.hstack([b["sc_hand"].transform(x_scale), 0.10 * b["sc_dgn_pc"].transform(c_ev)])
        w_sand_ev = b["reg_sand"].predict(x_sand)
        p_sand_ph = np.vstack([weibull_lnb_c_to_curve(float(a), float(c)) for a, c in w_sand_ev])
        p_sand = repair_isotonic(p_sand_ph.mean(axis=0))

        # Bayesian leverage & photo dispersion diagnostics
        lev_ph = np.einsum("ij,jk,ik->i", x_sand, b["sand_gram_inv"], x_sand)
        mean_lev = float(np.mean(lev_ph))
        photo_std = float(p_sand_ph[:, [4, 5]].std(axis=0).mean()) if len(p_sand_ph) > 1 else None

        # 5. Calibrated Soft Routing (threshold=90%, scale=5.0)
        f2 = float(opt1_curve[TARGET_COLUMNS.index("2")])
        w_route = float(1.0 / (1.0 + np.exp(-np.clip((f2 - float(b["route_threshold_2mm"])) / float(b["route_scale"]), -20.0, 20.0))))
        routed = repair_isotonic((1.0 - w_route) * opt1_curve + w_route * p_sand)

        # 6. Physical Large-Stone (Cobble) Correction
        a63, a20 = float(stone_area[0]), float(stone_area[1])
        cobble_applied = False
        final_curve = routed.copy()
        if a20 >= float(b["cobble_min_a20"]):
            i20, i63 = TARGET_COLUMNS.index("20"), TARGET_COLUMNS.index("63")
            m20 = min(a20, float(b["cobble_max_mass"]))
            m63 = min(a63, m20)
            f20 = min(final_curve[i20], 100.0 * (1.0 - m20))
            f63 = max(min(final_curve[i63], 100.0 * (1.0 - m63)), f20)
            final_curve[:i20] *= f20 / max(final_curve[i20], 1e-6)
            final_curve[i20], final_curve[i63] = f20, f63
            final_curve = repair_isotonic(final_curve)
            cobble_applied = True

        validate_curves(final_curve)
        diagnostics = {
            "sand_routing_weight": round(w_route, 4),
            "bayesian_leverage": round(mean_lev, 4),
            "ood_warning": bool(mean_lev > float(b["train_lev_p95"]) * 1.5),
            "photo_dispersion_pct": None if photo_std is None else round(photo_std, 2),
            "coarse_stone_area_20mm": round(a20, 4),
            "coarse_stone_area_63mm": round(a63, 4),
            "large_stone_correction_applied": cobble_applied,
        }
        return final_curve, diagnostics

    def predict_images(
        self,
        image_payloads: list[tuple[str, bytes]],
        sample_id: str,
        calibration_mode: str = "uncalibrated",
        effective_ppm: float | None = None,
        fov_width_mm: float | None = None,
        camera_preset: str | None = None,
        apply_stone_correction: bool = True,
    ) -> dict[str, Any]:
        """Run full end-to-end preprocessing, feature extraction, and inference on uploaded bytes."""
        if not image_payloads:
            raise ValueError("At least one soil photograph is required")

        tl_cols = self.bundle["tl_cols"]
        train_ppm = float(self.bundle["train_ppm"])
        ref_qtables = self.bundle["ref_qtables"]
        presets = self.bundle["camera_presets"]

        x_tl_list, x_b5_list, x_rgb_list, x_gn_list = [], [], [], []
        stone_a63_list, stone_a20_list = [], []
        warnings: list[str] = []
        cal_info: dict[str, Any] = {}

        with tempfile.TemporaryDirectory(prefix="soilvision_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            for idx, (fname, raw_bytes) in enumerate(image_payloads):
                with Image.open(io.BytesIO(raw_bytes)) as im:
                    oriented = ImageOps.exif_transpose(im).convert("RGB")
                    width, height = oriented.size
                    if min(width, height) < 224:
                        raise ValueError(
                            f"Image '{fname}' ({width}x{height}) is too small; minimum dimension is 224 px."
                        )
                    img_path = tmp_root / f"img_{idx}.jpg"
                    oriented.save(img_path, format="JPEG", quality=95)

                ppm, cal_meta, cal_warns = resolve_calibration_ppm(
                    width=width,
                    height=height,
                    calibration_mode=calibration_mode,
                    effective_ppm=effective_ppm,
                    fov_width_mm=fov_width_mm,
                    camera_preset=camera_preset,
                    presets=presets,
                    fallback_ppm=train_ppm,
                )
                if idx == 0:
                    cal_info = cal_meta
                    warnings.extend(cal_warns)

                img_tl = load_train_like_image(img_path, ppm, train_ppm=train_ppm, ref_qtables=ref_qtables)
                tl_dict = extract_dog_lab_granulometry_features(img_tl)
                b5_dict = extract_dog_band5_features(img_tl)
                x_tl_list.append([tl_dict[c] for c in tl_cols])
                x_b5_list.append([b5_dict[f"b{k}"] for k in range(5)])

                rgb_vec, gn_vec = self._extract_dinov2_pair(img_path, ppm)
                x_rgb_list.append(rgb_vec)
                x_gn_list.append(gn_vec)

                if apply_stone_correction and calibration_mode != "uncalibrated":
                    a63, a20 = estimate_coarse_stone_area(img_path, effective_ppm=ppm)
                    stone_a63_list.append(a63)
                    stone_a20_list.append(a20)
                else:
                    stone_a63_list.append(0.0)
                    stone_a20_list.append(0.0)

                del img_tl
                gc.collect()

        if len(image_payloads) == 1:
            warnings.append(
                "Single-image analysis: the trained pipeline was validated on multi-photo soil groups "
                "(3–8 photos per sample). Uploading 2–4 photos of the same sample improves stability."
            )

        stone_area = (float(np.mean(stone_a63_list)), float(np.mean(stone_a20_list)))
        curve, diag = self.predict_from_features(
            x_tl=np.asarray(x_tl_list, dtype=np.float32),
            x_b5=np.asarray(x_b5_list, dtype=np.float32),
            x_dino_rgb=np.asarray(x_rgb_list, dtype=np.float32),
            x_dino_gn=np.asarray(x_gn_list, dtype=np.float32),
            stone_area=stone_area,
        )

        if diag["ood_warning"]:
            warnings.append(
                f"Out-of-distribution notice: predictive leverage ({diag['bayesian_leverage']:.2f}) exceeds "
                "the training-set envelope. Visual texture or lighting may differ from the 24 training soils."
            )
        if diag["large_stone_correction_applied"]:
            warnings.append(
                f"Large-stone correction applied: detected {diag['coarse_stone_area_20mm']*100:.1f}% surface area "
                f">= 20 mm ({diag['coarse_stone_area_63mm']*100:.1f}% >= 63 mm)."
            )

        composition = compute_soil_composition(curve)
        engineering = compute_engineering_parameters(curve)
        classification = classify_soil_indicative(curve, composition, engineering)

        return {
            "schema_version": "1.0",
            "sample_id": sample_id,
            "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_version": str(self.bundle["model_version"]),
            "num_images": len(image_payloads),
            "calibration": cal_info,
            "sieve_diameters_mm": [float(d) for d in DIAMETERS],
            "sieve_labels": list(TARGET_COLUMNS),
            "cumulative_passing_pct": [round(float(v), 2) for v in curve],
            "composition": composition,
            "engineering_parameters": engineering,
            "classification": classification,
            "reliability": {
                **diag,
                "calibration_status": cal_info.get("status", "uncalibrated"),
                "disclaimer": DISCLAIMER_TEXT,
            },
            "warnings": warnings,
        }
