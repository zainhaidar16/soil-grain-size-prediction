"""Export fitted SoilVision AI v1 inference artifact from the 37.32208 Kaggle pipeline.

Fits the exact `SoftRoute(Option 1 -> Scale14 + DINO-GN Weibull, s=5)` model on all
24 labeled training soils (127 photographs) using the reusable functions in `src/`
and exports a compact, versioned joblib bundle (~1 MB, no raw dataset or PyTorch
weights) plus `metadata.json`. Verifies numerical equivalence against the preserved
`submissions/submission.csv` (commit `38100253e14ed13ab61d8acabee43138f30973f7`).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.baseline import read_labels
from src.dataset import build_image_manifest, resolve_data_dir
from src.features import (
    DEFAULT_DINOV2_MODEL,
    DINOv2FeatureExtractor,
    compute_reference_jpeg_qtables,
    extract_dog_band5_features,
    extract_dog_lab_granulometry_features,
    load_train_like_image,
)
from src.metric import (
    DEFAULT_QUANTILE_PROBS,
    TARGET_COLUMNS,
    curve_to_log_quantiles,
    emd_per_sample,
    fit_weibull_lnb_c,
)
from src.model import _photo_soil_weights

MODEL_VERSION = "soilvision-v1.0.0"
RECOVERY_COMMIT = "38100253e14ed13ab61d8acabee43138f30973f7"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "soilvision_v1.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

# Standard camera calibration presets derived from competition ppm_updated.csv
CAMERA_PRESETS: dict[str, dict[str, float | int | str]] = {
    "iphone_16": {
        "label": "iPhone 16 (210 mm stand-off)",
        "ref_width": 5712,
        "ref_height": 4284,
        "native_ppm": 19.525,
    },
    "iphone_14": {
        "label": "iPhone 14 (210 mm stand-off)",
        "ref_width": 4032,
        "ref_height": 3024,
        "native_ppm": 13.941,
    },
    "motorola_edge": {
        "label": "Motorola Edge (210 mm stand-off)",
        "ref_width": 4000,
        "ref_height": 1800,
        "native_ppm": 11.492,
    },
    "samsung_a52": {
        "label": "Samsung A52 (210 mm stand-off)",
        "ref_width": 4624,
        "ref_height": 3468,
        "native_ppm": 13.151,
    },
    "motorola_edge_60_fusion": {
        "label": "Motorola Edge 60 Fusion (210 mm stand-off)",
        "ref_width": 4096,
        "ref_height": 3072,
        "native_ppm": 17.325,
    },
}


def _agg_mean(x_arr: np.ndarray, sample_ids: list[str]) -> tuple[list[str], np.ndarray]:
    arr = np.asarray(sample_ids)
    uniq = list(dict.fromkeys(sample_ids))
    return uniq, np.vstack([x_arr[arr == s].mean(axis=0) for s in uniq])


def export_artifact(
    data_dir: Path | None = None,
    scratch_dir: Path | None = None,
    output_dir: Path = ARTIFACT_DIR,
) -> Path:
    """Fit the exact 37.32208 inference pipeline and write versioned artifacts."""
    resolved_dir = resolve_data_dir(data_dir)
    labels_df = (
        read_labels(resolved_dir / "Training_labels_updated.csv")
        .set_index("sample_id")[TARGET_COLUMNS]
        .astype(float)
    )
    train_m = build_image_manifest(resolved_dir, "train")
    train_ppm, ref_qtables = compute_reference_jpeg_qtables(train_m)

    n_tr = len(train_m)
    tr_sids = train_m["sample_id"].tolist()

    # Reuse cached feature matrices if available in scratch_dir; otherwise compute fresh.
    if scratch_dir is not None and (scratch_dir / "ref_feats.npz").exists() and (scratch_dir / "a1_rgb.npz").exists() and (scratch_dir / "a2_gn.npz").exists():
        ref = np.load(scratch_dir / "ref_feats.npz")
        a1 = np.load(scratch_dir / "a1_rgb.npz")
        a2 = np.load(scratch_dir / "a2_gn.npz")
        x_tl_tr = ref["X_sand2"][:n_tr].astype(np.float32)
        x_b5_tr = ref["X_b5"][:n_tr].astype(np.float32)
        tl_cols = [str(c).removeprefix("tl_") for c in ref["tl_cols"]]
        x_dino_rgb_tr = a1["tr"].astype(np.float32)
        x_dino_gn_tr = a2["tr"].astype(np.float32)
    else:
        rows_tl, rows_b5 = [], []
        for p, ppm in zip(train_m["image_path"], train_m["effective_ppm"]):
            img_tl = load_train_like_image(p, float(ppm), train_ppm=train_ppm, ref_qtables=ref_qtables)
            rows_tl.append(extract_dog_lab_granulometry_features(img_tl))
            rows_b5.append(extract_dog_band5_features(img_tl))
        tl_df = pd.DataFrame(rows_tl)
        tl_cols = list(tl_df.columns)
        x_tl_tr = tl_df.to_numpy(dtype=np.float32)
        x_b5_tr = pd.DataFrame(rows_b5)[["b0", "b1", "b2", "b3", "b4"]].to_numpy(dtype=np.float32)

        ext_rgb = DINOv2FeatureExtractor(model_name=DEFAULT_DINOV2_MODEL, gray_norm=False)
        x_dino_rgb_tr = ext_rgb.extract_manifest_features(train_m, batch_size=8)
        ext_gn = DINOv2FeatureExtractor(model_name=DEFAULT_DINOV2_MODEL, gray_norm=True)
        x_dino_gn_tr = ext_gn.extract_manifest_features(train_m, batch_size=8)

    dogfrac_idx = [i for i, c in enumerate(tl_cols) if "dogfrac" in c or "centroid" in c]
    x_scale_tr = np.hstack([x_b5_tr, x_tl_tr[:, dogfrac_idx]]).astype(np.float32)

    y_tr_photo = labels_df.loc[tr_sids].to_numpy(dtype=float)
    wb_map = {sid: fit_weibull_lnb_c(labels_df.loc[sid].to_numpy(dtype=float)) for sid in labels_df.index}
    w_tr_photo = np.vstack([wb_map[s] for s in tr_sids])
    logq_map = {sid: curve_to_log_quantiles(labels_df.loc[sid].to_numpy(dtype=float), DEFAULT_QUANTILE_PROBS) for sid in labels_df.index}
    soil_w = _photo_soil_weights(tr_sids)

    # 1. Option 1 Component 1: Logit-PCA Ridge (tl 29d, alpha=30, n_components=3)
    sc_logit = StandardScaler().fit(x_tl_tr)
    p_tr = np.clip(y_tr_photo, 0.5, 99.5) / 100.0
    z_tr = np.log(p_tr / (1.0 - p_tr))
    pca_logit = PCA(n_components=3).fit(z_tr)
    reg_logit = Ridge(alpha=30.0).fit(sc_logit.transform(x_tl_tr), pca_logit.transform(z_tr), sample_weight=soil_w)

    # 2. Option 1 Component 2: Weibull Ridge (tl 29d, alpha=100)
    sc_weib = StandardScaler().fit(x_tl_tr)
    reg_weib = Ridge(alpha=100.0).fit(sc_weib.transform(x_tl_tr), w_tr_photo)

    # 3. Option 1 Component 3: DINOv2-S RGB Latent-Kernel-Median (LogQ11, n_components=1, bandwidth=0.7)
    tr_u, xd_tr = _agg_mean(x_dino_rgb_tr, tr_sids)
    y_tr_q = np.vstack([logq_map[s] for s in tr_u])
    lkm_mu = xd_tr.mean(axis=0, keepdims=True)
    lkm_sd = xd_tr.std(axis=0, keepdims=True)
    lkm_sd[lkm_sd == 0.0] = 1.0
    z_lkm = np.clip((xd_tr - lkm_mu) / lkm_sd, -3.5, 3.5)
    pls_lkm = PLSRegression(n_components=1).fit(z_lkm, y_tr_q)
    u_tr = pls_lkm.transform(z_lkm)
    u_sd = u_tr.std(axis=0, keepdims=True) + 1e-8
    u_tr_norm = u_tr / u_sd

    # 4. Sand Specialist: Scale14 + DINOv2-GN PCA(8, w=0.10) Weibull Ridge (alpha=100)
    sc_hand = StandardScaler().fit(x_scale_tr)
    sc_dgn = StandardScaler().fit(x_dino_gn_tr)
    pca_dgn = PCA(n_components=8, svd_solver="randomized", random_state=42)
    c_tr = pca_dgn.fit_transform(sc_dgn.transform(x_dino_gn_tr))
    sc_dgn_pc = StandardScaler().fit(c_tr)
    x_sand_tr = np.hstack([sc_hand.transform(x_scale_tr), 0.10 * sc_dgn_pc.transform(c_tr)])
    reg_sand = Ridge(alpha=100.0).fit(x_sand_tr, w_tr_photo)

    # Bayesian leverage inverse Gram matrix for OOD / reliability diagnostics
    sand_gram_inv = np.linalg.inv(x_sand_tr.T @ x_sand_tr + 100.0 * np.eye(x_sand_tr.shape[1]))
    train_leverages = np.einsum("ij,jk,ik->i", x_sand_tr, sand_gram_inv, x_sand_tr)
    lev_p95 = float(np.percentile(train_leverages, 95))

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_version": MODEL_VERSION,
        "recovery_commit": RECOVERY_COMMIT,
        "dinov2_model_name": DEFAULT_DINOV2_MODEL,
        "train_ppm": float(train_ppm),
        "ref_qtables": ref_qtables,
        "tl_cols": tl_cols,
        "dogfrac_idx": dogfrac_idx,
        "camera_presets": CAMERA_PRESETS,
        # Component 1: Logit-PCA Ridge
        "sc_logit": sc_logit,
        "pca_logit": pca_logit,
        "reg_logit": reg_logit,
        # Component 2: Weibull Ridge
        "sc_weib": sc_weib,
        "reg_weib": reg_weib,
        # Component 3: DINOv2-RGB LKM
        "lkm_mu": lkm_mu.astype(np.float32),
        "lkm_sd": lkm_sd.astype(np.float32),
        "pls_lkm": pls_lkm,
        "u_sd": u_sd.astype(np.float32),
        "u_tr_norm": u_tr_norm.astype(np.float32),
        "y_tr_q": y_tr_q.astype(np.float32),
        "lkm_bandwidth": 0.7,
        "lkm_clip_z": 3.5,
        # Component 4: Scale14 + DINO-GN Weibull Sand Specialist
        "sc_hand": sc_hand,
        "sc_dgn": sc_dgn,
        "pca_dgn": pca_dgn,
        "sc_dgn_pc": sc_dgn_pc,
        "reg_sand": reg_sand,
        "sand_gram_inv": sand_gram_inv.astype(np.float64),
        "train_lev_p95": lev_p95,
        # Routing & blending hyperparameters
        "opt1_weights": (0.35, 0.35, 0.30),
        "route_threshold_2mm": 90.0,
        "route_scale": 5.0,
        "cobble_min_a20": 0.10,
        "cobble_max_mass": 0.90,
    }

    artifact_path = output_dir / "soilvision_v1.joblib"
    joblib.dump(bundle, artifact_path, compress=3)

    metadata = {
        "model_version": MODEL_VERSION,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "recovery_commit": RECOVERY_COMMIT,
        "architecture": "SoftRoute(Option 1 [0.35 Logit-PCA(tl) + 0.35 Weibull(tl) + 0.30 DINOv2-RGB LKM] -> Scale14 + DINOv2-GN Weibull, s=5) + Large-Stone Correction",
        "backbone": DEFAULT_DINOV2_MODEL,
        "training_samples": int(len(labels_df)),
        "training_photos": int(n_tr),
        "sieve_diameters_mm": [float(c) for c in TARGET_COLUMNS],
        "validation_metrics": {
            "kaggle_public_emd": 37.32208,
            "option1_kaggle_public_emd": 43.66208,
            "location_15_site_oof_emd": 40.10,
            "cross_phone_emd": 45.85,
            "option1_location_15_site_oof_emd": 37.83,
        },
        "train_ppm": round(float(train_ppm), 4),
        "train_leverage_p95": round(lev_p95, 4),
        "artifact_size_bytes": int(artifact_path.stat().st_size),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return artifact_path


def verify_against_submission(
    artifact_path: Path = ARTIFACT_PATH,
    submission_path: Path = Path("submissions/submission.csv"),
    scratch_dir: Path | None = None,
) -> float:
    """Verify exported artifact reproduces the 37.32208 submission within numerical tolerance."""
    from backend.inference import SoilVisionEngine

    data_dir = resolve_data_dir()
    test_m = build_image_manifest(data_dir, "test")
    sub_df = pd.read_csv(submission_path).set_index("sample_id")[TARGET_COLUMNS].astype(float)

    engine = SoilVisionEngine(artifact_path=artifact_path, load_backbone=False)
    ref = np.load(scratch_dir / "ref_feats.npz")
    a1 = np.load(scratch_dir / "a1_rgb.npz")
    a2 = np.load(scratch_dir / "a2_gn.npz")
    n_tr = len(build_image_manifest(data_dir, "train"))

    x_tl_te = ref["X_sand2"][n_tr:]
    x_b5_te = ref["X_b5"][n_tr:]
    x_dino_rgb_te = a1["te"]
    x_dino_gn_te = a2["te"]
    te_sids = test_m["sample_id"].to_numpy()

    sam_stones = {
        "HPC_Muenster_BS6_9_0-10m": (0.227, 0.655),
        "HPC_Testfeld Lidl WHV": (0.000, 0.372),
        "HPC_Audorfring": (0.000, 0.051),
    }

    diffs = []
    for sid in sub_df.index:
        m = te_sids == sid
        pred_curve, _ = engine.predict_from_features(
            x_tl=x_tl_te[m],
            x_b5=x_b5_te[m],
            x_dino_rgb=x_dino_rgb_te[m],
            x_dino_gn=x_dino_gn_te[m],
            stone_area=sam_stones.get(sid, (0.0, 0.0)),
        )
        d = float(emd_per_sample(sub_df.loc[sid].to_numpy(float)[None], pred_curve[None])[0])
        diffs.append(d)
    return float(max(diffs))


if __name__ == "__main__":
    scratch = Path("/home/antigravity/.gemini/jetski/brain/v1/default/jetski_remote_agents_3p_devbox/f303835a-8e03-4d31-9f25-9b428cc24723/scratch")
    path = export_artifact(scratch_dir=scratch if scratch.exists() else None)
    max_diff = verify_against_submission(artifact_path=path, scratch_dir=scratch)
    print(f"Exported {path} ({path.stat().st_size / 1024:.1f} KB); max EMD diff vs 37.32208 submission = {max_diff:.6f}")
