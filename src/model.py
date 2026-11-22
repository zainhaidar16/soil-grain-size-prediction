"""Leakage-free LOOCV Ridge regression model for cumulative grain-size prediction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.metric import TARGET_COLUMNS, emd_per_sample, mean_emd, project_curves, validate_curves

DEFAULT_ALPHA_GRID: tuple[float, ...] = (
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0,
    3000.0,
    10000.0,
)


@dataclass(frozen=True)
class LOOCVResult:
    """Container for leave-one-sample-out evaluation results."""

    mean_emd: float
    per_sample_emd: np.ndarray
    predictions: np.ndarray
    selected_alphas: list[float]


def fit_predict_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fit StandardScaler + Ridge on training samples only and return physically valid curves."""
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_train)
    x_ev = scaler.transform(x_eval)

    reg = Ridge(alpha=float(alpha))
    reg.fit(x_tr, y_train)
    raw_pred = reg.predict(x_ev)
    return project_curves(raw_pred)


def evaluate_loocv_ridge_fixed_alpha(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> LOOCVResult:
    """Leakage-free leave-one-sample-out CV for a single Ridge alpha."""
    x = np.asarray(x, dtype=float)
    y = validate_curves(y)
    n_samples = len(y)
    if len(x) != n_samples or n_samples < 2:
        raise ValueError("x and y must have identical length >= 2")

    preds = np.zeros_like(y, dtype=float)
    for i in range(n_samples):
        mask = np.arange(n_samples) != i
        preds[i] = fit_predict_ridge(x[mask], y[mask], x[i : i + 1], alpha=alpha)[0]

    per_sample = emd_per_sample(y, preds)
    return LOOCVResult(
        mean_emd=float(per_sample.mean()),
        per_sample_emd=per_sample,
        predictions=preds,
        selected_alphas=[float(alpha)] * n_samples,
    )


def evaluate_nested_loocv_ridge(
    x: np.ndarray,
    y: np.ndarray,
    alphas: Sequence[float] = DEFAULT_ALPHA_GRID,
) -> LOOCVResult:
    """Strictly leakage-free nested LOOCV: selects Ridge alpha inside each outer fold."""
    x = np.asarray(x, dtype=float)
    y = validate_curves(y)
    n_samples = len(y)
    if len(x) != n_samples or n_samples < 3:
        raise ValueError("Nested LOOCV requires at least 3 samples")
    if not alphas:
        raise ValueError("alphas must be non-empty")

    preds = np.zeros_like(y, dtype=float)
    selected_alphas: list[float] = []

    for i in range(n_samples):
        train_mask = np.arange(n_samples) != i
        x_tr, y_tr = x[train_mask], y[train_mask]

        best_alpha = float(alphas[0])
        best_inner_emd = float("inf")
        for alpha in alphas:
            inner = evaluate_loocv_ridge_fixed_alpha(x_tr, y_tr, alpha=float(alpha))
            if inner.mean_emd < best_inner_emd:
                best_inner_emd = inner.mean_emd
                best_alpha = float(alpha)

        selected_alphas.append(best_alpha)
        preds[i] = fit_predict_ridge(x_tr, y_tr, x[i : i + 1], alpha=best_alpha)[0]

    per_sample = emd_per_sample(y, preds)
    return LOOCVResult(
        mean_emd=float(per_sample.mean()),
        per_sample_emd=per_sample,
        predictions=preds,
        selected_alphas=selected_alphas,
    )


def select_best_alpha_loocv(
    x: np.ndarray,
    y: np.ndarray,
    alphas: Sequence[float] = DEFAULT_ALPHA_GRID,
) -> tuple[float, pd.DataFrame]:
    """Evaluate a grid of Ridge alphas using LOOCV and return (best_alpha, summary_df)."""
    rows = []
    for alpha in alphas:
        res = evaluate_loocv_ridge_fixed_alpha(x, y, alpha=float(alpha))
        rows.append({"alpha": float(alpha), "loocv_emd": res.mean_emd})
    summary = pd.DataFrame(rows).sort_values("loocv_emd", ascending=True).reset_index(drop=True)
    return float(summary.loc[0, "alpha"]), summary


def write_model_submission(
    test_sample_ids: Sequence[str],
    predictions: np.ndarray,
    template_path: Path | str,
    output_path: Path | str,
) -> pd.DataFrame:
    """Write and validate a competition submission CSV aligned with sample_submission.csv."""
    template = pd.read_csv(template_path)
    if "sample_id" not in template or template["sample_id"].isna().any():
        raise ValueError("Invalid sample_submission.csv")
    if template["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_ids in template")

    validated = validate_curves(predictions)
    if len(test_sample_ids) != len(validated):
        raise ValueError("test_sample_ids and predictions must have the same length")

    pred_df = pd.DataFrame(validated, columns=TARGET_COLUMNS)
    pred_df.insert(0, "sample_id", list(test_sample_ids))

    if set(pred_df["sample_id"]) != set(template["sample_id"]):
        raise ValueError("Predicted sample_ids do not match sample_submission.csv")

    # Reorder rows and columns to match the official template exactly.
    pred_df = pred_df.set_index("sample_id").loc[template["sample_id"]].reset_index()
    if set(pred_df.columns) != set(template.columns):
        raise ValueError("Submission columns do not match sample_submission.csv")
    pred_df = pred_df[template.columns]

    if pred_df.isna().any().any():
        raise ValueError("Submission contains missing values")
    validate_curves(pred_df[TARGET_COLUMNS].to_numpy(dtype=float))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_path, index=False)
    return pred_df


DEFAULT_PLS_COMPONENTS: tuple[int, ...] = (2, 3, 4, 5)


def fit_predict_pls(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """Fit StandardScaler + PLSRegression on training samples only and return valid curves."""
    from sklearn.cross_decomposition import PLSRegression

    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_train)
    x_ev = scaler.transform(x_eval)

    k = max(1, min(int(n_components), x_tr.shape[0] - 1, x_tr.shape[1]))
    pls = PLSRegression(n_components=k)
    pls.fit(x_tr, y_train)
    return project_curves(pls.predict(x_ev))


def evaluate_loocv_pls_fixed_components(
    x: np.ndarray,
    y: np.ndarray,
    n_components: int,
) -> LOOCVResult:
    """Leakage-free leave-one-sample-out CV for a fixed number of PLS components."""
    x = np.asarray(x, dtype=float)
    y = validate_curves(y)
    n_samples = len(y)
    if len(x) != n_samples or n_samples < 2:
        raise ValueError("x and y must have identical length >= 2")

    preds = np.zeros_like(y, dtype=float)
    for i in range(n_samples):
        mask = np.arange(n_samples) != i
        preds[i] = fit_predict_pls(x[mask], y[mask], x[i : i + 1], n_components=n_components)[0]

    per_sample = emd_per_sample(y, preds)
    return LOOCVResult(
        mean_emd=float(per_sample.mean()),
        per_sample_emd=per_sample,
        predictions=preds,
        selected_alphas=[float(n_components)] * n_samples,
    )


def evaluate_nested_loocv_pls(
    x: np.ndarray,
    y: np.ndarray,
    components: Sequence[int] = DEFAULT_PLS_COMPONENTS,
) -> LOOCVResult:
    """Strictly leakage-free nested LOOCV selecting PLS n_components inside each outer fold."""
    x = np.asarray(x, dtype=float)
    y = validate_curves(y)
    n_samples = len(y)
    if len(x) != n_samples or n_samples < 3:
        raise ValueError("Nested LOOCV requires at least 3 samples")
    if not components:
        raise ValueError("components must be non-empty")

    preds = np.zeros_like(y, dtype=float)
    selected: list[float] = []
    for i in range(n_samples):
        train_mask = np.arange(n_samples) != i
        x_tr, y_tr = x[train_mask], y[train_mask]

        best_k = int(components[0])
        best_inner_emd = float("inf")
        for k in components:
            inner = evaluate_loocv_pls_fixed_components(x_tr, y_tr, n_components=int(k))
            if inner.mean_emd < best_inner_emd:
                best_inner_emd = inner.mean_emd
                best_k = int(k)

        selected.append(float(best_k))
        preds[i] = fit_predict_pls(x_tr, y_tr, x[i : i + 1], n_components=best_k)[0]

    per_sample = emd_per_sample(y, preds)
    return LOOCVResult(
        mean_emd=float(per_sample.mean()),
        per_sample_emd=per_sample,
        predictions=preds,
        selected_alphas=selected,
    )


def select_best_pls_components_loocv(
    x: np.ndarray,
    y: np.ndarray,
    components: Sequence[int] = DEFAULT_PLS_COMPONENTS,
) -> tuple[int, pd.DataFrame]:
    """Select optimal PLS n_components using LOOCV EMD."""
    rows = []
    for k in components:
        res = evaluate_loocv_pls_fixed_components(x, y, n_components=int(k))
        rows.append({"n_components": int(k), "loocv_emd": res.mean_emd})
    summary = pd.DataFrame(rows).sort_values("loocv_emd", ascending=True).reset_index(drop=True)
    return int(summary.loc[0, "n_components"]), summary


def fit_predict_hybrid_ensemble(
    x_tex_train: np.ndarray,
    x_dino_train: np.ndarray,
    y_train: np.ndarray,
    x_tex_eval: np.ndarray,
    x_dino_eval: np.ndarray,
    ridge_alpha: float = 30.0,
    pls_components: int = 2,
    w_tex: float = 0.5,
) -> np.ndarray:
    """Blend physically calibrated texture Ridge + GrayNorm DINOv2 PLS predictions."""
    if not (0.0 <= w_tex <= 1.0):
        raise ValueError("w_tex must be in [0, 1]")
    pred_tex = fit_predict_ridge(x_tex_train, y_train, x_tex_eval, alpha=ridge_alpha)
    pred_dino = fit_predict_pls(x_dino_train, y_train, x_dino_eval, n_components=pls_components)
    return project_curves(w_tex * pred_tex + (1.0 - w_tex) * pred_dino)


def main() -> None:
    import argparse

    from src.baseline import loocv_mean_baseline, read_labels
    from src.dataset import build_image_manifest, resolve_data_dir
    from src.features import (
        DEFAULT_DINOV2_MODEL,
        DINOv2FeatureExtractor,
        aggregate_sample_features,
        extract_physical_texture_features,
    )

    parser = argparse.ArgumentParser(description="Frozen DINOv2 / Hybrid grain-size model.")
    parser.add_argument("--mode", choices=("ridge", "hybrid"), default="ridge")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default=DEFAULT_DINOV2_MODEL)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--w-tex", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("submissions/submission.csv"))
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    labels_df = read_labels(data_dir / "Training_labels_updated.csv")
    template_path = data_dir / "sample_submission.csv"
    test_ids = pd.read_csv(template_path)["sample_id"].tolist()

    train_manifest = build_image_manifest(data_dir, "train")
    test_manifest = build_image_manifest(data_dir, "test")
    train_ids = labels_df["sample_id"].tolist()
    y_train = labels_df[TARGET_COLUMNS].to_numpy(dtype=float)
    baseline_emd = loocv_mean_baseline(labels_df)

    if args.mode == "ridge":
        extractor = DINOv2FeatureExtractor(model_name=args.model_name, image_size=args.image_size)
        train_img_feats = extractor.extract_manifest_features(train_manifest, batch_size=args.batch_size)
        test_img_feats = extractor.extract_manifest_features(test_manifest, batch_size=args.batch_size)

        x_train = aggregate_sample_features(train_img_feats, train_manifest["sample_id"].tolist(), train_ids)
        x_test = aggregate_sample_features(test_img_feats, test_manifest["sample_id"].tolist(), test_ids)

        nested = evaluate_nested_loocv_ridge(x_train, y_train, alphas=DEFAULT_ALPHA_GRID)
        best_alpha, _ = select_best_alpha_loocv(x_train, y_train, alphas=DEFAULT_ALPHA_GRID)

        print(f"Train samples: {len(train_ids)} ({len(train_manifest)} images)")
        print(f"Test samples:  {len(test_ids)} ({len(test_manifest)} images)")
        print(f"Mean baseline LOOCV EMD:        {baseline_emd:.6f}")
        print(f"DINOv2+Ridge nested LOOCV EMD:  {nested.mean_emd:.6f}")
        print(f"Improvement over baseline:      {baseline_emd - nested.mean_emd:+.6f}")

        if nested.mean_emd < baseline_emd:
            test_preds = fit_predict_ridge(x_train, y_train, x_test, alpha=best_alpha)
            write_model_submission(test_ids, test_preds, template_path, args.output)
            print(f"Saved validated submission ({len(test_preds)} rows) using alpha={best_alpha}: {args.output}")
        else:
            print("Model did not beat baseline; submission not written.")
    else:
        tr_tex_img = extract_physical_texture_features(train_manifest)
        te_tex_img = extract_physical_texture_features(test_manifest)
        x_tex_tr = aggregate_sample_features(tr_tex_img, train_manifest["sample_id"].tolist(), train_ids, agg="mean")
        x_tex_te = aggregate_sample_features(te_tex_img, test_manifest["sample_id"].tolist(), test_ids, agg="mean")

        extractor = DINOv2FeatureExtractor(
            model_name=args.model_name,
            image_size=args.image_size,
            gray_norm=True,
        )
        tr_dino_img = extractor.extract_manifest_features(train_manifest, batch_size=args.batch_size)
        te_dino_img = extractor.extract_manifest_features(test_manifest, batch_size=args.batch_size)
        x_dino_tr = aggregate_sample_features(tr_dino_img, train_manifest["sample_id"].tolist(), train_ids, agg="mean")
        x_dino_te = aggregate_sample_features(te_dino_img, test_manifest["sample_id"].tolist(), test_ids, agg="mean")

        tex_alphas = (1.0, 3.0, 10.0, 30.0, 100.0)
        nested_tex = evaluate_nested_loocv_ridge(x_tex_tr, y_train, alphas=tex_alphas)
        nested_dino = evaluate_nested_loocv_pls(x_dino_tr, y_train, components=DEFAULT_PLS_COMPONENTS)
        hybrid_oof = project_curves(args.w_tex * nested_tex.predictions + (1.0 - args.w_tex) * nested_dino.predictions)
        hybrid_emd = mean_emd(y_train, hybrid_oof)

        best_alpha, _ = select_best_alpha_loocv(x_tex_tr, y_train, alphas=tex_alphas)
        best_k, _ = select_best_pls_components_loocv(x_dino_tr, y_train, components=DEFAULT_PLS_COMPONENTS)

        print(f"Mean baseline LOOCV EMD:         {baseline_emd:.6f}")
        print(f"PhysTex-L2 + Ridge Nested LOOCV: {nested_tex.mean_emd:.6f} (alpha={best_alpha})")
        print(f"GrayNorm DINOv2 + PLS Nested:    {nested_dino.mean_emd:.6f} (n_components={best_k})")
        print(f"Hybrid ({args.w_tex:.2f}/{1-args.w_tex:.2f}) Nested LOOCV:  {hybrid_emd:.6f}")

        if hybrid_emd < baseline_emd:
            test_preds = fit_predict_hybrid_ensemble(
                x_tex_tr, x_dino_tr, y_train, x_tex_te, x_dino_te,
                ridge_alpha=best_alpha, pls_components=best_k, w_tex=args.w_tex,
            )
            write_model_submission(test_ids, test_preds, template_path, args.output)
            print(f"Saved validated hybrid submission ({len(test_preds)} rows): {args.output}")
        else:
            print("Hybrid model did not beat baseline; submission not written.")


if __name__ == "__main__":
    main()
