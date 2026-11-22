from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.metric import TARGET_COLUMNS, validate_curves
from src.model import (
    evaluate_loocv_ridge_fixed_alpha,
    evaluate_nested_loocv_ridge,
    fit_predict_ridge,
    write_model_submission,
)


def _synthetic_dataset(n_samples: int = 12, n_features: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, n_features))
    # Construct smooth monotonic curves correlated with x[:, 0]
    t = np.linspace(0.0, 1.0, 11)
    shifts = 1.0 / (1.0 + np.exp(-x[:, 0]))
    y = np.vstack([np.clip(100.0 * (t ** (0.5 + 1.5 * s)), 0.0, 100.0) for s in shifts])
    y[:, -1] = 100.0
    return x, y


def test_loocv_ridge_produces_valid_curves_and_finite_emd():
    x, y = _synthetic_dataset(n_samples=12, n_features=8)
    fixed = evaluate_loocv_ridge_fixed_alpha(x, y, alpha=10.0)
    assert fixed.predictions.shape == y.shape
    assert validate_curves(fixed.predictions).shape == y.shape
    assert np.isfinite(fixed.mean_emd) and 0.0 <= fixed.mean_emd <= 500.0

    nested = evaluate_nested_loocv_ridge(x, y, alphas=(1.0, 10.0, 100.0))
    assert nested.predictions.shape == y.shape
    assert validate_curves(nested.predictions).shape == y.shape
    assert len(nested.selected_alphas) == len(y)
    assert np.isfinite(nested.mean_emd) and 0.0 <= nested.mean_emd <= 500.0


def test_loocv_is_leakage_free():
    """Perturbing a held-out sample's target must not change its own LOOCV prediction."""
    x, y = _synthetic_dataset(n_samples=10, n_features=6, seed=42)
    res_orig = evaluate_loocv_ridge_fixed_alpha(x, y, alpha=10.0)

    y_mod = y.copy()
    y_mod[0] = np.array([100.0] * 11)
    res_mod = evaluate_loocv_ridge_fixed_alpha(x, y_mod, alpha=10.0)

    assert np.allclose(res_orig.predictions[0], res_mod.predictions[0])


def test_write_model_submission_validates_and_reorders(tmp_path: Path):
    template_path = tmp_path / "sample_submission.csv"
    out_path = tmp_path / "submission.csv"

    template = pd.DataFrame(0.0, index=range(3), columns=["sample_id", *TARGET_COLUMNS])
    template["sample_id"] = ["S2", "S1", "S3"]
    template.to_csv(template_path, index=False)

    x_tr, y_tr = _synthetic_dataset(n_samples=10, n_features=5, seed=7)
    x_te = np.zeros((3, 5))
    preds = fit_predict_ridge(x_tr, y_tr, x_te, alpha=10.0)

    # Deliberately pass sample_ids in a different order than template
    written = write_model_submission(["S1", "S2", "S3"], preds, template_path, out_path)
    assert out_path.exists()
    assert list(written.columns) == list(template.columns)
    assert written["sample_id"].tolist() == ["S2", "S1", "S3"]
    assert not written.isna().any().any()
    validate_curves(written[TARGET_COLUMNS].to_numpy(dtype=float))

    with pytest.raises(ValueError, match="do not match"):
        write_model_submission(["S1", "S2", "BAD_ID"], preds, template_path, out_path)


def test_nested_pls_and_hybrid_ensemble():
    from src.model import evaluate_nested_loocv_pls, fit_predict_hybrid_ensemble

    x_tex, y = _synthetic_dataset(n_samples=12, n_features=6, seed=11)
    x_dino, _ = _synthetic_dataset(n_samples=12, n_features=10, seed=11)

    pls_res = evaluate_nested_loocv_pls(x_dino, y, components=(2, 3))
    assert pls_res.predictions.shape == y.shape
    assert validate_curves(pls_res.predictions).shape == y.shape
    assert np.isfinite(pls_res.mean_emd) and 0.0 <= pls_res.mean_emd <= 500.0

    hyb_preds = fit_predict_hybrid_ensemble(
        x_tex[:10], x_dino[:10], y[:10], x_tex[10:], x_dino[10:],
        ridge_alpha=10.0, pls_components=2, w_tex=0.5,
    )
    assert hyb_preds.shape == (2, 11)
    validate_curves(hyb_preds)
