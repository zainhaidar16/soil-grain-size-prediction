import numpy as np
import pytest

from src.metric import DIAMETERS, LOG_WEIGHTS, TARGET_COLUMNS, emd_per_sample, mean_emd, project_curves, validate_curves


def test_target_columns_match_official_schema():
    expected = ["0.002", "0.0063", "0.02", "0.063", "0.2", "0.63", "2", "6.3", "20", "63", "200"]
    assert TARGET_COLUMNS == expected
    assert np.allclose([float(c) for c in TARGET_COLUMNS], DIAMETERS)


def test_perfect_prediction():
    x = np.array([0, 0, 0, 5, 12, 40, 70, 90, 99, 100, 100])
    assert mean_emd(x, x) == 0


def test_logarithmic_weighting():
    y = np.array([0] * 10 + [100])
    p = np.array([10] * 10 + [100])
    assert emd_per_sample(y, p)[0] == pytest.approx(10 * LOG_WEIGHTS.sum())


def test_projection_is_valid():
    p = project_curves([20, -1, 4, 45, 35, 60, 75, 70, 95, 95, 10])
    assert validate_curves(p).shape == (1, 11)


def test_invalid_sequence_rejected():
    with pytest.raises(ValueError, match="nondecreasing"):
        validate_curves([0, 2, 1, 5, 10, 20, 40, 60, 80, 90, 100])


def test_emd_range_and_wasserstein_equivalence():
    from scipy.stats import wasserstein_distance

    # Maximum possible EMD is 100 * log10(200 / 0.002) = 500.0
    finest = np.array([100.0] * 11)
    coarsest = np.array([0.0] * 10 + [100.0])
    assert emd_per_sample(finest, coarsest)[0] == pytest.approx(500.0)

    # Exact equivalence to 100 * Wasserstein-1 distance on log10(DIAMETERS)
    y_true = np.array([5.2, 10.2, 23.7, 50.3, 75.8, 85.7, 91.0, 94.9, 98.6, 100.0, 100.0])
    y_pred = np.array([3.1, 6.2, 14.5, 30.2, 45.7, 55.9, 64.7, 79.4, 93.4, 99.5, 100.0])
    pdf_true = np.diff(np.concatenate([[0.0], y_true])) / 100.0
    pdf_pred = np.diff(np.concatenate([[0.0], y_pred])) / 100.0
    expected = 100.0 * wasserstein_distance(
        np.log10(DIAMETERS), np.log10(DIAMETERS), u_weights=pdf_true, v_weights=pdf_pred
    )
    assert emd_per_sample(y_true, y_pred)[0] == pytest.approx(expected, rel=1e-6)


def test_log_quantile_round_trip_and_weibull_fit():
    from src.metric import (
        curve_to_log_quantiles,
        fit_weibull_params,
        log_quantiles_to_curve,
        weibull_to_curve,
    )

    curve = np.array([5.2, 10.2, 23.7, 50.3, 75.8, 85.7, 91.0, 94.9, 98.6, 100.0, 100.0])
    log_q = curve_to_log_quantiles(curve)
    assert np.all(np.diff(log_q) >= -1e-9)
    reconstructed = log_quantiles_to_curve(log_q)
    validate_curves(reconstructed)
    assert emd_per_sample(curve, reconstructed)[0] < 6.0

    params = fit_weibull_params(curve)
    w_curve = weibull_to_curve(params)
    validate_curves(w_curve)
    assert emd_per_sample(curve, w_curve)[0] < 16.0
    assert emd_per_sample(curve, reconstructed)[0] < emd_per_sample(curve, w_curve)[0]


def test_repair_isotonic_and_weibull_lnb_c():
    from src.metric import fit_weibull_lnb_c, repair_isotonic, weibull_lnb_c_to_curve

    raw = np.array([15.0, 10.0, 25.0, 55.0, 50.0, 70.0, 92.0, 88.0, 99.0, 102.0, 97.0])
    repaired = repair_isotonic(raw)
    validate_curves(repaired)
    assert repaired[-1] == pytest.approx(100.0)

    curve = np.array([1.0, 2.5, 6.5, 16.0, 38.0, 72.0, 96.0, 99.8, 100.0, 100.0, 100.0])
    ln_b, c = fit_weibull_lnb_c(curve)
    pred = weibull_lnb_c_to_curve(ln_b, c)
    validate_curves(pred)
    assert emd_per_sample(curve, pred)[0] < 5.0

