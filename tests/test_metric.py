import numpy as np
import pytest

from src.metric import LOG_WEIGHTS, emd_per_sample, mean_emd, project_curves, validate_curves


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
