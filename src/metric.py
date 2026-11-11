"""Log-weighted cumulative EMD and competition output validation."""
from __future__ import annotations

import numpy as np

DIAMETERS = np.array([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2, 6.3, 20, 63, 200])
TARGET_COLUMNS = ["0.002", "0.0063", "0.02", "0.063", "0.2", "0.63", "2.0", "6.3", "20", "63", "200"]
LOG_WEIGHTS = np.log10(DIAMETERS[1:] / DIAMETERS[:-1])


def validate_curves(curves: np.ndarray) -> np.ndarray:
    arr = np.asarray(curves, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != 11:
        raise ValueError("Expected shape (n_samples, 11)")
    if not np.isfinite(arr).all():
        raise ValueError("Values must be finite")
    if ((arr < 0) | (arr > 100)).any():
        raise ValueError("Percentages must be between 0 and 100")
    if (np.diff(arr, axis=1) < -1e-8).any():
        raise ValueError("Cumulative percentages must be nondecreasing")
    if not np.allclose(arr[:, -1], 100, atol=1e-6):
        raise ValueError("Last percentage must be 100")
    return arr


def project_curves(raw: np.ndarray) -> np.ndarray:
    """Basic physical constraint projection (not optimal EMD projection)."""
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != 11 or not np.isfinite(arr).all():
        raise ValueError("Expected finite 11-point curves")
    arr = np.maximum.accumulate(np.clip(arr, 0, 100), axis=1)
    arr[:, -1] = 100.
    return arr


def emd_per_sample(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Weighted sum of absolute CDF differences over log10-diameter bins."""
    truth, pred = validate_curves(y_true), validate_curves(y_pred)
    if truth.shape != pred.shape:
        raise ValueError("Shapes must agree")
    return (np.abs(truth[:, :-1] - pred[:, :-1]) * LOG_WEIGHTS).sum(axis=1)


def mean_emd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(emd_per_sample(y_true, y_pred).mean())
