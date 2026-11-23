"""Log-weighted cumulative EMD and competition output validation."""
from __future__ import annotations

import numpy as np

DIAMETERS = np.array([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2, 6.3, 20, 63, 200])
TARGET_COLUMNS = ["0.002", "0.0063", "0.02", "0.063", "0.2", "0.63", "2", "6.3", "20", "63", "200"]
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


LOG_DIAMETERS = np.log10(DIAMETERS)
DEFAULT_QUANTILE_PROBS = np.array([5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 95.0])


def curve_to_log_quantiles(
    curve: np.ndarray,
    probs: np.ndarray = DEFAULT_QUANTILE_PROBS,
) -> np.ndarray:
    """Invert a cumulative passing curve (%) to log10(diameter) at given percentiles."""
    c = validate_curves(curve)[0]
    p_arr = np.asarray(probs, dtype=float)
    out = np.zeros(len(p_arr), dtype=float)
    for k, p in enumerate(p_arr):
        if p <= c[0]:
            if c[1] > c[0] + 1e-6:
                slope = (LOG_DIAMETERS[1] - LOG_DIAMETERS[0]) / (c[1] - c[0])
                val = LOG_DIAMETERS[0] - (c[0] - p) * slope
            else:
                val = LOG_DIAMETERS[0]
            out[k] = float(np.clip(val, LOG_DIAMETERS[0] - 1.0, LOG_DIAMETERS[0]))
        elif p >= c[-1]:
            out[k] = float(LOG_DIAMETERS[-1])
        else:
            idx = int(np.searchsorted(c, p, side="left"))
            if idx == 0:
                out[k] = float(LOG_DIAMETERS[0])
            else:
                c0, c1 = c[idx - 1], c[idx]
                if c1 <= c0 + 1e-12:
                    out[k] = float(LOG_DIAMETERS[idx])
                else:
                    t = (p - c0) / (c1 - c0)
                    out[k] = float(LOG_DIAMETERS[idx - 1] + t * (LOG_DIAMETERS[idx] - LOG_DIAMETERS[idx - 1]))
    return out


def log_quantiles_to_curve(
    log_q: np.ndarray,
    probs: np.ndarray = DEFAULT_QUANTILE_PROBS,
) -> np.ndarray:
    """Reconstruct a valid cumulative passing curve (%) at all 11 DIAMETERS from log10 quantiles."""
    q = np.maximum.accumulate(np.asarray(log_q, dtype=float))
    p_arr = np.asarray(probs, dtype=float)
    q_span = max(float(q[-1] - q[0]), 0.5)
    left_gap = max(float(q[1] - q[0]), 0.15 * q_span) if len(q) > 1 else 0.5
    right_gap = max(float(q[-1] - q[-2]), 0.15 * q_span) if len(q) > 1 else 0.5

    xp = np.concatenate([
        [q[0] - left_gap * (p_arr[0] / max(float(p_arr[1] - p_arr[0]), 1.0))],
        q,
        [q[-1] + right_gap * ((100.0 - p_arr[-1]) / max(float(p_arr[-1] - p_arr[-2]), 1.0))],
    ])
    fp = np.concatenate([[0.0], p_arr, [100.0]])
    for i in range(1, len(xp)):
        if xp[i] <= xp[i - 1]:
            xp[i] = xp[i - 1] + 1e-6

    raw = np.interp(LOG_DIAMETERS, xp, fp, left=0.0, right=100.0)
    raw[-1] = 100.0
    return project_curves(raw)[0]


def fit_weibull_params(curve: np.ndarray) -> np.ndarray:
    """Fit 2-parameter Weibull (log_b, log_c) by minimizing exact competition EMD."""
    from scipy.optimize import minimize

    row = validate_curves(curve)[0]

    def loss(params: np.ndarray) -> float:
        b, c = np.exp(params[0]), np.exp(params[1])
        pred = 100.0 * (1.0 - np.exp(-((DIAMETERS / b) ** c)))
        pred[-1] = 100.0
        return float((np.abs(row[:-1] - pred[:-1]) * LOG_WEIGHTS).sum())

    best = None
    for lb0 in (-2.0, -0.5, 1.0):
        for lc0 in (-0.7, 0.0, 0.5):
            res = minimize(loss, [lb0, lc0], method="Nelder-Mead")
            if best is None or res.fun < best.fun:
                best = res
    return np.asarray(best.x, dtype=float)


def weibull_to_curve(params: np.ndarray) -> np.ndarray:
    """Evaluate a 2-parameter Weibull (log_b, log_c) at all 11 DIAMETERS."""
    b = float(np.exp(np.clip(params[0], -8.0, 6.0)))
    c = float(np.exp(np.clip(params[1], -3.0, 3.0)))
    pred = 100.0 * (1.0 - np.exp(-((DIAMETERS / b) ** c)))
    pred[-1] = 100.0
    return project_curves(pred)[0]


def repair_isotonic(raw: np.ndarray) -> np.ndarray:
    """Project 11-point curves onto monotone [0, 100] CDFs via LOG_WEIGHTS-weighted isotonic regression."""
    from sklearn.isotonic import IsotonicRegression

    arr = np.asarray(raw, dtype=float)
    squeeze = arr.ndim == 1
    if squeeze:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != 11 or not np.isfinite(arr).all():
        raise ValueError("Expected finite 11-point curves")

    iso = IsotonicRegression(y_min=0.0, y_max=100.0)
    x_idx = np.arange(len(LOG_WEIGHTS), dtype=float)
    out = np.empty_like(arr, dtype=float)
    for i in range(len(arr)):
        clipped = np.clip(arr[i, :-1], 0.0, 100.0)
        out[i, :-1] = iso.fit_transform(x_idx, clipped, sample_weight=LOG_WEIGHTS)
        out[i, -1] = 100.0
    return out[0] if squeeze else out


def weibull_lnb_c_to_curve(ln_b: float, c: float) -> np.ndarray:
    """Evaluate 2-parameter Weibull F(d) = 100 * (1 - exp(-(d / exp(ln_b))^c)) at all 11 DIAMETERS."""
    b = float(np.exp(np.clip(ln_b, -10.0, 10.0)))
    shape = max(float(c), 0.05)
    pred = 100.0 * (1.0 - np.exp(-np.power(DIAMETERS / b, shape)))
    pred[-1] = 100.0
    return project_curves(pred)[0]


def fit_weibull_lnb_c(curve: np.ndarray) -> np.ndarray:
    """Fit Weibull (ln_b, c) by 2D grid initialization + Nelder-Mead minimization of exact EMD."""
    from scipy.optimize import minimize

    row = validate_curves(curve)[0]
    lnb_grid = np.linspace(np.log(1e-3), np.log(2e3), 80)
    c_grid = np.linspace(0.08, 3.0, 60)
    lg, cg = np.meshgrid(lnb_grid, c_grid, indexing="ij")
    grid = np.column_stack([lg.ravel(), cg.ravel()])
    g_curves = 100.0 * (1.0 - np.exp(-np.power(DIAMETERS[None, :-1] / np.exp(grid[:, :1]), grid[:, 1:])))
    errs = np.sum(LOG_WEIGHTS[None, :] * np.abs(g_curves - row[None, :-1]), axis=1)
    best_idx = int(np.argmin(errs))
    best_p = grid[best_idx]
    best_e = float(errs[best_idx])

    def loss(p: np.ndarray) -> float:
        pred = weibull_lnb_c_to_curve(float(p[0]), float(p[1]))
        return float(np.sum(LOG_WEIGHTS * np.abs(pred[:-1] - row[:-1])))

    res = minimize(loss, best_p, method="Nelder-Mead")
    if float(res.fun) < best_e:
        return np.array([float(res.x[0]), max(float(res.x[1]), 0.05)], dtype=float)
    return np.array([float(best_p[0]), float(best_p[1])], dtype=float)

