from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from src.eval.conformal import ConformalResults, coverage, compute_pi_metrics
from src.utils.logger import get_logger


def cqr_nonconformity_scores(
    y_true: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
) -> np.ndarray:
    """Compute CQR non-conformity scores.

    E_i = max(q_lo_i - y_i, y_i - q_hi_i)

    Args:
        y_true: True values (n_samples,).
        q_lo: Lower quantile predictions (n_samples,).
        q_hi: Upper quantile predictions (n_samples,).

    Returns:
        Non-conformity scores (n_samples,).
    """
    return np.maximum(q_lo - y_true, y_true - q_hi)


def cqr_calibrate(
    y_true: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    alpha: float,
) -> float:
    """Compute CQR conformal quantile.

    Uses finite-sample correction: quantile level = ceil((n+1)*(1-alpha)) / n.

    Args:
        y_true: True values on calibration set.
        q_lo: Lower quantile predictions on calibration set.
        q_hi: Upper quantile predictions on calibration set.
        alpha: Miscoverage rate.

    Returns:
        CQR conformal adjustment value.
    """
    scores = cqr_nonconformity_scores(y_true, q_lo, q_hi)
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = float(np.clip(q_level, 0.0, 1.0))
    return float(np.quantile(scores, q_level, method="higher"))


def cqr_apply_intervals(
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    q_cqr: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply CQR conformal adjustment to quantile predictions.

    Args:
        q_lo: Lower quantile predictions (n_samples,).
        q_hi: Upper quantile predictions (n_samples,).
        q_cqr: CQR conformal adjustment value.

    Returns:
        (lower, upper) adjusted prediction intervals.
    """
    return q_lo - q_cqr, q_hi + q_cqr


def cqr_calibrate_multiple_levels(
    y_true_cal: np.ndarray,
    q_lo_cal: np.ndarray,
    q_hi_cal: np.ndarray,
    q_lo_test: np.ndarray,
    q_hi_test: np.ndarray,
    y_true_test: np.ndarray,
    coverage_levels: list[float] | None = None,
) -> Dict[float, dict]:
    """CQR calibration for multiple coverage levels.

    Note: The quantile predictions (q_lo, q_hi) should correspond to the
    desired coverage level. For example, for 90% coverage, quantiles should
    be [0.05, 0.95].

    Args:
        y_true_cal: True values on calibration set.
        q_lo_cal: Lower quantile predictions on calibration set.
        q_hi_cal: Upper quantile predictions on calibration set.
        q_lo_test: Lower quantile predictions on test set.
        q_hi_test: Upper quantile predictions on test set.
        y_true_test: True values on test set.
        coverage_levels: Target coverage levels.

    Returns:
        Dictionary mapping coverage level to results dict.
    """
    logger = get_logger("eval.cqr")
    if coverage_levels is None:
        coverage_levels = [0.80, 0.90, 0.95]

    results = {}
    for target_coverage in coverage_levels:
        alpha = 1.0 - target_coverage
        q_cqr = cqr_calibrate(y_true_cal, q_lo_cal, q_hi_cal, alpha)
        lower, upper = cqr_apply_intervals(q_lo_test, q_hi_test, q_cqr)
        cov = float(np.mean((y_true_test >= lower) & (y_true_test <= upper)))
        widths = upper - lower
        mean_width = float(np.mean(widths))

        results[target_coverage] = {
            "q_cqr": q_cqr,
            "coverage": cov,
            "mean_width": mean_width,
            "lower": lower,
            "upper": upper,
        }

        logger.info(
            f"CQR Coverage {target_coverage:.0%}: q_cqr={q_cqr:.6f}, "
            f"actual_coverage={cov:.4%}, mean_width={mean_width:.6f}"
        )

    return results
