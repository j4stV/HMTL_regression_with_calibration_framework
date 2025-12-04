from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from src.utils.logger import get_logger


@dataclass
class ConformalResults:
    """Results from conformal calibration."""
    quantile: float
    coverage: float
    mean_width: float
    lower: np.ndarray
    upper: np.ndarray


def split_conformal_intervals(
    y_true_cal: np.ndarray,
    y_pred_cal: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """Compute conformal quantile for given alpha (miscoverage rate).
    
    Args:
        y_true_cal: True values on calibration set
        y_pred_cal: Predictions on calibration set
        alpha: Miscoverage rate (e.g., 0.1 for 90% coverage)
    
    Returns:
        Quantile q such that PI = [y_pred - q, y_pred + q] has coverage ~(1-alpha)
    """
    logger = get_logger("eval.conformal")
    logger.debug(f"Computing conformal intervals: alpha={alpha}, n_samples={len(y_true_cal)}")
    
    residuals = np.abs(y_true_cal - y_pred_cal)
    logger.debug(
        f"Residuals: mean={np.mean(residuals):.6f}, std={np.std(residuals):.6f}, "
        f"min={np.min(residuals):.6f}, max={np.max(residuals):.6f}"
    )
    
    q = np.quantile(residuals, 1 - alpha, method="higher")
    logger.info(f"Conformal quantile (1-alpha={1-alpha:.2f}): {q:.6f}")
    
    return float(q)


def apply_intervals(y_pred: np.ndarray, q: float) -> Tuple[np.ndarray, np.ndarray]:
    """Apply conformal intervals to predictions.
    
    Args:
        y_pred: Predictions
        q: Conformal quantile
    
    Returns:
        lower, upper: Prediction intervals
    """
    logger = get_logger("eval.conformal")
    logger.debug(f"Applying intervals with quantile q={q:.6f} to {len(y_pred)} predictions")
    
    lower = y_pred - q
    upper = y_pred + q
    
    interval_widths = upper - lower
    logger.debug(
        f"Interval widths: mean={np.mean(interval_widths):.6f}, "
        f"std={np.std(interval_widths):.6f}, "
        f"min={np.min(interval_widths):.6f}, max={np.max(interval_widths):.6f}"
    )
    
    return lower, upper


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Compute coverage rate of prediction intervals.
    
    Args:
        y_true: True values
        lower: Lower bounds
        upper: Upper bounds
    
    Returns:
        Coverage rate (fraction of points covered)
    """
    logger = get_logger("eval.conformal")
    
    coverage_rate = float(np.mean((y_true >= lower) & (y_true <= upper)))
    n_covered = np.sum((y_true >= lower) & (y_true <= upper))
    n_total = len(y_true)
    
    logger.info(f"Coverage: {n_covered}/{n_total} ({coverage_rate:.4%})")
    logger.debug(f"Lower bound: mean={np.mean(lower):.6f}, upper bound: mean={np.mean(upper):.6f}")
    
    return coverage_rate


def calibrate_multiple_levels(
    y_true_cal: np.ndarray,
    y_pred_cal: np.ndarray,
    coverage_levels: list[float] = [0.80, 0.90, 0.95],
) -> Dict[float, ConformalResults]:
    """Calibrate prediction intervals for multiple coverage levels.
    
    Args:
        y_true_cal: True values on calibration set
        y_pred_cal: Predictions on calibration set
        coverage_levels: List of desired coverage levels (e.g., [0.80, 0.90, 0.95])
    
    Returns:
        Dictionary mapping coverage level to ConformalResults
    """
    logger = get_logger("eval.conformal")
    logger.info(f"Calibrating for coverage levels: {coverage_levels}")
    
    results = {}
    
    for target_coverage in coverage_levels:
        alpha = 1.0 - target_coverage
        q = split_conformal_intervals(y_true_cal, y_pred_cal, alpha=alpha)
        lower, upper = apply_intervals(y_pred_cal, q)
        cov = coverage(y_true_cal, lower, upper)
        mean_width = np.mean(upper - lower)
        
        results[target_coverage] = ConformalResults(
            quantile=q,
            coverage=cov,
            mean_width=mean_width,
            lower=lower,
            upper=upper,
        )
        
        logger.info(
            f"Coverage {target_coverage:.0%}: quantile={q:.6f}, "
            f"actual_coverage={cov:.4%}, mean_width={mean_width:.6f}"
        )
    
    return results


def compute_pi_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Dict[str, float]:
    """Compute metrics for prediction intervals.
    
    Args:
        y_true: True values
        lower: Lower bounds
        upper: Upper bounds
    
    Returns:
        Dictionary with coverage, mean_width, median_width, etc.
    """
    cov = coverage(y_true, lower, upper)
    widths = upper - lower
    mean_width = float(np.mean(widths))
    median_width = float(np.median(widths))
    std_width = float(np.std(widths))
    
    return {
        "coverage": cov,
        "mean_width": mean_width,
        "median_width": median_width,
        "std_width": std_width,
        "min_width": float(np.min(widths)),
        "max_width": float(np.max(widths)),
    }


