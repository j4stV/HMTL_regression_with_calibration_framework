from __future__ import annotations

import numpy as np

from src.utils.logger import get_logger


def error_retention_curve(mse_per_point: np.ndarray, uncertainty: np.ndarray, num_thresholds: int = 50) -> tuple[np.ndarray, np.ndarray]:
    logger = get_logger("eval.r_auc_mse")
    assert mse_per_point.shape == uncertainty.shape
    
    logger.debug(f"Computing error retention curve: n_samples={len(mse_per_point)}, num_thresholds={num_thresholds}")
    logger.debug(f"MSE: mean={np.mean(mse_per_point):.6f}, std={np.std(mse_per_point):.6f}")
    logger.debug(f"Uncertainty: mean={np.mean(uncertainty):.6f}, std={np.std(uncertainty):.6f}")
    
    order = np.argsort(uncertainty)  # от наименьшей неопределённости к наибольшей
    mse_sorted = mse_per_point[order]
    x = []
    y = []
    n = len(mse_sorted)
    for k in np.linspace(1, n, num_thresholds, dtype=int):
        kept = mse_sorted[:k]
        x.append(k / n)
        y.append(np.mean(kept))
    
    logger.debug(f"Error retention curve computed: {len(x)} points")
    return np.array(x), np.array(y)


def r_auc_mse(mse_per_point: np.ndarray, uncertainty: np.ndarray) -> float:
    logger = get_logger("eval.r_auc_mse")
    
    x, y = error_retention_curve(mse_per_point, uncertainty)
    # интеграл по кусочно-линейной кривой
    score = float(np.trapz(y, x))
    
    logger.debug(f"R-AUC MSE score: {score:.6f}")
    return score


