from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.utils.logger import get_logger


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""
    rmse: float
    mse: float
    mae: float
    r_auc_mse: float
    mean_uncertainty: float
    mean_epistemic: float
    mean_aleatoric: float
    # Метрики из референсного репозитория
    rejection_ratio: float | None = None
    rejection_auc: float | None = None
    f_beta_auc: float | None = None
    f_beta_95: float | None = None


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray | None = None,
) -> Dict[str, float]:
    """Compute standard regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predictions
        uncertainty: Optional uncertainty estimates
    
    Returns:
        Dictionary with RMSE, MSE, MAE
    """
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    
    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
    }
    
    if uncertainty is not None:
        metrics["mean_uncertainty"] = float(np.mean(uncertainty))
        metrics["std_uncertainty"] = float(np.std(uncertainty))
    
    return metrics


def evaluate_comprehensive(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    epistemic: np.ndarray | None = None,
    aleatoric: np.ndarray | None = None,
    compute_reference_metrics: bool = True,
    error_threshold: float | None = None,
) -> EvaluationMetrics:
    """Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predictions
        uncertainty: Total uncertainty
        epistemic: Epistemic uncertainty (optional)
        aleatoric: Aleatoric uncertainty (optional)
        compute_reference_metrics: Вычислять ли метрики из референсного репозитория
        error_threshold: Порог ошибки для F-beta метрик (если None, используется медиана ошибок)
    
    Returns:
        EvaluationMetrics object
    """
    from src.eval.r_auc_mse import r_auc_mse
    from src.eval.reference_metrics import prr_regression, f_beta_metrics
    
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    
    errors = (y_true - y_pred) ** 2
    r_auc = r_auc_mse(errors, uncertainty)
    
    mean_uncertainty = float(np.mean(uncertainty))
    mean_epistemic = float(np.mean(epistemic)) if epistemic is not None else 0.0
    mean_aleatoric = float(np.mean(aleatoric)) if aleatoric is not None else 0.0
    
    # Метрики из референсного репозитория
    rejection_ratio = None
    rejection_auc = None
    f_beta_auc = None
    f_beta_95 = None
    
    if compute_reference_metrics:
        try:
            rejection_ratio, rejection_auc = prr_regression(y_true, y_pred, uncertainty)
            rejection_ratio = float(rejection_ratio)
            rejection_auc = float(rejection_auc)
            
            # F-beta метрики
            if error_threshold is None:
                error_threshold = float(np.median(errors))
            
            f_beta_auc_val, f_beta_95_val, _ = f_beta_metrics(errors, uncertainty, error_threshold)
            f_beta_auc = float(f_beta_auc_val)
            f_beta_95 = float(f_beta_95_val)
        except Exception as e:
            logger = get_logger("eval.metrics")
            logger.warning(f"Failed to compute reference metrics: {e}")
    
    return EvaluationMetrics(
        rmse=rmse,
        mse=mse,
        mae=mae,
        r_auc_mse=r_auc,
        mean_uncertainty=mean_uncertainty,
        mean_epistemic=mean_epistemic,
        mean_aleatoric=mean_aleatoric,
        rejection_ratio=rejection_ratio,
        rejection_auc=rejection_auc,
        f_beta_auc=f_beta_auc,
        f_beta_95=f_beta_95,
    )


def aggregate_metrics_across_seeds(
    metrics_list: list[EvaluationMetrics],
) -> Dict[str, tuple[float, float]]:
    """Aggregate metrics across multiple seeds.
    
    Args:
        metrics_list: List of EvaluationMetrics from different seeds
    
    Returns:
        Dictionary mapping metric name to (mean, std) tuple
    """
    aggregated = {}
    
    # Extract all metrics
    metric_names = [
        "rmse", "mse", "mae", "r_auc_mse",
        "mean_uncertainty", "mean_epistemic", "mean_aleatoric",
        "rejection_ratio", "rejection_auc", "f_beta_auc", "f_beta_95"
    ]
    
    for name in metric_names:
        values = [getattr(m, name) for m in metrics_list]
        aggregated[name] = (float(np.mean(values)), float(np.std(values)))
    
    return aggregated

