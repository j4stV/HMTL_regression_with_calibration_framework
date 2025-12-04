from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.eval.conformal import ConformalResults, calibrate_multiple_levels, compute_pi_metrics
from src.eval.ensemble import ensemble_predict
from src.eval.metrics import EvaluationMetrics, evaluate_comprehensive
from src.eval.r_auc_mse import error_retention_curve
from src.models.hmtl import HMTLModel
from src.utils.logger import get_logger


@dataclass
class EvaluationResults:
    """Complete evaluation results."""
    metrics: EvaluationMetrics
    conformal_results: Dict[float, ConformalResults]
    pi_metrics_before: Dict[float, Dict[str, float]]
    pi_metrics_after: Dict[float, Dict[str, float]]
    error_retention_x: np.ndarray
    error_retention_y: np.ndarray
    # Rejection curve из референсного репозитория
    rejection_curve: np.ndarray | None = None


def evaluate_on_dataset(
    models: List[HMTLModel],
    X: np.ndarray,
    y_true: np.ndarray,
    X_cal: np.ndarray | None = None,
    y_cal: np.ndarray | None = None,
    coverage_levels: list[float] = [0.80, 0.90, 0.95],
    device: torch.device | None = None,
    preprocessor=None,
) -> EvaluationResults:
    """Comprehensive evaluation on a dataset.
    
    Args:
        models: List of trained models
        X: Features
        y_true: True target values
        X_cal: Calibration features (optional, for conformal calibration)
        y_cal: Calibration targets (optional, for conformal calibration)
        coverage_levels: Desired coverage levels for conformal calibration
        device: Device to use for inference
    
    Returns:
        EvaluationResults with all metrics and calibration results
    """
    logger = get_logger("eval.evaluator")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Evaluating on {len(X)} samples with {len(models)} models")
    
    # Get ensemble predictions with uncertainty (in standardized space)
    y_pred, uncertainty_total, uncertainty_epistemic, uncertainty_aleatoric = ensemble_predict(
        models, X, device=device
    )
    
    # Transform uncertainty back to original space if preprocessor is provided
    if preprocessor is not None:
        logger.info("Transforming uncertainty from standardized to original space")
        uncertainty_total_original = preprocessor.inverse_transform_uncertainty(uncertainty_total)
        uncertainty_epistemic_original = preprocessor.inverse_transform_uncertainty(uncertainty_epistemic)
        uncertainty_aleatoric_original = preprocessor.inverse_transform_uncertainty(uncertainty_aleatoric)
        
        logger.info(
            f"Uncertainty transformation - "
            f"Standardized: total={np.mean(uncertainty_total):.6f}, "
            f"epistemic={np.mean(uncertainty_epistemic):.6f}, "
            f"aleatoric={np.mean(uncertainty_aleatoric):.6f}"
        )
        logger.info(
            f"Uncertainty transformation - "
            f"Original: total={np.mean(uncertainty_total_original):.6f}, "
            f"epistemic={np.mean(uncertainty_epistemic_original):.6f}, "
            f"aleatoric={np.mean(uncertainty_aleatoric_original):.6f}"
        )
        
        # Use original space uncertainty for metrics
        uncertainty_total = uncertainty_total_original
        uncertainty_epistemic = uncertainty_epistemic_original
        uncertainty_aleatoric = uncertainty_aleatoric_original
    else:
        logger.warning("No preprocessor provided - uncertainty remains in standardized space")
    
    # Compute base metrics
    metrics = evaluate_comprehensive(
        y_true=y_true,
        y_pred=y_pred,
        uncertainty=uncertainty_total,
        epistemic=uncertainty_epistemic,
        aleatoric=uncertainty_aleatoric,
    )
    
    logger.info(
        f"Metrics - RMSE: {metrics.rmse:.6f}, R-AUC MSE: {metrics.r_auc_mse:.6f}, "
        f"Mean uncertainty: {metrics.mean_uncertainty:.6f}"
    )
    
    # Compute error-retention curve (retention approach)
    mse_per_point = (y_true - y_pred) ** 2
    error_retention_x, error_retention_y = error_retention_curve(
        mse_per_point, uncertainty_total, num_thresholds=50
    )
    
    # Compute rejection curve (rejection approach from reference repo)
    from src.eval.reference_metrics import calc_uncertainty_rejection_curve
    rejection_curve = calc_uncertainty_rejection_curve(mse_per_point, uncertainty_total)
    
    # Conformal calibration
    conformal_results = {}
    pi_metrics_before = {}
    pi_metrics_after = {}
    
    if X_cal is not None and y_cal is not None:
        logger.info("Performing conformal calibration")
        
        # Get predictions on calibration set
        y_pred_cal, _, _, _ = ensemble_predict(models, X_cal, device=device)
        
        # Calibrate for multiple coverage levels
        conformal_results = calibrate_multiple_levels(
            y_true_cal=y_cal,
            y_pred_cal=y_pred_cal,
            coverage_levels=coverage_levels,
        )
        
        # Compute PI metrics before calibration (using raw uncertainty)
        for coverage_level in coverage_levels:
            # Before calibration: use uncertainty-based intervals
            z_score = {0.80: 1.28, 0.90: 1.645, 0.95: 1.96}.get(coverage_level, 1.645)
            lower_before = y_pred - z_score * uncertainty_total
            upper_before = y_pred + z_score * uncertainty_total
            pi_metrics_before[coverage_level] = compute_pi_metrics(y_true, lower_before, upper_before)
            
            # After calibration: use conformal intervals
            conf_result = conformal_results[coverage_level]
            # Apply conformal quantile to test set
            from src.eval.conformal import apply_intervals
            lower_after, upper_after = apply_intervals(y_pred, conf_result.quantile)
            pi_metrics_after[coverage_level] = compute_pi_metrics(y_true, lower_after, upper_after)
            
            logger.info(
                f"Coverage {coverage_level:.0%} - Before: {pi_metrics_before[coverage_level]['coverage']:.4%}, "
                f"After: {pi_metrics_after[coverage_level]['coverage']:.4%}"
            )
    else:
        logger.warning("No calibration set provided, skipping conformal calibration")
    
    return EvaluationResults(
        metrics=metrics,
        conformal_results=conformal_results,
        pi_metrics_before=pi_metrics_before,
        pi_metrics_after=pi_metrics_after,
        error_retention_x=error_retention_x,
        error_retention_y=error_retention_y,
        rejection_curve=rejection_curve,
    )

