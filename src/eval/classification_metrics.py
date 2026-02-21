"""Classification metrics including calibration and uncertainty metrics.

This module provides comprehensive evaluation metrics for classification tasks:
- Standard metrics: accuracy, F1, AUROC
- Calibration metrics: ECE (Expected Calibration Error), Brier score
- Uncertainty metrics: correlation between uncertainty and prediction errors
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from src.utils.logger import get_logger


def expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE measures the difference between predicted confidence and actual accuracy.
    Lower is better (0 = perfectly calibrated).

    Args:
        y_true: True class labels (n_samples,)
        probs: Predicted probabilities (n_samples, n_classes)
        n_bins: Number of bins for calibration

    Returns:
        ECE value (0 to 1)
    """
    y_true = y_true.astype(int)
    n_samples = len(y_true)

    # Get predicted class and confidence
    y_pred = np.argmax(probs, axis=-1)
    confidences = np.max(probs, axis=-1)

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            # Accuracy in this bin
            accuracy_in_bin = np.mean(y_pred[in_bin] == y_true[in_bin])
            # Average confidence in this bin
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            # Add weighted difference to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def brier_score_multi(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Compute multi-class Brier score.

    Brier score measures the mean squared error between predicted probabilities
    and one-hot encoded true labels. Lower is better (0 = perfect).

    Args:
        y_true: True class labels (n_samples,)
        probs: Predicted probabilities (n_samples, n_classes)

    Returns:
        Brier score (0 to 2 for binary, 0 to 2 for multi-class)
    """
    y_true = y_true.astype(int)
    n_samples, n_classes = probs.shape

    # One-hot encode true labels
    y_true_onehot = np.zeros((n_samples, n_classes))
    y_true_onehot[np.arange(n_samples), y_true] = 1

    # Brier score: mean squared difference
    brier = np.mean(np.sum((probs - y_true_onehot) ** 2, axis=-1))

    return float(brier)


def uncertainty_error_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
) -> dict[str, float]:
    """Compute correlation between uncertainty and prediction errors.

    Good uncertainty estimates should correlate with errors: high uncertainty
    should correspond to incorrect predictions.

    Args:
        y_true: True class labels (n_samples,)
        y_pred: Predicted class labels (n_samples,)
        uncertainty: Uncertainty estimates (n_samples,)

    Returns:
        Dictionary with:
        - pearson_r: Pearson correlation coefficient
        - pearson_p: Pearson p-value
        - spearman_rho: Spearman correlation coefficient
        - spearman_p: Spearman p-value
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    # Binary error indicator (1 = incorrect, 0 = correct)
    errors = (y_pred != y_true).astype(float)

    # Compute correlations
    try:
        pearson_r, pearson_p = pearsonr(uncertainty, errors)
    except Exception:
        pearson_r, pearson_p = np.nan, np.nan

    try:
        spearman_rho, spearman_p = spearmanr(uncertainty, errors)
    except Exception:
        spearman_rho, spearman_p = np.nan, np.nan

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    logits: np.ndarray,
    probs: np.ndarray,
    uncertainty: np.ndarray | None = None,
    num_classes: int | None = None,
) -> dict[str, float]:
    """Compute comprehensive classification metrics.

    Args:
        y_true: True class labels (n_samples,)
        logits: Predicted logits (n_samples, n_classes)
        probs: Predicted probabilities (n_samples, n_classes)
        uncertainty: Optional uncertainty estimates (n_samples,)
        num_classes: Number of classes (optional, inferred if not provided)

    Returns:
        Dictionary with all classification metrics:
        - accuracy, balanced_accuracy
        - f1_macro, f1_weighted
        - auroc (binary classification only)
        - ece (Expected Calibration Error)
        - brier (Brier score)
        - uncertainty_error_pearson_r, uncertainty_error_spearman_rho (if uncertainty provided)
    """
    logger = get_logger("eval.classification_metrics")

    y_true = y_true.astype(int)
    y_pred = np.argmax(probs, axis=-1)

    if num_classes is None:
        num_classes = probs.shape[-1]

    metrics = {}

    # Standard classification metrics
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

    # F1 scores
    try:
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    except Exception as e:
        logger.warning(f"Failed to compute F1 scores: {e}")
        metrics["f1_macro"] = 0.0
        metrics["f1_weighted"] = 0.0

    # AUROC (binary classification or OvR for multi-class)
    try:
        if num_classes == 2:
            # Binary classification: use probabilities of positive class
            metrics["auroc"] = float(roc_auc_score(y_true, probs[:, 1]))
        else:
            # Multi-class: use one-vs-rest with weighted average
            metrics["auroc"] = float(
                roc_auc_score(y_true, probs, multi_class="ovr", average="weighted")
            )
    except Exception as e:
        logger.warning(f"Failed to compute AUROC: {e}")
        metrics["auroc"] = np.nan

    # Calibration metrics
    try:
        metrics["ece"] = expected_calibration_error(y_true, probs)
    except Exception as e:
        logger.warning(f"Failed to compute ECE: {e}")
        metrics["ece"] = np.nan

    try:
        metrics["brier"] = brier_score_multi(y_true, probs)
    except Exception as e:
        logger.warning(f"Failed to compute Brier score: {e}")
        metrics["brier"] = np.nan

    # Uncertainty-error correlation (if uncertainty provided)
    if uncertainty is not None:
        try:
            corr_metrics = uncertainty_error_correlation(y_true, y_pred, uncertainty)
            metrics["uncertainty_error_pearson_r"] = corr_metrics["pearson_r"]
            metrics["uncertainty_error_pearson_p"] = corr_metrics["pearson_p"]
            metrics["uncertainty_error_spearman_rho"] = corr_metrics["spearman_rho"]
            metrics["uncertainty_error_spearman_p"] = corr_metrics["spearman_p"]
        except Exception as e:
            logger.warning(f"Failed to compute uncertainty-error correlation: {e}")
            metrics["uncertainty_error_pearson_r"] = np.nan
            metrics["uncertainty_error_spearman_rho"] = np.nan

    logger.debug(
        f"Classification metrics - "
        f"Accuracy: {metrics['accuracy']:.4f}, "
        f"F1 (macro): {metrics['f1_macro']:.4f}, "
        f"ECE: {metrics['ece']:.4f}, "
        f"Brier: {metrics['brier']:.4f}"
    )

    return metrics
