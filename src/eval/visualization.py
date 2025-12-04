from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.eval.evaluator import EvaluationResults
from src.utils.logger import get_logger

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def plot_error_retention_curve(
    x: np.ndarray,
    y: np.ndarray,
    save_path: Path | str | None = None,
    title: str = "Error-Retention Curve",
) -> None:
    """Plot error-retention curve.
    
    Args:
        x: Retention fractions (0 to 1)
        y: Mean MSE at each retention level
        save_path: Path to save figure (optional)
        title: Plot title
    """
    logger = get_logger("eval.visualization")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, linewidth=2, label="Error-Retention")
    ax.fill_between(x, 0, y, alpha=0.3)
    ax.set_xlabel("Retention Fraction", fontsize=12)
    ax.set_ylabel("Mean MSE", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved error-retention curve to {save_path}")
    
    plt.close(fig)


def plot_calibration_curve(
    coverage_levels: list[float],
    actual_coverages: list[float],
    save_path: Path | str | None = None,
    title: str = "Calibration Curve",
) -> None:
    """Plot calibration curve (actual vs nominal coverage).
    
    Args:
        coverage_levels: Nominal coverage levels (e.g., [0.80, 0.90, 0.95])
        actual_coverages: Actual coverage achieved
        save_path: Path to save figure (optional)
        title: Plot title
    """
    logger = get_logger("eval.visualization")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfect Calibration", alpha=0.5)
    
    # Actual calibration
    ax.scatter(coverage_levels, actual_coverages, s=100, zorder=5, label="Actual Coverage")
    ax.plot(coverage_levels, actual_coverages, linewidth=2, marker="o", markersize=8)
    
    ax.set_xlabel("Nominal Coverage", fontsize=12)
    ax.set_ylabel("Actual Coverage", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim([0.7, 1.0])
    ax.set_ylim([0.7, 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved calibration curve to {save_path}")
    
    plt.close(fig)


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    coverage_level: float = 0.90,
    n_bins: int = 10,
    save_path: Path | str | None = None,
    title: str = "Reliability Diagram",
) -> None:
    """Plot reliability diagram for prediction intervals.
    
    Args:
        y_true: True values
        y_pred: Predictions
        uncertainty: Uncertainty estimates
        coverage_level: Target coverage level
        n_bins: Number of bins for grouping
        save_path: Path to save figure (optional)
        title: Plot title
    """
    logger = get_logger("eval.visualization")
    
    # Compute prediction intervals
    z_score = {0.80: 1.28, 0.90: 1.645, 0.95: 1.96}.get(coverage_level, 1.645)
    lower = y_pred - z_score * uncertainty
    upper = y_pred + z_score * uncertainty
    
    # Bin by uncertainty
    uncertainty_sorted_idx = np.argsort(uncertainty)
    bin_size = len(uncertainty) // n_bins
    
    bin_coverages = []
    bin_uncertainties = []
    
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(uncertainty)
        bin_idx = uncertainty_sorted_idx[start_idx:end_idx]
        
        bin_y_true = y_true[bin_idx]
        bin_lower = lower[bin_idx]
        bin_upper = upper[bin_idx]
        
        bin_coverage = np.mean((bin_y_true >= bin_lower) & (bin_y_true <= bin_upper))
        bin_uncertainty = np.mean(uncertainty[bin_idx])
        
        bin_coverages.append(bin_coverage)
        bin_uncertainties.append(bin_uncertainty)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(n_bins), bin_coverages, alpha=0.7, label="Actual Coverage")
    ax.axhline(y=coverage_level, color="r", linestyle="--", linewidth=2, label=f"Target ({coverage_level:.0%})")
    ax.set_xlabel("Uncertainty Bin (low to high)", fontsize=12)
    ax.set_ylabel("Coverage", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels([f"Bin {i+1}" for i in range(n_bins)])
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved reliability diagram to {save_path}")
    
    plt.close(fig)


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path | str | None = None,
    title: str = "Predictions vs True Values",
) -> None:
    """Plot scatter plot of predictions vs true values.
    
    Args:
        y_true: True values
        y_pred: Predictions
        save_path: Path to save figure (optional)
        title: Plot title
    """
    logger = get_logger("eval.visualization")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
    
    ax.set_xlabel("True Values", fontsize=12)
    ax.set_ylabel("Predictions", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved prediction scatter to {save_path}")
    
    plt.close(fig)


def plot_rejection_curve(
    rejection_curve: np.ndarray,
    save_path: Path | str | None = None,
    title: str = "Uncertainty Rejection Curve",
) -> None:
    """Plot rejection curve (from reference repository).
    
    Args:
        rejection_curve: Rejection curve values (n_samples + 1)
        save_path: Path to save figure (optional)
        title: Plot title
    """
    logger = get_logger("eval.visualization")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # X-axis: fraction of rejected objects (0 to 1)
    n_samples = len(rejection_curve) - 1
    rejection_fractions = np.linspace(0, 1, len(rejection_curve))
    
    ax.plot(rejection_fractions, rejection_curve, linewidth=2, label="Rejection Curve")
    ax.fill_between(rejection_fractions, 0, rejection_curve, alpha=0.3)
    ax.set_xlabel("Rejection Fraction", fontsize=12)
    ax.set_ylabel("Error Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved rejection curve to {save_path}")
    
    plt.close(fig)


def plot_retention_vs_rejection_comparison(
    retention_x: np.ndarray,
    retention_y: np.ndarray,
    rejection_curve: np.ndarray,
    save_path: Path | str | None = None,
    title: str = "Retention vs Rejection Curves Comparison",
) -> None:
    """Plot comparison of retention and rejection curves.
    
    Args:
        retention_x: Retention fractions (0 to 1)
        retention_y: Mean MSE at each retention level
        rejection_curve: Rejection curve values
        save_path: Path to save figure (optional)
        title: Plot title
    """
    logger = get_logger("eval.visualization")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Retention curve
    ax1.plot(retention_x, retention_y, linewidth=2, label="Retention Curve", color="blue")
    ax1.fill_between(retention_x, 0, retention_y, alpha=0.3, color="blue")
    ax1.set_xlabel("Retention Fraction", fontsize=12)
    ax1.set_ylabel("Mean MSE", fontsize=12)
    ax1.set_title("Retention Curve", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Right plot: Rejection curve
    rejection_fractions = np.linspace(0, 1, len(rejection_curve))
    ax2.plot(rejection_fractions, rejection_curve, linewidth=2, label="Rejection Curve", color="red")
    ax2.fill_between(rejection_fractions, 0, rejection_curve, alpha=0.3, color="red")
    ax2.set_xlabel("Rejection Fraction", fontsize=12)
    ax2.set_ylabel("Error Rate", fontsize=12)
    ax2.set_title("Rejection Curve", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved comparison plot to {save_path}")
    
    plt.close(fig)


def visualize_evaluation_results(
    results: EvaluationResults,
    output_dir: Path | str,
    prefix: str = "",
) -> None:
    """Create all visualizations for evaluation results.
    
    Args:
        results: EvaluationResults object
        output_dir: Directory to save figures
        prefix: Prefix for filenames
    """
    logger = get_logger("eval.visualization")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating visualizations in {output_dir}")
    
    # Error-retention curve (retention approach)
    plot_error_retention_curve(
        results.error_retention_x,
        results.error_retention_y,
        save_path=output_dir / f"{prefix}error_retention.png",
        title=f"{prefix}Error-Retention Curve",
    )
    
    # Rejection curve (rejection approach from reference repo)
    if results.rejection_curve is not None:
        plot_rejection_curve(
            results.rejection_curve,
            save_path=output_dir / f"{prefix}rejection_curve.png",
            title=f"{prefix}Uncertainty Rejection Curve",
        )
        
        # Comparison plot
        plot_retention_vs_rejection_comparison(
            results.error_retention_x,
            results.error_retention_y,
            results.rejection_curve,
            save_path=output_dir / f"{prefix}retention_vs_rejection.png",
            title=f"{prefix}Retention vs Rejection Curves",
        )
    
    # Calibration curve
    if results.conformal_results:
        coverage_levels = sorted(results.conformal_results.keys())
        actual_coverages = [results.conformal_results[cl].coverage for cl in coverage_levels]
        plot_calibration_curve(
            coverage_levels,
            actual_coverages,
            save_path=output_dir / f"{prefix}calibration.png",
            title=f"{prefix}Calibration Curve",
        )
    
    logger.info(f"Visualizations saved to {output_dir}")

