from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.eval.evaluator import EvaluationResults
from src.utils.logger import get_logger
from src.eval.conformal import ConformalResults

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
    # Best point (min error)
    if len(retention_y) > 0:
        best_idx = int(np.argmin(retention_y))
        ax1.axvline(retention_x[best_idx], color="blue", linestyle="--", alpha=0.6, label="Best retention")
        ax1.scatter(retention_x[best_idx], retention_y[best_idx], color="blue", s=60, zorder=5)
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


def plot_calibration_before_after(
    pi_metrics_before: Dict[float, Dict[str, float]],
    pi_metrics_after: Dict[float, Dict[str, float]],
    save_path: Path | str,
    title: str = "Calibration Before/After Conformal",
) -> None:
    """Bar plot coverage/width before vs after conformal calibration."""
    logger = get_logger("eval.visualization")
    levels = sorted(pi_metrics_after.keys())
    if not levels:
        return
    cov_before = [pi_metrics_before[l]["coverage"] * 100 if l in pi_metrics_before else np.nan for l in levels]
    cov_after = [pi_metrics_after[l]["coverage"] * 100 if l in pi_metrics_after else np.nan for l in levels]
    width_before = [pi_metrics_before[l].get("mean_width", np.nan) if l in pi_metrics_before else np.nan for l in levels]
    width_after = [pi_metrics_after[l].get("mean_width", np.nan) if l in pi_metrics_after else np.nan for l in levels]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(levels))
    barw = 0.35
    
    axes[0].bar(x - barw/2, cov_before, width=barw, label="Before")
    axes[0].bar(x + barw/2, cov_after, width=barw, label="After")
    axes[0].axhline(80, color="r", linestyle="--", alpha=0.3)
    axes[0].axhline(90, color="r", linestyle="--", alpha=0.3)
    axes[0].axhline(95, color="r", linestyle="--", alpha=0.3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{int(l*100)}%" for l in levels])
    axes[0].set_ylabel("Coverage, %")
    axes[0].set_title("Coverage Before/After")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()
    
    axes[1].bar(x - barw/2, width_before, width=barw, label="Before")
    axes[1].bar(x + barw/2, width_after, width=barw, label="After")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{int(l*100)}%" for l in levels])
    axes[1].set_ylabel("Mean PI width")
    axes[1].set_title("PI Width Before/After")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()
    
    plt.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved calibration before/after plot to {save_path}")
    plt.close(fig)


def plot_residual_diagnostics(
    residuals: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray | None,
    save_dir: Path | str,
    prefix: str = "",
) -> None:
    """Hist + QQ + residual vs prediction."""
    logger = get_logger("eval.visualization")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Residual histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(residuals, bins=40, alpha=0.7, color="steelblue", edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="0")
    ax.set_title(f"{prefix} Residual Histogram", fontweight="bold")
    ax.set_xlabel("Residual (y_true - y_pred)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}residual_hist.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # QQ plot (normal)
    try:
        from scipy import stats  # type: ignore
        fig, ax = plt.subplots(figsize=(6, 6))
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f"{prefix} QQ-Plot (residuals)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_dir / f"{prefix}residual_qq.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Skipping QQ plot (scipy missing?): {e}")
    
    # Residual vs prediction
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, alpha=0.35, s=12)
    ax.axhline(0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Residual")
    ax.set_title(f"{prefix} Residuals vs Prediction", fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}residual_vs_pred.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    if uncertainty is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(uncertainty, residuals, alpha=0.35, s=12, color="darkorange")
        ax.axhline(0, color="red", linestyle="--", linewidth=2)
        ax.set_xlabel("Uncertainty")
        ax.set_ylabel("Residual")
        ax.set_title(f"{prefix} Residuals vs Uncertainty", fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / f"{prefix}residual_vs_uncertainty.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_uncertainty_vs_error(
    errors: np.ndarray,
    uncertainty: np.ndarray,
    save_dir: Path | str,
    prefix: str = "",
) -> None:
    """Scatter/box diagnostics of |error| vs uncertainty."""
    logger = get_logger("eval.visualization")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    abs_err = np.abs(errors)
    corr = np.corrcoef(abs_err, uncertainty)[0, 1] if len(abs_err) > 1 else np.nan
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(abs_err, uncertainty, alpha=0.3, s=10)
    ax.set_xlabel("|error|")
    ax.set_ylabel("uncertainty")
    ax.set_title(f"{prefix} |error| vs uncertainty (corr={corr:.3f})", fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}uncertainty_vs_error.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Boxplot by error quantiles
    quantiles = np.quantile(abs_err, [0, 0.25, 0.5, 0.75, 1.0])
    bins = np.digitize(abs_err, quantiles[1:-1], right=True)
    grouped = [uncertainty[bins == i] for i in range(len(quantiles)-1)]
    labels = [f"Q{i+1}" for i in range(len(quantiles)-1)]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(grouped, labels=labels, showfliers=False)
    ax.set_xlabel("Error quantile bin")
    ax.set_ylabel("Uncertainty")
    ax.set_title(f"{prefix} Uncertainty by error quantile", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}uncertainty_by_error_quantile.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pi_width_distribution(
    pi_intervals_after: Dict[float, Tuple[np.ndarray, np.ndarray]],
    save_dir: Path | str,
    prefix: str = "",
) -> None:
    """Distribution of PI widths for each coverage level."""
    logger = get_logger("eval.visualization")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    levels = sorted(pi_intervals_after.keys())
    if not levels:
        return
    
    for level in levels:
        lower, upper = pi_intervals_after[level]
        widths = upper - lower
        # Determine number of bins dynamically based on data
        if len(widths) == 0:
            continue
        
        # Check if all values are the same (or very close)
        unique_values = np.unique(widths)
        if len(unique_values) == 1:
            # All widths are the same, use a single bin
            n_bins = 1
        else:
            # Calculate range to ensure we can create bins
            width_range = np.max(widths) - np.min(widths)
            if width_range == 0 or np.isclose(width_range, 0):
                n_bins = 1
            else:
                n_bins = min(40, max(1, len(unique_values)))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(widths, bins=n_bins, alpha=0.7, color="seagreen", edgecolor="black")
        ax.set_title(f"{prefix} PI width dist @ {int(level*100)}%", fontweight="bold")
        ax.set_xlabel("Width")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / f"{prefix}pi_width_{int(level*100)}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_training_history(
    history: List[Dict[str, float]],
    save_path: Path | str,
    best_epoch: int | None = None,
    early_stop_epoch: int | None = None,
    title: str = "Training Curves",
) -> None:
    """Plot train/val curves from history."""
    if not history:
        return
    logger = get_logger("eval.visualization")
    epochs = [h["epoch"] for h in history]
    train_loss = [h.get("train_loss", np.nan) for h in history]
    val_score = [h.get("val_r_auc_mse", np.nan) for h in history]
    val_rmse = [h.get("val_rmse", np.nan) for h in history]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(epochs, train_loss, label="Train loss", color="blue")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train loss", fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, val_score, label="Val R-AUC MSE (lower better)", color="orange")
    axes[1].plot(epochs, val_rmse, label="Val RMSE", color="green", alpha=0.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_title("Validation metrics", fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    for ax in axes:
        if best_epoch is not None:
            ax.axvline(best_epoch, color="red", linestyle="--", alpha=0.6, label="Best")
        if early_stop_epoch is not None:
            ax.axvline(early_stop_epoch, color="purple", linestyle=":", alpha=0.6, label="Early stop")
    
    plt.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved training history to {save_path}")
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
        
        # Before/after coverage & width comparison
        plot_calibration_before_after(
            results.pi_metrics_before,
            results.pi_metrics_after,
            save_path=output_dir / f"{prefix}calibration_before_after.png",
            title=f"{prefix}Calibration Before/After",
        )
        
        # PI width distribution
        plot_pi_width_distribution(
            results.pi_intervals_after,
            save_dir=output_dir,
            prefix=prefix,
        )
    
    # Residual diagnostics & uncertainty-error relations
    if results.residuals is not None and results.y_pred is not None:
        plot_residual_diagnostics(
            residuals=results.residuals,
            y_pred=results.y_pred,
            uncertainty=results.uncertainty_total,
            save_dir=output_dir,
            prefix=prefix,
        )
    if results.residuals is not None and results.uncertainty_total is not None:
        plot_uncertainty_vs_error(
            errors=results.residuals,
            uncertainty=results.uncertainty_total,
            save_dir=output_dir,
            prefix=prefix,
        )
    
    logger.info(f"Visualizations saved to {output_dir}")

