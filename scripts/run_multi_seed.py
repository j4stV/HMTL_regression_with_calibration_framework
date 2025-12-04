from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.main import run_experiment, load_yaml
from src.eval.metrics import EvaluationMetrics, aggregate_metrics_across_seeds
from src.utils.logger import setup_logging, get_logger, log_metrics, log_timing

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def run_multi_seed_experiment(
    data_config: str = "configs/data.yaml",
    model_config: str = "configs/model_snn.yaml",
    train_config: str = "configs/train.yaml",
    ensemble_config: str = "configs/ensemble.yaml",
    n_seeds: int = 3,
    seeds: list[int] | None = None,
    output_dir: str | Path = "experiments/multi_seed",
) -> dict:
    """Run experiment with multiple seeds and aggregate results.
    
    Args:
        data_config: Path to data config
        model_config: Path to model config
        train_config: Path to train config
        ensemble_config: Path to ensemble config
        n_seeds: Number of seeds to run (if seeds not provided)
        seeds: List of specific seeds to use
        output_dir: Directory to save results
    
    Returns:
        Dictionary with aggregated metrics and individual results
    """
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info(f"Multi-Seed Experiment: {n_seeds if seeds is None else len(seeds)} seeds")
    logger.info("=" * 80)
    
    if seeds is None:
        seeds = list(range(42, 42 + n_seeds))
    
    logger.info(f"Seeds: {seeds}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base configs
    train_cfg = load_yaml(Path(train_config))
    
    # Run experiments for each seed
    all_metrics_list: list[EvaluationMetrics] = []
    all_results = []
    
    for i, seed in enumerate(seeds):
        logger.info("=" * 80)
        logger.info(f"Running experiment {i+1}/{len(seeds)} with seed {seed}")
        logger.info("=" * 80)
        
        # Update seed in config
        train_cfg["training"]["seed"] = seed
        
        # Save temporary config
        temp_config_path = output_dir / f"train_temp_seed_{seed}.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(train_cfg, f)
        
        try:
            # Run experiment
            with log_timing(f"Experiment with seed {seed}", logger):
                result = run_experiment(
                    data_config=data_config,
                    model_config=model_config,
                    train_config=str(temp_config_path),
                    ensemble_config=ensemble_config,
                    return_models=False,
                )
            
            # Extract metrics
            metrics = result["val_results"].metrics
            all_metrics_list.append(metrics)
            all_results.append({
                "seed": seed,
                "metrics": {
                    "rmse": metrics.rmse,
                    "mse": metrics.mse,
                    "mae": metrics.mae,
                    "r_auc_mse": metrics.r_auc_mse,
                    "mean_uncertainty": metrics.mean_uncertainty,
                    "mean_epistemic": metrics.mean_epistemic,
                    "mean_aleatoric": metrics.mean_aleatoric,
                },
                "coverage": {
                    level: result["val_results"].pi_metrics_after[level]["coverage"]
                    for level in [0.80, 0.90, 0.95]
                    if level in result["val_results"].pi_metrics_after
                },
            })
            
            logger.info(f"Seed {seed} completed - R-AUC MSE: {metrics.r_auc_mse:.6f}")
            
        except Exception as e:
            logger.error(f"Failed to run experiment with seed {seed}: {e}")
            logger.exception(e)
        finally:
            # Clean up temp config
            if temp_config_path.exists():
                temp_config_path.unlink()
    
    if not all_metrics_list:
        logger.error("No successful experiments!")
        return {}
    
    # Aggregate metrics
    logger.info("=" * 80)
    logger.info("Aggregating Results Across Seeds")
    logger.info("=" * 80)
    
    aggregated = aggregate_metrics_across_seeds(all_metrics_list)
    
    # Log aggregated metrics
    logger.info("Aggregated Metrics (mean ± std):")
    for metric_name, (mean_val, std_val) in aggregated.items():
        logger.info(f"  {metric_name}: {mean_val:.6f} ± {std_val:.6f}")
    
    # Save results to JSON
    results_summary = {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "aggregated_metrics": {k: {"mean": float(v[0]), "std": float(v[1])} for k, v in aggregated.items()},
        "individual_results": all_results,
    }
    
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results_summary, f, indent=2)
    logger.info(f"Results saved to {results_file}")
    
    # Create visualization of metric distributions
    logger.info("Creating visualizations...")
    create_metric_distribution_plots(all_results, output_dir)
    
    logger.info("=" * 80)
    logger.info("Multi-Seed Experiment Complete")
    logger.info("=" * 80)
    
    return results_summary


def create_metric_distribution_plots(results: list[dict], output_dir: Path) -> None:
    """Create plots showing distribution of metrics across seeds.
    
    Args:
        results: List of result dictionaries from individual seeds
        output_dir: Directory to save plots
    """
    logger = get_logger("multi_seed")
    
    # Extract metrics
    metric_names = ["rmse", "mse", "mae", "r_auc_mse", "mean_uncertainty"]
    metric_data = {name: [r["metrics"][name] for r in results] for name in metric_names}
    
    # Create box plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]
        values = metric_data[metric_name]
        ax.boxplot(values, labels=[metric_name])
        ax.set_ylabel("Value")
        ax.set_title(f"{metric_name.upper()} Distribution")
        ax.grid(True, alpha=0.3)
        
        # Add mean and std text
        mean_val = np.mean(values)
        std_val = np.std(values)
        ax.text(0.5, 0.95, f"Mean: {mean_val:.6f}\nStd: {std_val:.6f}",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Remove extra subplot
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plot_path = output_dir / "metric_distributions.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved metric distributions plot to {plot_path}")
    
    # Create coverage comparison plot
    coverage_levels = [0.80, 0.90, 0.95]
    coverage_data = {level: [r["coverage"].get(level, 0.0) for r in results] for level in coverage_levels}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = [1, 2, 3]
    bp = ax.boxplot([coverage_data[level] for level in coverage_levels],
                    labels=[f"{int(level*100)}%" for level in coverage_levels],
                    positions=positions)
    
    # Add target coverage lines
    for idx, level in enumerate(coverage_levels):
        ax.axhline(y=level, xmin=(positions[idx]-0.4)/3.5, xmax=(positions[idx]+0.4)/3.5,
                   color='r', linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_ylabel("Coverage")
    ax.set_xlabel("Target Coverage Level")
    ax.set_title("Coverage Distribution Across Seeds")
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = output_dir / "coverage_distributions.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved coverage distributions plot to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-seed experiments")
    parser.add_argument("--data", default="configs/data.yaml", help="Path to data config")
    parser.add_argument("--model", default="configs/model_snn.yaml", help="Path to model config")
    parser.add_argument("--train", default="configs/train.yaml", help="Path to train config")
    parser.add_argument("--ensemble", default="configs/ensemble.yaml", help="Path to ensemble config")
    parser.add_argument("--n_seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--seeds", type=int, nargs="+", help="Specific seeds to use")
    parser.add_argument("--output", default="experiments/multi_seed", help="Output directory for results")
    args = parser.parse_args()
    
    run_multi_seed_experiment(
        data_config=args.data,
        model_config=args.model,
        train_config=args.train,
        ensemble_config=args.ensemble,
        n_seeds=args.n_seeds,
        seeds=args.seeds,
        output_dir=args.output,
    )
