"""Script to run all experiments and generate comprehensive report."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import subprocess
import shutil
import time
from datetime import datetime
from typing import Dict, Any

import pandas as pd
import numpy as np

from scripts.main import run_experiment, load_yaml
from src.utils.logger import setup_logging, get_logger, log_timing


def run_command(cmd: list[str], description: str, logger) -> bool:
    """Run a command and return success status."""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        logger.info(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed")
        logger.error(f"Error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"✗ {description} failed with exception: {e}")
        return False


def collect_results(output_dir: Path, logger) -> Dict[str, Any]:
    """Collect all experiment results."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "main_experiment": None,
        "multi_seed": None,
        "baselines": None,
        "ablation": None,
    }
    
    # Collect main experiment results
    main_results_dir = output_dir / "main"
    if main_results_dir.exists():
        logger.info("Collecting main experiment results...")
        main_info: Dict[str, Any] = {}
        
        # Metrics
        main_metrics_file = main_results_dir / "results.json"
        if main_metrics_file.exists():
            try:
                with open(main_metrics_file, "r", encoding="utf-8") as f:
                    metrics_payload = json.load(f)
                main_info["metrics"] = metrics_payload.get("metrics", metrics_payload)
                logger.info(f"Loaded main experiment metrics from {main_metrics_file}")
            except Exception as e:
                logger.warning(f"Failed to read main experiment metrics: {e}")
        
        # Plots
        plots_dir = main_results_dir / "plots"
        if plots_dir.exists():
            main_info["plots_dir"] = str(plots_dir)
            main_info["plot_files"] = sorted([str(p) for p in plots_dir.glob("*.png")])
        
        results["main_experiment"] = main_info if main_info else None
    
    # Collect multi-seed results
    multi_seed_results_file = output_dir / "multi_seed" / "results.json"
    if multi_seed_results_file.exists():
        logger.info("Collecting multi-seed results...")
        with open(multi_seed_results_file) as f:
            results["multi_seed"] = json.load(f)
        multi_seed_plots = list((output_dir / "multi_seed").glob("*.png"))
        if multi_seed_plots:
            results["multi_seed"]["plot_files"] = [str(p) for p in multi_seed_plots]
    
    # Collect baseline comparison results
    baseline_locations = [
        output_dir / "baselines" / "comparison_table.csv",
        Path("experiments/baselines/comparison_table.csv"),
    ]
    
    baseline_results_file = None
    for loc in baseline_locations:
        if loc.exists():
            baseline_results_file = loc
            break
    
    if baseline_results_file:
        logger.info(f"Collecting baseline comparison results from {baseline_results_file}...")
        try:
            df = pd.read_csv(baseline_results_file)
            results["baselines"] = {
                "comparison_table": df.to_dict("records"),
                "best_models": {},
            }
            baseline_plot = baseline_results_file.parent / "baseline_comparison.png"
            if baseline_plot.exists():
                results["baselines"]["plot"] = str(baseline_plot)
            # Find best models
            for metric in ["RMSE", "R-AUC MSE", "MAE"]:
                if metric in df.columns:
                    best_idx = df[metric].idxmin()
                    results["baselines"]["best_models"][metric] = {
                        "model": df.loc[best_idx, "Model"],
                        "value": float(df.loc[best_idx, metric]),
                    }
        except Exception as e:
            logger.warning(f"Failed to read baseline results: {e}")
    else:
        logger.warning("Baseline comparison results not found in expected locations")
    
    return results


def generate_markdown_report(results: Dict[str, Any], output_file: Path, logger) -> None:
    """Generate markdown report from collected results."""
    logger.info(f"Generating markdown report: {output_file}")
    
    lines = []
    lines.append("# HMTL with Calibration - Experimental Report")
    lines.append("")
    lines.append(f"**Generated:** {results['timestamp']}")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("This report summarizes the results of comprehensive experiments on Hierarchical Multi-Task Learning (HMTL) with uncertainty estimation and conformal calibration for tabular regression.")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Main experiment section
    lines.append("## 1. Main HMTL Experiment")
    lines.append("")
    main = results.get("main_experiment")
    if main:
        metrics = main.get("metrics", {})
        def fmt_val(val: Any, pct: bool = False) -> str:
            if val is None:
                return "—"
            try:
                val = float(val)
                return f"{val*100:.2f}%" if pct else f"{val:.6f}"
            except Exception:
                return str(val)
        
        # Core metrics table
        lines.append("### Метрики (валидация / тест)")
        lines.append("")
        lines.append("| Metric | Validation | Test |")
        lines.append("|--------|------------|------|")
        metric_rows = [
            ("RMSE", "val_rmse", "test_rmse"),
            ("MAE", "val_mae", "test_mae"),
            ("R-AUC MSE", "val_r_auc_mse", "test_r_auc_mse"),
            ("Mean Uncertainty", "val_mean_uncertainty", "test_mean_uncertainty"),
            ("Mean Epistemic", "val_mean_epistemic", None),
            ("Mean Aleatoric", "val_mean_aleatoric", None),
        ]
        for label, val_key, test_key in metric_rows:
            lines.append(
                f"| {label} | {fmt_val(metrics.get(val_key))} | {fmt_val(metrics.get(test_key))} |"
            )
        lines.append("")
        
        # Coverage & interval width
        lines.append("### Покрытие после конформной калибровки")
        lines.append("")
        lines.append("| Level | Val Coverage | Val Width | Test Coverage | Test Width |")
        lines.append("|-------|--------------|-----------|---------------|------------|")
        for level in [80, 90, 95]:
            cov_val = metrics.get(f"val_coverage@{level}")
            width_val = metrics.get(f"val_width@{level}")
            cov_test = metrics.get(f"test_coverage@{level}")
            width_test = metrics.get(f"test_width@{level}")
            lines.append(
                f"| {level}% | {fmt_val(cov_val, pct=True)} | {fmt_val(width_val)} | "
                f"{fmt_val(cov_test, pct=True)} | {fmt_val(width_test)} |"
            )
        lines.append("")
        
        # Uncertainty breakdown
        if any(metrics.get(k) is not None for k in ["val_mean_uncertainty", "val_mean_epistemic", "val_mean_aleatoric"]):
            lines.append("### Разложение неопределенности")
            lines.append("")
            lines.append("- Вал. неопределенность: "
                         f"{fmt_val(metrics.get('val_mean_uncertainty'))} "
                         f"(эпистемическая: {fmt_val(metrics.get('val_mean_epistemic'))}, "
                         f"алеаторная: {fmt_val(metrics.get('val_mean_aleatoric'))})")
            if metrics.get("test_mean_uncertainty") is not None:
                lines.append(f"- Тестовая неопределенность: {fmt_val(metrics.get('test_mean_uncertainty'))}")
            if metrics.get("ensemble_avg_r_auc_mse") is not None:
                lines.append(f"- Средний R-AUC MSE по ансамблю: {fmt_val(metrics.get('ensemble_avg_r_auc_mse'))}")
            if metrics.get("n_models") is not None:
                lines.append(f"- Размер ансамбля: {metrics.get('n_models')}")
            lines.append("")
        
        # Plots
        plot_dir = Path(main.get("plots_dir", "")) if main.get("plots_dir") else None
        if plot_dir and plot_dir.exists():
            lines.append("### Ключевые графики")
            lines.append("")
            plot_defs = [
                ("val_error_retention.png", "Error-Retention (валидация)"),
                ("val_rejection_curve.png", "Rejection Curve (валидация)"),
                ("val_retention_vs_rejection.png", "Retention vs Rejection (валидация)"),
                ("val_calibration.png", "Calibration Curve (валидация)"),
                ("val_calibration_before_after.png", "Calibration before/after conformal (валидация)"),
                ("val_residual_hist.png", "Residual histogram (валидация)"),
                ("val_residual_qq.png", "Residual QQ (валидация)"),
                ("val_residual_vs_pred.png", "Residual vs pred (валидация)"),
                ("val_residual_vs_uncertainty.png", "Residual vs uncertainty (валидация)"),
                ("val_uncertainty_vs_error.png", "|error| vs uncertainty (валидация)"),
                ("val_uncertainty_by_error_quantile.png", "Uncertainty by error quantile (валидация)"),
                ("val_pi_width_80.png", "PI width dist @80%"),
                ("val_pi_width_90.png", "PI width dist @90%"),
                ("val_pi_width_95.png", "PI width dist @95%"),
                ("test_error_retention.png", "Error-Retention (тест)"),
                ("test_rejection_curve.png", "Rejection Curve (тест)"),
                ("test_retention_vs_rejection.png", "Retention vs Rejection (тест)"),
                ("test_calibration.png", "Calibration Curve (тест)"),
                ("test_calibration_before_after.png", "Calibration before/after conformal (тест)"),
                ("test_residual_hist.png", "Residual histogram (тест)"),
                ("test_residual_qq.png", "Residual QQ (тест)"),
                ("test_residual_vs_pred.png", "Residual vs pred (тест)"),
                ("test_residual_vs_uncertainty.png", "Residual vs uncertainty (тест)"),
                ("test_uncertainty_vs_error.png", "|error| vs uncertainty (тест)"),
                ("test_uncertainty_by_error_quantile.png", "Uncertainty by error quantile (тест)"),
                ("test_pi_width_80.png", "PI width dist @80% (тест)"),
                ("test_pi_width_90.png", "PI width dist @90% (тест)"),
                ("test_pi_width_95.png", "PI width dist @95% (тест)"),
            ]
            for fname, title in plot_defs:
                candidate = plot_dir / fname
                if candidate.exists():
                    lines.append(f"![{title}]({candidate.as_posix()})")
                    lines.append("")
            
            # Training curves (ensemble models)
            training_dir = plot_dir / "training"
            if training_dir.exists():
                lines.append("### Кривые обучения ансамбля HMTL")
                lines.append("")
                for img in sorted(training_dir.glob("model_*_training_curve.png")):
                    lines.append(f"![Training {img.name}]({img.as_posix()})")
                    lines.append("")
        lines.append("")
    else:
        lines.append("Main experiment results not found.")
        lines.append("")
    
    # Multi-seed results section
    lines.append("## 2. Multi-Seed Experiments")
    lines.append("")
    if results.get("multi_seed"):
        multi_seed = results["multi_seed"]
        lines.append(f"**Number of seeds:** {multi_seed.get('n_seeds', 'N/A')}")
        lines.append(f"**Seeds used:** {', '.join(map(str, multi_seed.get('seeds', [])))}")
        lines.append("")
        lines.append("### Aggregated Metrics (Mean ± Std)")
        lines.append("")
        lines.append("| Metric | Mean | Std |")
        lines.append("|--------|------|-----|")
        
        if "aggregated_metrics" in multi_seed:
            for metric, stats in multi_seed["aggregated_metrics"].items():
                mean_val = stats.get("mean", 0)
                std_val = stats.get("std", 0)
                lines.append(f"| {metric} | {mean_val:.6f} | {std_val:.6f} |")
        
        lines.append("")

        # Coverage aggregation across seeds (if available)
        coverage_levels = ["0.8", "0.9", "0.95"]
        coverage_stats = {}
        for level in coverage_levels:
            vals = []
            for seed_result in multi_seed.get("individual_results", []):
                cov = seed_result.get("coverage", {}).get(level)
                if cov is not None:
                    vals.append(cov)
            if vals:
                coverage_stats[level] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        
        if coverage_stats:
            lines.append("### Покрытие (Mean ± Std)")
            lines.append("")
            lines.append("| Level | Mean | Std |")
            lines.append("|-------|------|-----|")
            for level in coverage_levels:
                if level in coverage_stats:
                    lines.append(
                        f"| {int(float(level)*100)}% | {coverage_stats[level]['mean']*100:.2f}% | "
                        f"{coverage_stats[level]['std']*100:.2f}% |"
                    )
            lines.append("")
        lines.append("### Individual Seed Results")
        lines.append("")
        if "individual_results" in multi_seed:
            for seed_result in multi_seed["individual_results"]:
                seed = seed_result.get("seed", "N/A")
                metrics = seed_result.get("metrics", {})
                lines.append(f"**Seed {seed}:**")
                lines.append(f"- RMSE: {metrics.get('rmse', 0):.6f}")
                lines.append(f"- R-AUC MSE: {metrics.get('r_auc_mse', 0):.6f}")
                lines.append("")
        if multi_seed.get("plot_files"):
            lines.append("### Визуализация по сиду")
            lines.append("")
            for pf in multi_seed.get("plot_files", []):
                lines.append(f"![Multi-seed]({Path(pf).as_posix()})")
                lines.append("")
    else:
        lines.append("Multi-seed experiment results not found.")
    lines.append("")
    
    # Baseline comparison section
    lines.append("## 3. Baseline Comparison")
    lines.append("")
    if results.get("baselines"):
        baselines = results["baselines"]
        lines.append("### Comparison Table")
        lines.append("")
        
        if "comparison_table" in baselines:
            df = pd.DataFrame(baselines["comparison_table"])
            try:
                lines.append(df.to_markdown(index=False))
            except ImportError:
                if len(df) > 0:
                    # Create header
                    headers = list(df.columns)
                    lines.append("| " + " | ".join(headers) + " |")
                    lines.append("|" + "|".join(["---" for _ in headers]) + "|")
                    # Add rows
                    for _, row in df.iterrows():
                        values = []
                        for col in headers:
                            val = row[col]
                            if isinstance(val, float):
                                # Форматирование для разных типов метрик
                                if "Ratio" in col or "Coverage" in col:
                                    values.append(f"{val:.2f}")
                                elif "AUC" in col or "MSE" in col or "RMSE" in col or "MAE" in col:
                                    values.append(f"{val:.6f}")
                                else:
                                    values.append(f"{val:.6f}")
                            else:
                                values.append(str(val))
                        lines.append("| " + " | ".join(values) + " |")
            lines.append("")
            
            if "Rejection Ratio (%)" in df.columns:
                lines.append("### Метрики неопределенности")
                lines.append("")
                lines.append("- **Rejection Ratio**: Нормализованная метрика качества неопределенности (0-100%). ")
                lines.append("  Чем выше значение, тем лучше модель ранжирует ошибки по неопределенности.")
                lines.append("- **Rejection AUC**: Площадь под кривой отбрасывания (rejection curve).")
                lines.append("- **F-Beta AUC**: Площадь под кривой F-beta для оценки качества неопределенности.")
                lines.append("")
            
            # Визуализация барчартов
            if baselines.get("plot"):
                plot_path = Path(baselines["plot"])
                if plot_path.exists():
                    lines.append(f"![Baseline metrics]({plot_path.as_posix()})")
                    lines.append("")
            cb_curve = Path("experiments/baselines/catboost_training_curve.png")
            if cb_curve.exists():
                lines.append(f"![CatBoost training]({cb_curve.as_posix()})")
                lines.append("")
            delta_plot = Path("experiments/baselines/baseline_delta_vs_hmtl.png")
            if delta_plot.exists():
                lines.append(f"![Δ vs HMTL]({delta_plot.as_posix()})")
                lines.append("")
        
        lines.append("### Best Models")
        lines.append("")
        if "best_models" in baselines:
            for metric, best in baselines["best_models"].items():
                lines.append(f"- **{metric}:** {best['model']} ({best['value']:.6f})")
        lines.append("")
        lines.append("_HMTL включен в сравнение базлайнов по умолчанию._")
        lines.append("")
    else:
        lines.append("Baseline comparison results not found.")
    lines.append("")
    
    # Summary section
    lines.append("## 4. Summary and Conclusions")
    lines.append("")
    lines.append("### Key Findings")
    lines.append("")
    
    findings = []
    
    # Add key findings based on results
    if results.get("baselines") and results["baselines"].get("best_models"):
        best_r_auc = results["baselines"]["best_models"].get("R-AUC MSE")
        if best_r_auc:
            findings.append(f"- **Best R-AUC MSE:** {best_r_auc['model']} ({best_r_auc['value']:.6f})")
        
        best_rmse = results["baselines"]["best_models"].get("RMSE")
        if best_rmse:
            findings.append(f"- **Best RMSE:** {best_rmse['model']} ({best_rmse['value']:.6f})")
    
    if results.get("main_experiment") and results["main_experiment"].get("metrics"):
        m = results["main_experiment"]["metrics"]
        if m.get("val_rmse") is not None:
            findings.append(
                f"- **HMTL (val)** RMSE {m.get('val_rmse', 0):.6f}, R-AUC MSE {m.get('val_r_auc_mse', 0):.6f}"
            )
        if m.get("test_rmse") is not None:
            findings.append(
                f"- **HMTL (test)** RMSE {m.get('test_rmse', 0):.6f}, R-AUC MSE {m.get('test_r_auc_mse', 0):.6f}"
            )
        cov_90 = m.get("val_coverage@90")
        if cov_90 is not None:
            findings.append(f"- **Conformal coverage@90 (val):** {cov_90*100:.2f}%")
    
    if results.get("multi_seed") and results["multi_seed"].get("aggregated_metrics"):
        r_auc_stats = results["multi_seed"]["aggregated_metrics"].get("r_auc_mse", {})
        if r_auc_stats:
            mean_r_auc = r_auc_stats.get("mean", 0)
            std_r_auc = r_auc_stats.get("std", 0)
            findings.append(f"- **HMTL R-AUC MSE (multi-seed):** {mean_r_auc:.6f} ± {std_r_auc:.6f}")
        
        rmse_stats = results["multi_seed"]["aggregated_metrics"].get("rmse", {})
        if rmse_stats:
            mean_rmse = rmse_stats.get("mean", 0)
            std_rmse = rmse_stats.get("std", 0)
            findings.append(f"- **HMTL RMSE (multi-seed):** {mean_rmse:.6f} ± {std_rmse:.6f}")
    
    if findings:
        lines.extend(findings)
    else:
        lines.append("- Results collection incomplete. Please check individual experiment outputs.")
    
    lines.append("")
    lines.append("### Model Comparison")
    lines.append("")
    
    if results.get("baselines") and results["baselines"].get("comparison_table"):
        lines.append("Based on the baseline comparison:")
        lines.append("")
        df = pd.DataFrame(results["baselines"]["comparison_table"])
        
        # Compare HMTL vs baselines if available
        if "hmtl" in df["Model"].values or any("hmtl" in str(m).lower() for m in df["Model"].values):
            lines.append("- HMTL model performance compared to baselines:")
        else:
            lines.append("- Baseline models comparison:")
        
        # Show top 3 models by R-AUC MSE if available
        if "R-AUC MSE" in df.columns:
            cols_to_show = ["Model", "R-AUC MSE", "RMSE"]
            if "Rejection Ratio (%)" in df.columns:
                cols_to_show.append("Rejection Ratio (%)")
            top_models = df.nsmallest(3, "R-AUC MSE")[cols_to_show]
            lines.append("")
            lines.append("Top 3 models by R-AUC MSE:")
            lines.append("")
            try:
                lines.append(top_models.to_markdown(index=False))
            except ImportError:
                headers = " | ".join(cols_to_show)
                lines.append(f"| {headers} |")
                lines.append("|" + "|".join(["---" for _ in cols_to_show]) + "|")
                for _, row in top_models.iterrows():
                    values = []
                    for col in cols_to_show:
                        val = row[col]
                        if isinstance(val, float):
                            if "Ratio" in col:
                                values.append(f"{val:.2f}")
                            else:
                                values.append(f"{val:.6f}")
                        else:
                            values.append(str(val))
                    lines.append("| " + " | ".join(values) + " |")
    
    lines.append("")
    lines.append("### Recommendations")
    lines.append("")
    recommendations = [
        "- Review plots in `experiments/plots/` for detailed visualizations",
        "- Check `experiments/baselines/comparison_table.csv` for full baseline comparison",
        "- Examine `experiments/multi_seed/results.json` for detailed multi-seed statistics",
        "- Analyze error-retention curves to understand uncertainty calibration quality",
        "- Compare coverage metrics before and after conformal calibration",
    ]
    lines.extend(recommendations)
    lines.append("")
    
    # Add file locations
    lines.append("### Output Files")
    lines.append("")
    lines.append("All experiment outputs are saved in the following locations:")
    lines.append("")
    lines.append("- `experiments/plots/` - Visualization plots (error-retention, calibration, reliability)")
    lines.append("- `experiments/baselines/` - Baseline comparison results")
    lines.append("- `experiments/multi_seed/` - Multi-seed experiment results")
    lines.append("- `experiments/runs/` - Individual training runs (if MLflow disabled)")
    lines.append("")
    
    # Write report
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    logger.info(f"Report saved to: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all experiments and generate comprehensive report")
    parser.add_argument("--skip-main", action="store_true", help="Skip main experiment")
    parser.add_argument("--skip-multi-seed", action="store_true", help="Skip multi-seed experiments")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baseline comparisons")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip ablation studies")
    parser.add_argument("--n-seeds", type=int, default=3, help="Number of seeds for multi-seed experiment")
    parser.add_argument("--output", default="experiments/full_report", help="Output directory for report")
    parser.add_argument("--data", default="configs/data.yaml", help="Path to data config")
    parser.add_argument("--model", default="configs/model_snn.yaml", help="Path to model config")
    parser.add_argument("--train", default="configs/train.yaml", help="Path to train config")
    parser.add_argument("--ensemble", default="configs/ensemble.yaml", help="Path to ensemble config")
    parser.add_argument("--baseline-models", nargs="+", 
                       default=["single_mlp", "flat_mtl", "catboost", "hmtl"],
                       choices=["single_mlp", "flat_mtl", "catboost", "hmtl"],
                       help="Baseline models to compare (default: all)")
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("Full Experiment Pipeline and Report Generation")
    logger.info("=" * 80)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    # 1. Main experiment
    if not args.skip_main:
        logger.info("=" * 80)
        logger.info("Step 1: Running Main HMTL Experiment")
        logger.info("=" * 80)
        
        main_output_dir = output_dir / "main"
        main_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with log_timing("Main experiment", logger):
                result = run_experiment(
                    data_config=args.data,
                    model_config=args.model,
                    train_config=args.train,
                    ensemble_config=args.ensemble,
                    return_models=False,
                )
            
            # Save main experiment results
            main_results_file = main_output_dir / "results.json"
            with open(main_results_file, "w") as f:
                json.dump({"metrics": result["metrics"]}, f, indent=2, default=str)
            
            logger.info(f"Main experiment results saved to {main_results_file}")

            # Copy generated plots into the report folder for embedding
            plots_src = Path("experiments/plots")
            plots_dst = main_output_dir / "plots"
            if plots_src.exists():
                shutil.copytree(plots_src, plots_dst, dirs_exist_ok=True)
                logger.info(f"Main experiment plots copied to {plots_dst}")
            else:
                logger.warning("No plots found at experiments/plots to copy into report")
        except Exception as e:
            logger.error(f"Main experiment failed: {e}")
            logger.exception(e)
    else:
        logger.info("Skipping main experiment (--skip-main)")
    
    # 2. Multi-seed experiments
    if not args.skip_multi_seed:
        logger.info("=" * 80)
        logger.info(f"Step 2: Running Multi-Seed Experiments (n_seeds={args.n_seeds})")
        logger.info("=" * 80)
        
        multi_seed_output_dir = output_dir / "multi_seed"
        success = run_command(
            [
                "python", "scripts/run_multi_seed.py",
                "--n_seeds", str(args.n_seeds),
                "--output", str(multi_seed_output_dir),
                "--data", args.data,
                "--model", args.model,
                "--train", args.train,
                "--ensemble", args.ensemble,
            ],
            "Multi-seed experiments",
            logger,
        )
        if not success:
            logger.warning("Multi-seed experiments failed, continuing...")
    else:
        logger.info("Skipping multi-seed experiments (--skip-multi-seed)")
    
    # 3. Baseline comparisons
    if not args.skip_baselines:
        logger.info("=" * 80)
        logger.info("Step 3: Running Baseline Comparisons")
        logger.info("=" * 80)
        
        baseline_output_dir = output_dir / "baselines"
        
        # Try to use HMTL results from main experiment if available
        main_results_file = output_dir / "main" / "results.json"
        hmtl_from_main_arg = []
        if main_results_file.exists():
            hmtl_from_main_arg = ["--hmtl-from-main", str(main_results_file)]
            logger.info(f"Will use HMTL results from main experiment: {main_results_file}")
        
        baseline_cmd = [
            "python", "scripts/run_baselines.py",
            "--baselines"] + args.baseline_models + [
            "--data", args.data,
            "--train", args.train,
            "--output", str(baseline_output_dir),
        ] + hmtl_from_main_arg
        success = run_command(
            baseline_cmd,
            "Baseline comparisons",
            logger,
        )
        if not success:
            logger.warning("Baseline comparisons failed, continuing...")
    else:
        logger.info("Skipping baseline comparisons (--skip-baselines)")
    
    # 4. Ablation studies
    if not args.skip_ablation:
        logger.info("=" * 80)
        logger.info("Step 4: Running Ablation Studies")
        logger.info("=" * 80)
        
        ablation_output_dir = output_dir / "ablation"
        success = run_command(
            [
                "python", "scripts/run_ablation.py",
                "--data", args.data,
                "--model", args.model,
                "--train", args.train,
            ],
            "Ablation studies",
            logger,
        )
        if not success:
            logger.warning("Ablation studies failed, continuing...")
    else:
        logger.info("Skipping ablation studies (--skip-ablation)")
    
    # 5. Collect results and generate report
    logger.info("=" * 80)
    logger.info("Step 5: Collecting Results and Generating Report")
    logger.info("=" * 80)
    
    with log_timing("Report generation", logger):
        results = collect_results(output_dir, logger)
        
        # Generate markdown report
        report_file = output_dir / "report.md"
        generate_markdown_report(results, report_file, logger)
        
        # Also save results as JSON
        results_file = output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {results_file}")
    
    # Final summary
    elapsed_time = time.time() - start_time
    logger.info("=" * 80)
    logger.info("Pipeline Complete")
    logger.info("=" * 80)
    logger.info(f"Total time: {elapsed_time:.2f} seconds")
    logger.info(f"Report: {output_dir / 'report.md'}")
    logger.info(f"Results: {output_dir / 'results.json'}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

