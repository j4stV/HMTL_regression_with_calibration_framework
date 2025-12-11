"""Script to run baseline comparisons."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data.preprocess import PreprocessConfig, TabularPreprocessor
from src.baselines.trainer import (
    train_single_mlp_baseline,
    train_flat_mtl_baseline,
    train_catboost_baseline,
    train_hmtl_baseline,
)
from src.eval.evaluator import evaluate_on_dataset
from src.eval.metrics import EvaluationMetrics
from src.train.loop import TrainConfig
from src.train.ensemble import EnsembleConfig
from src.utils.logger import setup_logging, get_logger, log_timing, log_config, log_metrics


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_all_metrics(y_true, y_pred, uncertainty):
    """Вычисляет все метрики включая метрики из референсного репозитория."""
    from src.eval.metrics import evaluate_comprehensive
    
    metrics = evaluate_comprehensive(
        y_true=y_true,
        y_pred=y_pred,
        uncertainty=uncertainty,
        compute_reference_metrics=True,
    )
    
    result = {
        "rmse": metrics.rmse,
        "mse": metrics.mse,
        "mae": metrics.mae,
        "r_auc_mse": metrics.r_auc_mse,
        "mean_uncertainty": metrics.mean_uncertainty,
    }
    
    # Добавляем метрики из референсного репозитория
    if metrics.rejection_ratio is not None:
        result["rejection_ratio"] = metrics.rejection_ratio
        result["rejection_auc"] = metrics.rejection_auc
    if metrics.f_beta_auc is not None:
        result["f_beta_auc"] = metrics.f_beta_auc
        result["f_beta_95"] = metrics.f_beta_95
    
    return result


def main() -> None:
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("Baseline Comparisons")
    logger.info("=" * 80)
    
    parser = argparse.ArgumentParser(description="Run baseline comparisons")
    parser.add_argument("--data", default="configs/data.yaml", help="Path to data configuration file")
    parser.add_argument("--train", default="configs/train.yaml", help="Path to training configuration file")
    parser.add_argument("--baselines", nargs="+", default=["single_mlp", "flat_mtl", "catboost", "hmtl"], 
                       choices=["single_mlp", "flat_mtl", "catboost", "hmtl"],
                       help="Baselines to run")
    parser.add_argument("--output", default="experiments/baselines", help="Output directory for results")
    parser.add_argument("--hmtl-from-main", type=str, default=None, 
                       help="Path to main experiment results JSON to use HMTL metrics from there instead of training")
    args = parser.parse_args()

    training_dir = Path(args.output) / "training"

    # Гарантируем, что HMTL всегда участвует в сравнении (требование отчета)
    if "hmtl" not in args.baselines:
        args.baselines.append("hmtl")
        args.baselines = list(dict.fromkeys(args.baselines))  # сохраняем порядок без дублей
        logger.info("Добавляем HMTL к списку базлайнов для сравнения")
    
    # Load configs
    data_cfg = load_yaml(Path(args.data))
    train_cfg = load_yaml(Path(args.train))
    
    # Load data
    logger.info("Loading datasets...")
    train_path = data_cfg["paths"]["train_csv"]
    valid_path = data_cfg["paths"]["valid_csv"]
    target_col = data_cfg["paths"]["target"]
    
    df_train = pd.read_csv(train_path)
    df_valid = pd.read_csv(valid_path)
    
    # Preprocessing
    pre_cfg = PreprocessConfig(
        impute_const=float(data_cfg["preprocess"]["impute_const"]),
        quantile_binning_enabled=bool(data_cfg["preprocess"]["quantile_binning"]["enabled"]),
        quantile_binning_bins=int(data_cfg["preprocess"]["quantile_binning"]["bins"]),
        standardize=bool(data_cfg["preprocess"]["standardize"]),
        pca_enabled=bool(data_cfg["preprocess"]["pca"]["enabled"]),
        pca_n_components=data_cfg["preprocess"]["pca"]["n_components"],
        target_standardize=bool(data_cfg["preprocess"]["target_standardize"]),
    )
    
    pre = TabularPreprocessor(pre_cfg, target_column=target_col).fit(df_train)
    X_tr, y_tr = pre.transform(df_train)
    X_va, y_va = pre.transform(df_valid)
    
    input_dim = X_tr.shape[1]
    
    # Training config
    train_conf = TrainConfig(
        lr=float(train_cfg["optimizer"]["lr"]),
        epochs=int(train_cfg["training"]["epochs"]),
        batch_size=int(train_cfg["training"]["batch_size"]),
        patience=int(train_cfg["training"]["early_stop"]["patience"]) if train_cfg["training"].get("early_stop") else 10,
        optimizer=str(train_cfg["optimizer"].get("name", "radam_lookahead")),
        lookahead_k=int(train_cfg["optimizer"].get("lookahead_sync_period", 6)),
        lookahead_alpha=float(train_cfg["optimizer"].get("lookahead_slow_step", 0.5)),
        weight_decay=float(train_cfg["optimizer"].get("weight_decay", 0.0)),
        sigma_reg_weight=float(train_cfg["training"].get("sigma_reg_weight", 0.01)),
        seed=int(train_cfg["training"].get("seed")) if train_cfg["training"].get("seed") else None,
    )
    
    results = {}
    
    # Single MLP baseline
    if "single_mlp" in args.baselines:
        logger.info("=" * 80)
        logger.info("Training Single MLP Baseline")
        logger.info("=" * 80)
        model = train_single_mlp_baseline(
            X_tr, y_tr, X_va, y_va,
            input_dim=input_dim,
            train_cfg=train_conf,
        )
        
        # Evaluate
        from src.eval.ensemble import ensemble_predict
        y_pred, uncertainty, _, _ = ensemble_predict([model], X_va)
        results["single_mlp"] = compute_all_metrics(y_va, y_pred, uncertainty)
        logger.info(f"Single MLP - RMSE: {results['single_mlp']['rmse']:.6f}, "
                   f"R-AUC MSE: {results['single_mlp']['r_auc_mse']:.6f}")
        if "rejection_ratio" in results["single_mlp"]:
            logger.info(f"  Rejection Ratio: {results['single_mlp']['rejection_ratio']:.2f}%")
    
    # Flat MTL baseline
    if "flat_mtl" in args.baselines:
        logger.info("=" * 80)
        logger.info("Training Flat MTL Baseline")
        logger.info("=" * 80)
        model = train_flat_mtl_baseline(
            X_tr, y_tr, X_va, y_va,
            input_dim=input_dim,
            n_bins=5,
            train_cfg=train_conf,
        )
        
        # Evaluate
        from src.eval.ensemble import ensemble_predict
        y_pred, uncertainty, _, _ = ensemble_predict([model], X_va)
        results["flat_mtl"] = compute_all_metrics(y_va, y_pred, uncertainty)
        logger.info(f"Flat MTL - RMSE: {results['flat_mtl']['rmse']:.6f}, "
                   f"R-AUC MSE: {results['flat_mtl']['r_auc_mse']:.6f}")
        if "rejection_ratio" in results["flat_mtl"]:
            logger.info(f"  Rejection Ratio: {results['flat_mtl']['rejection_ratio']:.2f}%")
    
    # CatBoost baseline
    if "catboost" in args.baselines:
        logger.info("=" * 80)
        logger.info("Training CatBoost Baseline")
        logger.info("=" * 80)
        try:
            baseline = train_catboost_baseline(X_tr, y_tr, X_va, y_va, n_models=10)
            y_pred, uncertainty, _, _ = baseline.predict(X_va)
            
            # Plot training curves from first CatBoost model if available
            try:
                if baseline.models:
                    evals = baseline.models[0].get_evals_result()
                    learn = evals.get("learn", {})
                    valid = evals.get("validation", {})
                    if learn:
                        cb_fig, cb_ax = plt.subplots(figsize=(8, 5))
                        if "RMSE" in learn:
                            cb_ax.plot(learn["RMSE"], label="train RMSE")
                        if "RMSE" in valid:
                            cb_ax.plot(valid["RMSE"], label="val RMSE")
                        cb_ax.set_title("CatBoost training (RMSE)")
                        cb_ax.set_xlabel("Iteration")
                        cb_ax.set_ylabel("RMSE")
                        cb_ax.grid(True, alpha=0.3)
                        cb_ax.legend()
                        Path(args.output).mkdir(parents=True, exist_ok=True)
                        cb_plot = Path(args.output) / "catboost_training_curve.png"
                        plt.tight_layout()
                        plt.savefig(cb_plot, dpi=300, bbox_inches="tight")
                        plt.close(cb_fig)
                        logger.info(f"CatBoost training curve saved to {cb_plot}")
            except Exception as e:
                logger.warning(f"CatBoost training curve plotting failed: {e}")
            
            # Compute all metrics
            results["catboost"] = compute_all_metrics(y_va, y_pred, uncertainty)
            
            # Conformal calibration for CatBoost
            from src.eval.conformal import calibrate_multiple_levels, compute_pi_metrics
            conformal_results = calibrate_multiple_levels(
                y_true_cal=y_va,
                y_pred_cal=y_pred,
                coverage_levels=[0.80, 0.90, 0.95],
            )
            
            coverage_metrics = {}
            for level in [0.80, 0.90, 0.95]:
                if level in conformal_results:
                    from src.eval.conformal import apply_intervals
                    lower, upper = apply_intervals(y_pred, conformal_results[level].quantile)
                    pi_metrics = compute_pi_metrics(y_va, lower, upper)
                    coverage_metrics[f"coverage@{int(level*100)}"] = pi_metrics["coverage"]
                    coverage_metrics[f"width@{int(level*100)}"] = pi_metrics["mean_width"]
            
            results["catboost"].update(coverage_metrics)
            logger.info(f"CatBoost - RMSE: {results['catboost']['rmse']:.6f}, "
                       f"R-AUC MSE: {results['catboost']['r_auc_mse']:.6f}")
            if "rejection_ratio" in results["catboost"]:
                logger.info(f"  Rejection Ratio: {results['catboost']['rejection_ratio']:.2f}%")
            for level in [0.80, 0.90, 0.95]:
                if f"coverage@{int(level*100)}" in coverage_metrics:
                    logger.info(f"  Coverage {level:.0%}: {coverage_metrics[f'coverage@{int(level*100)}']:.4%}")
        except ImportError as e:
            logger.warning(f"CatBoost not available: {e}")
        except Exception as e:
            logger.error(f"CatBoost baseline failed: {e}")
            logger.exception(e)
    
    # HMTL baseline
    if "hmtl" in args.baselines:
        logger.info("=" * 80)
        logger.info("HMTL Baseline")
        logger.info("=" * 80)
        
        # Try to load from main experiment if path provided
        hmtl_from_main = None
        if args.hmtl_from_main:
            hmtl_main_path = Path(args.hmtl_from_main)
            if hmtl_main_path.exists():
                logger.info(f"Loading HMTL results from main experiment: {hmtl_main_path}")
                try:
                    import json
                    with open(hmtl_main_path) as f:
                        hmtl_from_main = json.load(f)
                    logger.info("Successfully loaded HMTL metrics from main experiment")
                except Exception as e:
                    logger.warning(f"Failed to load HMTL from main experiment: {e}, will train instead")
            else:
                logger.warning(f"HMTL main results path does not exist: {hmtl_main_path}, will train instead")
        
        if hmtl_from_main:
            # Use metrics from main experiment
            logger.info("Using HMTL metrics from main experiment (no training needed)")
            metrics_dict = hmtl_from_main.get("metrics", hmtl_from_main)
            
            results["hmtl"] = {
                "rmse": metrics_dict.get("val_rmse", 0.0),
                "mse": metrics_dict.get("val_mse", 0.0),
                "mae": metrics_dict.get("val_mae", 0.0),
                "r_auc_mse": metrics_dict.get("val_r_auc_mse", 0.0),
                "mean_uncertainty": metrics_dict.get("val_mean_uncertainty", 0.0),
            }
            
            # Add coverage metrics if available
            for level in [80, 90, 95]:
                cov_key = f"val_coverage@{level}"
                width_key = f"val_width@{level}"
                if cov_key in metrics_dict:
                    results["hmtl"][f"coverage@{level}"] = metrics_dict[cov_key]
                if width_key in metrics_dict:
                    results["hmtl"][f"width@{level}"] = metrics_dict[width_key]
            
            # Add reference metrics if available (check both metrics dict and val_results)
            # First check in metrics dict (new format)
            if metrics_dict.get("val_rejection_ratio") is not None:
                results["hmtl"]["rejection_ratio"] = metrics_dict["val_rejection_ratio"]
                results["hmtl"]["rejection_auc"] = metrics_dict.get("val_rejection_auc")
            if metrics_dict.get("val_f_beta_auc") is not None:
                results["hmtl"]["f_beta_auc"] = metrics_dict["val_f_beta_auc"]
                results["hmtl"]["f_beta_95"] = metrics_dict.get("val_f_beta_95")
            
            # Fallback: check if val_results exists in the JSON (old format)
            if "rejection_ratio" not in results["hmtl"] and "val_results" in hmtl_from_main and hmtl_from_main["val_results"]:
                val_res = hmtl_from_main["val_results"]
                if isinstance(val_res, dict) and "metrics" in val_res:
                    val_metrics = val_res["metrics"]
                    if isinstance(val_metrics, dict):
                        if val_metrics.get("rejection_ratio") is not None:
                            results["hmtl"]["rejection_ratio"] = val_metrics["rejection_ratio"]
                        if val_metrics.get("rejection_auc") is not None:
                            results["hmtl"]["rejection_auc"] = val_metrics["rejection_auc"]
                        if val_metrics.get("f_beta_auc") is not None:
                            results["hmtl"]["f_beta_auc"] = val_metrics["f_beta_auc"]
                        if val_metrics.get("f_beta_95") is not None:
                            results["hmtl"]["f_beta_95"] = val_metrics["f_beta_95"]
            
            logger.info(f"HMTL (from main) - RMSE: {results['hmtl']['rmse']:.6f}, "
                       f"R-AUC MSE: {results['hmtl']['r_auc_mse']:.6f}")
            if "rejection_ratio" in results["hmtl"]:
                logger.info(f"  Rejection Ratio: {results['hmtl']['rejection_ratio']:.2f}%")
            for level in [80, 90, 95]:
                cov_key = f"coverage@{level}"
                if cov_key in results["hmtl"]:
                    logger.info(f"  Coverage {level}%: {results['hmtl'][cov_key]:.4%}")
        else:
            # Train HMTL baseline (original logic)
            logger.info("Training HMTL Baseline (no main experiment results found)")
            try:
                # Load model config if available, otherwise use defaults
                model_cfg_path = Path("configs/model_snn.yaml")
                if model_cfg_path.exists():
                    model_cfg = load_yaml(model_cfg_path)
                    hidden_width = int(model_cfg["encoder"]["hidden_width"])
                    depth_low = int(model_cfg["hmtl"]["low_layer"])
                    depth_high = int(model_cfg["hmtl"]["high_layer"])
                    alpha_dropout = float(model_cfg["encoder"]["alpha_dropout"])
                    n_bins = int(model_cfg["hmtl"]["n_bins"])
                    aux_weight = float(model_cfg["hmtl"]["lambda_aux"])
                    sigma_max = float(model_cfg.get("regression_head", {}).get("sigma_max", 5.0))
                else:
                    # Use defaults
                    hidden_width = 512
                    depth_low = 12
                    depth_high = 18
                    alpha_dropout = 0.0003
                    n_bins = 5
                    aux_weight = 0.3
                    sigma_max = 5.0
                
                # Load ensemble config if available
                ensemble_cfg_path = Path("configs/ensemble.yaml")
                if ensemble_cfg_path.exists():
                    ensemble_cfg_yaml = load_yaml(ensemble_cfg_path)
                    n_models = int(ensemble_cfg_yaml["ensemble"]["n_models"])
                    bagging = str(ensemble_cfg_yaml["ensemble"]["bagging"])
                    ensemble_cfg = EnsembleConfig(n_models=n_models, bagging=bagging)
                else:
                    n_models = 10
                    ensemble_cfg = EnsembleConfig(n_models=n_models, bagging="stratified_bins")
                
                models = train_hmtl_baseline(
                    X_tr, y_tr, X_va, y_va,
                    input_dim=input_dim,
                    hidden_width=hidden_width,
                    depth_low=depth_low,
                    depth_high=depth_high,
                    alpha_dropout=alpha_dropout,
                    n_bins=n_bins,
                    aux_weight=aux_weight,
                    n_models=n_models,
                    train_cfg=train_conf,
                    ensemble_cfg=ensemble_cfg,
                    sigma_max=sigma_max,
                    history_dir=training_dir,
                )
                
                # Evaluate using comprehensive evaluator
                eval_results = evaluate_on_dataset(
                    models=models,
                    X=X_va,
                    y_true=y_va,
                    X_cal=X_va,  # Use validation set for calibration
                    y_cal=y_va,
                    coverage_levels=[0.80, 0.90, 0.95],
                    preprocessor=pre,
                    use_normalized_metrics=True,  # Compute metrics in standardized space (like baselines)
                )
                
                results["hmtl"] = {
                    "rmse": eval_results.metrics.rmse,
                    "mse": eval_results.metrics.mse,
                    "mae": eval_results.metrics.mae,
                    "r_auc_mse": eval_results.metrics.r_auc_mse,
                    "mean_uncertainty": eval_results.metrics.mean_uncertainty,
                }
                
                # Добавляем метрики из референсного репозитория
                if eval_results.metrics.rejection_ratio is not None:
                    results["hmtl"]["rejection_ratio"] = eval_results.metrics.rejection_ratio
                    results["hmtl"]["rejection_auc"] = eval_results.metrics.rejection_auc
                if eval_results.metrics.f_beta_auc is not None:
                    results["hmtl"]["f_beta_auc"] = eval_results.metrics.f_beta_auc
                    results["hmtl"]["f_beta_95"] = eval_results.metrics.f_beta_95
                
                # Add coverage metrics
                for level in [0.80, 0.90, 0.95]:
                    if level in eval_results.pi_metrics_after:
                        results["hmtl"][f"coverage@{int(level*100)}"] = eval_results.pi_metrics_after[level]["coverage"]
                        results["hmtl"][f"width@{int(level*100)}"] = eval_results.pi_metrics_after[level]["mean_width"]
                
                logger.info(f"HMTL - RMSE: {eval_results.metrics.rmse:.6f}, R-AUC MSE: {eval_results.metrics.r_auc_mse:.6f}")
                for level in [0.80, 0.90, 0.95]:
                    if level in eval_results.pi_metrics_after:
                        logger.info(f"  Coverage {level:.0%}: {eval_results.pi_metrics_after[level]['coverage']:.4%} "
                                  f"(width: {eval_results.pi_metrics_after[level]['mean_width']:.6f})")
            except Exception as e:
                logger.error(f"HMTL baseline failed: {e}")
                logger.exception(e)
    
    # Summary with detailed comparison
    logger.info("=" * 80)
    logger.info("Baseline Comparison Summary")
    logger.info("=" * 80)
    
    # Create comparison table
    comparison_data = []
    for name, metrics in results.items():
        row = {
            "Model": name,
            "RMSE": metrics.get("rmse", 0.0),
            "MSE": metrics.get("mse", 0.0),
            "MAE": metrics.get("mae", 0.0),
            "R-AUC MSE": metrics.get("r_auc_mse", 0.0),
            "Mean Uncertainty": metrics.get("mean_uncertainty", 0.0),
            "Coverage@80": metrics.get("coverage@80", 0.0),
            "Coverage@90": metrics.get("coverage@90", 0.0),
            "Coverage@95": metrics.get("coverage@95", 0.0),
        }
        
        # Добавляем метрики из референсного репозитория
        if "rejection_ratio" in metrics:
            row["Rejection Ratio (%)"] = metrics.get("rejection_ratio", 0.0)
            row["Rejection AUC"] = metrics.get("rejection_auc", 0.0)
        if "f_beta_auc" in metrics:
            row["F-Beta AUC"] = metrics.get("f_beta_auc", 0.0)
            row["F-Beta@95"] = metrics.get("f_beta_95", 0.0)
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    # Δ относительно HMTL (если есть)
    if "hmtl" in comparison_df["Model"].values:
        hmtl_row = comparison_df[comparison_df["Model"] == "hmtl"].iloc[0]
        comparison_df["ΔRMSE_vs_HMTL"] = comparison_df["RMSE"] - hmtl_row["RMSE"]
        comparison_df["ΔR-AUC_vs_HMTL"] = comparison_df["R-AUC MSE"] - hmtl_row["R-AUC MSE"]
    logger.info("\nComparison Table:")
    logger.info("\n" + comparison_df.to_string(index=False))
    
    # Find best model for each metric
    logger.info("\nBest Models:")
    for metric in ["RMSE", "R-AUC MSE", "MAE"]:
        if metric in comparison_df.columns:
            best_idx = comparison_df[metric].idxmin()
            best_model = comparison_df.loc[best_idx, "Model"]
            best_value = comparison_df.loc[best_idx, metric]
            logger.info(f"  {metric}: {best_model} ({best_value:.6f})")
    
    # Statistical comparison (if scipy available)
    try:
        from scipy import stats
        
        logger.info("\nStatistical Tests (if multiple runs available):")
        logger.info("Note: Statistical tests require multiple runs. For single run, use run_multi_seed.py")
        
        # If we had multiple runs, we could do paired t-tests here
        # For now, just log that statistical tests would go here
        
    except ImportError:
        logger.info("\nNote: scipy not available for statistical tests. Install with: pip install scipy")
    
    # Save comparison table
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_file = output_dir / "comparison_table.csv"
    comparison_df.to_csv(comparison_file, index=False)
    logger.info(f"\nComparison table saved to {comparison_file}")
    
    # Create visualization
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # RMSE comparison
        ax = axes[0, 0]
        comparison_df.plot(x="Model", y="RMSE", kind="bar", ax=ax, legend=False)
        ax.set_title("RMSE Comparison")
        ax.set_ylabel("RMSE")
        ax.tick_params(axis='x', rotation=45)
        
        # R-AUC MSE comparison
        ax = axes[0, 1]
        comparison_df.plot(x="Model", y="R-AUC MSE", kind="bar", ax=ax, legend=False, color='orange')
        ax.set_title("R-AUC MSE Comparison")
        ax.set_ylabel("R-AUC MSE")
        ax.tick_params(axis='x', rotation=45)
        
        # Coverage comparison
        ax = axes[1, 0]
        coverage_cols = [c for c in comparison_df.columns if c.startswith("Coverage@")]
        if coverage_cols:
            comparison_df.plot(x="Model", y=coverage_cols, kind="bar", ax=ax)
            ax.set_title("Coverage Comparison")
            ax.set_ylabel("Coverage")
            ax.axhline(y=0.80, color='r', linestyle='--', alpha=0.5, label='Target 80%')
            ax.axhline(y=0.90, color='g', linestyle='--', alpha=0.5, label='Target 90%')
            ax.axhline(y=0.95, color='b', linestyle='--', alpha=0.5, label='Target 95%')
            ax.legend()
            ax.tick_params(axis='x', rotation=45)
        
        # Uncertainty comparison
        ax = axes[1, 1]
        comparison_df.plot(x="Model", y="Mean Uncertainty", kind="bar", ax=ax, legend=False, color='green')
        ax.set_title("Mean Uncertainty Comparison")
        ax.set_ylabel("Mean Uncertainty")
        ax.tick_params(axis='x', rotation=45)
        
        # Δ vs HMTL (если есть)
        if "ΔRMSE_vs_HMTL" in comparison_df.columns:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            comparison_df.plot(x="Model", y=["ΔRMSE_vs_HMTL", "ΔR-AUC_vs_HMTL"], kind="bar", ax=ax2)
            ax2.set_title("Δметрики vs HMTL (ниже лучше)")
            ax2.set_ylabel("Δ (model - HMTL)")
            ax2.axhline(0, color="black", linewidth=1)
            ax2.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            delta_plot = output_dir / "baseline_delta_vs_hmtl.png"
            plt.savefig(delta_plot, dpi=300, bbox_inches="tight")
            plt.close(fig2)
            logger.info(f"Delta plot saved to {delta_plot}")
        
        plt.tight_layout()
        plot_file = output_dir / "baseline_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Comparison plot saved to {plot_file}")
        
    except Exception as e:
        logger.warning(f"Failed to create visualization: {e}")
    
    # Detailed logging
    for name, metrics in results.items():
        logger.info(f"\n{name}:")
        logger.info(f"  RMSE: {metrics.get('rmse', 0.0):.6f}")
        logger.info(f"  MSE: {metrics.get('mse', 0.0):.6f}")
        logger.info(f"  MAE: {metrics.get('mae', 0.0):.6f}")
        logger.info(f"  R-AUC MSE: {metrics.get('r_auc_mse', 0.0):.6f}")
        if 'mean_uncertainty' in metrics:
            logger.info(f"  Mean Uncertainty: {metrics['mean_uncertainty']:.6f}")
        for level in [80, 90, 95]:
            if f'coverage@{level}' in metrics:
                logger.info(f"  Coverage@{level}: {metrics[f'coverage@{level}']:.4%}")
    
    log_metrics({f"{k}_{m}": v for k, vs in results.items() for m, v in vs.items()}, logger)


if __name__ == "__main__":
    main()

