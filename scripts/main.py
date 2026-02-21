from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import yaml
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.preprocess import PreprocessConfig, TabularPreprocessor
from src.models.hmtl import HMTLModel
from src.train.loop import TrainConfig, train_model
from src.train.ensemble import EnsembleConfig, fit_ensemble
from src.eval.evaluator import evaluate_on_dataset, evaluate_classification_on_dataset
from src.eval.visualization import (
    visualize_classification_results,
    visualize_evaluation_results,
)
from src.utils.logger import setup_logging, get_logger, log_timing, log_config, log_metrics
from src.utils.mlflow_tracker import MLflowTracker


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_experiment(
    data_config: str | Path = "configs/data.yaml",
    model_config: str | Path = "configs/model_snn.yaml",
    train_config: str | Path = "configs/train.yaml",
    ensemble_config: str | Path = "configs/ensemble.yaml",
    return_models: bool = False,
) -> dict:
    """Run a single experiment and return metrics.
    
    Args:
        data_config: Path to data configuration file
        model_config: Path to model configuration file
        train_config: Path to training configuration file
        ensemble_config: Path to ensemble configuration file
        return_models: Whether to return trained models
    
    Returns:
        Dictionary with metrics and optionally models
    """
    logger = get_logger("main")
    
    logger.info("Loading configuration files...")
    with log_timing("Configuration loading", logger):
        data_cfg = load_yaml(Path(data_config))
        model_cfg = load_yaml(Path(model_config))
        train_cfg = load_yaml(Path(train_config))
        logger.info(f"Loaded configs: data={data_config}, model={model_config}, train={train_config}")
        log_config(data_cfg, logger, "Data config")
        log_config(model_cfg, logger, "Model config")
        log_config(train_cfg, logger, "Training config")

    # Determine task type from configuration
    task_cfg_dict = data_cfg.get("task", {"type": "regression"})
    task_type = task_cfg_dict.get("type", "regression")
    logger.info(f"Task type: {task_type}")

    # Create task configuration
    if task_type == "classification":
        from src.tasks.classification import ClassificationTask, ClassificationTaskConfig
        task_config = ClassificationTaskConfig(**task_cfg_dict)
        logger.info(f"Classification task: {task_config.num_classes} classes")
        if task_config.use_focal_loss:
            logger.info(f"Using focal loss (alpha={task_config.focal_alpha}, gamma={task_config.focal_gamma})")
    else:
        from src.tasks.regression import RegressionTask, RegressionTaskConfig
        task_config = RegressionTaskConfig(
            task_type="regression",
            sigma_reg_weight=float(train_cfg["training"].get("sigma_reg_weight", 0.0))
        )
        logger.info("Regression task")

    # Initialize MLflow tracking
    mlflow_enabled = train_cfg.get("logging", {}).get("mlflow", {}).get("enabled", False)
    mlflow_tracker = MLflowTracker(
        experiment_name="hmtl_calibration",
        tracking_uri=train_cfg.get("logging", {}).get("mlflow", {}).get("tracking_uri"),
        enabled=mlflow_enabled,
    )
    
    if mlflow_tracker.enabled:
        mlflow_tracker.start_run()
        # Log all configs
        mlflow_tracker.log_params({
            **{f"data.{k}": v for k, v in data_cfg.items()},
            **{f"model.{k}": v for k, v in model_cfg.items()},
            **{f"train.{k}": v for k, v in train_cfg.items()},
        })

    logger.info("Loading datasets...")
    with log_timing("Data loading", logger):
        train_path = data_cfg["paths"]["train_csv"]
        valid_path = data_cfg["paths"]["valid_csv"]
        cal_path = data_cfg["paths"].get("cal_csv")
        test_path = data_cfg["paths"].get("test_csv")
        target_col = data_cfg["paths"]["target"]
        
        logger.info(f"Loading training data from: {train_path}")
        df_train = pd.read_csv(train_path)
        logger.info(f"Training data loaded: {df_train.shape} (rows, cols)")
        
        logger.info(f"Loading validation data from: {valid_path}")
        df_valid = pd.read_csv(valid_path)
        logger.info(f"Validation data loaded: {df_valid.shape} (rows, cols)")
        
        df_cal = None
        if cal_path:
            logger.info(f"Loading calibration data from: {cal_path}")
            df_cal = pd.read_csv(cal_path)
            logger.info(f"Calibration data loaded: {df_cal.shape} (rows, cols)")
        else:
            logger.info("No separate calibration set provided, will use validation set for calibration")
        
        df_test = None
        if test_path:
            logger.info(f"Loading test data from: {test_path}")
            df_test = pd.read_csv(test_path)
            logger.info(f"Test data loaded: {df_test.shape} (rows, cols)")
        
        logger.info(f"Target column: {target_col}")

    logger.info("Setting up preprocessing...")
    pre_cfg = PreprocessConfig(
        impute_const=float(data_cfg["preprocess"]["impute_const"]),
        quantile_binning_enabled=bool(data_cfg["preprocess"]["quantile_binning"]["enabled"]),
        quantile_binning_bins=int(data_cfg["preprocess"]["quantile_binning"]["bins"]),
        standardize=bool(data_cfg["preprocess"]["standardize"]),
        pca_enabled=bool(data_cfg["preprocess"]["pca"]["enabled"]),
        pca_n_components=data_cfg["preprocess"]["pca"]["n_components"],
        target_standardize=bool(data_cfg["preprocess"]["target_standardize"]),
    )

    logger.info("Fitting preprocessor on training data...")
    pre = TabularPreprocessor(pre_cfg, target_column=target_col, task_type=task_type).fit(df_train)
    
    logger.info("Transforming datasets...")
    X_tr, y_tr = pre.transform(df_train)
    X_va, y_va = pre.transform(df_valid)
    
    X_cal, y_cal = None, None
    if df_cal is not None:
        X_cal, y_cal = pre.transform(df_cal)
        logger.info(f"Preprocessed calibration data: X={X_cal.shape}, y={y_cal.shape if y_cal is not None else None}")
    else:
        # Use validation set for calibration if no separate calibration set provided
        X_cal, y_cal = X_va, y_va
        logger.info("Using validation set for calibration")
    
    X_te, y_te = None, None
    if df_test is not None:
        X_te, y_te = pre.transform(df_test)
        logger.info(f"Preprocessed test data: X={X_te.shape}, y={y_te.shape if y_te is not None else None}")
    
    logger.info(f"Preprocessed training data: X={X_tr.shape}, y={y_tr.shape if y_tr is not None else None}")
    logger.info(f"Preprocessed validation data: X={X_va.shape}, y={y_va.shape if y_va is not None else None}")
    logger.info(f"Preprocessed calibration data: X={X_cal.shape}, y={y_cal.shape if y_cal is not None else None}")

    input_dim = X_tr.shape[1]
    logger.info(f"Model input dimension: {input_dim}")
    
    def build_model() -> HMTLModel:
        # Create task-specific head
        hidden_width = int(model_cfg["encoder"]["hidden_width"])

        if task_type == "classification":
            # Classification head
            task_head = ClassificationTask.create_task_head(task_config, in_dim=hidden_width)
        else:
            # Regression: use default RegressionHead (None = backward compatible)
            task_head = None

        # Get scale_coeff for regression (ignored for classification)
        scale_coeff = pre.target_std_ if pre.target_std_ is not None and pre.target_std_ > 1e-12 else 1.0

        return HMTLModel(
            input_dim=input_dim,
            hidden_width=hidden_width,
            depth_low=int(model_cfg["hmtl"]["low_layer"]),
            depth_high=int(model_cfg["hmtl"]["high_layer"]),
            alpha_dropout=float(model_cfg["encoder"]["alpha_dropout"]),
            n_bins=int(model_cfg["hmtl"]["n_bins"]),
            aux_weight=float(model_cfg["hmtl"]["lambda_aux"]),
            enable_aux=bool(model_cfg["hmtl"]["enabled"]),
            aux_task=str(model_cfg["hmtl"].get("aux_task", "contrastive")),  # Default to contrastive
            proj_dim=int(model_cfg["hmtl"].get("proj_dim", 50)),  # Default 50
            scale_coeff=scale_coeff,
            task_head=task_head,  # NEW: inject task head
        )
    
    logger.info("Model architecture:")
    logger.info(f"  Hidden width: {model_cfg['encoder']['hidden_width']}")
    logger.info(f"  Depth (low/high): {model_cfg['hmtl']['low_layer']}/{model_cfg['hmtl']['high_layer']}")
    logger.info(f"  Alpha dropout: {model_cfg['encoder']['alpha_dropout']}")
    logger.info(f"  N bins: {model_cfg['hmtl']['n_bins']}")
    logger.info(f"  Auxiliary weight: {model_cfg['hmtl']['lambda_aux']}")
    logger.info(f"  Auxiliary enabled: {model_cfg['hmtl']['enabled']}")

    # Create task-specific loss and metrics
    if task_type == "classification":
        task_loss = ClassificationTask.create_loss(task_config)
        task_metrics = ClassificationTask.create_metrics(task_config)
    else:
        task_loss = None  # Will use default gaussian_nll
        task_metrics = None

    train_conf = TrainConfig(
        lr=float(train_cfg["optimizer"]["lr"]),
        epochs=int(train_cfg["training"]["epochs"]),
        batch_size=int(train_cfg["training"]["batch_size"]),
        patience=int(train_cfg["training"]["early_stop"]["patience"]) if train_cfg["training"].get("early_stop") else 10,
        aux_weight=float(model_cfg["hmtl"]["lambda_aux"]),
        optimizer=str(train_cfg["optimizer"].get("name", "radam_lookahead")),
        lookahead_k=int(train_cfg["optimizer"].get("lookahead_sync_period", 6)),
        lookahead_alpha=float(train_cfg["optimizer"].get("lookahead_slow_step", 0.5)),
        weight_decay=float(train_cfg["optimizer"].get("weight_decay", 0.0)),
        sigma_reg_weight=float(train_cfg["training"].get("sigma_reg_weight", 0.01)),
        seed=int(train_cfg["training"].get("seed")) if train_cfg["training"].get("seed") else None,
        task_type=task_type,  # NEW
    )

    logger.info("Loading ensemble configuration...")
    ensemble_cfg = load_yaml(Path(ensemble_config))
    ens_conf = EnsembleConfig(
        n_models=int(ensemble_cfg["ensemble"]["n_models"]),
        bagging=str(ensemble_cfg["ensemble"]["bagging"]),
    )
    logger.info(f"Ensemble: {ens_conf.n_models} models, bagging={ens_conf.bagging}")

    logger.info("Starting ensemble training...")
    models, avg_score = fit_ensemble(
        build_model,
        X_tr,
        y_tr,
        X_va,
        y_va,
        n_bins=int(model_cfg["hmtl"]["n_bins"]),
        ens_cfg=ens_conf,
        train_cfg=train_conf,
        task_loss=task_loss,  # NEW
        task_metrics=task_metrics,  # NEW
        history_dir=Path("experiments/plots/training"),
    )
    logger.info(f"Ensemble training completed. Average score: {avg_score:.6f}")

    # Comprehensive evaluation on validation set
    logger.info("=" * 80)
    logger.info("Evaluating on Validation Set")
    logger.info("=" * 80)
    with log_timing("Validation evaluation", logger):
        if task_type == "classification":
            val_results = evaluate_classification_on_dataset(
                models=models,
                X=X_va,
                y_true=y_va,
                X_cal=X_cal,
                y_cal=y_cal,
                coverage_levels=[0.80, 0.90, 0.95],
            )

            logger.info("Validation Metrics:")
            metrics = val_results["metrics"]
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
            logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
            logger.info(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
            logger.info(f"  ECE: {metrics['ece']:.4f}")
            logger.info(f"  Brier: {metrics['brier']:.4f}")
            if 'uncertainty_error_pearson_r' in metrics:
                logger.info(f"  Uncertainty-Error Correlation (Pearson): {metrics['uncertainty_error_pearson_r']:.4f}")
                logger.info(f"  Uncertainty-Error Correlation (Spearman): {metrics['uncertainty_error_spearman_rho']:.4f}")

            for coverage_level in [0.80, 0.90, 0.95]:
                if coverage_level in val_results["conformal_results"]:
                    conf_res = val_results["conformal_results"][coverage_level]
                    logger.info(f"  Coverage {coverage_level:.0%}: {conf_res['coverage']:.4%} "
                              f"(mean set size: {conf_res['mean_set_size']:.2f})")
        else:
            val_results = evaluate_on_dataset(
                models=models,
                X=X_va,
                y_true=y_va,
                X_cal=X_cal,
                y_cal=y_cal,
                coverage_levels=[0.80, 0.90, 0.95],
                preprocessor=pre,
                use_normalized_metrics=True,
            )

            logger.info("Validation Metrics:")
            logger.info(f"  RMSE: {val_results.metrics.rmse:.6f}")
            logger.info(f"  MSE: {val_results.metrics.mse:.6f}")
            logger.info(f"  MAE: {val_results.metrics.mae:.6f}")
            logger.info(f"  R-AUC MSE: {val_results.metrics.r_auc_mse:.6f}")
            logger.info(f"  Mean Uncertainty: {val_results.metrics.mean_uncertainty:.6f}")
            logger.info(f"  Mean Epistemic: {val_results.metrics.mean_epistemic:.6f}")
            logger.info(f"  Mean Aleatoric: {val_results.metrics.mean_aleatoric:.6f}")

            for coverage_level in [0.80, 0.90, 0.95]:
                if coverage_level in val_results.pi_metrics_after:
                    logger.info(f"  Coverage {coverage_level:.0%}: {val_results.pi_metrics_after[coverage_level]['coverage']:.4%} "
                              f"(width: {val_results.pi_metrics_after[coverage_level]['mean_width']:.6f})")

    # Evaluate on test set if available
    test_results = None
    if X_te is not None and y_te is not None:
        logger.info("=" * 80)
        logger.info("Evaluating on Test Set")
        logger.info("=" * 80)
        with log_timing("Test evaluation", logger):
            if task_type == "classification":
                test_results = evaluate_classification_on_dataset(
                    models=models,
                    X=X_te,
                    y_true=y_te,
                    X_cal=X_cal,
                    y_cal=y_cal,
                    coverage_levels=[0.80, 0.90, 0.95],
                )

                logger.info("Test Metrics:")
                metrics = test_results["metrics"]
                logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
                logger.info(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
                logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
                logger.info(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
                logger.info(f"  ECE: {metrics['ece']:.4f}")
                logger.info(f"  Brier: {metrics['brier']:.4f}")
                if "uncertainty_error_pearson_r" in metrics:
                    logger.info(f"  Uncertainty-Error Correlation (Pearson): {metrics['uncertainty_error_pearson_r']:.4f}")
                    logger.info(f"  Uncertainty-Error Correlation (Spearman): {metrics['uncertainty_error_spearman_rho']:.4f}")

                for coverage_level in [0.80, 0.90, 0.95]:
                    if coverage_level in test_results["conformal_results"]:
                        conf_res = test_results["conformal_results"][coverage_level]
                        logger.info(
                            f"  Coverage {coverage_level:.0%}: {conf_res['coverage']:.4%} "
                            f"(mean set size: {conf_res['mean_set_size']:.2f})"
                        )
            else:
                test_results = evaluate_on_dataset(
                    models=models,
                    X=X_te,
                    y_true=y_te,
                    X_cal=X_cal,
                    y_cal=y_cal,
                    coverage_levels=[0.80, 0.90, 0.95],
                    preprocessor=pre,
                    use_normalized_metrics=True,  # Compute metrics in standardized space (like baselines)
                )

                logger.info("Test Metrics:")
                logger.info(f"  RMSE: {test_results.metrics.rmse:.6f}")
                logger.info(f"  MSE: {test_results.metrics.mse:.6f}")
                logger.info(f"  MAE: {test_results.metrics.mae:.6f}")
                logger.info(f"  R-AUC MSE: {test_results.metrics.r_auc_mse:.6f}")
                logger.info(f"  Mean Uncertainty: {test_results.metrics.mean_uncertainty:.6f}")

                for coverage_level in [0.80, 0.90, 0.95]:
                    if coverage_level in test_results.pi_metrics_after:
                        logger.info(f"  Coverage {coverage_level:.0%}: {test_results.pi_metrics_after[coverage_level]['coverage']:.4%} "
                                  f"(width: {test_results.pi_metrics_after[coverage_level]['mean_width']:.6f})")

    # Generate visualizations
    logger.info("=" * 80)
    logger.info("Generating Visualizations")
    logger.info("=" * 80)
    viz_output_dir = Path("experiments/plots")
    
    try:
        if task_type == "classification":
            visualize_classification_results(val_results, viz_output_dir, prefix="val_")
        else:
            visualize_evaluation_results(val_results, viz_output_dir, prefix="val_")
        logger.info(f"Validation visualizations saved to {viz_output_dir}")
    except Exception as e:
        logger.warning(f"Failed to generate validation visualizations: {e}")
        logger.exception(e)
        logger.info("Continuing pipeline execution despite visualization error...")
    
    if test_results is not None:
        try:
            if task_type == "classification":
                visualize_classification_results(test_results, viz_output_dir, prefix="test_")
            else:
                visualize_evaluation_results(test_results, viz_output_dir, prefix="test_")
            logger.info(f"Test visualizations saved to {viz_output_dir}")
        except Exception as e:
            logger.warning(f"Failed to generate test visualizations: {e}")
            logger.exception(e)
            logger.info("Continuing pipeline execution despite visualization error...")

    # Final summary
    logger.info("=" * 80)
    logger.info("Final Results Summary")
    logger.info("=" * 80)
    if task_type == "classification":
        val_metrics = val_results["metrics"]
        final_metrics = {
            "task_type": "classification",
            "ensemble_avg_val_nll": avg_score,
            "n_models": len(models),
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_f1_weighted": val_metrics["f1_weighted"],
            "val_auroc": val_metrics.get("auroc", np.nan),
            "val_ece": val_metrics["ece"],
            "val_brier": val_metrics["brier"],
            "val_mean_uncertainty": float(np.mean(val_results["uncertainty"]["total"])),
            "val_mean_epistemic": float(np.mean(val_results["uncertainty"]["epistemic"])),
            "val_mean_aleatoric": float(np.mean(val_results["uncertainty"]["aleatoric"])),
        }

        if "uncertainty_error_pearson_r" in val_metrics:
            final_metrics["val_uncertainty_error_pearson_r"] = val_metrics["uncertainty_error_pearson_r"]
            final_metrics["val_uncertainty_error_spearman_rho"] = val_metrics["uncertainty_error_spearman_rho"]

        for coverage_level in [0.80, 0.90, 0.95]:
            conf_res = val_results["conformal_results"].get(coverage_level)
            if conf_res is not None:
                final_metrics[f"val_coverage@{int(coverage_level*100)}"] = conf_res["coverage"]
                final_metrics[f"val_mean_set_size@{int(coverage_level*100)}"] = conf_res["mean_set_size"]

        if test_results is not None:
            test_metrics = test_results["metrics"]
            final_metrics["test_accuracy"] = test_metrics["accuracy"]
            final_metrics["test_balanced_accuracy"] = test_metrics["balanced_accuracy"]
            final_metrics["test_f1_macro"] = test_metrics["f1_macro"]
            final_metrics["test_f1_weighted"] = test_metrics["f1_weighted"]
            final_metrics["test_auroc"] = test_metrics.get("auroc", np.nan)
            final_metrics["test_ece"] = test_metrics["ece"]
            final_metrics["test_brier"] = test_metrics["brier"]
            final_metrics["test_mean_uncertainty"] = float(np.mean(test_results["uncertainty"]["total"]))
            final_metrics["test_mean_epistemic"] = float(np.mean(test_results["uncertainty"]["epistemic"]))
            final_metrics["test_mean_aleatoric"] = float(np.mean(test_results["uncertainty"]["aleatoric"]))

            if "uncertainty_error_pearson_r" in test_metrics:
                final_metrics["test_uncertainty_error_pearson_r"] = test_metrics["uncertainty_error_pearson_r"]
                final_metrics["test_uncertainty_error_spearman_rho"] = test_metrics["uncertainty_error_spearman_rho"]

            for coverage_level in [0.80, 0.90, 0.95]:
                conf_res = test_results["conformal_results"].get(coverage_level)
                if conf_res is not None:
                    final_metrics[f"test_coverage@{int(coverage_level*100)}"] = conf_res["coverage"]
                    final_metrics[f"test_mean_set_size@{int(coverage_level*100)}"] = conf_res["mean_set_size"]
    else:
        final_metrics = {
            "task_type": "regression",
            "ensemble_avg_r_auc_mse": avg_score,
            "n_models": len(models),
            "val_rmse": val_results.metrics.rmse,
            "val_mse": val_results.metrics.mse,
            "val_mae": val_results.metrics.mae,
            "val_r_auc_mse": val_results.metrics.r_auc_mse,
            "val_mean_uncertainty": val_results.metrics.mean_uncertainty,
            "val_mean_epistemic": val_results.metrics.mean_epistemic,
            "val_mean_aleatoric": val_results.metrics.mean_aleatoric,
        }

        # Add reference metrics (rejection ratio, rejection AUC, F-Beta metrics)
        if val_results.metrics.rejection_ratio is not None:
            final_metrics["val_rejection_ratio"] = val_results.metrics.rejection_ratio
            final_metrics["val_rejection_auc"] = val_results.metrics.rejection_auc
        if val_results.metrics.f_beta_auc is not None:
            final_metrics["val_f_beta_auc"] = val_results.metrics.f_beta_auc
            final_metrics["val_f_beta_95"] = val_results.metrics.f_beta_95

        if test_results is not None:
            final_metrics["test_rmse"] = test_results.metrics.rmse
            final_metrics["test_mse"] = test_results.metrics.mse
            final_metrics["test_mae"] = test_results.metrics.mae
            final_metrics["test_r_auc_mse"] = test_results.metrics.r_auc_mse
            final_metrics["test_mean_uncertainty"] = test_results.metrics.mean_uncertainty
            # Add reference metrics for test set
            if test_results.metrics.rejection_ratio is not None:
                final_metrics["test_rejection_ratio"] = test_results.metrics.rejection_ratio
                final_metrics["test_rejection_auc"] = test_results.metrics.rejection_auc
            if test_results.metrics.f_beta_auc is not None:
                final_metrics["test_f_beta_auc"] = test_results.metrics.f_beta_auc
                final_metrics["test_f_beta_95"] = test_results.metrics.f_beta_95

        for coverage_level in [0.80, 0.90, 0.95]:
            if coverage_level in val_results.pi_metrics_after:
                final_metrics[f"val_coverage@{int(coverage_level*100)}"] = val_results.pi_metrics_after[coverage_level]['coverage']
                final_metrics[f"val_width@{int(coverage_level*100)}"] = val_results.pi_metrics_after[coverage_level]['mean_width']
                if test_results is not None and coverage_level in test_results.pi_metrics_after:
                    final_metrics[f"test_coverage@{int(coverage_level*100)}"] = test_results.pi_metrics_after[coverage_level]['coverage']
                    final_metrics[f"test_width@{int(coverage_level*100)}"] = test_results.pi_metrics_after[coverage_level]['mean_width']
    
    log_metrics(final_metrics, logger, "Final Metrics")
    
    # Log to MLflow
    if mlflow_tracker.enabled:
        mlflow_tracker.log_metrics(final_metrics)
        if Path("experiments/plots").exists():
            mlflow_tracker.log_artifact("experiments/plots")
        mlflow_tracker.end_run()
    
    # Also print to console for easy access
    print("\n" + "=" * 80)
    print("Final Results:")
    print("=" * 80)
    for key, value in final_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    print("=" * 80)
    
    logger.info("Training pipeline completed successfully!")
    
    result = {
        "metrics": final_metrics,
        "val_results": val_results,
        "test_results": test_results,
    }
    
    if return_models:
        result["models"] = models
    
    return result


def main() -> None:
    # Initialize logging first
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("Starting HMTL with Calibration Training")
    logger.info("=" * 80)
    
    parser = argparse.ArgumentParser(description="Train HMTL ensemble model with conformal calibration")
    parser.add_argument("--data", default="configs/data.yaml", help="Path to data configuration file")
    parser.add_argument("--model", default="configs/model_snn.yaml", help="Path to model configuration file")
    parser.add_argument("--train", default="configs/train.yaml", help="Path to training configuration file")
    parser.add_argument("--ensemble", default="configs/ensemble.yaml", help="Path to ensemble configuration file")
    args = parser.parse_args()

    run_experiment(
        data_config=args.data,
        model_config=args.model,
        train_config=args.train,
        ensemble_config=args.ensemble,
    )


if __name__ == "__main__":
    main()
