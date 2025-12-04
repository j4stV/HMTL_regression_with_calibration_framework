"""Script for ablation studies."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any

from src.data.preprocess import PreprocessConfig, TabularPreprocessor
from src.models.hmtl import HMTLModel
from src.train.loop import TrainConfig, train_model
from src.train.ensemble import EnsembleConfig, fit_ensemble
from src.eval.evaluator import evaluate_on_dataset
from src.utils.logger import setup_logging, get_logger, log_timing, log_metrics


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_ablation_study(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    input_dim: int,
    base_config: Dict[str, Any],
    ablation_configs: list[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Run ablation study with different configurations.
    
    Args:
        X_tr, y_tr: Training data
        X_va, y_va: Validation data
        input_dim: Input dimension
        base_config: Base configuration
        ablation_configs: List of configurations to test
    
    Returns:
        Dictionary mapping config name to metrics
    """
    logger = get_logger("ablation")
    results = {}
    
    for i, config in enumerate(ablation_configs):
        config_name = config.get("name", f"config_{i+1}")
        logger.info("=" * 80)
        logger.info(f"Ablation Study: {config_name}")
        logger.info("=" * 80)
        
        # Merge with base config
        merged_config = {**base_config, **config}
        
        # Build model
        def build_model():
            return HMTLModel(
                input_dim=input_dim,
                hidden_width=int(merged_config.get("hidden_width", 512)),
                depth_low=int(merged_config.get("depth_low", 12)),
                depth_high=int(merged_config.get("depth_high", 18)),
                alpha_dropout=float(merged_config.get("alpha_dropout", 0.0003)),
                n_bins=int(merged_config.get("n_bins", 5)),
                aux_weight=float(merged_config.get("aux_weight", 0.3)),
                enable_aux=bool(merged_config.get("enable_aux", True)),
                aux_task=str(merged_config.get("aux_task", "bins")),
            )
        
        # Train single model (quick ablation)
        model = build_model()
        train_conf = TrainConfig(
            lr=float(merged_config.get("lr", 3e-4)),
            epochs=int(merged_config.get("epochs", 200)),
            batch_size=int(merged_config.get("batch_size", 256)),
            patience=int(merged_config.get("patience", 20)),
            aux_weight=float(merged_config.get("aux_weight", 0.3)),
            optimizer=str(merged_config.get("optimizer", "radam_lookahead")),
            weight_decay=float(merged_config.get("weight_decay", 0.0)),
        )
        
        score = train_model(model, X_tr, y_tr, X_va, y_va, n_bins=5, cfg=train_conf)
        
        # Evaluate
        from src.eval.ensemble import ensemble_predict
        y_pred, uncertainty, _, _ = ensemble_predict([model], X_va)
        mse = np.mean((y_va - y_pred) ** 2)
        rmse = np.sqrt(mse)
        
        from src.eval.r_auc_mse import r_auc_mse
        r_auc = r_auc_mse((y_va - y_pred) ** 2, uncertainty)
        
        results[config_name] = {
            "rmse": rmse,
            "mse": mse,
            "r_auc_mse": r_auc,
            "val_score": score,
        }
        
        logger.info(f"{config_name} - RMSE: {rmse:.6f}, R-AUC MSE: {r_auc:.6f}")
    
    return results


def main() -> None:
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("Ablation Study")
    logger.info("=" * 80)
    
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--data", default="configs/data.yaml", help="Path to data configuration file")
    parser.add_argument("--model", default="configs/model_snn.yaml", help="Path to model configuration file")
    parser.add_argument("--train", default="configs/train.yaml", help="Path to training configuration file")
    args = parser.parse_args()
    
    # Load configs
    data_cfg = load_yaml(Path(args.data))
    model_cfg = load_yaml(Path(args.model))
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
    
    # Base configuration
    base_config = {
        "hidden_width": int(model_cfg["encoder"]["hidden_width"]),
        "depth_low": int(model_cfg["hmtl"]["low_layer"]),
        "depth_high": int(model_cfg["hmtl"]["high_layer"]),
        "alpha_dropout": float(model_cfg["encoder"]["alpha_dropout"]),
        "n_bins": int(model_cfg["hmtl"]["n_bins"]),
        "aux_weight": float(model_cfg["hmtl"]["lambda_aux"]),
        "lr": float(train_cfg["optimizer"]["lr"]),
        "epochs": int(train_cfg["training"]["epochs"]),
        "batch_size": int(train_cfg["training"]["batch_size"]),
        "patience": int(train_cfg["training"]["early_stop"]["patience"]) if train_cfg["training"].get("early_stop") else 20,
        "optimizer": str(train_cfg["optimizer"].get("name", "radam_lookahead")),
        "weight_decay": float(train_cfg["optimizer"].get("weight_decay", 0.0)),
    }
    
    # Define ablation configurations
    ablation_configs = [
        {"name": "baseline_hmtl", "enable_aux": True, "aux_task": "bins"},
        {"name": "no_hmtl", "enable_aux": False},
        {"name": "hmtl_contrastive", "enable_aux": True, "aux_task": "contrastive"},
        {"name": "shallow", "depth_low": 6, "depth_high": 12},
        {"name": "deep", "depth_low": 18, "depth_high": 24},
        {"name": "low_dropout", "alpha_dropout": 0.0001},
        {"name": "high_dropout", "alpha_dropout": 0.001},
        {"name": "low_aux_weight", "aux_weight": 0.1},
        {"name": "high_aux_weight", "aux_weight": 0.5},
    ]
    
    # Run ablation study
    results = run_ablation_study(
        X_tr, y_tr, X_va, y_va,
        input_dim=input_dim,
        base_config=base_config,
        ablation_configs=ablation_configs,
    )
    
    # Summary
    logger.info("=" * 80)
    logger.info("Ablation Study Summary")
    logger.info("=" * 80)
    for name, metrics in results.items():
        logger.info(f"{name}:")
        logger.info(f"  RMSE: {metrics['rmse']:.6f}")
        logger.info(f"  R-AUC MSE: {metrics['r_auc_mse']:.6f}")
    
    # Find best configuration
    best_config = min(results.items(), key=lambda x: x[1]["r_auc_mse"])
    logger.info(f"\nBest configuration: {best_config[0]} (R-AUC MSE: {best_config[1]['r_auc_mse']:.6f})")
    
    log_metrics({f"{k}_{m}": v for k, vs in results.items() for m, v in vs.items()}, logger)


if __name__ == "__main__":
    main()

