from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from .loop import TrainConfig, compute_bins
from src.models.hmtl import HMTLModel
from src.utils.logger import get_logger, log_timing


@dataclass
class EnsembleConfig:
    n_models: int = 5
    bagging: str = "stratified_kfold"  # "stratified_kfold", "stratified_bins", or "bootstrap"


def stratified_bootstrap_indices(y_bins: np.ndarray, size: int, rng: np.random.RandomState) -> np.ndarray:
    """Stratified bootstrap sampling by target bins.
    
    Ensures each bin is represented proportionally in the bootstrap sample.
    This helps maintain diversity in the ensemble and improves coverage of rare regions.
    
    Args:
        y_bins: Target bins for stratification
        size: Desired sample size
        rng: Random number generator
    
    Returns:
        Array of indices sampled with stratification
    """
    unique_bins = np.unique(y_bins)
    n_bins = len(unique_bins)
    idxs = []
    
    # Sample proportionally from each bin
    for bin_val in unique_bins:
        bin_indices = np.where(y_bins == bin_val)[0]
        # Proportional sampling: each bin contributes size/n_bins samples
        bin_sample_size = max(1, int(round(len(bin_indices) * size / len(y_bins))))
        sampled = rng.choice(bin_indices, size=bin_sample_size, replace=True)
        idxs.append(sampled)
    
    result = np.concatenate(idxs)
    # Ensure we have exactly 'size' samples (may be slightly off due to rounding)
    if len(result) < size:
        # Add random samples to reach desired size
        additional = rng.choice(len(y_bins), size=size - len(result), replace=True)
        result = np.concatenate([result, additional])
    elif len(result) > size:
        # Randomly subsample if we have too many
        result = rng.choice(result, size=size, replace=False)
    
    return result


def fit_ensemble(
    build_model_fn,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    n_bins: int,
    ens_cfg: EnsembleConfig,
    train_cfg: TrainConfig,
    history_dir: Path | str | None = None,
) -> Tuple[List[HMTLModel], float]:
    logger = get_logger("ensemble")
    
    with log_timing(f"Ensemble training ({ens_cfg.n_models} models)", logger):
        logger.info(f"Starting ensemble training with {ens_cfg.n_models} models")
        logger.info(f"Bagging strategy: {ens_cfg.bagging}")
        logger.debug(f"Training data shape: {X_tr.shape}, Validation data shape: {X_va.shape}")
        
        # Use different seeds for each model in ensemble
        base_seed = train_cfg.seed if train_cfg.seed is not None else 42
        rng = np.random.RandomState(base_seed)
        models: list[HMTLModel] = []
        # Determine if we should use rounding (check first model)
        temp_model = build_model_fn()
        use_rounding = (getattr(temp_model, "aux_task", "bins") == "contrastive")
        del temp_model
        y_bins = compute_bins(y_tr, n_bins, use_rounding=use_rounding)
        scores: list[float] = []
        
        # Prepare splits if using StratifiedKFold
        splits = None
        if ens_cfg.bagging == "stratified_kfold":
            skf = StratifiedKFold(n_splits=ens_cfg.n_models, shuffle=True, random_state=base_seed)
            splits = list(skf.split(X_tr, y_bins))
            logger.info(f"Using StratifiedKFold with {ens_cfg.n_models} splits")
        
        # Progress bar for ensemble training
        ensemble_pbar = tqdm(range(ens_cfg.n_models), desc="Ensemble Training", unit="model", leave=True)
        
        for i in ensemble_pbar:
            logger.info(f"Training model {i+1}/{ens_cfg.n_models}")
            
            # Use different seed for each model
            model_seed = base_seed + i if train_cfg.seed is not None else None
            model_train_cfg = TrainConfig(
                lr=train_cfg.lr,
                epochs=train_cfg.epochs,
                batch_size=train_cfg.batch_size,
                patience=train_cfg.patience,
                aux_weight=train_cfg.aux_weight,
                optimizer=train_cfg.optimizer,
                lookahead_k=train_cfg.lookahead_k,
                lookahead_alpha=train_cfg.lookahead_alpha,
                seed=model_seed,
            )
            
            # Sample indices based on bagging strategy
            if ens_cfg.bagging == "stratified_kfold":
                # Use StratifiedKFold splits ()
                train_idx, val_idx = splits[i]
                X_tr_split = X_tr[train_idx]
                y_tr_split = y_tr[train_idx]
                X_va_split = X_tr[val_idx]  # Use fold validation set instead of global validation
                y_va_split = y_tr[val_idx]
                logger.debug(f"Model {i+1}: Using StratifiedKFold split (train: {len(train_idx)}, val: {len(val_idx)})")
            elif ens_cfg.bagging == "stratified_bins":
                idx = stratified_bootstrap_indices(y_bins, size=len(y_tr), rng=rng)
                X_tr_split = X_tr[idx]
                y_tr_split = y_tr[idx]
                X_va_split = X_va
                y_va_split = y_va
                logger.debug(f"Model {i+1}: Using stratified bootstrap (sampled {len(idx)} indices)")
            else:
                idx = rng.choice(len(y_tr), size=len(y_tr), replace=True)
                X_tr_split = X_tr[idx]
                y_tr_split = y_tr[idx]
                X_va_split = X_va
                y_va_split = y_va
                logger.debug(f"Model {i+1}: Using standard bootstrap (sampled {len(idx)} indices)")
            
            m = build_model_fn()
            
            # Train model
            from .loop import train_model
            history: list[dict] = []
            history_meta: dict = {}
            sc = train_model(
                m,
                X_tr_split,
                y_tr_split,
                X_va_split,
                y_va_split,
                n_bins=n_bins,
                cfg=model_train_cfg,
                history=history,
                history_meta=history_meta,
            )
            
            scores.append(sc)
            models.append(m)

            # Save history and plot
            if history_dir is not None:
                history_path = Path(history_dir)
                history_path.mkdir(parents=True, exist_ok=True)
                hist_file = history_path / f"model_{i+1}_history.json"
                with open(hist_file, "w", encoding="utf-8") as f:
                    json.dump({"history": history, "meta": history_meta}, f, indent=2)
                # Plot simple curves (loss / val metrics)
                if history:
                    epochs = [h["epoch"] for h in history]
                    train_loss = [h.get("train_loss", np.nan) for h in history]
                    val_r = [h.get("val_r_auc_mse", np.nan) for h in history]
                    val_rmse = [h.get("val_rmse", np.nan) for h in history]
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    axes[0].plot(epochs, train_loss, label="train_loss", color="blue")
                    axes[0].set_title("Train loss")
                    axes[0].set_xlabel("Epoch")
                    axes[0].grid(True, alpha=0.3)
                    axes[1].plot(epochs, val_r, label="val R-AUC MSE", color="orange")
                    axes[1].plot(epochs, val_rmse, label="val RMSE", color="green")
                    axes[1].set_title("Validation metrics")
                    axes[1].set_xlabel("Epoch")
                    axes[1].grid(True, alpha=0.3)
                    for ax in axes:
                        if history_meta.get("best_epoch") is not None:
                            ax.axvline(history_meta["best_epoch"], color="red", linestyle="--", alpha=0.6, label="best")
                        if history_meta.get("early_stop_epoch") is not None:
                            ax.axvline(history_meta["early_stop_epoch"], color="purple", linestyle=":", alpha=0.6, label="early stop")
                    axes[1].legend()
                    plt.tight_layout()
                    plot_file = history_path / f"model_{i+1}_training_curve.png"
                    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                    plt.close(fig)
            
            # Update progress bar
            avg_score = np.mean(scores)
            ensemble_pbar.set_postfix({
                "current_score": f"{sc:.6f}",
                "avg_score": f"{avg_score:.6f}",
                "best": f"{min(scores):.6f}"
            })
            
            logger.info(f"Model {i+1}/{ens_cfg.n_models} completed with score: {sc:.6f}")
        
        avg_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))
        
        logger.info("Ensemble training completed")
        logger.info(f"Individual model scores: {[f'{s:.6f}' for s in scores]}")
        logger.info(f"Ensemble statistics - Mean: {avg_score:.6f}, Std: {std_score:.6f}, Min: {min_score:.6f}, Max: {max_score:.6f}")
    
    return models, avg_score


