from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .loop import TrainConfig, compute_bins
from src.models.hmtl import HMTLModel
from src.utils.logger import get_logger, log_timing


@dataclass
class EnsembleConfig:
    n_models: int = 5
    bagging: str = "stratified_bins"  # or "bootstrap"


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
        y_bins = compute_bins(y_tr, n_bins)
        scores: list[float] = []
        
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
            if ens_cfg.bagging == "stratified_bins":
                idx = stratified_bootstrap_indices(y_bins, size=len(y_tr), rng=rng)
                logger.debug(f"Model {i+1}: Using stratified bootstrap (sampled {len(idx)} indices)")
            else:
                idx = rng.choice(len(y_tr), size=len(y_tr), replace=True)
                logger.debug(f"Model {i+1}: Using standard bootstrap (sampled {len(idx)} indices)")
            
            m = build_model_fn()
            
            # Train model
            from .loop import train_model
            sc = train_model(m, X_tr[idx], y_tr[idx], X_va, y_va, n_bins=n_bins, cfg=model_train_cfg)
            
            scores.append(sc)
            models.append(m)
            
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


