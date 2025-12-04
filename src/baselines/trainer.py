"""Training utilities for baseline models."""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch import nn

from src.baselines.catboost_baseline import CatBoostBaseline
from src.baselines.flat_mtl import FlatMTLModel
from src.baselines.single_mlp import SingleMLPModel
from src.models.hmtl import HMTLModel
from src.train.loop import TrainConfig, train_model
from src.train.ensemble import EnsembleConfig, fit_ensemble
from src.utils.logger import get_logger


def train_single_mlp_baseline(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    input_dim: int,
    hidden_width: int = 512,
    depth: int = 12,
    alpha_dropout: float = 0.0003,
    train_cfg: TrainConfig | None = None,
) -> SingleMLPModel:
    """Train single MLP baseline."""
    logger = get_logger("baselines.trainer")
    
    if train_cfg is None:
        train_cfg = TrainConfig()
    
    model = SingleMLPModel(
        input_dim=input_dim,
        hidden_width=hidden_width,
        depth=depth,
        alpha_dropout=alpha_dropout,
    )
    
    # Use dummy bins for compatibility with train_model
    n_bins = 5
    score = train_model(model, X_tr, y_tr, X_va, y_va, n_bins=n_bins, cfg=train_cfg)
    
    logger.info(f"Single MLP baseline trained. Validation score: {score:.6f}")
    return model


def train_flat_mtl_baseline(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    input_dim: int,
    hidden_width: int = 512,
    depth: int = 12,
    alpha_dropout: float = 0.0003,
    n_bins: int = 5,
    aux_weight: float = 0.3,
    train_cfg: TrainConfig | None = None,
) -> FlatMTLModel:
    """Train flat MTL baseline."""
    logger = get_logger("baselines.trainer")
    
    if train_cfg is None:
        train_cfg = TrainConfig(aux_weight=aux_weight)
    
    model = FlatMTLModel(
        input_dim=input_dim,
        hidden_width=hidden_width,
        depth=depth,
        alpha_dropout=alpha_dropout,
        n_bins=n_bins,
        aux_weight=aux_weight,
        enable_aux=True,
    )
    
    score = train_model(model, X_tr, y_tr, X_va, y_va, n_bins=n_bins, cfg=train_cfg)
    
    logger.info(f"Flat MTL baseline trained. Validation score: {score:.6f}")
    return model


def train_hmtl_baseline(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    input_dim: int,
    hidden_width: int = 512,
    depth_low: int = 12,
    depth_high: int = 18,
    alpha_dropout: float = 0.0003,
    n_bins: int = 5,
    aux_weight: float = 0.3,
    n_models: int = 10,
    train_cfg: TrainConfig | None = None,
    ensemble_cfg: EnsembleConfig | None = None,
    sigma_max: float = 5.0,
) -> List[HMTLModel]:
    """Train HMTL ensemble baseline."""
    logger = get_logger("baselines.trainer")
    
    if train_cfg is None:
        train_cfg = TrainConfig(aux_weight=aux_weight)
    else:
        # Ensure aux_weight is set
        train_cfg = TrainConfig(
            lr=train_cfg.lr,
            epochs=train_cfg.epochs,
            batch_size=train_cfg.batch_size,
            patience=train_cfg.patience,
            aux_weight=aux_weight,
            optimizer=train_cfg.optimizer,
            lookahead_k=train_cfg.lookahead_k,
            lookahead_alpha=train_cfg.lookahead_alpha,
            weight_decay=train_cfg.weight_decay,
            sigma_reg_weight=train_cfg.sigma_reg_weight,
            seed=train_cfg.seed,
        )
    
    if ensemble_cfg is None:
        ensemble_cfg = EnsembleConfig(n_models=n_models, bagging="stratified_bins")
    
    def build_model() -> HMTLModel:
        return HMTLModel(
            input_dim=input_dim,
            hidden_width=hidden_width,
            depth_low=depth_low,
            depth_high=depth_high,
            alpha_dropout=alpha_dropout,
            n_bins=n_bins,
            aux_weight=aux_weight,
            enable_aux=True,
            aux_task="bins",
            sigma_max=sigma_max,
        )
    
    models, avg_score = fit_ensemble(
        build_model_fn=build_model,
        X_tr=X_tr,
        y_tr=y_tr,
        X_va=X_va,
        y_va=y_va,
        n_bins=n_bins,
        ens_cfg=ensemble_cfg,
        train_cfg=train_cfg,
    )
    
    logger.info(f"HMTL baseline trained. Ensemble average score: {avg_score:.6f}")
    return models


def train_catboost_baseline(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    n_models: int = 10,
    iterations: int = 1000,
    learning_rate: float = 0.1,
    depth: int = 6,
    random_seed: int = 42,
) -> CatBoostBaseline:
    """Train CatBoost baseline."""
    logger = get_logger("baselines.trainer")
    
    baseline = CatBoostBaseline(
        n_models=n_models,
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        random_seed=random_seed,
    )
    
    baseline.fit(X_tr, y_tr)
    
    logger.info("CatBoost baseline trained")
    return baseline

