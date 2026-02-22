"""Smoke test for training path with AMP settings enabled."""

from __future__ import annotations

import numpy as np

from src.models.hmtl import HMTLModel
from src.train.loop import TrainConfig, train_model


def test_train_model_amp_smoke() -> None:
    rng = np.random.default_rng(42)
    X_tr = rng.normal(size=(96, 8)).astype(np.float32)
    y_tr = rng.normal(size=(96,)).astype(np.float32)
    X_va = rng.normal(size=(32, 8)).astype(np.float32)
    y_va = rng.normal(size=(32,)).astype(np.float32)

    model = HMTLModel(
        input_dim=8,
        hidden_width=32,
        depth_low=2,
        depth_high=4,
        alpha_dropout=0.0,
        n_bins=5,
        aux_weight=0.3,
        enable_aux=False,
    )

    cfg = TrainConfig(
        lr=1e-3,
        epochs=2,
        batch_size=16,
        patience=5,
        seed=123,
        task_type="regression",
        amp_enabled=True,
        amp_dtype="auto",
        amp_eval_enabled=True,
        show_progress=False,
    )

    score = train_model(
        model=model,
        X_tr=X_tr,
        y_tr=y_tr,
        X_va=X_va,
        y_va=y_va,
        n_bins=5,
        cfg=cfg,
    )

    assert np.isfinite(score)
