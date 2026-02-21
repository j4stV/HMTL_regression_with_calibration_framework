"""Regression tests for task-type propagation in ensemble training."""

from __future__ import annotations

import numpy as np

from src.train.ensemble import EnsembleConfig, fit_ensemble
from src.train.loop import TrainConfig


class _DummyModel:
    """Minimal model stub for fit_ensemble tests."""

    aux_task = "bins"



def test_fit_ensemble_propagates_task_type(monkeypatch):
    captured_task_types: list[str] = []
    captured_sigma_weights: list[float] = []

    def fake_train_model(
        model,
        X_tr,
        y_tr,
        X_va,
        y_va,
        n_bins,
        cfg,
        task_loss=None,
        task_metrics=None,
        history=None,
        history_meta=None,
    ):
        captured_task_types.append(cfg.task_type)
        captured_sigma_weights.append(cfg.sigma_reg_weight)
        return 0.123

    # fit_ensemble imports train_model from src.train.loop inside the function body.
    monkeypatch.setattr("src.train.loop.train_model", fake_train_model)

    X_tr = np.random.randn(16, 4)
    y_tr = np.random.randn(16)
    X_va = np.random.randn(8, 4)
    y_va = np.random.randn(8)

    train_cfg = TrainConfig(
        epochs=1,
        batch_size=8,
        task_type="classification",
        sigma_reg_weight=0.777,
        seed=123,
    )

    models, avg_score = fit_ensemble(
        build_model_fn=lambda: _DummyModel(),
        X_tr=X_tr,
        y_tr=y_tr,
        X_va=X_va,
        y_va=y_va,
        n_bins=3,
        ens_cfg=EnsembleConfig(n_models=2, bagging="bootstrap"),
        train_cfg=train_cfg,
    )

    assert len(models) == 2
    assert avg_score == 0.123
    assert captured_task_types == ["classification", "classification"]
    assert captured_sigma_weights == [0.777, 0.777]
