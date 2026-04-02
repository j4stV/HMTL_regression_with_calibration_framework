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
    captured_early_stop_metrics: list[str] = []
    captured_hybrid_weights: list[float] = []
    captured_grad_clip: list[float | None] = []
    captured_schedulers: list[str] = []

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
        captured_early_stop_metrics.append(cfg.early_stop_metric)
        captured_hybrid_weights.append(cfg.hybrid_r_auc_weight)
        captured_grad_clip.append(cfg.grad_clip_norm)
        captured_schedulers.append(cfg.lr_scheduler_name)
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
        early_stop_metric="rmse",
        hybrid_r_auc_weight=0.33,
        grad_clip_norm=0.75,
        lr_scheduler_name="cosine",
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
    assert captured_early_stop_metrics == ["rmse", "rmse"]
    assert captured_hybrid_weights == [0.33, 0.33]
    assert captured_grad_clip == [0.75, 0.75]
    assert captured_schedulers == ["cosine", "cosine"]


def test_fit_ensemble_stratified_kfold_cycles_when_requested_splits_too_high(monkeypatch):
    captured_val_sizes: list[int] = []

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
        captured_val_sizes.append(int(len(X_va)))
        return 0.5

    monkeypatch.setattr("src.train.loop.train_model", fake_train_model)

    X_tr = np.random.randn(12, 4)
    y_tr = np.array([0.0] * 4 + [1.0] * 4 + [2.0] * 4)
    X_va = np.random.randn(5, 4)
    y_va = np.random.randn(5)

    models, avg_score = fit_ensemble(
        build_model_fn=lambda: _DummyModel(),
        X_tr=X_tr,
        y_tr=y_tr,
        X_va=X_va,
        y_va=y_va,
        n_bins=3,
        ens_cfg=EnsembleConfig(n_models=6, bagging="stratified_kfold"),
        train_cfg=TrainConfig(epochs=1, batch_size=4, seed=321),
    )

    assert len(models) == 6
    assert avg_score == 0.5
    # With 12 samples and 3 balanced bins, effective_splits=4 -> fold val size is 3.
    assert captured_val_sizes == [3, 3, 3, 3, 3, 3]
