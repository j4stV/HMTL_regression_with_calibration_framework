"""Tests for CatBoost device selection and fallback behavior."""

from __future__ import annotations

import numpy as np
import pytest

from src.baselines import catboost_baseline as cb_mod


class _FakeCatBoostError(Exception):
    pass


def _patch_fake_catboost(monkeypatch: pytest.MonkeyPatch, *, fail_on_gpu: bool) -> None:
    class _FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.task_type = kwargs.get("task_type", "CPU")
            self.tree_count_ = 10

        def fit(self, X, y, **kwargs):
            if fail_on_gpu and self.task_type == "GPU":
                raise _FakeCatBoostError("GPU backend is unavailable")

    monkeypatch.setattr(cb_mod, "HAS_CATBOOST", True)
    monkeypatch.setattr(cb_mod, "CatBoostError", _FakeCatBoostError)
    monkeypatch.setattr(cb_mod, "CatBoostRegressor", _FakeModel)


def test_catboost_auto_falls_back_to_cpu_when_gpu_fails(monkeypatch: pytest.MonkeyPatch):
    _patch_fake_catboost(monkeypatch, fail_on_gpu=True)

    baseline = cb_mod.CatBoostBaseline(n_models=2, compute_device="auto")
    X = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    y = np.array([0.0, 1.0, 2.0], dtype=np.float32)

    baseline.fit(X, y)

    assert baseline._resolved_task_type == "CPU"
    assert len(baseline.models) == 2
    assert all(getattr(model, "task_type", None) == "CPU" for model in baseline.models)


def test_catboost_gpu_mode_raises_when_gpu_fails(monkeypatch: pytest.MonkeyPatch):
    _patch_fake_catboost(monkeypatch, fail_on_gpu=True)

    baseline = cb_mod.CatBoostBaseline(n_models=1, compute_device="gpu")
    X = np.array([[0.0], [1.0]], dtype=np.float32)
    y = np.array([0.0, 1.0], dtype=np.float32)

    with pytest.raises(_FakeCatBoostError):
        baseline.fit(X, y)


def test_catboost_cpu_mode_uses_cpu_only(monkeypatch: pytest.MonkeyPatch):
    _patch_fake_catboost(monkeypatch, fail_on_gpu=False)

    baseline = cb_mod.CatBoostBaseline(n_models=1, compute_device="cpu")
    X = np.array([[0.0], [1.0]], dtype=np.float32)
    y = np.array([0.0, 1.0], dtype=np.float32)

    baseline.fit(X, y)

    assert baseline._resolved_task_type == "CPU"
    assert len(baseline.models) == 1
    assert getattr(baseline.models[0], "task_type", None) == "CPU"
