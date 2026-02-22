"""Tests for CatBoost virtual ensemble backoff behavior."""

from __future__ import annotations

import numpy as np
import pytest

from src.baselines import catboost_baseline as cb_mod


class _FakeCatBoostError(Exception):
    pass


def test_virtual_ensemble_count_retries_down_until_success(monkeypatch: pytest.MonkeyPatch):
    attempts: list[int] = []

    class _FakeModel:
        def __init__(self, **kwargs):
            self.tree_count_ = 10

        def fit(self, X, y, **kwargs):
            return None

        def virtual_ensembles_predict(self, X, prediction_type, virtual_ensembles_count):
            attempts.append(int(virtual_ensembles_count))
            if virtual_ensembles_count > 2:
                raise _FakeCatBoostError(
                    f"catboost/private/libs/algo/apply.cpp:542: "
                    f"Not enough trees in model for {virtual_ensembles_count} virtual Ensembles"
                )
            n = len(X)
            return np.column_stack(
                [
                    np.full(n, 1.5, dtype=float),
                    np.full(n, 0.2, dtype=float),
                    np.full(n, 0.3, dtype=float),
                ]
            )

        def predict(self, X):
            return np.full(len(X), 1.5, dtype=float)

    monkeypatch.setattr(cb_mod, "HAS_CATBOOST", True)
    monkeypatch.setattr(cb_mod, "CatBoostError", _FakeCatBoostError)
    monkeypatch.setattr(cb_mod, "CatBoostRegressor", _FakeModel)

    baseline = cb_mod.CatBoostBaseline(n_models=10)
    X = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    y = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    baseline.fit(X, y)

    y_pred, unc_total, unc_epi, unc_alea = baseline.predict(X)

    assert attempts[0] == 10
    assert attempts[-1] == 2
    assert np.allclose(y_pred, np.full(3, 1.5))
    assert np.allclose(unc_epi, np.full(3, 0.2))
    assert np.allclose(unc_alea, np.full(3, 0.3))
    assert np.allclose(unc_total, np.sqrt(0.2**2 + 0.3**2))


def test_virtual_ensemble_all_failures_fall_back_to_mean_only(monkeypatch: pytest.MonkeyPatch):
    class _FakeModel:
        def __init__(self, **kwargs):
            self.tree_count_ = 1

        def fit(self, X, y, **kwargs):
            return None

        def virtual_ensembles_predict(self, X, prediction_type, virtual_ensembles_count):
            raise _FakeCatBoostError(
                f"catboost/private/libs/algo/apply.cpp:542: "
                f"Not enough trees in model for {virtual_ensembles_count} virtual Ensembles"
            )

        def predict(self, X):
            return np.asarray([0.1, 0.2, 0.3], dtype=float)

    monkeypatch.setattr(cb_mod, "HAS_CATBOOST", True)
    monkeypatch.setattr(cb_mod, "CatBoostError", _FakeCatBoostError)
    monkeypatch.setattr(cb_mod, "CatBoostRegressor", _FakeModel)

    baseline = cb_mod.CatBoostBaseline(n_models=10)
    X = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    y = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    baseline.fit(X, y)

    y_pred, unc_total, unc_epi, unc_alea = baseline.predict(X)

    assert np.allclose(y_pred, np.asarray([0.1, 0.2, 0.3], dtype=float))
    assert np.allclose(unc_epi, np.zeros(3))
    assert np.allclose(unc_alea, np.zeros(3))
    assert np.allclose(unc_total, np.full(3, 1e-5))
