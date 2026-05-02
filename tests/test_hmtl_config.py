"""Tests for :mod:`src.hmtl.config` and :mod:`src.hmtl.presets`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.hmtl.auto import summarize_data
from src.hmtl.config import Config
from src.hmtl.presets import resolve_preset


def _summary(n: int = 5_000) -> "DataSummary":
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(rng.normal(size=n))
    return summarize_data(X, y)


def test_config_copy_applies_overrides():
    cfg = Config(n_models=5, lr=1e-3)
    updated = cfg.copy(n_models=10, epochs=42)
    assert updated.n_models == 10
    assert updated.lr == 1e-3  # preserved
    assert updated.epochs == 42
    # original untouched
    assert cfg.n_models == 5


def test_resolve_preset_fast_sets_small_ensemble():
    cfg = resolve_preset("fast", _summary())
    assert cfg.preset == "fast"
    assert cfg.n_models == 3
    assert cfg.bagging == "full_dataset"


def test_resolve_preset_medium_uses_auto_aux():
    cfg = resolve_preset("medium", _summary())
    assert cfg.preset == "medium"
    assert cfg.aux_task == "auto"
    assert cfg.bagging == "stratified_kfold"


def test_resolve_preset_best_quality_has_larger_ensemble():
    cfg = resolve_preset("best_quality", _summary(n=50_000))
    assert cfg.preset == "best_quality"
    assert cfg.n_models >= 10


def test_resolve_preset_overrides_win():
    cfg = resolve_preset("fast", _summary(), overrides={"n_models": 7, "lr": 1e-2})
    assert cfg.n_models == 7
    assert cfg.lr == 1e-2


def test_resolve_preset_rejects_unknown_name():
    import pytest

    with pytest.raises(ValueError, match="Unknown preset"):
        resolve_preset("lightning", _summary())


def test_config_task_type_set_from_summary():
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"f": rng.normal(size=500)})
    y = pd.Series(rng.randint(0, 3, size=500))
    summary = summarize_data(X, y)
    cfg = resolve_preset("fast", summary)
    assert cfg.task_type == "classification"
