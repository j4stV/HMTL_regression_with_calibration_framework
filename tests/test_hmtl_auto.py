"""Tests for :mod:`src.hmtl.auto`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.hmtl.auto import (
    infer_size_class,
    infer_task_type,
    size_adaptive_overrides,
    summarize_data,
)


def test_infer_size_class_boundaries():
    assert infer_size_class(1) == "tiny"
    assert infer_size_class(999) == "tiny"
    assert infer_size_class(1_000) == "small"
    assert infer_size_class(9_999) == "small"
    assert infer_size_class(10_000) == "medium"
    assert infer_size_class(99_999) == "medium"
    assert infer_size_class(100_000) == "large"
    assert infer_size_class(999_999) == "large"
    assert infer_size_class(1_000_000) == "xl"
    assert infer_size_class(10_000_000) == "xl"


def test_infer_task_type_regression_on_continuous_float():
    y = np.random.RandomState(0).normal(size=200)
    assert infer_task_type(y) == "regression"


def test_infer_task_type_binary_on_zero_one():
    y = np.random.RandomState(0).randint(0, 2, size=200)
    assert infer_task_type(y) == "binary"


def test_infer_task_type_multiclass_on_small_int_cardinality():
    y = np.random.RandomState(0).randint(0, 5, size=200)
    assert infer_task_type(y) == "multiclass"


def test_infer_task_type_classification_on_string_targets():
    y = pd.Series(["a", "b", "a", "c", "b", "a"])
    assert infer_task_type(y) == "multiclass"


def test_summarize_data_regression():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(
        {
            "num1": rng.normal(size=500),
            "num2": rng.normal(size=500),
            "binflag": rng.randint(0, 2, size=500),
            "cat": rng.choice(["a", "b", "c"], size=500),
        }
    )
    y = pd.Series(rng.normal(size=500), name="target")

    summary = summarize_data(X, y, target_name="target")

    assert summary.n_rows == 500
    assert summary.n_features == 4
    assert summary.size_class == "tiny"  # 500 < 1000
    assert summary.task_type == "regression"
    assert "num1" in summary.numeric_columns
    assert "binflag" in summary.binary_columns
    assert "cat" in summary.low_card_cat_columns


def test_summarize_data_classification_imbalance():
    rng = np.random.RandomState(0)
    # Imbalanced: 90% class 0, 10% class 1
    y = np.concatenate([np.zeros(900), np.ones(100)])
    X = pd.DataFrame({"f": rng.normal(size=1000)})
    summary = summarize_data(X, y)

    assert summary.task_type == "binary"
    assert summary.n_classes == 2
    assert summary.class_imbalance_ratio == pytest.approx(9.0, rel=1e-6)


def test_size_adaptive_overrides_scales_with_size():
    rng = np.random.RandomState(0)
    for n, expected_bs_upper in [(500, 200), (5_000, 1024), (50_000, 4096)]:
        X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
        y = pd.Series(rng.normal(size=n))
        summary = summarize_data(X, y)
        overrides = size_adaptive_overrides(summary)
        assert "hidden_width" in overrides
        assert "batch_size" in overrides


def test_size_adaptive_overrides_sets_stratified_bagging_for_skewed_target():
    rng = np.random.RandomState(0)
    # Heavy right skew: exponential-like target
    y = rng.exponential(scale=1.0, size=2_000) ** 3
    X = pd.DataFrame({"f": rng.normal(size=2_000)})
    summary = summarize_data(X, y)
    overrides = size_adaptive_overrides(summary)
    assert overrides.get("bagging") == "stratified_bins"
