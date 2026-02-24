"""Unit tests for regression early-stop metric selection helpers."""

from __future__ import annotations

import math

import pytest

from src.train.loop import (
    _normalize_regression_early_stop_metric,
    _resolve_regression_validation_score,
)


def test_normalize_regression_early_stop_metric_aliases() -> None:
    assert _normalize_regression_early_stop_metric("hybrid") == "hybrid_rmse_rauc"
    assert _normalize_regression_early_stop_metric("hybrid_rmse_rauc") == "hybrid_rmse_rauc"
    assert _normalize_regression_early_stop_metric("rmse") == "rmse"
    assert _normalize_regression_early_stop_metric("r_auc_mse") == "r_auc_mse"


def test_resolve_regression_validation_score_rmse() -> None:
    score = _resolve_regression_validation_score(
        metric_name="rmse",
        rmse=0.5,
        r_auc_mse_score=0.3,
        hybrid_r_auc_weight=0.25,
    )
    assert score == 0.5


def test_resolve_regression_validation_score_legacy_r_auc() -> None:
    score = _resolve_regression_validation_score(
        metric_name="r_auc_mse",
        rmse=0.5,
        r_auc_mse_score=0.3,
        hybrid_r_auc_weight=0.25,
    )
    assert score == 0.3


def test_resolve_regression_validation_score_hybrid() -> None:
    score = _resolve_regression_validation_score(
        metric_name="hybrid_rmse_rauc",
        rmse=0.4,
        r_auc_mse_score=0.6,
        hybrid_r_auc_weight=0.25,
    )
    assert math.isclose(score, 0.55, rel_tol=0.0, abs_tol=1e-12)


def test_normalize_regression_early_stop_metric_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported regression early-stop metric"):
        _normalize_regression_early_stop_metric("unknown_metric")
