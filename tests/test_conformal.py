"""Tests for conformal calibration."""

import numpy as np
import pytest

from src.eval.conformal import (
    calibrate_multiple_levels,
    coverage,
    split_conformal_intervals,
    apply_intervals,
    compute_pi_metrics,
)


def test_split_conformal_intervals():
    """Test conformal quantile computation."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
    alpha = 0.1  # 90% coverage
    
    q = split_conformal_intervals(y_true, y_pred, alpha=alpha)
    
    assert q > 0
    assert isinstance(q, float)


def test_apply_intervals():
    """Test applying conformal intervals."""
    y_pred = np.array([1.0, 2.0, 3.0])
    q = 0.5
    
    lower, upper = apply_intervals(y_pred, q)
    
    assert len(lower) == len(y_pred)
    assert len(upper) == len(y_pred)
    assert np.all(lower <= y_pred)
    assert np.all(upper >= y_pred)
    assert np.all(upper - lower == 2 * q)


def test_coverage():
    """Test coverage computation."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lower = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    upper = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    
    cov = coverage(y_true, lower, upper)
    
    assert 0 <= cov <= 1
    assert cov == 1.0  # All points should be covered


def test_calibrate_multiple_levels():
    """Test multi-level conformal calibration."""
    y_true_cal = np.random.randn(100)
    y_pred_cal = y_true_cal + np.random.randn(100) * 0.1
    
    results = calibrate_multiple_levels(
        y_true_cal, y_pred_cal, coverage_levels=[0.80, 0.90, 0.95]
    )
    
    assert len(results) == 3
    assert 0.80 in results
    assert 0.90 in results
    assert 0.95 in results
    
    for level, result in results.items():
        assert result.quantile > 0
        assert 0 <= result.coverage <= 1
        assert result.mean_width > 0


def test_compute_pi_metrics():
    """Test PI metrics computation."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lower = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    upper = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    
    metrics = compute_pi_metrics(y_true, lower, upper)
    
    assert "coverage" in metrics
    assert "mean_width" in metrics
    assert "median_width" in metrics
    assert "std_width" in metrics
    assert "min_width" in metrics
    assert "max_width" in metrics
    assert metrics["coverage"] == 1.0
    assert metrics["mean_width"] == 1.0


def test_conformal_coverage_accuracy():
    """Test that conformal calibration achieves approximately correct coverage."""
    np.random.seed(42)
    n_samples = 1000
    y_true_cal = np.random.randn(n_samples)
    y_pred_cal = y_true_cal + np.random.randn(n_samples) * 0.5
    
    # Test 90% coverage
    results = calibrate_multiple_levels(
        y_true_cal, y_pred_cal, coverage_levels=[0.90]
    )
    
    # Coverage should be close to target (within reasonable bounds)
    actual_coverage = results[0.90].coverage
    assert 0.85 <= actual_coverage <= 0.95  # Allow some variance


def test_conformal_quantile_consistency():
    """Test that quantile increases with coverage level."""
    np.random.seed(42)
    y_true_cal = np.random.randn(100)
    y_pred_cal = y_true_cal + np.random.randn(100) * 0.5
    
    results = calibrate_multiple_levels(
        y_true_cal, y_pred_cal, coverage_levels=[0.80, 0.90, 0.95]
    )
    
    # Higher coverage should require larger quantile
    assert results[0.80].quantile <= results[0.90].quantile
    assert results[0.90].quantile <= results[0.95].quantile

