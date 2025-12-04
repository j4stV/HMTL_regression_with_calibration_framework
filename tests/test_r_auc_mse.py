"""Tests for R-AUC MSE metric."""

import numpy as np
import pytest

from src.eval.r_auc_mse import error_retention_curve, r_auc_mse


def test_error_retention_curve_shape():
    """Test error-retention curve shape."""
    mse_per_point = np.random.rand(100)
    uncertainty = np.random.rand(100)
    
    x, y = error_retention_curve(mse_per_point, uncertainty, num_thresholds=50)
    
    assert len(x) == 50
    assert len(y) == 50
    assert np.all(x >= 0) and np.all(x <= 1)
    assert np.all(y >= 0)


def test_error_retention_curve_monotonic():
    """Test that error-retention curve is monotonic (increasing MSE as retention increases)."""
    # Create data where uncertainty correlates with error
    uncertainty = np.linspace(0, 1, 100)
    mse_per_point = uncertainty ** 2  # Higher uncertainty -> higher error
    
    x, y = error_retention_curve(mse_per_point, uncertainty, num_thresholds=50)
    
    # As we retain more points (higher x), average MSE should increase
    # (we're keeping points with higher uncertainty/error)
    assert np.all(np.diff(y) >= -1e-10)  # Allow small numerical errors


def test_r_auc_mse():
    """Test R-AUC MSE computation."""
    mse_per_point = np.random.rand(100)
    uncertainty = np.random.rand(100)
    
    score = r_auc_mse(mse_per_point, uncertainty)
    
    assert score >= 0
    assert isinstance(score, float)


def test_r_auc_mse_perfect_ranking():
    """Test R-AUC MSE with perfect uncertainty ranking."""
    # Perfect case: uncertainty perfectly ranks errors
    uncertainty = np.linspace(0, 1, 100)
    mse_per_point = uncertainty ** 2  # Perfect correlation
    
    score = r_auc_mse(mse_per_point, uncertainty)
    
    # Score should be reasonable (not NaN or inf)
    assert np.isfinite(score)
    assert score >= 0


def test_r_auc_mse_consistency():
    """Test that R-AUC MSE is consistent across different sample sizes."""
    np.random.seed(42)
    
    # Test with different sample sizes
    for n_samples in [50, 100, 200]:
        mse_per_point = np.random.rand(n_samples)
        uncertainty = np.random.rand(n_samples)
        
        score = r_auc_mse(mse_per_point, uncertainty)
        
        assert np.isfinite(score)
        assert score >= 0


def test_error_retention_curve_edge_cases():
    """Test error-retention curve with edge cases."""
    # All zeros
    mse_per_point = np.zeros(100)
    uncertainty = np.random.rand(100)
    x, y = error_retention_curve(mse_per_point, uncertainty)
    assert len(x) > 0
    assert np.all(y == 0)
    
    # Constant uncertainty
    mse_per_point = np.random.rand(100)
    uncertainty = np.ones(100)
    x, y = error_retention_curve(mse_per_point, uncertainty)
    assert len(x) > 0
    assert np.all(y >= 0)
