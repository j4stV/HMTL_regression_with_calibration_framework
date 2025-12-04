"""Tests for ensemble uncertainty aggregation."""

import numpy as np
import pytest
import torch

from src.eval.ensemble import ensemble_predict
from src.models.hmtl import HMTLModel


@pytest.fixture
def dummy_models():
    """Create dummy models for testing."""
    models = []
    for i in range(3):
        model = HMTLModel(
            input_dim=10,
            hidden_width=32,
            depth_low=2,
            depth_high=4,
            alpha_dropout=0.0,
            n_bins=5,
            aux_weight=0.3,
            enable_aux=False,
        )
        # Initialize with dummy weights
        with torch.no_grad():
            for param in model.parameters():
                param.normal_(mean=0, std=0.1)
        models.append(model)
    return models


def test_ensemble_predict_shape(dummy_models):
    """Test that ensemble prediction returns correct shapes."""
    X = np.random.randn(20, 10).astype(np.float32)
    
    mu_mean, sigma_total, sigma_epistemic, sigma_aleatoric = ensemble_predict(dummy_models, X)
    
    assert mu_mean.shape == (20,)
    assert sigma_total.shape == (20,)
    assert sigma_epistemic.shape == (20,)
    assert sigma_aleatoric.shape == (20,)


def test_uncertainty_aggregation(dummy_models):
    """Test that uncertainty aggregation is correct."""
    X = np.random.randn(10, 10).astype(np.float32)
    
    mu_mean, sigma_total, sigma_epistemic, sigma_aleatoric = ensemble_predict(dummy_models, X)
    
    # Total uncertainty should be >= epistemic and aleatoric
    assert np.all(sigma_total >= sigma_epistemic)
    assert np.all(sigma_total >= sigma_aleatoric)
    
    # Total uncertainty should be approximately sqrt(Var_epi + E[σ²])
    # (allowing for numerical errors)
    expected_total_sq = sigma_epistemic ** 2 + sigma_aleatoric ** 2
    actual_total_sq = sigma_total ** 2
    np.testing.assert_allclose(actual_total_sq, expected_total_sq, rtol=1e-5)


def test_ensemble_predict_positive_uncertainty(dummy_models):
    """Test that uncertainty values are positive."""
    X = np.random.randn(10, 10).astype(np.float32)
    
    mu_mean, sigma_total, sigma_epistemic, sigma_aleatoric = ensemble_predict(dummy_models, X)
    
    assert np.all(sigma_total >= 0)
    assert np.all(sigma_epistemic >= 0)
    assert np.all(sigma_aleatoric >= 0)

