"""Tests for preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import PreprocessConfig, TabularPreprocessor


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    data = {
        f"feature_{i}": np.random.randn(n_samples) for i in range(n_features)
    }
    data["target"] = np.random.randn(n_samples)
    
    # Add some missing values
    data["feature_0"][:10] = np.nan
    
    return pd.DataFrame(data)


def test_preprocessor_fit_transform(sample_data):
    """Test preprocessor fit and transform."""
    config = PreprocessConfig(
        impute_const=-1.0,
        quantile_binning_enabled=False,
        standardize=True,
        pca_enabled=False,
        target_standardize=True,
    )
    
    preprocessor = TabularPreprocessor(
        config, target_column="target"
    )
    
    preprocessor.fit(sample_data)
    X_transformed, y_transformed = preprocessor.transform(sample_data)
    
    assert X_transformed.shape[0] == len(sample_data)
    assert y_transformed is not None
    assert len(y_transformed) == len(sample_data)
    assert not np.isnan(X_transformed).any()
    assert not np.isnan(y_transformed).any()


def test_preprocessor_pca(sample_data):
    """Test PCA preprocessing."""
    config = PreprocessConfig(
        impute_const=-1.0,
        quantile_binning_enabled=False,
        standardize=True,
        pca_enabled=True,
        pca_n_components=0.95,
        target_standardize=True,
    )
    
    preprocessor = TabularPreprocessor(
        config, target_column="target"
    )
    
    preprocessor.fit(sample_data)
    X_transformed, y_transformed = preprocessor.transform(sample_data)
    
    assert X_transformed.shape[0] == len(sample_data)
    # PCA should reduce dimensionality
    assert X_transformed.shape[1] <= sample_data.shape[1] - 1


def test_preprocessor_target_standardization(sample_data):
    """Test target standardization."""
    config = PreprocessConfig(
        target_standardize=True,
    )
    
    preprocessor = TabularPreprocessor(
        config, target_column="target"
    )
    
    preprocessor.fit(sample_data)
    _, y_transformed = preprocessor.transform(sample_data)
    
    # Standardized target should have mean ~0 and std ~1
    assert abs(np.mean(y_transformed)) < 1e-10
    assert abs(np.std(y_transformed) - 1.0) < 1e-10

