"""Integration tests for end-to-end classification pipeline."""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import tempfile

from src.data.preprocess import PreprocessConfig, TabularPreprocessor
from src.models.hmtl import HMTLModel
from src.tasks.classification import ClassificationTask, ClassificationTaskConfig
from src.train.loop import TrainConfig, train_model
from src.eval.ensemble import ensemble_predict_classification
from src.eval.evaluator import evaluate_classification_on_dataset


def generate_synthetic_classification_data(n_samples=200, n_features=10, n_classes=3, seed=42):
    """Generate synthetic classification dataset."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    # Create separable classes
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_classes):
        start_idx = i * (n_samples // n_classes)
        end_idx = (i + 1) * (n_samples // n_classes)
        y[start_idx:end_idx] = i
        X[start_idx:end_idx] += np.random.randn(n_features) * 2 + i * 3

    return X, y


def test_preprocessing_classification():
    """Test preprocessing for classification tasks."""
    # Generate data
    X, y = generate_synthetic_classification_data(n_samples=100, n_classes=3)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y

    # Create preprocessor
    config = PreprocessConfig(
        impute_const=-1.0,
        standardize=True,
        pca_enabled=False,
        target_standardize=False,  # Must be False for classification
    )

    preprocessor = TabularPreprocessor(
        config, target_column="target", task_type="classification"
    )
    preprocessor.fit(df)

    # Transform
    X_transformed, y_transformed = preprocessor.transform(df)

    assert X_transformed.shape == (100, 10)
    assert y_transformed.shape == (100,)
    assert y_transformed.dtype == np.int64
    assert preprocessor.num_classes_ == 3


def test_classification_training():
    """Test training a classification model."""
    # Generate data
    X_tr, y_tr = generate_synthetic_classification_data(n_samples=100, n_classes=3)
    X_va, y_va = generate_synthetic_classification_data(n_samples=50, n_classes=3, seed=43)

    # Create model
    task_config = ClassificationTaskConfig(num_classes=3)
    task_head = ClassificationTask.create_task_head(task_config, in_dim=32)
    task_loss = ClassificationTask.create_loss(task_config)

    model = HMTLModel(
        input_dim=10,
        hidden_width=32,
        depth_low=2,
        depth_high=4,
        alpha_dropout=0.0,
        n_bins=3,
        aux_weight=0.5,
        task_head=task_head,
    )

    # Train
    train_config = TrainConfig(
        lr=0.001,
        epochs=5,
        batch_size=32,
        patience=10,
        seed=42,
        task_type="classification",
    )

    score = train_model(
        model=model,
        X_tr=X_tr,
        y_tr=y_tr,
        X_va=X_va,
        y_va=y_va,
        n_bins=3,
        cfg=train_config,
        task_loss=task_loss,
    )

    assert score is not None
    assert 0 <= score <= 10  # NLL score


def test_ensemble_classification_prediction():
    """Test ensemble prediction for classification."""
    # Generate data
    X_test, y_test = generate_synthetic_classification_data(n_samples=50, n_classes=3, seed=44)

    # Create ensemble
    models = []
    for i in range(3):
        task_config = ClassificationTaskConfig(num_classes=3)
        task_head = ClassificationTask.create_task_head(task_config, in_dim=32)

        model = HMTLModel(
            input_dim=10,
            hidden_width=32,
            depth_low=2,
            depth_high=4,
            alpha_dropout=0.0,
            n_bins=3,
            aux_weight=0.5,
            task_head=task_head,
        )
        model.eval()
        models.append(model)

    # Predict
    logits_mean, probs_mean, unc_total, unc_epi, unc_alea = ensemble_predict_classification(
        models, X_test
    )

    assert logits_mean.shape == (50, 3)
    assert probs_mean.shape == (50, 3)
    assert np.allclose(probs_mean.sum(axis=-1), 1.0)
    assert unc_total.shape == (50,)
    assert unc_epi.shape == (50,)
    assert unc_alea.shape == (50,)

    # Check MI decomposition: total = epistemic + aleatoric (approximately)
    assert np.all(unc_total >= 0)
    assert np.all(unc_epi >= 0)
    assert np.all(unc_alea >= 0)


def test_end_to_end_classification_pipeline():
    """Test complete end-to-end classification pipeline."""
    # Generate datasets
    X_tr, y_tr = generate_synthetic_classification_data(n_samples=150, n_classes=3, seed=42)
    X_va, y_va = generate_synthetic_classification_data(n_samples=50, n_classes=3, seed=43)
    X_te, y_te = generate_synthetic_classification_data(n_samples=50, n_classes=3, seed=44)

    # Preprocess
    df_tr = pd.DataFrame(X_tr, columns=[f"f{i}" for i in range(X_tr.shape[1])])
    df_tr["target"] = y_tr

    config = PreprocessConfig(standardize=True, target_standardize=False)
    preprocessor = TabularPreprocessor(config, target_column="target", task_type="classification")
    preprocessor.fit(df_tr)

    X_tr_prep, y_tr_prep = preprocessor.transform(df_tr)

    df_va = pd.DataFrame(X_va, columns=[f"f{i}" for i in range(X_va.shape[1])])
    df_va["target"] = y_va
    X_va_prep, y_va_prep = preprocessor.transform(df_va)

    # Train ensemble
    task_config = ClassificationTaskConfig(num_classes=3)
    models = []

    for i in range(2):  # Small ensemble for speed
        task_head = ClassificationTask.create_task_head(task_config, in_dim=32)
        task_loss = ClassificationTask.create_loss(task_config)

        model = HMTLModel(
            input_dim=X_tr_prep.shape[1],
            hidden_width=32,
            depth_low=2,
            depth_high=4,
            alpha_dropout=0.0,
            n_bins=3,
            aux_weight=0.5,
            task_head=task_head,
        )

        train_config = TrainConfig(
            lr=0.01,
            epochs=3,
            batch_size=32,
            patience=10,
            seed=42 + i,
            task_type="classification",
        )

        train_model(
            model=model,
            X_tr=X_tr_prep,
            y_tr=y_tr_prep,
            X_va=X_va_prep,
            y_va=y_va_prep,
            n_bins=3,
            cfg=train_config,
            task_loss=task_loss,
        )

        models.append(model)

    # Evaluate
    df_te = pd.DataFrame(X_te, columns=[f"f{i}" for i in range(X_te.shape[1])])
    df_te["target"] = y_te
    X_te_prep, y_te_prep = preprocessor.transform(df_te)

    results = evaluate_classification_on_dataset(
        models=models,
        X=X_te_prep,
        y_true=y_te_prep,
        X_cal=X_va_prep,
        y_cal=y_va_prep,
        coverage_levels=[0.80, 0.90],
    )

    # Verify results structure
    assert "metrics" in results
    assert "conformal_results" in results
    assert "predictions" in results
    assert "uncertainty" in results

    metrics = results["metrics"]
    assert "accuracy" in metrics
    assert "ece" in metrics
    assert "brier" in metrics
    assert 0 <= metrics["accuracy"] <= 1

    # Check conformal results
    assert 0.80 in results["conformal_results"]
    assert 0.90 in results["conformal_results"]


def test_backward_compatibility_regression():
    """Test that existing regression functionality still works."""
    from src.tasks.regression import RegressionTask, RegressionTaskConfig

    # Generate regression data
    X_tr = np.random.randn(100, 10)
    y_tr = np.random.randn(100)
    X_va = np.random.randn(50, 10)
    y_va = np.random.randn(50)

    # Create regression model (default, no task_head)
    model = HMTLModel(
        input_dim=10,
        hidden_width=32,
        depth_low=2,
        depth_high=4,
        alpha_dropout=0.0,
        n_bins=5,
        aux_weight=0.5,
    )

    # Train
    train_config = TrainConfig(
        lr=0.001,
        epochs=3,
        batch_size=32,
        patience=10,
        seed=42,
        task_type="regression",  # Default
    )

    score = train_model(
        model=model,
        X_tr=X_tr,
        y_tr=y_tr,
        X_va=X_va,
        y_va=y_va,
        n_bins=5,
        cfg=train_config,
    )

    assert score is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
