"""Basic tests for classification functionality."""

import pytest
import numpy as np
import torch

from src.models.heads import ClassificationHead
from src.tasks.classification import ClassificationTask, ClassificationTaskConfig
from src.eval.classification_metrics import (
    expected_calibration_error,
    brier_score_multi,
    uncertainty_error_correlation,
)
from src.eval.ensemble import ensemble_predict_classification, softmax, entropy
from src.eval.conformal import (
    split_conformal_classification,
    apply_conformal_sets,
    coverage_classification,
)


def test_classification_head():
    """Test ClassificationHead forward pass."""
    head = ClassificationHead(in_dim=32, num_classes=3)
    h = torch.randn(10, 32)
    logits = head(h)
    assert logits.shape == (10, 3)


def test_classification_head_with_temperature():
    """Test ClassificationHead with temperature scaling."""
    head = ClassificationHead(in_dim=32, num_classes=3, use_temperature=True)
    assert hasattr(head, "temperature")
    h = torch.randn(10, 32)
    logits = head(h)
    assert logits.shape == (10, 3)


def test_classification_task_config():
    """Test ClassificationTaskConfig creation."""
    config = ClassificationTaskConfig(num_classes=5, use_focal_loss=True)
    assert config.task_type == "classification"
    assert config.num_classes == 5
    assert config.use_focal_loss is True


def test_classification_task_factory():
    """Test ClassificationTask factory methods."""
    config = ClassificationTaskConfig(num_classes=3)

    # Create head
    head = ClassificationTask.create_task_head(config, in_dim=32)
    assert head.num_classes == 3

    # Create loss
    loss = ClassificationTask.create_loss(config)
    assert loss is not None

    # Create metrics
    metrics = ClassificationTask.create_metrics(config)
    assert metrics is not None


def test_expected_calibration_error():
    """Test ECE computation."""
    n_samples = 100
    n_classes = 3

    # Perfect calibration
    y_true = np.random.randint(0, n_classes, n_samples)
    probs = np.zeros((n_samples, n_classes))
    probs[np.arange(n_samples), y_true] = 1.0  # Perfect confidence on true class

    ece = expected_calibration_error(y_true, probs)
    assert 0 <= ece <= 1
    assert ece < 0.1  # Should be well calibrated


def test_brier_score():
    """Test Brier score computation."""
    n_samples = 100
    n_classes = 3
    y_true = np.random.randint(0, n_classes, n_samples)
    probs = np.random.dirichlet(np.ones(n_classes), n_samples)

    brier = brier_score_multi(y_true, probs)
    assert 0 <= brier <= 2  # Brier score range for multi-class


def test_uncertainty_error_correlation():
    """Test uncertainty-error correlation."""
    n_samples = 100
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = np.random.randint(0, 2, n_samples)
    uncertainty = np.random.rand(n_samples)

    corr = uncertainty_error_correlation(y_true, y_pred, uncertainty)
    assert "pearson_r" in corr
    assert "spearman_rho" in corr


def test_softmax():
    """Test numerically stable softmax."""
    logits = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    probs = softmax(logits)

    assert probs.shape == logits.shape
    assert np.allclose(probs.sum(axis=-1), 1.0)
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_entropy():
    """Test entropy computation."""
    probs = np.array([[0.5, 0.5], [0.9, 0.1], [1.0, 0.0]])
    ent = entropy(probs)

    assert ent.shape == (3,)
    assert ent[0] > ent[1]  # Uniform has higher entropy than skewed
    assert ent[2] < 0.01  # Deterministic has near-zero entropy


def test_conformal_classification():
    """Test conformal prediction for classification."""
    n_samples = 100
    n_classes = 3

    # Generate synthetic data
    y_cal = np.random.randint(0, n_classes, n_samples)
    probs_cal = np.random.dirichlet(np.ones(n_classes), n_samples)

    # Compute conformal quantile
    q = split_conformal_classification(y_cal, probs_cal, alpha=0.1)
    assert 0 <= q <= 1

    # Apply to test data
    y_test = np.random.randint(0, n_classes, 50)
    probs_test = np.random.dirichlet(np.ones(n_classes), 50)

    prediction_sets = apply_conformal_sets(probs_test, q)
    assert len(prediction_sets) == 50
    assert all(isinstance(s, set) for s in prediction_sets)

    # Compute coverage
    cov = coverage_classification(y_test, prediction_sets)
    assert 0 <= cov <= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ensemble_predict_classification_gpu():
    """Test ensemble prediction on GPU."""
    from src.models.hmtl import HMTLModel
    from src.models.heads import ClassificationHead

    # Create models with classification heads
    models = []
    for _ in range(3):
        head = ClassificationHead(in_dim=32, num_classes=3)
        model = HMTLModel(
            input_dim=10,
            hidden_width=32,
            depth_low=2,
            depth_high=4,
            alpha_dropout=0.0,
            n_bins=3,
            aux_weight=0.5,
            task_head=head,
        )
        models.append(model)

    X = np.random.randn(20, 10)
    device = torch.device("cuda")

    logits_mean, probs_mean, unc_total, unc_epi, unc_alea = ensemble_predict_classification(
        models, X, device=device
    )

    assert logits_mean.shape == (20, 3)
    assert probs_mean.shape == (20, 3)
    assert unc_total.shape == (20,)
    assert unc_epi.shape == (20,)
    assert unc_alea.shape == (20,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
