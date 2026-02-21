"""Classification task implementation with uncertainty quantification.

This module provides classification-specific components including task heads,
losses, and metrics with support for temperature scaling and focal loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.models.heads import ClassificationHead
from .base import TaskConfig, TaskHead, TaskLoss, TaskMetrics

if TYPE_CHECKING:
    from typing import Any


@dataclass
class ClassificationTaskConfig(TaskConfig):
    """Configuration for classification tasks."""

    task_type: str = "classification"
    num_classes: int = 2  # Number of classes
    class_weights: list[float] | None = None  # Optional weights for imbalanced classes
    use_focal_loss: bool = False  # Use focal loss instead of cross-entropy
    focal_alpha: float = 0.25  # Focal loss alpha parameter
    focal_gamma: float = 2.0  # Focal loss gamma parameter
    temperature_scaling: bool = True  # Enable learnable temperature scaling
    label_smoothing: float = 0.0  # Label smoothing for cross-entropy


class ClassificationTaskHead(TaskHead):
    """Adapter wrapping ClassificationHead for the task interface."""

    def __init__(self, in_dim: int, num_classes: int, use_temperature: bool = False):
        """Initialize classification head.

        Args:
            in_dim: Input feature dimension
            num_classes: Number of classes
            use_temperature: If True, adds learnable temperature for calibration
        """
        super().__init__()
        self.head = ClassificationHead(in_dim, num_classes, use_temperature=use_temperature)
        self.num_classes = num_classes

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class logits.

        Args:
            h: Encoded features (batch_size, hidden_dim)

        Returns:
            logits: Class logits (batch_size, num_classes)
        """
        return self.head(h)

    def output_size(self) -> int:
        """Return output dimension (number of classes)."""
        return self.num_classes


class ClassificationTaskLoss(TaskLoss):
    """Classification loss (cross-entropy or focal loss)."""

    def __init__(self, config: ClassificationTaskConfig):
        """Initialize classification loss.

        Args:
            config: Classification task configuration
        """
        self.config = config
        self.class_weights = None
        if config.class_weights is not None:
            self.class_weights = torch.tensor(config.class_weights, dtype=torch.float32)

    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute classification loss.

        Args:
            predictions: Logits (batch_size, num_classes)
            targets: Class indices (batch_size,)
            **kwargs: Additional parameters

        Returns:
            Scalar loss tensor
        """
        # Ensure targets are long type
        targets = targets.long()

        # Move class weights to same device as predictions if needed
        class_weights = self.class_weights
        if class_weights is not None:
            class_weights = class_weights.to(predictions.device)

        if self.config.use_focal_loss:
            # Import focal loss from losses module
            from src.losses.nll import focal_loss
            return focal_loss(
                predictions,
                targets,
                alpha=self.config.focal_alpha,
                gamma=self.config.focal_gamma,
                class_weights=class_weights,
            )
        else:
            # Standard cross-entropy loss
            return F.cross_entropy(
                predictions,
                targets,
                weight=class_weights,
                label_smoothing=self.config.label_smoothing,
            )


class ClassificationTaskMetrics(TaskMetrics):
    """Compute classification metrics including calibration metrics."""

    def __init__(self, config: ClassificationTaskConfig):
        """Initialize classification metrics.

        Args:
            config: Classification task configuration
        """
        self.config = config

    def compute(
        self,
        y_true: np.ndarray,
        predictions: np.ndarray,
        uncertainty: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute classification metrics.

        Args:
            y_true: Ground truth class indices (n_samples,)
            predictions: Logits (n_samples, n_classes)
            uncertainty: Optional total uncertainty estimates (n_samples,)

        Returns:
            Dictionary with classification metrics:
            - accuracy, balanced_accuracy
            - f1_macro, f1_weighted
            - auroc (for binary classification)
            - ece (Expected Calibration Error)
            - brier (Brier score)
            - uncertainty_error_corr (if uncertainty provided)
        """
        from src.eval.classification_metrics import compute_classification_metrics

        # Convert logits to probabilities
        probs = self._softmax(predictions)

        return compute_classification_metrics(
            y_true=y_true,
            logits=predictions,
            probs=probs,
            uncertainty=uncertainty,
            num_classes=self.config.num_classes,
        )

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax.

        Args:
            logits: Logits array (n_samples, n_classes)

        Returns:
            Probabilities (n_samples, n_classes)
        """
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


class ClassificationTask:
    """Factory for creating classification task components.

    This class provides static methods to create task-specific
    heads, losses, and metrics for classification.
    """

    @staticmethod
    def create_task_head(config: ClassificationTaskConfig, in_dim: int) -> ClassificationTaskHead:
        """Create a classification task head.

        Args:
            config: Classification task configuration
            in_dim: Input feature dimension

        Returns:
            ClassificationTaskHead instance
        """
        return ClassificationTaskHead(
            in_dim=in_dim,
            num_classes=config.num_classes,
            use_temperature=config.temperature_scaling,
        )

    @staticmethod
    def create_loss(config: ClassificationTaskConfig) -> ClassificationTaskLoss:
        """Create a classification loss function.

        Args:
            config: Classification task configuration

        Returns:
            ClassificationTaskLoss instance
        """
        return ClassificationTaskLoss(config)

    @staticmethod
    def create_metrics(config: ClassificationTaskConfig) -> ClassificationTaskMetrics:
        """Create a classification metrics computer.

        Args:
            config: Classification task configuration

        Returns:
            ClassificationTaskMetrics instance
        """
        return ClassificationTaskMetrics(config)
