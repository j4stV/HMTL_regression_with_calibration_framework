"""Abstract base classes for task-specific components.

These abstractions allow the HMTL framework to support multiple task types
(regression, classification, etc.) through polymorphism rather than
conditional logic scattered throughout the codebase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass
class TaskConfig(ABC):
    """Base configuration for task-specific parameters.

    Concrete task implementations should subclass this and add
    task-specific configuration fields.
    """

    task_type: str  # "regression", "classification", etc.


class TaskHead(nn.Module, ABC):
    """Abstract interface for task-specific output heads.

    A TaskHead takes encoded features and produces task-specific outputs.
    For regression: (mean, sigma)
    For classification: logits
    """

    @abstractmethod
    def forward(self, h: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Forward pass producing task-specific outputs.

        Args:
            h: Encoded features from backbone network (batch_size, hidden_dim)

        Returns:
            Task-specific predictions. Shape and structure depends on task:
            - Regression: (mu, sigma) where mu and sigma are (batch_size, 1)
            - Classification: logits (batch_size, num_classes)
        """
        pass

    @abstractmethod
    def output_size(self) -> int | tuple[int, ...]:
        """Return the output dimension(s) of this head.

        Returns:
            Output size. For single output (e.g., classification logits),
            returns int. For multiple outputs (e.g., regression mu and sigma),
            returns tuple of ints.
        """
        pass


class TaskLoss(ABC):
    """Abstract interface for task-specific loss functions.

    A TaskLoss computes the loss for a specific task given
    model predictions and ground truth targets.
    """

    @abstractmethod
    def __call__(
        self,
        predictions: torch.Tensor | tuple[torch.Tensor, ...],
        targets: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute task-specific loss.

        Args:
            predictions: Model predictions. Structure depends on task:
                - Regression: (mu, sigma) tensors
                - Classification: logits tensor
            targets: Ground truth targets
                - Regression: continuous values (batch_size, 1)
                - Classification: class indices (batch_size,)
            **kwargs: Additional task-specific parameters

        Returns:
            Scalar loss tensor
        """
        pass


class TaskMetrics(ABC):
    """Abstract interface for computing task-specific evaluation metrics.

    A TaskMetrics implementation computes all relevant metrics for
    evaluating model performance on a specific task.
    """

    @abstractmethod
    def compute(
        self,
        y_true: np.ndarray,
        predictions: np.ndarray | tuple[np.ndarray, ...],
        uncertainty: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute task-specific evaluation metrics.

        Args:
            y_true: Ground truth targets
            predictions: Model predictions (structure depends on task)
            uncertainty: Optional uncertainty estimates

        Returns:
            Dictionary mapping metric names to values
        """
        pass
