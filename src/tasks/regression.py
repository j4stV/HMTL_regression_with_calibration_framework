"""Regression task implementation using existing components.

This module wraps the existing regression functionality (RegressionHead,
gaussian_nll, compute_regression_metrics) in the task abstraction layer.
This maintains full backward compatibility while enabling the framework
to support multiple task types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn

from src.losses.nll import gaussian_nll
from src.eval.metrics import compute_regression_metrics
from src.models.snn import RegressionHead
from .base import TaskConfig, TaskHead, TaskLoss, TaskMetrics


@dataclass
class RegressionTaskConfig(TaskConfig):
    """Configuration for regression tasks."""

    task_type: str = "regression"
    sigma_reg_weight: float = 0.0  # Weight for sigma regularization in loss


class RegressionTaskHead(TaskHead):
    """Adapter wrapping RegressionHead for the task interface.

    This is a thin wrapper around the existing RegressionHead that
    conforms to the TaskHead interface, maintaining full backward
    compatibility.
    """

    def __init__(self, in_dim: int, scale_coeff: float = 1.0):
        """Initialize regression head.

        Args:
            in_dim: Input feature dimension
            scale_coeff: Target standard deviation for sigma scaling
        """
        super().__init__()
        self.head = RegressionHead(in_dim, scale_coeff=scale_coeff)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (mu, sigma).

        Args:
            h: Encoded features (batch_size, hidden_dim)

        Returns:
            mu: Predicted mean (batch_size, 1)
            sigma: Predicted standard deviation (batch_size, 1)
        """
        return self.head(h)

    def output_size(self) -> Tuple[int, int]:
        """Return output dimensions: (1, 1) for mu and sigma."""
        return (1, 1)


class RegressionTaskLoss(TaskLoss):
    """Adapter wrapping gaussian_nll for the task interface."""

    def __init__(self, sigma_reg_weight: float = 0.0):
        """Initialize regression loss.

        Args:
            sigma_reg_weight: Weight for sigma regularization (prevents uncertainty explosion)
        """
        self.sigma_reg_weight = sigma_reg_weight

    def __call__(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute Gaussian negative log-likelihood loss.

        Args:
            predictions: (mu, sigma) tuple of tensors
            targets: Ground truth values (batch_size, 1)
            **kwargs: Additional parameters (ignored for backward compatibility)

        Returns:
            Scalar loss tensor
        """
        mu, sigma = predictions
        return gaussian_nll(mu, sigma, targets, sigma_reg_weight=self.sigma_reg_weight)


class RegressionTaskMetrics(TaskMetrics):
    """Adapter wrapping compute_regression_metrics for the task interface."""

    def compute(
        self,
        y_true: np.ndarray,
        predictions: Tuple[np.ndarray, np.ndarray],
        uncertainty: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute regression metrics (RMSE, MSE, MAE).

        Args:
            y_true: Ground truth values
            predictions: (mu, sigma) tuple - only mu is used for metrics
            uncertainty: Optional total uncertainty (not used directly)

        Returns:
            Dictionary with regression metrics
        """
        mu, sigma = predictions
        return compute_regression_metrics(y_true, mu, uncertainty)


class RegressionTask:
    """Factory for creating regression task components.

    This class provides static methods to create task-specific
    heads, losses, and metrics for regression.
    """

    @staticmethod
    def create_task_head(config: RegressionTaskConfig, in_dim: int, scale_coeff: float = 1.0) -> RegressionTaskHead:
        """Create a regression task head.

        Args:
            config: Regression task configuration
            in_dim: Input feature dimension
            scale_coeff: Target standard deviation for sigma scaling

        Returns:
            RegressionTaskHead instance
        """
        return RegressionTaskHead(in_dim, scale_coeff=scale_coeff)

    @staticmethod
    def create_loss(config: RegressionTaskConfig) -> RegressionTaskLoss:
        """Create a regression loss function.

        Args:
            config: Regression task configuration

        Returns:
            RegressionTaskLoss instance
        """
        return RegressionTaskLoss(sigma_reg_weight=config.sigma_reg_weight)

    @staticmethod
    def create_metrics(config: RegressionTaskConfig) -> RegressionTaskMetrics:
        """Create a regression metrics computer.

        Args:
            config: Regression task configuration

        Returns:
            RegressionTaskMetrics instance
        """
        return RegressionTaskMetrics()
