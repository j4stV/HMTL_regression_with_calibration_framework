"""Task abstraction layer for HMTL framework.

This module provides abstract interfaces for task-specific components,
allowing the framework to support multiple task types (regression, classification)
with minimal code duplication.
"""

from .base import TaskConfig, TaskHead, TaskLoss, TaskMetrics
from .regression import (
    RegressionTask,
    RegressionTaskConfig,
    RegressionTaskHead,
    RegressionTaskLoss,
    RegressionTaskMetrics,
)
from .classification import (
    ClassificationTask,
    ClassificationTaskConfig,
    ClassificationTaskHead,
    ClassificationTaskLoss,
    ClassificationTaskMetrics,
)

__all__ = [
    "TaskConfig",
    "TaskHead",
    "TaskLoss",
    "TaskMetrics",
    "RegressionTask",
    "RegressionTaskConfig",
    "RegressionTaskHead",
    "RegressionTaskLoss",
    "RegressionTaskMetrics",
    "ClassificationTask",
    "ClassificationTaskConfig",
    "ClassificationTaskHead",
    "ClassificationTaskLoss",
    "ClassificationTaskMetrics",
]
