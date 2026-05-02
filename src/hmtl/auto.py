"""Data introspection + size-adaptive config resolution.

Heuristics here are intentionally simple and deterministic. They are lifted
from ``scripts/run_automlbenchmark_experiment.py`` (the only place where
size-adaptive logic has been battle-tested) so that downstream callers can
reuse them as a library function instead of re-implementing them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd


SIZE_CLASS_BOUNDARIES = [
    (1_000, "tiny"),
    (10_000, "small"),
    (100_000, "medium"),
    (1_000_000, "large"),
    (float("inf"), "xl"),
]


@dataclass
class DataSummary:
    """Summary of a dataset used to drive auto-config."""

    n_rows: int
    n_features: int
    size_class: str  # tiny | small | medium | large | xl
    task_type: str  # regression | binary | multiclass

    # Column typing (feature names)
    numeric_columns: list[str]
    binary_columns: list[str]
    low_card_cat_columns: list[str]
    high_card_cat_columns: list[str]

    # Target characteristics
    target_name: Optional[str]
    n_classes: Optional[int] = None
    class_imbalance_ratio: Optional[float] = None  # max_class / min_class
    target_skewness: Optional[float] = None
    missing_target: int = 0

    def is_classification(self) -> bool:
        return self.task_type in {"binary", "multiclass"}


def infer_size_class(n_rows: int) -> str:
    for upper, label in SIZE_CLASS_BOUNDARIES:
        if n_rows < upper:
            return label
    return "xl"


def infer_task_type(y: pd.Series | np.ndarray) -> str:
    """Return 'regression' | 'binary' | 'multiclass'."""
    arr = pd.Series(y).dropna()
    if arr.empty:
        return "regression"

    # Non-numeric targets are classification.
    if not pd.api.types.is_numeric_dtype(arr):
        n_unique = arr.nunique()
        return "binary" if n_unique <= 2 else "multiclass"

    # Numeric targets: treat tiny-cardinality integer-valued series as classification.
    is_integer_valued = np.all(np.equal(np.mod(arr.to_numpy(), 1), 0))
    n_unique = int(arr.nunique())
    n = len(arr)
    # Heuristic: <= 20 distinct values and integer-valued, or (n_unique / n) < 0.02 — classification.
    if is_integer_valued and (n_unique <= 20 or n_unique / max(n, 1) < 0.02):
        return "binary" if n_unique <= 2 else "multiclass"

    return "regression"


def _classify_column(series: pd.Series) -> str:
    """Classify as numeric | binary | low_card_cat | high_card_cat."""
    s = series.dropna()
    if s.empty:
        return "numeric"

    if pd.api.types.is_numeric_dtype(s):
        n_unique = int(s.nunique())
        if n_unique == 2:
            return "binary"
        return "numeric"

    # Non-numeric: categorical by nunique threshold.
    n_unique = int(s.nunique())
    if n_unique <= 2:
        return "binary"
    # High cardinality threshold scales softly with dataset size.
    threshold = max(50, int(0.02 * len(s)))
    return "high_card_cat" if n_unique > threshold else "low_card_cat"


def summarize_data(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray | None = None,
    target_name: Optional[str] = None,
) -> DataSummary:
    """Introspect ``X``/``y`` and return a :class:`DataSummary`."""
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    else:
        X = X.copy()

    numeric_columns: list[str] = []
    binary_columns: list[str] = []
    low_card_cat_columns: list[str] = []
    high_card_cat_columns: list[str] = []
    for col in X.columns:
        kind = _classify_column(X[col])
        if kind == "numeric":
            numeric_columns.append(col)
        elif kind == "binary":
            binary_columns.append(col)
        elif kind == "low_card_cat":
            low_card_cat_columns.append(col)
        else:
            high_card_cat_columns.append(col)

    n_rows, n_features = X.shape
    size_class = infer_size_class(n_rows)

    task_type = "regression"
    n_classes = None
    class_imbalance_ratio = None
    target_skewness = None
    missing_target = 0

    if y is not None:
        y_series = pd.Series(y)
        missing_target = int(y_series.isna().sum())
        task_type = infer_task_type(y_series)
        if task_type in {"binary", "multiclass"}:
            vc = y_series.dropna().value_counts()
            n_classes = int(len(vc))
            if n_classes >= 2 and vc.min() > 0:
                class_imbalance_ratio = float(vc.max() / vc.min())
        else:
            y_num = y_series.dropna().astype(float).to_numpy()
            if y_num.size >= 3:
                target_skewness = float(pd.Series(y_num).skew())

    return DataSummary(
        n_rows=n_rows,
        n_features=n_features,
        size_class=size_class,
        task_type=task_type,
        numeric_columns=numeric_columns,
        binary_columns=binary_columns,
        low_card_cat_columns=low_card_cat_columns,
        high_card_cat_columns=high_card_cat_columns,
        target_name=target_name,
        n_classes=n_classes,
        class_imbalance_ratio=class_imbalance_ratio,
        target_skewness=target_skewness,
        missing_target=missing_target,
    )


def size_adaptive_overrides(summary: DataSummary) -> dict[str, Any]:
    """Return overrides that adapt model / training to the dataset size.

    The values mirror the heuristics used in
    ``scripts/run_automlbenchmark_experiment.py`` — shallower/smaller for tiny
    data, wider/deeper ceilings for very large data.
    """
    size = summary.size_class
    overrides: dict[str, Any] = {}

    if size == "tiny":
        overrides.update(
            hidden_width=64,
            depth_low=6,
            depth_high=10,
            batch_size=128,
            patience=25,
            pca_enabled=False,
        )
    elif size == "small":
        overrides.update(
            hidden_width=96,
            depth_low=8,
            depth_high=14,
            batch_size=512,
            patience=20,
        )
    elif size == "medium":
        # Current defaults are tuned for this regime.
        overrides.update(
            hidden_width=128,
            depth_low=12,
            depth_high=18,
            batch_size=2048,
            patience=15,
        )
    elif size == "large":
        overrides.update(
            hidden_width=192,
            depth_low=14,
            depth_high=22,
            batch_size=4096,
            patience=10,
            pca_enabled=False,
        )
    else:  # xl
        overrides.update(
            hidden_width=256,
            depth_low=16,
            depth_high=24,
            batch_size=8192,
            patience=8,
            pca_enabled=False,
        )

    # Classification with heavy imbalance → prefer focal-like loss path later.
    if summary.is_classification() and summary.class_imbalance_ratio and summary.class_imbalance_ratio > 10:
        overrides["extra_class_imbalance"] = summary.class_imbalance_ratio

    # Skewed regression → stratified bagging is safer than k-fold.
    if (
        summary.task_type == "regression"
        and summary.target_skewness is not None
        and abs(summary.target_skewness) > 1.5
    ):
        overrides["bagging"] = "stratified_bins"

    return overrides
