from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


_MISSING_CATEGORY_SENTINEL = "_missing_category__"


@dataclass
class RegressionOOFTargetEncoder:
    n_splits: int = 5
    smoothing: float = 20.0
    random_state: int = 42
    global_mean_: float | None = field(default=None, init=False)
    categorical_columns_: list[str] = field(default_factory=list, init=False)
    mappings_: dict[str, dict[object, float]] = field(default_factory=dict, init=False)

    def _normalize_key_series(self, series: pd.Series) -> pd.Series:
        normalized = series.astype("object").copy()
        return normalized.where(normalized.notna(), _MISSING_CATEGORY_SENTINEL)

    def _fit_column_mapping(
        self,
        series: pd.Series,
        y: np.ndarray,
        global_mean: float,
    ) -> dict[object, float]:
        keys = self._normalize_key_series(series)
        frame = pd.DataFrame({"key": keys, "target": y}, index=series.index)
        grouped = frame.groupby("key", dropna=False)["target"].agg(["mean", "count"])
        smoothed = (
            grouped["count"] * grouped["mean"] + float(self.smoothing) * global_mean
        ) / (grouped["count"] + float(self.smoothing))
        return {key: float(value) for key, value in smoothed.items()}

    def _apply_mapping(
        self,
        series: pd.Series,
        mapping: dict[object, float],
        global_mean: float,
    ) -> pd.Series:
        keys = self._normalize_key_series(series)
        encoded = keys.map(mapping)
        return encoded.fillna(float(global_mean)).astype(np.float64)

    def fit_transform(
        self,
        df_train: pd.DataFrame,
        y_train: np.ndarray,
        categorical_columns: list[str],
    ) -> pd.DataFrame:
        working = df_train.copy()
        y_array = np.asarray(y_train, dtype=np.float64).reshape(-1)
        if len(working) != len(y_array):
            raise ValueError("df_train and y_train must have the same length")

        self.global_mean_ = float(np.mean(y_array)) if len(y_array) > 0 else 0.0
        self.categorical_columns_ = [
            column for column in categorical_columns if column in working.columns
        ]
        self.mappings_ = {}

        if not self.categorical_columns_:
            return working

        transformed: dict[str, pd.Series] = {}
        split_count = min(int(self.n_splits), len(working))
        use_oof = split_count >= 2
        splitter = (
            KFold(n_splits=split_count, shuffle=True, random_state=self.random_state)
            if use_oof
            else None
        )

        for column in working.columns:
            if column not in self.categorical_columns_:
                transformed[column] = working[column]
                continue

            mapping_full = self._fit_column_mapping(
                working[column],
                y_array,
                global_mean=float(self.global_mean_),
            )
            self.mappings_[column] = mapping_full
            encoded_name = f"{column}__te"

            if not use_oof:
                transformed[encoded_name] = pd.Series(
                    np.full(len(working), float(self.global_mean_), dtype=np.float64),
                    index=working.index,
                )
                continue

            encoded_values = pd.Series(
                np.full(len(working), float(self.global_mean_), dtype=np.float64),
                index=working.index,
            )
            for fit_idx, holdout_idx in splitter.split(working):
                fold_mapping = self._fit_column_mapping(
                    working.iloc[fit_idx][column],
                    y_array[fit_idx],
                    global_mean=float(self.global_mean_),
                )
                encoded_values.iloc[holdout_idx] = self._apply_mapping(
                    working.iloc[holdout_idx][column],
                    fold_mapping,
                    global_mean=float(self.global_mean_),
                ).to_numpy(dtype=np.float64)

            transformed[encoded_name] = encoded_values.astype(np.float64)

        return pd.DataFrame(transformed, index=working.index)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.global_mean_ is None:
            raise ValueError("Encoder must be fit before calling transform")

        working = df.copy()
        if not self.categorical_columns_:
            return working

        transformed: dict[str, pd.Series] = {}
        for column in working.columns:
            if column not in self.categorical_columns_:
                transformed[column] = working[column]
                continue

            transformed[f"{column}__te"] = self._apply_mapping(
                working[column],
                self.mappings_.get(column, {}),
                global_mean=float(self.global_mean_),
            )

        return pd.DataFrame(transformed, index=working.index)
