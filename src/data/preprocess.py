from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, KBinsDiscretizer, LabelEncoder, MinMaxScaler, StandardScaler

from src.data.target_encoding import RegressionOOFTargetEncoder
from src.utils.logger import get_logger, log_timing, log_config

_MISSING_CATEGORY_SENTINEL = "__missing_category__"


@dataclass
class PreprocessConfig:
    impute_const: float = -1.0
    use_dynamic_binning: bool = True
    quantile_binning_enabled: bool = False 
    quantile_binning_bins: int = 5
    standardize: bool = True
    pca_enabled: bool = True
    pca_n_components: float | int = None  # None = все компоненты
    target_standardize: bool = True
    target_encoding_enabled: bool = False
    target_encoding_n_splits: int = 5
    target_encoding_smoothing: float = 20.0


class TabularPreprocessor:
    def __init__(
        self,
        config: PreprocessConfig,
        feature_columns: Optional[list[str]] = None,
        target_column: Optional[str] = None,
        categorical_columns: Optional[list[str]] = None,
        task_type: str = "regression",  # NEW: "regression" or "classification"
    ) -> None:
        self.config = config
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.categorical_columns = categorical_columns or []
        self.task_type = task_type
        self.pipeline: Optional[Pipeline] = None
        self.target_mean_: Optional[float] = None
        self.target_std_: Optional[float] = None
        self.num_classes_: Optional[int] = None  # NEW: For classification tasks
        self.label_encoder_: Optional[LabelEncoder] = None  # Encode string targets to ints
        self.categorical_columns_: list[str] = []
        self.target_encoder_: Optional[RegressionOOFTargetEncoder] = None
        self.categorical_code_mappings_: dict[str, dict[object, float]] = {}

    @staticmethod
    def _to_float_feature_matrix(X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Convert feature matrix to float64, coercing non-numeric values to NaN."""
        if isinstance(X, pd.DataFrame):
            return X.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)

        arr = np.asarray(X)
        if arr.dtype.kind in "biufc":
            return arr.astype(np.float64, copy=False)

        return pd.DataFrame(arr).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)

    @staticmethod
    def _normalize_categorical_series(series: pd.Series) -> pd.Series:
        normalized = series.astype("object").copy()
        return normalized.where(normalized.notna(), _MISSING_CATEGORY_SENTINEL)

    def _fit_categorical_features(
        self,
        features: pd.DataFrame | np.ndarray,
        y: Optional[np.ndarray],
    ) -> pd.DataFrame | np.ndarray:
        self.target_encoder_ = None
        self.categorical_code_mappings_ = {}

        if not isinstance(features, pd.DataFrame):
            self.categorical_columns_ = []
            return features

        self.categorical_columns_ = [
            column for column in self.categorical_columns if column in features.columns
        ]
        if not self.categorical_columns_:
            return features.copy()

        if self.config.target_encoding_enabled and self.task_type == "regression" and y is not None:
            encoder = RegressionOOFTargetEncoder(
                n_splits=int(self.config.target_encoding_n_splits),
                smoothing=float(self.config.target_encoding_smoothing),
            )
            self.target_encoder_ = encoder
            return encoder.fit_transform(features, np.asarray(y, dtype=np.float64), self.categorical_columns_)

        transformed = features.copy()
        for column in self.categorical_columns_:
            normalized = self._normalize_categorical_series(transformed[column])
            unique_values = pd.unique(normalized)
            mapping = {
                value: float(idx)
                for idx, value in enumerate(unique_values.tolist())
            }
            self.categorical_code_mappings_[column] = mapping
            transformed[column] = normalized.map(mapping).astype(np.float64)

        return transformed

    def _transform_categorical_features(self, features: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        if not isinstance(features, pd.DataFrame):
            return features

        if not self.categorical_columns_:
            return features.copy()

        if self.target_encoder_ is not None:
            return self.target_encoder_.transform(features)

        transformed = features.copy()
        for column in self.categorical_columns_:
            if column not in transformed.columns:
                continue
            mapping = self.categorical_code_mappings_.get(column, {})
            normalized = self._normalize_categorical_series(transformed[column])
            unseen_code = float(len(mapping))
            transformed[column] = normalized.map(mapping).fillna(unseen_code).astype(np.float64)

        return transformed

    @staticmethod
    def _column_transformer_name(kind: str, col_idx: int) -> str:
        """Return a stable sklearn-safe transformer name.

        ColumnTransformer rejects names containing ``__``. Encoded feature labels
        may contain that separator, so internal transformer names must not depend
        on raw column labels.
        """
        return f"{kind}_{int(col_idx)}"

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        logger = get_logger("preprocess")
        
        with log_timing("Preprocessor fitting", logger):
            logger.info(f"Fitting preprocessor on data shape: {df.shape}")
            
            features = df.drop(columns=[self.target_column]) if self.target_column in df.columns else df.copy()
            X = features if self.feature_columns is None else features[self.feature_columns]
            y_for_encoding = (
                df[self.target_column].to_numpy(dtype=float)
                if self.target_column is not None and self.target_column in df.columns and self.task_type == "regression"
                else None
            )
            X_prepared = self._fit_categorical_features(X, y_for_encoding)
            colnames = (
                X_prepared.columns.tolist()
                if isinstance(X_prepared, pd.DataFrame)
                else [f"feature_{i}" for i in range(X_prepared.shape[1])]
            )
            X_values = self._to_float_feature_matrix(X_prepared)
            
            logger.debug(f"Feature matrix shape: {X_values.shape}")
            log_config(vars(self.config), logger, "Preprocessing config")

            steps: list[tuple[str, object]] = []
            
            # Step 1: Imputation
            steps.append(("impute", SimpleImputer(strategy="constant", fill_value=self.config.impute_const)))
            logger.debug(f"Added imputation step (fill_value={self.config.impute_const})")
            
            if self.config.use_dynamic_binning:
                # Dynamic binning approach
                # First apply MinMaxScaler and round to integers
                temp_pipeline = Pipeline(steps=[
                    ("impute", SimpleImputer(strategy="constant", fill_value=self.config.impute_const)),
                    ("minmax_scaler", MinMaxScaler())
                ])
                X_scaled = temp_pipeline.fit_transform(X_values)
                X_int = np.rint(X_scaled * 100000.0).astype(np.int32)
                
                # Analyze features
                binary_features = {}
                categorical_features = {}
                removed_features = []
                
                for col_idx in range(X_values.shape[1]):
                    values = set(X_int[:, col_idx].tolist())
                    if len(values) > 1:
                        if len(values) < 3:
                            binary_features[col_idx] = np.min(X_values[:, col_idx])
                        else:
                            categorical_features[col_idx] = len(values)
                    else:
                        removed_features.append(col_idx)
                
                if len(removed_features) > 0:
                    logger.info(f"Removing {len(removed_features)} constant features: {[colnames[i] for i in removed_features]}")
                
                logger.info(f"Found {len(categorical_features)} categorical features, {len(binary_features)} binary features")
                
                # Build transformers
                transformers = []
                for col_idx in categorical_features:
                    n_unique_values = categorical_features[col_idx]
                    n_bins = min(max(n_unique_values // 3, 3), 256)
                    strategy = 'quantile' if n_unique_values > 50 else 'kmeans'
                    transformers.append((
                        self._column_transformer_name("bucket", col_idx),
                        KBinsDiscretizer(
                            n_bins=n_bins,
                            encode='ordinal',
                            strategy=strategy
                        ),
                        (col_idx,)
                    ))
                    logger.debug(f"Column {col_idx} '{colnames[col_idx]}': {n_unique_values} unique values -> {n_bins} bins ({strategy})")
                
                for col_idx in binary_features:
                    transformers.append((
                        self._column_transformer_name("binary", col_idx),
                        Binarizer(threshold=0.0),
                        (col_idx,)
                    ))
                
                # Add MinMaxScaler
                steps.append(("minmax_scaler", MinMaxScaler()))
                
                # Add composite transformer if we have any
                if len(transformers) > 0:
                    steps.append(("composite_transformer", ColumnTransformer(
                        transformers=transformers,
                        sparse_threshold=0.0,
                        n_jobs=1
                    )))
                
                # Add VarianceThreshold to remove constant features
                steps.append(("selector", VarianceThreshold()))
                
            elif self.config.quantile_binning_enabled:
                # Old approach (deprecated)
                from sklearn.preprocessing import QuantileTransformer
                n_quantiles = min(1000, len(X_values))
                steps.append(("quantile_bin", QuantileTransformer(n_quantiles=n_quantiles, output_distribution="uniform")))
                logger.debug(f"Added quantile binning step (n_quantiles={n_quantiles})")
            
            if self.config.standardize:
                steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
                logger.debug("Added standardization step")
            
            if self.config.pca_enabled:
                pca_n_components = self.config.pca_n_components if self.config.pca_n_components is not None else None
                steps.append(("pca", PCA(n_components=pca_n_components, svd_solver="auto", random_state=42)))
                logger.debug(f"Added PCA step (n_components={pca_n_components})")

            self.pipeline = Pipeline(steps=steps)
            logger.info(f"Fitting pipeline with {len(steps)} steps")
            self.pipeline.fit(X_values)
            
            # Log PCA information if enabled
            if self.config.pca_enabled:
                pca_step = self.pipeline.named_steps.get("pca")
                if pca_step is not None:
                    n_components = pca_step.n_components_
                    explained_variance = float(np.sum(pca_step.explained_variance_ratio_))
                    logger.info(f"PCA: {n_components} components explain {explained_variance:.4%} of variance")
                    logger.debug(f"PCA explained variance ratios: {pca_step.explained_variance_ratio_[:10]}...")

            # Target preprocessing - different for regression vs classification
            if self.target_column is not None and self.target_column in df.columns:
                if self.task_type == "regression" and self.config.target_standardize:
                    # Regression: standardize target
                    y = df[self.target_column].to_numpy(dtype=float)
                    self.target_mean_ = float(np.mean(y))
                    std = float(np.std(y))
                    self.target_std_ = std if std > 1e-12 else 1.0
                    logger.info(f"Target standardization (regression): mean={self.target_mean_:.6f}, std={self.target_std_:.6f}")
                elif self.task_type == "classification":
                    # Classification: encode labels to 0-indexed integers
                    raw_y = df[self.target_column].to_numpy()
                    self.label_encoder_ = LabelEncoder()
                    y = self.label_encoder_.fit_transform(raw_y)
                    self.num_classes_ = len(self.label_encoder_.classes_)
                    logger.info(f"Classification task: {self.num_classes_} classes detected (labels: {self.label_encoder_.classes_.tolist()})")
                else:
                    logger.debug("Target standardization disabled")

        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        logger = get_logger("preprocess")
        assert self.pipeline is not None, "Preprocessor must be fit before transform."
        
        logger.debug(f"Transforming data shape: {df.shape}")
        features = df.drop(columns=[self.target_column]) if self.target_column in df.columns else df.copy()
        X = features if self.feature_columns is None else features[self.feature_columns]
        X_prepared = self._transform_categorical_features(X)
        X_values = self._to_float_feature_matrix(X_prepared)
        X_t = self.pipeline.transform(X_values)
        logger.info(f"Transformed feature matrix: {X.shape} -> {X_t.shape}")

        y_t: Optional[np.ndarray] = None
        if self.target_column is not None and self.target_column in df.columns:
            if self.task_type == "regression":
                y = df[self.target_column].to_numpy(dtype=float)
                if self.config.target_standardize and self.target_mean_ is not None and self.target_std_ is not None:
                    y_t = (y - self.target_mean_) / self.target_std_
                    logger.debug(f"Target standardized: mean={np.mean(y_t):.6f}, std={np.std(y_t):.6f}")
                else:
                    y_t = y
                    logger.debug(f"Target not standardized: mean={np.mean(y_t):.6f}, std={np.std(y_t):.6f}")
            elif self.task_type == "classification":
                # Classification: encode labels using fitted LabelEncoder
                raw_y = df[self.target_column].to_numpy()
                if self.label_encoder_ is not None:
                    y_t = self.label_encoder_.transform(raw_y)
                else:
                    y_t = raw_y.astype(int)
                logger.debug(f"Classification target: {len(y_t)} samples, {self.num_classes_} classes")
        
        return X_t, y_t

    def inverse_transform_target(self, y_standardized: np.ndarray) -> np.ndarray:
        """Transform target from standardized space back to original space.
        
        Args:
            y_standardized: Target values in standardized space
            
        Returns:
            Target values in original space
        """
        if self.config.target_standardize and self.target_mean_ is not None and self.target_std_ is not None:
            return y_standardized * self.target_std_ + self.target_mean_
        return y_standardized
    
    def inverse_transform_uncertainty(self, uncertainty_standardized: np.ndarray) -> np.ndarray:
        """Transform uncertainty from standardized space back to original space.
        
        For standard deviation: sigma_original = sigma_standardized * std
        
        Args:
            uncertainty_standardized: Uncertainty values in standardized space (standard deviation)
            
        Returns:
            Uncertainty values in original space
        """
        if self.config.target_standardize and self.target_std_ is not None:
            return uncertainty_standardized * self.target_std_
        return uncertainty_standardized
