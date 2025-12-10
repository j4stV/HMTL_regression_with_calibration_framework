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
from sklearn.preprocessing import Binarizer, KBinsDiscretizer, MinMaxScaler, StandardScaler

from src.utils.logger import get_logger, log_timing, log_config


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


class TabularPreprocessor:
    def __init__(self, config: PreprocessConfig, feature_columns: Optional[list[str]] = None, target_column: Optional[str] = None) -> None:
        self.config = config
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.pipeline: Optional[Pipeline] = None
        self.target_mean_: Optional[float] = None
        self.target_std_: Optional[float] = None

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        logger = get_logger("preprocess")
        
        with log_timing("Preprocessor fitting", logger):
            logger.info(f"Fitting preprocessor on data shape: {df.shape}")
            
            features = df.drop(columns=[self.target_column]) if self.target_column in df.columns else df.copy()
            X = features if self.feature_columns is None else features[self.feature_columns]
            colnames = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]
            X_values = X.values if isinstance(X, pd.DataFrame) else X
            
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
                        colnames[col_idx],
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
                        colnames[col_idx],
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

            if self.target_column is not None and self.target_column in df.columns and self.config.target_standardize:
                y = df[self.target_column].to_numpy(dtype=float)
                self.target_mean_ = float(np.mean(y))
                std = float(np.std(y))
                self.target_std_ = std if std > 1e-12 else 1.0
                logger.info(f"Target standardization: mean={self.target_mean_:.6f}, std={self.target_std_:.6f}")
            elif self.target_column is not None and self.target_column in df.columns:
                logger.debug("Target standardization disabled")

        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        logger = get_logger("preprocess")
        assert self.pipeline is not None, "Preprocessor must be fit before transform."
        
        logger.debug(f"Transforming data shape: {df.shape}")
        features = df.drop(columns=[self.target_column]) if self.target_column in df.columns else df.copy()
        X = features if self.feature_columns is None else features[self.feature_columns]
        X_t = self.pipeline.transform(X.values)
        logger.info(f"Transformed feature matrix: {X.shape} -> {X_t.shape}")

        y_t: Optional[np.ndarray] = None
        if self.target_column is not None and self.target_column in df.columns:
            y = df[self.target_column].to_numpy(dtype=float)
            if self.config.target_standardize and self.target_mean_ is not None and self.target_std_ is not None:
                y_t = (y - self.target_mean_) / self.target_std_
                logger.debug(f"Target standardized: mean={np.mean(y_t):.6f}, std={np.std(y_t):.6f}")
            else:
                y_t = y
                logger.debug(f"Target not standardized: mean={np.mean(y_t):.6f}, std={np.std(y_t):.6f}")
        
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


