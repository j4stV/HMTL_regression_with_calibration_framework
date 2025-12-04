from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from src.utils.logger import get_logger, log_timing, log_config


@dataclass
class PreprocessConfig:
    impute_const: float = -1.0
    quantile_binning_enabled: bool = False
    quantile_binning_bins: int = 5
    standardize: bool = True
    pca_enabled: bool = True
    pca_n_components: float | int = 0.95
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
            
            logger.debug(f"Feature matrix shape: {X.shape}")
            log_config(vars(self.config), logger, "Preprocessing config")

            steps: list[tuple[str, object]] = []
            steps.append(("impute", SimpleImputer(strategy="constant", fill_value=self.config.impute_const)))
            logger.debug(f"Added imputation step (fill_value={self.config.impute_const})")
            
            if self.config.quantile_binning_enabled:
                n_quantiles = min(1000, len(X))
                steps.append(("quantile_bin", QuantileTransformer(n_quantiles=n_quantiles, output_distribution="uniform")))
                logger.debug(f"Added quantile binning step (n_quantiles={n_quantiles})")
            
            if self.config.standardize:
                steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
                logger.debug("Added standardization step")
            
            if self.config.pca_enabled:
                steps.append(("pca", PCA(n_components=self.config.pca_n_components, svd_solver="auto")))
                logger.debug(f"Added PCA step (n_components={self.config.pca_n_components})")

            self.pipeline = Pipeline(steps=steps)
            logger.info(f"Fitting pipeline with {len(steps)} steps")
            self.pipeline.fit(X.values)
            
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


