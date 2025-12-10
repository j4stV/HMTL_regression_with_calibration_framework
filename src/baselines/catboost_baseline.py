"""CatBoost baseline with uncertainty estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from src.utils.logger import get_logger


class CatBoostBaseline:
    """CatBoost baseline with uncertainty estimation via ensemble."""
    
    def __init__(
        self,
        n_models: int = 10,
        iterations: int = 1000,
        learning_rate: float = 0.1,
        depth: int = 6,
        random_seed: int = 42,
    ) -> None:
        if not HAS_CATBOOST:
            raise ImportError("CatBoost is not installed. Install with: pip install catboost")
        
        self.n_models = n_models
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.random_seed = random_seed
        self.models = []
        self.logger = get_logger("baselines.catboost")
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> None:
        """Train ensemble of CatBoost models with proper uncertainty estimation."""
        self.logger.info(f"Training CatBoost ensemble with {self.n_models} models")
        
        self.models = []
        for i in range(self.n_models):
            model = CatBoostRegressor(
                iterations=self.iterations,
                learning_rate=self.learning_rate,
                depth=self.depth,
                random_seed=self.random_seed + i,
                verbose=False,
                loss_function="RMSEWithUncertainty",
                posterior_sampling=True,
            )
            
            # Convert to DataFrame for CatBoost
            X_df = pd.DataFrame(X, columns=[f"feature_{j}" for j in range(X.shape[1])])
            eval_set = None
            if X_val is not None and y_val is not None:
                X_val_df = pd.DataFrame(X_val, columns=X_df.columns)
                eval_set = (X_val_df, y_val)
            
            model.fit(
                X_df,
                y,
                eval_set=eval_set,
                use_best_model=True if eval_set is not None else False,
                verbose=False,
                early_stopping_rounds=50 if eval_set is not None else None,
            )
            self.models.append(model)
            
            if (i + 1) % 5 == 0:
                self.logger.info(f"Trained {i+1}/{self.n_models} models")
        
        self.logger.info("CatBoost ensemble training completed")
    
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predict with uncertainty using CatBoost's virtual ensembles.
        
        Returns:
            mu_mean: Mean prediction
            sigma_total: Total uncertainty (epistemic + aleatoric, unified with HMTL)
            sigma_epistemic: Epistemic uncertainty (knowledge uncertainty)
            sigma_aleatoric: Aleatoric uncertainty (data uncertainty)
        
        Uses CatBoost's virtual_ensembles_predict with TotalUncertainty to get:
        - mean values (preds[:,0])
        - knowledge uncertainty (preds[:,1]) - epistemic
        - data uncertainty (preds[:,2]) - aleatoric
        
        For fair comparison with HMTL, we combine uncertainties using:
        sigma_total = sqrt(knowledge² + data²)
        """
        if not self.models:
            raise ValueError("Models not trained. Call fit() first.")
        
        X_df = pd.DataFrame(X, columns=[f"feature_{j}" for j in range(X.shape[1])])
        
        # Collect predictions from all models
        all_means = []
        all_knowledge = []
        all_data = []
        
        for model in self.models:
            # Use virtual_ensembles_predict with TotalUncertainty
            # Returns [mean, knowledge_uncertainty, data_uncertainty] for each sample
            preds = model.virtual_ensembles_predict(
                X_df,
                prediction_type='TotalUncertainty',
                virtual_ensembles_count=self.n_models,
            )
            
            
            if preds.ndim == 2 and preds.shape[1] == 3:
                mean_preds = preds[:, 0]  # Mean values
                knowledge = preds[:, 1]   # Knowledge uncertainty (epistemic)
                data = preds[:, 2]        # Data uncertainty (aleatoric)
            else:
                # Fallback: try to extract from different format
                self.logger.warning(f"Unexpected prediction shape: {preds.shape}, using fallback")
                mean_preds = 0
                knowledge = 0
                data = 0
            
            all_means.append(mean_preds)
            all_knowledge.append(knowledge)
            all_data.append(data)
        
        # Stack predictions: (n_models, n_samples)
        means_array = np.stack(all_means, axis=0)
        knowledge_array = np.stack(all_knowledge, axis=0)
        data_array = np.stack(all_data, axis=0)
        
        # Ensemble aggregation: average across models
        mu_mean = np.mean(means_array, axis=0)  # (n_samples,)
        
        # Epistemic uncertainty: average knowledge uncertainty across models
        # Knowledge uncertainty from CatBoost is already in standard deviation form
        sigma_epistemic = np.mean(knowledge_array, axis=0)  # (n_samples,)
        
        # Also add epistemic uncertainty from ensemble diversity (variance of means)
        # This captures additional uncertainty from model disagreement
        mu_var_epistemic = np.var(means_array, axis=0, ddof=0)  # (n_samples,)
        sigma_epistemic_from_ensemble = np.sqrt(mu_var_epistemic)  # (n_samples,)
        
        # Combine both sources: CatBoost's knowledge uncertainty + ensemble diversity
        # Using variance addition: Var(X+Y) = Var(X) + Var(Y) for independent sources
        sigma_epistemic_combined = np.sqrt(
            sigma_epistemic ** 2 + sigma_epistemic_from_ensemble ** 2
        )
        
        # Aleatoric uncertainty: average data uncertainty across models
        # Data uncertainty is already in standard deviation form
        sigma_aleatoric = np.mean(data_array, axis=0)  # (n_samples,)
        
        # Total uncertainty: sqrt(epistemic² + aleatoric²)
        sigma_total_squared = sigma_epistemic_combined ** 2 + sigma_aleatoric ** 2
        sigma_total = np.sqrt(np.maximum(sigma_total_squared, 1e-10))  # Ensure non-negative
        
        # Log uncertainty statistics for debugging
        mean_epistemic = np.mean(sigma_epistemic_combined)
        mean_aleatoric = np.mean(sigma_aleatoric)
        mean_total = np.mean(sigma_total)
        
        self.logger.info(
            f"CatBoost uncertainty stats - "
            f"Epistemic (knowledge): mean={np.mean(sigma_epistemic):.6f}, "
            f"Epistemic (ensemble): mean={np.mean(sigma_epistemic_from_ensemble):.6f}, "
            f"Epistemic (combined): mean={mean_epistemic:.6f}, "
            f"Aleatoric: mean={mean_aleatoric:.6f}, "
            f"Total: mean={mean_total:.6f}"
        )
        
        # Additional debug logging
        self.logger.debug(
            f"CatBoost uncertainty details - "
            f"Epistemic: min={np.min(sigma_epistemic_combined):.6f}, max={np.max(sigma_epistemic_combined):.6f}, "
            f"Aleatoric: min={np.min(sigma_aleatoric):.6f}, max={np.max(sigma_aleatoric):.6f}, "
            f"Total: min={np.min(sigma_total):.6f}, max={np.max(sigma_total):.6f}"
        )
        
        return mu_mean, sigma_total, sigma_epistemic_combined, sigma_aleatoric

