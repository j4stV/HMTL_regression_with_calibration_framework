"""CatBoost baseline with uncertainty estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

try:
    from catboost import CatBoostClassifier, CatBoostError, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    CatBoostRegressor = None  # type: ignore[assignment]
    CatBoostClassifier = None  # type: ignore[assignment]
    CatBoostError = Exception

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
        compute_device: str = "auto",
        gpu_devices: str | None = None,
    ) -> None:
        if not HAS_CATBOOST:
            raise ImportError("CatBoost is not installed. Install with: pip install catboost")
        if compute_device not in {"auto", "cpu", "gpu"}:
            raise ValueError("compute_device must be one of: 'auto', 'cpu', 'gpu'")
        
        self.n_models = n_models
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.random_seed = random_seed
        self.compute_device = compute_device
        self.gpu_devices = gpu_devices
        self.models = []
        self.logger = get_logger("baselines.catboost")
        self._resolved_task_type: str | None = None

    def _candidate_task_types(self) -> list[str]:
        if self.compute_device == "gpu":
            return ["GPU"]
        if self.compute_device == "cpu":
            return ["CPU"]
        return ["GPU", "CPU"]

    def _build_model(self, task_type: str, random_seed: int):
        params = {
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "random_seed": random_seed,
            "verbose": False,
            "loss_function": "RMSEWithUncertainty",
            "posterior_sampling": True,
            "task_type": task_type,
        }
        if task_type == "GPU" and self.gpu_devices:
            params["devices"] = self.gpu_devices
        return CatBoostRegressor(**params)

    @staticmethod
    def _is_not_enough_trees_error(exc: Exception) -> bool:
        return "Not enough trees in model for" in str(exc)

    def _infer_virtual_ensembles_cap(self, model: Any) -> int:
        caps: list[int] = [max(1, int(self.n_models))]

        tree_count = getattr(model, "tree_count_", None)
        if tree_count is not None:
            try:
                tree_count_int = int(tree_count)
            except (TypeError, ValueError):
                tree_count_int = 0
            if tree_count_int > 0:
                caps.append(tree_count_int)

        get_tree_count = getattr(model, "get_tree_count", None)
        if callable(get_tree_count):
            try:
                method_tree_count = int(get_tree_count())
            except Exception:
                method_tree_count = 0
            if method_tree_count > 0:
                caps.append(method_tree_count)

        get_best_iteration = getattr(model, "get_best_iteration", None)
        if callable(get_best_iteration):
            try:
                best_iteration = int(get_best_iteration())
            except Exception:
                best_iteration = -1
            if best_iteration >= 0:
                caps.append(best_iteration + 1)

        return max(1, min(caps))

    def _predict_with_virtual_ensembles_backoff(self, model: Any, X_df: pd.DataFrame) -> np.ndarray:
        max_virtual_ensembles = self._infer_virtual_ensembles_cap(model)

        for virtual_ensembles_count in range(max_virtual_ensembles, 0, -1):
            try:
                return model.virtual_ensembles_predict(
                    X_df,
                    prediction_type="TotalUncertainty",
                    virtual_ensembles_count=virtual_ensembles_count,
                )
            except CatBoostError as exc:
                if self._is_not_enough_trees_error(exc) and virtual_ensembles_count > 1:
                    continue
                raise

        raise RuntimeError("Failed to produce CatBoost virtual ensemble predictions")
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> None:
        """Train ensemble of CatBoost models with proper uncertainty estimation."""
        self.logger.info(f"Training CatBoost ensemble with {self.n_models} models")
        
        self.models = []
        self._resolved_task_type = None
        for i in range(self.n_models):
            # Convert to DataFrame for CatBoost
            X_df = pd.DataFrame(X, columns=[f"feature_{j}" for j in range(X.shape[1])])
            eval_set = None
            if X_val is not None and y_val is not None:
                X_val_df = pd.DataFrame(X_val, columns=X_df.columns)
                eval_set = (X_val_df, y_val)

            task_candidates = (
                [self._resolved_task_type]
                if self._resolved_task_type is not None
                else self._candidate_task_types()
            )
            last_error: Exception | None = None

            for task_type in task_candidates:
                model = self._build_model(task_type=task_type, random_seed=self.random_seed + i)
                try:
                    model.fit(
                        X_df,
                        y,
                        eval_set=eval_set,
                        use_best_model=True if eval_set is not None else False,
                        verbose=False,
                        early_stopping_rounds=50 if eval_set is not None else None,
                    )
                    if self._resolved_task_type is None:
                        self._resolved_task_type = task_type
                        self.logger.info(f"CatBoost execution device resolved to {task_type}")
                    self.models.append(model)
                    break
                except CatBoostError as exc:
                    last_error = exc
                    if self.compute_device == "auto" and task_type == "GPU":
                        self.logger.warning(
                            f"CatBoost GPU setup failed, falling back to CPU: {exc}"
                        )
                        continue
                    raise
            else:
                if last_error is not None:
                    raise last_error
                raise RuntimeError("Failed to initialize CatBoost model")
            
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
        n_samples = X_df.shape[0]
        
        # Collect predictions from all models
        all_means = []
        all_knowledge = []
        all_data = []
        
        for model_idx, model in enumerate(self.models, start=1):
            try:
                preds = self._predict_with_virtual_ensembles_backoff(model, X_df)
            except CatBoostError as exc:
                if self._is_not_enough_trees_error(exc):
                    self.logger.warning(
                        "CatBoost model %d: virtual_ensembles_predict failed (%s). "
                        "Falling back to deterministic prediction with zero uncertainty.",
                        model_idx,
                        exc,
                    )
                    raw_preds = np.asarray(model.predict(X_df), dtype=float)
                    if raw_preds.ndim == 2 and raw_preds.shape[1] >= 1:
                        mean_preds = raw_preds[:, 0]
                    else:
                        mean_preds = raw_preds.reshape(-1)
                    knowledge = np.zeros(n_samples, dtype=float)
                    data = np.zeros(n_samples, dtype=float)
                    all_means.append(mean_preds)
                    all_knowledge.append(knowledge)
                    all_data.append(data)
                    continue
                raise

            if preds.ndim == 2 and preds.shape[1] == 3:
                mean_preds = preds[:, 0]
                knowledge = preds[:, 1]
                data = preds[:, 2]
            else:
                self.logger.warning("Unexpected prediction shape: %s, using zero uncertainty fallback", preds.shape)
                preds_arr = np.asarray(preds, dtype=float)
                if preds_arr.ndim == 1:
                    mean_preds = preds_arr
                elif preds_arr.ndim == 2 and preds_arr.shape[1] >= 1:
                    mean_preds = preds_arr[:, 0]
                else:
                    mean_preds = np.zeros(n_samples, dtype=float)
                knowledge = np.zeros(n_samples, dtype=float)
                data = np.zeros(n_samples, dtype=float)
            
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


def _entropy(probs: np.ndarray) -> np.ndarray:
    """Compute Shannon entropy per row: H(p) = -sum(p * log(p))."""
    probs_safe = np.clip(probs, 1e-12, 1.0)
    return -np.sum(probs_safe * np.log(probs_safe), axis=-1)


class CatBoostClassificationBaseline:
    """CatBoost classification baseline with entropy-based uncertainty."""

    def __init__(
        self,
        n_models: int = 10,
        iterations: int = 1000,
        learning_rate: float = 0.1,
        depth: int = 6,
        random_seed: int = 42,
        num_classes: int = 2,
        compute_device: str = "auto",
        gpu_devices: str | None = None,
    ) -> None:
        if not HAS_CATBOOST:
            raise ImportError("CatBoost is not installed. Install with: pip install catboost")
        if compute_device not in {"auto", "cpu", "gpu"}:
            raise ValueError("compute_device must be one of: 'auto', 'cpu', 'gpu'")

        self.n_models = n_models
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.random_seed = random_seed
        self.num_classes = num_classes
        self.compute_device = compute_device
        self.gpu_devices = gpu_devices
        self.models: list[Any] = []
        self.logger = get_logger("baselines.catboost_cls")
        self._resolved_task_type: str | None = None

    def _candidate_task_types(self) -> list[str]:
        if self.compute_device == "gpu":
            return ["GPU"]
        if self.compute_device == "cpu":
            return ["CPU"]
        return ["GPU", "CPU"]

    def _build_model(self, task_type: str, random_seed: int):
        loss = "Logloss" if self.num_classes == 2 else "MultiClass"
        params: dict[str, Any] = {
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "random_seed": random_seed,
            "verbose": False,
            "loss_function": loss,
            "task_type": task_type,
        }
        if task_type == "GPU" and self.gpu_devices:
            params["devices"] = self.gpu_devices
        return CatBoostClassifier(**params)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        self.logger.info("Training CatBoost classification ensemble with %d models", self.n_models)
        self.models = []
        self._resolved_task_type = None
        col_names = [f"feature_{j}" for j in range(X.shape[1])]

        for i in range(self.n_models):
            X_df = pd.DataFrame(X, columns=col_names)
            eval_set = None
            if X_val is not None and y_val is not None:
                X_val_df = pd.DataFrame(X_val, columns=col_names)
                eval_set = (X_val_df, y_val)

            task_candidates = (
                [self._resolved_task_type]
                if self._resolved_task_type is not None
                else self._candidate_task_types()
            )
            last_error: Exception | None = None

            for task_type in task_candidates:
                model = self._build_model(task_type=task_type, random_seed=self.random_seed + i)
                try:
                    model.fit(
                        X_df,
                        y,
                        eval_set=eval_set,
                        use_best_model=eval_set is not None,
                        verbose=False,
                        early_stopping_rounds=50 if eval_set is not None else None,
                    )
                    if self._resolved_task_type is None:
                        self._resolved_task_type = task_type
                        self.logger.info("CatBoost cls execution device resolved to %s", task_type)
                    self.models.append(model)
                    break
                except CatBoostError as exc:
                    last_error = exc
                    if self.compute_device == "auto" and task_type == "GPU":
                        self.logger.warning("CatBoost cls GPU failed, falling back to CPU: %s", exc)
                        continue
                    raise
            else:
                if last_error is not None:
                    raise last_error
                raise RuntimeError("Failed to initialise CatBoost classification model")

            if (i + 1) % 5 == 0:
                self.logger.info("Trained %d/%d classification models", i + 1, self.n_models)

        self.logger.info("CatBoost classification ensemble training completed")

    def predict(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predict class probabilities with entropy-based uncertainty.

        Returns:
            probs_mean: (n_samples, n_classes) averaged probabilities
            unc_total:  (n_samples,) H[E[p]]  (entropy of mean probs)
            unc_epi:    (n_samples,) MI = H[E[p]] - E[H[p]]
            unc_alea:   (n_samples,) E[H[p]]  (mean per-model entropy)
        """
        if not self.models:
            raise ValueError("Models not trained. Call fit() first.")

        X_df = pd.DataFrame(X, columns=[f"feature_{j}" for j in range(X.shape[1])])

        all_probs: list[np.ndarray] = []
        for model in self.models:
            probs = np.asarray(model.predict_proba(X_df), dtype=float)
            if probs.ndim == 1:
                probs = np.column_stack([1.0 - probs, probs])
            all_probs.append(probs)

        probs_stack = np.stack(all_probs, axis=0)  # (n_models, n_samples, n_classes)
        probs_mean = np.mean(probs_stack, axis=0)  # (n_samples, n_classes)

        # Entropy-based uncertainty decomposition (same as ensemble_predict_classification)
        unc_total = _entropy(probs_mean)  # H[E[p]]
        per_model_entropy = np.stack([_entropy(p) for p in all_probs], axis=0)
        unc_alea = np.mean(per_model_entropy, axis=0)  # E[H[p]]
        unc_epi = np.maximum(unc_total - unc_alea, 0.0)  # MI

        self.logger.info(
            "CatBoost cls uncertainty — total: %.4f, epi: %.4f, alea: %.4f",
            float(np.mean(unc_total)),
            float(np.mean(unc_epi)),
            float(np.mean(unc_alea)),
        )

        return probs_mean, unc_total, unc_epi, unc_alea
