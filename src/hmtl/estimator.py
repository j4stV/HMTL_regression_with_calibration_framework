"""Sklearn-style estimators on top of the HMTL engine.

Wraps :mod:`src.train.ensemble`, :mod:`src.eval.evaluator`,
:mod:`src.eval.conformal`, :mod:`src.data.preprocess` into a clean
``fit``/``predict`` interface with presets, auto-detection, and persistence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from src.data.preprocess import PreprocessConfig, TabularPreprocessor
from src.eval.conformal import split_conformal_intervals
from src.eval.ensemble import ensemble_predict, ensemble_predict_classification
from src.hmtl.auto import DataSummary, summarize_data
from src.hmtl.config import Config
from src.hmtl.persistence import (
    FORMAT_VERSION,
    load_model as _load_bundle,
    save_model as _save_bundle,
    validate_input_schema,
)
from src.hmtl.presets import resolve_preset
from src.models.hmtl import HMTLModel
from src.train.ensemble import EnsembleConfig, fit_ensemble
from src.train.loop import TrainConfig
from src.utils.logger import get_logger


ArrayLike = Union[np.ndarray, pd.DataFrame]
TargetLike = Union[np.ndarray, pd.Series]


class _BaseEstimator:
    """Shared implementation for :class:`HMTLRegressor` and :class:`HMTLClassifier`."""

    _task_kind: str = "regression"  # "regression" | "classification"

    def __init__(
        self,
        preset: str = "medium",
        output_dir: Optional[str] = None,
        time_budget: Optional[int] = None,
        config: Optional[Config] = None,
        **overrides: Any,
    ) -> None:
        self.preset = preset
        self.output_dir = output_dir
        self.time_budget = time_budget
        self._initial_config = config  # optional: user-provided Config (e.g. from_yaml)
        self._overrides = dict(overrides)

        # Populated by fit()
        self._fitted: bool = False
        self.config_: Optional[Config] = None
        self.summary_: Optional[DataSummary] = None
        self.preprocessor_: Optional[TabularPreprocessor] = None
        self.models_: List[HMTLModel] = []
        self.conformal_q_: Dict[float, float] = {}
        self._feature_columns: List[str] = []
        self._feature_dtypes: Dict[str, str] = {}
        self._target_column: Optional[str] = None
        self._model_hparams: Dict[str, Any] = {}
        self._metrics: Dict[str, Any] = {}

    # ------------------------------------------------------------------ constructors

    @classmethod
    def from_yaml(
        cls,
        data_yaml: Optional[str] = None,
        model_yaml: Optional[str] = None,
        train_yaml: Optional[str] = None,
        ensemble_yaml: Optional[str] = None,
        **overrides: Any,
    ) -> "_BaseEstimator":
        cfg = Config.from_yaml(
            data_yaml=data_yaml,
            model_yaml=model_yaml,
            train_yaml=train_yaml,
            ensemble_yaml=ensemble_yaml,
        )
        return cls(preset="medium", config=cfg, **overrides)

    # ------------------------------------------------------------------ core API

    def fit(
        self,
        X: ArrayLike,
        y: Optional[TargetLike] = None,
        *,
        X_val: Optional[ArrayLike] = None,
        y_val: Optional[TargetLike] = None,
        target_column: Optional[str] = None,
        categorical_columns: Optional[List[str]] = None,
    ) -> "_BaseEstimator":
        logger = get_logger("hmtl.estimator")

        X_df, y_series, target_col = _coerce_frame(X, y, target_column)

        # Data summary + config resolution
        summary = summarize_data(X_df, y_series, target_name=target_col)

        if self._initial_config is not None:
            config = self._initial_config.copy(**self._overrides)
            config.preset = config.preset or self.preset
        else:
            config = resolve_preset(self.preset, summary, overrides=self._overrides)

        if self._task_kind == "regression":
            config.task_type = "regression"
        else:
            config.task_type = "classification"

        # Respect overrides for task-type (classifier forces classification etc.)
        logger.info(
            f"Resolved config: preset={config.preset}, n_models={config.n_models}, "
            f"bagging={config.bagging}, task_type={config.task_type}, "
            f"size_class={summary.size_class}"
        )

        # Preprocess
        pre_cfg = PreprocessConfig(
            impute_const=config.impute_const,
            standardize=config.standardize,
            pca_enabled=config.pca_enabled,
            pca_n_components=config.pca_n_components,
            target_standardize=config.target_standardize if config.task_type == "regression" else False,
            target_encoding_enabled=config.target_encoding_enabled,
        )

        # Form a DataFrame with the target column attached (preprocessor expects that).
        train_df = X_df.copy()
        train_df[target_col] = y_series.values

        pre = TabularPreprocessor(
            pre_cfg,
            target_column=target_col,
            categorical_columns=categorical_columns or _detect_categorical(X_df),
            task_type=config.task_type,
        ).fit(train_df)

        X_tr, y_tr = pre.transform(train_df)

        if X_val is not None and y_val is not None:
            val_df, y_val_series, _ = _coerce_frame(X_val, y_val, target_col)
            val_df = val_df.copy()
            val_df[target_col] = y_val_series.values
            X_va, y_va = pre.transform(val_df)
        else:
            # Simple 80/20 split off training if no validation supplied.
            rng = np.random.RandomState(config.seed)
            n = len(X_tr)
            idx = rng.permutation(n)
            split = int(0.8 * n)
            X_va, y_va = X_tr[idx[split:]], y_tr[idx[split:]]
            X_tr, y_tr = X_tr[idx[:split]], y_tr[idx[:split]]

        input_dim = X_tr.shape[1]
        scale_coeff = pre.target_std_ if pre.target_std_ is not None and pre.target_std_ > 1e-12 else 1.0

        # Build model factory
        task_head_factory, task_loss, task_metrics, num_classes = _build_task_components(
            config, summary, hidden_width=config.hidden_width
        )

        hparams = {
            "input_dim": input_dim,
            "hidden_width": config.hidden_width,
            "depth_low": config.depth_low,
            "depth_high": config.depth_high,
            "alpha_dropout": config.alpha_dropout,
            "n_bins": config.n_bins,
            "aux_weight": config.aux_weight,
            "enable_aux": config.aux_enabled,
            "aux_task": config.aux_task if config.aux_task != "auto" else "contrastive",
            "proj_dim": config.proj_dim,
            "scale_coeff": float(scale_coeff),
            "use_residual": config.use_residual,
            "num_classes": num_classes,
        }

        def build_model() -> HMTLModel:
            task_head = task_head_factory() if task_head_factory is not None else None
            return HMTLModel(
                input_dim=hparams["input_dim"],
                hidden_width=hparams["hidden_width"],
                depth_low=hparams["depth_low"],
                depth_high=hparams["depth_high"],
                alpha_dropout=hparams["alpha_dropout"],
                n_bins=hparams["n_bins"],
                aux_weight=hparams["aux_weight"],
                enable_aux=hparams["enable_aux"],
                aux_task=hparams["aux_task"],
                proj_dim=hparams["proj_dim"],
                scale_coeff=hparams["scale_coeff"],
                task_head=task_head,
                use_residual=hparams["use_residual"],
            )

        # Train config
        train_conf = TrainConfig(
            lr=config.lr,
            epochs=config.epochs,
            batch_size=config.batch_size,
            patience=config.patience,
            aux_weight=config.aux_weight,
            optimizer=config.optimizer if config.task_type == "regression" else "adamw",
            weight_decay=0.0,
            seed=config.seed,
            task_type=config.task_type,
            amp_enabled=config.amp_enabled,
            amp_dtype=config.amp_dtype,
            amp_eval_enabled=True,
            early_stop_metric=config.early_stop_metric if config.task_type == "regression" else "hybrid_rmse_rauc",
            grad_clip_norm=config.grad_clip_norm,
            lr_scheduler_name=config.lr_scheduler,
            lr_scheduler_eta_min_ratio=0.05,
            show_progress=False,
        )

        ens_conf = EnsembleConfig(
            n_models=config.n_models,
            bagging=config.bagging,
            show_progress=False,
        )

        # Optional aux-task auto-selection
        if config.aux_task == "auto":
            from src.train.aux_selector import select_best_aux_task

            def _build_for_selection(aux_task: str) -> HMTLModel:
                task_head = task_head_factory() if task_head_factory is not None else None
                return HMTLModel(
                    input_dim=hparams["input_dim"],
                    hidden_width=hparams["hidden_width"],
                    depth_low=hparams["depth_low"],
                    depth_high=hparams["depth_high"],
                    alpha_dropout=hparams["alpha_dropout"],
                    n_bins=hparams["n_bins"],
                    aux_weight=hparams["aux_weight"],
                    enable_aux=True,
                    aux_task=aux_task,
                    proj_dim=hparams["proj_dim"],
                    scale_coeff=hparams["scale_coeff"],
                    task_head=task_head,
                    use_residual=hparams["use_residual"],
                )

            chosen = select_best_aux_task(
                build_model_fn=_build_for_selection,
                X_tr=X_tr,
                y_tr=y_tr,
                X_va=X_va,
                y_va=y_va,
                n_bins=config.n_bins,
                train_cfg=train_conf,
                candidates=config.extra.get("auto_candidates", ["bins", "contrastive", "reconstruction", "rank"]),
                pilot_epochs=int(config.extra.get("auto_pilot_epochs", 30)),
            )
            logger.info(f"Auto-selected aux task: {chosen}")
            hparams["aux_task"] = chosen

        models, avg_score = fit_ensemble(
            build_model,
            X_tr,
            y_tr,
            X_va,
            y_va,
            n_bins=config.n_bins,
            ens_cfg=ens_conf,
            train_cfg=train_conf,
            task_loss=task_loss,
            task_metrics=task_metrics,
        )

        # Conformal calibration on validation set (regression only for now).
        conformal_q: Dict[float, float] = {}
        if config.task_type == "regression":
            mu_val, _, _, _ = ensemble_predict(
                models, X_va, amp_enabled=False
            )
            for cov in config.coverage_levels:
                alpha = 1.0 - float(cov)
                conformal_q[float(cov)] = split_conformal_intervals(y_va, mu_val, alpha=alpha)

        # Store state
        self.config_ = config
        self.summary_ = summary
        self.preprocessor_ = pre
        self.models_ = models
        self.conformal_q_ = conformal_q
        self._feature_columns = list(X_df.columns)
        self._feature_dtypes = {c: str(X_df[c].dtype) for c in X_df.columns}
        self._target_column = target_col
        self._model_hparams = hparams
        self._metrics = {"avg_score": float(avg_score)}
        self._fitted = True

        # Optional persistence
        if self.output_dir is not None:
            self.save(self.output_dir)

        return self

    # ------------------------------------------------------------------ inference

    def _preprocess_frame(self, X: ArrayLike) -> np.ndarray:
        self._check_fitted()
        df = _coerce_input_frame(X, self._feature_columns)
        # Preprocessor.transform expects a DataFrame that may include the target column.
        # Add a dummy target to satisfy signature; the preprocessor drops it.
        work = df.copy()
        if self._target_column not in work.columns:
            work[self._target_column] = 0.0
        X_arr, _ = self.preprocessor_.transform(work)
        return X_arr

    def predict(
        self, X: ArrayLike, return_uncertainty: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        self._check_fitted()
        X_arr = self._preprocess_frame(X)

        if self.config_.task_type == "classification":
            mean_probs, total, epi, alea, preds = ensemble_predict_classification(
                self.models_, X_arr, amp_enabled=False
            )
            if return_uncertainty:
                return preds, total
            return preds

        mu, sigma_total, _, _ = ensemble_predict(self.models_, X_arr, amp_enabled=False)
        mu_orig = self.preprocessor_.inverse_transform_target(mu)
        if return_uncertainty:
            sigma_orig = self.preprocessor_.inverse_transform_uncertainty(sigma_total)
            return mu_orig, sigma_orig
        return mu_orig

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Classification probabilities; not defined for regression."""
        self._check_fitted()
        if self.config_.task_type != "classification":
            raise NotImplementedError("predict_proba is only available for HMTLClassifier")
        X_arr = self._preprocess_frame(X)
        mean_probs, _, _, _, _ = ensemble_predict_classification(
            self.models_, X_arr, amp_enabled=False
        )
        return mean_probs

    def predict_interval(
        self, X: ArrayLike, coverage: float = 0.9
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._check_fitted()
        if self.config_.task_type != "regression":
            raise NotImplementedError("predict_interval requires a regression model")
        if not self.conformal_q_:
            raise RuntimeError(
                "Conformal quantiles were not computed. Refit with a validation set."
            )
        if float(coverage) not in self.conformal_q_:
            # Pick the closest available level.
            nearest = min(self.conformal_q_.keys(), key=lambda c: abs(c - coverage))
            q_std = self.conformal_q_[nearest]
            coverage_used = nearest
        else:
            q_std = self.conformal_q_[float(coverage)]
            coverage_used = float(coverage)

        X_arr = self._preprocess_frame(X)
        mu, _, _, _ = ensemble_predict(self.models_, X_arr, amp_enabled=False)
        lower_std = mu - q_std
        upper_std = mu + q_std
        lower = self.preprocessor_.inverse_transform_target(lower_std)
        upper = self.preprocessor_.inverse_transform_target(upper_std)
        return lower, upper

    # ------------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> Path:
        self._check_fitted()
        return _save_bundle(
            output_dir=path,
            config=self.config_,
            preprocessor=self.preprocessor_,
            models=self.models_,
            model_hparams=self._model_hparams,
            conformal_quantiles=self.conformal_q_,
            feature_columns=self._feature_columns,
            feature_dtypes=self._feature_dtypes,
            target_column=self._target_column,
            metrics=self._metrics,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "_BaseEstimator":
        bundle = _load_bundle(path, device=device)
        manifest = bundle["manifest"]
        task_type = manifest["task_type"]
        target_cls = HMTLClassifier if task_type == "classification" else HMTLRegressor
        inst = target_cls(preset=bundle["config"].preset or "medium")
        inst.config_ = bundle["config"]
        inst.preprocessor_ = bundle["preprocessor"]
        inst.models_ = bundle["models"]
        inst.conformal_q_ = bundle["conformal_quantiles"]
        inst._feature_columns = list(manifest["feature_columns"])
        inst._feature_dtypes = dict(manifest.get("feature_dtypes", {}))
        inst._target_column = manifest.get("target_column")
        inst._model_hparams = manifest.get("model_hparams", {})
        inst._metrics = manifest.get("metrics", {})
        inst._fitted = True
        return inst

    # ------------------------------------------------------------------ helpers

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} is not fitted. Call fit() or load() first.")


class HMTLRegressor(_BaseEstimator):
    """HMTL regressor with calibrated uncertainty intervals."""

    _task_kind = "regression"


class HMTLClassifier(_BaseEstimator):
    """HMTL classifier."""

    _task_kind = "classification"


def load(path: str | Path, device: str = "cpu") -> _BaseEstimator:
    """Load an estimator from ``path`` (regressor or classifier auto-detected)."""
    return _BaseEstimator.load(path, device=device)


# -------------------------------------------------------------------------- utils


def _coerce_frame(
    X: ArrayLike,
    y: Optional[TargetLike],
    target_column: Optional[str],
) -> tuple[pd.DataFrame, pd.Series, str]:
    """Return (X_df without target, y_series, target_column_name)."""
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    else:
        X_df = X.copy()

    target_col = target_column or "target"

    if y is None:
        # X must already contain the target column.
        if target_col not in X_df.columns:
            raise ValueError(
                "Must provide y, or pass a DataFrame that includes target_column."
            )
        y_series = X_df[target_col].copy()
        X_df = X_df.drop(columns=[target_col])
        return X_df, y_series, target_col

    if isinstance(y, pd.Series):
        y_series = y.copy()
        if target_col == "target" and y_series.name:
            target_col = str(y_series.name)
    else:
        y_series = pd.Series(np.asarray(y), name=target_col)
    # Ensure target column isn't also in X.
    if target_col in X_df.columns:
        X_df = X_df.drop(columns=[target_col])
    return X_df, y_series, target_col


def _coerce_input_frame(X: ArrayLike, expected_columns: List[str]) -> pd.DataFrame:
    if isinstance(X, np.ndarray):
        if X.shape[1] != len(expected_columns):
            raise ValueError(
                f"Input has {X.shape[1]} columns but model expects {len(expected_columns)}"
            )
        return pd.DataFrame(X, columns=expected_columns)
    missing = [c for c in expected_columns if c not in X.columns]
    if missing:
        raise ValueError(f"Input is missing expected columns: {missing}")
    return X[expected_columns].copy()


def _detect_categorical(df: pd.DataFrame) -> List[str]:
    cats: List[str] = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            cats.append(col)
    return cats


def _build_task_components(
    config: Config, summary: DataSummary, hidden_width: int
) -> tuple[Optional[Any], Optional[Any], Optional[Any], Optional[int]]:
    """Return (task_head_factory, task_loss, task_metrics, num_classes)."""
    if config.task_type == "regression":
        return None, None, None, None

    from src.tasks.classification import ClassificationTask, ClassificationTaskConfig

    num_classes = summary.n_classes or 2
    class_imbalance = summary.class_imbalance_ratio or 1.0
    use_focal = class_imbalance > 5

    task_cfg = ClassificationTaskConfig(
        task_type="classification",
        num_classes=num_classes,
        use_focal_loss=use_focal,
    )

    def _task_head_factory():
        return ClassificationTask.create_task_head(task_cfg, in_dim=hidden_width)

    task_loss = ClassificationTask.create_loss(task_cfg)
    task_metrics = ClassificationTask.create_metrics(task_cfg)
    return _task_head_factory, task_loss, task_metrics, num_classes
