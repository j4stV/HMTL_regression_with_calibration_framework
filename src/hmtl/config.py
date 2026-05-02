"""Typed config for the high-level HMTL API.

``Config`` holds the user-facing knobs that AutoML users care about. It can be
built programmatically, loaded from the existing YAML files (``from_yaml``),
or resolved from a preset (see :mod:`src.hmtl.presets`).

The full set of advanced knobs (adversarial, CQR, MLflow, multi-aux, etc.)
remains accessible through the YAML path and via the internal dataclasses
``PreprocessConfig`` / ``TrainConfig`` / ``EnsembleConfig``.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class Config:
    """User-facing configuration."""

    # Task
    task_type: str = "regression"  # "regression" | "classification"
    target_column: Optional[str] = None

    # Model architecture
    hidden_width: int = 128
    depth_low: int = 12
    depth_high: int = 18
    alpha_dropout: float = 3e-4
    use_residual: bool = True

    # HMTL auxiliary task
    aux_enabled: bool = True
    aux_task: str = "contrastive"  # "bins" | "contrastive" | "reconstruction" | "rank" | "auto"
    aux_weight: float = 0.5
    n_bins: int = 5
    proj_dim: int = 50

    # Training
    lr: float = 3e-4
    epochs: int = 1000
    batch_size: int = 4096
    patience: int = 15
    early_stop_metric: str = "r_auc_mse"
    optimizer: str = "radam_lookahead"
    grad_clip_norm: Optional[float] = 1.0
    lr_scheduler: str = "cosine"
    amp_enabled: bool = True
    amp_dtype: str = "auto"
    seed: int = 42

    # Preprocessing
    impute_const: float = -1.0
    standardize: bool = True
    pca_enabled: bool = True
    pca_n_components: Optional[float] = None
    target_standardize: bool = True
    target_encoding_enabled: bool = False

    # Ensemble
    n_models: int = 10
    bagging: str = "stratified_kfold"

    # Conformal calibration
    coverage_levels: tuple = (0.80, 0.90, 0.95)
    conformal_method: str = "symmetric"  # "symmetric" | "cqr"

    # AutoML meta
    preset: Optional[str] = None  # record of which preset produced this config
    time_budget: Optional[int] = None  # seconds, used by best_quality HPO

    # Extra (e.g. adversarial, cqr_quantiles) — opaque passthrough
    extra: Dict[str, Any] = field(default_factory=dict)

    def copy(self, **overrides: Any) -> "Config":
        data = asdict(self)
        data.pop("extra", None)
        data.update({k: v for k, v in overrides.items() if k != "extra"})
        merged_extra = dict(self.extra)
        if "extra" in overrides:
            merged_extra.update(overrides["extra"])
        return Config(**data, extra=merged_extra)

    @classmethod
    def from_yaml(
        cls,
        data_yaml: str | Path | None = None,
        model_yaml: str | Path | None = None,
        train_yaml: str | Path | None = None,
        ensemble_yaml: str | Path | None = None,
        strict: bool = False,
    ) -> "Config":
        """Build Config from the legacy 4-YAML set.

        Unknown top-level keys are ignored with a warning unless ``strict=True``.
        """
        data = _load_yaml(data_yaml) or {}
        model = _load_yaml(model_yaml) or {}
        train = _load_yaml(train_yaml) or {}
        ensemble = _load_yaml(ensemble_yaml) or {}

        cfg = cls()

        # Data
        paths = data.get("paths", {})
        if "target" in paths:
            cfg.target_column = paths["target"]

        task_cfg = data.get("task", {}) or {}
        task_type = task_cfg.get("type") or task_cfg.get("task_type") or "regression"
        cfg.task_type = task_type

        pp = data.get("preprocess", {}) or {}
        cfg.impute_const = float(pp.get("impute_const", cfg.impute_const))
        cfg.standardize = bool(pp.get("standardize", cfg.standardize))
        pca = pp.get("pca", {}) or {}
        cfg.pca_enabled = bool(pca.get("enabled", cfg.pca_enabled))
        cfg.pca_n_components = pca.get("n_components", cfg.pca_n_components)
        cfg.target_standardize = bool(pp.get("target_standardize", cfg.target_standardize))
        cfg.target_encoding_enabled = bool(pp.get("target_encoding_enabled", cfg.target_encoding_enabled))

        # Model
        enc = model.get("encoder", {}) or {}
        cfg.hidden_width = int(enc.get("hidden_width", cfg.hidden_width))
        cfg.alpha_dropout = float(enc.get("alpha_dropout", cfg.alpha_dropout))
        cfg.use_residual = bool(enc.get("residual", cfg.use_residual))

        hmtl = model.get("hmtl", {}) or {}
        cfg.aux_enabled = bool(hmtl.get("enabled", cfg.aux_enabled))
        cfg.aux_task = str(hmtl.get("aux_task", cfg.aux_task))
        cfg.aux_weight = float(hmtl.get("lambda_aux", cfg.aux_weight))
        cfg.n_bins = int(hmtl.get("n_bins", cfg.n_bins))
        cfg.proj_dim = int(hmtl.get("proj_dim", cfg.proj_dim))
        cfg.depth_low = int(hmtl.get("low_layer", cfg.depth_low))
        cfg.depth_high = int(hmtl.get("high_layer", cfg.depth_high))

        # Train
        opt = train.get("optimizer", {}) or {}
        cfg.optimizer = str(opt.get("name", cfg.optimizer))
        cfg.lr = float(opt.get("lr", cfg.lr))
        cfg.grad_clip_norm = opt.get("grad_clip_norm", cfg.grad_clip_norm)
        if cfg.grad_clip_norm is not None:
            cfg.grad_clip_norm = float(cfg.grad_clip_norm)
        sched = opt.get("scheduler", {}) or {}
        cfg.lr_scheduler = str(sched.get("name", cfg.lr_scheduler))

        tr = train.get("training", {}) or {}
        cfg.epochs = int(tr.get("epochs", cfg.epochs))
        cfg.batch_size = int(tr.get("batch_size", cfg.batch_size))
        es = tr.get("early_stop", {}) or {}
        cfg.patience = int(es.get("patience", cfg.patience))
        cfg.early_stop_metric = str(es.get("metric", cfg.early_stop_metric))
        cfg.seed = int(tr.get("seed", cfg.seed))

        amp = tr.get("amp") or train.get("amp", {}) or {}
        cfg.amp_enabled = bool(amp.get("enabled", cfg.amp_enabled))
        cfg.amp_dtype = str(amp.get("dtype", cfg.amp_dtype))

        conformal = train.get("conformal", {}) or {}
        cfg.conformal_method = str(conformal.get("method", cfg.conformal_method))

        # Ensemble
        ens = ensemble.get("ensemble", {}) or {}
        cfg.n_models = int(ens.get("n_models", cfg.n_models))
        cfg.bagging = str(ens.get("bagging", cfg.bagging))

        # Stash untranslated keys so nothing is silently dropped
        cfg.extra = {
            "adversarial": tr.get("adversarial"),
            "cqr_quantiles": conformal.get("quantiles"),
            "cqr_weight": conformal.get("cqr_weight"),
            "multi_aux_tasks": hmtl.get("multi_aux_tasks"),
            "multi_aux_weights": hmtl.get("multi_aux_weights"),
            "auto_candidates": hmtl.get("auto_candidates"),
            "auto_pilot_epochs": hmtl.get("auto_pilot_epochs"),
            "train_csv": paths.get("train_csv"),
            "valid_csv": paths.get("valid_csv"),
            "cal_csv": paths.get("cal_csv"),
            "test_csv": paths.get("test_csv"),
        }
        # Drop None entries so ``extra`` stays tidy
        cfg.extra = {k: v for k, v in cfg.extra.items() if v is not None}

        if strict:
            _warn_unknown_keys(data, model, train, ensemble, raise_on_unknown=True)

        return cfg

    def to_yaml(self, path: str | Path) -> None:
        """Serialize to a single resolved YAML (not the legacy 4-file layout)."""
        out = asdict(self)
        out["coverage_levels"] = list(out["coverage_levels"])
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(out, f, sort_keys=False)


# ResolvedConfig is Config + the DataSummary that produced it. Kept as a thin
# marker type so downstream callers can distinguish "raw user config" from
# "config after preset + auto-detection have run".
@dataclass
class ResolvedConfig:
    config: Config
    summary: Optional["DataSummary"] = None  # forward-ref; imported lazily

    def as_config(self) -> Config:
        return self.config


def _load_yaml(path: str | Path | None) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_KNOWN_TOP_LEVEL_KEYS = {
    "data": {"paths", "task", "preprocess"},
    "model": {"encoder", "hmtl", "regression_head"},
    "train": {"optimizer", "amp", "training", "conformal", "logging"},
    "ensemble": {"ensemble"},
}


def _warn_unknown_keys(data, model, train, ensemble, raise_on_unknown: bool) -> None:
    mapping = {"data": data, "model": model, "train": train, "ensemble": ensemble}
    unknown: list[str] = []
    for name, d in mapping.items():
        if not isinstance(d, dict):
            continue
        known = _KNOWN_TOP_LEVEL_KEYS[name]
        for key in d.keys():
            if key not in known:
                unknown.append(f"{name}.{key}")
    if unknown:
        msg = f"Unknown YAML keys (ignored): {unknown}"
        if raise_on_unknown:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=3)
