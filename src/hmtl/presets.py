"""Tiered AutoML presets.

A preset is a function ``(DataSummary, user_overrides) → Config``. The order of
precedence is:

1. User overrides (explicit keyword args on ``HMTLRegressor``)
2. Size-adaptive overrides (from ``auto.size_adaptive_overrides``)
3. Preset defaults
4. Base ``Config`` defaults
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from src.hmtl.auto import DataSummary, size_adaptive_overrides
from src.hmtl.config import Config


_BASE_TASK_TYPE_KEY = "task_type"


def _fast_defaults(summary: DataSummary) -> Dict[str, Any]:
    return {
        "n_models": 3,
        "bagging": "full_dataset",
        "aux_task": "contrastive",
        "aux_enabled": True,
        "epochs": 200,
        "patience": 10,
    }


def _medium_defaults(summary: DataSummary) -> Dict[str, Any]:
    n_models = 5 if summary.size_class in {"tiny", "small"} else 10
    return {
        "n_models": n_models,
        "bagging": "stratified_kfold",
        "aux_task": "auto",  # triggers aux_selector pilot
        "aux_enabled": True,
        "epochs": 500,
        "patience": 15,
    }


def _best_quality_defaults(summary: DataSummary) -> Dict[str, Any]:
    n_models = 10 if summary.size_class in {"tiny", "small"} else 20
    return {
        "n_models": n_models,
        "bagging": "stratified_kfold",
        "aux_task": "auto",
        "aux_enabled": True,
        "epochs": 1000,
        "patience": 20,
    }


PRESETS: Dict[str, Callable[[DataSummary], Dict[str, Any]]] = {
    "fast": _fast_defaults,
    "medium": _medium_defaults,
    "best_quality": _best_quality_defaults,
}


def resolve_preset(
    name: str,
    summary: DataSummary,
    overrides: Dict[str, Any] | None = None,
) -> Config:
    """Resolve a preset into a concrete :class:`Config`.

    ``overrides`` are user-supplied values that win over both the preset and
    size-adaptive heuristics.
    """
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Known: {list(PRESETS)}")

    overrides = dict(overrides or {})
    preset_defaults = PRESETS[name](summary)
    size_overrides = size_adaptive_overrides(summary)
    # Pop non-Config keys from size overrides; preserved separately.
    size_overrides.pop("extra_class_imbalance", None)

    # Merge in the documented order: base → preset → size → user overrides.
    merged: Dict[str, Any] = {}
    merged.update(preset_defaults)
    merged.update(size_overrides)
    merged.update(overrides)

    merged["preset"] = name

    # Task type must come from user / summary — respect user override first.
    merged.setdefault(
        _BASE_TASK_TYPE_KEY,
        "regression" if summary.task_type == "regression" else "classification",
    )

    # Guard against unknown fields to avoid silent misconfig.
    config_fields = {f for f in Config.__dataclass_fields__}
    extras = {k: v for k, v in merged.items() if k not in config_fields}
    known = {k: v for k, v in merged.items() if k in config_fields}
    cfg = Config(**known)
    if extras:
        cfg.extra.update(extras)
    return cfg
