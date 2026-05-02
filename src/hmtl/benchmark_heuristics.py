"""Size-adaptive heuristics lifted from ``scripts/run_automlbenchmark_experiment.py``.

That script contains the best-tuned heuristics in the repo — regime-dependent
batch sizes, sqrt-scaled learning rates, aux-task choice by feature count,
PCA policy for high-dim datasets, adversarial/CQR activation rules, etc.

This module mirrors them as a pure function so that
:class:`src.hmtl.estimator.HMTLRegressor` can opt into the same behavior
without depending on the benchmark script. The script itself is *not*
modified — it keeps its own private copy.

Public API::

    overrides = benchmark_size_overrides(n_rows, n_features,
                                         base_hidden_width=128,
                                         base_depth_low=12,
                                         base_depth_high=18,
                                         base_batch_size=4096,
                                         base_lr=3e-4,
                                         base_n_models=20)

Returned keys correspond to fields on :class:`src.hmtl.config.Config` and are
safe to pass as ``HMTLRegressor(..., **overrides)``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional


def _determine_regime(n_rows: int) -> str:
    """Match the benchmark's three-way split (tiny < 256 < small < 2048 ≤ large)."""
    if n_rows < 256:
        return "tiny"
    if n_rows < 2048:
        return "small"
    return "large"


def benchmark_size_overrides(
    n_rows: int,
    n_features: int,
    *,
    base_hidden_width: int = 128,
    base_depth_low: int = 12,
    base_depth_high: int = 18,
    base_batch_size: int = 4096,
    base_lr: float = 3e-4,
    base_lambda_aux: float = 0.5,
    base_n_models: int = 20,
    base_weight_decay: float = 0.0,
) -> Dict[str, Any]:
    """Return a Config-field override dict driven by dataset shape.

    The logic mirrors ``_build_effective_size_configs`` from
    ``run_automlbenchmark_experiment.py``. Returned keys line up with
    :class:`Config` fields so they can be fed directly to the estimator.

    Also stashes a ``_benchmark_policy`` key under ``extra`` so callers can
    inspect which branch was taken.
    """
    regime = _determine_regime(int(n_rows))

    # --- Batch size (per-regime divisor) ---
    batch_divisor_by_regime = {"tiny": 4, "small": 8, "large": 12}
    batch_divisor = batch_divisor_by_regime[regime]
    batch_size = int(min(base_batch_size, max(16, int(n_rows) // batch_divisor)))

    # --- LR scaling: gentle sqrt, clamped to [0.7, 1.5] ---
    reference_batch_size = 256
    lr_scale = math.sqrt(batch_size / reference_batch_size)
    lr_scale = max(0.7, min(lr_scale, 1.5))
    lr = float(base_lr * lr_scale)

    # --- Depth / width / aux task by regime (and n_features for large) ---
    hidden_width = base_hidden_width
    depth_low = base_depth_low
    depth_high = base_depth_high
    aux_task = "contrastive"
    aux_weight = base_lambda_aux
    weight_decay = float(base_weight_decay)

    if regime == "tiny":
        depth_high = max(1, min(base_depth_high, 6))
        depth_low = max(1, min(base_depth_low, 2, depth_high))
        hidden_width = min(base_hidden_width, 64)
        aux_task = "bins"
        aux_weight = min(base_lambda_aux, 0.20)
        weight_decay = max(weight_decay, 1e-4)
    elif regime == "small":
        depth_high = max(1, min(base_depth_high, 10))
        depth_low = max(1, min(base_depth_low, 4, depth_high))
        hidden_width = min(base_hidden_width, 96)
        aux_task = "bins"
        aux_weight = min(base_lambda_aux, 0.35)
        weight_decay = max(weight_decay, 5e-5)
    else:  # large
        if n_features <= 10:
            hidden_width = min(base_hidden_width, 64)
            depth_high, depth_low = 8, 3
            aux_task = "bins"
            aux_weight = min(base_lambda_aux, 0.25)
        elif n_features <= 20:
            hidden_width = min(base_hidden_width, 96)
            depth_high, depth_low = 10, 4
            aux_task = "bins"
            aux_weight = min(base_lambda_aux, 0.30)
        elif n_features <= 64:
            hidden_width = min(base_hidden_width, 128)
            depth_high, depth_low = 14, 6
            aux_task = "bins"
            aux_weight = min(base_lambda_aux, 0.35)
        elif n_features <= 256:
            hidden_width = base_hidden_width
            depth_high, depth_low = 16, 10
            aux_task = "bins"
            aux_weight = min(base_lambda_aux, 0.35)
        else:
            hidden_width = base_hidden_width
            depth_high, depth_low = base_depth_high, base_depth_low
            aux_task = "contrastive"
            aux_weight = base_lambda_aux

        # Extra weight-decay for high-dim large-regime datasets (p >> n guard).
        if n_features > 500:
            weight_decay = max(weight_decay, 1e-3)
        elif n_features > 100:
            weight_decay = max(weight_decay, 5e-4)

    # Enforce low_layer ≤ high_layer post-adjustment.
    depth_high = max(1, depth_high)
    depth_low = max(1, min(depth_low, depth_high))

    # --- Ensemble size / bagging ---
    if regime == "tiny":
        bagging = "stratified_bins"
        n_models = min(base_n_models, 8)
    elif regime == "small":
        bagging = "stratified_bins"
        n_models = min(base_n_models, 10)
    else:
        bagging = "stratified_kfold"
        n_models = min(base_n_models, 15)

    # --- PCA policy ---
    if n_features > 1000:
        pca_enabled = True
        max_components = max(50, int(n_rows) // 2)
        pca_n_components: Optional[float] = float(min(0.95, max_components))
        pca_policy = "enabled_extreme_dim"
    elif n_features > 500:
        pca_enabled = True
        pca_n_components = 0.99
        pca_policy = "enabled_high_dim"
    else:
        pca_enabled = False
        pca_n_components = None
        pca_policy = "disabled"

    # --- Conformal method (CQR off for tiny) ---
    if regime == "tiny":
        conformal_method = "symmetric"
        cqr_policy = "disabled_for_tiny"
    else:
        conformal_method = "cqr"
        cqr_policy = "cqr_light"

    # --- Auto aux on large regime; others keep the regime-chosen aux_task ---
    auto_aux_policy = "auto_selection" if regime == "large" else "hardcoded"
    auto_candidates = ["bins", "contrastive"] if regime == "large" else None
    auto_pilot_epochs = 40 if regime == "large" else None
    if auto_aux_policy == "auto_selection":
        aux_task = "auto"

    # --- Adversarial policy ---
    if regime == "tiny":
        adv_enabled = False
        adv_cfg: Optional[Dict[str, Any]] = {"enabled": False}
        adv_policy = "disabled_for_tiny"
    elif n_features > 500 or int(n_rows) < 5000:
        adv_enabled = True
        adv_cfg = {
            "enabled": True,
            "method": "fgsm",
            "epsilon": 0.002,
            "adv_weight": 0.05,
        }
        adv_policy = "fgsm_very_light"
    else:
        adv_enabled = True
        adv_cfg = {
            "enabled": True,
            "method": "fgsm",
            "epsilon": 0.005,
            "adv_weight": 0.15,
        }
        adv_policy = "fgsm_light"

    # --- Sigma regularization weight by regime ---
    sigma_reg_by_regime = {"tiny": 0.05, "small": 0.02, "large": 0.01}

    overrides: Dict[str, Any] = {
        "hidden_width": int(hidden_width),
        "depth_low": int(depth_low),
        "depth_high": int(depth_high),
        "aux_task": str(aux_task),
        "aux_weight": float(aux_weight),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "n_models": int(n_models),
        "bagging": str(bagging),
        "pca_enabled": bool(pca_enabled),
        "pca_n_components": pca_n_components,
        "conformal_method": str(conformal_method),
        # Benchmark-specific preferences stashed for introspection / passthrough.
        "extra": {
            "weight_decay": float(weight_decay),
            "sigma_reg_weight": float(sigma_reg_by_regime[regime]),
            "adversarial": adv_cfg,
            "auto_candidates": auto_candidates,
            "auto_pilot_epochs": auto_pilot_epochs,
            "_benchmark_policy": {
                "regime": regime,
                "batch_divisor": batch_divisor,
                "pca_policy": pca_policy,
                "cqr_policy": cqr_policy,
                "adversarial_policy": adv_policy,
                "auto_aux_policy": auto_aux_policy,
            },
        },
    }
    return overrides
