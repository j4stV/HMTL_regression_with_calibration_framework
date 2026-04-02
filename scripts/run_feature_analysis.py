#!/usr/bin/env python3
"""Run feature analysis experiments: CQR, adversarial training, extended aux tasks.

This script automatically tests how new features affect regression metrics by running
a grid of feature configurations on datasets with ensemble training, multiple seeds,
and saves comprehensive JSON results.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger, setup_logging
from src.data.openml_loader import (
    load_dataset_bundle,
    prepare_dataset_splits,
    sample_train_data,
)
from src.data.preprocess import PreprocessConfig, TabularPreprocessor
from src.eval.evaluator import evaluate_on_dataset
from src.eval.metrics import EvaluationMetrics, evaluate_comprehensive
from src.models.hmtl import HMTLModel
from src.models.quantile_head import QuantileHead
from src.train.adversarial import AdversarialConfig
from src.train.ensemble import EnsembleConfig, fit_ensemble
from src.train.loop import TrainConfig


# ---------------------------------------------------------------------------
# Helpers copied from run_automlbenchmark_experiment.py (self-contained)
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _build_preprocess_config(data_cfg: dict[str, Any] | None) -> PreprocessConfig:
    preprocess_cfg = (data_cfg or {}).get("preprocess", {})
    return PreprocessConfig(
        impute_const=float(preprocess_cfg.get("impute_const", -1.0)),
        use_dynamic_binning=bool(preprocess_cfg.get("use_dynamic_binning", True)),
        quantile_binning_enabled=bool(
            preprocess_cfg.get("quantile_binning", {}).get("enabled", False)
        ),
        quantile_binning_bins=int(
            preprocess_cfg.get("quantile_binning", {}).get("bins", 5)
        ),
        standardize=bool(preprocess_cfg.get("standardize", True)),
        pca_enabled=bool(preprocess_cfg.get("pca", {}).get("enabled", True)),
        pca_n_components=preprocess_cfg.get("pca", {}).get("n_components", None),
        target_standardize=bool(preprocess_cfg.get("target_standardize", True)),
        target_encoding_enabled=bool(preprocess_cfg.get("target_encoding_enabled", True)),
        target_encoding_n_splits=int(preprocess_cfg.get("target_encoding_n_splits", 5)),
        target_encoding_smoothing=float(preprocess_cfg.get("target_encoding_smoothing", 20.0)),
    )


def _resolve_seed_list(base_seed: int, seeds: list[int] | None, n_seeds: int) -> list[int]:
    if seeds:
        return list(dict.fromkeys(seeds))
    return [base_seed + i for i in range(n_seeds)]


def _normalize_early_stop_metric_name(metric_name: str) -> str:
    normalized = str(metric_name).strip().lower()
    aliases = {
        "hybrid": "hybrid_rmse_rauc",
        "hybrid_rmse_rauc": "hybrid_rmse_rauc",
        "hybrid_rmse_r_auc": "hybrid_rmse_rauc",
        "rmse_plus_r_auc": "hybrid_rmse_rauc",
        "rmse_plus_rauc": "hybrid_rmse_rauc",
        "rmse": "rmse",
        "r_auc_mse": "r_auc_mse",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported regression early-stop metric '{metric_name}'. "
            "Expected one of: hybrid_rmse_rauc, rmse, r_auc_mse."
        )
    return aliases[normalized]


def _resolve_regression_early_stop_settings(
    train_cfg_yaml: dict[str, Any],
) -> tuple[str, float]:
    early_stop_cfg = train_cfg_yaml.get("training", {}).get("early_stop", {})
    metric = _normalize_early_stop_metric_name(
        early_stop_cfg.get("metric", "hybrid_rmse_rauc")
    )
    hybrid_weight = float(early_stop_cfg.get("hybrid_r_auc_weight", 0.25))
    return metric, hybrid_weight


def _resolve_amp_config(train_cfg_yaml: dict[str, Any]) -> dict[str, Any]:
    training_amp = train_cfg_yaml.get("training", {}).get("amp", {})
    root_amp = train_cfg_yaml.get("amp", {})
    amp_cfg = training_amp if isinstance(training_amp, dict) and training_amp else root_amp
    if not isinstance(amp_cfg, dict):
        amp_cfg = {}
    return {
        "enabled": bool(amp_cfg.get("enabled", True)),
        "dtype": str(amp_cfg.get("dtype", "auto")),
        "eval_enabled": bool(amp_cfg.get("eval_enabled", True)),
    }


def _clone_train_cfg_with_fp16_amp(train_cfg_yaml: dict[str, Any]) -> dict[str, Any]:
    cloned = copy.deepcopy(train_cfg_yaml)
    top_level_amp = cloned.setdefault("amp", {})
    if isinstance(top_level_amp, dict):
        top_level_amp["enabled"] = True
        top_level_amp["dtype"] = "fp16"
        top_level_amp["eval_enabled"] = True
    training_amp = cloned.setdefault("training", {}).setdefault("amp", {})
    if isinstance(training_amp, dict):
        training_amp["enabled"] = True
        training_amp["dtype"] = "fp16"
        training_amp["eval_enabled"] = True
    return cloned


def _clone_train_cfg_with_amp_disabled(train_cfg_yaml: dict[str, Any]) -> dict[str, Any]:
    cloned = copy.deepcopy(train_cfg_yaml)
    top_level_amp = cloned.setdefault("amp", {})
    if isinstance(top_level_amp, dict):
        top_level_amp["enabled"] = False
        top_level_amp["dtype"] = "fp16"
        top_level_amp["eval_enabled"] = False
    training_amp = cloned.setdefault("training", {}).setdefault("amp", {})
    if isinstance(training_amp, dict):
        training_amp["enabled"] = False
        training_amp["dtype"] = "fp16"
        training_amp["eval_enabled"] = False
    return cloned


def _is_bfloat16_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "bfloat16" in message or "scalartype bfloat16" in message


def _metrics_to_dict(metrics: EvaluationMetrics) -> dict[str, float]:
    return {
        "rmse": float(metrics.rmse),
        "mse": float(metrics.mse),
        "mae": float(metrics.mae),
        "r_auc_mse": float(metrics.r_auc_mse),
        "mean_uncertainty": float(metrics.mean_uncertainty),
        "mean_epistemic": float(metrics.mean_epistemic),
        "mean_aleatoric": float(metrics.mean_aleatoric),
    }


def _aggregate_metric_dicts(
    metric_dicts: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    if not metric_dicts:
        return {}
    metric_names = sorted(set().union(*(d.keys() for d in metric_dicts)))
    aggregated: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        values = [
            d[metric_name]
            for d in metric_dicts
            if metric_name in d and np.isfinite(d[metric_name])
        ]
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        aggregated[metric_name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=0)),
        }
    return aggregated


def prepare_preprocessed_splits_for_size(
    *,
    df_train_full,
    df_valid,
    df_test,
    target_column: str,
    categorical_columns: list[str] | None,
    preprocess_config: PreprocessConfig,
    size_ratio: float,
    seed: int,
) -> dict[str, Any]:
    """Sample train subset for a given size and fit a new preprocessor."""
    df_train_sampled = sample_train_data(df_train_full, size_ratio=size_ratio, seed=seed)
    preprocessor = TabularPreprocessor(
        preprocess_config,
        target_column=target_column,
        categorical_columns=categorical_columns,
        task_type="regression",
    ).fit(df_train_sampled)
    X_tr, y_tr = preprocessor.transform(df_train_sampled)
    X_va, y_va = preprocessor.transform(df_valid)
    X_te, y_te = preprocessor.transform(df_test)
    return {
        "preprocessor": preprocessor,
        "n_train_samples": int(len(df_train_sampled)),
        "X_tr": X_tr,
        "y_tr": y_tr,
        "X_va": X_va,
        "y_va": y_va,
        "X_te": X_te,
        "y_te": y_te,
    }


# ---------------------------------------------------------------------------
# FeatureConfig and feature group definitions
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """A single feature configuration to evaluate."""
    name: str
    description: str
    train_overrides: dict[str, Any] = field(default_factory=dict)
    model_overrides: dict[str, Any] = field(default_factory=dict)


def build_feature_configs(feature_groups: list[str]) -> list[FeatureConfig]:
    """Assemble FeatureConfig list based on requested feature groups.

    Always includes the baseline config as the first entry.
    """
    configs: list[FeatureConfig] = []
    seen_names: set[str] = set()

    def _add(cfg: FeatureConfig) -> None:
        if cfg.name not in seen_names:
            seen_names.add(cfg.name)
            configs.append(cfg)

    # Always include baseline
    _add(FeatureConfig(
        name="baseline",
        description="Standard HMTL with bins aux",
        train_overrides={"cqr_enabled": False},
        model_overrides={"hmtl": {"aux_task": "bins"}},
    ))

    groups = set(feature_groups)
    if "all" in groups:
        groups = {"baseline", "cqr", "adversarial", "aux_tasks", "combined"}

    # CQR configs
    if "cqr" in groups:
        for w in [0.25, 0.5, 0.75]:
            _add(FeatureConfig(
                name=f"cqr_{w}",
                description=f"CQR calibration with cqr_weight={w}",
                train_overrides={
                    "cqr_enabled": True,
                    "cqr_quantiles": [0.05, 0.95],
                    "cqr_weight": w,
                },
                model_overrides={},
            ))

    # Adversarial configs
    if "adversarial" in groups:
        for eps in [0.005, 0.01, 0.02]:
            _add(FeatureConfig(
                name=f"fgsm_eps{eps}",
                description=f"FGSM adversarial training, epsilon={eps}",
                train_overrides={
                    "adversarial": {
                        "enabled": True,
                        "method": "fgsm",
                        "epsilon": eps,
                    },
                },
                model_overrides={},
            ))
        for eps in [0.005, 0.01, 0.02]:
            _add(FeatureConfig(
                name=f"pgd_eps{eps}",
                description=f"PGD adversarial training, epsilon={eps}",
                train_overrides={
                    "adversarial": {
                        "enabled": True,
                        "method": "pgd",
                        "epsilon": eps,
                        "alpha": 0.005,
                        "pgd_steps": 3,
                    },
                },
                model_overrides={},
            ))

    # Aux task configs
    if "aux_tasks" in groups:
        for aux in ["bins", "contrastive", "reconstruction", "rank"]:
            _add(FeatureConfig(
                name=f"aux_{aux}",
                description=f"Auxiliary task: {aux}",
                train_overrides={},
                model_overrides={"hmtl": {"aux_task": aux}},
            ))

    # Combined configs
    if "combined" in groups:
        _add(FeatureConfig(
            name="cqr0.5_fgsm0.01",
            description="CQR(0.5) + FGSM(0.01)",
            train_overrides={
                "cqr_enabled": True,
                "cqr_quantiles": [0.05, 0.95],
                "cqr_weight": 0.5,
                "adversarial": {
                    "enabled": True,
                    "method": "fgsm",
                    "epsilon": 0.01,
                },
            },
            model_overrides={},
        ))
        for aux in ["contrastive", "reconstruction", "rank"]:
            _add(FeatureConfig(
                name=f"cqr0.5_{aux}",
                description=f"CQR(0.5) + aux={aux}",
                train_overrides={
                    "cqr_enabled": True,
                    "cqr_quantiles": [0.05, 0.95],
                    "cqr_weight": 0.5,
                },
                model_overrides={"hmtl": {"aux_task": aux}},
            ))

    return configs


# ---------------------------------------------------------------------------
# Config application helpers
# ---------------------------------------------------------------------------

def apply_feature_config(
    base_model_cfg: dict[str, Any],
    base_train_cfg_yaml: dict[str, Any],
    feature_config: FeatureConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Deep-copy base configs and apply feature overrides.

    Returns:
        (model_cfg, train_cfg_yaml) with overrides applied.
    """
    model_cfg = copy.deepcopy(base_model_cfg)
    train_cfg_yaml = copy.deepcopy(base_train_cfg_yaml)

    # Apply model overrides (deep merge)
    for key, value in feature_config.model_overrides.items():
        if isinstance(value, dict) and key in model_cfg and isinstance(model_cfg[key], dict):
            model_cfg[key].update(value)
        else:
            model_cfg[key] = value

    # Train overrides are stored and applied later when building TrainConfig.
    # We stash them in train_cfg_yaml under a private key for retrieval.
    train_cfg_yaml["_feature_train_overrides"] = copy.deepcopy(feature_config.train_overrides)

    return model_cfg, train_cfg_yaml


def _build_model_with_features(
    input_dim: int,
    model_cfg: dict[str, Any],
    scale_coeff: float,
    cqr_enabled: bool,
    cqr_quantiles: list[float] | None,
):
    """Create a build_model_fn closure, optionally with a QuantileHead for CQR."""
    hidden_width = int(model_cfg["encoder"]["hidden_width"])
    use_residual = bool(model_cfg["encoder"].get("residual", True))

    def build_model() -> HMTLModel:
        quantile_head = None
        if cqr_enabled:
            q = cqr_quantiles if cqr_quantiles else [0.05, 0.95]
            quantile_head = QuantileHead(hidden_width, quantiles=q)

        return HMTLModel(
            input_dim=input_dim,
            hidden_width=hidden_width,
            depth_low=int(model_cfg["hmtl"]["low_layer"]),
            depth_high=int(model_cfg["hmtl"]["high_layer"]),
            alpha_dropout=float(model_cfg["encoder"]["alpha_dropout"]),
            n_bins=int(model_cfg["hmtl"]["n_bins"]),
            aux_weight=float(model_cfg["hmtl"]["lambda_aux"]),
            enable_aux=bool(model_cfg["hmtl"].get("enabled", True)),
            aux_task=str(model_cfg["hmtl"].get("aux_task", "contrastive")),
            proj_dim=int(model_cfg["hmtl"].get("proj_dim", 50)),
            scale_coeff=scale_coeff,
            use_residual=use_residual,
            quantile_head=quantile_head,
        )

    return build_model


def _build_train_config(
    model_cfg: dict[str, Any],
    train_cfg_yaml: dict[str, Any],
    seed: int,
    show_progress: bool,
) -> TrainConfig:
    """Build TrainConfig from YAML config, applying feature overrides."""
    early_stop_metric, hybrid_weight = _resolve_regression_early_stop_settings(train_cfg_yaml)
    amp_cfg = _resolve_amp_config(train_cfg_yaml)
    optimizer_cfg = train_cfg_yaml["optimizer"]
    scheduler_cfg = optimizer_cfg.get("scheduler", {})
    grad_clip_raw = optimizer_cfg.get("grad_clip_norm", 1.0)

    # Retrieve feature overrides stashed by apply_feature_config
    overrides = train_cfg_yaml.pop("_feature_train_overrides", {})

    # Base values
    cqr_enabled = bool(overrides.get("cqr_enabled", False))
    cqr_quantiles = overrides.get("cqr_quantiles", None)
    cqr_weight = float(overrides.get("cqr_weight", 0.5))

    adv_raw = overrides.get("adversarial", None)
    adversarial: AdversarialConfig | None = None
    if isinstance(adv_raw, dict) and adv_raw.get("enabled", False):
        adversarial = AdversarialConfig(
            enabled=True,
            method=str(adv_raw.get("method", "fgsm")),
            epsilon=float(adv_raw.get("epsilon", 0.01)),
            alpha=float(adv_raw.get("alpha", 0.005)),
            pgd_steps=int(adv_raw.get("pgd_steps", 3)),
            adv_weight=float(adv_raw.get("adv_weight", 0.5)),
        )

    return TrainConfig(
        lr=float(optimizer_cfg["lr"]),
        epochs=int(train_cfg_yaml["training"]["epochs"]),
        batch_size=int(train_cfg_yaml["training"]["batch_size"]),
        patience=int(
            train_cfg_yaml["training"].get("early_stop", {}).get("patience", 10)
        ),
        aux_weight=float(model_cfg["hmtl"]["lambda_aux"]),
        optimizer=str(optimizer_cfg.get("name", "radam_lookahead")),
        lookahead_k=int(optimizer_cfg.get("lookahead_sync_period", 6)),
        lookahead_alpha=float(optimizer_cfg.get("lookahead_slow_step", 0.5)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
        sigma_reg_weight=float(
            train_cfg_yaml["training"].get("sigma_reg_weight", 0.0)
        ),
        seed=seed,
        task_type="regression",
        show_progress=show_progress,
        amp_enabled=bool(amp_cfg["enabled"]),
        amp_dtype=str(amp_cfg["dtype"]),
        amp_eval_enabled=bool(amp_cfg["eval_enabled"]),
        early_stop_metric=early_stop_metric,
        hybrid_r_auc_weight=hybrid_weight,
        grad_clip_norm=None if grad_clip_raw is None else float(grad_clip_raw),
        lr_scheduler_name=str(scheduler_cfg.get("name", "none")),
        lr_scheduler_eta_min_ratio=float(scheduler_cfg.get("eta_min_ratio", 0.05)),
        cqr_enabled=cqr_enabled,
        cqr_quantiles=cqr_quantiles,
        cqr_weight=cqr_weight,
        adversarial=adversarial,
    )


# ---------------------------------------------------------------------------
# Core experiment logic
# ---------------------------------------------------------------------------

def run_single_config_seed(
    feature_config: FeatureConfig,
    seed: int,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    preprocessor: TabularPreprocessor,
    model_cfg: dict[str, Any],
    train_cfg_yaml: dict[str, Any],
    ensemble_cfg_yaml: dict[str, Any],
    show_progress: bool = False,
) -> dict[str, Any]:
    """Run a single feature config with a single seed. Returns metrics dict."""
    logger = get_logger("feature_analysis")

    # Apply feature overrides
    eff_model_cfg, eff_train_cfg_yaml = apply_feature_config(
        model_cfg, train_cfg_yaml, feature_config
    )

    # Build TrainConfig
    train_cfg = _build_train_config(eff_model_cfg, eff_train_cfg_yaml, seed, show_progress)

    # Ensemble config
    ens_block = ensemble_cfg_yaml.get("ensemble", {})
    ens_cfg = EnsembleConfig(
        n_models=int(ens_block.get("n_models", 5)),
        bagging=str(ens_block.get("bagging", "stratified_bins")),
        show_progress=show_progress,
    )

    input_dim = X_tr.shape[1]
    scale_coeff = (
        float(preprocessor.target_std_)
        if preprocessor.target_std_ is not None and preprocessor.target_std_ > 1e-12
        else 1.0
    )

    build_model = _build_model_with_features(
        input_dim=input_dim,
        model_cfg=eff_model_cfg,
        scale_coeff=scale_coeff,
        cqr_enabled=train_cfg.cqr_enabled,
        cqr_quantiles=train_cfg.cqr_quantiles,
    )

    # Fit ensemble with AMP error handling
    def _do_fit(tcfg_yaml: dict[str, Any] | None = None) -> tuple:
        nonlocal train_cfg
        if tcfg_yaml is not None:
            # rebuild train_cfg with patched AMP
            eff_mc, eff_tc = apply_feature_config(
                model_cfg, tcfg_yaml, feature_config
            )
            train_cfg = _build_train_config(eff_mc, eff_tc, seed, show_progress)
        models, avg_score = fit_ensemble(
            build_model_fn=build_model,
            X_tr=X_tr,
            y_tr=y_tr,
            X_va=X_va,
            y_va=y_va,
            n_bins=int(eff_model_cfg["hmtl"]["n_bins"]),
            ens_cfg=ens_cfg,
            train_cfg=train_cfg,
        )
        return models, avg_score

    try:
        models, avg_score = _do_fit()
    except Exception as exc:
        if not _is_bfloat16_error(exc):
            raise
        logger.warning(
            "BFloat16 error for config=%s seed=%d. Retrying with fp16.",
            feature_config.name,
            seed,
        )
        try:
            models, avg_score = _do_fit(_clone_train_cfg_with_fp16_amp(train_cfg_yaml))
        except Exception as exc2:
            if not _is_bfloat16_error(exc2):
                raise
            logger.warning(
                "FP16 also failed for config=%s seed=%d. Disabling AMP.",
                feature_config.name,
                seed,
            )
            models, avg_score = _do_fit(_clone_train_cfg_with_amp_disabled(train_cfg_yaml))

    # Evaluate
    conformal_method = "cqr" if train_cfg.cqr_enabled else "symmetric"
    eval_results = evaluate_on_dataset(
        models=models,
        X=X_te,
        y_true=y_te,
        X_cal=X_va,
        y_cal=y_va,
        coverage_levels=[0.80, 0.90, 0.95],
        preprocessor=preprocessor,
        use_normalized_metrics=True,
        conformal_method=conformal_method,
    )

    # Extract metrics
    result = _metrics_to_dict(eval_results.metrics)
    result["ensemble_avg_val_score"] = float(avg_score)

    # Standard conformal metrics (coverage/width at 90%)
    if eval_results.pi_metrics_after:
        pi90 = eval_results.pi_metrics_after.get(0.90)
        if pi90:
            result["conformal_coverage_90"] = float(pi90.get("coverage", float("nan")))
            result["conformal_width_90"] = float(pi90.get("mean_width", float("nan")))

    # CQR-specific metrics
    if eval_results.pi_metrics_cqr:
        pi_cqr90 = eval_results.pi_metrics_cqr.get(0.90)
        if pi_cqr90:
            result["cqr_coverage_90"] = float(pi_cqr90.get("coverage", float("nan")))
            result["cqr_width_90"] = float(pi_cqr90.get("mean_width", float("nan")))

    logger.info(
        "Config=%s seed=%d  rmse=%.6f r_auc_mse=%.6f",
        feature_config.name,
        seed,
        result["rmse"],
        result["r_auc_mse"],
    )
    return result


def _build_comparison_table(
    config_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build comparison table from a list of per-config aggregated results.

    Each entry in *config_results* should have keys ``name`` and ``aggregated``.
    """
    config_names = [cr["name"] for cr in config_results]
    all_metric_names: set[str] = set()
    for cr in config_results:
        all_metric_names.update(cr["aggregated"].keys())
    all_metric_names_sorted = sorted(all_metric_names)

    metrics_table: dict[str, list[dict[str, float]]] = {}
    for metric in all_metric_names_sorted:
        metrics_table[metric] = []
        for cr in config_results:
            agg = cr["aggregated"].get(metric, {"mean": float("nan"), "std": float("nan")})
            metrics_table[metric].append(agg)

    # Identify best config per metric (lower is better for all regression metrics)
    best_config: dict[str, str] = {}
    for metric in all_metric_names_sorted:
        means = [
            cr["aggregated"].get(metric, {}).get("mean", float("inf"))
            for cr in config_results
        ]
        best_idx = int(np.nanargmin(means))
        best_config[metric] = config_names[best_idx]

    return {
        "configs": config_names,
        "metrics": metrics_table,
        "best_config": best_config,
    }


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_feature_analysis(
    *,
    model_cfg_path: Path,
    train_cfg_path: Path,
    ensemble_cfg_path: Path,
    data_cfg_path: Path | None,
    output_dir: Path,
    dataset_ids: list[int],
    seed_list: list[int],
    sizes: list[float],
    feature_groups: list[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    show_progress: bool = False,
) -> dict[str, Any]:
    logger = get_logger("feature_analysis")

    output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = _load_yaml(model_cfg_path)
    train_cfg_yaml = _load_yaml(train_cfg_path)
    ensemble_cfg_yaml = _load_yaml(ensemble_cfg_path)
    data_cfg = _load_yaml(data_cfg_path) if data_cfg_path and data_cfg_path.exists() else None
    preprocess_config = _build_preprocess_config(data_cfg)

    feature_configs = build_feature_configs(feature_groups)
    logger.info(
        "Feature configs to evaluate (%d): %s",
        len(feature_configs),
        [fc.name for fc in feature_configs],
    )

    config_paths = {
        "model": str(model_cfg_path),
        "train": str(train_cfg_path),
        "ensemble": str(ensemble_cfg_path),
        "data": str(data_cfg_path) if data_cfg_path else "",
    }

    all_results: dict[str, Any] = {
        "experiment_meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_ids": dataset_ids,
            "n_seeds": len(seed_list),
            "seed_list": seed_list,
            "sizes": sizes,
            "feature_groups": feature_groups,
            "configs": config_paths,
        },
        "datasets": {},
    }

    for dataset_id in dataset_ids:
        logger.info("=" * 80)
        logger.info("Dataset ID: %d", dataset_id)
        logger.info("=" * 80)

        try:
            dataset_bundle = load_dataset_bundle(dataset_id)
        except Exception as exc:
            logger.error("Failed to load dataset %d: %s", dataset_id, exc)
            all_results["datasets"][str(dataset_id)] = {"error": str(exc)}
            continue

        df = dataset_bundle.df
        target_col = dataset_bundle.target_column
        categorical_columns = dataset_bundle.categorical_columns
        n_features = int(df.drop(columns=[target_col], errors="ignore").shape[1])

        dataset_name: str
        try:
            dataset_name = str(dataset_bundle.df.attrs.get("name", f"dataset_{dataset_id}"))
        except Exception:
            dataset_name = f"dataset_{dataset_id}"

        # Try to get the name from OpenML metadata
        try:
            import openml  # type: ignore
            ds_meta = openml.datasets.get_dataset(dataset_id, download_data=False)
            if hasattr(ds_meta, "name") and ds_meta.name:
                dataset_name = str(ds_meta.name)
        except Exception:
            pass

        df_train_full, df_valid, df_test = prepare_dataset_splits(
            df=df,
            target_column=target_col,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed_list[0],
        )

        dataset_result: dict[str, Any] = {
            "dataset_name": dataset_name,
            "n_features": n_features,
            "n_samples_total": int(len(df)),
            "sizes": {},
        }

        for size_ratio in sizes:
            size_key = str(int(round(size_ratio * 100)))
            logger.info("-" * 60)
            logger.info("Size %.0f%%", size_ratio * 100)
            logger.info("-" * 60)

            # Preprocess once per size (using first seed for sampling; preprocessor
            # is re-fitted per seed inside the loop below if size_ratio < 1.0).
            split_first = prepare_preprocessed_splits_for_size(
                df_train_full=df_train_full,
                df_valid=df_valid,
                df_test=df_test,
                target_column=target_col,
                categorical_columns=categorical_columns,
                preprocess_config=preprocess_config,
                size_ratio=size_ratio,
                seed=seed_list[0],
            )
            n_train_samples = split_first["n_train_samples"]

            size_config_results: list[dict[str, Any]] = []

            for feature_config in feature_configs:
                logger.info(
                    "  Config: %s -- %s",
                    feature_config.name,
                    feature_config.description,
                )
                per_seed_results: list[dict[str, float]] = []

                for seed in seed_list:
                    # Re-fit preprocessor per seed to capture sampling variation
                    split = prepare_preprocessed_splits_for_size(
                        df_train_full=df_train_full,
                        df_valid=df_valid,
                        df_test=df_test,
                        target_column=target_col,
                        categorical_columns=categorical_columns,
                        preprocess_config=preprocess_config,
                        size_ratio=size_ratio,
                        seed=seed,
                    )

                    try:
                        seed_metrics = run_single_config_seed(
                            feature_config=feature_config,
                            seed=seed,
                            X_tr=split["X_tr"],
                            y_tr=split["y_tr"],
                            X_va=split["X_va"],
                            y_va=split["y_va"],
                            X_te=split["X_te"],
                            y_te=split["y_te"],
                            preprocessor=split["preprocessor"],
                            model_cfg=model_cfg,
                            train_cfg_yaml=train_cfg_yaml,
                            ensemble_cfg_yaml=ensemble_cfg_yaml,
                            show_progress=show_progress,
                        )
                        per_seed_results.append(seed_metrics)
                    except Exception as exc:
                        logger.error(
                            "Config=%s seed=%d FAILED: %s",
                            feature_config.name,
                            seed,
                            exc,
                        )
                        per_seed_results.append({"error": str(exc)})

                # Filter out errors for aggregation
                valid_metrics = [
                    m for m in per_seed_results if "error" not in m
                ]
                aggregated = _aggregate_metric_dicts(valid_metrics)

                # Build per-seed list for JSON (include seed number)
                per_seed_with_seed = []
                for s, m in zip(seed_list, per_seed_results):
                    entry = {"seed": s}
                    entry.update(m)
                    per_seed_with_seed.append(entry)

                # Settings summary for the config
                settings: dict[str, Any] = {
                    "aux_task": str(
                        feature_config.model_overrides.get("hmtl", {}).get(
                            "aux_task",
                            model_cfg.get("hmtl", {}).get("aux_task", "bins"),
                        )
                    ),
                    "cqr_enabled": bool(feature_config.train_overrides.get("cqr_enabled", False)),
                    "adversarial_enabled": False,
                }
                adv_override = feature_config.train_overrides.get("adversarial")
                if isinstance(adv_override, dict):
                    settings["adversarial_enabled"] = bool(adv_override.get("enabled", False))
                    if settings["adversarial_enabled"]:
                        settings["adversarial_method"] = adv_override.get("method", "fgsm")
                        settings["adversarial_epsilon"] = adv_override.get("epsilon", 0.01)

                config_entry = {
                    "name": feature_config.name,
                    "description": feature_config.description,
                    "settings": settings,
                    "per_seed": per_seed_with_seed,
                    "aggregated": aggregated,
                }
                size_config_results.append(config_entry)

            # Comparison table
            comparison = _build_comparison_table(size_config_results)

            dataset_result["sizes"][size_key] = {
                "size_ratio": float(size_ratio),
                "n_train_samples": n_train_samples,
                "configs": size_config_results,
                "comparison": comparison,
            }

        # Save per-dataset results
        ds_folder = output_dir / f"dataset_{dataset_id}"
        ds_folder.mkdir(parents=True, exist_ok=True)
        ds_result_path = ds_folder / "results.json"
        with open(ds_result_path, "w", encoding="utf-8") as f:
            json.dump(dataset_result, f, indent=2, default=str)
        logger.info("Saved dataset results to %s", ds_result_path)

        all_results["datasets"][str(dataset_id)] = dataset_result

    # Save aggregated results
    all_results_path = output_dir / "feature_analysis_results.json"
    with open(all_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Saved all results to %s", all_results_path)

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feature analysis: test CQR, adversarial, aux tasks on regression metrics"
    )
    parser.add_argument("--dataset-id", type=int, default=None, help="Single dataset ID")
    parser.add_argument(
        "--dataset-ids",
        nargs="+",
        type=int,
        default=None,
        help="Multiple dataset IDs (space-separated)",
    )
    parser.add_argument("--n-seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None, help="Explicit seed list"
    )
    parser.add_argument(
        "--output",
        default="experiments/feature_analysis",
        help="Output directory",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=["baseline", "cqr", "adversarial", "aux_tasks"],
        choices=["baseline", "cqr", "adversarial", "aux_tasks", "combined", "all"],
        help="Feature groups to test",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=float,
        default=[1.0],
        help="Train size ratios to evaluate",
    )
    parser.add_argument("--model", default="configs/model_snn.yaml", help="Model config")
    parser.add_argument("--train", default="configs/train.yaml", help="Train config")
    parser.add_argument("--ensemble", default="configs/ensemble.yaml", help="Ensemble config")
    parser.add_argument("--data", default="configs/data.yaml", help="Data config")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    args = parser.parse_args()

    setup_logging(log_level=logging.INFO)
    logger = get_logger("feature_analysis")
    logger.info("=" * 80)
    logger.info("Feature Analysis Experiment")
    logger.info("=" * 80)

    # Resolve dataset IDs
    if args.dataset_id is not None and args.dataset_ids is not None:
        raise ValueError("Use either --dataset-id or --dataset-ids, not both")
    if args.dataset_id is not None:
        dataset_ids = [args.dataset_id]
    elif args.dataset_ids is not None:
        dataset_ids = list(dict.fromkeys(args.dataset_ids))
    else:
        # Default: use well-known regression datasets from OpenML
        dataset_ids = [44959, 44973, 44964]
        logger.info("No dataset specified, using default dataset_ids=44959 (concrete), 44973 (superconduct), 44964 (house_sales)")

    seed_list = _resolve_seed_list(args.seed, args.seeds, args.n_seeds)
    logger.info("Seeds: %s", seed_list)
    logger.info("Feature groups: %s", args.features)
    logger.info("Sizes: %s", args.sizes)

    if not np.isclose(args.train_ratio + args.val_ratio + args.test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    run_feature_analysis(
        model_cfg_path=Path(args.model),
        train_cfg_path=Path(args.train),
        ensemble_cfg_path=Path(args.ensemble),
        data_cfg_path=Path(args.data) if args.data else None,
        output_dir=Path(args.output),
        dataset_ids=dataset_ids,
        seed_list=seed_list,
        sizes=args.sizes,
        feature_groups=args.features,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    logger.info("Feature analysis completed.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
