#!/usr/bin/env python3
"""Run size-dependence experiments on OpenML regression datasets.

This script compares HMTL against configurable baselines on multiple dataset sizes
and multiple seeds while avoiding preprocessing leakage.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import logging
import multiprocessing
import os
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm
import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

HIGH_LEVEL_PROGRESS_FLAG = "--high-level-progress-only"
HIGH_LEVEL_PROGRESS_ONLY = HIGH_LEVEL_PROGRESS_FLAG in sys.argv
if HIGH_LEVEL_PROGRESS_ONLY:
    warnings.filterwarnings("ignore")
    mpl_cache_dir = project_root / ".cache" / "matplotlib"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

from src.utils.logger import get_logger, setup_logging

if HIGH_LEVEL_PROGRESS_ONLY:
    setup_logging(log_level=logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)

from src.baselines.trainer import (
    train_catboost_baseline,
    train_flat_mtl_baseline,
    train_single_mlp_baseline,
)
from src.data.openml_loader import (
    get_regression_datasets,
    load_dataset,
    prepare_dataset_splits,
    sample_train_data,
)
from src.data.preprocess import PreprocessConfig, TabularPreprocessor
from src.eval.ensemble import ensemble_predict
from src.eval.evaluator import evaluate_on_dataset
from src.eval.metrics import EvaluationMetrics, evaluate_comprehensive
from src.models.hmtl import HMTLModel
from src.train.ensemble import EnsembleConfig, fit_ensemble
from src.train.loop import TrainConfig


SUPPORTED_BASELINES = ("catboost", "single_mlp", "flat_mtl")


@dataclass
class DatasetMeta:
    dataset_id: int
    dataset_name: str
    task_id: int | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _slugify(text: str) -> str:
    lowered = text.lower().strip()
    cleaned = re.sub(r"[^a-z0-9]+", "_", lowered)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "unknown"


def _resolve_dataset_name(dataset_id: int, fallback_name: str | None) -> str:
    if fallback_name and fallback_name.strip() and fallback_name != f"dataset_{dataset_id}":
        return fallback_name

    # Try to get a meaningful name from OpenML metadata.
    try:
        import openml  # type: ignore

        dataset = openml.datasets.get_dataset(dataset_id, download_data=False)
        name = getattr(dataset, "name", None)
        if name:
            return str(name)
    except Exception:
        pass

    if fallback_name == f"dataset_{dataset_id}":
        return f"openml_{dataset_id}"
    return fallback_name or f"openml_{dataset_id}"


def _resolve_seed_list(base_seed: int, seeds: list[int] | None, n_seeds: int) -> list[int]:
    if seeds:
        deduped = list(dict.fromkeys(seeds))
        return deduped
    return [base_seed + i for i in range(n_seeds)]


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
    )


def prepare_preprocessed_splits_for_size(
    *,
    df_train_full,
    df_valid,
    df_test,
    target_column: str,
    preprocess_config: PreprocessConfig,
    size_ratio: float,
    seed: int,
) -> dict[str, Any]:
    """Sample train subset for a given size and fit a *new* preprocessor.

    Returns transformed arrays plus metadata. The preprocessor is fitted only on
    sampled training data (anti-leakage).
    """
    df_train_sampled = sample_train_data(df_train_full, size_ratio=size_ratio, seed=seed)

    preprocessor = TabularPreprocessor(
        preprocess_config,
        target_column=target_column,
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


def _aggregate_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not metric_dicts:
        return {}

    metric_names = sorted(set().union(*(d.keys() for d in metric_dicts)))
    aggregated: dict[str, dict[str, float]] = {}

    for metric_name in metric_names:
        values = [d[metric_name] for d in metric_dicts if metric_name in d and np.isfinite(d[metric_name])]
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        aggregated[metric_name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=0)),
        }

    return aggregated


def _aggregate_delta_dicts(delta_dicts: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not delta_dicts:
        return {}
    return _aggregate_metric_dicts(delta_dicts)


def _extract_mean_metrics(aggregated: dict[str, dict[str, float]]) -> dict[str, float]:
    return {metric: stats["mean"] for metric, stats in aggregated.items() if "mean" in stats}


def _compute_delta_vs_hmtl(baseline_metrics: dict[str, float], hmtl_metrics: dict[str, float]) -> dict[str, float]:
    return {
        "delta_rmse": float(baseline_metrics["rmse"] - hmtl_metrics["rmse"]),
        "delta_r_auc_mse": float(baseline_metrics["r_auc_mse"] - hmtl_metrics["r_auc_mse"]),
    }


def _build_hmtl_model_builder(
    *,
    input_dim: int,
    model_cfg: dict[str, Any],
    scale_coeff: float,
):
    hidden_width = int(model_cfg["encoder"]["hidden_width"])

    def build_model() -> HMTLModel:
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
        )

    return build_model


def train_and_evaluate_hmtl(
    *,
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
    seed: int,
    show_inner_progress: bool = True,
) -> dict[str, float]:
    logger = get_logger("automlbenchmark")

    train_cfg = TrainConfig(
        lr=float(train_cfg_yaml["optimizer"]["lr"]),
        epochs=int(train_cfg_yaml["training"]["epochs"]),
        batch_size=int(train_cfg_yaml["training"]["batch_size"]),
        patience=int(train_cfg_yaml["training"].get("early_stop", {}).get("patience", 10)),
        aux_weight=float(model_cfg["hmtl"]["lambda_aux"]),
        optimizer=str(train_cfg_yaml["optimizer"].get("name", "radam_lookahead")),
        lookahead_k=int(train_cfg_yaml["optimizer"].get("lookahead_sync_period", 6)),
        lookahead_alpha=float(train_cfg_yaml["optimizer"].get("lookahead_slow_step", 0.5)),
        weight_decay=float(train_cfg_yaml["optimizer"].get("weight_decay", 0.0)),
        sigma_reg_weight=float(train_cfg_yaml["training"].get("sigma_reg_weight", 0.0)),
        seed=seed,
        task_type="regression",
        show_progress=show_inner_progress,
    )

    ens_cfg = EnsembleConfig(
        n_models=int(ensemble_cfg_yaml["ensemble"]["n_models"]),
        bagging=str(ensemble_cfg_yaml["ensemble"].get("bagging", "stratified_bins")),
        show_progress=show_inner_progress,
    )

    input_dim = X_tr.shape[1]
    scale_coeff = (
        float(preprocessor.target_std_)
        if preprocessor.target_std_ is not None and preprocessor.target_std_ > 1e-12
        else 1.0
    )

    build_model = _build_hmtl_model_builder(
        input_dim=input_dim,
        model_cfg=model_cfg,
        scale_coeff=scale_coeff,
    )

    models, avg_score = fit_ensemble(
        build_model_fn=build_model,
        X_tr=X_tr,
        y_tr=y_tr,
        X_va=X_va,
        y_va=y_va,
        n_bins=int(model_cfg["hmtl"]["n_bins"]),
        ens_cfg=ens_cfg,
        train_cfg=train_cfg,
    )

    eval_results = evaluate_on_dataset(
        models=models,
        X=X_te,
        y_true=y_te,
        X_cal=X_va,
        y_cal=y_va,
        coverage_levels=[0.80, 0.90, 0.95],
        preprocessor=preprocessor,
        use_normalized_metrics=True,
    )

    metrics = _metrics_to_dict(eval_results.metrics)
    metrics["ensemble_avg_val_r_auc_mse"] = float(avg_score)

    logger.info(
        "HMTL metrics: rmse=%.6f r_auc_mse=%.6f",
        metrics["rmse"],
        metrics["r_auc_mse"],
    )
    return metrics


def train_and_evaluate_baseline(
    *,
    baseline_name: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    model_cfg: dict[str, Any],
    train_cfg_yaml: dict[str, Any],
    ensemble_cfg_yaml: dict[str, Any],
    seed: int,
    show_inner_progress: bool = True,
) -> dict[str, float]:
    train_cfg = TrainConfig(
        lr=float(train_cfg_yaml["optimizer"]["lr"]),
        epochs=int(train_cfg_yaml["training"]["epochs"]),
        batch_size=int(train_cfg_yaml["training"]["batch_size"]),
        patience=int(train_cfg_yaml["training"].get("early_stop", {}).get("patience", 10)),
        aux_weight=float(model_cfg["hmtl"]["lambda_aux"]),
        optimizer=str(train_cfg_yaml["optimizer"].get("name", "radam_lookahead")),
        lookahead_k=int(train_cfg_yaml["optimizer"].get("lookahead_sync_period", 6)),
        lookahead_alpha=float(train_cfg_yaml["optimizer"].get("lookahead_slow_step", 0.5)),
        weight_decay=float(train_cfg_yaml["optimizer"].get("weight_decay", 0.0)),
        sigma_reg_weight=float(train_cfg_yaml["training"].get("sigma_reg_weight", 0.0)),
        seed=seed,
        task_type="regression",
        show_progress=show_inner_progress,
    )

    input_dim = X_tr.shape[1]

    if baseline_name == "single_mlp":
        model = train_single_mlp_baseline(
            X_tr=X_tr,
            y_tr=y_tr,
            X_va=X_va,
            y_va=y_va,
            input_dim=input_dim,
            hidden_width=int(model_cfg["encoder"]["hidden_width"]),
            depth=int(model_cfg["hmtl"]["high_layer"]),
            alpha_dropout=float(model_cfg["encoder"]["alpha_dropout"]),
            train_cfg=train_cfg,
        )
        y_pred, unc_total, unc_epi, unc_alea = ensemble_predict([model], X_te)

    elif baseline_name == "flat_mtl":
        model = train_flat_mtl_baseline(
            X_tr=X_tr,
            y_tr=y_tr,
            X_va=X_va,
            y_va=y_va,
            input_dim=input_dim,
            hidden_width=int(model_cfg["encoder"]["hidden_width"]),
            depth=int(model_cfg["hmtl"]["high_layer"]),
            alpha_dropout=float(model_cfg["encoder"]["alpha_dropout"]),
            n_bins=int(model_cfg["hmtl"]["n_bins"]),
            aux_weight=float(model_cfg["hmtl"]["lambda_aux"]),
            train_cfg=train_cfg,
        )
        y_pred, unc_total, unc_epi, unc_alea = ensemble_predict([model], X_te)

    elif baseline_name == "catboost":
        catboost_n_models = min(10, int(ensemble_cfg_yaml["ensemble"].get("n_models", 10)))
        model = train_catboost_baseline(
            X_tr=X_tr,
            y_tr=y_tr,
            X_va=X_va,
            y_va=y_va,
            n_models=catboost_n_models,
            random_seed=seed,
        )
        y_pred, unc_total, unc_epi, unc_alea = model.predict(X_te)

    else:
        raise ValueError(f"Unsupported baseline: {baseline_name}")

    metrics = evaluate_comprehensive(
        y_true=y_te,
        y_pred=y_pred,
        uncertainty=unc_total,
        epistemic=unc_epi,
        aleatoric=unc_alea,
    )

    return _metrics_to_dict(metrics)


def run_size_seed_trial(
    *,
    size_ratio: float,
    seed: int,
    df_train_full,
    df_valid,
    df_test,
    target_column: str,
    preprocess_config: PreprocessConfig,
    model_cfg: dict[str, Any],
    train_cfg_yaml: dict[str, Any],
    ensemble_cfg_yaml: dict[str, Any],
    baselines: list[str],
    show_inner_progress: bool = True,
) -> dict[str, Any]:
    logger = get_logger("automlbenchmark")

    split = prepare_preprocessed_splits_for_size(
        df_train_full=df_train_full,
        df_valid=df_valid,
        df_test=df_test,
        target_column=target_column,
        preprocess_config=preprocess_config,
        size_ratio=size_ratio,
        seed=seed,
    )

    result: dict[str, Any] = {
        "seed": int(seed),
        "status": "ok",
        "n_train_samples": int(split["n_train_samples"]),
        "hmtl": None,
        "baselines": {},
        "delta_vs_hmtl": {},
    }

    try:
        hmtl_metrics = train_and_evaluate_hmtl(
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
            seed=seed,
            show_inner_progress=show_inner_progress,
        )
        result["hmtl"] = hmtl_metrics
    except Exception as exc:
        logger.error("HMTL failed for size %.0f%% seed %d: %s", size_ratio * 100, seed, exc)
        result["status"] = "failed"
        result["error"] = f"HMTL failed: {exc}"
        return result

    for baseline_name in baselines:
        try:
            baseline_metrics = train_and_evaluate_baseline(
                baseline_name=baseline_name,
                X_tr=split["X_tr"],
                y_tr=split["y_tr"],
                X_va=split["X_va"],
                y_va=split["y_va"],
                X_te=split["X_te"],
                y_te=split["y_te"],
                model_cfg=model_cfg,
                train_cfg_yaml=train_cfg_yaml,
                ensemble_cfg_yaml=ensemble_cfg_yaml,
                seed=seed,
                show_inner_progress=show_inner_progress,
            )
            result["baselines"][baseline_name] = baseline_metrics
            result["delta_vs_hmtl"][baseline_name] = _compute_delta_vs_hmtl(
                baseline_metrics,
                hmtl_metrics,
            )
        except Exception as exc:
            logger.error(
                "Baseline %s failed for size %.0f%% seed %d: %s",
                baseline_name,
                size_ratio * 100,
                seed,
                exc,
            )
            result["baselines"][baseline_name] = {"error": str(exc)}
            result["delta_vs_hmtl"][baseline_name] = {"error": str(exc)}

    return result


def aggregate_size_seed_runs(
    *,
    per_seed_runs: list[dict[str, Any]],
    baselines: list[str],
) -> dict[str, Any]:
    n_requested = len(per_seed_runs)

    hmtl_success_runs = [run for run in per_seed_runs if run.get("status") == "ok" and isinstance(run.get("hmtl"), dict)]
    failed_seeds = [int(run["seed"]) for run in per_seed_runs if run.get("status") != "ok"]

    hmtl_metric_dicts = [run["hmtl"] for run in hmtl_success_runs]
    hmtl_agg = _aggregate_metric_dicts(hmtl_metric_dicts)
    hmtl_means = _extract_mean_metrics(hmtl_agg)

    baselines_agg: dict[str, Any] = {}
    baselines_means: dict[str, dict[str, float]] = {}
    delta_agg: dict[str, Any] = {}
    delta_means: dict[str, dict[str, float]] = {}

    for baseline_name in baselines:
        baseline_success = []
        baseline_delta_success = []

        for run in hmtl_success_runs:
            baseline_metrics = run.get("baselines", {}).get(baseline_name)
            if isinstance(baseline_metrics, dict) and "error" not in baseline_metrics:
                baseline_success.append(baseline_metrics)
                delta_metrics = run.get("delta_vs_hmtl", {}).get(baseline_name)
                if isinstance(delta_metrics, dict) and "error" not in delta_metrics:
                    baseline_delta_success.append(delta_metrics)

        agg_metrics = _aggregate_metric_dicts(baseline_success)
        agg_delta = _aggregate_delta_dicts(baseline_delta_success)

        baselines_agg[baseline_name] = {
            "n_successful": int(len(baseline_success)),
            **agg_metrics,
        }
        baselines_means[baseline_name] = _extract_mean_metrics(agg_metrics)

        delta_agg[baseline_name] = {
            "n_successful": int(len(baseline_delta_success)),
            **agg_delta,
        }
        delta_means[baseline_name] = _extract_mean_metrics(agg_delta)

    aggregate_over_seeds = {
        "n_requested": int(n_requested),
        "n_successful": int(len(hmtl_success_runs)),
        "failed_seeds": failed_seeds,
        "hmtl": hmtl_agg,
        "baselines": baselines_agg,
        "delta_vs_hmtl": delta_agg,
    }

    size_summary: dict[str, Any] = {
        "status": "ok" if hmtl_success_runs else "failed",
        "aggregate_over_seeds": aggregate_over_seeds,
        "hmtl": hmtl_means,
        "baselines": baselines_means,
        "delta_vs_hmtl": delta_means,
    }

    # Backward-compatible aliases for CatBoost.
    catboost_mean = baselines_means.get("catboost")
    catboost_delta = delta_means.get("catboost")
    if catboost_mean:
        size_summary["catboost"] = catboost_mean
    if catboost_delta:
        if "delta_rmse" in catboost_delta:
            size_summary["delta_rmse"] = catboost_delta["delta_rmse"]
        if "delta_r_auc_mse" in catboost_delta:
            size_summary["delta_r_auc_mse"] = catboost_delta["delta_r_auc_mse"]

    return size_summary


def run_single_dataset_experiment(
    *,
    dataset_meta: DatasetMeta,
    sizes: list[float],
    seeds: list[int],
    model_cfg: dict[str, Any],
    train_cfg_yaml: dict[str, Any],
    ensemble_cfg_yaml: dict[str, Any],
    preprocess_config: PreprocessConfig,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    baselines: list[str],
    split_seed: int,
    output_dir: Path,
    study_id: int,
    config_paths: dict[str, str],
    show_trial_progress: bool = False,
    show_inner_progress: bool = True,
) -> dict[str, Any]:
    logger = get_logger("automlbenchmark")

    dataset_name = _resolve_dataset_name(dataset_meta.dataset_id, dataset_meta.dataset_name)
    slug = _slugify(dataset_name)
    dataset_folder = output_dir / f"dataset_{dataset_meta.dataset_id}_{slug}"
    dataset_folder.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Dataset: %s (ID: %d)", dataset_name, dataset_meta.dataset_id)
    logger.info("=" * 80)

    df, target_col = load_dataset(dataset_meta.dataset_id)
    n_features = int(df.drop(columns=[target_col], errors="ignore").shape[1])
    df_train_full, df_valid, df_test = prepare_dataset_splits(
        df=df,
        target_column=target_col,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=split_seed,
    )

    result: dict[str, Any] = {
        "dataset_id": int(dataset_meta.dataset_id),
        "dataset_name": dataset_name,
        "task_id": dataset_meta.task_id,
        "n_samples_total": int(len(df)),
        "n_samples_train": int(len(df_train_full)),
        "n_samples_valid": int(len(df_valid)),
        "n_samples_test": int(len(df_test)),
        "n_features": n_features,
        "target_column": target_col,
        "run_meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "study_id": int(study_id),
            "seed_list": [int(s) for s in seeds],
            "n_requested_seeds": int(len(seeds)),
            "sizes": [float(s) for s in sizes],
            "baselines": baselines,
            "configs": config_paths,
            "split": {
                "train_ratio": float(train_ratio),
                "val_ratio": float(val_ratio),
                "test_ratio": float(test_ratio),
                "split_seed": int(split_seed),
            },
            "output_folder": str(dataset_folder),
        },
        "sizes": {},
    }

    total_trials = len(sizes) * len(seeds)
    with tqdm(
        total=total_trials,
        desc=f"Dataset {dataset_meta.dataset_id} size*seed",
        unit="trial",
        leave=False,
        disable=not show_trial_progress,
    ) as trial_pbar:
        for size_ratio in sizes:
            size_key = str(int(round(size_ratio * 100)))
            logger.info("-" * 80)
            logger.info("Size %.0f%%", size_ratio * 100)
            logger.info("-" * 80)

            per_seed_runs = []
            for seed in seeds:
                seed_result = run_size_seed_trial(
                    size_ratio=size_ratio,
                    seed=seed,
                    df_train_full=df_train_full,
                    df_valid=df_valid,
                    df_test=df_test,
                    target_column=target_col,
                    preprocess_config=preprocess_config,
                    model_cfg=model_cfg,
                    train_cfg_yaml=train_cfg_yaml,
                    ensemble_cfg_yaml=ensemble_cfg_yaml,
                    baselines=baselines,
                    show_inner_progress=show_inner_progress,
                )
                per_seed_runs.append(seed_result)
                trial_pbar.update(1)

            size_summary = aggregate_size_seed_runs(
                per_seed_runs=per_seed_runs,
                baselines=baselines,
            )

            n_train_samples = (
                int(per_seed_runs[0]["n_train_samples"])
                if per_seed_runs
                else int(round(len(df_train_full) * size_ratio))
            )

            result["sizes"][size_key] = {
                "size_ratio": float(size_ratio),
                "n_train_samples": int(n_train_samples),
                "per_seed": {str(run["seed"]): run for run in per_seed_runs},
                **size_summary,
            }

    dataset_result_file = dataset_folder / "results.json"
    with open(dataset_result_file, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    logger.info("Saved dataset results to %s", dataset_result_file)
    return result


def _build_dataset_failure_result(
    *,
    dataset_meta: DatasetMeta,
    exc: Exception,
    study_id: int,
    seed_list: list[int],
    sizes: list[float],
    baselines: list[str],
) -> dict[str, Any]:
    return {
        "dataset_id": int(dataset_meta.dataset_id),
        "dataset_name": dataset_meta.dataset_name,
        "task_id": dataset_meta.task_id,
        "error": str(exc),
        "run_meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "study_id": int(study_id),
            "seed_list": [int(s) for s in seed_list],
            "sizes": [float(s) for s in sizes],
            "baselines": baselines,
        },
    }


def _write_aggregated_results(
    *,
    aggregated_results_by_index: list[dict[str, Any] | None],
    aggregated_path: Path,
) -> None:
    ordered_completed_results = [result for result in aggregated_results_by_index if result is not None]
    with open(aggregated_path, "w", encoding="utf-8") as file:
        json.dump(ordered_completed_results, file, indent=2)


def _accelerator_available() -> bool:
    try:
        import torch  # type: ignore
    except Exception:
        return False

    if bool(torch.cuda.is_available()):
        return True

    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend is not None and mps_backend.is_available())


def run_automlbenchmark_experiments(
    *,
    model_cfg_path: Path,
    train_cfg_path: Path,
    ensemble_cfg_path: Path,
    data_cfg_path: Path | None,
    output_dir: Path,
    sizes: list[float],
    dataset_id: int | None,
    study_id: int,
    seed: int,
    seeds: list[int] | None,
    n_seeds: int,
    max_datasets: int | None,
    max_dataset_workers: int = 1,
    baselines: list[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    high_level_progress_only: bool = False,
) -> list[dict[str, Any]]:
    logger = get_logger("automlbenchmark")

    baselines = [baseline.strip().lower() for baseline in baselines]
    baselines = list(dict.fromkeys(baselines))

    for baseline_name in baselines:
        if baseline_name not in SUPPORTED_BASELINES:
            raise ValueError(
                f"Unsupported baseline '{baseline_name}'. Supported: {SUPPORTED_BASELINES}"
            )

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    if max_dataset_workers < 1:
        raise ValueError("max_dataset_workers must be >= 1")

    output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = _load_yaml(model_cfg_path)
    train_cfg_yaml = _load_yaml(train_cfg_path)
    ensemble_cfg_yaml = _load_yaml(ensemble_cfg_path)
    data_cfg = _load_yaml(data_cfg_path) if data_cfg_path and data_cfg_path.exists() else None
    preprocess_config = _build_preprocess_config(data_cfg)

    seed_list = _resolve_seed_list(base_seed=seed, seeds=seeds, n_seeds=n_seeds)
    logger.info("Seeds: %s", seed_list)
    logger.info("Baselines: %s", baselines)

    datasets_meta: list[DatasetMeta]
    if dataset_id is not None:
        datasets_meta = [DatasetMeta(dataset_id=dataset_id, dataset_name=f"dataset_{dataset_id}")]
    else:
        datasets_info = get_regression_datasets(study_id=study_id)
        datasets_meta = [
            DatasetMeta(
                dataset_id=int(info["dataset_id"]),
                dataset_name=str(info.get("name", f"dataset_{info['dataset_id']}")),
                task_id=int(info["task_id"]) if info.get("task_id") is not None else None,
            )
            for info in datasets_info
        ]

    if max_datasets is not None:
        datasets_meta = datasets_meta[:max_datasets]

    effective_workers = min(max_dataset_workers, len(datasets_meta))

    logger.info("Processing %d datasets", len(datasets_meta))
    logger.info(
        "Dataset workers: requested=%d effective=%d",
        max_dataset_workers,
        effective_workers,
    )
    if max_dataset_workers > 1 and _accelerator_available():
        logger.warning(
            "Parallel dataset workers requested while CUDA/MPS is available. "
            "This may cause accelerator memory contention or OOM."
        )

    config_paths = {
        "model": str(model_cfg_path),
        "train": str(train_cfg_path),
        "ensemble": str(ensemble_cfg_path),
        "data": str(data_cfg_path) if data_cfg_path else "",
    }

    aggregated_results_by_index: list[dict[str, Any] | None] = [None] * len(datasets_meta)
    aggregated_path = output_dir / "aggregated_results.json"

    with tqdm(
        total=len(datasets_meta),
        desc="Datasets",
        unit="dataset",
        leave=True,
        disable=not high_level_progress_only,
    ) as dataset_pbar:
        if effective_workers <= 1 or len(datasets_meta) <= 1:
            for idx, dataset_meta in enumerate(datasets_meta, start=1):
                if high_level_progress_only:
                    dataset_pbar.set_postfix_str(f"id={dataset_meta.dataset_id}")
                else:
                    logger.info("\nDataset %d/%d", idx, len(datasets_meta))

                try:
                    dataset_result = run_single_dataset_experiment(
                        dataset_meta=dataset_meta,
                        sizes=sizes,
                        seeds=seed_list,
                        model_cfg=model_cfg,
                        train_cfg_yaml=train_cfg_yaml,
                        ensemble_cfg_yaml=ensemble_cfg_yaml,
                        preprocess_config=preprocess_config,
                        train_ratio=train_ratio,
                        val_ratio=val_ratio,
                        test_ratio=test_ratio,
                        baselines=baselines,
                        split_seed=seed,
                        output_dir=output_dir,
                        study_id=study_id,
                        config_paths=config_paths,
                        show_trial_progress=high_level_progress_only,
                        show_inner_progress=not high_level_progress_only,
                    )
                except Exception as exc:
                    logger.error(
                        "Dataset %d failed (%s): %s",
                        dataset_meta.dataset_id,
                        dataset_meta.dataset_name,
                        exc,
                    )
                    dataset_result = _build_dataset_failure_result(
                        dataset_meta=dataset_meta,
                        exc=exc,
                        study_id=study_id,
                        seed_list=seed_list,
                        sizes=sizes,
                        baselines=baselines,
                    )

                aggregated_results_by_index[idx - 1] = dataset_result
                _write_aggregated_results(
                    aggregated_results_by_index=aggregated_results_by_index,
                    aggregated_path=aggregated_path,
                )
                dataset_pbar.update(1)
        else:
            mp_context = multiprocessing.get_context("spawn")
            # Keep only the parent dataset-level progress bar in parallel mode.
            # Multiple per-worker tqdm bars tend to corrupt/hide the global bar.
            worker_show_trial_progress = False
            with futures.ProcessPoolExecutor(
                max_workers=effective_workers,
                mp_context=mp_context,
            ) as executor:
                future_to_dataset: dict[futures.Future[dict[str, Any]], tuple[int, DatasetMeta]] = {}

                for idx, dataset_meta in enumerate(datasets_meta):
                    if not high_level_progress_only:
                        logger.info(
                            "Submitting dataset %d/%d (id=%d)",
                            idx + 1,
                            len(datasets_meta),
                            dataset_meta.dataset_id,
                        )
                    future = executor.submit(
                        run_single_dataset_experiment,
                        dataset_meta=dataset_meta,
                        sizes=sizes,
                        seeds=seed_list,
                        model_cfg=model_cfg,
                        train_cfg_yaml=train_cfg_yaml,
                        ensemble_cfg_yaml=ensemble_cfg_yaml,
                        preprocess_config=preprocess_config,
                        train_ratio=train_ratio,
                        val_ratio=val_ratio,
                        test_ratio=test_ratio,
                        baselines=baselines,
                        split_seed=seed,
                        output_dir=output_dir,
                        study_id=study_id,
                        config_paths=config_paths,
                        show_trial_progress=worker_show_trial_progress,
                        show_inner_progress=not high_level_progress_only,
                    )
                    future_to_dataset[future] = (idx, dataset_meta)

                for future in futures.as_completed(future_to_dataset):
                    idx, dataset_meta = future_to_dataset[future]
                    if high_level_progress_only:
                        dataset_pbar.set_postfix_str(f"id={dataset_meta.dataset_id}")

                    try:
                        dataset_result = future.result()
                    except Exception as exc:
                        logger.error(
                            "Dataset %d failed (%s): %s",
                            dataset_meta.dataset_id,
                            dataset_meta.dataset_name,
                            exc,
                        )
                        dataset_result = _build_dataset_failure_result(
                            dataset_meta=dataset_meta,
                            exc=exc,
                            study_id=study_id,
                            seed_list=seed_list,
                            sizes=sizes,
                            baselines=baselines,
                        )

                    aggregated_results_by_index[idx] = dataset_result
                    _write_aggregated_results(
                        aggregated_results_by_index=aggregated_results_by_index,
                        aggregated_path=aggregated_path,
                    )
                    dataset_pbar.update(1)

    return [result for result in aggregated_results_by_index if result is not None]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AutoMLBenchmark size experiments")
    parser.add_argument("--model", default="configs/model_snn.yaml", help="Path to model config")
    parser.add_argument("--train", default="configs/train.yaml", help="Path to train config")
    parser.add_argument("--ensemble", default="configs/ensemble.yaml", help="Path to ensemble config")
    parser.add_argument(
        "--data",
        default="configs/data.yaml",
        help="Optional data config (preprocessing settings only)",
    )
    parser.add_argument(
        "--output",
        default="experiments/automlbenchmark",
        help="Directory for experiment outputs",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        help="Train size ratios to evaluate",
    )
    parser.add_argument("--dataset-id", type=int, default=None, help="Run only this dataset ID")
    parser.add_argument("--study-id", type=int, default=269, help="OpenML study ID")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Explicit seed list")
    parser.add_argument(
        "--n-seeds",
        "--n_seeds",
        dest="n_seeds",
        type=int,
        default=3,
        help="Number of seeds (when --seeds not provided)",
    )
    parser.add_argument("--max-datasets", type=int, default=None, help="Limit number of datasets")
    parser.add_argument(
        "--max-dataset-workers",
        "--max_dataset_workers",
        dest="max_dataset_workers",
        type=int,
        default=1,
        help="Maximum number of datasets to process in parallel (default: 1)",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["catboost", "single_mlp"],
        help=(
            "Baselines to compare against HMTL. "
            "Supported: catboost single_mlp flat_mtl"
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio")
    parser.add_argument(
        "--high-level-progress-only",
        action="store_true",
        help=(
            "Show only high-level tqdm bars (datasets and per-dataset size*seed progress). "
            "Suppresses detailed console logs and inner training progress bars."
        ),
    )

    args = parser.parse_args()

    console_log_level = logging.ERROR if args.high_level_progress_only else logging.INFO
    setup_logging(log_level=console_log_level)
    if args.high_level_progress_only:
        logging.getLogger().setLevel(logging.ERROR)
        for noisy_logger in ("openml", "urllib3", "matplotlib", "numexpr"):
            logging.getLogger(noisy_logger).setLevel(logging.ERROR)

    logger = get_logger("automlbenchmark")
    logger.info("=" * 80)
    logger.info("AutoML Benchmark Regression Experiments")
    logger.info("=" * 80)

    output_dir = Path(args.output)

    results = run_automlbenchmark_experiments(
        model_cfg_path=Path(args.model),
        train_cfg_path=Path(args.train),
        ensemble_cfg_path=Path(args.ensemble),
        data_cfg_path=Path(args.data) if args.data else None,
        output_dir=output_dir,
        sizes=args.sizes,
        dataset_id=args.dataset_id,
        study_id=args.study_id,
        seed=args.seed,
        seeds=args.seeds,
        n_seeds=args.n_seeds,
        max_datasets=args.max_datasets,
        max_dataset_workers=args.max_dataset_workers,
        baselines=args.baselines,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        high_level_progress_only=args.high_level_progress_only,
    )

    aggregated_path = output_dir / "aggregated_results.json"
    with open(aggregated_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    logger.info("\nAll experiments completed")
    logger.info("Aggregated results saved to %s", aggregated_path)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
