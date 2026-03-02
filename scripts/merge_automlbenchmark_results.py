#!/usr/bin/env python3
"""Merge two AutoMLBenchmark aggregated_results.json files.

The second file has priority on conflicting fields. After merging per-seed runs,
this script rebuilds size-level aggregates (`aggregate_over_seeds`, `hmtl`,
`baselines`, `delta_vs_hmtl`) so the output is internally consistent.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any


def _load_results(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        loaded = json.load(file)

    if isinstance(loaded, list):
        return loaded
    if isinstance(loaded, dict) and isinstance(loaded.get("results"), list):
        return loaded["results"]
    raise ValueError(f"Unsupported JSON format in {path}. Expected list or {{'results': [...]}}.")


def _safe_float(value: Any) -> float | None:
    try:
        casted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(casted):
        return None
    return casted


def _aggregate_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not metric_dicts:
        return {}

    metric_names = sorted(set().union(*(metrics.keys() for metrics in metric_dicts)))
    aggregated: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        values = [
            float(metrics[metric_name])
            for metrics in metric_dicts
            if metric_name in metrics and _safe_float(metrics[metric_name]) is not None
        ]
        if not values:
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        aggregated[metric_name] = {
            "mean": float(mean),
            "std": float(math.sqrt(variance)),
        }
    return aggregated


def _extract_mean_metrics(aggregated: dict[str, dict[str, float]]) -> dict[str, float]:
    return {metric: values["mean"] for metric, values in aggregated.items() if "mean" in values}


def _compute_delta_vs_hmtl(baseline_metrics: dict[str, float], hmtl_metrics: dict[str, float]) -> dict[str, float]:
    return {
        "delta_rmse": float(baseline_metrics["rmse"] - hmtl_metrics["rmse"]),
        "delta_r_auc_mse": float(baseline_metrics["r_auc_mse"] - hmtl_metrics["r_auc_mse"]),
    }


def _seed_sort_key(seed_value: Any) -> tuple[int, int | str]:
    try:
        return (0, int(seed_value))
    except (TypeError, ValueError):
        return (1, str(seed_value))


def _normalize_metrics_block(metrics: Any) -> dict[str, float] | None:
    if not isinstance(metrics, dict) or "error" in metrics:
        return None

    normalized: dict[str, float] = {}
    for key, value in metrics.items():
        numeric = _safe_float(value)
        if numeric is not None:
            normalized[str(key)] = float(numeric)
    if not normalized:
        return None
    return normalized


def _refresh_seed_deltas(seed_run: dict[str, Any]) -> None:
    hmtl_metrics = _normalize_metrics_block(seed_run.get("hmtl"))
    if hmtl_metrics is None:
        return

    baselines = seed_run.setdefault("baselines", {})
    deltas = seed_run.setdefault("delta_vs_hmtl", {})
    for baseline_name, baseline_metrics_any in baselines.items():
        baseline_metrics = _normalize_metrics_block(baseline_metrics_any)
        if baseline_metrics is None:
            continue
        if "rmse" not in baseline_metrics or "r_auc_mse" not in baseline_metrics:
            continue
        if "rmse" not in hmtl_metrics or "r_auc_mse" not in hmtl_metrics:
            continue
        deltas[str(baseline_name)] = _compute_delta_vs_hmtl(
            baseline_metrics=baseline_metrics,
            hmtl_metrics=hmtl_metrics,
        )


def _merge_seed_runs(left_seed: dict[str, Any] | None, right_seed: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}

    if isinstance(left_seed, dict):
        merged = copy.deepcopy(left_seed)
    if isinstance(right_seed, dict):
        for key, value in right_seed.items():
            if key in {"baselines", "delta_vs_hmtl"}:
                continue
            if key == "hmtl":
                if isinstance(value, dict) and "error" not in value:
                    merged[key] = copy.deepcopy(value)
                elif merged.get(key) is None:
                    merged[key] = copy.deepcopy(value)
                continue
            merged[key] = copy.deepcopy(value)

    merged_baselines: dict[str, Any] = {}
    if isinstance(left_seed, dict) and isinstance(left_seed.get("baselines"), dict):
        merged_baselines.update(copy.deepcopy(left_seed["baselines"]))
    if isinstance(right_seed, dict) and isinstance(right_seed.get("baselines"), dict):
        merged_baselines.update(copy.deepcopy(right_seed["baselines"]))
    merged["baselines"] = merged_baselines

    merged_deltas: dict[str, Any] = {}
    if isinstance(left_seed, dict) and isinstance(left_seed.get("delta_vs_hmtl"), dict):
        merged_deltas.update(copy.deepcopy(left_seed["delta_vs_hmtl"]))
    if isinstance(right_seed, dict) and isinstance(right_seed.get("delta_vs_hmtl"), dict):
        merged_deltas.update(copy.deepcopy(right_seed["delta_vs_hmtl"]))
    merged["delta_vs_hmtl"] = merged_deltas

    left_status = left_seed.get("status") if isinstance(left_seed, dict) else None
    right_status = right_seed.get("status") if isinstance(right_seed, dict) else None
    if left_status == "ok" or right_status == "ok":
        merged["status"] = "ok"
    elif right_status is not None:
        merged["status"] = right_status
    elif left_status is not None:
        merged["status"] = left_status

    _refresh_seed_deltas(merged)
    return merged


def _aggregate_size_seed_runs(per_seed_runs: list[dict[str, Any]], baselines: list[str]) -> dict[str, Any]:
    n_requested = len(per_seed_runs)
    successful_runs = [run for run in per_seed_runs if run.get("status") == "ok"]
    hmtl_success_runs = [
        run
        for run in successful_runs
        if isinstance(run.get("hmtl"), dict) and "error" not in run.get("hmtl", {})
    ]
    failed_seeds = [run.get("seed") for run in per_seed_runs if run.get("status") != "ok"]

    hmtl_metric_dicts = [
        _normalize_metrics_block(run.get("hmtl"))
        for run in hmtl_success_runs
    ]
    hmtl_metric_dicts = [metrics for metrics in hmtl_metric_dicts if metrics]
    hmtl_agg = _aggregate_metric_dicts(hmtl_metric_dicts)
    hmtl_means = _extract_mean_metrics(hmtl_agg)

    baselines_agg: dict[str, Any] = {}
    baselines_means: dict[str, dict[str, float]] = {}
    delta_agg: dict[str, Any] = {}
    delta_means: dict[str, dict[str, float]] = {}

    for baseline_name in baselines:
        baseline_success: list[dict[str, float]] = []
        baseline_delta_success: list[dict[str, float]] = []

        for run in successful_runs:
            baseline_metrics = _normalize_metrics_block(
                run.get("baselines", {}).get(baseline_name)
            )
            if baseline_metrics:
                baseline_success.append(baseline_metrics)

            delta_metrics = _normalize_metrics_block(
                run.get("delta_vs_hmtl", {}).get(baseline_name)
            )
            if delta_metrics:
                baseline_delta_success.append(delta_metrics)

        agg_metrics = _aggregate_metric_dicts(baseline_success)
        agg_delta = _aggregate_metric_dicts(baseline_delta_success)

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
        "n_successful": int(len(successful_runs)),
        "n_successful_hmtl": int(len(hmtl_success_runs)),
        "failed_seeds": failed_seeds,
        "hmtl": hmtl_agg,
        "baselines": baselines_agg,
        "delta_vs_hmtl": delta_agg,
    }

    size_summary: dict[str, Any] = {
        "status": "ok" if successful_runs else "failed",
        "aggregate_over_seeds": aggregate_over_seeds,
        "hmtl": hmtl_means,
        "baselines": baselines_means,
        "delta_vs_hmtl": delta_means,
    }

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


def _discover_size_baselines(size_data: dict[str, Any], per_seed_map: dict[str, dict[str, Any]]) -> list[str]:
    baselines: list[str] = []

    def _extend(values: list[str]) -> None:
        for value in values:
            if value not in baselines:
                baselines.append(value)

    legacy = size_data.get("baselines")
    if isinstance(legacy, dict):
        _extend([str(name) for name in legacy.keys()])

    aggregate = size_data.get("aggregate_over_seeds", {})
    if isinstance(aggregate, dict) and isinstance(aggregate.get("baselines"), dict):
        _extend([str(name) for name in aggregate["baselines"].keys()])

    for seed_run in per_seed_map.values():
        if isinstance(seed_run, dict) and isinstance(seed_run.get("baselines"), dict):
            _extend([str(name) for name in seed_run["baselines"].keys()])

    return baselines


def _merge_size_data(left_size: dict[str, Any] | None, right_size: dict[str, Any] | None) -> dict[str, Any]:
    merged_size: dict[str, Any] = {}
    if isinstance(left_size, dict):
        merged_size = copy.deepcopy(left_size)
    if isinstance(right_size, dict):
        for key, value in right_size.items():
            if key in {"per_seed", "aggregate_over_seeds", "hmtl", "baselines", "delta_vs_hmtl"}:
                continue
            merged_size[key] = copy.deepcopy(value)

    left_per_seed = left_size.get("per_seed", {}) if isinstance(left_size, dict) else {}
    right_per_seed = right_size.get("per_seed", {}) if isinstance(right_size, dict) else {}
    if not isinstance(left_per_seed, dict):
        left_per_seed = {}
    if not isinstance(right_per_seed, dict):
        right_per_seed = {}

    seed_keys = set(left_per_seed.keys()) | set(right_per_seed.keys())
    merged_per_seed: dict[str, dict[str, Any]] = {}
    for seed_key in sorted(seed_keys, key=_seed_sort_key):
        merged_per_seed[str(seed_key)] = _merge_seed_runs(
            left_seed=left_per_seed.get(seed_key),
            right_seed=right_per_seed.get(seed_key),
        )

    baselines = _discover_size_baselines(left_size or {}, merged_per_seed)
    for baseline in _discover_size_baselines(right_size or {}, merged_per_seed):
        if baseline not in baselines:
            baselines.append(baseline)

    per_seed_runs = [merged_per_seed[key] for key in sorted(merged_per_seed.keys(), key=_seed_sort_key)]
    rebuilt_summary = _aggregate_size_seed_runs(per_seed_runs=per_seed_runs, baselines=baselines)

    merged_size["per_seed"] = merged_per_seed
    merged_size.update(rebuilt_summary)
    return merged_size


def _dataset_key(dataset_result: dict[str, Any]) -> str:
    dataset_id = dataset_result.get("dataset_id")
    if dataset_id is None:
        return f"name:{dataset_result.get('dataset_name', 'unknown')}"
    return f"id:{dataset_id}"


def _merge_run_meta(left_meta: dict[str, Any], right_meta: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(left_meta)
    merged.update(copy.deepcopy(right_meta))

    baselines: list[str] = []
    for source in (left_meta, right_meta):
        values = source.get("baselines")
        if isinstance(values, list):
            for baseline in values:
                baseline_name = str(baseline)
                if baseline_name not in baselines:
                    baselines.append(baseline_name)
    merged["baselines"] = baselines

    left_hmtl = bool(left_meta.get("hmtl_enabled", True))
    right_hmtl = bool(right_meta.get("hmtl_enabled", True))
    merged["hmtl_enabled"] = left_hmtl or right_hmtl
    return merged


def _merge_dataset_results(left_ds: dict[str, Any] | None, right_ds: dict[str, Any] | None) -> dict[str, Any]:
    if left_ds is None:
        return copy.deepcopy(right_ds) if isinstance(right_ds, dict) else {}
    if right_ds is None:
        return copy.deepcopy(left_ds)

    merged = copy.deepcopy(left_ds)
    for key, value in right_ds.items():
        if key in {"sizes", "run_meta"}:
            continue
        merged[key] = copy.deepcopy(value)

    left_meta = left_ds.get("run_meta", {})
    right_meta = right_ds.get("run_meta", {})
    if isinstance(left_meta, dict) and isinstance(right_meta, dict):
        merged["run_meta"] = _merge_run_meta(left_meta, right_meta)
    elif isinstance(right_meta, dict):
        merged["run_meta"] = copy.deepcopy(right_meta)
    elif isinstance(left_meta, dict):
        merged["run_meta"] = copy.deepcopy(left_meta)

    left_sizes = left_ds.get("sizes", {})
    right_sizes = right_ds.get("sizes", {})
    if not isinstance(left_sizes, dict):
        left_sizes = {}
    if not isinstance(right_sizes, dict):
        right_sizes = {}

    merged_sizes: dict[str, Any] = {}
    size_keys = set(left_sizes.keys()) | set(right_sizes.keys())
    for size_key in sorted(size_keys, key=lambda value: _seed_sort_key(value)):
        merged_sizes[str(size_key)] = _merge_size_data(
            left_size=left_sizes.get(size_key),
            right_size=right_sizes.get(size_key),
        )

    merged["sizes"] = merged_sizes
    return merged


def _dataset_sort_key(dataset: dict[str, Any]) -> tuple[int, int | str]:
    return _seed_sort_key(dataset.get("dataset_id"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge two AutoMLBenchmark aggregated_results.json files")
    parser.add_argument("--first", required=True, help="Path to the first aggregated_results.json")
    parser.add_argument("--second", required=True, help="Path to the second aggregated_results.json")
    parser.add_argument("--output", required=True, help="Path for merged aggregated_results.json")
    args = parser.parse_args()

    first_path = Path(args.first)
    second_path = Path(args.second)
    output_path = Path(args.output)

    first_results = _load_results(first_path)
    second_results = _load_results(second_path)

    first_by_key = {_dataset_key(dataset): dataset for dataset in first_results}
    second_by_key = {_dataset_key(dataset): dataset for dataset in second_results}

    merged: list[dict[str, Any]] = []
    for key in sorted(set(first_by_key.keys()) | set(second_by_key.keys())):
        merged.append(
            _merge_dataset_results(
                left_ds=first_by_key.get(key),
                right_ds=second_by_key.get(key),
            )
        )

    merged = sorted(merged, key=_dataset_sort_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(merged, file, indent=2)

    print(f"Merged {len(merged)} dataset entries into {output_path}")


if __name__ == "__main__":
    main()
