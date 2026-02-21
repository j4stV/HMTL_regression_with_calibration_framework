#!/usr/bin/env python3
"""Analyze size dependence of delta metrics (baseline - HMTL)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger, setup_logging


logger = get_logger("analyze_size_dependence")

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (11, 7)


def load_results(results_file: Path) -> list[dict[str, Any]]:
    with open(results_file, "r", encoding="utf-8") as file:
        loaded = json.load(file)

    if isinstance(loaded, list):
        return loaded
    if isinstance(loaded, dict) and isinstance(loaded.get("results"), list):
        return loaded["results"]

    raise ValueError("Unsupported results JSON format: expected list or {'results': [...]}.")


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        casted = float(value)
        if not np.isfinite(casted):
            return None
        return casted
    except Exception:
        return None


def _extract_from_aggregate_block(size_data: dict[str, Any], baseline: str) -> dict[str, float | None]:
    aggregate = size_data.get("aggregate_over_seeds", {})

    hmtl = aggregate.get("hmtl", {})
    baselines = aggregate.get("baselines", {})
    deltas = aggregate.get("delta_vs_hmtl", {})

    baseline_block = baselines.get(baseline, {})
    delta_block = deltas.get(baseline, {})

    return {
        "hmtl_rmse": _safe_float(hmtl.get("rmse", {}).get("mean")),
        "hmtl_r_auc_mse": _safe_float(hmtl.get("r_auc_mse", {}).get("mean")),
        "baseline_rmse": _safe_float(baseline_block.get("rmse", {}).get("mean")),
        "baseline_r_auc_mse": _safe_float(baseline_block.get("r_auc_mse", {}).get("mean")),
        "delta_rmse": _safe_float(delta_block.get("delta_rmse", {}).get("mean")),
        "delta_r_auc_mse": _safe_float(delta_block.get("delta_r_auc_mse", {}).get("mean")),
    }


def _extract_legacy_block(size_data: dict[str, Any], baseline: str) -> dict[str, float | None]:
    baseline_data = size_data.get("catboost", {}) if baseline == "catboost" else size_data.get("baselines", {}).get(baseline, {})
    hmtl_data = size_data.get("hmtl", {})

    return {
        "hmtl_rmse": _safe_float(hmtl_data.get("rmse")),
        "hmtl_r_auc_mse": _safe_float(hmtl_data.get("r_auc_mse")),
        "baseline_rmse": _safe_float(baseline_data.get("rmse")),
        "baseline_r_auc_mse": _safe_float(baseline_data.get("r_auc_mse")),
        "delta_rmse": _safe_float(size_data.get("delta_rmse")),
        "delta_r_auc_mse": _safe_float(size_data.get("delta_r_auc_mse")),
    }


def extract_metrics_by_size(results: list[dict[str, Any]], baseline: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for dataset_result in results:
        if "error" in dataset_result:
            continue

        dataset_id = dataset_result.get("dataset_id")
        dataset_name = dataset_result.get("dataset_name")

        sizes = dataset_result.get("sizes", {})
        if not isinstance(sizes, dict):
            continue

        for size_key, size_data in sizes.items():
            if not isinstance(size_data, dict) or "error" in size_data:
                continue

            size_ratio = _safe_float(size_data.get("size_ratio"))
            if size_ratio is None:
                try:
                    size_ratio = float(size_key) / 100.0
                except Exception:
                    continue

            if "aggregate_over_seeds" in size_data:
                extracted = _extract_from_aggregate_block(size_data, baseline=baseline)
            else:
                extracted = _extract_legacy_block(size_data, baseline=baseline)

            # Backfill deltas if only raw metrics are present.
            if extracted["delta_rmse"] is None and extracted["baseline_rmse"] is not None and extracted["hmtl_rmse"] is not None:
                extracted["delta_rmse"] = float(extracted["baseline_rmse"] - extracted["hmtl_rmse"])
            if (
                extracted["delta_r_auc_mse"] is None
                and extracted["baseline_r_auc_mse"] is not None
                and extracted["hmtl_r_auc_mse"] is not None
            ):
                extracted["delta_r_auc_mse"] = float(
                    extracted["baseline_r_auc_mse"] - extracted["hmtl_r_auc_mse"]
                )

            rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    "size_ratio": float(size_ratio),
                    "hmtl_rmse": extracted["hmtl_rmse"],
                    f"{baseline}_rmse": extracted["baseline_rmse"],
                    "hmtl_r_auc_mse": extracted["hmtl_r_auc_mse"],
                    f"{baseline}_r_auc_mse": extracted["baseline_r_auc_mse"],
                    "delta_rmse": extracted["delta_rmse"],
                    "delta_r_auc_mse": extracted["delta_r_auc_mse"],
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["size_ratio", "delta_rmse", "delta_r_auc_mse"], how="any")

    logger.info(
        "Extracted %d rows from %d datasets",
        len(df),
        int(df["dataset_id"].nunique()) if not df.empty else 0,
    )
    return df


def _one_sample_ttest_greater(values: np.ndarray) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    try:
        test = stats.ttest_1samp(values, popmean=0.0, alternative="greater")
        return float(test.statistic), float(test.pvalue)
    except TypeError:
        # Fallback for old SciPy versions without "alternative".
        test = stats.ttest_1samp(values, popmean=0.0)
        if test.statistic is None or test.pvalue is None:
            return None, None
        p_one_sided = float(test.pvalue / 2.0) if float(test.statistic) > 0 else 1.0
        return float(test.statistic), p_one_sided


def _wilcoxon_greater(values: np.ndarray) -> tuple[float | None, float | None]:
    if len(values) < 1:
        return None, None
    if np.allclose(values, 0.0):
        return None, None
    try:
        statistic, p_value = stats.wilcoxon(values, zero_method="wilcox", alternative="greater")
        return float(statistic), float(p_value)
    except Exception:
        return None, None


def _effect_size(values: np.ndarray) -> float | None:
    if len(values) < 2:
        return None
    std = float(np.std(values, ddof=1))
    if std < 1e-12:
        return None
    return float(np.mean(values) / std)


def perform_statistical_analysis(df: pd.DataFrame) -> dict[str, Any]:
    analysis: dict[str, Any] = {}

    for size_ratio in sorted(df["size_ratio"].unique()):
        subset = df[df["size_ratio"] == size_ratio]
        deltas_rmse = subset["delta_rmse"].astype(float).values
        deltas_r_auc = subset["delta_r_auc_mse"].astype(float).values

        t_rmse, p_rmse = _one_sample_ttest_greater(deltas_rmse)
        w_rmse, pw_rmse = _wilcoxon_greater(deltas_rmse)

        t_rauc, p_rauc = _one_sample_ttest_greater(deltas_r_auc)
        w_rauc, pw_rauc = _wilcoxon_greater(deltas_r_auc)

        analysis[str(size_ratio)] = {
            "size_ratio": float(size_ratio),
            "n_datasets": int(len(subset)),
            "mean_delta_rmse": float(np.mean(deltas_rmse)),
            "std_delta_rmse": float(np.std(deltas_rmse, ddof=1)) if len(deltas_rmse) > 1 else 0.0,
            "mean_delta_r_auc_mse": float(np.mean(deltas_r_auc)),
            "std_delta_r_auc_mse": float(np.std(deltas_r_auc, ddof=1)) if len(deltas_r_auc) > 1 else 0.0,
            "ttest_greater_rmse": {
                "statistic": t_rmse,
                "p_value": p_rmse,
            },
            "wilcoxon_greater_rmse": {
                "statistic": w_rmse,
                "p_value": pw_rmse,
            },
            "ttest_greater_r_auc_mse": {
                "statistic": t_rauc,
                "p_value": p_rauc,
            },
            "wilcoxon_greater_r_auc_mse": {
                "statistic": w_rauc,
                "p_value": pw_rauc,
            },
            "cohens_d_rmse": _effect_size(deltas_rmse),
            "cohens_d_r_auc_mse": _effect_size(deltas_r_auc),
            "n_favor_hmtl_rmse": int(np.sum(deltas_rmse > 0)),
            "n_favor_hmtl_r_auc_mse": int(np.sum(deltas_r_auc > 0)),
        }

    return analysis


def compute_dataset_slopes(
    df: pd.DataFrame,
    *,
    metric_col: str,
    min_size_points: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for dataset_id, dataset_df in df.groupby("dataset_id"):
        ordered = dataset_df.sort_values("size_ratio")
        ordered = ordered.dropna(subset=[metric_col])

        if len(ordered) < min_size_points or ordered["size_ratio"].nunique() < min_size_points:
            continue

        x = ordered["size_ratio"].astype(float).values
        y = ordered[metric_col].astype(float).values
        slope, intercept = np.polyfit(x, y, deg=1)

        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_name": ordered["dataset_name"].iloc[0],
                "metric": metric_col,
                "n_points": int(len(ordered)),
                "slope": float(slope),
                "intercept": float(intercept),
            }
        )

    return pd.DataFrame(rows)


def analyze_slopes(
    slope_df: pd.DataFrame,
    *,
    min_datasets: int,
) -> dict[str, Any]:
    if slope_df.empty:
        return {
            "n_datasets": 0,
            "mean_slope": None,
            "std_slope": None,
            "ttest_greater": {"statistic": None, "p_value": None},
            "wilcoxon_greater": {"statistic": None, "p_value": None},
            "insufficient": True,
        }

    slopes = slope_df["slope"].astype(float).values
    t_stat, p_t = _one_sample_ttest_greater(slopes)
    w_stat, p_w = _wilcoxon_greater(slopes)

    return {
        "n_datasets": int(len(slopes)),
        "mean_slope": float(np.mean(slopes)),
        "std_slope": float(np.std(slopes, ddof=1)) if len(slopes) > 1 else 0.0,
        "ttest_greater": {"statistic": t_stat, "p_value": p_t},
        "wilcoxon_greater": {"statistic": w_stat, "p_value": p_w},
        "insufficient": bool(len(slopes) < min_datasets),
    }


def determine_verdict(
    *,
    slope_stats: dict[str, Any],
    n_unique_sizes: int,
    min_datasets: int,
    min_size_points: int,
    alpha: float,
) -> str:
    if n_unique_sizes < min_size_points:
        return "insufficient_evidence"

    if slope_stats.get("n_datasets", 0) < min_datasets:
        return "insufficient_evidence"

    mean_slope = slope_stats.get("mean_slope")
    if mean_slope is None:
        return "insufficient_evidence"

    p_t = slope_stats.get("ttest_greater", {}).get("p_value")
    p_w = slope_stats.get("wilcoxon_greater", {}).get("p_value")

    if p_t is None or p_w is None:
        return "insufficient_evidence"

    if mean_slope > 0 and p_t < alpha and p_w < alpha:
        return "supported"

    return "not_supported"


def visualize_results(df: pd.DataFrame, by_size: dict[str, Any], output_dir: Path) -> None:
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    sizes = sorted(df["size_ratio"].unique())

    fig, ax = plt.subplots()
    means_rmse = [by_size[str(s)]["mean_delta_rmse"] for s in sizes]
    std_rmse = [by_size[str(s)]["std_delta_rmse"] for s in sizes]
    ax.errorbar([s * 100 for s in sizes], means_rmse, yerr=std_rmse, marker="o", linewidth=2)
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Dataset size (%)")
    ax.set_ylabel("ΔRMSE (baseline - HMTL)")
    ax.set_title("ΔRMSE by dataset size")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / "delta_rmse_vs_size.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    means_rauc = [by_size[str(s)]["mean_delta_r_auc_mse"] for s in sizes]
    std_rauc = [by_size[str(s)]["std_delta_r_auc_mse"] for s in sizes]
    ax.errorbar(
        [s * 100 for s in sizes],
        means_rauc,
        yerr=std_rauc,
        marker="o",
        linewidth=2,
        color="darkorange",
    )
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Dataset size (%)")
    ax.set_ylabel("ΔR-AUC MSE (baseline - HMTL)")
    ax.set_title("ΔR-AUC MSE by dataset size")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / "delta_r_auc_mse_vs_size.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    df_plot = df.copy()
    df_plot["size_pct"] = (df_plot["size_ratio"] * 100).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.boxplot(data=df_plot, x="size_pct", y="delta_rmse", ax=axes[0])
    axes[0].axhline(0, color="red", linestyle="--", alpha=0.5)
    axes[0].set_title("ΔRMSE distribution by size")
    axes[0].set_xlabel("Dataset size (%)")
    axes[0].set_ylabel("ΔRMSE")

    sns.boxplot(data=df_plot, x="size_pct", y="delta_r_auc_mse", ax=axes[1])
    axes[1].axhline(0, color="red", linestyle="--", alpha=0.5)
    axes[1].set_title("ΔR-AUC MSE distribution by size")
    axes[1].set_xlabel("Dataset size (%)")
    axes[1].set_ylabel("ΔR-AUC MSE")

    plt.tight_layout()
    plt.savefig(viz_dir / "boxplots_by_size.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def analyze_size_dependence(
    *,
    results_file: Path,
    output_dir: Path,
    baseline: str,
    min_datasets: int,
    min_size_points: int,
    alpha: float,
) -> dict[str, Any]:
    results = load_results(results_file)
    df = extract_metrics_by_size(results, baseline=baseline)

    if df.empty:
        report = {
            "summary": {
                "n_datasets": 0,
                "n_data_points": 0,
                "sizes_tested": [],
                "baseline": baseline,
            },
            "status": "insufficient_evidence",
            "reason": "no_valid_rows",
            "by_size": {},
            "slope_analysis": {},
            "verdict": {
                "delta_rmse": "insufficient_evidence",
                "delta_r_auc_mse": "insufficient_evidence",
            },
        }
        return report

    by_size = perform_statistical_analysis(df)

    slope_rmse_df = compute_dataset_slopes(
        df,
        metric_col="delta_rmse",
        min_size_points=min_size_points,
    )
    slope_rauc_df = compute_dataset_slopes(
        df,
        metric_col="delta_r_auc_mse",
        min_size_points=min_size_points,
    )

    slope_rmse_stats = analyze_slopes(slope_rmse_df, min_datasets=min_datasets)
    slope_rauc_stats = analyze_slopes(slope_rauc_df, min_datasets=min_datasets)

    n_unique_sizes = int(df["size_ratio"].nunique())

    verdict_rmse = determine_verdict(
        slope_stats=slope_rmse_stats,
        n_unique_sizes=n_unique_sizes,
        min_datasets=min_datasets,
        min_size_points=min_size_points,
        alpha=alpha,
    )
    verdict_rauc = determine_verdict(
        slope_stats=slope_rauc_stats,
        n_unique_sizes=n_unique_sizes,
        min_datasets=min_datasets,
        min_size_points=min_size_points,
        alpha=alpha,
    )

    overall_status = (
        "insufficient_evidence"
        if verdict_rmse == "insufficient_evidence" and verdict_rauc == "insufficient_evidence"
        else "completed"
    )

    report = {
        "summary": {
            "n_datasets": int(df["dataset_id"].nunique()),
            "n_data_points": int(len(df)),
            "sizes_tested": sorted(float(v) for v in df["size_ratio"].unique()),
            "baseline": baseline,
            "min_datasets_required": int(min_datasets),
            "min_size_points_required": int(min_size_points),
            "alpha": float(alpha),
        },
        "status": overall_status,
        "by_size": by_size,
        "slope_analysis": {
            "delta_rmse": {
                "stats": slope_rmse_stats,
                "per_dataset": slope_rmse_df.to_dict(orient="records"),
            },
            "delta_r_auc_mse": {
                "stats": slope_rauc_stats,
                "per_dataset": slope_rauc_df.to_dict(orient="records"),
            },
        },
        "verdict": {
            "delta_rmse": verdict_rmse,
            "delta_r_auc_mse": verdict_rauc,
        },
    }

    summary_rows = []
    for size_ratio, values in by_size.items():
        summary_rows.append(
            {
                "size_ratio": float(size_ratio),
                "n_datasets": values["n_datasets"],
                "mean_delta_rmse": values["mean_delta_rmse"],
                "std_delta_rmse": values["std_delta_rmse"],
                "p_ttest_greater_rmse": values["ttest_greater_rmse"]["p_value"],
                "p_wilcoxon_greater_rmse": values["wilcoxon_greater_rmse"]["p_value"],
                "mean_delta_r_auc_mse": values["mean_delta_r_auc_mse"],
                "std_delta_r_auc_mse": values["std_delta_r_auc_mse"],
                "p_ttest_greater_r_auc_mse": values["ttest_greater_r_auc_mse"]["p_value"],
                "p_wilcoxon_greater_r_auc_mse": values["wilcoxon_greater_r_auc_mse"]["p_value"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    report_file = output_dir / "statistical_analysis.json"
    with open(report_file, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    summary_file = output_dir / "summary_by_size.csv"
    pd.DataFrame(summary_rows).to_csv(summary_file, index=False)

    visualize_results(df, by_size=by_size, output_dir=output_dir)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze size dependence of HMTL performance")
    parser.add_argument("--results", required=True, help="Path to aggregated_results.json")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: same directory as results file)",
    )
    parser.add_argument(
        "--baseline",
        default="catboost",
        help="Which baseline to analyze against HMTL",
    )
    parser.add_argument(
        "--min-datasets",
        type=int,
        default=3,
        help="Minimum number of datasets required for verdict",
    )
    parser.add_argument(
        "--min-size-points",
        type=int,
        default=3,
        help="Minimum size points per dataset for slope analysis",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for statistical tests",
    )

    args = parser.parse_args()

    setup_logging()

    results_path = Path(args.results)
    output_path = Path(args.output) if args.output else results_path.parent

    report = analyze_size_dependence(
        results_file=results_path,
        output_dir=output_path,
        baseline=args.baseline,
        min_datasets=args.min_datasets,
        min_size_points=args.min_size_points,
        alpha=args.alpha,
    )

    logger.info("Analysis complete. Verdict ΔRMSE: %s", report["verdict"]["delta_rmse"])
    logger.info(
        "Analysis complete. Verdict ΔR-AUC MSE: %s",
        report["verdict"]["delta_r_auc_mse"],
    )


if __name__ == "__main__":
    main()
