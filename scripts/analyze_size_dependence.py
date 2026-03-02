#!/usr/bin/env python3
"""Analyze dependence of baseline-vs-HMTL deltas on dataset size and n_features.

Produces a single self-contained HTML report with interactive Plotly charts.
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger, setup_logging


logger = get_logger("analyze_size_dependence")

METRICS = ("delta_rmse", "delta_r_auc_mse")
FULL_SIZE_RATIO = 1.0
METRIC_LABELS = {
    "delta_rmse": "ΔRMSE",
    "delta_r_auc_mse": "ΔR-AUC MSE",
}
METRIC_ANALYSIS_KEYS = {
    "delta_rmse": {
        "suffix": "rmse",
        "mean_col": "mean_delta_rmse",
        "std_col": "std_delta_rmse",
        "cohens_col": "cohens_d_rmse",
        "favor_col": "n_favor_hmtl_rmse",
    },
    "delta_r_auc_mse": {
        "suffix": "r_auc_mse",
        "mean_col": "mean_delta_r_auc_mse",
        "std_col": "std_delta_r_auc_mse",
        "cohens_col": "cohens_d_r_auc_mse",
        "favor_col": "n_favor_hmtl_r_auc_mse",
    },
}
MODEL_TERM_TO_FACTOR = {
    "log_n_samples": "n_samples_absolute",
    "size_ratio": "size_ratio",
    "log_n_features": "n_features",
}
FACTOR_LABELS = {
    "n_samples_absolute": "log1p(n_samples)",
    "size_ratio": "size_ratio",
    "n_features": "log1p(n_features)",
}


def load_results(results_file: Path) -> list[dict[str, Any]]:
    with open(results_file, "r", encoding="utf-8") as file:
        loaded = json.load(file)

    if isinstance(loaded, list):
        return loaded
    if isinstance(loaded, dict) and isinstance(loaded.get("results"), list):
        return loaded["results"]

    raise ValueError("Unsupported results JSON format: expected list or {'results': [...]} .")


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


def _discover_available_baselines(results: list[dict[str, Any]]) -> list[str]:
    baselines: set[str] = set()

    for dataset_result in results:
        sizes = dataset_result.get("sizes", {})
        if not isinstance(sizes, dict):
            continue

        for size_data in sizes.values():
            if not isinstance(size_data, dict):
                continue

            aggregate = size_data.get("aggregate_over_seeds")
            if isinstance(aggregate, dict):
                agg_baselines = aggregate.get("baselines")
                if isinstance(agg_baselines, dict):
                    baselines.update(str(name) for name in agg_baselines.keys())

            legacy_baselines = size_data.get("baselines")
            if isinstance(legacy_baselines, dict):
                baselines.update(str(name) for name in legacy_baselines.keys())

            if isinstance(size_data.get("catboost"), dict):
                baselines.add("catboost")

    return sorted(baselines)


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
    baseline_data = (
        size_data.get("catboost", {})
        if baseline == "catboost"
        else size_data.get("baselines", {}).get(baseline, {})
    )
    hmtl_data = size_data.get("hmtl", {})

    return {
        "hmtl_rmse": _safe_float(hmtl_data.get("rmse")),
        "hmtl_r_auc_mse": _safe_float(hmtl_data.get("r_auc_mse")),
        "baseline_rmse": _safe_float(baseline_data.get("rmse")),
        "baseline_r_auc_mse": _safe_float(baseline_data.get("r_auc_mse")),
        "delta_rmse": _safe_float(size_data.get("delta_rmse")),
        "delta_r_auc_mse": _safe_float(size_data.get("delta_r_auc_mse")),
    }


def extract_metrics_long_form(results: list[dict[str, Any]], baseline: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for dataset_result in results:
        if "error" in dataset_result:
            continue

        dataset_id = dataset_result.get("dataset_id")
        dataset_name = dataset_result.get("dataset_name")
        n_features = _safe_float(dataset_result.get("n_features"))
        dataset_train_samples = _safe_float(dataset_result.get("n_samples_train"))
        dataset_total_samples = _safe_float(dataset_result.get("n_samples_total"))

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

            n_samples = _safe_float(size_data.get("n_train_samples"))
            if n_samples is None:
                n_samples = _safe_float(size_data.get("n_samples_train"))
            if n_samples is None:
                n_samples = _safe_float(size_data.get("n_samples"))
            if n_samples is None and dataset_train_samples is not None:
                n_samples = float(dataset_train_samples * size_ratio)
            if n_samples is None and dataset_total_samples is not None:
                n_samples = float(dataset_total_samples * size_ratio)

            if (
                extracted["delta_rmse"] is None
                and extracted["baseline_rmse"] is not None
                and extracted["hmtl_rmse"] is not None
            ):
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
                    "size_pct": float(size_ratio) * 100.0,
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "hmtl_rmse": extracted["hmtl_rmse"],
                    f"{baseline}_rmse": extracted["baseline_rmse"],
                    "hmtl_r_auc_mse": extracted["hmtl_r_auc_mse"],
                    f"{baseline}_r_auc_mse": extracted["baseline_r_auc_mse"],
                    "delta_rmse": extracted["delta_rmse"],
                    "delta_r_auc_mse": extracted["delta_r_auc_mse"],
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.dropna(subset=["size_ratio", "delta_rmse", "delta_r_auc_mse"])
    df["dataset_id"] = df["dataset_id"].astype(str)
    return df


def _one_sample_ttest_greater(values: np.ndarray) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    try:
        test = stats.ttest_1samp(values, popmean=0.0, alternative="greater")
        return float(test.statistic), float(test.pvalue)
    except TypeError:
        test = stats.ttest_1samp(values, popmean=0.0)
        if test.statistic is None or test.pvalue is None:
            return None, None
        p_one_sided = float(test.pvalue / 2.0) if float(test.statistic) > 0 else 1.0
        return float(test.statistic), p_one_sided


def _wilcoxon_greater(values: np.ndarray) -> tuple[float | None, float | None]:
    if len(values) < 1 or np.allclose(values, 0.0):
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


def _overall_delta_significance(values: np.ndarray, *, alpha: float) -> dict[str, Any]:
    n_points = int(len(values))
    result = {
        "n_points": n_points,
        "mean_delta": None,
        "std_delta": None,
        "ci_low": None,
        "ci_high": None,
        "t_stat": None,
        "p_value_two_sided": None,
        "p_value_one_sided_greater": None,
        "verdict": "insufficient_evidence",
    }
    if n_points < 2:
        return result

    mean_delta = float(np.mean(values))
    std_delta = float(np.std(values, ddof=1))
    result["mean_delta"] = mean_delta
    result["std_delta"] = std_delta

    t_stat = None
    p_two = None
    try:
        ttest = stats.ttest_1samp(values, popmean=0.0)
        t_stat = _safe_float(ttest.statistic)
        p_two = _safe_float(ttest.pvalue)
    except Exception:
        t_stat = None
        p_two = None
    p_one = _one_sided_greater_from_two_sided(t_stat=t_stat, p_two_sided=p_two)

    result["t_stat"] = t_stat
    result["p_value_two_sided"] = p_two
    result["p_value_one_sided_greater"] = p_one

    if np.isfinite(std_delta) and std_delta >= 0.0:
        sem = std_delta / np.sqrt(float(n_points))
        if sem >= 0.0:
            try:
                t_crit = float(stats.t.ppf(1.0 - (alpha / 2.0), df=n_points - 1))
                if np.isfinite(t_crit):
                    result["ci_low"] = float(mean_delta - (t_crit * sem))
                    result["ci_high"] = float(mean_delta + (t_crit * sem))
            except Exception:
                pass

    if p_one is not None and mean_delta > 0.0 and p_one < alpha:
        result["verdict"] = "supported"
    else:
        result["verdict"] = "not_supported"
    return result


def _format_interval(interval: pd.Interval) -> str:
    return f"{interval.left:.0f}-{interval.right:.0f}"


def _assign_feature_bins(dataset_df: pd.DataFrame, max_bins: int = 4) -> pd.DataFrame:
    working = dataset_df[["dataset_id", "n_features"]].dropna().drop_duplicates().copy()
    if working.empty:
        working["feature_bin"] = pd.Series(dtype="object")
        return working

    n_unique = int(working["n_features"].nunique())
    if n_unique <= 1:
        working["feature_bin"] = f"{int(round(float(working['n_features'].iloc[0])))}"
        return working

    q = min(max_bins, n_unique)
    bins = pd.qcut(working["n_features"], q=q, duplicates="drop")
    working["feature_bin"] = bins.map(_format_interval).astype(str)
    return working


def _relationship_analysis(
    *,
    x: np.ndarray,
    y: np.ndarray,
    min_datasets: int,
    alpha: float,
    positive_effect_required: bool,
) -> dict[str, Any]:
    n_points = int(len(x))
    if n_points < min_datasets:
        return {
            "n_points": n_points,
            "spearman": {"rho": None, "p_value": None},
            "linear": {"slope": None, "intercept": None, "r_value": None, "p_value": None, "std_err": None},
            "verdict": "insufficient_evidence",
        }

    rho, p_spearman = stats.spearmanr(x, y)
    lin = stats.linregress(x, y)
    slope = float(lin.slope)
    p_linear = float(lin.pvalue)
    rho_f = None if not np.isfinite(rho) else float(rho)
    p_spearman_f = None if not np.isfinite(p_spearman) else float(p_spearman)

    passes = p_linear < alpha
    if positive_effect_required:
        passes = passes and slope > 0.0
    if rho_f is not None and p_spearman_f is not None:
        if positive_effect_required:
            passes = passes and rho_f > 0.0 and p_spearman_f < alpha
        else:
            passes = passes and p_spearman_f < alpha
    else:
        passes = False

    verdict = "supported" if passes else "not_supported"

    return {
        "n_points": n_points,
        "spearman": {"rho": rho_f, "p_value": p_spearman_f},
        "linear": {
            "slope": slope,
            "intercept": float(lin.intercept),
            "r_value": float(lin.rvalue),
            "p_value": p_linear,
            "std_err": float(lin.stderr),
        },
        "verdict": verdict,
    }


def _require_statsmodels_modules() -> tuple[Any, Any]:
    try:
        import statsmodels.api as sm  # type: ignore
        import statsmodels.formula.api as smf  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "statsmodels is required for hypotheses and factorial analysis. "
            "Install dependencies from requirements.txt or run: pip install statsmodels"
        ) from exc
    return sm, smf


def _one_sided_greater_from_two_sided(*, t_stat: float | None, p_two_sided: float | None) -> float | None:
    if t_stat is None or p_two_sided is None:
        return None
    p_two = float(max(0.0, min(1.0, p_two_sided)))
    if t_stat >= 0.0:
        return p_two / 2.0
    return 1.0 - (p_two / 2.0)


def _build_factorial_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["n_samples", "size_ratio", "n_features"]
    working = df.dropna(subset=required_cols).copy()
    if working.empty:
        return working
    working["log_n_samples"] = np.log1p(working["n_samples"].astype(float))
    working["log_n_features"] = np.log1p(working["n_features"].astype(float))
    return working


def _empty_factorial_metric_result(*, formula: str, n_points: int, min_points_required: int) -> dict[str, Any]:
    return {
        "formula": formula,
        "n_points": n_points,
        "min_points_required": int(min_points_required),
        "degrees_of_freedom": None,
        "r_squared": None,
        "adjusted_r_squared": None,
        "f_statistic": None,
        "f_p_value": None,
        "intercept": {"coef": None, "std_err": None, "t_stat": None, "p_value": None, "ci_low": None, "ci_high": None},
        "factors": {
            factor: {
                "term": term,
                "coef": None,
                "std_err": None,
                "t_stat": None,
                "p_value": None,
                "ci_low": None,
                "ci_high": None,
            }
            for term, factor in MODEL_TERM_TO_FACTOR.items()
        },
        "significant_factors": [],
        "raw_output": {"summary": "", "anova_type_2": ""},
        "error": None,
        "verdict": "insufficient_evidence",
    }


def _factor_block_from_model(
    *,
    model: Any,
    param_name: str,
    conf_int_frame: pd.DataFrame,
) -> dict[str, float | None]:
    ci_low = None
    ci_high = None
    if param_name in conf_int_frame.index:
        ci_low = _safe_float(conf_int_frame.loc[param_name, 0])
        ci_high = _safe_float(conf_int_frame.loc[param_name, 1])
    return {
        "coef": _safe_float(model.params.get(param_name)),
        "std_err": _safe_float(model.bse.get(param_name)),
        "t_stat": _safe_float(model.tvalues.get(param_name)),
        "p_value": _safe_float(model.pvalues.get(param_name)),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def _build_factorial_metric_result(
    *,
    working_df: pd.DataFrame,
    metric_col: str,
    min_datasets: int,
    alpha: float,
    sm: Any,
    smf: Any,
) -> dict[str, Any]:
    formula = f"{metric_col} ~ log_n_samples + size_ratio + log_n_features"
    min_points_required = max(int(min_datasets), 5)
    metric_df = working_df.dropna(subset=[metric_col]).copy()
    n_points = int(len(metric_df))
    result = _empty_factorial_metric_result(
        formula=formula,
        n_points=n_points,
        min_points_required=min_points_required,
    )
    if n_points < min_points_required:
        return result

    try:
        fitted = smf.ols(formula=formula, data=metric_df).fit()
    except Exception as exc:
        result["error"] = str(exc)
        return result

    conf_int_frame = fitted.conf_int(alpha=alpha)
    result["degrees_of_freedom"] = _safe_float(fitted.df_resid)
    result["r_squared"] = _safe_float(fitted.rsquared)
    result["adjusted_r_squared"] = _safe_float(fitted.rsquared_adj)
    result["f_statistic"] = _safe_float(getattr(fitted, "fvalue", None))
    result["f_p_value"] = _safe_float(getattr(fitted, "f_pvalue", None))
    result["intercept"] = _factor_block_from_model(
        model=fitted,
        param_name="Intercept",
        conf_int_frame=conf_int_frame,
    )

    factors: dict[str, Any] = {}
    for term, factor in MODEL_TERM_TO_FACTOR.items():
        factor_block = _factor_block_from_model(
            model=fitted,
            param_name=term,
            conf_int_frame=conf_int_frame,
        )
        factor_block["term"] = term
        factors[factor] = factor_block

    significant_factors = [
        factor
        for factor, block in factors.items()
        if block.get("p_value") is not None and float(block["p_value"]) < float(alpha)
    ]
    result["factors"] = factors
    result["significant_factors"] = significant_factors
    result["raw_output"]["summary"] = fitted.summary().as_text()
    try:
        anova_frame = sm.stats.anova_lm(fitted, typ=2)
        result["raw_output"]["anova_type_2"] = anova_frame.to_string()
    except Exception as exc:
        result["raw_output"]["anova_type_2"] = f"ANOVA Type II unavailable: {exc}"

    result["verdict"] = "completed"
    return result


def _build_hypotheses_from_factorial(
    *,
    metric_col: str,
    factorial_metric_result: dict[str, Any],
    alpha: float,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    for factor, factor_label in FACTOR_LABELS.items():
        block = factorial_metric_result.get("factors", {}).get(factor, {})
        coef = _safe_float(block.get("coef"))
        t_stat = _safe_float(block.get("t_stat"))
        p_two = _safe_float(block.get("p_value"))
        p_one = _one_sided_greater_from_two_sided(t_stat=t_stat, p_two_sided=p_two)
        if coef is None or p_one is None:
            verdict = "insufficient_evidence"
        elif coef > 0.0 and p_one < alpha:
            verdict = "supported"
        else:
            verdict = "not_supported"

        checks[factor] = {
            "factor_label": factor_label,
            "hypothesis": f"{factor_label} coefficient > 0",
            "coef": coef,
            "t_stat": t_stat,
            "p_value_two_sided": p_two,
            "p_value_one_sided_greater": p_one,
            "verdict": verdict,
        }

    verdicts = [checks[factor]["verdict"] for factor in FACTOR_LABELS]
    if any(verdict == "insufficient_evidence" for verdict in verdicts):
        overall_verdict = "insufficient_evidence"
    elif all(verdict == "supported" for verdict in verdicts):
        overall_verdict = "supported"
    else:
        overall_verdict = "not_supported"

    return {
        "metric": metric_col,
        "metric_label": METRIC_LABELS.get(metric_col, metric_col),
        "n_points": int(factorial_metric_result.get("n_points", 0)),
        "checks": checks,
        "verdict": overall_verdict,
    }


def build_hypotheses_and_factorial_analysis(
    df: pd.DataFrame,
    *,
    min_datasets: int,
    alpha: float,
    sm: Any,
    smf: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    working = _build_factorial_dataset(df)
    hypotheses: dict[str, Any] = {
        "n_points_total": int(len(working)),
        "verdict": {},
    }
    factorial_analysis: dict[str, Any] = {
        "n_points_total": int(len(working)),
        "formula_template": "metric ~ log_n_samples + size_ratio + log_n_features",
        "metrics": {},
    }

    for metric_col in METRICS:
        factorial_metric_result = _build_factorial_metric_result(
            working_df=working,
            metric_col=metric_col,
            min_datasets=min_datasets,
            alpha=alpha,
            sm=sm,
            smf=smf,
        )
        hypotheses_metric = _build_hypotheses_from_factorial(
            metric_col=metric_col,
            factorial_metric_result=factorial_metric_result,
            alpha=alpha,
        )
        metric_values = working[metric_col].dropna().astype(float).values
        hypotheses_metric["overall_delta"] = _overall_delta_significance(metric_values, alpha=alpha)
        hypotheses[metric_col] = hypotheses_metric
        hypotheses["verdict"][metric_col] = hypotheses_metric["verdict"]
        factorial_analysis["metrics"][metric_col] = factorial_metric_result

    return hypotheses, factorial_analysis


def _empty_two_factor_result(*, n_points: int = 0, degrees_of_freedom: int = 0) -> dict[str, Any]:
    return {
        "n_points": n_points,
        "degrees_of_freedom": degrees_of_freedom,
        "r_squared": None,
        "adjusted_r_squared": None,
        "intercept": {"coef": None, "std_err": None, "t_stat": None, "p_value": None},
        "n_features": {"coef": None, "std_err": None, "t_stat": None, "p_value": None},
        "size_ratio": {"coef": None, "std_err": None, "t_stat": None, "p_value": None},
        "verdict": "insufficient_evidence",
    }


def _two_factor_linear_analysis(
    df: pd.DataFrame,
    *,
    metric_col: str,
    min_points: int,
    alpha: float,
    positive_effect_required: bool,
) -> dict[str, Any]:
    working = df.dropna(subset=["n_features", "size_ratio", metric_col]).copy()
    if working.empty:
        return _empty_two_factor_result(n_points=0)

    working["log_n_features"] = np.log1p(working["n_features"].astype(float))
    n_points = int(len(working))
    n_params = 3  # intercept + log_n_features + size_ratio
    min_required = max(int(min_points), n_params + 1)
    if n_points < min_required:
        return _empty_two_factor_result(n_points=n_points, degrees_of_freedom=max(n_points - n_params, 0))

    y = working[metric_col].astype(float).values
    x_features = working["log_n_features"].astype(float).values
    x_size = working["size_ratio"].astype(float).values
    X = np.column_stack([np.ones(n_points), x_features, x_size])

    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    df_resid = n_points - n_params
    if int(rank) < n_params or df_resid <= 0:
        return _empty_two_factor_result(n_points=n_points, degrees_of_freedom=max(df_resid, 0))

    residuals = y - (X @ beta)
    rss = float(np.dot(residuals, residuals))
    centered_y = y - float(np.mean(y))
    tss = float(np.dot(centered_y, centered_y))
    r_squared = None if tss <= 1e-12 else float(1.0 - (rss / tss))
    adjusted_r_squared = None
    if r_squared is not None:
        adjusted_r_squared = float(1.0 - (1.0 - r_squared) * (n_points - 1) / df_resid)

    sigma_sq = rss / df_resid
    xtx_inv = np.linalg.pinv(X.T @ X)
    cov_beta = sigma_sq * xtx_inv
    std_err = np.sqrt(np.maximum(np.diag(cov_beta), 0.0))

    t_stats = np.full(beta.shape, np.nan, dtype=float)
    valid_se = std_err > 0.0
    t_stats[valid_se] = beta[valid_se] / std_err[valid_se]
    p_values = np.full(beta.shape, np.nan, dtype=float)
    p_values[valid_se] = 2.0 * stats.t.sf(np.abs(t_stats[valid_se]), df=df_resid)

    result = {
        "n_points": n_points,
        "degrees_of_freedom": int(df_resid),
        "r_squared": r_squared,
        "adjusted_r_squared": adjusted_r_squared,
        "intercept": {
            "coef": _safe_float(beta[0]),
            "std_err": _safe_float(std_err[0]),
            "t_stat": _safe_float(t_stats[0]),
            "p_value": _safe_float(p_values[0]),
        },
        "n_features": {
            "coef": _safe_float(beta[1]),
            "std_err": _safe_float(std_err[1]),
            "t_stat": _safe_float(t_stats[1]),
            "p_value": _safe_float(p_values[1]),
        },
        "size_ratio": {
            "coef": _safe_float(beta[2]),
            "std_err": _safe_float(std_err[2]),
            "t_stat": _safe_float(t_stats[2]),
            "p_value": _safe_float(p_values[2]),
        },
        "verdict": "insufficient_evidence",
    }

    coef_features = result["n_features"]["coef"]
    coef_size = result["size_ratio"]["coef"]
    p_features = result["n_features"]["p_value"]
    p_size = result["size_ratio"]["p_value"]
    if coef_features is None or coef_size is None or p_features is None or p_size is None:
        return result

    passes = p_features < alpha and p_size < alpha
    if positive_effect_required:
        passes = passes and coef_features > 0.0 and coef_size > 0.0
    result["verdict"] = "supported" if passes else "not_supported"
    return result


def perform_size_by_ratio_analysis(df: pd.DataFrame) -> dict[str, Any]:
    analysis: dict[str, Any] = {}

    for size_ratio in sorted(df["size_ratio"].unique()):
        subset = df[df["size_ratio"] == size_ratio]
        row: dict[str, Any] = {
            "size_ratio": float(size_ratio),
            "size_pct": float(size_ratio) * 100.0,
            "n_datasets": int(len(subset)),
        }
        for metric_col, keys in METRIC_ANALYSIS_KEYS.items():
            values = subset[metric_col].astype(float).values
            t_stat, p_t = _one_sample_ttest_greater(values)
            w_stat, p_w = _wilcoxon_greater(values)
            row[keys["mean_col"]] = float(np.mean(values))
            row[keys["std_col"]] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"ttest_greater_{keys['suffix']}"] = {"statistic": t_stat, "p_value": p_t}
            row[f"wilcoxon_greater_{keys['suffix']}"] = {"statistic": w_stat, "p_value": p_w}
            row[keys["cohens_col"]] = _effect_size(values)
            row[keys["favor_col"]] = int(np.sum(values > 0))

        analysis[str(size_ratio)] = row

    return analysis


def compute_dataset_slopes(
    df: pd.DataFrame,
    *,
    metric_col: str,
    min_size_points: int,
) -> pd.DataFrame:
    columns = [
        "dataset_id",
        "dataset_name",
        "metric",
        "n_points",
        "slope",
        "intercept",
        "n_features",
        "log_n_features",
    ]
    rows: list[dict[str, Any]] = []

    for dataset_id, dataset_df in df.groupby("dataset_id"):
        ordered = dataset_df.sort_values("size_ratio").dropna(subset=[metric_col])
        if len(ordered) < min_size_points or ordered["size_ratio"].nunique() < min_size_points:
            continue

        x = ordered["size_ratio"].astype(float).values
        y = ordered[metric_col].astype(float).values
        slope, intercept = np.polyfit(x, y, deg=1)
        n_features = _safe_float(ordered["n_features"].iloc[0])

        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_name": ordered["dataset_name"].iloc[0],
                "metric": metric_col,
                "n_points": int(len(ordered)),
                "slope": float(slope),
                "intercept": float(intercept),
                "n_features": n_features,
                "log_n_features": float(np.log1p(n_features)) if n_features is not None else None,
            }
        )

    return pd.DataFrame(rows, columns=columns)


def analyze_slopes(slope_df: pd.DataFrame, *, min_datasets: int) -> dict[str, Any]:
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


def determine_size_verdict(
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
    p_t = slope_stats.get("ttest_greater", {}).get("p_value")
    p_w = slope_stats.get("wilcoxon_greater", {}).get("p_value")
    if mean_slope is None or p_t is None or p_w is None:
        return "insufficient_evidence"

    if mean_slope > 0.0 and p_t < alpha and p_w < alpha:
        return "supported"
    return "not_supported"


def build_size_analysis(
    df: pd.DataFrame,
    *,
    min_datasets: int,
    min_size_points: int,
    alpha: float,
) -> dict[str, Any]:
    if df.empty:
        return {
            "n_unique_sizes": 0,
            "by_size": {},
            "slope_analysis": {},
            "verdict": {metric: "insufficient_evidence" for metric in METRICS},
        }

    by_size = perform_size_by_ratio_analysis(df)
    n_unique_sizes = int(df["size_ratio"].nunique())

    slope_analysis: dict[str, Any] = {}
    verdict: dict[str, str] = {}
    for metric_col in METRICS:
        slope_df = compute_dataset_slopes(df, metric_col=metric_col, min_size_points=min_size_points)
        slope_stats = analyze_slopes(slope_df, min_datasets=min_datasets)
        slope_analysis[metric_col] = {
            "stats": slope_stats,
            "per_dataset": slope_df.to_dict(orient="records"),
        }
        verdict[metric_col] = determine_size_verdict(
            slope_stats=slope_stats,
            n_unique_sizes=n_unique_sizes,
            min_datasets=min_datasets,
            min_size_points=min_size_points,
            alpha=alpha,
        )

    return {
        "n_unique_sizes": n_unique_sizes,
        "by_size": by_size,
        "slope_analysis": slope_analysis,
        "verdict": verdict,
    }


def _build_dataset_feature_df(
    df: pd.DataFrame,
    *,
    target_size_ratio: float = FULL_SIZE_RATIO,
) -> pd.DataFrame:
    feature_df = df.dropna(subset=["n_features"]).copy()
    if feature_df.empty:
        return pd.DataFrame()

    feature_df = feature_df[np.isclose(feature_df["size_ratio"].astype(float), float(target_size_ratio), atol=1e-9)]
    if feature_df.empty:
        return pd.DataFrame()

    grouped = (
        feature_df.groupby(["dataset_id", "dataset_name", "n_features"], as_index=False)
        .agg(
            mean_delta_rmse=("delta_rmse", "mean"),
            mean_delta_r_auc_mse=("delta_r_auc_mse", "mean"),
            std_delta_rmse=("delta_rmse", "std"),
            std_delta_r_auc_mse=("delta_r_auc_mse", "std"),
            n_size_points=("size_ratio", "nunique"),
        )
        .fillna(0.0)
    )
    grouped["log_n_features"] = np.log1p(grouped["n_features"].astype(float))
    grouped["analysis_size_ratio"] = float(target_size_ratio)
    grouped["analysis_size_pct"] = float(target_size_ratio) * 100.0
    return grouped


def _feature_bin_summary(dataset_df: pd.DataFrame, metric_col: str) -> list[dict[str, Any]]:
    if dataset_df.empty:
        return []

    binned = _assign_feature_bins(dataset_df, max_bins=4)
    merged = dataset_df.merge(binned[["dataset_id", "feature_bin"]], on="dataset_id", how="inner")
    if merged.empty:
        return []

    summary = (
        merged.groupby("feature_bin", as_index=False)
        .agg(
            n_datasets=("dataset_id", "nunique"),
            mean=(metric_col, "mean"),
            std=(metric_col, "std"),
            median=(metric_col, "median"),
            min_value=(metric_col, "min"),
            max_value=(metric_col, "max"),
        )
        .fillna(0.0)
    )
    return summary.to_dict(orient="records")


def build_feature_analysis(
    dataset_feature_df: pd.DataFrame,
    *,
    min_datasets: int,
    alpha: float,
) -> dict[str, Any]:
    if dataset_feature_df.empty:
        return {
            "n_datasets_with_features": 0,
            "per_dataset": [],
            "relationships": {},
            "feature_bin_summary": {"delta_rmse": [], "delta_r_auc_mse": []},
            "verdict": {"delta_rmse": "insufficient_evidence", "delta_r_auc_mse": "insufficient_evidence"},
        }

    x = dataset_feature_df["log_n_features"].astype(float).values
    rmse_relationship = _relationship_analysis(
        x=x,
        y=dataset_feature_df["mean_delta_rmse"].astype(float).values,
        min_datasets=min_datasets,
        alpha=alpha,
        positive_effect_required=True,
    )
    rauc_relationship = _relationship_analysis(
        x=x,
        y=dataset_feature_df["mean_delta_r_auc_mse"].astype(float).values,
        min_datasets=min_datasets,
        alpha=alpha,
        positive_effect_required=True,
    )

    return {
        "n_datasets_with_features": int(len(dataset_feature_df)),
        "per_dataset": dataset_feature_df.to_dict(orient="records"),
        "relationships": {
            "delta_rmse": rmse_relationship,
            "delta_r_auc_mse": rauc_relationship,
        },
        "feature_bin_summary": {
            "delta_rmse": _feature_bin_summary(dataset_feature_df, "mean_delta_rmse"),
            "delta_r_auc_mse": _feature_bin_summary(dataset_feature_df, "mean_delta_r_auc_mse"),
        },
        "verdict": {
            "delta_rmse": rmse_relationship["verdict"],
            "delta_r_auc_mse": rauc_relationship["verdict"],
        },
    }


def _build_size_feature_matrix(df: pd.DataFrame, metric_col: str) -> dict[str, Any]:
    working = df.dropna(subset=["n_features"]).copy()
    if working.empty:
        return {"size_labels": [], "feature_bins": [], "values": [], "table": []}

    feature_map = _assign_feature_bins(working[["dataset_id", "n_features"]].drop_duplicates())
    working = working.merge(feature_map[["dataset_id", "feature_bin"]], on="dataset_id", how="inner")
    if working.empty:
        return {"size_labels": [], "feature_bins": [], "values": [], "table": []}

    size_values = sorted(working["size_ratio"].unique())
    size_order = [f"{s * 100:.0f}%" for s in size_values]
    size_label_map = {s: f"{s * 100:.0f}%" for s in size_values}
    working["size_label"] = working["size_ratio"].map(size_label_map)

    pivot = (
        working.pivot_table(
            index="feature_bin",
            columns="size_label",
            values=metric_col,
            aggfunc="mean",
        )
        .reindex(columns=size_order)
        .sort_index()
    )
    table_rows = pivot.reset_index().to_dict(orient="records")

    return {
        "size_labels": size_order,
        "feature_bins": [str(idx) for idx in pivot.index.tolist()],
        "values": pivot.fillna(np.nan).values.tolist(),
        "table": table_rows,
    }


def build_joint_analysis(
    df: pd.DataFrame,
    *,
    min_datasets: int,
    min_size_points: int,
    alpha: float,
) -> dict[str, Any]:
    slope_rmse_df = compute_dataset_slopes(df, metric_col="delta_rmse", min_size_points=min_size_points)
    slope_rauc_df = compute_dataset_slopes(df, metric_col="delta_r_auc_mse", min_size_points=min_size_points)

    slope_rmse_valid = (
        slope_rmse_df.dropna(subset=["log_n_features"])
        if "log_n_features" in slope_rmse_df.columns
        else slope_rmse_df.iloc[0:0]
    )
    slope_rauc_valid = (
        slope_rauc_df.dropna(subset=["log_n_features"])
        if "log_n_features" in slope_rauc_df.columns
        else slope_rauc_df.iloc[0:0]
    )

    rmse_relation = _relationship_analysis(
        x=slope_rmse_valid["log_n_features"].astype(float).values if not slope_rmse_valid.empty else np.array([]),
        y=slope_rmse_valid["slope"].astype(float).values if not slope_rmse_valid.empty else np.array([]),
        min_datasets=min_datasets,
        alpha=alpha,
        positive_effect_required=True,
    )
    rauc_relation = _relationship_analysis(
        x=slope_rauc_valid["log_n_features"].astype(float).values if not slope_rauc_valid.empty else np.array([]),
        y=slope_rauc_valid["slope"].astype(float).values if not slope_rauc_valid.empty else np.array([]),
        min_datasets=min_datasets,
        alpha=alpha,
        positive_effect_required=True,
    )
    two_factor_rmse = _two_factor_linear_analysis(
        df,
        metric_col="delta_rmse",
        min_points=min_datasets,
        alpha=alpha,
        positive_effect_required=True,
    )
    two_factor_rauc = _two_factor_linear_analysis(
        df,
        metric_col="delta_r_auc_mse",
        min_points=min_datasets,
        alpha=alpha,
        positive_effect_required=True,
    )

    return {
        "interaction_relationships": {
            "delta_rmse": rmse_relation,
            "delta_r_auc_mse": rauc_relation,
        },
        "two_factor_model": {
            "delta_rmse": two_factor_rmse,
            "delta_r_auc_mse": two_factor_rauc,
        },
        "slope_vs_features": {
            "delta_rmse": slope_rmse_valid.to_dict(orient="records"),
            "delta_r_auc_mse": slope_rauc_valid.to_dict(orient="records"),
        },
        "size_feature_matrix": {
            "delta_rmse": _build_size_feature_matrix(df, "delta_rmse"),
            "delta_r_auc_mse": _build_size_feature_matrix(df, "delta_r_auc_mse"),
        },
        "verdict": {
            "delta_rmse": rmse_relation["verdict"],
            "delta_r_auc_mse": rauc_relation["verdict"],
        },
    }


def _build_size_trend_figure(by_size: dict[str, Any], metric: str, title: str, y_title: str) -> go.Figure:
    if not by_size:
        return go.Figure().update_layout(title=title)

    ordered = sorted(by_size.values(), key=lambda item: item["size_ratio"])
    x = [item["size_pct"] for item in ordered]
    if metric == "delta_rmse":
        y = [item["mean_delta_rmse"] for item in ordered]
        err = [item["std_delta_rmse"] for item in ordered]
    else:
        y = [item["mean_delta_r_auc_mse"] for item in ordered]
        err = [item["std_delta_r_auc_mse"] for item in ordered]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name=metric,
            error_y={"type": "data", "array": err, "visible": True},
        )
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="#D62839")
    fig.update_layout(title=title, xaxis_title="Dataset size (%)", yaxis_title=y_title, template="plotly_white")
    return fig


def _build_size_boxplot_figure(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ΔRMSE by size", "ΔR-AUC MSE by size"))
    if not df.empty:
        for size_ratio in sorted(df["size_ratio"].unique()):
            size_label = f"{size_ratio * 100:.0f}%"
            subset = df[df["size_ratio"] == size_ratio]
            fig.add_trace(go.Box(y=subset["delta_rmse"], name=size_label, boxmean=True), row=1, col=1)
            fig.add_trace(go.Box(y=subset["delta_r_auc_mse"], name=size_label, boxmean=True), row=1, col=2)
    fig.update_layout(template="plotly_white", showlegend=False, title="Distribution by size")
    return fig


def _hist_nbins(values: np.ndarray) -> int:
    n_points = int(len(values))
    if n_points <= 1:
        return 1
    return int(max(8, min(30, round(np.sqrt(float(n_points)) * 2.0))))


def _build_delta_histogram_figure(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Distribution of ΔRMSE", "Distribution of ΔR-AUC MSE"),
    )
    if not df.empty:
        rmse_vals = pd.to_numeric(df["delta_rmse"], errors="coerce").dropna().astype(float).values
        rauc_vals = pd.to_numeric(df["delta_r_auc_mse"], errors="coerce").dropna().astype(float).values
        if len(rmse_vals) > 0:
            fig.add_trace(
                go.Histogram(
                    x=rmse_vals,
                    nbinsx=_hist_nbins(rmse_vals),
                    marker_color="#0D6EFD",
                    opacity=0.8,
                    name="ΔRMSE",
                ),
                row=1,
                col=1,
            )
        if len(rauc_vals) > 0:
            fig.add_trace(
                go.Histogram(
                    x=rauc_vals,
                    nbinsx=_hist_nbins(rauc_vals),
                    marker_color="#FD7E14",
                    opacity=0.8,
                    name="ΔR-AUC MSE",
                ),
                row=1,
                col=2,
            )

    fig.add_vline(x=0.0, line_dash="dash", line_color="#6C757D", row=1, col=1)
    fig.add_vline(x=0.0, line_dash="dash", line_color="#6C757D", row=1, col=2)
    fig.update_layout(
        template="plotly_white",
        title="Delta distributions",
        showlegend=False,
        bargap=0.1,
    )
    fig.update_xaxes(title_text="Delta", row=1, col=1)
    fig.update_xaxes(title_text="Delta", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    return fig


def _extract_slope_values(size_analysis: dict[str, Any], metric_col: str) -> np.ndarray:
    rows = size_analysis.get("slope_analysis", {}).get(metric_col, {}).get("per_dataset", [])
    if not isinstance(rows, list) or not rows:
        return np.array([], dtype=float)
    frame = pd.DataFrame(rows)
    if "slope" not in frame.columns:
        return np.array([], dtype=float)
    values = pd.to_numeric(frame["slope"], errors="coerce").dropna().astype(float).values
    return values


def _build_slope_histogram_figure(size_analysis: dict[str, Any]) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Distribution of size slopes (ΔRMSE)", "Distribution of size slopes (ΔR-AUC MSE)"),
    )
    rmse_slopes = _extract_slope_values(size_analysis, "delta_rmse")
    rauc_slopes = _extract_slope_values(size_analysis, "delta_r_auc_mse")

    if len(rmse_slopes) > 0:
        fig.add_trace(
            go.Histogram(
                x=rmse_slopes,
                nbinsx=_hist_nbins(rmse_slopes),
                marker_color="#20C997",
                opacity=0.8,
                name="Slope ΔRMSE",
            ),
            row=1,
            col=1,
        )
    if len(rauc_slopes) > 0:
        fig.add_trace(
            go.Histogram(
                x=rauc_slopes,
                nbinsx=_hist_nbins(rauc_slopes),
                marker_color="#845EF7",
                opacity=0.8,
                name="Slope ΔR-AUC MSE",
            ),
            row=1,
            col=2,
        )

    fig.add_vline(x=0.0, line_dash="dash", line_color="#6C757D", row=1, col=1)
    fig.add_vline(x=0.0, line_dash="dash", line_color="#6C757D", row=1, col=2)
    fig.update_layout(
        template="plotly_white",
        title="Slope distributions",
        showlegend=False,
        bargap=0.1,
    )
    fig.update_xaxes(title_text="Slope", row=1, col=1)
    fig.update_xaxes(title_text="Slope", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    return fig


def _assign_absolute_size_bins(values: pd.Series, max_bins: int = 4) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    bins = pd.Series(index=values.index, dtype="object")
    valid = numeric.dropna().astype(float)
    if valid.empty:
        return bins

    n_unique = int(valid.nunique())
    if n_unique <= 1:
        value = float(valid.iloc[0])
        bins.loc[valid.index] = f"{value:.0f}-{value:.0f}"
        return bins

    q = min(max_bins, n_unique)
    qcut_bins = pd.qcut(valid, q=q, duplicates="drop")
    categories = list(qcut_bins.cat.categories)
    labels = [_format_interval(interval) for interval in categories]
    relabeled = qcut_bins.cat.rename_categories(labels).astype(str)
    bins.loc[valid.index] = relabeled
    return bins


def _build_absolute_size_scatter_figure(
    df: pd.DataFrame,
    *,
    metric_col: str,
    title: str,
    y_title: str,
) -> go.Figure:
    fig = go.Figure()
    working = df.dropna(subset=["n_samples", metric_col]).copy()
    if working.empty:
        return fig.update_layout(title=title, template="plotly_white")

    working["n_samples"] = pd.to_numeric(working["n_samples"], errors="coerce")
    working[metric_col] = pd.to_numeric(working[metric_col], errors="coerce")
    working = working[np.isfinite(working["n_samples"]) & np.isfinite(working[metric_col])]
    working = working[working["n_samples"] > 0.0]
    if working.empty:
        return fig.update_layout(title=title, template="plotly_white")

    names = working["dataset_name"].astype(str).values
    x_vals = working["n_samples"].astype(float).values
    y_vals = working[metric_col].astype(float).values
    size_ratio_vals = working["size_ratio"].astype(float).values
    marker: dict[str, Any] = {"size": 9, "opacity": 0.82}
    if "n_features" in working.columns and working["n_features"].notna().any():
        feature_counts = pd.to_numeric(working["n_features"], errors="coerce").astype(float).values
        q_low = float(np.nanpercentile(feature_counts, 10))
        q_high = float(np.nanpercentile(feature_counts, 95))
        if not np.isfinite(q_low):
            q_low = float(np.nanmin(feature_counts))
        if not np.isfinite(q_high):
            q_high = float(np.nanmax(feature_counts))
        if q_high <= q_low:
            q_low = float(np.nanmin(feature_counts))
            q_high = float(np.nanmax(feature_counts))
        marker.update(
            {
                "color": feature_counts,
                "colorscale": [
                    [0.0, "#74C476"],
                    [0.4, "#41AB5D"],
                    [0.7, "#238B45"],
                    [1.0, "#005A32"],
                ],
                "cmin": q_low,
                "cmax": q_high,
                "colorbar": {"title": "n_features"},
                "showscale": True,
            }
        )
        customdata = np.column_stack([size_ratio_vals, feature_counts])
        hovertemplate = (
            "Dataset=%{text}<br>n_samples=%{x:.0f}<br>"
            "size_ratio=%{customdata[0]:.2f}<br>n_features=%{customdata[1]:.0f}<br>"
            "value=%{y:.4f}<extra></extra>"
        )
    else:
        marker["color"] = "#2F9E44"
        customdata = np.column_stack([size_ratio_vals])
        hovertemplate = (
            "Dataset=%{text}<br>n_samples=%{x:.0f}<br>"
            "size_ratio=%{customdata[0]:.2f}<br>value=%{y:.4f}<extra></extra>"
        )

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker=marker,
            text=names,
            customdata=customdata,
            hovertemplate=hovertemplate,
            name="dataset-size points",
        )
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="#6C757D")
    fig.update_layout(
        title=title,
        xaxis_title="n_samples (log scale)",
        yaxis_title=y_title,
        xaxis_type="log",
        template="plotly_white",
    )
    return fig


def _build_absolute_size_boxplot_figure(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("ΔRMSE by absolute n_samples bins", "ΔR-AUC MSE by absolute n_samples bins"),
    )
    working = df.dropna(subset=["n_samples"]).copy()
    if working.empty:
        return fig.update_layout(template="plotly_white", showlegend=False, title="Delta by absolute n_samples bins")

    working["n_samples"] = pd.to_numeric(working["n_samples"], errors="coerce")
    working = working[np.isfinite(working["n_samples"]) & (working["n_samples"] > 0.0)]
    if working.empty:
        return fig.update_layout(template="plotly_white", showlegend=False, title="Delta by absolute n_samples bins")

    working["n_samples_bin"] = _assign_absolute_size_bins(working["n_samples"])
    working = working.dropna(subset=["n_samples_bin"])
    if working.empty:
        return fig.update_layout(template="plotly_white", showlegend=False, title="Delta by absolute n_samples bins")

    ordered_bins = (
        working.groupby("n_samples_bin", as_index=False)["n_samples"]
        .min()
        .sort_values("n_samples")
        ["n_samples_bin"]
        .tolist()
    )
    for bin_label in ordered_bins:
        subset = working[working["n_samples_bin"] == bin_label]
        rmse_vals = pd.to_numeric(subset["delta_rmse"], errors="coerce").dropna().astype(float).values
        rauc_vals = pd.to_numeric(subset["delta_r_auc_mse"], errors="coerce").dropna().astype(float).values
        if len(rmse_vals) > 0:
            fig.add_trace(go.Box(y=rmse_vals, name=bin_label, boxmean=True), row=1, col=1)
        if len(rauc_vals) > 0:
            fig.add_trace(go.Box(y=rauc_vals, name=bin_label, boxmean=True), row=1, col=2)

    fig.add_hline(y=0.0, line_dash="dash", line_color="#6C757D", row=1, col=1)
    fig.add_hline(y=0.0, line_dash="dash", line_color="#6C757D", row=1, col=2)
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        title="Delta by absolute n_samples bins",
    )
    return fig


def _build_feature_scatter_figure(
    dataset_feature_df: pd.DataFrame,
    *,
    metric_col: str,
    title: str,
    y_title: str,
) -> go.Figure:
    fig = go.Figure()
    if dataset_feature_df.empty:
        fig.update_layout(title=title, template="plotly_white")
        return fig

    x = dataset_feature_df["n_features"].astype(float).values
    y = dataset_feature_df[metric_col].astype(float).values
    names = dataset_feature_df["dataset_name"].astype(str).values
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker={"size": 10, "opacity": 0.75, "color": "#0D6EFD"},
            text=names,
            hovertemplate="Dataset=%{text}<br>n_features=%{x:.0f}<br>value=%{y:.4f}<extra></extra>",
            name="datasets",
        )
    )

    if len(dataset_feature_df) >= 2:
        slope, intercept = np.polyfit(x, y, deg=1)
        x_line = np.linspace(np.min(x), np.max(x), 100)
        y_line = slope * x_line + intercept
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line={"color": "#DC3545", "width": 2},
                name="linear trend",
            )
        )

    fig.add_hline(y=0.0, line_dash="dash", line_color="#6C757D")
    fig.update_layout(
        title=title,
        xaxis_title="n_features (log scale)",
        yaxis_title=y_title,
        xaxis_type="log",
        template="plotly_white",
    )
    return fig


def _build_size_feature_scatter_figure(
    df: pd.DataFrame,
    *,
    metric_col: str,
    title: str,
    y_title: str,
) -> go.Figure:
    fig = go.Figure()
    working = df.dropna(subset=["n_features", metric_col]).copy()
    if working.empty:
        return fig.update_layout(title=title, template="plotly_white")

    names = working["dataset_name"].astype(str).values
    size_ratio = working["size_ratio"].astype(float).values
    marker: dict[str, Any] = {"size": 9, "opacity": 0.8}
    if "n_samples" in working.columns and working["n_samples"].notna().any():
        sample_counts = working["n_samples"].astype(float).values
        q_low = float(np.nanpercentile(sample_counts, 10))
        q_high = float(np.nanpercentile(sample_counts, 95))
        if not np.isfinite(q_low):
            q_low = float(np.nanmin(sample_counts))
        if not np.isfinite(q_high):
            q_high = float(np.nanmax(sample_counts))
        if q_high <= q_low:
            q_low = float(np.nanmin(sample_counts))
            q_high = float(np.nanmax(sample_counts))
        # Use only the darker half of the green palette to avoid near-white markers.
        marker.update(
            {
                "color": sample_counts,
                "colorscale": [
                    [0.0, "#74C476"],
                    [0.4, "#41AB5D"],
                    [0.7, "#238B45"],
                    [1.0, "#005A32"],
                ],
                "cmin": q_low,
                "cmax": q_high,
                "colorbar": {"title": "n_samples"},
                "showscale": True,
            }
        )
        hovertemplate = (
            "Dataset=%{text}<br>n_features=%{x:.0f}<br>"
            "size_ratio=%{customdata[0]:.2f}<br>n_samples=%{customdata[1]:.0f}<br>"
            "value=%{y:.4f}<extra></extra>"
        )
        customdata = np.column_stack([size_ratio, sample_counts])
    else:
        marker["color"] = "#2F9E44"
        hovertemplate = (
            "Dataset=%{text}<br>n_features=%{x:.0f}<br>"
            "size_ratio=%{customdata[0]:.2f}<br>value=%{y:.4f}<extra></extra>"
        )
        customdata = np.column_stack([size_ratio])

    fig.add_trace(
        go.Scatter(
            x=working["n_features"].astype(float).values,
            y=working[metric_col].astype(float).values,
            mode="markers",
            marker=marker,
            text=names,
            customdata=customdata,
            hovertemplate=hovertemplate,
            name="dataset-size points",
        )
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="#6C757D")
    fig.update_layout(
        title=title,
        xaxis_title="n_features (log scale)",
        yaxis_title=y_title,
        xaxis_type="log",
        template="plotly_white",
    )
    return fig


def _build_feature_bin_bar_figure(feature_analysis: dict[str, Any]) -> go.Figure:
    fig = go.Figure()
    rmse_bins = feature_analysis.get("feature_bin_summary", {}).get("delta_rmse", [])
    rauc_bins = feature_analysis.get("feature_bin_summary", {}).get("delta_r_auc_mse", [])
    if not rmse_bins and not rauc_bins:
        return fig.update_layout(title="Feature-bin summary", template="plotly_white")

    x_labels = [row["feature_bin"] for row in rmse_bins] if rmse_bins else [row["feature_bin"] for row in rauc_bins]
    rmse_vals = {row["feature_bin"]: row["mean"] for row in rmse_bins}
    rauc_vals = {row["feature_bin"]: row["mean"] for row in rauc_bins}

    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=[rmse_vals.get(label, np.nan) for label in x_labels],
            name="Mean ΔRMSE",
            marker_color="#0D6EFD",
        )
    )
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=[rauc_vals.get(label, np.nan) for label in x_labels],
            name="Mean ΔR-AUC MSE",
            marker_color="#FD7E14",
        )
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="#6C757D")
    fig.update_layout(
        title="Mean delta by feature-count bins",
        barmode="group",
        xaxis_title="Feature-count bin",
        yaxis_title="Mean delta",
        template="plotly_white",
    )
    return fig


def _build_heatmap_figure(matrix: dict[str, Any], title: str) -> go.Figure:
    fig = go.Figure()
    if not matrix["values"]:
        return fig.update_layout(title=title, template="plotly_white")

    fig.add_trace(
        go.Heatmap(
            z=matrix["values"],
            x=matrix["size_labels"],
            y=matrix["feature_bins"],
            colorscale="RdBu",
            colorbar={"title": "Mean delta"},
            zmid=0.0,
        )
    )
    fig.update_layout(title=title, xaxis_title="Dataset size", yaxis_title="Feature-count bin", template="plotly_white")
    return fig


def _build_slope_vs_features_figure(per_dataset_rows: list[dict[str, Any]], title: str, y_title: str) -> go.Figure:
    fig = go.Figure()
    if not per_dataset_rows:
        return fig.update_layout(title=title, template="plotly_white")

    frame = pd.DataFrame(per_dataset_rows)
    x = frame["n_features"].astype(float).values
    y = frame["slope"].astype(float).values
    names = frame["dataset_name"].astype(str).values

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker={"size": 10, "opacity": 0.75, "color": "#20C997"},
            text=names,
            hovertemplate="Dataset=%{text}<br>n_features=%{x:.0f}<br>slope=%{y:.5f}<extra></extra>",
            name="dataset slopes",
        )
    )

    if len(frame) >= 2:
        slope, intercept = np.polyfit(x, y, deg=1)
        x_line = np.linspace(np.min(x), np.max(x), 100)
        y_line = slope * x_line + intercept
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line={"color": "#DC3545", "width": 2},
                name="linear trend",
            )
        )

    fig.add_hline(y=0.0, line_dash="dash", line_color="#6C757D")
    fig.update_layout(
        title=title,
        xaxis_title="n_features (log scale)",
        yaxis_title=y_title,
        xaxis_type="log",
        template="plotly_white",
    )
    return fig


def _fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        casted = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(casted):
        return "NA"
    return f"{casted:.{digits}f}"


def _verdict_badge(verdict: str) -> str:
    cls = {
        "supported": "badge-supported",
        "not_supported": "badge-not-supported",
        "insufficient_evidence": "badge-insufficient",
    }.get(verdict, "badge-insufficient")
    label = verdict.replace("_", " ").title()
    return f'<span class="badge {cls}">{label}</span>'


def _table_html(rows: list[dict[str, Any]], columns: list[str], column_labels: dict[str, str]) -> str:
    if not rows:
        return "<p class='muted'>No data available.</p>"
    frame = pd.DataFrame(rows)
    frame = frame[[col for col in columns if col in frame.columns]]
    frame = frame.rename(columns=column_labels)
    return frame.to_html(index=False, classes="data-table", border=0, justify="left")


def _fig_html(fig: go.Figure, *, include_plotlyjs: bool) -> str:
    return fig.to_html(
        full_html=False,
        include_plotlyjs=True if include_plotlyjs else False,
        config={"displaylogo": False, "responsive": True},
    )


def generate_html_report(
    *,
    report: dict[str, Any],
    df: pd.DataFrame,
    dataset_feature_df: pd.DataFrame,
    output_path: Path,
) -> None:
    size_analysis = report.get("size_analysis", {})
    feature_analysis = report.get("feature_analysis", {})
    joint_analysis = report.get("joint_analysis", {})
    hypotheses = report.get("hypotheses", {})
    factorial_analysis = report.get("factorial_analysis", {})
    summary = report["summary"]

    by_size = size_analysis.get("by_size", {})
    figures_by_key: dict[str, go.Figure] = {
        "size_trend_rmse": _build_size_trend_figure(
            by_size,
            "delta_rmse",
            "Size trend: Mean ΔRMSE",
            "Mean ΔRMSE",
        ),
        "size_trend_rauc": _build_size_trend_figure(
            by_size,
            "delta_r_auc_mse",
            "Size trend: Mean ΔR-AUC MSE",
            "Mean ΔR-AUC MSE",
        ),
        "size_boxplot_by_ratio": _build_size_boxplot_figure(df),
        "delta_histograms": _build_delta_histogram_figure(df),
        "abs_size_scatter_rmse": _build_absolute_size_scatter_figure(
            df,
            metric_col="delta_rmse",
            title="n_samples vs ΔRMSE (absolute size)",
            y_title="ΔRMSE",
        ),
        "abs_size_scatter_rauc": _build_absolute_size_scatter_figure(
            df,
            metric_col="delta_r_auc_mse",
            title="n_samples vs ΔR-AUC MSE (absolute size)",
            y_title="ΔR-AUC MSE",
        ),
        "abs_size_boxplot": _build_absolute_size_boxplot_figure(df),
        "feature_scatter_rmse_100": _build_feature_scatter_figure(
            dataset_feature_df,
            metric_col="mean_delta_rmse",
            title="n_features vs ΔRMSE at 100% size (per dataset)",
            y_title="ΔRMSE at 100% size",
        ),
        "feature_scatter_rauc_100": _build_feature_scatter_figure(
            dataset_feature_df,
            metric_col="mean_delta_r_auc_mse",
            title="n_features vs ΔR-AUC MSE at 100% size (per dataset)",
            y_title="ΔR-AUC MSE at 100% size",
        ),
        "feature_bin_bar": _build_feature_bin_bar_figure(feature_analysis),
        "size_feature_scatter_rmse": _build_size_feature_scatter_figure(
            df,
            metric_col="delta_rmse",
            title="n_features vs ΔRMSE (all size ratios)",
            y_title="ΔRMSE",
        ),
        "size_feature_scatter_rauc": _build_size_feature_scatter_figure(
            df,
            metric_col="delta_r_auc_mse",
            title="n_features vs ΔR-AUC MSE (all size ratios)",
            y_title="ΔR-AUC MSE",
        ),
        "heatmap_rmse": _build_heatmap_figure(
            joint_analysis.get("size_feature_matrix", {}).get("delta_rmse", {"values": []}),
            "Mean ΔRMSE heatmap: size × feature bin",
        ),
        "heatmap_rauc": _build_heatmap_figure(
            joint_analysis.get("size_feature_matrix", {}).get("delta_r_auc_mse", {"values": []}),
            "Mean ΔR-AUC MSE heatmap: size × feature bin",
        ),
        "slope_vs_features_rmse": _build_slope_vs_features_figure(
            joint_analysis.get("slope_vs_features", {}).get("delta_rmse", []),
            "Size-slope(ΔRMSE) vs n_features",
            "Slope of ΔRMSE over size",
        ),
        "slope_vs_features_rauc": _build_slope_vs_features_figure(
            joint_analysis.get("slope_vs_features", {}).get("delta_r_auc_mse", []),
            "Size-slope(ΔR-AUC MSE) vs n_features",
            "Slope of ΔR-AUC MSE over size",
        ),
        "slope_histograms": _build_slope_histogram_figure(size_analysis),
    }

    figure_caption_by_key = {
        "delta_histograms": "Distribution of ΔRMSE | Distribution of ΔR-AUC MSE",
        "slope_histograms": "Distribution of size slopes (ΔRMSE) | Distribution of size slopes (ΔR-AUC MSE)",
        "abs_size_scatter_rmse": "n_samples vs ΔRMSE (absolute size)",
        "abs_size_scatter_rauc": "n_samples vs ΔR-AUC MSE (absolute size)",
        "abs_size_boxplot": "Delta by absolute n_samples bins",
    }

    include_js = True
    figure_html_by_key: dict[str, str] = {}
    for key, fig in figures_by_key.items():
        caption = figure_caption_by_key.get(key)
        caption_html = f"<h3>{html_lib.escape(caption)}</h3>" if caption else ""
        figure_html_by_key[key] = (
            f"<div class='chart'>{caption_html}{_fig_html(fig, include_plotlyjs=include_js)}</div>"
        )
        include_js = False

    size_charts_html = "".join(
        figure_html_by_key[key]
        for key in [
            "size_trend_rmse",
            "size_trend_rauc",
            "size_boxplot_by_ratio",
            "delta_histograms",
            "abs_size_scatter_rmse",
            "abs_size_scatter_rauc",
            "abs_size_boxplot",
        ]
    )
    feature_charts_html = "".join(
        figure_html_by_key[key]
        for key in [
            "feature_scatter_rmse_100",
            "feature_scatter_rauc_100",
            "feature_bin_bar",
        ]
    )
    joint_charts_html = "".join(
        figure_html_by_key[key]
        for key in [
            "size_feature_scatter_rmse",
            "size_feature_scatter_rauc",
            "heatmap_rmse",
            "heatmap_rauc",
            "slope_vs_features_rmse",
            "slope_vs_features_rauc",
            "slope_histograms",
        ]
    )

    size_rows = [
        {
            "size_pct": values.get("size_pct"),
            "n_datasets": values.get("n_datasets"),
            "mean_delta_rmse": values.get("mean_delta_rmse"),
            "mean_delta_r_auc_mse": values.get("mean_delta_r_auc_mse"),
        }
        for _, values in sorted(by_size.items(), key=lambda item: float(item[1]["size_ratio"]))
    ]
    for row in size_rows:
        row["size_pct"] = _fmt_float(row["size_pct"], 0)
        row["mean_delta_rmse"] = _fmt_float(row["mean_delta_rmse"])
        row["mean_delta_r_auc_mse"] = _fmt_float(row["mean_delta_r_auc_mse"])

    hypothesis_rows: list[dict[str, Any]] = []
    overall_delta_rows: list[dict[str, Any]] = []
    factorial_overview_rows: list[dict[str, Any]] = []
    factorial_factor_rows: list[dict[str, Any]] = []
    factorial_raw_blocks: list[str] = []

    for metric_col in METRICS:
        metric_label = METRIC_LABELS.get(metric_col, metric_col)
        metric_hyp = hypotheses.get(metric_col, {})
        checks = metric_hyp.get("checks", {})
        overall_delta = metric_hyp.get("overall_delta", {})
        for factor in FACTOR_LABELS:
            check = checks.get(factor, {})
            hypothesis_rows.append(
                {
                    "metric": metric_label,
                    "factor": FACTOR_LABELS[factor],
                    "coef": _fmt_float(check.get("coef")),
                    "t_stat": _fmt_float(check.get("t_stat")),
                    "p_one_sided": _fmt_float(check.get("p_value_one_sided_greater")),
                    "verdict": check.get("verdict", "insufficient_evidence").replace("_", " "),
                }
            )
        overall_delta_rows.append(
            {
                "metric": metric_label,
                "n_points": int(overall_delta.get("n_points", 0)),
                "mean_delta": _fmt_float(overall_delta.get("mean_delta")),
                "ci_low": _fmt_float(overall_delta.get("ci_low")),
                "ci_high": _fmt_float(overall_delta.get("ci_high")),
                "p_one_sided": _fmt_float(overall_delta.get("p_value_one_sided_greater")),
                "verdict": overall_delta.get("verdict", "insufficient_evidence").replace("_", " "),
            }
        )

        metric_factorial = factorial_analysis.get("metrics", {}).get(metric_col, {})
        significant = metric_factorial.get("significant_factors", [])
        factor_labels = [FACTOR_LABELS.get(name, name) for name in significant]
        factorial_overview_rows.append(
            {
                "metric": metric_label,
                "n_points": int(metric_factorial.get("n_points", 0)),
                "r_squared": _fmt_float(metric_factorial.get("r_squared")),
                "adj_r_squared": _fmt_float(metric_factorial.get("adjusted_r_squared")),
                "f_p_value": _fmt_float(metric_factorial.get("f_p_value")),
                "significant_factors": ", ".join(factor_labels) if factor_labels else "none",
                "verdict": metric_factorial.get("verdict", "insufficient_evidence").replace("_", " "),
            }
        )
        for factor in FACTOR_LABELS:
            block = metric_factorial.get("factors", {}).get(factor, {})
            p_value = block.get("p_value")
            is_significant = p_value is not None and float(p_value) < float(summary["alpha"])
            factorial_factor_rows.append(
                {
                    "metric": metric_label,
                    "factor": FACTOR_LABELS[factor],
                    "coef": _fmt_float(block.get("coef")),
                    "p_value": _fmt_float(p_value),
                    "ci_low": _fmt_float(block.get("ci_low")),
                    "ci_high": _fmt_float(block.get("ci_high")),
                    "significant": "yes" if is_significant else "no",
                }
            )

        raw_summary = metric_factorial.get("raw_output", {}).get("summary", "")
        raw_anova = metric_factorial.get("raw_output", {}).get("anova_type_2", "")
        if raw_summary:
            factorial_raw_blocks.append(
                "<h3>{title} OLS Summary</h3><pre>{content}</pre>".format(
                    title=html_lib.escape(metric_label),
                    content=html_lib.escape(raw_summary),
                )
            )
        if raw_anova:
            factorial_raw_blocks.append(
                "<h3>{title} ANOVA Type II</h3><pre>{content}</pre>".format(
                    title=html_lib.escape(metric_label),
                    content=html_lib.escape(raw_anova),
                )
            )

    feature_range = "NA"
    if not dataset_feature_df.empty:
        feature_range = (
            f"{int(dataset_feature_df['n_features'].min())} - {int(dataset_feature_df['n_features'].max())}"
        )

    overall = report.get("status", "insufficient_evidence")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Size Dependence Report</title>
  <style>
    :root {{
      --bg: #f3f7fb;
      --card: #ffffff;
      --ink: #0f1b2a;
      --muted: #5d6b7a;
      --brand: #0d6efd;
      --good: #1f9d55;
      --warn: #c77700;
      --bad: #c92a2a;
      --border: #dce6f1;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      background: radial-gradient(circle at top right, #eaf3ff 0%, var(--bg) 45%, #edf6f4 100%);
      color: var(--ink);
    }}
    .container {{
      width: min(1200px, 96vw);
      margin: 24px auto 40px;
    }}
    .hero {{
      background: linear-gradient(130deg, #0d6efd 0%, #0b4ea7 55%, #0a8f74 100%);
      color: #fff;
      border-radius: 16px;
      padding: 24px;
      box-shadow: 0 16px 35px rgba(0, 35, 90, 0.22);
    }}
    .hero h1 {{ margin: 0 0 8px 0; font-size: 1.9rem; }}
    .hero p {{ margin: 0; opacity: 0.95; }}
    .grid {{
      margin-top: 18px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px 16px;
      box-shadow: 0 8px 18px rgba(13, 37, 61, 0.06);
    }}
    .card .label {{ color: var(--muted); font-size: 0.85rem; }}
    .card .value {{ margin-top: 6px; font-size: 1.15rem; font-weight: 700; }}
    h2 {{
      margin: 26px 0 10px;
      font-size: 1.35rem;
      border-left: 4px solid var(--brand);
      padding-left: 10px;
    }}
    .section {{
      margin-top: 14px;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px 16px;
      box-shadow: 0 10px 22px rgba(13, 37, 61, 0.06);
    }}
    .muted {{ color: var(--muted); }}
    .badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 10px 0 2px;
    }}
    .badge {{
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 0.84rem;
      font-weight: 600;
      border: 1px solid transparent;
    }}
    .badge-supported {{ background: #d9f5e7; border-color: #96dfbb; color: #0d6d3d; }}
    .badge-not-supported {{ background: #ffe9cc; border-color: #f4c27a; color: #8a4f00; }}
    .badge-insufficient {{ background: #fbe0e2; border-color: #efafb4; color: #8c1d25; }}
    .chart {{ margin: 14px 0; }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
      margin-top: 10px;
    }}
    .data-table th, .data-table td {{
      border-bottom: 1px solid var(--border);
      padding: 8px 9px;
      text-align: left;
    }}
    .data-table th {{
      background: #edf4fb;
      font-weight: 700;
    }}
    pre {{
      background: #0f1b2a;
      color: #e9f0f8;
      border-radius: 10px;
      padding: 12px;
      overflow-x: auto;
      font-size: 0.8rem;
      line-height: 1.35;
      white-space: pre;
    }}
    h3 {{
      margin: 16px 0 8px;
      font-size: 1rem;
    }}
    .footer {{
      margin-top: 20px;
      color: var(--muted);
      font-size: 0.84rem;
      text-align: center;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="hero">
      <h1>HMTL Delta Dependence Report</h1>
      <p>Interactive statistical report for baseline <strong>{summary["baseline"]}</strong>. Generated {report["generated_at_utc"]}.</p>
      <div class="badges">
        <span class="badge {("badge-supported" if overall == "completed" else "badge-insufficient")}">Overall status: {overall.replace("_", " ").title()}</span>
      </div>
    </div>

    <div class="grid">
      <div class="card"><div class="label">Datasets</div><div class="value">{summary["n_datasets"]}</div></div>
      <div class="card"><div class="label">Data points</div><div class="value">{summary["n_data_points"]}</div></div>
      <div class="card"><div class="label">Datasets with n_features</div><div class="value">{summary["n_datasets_with_features"]}</div></div>
      <div class="card"><div class="label">Feature range</div><div class="value">{feature_range}</div></div>
      <div class="card"><div class="label">Sizes tested</div><div class="value">{", ".join(f"{v*100:.0f}%" for v in summary["sizes_tested"]) if summary["sizes_tested"] else "NA"}</div></div>
      <div class="card"><div class="label">Significance alpha</div><div class="value">{_fmt_float(summary["alpha"], 3)}</div></div>
    </div>

    <h2>Size Analysis</h2>
    <div class="section">
      <p class="muted">Visual overview of delta behavior over relative and absolute dataset sizes.</p>
      {size_charts_html}
      {_table_html(
        size_rows,
        ["size_pct", "n_datasets", "mean_delta_rmse", "mean_delta_r_auc_mse"],
        {
          "size_pct": "Size (%)",
          "n_datasets": "N datasets",
          "mean_delta_rmse": "Mean ΔRMSE",
          "mean_delta_r_auc_mse": "Mean ΔR-AUC MSE",
        },
      )}
    </div>

    <h2>Feature Analysis</h2>
    <div class="section">
      <p class="muted">Analyzes how delta at <code>100%</code> size depends on <code>n_features</code> (solo feature effect).</p>
      {feature_charts_html}
    </div>

    <h2>Joint Analysis</h2>
    <div class="section">
      <p class="muted">Interaction visualizations across size and feature complexity.</p>
      {joint_charts_html}
    </div>

    <h2>Hypotheses</h2>
    <div class="section">
      <p class="muted">
        One-sided checks for positive growth of deltas over the three factors:
        <code>log1p(n_samples)</code>, <code>size_ratio</code>, <code>log1p(n_features)</code>.
      </p>
      <div class="badges">
        {_verdict_badge(hypotheses.get("verdict", {}).get("delta_rmse", "insufficient_evidence"))}
        {_verdict_badge(hypotheses.get("verdict", {}).get("delta_r_auc_mse", "insufficient_evidence"))}
      </div>
      {_table_html(
        hypothesis_rows,
        ["metric", "factor", "coef", "t_stat", "p_one_sided", "verdict"],
        {
          "metric": "Metric",
          "factor": "Hypothesis factor",
          "coef": "Coef",
          "t_stat": "t-stat",
          "p_one_sided": "p-value (one-sided)",
          "verdict": "Verdict",
        },
      )}
      <p class="muted">Overall significance test of mean delta across all available points (one-sided, mean delta &gt; 0).</p>
      {_table_html(
        overall_delta_rows,
        ["metric", "n_points", "mean_delta", "ci_low", "ci_high", "p_one_sided", "verdict"],
        {
          "metric": "Metric",
          "n_points": "N points",
          "mean_delta": "Mean delta",
          "ci_low": "CI low",
          "ci_high": "CI high",
          "p_one_sided": "p-value (one-sided)",
          "verdict": "Verdict",
        },
      )}
    </div>

    <h2>Factorial Analysis</h2>
    <div class="section">
      <p class="muted">
        Main-effects model formula: <code>{summary.get("factorial_formula", "metric ~ log_n_samples + size_ratio + log_n_features")}</code>.
      </p>
      {_table_html(
        factorial_overview_rows,
        ["metric", "n_points", "r_squared", "adj_r_squared", "f_p_value", "significant_factors", "verdict"],
        {
          "metric": "Metric",
          "n_points": "N points",
          "r_squared": "R²",
          "adj_r_squared": "Adj. R²",
          "f_p_value": "F-test p-value",
          "significant_factors": "Significant factors",
          "verdict": "Verdict",
        },
      )}
      {_table_html(
        factorial_factor_rows,
        ["metric", "factor", "coef", "p_value", "ci_low", "ci_high", "significant"],
        {
          "metric": "Metric",
          "factor": "Factor",
          "coef": "Coef",
          "p_value": "p-value",
          "ci_low": "CI low",
          "ci_high": "CI high",
          "significant": "Significant",
        },
      )}
      {"".join(factorial_raw_blocks) if factorial_raw_blocks else "<p class='muted'>No statsmodels output available.</p>"}
    </div>

    <div class="footer">
      Report file: {output_path.name}
    </div>
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


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
    available_baselines = _discover_available_baselines(results)

    if baseline not in available_baselines:
        available = ", ".join(available_baselines) if available_baselines else "none"
        raise ValueError(
            f"Requested baseline '{baseline}' not found in results. Available baselines: {available}"
        )

    sm, smf = _require_statsmodels_modules()
    df = extract_metrics_long_form(results, baseline=baseline)
    dataset_feature_df = _build_dataset_feature_df(df)

    size_analysis = build_size_analysis(
        df,
        min_datasets=min_datasets,
        min_size_points=min_size_points,
        alpha=alpha,
    )
    feature_analysis = build_feature_analysis(
        dataset_feature_df,
        min_datasets=min_datasets,
        alpha=alpha,
    )
    joint_analysis = build_joint_analysis(
        df,
        min_datasets=min_datasets,
        min_size_points=min_size_points,
        alpha=alpha,
    )
    hypotheses, factorial_analysis = build_hypotheses_and_factorial_analysis(
        df,
        min_datasets=min_datasets,
        alpha=alpha,
        sm=sm,
        smf=smf,
    )

    status = "insufficient_evidence"
    if not df.empty:
        status = "completed"

    output_dir.mkdir(parents=True, exist_ok=True)
    html_report_path = output_dir / "size_dependence_report.html"

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "n_datasets": int(df["dataset_id"].nunique()) if not df.empty else 0,
            "n_data_points": int(len(df)),
            "n_datasets_with_features": int(dataset_feature_df["dataset_id"].nunique())
            if not dataset_feature_df.empty
            else 0,
            "sizes_tested": sorted(float(v) for v in df["size_ratio"].unique()) if not df.empty else [],
            "baseline": baseline,
            "min_datasets_required": int(min_datasets),
            "min_size_points_required": int(min_size_points),
            "alpha": float(alpha),
            "available_baselines": available_baselines,
            "factorial_formula": "metric ~ log_n_samples + size_ratio + log_n_features",
            "factorial_n_points_total": int(factorial_analysis.get("n_points_total", 0)),
            "factorial_n_points_by_metric": {
                metric: int(factorial_analysis.get("metrics", {}).get(metric, {}).get("n_points", 0))
                for metric in METRICS
            },
        },
        "status": status,
        "size_analysis": size_analysis,
        "feature_analysis": feature_analysis,
        "joint_analysis": joint_analysis,
        "hypotheses": hypotheses,
        "factorial_analysis": factorial_analysis,
        "html_report_path": str(html_report_path),
    }

    generate_html_report(
        report=report,
        df=df,
        dataset_feature_df=dataset_feature_df,
        output_path=html_report_path,
    )

    logger.info(
        "Analysis complete. Size verdicts: ΔRMSE=%s, ΔR-AUC MSE=%s",
        size_analysis["verdict"]["delta_rmse"],
        size_analysis["verdict"]["delta_r_auc_mse"],
    )
    logger.info("HTML report written to %s", html_report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze dependence of HMTL deltas on size and n_features")
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

    logger.info("Report ready: %s", report["html_report_path"])


if __name__ == "__main__":
    main()
