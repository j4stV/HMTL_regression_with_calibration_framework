"""Tests for size dependence analysis with hypotheses and factorial modeling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scripts.analyze_size_dependence as mod


SIZE_GRID = [0.2, 0.4, 0.6, 0.8, 1.0]


def _build_legacy_dataset(
    *,
    dataset_id: int,
    dataset_name: str,
    n_features: int,
    n_samples_train: int,
    sizes: list[float],
    rmse_coeffs: tuple[float, float, float, float],
    rauc_coeffs: tuple[float, float, float, float],
) -> dict:
    payload: dict[str, object] = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "n_features": n_features,
        "n_samples_train": n_samples_train,
        "sizes": {},
    }

    size_block = payload["sizes"]
    assert isinstance(size_block, dict)

    rmse_intercept, rmse_log_samples, rmse_size_ratio, rmse_log_features = rmse_coeffs
    rauc_intercept, rauc_log_samples, rauc_size_ratio, rauc_log_features = rauc_coeffs

    for ratio in sizes:
        n_train = int(max(1, int(round(n_samples_train * ratio))))
        log_samples = float(np.log1p(n_train))
        log_features = float(np.log1p(n_features))
        # Keep deterministic micro-offset so rows are not perfectly collinear.
        tiny_offset = float(dataset_id) * 1e-4 + float(ratio) * 1e-5

        delta_rmse = (
            rmse_intercept
            + rmse_log_samples * log_samples
            + rmse_size_ratio * ratio
            + rmse_log_features * log_features
            + tiny_offset
        )
        delta_rauc = (
            rauc_intercept
            + rauc_log_samples * log_samples
            + rauc_size_ratio * ratio
            + rauc_log_features * log_features
            + tiny_offset
        )

        hmtl_rmse = 1.25 - 0.16 * ratio + 0.01 * dataset_id
        hmtl_rauc = 0.36 - 0.05 * ratio + 0.003 * dataset_id

        size_block[str(int(round(ratio * 100)))] = {
            "size_ratio": ratio,
            "n_train_samples": n_train,
            "hmtl": {"rmse": hmtl_rmse, "r_auc_mse": hmtl_rauc},
            "catboost": {
                "rmse": hmtl_rmse + delta_rmse,
                "r_auc_mse": hmtl_rauc + delta_rauc,
            },
            "delta_rmse": delta_rmse,
            "delta_r_auc_mse": delta_rauc,
        }

    return payload


def _build_factorial_payload(
    *,
    n_datasets: int,
    size_ratios: list[float] | None = None,
    rmse_coeffs: tuple[float, float, float, float],
    rauc_coeffs: tuple[float, float, float, float],
) -> list[dict]:
    ratios = size_ratios or SIZE_GRID
    payload: list[dict] = []
    for idx in range(1, n_datasets + 1):
        payload.append(
            _build_legacy_dataset(
                dataset_id=idx,
                dataset_name=f"d{idx}",
                n_features=10 + 16 * idx,
                n_samples_train=900 + 250 * idx,
                sizes=ratios,
                rmse_coeffs=rmse_coeffs,
                rauc_coeffs=rauc_coeffs,
            )
        )
    return payload


def _write_results(tmp_path: Path, payload: list[dict]) -> Path:
    results_file = tmp_path / "aggregated_results.json"
    results_file.write_text(json.dumps(payload), encoding="utf-8")
    return results_file


def _require_statsmodels() -> None:
    pytest.importorskip("statsmodels")


def test_analyze_size_dependence_positive_hypotheses_and_factorial(tmp_path: Path):
    _require_statsmodels()
    aggregated = _build_factorial_payload(
        n_datasets=9,
        rmse_coeffs=(0.01, 0.032, 0.20, 0.028),
        rauc_coeffs=(0.005, 0.016, 0.11, 0.012),
    )
    results_file = _write_results(tmp_path, aggregated)

    report = mod.analyze_size_dependence(
        results_file=results_file,
        output_dir=tmp_path,
        baseline="catboost",
        min_datasets=6,
        min_size_points=4,
        alpha=0.05,
    )

    assert report["status"] == "completed"
    assert "size_analysis" in report
    assert "feature_analysis" in report
    assert "joint_analysis" in report
    assert "hypotheses" in report
    assert "factorial_analysis" in report

    for metric_col in mod.METRICS:
        metric_hyp = report["hypotheses"][metric_col]
        checks = metric_hyp["checks"]
        assert checks["n_samples_absolute"]["verdict"] == "supported"
        assert checks["size_ratio"]["verdict"] == "supported"
        assert checks["n_features"]["verdict"] == "supported"
        assert metric_hyp["verdict"] == "supported"
        overall_delta = metric_hyp["overall_delta"]
        assert overall_delta["verdict"] == "supported"
        assert overall_delta["mean_delta"] is not None
        assert overall_delta["ci_low"] is not None
        assert overall_delta["ci_high"] is not None
        assert overall_delta["p_value_one_sided_greater"] is not None

        metric_factorial = report["factorial_analysis"]["metrics"][metric_col]
        assert metric_factorial["verdict"] == "completed"
        assert "n_samples_absolute" in metric_factorial["factors"]
        assert "size_ratio" in metric_factorial["factors"]
        assert "n_features" in metric_factorial["factors"]
        assert "OLS Regression Results" in metric_factorial["raw_output"]["summary"]
        assert "sum_sq" in metric_factorial["raw_output"]["anova_type_2"]

    assert 0.2 in report["summary"]["sizes_tested"]
    expected_points = len(aggregated) * len(SIZE_GRID)
    assert report["summary"]["factorial_n_points_total"] == expected_points
    assert report["summary"]["factorial_n_points_by_metric"]["delta_rmse"] == expected_points
    assert report["summary"]["factorial_n_points_by_metric"]["delta_r_auc_mse"] == expected_points

    html_file = tmp_path / "size_dependence_report.html"
    assert html_file.exists()
    html = html_file.read_text(encoding="utf-8")
    assert "Hypotheses" in html
    assert "Factorial Analysis" in html
    assert "OLS Regression Results" in html

    pre_hypothesis, _, _ = html.partition("<h2>Hypotheses</h2>")
    assert "Linear p-value" not in pre_hypothesis
    assert "Two-factor model p-values" not in pre_hypothesis
    assert "p-value t-test (ΔRMSE)" not in pre_hypothesis

    assert "Distribution of ΔRMSE" in html
    assert "Distribution of ΔR-AUC MSE" in html
    assert "Distribution of size slopes (ΔRMSE)" in html
    assert "Distribution of size slopes (ΔR-AUC MSE)" in html
    assert "n_samples vs ΔRMSE (absolute size)" in html
    assert "n_samples vs ΔR-AUC MSE (absolute size)" in html
    assert "Delta by absolute n_samples bins" in html
    assert "Overall significance test of mean delta across all available points" in html


def test_analyze_size_dependence_negative_size_ratio_hypothesis(tmp_path: Path):
    _require_statsmodels()
    aggregated = _build_factorial_payload(
        n_datasets=9,
        rmse_coeffs=(0.02, 0.020, -0.80, 0.030),
        rauc_coeffs=(0.01, 0.010, -0.50, 0.014),
    )
    results_file = _write_results(tmp_path, aggregated)

    report = mod.analyze_size_dependence(
        results_file=results_file,
        output_dir=tmp_path,
        baseline="catboost",
        min_datasets=6,
        min_size_points=4,
        alpha=0.05,
    )

    for metric_col in mod.METRICS:
        check = report["hypotheses"][metric_col]["checks"]["size_ratio"]
        assert check["verdict"] == "not_supported"


def test_analyze_size_dependence_insufficient_evidence_on_too_few_points(tmp_path: Path):
    _require_statsmodels()
    aggregated = _build_factorial_payload(
        n_datasets=1,
        size_ratios=[0.2, 1.0],
        rmse_coeffs=(0.02, 0.03, 0.15, 0.02),
        rauc_coeffs=(0.01, 0.015, 0.08, 0.01),
    )
    results_file = _write_results(tmp_path, aggregated)

    report = mod.analyze_size_dependence(
        results_file=results_file,
        output_dir=tmp_path,
        baseline="catboost",
        min_datasets=5,
        min_size_points=3,
        alpha=0.05,
    )

    for metric_col in mod.METRICS:
        assert report["hypotheses"][metric_col]["verdict"] == "insufficient_evidence"
        assert report["factorial_analysis"]["metrics"][metric_col]["verdict"] == "insufficient_evidence"


def test_analyze_size_dependence_handles_missing_n_samples_for_absolute_size_views(tmp_path: Path):
    _require_statsmodels()
    aggregated = _build_factorial_payload(
        n_datasets=4,
        rmse_coeffs=(0.01, 0.03, 0.18, 0.02),
        rauc_coeffs=(0.005, 0.015, 0.10, 0.01),
    )
    for dataset in aggregated:
        dataset.pop("n_samples_train", None)
        dataset.pop("n_samples_total", None)
        sizes = dataset.get("sizes", {})
        assert isinstance(sizes, dict)
        for size_data in sizes.values():
            assert isinstance(size_data, dict)
            size_data.pop("n_train_samples", None)
            size_data.pop("n_samples_train", None)
            size_data.pop("n_samples", None)

    results_file = _write_results(tmp_path, aggregated)

    report = mod.analyze_size_dependence(
        results_file=results_file,
        output_dir=tmp_path,
        baseline="catboost",
        min_datasets=3,
        min_size_points=3,
        alpha=0.05,
    )

    assert report["status"] == "completed"
    html = (tmp_path / "size_dependence_report.html").read_text(encoding="utf-8")
    assert "n_samples vs ΔRMSE (absolute size)" in html
    assert "n_samples vs ΔR-AUC MSE (absolute size)" in html
    assert "Delta by absolute n_samples bins" in html


def test_analyze_size_dependence_raises_for_missing_baseline(tmp_path: Path):
    aggregated = _build_factorial_payload(
        n_datasets=1,
        rmse_coeffs=(0.02, 0.03, 0.15, 0.02),
        rauc_coeffs=(0.01, 0.015, 0.08, 0.01),
    )
    results_file = _write_results(tmp_path, aggregated)

    with pytest.raises(ValueError, match="Available baselines: catboost"):
        mod.analyze_size_dependence(
            results_file=results_file,
            output_dir=tmp_path,
            baseline="single_mlp",
            min_datasets=5,
            min_size_points=3,
            alpha=0.05,
        )
