"""Tests for size/feature dependence analysis HTML report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_size_dependence import analyze_size_dependence


def _build_legacy_dataset(
    *,
    dataset_id: int,
    dataset_name: str,
    n_features: int | None,
    sizes: list[float],
    base_delta_rmse: float,
    base_delta_rauc: float,
) -> dict:
    payload: dict[str, object] = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "sizes": {},
    }
    if n_features is not None:
        payload["n_features"] = n_features

    size_block = payload["sizes"]
    assert isinstance(size_block, dict)

    for ratio in sizes:
        key = str(int(round(ratio * 100)))
        hmtl_rmse = 1.0 - 0.1 * ratio
        hmtl_rauc = 0.30 - 0.05 * ratio
        delta_rmse = base_delta_rmse + 0.04 * ratio
        delta_rauc = base_delta_rauc + 0.02 * ratio
        size_block[key] = {
            "size_ratio": ratio,
            "hmtl": {"rmse": hmtl_rmse, "r_auc_mse": hmtl_rauc},
            "catboost": {
                "rmse": hmtl_rmse + delta_rmse,
                "r_auc_mse": hmtl_rauc + delta_rauc,
            },
            "delta_rmse": delta_rmse,
            "delta_r_auc_mse": delta_rauc,
        }

    return payload


def test_analyze_size_dependence_insufficient_evidence(tmp_path: Path):
    aggregated = [
        {
            "dataset_id": 1,
            "dataset_name": "toy",
            "n_features": 10,
            "sizes": {
                "50": {
                    "size_ratio": 0.5,
                    "hmtl": {"rmse": 1.0, "r_auc_mse": 0.20},
                    "catboost": {"rmse": 1.1, "r_auc_mse": 0.25},
                    "delta_rmse": 0.1,
                    "delta_r_auc_mse": 0.05,
                },
                "100": {
                    "size_ratio": 1.0,
                    "hmtl": {"rmse": 0.9, "r_auc_mse": 0.18},
                    "catboost": {"rmse": 1.0, "r_auc_mse": 0.21},
                    "delta_rmse": 0.1,
                    "delta_r_auc_mse": 0.03,
                },
            },
        }
    ]

    results_file = tmp_path / "aggregated_results.json"
    results_file.write_text(json.dumps(aggregated), encoding="utf-8")

    report = analyze_size_dependence(
        results_file=results_file,
        output_dir=tmp_path,
        baseline="catboost",
        min_datasets=2,
        min_size_points=3,
        alpha=0.05,
    )

    assert report["size_analysis"]["verdict"]["delta_rmse"] == "insufficient_evidence"
    assert report["size_analysis"]["verdict"]["delta_r_auc_mse"] == "insufficient_evidence"

    html_file = tmp_path / "size_dependence_report.html"
    assert html_file.exists()
    html = html_file.read_text(encoding="utf-8")
    assert "Size Analysis" in html
    assert "Feature Analysis" in html
    assert "Joint Analysis" in html


def test_analyze_size_dependence_feature_and_joint_analysis(tmp_path: Path):
    aggregated = [
        _build_legacy_dataset(
            dataset_id=1,
            dataset_name="d1",
            n_features=10,
            sizes=[0.4, 0.7, 1.0],
            base_delta_rmse=0.01,
            base_delta_rauc=0.005,
        ),
        _build_legacy_dataset(
            dataset_id=2,
            dataset_name="d2",
            n_features=30,
            sizes=[0.4, 0.7, 1.0],
            base_delta_rmse=0.03,
            base_delta_rauc=0.012,
        ),
        _build_legacy_dataset(
            dataset_id=3,
            dataset_name="d3",
            n_features=60,
            sizes=[0.4, 0.7, 1.0],
            base_delta_rmse=0.05,
            base_delta_rauc=0.02,
        ),
        _build_legacy_dataset(
            dataset_id=4,
            dataset_name="d4",
            n_features=120,
            sizes=[0.4, 0.7, 1.0],
            base_delta_rmse=0.08,
            base_delta_rauc=0.03,
        ),
    ]
    results_file = tmp_path / "aggregated_results.json"
    results_file.write_text(json.dumps(aggregated), encoding="utf-8")

    report = analyze_size_dependence(
        results_file=results_file,
        output_dir=tmp_path,
        baseline="catboost",
        min_datasets=3,
        min_size_points=3,
        alpha=0.05,
    )

    assert report["feature_analysis"]["n_datasets_with_features"] == 4
    assert report["feature_analysis"]["relationships"]["delta_rmse"]["n_points"] == 4
    per_dataset = {
        str(row["dataset_id"]): row
        for row in report["feature_analysis"]["per_dataset"]
    }
    assert per_dataset["1"]["mean_delta_rmse"] == pytest.approx(0.05)
    assert per_dataset["2"]["mean_delta_rmse"] == pytest.approx(0.07)
    assert all(row["n_size_points"] == 1 for row in per_dataset.values())

    assert len(report["joint_analysis"]["slope_vs_features"]["delta_rmse"]) == 4
    assert report["joint_analysis"]["size_feature_matrix"]["delta_rmse"]["values"]
    joint_two_factor_rmse = report["joint_analysis"]["two_factor_model"]["delta_rmse"]
    assert joint_two_factor_rmse["n_features"]["p_value"] is not None
    assert joint_two_factor_rmse["size_ratio"]["p_value"] is not None

    html = (tmp_path / "size_dependence_report.html").read_text(encoding="utf-8")
    assert "plotly-graph-div" in html
    assert "p-value n_features" in html
    assert "p-value size_ratio" in html


def test_analyze_size_dependence_missing_n_features_excluded_from_feature_blocks(tmp_path: Path):
    aggregated = [
        _build_legacy_dataset(
            dataset_id=1,
            dataset_name="d1",
            n_features=8,
            sizes=[0.3, 0.6, 1.0],
            base_delta_rmse=0.02,
            base_delta_rauc=0.01,
        ),
        _build_legacy_dataset(
            dataset_id=2,
            dataset_name="d2",
            n_features=None,
            sizes=[0.3, 0.6, 1.0],
            base_delta_rmse=0.01,
            base_delta_rauc=0.005,
        ),
        _build_legacy_dataset(
            dataset_id=3,
            dataset_name="d3",
            n_features=64,
            sizes=[0.3, 0.6, 1.0],
            base_delta_rmse=0.04,
            base_delta_rauc=0.02,
        ),
    ]
    results_file = tmp_path / "aggregated_results.json"
    results_file.write_text(json.dumps(aggregated), encoding="utf-8")

    report = analyze_size_dependence(
        results_file=results_file,
        output_dir=tmp_path,
        baseline="catboost",
        min_datasets=2,
        min_size_points=3,
        alpha=0.05,
    )

    assert report["summary"]["n_datasets"] == 3
    assert report["summary"]["n_datasets_with_features"] == 2
    assert report["feature_analysis"]["n_datasets_with_features"] == 2
    assert report["size_analysis"]["n_unique_sizes"] == 3


def test_analyze_size_dependence_raises_for_missing_baseline(tmp_path: Path):
    aggregated = [
        _build_legacy_dataset(
            dataset_id=1,
            dataset_name="d1",
            n_features=10,
            sizes=[0.5, 1.0],
            base_delta_rmse=0.02,
            base_delta_rauc=0.01,
        )
    ]
    results_file = tmp_path / "aggregated_results.json"
    results_file.write_text(json.dumps(aggregated), encoding="utf-8")

    with pytest.raises(ValueError, match="Available baselines: catboost"):
        analyze_size_dependence(
            results_file=results_file,
            output_dir=tmp_path,
            baseline="single_mlp",
            min_datasets=2,
            min_size_points=2,
            alpha=0.05,
        )
