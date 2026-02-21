"""Tests for size-dependence analysis script."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_size_dependence import analyze_size_dependence



def test_analyze_size_dependence_insufficient_evidence(tmp_path: Path):
    # One dataset with only two size points -> insufficient for default criteria.
    aggregated = [
        {
            "dataset_id": 1,
            "dataset_name": "toy",
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

    assert report["verdict"]["delta_rmse"] == "insufficient_evidence"
    assert report["verdict"]["delta_r_auc_mse"] == "insufficient_evidence"

    saved_report = json.loads((tmp_path / "statistical_analysis.json").read_text(encoding="utf-8"))
    assert saved_report["verdict"]["delta_rmse"] == "insufficient_evidence"
