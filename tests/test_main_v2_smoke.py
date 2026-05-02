"""Smoke test for ``scripts/main_v2.py`` — exercises the full YAML → API flow."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _write_csv(path: Path, n: int = 200, d: int = 4, seed: int = 0) -> None:
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    y = X @ w + 0.3 * rng.normal(size=n)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(d)])
    df["target"] = y
    df.to_csv(path, index=False)


def _write_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


def test_main_v2_runs_end_to_end(tmp_path):
    train_csv = tmp_path / "train.csv"
    valid_csv = tmp_path / "valid.csv"
    test_csv = tmp_path / "test.csv"
    _write_csv(train_csv, n=200, seed=0)
    _write_csv(valid_csv, n=50, seed=1)
    _write_csv(test_csv, n=50, seed=2)

    data_yaml = tmp_path / "data.yaml"
    _write_yaml(
        data_yaml,
        {
            "paths": {
                "train_csv": str(train_csv),
                "valid_csv": str(valid_csv),
                "test_csv": str(test_csv),
                "target": "target",
            },
            "preprocess": {
                "impute_const": -1.0,
                "standardize": True,
                "pca": {"enabled": False, "n_components": None},
                "target_standardize": True,
            },
        },
    )
    model_yaml = tmp_path / "model.yaml"
    _write_yaml(
        model_yaml,
        {
            "encoder": {"hidden_width": 32, "alpha_dropout": 0.0, "residual": True},
            "hmtl": {
                "enabled": True,
                "aux_task": "contrastive",
                "low_layer": 2,
                "high_layer": 3,
                "lambda_aux": 0.3,
                "n_bins": 5,
                "proj_dim": 16,
            },
        },
    )
    train_yaml = tmp_path / "train.yaml"
    _write_yaml(
        train_yaml,
        {
            "optimizer": {
                "name": "adamw",
                "lr": 1e-3,
                "grad_clip_norm": 1.0,
                "scheduler": {"name": "none"},
            },
            "amp": {"enabled": False, "dtype": "fp16"},
            "training": {
                "seed": 0,
                "epochs": 4,
                "batch_size": 64,
                "early_stop": {"metric": "r_auc_mse", "patience": 3},
            },
            "conformal": {"method": "symmetric"},
        },
    )
    ensemble_yaml = tmp_path / "ensemble.yaml"
    _write_yaml(
        ensemble_yaml,
        {"ensemble": {"n_models": 2, "bagging": "full_dataset"}},
    )

    run_dir = tmp_path / "run"
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "scripts" / "main_v2.py"),
        "--data",
        str(data_yaml),
        "--model",
        str(model_yaml),
        "--train",
        str(train_yaml),
        "--ensemble",
        str(ensemble_yaml),
        "--output",
        str(run_dir),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"main_v2 failed: {r.stderr}\n{r.stdout}"

    # Resulting manifest should be populated.
    manifest_path = run_dir / "manifest.json"
    assert manifest_path.exists()
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["task_type"] == "regression"
    assert manifest["n_models"] == 2
