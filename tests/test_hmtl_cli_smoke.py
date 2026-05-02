"""Smoke tests for the ``hmtl`` CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _synthetic_csv(path: Path, n: int = 300, d: int = 5, seed: int = 0) -> None:
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    y = X @ w + 0.3 * rng.normal(size=n)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(d)])
    df["target"] = y
    df.to_csv(path, index=False)


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "src.hmtl", *args]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_cli_train_predict_info(tmp_path):
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    _synthetic_csv(train_csv, n=300, seed=0)
    _synthetic_csv(test_csv, n=60, seed=1)

    run_dir = tmp_path / "run"
    preds_csv = tmp_path / "preds.csv"

    # train
    r = _run_cli(
        [
            "train",
            str(train_csv),
            "--target",
            "target",
            "--output",
            str(run_dir),
            "--preset",
            "fast",
            "--n-models",
            "2",
            "--epochs",
            "4",
        ]
    )
    assert r.returncode == 0, f"train failed: {r.stderr}\n{r.stdout}"
    assert (run_dir / "manifest.json").exists()

    # predict
    r = _run_cli(
        [
            "predict",
            str(run_dir),
            str(test_csv),
            "--out",
            str(preds_csv),
            "--with-uncertainty",
            "--coverage",
            "0.9",
        ]
    )
    assert r.returncode == 0, f"predict failed: {r.stderr}\n{r.stdout}"
    assert preds_csv.exists()

    out = pd.read_csv(preds_csv)
    assert "prediction" in out.columns
    assert "uncertainty" in out.columns
    assert "lower_90" in out.columns
    assert "upper_90" in out.columns
    assert len(out) == 60

    # info
    r = _run_cli(["info", str(run_dir)])
    assert r.returncode == 0
    manifest = json.loads(r.stdout)
    assert manifest["task_type"] == "regression"
    assert manifest["n_models"] == 2
