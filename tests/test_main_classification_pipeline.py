"""Integration smoke test for classification path in scripts/main.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.main import run_experiment



def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")



def test_main_classification_outputs_classification_metrics(tmp_path: Path, monkeypatch):
    rng = np.random.default_rng(42)

    n_features = 4
    feature_names = [f"f{i}" for i in range(n_features)]

    def make_split(n_rows: int) -> pd.DataFrame:
        X = rng.normal(size=(n_rows, n_features))
        # Build simple 3-class labels from a linear score.
        scores = X[:, 0] + 0.5 * X[:, 1]
        y = np.digitize(scores, bins=[-0.3, 0.4]).astype(int)
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y
        return df

    train_df = make_split(90)
    valid_df = make_split(30)
    test_df = make_split(30)

    train_csv = tmp_path / "train.csv"
    valid_csv = tmp_path / "valid.csv"
    test_csv = tmp_path / "test.csv"
    train_df.to_csv(train_csv, index=False)
    valid_df.to_csv(valid_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    data_cfg = {
        "task": {
            "type": "classification",
            "num_classes": 3,
            "class_weights": None,
            "temperature_scaling": True,
            "use_focal_loss": False,
            "focal_alpha": 0.25,
            "focal_gamma": 2.0,
            "label_smoothing": 0.0,
        },
        "paths": {
            "train_csv": str(train_csv),
            "valid_csv": str(valid_csv),
            "cal_csv": None,
            "test_csv": str(test_csv),
            "target": "target",
        },
        "preprocess": {
            "impute_const": -1.0,
            "use_dynamic_binning": False,
            "quantile_binning": {"enabled": False, "bins": 5},
            "standardize": True,
            "pca": {"enabled": False, "n_components": None},
            "target_standardize": False,
        },
    }

    model_cfg = {
        "encoder": {
            "hidden_width": 16,
            "depth_base": 2,
            "depth_hmtl": 4,
            "alpha_dropout": 0.0,
            "activation": "SELU",
            "init": "lecun",
        },
        "hmtl": {
            "enabled": True,
            "aux_task": "bins",
            "low_layer": 2,
            "high_layer": 4,
            "lambda_aux": 0.3,
            "n_bins": 3,
            "proj_dim": 8,
        },
    }

    train_cfg = {
        "optimizer": {
            "name": "adamw",
            "lr": 0.005,
            "lookahead_sync_period": 6,
            "lookahead_slow_step": 0.5,
            "weight_decay": 0.0,
        },
        "training": {
            "seed": 7,
            "epochs": 2,
            "batch_size": 16,
            "sigma_reg_weight": 0.0,
            "early_stop": {
                "metric": "nll",
                "patience": 2,
                "mode": "min",
            },
        },
        "logging": {
            "mlflow": {"enabled": False, "tracking_uri": None},
            "save.dir": str(tmp_path / "runs"),
        },
    }

    ensemble_cfg = {
        "ensemble": {
            "n_models": 1,
            "bagging": "bootstrap",
            "val_metric": "nll",
        }
    }

    data_yaml = tmp_path / "data.yaml"
    model_yaml = tmp_path / "model.yaml"
    train_yaml = tmp_path / "train.yaml"
    ensemble_yaml = tmp_path / "ensemble.yaml"

    _write_yaml(data_yaml, data_cfg)
    _write_yaml(model_yaml, model_cfg)
    _write_yaml(train_yaml, train_cfg)
    _write_yaml(ensemble_yaml, ensemble_cfg)

    monkeypatch.chdir(tmp_path)

    result = run_experiment(
        data_config=str(data_yaml),
        model_config=str(model_yaml),
        train_config=str(train_yaml),
        ensemble_config=str(ensemble_yaml),
        return_models=False,
    )

    metrics = result["metrics"]
    assert metrics["task_type"] == "classification"
    assert "val_accuracy" in metrics
    assert "val_f1_macro" in metrics
    assert "val_ece" in metrics
    assert "val_rmse" not in metrics
    assert "val_r_auc_mse" not in metrics
