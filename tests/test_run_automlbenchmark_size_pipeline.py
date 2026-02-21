"""Tests for size experiment runner utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import scripts.run_automlbenchmark_experiment as size_script
from scripts.run_automlbenchmark_experiment import DatasetMeta
from src.data.preprocess import PreprocessConfig



def _make_df(n_rows: int = 40) -> pd.DataFrame:
    x = np.linspace(0, 1, n_rows)
    return pd.DataFrame(
        {
            "f1": x,
            "f2": x * 2,
            "target": 10 + 3 * x,
        }
    )



def test_prepare_preprocessed_splits_for_size_refits_preprocessor():
    df = _make_df(40)
    df_train = df.iloc[:24].copy()
    df_valid = df.iloc[24:32].copy()
    df_test = df.iloc[32:].copy()

    cfg = PreprocessConfig(
        impute_const=-1.0,
        use_dynamic_binning=False,
        quantile_binning_enabled=False,
        standardize=True,
        pca_enabled=False,
        target_standardize=True,
    )

    split_small = size_script.prepare_preprocessed_splits_for_size(
        df_train_full=df_train,
        df_valid=df_valid,
        df_test=df_test,
        target_column="target",
        preprocess_config=cfg,
        size_ratio=0.5,
        seed=1,
    )
    split_full = size_script.prepare_preprocessed_splits_for_size(
        df_train_full=df_train,
        df_valid=df_valid,
        df_test=df_test,
        target_column="target",
        preprocess_config=cfg,
        size_ratio=1.0,
        seed=1,
    )

    assert split_small["n_train_samples"] < split_full["n_train_samples"]
    assert split_small["preprocessor"] is not split_full["preprocessor"]



def test_run_single_dataset_experiment_smoke_with_two_baselines(monkeypatch, tmp_path: Path):
    df = _make_df(30)

    def fake_load_dataset(dataset_id: int):
        return df.copy(), "target"

    def fake_run_size_seed_trial(
        *,
        size_ratio,
        seed,
        df_train_full,
        df_valid,
        df_test,
        target_column,
        preprocess_config,
        model_cfg,
        train_cfg_yaml,
        ensemble_cfg_yaml,
        baselines,
        show_inner_progress=True,
    ):
        hmtl_rmse = 1.0 - 0.1 * size_ratio + 0.01 * seed
        hmtl_rauc = 0.20 - 0.02 * size_ratio + 0.005 * seed

        result = {
            "seed": seed,
            "status": "ok",
            "n_train_samples": int(len(df_train_full) * size_ratio),
            "hmtl": {
                "rmse": hmtl_rmse,
                "mse": hmtl_rmse ** 2,
                "mae": hmtl_rmse * 0.8,
                "r_auc_mse": hmtl_rauc,
                "mean_uncertainty": 0.5,
                "mean_epistemic": 0.2,
                "mean_aleatoric": 0.3,
            },
            "baselines": {},
            "delta_vs_hmtl": {},
        }

        for baseline_name in baselines:
            shift = 0.05 if baseline_name == "catboost" else 0.03
            baseline_metrics = {
                "rmse": hmtl_rmse + shift,
                "mse": (hmtl_rmse + shift) ** 2,
                "mae": (hmtl_rmse + shift) * 0.8,
                "r_auc_mse": hmtl_rauc + shift / 2,
                "mean_uncertainty": 0.4,
                "mean_epistemic": 0.1,
                "mean_aleatoric": 0.3,
            }
            result["baselines"][baseline_name] = baseline_metrics
            result["delta_vs_hmtl"][baseline_name] = {
                "delta_rmse": baseline_metrics["rmse"] - result["hmtl"]["rmse"],
                "delta_r_auc_mse": baseline_metrics["r_auc_mse"] - result["hmtl"]["r_auc_mse"],
            }

        return result

    monkeypatch.setattr(size_script, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(size_script, "run_size_seed_trial", fake_run_size_seed_trial)

    model_cfg = {
        "encoder": {"hidden_width": 16, "alpha_dropout": 0.0},
        "hmtl": {
            "low_layer": 2,
            "high_layer": 4,
            "n_bins": 3,
            "lambda_aux": 0.5,
            "enabled": True,
            "aux_task": "bins",
            "proj_dim": 8,
        },
    }
    train_cfg = {
        "optimizer": {
            "lr": 1e-3,
            "name": "adamw",
            "lookahead_sync_period": 6,
            "lookahead_slow_step": 0.5,
            "weight_decay": 0.0,
        },
        "training": {
            "epochs": 1,
            "batch_size": 8,
            "sigma_reg_weight": 0.0,
            "early_stop": {"patience": 1},
        },
    }
    ensemble_cfg = {"ensemble": {"n_models": 2, "bagging": "bootstrap"}}

    result = size_script.run_single_dataset_experiment(
        dataset_meta=DatasetMeta(dataset_id=123, dataset_name="Toy Dataset", task_id=555),
        sizes=[0.5, 1.0],
        seeds=[11, 12],
        model_cfg=model_cfg,
        train_cfg_yaml=train_cfg,
        ensemble_cfg_yaml=ensemble_cfg,
        preprocess_config=PreprocessConfig(pca_enabled=False),
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        baselines=["catboost", "single_mlp"],
        split_seed=42,
        output_dir=tmp_path,
        study_id=269,
        config_paths={"model": "m", "train": "t", "ensemble": "e", "data": "d"},
    )

    assert "50" in result["sizes"]
    assert "100" in result["sizes"]

    size_50 = result["sizes"]["50"]
    assert size_50["aggregate_over_seeds"]["n_requested"] == 2
    assert size_50["aggregate_over_seeds"]["n_successful"] == 2
    assert "catboost" in size_50["baselines"]
    assert "single_mlp" in size_50["baselines"]

    # Backward-compatible aliases for CatBoost must exist.
    assert "catboost" in size_50
    assert "delta_rmse" in size_50
    assert "delta_r_auc_mse" in size_50

    saved_path = tmp_path / "dataset_123_toy_dataset" / "results.json"
    assert saved_path.exists()


def test_run_automlbenchmark_experiments_high_level_progress_only(monkeypatch, tmp_path: Path):
    model_cfg = tmp_path / "model.yaml"
    train_cfg = tmp_path / "train.yaml"
    ensemble_cfg = tmp_path / "ensemble.yaml"
    data_cfg = tmp_path / "data.yaml"

    model_cfg.write_text("{}", encoding="utf-8")
    train_cfg.write_text("{}", encoding="utf-8")
    ensemble_cfg.write_text("{}", encoding="utf-8")
    data_cfg.write_text("{}", encoding="utf-8")

    captured: dict[str, bool] = {}

    def fake_run_single_dataset_experiment(
        *,
        dataset_meta,
        sizes,
        seeds,
        model_cfg,
        train_cfg_yaml,
        ensemble_cfg_yaml,
        preprocess_config,
        train_ratio,
        val_ratio,
        test_ratio,
        baselines,
        split_seed,
        output_dir,
        study_id,
        config_paths,
        show_trial_progress=False,
        show_inner_progress=True,
    ):
        captured["show_trial_progress"] = bool(show_trial_progress)
        captured["show_inner_progress"] = bool(show_inner_progress)
        return {"dataset_id": int(dataset_meta.dataset_id), "sizes": {}}

    monkeypatch.setattr(size_script, "run_single_dataset_experiment", fake_run_single_dataset_experiment)

    results = size_script.run_automlbenchmark_experiments(
        model_cfg_path=model_cfg,
        train_cfg_path=train_cfg,
        ensemble_cfg_path=ensemble_cfg,
        data_cfg_path=data_cfg,
        output_dir=tmp_path / "out",
        sizes=[0.5],
        dataset_id=123,
        study_id=269,
        seed=42,
        seeds=None,
        n_seeds=2,
        max_datasets=None,
        baselines=["catboost"],
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        high_level_progress_only=True,
    )

    assert len(results) == 1
    assert captured["show_trial_progress"] is True
    assert captured["show_inner_progress"] is False
