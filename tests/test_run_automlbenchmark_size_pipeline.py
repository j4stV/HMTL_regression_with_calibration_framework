"""Tests for size experiment runner utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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



def test_build_effective_size_configs_balanced_policy():
    base_model_cfg = {
        "encoder": {"hidden_width": 128, "alpha_dropout": 0.0},
        "hmtl": {
            "low_layer": 12,
            "high_layer": 18,
            "n_bins": 5,
            "lambda_aux": 0.5,
            "enabled": True,
            "aux_task": "contrastive",
            "proj_dim": 16,
        },
    }
    base_train_cfg = {
        "optimizer": {"lr": 1e-3, "name": "adamw"},
        "training": {"epochs": 2, "batch_size": 4096, "early_stop": {"patience": 3}},
    }
    base_ensemble_cfg = {"ensemble": {"n_models": 20, "bagging": "stratified_kfold"}}
    base_preprocess = PreprocessConfig(pca_enabled=True, pca_n_components=None)

    tiny = size_script._build_effective_size_configs(
        base_model_cfg=base_model_cfg,
        base_train_cfg_yaml=base_train_cfg,
        base_ensemble_cfg_yaml=base_ensemble_cfg,
        base_preprocess_config=base_preprocess,
        size_ratio=0.2,
        n_train_size=200,
        n_features=20,
    )
    assert tiny["adaptive_policy"]["regime"] == "tiny"
    assert tiny["train_cfg_yaml"]["training"]["batch_size"] == 50
    assert tiny["ensemble_cfg_yaml"]["ensemble"]["n_models"] == 8
    assert tiny["ensemble_cfg_yaml"]["ensemble"]["bagging"] == "stratified_bins"
    assert tiny["ensemble_cfg_yaml"]["ensemble"]["baseline_n_models"] == 20
    assert tiny["model_cfg"]["hmtl"]["enabled"] is True
    assert tiny["model_cfg"]["hmtl"]["aux_task"] == "bins"
    assert tiny["model_cfg"]["hmtl"]["lambda_aux"] == 0.2
    assert tiny["model_cfg"]["encoder"]["hidden_width"] == 64
    assert tiny["model_cfg"]["hmtl"]["high_layer"] == 6
    assert tiny["model_cfg"]["hmtl"]["low_layer"] == 2
    assert tiny["preprocess_config"].pca_enabled is False

    small = size_script._build_effective_size_configs(
        base_model_cfg=base_model_cfg,
        base_train_cfg_yaml=base_train_cfg,
        base_ensemble_cfg_yaml=base_ensemble_cfg,
        base_preprocess_config=base_preprocess,
        size_ratio=0.5,
        n_train_size=1000,
        n_features=300,
    )
    assert small["adaptive_policy"]["regime"] == "small"
    assert small["train_cfg_yaml"]["training"]["batch_size"] == 125
    assert small["ensemble_cfg_yaml"]["ensemble"]["n_models"] == 10
    assert small["ensemble_cfg_yaml"]["ensemble"]["bagging"] == "stratified_bins"
    assert small["model_cfg"]["hmtl"]["enabled"] is True
    assert small["model_cfg"]["hmtl"]["aux_task"] == "bins"
    assert small["model_cfg"]["encoder"]["hidden_width"] == 96
    assert small["model_cfg"]["hmtl"]["high_layer"] == 10
    assert small["model_cfg"]["hmtl"]["low_layer"] == 4
    assert small["preprocess_config"].pca_enabled is True
    assert small["preprocess_config"].pca_n_components == 0.95

    large = size_script._build_effective_size_configs(
        base_model_cfg=base_model_cfg,
        base_train_cfg_yaml=base_train_cfg,
        base_ensemble_cfg_yaml=base_ensemble_cfg,
        base_preprocess_config=base_preprocess,
        size_ratio=1.0,
        n_train_size=5000,
        n_features=200,
    )
    assert large["adaptive_policy"]["regime"] == "large"
    assert large["train_cfg_yaml"]["training"]["batch_size"] == 416
    assert large["ensemble_cfg_yaml"]["ensemble"]["n_models"] == 20
    assert large["ensemble_cfg_yaml"]["ensemble"]["bagging"] == "stratified_kfold"
    assert large["model_cfg"]["hmtl"]["enabled"] is True
    assert large["model_cfg"]["hmtl"]["aux_task"] == "bins"
    assert large["preprocess_config"].pca_enabled is True
    assert large["preprocess_config"].pca_n_components == 0.99


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
    assert "adaptive_policy" in size_50
    assert "effective_config" in size_50
    assert "ensemble_val_metric" in size_50
    assert "ensemble_avg_val_score" in size_50

    seed_payload = size_50["per_seed"]["11"]
    assert "adaptive_policy" in seed_payload
    assert "effective_config" in seed_payload
    assert "ensemble_val_metric" in seed_payload
    assert seed_payload["ensemble_val_metric"] == "hybrid_rmse_rauc"

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
        dataset_ids=None,
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


def test_run_automlbenchmark_experiments_rejects_nonpositive_dataset_workers(tmp_path: Path):
    model_cfg = tmp_path / "model.yaml"
    train_cfg = tmp_path / "train.yaml"
    ensemble_cfg = tmp_path / "ensemble.yaml"
    data_cfg = tmp_path / "data.yaml"

    model_cfg.write_text("{}", encoding="utf-8")
    train_cfg.write_text("{}", encoding="utf-8")
    ensemble_cfg.write_text("{}", encoding="utf-8")
    data_cfg.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="max_dataset_workers must be >= 1"):
        size_script.run_automlbenchmark_experiments(
            model_cfg_path=model_cfg,
            train_cfg_path=train_cfg,
            ensemble_cfg_path=ensemble_cfg,
            data_cfg_path=data_cfg,
            output_dir=tmp_path / "out",
            sizes=[0.5],
            dataset_id=123,
            dataset_ids=None,
            study_id=269,
            seed=42,
            seeds=None,
            n_seeds=2,
            max_datasets=None,
            max_dataset_workers=0,
            baselines=["catboost"],
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            high_level_progress_only=True,
        )


def test_run_automlbenchmark_experiments_accepts_explicit_dataset_ids(monkeypatch, tmp_path: Path):
    model_cfg = tmp_path / "model.yaml"
    train_cfg = tmp_path / "train.yaml"
    ensemble_cfg = tmp_path / "ensemble.yaml"
    data_cfg = tmp_path / "data.yaml"

    model_cfg.write_text("{}", encoding="utf-8")
    train_cfg.write_text("{}", encoding="utf-8")
    ensemble_cfg.write_text("{}", encoding="utf-8")
    data_cfg.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        size_script,
        "get_regression_datasets",
        lambda study_id: (_ for _ in ()).throw(AssertionError("should not query study datasets")),
    )

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
        return {"dataset_id": int(dataset_meta.dataset_id), "dataset_name": dataset_meta.dataset_name, "sizes": {}}

    monkeypatch.setattr(size_script, "run_single_dataset_experiment", fake_run_single_dataset_experiment)

    results = size_script.run_automlbenchmark_experiments(
        model_cfg_path=model_cfg,
        train_cfg_path=train_cfg,
        ensemble_cfg_path=ensemble_cfg,
        data_cfg_path=data_cfg,
        output_dir=tmp_path / "out",
        sizes=[0.5],
        dataset_id=None,
        dataset_ids=[102, 101, 102],
        study_id=269,
        seed=42,
        seeds=None,
        n_seeds=1,
        max_datasets=None,
        max_dataset_workers=1,
        baselines=["catboost"],
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        high_level_progress_only=True,
    )

    assert [item["dataset_id"] for item in results] == [102, 101]


def test_run_automlbenchmark_experiments_rejects_dataset_id_and_dataset_ids(tmp_path: Path):
    model_cfg = tmp_path / "model.yaml"
    train_cfg = tmp_path / "train.yaml"
    ensemble_cfg = tmp_path / "ensemble.yaml"
    data_cfg = tmp_path / "data.yaml"

    model_cfg.write_text("{}", encoding="utf-8")
    train_cfg.write_text("{}", encoding="utf-8")
    ensemble_cfg.write_text("{}", encoding="utf-8")
    data_cfg.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Use either dataset_id or dataset_ids, not both"):
        size_script.run_automlbenchmark_experiments(
            model_cfg_path=model_cfg,
            train_cfg_path=train_cfg,
            ensemble_cfg_path=ensemble_cfg,
            data_cfg_path=data_cfg,
            output_dir=tmp_path / "out",
            sizes=[0.5],
            dataset_id=123,
            dataset_ids=[123, 124],
            study_id=269,
            seed=42,
            seeds=None,
            n_seeds=1,
            max_datasets=None,
            max_dataset_workers=1,
            baselines=["catboost"],
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            high_level_progress_only=True,
        )


def test_run_automlbenchmark_experiments_parallel_branch_and_ordering(monkeypatch, tmp_path: Path):
    model_cfg = tmp_path / "model.yaml"
    train_cfg = tmp_path / "train.yaml"
    ensemble_cfg = tmp_path / "ensemble.yaml"
    data_cfg = tmp_path / "data.yaml"

    model_cfg.write_text("{}", encoding="utf-8")
    train_cfg.write_text("{}", encoding="utf-8")
    ensemble_cfg.write_text("{}", encoding="utf-8")
    data_cfg.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        size_script,
        "get_regression_datasets",
        lambda study_id: [
            {"dataset_id": 101, "name": "dataset_a", "task_id": 1},
            {"dataset_id": 102, "name": "dataset_b", "task_id": 2},
        ],
    )

    captured: dict[str, object] = {
        "submitted": [],
        "called_flags": [],
    }

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
        cast_flags = captured["called_flags"]
        assert isinstance(cast_flags, list)
        cast_flags.append(
            {
                "dataset_id": int(dataset_meta.dataset_id),
                "show_trial_progress": bool(show_trial_progress),
                "show_inner_progress": bool(show_inner_progress),
            }
        )
        return {"dataset_id": int(dataset_meta.dataset_id), "dataset_name": dataset_meta.dataset_name, "sizes": {}}

    monkeypatch.setattr(size_script, "run_single_dataset_experiment", fake_run_single_dataset_experiment)

    class _FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _FakeExecutor:
        def __init__(self, *, max_workers, mp_context):
            captured["max_workers"] = int(max_workers)
            captured["mp_context_start_method"] = mp_context.get_start_method()
            self._futures = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, **kwargs):
            submitted = captured["submitted"]
            assert isinstance(submitted, list)
            submitted.append(
                {
                    "fn_name": fn.__name__,
                    "dataset_id": int(kwargs["dataset_meta"].dataset_id),
                    "show_trial_progress": bool(kwargs["show_trial_progress"]),
                    "show_inner_progress": bool(kwargs["show_inner_progress"]),
                }
            )
            future = _FakeFuture(fn(**kwargs))
            self._futures.append(future)
            return future

    def fake_as_completed(future_to_dataset):
        ordered = list(future_to_dataset.keys())
        return iter(list(reversed(ordered)))

    monkeypatch.setattr(size_script.futures, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(size_script.futures, "as_completed", fake_as_completed)

    results = size_script.run_automlbenchmark_experiments(
        model_cfg_path=model_cfg,
        train_cfg_path=train_cfg,
        ensemble_cfg_path=ensemble_cfg,
        data_cfg_path=data_cfg,
        output_dir=tmp_path / "out",
        sizes=[0.5],
        dataset_id=None,
        dataset_ids=None,
        study_id=269,
        seed=42,
        seeds=None,
        n_seeds=2,
        max_datasets=None,
        max_dataset_workers=2,
        baselines=["catboost"],
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        high_level_progress_only=True,
    )

    assert captured["max_workers"] == 2
    assert captured["mp_context_start_method"] == "spawn"

    submitted = captured["submitted"]
    assert isinstance(submitted, list)
    assert [item["dataset_id"] for item in submitted] == [101, 102]
    assert all(item["show_trial_progress"] is False for item in submitted)
    assert all(item["show_inner_progress"] is False for item in submitted)

    called_flags = captured["called_flags"]
    assert isinstance(called_flags, list)
    assert len(called_flags) == 2
    assert all(item["show_trial_progress"] is False for item in called_flags)
    assert all(item["show_inner_progress"] is False for item in called_flags)

    # Completion is reversed, but final ordering must follow input dataset order.
    assert [item["dataset_id"] for item in results] == [101, 102]

    aggregated_file = tmp_path / "out" / "aggregated_results.json"
    aggregated_payload = json.loads(aggregated_file.read_text(encoding="utf-8"))
    assert [item["dataset_id"] for item in aggregated_payload] == [101, 102]


def test_run_automlbenchmark_experiments_parallel_future_failure(monkeypatch, tmp_path: Path):
    model_cfg = tmp_path / "model.yaml"
    train_cfg = tmp_path / "train.yaml"
    ensemble_cfg = tmp_path / "ensemble.yaml"
    data_cfg = tmp_path / "data.yaml"

    model_cfg.write_text("{}", encoding="utf-8")
    train_cfg.write_text("{}", encoding="utf-8")
    ensemble_cfg.write_text("{}", encoding="utf-8")
    data_cfg.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        size_script,
        "get_regression_datasets",
        lambda study_id: [
            {"dataset_id": 201, "name": "dataset_ok", "task_id": 11},
            {"dataset_id": 202, "name": "dataset_fail", "task_id": 12},
        ],
    )

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
        return {"dataset_id": int(dataset_meta.dataset_id), "dataset_name": dataset_meta.dataset_name, "sizes": {}}

    monkeypatch.setattr(size_script, "run_single_dataset_experiment", fake_run_single_dataset_experiment)

    class _FakeFuture:
        def __init__(self, *, value=None, error: Exception | None = None):
            self._value = value
            self._error = error

        def result(self):
            if self._error is not None:
                raise self._error
            return self._value

    class _FakeExecutor:
        def __init__(self, *, max_workers, mp_context):
            self._futures = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, **kwargs):
            dataset_id = int(kwargs["dataset_meta"].dataset_id)
            if dataset_id == 202:
                future = _FakeFuture(error=RuntimeError("worker failed"))
            else:
                future = _FakeFuture(value=fn(**kwargs))
            self._futures.append(future)
            return future

    def fake_as_completed(future_to_dataset):
        ordered = list(future_to_dataset.keys())
        return iter(list(reversed(ordered)))

    monkeypatch.setattr(size_script.futures, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(size_script.futures, "as_completed", fake_as_completed)

    results = size_script.run_automlbenchmark_experiments(
        model_cfg_path=model_cfg,
        train_cfg_path=train_cfg,
        ensemble_cfg_path=ensemble_cfg,
        data_cfg_path=data_cfg,
        output_dir=tmp_path / "out",
        sizes=[0.5],
        dataset_id=None,
        dataset_ids=None,
        study_id=269,
        seed=42,
        seeds=None,
        n_seeds=2,
        max_datasets=None,
        max_dataset_workers=2,
        baselines=["catboost"],
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        high_level_progress_only=True,
    )

    # Final results remain in original dataset order.
    assert [item["dataset_id"] for item in results] == [201, 202]
    assert "error" not in results[0]
    assert results[1]["error"] == "worker failed"
    assert results[1]["dataset_name"] == "dataset_fail"
    assert results[1]["task_id"] == 12
    assert "run_meta" in results[1]
    assert results[1]["run_meta"]["study_id"] == 269
    assert results[1]["run_meta"]["seed_list"] == [42, 43]
    assert results[1]["run_meta"]["sizes"] == [0.5]
    assert results[1]["run_meta"]["baselines"] == ["catboost"]
    assert "timestamp_utc" in results[1]["run_meta"]


def test_run_automlbenchmark_experiments_parallel_keyboard_interrupt_forces_shutdown(
    monkeypatch,
    tmp_path: Path,
):
    model_cfg = tmp_path / "model.yaml"
    train_cfg = tmp_path / "train.yaml"
    ensemble_cfg = tmp_path / "ensemble.yaml"
    data_cfg = tmp_path / "data.yaml"

    model_cfg.write_text("{}", encoding="utf-8")
    train_cfg.write_text("{}", encoding="utf-8")
    ensemble_cfg.write_text("{}", encoding="utf-8")
    data_cfg.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        size_script,
        "get_regression_datasets",
        lambda study_id: [
            {"dataset_id": 301, "name": "dataset_a", "task_id": 21},
            {"dataset_id": 302, "name": "dataset_b", "task_id": 22},
        ],
    )
    monkeypatch.setattr(
        size_script,
        "run_single_dataset_experiment",
        lambda **kwargs: {
            "dataset_id": int(kwargs["dataset_meta"].dataset_id),
            "dataset_name": kwargs["dataset_meta"].dataset_name,
            "sizes": {},
        },
    )

    captured: dict[str, object] = {}

    class _FakeProcess:
        def __init__(self):
            self.terminated = False
            self.killed = False
            self.join_calls = 0
            self.alive = True

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True
            self.alive = False

        def kill(self):
            self.killed = True
            self.alive = False

        def join(self, timeout=None):
            self.join_calls += 1

    class _FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _FakeExecutor:
        def __init__(self, *, max_workers, mp_context):
            self._processes = {0: _FakeProcess(), 1: _FakeProcess()}
            self.shutdown_calls: list[tuple[bool, bool | None]] = []
            captured["executor"] = self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, **kwargs):
            return _FakeFuture(fn(**kwargs))

        def shutdown(self, wait=True, cancel_futures=None):
            self.shutdown_calls.append((bool(wait), cancel_futures))

    def fake_as_completed(future_to_dataset):
        def _gen():
            raise KeyboardInterrupt()
            yield from future_to_dataset

        return _gen()

    monkeypatch.setattr(size_script.futures, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(size_script.futures, "as_completed", fake_as_completed)

    with pytest.raises(KeyboardInterrupt):
        size_script.run_automlbenchmark_experiments(
            model_cfg_path=model_cfg,
            train_cfg_path=train_cfg,
            ensemble_cfg_path=ensemble_cfg,
            data_cfg_path=data_cfg,
            output_dir=tmp_path / "out",
            sizes=[0.5],
            dataset_id=None,
            dataset_ids=None,
            study_id=269,
            seed=42,
            seeds=None,
            n_seeds=2,
            max_datasets=None,
            max_dataset_workers=2,
            baselines=["catboost"],
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            high_level_progress_only=True,
        )

    executor = captured["executor"]
    assert isinstance(executor, _FakeExecutor)
    assert (False, True) in executor.shutdown_calls
    assert all(proc.terminated for proc in executor._processes.values())
