"""Tests AMP config propagation from YAML into run_experiment runtime calls."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.main import run_experiment


def _write_yaml(path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_run_experiment_amp_config_propagation(tmp_path, monkeypatch) -> None:
    rng = np.random.default_rng(7)
    train_df = pd.DataFrame(
        {
            "f1": rng.normal(size=32),
            "f2": rng.normal(size=32),
            "target": rng.normal(size=32),
        }
    )
    valid_df = pd.DataFrame(
        {
            "f1": rng.normal(size=16),
            "f2": rng.normal(size=16),
            "target": rng.normal(size=16),
        }
    )

    train_csv = tmp_path / "train.csv"
    valid_csv = tmp_path / "valid.csv"
    train_df.to_csv(train_csv, index=False)
    valid_df.to_csv(valid_csv, index=False)

    data_yaml = tmp_path / "data.yaml"
    _write_yaml(
        data_yaml,
        f"""
paths:
  train_csv: {train_csv}
  valid_csv: {valid_csv}
  cal_csv: null
  test_csv: null
  target: target
preprocess:
  impute_const: -1.0
  quantile_binning:
    enabled: false
    bins: 5
  standardize: true
  pca:
    enabled: false
    n_components: null
  target_standardize: true
""".strip(),
    )

    model_yaml = tmp_path / "model.yaml"
    _write_yaml(
        model_yaml,
        """
encoder:
  hidden_width: 16
  alpha_dropout: 0.0
hmtl:
  low_layer: 2
  high_layer: 4
  n_bins: 5
  lambda_aux: 0.3
  enabled: true
  aux_task: bins
  proj_dim: 8
""".strip(),
    )

    train_yaml = tmp_path / "train.yaml"
    _write_yaml(
        train_yaml,
        """
optimizer:
  name: adamw
  lr: 0.001
  lookahead_sync_period: 5
  lookahead_slow_step: 0.5
  weight_decay: 0.0
training:
  seed: 1
  epochs: 2
  batch_size: 8
  sigma_reg_weight: 0.0
  amp:
    enabled: true
    dtype: bf16
    eval_enabled: true
  early_stop:
    metric: r_auc_mse
    patience: 2
    mode: min
logging:
  mlflow:
    enabled: false
    tracking_uri: null
  save.dir: experiments/runs
""".strip(),
    )

    ensemble_yaml = tmp_path / "ensemble.yaml"
    _write_yaml(
        ensemble_yaml,
        """
ensemble:
  n_models: 1
  bagging: bootstrap
  val_metric: r_auc_mse
""".strip(),
    )

    captured: dict[str, object] = {}

    class _DummyModel:
        def eval(self):
            return self

        def __call__(self, x):
            import torch

            mu = torch.zeros((x.shape[0], 1), device=x.device)
            sigma = torch.ones((x.shape[0], 1), device=x.device)
            return mu, sigma, None

    def fake_fit_ensemble(*args, **kwargs):
        captured["train_cfg"] = kwargs["train_cfg"]
        return [_DummyModel()], 0.0

    def fake_evaluate_on_dataset(*args, **kwargs):
        captured["eval_amp_enabled"] = kwargs["amp_enabled"]
        captured["eval_amp_dtype"] = kwargs["amp_dtype"]
        metrics = SimpleNamespace(
            rmse=1.0,
            mse=1.0,
            mae=1.0,
            r_auc_mse=1.0,
            mean_uncertainty=1.0,
            mean_epistemic=0.5,
            mean_aleatoric=0.5,
            rejection_ratio=None,
            rejection_auc=None,
            f_beta_auc=None,
            f_beta_95=None,
        )
        return SimpleNamespace(
            metrics=metrics,
            pi_metrics_after={},
            error_retention_x=np.array([1.0]),
            error_retention_y=np.array([1.0]),
        )

    monkeypatch.setattr("scripts.main.fit_ensemble", fake_fit_ensemble)
    monkeypatch.setattr("scripts.main.evaluate_on_dataset", fake_evaluate_on_dataset)
    monkeypatch.setattr("scripts.main.visualize_evaluation_results", lambda *args, **kwargs: None)

    run_experiment(
        data_config=data_yaml,
        model_config=model_yaml,
        train_config=train_yaml,
        ensemble_config=ensemble_yaml,
        return_models=False,
    )

    train_cfg = captured["train_cfg"]
    assert train_cfg.amp_enabled is True
    assert train_cfg.amp_dtype == "bf16"
    assert train_cfg.amp_eval_enabled is True

    assert captured["eval_amp_enabled"] is True
    assert captured["eval_amp_dtype"] == "bf16"
