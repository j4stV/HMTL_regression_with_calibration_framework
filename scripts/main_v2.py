"""Thin drop-in replacement for ``scripts/main.py`` using the new HMTL API.

Same four-YAML interface as the legacy flow, but the training / evaluation /
conformal-calibration pipeline is delegated to
:class:`src.hmtl.estimator.HMTLRegressor` (or ``HMTLClassifier``). ~50 lines of
glue instead of 700.

The original ``main.py`` is **not** modified — this lives alongside it so you
can compare runs and fall back if needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.hmtl import HMTLClassifier, HMTLRegressor
from src.hmtl.config import Config
from src.utils.logger import get_logger, setup_logging


def _load_csv(path: str | None) -> pd.DataFrame | None:
    if path is None:
        return None
    return pd.read_csv(path)


def run_experiment(
    data_config: str = "configs/data.yaml",
    model_config: str = "configs/model_snn.yaml",
    train_config: str = "configs/train.yaml",
    ensemble_config: str = "configs/ensemble.yaml",
    output_dir: str | None = "experiments/runs/latest_v2",
) -> dict:
    setup_logging()
    logger = get_logger("main_v2")

    cfg = Config.from_yaml(
        data_yaml=data_config,
        model_yaml=model_config,
        train_yaml=train_config,
        ensemble_yaml=ensemble_config,
    )
    logger.info(
        f"Loaded config: task={cfg.task_type}, n_models={cfg.n_models}, "
        f"bagging={cfg.bagging}, target={cfg.target_column}"
    )

    train_csv = cfg.extra.get("train_csv")
    valid_csv = cfg.extra.get("valid_csv")
    test_csv = cfg.extra.get("test_csv")
    if not train_csv:
        raise ValueError("data config must set paths.train_csv")

    df_train = _load_csv(train_csv)
    df_valid = _load_csv(valid_csv)
    df_test = _load_csv(test_csv)

    target_col = cfg.target_column
    assert target_col is not None, "data config must set paths.target"

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    X_valid, y_valid = None, None
    if df_valid is not None:
        X_valid = df_valid.drop(columns=[target_col])
        y_valid = df_valid[target_col]

    EstimatorCls = HMTLClassifier if cfg.task_type == "classification" else HMTLRegressor
    estimator = EstimatorCls(
        preset="medium",  # preset is overridden by the YAML-derived Config below
        output_dir=output_dir,
        config=cfg,
    )
    estimator.fit(X_train, y_train, X_val=X_valid, y_val=y_valid, target_column=target_col)

    # Report on validation + test if present.
    report: dict = {"avg_val_score": estimator._metrics.get("avg_score")}
    if df_test is not None:
        X_test = df_test.drop(columns=[target_col])
        y_test = df_test[target_col].values
        preds = estimator.predict(X_test)
        if cfg.task_type == "regression":
            import numpy as np

            rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
            mae = float(np.mean(np.abs(preds - y_test)))
            report["test_rmse"] = rmse
            report["test_mae"] = mae
            logger.info(f"Test RMSE: {rmse:.6f}  MAE: {mae:.6f}")
        else:
            from sklearn.metrics import accuracy_score

            acc = float(accuracy_score(y_test, preds))
            report["test_accuracy"] = acc
            logger.info(f"Test accuracy: {acc:.4f}")

    logger.info(f"Results summary: {report}")
    if output_dir:
        logger.info(f"Model saved to {output_dir}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train an HMTL model from the legacy 4-YAML config layout (via new API)."
    )
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--model", default="configs/model_snn.yaml")
    parser.add_argument("--train", default="configs/train.yaml")
    parser.add_argument("--ensemble", default="configs/ensemble.yaml")
    parser.add_argument("--output", default="experiments/runs/latest_v2")
    args = parser.parse_args(argv)

    run_experiment(
        data_config=args.data,
        model_config=args.model,
        train_config=args.train,
        ensemble_config=args.ensemble,
        output_dir=args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
