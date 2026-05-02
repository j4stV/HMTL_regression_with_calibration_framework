"""Command-line interface for the HMTL AutoML API.

Subcommands
-----------

``hmtl train``   — fit an estimator on a CSV and write it to disk.
``hmtl predict`` — load a saved estimator and score a CSV.
``hmtl info``    — print the saved manifest.
``hmtl report``  — regenerate the report plots and metrics summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from src.hmtl.auto import summarize_data
from src.hmtl.estimator import HMTLClassifier, HMTLRegressor, load


def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _choose_estimator(task_type: str):
    return HMTLClassifier if task_type == "classification" else HMTLRegressor


def cmd_train(args: argparse.Namespace) -> int:
    df = _load_csv(args.data)
    if args.target not in df.columns:
        print(f"error: target column '{args.target}' not in data", file=sys.stderr)
        return 2

    y = df[args.target]
    X = df.drop(columns=[args.target])

    # Auto-detect task type unless user specified
    task_type = args.task_type
    if task_type == "auto":
        summary = summarize_data(X, y)
        task_type = "classification" if summary.is_classification() else "regression"

    cls = _choose_estimator(task_type)
    overrides = {}
    if args.n_models is not None:
        overrides["n_models"] = args.n_models
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.epochs is not None:
        overrides["epochs"] = args.epochs

    estimator = cls(preset=args.preset, output_dir=args.output, **overrides)
    estimator.fit(X, y, target_column=args.target)
    print(f"Saved model to {args.output}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    est = load(args.run_dir, device=args.device)
    df = _load_csv(args.data)
    # Drop the target column if it happens to be present.
    if est._target_column and est._target_column in df.columns:
        df = df.drop(columns=[est._target_column])

    if args.with_uncertainty:
        result = est.predict(df, return_uncertainty=True)
        pred, sigma = result
        out = pd.DataFrame({"prediction": pred, "uncertainty": sigma})
    else:
        pred = est.predict(df)
        out = pd.DataFrame({"prediction": pred})

    if args.coverage is not None and est.config_.task_type == "regression":
        lower, upper = est.predict_interval(df, coverage=args.coverage)
        out[f"lower_{int(args.coverage * 100)}"] = lower
        out[f"upper_{int(args.coverage * 100)}"] = upper

    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} predictions to {args.out}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    manifest_path = Path(args.run_dir) / "manifest.json"
    if not manifest_path.exists():
        print(f"error: no manifest at {manifest_path}", file=sys.stderr)
        return 2
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    print(json.dumps(manifest, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    # Minimal MVP: re-load and print key stats + metrics.
    est = load(args.run_dir)
    print(f"Task: {est.config_.task_type}")
    print(f"Preset: {est.config_.preset}")
    print(f"Ensemble size: {len(est.models_)}")
    print(f"Features: {len(est._feature_columns)}")
    if est.conformal_q_:
        print("Conformal quantiles (standardized space):")
        for cov, q in sorted(est.conformal_q_.items()):
            print(f"  {int(cov * 100)}%: q={q:.4f}")
    if est._metrics:
        print("Training metrics:")
        for k, v in est._metrics.items():
            print(f"  {k}: {v}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hmtl",
        description="HMTL AutoML CLI — train, predict, and inspect tabular models with calibrated uncertainty.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    pt = sub.add_parser("train", help="Fit a model and save it to disk")
    pt.add_argument("data", help="Path to training CSV")
    pt.add_argument("--target", required=True, help="Target column name")
    pt.add_argument("--output", required=True, help="Output directory for the trained model")
    pt.add_argument("--preset", default="medium", choices=["fast", "medium", "best_quality"])
    pt.add_argument("--task-type", default="auto", choices=["auto", "regression", "classification"])
    pt.add_argument("--n-models", type=int, default=None)
    pt.add_argument("--epochs", type=int, default=None)
    pt.add_argument("--seed", type=int, default=None)
    pt.set_defaults(func=cmd_train)

    # predict
    pp = sub.add_parser("predict", help="Score a CSV using a saved model")
    pp.add_argument("run_dir", help="Directory produced by `hmtl train`")
    pp.add_argument("data", help="Path to CSV to score")
    pp.add_argument("--out", required=True, help="Path to output CSV with predictions")
    pp.add_argument("--with-uncertainty", action="store_true")
    pp.add_argument("--coverage", type=float, default=None, help="Emit prediction intervals at this coverage (regression only)")
    pp.add_argument("--device", default="cpu")
    pp.set_defaults(func=cmd_predict)

    # info
    pi = sub.add_parser("info", help="Print a saved model's manifest")
    pi.add_argument("run_dir")
    pi.set_defaults(func=cmd_info)

    # report
    pr = sub.add_parser("report", help="Print a quick summary of a saved model")
    pr.add_argument("run_dir")
    pr.set_defaults(func=cmd_report)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
