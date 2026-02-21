#!/usr/bin/env python3
"""Run multi-size experiments on the local superconductor dataset."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

HIGH_LEVEL_PROGRESS_FLAG = "--high-level-progress-only"
HIGH_LEVEL_PROGRESS_ONLY = HIGH_LEVEL_PROGRESS_FLAG in sys.argv
if HIGH_LEVEL_PROGRESS_ONLY:
    warnings.filterwarnings("ignore")
    mpl_cache_dir = project_root / ".cache" / "matplotlib"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

from src.utils.logger import get_logger, setup_logging
from scripts.run_automlbenchmark_experiment import (
    SUPPORTED_BASELINES,
    _build_preprocess_config,
    _load_yaml,
    _resolve_seed_list,
    _slugify,
    aggregate_size_seed_runs,
    run_size_seed_trial,
)


def run_superconductor_size_experiments(
    *,
    data_cfg_path: Path,
    model_cfg_path: Path,
    train_cfg_path: Path,
    ensemble_cfg_path: Path,
    output_dir: Path,
    sizes: list[float],
    seed: int,
    seeds: list[int] | None,
    n_seeds: int,
    baselines: list[str],
    dataset_name: str,
    high_level_progress_only: bool = False,
) -> list[dict[str, Any]]:
    logger = get_logger("superconductor_size")

    baselines = [baseline.strip().lower() for baseline in baselines]
    baselines = list(dict.fromkeys(baselines))
    for baseline_name in baselines:
        if baseline_name not in SUPPORTED_BASELINES:
            raise ValueError(
                f"Unsupported baseline '{baseline_name}'. Supported: {SUPPORTED_BASELINES}"
            )

    data_cfg = _load_yaml(data_cfg_path)
    model_cfg = _load_yaml(model_cfg_path)
    train_cfg_yaml = _load_yaml(train_cfg_path)
    ensemble_cfg_yaml = _load_yaml(ensemble_cfg_path)
    preprocess_config = _build_preprocess_config(data_cfg)
    seed_list = _resolve_seed_list(base_seed=seed, seeds=seeds, n_seeds=n_seeds)

    paths = data_cfg["paths"]
    target_col = paths["target"]
    df_train_full = pd.read_csv(paths["train_csv"])
    df_valid = pd.read_csv(paths["valid_csv"])
    df_test = pd.read_csv(paths["test_csv"])
    n_features = int(df_train_full.drop(columns=[target_col], errors="ignore").shape[1])

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_slug = _slugify(dataset_name)
    dataset_folder = output_dir / f"dataset_local_{dataset_slug}"
    dataset_folder.mkdir(parents=True, exist_ok=True)

    logger.info("Seeds: %s", seed_list)
    logger.info("Baselines: %s", baselines)
    logger.info("Dataset: %s", dataset_name)

    result: dict[str, Any] = {
        "dataset_id": -1,
        "dataset_name": dataset_name,
        "task_id": None,
        "n_samples_total": int(len(df_train_full) + len(df_valid) + len(df_test)),
        "n_samples_train": int(len(df_train_full)),
        "n_samples_valid": int(len(df_valid)),
        "n_samples_test": int(len(df_test)),
        "n_features": n_features,
        "target_column": target_col,
        "run_meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "seed_list": [int(s) for s in seed_list],
            "n_requested_seeds": int(len(seed_list)),
            "sizes": [float(s) for s in sizes],
            "baselines": baselines,
            "configs": {
                "data": str(data_cfg_path),
                "model": str(model_cfg_path),
                "train": str(train_cfg_path),
                "ensemble": str(ensemble_cfg_path),
            },
            "output_folder": str(dataset_folder),
        },
        "sizes": {},
    }

    total_trials = len(sizes) * len(seed_list)
    with tqdm(
        total=total_trials,
        desc="Superconductor size*seed",
        unit="trial",
        leave=True,
        disable=not high_level_progress_only,
    ) as trial_pbar:
        for size_ratio in sizes:
            size_key = str(int(round(size_ratio * 100)))
            per_seed_runs = []
            for run_seed in seed_list:
                seed_result = run_size_seed_trial(
                    size_ratio=size_ratio,
                    seed=run_seed,
                    df_train_full=df_train_full,
                    df_valid=df_valid,
                    df_test=df_test,
                    target_column=target_col,
                    preprocess_config=preprocess_config,
                    model_cfg=model_cfg,
                    train_cfg_yaml=train_cfg_yaml,
                    ensemble_cfg_yaml=ensemble_cfg_yaml,
                    baselines=baselines,
                    show_inner_progress=not high_level_progress_only,
                )
                per_seed_runs.append(seed_result)
                trial_pbar.update(1)

            size_summary = aggregate_size_seed_runs(
                per_seed_runs=per_seed_runs,
                baselines=baselines,
            )
            n_train_samples = (
                int(per_seed_runs[0]["n_train_samples"])
                if per_seed_runs
                else int(round(len(df_train_full) * size_ratio))
            )
            result["sizes"][size_key] = {
                "size_ratio": float(size_ratio),
                "n_train_samples": int(n_train_samples),
                "per_seed": {str(run["seed"]): run for run in per_seed_runs},
                **size_summary,
            }

    dataset_result_file = dataset_folder / "results.json"
    with open(dataset_result_file, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    aggregated_results = [result]
    aggregated_path = output_dir / "aggregated_results.json"
    with open(aggregated_path, "w", encoding="utf-8") as file:
        json.dump(aggregated_results, file, indent=2)

    logger.info("Saved dataset results to %s", dataset_result_file)
    logger.info("Saved aggregated results to %s", aggregated_path)
    return aggregated_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run size-dependence experiments on local superconductor dataset"
    )
    parser.add_argument("--data", default="configs/data_superconductor.yaml", help="Path to data config")
    parser.add_argument("--model", default="configs/model_snn.yaml", help="Path to model config")
    parser.add_argument("--train", default="configs/train.yaml", help="Path to train config")
    parser.add_argument("--ensemble", default="configs/ensemble.yaml", help="Path to ensemble config")
    parser.add_argument(
        "--output",
        default="experiments/superconductor_sizes",
        help="Directory for experiment outputs",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=float,
        default=[0.2, 0.4, 0.6, 0.8, 1.0],
        help="Train size ratios to evaluate",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Explicit seed list")
    parser.add_argument(
        "--n-seeds",
        "--n_seeds",
        dest="n_seeds",
        type=int,
        default=3,
        help="Number of seeds (when --seeds not provided)",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["catboost", "single_mlp"],
        help="Baselines to compare against HMTL. Supported: catboost single_mlp flat_mtl",
    )
    parser.add_argument("--dataset-name", default="superconductor", help="Dataset name label in outputs")
    parser.add_argument(
        "--high-level-progress-only",
        action="store_true",
        help="Show only high-level progress bars and suppress inner training progress",
    )

    args = parser.parse_args()
    console_log_level = logging.ERROR if args.high_level_progress_only else logging.INFO
    setup_logging(log_level=console_log_level)
    if args.high_level_progress_only:
        logging.getLogger().setLevel(logging.ERROR)
        for noisy_logger in ("openml", "urllib3", "matplotlib", "numexpr"):
            logging.getLogger(noisy_logger).setLevel(logging.ERROR)

    run_superconductor_size_experiments(
        data_cfg_path=Path(args.data),
        model_cfg_path=Path(args.model),
        train_cfg_path=Path(args.train),
        ensemble_cfg_path=Path(args.ensemble),
        output_dir=Path(args.output),
        sizes=args.sizes,
        seed=args.seed,
        seeds=args.seeds,
        n_seeds=args.n_seeds,
        baselines=args.baselines,
        dataset_name=args.dataset_name,
        high_level_progress_only=args.high_level_progress_only,
    )


if __name__ == "__main__":
    main()
