#!/usr/bin/env python3
"""Quick single-dataset validation for HMTL fixes.

Usage:
    python scripts/quick_validation_single_dataset.py --dataset-id 2178 --n-seeds 1 --sizes 1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.run_automlbenchmark_experiment import (
    DatasetMeta,
    run_single_dataset_experiment,
)
from src.data.preprocess import PreprocessConfig
from src.utils.logger import setup_logging

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", type=int, default=2178, help="OpenML dataset ID")
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--sizes", type=str, default="0.5,1.0",
                        help="Comma-separated size ratios")
    parser.add_argument("--output", type=str, default="experiments/quick_validation")
    args = parser.parse_args()

    setup_logging()

    sizes = [float(s) for s in args.sizes.split(",")]
    seeds = list(range(42, 42 + args.n_seeds))

    configs_dir = project_root / "configs"
    model_cfg = yaml.safe_load((configs_dir / "model_snn.yaml").read_text())
    train_cfg = yaml.safe_load((configs_dir / "train.yaml").read_text())
    ensemble_cfg = yaml.safe_load((configs_dir / "ensemble.yaml").read_text())

    # Use fewer ensemble members for speed
    ensemble_cfg.setdefault("ensemble", {})["n_models"] = 5

    preprocess_config = PreprocessConfig(
        impute_const=-1.0,
        use_dynamic_binning=False,
        standardize=True,
        pca_enabled=False,
        target_standardize=True,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_meta = DatasetMeta(
        dataset_id=args.dataset_id,
        task_id=0,
        dataset_name=None,
    )

    result = run_single_dataset_experiment(
        dataset_meta=dataset_meta,
        sizes=sizes,
        seeds=seeds,
        model_cfg=model_cfg,
        train_cfg_yaml=train_cfg,
        ensemble_cfg_yaml=ensemble_cfg,
        preprocess_config=preprocess_config,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        baselines=["catboost"],
        split_seed=42,
        output_dir=output_dir,
        study_id=0,
        config_paths={
            "model": str(configs_dir / "model_snn.yaml"),
            "train": str(configs_dir / "train.yaml"),
            "ensemble": str(configs_dir / "ensemble.yaml"),
            "data": str(configs_dir / "data.yaml"),
        },
        show_trial_progress=True,
    )

    # Print comparison
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    for sz_key, sz_data in result.get("sizes", {}).items():
        for seed, sv in sz_data.get("per_seed", {}).items():
            hmtl_rmse = sv.get("hmtl", {}).get("rmse", "N/A")
            cb_rmse = sv.get("baselines", {}).get("catboost", {}).get("rmse", "N/A")
            print(f"size={sz_key}% seed={seed}: HMTL={hmtl_rmse:.4f}  CatBoost={cb_rmse:.4f}")

    print(f"\nBefore fix: HMTL=1.847, CatBoost=0.923")
    print("If HMTL RMSE dropped significantly, the fixes are working.")


if __name__ == "__main__":
    main()
