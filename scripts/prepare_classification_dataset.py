#!/usr/bin/env python3
"""Prepare classification datasets from OpenML for experimentation.

This script downloads tabular classification datasets from OpenML,
splits them into train/valid/test sets, and generates corresponding
configuration files for the HMTL framework.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.data.openml_loader import load_dataset_from_task
from src.utils.logger import setup_logging, get_logger


# OpenML dataset catalog for classification
DATASET_CATALOG = {
    "iris": {
        "task_id": 59,
        "name": "Iris",
        "n_classes": 3,
        "n_samples": 150,
        "n_features": 4,
        "description": "Classic iris flowers classification (3 classes, 150 samples)",
    },
    "phoneme": {
        "task_id": 9952,
        "name": "Phoneme",
        "n_classes": 2,
        "n_samples": 5404,
        "n_features": 5,
        "description": "Phoneme classification (2 classes, 5,404 samples)",
    },
    "letter": {
        "task_id": 6,
        "name": "Letter Recognition",
        "n_classes": 26,
        "n_samples": 20000,
        "n_features": 16,
        "description": "Letter recognition (26 classes, 20,000 samples)",
    },
}


def prepare_dataset(
    dataset_name: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    force: bool = False,
) -> dict:
    """Prepare a single classification dataset.

    Args:
        dataset_name: Dataset name from DATASET_CATALOG
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
        force: Force re-download even if files exist

    Returns:
        Dictionary with dataset info
    """
    logger = get_logger("prepare_classification")

    if dataset_name not in DATASET_CATALOG:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CATALOG.keys())}")

    catalog_entry = DATASET_CATALOG[dataset_name]
    logger.info(f"Preparing {catalog_entry['name']} dataset")
    logger.info(f"  Description: {catalog_entry['description']}")

    # Check if files already exist
    data_dir = project_root / "data" / dataset_name
    train_path = data_dir / "train.csv"
    valid_path = data_dir / "valid.csv"
    test_path = data_dir / "test.csv"
    config_path = project_root / "configs" / f"data_{dataset_name}.yaml"

    if not force and all(p.exists() for p in [train_path, valid_path, test_path, config_path]):
        logger.info(f"Dataset files already exist. Use --force to re-generate.")
        return {
            "train_path": str(train_path),
            "valid_path": str(valid_path),
            "test_path": str(test_path),
            "config_path": str(config_path),
        }

    # Download from OpenML
    logger.info(f"Downloading from OpenML (task_id={catalog_entry['task_id']})...")
    df, target_col = load_dataset_from_task(catalog_entry["task_id"])

    logger.info(f"Loaded dataset: {df.shape} (rows, cols)")
    logger.info(f"Target column: {target_col}")
    logger.info(f"Class distribution:\n{df[target_col].value_counts()}")

    # Ensure class labels are 0-indexed consecutive integers
    unique_labels = sorted(df[target_col].unique())
    expected_labels = list(range(len(unique_labels)))
    has_integer_labels = all(isinstance(label, (int, np.integer)) for label in unique_labels)
    is_zero_indexed = has_integer_labels and list(unique_labels) == expected_labels
    if not is_zero_indexed:
        logger.info("Converting class labels to 0-indexed consecutive integers")
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        df[target_col] = df[target_col].map(label_mapping)
        logger.info(f"Label mapping: {label_mapping}")

    # Verify class count
    actual_n_classes = df[target_col].nunique()
    if actual_n_classes != catalog_entry["n_classes"]:
        logger.warning(
            f"Expected {catalog_entry['n_classes']} classes, found {actual_n_classes}"
        )

    # Split into train/val/test
    logger.info(f"Splitting: train={train_ratio}, val={val_ratio}, test={test_ratio}")

    # First split: train+val vs test
    train_val_ratio = train_ratio + val_ratio
    df_train_val, df_test = train_test_split(
        df, test_size=test_ratio, random_state=seed, stratify=df[target_col]
    )

    # Second split: train vs val
    val_ratio_adjusted = val_ratio / train_val_ratio
    df_train, df_val = train_test_split(
        df_train_val, test_size=val_ratio_adjusted, random_state=seed, stratify=df_train_val[target_col]
    )

    logger.info(f"Split sizes: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    # Save to CSV
    data_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(valid_path, index=False)
    df_test.to_csv(test_path, index=False)

    logger.info(f"Saved datasets to {data_dir}")

    # Generate configuration file
    config = {
        "task": {
            "type": "classification",
            "num_classes": actual_n_classes,
            "class_weights": None,
            "temperature_scaling": True,
            "use_focal_loss": False,
            "focal_alpha": 0.25,
            "focal_gamma": 2.0,
            "label_smoothing": 0.0,
        },
        "paths": {
            "train_csv": f"data/{dataset_name}/train.csv",
            "valid_csv": f"data/{dataset_name}/valid.csv",
            "cal_csv": None,
            "test_csv": f"data/{dataset_name}/test.csv",
            "target": target_col,
        },
        "preprocess": {
            "impute_const": -1.0,
            "use_dynamic_binning": True,
            "quantile_binning": {
                "enabled": False,
                "bins": 5,
            },
            "standardize": True,
            "pca": {
                "enabled": True,
                "n_components": None,
            },
            "target_standardize": False,  # MUST be false for classification
        },
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Generated configuration: {config_path}")

    return {
        "dataset_name": dataset_name,
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "test_path": str(test_path),
        "config_path": str(config_path),
        "n_classes": actual_n_classes,
        "n_train": len(df_train),
        "n_val": len(df_val),
        "n_test": len(df_test),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare classification datasets from OpenML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available datasets:
{chr(10).join(f"  - {name}: {info['description']}" for name, info in DATASET_CATALOG.items())}
        """,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_CATALOG.keys()) + ["all"],
        default=["iris"],
        help="Datasets to prepare (default: iris)",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--force", action="store_true", help="Force re-download")

    args = parser.parse_args()

    setup_logging()
    logger = get_logger("main")

    # Validate ratios
    if not np.isclose(args.train_ratio + args.val_ratio + args.test_ratio, 1.0):
        raise ValueError("Train/val/test ratios must sum to 1.0")

    # Determine which datasets to prepare
    if "all" in args.datasets:
        datasets_to_prepare = list(DATASET_CATALOG.keys())
    else:
        datasets_to_prepare = args.datasets

    logger.info(f"Preparing {len(datasets_to_prepare)} dataset(s): {datasets_to_prepare}")

    # Prepare each dataset
    results = []
    for dataset_name in datasets_to_prepare:
        logger.info("=" * 80)
        try:
            result = prepare_dataset(
                dataset_name=dataset_name,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
                force=args.force,
            )
            results.append(result)
            logger.info(f"✓ Successfully prepared {dataset_name}")
        except Exception as e:
            logger.error(f"✗ Failed to prepare {dataset_name}: {e}")
            continue

    logger.info("=" * 80)
    logger.info("Summary:")
    for result in results:
        if result:
            logger.info(
                f"  {result['dataset_name']}: "
                f"{result['n_classes']} classes, "
                f"{result['n_train']} train, "
                f"{result['n_val']} val, "
                f"{result['n_test']} test"
            )

    logger.info(f"Prepared {len(results)}/{len(datasets_to_prepare)} datasets successfully")


if __name__ == "__main__":
    main()
