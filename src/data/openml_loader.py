"""Utilities for loading datasets from OpenML."""

from __future__ import annotations

import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
from pathlib import Path
from http.client import IncompleteRead

try:
    import openml
except ImportError:
    openml = None

from src.utils.logger import get_logger, log_timing


logger = get_logger("data.openml_loader")


def _require_openml():
    if openml is None:
        raise ImportError("openml is not installed. Install with: pip install openml")
    return openml


def _is_transient_openml_download_error(exc: Exception) -> bool:
    """Return True if an OpenML download error is likely transient/network-related."""
    if isinstance(exc, IncompleteRead):
        return True

    cursor: BaseException | None = exc
    inspected: set[int] = set()
    while cursor is not None and id(cursor) not in inspected:
        inspected.add(id(cursor))
        if isinstance(cursor, IncompleteRead):
            return True
        text = str(cursor)
        if any(
            marker in text
            for marker in (
                "IncompleteRead",
                "Connection broken",
                "Read timed out",
                "ChunkedEncodingError",
                "RemoteDisconnected",
                "ConnectionResetError",
                "ProtocolError",
                "Temporary failure in name resolution",
            )
        ):
            return True
        cursor = cursor.__cause__ or cursor.__context__
    return False


def _load_openml_dataset_with_retry(dataset_id: int):
    _require_openml()

    retry_count = int(os.getenv("OPENML_DOWNLOAD_RETRIES", "3"))
    base_backoff = float(os.getenv("OPENML_DOWNLOAD_RETRY_BACKOFF_SEC", "2.0"))
    attempts = max(1, retry_count + 1)

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with log_timing(f"Loading dataset {dataset_id}", logger):
                return openml.datasets.get_dataset(dataset_id, download_data=True)
        except Exception as exc:
            last_error = exc
            should_retry = _is_transient_openml_download_error(exc) and attempt < attempts
            if not should_retry:
                raise
            sleep_seconds = base_backoff * (2 ** (attempt - 1))
            logger.warning(
                "Transient OpenML download failure for dataset %d (attempt %d/%d): %s. "
                "Retrying in %.1fs.",
                dataset_id,
                attempt,
                attempts,
                exc,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to load OpenML dataset {dataset_id} for unknown reasons")


def get_regression_datasets(study_id: int = 269) -> list[dict]:
    """Get list of regression datasets from OpenML study.
    
    Args:
        study_id: OpenML study ID (default: 269 for regression suite)
        
    Returns:
        List of dictionaries with dataset metadata (id, name, etc.)
    """
    logger.info(f"Fetching regression datasets from OpenML study {study_id}...")
    _require_openml()
    
    try:
        suite = openml.study.get_suite(study_id)
        logger.info(f"Found {len(suite.tasks)} tasks in study {study_id}")
        
        # #region agent log
        import json
        log_path = Path(__file__).parent.parent.parent / ".cursor" / "debug.log"
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A",
                    "location": "openml_loader.py:33",
                    "message": "Checking suite.tasks type",
                    "data": {
                        "suite_tasks_type": str(type(suite.tasks)),
                        "suite_tasks_len": len(suite.tasks),
                        "suite_tasks_first_few": [str(t) for t in list(suite.tasks)[:3]] if hasattr(suite.tasks, "__iter__") else None,
                        "suite_type": str(type(suite)),
                        "suite_attrs": [attr for attr in dir(suite) if not attr.startswith("_")][:10]
                    },
                    "timestamp": int(__import__("time").time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion agent log
        
        datasets = []
        for idx, task in enumerate(suite.tasks):
            # #region agent log
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "B",
                        "location": "openml_loader.py:50",
                        "message": "Checking task type in loop",
                        "data": {
                            "idx": idx,
                            "task_type": str(type(task)),
                            "task_value": str(task),
                            "task_is_int": isinstance(task, int),
                            "has_task_id": hasattr(task, "task_id") if not isinstance(task, int) else False,
                            "has_dataset_id": hasattr(task, "dataset_id") if not isinstance(task, int) else False,
                        },
                        "timestamp": int(__import__("time").time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion agent log
            
            try:
                # #region agent log
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "C",
                            "location": "openml_loader.py:70",
                            "message": "Before accessing task attributes",
                            "data": {
                                "idx": idx,
                                "task": str(task),
                                "task_type": str(type(task)),
                            },
                            "timestamp": int(__import__("time").time() * 1000)
                        }) + "\n")
                except Exception:
                    pass
                # #endregion agent log
                
                if isinstance(task, int):
                    # Task is an ID, need to fetch the task object
                    task_obj = openml.tasks.get_task(task)
                else:
                    # Task is already an object
                    task_obj = task
                
                # #region agent log
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        task_attrs = [attr for attr in dir(task_obj) if not attr.startswith("_")]
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run2",
                            "hypothesisId": "E",
                            "location": "openml_loader.py:85",
                            "message": "Checking task object attributes",
                            "data": {
                                "task_id": getattr(task_obj, "task_id", None),
                                "dataset_id": getattr(task_obj, "dataset_id", None),
                                "has_name": hasattr(task_obj, "name"),
                                "has_dataset_name": hasattr(task_obj, "dataset_name"),
                                "has_get_dataset": hasattr(task_obj, "get_dataset"),
                                "task_type": str(type(task_obj)),
                                "all_attrs": task_attrs[:20],
                            },
                            "timestamp": int(__import__("time").time() * 1000)
                        }) + "\n")
                except Exception:
                    pass
                # #endregion agent log
                
                # Get dataset name - try different approaches
                dataset_name = None
                if hasattr(task_obj, "name"):
                    dataset_name = task_obj.name
                elif hasattr(task_obj, "dataset_name"):
                    dataset_name = task_obj.dataset_name
                else:
                    # Try to get name from dataset
                    try:
                        dataset = openml.datasets.get_dataset(task_obj.dataset_id)
                        dataset_name = dataset.name
                    except Exception as e:
                        # #region agent log
                        try:
                            with open(log_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps({
                                    "sessionId": "debug-session",
                                    "runId": "run2",
                                    "hypothesisId": "F",
                                    "location": "openml_loader.py:110",
                                    "message": "Failed to get dataset name",
                                    "data": {
                                        "task_id": getattr(task_obj, "task_id", None),
                                        "dataset_id": getattr(task_obj, "dataset_id", None),
                                        "error": str(e),
                                    },
                                    "timestamp": int(__import__("time").time() * 1000)
                                }) + "\n")
                        except Exception:
                            pass
                        # #endregion agent log
                        dataset_name = f"dataset_{task_obj.dataset_id}"
                
                dataset_info = {
                    "task_id": task_obj.task_id,
                    "dataset_id": task_obj.dataset_id,
                    "name": dataset_name,
                }
                datasets.append(dataset_info)
            except Exception as e:
                # #region agent log
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "D",
                            "location": "openml_loader.py:95",
                            "message": "Exception in task processing",
                            "data": {
                                "idx": idx,
                                "task": str(task),
                                "error": str(e),
                                "error_type": str(type(e).__name__),
                            },
                            "timestamp": int(__import__("time").time() * 1000)
                        }) + "\n")
                except Exception:
                    pass
                # #endregion agent log
                
                task_id_str = str(task) if isinstance(task, int) else (getattr(task, "task_id", "unknown") if hasattr(task, "task_id") else "unknown")
                logger.warning(f"Failed to get info for task {task_id_str}: {e}")
                continue
        
        logger.info(f"Successfully retrieved {len(datasets)} datasets")
        return datasets
        
    except Exception as e:
        logger.error(f"Failed to fetch datasets from study {study_id}: {e}")
        raise


def load_dataset(
    dataset_id: int,
    target_column: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """Load a dataset from OpenML.
    
    Args:
        dataset_id: OpenML dataset ID
        target_column: Name of target column (if None, uses default)
        cache_dir: Directory to cache datasets (optional)
        
    Returns:
        Tuple of (dataframe, target_column_name)
    """
    logger.info(f"Loading dataset {dataset_id} from OpenML...")
    _require_openml()
    
    try:
        dataset = _load_openml_dataset_with_retry(dataset_id)
        logger.info(f"Dataset loaded: {dataset.name}")
        
        # Get data - check API compatibility
        # #region agent log
        try:
            import inspect
            with open(log_path, "a", encoding="utf-8") as f:
                get_data_sig = inspect.signature(dataset.get_data)
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run3",
                    "hypothesisId": "G",
                    "location": "openml_loader.py:80",
                    "message": "Checking get_data signature",
                    "data": {
                        "dataset_id": dataset_id,
                        "get_data_params": list(get_data_sig.parameters.keys()),
                        "get_data_defaults": {k: str(v.default) if v.default != inspect.Parameter.empty else None 
                                             for k, v in get_data_sig.parameters.items()},
                    },
                    "timestamp": int(__import__("time").time() * 1000)
                }) + "\n")
        except Exception as e:
            pass
        # #endregion agent log
        
        # Try different API versions
        try:
            # Try new API (without return_categorical_indicator)
            result = dataset.get_data(
                target=target_column or dataset.default_target_attribute,
                return_attribute_names=True,
            )
            # #region agent log
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run3",
                        "hypothesisId": "H",
                        "location": "openml_loader.py:265",
                        "message": "get_data result structure",
                        "data": {
                            "dataset_id": dataset_id,
                            "result_type": str(type(result)),
                            "result_len": len(result) if isinstance(result, (tuple, list)) else None,
                            "result_types": [str(type(r)) for r in result] if isinstance(result, (tuple, list)) else None,
                        },
                        "timestamp": int(__import__("time").time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion agent log
            
            if isinstance(result, tuple):
                if len(result) >= 2:
                    X, y = result[0], result[1]
                    # Try to get categorical indicator and attribute names
                    if len(result) >= 3:
                        attribute_names = result[2] if result[2] is not None else dataset.features
                    else:
                        attribute_names = dataset.features
                    categorical_indicator = None  # Not available in new API
                else:
                    raise ValueError(f"Unexpected get_data() return format: {len(result)} elements")
            else:
                raise ValueError(f"Unexpected get_data() return type: {type(result)}")
        except Exception as e:
            # Try old API with return_categorical_indicator (catch all exceptions, not just TypeError)
            # #region agent log
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run3",
                        "hypothesisId": "I",
                        "location": "openml_loader.py:299",
                        "message": "Trying old API after new API failed",
                        "data": {
                            "dataset_id": dataset_id,
                            "error_type": str(type(e).__name__),
                            "error": str(e),
                        },
                        "timestamp": int(__import__("time").time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion agent log
            
            try:
                X, y, categorical_indicator, attribute_names = dataset.get_data(
                    target=target_column or dataset.default_target_attribute,
                    return_categorical_indicator=True,
                    return_attribute_names=True,
                )
            except Exception as e2:
                # If old API also fails, try minimal API
                # #region agent log
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run3",
                            "hypothesisId": "J",
                            "location": "openml_loader.py:325",
                            "message": "Trying minimal API",
                            "data": {
                                "dataset_id": dataset_id,
                                "error_type": str(type(e2).__name__),
                                "error": str(e2),
                            },
                            "timestamp": int(__import__("time").time() * 1000)
                        }) + "\n")
                except Exception:
                    pass
                # #endregion agent log
                
                # Try minimal API - just get data without extra parameters
                result = dataset.get_data(target=target_column or dataset.default_target_attribute)
                if isinstance(result, tuple) and len(result) >= 2:
                    X, y = result[0], result[1]
                    attribute_names = dataset.features if hasattr(dataset, "features") else None
                    categorical_indicator = None
                else:
                    raise ValueError(f"Unexpected get_data() return format: {type(result)}")
        
        # Convert to DataFrame
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=attribute_names)
        
        # Normalize feature dtypes to numeric floats for downstream preprocessing.
        # This handles object/category/string columns and preserves missing values.
        normalized_features: dict[str, pd.Series] = {}
        for col in df.columns:
            col_series = df[col]

            if pd.api.types.is_numeric_dtype(col_series):
                normalized_features[col] = pd.to_numeric(col_series, errors="coerce").astype(np.float64)
                continue

            numeric_series = pd.to_numeric(col_series, errors="coerce")
            n_non_null = int(col_series.notna().sum())
            n_numeric = int(numeric_series.notna().sum())
            if n_numeric == n_non_null:
                normalized_features[col] = numeric_series.astype(np.float64)
                logger.debug("Column '%s' parsed as numeric from non-numeric dtype", col)
                continue

            # Fallback: stable categorical coding with NaN for missing entries.
            categorical_codes = pd.Categorical(col_series).codes.astype(np.float64)
            categorical_codes[categorical_codes < 0] = np.nan
            normalized_features[col] = pd.Series(categorical_codes, index=col_series.index)
            n_categories = int(pd.Series(categorical_codes).dropna().nunique())
            logger.debug(
                "Encoded categorical column '%s' using category codes (%d categories)",
                col,
                n_categories,
            )

        df = pd.DataFrame(normalized_features, index=df.index)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)
        
        # Add target column
        target_col = target_column or dataset.default_target_attribute
        df[target_col] = y
        
        logger.info(f"Dataset shape: {df.shape}, target: {target_col}")
        logger.info(f"Features: {len(df.columns) - 1}, samples: {len(df)}")
        
        return df, target_col
        
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_id}: {e}")
        raise


def load_dataset_from_task(
    task_id: int,
    target_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """Load dataset by OpenML task id.

    This helper is useful for classification tasks where user-facing catalogs
    often specify OpenML task ids instead of dataset ids.
    """
    logger.info(f"Resolving OpenML task {task_id}")
    _require_openml()
    try:
        task = openml.tasks.get_task(task_id)
    except Exception as e:
        logger.error(f"Failed to fetch OpenML task {task_id}: {e}")
        raise

    dataset_id = getattr(task, "dataset_id", None)
    if dataset_id is None:
        raise ValueError(f"OpenML task {task_id} does not expose dataset_id")

    resolved_target = target_column
    if resolved_target is None:
        resolved_target = getattr(task, "target_name", None)

    logger.info(f"Task {task_id} resolved to dataset {dataset_id}, target={resolved_target}")
    return load_dataset(dataset_id=dataset_id, target_column=resolved_target)


def prepare_dataset_splits(
    df: pd.DataFrame,
    target_column: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train, validation, and test sets.
    
    Args:
        df: Full dataset DataFrame
        target_column: Name of target column
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, valid_df, test_df)
    """
    logger.info(f"Splitting dataset: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}")
    
    # Check ratios sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # First split: train+val vs test
    test_size = test_ratio
    train_val_size = train_ratio + val_ratio
    
    df_train_val, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )
    
    # Second split: train vs val
    val_size_in_train_val = val_ratio / train_val_size
    
    df_train, df_valid = train_test_split(
        df_train_val,
        test_size=val_size_in_train_val,
        random_state=seed,
        shuffle=True,
    )
    
    logger.info(f"Split completed:")
    logger.info(f"  Train: {len(df_train)} samples ({len(df_train)/len(df):.1%})")
    logger.info(f"  Valid: {len(df_valid)} samples ({len(df_valid)/len(df):.1%})")
    logger.info(f"  Test:  {len(df_test)} samples ({len(df_test)/len(df):.1%})")
    
    return df_train, df_valid, df_test


def sample_train_data(
    df_train: pd.DataFrame,
    size_ratio: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample a subset of training data.
    
    Args:
        df_train: Full training DataFrame
        size_ratio: Proportion of data to sample (0.0 to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Sampled DataFrame
    """
    if size_ratio <= 0.0 or size_ratio > 1.0:
        raise ValueError(f"size_ratio must be in (0, 1], got {size_ratio}")
    
    if size_ratio == 1.0:
        return df_train.copy()
    
    n_samples = int(len(df_train) * size_ratio)
    if n_samples == 0:
        n_samples = 1
    
    sampled = df_train.sample(n=n_samples, random_state=seed, replace=False)
    
    logger.debug(f"Sampled {len(sampled)}/{len(df_train)} samples ({size_ratio:.1%})")
    
    return sampled
