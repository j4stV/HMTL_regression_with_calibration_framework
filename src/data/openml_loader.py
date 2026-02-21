"""Utilities for loading datasets from OpenML."""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
from pathlib import Path

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
        with log_timing(f"Loading dataset {dataset_id}", logger):
            dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
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
        
        # Handle categorical features - convert string categories to numeric codes
        # OpenML datasets may have categorical features as strings, but preprocessor expects numeric
        from sklearn.preprocessing import LabelEncoder
        
        for col in df.columns:
            if df[col].dtype == 'object' or (len(df) > 0 and isinstance(df[col].iloc[0], str)):
                # Try to convert to numeric first
                try:
                    df[col] = pd.to_numeric(df[col], errors='raise')
                except (ValueError, TypeError):
                    # If conversion fails, encode as numeric codes (LabelEncoder)
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    logger.debug(f"Encoded categorical column '{col}' using LabelEncoder ({len(le.classes_)} categories)")
        
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
