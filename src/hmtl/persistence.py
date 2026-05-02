"""Inference-only model persistence for the high-level HMTL API.

Directory layout ::

    runs/<name>/
    ├── manifest.json       # version, task_type, feature schema, model hparams
    ├── preprocessor.pkl    # joblib-pickled TabularPreprocessor
    ├── models/model_*.pt   # per-ensemble-member state_dict + init hparams
    ├── conformal.json      # {0.8: q, 0.9: q, 0.95: q}
    └── config.yaml         # resolved Config
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import torch

from src.hmtl.config import Config

FORMAT_VERSION = 1


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _pkg_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    try:
        import torch as _torch
        versions["torch"] = _torch.__version__
    except Exception:
        pass
    try:
        import numpy as _np
        versions["numpy"] = _np.__version__
    except Exception:
        pass
    try:
        import sklearn as _sk
        versions["sklearn"] = _sk.__version__
    except Exception:
        pass
    try:
        from src.hmtl import __version__ as _hmtl_version
        versions["hmtl"] = _hmtl_version
    except Exception:
        pass
    return versions


def save_model(
    output_dir: str | Path,
    config: Config,
    preprocessor,
    models: List,
    model_hparams: Dict[str, Any],
    conformal_quantiles: Dict[float, float],
    feature_columns: List[str],
    feature_dtypes: Dict[str, str],
    target_column: Optional[str],
    metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    """Serialize a trained estimator to ``output_dir``.

    ``model_hparams`` are the kwargs needed to reconstruct each ensemble member
    via :class:`src.models.hmtl.HMTLModel`.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)

    # Preprocessor
    joblib.dump(preprocessor, out / "preprocessor.pkl")

    # Model state dicts
    model_paths: list[str] = []
    for i, model in enumerate(models):
        path = out / "models" / f"model_{i}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hparams": model_hparams,
            },
            path,
        )
        model_paths.append(str(path.relative_to(out)))

    # Conformal quantiles — JSON keys must be strings
    with open(out / "conformal.json", "w", encoding="utf-8") as f:
        json.dump({str(k): float(v) for k, v in conformal_quantiles.items()}, f, indent=2)

    # Resolved config
    config.to_yaml(out / "config.yaml")

    # Manifest
    manifest = {
        "format_version": FORMAT_VERSION,
        "created_at": _timestamp(),
        "task_type": config.task_type,
        "preset": config.preset,
        "target_column": target_column,
        "feature_columns": list(feature_columns),
        "feature_dtypes": {k: str(v) for k, v in feature_dtypes.items()},
        "n_models": len(models),
        "model_paths": model_paths,
        "coverage_levels": [float(c) for c in config.coverage_levels],
        "model_hparams": _to_jsonable(model_hparams),
        "package_versions": _pkg_versions(),
        "metrics": _to_jsonable(metrics or {}),
    }
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return out


def load_model(run_dir: str | Path, device: str = "cpu") -> Dict[str, Any]:
    """Inverse of :func:`save_model`. Returns a dict the estimator can consume."""
    from src.models.hmtl import HMTLModel

    root = Path(run_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {root}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("format_version", 0) > FORMAT_VERSION:
        raise ValueError(
            f"Model format_version={manifest['format_version']} is newer than "
            f"library FORMAT_VERSION={FORMAT_VERSION}. Upgrade the hmtl package."
        )

    # Preprocessor
    preprocessor = joblib.load(root / "preprocessor.pkl")

    # Models
    hparams = manifest["model_hparams"]
    models: list[HMTLModel] = []
    model_paths = manifest["model_paths"]
    for rel_path in model_paths:
        ckpt = torch.load(root / rel_path, map_location=device, weights_only=False)
        # Classification models need a task_head; we reconstruct minimally here
        # by letting HMTLModel default to RegressionHead unless the manifest
        # captured a classification task. For classification, we pass
        # ``num_classes`` via a lightweight head reconstruction.
        model = _reconstruct_model(
            hparams, task_type=manifest["task_type"], state_dict=ckpt["state_dict"]
        )
        model.to(device)
        model.eval()
        models.append(model)

    # Conformal quantiles
    conformal_path = root / "conformal.json"
    conformal_quantiles: dict[float, float] = {}
    if conformal_path.exists():
        with open(conformal_path, "r", encoding="utf-8") as f:
            conformal_quantiles = {float(k): float(v) for k, v in json.load(f).items()}

    # Config
    from yaml import safe_load

    with open(root / "config.yaml", "r", encoding="utf-8") as f:
        cfg_dict = safe_load(f) or {}
    cfg = Config(**{k: v for k, v in cfg_dict.items() if k in Config.__dataclass_fields__})

    return {
        "config": cfg,
        "preprocessor": preprocessor,
        "models": models,
        "conformal_quantiles": conformal_quantiles,
        "manifest": manifest,
    }


def _reconstruct_model(hparams: Dict[str, Any], task_type: str, state_dict):
    """Build an HMTLModel skeleton matching stored hparams, then load weights."""
    from src.models.hmtl import HMTLModel

    # Classification heads are rebuilt via the task factory.
    task_head = None
    if task_type == "classification":
        from src.tasks.classification import ClassificationTask, ClassificationTaskConfig

        num_classes = int(hparams.get("num_classes", 2))
        task_cfg = ClassificationTaskConfig(task_type="classification", num_classes=num_classes)
        task_head = ClassificationTask.create_task_head(task_cfg, in_dim=int(hparams["hidden_width"]))

    init_kwargs = {
        "input_dim": int(hparams["input_dim"]),
        "hidden_width": int(hparams["hidden_width"]),
        "depth_low": int(hparams["depth_low"]),
        "depth_high": int(hparams["depth_high"]),
        "alpha_dropout": float(hparams["alpha_dropout"]),
        "n_bins": int(hparams["n_bins"]),
        "aux_weight": float(hparams["aux_weight"]),
        "enable_aux": bool(hparams["enable_aux"]),
        "aux_task": str(hparams["aux_task"]),
        "proj_dim": int(hparams["proj_dim"]),
        "scale_coeff": float(hparams.get("scale_coeff", 1.0)),
        "task_head": task_head,
        "use_residual": bool(hparams.get("use_residual", True)),
    }
    model = HMTLModel(**init_kwargs)
    model.load_state_dict(state_dict, strict=False)
    return model


def validate_input_schema(
    df, manifest: Dict[str, Any], allow_missing: bool = False
) -> None:
    """Raise if ``df`` columns don't match the manifest feature schema."""
    expected = manifest["feature_columns"]
    actual = list(df.columns)
    if set(expected) - set(actual):
        missing = set(expected) - set(actual)
        if not allow_missing:
            raise ValueError(
                f"Input data is missing feature columns: {sorted(missing)}. "
                f"Expected columns: {expected}"
            )


def _to_jsonable(obj: Any) -> Any:
    """Make a dict-tree safe for json.dump (tuples → lists, paths → str, numpy → python)."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    try:
        # numpy scalars
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return str(obj)
