"""High-level HMTL AutoML API.

Entry points for end users: ``HMTLRegressor`` and ``HMTLClassifier`` expose a
sklearn-style ``fit`` / ``predict`` interface with tiered presets (``fast`` |
``medium`` | ``best_quality``), inference-only model persistence, and
size-adaptive defaults.

The underlying ``src.models``, ``src.train``, ``src.eval``, ``src.data``
modules remain the engine — this package wraps them, it does not replace them.
"""

from src.hmtl.auto import DataSummary, summarize_data
from src.hmtl.config import Config, ResolvedConfig
from src.hmtl.estimator import HMTLClassifier, HMTLRegressor, load
from src.hmtl.presets import PRESETS, resolve_preset

__all__ = [
    "Config",
    "DataSummary",
    "HMTLClassifier",
    "HMTLRegressor",
    "PRESETS",
    "ResolvedConfig",
    "load",
    "resolve_preset",
    "summarize_data",
]

__version__ = "0.1.0"
