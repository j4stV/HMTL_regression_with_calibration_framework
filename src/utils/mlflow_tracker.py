from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:
    import mlflow
    import mlflow.pytorch
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from src.utils.logger import get_logger


class MLflowTracker:
    """MLflow experiment tracker."""
    
    def __init__(
        self,
        experiment_name: str = "hmtl_calibration",
        tracking_uri: str | None = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled and HAS_MLFLOW
        self.logger = get_logger("mlflow")
        
        if not self.enabled:
            if not HAS_MLFLOW:
                self.logger.warning("MLflow not available, tracking disabled")
            return
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        mlflow.set_experiment(experiment_name)
        self.logger.info(f"MLflow tracking enabled: experiment={experiment_name}")
    
    def start_run(self, run_name: str | None = None) -> None:
        """Start a new MLflow run."""
        if not self.enabled:
            return
        
        mlflow.start_run(run_name=run_name)
        self.logger.info(f"Started MLflow run: {run_name}")
    
    def end_run(self) -> None:
        """End current MLflow run."""
        if not self.enabled:
            return
        
        mlflow.end_run()
        self.logger.info("Ended MLflow run")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters."""
        if not self.enabled:
            return
        
        mlflow.log_params(params)
        self.logger.debug(f"Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: int | None = None) -> None:
        """Log metrics."""
        if not self.enabled:
            return
        
        mlflow.log_metrics(metrics, step=step)
        self.logger.debug(f"Logged {len(metrics)} metrics")
    
    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        """Log artifact (file or directory)."""
        if not self.enabled:
            return
        
        mlflow.log_artifacts(str(local_path), artifact_path=artifact_path)
        self.logger.debug(f"Logged artifact: {local_path}")
    
    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        """Log PyTorch model."""
        if not self.enabled:
            return
        
        try:
            mlflow.pytorch.log_model(model, artifact_path)
            self.logger.info(f"Logged model to {artifact_path}")
        except Exception as e:
            self.logger.warning(f"Failed to log model: {e}")

