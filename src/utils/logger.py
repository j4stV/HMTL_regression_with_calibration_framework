from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False


_logger: logging.Logger | None = None
_log_file_path: Path | None = None


def setup_logging(log_dir: Path | str = "logs", log_level: int = logging.INFO) -> logging.Logger:
    """Initialize and configure logging system.
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    global _logger, _log_file_path
    
    if _logger is not None:
        return _logger
    
    # Create log directory
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir_path / f"run_{timestamp}.log"
    _log_file_path = log_file
    
    # Create logger
    logger = logging.getLogger("hmtl")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # Remove any existing handlers
    
    # File handler (DEBUG level - detailed logs)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] [%(name)s.%(funcName)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (INFO level - concise output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    if HAS_COLORLOG:
        console_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s [%(levelname)-8s]%(reset)s %(name)s - %(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            }
        )
    else:
        console_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
            datefmt="%H:%M:%S"
        )
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    _logger = logger
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get logger instance, initializing if necessary.
    
    Args:
        name: Optional logger name (defaults to 'hmtl')
        
    Returns:
        Logger instance
    """
    if _logger is None:
        setup_logging()
    
    if name:
        return logging.getLogger(f"hmtl.{name}")
    return _logger or logging.getLogger("hmtl")


@contextmanager
def log_timing(operation: str, logger: logging.Logger | None = None):
    """Context manager to log timing of operations.
    
    Args:
        operation: Description of the operation
        logger: Optional logger instance (uses default if None)
        
    Example:
        with log_timing("Data preprocessing"):
            # ... preprocessing code ...
    """
    if logger is None:
        logger = get_logger()
    
    start_time = datetime.now()
    logger.info(f"Starting: {operation}")
    
    try:
        yield
    finally:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Completed: {operation} (took {elapsed:.2f}s)")


def log_config(config: Dict[str, Any], logger: logging.Logger | None = None, prefix: str = "Config"):
    """Log configuration dictionary in a structured way.
    
    Args:
        config: Configuration dictionary
        logger: Optional logger instance
        prefix: Prefix for log message
    """
    if logger is None:
        logger = get_logger()
    
    logger.info(f"{prefix}:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.debug(f"  {key}:")
            for k, v in value.items():
                logger.debug(f"    {k}: {v}")
        else:
            logger.debug(f"  {key}: {value}")


def log_metrics(metrics: Dict[str, float], logger: logging.Logger | None = None, prefix: str = "Metrics"):
    """Log metrics dictionary.
    
    Args:
        metrics: Dictionary of metric names to values
        logger: Optional logger instance
        prefix: Prefix for log message
    """
    if logger is None:
        logger = get_logger()
    
    logger.info(f"{prefix}:")
    for name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {name}: {value:.6f}")
        else:
            logger.info(f"  {name}: {value}")

