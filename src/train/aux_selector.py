from __future__ import annotations

from copy import deepcopy
from typing import Callable

import numpy as np

from src.utils.logger import get_logger


def select_best_aux_task(
    build_model_fn: Callable,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    n_bins: int,
    train_cfg,
    candidates: list[str] | None = None,
    pilot_epochs: int = 30,
) -> str:
    """Select the best auxiliary task by running short pilot training runs.

    Trains a model with each candidate aux task for a limited number of epochs,
    then selects the candidate that achieves the best validation score.

    Args:
        build_model_fn: Callable that takes aux_task string and returns a fresh model.
        X_tr: Training features.
        y_tr: Training targets.
        X_va: Validation features.
        y_va: Validation targets.
        n_bins: Number of bins for aux tasks.
        train_cfg: TrainConfig instance (will be copied with reduced epochs).
        candidates: List of aux task names to try. Defaults to all available.
        pilot_epochs: Number of epochs per pilot run.

    Returns:
        Name of the best auxiliary task.
    """
    from src.train.loop import TrainConfig, train_model

    logger = get_logger("train.aux_selector")

    if candidates is None:
        candidates = ["bins", "contrastive", "reconstruction", "rank"]

    logger.info(
        f"Auto-selecting aux task from candidates: {candidates} "
        f"(pilot_epochs={pilot_epochs})"
    )

    best_task = candidates[0]
    best_score = float("inf")

    for candidate in candidates:
        logger.info(f"Pilot run: aux_task={candidate}")

        try:
            model = build_model_fn(aux_task=candidate)
        except Exception as e:
            logger.warning(f"Failed to build model with aux_task={candidate}: {e}")
            continue

        # Create pilot config with reduced epochs and no progress bar
        pilot_cfg = deepcopy(train_cfg)
        pilot_cfg.epochs = pilot_epochs
        pilot_cfg.patience = max(pilot_epochs // 3, 5)
        pilot_cfg.show_progress = False

        try:
            score = train_model(
                model=model,
                X_tr=X_tr,
                y_tr=y_tr,
                X_va=X_va,
                y_va=y_va,
                n_bins=n_bins,
                cfg=pilot_cfg,
            )
        except Exception as e:
            logger.warning(f"Pilot training failed for aux_task={candidate}: {e}")
            continue

        logger.info(f"Pilot result: aux_task={candidate}, score={score:.6f}")

        if score < best_score:
            best_score = score
            best_task = candidate

    logger.info(
        f"Auto-selected aux_task: {best_task} (score={best_score:.6f})"
    )

    return best_task
