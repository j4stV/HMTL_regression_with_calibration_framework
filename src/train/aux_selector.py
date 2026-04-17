from __future__ import annotations

import math
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
    pilot_epochs: int = 60,
    stability_weight: float = 0.1,
) -> str:
    """Select the best auxiliary task by running short pilot training runs.

    Trains a model with each candidate aux task for a limited number of epochs,
    then selects the candidate that achieves the best stability-penalized
    validation score.

    The stability penalty discourages candidates that look good on their best
    epoch but exhibit high variance — a sign of training instability that
    worsens over the full training run.

    Args:
        build_model_fn: Callable that takes aux_task string and returns a fresh model.
        X_tr: Training features.
        y_tr: Training targets.
        X_va: Validation features.
        y_va: Validation targets.
        n_bins: Number of bins for aux tasks.
        train_cfg: TrainConfig instance (will be copied with reduced epochs).
        candidates: List of aux task names to try. Defaults to ["bins", "contrastive"].
        pilot_epochs: Number of epochs per pilot run.
        stability_weight: Weight for the stability penalty (std of last 15 val scores).

    Returns:
        Name of the best auxiliary task.
    """
    from src.train.loop import train_model

    logger = get_logger("train.aux_selector")

    if candidates is None:
        candidates = ["bins", "contrastive"]

    logger.info(
        f"Auto-selecting aux task from candidates: {candidates} "
        f"(pilot_epochs={pilot_epochs}, stability_weight={stability_weight})"
    )

    best_task = candidates[0]  # fallback: first candidate (bins)
    best_penalized_score = float("inf")

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

        history: list[dict] = []
        try:
            score = train_model(
                model=model,
                X_tr=X_tr,
                y_tr=y_tr,
                X_va=X_va,
                y_va=y_va,
                n_bins=n_bins,
                cfg=pilot_cfg,
                history=history,
            )
        except Exception as e:
            logger.warning(f"Pilot training failed for aux_task={candidate}: {e}")
            continue

        # Skip candidates that produced NaN/Inf
        if not math.isfinite(score):
            logger.warning(
                f"Pilot result for aux_task={candidate}: score={score} (NaN/Inf). Skipping."
            )
            continue

        # Compute stability penalty from the last 15 validation scores
        val_scores = [
            h["val_score"]
            for h in history
            if "val_score" in h and math.isfinite(h["val_score"])
        ]
        tail_n = min(15, len(val_scores))
        if tail_n >= 3:
            tail_std = float(np.std(val_scores[-tail_n:]))
        else:
            tail_std = 0.0

        penalized_score = score + stability_weight * tail_std

        logger.info(
            f"Pilot result: aux_task={candidate}, best_score={score:.6f}, "
            f"tail_std={tail_std:.6f}, penalized_score={penalized_score:.6f}"
        )

        if penalized_score < best_penalized_score:
            best_penalized_score = penalized_score
            best_task = candidate

    logger.info(
        f"Auto-selected aux_task: {best_task} "
        f"(penalized_score={best_penalized_score:.6f})"
    )

    return best_task
