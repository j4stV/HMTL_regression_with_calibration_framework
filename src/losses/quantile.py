from __future__ import annotations

import torch
from torch import Tensor


def pinball_loss(
    y_pred_quantiles: Tensor,
    y_true: Tensor,
    quantiles: list[float],
) -> Tensor:
    """Pinball (quantile) loss for quantile regression.

    For each quantile tau:
        L_tau(y, q) = tau * max(y - q, 0) + (1 - tau) * max(q - y, 0)

    Args:
        y_pred_quantiles: Predicted quantiles (batch_size, n_quantiles).
        y_true: True target values (batch_size, 1) or (batch_size,).
        quantiles: List of quantile levels corresponding to columns of y_pred_quantiles.

    Returns:
        Scalar mean loss.
    """
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(1)

    errors = y_true - y_pred_quantiles  # (batch, n_quantiles)

    loss = torch.zeros(1, device=y_pred_quantiles.device, dtype=y_pred_quantiles.dtype)
    for i, tau in enumerate(quantiles):
        e = errors[:, i]
        loss = loss + torch.mean(torch.where(e >= 0, tau * e, (tau - 1.0) * e))

    return loss / len(quantiles)
