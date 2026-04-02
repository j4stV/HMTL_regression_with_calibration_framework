from __future__ import annotations

import torch
from torch import Tensor


def reconstruction_loss(h_original: Tensor, h_reconstructed: Tensor) -> Tensor:
    """MSE reconstruction loss for autoencoder auxiliary task.

    Args:
        h_original: Original encoder output (batch_size, dim).
        h_reconstructed: Reconstructed encoder output (batch_size, dim).

    Returns:
        Scalar MSE loss.
    """
    return torch.nn.functional.mse_loss(h_reconstructed, h_original)


def pairwise_ranking_loss(
    scores: Tensor,
    targets: Tensor,
    margin: float = 1.0,
    n_pairs: int = 256,
) -> Tensor:
    """Pairwise ranking loss for auxiliary ranking task.

    For pairs (i, j) where y_i > y_j, penalizes when score_i <= score_j + margin.
    Uses random pair sampling for efficiency.

    Args:
        scores: Ranking scores (batch_size, 1).
        targets: Target values (batch_size,) or (batch_size, 1).
        margin: Margin for hinge loss.
        n_pairs: Number of random pairs to sample.

    Returns:
        Scalar ranking loss.
    """
    scores = scores.view(-1)
    targets = targets.view(-1)
    batch_size = scores.shape[0]

    if batch_size < 2:
        return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)

    # Sample random pairs
    n_pairs = min(n_pairs, batch_size * (batch_size - 1) // 2)
    idx_i = torch.randint(0, batch_size, (n_pairs,), device=scores.device)
    idx_j = torch.randint(0, batch_size, (n_pairs,), device=scores.device)

    # Ensure i != j
    mask = idx_i != idx_j
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]

    if idx_i.shape[0] == 0:
        return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)

    # Get target differences: positive means y_i > y_j
    target_diff = targets[idx_i] - targets[idx_j]
    score_diff = scores[idx_i] - scores[idx_j]

    # Sign: +1 if y_i > y_j, -1 otherwise
    sign = torch.sign(target_diff)
    # Filter out equal targets
    nonzero = sign != 0
    if nonzero.sum() == 0:
        return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)

    sign = sign[nonzero]
    score_diff = score_diff[nonzero]

    # Hinge loss: max(0, margin - sign * score_diff)
    loss = torch.clamp(margin - sign * score_diff, min=0.0)
    return loss.mean()
