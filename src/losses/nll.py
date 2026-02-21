from __future__ import annotations

import torch


def gaussian_nll(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    y: torch.Tensor,
    sigma_reg_weight: float = 0.01,
) -> torch.Tensor:
    """Gaussian negative log-likelihood with optional sigma regularization.

    Args:
        mu: Predicted mean
        sigma: Predicted standard deviation
        y: Ground truth targets
        sigma_reg_weight: Weight for sigma regularization term (penalizes large sigma values)
    """
    var = sigma.clamp_min(1e-6) ** 2
    nll = 0.5 * (torch.log(2 * torch.pi * var) + (y - mu) ** 2 / var).mean()

    # Регуляризация sigma: штрафуем большие значения для предотвращения взрыва неопределенности
    # Используем L2 регуляризацию на среднее значение sigma
    if sigma_reg_weight > 0:
        sigma_reg = sigma_reg_weight * torch.mean(sigma ** 2)
        nll = nll + sigma_reg

    return nll


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    class_weights: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal loss for imbalanced classification.

    Focal loss down-weights easy examples and focuses on hard examples.
    Particularly useful for class-imbalanced datasets.

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        logits: Predicted logits (batch_size, num_classes)
        targets: Ground truth class indices (batch_size,)
        alpha: Weighting factor in [0, 1] to balance positive/negative examples
        gamma: Focusing parameter (gamma >= 0). Higher gamma = more focus on hard examples
        class_weights: Optional per-class weights (num_classes,)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Focal loss (scalar if reduction != 'none', otherwise per-sample losses)
    """
    # Compute cross-entropy loss (no reduction yet)
    ce_loss = torch.nn.functional.cross_entropy(
        logits, targets, weight=class_weights, reduction="none"
    )

    # Compute p_t: probability of true class
    probs = torch.nn.functional.softmax(logits, dim=-1)
    p_t = probs[torch.arange(len(targets), device=targets.device), targets]

    # Focal weight: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** gamma

    # Focal loss: alpha * focal_weight * ce_loss
    focal_loss_values = alpha * focal_weight * ce_loss

    # Apply reduction
    if reduction == "mean":
        return focal_loss_values.mean()
    elif reduction == "sum":
        return focal_loss_values.sum()
    else:  # reduction == "none"
        return focal_loss_values


