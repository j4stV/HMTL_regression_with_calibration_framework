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


