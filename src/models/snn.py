from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


def lecun_normal_(tensor: torch.Tensor) -> torch.Tensor:
    fan_in = nn.init._calculate_correct_fan(tensor, "fan_in")
    std = (1.0 / max(1.0, fan_in)) ** 0.5
    return nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)


class AlphaDropout(nn.AlphaDropout):
    pass


class SNNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_width: int, depth: int, alpha_dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(depth):
            linear = nn.Linear(in_dim, hidden_width)
            lecun_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.SELU(inplace=True))
            if alpha_dropout > 0:
                layers.append(AlphaDropout(p=alpha_dropout))
            in_dim = hidden_width
        self.net = nn.Sequential(*layers)
        self.output_dim = hidden_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegressionHead(nn.Module):
    """Regression head.
    
    Scale = 1e-6 + softplus((1.0 / scale_coeff) * raw_sigma)
    where scale_coeff is the target standard deviation.
    This allows sigma to be learned in standardized space and scaled back.
    """ 
    def __init__(self, in_dim: int, scale_coeff: float = 1.0) -> None:
        super().__init__()
        self.mu = nn.Linear(in_dim, 1, bias=False)
        self.raw_sigma = nn.Linear(in_dim, 1, bias=False)
        self.scale_coeff = scale_coeff
        lecun_normal_(self.mu.weight)
        lecun_normal_(self.raw_sigma.weight)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu(h)
        # Scale = 1e-6 + softplus((1.0 / scale_coeff) * raw_sigma)
        raw_sigma = self.raw_sigma(h)
        sigma = 1e-6 + torch.nn.functional.softplus((1.0 / self.scale_coeff) * raw_sigma)
        return mu, sigma

# Повышение устойчивости через аугментации - подумать
# Оценка устойчивости к сдвигу данных - подумать как промоделировать, возможно HMTL более устойчива
