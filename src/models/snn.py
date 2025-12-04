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
    def __init__(self, in_dim: int, sigma_max: float = 5.0) -> None:
        super().__init__()
        self.mu = nn.Linear(in_dim, 1)
        self.raw_sigma = nn.Linear(in_dim, 1)
        self.sigma_max = sigma_max  # Верхний предел для sigma (в стандартизированном пространстве)
        lecun_normal_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        lecun_normal_(self.raw_sigma.weight)
        nn.init.zeros_(self.raw_sigma.bias)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu(h)
        # Ограничиваем sigma сверху для предотвращения взрыва неопределенности
        # Используем более мягкую активацию с ограничением
        sigma = torch.nn.functional.softplus(self.raw_sigma(h)) + 1e-6
        sigma = torch.clamp(sigma, min=1e-6, max=self.sigma_max)
        return mu, sigma

# Повышение устойчивости через аугментации - подумать
# Оценка устойчивости к сдвигу данных - подумать как промоделировать, возможно HMTL более устойчива
