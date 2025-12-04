from __future__ import annotations

import torch
from torch import nn


class AuxBinsHead(nn.Module):
    def __init__(self, in_dim: int, n_bins: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, n_bins)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)


