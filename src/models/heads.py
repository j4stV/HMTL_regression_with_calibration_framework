from __future__ import annotations

import torch
from torch import nn

from .snn import lecun_normal_


class AuxBinsHead(nn.Module):
    def __init__(self, in_dim: int, n_bins: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, n_bins)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)


class ClassificationHead(nn.Module):
    """Classification head with logits output.

    Uses a 2-layer MLP (hidden + output) with SELU activation for nonlinear
    decision boundaries. Optionally supports temperature scaling for calibration.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        use_temperature: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_temperature = use_temperature

        hidden = max(in_dim // 2, num_classes * 4, 32)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.SELU(inplace=True)
        self.drop = nn.AlphaDropout(p=0.05)
        self.fc2 = nn.Linear(hidden, num_classes)

        if use_temperature:
            self.temperature = nn.Parameter(torch.ones(1))

        # LeCun normal init (appropriate for SELU networks)
        for layer in [self.fc1, self.fc2]:
            lecun_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.act(self.fc1(h)))
        logits = self.fc2(h)
        if self.use_temperature:
            logits = logits / self.temperature
        return logits


