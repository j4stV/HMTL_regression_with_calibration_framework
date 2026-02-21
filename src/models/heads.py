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

    Outputs logits for K classes. Optionally supports temperature scaling
    for calibration (temperature is a learnable parameter).
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        use_temperature: bool = False,
    ) -> None:
        """Initialize classification head.

        Args:
            in_dim: Input feature dimension
            num_classes: Number of classes
            use_temperature: If True, adds learnable temperature parameter for calibration
        """
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        self.num_classes = num_classes
        self.use_temperature = use_temperature

        if use_temperature:
            self.temperature = nn.Parameter(torch.ones(1))

        # Initialize with LeCun normal (appropriate for SELU networks)
        lecun_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class logits.

        Args:
            h: Encoded features (batch_size, hidden_dim)

        Returns:
            logits: Class logits (batch_size, num_classes)
        """
        logits = self.fc(h)

        if self.use_temperature:
            # Temperature scaling for calibration
            logits = logits / self.temperature

        return logits


