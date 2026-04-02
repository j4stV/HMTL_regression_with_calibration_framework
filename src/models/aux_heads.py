from __future__ import annotations

import math

import torch
from torch import nn


class ReconstructionHead(nn.Module):
    """Autoencoder-style reconstruction head for auxiliary task.

    Reconstructs the low-level encoder output through a bottleneck.

    Args:
        in_dim: Input/output dimension (matches encoder_low output).
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        bottleneck_dim = max(in_dim // 4, 8)
        self.encoder = nn.Linear(in_dim, bottleneck_dim)
        self.decoder = nn.Linear(bottleneck_dim, in_dim)
        self.activation = nn.SELU()

        # LeCun normal initialization
        nn.init.normal_(self.encoder.weight, 0.0, 1.0 / math.sqrt(in_dim))
        nn.init.zeros_(self.encoder.bias)
        nn.init.normal_(self.decoder.weight, 0.0, 1.0 / math.sqrt(bottleneck_dim))
        nn.init.zeros_(self.decoder.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Reconstruct input features.

        Args:
            h: Low-level encoder output (batch_size, in_dim).

        Returns:
            Reconstructed features (batch_size, in_dim).
        """
        return self.decoder(self.activation(self.encoder(h)))


class RankHead(nn.Module):
    """Ranking head for auxiliary pairwise ranking task.

    Outputs a scalar ranking score per sample.

    Args:
        in_dim: Input dimension.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

        # LeCun normal initialization
        nn.init.normal_(self.linear.weight, 0.0, 1.0 / math.sqrt(in_dim))
        nn.init.zeros_(self.linear.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Compute ranking score.

        Args:
            h: Low-level encoder output (batch_size, in_dim).

        Returns:
            Ranking scores (batch_size, 1).
        """
        return self.linear(h)
