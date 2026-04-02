from __future__ import annotations

import math

import torch
from torch import nn


class QuantileHead(nn.Module):
    """Head for quantile regression predictions.

    Predicts conditional quantiles of the target distribution.
    Enforces monotonicity (non-crossing) via softplus parameterization.

    Args:
        in_dim: Input feature dimension.
        quantiles: List of quantile levels (e.g., [0.05, 0.95]).
    """

    def __init__(self, in_dim: int, quantiles: list[float] | None = None) -> None:
        super().__init__()
        if quantiles is None:
            quantiles = [0.05, 0.95]
        self.quantiles = sorted(quantiles)
        n_quantiles = len(self.quantiles)

        # First quantile predicted directly, subsequent as offsets (softplus)
        self.linear = nn.Linear(in_dim, n_quantiles)

        # LeCun normal initialization
        nn.init.normal_(self.linear.weight, 0.0, 1.0 / math.sqrt(in_dim))
        nn.init.zeros_(self.linear.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            h: Encoded features (batch_size, in_dim).

        Returns:
            Quantile predictions (batch_size, n_quantiles), monotonically ordered.
        """
        raw = self.linear(h)  # (batch, n_quantiles)

        if raw.shape[1] == 1:
            return raw

        # Enforce monotonicity: first quantile raw, subsequent = prev + softplus(delta)
        outputs = [raw[:, 0:1]]
        for i in range(1, raw.shape[1]):
            delta = torch.nn.functional.softplus(raw[:, i:i+1])
            outputs.append(outputs[-1] + delta)
        return torch.cat(outputs, dim=1)
