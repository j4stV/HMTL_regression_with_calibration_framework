"""Single MLP baseline without HMTL."""

from __future__ import annotations

import torch
from torch import nn

from src.models.snn import SNNEncoder, RegressionHead


class SingleMLPModel(nn.Module):
    """Single MLP model without hierarchical multitask learning."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_width: int,
        depth: int,
        alpha_dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = SNNEncoder(input_dim, hidden_width, depth, alpha_dropout)
        self.reg_head = RegressionHead(self.encoder.output_dim)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Returns:
            mu, sigma: Mean and standard deviation predictions
        """
        h = self.encoder(x)
        mu, sigma = self.reg_head(h)
        return mu, sigma

