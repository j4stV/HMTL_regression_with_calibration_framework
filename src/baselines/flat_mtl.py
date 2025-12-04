"""Flat MTL baseline without hierarchy."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from src.models.snn import SNNEncoder, RegressionHead
from src.models.heads import AuxBinsHead


class FlatMTLModel(nn.Module):
    """Flat multitask learning model (no hierarchy).
    
    Both regression and auxiliary tasks use the same encoder output.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_width: int,
        depth: int,
        alpha_dropout: float,
        n_bins: int,
        aux_weight: float,
        enable_aux: bool = True,
    ) -> None:
        super().__init__()
        self.enable_aux = enable_aux
        self.aux_weight = aux_weight
        
        self.encoder = SNNEncoder(input_dim, hidden_width, depth, alpha_dropout)
        self.reg_head = RegressionHead(self.encoder.output_dim)
        self.aux_head = AuxBinsHead(self.encoder.output_dim, n_bins) if enable_aux else None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Forward pass.
        
        Returns:
            mu, sigma, logits: Regression outputs and auxiliary task logits
        """
        h = self.encoder(x)
        mu, sigma = self.reg_head(h)
        logits = self.aux_head(h) if self.enable_aux and self.aux_head is not None else None
        return mu, sigma, logits

