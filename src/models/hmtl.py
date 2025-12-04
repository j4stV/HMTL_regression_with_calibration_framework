from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .snn import SNNEncoder, RegressionHead
from .heads import AuxBinsHead
from .contrastive import ProjectionHead


class HMTLModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_width: int,
        depth_low: int,
        depth_high: int,
        alpha_dropout: float,
        n_bins: int,
        aux_weight: float,
        enable_aux: bool = True,
        aux_task: str = "bins",  # "bins" or "contrastive"
        proj_dim: int = 128,
        sigma_max: float = 5.0,
    ) -> None:
        super().__init__()
        self.enable_aux = enable_aux
        self.aux_weight = aux_weight
        self.aux_task = aux_task

        self.encoder_low = SNNEncoder(input_dim, hidden_width, depth_low, alpha_dropout)
        self.encoder_high = SNNEncoder(self.encoder_low.output_dim, hidden_width, depth_high - depth_low, alpha_dropout)
        self.reg_head = RegressionHead(self.encoder_high.output_dim, sigma_max=sigma_max)
        
        if enable_aux:
            if aux_task == "bins":
                self.aux_head = AuxBinsHead(self.encoder_low.output_dim, n_bins)
                self.proj_head = None
            elif aux_task == "contrastive":
                self.aux_head = None
                self.proj_head = ProjectionHead(self.encoder_low.output_dim, proj_dim=proj_dim)
            else:
                raise ValueError(f"Unknown aux_task: {aux_task}")
        else:
            self.aux_head = None
            self.proj_head = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        h_low = self.encoder_low(x)
        h_high = self.encoder_high(h_low)
        mu, sigma = self.reg_head(h_high)
        
        if self.enable_aux:
            if self.aux_task == "bins" and self.aux_head is not None:
                logits = self.aux_head(h_low)
                return mu, sigma, logits
            elif self.aux_task == "contrastive" and self.proj_head is not None:
                projection = self.proj_head(h_low)
                return mu, sigma, projection
            else:
                return mu, sigma, None
        else:
            return mu, sigma, None


