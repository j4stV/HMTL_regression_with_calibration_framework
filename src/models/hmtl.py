from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .snn import SNNEncoder, RegressionHead
from .heads import AuxBinsHead
from .contrastive import ProjectionHead


class HMTLModel(nn.Module):
    """HMTL model.
    
    Architecture:
    - Low-level encoder: first 2/3 of layers (e.g., 12 out of 18)
    - Projection head: attached after low-level encoder (for contrastive learning)
    - High-level encoder: remaining layers (e.g., 6 out of 18)
    - Regression head: attached after high-level encoder
    """
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
        aux_task: str = "contrastive",  # "bins" or "contrastive" (default: contrastive )
        proj_dim: int = 50,  # Default 50
        scale_coeff: float = 1.0,  # Target std for sigma scaling
        task_head: nn.Module | None = None,  # NEW: Injectable task head for multi-task support
    ) -> None:
        super().__init__()
        self.enable_aux = enable_aux
        self.aux_weight = aux_weight
        self.aux_task = aux_task

        # Low-level encoder: first 2/3 of layers (e.g., 12 layers)
        self.encoder_low = SNNEncoder(input_dim, hidden_width, depth_low, alpha_dropout)
        # High-level encoder: remaining layers (e.g., 6 layers)
        self.encoder_high = SNNEncoder(self.encoder_low.output_dim, hidden_width, depth_high - depth_low, alpha_dropout)

        # Task head: injected for flexibility, defaults to RegressionHead for backward compatibility
        if task_head is None:
            self.task_head = RegressionHead(self.encoder_high.output_dim, scale_coeff=scale_coeff)
        else:
            self.task_head = task_head

        # Maintain backward compatibility: keep reg_head attribute
        self.reg_head = self.task_head if isinstance(self.task_head, RegressionHead) else None
        
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        h_low = self.encoder_low(x)
        h_high = self.encoder_high(h_low)

        # Use task_head (which may be RegressionHead or ClassificationHead)
        task_output = self.task_head(h_high)

        # Handle task output - maintain backward compatibility with (mu, sigma) naming
        # For regression: task_output = (mu, sigma)
        # For classification: task_output = logits (single tensor)
        if isinstance(task_output, tuple):
            # Regression case: (mu, sigma)
            output1, output2 = task_output
        else:
            # Classification case: logits
            # Return as tuple for consistency: (logits, None)
            output1, output2 = task_output, None

        # Auxiliary task output
        if self.enable_aux:
            if self.aux_task == "bins" and self.aux_head is not None:
                aux_logits = self.aux_head(h_low)
                return output1, output2, aux_logits
            elif self.aux_task == "contrastive" and self.proj_head is not None:
                projection = self.proj_head(h_low)
                return output1, output2, projection
            else:
                return output1, output2, None
        else:
            return output1, output2, None


