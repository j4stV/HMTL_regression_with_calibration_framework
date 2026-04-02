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
        aux_task: str = "contrastive",  # "bins", "contrastive", "reconstruction", "rank", "multi"
        proj_dim: int = 50,  # Default 50
        scale_coeff: float = 1.0,  # Target std for sigma scaling
        task_head: nn.Module | None = None,  # Injectable task head for multi-task support
        use_residual: bool = True,
        quantile_head: nn.Module | None = None,  # Optional CQR quantile head
        # Multi-aux settings (used when aux_task="multi")
        multi_aux_tasks: list[str] | None = None,
        multi_aux_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.enable_aux = enable_aux
        self.aux_weight = aux_weight
        self.aux_task = aux_task
        self._cached_h_low: torch.Tensor | None = None

        # Low-level encoder: first 2/3 of layers (e.g., 12 layers)
        self.encoder_low = SNNEncoder(
            input_dim,
            hidden_width,
            depth_low,
            alpha_dropout,
            use_residual=use_residual,
        )
        # High-level encoder: remaining layers (e.g., 6 layers)
        self.encoder_high = SNNEncoder(
            self.encoder_low.output_dim,
            hidden_width,
            depth_high - depth_low,
            alpha_dropout,
            use_residual=use_residual,
        )

        # Task head: injected for flexibility, defaults to RegressionHead for backward compatibility
        if task_head is None:
            self.task_head = RegressionHead(self.encoder_high.output_dim, scale_coeff=scale_coeff)
        else:
            self.task_head = task_head

        # Maintain backward compatibility: keep reg_head attribute
        self.reg_head = self.task_head if isinstance(self.task_head, RegressionHead) else None

        # Optional CQR quantile head
        self.quantile_head = quantile_head

        # Auxiliary heads
        if enable_aux:
            if aux_task == "bins":
                self.aux_head = AuxBinsHead(self.encoder_low.output_dim, n_bins)
                self.proj_head = None
            elif aux_task == "contrastive":
                self.aux_head = None
                self.proj_head = ProjectionHead(self.encoder_low.output_dim, proj_dim=proj_dim)
            elif aux_task == "reconstruction":
                from .aux_heads import ReconstructionHead
                self.aux_head = None
                self.proj_head = None
                self.reconstruction_head = ReconstructionHead(self.encoder_low.output_dim)
            elif aux_task == "rank":
                from .aux_heads import RankHead
                self.aux_head = None
                self.proj_head = None
                self.rank_head = RankHead(self.encoder_low.output_dim)
            elif aux_task == "multi":
                from .aux_heads import ReconstructionHead, RankHead
                self.aux_head = None
                self.proj_head = None
                self.multi_aux_heads = nn.ModuleDict()
                self.multi_aux_weights = multi_aux_weights or {}
                tasks = multi_aux_tasks or []
                low_dim = self.encoder_low.output_dim
                for t in tasks:
                    if t == "bins":
                        self.multi_aux_heads[t] = AuxBinsHead(low_dim, n_bins)
                    elif t == "contrastive":
                        self.multi_aux_heads[t] = ProjectionHead(low_dim, proj_dim=proj_dim)
                    elif t == "reconstruction":
                        self.multi_aux_heads[t] = ReconstructionHead(low_dim)
                    elif t == "rank":
                        self.multi_aux_heads[t] = RankHead(low_dim)
                    else:
                        raise ValueError(f"Unknown multi-aux task: {t}")
            else:
                raise ValueError(f"Unknown aux_task: {aux_task}")
        else:
            self.aux_head = None
            self.proj_head = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        h_low = self.encoder_low(x)
        self._cached_h_low = h_low
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
            elif self.aux_task == "reconstruction" and hasattr(self, "reconstruction_head"):
                reconstructed = self.reconstruction_head(h_low)
                return output1, output2, reconstructed
            elif self.aux_task == "rank" and hasattr(self, "rank_head"):
                rank_scores = self.rank_head(h_low)
                return output1, output2, rank_scores
            elif self.aux_task == "multi" and hasattr(self, "multi_aux_heads"):
                aux_outputs = {}
                for name, head in self.multi_aux_heads.items():
                    aux_outputs[name] = head(h_low)
                return output1, output2, aux_outputs
            else:
                return output1, output2, None
        else:
            return output1, output2, None

    def predict_quantiles(self, x: torch.Tensor) -> torch.Tensor:
        """Predict quantiles for CQR.

        Args:
            x: Input features (batch_size, input_dim).

        Returns:
            Quantile predictions (batch_size, n_quantiles).

        Raises:
            RuntimeError: If no quantile_head is attached.
        """
        if self.quantile_head is None:
            raise RuntimeError("No quantile_head attached to this model.")
        h_low = self.encoder_low(x)
        h_high = self.encoder_high(h_low)
        return self.quantile_head(h_high)
