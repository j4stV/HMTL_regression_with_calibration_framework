from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from src.utils.logger import get_logger


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning.
    
    Projection is a single linear layer without bias, output size 50.
    """
    
    def __init__(self, in_dim: int, proj_dim: int = 50) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, proj_dim, bias=False)
        # Initialize with Lecun normal
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_in', nonlinearity='linear')
        # Lecun normal for linear layer
        fan_in = self.fc.weight.size(1)
        std = (1.0 / max(1.0, fan_in)) ** 0.5
        with torch.no_grad():
            self.fc.weight.data = torch.clamp(
                torch.normal(0, std, size=self.fc.weight.shape),
                -2 * std, 2 * std
            )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.fc(h)
        # L2 normalize ()
        z = z / (torch.norm(z, dim=1, keepdim=True) + 1e-8)
        return z


class NPairsLoss(nn.Module):
    """N-pairs loss for supervised contrastive learning.
    
    For each anchor, pull positive samples (same class) together
    and push negative samples (different classes) apart.
    """
    
    def __init__(self, temperature: float = 0.5) -> None:
        """N-pairs loss with temperature parameter.
        
        Temperature is 0.5.
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, projections: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute N-pairs contrastive loss.
        
        Args:
            projections: Projected features (batch_size, proj_dim)
            labels: Class labels (batch_size,)
        
        Returns:
            Contrastive loss
        """
        batch_size = projections.size(0)
        device = projections.device
        
        # Compute similarity matrix
        similarity = torch.matmul(projections, projections.t()) / self.temperature
        
        # Create mask for positive pairs (same label)
        labels = labels.unsqueeze(1)
        mask = (labels == labels.t()).float().to(device)
        
        # Remove diagonal (self-similarity)
        mask.fill_diagonal_(0)
        
        # For each anchor, compute log-sum-exp over positives
        # and subtract log-sum-exp over all pairs
        exp_sim = torch.exp(similarity)
        
        # Sum over positive pairs for each anchor
        pos_sum = (exp_sim * mask).sum(dim=1, keepdim=True)
        
        # Sum over all pairs (excluding self)
        all_sum = exp_sim.sum(dim=1, keepdim=True) - torch.exp(similarity.diag()).unsqueeze(1)
        
        # Loss: -log(pos_sum / all_sum) = log(all_sum) - log(pos_sum)
        # Add small epsilon for numerical stability
        loss = torch.log(all_sum + 1e-8) - torch.log(pos_sum + 1e-8)
        
        # Average over batch
        return loss.mean()


def n_pairs_loss(projections: torch.Tensor, labels: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Convenience function for N-pairs loss.
    
    Default temperature is 0.5.
    """
    criterion = NPairsLoss(temperature=temperature)
    return criterion(projections, labels)

