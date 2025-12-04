from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.optim import Optimizer


class RAdam(Optimizer):
    """Rectified Adam optimizer.
    
    Implementation based on: https://arxiv.org/abs/1908.03265
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute bias-corrected moving averages
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Rectification
                rho_inf = 2 / (1 - beta2) - 1
                rho_t = rho_inf - 2 * state['step'] * beta2 ** state['step'] / (1 - beta2 ** state['step'])
                
                if rho_t > 4:
                    # Rectified update
                    r_t = ((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t)) ** 0.5
                    exp_avg_sq_rect = exp_avg_sq / bias_correction2
                    denom = (exp_avg_sq_rect.sqrt() / r_t).add_(group['eps'])
                    step_size = group['lr'] * r_t / bias_correction1
                else:
                    # Unrectified update (like Adam)
                    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                    step_size = group['lr'] / bias_correction1
                
                if group['weight_decay'] > 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss


class Lookahead:
    """Lookahead optimizer wrapper.
    
    Implementation based on: https://arxiv.org/abs/1907.08610
    
    Args:
        base_optimizer: Base optimizer (e.g., RAdam)
        k: Synchronization period (number of steps before updating slow weights)
        alpha: Slow step size (fraction of update to apply to slow weights)
    """
    
    def __init__(
        self,
        base_optimizer: Optimizer,
        k: int = 6,
        alpha: float = 0.5,
    ):
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0
        
        # Initialize slow weights
        self.slow_weights = {}
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                self.slow_weights[id(p)] = p.data.clone()
    
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = self.base_optimizer.step(closure)
        self.step_count += 1
        
        # Update slow weights every k steps
        if self.step_count % self.k == 0:
            for group in self.base_optimizer.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    slow_p = self.slow_weights[id(p)]
                    # Update slow weights: slow = slow + alpha * (fast - slow)
                    slow_p.add_(p.data - slow_p, alpha=self.alpha)
                    # Copy slow weights back to fast weights
                    p.data.copy_(slow_p)
        
        return loss
    
    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        self.base_optimizer.zero_grad()
    
    def state_dict(self):
        """Returns the state of the optimizer as a dict."""
        state = {
            'base_optimizer': self.base_optimizer.state_dict(),
            'slow_weights': {k: v.clone() for k, v in self.slow_weights.items()},
            'step_count': self.step_count,
            'k': self.k,
            'alpha': self.alpha,
        }
        return state
    
    def load_state_dict(self, state_dict):
        """Loads the optimizer state."""
        self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        self.slow_weights = {k: v.clone() for k, v in state_dict['slow_weights'].items()}
        self.step_count = state_dict['step_count']
        self.k = state_dict.get('k', self.k)
        self.alpha = state_dict.get('alpha', self.alpha)
    
    def __getattr__(self, name):
        """Delegate attribute access to base optimizer."""
        return getattr(self.base_optimizer, name)


def create_radam_lookahead(
    model: nn.Module,
    lr: float = 3e-4,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    lookahead_k: int = 6,
    lookahead_alpha: float = 0.5,
) -> Lookahead:
    """Create RAdam + Lookahead optimizer.
    
    Args:
        model: Model to optimize
        lr: Learning rate
        betas: Beta parameters for RAdam
        eps: Epsilon for numerical stability
        weight_decay: Weight decay
        lookahead_k: Lookahead synchronization period
        lookahead_alpha: Lookahead slow step size
    
    Returns:
        Lookahead optimizer wrapping RAdam
    """
    base_optimizer = RAdam(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )
    return Lookahead(base_optimizer, k=lookahead_k, alpha=lookahead_alpha)

