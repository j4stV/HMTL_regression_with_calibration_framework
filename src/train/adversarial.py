from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


@dataclass
class AdversarialConfig:
    """Configuration for adversarial augmentations during training."""
    enabled: bool = False
    method: str = "fgsm"       # "fgsm" or "pgd"
    epsilon: float = 0.01      # perturbation magnitude
    alpha: float = 0.005       # PGD step size
    pgd_steps: int = 3         # PGD iterations
    adv_weight: float = 0.5    # weight of adversarial loss


def fgsm_attack(
    model: torch.nn.Module,
    x: Tensor,
    loss_fn: Callable[[Tensor], Tensor],
    epsilon: float,
) -> Tensor:
    """Generate FGSM adversarial examples.

    Args:
        model: The model (must be in train mode).
        x: Clean input batch.
        loss_fn: Callable that takes x_input and returns scalar loss.
        epsilon: Perturbation magnitude.

    Returns:
        Adversarial input tensor (detached).
    """
    x_input = x.detach().clone().requires_grad_(True)
    loss = loss_fn(x_input)
    grad = torch.autograd.grad(loss, x_input, create_graph=False)[0]
    x_adv = x_input + epsilon * grad.sign()
    return x_adv.detach()


def pgd_attack(
    model: torch.nn.Module,
    x: Tensor,
    loss_fn: Callable[[Tensor], Tensor],
    epsilon: float,
    alpha: float,
    steps: int,
) -> Tensor:
    """Generate PGD adversarial examples.

    Args:
        model: The model (must be in train mode).
        x: Clean input batch.
        loss_fn: Callable that takes x_input and returns scalar loss.
        epsilon: Maximum perturbation magnitude (L-inf ball).
        alpha: Step size per iteration.
        steps: Number of PGD steps.

    Returns:
        Adversarial input tensor (detached).
    """
    # Initialize with random perturbation within epsilon-ball
    delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
    x_adv = (x + delta).detach()

    for _ in range(steps):
        x_adv = x_adv.clone().requires_grad_(True)
        loss = loss_fn(x_adv)
        grad = torch.autograd.grad(loss, x_adv, create_graph=False)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        # Project back to epsilon-ball around original x
        perturbation = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = (x + perturbation).detach()

    return x_adv


def generate_adversarial(
    model: torch.nn.Module,
    x: Tensor,
    loss_fn: Callable[[Tensor], Tensor],
    config: AdversarialConfig,
) -> Tensor:
    """Generate adversarial examples based on config.

    Args:
        model: The model.
        x: Clean input batch.
        loss_fn: Callable that takes x_input and returns scalar loss.
        config: Adversarial configuration.

    Returns:
        Adversarial input tensor (detached).
    """
    if config.method == "fgsm":
        return fgsm_attack(model, x, loss_fn, config.epsilon)
    elif config.method == "pgd":
        return pgd_attack(
            model, x, loss_fn,
            epsilon=config.epsilon,
            alpha=config.alpha,
            steps=config.pgd_steps,
        )
    else:
        raise ValueError(f"Unknown adversarial method: {config.method}. Expected 'fgsm' or 'pgd'.")
