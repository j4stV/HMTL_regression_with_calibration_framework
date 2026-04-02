from __future__ import annotations

import torch
from torch import nn

from src.models.snn import SNNEncoder


def _set_constant_block_parameters(encoder: SNNEncoder, bias: float = 1.0) -> None:
    for block in encoder.blocks:
        linear = block[0]
        nn.init.zeros_(linear.weight)
        nn.init.constant_(linear.bias, bias)


def test_snn_encoder_first_layer_skips_residual_when_input_dim_differs() -> None:
    x = torch.tensor([[1.0, -2.0]], dtype=torch.float32)

    with_residual = SNNEncoder(2, 4, 1, 0.0, use_residual=True)
    without_residual = SNNEncoder(2, 4, 1, 0.0, use_residual=False)
    _set_constant_block_parameters(with_residual)
    _set_constant_block_parameters(without_residual)

    out_with_residual = with_residual(x)
    out_without_residual = without_residual(x)

    assert out_with_residual.shape == (1, 4)
    assert torch.allclose(out_with_residual, out_without_residual)


def test_snn_encoder_applies_residual_once_width_matches() -> None:
    x = torch.tensor([[1.0, -2.0]], dtype=torch.float32)

    with_residual = SNNEncoder(2, 4, 2, 0.0, use_residual=True)
    without_residual = SNNEncoder(2, 4, 2, 0.0, use_residual=False)
    _set_constant_block_parameters(with_residual)
    _set_constant_block_parameters(without_residual)

    out_with_residual = with_residual(x)
    out_without_residual = without_residual(x)

    assert out_with_residual.shape == out_without_residual.shape == (1, 4)
    assert torch.allclose(out_with_residual, out_without_residual * 2.0, atol=1e-6)
