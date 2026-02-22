"""Tests for AMP mode resolution in training and inference."""

from __future__ import annotations

import torch

from src.eval.ensemble import _resolve_inference_amp_mode
from src.train.loop import _resolve_amp_mode


def test_train_amp_disabled_in_config() -> None:
    state = _resolve_amp_mode(
        device=torch.device("cuda"),
        amp_enabled=False,
        amp_dtype="auto",
    )
    assert state.enabled is False
    assert state.dtype is None
    assert state.use_grad_scaler is False


def test_train_amp_disabled_on_non_cuda() -> None:
    state = _resolve_amp_mode(
        device=torch.device("cpu"),
        amp_enabled=True,
        amp_dtype="auto",
    )
    assert state.enabled is False
    assert state.dtype is None
    assert state.use_grad_scaler is False


def test_train_amp_auto_prefers_bf16(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    state = _resolve_amp_mode(
        device=torch.device("cuda"),
        amp_enabled=True,
        amp_dtype="auto",
    )
    assert state.enabled is True
    assert state.dtype == torch.bfloat16
    assert state.use_grad_scaler is False


def test_train_amp_auto_falls_back_to_fp16(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    state = _resolve_amp_mode(
        device=torch.device("cuda"),
        amp_enabled=True,
        amp_dtype="auto",
    )
    assert state.enabled is True
    assert state.dtype == torch.float16
    assert state.use_grad_scaler is True
    assert state.reason is not None and "falling back to fp16" in state.reason.lower()


def test_train_amp_bf16_falls_back_to_fp16(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    state = _resolve_amp_mode(
        device=torch.device("cuda"),
        amp_enabled=True,
        amp_dtype="bf16",
    )
    assert state.enabled is True
    assert state.dtype == torch.float16
    assert state.use_grad_scaler is True
    assert state.reason is not None and "requested bf16" in state.reason.lower()


def test_inference_amp_disabled_on_non_cuda() -> None:
    state = _resolve_inference_amp_mode(
        device=torch.device("cpu"),
        amp_enabled=True,
        amp_dtype="auto",
    )
    assert state.enabled is False
    assert state.dtype is None
