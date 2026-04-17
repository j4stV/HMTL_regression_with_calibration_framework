from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
from torch import nn

import src.train.loop as train_loop
from src.train.loop import TrainConfig, train_model


class _TinyRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.raw_sigma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.linear(x)
        sigma = 1e-3 + torch.nn.functional.softplus(self.raw_sigma).expand_as(mu)
        return mu, sigma


def _make_regression_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.linspace(-1.0, 1.0, 12, dtype=np.float32).reshape(-1, 1)
    y_train = (2.0 * x_train[:, 0] + 0.5).astype(np.float32)
    x_valid = np.linspace(-0.5, 0.5, 6, dtype=np.float32).reshape(-1, 1)
    y_valid = (2.0 * x_valid[:, 0] + 0.5).astype(np.float32)
    return x_train, y_train, x_valid, y_valid


def test_train_model_cosine_scheduler_reduces_learning_rate() -> None:
    x_train, y_train, x_valid, y_valid = _make_regression_data()
    history: list[dict] = []

    train_model(
        model=_TinyRegressor(),
        X_tr=x_train,
        y_tr=y_train,
        X_va=x_valid,
        y_va=y_valid,
        n_bins=5,
        cfg=TrainConfig(
            lr=1e-2,
            epochs=3,
            batch_size=4,
            patience=10,
            seed=7,
            optimizer="adamw",
            show_progress=False,
            grad_clip_norm=None,
            lr_scheduler_name="cosine",
            lr_scheduler_eta_min_ratio=0.05,
            lr_warmup_epochs=0,
        ),
        history=history,
    )

    assert len(history) == 3
    assert history[0]["lr"] > history[-1]["lr"]


def test_train_model_clips_gradients_before_optimizer_step(monkeypatch) -> None:
    x_train, y_train, x_valid, y_valid = _make_regression_data()
    events: list[str] = []

    class _RecordingOptimizer:
        def __init__(self, params, lr: float, weight_decay: float) -> None:
            del weight_decay
            self.params = list(params)
            self.param_groups = [{"lr": lr, "params": self.params}]

        def zero_grad(self) -> None:
            for param in self.params:
                if param.grad is not None:
                    param.grad.zero_()

        def step(self) -> None:
            events.append("step")

    monkeypatch.setattr(
        train_loop.torch.optim,
        "AdamW",
        lambda params, lr, weight_decay: _RecordingOptimizer(params, lr, weight_decay),
    )

    def fake_clip_grad_norm_(parameters, max_norm):
        del max_norm
        list(parameters)
        events.append("clip")
        return torch.tensor(0.0)

    monkeypatch.setattr(train_loop.torch.nn.utils, "clip_grad_norm_", fake_clip_grad_norm_)

    train_model(
        model=_TinyRegressor(),
        X_tr=x_train,
        y_tr=y_train,
        X_va=x_valid,
        y_va=y_valid,
        n_bins=5,
        cfg=TrainConfig(
            lr=1e-2,
            epochs=1,
            batch_size=4,
            patience=5,
            seed=11,
            optimizer="adamw",
            show_progress=False,
            grad_clip_norm=0.5,
        ),
    )

    assert "clip" in events
    assert "step" in events
    assert events.index("clip") < events.index("step")


def test_train_model_unscales_before_clipping_with_grad_scaler(monkeypatch) -> None:
    x_train, y_train, x_valid, y_valid = _make_regression_data()
    events: list[str] = []

    class _RecordingOptimizer:
        def __init__(self, params, lr: float, weight_decay: float) -> None:
            del weight_decay
            self.params = list(params)
            self.param_groups = [{"lr": lr, "params": self.params}]

        def zero_grad(self) -> None:
            for param in self.params:
                if param.grad is not None:
                    param.grad.zero_()

        def step(self) -> None:
            events.append("optimizer_step")

    class _FakeScaler:
        def scale(self, loss: torch.Tensor) -> torch.Tensor:
            return loss

        def unscale_(self, optimizer) -> None:
            del optimizer
            events.append("unscale")

        def step(self, optimizer) -> None:
            events.append("scaler_step")
            optimizer.step()

        def update(self) -> None:
            events.append("update")

    monkeypatch.setattr(
        train_loop.torch.optim,
        "AdamW",
        lambda params, lr, weight_decay: _RecordingOptimizer(params, lr, weight_decay),
    )
    monkeypatch.setattr(
        train_loop,
        "_resolve_amp_mode",
        lambda device, amp_enabled, amp_dtype: train_loop._AmpState(
            enabled=False,
            dtype=None,
            use_grad_scaler=True,
        ),
    )
    monkeypatch.setattr(train_loop, "_autocast_if_needed", lambda enabled, dtype: nullcontext())
    monkeypatch.setattr(train_loop.torch.cuda.amp, "GradScaler", lambda enabled=True: _FakeScaler())

    def fake_clip_grad_norm_(parameters, max_norm):
        del max_norm
        list(parameters)
        events.append("clip")
        return torch.tensor(0.0)

    monkeypatch.setattr(train_loop.torch.nn.utils, "clip_grad_norm_", fake_clip_grad_norm_)

    train_model(
        model=_TinyRegressor(),
        X_tr=x_train,
        y_tr=y_train,
        X_va=x_valid,
        y_va=y_valid,
        n_bins=5,
        cfg=TrainConfig(
            lr=1e-2,
            epochs=1,
            batch_size=4,
            patience=5,
            seed=13,
            optimizer="adamw",
            show_progress=False,
            amp_enabled=True,
            grad_clip_norm=0.5,
        ),
    )

    assert "unscale" in events
    assert "clip" in events
    assert "scaler_step" in events
    assert events.index("unscale") < events.index("clip") < events.index("scaler_step")
