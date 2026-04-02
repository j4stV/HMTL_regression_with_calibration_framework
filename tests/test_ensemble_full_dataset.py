from __future__ import annotations

import numpy as np
import pytest

from src.train import loop as loop_mod
from src.train.ensemble import EnsembleConfig, fit_ensemble
from src.train.loop import TrainConfig


def test_fit_ensemble_full_dataset_uses_full_train_and_shared_validation(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[tuple[int, ...], tuple[int, ...], int | None]] = []

    class _FakeModel:
        aux_task = "bins"

    def fake_train_model(model, X_tr, y_tr, X_va, y_va, n_bins, cfg, **kwargs):
        calls.append((X_tr.shape, X_va.shape, cfg.seed))
        return 0.1

    monkeypatch.setattr(loop_mod, "train_model", fake_train_model)

    X_tr = np.zeros((6, 3), dtype=np.float32)
    y_tr = np.linspace(0.0, 1.0, 6, dtype=np.float32)
    X_va = np.zeros((2, 3), dtype=np.float32)
    y_va = np.zeros(2, dtype=np.float32)

    models, score = fit_ensemble(
        build_model_fn=lambda: _FakeModel(),
        X_tr=X_tr,
        y_tr=y_tr,
        X_va=X_va,
        y_va=y_va,
        n_bins=3,
        ens_cfg=EnsembleConfig(n_models=3, bagging="full_dataset", show_progress=False),
        train_cfg=TrainConfig(epochs=1, batch_size=2, seed=9, show_progress=False),
    )

    assert len(models) == 3
    assert np.isclose(score, 0.1)
    assert calls == [
        ((6, 3), (2, 3), 9),
        ((6, 3), (2, 3), 10),
        ((6, 3), (2, 3), 11),
    ]
