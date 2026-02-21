"""Tests for classification dataset preparation script."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import scripts.prepare_classification_dataset as prep_script



def test_prepare_classification_dataset_uses_task_loader(monkeypatch, tmp_path: Path):
    calls: list[int] = []

    def fake_loader(task_id: int, target_column=None):
        calls.append(task_id)
        df = pd.DataFrame(
            {
                "f1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
                "f2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "target": [10, 20, 10, 20, 10, 20, 10, 20, 10, 20, 10, 20],
            }
        )
        return df, "target"

    monkeypatch.setattr(prep_script, "project_root", tmp_path)
    monkeypatch.setattr(prep_script, "load_dataset_from_task", fake_loader)
    monkeypatch.setattr(
        prep_script,
        "DATASET_CATALOG",
        {
            "tiny": {
                "task_id": 999,
                "name": "Tiny",
                "n_classes": 2,
                "n_samples": 12,
                "n_features": 2,
                "description": "tiny test dataset",
            }
        },
    )

    result = prep_script.prepare_dataset(
        dataset_name="tiny",
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=7,
        force=True,
    )

    assert calls == [999]

    train_df = pd.read_csv(result["train_path"])
    valid_df = pd.read_csv(result["valid_path"])
    test_df = pd.read_csv(result["test_path"])

    # Labels must be remapped to 0..K-1.
    for df in (train_df, valid_df, test_df):
        assert set(df["target"].unique()).issubset({0, 1})

    config_text = Path(result["config_path"]).read_text(encoding="utf-8")
    assert "type: classification" in config_text
    assert "num_classes: 2" in config_text
