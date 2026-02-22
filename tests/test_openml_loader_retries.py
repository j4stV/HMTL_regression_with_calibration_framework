from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.data.openml_loader as openml_loader


class _FakeDataset:
    name = "Buzzinsocialmedia_Twitter"
    default_target_attribute = "target"
    features = ["x"]

    def get_data(self, target=None, return_attribute_names=False, **kwargs):
        X = pd.DataFrame({"x": [1.0, 2.0]})
        y = pd.Series([0.1, 0.2])
        if return_attribute_names:
            return X, y, ["x"]
        return X, y


class _FakeMixedTypeDataset:
    name = "house_prices_nominal_like"
    default_target_attribute = "target"
    features = ["text_num", "road_type", "nullable_cat", "plain_int"]

    def get_data(self, target=None, return_attribute_names=False, **kwargs):
        X = pd.DataFrame(
            {
                "text_num": pd.Series([None, "1.5", "2", "3.25"], dtype="object"),
                "road_type": pd.Series(pd.Categorical([None, "Grvl", "Pave", "Grvl"])),
                "nullable_cat": pd.Series(
                    [None, "Seventh Day Adventists", "Catholic", "Catholic"],
                    dtype="object",
                ),
                "plain_int": pd.Series([1, 2, 3, 4], dtype="uint8"),
            }
        )
        y = pd.Series([10.0, 20.0, 30.0, 40.0])
        if return_attribute_names:
            return X, y, self.features
        return X, y


def test_load_dataset_retries_transient_openml_download_error(monkeypatch):
    calls = {"count": 0}
    sleep_calls: list[float] = []

    def fake_get_dataset(dataset_id: int, download_data: bool = True):
        calls["count"] += 1
        if calls["count"] == 1:
            raise Exception(
                "Connection broken: IncompleteRead(45305856 bytes read, 14771245 more expected)"
            )
        return _FakeDataset()

    monkeypatch.setenv("OPENML_DOWNLOAD_RETRIES", "2")
    monkeypatch.setenv("OPENML_DOWNLOAD_RETRY_BACKOFF_SEC", "0")
    monkeypatch.setattr(openml_loader.time, "sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setattr(
        openml_loader,
        "openml",
        SimpleNamespace(datasets=SimpleNamespace(get_dataset=fake_get_dataset)),
    )

    df, target_col = openml_loader.load_dataset(4549)

    assert calls["count"] == 2
    assert sleep_calls == [0.0]
    assert target_col == "target"
    assert "target" in df.columns
    assert len(df) == 2


def test_load_dataset_does_not_retry_non_transient_openml_download_error(monkeypatch):
    calls = {"count": 0}
    sleep_calls: list[float] = []

    def fake_get_dataset(dataset_id: int, download_data: bool = True):
        calls["count"] += 1
        raise ValueError("Dataset id not found")

    monkeypatch.setenv("OPENML_DOWNLOAD_RETRIES", "5")
    monkeypatch.setenv("OPENML_DOWNLOAD_RETRY_BACKOFF_SEC", "0")
    monkeypatch.setattr(openml_loader.time, "sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setattr(
        openml_loader,
        "openml",
        SimpleNamespace(datasets=SimpleNamespace(get_dataset=fake_get_dataset)),
    )

    with pytest.raises(ValueError, match="Dataset id not found"):
        openml_loader.load_dataset(999999)

    assert calls["count"] == 1
    assert sleep_calls == []


def test_load_dataset_converts_mixed_feature_types_to_numeric(monkeypatch):
    monkeypatch.setattr(
        openml_loader,
        "openml",
        SimpleNamespace(
            datasets=SimpleNamespace(
                get_dataset=lambda dataset_id, download_data=True: _FakeMixedTypeDataset()
            )
        ),
    )

    df, target_col = openml_loader.load_dataset(12345)

    assert target_col == "target"
    feature_df = df.drop(columns=[target_col])
    assert list(feature_df.columns) == _FakeMixedTypeDataset.features
    assert all(pd.api.types.is_numeric_dtype(feature_df[col]) for col in feature_df.columns)
    assert all(pd.api.types.is_float_dtype(feature_df[col]) for col in feature_df.columns)

    assert np.isclose(feature_df.loc[1, "text_num"], 1.5)
    assert np.isnan(feature_df.loc[0, "road_type"])
    assert np.isfinite(feature_df.loc[1, "road_type"])
    assert np.isfinite(feature_df.loc[1, "nullable_cat"])
