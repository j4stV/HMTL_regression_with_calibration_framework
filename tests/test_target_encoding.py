from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.preprocess import PreprocessConfig, TabularPreprocessor
from src.data.target_encoding import RegressionOOFTargetEncoder


def test_regression_oof_target_encoder_replaces_categoricals_and_handles_unseen_values():
    train_df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0],
            "city": ["A", "A", "B", None],
            "merchant": ["x", "y", "x", "z"],
        }
    )
    y_train = np.asarray([10.0, 14.0, 30.0, 40.0], dtype=float)

    encoder = RegressionOOFTargetEncoder(n_splits=2, smoothing=5.0, random_state=11)
    encoded_train = encoder.fit_transform(train_df, y_train, ["city", "merchant"])

    assert list(encoded_train.columns) == ["num", "city__te", "merchant__te"]
    assert all(pd.api.types.is_numeric_dtype(encoded_train[column]) for column in encoded_train.columns)
    assert np.isclose(encoder.global_mean_, 23.5)

    valid_df = pd.DataFrame(
        {
            "num": [5.0, 6.0],
            "city": ["A", "unseen"],
            "merchant": [None, "z"],
        }
    )
    encoded_valid = encoder.transform(valid_df)

    assert list(encoded_valid.columns) == ["num", "city__te", "merchant__te"]
    assert np.isclose(encoded_valid.loc[1, "city__te"], encoder.global_mean_)
    assert np.isfinite(encoded_valid.loc[0, "merchant__te"])


def test_regression_oof_target_encoder_singleton_category_does_not_self_leak():
    train_df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0],
            "singleton": ["only", "shared", "shared", "shared"],
        }
    )
    y_train = np.asarray([100.0, 10.0, 20.0, 30.0], dtype=float)

    encoder = RegressionOOFTargetEncoder(n_splits=2, smoothing=10.0, random_state=3)
    encoded_train = encoder.fit_transform(train_df, y_train, ["singleton"])

    assert not np.isclose(encoded_train.loc[0, "singleton__te"], 100.0)
    assert np.isclose(encoded_train.loc[0, "singleton__te"], encoder.global_mean_)


def test_tabular_preprocessor_applies_target_encoding_and_handles_unseen_categories():
    train_df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0],
            "city": ["A", "A", "B", None],
            "merchant": ["x", "y", "x", "z"],
            "target": [10.0, 14.0, 30.0, 40.0],
        }
    )
    valid_df = pd.DataFrame(
        {
            "num": [5.0, 6.0],
            "city": ["A", "unseen"],
            "merchant": [None, "z"],
            "target": [0.0, 0.0],
        }
    )

    preprocessor = TabularPreprocessor(
        PreprocessConfig(
            use_dynamic_binning=False,
            quantile_binning_enabled=False,
            standardize=False,
            pca_enabled=False,
            target_standardize=False,
            target_encoding_enabled=True,
            target_encoding_n_splits=2,
            target_encoding_smoothing=5.0,
        ),
        target_column="target",
        categorical_columns=["city", "merchant"],
        task_type="regression",
    ).fit(train_df)

    transformed_train, y_train = preprocessor.transform(train_df)
    transformed_valid, y_valid = preprocessor.transform(valid_df)

    assert transformed_train.shape == (4, 3)
    assert transformed_valid.shape == (2, 3)
    assert np.all(np.isfinite(transformed_train))
    assert np.all(np.isfinite(transformed_valid))
    assert np.isclose(y_train[0], 10.0)
    assert np.isclose(y_valid[0], 0.0)

    encoder = preprocessor.target_encoder_
    assert encoder is not None
    assert list(encoder.transform(valid_df.drop(columns=["target"])).columns) == [
        "num",
        "city__te",
        "merchant__te",
    ]
    assert np.isclose(
        encoder.transform(valid_df.drop(columns=["target"])).loc[1, "city__te"],
        encoder.global_mean_,
    )


def test_tabular_preprocessor_target_encoding_works_with_dynamic_binning_enabled():
    train_df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0, 5.0],
            "algorithm": ["a", "b", "a", "c", "b"],
            "target": [10.0, 11.0, 12.0, 13.0, 14.0],
        }
    )
    valid_df = pd.DataFrame(
        {
            "num": [6.0, 7.0],
            "algorithm": ["a", "unseen"],
            "target": [15.0, 16.0],
        }
    )

    preprocessor = TabularPreprocessor(
        PreprocessConfig(
            use_dynamic_binning=True,
            standardize=False,
            pca_enabled=False,
            target_standardize=False,
            target_encoding_enabled=True,
            target_encoding_n_splits=2,
            target_encoding_smoothing=5.0,
        ),
        target_column="target",
        categorical_columns=["algorithm"],
        task_type="regression",
    ).fit(train_df)

    transformed_valid, y_valid = preprocessor.transform(valid_df)

    assert transformed_valid.shape[0] == 2
    assert np.all(np.isfinite(transformed_valid))
    assert np.allclose(y_valid, np.array([15.0, 16.0]))
