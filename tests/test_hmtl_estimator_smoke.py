"""End-to-end smoke test for the high-level HMTL API.

This test fits a tiny regression model on synthetic data, verifies predictions
have the right shape, round-trips through ``save``/``load``, and confirms the
reloaded model produces identical predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.hmtl import HMTLRegressor, load


@pytest.fixture(scope="module")
def synthetic_regression():
    rng = np.random.RandomState(0)
    n_train, n_test, d = 400, 100, 6
    X_train = rng.normal(size=(n_train, d))
    # Simple linear target with heteroskedastic noise
    w = rng.normal(size=d)
    y_train = X_train @ w + 0.3 * rng.normal(size=n_train)
    X_test = rng.normal(size=(n_test, d))
    y_test = X_test @ w + 0.3 * rng.normal(size=n_test)

    cols = [f"x{i}" for i in range(d)]
    X_train_df = pd.DataFrame(X_train, columns=cols)
    X_test_df = pd.DataFrame(X_test, columns=cols)
    return X_train_df, pd.Series(y_train, name="y"), X_test_df, y_test


def test_regressor_fit_predict_tiny(synthetic_regression, tmp_path):
    X_train_df, y_train, X_test_df, y_test = synthetic_regression

    reg = HMTLRegressor(
        preset="fast",
        n_models=2,
        epochs=5,
        patience=3,
        hidden_width=32,
        depth_low=2,
        depth_high=3,
        amp_enabled=False,
    )
    reg.fit(X_train_df, y_train, target_column="y")

    assert reg._fitted
    pred = reg.predict(X_test_df)
    assert pred.shape == (len(X_test_df),)
    assert np.all(np.isfinite(pred))

    mu, sigma = reg.predict(X_test_df, return_uncertainty=True)
    assert sigma.shape == (len(X_test_df),)
    assert np.all(sigma >= 0)


def test_regressor_save_load_roundtrip(synthetic_regression, tmp_path):
    X_train_df, y_train, X_test_df, _ = synthetic_regression

    reg = HMTLRegressor(
        preset="fast",
        n_models=2,
        epochs=5,
        patience=3,
        hidden_width=32,
        depth_low=2,
        depth_high=3,
        amp_enabled=False,
    )
    reg.fit(X_train_df, y_train, target_column="y")

    run_dir = tmp_path / "run"
    reg.save(run_dir)

    # Manifest files exist
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "preprocessor.pkl").exists()
    assert (run_dir / "conformal.json").exists()
    assert (run_dir / "config.yaml").exists()
    assert any((run_dir / "models").iterdir())

    # Load returns a fitted estimator producing the same predictions.
    loaded = load(run_dir)
    assert loaded._fitted
    pred_original = reg.predict(X_test_df)
    pred_loaded = loaded.predict(X_test_df)
    np.testing.assert_allclose(pred_original, pred_loaded, rtol=1e-4, atol=1e-4)


def test_regressor_predict_interval_shapes(synthetic_regression):
    X_train_df, y_train, X_test_df, _ = synthetic_regression

    reg = HMTLRegressor(
        preset="fast",
        n_models=2,
        epochs=5,
        patience=3,
        hidden_width=32,
        depth_low=2,
        depth_high=3,
        amp_enabled=False,
    )
    reg.fit(X_train_df, y_train, target_column="y")

    lower, upper = reg.predict_interval(X_test_df, coverage=0.9)
    assert lower.shape == upper.shape == (len(X_test_df),)
    assert np.all(upper >= lower)


def test_load_errors_if_column_mismatch(synthetic_regression, tmp_path):
    X_train_df, y_train, X_test_df, _ = synthetic_regression

    reg = HMTLRegressor(
        preset="fast",
        n_models=2,
        epochs=5,
        patience=3,
        hidden_width=32,
        depth_low=2,
        depth_high=3,
        amp_enabled=False,
    )
    reg.fit(X_train_df, y_train, target_column="y")

    run_dir = tmp_path / "run"
    reg.save(run_dir)
    loaded = load(run_dir)

    # Drop a feature — predict() should complain.
    bad_df = X_test_df.drop(columns=[X_test_df.columns[0]])
    with pytest.raises(ValueError, match="missing expected columns"):
        loaded.predict(bad_df)
