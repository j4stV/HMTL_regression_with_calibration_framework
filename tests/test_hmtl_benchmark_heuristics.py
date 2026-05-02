"""Tests for :mod:`src.hmtl.benchmark_heuristics`."""

from __future__ import annotations

from src.hmtl.benchmark_heuristics import benchmark_size_overrides


def test_tiny_regime_uses_bins_small_ensemble_no_adversarial():
    ov = benchmark_size_overrides(n_rows=200, n_features=40)
    assert ov["aux_task"] == "bins"
    assert ov["bagging"] == "stratified_bins"
    assert ov["n_models"] <= 8
    assert ov["hidden_width"] <= 64
    assert ov["conformal_method"] == "symmetric"
    assert ov["extra"]["adversarial"]["enabled"] is False


def test_small_regime_uses_bins_stratified_bagging():
    ov = benchmark_size_overrides(n_rows=1500, n_features=40)
    assert ov["aux_task"] == "bins"
    assert ov["bagging"] == "stratified_bins"
    assert ov["n_models"] <= 10
    assert ov["hidden_width"] <= 96
    assert ov["conformal_method"] == "cqr"


def test_large_low_dim_uses_bins_depth_cap():
    ov = benchmark_size_overrides(n_rows=50_000, n_features=8)
    # large regime activates auto aux selection
    assert ov["aux_task"] == "auto"
    assert ov["extra"]["auto_candidates"] == ["bins", "contrastive"]
    assert ov["depth_high"] == 8
    assert ov["depth_low"] == 3
    assert ov["hidden_width"] <= 64
    assert ov["pca_enabled"] is False


def test_large_high_dim_enables_pca():
    ov = benchmark_size_overrides(n_rows=20_000, n_features=1200)
    assert ov["pca_enabled"] is True
    assert ov["pca_n_components"] is not None
    assert ov["extra"]["weight_decay"] >= 1e-3


def test_batch_size_respects_regime_divisor():
    # large regime divides by 12
    ov_large = benchmark_size_overrides(n_rows=12_000, n_features=40)
    assert ov_large["batch_size"] == min(4096, max(16, 12_000 // 12))

    # tiny regime divides by 4
    ov_tiny = benchmark_size_overrides(n_rows=200, n_features=40)
    assert ov_tiny["batch_size"] == max(16, 200 // 4)


def test_lr_scales_with_sqrt_of_batch_but_clamped():
    # Very small batch → clamped to 0.7x the base LR
    ov = benchmark_size_overrides(n_rows=100, n_features=5, base_lr=3e-4)
    assert ov["lr"] >= 3e-4 * 0.7 - 1e-12
    assert ov["lr"] <= 3e-4 * 1.5 + 1e-12
