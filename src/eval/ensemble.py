from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from src.models.hmtl import HMTLModel
from src.utils.logger import get_logger


def ensemble_predict(
    models: List[HMTLModel],
    X: np.ndarray,
    device: torch.device | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict using ensemble with proper uncertainty aggregation.
    
    This function implements a unified uncertainty computation approach used by all models
    (HMTL, single_mlp, flat_mtl) for fair comparison. CatBoost uses the same formula
    in its predict() method.
    
    Uncertainty formula: sigma_total = sqrt(Var_epi(μ_i) + E[σ_i²])
    where:
    - Var_epi(μ_i) = variance of predictions across ensemble (epistemic uncertainty)
    - E[σ_i²] = mean of predicted variances (aleatoric uncertainty)
    
    Returns:
        mu_mean: Mean prediction across ensemble (μ̄)
        sigma_total: Total uncertainty = sqrt(Var_epi(μ_i) + E[σ_i²])
        sigma_epistemic: Epistemic uncertainty = std(μ_i)
        sigma_aleatoric: Aleatoric uncertainty = sqrt(mean(σ_i²))
    """
    logger = get_logger("eval.ensemble")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.debug(f"Ensemble prediction: {len(models)} models, {len(X)} samples")
    
    mus = []
    sigmas = []
    
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        
        for i, model in enumerate(models):
            model.eval()
            output = model(X_tensor)
            # Handle different model types
            if len(output) == 2:
                # Single MLP or models without aux task
                mu, sigma = output
            else:
                # HMTL models with aux task
                mu, sigma, _ = output
            mus.append(mu.cpu().numpy().ravel())
            sigmas.append(sigma.cpu().numpy().ravel())
    
    # Stack predictions: shape (n_models, n_samples)
    mus_array = np.stack(mus, axis=0)  # (n_models, n_samples)
    sigmas_array = np.stack(sigmas, axis=0)  # (n_models, n_samples)
    
    # Mean prediction
    mu_mean = np.mean(mus_array, axis=0)  # (n_samples,)
    
    # Epistemic uncertainty: variance of means across models
    mu_var_epistemic = np.var(mus_array, axis=0, ddof=0)  # (n_samples,)
    sigma_epistemic = np.sqrt(mu_var_epistemic)  # (n_samples,)
    
    # Aleatoric uncertainty: mean of predicted variances
    sigma_squared_mean = np.mean(sigmas_array ** 2, axis=0)  # (n_samples,)
    sigma_aleatoric = np.sqrt(sigma_squared_mean)  # (n_samples,)
    
    # Total uncertainty: Var_epi(μ_i) + E[σ_i²]
    sigma_total_squared = mu_var_epistemic + sigma_squared_mean
    sigma_total = np.sqrt(sigma_total_squared)  # (n_samples,)
    
    mean_epistemic = np.mean(sigma_epistemic)
    mean_aleatoric = np.mean(sigma_aleatoric)
    mean_total = np.mean(sigma_total)
    
    logger.info(
        f"Ensemble prediction uncertainty (standardized space) - "
        f"Epistemic: mean={mean_epistemic:.6f} (std={np.std(sigma_epistemic):.6f}), "
        f"Aleatoric: mean={mean_aleatoric:.6f} (std={np.std(sigma_aleatoric):.6f}), "
        f"Total: mean={mean_total:.6f} (std={np.std(sigma_total):.6f})"
    )
    
    logger.debug(
        f"Uncertainty stats - Epistemic: mean={mean_epistemic:.6f}, "
        f"Aleatoric: mean={mean_aleatoric:.6f}, "
        f"Total: mean={mean_total:.6f}"
    )
    
    return mu_mean, sigma_total, sigma_epistemic, sigma_aleatoric


def ensemble_predict_mean(
    models: List[HMTLModel],
    X: np.ndarray,
    device: torch.device | None = None,
) -> np.ndarray:
    """Simple mean prediction (backward compatibility)."""
    mu_mean, _, _, _ = ensemble_predict(models, X, device)
    return mu_mean

