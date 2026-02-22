from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from src.models.hmtl import HMTLModel
from src.utils.logger import get_logger


def _is_mps_available() -> bool:
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend is not None and mps_backend.is_available())


def _select_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class _InferenceAmpState:
    enabled: bool
    dtype: torch.dtype | None
    reason: str | None = None


def _normalize_amp_dtype(amp_dtype: str) -> str:
    normalized = str(amp_dtype).strip().lower()
    if normalized not in {"auto", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported AMP dtype '{amp_dtype}'. Expected one of: auto, fp16, bf16."
        )
    return normalized


def _cuda_supports_native_bf16() -> bool:
    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    if not callable(is_bf16_supported):
        return False

    try:
        bf16_supported = bool(is_bf16_supported(including_emulation=False))
    except TypeError:
        bf16_supported = bool(is_bf16_supported())
    except Exception:
        return False

    if not bf16_supported:
        return False

    get_capability = getattr(torch.cuda, "get_device_capability", None)
    if callable(get_capability):
        try:
            major, _minor = get_capability(torch.cuda.current_device())
            if int(major) < 8:
                return False
        except TypeError:
            try:
                major, _minor = get_capability()
                if int(major) < 8:
                    return False
            except Exception:
                pass
        except Exception:
            pass

    return True


def _resolve_inference_amp_mode(
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: str,
) -> _InferenceAmpState:
    if not amp_enabled:
        return _InferenceAmpState(enabled=False, dtype=None, reason="AMP disabled by caller")

    if device.type != "cuda":
        return _InferenceAmpState(
            enabled=False,
            dtype=None,
            reason=f"AMP supports only CUDA; current device={device.type}",
        )

    requested_dtype = _normalize_amp_dtype(amp_dtype)
    bf16_supported = _cuda_supports_native_bf16()

    if requested_dtype == "auto":
        if bf16_supported:
            return _InferenceAmpState(enabled=True, dtype=torch.bfloat16)
        return _InferenceAmpState(
            enabled=True,
            dtype=torch.float16,
            reason="BF16 native support not available on this CUDA device; falling back to FP16",
        )

    if requested_dtype == "bf16":
        if bf16_supported:
            return _InferenceAmpState(enabled=True, dtype=torch.bfloat16)
        return _InferenceAmpState(
            enabled=True,
            dtype=torch.float16,
            reason="Requested BF16, but native BF16 support is unavailable; falling back to FP16",
        )

    return _InferenceAmpState(enabled=True, dtype=torch.float16)


def _autocast_if_needed(enabled: bool, dtype: torch.dtype | None):
    if enabled and dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def ensemble_predict(
    models: List[HMTLModel],
    X: np.ndarray,
    device: torch.device | None = None,
    amp_enabled: bool = False,
    amp_dtype: str = "auto",
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
        device = _select_default_device()

    amp_state = _resolve_inference_amp_mode(
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )
    logger.info(
        f"Inference AMP requested: enabled={amp_enabled}, dtype={amp_dtype}. "
        f"effective_enabled={amp_state.enabled}, effective_dtype={amp_state.dtype}"
    )
    if amp_state.reason is not None:
        if "falling back" in amp_state.reason.lower():
            logger.warning(f"Inference AMP note: {amp_state.reason}")
        else:
            logger.info(f"Inference AMP note: {amp_state.reason}")
    
    logger.debug(f"Ensemble prediction: {len(models)} models, {len(X)} samples")
    
    mus = []
    sigmas = []
    
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        
        for i, model in enumerate(models):
            model.eval()
            with _autocast_if_needed(enabled=amp_state.enabled, dtype=amp_state.dtype):
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
    amp_enabled: bool = False,
    amp_dtype: str = "auto",
) -> np.ndarray:
    """Simple mean prediction (backward compatibility)."""
    mu_mean, _, _, _ = ensemble_predict(
        models,
        X,
        device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )
    return mu_mean


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax.

    Args:
        logits: Logits array (..., n_classes)
        axis: Axis along which to compute softmax

    Returns:
        Probabilities with same shape as logits
    """
    exp_logits = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)


def entropy(probs: np.ndarray, axis: int = -1, eps: float = 1e-10) -> np.ndarray:
    """Compute entropy of probability distribution.

    H(p) = -sum(p * log(p))

    Args:
        probs: Probability array (..., n_classes)
        axis: Axis along which to compute entropy
        eps: Small constant to avoid log(0)

    Returns:
        Entropy values with shape obtained by removing axis
    """
    # Clip to avoid log(0)
    probs_clipped = np.clip(probs, eps, 1.0)
    return -np.sum(probs_clipped * np.log(probs_clipped), axis=axis)


def ensemble_predict_classification(
    models: List[HMTLModel],
    X: np.ndarray,
    device: torch.device | None = None,
    amp_enabled: bool = False,
    amp_dtype: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Predict using ensemble for classification with uncertainty decomposition.

    Uncertainty is decomposed using mutual information (MI):
    - Total uncertainty: H[E[p]] = entropy of mean probabilities (ensemble disagreement)
    - Aleatoric uncertainty: E[H[p]] = mean entropy across models (average individual model uncertainty)
    - Epistemic uncertainty: MI = H[E[p]] - E[H[p]] (uncertainty reducible by more data)

    This decomposition is principled for classification and mirrors the
    epistemic/aleatoric split used in regression.

    Args:
        models: List of trained models
        X: Input features (n_samples, n_features)
        device: Device to run predictions on

    Returns:
        logits_mean: Mean logits across ensemble (n_samples, n_classes)
        probs_mean: Mean probabilities across ensemble (n_samples, n_classes)
        uncertainty_total: Total uncertainty = H[E[p]] (n_samples,)
        uncertainty_epistemic: Epistemic = MI = H[E[p]] - E[H[p]] (n_samples,)
        uncertainty_aleatoric: Aleatoric = E[H[p]] (n_samples,)
    """
    logger = get_logger("eval.ensemble")

    if device is None:
        device = _select_default_device()

    amp_state = _resolve_inference_amp_mode(
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )
    logger.info(
        f"Inference AMP requested: enabled={amp_enabled}, dtype={amp_dtype}. "
        f"effective_enabled={amp_state.enabled}, effective_dtype={amp_state.dtype}"
    )
    if amp_state.reason is not None:
        if "falling back" in amp_state.reason.lower():
            logger.warning(f"Inference AMP note: {amp_state.reason}")
        else:
            logger.info(f"Inference AMP note: {amp_state.reason}")

    logger.debug(f"Ensemble classification prediction: {len(models)} models, {len(X)} samples")

    logits_list = []
    probs_list = []

    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

        for model in models:
            model.eval()
            with _autocast_if_needed(enabled=amp_state.enabled, dtype=amp_state.dtype):
                output = model(X_tensor)

            # Handle different model output types
            if len(output) == 2:
                # Model without aux task: (logits, None)
                logits = output[0]
            elif len(output) == 3:
                # Model with aux task: (logits, None, aux_output)
                logits = output[0]
            else:
                logits = output

            # Ensure logits is a tensor
            if not isinstance(logits, torch.Tensor):
                raise ValueError(f"Expected logits to be torch.Tensor, got {type(logits)}")

            logits_np = logits.cpu().numpy()
            probs_np = softmax(logits_np, axis=-1)

            logits_list.append(logits_np)
            probs_list.append(probs_np)

    # Stack predictions: (n_models, n_samples, n_classes)
    logits_array = np.stack(logits_list, axis=0)
    probs_array = np.stack(probs_list, axis=0)

    # Mean predictions
    logits_mean = np.mean(logits_array, axis=0)  # (n_samples, n_classes)
    probs_mean = np.mean(probs_array, axis=0)  # (n_samples, n_classes)

    # Uncertainty computation via mutual information decomposition
    # 1. Total uncertainty: entropy of mean probabilities
    uncertainty_total = entropy(probs_mean, axis=-1)  # (n_samples,)

    # 2. Aleatoric uncertainty: mean entropy of individual predictions
    entropies = entropy(probs_array, axis=-1)  # (n_models, n_samples)
    uncertainty_aleatoric = np.mean(entropies, axis=0)  # (n_samples,)

    # 3. Epistemic uncertainty: mutual information = total - aleatoric
    uncertainty_epistemic = uncertainty_total - uncertainty_aleatoric  # (n_samples,)

    # Log statistics
    mean_total = np.mean(uncertainty_total)
    mean_epistemic = np.mean(uncertainty_epistemic)
    mean_aleatoric = np.mean(uncertainty_aleatoric)

    logger.info(
        f"Classification uncertainty - "
        f"Total: mean={mean_total:.6f} (std={np.std(uncertainty_total):.6f}), "
        f"Epistemic: mean={mean_epistemic:.6f} (std={np.std(uncertainty_epistemic):.6f}), "
        f"Aleatoric: mean={mean_aleatoric:.6f} (std={np.std(uncertainty_aleatoric):.6f})"
    )

    return logits_mean, probs_mean, uncertainty_total, uncertainty_epistemic, uncertainty_aleatoric

