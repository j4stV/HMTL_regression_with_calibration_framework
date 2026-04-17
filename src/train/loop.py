from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.losses.nll import gaussian_nll
from src.eval.r_auc_mse import r_auc_mse
from src.utils.logger import get_logger
from src.train.optimizers import create_radam_lookahead
from src.train.adversarial import AdversarialConfig, generate_adversarial
from src.models.contrastive import n_pairs_loss


@dataclass
class TrainConfig:
    lr: float = 3e-4
    epochs: int = 200
    batch_size: int = 256
    patience: int = 20
    aux_weight: float = 0.5  # Default 0.5  (loss_weights for projection)
    sigma_reg_weight: float = 0.0  # Disabled by default (regression only)
    optimizer: str = "radam_lookahead"  # "radam_lookahead" or "adamw"
    lookahead_k: int = 6
    lookahead_alpha: float = 0.5
    weight_decay: float = 0.0
    seed: int | None = None
    task_type: str = "regression"  # NEW: "regression" or "classification"
    show_progress: bool = True
    amp_enabled: bool = True
    amp_dtype: str = "auto"  # "auto", "fp16", or "bf16"
    amp_eval_enabled: bool = True
    early_stop_metric: str = "hybrid_rmse_rauc"  # "hybrid_rmse_rauc", "rmse", or "r_auc_mse"
    hybrid_r_auc_weight: float = 0.25
    grad_clip_norm: float | None = 1.0
    lr_scheduler_name: str = "none"  # "none" or "cosine"
    lr_scheduler_eta_min_ratio: float = 0.05
    lr_warmup_epochs: int = 2
    # CQR (Conformal Quantile Regression)
    cqr_enabled: bool = False
    cqr_quantiles: list[float] | None = None
    cqr_weight: float = 0.5
    # Adversarial augmentations
    adversarial: AdversarialConfig | None = None


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
class _AmpState:
    enabled: bool
    dtype: torch.dtype | None
    use_grad_scaler: bool
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


def _resolve_amp_mode(device: torch.device, amp_enabled: bool, amp_dtype: str) -> _AmpState:
    if not amp_enabled:
        return _AmpState(
            enabled=False,
            dtype=None,
            use_grad_scaler=False,
            reason="AMP disabled in config",
        )

    if device.type != "cuda":
        return _AmpState(
            enabled=False,
            dtype=None,
            use_grad_scaler=False,
            reason=f"AMP supports only CUDA; current device={device.type}",
        )

    requested_dtype = _normalize_amp_dtype(amp_dtype)
    bf16_supported = _cuda_supports_native_bf16()

    if requested_dtype == "auto":
        if bf16_supported:
            return _AmpState(enabled=True, dtype=torch.bfloat16, use_grad_scaler=False)
        return _AmpState(
            enabled=True,
            dtype=torch.float16,
            use_grad_scaler=True,
            reason="BF16 native support not available on this CUDA device; falling back to FP16",
        )

    if requested_dtype == "bf16":
        if bf16_supported:
            return _AmpState(enabled=True, dtype=torch.bfloat16, use_grad_scaler=False)
        return _AmpState(
            enabled=True,
            dtype=torch.float16,
            use_grad_scaler=True,
            reason="Requested BF16, but native BF16 support is unavailable; falling back to FP16",
        )

    return _AmpState(enabled=True, dtype=torch.float16, use_grad_scaler=True)


def _autocast_if_needed(enabled: bool, dtype: torch.dtype | None):
    if enabled and dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _to_numpy_compatible(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to a NumPy-compatible CPU array.

    NumPy does not support torch.bfloat16 tensors directly.
    """
    cpu_tensor = tensor.detach().cpu()
    if cpu_tensor.dtype == torch.bfloat16:
        cpu_tensor = cpu_tensor.to(dtype=torch.float32)
    return cpu_tensor.numpy()


def _normalize_regression_early_stop_metric(metric_name: str) -> str:
    normalized = str(metric_name).strip().lower()
    aliases = {
        "hybrid": "hybrid_rmse_rauc",
        "hybrid_rmse_rauc": "hybrid_rmse_rauc",
        "hybrid_rmse_r_auc": "hybrid_rmse_rauc",
        "rmse_plus_r_auc": "hybrid_rmse_rauc",
        "rmse_plus_rauc": "hybrid_rmse_rauc",
        "rmse": "rmse",
        "r_auc_mse": "r_auc_mse",
    }
    if normalized not in aliases:
        raise ValueError(
            "Unsupported regression early-stop metric "
            f"'{metric_name}'. Expected one of: hybrid_rmse_rauc, rmse, r_auc_mse."
        )
    return aliases[normalized]


def _normalize_lr_scheduler_name(scheduler_name: str) -> str:
    normalized = str(scheduler_name).strip().lower()
    aliases = {
        "none": "none",
        "off": "none",
        "disabled": "none",
        "cosine": "cosine",
        "cosineannealing": "cosine",
        "cosine_annealing": "cosine",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported LR scheduler '{scheduler_name}'. Expected one of: none, cosine."
        )
    return aliases[normalized]


def _resolve_regression_validation_score(
    *,
    metric_name: str,
    rmse: float,
    r_auc_mse_score: float,
    hybrid_r_auc_weight: float,
) -> float:
    normalized = _normalize_regression_early_stop_metric(metric_name)
    if normalized == "rmse":
        return float(rmse)
    if normalized == "r_auc_mse":
        return float(r_auc_mse_score)
    return float(rmse + hybrid_r_auc_weight * r_auc_mse_score)


def compute_bins(y: np.ndarray, n_bins: int, use_rounding: bool = False) -> np.ndarray:
    """Compute bins for auxiliary task.
    
    Args:
        y: Target values
        n_bins: Number of bins (used only if use_rounding=False)
        use_rounding: If True, use rounding approach (round to integers, 
                     clip to 0.1%-99.9% percentile range)
    
    Returns:
        Array of bin indices
    """
    if use_rounding:
        # Approach: round to integers, create classes from rounded values
        all_values = sorted(y.tolist())
        n = len(all_values)
        if n <= 10000:
            # Fallback to quantile approach for small datasets
            quantiles = np.quantile(y, np.linspace(0, 1, n_bins + 1))
            quantiles[0] -= 1e-9
            quantiles[-1] += 1e-9
            return np.digitize(y, quantiles[1:-1])
        
        # Get 0.1% and 99.9% percentiles
        y001 = all_values[int(round(0.001 * (n - 1)))]
        y999 = all_values[int(round(0.999 * (n - 1)))]
        
        # Create integer classes
        min_temp = int(np.floor(y001))
        max_temp = int(np.ceil(y999))
        n_classes = max_temp - min_temp + 1
        
        # Map rounded values to class indices
        y_rounded = np.round(y).astype(np.int32)
        y_bins = np.clip(y_rounded, min_temp, max_temp) - min_temp
        
        return y_bins.astype(np.int64)
    else:
        # Equal quantiles approach (original)
        quantiles = np.quantile(y, np.linspace(0, 1, n_bins + 1))
        quantiles[0] -= 1e-9
        quantiles[-1] += 1e-9
        return np.digitize(y, quantiles[1:-1])


def _compute_aux_loss(aux_task, aux_output, yb_bin, yb, ce, model):
    """Compute auxiliary task loss based on aux_task type."""
    if aux_task == "bins":
        return ce(aux_output, yb_bin)
    elif aux_task == "contrastive":
        return n_pairs_loss(aux_output, yb_bin, temperature=0.5)
    elif aux_task == "reconstruction":
        from src.losses.aux_losses import reconstruction_loss
        h_low = getattr(model, '_cached_h_low', None)
        if h_low is not None:
            return reconstruction_loss(h_low.detach(), aux_output)
        return torch.tensor(0.0, device=aux_output.device)
    elif aux_task == "rank":
        from src.losses.aux_losses import pairwise_ranking_loss
        return pairwise_ranking_loss(aux_output, yb)
    elif aux_task == "multi":
        # aux_output is a dict {task_name: output}
        total_aux = torch.tensor(0.0, device=yb.device)
        multi_weights = getattr(model, 'multi_aux_weights', {})
        h_low = getattr(model, '_cached_h_low', None)
        for name, out in aux_output.items():
            w = multi_weights.get(name, 1.0)
            if name == "bins":
                total_aux = total_aux + w * ce(out, yb_bin)
            elif name == "contrastive":
                total_aux = total_aux + w * n_pairs_loss(out, yb_bin, temperature=0.5)
            elif name == "reconstruction":
                from src.losses.aux_losses import reconstruction_loss
                if h_low is not None:
                    total_aux = total_aux + w * reconstruction_loss(h_low.detach(), out)
            elif name == "rank":
                from src.losses.aux_losses import pairwise_ranking_loss
                total_aux = total_aux + w * pairwise_ranking_loss(out, yb)
        return total_aux
    else:
        return torch.tensor(0.0, device=aux_output.device if hasattr(aux_output, 'device') else yb.device)


def _resolve_classification_val_score(
    metric_name: str,
    nll: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Return a classification validation score (lower is better)."""
    from sklearn.metrics import balanced_accuracy_score, f1_score

    metric = str(metric_name).strip().lower()
    if metric == "balanced_accuracy":
        return -float(balanced_accuracy_score(y_true, y_pred))
    if metric == "accuracy":
        return -float(np.mean(y_pred == y_true))
    if metric == "f1_weighted":
        return -float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    return nll


def train_model(
    model: nn.Module,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    n_bins: int,
    cfg: TrainConfig,
    task_loss = None,  # NEW: Optional task-specific loss function
    task_metrics = None,  # NEW: Optional task-specific metrics
    history: list[dict] | None = None,
    history_meta: dict | None = None,
) -> float:
    logger = get_logger("train")
    
    # Set random seed if provided
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        logger.info(f"Set random seed to {cfg.seed}")
    
    # Select execution device with fallback: CUDA -> MPS -> CPU.
    device = _select_default_device()
    if device.type == "cuda":
        logger.info(f"Using device: {device}")
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  GPU count: {torch.cuda.device_count()}")
        logger.info(f"  GPU name: {torch.cuda.get_device_name(0)}")
    elif device.type == "mps":
        logger.info(f"Using device: {device}")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and hasattr(mps_backend, "is_built"):
            logger.info(f"  MPS backend built: {mps_backend.is_built()}")
        logger.info(f"  PyTorch version: {torch.__version__}")
    else:
        logger.warning(f"Using device: {device}")
        logger.warning("  CUDA and MPS are unavailable. Possible reasons:")
        logger.warning("    1. No compatible GPU detected on this system")
        logger.warning("    2. PyTorch was installed without GPU backend support")
        logger.warning("    3. GPU drivers/runtime are not properly configured")
        logger.warning(f"  PyTorch version: {torch.__version__}")
        logger.warning(f"  CUDA available in PyTorch build: {torch.version.cuda is not None}")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and hasattr(mps_backend, "is_built"):
            logger.warning(f"  MPS available in PyTorch build: {mps_backend.is_built()}")
    model.to(device)

    amp_state = _resolve_amp_mode(
        device=device,
        amp_enabled=cfg.amp_enabled,
        amp_dtype=cfg.amp_dtype,
    )
    logger.info(
        f"AMP requested: enabled={cfg.amp_enabled}, dtype={cfg.amp_dtype}, "
        f"eval_enabled={cfg.amp_eval_enabled}"
    )
    logger.info(
        f"AMP effective: enabled={amp_state.enabled}, dtype={amp_state.dtype}, "
        f"grad_scaler={amp_state.use_grad_scaler}"
    )
    if amp_state.reason is not None:
        if "falling back" in amp_state.reason.lower():
            logger.warning(f"AMP note: {amp_state.reason}")
        else:
            logger.info(f"AMP note: {amp_state.reason}")
    use_eval_amp = amp_state.enabled and cfg.amp_eval_enabled
    logger.info(f"AMP eval effective: enabled={use_eval_amp}, dtype={amp_state.dtype}")
    if cfg.task_type == "regression":
        normalized_metric = _normalize_regression_early_stop_metric(cfg.early_stop_metric)
        logger.info(
            "Regression early-stop metric: %s (hybrid_r_auc_weight=%.4f)",
            normalized_metric,
            cfg.hybrid_r_auc_weight,
        )

    scaler = (
        torch.cuda.amp.GradScaler(enabled=True)
        if amp_state.use_grad_scaler
        else None
    )

    logger.debug(f"Training data shape: {X_tr.shape}, Validation data shape: {X_va.shape}")
    logger.debug(f"Training config: lr={cfg.lr}, epochs={cfg.epochs}, batch_size={cfg.batch_size}, patience={cfg.patience}, optimizer={cfg.optimizer}")

    # Determine aux task type (needed for classification bin handling below)
    aux_task = getattr(model, "aux_task", "bins")
    # Always use quantile-based bins (balanced bin sizes).
    # Rounding was previously used for contrastive but produced unbalanced bins.
    use_rounding = False

    # Prepare datasets based on task type
    if cfg.task_type == "regression":
        y_bins_tr = compute_bins(y_tr, n_bins, use_rounding=use_rounding)
        train_ds = TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr.reshape(-1, 1), dtype=torch.float32),
            torch.tensor(y_bins_tr, dtype=torch.long),
        )
        val_ds = TensorDataset(
            torch.tensor(X_va, dtype=torch.float32),
            torch.tensor(y_va.reshape(-1, 1), dtype=torch.float32),
        )
    else:  # classification
        # For classification, targets are class indices. If bins-aux is enabled
        # with fewer bins than classes, fall back to quantile bins to avoid
        # out-of-range targets in auxiliary cross-entropy.
        y_bins_tr = y_tr.astype(np.int64)
        if aux_task == "bins":
            min_label = int(np.min(y_bins_tr))
            max_label = int(np.max(y_bins_tr))
            if min_label < 0 or max_label >= n_bins:
                logger.warning(
                    "Classification aux_task=bins received class labels outside [0, %d). "
                    "Remapping labels to %d quantile bins for auxiliary supervision.",
                    n_bins,
                    n_bins,
                )
                y_bins_tr = compute_bins(y_tr.astype(np.float64), n_bins, use_rounding=False).astype(np.int64)
        train_ds = TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.long),  # Class indices
            torch.tensor(y_bins_tr, dtype=torch.long),  # Aux labels (classes or remapped bins)
        )
        val_ds = TensorDataset(
            torch.tensor(X_va, dtype=torch.float32),
            torch.tensor(y_va, dtype=torch.long),  # Class indices
        )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    logger.info(f"Training batches per epoch: {len(train_loader)}, Validation batches: {len(val_loader)}")

    # Create optimizer
    if cfg.optimizer == "radam_lookahead":
        optim = create_radam_lookahead(
            model,
            lr=cfg.lr,
            lookahead_k=cfg.lookahead_k,
            lookahead_alpha=cfg.lookahead_alpha,
            weight_decay=cfg.weight_decay,
        )
        logger.info(f"Using RAdam + Lookahead (k={cfg.lookahead_k}, alpha={cfg.lookahead_alpha})")
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        logger.info(f"Using AdamW optimizer (weight_decay={cfg.weight_decay})")

    normalized_scheduler = _normalize_lr_scheduler_name(cfg.lr_scheduler_name)
    scheduler_optimizer = getattr(optim, "base_optimizer", optim)
    scheduler = None
    if normalized_scheduler == "cosine":
        eta_min_ratio = max(0.0, float(cfg.lr_scheduler_eta_min_ratio))
        eta_min = float(cfg.lr) * eta_min_ratio
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            scheduler_optimizer,
            T_max=max(1, int(cfg.epochs)),
            eta_min=eta_min,
        )
        logger.info(
            "Using CosineAnnealingLR scheduler (T_max=%d, eta_min=%.8f, eta_min_ratio=%.4f)",
            max(1, int(cfg.epochs)),
            eta_min,
            eta_min_ratio,
        )
    else:
        logger.info("LR scheduler disabled")

    # LR warmup: linearly ramp from lr/10 to lr over warmup_epochs
    if cfg.lr_warmup_epochs > 0 and isinstance(scheduler_optimizer, torch.optim.Optimizer):
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            scheduler_optimizer,
            start_factor=0.1,
            total_iters=cfg.lr_warmup_epochs,
        )
        if scheduler is not None:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                scheduler_optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[cfg.lr_warmup_epochs],
            )
        else:
            scheduler = warmup_scheduler
        logger.info(
            "LR warmup: %d epochs (start_factor=0.1)",
            cfg.lr_warmup_epochs,
        )

    logger.info("Gradient clipping: %s", cfg.grad_clip_norm)
    
    ce = nn.CrossEntropyLoss()

    best = float("inf")
    best_epoch = None
    best_state_dict: dict[str, torch.Tensor] | None = None
    wait = 0
    
    # Determine aux task type from model
    aux_task = getattr(model, "aux_task", "bins")
    
    # Epoch loop with progress bar
    epoch_pbar = tqdm(
        range(cfg.epochs),
        desc="Training",
        unit="epoch",
        leave=True,
        disable=not cfg.show_progress,
    )
    early_stop_epoch = None
    for epoch in epoch_pbar:
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Batch loop with nested progress bar
        batch_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{cfg.epochs}",
            leave=False,
            unit="batch",
            disable=not cfg.show_progress,
        )
        for xb, yb, yb_bin in batch_pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            yb_bin = yb_bin.to(device)

            with _autocast_if_needed(enabled=amp_state.enabled, dtype=amp_state.dtype):
                output = model(xb)

                # Compute task loss based on task type
                if cfg.task_type == "regression":
                    # Regression: use gaussian_nll
                    if len(output) == 2:
                        # Single MLP or models without aux task
                        mu, sigma = output
                        loss = gaussian_nll(mu, sigma, yb, sigma_reg_weight=cfg.sigma_reg_weight)
                    else:
                        # HMTL models with aux task
                        mu, sigma, aux_output = output
                        loss = gaussian_nll(mu, sigma, yb, sigma_reg_weight=cfg.sigma_reg_weight)

                        if aux_output is not None:
                            loss = loss + cfg.aux_weight * _compute_aux_loss(
                                aux_task, aux_output, yb_bin, yb, ce, model
                            )

                else:  # classification
                    # Classification: use task_loss or cross-entropy
                    if len(output) == 2:
                        # Model without aux task
                        logits = output[0]
                    else:
                        # Model with aux task
                        logits, _, aux_output = output

                    if task_loss is not None:
                        loss = task_loss(logits, yb)
                    else:
                        # Fallback to cross-entropy
                        loss = ce(logits, yb)

                    # Add auxiliary loss if available
                    if len(output) == 3 and output[2] is not None:
                        aux_output = output[2]
                        loss = loss + cfg.aux_weight * _compute_aux_loss(
                            aux_task, aux_output, yb_bin, yb, ce, model
                        )

                # CQR: add quantile loss if enabled
                if cfg.cqr_enabled and hasattr(model, 'quantile_head') and model.quantile_head is not None:
                    from src.losses.quantile import pinball_loss
                    q_preds = model.predict_quantiles(xb)
                    q_target = yb.view(-1, 1) if yb.dim() == 1 else yb
                    cqr_quantiles = cfg.cqr_quantiles or [0.05, 0.95]
                    q_loss = pinball_loss(q_preds, q_target, cqr_quantiles)
                    loss = loss + cfg.cqr_weight * q_loss

                # Adversarial augmentations
                adv_cfg = cfg.adversarial
                if adv_cfg is not None and adv_cfg.enabled:
                    def _adv_loss_fn(x_input):
                        with _autocast_if_needed(enabled=amp_state.enabled, dtype=amp_state.dtype):
                            adv_out = model(x_input)
                            if cfg.task_type == "regression":
                                if len(adv_out) == 2:
                                    mu_a, sigma_a = adv_out
                                else:
                                    mu_a, sigma_a, _ = adv_out
                                return gaussian_nll(mu_a, sigma_a, yb, sigma_reg_weight=cfg.sigma_reg_weight)
                            else:
                                logits_a = adv_out[0]
                                if task_loss is not None:
                                    return task_loss(logits_a, yb)
                                return ce(logits_a, yb)

                    x_adv = generate_adversarial(model, xb, _adv_loss_fn, adv_cfg)
                    with _autocast_if_needed(enabled=amp_state.enabled, dtype=amp_state.dtype):
                        adv_output = model(x_adv)
                        if cfg.task_type == "regression":
                            if len(adv_output) == 2:
                                mu_adv, sigma_adv = adv_output
                            else:
                                mu_adv, sigma_adv, _ = adv_output
                            adv_loss_val = gaussian_nll(mu_adv, sigma_adv, yb, sigma_reg_weight=cfg.sigma_reg_weight)
                        else:
                            logits_adv = adv_output[0]
                            if task_loss is not None:
                                adv_loss_val = task_loss(logits_adv, yb)
                            else:
                                adv_loss_val = ce(logits_adv, yb)
                    loss = loss + adv_cfg.adv_weight * adv_loss_val

            optim.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optim.step()
            
            loss_val = loss.item()
            if not math.isfinite(loss_val):
                logger.warning(
                    f"Epoch {epoch+1}: NaN/Inf loss detected ({loss_val}). Skipping batch."
                )
                continue
            epoch_loss += loss_val
            num_batches += 1
            batch_pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        logger.debug(f"Epoch {epoch+1}: Average training loss = {avg_train_loss:.6f}")

        # Periodic check for NaN in model parameters (every 10 epochs)
        if epoch % 10 == 0:
            has_nan = any(
                torch.isnan(p).any().item()
                for p in model.parameters()
                if p is not None
            )
            if has_nan:
                logger.error(
                    f"Epoch {epoch+1}: NaN detected in model parameters. "
                    f"Restoring best checkpoint and stopping training."
                )
                if best_state_dict is not None:
                    model.load_state_dict(best_state_dict)
                break

        # Validation
        model.eval()
        with torch.no_grad():
            if cfg.task_type == "regression":
                # Regression validation
                preds = []
                sigmas = []
                gts = []
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    with _autocast_if_needed(enabled=use_eval_amp, dtype=amp_state.dtype):
                        output = model(xb)
                    # Handle different model types
                    if len(output) == 2:
                        mu, sigma = output
                    else:
                        mu, sigma, _ = output
                    preds.append(_to_numpy_compatible(mu.float()).ravel())
                    sigmas.append(_to_numpy_compatible(sigma.float()).ravel())
                    gts.append(_to_numpy_compatible(yb).ravel())
                y_pred = np.concatenate(preds)
                y_sigma = np.concatenate(sigmas)
                y_true = np.concatenate(gts)
                mse = float(np.mean((y_true - y_pred) ** 2))
                rmse = float(np.sqrt(mse))
                mae = float(np.mean(np.abs(y_true - y_pred)))
                val_r_auc_mse = float(r_auc_mse((y_true - y_pred) ** 2, y_sigma))
                score = _resolve_regression_validation_score(
                    metric_name=cfg.early_stop_metric,
                    rmse=rmse,
                    r_auc_mse_score=val_r_auc_mse,
                    hybrid_r_auc_weight=float(cfg.hybrid_r_auc_weight),
                )

            else:  # classification
                # Classification validation
                logits_list = []
                gts = []
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    with _autocast_if_needed(enabled=use_eval_amp, dtype=amp_state.dtype):
                        output = model(xb)
                    if len(output) == 2:
                        logits = output[0]
                    else:
                        logits = output[0]
                    logits_list.append(_to_numpy_compatible(logits.float()))
                    gts.append(_to_numpy_compatible(yb))

                logits_val = np.concatenate(logits_list)
                y_true = np.concatenate(gts)

                # Compute softmax probabilities
                from src.eval.ensemble import softmax
                probs_val = softmax(logits_val, axis=-1)
                y_pred = np.argmax(probs_val, axis=-1)

                # Compute metrics
                accuracy = float(np.mean(y_pred == y_true))
                nll = float(np.mean(-np.log(probs_val[np.arange(len(y_true)), y_true.astype(int)] + 1e-10)))

                # Resolve classification early-stop score (lower is better)
                score = _resolve_classification_val_score(
                    metric_name=cfg.early_stop_metric,
                    nll=nll,
                    y_true=y_true,
                    y_pred=y_pred,
                )
                rmse = 0.0  # Not applicable for classification
                mae = 0.0  # Not applicable for classification
                val_r_auc_mse = 0.0

        # Update progress bar with metrics
        epoch_pbar.set_postfix({
            "train_loss": f"{avg_train_loss:.4f}",
            "val_score": f"{score:.6f}",
            "best": f"{best:.6f}" if best != float("inf") else "inf",
            "patience": f"{wait}/{cfg.patience}"
        })

        stop_training = False
        if not math.isfinite(score):
            logger.warning(
                f"Epoch {epoch+1}: NaN/Inf validation score ({score}). "
                f"Treating as failed epoch."
            )
            wait += 1
            if wait >= cfg.patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch+1} "
                    f"(patience: {cfg.patience}, last score was NaN/Inf)"
                )
                early_stop_epoch = epoch
                stop_training = True
        elif score < best:
            improvement = best - score if best != float("inf") else 0.0
            best = score
            best_epoch = epoch
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            wait = 0
            logger.info(f"Epoch {epoch+1}: New best score! {score:.6f} (improvement: {improvement:.6f})")
        else:
            wait += 1
            logger.debug(f"Epoch {epoch+1}: Score {score:.6f} (no improvement, patience: {wait}/{cfg.patience})")
            if wait >= cfg.patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1} (patience: {cfg.patience})")
                early_stop_epoch = epoch
                stop_training = True

        current_lr = float(optim.param_groups[0]["lr"]) if optim.param_groups else float(cfg.lr)

        if history is not None:
            if cfg.task_type == "regression":
                history.append({
                    "epoch": epoch,
                    "train_loss": float(avg_train_loss),
                    "lr": current_lr,
                    "val_score": float(score),
                    "val_metric": _normalize_regression_early_stop_metric(cfg.early_stop_metric),
                    "val_r_auc_mse": float(val_r_auc_mse),
                    "val_rmse": float(rmse),
                    "val_mae": float(mae),
                })
            else:  # classification
                history.append({
                    "epoch": epoch,
                    "train_loss": float(avg_train_loss),
                    "lr": current_lr,
                    "val_nll": float(score),
                    "val_accuracy": float(accuracy),
                })

        if scheduler is not None:
            scheduler.step()

        if stop_training:
            epoch_pbar.close()
            break
    
    if history_meta is not None:
        history_meta["best_epoch"] = best_epoch
        history_meta["early_stop_epoch"] = early_stop_epoch
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logger.info("Restored model weights from best validation epoch: %s", best_epoch)
    logger.info(f"Training completed. Best validation score: {best:.6f}")
    return best
