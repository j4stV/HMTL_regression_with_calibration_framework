from __future__ import annotations

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
from src.models.contrastive import n_pairs_loss


@dataclass
class TrainConfig:
    lr: float = 3e-4
    epochs: int = 200
    batch_size: int = 256
    patience: int = 20
    aux_weight: float = 0.3
    sigma_reg_weight: float = 0.01  # Регуляризация для предотвращения взрыва sigma
    optimizer: str = "radam_lookahead"  # "radam_lookahead" or "adamw"
    lookahead_k: int = 6
    lookahead_alpha: float = 0.5
    weight_decay: float = 0.0
    seed: int | None = None


def compute_bins(y: np.ndarray, n_bins: int) -> np.ndarray:
    # равные квантили
    quantiles = np.quantile(y, np.linspace(0, 1, n_bins + 1))
    quantiles[0] -= 1e-9; quantiles[-1] += 1e-9
    return np.digitize(y, quantiles[1:-1])


def train_model(model: nn.Module, X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray, n_bins: int, cfg: TrainConfig) -> float:
    logger = get_logger("train")
    
    # Set random seed if provided
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        logger.info(f"Set random seed to {cfg.seed}")
    
    # Check CUDA availability with detailed diagnostics
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using device: {device}")
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  GPU count: {torch.cuda.device_count()}")
        logger.info(f"  GPU name: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.warning(f"Using device: {device}")
        logger.warning("  CUDA is not available. Possible reasons:")
        logger.warning("    1. No GPU detected on this system")
        logger.warning("    2. PyTorch was installed without CUDA support (CPU-only version)")
        logger.warning("    3. CUDA drivers are not properly installed")
        logger.warning("    4. CUDA version mismatch between PyTorch and system")
        logger.warning(f"  PyTorch version: {torch.__version__}")
        logger.warning(f"  CUDA available in PyTorch build: {torch.version.cuda is not None}")
    model.to(device)

    logger.debug(f"Training data shape: {X_tr.shape}, Validation data shape: {X_va.shape}")
    logger.debug(f"Training config: lr={cfg.lr}, epochs={cfg.epochs}, batch_size={cfg.batch_size}, patience={cfg.patience}, optimizer={cfg.optimizer}")

    y_bins_tr = compute_bins(y_tr, n_bins)
    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr.reshape(-1, 1), dtype=torch.float32),
        torch.tensor(y_bins_tr, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_va, dtype=torch.float32),
        torch.tensor(y_va.reshape(-1, 1), dtype=torch.float32),
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
    
    ce = nn.CrossEntropyLoss()

    best = float("inf")
    wait = 0
    
    # Determine aux task type from model
    aux_task = getattr(model, "aux_task", "bins")
    
    # Epoch loop with progress bar
    epoch_pbar = tqdm(range(cfg.epochs), desc="Training", unit="epoch", leave=True)
    for epoch in epoch_pbar:
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Batch loop with nested progress bar
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}", leave=False, unit="batch")
        for xb, yb, yb_bin in batch_pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            yb_bin = yb_bin.to(device)
            
            output = model(xb)
            
            # Handle different model types
            if len(output) == 2:
                # Single MLP or models without aux task
                mu, sigma = output
                loss = gaussian_nll(mu, sigma, yb, sigma_reg_weight=cfg.sigma_reg_weight)
            else:
                # HMTL models with aux task
                mu, sigma, aux_output = output
                loss = gaussian_nll(mu, sigma, yb, sigma_reg_weight=cfg.sigma_reg_weight)
                
                if aux_output is not None:
                    if aux_task == "bins":
                        # Classification head
                        loss = loss + cfg.aux_weight * ce(aux_output, yb_bin)
                    elif aux_task == "contrastive":
                        # Contrastive learning
                        aux_loss = n_pairs_loss(aux_output, yb_bin, temperature=0.1)
                        loss = loss + cfg.aux_weight * aux_loss
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            batch_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        logger.debug(f"Epoch {epoch+1}: Average training loss = {avg_train_loss:.6f}")

        # Validation
        model.eval()
        with torch.no_grad():
            preds = []
            sigmas = []
            gts = []
            for xb, yb in val_loader:
                xb = xb.to(device)
                output = model(xb)
                # Handle different model types
                if len(output) == 2:
                    mu, sigma = output
                else:
                    mu, sigma, _ = output
                preds.append(mu.cpu().numpy().ravel())
                sigmas.append(sigma.cpu().numpy().ravel())
                gts.append(yb.numpy().ravel())
            y_pred = np.concatenate(preds)
            y_sigma = np.concatenate(sigmas)
            y_true = np.concatenate(gts)
            score = r_auc_mse((y_true - y_pred) ** 2, y_sigma)

        # Update progress bar with metrics
        epoch_pbar.set_postfix({
            "train_loss": f"{avg_train_loss:.4f}",
            "val_score": f"{score:.6f}",
            "best": f"{best:.6f}" if best != float("inf") else "inf",
            "patience": f"{wait}/{cfg.patience}"
        })

        if score < best:
            improvement = best - score if best != float("inf") else 0.0
            best = score
            wait = 0
            logger.info(f"Epoch {epoch+1}: New best score! {score:.6f} (improvement: {improvement:.6f})")
        else:
            wait += 1
            logger.debug(f"Epoch {epoch+1}: Score {score:.6f} (no improvement, patience: {wait}/{cfg.patience})")
            if wait >= cfg.patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1} (patience: {cfg.patience})")
                epoch_pbar.close()
                break
    
    logger.info(f"Training completed. Best validation score: {best:.6f}")
    return best

