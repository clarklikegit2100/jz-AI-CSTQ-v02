"""
Training and evaluation engine for BSGM-CellTrack.

Supports:
  - Mixed precision training (torch.cuda.amp)
  - Gradient clipping
  - Multi-stage training (detection warmup → full tracking)
  - Evaluation on validation set
"""

import math
import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .util.misc import MetricLogger, get_total_grad_norm, save_checkpoint


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0.1,
    use_amp: bool = True,
    print_freq: int = 50,
    stage: str = "full",            # "warmup" | "track" | "full"
) -> Dict[str, float]:
    """
    Train for one epoch.

    Parameters
    ----------
    stage : controls which queries are active:
        "warmup" — object queries only (no track queries, backbone frozen)
        "track"  — enable tracking (but no dn_track)
        "full"   — full training including dn_track
    """
    model.train()
    criterion.train()

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    logger = MetricLogger()
    start = time.time()

    for batch_idx, (images, targets) in enumerate(data_loader):
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, Tensor) else v
                    for k, v in t.items()} for t in targets]

        # Forward (with optional mixed precision)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            # During warmup: don't pass track queries
            if stage == "warmup":
                outputs = model(images)
            else:
                # TODO: extract track queries from targets if DN-track is used
                outputs = model(images)

            losses = criterion(outputs, targets)
            loss = losses["loss_total"]

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training at epoch {epoch} batch {batch_idx}")
            raise RuntimeError(f"Non-finite loss: {loss.item()}")

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()

        # Logging
        log_dict = {k: v.item() for k, v in losses.items()}
        logger.update(**log_dict)

        if (batch_idx + 1) % print_freq == 0:
            elapsed = time.time() - start
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch}] [{batch_idx+1}/{len(data_loader)}]  "
                f"loss: {loss.item():.4f}  lr: {lr:.2e}  "
                f"time: {elapsed:.1f}s"
            )

    return {k: v.global_avg for k, v in logger.meters.items()}


# ---------------------------------------------------------------------------
# Evaluation step
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> Dict[str, float]:
    """Evaluate on validation set, compute average losses."""
    model.eval()
    criterion.eval()

    logger = MetricLogger()

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, Tensor) else v
                    for k, v in t.items()} for t in targets]

        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            outputs = model(images)
            losses = criterion(outputs, targets)

        logger.update(**{k: v.item() for k, v in losses.items()})

    return {k: v.global_avg for k, v in logger.meters.items()}


# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    lr_scheduler,
    cfg: dict,
    device: torch.device,
    output_dir: str,
    start_epoch: int = 0,
):
    """
    Full training loop with staged curriculum:
      - Epochs [0, warmup_epochs):  detection only, backbone frozen
      - Epochs [warmup_epochs, total_epochs): full tracking
    """
    total_epochs = cfg.get("epochs", 24)
    warmup_epochs = cfg.get("warmup_epochs", 8)
    use_amp = cfg.get("use_amp", True)
    max_norm = cfg.get("clip_max_norm", 0.1)
    save_every = cfg.get("save_checkpoint_every", 4)

    best_val_loss = float("inf")

    for epoch in range(start_epoch, total_epochs):
        # Determine training stage
        if epoch < warmup_epochs:
            stage = "warmup"
            # Freeze backbone during warmup
            for p in model.backbone.parameters():
                p.requires_grad_(False)
        else:
            stage = "full"
            # Unfreeze backbone with lower LR
            for p in model.backbone.parameters():
                p.requires_grad_(True)

        print(f"\n===== Epoch {epoch}/{total_epochs - 1}  stage={stage} =====")
        train_stats = train_one_epoch(
            model, criterion, train_loader, optimizer, device,
            epoch, max_norm=max_norm, use_amp=use_amp, stage=stage,
        )

        val_stats = evaluate(model, criterion, val_loader, device, use_amp=use_amp)

        lr_scheduler.step()

        # Logging
        print(f"Train: {' | '.join(f'{k}: {v:.4f}' for k, v in train_stats.items() if 'total' in k or 'cls' in k or 'bbox' in k)}")
        print(f"Val:   {' | '.join(f'{k}: {v:.4f}' for k, v in val_stats.items() if 'total' in k or 'cls' in k or 'bbox' in k)}")

        val_loss = val_stats.get("loss_total", float("inf"))
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "train_stats": train_stats,
            "val_stats": val_stats,
            "cfg": cfg,
        }
        save_checkpoint(ckpt, os.path.join(output_dir, "checkpoint_last.pth"))
        if is_best:
            save_checkpoint(ckpt, os.path.join(output_dir, "checkpoint_best.pth"))
        if (epoch + 1) % save_every == 0:
            save_checkpoint(ckpt, os.path.join(output_dir, f"checkpoint_epoch{epoch:04d}.pth"))

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# Build optimizer
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg: dict) -> Optimizer:
    """AdamW with different LRs for backbone vs. rest."""
    lr = cfg.get("lr", 2e-4)
    lr_backbone = cfg.get("lr_backbone", 2e-5)
    wd = cfg.get("weight_decay", 1e-4)

    backbone_params = list(model.backbone.parameters())
    backbone_ids = set(id(p) for p in backbone_params)
    other_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    param_groups = [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": other_params, "lr": lr},
    ]
    return torch.optim.AdamW(param_groups, weight_decay=wd)


def build_lr_scheduler(optimizer: Optimizer, cfg: dict):
    """MultiStepLR that drops LR by 0.1 at lr_drop epoch."""
    lr_drop = cfg.get("lr_drop", 20)
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[lr_drop], gamma=0.1)
