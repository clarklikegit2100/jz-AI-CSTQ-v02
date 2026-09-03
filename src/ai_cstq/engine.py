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


@torch.no_grad()
def _previous_frame_track_queries(
    model: nn.Module, images: List[Tensor], max_track_queries: int,
):
    """Create detached track queries from a previous-centred temporal window."""
    if max_track_queries <= 0 or len(images) < 2:
        return None, None
    previous_window = [images[0], images[0], images[1]]
    previous = model(previous_window)
    scores = previous["pred_logits"][..., 0].sigmoid()
    count = min(max_track_queries, scores.shape[1])
    topk = scores.topk(count, dim=1).indices
    hidden = torch.gather(
        previous["hs_embed"], 1,
        topk.unsqueeze(-1).expand(-1, -1, previous["hs_embed"].shape[-1]),
    )
    boxes = torch.gather(
        previous["pred_boxes"], 1,
        topk.unsqueeze(-1).expand(-1, -1, previous["pred_boxes"].shape[-1]),
    )
    return hidden.detach(), boxes.detach()


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
    max_track_queries: int = 50,
    max_steps: Optional[int] = None,
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
                track_hs, track_boxes = _previous_frame_track_queries(
                    model, images, max_track_queries,
                )
                outputs = model(
                    images,
                    track_query_hs_embeds=track_hs,
                    track_query_boxes=track_boxes,
                )

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

        if max_steps is not None and batch_idx + 1 >= max_steps:
            break

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
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate on validation set, compute average losses."""
    model.eval()
    criterion.eval()

    logger = MetricLogger()

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, Tensor) else v
                    for k, v in t.items()} for t in targets]

        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            outputs = model(images)
            losses = criterion(outputs, targets)

        logger.update(**{k: v.item() for k, v in losses.items()})
        if max_steps is not None and batch_idx + 1 >= max_steps:
            break

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
    max_track_queries = cfg.get("max_track_queries", 50)
    max_train_steps = cfg.get("max_train_steps")
    max_val_steps = cfg.get("max_val_steps")
    mask_warmup_epochs = cfg.get("mask_warmup_epochs", 0)

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

        criterion.mask_enabled = epoch >= mask_warmup_epochs
        if hasattr(criterion, "module"):
            criterion.module.mask_enabled = criterion.mask_enabled

        print(f"\n===== Epoch {epoch}/{total_epochs - 1}  stage={stage}  "
              f"mask={'on' if criterion.mask_enabled else 'warmup'} =====")
        train_stats = train_one_epoch(
            model, criterion, train_loader, optimizer, device,
            epoch, max_norm=max_norm, use_amp=use_amp, stage=stage,
            max_track_queries=max_track_queries, max_steps=max_train_steps,
        )

        val_stats = evaluate(
            model, criterion, val_loader, device, use_amp=use_amp,
            max_steps=max_val_steps,
        )

        lr_scheduler.step()

        # Logging
        reported = ("loss_cls", "loss_bbox", "loss_giou", "loss_mask_focal",
                    "loss_mask_dice", "loss_total",
                    "stat_mask_iou", "stat_mask_pred_fg", "stat_mask_matched")
        print(f"Train: {' | '.join(f'{k}: {train_stats[k]:.4f}' for k in reported if k in train_stats)}")
        print(f"Val:   {' | '.join(f'{k}: {val_stats[k]:.4f}' for k in reported if k in val_stats)}")

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
    mask_lr_mult = cfg.get("mask_lr_mult", 1.0)
    wd = cfg.get("weight_decay", 1e-4)

    backbone_params = list(model.backbone.parameters())
    backbone_ids = set(id(p) for p in backbone_params)

    def is_mask_param(name: str) -> bool:
        return ("mask_head" in name) or ("pixel_decoder" in name)

    mask_ids = set(
        id(p) for n, p in model.named_parameters() if is_mask_param(n)
    )
    other_params = [
        p for p in model.parameters()
        if id(p) not in backbone_ids and id(p) not in mask_ids
    ]
    mask_params = [
        p for n, p in model.named_parameters()
        if is_mask_param(n) and id(p) not in backbone_ids
    ]

    param_groups = [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": other_params, "lr": lr},
    ]
    if mask_params:
        param_groups.append({"params": mask_params, "lr": lr * mask_lr_mult})
    return torch.optim.AdamW(param_groups, weight_decay=wd)


def build_lr_scheduler(optimizer: Optimizer, cfg: dict):
    """MultiStepLR that drops LR by 0.1 at lr_drop epoch."""
    lr_drop = cfg.get("lr_drop", 20)
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[lr_drop], gamma=0.1)
