"""
BSGM-CellTrack training entry point.

Usage:
    python train.py --config cfgs/ctchuh7_bsgm.yaml
    python train.py --config cfgs/ctcgowt1_bsgm.yaml --resume results/ctchuh7_bsgm/checkpoint_last.pth
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ai_cstq.models import build_model
from ai_cstq.models.criterion import build_criterion
from ai_cstq.datasets import build_dataset
from ai_cstq.datasets.ctc_coco import collate_fn
from ai_cstq.engine import build_optimizer, build_lr_scheduler, train
from ai_cstq.util.misc import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser("BSGM-CellTrack trainer")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--device", default="cuda", help="Device (cuda / cpu)")
    p.add_argument("--output_dir", default=None, help="Override output directory")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Handle base_config inheritance
    if "base_config" in cfg:
        base_path = os.path.join(os.path.dirname(path), cfg.pop("base_config"))
        with open(base_path) as f:
            base = yaml.safe_load(f)
        base.update(cfg)
        cfg = base
    return cfg


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    output_dir = cfg.get("output_dir", "results/default")
    os.makedirs(output_dir, exist_ok=True)

    # Save merged config
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Seed: {seed}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ---- Model ----
    print("Building model...")
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params / 1e6:.1f}M")

    # ---- Criterion ----
    criterion = build_criterion(cfg).to(device)

    # ---- Datasets ----
    print("Loading datasets...")
    train_dataset = build_dataset(cfg, split="train")
    val_dataset = build_dataset(cfg, split="val")
    print(f"  Train: {len(train_dataset)} samples  |  Val: {len(val_dataset)} samples")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.get("batch_size", 1),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.get("num_workers", 2),
        collate_fn=collate_fn,
    )

    # ---- Optimizer ----
    optimizer = build_optimizer(model, cfg)
    lr_scheduler = build_lr_scheduler(optimizer, cfg)

    # ---- Resume ----
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        start_epoch = load_checkpoint(args.resume, model, optimizer) + 1
        # Advance LR scheduler
        for _ in range(start_epoch):
            lr_scheduler.step()

    # ---- Train ----
    print("\nStarting training...")
    train(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        cfg=cfg,
        device=device,
        output_dir=output_dir,
        start_epoch=start_epoch,
    )


if __name__ == "__main__":
    main()
