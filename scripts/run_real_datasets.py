"""
Run all 3 models (BSGM / EmbedTrack / Hybrid) on real CTC datasets.

For each of the 6 datasets:
  - Loads one real batch via CTCCocoDataset (target_size=256x256)
  - Runs forward + loss backward for each model
  - Reports: orig image size, #cells, forward time, loss, grad norm

Terminal output: Chinese
Code: English

Usage:
    python scripts/run_real_datasets.py
    python scripts/run_real_datasets.py --device cuda --img_size 512
    python scripts/run_real_datasets.py --models bsgm hybrid
    python scripts/run_real_datasets.py --datasets dhela sim
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

DATA_ROOT = ROOT / "data"

# 6 datasets: key -> (dir_tag, full_name, orig_shape)
DATASETS = {
    "dhela":  ("ctc-dhela",  "DIC-C2DH-HeLa",   "(512,512)"),
    "sim":    ("ctc-sim",    "Fluo-N2DH-SIM+",   "(773,739)"),
    "gowt1":  ("ctc-gowt1",  "Fluo-N2DH-GOWT1", "(1024,1024)"),
    "huh7":   ("ctc-huh7",   "Fluo-C2DL-Huh7",  "(1024,1024)"),
    "u373":   ("ctc-u373",   "PhC-C2DH-U373",   "(520,696)"),
    "psc":    ("ctc-psc",    "PhC-C2DL-PSC",    "(576,720)"),
}

# Tiny model configs (same as dry_run.py — fast on CPU)
BSGM_CFG = dict(
    backbone="swin_t", backbone_in_channels=3, swin_window_size=4,
    hidden_dim=256, nheads=8, enc_layers=1, dec_layers=2,
    dim_feedforward=256, dropout=0.0, num_feature_levels=4, dec_n_points=2,
    num_queries=20, num_classes=1, tracking=True, with_div=False,
    masks=True, mask_channels=32, bayesian_dropout=0.0, bayesian_eval=False,
    mamba_d_state=4, mamba_d_conv=4, graph_topk=4, graph_heads=2,
    two_stage=True, with_box_refine=True, dn_track=False,
)
EMBED_CFG = dict(backbone_in_channels=3, embed_base_ch=32)
HYBRID_CFG = dict(
    backbone="swin_t", backbone_in_channels=3, swin_window_size=4,
    hidden_dim=256, nheads=8, enc_layers=1, dec_layers=2,
    dim_feedforward=256, dropout=0.0, num_feature_levels=4, dec_n_points=2,
    num_queries=20, num_classes=1, tracking=True, with_div=False,
    with_box_refine=True, bayesian_dropout=0.0, bayesian_eval=False,
    mamba_d_state=4, mamba_d_conv=4, graph_topk=4, graph_heads=2,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_real_batch(ds_tag: str, split: str, img_size: int, device: torch.device):
    """
    Load one batch (first sample) from a real CTC dataset.
    Returns (frames_list, targets, orig_hw, n_cells) or None on failure.
    """
    from ai_cstq.datasets.ctc_coco import CTCCocoDataset, collate_fn
    from torch.utils.data import DataLoader

    data_dir = DATA_ROOT / ds_tag
    img_dir  = str(data_dir / "CTC" / split)
    ann_file = str(data_dir / "COCO" / "annotations" / f"instances_{split}.json")

    if not Path(ann_file).exists():
        return None

    dataset = CTCCocoDataset(
        img_dir=img_dir,
        ann_file=ann_file,
        transforms=None,
        target_size=(img_size, img_size),
        in_channels=3,
        temporal_window=3,
        augment=False,
    )
    if len(dataset) == 0:
        return None

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)
    batch = next(iter(loader))

    frames_list, targets = batch
    # frames_list: list of T tensors, each (B, C, H, W)
    frames = [f.to(device) for f in frames_list]
    targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()} for t in targets]

    orig_hw = targets[0]["orig_size"]
    n_cells = targets[0]["labels"].shape[0]
    return frames, targets, orig_hw, n_cells


# ---------------------------------------------------------------------------
# Model testers
# ---------------------------------------------------------------------------

def _resize_masks_to(targets, new_h, new_w):
    """Resize target masks from (H, W) to (new_h, new_w) for criterion."""
    import torch.nn.functional as F
    resized = []
    for t in targets:
        m = t["masks"].float()  # (M, H, W)
        if m.shape[0] == 0:
            resized.append(t)
            continue
        m = F.interpolate(m.unsqueeze(0), size=(new_h, new_w),
                          mode="nearest").squeeze(0).bool()
        resized.append({**t, "masks": m})
    return resized


def run_bsgm(frames, targets, device, img_size):
    from ai_cstq.models import build_model
    from ai_cstq.models.criterion import build_criterion

    model = build_model(BSGM_CFG).to(device)
    model.eval()
    with torch.no_grad():
        out_probe = model(frames)

    # Get predicted mask resolution for target resizing
    pmasks = out_probe.get("pred_masks")
    mh, mw = (pmasks.shape[-2], pmasks.shape[-1]) if pmasks is not None else (img_size // 4, img_size // 4)
    targets_resized = _resize_masks_to(targets, mh, mw)

    loss_cfg = dict(
        num_classes=1, cls_loss_coef=4.0, bbox_loss_coef=5.0, giou_loss_coef=2.0,
        mask_loss_coef=5.0, dice_loss_coef=5.0, set_cost_class=1.0,
        set_cost_bbox=5.0, set_cost_giou=2.0, set_cost_mask=1.0,
        focal_alpha=0.25, focal_gamma=2.0, with_div=False, masks=True,
    )
    criterion = build_criterion(loss_cfg).to(device)

    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optim.zero_grad()

    t0 = time.perf_counter()
    out = model(frames, targets=targets_resized)
    loss_dict = criterion(out, targets_resized)
    total_loss = loss_dict["loss_total"]
    total_loss.backward()
    dt = time.perf_counter() - t0

    grad_norm = sum(p.grad.norm().item() ** 2
                    for p in model.parameters() if p.grad is not None) ** 0.5
    optim.step()

    mem_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0
    return total_loss.item(), dt * 1000, grad_norm, mem_mb


def run_embedtrack(frames, targets, device, img_size):
    from ai_cstq.models.embedtrack_net import build_embedtrack, embedtrack_loss

    model = build_embedtrack(EMBED_CFG).to(device)
    frames_2 = frames[:2]   # EmbedTrack takes [t, t-1]

    # EmbedTrack targets: masks at full img_size, boxes
    targets_embed = [{"masks": t["masks"].float(), "boxes": t["boxes"]}
                     for t in targets]
    # Resize masks to img_size if needed (already at img_size from loader)
    import torch.nn.functional as F
    for i, t in enumerate(targets_embed):
        m = t["masks"]
        if m.shape[0] > 0 and (m.shape[-2] != img_size or m.shape[-1] != img_size):
            m = F.interpolate(m.unsqueeze(0), size=(img_size, img_size),
                              mode="nearest").squeeze(0)
        targets_embed[i]["masks"] = m

    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optim.zero_grad()

    t0 = time.perf_counter()
    out = model(frames_2)
    loss_dict = embedtrack_loss(out, targets_embed)
    total_loss = loss_dict["loss_total"]
    total_loss.backward()
    dt = time.perf_counter() - t0

    grad_norm = sum(p.grad.norm().item() ** 2
                    for p in model.parameters() if p.grad is not None) ** 0.5
    optim.step()

    mem_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0
    return total_loss.item(), dt * 1000, grad_norm, mem_mb


def run_hybrid(frames, targets, device, img_size):
    from ai_cstq.models.hybrid_net import build_hybrid_model
    from ai_cstq.models.hybrid_criterion import build_hybrid_criterion

    model = build_hybrid_model(HYBRID_CFG).to(device)
    seg_h, seg_w = img_size // 4, img_size // 4
    targets_hybrid = _resize_masks_to(targets, seg_h, seg_w)

    crit_cfg = dict(
        num_classes=1, cls_loss_coef=4.0, bbox_loss_coef=5.0, giou_loss_coef=2.0,
        set_cost_class=1.0, set_cost_bbox=5.0, set_cost_giou=2.0, set_cost_mask=0.0,
        focal_alpha=0.25, focal_gamma=2.0,
        seg_loss_coef=1.0, track_offset_coef=1.0,
        lambda_seg=1.0, lambda_track=1.0, lambda_aux=0.5,
    )
    criterion = build_hybrid_criterion(crit_cfg).to(device)

    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optim.zero_grad()

    t0 = time.perf_counter()
    out = model(frames, targets=targets_hybrid)
    loss_dict = criterion(out, targets_hybrid)
    total_loss = loss_dict["loss_total"]
    total_loss.backward()
    dt = time.perf_counter() - t0

    grad_norm = sum(p.grad.norm().item() ** 2
                    for p in model.parameters() if p.grad is not None) ** 0.5
    optim.step()

    mem_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0
    return total_loss.item(), dt * 1000, grad_norm, mem_mb


MODEL_RUNNERS = {
    "bsgm":    ("BSGM-CellTrack",  run_bsgm),
    "embed":   ("EmbedTrack 风格",  run_embedtrack),
    "hybrid":  ("混合架构",          run_hybrid),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Real-data pipeline test")
    p.add_argument("--device",   default="auto")
    p.add_argument("--img_size", type=int, default=256,
                   help="Resize all images to this size (default 256)")
    p.add_argument("--split",    default="train", choices=["train", "val", "test"])
    p.add_argument("--models",   nargs="+",
                   choices=list(MODEL_RUNNERS.keys()),
                   default=list(MODEL_RUNNERS.keys()),
                   help="Which models to test (default: all 3)")
    p.add_argument("--datasets", nargs="+",
                   choices=list(DATASETS.keys()),
                   default=list(DATASETS.keys()),
                   help="Which datasets to test (default: all 6)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print()
    print("=" * 72)
    print("  真实 CTC 数据流水线测试 — jz-AI-CSTQ-v02")
    print("=" * 72)
    print(f"  设备: {device}  |  图像尺寸: {args.img_size}×{args.img_size}  "
          f"|  Split: {args.split}")
    print(f"  数据集: {', '.join(args.datasets)}")
    print(f"  模型:   {', '.join(args.models)}")
    print()

    # results[ds_key][model_key] = (loss, ms, grad_norm, mem_mb) or None
    results = {ds: {} for ds in args.datasets}
    ds_meta = {}   # ds_key -> (n_cells, orig_hw)

    for ds_key in args.datasets:
        ds_tag, ds_name, orig_shape = DATASETS[ds_key]
        print(f"[ {ds_name} ({ds_tag}) ]")

        # Load real batch once, reuse across models
        try:
            batch = load_real_batch(ds_tag, args.split, args.img_size, device)
            if batch is None:
                print(f"  跳过：数据不存在或 COCO 标注缺失")
                continue
            frames, targets, orig_hw, n_cells = batch
            ds_meta[ds_key] = (n_cells, orig_hw)
            print(f"  原始尺寸: {orig_hw}  -> 调整至 {args.img_size}×{args.img_size}  "
                  f"|  本批次细胞数: {n_cells}")
        except Exception as e:
            print(f"  数据加载失败: {e}")
            traceback.print_exc()
            continue

        for model_key in args.models:
            model_name, runner = MODEL_RUNNERS[model_key]
            try:
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                loss, ms, gn, mem = runner(frames, targets, device, args.img_size)
                results[ds_key][model_key] = (loss, ms, gn, mem)
                print(f"  [{model_name}]  loss={loss:.3f}  "
                      f"{ms:.0f}ms  grad_norm={gn:.3f}  "
                      f"{'mem=' + f'{mem:.0f}MB' if mem > 0 else 'CPU'}")
            except Exception as e:
                results[ds_key][model_key] = None
                print(f"  [{model_name}]  失败: {e}")
                if "--verbose" in sys.argv:
                    traceback.print_exc()
        print()

    # ---------- Summary table ----------
    model_names = [MODEL_RUNNERS[k][0] for k in args.models]
    col = 20

    print("=" * 72)
    print("  汇总结果（前向+反向，ms = forward+backward 总耗时）")
    print("=" * 72)

    # Header
    print(f"  {'数据集':20s} {'细胞':>5s}", end="")
    for mn in model_names:
        print(f"  {mn[:col]:>{col}s}(ms/loss)", end="")
    print()
    print(f"  {'-'*20} {'-'*5}", end="")
    for _ in model_names:
        print(f"  {'-'*(col+9)}", end="")
    print()

    for ds_key in args.datasets:
        if ds_key not in ds_meta:
            continue
        n_cells, _ = ds_meta[ds_key]
        _, ds_name, _ = DATASETS[ds_key]
        print(f"  {ds_name:20s} {n_cells:>5d}", end="")
        for model_key in args.models:
            r = results[ds_key].get(model_key)
            if r:
                loss, ms, gn, _ = r
                val = f"{ms:.0f}ms / {loss:.3f}"
            else:
                val = "失败"
            print(f"  {val:>{col+9}s}", end="")
        print()

    print("=" * 72)
    n_pass = sum(1 for ds in results.values() for r in ds.values() if r is not None)
    n_total = sum(1 for ds in results.values() for r in ds.values())
    print(f"  通过 {n_pass}/{n_total} 项测试")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
