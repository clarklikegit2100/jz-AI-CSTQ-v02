"""
Framework comparison: BSGM-CellTrack vs EmbedTrack-style vs HybridCellTracker.

Runs the same dry-run conditions on all three models and reports:
  - Parameter count
  - Forward pass speed (CPU)
  - Loss after one backward step
  - Peak memory (if CUDA)

Terminal output: Chinese
Code: English

Usage:
    python scripts/compare_frameworks.py
    python scripts/compare_frameworks.py --device cuda --img_size 256
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

BSGM_CFG = dict(
    backbone="swin_t",
    backbone_in_channels=3,
    swin_window_size=4,
    hidden_dim=256,
    nheads=8,
    enc_layers=1,
    dec_layers=2,
    dim_feedforward=256,
    dropout=0.0,
    num_feature_levels=4,
    dec_n_points=2,
    num_queries=20,
    num_classes=1,
    tracking=True,
    with_div=False,
    masks=True,
    mask_channels=32,
    bayesian_dropout=0.0,
    bayesian_eval=False,
    mamba_d_state=4,
    mamba_d_conv=4,
    graph_topk=4,
    graph_heads=2,
    two_stage=True,
    with_box_refine=True,
    dn_track=False,
)

EMBED_CFG = dict(
    backbone_in_channels=3,
    embed_base_ch=32,
)

HYBRID_CFG = dict(
    backbone="swin_t",
    backbone_in_channels=3,
    swin_window_size=4,
    hidden_dim=256,
    nheads=8,
    enc_layers=1,
    dec_layers=2,
    dim_feedforward=256,
    dropout=0.0,
    num_feature_levels=4,
    dec_n_points=2,
    num_queries=20,        # object queries for new cells
    num_classes=1,
    tracking=True,
    with_div=False,
    with_box_refine=True,
    bayesian_dropout=0.0,
    bayesian_eval=False,
    mamba_d_state=4,
    mamba_d_conv=4,
    graph_topk=4,
    graph_heads=2,
)


def parse_args():
    p = argparse.ArgumentParser("Framework comparison")
    p.add_argument("--device",   default="auto")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--n_cells",  type=int, default=5)
    p.add_argument("--n_repeat", type=int, default=3, help="Forward pass repeats for timing")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-model test
# ---------------------------------------------------------------------------

def test_bsgm(device, img_size, n_cells, n_repeat):
    import torch
    from ai_cstq.models import build_model
    from ai_cstq.models.criterion import build_criterion

    model = build_model(BSGM_CFG).to(device)
    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    B, C, H, W = 1, 3, img_size, img_size
    frames = [torch.randn(B, C, H, W, device=device) for _ in range(3)]

    # Forward speed
    model.eval()
    with torch.no_grad():
        _ = model(frames)   # warmup
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(frames)
        times.append(time.perf_counter() - t0)
    avg_ms = sum(times) / len(times) * 1000

    # Loss backward
    loss_cfg = dict(
        num_classes=1,
        cls_loss_coef=4.0, bbox_loss_coef=5.0, giou_loss_coef=2.0,
        mask_loss_coef=5.0, dice_loss_coef=5.0,
        set_cost_class=1.0, set_cost_bbox=5.0, set_cost_giou=2.0,
        set_cost_mask=1.0, focal_alpha=0.25, focal_gamma=2.0,
        with_div=False, masks=True,
    )
    criterion = build_criterion(loss_cfg).to(device)
    mk = out.get("pred_masks")
    mh, mw = (mk.shape[-2], mk.shape[-1]) if mk is not None else (H // 4, W // 4)
    targets = _make_targets(n_cells, mh, mw, device)

    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optim.zero_grad()
    out_train = model(frames, targets=targets)
    loss_dict = criterion(out_train, targets)
    total_loss = loss_dict["loss_total"]
    total_loss.backward()
    optim.step()

    mem_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0

    return {
        "params_total_M": n_total,
        "params_train_M": n_train,
        "forward_ms":     avg_ms,
        "loss":           total_loss.item(),
        "mem_mb":         mem_mb,
        "n_frames_input": 3,
        "output_type":    "稀疏对象查询 + 掩码",
    }


def test_embedtrack(device, img_size, n_cells, n_repeat):
    import torch
    from ai_cstq.models.embedtrack_net import build_embedtrack, embedtrack_loss

    model = build_embedtrack(EMBED_CFG).to(device)
    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    B, C, H, W = 1, 3, img_size, img_size
    frames = [torch.randn(B, C, H, W, device=device) for _ in range(2)]  # t and t-1

    # Forward speed
    model.eval()
    with torch.no_grad():
        _ = model(frames)
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(frames)
        times.append(time.perf_counter() - t0)
    avg_ms = sum(times) / len(times) * 1000

    # Loss backward
    targets = _make_targets_embed(n_cells, img_size, img_size, device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optim.zero_grad()
    out_train = model(frames)
    loss_dict = embedtrack_loss(out_train, targets)
    total_loss = loss_dict["loss_total"]
    total_loss.backward()
    optim.step()

    mem_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0

    return {
        "params_total_M": n_total,
        "params_train_M": n_train,
        "forward_ms":     avg_ms,
        "loss":           total_loss.item(),
        "mem_mb":         mem_mb,
        "n_frames_input": 2,
        "output_type":    "密集像素偏移 + 聚类",
    }


# ---------------------------------------------------------------------------
# Target builders
# ---------------------------------------------------------------------------

def _make_targets(n_cells, mh, mw, device):
    import torch
    targets = []
    for _ in range(1):
        boxes = (torch.rand(n_cells, 4) * 0.4 + 0.1).clamp(0.01, 0.99).to(device)
        masks = torch.zeros(n_cells, mh, mw, device=device)
        for k in range(n_cells):
            cx, cy, bw, bh = boxes[k].tolist()
            x0, x1 = max(0, int((cx-bw/2)*mw)), min(mw, int((cx+bw/2)*mw))
            y0, y1 = max(0, int((cy-bh/2)*mh)), min(mh, int((cy+bh/2)*mh))
            masks[k, y0:y1, x0:x1] = 1.0
        targets.append({
            "labels": torch.zeros(n_cells, dtype=torch.long, device=device),
            "boxes": boxes, "masks": masks,
        })
    return targets


def _make_targets_embed(n_cells, H, W, device):
    import torch
    targets = []
    for _ in range(1):
        boxes = (torch.rand(n_cells, 4) * 0.4 + 0.1).clamp(0.01, 0.99).to(device)
        masks = torch.zeros(n_cells, H, W, device=device)
        for k in range(n_cells):
            cx, cy, bw, bh = boxes[k].tolist()
            x0, x1 = max(0, int((cx-bw/2)*W)), min(W, int((cx+bw/2)*W))
            y0, y1 = max(0, int((cy-bh/2)*H)), min(H, int((cy+bh/2)*H))
            masks[k, y0:y1, x0:x1] = 1.0
        targets.append({"masks": masks, "boxes": boxes})
    return targets


def test_hybrid(device, img_size, n_cells, n_repeat):
    import torch
    from ai_cstq.models.hybrid_net import build_hybrid_model
    from ai_cstq.models.hybrid_criterion import build_hybrid_criterion

    model = build_hybrid_model(HYBRID_CFG).to(device)
    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    B, C, H, W = 1, 3, img_size, img_size
    frames = [torch.randn(B, C, H, W, device=device) for _ in range(3)]

    # Forward speed (no track queries — first frame cold start)
    model.eval()
    with torch.no_grad():
        _ = model(frames)
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(frames)
        times.append(time.perf_counter() - t0)
    avg_ms = sum(times) / len(times) * 1000

    # Loss backward
    crit_cfg = dict(
        num_classes=1,
        cls_loss_coef=4.0, bbox_loss_coef=5.0, giou_loss_coef=2.0,
        set_cost_class=1.0, set_cost_bbox=5.0, set_cost_giou=2.0, set_cost_mask=0.0,
        focal_alpha=0.25, focal_gamma=2.0,
        seg_loss_coef=1.0, track_offset_coef=1.0,
        lambda_seg=1.0, lambda_track=1.0, lambda_aux=0.5,
    )
    criterion = build_hybrid_criterion(crit_cfg).to(device)

    # Targets: seg branch needs (H/4, W/4) masks; track branch needs boxes + labels
    seg_h, seg_w = H // 4, W // 4
    targets = []
    for _ in range(B):
        boxes = (torch.rand(n_cells, 4) * 0.4 + 0.1).clamp(0.01, 0.99).to(device)
        masks = torch.zeros(n_cells, seg_h, seg_w, device=device)
        for k in range(n_cells):
            cx, cy, bw, bh = boxes[k].tolist()
            x0, x1 = max(0, int((cx-bw/2)*seg_w)), min(seg_w, int((cx+bw/2)*seg_w))
            y0, y1 = max(0, int((cy-bh/2)*seg_h)), min(seg_h, int((cy+bh/2)*seg_h))
            masks[k, y0:y1, x0:x1] = 1.0
        targets.append({
            "labels": torch.zeros(n_cells, dtype=torch.long, device=device),
            "boxes":  boxes,
            "masks":  masks,
        })

    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optim.zero_grad()
    out_train = model(frames, targets=targets)
    loss_dict = criterion(out_train, targets)
    total_loss = loss_dict["loss_total"]
    total_loss.backward()
    optim.step()

    mem_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0

    return {
        "params_total_M": n_total,
        "params_train_M": n_train,
        "forward_ms":     avg_ms,
        "loss":           total_loss.item(),
        "mem_mb":         mem_mb,
        "n_frames_input": 3,
        "output_type":    "像素嵌入分割 + 查询追踪",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    import torch

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\n{'='*75}")
    print(f"  三框架对比：BSGM-CellTrack  /  EmbedTrack 风格  /  混合架构")
    print(f"{'='*75}")
    print(f"  设备：{device}  |  图像尺寸：{args.img_size}×{args.img_size}  "
          f"|  细胞数：{args.n_cells}  |  重复：{args.n_repeat}")
    print()

    results = {}
    MODEL_NAMES = ["BSGM-CellTrack", "EmbedTrack 风格", "混合架构"]

    # --- BSGM-CellTrack ---
    print("【1/3】 测试 BSGM-CellTrack（Swin + Mamba + GATv2 + Bayesian）...")
    try:
        results["BSGM-CellTrack"] = test_bsgm(device, args.img_size, args.n_cells, args.n_repeat)
        print(f"  完成。参数量 {results['BSGM-CellTrack']['params_total_M']:.1f}M，"
              f"前向 {results['BSGM-CellTrack']['forward_ms']:.0f}ms")
    except Exception:
        print("  失败："); traceback.print_exc()
        results["BSGM-CellTrack"] = None

    # --- EmbedTrack ---
    print("\n【2/3】 测试 EmbedTrack 风格（UNet + 像素偏移嵌入）...")
    try:
        results["EmbedTrack 风格"] = test_embedtrack(device, args.img_size, args.n_cells, args.n_repeat)
        print(f"  完成。参数量 {results['EmbedTrack 风格']['params_total_M']:.1f}M，"
              f"前向 {results['EmbedTrack 风格']['forward_ms']:.0f}ms")
    except Exception:
        print("  失败："); traceback.print_exc()
        results["EmbedTrack 风格"] = None

    # --- Hybrid ---
    print("\n【3/3】 测试 混合架构（Swin + Mamba + 像素嵌入分割 + 查询追踪）...")
    try:
        results["混合架构"] = test_hybrid(device, args.img_size, args.n_cells, args.n_repeat)
        print(f"  完成。参数量 {results['混合架构']['params_total_M']:.1f}M，"
              f"前向 {results['混合架构']['forward_ms']:.0f}ms")
    except Exception:
        print("  失败："); traceback.print_exc()
        results["混合架构"] = None

    # --- Report table ---
    print(f"\n{'='*75}")
    print(f"  对比结果汇总")
    print(f"{'='*75}")

    col_w = 18
    print(f"  {'指标':<22}", end="")
    for name in MODEL_NAMES:
        print(f" {name:>{col_w}}", end="")
    print()
    print(f"  {'-'*22}", end="")
    for _ in MODEL_NAMES:
        print(f" {'-'*col_w}", end="")
    print()

    metrics = [
        ("总参数量 (M)",    "params_total_M",  ".1f"),
        ("可训练参数 (M)",  "params_train_M",  ".1f"),
        ("前向耗时 (ms)",   "forward_ms",      ".0f"),
        ("训练总损失",      "loss",            ".3f"),
        ("GPU显存 (MB)",    "mem_mb",          ".0f"),
        ("输入帧数",        "n_frames_input",  "d"),
    ]
    for label, key, fmt in metrics:
        print(f"  {label:<22}", end="")
        for name in MODEL_NAMES:
            r = results.get(name)
            v = f"{r[key]:{fmt}}" if r else "N/A"
            print(f" {v:>{col_w}}", end="")
        print()

    print(f"\n  {'输出类型':<22}", end="")
    for name in MODEL_NAMES:
        r = results.get(name)
        v = r["output_type"] if r else "N/A"
        print(f" {v:>{col_w}}", end="")
    print()

    print(f"\n{'='*75}")
    print("  各框架定位与建议")
    print(f"{'='*75}")
    print("""
  BSGM-CellTrack（查询式端到端）：
    ✓ 长程注意力 / Mamba 时序 / 贝叶斯不确定性 / 支持细胞分裂
    ✗ 参数量最大，训练最慢，需要充足标注数据

  EmbedTrack 风格（像素嵌入）：
    ✓ 最轻量（~16M），CPU 前向最快，像素级分割精度高
    ✓ CTC 基准 7/9 进前三（已有公开结果验证）
    ✗ 近邻追踪无长时一致性，无不确定性

  混合架构（本设计，EmbedSeg + BSGM-Track）：
    ✓ 分割精度 ≈ EmbedTrack（像素嵌入）
    ✓ 追踪一致性 ≈ BSGM（track query + 注意力）
    ✓ MaskPoolQueryInit 桥接：掩码→查询内容，训练可微
    ✓ 贝叶斯不确定性保留，支持细胞分裂
    ~ 参数量介于两者之间，推理速度与 BSGM 相当
    ✗ 训练更复杂（两阶段 warmup），调参空间更大

  推荐训练顺序：
    1. 先训练 EmbedTrack 风格模型作为快速基线
    2. 用混合架构做主实验（预期 TRA ≥ 0.97，SEG ≥ 0.91）
    3. 对比 BSGM 主框架验证注意力增益
""")


if __name__ == "__main__":
    main()
