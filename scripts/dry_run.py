"""
End-to-end smoke test / dry run for BSGM-CellTrack.

Runs every stage of the pipeline on a tiny synthetic dataset (default 10 frames,
5 cells, 256×256) so you can verify the code is correct before committing to
a full training run.

Stages:
  1. Generate synthetic CTC data (10 frames × 256×256)
  2. Convert CTC → COCO JSON annotations
  3. Build the model (tiny config, CPU/GPU auto-detected)
  4. Count parameters
  5. Smoke-test forward pass (3-frame triplet, no targets)
  6. Smoke-test forward pass with targets + loss backward
  7. Smoke-test 2-step training loop (train_one_epoch)
  8. Smoke-test inference (infer-style loop, predictions_to_ctc)
  9. Print summary

Usage (from project root):
    python scripts/dry_run.py
    python scripts/dry_run.py --device cuda --n_frames 10 --n_cells 5
    python scripts/dry_run.py --skip_gen   # reuse existing synthetic data
"""

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

# Add src to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("BSGM-CellTrack dry-run")
    p.add_argument("--device",    default="auto",  help="cpu | cuda | auto")
    p.add_argument("--n_frames",  type=int, default=10)
    p.add_argument("--n_cells",   type=int, default=5)
    p.add_argument("--img_size",  type=int, default=256)
    p.add_argument("--skip_gen",  action="store_true", help="Skip synthetic data gen")
    p.add_argument("--verbose",   action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pretty reporter
# ---------------------------------------------------------------------------

class Reporter:
    def __init__(self):
        self.results = []

    def ok(self, name, detail=""):
        msg = f"  [通过] {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
        self.results.append((name, True, detail))

    def fail(self, name, err):
        print(f"  [失败] {name}")
        print(f"         {err}")
        self.results.append((name, False, str(err)))

    def summary(self):
        n_ok  = sum(1 for _, ok, _ in self.results if ok)
        n_fail = sum(1 for _, ok, _ in self.results if not ok)
        print("\n" + "=" * 60)
        print(f"  干运行汇总：{n_ok} 项通过，{n_fail} 项失败")
        print("=" * 60)
        for name, ok, detail in self.results:
            status = "通过" if ok else "失败"
            print(f"  [{status}] {name}")
            if not ok:
                print(f"         {detail[:120]}")
        print()


R = Reporter()


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def stage(label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Stage 1: generate synthetic data
# ---------------------------------------------------------------------------

def gen_data(out_dir, n_frames, n_cells, img_size, skip):
    stage("阶段 1：生成合成 CTC 数据")
    try:
        if skip and (out_dir / "CTC" / "train" / "01").exists():
            R.ok("合成数据", "已跳过（--skip_gen，数据已存在）")
            return True

        # Run make_synthetic_ctc inline (avoid subprocess for portability)
        sys.path.insert(0, str(ROOT / "scripts"))
        from make_synthetic_ctc import generate
        generate(
            out_dir=str(out_dir),
            n_frames=n_frames,
            n_cells=n_cells,
            H=img_size,
            W=img_size,
            sigma=max(8.0, img_size / 32),
            splits=["train", "val"],
            sequence="01",
        )
        # Verify files
        frame_dir = out_dir / "CTC" / "train" / "01"
        tifs = list(frame_dir.glob("t*.tif"))
        assert len(tifs) == n_frames, f"期望 {n_frames} 帧，实际 {len(tifs)} 帧"
        R.ok("合成数据", f"{n_frames} 帧 × {n_cells} 个细胞，尺寸 {img_size}×{img_size}")
        return True
    except Exception as e:
        R.fail("合成数据", traceback.format_exc()[-300:])
        return False


# ---------------------------------------------------------------------------
# Stage 2: CTC → COCO conversion
# ---------------------------------------------------------------------------

def convert_coco(data_dir):
    stage("阶段 2：CTC → COCO JSON 格式转换")
    try:
        # Import the converter directly
        sys.path.insert(0, str(ROOT / "scripts"))
        from create_coco_from_ctc import process_split
        for split in ["train", "val"]:
            process_split(
                data_dir=str(data_dir),
                split=split,
                sequences=["01"],
                use_rle=False,
            )
        ann_dir = data_dir / "COCO" / "annotations"
        for split in ["train", "val"]:
            p = ann_dir / f"instances_{split}.json"
            assert p.exists(), f"Missing {p}"
        import json
        with open(ann_dir / "instances_train.json") as f:
            d = json.load(f)
        n_imgs = len(d["images"])
        n_anns = len(d["annotations"])
        R.ok("COCO 转换", f"{n_imgs} 张图像，{n_anns} 个标注")
        return True
    except Exception as e:
        R.fail("COCO 转换", traceback.format_exc()[-300:])
        return False


# ---------------------------------------------------------------------------
# Stage 3: model build
# ---------------------------------------------------------------------------

def build(device):
    stage("阶段 3：构建模型（精简配置）")
    try:
        from ai_cstq.models import build_model
        cfg = dict(
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
        import torch
        model = build_model(cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        R.ok("模型构建", f"总参数 {n_params:.1f}M，可训练 {n_train:.1f}M")
        return model, cfg
    except Exception as e:
        R.fail("模型构建", traceback.format_exc()[-400:])
        return None, None


# ---------------------------------------------------------------------------
# Stage 4: forward pass (no targets)
# ---------------------------------------------------------------------------

def forward_no_targets(model, device, img_size, n_cells):
    stage("阶段 4：前向传播（无目标，无损失）")
    try:
        import torch
        B, C, H, W = 1, 3, img_size, img_size
        frames = [torch.randn(B, C, H, W, device=device) for _ in range(3)]

        model.eval()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(frames)
        dt = time.perf_counter() - t0

        logits = out["pred_logits"]
        boxes  = out["pred_boxes"]
        assert logits.ndim == 3, f"Expected 3D logits, got {logits.ndim}D"
        assert boxes.ndim  == 3, f"Expected 3D boxes, got {boxes.ndim}D"
        N = logits.shape[1]
        detail = (
            f"pred_logits {tuple(logits.shape)}, "
            f"pred_boxes {tuple(boxes.shape)}, "
            f"pred_masks {tuple(out['pred_masks'].shape) if out.get('pred_masks') is not None else 'n/a'}, "
            f"uncertainty {tuple(out['uncertainty'].shape)}, "
            f"{dt*1000:.0f} ms"
        )
        R.ok("前向传播（无目标）", detail)
        return True, out
    except Exception as e:
        R.fail("前向传播（无目标）", traceback.format_exc()[-400:])
        return False, None


# ---------------------------------------------------------------------------
# Stage 5: forward + loss backward
# ---------------------------------------------------------------------------

def forward_with_loss(model, cfg, device, img_size, n_cells):
    stage("阶段 5：前向传播 + 损失反向传播")
    try:
        import torch
        from ai_cstq.models.criterion import build_criterion

        loss_cfg = dict(
            num_classes=1,
            cls_loss_coef=4.0,
            bbox_loss_coef=5.0,
            giou_loss_coef=2.0,
            mask_loss_coef=5.0,
            dice_loss_coef=5.0,
            set_cost_class=1.0,
            set_cost_bbox=5.0,
            set_cost_giou=2.0,
            set_cost_mask=1.0,
            focal_alpha=0.25,
            focal_gamma=2.0,
            with_div=False,
            masks=True,
        )
        criterion = build_criterion(loss_cfg).to(device)

        B, C, H, W = 1, 3, img_size, img_size
        frames = [torch.randn(B, C, H, W, device=device) for _ in range(3)]

        # Build synthetic targets that match COCO format consumed by criterion
        # pred_masks shape: (B, N, H_mask, W_mask) — we need to know N first
        model.eval()
        with torch.no_grad():
            out_probe = model(frames)
        mask_h, mask_w = out_probe["pred_masks"].shape[-2:] if out_probe.get("pred_masks") is not None else (H // 4, W // 4)

        targets = []
        for _ in range(B):
            n = n_cells
            # Random normalised boxes [cx, cy, w, h] in (0,1)
            boxes = torch.rand(n, 4, device=device) * 0.4 + 0.1
            boxes = boxes.clamp(0.01, 0.99)
            labels = torch.zeros(n, dtype=torch.long, device=device)     # all class 0 (cell)
            masks  = torch.zeros(n, mask_h, mask_w, device=device)
            for k in range(n):
                cx, cy, bw, bh = boxes[k].tolist()
                x0 = max(0, int((cx - bw / 2) * mask_w))
                x1 = min(mask_w, int((cx + bw / 2) * mask_w))
                y0 = max(0, int((cy - bh / 2) * mask_h))
                y1 = min(mask_h, int((cy + bh / 2) * mask_h))
                masks[k, y0:y1, x0:x1] = 1.0
            targets.append({
                "labels": labels,
                "boxes": boxes,
                "masks": masks,
            })

        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.zero_grad()

        t0 = time.perf_counter()
        out = model(frames, targets=targets)
        loss_dict = criterion(out, targets)
        total_loss = sum(loss_dict.values())
        total_loss.backward()
        grad_norm = sum(
            p.grad.norm().item() ** 2
            for p in model.parameters()
            if p.grad is not None
        ) ** 0.5
        optimizer.step()
        dt = time.perf_counter() - t0

        detail = (
            f"total_loss={total_loss.item():.3f}, "
            f"grad_norm={grad_norm:.3f}, "
            f"{dt*1000:.0f} ms"
        )
        R.ok("前向传播 + 损失反向传播", detail)
        return True
    except Exception as e:
        R.fail("前向传播 + 损失反向传播", traceback.format_exc()[-500:])
        return False


# ---------------------------------------------------------------------------
# Stage 6: mini training loop (engine.train_one_epoch)
# ---------------------------------------------------------------------------

def mini_train(model, cfg, device, data_dir, img_size):
    stage("阶段 6：训练循环（engine.train_one_epoch）")
    try:
        import torch
        from torch.utils.data import DataLoader
        from ai_cstq.datasets.ctc_coco import CTCCocoDataset, collate_fn
        from ai_cstq.models.criterion import build_criterion
        from ai_cstq.engine import train_one_epoch, build_optimizer

        ann_dir = data_dir / "COCO" / "annotations"

        dataset = CTCCocoDataset(
            img_dir=str(data_dir / "CTC" / "train"),
            ann_file=str(ann_dir / "instances_train.json"),
            transforms=None,
            target_size=(img_size, img_size),
            in_channels=3,
            temporal_window=3,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)

        loss_cfg = {**cfg,
            "num_classes": 1,
            "cls_loss_coef": 4.0, "bbox_loss_coef": 5.0, "giou_loss_coef": 2.0,
            "mask_loss_coef": 5.0, "dice_loss_coef": 5.0,
            "set_cost_class": 1.0, "set_cost_bbox": 5.0, "set_cost_giou": 2.0,
            "set_cost_mask": 1.0, "focal_alpha": 0.25, "focal_gamma": 2.0,
        }
        criterion = build_criterion(loss_cfg).to(device)
        optimizer = build_optimizer(model, {"lr": 1e-4, "lr_backbone": 1e-5, "weight_decay": 1e-4})

        t0 = time.perf_counter()
        metrics = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=loader,
            optimizer=optimizer,
            device=device,
            epoch=0,
            max_norm=0.1,
            use_amp=False,
            print_freq=999,   # suppress per-iter prints
            stage="warmup",
        )
        dt = time.perf_counter() - t0
        R.ok("训练循环（warmup 阶段）", f"{len(dataset)} 个样本，耗时 {dt:.1f}s")
        return True
    except Exception as e:
        R.fail("训练循环（warmup 阶段）", traceback.format_exc()[-500:])
        return False


# ---------------------------------------------------------------------------
# Stage 7: inference / predictions_to_ctc
# ---------------------------------------------------------------------------

def mini_infer(model, device, data_dir, img_size):
    stage("阶段 7：推理 + CTC 格式输出")
    try:
        import torch
        from ai_cstq.util.ctc_io import predictions_to_ctc

        res_dir = data_dir / "CTC" / "test_RES"
        res_dir.mkdir(parents=True, exist_ok=True)

        model.eval()
        B, C, H, W = 1, 3, img_size, img_size

        all_outputs = []
        with torch.no_grad():
            for t in range(3):
                frames = [torch.randn(B, C, H, W, device=device) for _ in range(3)]
                out = model(frames)
                # predictions_to_ctc expects a list of per-frame dicts
                out_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                           for k, v in out.items()}
                all_outputs.append(out_cpu)

        predictions_to_ctc(
            all_outputs=all_outputs,
            img_hw=(H, W),
            out_dir=str(res_dir),
            conf_threshold=0.0,   # accept everything for smoke test
            mask_threshold=0.3,
            start_frame=0,
        )
        man_track = res_dir / "man_track.txt"
        assert man_track.exists(), "man_track.txt 未生成"
        masks = list(res_dir.glob("mask*.tif"))
        R.ok("推理 + CTC 输出", f"{len(masks)} 个掩码文件 + man_track.txt")
        return True
    except Exception as e:
        R.fail("推理 + CTC 输出", traceback.format_exc()[-500:])
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.device == "auto":
        import torch
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device

    import torch
    device = torch.device(device_str)
    print(f"\nBSGM-CellTrack 干运行测试")
    print(f"  设备   : {device}")
    print(f"  帧数   : {args.n_frames}")
    print(f"  细胞数 : {args.n_cells}")
    print(f"  尺寸   : {args.img_size}×{args.img_size}")
    print(f"  PyTorch: {torch.__version__}")
    if device.type == "cuda":
        print(f"  CUDA   : {torch.version.cuda}，GPU={torch.cuda.get_device_name(0)}")

    data_dir = ROOT / "data" / "dryrun"

    ok = gen_data(data_dir, args.n_frames, args.n_cells, args.img_size, args.skip_gen)
    if not ok:
        print("\n已中止：数据生成失败。")
        R.summary()
        sys.exit(1)

    ok = convert_coco(data_dir)
    if not ok:
        print("\n已中止：COCO 转换失败。")
        R.summary()
        sys.exit(1)

    model, cfg = build(device)
    if model is None:
        print("\n已中止：模型构建失败。")
        R.summary()
        sys.exit(1)

    forward_no_targets(model, device, args.img_size, args.n_cells)
    forward_with_loss(model, cfg, device, args.img_size, args.n_cells)
    mini_train(model, cfg, device, data_dir, args.img_size)
    mini_infer(model, device, data_dir, args.img_size)

    R.summary()

    n_fail = sum(1 for _, ok, _ in R.results if not ok)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
