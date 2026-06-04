"""
Run one full training epoch on each of the 6 CTC datasets (smallest → largest).

Source data  : F:/GitHub/99-CellTracktor/code-ubu2004/data/{src_tag}/CTC/train/
COCO cache   : F:/GitHub/jz-AI-CSTQ-v02/data/{dst_tag}/COCO/annotations/instances_train_full.json
Model        : BSGM-CellTrack (tiny config, same as dry_run.py)
Device       : CUDA if available, else CPU

Dataset order (smallest → largest by train triplets):
  1. Fluo-C2DL-Huh7     672 samples   ~5.8 min
  2. DIC-C2DH-HeLa     1968 samples  ~26.6 min
  3. Fluo-N2DH-GOWT1   2160 samples  ~19.2 min
  4. Fluo-N2DH-SIM+    2368 samples  ~21.1 min
  5. PhC-C2DH-U373     2712 samples  ~24.4 min
  6. PhC-C2DL-PSC      1192 samples  ~10.9 min  (4 seqs only)

Total estimated: ~1h 47min (BSGM, GPU, 256×256, batch_size=1)

Usage:
    python scripts/run_full_epoch.py
    python scripts/run_full_epoch.py --device cpu --img_size 256
    python scripts/run_full_epoch.py --datasets huh7 dhela   # subset
    python scripts/run_full_epoch.py --regen_coco            # force COCO rebuild
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT    = Path(__file__).parent.parent
SRC     = Path("F:/GitHub/99-CellTracktor/code-ubu2004/data")
DST     = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Dataset registry: key -> (src_tag, dst_tag, full_name, max_train_seqs)
DATASETS = {
    "huh7":  ("ctchuh7",  "ctc-huh7",  "Fluo-C2DL-Huh7",   None),
    "dhela": ("ctcdhela", "ctc-dhela", "DIC-C2DH-HeLa",     None),
    "gowt1": ("ctcgowt1", "ctc-gowt1", "Fluo-N2DH-GOWT1",  None),
    "sim":   ("ctcsim",   "ctc-sim",   "Fluo-N2DH-SIM+",   None),
    "u373":  ("ctcu373",  "ctc-u373",  "PhC-C2DH-U373",    None),
    "psc":   ("ctcpscv2", "ctc-psc",  "PhC-C2DL-PSC",     4),   # cap at 4 seqs
}

DATASET_ORDER = ["huh7", "dhela", "gowt1", "sim", "u373", "psc"]

# Tiny BSGM config (fast on CPU/GPU, proves the pipeline works)
BSGM_CFG = dict(
    backbone="swin_t", backbone_in_channels=3, swin_window_size=4,
    hidden_dim=256, nheads=8, enc_layers=1, dec_layers=2,
    dim_feedforward=256, dropout=0.0, num_feature_levels=4, dec_n_points=2,
    num_queries=20, num_classes=1, tracking=True, with_div=False,
    masks=True, mask_channels=32, bayesian_dropout=0.0, bayesian_eval=False,
    mamba_d_state=4, mamba_d_conv=4, graph_topk=4, graph_heads=2,
    two_stage=True, with_box_refine=True, dn_track=False,
)

LOSS_CFG = dict(
    num_classes=1, cls_loss_coef=4.0, bbox_loss_coef=5.0, giou_loss_coef=2.0,
    mask_loss_coef=5.0, dice_loss_coef=5.0, set_cost_class=1.0,
    set_cost_bbox=5.0, set_cost_giou=2.0, set_cost_mask=1.0,
    focal_alpha=0.25, focal_gamma=2.0, with_div=False, masks=True,
)


# ---------------------------------------------------------------------------
# COCO generation
# ---------------------------------------------------------------------------

def get_train_seqs(src_tag: str, max_seqs=None):
    train_dir = SRC / src_tag / "CTC" / "train"
    seqs = sorted([d.name for d in train_dir.iterdir()
                   if d.is_dir() and not d.name.endswith("_GT")])
    if max_seqs:
        seqs = seqs[:max_seqs]
    return seqs


def ensure_coco(src_tag: str, dst_tag: str, seqs: list, regen: bool) -> Path:
    """
    Generate (or reuse) COCO JSON for the given train sequences.
    img_dir = SRC/{src_tag}/CTC/train/   (relative file_name: {seq}/t000.tif)
    Returns path to JSON file.
    """
    from create_coco_from_ctc import process_split

    ann_dir  = DST / dst_tag / "COCO" / "annotations"
    ann_file = ann_dir / "instances_train_full.json"
    ann_dir.mkdir(parents=True, exist_ok=True)

    if ann_file.exists() and not regen:
        import json
        with open(ann_file) as f:
            d = json.load(f)
        n_img = len(d["images"])
        n_ann = len(d.get("annotations", []))
        print(f"    [缓存] instances_train_full.json 已存在 "
              f"({n_img} 张图, {n_ann} 标注) — 跳过生成")
        return ann_file

    print(f"    生成 COCO 标注，共 {len(seqs)} 个序列 ...")
    t0 = time.perf_counter()
    process_split(
        data_dir=str(SRC / src_tag),
        split="train",
        sequences=seqs,
        use_rle=False,
        out_filename="instances_train_full.json",
        out_dir=str(ann_dir),
    )
    elapsed = time.perf_counter() - t0
    print(f"    COCO 生成完毕 ({elapsed:.1f}s)")
    return ann_file


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_one_epoch(src_tag: str, dst_tag: str, name: str, seqs: list,
                  ann_file: Path, device: torch.device,
                  img_size: int, print_every: int) -> dict:
    from ai_cstq.datasets.ctc_coco import CTCCocoDataset, collate_fn
    from ai_cstq.models import build_model
    from ai_cstq.models.criterion import build_criterion
    from ai_cstq.engine import build_optimizer
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    img_dir = str(SRC / src_tag / "CTC" / "train")

    dataset = CTCCocoDataset(
        img_dir=img_dir,
        ann_file=str(ann_file),
        transforms=None,
        target_size=(img_size, img_size),
        in_channels=3,
        temporal_window=3,
        augment=False,
    )
    n_samples = len(dataset)
    print(f"    数据集: {n_samples} 个三元组样本")

    loader = DataLoader(dataset, batch_size=1, shuffle=True,
                        num_workers=2, collate_fn=collate_fn,
                        pin_memory=(device.type == "cuda"))

    model = build_model(BSGM_CFG).to(device)
    criterion = build_criterion(LOSS_CFG).to(device)
    optimizer = build_optimizer(
        model, {"lr": 1e-4, "lr_backbone": 1e-5, "weight_decay": 1e-4}
    )

    model.train()

    losses_total = []
    batch_times  = []
    n_err        = 0
    t_epoch_start = time.perf_counter()

    for i, (frames_list, targets) in enumerate(loader):
        frames  = [f.to(device) for f in frames_list]
        targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in t.items()} for t in targets]

        # Resize masks to match model's mask head output size
        # Probe on first batch only
        if i == 0:
            model.eval()
            with torch.no_grad():
                probe = model(frames)
            pm = probe.get("pred_masks")
            mask_h = pm.shape[-2] if pm is not None else img_size // 4
            mask_w = pm.shape[-1] if pm is not None else img_size // 4
            model.train()

        targets_r = []
        for t in targets:
            m = t["masks"].float()
            if m.shape[0] > 0:
                m = F.interpolate(m.unsqueeze(0), size=(mask_h, mask_w),
                                  mode="nearest").squeeze(0).bool()
            targets_r.append({**t, "masks": m})

        t0 = time.perf_counter()
        try:
            optimizer.zero_grad()
            out = model(frames, targets=targets_r)
            loss_dict = criterion(out, targets_r)
            total_loss = loss_dict["loss_total"]
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            losses_total.append(total_loss.item())
            batch_times.append(time.perf_counter() - t0)
        except Exception as e:
            n_err += 1
            if n_err <= 3:
                print(f"      [批次 {i} 错误] {e}")
            continue

        if (i + 1) % print_every == 0 or (i + 1) == n_samples:
            elapsed = time.perf_counter() - t_epoch_start
            remaining = elapsed / (i + 1) * (n_samples - i - 1)
            avg_loss = sum(losses_total[-print_every:]) / max(len(losses_total[-print_every:]), 1)
            avg_ms   = sum(batch_times[-print_every:]) / max(len(batch_times[-print_every:]), 1) * 1000
            print(f"    [{i+1:>5d}/{n_samples}]  "
                  f"loss={avg_loss:.3f}  "
                  f"{avg_ms:.0f}ms/batch  "
                  f"已用:{elapsed/60:.1f}min  剩余:~{remaining/60:.1f}min")

    t_total = time.perf_counter() - t_epoch_start
    avg_loss_final = sum(losses_total) / max(len(losses_total), 1)
    avg_ms_final   = sum(batch_times)  / max(len(batch_times),  1) * 1000

    return {
        "name":        name,
        "n_samples":   n_samples,
        "n_ok":        len(losses_total),
        "n_err":       n_err,
        "avg_loss":    avg_loss_final,
        "avg_ms":      avg_ms_final,
        "total_min":   t_total / 60,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
    }


# ---------------------------------------------------------------------------
# Patch create_coco_from_ctc to accept custom out_dir / out_filename
# ---------------------------------------------------------------------------

def _patch_coco_script():
    """
    create_coco_from_ctc.process_split doesn't accept out_dir/out_filename.
    Monkey-patch it to support that.
    """
    import create_coco_from_ctc as m
    import os, json
    from pathlib import Path as P

    _orig = m.process_split

    def patched(data_dir, split, sequences, use_rle,
                out_filename=None, out_dir=None):
        # Call the original but intercept the output path
        if out_filename is None and out_dir is None:
            return _orig(data_dir, split, sequences, use_rle)

        # Temporarily run in a way that writes to a custom location
        # by calling the internals directly
        import tifffile
        import numpy as np

        def read_tif(p):
            return tifffile.imread(str(p))

        ctc_dir = os.path.join(data_dir, "CTC", split)
        _out_dir = out_dir or os.path.join(data_dir, "COCO", "annotations")
        _out_fn  = out_filename or f"instances_{split}.json"
        os.makedirs(_out_dir, exist_ok=True)

        if sequences is None:
            sequences = sorted(
                d for d in os.listdir(ctc_dir)
                if os.path.isdir(os.path.join(ctc_dir, d))
                and not d.endswith("_GT") and not d.endswith("_RES")
            )

        images_list, annotations_list = [], []
        image_id = ann_id = 0

        for seq in sequences:
            seq_dir    = os.path.join(ctc_dir, seq)
            gt_tra_dir = os.path.join(ctc_dir, f"{seq}_GT", "TRA")
            if not os.path.isdir(gt_tra_dir):
                continue

            from pathlib import Path as PP
            frame_files = sorted(PP(seq_dir).glob("t*.tif"))
            print(f"      seq {seq}: {len(frame_files)} 帧")

            for fp in frame_files:
                fname = fp.stem
                t = int(fname[1:])
                raw = read_tif(fp)
                H, W = raw.shape[:2]
                img_fn = f"{seq}/{fname}.tif"
                images_list.append({
                    "id": image_id, "file_name": img_fn,
                    "height": H, "width": W, "seq": seq, "frame": t,
                })

                mask_fn = os.path.join(gt_tra_dir, f"man_track{t:03d}.tif")
                if os.path.isfile(mask_fn):
                    mask = read_tif(mask_fn).astype(np.uint16)
                    for cell_id in np.unique(mask):
                        if cell_id == 0:
                            continue
                        bm = (mask == cell_id).astype(np.uint8)
                        ys, xs = np.where(bm)
                        if len(ys) == 0:
                            continue
                        x0, x1 = int(xs.min()), int(xs.max())
                        y0, y1 = int(ys.min()), int(ys.max())
                        bw, bh = x1 - x0 + 1, y1 - y0 + 1
                        annotations_list.append({
                            "id": ann_id, "image_id": image_id,
                            "category_id": 1,
                            "bbox": [int(x0), int(y0), int(bw), int(bh)],
                            "area": int(bm.sum()),
                            "segmentation": [],
                            "iscrowd": 0,
                            "track_id": int(cell_id),   # cell_id == track_id in CTC format
                        })
                        ann_id += 1
                image_id += 1

        coco_out = {
            "info": {"description": f"CTC full {split}", "version": "1.0"},
            "categories": [{"id": 1, "name": "cell"}],
            "images": images_list,
            "annotations": annotations_list,
        }
        out_path = os.path.join(_out_dir, _out_fn)
        with open(out_path, "w") as f:
            json.dump(coco_out, f)
        print(f"    写入 {len(images_list)} 张图, "
              f"{len(annotations_list)} 标注 -> {out_path}")

    m.process_split = patched


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Full-epoch run on all 6 CTC datasets")
    p.add_argument("--device",      default="auto")
    p.add_argument("--img_size",    type=int, default=256)
    p.add_argument("--datasets",    nargs="+",
                   choices=list(DATASETS.keys()), default=DATASET_ORDER)
    p.add_argument("--print_every", type=int, default=100,
                   help="Print progress every N batches")
    p.add_argument("--regen_coco",  action="store_true",
                   help="Force COCO annotation regeneration")
    return p.parse_args()


def fmt_time(s):
    if s < 60:
        return f"{s:.0f}s"
    return f"{s/60:.1f} min"


def main():
    args = parse_args()
    _patch_coco_script()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print()
    print("=" * 72)
    print("  完整数据集训练（一个 Epoch） — jz-AI-CSTQ-v02")
    print("=" * 72)
    print(f"  设备: {device}  |  尺寸: {args.img_size}×{args.img_size}  "
          f"|  数据源: 99-CellTracktor")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()

    all_results = []
    t_grand_start = time.perf_counter()

    for key in args.datasets:
        src_tag, dst_tag, name, max_seqs = DATASETS[key]
        print(f"{'='*72}")
        print(f"  [{name}]  ({dst_tag})")
        print(f"{'='*72}")

        try:
            seqs = get_train_seqs(src_tag, max_seqs)
            print(f"    训练序列: {seqs[:4]}{'...' if len(seqs)>4 else ''} "
                  f"共 {len(seqs)} 条")

            ann_file = ensure_coco(src_tag, dst_tag, seqs, args.regen_coco)

            result = run_one_epoch(
                src_tag=src_tag, dst_tag=dst_tag, name=name,
                seqs=seqs, ann_file=ann_file,
                device=device, img_size=args.img_size,
                print_every=args.print_every,
            )
            all_results.append(result)
            print(f"\n  [{name}] Epoch 完成")
            print(f"    平均损失: {result['avg_loss']:.3f}  "
                  f"平均批次: {result['avg_ms']:.0f}ms  "
                  f"总耗时: {result['total_min']:.1f} min  "
                  f"错误批次: {result['n_err']}")

            # Save checkpoint
            ckpt_dir = ROOT / "results" / dst_tag
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / "checkpoint_epoch1.pth"
            torch.save({
                "model_state": result["model_state"],
                "cfg": BSGM_CFG,
                "epoch": 1,
                "avg_loss": result["avg_loss"],
                "dataset": name,
            }, ckpt_path)
            print(f"    Checkpoint 已保存: {ckpt_path}")
        except Exception:
            print(f"\n  [{name}] 失败:")
            traceback.print_exc()
            all_results.append({"name": name, "failed": True})

        print()

    # ---- Grand summary ----
    t_grand = time.perf_counter() - t_grand_start
    print("=" * 72)
    print("  汇总结果")
    print("=" * 72)
    print(f"  {'数据集':25s} {'样本':>7s} {'平均Loss':>10s} "
          f"{'ms/batch':>9s} {'总耗时':>10s} {'错误':>6s}")
    print(f"  {'-'*25} {'-'*7} {'-'*10} {'-'*9} {'-'*10} {'-'*6}")
    for r in all_results:
        if r.get("failed"):
            print(f"  {r['name']:25s} {'FAILED':>7s}")
        else:
            print(f"  {r['name']:25s} {r['n_samples']:>7d} "
                  f"{r['avg_loss']:>10.3f} {r['avg_ms']:>9.0f} "
                  f"{r['total_min']:>9.1f}m {r['n_err']:>6d}")
    print(f"\n  全部完成，总耗时: {t_grand/60:.1f} min")
    print("=" * 72)


if __name__ == "__main__":
    main()
