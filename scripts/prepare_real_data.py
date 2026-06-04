"""
Prepare real CTC dataset samples for jz-AI-CSTQ-v02 training and testing.

Reads from: F:/GitHub/Deep_CSTQ_Datasets/src/data/{dataset}/{seq}/
Writes  to: F:/GitHub/jz-AI-CSTQ-v02/data/{dataset_tag}/CTC/{split}/{seq}/

Datasets sampled:
  Fluo-N2DH-SIM+    -> data/ctc-sim/
  Fluo-N2DH-GOWT1   -> data/ctc-gowt1/
  Fluo-C2DL-Huh7    -> data/ctc-huh7/
  PhC-C2DH-U373     -> data/ctc-u373/
  Fluo-C2DL-MSC     -> data/ctc-msc/

For each dataset:
  train split <- seq 01, first N frames
  val   split <- seq 02, first N frames

Usage:
    python scripts/prepare_real_data.py
    python scripts/prepare_real_data.py --n_frames 20 --datasets sim gowt1
    python scripts/prepare_real_data.py --n_frames 10 --list_info
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import tifffile
    def read_tif(p: Path) -> np.ndarray:
        return tifffile.imread(str(p))
    def write_tif(p: Path, arr: np.ndarray):
        tifffile.imwrite(str(p), arr)
except ImportError:
    from PIL import Image
    def read_tif(p: Path) -> np.ndarray:
        return np.array(Image.open(str(p)))
    def write_tif(p: Path, arr: np.ndarray):
        if arr.dtype == np.uint16:
            Image.fromarray(arr.astype(np.int32), mode="I").save(str(p))
        else:
            Image.fromarray(arr).save(str(p))


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

SRC_ROOT = Path("F:/GitHub/Deep_CSTQ_Datasets/src/data")
DST_ROOT = Path("F:/GitHub/jz-AI-CSTQ-v02/data")

DATASETS = {
    "sim":   ("Fluo-N2DH-SIM+",  "ctc-sim"),
    "gowt1": ("Fluo-N2DH-GOWT1", "ctc-gowt1"),
    "huh7":  ("Fluo-C2DL-Huh7",  "ctc-huh7"),
    "u373":  ("PhC-C2DH-U373",   "ctc-u373"),
    "msc":   ("Fluo-C2DL-MSC",   "ctc-msc"),
}

SPLIT_TO_SEQ = {
    "train": "01",
    "val":   "02",
}


# ---------------------------------------------------------------------------
# man_track.txt helpers
# ---------------------------------------------------------------------------

def read_man_track(txt_path: Path) -> List[Tuple[int, int, int, int]]:
    """Parse man_track.txt -> list of (cell_id, t_start, t_end, parent_id)."""
    rows = []
    if not txt_path.exists():
        return rows
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            rows.append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])))
    return rows


def filter_man_track(rows: List[Tuple[int, int, int, int]],
                     t_start: int, t_end: int) -> List[Tuple[int, int, int, int]]:
    """
    Keep only cells whose lifetime overlaps [t_start, t_end].
    Clip t_start/t_end to window. Reset parent to 0 if parent not in surviving set.
    """
    surviving_ids = set()
    clipped = []
    for cell_id, ts, te, par in rows:
        if te < t_start or ts > t_end:
            continue
        new_ts = max(ts, t_start) - t_start
        new_te = min(te, t_end) - t_start
        surviving_ids.add(cell_id)
        clipped.append((cell_id, new_ts, new_te, par))

    # Clear parent references to cells that were cut
    result = []
    for cell_id, ts, te, par in clipped:
        par_out = par if par in surviving_ids else 0
        result.append((cell_id, ts, te, par_out))
    return result


def write_man_track(txt_path: Path, rows: List[Tuple[int, int, int, int]]):
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w") as f:
        for cell_id, ts, te, par in rows:
            f.write(f"{cell_id} {ts} {te} {par}\n")


# ---------------------------------------------------------------------------
# Core copy logic
# ---------------------------------------------------------------------------

def copy_sequence(src_dataset: str, dst_tag: str, seq: str,
                  split: str, n_frames: int, verbose: bool = True) -> dict:
    """
    Copy n_frames from a CTC sequence into the project data directory.

    Returns a dict with stats: frames_copied, cells_in_window, has_seg.
    """
    src_seq   = SRC_ROOT / src_dataset / seq
    src_gt    = SRC_ROOT / src_dataset / f"{seq}_GT"
    src_tra   = src_gt / "TRA"
    src_seg   = src_gt / "SEG"

    dst_base  = DST_ROOT / dst_tag / "CTC" / split
    dst_seq   = dst_base / seq
    dst_tra   = dst_base / f"{seq}_GT" / "TRA"
    dst_seg   = dst_base / f"{seq}_GT" / "SEG"

    dst_seq.mkdir(parents=True, exist_ok=True)
    dst_tra.mkdir(parents=True, exist_ok=True)

    # Collect available raw frames
    raw_files = sorted(src_seq.glob("t???.tif"))
    if not raw_files:
        raw_files = sorted(src_seq.glob("t????.tif"))

    if not raw_files:
        print(f"  [警告] 未找到原始帧: {src_seq}")
        return {"frames_copied": 0, "cells": 0, "has_seg": False}

    frames_to_copy = raw_files[:n_frames]
    t_indices = []

    # Copy raw images
    for i, src_f in enumerate(frames_to_copy):
        dst_f = dst_seq / f"t{i:03d}.tif"
        shutil.copy2(src_f, dst_f)
        t_indices.append(i)

    frames_copied = len(t_indices)
    t_end_idx = frames_copied - 1

    # Copy TRA masks
    tra_files = sorted(src_tra.glob("man_track???.tif"))
    for i, src_f in enumerate(tra_files[:n_frames]):
        dst_f = dst_tra / f"man_track{i:03d}.tif"
        shutil.copy2(src_f, dst_f)

    # Filter and write man_track.txt
    txt_src = src_tra / "man_track.txt"
    track_rows = read_man_track(txt_src)
    filtered = filter_man_track(track_rows, 0, t_end_idx)
    write_man_track(dst_tra / "man_track.txt", filtered)
    n_cells = len(filtered)

    # Copy SEG masks (sparse — only copy ones that exist)
    seg_files = sorted(src_seg.glob("man_seg???.tif")) if src_seg.exists() else []
    has_seg = len(seg_files) > 0
    if has_seg:
        dst_seg.mkdir(parents=True, exist_ok=True)
        for src_f in seg_files:
            frame_idx = int(src_f.stem.replace("man_seg", ""))
            if frame_idx < n_frames:
                dst_f = dst_seg / f"man_seg{frame_idx:03d}.tif"
                shutil.copy2(src_f, dst_f)

    if verbose:
        seg_str = f", SEG={len([f for f in seg_files if int(f.stem.replace('man_seg',''))<n_frames])}帧" if has_seg else ""
        print(f"    [{split}/{seq}] {frames_copied} 帧, {n_cells} 个细胞{seg_str}  ->  {dst_base}")

    return {"frames_copied": frames_copied, "cells": n_cells, "has_seg": has_seg}


# ---------------------------------------------------------------------------
# Per-dataset info
# ---------------------------------------------------------------------------

def print_dataset_info(src_dataset: str, dst_tag: str):
    """Print frame counts and image shapes for one dataset."""
    for seq in ("01", "02"):
        src_seq = SRC_ROOT / src_dataset / seq
        raw = sorted(src_seq.glob("t???.tif"))
        if not raw:
            print(f"  {src_dataset}/{seq}: 未找到文件")
            continue
        try:
            img = read_tif(raw[0])
            shape_str = f"{img.shape} {img.dtype}"
        except Exception as e:
            shape_str = f"(读取失败: {e})"
        tra_txt = SRC_ROOT / src_dataset / f"{seq}_GT" / "TRA" / "man_track.txt"
        n_cells = len(read_man_track(tra_txt))
        has_seg = (SRC_ROOT / src_dataset / f"{seq}_GT" / "SEG").exists()
        print(f"  {src_dataset}/{seq}: {len(raw)} 帧, {n_cells} 条轨迹, "
              f"形状={shape_str}, SEG={'有' if has_seg else '无'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Prepare real CTC data for jz-AI-CSTQ-v02")
    p.add_argument("--n_frames", type=int, default=15,
                   help="Number of frames to sample per split (default: 15)")
    p.add_argument("--datasets", nargs="+",
                   choices=list(DATASETS.keys()), default=list(DATASETS.keys()),
                   help="Datasets to prepare (default: all 5)")
    p.add_argument("--splits", nargs="+", choices=["train", "val"],
                   default=["train", "val"])
    p.add_argument("--list_info", action="store_true",
                   help="Print dataset info without copying")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing destination directories")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 64)
    print("  CTC 真实数据采样工具 — jz-AI-CSTQ-v02")
    print("=" * 64)
    print(f"  源数据: {SRC_ROOT}")
    print(f"  目标:   {DST_ROOT}")
    print(f"  采样帧数: {args.n_frames}/split")
    print()

    if args.list_info:
        print("[ 数据集信息 ]")
        for key in args.datasets:
            src_name, dst_tag = DATASETS[key]
            print(f"\n  {src_name} -> {dst_tag}")
            print_dataset_info(src_name, dst_tag)
        return

    total_stats = {"frames": 0, "cells": 0, "datasets": 0}

    for key in args.datasets:
        src_name, dst_tag = DATASETS[key]
        print(f"[ {src_name}  ->  {dst_tag} ]")

        for split in args.splits:
            seq = SPLIT_TO_SEQ[split]
            src_seq = SRC_ROOT / src_name / seq

            if not src_seq.exists():
                print(f"    [{split}/{seq}] 跳过 — 源目录不存在: {src_seq}")
                continue

            dst_seq = DST_ROOT / dst_tag / "CTC" / split / seq
            if dst_seq.exists() and not args.overwrite:
                existing = list(dst_seq.glob("t???.tif"))
                print(f"    [{split}/{seq}] 已存在 ({len(existing)} 帧)，跳过 (--overwrite 强制覆盖)")
                total_stats["frames"] += len(existing)
                continue

            stats = copy_sequence(src_name, dst_tag, seq, split, args.n_frames)
            total_stats["frames"] += stats["frames_copied"]
            total_stats["cells"]  += stats["cells"]

        total_stats["datasets"] += 1
        print()

    print("=" * 64)
    print(f"  完成！共处理 {total_stats['datasets']} 个数据集")
    print(f"  总帧数: {total_stats['frames']}")
    print(f"  总细胞轨迹: {total_stats['cells']}")
    print()
    print("  接下来可运行:")
    for key in args.datasets:
        _, dst_tag = DATASETS[key]
        print(f"    python scripts/create_coco_from_ctc.py --data_dir data/{dst_tag} "
              f"--splits train val --sequences 01 02")
    print("=" * 64)


if __name__ == "__main__":
    main()
