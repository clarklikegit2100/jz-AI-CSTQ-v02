"""
Sample real CTC data from F:/GitHub/99-CellTracktor/code-ubu2004/data
into F:/GitHub/jz-AI-CSTQ-v02/data/ for pipeline smoke-testing.

Source layout (99-CellTracktor):
    {src_tag}/CTC/{split}/{seq}/t{t:03d}.tif
    {src_tag}/CTC/{split}/{seq}_GT/TRA/man_track.txt
    {src_tag}/CTC/{split}/{seq}_GT/TRA/man_track{t:03d}.tif
    {src_tag}/CTC/{split}/{seq}_GT/SEG/man_seg{t:03d}.tif   (sparse)

Destination layout (jz-AI-CSTQ-v02):
    {dst_tag}/CTC/{split}/01/t{t:03d}.tif          <- always renamed to seq "01"
    {dst_tag}/CTC/{split}/01_GT/TRA/man_track.txt
    ...

For each dataset we sample:
    train: first seq (e.g. "01"), first N frames
    val  : first val seq,  first N frames
    test : first test seq, first N frames

Datasets (6):
    ctcdhela  -> DIC-C2DH-HeLa
    ctcsim    -> Fluo-N2DH-SIM+
    ctcgowt1  -> Fluo-N2DH-GOWT1
    ctchuh7   -> Fluo-C2DL-Huh7
    ctcu373   -> PhC-C2DH-U373
    ctcpscv2  -> PhC-C2DL-PSC

Usage:
    python scripts/prepare_data_from_ctracktor.py
    python scripts/prepare_data_from_ctracktor.py --n_frames 20
    python scripts/prepare_data_from_ctracktor.py --datasets dhela sim --n_frames 10
    python scripts/prepare_data_from_ctracktor.py --info
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
# Registry
# ---------------------------------------------------------------------------

SRC_ROOT = Path("F:/GitHub/99-CellTracktor/code-ubu2004/data")
DST_ROOT = Path("F:/GitHub/jz-AI-CSTQ-v02/data")

# key -> (src_tag, dst_tag, full_name)
DATASETS = {
    "dhela":  ("ctcdhela",  "ctc-dhela",  "DIC-C2DH-HeLa"),
    "sim":    ("ctcsim",    "ctc-sim",    "Fluo-N2DH-SIM+"),
    "gowt1":  ("ctcgowt1",  "ctc-gowt1",  "Fluo-N2DH-GOWT1"),
    "huh7":   ("ctchuh7",   "ctc-huh7",   "Fluo-C2DL-Huh7"),
    "u373":   ("ctcu373",   "ctc-u373",   "PhC-C2DH-U373"),
    "psc":    ("ctcpscv2",  "ctc-psc",    "PhC-C2DL-PSC"),
}


# ---------------------------------------------------------------------------
# man_track helpers (same as prepare_real_data.py)
# ---------------------------------------------------------------------------

def read_man_track(p: Path) -> List[Tuple[int, int, int, int]]:
    rows = []
    if not p.exists():
        return rows
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            parts = line.split()
            rows.append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])))
    return rows


def filter_man_track(rows, t_start: int, t_end: int):
    surviving = set()
    clipped = []
    for cid, ts, te, par in rows:
        if te < t_start or ts > t_end:
            continue
        surviving.add(cid)
        clipped.append((cid, max(ts, t_start) - t_start, min(te, t_end) - t_start, par))
    return [(cid, ts, te, par if par in surviving else 0)
            for cid, ts, te, par in clipped]


def write_man_track(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for cid, ts, te, par in rows:
            f.write(f"{cid} {ts} {te} {par}\n")


# ---------------------------------------------------------------------------
# Sequence discovery
# ---------------------------------------------------------------------------

def get_first_seq(src_split_dir: Path) -> Optional[str]:
    """Return the name of the first sequence directory in a split."""
    seqs = sorted([d.name for d in src_split_dir.iterdir()
                   if d.is_dir() and not d.name.endswith("_GT")])
    return seqs[0] if seqs else None


# ---------------------------------------------------------------------------
# Copy one sequence
# ---------------------------------------------------------------------------

def copy_seq(src_root_tag: Path, src_split: str, src_seq: str,
             dst_root_tag: Path, dst_split: str, dst_seq: str,
             n_frames: int, verbose: bool = True) -> dict:
    """
    Copy first n_frames from src into dst, renaming sequence to dst_seq.
    Returns stats dict.
    """
    src_img  = src_root_tag / "CTC" / src_split / src_seq
    src_tra  = src_root_tag / "CTC" / src_split / f"{src_seq}_GT" / "TRA"
    src_seg  = src_root_tag / "CTC" / src_split / f"{src_seq}_GT" / "SEG"

    dst_img  = dst_root_tag / "CTC" / dst_split / dst_seq
    dst_tra  = dst_root_tag / "CTC" / dst_split / f"{dst_seq}_GT" / "TRA"
    dst_seg  = dst_root_tag / "CTC" / dst_split / f"{dst_seq}_GT" / "SEG"

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_tra.mkdir(parents=True, exist_ok=True)

    # raw frames
    raw = sorted(src_img.glob("t???.tif"))
    if not raw:
        print(f"    [警告] 未找到原始帧: {src_img}")
        return {"frames": 0, "tracks": 0}

    for i, f in enumerate(raw[:n_frames]):
        shutil.copy2(f, dst_img / f"t{i:03d}.tif")
    frames_copied = min(len(raw), n_frames)

    # TRA masks
    tra_masks = sorted(src_tra.glob("man_track???.tif"))
    for i, f in enumerate(tra_masks[:n_frames]):
        shutil.copy2(f, dst_tra / f"man_track{i:03d}.tif")

    # man_track.txt (filter to window)
    rows = read_man_track(src_tra / "man_track.txt")
    filtered = filter_man_track(rows, 0, frames_copied - 1)
    write_man_track(dst_tra / "man_track.txt", filtered)

    # SEG masks (sparse — copy only frames within window)
    has_seg = False
    seg_count = 0
    if src_seg.exists():
        seg_files = sorted(src_seg.glob("man_seg???.tif"))
        for sf in seg_files:
            fidx = int(sf.stem.replace("man_seg", ""))
            if fidx < n_frames:
                dst_seg.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sf, dst_seg / f"man_seg{fidx:03d}.tif")
                seg_count += 1
                has_seg = True

    if verbose:
        seg_str = f", SEG={seg_count}" if has_seg else ""
        print(f"    [{dst_split}/{dst_seq}] {frames_copied} 帧, "
              f"{len(filtered)} 条轨迹{seg_str}")

    return {"frames": frames_copied, "tracks": len(filtered)}


# ---------------------------------------------------------------------------
# Dataset info (--info mode)
# ---------------------------------------------------------------------------

def print_info(src_tag: str, full_name: str):
    base = SRC_ROOT / src_tag / "CTC"
    for split in ("train", "val", "test"):
        sp_dir = base / split
        if not sp_dir.exists():
            continue
        seqs = sorted([d.name for d in sp_dir.iterdir()
                       if d.is_dir() and not d.name.endswith("_GT")])
        if not seqs:
            continue
        # Show first seq only
        seq = seqs[0]
        raw = sorted((sp_dir / seq).glob("t???.tif"))
        tra_txt = sp_dir / f"{seq}_GT" / "TRA" / "man_track.txt"
        n_tracks = len(read_man_track(tra_txt))
        try:
            img = read_tif(raw[0])
            shape_str = f"{img.shape} {img.dtype}"
        except Exception as e:
            shape_str = f"(读取失败)"
        print(f"  {full_name:25s} [{split:5s}] {len(seqs)} seqs, "
              f"seq{seq}: {len(raw)} 帧 / {n_tracks} 轨迹  {shape_str}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Prepare CellTracktor data for jz-AI-CSTQ-v02")
    p.add_argument("--n_frames", type=int, default=15,
                   help="Frames per split to sample (default: 15)")
    p.add_argument("--datasets", nargs="+",
                   choices=list(DATASETS.keys()), default=list(DATASETS.keys()),
                   help="Which datasets (default: all 6)")
    p.add_argument("--splits", nargs="+",
                   choices=["train", "val", "test"], default=["train", "val", "test"])
    p.add_argument("--info", action="store_true",
                   help="Just print dataset info, no copying")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing destination data")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 68)
    print("  CellTracktor 数据采样 -> jz-AI-CSTQ-v02")
    print("=" * 68)
    print(f"  源:    {SRC_ROOT}")
    print(f"  目标:  {DST_ROOT}")
    print(f"  采样帧数: {args.n_frames} 帧/split")
    print()

    if args.info:
        print("[ 数据集信息 ]")
        for key in args.datasets:
            src_tag, dst_tag, full_name = DATASETS[key]
            print_info(src_tag, full_name)
        return

    total = {"frames": 0, "tracks": 0, "datasets": 0}

    for key in args.datasets:
        src_tag, dst_tag, full_name = DATASETS[key]
        src_base = SRC_ROOT / src_tag
        dst_base = DST_ROOT / dst_tag

        print(f"[ {full_name}  ->  {dst_tag} ]")

        for split in args.splits:
            src_split_dir = src_base / "CTC" / split
            if not src_split_dir.exists():
                print(f"    [{split}] 跳过 — 不存在: {src_split_dir}")
                continue

            src_seq = get_first_seq(src_split_dir)
            if src_seq is None:
                print(f"    [{split}] 跳过 — 无序列目录")
                continue

            dst_seq = "01"  # always normalize to 01
            dst_img = dst_base / "CTC" / split / dst_seq
            if dst_img.exists() and not args.overwrite:
                existing = list(dst_img.glob("t???.tif"))
                print(f"    [{split}/01] 已存在 ({len(existing)} 帧)，"
                      f"跳过 (--overwrite 强制覆盖)")
                total["frames"] += len(existing)
                continue

            stats = copy_seq(src_base, split, src_seq,
                             dst_base, split, dst_seq,
                             args.n_frames)
            total["frames"]  += stats["frames"]
            total["tracks"]  += stats["tracks"]

        total["datasets"] += 1
        print()

    print("=" * 68)
    print(f"  完成！{total['datasets']} 个数据集，{total['frames']} 帧，"
          f"{total['tracks']} 条轨迹")
    print()
    print("  COCO 转换命令:")
    for key in args.datasets:
        _, dst_tag, _ = DATASETS[key]
        splits_arg = " ".join(args.splits)
        print(f"    python scripts/create_coco_from_ctc.py "
              f"--data_dir data/{dst_tag} --splits {splits_arg} --sequences 01")
    print("=" * 68)


if __name__ == "__main__":
    main()
