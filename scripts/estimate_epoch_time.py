"""
Scan all 6 full datasets from 99-CellTracktor and estimate one-epoch training time.

Uses measured per-batch GPU speed from run_real_datasets.py to project:
  estimated_time = n_triplets * per_batch_ms

Prints the estimate table, then exits (no actual training).
"""

import sys
from pathlib import Path

SRC = Path("F:/GitHub/99-CellTracktor/code-ubu2004/data")

# src_tag -> (dst_tag, full_name)
DATASETS_ORDER = [
    # sorted by total train frames (smallest first)
    ("ctchuh7",  "ctc-huh7",  "Fluo-C2DL-Huh7",   "huh7"),
    ("ctcdhela", "ctc-dhela", "DIC-C2DH-HeLa",     "dhela"),
    ("ctcgowt1", "ctc-gowt1", "Fluo-N2DH-GOWT1",  "gowt1"),
    ("ctcsim",   "ctc-sim",   "Fluo-N2DH-SIM+",   "sim"),
    ("ctcu373",  "ctc-u373",  "PhC-C2DH-U373",    "u373"),
    ("ctcpscv2", "ctc-psc",   "PhC-C2DL-PSC",     "psc"),
]

# Measured GPU forward+backward time (ms) per batch at 256×256
# from run_real_datasets.py results
MEASURED_MS = {
    "huh7":  521,
    "dhela": 811,
    "gowt1": 533,
    "sim":   534,
    "u373":  539,
    "psc":   548,
}

# PSC has ~1006 cells/frame — limit train sequences for manageable COCO JSON
PSC_MAX_TRAIN_SEQS = 4


def scan_dataset(src_tag: str, ds_key: str):
    """
    Returns dict with train sequence info.
    """
    result = {"seqs": [], "total_frames": 0, "cells_sample": 0}
    train_dir = SRC / src_tag / "CTC" / "train"
    if not train_dir.exists():
        return result

    seq_dirs = sorted([d for d in train_dir.iterdir()
                       if d.is_dir() and not d.name.endswith("_GT")])

    if ds_key == "psc":
        seq_dirs = seq_dirs[:PSC_MAX_TRAIN_SEQS]   # cap PSC

    for sd in seq_dirs:
        frames = sorted(sd.glob("t???.tif"))
        n = len(frames)
        tra_txt = train_dir / f"{sd.name}_GT" / "TRA" / "man_track.txt"
        if tra_txt.exists():
            lines = [l for l in tra_txt.read_text().splitlines()
                     if l.strip() and not l.startswith("#")]
            n_tracks = len(lines)
        else:
            n_tracks = 0
        result["seqs"].append((sd.name, n, n_tracks))
        result["total_frames"] += n

    # sample cells from first seq
    if result["seqs"]:
        _, _, n_cells = result["seqs"][0]
        result["cells_sample"] = n_cells

    return result


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f} min"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}min"


def main():
    print()
    print("=" * 74)
    print("  全量 CTC 数据集规模扫描 & 训练时间估算（BSGM，GPU，256×256）")
    print("=" * 74)
    print(f"  PSC 限制为前 {PSC_MAX_TRAIN_SEQS} 个训练序列（避免 COCO JSON 过大）")
    print()

    total_est_sec = 0
    rows = []

    for src_tag, dst_tag, name, key in DATASETS_ORDER:
        info = scan_dataset(src_tag, key)
        n_seqs = len(info["seqs"])
        total_frames = info["total_frames"]
        # CTCCocoDataset returns triplets: n - 2 per sequence
        n_triplets = sum(max(0, n - 2) for _, n, _ in info["seqs"])
        ms_per = MEASURED_MS.get(key, 600)
        est_sec = n_triplets * ms_per / 1000

        cells_per_seq = info["cells_sample"]
        total_est_sec += est_sec

        rows.append({
            "name":     name,
            "key":      key,
            "n_seqs":   n_seqs,
            "frames":   total_frames,
            "triplets": n_triplets,
            "cells":    cells_per_seq,
            "ms_batch": ms_per,
            "est_sec":  est_sec,
        })

        print(f"  {name}")
        print(f"    训练序列:  {n_seqs} 条  |  总帧数: {total_frames}  "
              f"|  三元组样本: {n_triplets}")
        print(f"    细胞/序列: ~{cells_per_seq}  |  "
              f"实测单批 {ms_per}ms  |  "
              f"估算耗时: {fmt_time(est_sec)}")
        print()

    print("=" * 74)
    print(f"  {'数据集':25s} {'序列':>5s} {'帧数':>7s} {'样本':>7s} "
          f"{'细胞/帧':>8s} {'估算时间':>10s}")
    print(f"  {'-'*25} {'-'*5} {'-'*7} {'-'*7} {'-'*8} {'-'*10}")
    for r in rows:
        print(f"  {r['name']:25s} {r['n_seqs']:>5d} {r['frames']:>7d} "
              f"{r['triplets']:>7d} {r['cells']:>8d} {fmt_time(r['est_sec']):>10s}")

    print(f"  {'合计':25s} {sum(r['n_seqs'] for r in rows):>5d} "
          f"{sum(r['frames'] for r in rows):>7d} "
          f"{sum(r['triplets'] for r in rows):>7d} "
          f"{'':>8s} {fmt_time(total_est_sec):>10s}")
    print("=" * 74)
    print()
    print("  注：以上为一个 epoch（BSGM 模型，GPU，256×256，batch_size=1）")
    print("  COCO 标注生成时间未计入（仅需一次，约 2–15 min/数据集）")
    print()

    return rows


if __name__ == "__main__":
    main()
