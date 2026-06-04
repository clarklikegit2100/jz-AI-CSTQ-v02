"""
Generate a tiny synthetic CTC dataset for smoke-testing.

Produces N frames of HxW uint16 grayscale images with K Gaussian-blob cells
that perform a random walk.  Ground-truth instance masks and man_track.txt
are written in the standard CTC format so the rest of the pipeline
(create_coco_from_ctc → CTCCocoDataset → model) can run end-to-end.

Output layout (default: data/dryrun/CTC/train/):
    01/t000.tif … t009.tif          raw 16-bit images
    01_GT/TRA/man_track.txt          track table
    01_GT/TRA/man_track000.tif … 009.tif  instance masks (uint16)

Usage:
    python scripts/make_synthetic_ctc.py
    python scripts/make_synthetic_ctc.py --out_dir data/dryrun --n_frames 10 --n_cells 5
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# tifffile or Pillow for writing
try:
    import tifffile
    def save_tif(arr, path):
        tifffile.imwrite(str(path), arr)
except ImportError:
    from PIL import Image
    def save_tif(arr, path):
        # PIL doesn't support uint16; store as int32 ('I' mode) — uint16 values fit
        if arr.dtype == np.uint16:
            Image.fromarray(arr.astype(np.int32), mode="I").save(str(path))
        else:
            Image.fromarray(arr).save(str(path))


# ---------------------------------------------------------------------------
# Gaussian blob helper
# ---------------------------------------------------------------------------

def gaussian_blob(H, W, cy, cx, sigma=8.0, peak=30000):
    """Return a float32 array of shape (H, W) with a Gaussian centred at (cy, cx)."""
    yy, xx = np.ogrid[:H, :W]
    return (peak * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))).astype(np.float32)


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------

def generate(
    out_dir: str,
    n_frames: int = 10,
    n_cells: int = 5,
    H: int = 256,
    W: int = 256,
    sigma: float = 10.0,
    noise_std: float = 500.0,
    step_std: float = 6.0,
    splits=("train", "val"),
    sequence: str = "01",
):
    rng = np.random.default_rng(42)

    # Initial cell positions (cy, cx) within [sigma, H-sigma]
    positions = np.column_stack([
        rng.uniform(sigma * 2, H - sigma * 2, n_cells),
        rng.uniform(sigma * 2, W - sigma * 2, n_cells),
    ])

    for split in splits:
        seq_dir = Path(out_dir) / "CTC" / split / sequence
        tra_dir = Path(out_dir) / "CTC" / split / f"{sequence}_GT" / "TRA"
        seq_dir.mkdir(parents=True, exist_ok=True)
        tra_dir.mkdir(parents=True, exist_ok=True)

        # Independently jitter positions per split (so train ≠ val)
        pos = positions.copy()
        if split == "val":
            pos = pos + rng.normal(0, sigma, pos.shape)
            pos = np.clip(pos, sigma * 2, [H - sigma * 2, W - sigma * 2])

        frame_positions = []  # list[array(n_cells, 2)]

        for t in range(n_frames):
            # Random walk step
            if t > 0:
                pos = pos + rng.normal(0, step_std, pos.shape)
                pos = np.clip(pos, sigma * 2, [H - sigma * 2, W - sigma * 2])

            frame_positions.append(pos.copy())

            # --- Raw image ---
            raw = rng.normal(0, noise_std, (H, W)).astype(np.float32)
            raw = np.clip(raw, 0, None)
            for k in range(n_cells):
                cy, cx = pos[k]
                raw += gaussian_blob(H, W, cy, cx, sigma)
            raw = np.clip(raw, 0, 65535).astype(np.uint16)
            save_tif(raw, seq_dir / f"t{t:03d}.tif")

            # --- Instance mask (pixel value = track_id 1..n_cells) ---
            mask = np.zeros((H, W), dtype=np.uint16)
            for k in range(n_cells):
                cy, cx = int(round(pos[k, 0])), int(round(pos[k, 1]))
                r = max(1, int(sigma * 1.5))
                y0, y1 = max(0, cy - r), min(H, cy + r + 1)
                x0, x1 = max(0, cx - r), min(W, cx + r + 1)
                # Disk mask
                yy, xx = np.ogrid[y0:y1, x0:x1]
                disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
                mask[y0:y1, x0:x1][disk] = k + 1   # track_id = k+1
            save_tif(mask, tra_dir / f"man_track{t:03d}.tif")

        # --- man_track.txt ---
        # Format: track_id  start_frame  end_frame  parent_id
        with open(tra_dir / "man_track.txt", "w") as f:
            for k in range(n_cells):
                f.write(f"{k+1} 0 {n_frames - 1} 0\n")

        print(f"[{split}] {n_frames} frames, {n_cells} cells → {seq_dir}")

    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Synthetic CTC data generator")
    p.add_argument("--out_dir",  default="data/dryrun",  help="Root output dir")
    p.add_argument("--n_frames", type=int, default=10,   help="Number of frames")
    p.add_argument("--n_cells",  type=int, default=5,    help="Number of cells")
    p.add_argument("--height",   type=int, default=256)
    p.add_argument("--width",    type=int, default=256)
    p.add_argument("--sigma",    type=float, default=10.0, help="Cell radius (px)")
    p.add_argument("--splits",   nargs="+", default=["train", "val"])
    p.add_argument("--sequence", default="01")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Resolve relative paths from project root
    root = Path(__file__).parent.parent
    out = args.out_dir if os.path.isabs(args.out_dir) else str(root / args.out_dir)
    generate(
        out_dir=out,
        n_frames=args.n_frames,
        n_cells=args.n_cells,
        H=args.height,
        W=args.width,
        sigma=args.sigma,
        splits=args.splits,
        sequence=args.sequence,
    )
