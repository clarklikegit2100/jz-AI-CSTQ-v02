"""
Convert CTC format datasets to COCO JSON annotation format.

CTC input format:
    data/{dataset}/CTC/{split}/{seq}/t{t:03d}.tif          — raw image
    data/{dataset}/CTC/{split}/{seq}_GT/TRA/man_track.txt   — track table
    data/{dataset}/CTC/{split}/{seq}_GT/TRA/man_track{t:03d}.tif — instance masks
    data/{dataset}/CTC/{split}/{seq}_GT/SEG/man_seg{t:03d}.tif   — seg masks (if available)

COCO output:
    data/{dataset}/COCO/annotations/instances_{split}.json

Usage:
    python scripts/create_coco_from_ctc.py \
        --data_dir  data/ctchuh7 \
        --splits    train val \
        --sequences 01 02
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ai_cstq.util.ctc_io import read_man_track

try:
    import tifffile
    def read_tif(p): return tifffile.imread(str(p))
except ImportError:
    from PIL import Image
    def read_tif(p): return np.array(Image.open(str(p)))

try:
    from pycocotools import mask as coco_mask
    HAS_COCO = True
except ImportError:
    HAS_COCO = False


def mask_to_polygon(binary_mask: np.ndarray):
    """Convert binary mask → list of polygon coordinate lists."""
    try:
        import cv2
        contours, _ = cv2.findContours(
            binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        polys = []
        for c in contours:
            if c.shape[0] >= 3:
                polys.append(c.flatten().tolist())
        return polys
    except ImportError:
        # Fallback: return empty (masks will be encoded as RLE)
        return []


def mask_to_rle(binary_mask: np.ndarray) -> dict:
    """Encode binary mask as COCO RLE."""
    if HAS_COCO:
        rle = coco_mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle
    # Fallback: uncompressed RLE
    flat = binary_mask.flatten(order="F").tolist()
    counts = []
    cur = 0
    for pixel in flat:
        if pixel == cur:
            counts[-1] += 1 if counts else counts.append(1)
        else:
            counts.append(1)
            cur = pixel
    return {"size": list(binary_mask.shape), "counts": counts}


def parse_args():
    p = argparse.ArgumentParser("CTC → COCO converter")
    p.add_argument("--data_dir", required=True, help="e.g. data/ctchuh7")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--sequences", nargs="+", default=None,
                   help="Sequence IDs to process (e.g. 01 02). None = auto-detect.")
    p.add_argument("--use_rle", action="store_true",
                   help="Encode masks as RLE instead of polygons (more compact)")
    return p.parse_args()


def process_split(
    data_dir: str,
    split: str,
    sequences,
    use_rle: bool,
):
    ctc_dir = os.path.join(data_dir, "CTC", split)
    out_dir = os.path.join(data_dir, "COCO", "annotations")
    os.makedirs(out_dir, exist_ok=True)

    if sequences is None:
        # Auto-detect: folders without _GT suffix
        sequences = sorted(
            d for d in os.listdir(ctc_dir)
            if os.path.isdir(os.path.join(ctc_dir, d)) and not d.endswith("_GT") and not d.endswith("_RES")
        )
    print(f"  Sequences ({split}): {sequences}")

    images_list = []
    annotations_list = []
    image_id = 0
    ann_id = 0

    for seq in sequences:
        seq_dir = os.path.join(ctc_dir, seq)
        gt_tra_dir = os.path.join(ctc_dir, f"{seq}_GT", "TRA")
        gt_seg_dir = os.path.join(ctc_dir, f"{seq}_GT", "SEG")

        if not os.path.isdir(gt_tra_dir):
            print(f"    [skip] no GT/TRA for seq {seq}")
            continue

        # Load tracking table
        track_file = os.path.join(gt_tra_dir, "man_track.txt")
        tracks = read_man_track(track_file) if os.path.isfile(track_file) else {}

        # Discover frames
        frame_files = sorted(Path(seq_dir).glob("t*.tif"))
        print(f"    Seq {seq}: {len(frame_files)} frames")

        for frame_path in frame_files:
            frame_name = frame_path.stem  # e.g. t000
            t = int(frame_name[1:])       # frame number

            # Image info
            raw = read_tif(frame_path)
            H, W = raw.shape[:2]
            img_filename = f"{seq}/{frame_name}.tif"

            img_info = {
                "id": image_id,
                "file_name": img_filename,
                "height": H,
                "width": W,
                "seq": seq,
                "frame": t,
            }
            images_list.append(img_info)

            # Load TRA mask (instance mask, pixel=track_id)
            tra_mask_path = os.path.join(gt_tra_dir, f"man_track{t:03d}.tif")
            if not os.path.isfile(tra_mask_path):
                image_id += 1
                continue

            tra_mask = read_tif(tra_mask_path).astype(np.uint16)
            instance_ids = np.unique(tra_mask)
            instance_ids = instance_ids[instance_ids > 0]

            for tid in instance_ids:
                bin_mask = (tra_mask == tid).astype(np.uint8)

                # Bounding box
                rows = np.any(bin_mask, axis=1)
                cols = np.any(bin_mask, axis=0)
                if not rows.any():
                    continue
                r_min, r_max = np.where(rows)[0][[0, -1]]
                c_min, c_max = np.where(cols)[0][[0, -1]]
                bw = int(c_max - c_min + 1)
                bh = int(r_max - r_min + 1)
                bbox = [int(c_min), int(r_min), bw, bh]
                area = int(bin_mask.sum())

                # Segmentation
                if use_rle:
                    seg = mask_to_rle(bin_mask)
                else:
                    polys = mask_to_polygon(bin_mask)
                    seg = polys if polys else mask_to_rle(bin_mask)

                # Parent ID from tracking table
                parent_id = 0
                if tid in tracks:
                    parent_id = tracks[tid][2]

                ann = {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "area": area,
                    "segmentation": seg,
                    "iscrowd": 0,
                    "track_id": int(tid),
                    "parent_id": int(parent_id),
                }
                annotations_list.append(ann)
                ann_id += 1

            image_id += 1

    coco_json = {
        "images": images_list,
        "annotations": annotations_list,
        "categories": [{"id": 1, "name": "cell", "supercategory": "cell"}],
    }
    out_path = os.path.join(out_dir, f"instances_{split}.json")
    with open(out_path, "w") as f:
        json.dump(coco_json, f)
    print(f"  Wrote {len(images_list)} images, {len(annotations_list)} annotations → {out_path}")


def main():
    args = parse_args()
    for split in args.splits:
        print(f"\nProcessing split: {split}")
        process_split(args.data_dir, split, args.sequences, args.use_rle)
    print("\nDone.")


if __name__ == "__main__":
    main()
