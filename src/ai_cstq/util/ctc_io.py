"""
CTC format I/O utilities for BSGM-CellTrack.

Handles reading/writing of:
  - man_track.txt: <track_id> <start_frame> <end_frame> <parent_id>
  - man_track{t:03d}.tif: uint16 instance masks (pixel value = track_id)
  - CTC evaluation: wrapping the binary evaluation tools
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


# ---------------------------------------------------------------------------
# man_track.txt I/O
# ---------------------------------------------------------------------------

def read_man_track(path: str) -> Dict[int, Tuple[int, int, int]]:
    """
    Read CTC man_track.txt.

    Returns
    -------
    tracks : dict[track_id → (start_frame, end_frame, parent_id)]
    """
    tracks = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                tid, t_start, t_end, parent = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                tracks[tid] = (t_start, t_end, parent)
    return tracks


def write_man_track(path: str, tracks: Dict[int, Tuple[int, int, int]]):
    """Write CTC man_track.txt."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for tid, (t_start, t_end, parent) in sorted(tracks.items()):
            f.write(f"{tid} {t_start} {t_end} {parent}\n")


# ---------------------------------------------------------------------------
# Mask tif I/O
# ---------------------------------------------------------------------------

def read_ctc_mask(path: str) -> np.ndarray:
    """Read CTC uint16 mask tif → (H, W) numpy array."""
    if HAS_TIFFFILE:
        return tifffile.imread(path).astype(np.uint16)
    try:
        from PIL import Image
        return np.array(Image.open(path), dtype=np.uint16)
    except Exception:
        raise ImportError("Install tifffile or Pillow to read CTC masks.")


def write_ctc_mask(path: str, mask: np.ndarray):
    """Write uint16 mask to CTC tif file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mask = mask.astype(np.uint16)
    if HAS_TIFFFILE:
        tifffile.imwrite(path, mask, compression=None)
    else:
        from PIL import Image
        Image.fromarray(mask).save(path)


# ---------------------------------------------------------------------------
# Prediction → CTC output
# ---------------------------------------------------------------------------

def predictions_to_ctc(
    all_outputs: List[Dict],
    img_hw: Tuple[int, int],
    out_dir: str,
    conf_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    start_frame: int = 0,
) -> Dict[int, Tuple[int, int, int]]:
    """
    Convert per-frame model outputs to CTC format.

    Parameters
    ----------
    all_outputs : list of T dicts from BSGMCellTrack.forward()
                  Each dict must contain:
                    'pred_logits': (B=1, N, C+1)
                    'pred_boxes':  (B=1, N, 4 or 8)
                    'pred_masks':  (B=1, N, H_m, W_m)   optional
                    'hs_embed':    (B=1, N, d_model)
                    'track_ids':   (N,) int or None
    img_hw      : (H, W) of the original image
    out_dir     : output directory for mask tifs and man_track.txt
    conf_threshold : detection score threshold
    mask_threshold : binary mask threshold (sigmoid)
    start_frame : first frame index

    Returns
    -------
    tracks : dict[track_id → (start_frame, end_frame, parent_id)]
    """
    H, W = img_hw
    tracks: Dict[int, Tuple[int, int, int]] = {}
    os.makedirs(out_dir, exist_ok=True)

    next_id = 1
    # Map: query_index → track_id  (maintained across frames)
    active_tracks: Dict[int, int] = {}

    for frame_idx, out in enumerate(all_outputs):
        t = frame_idx + start_frame
        logits = out["pred_logits"][0]       # (N, C+1)
        boxes = out["pred_boxes"][0]         # (N, 4 or 8)
        scores = logits[:, 0].sigmoid()      # cell class score
        keep = scores > conf_threshold

        # Instance mask (H, W) uint16
        frame_mask = np.zeros((H, W), dtype=np.uint16)

        if keep.any():
            # Assign track IDs
            track_ids_frame = out.get("track_ids", None)

            for qi in keep.nonzero(as_tuple=True)[0].tolist():
                # Determine track ID
                if track_ids_frame is not None and track_ids_frame[qi] > 0:
                    tid = int(track_ids_frame[qi])
                elif qi in active_tracks:
                    tid = active_tracks[qi]
                else:
                    tid = next_id
                    next_id += 1
                active_tracks[qi] = tid

                # Update track entry
                if tid not in tracks:
                    tracks[tid] = (t, t, 0)
                else:
                    t_start, _, parent = tracks[tid]
                    tracks[tid] = (t_start, t, parent)

                # Rasterise mask
                if "pred_masks" in out:
                    pred_mask_i = out["pred_masks"][0, qi]   # (H_m, W_m) logits
                    import torch.nn.functional as F
                    mask_bin = F.interpolate(
                        pred_mask_i.unsqueeze(0).unsqueeze(0).float(),
                        size=(H, W), mode="bilinear", align_corners=False
                    )[0, 0]
                    mask_bin = (mask_bin.sigmoid() > mask_threshold).cpu().numpy()
                    # Assign track_id to mask pixels (later ids overwrite earlier if overlapping)
                    frame_mask[mask_bin] = tid
                else:
                    # Fallback: draw box
                    cx, cy, bw, bh = boxes[qi, :4].cpu().tolist()
                    x1 = max(0, int((cx - bw / 2) * W))
                    y1 = max(0, int((cy - bh / 2) * H))
                    x2 = min(W, int((cx + bw / 2) * W))
                    y2 = min(H, int((cy + bh / 2) * H))
                    frame_mask[y1:y2, x1:x2] = tid

        # Write mask tif
        mask_path = os.path.join(out_dir, f"mask{t:03d}.tif")
        write_ctc_mask(mask_path, frame_mask)

    # Write man_track.txt
    track_path = os.path.join(out_dir, "man_track.txt")
    write_man_track(track_path, tracks)
    return tracks


# ---------------------------------------------------------------------------
# CTC evaluation wrapper
# ---------------------------------------------------------------------------

def run_ctc_eval(
    res_dir: str,
    gt_dir: str,
    eval_binary_dir: str,
    sequence: str,
    metrics: List[str] = ("TRA", "SEG", "DET"),
) -> Dict[str, float]:
    """
    Run official CTC evaluation binaries and parse results.

    Parameters
    ----------
    res_dir        : path to RES/ folder (model output masks)
    gt_dir         : path to GT/ folder (ground truth)
    eval_binary_dir: directory containing TRAMeasure, SEGMeasure, DETMeasure
    sequence       : sequence name (e.g., '01')
    metrics        : which metrics to compute

    Returns
    -------
    scores : dict of metric name → float score
    """
    import platform
    suffix = ".exe" if platform.system() == "Windows" else ""

    scores = {}
    for metric in metrics:
        binary = os.path.join(eval_binary_dir, f"{metric}Measure{suffix}")
        if not os.path.exists(binary):
            print(f"[ctc_io] Warning: {binary} not found, skipping {metric}.")
            scores[metric] = float("nan")
            continue
        try:
            cmd = [binary, res_dir, gt_dir, "3"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            # Parse score from last line like "TRA measure: 0.9321"
            for line in reversed(result.stdout.splitlines()):
                m = re.search(r"[\d.]+$", line)
                if m:
                    scores[metric] = float(m.group())
                    break
            else:
                scores[metric] = float("nan")
        except Exception as e:
            print(f"[ctc_io] Error running {metric}: {e}")
            scores[metric] = float("nan")
    return scores
