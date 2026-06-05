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

                # Rasterise mask; fall back to box when mask is all-zero
                mask_filled = False
                if "pred_masks" in out:
                    pred_mask_i = out["pred_masks"][0, qi]   # (H_m, W_m) logits
                    import torch.nn.functional as F
                    mask_bin = F.interpolate(
                        pred_mask_i.unsqueeze(0).unsqueeze(0).float(),
                        size=(H, W), mode="bilinear", align_corners=False
                    )[0, 0]
                    mask_bin = (mask_bin.sigmoid() > mask_threshold).cpu().numpy()
                    if mask_bin.any():
                        frame_mask[mask_bin] = tid
                        mask_filled = True
                if not mask_filled:
                    # Box fallback (mask empty or no pred_masks)
                    cx, cy, bw, bh = boxes[qi, :4].cpu().tolist()
                    x1 = max(0, int((cx - bw / 2) * W))
                    y1 = max(0, int((cy - bh / 2) * H))
                    x2 = min(W, int((cx + bw / 2) * W))
                    y2 = min(H, int((cy + bh / 2) * H))
                    if x2 > x1 and y2 > y1:
                        frame_mask[y1:y2, x1:x2] = tid

        # Write mask tif
        mask_path = os.path.join(out_dir, f"mask{t:03d}.tif")
        write_ctc_mask(mask_path, frame_mask)

    # Rebuild tracking table from actual mask pixels.
    # CTC requires each track to appear in EVERY frame from start to end.
    # Split intermittent tracks into consecutive segments; rewrite mask pixels
    # for non-first segments so res_track.txt IDs match mask pixel values.
    mask_files = sorted(Path(out_dir).glob("mask???.tif"))
    masks_by_frame: Dict[int, np.ndarray] = {}
    tid_frames: Dict[int, List[int]] = {}
    for mf in mask_files:
        t_frame = int(mf.stem.replace("mask", ""))
        m = read_ctc_mask(str(mf))
        masks_by_frame[t_frame] = m
        for tid in np.unique(m):
            if tid == 0:
                continue
            tid_frames.setdefault(tid, []).append(t_frame)

    max_id = max(tid_frames.keys(), default=0)
    next_seg_id = max_id + 1
    # remap[(old_tid, t_frame)] = new_tid  for frames needing pixel rewrite
    remap: Dict[Tuple[int, int], int] = {}
    verified_tracks: Dict[int, Tuple[int, int, int]] = {}

    for tid, frames in tid_frames.items():
        frames = sorted(set(frames))
        # Split into consecutive segments
        segments: List[Tuple[int, int]] = []
        seg_start = frames[0]
        prev = frames[0]
        for f in frames[1:]:
            if f == prev + 1:
                prev = f
            else:
                segments.append((seg_start, prev))
                seg_start = f
                prev = f
        segments.append((seg_start, prev))

        for i, (s, e) in enumerate(segments):
            if i == 0:
                new_tid = tid  # first segment keeps original ID
            else:
                new_tid = next_seg_id
                next_seg_id += 1
                for f in range(s, e + 1):
                    remap[(tid, f)] = new_tid
            verified_tracks[new_tid] = (s, e, 0)

    # Rewrite mask pixels where IDs were reassigned
    if remap:
        dirty_frames = set(f for (_, f) in remap)
        for t_frame in dirty_frames:
            m = masks_by_frame[t_frame]
            changed = False
            for (old_tid, tf), new_tid in remap.items():
                if tf == t_frame:
                    mask_pixels = m == old_tid
                    if mask_pixels.any():
                        m[mask_pixels] = new_tid
                        changed = True
            if changed:
                mf_path = os.path.join(out_dir, f"mask{t_frame:03d}.tif")
                write_ctc_mask(mf_path, m)

    # Write res_track.txt (CTC evaluation expects this name in RES dir)
    track_path = os.path.join(out_dir, "res_track.txt")
    write_man_track(track_path, verified_tracks)
    return verified_tracks


# ---------------------------------------------------------------------------
# CTC evaluation wrapper
# ---------------------------------------------------------------------------

def run_ctc_eval(
    parent_dir: str,
    sequence: str,
    eval_binary_dir: str,
    num_digits: int = 3,
    metrics: List[str] = ("TRA", "SEG", "DET"),
) -> Dict[str, float]:
    """
    Run official CTC evaluation binaries and parse results.

    Binary usage: TRAMeasure.exe <parent_dir> <sequence> <num_digits>
    parent_dir must contain both {sequence}_RES/ and {sequence}_GT/.

    Parameters
    ----------
    parent_dir     : directory containing {seq}_RES/ and {seq}_GT/
    sequence       : sequence name, e.g. '01'
    eval_binary_dir: directory containing TRAMeasure, SEGMeasure, DETMeasure
    num_digits     : frame number digits (3 for mask000.tif)
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
            cmd = [binary, str(parent_dir), str(sequence), str(num_digits)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            output = result.stdout + result.stderr
            # Parse score from lines like "TRA measure: 0.9321" or "SEG measure: 0.8800"
            found = False
            for line in reversed(output.splitlines()):
                m = re.search(r"measure[:\s]+([\d.]+)", line, re.IGNORECASE)
                if m:
                    scores[metric] = float(m.group(1))
                    found = True
                    break
            if not found:
                # Fallback: last float on last non-empty line
                for line in reversed(output.splitlines()):
                    m = re.search(r"[\d.]+$", line.strip())
                    if m:
                        try:
                            scores[metric] = float(m.group())
                            found = True
                            break
                        except ValueError:
                            pass
            if not found:
                scores[metric] = float("nan")
                print(f"[ctc_io] Could not parse {metric} score. Output:\n{output[:300]}")
        except Exception as e:
            print(f"[ctc_io] Error running {metric}: {e}")
            scores[metric] = float("nan")
    return scores
