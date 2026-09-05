"""
Tiled inference and stitching (resolution_scaling_plan.md Phase 1, inference half).

Training now samples native-resolution 256x256 tiles from the full-resolution
CTC frame instead of downscaling the whole frame (see
`CTCCocoDataset.tile_size` in `datasets/ctc_coco.py`). A checkpoint trained
that way must also be *run* on native-resolution tiles at inference, or the
train/inference resolution would mismatch again. This module slides a
256x256 window with overlap over each full frame, runs the (untiled) model on
every window, and stitches per-tile detections back into full-frame
instances:

  1. `compute_tile_origins`   -- a grid of tile origins covering the frame,
     with the last row/column aligned to the frame edge (not overshooting).
  2. `center_weight_map`      -- a (ts, ts) weight, 1.0 in the tile core and
     ramped down over the overlap band at each edge, used to blend two tiles'
     mask probabilities for the same cell instead of a hard seam.
  3. `detect_frame_tiled`     -- runs the model on every tile of one frame
     (no track queries: each tile is independent, detection-only) and merges
     duplicate detections in the overlap bands by mask IoU + centroid
     distance into one full-frame instance per cell.
  4. `assign_track_ids`       -- cross-frame identity by Hungarian matching on
     mask IoU between consecutive frames' *merged* detections (a
     tracking-by-detection association, not the model's DN track-query
     mechanism -- track queries would need to be re-projected into whichever
     tile a track's box falls into next frame, which is unvalidated and out
     of scope here). Supports `max_age` frames of a missed detection before a
     track is retired, so a single dropped detection does not always start a
     new ID.

`run_tiled_inference` ties these together into the same per-frame output
dict shape (`pred_logits`, `pred_boxes`, `pred_masks`, `hs_embed`,
`track_ids`) that `predictions_to_ctc` (ctc_io.py) already consumes, so no
changes are needed downstream: each merged instance's mask is written as a
full-resolution (H, W) logit canvas (+10 inside / -10 outside the stitched
binary mask) so `predictions_to_ctc`'s own upsample-and-threshold step is a
no-op resize.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Tile geometry
# ---------------------------------------------------------------------------

def compute_tile_origins(H: int, W: int, tile_size: int, stride: int) -> List[Tuple[int, int]]:
    """Grid of (y0, x0) tile origins covering [0,H) x [0,W); last row/column
    is shifted to align with the frame edge instead of overshooting it."""
    def axis_origins(size: int) -> List[int]:
        if size <= tile_size:
            return [0]
        origins = list(range(0, size - tile_size + 1, stride))
        if origins[-1] != size - tile_size:
            origins.append(size - tile_size)
        return origins

    ys = axis_origins(H)
    xs = axis_origins(W)
    return [(y, x) for y in ys for x in xs]


def center_weight_map(tile_size: int, overlap: int) -> np.ndarray:
    """(tile_size, tile_size) float32 weight: 1.0 in the core, linearly
    ramped down to a 0.05 floor over `overlap` pixels at each edge. Used to
    favour the tile whose centre is closest to a seam-straddling cell when
    blending two tiles' mask probabilities."""
    ramp = np.ones(tile_size, dtype=np.float32)
    r = max(min(overlap, tile_size // 2), 1)
    for i in range(r):
        w = 0.05 + 0.95 * (i + 1) / r
        ramp[i] = min(ramp[i], w)
        ramp[tile_size - 1 - i] = min(ramp[tile_size - 1 - i], w)
    return ramp[:, None] * ramp[None, :]


# ---------------------------------------------------------------------------
# Per-frame tiled detection + duplicate merge
# ---------------------------------------------------------------------------

def _box_iou_xyxy(a: Tensor, b: Tensor) -> Tensor:
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


@torch.no_grad()
def detect_frame_tiled(
    model,
    full_frames: List[Tensor],       # 3 tensors (C, H, W): [prev, curr, next], on `device`
    img_hw: Tuple[int, int],
    device: torch.device,
    tile_size: int = 256,
    tile_overlap: int = 64,
    conf_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    merge_iou_thresh: float = 0.3,
    merge_centroid_frac: float = 0.3,
) -> List[Dict]:
    """
    Run the model on every tile of one frame and merge overlap-band
    duplicates. Returns a list of merged instances, each:
        {"box": (cx, cy, w, h) normalised to the full frame,
         "score": float, "hs": (d_model,) tensor,
         "mask_bool": (H, W) bool tensor}
    """
    H, W = img_hw
    ts = tile_size
    stride = max(ts - tile_overlap, 1)
    origins = compute_tile_origins(H, W, ts, stride)
    cweight = torch.from_numpy(center_weight_map(ts, tile_overlap)).to(device)

    raw = []  # dicts with tile-local prob patch + full-frame placement/box
    for (y0, x0) in origins:
        tile_imgs = [f[:, y0:y0 + ts, x0:x0 + ts].unsqueeze(0) for f in full_frames]
        out = model(tile_imgs)
        logits = out["pred_logits"][0]                     # (N, C+1)
        boxes = out["pred_boxes"][0]                        # (N, 4 or 8)
        scores = logits[:, 0].sigmoid()
        keep = (scores > conf_threshold).nonzero(as_tuple=True)[0]
        if keep.numel() == 0:
            continue
        pm = out["pred_masks"][0, keep]                      # (K, Hm, Wm)
        pm_up = F.interpolate(
            pm.unsqueeze(1).float(), size=(ts, ts), mode="bilinear", align_corners=False,
        )[:, 0].sigmoid()                                    # (K, ts, ts)
        for k, qi in enumerate(keep.tolist()):
            cx, cy, bw, bh = boxes[qi, :4].tolist()
            full_cx, full_cy = (x0 + cx * ts) / W, (y0 + cy * ts) / H
            full_bw, full_bh = bw * ts / W, bh * ts / H
            raw.append({
                "box_xyxy": torch.tensor([
                    full_cx - full_bw / 2, full_cy - full_bh / 2,
                    full_cx + full_bw / 2, full_cy + full_bh / 2,
                ]),
                "centre": (full_cx, full_cy),
                "prob": pm_up[k],           # (ts, ts) on device
                "y0": y0, "x0": x0,
                "score": scores[qi].item(),
                "hs": out["hs_embed"][0, qi].detach().cpu(),
            })

    if not raw:
        return []

    boxes_xyxy = torch.stack([r["box_xyxy"] for r in raw])
    iou = _box_iou_xyxy(boxes_xyxy, boxes_xyxy)
    centres = torch.tensor([r["centre"] for r in raw])
    cdist = torch.cdist(centres, centres) * max(H, W)  # back to pixel units-ish
    same = (iou > merge_iou_thresh) | (cdist < merge_centroid_frac * ts)

    order = sorted(range(len(raw)), key=lambda i: -raw[i]["score"])
    assigned = [-1] * len(raw)
    clusters: List[List[int]] = []
    for i in order:
        if assigned[i] != -1:
            continue
        cid = len(clusters)
        cluster = [i]
        assigned[i] = cid
        for j in order:
            if assigned[j] == -1 and same[i, j]:
                assigned[j] = cid
                cluster.append(j)
        clusters.append(cluster)

    merged = []
    for cluster in clusters:
        members = [raw[i] for i in cluster]
        y0s = [m["y0"] for m in members]
        x0s = [m["x0"] for m in members]
        uy0, ux0 = min(y0s), min(x0s)
        uy1, ux1 = max(y0s) + ts, max(x0s) + ts
        prob_sum = torch.zeros(uy1 - uy0, ux1 - ux0, device=device)
        weight_sum = torch.zeros_like(prob_sum)
        for m in members:
            ry, rx = m["y0"] - uy0, m["x0"] - ux0
            w = cweight
            prob_sum[ry:ry + ts, rx:rx + ts] += m["prob"] * w
            weight_sum[ry:ry + ts, rx:rx + ts] += w
        prob = prob_sum / weight_sum.clamp(min=1e-6)
        local_bin = prob > mask_threshold

        full_mask = torch.zeros(H, W, dtype=torch.bool, device=device)
        full_mask[uy0:uy1, ux0:ux1] = local_bin
        ys, xs = torch.where(full_mask)
        best = max(members, key=lambda m: m["score"])
        if len(ys) > 0:
            y1, y2 = ys.min().item(), ys.max().item() + 1
            x1, x2 = xs.min().item(), xs.max().item() + 1
            box = ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)
        else:
            bx = best["box_xyxy"]
            box = ((bx[0] + bx[2]).item() / 2, (bx[1] + bx[3]).item() / 2,
                   (bx[2] - bx[0]).item(), (bx[3] - bx[1]).item())
        merged.append({
            "box": box,
            "score": max(m["score"] for m in members),
            "hs": best["hs"],
            "mask_bool": full_mask.cpu(),
        })
    return merged


# ---------------------------------------------------------------------------
# Cross-frame identity (tracking-by-detection on the merged instances)
# ---------------------------------------------------------------------------

def assign_track_ids(
    frames_dets: List[List[Dict]],
    iou_thresh: float = 0.3,
    max_age: int = 0,
) -> List[List[int]]:
    """
    Hungarian-match merged detections frame-to-frame by mask IoU.

    A previous track that goes unmatched is kept alive (without appearing in
    the output for the missed frame) for up to `max_age` frames before it is
    retired; `max_age=0` retires it immediately (a missed detection always
    starts a new ID on re-detection), matching CTC's usual assumption that a
    track is not expected to have gaps.

    Returns a list (length T) of per-frame track-id lists, aligned with
    `frames_dets[t]`.
    """
    from scipy.optimize import linear_sum_assignment

    next_id = 1
    alive: List[Dict] = []   # {"id", "mask_bool", "missed"}
    all_ids: List[List[int]] = []

    for dets in frames_dets:
        n = len(dets)
        ids_t = [0] * n
        matched_alive = set()

        if alive and n:
            iou = np.zeros((len(alive), n), dtype=np.float32)
            for a, tr in enumerate(alive):
                pm = tr["mask_bool"]
                for b, d in enumerate(dets):
                    cm = d["mask_bool"]
                    inter = (pm & cm).sum().item()
                    union = (pm | cm).sum().item()
                    iou[a, b] = inter / union if union > 0 else 0.0
            ra, rb = linear_sum_assignment(-iou)
            for a, b in zip(ra, rb):
                if iou[a, b] >= iou_thresh:
                    ids_t[b] = alive[a]["id"]
                    matched_alive.add(a)

        for b in range(n):
            if ids_t[b] == 0:
                ids_t[b] = next_id
                next_id += 1

        new_alive = []
        for a, tr in enumerate(alive):
            if a in matched_alive:
                continue
            missed = tr["missed"] + 1
            if missed <= max_age:
                new_alive.append({**tr, "missed": missed})
        for b in range(n):
            new_alive.append({"id": ids_t[b], "mask_bool": dets[b]["mask_bool"], "missed": 0})
        alive = new_alive

        all_ids.append(ids_t)

    return all_ids


# ---------------------------------------------------------------------------
# Full sequence: tiled detection + stitching + tracking-by-detection
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_tiled_inference(
    model,
    frame_files,
    device: torch.device,
    in_channels: int = 3,
    tile_size: int = 256,
    tile_overlap: int = 64,
    conf_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    merge_iou_thresh: float = 0.3,
    merge_centroid_frac: float = 0.3,
    track_iou_thresh: float = 0.3,
    track_max_age: int = 0,
) -> Tuple[List[Dict], Tuple[int, int]]:
    """
    Full-sequence tiled inference. Returns (all_outputs, img_hw) in the same
    shape `predictions_to_ctc` expects, so it needs no changes:
    each output dict holds `pred_logits`/`pred_boxes`/`pred_masks`/`hs_embed`
    at full frame resolution (one "query" per merged instance) plus
    externally-assigned `track_ids`.
    """
    from PIL import Image

    sample_img = np.array(Image.open(frame_files[0]))
    H, W = sample_img.shape[:2]
    T = len(frame_files)

    def load_full(path) -> Tensor:
        img = np.array(Image.open(path)).astype(np.float32)
        if img.ndim == 2:
            img = img[:, :, None]
        if img.max() > 1.0:
            img = img / (img.max() + 1e-8)
        t = torch.from_numpy(img.transpose(2, 0, 1))
        if t.shape[0] == 1 and in_channels == 3:
            t = t.expand(3, -1, -1)
        return t.to(device)

    frames_tensors = [load_full(f) for f in frame_files]

    frames_dets: List[List[Dict]] = []
    for t_idx in range(T):
        i_prev, i_curr, i_next = max(0, t_idx - 1), t_idx, min(T - 1, t_idx + 1)
        clip = [frames_tensors[i_prev], frames_tensors[i_curr], frames_tensors[i_next]]
        dets = detect_frame_tiled(
            model, clip, (H, W), device,
            tile_size=tile_size, tile_overlap=tile_overlap,
            conf_threshold=conf_threshold, mask_threshold=mask_threshold,
            merge_iou_thresh=merge_iou_thresh, merge_centroid_frac=merge_centroid_frac,
        )
        frames_dets.append(dets)
        print(f"  Frame {t_idx + 1}/{T}  tiles merged -> {len(dets)} instances")

    all_track_ids = assign_track_ids(frames_dets, iou_thresh=track_iou_thresh, max_age=track_max_age)

    all_outputs = []
    for dets, ids_t in zip(frames_dets, all_track_ids):
        n = len(dets)
        if n == 0:
            # hs_embed's feature dim is irrelevant with zero rows; downstream
            # (predictions_to_ctc) never reads it in this branch.
            all_outputs.append({
                "pred_logits": torch.zeros(1, 0, 2),
                "pred_boxes": torch.zeros(1, 0, 4),
                "pred_masks": torch.zeros(1, 0, H, W),
                "hs_embed": torch.zeros(1, 0, 1),
                "track_ids": torch.zeros(0, dtype=torch.long),
            })
            continue
        scores = torch.tensor([d["score"] for d in dets]).clamp(1e-4, 1 - 1e-4)
        logit0 = torch.log(scores / (1 - scores))
        pred_logits = torch.stack([logit0, torch.zeros_like(logit0)], dim=-1).unsqueeze(0)
        pred_boxes = torch.tensor([d["box"] for d in dets]).unsqueeze(0)
        pred_masks = torch.stack([
            torch.where(d["mask_bool"], torch.tensor(10.0), torch.tensor(-10.0)) for d in dets
        ]).unsqueeze(0)
        hs_embed = torch.stack([d["hs"] for d in dets]).unsqueeze(0)
        all_outputs.append({
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "pred_masks": pred_masks,
            "hs_embed": hs_embed,
            "track_ids": torch.tensor(ids_t, dtype=torch.long),
        })

    return all_outputs, (H, W)
