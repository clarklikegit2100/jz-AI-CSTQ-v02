"""
BSGM-CellTrack inference script.

Runs inference on a CTC sequence, outputs:
  - man_track.txt (CTC tracking format)
  - mask{t:03d}.tif (instance segmentation masks)
  - (Optional) CTC evaluation scores

Usage:
    python infer.py --config cfgs/ctchuh7_bsgm.yaml \
                    --checkpoint results/ctchuh7_bsgm/checkpoint_best.pth \
                    --sequence data/ctchuh7/CTC/test/01 \
                    --output    data/ctchuh7/CTC/test/01_RES
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ai_cstq.models import build_model
from ai_cstq.util.ctc_io import predictions_to_ctc, run_ctc_eval
from ai_cstq.util.misc import load_checkpoint
from ai_cstq.util.tiled_inference import run_tiled_inference

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


def parse_args():
    p = argparse.ArgumentParser("BSGM-CellTrack inference")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sequence", required=True, help="Path to sequence folder (e.g., .../01/)")
    p.add_argument("--output", required=True, help="Path to output RES folder")
    p.add_argument("--device", default="cuda")
    p.add_argument("--conf_threshold", type=float, default=0.5)
    p.add_argument("--mc_samples", type=int, default=1,
                   help="MC Dropout samples for uncertainty (1 = deterministic)")
    p.add_argument("--eval", action="store_true", help="Run CTC evaluation after inference")
    p.add_argument("--gt_dir", default=None, help="GT directory for evaluation")
    p.add_argument("--eval_bin_dir", default=None, help="CTC binary evaluation tools directory")
    p.add_argument("--tile_size", type=int, default=None,
                   help="Enable tiled inference: slide a tile_size x tile_size window over "
                        "the full-resolution frame instead of downscaling it. Use the same "
                        "value the checkpoint was trained with (see cfg tile_size).")
    p.add_argument("--tile_overlap", type=int, default=64)
    p.add_argument("--merge_iou_thresh", type=float, default=0.3,
                   help="Mask IoU above which two tiles' detections are merged as one cell.")
    p.add_argument("--merge_centroid_frac", type=float, default=0.3,
                   help="Centroid distance (as a fraction of tile_size) below which two "
                        "tiles' detections are merged as one cell.")
    p.add_argument("--mask_roi_pad", type=float, default=0.3,
                   help="Confine each query's mask to its own predicted box padded by this "
                        "fraction on each side, so a degenerate mask cannot grow past a "
                        "plausible cell region.")
    p.add_argument("--track_iou_thresh", type=float, default=0.3,
                   help="Mask IoU above which a detection is matched to a previous-frame track.")
    p.add_argument("--track_max_age", type=int, default=0,
                   help="Frames a track may go undetected before it is retired.")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "base_config" in cfg:
        base_path = os.path.join(os.path.dirname(path), cfg.pop("base_config"))
        with open(base_path) as f:
            base = yaml.safe_load(f)
        base.update(cfg)
        cfg = base
    return cfg


def load_frame(path: str, target_size=None, in_channels: int = 3) -> torch.Tensor:
    """Load a single frame as a normalised float tensor."""
    img = np.array(Image.open(path))
    if img.ndim == 2:
        img = img[:, :, None]
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= (img.max() + 1e-8)

    t = torch.from_numpy(img.transpose(2, 0, 1))  # (C, H, W)
    if t.shape[0] == 1 and in_channels == 3:
        t = t.expand(3, -1, -1)

    if target_size is not None:
        import torch.nn.functional as F
        t = F.interpolate(t.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False).squeeze(0)
    return t


def get_frame_files(seq_dir: str):
    """Return sorted list of .tif frame files in a CTC sequence directory."""
    exts = (".tif", ".tiff", ".png", ".jpg")
    files = sorted(
        p for p in Path(seq_dir).iterdir()
        if p.suffix.lower() in exts and not p.name.startswith("man_")
    )
    return files


@torch.no_grad()
def run_inference(
    model,
    frame_files,
    device,
    target_size,
    in_channels: int,
    conf_threshold: float,
    mc_samples: int = 1,
):
    """
    Iterate through frames, maintain track queries across time.

    Returns
    -------
    all_outputs : list of T output dicts (with track_ids assigned)
    img_hw      : (H, W) of original images
    """
    # Determine original image size
    sample_img = np.array(Image.open(frame_files[0]))
    img_hw = sample_img.shape[:2]

    T = len(frame_files)
    all_outputs = []
    track_query_hs = None
    track_query_boxes = None
    next_track_id = 1
    active_queries: dict = {}   # query_idx → track_id

    # Preload all frames as tensors
    frames_tensors = [
        load_frame(str(f), target_size=target_size, in_channels=in_channels).to(device)
        for f in frame_files
    ]

    for t_idx in range(T):
        # Frame triplet [t-1, t, t+1] (clamp at boundaries)
        i_prev = max(0, t_idx - 1)
        i_curr = t_idx
        i_next = min(T - 1, t_idx + 1)
        frames = [
            frames_tensors[i_prev].unsqueeze(0),
            frames_tensors[i_curr].unsqueeze(0),
            frames_tensors[i_next].unsqueeze(0),
        ]

        if mc_samples > 1:
            out = model.mc_forward(
                frames,
                track_query_hs_embeds=track_query_hs,
                track_query_boxes=track_query_boxes,
                num_samples=mc_samples,
            )
        else:
            out = model(
                frames,
                track_query_hs_embeds=track_query_hs,
                track_query_boxes=track_query_boxes,
            )

        # Assign track IDs
        logits = out["pred_logits"]           # (1, N, C+1)
        scores = logits[0, :, 0].sigmoid()   # (N,)
        keep_mask = scores > conf_threshold

        track_ids = torch.zeros(logits.shape[1], dtype=torch.long, device=device)
        n_track_queries = track_query_hs.shape[1] if track_query_hs is not None else 0

        for qi in range(n_track_queries):
            if keep_mask[qi]:
                if qi in active_queries:
                    track_ids[qi] = active_queries[qi]
                # else: was a track query but not detected → track ends

        for qi in range(n_track_queries, logits.shape[1]):
            if keep_mask[qi]:
                track_ids[qi] = next_track_id
                active_queries[qi] = next_track_id
                next_track_id += 1

        out["track_ids"] = track_ids

        # Prepare track queries for next frame: only the kept queries
        kept_indices = keep_mask.nonzero(as_tuple=True)[0]
        if len(kept_indices) > 0:
            track_query_hs = out["hs_embed"][:, kept_indices, :]        # (1, K, d)
            track_query_boxes = out["pred_boxes"][:, kept_indices, :4]  # (1, K, 4)
            # Update active_queries for next frame
            new_active = {}
            for new_qi, old_qi in enumerate(kept_indices.tolist()):
                if track_ids[old_qi].item() > 0:
                    new_active[new_qi] = track_ids[old_qi].item()
            active_queries = new_active
        else:
            track_query_hs = None
            track_query_boxes = None
            active_queries = {}

        all_outputs.append({k: v.cpu() for k, v in out.items() if isinstance(v, torch.Tensor)})
        print(f"  Frame {t_idx+1}/{T}  detected: {keep_mask.sum().item()}")

    return all_outputs, img_hw


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build model
    print("Loading model...")
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, strict=False)
    model.eval()

    target_size = cfg.get("target_size", None)
    if isinstance(target_size, (list, tuple)) and len(target_size) == 2:
        target_size = tuple(target_size)
    elif isinstance(target_size, int):
        target_size = (target_size, target_size)

    in_channels = cfg.get("backbone_in_channels", 3)

    # Get frames
    frame_files = get_frame_files(args.sequence)
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {args.sequence}")
    print(f"Found {len(frame_files)} frames in {args.sequence}")

    # Run inference
    print("Running inference...")
    if args.tile_size:
        print(f"  tiled mode: tile_size={args.tile_size} overlap={args.tile_overlap}")
        all_outputs, img_hw = run_tiled_inference(
            model=model,
            frame_files=frame_files,
            device=device,
            in_channels=in_channels,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            conf_threshold=args.conf_threshold,
            merge_iou_thresh=args.merge_iou_thresh,
            merge_centroid_frac=args.merge_centroid_frac,
            mask_roi_pad=args.mask_roi_pad,
            track_iou_thresh=args.track_iou_thresh,
            track_max_age=args.track_max_age,
        )
    else:
        all_outputs, img_hw = run_inference(
            model=model,
            frame_files=frame_files,
            device=device,
            target_size=target_size,
            in_channels=in_channels,
            conf_threshold=args.conf_threshold,
            mc_samples=args.mc_samples,
        )

    # Write CTC output
    print(f"Writing CTC output to {args.output}...")
    os.makedirs(args.output, exist_ok=True)
    tracks = predictions_to_ctc(
        all_outputs=all_outputs,
        img_hw=img_hw,
        out_dir=args.output,
        conf_threshold=args.conf_threshold,
        mask_threshold=0.5,
        start_frame=0,
    )
    print(f"  Wrote {len(tracks)} tracks.")

    # Optional: CTC evaluation
    if args.eval and args.gt_dir and args.eval_bin_dir:
        print("Running CTC evaluation...")
        scores = run_ctc_eval(
            res_dir=args.output,
            gt_dir=args.gt_dir,
            eval_binary_dir=args.eval_bin_dir,
            sequence=Path(args.sequence).name,
        )
        for metric, score in scores.items():
            print(f"  {metric}: {score:.4f}")

    print("Done.")


if __name__ == "__main__":
    main()
