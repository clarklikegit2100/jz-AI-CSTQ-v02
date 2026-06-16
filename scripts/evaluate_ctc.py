"""
Compute official CTC TRA / SEG / DET scores for all 6 datasets.

Pipeline per dataset:
  1. Load BSGM model + checkpoint (results/{dst_tag}/checkpoint_epoch1.pth)
  2. Run inference on test/01 sequence (15 frames)
  3. Write predictions to CTC RES format  (data/{dst_tag}/CTC/test/01_RES/)
  4. Invoke TRAMeasure.exe / SEGMeasure.exe / DETMeasure.exe
  5. Print scores

CTC evaluation binary location:
  F:/GitHub/99-CellTracktor/EvaluationSoftware/Win/

Usage:
    python scripts/evaluate_ctc.py
    python scripts/evaluate_ctc.py --datasets huh7 sim
    python scripts/evaluate_ctc.py --conf_threshold 0.3
"""

import argparse
import sys
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT           = Path(__file__).parent.parent
# Env-overridable for cloud (Linux). On Linux set CTC_EVAL_BINS to the Linux
# binaries dir (EvaluationSoftware/Linux). Defaults keep local Windows behaviour.
EVAL_BINS      = Path(os.environ.get("CTC_EVAL_BINS", "F:/GitHub/99-CellTracktor/EvaluationSoftware/Win"))
DEEP_CSTQ_OUT  = Path(os.environ.get("DEEP_CSTQ_DATA", "F:/GitHub/Deep_CSTQ_Datasets/src/output"))
sys.path.insert(0, str(ROOT / "src"))

from ai_cstq.models import build_model
from ai_cstq.util.ctc_io import predictions_to_ctc, run_ctc_eval

# Dataset registry: key -> (dst_tag, full_name, test_src)
#   test_src="local"  : ROOT/data/{dst_tag}/CTC/test/01/
#   test_src="deep"   : DEEP_CSTQ_OUT/{ctc_name}/test/01/  (clean continuous GT)
DATASETS_ORDER = [
    ("huh7",  "ctc-huh7",  "Fluo-C2DL-Huh7",  "deep",  "Fluo-C2DL-Huh7"),
    ("dhela", "ctc-dhela", "DIC-C2DH-HeLa",   "local", None),
    ("gowt1", "ctc-gowt1", "Fluo-N2DH-GOWT1", "deep",  "Fluo-N2DH-GOWT1"),
    ("sim",   "ctc-sim",   "Fluo-N2DH-SIM+",  "local", None),
    ("u373",  "ctc-u373",  "PhC-C2DH-U373",   "deep",  "PhC-C2DH-U373"),
    ("psc",   "ctc-psc",   "PhC-C2DL-PSC",    "local", None),
]

IMG_SIZE    = 256     # must match training
IN_CHANNELS = 3
SEQ         = "01"
NUM_DIGITS  = 3


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------

def load_frame(path: Path, target_size, in_channels: int) -> torch.Tensor:
    from PIL import Image
    img = np.array(Image.open(str(path))).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, None]
    if img.max() > 1.0:
        img /= img.max() + 1e-8
    t = torch.from_numpy(img.transpose(2, 0, 1))   # (C, H, W)
    if t.shape[0] == 1 and in_channels == 3:
        t = t.expand(3, -1, -1).contiguous()
    if target_size:
        t = F.interpolate(t.unsqueeze(0), size=target_size,
                          mode="bilinear", align_corners=False).squeeze(0)
    return t


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

MAX_TRACK_QUERIES = 20   # default cap; override with --max_track_queries


@torch.no_grad()
def run_inference_iou_tracker(model, frame_files, device, conf_threshold,
                               gt_tra_dir=None, iou_link_thresh=0.05, top_k=None):
    """Detection-only inference + IoU Hungarian tracker post-processing.
    No track queries are passed to the model — each frame is detected independently,
    then detections are linked across frames via IoU-based Hungarian matching.
    This avoids FP track query accumulation from imprecise low-res boxes.
    """
    from scipy.optimize import linear_sum_assignment
    target_size = (IMG_SIZE, IMG_SIZE)
    from PIL import Image as PILImage
    sample = np.array(PILImage.open(str(frame_files[0])))
    img_hw = sample.shape[:2]
    if gt_tra_dir is not None:
        gt_masks = sorted(Path(gt_tra_dir).glob("man_track???.tif"))
        if gt_masks:
            gt_sample = np.array(PILImage.open(str(gt_masks[0])))
            if gt_sample.shape[:2] != img_hw:
                img_hw = gt_sample.shape[:2]

    frames_t = [load_frame(f, target_size, IN_CHANNELS).to(device) for f in frame_files]
    T = len(frames_t)
    model.eval()

    # Pass 1: detect each frame independently (no track queries)
    per_frame_dets = []   # list of {"boxes": (K,4), "scores": (K,), "out": dict}
    for t_idx in range(T):
        i_p, i_n = max(0, t_idx - 1), min(T - 1, t_idx + 1)
        frames = [frames_t[i_p].unsqueeze(0),
                  frames_t[t_idx].unsqueeze(0),
                  frames_t[i_n].unsqueeze(0)]
        out = model(frames)
        scores = out["pred_logits"][0, :, 0].sigmoid()
        keep   = scores > conf_threshold
        kept   = keep.nonzero(as_tuple=True)[0]
        kept_scores = scores[kept]
        order = kept_scores.argsort(descending=True)
        kept = kept[order]
        if top_k is not None:
            kept = kept[:top_k]
        boxes = out["pred_boxes"][0, kept, :4].cpu()
        per_frame_dets.append({
            "boxes": boxes,
            "scores": kept_scores[order].cpu(),
            "kept_idx": kept.cpu(),
            "out": {k: v.cpu() for k, v in out.items() if isinstance(v, torch.Tensor)},
        })
        print(f"    t={t_idx:03d}  检测到 {len(kept)} 个细胞  (det-only)")

    # Pass 2: Hungarian IoU link across frames
    def iou_matrix(boxes_a, boxes_b):
        n, m = len(boxes_a), len(boxes_b)
        if n == 0 or m == 0:
            return np.zeros((n, m))
        mat = np.zeros((n, m))
        for i, ba in enumerate(boxes_a):
            iou = _box_iou_single(ba, boxes_b)
            mat[i] = iou.numpy()
        return mat

    next_tid = 1
    active_boxes = None   # (K, 4) boxes from previous frame
    active_tids  = []     # track IDs from previous frame

    all_outputs = []
    for t_idx, det in enumerate(per_frame_dets):
        boxes   = det["boxes"]   # (K, 4)
        kept_idx = det["kept_idx"]
        out = det["out"]
        n_det = len(boxes)

        track_ids = torch.zeros(out["pred_logits"].shape[1], dtype=torch.long)

        if t_idx == 0 or active_boxes is None or len(active_tids) == 0:
            # First frame: assign fresh IDs
            new_tids = list(range(next_tid, next_tid + n_det))
            next_tid += n_det
        else:
            # Match current detections to previous active tracks via IoU
            iou_mat = iou_matrix(boxes, active_boxes)   # (n_det, n_prev)
            cost = 1.0 - iou_mat
            row_ind, col_ind = linear_sum_assignment(cost)
            new_tids = [0] * n_det
            used_prev = set()
            for r, c in zip(row_ind, col_ind):
                if iou_mat[r, c] >= iou_link_thresh:
                    new_tids[r] = active_tids[c]
                    used_prev.add(c)
            # Unmatched detections get new IDs
            for i in range(n_det):
                if new_tids[i] == 0:
                    new_tids[i] = next_tid
                    next_tid += 1

        for qi_pos, qi in enumerate(kept_idx.tolist()):
            track_ids[qi] = new_tids[qi_pos]

        out["track_ids"] = track_ids
        all_outputs.append(out)

        active_boxes = boxes
        active_tids  = new_tids

    return all_outputs, img_hw


def _box_iou_single(box_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """IoU of one cxcywh box vs many. Returns (N,) tensor."""
    def to_xyxy(b):
        x1 = b[..., 0] - b[..., 2] / 2
        y1 = b[..., 1] - b[..., 3] / 2
        x2 = b[..., 0] + b[..., 2] / 2
        y2 = b[..., 1] + b[..., 3] / 2
        return x1, y1, x2, y2
    ax1, ay1, ax2, ay2 = to_xyxy(box_a)
    bx1, by1, bx2, by2 = to_xyxy(boxes_b)
    inter_w = (torch.min(ax2, bx2) - torch.max(ax1, bx1)).clamp(0)
    inter_h = (torch.min(ay2, by2) - torch.max(ay1, by1)).clamp(0)
    inter   = inter_w * inter_h
    area_a  = (ax2 - ax1) * (ay2 - ay1)
    area_b  = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter).clamp(min=1e-6)


@torch.no_grad()
def run_inference(model, frame_files, device, conf_threshold, gt_tra_dir=None,
                  max_track_queries=MAX_TRACK_QUERIES,
                  track_threshold=None, new_det_iou_thresh=0.4):
    """
    track_threshold   : min score for an existing track to be propagated.
                        Defaults to conf_threshold + 0.05 to kill low-conf FP tracks.
    new_det_iou_thresh: suppress a new-detection query if its box overlaps an
                        existing track box by more than this IoU (avoids double-counting).
    """
    if track_threshold is None:
        track_threshold = min(conf_threshold + 0.05, 0.9)

    target_size = (IMG_SIZE, IMG_SIZE)
    from PIL import Image as PILImage
    sample = np.array(PILImage.open(str(frame_files[0])))
    img_hw = sample.shape[:2]

    # Use GT mask size when it differs from image size (avoids TRAMeasure incompatible-size error)
    if gt_tra_dir is not None:
        gt_masks = sorted(Path(gt_tra_dir).glob("man_track???.tif"))
        if gt_masks:
            gt_sample = np.array(PILImage.open(str(gt_masks[0])))
            if gt_sample.shape[:2] != img_hw:
                img_hw = gt_sample.shape[:2]

    frames_t = [load_frame(f, target_size, IN_CHANNELS).to(device)
                for f in frame_files]
    T = len(frames_t)

    all_outputs = []
    track_hs    = None
    track_boxes = None
    active      = {}    # query_idx -> track_id
    next_tid    = 1

    model.eval()
    for t_idx in range(T):
        i_p = max(0, t_idx - 1)
        i_n = min(T - 1, t_idx + 1)
        frames = [frames_t[i_p].unsqueeze(0),
                  frames_t[t_idx].unsqueeze(0),
                  frames_t[i_n].unsqueeze(0)]

        out = model(frames,
                    track_query_hs_embeds=track_hs,
                    track_query_boxes=track_boxes)

        logits = out["pred_logits"]           # (1, N, C+1)
        scores = logits[0, :, 0].sigmoid()
        boxes  = out["pred_boxes"][0, :, :4]  # (N, 4) cxcywh normalised

        n_track = track_hs.shape[1] if track_hs is not None else 0
        track_ids = torch.zeros(logits.shape[1], dtype=torch.long, device=device)

        # --- Track queries: propagate only if score >= track_threshold ---
        active_track_boxes = []
        for qi in range(n_track):
            if scores[qi] >= track_threshold and qi in active:
                track_ids[qi] = active[qi]
                active_track_boxes.append(boxes[qi])

        # --- New-detection queries: skip if overlapping an active track ---
        track_box_tensor = (torch.stack(active_track_boxes)
                            if active_track_boxes else None)
        for qi in range(n_track, logits.shape[1]):
            if scores[qi] < conf_threshold:
                continue
            if track_box_tensor is not None:
                iou = _box_iou_single(boxes[qi], track_box_tensor)
                if iou.max().item() > new_det_iou_thresh:
                    continue   # already covered by an active track
            track_ids[qi] = next_tid
            active[qi] = next_tid
            next_tid += 1

        out["track_ids"] = track_ids

        # Build next-frame track queries from ALL kept detections (track + new),
        # capped at max_track_queries by score
        keep = track_ids > 0
        kept = keep.nonzero(as_tuple=True)[0]
        if len(kept) > 0:
            kept_scores = scores[kept]
            order = kept_scores.argsort(descending=True)
            kept = kept[order[:max_track_queries]]

            track_hs    = out["hs_embed"][:, kept, :]
            track_boxes = out["pred_boxes"][:, kept, :4]
            new_active  = {}
            for new_qi, old_qi in enumerate(kept.tolist()):
                tid = track_ids[old_qi].item()
                if tid > 0:
                    new_active[new_qi] = tid
            active = new_active
        else:
            track_hs = track_boxes = None
            active = {}

        all_outputs.append({k: v.cpu() for k, v in out.items()
                            if isinstance(v, torch.Tensor)})
        n_det = keep.sum().item()
        print(f"    t={t_idx:03d}  检测到 {n_det} 个细胞  (track_q={len(kept) if len(kept)>0 else 0})")

    return all_outputs, img_hw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("CTC TRA/SEG/DET evaluation")
    p.add_argument("--datasets", nargs="+",
                   default=[row[0] for row in DATASETS_ORDER],
                   help="Dataset keys to evaluate")
    p.add_argument("--conf_threshold", type=float, default=0.3)
    p.add_argument("--device", default="auto")
    p.add_argument("--ckpt_epoch", type=int, default=1,
                   help="Checkpoint epoch to load")
    p.add_argument("--ckpt_dir", default=None,
                   help="Override results subdir holding checkpoints "
                        "(e.g. ctc-huh7-fixed). Defaults to each dataset's dst_tag.")
    p.add_argument("--max_track_queries", type=int, default=MAX_TRACK_QUERIES,
                   help="Cap on track queries propagated per frame (raise for high num_queries models)")
    p.add_argument("--track_threshold", type=float, default=None,
                   help="Min score to keep propagating an existing track (default: conf+0.05)")
    p.add_argument("--new_det_iou_thresh", type=float, default=0.4,
                   help="Suppress new-detection query if IoU with any active track > this (avoids double-count)")
    p.add_argument("--iou_tracker", action="store_true",
                   help="Detection-only mode: no track queries; link frames with IoU Hungarian tracker")
    p.add_argument("--iou_tracker_thresh", type=float, default=0.05,
                   help="Min IoU to link a detection to an existing track (default 0.05)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print()
    print("=" * 68)
    print("  CTC 官方评测  TRA / SEG / DET — jz-AI-CSTQ-v02")
    print("=" * 68)
    print(f"  设备: {device}  |  conf_threshold: {args.conf_threshold}")
    print(f"  评测二进制: {EVAL_BINS}")
    print()

    rows = []

    for key, dst_tag, name, test_src, ctc_name in DATASETS_ORDER:
        if key not in args.datasets:
            continue

        print(f"{'='*68}")
        print(f"  [{name}]  ({dst_tag})  [GT源: {test_src}]")
        print(f"{'='*68}")

        ckpt_subdir = args.ckpt_dir if args.ckpt_dir else dst_tag
        ckpt_path = ROOT / "results" / ckpt_subdir / f"checkpoint_epoch{args.ckpt_epoch}.pth"
        if not ckpt_path.exists():
            print(f"  [跳过] Checkpoint 不存在: {ckpt_path}")
            print(f"         先运行: python scripts/run_full_epoch.py --datasets {key}")
            rows.append({"name": name, "TRA": float("nan"),
                         "SEG": float("nan"), "DET": float("nan"),
                         "note": "no_ckpt"})
            continue

        # ---- Load model ----
        print(f"  加载 checkpoint: {ckpt_path.name}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg  = ckpt.get("cfg", {})
        model = build_model(cfg).to(device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval()
        print(f"  Epoch {ckpt.get('epoch', '?')}  "
              f"训练 loss={ckpt.get('avg_loss', 0):.3f}")

        # ---- Test sequence (local copy or Deep_CSTQ_Datasets) ----
        if test_src == "deep":
            test_parent = DEEP_CSTQ_OUT / ctc_name / "test"
            # Sequence numbers vary (e.g., 37-40); pick the first raw-image dir
            seq_candidates = sorted(
                d.name for d in test_parent.iterdir()
                if d.is_dir() and not d.name.endswith(("_GT", "_RES"))
            ) if test_parent.exists() else []
            seq = seq_candidates[0] if seq_candidates else SEQ
        else:
            test_parent = ROOT / "data" / dst_tag / "CTC" / "test"
            seq = SEQ
        test_seq_dir = test_parent / seq
        frame_files  = sorted(test_seq_dir.glob("t???.tif"))
        if not frame_files:
            print(f"  [跳过] 未找到测试帧: {test_seq_dir}")
            rows.append({"name": name, "TRA": float("nan"),
                         "SEG": float("nan"), "DET": float("nan"),
                         "note": "no_test_frames"})
            continue
        print(f"  测试帧: {len(frame_files)} 帧  ({test_seq_dir})")

        # ---- Inference ----
        gt_tra_dir = test_parent / f"{seq}_GT" / "TRA"
        print("  推理中 ...")
        try:
            t0 = time.perf_counter()
            if args.iou_tracker:
                all_outputs, img_hw = run_inference_iou_tracker(
                    model, frame_files, device, args.conf_threshold,
                    gt_tra_dir=str(gt_tra_dir) if gt_tra_dir.exists() else None,
                    iou_link_thresh=args.iou_tracker_thresh,
                    top_k=args.max_track_queries if args.max_track_queries != MAX_TRACK_QUERIES else None)
            else:
                all_outputs, img_hw = run_inference(
                    model, frame_files, device, args.conf_threshold,
                    gt_tra_dir=str(gt_tra_dir) if gt_tra_dir.exists() else None,
                    max_track_queries=args.max_track_queries,
                    track_threshold=args.track_threshold,
                    new_det_iou_thresh=args.new_det_iou_thresh)
            dt = time.perf_counter() - t0
            print(f"  推理完成，耗时 {dt:.1f}s")
        except Exception:
            print("  推理失败:")
            traceback.print_exc()
            rows.append({"name": name, "TRA": float("nan"),
                         "SEG": float("nan"), "DET": float("nan"),
                         "note": "infer_error"})
            continue

        # ---- Write RES ----
        parent_dir = test_parent
        res_dir    = parent_dir / f"{seq}_RES"
        res_dir.mkdir(parents=True, exist_ok=True)
        predictions_to_ctc(
            all_outputs=all_outputs,
            img_hw=img_hw,
            out_dir=str(res_dir),
            conf_threshold=args.conf_threshold,
            mask_threshold=0.5,
            start_frame=0,
        )
        n_masks = len(list(res_dir.glob("mask*.tif")))
        print(f"  RES 输出: {n_masks} 个掩码  ->  {res_dir}")

        # ---- CTC Evaluation ----
        print("  运行 CTC 评测 ...")
        scores = run_ctc_eval(
            parent_dir=str(parent_dir),
            sequence=seq,
            eval_binary_dir=str(EVAL_BINS),
            num_digits=NUM_DIGITS,
            metrics=["TRA", "SEG", "DET"],
        )
        rows.append({
            "name": name,
            "TRA": scores.get("TRA", float("nan")),
            "SEG": scores.get("SEG", float("nan")),
            "DET": scores.get("DET", float("nan")),
            "note": f"epoch{args.ckpt_epoch}",
        })
        print(f"  TRA={scores.get('TRA', 'nan'):.4f}  "
              f"SEG={scores.get('SEG', 'nan'):.4f}  "
              f"DET={scores.get('DET', 'nan'):.4f}")
        print()

    # ---- Summary ----
    print("=" * 68)
    print("  评测汇总")
    print("=" * 68)
    print(f"  {'数据集':25s} {'TRA':>8s} {'SEG':>8s} {'DET':>8s}  备注")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}  ----")
    for r in rows:
        def fmt(v):
            return f"{v:.4f}" if not (v != v) else "  N/A  "  # nan check
        print(f"  {r['name']:25s} {fmt(r['TRA']):>8s} "
              f"{fmt(r['SEG']):>8s} {fmt(r['DET']):>8s}  {r.get('note','')}")
    print("=" * 68)
    print()
    print("  说明：当前为 1 epoch 初始训练，分数低属正常。")
    print("  目标分数（20+ epoch，标准配置）：TRA ≥ 0.90，SEG ≥ 0.85")
    print()


if __name__ == "__main__":
    main()
