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
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT       = Path(__file__).parent.parent
EVAL_BINS  = Path("F:/GitHub/99-CellTracktor/EvaluationSoftware/Win")
sys.path.insert(0, str(ROOT / "src"))

from ai_cstq.models import build_model
from ai_cstq.util.ctc_io import predictions_to_ctc, run_ctc_eval

# Dataset registry: key -> (dst_tag, full_name)
DATASETS_ORDER = [
    ("huh7",  "ctc-huh7",  "Fluo-C2DL-Huh7"),
    ("dhela", "ctc-dhela", "DIC-C2DH-HeLa"),
    ("gowt1", "ctc-gowt1", "Fluo-N2DH-GOWT1"),
    ("sim",   "ctc-sim",   "Fluo-N2DH-SIM+"),
    ("u373",  "ctc-u373",  "PhC-C2DH-U373"),
    ("psc",   "ctc-psc",   "PhC-C2DL-PSC"),
]

IMG_SIZE   = 256     # must match training
IN_CHANNELS = 3
SEQ        = "01"
NUM_DIGITS = 3


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

MAX_TRACK_QUERIES = 20   # cap to model's num_queries to avoid explosion


@torch.no_grad()
def run_inference(model, frame_files, device, conf_threshold):
    target_size = (IMG_SIZE, IMG_SIZE)
    from PIL import Image as PILImage
    sample = np.array(PILImage.open(str(frame_files[0])))
    img_hw = sample.shape[:2]

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
        keep   = scores > conf_threshold

        n_track = track_hs.shape[1] if track_hs is not None else 0
        track_ids = torch.zeros(logits.shape[1], dtype=torch.long, device=device)

        # Propagate existing tracks
        for qi in range(n_track):
            if keep[qi] and qi in active:
                track_ids[qi] = active[qi]

        # Assign new IDs to new-object queries only
        for qi in range(n_track, logits.shape[1]):
            if keep[qi]:
                track_ids[qi] = next_tid
                active[qi] = next_tid
                next_tid += 1

        out["track_ids"] = track_ids

        # Build next-frame track queries: only from kept detections,
        # capped at MAX_TRACK_QUERIES (top by score to avoid explosion)
        kept = keep.nonzero(as_tuple=True)[0]
        if len(kept) > 0:
            # Sort kept by score descending, cap
            kept_scores = scores[kept]
            order = kept_scores.argsort(descending=True)
            kept = kept[order[:MAX_TRACK_QUERIES]]

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
                   default=[k for k, _, _ in DATASETS_ORDER],
                   help="Dataset keys to evaluate")
    p.add_argument("--conf_threshold", type=float, default=0.3)
    p.add_argument("--device", default="auto")
    p.add_argument("--ckpt_epoch", type=int, default=1,
                   help="Checkpoint epoch to load")
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

    for key, dst_tag, name in DATASETS_ORDER:
        if key not in args.datasets:
            continue

        print(f"{'='*68}")
        print(f"  [{name}]  ({dst_tag})")
        print(f"{'='*68}")

        ckpt_path = ROOT / "results" / dst_tag / f"checkpoint_epoch{args.ckpt_epoch}.pth"
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

        # ---- Test sequence ----
        test_seq_dir = ROOT / "data" / dst_tag / "CTC" / "test" / SEQ
        frame_files  = sorted(test_seq_dir.glob("t???.tif"))
        if not frame_files:
            print(f"  [跳过] 未找到测试帧: {test_seq_dir}")
            rows.append({"name": name, "TRA": float("nan"),
                         "SEG": float("nan"), "DET": float("nan"),
                         "note": "no_test_frames"})
            continue
        print(f"  测试帧: {len(frame_files)} 帧  ({test_seq_dir})")

        # ---- Inference ----
        print("  推理中 ...")
        try:
            t0 = time.perf_counter()
            all_outputs, img_hw = run_inference(
                model, frame_files, device, args.conf_threshold)
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
        parent_dir = ROOT / "data" / dst_tag / "CTC" / "test"
        res_dir    = parent_dir / f"{SEQ}_RES"
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
            sequence=SEQ,
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
