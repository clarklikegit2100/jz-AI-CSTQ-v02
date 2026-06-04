"""
BSGM-CellTrack Interactive Dashboard  —  Gradio UI (English)
Code language: English  |  Display language: English

Launch:
    conda activate jz-AI-CSTQ-v02
    python app.py
"""

import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

import gradio as gr
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

# ---------------------------------------------------------------------------
# Tiny model config used for the dry-run (mirrors dry_run.py Stage 3)
# ---------------------------------------------------------------------------
DRYRUN_CFG = dict(
    backbone="swin_t",
    backbone_in_channels=3,
    swin_window_size=4,
    hidden_dim=256,
    nheads=8,
    enc_layers=1,
    dec_layers=2,
    dim_feedforward=256,
    dropout=0.0,
    num_feature_levels=4,
    dec_n_points=2,
    num_queries=20,
    num_classes=1,
    tracking=True,
    with_div=False,
    masks=True,
    mask_channels=32,
    bayesian_dropout=0.0,
    bayesian_eval=False,
    mamba_d_state=4,
    mamba_d_conv=4,
    graph_topk=4,
    graph_heads=2,
    two_stage=True,
    with_box_refine=True,
    dn_track=False,
)

STAGE_LABELS = {
    "synthetic_data":     "Synthetic data generation",
    "coco_conversion":    "CTC → COCO conversion",
    "model_build":        "Model build",
    "forward_no_target":  "Forward pass (no targets)",
    "forward_with_loss":  "Forward pass + loss backward",
    "train_one_epoch":    "Training loop (warmup epoch)",
    "inference_ctc":      "Inference + CTC output",
}


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def make_log():
    lines = []
    def log(msg): lines.append(msg)
    def get(): return "\n".join(lines)
    return log, get


def run_stage_synthetic(n_frames, n_cells, img_size, log):
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from make_synthetic_ctc import generate
        data_dir = ROOT / "data" / "dryrun"
        t0 = time.perf_counter()
        generate(str(data_dir), int(n_frames), int(n_cells),
                 int(img_size), int(img_size),
                 sigma=max(8.0, int(img_size) / 32),
                 splits=["train", "val"], sequence="01")
        dt = time.perf_counter() - t0
        n = len(list((data_dir / "CTC" / "train" / "01").glob("t*.tif")))
        detail = f"{n} frames × {n_cells} cells at {img_size}×{img_size}"
        log(f"  Written: {n} frames to data/dryrun/")
        return True, detail, dt
    except Exception:
        return False, traceback.format_exc()[-200:], 0.0


def run_stage_coco(log):
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from create_coco_from_ctc import process_split
        data_dir = ROOT / "data" / "dryrun"
        t0 = time.perf_counter()
        for split in ["train", "val"]:
            process_split(str(data_dir), split, ["01"], use_rle=False)
        dt = time.perf_counter() - t0
        ann = data_dir / "COCO" / "annotations" / "instances_train.json"
        d = json.loads(ann.read_text())
        detail = f"{len(d['images'])} images, {len(d['annotations'])} annotations"
        log(f"  Annotations written to: {ann}")
        return True, detail, dt
    except Exception:
        return False, traceback.format_exc()[-200:], 0.0


def run_stage_model_build(log):
    try:
        from ai_cstq.models import build_model
        t0 = time.perf_counter()
        model = build_model(DRYRUN_CFG)
        dt = time.perf_counter() - t0
        n_total = sum(p.numel() for p in model.parameters()) / 1e6
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        detail = f"{n_total:.1f}M total params, {n_train:.1f}M trainable"
        log(f"  Model: {n_total:.1f}M params")
        return True, detail, dt, model
    except Exception:
        return False, traceback.format_exc()[-200:], 0.0, None


def run_stage_forward(model, img_size, log):
    try:
        import torch
        frames = [torch.randn(1, 3, int(img_size), int(img_size)) for _ in range(3)]
        model.eval()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(frames)
        dt = time.perf_counter() - t0
        mk = out.get("pred_masks")
        detail = (
            f"logits {tuple(out['pred_logits'].shape)}, "
            f"boxes {tuple(out['pred_boxes'].shape)}, "
            f"masks {tuple(mk.shape) if mk is not None else 'N/A'}, "
            f"uncertainty {tuple(out['uncertainty'].shape)}"
        )
        log(f"  Forward: {dt*1000:.0f} ms")
        return True, detail, dt
    except Exception:
        return False, traceback.format_exc()[-200:], 0.0


def run_stage_loss(model, n_cells, img_size, log):
    try:
        import torch
        from ai_cstq.models.criterion import build_criterion
        loss_cfg = dict(
            num_classes=1,
            cls_loss_coef=4.0, bbox_loss_coef=5.0, giou_loss_coef=2.0,
            mask_loss_coef=5.0, dice_loss_coef=5.0,
            set_cost_class=1.0, set_cost_bbox=5.0, set_cost_giou=2.0,
            set_cost_mask=1.0, focal_alpha=0.25, focal_gamma=2.0,
            with_div=False, masks=True,
        )
        criterion = build_criterion(loss_cfg)
        H = W = int(img_size)
        frames = [torch.randn(1, 3, H, W) for _ in range(3)]
        model.eval()
        with torch.no_grad():
            probe = model(frames)
        mk = probe.get("pred_masks")
        mh, mw = (mk.shape[-2], mk.shape[-1]) if mk is not None else (H // 4, W // 4)
        targets = []
        for _ in range(1):
            n = int(n_cells)
            boxes = (torch.rand(n, 4) * 0.4 + 0.1).clamp(0.01, 0.99)
            masks = torch.zeros(n, mh, mw)
            for k in range(n):
                cx, cy, bw, bh = boxes[k].tolist()
                x0, x1 = max(0, int((cx-bw/2)*mw)), min(mw, int((cx+bw/2)*mw))
                y0, y1 = max(0, int((cy-bh/2)*mh)), min(mh, int((cy+bh/2)*mh))
                masks[k, y0:y1, x0:x1] = 1.0
            targets.append({"labels": torch.zeros(n, dtype=torch.long),
                            "boxes": boxes, "masks": masks})
        model.train()
        optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optim.zero_grad()
        t0 = time.perf_counter()
        out = model(frames, targets=targets)
        total = sum(build_criterion(loss_cfg)(out, targets).values())
        total.backward()
        grad_norm = sum(p.grad.norm().item()**2
                        for p in model.parameters() if p.grad is not None) ** 0.5
        optim.step()
        dt = time.perf_counter() - t0
        detail = f"loss={total.item():.3f}, grad_norm={grad_norm:.1f}"
        log(f"  Loss: {total.item():.3f}  grad_norm: {grad_norm:.1f}")
        return True, detail, dt
    except Exception:
        return False, traceback.format_exc()[-200:], 0.0


def run_stage_train(model, img_size, log):
    try:
        import torch
        from torch.utils.data import DataLoader
        from ai_cstq.datasets.ctc_coco import CTCCocoDataset, collate_fn
        from ai_cstq.models.criterion import build_criterion
        from ai_cstq.engine import train_one_epoch, build_optimizer
        data_dir = ROOT / "data" / "dryrun"
        dataset = CTCCocoDataset(
            img_dir=str(data_dir / "CTC" / "train"),
            ann_file=str(data_dir / "COCO" / "annotations" / "instances_train.json"),
            target_size=(int(img_size), int(img_size)),
            in_channels=3, temporal_window=3, augment=False,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)
        loss_cfg = dict(
            num_classes=1,
            cls_loss_coef=4.0, bbox_loss_coef=5.0, giou_loss_coef=2.0,
            mask_loss_coef=5.0, dice_loss_coef=5.0,
            set_cost_class=1.0, set_cost_bbox=5.0, set_cost_giou=2.0,
            set_cost_mask=1.0, focal_alpha=0.25, focal_gamma=2.0,
            with_div=False, masks=True,
        )
        optimizer = build_optimizer(model, {"lr": 1e-4, "lr_backbone": 1e-5, "weight_decay": 1e-4})
        t0 = time.perf_counter()
        train_one_epoch(model, build_criterion(loss_cfg), loader, optimizer,
                        device=torch.device("cpu"), epoch=0,
                        max_norm=0.1, use_amp=False, print_freq=9999, stage="warmup")
        dt = time.perf_counter() - t0
        detail = f"{len(dataset)} samples in {dt:.1f}s"
        log(f"  Epoch done: {len(dataset)} samples / {dt:.1f}s")
        return True, detail, dt
    except Exception:
        return False, traceback.format_exc()[-200:], 0.0


def run_stage_infer(model, img_size, log):
    try:
        import torch
        from ai_cstq.util.ctc_io import predictions_to_ctc
        res_dir = ROOT / "data" / "dryrun" / "CTC" / "test_RES"
        res_dir.mkdir(parents=True, exist_ok=True)
        H = W = int(img_size)
        model.eval()
        all_outputs = []
        with torch.no_grad():
            for _ in range(3):
                frames = [torch.randn(1, 3, H, W) for _ in range(3)]
                out = model(frames)
                all_outputs.append({k: v.cpu() if isinstance(v, torch.Tensor) else v
                                    for k, v in out.items()})
        t0 = time.perf_counter()
        predictions_to_ctc(all_outputs, (H, W), str(res_dir),
                           conf_threshold=0.0, mask_threshold=0.3, start_frame=0)
        dt = time.perf_counter() - t0
        masks = list(res_dir.glob("mask*.tif"))
        detail = f"{len(masks)} mask files + man_track.txt"
        log(f"  CTC output written to: {res_dir}")
        return True, detail, dt
    except Exception:
        return False, traceback.format_exc()[-200:], 0.0


# ---------------------------------------------------------------------------
# Full dry-run pipeline
# ---------------------------------------------------------------------------

def full_dry_run(n_frames, n_cells, img_size, progress=gr.Progress()):
    import pandas as pd
    log, get_log = make_log()
    results = []
    model = None

    def record(key, ok, detail, dt):
        status = "✅ PASS" if ok else "❌ FAIL"
        results.append([STAGE_LABELS[key], status, detail, f"{dt:.1f}s"])

    keys = list(STAGE_LABELS.keys())
    for step, (key, label) in enumerate(STAGE_LABELS.items()):
        progress(step / len(keys), desc=f"Running: {label}...")

        if key == "synthetic_data":
            ok, detail, dt = run_stage_synthetic(n_frames, n_cells, img_size, log)
        elif key == "coco_conversion":
            ok, detail, dt = run_stage_coco(log)
        elif key == "model_build":
            ok, detail, dt, model = run_stage_model_build(log)
        elif key == "forward_no_target":
            ok, detail, dt = run_stage_forward(model, img_size, log) if model else (False, "model not built", 0.0)
        elif key == "forward_with_loss":
            ok, detail, dt = run_stage_loss(model, n_cells, img_size, log) if model else (False, "model not built", 0.0)
        elif key == "train_one_epoch":
            ok, detail, dt = run_stage_train(model, img_size, log) if model else (False, "model not built", 0.0)
        elif key == "inference_ctc":
            ok, detail, dt = run_stage_infer(model, img_size, log) if model else (False, "model not built", 0.0)

        record(key, ok, detail, dt)

    progress(1.0, desc="Done")
    n_pass = sum(1 for r in results if "PASS" in r[1])
    log(f"\nSummary: {n_pass}/{len(results)} passed")
    df = pd.DataFrame(results, columns=["Stage", "Status", "Details", "Time"])
    return df, get_log(), _model_card(model)


def _model_card(model):
    if model is None:
        return "*Model not built*"
    total = sum(p.numel() for p in model.parameters()) / 1e6
    train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return (
        f"**Backbone:** Swin-T (window={DRYRUN_CFG['swin_window_size']})\n\n"
        f"**FPN levels:** {DRYRUN_CFG['num_feature_levels']}\n\n"
        f"**Temporal Mamba:** d_state={DRYRUN_CFG['mamba_d_state']}\n\n"
        f"**Graph (GATv2):** k={DRYRUN_CFG['graph_topk']} neighbors\n\n"
        f"**Decoder:** {DRYRUN_CFG['dec_layers']} BSGM layers\n\n"
        f"**Queries:** {DRYRUN_CFG['num_queries']}\n\n"
        f"**Parameters:** {total:.1f}M total / {train:.1f}M trainable"
    )


# ---------------------------------------------------------------------------
# Data preview
# ---------------------------------------------------------------------------

def preview_data(n_frames, n_cells, img_size):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from make_synthetic_ctc import generate

    data_dir = ROOT / "data" / "dryrun_preview"
    generate(str(data_dir), int(n_frames), int(n_cells),
             int(img_size), int(img_size),
             sigma=max(8.0, int(img_size) / 32),
             splits=["train"], sequence="01")

    frame_dir = data_dir / "CTC" / "train" / "01"
    mask_dir  = data_dir / "CTC" / "train" / "01_GT" / "TRA"

    try:
        import tifffile
        read = lambda p: tifffile.imread(str(p))
    except ImportError:
        from PIL import Image as PILImage
        read = lambda p: np.array(PILImage.open(str(p)))

    frames = sorted(frame_dir.glob("t*.tif"))
    show_n = min(4, len(frames))
    fig, axes = plt.subplots(2, show_n, figsize=(3 * show_n, 6))
    fig.patch.set_facecolor("#1e1e2e")

    for col, fp in enumerate(frames[:show_n]):
        t_idx = int(fp.stem[1:])
        raw = read(fp).astype(float)
        raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
        mask_p = mask_dir / f"man_track{t_idx:03d}.tif"
        msk = (read(mask_p) > 0).astype(float) if mask_p.exists() else np.zeros_like(raw)
        for row, (data, title, cmap) in enumerate([
            (raw, f"Frame {t_idx:03d} (raw)", "gray"),
            (msk, f"Frame {t_idx:03d} (mask)", "hot"),
        ]):
            ax = axes[row][col]
            ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(title, color="white", fontsize=9)
            ax.axis("off")

    fig.suptitle(f"Synthetic data preview — {n_cells} cells, {img_size}×{img_size}",
                 color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    from PIL import Image as PILImage
    return PILImage.open(buf)


# ---------------------------------------------------------------------------
# Config file reader
# ---------------------------------------------------------------------------

def read_config():
    p = ROOT / "cfgs" / "dryrun_bsgm.yaml"
    return p.read_text(encoding="utf-8") if p.exists() else "# config not found"


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="BSGM-CellTrack Dashboard") as demo:

    gr.Markdown(
        """
        # BSGM-CellTrack — End-to-End CTC Cell Tracking Dashboard
        **B**ayesian · **S**win · **G**raph · **M**amba  |  Segmentation + Tracking simultaneously
        ---
        """
    )

    with gr.Tabs():

        # ── Dry Run ─────────────────────────────────────────────────────────
        with gr.Tab("Dry Run"):
            gr.Markdown("Run the full 7-stage smoke test on a tiny synthetic dataset.")
            with gr.Row():
                n_frames_sl = gr.Slider(5, 30, value=10, step=1,  label="Number of frames")
                n_cells_sl  = gr.Slider(3, 15, value=5,  step=1,  label="Number of cells")
                img_size_sl = gr.Slider(128, 512, value=256, step=64, label="Image size (px)")
            run_btn = gr.Button("Run Dry Run", variant="primary", size="lg")

            result_df = gr.Dataframe(
                headers=["Stage", "Status", "Details", "Time"],
                datatype=["str", "str", "str", "str"],
                interactive=False,
            )
            with gr.Row():
                with gr.Column(scale=2):
                    log_box = gr.Textbox(label="Run log", lines=10, interactive=False)
                with gr.Column(scale=1):
                    model_card = gr.Markdown("*Run the test to see model info*")

            run_btn.click(
                fn=full_dry_run,
                inputs=[n_frames_sl, n_cells_sl, img_size_sl],
                outputs=[result_df, log_box, model_card],
            )

        # ── Data Preview ────────────────────────────────────────────────────
        with gr.Tab("Data Preview"):
            gr.Markdown("Generate and visualise synthetic CTC frames (raw + instance mask).")
            with gr.Row():
                pv_frames = gr.Slider(4, 10, value=4, step=1,  label="Frames")
                pv_cells  = gr.Slider(2, 10, value=5, step=1,  label="Cells")
                pv_size   = gr.Slider(128, 512, value=256, step=64, label="Image size")
            pv_btn   = gr.Button("Generate Preview", variant="secondary")
            pv_image = gr.Image(label="Top: raw frames  |  Bottom: instance masks",
                                type="pil", interactive=False)
            pv_btn.click(fn=preview_data,
                         inputs=[pv_frames, pv_cells, pv_size],
                         outputs=[pv_image])

        # ── Architecture ────────────────────────────────────────────────────
        with gr.Tab("Architecture"):
            gr.Markdown(
                """
                ## BSGM-CellTrack Pipeline

                ```
                Frame triplet [t-1, t, t+1]
                        │
                        ▼
                ┌─────────────────────┐
                │  Swin-T Backbone    │  4-stage hierarchical ViT
                │  W-MSA / SW-MSA     │  Relative position bias
                └────────┬────────────┘
                         │ [C2, C3, C4, C5]
                         ▼
                ┌─────────────────────┐
                │     FPN Neck        │  Top-down lateral connections
                └────────┬────────────┘
                         │ [P2, P3, P4, P5]
                         ▼
                ┌─────────────────────┐
                │  Multi-Scale        │  Mamba SSM across T=3 frames
                │  Temporal Mamba     │  Pure PyTorch — no CUDA ext
                └────────┬────────────┘
                         ▼
                ┌─────────────────────┐
                │  Deformable Encoder │  Multi-scale self-attention
                └────────┬────────────┘
                         ▼
                ┌───────────────────────────────────────────┐
                │       BSGM Decoder  (×N layers)           │
                │  CellGraphLayer (GATv2, kNN)              │
                │  → QueryMamba → SelfAttn                  │
                │  → DeformableCrossAttn → FFN              │
                └────────┬──────────────────────────────────┘
                         │
                  ┌──────┴─────────┬──────────────┐
                  ▼                ▼               ▼
                Cls + Box       Mask Head    Uncertainty Head
                  head           (FPN)         (Bayesian)
                         │
                          ▼
                  CTC: man_track.txt + mask*.tif
                ```

                ### Cell-TRACTR stubs → BSGM-CellTrack real implementations

                | Component | Cell-TRACTR (stub) | BSGM-CellTrack (this project) |
                |-----------|-------------------|-------------------------------|
                | Backbone  | ResNet-50 | **Swin-T** (W-MSA / SW-MSA, relative position bias) |
                | Temporal  | Conv1D + kNN softmax | **Selective SSM (Mamba)** |
                | Graph     | Affinity matrix mixing | **GATv2** (kNN message passing) |
                | Uncertainty | Scalar MC Dropout | **BayesianDropout + mc_forward()** |
                | Segmentation | — | **FPNPixelDecoder + MaskHead** |
                """
            )

        # ── Config ──────────────────────────────────────────────────────────
        with gr.Tab("Config"):
            gr.Markdown("Dry-run configuration file (`cfgs/dryrun_bsgm.yaml`)")
            gr.Code(value=read_config(), language="yaml", interactive=False,
                    label="dryrun_bsgm.yaml")

        # ── How to Use ──────────────────────────────────────────────────────
        with gr.Tab("How to Use"):
            gr.Markdown(
                """
                ## Quick Start

                ```bash
                conda activate jz-AI-CSTQ-v02
                cd F:/GitHub/jz-AI-CSTQ-v02

                # Smoke test (CLI, Chinese terminal log)
                python scripts/dry_run.py --device cpu
                python scripts/dry_run.py --device cuda

                # Prepare real CTC data
                python scripts/create_coco_from_ctc.py \\
                    --data_dir data/ctchuh7 --splits train val

                # Train
                python train.py --config cfgs/ctchuh7_bsgm.yaml

                # Infer + eval
                python infer.py --config cfgs/ctchuh7_bsgm.yaml \\
                    --checkpoint results/ctchuh7_bsgm/checkpoint_best.pth \\
                    --sequence data/ctchuh7/CTC/test/01 \\
                    --output   data/ctchuh7/CTC/test/01_RES --eval
                ```

                ## Dataset configs

                | Config | Dataset | Notes |
                |--------|---------|-------|
                | `ctchuh7_bsgm.yaml`  | HuH7 hepatocyte | Default experiment |
                | `ctcgowt1_bsgm.yaml` | GFP-GOWT1 | Fluorescence stem cells |
                | `ctcsim_bsgm.yaml`   | SIM+ simulated | Synthetic benchmark |
                | `ctcpsc_bsgm.yaml`   | PSC stem cells | Dense colony |
                | `dryrun_bsgm.yaml`   | Synthetic | Smoke test |

                ## Target metrics

                | Metric | Target |
                |--------|--------|
                | TRA (tracking accuracy) | ≥ 0.94 |
                | SEG (segmentation)      | ≥ 0.88 |
                | DET (detection)         | ≥ 0.92 |
                | Inference speed         | ≥ 4 fps |
                """
            )

    gr.Markdown(
        "<center><sub>BSGM-CellTrack | Code: English | jz-AI-CSTQ-v02</sub></center>"
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
    )
