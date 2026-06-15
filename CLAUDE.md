# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**BSGM-CellTrack** is an end-to-end deep learning cell tracking system for Cell Tracking Challenge (CTC) benchmarks. It replaces a prior Cell-TRACTR stub-based design with a full production implementation combining:
- **Swin-T backbone** (hierarchical window attention, 4 stages)
- **Temporal Mamba** (selective SSM for 3-frame fusion)
- **GATv2 graph layer** (kNN cell-graph attention per decoder layer)
- **Bayesian Dropout** (MC uncertainty at inference)
- **Deformable DETR** encoder/decoder (4 enc + 6 dec layers)
- **Instance mask head** (FPN pixel decoder → sigmoid)

Target metrics: TRA ≥ 0.94, SEG ≥ 0.88, DET ≥ 0.92.

## Common Commands

### Training

```bash
# Train on a specific dataset (real config, 24 epochs)
python train.py --config cfgs/ctchuh7_bsgm.yaml

# Resume from checkpoint
python train.py --config cfgs/ctchuh7_bsgm.yaml --resume results/ctc-huh7-real/checkpoint_last.pth

# Cloud (RunPod) — real HuH7 training with fixed focal_alpha
python scripts/retrain_huh7_real.py --epochs 24 --lr_drop 20
```

### Inference & Evaluation

```bash
# Generate CTC output (man_track.txt + mask .tif files)
python infer.py --config cfgs/ctchuh7_bsgm.yaml \
                --checkpoint results/ctc-huh7-real/checkpoint_epoch24.pth \
                --sequence data/ctc-huh7/CTC/test/01 \
                --output data/ctc-huh7/CTC/test/01_RES --eval

# Official CTC metrics (TRA/SEG/DET)
python scripts/evaluate_ctc.py --datasets huh7 \
       --ckpt_epoch 24 --ckpt_dir ctc-huh7-real \
       --conf_threshold 0.5 --max_track_queries 300

# Score distribution sanity check (debug classification collapse)
python scripts/diag_scores.py --ckpt results/ctc-huh7-real/checkpoint_epoch24.pth
```

### Smoke Tests & Data Prep

```bash
# Quick dry-run (synthetic data, tiny model, validates entire pipeline)
python scripts/dry_run.py

# One epoch on all 6 real datasets (tiny model, confirms data pipeline)
python scripts/run_full_epoch.py --datasets huh7 --epochs 1

# Convert CTC TIF sequences to COCO JSON annotations
python scripts/create_coco_from_ctc.py

# Interactive Gradio dashboard
python app.py
```

## Architecture

### Data Flow

```
Input: 3 consecutive frames [t-1, t, t+1]  (grayscale replicated to 3-channel)
→ Swin-T Backbone  →  C2/C3/C4/C5 feature pyramid
→ FPN Neck         →  P2/P3/P4/P5 (all 256-dim)
→ MultiScaleTemporalMamba  →  fuses T dimension via selective SSM
→ Deformable Encoder (4 layers)
→ BSGM Decoder (6 layers, each: CellGraphLayer → QueryMamba → Self-Attn → Deformable Cross-Attn → BayesianDropout → FFN)
→ Heads: classification, box (4D normal / 8D division), mask, uncertainty
→ Post-process: Hungarian matching → CTC format (man_track.txt + mask .tif)
```

### Key Source Files

| File | Role |
|---|---|
| `src/ai_cstq/models/bsgm_net.py` | Main model class `BSGMCellTrack` |
| `src/ai_cstq/models/swin_backbone.py` | Pure-PyTorch SwinTransformerBackbone |
| `src/ai_cstq/models/mamba_module.py` | `SelectiveSSM`, `MultiScaleTemporalMamba` |
| `src/ai_cstq/models/graph_layer.py` | `CellGraphLayer` (GATv2, kNN) |
| `src/ai_cstq/models/bsgm_decoder.py` | `BSGMDecoder`, `BayesianDropout` |
| `src/ai_cstq/models/criterion.py` | `SetCriterion` (focal + L1 + GIoU + mask dice) |
| `src/ai_cstq/models/matcher.py` | `HungarianMatcher` |
| `src/ai_cstq/engine.py` | `train()` loop, optimizer/scheduler builders |
| `src/ai_cstq/datasets/ctc_coco.py` | `CTCCocoDataset`, `collate_fn` |

### Three-Stage Training Schedule

| Stage | Epochs | Frozen | Active losses |
|---|---|---|---|
| Detection warmup | 0–7 | Backbone + tracking heads | cls + box + mask |
| Tracking | 8–19 | Backbone only | + division loss |
| Full joint | 20–23 | Nothing | + DN-track loss |

### Configuration System

Configs in `cfgs/` use YAML inheritance via `base_config: base_bsgm.yaml`. Dataset-specific YAMLs override only what differs. Key parameters:

- `backbone`: `swin_t` | `swin_s` | `swin_b`
- `num_queries`: 300–400 (real configs), 20 (dry-run/toy)
- `target_size`: `[512, 512]` for real training; `256` for smoke tests
- `use_amp: false` — AMP is **disabled** (matcher produces NaN under autocast)
- `mamba_d_state`: 16–32 (SSM state dimension)
- `graph_topk`: 16–20 (kNN neighbourhood per decoder layer)

### Data Paths (env-overridable for cloud)

The code reads two environment variables so Windows paths don't need editing on Linux pods:

```bash
export DEEP_CSTQ_DATA=/workspace/data/Deep_CSTQ_Datasets/src/output  # Deep_CSTQ augmented data
export CTRACKTOR_DATA=/workspace/data/99-CellTracktor/code-ubu2004/data  # raw CTC data
export CTC_EVAL_BINS=/workspace/eval_bins  # official CTC evaluation binaries
```

Defaults fall back to the original Windows paths (`F:/GitHub/...`).

## RunPod Cloud Training

See `docs/RUNPOD_REAL_TRAINING.md` for the full guide. Key points:
- VRAM needed: **~10–12 GB** for fp32 at 512²; minimum 16 GB recommended
- RTX 4090 baseline: **15–30 min/epoch → 24 epochs ≈ 6–12 h**
- Use `nohup python scripts/retrain_huh7_real.py ... > train.log 2>&1 &` so the job survives SSH disconnect
- Checkpoints land in `results/ctc-huh7-real/checkpoint_epoch{N}.pth`, resumable

## Datasets

Six CTC benchmarks supported (all `CTC/train/<seq>/t*.tif` + `<seq>_GT/TRA/man_track*.tif`):

| Tag | Full name | Source type |
|---|---|---|
| `huh7` | Fluo-C2DL-Huh7 | Deep_CSTQ_Datasets |
| `gowt1` | Fluo-N2DH-GOWT1 | Deep_CSTQ_Datasets |
| `u373` | PhC-C2DH-U373 | Deep_CSTQ_Datasets |
| `dhela` | DIC-C2DH-HeLa | 99-CellTracktor |
| `sim` | Fluo-N2DH-SIM+ | 99-CellTracktor |
| `psc` | PhC-C2DL-PSC | 99-CellTracktor |

CTC output format: `man_track.txt` (track_id, start, end, parent) + `man_track{NNN}.tif` (uint16 instance masks where pixel value = track_id).

## Known Issues / Constraints

- **AMP must stay off**: `use_amp: false` in all real configs — the Hungarian matcher overflows in fp16.
- **`focal_alpha` fix**: real training uses `focal_alpha=0.5` (not the default 0.25) to prevent classification collapse on the rare-cell class; this is applied automatically by `retrain_huh7_real.py`.
- **Mamba-SSM CUDA extension is optional**: code auto-falls back to pure-PyTorch if `mamba-ssm` is not installed.
- **`mamba_d_state=4, graph_topk=4`** in `run_full_epoch.py`'s `BSGM_CFG` is intentionally a toy/smoke-test config — not the real architecture.
