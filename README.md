# BSGM-CellTrack

**Bayesian Swin Graph Mamba Cell Tracker** — an end-to-end deep learning system for the [Cell Tracking Challenge](http://celltrackingchallenge.net/) (CTC) benchmarks.

Combines four complementary modules around a Deformable DETR backbone to produce simultaneous detection, segmentation, and multi-object tracking from raw microscopy images, with no segmentation mask input required:

| Module | Role |
|---|---|
| **Swin-T backbone** | Hierarchical shifted-window attention, 4 stages |
| **Temporal Mamba** | Selective SSM fusing 3 consecutive frames |
| **GATv2 graph layer** | kNN cell-graph attention inside each decoder layer |
| **Bayesian Dropout** | MC uncertainty estimation at inference |

Target metrics: **TRA ≥ 0.94 · SEG ≥ 0.88 · DET ≥ 0.92**

---

## Architecture

```
Input: [t-1, t, t+1]  (grayscale replicated to 3-ch)
  → Swin-T Backbone          C2/C3/C4/C5 feature pyramid
  → FPN Neck                 P2/P3/P4/P5, all 256-dim
  → MultiScaleTemporalMamba  3-frame SSM fusion
  → Deformable Encoder       4 layers (multi-scale attn)
  → BSGM Decoder  ×6        CellGraphLayer → QueryMamba
                              → Self-Attn → Deformable Cross-Attn
                              → BayesianDropout → FFN
  → Heads                    cls · box (4D/8D) · mask · uncertainty
  → Post-process             Hungarian match → CTC output
```

~50M parameters, fp32, no AMP (Hungarian matcher overflows in fp16).

---

## Installation

```bash
pip install -r requirements.txt
pip install -e .                   # installs the ai_cstq package

# Optional: COCO annotation tools
pip install pycocotools

# Optional: Mamba CUDA extension (falls back to pure-PyTorch if absent)
pip install mamba-ssm causal-conv1d
```

**Python ≥ 3.9, PyTorch ≥ 2.1.**  
On Blackwell GPUs (sm_120) use PyTorch 2.11+cu128:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

## Quick start

```bash
# Validate the entire pipeline in ~30 s on any GPU/CPU (synthetic data, tiny model)
python scripts/dry_run.py

# Interactive Gradio dashboard
python app.py
```

---

## Datasets

Six CTC benchmarks are supported. Set environment variables to point at your data roots:

```bash
export DEEP_CSTQ_DATA=/path/to/Deep_CSTQ_Datasets/src/output   # huh7, gowt1, u373
export CTRACKTOR_DATA=/path/to/99-CellTracktor/data             # dhela, sim, psc
export CTC_EVAL_BINS=/path/to/eval_bins                         # official CTC binaries
```

Defaults fall back to the original Windows paths in `cfgs/base_bsgm.yaml`.

| Tag | Dataset | Source |
|---|---|---|
| `huh7` | Fluo-C2DL-Huh7 | Deep_CSTQ_Datasets |
| `gowt1` | Fluo-N2DH-GOWT1 | Deep_CSTQ_Datasets |
| `u373` | PhC-C2DH-U373 | Deep_CSTQ_Datasets |
| `dhela` | DIC-C2DH-HeLa | 99-CellTracktor |
| `sim` | Fluo-N2DH-SIM+ | 99-CellTracktor |
| `psc` | PhC-C2DL-PSC | 99-CellTracktor |

Generate COCO JSON annotations from CTC TIF sequences:

```bash
python scripts/create_coco_from_ctc.py
```

---

## Training

### Smoke test — one epoch on real data (tiny model, confirms data pipeline)

```bash
python scripts/run_full_epoch.py --datasets huh7 --epochs 1
```

### Real training — Huh7, 24 epochs, full architecture

```bash
python train.py --config cfgs/ctchuh7_bsgm.yaml
```

### Resume from checkpoint

```bash
python train.py --config cfgs/ctchuh7_bsgm.yaml \
                --resume results/ctc-huh7-real/checkpoint_last.pth
```

### Cloud (RunPod) — real HuH7 training script with all fixes applied

```bash
export DEEP_CSTQ_DATA=/workspace/data/Deep_CSTQ_Datasets/src/output
python -u scripts/retrain_huh7_real.py --epochs 24 --lr_drop 20
```

Checkpoints are saved to `results/ctc-huh7-real/checkpoint_epoch{N}.pth` and auto-resume on restart.

#### Three-stage training schedule

| Stage | Epochs | Frozen | Active losses |
|---|---|---|---|
| Detection warmup | 0–7 | Backbone + tracking heads | cls · box · mask |
| Tracking | 8–19 | Backbone only | + division loss |
| Full joint | 20–23 | Nothing | + DN-track loss |

---

## Inference & Evaluation

```bash
# Generate CTC output (man_track.txt + mask .tif files)
python infer.py --config cfgs/ctchuh7_bsgm.yaml \
                --checkpoint results/ctc-huh7-real/checkpoint_epoch24.pth \
                --sequence data/ctc-huh7/CTC/test/01 \
                --output data/ctc-huh7/CTC/test/01_RES --eval

# Official CTC metrics (TRA / SEG / DET)
python scripts/evaluate_ctc.py --datasets huh7 \
       --ckpt_epoch 24 --ckpt_dir ctc-huh7-real \
       --conf_threshold 0.5 --max_track_queries 300

# Classification collapse diagnostic
python scripts/diag_scores.py --ckpt results/ctc-huh7-real/checkpoint_epoch24.pth
```

CTC output format: `man_track.txt` (track\_id, start, end, parent) + `man_track{NNN}.tif` (uint16 instance masks, pixel value = track\_id).

---

## Configuration

Configs live in `cfgs/` and use YAML inheritance (`base_config: base_bsgm.yaml`). Dataset YAMLs override only what differs from the base.

Key parameters:

| Parameter | Default | Notes |
|---|---|---|
| `backbone` | `swin_t` | `swin_t` \| `swin_s` \| `swin_b` |
| `target_size` | `[512, 512]` | `[256, 256]` when VRAM < 24 GB |
| `num_queries` | 300–400 | 20 for dry-run |
| `use_amp` | `false` | AMP **must stay off** — matcher NaNs in fp16 |
| `focal_alpha` | `0.5` | Real training uses 0.5 (prevents classification collapse) |
| `mamba_d_state` | 16–32 | SSM state dimension |
| `graph_topk` | 16–20 | kNN neighbourhood per decoder layer |

---

## Results

### Paradigm A — Tracking-by-Detection (segmentation mask required as input)

v2.0 Deep-CSTQ-MG (Mamba, GPU):

| Dataset | TRA | SEG | DET |
|---|---|---|---|
| Fluo-C2DL-Huh7 | 0.853 | 0.618 | 0.876 |
| Fluo-N2DH-GOWT1 seq01 | 0.983 | 0.869 | — |
| Fluo-N2DH-GOWT1 seq02 | 0.877 | 0.898 | 0.876 |
| PhC-C2DH-U373 | 0.926 | 0.727 | 0.930 |
| PhC-C2DL-PSC | 0.878 | 0.616 | 0.904 |
| Fluo-N2DH-SIM+ | 0.987–0.990 | 0.858–0.997 | 0.991–0.997 |

### Paradigm B — End-to-End (raw image only)

v3.0 Cell-TRACTR (ResNet50 + Deformable DETR, 24 epochs):

| Dataset | TRA avg | SEG avg | DET avg |
|---|---|---|---|
| Fluo-N2DH-GOWT1 | 0.593 | 0.157 | 0.611 |
| PhC-C2DH-U373 | 0.056 | 0.000 | 0.058 |
| DIC-C2DH-HeLa | 0.443 | — | 0.444 |
| Fluo-N2DH-SIM+ | 0.474 | ~0.345 | ~0.476 |

v3.2 BSGM-CellTrack (this project): training in progress.

---

## Project Structure

```
src/ai_cstq/
  models/
    bsgm_net.py          main model class (BSGMCellTrack)
    swin_backbone.py     SwinTransformerBackbone + FPNNeck
    mamba_module.py      SelectiveSSM, MultiScaleTemporalMamba
    graph_layer.py       CellGraphLayer (GATv2, kNN)
    bsgm_decoder.py      BSGMDecoder, BayesianDropout
    criterion.py         SetCriterion (focal + L1 + GIoU + mask dice)
    matcher.py           HungarianMatcher
  datasets/
    ctc_coco.py          CTCCocoDataset, collate_fn
  engine.py              train() loop, optimizer/scheduler builders
  util/
    ctc_io.py            CTC TIF I/O helpers
cfgs/                    YAML configs (base + per-dataset)
scripts/                 Training, evaluation, and data-prep utilities
docs/                    RUNPOD_REAL_TRAINING.md and other guides
```

---

## Cloud Training (RunPod)

See [docs/RUNPOD_REAL_TRAINING.md](docs/RUNPOD_REAL_TRAINING.md) for the full guide.

- VRAM needed: ~10–12 GB at 256²; 24 GB recommended for 512²
- RTX 4090 at 512²: ~15–30 min/epoch → 24 epochs ≈ 6–12 h
- Use `nohup python -u scripts/retrain_huh7_real.py ... > train.log 2>&1 &` to survive SSH disconnect
- The script auto-resumes from the latest `checkpoint_epoch{N}.pth` on restart

---

## Project Registration

```yaml
project_id: jz-ai-cstq-v02
name: BSGM-CellTrack
description: >
  End-to-end deep learning cell tracker for Cell Tracking Challenge (CTC) benchmarks.
  Bayesian Swin-T backbone + Temporal Mamba + GATv2 graph attention + Deformable DETR —
  simultaneous detection, segmentation, and multi-object tracking from raw microscopy images.
version: v3.2

paths:
  primary: "%Github%\\jz-AI-CSTQ-v02"
  sibling:  "%Github%\\jz-AI-CSTQ"

type: python-ml

conda_env: jz-AI-CSTQ-v02

python_requires: ">=3.9"
gpu_vram_min_gb: 6      # dry-run / smoke test
gpu_vram_rec_gb: 24     # full 512² training

commands:
  dry-run:   "python scripts/dry_run.py"
  train:     "python train.py --config cfgs/ctchuh7_bsgm.yaml"
  resume:    "python train.py --config cfgs/ctchuh7_bsgm.yaml --resume results/ctc-huh7-real/checkpoint_last.pth"
  retrain:   "python scripts/retrain_huh7_real.py --epochs 24 --lr_drop 20"
  infer:     "python infer.py --config cfgs/ctchuh7_bsgm.yaml --checkpoint <ckpt> --sequence <seq_dir> --output <out_dir>"
  evaluate:  "python scripts/evaluate_ctc.py --datasets huh7 --ckpt_epoch 24 --ckpt_dir ctc-huh7-real"
  diag:      "python scripts/diag_scores.py --ckpt <ckpt>"
  app:       "python app.py"
  prep-data: "python scripts/create_coco_from_ctc.py"

datasets:
  - id: huh7    name: Fluo-C2DL-Huh7      source: Deep_CSTQ_Datasets
  - id: gowt1   name: Fluo-N2DH-GOWT1     source: Deep_CSTQ_Datasets
  - id: u373    name: PhC-C2DH-U373        source: Deep_CSTQ_Datasets
  - id: dhela   name: DIC-C2DH-HeLa        source: 99-CellTracktor
  - id: sim     name: Fluo-N2DH-SIM+       source: 99-CellTracktor
  - id: psc     name: PhC-C2DL-PSC         source: 99-CellTracktor

env_vars:
  DEEP_CSTQ_DATA: "/path/to/Deep_CSTQ_Datasets/src/output"
  CTRACKTOR_DATA: "/path/to/99-CellTracktor/data"
  CTC_EVAL_BINS:  "/path/to/eval_bins"

target_metrics:
  TRA: 0.94
  SEG: 0.88
  DET: 0.92
```
