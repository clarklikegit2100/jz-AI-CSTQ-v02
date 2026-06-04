# BSGM-CellTrack: Bayesian Swin Graph Mamba Cell Tracker

## Motivation

Cell-TRACTR (99-CellTracktor) achieves state-of-the-art CTC tracking using Deformable DETR with
a ResNet50 backbone. Its experimental `bayes_swin_graph_mamba` backend introduces placeholders for
Bayesian uncertainty, Swin local attention, and Graph-Mamba temporal mixing — but the actual
implementations are stubs:

| Component | Cell-TRACTR stub | BSGM-CellTrack (this project) |
|---|---|---|
| Backbone | ResNet50/101 | **Swin-T** (hierarchical shifted-window attn) |
| Local spatial attn | DepthwiseConv + sigmoid gate | **SwinLocalBlock** (real W-MSA + SW-MSA) |
| Temporal / SSM | Conv1D + kNN softmax | **SelectiveSSM** (Mamba-style discretized recurrence) |
| Uncertainty | MC Dropout scalar | **BayesHead** (MC Dropout + per-query variance output) |
| Graph | kNN affinity matrix | **CellGraphLayer** (GATv2 message passing) |

The goal: **accurate, fast, end-to-end CTC cell tracking** — simultaneous segmentation + tracking —
outperforming Cell-TRACTR's backbone while keeping the proven DN-Track + track-query paradigm.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     BSGM-CellTrack Pipeline                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input: Frame triplet [t-1, t, t+1]  (grayscale → 3-ch stack)     │
│                      │                                              │
│          ┌───────────▼───────────┐                                 │
│          │  1. Swin-T Backbone   │  4 stages, shifted windows      │
│          │     (per-frame)       │  Output: C2(H/4), C3(H/8),     │
│          └───────────┬───────────┘         C4(H/16), C5(H/32)     │
│                      │                                              │
│          ┌───────────▼───────────┐                                 │
│          │  2. FPN Neck          │  Top-down lateral connections   │
│          │     + Proj → d_model  │  P2, P3, P4, P5 → d=256        │
│          └───────────┬───────────┘                                 │
│                      │                                              │
│          ┌───────────▼───────────┐                                 │
│          │  3. Mamba Temporal    │  3-frame SSM fusion:            │
│          │     Fusion            │  SelectiveSSM along T dimension  │
│          └───────────┬───────────┘  → temporally-aware features    │
│                      │                                              │
│          ┌───────────▼───────────┐                                 │
│          │  4. Deformable Enc.   │  Multi-scale deformable attn    │
│          │     + BayesDropout    │  4 encoder layers               │
│          └───────────┬───────────┘                                 │
│                      │                                              │
│          ┌───────────▼────────────────────────────────┐           │
│          │  5. BSGM Decoder  (per layer):              │           │
│          │     a) CellGraphLayer (kNN GATv2)           │           │
│          │     b) SelectiveSSM over queries            │           │
│          │     c) Self-attention (masked)              │           │
│          │     d) Deformable cross-attention           │           │
│          │     e) BayesianDropout + FFN                │           │
│          │                                             │           │
│          │  Query types:                               │           │
│          │    • Object queries  (new cells, N=400)     │           │
│          │    • Track queries   (prev-frame cells)     │           │
│          │    • DN-Track queries (denoised training)   │           │
│          └───────────┬────────────────────────────────┘           │
│                      │                                              │
│    ┌─────────────────┼──────────────────────────────┐             │
│    │                 │                               │             │
│    ▼                 ▼                               ▼             │
│  Class head       Box head                     Mask head           │
│  (focal loss)     (4D / 8D div boxes)          (FPN pixel dec.)   │
│                                                                     │
│  + BayesHead: per-query epistemic uncertainty (MC-Dropout var)     │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                  CTC Post-processing                                 │
│   Hungarian matching → Track IDs → Division detection              │
│   Output: man_track.txt + man_track*.tif masks                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Designs

### 1. Swin Transformer Backbone (`models/swin_backbone.py`)

Swin-T configuration (tiny):  depths=[2,2,6,2], channels=[96,192,384,768]

```
PatchEmbed: H×W×1 → H/4×W/4×96      (4×4 non-overlapping patches)
Stage 1:  2 × SwinBlock (W-MSA)       → H/4 × W/4 × 96   = C2
          PatchMerging                 → H/8 × W/8 × 192
Stage 2:  2 × SwinBlock (W-MSA / SW-MSA alternating)      = C3
          PatchMerging                 → H/16 × W/16 × 384
Stage 3:  6 × SwinBlock               → H/16 × W/16 × 384 = C4
          PatchMerging                 → H/32 × W/32 × 768
Stage 4:  2 × SwinBlock               → H/32 × W/32 × 768 = C5

FPN lateral projections: C2→256, C3→256, C4→256, C5→256
```

**Key properties**:
- Window size: 7×7 (local attention, O(HW) complexity vs O(H²W²) for global)
- Shifted windows: alternating shift=(0,0) / shift=(3,3) for cross-window connections
- Relative position bias: learned per-window bias table
- Imagenet-22K pretrained weights loadable via `timm` or manual checkpoint

### 2. Mamba Temporal Module (`models/mamba_module.py`)

Selective State Space Model (Mamba-style), pure PyTorch (no CUDA dependency):

```
Input: x ∈ R^{B, T, L, D}   (B=batch, T=3 frames, L=spatial_tokens, D=d_model)

For each (B, L):
  x_seq ∈ R^{T, D}   (temporal sequence per spatial token)

  SelectiveSSM:
    Δ = softplus(Linear_Δ(x) + b_Δ)    # input-dependent timescale
    B = Linear_B(x)                      # input-dependent input matrix
    C = Linear_C(x)                      # input-dependent output matrix
    A_bar = exp(-exp(A) * Δ)            # discretized A (ZOH)
    B_bar = Δ * B                        # discretized B
    
    Scan (recurrence):
      h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
      y_t = C_t · h_t + D * x_t

Output: y ∈ R^{B, T, L, D}  (temporal context for each frame)
Extract: y[:, T//2, :, :]   (current frame features)
```

**Usage in pipeline**: Applied to flattened multi-scale features from 3-frame FPN outputs,
fusing temporal context before the deformable encoder.

### 3. Cell Graph Layer (`models/graph_layer.py`)

Graph Attention v2 (GATv2) for spatial cell-cell interactions:

```
Input: query embeddings Q ∈ R^{B, N, D}
       reference points ref ∈ R^{B, N, 2}   (cell center coordinates)

1. Build kNN graph: k=16 nearest cells by L2 distance on ref
2. GATv2 message passing:
   e_{ij} = LeakyReLU( W_a^T [W_l h_i || W_r h_j] )
   α_{ij} = softmax_j(e_{ij})
   h_i' = ELU( Σ_j α_{ij} W_v h_j )
3. Multi-head: num_heads=4, concat → project → residual

Complexity: O(N × k)  instead of O(N²)
```

### 4. BSGM Decoder Layer (`models/bsgm_decoder.py`)

Each decoder layer applies (in order):

```
tgt = CellGraphLayer(tgt, ref_pts)         # spatial cell relationships
tgt = SelectiveSSM_1D(tgt)                 # query-level SSM (along N dimension)
tgt = BayesDropout(tgt)
tgt = SelfAttention(tgt, attn_mask)        # masked self-attn (DN groups)
tgt = DeformCrossAttention(tgt, memory)    # attend to encoder features
tgt = BayesDropout(tgt)
tgt = FFN(tgt)
```

### 5. Bayesian Uncertainty Head (`models/bsgm_net.py`)

MC Dropout inference: run K=10 forward passes with dropout active, compute:
```
μ_k = mean( logit(class_k) )   over K passes
σ_k = std(  logit(class_k) )   over K passes  → epistemic uncertainty
```

Uncertainty used at inference for:
- Track assignment confidence: high uncertainty → Hungarian cost penalty
- Division detection: require low uncertainty for division acceptance

---

## Training Strategy

### Stage 1: Detection Warmup (epochs 0-8)
- Train detection + segmentation only (disable track queries)
- Swin backbone frozen (pretrained weights)
- Loss: focal + L1 + GIoU + mask

### Stage 2: Tracking (epochs 8-20)
- Unfreeze Swin backbone (lower LR: 2e-5)
- Enable track queries + DN-Track
- Full loss including div_loss

### Stage 3: Fine-tuning (epochs 20-24)
- BayesHead active
- LR decay × 0.1

### Optimizer
- AdamW, weight_decay=1e-4
- Backbone LR: 2e-5, Others: 2e-4
- Clip grad norm: 0.1
- Mixed precision: torch.cuda.amp

### Losses
| Loss | Weight | Notes |
|---|---|---|
| Focal classification | 4.0 | α=0.25, γ=2 |
| L1 box | 5.0 | Normalized coords |
| GIoU box | 2.0 | IoU-aware |
| Mask focal | 5.0 | Per-pixel |
| Mask dice | 5.0 | F1-aware |
| Division | 5.0 | Extra weight on 8D boxes |
| DN-track | 1.0 | Denoised supervision |
| Auxiliary | × all | Each decoder layer |

---

## CTC Output Pipeline

```
Model output per frame:
  pred_logits: [B, N_queries, num_classes]
  pred_boxes:  [B, N_queries, 4 or 8]   (4=cell, 8=division)
  pred_masks:  [B, N_queries, H_mask, W_mask]
  hs_embed:    [B, N_queries, d_model]  (for next-frame track queries)
  uncertainty: [B, N_queries]

Post-processing:
  1. Filter by confidence threshold (0.5 default)
  2. Assign track IDs:
       - Object queries → new track ID
       - Track queries  → inherited track ID from previous frame
  3. Division detection:
       - 8D box predictions → 2 child bounding boxes
       - Link: parent_id → (child1_id, child2_id)
  4. Write man_track.txt:
       <track_id> <start_frame> <end_frame> <parent_id>
  5. Write man_track{t:03d}.tif:
       uint16 mask where pixel value = track_id
```

---

## Configuration

All hyperparameters controlled by YAML files in `cfgs/`.
Base config (`cfgs/base_bsgm.yaml`) defines defaults; dataset configs override.

Key new parameters vs Cell-TRACTR:
```yaml
backbone: swin_t           # swin_t | swin_s | swin_b | resnet50
swin_window_size: 7
swin_pretrained: null      # path to pretrained Swin weights
mamba_d_state: 16          # SSM state dimension
mamba_d_conv: 4            # SSM conv width
mamba_expand: 2            # SSM expansion factor
graph_topk: 16             # kNN graph neighborhood
graph_heads: 4             # GATv2 attention heads
bayesian_dropout: 0.1      # dropout rate (active at eval for MC)
bayesian_eval: false       # True: always active (MC inference)
mc_samples: 10             # MC forward passes for uncertainty
uncertainty_threshold: 0.3 # High uncertainty → conservative tracking
```

---

## File Structure

```
jz-AI-CSTQ-v02/
├── src/ai_cstq/
│   ├── models/
│   │   ├── __init__.py          # build_model()
│   │   ├── swin_backbone.py     # SwinTransformerBackbone
│   │   ├── mamba_module.py      # SelectiveSSM, MambaBlock, TemporalMamba
│   │   ├── graph_layer.py       # CellGraphLayer (GATv2)
│   │   ├── bsgm_decoder.py      # BSGMDecoderLayer, BSGMDecoder
│   │   ├── bsgm_net.py          # BSGMCellTrack (main model)
│   │   ├── criterion.py         # SetCriterion, loss functions
│   │   ├── matcher.py           # HungarianMatcher
│   │   ├── mask_head.py         # FPNMaskHead
│   │   └── position_encoding.py # PositionEmbeddingSine/Learned
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── ctc_coco.py          # CTC/COCO dataset
│   │   └── transforms.py        # Augmentations
│   ├── util/
│   │   ├── box_ops.py           # GIoU, box conversions
│   │   ├── misc.py              # Misc helpers
│   │   └── ctc_io.py            # CTC I/O (man_track.txt, tif masks)
│   └── engine.py                # train_one_epoch, evaluate
├── scripts/
│   └── create_coco_from_ctc.py  # CTC → COCO JSON conversion
├── cfgs/
│   ├── base_bsgm.yaml
│   ├── ctchuh7_bsgm.yaml
│   ├── ctcgowt1_bsgm.yaml
│   └── ctcsim_bsgm.yaml
├── train.py                     # Training entry point
├── infer.py                     # Inference + CTC output
├── requirements.txt
├── setup.py
└── DESIGN.md                    # This file
```

---

## Performance Targets

| Metric | Cell-TRACTR (HuH7) | BSGM-CellTrack target |
|---|---|---|
| TRA | ~0.93 | ≥ 0.94 |
| SEG (Dice) | ~0.87 | ≥ 0.88 |
| DET (F1) | ~0.91 | ≥ 0.92 |
| Inference FPS | ~2-3 | ≥ 4 (Swin-T faster than ResNet101) |
| Uncertainty calibration | N/A | ECE < 0.05 |

---

## References

1. Cell-TRACTR: "Cell tracking with a transformer", PLOS Comp. Biol. 2025
2. Swin Transformer: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021
3. Mamba: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", arXiv 2312
4. Deformable DETR: "Deformable DETR: Deformable Transformers for End-to-End Object Detection", ICLR 2021
5. GATv2: "How Attentive are Graph Attention Networks?", ICLR 2022
6. CTC: Cell Tracking Challenge (celltrackingchallenge.net)
