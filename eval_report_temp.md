# Evaluation Report (Temporary, 2026-06-15)
# Status: partial — v3.2 BSGM pending RunPod training

---

## Architecture Comparison

| Feature            | v1.0 Deep-CSTQ-GR | v2.0 Deep-CSTQ-MG | v3.0 Cell-TRACTR | v3.2 BSGM       |
|--------------------|-------------------|-------------------|------------------|-----------------|
| Paradigm           | TbD               | TbD               | End-to-End       | End-to-End      |
| Backbone           | ResNet + GNN      | ResNet + Mamba    | ResNet50         | Swin-T          |
| Temporal           | GRU               | Mamba SSM         | DETR tracking    | Mamba SSM       |
| Graph              | GNN               | GNN               | —                | GATv2           |
| Uncertainty        | —                 | —                 | —                | Bayesian MC     |
| Encoder/Decoder    | —                 | —                 | 4E / 4D          | 4E / 6D         |
| Queries            | —                 | —                 | 400              | 300             |
| Seg input needed   | YES               | YES               | NO               | NO              |
| Params (approx)    | ~10M              | ~12M              | ~45M             | ~50M            |

**Note:** v1.0 and v2.0 use the same codebase; Mamba is auto-selected on CUDA GPU → existing
results are all v2.0 (Mamba). True v1.0 (GRU) requires a CPU run or no mamba-ssm installed.

---

## Results

### Paradigm A: Tracking-by-Detection  (seg mask required as input)

#### v2.0 Deep-CSTQ-MG  (Mamba auto-selected on GPU)

| Dataset              | Seq | Seg Input      | TRA    | SEG    | DET    |
|----------------------|-----|----------------|--------|--------|--------|
| Fluo-C2DL-Huh7       | 02  | GT mask ⚠      | 0.853  | 0.618  | 0.876  |
| Fluo-N2DH-GOWT1      | 01  | CSTQ auto-seg  | 0.983  | 0.869  | —      |
| Fluo-N2DH-GOWT1      | 02  | CSTQ auto-seg  | 0.877  | 0.898  | 0.876  |
| PhC-C2DH-U373        | 01  | ST mask        | 0.926  | 0.727  | 0.930  |
| PhC-C2DL-PSC         | 02  | CSTQ auto-seg  | 0.878  | 0.616  | 0.904  |
| DIC-C2DH-HeLa        | 02  | ST mask        | 0.000  | —      | —      | ← FAILED
| Fluo-N2DH-SIM+       | 01  | GT mask ⚠      | 0.990  | 0.858  | 0.991  |
| Fluo-N2DH-SIM+       | 02  | GT mask ⚠      | 0.987  | 0.997  | 0.997  |

⚠ GT mask = oracle condition (not realistic). ST/CSTQ = realistic deployment.

---

### Paradigm B: End-to-End  (raw image only, no seg input)

#### v3.0 Cell-TRACTR  (ResNet50 + Deformable DETR, 24 epochs, CoMOT + DN-track)

Source: F:\GitHub\99-CellTracktor\code-win11\results\

| Dataset              | Seq01  TRA | Seq02  TRA | TRA avg | SEG avg | DET avg | Notes                     |
|----------------------|------------|------------|---------|---------|---------|---------------------------|
| Fluo-C2DL-Huh7       | —          | —          | —       | —       | —       | no checkpoint, not trained |
| Fluo-N2DH-GOWT1      | 0.2846     | 0.9011     | 0.5929  | 0.1566  | 0.6105  |                           |
| PhC-C2DH-U373        | 0.0000     | 0.1127     | 0.0563  | 0.0000  | 0.0579  | seq01: 35 frames missing → zero-filled |
| PhC-C2DL-PSC         | —          | —          | —       | —       | —       | no checkpoint, not trained |
| DIC-C2DH-HeLa        | 0.0000     | 0.8854     | 0.4427  | 0.0000  | 0.4442  | SEG GT missing            |
| Fluo-N2DH-SIM+       | 0.1810     | 0.7670     | 0.4740  | ~0.345  | ~0.476  | seqs 19 & 20              |

Existing result files:
- F:\GitHub\99-CellTracktor\code-win11\results\ctcgowt1\ctc_eval_results.txt  ✓
- F:\GitHub\99-CellTracktor\code-win11\results\ctcdhela\ctc_eval_results.txt  ✓
- F:\GitHub\99-CellTracktor\code-win11\results\ctcu373\ctc_eval_results.txt   ✓ (seq01 fixed with zero-fill)
- F:\GitHub\99-CellTracktor\code-win11\results\ctcsim\ctc_eval_results.txt    ✓

#### v3.2 BSGM-CellTrack  (Swin-T + Mamba + GATv2 + Bayesian, ~50M params)

| Dataset              | TRA | SEG | DET | Notes                        |
|----------------------|-----|-----|-----|------------------------------|
| Fluo-C2DL-Huh7       | ⏳  | ⏳  | ⏳  | pending RunPod training      |
| Fluo-N2DH-GOWT1      | ⏳  | ⏳  | ⏳  | pending RunPod training      |
| PhC-C2DH-U373        | ⏳  | ⏳  | ⏳  | pending RunPod training      |
| PhC-C2DL-PSC         | ⏳  | ⏳  | ⏳  | pending RunPod training      |
| DIC-C2DH-HeLa        | ⏳  | ⏳  | ⏳  | pending RunPod training      |
| Fluo-N2DH-SIM+       | ⏳  | ⏳  | ⏳  | pending RunPod training      |

Train cmd (after criterion fix applied):
  python scripts/retrain_huh7_real.py --epochs 24 --lr_drop 20
Eval cmd:
  python scripts/evaluate_ctc.py --datasets huh7 gowt1 u373 psc dhela sim \
         --ckpt_epoch 24 --ckpt_dir ctc-huh7-real \
         --conf_threshold 0.5 --max_track_queries 300

---

## Gaps / TODO

1. v1.0 GRU explicit results — run Deep_CSTQ without mamba-ssm (CPU or uninstall)
2. v3.0 Huh7 — train ctchuh7 in 99-CellTracktor (~6-12h)
3. v3.0 PSC  — train ctcpsc in 99-CellTracktor (~3-6h)
4. v3.0 SEG  — all SEG=0 for HeLa and U373 (GT/SEG dirs missing or mismatched)
5. v3.2 BSGM — full RunPod training (24 epochs, 512², fp32, focal_alpha=0.5)
6. v2.0 fairness — Huh7 and SIM+ use GT masks (oracle); rerun with ST masks

---

## Fairness Notes

- v1.0/v2.0 cannot be compared directly to v3.0/v3.2 on raw metrics because they
  require ground-truth or auto-generated segmentation masks as input.
- Deep_CSTQ Huh7 and SIM+ results are oracle (GT masks), inflating TRA vs realistic.
- v3.0 SEG=0 for most datasets is likely a GT/SEG directory mismatch, not model failure.
- v3.0 seq01 TRA consistently lower than seq02 across all datasets — warrants investigation.
