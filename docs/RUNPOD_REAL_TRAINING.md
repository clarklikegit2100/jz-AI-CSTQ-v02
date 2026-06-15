# Running the REAL v3.2 BSGM config on RunPod

The full v3.2 architecture (`512²`, `enc_layers=4`, `dec_layers=6`, `num_queries=300`,
`dim_feedforward=1024`, ~50M params) needs **~10–12 GB** for a training step and does
**not** fit the local 6 GB RTX 3050. This guide runs it on a RunPod cloud GPU.

It trains the same model with the two correctness fixes already validated locally:
- `criterion.loss_labels` fix (only matched queries get the cell target) — already in the repo.
- `focal_alpha 0.25 → 0.5` (up-weights the rare cell class) — applied by `retrain_huh7_real.py`.

AMP stays **off** (the matcher NaNs under autocast); a ≥16 GB GPU runs fp32 comfortably.

---

## 0. Before you start (on your local machine)

Commit and push the enabling changes so the pod can `git clone` them:

```powershell
cd F:\GitHub\jz-AI-CSTQ-v02
git add src/ai_cstq/models/criterion.py scripts/run_full_epoch.py scripts/evaluate_ctc.py `
        scripts/retrain_huh7_real.py scripts/diag_scores.py docs/RUNPOD_REAL_TRAINING.md
git commit -m "Real-config cloud training: env-var data paths + retrain_huh7_real + loss fix"
git push origin main
```

What must be on the remote: the `criterion.py` fix, the env-var path overrides in
`run_full_epoch.py` / `evaluate_ctc.py`, and `retrain_huh7_real.py`.

---

## 1. Pick the pod

### Option A — RTX 5090 (recommended, cheapest total, no network volume)

| Item | Value |
|---|---|
| GPU | **RTX 5090 (32 GB, $0.99/hr)** — fastest, cheapest total cost |
| Template | RunPod PyTorch 2.x (CUDA 12.8) |
| Storage | Container disk only (50 GB) — no network volume needed |
| Total cost | ~$4–8 for 24 epochs (~4–8 h) |

**Because the 5090 has no network volume**, use the auto-upload script (`scripts/cloud_train_huh7_5090.sh`)
which pushes checkpoints to HuggingFace Hub and auto-stops the pod when done (§ Option A workflow below).

### Option B — RTX 4090 / L40S / A100 (network volume, manual workflow)

| Item | Recommended |
|---|---|
| GPU | RTX 4090 (24 GB, $0.69/hr) · L40S (48 GB, $0.86/hr) · A100 40 GB |
| Template | RunPod PyTorch 2.x (CUDA 12.x) |
| Disk | Container 20 GB + **Volume 30 GB** (mounted at `/workspace`, persists across restarts) |

Create the pod with a **Network Volume** mounted at `/workspace` so data and
checkpoints survive a pod stop/restart.

---

## 1b. Option A workflow (RTX 5090 — auto-upload + auto-stop)

### Step 1: Set Environment Variables in RunPod pod settings (before starting)

| Variable | Value |
|---|---|
| `HF_TOKEN` | Your HuggingFace write token (huggingface.co/settings/tokens) |
| `HF_REPO` | e.g. `yourname/bsgm-huh7` (will be created automatically) |
| `RUNPOD_API_KEY` | RunPod API key (runpod.io → User Settings → API Keys) |
| `DEEP_CSTQ_DATA` | `/workspace/data/Deep_CSTQ_Datasets/src/output` |

### Step 2: Upload data to the pod (one-time)

```bash
# On local (install runpodctl first):
runpodctl send "F:/GitHub/Deep_CSTQ_Datasets/src/output/Fluo-C2DL-Huh7"
# On pod:
mkdir -p /workspace/data/Deep_CSTQ_Datasets/src/output
runpodctl receive <code>
mv Fluo-C2DL-Huh7 /workspace/data/Deep_CSTQ_Datasets/src/output/
```

### Step 3: Run the auto script (trains → uploads → stops pod)

```bash
cd /workspace
git clone https://github.com/clarklikegit2100/jz-AI-CSTQ-v02.git
bash jz-AI-CSTQ-v02/scripts/cloud_train_huh7_5090.sh
```

That's it. The script handles: pip install → train 24 epochs → upload last 3 checkpoints + log to HF Hub → stop pod.

### Step 4: Download results locally

```bash
pip install huggingface_hub
huggingface-cli download yourname/bsgm-huh7 --local-dir ./hf_results
```

---

## 2. What to upload  (Option B — network volume pods only)

| Source (local) | Size | Destination (pod) |
|---|---|---|
| `Deep_CSTQ_Datasets/src/output/Fluo-C2DL-Huh7/` | ~3.7 GB | `/workspace/data/Deep_CSTQ_Datasets/src/output/Fluo-C2DL-Huh7/` |
| `99-CellTracktor/EvaluationSoftware/Linux/` | small | `/workspace/eval_bins/` |
| The repo | small | clone from GitHub (step 3) |

The COCO annotations are **regenerated on the pod** from the raw Huh7 data, so you only
need the raw images + GT (`train/` for training, `test/37` + `test/37_GT` for eval).

### Upload options
- **runpodctl** (simplest for files/folders):
  ```bash
  # on local (install runpodctl first):
  runpodctl send "F:/GitHub/Deep_CSTQ_Datasets/src/output/Fluo-C2DL-Huh7"
  # it prints a one-time code; on the pod:
  runpodctl receive <code>
  ```
- **scp / rsync** (if the pod exposes SSH): `scp -r -P <port> <local> root@<ip>:/workspace/data/...`
- **Cloud bucket**: push to S3/GDrive, pull on the pod with `aws s3 cp` / `gdown`.

---

## 3. Pod setup (run once, in the pod terminal)

```bash
cd /workspace
git clone https://github.com/clarklikegit2100/jz-AI-CSTQ-v02.git
cd jz-AI-CSTQ-v02

# --- Python env ---
conda create -n bsgm python=3.10 -y
conda activate bsgm
# Match the CUDA wheel to the pod's CUDA (cu121/cu124 typical on RunPod):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# (mamba-ssm is optional; the code falls back to pure-PyTorch if absent — skip it.)

# --- Make the Linux CTC eval binaries executable ---
chmod +x /workspace/eval_bins/*
```

Quick GPU sanity check:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.mem_get_info())"
```

---

## 4. Point the code at the uploaded data (env vars)

These override the hardcoded Windows paths (no code edits needed):

```bash
export DEEP_CSTQ_DATA=/workspace/data/Deep_CSTQ_Datasets/src/output
export CTC_EVAL_BINS=/workspace/eval_bins
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
```

Add them to `~/.bashrc` so they persist in new shells.

---

## 5. Launch training

```bash
cd /workspace/jz-AI-CSTQ-v02
# First run regenerates COCO from the raw Huh7 data (~1 min), then trains.
nohup python scripts/retrain_huh7_real.py --epochs 24 --lr_drop 20 \
      > train_real.log 2>&1 &
tail -f train_real.log
```

Expected first lines:
```
REAL config | img=512 nq=300 enc=4 dec=6 ff=1024 focal_alpha=0.5 | epochs=24 lr_drop=20
params: 49.8M
--- Epoch 1/24  (lr=2.00e-04) ---
```

- Checkpoints land in `results/ctc-huh7-real/checkpoint_epoch{N}.pth` (one per epoch, resumable).
- `nohup ... &` keeps it running if your SSH/web terminal disconnects.
- **Rough time:** ~960 batches/epoch. RTX 4090 ≈ 15–30 min/epoch → 24 epochs ≈ **6–12 h**.
  A100 is faster. Watch the live `ms/batch` print to refine the estimate.

---

## 6. Evaluate (after training, or on the best epoch)

The fixed model should de-saturate AND become confident. First check the score
distribution, then run the official CTC metrics:

```bash
# 1) score sanity — cell scores should now exceed 0.3 and separate from background
python scripts/diag_scores.py --ckpt results/ctc-huh7-real/checkpoint_epoch24.pth

# 2) official TRA / SEG / DET (uses the Linux binaries via $CTC_EVAL_BINS)
python scripts/evaluate_ctc.py --datasets huh7 \
       --ckpt_epoch 24 --ckpt_dir ctc-huh7-real \
       --conf_threshold 0.5 --max_track_queries 300
```

Pick the **best epoch** by re-running `--ckpt_epoch N` for a few late epochs (the local
runs peaked mid-to-late, not always the last). Sweep `--conf_threshold` 0.3/0.5/0.7 to
find the calibrated operating point.

---

## 7. Retrieve results

```bash
# from the pod, send checkpoints + logs back:
runpodctl send results/ctc-huh7-real train_real.log
# then `runpodctl receive <code>` locally
```
Or commit just the logs/metrics to a branch (don't commit multi-hundred-MB `.pth` files).

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | Lower `target_size` to 448 or 384 in `cfgs/ctchuh7_bsgm.yaml`, or pick a bigger GPU. fp32 at 512² needs ~12 GB. |
| `NaN` loss after a few steps | AMP is off by design — if you re-enabled it, turn it back off (matcher overflows in fp16). |
| COCO step says 0 images | `DEEP_CSTQ_DATA` is wrong — it must contain `Fluo-C2DL-Huh7/train/<seq>/t*.tif`. |
| Eval: `directory ... does not exist` | `CTC_EVAL_BINS` unset, or test GT (`test/37_GT`) not uploaded, or binaries not `chmod +x`. |
| `mamba-ssm` build errors | Skip it — the model auto-falls back to pure-PyTorch Mamba. |

---

## 9. Scaling to all 6 datasets

Once Huh7 validates, generalize: `retrain_huh7_real.py` is Huh7-specific, but
`run_full_epoch.py` already iterates all datasets and now honors `$DEEP_CSTQ_DATA` /
`$CTRACKTOR_DATA`. Upload the other datasets' raw data, set those env vars, and run
`run_full_epoch.py` with an upgraded config (lift `BSGM_CFG` to the real arch, or add a
`--config` hook). Note PSC/U373/etc. also need `$CTRACKTOR_DATA` for the `ctracktor`-source
datasets.
```
