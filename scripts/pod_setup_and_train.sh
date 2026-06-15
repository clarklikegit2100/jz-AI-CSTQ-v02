#!/usr/bin/env bash
# =============================================================================
# pod_setup_and_train.sh
# Paste and run this ONCE inside the RunPod terminal (SSH or web console).
# RTX PRO 4500 | network volume at /workspace | CUDA 12.4 | PyTorch 2.4 image
#
# What it does:
#   1. Clones / updates the repo
#   2. pip-installs missing deps (torch already in image)
#   3. Starts training in background (survives SSH disconnect)
#   4. Writes /workspace/training_complete.txt when finished
#      so the local download script can detect completion.
# =============================================================================
set -eo pipefail

# ── Persistent paths (all on the 50 GB network volume) ──────────────────────
export DEEP_CSTQ_DATA=/workspace/data/Deep_CSTQ_Datasets/src/output
export CTC_EVAL_BINS=/workspace/eval_bins
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
REPO=/workspace/jz-AI-CSTQ-v02
LOG=/workspace/train_real.log

echo "================================================================"
echo "  Pod: $(hostname)  |  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo "  CUDA: $(nvcc --version | grep release | awk '{print $5}' | tr -d ,)"
echo "================================================================"

# ── 1. Clone / update repo ───────────────────────────────────────────────────
echo "[1/4] Repo setup..."
if [ -d "$REPO/.git" ]; then
    git -C "$REPO" pull --ff-only
    echo "  Repo updated."
else
    git clone https://github.com/clarklikegit2100/jz-AI-CSTQ-v02.git "$REPO"
    echo "  Repo cloned."
fi
cd "$REPO"

# ── 2. Install Python deps ───────────────────────────────────────────────────
echo "[2/4] Installing dependencies (torch already in image, skipping)..."
pip install -q -r requirements.txt 2>&1 | tail -5

# Verify GPU access
python -c "import torch; assert torch.cuda.is_available(), 'No CUDA!'; \
    print(f'  torch {torch.__version__} | {torch.cuda.get_device_name(0)} | \
{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')"

# ── 3. Verify data is present ────────────────────────────────────────────────
echo "[3/4] Checking data..."
HUH7_TRAIN="$DEEP_CSTQ_DATA/Fluo-C2DL-Huh7/train"
if [ ! -d "$HUH7_TRAIN" ]; then
    echo ""
    echo "ERROR: Huh7 training data not found at $HUH7_TRAIN"
    echo "Upload it first from local (run this on your LOCAL machine):"
    echo ""
    echo "  # Option A — runpodctl (easiest):"
    echo "  runpodctl send \"F:/GitHub/Deep_CSTQ_Datasets/src/output/Fluo-C2DL-Huh7\""
    echo "  # Then on pod:"
    echo "  mkdir -p $DEEP_CSTQ_DATA && runpodctl receive <CODE>"
    echo "  mv Fluo-C2DL-Huh7 $DEEP_CSTQ_DATA/"
    echo ""
    echo "  # Option B — scp (replace HOST and PORT):"
    echo "  scp -P PORT -r \"F:/GitHub/Deep_CSTQ_Datasets/src/output/Fluo-C2DL-Huh7\" root@HOST:$DEEP_CSTQ_DATA/"
    echo ""
    exit 1
fi
N_FRAMES=$(find "$HUH7_TRAIN" -name "t*.tif" | wc -l)
echo "  Huh7 training frames found: $N_FRAMES"
[ "$N_FRAMES" -lt 10 ] && echo "WARNING: very few frames — data may be incomplete"

# ── 4. Launch training in background ────────────────────────────────────────
echo "[4/4] Starting training (background, survives SSH disconnect)..."
rm -f /workspace/training_complete.txt /workspace/training_failed.txt

nohup bash -c "
    cd $REPO
    export DEEP_CSTQ_DATA=$DEEP_CSTQ_DATA
    export CTC_EVAL_BINS=$CTC_EVAL_BINS
    export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
    python scripts/retrain_huh7_real.py --epochs 24 --lr_drop 20
    if [ \$? -eq 0 ]; then
        echo 'DONE' > /workspace/training_complete.txt
        echo 'Training finished successfully at '$(date) >> /workspace/training_complete.txt
    else
        echo 'FAILED' > /workspace/training_failed.txt
        echo 'Training failed at '$(date) >> /workspace/training_failed.txt
    fi
" >> "$LOG" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > /workspace/train_pid.txt
echo ""
echo "================================================================"
echo "  Training started  PID=$TRAIN_PID"
echo "  Log : tail -f $LOG"
echo "  Done: /workspace/training_complete.txt  (written when finished)"
echo ""
echo "  RTX PRO 4500 estimate: ~12–22 h for 24 epochs"
echo "  Cost estimate        : ~\$9–16 at \$0.74/hr"
echo "================================================================"
echo ""
echo "Tailing log (Ctrl+C to detach, training continues)..."
tail -f "$LOG"
