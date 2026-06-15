#!/usr/bin/env bash
# =============================================================================
# cloud_train_huh7_5090.sh
# RTX 5090 (no network volume) — full pipeline:
#   setup → train 24 epochs → upload to HuggingFace Hub → auto-stop pod
#
# Required env vars (set in pod > Environment Variables tab before starting):
#   HF_TOKEN        — huggingface.co/settings/tokens  (write permission)
#   HF_REPO         — e.g.  "yourname/bsgm-huh7"
#   RUNPOD_API_KEY  — runpod.io/console/user/settings > API Keys
#   DEEP_CSTQ_DATA  — /workspace/data/Deep_CSTQ_Datasets/src/output
#
# Optional:
#   CTC_EVAL_BINS   — /workspace/eval_bins  (for post-training CTC eval)
#   EPOCHS          — default 24
#   LR_DROP         — default 20
# =============================================================================
set -eo pipefail

# --- Validate required vars ---
: "${HF_TOKEN:?Set HF_TOKEN (huggingface.co/settings/tokens, write access)}"
: "${HF_REPO:?Set HF_REPO e.g. yourname/bsgm-huh7}"
: "${RUNPOD_API_KEY:?Set RUNPOD_API_KEY (runpod.io > User Settings > API Keys)}"
: "${DEEP_CSTQ_DATA:?Set DEEP_CSTQ_DATA e.g. /workspace/data/Deep_CSTQ_Datasets/src/output}"

export CTC_EVAL_BINS="${CTC_EVAL_BINS:-/workspace/eval_bins}"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8

EPOCHS="${EPOCHS:-24}"
LR_DROP="${LR_DROP:-20}"
REPO_DIR="/workspace/jz-AI-CSTQ-v02"
CKPT_DIR="$REPO_DIR/results/ctc-huh7-real"
LOG="$REPO_DIR/train_real.log"

echo "================================================================"
echo " BSGM HuH7 — RTX 5090 auto-train + upload + stop"
echo " HF repo : $HF_REPO"
echo " Epochs  : $EPOCHS   lr_drop: $LR_DROP"
echo " Data    : $DEEP_CSTQ_DATA"
echo " Started : $(date)"
echo "================================================================"

# =============================================================================
# 1. Clone / update repo
# =============================================================================
cd /workspace
if [ -d "$REPO_DIR/.git" ]; then
    echo "[1/5] Updating existing repo..."
    git -C "$REPO_DIR" pull --ff-only
else
    echo "[1/5] Cloning repo..."
    git clone https://github.com/clarklikegit2100/jz-AI-CSTQ-v02.git "$REPO_DIR"
fi
cd "$REPO_DIR"

# =============================================================================
# 2. Python environment
# =============================================================================
echo "[2/5] Installing dependencies..."
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -3
pip install -q -r requirements.txt 2>&1 | tail -3
pip install -q huggingface_hub 2>&1 | tail -1

# Quick GPU sanity check
python -c "import torch; print(f'  PyTorch {torch.__version__} | GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.mem_get_info()[1]/1e9:.1f} GB')"

# =============================================================================
# 3. Train
# =============================================================================
echo "[3/5] Training started: $(date)"
python scripts/retrain_huh7_real.py \
    --epochs "$EPOCHS" \
    --lr_drop "$LR_DROP" \
    2>&1 | tee "$LOG"
echo "Training finished: $(date)"

# =============================================================================
# 4. (Optional) CTC eval if binaries are present
# =============================================================================
if [ -f "$CTC_EVAL_BINS/TRAMeasure" ]; then
    echo "[4/5] Running CTC evaluation..."
    chmod +x "$CTC_EVAL_BINS"/*
    python scripts/evaluate_ctc.py \
        --datasets huh7 \
        --ckpt_epoch "$EPOCHS" \
        --ckpt_dir ctc-huh7-real \
        --conf_threshold 0.5 \
        --max_track_queries 300 \
        2>&1 | tee -a "$LOG"
else
    echo "[4/5] eval_bins not found — skipping CTC eval (upload CTC eval binaries to $CTC_EVAL_BINS)"
fi

# =============================================================================
# 5. Upload to HuggingFace Hub  (last 3 epoch checkpoints + log + eval results)
# =============================================================================
echo "[5/5] Uploading to HuggingFace Hub: $HF_REPO ..."
python3 - <<'PYEOF'
import os, glob, sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("huggingface_hub not installed — skipping upload")
    sys.exit(0)

token   = os.environ["HF_TOKEN"]
repo    = os.environ["HF_REPO"]
epochs  = int(os.environ.get("EPOCHS", "24"))
ckpt_dir = Path(os.environ.get("CKPT_DIR", "/workspace/jz-AI-CSTQ-v02/results/ctc-huh7-real"))
log_file = Path(os.environ.get("LOG", "/workspace/jz-AI-CSTQ-v02/train_real.log"))

api = HfApi()
try:
    create_repo(repo, token=token, exist_ok=True, private=False)
    print(f"  Repo: huggingface.co/{repo}")
except Exception as e:
    print(f"  Repo create/check: {e}")

def upload(local, remote):
    size_mb = os.path.getsize(local) / 1e6
    print(f"  -> {remote}  ({size_mb:.1f} MB)")
    api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                    repo_id=repo, token=token)

# Upload last 3 epoch checkpoints (e.g. epoch22, 23, 24)
start_epoch = max(1, epochs - 2)
for ep in range(start_epoch, epochs + 1):
    p = ckpt_dir / f"checkpoint_epoch{ep}.pth"
    if p.exists():
        upload(p, f"checkpoints/checkpoint_epoch{ep}.pth")

# checkpoint_last.pth (if saved separately)
p = ckpt_dir / "checkpoint_last.pth"
if p.exists():
    upload(p, "checkpoints/checkpoint_last.pth")

# Training log
if log_file.exists():
    upload(log_file, "train_real.log")

# CTC eval results
for txt in ckpt_dir.glob("*.txt"):
    upload(txt, f"eval/{txt.name}")

print("  Upload complete.")
print(f"  Download locally: huggingface-cli download {repo} --local-dir ./hf_results")
PYEOF

# =============================================================================
# Auto-stop pod
# =============================================================================
POD_ID="${RUNPOD_POD_ID:-}"
if [ -n "$POD_ID" ]; then
    echo "Sending stop signal for pod $POD_ID ..."
    STOP_RESP=$(curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"mutation { podStop(input: {podId: \\\"${POD_ID}\\\"}) { id desiredStatus } }\"}")
    echo "RunPod response: $STOP_RESP"
    echo "Pod stop requested. Results are safe on HuggingFace Hub."
else
    echo "RUNPOD_POD_ID not set — stop the pod manually in the RunPod dashboard."
fi

echo "================================================================"
echo " ALL DONE — $(date)"
echo " Download checkpoints locally:"
echo "   pip install huggingface_hub"
echo "   huggingface-cli download $HF_REPO --local-dir ./hf_results"
echo "================================================================"
