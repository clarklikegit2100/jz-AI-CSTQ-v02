#!/usr/bin/env bash
# Wait for the Huh7-fixed retrain to finish, then auto-run diagnosis + CTC eval.
set -u
cd /f/GitHub/jz-AI-CSTQ-v02
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
PY="/c/Users/jack/Miniconda3/envs/jz-AI-CSTQ-v02/python.exe"
TRAIN_PID="${1:-1676}"
CKPT_DIR="results/ctc-huh7-fixed"
LOG="eval_huh7_fixed.log"
: > "$LOG"

echo "[auto-eval] waiting for training pid $TRAIN_PID ..." | tee -a "$LOG"
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 30; done
echo "[auto-eval] training finished at $(date)" | tee -a "$LOG"

# latest fixed checkpoint epoch
EP=$(ls "$CKPT_DIR"/checkpoint_epoch*.pth 2>/dev/null \
     | grep -oE 'epoch[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1)
if [ -z "$EP" ]; then echo "[auto-eval] NO checkpoints in $CKPT_DIR" | tee -a "$LOG"; exit 1; fi
echo "[auto-eval] latest fixed checkpoint: epoch $EP" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "########## DE-SATURATION CHECK (diag_scores) ##########" | tee -a "$LOG"
"$PY" scripts/diag_scores.py --ckpt "$CKPT_DIR/checkpoint_epoch$EP.pth" 2>&1 \
   | grep -vE "Warning|warn" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "########## CTC EVAL  conf=0.5  (fixed model) ##########" | tee -a "$LOG"
"$PY" scripts/evaluate_ctc.py --datasets huh7 --ckpt_dir ctc-huh7-fixed \
   --ckpt_epoch "$EP" --conf_threshold 0.5 --max_track_queries 300 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "########## CTC EVAL  conf=0.3  (fixed model) ##########" | tee -a "$LOG"
"$PY" scripts/evaluate_ctc.py --datasets huh7 --ckpt_dir ctc-huh7-fixed \
   --ckpt_epoch "$EP" --conf_threshold 0.3 --max_track_queries 300 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[auto-eval] DONE" | tee -a "$LOG"
