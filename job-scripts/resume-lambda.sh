#!/bin/bash
set -euo pipefail

# ── Usage ──────────────────────────────────────────────────────
# MANUALLY resume a failed/stopped training run.
#
# Before running this, you should:
#   1. Check the NFS backup exists:  ls ~/iris-stu-transformer-runs/runs/
#   2. If wandb logged past the checkpoint, reset it first:
#        python job-scripts/reset_wandb_to_checkpoint.py <NFS_RUN_DIR>
#   3. Then resume:
#        RUN_NAME=iris-stu-transformer-runs_seed0_ml20 bash job-scripts/resume-lambda.sh
#
# This restores checkpoints + wandb state from NFS to local SSD,
# resumes training, then syncs everything back to NFS on exit.
# ───────────────────────────────────────────────────────────────

RUN_NAME=${RUN_NAME:?ERROR: Set RUN_NAME (e.g. iris-stu-transformer-runs_seed0_ml20)}
IRIS_DIR=${IRIS_DIR:-/home/ubuntu/iris}
PERSISTENT_DIR=${PERSISTENT_DIR:-/home/ubuntu/iris-stu-transformer-runs}
CONDA_ENV="iris"

NFS_RUN_DIR="$PERSISTENT_DIR/runs/$RUN_NAME"

# ── Preflight checks ──────────────────────────────────────────
if [ ! -d "$NFS_RUN_DIR/checkpoints" ]; then
    echo "ERROR: No checkpoint backup found at $NFS_RUN_DIR/checkpoints"
    echo "Available runs:"
    ls "$PERSISTENT_DIR/runs/" 2>/dev/null || echo "  (none)"
    exit 1
fi

if [ ! -f "$NFS_RUN_DIR/checkpoints/last.pt" ]; then
    echo "ERROR: $NFS_RUN_DIR/checkpoints/last.pt not found"
    exit 1
fi

# ── Environment setup ─────────────────────────────────────────
source "$PERSISTENT_DIR/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export HYDRA_FULL_ERROR=1

# ── Restore from NFS to local SSD ────────────────────────────
# Create a local working directory for this resumed run
LOCAL_RUN_DIR="$IRIS_DIR/outputs/resumed_${RUN_NAME}_$(date +%Y%m%d_%H%M%S)"
echo "Restoring from NFS to local SSD: $LOCAL_RUN_DIR"
mkdir -p "$LOCAL_RUN_DIR"

# Restore checkpoints
rsync -a --info=progress2 "$NFS_RUN_DIR/checkpoints/" "$LOCAL_RUN_DIR/checkpoints/"

# Restore wandb directory (preserves run ID for graph continuity)
if [ -d "$NFS_RUN_DIR/wandb" ]; then
    rsync -a "$NFS_RUN_DIR/wandb/" "$LOCAL_RUN_DIR/wandb/"
fi

# Restore Hydra config
if [ -d "$NFS_RUN_DIR/.hydra" ]; then
    rsync -a "$NFS_RUN_DIR/.hydra/" "$LOCAL_RUN_DIR/.hydra/"
fi

# Read metadata
CHECKPOINT_EPOCH=$(python3 -c "import torch; print(torch.load('$LOCAL_RUN_DIR/checkpoints/epoch.pt', map_location='cpu'))")

echo "============================================"
echo "RESUMING: $RUN_NAME"
echo "Checkpoint epoch: $CHECKPOINT_EPOCH"
echo "Will resume from epoch: $((CHECKPOINT_EPOCH + 1))"
echo "Local dir: $LOCAL_RUN_DIR"
echo "NFS backup: $NFS_RUN_DIR"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time: $(date)"
echo "============================================"

cd "$LOCAL_RUN_DIR"

# ── Signal handling ───────────────────────────────────────────
TRAINING_PID=""

cleanup() {
    echo ""
    echo "[$(date)] Signal received, stopping training..."
    if [ -n "$TRAINING_PID" ] && kill -0 "$TRAINING_PID" 2>/dev/null; then
        kill -TERM "$TRAINING_PID"
        wait "$TRAINING_PID" 2>/dev/null || true
    fi
}
trap cleanup SIGTERM SIGINT SIGHUP

# ── Resume training ──────────────────────────────────────────
python "$IRIS_DIR/src/main.py" \
    common.resume=True \
    hydra.output_subdir=null \
    hydra.run.dir=. &

TRAINING_PID=$!
wait "$TRAINING_PID"
EXIT_CODE=$?
TRAINING_PID=""

# ── Sync back to persistent NFS ──────────────────────────────
echo ""
echo "============================================"
echo "Training exited with code: $EXIT_CODE"
echo "End time: $(date)"
echo "Syncing to NFS: $NFS_RUN_DIR"

# Sync checkpoints (overwrite old with new)
rsync -a --info=progress2 "$LOCAL_RUN_DIR/checkpoints/" "$NFS_RUN_DIR/checkpoints/"

# Sync wandb directory
if [ -d "$LOCAL_RUN_DIR/wandb" ]; then
    rsync -a "$LOCAL_RUN_DIR/wandb/" "$NFS_RUN_DIR/wandb/"
fi

# Update metadata
cat > "$NFS_RUN_DIR/run_info.txt" <<INFO
run_name=$RUN_NAME
exit_code=$EXIT_CODE
resumed_from_epoch=$CHECKPOINT_EPOCH
local_run_dir=$LOCAL_RUN_DIR
synced_at=$(date -Iseconds)
hostname=$(hostname)
INFO

echo "Sync complete."
echo "============================================"
exit $EXIT_CODE
