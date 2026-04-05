#!/bin/bash
set -euo pipefail

# ── Usage ──────────────────────────────────────────────────────
# Start a FRESH training run on a Lambda Labs instance.
#
#   GAME=pen-human-v1 SEED=0 MAX_LEN=20 bash job-scripts/train-lambda.sh
#
# Training runs on local SSD for speed. On ANY exit (success, failure,
# or signal), checkpoints + wandb metadata are synced to persistent NFS
# so you can manually resume later from any instance.
# ───────────────────────────────────────────────────────────────

GAME=${GAME:-AsterixNoFrameskip-v4}
SEED=${SEED:-0}
MAX_LEN=${MAX_LEN:-20}
IRIS_DIR=${IRIS_DIR:-/home/ubuntu/iris}
PERSISTENT_DIR=${PERSISTENT_DIR:-/home/ubuntu/iris-runs}
CONDA_ENV="iris"

# Unique run name used as the NFS backup directory
RUN_NAME=${RUN_NAME:-"iris-runs_seed${SEED}_ml${MAX_LEN}"}
NFS_RUN_DIR="$PERSISTENT_DIR/runs/$RUN_NAME"

# ── Preflight checks ──────────────────────────────────────────
if [ -d "$NFS_RUN_DIR/checkpoints" ]; then
    echo "ERROR: A backup already exists at $NFS_RUN_DIR"
    echo "If you want to resume, use:  resume-lambda.sh"
    echo "If you want to start fresh, remove: $NFS_RUN_DIR"
    exit 1
fi

# ── Environment setup ─────────────────────────────────────────
source "$PERSISTENT_DIR/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export HYDRA_FULL_ERROR=1

echo "============================================"
echo "FRESH training: $GAME"
echo "Seed: $SEED | Max blocks: $MAX_LEN"
echo "Run name: $RUN_NAME"
echo "Local dir: $IRIS_DIR"
echo "NFS backup: $NFS_RUN_DIR"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time: $(date)"
echo "============================================"

cd "$IRIS_DIR"

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

# ── Run training ──────────────────────────────────────────────
python src/main.py \
    env.train.id="$GAME" \
    env.test.id="$GAME" \
    world_model.max_blocks="$MAX_LEN" \
    common.device=cuda:0 \
    common.seed="$SEED" \
    common.do_checkpoint=True \
    wandb.mode=online \
    wandb.name="$RUN_NAME" &

TRAINING_PID=$!
wait "$TRAINING_PID"
EXIT_CODE=$?
TRAINING_PID=""

# ── Sync to persistent NFS ───────────────────────────────────
echo ""
echo "============================================"
echo "Training exited with code: $EXIT_CODE"
echo "End time: $(date)"

# Find the Hydra output directory (most recent with checkpoints)
HYDRA_RUN_DIR=$(find "$IRIS_DIR/outputs" -path "*/checkpoints/last.pt" -printf '%T@ %h\n' 2>/dev/null \
    | sort -rn | head -1 | awk '{print $2}' | sed 's|/checkpoints$||')

if [ -n "$HYDRA_RUN_DIR" ] && [ -d "$HYDRA_RUN_DIR/checkpoints" ]; then
    echo "Syncing to NFS: $NFS_RUN_DIR"
    mkdir -p "$NFS_RUN_DIR"

    # Sync checkpoints
    rsync -a --info=progress2 "$HYDRA_RUN_DIR/checkpoints/" "$NFS_RUN_DIR/checkpoints/"

    # Sync wandb directory (contains run ID needed for resumption)
    if [ -d "$HYDRA_RUN_DIR/wandb" ]; then
        rsync -a "$HYDRA_RUN_DIR/wandb/" "$NFS_RUN_DIR/wandb/"
    fi

    # Sync Hydra config for reference
    if [ -d "$HYDRA_RUN_DIR/.hydra" ]; then
        rsync -a "$HYDRA_RUN_DIR/.hydra/" "$NFS_RUN_DIR/.hydra/"
    fi
    if [ -d "$HYDRA_RUN_DIR/config" ]; then
        rsync -a "$HYDRA_RUN_DIR/config/" "$NFS_RUN_DIR/config/"
    fi

    # Save metadata
    cat > "$NFS_RUN_DIR/run_info.txt" <<INFO
game=$GAME
seed=$SEED
max_len=$MAX_LEN
exit_code=$EXIT_CODE
local_hydra_dir=$HYDRA_RUN_DIR
synced_at=$(date -Iseconds)
hostname=$(hostname)
INFO

    echo "Sync complete."
else
    echo "WARNING: No checkpoint found to sync."
fi

echo "============================================"
exit $EXIT_CODE
