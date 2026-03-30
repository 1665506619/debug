#!/bin/bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${REPO_ROOT}"

export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# Conservative defaults for 8x A100 80G. Override by env if you want to push harder.
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
export MAX_FRAMES="${MAX_FRAMES:-32}"
export FPS="${FPS:-1}"
export MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-8192}"
export MM_MAX_LENGTH="${MM_MAX_LENGTH:-4096}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export RUN_NAME="${RUN_NAME:-revos_video_v1_8gpu}"

exec bash debug/scripts/train_revos_video.sh "$@"
