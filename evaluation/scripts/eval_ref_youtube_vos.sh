#!/bin/bash
export PYTHONWARNINGS="ignore"

MODEL_PATH=${1:-"1125_new_v0_lora/checkpoint-3000"}

ARG_WORLD_SIZE=${2:-1}
ARG_NPROC_PER_NODE=${3:-4} # 默认 4 卡
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16670
ARG_RANK=${4:-0}

if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "=================================================="
echo "Ref-YouTube-VOS Evaluation"
echo "Model Path: $MODEL_PATH"
echo "GPUs: $NPROC_PER_NODE (per node)"
echo "=================================================="

SAVE_DIR=./evaluation_results
DATA_ROOT=/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data
DATASET="ref_youtube_vos_valid"
QUESTION_FILE="${DATA_ROOT}/eval/${DATASET}.json"


torchrun --nnodes="$WORLD_SIZE" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --node_rank="$RANK" \
    evaluation/eval_video.py \
    --model_path "/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/work_dirs/${MODEL_PATH}" \
    --question_file "${QUESTION_FILE}" \
    --video_folder "${DATA_ROOT}" \
    --output_file "${SAVE_DIR}/$MODEL_PATH/${DATASET}.json"
