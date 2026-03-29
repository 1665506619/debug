#!/bin/bash
export PYTHONWARNINGS="ignore"
# export CUDA_VISIBLE_DEVICES=4,5,6,7

MODEL_PATH=${1:-"your_model_path_here"}
SPLIT=${2:-"object_segmentation"}

ARG_WORLD_SIZE=${3:-1}
ARG_NPROC_PER_NODE=${4:-4}

ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16667
ARG_RANK=${5:-0}

if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "MODEL_PATH: $MODEL_PATH"

SAVE_DIR=./evaluation_results
DATA_ROOT=/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/RynnEC-Bench

DATASET="rynnec_${SPLIT}"
QUESTION_FILE="/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/eval/${DATASET}.json"

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node 8 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    evaluation/eval_rynnec.py \
    --model_path "/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/work_dirs/${MODEL_PATH}" \
    --question_file ${QUESTION_FILE} \
    --video_folder ${DATA_ROOT} \
    --output_file "${SAVE_DIR}/$MODEL_PATH/${DATASET}.json" \
#    --only_mask_img
