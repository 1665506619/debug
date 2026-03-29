#!/bin/bash
export PYTHONWARNINGS="ignore"
# export CUDA_VISIBLE_DEVICES=4,5,6,7

MODEL_PATH=${1:-"1228_pretrain_v5_50_lora"}

ARG_WORLD_SIZE=${2:-1}
ARG_NPROC_PER_NODE=${3:-2}

ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16668
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


echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "MODEL_PATH: $MODEL_PATH"


SAVE_DIR=evaluation_results
DATA_ROOT=/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data

# DATASET_LIST=("grefcoco_val" "grefcoco_testA" "grefcoco_testB")
DATASET_LIST=("grefcoco_val")


for DATASET in "${DATASET_LIST[@]}"; do
    echo "run ${DATASET}..."
    QUESTION_FILE="/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/eval/${DATASET}.json"

    torchrun --nnodes="$WORLD_SIZE" \
        --nproc_per_node=8 \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        --node_rank="$RANK" \
        evaluation/eval_refcoco.py \
        --model_path "work_dirs/${MODEL_PATH}" \
        --question_file "${QUESTION_FILE}" \
        --image_folder "${DATA_ROOT}" \
        --output_file "${SAVE_DIR}/${MODEL_PATH}/${DATASET}.json"
done