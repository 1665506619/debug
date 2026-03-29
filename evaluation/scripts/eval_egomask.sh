#!/bin/bash
export PYTHONWARNINGS="ignore"

# 模型路径
MODEL_PATH=${1:-"1127_v2_lora/checkpoint-8623"}

# EgoMask 数据集类型: long, medium, short, full
DATASET_TYPE=${2:-"medium"}

# 分布式训练参数
ARG_WORLD_SIZE=${3:-1}
ARG_NPROC_PER_NODE=${4:-4}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16668
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

echo "========================================="
echo "EgoMask 评估配置"
echo "========================================="
echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "MODEL_PATH: $MODEL_PATH"
echo "DATASET_TYPE: $DATASET_TYPE"
echo "========================================="

# 路径配置
SAVE_DIR=evaluation_results
DATA_ROOT=/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data
DATASET="egomask_${DATASET_TYPE}"
QUESTION_FILE="/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/eval/${DATASET}.json"


# 第一步：模型推理
echo ""
echo "========================================="
echo "步骤 1/3: 运行模型推理"
echo "========================================="

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

if [ $? -ne 0 ]; then
    echo "错误: 模型推理失败"
    exit 1
fi

echo "推理完成！"

# 第二步：后处理 - 将推理结果转换为 EgoMask 评估格式
echo ""
echo "========================================="
echo "步骤 2/3: 后处理推理结果"
echo "========================================="

PRED_RESULT="${SAVE_DIR}/$MODEL_PATH/${DATASET}.json"
PRED_OUTPUT_DIR="${SAVE_DIR}/$MODEL_PATH/egomask_pred/${DATASET_TYPE}"

python evaluation/eval_egomask_postprocess.py \
    --pred_result "${PRED_RESULT}" \
    --output_dir "${PRED_OUTPUT_DIR}" \
    --dataset_type "${DATASET_TYPE}"

if [ $? -ne 0 ]; then
    echo "错误: 后处理失败"
    exit 1
fi

echo "后处理完成！"

# 第三步：使用 EgoMask 官方评估脚本计算指标
echo ""
echo "========================================="
echo "步骤 3/3: 计算 EgoMask 评估指标"
echo "========================================="


python evaluation/egomask/eval_egomask.py \
    --dataset_type "${DATASET_TYPE}" \
    --pred_path "${SAVE_DIR}/$MODEL_PATH/egomask_pred" \
    --save_name "results_${DATASET_TYPE}.json"
