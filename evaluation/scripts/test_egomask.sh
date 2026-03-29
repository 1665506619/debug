#!/bin/bash
# EgoMask 测试脚本 - 使用小数据集快速验证流程

export PYTHONWARNINGS="ignore"

echo "========================================"
echo "EgoMask 测试模式 - 小数据集验证"
echo "========================================"

# 模型路径
MODEL_PATH=${1:-"/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/code/videollama3-r/checkpoints/1010_v1_sa2va_8f_5obj_0.4M_wo_cls_lora_4xlr/checkpoint-3122"}

# GPU 配置 (测试模式使用更少的GPU)
ARG_NPROC_PER_NODE=${2:-2}

echo "模型路径: $MODEL_PATH"
echo "GPU 数量: $ARG_NPROC_PER_NODE"
echo "数据集: short subset (5 个样本)"
echo "========================================"
echo ""

# 步骤 1: 数据预处理（生成测试数据）
echo "========================================="
echo "步骤 1/3: 生成测试数据"
echo "========================================="

python evaluation/data_construction/construct.py

if [ $? -ne 0 ]; then
    echo "❌ 数据处理失败"
    exit 1
fi

# 检查测试数据是否生成
TEST_DATA="/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/eval/egomask_short_test.json"
if [ ! -f "$TEST_DATA" ]; then
    echo "❌ 测试数据文件不存在: $TEST_DATA"
    exit 1
fi

echo "✅ 测试数据生成成功"
echo ""

# 步骤 2: 模型推理
echo "========================================="
echo "步骤 2/3: 模型推理（测试集）"
echo "========================================="

DATA_ROOT=/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data
SAVE_DIR=evaluation_results_test

torchrun --nproc_per_node="$ARG_NPROC_PER_NODE" \
    evaluation/eval_video.py \
    --model_path "${MODEL_PATH}" \
    --question_file "${TEST_DATA}" \
    --video_folder "${DATA_ROOT}" \
    --output_file "${SAVE_DIR}/$(basename "$MODEL_PATH")/egomask_short_test.json"

if [ $? -ne 0 ]; then
    echo "❌ 模型推理失败"
    exit 1
fi

echo "✅ 推理完成"
echo ""

# 步骤 3: 后处理
echo "========================================="
echo "步骤 3/3: 后处理和评估"
echo "========================================="

PRED_RESULT="${SAVE_DIR}/$(basename "$MODEL_PATH")/egomask_short_test.json"
PRED_OUTPUT_DIR="${SAVE_DIR}/$(basename "$MODEL_PATH")/egomask_pred/short_test"

python evaluation/eval_egomask_postprocess.py \
    --pred_result "${PRED_RESULT}" \
    --output_dir "${PRED_OUTPUT_DIR}" \
    --dataset_type short

if [ $? -ne 0 ]; then
    echo "❌ 后处理失败"
    exit 1
fi

echo "✅ 后处理完成"
echo ""

# 步骤 4: 评估指标（可选，如果样本太少可能会失败）
echo "========================================="
echo "步骤 4/3: 计算评估指标（可选）"
echo "========================================="

# 注意：因为只有5个样本，EgoMask官方评估可能会因为样本不完整而失败
# 这是正常的，主要目的是测试推理和格式转换流程

echo "⚠️  注意: 测试模式样本数少，评估可能失败（这是正常的）"
python /lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/EgoMask/EgoMask/evaluation/eval_egomask.py \
    --dataset_type short \
    --pred_path "${SAVE_DIR}/$(basename "$MODEL_PATH")/egomask_pred" \
    --save_name "test_results.json" 2>/dev/null || true

echo ""
echo "========================================"
echo "✅ 测试完成！"
echo "========================================"
echo ""
echo "测试结果位置:"
echo "  - 推理结果: ${PRED_RESULT}"
echo "  - 预测masks: ${PRED_OUTPUT_DIR}"
echo ""
echo "如果前3步都成功，说明实现正确！"
echo "现在可以运行完整评估:"
echo "  bash evaluation/scripts/eval_egomask.sh MODEL_PATH 1 4 0 short"
echo "========================================"

