#!/bin/bash
# 批量运行所有 benchmark 评估

set -e  # 遇到错误立即退出

# 模型路径
MODEL_PATH=${1:-"/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/code/videollama3-r/checkpoints/1010_v1_sa2va_8f_5obj_0.4M_wo_cls_lora_4xlr/checkpoint-3122"}

# GPU 配置
WORLD_SIZE=${2:-1}
NPROC_PER_NODE=${3:-4}
RANK=${4:-0}

echo "========================================"
echo "批量评估所有 Benchmarks"
echo "========================================"
echo "模型路径: $MODEL_PATH"
echo "GPU 配置: ${NPROC_PER_NODE} GPUs"
echo "========================================"
echo ""

# 1. Ref-YouTube-VOS
echo "📊 [1/7] 开始评估 Ref-YouTube-VOS..."
bash evaluation/scripts/eval_ref_youtube_vos.sh $MODEL_PATH $WORLD_SIZE $NPROC_PER_NODE $RANK
echo "✅ Ref-YouTube-VOS 评估完成"
echo ""

# 2. ReasonVOS
echo "📊 [2/7] 开始评估 ReasonVOS..."
bash evaluation/scripts/eval_reason_vos.sh $MODEL_PATH $WORLD_SIZE $NPROC_PER_NODE $RANK
echo "✅ ReasonVOS 评估完成"
echo ""

# 3. LISA-Reasoning
echo "📊 [3/7] 开始评估 LISA-Reasoning..."
bash evaluation/scripts/eval_reasonseg.sh $MODEL_PATH $WORLD_SIZE $NPROC_PER_NODE $RANK
echo "✅ LISA-Reasoning 评估完成"
echo ""

# 4. RynnEC-Bench
echo "📊 [4/7] 开始评估 RynnEC-Bench..."
bash evaluation/scripts/eval_rynnec.sh $MODEL_PATH $WORLD_SIZE $NPROC_PER_NODE $RANK
echo "✅ RynnEC-Bench 评估完成"
echo ""

# 5. EgoMask - Long
echo "📊 [5/7] 开始评估 EgoMask (long)..."
bash evaluation/scripts/eval_egomask.sh $MODEL_PATH $WORLD_SIZE $NPROC_PER_NODE $RANK long
echo "✅ EgoMask (long) 评估完成"
echo ""

# 6. EgoMask - Medium
echo "📊 [6/7] 开始评估 EgoMask (medium)..."
bash evaluation/scripts/eval_egomask.sh $MODEL_PATH $WORLD_SIZE $NPROC_PER_NODE $RANK medium
echo "✅ EgoMask (medium) 评估完成"
echo ""

# 7. EgoMask - Short
echo "📊 [7/7] 开始评估 EgoMask (short)..."
bash evaluation/scripts/eval_egomask.sh $MODEL_PATH $WORLD_SIZE $NPROC_PER_NODE $RANK short
echo "✅ EgoMask (short) 评估完成"
echo ""

echo "========================================"
echo "🎉 所有 Benchmark 评估完成！"
echo "========================================"
echo ""
echo "结果保存位置:"
echo "  evaluation_results/$(basename "$MODEL_PATH")/"
echo ""
echo "使用以下命令查看结果汇总:"
echo "  python evaluation/collect.py --result_dir evaluation_results/$(basename "$MODEL_PATH")"
echo "========================================"

