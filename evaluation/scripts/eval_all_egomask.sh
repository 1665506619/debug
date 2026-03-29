#!/bin/bash

# EgoMask 完整评估脚本（short + medium + long）
# 用法: bash evaluation/scripts/eval_all_egomask.sh

set -e  # 遇到错误立即退出

MODEL_PATH="/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/code/videollama3-r/checkpoints/1010_v1_sa2va_8f_5obj_0.4M_wo_cls_lora_4xlr/checkpoint-3122"
CODE_DIR="/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/code/videollama3-r"

cd "$CODE_DIR"

echo "========================================="
echo "   EgoMask 完整评估（short/medium/long）"
echo "========================================="

# ========================================
# 第一步：构建数据集
# ========================================
echo ""
echo "========== 第一步：构建数据集 =========="
echo ""

# 保存原始 construct.py
cp evaluation/data_construction/construct.py evaluation/data_construction/construct.py.bak

for split in short medium long; do
    echo "正在构建 ${split} 数据集..."
    
    # 修改最后一行
    sed -i "s/data.EgoMask_EVAL(split='.*')/data.EgoMask_EVAL(split='${split}')/" evaluation/data_construction/construct.py
    
    # 运行构建
    python evaluation/data_construction/construct.py
    
    # 检查结果
    if [ -f "data/eval/egomask_${split}.json" ]; then
        sample_count=$(python -c "import json; print(len(json.load(open('data/eval/egomask_${split}.json'))))")
        echo "✅ ${split} 数据集构建完成：${sample_count} 样本"
    else
        echo "❌ ${split} 数据集构建失败！"
        exit 1
    fi
    echo ""
done

# 恢复原始文件
mv evaluation/data_construction/construct.py.bak evaluation/data_construction/construct.py

# ========================================
# 第二步：运行评估
# ========================================
echo ""
echo "========== 第二步：运行评估 =========="
echo ""

for split in short medium long; do
    echo "========================================="
    echo "开始评估 ${split} 数据集"
    echo "========================================="
    
    start_time=$(date +%s)
    
    # 运行评估
    bash evaluation/scripts/eval_egomask.sh "$MODEL_PATH" 1 4 0 "$split"
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo ""
    echo "✅ ${split} 评估完成！耗时: ${duration} 秒 ($((duration/60)) 分钟)"
    echo ""
    
    # 显示结果
    result_file="evaluation_results/checkpoint-3122/egomask_pred/${split}/default_results_${split}.json"
    if [ -f "$result_file" ]; then
        echo "===== ${split} 结果 ====="
        cat "$result_file"
        echo ""
    fi
done

# ========================================
# 第三步：汇总结果
# ========================================
echo ""
echo "========================================="
echo "         所有评估结果汇总"
echo "========================================="
echo ""

for split in short medium long; do
    result_file="evaluation_results/checkpoint-3122/egomask_pred/${split}/default_results_${split}.json"
    
    if [ -f "$result_file" ]; then
        echo "===== ${split} ====="
        cat "$result_file"
        echo ""
    else
        echo "===== ${split} ====="
        echo "❌ 未找到结果文件"
        echo ""
    fi
done

echo "========================================="
echo "   所有评估完成！"
echo "========================================="

