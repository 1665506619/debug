#!/bin/bash
# 准备 EgoMask 评测数据
# 用于生成评测所需的 JSON 文件

export PYTHONWARNINGS="ignore"

DATASET_TYPE=${1:-"long"}  # 默认准备 long subset

echo "========================================="
echo "准备 EgoMask 评测数据"
echo "========================================="
echo "数据集类型: $DATASET_TYPE"
echo ""

# 备份原始 construct.py
if [ ! -f "evaluation/data_construction/construct.py.bak" ]; then
    echo "备份原始 construct.py..."
    cp evaluation/data_construction/construct.py evaluation/data_construction/construct.py.bak
fi

# 创建临时的 construct.py
cat > evaluation/data_construction/construct_temp.py << 'EOF'
import sys
sys.path.append('./')

# 导入原始的 SEG_DATA 类
from evaluation.data_construction.construct import SEG_DATA

# 创建实例并运行
data = SEG_DATA()

# 根据命令行参数决定处理哪个 split
import sys
if len(sys.argv) > 1:
    split = sys.argv[1]
else:
    split = 'long'

print(f'正在处理 EgoMask {split} subset...')
data.EgoMask_EVAL(split=split)
print(f'✅ EgoMask {split} subset 数据准备完成！')
EOF

# 运行数据构造
echo "正在生成 ${DATASET_TYPE} 数据集..."
python evaluation/data_construction/construct_temp.py "${DATASET_TYPE}"

if [ $? -ne 0 ]; then
    echo "❌ 数据构造失败"
    rm -f evaluation/data_construction/construct_temp.py
    exit 1
fi

# 清理临时文件
rm -f evaluation/data_construction/construct_temp.py

# 检查输出文件
OUTPUT_FILE="/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/eval/egomask_${DATASET_TYPE}.json"

if [ -f "$OUTPUT_FILE" ]; then
    # 统计样本数量
    SAMPLE_COUNT=$(python -c "import json; print(len(json.load(open('$OUTPUT_FILE'))))" 2>/dev/null || echo "unknown")
    echo ""
    echo "========================================="
    echo "✅ 数据准备完成！"
    echo "========================================="
    echo "数据集: ${DATASET_TYPE}"
    echo "样本数量: ${SAMPLE_COUNT}"
    echo "输出文件: ${OUTPUT_FILE}"
    echo "========================================="
else
    echo "❌ 错误: 输出文件不存在: $OUTPUT_FILE"
    exit 1
fi

echo ""
echo "现在可以运行评测了："
echo "bash evaluation/scripts/eval_checkpoint_1517.sh 4 ${DATASET_TYPE}"

