#!/bin/bash

# RoboRefIt Data Construction Script
# This script converts RoboRefIt raw data to evaluation format

echo "========================================="
echo "RoboRefIt Data Construction"
echo "========================================="

DATA_ROOT="/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/RoboRefIt"
OUTPUT_DIR="/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/eval"

echo "Data root: ${DATA_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "========================================="
echo ""

# Run construction script
python evaluation/data_construction/construct_roborefit.py \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --splits testA testB

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ Data construction completed!"
    echo "========================================="
    echo "Output files:"
    ls -lh "${OUTPUT_DIR}"/roborefit_*.json
    echo "========================================="
else
    echo ""
    echo "========================================="
    echo "❌ Data construction failed!"
    echo "========================================="
    exit 1
fi

