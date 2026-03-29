# EgoMask 评估指南

## 概述

EgoMask 是一个第一人称视角的视频分割数据集，包含三个子集：
- **long**: 长视频片段 (来自 egotracks)
- **medium**: 中等长度视频片段 (来自 egotracks)
- **short**: 短视频片段 (来自 refego)
- **full**: 完整数据集 (包含所有子集)

## 数据路径配置

### 服务器上的数据路径：
- **Annotation 路径**: `/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/EgoMask/egomask`
- **图片路径**: `/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/EgoMask/EgoMask/dataset/egomask`
- **模型检查点**: `/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/code/videollama3-r/checkpoints/1010_v1_sa2va_8f_5obj_0.4M_wo_cls_lora_4xlr/checkpoint-3122`

## 评估流程

### 步骤 1: 数据预处理

运行数据构造脚本，将 EgoMask 原始数据转换为评估格式：

```bash
cd /lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/code/videollama3-r

# 处理不同的 subset
python evaluation/data_construction/construct.py
```

在 `construct.py` 中取消注释相应的行：

```python
data = SEG_DATA()

# 处理 EgoMask 数据集
data.EgoMask_EVAL(split='long')      # 处理 long subset
# data.EgoMask_EVAL(split='medium')  # 处理 medium subset
# data.EgoMask_EVAL(split='short')   # 处理 short subset
# data.EgoMask_EVAL(split='full')    # 处理完整数据集
```

处理完成后，数据将保存到：
- `/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/eval/egomask_long.json`
- `/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/eval/egomask_medium.json`
- `/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/eval/egomask_short.json`
- `/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/eval/egomask_full.json`

### 步骤 2: 运行评估

使用评估脚本进行模型推理和评估：

```bash
# 评估 long subset (使用 4 张 GPU)
bash evaluation/scripts/eval_egomask.sh \
    /lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/code/videollama3-r/checkpoints/1010_v1_sa2va_8f_5obj_0.4M_wo_cls_lora_4xlr/checkpoint-3122 \
    1 \
    4 \
    0 \
    long

# 评估 medium subset
bash evaluation/scripts/eval_egomask.sh \
    /lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/code/videollama3-r/checkpoints/1010_v1_sa2va_8f_5obj_0.4M_wo_cls_lora_4xlr/checkpoint-3122 \
    1 \
    4 \
    0 \
    medium

# 评估 short subset
bash evaluation/scripts/eval_egomask.sh \
    /lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/code/videollama3-r/checkpoints/1010_v1_sa2va_8f_5obj_0.4M_wo_cls_lora_4xlr/checkpoint-3122 \
    1 \
    4 \
    0 \
    short

# 评估完整数据集
bash evaluation/scripts/eval_egomask.sh \
    /lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/code/videollama3-r/checkpoints/1010_v1_sa2va_8f_5obj_0.4M_wo_cls_lora_4xlr/checkpoint-3122 \
    1 \
    4 \
    0 \
    full
```

### 脚本参数说明：

```bash
bash evaluation/scripts/eval_egomask.sh [MODEL_PATH] [WORLD_SIZE] [NPROC_PER_NODE] [RANK] [DATASET_TYPE]
```

- **MODEL_PATH**: 模型检查点路径（默认已配置）
- **WORLD_SIZE**: 节点数量（默认: 1）
- **NPROC_PER_NODE**: 每个节点的 GPU 数量（默认: 4，可根据实际情况调整）
- **RANK**: 节点排名（默认: 0）
- **DATASET_TYPE**: 数据集类型 - long/medium/short/full（默认: long）

### 步骤 3: 评估流程说明

评估脚本会自动执行三个步骤：

1. **模型推理**: 使用 `evaluation/eval_video.py` 进行视频分割推理
2. **结果后处理**: 使用 `evaluation/eval_egomask_postprocess.py` 将推理结果转换为 EgoMask 评估格式
3. **计算指标**: 使用 `EgoMask-main/evaluation/eval_egomask.py` 计算 EgoMask 官方指标

### 步骤 4: 查看结果

评估结果将保存在以下位置：

```
evaluation_results/checkpoint-3122/
├── egomask_long.json                    # 模型推理原始结果
├── egomask_pred/                        # EgoMask 格式的预测结果
│   └── long/
│       ├── video_id_1/
│       │   └── exp_id_1/
│       │       └── exp_id_1-obj_id.json
│       ├── video_id_2/
│       │   └── exp_id_1/
│       │       └── exp_id_1-obj_id.json
│       └── default_results_long.json    # 最终评估指标
└── ...
```

### 评估指标说明

EgoMask 评估会计算以下指标：

- **J (Region Similarity)**: IoU 指标
- **F (Contour Accuracy)**: 边界准确度
- **J&F**: J 和 F 的平均值
- **T_f1, T_acc, T_precision, T_recall**: 时序一致性指标
- **iou_overall**: 所有帧的平均 IoU
- **iou_gold**: 仅在 gold frame 上的 IoU
- **iou_gold_with_pred**: gold frame 和预测帧上的 IoU
- **detection_flag**: 检测成功率
- **detection_bbox_iou**: 检测框 IoU
- **detection_mask_iou**: 检测 mask IoU

## 调试模式

如果需要在 4h4gpu 调试机器上运行：

```bash
# 1. 申请调试机器
cd /lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/
srun -p interactive_singlenode --time=240 --account=nvr_lpr_nvgptvision \
    --container-image=./region.sqsh --container-save=./region.sqsh \
    --container-mounts=/lustre/ --gres=gpu:4 --cpus-per-task=120 --pty bash

# 2. 进入代码目录
cd /lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/code/videollama3-r

# 3. 运行评估（使用较小的 subset 进行测试）
bash evaluation/scripts/eval_egomask.sh \
    /lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/code/videollama3-r/checkpoints/1010_v1_sa2va_8f_5obj_0.4M_wo_cls_lora_4xlr/checkpoint-3122 \
    1 \
    4 \
    0 \
    short

# 4. 任务完成后释放机器
# Ctrl+D
```

## 常见问题

### Q1: 数据路径错误？
A: 确保在服务器上运行，并且数据路径正确指向 EgoMask 数据集

### Q2: GPU 数量不够？
A: 可以通过修改 `--nproc_per_node` 参数来调整使用的 GPU 数量，例如使用 2 张 GPU：
```bash
bash evaluation/scripts/eval_egomask.sh MODEL_PATH 1 2 0 long
```

### Q3: 如何只运行推理而不评估？
A: 可以直接运行：
```bash
torchrun --nproc_per_node=4 evaluation/eval_video.py \
    --model_path YOUR_MODEL_PATH \
    --question_file /lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/eval/egomask_long.json \
    --video_folder /lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data \
    --output_file evaluation_results/YOUR_CHECKPOINT/egomask_long.json
```

### Q4: 如何只运行评估而不重新推理？
A: 如果已经有推理结果，可以从步骤 2 开始：
```bash
# 后处理
python evaluation/eval_egomask_postprocess.py \
    --pred_result evaluation_results/YOUR_CHECKPOINT/egomask_long.json \
    --output_dir evaluation_results/YOUR_CHECKPOINT/egomask_pred/long \
    --dataset_type long

# 评估
python EgoMask-main/evaluation/eval_egomask.py \
    --dataset_type long \
    --pred_path evaluation_results/YOUR_CHECKPOINT/egomask_pred \
    --save_name results_long.json
```

## 总结

EgoMask 评估已完全集成到项目中，您只需：

1. ✅ 运行 `construct.py` 预处理数据（只需运行一次）
2. ✅ 运行 `eval_egomask.sh` 进行评估
3. ✅ 查看 `default_results_*.json` 获取评估指标

所有路径都已配置好，直接使用即可！

