#!/bin/bash
NPROC_PER_NODE=$gpu_per_pod
echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"


pip install /mnt/workspace/workgroup/yuanyq/code/transformers

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

WORK_DIR=/mnt/workspace/workgroup/yuanyq/code/video_seg/EasyVLM/work_dirs
RUN_NAME=1220_pretrain_v1_lora
OUTPUT_DIR=$WORK_DIR/$RUN_NAME

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

MODEL_ARGS=(
    --model_path /mnt/workspace/workgroup/yuanyq/checkpoints/Qwen3-VL-2B-Instruct
    --gradient_checkpointing True
    --use_liger_kernel False
)

DATA_ARGS=(
    --ann_path /mnt/workspace/workgroup/yuanyq/code/video_seg/EasyVLM/data/v1.txt
    --data_root /mnt/workspace/workgroup/yuanyq/video_data
    --data_path_root /mnt/workspace/workgroup/yuanyq/code/video_seg/datasets/new_format
    --model_max_length 16384
    --mm_max_length 10240
    --fps 2
    --max_frames 512
    --per_device_train_batch_size 2
    --gradient_accumulation_steps 1
    --num_train_epochs 3
    --remove_unused_columns False
)

OPTIMIZER_ARGS=(
    --llm_lr 1e-5
    --projector_lr 1e-5
    --vision_encoder_lr 2e-6
    --sam_decoder_lr 5e-5
    --weight_decay 0.0
    --warmup_ratio 0.03
    --lr_scheduler_type "cosine"
)

TRAINING_ARGS=(
    --deepspeed scripts/zero1.json
    --bf16 True
    --lora_enable True
    --bf16 True
    --tf32 True
    --fp16 False
    --dataloader_num_workers 16
    --loss_reduction_scope sequence
    --average_tokens_across_devices True
)

LOG_ARGS=(
    --output_dir $OUTPUT_DIR
    --run_name $RUN_NAME
    --logging_steps 1
    --report_to tensorboard
    --save_strategy "steps"
    --save_steps 1000
    --save_total_limit 2
)

set -x

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --node_rank $RANK \
    --rdzv_conf="timeout=7200,join_timeout=7200" \
    -m easy_vlm.train \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${LOG_ARGS[@]} 2>&1 | tee -a $OUTPUT_DIR/$RANK.log
