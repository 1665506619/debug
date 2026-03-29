#!/bin/bash
export WANDB_PROJECT=qwen3_vl
WORK_DIR=work_dirs
RUN_NAME=test
OUTPUT_DIR=$WORK_DIR/$WANDB_PROJECT/$RUN_NAME

MODEL_ARGS=(
    --model_path /mnt/damovl/CKPT/Qwen3-VL-2B-Instruct
    --gradient_checkpointing True
    --use_liger_kernel False
)

DATA_ARGS=(
    --ann_path brain_v3_20.jsonl
    --model_max_length 16384
    --mm_max_length 10240
    --fps 2
    --max_frames 512
    --per_device_train_batch_size 4
    --gradient_accumulation_steps 1
    --num_train_epochs 1
    --remove_unused_columns False
)

OPTIMIZER_ARGS=(
    --llm_lr 1e-5
    --projector_lr 1e-5
    --vision_encoder_lr 2e-6
    --weight_decay 0.0
    --warmup_ratio 0.03
    --lr_scheduler_type "cosine"
)

TRAINING_ARGS=(
    --deepspeed scripts/zero1.json
    --bf16 True
    --tf32 True
    --fp16 False
    --dataloader_num_workers 8
    --decoder_load_balancing True
    --loss_reduction_scope sequence
    --average_tokens_across_devices True
)

LOG_ARGS=(
    --output_dir $OUTPUT_DIR
    --run_name $RUN_NAME
    --logging_steps 1
    --report_to tensorboard
    --save_strategy "steps"
    --save_steps 2000
    --save_total_limit 2
)

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --node_rank $RANK \
    --rdzv_conf="timeout=7200,join_timeout=7200" \
    -m easy_vlm.api.train \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${LOG_ARGS[@]}
