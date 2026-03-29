#!/bin/bash
GIT_SSH_COMMAND="ssh -i /lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/smajumdar/region/sshkey" git pull origin sam3

echo "MASTER_ADDR=$MASTER_ADDR"
n_node=$SLURM_JOB_NUM_NODES
echo "number of nodes:" $n_node
echo "node rank:" $SLURM_PROCID

if [[ "${SLURM_PROCID}" -eq 0 ]]; then
    echo "==== GPU Model on node rank 0 ===="
    nvidia-smi --query-gpu=name --format=csv,noheader
    echo "==============================================="
fi

export TRANSFORMERS_OFFLINE=1
export WANDB_API_KEY="ee8b73e5623332aab6d3ddbe9bd2b4ccb44ecd62"
export WANDB_PROJECT='Star_Nemotron'
export WANDB_LOG_MODEL=False
echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

WORK_DIR=./work_dirs
RUN_NAME=0109_pretrain_v7_w_none_multi_objs_lora
OUTPUT_DIR=$WORK_DIR/$RUN_NAME

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

MODEL_ARGS=(
    --model_path /lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/checkpoint/Qwen3-VL-2B-Instruct
    --mask_decoder_model /lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/checkpoint/sam3
    --gradient_checkpointing True
    --use_liger_kernel False
)

DATA_ARGS=(
    --ann_path ./data/v7.txt
    --data_root /lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data
    --data_path_root /lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/seg_train/seg-train/new
    --data_cache_dir /lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/seg_train/seg-train/cache
    --model_max_length 16384
    --mm_max_length 8192
    --fps 2
    --max_frames 512
    --per_device_train_batch_size 2
    --gradient_accumulation_steps 1
    --num_train_epochs 1
    --remove_unused_columns False
    --use_multi_objs True
    --skip_none False
)

OPTIMIZER_ARGS=(
    --llm_lr 5e-6
    --projector_lr 5e-6
    --vision_encoder_lr 5e-6
    --sam_decoder_lr 1e-5
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
    --loss_reduction_scope batch
    --average_tokens_across_devices False
    --group_by_modality_length True
)

LOG_ARGS=(
    --output_dir $OUTPUT_DIR
    --run_name $RUN_NAME
    --logging_steps 1
    --report_to "wandb"
    --use_onelogger True
    --save_strategy "steps"
    --save_steps 1000
    --save_total_limit 2
)

set -x

torchrun --nnodes $SLURM_NNODES \
    --node_rank $SLURM_PROCID \
    --nproc_per_node 8 \
    --master_addr $MASTER_ADDR \
    --master_port 25031 \
    -m easy_vlm.train \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${LOG_ARGS[@]} 2>&1 | tee -a logs/${RUN_NAME}_${SLURM_PROCID}.log
