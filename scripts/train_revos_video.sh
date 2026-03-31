#!/bin/bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/debug:${PYTHONPATH:-}"

PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

MODEL_PATH=${MODEL_PATH:-weights/pretrain_weights}
SAM3_MODEL=${SAM3_MODEL:-}
DATA_ROOT=${DATA_ROOT:-/root/InstructSAM/video_dataset/revos}
META_PATH=${META_PATH:-${DATA_ROOT}/meta_expressions_train_.json}
MASK_DICT_PATH=${MASK_DICT_PATH:-${DATA_ROOT}/mask_dict.json}
TRAIN_JSON=${TRAIN_JSON:-${REPO_ROOT}/outputs/revos_train/revos_train_sft.json}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/outputs/revos_train/checkpoints}
RUN_NAME=${RUN_NAME:-revos_video_v1}

MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-8192}
MM_MAX_LENGTH=${MM_MAX_LENGTH:-4096}
FPS=${FPS:-1}
MAX_FRAMES=${MAX_FRAMES:-32}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
SAVE_STEPS=${SAVE_STEPS:-500}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
MAX_SEG_NUMS=${MAX_SEG_NUMS:-10}
LLM_LR=${LLM_LR:-2e-6}
PROJECTOR_LR=${PROJECTOR_LR:-2e-6}
VISION_ENCODER_LR=${VISION_ENCODER_LR:-2e-6}
SAM_DECODER_LR=${SAM_DECODER_LR:-5e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-cosine}
REPORT_TO=${REPORT_TO:-none}
BADCASE_LOG_PATH=${BADCASE_LOG_PATH:-${OUTPUT_DIR}/badcases.jsonl}

mkdir -p "$(dirname "${TRAIN_JSON}")" "${OUTPUT_DIR}"

if [ -z "${SAM3_MODEL}" ]; then
  echo "SAM3_MODEL is not set."
  echo "Please point it to a local SAM3 checkpoint file or model directory."
  echo "Examples:"
  echo "  SAM3_MODEL=weights/sam3/sam3.pt bash debug/scripts/train_revos_video.sh"
  echo "  SAM3_MODEL=/path/to/local/sam3_dir bash debug/scripts/train_revos_video.sh"
  exit 1
fi

if [ ! -e "${SAM3_MODEL}" ]; then
  echo "SAM3_MODEL path does not exist: ${SAM3_MODEL}"
  exit 1
fi

if [ ! -f "${TRAIN_JSON}" ]; then
  "${PYTHON_BIN}" debug/evaluation/data_construction/construct_revos_train.py \
    --meta-path "${META_PATH}" \
    --mask-dict-path "${MASK_DICT_PATH}" \
    --data-root "${DATA_ROOT}" \
    --output-path "${TRAIN_JSON}"
fi

set -x

export OMP_NUM_THREADS
export BADCASE_LOG_PATH

"${TORCHRUN_BIN}" \
  --nnodes "${NNODES}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  -m easy_vlm.train \
  --model_path "${MODEL_PATH}" \
  --mask_decoder_model "${SAM3_MODEL}" \
  --seg_encoder sam3 \
  --seg_decoder sam3 \
  --ann_path "${TRAIN_JSON}" \
  --data_root "${DATA_ROOT}" \
  --data_path_root "${REPO_ROOT}" \
  --data_cache_dir "${OUTPUT_DIR}/hf_cache" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_NAME}" \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --mm_max_length "${MM_MAX_LENGTH}" \
  --fps "${FPS}" \
  --max_frames "${MAX_FRAMES}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --remove_unused_columns False \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --report_to "${REPORT_TO}" \
  --group_by_modality_length True \
  --use_multi_objs False \
  --skip_none False \
  --attn_implementation "${ATTN_IMPLEMENTATION}" \
  --max_seg_nums "${MAX_SEG_NUMS}" \
  --bf16 True \
  --fp16 False \
  --tf32 True \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --llm_lr "${LLM_LR}" \
  --projector_lr "${PROJECTOR_LR}" \
  --vision_encoder_lr "${VISION_ENCODER_LR}" \
  --sam_decoder_lr "${SAM_DECODER_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
  "$@"
