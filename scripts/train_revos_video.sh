#!/bin/bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/debug:${PYTHONPATH:-}"

PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

MODEL_PATH=${MODEL_PATH:-weights/pretrain_weights}
SAM3_MODEL=${SAM3_MODEL:-}
DATA_ROOT=${DATA_ROOT:-/root/InstructSAM/video_dataset/revos}
META_PATH=${META_PATH:-${DATA_ROOT}/meta_expressions_train_.json}
MASK_DICT_PATH=${MASK_DICT_PATH:-${DATA_ROOT}/mask_dict.json}
TRAIN_JSON=${TRAIN_JSON:-${REPO_ROOT}/outputs/revos_train/revos_train_sft.json}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/outputs/revos_train/checkpoints}
RUN_NAME=${RUN_NAME:-revos_video_v1}

mkdir -p "$(dirname "${TRAIN_JSON}")" "${OUTPUT_DIR}"

if [ -z "${SAM3_MODEL}" ]; then
  echo "SAM3_MODEL is not set."
  echo "Please point it to a local SAM3 model directory that contains config/processor/weights."
  echo "Example:"
  echo "  SAM3_MODEL=/path/to/local/sam3_dir bash debug/scripts/train_revos_video.sh"
  exit 1
fi

if [ ! -d "${SAM3_MODEL}" ]; then
  echo "SAM3_MODEL must be a local directory, got: ${SAM3_MODEL}"
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

"${TORCHRUN_BIN}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  -m easy_vlm.train \
  --model_path "${MODEL_PATH}" \
  --mask_decoder_model "${SAM3_MODEL}" \
  --seg_encoder sam3 \
  --seg_decoder sam3 \
  --ann_path "${TRAIN_JSON}" \
  --data_root "${DATA_ROOT}" \
  --data_path_root / \
  --data_cache_dir "${OUTPUT_DIR}/hf_cache" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_NAME}" \
  --model_max_length 8192 \
  --mm_max_length 4096 \
  --fps 1 \
  --max_frames 32 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 1 \
  --remove_unused_columns False \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 2 \
  --report_to none \
  --group_by_modality_length True \
  --use_multi_objs False \
  --skip_none False \
  --attn_implementation sdpa \
  --max_seg_nums 10 \
  --bf16 True \
  --fp16 False \
  --tf32 True \
  --dataloader_num_workers 4 \
  --llm_lr 2e-6 \
  --projector_lr 2e-6 \
  --vision_encoder_lr 2e-6 \
  --sam_decoder_lr 5e-6 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine
