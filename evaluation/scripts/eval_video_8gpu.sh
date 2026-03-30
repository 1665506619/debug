#!/bin/bash
set -euo pipefail

export PYTHONWARNINGS="ignore"

if [ "$#" -lt 6 ]; then
    echo "Usage: $0 MODEL_PATH MODEL_BASE SAM3_CKPT VIDEO_FOLDER QUESTION_FILE OUTPUT_FILE [extra eval_video args...]"
    exit 1
fi

MODEL_PATH=$1
MODEL_BASE=$2
SAM3_CKPT=$3
VIDEO_FOLDER=$4
QUESTION_FILE=$5
OUTPUT_FILE=$6
shift 6

NUM_CHUNKS=${NUM_CHUNKS:-8}
GPU_IDS_STR=${GPU_IDS:-"0,1,2,3,4,5,6,7"}
DEFAULT_NUM_WORKERS=${NUM_WORKERS:-4}
DEFAULT_VIDEO_LOADER_TYPE=${VIDEO_LOADER_TYPE:-torchcodec}
DEFAULT_ASYNC_LOADING_FRAMES=${ASYNC_LOADING_FRAMES:-true}
IFS=',' read -r -a GPU_IDS_ARR <<< "$GPU_IDS_STR"

if [ "${#GPU_IDS_ARR[@]}" -lt "$NUM_CHUNKS" ]; then
    echo "Need at least $NUM_CHUNKS GPU ids, got ${#GPU_IDS_ARR[@]} from GPU_IDS=$GPU_IDS_STR"
    exit 1
fi

OUTPUT_STEM=${OUTPUT_FILE%.*}
OUTPUT_EXT=${OUTPUT_FILE##*.}
CHUNK_DIR="${OUTPUT_STEM}_chunks"
LOG_DIR="${CHUNK_DIR}/logs"
mkdir -p "$CHUNK_DIR" "$LOG_DIR"

PIDS=()
STATUS=0

for ((CHUNK_IDX=0; CHUNK_IDX<NUM_CHUNKS; CHUNK_IDX++)); do
    GPU_ID=${GPU_IDS_ARR[$CHUNK_IDX]}
    CHUNK_OUTPUT="${CHUNK_DIR}/chunk_${CHUNK_IDX}.${OUTPUT_EXT}"
    LOG_PATH="${LOG_DIR}/chunk_${CHUNK_IDX}.log"

    CMD=(
        python debug/evaluation/eval_video.py
        --model_path "${MODEL_PATH}"
        --sam3_video_checkpoint "${SAM3_CKPT}"
        --video_folder "${VIDEO_FOLDER}"
        --question_file "${QUESTION_FILE}"
        --output_file "${CHUNK_OUTPUT}"
        --num-chunks "${NUM_CHUNKS}"
        --chunk-idx "${CHUNK_IDX}"
        --num-workers "${DEFAULT_NUM_WORKERS}"
        --video-loader-type "${DEFAULT_VIDEO_LOADER_TYPE}"
    )

    if [[ "${MODEL_BASE}" != "-" && "${MODEL_BASE}" != "None" && "${MODEL_BASE}" != "none" ]]; then
        CMD+=(--model_base "${MODEL_BASE}")
    fi

    if [[ "${DEFAULT_ASYNC_LOADING_FRAMES}" == "true" ]]; then
        CMD+=(--async-loading-frames)
    fi

    echo "Launching chunk ${CHUNK_IDX}/${NUM_CHUNKS} on GPU ${GPU_ID} -> ${CHUNK_OUTPUT}"
    CUDA_VISIBLE_DEVICES=${GPU_ID} PYTHONPATH=debug "${CMD[@]}" "$@" > "${LOG_PATH}" 2>&1 &

    PIDS+=($!)
done

for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        STATUS=1
    fi
done

if [ "$STATUS" -ne 0 ]; then
    echo "At least one chunk failed. Check logs under ${LOG_DIR}"
    exit "$STATUS"
fi

PYTHONPATH=debug python debug/evaluation/merge_video_chunk_results.py \
    --chunk-dir "${CHUNK_DIR}" \
    --pattern "chunk_*.${OUTPUT_EXT}" \
    --output-file "${OUTPUT_FILE}"

echo "Merged result saved to ${OUTPUT_FILE}"
