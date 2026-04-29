#!/bin/bash
# Copyright (c) jiamingda (https://github.com/Luyitas)
set -euo pipefail
# ============================================================
# Step 03: Qwen2.5-VL video captioning
# conda env: caption
# ============================================================
#
# Usage:
#   source data_process/run_pipeline.env.sh
#   conda activate caption
#   bash data_process/03_caption/run_step03_caption.sh
#
# Optional flags (passed through to caption_video.py):
#   SKIP_EXISTING=1  bash data_process/03_caption/run_step03_caption.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${VIDEO_PATH:?Set VIDEO_PATH in run_pipeline.env.sh and source it first}"
: "${CAPTION_MODEL:?Set CAPTION_MODEL in run_pipeline.env.sh and source it first}"

CLIP_ID="$(basename "${VIDEO_PATH%.*}")"
CLIP_DIR="${REPO_ROOT}/tests/samples/${CLIP_ID}"

EXTRA_ARGS=""
if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then
    EXTRA_ARGS="--skip_existing"
fi

echo "[Step 03] Qwen2.5-VL Video Captioning"
echo "  Clip:   ${CLIP_DIR}/video.mp4"
echo "  Model:  ${CAPTION_MODEL}"
echo "  Output: ${CLIP_DIR}/caption.txt"
echo ""

python "${SCRIPT_DIR}/caption_video.py" \
    --clip_dir "${CLIP_DIR}" \
    --model_path "${CAPTION_MODEL}" \
    ${EXTRA_ARGS}

echo "[Step 03] Done."
