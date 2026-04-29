#!/bin/bash
# Copyright (c) jiamingda (https://github.com/Luyitas)
set -euo pipefail
# ============================================================
# Step 00: Convert source video to 16fps 720p 61-frame
# conda env: base (ffmpeg only)
# ============================================================
#
# Usage:
#   source data_process/run_pipeline.env.sh   # set VIDEO_PATH
#   bash data_process/run_step00_convert.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${VIDEO_PATH:?Set VIDEO_PATH in run_pipeline.env.sh and source it first}"

CLIP_ID="$(basename "${VIDEO_PATH%.*}")"
CLIP_DIR="${REPO_ROOT}/tests/samples/${CLIP_ID}"
PROC_DIR="${CLIP_DIR}/_proc"

mkdir -p "${CLIP_DIR}" "${PROC_DIR}"

# Copy source video if not already in place
if [[ "$(realpath "${VIDEO_PATH}")" != "$(realpath "${CLIP_DIR}/video.mp4" 2>/dev/null || echo '')" ]]; then
    cp "${VIDEO_PATH}" "${CLIP_DIR}/video.mp4"
fi

echo "[Step 00] Convert video → 16fps 720p 61-frame"
echo "  Input:  ${VIDEO_PATH}"
echo "  Output: ${CLIP_DIR}/video_16fps.mp4"
echo ""

ffmpeg -y -i "${CLIP_DIR}/video.mp4" \
    -vf "fps=16,scale=-2:720" \
    -frames:v 61 \
    -c:v libx264 -preset fast -crf 18 -an \
    "${CLIP_DIR}/video_16fps.mp4"

echo "[Step 00] Done."
