#!/bin/bash
# Copyright (c) jiamingda (https://github.com/Luyitas)
set -euo pipefail
# ============================================================
# Step 01b: Kalman smoothing of camera parameters
# conda env: da3 (numpy + scipy only)
# ============================================================
#
# Usage:
#   source data_process/run_pipeline.env.sh
#   conda activate da3
#   bash data_process/01_depth_pose_da3/run_step01b_smooth.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${VIDEO_PATH:?Set VIDEO_PATH in run_pipeline.env.sh and source it first}"
NUM_WORKERS="${NUM_WORKERS:-8}"

CLIP_ID="$(basename "${VIDEO_PATH%.*}")"
CLIP_DIR="${REPO_ROOT}/tests/samples/${CLIP_ID}"
PROC_DIR="${CLIP_DIR}/_proc"

echo "[Step 01b] Camera Kalman Smoothing"
echo "  Input:  ${PROC_DIR}/poses_da3"
echo "  Output: ${PROC_DIR}/poses_da3_smoothed"
echo ""

python "${SCRIPT_DIR}/smooth_camera_kalman_egovid.py" \
    --input_dir "${PROC_DIR}/poses_da3" \
    --output_dir "${PROC_DIR}/poses_da3_smoothed" \
    --fps 30 \
    --num_workers "${NUM_WORKERS}" \
    --start_idx 0 \
    --end_idx 1

echo "[Step 01b] Done."
