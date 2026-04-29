#!/bin/bash
# Copyright (c) jiamingda (https://github.com/Luyitas)
set -euo pipefail
# ============================================================
# Step 04: Generate metadata CSV for inference
# conda env: caption (or any env with Python + standard library)
# ============================================================
#
# Usage:
#   source data_process/run_pipeline.env.sh
#   bash data_process/03_caption/run_step04_metadata.sh
#
# Output: tests/samples/<clip_name>_metadata.csv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${VIDEO_PATH:?Set VIDEO_PATH in run_pipeline.env.sh and source it first}"

CLIP_ID="$(basename "${VIDEO_PATH%.*}")"
CLIP_DIR="${REPO_ROOT}/tests/samples/${CLIP_ID}"
OUTPUT_CSV="${REPO_ROOT}/tests/samples/${CLIP_ID}_metadata.csv"

echo "[Step 04] Generate Metadata CSV"
echo "  Clip:   ${CLIP_DIR}"
echo "  Output: ${OUTPUT_CSV}"
echo ""

python "${SCRIPT_DIR}/generate_metadata.py" \
    --clip_dir "${CLIP_DIR}" \
    --output "${OUTPUT_CSV}"

echo "[Step 04] Done."
echo ""
echo "Run inference with:"
echo "  PYTHONPATH=. python egowm/inference/runner.py \\"
echo "    --dataset egovid \\"
echo "    --model_root ../EgoSim-14B \\"
echo "    --dataset_root tests/samples \\"
echo "    --metadata_path ${OUTPUT_CSV} \\"
echo "    --output_dir output_${CLIP_ID} \\"
echo "    --gpu_id 0"
