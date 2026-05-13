#!/bin/bash
# EgoHOI baseline on EgoSim demo metadata.

set -e

MODEL_ROOT="${MODEL_ROOT:-./Wan2.1-Fun-14B-InP}"
CHECKPOINT="${CHECKPOINT:-${EGOHOI_CHECKPOINT:-}}"
DATA_ROOT="${DATA_ROOT:-tests/samples/demo_data}"
GPU_ID="${GPU_ID:-0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-480}"
NUM_FRAMES="${NUM_FRAMES:-81}"
FPS="${FPS:-24}"

if [ -z "$CHECKPOINT" ]; then
  echo "[EgoHOI-baseline] ERROR: set CHECKPOINT or EGOHOI_CHECKPOINT to the EgoHOI finetuned checkpoint."
  exit 1
fi

COMMON_ARGS=(
  --model_root "$MODEL_ROOT"
  --checkpoint "$CHECKPOINT"
  --gpu_id "$GPU_ID"
  --num_inference_steps "$NUM_INFERENCE_STEPS"
  --height "$HEIGHT"
  --width "$WIDTH"
  --num_frames "$NUM_FRAMES"
  --fps "$FPS"
)

echo "[EgoHOI-baseline] Running egodex ..."
PYTHONPATH=. python baseline/EgoHOI/inference.py \
  --dataset egodex \
  --dataset_root "$DATA_ROOT/egodex" \
  --metadata_path "$DATA_ROOT/egodex_metadata.csv" \
  --output_dir baseline/EgoHOI/outputs/egodex \
  "${COMMON_ARGS[@]}"

echo "[EgoHOI-baseline] Running egovid ..."
PYTHONPATH=. python baseline/EgoHOI/inference.py \
  --dataset egovid \
  --dataset_root "$DATA_ROOT/egovid" \
  --metadata_path "$DATA_ROOT/egovid_metadata.csv" \
  --output_dir baseline/EgoHOI/outputs/egovid \
  "${COMMON_ARGS[@]}"

echo "[EgoHOI-baseline] All done. Outputs under baseline/EgoHOI/outputs/"
