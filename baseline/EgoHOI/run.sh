#!/bin/bash
# EgoHOI baseline on EgoSim demo metadata.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_ROOT="${MODEL_ROOT:-./Wan2.1-Fun-14B-InP}"
CHECKPOINT="${CHECKPOINT:-${EGOHOI_CHECKPOINT:-}}"
DATA_ROOT="${DATA_ROOT:-tests/samples/demo_data}"
GPU_ID="${GPU_ID:-0}"
USE_USP="${USE_USP:-1}"
USP_GPUS="${USP_GPUS:-4}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-480}"
NUM_FRAMES="${NUM_FRAMES:-81}"
FPS="${FPS:-24}"

if [ -z "$CHECKPOINT" ]; then
  echo "[EgoHOI-baseline] ERROR: set CHECKPOINT or EGOHOI_CHECKPOINT to the EgoHOI finetuned checkpoint."
  return 1 2>/dev/null || exit 1
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

run_egohoi() {
  if [ "$USE_USP" = "1" ]; then
    PYTHONPATH=. CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" torchrun --standalone --nproc_per_node "$USP_GPUS" \
      baseline/EgoHOI/inference.py --use_usp "$@"
  else
    PYTHONPATH=. python baseline/EgoHOI/inference.py "$@"
  fi
}

echo "[EgoHOI-baseline] Running egodex ..."
run_egohoi \
  --dataset egodex \
  --dataset_root "$DATA_ROOT/egodex" \
  --metadata_path "$DATA_ROOT/egodex_metadata.csv" \
  --output_dir baseline/EgoHOI/outputs/egodex \
  "${COMMON_ARGS[@]}"

echo "[EgoHOI-baseline] Running egovid ..."
run_egohoi \
  --dataset egovid \
  --dataset_root "$DATA_ROOT/egovid" \
  --metadata_path "$DATA_ROOT/egovid_metadata.csv" \
  --output_dir baseline/EgoHOI/outputs/egovid \
  "${COMMON_ARGS[@]}"

echo "[EgoHOI-baseline] All done. Outputs under baseline/EgoHOI/outputs/"
