#!/bin/bash
# EgoHOI original-format inference on examples/val.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_ROOT="${MODEL_ROOT:-/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/wan2.1_inp}"
CHECKPOINT="${CHECKPOINT:-/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/EgoHOI/mp_rank_00_model_states.pt}"
DATASET_PATH="${DATASET_PATH:-$SCRIPT_DIR/examples}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs/original}"
SPLIT="${SPLIT:-val}"
GPU_ID="${GPU_ID:-0}"
USE_USP="${USE_USP:-1}"
USP_GPUS="${USP_GPUS:-8}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-480}"
NUM_FRAMES="${NUM_FRAMES:-81}"
OUTPUT_FPS="${OUTPUT_FPS:-24}"
TORCH_DTYPE="${TORCH_DTYPE:-bf16}"

DIT_PATH="${DIT_PATH:-$MODEL_ROOT/diffusion_pytorch_model.safetensors}"
TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:-$MODEL_ROOT/models_t5_umt5-xxl-enc-bf16.pth}"
VAE_PATH="${VAE_PATH:-$MODEL_ROOT/Wan2.1_VAE.pth}"
IMAGE_ENCODER_PATH="${IMAGE_ENCODER_PATH:-$MODEL_ROOT/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth}"

if [ -z "$CHECKPOINT" ]; then
  echo "[EgoHOI-original] ERROR: set CHECKPOINT to the EgoHOI finetuned checkpoint."
  return 1 2>/dev/null || exit 1
fi

if [ ! -d "$DATASET_PATH/$SPLIT" ]; then
  echo "[EgoHOI-original] ERROR: expected split directory at $DATASET_PATH/$SPLIT"
  return 1 2>/dev/null || exit 1
fi

echo "[EgoHOI-original] Running split '$SPLIT' from $DATASET_PATH ..."
COMMON_ARGS=(
  --dataset_path "$DATASET_PATH"
  --splits "$SPLIT"
  --output_root "$OUTPUT_ROOT"
  --dit_path "$DIT_PATH"
  --text_encoder_path "$TEXT_ENCODER_PATH"
  --vae_path "$VAE_PATH"
  --image_encoder_path "$IMAGE_ENCODER_PATH"
  --checkpoint "$CHECKPOINT"
  --device "cuda:$GPU_ID"
  --num_inference_steps "$NUM_INFERENCE_STEPS"
  --num_frames "$NUM_FRAMES"
  --height "$HEIGHT"
  --width "$WIDTH"
  --output_fps "$OUTPUT_FPS"
  --torch_dtype "$TORCH_DTYPE"
  --disable_multiprocessing
)

if [ "$USE_USP" = "1" ]; then
  PYTHONPATH="$SCRIPT_DIR:$PROJECT_ROOT" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
    torchrun --standalone --nproc_per_node "$USP_GPUS" \
    baseline/EgoHOI/infer.py --use_usp "${COMMON_ARGS[@]}"
else
  PYTHONPATH="$SCRIPT_DIR:$PROJECT_ROOT" python baseline/EgoHOI/infer.py "${COMMON_ARGS[@]}"
fi

echo "[EgoHOI-original] All done. Outputs under $OUTPUT_ROOT/$SPLIT"
