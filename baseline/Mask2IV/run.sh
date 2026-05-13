#!/bin/bash
# Mask2IV Wan2.1 zero-shot baseline on demo_data.
# Produces first-frame + prompt videos for direct comparison with Mask2IV.

set -e

MODEL_ROOT="${MODEL_ROOT:-./Wan2.1-Fun-14B-InP}"
GPU_ID="${GPU_ID:-0}"

echo "[Mask2IV-baseline] Running egodex ..."
PYTHONPATH=. python baseline/Wan2.1-Fun-14B-InP/inference.py \
  --dataset egodex \
  --dataset_root demo_data/egodex \
  --metadata_path demo_data/egodex_metadata.csv \
  --output_dir baseline/Mask2IV/outputs/egodex \
  --model_root "$MODEL_ROOT" \
  --gpu_id "$GPU_ID" \
  --num_inference_steps 50 \
  --height 480 \
  --width 832 \
  --num_frames 81

echo "[Mask2IV-baseline] Running egovid ..."
PYTHONPATH=. python baseline/Wan2.1-Fun-14B-InP/inference.py \
  --dataset egovid \
  --dataset_root demo_data/egovid \
  --metadata_path demo_data/egovid_metadata.csv \
  --output_dir baseline/Mask2IV/outputs/egovid \
  --model_root "$MODEL_ROOT" \
  --gpu_id "$GPU_ID" \
  --num_inference_steps 50 \
  --height 480 \
  --width 832 \
  --num_frames 81

echo "[Mask2IV-baseline] Running continuous_generation ..."
PYTHONPATH=. python baseline/Wan2.1-Fun-14B-InP/inference.py \
  --dataset continuous_generation \
  --dataset_root demo_data/continuous_generation \
  --metadata_path demo_data/continuous_generation/metadata.csv \
  --output_dir baseline/Mask2IV/outputs/continuous_generation \
  --model_root "$MODEL_ROOT" \
  --gpu_id "$GPU_ID" \
  --num_inference_steps 50 \
  --height 480 \
  --width 832 \
  --num_frames 81

echo "[Mask2IV-baseline] All done. Outputs under baseline/Mask2IV/outputs/"
