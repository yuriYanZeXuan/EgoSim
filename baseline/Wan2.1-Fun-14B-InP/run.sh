#!/bin/bash
# Generic Wan2.1-Fun-14B-InP baseline inference on demo_data.
# Usage: bash baseline/run.sh

set -e

MODEL_ROOT="${MODEL_ROOT:-./Wan2.1-Fun-14B-InP}"
GPU_ID="${GPU_ID:-0}"

# ---------- Egodex ----------
echo "[baseline] Running egodex ..."
PYTHONPATH=. python baseline/Wan2.1-Fun-14B-InP/inference.py \
  --dataset egodex \
  --dataset_root demo_data/egodex \
  --metadata_path demo_data/egodex_metadata.csv \
  --output_dir output_baseline/egodex \
  --model_root "$MODEL_ROOT" \
  --gpu_id "$GPU_ID" \
  --num_inference_steps 50 \
  --height 480 \
  --width 832 \
  --num_frames 81

# ---------- EgoVid ----------
echo "[baseline] Running egovid ..."
PYTHONPATH=. python baseline/Wan2.1-Fun-14B-InP/inference.py \
  --dataset egovid \
  --dataset_root demo_data/egovid \
  --metadata_path demo_data/egovid_metadata.csv \
  --output_dir output_baseline/egovid \
  --model_root "$MODEL_ROOT" \
  --gpu_id "$GPU_ID" \
  --num_inference_steps 50 \
  --height 480 \
  --width 832 \
  --num_frames 81

# ---------- Continuous Generation ----------
echo "[baseline] Running continuous_generation ..."
PYTHONPATH=. python baseline/Wan2.1-Fun-14B-InP/inference.py \
  --dataset continuous_generation \
  --dataset_root demo_data/continuous_generation \
  --metadata_path demo_data/continuous_generation/metadata.csv \
  --output_dir output_baseline/continuous_generation \
  --model_root "$MODEL_ROOT" \
  --gpu_id "$GPU_ID" \
  --num_inference_steps 50 \
  --height 480 \
  --width 832 \
  --num_frames 81

echo "[baseline] All done. Outputs under output_baseline/"
