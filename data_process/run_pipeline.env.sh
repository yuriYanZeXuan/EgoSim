#!/bin/bash
# Copyright (c) jiamingda (https://github.com/Luyitas)
# ============================================================
# Pipeline configuration — edit this file once, then run each
# step with:  bash data_process/run_stepXX_*.sh
# ============================================================

# ── Input video ─────────────────────────────────────────────
export VIDEO_PATH="/path/to/your/video.mp4"

# ── Repo roots ──────────────────────────────────────────────
export DA3_ROOT="/path/to/Depth-Anything-3"
export HAMER_ROOT="/path/to/hamer"
export SAM3_ROOT="/path/to/sam3"

# ── Model checkpoints ───────────────────────────────────────
export DA3_MODEL="${DA3_ROOT}/checkpoints/DA3NESTED-GIANT-LARGE-1.1"
export SAM3_CHECKPOINT="${SAM3_ROOT}/checkpoints/sam3.pt"
export INPAINT_MODEL="/path/to/models/Qwen-Image-Edit-2511"
export VITDET_INIT_CHECKPOINT="${HAMER_ROOT}/_DATA/model_final_f05665.pkl"
export MANO_PATH="${HAMER_ROOT}/_DATA/data/mano"
export CAPTION_MODEL="/path/to/models/Qwen2.5-VL-32B-Instruct"

# ── Optional overrides ──────────────────────────────────────
# export DEVICE=0            # GPU index (default: 0)
# export NUM_WORKERS=8       # parallel workers (default: 8)
# export PYOPENGL_PLATFORM=egl   # or osmesa if no display
