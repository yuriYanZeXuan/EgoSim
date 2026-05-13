# EgoHOI: Egocentric World Model for Photorealistic Hand-Object Interaction Synthesis

This repository contains the EgoHOI codebase. It packages the release entrypoints, the `egohoi` modules, and a vendored `diffsynth` dependency required by the current code.

## Repository Layout

```text
egohoi_release/
├── infer.py
├── egohoi/
│   ├── __init__.py
│   ├── camera.py
│   ├── dataset.py
│   ├── inference.py
│   └── model.py
└── diffsynth/
```

- `infer.py`: batch inference over an entire dataset split or dataset root.
- `egohoi/inference.py`: single-clip inference utility and shared inference helpers.
- `egohoi/dataset.py`: dataset loading, pose/object frame lookup, and camera embedding preparation.
- `egohoi/model.py`: inference-time model classes and conditioning modules.
- `diffsynth/`: local dependency used by the EgoHOI pipeline.

## Environment

Recommended:

- Python 3.10+
- CUDA-enabled PyTorch environment
- A GPU with enough memory for Wan I2V inference

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Note: `cupy-cuda12x` is CUDA-version-specific. If your environment uses a different CUDA version, replace it with the matching CuPy package or remove it if your runtime does not need it.

## Required Assets

This repository does not include model weights or datasets. You must provide paths for:

- Wan DiT weights via `--dit_path`
- text encoder weights via `--text_encoder_path`
- VAE weights via `--vae_path`
- image encoder weights via `--image_encoder_path`
- fine-tuned EgoHOI checkpoint via `--checkpoint`

> Checkpoints and base weights: [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-edisondd%2Fegohoi__ckpt-yellow?logo=huggingface)](https://huggingface.co/edisondd/egohoi_ckpt/tree/main)

## Dataset Layout

`infer.py` accepts either a dataset root with split subdirectories or a single split directory directly.

Expected split layout:

```text
SPLIT_ROOT/
├── videos/
├── saved_pose/
├── obj_mask/
└── camera_traj1/
```

If you pass a dataset root, it should look like:

```text
DATA_ROOT/
├── train/
│   ├── videos/
│   ├── saved_pose/
│   ├── obj_mask/
│   └── camera_traj1/
└── val/
    ├── videos/
    ├── saved_pose/
    ├── obj_mask/
    └── camera_traj1/
```

Optional hand-pose override for batch inference:

```text
HAND_POSE_ROOT/
└── <clip_id>/
    ├── 000000.png
    ├── 000001.png
    └── ...
```

## Usage

Batch inference over a split or dataset root:

```bash
python infer.py \
  --dataset_path /path/to/data_root \
  --output_root outputs/batch \
  --dit_path /path/to/dit_01.safetensors,/path/to/dit_02.safetensors \
  --text_encoder_path /path/to/text_encoder.pth \
  --vae_path /path/to/vae.pth \
  --image_encoder_path /path/to/image_encoder.pth \
  --checkpoint /path/to/egohoi_checkpoint.pt \
  --num_frames 81 \
  --height 480 \
  --width 480 \
  --output_fps 24 \
  --torch_dtype bf16
```

Outputs are written under:

```text
<output_root>/<split>/<clip_id>.mp4
```

Single-clip inference:

```bash
python egohoi/inference.py \
  --dataset_path /path/to/data_root \
  --split train \
  --clip_id clip-001947 \
  --output_path outputs/single/clip-001947.mp4 \
  --dit_path /path/to/dit_01.safetensors,/path/to/dit_02.safetensors \
  --text_encoder_path /path/to/text_encoder.pth \
  --vae_path /path/to/vae.pth \
  --image_encoder_path /path/to/image_encoder.pth \
  --checkpoint /path/to/egohoi_checkpoint.pt \
  --num_frames 81 \
  --height 480 \
  --width 480 \
  --output_fps 24 \
  --torch_dtype bf16
```

Useful options:

- `--skip_existing`: skip clips that already have output videos.
- `--splits train val`: restrict batch inference to selected splits.
- `--max_clips_per_split N`: run a smoke test on a subset.
- `--hand_pose_root /path/to/override_pose_root`: override the default pose directory for batch inference.

