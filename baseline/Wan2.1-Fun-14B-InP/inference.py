#!/usr/bin/env python
"""
Generic Wan2.1-Fun-14B-InP baseline inference.

Uses first-frame image + prompt to generate video via Wan2.1 I2V pipeline.
Supports all demo_data formats: egodex, egovid, continuous_generation.

Example:
    PYTHONPATH=. python baseline/Wan2.1-Fun-14B-InP/inference.py \
        --dataset egodex \
        --dataset_root demo_data/egodex \
        --metadata_path demo_data/egodex_metadata.csv \
        --output_dir output_baseline/egodex \
        --model_root ./Wan2.1-Fun-14B-InP
"""
import argparse
import importlib
import os
import sys
from pathlib import Path

import imageio
import torch
from PIL import Image
from tqdm import tqdm

# Keep the project root importable for baseline.data, but do not let the local
# EgoSim/diffsynth shadow an installed diffsynth package for this baseline.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from baseline.data import egodex, egovid, continuous
import torch.distributed as dist

def _path_is_project_root(path: str) -> bool:
    try:
        return Path(path or os.getcwd()).resolve() == _PROJECT_ROOT
    except OSError:
        return False


def _import_external_wan_pipeline():
    original_sys_path = list(sys.path)
    try:
        sys.path = [path for path in sys.path if not _path_is_project_root(path)]
        for module_name in list(sys.modules):
            if module_name == "diffsynth" or module_name.startswith("diffsynth."):
                module_file = getattr(sys.modules[module_name], "__file__", "")
                if module_file and Path(module_file).resolve().is_relative_to(_PROJECT_ROOT):
                    del sys.modules[module_name]
        importlib.invalidate_caches()
        wan_video = importlib.import_module("diffsynth.pipelines.wan_video")
        return wan_video.WanVideoPipeline, wan_video.ModelConfig
    finally:
        sys.path = original_sys_path


WanVideoPipeline, ModelConfig = _import_external_wan_pipeline()


def _resolve_dataset_loader(dataset: str):
    loaders = {
        "egodex": egodex,
        "egovid": egovid,
        "continuous_generation": continuous,
    }
    if dataset not in loaders:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {list(loaders.keys())}.")
    return loaders[dataset]


def load_pipeline(model_root: str, device: str) -> WanVideoPipeline:
    """Load Wan2.1-Fun-14B-InP pipeline from local model directory."""
    model_root = Path(model_root)
    diffsynth_module = importlib.import_module("diffsynth")
    print(f"[INFO] Using diffsynth from {getattr(diffsynth_module, '__file__', 'unknown')}")
    required = [
        "diffusion_pytorch_model.safetensors",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    ]
    for f in required:
        p = model_root / f
        if not p.exists():
            raise FileNotFoundError(f"Required model file not found: {p}")

    tokenizer_path = model_root / "google" / "umt5-xxl"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    model_configs = [
        ModelConfig(path=str(model_root / "diffusion_pytorch_model.safetensors")),
        ModelConfig(path=str(model_root / "models_t5_umt5-xxl-enc-bf16.pth")),
        ModelConfig(path=str(model_root / "Wan2.1_VAE.pth")),
        ModelConfig(path=str(model_root / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")),
    ]

    print(f"[INFO] Loading Wan2.1-Fun-14B-InP from {model_root} on {device} ...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        use_usp=True,
        model_configs=model_configs,
        tokenizer_config=ModelConfig(path=str(tokenizer_path)),
    )
    print("[INFO] Pipeline ready.")
    return pipe


def run_inference(args: argparse.Namespace) -> None:
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    pipe = load_pipeline(args.model_root, device)

    loader = _resolve_dataset_loader(args.dataset)
    print(f"[INFO] Loading {args.dataset} samples from {args.metadata_path} ...")

    if args.dataset == "egovid" and getattr(args, "eval_set_path", None):
        samples = loader.load_samples(args.metadata_path, args.eval_set_path)
    else:
        samples = loader.load_samples(args.metadata_path)

    if not samples:
        print("[WARN] No samples loaded.")
        return

    print(f"[INFO] {len(samples)} samples to process.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    skipped = 0
    failed = 0

    for sample in tqdm(samples, desc=f"[{args.dataset}]"):
        out_path = output_dir / f"{sample.output_id}.mp4"
        if out_path.exists() and args.skip_existing:
            skipped += 1
            continue

        first_frame_path = loader.get_first_frame_path(args.dataset_root, sample)
        if not first_frame_path.exists():
            print(f"  [WARN] Missing first frame: {first_frame_path}")
            failed += 1
            continue

        first_frame = Image.open(first_frame_path).convert("RGB")

        video = pipe(
            prompt=sample.prompt,
            input_image=first_frame,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
            tiled=args.tiled,
        )
        if dist.get_rank() == 0:
            imageio.mimwrite(str(out_path), video, fps=args.fps, quality=8)
        success += 1

    print(
        f"[INFO] Done. success={success}, skipped={skipped}, failed={failed}, "
        f"total={len(samples)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wan2.1-Fun-14B-InP zero-shot baseline inference (first-frame + prompt)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["egodex", "egovid", "continuous_generation"],
        help="Dataset type to infer on.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Root directory containing the dataset assets.",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        required=True,
        help="Path to the dataset metadata CSV.",
    )
    parser.add_argument(
        "--eval_set_path",
        type=str,
        default=None,
        help="Optional eval_set.txt for egovid.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write generated videos.",
    )
    parser.add_argument(
        "--model_root",
        type=str,
        required=True,
        help="Local directory containing Wan2.1-Fun-14B-InP weights.",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="CUDA device id.")
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Denoising steps."
    )
    parser.add_argument("--height", type=int, default=480, help="Video height.")
    parser.add_argument("--width", type=int, default=832, help="Video width.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames.")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="CFG scale.")
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--tiled", action="store_true", default=True, help="VAE tiling.")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Negative prompt for CFG.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip samples that already have output videos.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
