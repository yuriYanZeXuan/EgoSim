#!/usr/bin/env python
"""Run EgoHOI on EgoSim metadata samples.

This entrypoint keeps the common EgoSim CLI contract
(`--dataset`, `--dataset_root`, `--metadata_path`, `--output_dir`) while using
the EgoHOI inference modules under `baseline/EgoHOI`.
"""
import argparse
import os
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EGOHOI_ROOT = Path(__file__).resolve().parent
if str(_EGOHOI_ROOT) not in sys.path:
    sys.path.insert(0, str(_EGOHOI_ROOT))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from diffsynth import save_video  # noqa: E402
from baseline.data import agibot, continuous, egodex, egovid  # noqa: E402
from egohoi.inference import (  # noqa: E402
    debug_tensor_stats,
    prepare_camera_tokens,
    prepare_object_tokens,
    prepare_pose_condition,
    prepare_random_ref_pose,
    set_seed,
)
from infer import initialize_models, prepare_device, resolve_torch_dtype  # noqa: E402


def _dist_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _is_main_process() -> bool:
    return not _dist_is_initialized() or torch.distributed.get_rank() == 0


def _initialize_usp(args: argparse.Namespace) -> None:
    if not args.use_usp:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("--use_usp requires CUDA.")
    if not _dist_is_initialized():
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment

        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        init_distributed_environment(
            rank=torch.distributed.get_rank(),
            world_size=torch.distributed.get_world_size(),
        )
        initialize_model_parallel(
            sequence_parallel_degree=torch.distributed.get_world_size(),
            ring_degree=1,
            ulysses_degree=torch.distributed.get_world_size(),
        )
    local_rank = int(os.environ.get("LOCAL_RANK", args.gpu_id))
    torch.cuda.set_device(local_rank)
    args.gpu_id = local_rank
    args.device = f"cuda:{local_rank}"


def _resolve_dataset_loader(dataset: str):
    loaders = {
        "egodex": egodex,
        "egovid": egovid,
        "agibot": agibot,
        "continuous_generation": continuous,
    }
    if dataset not in loaders:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {list(loaders.keys())}.")
    return loaders[dataset]


def _load_samples(args: argparse.Namespace):
    loader = _resolve_dataset_loader(args.dataset)
    if args.dataset == "egovid" and args.eval_set_path:
        return loader.load_samples(args.metadata_path, args.eval_set_path)
    if args.dataset == "continuous_generation":
        return loader.load_samples(args.metadata_path, args.eval_set_path)
    return loader.load_samples(args.metadata_path)


def _get_path(loader, name: str, dataset_root: str, sample) -> Path:
    return getattr(loader, name)(dataset_root, sample)


def _resize_image(image: Image.Image, height: int, width: int, interpolation) -> Image.Image:
    resample = {
        InterpolationMode.NEAREST: Image.Resampling.NEAREST,
        InterpolationMode.BILINEAR: Image.Resampling.BILINEAR,
        InterpolationMode.BICUBIC: Image.Resampling.BICUBIC,
    }.get(interpolation, Image.Resampling.BILINEAR)
    return image.resize((width, height), resample=resample)


def _select_indices(total: int, start: int, num_frames: int, interval: int) -> list[int]:
    if total <= 0:
        raise ValueError("Video contains no frames.")
    return [max(min(start + idx * interval, total - 1), 0) for idx in range(num_frames)]


def _video_num_frames(path: Path) -> int:
    reader = imageio.get_reader(str(path))
    try:
        try:
            count = reader.count_frames()
            if count and count > 0:
                return int(count)
        except Exception:
            pass
        return sum(1 for _ in reader)
    finally:
        reader.close()


def _read_video_frames(path: Path, indices: list[int], height: int, width: int, interpolation) -> list[Image.Image]:
    reader = imageio.get_reader(str(path))
    frames = []
    try:
        for idx in indices:
            frame = Image.fromarray(reader.get_data(idx)).convert("RGB")
            frames.append(_resize_image(frame, height, width, interpolation))
    finally:
        reader.close()
    return frames


def _frames_to_chw_video(frames: list[Image.Image]) -> torch.Tensor:
    tensors = [torch.from_numpy(np.array(frame, dtype=np.uint8)) for frame in frames]
    return torch.stack(tensors, dim=0).permute(3, 0, 1, 2).contiguous()


def _first_rgb_frame(path: Path, height: int, width: int, interpolation) -> torch.Tensor:
    frame = _read_video_frames(path, [0], height, width, interpolation)[0]
    return torch.from_numpy(np.array(frame, dtype=np.uint8))


def _load_first_frame(path: Path, height: int, width: int) -> Image.Image:
    if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        return _read_video_frames(path, [0], height, width, InterpolationMode.BILINEAR)[0]
    with Image.open(path) as image:
        return _resize_image(image.convert("RGB"), height, width, InterpolationMode.BILINEAR)


def _load_baseline_sample(loader, args: argparse.Namespace, sample) -> dict:
    hand_path = _get_path(loader, "get_hand_video_path", args.dataset_root, sample)
    first_frame_path = _get_path(loader, "get_first_frame_path", args.dataset_root, sample)
    mask_path = _get_path(loader, "get_mask_path", args.dataset_root, sample)

    total_frames = _video_num_frames(hand_path)
    frame_indices = _select_indices(total_frames, args.start_frame, args.num_frames, args.frame_interval)
    hand_frames = _read_video_frames(
        hand_path,
        frame_indices,
        args.height,
        args.width,
        InterpolationMode.BILINEAR,
    )

    if mask_path.exists():
        object_ref = _first_rgb_frame(mask_path, args.height, args.width, InterpolationMode.NEAREST)
    else:
        object_ref = torch.zeros(args.height, args.width, 3, dtype=torch.uint8)

    return {
        "frame_indices": frame_indices,
        "first_frame": _load_first_frame(first_frame_path, args.height, args.width),
        "dwpose": _frames_to_chw_video(hand_frames),
        "random_ref_dwpose": torch.from_numpy(np.array(hand_frames[0], dtype=np.uint8)),
        "random_ref_object": object_ref,
        # EgoSim metadata does not carry EgoHOI's camera_traj1 JSON; use neutral camera tokens.
        "camera_embedding": torch.zeros(6, args.num_frames, args.height, args.width, dtype=torch.float32),
    }


@torch.no_grad()
def run_sample_inference(
    args: argparse.Namespace,
    model_bundle: tuple,
    sample_data: dict,
    prompt: str,
    clip_seed: int,
    output_path: Path,
) -> None:
    train_model, pipe, pipe_vae, tiler_kwargs, device, torch_dtype = model_bundle
    pipe.scheduler.set_timesteps(args.num_inference_steps, shift=args.sigma_shift)

    height, width = args.height, args.width
    latent_shape = (1, 16, (args.num_frames - 1) // 4 + 1, height // 8, width // 8)
    latents = pipe.generate_noise(latent_shape, seed=clip_seed, device=device, dtype=torch.float32)
    latents = latents.to(device=device, dtype=torch_dtype)

    if device.type == "cuda" and pipe.text_encoder is not None:
        pipe.text_encoder.to(device=device, dtype=torch_dtype).eval()
    prompt_emb_pos = pipe_vae.encode_prompt(prompt or args.prompt, positive=True)
    prompt_emb_pos["context"] = prompt_emb_pos["context"].to(device=device)
    if args.cfg_scale != 1.0:
        prompt_emb_neg = pipe_vae.encode_prompt(args.negative_prompt, positive=False)
        prompt_emb_neg["context"] = prompt_emb_neg["context"].to(device=device)
    else:
        prompt_emb_neg = None
    if device.type == "cuda" and pipe.text_encoder is not None and not args.keep_modules_on_device:
        pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()

    if device.type == "cuda":
        if pipe.image_encoder is not None:
            pipe.image_encoder.to(device=device, dtype=torch.float32).eval()
        if pipe.vae is not None:
            pipe.vae.to(device=device, dtype=torch_dtype).eval()
    image_emb_pos = pipe_vae.encode_image(sample_data["first_frame"], args.num_frames, height, width)
    for key, value in image_emb_pos.items():
        image_emb_pos[key] = value.to(device=device)

    condition_tokens = prepare_pose_condition(train_model, sample_data["dwpose"].unsqueeze(0), device, torch_dtype)
    random_ref_pose_emb = prepare_random_ref_pose(
        train_model,
        sample_data["random_ref_dwpose"].unsqueeze(0),
        device,
        torch_dtype,
    )
    if "y" in image_emb_pos:
        image_emb_pos["y"] = image_emb_pos["y"] + random_ref_pose_emb.to(dtype=image_emb_pos["y"].dtype)

    image_emb_neg = {}
    if args.cfg_scale != 1.0:
        if "clip_feature" in image_emb_pos:
            image_emb_neg["clip_feature"] = image_emb_pos["clip_feature"]
        if "y" in image_emb_pos:
            image_emb_neg["y"] = torch.zeros_like(image_emb_pos["y"])

    object_tokens, object_grid = prepare_object_tokens(
        train_model,
        sample_data["random_ref_object"].unsqueeze(0),
        pipe_vae,
        device,
        tiler_kwargs,
        torch_dtype,
    )
    camera_tokens = prepare_camera_tokens(
        train_model,
        sample_data["camera_embedding"].unsqueeze(0),
        target_shape=latents.shape[2:],
        device=device,
        target_dtype=torch_dtype,
    )

    if args.debug_conditions:
        debug_tensor_stats("condition_tokens", condition_tokens)
        debug_tensor_stats("object_tokens", object_tokens)
        debug_tensor_stats("camera_tokens", camera_tokens)

    if device.type == "cuda" and not args.keep_modules_on_device:
        if pipe.image_encoder is not None:
            pipe.image_encoder.to("cpu")
        if pipe.vae is not None:
            pipe.vae.to("cpu")
        torch.cuda.empty_cache()

    condition_tokens_uncond = torch.zeros_like(condition_tokens)
    camera_tokens_uncond = torch.zeros_like(camera_tokens)
    timesteps = pipe.scheduler.timesteps
    iterator = (
        tqdm(range(len(timesteps)), desc="Denoising", leave=False, dynamic_ncols=True)
        if args.denoise_progress
        else range(len(timesteps))
    )
    for progress_id in iterator:
        timestep = timesteps[progress_id]
        timestep_tensor = timestep.unsqueeze(0).to(device=device, dtype=torch_dtype)
        noise_pred_pos = train_model.forward_dit_with_conditions(
            noisy_latents=latents,
            timestep=timestep_tensor,
            prompt_emb=prompt_emb_pos,
            image_emb=image_emb_pos,
            condition_tokens=condition_tokens,
            object_tokens=object_tokens,
            object_grid=object_grid,
            camera_tokens=camera_tokens,
        )
        if args.cfg_scale != 1.0 and prompt_emb_neg is not None:
            noise_pred_neg = train_model.forward_dit_with_conditions(
                noisy_latents=latents,
                timestep=timestep_tensor,
                prompt_emb=prompt_emb_neg,
                image_emb=image_emb_neg,
                condition_tokens=condition_tokens_uncond,
                object_tokens=None,
                object_grid=None,
                camera_tokens=camera_tokens_uncond,
            )
            noise_pred = noise_pred_neg + args.cfg_scale * (noise_pred_pos - noise_pred_neg)
        else:
            noise_pred = noise_pred_pos
        latents = pipe.scheduler.step(noise_pred, pipe.scheduler.timesteps[progress_id], latents)

    if device.type == "cuda" and pipe.vae is not None:
        pipe.vae.to(device=device, dtype=torch_dtype).eval()
    frames = pipe.decode_video(latents, **tiler_kwargs)
    video_frames = pipe.tensor2video(frames[0])
    if _is_main_process():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_video(video_frames, str(output_path), fps=args.fps)


def _default_model_file(model_root: str, filename: str) -> str:
    return str(Path(model_root) / filename)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EgoHOI inference on EgoSim baseline/data samples.")
    parser.add_argument("--dataset", required=True, choices=["egodex", "egovid", "agibot", "continuous_generation"])
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--eval_set_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_root", required=True, help="Directory containing Wan base weights.")
    parser.add_argument("--checkpoint", default=os.environ.get("EGOHOI_CHECKPOINT"), help="EgoHOI finetuned checkpoint.")
    parser.add_argument("--dit_path", default=None)
    parser.add_argument("--text_encoder_path", default=None)
    parser.add_argument("--vae_path", default=None)
    parser.add_argument("--image_encoder_path", default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch_dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--prompt", default="a person interacts with objects in an indoor scene")
    parser.add_argument("--negative_prompt", default="blurry, low quality, flicker, distorted hands, color bleeding")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--seed_stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=float, default=128)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile_size", type=int, nargs=2, default=[34, 34])
    parser.add_argument("--tile_stride", type=int, nargs=2, default=[18, 16])
    parser.add_argument("--keep_modules_on_device", action="store_true")
    parser.add_argument("--denoise_progress", action="store_true")
    parser.add_argument("--debug_conditions", action="store_true")
    parser.add_argument("--use_usp", action="store_true", help="Enable Unified Sequence Parallel. Launch with torchrun.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def _finalize_weight_args(args: argparse.Namespace) -> None:
    if not args.checkpoint:
        raise ValueError("EgoHOI requires --checkpoint or EGOHOI_CHECKPOINT.")
    args.dit_path = args.dit_path or _default_model_file(args.model_root, "diffusion_pytorch_model.safetensors")
    args.text_encoder_path = args.text_encoder_path or _default_model_file(args.model_root, "models_t5_umt5-xxl-enc-bf16.pth")
    args.vae_path = args.vae_path or _default_model_file(args.model_root, "Wan2.1_VAE.pth")
    args.image_encoder_path = args.image_encoder_path or _default_model_file(
        args.model_root,
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    )
    args.device = args.device or (f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    _initialize_usp(args)
    _finalize_weight_args(args)
    set_seed(args.seed)

    loader = _resolve_dataset_loader(args.dataset)
    all_samples = _load_samples(args)
    if args.max_samples is not None:
        all_samples = all_samples[: args.max_samples]
    if args.use_usp:
        samples = all_samples
    else:
        samples = [sample for idx, sample in enumerate(all_samples) if idx % args.world_size == args.rank]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = prepare_device(args.device)
    torch_dtype = resolve_torch_dtype(args.torch_dtype)
    train_model, pipe, pipe_vae, tiler_kwargs = initialize_models(args, device, torch_dtype)
    model_bundle = (train_model, pipe, pipe_vae, tiler_kwargs, device, torch_dtype)

    success = 0
    skipped = 0
    failed = 0
    disable_progress = args.use_usp and not _is_main_process()
    for global_idx, sample in enumerate(tqdm(samples, desc=f"[EgoHOI:{args.dataset}]", dynamic_ncols=True, disable=disable_progress)):
        output_path = output_dir / f"{sample.output_id}.mp4"
        if args.skip_existing and output_path.exists():
            skipped += 1
            continue
        try:
            sample_data = _load_baseline_sample(loader, args, sample)
            run_sample_inference(
                args=args,
                model_bundle=model_bundle,
                sample_data=sample_data,
                prompt=sample.prompt,
                clip_seed=args.seed + global_idx * args.seed_stride,
                output_path=output_path,
            )
            success += 1
        except Exception as exc:  # pylint: disable=broad-except
            failed += 1
            if _is_main_process():
                print(f"  [ERROR] {sample.output_id}: {exc}")

    if _is_main_process():
        print(f"[INFO] Done. success={success}, skipped={skipped}, failed={failed}, total={len(samples)}")


if __name__ == "__main__":
    main()
