import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import imageio.v2 as imageio
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from diffsynth import save_video
from egohoi.dataset import TextVideoDataset_onestage  # noqa: E402
from egohoi.model import (  # noqa: E402
    LightningModelForDataProcess,
    LightningModelForTrain_onestage,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Inference utilities for EgoHOI.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Root path of HOT3D dataset.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train/val).")
    parser.add_argument("--clip_id", type=str, required=True, help="Clip identifier, e.g. clip-001849.")
    parser.add_argument("--start_frame", type=int, default=0, help="Starting frame index for conditioning.")
    parser.add_argument("--frame_interval", type=int, default=1, help="Frame interval when sampling sequence.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to synthesize.")
    parser.add_argument("--height", type=int, default=480, help="Output video height.")
    parser.add_argument("--width", type=int, default=480, help="Output video width.")
    parser.add_argument("--prompt", type=str, default="a person interacts with objects in an indoor scene")
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality, flicker, distorted hands, color bleeding")
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--output_fps", type=int, default=24)
    parser.add_argument("--lora_rank", type=int, default=128, help="LoRA rank used during finetuning.")
    parser.add_argument("--lora_alpha", type=float, default=128, help="LoRA alpha used during finetuning.")
    parser.add_argument("--dit_path", type=str, required=True, help="Comma separated list of Wan DiT weights.")
    parser.add_argument("--text_encoder_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--image_encoder_path", type=str, required=True)
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to finetuned checkpoint (PyTorch state dict or DeepSpeed directory).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile_size", type=int, nargs=2, default=[34, 34])
    parser.add_argument("--tile_stride", type=int, nargs=2, default=[18, 16])
    parser.add_argument("--debug_shapes", action="store_true", help="Print intermediate tensor shapes for debugging.")
    parser.add_argument(
        "--debug_conditions",
        action="store_true",
        help="Print statistics for condition embeddings to verify they are applied.",
    )
    return parser.parse_args()


def select_frame_indices(total: int, start: int, num_frames: int, interval: int) -> List[int]:
    indices: List[int] = []
    cur = start
    for _ in range(num_frames):
        idx = max(min(cur, total - 1), 0)
        indices.append(idx)
        cur += interval
    return indices


def load_clip_sample(
    dataset: TextVideoDataset_onestage,
    clip_id: str,
    start_frame: int,
    num_frames: int,
    frame_interval: int,
) -> Dict[str, torch.Tensor]:
    clip_info = None
    for info in dataset.video_list:
        if info["clip_id"] == clip_id:
            clip_info = info
            break
    if clip_info is None:
        raise ValueError(f"Clip {clip_id} not found under {dataset.base_path}.")

    camera_meta = dataset._load_camera_meta(clip_info)
    total_frames = min(
        len(camera_meta.get("frames", [])),
        len(clip_info["pose_frames"]),
        len(clip_info["object_frames"]),
    )
    if total_frames <= 0:
        raise ValueError(f"Clip {clip_id} has no frames.")

    frame_indices = select_frame_indices(total_frames, start_frame, num_frames, frame_interval)
    ref_index = frame_indices[0]

    with imageio.get_reader(clip_info["video_path"]) as reader:
        first_frame = Image.fromarray(reader.get_data(ref_index)).convert("RGB")
    first_frame = dataset._resize_image(first_frame, InterpolationMode.BILINEAR)

    dwpose_data = dataset._load_image_sequence(
        clip_info["pose_dir"],
        clip_info["pose_frames"],
        frame_indices,
        InterpolationMode.BILINEAR,
    )

    random_ref_dwpose = dataset._load_image_tensor(
        os.path.join(
            clip_info["pose_dir"],
            clip_info["pose_frames"][min(ref_index, len(clip_info["pose_frames"]) - 1)],
        ),
        InterpolationMode.BILINEAR,
        channel_last=True,
    )
    random_ref_object = dataset._load_image_tensor(
        os.path.join(
            clip_info["object_dir"],
            clip_info["object_frames"][min(ref_index, len(clip_info["object_frames"]) - 1)],
        ),
        InterpolationMode.NEAREST,
        channel_last=True,
    )

    camera_embedding = dataset._compute_camera_embedding(camera_meta, frame_indices)

    return {
        "frame_indices": frame_indices,
        "first_frame": first_frame,
        "dwpose": dwpose_data,  # [C, F, H, W]
        "random_ref_dwpose": random_ref_dwpose,  # [H, W, 3]
        "random_ref_object": random_ref_object,  # [H, W, 3]
        "camera_embedding": camera_embedding,  # [6, F, H, W]
        "clip_info": clip_info,
    }


def prepare_pose_condition(model, dwpose_uint8: torch.Tensor, device: torch.device, target_dtype: torch.dtype):
    pose_sequence = torch.cat(
        [
            dwpose_uint8[:, :, :1].repeat(1, 1, 3, 1, 1),
            dwpose_uint8,
        ],
        dim=2,
    )
    pose_input = (pose_sequence / 255.0).to(device=device, dtype=model.dwpose_embedding[0].weight.dtype)
    pose_features = model.dwpose_embedding(pose_input)
    condition_tokens = rearrange(
        pose_features.to(dtype=target_dtype),
        "b c f h w -> b (f h w) c",
    ).contiguous()
    return condition_tokens


def prepare_random_ref_pose(model, pose_ref: torch.Tensor, device: torch.device, target_dtype: torch.dtype):
    pose_tensor = (
        pose_ref.permute(0, 3, 1, 2).to(device=device, dtype=model.randomref_embedding_pose[0].weight.dtype) / 255.0
    )
    pose_embedding = model.randomref_embedding_pose(pose_tensor).unsqueeze(2)
    return pose_embedding.to(dtype=target_dtype)


def prepare_object_tokens(
    model,
    object_image: torch.Tensor,
    vae,
    device: torch.device,
    tiler_kwargs,
    target_dtype: torch.dtype,
):
    image = (object_image.permute(0, 3, 1, 2) / 255.0).to(device)
    image = image.unsqueeze(2)  # [B, 3, 1, H, W]
    image = image * 2.0 - 1.0
    latents = vae.encode_video(image.to(dtype=vae.torch_dtype, device=vae.device), **tiler_kwargs)[0]
    latents = latents.unsqueeze(0).to(device=device)
    patchifier_dtype = model.object_patchifier.proj.weight.dtype
    tokens, grid = model.object_patchifier(latents.to(dtype=patchifier_dtype))
    tokens = tokens.to(dtype=target_dtype)
    return tokens, grid


def prepare_camera_tokens(model, camera_embedding, target_shape, device, target_dtype):
    camera_encoder = model.camera_encoder.to("cpu")
    first_param = next(camera_encoder.parameters(), None)
    encode_dtype = first_param.dtype if first_param is not None else torch.float32
    camera_input = camera_embedding.to(device="cpu", dtype=encode_dtype)
    with torch.cuda.amp.autocast(enabled=False):
        tokens, _ = camera_encoder(camera_input, target_shape=target_shape)
    return tokens.to(device=device, dtype=target_dtype)


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def debug_tensor_stats(name: str, tensor: torch.Tensor, num_values: int = 5):
    if tensor is None:
        print(f"[DEBUG] {name}: None")
        return
    if not isinstance(tensor, torch.Tensor):
        print(f"[DEBUG] {name}: {type(tensor).__name__} = {tensor}")
        return
    with torch.no_grad():
        detatched = tensor.detach()
        if detatched.is_cuda:
            detatched = detatched.to("cpu")
        flat = detatched.reshape(-1)
        info = f"[DEBUG] {name}: shape={tuple(detatched.shape)} dtype={detatched.dtype}"
        if flat.numel() == 0:
            print(info + " (empty tensor)")
            return
        flat_fp32 = flat.to(torch.float32)
        stats = {
            "min": flat_fp32.min().item(),
            "max": flat_fp32.max().item(),
            "mean": flat_fp32.mean().item(),
            "std": flat_fp32.std(unbiased=False).item(),
        }
        info += " " + " ".join(f"{key}={value:.6f}" for key, value in stats.items())
        print(info)
        if num_values > 0:
            sample = flat_fp32[:num_values].tolist()
            print(f"[DEBUG] {name}: first_values={sample}")


def load_checkpoint_state(path: str) -> Dict[str, torch.Tensor]:
    if os.path.isfile(path):
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict):
            if isinstance(ckpt.get("state_dict"), dict):
                return ckpt["state_dict"]
            if isinstance(ckpt.get("module"), dict):
                return ckpt["module"]
        if not isinstance(ckpt, dict):
            raise ValueError(f"Checkpoint at {path} does not contain a state dict.")
        return ckpt

    if not os.path.isdir(path):
        raise ValueError(f"No checkpoint file found at {path}.")

    search_dirs = []
    latest_file = os.path.join(path, "latest")
    if os.path.isfile(latest_file):
        with open(latest_file, "r") as f:
            latest_subdir = f.read().strip()
        latest_path = os.path.join(path, latest_subdir)
        if os.path.isdir(latest_path):
            search_dirs.append(latest_path)
    search_dirs.append(path)

    for base_dir in search_dirs:
        model_state_path = os.path.join(base_dir, "mp_rank_00_model_states.pt")
        if os.path.isfile(model_state_path):
            ckpt = torch.load(model_state_path, map_location="cpu")
            state_dict = ckpt.get("module", ckpt)
            if not isinstance(state_dict, dict):
                raise ValueError(f"Checkpoint shard at {model_state_path} does not contain a state dict.")
            return state_dict

    raise ValueError(f"No checkpoint file found at {path}.")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    set_seed(args.seed)

    split_dir = os.path.join(args.dataset_path, args.split)
    dataset = TextVideoDataset_onestage(
        base_path=split_dir,
        metadata_path="",
        max_num_frames=args.num_frames,
        frame_interval=args.frame_interval,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        is_i2v=True,
    )

    sample = load_clip_sample(
        dataset,
        clip_id=args.clip_id,
        start_frame=args.start_frame,
        num_frames=args.num_frames,
        frame_interval=args.frame_interval,
    )

    model_VAE = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=tuple(args.tile_size),
        tile_stride=tuple(args.tile_stride),
    )

    train_model = LightningModelForTrain_onestage(
        dit_path=args.dit_path,
        train_architecture="lora",
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        model_VAE=model_VAE,
    )
    train_model.use_gradient_checkpointing = False
    train_model.use_gradient_checkpointing_offload = False
    train_model.eval()  # ensure eval mode to control VRAM usage

    state_dict = load_checkpoint_state(args.checkpoint)
    missing, unexpected = train_model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"[WARN] Unexpected keys in checkpoint: {unexpected}")
    if missing:
        print(f"[INFO] Missing keys during load (typically optimizer stats): {missing}")

    train_model.to(device=device, dtype=torch_dtype)
    train_model.dwpose_embedding.to(device=device, dtype=torch_dtype)
    train_model.randomref_embedding_pose.to(device=device, dtype=torch_dtype)
    train_model.object_patchifier.to(device=device, dtype=torch_dtype)
    train_model.camera_encoder.to("cpu")
    train_model.camera_residual_layers.to(device=device, dtype=torch_dtype)
    train_model.pipe.denoising_model().to(device=device, dtype=torch_dtype).eval()

    pipe = train_model.pipe
    pipe.device = device
    pipe.torch_dtype = torch_dtype
    pipe.scheduler.set_timesteps(args.num_inference_steps, shift=args.sigma_shift)
    pipe_VAE = train_model.pipe_VAE
    pipe_VAE.device = device
    pipe_VAE.torch_dtype = torch_dtype
    pipe.text_encoder = pipe_VAE.text_encoder
    pipe.image_encoder = pipe_VAE.image_encoder
    pipe.vae = pipe_VAE.vae

    tiler_kwargs = {"tiled": args.tiled, "tile_size": tuple(args.tile_size), "tile_stride": tuple(args.tile_stride)}

    height, width = args.height, args.width
    latent_shape = (1, 16, (args.num_frames - 1) // 4 + 1, height // 8, width // 8)
    noise = pipe.generate_noise(latent_shape, seed=args.seed, device=device, dtype=torch.float32)
    latents = noise.to(device=device, dtype=torch_dtype)
    if args.debug_shapes:
        print(f"[DEBUG] latents shape (after noise init): {tuple(latents.shape)}")
        print(f"[DEBUG] camera embedding shape (from dataset): {tuple(sample['camera_embedding'].shape)}")

    if device.type == "cuda" and pipe.text_encoder is not None:
        pipe.text_encoder.to(device=device, dtype=torch_dtype).eval()
    prompt_emb_pos = pipe_VAE.encode_prompt(args.prompt, positive=True)
    prompt_emb_pos["context"] = prompt_emb_pos["context"].to(device=device)
    if args.cfg_scale != 1.0:
        prompt_emb_neg = pipe_VAE.encode_prompt(args.negative_prompt, positive=False)
        prompt_emb_neg["context"] = prompt_emb_neg["context"].to(device=device)
    else:
        prompt_emb_neg = None
    if device.type == "cuda" and pipe.text_encoder is not None:
        pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()

    first_frame = sample["first_frame"]
    if device.type == "cuda":
        if pipe.image_encoder is not None:
            pipe.image_encoder.to(device=device, dtype=torch.float32).eval()
        if pipe.vae is not None:
            pipe.vae.to(device=device, dtype=torch_dtype).eval()
    image_emb_pos = pipe_VAE.encode_image(first_frame, args.num_frames, height, width)
    for key, value in image_emb_pos.items():
        image_emb_pos[key] = value.to(device=device)
    if args.debug_conditions:
        for key, value in image_emb_pos.items():
            debug_tensor_stats(f"image_emb_pos[{key}]", value)

    dwpose_uint8 = sample["dwpose"].unsqueeze(0)  # [1, 3, F, H, W]
    condition_tokens = prepare_pose_condition(train_model, dwpose_uint8, device, torch_dtype)
    if args.debug_conditions:
        debug_tensor_stats("condition_tokens", condition_tokens)

    random_ref_dwpose = sample["random_ref_dwpose"].unsqueeze(0)
    random_ref_pose_emb = prepare_random_ref_pose(train_model, random_ref_dwpose, device, torch_dtype)
    if args.debug_conditions:
        debug_tensor_stats("random_ref_pose_emb", random_ref_pose_emb)
    if "y" in image_emb_pos:
        if args.debug_conditions:
            debug_tensor_stats("image_emb_pos[y]_before_random_ref_pose", image_emb_pos["y"])
        image_emb_pos["y"] = image_emb_pos["y"] + random_ref_pose_emb.to(dtype=image_emb_pos["y"].dtype)
        if args.debug_conditions:
            debug_tensor_stats("image_emb_pos[y]_after_random_ref_pose", image_emb_pos["y"])

    image_emb_neg = {}
    if args.cfg_scale != 1.0:
        if "clip_feature" in image_emb_pos:
            image_emb_neg["clip_feature"] = image_emb_pos["clip_feature"]
        if "y" in image_emb_pos:
            image_emb_neg["y"] = torch.zeros_like(image_emb_pos["y"])
        if args.debug_conditions:
            for key, value in image_emb_neg.items():
                debug_tensor_stats(f"image_emb_neg[{key}]", value)

    random_ref_object = sample["random_ref_object"].unsqueeze(0)
    object_tokens, object_grid = prepare_object_tokens(
        train_model,
        random_ref_object,
        pipe_VAE,
        device,
        tiler_kwargs,
        torch_dtype,
    )
    if args.debug_conditions:
        debug_tensor_stats("object_tokens", object_tokens)
        debug_tensor_stats("object_grid", object_grid)

    camera_embedding = sample["camera_embedding"].unsqueeze(0)
    if args.debug_shapes and camera_embedding.shape[2] != latents.shape[2]:
        print(f"[DEBUG] camera embedding temporal len {camera_embedding.shape[2]} will be resampled to {latents.shape[2]}")
    camera_tokens = prepare_camera_tokens(
        train_model,
        camera_embedding,
        target_shape=latents.shape[2:],
        device=device,
        target_dtype=torch_dtype,
    )
    if args.debug_shapes:
        print(f"[DEBUG] camera tokens shape: {tuple(camera_tokens.shape)}")
    if args.debug_conditions:
        debug_tensor_stats("camera_tokens", camera_tokens)

    if device.type == "cuda":
        if pipe.image_encoder is not None:
            pipe.image_encoder.to("cpu")
        if pipe.vae is not None:
            pipe.vae.to("cpu")
        torch.cuda.empty_cache()

    condition_tokens_uncond = torch.zeros_like(condition_tokens)
    camera_tokens_uncond = torch.zeros_like(camera_tokens)

    torch.set_grad_enabled(False)
    timesteps = pipe.scheduler.timesteps
    progress_bar = tqdm(range(len(timesteps)), desc="Denoising", dynamic_ncols=True)
    for progress_id in progress_bar:
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
    progress_bar.close()

    if device.type == "cuda" and pipe.vae is not None:
        pipe.vae.to(device=device, dtype=torch_dtype).eval()
    frames = pipe.decode_video(latents, **tiler_kwargs)
    video_frames = pipe.tensor2video(frames[0])
    save_dir = os.path.dirname(args.output_path)
    os.makedirs(save_dir, exist_ok=True)
    save_video(video_frames, args.output_path, fps=args.output_fps)
    print(f"Saved video to {args.output_path}")


if __name__ == "__main__":
    main()
