import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import imageio
import torch
import torch.multiprocessing as mp
from PIL import Image
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from diffsynth import save_video  # noqa: E402
from egohoi.inference import (  # noqa: E402
    debug_tensor_stats,
    load_checkpoint_state,
    load_clip_sample,
    prepare_camera_tokens,
    prepare_object_tokens,
    prepare_pose_condition,
    prepare_random_ref_pose,
    select_frame_indices,
    set_seed,
)
from egohoi.dataset import TextVideoDataset_onestage  # noqa: E402
from egohoi.model import (  # noqa: E402
    LightningModelForDataProcess,
    LightningModelForTrain_onestage,
)

DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    try:
        return DTYPE_MAP[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype '{dtype_name}'.") from exc

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch inference for EgoHOI over an entire dataset."
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Root path containing dataset splits.")
    parser.add_argument("--output_root", type=str, required=True, help="Directory where generated videos are saved.")
    parser.add_argument(
        "--splits",
        type=str,
        nargs="*",
        default=None,
        help="Dataset splits to process. When omitted or set to 'all', splits are auto-discovered.",
    )
    parser.add_argument("--start_frame", type=int, default=0, help="Starting frame index for conditioning.")
    parser.add_argument("--frame_interval", type=int, default=1, help="Frame interval when sampling sequence.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to synthesize.")
    parser.add_argument("--height", type=int, default=480, help="Output video height.")
    parser.add_argument("--width", type=int, default=480, help="Output video width.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="a person interacts with objects in an indoor scene",
        help="Positive text prompt used during inference.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="blurry, low quality, flicker, distorted hands, color bleeding",
        help="Negative text prompt for classifier-free guidance.",
    )
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier-free guidance scale.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps.")
    parser.add_argument("--sigma_shift", type=float, default=5.0, help="Sigma shift applied to the scheduler.")
    parser.add_argument("--seed", type=int, default=123, help="Base random seed.")
    parser.add_argument(
        "--seed_stride",
        type=int,
        default=1,
        help="Offset added to the base seed for each clip (seed + idx * seed_stride).",
    )
    parser.add_argument("--output_fps", type=int, default=24, help="Frames per second for the saved video.")
    parser.add_argument("--lora_rank", type=int, default=128, help="LoRA rank used during finetuning.")
    parser.add_argument("--lora_alpha", type=float, default=128, help="LoRA alpha used during finetuning.")
    parser.add_argument("--dit_path", type=str, required=True, help="Comma separated list of Wan DiT weights.")
    parser.add_argument("--text_encoder_path", type=str, required=True, help="Path to the text encoder weights.")
    parser.add_argument("--vae_path", type=str, required=True, help="Path to the VAE weights.")
    parser.add_argument("--image_encoder_path", type=str, required=True, help="Path to the image encoder weights.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to finetuned checkpoint (PyTorch state dict or DeepSpeed directory).",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference (cpu/cuda).")
    parser.add_argument("--torch_dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--num_workers", type=int, default=None, help="Number of worker processes to use.")
    parser.add_argument(
        "--worker_devices",
        type=str,
        default=None,
        help="Comma separated list of devices for workers (e.g., 'cuda:0,cuda:1').",
    )
    parser.add_argument(
        "--disable_multiprocessing",
        action="store_true",
        help="Force single-process execution even if multiple devices are available.",
    )
    parser.add_argument(
        "--keep_modules_on_device",
        action="store_true",
        help="Keep text/image encoders and VAE on the selected device to reduce CPU memory usage.",
    )
    parser.add_argument(
        "--denoise_progress",
        action="store_true",
        help="Show a progress bar for denoising steps inside each clip.",
    )
    parser.add_argument(
        "--log_condition_tokens",
        action="store_true",
        help="Print representative values of condition tokens fed into the model.",
    )
    parser.add_argument("--tiled", action="store_true", help="Enable VAE tiling for decoding.")
    parser.add_argument("--tile_size", type=int, nargs=2, default=[34, 34], help="Tile size for VAE tiling.")
    parser.add_argument("--tile_stride", type=int, nargs=2, default=[18, 16], help="Tile stride for VAE tiling.")
    parser.add_argument("--debug_shapes", action="store_true", help="Print intermediate tensor shapes for debugging.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip clips whose outputs already exist.")
    parser.add_argument(
        "--max_clips_per_split",
        type=int,
        default=None,
        help="Limit the number of clips processed per split (useful for smoke tests).",
    )
    parser.add_argument(
        "--clip_offset",
        type=int,
        default=0,
        help="Number of clips to skip at the start of each split before processing.",
    )
    parser.add_argument(
        "--debug_conditions",
        action="store_true",
        help="Print statistics for condition embeddings to verify they are applied.",
    )
    parser.add_argument(
        "--hand_pose_root",
        type=str,
        default=None,
        help="Override pose sequence directory root (expects <root>/<clip_id> with pose frames).",
    )
    return parser.parse_args()


def _dataset_path_is_split(dataset_root: str) -> bool:
    return os.path.isdir(os.path.join(dataset_root, "videos"))


def _split_name_from_path(dataset_root: str) -> str:
    return os.path.basename(os.path.normpath(dataset_root))


def resolve_split_dir(dataset_root: str, split: str) -> str:
    if _dataset_path_is_split(dataset_root):
        return dataset_root
    return os.path.join(dataset_root, split)


def discover_available_splits(dataset_root: str) -> List[str]:
    if not os.path.isdir(dataset_root):
        return []

    if _dataset_path_is_split(dataset_root):
        return [_split_name_from_path(dataset_root)]

    splits: List[str] = []
    for entry in sorted(os.listdir(dataset_root)):
        split_dir = os.path.join(dataset_root, entry)
        if not os.path.isdir(split_dir):
            continue
        if os.path.isdir(os.path.join(split_dir, "videos")):
            splits.append(entry)
    return splits


def build_dataset(
    split_dir: str,
    args,
) -> Optional[TextVideoDataset_onestage]:
    try:
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
    except ValueError as exc:
        print(f"[WARN] {exc}")
        return None

    dataset.video_list.sort(key=lambda info: info["clip_id"])
    return dataset


def load_clip_sample_with_pose_root(
    dataset: TextVideoDataset_onestage,
    clip_id: str,
    start_frame: int,
    num_frames: int,
    frame_interval: int,
    pose_root: str,
) -> Dict[str, torch.Tensor]:
    clip_info = None
    for info in dataset.video_list:
        if info["clip_id"] == clip_id:
            clip_info = info
            break
    if clip_info is None:
        raise ValueError(f"Clip {clip_id} not found under {dataset.base_path}.")

    camera_meta = dataset._load_camera_meta(clip_info)
    noisy_pose_dir = os.path.join(pose_root, clip_id)
    if not os.path.isdir(noisy_pose_dir):
        raise FileNotFoundError(f"Pose directory not found: {noisy_pose_dir}")
    noisy_pose_frames = dataset._list_frames(noisy_pose_dir)
    if not noisy_pose_frames:
        raise FileNotFoundError(f"No pose frames found under: {noisy_pose_dir}")

    total_frames = min(
        len(camera_meta.get("frames", [])),
        len(noisy_pose_frames),
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
        noisy_pose_dir,
        noisy_pose_frames,
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
        "dwpose": dwpose_data,  # [C, F, H, W] from noisy pose root
        "random_ref_dwpose": random_ref_dwpose,  # [H, W, 3] from original saved_pose
        "random_ref_object": random_ref_object,  # [H, W, 3]
        "camera_embedding": camera_embedding,  # [6, F, H, W]
        "clip_info": clip_info,
    }


def log_tensor_stats(label: str, tensor: torch.Tensor) -> None:
    if tensor is None:
        print(f"[COND] {label}: None")
        return
    if not isinstance(tensor, torch.Tensor):
        print(f"[COND] {label}: {type(tensor).__name__} = {tensor}")
        return
    with torch.no_grad():
        flat = tensor.detach().float().flatten()
        if flat.numel() == 0:
            print(f"[COND] {label}: empty tensor")
            return
        stats = {
            "min": flat.min().item(),
            "max": flat.max().item(),
            "mean": flat.mean().item(),
            "var": flat.var(unbiased=False).item(),
        }
        stats_str = ", ".join(f"{key}={value:.6f}" for key, value in stats.items())
        print(f"[COND] {label}: shape={tuple(tensor.shape)} {stats_str}")


def _normalize_device_token(token: str) -> str:
    token = token.strip()
    if not token:
        return token
    if token.isdigit():
        return f"cuda:{token}"
    if token.startswith("cuda") and ":" not in token and token != "cuda":
        # e.g. "cuda0" -> "cuda:0"
        index = token[4:]
        if index.isdigit():
            return f"cuda:{index}"
    return token


def determine_worker_devices(args) -> List[str]:
    if args.worker_devices:
        raw_tokens = [tok for tok in args.worker_devices.split(",") if tok.strip()]
        devices = [_normalize_device_token(tok) for tok in raw_tokens]
    else:
        normalized = _normalize_device_token(args.device)
        if normalized == "cuda" and torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count <= 0:
                devices = ["cpu"]
            else:
                devices = [f"cuda:{idx}" for idx in range(device_count)]
        elif normalized.startswith("cuda") and torch.cuda.is_available():
            devices = [normalized]
        elif normalized.startswith("cuda") and not torch.cuda.is_available():
            devices = ["cpu"]
        else:
            devices = [normalized]

    if args.num_workers is not None:
        desired = max(1, int(args.num_workers))
        devices = devices[:desired]

    if not devices:
        return ["cpu"]

    available_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    filtered: List[str] = []
    for dev in devices:
        if dev.startswith("cuda:"):
            index_part = dev.split(":", 1)[1]
            if index_part.isdigit() and int(index_part) >= available_cuda:
                continue
        filtered.append(dev)
    if not filtered:
        return ["cpu"]
    return filtered


def split_tasks_among_workers(tasks: Sequence[Dict[str, Any]], num_workers: int) -> List[List[Dict[str, Any]]]:
    total_tasks = len(tasks)
    if total_tasks == 0:
        return [[]]
    if num_workers <= 1:
        return [list(tasks)]
    effective_workers = min(num_workers, total_tasks)
    base = total_tasks // effective_workers
    remainder = total_tasks % effective_workers

    result: List[List[Dict[str, Any]]] = []
    start = 0
    for idx in range(effective_workers):
        extra = 1 if idx < remainder else 0
        end = start + base + extra
        result.append(list(tasks[start:end]))
        start = end
    return result


def collect_clip_tasks(args, splits: Sequence[str]) -> Dict[str, Any]:
    tasks: List[Dict[str, Any]] = []
    skipped = 0
    global_clip_index = 0
    for split in splits:
        split_dir = resolve_split_dir(args.dataset_path, split)
        if not os.path.isdir(split_dir):
            print(f"[WARN] Split '{split}' not found, skipping.")
            continue

        dataset = build_dataset(split_dir, args)
        if dataset is None:
            continue

        clip_offset = max(args.clip_offset or 0, 0)
        clip_infos = dataset.video_list[clip_offset:]
        if args.max_clips_per_split is not None:
            clip_infos = clip_infos[: args.max_clips_per_split]

        global_clip_index += clip_offset

        if not clip_infos:
            print(f"[INFO] Split '{split}' has no clips to process after applying offset/limit, skipping.")
            continue

        split_output_dir = os.path.join(args.output_root, split)
        os.makedirs(split_output_dir, exist_ok=True)

        for clip_info in clip_infos:
            clip_id = clip_info["clip_id"]
            clip_seed = args.seed + global_clip_index * args.seed_stride
            output_path = os.path.join(split_output_dir, f"{clip_id}.mp4")

            if args.skip_existing and os.path.isfile(output_path):
                skipped += 1
                global_clip_index += 1
                continue

            tasks.append(
                {
                    "split": split,
                    "clip_id": clip_id,
                    "clip_seed": clip_seed,
                    "output_path": output_path,
                }
            )
            global_clip_index += 1

    return {
        "tasks": tasks,
        "skipped": skipped,
        "total_planned": global_clip_index,
    }


def run_single_clip_inference(
    clip_id: str,
    dataset: TextVideoDataset_onestage,
    args,
    train_model: LightningModelForTrain_onestage,
    pipe,
    pipe_VAE,
    tiler_kwargs: Dict,
    device: torch.device,
    torch_dtype: torch.dtype,
    clip_seed: int,
    output_path: str,
) -> None:
    if args.hand_pose_root:
        sample = load_clip_sample_with_pose_root(
            dataset,
            clip_id=clip_id,
            start_frame=args.start_frame,
            num_frames=args.num_frames,
            frame_interval=args.frame_interval,
            pose_root=args.hand_pose_root,
        )
    else:
        sample = load_clip_sample(
            dataset,
            clip_id=clip_id,
            start_frame=args.start_frame,
            num_frames=args.num_frames,
            frame_interval=args.frame_interval,
        )

    pipe.scheduler.set_timesteps(args.num_inference_steps, shift=args.sigma_shift)
    height, width = args.height, args.width
    latent_shape = (1, 16, (args.num_frames - 1) // 4 + 1, height // 8, width // 8)
    noise = pipe.generate_noise(latent_shape, seed=clip_seed, device=device, dtype=torch.float32)
    latents = noise.to(device=device, dtype=torch_dtype)
    if args.debug_shapes:
        print(f"[DEBUG][{clip_id}] latents shape: {tuple(latents.shape)}")
        print(f"[DEBUG][{clip_id}] camera embedding shape: {tuple(sample['camera_embedding'].shape)}")

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
        if not args.keep_modules_on_device:
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
            debug_tensor_stats(f"{clip_id}: image_emb_pos[{key}]", value)
    if args.log_condition_tokens:
        for key, value in image_emb_pos.items():
            log_tensor_stats(f"{clip_id}: image_emb_pos[{key}]", value)

    dwpose_uint8 = sample["dwpose"].unsqueeze(0)
    condition_tokens = prepare_pose_condition(train_model, dwpose_uint8, device, torch_dtype)
    if args.debug_conditions:
        debug_tensor_stats(f"{clip_id}: condition_tokens", condition_tokens)
    if args.log_condition_tokens:
        log_tensor_stats(f"{clip_id}: condition_tokens", condition_tokens)

    random_ref_dwpose = sample["random_ref_dwpose"].unsqueeze(0)
    random_ref_pose_emb = prepare_random_ref_pose(train_model, random_ref_dwpose, device, torch_dtype)
    if args.debug_conditions:
        debug_tensor_stats(f"{clip_id}: random_ref_pose_emb", random_ref_pose_emb)
    if "y" in image_emb_pos:
        if args.debug_conditions:
            debug_tensor_stats(f"{clip_id}: image_emb_pos[y]_before_random_ref_pose", image_emb_pos["y"])
        image_emb_pos["y"] = image_emb_pos["y"] + random_ref_pose_emb.to(dtype=image_emb_pos["y"].dtype)
        if args.debug_conditions:
            debug_tensor_stats(f"{clip_id}: image_emb_pos[y]_after_random_ref_pose", image_emb_pos["y"])
        if args.log_condition_tokens:
            log_tensor_stats(f"{clip_id}: random_ref_pose_emb", random_ref_pose_emb)
            log_tensor_stats(f"{clip_id}: image_emb_pos[y]_with_pose", image_emb_pos["y"])
    elif args.log_condition_tokens:
        log_tensor_stats(f"{clip_id}: random_ref_pose_emb", random_ref_pose_emb)

    image_emb_neg: Dict[str, torch.Tensor] = {}
    if args.cfg_scale != 1.0:
        if "clip_feature" in image_emb_pos:
            image_emb_neg["clip_feature"] = image_emb_pos["clip_feature"]
        if "y" in image_emb_pos:
            image_emb_neg["y"] = torch.zeros_like(image_emb_pos["y"])
        if args.debug_conditions:
            for key, value in image_emb_neg.items():
                debug_tensor_stats(f"{clip_id}: image_emb_neg[{key}]", value)
        if args.log_condition_tokens:
            for key, value in image_emb_neg.items():
                log_tensor_stats(f"{clip_id}: image_emb_neg[{key}]", value)

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
        debug_tensor_stats(f"{clip_id}: object_tokens", object_tokens)
        debug_tensor_stats(f"{clip_id}: object_grid", object_grid)
    if args.log_condition_tokens:
        log_tensor_stats(f"{clip_id}: object_tokens", object_tokens)
        log_tensor_stats(f"{clip_id}: object_grid", object_grid)

    camera_embedding = sample["camera_embedding"].unsqueeze(0)
    if args.debug_shapes and camera_embedding.shape[2] != latents.shape[2]:
        print(
            f"[DEBUG][{clip_id}] camera embedding temporal len {camera_embedding.shape[2]} "
            f"will be resampled to {latents.shape[2]}"
        )
    camera_tokens = prepare_camera_tokens(
        train_model,
        camera_embedding,
        target_shape=latents.shape[2:],
        device=device,
        target_dtype=torch_dtype,
    )
    if args.debug_shapes:
        print(f"[DEBUG][{clip_id}] camera tokens shape: {tuple(camera_tokens.shape)}")
    if args.debug_conditions:
        debug_tensor_stats(f"{clip_id}: camera_tokens", camera_tokens)
    if args.log_condition_tokens:
        log_tensor_stats(f"{clip_id}: camera_tokens", camera_tokens)

    if device.type == "cuda":
        if not args.keep_modules_on_device:
            if pipe.image_encoder is not None:
                pipe.image_encoder.to("cpu")
            if pipe.vae is not None:
                pipe.vae.to("cpu")
        torch.cuda.empty_cache()

    condition_tokens_uncond = torch.zeros_like(condition_tokens)
    camera_tokens_uncond = torch.zeros_like(camera_tokens)

    torch.set_grad_enabled(False)
    timesteps = pipe.scheduler.timesteps
    if args.denoise_progress:
        desc = f"{clip_id} denoise"
        with tqdm(range(len(timesteps)), desc=desc, leave=False, dynamic_ncols=True) as step_bar:
            for progress_id in step_bar:
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
    else:
        for progress_id, timestep in enumerate(timesteps):
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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_video(video_frames, output_path, fps=args.output_fps)
    if device.type == "cuda":
        torch.cuda.empty_cache()


def prepare_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            print(f"[WARN] CUDA device '{device_str}' requested but CUDA is unavailable. Falling back to CPU.")
            return torch.device("cpu")
        if device_str == "cuda":
            torch.cuda.set_device(torch.cuda.current_device())
            return torch.device("cuda")
        index_part = device_str.split(":", 1)[1] if ":" in device_str else None
        if index_part and index_part.isdigit():
            torch.cuda.set_device(int(index_part))
        return torch.device(device_str)
    return torch.device(device_str)


def initialize_models(args, device: torch.device, torch_dtype: torch.dtype):
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
    train_model.eval()

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
    pipe_VAE = train_model.pipe_VAE
    pipe_VAE.device = device
    pipe_VAE.torch_dtype = torch_dtype
    pipe.text_encoder = pipe_VAE.text_encoder
    pipe.image_encoder = pipe_VAE.image_encoder
    pipe.vae = pipe_VAE.vae

    tiler_kwargs = {"tiled": args.tiled, "tile_size": tuple(args.tile_size), "tile_stride": tuple(args.tile_stride)}
    return train_model, pipe, pipe_VAE, tiler_kwargs


def run_serial_inference(
    args,
    tasks: Sequence[Dict[str, Any]],
    torch_dtype: torch.dtype,
    device_str: str,
):
    if not tasks:
        return {"processed": 0, "successful": 0, "failed": []}

    device = prepare_device(device_str)
    set_seed(args.seed)
    train_model, pipe, pipe_VAE, tiler_kwargs = initialize_models(args, device, torch_dtype)

    tasks_by_split: Dict[str, List[Dict[str, Any]]] = {}
    for task in tasks:
        tasks_by_split.setdefault(task["split"], []).append(task)

    successful = 0
    failed: List[str] = []
    processed = 0

    for split, split_tasks in tasks_by_split.items():
        dataset = build_dataset(resolve_split_dir(args.dataset_path, split), args)
        if dataset is None:
            for task in split_tasks:
                processed += 1
                failed.append(f"{split}/{task['clip_id']}: dataset creation failed.")
            continue

        iterator = tqdm(split_tasks, desc=f"{split} clips", dynamic_ncols=True)
        for task in iterator:
            processed += 1
            clip_id = task["clip_id"]
            try:
                run_single_clip_inference(
                    clip_id=clip_id,
                    dataset=dataset,
                    args=args,
                    train_model=train_model,
                    pipe=pipe,
                    pipe_VAE=pipe_VAE,
                    tiler_kwargs=tiler_kwargs,
                    device=device,
                    torch_dtype=torch_dtype,
                    clip_seed=task["clip_seed"],
                    output_path=task["output_path"],
                )
                successful += 1
                iterator.set_postfix_str(f"done {clip_id}")
            except Exception as exc:  # pylint: disable=broad-except
                failed.append(f"{split}/{clip_id}: {exc}")
                iterator.set_postfix_str(f"fail {clip_id}")
                print(f"[ERROR] Failed to process {split}/{clip_id}: {exc}")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {"processed": processed, "successful": successful, "failed": failed}


def worker_entry(
    worker_id: int,
    device_str: str,
    tasks: Sequence[Dict[str, Any]],
    args_dict: Dict[str, Any],
    torch_dtype_name: str,
    result_queue,
):
    summary = {"worker_id": worker_id, "processed": 0, "successful": 0, "failed": []}
    try:
        args = argparse.Namespace(**args_dict)
        torch_dtype = resolve_torch_dtype(torch_dtype_name)
        device = prepare_device(device_str)

        set_seed(args.seed + worker_id)
        train_model, pipe, pipe_VAE, tiler_kwargs = initialize_models(args, device, torch_dtype)

        dataset_cache: Dict[str, Optional[TextVideoDataset_onestage]] = {}
        for task in tasks:
            summary["processed"] += 1
            split = task["split"]
            clip_id = task["clip_id"]

            if split not in dataset_cache:
                dataset_cache[split] = build_dataset(resolve_split_dir(args.dataset_path, split), args)

            dataset = dataset_cache[split]
            if dataset is None:
                msg = f"{split}/{clip_id}: dataset creation failed."
                summary["failed"].append(msg)
                print(f"[WORKER {worker_id}] {msg}")
                continue

            try:
                run_single_clip_inference(
                    clip_id=clip_id,
                    dataset=dataset,
                    args=args,
                    train_model=train_model,
                    pipe=pipe,
                    pipe_VAE=pipe_VAE,
                    tiler_kwargs=tiler_kwargs,
                    device=device,
                    torch_dtype=torch_dtype,
                    clip_seed=task["clip_seed"],
                    output_path=task["output_path"],
                )
                summary["successful"] += 1
                print(f"[WORKER {worker_id}] done {split}/{clip_id}")
            except Exception as exc:  # pylint: disable=broad-except
                msg = f"{split}/{clip_id}: {exc}"
                summary["failed"].append(msg)
                print(f"[WORKER {worker_id}] fail {msg}")
    except Exception as exc:  # pylint: disable=broad-except
        err_msg = f"[worker {worker_id}] fatal error: {exc}"
        summary["failed"].append(err_msg)
        print(f"[WORKER {worker_id}] Fatal error: {exc}")
    finally:
        if summary["processed"] == 0 and not summary["failed"] and tasks:
            summary["failed"].append(f"[worker {worker_id}] no tasks processed due to initialization failure.")
        if device_str.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        result_queue.put(summary)


def run_parallel_inference(
    args,
    tasks: Sequence[Dict[str, Any]],
    devices: Sequence[str],
):
    if not tasks:
        return {"processed": 0, "successful": 0, "failed": []}

    worker_task_lists = split_tasks_among_workers(tasks, len(devices))
    effective_workers = min(len(worker_task_lists), len(devices))
    if effective_workers == 0:
        return {"processed": 0, "successful": 0, "failed": []}

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []

    args_dict = vars(args)
    dtype_name = args.torch_dtype

    for worker_id in range(effective_workers):
        worker_tasks = worker_task_lists[worker_id]
        if not worker_tasks:
            continue
        device_str = devices[worker_id]
        process = ctx.Process(
            target=worker_entry,
            args=(worker_id, device_str, worker_tasks, args_dict, dtype_name, result_queue),
        )
        process.start()
        processes.append(process)

    summaries = []
    for _ in processes:
        summaries.append(result_queue.get())
    for process in processes:
        process.join()

    processed = sum(item["processed"] for item in summaries)
    successful = sum(item["successful"] for item in summaries)
    failed: List[str] = []
    for item in summaries:
        failed.extend(item["failed"])

    return {"processed": processed, "successful": successful, "failed": failed}


def main():
    args = parse_args()
    torch_dtype = resolve_torch_dtype(args.torch_dtype)

    if not os.path.isdir(args.dataset_path):
        raise ValueError(f"Dataset path '{args.dataset_path}' does not exist.")
    os.makedirs(args.output_root, exist_ok=True)

    requested_splits = args.splits or []
    if _dataset_path_is_split(args.dataset_path):
        split_name = _split_name_from_path(args.dataset_path)
        if requested_splits and not any(split.lower() == "all" for split in requested_splits):
            if split_name not in requested_splits:
                print(
                    f"[WARN] dataset_path points to a split; overriding requested splits {requested_splits} "
                    f"with '{split_name}'."
                )
        splits = [split_name]
    else:
        if not requested_splits or any(split.lower() == "all" for split in requested_splits):
            splits = discover_available_splits(args.dataset_path)
        else:
            splits = requested_splits

    if not splits:
        raise ValueError(f"No dataset splits found under {args.dataset_path}.")

    task_info = collect_clip_tasks(args, splits)
    tasks: List[Dict[str, Any]] = task_info["tasks"]
    skipped = task_info["skipped"]

    if not tasks:
        total_clips = skipped
        summary = (
            f"Inference completed - processed {total_clips} clips "
            f"(0 succeeded, {skipped} skipped, 0 failed)."
        )
        print(summary)
        return

    devices = determine_worker_devices(args)
    if args.disable_multiprocessing or len(tasks) == 1:
        devices = [devices[0]]
    if len(devices) == 1:
        device_str = devices[0]
        print(f"[INFO] Using device: {device_str}")
        result = run_serial_inference(args, tasks, torch_dtype, device_str)
    else:
        effective_devices = devices[: min(len(devices), len(tasks))]
        print(f"[INFO] Launching {len(effective_devices)} workers on devices: {', '.join(effective_devices)}")
        result = run_parallel_inference(args, tasks, effective_devices)

    processed = result["processed"]
    successful = result["successful"]
    failed = result["failed"]

    total_clips = processed + skipped
    summary = (
        f"Inference completed - processed {total_clips} clips "
        f"({successful} succeeded, {skipped} skipped, {len(failed)} failed)."
    )
    print(summary)
    if failed:
        print("Failures:")
        for item in failed:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
