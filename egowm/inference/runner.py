"""
Main inference runner for EgoWM baseline.

Replaces cache-based loading with online encoding via egowm.inference.encoders.
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import torch
import numpy as np
import imageio
from pathlib import Path
from tqdm import tqdm

from egowm.data import egodex, egovid, agibot
from egowm.inference.pipeline import load_pipeline, encode_mask_to_latent, model_fn
from egowm.inference.encoders import encode_ego_prior, encode_prompt, encode_first_frame


def load_video_rgb(video_path: str, target_frames: int = 61, height: int = 480, width: int = 832) -> np.ndarray:
    """Load video → uint8 numpy array [F, H, W, 3]."""
    from PIL import Image
    reader = imageio.get_reader(video_path)
    frames = [frame for frame in reader]
    reader.close()
    if len(frames) < target_frames:
        frames += [frames[-1]] * (target_frames - len(frames))
    else:
        frames = frames[:target_frames]
    resized = [np.array(Image.fromarray(f).resize((width, height), Image.LANCZOS)) for f in frames]
    return np.stack(resized, axis=0).astype(np.uint8)


def save_comparison_video(
    out_path: str,
    ego_prior_frames: np.ndarray,
    hand_frames: np.ndarray,
    generated_frames: np.ndarray,
    fps: int = 16,
) -> None:
    """Save side-by-side comparison: [ego_prior | hand | generated]."""
    combined = np.concatenate([ego_prior_frames, hand_frames, generated_frames], axis=2)
    imageio.mimwrite(out_path, combined, fps=fps, quality=8)


def load_mask_video(mask_path: str, target_frames: int = 61, height: int = 480, width: int = 832) -> torch.Tensor:
    """Load mask video → tensor [1, F, H, W] in [0, 1]."""
    from PIL import Image

    reader = imageio.get_reader(mask_path)
    frames = []
    for frame_data in reader:
        frame = Image.fromarray(frame_data).resize((width, height), Image.BILINEAR)
        frames.append(frame)
    reader.close()

    if len(frames) < target_frames:
        last = frames[-1] if frames else Image.new("RGB", (width, height))
        frames += [last] * (target_frames - len(frames))
    else:
        frames = frames[:target_frames]

    import numpy as np
    arr = np.stack([np.array(f) for f in frames], axis=0)
    tensor = torch.from_numpy(arr).float().permute(3, 0, 1, 2)  # [C, F, H, W]
    tensor = tensor / 255.0 * 2.0 - 1.0
    tensor = tensor.unsqueeze(0)                                  # [1, C, F, H, W]

    mask_video_raw = (tensor + 1.0) / 2.0
    mask_video_raw = mask_video_raw.clamp(0, 1)
    mask_video_raw = mask_video_raw[:, :1].squeeze(0)             # [1, F, H, W]
    mask_video_raw[:, 0, :, :] = 0.0                              # keep first frame unmasked
    return mask_video_raw


@torch.no_grad()
def run_inference_single(
    pipe,
    cloud_latent: torch.Tensor,      # [16, f, h, w]
    hand_latent: torch.Tensor,       # [16, f, h, w]
    mask_video_raw: torch.Tensor,    # [1, F, H, W]
    prompt_embedding: torch.Tensor,  # [seq, 4096]
    image_embedding: torch.Tensor,   # [257, 1280]
    device: torch.device,
    num_inference_steps: int = 50,
) -> np.ndarray:
    dtype = torch.bfloat16

    cloud_latent   = cloud_latent.unsqueeze(0).to(device, dtype=dtype)   # [1,16,f,h,w]
    hand_latent    = hand_latent.unsqueeze(0).to(device, dtype=dtype)    # [1,16,f,h,w]
    mask_video_raw = mask_video_raw.to(device, dtype=dtype)

    _, f, h, w = cloud_latent.shape[1:]
    mask_latent = encode_mask_to_latent(mask_video_raw, (16, f, h, w))
    mask_latent = mask_latent.unsqueeze(0).to(device, dtype=dtype)       # [1,4,f,h,w]

    ctx  = prompt_embedding.unsqueeze(0).to(device, dtype=dtype)         # [1,seq,4096]
    clip = image_embedding.unsqueeze(0).to(device, dtype=dtype)          # [1,257,1280]

    latents = torch.randn(cloud_latent.shape, device=device, dtype=dtype)

    # DiT on GPU for denoising
    pipe.dit.to(device)
    pipe.scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(pipe.scheduler.timesteps.to(device), desc="Denoising", leave=False):
        mask_weight = mask_latent[:, :1].expand_as(cloud_latent)
        masked_ego  = cloud_latent * (1.0 - mask_weight)
        # 52ch: [noisy(16), mask(4), ego*(1-mask)(16), hand(16)]
        model_in = torch.cat([latents, mask_latent, masked_ego, hand_latent], dim=1)
        ts = t.unsqueeze(0).to(dtype=dtype, device=device)

        noise_pred = model_fn(pipe.dit, model_in, ts, ctx, clip)
        latents    = pipe.scheduler.step(noise_pred, t, latents)

    # VAE decode
    pipe.vae.to(device)
    decoded = pipe.vae.decode([latents.squeeze(0)], device=device)
    out = decoded.squeeze(0).permute(1, 2, 3, 0)   # [T, H, W, C]
    out = ((out + 1.0) / 2.0).clamp(0, 1)
    return (out * 255).float().cpu().numpy().astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(
        description="EgoWM baseline inference: ego prior + mask, online encoding, no preprocess required."
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["egodex", "egovid", "agibot"],
                        help="Dataset to run inference on")
    parser.add_argument("--model_root", type=str, required=True,
                        help="Path to EgoSim-14B model directory")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Dataset root directory")
    parser.add_argument("--metadata_path", type=str, required=True,
                        help="Path to metadata CSV")
    parser.add_argument("--eval_set_path", type=str, default=None,
                        help="Path to eval_set.txt (egovid only)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for generated videos")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=61)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)
    print(f"[Rank {args.rank}] Dataset: {args.dataset}, device: {device}")

    # Load samples
    if args.dataset == "egodex":
        all_samples = egodex.load_samples(args.metadata_path)
    elif args.dataset == "egovid":
        all_samples = egovid.load_samples(args.metadata_path, args.eval_set_path)
    else:
        all_samples = agibot.load_samples(args.metadata_path)

    if args.max_samples:
        all_samples = all_samples[:args.max_samples]

    # Shard across GPUs
    samples = [s for i, s in enumerate(all_samples) if i % args.world_size == args.rank]
    print(f"[Rank {args.rank}] {len(samples)} samples (total {len(all_samples)})")

    # Load pipeline
    print(f"[Rank {args.rank}] Loading pipeline from {args.model_root}...")
    pipe = load_pipeline(args.model_root, device=str(device))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    success, skipped, failed = 0, 0, 0

    for sample in tqdm(samples, desc=f"[Rank {args.rank}]"):
        output_id = sample.output_id
        out_path = output_dir / f"{output_id}.mp4"

        if args.skip_existing and out_path.exists():
            skipped += 1
            continue

        try:
            # Online encode: ego prior → VAE latent
            ego_prior_path = str(
                egodex.get_ego_prior_video_path(args.dataset_root, sample)
                if args.dataset == "egodex"
                else egovid.get_ego_prior_video_path(args.dataset_root, sample)
                if args.dataset == "egovid"
                else agibot.get_ego_prior_video_path(args.dataset_root, sample)
            )
            cloud_latent = encode_ego_prior(
                pipe, ego_prior_path, device,
                target_frames=args.num_frames, height=args.height, width=args.width,
            )

            # Online encode: hand keypoint video → VAE latent
            hand_video_path = str(
                egodex.get_hand_video_path(args.dataset_root, sample)
                if args.dataset == "egodex"
                else egovid.get_hand_video_path(args.dataset_root, sample)
                if args.dataset == "egovid"
                else agibot.get_hand_video_path(args.dataset_root, sample)
            )
            hand_latent = encode_ego_prior(
                pipe, hand_video_path, device,
                target_frames=args.num_frames, height=args.height, width=args.width,
            )

            # Online encode: prompt → T5 embedding
            prompt_embedding = encode_prompt(pipe, sample.prompt, device)

            # Online encode: first frame → CLIP embedding
            first_frame_path = str(
                egodex.get_first_frame_path(args.dataset_root, sample)
                if args.dataset == "egodex"
                else egovid.get_first_frame_path(args.dataset_root, sample)
                if args.dataset == "egovid"
                else agibot.get_first_frame_path(args.dataset_root, sample)
            )
            image_embedding = encode_first_frame(
                pipe, first_frame_path, device, height=args.height, width=args.width,
            )

            # Load mask
            if args.dataset == "egodex":
                mask_path = egodex.get_mask_path(args.dataset_root, sample)
            elif args.dataset == "egovid":
                mask_path = egovid.get_mask_path(args.dataset_root, sample)
            else:
                mask_path = agibot.get_mask_path(args.dataset_root, sample)

            target_frames_pixel = cloud_latent.shape[1] * 4 + 1
            if mask_path.exists():
                mask_video_raw = load_mask_video(
                    str(mask_path), target_frames=target_frames_pixel,
                    height=args.height, width=args.width,
                )
            else:
                print(f"  No mask found at {mask_path}, using zeros")
                mask_video_raw = torch.zeros(1, target_frames_pixel, args.height, args.width)

            # Run denoising
            generated_video = run_inference_single(
                pipe=pipe,
                cloud_latent=cloud_latent,
                hand_latent=hand_latent,
                mask_video_raw=mask_video_raw,
                prompt_embedding=prompt_embedding,
                image_embedding=image_embedding,
                device=device,
                num_inference_steps=args.num_inference_steps,
            )

            imageio.mimwrite(str(out_path), generated_video, fps=args.fps, quality=8)

            # Save side-by-side comparison: ego_prior | hand | generated
            cmp_path = output_dir / f"{output_id}_cmp.mp4"
            ego_frames = load_video_rgb(ego_prior_path, target_frames=args.num_frames, height=args.height, width=args.width)
            hand_frames = load_video_rgb(hand_video_path, target_frames=args.num_frames, height=args.height, width=args.width)
            save_comparison_video(str(cmp_path), ego_frames, hand_frames, generated_video, fps=args.fps)

            success += 1

        except Exception as e:
            import traceback
            print(f"  Failed [{output_id}]: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n[Rank {args.rank}] Done. success={success}, skipped={skipped}, failed={failed}")
    print(f"[Rank {args.rank}] Output: {output_dir}")


if __name__ == "__main__":
    main()
