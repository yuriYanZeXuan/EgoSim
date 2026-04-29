"""
Online encoders for EgoWM inference.
Aligned with inference_realcap_wan21.py (the working reference).
"""
import torch
import numpy as np
import imageio
from PIL import Image


def _load_video_frames(video_path: str, target_frames: int, height: int, width: int) -> torch.Tensor:
    """Read video → [3, F, H, W] float32 in [-1, 1]."""
    reader = imageio.get_reader(video_path)
    frames = [frame for frame in reader]
    reader.close()

    if len(frames) < target_frames:
        frames += [frames[-1]] * (target_frames - len(frames))
    else:
        frames = frames[:target_frames]

    resized = []
    for f in frames:
        pil = Image.fromarray(f)
        if pil.size != (width, height):
            pil = pil.resize((width, height), Image.LANCZOS)
        resized.append(np.array(pil))

    arr = np.stack(resized, axis=0).astype(np.float32)   # [F, H, W, 3]
    t = torch.from_numpy(arr).permute(3, 0, 1, 2)        # [3, F, H, W]
    t = t / 255.0 * 2.0 - 1.0                            # [-1, 1]
    return t


@torch.no_grad()
def encode_ego_prior(
    pipe,
    video_path: str,
    device: torch.device,
    target_frames: int = 61,
    height: int = 480,
    width: int = 832,
) -> torch.Tensor:
    """Encode video → VAE latent [16, f, h, w]."""
    video_tensor = _load_video_frames(video_path, target_frames, height, width)
    vae = pipe.vae
    vae.to(device)
    dtype = next(vae.parameters()).dtype
    x = video_tensor.unsqueeze(0).to(device, dtype=dtype)  # [1, 3, F, H, W]
    latent = vae.encode(x, device=device)                  # [1, 16, f, h, w]
    return latent.squeeze(0).to(dtype=torch.bfloat16)       # [16, f, h, w]


@torch.no_grad()
def encode_prompt(pipe, text: str, device: torch.device) -> torch.Tensor:
    """Encode text → T5 embedding [seq, 4096]."""
    if not hasattr(pipe, "prompter") or pipe.prompter is None:
        raise RuntimeError("pipe.prompter is not set.")
    pipe.prompter.fetch_models(text_encoder=pipe.text_encoder)
    pipe.text_encoder.to(device)
    emb = pipe.prompter.encode_prompt(text, device=device)  # [1, seq, 4096]
    return emb.squeeze(0).to(dtype=torch.bfloat16)           # [seq, 4096]


@torch.no_grad()
def encode_first_frame(
    pipe,
    image_path: str,
    device: torch.device,
    height: int = 480,
    width: int = 832,
) -> torch.Tensor:
    """
    Encode image (or first frame of video) → CLIP embedding [257, 1280].
    image_path: path to .png/.jpg or .mp4 (first frame used).
    """
    ext = image_path.lower().rsplit(".", 1)[-1]
    if ext in ("mp4", "avi", "mov", "mkv"):
        reader = imageio.get_reader(image_path)
        frame = next(iter(reader))
        reader.close()
        pil = Image.fromarray(frame)
    else:
        pil = Image.open(image_path).convert("RGB")

    if pil.size != (width, height):
        pil = pil.resize((width, height), Image.LANCZOS)

    arr = np.array(pil).astype(np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
    tensor = tensor / 255.0                           # [0, 1]  — matches training cache

    pipe.image_encoder.to(device)
    dtype = next(pipe.image_encoder.parameters()).dtype
    emb = pipe.image_encoder.encode_image(
        tensor.unsqueeze(0).to(device, dtype=dtype)
    )  # [1, 257, 1280]
    return emb.squeeze(0).to(dtype=torch.bfloat16)  # [257, 1280]
