"""
Pipeline loading and DiT forward pass for EgoSim inference.

Architecture: EgoSim-14B (fine-tuned from Wan2.1-Fun-14B-InP), in_dim=52
  52-channel InP format: [noisy(16), mask(4), ego*(1-mask)(16), hand(16)]

Model directory layout (EgoSim-14B/):
  diffusion_pytorch_model.safetensors          — fine-tuned DiT (in_dim=52)
  Wan2.1_VAE.pth                               — VAE
  models_t5_umt5-xxl-enc-bf16.pth             — T5 encoder
  models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth  — CLIP encoder
  google/umt5-xxl/                             — T5 tokenizer
"""
import os
import torch
from einops import rearrange


def load_pipeline(model_root: str, device: str = "cuda"):
    """
    Load WanVideoPipeline with EgoSim-14B fine-tuned weights (in_dim=52).

    Args:
        model_root: path to EgoSim-14B model directory
        device: target device, e.g. "cuda" or "cuda:0"
    """
    from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
    from diffsynth.models.wan_video_dit import WanModel
    from diffsynth.prompters.wan_prompter import WanPrompter
    from safetensors.torch import load_file as load_safetensors

    model_root = os.path.abspath(model_root)

    dit_config = {
        "has_image_input":               True,
        "patch_size":                    [1, 2, 2],
        "in_dim":                        52,
        "dim":                           5120,
        "ffn_dim":                       13824,
        "freq_dim":                      256,
        "text_dim":                      4096,
        "out_dim":                       16,
        "num_heads":                     40,
        "num_layers":                    40,
        "eps":                           1e-6,
        "seperated_timestep":            True,
        "require_clip_embedding":        True,
        "require_vae_embedding":         False,
        "fuse_vae_embedding_in_latents": True,
    }

    dit_path = os.path.join(model_root, "diffusion_pytorch_model.safetensors")
    if not os.path.exists(dit_path):
        raise FileNotFoundError(f"DiT weights not found: {dit_path}")

    print(f"  Building WanModel (in_dim=52)...")
    dit = WanModel(**dit_config)
    print(f"  Loading DiT: {dit_path}")
    state_dict = load_safetensors(dit_path)
    missing, unexpected = dit.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:3]}{'...' if len(missing) > 3 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
    dit = dit.to(dtype=torch.bfloat16)
    print(f"  DiT loaded")

    model_configs = [
        ModelConfig(path=os.path.join(model_root, "models_t5_umt5-xxl-enc-bf16.pth")),
        ModelConfig(path=os.path.join(model_root, "Wan2.1_VAE.pth")),
        ModelConfig(path=os.path.join(model_root, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")),
        ModelConfig(path=os.path.join(model_root, "google/umt5-xxl")),
    ]

    print(f"  Loading pipeline (VAE + text/image encoders)...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cpu",
        model_configs=model_configs,
        tokenizer_config=ModelConfig(path=os.path.join(model_root, "google/umt5-xxl")),
    )
    pipe.dit = dit

    tokenizer_path = os.path.join(model_root, "google/umt5-xxl")
    if os.path.exists(tokenizer_path):
        pipe.prompter = WanPrompter(tokenizer_path=tokenizer_path)
    else:
        print(f"  Warning: tokenizer not found at {tokenizer_path}")

    print(f"  Pipeline ready")
    return pipe


def encode_mask_to_latent(mask_video_raw: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """Encode binary mask [1, F, H, W] → latent [4, f, h, w] via nearest patch."""
    import torch.nn.functional as F

    _, target_f, target_h, target_w = target_shape

    mask = torch.where(mask_video_raw > 0.5, 1.0, 0.0)
    mask_ds = F.interpolate(
        mask.unsqueeze(0),
        size=(target_f, target_h * 2, target_w * 2),
        mode="nearest",
    ).squeeze(0).squeeze(0)

    mask_p = mask_ds.view(target_f, target_h, 2, target_w, 2)
    mask_p = mask_p.permute(2, 4, 0, 1, 3).reshape(4, target_f, target_h, target_w)
    return mask_p


def model_fn(dit, latents, timestep, context, clip_feature):
    """
    DiT forward pass for EgoSim (in_dim=52).

    latents: [B, 52, f, h, w]  = [noisy(16), mask(4), ego*(1-mask)(16), hand(16)]
    """
    from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d

    dev   = next(dit.parameters()).device
    dtype = next(dit.parameters()).dtype

    timestep     = timestep.to(device=dev)
    latents      = latents.to(device=dev, dtype=dtype)
    context      = context.to(device=dev, dtype=dtype)
    if clip_feature is not None:
        clip_feature = clip_feature.to(device=dev, dtype=dtype)

    t     = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).to(dtype=dtype))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    if clip_feature is not None and hasattr(dit, "img_emb"):
        context = torch.cat([dit.img_emb(clip_feature), context], dim=1)

    x = dit.patch_embedding(latents)
    b, c, f, h, w = x.shape
    x = rearrange(x, "b c f h w -> b (f h w) c")

    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    for block in dit.blocks:
        x = block(x, context, t_mod, freqs)

    t_mod_head = t.unsqueeze(1).expand(-1, 2, -1)
    x = dit.head(x, t_mod_head)
    return dit.unpatchify(x, (f, h, w))
