import os
import math
import sys

import lightning as pl
import torch
from PIL import Image
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(FILE_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from diffsynth import WanVideoPipeline, ModelManager, load_state_dict, load_state_dict_from_folder
from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
from peft import LoraConfig, inject_adapter_in_model
os.environ["TOKENIZERS_PARALLELISM"] = "false"




class ObjectPatchifier(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, latents: torch.Tensor):
        tokens = self.proj(latents)
        grid = tokens.shape[2:]
        tokens = rearrange(tokens, "b c f h w -> b (f h w) c").contiguous()
        return tokens, grid


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int, expansion: int = 2, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv3d(channels, channels * expansion, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels * expansion)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv3d(channels * expansion, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        identity = x
        x = self.conv1(self.act1(self.norm1(x)))
        x = self.conv2(self.act2(self.norm2(x)))
        return F.silu(identity + x)


class ZeroLinear(nn.Linear):
    def __init__(self, dim: int):
        super().__init__(dim, dim, bias=False)
        nn.init.zeros_(self.weight)

    def reset_parameters(self):
        nn.init.zeros_(self.weight)


def _make_group_norm(num_channels: int, groups: int = 8):
    while num_channels % groups != 0 and groups > 1:
        groups //= 2
    return nn.GroupNorm(groups, num_channels)


class TemporalSelfAttention2D(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        max_frames: int = 128,
        dropout: float = 0.0,
        use_learnable_pe: bool = True,
        t_max: int = 1024,
    ):
        super().__init__()
        self.channels = channels
        self.max_frames = max_frames
        self.use_learnable_pe = use_learnable_pe
        self.t_max = t_max

        if use_learnable_pe:
            self.time_embed = nn.Parameter(torch.zeros(t_max, channels))
            nn.init.normal_(self.time_embed, std=0.01)
        else:
            self.register_buffer("time_pe_sin", self._build_sin_pe(t_max, channels), persistent=False)

        self.attn_norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)

        self.ff_norm = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
        )
        self.ff_drop = nn.Dropout(dropout)

        self.gamma_attn = nn.Parameter(torch.zeros(1))
        self.gamma_ff = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _build_sin_pe(t_max: int, dim: int):
        pos = torch.arange(t_max, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(dim, dtype=torch.float32).unsqueeze(0)
        div = torch.exp(-(2 * (i // 2)) * math.log(10000.0) / dim)
        pe = pos * div
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe

    def forward(self, x: torch.Tensor):
        # x: [B, C, F, H, W]
        b, c, f, h, w = x.shape
        if f > self.max_frames:
            raise ValueError(f"TemporalSelfAttention2D max_frames={self.max_frames}, but got {f} frames.")

        tokens = x.permute(0, 3, 4, 2, 1).reshape(b * h * w, f, c)

        if self.use_learnable_pe:
            if f > self.t_max:
                pe = self._build_sin_pe(f, c).to(dtype=tokens.dtype, device=tokens.device)
            else:
                pe = self.time_embed[:f].to(dtype=tokens.dtype, device=tokens.device)
        else:
            pe = self.time_pe_sin[:f].to(dtype=tokens.dtype, device=tokens.device)
        tokens = tokens + pe.unsqueeze(0)

        attn_in = self.attn_norm(tokens)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        tokens = tokens + self.attn_drop(self.gamma_attn * attn_out)

        ff_in = self.ff_norm(tokens)
        ff_out = self.ff(ff_in)
        tokens = tokens + self.ff_drop(self.gamma_ff * ff_out)

        tokens = tokens.view(b, h, w, f, c).permute(0, 4, 3, 1, 2).contiguous()
        return tokens


class Conv2DTemporalStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool,
        num_heads: int,
        max_frames: int,
        dropout: float = 0.0,
        use_learnable_pe: bool = True,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            _make_group_norm(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            _make_group_norm(out_channels),
            nn.SiLU(),
        )
        self.downsample = (
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
            if downsample
            else None
        )
        self.temporal_attn = TemporalSelfAttention2D(
            out_channels,
            num_heads=num_heads,
            max_frames=max_frames,
            dropout=dropout,
            use_learnable_pe=use_learnable_pe,
        )

    def forward(self, x: torch.Tensor, downsample: bool):
        # x: [B, C, F, H, W]
        b, c, f, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
        x = self.conv(x)
        if self.downsample is not None and downsample:
            x = self.downsample(x)
        _, c_out, h_out, w_out = x.shape
        x = x.reshape(b, f, c_out, h_out, w_out).permute(0, 2, 1, 3, 4)
        x = self.temporal_attn(x)
        return x


class CausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        pad_t = (self.kernel_size[0] - 1) * self.dilation[0]
        if pad_t > 0:
            x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        return super().forward(x)


class CausalDownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        padding_h = kernel_size[1] // 2
        padding_w = kernel_size[2] // 2
        self.conv = CausalConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(0, padding_h, padding_w),
        )
        self.norm = _make_group_norm(out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, debug: bool = False, name: str = ""):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if debug:
            print(f"[DEBUG][camera][{name}] shape after block: {tuple(x.shape)}")
        return x


class TemporalCausalDownsampler(nn.Module):
    def __init__(self, in_channels: int, plan: tuple):
        super().__init__()
        blocks = []
        current_channels = in_channels
        self.total_temporal_stride = 1
        self.total_spatial_stride = 1
        for idx, (out_channels, kernel_size, stride) in enumerate(plan):
            block = CausalDownsampleBlock(
                current_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
            )
            blocks.append(block)
            current_channels = out_channels
            self.total_temporal_stride *= stride[0]
            self.total_spatial_stride *= stride[1]
        self.blocks = nn.ModuleList(blocks)
        self.output_channels = current_channels

    def forward(self, x: torch.Tensor, debug: bool = False):
        for idx, block in enumerate(self.blocks):
            x = block(x, debug=debug, name=f"downsample_{idx}")
        return x


class CameraEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        embed_dim: int,
        patch_size,
        stage_channels: tuple = (64, 128, 256),
        num_heads: int = 4,
        max_frames: int = 128,
        temporal_dropout: float = 0.0,
        use_learnable_pe: bool = True,
    ):
        super().__init__()
        del hidden_channels  # kept for backwards compatibility
        self.patch_size = patch_size
        self.input_channels = in_channels
        if len(stage_channels) == 0:
            raise ValueError("stage_channels must contain at least one entry.")
        downsample_plan = (
            (stage_channels[0], (3, 3, 3), (2, 2, 2)),
            (stage_channels[0], (3, 3, 3), (2, 2, 2)),
            (stage_channels[0], (3, 3, 3), (1, 2, 2)),
        )
        self.downsampler = TemporalCausalDownsampler(in_channels, downsample_plan)
        stages = []
        prev_channels = self.downsampler.output_channels
        for idx, out_channels in enumerate(stage_channels):
            stage = Conv2DTemporalStage(
                in_channels=prev_channels,
                out_channels=out_channels,
                downsample=True,
                num_heads=num_heads,
                max_frames=max_frames,
                dropout=temporal_dropout,
                use_learnable_pe=use_learnable_pe,
            )
            stages.append(stage)
            prev_channels = out_channels
        self.stages = nn.ModuleList(stages)
        self.output_channels = prev_channels
        self.norm = _make_group_norm(self.output_channels)
        self.proj = nn.Conv3d(self.output_channels, self.output_channels, kernel_size=1)
        self.patchifier = nn.Conv3d(
            self.output_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        nn.init.kaiming_normal_(self.patchifier.weight, nonlinearity="linear")
        self.patchifier.weight.data.mul_(0.3)
        nn.init.zeros_(self.patchifier.bias)
        self.gamma = nn.Parameter(torch.ones(1))
        self.temporal_aligners = nn.ModuleDict()

    def _init_temporal_weights(self, f_in: int, target_f: int, dtype: torch.dtype) -> torch.Tensor:
        weights = torch.zeros(target_f, f_in, dtype=dtype)
        if target_f <= 1:
            weights.fill_(1.0 / f_in)
            return weights
        for t in range(target_f):
            pos = t * (f_in - 1) / (target_f - 1)
            left = int(math.floor(pos))
            right = min(left + 1, f_in - 1)
            if right == left:
                weights[t, left] = 1.0
            else:
                right_w = pos - left
                left_w = 1.0 - right_w
                weights[t, left] = left_w
                weights[t, right] = right_w
        return weights

    def _get_temporal_aligner(self, f_in: int, target_f: int, device, dtype) -> nn.Linear:
        key = f"{f_in}->{target_f}"
        if key not in self.temporal_aligners:
            layer = nn.Linear(f_in, target_f, bias=False)
            init_weights = self._init_temporal_weights(f_in, target_f, dtype=torch.float32)
            layer.weight.data.copy_(init_weights)
            self.temporal_aligners[key] = layer
        aligner = self.temporal_aligners[key]
        return aligner.to(device=device, dtype=dtype)

    def forward(self, camera_embedding: torch.Tensor, target_shape, debug: bool = False):
        camera_embedding = camera_embedding.to(dtype=self.patchifier.weight.dtype)
        features = self.downsampler(camera_embedding, debug=debug)
        _, _, f_in, h_in, w_in = features.shape
        target_f, target_h, target_w = target_shape
        if f_in != target_f:
            if target_f <= 0:
                raise ValueError("target temporal length must be positive.")
            aligner = self._get_temporal_aligner(f_in, target_f, features.device, features.dtype)
            b, c, _, h, w = features.shape
            flat = rearrange(features, "b c f h w -> (b c h w) f")
            flat = aligner(flat)
            features = rearrange(flat, "(b c h w) f -> b c f h w", b=b, c=c, h=h, w=w)
            f_in = features.shape[2]
            if debug:
                print(f"[DEBUG][camera] temporal alignment applied: {f_in}->{target_f}")
        if h_in < target_h or w_in < target_w:
            raise ValueError(
                f"Camera encoder expects spatial input >= target resolution. "
                f"Got input ({h_in}, {w_in}) vs target ({target_h}, {target_w})."
            )

        required_downsamples_h = int(math.log2(h_in / target_h)) if h_in != target_h else 0
        required_downsamples_w = int(math.log2(w_in / target_w)) if w_in != target_w else 0
        if 2 ** required_downsamples_h * target_h != h_in or 2 ** required_downsamples_w * target_w != w_in:
            raise ValueError(
                "Camera encoder currently supports power-of-two downsampling only. "
                f"Input ({h_in}, {w_in}) vs target ({target_h}, {target_w})."
            )
        if required_downsamples_h != required_downsamples_w:
            raise ValueError(
                f"Camera encoder requires uniform downsampling. ratio_h={h_in/target_h}, ratio_w={w_in/target_w}."
            )
        required_downsamples = required_downsamples_h
        if required_downsamples > len(self.stages):
            raise ValueError(
                f"Not enough stages to reach target resolution. Need {required_downsamples}, but only "
                f"{len(self.stages)} stages are configured."
            )

        for idx, stage in enumerate(self.stages):
            features = stage(features, downsample=idx < required_downsamples)
        if features.shape[-2:] != (target_h, target_w):
            raise ValueError(
                f"Camera encoder spatial size mismatch. Expected ({target_h}, {target_w}), got {features.shape[-2:]}."
            )
        features = self.proj(features)
        features = F.silu(self.norm(features))
        projected = self.patchifier(features)
        projected = self.gamma * projected
        grid = projected.shape[2:]
        tokens = rearrange(projected, "b c f h w -> b (f h w) c").contiguous()
        if debug:
            print(f"[DEBUG][camera] output token shape: {tuple(tokens.shape)}")
        return tokens, grid


class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, image_encoder_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        super().__init__()
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

    def test_step(self, batch, batch_idx):
        text, video, path = batch["text"][0], batch["video"], batch["path"][0]

        self.pipe.device = self.device
        if video is not None:
            prompt_emb = self.pipe.encode_prompt(text)
            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            latents = self.pipe.encode_video(video, **self.tiler_kwargs)[0]
            if "first_frame" in batch:
                first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
                _, _, num_frames, height, width = video.shape
                image_emb = self.pipe.encode_image(first_frame, num_frames, height, width)
            else:
                image_emb = {}
            data = {"latents": latents, "prompt_emb": prompt_emb, "image_emb": image_emb}
            torch.save(data, path + ".tensors.pth")


class LightningModelForTrain_onestage(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        lora_rank=4, lora_alpha=4, train_architecture="lora", lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming",
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None,
        model_VAE=None,
        debug_shapes=False,
        # 
    ):
        super().__init__()
        self.debug_shapes = debug_shapes
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])
        
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)

        self.pipe_VAE = model_VAE.pipe.eval()
        self.tiler_kwargs = model_VAE.tiler_kwargs
        dit_model = self.pipe.denoising_model()
        latent_channels = dit_model.patch_embedding.in_channels
        if dit_model.has_image_input:
            # Wan DiT expects 20 extra channels from image conditioning (masks + first-frame features)
            image_condition_channels = 20
            latent_channels = latent_channels - image_condition_channels
        embed_dim = dit_model.dim
        patch_size = dit_model.patch_size

        concat_dim = 4
        self.dwpose_embedding = nn.Sequential(
                    nn.Conv3d(3, concat_dim * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, (3,3,3), stride=(1,2,2), padding=(1,1,1)),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2,2,2), padding=1),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2,2,2), padding=1),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, 5120, (1,2,2), stride=(1,2,2), padding=0))

        randomref_dim = 20
        self.randomref_embedding_pose = nn.Sequential(
                    nn.Conv2d(3, concat_dim * 4, 3, stride=1, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, randomref_dim, 3, stride=2, padding=1),
                    
                    )
        self.object_patchifier = ObjectPatchifier(
            in_channels=latent_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        self.camera_encoder = CameraEncoder(
            in_channels=6,
            hidden_channels=64,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        self.reference_rope_offset = (256, 256, 256)
        self.camera_injection_depth = min(16, len(dit_model.blocks))
        self.camera_residual_layers = nn.ModuleList(
            [ZeroLinear(embed_dim) for _ in range(self.camera_injection_depth)]
        )
        self.freeze_parameters()

        # self.freeze_parameters()
        if train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        else:
            self.pipe.denoising_model().requires_grad_(True)
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        

        
        
    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        dit_model = self.pipe.denoising_model()
        dit_model.train()
        self.pipe_VAE.requires_grad_(False)
        self.pipe_VAE.eval()
        self.randomref_embedding_pose.train()
        self.dwpose_embedding.train()
        self.object_patchifier.train()
        self.object_patchifier.requires_grad_(True)
        self.camera_encoder.train()
        self.camera_encoder.requires_grad_(True)
        for layer in self.camera_residual_layers:
            layer.train()
            layer.requires_grad_(True)
        self.debug_camera_log_steps = 10000
        
        
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)
                
        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            # 
            try:
                state_dict = load_state_dict(pretrained_lora_path)
            except:
                state_dict = load_state_dict_from_folder(pretrained_lora_path)
            # 
            state_dict_new = {}
            state_dict_new_module = {}
            for key in state_dict.keys():
                
                if 'pipe.dit.' in key:
                    key_new = key.split("pipe.dit.")[1]
                    state_dict_new[key_new] = state_dict[key]
                if "dwpose_embedding" in key or "randomref_embedding_pose" in key:
                    state_dict_new_module[key] = state_dict[key]
            state_dict = state_dict_new
            state_dict_new = {}

            for key in state_dict_new_module:
                if "dwpose_embedding" in key:
                    state_dict_new[key.split("dwpose_embedding.")[1]] = state_dict_new_module[key]
            self.dwpose_embedding.load_state_dict(state_dict_new, strict=True)

            state_dict_new = {}
            for key in state_dict_new_module:
                if "randomref_embedding_pose" in key:
                    state_dict_new[key.split("randomref_embedding_pose.")[1]] = state_dict_new_module[key]
            self.randomref_embedding_pose.load_state_dict(state_dict_new,strict=True)

            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")
    
    

    def _should_log_debug(self, batch_idx):
        if not self.debug_shapes:
            return False
        if batch_idx == 0:
            return True
        if getattr(self, "debug_camera_log_steps", 0) > 0:
            current_step = getattr(self, "global_step", 0)
            return current_step < self.debug_camera_log_steps
        return False

    def training_step(self, batch, batch_idx):
        text, video, path = batch["text"][0], batch["video"], batch["path"][0]
        self.pipe_VAE.device = self.device
        debug_active = self._should_log_debug(batch_idx)
        if debug_active:
            print(f"[DEBUG][train] step={getattr(self, 'global_step', 0)} video shape: {tuple(video.shape) if video is not None else None}")
            if "camera_embedding" in batch:
                camera_raw = batch["camera_embedding"]
                print(f"[DEBUG][train] camera_embedding shape (raw): {tuple(camera_raw.shape)}")
                cam_raw_mean = camera_raw.to(dtype=torch.float32).mean().item()
                cam_raw_std = camera_raw.to(dtype=torch.float32).std().item()
                cam_raw_min = camera_raw.min().item()
                cam_raw_max = camera_raw.max().item()
                print(
                    f"[DEBUG][train] camera_embedding stats raw "
                    f"mean={cam_raw_mean:.4f}, std={cam_raw_std:.4f}, "
                    f"min={cam_raw_min:.4f}, max={cam_raw_max:.4f}"
                )
            else:
                print("[DEBUG][train] camera_embedding missing from batch!")
            for key in ("dwpose_data", "random_ref_dwpose_data", "random_ref_object_data", "object_data"):
                if key in batch:
                    tensor = batch[key]
                    tensor_float = tensor.to(dtype=torch.float32)
                    stats = (
                        tensor_float.mean().item(),
                        tensor_float.std().item(),
                        tensor.min().item(),
                        tensor.max().item(),
                    )
                    print(
                        f"[DEBUG][train] {key} stats "
                        f"mean={stats[0]:.4f}, std={stats[1]:.4f}, "
                        f"min={stats[2]:.4f}, max={stats[3]:.4f}"
                    )
                else:
                    print(f"[DEBUG][train] {key} missing from batch!")
        pose_sequence = torch.cat(
            [batch["dwpose_data"][:, :, :1].repeat(1, 1, 3, 1, 1), batch["dwpose_data"]],
            dim=2,
        )
        dwpose_dtype = self.dwpose_embedding[0].weight.dtype
        dwpose_input = (pose_sequence / 255.0).to(self.device, dtype=dwpose_dtype)
        dwpose_data = self.dwpose_embedding(dwpose_input)
        random_ref_dtype = self.randomref_embedding_pose[0].weight.dtype
        random_ref_dwpose_data = self.randomref_embedding_pose(
            (batch["random_ref_dwpose_data"] / 255.0)
            .to(self.device, dtype=random_ref_dtype)
            .permute(0, 3, 1, 2)
        ).unsqueeze(2)
        random_ref_dwpose_data = random_ref_dwpose_data.to(dtype=self.pipe.torch_dtype)

        object_latents = None
        camera_embedding = batch.get("camera_embedding", None)
        with torch.no_grad():
            if video is not None:
                prompt_emb = self.pipe_VAE.encode_prompt(text)
                video = video.to(dtype=self.pipe_VAE.torch_dtype, device=self.pipe_VAE.device)
                latents = self.pipe_VAE.encode_video(video, **self.tiler_kwargs)[0]
                if "first_frame" in batch: # [1, 853, 480, 3]
                    first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
                    _, _, num_frames, height, width = video.shape
                    image_emb = self.pipe_VAE.encode_image(first_frame, num_frames, height, width)
                else:
                    image_emb = {}
                if "random_ref_object_data" in batch:
                    object_ref = (
                        batch["random_ref_object_data"]
                        .to(self.pipe_VAE.device, dtype=self.pipe_VAE.torch_dtype)
                        .permute(0, 3, 1, 2)
                        .unsqueeze(2)
                    )
                    object_ref = object_ref / 255.0 * 2.0 - 1.0
                    object_latents = self.pipe_VAE.encode_video(object_ref, **self.tiler_kwargs)[0].unsqueeze(0)
                batch = {
                    "latents": latents.unsqueeze(0),
                    "prompt_emb": prompt_emb,
                    "image_emb": image_emb,
                }
                if camera_embedding is not None:
                    batch["camera_embedding"] = camera_embedding
                    if debug_active:
                        cam_mean = camera_embedding.to(dtype=torch.float32).mean().item()
                        cam_std = camera_embedding.to(dtype=torch.float32).std().item()
                        cam_min = camera_embedding.min().item()
                        cam_max = camera_embedding.max().item()
                        print(
                            f"[DEBUG][train] preserved camera embedding stats "
                            f"mean={cam_mean:.4f}, std={cam_std:.4f}, "
                            f"min={cam_min:.4f}, max={cam_max:.4f}"
                        )
        
        # p1 = random.random()  # (disabled random dropout; keep for future reference)
        # p = random.random()  # (disabled random dropout; keep for future reference)
        # if p1 < 0.05:
        #     dwpose_data = torch.zeros_like(dwpose_data)
        #     random_ref_dwpose_data = torch.zeros_like(random_ref_dwpose_data)
        latents = batch["latents"].to(self.device)  # [1, 16, 21, 60, 104]
        if debug_active:
            print(f"[DEBUG][train] latents shape after VAE encode: {tuple(latents.shape)}")
        latents_dtype = latents.dtype
        prompt_emb = batch["prompt_emb"] # batch["prompt_emb"]["context"]:  [1, 1, 512, 4096]
        
        prompt_emb["context"] = prompt_emb["context"].to(self.device)
        image_emb = batch["image_emb"]
        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"].to(self.device) # [1, 257, 1280]
            # if p < 0.1:
            #     image_emb["clip_feature"] = torch.zeros_like(image_emb["clip_feature"]) # [1, 257, 1280]
        if "y" in image_emb:
            
            # if p < 0.1:
            #     image_emb["y"] = torch.zeros_like(image_emb["y"])
            image_emb["y"] = image_emb["y"].to(self.device)
            image_emb["y"] = image_emb["y"] + random_ref_dwpose_data.to(image_emb["y"].dtype)  # [1, 20, 21, 104, 60]
        
        object_tokens, object_grid = None, None
        patchifier_dtype = self.object_patchifier.proj.weight.dtype
        if object_latents is not None:
            object_input = object_latents.to(device=self.device, dtype=patchifier_dtype)
        else:
            object_input = torch.zeros(
                (latents.shape[0], self.object_patchifier.proj.in_channels, *latents.shape[2:]),
                device=self.device,
                dtype=patchifier_dtype,
            )
        object_tokens, object_grid = self.object_patchifier(object_input)
        object_tokens = object_tokens.to(device=self.device, dtype=latents_dtype)
        if debug_active:
            if object_tokens is not None:
                obj_mean = object_tokens.to(dtype=torch.float32).mean().item()
                obj_std = object_tokens.to(dtype=torch.float32).std().item()
                obj_min = object_tokens.min().item()
                obj_max = object_tokens.max().item()
                print(
                    f"[DEBUG][train] object_tokens stats "
                    f"mean={obj_mean:.4f}, std={obj_std:.4f}, "
                    f"min={obj_min:.4f}, max={obj_max:.4f}"
                )
                print(f"[DEBUG][train] object_tokens shape: {tuple(object_tokens.shape)}")
            else:
                print("[DEBUG][train] object_tokens is None")

        camera_tokens, camera_grid = None, None
        first_param = next(self.camera_encoder.parameters(), None)
        camera_dtype = first_param.dtype if first_param is not None else latents_dtype
        if "camera_embedding" in batch:
            camera_input = batch["camera_embedding"].to(device=self.device, dtype=camera_dtype)
            if debug_active:
                cam_mean = camera_input.to(dtype=torch.float32).mean().item()
                cam_std = camera_input.to(dtype=torch.float32).std().item()
                cam_min = camera_input.min().item()
                cam_max = camera_input.max().item()
                print(
                    f"[DEBUG][train] camera_input before encoder "
                    f"mean={cam_mean:.4f}, std={cam_std:.4f}, "
                    f"min={cam_min:.4f}, max={cam_max:.4f}"
                )
                print(f"[DEBUG][train] camera_input before encoder shape: {tuple(camera_input.shape)}")
        else:
            camera_in_channels = getattr(self.camera_encoder, "input_channels", None)
            if camera_in_channels is None:
                raise ValueError("Camera encoder must define input_channels when camera_embedding is absent.")
            camera_input = torch.zeros(
                (latents.shape[0], camera_in_channels, *latents.shape[2:]),
                device=self.device,
                dtype=camera_dtype,
            )
        camera_tokens, camera_grid = self.camera_encoder(
            camera_input,
            target_shape=latents.shape[2:],
            debug=debug_active,
        )
        camera_tokens = camera_tokens.to(device=self.device, dtype=latents_dtype)
        if debug_active:
            token_mean = camera_tokens.to(dtype=torch.float32).mean().item()
            token_std = camera_tokens.to(dtype=torch.float32).std().item()
            token_min = camera_tokens.min().item()
            token_max = camera_tokens.max().item()
            print(
                f"[DEBUG][train] camera_tokens stats "
                f"mean={token_mean:.4f}, std={token_std:.4f}, "
                f"min={token_min:.4f}, max={token_max:.4f}"
            )
            print(f"[DEBUG][train] camera_tokens shape: {tuple(camera_tokens.shape)}")
            with torch.no_grad():
                num_layers_preview = min(3, len(self.camera_residual_layers))
                for idx in range(num_layers_preview):
                    layer = self.camera_residual_layers[idx]
                    weight = layer.weight.detach()
                    w_mean = weight.to(dtype=torch.float32).mean().item()
                    w_std = weight.to(dtype=torch.float32).std().item()
                    w_min = weight.min().item()
                    w_max = weight.max().item()
                    print(
                        f"[DEBUG][train] camera_residual_layers[{idx}] weight stats "
                        f"mean={w_mean:.6f}, std={w_std:.6f}, "
                        f"min={w_min:.6f}, max={w_max:.6f}"
                    )
        # if p1 < 0.05:
        #     if object_tokens is not None:
        #         object_tokens = torch.zeros_like(object_tokens)
        #     if camera_tokens is not None:
        #         camera_tokens = torch.zeros_like(camera_tokens)

        condition = rearrange(dwpose_data.to(dtype=latents_dtype), 'b c f h w -> b (f h w) c').contiguous()
        if debug_active:
            cond_mean = condition.to(dtype=torch.float32).mean().item()
            cond_std = condition.to(dtype=torch.float32).std().item()
            cond_min = condition.min().item()
            cond_max = condition.max().item()
            print(
                f"[DEBUG][train] condition tokens stats "
                f"mean={cond_mean:.4f}, std={cond_std:.4f}, "
                f"min={cond_min:.4f}, max={cond_max:.4f}"
            )
            print(f"[DEBUG][train] condition tokens shape: {tuple(condition.shape)}")
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        self.pipe.prepare_extra_input(latents)  # keep compatibility; returns {}
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # Compute loss
        noise_pred = self.forward_dit_with_conditions(
            noisy_latents=noisy_latents,
            timestep=timestep,
            prompt_emb=prompt_emb,
            image_emb=image_emb,
            condition_tokens=condition,
            object_tokens=object_tokens,
            object_grid=object_grid,
            camera_tokens=camera_tokens,
        )
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        # Record log
        self.log("train_loss_avg", loss, prog_bar=True, sync_dist=True)
        if self.trainer is not None:
            rank_id = getattr(self.trainer, "global_rank", 0)
            self.log(f"train_loss_rank{rank_id}", loss, sync_dist=False)
        return loss

    def build_freqs(self, model, grid_size, offset=(0, 0, 0), device=None):
        f, h, w = grid_size
        of, oh, ow = offset
        f_slice = slice(of, of + f)
        h_slice = slice(oh, oh + h)
        w_slice = slice(ow, ow + w)
        freqs = torch.cat(
            [
                model.freqs[0][f_slice].view(f, 1, 1, -1).expand(f, h, w, -1),
                model.freqs[1][h_slice].view(1, h, 1, -1).expand(f, h, w, -1),
                model.freqs[2][w_slice].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1)
        if device is not None:
            freqs = freqs.to(device)
        return freqs

    def forward_dit_with_conditions(
        self,
        noisy_latents,
        timestep,
        prompt_emb,
        image_emb,
        condition_tokens,
        object_tokens,
        object_grid,
        camera_tokens,
    ):
        model = self.pipe.denoising_model()
        context = prompt_emb["context"]
        t = model.time_embedding(sinusoidal_embedding_1d(model.freq_dim, timestep))
        t_mod = model.time_projection(t).unflatten(1, (6, model.dim))
        context = model.text_embedding(context)

        x = noisy_latents
        clip_feature = image_emb.get("clip_feature", None)
        image_y = image_emb.get("y", None)
        if model.has_image_input:
            if image_y is None or clip_feature is None:
                raise ValueError("Image-to-video conditioning requires both 'y' and 'clip_feature'.")
            image_y = image_y.to(dtype=x.dtype, device=x.device)
            x = torch.cat([x, image_y], dim=1)
            clip_embedding = model.img_emb(clip_feature.to(self.device))
            context = torch.cat([clip_embedding, context], dim=1)

        x, grid_size = model.patchify(x)
        main_token_len = x.shape[1]

        if condition_tokens is not None:
            x = x + condition_tokens

        freqs = self.build_freqs(model, grid_size, device=x.device)

        reference_len = 0
        if object_tokens is not None:
            if object_grid is None:
                raise ValueError("object_grid must be provided when object_tokens is not None.")
            ref_freqs = self.build_freqs(
                model, object_grid, offset=self.reference_rope_offset, device=x.device
            )
            x = torch.cat([x, object_tokens], dim=1)
            freqs = torch.cat([freqs, ref_freqs], dim=0)
            reference_len = object_tokens.shape[1]

        camera_full = None
        if camera_tokens is not None:
            if camera_tokens.shape[1] != main_token_len:
                raise ValueError("camera_tokens must match the number of main tokens.")
            if reference_len > 0:
                pad = torch.zeros(
                    camera_tokens.size(0),
                    reference_len,
                    camera_tokens.size(2),
                    dtype=camera_tokens.dtype,
                    device=camera_tokens.device,
                )
                camera_full = torch.cat([camera_tokens, pad], dim=1)
            else:
                camera_full = camera_tokens

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for idx, block in enumerate(model.blocks):
            if camera_full is not None and idx < self.camera_injection_depth:
                camera_layer = self.camera_residual_layers[idx]
                camera_dtype = camera_layer.weight.dtype
                camera_input = camera_full.to(dtype=camera_dtype)
                injected = camera_layer(camera_input)
                x = x + injected.to(dtype=x.dtype)
            if self.training and self.use_gradient_checkpointing:
                if self.use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            context,
                            t_mod,
                            freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        context,
                        t_mod,
                        freqs,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs)

        x_main = x[:, :main_token_len]
        x_main = model.head(x_main, t)
        x_main = model.unpatchify(x_main, grid_size)
        return x_main


    def configure_optimizers(self):
        # trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        # optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        # return optimizer
        trainable_modules = [
            {'params': filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())},
            {'params': self.dwpose_embedding.parameters()},
            {'params': self.randomref_embedding_pose.parameters()},
            {'params': self.object_patchifier.parameters()},
            {'params': self.camera_encoder.parameters()},
            {'params': self.camera_residual_layers.parameters()},
        ]
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        # trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters())) + \
        #                         list(filter(lambda named_param: named_param[1].requires_grad, self.dwpose_embedding.named_parameters())) + \
        #                         list(filter(lambda named_param: named_param[1].requires_grad, self.randomref_embedding_pose.named_parameters()))
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters())) 
        
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        # state_dict = self.pipe.denoising_model().state_dict()
        state_dict = self.state_dict()
        # state_dict.update()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        checkpoint.update(lora_state_dict)
