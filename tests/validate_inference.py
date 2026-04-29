#!/usr/bin/env python3
"""
Validation script for egosim-opensource inference pipeline.

  Layer 1: Unit tests (no GPU)
  Layer 2: Data loading (no GPU)
  Layer 3: Pipeline loading (GPU required)

Usage:
    cd egosim-opensource
    MODEL_ROOT=/path/to/EgoSim-14B python tests/validate_inference.py
"""
import sys, os

# Allow running from repo root: python tests/validate_inference.py
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

MODEL_ROOT = os.environ.get(
    "MODEL_ROOT",
    os.path.join(os.path.dirname(_REPO_ROOT), "EgoSim-14B"),
)
FIXTURES = os.path.join(_REPO_ROOT, "tests", "samples", "mini_sample")

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ============================================================
# Layer 1: Unit tests
# ============================================================
section("Layer 1: Unit tests (no GPU)")

# 1a. EgoSample schema
try:
    from egowm.data.schema import EgoSample
    s = EgoSample(video_id="v001", output_id="v001", prompt="pick up the cup", dataset="egodex")
    assert s.ego_prior_video is None
    assert s.hand_keypoint_video is None
    print(f"{PASS} EgoSample schema")
except Exception as e:
    print(f"{FAIL} EgoSample schema: {e}")

# 1b. encode_mask_to_latent shape
try:
    import torch
    from egowm.inference.pipeline import encode_mask_to_latent
    mask = torch.zeros(1, 61, 480, 832)
    out = encode_mask_to_latent(mask, (16, 15, 60, 104))
    assert out.shape == (4, 15, 60, 104), f"got {out.shape}"
    print(f"{PASS} encode_mask_to_latent shape: {out.shape}")
except Exception as e:
    print(f"{FAIL} encode_mask_to_latent: {e}")

# 1c. encode_mask_to_latent values
try:
    import torch
    from egowm.inference.pipeline import encode_mask_to_latent
    mask = torch.ones(1, 61, 480, 832)
    out = encode_mask_to_latent(mask, (16, 15, 60, 104))
    assert out.min() >= 0.0 and out.max() <= 1.0
    print(f"{PASS} encode_mask_to_latent values in [0,1]")
except Exception as e:
    print(f"{FAIL} encode_mask_to_latent values: {e}")

# ============================================================
# Layer 2: Data loading
# ============================================================
section("Layer 2: Data loading (no GPU)")

# 2a. egodex loader
try:
    from egowm.data import egodex
    samples = egodex.load_samples(f"{FIXTURES}/egodex_metadata.csv")
    assert len(samples) == 2
    s = samples[0]
    assert s.dataset == "egodex"
    assert s.ego_prior_video != ""
    assert s.hand_keypoint_video != ""
    assert s.first_frame != ""
    print(f"{PASS} egodex.load_samples: {len(samples)} samples, video_id={s.video_id}")
    egodex_root = f"{FIXTURES}/egodex"
    p = egodex.get_ego_prior_video_path(egodex_root, s)
    h = egodex.get_hand_video_path(egodex_root, s)
    m = egodex.get_mask_path(egodex_root, s)
    f_ = egodex.get_first_frame_path(egodex_root, s)
    assert p.exists(), f"ego_prior not found: {p}"
    assert h.exists(), f"hand_video not found: {h}"
    assert m.exists(), f"mask not found: {m}"
    assert f_.exists(), f"first_frame not found: {f_}"
    print(f"       ego_prior:   {p.name}")
    print(f"       hand_video:  {h.name}")
    print(f"       mask:        {m.name}")
    print(f"       first_frame: {f_.name}")
except Exception as e:
    print(f"{FAIL} egodex loader: {e}")
    import traceback; traceback.print_exc()

# 2b. egovid loader
try:
    from egowm.data import egovid
    samples = egovid.load_samples(
        f"{FIXTURES}/egovid_metadata.csv",
    )
    assert len(samples) == 2
    s = samples[0]
    assert s.dataset == "egovid"
    assert s.ego_prior_video != ""
    assert s.hand_keypoint_video != ""
    assert s.first_frame != ""
    print(f"{PASS} egovid.load_samples: {len(samples)} samples, video_id={s.video_id}")
    egovid_root = f"{FIXTURES}/egovid"
    p = egovid.get_ego_prior_video_path(egovid_root, s)
    h = egovid.get_hand_video_path(egovid_root, s)
    m = egovid.get_mask_path(egovid_root, s)
    f_ = egovid.get_first_frame_path(egovid_root, s)
    assert p.exists(), f"ego_prior not found: {p}"
    assert h.exists(), f"hand_video not found: {h}"
    assert m.exists(), f"mask not found: {m}"
    assert f_.exists(), f"first_frame not found: {f_}"
    print(f"       ego_prior:   {p.name}")
    print(f"       hand_video:  {h.name}")
    print(f"       mask:        {m.name}")
    print(f"       first_frame: {f_.name}")
except Exception as e:
    print(f"{FAIL} egovid loader: {e}")
    import traceback; traceback.print_exc()

# 2c. agibot loader
try:
    from egowm.data import agibot
    samples = agibot.load_samples(f"{FIXTURES}/agibot_metadata.csv")
    assert len(samples) == 2
    s = samples[0]
    assert s.dataset == "agibot"
    print(f"{PASS} agibot.load_samples: {len(samples)} samples, video_id={s.video_id}")
except Exception as e:
    print(f"{FAIL} agibot loader: {e}")
    import traceback; traceback.print_exc()

# ============================================================
# Layer 3: Pipeline loading (GPU required)
# ============================================================
section("Layer 3: Pipeline loading (GPU required)")

try:
    import torch
    if not torch.cuda.is_available():
        print(f"  [SKIP] No CUDA available")
    else:
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  Loading pipeline from: {MODEL_ROOT}")
        from egowm.inference.pipeline import load_pipeline
        pipe = load_pipeline(MODEL_ROOT)
        print(f"{PASS} Pipeline loaded")

        # Check DiT in_dim
        in_dim = pipe.dit.in_dim if hasattr(pipe.dit, 'in_dim') else pipe.dit.patch_embedding.weight.shape[1]
        print(f"  DiT in_dim: {in_dim}  (expected 52)")
        assert in_dim == 52, f"Expected in_dim=52, got {in_dim}"
        print(f"{PASS} DiT in_dim == 52")

        # Check components
        print(f"  VAE:          {type(pipe.vae).__name__}")
        print(f"  text_encoder: {type(pipe.text_encoder).__name__}")
        print(f"  image_encoder:{type(pipe.image_encoder).__name__}")
        print(f"  scheduler:    {type(pipe.scheduler).__name__}")
        print(f"  torch_dtype:  {pipe.torch_dtype}")

        assert pipe.vae is not None,          "pipe.vae is None"
        assert pipe.text_encoder is not None, "pipe.text_encoder is None"
        assert pipe.image_encoder is not None,"pipe.image_encoder is None"
        print(f"{PASS} All pipeline components present")

except Exception as e:
    print(f"{FAIL} Pipeline loading: {e}")
    import traceback; traceback.print_exc()

section("Done")
