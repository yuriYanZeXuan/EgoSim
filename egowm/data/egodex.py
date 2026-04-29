"""
Egodex dataset sample loader.

CSV columns: video, ego_prior_video, prompt
  video: relative path like "test_egodex/some/ego_prior.mp4"
"""
import pandas as pd
from pathlib import Path
from .schema import EgoSample


def load_samples(metadata_path: str) -> list[EgoSample]:
    df = pd.read_csv(metadata_path)
    samples = []
    for _, row in df.iterrows():
        video = str(row["video"])
        video_id = video.replace("/", "_").replace(".mp4", "")
        parts = video.split("/", 1)
        output_id = (parts[1] if len(parts) > 1 else video).replace("/", "_").replace(".mp4", "")
        samples.append(EgoSample(
            video_id=video_id,
            output_id=output_id,
            prompt=str(row.get("prompt", "")),
            dataset="egodex",
            ego_prior_video=str(row.get("ego_prior_video", "")),
            hand_keypoint_video=str(row.get("hand_keypoint_video", "")),
            first_frame=str(row.get("first_frame", "")),
        ))
    return samples


def get_mask_path(dataset_root: str, sample: EgoSample) -> Path:
    """Return path to pc_mask_video.mp4 for this sample."""
    ego_prior = sample.ego_prior_video or ""
    if ego_prior:
        return Path(dataset_root) / Path(ego_prior).parent / "pc_mask_video.mp4"
    return Path(dataset_root) / sample.video_id / "pc_mask_video.mp4"


def get_ego_prior_video_path(dataset_root: str, sample: EgoSample) -> Path:
    """Return absolute path to ego prior video."""
    ego_prior = sample.ego_prior_video or ""
    if ego_prior:
        return Path(dataset_root) / ego_prior
    raise ValueError(f"No ego_prior_video for sample {sample.video_id}")


def get_hand_video_path(dataset_root: str, sample: EgoSample) -> Path:
    """Return absolute path to hand keypoint video."""
    hand = sample.hand_keypoint_video or ""
    if hand:
        return Path(dataset_root) / hand
    # fallback: same directory as ego prior, named hand_kp_video.mp4
    ego_prior = sample.ego_prior_video or ""
    if ego_prior:
        return Path(dataset_root) / Path(ego_prior).parent / "hand_kp_video.mp4"
    raise ValueError(f"No hand_keypoint_video for sample {sample.video_id}")


def get_first_frame_path(dataset_root: str, sample: EgoSample) -> Path:
    """Return absolute path to first frame image (for CLIP encoding)."""
    ff = sample.first_frame or ""
    if ff:
        return Path(dataset_root) / ff
    ego_prior = sample.ego_prior_video or ""
    if ego_prior:
        return Path(dataset_root) / Path(ego_prior).parent / "hand_inpaint.png"
    raise ValueError(f"No first_frame for sample {sample.video_id}")
