"""
Agibot dataset sample loader.

CSV columns: video_id, ego_prior_video, prompt
"""
import pandas as pd
from pathlib import Path
from .schema import EgoSample


def load_samples(metadata_path: str) -> list[EgoSample]:
    df = pd.read_csv(metadata_path)
    samples = []
    for _, row in df.iterrows():
        video_id = str(row["video_id"]).strip()
        samples.append(EgoSample(
            video_id=video_id,
            output_id=video_id,
            prompt=str(row.get("prompt", "")).strip(),
            dataset="agibot",
            ego_prior_video=str(row.get("ego_prior_video", "")),
            hand_keypoint_video=str(row.get("hand_keypoint_video", "")),
            first_frame=str(row.get("first_frame", "")),
        ))
    return samples


def get_mask_path(dataset_root: str, sample: EgoSample) -> Path:
    return Path(dataset_root) / sample.video_id / "pc_mask_video.mp4"


def get_ego_prior_video_path(dataset_root: str, sample: EgoSample) -> Path:
    ego_prior = sample.ego_prior_video or ""
    if ego_prior:
        return Path(dataset_root) / ego_prior
    return Path(dataset_root) / sample.video_id / "ego_prior_video.mp4"


def get_hand_video_path(dataset_root: str, sample: EgoSample) -> Path:
    hand = sample.hand_keypoint_video or ""
    if hand:
        return Path(dataset_root) / hand
    return Path(dataset_root) / sample.video_id / "hand_kp_video.mp4"


def get_first_frame_path(dataset_root: str, sample: EgoSample) -> Path:
    ff = sample.first_frame or ""
    if ff:
        return Path(dataset_root) / ff
    return Path(dataset_root) / sample.video_id / "hand_inpaint.png"
