"""
Continuous generation dataset sample loader.

CSV columns: video, ego_prior_video, hand_keypoint_video, first_frame, prompt, task_name, part_idx, process_result_dir, hdf5_path, gt_process_result_dir
"""
import pandas as pd
from pathlib import Path
from .schema import EgoSample


def load_samples(metadata_path: str, eval_set_path: str | None = None) -> list[EgoSample]:
    df = pd.read_csv(metadata_path)
    samples = []
    for _, row in df.iterrows():
        video = str(row.get("video", ""))
        video_id = video.replace("/", "_").replace(".mp4", "")
        samples.append(EgoSample(
            video_id=video_id,
            output_id=video_id,
            prompt=str(row.get("prompt", "")),
            dataset="continuous_generation",
            ego_prior_video=str(row.get("ego_prior_video", "")),
            hand_keypoint_video=str(row.get("hand_keypoint_video", "")),
            first_frame=str(row.get("first_frame", "")),
        ))
    return samples


def get_mask_path(dataset_root: str, sample: EgoSample) -> Path:
    ego_prior = sample.ego_prior_video or ""
    if ego_prior:
        return Path(dataset_root) / Path(ego_prior).parent / "pc_mask_video.mp4"
    return Path(dataset_root) / sample.video_id / "pc_mask_video.mp4"


def get_ego_prior_video_path(dataset_root: str, sample: EgoSample) -> Path:
    ego_prior = sample.ego_prior_video or ""
    if ego_prior:
        return Path(dataset_root) / ego_prior
    raise ValueError(f"No ego_prior_video for sample {sample.video_id}")


def get_hand_video_path(dataset_root: str, sample: EgoSample) -> Path:
    hand = sample.hand_keypoint_video or ""
    if hand:
        return Path(dataset_root) / hand
    ego_prior = sample.ego_prior_video or ""
    if ego_prior:
        return Path(dataset_root) / Path(ego_prior).parent / "skeleton_3d.mp4"
    raise ValueError(f"No hand_keypoint_video for sample {sample.video_id}")


def get_first_frame_path(dataset_root: str, sample: EgoSample) -> Path:
    ff = sample.first_frame or ""
    if ff:
        return Path(dataset_root) / ff
    ego_prior = sample.ego_prior_video or ""
    if ego_prior:
        return Path(dataset_root) / Path(ego_prior).parent / "hand_inpaint.png"
    raise ValueError(f"No first_frame for sample {sample.video_id}")
