"""
EgoVid dataset sample loader.

eval_set.txt: one relative path per line, e.g. "abc_def_ghi/video.mp4" (optional)
CSV columns: video_id (or video), ego_prior_video, hand_keypoint_video, first_frame, prompt
"""
import pandas as pd
from pathlib import Path
from .schema import EgoSample


def load_samples(metadata_path: str, eval_set_path: str | None = None) -> list[EgoSample]:
    df = pd.read_csv(metadata_path)
    if "video_id" not in df.columns:
        if "video" in df.columns:
            df["video_id"] = df["video"].apply(
                lambda x: x.replace("/video.mp4", "").strip() if isinstance(x, str) else ""
            )
        else:
            raise ValueError("EgoVid metadata CSV must have 'video_id' or 'video' column")

    video_to_meta = {str(row["video_id"]).strip(): row.to_dict()
                     for _, row in df.iterrows() if str(row["video_id"]).strip()}

    if eval_set_path is not None:
        with open(eval_set_path) as f:
            eval_paths = [l.strip() for l in f if l.strip()]
        video_ids = []
        for vp in eval_paths:
            vid = vp.split("/")[0] if "/" in vp else vp.replace(".mp4", "")
            video_ids.append(vid)
    else:
        video_ids = list(video_to_meta.keys())

    samples = []
    for video_id in video_ids:
        meta = video_to_meta.get(video_id)
        if meta is None:
            print(f"  No metadata for video_id={video_id}, skipping")
            continue
        prompt = str(meta.get("prompt", "")).strip()
        if not prompt:
            print(f"  Empty prompt for video_id={video_id}, skipping")
            continue
        samples.append(EgoSample(
            video_id=video_id,
            output_id=video_id.replace("/", "_"),
            prompt=prompt,
            dataset="egovid",
            ego_prior_video=str(meta.get("ego_prior_video", "")),
            hand_keypoint_video=str(meta.get("hand_keypoint_video", "")),
            first_frame=str(meta.get("first_frame", "")),
        ))
    return samples


def get_mask_path(dataset_root: str, sample: EgoSample) -> Path:
    return Path(dataset_root) / sample.video_id / "pc_mask_video.mp4"


def get_ego_prior_video_path(dataset_root: str, sample: EgoSample) -> Path:
    ego = sample.ego_prior_video or ""
    if ego:
        return Path(dataset_root) / ego
    # fallback: rendered_scene.mp4 is the standard EgoVid ego prior filename
    return Path(dataset_root) / sample.video_id / "rendered_scene.mp4"


def get_hand_video_path(dataset_root: str, sample: EgoSample) -> Path:
    hand = sample.hand_keypoint_video or ""
    if hand:
        return Path(dataset_root) / hand
    return Path(dataset_root) / sample.video_id / "skeleton_3d.mp4"


def get_first_frame_path(dataset_root: str, sample: EgoSample) -> Path:
    ff = sample.first_frame or ""
    if ff:
        return Path(dataset_root) / ff
    return Path(dataset_root) / sample.video_id / "hand_inpaint.png"
