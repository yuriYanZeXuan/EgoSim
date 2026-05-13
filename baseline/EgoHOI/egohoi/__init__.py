__all__ = [
    "LightningModelForDataProcess",
    "LightningModelForTrain_onestage",
    "TextVideoDataset",
    "TextVideoDataset_onestage",
    "debug_tensor_stats",
    "get_camera_embedding",
    "load_checkpoint_state",
    "load_clip_sample",
    "prepare_camera_tokens",
    "prepare_object_tokens",
    "prepare_pose_condition",
    "prepare_random_ref_pose",
    "select_frame_indices",
    "set_seed",
]


def __getattr__(name):
    if name == "get_camera_embedding":
        from .camera import get_camera_embedding

        return get_camera_embedding

    if name in {"TextVideoDataset", "TextVideoDataset_onestage"}:
        from .dataset import TextVideoDataset, TextVideoDataset_onestage

        return {
            "TextVideoDataset": TextVideoDataset,
            "TextVideoDataset_onestage": TextVideoDataset_onestage,
        }[name]

    if name in {"LightningModelForDataProcess", "LightningModelForTrain_onestage"}:
        from .model import LightningModelForDataProcess, LightningModelForTrain_onestage

        return {
            "LightningModelForDataProcess": LightningModelForDataProcess,
            "LightningModelForTrain_onestage": LightningModelForTrain_onestage,
        }[name]

    if name in {
        "debug_tensor_stats",
        "load_checkpoint_state",
        "load_clip_sample",
        "prepare_camera_tokens",
        "prepare_object_tokens",
        "prepare_pose_condition",
        "prepare_random_ref_pose",
        "select_frame_indices",
        "set_seed",
    }:
        from .inference import (
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

        return {
            "debug_tensor_stats": debug_tensor_stats,
            "load_checkpoint_state": load_checkpoint_state,
            "load_clip_sample": load_clip_sample,
            "prepare_camera_tokens": prepare_camera_tokens,
            "prepare_object_tokens": prepare_object_tokens,
            "prepare_pose_condition": prepare_pose_condition,
            "prepare_random_ref_pose": prepare_random_ref_pose,
            "select_frame_indices": select_frame_indices,
            "set_seed": set_seed,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
