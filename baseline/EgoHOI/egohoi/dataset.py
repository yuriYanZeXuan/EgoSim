import json
import os
import random
import sys

import imageio
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from einops import rearrange
from torchvision.transforms import InterpolationMode, v2
import torchvision.transforms.functional as TF

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(FILE_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from egohoi.camera import get_camera_embedding

class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832, is_i2v=False):
        metadata = pd.read_csv(metadata_path)
        self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        self.text = metadata["text"].to_list()
        
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
            
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image

    def resize(self, image):
        width, height = image.size
        # scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (self.height, self.width),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return torch.from_numpy(np.array(image))


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames


    def load_video(self, file_path):
        start_frame_id = torch.randint(0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame


    def __getitem__(self, data_id):
        text = self.text[data_id]
        path = self.path[data_id]
        if self.is_image(path):
            if self.is_i2v:
                raise ValueError(f"{path} is not a video. I2V model doesn't support image-to-image training.")
            video = self.load_image(path)
        else:
            video = self.load_video(path)
        if self.is_i2v:
            video, first_frame = video
            data = {"text": text, "video": video, "path": path, "first_frame": first_frame}
        else:
            data = {"text": text, "video": video, "path": path}
        return data
    

    def __len__(self):
        return len(self.path)





class TextVideoDataset_onestage(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path,
        metadata_path,
        max_num_frames=81,
        frame_interval=1,
        num_frames=81,
        height=480,
        width=480,
        is_i2v=False,
        steps_per_epoch=1,
    ):
        del metadata_path  # metadata no longer used in HOT3D setting

        self.base_path = base_path
        self.video_dir = os.path.join(base_path, "videos")
        self.pose_dir = os.path.join(base_path, "saved_pose")
        self.object_dir = os.path.join(base_path, "obj_mask")
        self.camera_dir = os.path.join(base_path, "camera_traj1")

        self.max_num_frames = max_num_frames
        self.frame_interval = max(frame_interval, 1)
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
        self.steps_per_epoch = steps_per_epoch

        self.sample_fps = self.frame_interval
        self.max_retries = 5

        self.video_transform = v2.Compose([
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.video_list = self._build_index()
        if not self.video_list:
            raise ValueError(f"No valid clips found under {base_path}.")
        self._clip_offsets = {clip["clip_id"]: 0 for clip in self.video_list}

        self.camera_cache = {}

    def _build_index(self):
        clips = []
        if not os.path.isdir(self.video_dir):
            return clips

        supported_video_ext = (".mp4", ".mov", ".avi", ".mkv")
        for file_name in sorted(os.listdir(self.video_dir)):
            if not file_name.lower().endswith(supported_video_ext):
                continue
            clip_id = os.path.splitext(file_name)[0]
            pose_dir = os.path.join(self.pose_dir, clip_id)
            object_dir = os.path.join(self.object_dir, clip_id)
            camera_path = os.path.join(self.camera_dir, f"{clip_id}.json")

            if not (os.path.isdir(pose_dir) and os.path.isdir(object_dir) and os.path.isfile(camera_path)):
                continue

            pose_frames = self._list_frames(pose_dir)
            object_frames = self._list_frames(object_dir)
            if not pose_frames or not object_frames:
                continue

            clips.append(
                {
                    "clip_id": clip_id,
                    "video_path": os.path.join(self.video_dir, file_name),
                    "pose_dir": pose_dir,
                    "object_dir": object_dir,
                    "pose_frames": pose_frames,
                    "object_frames": object_frames,
                    "camera_path": camera_path,
                }
            )

        random.shuffle(clips)
        return clips

    def _list_frames(self, directory):
        supported_ext = (".jpg", ".jpeg", ".png", ".webp")
        return sorted([file for file in os.listdir(directory) if file.lower().endswith(supported_ext)])

    def __len__(self):
        return len(self.video_list)

    def _load_camera_meta(self, clip_info):
        clip_id = clip_info["clip_id"]
        if clip_id not in self.camera_cache:
            with open(clip_info["camera_path"], "r") as fp:
                self.camera_cache[clip_id] = json.load(fp)
        return self.camera_cache[clip_id]

    def _resize_image(self, image, interpolation):
        antialias = interpolation in {InterpolationMode.BILINEAR, InterpolationMode.BICUBIC}
        return TF.resize(image, [self.height, self.width], interpolation=interpolation, antialias=antialias)

    def _load_image_tensor(self, path, interpolation, channel_last=False):
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = self._resize_image(img, interpolation)
            array = np.array(img, dtype=np.uint8)
        tensor = torch.from_numpy(array)
        if channel_last:
            return tensor
        return tensor.permute(2, 0, 1)

    def _load_image_sequence(self, directory, filenames, indices, interpolation):
        frames = []
        limit = len(filenames) - 1
        for idx in indices:
            file_idx = min(idx, limit)
            img_path = os.path.join(directory, filenames[file_idx])
            frames.append(self._load_image_tensor(img_path, interpolation, channel_last=False))
        stacked = torch.stack(frames, dim=0)  # [F, C, H, W]
        return stacked.permute(1, 0, 2, 3).contiguous()  # [C, F, H, W]

    def _resize_intrinsic(self, intrinsic, original_height, original_width):
        scale_x = self.width / float(original_width)
        scale_y = self.height / float(original_height)
        scale_mat = np.array(
            [
                [scale_x, 0.0, 0.0],
                [0.0, scale_y, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return scale_mat @ intrinsic

    def _select_frame_indices(self, clip_id, total_frames):
        stride = self.frame_interval
        if total_frames <= 0:
            raise ValueError("Video contains no frames.")

        start = self._clip_offsets.get(clip_id, 0) % total_frames
        indices = [(start + i * stride) % total_frames for i in range(self.num_frames)]
        self._clip_offsets[clip_id] = (start + stride) % total_frames
        return indices

    def _load_video_window(self, clip_info, frame_indices, ref_index):
        reader = imageio.get_reader(clip_info["video_path"])
        video_frames = []
        try:
            for idx in frame_indices:
                frame = reader.get_data(idx)
                frame = Image.fromarray(frame).convert("RGB")
                video_frames.append(self.video_transform(frame))

            ref_frame = reader.get_data(ref_index)
            ref_frame = Image.fromarray(ref_frame).convert("RGB")
            ref_frame = self._resize_image(ref_frame, InterpolationMode.BILINEAR)
            ref_frame_tensor = torch.from_numpy(np.array(ref_frame, dtype=np.uint8))
        finally:
            reader.close()

        video_tensor = torch.stack(video_frames, dim=0).permute(1, 0, 2, 3).contiguous()
        return video_tensor, ref_frame_tensor

    def _compute_camera_embedding(self, camera_meta, frame_indices):
        intrinsics_raw = camera_meta["intrinsics"]
        extrinsics_raw = camera_meta["extrinsics"]
        total_intrinsics = len(intrinsics_raw)
        total_extrinsics = len(extrinsics_raw)
        original_width = camera_meta.get("image_width", self.width)
        original_height = camera_meta.get("image_height", self.height)

        intrinsics = []
        extrinsics = []
        for idx in frame_indices:
            src_idx = min(idx, total_intrinsics - 1)
            intrinsic = np.asarray(intrinsics_raw[src_idx], dtype=np.float32)
            intrinsic = self._resize_intrinsic(intrinsic, original_height, original_width)
            intrinsics.append(intrinsic)

            src_idx = min(idx, total_extrinsics - 1)
            extrinsic = np.asarray(extrinsics_raw[src_idx], dtype=np.float32)
            extrinsics.append(extrinsic)

        intrinsic_tensor = torch.from_numpy(np.stack(intrinsics, axis=0))
        extrinsic_tensor = torch.from_numpy(np.stack(extrinsics, axis=0))
        camera_embedding = get_camera_embedding(
            intrinsic_tensor, extrinsic_tensor, f=len(frame_indices), h=self.height, w=self.width
        )
        return camera_embedding.squeeze(0).contiguous().to(torch.float32)  # [C, F, H, W]

    def __getitem__(self, index):
        index = index % len(self.video_list)
        last_error = None

        for _ in range(self.max_retries):
            clip_info = self.video_list[index]
            try:
                camera_meta = self._load_camera_meta(clip_info)
                total_frames = min(
                    len(camera_meta.get("frames", [])),
                    len(clip_info["pose_frames"]),
                    len(clip_info["object_frames"]),
                )
                if total_frames <= 0:
                    raise ValueError(f"Clip {clip_info['clip_id']} contains no usable frames.")

                frame_indices = self._select_frame_indices(clip_info["clip_id"], total_frames)
                ref_index = frame_indices[0]

                video, first_frame = self._load_video_window(clip_info, frame_indices, ref_index)
                dwpose_data = self._load_image_sequence(
                    clip_info["pose_dir"], clip_info["pose_frames"], frame_indices, InterpolationMode.BILINEAR
                )
                object_data = self._load_image_sequence(
                    clip_info["object_dir"], clip_info["object_frames"], frame_indices, InterpolationMode.NEAREST
                )

                random_ref_dwpose = self._load_image_tensor(
                    os.path.join(
                        clip_info["pose_dir"],
                        clip_info["pose_frames"][min(ref_index, len(clip_info["pose_frames"]) - 1)],
                    ),
                    InterpolationMode.BILINEAR,
                    channel_last=True,
                )
                random_ref_object = self._load_image_tensor(
                    os.path.join(
                        clip_info["object_dir"],
                        clip_info["object_frames"][min(ref_index, len(clip_info["object_frames"]) - 1)],
                    ),
                    InterpolationMode.NEAREST,
                    channel_last=True,
                )

                camera_embedding = self._compute_camera_embedding(camera_meta, frame_indices)

                caption = "An egocentric video showing human hands interacting with everyday objects in indoor environments."
                data = {
                    "text": caption,
                    "video": video,
                    "path": clip_info["video_path"],
                    "dwpose_data": dwpose_data,
                    "object_data": object_data,
                    "camera_embedding": camera_embedding,
                    "random_ref_dwpose_data": random_ref_dwpose,
                    "random_ref_object_data": random_ref_object,
                }

                if self.is_i2v:
                    data["first_frame"] = first_frame

                return data

            except Exception as exc:
                last_error = exc
                print(f"[WARN] Failed to load clip {clip_info['clip_id']}: {exc}")
                index = random.randint(0, len(self.video_list) - 1)

        raise RuntimeError(f"Unable to load data sample after {self.max_retries} retries. Last error: {last_error}")

__all__ = ["TextVideoDataset", "TextVideoDataset_onestage"]
