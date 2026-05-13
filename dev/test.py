import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download
dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/examples_in_diffsynth",
    local_dir="./",
    allow_file_pattern=f"data/examples/wan/input_image.jpg"
)
image = Image.open("data/examples/wan/input_image.jpg")