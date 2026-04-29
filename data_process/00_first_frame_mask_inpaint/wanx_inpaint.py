# Copyright (c) jiamingda (https://github.com/Luyitas)

"""
Inpaint regions marked by the blue mask using Qwen-Image-Edit.
Reads hand_seg_vis.jpg per clip and writes hand_inpaint.png.

Input:  <input_base>/<video_name>/hand_seg_vis.jpg
Output: <output_base>/<video_name>/hand_inpaint.png
"""

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import sys
from tqdm import tqdm

from diffusers import QwenImageEditPlusPipeline


def is_image_all_black(image_path, threshold=1):
    try:
        img = Image.open(image_path).convert('L')
        img_array = np.array(img)
        return np.all(img_array <= threshold)
    except Exception as e:
        print(f"Image check failed {image_path}: {str(e)}", flush=True)
        return False


def load_log_set(log_file):
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def append_log(log_file, value):
    with open(log_file, 'a') as f:
        f.write(f"{value}\n")


def process_single_video(input_vis_path, output_inpaint_path, pipeline):
    """Run inpainting on one visualization image."""
    try:
        if not input_vis_path.exists():
            return False, "Input file does not exist"

        image = Image.open(input_vis_path).convert("RGB")

        prompt = "Discard and remove the hand and arms according to the red masks of this image, keep the other part the same."

        # Under cpu offload, generator must live on CPU
        generator = torch.Generator(device="cpu").manual_seed(42)

        with torch.inference_mode():
            output = pipeline(
                image=[image],
                prompt=prompt,
                generator=generator,
                true_cfg_scale=6.0,
                negative_prompt="hands, fingers, arms, human body parts, skin, person, red",
                num_inference_steps=20,
                guidance_scale=3.5,
                num_images_per_prompt=1,
            )
            output_image = output.images[0]

        output_inpaint_path.parent.mkdir(parents=True, exist_ok=True)
        output_image.save(output_inpaint_path)

        del output, image, generator
        torch.cuda.empty_cache()

        return True, "ok"
    except Exception as e:
        torch.cuda.empty_cache()
        return False, str(e)


def process_dataset(video_list_file, input_base, output_base, model_path, device_id=0, log_dir="./logs"):
    """Process all clip names listed in video_list_file."""

    print(f"[GPU {device_id}] Loading Qwen-Image-Edit...", flush=True)

    # Official pipeline + cpu offload: ~30GB bf16; true_cfg_scale>1 runs CFG twice
    # Peak VRAM can exceed 48GB without offload
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    # enable_model_cpu_offload: move submodule to GPU only while needed
    # With CUDA_VISIBLE_DEVICES set, use logical gpu_id=0
    pipeline.enable_model_cpu_offload(gpu_id=0)
    pipeline.set_progress_bar_config(disable=True)
    print(f"[GPU {device_id}] Model ready.", flush=True)

    with open(video_list_file, 'r') as f:
        video_names = [line.strip() for line in f if line.strip()]

    print(f"[GPU {device_id}] Clips to process: {len(video_names)}", flush=True)

    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    processed_log = log_dir_path / f"processed_folders_gpu{device_id}.txt"
    no_hand_log = log_dir_path / f"no_hand_folders_gpu{device_id}.txt"

    processed_folders = load_log_set(processed_log)
    no_hand_folders = load_log_set(no_hand_log)

    print(f"[GPU {device_id}] Already processed: {len(processed_folders)}, no-hand logged: {len(no_hand_folders)}", flush=True)

    success_count = 0
    fail_count = 0
    skip_count = 0
    no_hand_count = 0

    input_base_path = Path(input_base)
    output_base_path = Path(output_base)

    with tqdm(total=len(video_names), desc=f"GPU{device_id}",
              position=0, leave=True, ncols=100, file=sys.stderr) as pbar:

        for video_name in video_names:
            vis_file = input_base_path / video_name / "hand_seg_vis.jpg"

            if not vis_file.exists():
                fail_count += 1
                print(f"\n[GPU {device_id}] ✗ Missing file: {vis_file}", flush=True)
                pbar.set_postfix({"✓": success_count, "⊙": skip_count, "✗": fail_count, "∅": no_hand_count})
                pbar.update(1)
                continue

            is_no_hand = False
            hand_seg_path = input_base_path / video_name / "hand_seg.png"
            if hand_seg_path.exists() and is_image_all_black(hand_seg_path):
                is_no_hand = True
                if video_name not in no_hand_folders:
                    append_log(no_hand_log, video_name)
                    no_hand_folders.add(video_name)
                no_hand_count += 1

            output_inpaint_path = output_base_path / video_name / "hand_inpaint.png"

            success, msg = process_single_video(vis_file, output_inpaint_path, pipeline)

            if success:
                success_count += 1
                append_log(processed_log, video_name)
                processed_folders.add(video_name)
                tag = " [no-hand]" if is_no_hand else ""
                print(f"\n[GPU {device_id}] ✓{tag}: {video_name}", flush=True)
            else:
                fail_count += 1
                tag = " [no-hand]" if is_no_hand else ""
                print(f"\n[GPU {device_id}] ✗{tag} {video_name}: {msg}", flush=True)

            pbar.set_postfix({"✓": success_count, "⊙": skip_count, "✗": fail_count, "∅": no_hand_count})
            pbar.update(1)

    print(f"\n[GPU {device_id}] Done. ok={success_count}, skip={skip_count}, fail={fail_count}, no-hand={no_hand_count}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_list', type=str, required=True)
    parser.add_argument('--input_base', type=str, required=True)
    parser.add_argument('--output_base', type=str, required=True,
                        help='Inpaint output root (one subdir per clip)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Local Qwen-Image-Edit-2511 directory (from_pretrained)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='./inpaint_logs')
    args = parser.parse_args()

    process_dataset(args.video_list, args.input_base, args.output_base, args.model_path, args.device, args.log_dir)
