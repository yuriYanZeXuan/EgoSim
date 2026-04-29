# Copyright (c) jiamingda (https://github.com/Luyitas)

"""
Take the first frame of each input mp4 clip, segment hands with SAM3, save masks as PNG.

Example paths (replace roots with your dataset layout):
  <dataset_root>/<part>/<action>/<clip>.mp4
  <output_root>/<part>/<action>/<clip>/hand_seg.png
"""

import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import sys
import argparse
from tqdm import tqdm
import os

# SAM3 sources: export SAM3_ROOT=/path/to/sam3 (directory containing the `sam3` package)
_sam3_root = os.environ.get("SAM3_ROOT", "").strip()
if _sam3_root:
    sys.path.insert(0, os.path.abspath(_sam3_root))

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as exc:
    raise SystemExit(
        "Cannot import sam3. Clone the SAM3 repo and set SAM3_ROOT to its root "
        "(directory that contains the `sam3` Python package), e.g.\n"
        "  export SAM3_ROOT=\"${REPOS_DIR}/sam3\"\n"
        f"Original error: {exc}"
    ) from exc


def extract_first_frame(video_path):
    """Read the first frame from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Cannot read first frame: {video_path}")
    
    # BGR -> RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb


def segment_hand(image_np, processor, inference_state, score_threshold=0.5):
    """Segment hands with SAM3."""
    # Text prompt
    text_prompt = "hand and arm"
    output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)
    
    masks = output["masks"]
    scores = output["scores"]
    
    # Merge high-confidence masks
    combined_mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
    
    for mask, score in zip(masks, scores):
        if score > score_threshold:
            mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
            while mask_np.ndim > 2:
                mask_np = mask_np.squeeze(0)
            
            combined_mask[mask_np > 0] = 255
    
    return combined_mask


def process_single_video(video_path, output_mask_path, output_vis_path, processor, device):
    """Process one video file."""
    try:
        frame = extract_first_frame(video_path)
        image_pil = Image.fromarray(frame)
        inference_state = processor.set_image(image_pil)
        mask = segment_hand(frame, processor, inference_state)
        
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)
        output_vis_path.parent.mkdir(parents=True, exist_ok=True)
        
        Image.fromarray(mask).save(output_mask_path)
        
        # Blue overlay visualization (similar to test_hand.py)
        frame_float = frame.astype(np.float32)
        mask_bool = mask > 0
        alpha = 0.5
        mask_color = np.array([0, 0, 255], dtype=np.float32)  # blue (RGB)
        
        vis_img = frame_float.copy()
        vis_img[mask_bool] = vis_img[mask_bool] * (1 - alpha) + mask_color * alpha
        
        vis_img = vis_img.astype(np.uint8)
        Image.fromarray(vis_img).save(output_vis_path, quality=95)
        
        return True
    except Exception as e:
        print(f"Failed {video_path}: {str(e)}")
        return False


def process_dataset(input_dir, output_dir, checkpoint_path, device_id=0):
    """Process a dataset directory or single video file."""
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    
    print(f"[GPU {device_id}] Device: {device}", flush=True)
    
    print(f"[GPU {device_id}] Loading SAM3...", flush=True)
    model = build_sam3_image_model(checkpoint_path=checkpoint_path, load_from_HF=False)
    model = model.to(device)
    processor = Sam3Processor(model)
    print(f"[GPU {device_id}] Model ready.", flush=True)
    
    # Infer `part_name` from parent of input path
    input_path = Path(input_dir)
    file_mode = input_path.is_file()
    part_name = input_path.parent.name

    if file_mode:
        action_name = input_path.stem
        video_files = [input_path]
    else:
        action_name = input_path.name
        video_files = list(input_path.rglob("*.mp4"))

    print(f"[GPU {device_id}] Part: {part_name}, Action: {action_name}", flush=True)
    print(f"[GPU {device_id}] Found {len(video_files)} video(s)", flush=True)
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    with tqdm(total=len(video_files), desc=f"GPU{device_id}-{action_name[:15]}", 
              position=device_id, leave=True, 
              ncols=100, file=sys.stderr, dynamic_ncols=False) as pbar:
        
        for video_file in video_files:
            # <output_dir>/<part>/<action>/<video_name>/hand_seg.png
            video_name = video_file.stem
            
            if file_mode:
                output_mask_path = Path(output_dir) / part_name / action_name / "hand_seg.png"
                output_vis_path = Path(output_dir) / part_name / action_name / "hand_seg_vis.jpg"
            else:
                output_mask_path = Path(output_dir) / part_name / action_name / video_name / "hand_seg.png"
                output_vis_path = Path(output_dir) / part_name / action_name / video_name / "hand_seg_vis.jpg"
            
            # Always overwrite (do not skip existing outputs)
            if process_single_video(video_file, output_mask_path, output_vis_path, processor, device):
                success_count += 1
            else:
                fail_count += 1
            
            pbar.set_postfix({"✓": success_count, "⊙": skip_count, "✗": fail_count}, refresh=True)
            pbar.update(1)
    
    print(f"\n[GPU {device_id}] Done. ok={success_count}, skip={skip_count}, fail={fail_count}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAM3 hand segmentation on the first frame of each video')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input video file or directory containing .mp4 files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output root (hand_seg.png / hand_seg_vis.jpg)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to SAM3 weights (sam3.pt)')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU id (default: 0)')
    
    args = parser.parse_args()
    
    process_dataset(args.input_dir, args.output_dir, args.checkpoint, args.device)
