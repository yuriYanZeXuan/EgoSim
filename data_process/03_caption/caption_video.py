# Copyright (c) jiamingda (https://github.com/Luyitas)
"""
Use Qwen2.5-VL to generate a text caption for a single clip's video.mp4.
Writes the caption to <clip_dir>/caption.txt.

Usage (called by run_step03_caption.sh via VIDEO_PATH env var):
    python caption_video.py --clip_dir /path/to/clip --model_path /path/to/Qwen2.5-VL

Direct usage:
    python caption_video.py \
        --clip_dir tests/samples/my_clip \
        --model_path /path/to/Qwen2.5-VL-32B-Instruct \
        [--skip_existing] \
        [--dry_run]
"""

import argparse
from pathlib import Path

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


PROMPT = (
    "Please provide a detailed description of the video, focusing on the main subjects, "
    "their detailed actions, and the background scene."
)

MAX_NEW_TOKENS = 1024


def infer_caption(model, processor, video_path: Path) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path)},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL video captioning for a single clip")
    parser.add_argument("--clip_dir", type=Path, required=True,
                        help="Clip directory containing video.mp4 (output: caption.txt)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to Qwen2.5-VL model directory")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip if caption.txt already exists")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be processed without running inference")
    args = parser.parse_args()

    clip_dir = args.clip_dir
    video_path = clip_dir / "video.mp4"
    caption_path = clip_dir / "caption.txt"

    if not video_path.exists():
        raise FileNotFoundError(f"video.mp4 not found: {video_path}")

    if args.skip_existing and caption_path.exists():
        print(f"[Step 03] Skipping (caption.txt exists): {clip_dir.name}")
        return

    if args.dry_run:
        print(f"[Step 03] Would caption: {video_path}")
        return

    print(f"[Step 03] Loading model: {args.model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    print("[Step 03] Model ready.\n")

    print(f"[Step 03] Captioning: {video_path}")
    caption = infer_caption(model, processor, video_path)
    caption_path.write_text(caption, encoding="utf-8")
    print(f"[Step 03] Caption saved: {caption_path}")
    print(f"[Step 03] Caption: {caption}")


if __name__ == "__main__":
    main()
