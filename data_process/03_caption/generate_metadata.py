# Copyright (c) jiamingda (https://github.com/Luyitas)
"""
Generate a metadata CSV for a single processed clip, ready for EgoSim inference.

Reads caption.txt from the clip directory and assembles the standard CSV columns:
    video, ego_prior_video, hand_keypoint_video, first_frame, prompt, video_id

The CSV is written to <output_path> (default: tests/samples/<clip_name>_metadata.csv).

Usage (called by run_step04_metadata.sh via VIDEO_PATH env var):
    python generate_metadata.py --clip_dir /path/to/clip [--output /path/to/out.csv]

Direct usage:
    python generate_metadata.py \
        --clip_dir tests/samples/my_clip \
        --output tests/samples/my_clip_metadata.csv
"""

import argparse
import csv
from pathlib import Path


REQUIRED_FILES = [
    "video.mp4",
    "rendered_scene.mp4",
    "pc_mask_video.mp4",
    "skeleton_3d.mp4",
    "hand_inpaint.png",
    "caption.txt",
]


def check_files(clip_dir: Path) -> list[str]:
    missing = [f for f in REQUIRED_FILES if not (clip_dir / f).exists()]
    return missing


def main():
    parser = argparse.ArgumentParser(description="Generate EgoSim metadata CSV for a single clip")
    parser.add_argument("--clip_dir", type=Path, required=True,
                        help="Processed clip directory (must contain all pipeline outputs)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV path (default: <clip_dir>/../<clip_name>_metadata.csv)")
    args = parser.parse_args()

    clip_dir = args.clip_dir.resolve()
    clip_name = clip_dir.name

    missing = check_files(clip_dir)
    if missing:
        raise FileNotFoundError(
            f"Missing required files in {clip_dir}:\n" + "\n".join(f"  {f}" for f in missing)
        )

    caption = (clip_dir / "caption.txt").read_text(encoding="utf-8").strip()
    if not caption:
        raise ValueError(f"caption.txt is empty: {clip_dir / 'caption.txt'}")

    output_path = args.output or clip_dir.parent / f"{clip_name}_metadata.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "video":               f"{clip_name}/video.mp4",
        "ego_prior_video":     f"{clip_name}/rendered_scene.mp4",
        "hand_keypoint_video": f"{clip_name}/skeleton_3d.mp4",
        "first_frame":         f"{clip_name}/hand_inpaint.png",
        "prompt":              caption,
        "video_id":            clip_name,
    }

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print(f"[Step 04] Metadata CSV written: {output_path}")
    print(f"[Step 04] prompt: {caption[:120]}{'...' if len(caption) > 120 else ''}")


if __name__ == "__main__":
    main()
