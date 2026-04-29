# Copyright (c) jiamingda (https://github.com/Luyitas)

"""
Reconstruct full MANO (vertices 778×3, keypoints_3d 21×3) from lightweight JSON
(global_orient / hand_pose / betas) and write merged JSON files.

Input layout (same as lightweight HaMeR export):
  clips_16fps_mano/<uuid>/<clip_name>.json

Output layout (same depth):
  clips_16fps_mano_full/<uuid>/<clip_name>.json

Usage:
  python reconstruct_full_mano.py \
      --input_dir  .../ego_demo/clips_16fps_mano \
      --output_dir .../ego_demo/clips_16fps_mano_full \
      [--mano_path /path/to/mano] \
      [--device cuda] \
      [--no_resume]
"""
import sys
from unittest.mock import MagicMock

# Mock pyrender / OpenGL for headless
sys.modules['pyrender'] = MagicMock()
sys.modules['OpenGL'] = MagicMock()
sys.modules['OpenGL.GL'] = MagicMock()
sys.modules['OpenGL.error'] = MagicMock()
sys.modules['OpenGL.platform'] = MagicMock()

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from hamer.models import MANO


# ─────────────────────────── helpers ────────────────────────────

def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)  # compact JSON (no indent) to save space


def to_tensor(x, device):
    if isinstance(x, list):
        x = np.array(x, dtype=np.float32)
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to(device).float()


def reconstruct_one_hand(hand: dict, mano_model, device) -> dict:
    """Rebuild vertices and keypoints_3d from lightweight MANO params."""
    global_orient = to_tensor(hand['global_orient'], device)  # (1,3,3) or (3,3)
    hand_pose     = to_tensor(hand['hand_pose'],     device)  # (15,3,3)
    betas         = to_tensor(hand['betas'],         device)  # (10,)

    # Add batch dims as needed
    if global_orient.dim() == 2:
        global_orient = global_orient.unsqueeze(0).unsqueeze(0)   # (1,1,3,3)
    elif global_orient.dim() == 3:
        global_orient = global_orient.unsqueeze(0)                 # (1,1,3,3)
    if hand_pose.dim() == 3:
        hand_pose = hand_pose.unsqueeze(0)                         # (1,15,3,3)
    if betas.dim() == 1:
        betas = betas.unsqueeze(0)                                 # (1,10)

    with torch.no_grad():
        mano_out = mano_model(
            global_orient=global_orient,
            hand_pose=hand_pose,
            betas=betas,
            pose2rot=False,
        )

    return {
        'vertices':      mano_out.vertices.squeeze(0).cpu().numpy().tolist(),   # (778,3)
        'keypoints_3d':  mano_out.joints.squeeze(0).cpu().numpy().tolist(),     # (21,3)
    }


# ─────────────────────────── main logic ─────────────────────────

def process_json(src: Path, dst: Path, mano_model, device: str) -> int:
    """Process one JSON file; return number of hands updated."""
    data = load_json(src)

    total_hands = 0
    for frame in data.get('frames', []):
        for hand in frame.get('hands', []):
            recon = reconstruct_one_hand(hand, mano_model, device)
            hand['vertices']     = recon['vertices']
            hand['keypoints_3d'] = recon['keypoints_3d']
            total_hands += 1

    save_json(data, dst)
    return total_hands


def main():
    parser = argparse.ArgumentParser(
        description='Reconstruct full MANO (vertices + keypoints_3d) from lightweight annotations'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing lightweight MANO JSON files (recursive)',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for full MANO JSON files',
    )
    parser.add_argument(
        '--mano_path',
        type=str,
        required=True,
        help='Path to MANO model directory',
    )
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no_resume', action='store_true',
                        help='Re-process all files even if output already exists')
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    device     = args.device if torch.cuda.is_available() else 'cpu'

    print('=' * 60)
    print('Reconstruct Full MANO from Lightweight Annotations')
    print('=' * 60)
    print(f'Input  : {input_dir}')
    print(f'Output : {output_dir}')
    print(f'MANO   : {args.mano_path}')
    print(f'Device : {device}')
    print('=' * 60)

    print('Loading MANO model...')
    mano_model = MANO(
        model_path=args.mano_path,
        gender='neutral',
        num_hand_joints=15,
    ).to(device)
    mano_model.eval()
    print('MANO model loaded.')

    all_jsons = sorted(input_dir.rglob('*.json'))
    print(f'Found {len(all_jsons)} JSON files.')

    if args.no_resume:
        pending = all_jsons
    else:
        pending = []
        for src in all_jsons:
            rel = src.relative_to(input_dir)
            if not (output_dir / rel).exists():
                pending.append(src)
        print(f'Pending (output missing): {len(pending)}')

    total_files = 0
    total_hands = 0
    failed = 0

    for src in tqdm(pending, desc='Reconstructing'):
        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        try:
            n = process_json(src, dst, mano_model, device)
            total_hands += n
            total_files += 1
        except Exception as e:
            failed += 1
            print(f'\n  ✗ Failed {rel}: {e}')

    print('\n' + '=' * 60)
    print(f'Done.')
    print(f'  Files processed : {total_files}')
    print(f'  Hands rebuilt   : {total_hands}')
    print(f'  Failed          : {failed}')
    print(f'  Skipped         : {len(all_jsons) - len(pending)}')
    print('=' * 60)


if __name__ == '__main__':
    main()
