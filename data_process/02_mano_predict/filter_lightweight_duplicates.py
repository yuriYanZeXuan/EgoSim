# Copyright (c) jiamingda (https://github.com/Luyitas)
import argparse
import json
import os
import sys
import multiprocessing as mp
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock

# Mock pyrender and OpenGL to avoid EGL/OSMesa errors since we only use OpenCV for 2D visualization
sys.modules['pyrender'] = MagicMock()
sys.modules['OpenGL'] = MagicMock()
sys.modules['OpenGL.GL'] = MagicMock()
sys.modules['OpenGL.error'] = MagicMock()
sys.modules['OpenGL.platform'] = MagicMock()

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add hamer to python path to import modules
sys.path.append(str(Path(__file__).resolve().parents[1]))

from hamer.models import MANO

# Global variables for worker processes
g_mano_model = None
g_device = None

def init_worker(mano_path):
    global g_mano_model
    global g_device
    # Force single thread per process to avoid oversubscription
    torch.set_num_threads(1)
    # Use CPU
    g_device = torch.device('cpu')
    g_mano_model = MANO(model_path=mano_path, gender="neutral", num_hand_joints=15).to(g_device)

# Colors from batch_render_manip_mask_3d.py (normalized 0-1 RGB)
# Converted to 0-255 BGR for OpenCV
FINGER_COLORS_RGB = {
    'little': np.array([0, 152, 191]),
    'ring': np.array([173, 255, 47]),
    'middle': np.array([230, 245, 250]),
    'index': np.array([255, 99, 71]),
    'thumb': np.array([238, 130, 238]),
}


def rgb_to_bgr(color_array):
    return tuple(map(int, color_array[::-1]))


FINGER_COLORS_BGR = {k: rgb_to_bgr(v) for k, v in FINGER_COLORS_RGB.items()}

# MANO Joint Indices (User Provided / Convention)
# 0: Wrist
# Thumb: 1, 2, 3, 4
# Index: 5, 6, 7, 8
# Middle: 9, 10, 11, 12
# Ring: 13, 14, 15, 16
# Little: 17, 18, 19, 20
FINGER_CONNECTIONS = {
    'thumb': [0, 1, 2, 3, 4],
    'index': [0, 5, 6, 7, 8],
    'middle': [0, 9, 10, 11, 12],
    'ring': [0, 13, 14, 15, 16],
    'little': [0, 17, 18, 19, 20]
}

# Mapping from MANO joint index to finger name for keypoint coloring
JOINT_TO_FINGER = {}
for finger, indices in FINGER_CONNECTIONS.items():
    for idx in indices:
        if idx == 0:
            continue
        JOINT_TO_FINGER[idx] = finger


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def to_tensor(x, device):
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to(device).float()


def draw_skeleton(frame, keypoints_2d, confidence=None, is_right=None):
    for finger, indices in FINGER_CONNECTIONS.items():
        color = FINGER_COLORS_BGR[finger]
        for i in range(len(indices) - 1):
            idx1, idx2 = indices[i], indices[i + 1]
            pt1 = tuple(map(int, keypoints_2d[idx1]))
            pt2 = tuple(map(int, keypoints_2d[idx2]))
            cv2.line(frame, pt1, pt2, color, 2)

    for i in range(keypoints_2d.shape[0]):
        pt = tuple(map(int, keypoints_2d[i]))
        if i == 0:
            color = FINGER_COLORS_BGR['middle']
        else:
            finger = JOINT_TO_FINGER.get(i, 'middle')
            color = FINGER_COLORS_BGR[finger]
        cv2.circle(frame, pt, 4, color, -1)

    # Draw label if provided
    # if confidence is not None or is_right is not None:
    #     wrist = keypoints_2d[0]
    #     text = ""
    #     if is_right is not None:
    #         text += "R " if is_right else "L "
    #     if confidence is not None:
    #         text += f"{confidence:.2f}"
        
    #     pt = (int(wrist[0]), int(wrist[1]) - 10)
    #     cv2.putText(frame, text, pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    #     cv2.putText(frame, text, pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)



def compute_bbox_from_kps(kps_2d):
    if kps_2d is None:
        return None
    if not np.isfinite(kps_2d).all():
        return None
    x_min = float(np.min(kps_2d[:, 0]))
    y_min = float(np.min(kps_2d[:, 1]))
    x_max = float(np.max(kps_2d[:, 0]))
    y_max = float(np.max(kps_2d[:, 1]))
    if x_max <= x_min or y_max <= y_min:
        return None
    return (x_min, y_min, x_max, y_max)


def compute_iou(box_a, box_b):
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0
    return inter_area / denom


def project_hand_keypoints_2d(hand, mano_model, device, scaled_focal_length, cx, cy):
    batch = {
        'global_orient': to_tensor(hand['global_orient'], device),
        'hand_pose': to_tensor(hand['hand_pose'], device),
        'betas': to_tensor(hand['betas'], device)
    }

    if batch['global_orient'].ndim == 2:
        batch['global_orient'] = batch['global_orient'].unsqueeze(0).unsqueeze(0)
    if batch['global_orient'].ndim == 3:
        batch['global_orient'] = batch['global_orient'].unsqueeze(0)
    if batch['hand_pose'].ndim == 3:
        batch['hand_pose'] = batch['hand_pose'].unsqueeze(0)
    if batch['betas'].ndim == 1:
        batch['betas'] = batch['betas'].unsqueeze(0)

    if 'cam_t_full' in hand:
        cam_t = to_tensor(hand['cam_t_full'], device)
    elif 'cam_t' in hand:
        cam_t = to_tensor(hand['cam_t'], device)
    else:
        return None

    if cam_t.ndim == 1:
        cam_t = cam_t.unsqueeze(0)

    with torch.no_grad():
        mano_output = mano_model(
            global_orient=batch['global_orient'],
            hand_pose=batch['hand_pose'],
            betas=batch['betas'],
            pose2rot=False
        )
        joints_3d = mano_output.joints

        is_right = int(hand['is_right'])
        multiplier = 2 * is_right - 1
        joints_3d[:, :, 0] = multiplier * joints_3d[:, :, 0]

        joints_3d_trans = joints_3d + cam_t.unsqueeze(1)

        x = joints_3d_trans[..., 0]
        y = joints_3d_trans[..., 1]
        z = joints_3d_trans[..., 2]

        u = scaled_focal_length * (x / z) + cx
        v = scaled_focal_length * (y / z) + cy

        pred_kps_2d = torch.stack([u, v], dim=-1).squeeze(0).cpu().numpy()

    return pred_kps_2d


def filter_duplicate_hands(hands, hand_bboxes, iou_thresh):
    num_hands = len(hands)
    if num_hands <= 1:
        return hands, False, list(range(num_hands))

    confidences = [float(h.get('confidence', 0.0)) for h in hands]
    keep_indices = set()

    # Process separated by handedness to avoid mixing Left/Right
    for handedness in [False, True]:
        idxs = [i for i, h in enumerate(hands) if bool(h.get('is_right', False)) == handedness]
        
        if not idxs:
            continue
            
        # Build Connected Components based on IoU > thresh
        # 1. Build Adjacency Graph
        adj = {i: [] for i in idxs}
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                idx1 = idxs[i]
                idx2 = idxs[j]
                
                box1 = hand_bboxes[idx1]
                box2 = hand_bboxes[idx2]
                
                if box1 is None or box2 is None:
                    continue
                
                if compute_iou(box1, box2) > iou_thresh:
                    adj[idx1].append(idx2)
                    adj[idx2].append(idx1)
        
        # 2. Find Components and Keep Max Confidence per Component
        visited = set()
        for i in idxs:
            if i in visited:
                continue
            
            # BFS for component
            component = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            # Find max conf in this component
            if not component:
                continue
            
            best_idx = component[0]
            max_conf = confidences[best_idx]
            
            for c_idx in component[1:]:
                conf = confidences[c_idx]
                if conf > max_conf:
                    max_conf = conf
                    best_idx = c_idx
            
            keep_indices.add(best_idx)

    keep_indices = sorted(list(keep_indices))
    filtered_hands = [hands[i] for i in keep_indices]
    duplicates_found = len(filtered_hands) < num_hands
    return filtered_hands, duplicates_found, keep_indices



def compute_box_center(box):
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def compute_rotation_angle(rod1, rod2):
    # rod1, rod2: (3,) or (1,3) arrays (axis-angle)
    if rod1 is None or rod2 is None:
        return 0.0
    r1 = np.array(rod1)
    r2 = np.array(rod2)

    if r1.size == 9:
        R1 = r1.reshape(3, 3)
    else:
        R1, _ = cv2.Rodrigues(r1.flatten())

    if r2.size == 9:
        R2 = r2.reshape(3, 3)
    else:
        R2, _ = cv2.Rodrigues(r2.flatten())

    # Relative rotation R_rel = R1^T * R2
    R_rel = np.dot(R1.T, R2)
    # trace = 1 + 2cos(theta)
    tr = np.trace(R_rel)
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return theta # radians

def filter_temporal_outliers(filtered_results_map, pose_angle_thresh=1.5, velocity_thresh=0.2):
    """
    Filter hands that exhibit unnatural temporal jumps (spikes) in pose or position.
    
    Args:
        filtered_results_map: frame_idx -> (filtered_hands, keep_indices, hand_kps)
        pose_angle_thresh: Radians per frame (max allowed angular velocity spike)
        velocity_thresh: Units/frame (max allowed translational velocity spike)
    """
    # 1. Organize tracks separated by handedness
    frames = sorted(filtered_results_map.keys())
    # Struct: is_right (0/1) -> list of dict(frame_idx, cam_t, orient, list_idx)
    tracks = {False: [], True: []}
    
    for f in frames:
        hands_list, keep_indices, kps_list = filtered_results_map[f]
        
        for i, hand in enumerate(hands_list):
            is_right = bool(hand.get('is_right', False))
            # Try to get 3D translation
            cam_t = hand.get('cam_t_full') if 'cam_t_full' in hand else hand.get('cam_t')
            orient = hand.get('global_orient')
            
            track_item = {
                'frame_idx': f,
                'cam_t': cam_t,
                'orient': orient,
                'list_idx': i # Index within the filtered_hands list of this frame
            }
            tracks[is_right].append(track_item)

    outliers_removed = False
    violation_details = []

    # 2. Check each track
    for is_right, track in tracks.items():
        if len(track) < 3:
            continue
            
        bad_indices = set()
        
        # Check triads (prev, curr, next)
        for i in range(1, len(track) - 1):
            curr = track[i]
            prev = track[i-1]
            next_fr = track[i+1]
            
            # Check Frame Continuity (if gap is too large, tracking broken, skip check)
            if (curr['frame_idx'] - prev['frame_idx']) > 5:
                continue
            if (next_fr['frame_idx'] - curr['frame_idx']) > 5:
                continue

            # Calculate metrics (prev -> curr)
            dt_p = float(curr['frame_idx'] - prev['frame_idx'])
            v_trans_p = 0.0
            if curr['cam_t'] is not None and prev['cam_t'] is not None:
                dist_val = np.linalg.norm(np.array(curr['cam_t']) - np.array(prev['cam_t']))
                v_trans_p = dist_val / dt_p
                
            v_ang_p = 0.0
            if curr['orient'] is not None and prev['orient'] is not None:
                ang = compute_rotation_angle(prev['orient'], curr['orient'])
                v_ang_p = ang / dt_p

            # Calculate metrics (curr -> next)
            dt_n = float(next_fr['frame_idx'] - curr['frame_idx'])
            v_trans_n = 0.0
            if next_fr['cam_t'] is not None and curr['cam_t'] is not None:
                dist_val = np.linalg.norm(np.array(next_fr['cam_t']) - np.array(curr['cam_t']))
                v_trans_n = dist_val / dt_n

            v_ang_n = 0.0
            if next_fr['orient'] is not None and curr['orient'] is not None:
                ang = compute_rotation_angle(curr['orient'], next_fr['orient'])
                v_ang_n = ang / dt_n
            
            # Check "Spike": Jump In AND Jump Out
            # If motion is unusually large in both directions relative to neighbors, it's a glitch
            is_pos_glitch = (v_trans_p > velocity_thresh) and (v_trans_n > velocity_thresh)
            is_rot_glitch = (v_ang_p > pose_angle_thresh) and (v_ang_n > pose_angle_thresh)
            
            if is_pos_glitch or is_rot_glitch:
                bad_indices.add(i)
                
                hand_str = "Right" if is_right else "Left"
                detail_parts = []
                if is_pos_glitch:
                    detail_parts.append(f"Vel=({v_trans_p:.3f},{v_trans_n:.3f})")
                if is_rot_glitch:
                    detail_parts.append(f"Rot=({v_ang_p:.3f},{v_ang_n:.3f})")
                
                violation_details.append(f"Fr{curr['frame_idx']}({hand_str}):{' '.join(detail_parts)}")
                # print(f"Frame {curr['frame_idx']} Right={is_right} Outlier: pos={v_trans_p:.2f}/{v_trans_n:.2f}, rot={v_ang_p:.2f}/{v_ang_n:.2f}")

        if bad_indices:
            outliers_removed = True
            
            # Identify which hands to remove per frame
            frame_to_bad_list_idxs = {}
            for idx in bad_indices:
                item = track[idx]
                f = item['frame_idx']
                if f not in frame_to_bad_list_idxs:
                    frame_to_bad_list_idxs[f] = set()
                frame_to_bad_list_idxs[f].add(item['list_idx'])
            
            # Update map
            for f, remove_set in frame_to_bad_list_idxs.items():
                hands, keep, kps = filtered_results_map[f]
                
                # Keep items not in remove_set
                new_hands = [h for i, h in enumerate(hands) if i not in remove_set]
                new_keep = [k for i, k in enumerate(keep) if i not in remove_set]
                
                filtered_results_map[f] = (new_hands, new_keep, kps)

    return outliers_removed, violation_details



def process_clip(clip_id, video_dir, annot_dir, output_dir, output_annot_dir, iou_thresh, modified_log_path, args, mano_model=None, device=None):
    # Use global model/device if not provided (worker process case)
    if mano_model is None:
        global g_mano_model
        mano_model = g_mano_model
    if device is None:
        global g_device
        device = g_device

    video_path = os.path.join(video_dir, f"{clip_id}.mp4")
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return None

    annot_path_no_ext = os.path.join(annot_dir, clip_id)
    annot_path_json = os.path.join(annot_dir, f"{clip_id}.json")

    annot_path = None
    # Handle filesystem quirks where exists() might raise FileNotFoundError
    try:
        if os.path.exists(annot_path_json):
            annot_path = annot_path_json
    except OSError:
        pass

    if annot_path is None:
        try:
            if os.path.exists(annot_path_no_ext):
                annot_path = annot_path_no_ext
        except OSError:
            pass

    if annot_path is None:
        print(f"Annotation not found for {clip_id}")
        return None

    try:
        annot_data = load_json(annot_path)
    except json.JSONDecodeError:
        print(f"Failed to load JSON: {annot_path}")
        return None


    # Pass 1: Filter and Check
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    HAMER_FOCAL_LENGTH = 5000.0
    HAMER_IMAGE_SIZE = 256.0
    scaled_focal_length = HAMER_FOCAL_LENGTH / HAMER_IMAGE_SIZE * max(width, height)
    cx = width / 2.0
    cy = height / 2.0

    annot_frames_map = {f['frame_idx']: f for f in annot_data.get('frames', [])}
    
    # Pre-calculate data for all frames
    frame_data_list = []
    
    any_duplicates_found = False

    # We iterate based on frames present in annotation to be faster, 
    # but to match video we might need total_frames loop if we want to visualize correctly.
    # However, for filtering check we only care about annotated frames.
    # Let's verify if modification happens first.
    
    sorted_frame_indices = sorted(annot_frames_map.keys())
    
    filtered_results_map = {} # frame_idx -> (filtered_hands, keep_indices, hand_kps)
    
    for frame_idx in sorted_frame_indices:
        hands = annot_frames_map[frame_idx].get('hands', [])
        if not hands:
            filtered_results_map[frame_idx] = ([], [], [])
            continue

        hand_kps = []
        hand_bboxes = []
        for hand in hands:
            try:
                kps_2d = project_hand_keypoints_2d(
                    hand,
                    mano_model,
                    device,
                    scaled_focal_length,
                    cx,
                    cy
                )
            except Exception as e:
                # print(f"Error processing hand in frame {frame_idx}: {e}")
                kps_2d = None
            hand_kps.append(kps_2d)
            hand_bboxes.append(compute_bbox_from_kps(kps_2d))
        
        filtered_hands = hands
        keep_indices = list(range(len(hands)))
        
        if len(hands) > 1:
            filtered_hands, duplicates_found, keep_indices = filter_duplicate_hands(
                hands, hand_bboxes, iou_thresh
            )
            if duplicates_found:
                any_duplicates_found = True

        # --- New Requirement: Keep only MAX confidence per hand side ---
        # After IoU filtering, if we still have multiple left or multiple right hands, keep only the best one.
        
        # Split remaining hands by side
        single_side_hands = []
        single_side_keep_indices = []
        
        best_left_idx = -1
        best_left_conf = -1.0
        
        best_right_idx = -1
        best_right_conf = -1.0

        # Indices in filtered_hands match indices in keep_indices
        for i, hand in enumerate(filtered_hands):
            orig_idx = keep_indices[i]
            conf = float(hand.get('confidence', 0.0))
            is_right = bool(hand.get('is_right', False))
            
            if is_right:
                if conf > best_right_conf:
                    best_right_conf = conf
                    best_right_idx = i # index in filtered_hands
            else:
                if conf > best_left_conf:
                    best_left_conf = conf
                    best_left_idx = i # index in filtered_hands
        
        # Rebuild list with only best left and best right
        final_frame_hands = []
        final_frame_keep = []
        
        if best_left_idx != -1:
            final_frame_hands.append(filtered_hands[best_left_idx])
            final_frame_keep.append(keep_indices[best_left_idx])
            
        if best_right_idx != -1:
            final_frame_hands.append(filtered_hands[best_right_idx])
            final_frame_keep.append(keep_indices[best_right_idx])
            
        # Check if we removed any extra hands in this step
        if len(final_frame_hands) < len(filtered_hands):
             any_duplicates_found = True

        # Pass 3: Update map with strictly 1L 1R max
        filtered_results_map[frame_idx] = (final_frame_hands, final_frame_keep, hand_kps)
        

    # --- New Step: Temporal Outlier Check ---
    # Check if the remaining single-hand streams have physics violations
    is_temporal_outlier = False
    outlier_details = []
    if True:
        # We reuse the outlier detection function but just to check boolean status
        # We DO NOT modify the results here, just detect "bad video"
        # Create a copy so we don't mutate if function mutates
        # Actually filter_temporal_outliers modifies the map. 
        # But per instruction: "If temporal outlier -> Do not save annotation, move to bad folder"
        # So we can let it run. If it returns True, we flag this entire clip as bad.
        
        # We need a fresh check without modification first?
        # The function `filter_temporal_outliers` returns True if it removed something.
        # But here logic is: if huge jump exists, the WHOLE CLIP is bad (or at least we identify it).
        # Save videos filtered by temporal criteria to a separate folder (no JSON).
        
        # Let's run detection.
        # Modified filter_temporal_outliers to optionally NOT remove, just report?
        # Or we rely on `outliers_removed` return value.
        # If it returns True, it means it found something bad.
        
        # However, `filter_temporal_outliers` modifies `filtered_results_map` in place to remove bad frames.
        # If we want to discard the whole video, that is fine.
        
        # Let's clone map for checking to preserve original filtered state?
        # Actually if we are discarding the annotation, we don't care if map is modified.
        
        is_temporal_outlier, outlier_details = filter_temporal_outliers(filtered_results_map, args.temp_pose_thresh, args.temp_vel_thresh)

    if is_temporal_outlier:
        # This is a "Bad" clip with temporal jumps
        # 1. Do not save annotation
        # 2. Add to bad video list
        # 3. Save visualization to special folder
        
        bad_video_dir = os.path.join(output_dir, "temporal_outliers_vis")
        os.makedirs(bad_video_dir, exist_ok=True)
        
        bad_video_path = os.path.join(bad_video_dir, f"{clip_id}_bad_temp.mp4")
        
        # Save bad list
        bad_list_path = os.path.join(output_dir, "temporal_outliers.txt")
        # We can't write to single file safely in mp. 
        # We can treat this as another return type to collect in main.
        
        # But user asked to "Save video".
        # We render the video showing the jumps (or just the cleaned version? Usually evidence of bad tracking).
        # Let's render the version *before* temporal cleaning (which shows the jump) vs after?
        # Since `filter_temporal_outliers` modified the map, the "after" frames are gone.
        # If we want to see the jump, we should have rendered before modification.
        
        # If we just want to save "the video that was filtered out", we can visualize the *surviving* duplicates-filtered hands.
        # But since we called filter_temporal_outliers, the map is already cleaned of the jumps.
        # The "jump" is gaps in the track now.
        
        # Let's just save the visualization of what we have (cleaned or semi-cleaned)
        
        # Write video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(bad_video_path, fourcc, fps, (width, height))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        for frame_idx in range(total_frames):
             ret, frame = cap.read()
             if not ret: break
             
             # Draw what remains (if any) or just the raw input to see why it failed?
             # Usually we want to see the Skeleton that caused the issue.
             # But `filter_temporal_outliers` REMOVED it.
             # So we can't see it if we use `filtered_results_map`.
             
             # To properly visualize the "Bad" jumps, we should probably NOT have modified the map
             # or we should visualize the state before temporal filtering.
             # Given the flow, let's just visualize the `filtered_results_map` which represents "Best attempt".
             
             if frame_idx in filtered_results_map:
                hands_list, k_idxs, h_kps = filtered_results_map[frame_idx]
                
                # Retrieve original hands info for labeling
                original_hands_info = annot_frames_map.get(frame_idx, {}).get('hands', [])
                
                for idx in k_idxs:
                    if idx < len(h_kps) and h_kps[idx] is not None:
                        # Extract info
                        conf = None
                        is_r = None
                        if idx < len(original_hands_info):
                            item = original_hands_info[idx]
                            conf = float(item.get('confidence', 0))
                            is_r = bool(item.get('is_right', False))
                        
                        draw_skeleton(frame, h_kps[idx], confidence=conf, is_right=is_r)
             
             out_writer.write(frame)
        
        out_writer.release()
        cap.release()
        
        details_str = "|".join(outlier_details)
        return f"TEMP_OUTLIER:{clip_id}:{details_str}"


    # Construct filtered frames list for saving JSON
    final_filtered_frames = []
    # We should preserve frame structure. The original annot_data['frames'] might be sparse.
    # We iterate the map we just built.
    for frame_idx in sorted_frame_indices:
        filtered_hands, _, _ = filtered_results_map[frame_idx]
        final_filtered_frames.append({'frame_idx': frame_idx, 'hands': filtered_hands})

    # Save Filtered JSON (Only if NOT temporal outlier)
    filtered_path = os.path.join(output_annot_dir, f"{clip_id}.json")
    os.makedirs(os.path.dirname(filtered_path), exist_ok=True)
    with open(filtered_path, "w") as f:
        json.dump({"video_id": clip_id, "frames": final_filtered_frames}, f)
    
    # Check if we should exit early (no visualization)
    if not any_duplicates_found:
        cap.release()
        # print(f"No duplicates found for {clip_id}. Saved filtered annotations only.")
        return None
    
    if args.no_vis:
        cap.release()
        # Return clip_id to signal modification
        return clip_id

    # Pass 2: Visualization (Only if duplicates found and valid)
    # We return clip_id at the end to signal modification
    
    os.makedirs(output_dir, exist_ok=True)
    before_path = os.path.join(output_dir, f"{clip_id}_vis_before.mp4")
    after_path = os.path.join(output_dir, f"{clip_id}_vis_after.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    before_writer = cv2.VideoWriter(before_path, fourcc, fps, (width, height))
    after_writer = cv2.VideoWriter(after_path, fourcc, fps, (width, height))
    
    # We need to restart reading video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    for frame_idx in tqdm(range(total_frames), desc=f"Visualizing {clip_id}", leave=False):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get precomputed data
        if frame_idx in filtered_results_map:
            filtered_hands, keep_indices, hand_kps = filtered_results_map[frame_idx]
            
            # Retrieve original hands info for labeling
            original_hands_info = annot_frames_map.get(frame_idx, {}).get('hands', [])
            
            before_frame = frame.copy()
            after_frame = frame.copy()

            # Before: Draw all
            for i, kps_2d in enumerate(hand_kps):
                if kps_2d is not None:
                    # Extract info
                    conf = None
                    is_r = None
                    if i < len(original_hands_info):
                        item = original_hands_info[i]
                        conf = float(item.get('confidence', 0))
                        is_r = bool(item.get('is_right', False))
                    
                    draw_skeleton(before_frame, kps_2d, confidence=conf, is_right=is_r)
            
            # Add count labeling to before frame
            if len(hand_kps) > 0:
                cv2.putText(before_frame, f"Det: {len(hand_kps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # After: Draw kept
            for idx in keep_indices:
                if idx < len(hand_kps) and hand_kps[idx] is not None:
                    # Extract info
                    conf = None
                    is_r = None
                    if idx < len(original_hands_info):
                        item = original_hands_info[idx]
                        conf = float(item.get('confidence', 0))
                        is_r = bool(item.get('is_right', False))
                        
                    draw_skeleton(after_frame, hand_kps[idx], confidence=conf, is_right=is_r)
            
            before_writer.write(before_frame)
            after_writer.write(after_frame)
        else:
            before_writer.write(frame)
            after_writer.write(frame)

    cap.release()
    before_writer.release()
    after_writer.release()

    print(f"Saved: {before_path}")
    print(f"Saved: {after_path}")
    print(f"Saved filtered annotations: {filtered_path}")
    return clip_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_list", required=True, help="Path to kept_clips.txt")
    parser.add_argument("--annot_dir", required=True, help="Directory of lightweight annotations")
    parser.add_argument("--video_dir", required=True, help="Directory of original videos")
    parser.add_argument("--output_dir", required=True, help="Output directory for visualization")
    parser.add_argument("--output_annot_dir", default=None, help="Output directory for filtered annotations")
    parser.add_argument("--mano_path", required=True, help="MANO model path")
    parser.add_argument("--iou_thresh", type=float, default=0.6, help="IoU threshold for duplicate filtering")
    parser.add_argument("--modified_log", default=None, help="Output path for modified video list")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of worker processes")
    parser.add_argument("--no_vis", action="store_true", help="Do not save visualization videos (annotations only)")
    parser.add_argument("--temp_pose_thresh", type=float, default=5.0, help="Temporal pose outlier threshold (radians/frame), default 2.5 (~143deg)")
    parser.add_argument("--temp_vel_thresh", type=float, default=10.0, help="Temporal velocity outlier threshold (m/frame), default 5.0")

    args = parser.parse_args()

    if args.output_annot_dir is None:
        args.output_annot_dir = os.path.join(args.output_dir, "filtered_annotations")

    # Set multiprocessing start method to spawn (safer for PyTorch)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    with open(args.clip_list, "r") as f:
        clips = [line.strip() for line in f if line.strip()]

    print(f"Found {len(clips)} clips to process.")
    print(f"Running with {args.num_workers} workers.")

    if args.modified_log is None:
        args.modified_log = os.path.join(args.output_dir, "modified_video.txt")
    
    # Ensure output dir exists
    os.makedirs(os.path.dirname(args.modified_log), exist_ok=True)
    
    # Partial function for the worker
    worker_fn = partial(
        process_clip,
        video_dir=args.video_dir,
        annot_dir=args.annot_dir,
        output_dir=args.output_dir,
        output_annot_dir=args.output_annot_dir,
        iou_thresh=args.iou_thresh,
        modified_log_path=None, # Not used inside anymore for writing
        args=args
    )

    # Prepare log files
    temp_outlier_log = os.path.join(args.output_dir, "temporal_outliers.txt")
    temp_outlier_details_log = os.path.join(args.output_dir, "temporal_outliers_details.txt")
    
    # Clear/Init files
    for p in [args.modified_log, temp_outlier_log, temp_outlier_details_log]:
        with open(p, "w") as f:
            pass # Create empty file

    print(f"Logging to:\n  {args.modified_log}\n  {temp_outlier_log}\n  {temp_outlier_details_log}")
    
    # Run with Pool
    with mp.Pool(processes=args.num_workers, initializer=init_worker, initargs=(args.mano_path,)) as pool:
        with open(args.modified_log, "a") as f_mod, \
             open(temp_outlier_log, "a") as f_out, \
             open(temp_outlier_details_log, "a") as f_det:
             
            for res in tqdm(pool.imap_unordered(worker_fn, clips), total=len(clips), desc="Processing clips"):
                if res is not None:
                    if isinstance(res, str) and res.startswith("TEMP_OUTLIER:"):
                        parts = res.split(":", 2)
                        clip_id = parts[1]
                        
                        f_out.write(f"{clip_id}\n")
                        f_out.flush()
                        
                        if len(parts) > 2:
                            f_det.write(f"{clip_id} -> {parts[2]}\n")
                            f_det.flush()
                    else:
                        f_mod.write(f"{res}\n")
                        f_mod.flush()
    
    print(f"Done.")

if __name__ == "__main__":
    main()
