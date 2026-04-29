# Copyright (c) jiamingda (https://github.com/Luyitas)

'''
Camera-trajectory rendering (PyRender): render a point cloud from moving cameras.
Tuned for headless servers.
'''

import os
import sys

# Configure headless GL before importing PyRender.
os.environ.setdefault('PYOPENGL_PLATFORM', os.environ.get('RENDER_PLATFORM', 'osmesa'))
# Relax PyOpenGL strict checks for headless use.
os.environ['PYOPENGL_ERROR_ON_COPY'] = '0'

import numpy as np
import h5py
import cv2
from pathlib import Path
from tqdm import tqdm
import pyrender
import trimesh
import subprocess
import tempfile
import shutil
import uuid

# Local temp base (avoid /tmp; keep next to this script).
BASE_TEMP_DIR = Path(__file__).parent / "tmp_render_cache"
BASE_TEMP_DIR.mkdir(parents=True, exist_ok=True)


def pointcloud_cam_to_world(points_cam, camera_pose):
    """
    Map points from camera frame to world frame.
    
    Args:
        points_cam: [N, 3] points in camera coordinates.
        camera_pose: [4, 4] camera-to-world transform.
    
    Returns:
        points_world: [N, 3] world coordinates.
    """
    points_cam_h = np.hstack([points_cam, np.ones((len(points_cam), 1))])
    points_world = (camera_pose @ points_cam_h.T).T[:, :3]
    return points_world


def create_point_cloud_mesh(points, colors, point_size=0.002):
    """
    Method 1 (spheres): approximate each point with a small mesh (slow, pretty).
    PyRender has no native point primitives; instanced spheres are heavy.
    
    Args:
        points: [N, 3]
        colors: [N, 3] RGB in [0,1]
        point_size: sphere radius in meters
    
    Returns:
        Combined trimesh.Trimesh
    """
    # Unit sphere template.
    sphere = trimesh.creation.icosphere(subdivisions=1, radius=point_size)
    
    # Instance per point.
    spheres = []
    for i, (point, color) in enumerate(zip(points, colors)):
        # Translate sphere to sample.
        sphere_copy = sphere.copy()
        sphere_copy.apply_translation(point)
        
        # Vertex color.
        sphere_copy.visual.vertex_colors = np.tile(
            (color * 255).astype(np.uint8), (len(sphere_copy.vertices), 1)
        )
        
        spheres.append(sphere_copy)
        
        # Cap count for speed.
        if i >= 50000:  # hard cap ~50k
            print(f"  ⚠ large cloud; rendering first {i+1} points only")
            break
    
    # Concatenate meshes.
    combined_mesh = trimesh.util.concatenate(spheres)
    return combined_mesh


def create_point_cloud_fast(points, colors):
    """
    Method 2 (point sprites): fast path for large clouds (billboard quads).
    Point radius is controlled by OffscreenRenderer(point_size=...).
    
    Args:
        points: [N, 3]
        colors: [N, 3] RGB in [0,1]
    
    Returns:
        pyrender.Mesh
    """
    # Colors -> uint8.
    colors_uint8 = (colors * 255).astype(np.uint8)
    
    # trimesh PointCloud (unused mesh path).
    point_cloud = trimesh.points.PointCloud(points, colors=colors_uint8)
    
    # pyrender point mesh.
    mesh = pyrender.Mesh.from_points(points, colors=colors_uint8)
    return mesh
def render_video_with_pyrender(points_world, colors, camera_transforms, 
                                intrinsics, image_width, image_height,
                                frame_stride, render_frames, total_frames,
                                temp_dir, fps, output_video,
                                point_size=2.0, mask_mode=False,
                                overlay_video_path=None, original_video_path=None):
    """
    Render point-cloud video from moving cameras.
    Streams to mp4 in memory (no per-frame PNG round-trip).
    
    Args:
        output_video: path for render-only mp4
        overlay_video_path: optional blended mp4 path
        original_video_path: source video for blending
    """
    if mask_mode:
        # Use float white for EGL/OSMesa consistency (some EGL builds dislike int white).
        bg_color = [1.0, 1.0, 1.0]
        colors = np.zeros_like(colors)
    else:
        bg_color = [0.0, 0.0, 0.0]

    scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=bg_color)
    mesh = create_point_cloud_fast(points_world, colors)
    scene.add(mesh)
    
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    camera = pyrender.IntrinsicsCamera(
        fx=fx, fy=fy, cx=cx, cy=cy,
        znear=0.01, zfar=100.0
    )
    
    renderer = pyrender.OffscreenRenderer(image_width, image_height, point_size=point_size)
    
    # Video writers.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # 1) Render-only writer.
    # mkdir parent.
    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    # Write to a unique local file first, then move (safer on some network FS).
    
    # Use UUID to avoid collision in parallel processing where output basename is identical (e.g. render.mp4)
    unique_id = uuid.uuid4().hex
    local_render_path = BASE_TEMP_DIR / f"render_{unique_id}.mp4"
    writer_render = cv2.VideoWriter(str(local_render_path), fourcc, fps, (image_width, image_height))
    
    # 2) Optional overlay writer.
    writer_overlay = None
    cap_orig = None
    if overlay_video_path and original_video_path:
        Path(overlay_video_path).parent.mkdir(parents=True, exist_ok=True)
        local_overlay_path = BASE_TEMP_DIR / f"overlay_{unique_id}.mp4"
        writer_overlay = cv2.VideoWriter(str(local_overlay_path), fourcc, fps, (image_width, image_height))
        
        cap_orig = cv2.VideoCapture(original_video_path)
    
    print(f"[Render] Starting processing {render_frames} frames...")
    
    # Each frame.
    for i in range(render_frames):
        actual_frame = i * frame_stride
        if actual_frame >= len(camera_transforms):
            break
        
        # 1. Update Camera
        camera_pose = camera_transforms[actual_frame]
        camera_nodes = [node for node in scene.get_nodes() if node.camera is not None]
        for node in camera_nodes:
            scene.remove_node(node)
        scene.add(camera, pose=camera_pose)
        
        # 2. Render
        color, depth = renderer.render(scene)
        
        if color.dtype == np.uint8:
            color_uint8 = color
        else:
            color_uint8 = (np.clip(color, 0.0, 1.0) * 255.0).astype(np.uint8)
        image_bgr = cv2.cvtColor(color_uint8, cv2.COLOR_RGB2BGR)
        
        # 3. Write Render Frame
        writer_render.write(image_bgr)
        
        # 4. Overlay if needed
        if writer_overlay is not None and cap_orig is not None:
            # Matching source frame.
            if i == 0:
                cap_orig.set(cv2.CAP_PROP_POS_FRAMES, actual_frame)
            # Seek when stride != 1 (sequential read otherwise).
            if frame_stride != 1:
                cap_orig.set(cv2.CAP_PROP_POS_FRAMES, actual_frame)
            
            ret, original_frame = cap_orig.read()
            if ret:
                # Resize if needed
                if original_frame.shape[:2] != (image_height, image_width):
                    original_frame = cv2.resize(original_frame, (image_width, image_height))
                
                # Blend
                # alpha = 0.5
                overlay_frame = cv2.addWeighted(
                    original_frame, 0.5,
                    image_bgr, 0.5,
                    0
                )
                writer_overlay.write(overlay_frame)
            else:
                # Fallback: Just write render if orig missing
                writer_overlay.write(image_bgr)

    # Teardown GL + writers.
    renderer.delete()
    writer_render.release()
    if writer_overlay:
        writer_overlay.release()
    if cap_orig:
        cap_orig.release()
        
    # Move files to final destination
    if os.path.exists(str(local_render_path)):
        shutil.move(str(local_render_path), output_video)
        print(f"✓ Render video saved: {output_video}")
        
    if writer_overlay and os.path.exists(str(local_overlay_path)):
        shutil.move(str(local_overlay_path), overlay_video_path)
        print(f"✓ Overlay video saved: {overlay_video_path}")


def render_overlay_video(video_path, rendered_frames_dir, output_video, alpha=0.5, fps=30, frame_stride=1):
    """
Blend rendered RGB frames with the source video (legacy helper).
    """
    # Load rendered PNGs.
    rendered_frames = sorted(Path(rendered_frames_dir).glob("frame_*.png"))
    if len(rendered_frames) == 0:
        return
    
    # Source capture.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Staging.
    overlay_dir = Path(output_video).parent / "overlay_frames"
    overlay_dir.mkdir(exist_ok=True)
    
    # Per frame.
    for i, rendered_frame_path in enumerate(rendered_frames):
        # load render.
        rendered_frame = cv2.imread(str(rendered_frame_path))
        if rendered_frame is None:
            continue
        
        # resize to source res.
        if rendered_frame.shape[:2] != (video_height, video_width):
            rendered_frame = cv2.resize(rendered_frame, (video_width, video_height))
        
        # source frame.
        actual_video_frame = i * frame_stride
        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_video_frame)
        ret, original_frame = cap.read()
        
        if not ret:
            break
        
        # alpha blend.
        overlay_frame = cv2.addWeighted(
            original_frame, 1 - alpha,
            rendered_frame, alpha,
            0
        )
        
        # write PNG.
        overlay_frame_path = overlay_dir / f"overlay_{i:06d}.png"
        cv2.imwrite(str(overlay_frame_path), overlay_frame)
    
    cap.release()
    
    # Encode blended mp4.
    overlay_frames = sorted(overlay_dir.glob("overlay_*.png"))
    
    if len(overlay_frames) > 0:
        first_frame = cv2.imread(str(overlay_frames[0]))
        height, width = first_frame.shape[:2]
        
        # temp mp4 then copy (network FS safe).
        temp_video = tempfile.NamedTemporaryFile(
            suffix='.mp4', delete=False, dir=str(BASE_TEMP_DIR)
        )
        temp_video_path = temp_video.name
        temp_video.close()
        
        # Prefer ffmpeg x264.
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(fps),
            "-pattern_type", "glob",
            "-i", str(overlay_frames[0].parent / "overlay_*.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-crf", "18",
            temp_video_path
        ]
        try:
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
            
            for frame_file in overlay_frames:
                frame = cv2.imread(str(frame_file))
                video_writer.write(frame)
            
            video_writer.release()
        
        # copy to final path.
        shutil.copyfile(temp_video_path, str(output_video))
        os.remove(temp_video_path)
        
        print(f"✓ Overlay saved: {output_video}")
        
        # cleanup frames.
        if overlay_dir.exists():
            shutil.rmtree(overlay_dir, ignore_errors=True)


def render_pointcloud_trajectory(pointcloud_path, hdf5_path, output_html, output_video=None,
                                 video_path=None, overlay_alpha=0.5,
                                 frame_stride=1, max_frames=None, fps=30,
                                 point_size=2.0, mask_mode=False):
    """
    Render trajectory from HDF5 poses + NPZ point cloud.
    
    Args:
        pointcloud_path: .npz with points/colors/intrinsics
        hdf5_path: HDF5 with /transforms/camera
        output_html: legacy argument (unused)
        output_video: optional mp4 path
        video_path: optional source for overlays (not wired here)
        overlay_alpha: blend weight
        frame_stride: temporal subsample
        max_frames: cap
        fps: output fps
        point_size: GL point size
        mask_mode: black points on white bg
    """
    # Load NPZ.
    pc_data = np.load(pointcloud_path)
    points_cam = pc_data['points']
    colors = pc_data['colors']
    intrinsics = pc_data['intrinsics']
    original_size = tuple(pc_data['original_size'])
    
    # Load poses.
    root = h5py.File(hdf5_path, 'r')
    camera_transforms = root['/transforms/camera'][:]
    total_frames = len(camera_transforms)
    
    # Frame count.
    if max_frames is not None:
        render_frames = min(max_frames, total_frames // frame_stride)
    else:
        render_frames = total_frames // frame_stride
    
    # OpenCV camera frame -> OpenGL (flip Y,Z).
    opencv_to_opengl = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],  # flip Y
        [0,  0, -1, 0],  # flip Z
        [0,  0,  0, 1]
    ], dtype=np.float32)
    
    # Points to GL camera space.
    points_cam_gl = points_cam.copy()
    points_cam_gl[:, 1] = -points_cam[:, 1]  # flip Y
    points_cam_gl[:, 2] = -points_cam[:, 2]  # flip Z
    
    # Camera c2w -> GL.
    camera_transforms_gl = np.array([c2w @ opencv_to_opengl for c2w in camera_transforms])
    
    # Freeze cloud in world using first pose.
    first_camera_pose_gl = camera_transforms_gl[0]
    points_world = pointcloud_cam_to_world(points_cam_gl, first_camera_pose_gl)
    
    
    # Optional mp4 export.
    if output_video is not None:
        # temp folder (mostly unused now).
        temp_dir = Path(output_video).parent / "temp_frames"
        temp_dir.mkdir(exist_ok=True)
        
        # Offscreen render with GL poses.
        render_video_with_pyrender(
            points_world, colors, camera_transforms_gl,
            intrinsics, original_size[0], original_size[1],
            frame_stride, render_frames, total_frames,
            temp_dir, fps, output_video,
            point_size=point_size,
            mask_mode=mask_mode
        )
        
        # Optional overlay hook (commented).
        # if video_path is not None and Path(video_path).exists():
        #     overlay_video = str(Path(output_video).parent / f"{Path(output_video).stem}_overlay.mp4")
        #     render_overlay_video(
        #         video_path, temp_dir, overlay_video, 
        #         alpha=overlay_alpha, fps=fps, frame_stride=frame_stride
        #     )
        
        # Remove temp dir.
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    root.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PyRender camera-trajectory export (headless-friendly)')
    parser.add_argument('--pointcloud', type=str, required=True, help='Point cloud .npz path')
    parser.add_argument('--hdf5', type=str, required=True, help='HDF5 trajectory path')
    parser.add_argument('--output_html', type=str, default='trajectory_render.html', help='Legacy HTML path (unused)')
    parser.add_argument('--output_video', type=str, default=None, help='Output mp4 (optional)')
    parser.add_argument('--video', type=str, default=None, help='Source video for overlays (optional)')
    parser.add_argument('--overlay_alpha', type=float, default=0.5, help='Overlay blend alpha [0,1]')
    parser.add_argument('--frame_stride', type=int, default=1, help='Temporal stride')
    parser.add_argument('--max_frames', type=int, default=None, help='Max frames to render')
    parser.add_argument('--fps', type=int, default=30, help='Output FPS')
    parser.add_argument('--point_size', type=float, default=2.0, help='Point sprite radius (pixels)')
    parser.add_argument('--mask_mode', action='store_true', help='Black points on white background')
    
    args = parser.parse_args()
    
    render_pointcloud_trajectory(
        args.pointcloud,
        args.hdf5,
        args.output_html,
        args.output_video,
        args.video,
        args.overlay_alpha,
        args.frame_stride,
        args.max_frames,
        args.fps,
        args.point_size,
        args.mask_mode
    )

