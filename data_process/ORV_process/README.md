# ORV Process

This folder contains the ORV data preparation pipeline needed to turn RGB videos into:

- `points/`: MonST3R sparse point reconstruction and camera parameters.
- `mesh/`: NKSR dense mesh reconstruction.
- `occ/`: 4D occupancy voxel data, saved as `frame_*.npy`.
- `render/`: 3D Gaussian Splatting rendered depth/semantic conditions, saved as `{traj_id}.npz`.

## Input Layout

`prepare_dataset.py` expects:

```text
${ORV_DATA_DIR}/{train,val,test}/{traj_id}/rgb.mp4
```

For a single video:

```bash
bash data_process/ORV_process/run_orv_pipeline.sh prepare-video /path/to/video.mp4 train my_clip
```

## Setup

```bash
cd /path/to/EgoSim
bash data_process/ORV_process/run_orv_pipeline.sh setup
conda activate orv
bash data_process/ORV_process/run_orv_pipeline.sh weights
```

Weights default to:

```text
/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/ORV
```

Override paths by exporting environment variables before running the script.

## Run

```bash
conda activate orv
bash data_process/ORV_process/run_orv_pipeline.sh run train
```

Prepare one video and run the default pipeline in one command:

```bash
bash data_process/ORV_process/run_orv_pipeline.sh all /path/to/video.mp4 train my_clip
```

Distributed sharding keeps ORV's original `rank/all_ranks` format:

```bash
bash data_process/ORV_process/run_orv_pipeline.sh run train 0/4
bash data_process/ORV_process/run_orv_pipeline.sh run train 1/4
bash data_process/ORV_process/run_orv_pipeline.sh run train 2/4
bash data_process/ORV_process/run_orv_pipeline.sh run train 3/4
```

To run only selected stages:

```bash
ORV_PROCESS_KEYS=points,mesh,occupancy bash data_process/ORV_process/run_orv_pipeline.sh run train
ORV_PROCESS_KEYS=rendering bash data_process/ORV_process/run_orv_pipeline.sh run train
```

## Notes

- `ops/voxelize` is JIT-compiled on first import and requires CUDA/nvcc.
- `ops/diff-gaussian-rasterization` is installed by `run_orv_pipeline.sh setup`.
- Semantic labeling is optional. Without generated `semantics/`, the upstream occupancy stage currently raises and skips those frames; run `caption`, `caption_post_process`, `labeling`, and `labels_post_process` first if semantic labels are required.
