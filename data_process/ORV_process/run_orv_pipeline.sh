#!/usr/bin/env bash
# One entry point for ORV RGB video -> 4D occupancy / 3DGS-rendered data.

set -euo pipefail

ORV_PROCESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EGOSIM_ROOT="$(cd "${ORV_PROCESS_ROOT}/../.." && pwd)"

export ORV_PROCESS_ROOT EGOSIM_ROOT
export ORV_DATA_DIR="${ORV_DATA_DIR:-${EGOSIM_ROOT}/data/orv/videos}"
export ORV_SAVE_DIR="${ORV_SAVE_DIR:-${EGOSIM_ROOT}/data/orv/renderings}"
export ORV_EMBEDDING_DIR="${ORV_EMBEDDING_DIR:-${EGOSIM_ROOT}/data/orv/embeddings_full}"
export ORV_N_VIEW="${ORV_N_VIEW:-1}"
export ORV_PROCESS_KEYS="${ORV_PROCESS_KEYS:-points,mesh,occupancy,rendering}"

export WEIGHT_ROOT="${WEIGHT_ROOT:-/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight}"
export HF_HOME="${HF_HOME:-${WEIGHT_ROOT}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${WEIGHT_ROOT}/torch}"
export ORV_MONST3R_CKPT="${ORV_MONST3R_CKPT:-${WEIGHT_ROOT}/ORV/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth}"
export ORV_VGGT_CKPT="${ORV_VGGT_CKPT:-${WEIGHT_ROOT}/ORV/VGGT-1B/model.pt}"
export ORV_SAM2_CKPT="${ORV_SAM2_CKPT:-${WEIGHT_ROOT}/ORV/sam2.1_hiera_large.pt}"

export PYTHONPATH="${ORV_PROCESS_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

usage() {
  cat <<'EOF'
Usage:
  bash run_orv_pipeline.sh setup
  bash run_orv_pipeline.sh weights
  bash run_orv_pipeline.sh prepare-video /path/to/video.mp4 [split] [traj_id]
  bash run_orv_pipeline.sh run [split] [rank/all_ranks]
  bash run_orv_pipeline.sh all /path/to/video.mp4 [split] [traj_id]

Common overrides:
  WEIGHT_ROOT=/path/to/weights
  ORV_DATA_DIR=/path/to/videos
  ORV_SAVE_DIR=/path/to/renderings
  ORV_PROCESS_KEYS=points,mesh,occupancy,rendering
EOF
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing command: $1" >&2; exit 1; }
}

clone_if_missing() {
  local url="$1" dst="$2"
  [[ -d "${dst}/.git" ]] || git clone "${url}" "${dst}"
}

install_requirements_if_present() {
  local req="$1"
  [[ -f "${req}" ]] && python -m pip install -r "${req}"
}

setup_env() {
  local env_name="${ORV_CONDA_ENV:-orv}"
  local python_version="${ORV_PYTHON_VERSION:-3.10}"

  need_cmd conda
  eval "$(conda shell.bash hook)"
  conda env list | awk '{print $1}' | grep -qx "${env_name}" || conda create -n "${env_name}" "python=${python_version}" -y
  conda activate "${env_name}"

  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -r "${ORV_PROCESS_ROOT}/requirements_orv.txt"

  mkdir -p "${ORV_PROCESS_ROOT}/thirdparty"
  clone_if_missing "https://github.com/Junyi42/monst3r" "${ORV_PROCESS_ROOT}/thirdparty/monst3r"
  clone_if_missing "https://github.com/facebookresearch/vggt" "${ORV_PROCESS_ROOT}/thirdparty/vggt"
  clone_if_missing "https://github.com/IDEA-Research/Grounded-SAM-2" "${ORV_PROCESS_ROOT}/thirdparty/grounded_sam_2"

  install_requirements_if_present "${ORV_PROCESS_ROOT}/thirdparty/monst3r/requirements.txt"
  install_requirements_if_present "${ORV_PROCESS_ROOT}/thirdparty/vggt/requirements.txt"
  install_requirements_if_present "${ORV_PROCESS_ROOT}/thirdparty/grounded_sam_2/requirements.txt"
  python -m pip install -e "${ORV_PROCESS_ROOT}/ops/diff-gaussian-rasterization"

  echo "ORV environment is ready: conda activate ${env_name}"
}

prepare_weights() {
  mkdir -p \
    "${WEIGHT_ROOT}/ORV" "${HF_HOME}" "${TORCH_HOME}" \
    "${ORV_PROCESS_ROOT}/thirdparty/monst3r/checkpoints" \
    "${ORV_PROCESS_ROOT}/thirdparty/vggt/vggt/checkpoints" \
    "${ORV_PROCESS_ROOT}/thirdparty/grounded_sam_2/checkpoints"

  command -v huggingface-cli >/dev/null 2>&1 || python -m pip install "huggingface-hub[cli]"

  [[ -f "${ORV_MONST3R_CKPT}" ]] || huggingface-cli download Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt --local-dir "${WEIGHT_ROOT}/ORV/monst3r"
  [[ -f "${WEIGHT_ROOT}/ORV/monst3r/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth" ]] && ln -sfn "${WEIGHT_ROOT}/ORV/monst3r/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth" "${ORV_MONST3R_CKPT}"
  ln -sfn "${ORV_MONST3R_CKPT}" "${ORV_PROCESS_ROOT}/thirdparty/monst3r/checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"

  [[ -f "${ORV_VGGT_CKPT}" ]] || huggingface-cli download facebook/VGGT-1B model.pt --local-dir "$(dirname "${ORV_VGGT_CKPT}")"
  ln -sfn "${ORV_VGGT_CKPT}" "${ORV_PROCESS_ROOT}/thirdparty/vggt/vggt/checkpoints/model.pt"

  [[ -f "${ORV_SAM2_CKPT}" ]] || [[ ! -f "${ORV_PROCESS_ROOT}/thirdparty/grounded_sam_2/checkpoints/download_ckpts.sh" ]] || (cd "${ORV_PROCESS_ROOT}/thirdparty/grounded_sam_2/checkpoints" && bash download_ckpts.sh)
  [[ -f "${ORV_PROCESS_ROOT}/thirdparty/grounded_sam_2/checkpoints/sam2.1_hiera_large.pt" && ! -f "${ORV_SAM2_CKPT}" ]] && cp "${ORV_PROCESS_ROOT}/thirdparty/grounded_sam_2/checkpoints/sam2.1_hiera_large.pt" "${ORV_SAM2_CKPT}"
  [[ -f "${ORV_SAM2_CKPT}" ]] && ln -sfn "${ORV_SAM2_CKPT}" "${ORV_PROCESS_ROOT}/thirdparty/grounded_sam_2/checkpoints/sam2.1_hiera_large.pt"

  echo "Weights prepared under ${WEIGHT_ROOT}/ORV"
}

prepare_video() {
  local video_path="${1:?Missing video path.}"
  local split="${2:-train}"
  local traj_id="${3:-$(basename "${video_path%.*}")}"
  local target_dir="${ORV_DATA_DIR}/${split}/${traj_id}"

  mkdir -p "${target_dir}"
  [[ "${ORV_COPY_VIDEO:-0}" == "1" ]] && cp "${video_path}" "${target_dir}/rgb.mp4" || ln -sfn "${video_path}" "${target_dir}/rgb.mp4"
  printf "Prepared ORV video: %s\n" "${target_dir}/rgb.mp4"
}

run_pipeline() {
  local split="${1:-train}"
  local rank_arg="${2:-}"
  local args=(
    --split "${split}"
    --action reconstruction
    --data_dir "${ORV_DATA_DIR}"
    --save_dir "${ORV_SAVE_DIR}"
    --embedding_dir "${ORV_EMBEDDING_DIR}"
    --n_view "${ORV_N_VIEW}"
    --process_keys "${ORV_PROCESS_KEYS}"
  )

  [[ -n "${rank_arg}" ]] && args+=(--rank "${rank_arg}")
  cd "${ORV_PROCESS_ROOT}"
  python prepare_dataset.py "${args[@]}"
}

cmd="${1:-help}"
shift || true

case "${cmd}" in
  setup) setup_env ;;
  weights) prepare_weights ;;
  prepare-video) prepare_video "$@" ;;
  run) run_pipeline "$@" ;;
  all) prepare_video "$@"; run_pipeline "${2:-train}" ;;
  help|-h|--help) usage ;;
  *) usage; exit 1 ;;
esac
