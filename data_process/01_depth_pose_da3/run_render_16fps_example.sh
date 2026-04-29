#!/bin/bash
# Copyright (c) jiamingda (https://github.com/Luyitas)
set -euo pipefail

# Sharded 16 fps rendering:
# - Optional 8-way sharding (SHARD_ID=1..8)
# - Default 64 concurrent workers per node (MAX_JOBS=64)
# - Scan clips under INPAINT_ROOT, or pass a list file

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="${SCRIPT_DIR}/render_16fps_aligned.py"

# Use system python by default; set USE_CONDA=1 to try conda
USE_CONDA="${USE_CONDA:-0}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-depthanything}"
if [[ "${USE_CONDA}" == "1" ]] && command -v conda >/dev/null 2>&1; then
    export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
    export CONDA_NO_PLUGINS=1
    if conda run --no-capture-output -n "${CONDA_ENV_NAME}" python -c "import sys" >/dev/null 2>&1; then
        PYTHON_CMD=(conda run --no-capture-output -n "${CONDA_ENV_NAME}" python)
    else
        echo "[warn] conda env not usable: ${CONDA_ENV_NAME}, fallback to system python"
        PYTHON_CMD=(python)
    fi
else
    PYTHON_CMD=(python)
fi

# ================= Usage =================
# 1) Auto-scan INPAINT_ROOT, run shard 1 of 8:
#    ./run_render_16fps_example.sh 1
#
# 2) Use a list file (clip name or path per line), run shard 3:
#    ./run_render_16fps_example.sh 3 ./test8_inpainted_folders.txt
#
# Required env (dataset roots); others optional:
#   VIDEO_ROOT   Required: parent of <clip>/video.mp4
#   INPAINT_ROOT Required: parent of <clip>/hand_inpaint.png, depth, intrinsics, etc.
#   POSE_ROOT    Required: DA3 pose root (one subdir per clip)
#   TOTAL_SHARDS=8
#   MAX_JOBS=32  (workers per node; 32 is a reasonable default)
#   SKIP_DONE=1
#   OUT_ROOT (optional; default ./render_16fps_outputs_mask_refined)
#   FPS=16 / POINT_SIZE=2.0
#   MASK_MODE=0|1 (1: mask-style output video)
#   NUM_GPUS=0   (>0: bind workers to GPU via worker_id % NUM_GPUS)
#   RENDER_PLATFORM=egl|osmesa|pyglet  (default egl; egl recommended on GPU nodes)
#   AUTO_INSTALL_RENDER_DEPS=1|0 (default 1: apt-install missing GL deps if root; 0 to disable)

SHARD_ID="${1:-1}"
INPUT_LIST="${2:-}"

TOTAL_SHARDS="${TOTAL_SHARDS:-8}"
MAX_JOBS="${MAX_JOBS:-8}"
SKIP_DONE="${SKIP_DONE:-1}"
FPS="${FPS:-16}"
POINT_SIZE="${POINT_SIZE:-2.0}"
MASK_MODE="${MASK_MODE:-0}"
AUTO_INSTALL_RENDER_DEPS="${AUTO_INSTALL_RENDER_DEPS:-1}"

# Limit BLAS/OpenCV threads per process to reduce CPU oversubscription
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export OPENCV_OPENCL_RUNTIME="${OPENCV_OPENCL_RUNTIME:-disabled}"

if [[ -z "${NUM_GPUS:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        NUM_GPUS=$(nvidia-smi -L | wc -l)
        echo "[info] Auto-detected NUM_GPUS=${NUM_GPUS}"
    else
        NUM_GPUS=0
    fi
fi
NUM_GPUS="${NUM_GPUS:-0}"

RENDER_PLATFORM="${RENDER_PLATFORM:-egl}"

export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${RENDER_PLATFORM}}"

if [[ "${PYOPENGL_PLATFORM}" == "egl" ]] && [[ -f "/usr/lib/x86_64-linux-gnu/libEGL.so.1" ]]; then
    if [[ -n "${LD_PRELOAD:-}" ]]; then
        export LD_PRELOAD="${LD_PRELOAD}:/usr/lib/x86_64-linux-gnu/libEGL.so.1"
    else
        export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libEGL.so.1"
    fi
fi

if [[ "${PYOPENGL_PLATFORM}" == "egl" ]] && [[ "${NUM_GPUS}" -gt 0 ]]; then
    local_recommended=$((NUM_GPUS * 2))
    if [[ "${MAX_JOBS}" -gt "${local_recommended}" ]]; then
        echo "[warn] MAX_JOBS=${MAX_JOBS} may be too high for egl with NUM_GPUS=${NUM_GPUS}." >&2
        echo "[warn] Suggested MAX_JOBS <= ${local_recommended} (about 1-2 workers per GPU)." >&2
    fi
fi

check_render_backend() {
    local platform="$1"
    local check_script='import ctypes, sys
from ctypes.util import find_library
p = sys.argv[1].lower()
name = {"egl": "EGL", "osmesa": "OSMesa"}.get(p)
if name is None:
    sys.exit(0)
lib = find_library(name)
if not lib:
    sys.exit(2)
try:
    ctypes.CDLL(lib)
except Exception:
    sys.exit(3)
'

    if ! "${PYTHON_CMD[@]}" -c "${check_script}" "${platform}" >/dev/null 2>&1; then
        local rc=$?

        maybe_auto_install_render_dep() {
            local dep_platform="$1"
            if [[ "${AUTO_INSTALL_RENDER_DEPS}" != "1" ]]; then
                return 1
            fi
            if ! command -v apt-get >/dev/null 2>&1; then
                echo "[warn] AUTO_INSTALL_RENDER_DEPS=1 but apt-get not found; skip auto install." >&2
                return 1
            fi
            if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
                echo "[warn] AUTO_INSTALL_RENDER_DEPS=1 needs root (run as root or with sudo)." >&2
                return 1
            fi

            echo "[info] Auto-installing render deps for platform=${dep_platform} ..."
            if [[ "${dep_platform}" == "osmesa" ]]; then
                apt-get update && apt-get install -y libosmesa6 libosmesa6-dev
            elif [[ "${dep_platform}" == "egl" ]]; then
                apt-get update && apt-get install -y libegl1 libegl-mesa0 libgl1 libglx0 libgles2
            else
                return 1
            fi
            return 0
        }

        if [[ "${platform}" == "egl" ]]; then
            echo "[err] Render backend '${platform}' is selected, but EGL library is not available." >&2
            echo "[hint] Install command (Ubuntu/Debian): sudo apt-get update && sudo apt-get install -y libegl1 libegl-mesa0 libgl1 libglx0 libgles2" >&2
            echo "[hint] Conda alternative: conda install -c conda-forge mesa-libegl-cos7-x86_64" >&2
            echo "[hint] Or switch backend: export RENDER_PLATFORM=osmesa" >&2
            maybe_auto_install_render_dep "egl" || true
        elif [[ "${platform}" == "osmesa" ]]; then
            echo "[err] Render backend '${platform}' is selected, but OSMesa library is not available." >&2
            echo "[hint] Install command (Ubuntu/Debian): sudo apt-get update && sudo apt-get install -y libosmesa6 libosmesa6-dev" >&2
            echo "[hint] Conda alternative: conda install -c conda-forge mesalib" >&2
            echo "[hint] On GPU machine you can switch backend: export RENDER_PLATFORM=egl" >&2
            maybe_auto_install_render_dep "osmesa" || true
        else
            echo "[err] Render backend '${platform}' check failed (rc=${rc})." >&2
        fi

        if "${PYTHON_CMD[@]}" -c "${check_script}" "${platform}" >/dev/null 2>&1; then
            echo "[info] Render backend dependency recovered: ${platform}"
            return 0
        fi

        exit 1
    fi
}

check_render_backend "${PYOPENGL_PLATFORM}"

: "${VIDEO_ROOT:?Set VIDEO_ROOT: parent of <clip_name>/video.mp4}"
: "${INPAINT_ROOT:?Set INPAINT_ROOT: parent of <clip_name>/hand_inpaint.png (and depth/intrinsics)}"
: "${POSE_ROOT:?Set POSE_ROOT: DA3 pose output root (<clip_name>/extrinsics_*.npy etc.)}"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/render_16fps_outputs_mask_refined}"

if ! [[ "${SHARD_ID}" =~ ^[0-9]+$ ]]; then
    echo "[err] SHARD_ID must be integer, got: ${SHARD_ID}" >&2
    exit 1
fi
if ! [[ "${TOTAL_SHARDS}" =~ ^[0-9]+$ ]] || [[ "${TOTAL_SHARDS}" -le 0 ]]; then
    echo "[err] TOTAL_SHARDS must be positive integer, got: ${TOTAL_SHARDS}" >&2
    exit 1
fi
if ! [[ "${MAX_JOBS}" =~ ^[0-9]+$ ]] || [[ "${MAX_JOBS}" -le 0 ]]; then
    echo "[err] MAX_JOBS must be positive integer, got: ${MAX_JOBS}" >&2
    exit 1
fi
if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]] || [[ "${NUM_GPUS}" -lt 0 ]]; then
    echo "[err] NUM_GPUS must be non-negative integer, got: ${NUM_GPUS}" >&2
    exit 1
fi
if [[ "${SHARD_ID}" -lt 1 ]] || [[ "${SHARD_ID}" -gt "${TOTAL_SHARDS}" ]]; then
    echo "[err] SHARD_ID out of range: ${SHARD_ID}, expected 1..${TOTAL_SHARDS}" >&2
    exit 1
fi

mkdir -p "${OUT_ROOT}"
SHARD_DIR="${OUT_ROOT}/_shards"
mkdir -p "${SHARD_DIR}"
SHARD_LIST_FILE="${SHARD_DIR}/shard_${SHARD_ID}_of_${TOTAL_SHARDS}.txt"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
SHARD_LOG_FILE="${SHARD_DIR}/shard_${SHARD_ID}_of_${TOTAL_SHARDS}_${RUN_TS}.log"
PROGRESS_LOG_FILE="${SHARD_DIR}/progress_shard_${SHARD_ID}_of_${TOTAL_SHARDS}_${RUN_TS}.log"
LOG_LOCK_FILE="${SHARD_DIR}/log_lock_${SHARD_ID}_of_${TOTAL_SHARDS}_${RUN_TS}.lock"
PROGRESS_INTERVAL_SEC="${PROGRESS_INTERVAL_SEC:-2}"
IS_TTY=0
if [[ -t 1 ]]; then
    IS_TTY=1
fi

log_msg() {
    (
        flock -x 201
        printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "${SHARD_LOG_FILE}"
    ) 201>"${LOG_LOCK_FILE}"
}

format_duration() {
    local sec="$1"
    if [[ -z "${sec}" ]] || [[ "${sec}" -lt 0 ]]; then
        sec=0
    fi
    local h=$((sec / 3600))
    local m=$(((sec % 3600) / 60))
    local s=$((sec % 60))
    printf '%02d:%02d:%02d' "${h}" "${m}" "${s}"
}

print_progress() {
    local done="$1"
    local total="$2"
    local ok_count="$3"
    local done_count="$4"
    local skip_count="$5"
    local fail_count="$6"
    local tick="$7"
    local elapsed_sec="$8"
    local eta_sec="$9"
    local avg_sec_per_clip="${10}"

    local width=40
    local filled=0
    local pct=0
    local remaining=0
    local spinner='|/-\\'
    local spin_char='|'

    if [[ "${total}" -gt 0 ]]; then
        filled=$(( done * width / total ))
        pct=$(( done * 100 / total ))
    fi
    remaining=$(( total - done ))
    if [[ "${remaining}" -lt 0 ]]; then
        remaining=0
    fi
    spin_char="${spinner:$((tick % 4)):1}"

    local bar=""
    local i
    for ((i=0; i<filled; i++)); do bar+="#"; done
    for ((i=filled; i<width; i++)); do bar+="-"; done

    local elapsed_text eta_text
    elapsed_text=$(format_duration "${elapsed_sec}")
    if [[ "${eta_sec}" -ge 0 ]]; then
        eta_text=$(format_duration "${eta_sec}")
    else
        eta_text="--:--:--"
    fi

    local line="[progress ${spin_char}] [${bar}] ${pct}% (${done}/${total}) remaining=${remaining} ok=${ok_count} already_done=${done_count} skip=${skip_count} fail=${fail_count} elapsed=${elapsed_text} eta=${eta_text} avg=${avg_sec_per_clip}s/clip"
    if [[ "${IS_TTY}" -eq 1 ]]; then
        printf '\r%s' "${line}"
    else
        printf '%s\n' "${line}"
    fi
}

init_counter() {
    local file="$1"
    echo "0" > "${file}"
}

inc_counter() {
    local file="$1"
    (
        flock -x 200
        local cur=0
        cur=$(cat "${file}" 2>/dev/null || echo "0")
        echo $((cur + 1)) > "${file}"
    ) 200>"${COUNTER_LOCK_FILE}"
}

read_counter() {
    local file="$1"
    cat "${file}" 2>/dev/null || echo "0"
}

monitor_progress() {
    local total="$1"
    local tick=0
    local start_ts
    start_ts=$(date +%s)

    while true; do
        local ok_count done_count skip_count fail_count done_all
        local now_ts elapsed_sec eta_sec avg_sec_per_clip remaining
        ok_count=$(read_counter "${COUNTER_OK_FILE}")
        done_count=$(read_counter "${COUNTER_ALREADY_DONE_FILE}")
        skip_count=$(read_counter "${COUNTER_SKIP_FILE}")
        fail_count=$(read_counter "${COUNTER_FAIL_FILE}")
        done_all=$((ok_count + done_count + skip_count + fail_count))

        now_ts=$(date +%s)
        elapsed_sec=$((now_ts - start_ts))
        remaining=$((total - done_all))
        if [[ "${remaining}" -lt 0 ]]; then
            remaining=0
        fi

        eta_sec=-1
        avg_sec_per_clip="-"
        if [[ "${done_all}" -gt 0 ]]; then
            # Use ms precision to avoid integer division causing static ETA
            local avg_ms=$(( (elapsed_sec * 1000) / done_all ))
            if [[ "${avg_ms}" -lt 0 ]]; then avg_ms=0; fi

            eta_sec=$(( (avg_ms * remaining) / 1000 ))

            # Display with 2 decimal places
            avg_sec_per_clip=$(printf "%d.%02d" $((avg_ms / 1000)) $(( (avg_ms % 1000) / 10 )))
        fi

        print_progress "${done_all}" "${total}" "${ok_count}" "${done_count}" "${skip_count}" "${fail_count}" "${tick}" "${elapsed_sec}" "${eta_sec}" "${avg_sec_per_clip}"

        local eta_text
        if [[ "${eta_sec}" -ge 0 ]]; then
            eta_text=$(format_duration "${eta_sec}")
        else
            eta_text="--:--:--"
        fi
        printf '[%s] done=%d/%d ok=%d already_done=%d skip=%d fail=%d elapsed=%s eta=%s avg=%ss_per_clip\n' \
            "$(date '+%F %T')" "${done_all}" "${total}" "${ok_count}" "${done_count}" "${skip_count}" "${fail_count}" "$(format_duration "${elapsed_sec}")" "${eta_text}" "${avg_sec_per_clip}" >> "${PROGRESS_LOG_FILE}"

        if [[ "${done_all}" -ge "${total}" ]]; then
            break
        fi

        tick=$((tick + 1))
        sleep "${PROGRESS_INTERVAL_SEC}"
    done
}

build_full_clip_list() {
    local full_list_file="$1"

    if [[ -n "${INPUT_LIST}" ]]; then
        if [[ ! -f "${INPUT_LIST}" ]]; then
            echo "[err] input list not found: ${INPUT_LIST}" >&2
            exit 1
        fi
        : > "${full_list_file}"
        while IFS= read -r line || [[ -n "${line}" ]]; do
            line="${line//$'\r'/}"
            line="${line#"${line%%[![:space:]]*}"}"
            line="${line%"${line##*[![:space:]]}"}"
            [[ -z "${line}" ]] && continue
            [[ "${line}" == \#* ]] && continue
            if [[ "${line}" == */* ]]; then
                basename "${line}" >> "${full_list_file}"
            else
                echo "${line}" >> "${full_list_file}"
            fi
        done < "${INPUT_LIST}"
    else
        if [[ ! -d "${INPAINT_ROOT}" ]]; then
            echo "[err] INPAINT_ROOT not found: ${INPAINT_ROOT}" >&2
            exit 1
        fi
        find "${INPAINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort > "${full_list_file}"
    fi
}

split_shard() {
    local full_list_file="$1"
    local shard_list_file="$2"
    awk -v shard_id="${SHARD_ID}" -v total_shards="${TOTAL_SHARDS}" 'NF{ if ((NR-1) % total_shards == (shard_id-1)) print $0 }' "${full_list_file}" > "${shard_list_file}"
}

split_workers() {
    local shard_list_file="$1"
    local worker_dir="$2"
    local worker_count="$3"

    mkdir -p "${worker_dir}"
    rm -f "${worker_dir}"/worker_*.txt

    awk -v worker_count="${worker_count}" -v worker_dir="${worker_dir}" 'NF{
        wid = (NR - 1) % worker_count
        file = sprintf("%s/worker_%02d.txt", worker_dir, wid)
        print $0 >> file
    }' "${shard_list_file}"

    local i
    for ((i=0; i<worker_count; i++)); do
        local wf
        wf=$(printf '%s/worker_%02d.txt' "${worker_dir}" "${i}")
        [[ -f "${wf}" ]] || : > "${wf}"
    done
}

process_worker() {
    local worker_id="$1"
    local worker_file="$2"

    if [[ "${NUM_GPUS}" -gt 0 ]]; then
        local gpu_id=$((worker_id % NUM_GPUS))
        export CUDA_VISIBLE_DEVICES="${gpu_id}"
        log_msg "worker=${worker_id} gpu=${gpu_id} list=${worker_file}"
    else
        export CUDA_VISIBLE_DEVICES=""
        log_msg "worker=${worker_id} gpu=none (cpu-only) list=${worker_file}"
    fi

    # Use cat and pipe to avoid potential race conditions with direct file redirection
    cat "${worker_file}" | while IFS=$'\n' read -r clip; do
        clip="${clip//$'\r'/}" # Remove CR
        [[ -z "${clip}" ]] && continue
        render_one "${clip}"
    done

    log_msg "worker=${worker_id} done"
}


log_summary_skip() {
    local clip="$1"
    local reason="$2"
    (
        flock -x 202
        printf '%s %s\n' "${clip}" "${reason}" >> "${SUMMARY_SKIP_FILE}"
    ) 202>"${SUMMARY_LOCK_FILE}"
}

log_summary_ok() {
    local clip="$1"
    (
        flock -x 202
        printf '%s\n' "${clip}" >> "${SUMMARY_OK_FILE}"
    ) 202>"${SUMMARY_LOCK_FILE}"
}

log_summary_fail() {
    local clip="$1"
    local reason="$2"
    (
        flock -x 202
        printf '%s %s\n' "${clip}" "${reason}" >> "${SUMMARY_FAIL_FILE}"
    ) 202>"${SUMMARY_LOCK_FILE}"
}

log_summary_missing_video() {
    local clip="$1"
    (
        flock -x 202
        printf '%s\n' "${clip}" >> "${SUMMARY_MISSING_VIDEO_FILE}"
    ) 202>"${SUMMARY_LOCK_FILE}"
}

render_one() {
    local clip="$1"
    
    # Sanity check: clip name should be reasonable length (e.g. UUID-like > 20 chars)
    if [[ ${#clip} -lt 20 ]]; then
        log_msg "skip_invalid_name clip='${clip}' len=${#clip}"
        log_summary_skip "${clip}" "invalid_name"
        return 0
    fi

    local video_path="${VIDEO_ROOT}/${clip}/video.mp4"
    local rgb_path="${INPAINT_ROOT}/${clip}/hand_inpaint.png"
    local depth_path="${INPAINT_ROOT}/${clip}/depth_first_frame.npy"
    local pose_dir="${POSE_ROOT}/${clip}"
    local intrinsics_path="${INPAINT_ROOT}/${clip}/intrinsics_first_frame.npy"

    local out_dir="${OUT_ROOT}/${clip}"
    mkdir -p "${out_dir}"

    local output_render="${out_dir}/render.mp4"
    local output_overlay="${out_dir}/overlay.mp4"
    if [[ "${MASK_MODE}" == "1" ]]; then
        output_render="${out_dir}/render_mask.mp4"
        output_overlay="${out_dir}/overlay_mask.mp4"
    fi
    local log_file="${out_dir}/run.log"
    local status_file="${out_dir}/status.txt"

    log_msg "start clip=${clip}"

    if [[ "${SKIP_DONE}" == "1" ]] && [[ -f "${output_overlay}" ]]; then
        # Mask mode: sanity-check first frame (all-black means bad render)
        if [[ "${MASK_MODE}" == "1" ]]; then
            # Valid mask video: white background, black points (bg=1, points=0).
            # Max grayscale of first frame should be ~255 for white; ~0 if render failed.
            local max_val
            max_val=$(ffmpeg -i "${output_render}" -vframes 1 -f image2pipe -vcodec rawvideo -pix_fmt gray - 2>/dev/null | od -An -t u1 | tr -s ' ' '\n' | sort -nr | head -n 1)            
            
            # max_val < 200 => invalid (not white enough); re-run
            if [[ -z "${max_val}" ]] || [[ "${max_val}" -lt 200 ]]; then
                log_msg "re-run bad mask video detected (no white background): ${output_render} Max_val=${max_val}"
                # fall through: treat as not done and re-render
            else
                echo "already_done" > "${status_file}"
                inc_counter "${COUNTER_ALREADY_DONE_FILE}"
                log_msg "already_done clip=${clip}"
                log_summary_ok "${clip}"
                return 0
            fi
        else
            echo "already_done" > "${status_file}"
            inc_counter "${COUNTER_ALREADY_DONE_FILE}"
            log_msg "already_done clip=${clip}"
            log_summary_ok "${clip}"
            return 0
        fi
    fi

    if [[ ! -f "${video_path}" ]]; then echo "skip_video" > "${status_file}"; inc_counter "${COUNTER_SKIP_FILE}"; log_msg "skip_video clip=${clip}"; log_summary_skip "${clip}" "skip_video"; log_summary_missing_video "${clip}"; return 0; fi
    if [[ ! -f "${rgb_path}" ]]; then echo "skip_rgb" > "${status_file}"; inc_counter "${COUNTER_SKIP_FILE}"; log_msg "skip_rgb clip=${clip}"; log_summary_skip "${clip}" "skip_rgb"; return 0; fi
    if [[ ! -f "${depth_path}" ]]; then echo "skip_depth" > "${status_file}"; inc_counter "${COUNTER_SKIP_FILE}"; log_msg "skip_depth clip=${clip}"; log_summary_skip "${clip}" "skip_depth"; return 0; fi
    if [[ ! -f "${intrinsics_path}" ]]; then echo "skip_intrinsics" > "${status_file}"; inc_counter "${COUNTER_SKIP_FILE}"; log_msg "skip_intrinsics clip=${clip}"; log_summary_skip "${clip}" "skip_intrinsics"; return 0; fi
    if [[ ! -d "${pose_dir}" ]]; then echo "skip_pose" > "${status_file}"; inc_counter "${COUNTER_SKIP_FILE}"; log_msg "skip_pose clip=${clip}"; log_summary_skip "${clip}" "skip_pose"; return 0; fi

    set +e
    "${PYTHON_CMD[@]}" "${PYTHON_SCRIPT}" \
        --video_path "${video_path}" \
        --pose_dir "${pose_dir}" \
        --intrinsics_path "${intrinsics_path}" \
        --rgb_path "${rgb_path}" \
        --depth_path "${depth_path}" \
        --output_video "${output_render}" \
        --overlay_video "${output_overlay}" \
        --point_size "${POINT_SIZE}" \
        --fps "${FPS}" \
        $([[ "${MASK_MODE}" == "1" ]] && echo "--mask_mode") < /dev/null >"${log_file}" 2>&1
    local rc=$?
    set -e

    if [[ ${rc} -ne 0 ]]; then
        echo "fail:${rc}" > "${status_file}"
        inc_counter "${COUNTER_FAIL_FILE}"
        log_msg "fail clip=${clip} rc=${rc} log=${log_file}"
        log_summary_fail "${clip}" "rc=${rc}"
        return 0
    fi

    echo "ok" > "${status_file}"
    inc_counter "${COUNTER_OK_FILE}"
    log_msg "ok clip=${clip} overlay=${output_overlay}"
    log_summary_ok "${clip}"
}


WORKER_DIR="${SHARD_DIR}/workers_${SHARD_ID}_of_${TOTAL_SHARDS}_${RUN_TS}"
INPUT_SOURCE="${INPUT_LIST}"
if [[ -z "${INPUT_SOURCE}" ]]; then
    INPUT_SOURCE="${INPAINT_ROOT}"
fi

echo "[info] Running split_data.py to prepare worker lists..."
"${PYTHON_CMD[@]}" "${SCRIPT_DIR}/split_data.py" \
    "${INPUT_SOURCE}" \
    "${SHARD_ID}" \
    "${TOTAL_SHARDS}" \
    "${MAX_JOBS}" \
    "${WORKER_DIR}"

sync
sleep 2

SHARD_LIST_FILE="${SHARD_DIR}/shard_list_${SHARD_ID}.txt"
# Concatenate all worker files to create the full shard list for summary
if [[ -d "${WORKER_DIR}" ]]; then
    find "${WORKER_DIR}" -name "worker_*.txt" -exec cat {} + > "${SHARD_LIST_FILE}"
fi

SHARD_CLIPS=0
# Sum up lines in worker files to get SHARD_CLIPS (approx or exact)
if [[ -d "${WORKER_DIR}" ]]; then
    SHARD_CLIPS=$(find "${WORKER_DIR}" -name "worker_*.txt" -exec cat {} + | wc -l)
fi

WORKER_COUNT="${MAX_JOBS}"
# If fewer clips than workers, we still have WORKER_COUNT files, some empty.
# We can just run all WORKER_COUNT workers.

echo "[info] shard ${SHARD_ID}/${TOTAL_SHARDS}: ${SHARD_CLIPS} clips (approx)"
echo "[info] max parallel jobs: ${MAX_JOBS}"
echo "[info] num_gpus: ${NUM_GPUS}"
echo "[info] render_platform: ${PYOPENGL_PLATFORM}"
echo "[info] shard log: ${SHARD_LOG_FILE}"
echo "[info] progress log: ${PROGRESS_LOG_FILE}"
log_msg "start shard=${SHARD_ID}/${TOTAL_SHARDS} shard_clips=${SHARD_CLIPS} max_jobs=${MAX_JOBS}"

if [[ "${SHARD_CLIPS}" -eq 0 ]]; then
    echo "[done] no clips for this shard"
    log_msg "done_no_clips shard=${SHARD_ID}/${TOTAL_SHARDS}"
    exit 0
fi

active_jobs=0
COUNTER_DIR="${SHARD_DIR}/counter_${SHARD_ID}_of_${TOTAL_SHARDS}_${RUN_TS}"
mkdir -p "${COUNTER_DIR}"
COUNTER_LOCK_FILE="${COUNTER_DIR}/counter.lock"
COUNTER_OK_FILE="${COUNTER_DIR}/ok.txt"
COUNTER_ALREADY_DONE_FILE="${COUNTER_DIR}/already_done.txt"
COUNTER_SKIP_FILE="${COUNTER_DIR}/skip.txt"
COUNTER_FAIL_FILE="${COUNTER_DIR}/fail.txt"

# Create Summary Files
SUMMARY_DIR="${SHARD_DIR}/summaries_${SHARD_ID}_of_${TOTAL_SHARDS}_${RUN_TS}"
mkdir -p "${SUMMARY_DIR}"
SUMMARY_LOCK_FILE="${SUMMARY_DIR}/summary.lock"
SUMMARY_OK_FILE="${SUMMARY_DIR}/processed_ok.txt"
SUMMARY_SKIP_FILE="${SUMMARY_DIR}/processed_skip.txt"
SUMMARY_FAIL_FILE="${SUMMARY_DIR}/processed_fail.txt"
SUMMARY_MISSING_VIDEO_FILE="${SUMMARY_DIR}/missing_videos.txt"
touch "${SUMMARY_OK_FILE}" "${SUMMARY_SKIP_FILE}" "${SUMMARY_FAIL_FILE}" "${SUMMARY_MISSING_VIDEO_FILE}"

init_counter "${COUNTER_OK_FILE}"
init_counter "${COUNTER_ALREADY_DONE_FILE}"
init_counter "${COUNTER_SKIP_FILE}"
init_counter "${COUNTER_FAIL_FILE}"

echo "[info] progress bar refresh interval: ${PROGRESS_INTERVAL_SEC}s"
echo "[info] running shared progress monitor..."
monitor_progress "${SHARD_CLIPS}" &
MONITOR_PID=$!

echo "[info] split shard into ${WORKER_COUNT} worker lists: ${WORKER_DIR}"
log_msg "split_workers worker_count=${WORKER_COUNT} worker_dir=${WORKER_DIR}"

for ((wid=0; wid<WORKER_COUNT; wid++)); do
    wf=$(printf '%s/worker_%02d.txt' "${WORKER_DIR}" "${wid}")
    if [[ -s "${wf}" ]]; then
        process_worker "${wid}" "${wf}" &
        active_jobs=$((active_jobs + 1))
    fi
done

while [[ "${active_jobs}" -gt 0 ]]; do
    wait -n || true
    active_jobs=$((active_jobs - 1))
done

wait "${MONITOR_PID}" || true
if [[ "${IS_TTY}" -eq 1 ]]; then
    echo
fi

OK_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
DONE_COUNT=0

while IFS= read -r clip; do
    [[ -z "${clip}" ]] && continue
    status_file="${OUT_ROOT}/${clip}/status.txt"
    if [[ ! -f "${status_file}" ]]; then
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    status=$(cat "${status_file}")
    case "${status}" in
        ok)
            OK_COUNT=$((OK_COUNT + 1))
            ;;
        already_done)
            DONE_COUNT=$((DONE_COUNT + 1))
            ;;
        skip_*)
            SKIP_COUNT=$((SKIP_COUNT + 1))
            ;;
        fail:*)
            FAIL_COUNT=$((FAIL_COUNT + 1))
            ;;
        *)
            FAIL_COUNT=$((FAIL_COUNT + 1))
            ;;
    esac
done < "${SHARD_LIST_FILE}"

echo "[summary] shard ${SHARD_ID}/${TOTAL_SHARDS} done"
echo "          ok=${OK_COUNT}, already_done=${DONE_COUNT}, skip=${SKIP_COUNT}, fail=${FAIL_COUNT}"
log_msg "summary shard=${SHARD_ID}/${TOTAL_SHARDS} ok=${OK_COUNT} already_done=${DONE_COUNT} skip=${SKIP_COUNT} fail=${FAIL_COUNT}"
