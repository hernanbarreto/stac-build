#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# GauS-SLAM Pipeline Launcher
# Orchestrates: DA3 depth+poses → TUM bridge → GauS-SLAM
#
# Uses TWO conda environments:
#   - da3:        DA3 NESTED-GIANT-LARGE-1.1 inference
#   - gaus_slam:  GauS-SLAM Gaussian Surfel SLAM
#
# Same pattern as run_da3.sh / run_mapanything.sh
#
# Hernán Barreto — Ingerop IN3 Session IV — STAC
# ─────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Parse arguments ──
IMAGE_DIR=""
OUTPUT_DIR=""
SELECTED_FRAMES=""
DA3_MODEL="depth-anything/DA3NESTED-GIANT-LARGE-1.1"
PROCESS_RES=504
USE_RAY_POSE=""
GAUS_CONFIG=""
DEVICE="auto"
DEPTH_SOURCE="lidar"       # lidar | da3 | hybrid
STRAY_DIR=""               # requerido para lidar y hybrid
LIDAR_TRUST_M="5.0"
CALIB_MIN_DEPTH="0.3"
DA3_MAX_RANGE="20.0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image_dir)      IMAGE_DIR="$2";      shift 2 ;;
        --output_dir)     OUTPUT_DIR="$2";     shift 2 ;;
        --selected_frames) SELECTED_FRAMES="$2"; shift 2 ;;
        --da3_model)      DA3_MODEL="$2";      shift 2 ;;
        --process_res)    PROCESS_RES="$2";    shift 2 ;;
        --use_ray_pose)   USE_RAY_POSE="--use_ray_pose"; shift ;;
        --gaus_config)    GAUS_CONFIG="$2";    shift 2 ;;
        --device)         DEVICE="$2";         shift 2 ;;
        --frame_step)     FRAME_STEP="$2";     shift 2 ;;
        --depth_source)   DEPTH_SOURCE="$2";   shift 2 ;;
        --stray_dir)      STRAY_DIR="$2";      shift 2 ;;
        --lidar_trust_m)  LIDAR_TRUST_M="$2";  shift 2 ;;
        *) echo "[GauS-SLAM] Unknown argument: $1"; shift ;;
    esac
done

if [ -z "$IMAGE_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "[GauS-SLAM] ❌ Usage: $0 --image_dir DIR --output_dir DIR [--selected_frames PATH]"
    exit 1
fi

# ── Conda setup ──
CONDA_ROOT="${CONDA_ROOT:-/home/hernan/miniforge3}"
DA3_ENV="${DA3_CONDA_ENV:-da3}"
GAUS_ENV="${GAUS_CONDA_ENV:-gaus_slam}"

# Prefer conda when it's available (the pod is itself a container, so /.dockerenv
# exists there too). conda.sh must be sourced so the `conda activate` calls below
# (DA3_ENV / GAUS_ENV) work.
if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
elif [ -f "/.dockerenv" ]; then
    echo "[GauS-SLAM] Running in Docker mode (no conda)"
else
    echo "[GauS-SLAM] ⚠️ Conda not found at ${CONDA_ROOT}"
    exit 1
fi

export PYTHONUNBUFFERED=1

# ── Directory structure ──
DA3_OUTPUT="${OUTPUT_DIR}/da3_full"
TUM_BRIDGE="${OUTPUT_DIR}/tum_bridge"
GAUS_OUTPUT="${OUTPUT_DIR}/gaus_slam_run"

mkdir -p "${DA3_OUTPUT}" "${TUM_BRIDGE}" "${GAUS_OUTPUT}"

# ════════════════════════════════════════════════════════════════
# STEP 1: Depth Acquisition (varía según depth_source)
#   lidar  → da3_full/ ya está pre-poblado por map_worker (skip)
#   da3    → extract_da3_full.py (DA3 depth + DA3 poses)
#   hybrid → extract_da3_full.py + calibrate_depth_hybrid.py
# ════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  STEP 1/3: Depth Acquisition (mode: ${DEPTH_SOURCE})                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"

if [ "${DEPTH_SOURCE}" = "lidar" ]; then
    echo "[GauS-SLAM] LiDAR mode — da3_full/ pre-poblado, skipping DA3 inference"

elif [ -f "${DA3_OUTPUT}/da3_manifest.json" ]; then
    # Resume: da3_full/ already populated (DA3 + calibration already done)
    # Skip both DA3 extraction AND calibration to avoid double-calibrating
    echo "[GauS-SLAM] ✅ da3_manifest.json found — skipping DA3 + calibration (resume)"

elif [ "${DEPTH_SOURCE}" = "da3" ] || [ "${DEPTH_SOURCE}" = "hybrid" ]; then
    conda activate "${DA3_ENV}"
    export PYTHONPATH="${PROJECT_ROOT}/vendor/depth-anything-3/src:${PYTHONPATH}"

    DA3_CMD=(
        python -u "${SCRIPT_DIR}/extract_da3_full.py"
        --image_dir "${IMAGE_DIR}"
        --output_dir "${DA3_OUTPUT}"
        --model "${DA3_MODEL}"
        --process_res "${PROCESS_RES}"
        --device "${DEVICE}"
    )

    if [ -n "${SELECTED_FRAMES}" ] && [ -f "${SELECTED_FRAMES}" ]; then
        DA3_CMD+=(--selected_frames "${SELECTED_FRAMES}")
    fi
    if [ -n "${USE_RAY_POSE}" ]; then
        DA3_CMD+=("${USE_RAY_POSE}")
    fi

    # Hybrid mode: process frame-by-frame (only depth needed, poses come from ARKit)
    # This drops RAM usage from ~16GB (N=250 multi-view) to ~6GB (N=1 monocular)
    if [ "${DEPTH_SOURCE}" = "hybrid" ]; then
        DA3_CMD+=(--per_frame)
        echo "[GauS-SLAM] Hybrid mode: using --per_frame (poses will come from ARKit)"
    fi

    echo "[GauS-SLAM] Running: ${DA3_CMD[*]}"
    "${DA3_CMD[@]}"
    echo "[GauS-SLAM] ✅ DA3 extraction complete"

    # ── STEP 1b: Calibración LiDAR (solo hybrid) ──────────────────────────
    if [ "${DEPTH_SOURCE}" = "hybrid" ]; then
        echo ""
        echo "╔══════════════════════════════════════════════════════════════╗"
        echo "║  STEP 1b: LiDAR Depth Calibration                          ║"
        echo "╚══════════════════════════════════════════════════════════════╝"

        if [ -z "${STRAY_DIR}" ]; then
            echo "[GauS-SLAM] ❌ --stray_dir requerido para depth_source=hybrid"
            exit 1
        fi

        python -u "${SCRIPT_DIR}/calibrate_depth_hybrid.py" \
            --da3_dir  "${DA3_OUTPUT}" \
            --stray_dir "${STRAY_DIR}" \
            --lidar_trust_m  "${LIDAR_TRUST_M}" \
            --calib_min_depth "${CALIB_MIN_DEPTH}" \
            --da3_max_range   "${DA3_MAX_RANGE}"

        echo "[GauS-SLAM] ✅ Hybrid depth calibration complete"
    fi
else
    echo "[GauS-SLAM] ❌ depth_source desconocido: ${DEPTH_SOURCE}"
    exit 1
fi

# ════════════════════════════════════════════════════════════════
# STEP 2: TUM-RGBD Format Bridge (no GPU needed)
# Converts DA3 output to TUM-RGBD directory structure
# ════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  STEP 2/3: TUM-RGBD Format Bridge                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"

python -u "${SCRIPT_DIR}/bridge_tum_format.py" \
    --frames_dir "${IMAGE_DIR}" \
    --da3_dir "${DA3_OUTPUT}" \
    --output_dir "${TUM_BRIDGE}" \
    --use_processed_images \
    ${SELECTED_FRAMES:+--selected_frames "${SELECTED_FRAMES}"} \
    ${FRAME_STEP:+--frame_step "${FRAME_STEP}"}

echo "[GauS-SLAM] ✅ TUM bridge complete"

# ════════════════════════════════════════════════════════════════
# STEP 3: GauS-SLAM (conda: gaus_slam)
# Runs Gaussian Surfel SLAM on the TUM-RGBD data
# ════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  STEP 3/3: GauS-SLAM Reconstruction                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"

conda activate "${GAUS_ENV}"

GAUS_SLAM_DIR="${PROJECT_ROOT}/vendor/gaus-slam"

# Siempre regenerar gaus_slam_config.py — evita usar config stale de corridas anteriores
GAUS_CONFIG="${TUM_BRIDGE}/gaus_slam_config.py"

# Determine device for GauS-SLAM (use CUDA if available)
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GAUS_DEVICE="cuda"
else
    GAUS_DEVICE="cpu"
fi

python -u "${SCRIPT_DIR}/generate_gaus_config.py" \
    --tum_dir "${TUM_BRIDGE}" \
    --output "${GAUS_CONFIG}" \
    --device "${GAUS_DEVICE}"

echo "[GauS-SLAM] Config: ${GAUS_CONFIG}"
echo "[GauS-SLAM] TUM data: ${TUM_BRIDGE}"

# Run GauS-SLAM
cd "${GAUS_SLAM_DIR}"
export PYTHONPATH="${GAUS_SLAM_DIR}:${PYTHONPATH}"

# CUDA extensions (simple_knn, gaus_2dgs_rasterization) need PyTorch's libc10.so
TORCH_LIB=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH}"

python -u scripts/gaus_mp.py "${GAUS_CONFIG}" \
    2>&1 | while IFS= read -r line; do
    echo "$line"
    # Emit progress markers for map_worker to parse
    if echo "$line" | grep -q "Frame"; then
        echo "[Progress]: $line"
    fi
done

GAUS_EXIT=${PIPESTATUS[0]}
if [ $GAUS_EXIT -ne 0 ]; then
    echo "[GauS-SLAM] ❌ GauS-SLAM failed with exit code $GAUS_EXIT"
    exit $GAUS_EXIT
fi

echo ""
echo "[GauS-SLAM] ✅ Pipeline complete"
echo "  DA3 output:    ${DA3_OUTPUT}"
echo "  TUM bridge:    ${TUM_BRIDGE}"
echo "  GauS-SLAM out: ${GAUS_OUTPUT}"
