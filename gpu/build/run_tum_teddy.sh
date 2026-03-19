#!/usr/bin/env bash
# run_tum_teddy.sh
# Run KinectFusion on the TUM RGB-D freiburg1_teddy dataset.
#
# Usage:
#   ./run_tum_teddy.sh [--frames N] [--skip K] [--mode MODE] [--no-mask] [--remask]
#
# Options:
#   --frames N      process only first N frames
#   --skip K        process every K-th frame (default: 1)
#   --mode MODE     preprocessing mode passed to mask_teddy_depth.py:
#                     none      — no masking (raw depth)
#                     maskrcnn  — Mask R-CNN instance segmentation (class-specific)
#                     depth     — depth-only foreground extraction (default, general)
#   --no-mask       skip mask_teddy_depth.py entirely, use raw depth.txt
#   --remask        re-run mask_teddy_depth.py even if depth_masked/ already exists
#
# Output:
#   output_tum.ply  — reconstructed mesh (ASCII PLY)

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
DATASET="rgbd_dataset_freiburg1_teddy"
CONFIG="kinfu_tum_freiburg1.cfg"
MASKED_TXT="$DATASET/depth_masked.txt"

# Default run options
FRAMES=""
SKIP=1
USE_MASK=true
REMASK=true
MASK_MODE="depth"

# Parse optional overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --frames)  FRAMES="--frames $2"; shift 2 ;;
        --skip)    SKIP="$2";            shift 2 ;;
        --mode)    MASK_MODE="$2";       shift 2 ;;
        --no-mask) USE_MASK=false;       shift   ;;
        --remask)  REMASK=true;          shift   ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# -----------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use system NVIDIA OpenCL ICD instead of conda stub
export LD_PRELOAD=/lib/x86_64-linux-gnu/libOpenCL.so.1

# -----------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------
if [[ ! -d "$DATASET" ]]; then
    echo "ERROR: Dataset not found: $SCRIPT_DIR/$DATASET"
    echo "       Extract rgbd_dataset_freiburg1_teddy.tgz first:"
    echo "       tar xzf rgbd_dataset_freiburg1_teddy.tgz"
    exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config not found: $SCRIPT_DIR/$CONFIG"
    exit 1
fi

# -----------------------------------------------------------------------
# Teddy segmentation (mask_teddy_depth.py)
# -----------------------------------------------------------------------
if $USE_MASK; then
    if $REMASK || [[ ! -f "$MASKED_TXT" ]]; then
        echo "Running foreground extraction (mask_teddy_depth.py --mode $MASK_MODE)…"
        python3 mask_teddy_depth.py "$DATASET" --mode "$MASK_MODE"
        if [[ $? -ne 0 ]]; then
            echo "ERROR: mask_teddy_depth.py failed."
            exit 1
        fi
    else
        echo "Using existing masked depth: $MASKED_TXT"
        echo "  (pass --remask to regenerate)"
    fi
    DEPTH_ARG="--depth-txt $MASKED_TXT"
else
    echo "Skipping mask — using raw depth.txt"
    DEPTH_ARG=""
fi

# -----------------------------------------------------------------------
# Run KinFu
# -----------------------------------------------------------------------
echo ""
echo "Dataset : $DATASET"
echo "Config  : $CONFIG"
echo "Mode    : ${MASK_MODE}"
echo "Depth   : ${DEPTH_ARG:-(default depth.txt)}"
echo "Skip    : $SKIP"
echo ""

exec ./run_kinfu_tum "$DATASET" \
     --config  "$CONFIG" \
     --skip    "$SKIP" \
     $DEPTH_ARG \
     $FRAMES
