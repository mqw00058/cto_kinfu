#!/usr/bin/env bash
# run_kinfu.sh — wrapper to run KinectFusion with correct OpenCL library
#
# The conda environment provides a stub libOpenCL.so that has no ICD support.
# We preload the system NVIDIA OpenCL implementation instead.
#
# Usage:
#   ./run_kinfu.sh [--frames N]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure NVIDIA OpenCL is used instead of the conda stub
export LD_PRELOAD=/lib/x86_64-linux-gnu/libOpenCL.so.1

exec ./run_kinfu_oni "$@"
