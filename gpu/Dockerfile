# =============================================================================
# KinectFusion OpenCL — Docker image
#
# Build (from gpu/ directory):
#   docker build -f build/Dockerfile -t kinfu-opencl .
#
# Build with CUDA-enabled PyTorch (for maskrcnn mode on GPU):
#   docker build -f build/Dockerfile \
#     --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 \
#     -t kinfu-opencl .
#
# Run (CPU only):
#   docker run --rm -it \
#     -v /path/to/dataset:/data \
#     kinfu-opencl
#
# Run (NVIDIA GPU — requires nvidia-container-toolkit on host):
#   docker run --rm -it --gpus all \
#     -v /path/to/dataset:/data \
#     kinfu-opencl
# =============================================================================

FROM ubuntu:22.04

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV WORKDIR=/opt/kinfu

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        # Build tools
        build-essential \
        cmake \
        # KinFu C++ dependencies
        libboost-dev \
        libeigen3-dev \
        libpng-dev \
        # OpenCL (headers + ICD loader)
        opencl-headers \
        ocl-icd-opencl-dev \
        ocl-icd-libopencl1 \
        # Python runtime
        python3 \
        python3-pip \
        python3-dev \
        # Utilities
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Ensure libOpenCL.so.1.0.0 exists (ocl-icd provides .so.1.0.0)
# Create a symlink if it doesn't exist yet for CMakeLists compatibility
RUN ldconfig && \
    if [ ! -f /usr/lib/x86_64-linux-gnu/libOpenCL.so.1.0.0 ]; then \
        ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 \
               /usr/lib/x86_64-linux-gnu/libOpenCL.so.1.0.0; \
    fi

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
# Core preprocessing deps (all modes)
RUN pip3 install --no-cache-dir \
        numpy \
        opencv-python-headless \
        pillow

# PyTorch — CPU by default; pass --build-arg TORCH_INDEX=.../cu128 for CUDA
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip3 install --no-cache-dir \
        torch \
        torchvision \
        --index-url ${TORCH_INDEX}

# ---------------------------------------------------------------------------
# Copy source (build context = gpu/ directory)
# ---------------------------------------------------------------------------
WORKDIR ${WORKDIR}/gpu
COPY . .

# ---------------------------------------------------------------------------
# Build C++ library and executables
# ---------------------------------------------------------------------------
WORKDIR ${WORKDIR}/gpu/build

RUN cmake . \
        -DCONDA_CELLSEG1=/usr \
        -DCONDA_GLUE=/usr \
        -DOpenCL_LIBRARY=/usr/lib/x86_64-linux-gnu/libOpenCL.so \
    && make -j$(nproc)

# ---------------------------------------------------------------------------
# Runtime environment
# ---------------------------------------------------------------------------
# Mount point for TUM dataset / output
VOLUME ["/data"]

# LD_PRELOAD for NVIDIA OpenCL ICD (set at runtime when GPU is available)
# When running with --gpus all, the NVIDIA driver provides:
#   /usr/lib/x86_64-linux-gnu/libOpenCL.so.1  →  NVIDIA ICD
# The ocl-icd loader inside the container dispatches to it automatically.
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libOpenCL.so

WORKDIR ${WORKDIR}/gpu/build

CMD ["/bin/bash"]
