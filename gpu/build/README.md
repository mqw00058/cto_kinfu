# KinectFusion OpenCL — Linux Standalone Build

A standalone Linux build of the **LG Electronics OpenCL port** of KinectFusion, derived from `pcl-master/gpu/kinfu_opencl`. This build produces a static library (`libkinfu_opencl.a`) and runner executables that exercise the full TSDF + ICP + Marching Cubes pipeline without requiring PCL, VTK, or visualization dependencies.

---

## Overview

KinectFusion reconstructs a 3-D surface from a live depth-camera stream by fusing depth measurements into a Truncated Signed Distance Function (TSDF) volume and tracking the camera pose via Iterative Closest Point (ICP). This port replaces the original CUDA kernels with OpenCL kernels, making it compatible with any OpenCL 1.2+ GPU.

### Pipeline

```
Depth Frame (uint16 mm)
    ↓
[Preprocessing]           (mask_teddy_depth.py — optional)
    ↓
Bilateral Filter          (cl/bilateral.cl)
    ↓
Vertex / Normal Maps      (cl/maps.cl)
    ↓
ICP Pose Estimation       (cl/estimate_combined.cl)
    ↓
TSDF Integration          (cl/tsdf.cl)
    ↓
Ray Casting               (cl/ray_caster.cl)
    ↓
Marching Cubes            (cl/marchingcube.cl)
    ↓
PLY / STL Mesh
```

---

## Results

Tested on **NVIDIA RTX 5000 Ada Generation** (OpenCL 3.0, CUDA 12.8):

| Item | Value |
|---|---|
| Input | 200 synthetic depth frames (sphere scene, VGA 640×480) |
| Output | `output.stl` — ASCII STL |
| TSDF volume | 512³ voxels |
| Voxel size | ~3.9 mm |

TUM RGB-D freiburg1_teddy results (1418 frames):

| Preprocessing mode | Triangles | Notes |
|---|---|---|
| None (raw depth) | 447,182 | full room geometry |
| `maskrcnn` | 422,402 | teddy-only, class-specific |
| `depth` | 518,736 | foreground object, general-purpose |

---

## Directory Layout

```
gpu/
├── Dockerfile                  ← Docker image definition
├── build/                      ← This directory (CMake build + runner source)
│   ├── CMakeLists.txt          ← Standalone CMake build
│   ├── run_kinfu_oni.cpp       ← Synthetic-depth runner (no OpenNI2 needed)
│   ├── run_kinfu_tum.cpp       ← TUM RGB-D dataset runner
│   ├── run_kinfu.sh            ← Launcher for synthetic runner
│   ├── run_tum_teddy.sh        ← Full pipeline launcher for TUM dataset
│   ├── mask_teddy_depth.py     ← Depth preprocessing / foreground extraction
│   ├── dump_depth_ply.py       ← Export per-frame point clouds as PLY
│   ├── libkinfu_opencl.a       ← Built static library
│   ├── run_kinfu_oni           ← Built executable (synthetic)
│   ├── run_kinfu_tum           ← Built executable (TUM RGB-D)
│   └── include/                ← Stub headers (Linux/PCL stubs)
├── include/pcl/gpu/kinfu/      ← Public API headers
│   ├── kinfu.h
│   ├── kinfu_config.h
│   ├── tsdf_volume.h
│   └── ...
├── src/                        ← C++ implementation files
└── kinfu_opencl/src/cl/        ← OpenCL kernel source (loaded at runtime)
    ├── bilateral.cl
    ├── maps.cl
    ├── tsdf.cl
    ├── estimate_combined.cl
    ├── marchingcube.cl
    └── ray_caster.cl
```

---

## Prerequisites

| Dependency | Version tested | Notes |
|---|---|---|
| CMake | ≥ 3.10 | |
| GCC / G++ | ≥ 7 | C++11 required |
| OpenCL headers | any | from conda or system package |
| OpenCL runtime | NVIDIA 3.0 | system `/usr/lib/x86_64-linux-gnu/libOpenCL.so.1` |
| Boost (headers) | ≥ 1.65 | `shared_ptr`, `make_shared` — header-only |
| Eigen3 | ≥ 3.3 | header-only |
| libpng | any | for `run_kinfu_tum` (depth PNG loading) |

**Python preprocessing dependencies** (for `mask_teddy_depth.py`):

| Package | Required for |
|---|---|
| numpy, opencv-python, Pillow | all modes |
| torch, torchvision | `maskrcnn` mode only |

---

## Docker (recommended)

The easiest way to get a fully working environment. The image bundles all C++ build tools, OpenCL headers, Python packages, and the compiled binaries.

### Host requirements

| Requirement | Purpose |
|---|---|
| Docker ≥ 20.10 | Container runtime |
| [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) | Exposes host GPU to container (`--gpus all`) |
| NVIDIA driver with OpenCL support | Provides the actual OpenCL ICD at runtime |

Install NVIDIA Container Toolkit on Ubuntu:
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor \
    -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Build the image

The `Dockerfile` lives in `gpu/` and the build context is also `gpu/`:

```bash
cd /path/to/pcl-master/gpu

# CPU-only PyTorch (default — sufficient for depth mode)
docker build -t kinfu-opencl .

# CUDA-enabled PyTorch (required for maskrcnn mode on GPU)
# Match the cu<VER> suffix to your installed CUDA version
docker build \
    --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 \
    -t kinfu-opencl .
```

Build time: ~5–10 min (downloads PyTorch). Resulting image: ~4–5 GB.

### What is installed in the image

| Layer | Contents |
|---|---|
| `ubuntu:22.04` | Base OS |
| apt | `cmake`, `gcc/g++`, `libboost-dev`, `libeigen3-dev`, `libpng-dev`, `opencl-headers`, `ocl-icd-opencl-dev`, `ocl-icd-libopencl1` |
| pip | `numpy`, `opencv-python-headless`, `pillow`, `torch`, `torchvision` |
| Built | `libkinfu_opencl.a`, `run_kinfu_oni`, `run_kinfu_tum` |
| Working directory | `/opt/kinfu/gpu/build` |

### Run

```bash
# NVIDIA GPU — KinFu requires OpenCL which needs the host GPU
docker run --rm -it --gpus all \
    -v /path/to/rgbd_dataset_freiburg1_teddy:/data/rgbd_dataset_freiburg1_teddy \
    -v /path/to/output:/output \
    kinfu-opencl
```

Inside the container (`/opt/kinfu/gpu/build` is the working directory):

```bash
# Symlink the mounted dataset into the working directory
ln -s /data/rgbd_dataset_freiburg1_teddy .

# Run the full pipeline (segmentation + KinFu) — one command
bash run_tum_teddy.sh --mode depth

# Other modes
bash run_tum_teddy.sh --mode maskrcnn   # Mask R-CNN (CUDA PyTorch required)
bash run_tum_teddy.sh --mode none       # Raw depth, no masking

# Quick test with first 100 frames
bash run_tum_teddy.sh --mode depth --frames 100

# Copy output mesh to the mounted output directory
cp output_tum.ply /output/
```

> `run_tum_teddy.sh` sets `LD_PRELOAD` automatically — no need to set it manually inside the container.

### Step-by-step (inside container)

```bash
# 1. Preprocessing only
python3 mask_teddy_depth.py rgbd_dataset_freiburg1_teddy \
    --mode depth --vis

# 2. KinFu only (after preprocessing)
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libOpenCL.so \
./run_kinfu_tum rgbd_dataset_freiburg1_teddy \
    --config kinfu_tum_freiburg1.cfg \
    --depth-txt rgbd_dataset_freiburg1_teddy/depth_masked.txt

# 3. Export aligned point clouds for visual inspection
python3 dump_depth_ply.py rgbd_dataset_freiburg1_teddy --skip 5 --outdir /output/ply_frames
```

### Retrieve output without a volume mount

```bash
# Start a container in the background, run the pipeline, then copy out
CONTAINER=$(docker run -d --gpus all \
    -v /path/to/dataset:/data/rgbd_dataset_freiburg1_teddy \
    kinfu-opencl \
    bash -c "ln -s /data/rgbd_dataset_freiburg1_teddy . && bash run_tum_teddy.sh --mode depth")

docker wait $CONTAINER
docker cp $CONTAINER:/opt/kinfu/gpu/build/output_tum.ply ./output_tum.ply
docker rm $CONTAINER
```

---

## Native Build

Edit the paths at the top of `CMakeLists.txt` to match your conda environment:

```cmake
set(CONDA_CELLSEG1 "/home/jiy/miniconda3/envs/cellseg1")   # Boost + Eigen
set(CONDA_GLUE    "/home/jiy/miniconda3/envs/glue_env")    # OpenCL headers
```

Then build:

```bash
cd gpu/build
cmake .
make -j$(nproc)
```

---

## Synthetic Runner

```bash
cd gpu/build
./run_kinfu.sh              # 200 synthetic frames
./run_kinfu.sh --frames 50  # quick test
```

Output: `output.stl` (ASCII STL / PLY, viewable in MeshLab or CloudCompare).

---

## TUM RGB-D Runner

### Quick start

```bash
cd gpu/build

# Extract dataset (if not already done)
tar xzf rgbd_dataset_freiburg1_teddy.tgz

# Run full pipeline (preprocessing + KinFu)
./run_tum_teddy.sh
```

Output: `output_tum.ply` — ASCII PLY mesh.

### `run_tum_teddy.sh` options

```
./run_tum_teddy.sh [--mode MODE] [--frames N] [--skip K] [--no-mask] [--remask]

  --mode MODE     Preprocessing mode (default: depth):
                    none      no masking, raw depth frames
                    maskrcnn  Mask R-CNN instance segmentation (class-specific)
                    depth     depth-only foreground extraction (general-purpose)
  --frames N      Process only the first N frames
  --skip K        Process every K-th frame (default: 1)
  --no-mask       Skip preprocessing entirely, use raw depth.txt
  --remask        Re-run mask_teddy_depth.py even if depth_masked/ already exists
```

Examples:

```bash
# General foreground extraction — works for any object/scene (default)
./run_tum_teddy.sh --mode depth

# Class-specific Mask R-CNN segmentation (detects 'teddy bear' via COCO)
./run_tum_teddy.sh --mode maskrcnn

# Raw depth, no masking
./run_tum_teddy.sh --mode none

# Quick test on first 100 frames
./run_tum_teddy.sh --frames 100

# Force re-segmentation then run KinFu
./run_tum_teddy.sh --remask --mode depth
```

### Running `run_kinfu_tum` directly

```bash
LD_PRELOAD=/lib/x86_64-linux-gnu/libOpenCL.so.1 \
./run_kinfu_tum rgbd_dataset_freiburg1_teddy \
    --config kinfu_tum_freiburg1.cfg \
    --depth-txt rgbd_dataset_freiburg1_teddy/depth_masked.txt \
    [--frames N] [--skip K] [--no-gt]
```

| Flag | Description |
|---|---|
| `--config FILE` | KinFu config file (intrinsics, volume size, etc.) |
| `--depth-txt FILE` | Override default `depth.txt` with a custom list (e.g. `depth_masked.txt`) |
| `--frames N` | Process only first N frames |
| `--skip K` | Process every K-th frame |
| `--no-gt` | Disable ground-truth pose injection (pure ICP) |

> **Note:** Must be run from `gpu/build/` with `LD_PRELOAD` set, or via `run_tum_teddy.sh` which handles both automatically.

---

## Preprocessing: `mask_teddy_depth.py`

Extracts the foreground object from each depth frame before KinFu integration, reducing background geometry and outliers. Three modes are available.

### Preprocessing pipeline (all modes)

```
Raw depth PNG
    │
    ├─ [Mode-specific initial mask]
    │      none      → pass-through
    │      maskrcnn  → Mask R-CNN binary instance mask (from paired RGB)
    │      depth     → depth range filter using global reference depth
    │
    ├─ adaptive_depth_threshold()
    │      Otsu split on masked depth histogram — removes far-background leakage
    │
    └─ largest_component_depth()
           Keep only the largest connected component — removes isolated outliers
```

### Modes

#### `none` — pass-through
Copies depth frames unchanged. Useful as a baseline or when the scene is already clean.

#### `maskrcnn` — Mask R-CNN instance segmentation
Uses a COCO-pretrained Mask R-CNN (torchvision) to detect `teddy bear` (class 88) or `bear` (class 23) in the paired RGB frame and applies the instance mask to the depth. Frames where no object is detected are zeroed. Requires CUDA + torch + torchvision.

```bash
python3 mask_teddy_depth.py <dataset_dir> --mode maskrcnn [--score-thr 0.5] [--device cuda]
```

#### `depth` — depth-only foreground extraction (recommended for general use)
No RGB or ML model required. Estimates the object's depth from the centre region of the first `--ref-frames` frames, then keeps all depth pixels within `--tolerance-frac` of that reference depth. Works for any object/scene where the camera orbits a single foreground subject.

```bash
python3 mask_teddy_depth.py <dataset_dir> --mode depth \
    [--ref-frames 10] [--center-frac 0.25] [--tolerance-frac 0.50]
```

### All options

```
python3 mask_teddy_depth.py <dataset_dir> [options]

  --mode MODE           none | maskrcnn | depth  (default: depth)
  --outdir DIR          output directory for masked depth PNGs
                        (default: <dataset_dir>/depth_masked/)
  --frames N            process only first N frames
  --vis                 save depth-colourmap visualisations to vis_mask/

  [maskrcnn mode]
  --score-thr F         confidence threshold (default: 0.5)
  --device DEVICE       cuda or cpu (default: cuda if available)

  [depth mode]
  --ref-frames N        frames used to estimate reference depth (default: 10)
  --center-frac F       centre region fraction for d_ref sampling (default: 0.25)
  --tolerance-frac F    depth tolerance as fraction of d_ref (default: 0.50)
```

---

## Point Cloud Export: `dump_depth_ply.py`

Exports each depth frame as a world-space PLY point cloud (applying ground-truth poses) for alignment verification before running KinFu.

```bash
python3 dump_depth_ply.py <dataset_dir> [options]

  --skip K        export every K-th frame (default: 10)
  --frames N      limit to first N frames
  --outdir DIR    output directory (default: ply_frames/)
  --no-gt         export in camera frame only (no pose applied)
```

Example:

```bash
# Export every 5th frame with GT pose applied
python3 dump_depth_ply.py rgbd_dataset_freiburg1_teddy --skip 5

# Load all ply_frames/*.ply in CloudCompare and merge to verify alignment
```

---

## Stub Headers

The following minimal stub headers allow the library to compile without the full PCL suite:

| File | Purpose |
|---|---|
| `include/direct.h` | Linux stub — maps Windows `_getcwd` → `getcwd` |
| `include/pcl/pcl_macros.h` | `PCL_EXPORTS`, `pcl_isnan` |
| `include/pcl/point_types.h` | `pcl::PointXYZ`, `pcl::Normal` |
| `include/pcl/point_cloud.h` | `pcl::PointCloud<T>` |
| `include/pcl/gpu/containers/device_array_cl.h` | `CLDeviceArray`, `CLDeviceArray2D`, `CLDeviceImage2D` |
| `include/pcl/gpu/containers/kernel_containers_cl.h` | `CLPtrStep`, `CLPtrStepSz`, `CLPtrSz` |
| `include/opencv2/core/types.hpp` | minimal `ushort` guard |

---

## OpenCL Kernel Fixes

Three bugs were found in the original `.cl` files and patched:

| File | Fix |
|---|---|
| `maps.cl` | Removed `__write_only` qualifier from buffer pointer (OpenCL 3.0 stricter) |
| `marchingcube.cl` | Removed `__global __read_only` from buffer pointer params |
| `tsdf.cl` | Changed `-1.0`, `1.0` → `-1.0f`, `1.0f` to avoid float/double type mismatch |

---

## Known Limitations

- The OpenCL kernels are loaded at runtime via relative path `../kinfu_opencl/src/cl/`; the binary must be run from `gpu/build/`.
- Conda provides a stub `libOpenCL.so` with no ICD support; always use the system library via `LD_PRELOAD` or the provided shell scripts.
- The `depth` preprocessing mode assumes the target object stays near the depth established in the first few frames. For scenes where the camera starts far away and approaches, increase `--ref-frames` or adjust `--tolerance-frac`.
- The `maskrcnn` mode zeros frames where the object is not detected (e.g. when viewed from behind or heavily motion-blurred). Since KinFu uses ground-truth poses directly (`disableIcp()`), zeroed frames are skipped without breaking tracking.

---

## Origin

This code is based on the PCL GPU KinFu module. The original CUDA-based PCL KinFu is from the [Point Cloud Library (PCL)](https://pointclouds.org/) project.
