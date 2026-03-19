#!/usr/bin/env python3
"""
dump_depth_ply.py
-----------------
For each depth frame in a TUM RGB-D dataset, back-project to 3D and apply the
nearest ground-truth affine pose, then save as a PLY point cloud.

Usage:
    python3 dump_depth_ply.py <dataset_dir> [options]

Options:
    --frames N      process only first N frames (default: all)
    --skip K        save every K-th frame (default: 10)
    --outdir DIR    output directory for PLY files (default: ply_frames/)
    --no-gt         write points in camera frame only (no pose applied)
    --max-dt DT     max timestamp delta for GT matching, seconds (default: 0.1)

TUM freiburg1 intrinsics (hard-coded, matching run_kinfu_tum.cpp):
    fx=517.3  fy=516.5  cx=318.6  cy=255.3
    depth scale: raw / 5000 = metres
"""

import sys
import os
import struct
import argparse
import numpy as np

try:
    from PIL import Image
    def load_depth(path):
        img = Image.open(path)
        return np.array(img, dtype=np.uint16)
except ImportError:
    import png  # pypng fallback
    def load_depth(path):
        r = png.Reader(filename=path)
        w, h, rows, meta = r.read()
        return np.array(list(rows), dtype=np.uint16)

# ── TUM freiburg1 intrinsics ──────────────────────────────────────────────────
FX, FY = 517.3, 516.5
CX, CY = 318.6, 255.3
DEPTH_SCALE = 5000.0   # raw / 5000 = metres


def parse_tum_file(path):
    """Return list of (timestamp_float, rest_of_line_string) tuples."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            entries.append((float(parts[0]), parts[1:]))
    entries.sort(key=lambda x: x[0])
    return entries


def nearest_entry(entries, ts, max_dt=0.1):
    """Binary-search for the entry whose timestamp is closest to ts."""
    lo, hi = 0, len(entries)
    while lo < hi:
        mid = (lo + hi) // 2
        if entries[mid][0] < ts:
            lo = mid + 1
        else:
            hi = mid
    best_dt = 1e18
    best = None
    for idx in (lo, lo - 1):
        if 0 <= idx < len(entries):
            dt = abs(entries[idx][0] - ts)
            if dt < best_dt:
                best_dt = dt
                best = entries[idx]
    if best_dt > max_dt:
        return None
    return best


def gt_to_affine(parts):
    """parts = [tx, ty, tz, qx, qy, qz, qw]  → 4×4 numpy float32 matrix."""
    tx, ty, tz = float(parts[0]), float(parts[1]), float(parts[2])
    qx, qy, qz, qw = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
    # normalise quaternion
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n

    R = np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float32)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3,  3] = [tx, ty, tz]
    return T


def backproject(depth_mm, valid_mask=None):
    """
    depth_mm : H×W uint16 array, values in millimetres
    Returns (N,3) float32 array of (X,Y,Z) in metres for valid pixels.
    """
    H, W = depth_mm.shape
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    Z = depth_mm.astype(np.float32) / 1000.0  # mm → metres
    mask = Z > 0
    if valid_mask is not None:
        mask &= valid_mask

    Z = Z[mask]
    X = (uu[mask] - CX) * Z / FX
    Y = (vv[mask] - CY) * Z / FY

    return np.stack([X, Y, Z], axis=1)  # (N,3)


def apply_affine(pts_cam, T):
    """pts_cam: (N,3), T: 4×4  → (N,3) in world frame."""
    ones = np.ones((len(pts_cam), 1), dtype=np.float32)
    pts_h = np.hstack([pts_cam, ones])          # (N,4)
    pts_world = (T @ pts_h.T).T                 # (N,4)
    return pts_world[:, :3]


def write_ply(path, pts):
    """Write ASCII PLY with x,y,z float properties."""
    N = len(pts)
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset_dir')
    ap.add_argument('--frames', type=int, default=-1)
    ap.add_argument('--skip',   type=int, default=10)
    ap.add_argument('--outdir', default='ply_frames')
    ap.add_argument('--no-gt',  action='store_true')
    ap.add_argument('--max-dt', type=float, default=0.1)
    args = ap.parse_args()

    dataset = args.dataset_dir
    outdir  = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Load depth list
    depth_list = parse_tum_file(os.path.join(dataset, 'depth.txt'))
    if not depth_list:
        sys.exit("No entries in depth.txt")

    # Load GT
    gt_list = []
    gt_path = os.path.join(dataset, 'groundtruth.txt')
    if not args.no_gt and os.path.exists(gt_path):
        gt_list = parse_tum_file(gt_path)
        print(f"GT poses loaded: {len(gt_list)}")
    else:
        print("GT not used – writing in camera frame")

    total = len(depth_list)
    if args.frames > 0:
        total = min(total, args.frames)

    saved = 0
    for i in range(0, total, args.skip):
        ts, rest = depth_list[i]
        fname = rest[0]
        fpath = os.path.join(dataset, fname)

        if not os.path.exists(fpath):
            print(f"  missing: {fpath}")
            continue

        depth_raw = load_depth(fpath)              # uint16, raw / 5000 = m
        depth_mm  = (depth_raw / 5).astype(np.uint16)  # raw / 5 = mm  (same as KinFu)

        pts = backproject(depth_mm)
        if len(pts) == 0:
            print(f"  frame {i:04d}: no valid depth pixels, skipped")
            continue

        # Apply GT pose if available
        if gt_list:
            gt = nearest_entry(gt_list, ts, args.max_dt)
            if gt is not None:
                T = gt_to_affine(gt[1])
                pts = apply_affine(pts, T)
                pose_str = f"t=({gt[1][0]},{gt[1][1]},{gt[1][2]})"
            else:
                pose_str = "no GT (camera frame)"
        else:
            pose_str = "camera frame"

        out_name = f"frame_{i:05d}_{ts:.3f}.ply"
        out_path = os.path.join(outdir, out_name)
        write_ply(out_path, pts)
        saved += 1
        print(f"  [{i:4d}/{total}] {out_name}  pts={len(pts)}  {pose_str}")

    print(f"\nDone. {saved} PLY files written to '{outdir}/'")


if __name__ == '__main__':
    main()
