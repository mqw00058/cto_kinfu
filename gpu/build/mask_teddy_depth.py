#!/usr/bin/env python3
"""
mask_teddy_depth.py
-------------------
Foreground extractor for RGB-D KinFu datasets.

Three preprocessing modes (--mode):

  none      Copy depth frames unchanged (no masking).

  maskrcnn  Segment the foreground using Mask R-CNN (COCO pretrained).
            Detects 'teddy bear' (class 88) or 'bear' (class 23) in the
            paired RGB frame and uses the instance mask on the depth.
            Falls back to zeroing the frame when nothing is detected.
            Requires: torch, torchvision, Pillow.

  depth     Class-agnostic depth-only foreground extraction.
            Estimates the object depth from the centre of the first
            --ref-frames frames, then keeps pixels within --tolerance-frac
            of that reference depth. No RGB / ML model required.
            Works for any object/scene.

All modes apply two common post-processing steps:
  • Otsu adaptive threshold  – trims far-background leakage
  • Largest connected component – removes isolated outlier clusters

Usage:
    python3 mask_teddy_depth.py <dataset_dir> [options]

Options:
    --mode MODE           Preprocessing mode: none | maskrcnn | depth
                          (default: depth)
    --outdir DIR          Output directory for masked depth PNGs
                          (default: <dataset_dir>/depth_masked/)
    --frames N            Process only first N frames (default: all)
    --vis                 Save depth-colourmap overlays to <dataset_dir>/vis_mask/

  [maskrcnn mode]
    --score-thr F         Mask R-CNN confidence threshold (default: 0.5)
    --device DEVICE       'cuda' or 'cpu' (default: cuda if available)

  [depth mode]
    --center-frac F       Centre region fraction for d_ref estimation
                          (default: 0.25)
    --tolerance-frac F    Depth tolerance as fraction of d_ref (default: 0.50)
    --ref-frames N        Number of initial frames used to estimate d_ref
                          (default: 10)

Output:
    <dataset_dir>/depth_masked/     Masked 16-bit depth PNGs
    <dataset_dir>/depth_masked.txt  TUM-format depth list for run_kinfu_tum
"""

import sys, os, argparse
import numpy as np
import cv2
from PIL import Image

# ── COCO IDs ──────────────────────────────────────────────────────────────────
COCO_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush',
]
TEDDY_ID = COCO_NAMES.index('teddy bear')   # 88
BEAR_ID  = COCO_NAMES.index('bear')         # 23

# ── helpers ───────────────────────────────────────────────────────────────────

def parse_tum_file(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            entries.append((float(parts[0]), parts[1]))
    entries.sort(key=lambda x: x[0])
    return entries


def associate(depth_list, rgb_list, max_dt=0.05):
    import bisect
    rgb_ts = [e[0] for e in rgb_list]
    pairs = []
    for d_ts, d_file in depth_list:
        idx = bisect.bisect_left(rgb_ts, d_ts)
        best, best_dt = None, 1e18
        for i in (idx - 1, idx):
            if 0 <= i < len(rgb_list):
                dt = abs(rgb_list[i][0] - d_ts)
                if dt < best_dt:
                    best_dt, best = dt, rgb_list[i][1]
        if best_dt <= max_dt and best is not None:
            pairs.append((d_ts, d_file, best))
    return pairs


def load_depth_png(path):
    return np.array(Image.open(path), dtype=np.uint16)


def save_depth_png(path, arr):
    Image.fromarray(arr).save(path)


# ── shared post-processing ────────────────────────────────────────────────────

def adaptive_depth_threshold(depth_masked):
    """Otsu split: keep nearer cluster, discard far tail."""
    valid = depth_masked[depth_masked > 0]
    if len(valid) == 0:
        return depth_masked
    d_min, d_max = int(valid.min()), int(valid.max())
    if d_max == d_min:
        return depth_masked

    scale = 255.0 / (d_max - d_min)
    norm = ((depth_masked.astype(np.float32) - d_min) * scale).clip(0, 255).astype(np.uint8)
    norm[depth_masked == 0] = 0

    hist = cv2.calcHist([norm], [0], (depth_masked > 0).astype(np.uint8),
                        [255], [1, 256]).flatten()
    total = hist.sum()
    if total == 0:
        return depth_masked

    sum_all = np.dot(np.arange(255), hist)
    sum_b, w_b, max_var, thresh_norm = 0.0, 0.0, 0.0, 255
    for t in range(255):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        mean_b = sum_b / w_b
        mean_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (mean_b - mean_f) ** 2
        if var > max_var:
            max_var = var
            thresh_norm = t

    thresh_depth = int(thresh_norm / scale) + d_min
    result = depth_masked.copy()
    result[depth_masked > thresh_depth] = 0
    return result


def largest_component_depth(depth_masked):
    """Keep only the largest connected component of non-zero pixels."""
    valid = (depth_masked > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(valid, connectivity=8)
    if n_labels <= 1:
        return depth_masked
    biggest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return depth_masked * (labels == biggest).astype(np.uint16)


def postprocess(depth_masked):
    depth_masked = adaptive_depth_threshold(depth_masked)
    depth_masked = largest_component_depth(depth_masked)
    return depth_masked


# ── mode: maskrcnn ────────────────────────────────────────────────────────────

def load_maskrcnn(device):
    import torch, torchvision
    from torchvision import transforms as T
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
    model.eval().to(device)
    return model, T.ToTensor()


def maskrcnn_fg_mask(model, to_tensor, rgb_path, device, score_thr):
    import torch
    rgb_img = Image.open(rgb_path).convert('RGB')
    with torch.no_grad():
        pred = model([to_tensor(rgb_img).to(device)])[0]

    labels = pred['labels'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    masks  = pred['masks'].cpu().numpy()   # (N,1,H,W)

    best_score, best_mask = -1, None
    for label_id in (TEDDY_ID, BEAR_ID):
        for i, (lbl, sc) in enumerate(zip(labels, scores)):
            if lbl == label_id and sc > score_thr and sc > best_score:
                best_score = sc
                best_mask  = (masks[i, 0] > 0.5).astype(np.uint8)

    h, w = np.array(rgb_img).shape[:2]
    if best_mask is None:
        return np.zeros((h, w), dtype=np.uint8), 0.0, False
    return best_mask, best_score, True


# ── mode: depth ───────────────────────────────────────────────────────────────

def estimate_global_depth_ref(depth_list, dataset, center_frac, n_ref_frames):
    samples = []
    for ts, fname in depth_list[:n_ref_frames]:
        path = os.path.join(dataset, fname)
        if not os.path.exists(path):
            continue
        depth = load_depth_png(path)
        h, w = depth.shape
        cy0, cy1 = int(h*(1-center_frac)/2), int(h*(1+center_frac)/2)
        cx0, cx1 = int(w*(1-center_frac)/2), int(w*(1+center_frac)/2)
        valid = depth[cy0:cy1, cx0:cx1]
        valid = valid[valid > 0]
        if len(valid) >= 10:
            samples.append(float(np.median(valid)))
    return float(np.median(samples)) if samples else None


def depth_fg_mask(depth_raw, d_ref, tolerance_frac):
    if d_ref is None or d_ref <= 0:
        return np.ones(depth_raw.shape, dtype=np.uint8)
    tol = tolerance_frac * d_ref
    return ((depth_raw > 0) &
            (depth_raw >= d_ref - tol) &
            (depth_raw <= d_ref + tol)).astype(np.uint8)


# ── visualisation ─────────────────────────────────────────────────────────────

def write_vis(vis_dir, i, depth_masked):
    d_vis = depth_masked.astype(np.float32)
    valid = d_vis > 0
    if valid.any():
        d_vis[valid] = (d_vis[valid] - d_vis[valid].min()) / \
                       (d_vis[valid].max() - d_vis[valid].min() + 1e-6)
    d8 = (d_vis * 255).astype(np.uint8)
    colored = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
    colored[~valid] = 30
    cv2.imwrite(os.path.join(vis_dir, f"vis_{i:05d}.jpg"), colored)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset_dir')
    ap.add_argument('--mode',           choices=['none', 'maskrcnn', 'depth'],
                                        default='depth')
    ap.add_argument('--outdir',         default=None)
    ap.add_argument('--frames',         type=int,   default=-1)
    ap.add_argument('--vis',            action='store_true')
    # maskrcnn options
    ap.add_argument('--score-thr',      type=float, default=0.5)
    ap.add_argument('--device',         default=None)
    # depth options
    ap.add_argument('--center-frac',    type=float, default=0.25)
    ap.add_argument('--tolerance-frac', type=float, default=0.50)
    ap.add_argument('--ref-frames',     type=int,   default=10)
    args = ap.parse_args()

    dataset    = args.dataset_dir
    out_subdir = args.outdir or os.path.join(dataset, 'depth_masked')
    os.makedirs(out_subdir, exist_ok=True)

    vis_dir = None
    if args.vis:
        vis_dir = os.path.join(dataset, 'vis_mask')
        os.makedirs(vis_dir, exist_ok=True)

    depth_list = parse_tum_file(os.path.join(dataset, 'depth.txt'))
    total = len(depth_list)
    if args.frames > 0:
        total = min(total, args.frames)

    print(f"Mode         : {args.mode}")
    print(f"Depth frames : {total}")

    # ── mode-specific setup ───────────────────────────────────────────────────
    if args.mode == 'maskrcnn':
        import torch
        device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device       : {device}")
        print(f"Score thr    : {args.score_thr}")
        rgb_list = parse_tum_file(os.path.join(dataset, 'rgb.txt'))
        pairs = associate(depth_list, rgb_list)
        pairs_map = {d: (r,) for d, r, *_ in
                     [(p[0], p[1], p[2]) for p in pairs]}
        # rebuild as dict keyed by depth ts
        pairs_map = {}
        for d_ts, d_file, r_file in pairs:
            pairs_map[d_ts] = r_file
        print(f"RGB matched  : {len(pairs_map)} / {total}")
        model, to_tensor = load_maskrcnn(device)

    elif args.mode == 'depth':
        d_ref = estimate_global_depth_ref(depth_list, dataset,
                                          args.center_frac, args.ref_frames)
        if d_ref:
            print(f"d_ref        : {d_ref:.0f} raw  ({d_ref/5000:.3f} m)")
            print(f"Tolerance    : ±{args.tolerance_frac*100:.0f}%  "
                  f"({d_ref*args.tolerance_frac/5000:.3f} m)")
        else:
            print("d_ref        : None — no valid centre pixels found")

    print()

    # ── per-frame loop ────────────────────────────────────────────────────────
    out_txt = os.path.join(dataset, 'depth_masked.txt')
    stats = {'kept': 0, 'zeroed': 0,
             'detected': 0, 'missed': 0}   # maskrcnn extras

    with open(out_txt, 'w') as fp:
        fp.write(f"# depth_masked — mode={args.mode} — mask_teddy_depth.py\n")
        fp.write("# timestamp filename\n")

        for i, (d_ts, d_file) in enumerate(depth_list[:total]):
            d_path = os.path.join(dataset, d_file)
            if not os.path.exists(d_path):
                continue

            depth_raw = load_depth_png(d_path)

            if args.mode == 'none':
                depth_masked = depth_raw

            elif args.mode == 'maskrcnn':
                r_file = pairs_map.get(d_ts)
                if r_file:
                    r_path = os.path.join(dataset, r_file)
                    mask, score, detected = maskrcnn_fg_mask(
                        model, to_tensor, r_path, device, args.score_thr)
                    if detected:
                        stats['detected'] += 1
                    else:
                        stats['missed'] += 1
                else:
                    mask = np.zeros(depth_raw.shape, dtype=np.uint8)
                    stats['missed'] += 1
                depth_masked = postprocess(depth_raw * mask)

            else:  # depth
                mask = depth_fg_mask(depth_raw, d_ref, args.tolerance_frac)
                depth_masked = postprocess(depth_raw * mask)

            out_name = os.path.basename(d_file)
            out_path  = os.path.join(out_subdir, out_name)
            save_depth_png(out_path, depth_masked)
            fp.write(f"{d_ts:.6f} {os.path.relpath(out_path, dataset)}\n")

            if depth_masked.max() > 0:
                stats['kept'] += 1
            else:
                stats['zeroed'] += 1

            if vis_dir:
                write_vis(vis_dir, i, depth_masked)

            if (i + 1) % 50 == 0 or i == total - 1:
                n_pts = int((depth_masked > 0).sum())
                print(f"  [{i+1:4d}/{total}]  pts={n_pts:>7d}  {d_file}")

    print(f"\n=== Done ===")
    print(f"  kept   : {stats['kept']}")
    print(f"  zeroed : {stats['zeroed']}")
    if args.mode == 'maskrcnn':
        print(f"  detected : {stats['detected']}  missed : {stats['missed']}")
    print(f"  output : {out_subdir}/")
    print(f"  list   : {out_txt}")


if __name__ == '__main__':
    main()
