#!/usr/bin/env python3
# =============================================================================
# color_precise_cloud.py  --  paint the SAM3->SAM2 object IDs onto the PRECISE
# ICP+TSDF (GPU) fused cloud, instead of the noisy naive back-projection.
#
# Geometry source : <new_env>/static_scene/depth_camera_init_points.ply
#                   (the GPU ICP+TSDF fusion seed the capture pipeline wrote;
#                    1.4M pts, ~1.5mm near / 5mm far, free-space carved -> no
#                    flying-pixel edge leakage).
# Labels source   : output/seg_ids.npz  (per-frame object IDs from SAM2).
#
# TRANSFER = visibility-aware projection voting:
#   for each precise point, project into every camera; where the point is
#   actually visible (in-frame AND its camera-depth matches the depth map),
#   read that frame's SAM2 label and cast a vote. Majority label wins. This
#   uses real occlusion, so a table point never inherits an object's color the
#   way a blind nearest-neighbor would, and the carved TSDF surface stays clean.
#
# OUT: output/precise_objects.ply  (objects = palette color, background = gray)
#
# RUN: /home/mrc-cuhk/miniconda3/envs/dynamic_gs/bin/python \
#        experiments/sam3_seed_sam2_mvp/color_precise_cloud.py
# =============================================================================

import os
import json
import colorsys

import numpy as np
import cv2
import open3d as o3d

DATASET_DIR = (
    "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/"
    "new_env/static_scene"
)
PRECISE_PLY = ("/home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts/experiments/"
               "icp_fusion_mvp/output/online_seed.ply")   # freshly GPU-fused, production params

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(THIS_DIR, "output")
SEG_IDS_NPZ = os.path.join(OUTPUT_DIR, "seg_ids.npz")   # SAM3-seed -> SAM2-propagate (table labeled @ ct=0.10)
OUT_PLY = os.path.join(OUTPUT_DIR, "precise_objects.ply")

DEPTH_SCALE = 1000.0
OCC_TOL_M = 0.02           # |point_z - depth_map| under this => point is visible in that cam
MIN_OBJ_VOTES = 3          # an object label needs >=N visible votes, else -> background
EROSION_PX = 0             # OFF (3px ate into the objects -> grey holes); table-as-object fixes the leak instead
TINT_ALPHA = 0.55          # object points = (1-a)*TSDF_RGB + a*palette; background keeps full TSDF RGB
MAX_IDS = 64


def color_for_id(k: int) -> tuple:
    if k <= 0:
        return (0.62, 0.62, 0.62)
    h = (k * 0.6180339887498949) % 1.0
    s = 0.55 + 0.35 * ((k * 2) % 3) / 2.0
    v = 0.75 + 0.25 * ((k * 5) % 2)
    return colorsys.hsv_to_rgb(h, s, v)


PALETTE = np.array([color_for_id(k) for k in range(MAX_IDS + 1)], dtype=np.float32)


def abspath(rel):
    return os.path.join(DATASET_DIR, rel.lstrip("./"))


def erode_per_object(seg, px):
    """Shrink each object mask by `px` pixels (eroded edges -> background id 0).
    Per-object so neighbouring objects don't bleed into each other."""
    if px <= 0:
        return seg
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    out = np.zeros_like(seg)
    for oid in np.unique(seg):
        if oid == 0:
            continue
        m = cv2.erode((seg == oid).astype(np.uint8), k)
        out[m > 0] = oid
    return out


def main():
    import sys
    seg_path = sys.argv[1] if len(sys.argv) > 1 else SEG_IDS_NPZ
    out_path = sys.argv[2] if len(sys.argv) > 2 else OUT_PLY

    # --- precise geometry (KEEP the TSDF RGB; we only tint object points) ---
    pc = o3d.io.read_point_cloud(PRECISE_PLY)
    world = np.asarray(pc.points, np.float64)                 # (N,3)
    N = len(world)
    tsdf_rgb = np.asarray(pc.colors, np.float64).copy()       # real TSDF colors
    if tsdf_rgb.shape != world.shape:                         # geometry-only cloud -> neutral base
        tsdf_rgb = np.full((N, 3), 0.62)
    print(f"[geom] precise cloud: {N:,} points  (TSDF RGB preserved)")

    # --- labels + cameras ---
    seg_ids = np.load(seg_path)["seg_ids"]                    # (F,H,W) uint8
    F, H, W = seg_ids.shape
    K = int(seg_ids.max())
    with open(os.path.join(DATASET_DIR, "transforms.json")) as f:
        meta = json.load(f)
    fx, fy, cx, cy = meta["fl_x"], meta["fl_y"], meta["cx"], meta["cy"]
    import re
    frames = sorted(meta["frames"], key=lambda fr: int(re.findall(r"\d+", fr["file_path"])[-1]))
    assert len(frames) == F, f"seg_ids has {F} frames but transforms has {len(frames)}"
    print(f"[labels] {F} frames, {K} object ids, {H}x{W}")

    seg_ids = np.stack([erode_per_object(seg_ids[i], EROSION_PX) for i in range(F)])
    if EROSION_PX:
        print(f"[erode] each object mask shrunk by {EROSION_PX} px before voting")

    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    homog = np.concatenate([world, np.ones((N, 1))], 1).T     # (4,N)
    votes = np.zeros((N, K + 1), np.int32)                    # incl. id 0 = background

    for i, fr in enumerate(frames):
        c2w_cv = np.asarray(fr["transform_matrix"], np.float64) @ flip
        cam = (np.linalg.inv(c2w_cv) @ homog).T[:, :3]        # (N,3) in OpenCV cam frame
        z = cam[:, 2]
        front = z > 1e-6
        u = np.full(N, -1.0); v = np.full(N, -1.0)
        u[front] = fx * cam[front, 0] / z[front] + cx
        v[front] = fy * cam[front, 1] / z[front] + cy
        in_img = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = np.where(in_img)[0]
        ui = u[idx].astype(np.int32); vi = v[idx].astype(np.int32); zi = z[idx]

        depth = cv2.imread(abspath(fr["depth_file_path"]), cv2.IMREAD_UNCHANGED).astype(np.float32) / DEPTH_SCALE
        dmap = depth[vi, ui]
        vis = (dmap > 0) & (np.abs(zi - dmap) < OCC_TOL_M)    # occlusion test
        idx_v = idx[vis]
        lbl = seg_ids[i][vi[vis], ui[vis]].astype(np.int64)
        np.add.at(votes, (idx_v, lbl), 1)
        if i % 20 == 0 or i == F - 1:
            print(f"[vote] frame {i + 1}/{F}: {len(idx_v):,} visible votes")

    # Label = best object, but ONLY if it BEATS the background votes (true majority).
    # This is the fix for border leakage: a table point with many "background" votes
    # and a few stray "object" votes (grazing-view mask edges) stays background,
    # instead of being labeled the object because background was ignored.
    bg_cnt = votes[:, 0]
    obj_votes = votes.copy(); obj_votes[:, 0] = 0
    best_obj = obj_votes.argmax(1)
    best_cnt = obj_votes[np.arange(N), best_obj]
    ids = np.where((best_cnt >= MIN_OBJ_VOTES) & (best_cnt > bg_cnt), best_obj, 0).astype(np.int64)

    uniq, cnt = np.unique(ids, return_counts=True)
    print(f"\n[result] points per id (0=background):")
    for k, c in zip(uniq, cnt):
        print(f"   id {int(k):>2}: {int(c):>9,}  ({100*c/N:4.1f}%)")

    # keep TSDF RGB everywhere; tint ONLY object points so the real scene stays visible
    out = tsdf_rgb.copy()
    objm = ids > 0
    out[objm] = (1.0 - TINT_ALPHA) * tsdf_rgb[objm] + TINT_ALPHA * PALETTE[ids[objm]]
    pc.colors = o3d.utility.Vector3dVector(np.clip(out, 0, 1))
    o3d.io.write_point_cloud(out_path, pc)
    print(f"\n[out] {out_path}")


if __name__ == "__main__":
    main()
