"""ICP-refine dynamic_scene/transforms.json against the static-fused PLY.

The dynamic frames are typically captured from raw URDF FK and have the same
sub-mm/sub-degree drift as the original static frames. The static frames have
already been ICP-aligned via ``rewrite_transforms_with_icp.py`` against the
TSDF-fused cloud. This script does the same per-frame alignment for the
DYNAMIC frames, but with the static PLY as a FIXED target — much faster than
re-running TSDF fusion.

Pipeline (per frame):
    1. Back-project the dynamic frame's depth (masked by gripper mask if
       present) through scene intrinsics + current pose → source point cloud
       in world coords.
    2. Point-to-plane ICP, source → static PLY (target), small max-iter.
    3. Right-compose the refined transform into the frame's c2w.

Original poses preserved at ``dynamic_scene/transforms_urdf_backup.json``.

Run:
    conda activate dynamic_gs
    python scripts/icp_refine_dynamic_transforms.py <data_dir>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

# OpenGL ↔ OpenCV convention flip (involutive).
_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def _load_depth(path: Path) -> np.ndarray:
    """Load uint16 depth in millimetres → float32 metres."""
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise RuntimeError(f"could not read depth: {path}")
    return d.astype(np.float32) * 1e-3  # mm → m


def _load_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return (m > 127) if m is not None else None


def _depth_to_world_cloud(
    depth_m: np.ndarray,
    mask: np.ndarray | None,
    fx: float, fy: float, cx: float, cy: float,
    c2w_gl: np.ndarray,
    stride: int = 4,
) -> np.ndarray:
    """Stride-subsampled back-projection of depth → world-frame xyz (Nx3).

    c2w is OpenGL convention. Output is in the same world frame as the static PLY.
    """
    H, W = depth_m.shape[:2]
    vv, uu = np.meshgrid(np.arange(0, H, stride), np.arange(0, W, stride), indexing="ij")
    d = depth_m[vv, uu]
    if mask is not None:
        m = mask[vv, uu]
    else:
        m = np.ones_like(d, dtype=bool)
    valid = (d > 0.05) & (d < 5.0) & m
    if not valid.any():
        return np.zeros((0, 3), dtype=np.float64)
    uu_v = uu[valid].astype(np.float64)
    vv_v = vv[valid].astype(np.float64)
    d_v  = d[valid].astype(np.float64)
    # Camera-frame OpenGL: x=right, y=up, z=back
    p_cam_gl = np.stack([
        d_v * (uu_v - cx) / fx,
        -d_v * (vv_v - cy) / fy,
        -d_v,
    ], axis=-1)
    R = c2w_gl[:3, :3]
    t = c2w_gl[:3, 3]
    return (R @ p_cam_gl.T).T + t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", type=Path, help="Dataset dir containing dynamic_scene/ and static_scene/")
    ap.add_argument("--max-dist-m", type=float, default=0.02,
                    help="ICP max correspondence distance (m). Default: 0.02 (2 cm).")
    ap.add_argument("--max-iters", type=int, default=30,
                    help="ICP max iterations per frame. Default: 30.")
    ap.add_argument("--stride", type=int, default=4,
                    help="Depth subsample stride. Default: 4 (~40k points at 800x800).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Do not write back; just report drift stats.")
    args = ap.parse_args()

    data_dir = args.data_dir.resolve()
    dyn_dir = data_dir / "dynamic_scene"
    static_dir = data_dir / "static_scene"

    transforms_path = dyn_dir / "transforms.json"
    backup_path = dyn_dir / "transforms_urdf_backup.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"no transforms.json at {transforms_path}")
    static_ply = static_dir / "depth_camera_init_points.ply"
    if not static_ply.exists():
        raise FileNotFoundError(f"static PLY not found at {static_ply}")

    meta = json.loads(transforms_path.read_text())
    frames = meta.get("frames", [])
    if not frames:
        raise RuntimeError(f"no frames in {transforms_path}")

    if meta.get("pose_source") == "icp_refined_from_urdf_v1":
        print("[icp-refine-dyn] dynamic transforms already marked ICP-refined; aborting")
        return

    # Backup before overwrite.
    if not args.dry_run and not backup_path.exists():
        shutil.copy2(transforms_path, backup_path)
        print(f"[icp-refine-dyn] backed up URDF transforms → {backup_path}")

    fx = float(meta["fl_x"]); fy = float(meta["fl_y"])
    cx = float(meta["cx"]);   cy = float(meta["cy"])
    W  = int(meta["w"]);      H  = int(meta["h"])
    print(f"[icp-refine-dyn] intrinsics: fx={fx:.2f} fy={fy:.2f} cx={cx:.1f} cy={cy:.1f} W={W} H={H}")

    # Load + prep the static PLY once.
    print(f"[icp-refine-dyn] loading static PLY: {static_ply}")
    target = o3d.io.read_point_cloud(str(static_ply))
    if len(target.points) == 0:
        raise RuntimeError("static PLY has zero points")
    print(f"[icp-refine-dyn] target points: {len(target.points)}, estimating normals...")
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))

    t_total0 = time.time()
    drifts_t = []
    drifts_r = []
    fitnesses = []

    for i, fr in enumerate(frames):
        c2w_gl = np.asarray(fr["transform_matrix"], dtype=np.float64)
        depth_p = (dyn_dir / fr["depth_file_path"].lstrip("./")).resolve()
        mask_p  = (dyn_dir / fr["mask_path"].lstrip("./")).resolve() if fr.get("mask_path") else None
        try:
            depth_m = _load_depth(depth_p)
            mask = _load_mask(mask_p) if mask_p else None
        except Exception as exc:
            print(f"[icp-refine-dyn] frame {i}: skip ({exc})")
            continue

        src_world = _depth_to_world_cloud(depth_m, mask, fx, fy, cx, cy, c2w_gl, stride=args.stride)
        if src_world.shape[0] < 1000:
            print(f"[icp-refine-dyn] frame {i}: too few valid src pts ({src_world.shape[0]}); keeping URDF")
            continue

        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(src_world)
        reg = o3d.pipelines.registration.registration_icp(
            src, target, args.max_dist_m, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=args.max_iters),
        )
        T = np.asarray(reg.transformation, dtype=np.float64)
        # T maps src(world) → target(world). New c2w = T @ c2w.
        c2w_refined = T @ c2w_gl

        # Drift stats.
        dt = T[:3, 3]
        drift_t_mm = float(np.linalg.norm(dt) * 1000.0)
        # Rotation angle from axis-angle of T's rotation:
        R_t = T[:3, :3]
        cos_th = (np.trace(R_t) - 1.0) / 2.0
        cos_th = float(np.clip(cos_th, -1.0, 1.0))
        drift_r_deg = float(np.degrees(np.arccos(cos_th)))
        drifts_t.append(drift_t_mm)
        drifts_r.append(drift_r_deg)
        fitnesses.append(float(reg.fitness))

        fr["transform_matrix"] = c2w_refined.tolist()

        if (i + 1) % 20 == 0 or i == len(frames) - 1:
            print(f"[icp-refine-dyn] {i+1}/{len(frames)} done, last: "
                  f"fitness={reg.fitness:.3f} drift={drift_t_mm:.2f}mm/{drift_r_deg:.3f}°")

    total_s = time.time() - t_total0
    if drifts_t:
        a = np.asarray(drifts_t); b = np.asarray(drifts_r); c = np.asarray(fitnesses)
        print(f"[icp-refine-dyn] DONE — {len(drifts_t)}/{len(frames)} frames refined in {total_s:.1f}s")
        print(f"[icp-refine-dyn]   translation drift  mm: median={np.median(a):.2f} p90={np.percentile(a,90):.2f} max={a.max():.2f}")
        print(f"[icp-refine-dyn]   rotation drift    deg: median={np.median(b):.3f} p90={np.percentile(b,90):.3f} max={b.max():.3f}")
        print(f"[icp-refine-dyn]   fitness:               median={np.median(c):.3f} min={c.min():.3f}")

    if args.dry_run:
        print("[icp-refine-dyn] dry-run mode: NOT writing transforms.json")
        return

    meta["pose_source"] = "icp_refined_from_urdf_v1"
    tmp = transforms_path.with_name(f".{transforms_path.name}.tmp")
    tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    tmp.replace(transforms_path)
    print(f"[icp-refine-dyn] wrote refined → {transforms_path}")


if __name__ == "__main__":
    main()
