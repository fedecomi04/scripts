"""Rewrite a captured dataset's transforms.json with ICP-refined poses.

Background. The capture flow writes URDF-FK poses (Gazebo pose-plugin output
slerped to image stamps) to ``<data>/static_scene/transforms.json``. The
streaming OnlineFusion runs point-to-plane ICP between each new frame's
depth cloud and the running TSDF model, producing a refined per-frame pose
— but in the shipped pipeline that refined pose is discarded after the
integrate() call. The seed PLY (``depth_camera_init_points.ply``) lives in
the ICP-refined frame while transforms.json keeps raw FK poses, so
Splatfacto trains against a small but systematic misalignment.

This script closes that loop. It re-runs ICP over the recorded frames using
the same OnlineFusion class, captures the refined poses, and overwrites
``transforms.json``. The original is preserved as
``transforms_urdf_backup.json``.

Frame 0 has no ICP (it seeds the TSDF) — its URDF pose is kept verbatim, and
the global frame is anchored there by definition.

Pose conventions:
- transforms.json stores OpenGL c2w (Nerfstudio convention).
- OnlineFusion works in OpenCV (flip y/z); it converts via
  ``diag(1,-1,-1,1)`` at the boundary. We pass OpenGL c2w in, the ICP
  result returned from add_frame() is already in the same OpenGL frame
  the caller passed in (see online_fusion.py: ``c2w_cv = c2w @ FLIP``,
  then ``refined = T @ c2w_cv`` (in CV), then ``refined_gl =
  refined @ inv(FLIP)`` for return) — so we write what add_frame()
  returns directly into transforms.json.

Run:
    conda activate dynamic_gs
    python scripts/rewrite_transforms_with_icp.py <data_dir>
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

# Pull OnlineFusion + its add_frame contract directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from dynamic_gs.utils.online_fusion import OnlineFusion  # noqa: E402


_OPENGL_OPENCV_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])  # involutive (its own inverse)


def _load_intrinsics(meta: dict) -> tuple[float, float, float, float, int, int]:
    fx = float(meta["fl_x"]); fy = float(meta["fl_y"])
    cx = float(meta["cx"]);   cy = float(meta["cy"])
    W = int(meta["w"]);       H = int(meta["h"])
    return fx, fy, cx, cy, W, H


def _maybe_load_mask(static_dir: Path, fr: dict) -> np.ndarray | None:
    mp = fr.get("mask_path")
    if not mp:
        return None
    p = (static_dir / mp.lstrip("./")).resolve()
    if not p.exists():
        return None
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return (m > 0).astype(np.uint8) if m is not None else None


def _convert_refined_to_gl(refined_cv: np.ndarray) -> np.ndarray:
    """OnlineFusion.add_frame returns refined c2w in OpenCV frame
    (online_fusion.py:516-518). Apply the OpenGL↔OpenCV flip to round-trip
    back to the Nerfstudio convention used in transforms.json.
    """
    return np.asarray(refined_cv, dtype=np.float64) @ _OPENGL_OPENCV_FLIP


def rewrite_transforms_with_icp(data_dir: Path, *, dry_run: bool = False) -> dict:
    data_dir = Path(data_dir).resolve()
    static_dir = data_dir / "static_scene"
    transforms_path = static_dir / "transforms.json"
    backup_path = static_dir / "transforms_urdf_backup.json"

    if not transforms_path.exists():
        raise FileNotFoundError(f"transforms.json not found: {transforms_path}")

    meta = json.loads(transforms_path.read_text())
    frames = meta.get("frames", [])
    if not frames:
        raise RuntimeError(f"no frames in {transforms_path}")

    # Backup original FK transforms. Skip if a backup already exists so
    # repeated runs don't clobber the URDF ground-truth.
    if not backup_path.exists():
        shutil.copy2(transforms_path, backup_path)
        print(f"[icp-rewrite] backed up URDF transforms → {backup_path}")
    else:
        print(f"[icp-rewrite] backup already exists at {backup_path} (not overwriting)")

    fx, fy, cx, cy, W, H = _load_intrinsics(meta)
    depth_scale = float(meta.get("depth_unit_scale_factor", 1e-3))

    print(f"[icp-rewrite] {len(frames)} frames, {W}x{H}, depth_unit_scale_factor={depth_scale}")

    # Build a fresh OnlineFusion. The first add_frame call seeds the TSDF;
    # its returned "refined" pose is the caller's input verbatim (no ICP
    # for the seed). Subsequent calls run ICP against the accumulated model
    # and return the refined pose IN OPENCV — we flip back to OpenGL via
    # _convert_refined_to_gl before writing transforms.json.
    fuser = OnlineFusion(fx, fy, cx, cy, W, H)
    print(f"[icp-rewrite] OnlineFusion device={fuser.device}")

    refined_per_frame: list[np.ndarray] = []
    deltas_mm: list[float] = []
    deltas_deg: list[float] = []
    t0 = time.time()

    for i, fr in enumerate(frames):
        rgb_path = (static_dir / fr["file_path"].lstrip("./")).resolve()
        depth_path = (static_dir / fr["depth_file_path"].lstrip("./")).resolve()
        if not rgb_path.exists():
            raise FileNotFoundError(rgb_path)
        if not depth_path.exists():
            raise FileNotFoundError(depth_path)

        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f"failed to read {rgb_path}")
        rgb = rgb[:, :, ::-1].copy()  # BGR → RGB

        # OnlineFusion expects uint16 mm depth with gripper pixels zeroed.
        depth_u16 = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_u16 is None:
            raise RuntimeError(f"failed to read {depth_path}")
        if depth_u16.dtype != np.uint16:
            # Should already be uint16 mm on disk; convert defensively.
            depth_u16 = (depth_u16.astype(np.float32) * depth_scale * 1000.0).astype(np.uint16)

        mask = _maybe_load_mask(static_dir, fr)
        if mask is not None:
            depth_u16 = np.where(mask > 0, depth_u16, np.uint16(0))

        c2w_gl = np.asarray(fr["transform_matrix"], dtype=np.float64)

        # add_frame(depth_u16, c2w_opengl, rgb_u8); returns refined CV c2w.
        refined_cv = fuser.add_frame(depth_u16, c2w_gl, rgb)
        refined_gl = _convert_refined_to_gl(refined_cv)
        refined_per_frame.append(refined_gl)

        # Drift stats vs URDF (OpenGL frame; translation in metres,
        # rotation via the trace-formula angle of the relative rotation).
        dt = float(np.linalg.norm(refined_gl[:3, 3] - c2w_gl[:3, 3]) * 1000.0)
        R_delta = refined_gl[:3, :3].T @ c2w_gl[:3, :3]
        cos_theta = (np.trace(R_delta) - 1.0) / 2.0
        cos_theta = max(-1.0, min(1.0, cos_theta))
        dR = float(np.degrees(np.arccos(cos_theta)))
        deltas_mm.append(dt); deltas_deg.append(dR)
        if (i % 10 == 0) or (i == len(frames) - 1):
            print(f"[icp-rewrite] frame {i:3d}/{len(frames)} dt={dt:7.3f} mm dR={dR:6.3f} deg")

    elapsed = time.time() - t0
    print(f"[icp-rewrite] ICP done in {elapsed:.1f}s")
    print(f"[icp-rewrite] drift stats — dt(mm): mean={np.mean(deltas_mm):.3f} "
          f"median={np.median(deltas_mm):.3f} p90={np.percentile(deltas_mm, 90):.3f} "
          f"max={np.max(deltas_mm):.3f}")
    print(f"[icp-rewrite] drift stats — dR(deg): mean={np.mean(deltas_deg):.4f} "
          f"median={np.median(deltas_deg):.4f} p90={np.percentile(deltas_deg, 90):.4f} "
          f"max={np.max(deltas_deg):.4f}")

    # Build the rewritten meta. Preserve everything except per-frame
    # transform_matrix. Frame 0 keeps URDF (anchor of the global frame —
    # add_frame returns it verbatim when idx==0, but we skip explicitly
    # to make the invariant obvious).
    new_meta = json.loads(json.dumps(meta))  # deep copy via JSON round-trip
    for i, fr in enumerate(new_meta["frames"]):
        if i == 0:
            continue
        fr["transform_matrix"] = refined_per_frame[i].tolist()
    new_meta["pose_source"] = "icp_refined_from_urdf_v1"

    if dry_run:
        print("[icp-rewrite] dry run — NOT writing transforms.json")
        return {"frames": len(frames), "dt_mm_median": float(np.median(deltas_mm))}

    tmp = transforms_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(new_meta, indent=2) + "\n")
    tmp.replace(transforms_path)
    print(f"[icp-rewrite] wrote refined transforms → {transforms_path}")

    return {
        "frames": len(frames),
        "dt_mm": deltas_mm,
        "dR_deg": deltas_deg,
        "elapsed_seconds": elapsed,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("data_dir", type=Path, help="Dataset root (contains static_scene/)")
    p.add_argument("--dry-run", action="store_true", help="Run ICP but do not overwrite transforms.json")
    return p.parse_args()


def _main() -> int:
    args = _parse_args()
    rewrite_transforms_with_icp(args.data_dir, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
