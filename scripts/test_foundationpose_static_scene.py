"""Standalone smoke test for the FoundationPoseTracker.

Iterates over the ``static_scene/`` frames of the supplied dataset (the
test dataset hard-coded below by default), seeds the FP tracker from the
mesh-to-world transform persisted in
``dynamic_scene/initialization_artifacts/phase0_manifest.json``, and
checks that the world-frame ``(R, t)`` returned for each subsequent
frame stays near identity (the object is physically stationary across
``static_scene/``). Reports per-frame timing and steady-state FPS.

Run with the ``foundationpose`` conda env so all FP runtime deps are
available::

    conda run -n foundationpose python scripts/test_foundationpose_static_scene.py

Pass ``--data <path>`` to override the dataset. Pass
``--instance-id <id>`` to test a different fused instance (default: 1
which is ``static0_obj_00``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_DATA = (
    "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/"
    "dynamic_gs_test_2026-03-28_19-49-45_w_background"
)


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.ascontiguousarray(np.array(img, dtype=np.uint8))


def _load_depth_meters(path: Path) -> np.ndarray:
    """Load uint16 mm depth TIFF and convert to float32 meters."""
    depth = np.array(Image.open(path))
    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) / 1000.0
    elif depth.dtype in (np.float32, np.float64):
        depth = depth.astype(np.float32)
    else:
        depth = depth.astype(np.float32) / 1000.0
    return np.ascontiguousarray(depth)


def _ns_c2w_to_cv(transform_matrix: list) -> np.ndarray:
    """Convert nerfstudio (y up, z back) c2w to FP (y down, z forward) c2w."""
    ns_c2w = np.asarray(transform_matrix, dtype=np.float64).reshape(4, 4)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return ns_c2w @ flip


def _rotation_deg_from_identity(R: np.ndarray) -> float:
    """Geodesic angle (deg) between R and identity."""
    cos_theta = (np.trace(R) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--instance-id", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0, help="0 = all")
    parser.add_argument("--init-refine-iter", type=int, default=6)
    parser.add_argument("--track-refine-iter", type=int, default=2)
    parser.add_argument(
        "--rotation-tol-deg",
        type=float,
        default=5.0,
        help="Sanity-check threshold for max rotation deviation in degrees",
    )
    parser.add_argument(
        "--translation-tol-cm",
        type=float,
        default=5.0,
        help="Sanity-check threshold for max translation deviation in centimetres",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    _ensure_repo_on_path()
    from dynamic_gs.utils.foundationpose_tracker import FoundationPoseTracker

    data_root = Path(args.data)
    static_dir = data_root / "static_scene"
    artifacts = data_root / "dynamic_scene" / "initialization_artifacts"
    manifest_path = artifacts / "phase0_manifest.json"
    if not manifest_path.exists():
        print(f"[FAIL] manifest missing: {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text())
    entry = manifest.get(str(args.instance_id))
    if entry is None:
        print(
            f"[FAIL] no manifest entry for instance_id={args.instance_id}; "
            f"available: {list(manifest.keys())}",
            file=sys.stderr,
        )
        return 2
    mesh_path = entry.get("mesh_path")
    mesh_to_world = entry.get("mesh_to_world_4x4")
    if not mesh_path or not Path(mesh_path).exists():
        print(
            f"[FAIL] manifest entry missing mesh_path or file not found: {mesh_path}\n"
            f"       Re-run Phase 0 with the SAM3D mesh decoder enabled.",
            file=sys.stderr,
        )
        return 2
    if mesh_to_world is None:
        print(
            f"[FAIL] manifest entry missing mesh_to_world_4x4 — re-run Phase 0 "
            f"so _fuse_sam3d_objects_into_scene writes the post-CPD transform.",
            file=sys.stderr,
        )
        return 2

    transforms_path = static_dir / "transforms.json"
    if not transforms_path.exists():
        print(f"[FAIL] static_scene/transforms.json missing: {transforms_path}", file=sys.stderr)
        return 2
    transforms = json.loads(transforms_path.read_text())
    K = np.array(
        [[transforms["fl_x"], 0.0, transforms["cx"]],
         [0.0, transforms["fl_y"], transforms["cy"]],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    frames = transforms["frames"]
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    if len(frames) < 2:
        print("[FAIL] need at least 2 frames to test tracking", file=sys.stderr)
        return 2

    tracker = FoundationPoseTracker(
        mesh_path=mesh_path,
        mesh_to_world=np.asarray(mesh_to_world, dtype=np.float64),
        mesh_unit_scale=1.0,
    )

    rot_devs_deg: list[float] = []
    trans_devs_cm: list[float] = []
    track_times_ms: list[float] = []

    print(f"[FP-test] frames={len(frames)} mesh={mesh_path} instance={args.instance_id}")

    for i, frame in enumerate(frames):
        rgb_path = static_dir / frame["file_path"]
        depth_path = static_dir / frame["depth_file_path"]
        if not rgb_path.exists() or not depth_path.exists():
            print(f"[FP-test] skip frame {i}: file missing")
            continue
        rgb = _load_rgb(rgb_path)
        depth = _load_depth_meters(depth_path)
        c2w = _ns_c2w_to_cv(frame["transform_matrix"])

        t0 = time.time()
        if i == 0:
            R, t = tracker.initialize_from_known_pose(
                rgb=rgb,
                depth=depth,
                K=K,
                camera_to_world=c2w,
                refine_iterations=args.init_refine_iter,
            )
            init_ms = (time.time() - t0) * 1000.0
        else:
            R, t = tracker.track_one(
                rgb=rgb,
                depth=depth,
                K=K,
                camera_to_world=c2w,
                iterations=args.track_refine_iter,
            )
            track_times_ms.append((time.time() - t0) * 1000.0)

        rot_dev = _rotation_deg_from_identity(R)
        trans_dev_cm = float(np.linalg.norm(t)) * 100.0
        rot_devs_deg.append(rot_dev)
        trans_devs_cm.append(trans_dev_cm)
        if i == 0:
            print(
                f"[FP-test] frame {i:03d} init: rot_dev={rot_dev:.3f}deg "
                f"trans_dev={trans_dev_cm:.3f}cm init_ms={init_ms:.1f}"
            )
        else:
            print(
                f"[FP-test] frame {i:03d}: rot_dev={rot_dev:.3f}deg "
                f"trans_dev={trans_dev_cm:.3f}cm track_ms={track_times_ms[-1]:.1f}"
            )

    if not track_times_ms:
        print("[FAIL] no track_one calls completed", file=sys.stderr)
        return 2
    avg_track_ms = float(np.mean(track_times_ms))
    avg_fps = 1000.0 / avg_track_ms if avg_track_ms > 0 else float("inf")
    max_rot = max(rot_devs_deg)
    max_trans = max(trans_devs_cm)
    print("\n[FP-test] === summary ===")
    print(f"frames_tracked: {len(track_times_ms)}")
    print(f"max_rotation_deviation_deg: {max_rot:.3f}")
    print(f"max_translation_deviation_cm: {max_trans:.3f}")
    print(f"avg_track_ms: {avg_track_ms:.2f}")
    print(f"avg_fps: {avg_fps:.2f}")

    rc = 0
    if max_rot > args.rotation_tol_deg:
        print(
            f"[WARN] max rotation {max_rot:.3f}deg exceeds tolerance "
            f"{args.rotation_tol_deg:.3f}deg",
            file=sys.stderr,
        )
        rc = 1
    if max_trans > args.translation_tol_cm:
        print(
            f"[WARN] max translation {max_trans:.3f}cm exceeds tolerance "
            f"{args.translation_tol_cm:.3f}cm",
            file=sys.stderr,
        )
        rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
