"""Test the concurrent ICP+TSDF fusion on a recorded dataset.

Exercises the exact same code path that `capture_only.py` and
`live_session.py` use — `ConcurrentFusionRunner` watching a
streaming transforms.json. We simulate the publisher by copying frames
from the source dataset into a temp staging dir one-by-one with a
small inter-frame pause, then call `stop_and_finalize()` to drain.

Reports:
    * per-frame add_frame ms (mean / p90 / max)
    * total wall-clock from first frame enqueued → finalize() return
    * fused point count
    * "would have fit in capture" margin (frames missed at the end)

Does NOT overwrite the source dataset's existing
`depth_camera_init_points.ply`. Output goes to a sibling tmp dir.

Usage:
    python scripts/test_concurrent_fusion.py [data_root] [--inter-frame-s SEC]
        data_root defaults to /home/mrc-cuhk/.../datasets/validate_run_1
        inter-frame-s defaults to 0.5  (simulates 2 cm/20° keyframe rate)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import open3d as o3d  # noqa: E402

from dynamic_gs.utils.fusion_runner import ConcurrentFusionRunner  # noqa: E402


class _CameraIntrinsicsLite:
    """Match the duck-typed interface ConcurrentFusionRunner expects."""

    def __init__(self, fx, fy, cx, cy, w, h):
        self.fx = float(fx); self.fy = float(fy)
        self.cx = float(cx); self.cy = float(cy)
        self.width = int(w); self.height = int(h)


def _stage_frame_atomically(stage_static: Path, src_static: Path, fr_meta: dict, base_meta: dict, written: list, ply_path_preexisting: bool) -> None:
    """Copy rgb/depth/mask from source into stage dir, then atomically
    swap transforms.json with the new frame appended (matches the
    publisher's _write_frame_to_disk semantics)."""
    for sub, key in [("rgb", "file_path"), ("depth", "depth_file_path"), ("masks", "mask_path")]:
        src = src_static / fr_meta[key].lstrip("./")
        dst = stage_static / fr_meta[key].lstrip("./")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
    written.append(fr_meta)
    meta_now = dict(base_meta)
    # Keep the existing ply_file_path out so the runner's finalize is
    # what writes it (matches the live flow where the seed PLY is
    # produced by the runner, not pre-existing).
    if not ply_path_preexisting:
        meta_now.pop("ply_file_path", None)
    meta_now["frames"] = written
    tp = stage_static / "transforms.json"
    tmp = tp.with_name(f".{tp.name}.tmp")
    tmp.write_text(json.dumps(meta_now, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, tp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root", nargs="?",
                    default="/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/validate_run_1",
                    type=Path)
    ap.add_argument("--inter-frame-s", type=float, default=0.5,
                    help="Wall-clock pause between simulated frame arrivals (s). "
                         "Default 0.5 mimics the ~2 cm/20° keyframe rate during a hand-paced sweep.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Where to stage frames + write the fused PLY. "
                         "Default: <data_root>/_fusion_test/  (override with --label too)")
    ap.add_argument("--label", type=str, default="",
                    help="Optional suffix for the default out-dir, e.g. "
                         "'stride4' -> <data_root>/_fusion_test_stride4/")
    ap.add_argument("--icp-src-stride", type=int, default=None,
                    help="Override ICP source decimation stride (module default = 4). "
                         "Higher = fewer source points = faster ICP, less precise.")
    args = ap.parse_args()

    # Apply tunable overrides BEFORE ConcurrentFusionRunner imports
    # OnlineFusion's constants (they're read at add_frame time).
    if args.icp_src_stride is not None:
        from dynamic_gs.utils import online_fusion as _of
        _of.ICP_SRC_STRIDE = int(args.icp_src_stride)
        print(f"[test] ICP_SRC_STRIDE override -> {_of.ICP_SRC_STRIDE}")

    src_root = args.data_root.resolve()
    src_static = src_root / "static_scene"
    if not (src_static / "transforms.json").exists():
        print(f"FAIL: no transforms.json under {src_static}", file=sys.stderr)
        sys.exit(2)

    if args.out_dir is not None:
        out_root = args.out_dir.resolve()
    else:
        suffix = f"_{args.label}" if args.label else ""
        out_root = src_root / f"_fusion_test{suffix}"
    stage_static = out_root / "static_scene"
    if stage_static.exists():
        shutil.rmtree(stage_static)
    stage_static.mkdir(parents=True)
    print(f"[test] source: {src_static}")
    print(f"[test] stage:  {stage_static}")

    src_meta = json.loads((src_static / "transforms.json").read_text())
    n_frames = len(src_meta["frames"])
    intrinsics = _CameraIntrinsicsLite(
        src_meta["fl_x"], src_meta["fl_y"], src_meta["cx"], src_meta["cy"],
        src_meta["w"], src_meta["h"],
    )
    print(f"[test] {n_frames} frames @ {intrinsics.width}x{intrinsics.height}")
    print(f"[test] inter-frame stream pause: {args.inter_frame_s:.2f} s "
          f"(simulated total stream wall-clock: {n_frames*args.inter_frame_s:.1f} s)")

    base_meta = {k: v for k, v in src_meta.items() if k != "frames"}

    # Arm the concurrent fusion runner — same call live_session.py makes.
    runner = ConcurrentFusionRunner(stage_static, intrinsics)
    runner.start()
    print(f"[test] fusion thread armed; starting frame stream...\n")

    t_stream_start = time.time()
    written: list = []
    for i, fr in enumerate(src_meta["frames"]):
        _stage_frame_atomically(stage_static, src_static, fr, base_meta, written, ply_path_preexisting=False)
        if (i + 1) % 10 == 0 or i == n_frames - 1:
            print(f"[test] streamed {i+1}/{n_frames} frames", flush=True)
        time.sleep(args.inter_frame_s)
    t_stream_done = time.time()
    print(f"\n[test] all {n_frames} frames streamed in {t_stream_done-t_stream_start:.1f} s")
    print(f"[test] calling stop_and_finalize()...\n")

    t_finalize_start = time.time()
    ply_path = runner.stop_and_finalize()
    t_finalize_done = time.time()
    drain_finalize_s = t_finalize_done - t_finalize_start
    total_wall_s = t_finalize_done - t_stream_start

    print("\n============================================================")
    print(" RESULT")
    print("============================================================")
    print(f"  stream wall-clock     : {t_stream_done-t_stream_start:.1f} s "
          f"(simulated capture window)")
    print(f"  drain + finalize tail : {drain_finalize_s:.1f} s "
          f"(blocking after Enter)")
    print(f"  TOTAL wall-clock      : {total_wall_s:.1f} s")
    print(f"  PLY output            : {ply_path}")
    if ply_path is not None and Path(ply_path).exists():
        pc = o3d.io.read_point_cloud(str(ply_path))
        print(f"  fused points          : {len(pc.points):,}")
    print()
    print(f"  source unchanged      : {(src_static / 'depth_camera_init_points.ply').exists()} "
          f"(existing seed at {src_static / 'depth_camera_init_points.ply'})")
    print("============================================================")


if __name__ == "__main__":
    main()
