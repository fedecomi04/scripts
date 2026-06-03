"""Capture-only dataset recorder — no SAM3 / SAM3D / training.

Flow:
    1. Spawn the ROS publisher (same `LiveShmSubscriber` as bootstrap_live).
       Wait for first synced frame.
    2. STATIC phase: publisher's dedup-filtered recorder (2 cm OR 20°)
       writes accepted keyframes to `<data_dir>/static_scene/`. A
       concurrent fusion worker watches `transforms.json` and feeds
       each newly-written frame into `OnlineFusion.add_frame()` in the
       background. Press ENTER to end. At end: drain the queue, call
       `finalize()`, write `depth_camera_init_points.ply` directly.
    3. DYNAMIC phase: reader polls SHM at 30 fps from this process and
       writes every frame to `<data_dir>/dynamic_scene/` (no dedup).
       Press ENTER to stop and exit.

Concurrent fusion design:
    * Watcher thread polls `static_scene/transforms.json` every 250 ms.
      The publisher's `_write_frame_to_disk` writes rgb/depth/mask
      first, then atomically swaps in the new transforms.json
      (tmpfile + os.replace). So when `len(meta["frames"])` ticks up,
      every byte of frame N is on disk.
    * For each new index, watcher enqueues `(frame_idx, depth_path,
      rgb_path, mask_path, c2w_4x4)` onto a queue.
    * Fusion thread drains the queue: load depth+mask+rgb from disk,
      zero gripper pixels, call `OnlineFusion.add_frame()`. Per-frame
      cost ~250 ms — fits the inter-keyframe gap; if it falls behind,
      the queue grows and the static-end finalize blocks until drained.

Output layout (matches `bootstrap_live.sh`):

    <data_dir>/
        static_scene/
            rgb/        frame_NNNNN.png        (BGR uint8)
            depth/      frame_NNNNN.tiff       (uint16 mm)
            masks/      frame_NNNNN.png        (uint8, 255 = keep)
            transforms.json                    (updated with ply_file_path)
            depth_camera_init_points.ply       (TSDF-fused seed, RGB-coloured)
        dynamic_scene/
            rgb/  depth/  masks/  transforms.json

Usage:
    scripts/capture_only.sh                       # default: datasets/<timestamp>/
    scripts/capture_only.sh <dir>                 # bare name → datasets/<dir>/
    scripts/capture_only.sh /abs/path             # absolute → as-is
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# Make `dynamic_gs` importable when this file is run directly from the
# scripts/ subdir, before the package is installed.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dynamic_gs.utils.fusion_runner import ConcurrentFusionRunner  # noqa: E402
from dynamic_gs.utils.live_shm_reader import LiveShmSubscriber, LiveFrame  # noqa: E402

IMAGE_NAME_PREFIX = "frame"


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _wipe_or_confirm(data_dir: Path) -> None:
    """Wipe the dataset dir if non-empty, after a y/N confirm."""
    if data_dir.exists():
        contents = list(data_dir.iterdir())
        if contents:
            print(f"[capture] target dir is non-empty ({len(contents)} entries): {data_dir}", flush=True)
            reply = input("       wipe it? [y/N]: ").strip().lower()
            if reply not in ("y", "yes"):
                print("[capture] aborting (target not wiped)", flush=True)
                sys.exit(1)
            shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)


def _enter_listener(stop_evt: threading.Event) -> threading.Thread:
    """Background thread: signal `stop_evt` on the next Enter on stdin."""
    def _wait():
        try:
            sys.stdin.readline()
        except Exception:
            pass
        stop_evt.set()
    t = threading.Thread(target=_wait, name="enter_listener", daemon=True)
    t.start()
    return t


def _write_frame_dynamic(frame: LiveFrame, dyn_dir: Path, meta: dict, written: list) -> None:
    """Write a single frame to dynamic_scene/ and update transforms.json."""
    frame_index = len(written)
    stem = f"{IMAGE_NAME_PREFIX}_{frame_index:05d}"
    cv2.imwrite(str(dyn_dir / "rgb" / f"{stem}.png"), frame.rgb_bgr)
    depth_mm_u16 = np.clip(frame.depth_m * 1000.0, 0.0, 65535.0).astype(np.uint16)
    cv2.imwrite(str(dyn_dir / "depth" / f"{stem}.tiff"), depth_mm_u16)
    cv2.imwrite(str(dyn_dir / "masks" / f"{stem}.png"), frame.mask_keep)
    written.append({
        "file_path": f"./rgb/{stem}.png",
        "depth_file_path": f"./depth/{stem}.tiff",
        "mask_path": f"./masks/{stem}.png",
        "transform_matrix": frame.c2w_4x4.tolist(),
    })
    meta["frames"] = written
    tp = dyn_dir / "transforms.json"
    tmp = tp.with_name(f".{tp.name}.tmp")
    tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, tp)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", type=Path, help="Output dataset dir (will be wiped after confirm).")
    parser.add_argument("--dynamic-fps", type=float, default=30.0,
                        help="Polling rate for the dynamic phase (Hz). Default: 30.")
    parser.add_argument("--static-translation-m", type=float, default=0.02,
                        help="Static-phase keyframe dedup translation threshold (m). Default: 0.02 (2 cm).")
    parser.add_argument("--static-rotation-deg", type=float, default=20.0,
                        help="Static-phase keyframe dedup rotation threshold (deg). Default: 20.")
    parser.add_argument("--no-fusion", action="store_true",
                        help="Skip the concurrent ICP+TSDF fusion; no init PLY written. "
                             "Useful for fast capture iteration when you don't need a seed.")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    _wipe_or_confirm(data_dir)

    # The publisher's `LIVE_ROOT` is env-overridable. Point it at our
    # target dataset dir so its `start_recording` writes into
    # `<data_dir>/static_scene/`.
    os.environ["DGS_LIVE_ROOT"] = str(data_dir)

    print(f"\n========================================================")
    print(f"  capture_only :: {data_dir}")
    print(f"  static dedup : {args.static_translation_m * 100:.1f} cm  OR  {args.static_rotation_deg:.0f}°")
    print(f"  dynamic fps  : {args.dynamic_fps:.0f}")
    print(f"  fusion       : {'OFF' if args.no_fusion else 'ON (concurrent ICP+TSDF)'}")
    print(f"========================================================\n")

    # ------------------------------------------------------------------
    # Publisher + first frame
    # ------------------------------------------------------------------
    print("[capture] spawning ROS publisher subprocess...", flush=True)
    sub = LiveShmSubscriber(
        live_root=data_dir,
        keyframe_translation_m=args.static_translation_m,
        keyframe_rotation_deg=args.static_rotation_deg,
        wipe_live_root=True,
    )

    # Make absolutely sure the publisher dies on any exit path — Ctrl-C,
    # exception, normal return. Without this the orphaned publisher keeps
    # subscribing to /camera_info and the NEXT run's publisher times out
    # waiting for the first message.
    def _shutdown(*_unused) -> None:
        try:
            sub.close()
        except Exception:
            pass
    signal.signal(signal.SIGINT,  lambda *_: (_shutdown(), sys.exit(130)))
    signal.signal(signal.SIGTERM, lambda *_: (_shutdown(), sys.exit(143)))

    try:
        sub.wait_for_first_frame(timeout_s=90.0)
        print("[capture] ready!", flush=True)
        _run_capture(sub, args, data_dir)
    finally:
        _shutdown()


def _run_capture(sub, args, data_dir: Path) -> None:
    intrinsics = sub.intrinsics

    # ------------------------------------------------------------------
    # STATIC phase — publisher dedup + concurrent fusion
    # ------------------------------------------------------------------
    static_dir = data_dir / "static_scene"

    fusion_runner: ConcurrentFusionRunner | None = None
    if not args.no_fusion:
        fusion_runner = ConcurrentFusionRunner(static_dir, intrinsics)
        fusion_runner.start()
        print("[capture] online fusion thread armed", flush=True)

    bootstrap_anchor = sub.capture_anchor()
    sub.start_recording(bootstrap_anchor)
    print(f"[capture] STATIC recording started "
          f"(dedup {args.static_translation_m*100:.1f}cm/{args.static_rotation_deg:.0f}°).")
    print("          sweep the scene; press ENTER to switch to DYNAMIC.\n", flush=True)
    input()

    n_static = sub.stop_recording()
    print(f"[capture] STATIC done — {n_static} keyframes written to {static_dir}/", flush=True)

    # Drain + finalize fusion (blocks until ICP+TSDF has caught up).
    if fusion_runner is not None:
        fusion_runner.stop_and_finalize()

    # ------------------------------------------------------------------
    # DYNAMIC phase — reader-side polling, no dedup, fixed FPS
    # ------------------------------------------------------------------
    dyn_dir = data_dir / "dynamic_scene"
    (dyn_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (dyn_dir / "depth").mkdir(parents=True, exist_ok=True)
    (dyn_dir / "masks").mkdir(parents=True, exist_ok=True)

    dyn_meta = {
        "fl_x": intrinsics.fx, "fl_y": intrinsics.fy,
        "cx": intrinsics.cx, "cy": intrinsics.cy,
        "w": intrinsics.width, "h": intrinsics.height,
        "frames": [],
    }
    dyn_written: list = []

    period = 1.0 / max(args.dynamic_fps, 1.0)
    dyn_stop_evt = threading.Event()
    _enter_listener(dyn_stop_evt)
    print(f"\n[capture] DYNAMIC recording started ({args.dynamic_fps:.0f} fps, no dedup).")
    print("          move the object; press ENTER to stop and exit.\n", flush=True)

    last_seq = -1
    last_log = time.time()
    next_tick = time.time()
    while not dyn_stop_evt.is_set():
        now = time.time()
        if now < next_tick:
            time.sleep(min(period * 0.25, next_tick - now))
            continue
        next_tick += period
        if next_tick < now:
            next_tick = now + period

        frame = sub.peek_latest()
        if frame is None or frame.seq == last_seq:
            continue
        last_seq = frame.seq
        _write_frame_dynamic(frame, dyn_dir, dyn_meta, dyn_written)

        if now - last_log >= 2.0:
            print(f"[capture] dynamic frames: {len(dyn_written)}", flush=True)
            last_log = now

    print(f"\n[capture] DYNAMIC done — {len(dyn_written)} frames written to {dyn_dir}/", flush=True)
    print(f"[capture] dataset complete: {data_dir}", flush=True)


if __name__ == "__main__":
    main()
