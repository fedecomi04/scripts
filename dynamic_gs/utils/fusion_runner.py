"""Concurrent ICP+TSDF fusion runner shared by ``capture_only.py`` and
``live_session.py``.

The publisher writes accepted keyframes to disk and atomically swaps
``static_scene/transforms.json`` (tmpfile + ``os.replace``). The
``ConcurrentFusionRunner`` watches that file, enqueues each newly-
written index, and feeds it to ``OnlineFusion.add_frame()`` on a
worker thread. When the caller calls ``stop_and_finalize()``:

  1. The watcher does one last sweep of ``transforms.json`` (so frames
     written between the last poll and stop are not lost).
  2. The worker drains the queue (blocking; logs "draining: N pending"
     every 2 s for visibility).
  3. ``OnlineFusion.finalize()`` runs (~0.6 s).
  4. The fused cloud is written to
     ``<static_dir>/depth_camera_init_points.ply``.
  5. ``transforms.json["ply_file_path"]`` is set so the Splatfacto
     dataparser picks the seed up unchanged.

Usage:
    runner = ConcurrentFusionRunner(static_dir, intrinsics)
    runner.start()
    # ... publisher records keyframes concurrently ...
    ply_path = runner.stop_and_finalize()   # blocks until drained + finalized

Pass ``intrinsics`` as anything with ``fx``, ``fy``, ``cx``, ``cy``,
``width``, ``height`` attributes (matches ``CameraIntrinsicsLite`` from
``live_shm_reader``).
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import open3d as o3d

from .online_fusion import (
    FAR_VOXEL_M,
    NEAR_RADIUS_M,
    OnlineFusion,
    WITH_COLOR,
    adaptive_downsample,
)

INIT_PLY_NAME = "depth_camera_init_points.ply"


# ---------------------------------------------------------------------------
# Worker + watcher
# ---------------------------------------------------------------------------


class _FusionWorker(threading.Thread):
    """Drains a queue of frame tuples; runs ICP+TSDF per item.

    Sentinel ``None`` on the queue tells the worker to exit. Per-frame
    timings (ms) and failure count are exposed for the final summary.
    """

    def __init__(self, fuser: OnlineFusion, q: "queue.Queue") -> None:
        super().__init__(name="fusion_worker", daemon=True)
        self.fuser = fuser
        self.q = q
        self.timings_ms: List[float] = []
        self.fail_count = 0

    def run(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                return
            frame_idx, depth_path, rgb_path, mask_path, c2w_opengl = item
            try:
                t = time.time()
                depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                if depth is None:
                    raise RuntimeError(f"failed to read depth {depth_path}")
                depth = depth.astype(np.uint16).copy()
                if mask_path is not None:
                    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if m is not None:
                        depth[m == 0] = 0
                rgb_u8 = None
                if WITH_COLOR:
                    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                    if bgr is not None:
                        rgb_u8 = bgr[:, :, ::-1].copy()
                self.fuser.add_frame(depth, np.asarray(c2w_opengl, dtype=np.float64), rgb_u8)
                self.timings_ms.append(1000.0 * (time.time() - t))
            except Exception as exc:
                self.fail_count += 1
                print(f"[fusion] frame {frame_idx} failed: {exc}", flush=True)
            finally:
                self.q.task_done()


class _FrameWatcher(threading.Thread):
    """Polls ``<static_dir>/transforms.json``; enqueues newly-written frames.

    The publisher writes rgb/depth/mask BEFORE the atomic transforms.json
    swap, so a strictly-increasing ``len(meta["frames"])`` is a safe
    "frame N is fully on disk" signal.
    """

    def __init__(
        self,
        static_dir: Path,
        q: "queue.Queue",
        stop_evt: threading.Event,
        poll_period_s: float = 0.25,
    ) -> None:
        super().__init__(name="frame_watcher", daemon=True)
        self.static_dir = static_dir
        self.q = q
        self.stop_evt = stop_evt
        self.poll_period_s = float(poll_period_s)
        self._last_count = 0

    def _enqueue_from_meta(self, frames: list) -> None:
        for idx in range(self._last_count, len(frames)):
            fr = frames[idx]
            depth_path = (self.static_dir / fr["depth_file_path"].lstrip("./")).resolve()
            rgb_path = (self.static_dir / fr["file_path"].lstrip("./")).resolve()
            mask_path = (
                (self.static_dir / fr["mask_path"].lstrip("./")).resolve()
                if fr.get("mask_path") else None
            )
            c2w = np.asarray(fr["transform_matrix"], dtype=np.float64)
            self.q.put((idx, depth_path, rgb_path, mask_path, c2w))
        self._last_count = len(frames)

    def run(self) -> None:
        tp = self.static_dir / "transforms.json"
        while not self.stop_evt.is_set():
            try:
                if tp.exists():
                    meta = json.loads(tp.read_text())
                    frames = meta.get("frames", [])
                    if len(frames) > self._last_count:
                        self._enqueue_from_meta(frames)
            except Exception as exc:
                # Race window: the publisher may rewrite transforms.json
                # while we read it. os.replace is atomic on POSIX, but a
                # partial JSON in the rare unlucky read is possible —
                # just retry next tick.
                print(f"[watcher] transient: {exc}", flush=True)
            time.sleep(self.poll_period_s)
        # Final sweep: pick up anything written between the last poll
        # and the stop signal.
        try:
            if tp.exists():
                meta = json.loads(tp.read_text())
                self._enqueue_from_meta(meta.get("frames", []))
        except Exception as exc:
            print(f"[watcher] final pass failed: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ConcurrentFusionRunner:
    """Owns the watcher + worker + ``OnlineFusion``. Call once per
    capture session.

    Lifecycle:
        runner = ConcurrentFusionRunner(static_dir, intrinsics)
        runner.start()
        ... user-controlled recording window ...
        ply_path = runner.stop_and_finalize()  # blocks until done
    """

    def __init__(
        self,
        static_dir: Path,
        intrinsics,
        poll_period_s: float = 0.25,
    ) -> None:
        self.static_dir = Path(static_dir)
        self.intrinsics = intrinsics
        self.poll_period_s = poll_period_s
        self._fuser: Optional[OnlineFusion] = None
        self._q: Optional[queue.Queue] = None
        self._watcher: Optional[_FrameWatcher] = None
        self._worker: Optional[_FusionWorker] = None
        self._watcher_stop_evt: Optional[threading.Event] = None
        self._started = False
        self._finalized = False

    def start(self) -> None:
        if self._started:
            return
        self._fuser = OnlineFusion(
            self.intrinsics.fx, self.intrinsics.fy,
            self.intrinsics.cx, self.intrinsics.cy,
            self.intrinsics.width, self.intrinsics.height,
        )
        self._q = queue.Queue()
        self._watcher_stop_evt = threading.Event()
        self._watcher = _FrameWatcher(
            self.static_dir, self._q, self._watcher_stop_evt,
            poll_period_s=self.poll_period_s,
        )
        self._worker = _FusionWorker(self._fuser, self._q)
        self._worker.start()
        self._watcher.start()
        self._started = True

    def _last_camera_world_xyz(self) -> Optional[np.ndarray]:
        """Return the world-frame position of the LAST captured frame's
        camera, read from ``<static_dir>/transforms.json``. Returns
        ``None`` if the file is missing or has no frames."""
        tp = self.static_dir / "transforms.json"
        if not tp.exists():
            return None
        try:
            meta = json.loads(tp.read_text())
            frames = meta.get("frames", [])
            if not frames:
                return None
            T = np.asarray(frames[-1]["transform_matrix"], dtype=np.float64)
            return T[:3, 3].copy()
        except Exception:
            return None

    def stop_and_finalize(self) -> Optional[Path]:
        """Stop the watcher, drain the queue, run ``finalize()``, write
        the PLY, update transforms.json. Returns the PLY path (or None
        if start() was never called).

        Blocks until ICP+TSDF has consumed every queued frame.
        """
        if not self._started or self._finalized:
            return None
        self._finalized = True
        assert self._fuser is not None
        assert self._q is not None
        assert self._worker is not None
        assert self._watcher is not None
        assert self._watcher_stop_evt is not None

        # Stop watcher (it flushes its tail). M3: unbounded join so a slow
        # final read can't drop frames silently. If the watcher is genuinely
        # wedged, a 30 s hard timeout still lets us proceed rather than hang
        # the user — but we now report it loudly instead of dropping data.
        self._watcher_stop_evt.set()
        self._watcher.join(timeout=30.0)
        if self._watcher.is_alive():
            print("[fusion] WARNING: watcher thread did not exit within 30 s; final keyframes may be dropped", flush=True)
        # Sentinel to exit worker after queue drains.
        self._q.put(None)
        t_drain = time.time()
        while self._worker.is_alive():
            qsize = self._q.qsize()
            if qsize > 0:
                print(f"[fusion] draining queue: {qsize} frames pending...", flush=True)
                time.sleep(2.0)
            else:
                self._worker.join(timeout=2.0)
        drain_s = time.time() - t_drain

        timings = self._worker.timings_ms
        if timings:
            arr = np.asarray(timings)
            print(
                f"[fusion] per-frame add_frame: mean {arr.mean():.0f} ms  "
                f"p90 {np.percentile(arr, 90):.0f} ms  "
                f"max {arr.max():.0f} ms  n={arr.size}  fail={self._worker.fail_count}",
                flush=True,
            )
        print(f"[fusion] drained in {drain_s:.1f} s; calling finalize()...", flush=True)
        t_fin = time.time()
        pc = self._fuser.finalize()
        n_full = len(pc.points)
        print(
            f"[fusion] finalize() {time.time()-t_fin:.1f} s; fused {n_full:,} pts",
            flush=True,
        )

        # Adaptive near/far downsample: keep <NEAR_RADIUS_M of the LAST
        # camera pose at native TSDF density, voxel-downsample the rest
        # to FAR_VOXEL_M. Sub-second; shrinks the Splatfacto seed ~9×.
        last_cam = self._last_camera_world_xyz()
        if last_cam is not None and n_full > 0:
            t_ds = time.time()
            pc = adaptive_downsample(pc, last_cam, NEAR_RADIUS_M, FAR_VOXEL_M)
            n_out = len(pc.points)
            print(
                f"[fusion] adaptive downsample ({time.time()-t_ds:.2f} s): "
                f"{n_full:,} → {n_out:,}  "
                f"(near<{NEAR_RADIUS_M:.1f}m full, far→{FAR_VOXEL_M*1000:.1f}mm voxel)",
                flush=True,
            )
        else:
            print("[fusion] adaptive downsample SKIPPED (no transforms.json / empty cloud)", flush=True)

        ply_path = self.static_dir / INIT_PLY_NAME
        o3d.io.write_point_cloud(str(ply_path), pc)
        print(f"[fusion] wrote {len(pc.points):,} pts → {ply_path}", flush=True)

        # Point transforms.json at the seed PLY so the Splatfacto
        # dataparser picks it up unchanged.
        tp = self.static_dir / "transforms.json"
        if tp.exists():
            meta = json.loads(tp.read_text())
            meta["ply_file_path"] = INIT_PLY_NAME
            tmp = tp.with_name(f".{tp.name}.tmp")
            tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, tp)
            # M1: bump PLY mtime past transforms.json so dynamic_gs.utils.
            # rgbd_fusion_init._output_is_fresh treats the GPU-generated seed
            # as up-to-date. Without this, every fresh capture leaves
            # transforms.json.mtime > out_ply.mtime → the redundant
            # bootstrap_live.sh re-fusion overwrites the 1.5 mm GPU seed
            # with a 2.5 mm CPU one.
            os.utime(ply_path, None)
        return ply_path
