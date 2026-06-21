"""static_seed_stream.py — build the TSDF seed INCREMENTALLY during the live sweep on CPU.

The old path (_build_seed_deferred) waits until after SAM3D frees the GPU, then spawns a COLD
subprocess that imports torch+Open3D+CUDA from scratch (~12 s) just to TSDF-fuse ~18-30 keyframes
whose actual integrate is ~3 s. This module moves that work onto a background CPU thread that fuses
each kept keyframe AS IT ARRIVES during the operator sweep, so by trigger time only finalize()
remains (~2 s) — and it never touches the GPU SAM3D is using.

Measured (2026-06-21, 18-keyframe live set, 3 mm voxel, ICP_SRC_STRIDE=8):
  per-keyframe add ≈ 340 ms CPU  (ICP-refine + full-res TSDF integrate)
  finalize + adaptive_downsample ≈ 1.7 s at trigger
Keyframes pass the recorder's 2 cm / 20° gate, so they arrive every ~1-3 s of hand-sweep — slower
than the 340 ms it takes to fuse one, so the worker drains in real time. ICP_SRC_STRIDE=8 (vs the
default 4) halves the per-frame cost with NO seed change: A/B vs stride 4 = poses within 0.024 mm,
fused surface within 0.01 mm median (both << the 3 mm voxel). The full-res TSDF integrate is
untouched, so seed geometry is identical to the GPU build.
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .frame import Frame, Intrinsics


class SweepSeedBuilder:
    """Background CPU TSDF seed builder. start() -> submit(frame) per kept keyframe -> finalize().

    Thread-safe submit (non-blocking). All Open3D work happens on the single worker thread, so the
    sweep loop never blocks on fusion. finalize() drains + joins, then writes the seed PLY and
    patches transforms.json's ply_file_path (same output contract as fuse_recorded_dataset)."""

    # Lighter ICP source decimation for the seed: stride 8 (vs the fusion default 4) halves CPU
    # per-frame cost with no measurable seed change (the TSDF integrate stays full-res).
    SEED_ICP_STRIDE = 8
    # MUST match the GPU-subprocess build (_build_seed_deferred sets DGS_TSDF_VOXEL_M=0.003): the
    # online_fusion module default is 2 mm, which yields a DIFFERENT seed (+34% points, +0.4 m taller
    # bbox from kept far-surface fragments) that splatfacto can't fit (loss plateaus ~0.05, blurry).
    SEED_VOXEL_M = 0.003
    SEED_TRUNC_M = 0.008          # ~4x voxel, matches the GPU build's TSDF_TRUNC_M

    def __init__(self, intr: Intrinsics, tm=None):
        self.intr = intr
        self._tm = tm                                     # TimingLedger: per-keyframe ICP + TSDF means
        self._q: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._fuser = None
        self._n_fused = 0
        self._last_cam_world: Optional[np.ndarray] = None
        self._error: Optional[BaseException] = None
        self._started = False

    def start(self) -> None:
        """Construct the CPU OnlineFusion and spawn the worker thread."""
        import os
        os.environ["DGS_FUSION_DEVICE"] = "cpu"          # force the legacy CPU pipeline (no GPU contention)
        import dynamic_gs.utils.online_fusion as OF
        # Override the module globals (read at IMPORT time, so env/setdefault here is too late): match
        # the GPU subprocess build's voxel/trunc EXACTLY so the seed is identical, and lighten ICP.
        OF.TSDF_VOXEL_M = self.SEED_VOXEL_M
        OF.TSDF_TRUNC_M = self.SEED_TRUNC_M
        OF.ICP_SRC_STRIDE = self.SEED_ICP_STRIDE          # lighter ICP source for the seed only
        self._OF = OF
        self._fuser = OF.OnlineFusion(self.intr.fx, self.intr.fy, self.intr.cx, self.intr.cy,
                                      self.intr.width, self.intr.height)
        print(f"[static] seed fusion: voxel={self.SEED_VOXEL_M*1000:.0f}mm trunc={self.SEED_TRUNC_M*1000:.0f}mm "
              f"icp_stride={self.SEED_ICP_STRIDE}", flush=True)
        self._thread = threading.Thread(target=self._run, name="sweep-seed", daemon=True)
        self._thread.start()
        self._started = True
        print(f"[static] incremental CPU TSDF seed builder started (stride={self.SEED_ICP_STRIDE})", flush=True)

    def submit(self, frame: Frame) -> None:
        """Enqueue one kept keyframe. Copies the arrays it needs so the SHM/Frame can be recycled."""
        if not self._started or self._error is not None:
            return
        depth_m = np.nan_to_num(np.asarray(frame.depth_m, dtype=np.float32),
                                nan=0.0, posinf=0.0, neginf=0.0)
        depth_mm = np.clip(depth_m * 1000.0, 0.0, 65535.0).astype(np.uint16)
        keep = np.asarray(frame.mask_keep)
        keep = keep[..., 0] if keep.ndim == 3 else keep
        depth_mm[keep == 0] = 0                            # gripper/robot excluded, same as the recorder
        c2w_gl = np.asarray(frame.c2w_4x4, dtype=np.float64).copy()
        # RGB for color fusion (online_fusion WITH_COLOR=True). Frame.rgb_bgr is BGR; online_fusion
        # wants RGB uint8 (matches fuse_recorded_dataset's cv2.imread(...)[:, :, ::-1]). WITHOUT this
        # the seed is colorless (all-black) -> gaussians init black -> can't match the bright scene ->
        # the static train drives opacity to ~0 -> the end-of-train purge culls ~68% -> holes.
        rgb = frame.rgb_bgr
        rgb_u8 = np.ascontiguousarray(np.asarray(rgb)[..., ::-1], dtype=np.uint8) if rgb is not None else None
        self._q.put((depth_mm, c2w_gl, rgb_u8))

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:                              # sentinel -> drain done
                self._q.task_done()
                return
            depth_mm, c2w_gl, rgb_u8 = item
            try:
                # Per-keyframe ICP vs TSDF split -> ledger aggregate per-step table (mean/max/n),
                # mirroring the tracker-tick timing. record_ms is thread-safe.
                def _timer(stage, ms):
                    if self._tm is not None:
                        self._tm.record_ms(f"seed.{stage}_per_kf", ms)
                refined_cv = self._fuser.add_frame(depth_mm, c2w_gl, rgb_u8, timer=_timer)
                self._last_cam_world = refined_cv[:3, 3].copy()
                self._n_fused += 1
            except BaseException as exc:                   # keep the sweep alive; finalize() reports it
                self._error = exc
            finally:
                self._q.task_done()

    def finalize(self, static_dir) -> Optional[Path]:
        """Stop the worker, extract + downsample the fused cloud, write the seed PLY + patch
        transforms.json. Returns the PLY path, or None if nothing was fused / an error occurred
        (caller then falls back to the GPU subprocess build)."""
        if not self._started:
            return None
        # Time the DRAIN-WAIT separately from the extract: drain = how long finalize blocks for the bg
        # worker to finish any keyframes still queued at trigger (= the part splatfacto would wait on).
        # If the worker kept up with the sweep, this is ~0; a large value means the sweep outran fusion.
        _t_drain = time.time()
        self._q.put(None)                                 # sentinel
        if self._thread is not None:
            self._thread.join(timeout=120.0)
        if self._tm is not None:
            self._tm.record_ms("seed.drain_wait", (time.time() - _t_drain) * 1000.0)
        if self._error is not None:
            print(f"[static] WARNING incremental seed builder errored ({self._error}); "
                  f"caller will fall back to GPU build", flush=True)
            return None
        if self._n_fused == 0:
            print("[static] WARNING incremental seed builder fused 0 frames; falling back", flush=True)
            return None
        import open3d as o3d
        OF = self._OF
        t0 = time.time()
        pc = self._fuser.finalize()
        n_full = len(pc.points)
        if n_full > 0 and self._last_cam_world is not None:
            pc = OF.adaptive_downsample(pc, self._last_cam_world)
        static_dir = Path(static_dir)
        ply_path = static_dir / "depth_camera_init_points.ply"
        o3d.io.write_point_cloud(str(ply_path), pc)
        # Patch transforms.json's ply_file_path so the dataparser picks it up (matches
        # fuse_recorded_dataset's contract).
        import json
        meta_path = static_dir / "transforms.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta["ply_file_path"] = "depth_camera_init_points.ply"
            tmp = meta_path.with_name(f".{meta_path.name}.tmp")
            tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            import os as _os
            _os.replace(tmp, meta_path)
        print(f"[static] incremental CPU TSDF seed: {self._n_fused} keyframes fused, "
              f"{n_full:,} -> {len(pc.points):,} pts, finalize {time.time()-t0:.1f}s -> {ply_path.name}",
              flush=True)
        return ply_path
