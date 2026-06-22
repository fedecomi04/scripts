"""static_capture.py — record live sweep frames into a recorded static_scene/ dataset.

The live static schedule produces an on-disk static_scene/ that the proven old static-gs
then consumes (train + Phase-0a/0b). This writer byte-matches the old recorder's format
(BGR png / uint16-mm tiff / uint8 mask / transforms.json with relative paths + intrinsics)
so the old dataparser reads it unchanged. A greedy ORB-SLAM keyframe gate (2 cm / 20°)
drops near-duplicate views, exactly like the old capture, so training isn't fed redundant
frames.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .frame import Frame, Intrinsics


class StaticRecorder:
    """Accumulate keyframe-deduped sweep frames, then write static_scene/{rgb,depth,masks} +
    transforms.json. One recorder per live static run."""

    def __init__(self, data_dir, intr: Intrinsics, *,
                 trans_thresh_m: float = 0.02, rot_thresh_deg: float = 20.0,
                 interval_s: float = 0.0):
        self._root = Path(data_dir) / "static_scene"
        for sub in ("rgb", "depth", "masks"):
            (self._root / sub).mkdir(parents=True, exist_ok=True)
        self._intr = intr
        self._tt = float(trans_thresh_m)
        self._rt = float(np.deg2rad(rot_thresh_deg))
        # interval_s > 0 -> time-based capture (every N s), for a FIXED camera that doesn't sweep
        # (real-HW elbow mount). Defaults to 1.0 under DGS_REAL_HW_CAMERA, else 0 (motion dedup).
        if interval_s <= 0.0:
            _default = 1.0 if os.environ.get("DGS_REAL_HW_CAMERA", "0") != "0" else 0.0
            try:
                interval_s = float(os.environ.get("DGS_STATIC_INTERVAL_S", _default))
            except ValueError:
                interval_s = _default
        self._interval_s = float(interval_s)
        self._last_accept_t: float | None = None
        self._kept_R: List[np.ndarray] = []
        self._kept_t: List[np.ndarray] = []
        self._frames_meta: List[dict] = []
        self._n = 0

    def _is_keyframe(self, c2w: np.ndarray) -> bool:
        """Accept gate. Time mode (interval_s>0): accept frame 0, then one every interval_s seconds
        regardless of camera motion (fixed camera). Motion mode (default): ORB-SLAM gate — accept
        frame 0; later frames only if no kept frame is within BOTH the translation AND rotation
        thresholds (i.e. far enough in T or R from all kept)."""
        R, t = c2w[:3, :3], c2w[:3, 3]
        if self._interval_s > 0.0:
            now = time.time()
            if self._last_accept_t is not None and (now - self._last_accept_t) < self._interval_s:
                return False
            self._last_accept_t = now
            return True
        if not self._kept_R:
            return True
        for Rk, tk in zip(self._kept_R, self._kept_t):
            dt = float(np.linalg.norm(t - tk))
            cos = np.clip(0.5 * (float(np.trace(R.T @ Rk)) - 1.0), -1.0, 1.0)
            dr = float(np.arccos(cos))
            if dt <= self._tt and dr <= self._rt:
                return False
        return True

    def add(self, frame: Frame) -> bool:
        """Keyframe-gate + write one frame (rgb/depth/mask) and stage its transforms entry.
        Returns True if kept. Robot is excluded by the saved mask, same as the old recorder.
        Skips frames with missing/empty rgb or depth (a torn SHM read can hand back None/0-size
        arrays); a frame is only STAGED in transforms.json after all 3 imwrites are verified on
        disk, so transforms.json can never list a file that isn't there (the seed-build crash)."""
        if (frame.rgb_bgr is None or frame.depth_m is None or frame.mask_keep is None
                or getattr(frame.rgb_bgr, "size", 0) == 0 or getattr(frame.depth_m, "size", 0) == 0):
            return False
        c2w = np.asarray(frame.c2w_4x4, dtype=np.float64)
        if not self._is_keyframe(c2w):
            return False
        # Defensive: recreate the dirs each add (cheap, exist_ok) so a frame is never silently
        # dropped because the dir vanished (the 2026-06-21 '0 files / 22 transforms entries' bug).
        for sub in ("rgb", "depth", "masks"):
            (self._root / sub).mkdir(parents=True, exist_ok=True)
        stem = f"arm_{self._n:05d}"
        rgb_p = self._root / "rgb" / f"{stem}.png"
        depth_p = self._root / "depth" / f"{stem}.tiff"
        mask_p = self._root / "masks" / f"{stem}.png"
        ok_rgb = cv2.imwrite(str(rgb_p), frame.rgb_bgr)
        # NaN/inf -> 0 (the "invalid/no-return" sentinel) BEFORE the uint16 cast: live ZED depth has
        # NaN holes, and astype(uint16) on NaN is undefined (RuntimeWarning + garbage). nan_to_num first.
        depth_m = np.nan_to_num(frame.depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        depth_mm = np.clip(depth_m * 1000.0, 0.0, 65535.0).astype(np.uint16)
        ok_depth = cv2.imwrite(str(depth_p), depth_mm)
        keep = frame.mask_keep
        keep = keep[..., 0] if keep.ndim == 3 else keep
        ok_mask = cv2.imwrite(str(mask_p), (keep > 0).astype(np.uint8) * 255)
        # Stage the transforms entry ONLY if all three files are verified on disk. If any imwrite
        # failed (returns False, or the file is absent), drop the frame entirely so transforms.json
        # never references a missing file -> the deferred TSDF seed can't crash on a None depth.
        if not (ok_rgb and ok_depth and ok_mask and rgb_p.exists() and depth_p.exists() and mask_p.exists()):
            print(f"[static-recorder] WARNING dropped frame {stem}: imwrite failed "
                  f"(rgb={ok_rgb} depth={ok_depth} mask={ok_mask}) — NOT staged", flush=True)
            for p in (rgb_p, depth_p, mask_p):
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
            return False
        self._frames_meta.append({
            "file_path": f"./rgb/{stem}.png",
            "depth_file_path": f"./depth/{stem}.tiff",
            "mask_path": f"./masks/{stem}.png",
            "transform_matrix": c2w.tolist(),
            "stamp_wall": float(frame.stamp_sec),
        })
        self._kept_R.append(c2w[:3, :3]); self._kept_t.append(c2w[:3, 3])
        self._n += 1
        return True

    @property
    def num_kept(self) -> int:
        return self._n

    def finalize(self) -> Path:
        """Atomically write transforms.json (intrinsics + the staged frame list). Returns its path."""
        meta = {
            "w": int(self._intr.width), "h": int(self._intr.height),
            "fl_x": float(self._intr.fx), "fl_y": float(self._intr.fy),
            "cx": float(self._intr.cx), "cy": float(self._intr.cy),
            "frames": self._frames_meta,
        }
        path = self._root / "transforms.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=2) + "\n")
        os.replace(tmp, path)
        return path
