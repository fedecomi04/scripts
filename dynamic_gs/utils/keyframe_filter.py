"""Stateful ORB-SLAM-style keyframe filter.

Used at the dynamic-phase boundary to drop near-duplicate frames before
the per-frame ``_prepare_dynamic_frame`` pipeline runs (change mask +
50 optimization steps). Frame ``i`` is rejected iff some already-
accepted keyframe ``j`` is within both ``τ_t`` in translation AND
``τ_r`` in rotation.

Two modes share the same state:

* ``accept(c2w_3x4)`` — incremental, called per arriving frame. This
  is the API for live-data ingestion: as each camera frame streams in,
  ask the filter whether it should trigger optimization, and append
  it to the kept list on True.
* ``bulk_filter(c2w_Nx3x4)`` — convenience for recorded datasets that
  loops ``accept`` over all frames. Returns the accepted dataset
  indices in order.

The math matches the static keyframe filter in
``DynamicGSDataManager._filter_static_keyframes``: translation distance
is Euclidean on the c2w translation column, rotation distance is the
SO(3) geodesic ``arccos(clip(0.5·(trace(R_iᵀ R_j) − 1), −1, 1))``.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch


class DynamicKeyframeFilter:
    """Greedy keyframe filter with stateful accept/reject decisions."""

    def __init__(self, translation_thresh_m: float, rotation_thresh_deg: float):
        self.translation_thresh_m = float(translation_thresh_m)
        self.rotation_thresh_rad = float(np.deg2rad(rotation_thresh_deg))
        self._kept_R: List[np.ndarray] = []
        self._kept_t: List[np.ndarray] = []

    @property
    def num_kept(self) -> int:
        return len(self._kept_R)

    def reset(self) -> None:
        self._kept_R.clear()
        self._kept_t.clear()

    def accept(self, c2w_3x4) -> bool:
        """Return True if this c2w should trigger optimization.

        The first call is always accepted (bootstrap). On accept, the
        keyframe's (R, t) is appended to the internal state so future
        candidates are tested against it. ``c2w_3x4`` may be a torch
        Tensor (CPU or GPU) or a numpy array; we normalize to float64
        numpy on the host.
        """
        if isinstance(c2w_3x4, torch.Tensor):
            c2w = c2w_3x4.detach().cpu().numpy().astype(np.float64)
        else:
            c2w = np.asarray(c2w_3x4, dtype=np.float64)
        if c2w.shape != (3, 4):
            raise ValueError(f"expected c2w of shape (3, 4), got {c2w.shape}")
        R_i = c2w[:, :3]
        t_i = c2w[:, 3]

        if not self._kept_R:
            self._kept_R.append(R_i)
            self._kept_t.append(t_i)
            return True

        K_R = np.stack(self._kept_R, axis=0)
        K_t = np.stack(self._kept_t, axis=0)
        dt = np.linalg.norm(t_i - K_t, axis=1)
        traces = np.einsum("ab,kab->k", R_i, K_R)
        cos_theta = np.clip(0.5 * (traces - 1.0), -1.0, 1.0)
        dr = np.arccos(cos_theta)
        near = (dt <= self.translation_thresh_m) & (dr <= self.rotation_thresh_rad)
        if near.any():
            return False
        self._kept_R.append(R_i)
        self._kept_t.append(t_i)
        return True

    def bulk_filter(self, c2w_Nx3x4) -> List[int]:
        """Iterate ``accept`` over a stack of poses; return kept indices.

        Use this for recorded datasets where all camera poses are known
        upfront. State accumulates across calls — call ``reset()`` first
        if you want a fresh pass.
        """
        if isinstance(c2w_Nx3x4, torch.Tensor):
            c2w = c2w_Nx3x4.detach().cpu().numpy().astype(np.float64)
        else:
            c2w = np.asarray(c2w_Nx3x4, dtype=np.float64)
        if c2w.ndim != 3 or c2w.shape[1:] != (3, 4):
            raise ValueError(f"expected (N, 3, 4) c2w stack, got {c2w.shape}")
        kept: List[int] = []
        for i in range(c2w.shape[0]):
            if self.accept(c2w[i]):
                kept.append(i)
        return kept
