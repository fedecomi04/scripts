"""dynamic_feedforward.py — single-in-flight FF decode+insert worker (bg thread).

THE architectural fixes live here:
  P0 (highest priority): the bg thread reads camera/rgb/depth/masks ONLY from the
      immutable FeedforwardDispatch frozen at dispatch time — NEVER any live
      pipeline latest-frame attribute. So the pose the CDN scores against == the pose
      the inserts are placed against, BY CONSTRUCTION (kills the staleness re-insertion
      loop = "insert much less"). Enforced by a grep-check (test asserts the live-frame
      attribute token never appears in this file).
  Load-shed (the zed_final 3M blowup guard): enforce_ceiling refuses inserts past
      live_gaussian_ceiling (purge-then-trim); purge_ff_inserts drops low-opacity FF
      inserts (instance_id==insert_id) — NEVER the tracked object (protect_mask).

Lock discipline: _model_lock held only for the ms snapshot-read / cull / insert; the
~400 ms AnySplat decode runs lock-free. Heavy CDN + AnySplat decode are injected
callbacks (default factories wrap the proven dynamic_gs utils); the worker owns the
orchestration. (rewrite_spec/dynamic_feedforward.md, Invariants #4/#8/#9.)
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from .gaussian_set import GaussTensors, GaussianSet, activated_opacity


@dataclass(frozen=True)
class FeedforwardDispatch:
    """Immutable per-tick snapshot. The bg thread reads ONLY this (P0)."""
    seq: int
    camera: object               # dispatch-time scene Cameras (frozen)
    rgb_bgr: np.ndarray          # (H,W,3) uint8 BGR, dispatch-time
    depth_m: torch.Tensor        # (H,W) float32 metres, filtered, 0==invalid
    object_mask: torch.Tensor    # (H,W[,1]) tracked-object footprint at dispatch
    gripper_keep: torch.Tensor   # (H,W[,1]) robot-exclusion keep mask
    scene_intr: dict             # {fl_x,fl_y,cx,cy,w,h}
    d0_instance_id: int          # tracked id to PROTECT from cull/purge


class FeedforwardWorker:
    """Single-in-flight FF worker. due()/dispatch() on main; _run() on the bg daemon."""

    def __init__(self, gaussian_set: GaussianSet, lock: "threading.RLock",
                 ff_cfg, budget_cfg, *, cdn_fn: Callable, decode_fn: Callable,
                 on_insert: Optional[Callable] = None):
        self.g = gaussian_set
        self.lock = lock
        self.cfg = ff_cfg
        self.budget = budget_cfg
        self._cdn_fn = cdn_fn           # (dispatch) -> list[region masks] (locked render inside)
        self._decode_fn = decode_fn     # (dispatch, regions, snapshot) -> GaussTensors (lock-free)
        self._on_insert = on_insert
        self._slot = threading.Lock()
        self._inflight = False
        self._thread: Optional[threading.Thread] = None
        self._ff_calls = 0
        self.last_inserted = 0
        self.last_culled = 0

    # ---- main-thread gate ----
    def due(self, tick: int, now_s: float = 0.0) -> bool:
        """PURE: cadence_ticks boundary AND not in-flight. No side effects, no model touch."""
        if self._inflight:
            return False
        c = int(self.cfg.cadence_ticks)
        return c > 0 and tick > 0 and (tick % c == 0)

    def in_flight(self) -> bool:
        return self._inflight

    def dispatch(self, d: FeedforwardDispatch) -> bool:
        """Non-blocking. Acquire the single slot; start the bg thread. Release slot on
        Thread.start() failure (no permanent FF death — H7)."""
        if not self._slot.acquire(blocking=False):
            return False
        self._inflight = True
        try:
            self._thread = threading.Thread(target=self._run_guarded, args=(d,), daemon=True)
            self._thread.start()
        except Exception:
            self._inflight = False
            self._slot.release()
            return False
        return True

    def _run_guarded(self, d: FeedforwardDispatch) -> None:
        try:
            self._run(d)
        except Exception as e:                          # never let the bg thread crash silently
            print(f"[FF] _run error: {e}")
        finally:
            self._inflight = False
            self._ff_calls += 1
            try:
                self._slot.release()
            except RuntimeError:
                pass

    # ---- bg-thread body (reads ONLY `d`) ----
    def _run(self, d: FeedforwardDispatch) -> None:
        self.last_inserted = 0
        self.last_culled = 0
        regions = self._cdn_fn(d)                        # locked render inside cdn_fn; lock-free score
        if not regions:
            return
        snap = self.g.snapshot()                         # atomic detached read for ICP target + count
        tensors = self._decode_fn(d, regions, snap)      # ~400ms, LOCK-FREE
        if tensors is None or tensors.means.shape[0] == 0:
            return
        tensors, shed = self._enforce_ceiling(tensors, d.d0_instance_id)
        if tensors is None or tensors.means.shape[0] == 0:
            if shed:
                print(f"[FF] ceiling reached ({self.g.num_points}/{self.budget.live_gaussian_ceiling}); shed {shed}")
            return
        rng = self.g.insert(tensors, object_flag=1.0, instance_id=int(self.cfg.insert_id))   # UNDER lock
        self.last_inserted = int(rng.numel())
        n = int(self.budget.dynamic_purge_every_n_ff)
        if n > 0 and (self._ff_calls % n == 0):
            self.last_culled = self._purge_ff_inserts(d.d0_instance_id)
        if self._on_insert is not None:
            self._on_insert(rng)

    # ---- load-shed (NEW; deterministically testable) ----
    def _enforce_ceiling(self, tensors: GaussTensors, d0_instance_id: int):
        """Refuse inserts past live_gaussian_ceiling: purge first, then trim the batch to fit.
        Returns (tensors_or_None, shed_count)."""
        ceiling = int(self.budget.live_gaussian_ceiling)
        m = tensors.means.shape[0]
        if self.g.num_points + m <= ceiling:
            return tensors, 0
        self._purge_ff_inserts(d0_instance_id)
        room = max(0, ceiling - self.g.num_points)
        if room >= m:
            return tensors, 0
        if room == 0:
            return None, m
        keep = slice(0, room)                            # trim the batch to fit
        trimmed = GaussTensors(
            means=tensors.means[keep], features_dc=tensors.features_dc[keep],
            features_rest=tensors.features_rest[keep], scales=tensors.scales[keep],
            quats=tensors.quats[keep], opacities=tensors.opacities[keep])
        return trimmed, m - room

    def _purge_ff_inserts(self, d0_instance_id: int) -> int:
        """Cull low-opacity FF inserts (instance_id==insert_id) below the purge floor.
        NEVER drops the tracked object (protect_mask = object_instance_ids==d0_id)."""
        thr = float(self.budget.dynamic_purge_opacity_below)
        if thr <= 0:
            return 0
        snap = self.g.snapshot()
        ids = snap.buffers["object_instance_ids"][:, 0]
        ins = snap.buffers["inserted_flags"][:, 0] > 0.5
        elig = ins & (ids == int(self.cfg.insert_id))
        low = activated_opacity(snap.params["opacities"])[:, 0] < thr
        idx = torch.nonzero(elig & low, as_tuple=False).flatten()
        if idx.numel() == 0:
            return 0
        protect = ids == int(d0_instance_id)
        return self.g.cull(idx, protect_mask=protect)

    def close(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
