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

This file = everything that MUTATES the scene (cull/insert/purge) + threading.
dynamic_ff_backends.py = the pure compute that does NOT touch the scene (render+CDN, AnySplat decode).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from .gaussian_set import GaussTensors, GaussianSet, activated_opacity
from .timing import get_ledger


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
                 on_insert: Optional[Callable] = None, set_hidden_fn: Optional[Callable] = None):
        # ---- setup: store the scene, the config, the callbacks, and the single-run slot ----
        self.g = gaussian_set
        self.lock = lock
        self.cfg = ff_cfg
        self.budget = budget_cfg
        self._cdn_fn = cdn_fn           # (dispatch) -> list[region masks] (locked render inside)
        self._decode_fn = decode_fn     # (dispatch, regions, snapshot) -> GaussTensors (lock-free)
        self._on_insert = on_insert
        # Deferred cull: set_hidden_fn(idx_or_None) tells the renderer to VIRTUALLY hide those rows so
        # the re-CDN sees the scene as-if-culled while the real cull is held to commit atomically with
        # the insert. None -> deferral off (the cull commits on its own surgery before the decode).
        self._set_hidden_fn = set_hidden_fn
        self._slot = threading.Lock()
        self._inflight = False
        self._thread: Optional[threading.Thread] = None
        self._ff_calls = 0
        self.last_inserted = 0
        self.last_culled = 0
        self._t = get_ledger()          # always-on timing ledger (FF stages + gaussian-count gauge)

    # ---- checks when to run (called on the main thread, every tick) ----
    def due(self, tick: int, now_s: float = 0.0) -> bool:
        """PURE: cadence_ticks boundary AND not in-flight. No side effects, no model touch."""
        if self._inflight:
            return False
        c = int(self.cfg.cadence_ticks)
        return c > 0 and tick > 0 and (tick % c == 0)

    def in_flight(self) -> bool:
        """Public read-only accessor: is an FF decode currently running on the bg thread?"""
        return self._inflight

    # ---- threading: launch the work on the side thread, with an error guard ----
    def dispatch(self, d: FeedforwardDispatch) -> bool:
        """Non-blocking. Acquire the single slot; start the bg thread. Release the slot if
        Thread.start() raises, so a launch failure doesn't permanently wedge FF."""
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
        """bg-thread entry: run _run, swallow+log any crash so the thread never dies silently,
        and ALWAYS free the in-flight slot in `finally` so the next tick can fire FF."""
        try:
            self._run(d)
        except Exception as e:                          # never let the bg thread crash silently
            print(f"[FF] _run error: {e}")
        finally:
            # Release this cycle's CUDA transients (render / cull projection / decode / surgery) back
            # to the allocator EVERY cycle, so the AnySplat worker — which shares the GPU — isn't
            # starved of its ~50MB and OOMs. Runs regardless of how _run exited.
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            self._inflight = False
            self._ff_calls += 1
            try:
                self._slot.release()
            except RuntimeError:
                pass

    # ---- cleanup: delete scene gaussians floating in front of the real surface ----
    @staticmethod
    def _cam_scalars(cam):
        """Pull the 6 camera numbers (fx,fy,cx,cy,width,height) out as plain floats/ints.
        Needed because nerfstudio stores them as tensors (sometimes shape (1,) for a batched
        camera); s() handles both tensor and bare-number cases so the projection math gets clean
        scalars."""
        def s(x):
            return float(x.detach().cpu().reshape(-1)[0].item()) if torch.is_tensor(x) else float(x)
        return (s(cam.fx), s(cam.fy), s(cam.cx), s(cam.cy),
                int(s(cam.width)), int(s(cam.height)))

    def _compute_cull_in_front(self, d: FeedforwardDispatch, cdn_np: np.ndarray):
        """COMPUTE (no commit) the eligible gaussians (instance_id in {0, insert_id}, i.e. not the
        tracked object) projecting inside the CDN region that sit IN FRONT of the live sensor surface
        (artifacts/occluders). Direct projection — no render. Returns (cull_idx, protect_mask)."""
        snap = self.g.snapshot()
        means = snap.params["means"]
        dev = means.device
        ids = snap.buffers["object_instance_ids"][:, 0].to(dev)
        fx, fy, cx, cy, W_cam, H_cam = self._cam_scalars(d.camera)
        depth = d.depth_m
        depth = depth[..., 0] if depth.ndim == 3 else depth
        depth = depth.to(dev)
        H, W = int(depth.shape[0]), int(depth.shape[1])
        if (H, W) != (H_cam, W_cam):                      # depth grid != camera res -> scale K
            sx, sy = W / float(W_cam), H / float(H_cam)
            fx *= sx; cx *= sx; fy *= sy; cy *= sy
        c2w = d.camera.camera_to_worlds
        c2w = c2w[0] if c2w.ndim == 3 else c2w
        c2w = c2w.to(dev, dtype=means.dtype)
        R, t = c2w[:3, :3], c2w[:3, 3]
        means_cam = (means - t[None, :]) @ R
        depths_g = -means_cam[:, 2]
        in_front_cam = depths_g > 1e-6
        safe_d = torch.where(in_front_cam, depths_g, torch.ones_like(depths_g))
        u = fx * (means_cam[:, 0] / safe_d) + cx
        v = fy * (-means_cam[:, 1] / safe_d) + cy
        u_idx = u.round().long().clamp(0, W - 1)
        v_idx = v.round().long().clamp(0, H - 1)
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & in_front_cam
        comp = torch.as_tensor(cdn_np, device=dev).bool()
        if comp.shape[-2:] != (H, W):
            import torch.nn.functional as _F
            comp = _F.interpolate(comp.float()[None, None], size=(H, W), mode="nearest")[0, 0].bool()
        sensor_at = depth[v_idx, u_idx]
        in_region = comp[v_idx, u_idx] & in_bounds
        in_front = depths_g < (sensor_at - float(self.cfg.cull_in_front_depth_tol_m))
        eligible = (ids == 0) | (ids == int(self.cfg.insert_id))
        idx = torch.nonzero(in_region & (sensor_at > 0) & in_front & eligible, as_tuple=False).flatten()
        return idx, (ids == int(d.d0_instance_id))

    def _compute_cull_replaced(self, d: FeedforwardDispatch, cdn_np: np.ndarray):
        """COMPUTE (no commit) the OLD geometry the fresh insert REPLACES: eligible (non-tracked)
        gaussians projecting into the changed region that sit in a thin slab just behind the live surface.
        Committed ATOMICALLY with the insert so the deletion is never visible (new replaces old in one
        surgery → caps cumulative growth). Returns an empty tensor when cull_replaced_enabled is False.
        Same projection math as _compute_cull_in_front."""
        if not bool(self.cfg.cull_replaced_enabled):
            return torch.empty(0, dtype=torch.long, device=self.g.device)
        tol = float(self.cfg.cull_replaced_depth_tol_m)
        snap = self.g.snapshot()
        means = snap.params["means"]
        dev = means.device
        ids = snap.buffers["object_instance_ids"][:, 0].to(dev)
        fx, fy, cx, cy, W_cam, H_cam = self._cam_scalars(d.camera)
        depth = d.depth_m
        depth = depth[..., 0] if depth.ndim == 3 else depth
        depth = depth.to(dev)
        H, W = int(depth.shape[0]), int(depth.shape[1])
        if (H, W) != (H_cam, W_cam):
            sx, sy = W / float(W_cam), H / float(H_cam)
            fx *= sx; cx *= sx; fy *= sy; cy *= sy
        c2w = d.camera.camera_to_worlds
        c2w = c2w[0] if c2w.ndim == 3 else c2w
        c2w = c2w.to(dev, dtype=means.dtype)
        R, t = c2w[:3, :3], c2w[:3, 3]
        means_cam = (means - t[None, :]) @ R
        depths_g = -means_cam[:, 2]
        in_front_cam = depths_g > 1e-6
        safe_d = torch.where(in_front_cam, depths_g, torch.ones_like(depths_g))
        u = fx * (means_cam[:, 0] / safe_d) + cx
        v = fy * (-means_cam[:, 1] / safe_d) + cy
        u_idx = u.round().long().clamp(0, W - 1)
        v_idx = v.round().long().clamp(0, H - 1)
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & in_front_cam
        comp = torch.as_tensor(cdn_np, device=dev).bool()
        if comp.shape[-2:] != (H, W):
            import torch.nn.functional as _F
            comp = _F.interpolate(comp.float()[None, None], size=(H, W), mode="nearest")[0, 0].bool()
        sensor_at = depth[v_idx, u_idx]
        in_region = comp[v_idx, u_idx] & in_bounds
        # The old surface the insert overwrites: ONLY gaussians in a thin slab just BEHIND the live
        # surface — between the real depth and `tol` (0.5mm) behind it: sensor <= depth_g < sensor + tol.
        # A bounded slab (not the whole column behind) keeps deep walls/background intact; in front of the
        # surface is the in-front cull's job.
        eps = 1e-4                                            # 0.1mm float-compare grace at the AT-surface boundary
        in_slab = (depths_g >= (sensor_at - eps)) & (depths_g < (sensor_at + tol))
        eligible = (ids == 0) | (ids == int(self.cfg.insert_id))
        return torch.nonzero(in_region & (sensor_at > 0) & in_slab & eligible, as_tuple=False).flatten()

    def cull_in_front(self, d: FeedforwardDispatch, cdn_np: np.ndarray) -> int:
        """Compute + COMMIT the in-front cull immediately (non-deferred path). Returns n deleted."""
        idx, protect = self._compute_cull_in_front(d, cdn_np)
        if idx.numel() == 0:
            return 0
        return self.g.cull(idx, protect_mask=protect)

    # ---- the actual work, run on the side thread (reads ONLY the frozen `d`) ----
    def _run(self, d: FeedforwardDispatch) -> None:
        """The full FF cycle: render the scene + compute the change region (CDN); find the gaussians
        floating in front of that changed region and virtually hide them; re-render restricted to that
        region + re-CDN (AND-gated to the first region); run the AnySplat decode; then commit the cull
        (in-front + the replaced-surface slab + any due purge) and the insert TOGETHER in one atomic
        surgery so the viewer never sees a hole with nothing filling it. When deferral is not wired,
        the in-front cull commits on its own surgery before the decode instead."""
        self.last_inserted = 0
        self.last_culled = 0
        self._t.cycle(self._ff_calls)                    # tag every stage below with THIS FF cycle id
        # CDN render + clean (steps 1-2 are timed inside cdn_fn under this cycle).
        regions = self._cdn_fn(d)                        # render + score + CLEAN (object-footprint subtract)
        if not regions:
            self._t.event("ff_skipped", reason="no_cdn")
            return
        deferred = self.cfg.cull_before_decode and self._set_hidden_fn is not None
        first_region = regions[0]                        # the FIRST CDN region — the re-CDN is restricted
                                                          # to it (render only its gaussians + AND-gate)
        cull_idx = None
        protect = None
        if self.cfg.cull_before_decode:
            with self._t.stage("cull_infront.compute"):
                cull_idx, protect = self._compute_cull_in_front(d, first_region)
            if cull_idx.numel() == 0:
                cull_idx = None
            elif deferred:
                # hide the to-be-culled rows for the re-CDN ONLY (held under the lock so the render
                # honors it), then unhide — the real delete waits for the atomic commit below.
                with self.lock:
                    with self._t.stage("cull_infront.hide"):
                        self._set_hidden_fn(cull_idx)
                    regions = self._cdn_fn(d, restrict_to=first_region)   # re-CDN: as-if-culled, region-restricted
                    self._set_hidden_fn(None)
                if not regions:
                    self._t.event("ff_skipped", reason="no_recdn")
                    return
            else:                                        # no deferral: commit the cull on its own surgery
                with self._t.stage("surgery.cull_insert"):
                    self.last_culled = self.g.cull(cull_idx, protect_mask=protect)
                cull_idx = None
                regions = self._cdn_fn(d, restrict_to=first_region)
                if not regions:
                    self._t.event("ff_skipped", reason="no_recdn")
                    return
        snap = self.g.snapshot()                         # atomic detached read for ICP target + count
        # AnySplat decode + density shaping + clamp (steps 7-9 timed inside decode_fn).
        tensors = self._decode_fn(d, regions, snap)      # ~400ms, LOCK-FREE
        if tensors is None or tensors.means.shape[0] == 0:
            self._t.event("ff_skipped", reason="empty_decode")
            return
        with self._t.stage("enforce_ceiling"):
            tensors, shed = self._enforce_ceiling(tensors, d.d0_instance_id)
        if tensors is None or tensors.means.shape[0] == 0:
            if shed:
                print(f"[FF] ceiling reached ({self.g.num_points}/{self.budget.live_gaussian_ceiling}); shed {shed}")
            self._t.event("ff_skipped", reason="ceiling")
            return
        n = int(self.budget.dynamic_purge_every_n_ff)
        purge_due = n > 0 and (self._ff_calls % n == 0)
        if deferred:
            # Fold THREE deletes into the SAME atomic cull_and_insert (one surgery + one rebind):
            #  1. the in-front cull (artifacts/occluders, hidden during the re-CDN above),
            #  2. the REPLACED cull — old geometry at-or-behind the live surface the insert overwrites,
            #     computed NOW (after decode confirmed we're inserting) so it never affects the re-CDN
            #     and the deletion is invisible (new replaces old in the same surgery), and
            #  3. (when due) the periodic low-opacity purge.
            idx = cull_idx if cull_idx is not None else torch.empty(0, dtype=torch.long, device=self.g.device)
            with self._t.stage("cull_replaced.compute"):
                replaced = self._compute_cull_replaced(d, regions[0])
            if replaced.numel():
                idx = torch.cat([idx, replaced.to(idx.device)]).unique()
            if purge_due:
                idx = torch.cat([idx, self._purge_indices().to(idx.device)]).unique()
            # Row layout is unchanged across all the snapshots above: the ONLY concurrent mutation
            # during the lock-free decode is the tracker's write_object_pose, which writes VALUES in
            # place (no cull/insert/reorder). cull/insert come only from this single-in-flight FF thread.
            # So the indices stay valid; still, build protect fresh here so it matches the commit state.
            if idx.numel():
                protect = self.g.snapshot().buffers["object_instance_ids"][:, 0] == int(d.d0_instance_id)
            with self._t.stage("surgery.cull_insert"):
                n_culled, rng = self.g.cull_and_insert(
                    idx, tensors, object_flag=1.0, instance_id=int(self.cfg.insert_id),
                    protect_mask=protect)
            self.last_culled = n_culled
        else:
            rng = self.g.insert(tensors, object_flag=1.0, instance_id=int(self.cfg.insert_id))   # UNDER lock
            if purge_due:
                self.last_culled += self._purge_ff_inserts(d.d0_instance_id)   # separate surgery (non-deferred path)
        self.last_inserted = int(rng.numel())
        self._t.event("ff_inserted", n=self.last_inserted, culled=self.last_culled)
        self._t.gauge("gaussian_count", self.g.num_points)   # bounded-growth watchdog
        if self._on_insert is not None:
            self._on_insert(rng)

    # ---- keep the scene from growing forever (cap the total gaussian count) ----
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

    def _purge_indices(self) -> torch.Tensor:
        """COMPUTE (no commit) the low-opacity FF inserts (inserted_flags & instance_id==insert_id,
        below the purge floor) to drop. Returns the index tensor (empty if disabled/none). The
        eligibility filter already excludes the tracked object (its id != insert_id), so these
        indices never contain a d0 row — the caller still passes protect_mask as cheap insurance."""
        thr = float(self.budget.dynamic_purge_opacity_below)
        if thr <= 0:
            return torch.empty(0, dtype=torch.long, device=self.g.device)
        snap = self.g.snapshot()
        ids = snap.buffers["object_instance_ids"][:, 0]
        ins = snap.buffers["inserted_flags"][:, 0] > 0.5
        elig = ins & (ids == int(self.cfg.insert_id))
        low = activated_opacity(snap.params["opacities"])[:, 0] < thr
        return torch.nonzero(elig & low, as_tuple=False).flatten()

    def _purge_ff_inserts(self, d0_instance_id: int) -> int:
        """Standalone purge commit (the periodic maintenance sweep). Computes + culls in one surgery.
        NEVER drops the tracked object (protect_mask = object_instance_ids==d0_id)."""
        idx = self._purge_indices()
        if idx.numel() == 0:
            return 0
        protect = self.g.snapshot().buffers["object_instance_ids"][:, 0] == int(d0_instance_id)
        return self.g.cull(idx, protect_mask=protect)

    # ---- shutdown: wait for the side thread to finish ----
    def close(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
