"""Live ROS-fed sibling of :class:`RecordedDynamicGSPipeline`.

Frame source: a ``LiveShmSubscriber`` (in :mod:`dynamic_gs.utils.live_shm_reader`)
that spawns the publisher subprocess (``live_ros_publisher.py``) in the
``dynamic_gs_ros`` conda env and exposes ``peek_latest()`` for lock-free
per-tick frame reads from shared memory.

D0 object selection: 3D-centroid-to-camera distance — pick the closest
prefused instance. Works in live because the teleoperated object is the
nearest thing in front of the camera; no pixel anchor required.

Per-tick flow (mirrors :class:`RecordedDynamicGSPipeline._tracker_tick`):
  1. Peek the latest SHM frame; skip if none / already processed (dedup
     on ``stamp_sec``).
  2. Build a :class:`Cameras` slice from the live intrinsics + c2w, and a
     batch dict (image / depth_image / mask) from the LiveFrame buffers.
  3. D0 bootstrap on the first frame; XFeat motion estimate on N>=1.
  4. Render + change mask (CDN).
  5. Publish :attr:`_latest_tracker_frame`; refresh viser-direct.
  6. Fire :meth:`_on_tracker_frame` -> Mode B feedforward (same cadence
     gate as recorded).

Termination: type ``stop`` on stdin (caught by a daemon watcher) OR
Ctrl+C / max_num_iterations. SHM publisher subprocess is shut down via
``LiveShmSubscriber.close()`` registered atexit.

AnySplat FF context image: the AnySplat worker runs in a different conda
env so it reads its input from disk. We dump the latest live RGB to
``/dev/shm/dgs_live_ff_frame_<pid>.png`` per FF call -- tmpfs (RAM) so
the write is ~1 ms and the file lives only until the next dump.
"""

from __future__ import annotations

import atexit
import os
import time
import signal
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type

import numpy as np
import torch

from nerfstudio.utils.rich_utils import CONSOLE

from .dynamic_gs_pipeline_base import (
    DynamicGSPipelineBase,
    DynamicGSPipelineBaseConfig,
)


@dataclass
class LiveDynamicGSPipelineConfig(DynamicGSPipelineBaseConfig):
    """Config for :class:`LiveDynamicGSPipeline`. Adds the SHM publisher
    knobs; everything else is inherited from the base."""

    _target: Type = field(default_factory=lambda: LiveDynamicGSPipeline)

    live_shm_name: str = "/dgs_live_shm"
    """Name of the POSIX shared-memory region the publisher writes into.
    Must match the publisher's --shm-name (default matches)."""

    live_keyframe_translation_m: float = 0.02
    """Publisher-side keyframe filter: emit only when the camera has
    moved at least this far (meters) since the last published frame."""

    live_keyframe_rotation_deg: float = 20.0
    """Publisher-side keyframe filter: emit only when the camera has
    rotated at least this much (degrees) since the last published frame."""

    live_wipe_root: bool = False
    """If True, the publisher wipes its live-root directory at startup
    -- DESTROYS any recorded frames + post_fusion_state.pt on the way in.
    Default False so re-runs against a previously-captured scene are
    non-destructive. Set True only when you actually want a fresh capture
    (the bootstrap_live.sh script handles fresh captures via
    live_session.py directly, which has its own wipe logic)."""


class LiveDynamicGSPipeline(DynamicGSPipelineBase):
    """Live-ROS dynamic pipeline. See module docstring."""

    config: LiveDynamicGSPipelineConfig

    def __init__(
        self,
        config: LiveDynamicGSPipelineConfig,
        device: str,
        test_mode="val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler=None,
    ):
        # Initialize live-specific state BEFORE super().__init__() because
        # the base's __init__ may invoke things that need them (e.g. the
        # AnySplat persistent worker spawn, which checks the config).
        self._next_live_frame_counter: int = 0
        self._last_processed_stamp_sec: Optional[float] = None
        self._latest_live_rgb_bgr: Optional[np.ndarray] = None
        self._live_stop_requested: bool = False
        self._shm_sub = None  # set below
        self._d0_completed: bool = False
        self._d0_defer_attempts: int = 0

        super().__init__(
            config=config, device=device, test_mode=test_mode,
            world_size=world_size, local_rank=local_rank,
            grad_scaler=grad_scaler,
        )

        # Switch datamanager to dynamic phase so any state checks pass
        # (the live pipeline doesn't actually pull from the datamanager
        # at runtime, but the model + datamanager initialization paths
        # do read `phase`).
        if hasattr(self.datamanager, "set_phase"):
            self.datamanager.set_phase("dynamic")
        # ALSO switch the model into dynamic phase so _apply_phase_trainability
        # keeps means.requires_grad=True (otherwise the static-phase branch
        # would flip it to False and break register_hook on later FF inserts).
        if hasattr(self.model, "set_phase"):
            self.model.set_phase("dynamic")
        # Empty accepted-frames list: the FF anysplat path doesn't need
        # one in live mode (we override _resolve_anysplat_context_image_paths).
        self._accepted_dynamic_frames = []

        # Spawn the ROS publisher subprocess + SHM subscriber. Blocks
        # until the publisher reports {"event": "ready"} (intrinsics +
        # first /camera_info received).
        from .utils.live_shm_reader import LiveShmSubscriber
        CONSOLE.log("[dynamic-gs-live] spawning ROS publisher + SHM subscriber...")
        self._shm_sub = LiveShmSubscriber(
            shm_name=config.live_shm_name,
            keyframe_translation_m=config.live_keyframe_translation_m,
            keyframe_rotation_deg=config.live_keyframe_rotation_deg,
            wipe_live_root=config.live_wipe_root,
        )
        intr = self._shm_sub.intrinsics
        CONSOLE.log(
            f"[dynamic-gs-live] publisher ready: {intr.width}x{intr.height}, "
            f"K=(fx={intr.fx:.1f}, fy={intr.fy:.1f}, cx={intr.cx:.1f}, cy={intr.cy:.1f})"
        )

        # Wait for the first (rgb, depth, pose) tuple to ensure the
        # publisher really is producing frames before training starts.
        first = self._shm_sub.wait_for_first_frame(timeout_s=30.0)
        CONSOLE.log(f"[dynamic-gs-live] first frame received at stamp_sec={first.stamp_sec:.3f}")

        # Stdin "stop" watcher: lets the user end the session cleanly.
        self._start_stdin_stop_watcher()

        # Cleanup the SHM + publisher BEFORE the AnySplat worker is closed
        # (atexit is LIFO; the base's worker cleanup is already registered).
        atexit.register(self._cleanup_live_subscriber)
        atexit.register(self._cleanup_live_ff_dump)

        # Explicit signal handlers: atexit doesn't fire reliably on SIGTERM,
        # and on SIGINT it only fires if the main thread reaches normal
        # interpreter shutdown — which it doesn't if Nerfstudio's trainer
        # is mid-iteration in a background thread. Install handlers that
        # explicitly drop the publisher subprocess, so the NEXT run doesn't
        # find an orphan holding /camera_info subscriptions.
        def _on_signal(signum, _frame) -> None:
            CONSOLE.log(f"[dynamic-gs-live] signal {signum} received — closing publisher")
            try:
                self._cleanup_live_subscriber()
            except Exception:
                pass
            try:
                self._cleanup_live_ff_dump()
            except Exception:
                pass
            # Re-raise to let the normal shutdown finish (KeyboardInterrupt
            # for SIGINT, SystemExit for SIGTERM).
            if signum == signal.SIGINT:
                raise KeyboardInterrupt()
            sys.exit(128 + signum)
        try:
            signal.signal(signal.SIGINT,  _on_signal)
            signal.signal(signal.SIGTERM, _on_signal)
        except ValueError:
            # signal.signal only works from the main thread. If we're not
            # on it, atexit + Nerfstudio's own handler are the fallback.
            pass

    # ====================================================================
    # Cleanup hooks
    # ====================================================================

    def _cleanup_live_subscriber(self) -> None:
        sub = getattr(self, "_shm_sub", None)
        if sub is None:
            return
        try:
            sub.close()
        except Exception:
            pass
        self._shm_sub = None

    def _cleanup_live_ff_dump(self) -> None:
        """Remove the per-process FF input PNG from /dev/shm."""
        try:
            p = Path(f"/dev/shm/dgs_live_ff_frame_{os.getpid()}.png")
            if p.exists():
                p.unlink()
        except Exception:
            pass

    def _start_stdin_stop_watcher(self) -> None:
        """Daemon thread watching stdin for control input. Non-blocking; if
        stdin is not a TTY (CI / nohup) it simply never fires.

        * ``stop`` / ``quit`` / ``exit`` — end the live session (returns).
        * bare Enter (empty line) — when ``interactive_object_selection`` is on,
          reopen the object picker so the operator can switch objects. The
          watcher keeps running so further Enters / ``stop`` are still caught."""
        def _watch() -> None:
            try:
                for line in sys.stdin:
                    s = line.strip().lower()
                    if s in ("stop", "quit", "exit"):
                        CONSOLE.log("[dynamic-gs-live] 'stop' received on stdin -- ending session")
                        self._live_stop_requested = True
                        return
                    if s == "" and self.config.interactive_object_selection:
                        CONSOLE.log("[dynamic-gs-live] bare-Enter -> reopening object picker")
                        self._reselect_requested = True
            except Exception:
                pass  # stdin closed; that's fine
        threading.Thread(target=_watch, daemon=True, name="dgs-live-stop-watcher").start()

    # ====================================================================
    # Abstract hook implementations
    # ====================================================================

    def _tracker_tick(self, step: int) -> None:
        """Live tick: peek latest SHM frame, dedup, run XFeat, publish."""
        if self._live_stop_requested:
            return
        if self._shm_sub is None:
            return

        latest = self._shm_sub.peek_latest()
        if latest is None:
            return  # publisher hasn't emitted a new frame yet
        if self._last_processed_stamp_sec is not None:
            # Sim-clock reset detection: if the new stamp is more than 1 s
            # behind the last, Gazebo's /clock was reset and the dedup gate
            # would otherwise drop every subsequent frame forever. Re-arm.
            if latest.stamp_sec < self._last_processed_stamp_sec - 1.0:
                CONSOLE.log(
                    f"[dynamic-gs-live] sim clock reset detected "
                    f"(last={self._last_processed_stamp_sec:.3f}s -> "
                    f"new={latest.stamp_sec:.3f}s); re-arming stamp gate"
                )
                self._last_processed_stamp_sec = None
            elif latest.stamp_sec <= self._last_processed_stamp_sec:
                return  # we already processed this (or older) frame
        self._last_processed_stamp_sec = float(latest.stamp_sec)

        # Build Nerfstudio Cameras + a Splatfacto-shaped batch dict.
        from .utils.live_shm_reader import cameras_from_live_frame
        device = torch.device(self.device) if isinstance(self.device, str) else self.device
        camera = cameras_from_live_frame(latest, self._shm_sub.intrinsics, device=device)
        batch = self._batch_from_live_frame(latest, device=device)
        # New frame/camera → invalidate the per-tick object-mask cache so this
        # tick's first consumer renders it once and the rest reuse it.
        self._invalidate_object_mask_cache()

        frame_idx = self._next_live_frame_counter
        self._next_live_frame_counter += 1
        is_first = not self._d0_completed

        # Interactive object picker (live): when active, this BLOCKS the tick
        # until the operator selects in viser (or the timeout fires), then
        # reseeds + marks D0 complete via the live _reset_d0_guard. SHM frames
        # are dropped-oldest during the block (acceptable: nothing is being
        # tracked yet / the prior object is frozen while choosing the next).
        sel_status = "none"
        if self.config.interactive_object_selection:
            sel_status = self._tick_interactive_selection(camera, batch, is_first)

        if sel_status == "seeded":
            # Picker already reseeded + marked D0 complete; nothing more to do
            # at the D0 stage. Fall through to CDN / publish.
            pass
        elif is_first:
            self._bootstrap_d0(camera, batch)
            if not self._d0_completed:
                # D0 deferred (no object visible enough this frame). Don't
                # advance _tracker_tick_count so the next frame re-enters
                # the bootstrap branch — and don't run CDN / FF either,
                # there's nothing to compare against yet.
                return
        else:
            self._apply_motion_estimator(camera, batch)

        t_cdn = time.time()
        cdn = self._compute_tick_cdn(camera, batch)
        if os.environ.get("DGS_DIAG_SYNC") == "1" and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._timing["DN.2_cdn_render"].append(time.time() - t_cdn)

        # Cache the latest BGR frame for FF AnySplat dump (see
        # _resolve_anysplat_context_image_paths).
        self._latest_live_rgb_bgr = latest.rgb_bgr

        self._latest_tracker_frame = {
            "frame_idx": frame_idx,
            "camera": camera,
            "cdn": cdn,
            "batch": batch,
            "stamp_sec": float(latest.stamp_sec),
        }
        self._global_frame_counter += 1
        self._tracker_tick_count += 1

        self._build_viser_direct_handles(camera)
        self._push_viser_direct_transforms()
        self._push_viser_camera_feed(camera, batch)
        self._force_viewer_rerender()

        self._on_tracker_frame(camera, batch, cdn, is_first)

    _D0_MIN_VISIBLE_GAUSSIANS = 200
    """At D0, the picked instance must project at least this many of its
    Gaussian centres into the camera frustum before we accept the seed.
    Below this the camera isn't pointing at the object well enough — defer
    D0 to a later frame instead of seeding the XFeat anchor on background."""

    def _pick_d0_object(
        self,
        camera,
        prefused_instance_ids: torch.Tensor,
    ) -> int:
        """Live D0: of the prefused instances with ``visible_gauss >=
        _D0_MIN_VISIBLE_GAUSSIANS`` in the current camera frustum, pick
        the one whose centroid is closest to the camera. Visibility is a
        gate (the object must actually be on screen, not behind the camera
        or out of frame); distance is the ranking. Returns 0 if no instance
        passes the visibility gate — D0 is then deferred."""
        ids = prefused_instance_ids.squeeze(-1) if prefused_instance_ids.ndim > 1 else prefused_instance_ids
        unique_ids = torch.unique(ids[ids > 0]).tolist()
        if not unique_ids:
            return 0

        # Build the projection once for ALL means; per-instance counts are
        # then a cheap boolean count on the already-projected tensor.
        c2w = camera.camera_to_worlds
        if c2w.ndim == 3:
            c2w = c2w[0]
        device = self.model.means.device
        dtype = self.model.means.dtype
        c2w = c2w.to(device=device, dtype=dtype)
        R = c2w[:3, :3]
        t = c2w[:3, 3]

        def _scalar(x):
            if isinstance(x, torch.Tensor):
                return float(x.detach().cpu().reshape(-1)[0].item())
            return float(x)
        fx = _scalar(camera.fx); fy = _scalar(camera.fy)
        cx = _scalar(camera.cx); cy = _scalar(camera.cy)
        W = int(_scalar(camera.width)); H = int(_scalar(camera.height))

        means_cam = (self.model.means - t[None, :]) @ R  # camera-space
        depths = -means_cam[:, 2]
        in_front = depths > 1e-6
        safe_d = torch.where(in_front, depths, torch.ones_like(depths))
        u = fx * (means_cam[:, 0] / safe_d) + cx
        v = fy * (-means_cam[:, 1] / safe_d) + cy
        visible = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)

        cam_pos = t

        best_id = 0
        best_dist = float("inf")
        for iid in unique_ids:
            inst_mask = (ids == iid)
            if not inst_mask.any():
                continue
            n_visible = int((visible & inst_mask).sum().item())
            centroid = self.model.means[inst_mask].mean(dim=0)
            d = float((centroid - cam_pos).norm())
            CONSOLE.log(
                f"[dynamic-gs-live] D0 candidate instance_id={int(iid)} "
                f"visible_gauss={n_visible} dist_to_cam={d:.3f} m"
            )
            if n_visible < self._D0_MIN_VISIBLE_GAUSSIANS:
                continue
            # Closest centroid among instances that pass the visibility
            # gate. Distance is the ranking — visibility only ensures the
            # object is actually on screen.
            if d < best_dist:
                best_dist = d
                best_id = int(iid)
        return best_id

    def _on_tracker_frame(
        self,
        camera,
        batch: dict,
        cdn: Optional[torch.Tensor],
        is_first: bool,
    ) -> None:
        """Live: Mode B feedforward cadence (same gate as recorded)."""
        if is_first:
            return
        N = int(self.config.feedforward_recurring_every_n_ticks)
        if N <= 0:
            return
        if (self._tracker_tick_count % N) != 0:
            return
        import time as _time
        gap = (
            self.config.feedforward_anysplat_min_gap_s
            if str(self.config.enable_feedforward_inpaint) == "anysplat_decode"
            else self.config.feedforward_recurring_min_gap_s
        )
        now = _time.time()
        if gap > 0 and (now - self._last_feedforward_wall_time) < gap:
            return
        self._last_feedforward_wall_time = now
        if self._latest_tracker_frame is None:
            return
        self._run_feedforward(self._latest_tracker_frame, mode_label="recurring")

    # ====================================================================
    # Recorded-method overrides (live equivalents)
    # ====================================================================

    def _resolve_anysplat_context_image_paths(
        self, target_frame_idx: int
    ) -> tuple[list[Path], list[int]]:
        """Live override: AnySplat worker is a subprocess in a different
        conda env, so it reads its input from disk. Dump the latest live
        RGB to a /dev/shm tmpfs PNG (single fixed filename per process --
        FF calls are serialized by ``_anysplat_slot_lock`` so we never
        race the dump). Returns ([path], [frame_idx])."""
        rgb_bgr = self._latest_live_rgb_bgr
        if rgb_bgr is None:
            return [], []
        try:
            import cv2
        except Exception as e:
            CONSOLE.log(f"[dynamic-gs-live] cv2 unavailable for FF image dump: {e}")
            return [], []
        path = Path(f"/dev/shm/dgs_live_ff_frame_{os.getpid()}.png")
        cv2.imwrite(str(path), rgb_bgr)
        return [path], [int(target_frame_idx)]

    def _scene_c2w_for_frame(self, frame_idx: int) -> np.ndarray:
        """Live override: live 'frame_idx' is just a monotonic counter,
        not a dataset index. Return the c2w of the current tracker frame
        (the FF dispatcher snapshots state at dispatch time so this is
        the right pose for the frame the dispatcher just saw)."""
        if self._latest_tracker_frame is None:
            raise RuntimeError("_scene_c2w_for_frame called before first tracker tick")
        cam = self._latest_tracker_frame["camera"]
        c2w = cam.camera_to_worlds
        if c2w.ndim == 3:
            c2w = c2w[0]
        if c2w.shape == (3, 4):
            bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=c2w.device, dtype=c2w.dtype)
            c2w = torch.cat([c2w, bottom], dim=0)
        return c2w.detach().cpu().numpy().astype(np.float32)

    # ====================================================================
    # Helpers (copied from RecordedDynamicGSPipeline -- shared logic; if a
    # third subclass appears these should move to the base)
    # ====================================================================

    @torch.no_grad()
    def _bootstrap_d0(self, camera, batch) -> None:
        """First-tick bootstrap: pick the moved object, write object_flags,
        capture reference pose, seed the XFeat tracker. Identical to the
        recorded subclass's _bootstrap_d0; the only live-specific bit is
        which :meth:`_pick_d0_object` overload runs (3D vs 2D)."""
        instance_ids_buf = self.model.object_instance_ids
        picked = self._pick_d0_object(camera, instance_ids_buf)
        if picked == 0:
            # Defer D0: no instance has enough visible Gaussians in this
            # frame's frustum. Stay quiet after the first few attempts —
            # the user just needs to point the camera at the object.
            self._d0_defer_attempts += 1
            if self._d0_defer_attempts in (1, 5, 25, 125):
                CONSOLE.log(
                    f"[dynamic-gs-live] D0 deferred (attempt "
                    f"#{self._d0_defer_attempts}): no prefused instance has "
                    f">={self._D0_MIN_VISIBLE_GAUSSIANS} centres in the current "
                    f"camera frustum. Point the camera at the object."
                )
            return

        CONSOLE.log(
            f"[dynamic-gs-live] D0 picked instance_id={picked} "
            f"after {self._d0_defer_attempts} deferred attempt(s)"
        )
        # Shared reseed (object_flags + reference pose + object mask + XFeat
        # anchor seed). The live ``_reset_d0_guard`` override sets
        # ``_d0_completed = True`` at the end, so the next tick runs DN.
        self._reseed_tracked_object(int(picked), camera, batch)

    def _reset_d0_guard(self) -> None:
        """Live override: D0 is complete once a reseed succeeds. Also resets the
        tick counter so the new object is treated as a fresh D0 by DN gating."""
        self._tracker_tick_count = 0
        self._d0_completed = True

    @torch.no_grad()
    def _compute_tick_cdn(self, camera, batch):
        """Render + compare for the change mask. Returns the CDN tensor.
        Identical to recorded's helper."""
        try:
            outputs = self._render_from_camera(camera)
        except Exception as exc:
            CONSOLE.log(f"[dynamic-gs-live] render for CDN failed: {exc}")
            return None
        rendered_rgb = outputs.get("rgb")
        rendered_depth = outputs.get("depth")
        rendered_alpha = outputs.get("accumulation")
        if rendered_rgb is None:
            return None

        bg = self.model._get_background_color()
        live_rgb = self.model.composite_with_background(
            self.model.get_gt_img(batch["image"]), bg
        )
        gt_depth = self.model._get_gt_depth(batch)
        gripper_mask = self.model._get_batch_mask(batch)

        try:
            obj_mask = self._render_object_mask_cached(camera)
        except Exception:
            obj_mask = None

        try:
            cdn = self._compute_change_mask(
                rendered_rgb=rendered_rgb,
                rendered_depth=rendered_depth,
                live_rgb=live_rgb,
                gt_depth=gt_depth,
                gripper_mask=gripper_mask,
                object_mask=obj_mask,
                rendered_alpha=rendered_alpha,
            )
            return cdn
        except Exception as exc:
            CONSOLE.log(f"[dynamic-gs-live] _compute_change_mask failed: {exc}")
            return None

    # ====================================================================
    # LiveFrame -> Splatfacto batch dict
    # ====================================================================

    def _batch_from_live_frame(self, frame, device) -> dict:
        """Convert a :class:`LiveFrame` (BGR uint8 + float32 depth + uint8
        keep-mask) into the Splatfacto batch shape:
          - ``"image"``       : (H, W, 3) float32 in [0, 1], RGB order
          - ``"depth_image"`` : (H, W, 1) float32 in meters
          - ``"mask"``        : (H, W, 1) bool (True = keep, i.e. not gripper)
        """
        rgb_uint8 = frame.rgb_bgr[..., ::-1]  # BGR -> RGB, no copy
        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb_uint8)).to(device).float() / 255.0
        depth_t = torch.from_numpy(frame.depth_m).to(device).float().unsqueeze(-1)
        mask_t = torch.from_numpy(frame.mask_keep).to(device).bool().unsqueeze(-1)
        return {
            "image": rgb_t,
            "depth_image": depth_t,
            "mask": mask_t,
        }
