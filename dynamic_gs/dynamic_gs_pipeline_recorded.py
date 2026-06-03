"""Recorded-mode subclass of :class:`DynamicGSPipelineBase`.

Frame source: iterates the recorded dynamic dataset frame-by-frame.

D0 object selection: 2D anchor pick — closest prefused 2D centroid to
the lower-centre anchor ``(W/2, 0.75*H)``, validated by CD0 overlap.
Robust for teleoperation datasets where the gripper-held object sits
in the lower-centre of the image.

Use this pipeline for:
* Testing new tracker / decoder algorithms against existing recorded data.
* Evaluating feedforward hole-fill quality on benchmark datasets.
* Reproducing past runs deterministically.

For live ROS-driven operation see :mod:`dynamic_gs_pipeline_live`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Type

import numpy as np
import torch

from nerfstudio.utils.rich_utils import CONSOLE

from .dynamic_gs_pipeline_base import (
    DynamicGSPipelineBase,
    DynamicGSPipelineBaseConfig,
    TrackerFrame,
)


@dataclass
class RecordedDynamicGSPipelineConfig(DynamicGSPipelineBaseConfig):
    """Config for ``RecordedDynamicGSPipeline``. Currently a thin
    extension of the base — all recorded-specific behavior is hard-coded
    rather than configurable."""

    _target: Type = field(default_factory=lambda: RecordedDynamicGSPipeline)

    d0_anchor_x_ratio: float = 0.5
    """Horizontal anchor position as a fraction of image width. 0.5 =
    center. Used by :meth:`_pick_d0_object` to pick the closest
    prefused-object centroid."""
    d0_anchor_y_ratio: float = 0.75
    """Vertical anchor position. 0.75 = lower-third — matches the
    typical position of a gripper-held object in teleop datasets."""


class RecordedDynamicGSPipeline(DynamicGSPipelineBase):
    """Recorded-mode dynamic pipeline.

    Per-step flow:
      1. Pull dataset frame ``[self._next_frame_to_track]`` via the
         datamanager.
      2. On D0: render the prefused scene; pick the moved-object
         instance via :meth:`_pick_d0_object`; set ``model.object_flags``
         to that instance; seed the XFeat motion estimator.
      3. On DN: call :meth:`_apply_motion_estimator` to advance the
         rigid transform on the tracked-object Gaussians.
      4. Compute CDN (render + change_mask).
      5. Write to :attr:`_latest_tracker_frame`.
      6. Push viser-direct + viewer re-render.
      7. Fire :meth:`_on_tracker_frame` (Mode B cadence check happens
         there if enabled).
      8. Advance ``_next_frame_to_track``.
    """

    config: RecordedDynamicGSPipelineConfig

    def __init__(
        self,
        config: RecordedDynamicGSPipelineConfig,
        device: str,
        test_mode="val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler=None,
    ):
        self._next_frame_to_track: int = 0
        super().__init__(
            config=config, device=device, test_mode=test_mode,
            world_size=world_size, local_rank=local_rank,
            grad_scaler=grad_scaler,
        )
        # Switch BOTH datamanager AND model to dynamic phase. The datamanager
        # switch makes next_train pull dynamic frames; the model switch flips
        # _apply_phase_trainability into its dynamic branch so means keeps
        # requires_grad=True. Without the model switch, the static-phase
        # branch fires and sets means.requires_grad=False, which breaks
        # register_hook on the next insert_inpaint_gaussians call.
        if hasattr(self.datamanager, "set_phase"):
            self.datamanager.set_phase("dynamic")
        if hasattr(self.model, "set_phase"):
            self.model.set_phase("dynamic")
        n_dyn = self.datamanager.get_num_dynamic_frames() if hasattr(
            self.datamanager, "get_num_dynamic_frames"
        ) else 0
        self._accepted_dynamic_frames = list(range(n_dyn))
        CONSOLE.log(
            f"[dynamic-gs-recorded] ready: {n_dyn} dynamic frames available "
            f"(D0 anchor at ({config.d0_anchor_x_ratio:.2f}*W, {config.d0_anchor_y_ratio:.2f}*H))"
        )

    # ====================================================================
    # Abstract hook implementations
    # ====================================================================

    def _tracker_tick(self, step: int) -> None:
        """Advance one dataset frame, run XFeat, update :attr:`_latest_tracker_frame`."""
        n_total = self.datamanager.get_num_dynamic_frames()
        if self._next_frame_to_track >= n_total:
            return  # nothing more to process

        # Pacing: only advance to the next dataset frame every N trainer steps.
        N = max(1, int(self.config.dynamic_steps_per_frame))
        if step > 0 and (step % N) != 0 and self._tracker_tick_count > 0:
            # No-op tick — the trainer just spins until the next frame is due.
            return

        frame_idx = self._next_frame_to_track
        is_first = self._tracker_tick_count == 0

        # Pin datamanager to this frame and pull the batch.
        self.datamanager.set_dynamic_frame_idx(frame_idx)
        camera, batch = self.datamanager.get_current_dynamic_train_batch()

        # D0 bootstrap: pick the moved object, init XFeat anchor.
        if is_first:
            self._bootstrap_d0(camera, batch)
        else:
            self._apply_motion_estimator(camera, batch)

        # Render + CDN for downstream feedforward.
        cdn = self._compute_tick_cdn(camera, batch)

        # Publish to latest tracker frame.
        self._latest_tracker_frame = {
            "frame_idx": frame_idx,
            "camera": camera,
            "cdn": cdn,
            "batch": batch,
            "stamp_sec": None,
        }
        self._global_frame_counter += 1
        self._tracker_tick_count += 1

        # Visualization: push to viser-direct, also kick the Nerfstudio viewer.
        self._build_viser_direct_handles(camera)
        self._push_viser_direct_transforms()
        self._force_viewer_rerender()

        # Subclass / Mode B hook.
        self._on_tracker_frame(camera, batch, cdn, is_first)

        # Advance frame index for next tick.
        self._next_frame_to_track += 1

    def _pick_d0_object(
        self,
        camera,
        prefused_instance_ids: torch.Tensor,
    ) -> int:
        """2D anchor pick: closest prefused 2D centroid to
        ``(d0_anchor_x_ratio*W, d0_anchor_y_ratio*H)``."""
        ids = prefused_instance_ids.squeeze(-1) if prefused_instance_ids.ndim > 1 else prefused_instance_ids
        unique_ids = torch.unique(ids[ids > 0]).tolist()
        if not unique_ids:
            return 0

        def _scalar(x):
            return float(x.detach().cpu().reshape(-1)[0].item()) if isinstance(x, torch.Tensor) else float(x)
        W = int(_scalar(camera.width))
        H = int(_scalar(camera.height))
        anchor_x = self.config.d0_anchor_x_ratio * W
        anchor_y = self.config.d0_anchor_y_ratio * H

        # Project each prefused instance's 3D centroid into the camera.
        fx = _scalar(camera.fx); fy = _scalar(camera.fy)
        cx = _scalar(camera.cx); cy = _scalar(camera.cy)
        c2w = camera.camera_to_worlds
        if c2w.ndim == 3:
            c2w = c2w[0]
        c2w = c2w.to(self.model.means.device, dtype=self.model.means.dtype)
        R = c2w[:3, :3]; t = c2w[:3, 3]

        best_id = 0
        best_dist = float("inf")
        for iid in unique_ids:
            mask = (ids == iid)
            centroid = self.model.means[mask].mean(dim=0)
            centroid_cam = (centroid - t) @ R
            depth = -centroid_cam[2]
            if depth <= 1e-6:
                continue
            u = fx * (centroid_cam[0] / depth) + cx
            v = fy * (-centroid_cam[1] / depth) + cy
            d = ((u - anchor_x) ** 2 + (v - anchor_y) ** 2) ** 0.5
            CONSOLE.log(
                f"[dynamic-gs-recorded] D0 candidate instance_id={int(iid)} "
                f"at pixel ({float(u):.0f}, {float(v):.0f}), depth={float(depth):.2f} m, "
                f"anchor_distance={float(d):.0f} px"
            )
            if float(d) < best_dist:
                best_dist = float(d)
                best_id = int(iid)
        return best_id

    def _on_tracker_frame(
        self,
        camera,
        batch: dict,
        cdn: Optional[torch.Tensor],
        is_first: bool,
    ) -> None:
        """Recorded: Mode B feedforward cadence + optional anchor-video write.

        Mode A (oneshot) fires from :meth:`DynamicGSPipelineBase.get_train_loss_dict`
        — not here. Mode B fires every Nth tracker tick, additionally
        gated by a wall-clock floor (``feedforward_recurring_min_gap_s``)
        so high tracker rates don't dominate FF cost.
        """
        if is_first:
            return  # no FF on D0
        N = int(self.config.feedforward_recurring_every_n_ticks)
        if N <= 0:
            return
        if (self._tracker_tick_count % N) != 0:
            return
        # Wall-clock floor.
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
    # D0 bootstrap (recorded-specific implementation)
    # ====================================================================

    @torch.no_grad()
    def _bootstrap_d0(self, camera, batch) -> None:
        """First-tick bootstrap: pick moved-object instance, write
        ``model.object_flags``, capture reference object pose, seed the
        XFeat tracker.

        After this method:
          * ``model.object_flags`` has 1s for the picked instance only.
          * Reference pose captured (so future ``apply_rigid_object_transform_from_reference``
            applies deltas relative to D0).
          * :attr:`_motion_estimator` is initialized + anchor frame seeded.
          * :attr:`_d0_selected_instance_id` is set (so viser-direct knows
            which Gaussians are the tracked object).
        """
        # 1) Pick the moved-object instance from prefused candidates.
        instance_ids_buf = self.model.object_instance_ids
        picked = self._pick_d0_object(camera, instance_ids_buf)
        self._d0_selected_instance_id = int(picked)
        if picked == 0:
            CONSOLE.log(
                "[dynamic-gs-recorded] WARNING: no prefused instance picked at D0. "
                "Tracker will be disabled (no object to follow)."
            )
            return

        CONSOLE.log(f"[dynamic-gs-recorded] D0 picked instance_id={picked}")

        # 2) Write object_flags = (instance_ids == picked).
        ids = instance_ids_buf.squeeze(-1) if instance_ids_buf.ndim > 1 else instance_ids_buf
        new_flags = (ids == picked).float().unsqueeze(-1)
        self.model.object_flags.copy_(new_flags)
        n_picked = int((self.model.object_flags.squeeze(-1) > 0.5).sum().item())
        CONSOLE.log(
            f"[dynamic-gs-recorded] object_flags written: {n_picked} Gaussians "
            f"flagged for instance_id={picked}"
        )

        # 3) Render the object mask (the picked instance's projected silhouette).
        try:
            obj_mask = self.model.render_object_mask(camera)
        except Exception as exc:
            CONSOLE.log(f"[dynamic-gs-recorded] render_object_mask failed at D0: {exc}")
            obj_mask = None

        # 4) Capture reference object pose so DN deltas apply from this baseline.
        # Pass the picked instance id so _tracked_object_mask becomes a STRICT
        # `instance_ids == d0_id` mask — survives FF inserts without count drift.
        if hasattr(self.model, "capture_reference_object_pose"):
            with self._viser_lock_ctx():
                self.model.capture_reference_object_pose(
                    instance_id=int(self._d0_selected_instance_id),
                )

        # 5) Seed the XFeat motion estimator.
        live_rgb = self._build_tracking_rgb(batch)
        depth = batch.get("depth_image")
        try:
            self._initialize_motion_estimator(live_rgb, depth, camera, obj_mask)
        except Exception as exc:
            CONSOLE.log(f"[dynamic-gs-recorded] _initialize_motion_estimator failed: {exc}")

    @torch.no_grad()
    def _compute_tick_cdn(self, camera, batch):
        """Render + compare for the change mask. Returns the CDN tensor."""
        try:
            outputs = self._render_from_camera(camera)
        except Exception as exc:
            CONSOLE.log(f"[dynamic-gs-recorded] render for CDN failed: {exc}")
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
            obj_mask = self.model.render_object_mask(camera)
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
            CONSOLE.log(f"[dynamic-gs-recorded] _compute_change_mask failed: {exc}")
            return None
