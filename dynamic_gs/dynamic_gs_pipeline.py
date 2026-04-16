from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type

import torch
import torch.nn.functional as TF
from PIL import Image, ImageDraw
from nerfstudio.engine.callbacks import TrainingCallbackAttributes
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler
from nerfstudio.utils.rich_utils import CONSOLE

from .dynamic_gs_datamanager import DynamicGSDataManagerConfig
from .dynamic_gs_model import DynamicGSModelConfig
from .utils import (
    CoTrackerMotionEstimator,
    build_change_mask,
    dilate_binary_mask,
    query_sam2_propagated_mask,
)


@dataclass
class DynamicGSPipelineConfig(VanillaPipelineConfig):
    _target: Type = field(default_factory=lambda: DynamicGSPipeline)

    datamanager: DynamicGSDataManagerConfig = field(default_factory=DynamicGSDataManagerConfig)
    model: DynamicGSModelConfig = field(default_factory=DynamicGSModelConfig)

    static_num_steps: int = 3000
    dynamic_steps_per_frame: int = 300
class DynamicGSPipeline(VanillaPipeline):
    config: DynamicGSPipelineConfig

    def __init__(self, config, device, test_mode="val", world_size=1, local_rank=0, grad_scaler=None):
        self.current_phase = None  # type: Optional[Literal["static", "dynamic"]]
        self.current_dynamic_frame_idx = None  # type: Optional[int]
        self.total_dynamic_frames = 0
        self.total_dynamic_steps = 0
        self._sam3d_inserted = False
        self._cotracker_motion = None
        self._global_frame_counter = 0
        # Live SAM2 tracker: D0 → D1 → D2 → ...
        self._live_tracker_rgb = None
        self._live_tracker_mask = None
        self._live_tracker_seeded = False
        self._timing = defaultdict(list)
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            grad_scaler=grad_scaler,
        )
        self.total_dynamic_frames = self.datamanager.get_num_dynamic_frames()
        self.total_dynamic_steps = self.total_dynamic_frames * self.config.dynamic_steps_per_frame
        self._sync_phase(0)

    def _reset_dynamic_segmentation_state(self) -> None:
        self._sam3d_inserted = False
        self._cotracker_motion = None
        self._global_frame_counter = 0
        self._live_tracker_rgb = None
        self._live_tracker_mask = None
        self._live_tracker_seeded = False

    @staticmethod
    def _resize_mask_to(mask: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """Resize a (H,W,C) mask to (target_h, target_w, C) using nearest interpolation."""
        if mask.shape[0] == target_h and mask.shape[1] == target_w:
            return mask
        return TF.interpolate(
            mask.permute(2, 0, 1).unsqueeze(0), size=(target_h, target_w), mode="nearest",
        ).squeeze(0).permute(1, 2, 0)

    # ---- CoTracker helpers ----

    @staticmethod
    def _has_nonempty_mask(mask) -> bool:
        return mask is not None and bool(torch.any(mask > 0.5))

    def _write_cotracker_motion_log(self, frame_name: str, motion_estimate) -> None:
        debug_dir = self._get_debug_dir()
        debug_dir.mkdir(parents=True, exist_ok=True)
        log_path = debug_dir / f"{frame_name}_cotracker_motion.txt"
        log_lines = [
            f"success: {motion_estimate.success}",
            f"ready: {motion_estimate.ready}",
            f"correspondence_count: {motion_estimate.correspondence_count}",
            f"inlier_count: {motion_estimate.inlier_count}",
            f"track_count_before: {motion_estimate.track_count_before}",
            f"track_count_after: {motion_estimate.track_count_after}",
            f"raw_visible_count: {motion_estimate.raw_visible_count}",
            f"mask_visible_count: {motion_estimate.mask_visible_count}",
            f"depth_valid_count: {motion_estimate.depth_valid_count}",
            f"used_mask_fallback: {motion_estimate.used_mask_fallback}",
            f"mean_residual: {motion_estimate.mean_residual}",
            f"median_residual: {motion_estimate.median_residual}",
            f"rotation: {motion_estimate.rotation.tolist()}",
            f"translation: {motion_estimate.translation.tolist()}",
        ]
        log_path.write_text("\n".join(log_lines) + "\n")

    def _apply_cotracker_motion(self, camera, batch, current_mask=None) -> None:
        if self._cotracker_motion is None:
            return
        if not self._cotracker_motion.ready:
            frame_name = self.datamanager.get_current_dynamic_frame_name()
            CONSOLE.log(
                f"[dynamic-gs] CoTracker skipped for {frame_name}: tracker not ready "
                f"({self._cotracker_motion.current_track_count} tracks, "
                f"min={self._cotracker_motion.min_track_points})"
            )
            return
        current_live_rgb = self.model.get_live_rgb(batch, apply_training_downscale=False)
        motion_estimate = self._cotracker_motion.estimate_and_advance(
            current_rgb=current_live_rgb,
            current_depth=batch["depth_image"],
            current_camera=camera,
            current_mask=current_mask,
        )
        frame_name = self.datamanager.get_current_dynamic_frame_name()
        self._write_cotracker_motion_log(frame_name, motion_estimate)
        self._save_cotracker_debug(frame_name, motion_estimate)
        if not motion_estimate.success:
            CONSOLE.log(
                f"[dynamic-gs] CoTracker rigid motion unavailable for {frame_name}: "
                f"raw={motion_estimate.raw_visible_count}, "
                f"mask={motion_estimate.mask_visible_count}, "
                f"depth={motion_estimate.depth_valid_count}, "
                f"correspondences={motion_estimate.correspondence_count}, "
                f"inliers={motion_estimate.inlier_count}, "
                f"mask_fallback={motion_estimate.used_mask_fallback}"
            )
            return
        moved_count = self.model.apply_rigid_object_transform_from_reference(
            motion_estimate.rotation, motion_estimate.translation,
        )
        if moved_count == 0:
            CONSOLE.log(
                f"[dynamic-gs] CoTracker estimated motion for {frame_name}, "
                "but no object Gaussians were moved. Check object_flags/reference pose consistency."
            )
        CONSOLE.log(
            f"[dynamic-gs] CoTracker rigid motion -> {frame_name}, moved={moved_count}, "
            f"inliers={motion_estimate.inlier_count}/{motion_estimate.correspondence_count}, "
            f"median residual={motion_estimate.median_residual:.5f}, "
            f"mask_fallback={motion_estimate.used_mask_fallback}"
        )

    def _initialize_cotracker(self, rs00_rgb, rs00_depth, camera, mask) -> None:
        if not self.model.config.enable_cotracker_rigid_motion:
            return
        self._cotracker_motion = CoTrackerMotionEstimator(
            device=self.model.device,
            query_point_count=self.model.config.cotracker_query_point_count,
            min_track_points=self.model.config.cotracker_min_track_points,
            ransac_iterations=self.model.config.cotracker_ransac_iterations,
            ransac_inlier_threshold=self.model.config.cotracker_ransac_inlier_threshold,
            point_refresh_min_distance=self.model.config.cotracker_point_refresh_min_distance,
            checkpoint_path=self.model.config.cotracker_checkpoint_path,
            hub_repo=self.model.config.cotracker_hub_repo,
            hub_model=self.model.config.cotracker_hub_model,
        )
        seeded = self._cotracker_motion.initialize(
            rgb=rs00_rgb, depth=rs00_depth, camera=camera, mask=mask,
        )
        CONSOLE.log(
            f"[dynamic-gs] CoTracker reference seed on D0 -> "
            f"fast={self._cotracker_motion.last_init_fast_point_count}, "
            f"sampled={self._cotracker_motion.last_init_sampled_count}, "
            f"depth_valid={self._cotracker_motion.last_init_depth_valid_count}, "
            f"dense_fallback={self._cotracker_motion.last_init_used_dense_fallback}, "
            f"tracks={seeded}, ready={self._cotracker_motion.ready}"
        )
        if seeded < self._cotracker_motion.min_track_points:
            CONSOLE.log(
                f"[dynamic-gs] CoTracker seeded too few D0 points: "
                f"{seeded} < min_track_points={self._cotracker_motion.min_track_points}"
            )

    # ---- Image helpers ----

    @staticmethod
    def _save_image(image, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensor = image.detach().float().clamp(0.0, 1.0)
        if tensor.ndim == 2:
            tensor = tensor[..., None]
        if tensor.shape[-1] == 1:
            tensor = tensor.repeat(1, 1, 3)
        image_uint8 = tensor.mul(255).byte().cpu().numpy()
        Image.fromarray(image_uint8).save(path)

    @staticmethod
    def _save_image_with_points(image, points_xy, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        tensor = image.detach().float().clamp(0.0, 1.0)
        if tensor.ndim == 2:
            tensor = tensor[..., None]
        if tensor.shape[-1] == 1:
            tensor = tensor.repeat(1, 1, 3)

        image_uint8 = tensor.mul(255).byte().cpu().numpy()
        pil_image = Image.fromarray(image_uint8)

        if points_xy is not None and points_xy.numel() > 0:
            draw = ImageDraw.Draw(pil_image)
            radius = max(2, int(round(0.006 * max(pil_image.size))))
            for point in points_xy.detach().cpu().tolist():
                x = int(round(point[0]))
                y = int(round(point[1]))
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0), outline=(255, 255, 255))

        pil_image.save(path)

    @staticmethod
    def _resize_points(points_xy, source_shape, target_shape):
        if points_xy is None or points_xy.numel() == 0:
            return points_xy
        source_h, source_w = source_shape[:2]
        target_h, target_w = target_shape[:2]
        scaled = points_xy.detach().clone().float()
        scaled[:, 0] *= float(target_w) / float(max(source_w, 1))
        scaled[:, 1] *= float(target_h) / float(max(source_h, 1))
        return scaled

    @staticmethod
    def _save_depth_image(depth, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensor = depth.detach().float()
        if tensor.ndim == 3 and tensor.shape[-1] == 1:
            tensor = tensor[..., 0]
        valid = torch.isfinite(tensor) & (tensor > 0.0)
        image = torch.zeros((*tensor.shape, 3), dtype=torch.float32, device=tensor.device)
        if bool(valid.any()):
            valid_values = tensor[valid]
            depth_min = float(valid_values.min().item())
            depth_max = float(valid_values.max().item())
            if depth_max > depth_min:
                normalized = (tensor - depth_min) / (depth_max - depth_min)
            else:
                normalized = torch.zeros_like(tensor)
            normalized = (1.0 - normalized).clamp(0.0, 1.0)
            image[valid] = normalized[valid][..., None].expand(-1, 3)
        image_uint8 = image.mul(255).byte().cpu().numpy()
        Image.fromarray(image_uint8).save(path)

    @staticmethod
    def _save_overlay(rgb, mask, path, color=(1.0, 0.0, 0.0), alpha=0.5):
        """Save rgb with a transparent colored mask overlay."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        img = rgb.detach().float().clamp(0.0, 1.0).clone()
        m = mask.detach().float()
        if m.ndim == 3 and m.shape[-1] == 1:
            m = m[..., 0]
        if m.ndim == 2:
            m = (m > 0.5).float()
        # Resize mask to match image if needed
        if m.shape[:2] != img.shape[:2]:
            m = TF.interpolate(
                m.unsqueeze(0).unsqueeze(0), size=img.shape[:2], mode="nearest",
            ).squeeze(0).squeeze(0)
        overlay = torch.tensor(color, device=img.device).view(1, 1, 3)
        img[m > 0.5] = img[m > 0.5] * (1 - alpha) + overlay.expand_as(img)[m > 0.5] * alpha
        image_uint8 = img.mul(255).byte().cpu().numpy()
        Image.fromarray(image_uint8).save(path)

    def _get_debug_dir(self) -> Path:
        return Path(self.datamanager.config.data) / self.datamanager.config.dynamic_subdir / "debug"

    def _get_cotracker_debug_dir(self) -> Path:
        return self._get_debug_dir() / "cotracker_debug"

    def _save_cotracker_debug(self, frame_name: str, est) -> None:
        """Save side-by-side image: previous frame with points → current frame with tracked points + lines."""
        if est.previous_points_xy is None or est.current_points_xy is None:
            return
        if est.previous_rgb is None or est.current_rgb is None:
            return
        import numpy as np

        # CoTracker stores images in 0-255 range
        prev_img = est.previous_rgb.detach().float().cpu().numpy()
        curr_img = est.current_rgb.detach().float().cpu().numpy()
        if prev_img.max() > 1.5:
            prev_img = prev_img / 255.0
        if curr_img.max() > 1.5:
            curr_img = curr_img / 255.0
        prev_img = prev_img.clip(0, 1)
        curr_img = curr_img.clip(0, 1)
        h, w = prev_img.shape[:2]

        # Create side-by-side canvas
        canvas = np.concatenate([prev_img, curr_img], axis=1)
        canvas = (canvas * 255).astype(np.uint8).copy()

        prev_pts = est.previous_points_xy  # (K, 2) x,y
        curr_pts = est.current_points_xy   # (K, 2) x,y
        inlier_mask = est.tracked_inlier_mask
        n = min(len(prev_pts), len(curr_pts))

        for i in range(n):
            px, py = int(prev_pts[i, 0]), int(prev_pts[i, 1])
            cx, cy = int(curr_pts[i, 0]) + w, int(curr_pts[i, 1])
            is_inlier = bool(inlier_mask[i]) if inlier_mask is not None and i < len(inlier_mask) else False
            point_color = [0, 255, 0] if is_inlier else [255, 0, 0]
            line_color = [0, 180, 0] if is_inlier else [180, 0, 0]

            # Draw the correspondence line first so the point color remains visible on top.
            steps = max(abs(cx - px), abs(cy - py), 1)
            for t in range(steps + 1):
                lx = int(px + (cx - px) * t / steps)
                ly = int(py + (cy - py) * t / steps)
                if 0 <= ly < h and 0 <= lx < 2 * w:
                    canvas[ly, lx] = line_color

            # Draw inliers in green and everything else in red, with a larger marker.
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if 0 <= py + dy < h and 0 <= px + dx < w:
                        canvas[py + dy, px + dx] = point_color
                    if 0 <= cy + dy < h and 0 <= cx + dx < 2 * w:
                        canvas[cy + dy, cx + dx] = point_color

        dbg = self._get_cotracker_debug_dir()
        dbg.mkdir(parents=True, exist_ok=True)
        Image.fromarray(canvas).save(dbg / f"{frame_name}_cotracker.png")

    @torch.no_grad()
    def _render_from_camera(self, camera):
        """Render from camera in training mode (to get training-resolution output)."""
        was_training = self.model.training
        self.model.train()
        try:
            return self.model.get_outputs(camera.to(self.model.device))
        finally:
            if not was_training:
                self.model.eval()

    def _compute_change_mask(self, rendered_rgb, rendered_depth, live_rgb, gt_depth, gripper_mask, object_mask):
        """Compute change mask between render and live, excluding gripper + object regions."""
        target_h, target_w = rendered_rgb.shape[:2]
        # Combine gripper mask and object mask into valid_mask
        valid_mask = None
        if object_mask is not None:
            obj = object_mask.float()
            if obj.ndim == 2:
                obj = obj[..., None]
            obj = self._resize_mask_to(obj.to(self.model.device), target_h, target_w)
            valid_mask = 1.0 - obj
        if gripper_mask is not None:
            grip = gripper_mask.float().to(self.model.device)
            if grip.ndim == 2:
                grip = grip[..., None]
            grip = self._resize_mask_to(grip, target_h, target_w)
            valid_mask = grip * valid_mask if valid_mask is not None else grip

        change_mask = build_change_mask(
            rendered_depth, gt_depth,
            pred_rgb=rendered_rgb, gt_rgb=live_rgb,
            valid_mask=valid_mask,
            depth_threshold=self.model.config.change_mask_depth_threshold,
            rgb_threshold=self.model.config.change_mask_rgb_threshold,
            use_rgb=self.model.config.change_mask_use_rgb,
            blur_kernel_size=self.model.config.change_mask_blur_kernel_size,
            blur_sigma=self.model.config.change_mask_blur_sigma,
            filter_radius=self.model.config.change_mask_filter_radius,
            min_component_size=self.model.config.change_mask_min_component_size,
        )
        if self.model.config.active_mask_dilate_radius > 0:
            change_mask = dilate_binary_mask(change_mask, self.model.config.active_mask_dilate_radius)
        return change_mask

    # ---- Step/phase management ----

    def _total_train_steps(self) -> int:
        return self.config.static_num_steps + self.total_dynamic_steps

    def _phase_for_step(self, step: int) -> Literal["static", "dynamic"]:
        if self.total_dynamic_frames == 0 or step < self.config.static_num_steps:
            return "static"
        return "dynamic"

    def _dynamic_frame_for_step(self, step: int) -> int:
        dynamic_step = max(step - self.config.static_num_steps, 0)
        return min(dynamic_step // self.config.dynamic_steps_per_frame, self.total_dynamic_frames - 1)

    # ---- Core: per-frame processing ----

    def _prepare_dynamic_frame(self) -> None:
        frame_idx = self.current_dynamic_frame_idx
        self.datamanager.set_dynamic_frame_idx(frame_idx)
        camera, batch = self.datamanager.get_current_dynamic_train_batch()
        frame_name = self.datamanager.get_dynamic_frame_name(frame_idx)
        is_first = self._global_frame_counter == 0

        # All mask/change operations at training resolution for consistency
        bg = self.model._get_background_color()
        live_rgb = self.model.get_live_rgb(batch, background=bg, apply_training_downscale=True)
        gt_rgb = self.model.composite_with_background(self.model.get_gt_img(batch["image"]), bg)
        gt_depth = self.model._get_gt_depth(batch)
        gripper_mask = self.model._get_batch_mask(batch)

        if is_first:
            init_debug_dir = self.datamanager.get_initialization_debug_dir()
            init_artifact_dir = self.datamanager.get_initialization_artifact_dir()
            self._prepare_frame_0(
                camera, batch, live_rgb, gt_rgb, gt_depth, gripper_mask,
                frame_name, init_debug_dir, init_artifact_dir,
            )
        else:
            debug_dir = self._get_debug_dir()
            self._prepare_frame_n(camera, batch, live_rgb, gt_rgb, gt_depth, gripper_mask, frame_name, debug_dir)

        self._global_frame_counter += 1

    def _prepare_frame_0(
        self, camera, batch, live_rgb, gt_rgb, gt_depth, gripper_mask,
        frame_name, debug_dir, artifact_dir,
    ):
        """Bootstrap: ESAM → SAM3D → rendered object mask → seed live tracker → CoTracker → CD0."""
        t_total = time.time()

        # --- TIMING: D0.1 Initial change detection (render RS, MSSIM change mask, ESAM on RS + D0, flag Gaussians) ---
        t0 = time.time()
        stats = self.model.prepare_dynamic_update(camera, batch)
        self._timing["D0.1_initial_change_detection"].append(time.time() - t0)
        # Record substep breakdown for the timing report
        for k, v in stats.get("prepare_dynamic_update_substeps", {}).items():
            self._timing[k].append(v)
        render_mask_plain_path = debug_dir / f"{frame_name}_render_object_mask_binary.png"
        self._save_image(live_rgb, debug_dir / f"{frame_name}_live_input.png")
        self._save_depth_image(gt_depth, debug_dir / f"{frame_name}_live_depth.png")
        self._save_image(stats["rendered_rgb"], debug_dir / f"{frame_name}_render.png")
        self._save_depth_image(stats["rendered_depth"], debug_dir / f"{frame_name}_render_depth.png")
        self._save_image_with_points(stats["change_mask"], None, debug_dir / f"{frame_name}_change_mask.png")
        self._save_image(stats["render_object_mask"], render_mask_plain_path)
        self._save_image_with_points(
            stats["render_object_mask"],
            stats.get("render_prompt_points"),
            debug_dir / f"{frame_name}_render_object_mask.png",
        )
        self._save_image(stats["live_object_mask"], debug_dir / f"{frame_name}_live_object_mask_binary.png")
        self._save_image_with_points(
            stats["live_object_mask"],
            stats.get("live_prompt_points"),
            debug_dir / f"{frame_name}_live_object_mask.png",
        )
        self._save_image_with_points(stats["optim_mask"], None, debug_dir / f"{frame_name}_optim_mask.png")
        self._save_image_with_points(
            stats["render_propagation_mask"],
            stats.get("render_prompt_points"),
            debug_dir / f"{frame_name}_render_propagation_mask.png",
        )
        self._save_image_with_points(
            stats["live_propagation_mask"],
            stats.get("live_prompt_points"),
            debug_dir / f"{frame_name}_live_propagation_mask.png",
        )
        if gripper_mask is not None:
            self._save_image_with_points(gripper_mask.float(), None, debug_dir / f"{frame_name}_gripper_mask.png")

        # --- TIMING: D0.2 SAM3D generation (subprocess) + D0.3 SAM3D insertion (CPD similarity + dedup + insert) ---
        if not self._sam3d_inserted and self.model.config.use_sam3d_object_init:
            sam3d_stats = self.model.initialize_object_from_sam3d(
                render_image_path=debug_dir / f"{frame_name}_render.png",
                object_mask_path=render_mask_plain_path,
                render_object_mask=stats["render_object_mask"],
                rendered_depth=stats["rendered_depth"],
                camera=camera, image_debug_dir=debug_dir, artifact_dir=artifact_dir, frame_name=frame_name,
            )
            if sam3d_stats:
                self._timing["D0.2_sam3d_generation"].append(sam3d_stats.get("sam3d_generation_time", 0.0))
                self._timing["D0.3_sam3d_insertion"].append(sam3d_stats.get("sam3d_insertion_time", 0.0))
                self._sam3d_inserted = True
                self.model.refresh_dynamic_state_after_insertion(
                    camera, stats["render_object_mask"], stats["optim_mask"],
                )
                CONSOLE.log(
                    f"[dynamic-gs] SAM3D object init -> existing={sam3d_stats['existing_object_gaussians']}, "
                    f"scale={sam3d_stats['chosen_scale']:.4f}, "
                    f"generated={sam3d_stats['sam3d_generated_points']}, kept={sam3d_stats['kept_points_after_dedup']}"
                )
            else:
                self._timing["D0.2_sam3d_generation"].append(0.0)
                self._timing["D0.3_sam3d_insertion"].append(0.0)
        else:
            self._timing["D0.2_sam3d_generation"].append(0.0)
            self._timing["D0.3_sam3d_insertion"].append(0.0)

        # --- TIMING: D0.4 Render object mask (rasterize only object_flags > 0.5 Gaussians, threshold, dilate) ---
        t0 = time.time()
        rendered_obj_mask = self.model.render_object_mask(camera)
        self._timing["D0.4_render_object_mask"].append(time.time() - t0)

        # --- TIMING: D0.5 Seed live SAM2 tracker (store D0 live RGB + f0_live for SAM2 propagation chain) ---
        t0 = time.time()
        f0_live = stats["live_object_mask"]
        self._live_tracker_rgb = live_rgb.detach().clone()
        self._live_tracker_mask = f0_live.detach().clone()
        self._live_tracker_seeded = True
        self._timing["D0.5_seed_live_tracker"].append(time.time() - t0)

        # --- TIMING: D0.6 CoTracker init (freeze D0 as reference, sample points once, backproject reference 3D once) ---
        t0 = time.time()
        live_rgb_fullres = self.model.get_live_rgb(batch, apply_training_downscale=False)
        self._initialize_cotracker(live_rgb_fullres, batch["depth_image"], camera, f0_live)
        self.model.capture_reference_object_pose()
        self._timing["D0.6_cotracker_init"].append(time.time() - t0)

        f0_live_resized = self._resize_mask_to(
            f0_live.float() if f0_live.ndim == 3 else f0_live[..., None].float(),
            rendered_obj_mask.shape[0], rendered_obj_mask.shape[1],
        ).to(self.model.device)
        combined_obj_mask = torch.maximum(rendered_obj_mask, f0_live_resized)
        resized_live_prompt_points = self._resize_points(
            stats.get("live_prompt_points"),
            f0_live.shape,
            rendered_obj_mask.shape,
        )
        self._save_image_with_points(rendered_obj_mask, None, debug_dir / f"{frame_name}_rendered_object_mask.png")
        self._save_image_with_points(
            f0_live_resized,
            resized_live_prompt_points,
            debug_dir / f"{frame_name}_live_object_mask_resized.png",
        )
        self._save_image_with_points(combined_obj_mask, None, debug_dir / f"{frame_name}_combined_object_mask.png")

        # --- TIMING: D0.7 Render RS00 (re-render scene after SAM3D object insertion) ---
        t0 = time.time()
        rs00_outputs = self._render_from_camera(camera)
        self._timing["D0.7_render_rs00"].append(time.time() - t0)
        rs00_rgb = rs00_outputs["rgb"]

        # --- TIMING: D0.8 Change mask CD0 (MSSIM comparison RS00 vs D0, excluding gripper + object union mask) ---
        t0 = time.time()
        cd0 = self._compute_change_mask(rs00_rgb, rs00_outputs["depth"], gt_rgb, gt_depth, gripper_mask, combined_obj_mask)
        self._timing["D0.8_change_mask_cd0"].append(time.time() - t0)

        # --- TIMING: D0.9 Debug images (save ~9 overlay PNGs to disk) ---
        t0 = time.time()
        dbg = debug_dir
        self._save_overlay(gt_rgb, cd0, dbg / f"{frame_name}_live_w_cd0.png")
        self._save_overlay(rs00_rgb, cd0, dbg / f"{frame_name}_render_w_cd0.png")
        self._save_overlay(rs00_rgb, rendered_obj_mask, dbg / f"{frame_name}_render_w_objmask.png", color=(0, 0, 1))
        self._save_overlay(gt_rgb, f0_live_resized, dbg / f"{frame_name}_live_w_f0live.png", color=(0, 0, 1))
        self._save_overlay(rs00_rgb, combined_obj_mask, dbg / f"{frame_name}_render_w_combined.png", color=(0, 1, 1))
        if gripper_mask is not None:
            self._save_overlay(gt_rgb, gripper_mask, dbg / f"{frame_name}_live_w_gripper.png", color=(0, 1, 0))
        self._save_image(gt_rgb, dbg / f"{frame_name}_live.png")
        self._save_image(rs00_rgb, dbg / f"{frame_name}_rs00.png")
        self._save_image_with_points(cd0, None, dbg / f"{frame_name}_cd0.png")
        self._timing["D0.9_debug_images"].append(time.time() - t0)

        self.model._set_optim_mask(cd0)
        self.model._dynamic_ready = True

        self._timing["D0.10_total_frame_0"].append(time.time() - t_total)

        change_px = int((cd0[..., 0] > 0.5).sum().item()) if cd0.ndim >= 3 else int((cd0 > 0.5).sum().item())
        CONSOLE.log(
            f"[dynamic-gs] frame 0 ({frame_name}): bootstrap complete, "
            f"change px={change_px}, object flags={int((self.model.object_flags.squeeze(-1) > 0.5).sum().item())}"
        )
        CONSOLE.log(
            f"[timing] frame 0: total={self._timing['D0.10_total_frame_0'][-1]:.2f}s, "
            f"change_detect={self._timing['D0.1_initial_change_detection'][-1]:.2f}s, "
            f"sam3d_gen={self._timing['D0.2_sam3d_generation'][-1]:.2f}s, "
            f"sam3d_ins={self._timing['D0.3_sam3d_insertion'][-1]:.2f}s, "
            f"obj_mask={self._timing['D0.4_render_object_mask'][-1]:.2f}s, "
            f"cotracker_init={self._timing['D0.6_cotracker_init'][-1]:.2f}s, "
            f"render_rs00={self._timing['D0.7_render_rs00'][-1]:.2f}s, "
            f"change_mask={self._timing['D0.8_change_mask_cd0'][-1]:.2f}s, "
            f"debug_imgs={self._timing['D0.9_debug_images'][-1]:.2f}s"
        )

    def _prepare_frame_n(self, camera, batch, live_rgb, gt_rgb, gt_depth, gripper_mask, frame_name, debug_dir):
        """Frame N>=1: live SAM2 → reference-frame CoTracker → absolute rigid transform → render → rendered obj mask → CDN."""
        t_total = time.time()

        # --- TIMING: DN.1 SAM2 live mask propagation (propagate object mask D(N-1) → DN on live images) ---
        t0 = time.time()
        fdn_live = None
        if self._live_tracker_seeded:
            fdn_live = query_sam2_propagated_mask(
                self.model._get_sam2_video_predictor(),
                self._live_tracker_rgb, live_rgb, self._live_tracker_mask,
            )
            fdn_live = fdn_live[..., None].float() if fdn_live.ndim == 2 else fdn_live.float()
            fdn_live = fdn_live.to(self.model.device)
            self._live_tracker_rgb = live_rgb.detach().clone()
            self._live_tracker_mask = fdn_live.detach().clone()
        self._timing["DN.1_sam2_live_propagation"].append(time.time() - t0)

        # --- TIMING: DN.2 CoTracker reference mode (no reseed; keep D0 query points fixed) ---
        if self._sam3d_inserted:
            t0 = time.time()
            self._timing["DN.2_cotracker_refresh"].append(time.time() - t0)

            # --- TIMING: DN.3 CoTracker absolute rigid transform (reference D0 -> DN, current-mask filter, RANSAC, apply absolute SE(3)) ---
            t0 = time.time()
            self._apply_cotracker_motion(camera, batch, current_mask=fdn_live)
            self._timing["DN.3_cotracker_advance"].append(time.time() - t0)

            # --- TIMING: DN.4 CoTracker post-filter skipped (reference queries stay fixed) ---
            t0 = time.time()
            self._timing["DN.4_cotracker_filter"].append(time.time() - t0)

        # --- TIMING: DN.5 Render RDN (render full scene after rigid transform applied to object Gaussians) ---
        t0 = time.time()
        rdn_outputs = self._render_from_camera(camera)
        rdn_rgb = rdn_outputs["rgb"]
        rdn_depth = rdn_outputs["depth"]
        self._timing["DN.5_render_rdn"].append(time.time() - t0)

        # --- TIMING: DN.6 Render object mask (rasterize only object_flags > 0.5 Gaussians from simulation) ---
        t0 = time.time()
        rendered_obj_mask = self.model.render_object_mask(camera)
        self._timing["DN.6_render_object_mask"].append(time.time() - t0)

        # Union mask: max(rendered simulation mask, SAM2 live mask)
        if fdn_live is not None:
            fdn_resized = self._resize_mask_to(
                fdn_live, rendered_obj_mask.shape[0], rendered_obj_mask.shape[1],
            )
            combined_obj_mask = torch.maximum(rendered_obj_mask, fdn_resized)
        else:
            combined_obj_mask = rendered_obj_mask

        # --- TIMING: DN.7 Change mask CDN (MSSIM comparison RDN vs DN, excluding gripper + union object mask) ---
        t0 = time.time()
        cdn = self._compute_change_mask(rdn_rgb, rdn_depth, gt_rgb, gt_depth, gripper_mask, combined_obj_mask)
        self._timing["DN.7_change_mask_cdn"].append(time.time() - t0)

        # --- TIMING: DN.8 Debug images (save ~9 overlay PNGs to disk) ---
        t0 = time.time()
        dbg = self._get_debug_dir()
        self._save_overlay(gt_rgb, cdn, dbg / f"{frame_name}_live_w_cdn.png")
        self._save_overlay(rdn_rgb, cdn, dbg / f"{frame_name}_render_w_cdn.png")
        self._save_overlay(rdn_rgb, rendered_obj_mask, dbg / f"{frame_name}_render_w_objmask.png", color=(0, 0, 1))
        if fdn_live is not None:
            self._save_overlay(gt_rgb, fdn_live, dbg / f"{frame_name}_live_w_fdn.png", color=(0, 0, 1))
        self._save_overlay(rdn_rgb, combined_obj_mask, dbg / f"{frame_name}_render_w_combined.png", color=(0, 1, 1))
        if gripper_mask is not None:
            self._save_overlay(gt_rgb, gripper_mask, dbg / f"{frame_name}_live_w_gripper.png", color=(0, 1, 0))
        self._save_image(gt_rgb, dbg / f"{frame_name}_live.png")
        self._save_image(rdn_rgb, dbg / f"{frame_name}_rdn.png")
        self._save_image(cdn, dbg / f"{frame_name}_cdn.png")
        self._timing["DN.8_debug_images"].append(time.time() - t0)

        self.model._set_optim_mask(cdn)
        self.model._dynamic_ready = True

        self._timing["DN.9_total_frame_n"].append(time.time() - t_total)

        change_px = int((cdn[..., 0] > 0.5).sum().item()) if cdn.ndim >= 3 else int((cdn > 0.5).sum().item())
        CONSOLE.log(
            f"[dynamic-gs] frame {self.current_dynamic_frame_idx} ({frame_name}): "
            f"change px={change_px}, "
            f"object flags={int((self.model.object_flags.squeeze(-1) > 0.5).sum().item())}"
        )
        CONSOLE.log(
            f"[timing] frame {self.current_dynamic_frame_idx}: "
            f"total={self._timing['DN.9_total_frame_n'][-1]:.3f}s, "
            f"sam2={self._timing['DN.1_sam2_live_propagation'][-1]:.3f}s, "
            f"cotracker={self._timing.get('DN.3_cotracker_advance', [0])[-1]:.3f}s, "
            f"render={self._timing['DN.5_render_rdn'][-1]:.3f}s, "
            f"obj_mask={self._timing['DN.6_render_object_mask'][-1]:.3f}s, "
            f"change={self._timing['DN.7_change_mask_cdn'][-1]:.3f}s, "
            f"debug={self._timing['DN.8_debug_images'][-1]:.3f}s"
        )

    # ---- Phase sync and training loop ----

    def _sync_phase(self, step: int) -> None:
        phase = self._phase_for_step(step)
        phase_changed = phase != self.current_phase

        if phase_changed:
            self.current_phase = phase
            self.datamanager.set_phase(phase)
            self.model.set_phase(phase, reset_means_optimizer=phase == "dynamic")
            if phase == "dynamic":
                self._reset_dynamic_segmentation_state()
            CONSOLE.log(f"[dynamic-gs] phase -> {phase} at step {step}")

        if phase == "static":
            self.current_dynamic_frame_idx = None
            return

        frame_idx = self._dynamic_frame_for_step(step)
        if frame_idx != self.current_dynamic_frame_idx:
            self.current_dynamic_frame_idx = frame_idx
            self.datamanager.set_dynamic_frame_idx(frame_idx)
            self._prepare_dynamic_frame()

    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes):
        callbacks = super().get_training_callbacks(training_callback_attributes)
        trainer = training_callback_attributes.trainer
        if trainer is not None:
            trainer.config.max_num_iterations = self._total_train_steps()
        return callbacks

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        t0 = time.time()
        self._sync_phase(step)
        result = super().get_train_loss_dict(step)
        elapsed = time.time() - t0
        phase_key = "static_step" if self.current_phase == "static" else "dynamic_step"
        self._timing[phase_key].append(elapsed)

        # Print summary at end of static phase
        if self.current_phase == "dynamic" and step == self.config.static_num_steps:
            s = self._timing["static_step"]
            if s:
                CONSOLE.log(
                    f"[timing] === STATIC PHASE SUMMARY ===\n"
                    f"  steps: {len(s)}, total: {sum(s):.1f}s, avg: {sum(s)/len(s)*1000:.1f}ms/step"
                )
        # Write full report at the very last step
        total_steps = self._total_train_steps()
        if step == total_steps - 1:
            self._print_timing_summary()
            self._write_timing_report()

        return result

    def _print_timing_summary(self):
        """Print a concise timing summary to the console log."""
        CONSOLE.log("[timing] === FULL PIPELINE SUMMARY ===")
        s = self._timing["static_step"]
        if s:
            CONSOLE.log(f"  Static phase: {len(s)} steps, {sum(s):.1f}s total, {sum(s)/len(s)*1000:.1f}ms/step avg")
        for key in sorted(k for k in self._timing if k.startswith("D0.")):
            vals = self._timing[key]
            CONSOLE.log(f"  {key}: {sum(vals):.2f}s")
        for key in sorted(k for k in self._timing if k.startswith("DN.")):
            vals = self._timing[key]
            if vals:
                CONSOLE.log(f"  {key}: avg={sum(vals)/len(vals)*1000:.1f}ms, total={sum(vals):.2f}s ({len(vals)} frames)")
        d = self._timing["dynamic_step"]
        if d:
            CONSOLE.log(f"  Dynamic training: {len(d)} steps, {sum(d):.1f}s total, {sum(d)/len(d)*1000:.1f}ms/step avg")
        all_times = sum(sum(v) for v in self._timing.values())
        CONSOLE.log(f"  Grand total measured: {all_times:.1f}s")

    def _write_timing_report(self):
        """Write a detailed timing report file with per-phase breakdowns and percentages."""
        from datetime import datetime

        # --- Descriptions for each timer key (chronological within phase) ---
        d0_keys = [
            ("D0.1_initial_change_detection", "Initial change detection (total)"),
            ("D0.1a_forward_render", "  -> Forward render (get_outputs in eval mode)"),
            ("D0.1b_change_mask", "  -> Change mask (MSSIM depth+RGB, morphological filtering)"),
            ("D0.1c_esam_render", "  -> ESAM on render (includes model load on first call)"),
            ("D0.1d_esam_live", "  -> ESAM on live image"),
            ("D0.1e_gaussian_flagging", "  -> Gaussian flagging (project centers, build active mask)"),
            ("D0.2_sam3d_generation", "SAM3D object generation (subprocess)"),
            ("D0.3_sam3d_insertion", "SAM3D object insertion (CPD similarity + dedup + insert Gaussians)"),
            ("D0.4_render_object_mask", "Render object mask (rasterize object_flags Gaussians from simulation)"),
            ("D0.5_seed_live_tracker", "Seed live SAM2 tracker (store D0 live RGB + mask)"),
            ("D0.6_cotracker_init", "CoTracker initialization (freeze D0 reference frame, sample points once, cache reference 3D)"),
            ("D0.7_render_rs00", "Render RS00 (re-render scene after SAM3D insertion)"),
            ("D0.8_change_mask_cd0", "Change mask CD0 (MSSIM RS00 vs D0, excluding gripper + object)"),
            ("D0.9_debug_images", "Debug images (save overlay PNGs to disk)"),
        ]
        dn_keys = [
            ("DN.1_sam2_live_propagation", "SAM2 live mask propagation (D(N-1) -> DN on live images)"),
            ("DN.2_cotracker_refresh", "CoTracker reference mode (no reseed; D0 query points stay fixed)"),
            ("DN.3_cotracker_advance", "CoTracker absolute rigid transform (D0 -> DN, RANSAC, apply absolute SE(3))"),
            ("DN.4_cotracker_filter", "CoTracker post-filter skipped in reference mode"),
            ("DN.5_render_rdn", "Render RDN (render scene after rigid transform)"),
            ("DN.6_render_object_mask", "Render object mask (rasterize object_flags Gaussians from simulation)"),
            ("DN.7_change_mask_cdn", "Change mask CDN (MSSIM RDN vs DN, excluding gripper + union object mask)"),
            ("DN.8_debug_images", "Debug images (save overlay PNGs to disk)"),
        ]

        lines = []
        lines.append("=== PIPELINE TIMING REPORT ===")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(
            f"Config: static_num_steps={self.config.static_num_steps}, "
            f"dynamic_steps_per_frame={self.config.dynamic_steps_per_frame}, "
            f"total_dynamic_frames={self.total_dynamic_frames}"
        )
        lines.append("")

        # --- Phase 1: Static ---
        s = self._timing["static_step"]
        static_total = sum(s) if s else 0.0
        lines.append("--- PHASE 1: STATIC TRAINING ---")
        lines.append(f"Phase total: {static_total:.1f}s")
        lines.append("")
        if s:
            avg_ms = static_total / len(s) * 1000
            lines.append(f"  S.1  Full training step (avg over {len(s)} steps)  {avg_ms:>10.1f}ms  {100.0:>6.1f}%")
        lines.append("")

        # --- Phase 2: Dynamic initialization (Frame 0) ---
        d0_total_vals = self._timing.get("D0.10_total_frame_0", [])
        d0_phase_total = sum(d0_total_vals) if d0_total_vals else 0.0
        lines.append("--- PHASE 2: DYNAMIC INITIALIZATION (Frame 0) ---")
        lines.append(f"Phase total: {d0_phase_total:.1f}s")
        lines.append("")
        for key, desc in d0_keys:
            vals = self._timing.get(key, [])
            t = sum(vals) if vals else 0.0
            pct = (t / d0_phase_total * 100) if d0_phase_total > 0 else 0.0
            lines.append(f"  {key:<42s} {t:>8.2f}s  {pct:>6.1f}%    {desc}")
        lines.append("")

        # --- Phase 3: Dynamic loop ---
        dn_total_vals = self._timing.get("DN.9_total_frame_n", [])
        frame_prep_total = sum(dn_total_vals) if dn_total_vals else 0.0
        n_frames = len(dn_total_vals)
        d = self._timing["dynamic_step"]
        dyn_train_total = sum(d) if d else 0.0
        dyn_phase_total = frame_prep_total + dyn_train_total

        lines.append(f"--- PHASE 3: DYNAMIC LOOP (Frames 1-{self.total_dynamic_frames - 1}) ---")
        lines.append(f"Phase total: {dyn_phase_total:.1f}s")
        lines.append(f"  Frame prep total: {frame_prep_total:.1f}s")
        lines.append(f"  Training total: {dyn_train_total:.1f}s")
        lines.append("")

        lines.append(f"  [Per-frame prep averages over {n_frames} frames]")
        avg_frame_total = (frame_prep_total / n_frames) if n_frames > 0 else 0.0
        for key, desc in dn_keys:
            vals = self._timing.get(key, [])
            if vals:
                avg_ms = sum(vals) / len(vals) * 1000
                pct = (avg_ms / (avg_frame_total * 1000) * 100) if avg_frame_total > 0 else 0.0
                lines.append(f"  {key:<42s} {avg_ms:>8.1f}ms  {pct:>6.1f}%    {desc}")
            else:
                lines.append(f"  {key:<42s}      N/A     N/A    {desc}")
        lines.append("")

        lines.append(f"  [Per-epoch training average over {len(d)} steps]")
        if d:
            avg_dyn_ms = dyn_train_total / len(d) * 1000
            lines.append(f"  {'DT.1 dynamic_step':<42s} {avg_dyn_ms:>8.1f}ms  {100.0:>6.1f}%    Full training iteration (masked loss + backward + optimizer)")
        lines.append("")

        # --- Grand total ---
        grand_total = static_total + d0_phase_total + dyn_phase_total
        lines.append("--- GRAND TOTAL ---")
        if grand_total > 0:
            lines.append(f"  Static phase:           {static_total:>8.1f}s  {static_total/grand_total*100:>6.1f}%")
            lines.append(f"  Dynamic initialization: {d0_phase_total:>8.1f}s  {d0_phase_total/grand_total*100:>6.1f}%")
            lines.append(f"  Dynamic loop:           {dyn_phase_total:>8.1f}s  {dyn_phase_total/grand_total*100:>6.1f}%")
            lines.append(f"    Frame prep subtotal:  {frame_prep_total:>8.1f}s  {frame_prep_total/grand_total*100:>6.1f}%")
            lines.append(f"    Training subtotal:    {dyn_train_total:>8.1f}s  {dyn_train_total/grand_total*100:>6.1f}%")
            lines.append(f"  Pipeline total:         {grand_total:>8.1f}s")
        lines.append("")

        report_text = "\n".join(lines)

        # Write to data root (same level as CLAUDE.md equivalent for the data)
        data_root = Path(self.datamanager.config.data)
        report_path = data_root / "timing_report.txt"
        report_path.write_text(report_text)
        CONSOLE.log(f"[timing] Report written to {report_path}")

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        self._sync_phase(step)
        return super().get_eval_loss_dict(step)

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        self._sync_phase(step)
        return super().get_eval_image_metrics_and_images(step)

    @profiler.time_function
    def get_average_eval_image_metrics(self, step=None, output_path=None, get_std=False):
        if step is not None:
            self._sync_phase(step)
        return super().get_average_eval_image_metrics(step=step, output_path=output_path, get_std=get_std)
