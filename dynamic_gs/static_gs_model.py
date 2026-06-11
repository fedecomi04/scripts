"""Stripped Splatfacto model for ``static-gs``.

What's here:

* The four persistent Gaussian-identity buffers (``object_flags``,
  ``object_instance_ids``, ``sam3d_init_target_flags``, ``inserted_flags``).
  Shapes and names match ``DynamicGSModel`` so a future dynamic-gs warm
  restart can load a static-gs cache.
* Insert / delete helpers that Phase 0b fusion calls, plus their internal
  buffer-resize + optimizer-refresh machinery.
* The two Gaussian-subset helpers Phase 0b uses to find existing object
  Gaussians under a 2D mask (``_get_existing_object_subset``,
  ``_get_object_mask_slab_indices``) and the spacing estimator they need.
* Simulator-background override + the ``NoRefineStrategy`` registration
  that keeps Splatfacto from densifying / pruning during static training.

What's intentionally not here (vs ``DynamicGSModel``):

* ``set_phase`` / ``_apply_phase_trainability`` / ``_apply_phase_optimizers``
  — there is no dynamic phase.
* ``_mask_means_grad`` and the scene-opt gradient hooks — no gradient
  masking, no per-Gaussian scene-opt gate.
* ``current_active_mask``, ``change_mask_image`` (both ``persistent=False``
  so they were never in the saved state_dict anyway — dropping the
  registrations just removes attribute clutter).
* ``render_object_mask``, ``apply_rigid_object_transform_from_reference``,
  ``capture_reference_object_pose``, ``prepare_dynamic_update``,
  ``initialize_object_from_sam3d``, ESAM lazy load, bilateral grid handling,
  the customised ``get_outputs`` with the three subset renders, the
  customised ``get_loss_dict`` with the masked dynamic path, and
  ``step_post_backward`` overrides.
* All tracker / feedforward / scene-opt config fields. The kept config
  surface is just what Splatfacto needs + what Phase 0a/0b touch.

``StaticGSModel`` does NOT subclass ``DynamicGSModel`` — it goes straight
to ``SplatfactoModel`` so the dynamic-phase code paths can't be reached
even by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type

import numpy as np
import torch
from torch import Tensor

from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
from nerfstudio.utils.math import k_nearest_sklearn
from nerfstudio.utils.spherical_harmonics import RGB2SH

from .utils import (
    NoRefineStrategy,
    extract_projected_centers_and_radii,
)


@dataclass
class StaticGSModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: StaticGSModel)

    # ---- Render background ----
    use_simulator_background: bool = True
    simulator_background_rgb: Tuple[float, float, float] = (0.86, 0.92, 1.0)

    # ---- Splatfacto schedule overrides (kept identical to dynamic-gs so
    # the trained scene is byte-comparable). ----
    output_depth_during_training: bool = True
    sh_degree_interval: int = 500
    resolution_schedule: int = 100

    # ---- Phase 0a (SAM3 text-prompted segmentation) ----
    use_sam3_graspable_prefusion: bool = True
    sam3_prompt_text: str = "the can of coke on the table"
    sam3_conda_env_name: str = "sam3_dynamic_gs"
    sam3_candidate_min_area_ratio: float = 0.002
    sam3_candidate_max_area_ratio: float = 0.25
    sam3_candidate_dedup_iou: float = 0.6
    sam3_candidate_max_objects: int = 8
    sam3_confidence_threshold: float = 0.3
    sam3_min_score: float = 0.2
    sam3_reuse_cached: bool = True

    # ---- Segmentation backend (SAM3 vs FastSAM) ----
    segmentation_backend: Literal["sam3", "fastsam"] = "fastsam"
    """Which text-prompted segmenter feeds Phase 0a. ``fastsam`` (DEFAULT) =
    FastSAM-x + CLIP, ~0.85 GB resident vs SAM3's ~3.8 GB — light enough to
    co-reside with SAM3D (~12 GB) so SAM3D can load from the start. ``sam3`` =
    the original Meta SAM3 grounding model (heavier, slightly tighter masks).
    Quality gate (screwdriver, recording_15fps): top-1 IoU 0.79 fastsam-vs-sam3.
    Both run in the ``sam3_dynamic_gs`` env via SamWorkerClient."""
    fastsam_weights: str = "FastSAM-x.pt"
    fastsam_clip_model: str = "ViT-B-32-quickgelu"
    fastsam_clip_pretrained: str = "openai"
    fastsam_conf: float = 0.4
    fastsam_iou: float = 0.9

    # ---- Phase 0a (Fast-SAM3D 3D generation) + Phase 0b (registration) ----
    use_sam3d_object_init: bool = True
    reuse_sam3d_generated_ply: bool = True
    sam3d_registration_backend: Literal["ndp", "cpd", "teaser"] = "ndp"
    """``ndp`` = Neural Deformation Pyramid non-rigid registration (DEFAULT).
    Reuses the rigid init (SAM3D-rotation + bbox-scale + centroid) then NON-RIGIDLY
    deforms the complete SAM3D cloud onto the accurate partial target with a
    hierarchical Sim3 deformation pyramid (no learned weights, GPU, in-process via
    ``dynamic_gs/utils/ndp_register.py``). Conforms the approximate model to the
    real geometry far better than a single rigid+scale fit.
    ``cpd`` = legacy probreg coherent point drift (rigid+scale, robust, slow).
    ``teaser`` = the "v13" 3-stage recipe (FPFH+TEASER -> Euclidean-NN reproject
    +TEASER -> point-to-plane ICP), rigid+scale. Fast; requires ``teaserpp_python``.
    Keep this block byte-for-byte in sync with DynamicGSModelConfig — both feed
    the same ``register_and_fuse_sam3d_object`` via run_phase0b_fusion(model)."""
    sam3d_teaser_noise_bound: float = 0.02
    sam3d_teaser_max_correspondences: int = 5000
    sam3d_teaser_fpfh_normal_radius_mult: float = 2.0
    sam3d_teaser_fpfh_feature_radius_mult: float = 5.0
    # 0.0: SAM3D-decoded colors disagree with the real-camera target; any color
    # weight degraded matching in benchmarks.
    sam3d_teaser_color_weight: float = 0.0
    # fpfh_max_nn=500 unlocks v5+ quality (default 100 silently caps the radius).
    sam3d_teaser_fpfh_max_nn: int = 500
    sam3d_teaser_normal_max_nn: int = 30
    # Stage 2 — Euclidean-NN reproject + TEASER (biggest single quality win).
    sam3d_teaser_enable_reproject: bool = True
    sam3d_teaser_reproject_max_corr_mult: float = 3.0
    sam3d_teaser_reproject_noise_bound: float = 0.005
    # Stage 3 — point-to-plane ICP polish.
    sam3d_teaser_enable_post_icp: bool = True
    sam3d_teaser_post_icp_max_corr_mult: float = 2.0
    sam3d_teaser_post_icp_iterations: int = 50

    # ---- Change-mask knobs (for a future static-convergence early-exit;
    # not consumed yet — kept so the StaticGSModelConfig surface matches
    # what the convergence check will need). ----
    change_mask_depth_threshold: float = 0.02
    change_mask_rgb_threshold: float = 0.07
    change_mask_use_rgb: bool = False
    change_mask_blur_kernel_size: int = 5
    change_mask_blur_sigma: float = 1.0
    change_mask_filter_radius: int = 1
    change_mask_min_component_size: int = 64


class StaticGSModel(SplatfactoModel):
    config: StaticGSModelConfig

    def __init__(self, config, metadata=None, **kwargs):
        self._optimizers_wrapper = None
        super().__init__(config=config, metadata=metadata, **kwargs)

    # ------------------------------------------------------------------
    # Module setup
    # ------------------------------------------------------------------

    def populate_modules(self):
        super().populate_modules()

        if self.config.use_simulator_background:
            self.set_background(
                torch.tensor(
                    self.config.simulator_background_rgb,
                    device=self.means.device,
                    dtype=torch.float32,
                )
            )

        num_points = self.num_points
        # Persistent identity buffers — shape + name match DynamicGSModel
        # so a static-gs cache loads cleanly into the future dynamic-gs.
        self.register_buffer(
            "object_flags",
            torch.zeros(num_points, 1, dtype=self.means.dtype, device=self.means.device),
            persistent=True,
        )
        self.register_buffer(
            "sam3d_init_target_flags",
            torch.zeros(num_points, 1, dtype=self.means.dtype, device=self.means.device),
            persistent=True,
        )
        self.register_buffer(
            "object_instance_ids",
            torch.zeros(num_points, 1, dtype=torch.long, device=self.means.device),
            persistent=True,
        )
        self.register_buffer(
            "inserted_flags",
            torch.zeros(num_points, 1, dtype=self.means.dtype, device=self.means.device),
            persistent=True,
        )

        # Disable Splatfacto's default densify/prune strategy for the
        # static phase. Phase 0b mutates Gaussian count surgically, not
        # through densification.
        self.strategy = NoRefineStrategy()
        self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)

    def _get_background_color(self):
        if self.config.use_simulator_background:
            return self.background_color.to(self.means.device)
        return super()._get_background_color()

    def step_cb(self, optimizers, step):
        super().step_cb(optimizers, step)
        self._optimizers_wrapper = optimizers

    def step_post_backward(self, step):  # type: ignore[override]
        """No-op: ``NoRefineStrategy`` means no densify/prune. Splatfacto's
        parent ``step_post_backward`` raises on any non-``DefaultStrategy`` /
        non-``MCMCStrategy``, so we must override to bypass it."""
        return

    # ------------------------------------------------------------------
    # State-dict load (warm-cache / resume support)
    # ------------------------------------------------------------------

    def load_state_dict(self, state_dict, **kwargs):  # type: ignore[override]
        """Reshape the four persistent buffers to match the saved Gaussian
        count before delegating to ``super().load_state_dict``. This lets
        a static-gs run resume from a snapshot whose ``num_points`` differs
        from the cold-start SfM seed.
        """
        state_dict = state_dict.copy()
        if "gauss_params.means" in state_dict:
            num_points = state_dict["gauss_params.means"].shape[0]
        elif "means" in state_dict:
            num_points = state_dict["means"].shape[0]
        else:
            num_points = self.num_points

        if self.object_flags.shape[0] != num_points:
            self._buffers["object_flags"] = torch.zeros(
                num_points, 1,
                dtype=self.object_flags.dtype, device=self.object_flags.device,
            )
            self._buffers["sam3d_init_target_flags"] = torch.zeros(
                num_points, 1,
                dtype=self.sam3d_init_target_flags.dtype,
                device=self.sam3d_init_target_flags.device,
            )
            self._buffers["object_instance_ids"] = torch.zeros(
                num_points, 1,
                dtype=torch.long, device=self.object_instance_ids.device,
            )
            self._buffers["inserted_flags"] = torch.zeros(
                num_points, 1,
                dtype=self.inserted_flags.dtype, device=self.inserted_flags.device,
            )

        for key, buf in (
            ("object_flags", self.object_flags),
            ("sam3d_init_target_flags", self.sam3d_init_target_flags),
            ("object_instance_ids", self.object_instance_ids),
            ("inserted_flags", self.inserted_flags),
        ):
            if key not in state_dict:
                state_dict[key] = torch.zeros_like(buf)

        super().load_state_dict(state_dict, **kwargs)

    # ------------------------------------------------------------------
    # Insert / delete machinery (called by Phase 0b)
    # ------------------------------------------------------------------

    def _resize_dynamic_buffers(self, num_points: int) -> None:
        """Resize the four identity buffers in lockstep with a
        ``gauss_params`` insert/delete. Preserves the leading entries."""
        if all(
            b.shape[0] == num_points
            for b in (
                self.object_flags,
                self.sam3d_init_target_flags,
                self.object_instance_ids,
                self.inserted_flags,
            )
        ):
            return

        def _resize(old: Tensor, *, long: bool = False) -> Tensor:
            new = torch.zeros(
                num_points, 1,
                dtype=torch.long if long else old.dtype,
                device=old.device,
            )
            keep = min(old.shape[0], num_points)
            if keep > 0:
                new[:keep] = old[:keep]
            return new

        self._buffers["object_flags"] = _resize(self.object_flags)
        self._buffers["sam3d_init_target_flags"] = _resize(self.sam3d_init_target_flags)
        self._buffers["object_instance_ids"] = _resize(self.object_instance_ids, long=True)
        self._buffers["inserted_flags"] = _resize(self.inserted_flags)

    def _refresh_gaussian_optimizers(self, reset_means_optimizer: bool) -> None:
        """Re-point each optimizer at the freshly-allocated ``gauss_params``
        Parameter after an insert/delete. ``reset_means_optimizer`` clears
        the ``means`` Adam state too; the others always get cleared because
        their ``m``/``v`` shapes no longer match the new tensor."""
        if not hasattr(self, "optimizers"):
            return
        for name, optimizer in self.optimizers.items():
            if name not in self.gauss_params:
                continue
            optimizer.param_groups[0]["params"] = [self.gauss_params[name]]
            optimizer.state.clear()
        if self._optimizers_wrapper is not None:
            for name in self.gauss_params:
                if name in self._optimizers_wrapper.parameters:
                    self._optimizers_wrapper.parameters[name] = [self.gauss_params[name]]

    def _build_new_gaussian_tensors(
        self, new_xyz: Tensor, new_rgb: Tensor
    ) -> Dict[str, Tensor]:
        """Default per-Gaussian attribute tensors for a batch of new points.
        Scales seed from k-NN spacing; opacity = 0.1 (pre-sigmoid)."""
        device = self.means.device
        dtype = self.means.dtype
        new_xyz = new_xyz.to(device=device, dtype=dtype)
        new_rgb = new_rgb.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        num_new = new_xyz.shape[0]
        dim_sh = self.features_rest.shape[1] + 1

        if num_new > 1:
            neighbor_k = min(3, num_new - 1)
            distances, _ = k_nearest_sklearn(new_xyz.detach().cpu(), neighbor_k)
            avg_dist = distances.mean(dim=-1, keepdim=True).to(device=device, dtype=dtype)
        else:
            avg_dist = torch.full((num_new, 1), 1e-3, device=device, dtype=dtype)
        avg_dist = avg_dist.clamp_min(1e-6)

        if self.config.sh_degree > 0:
            features_dc = RGB2SH(new_rgb)
        else:
            features_dc = torch.logit(new_rgb, eps=1e-10)
        features_rest = torch.zeros((num_new, dim_sh - 1, 3), device=device, dtype=dtype)
        scales = torch.log(avg_dist.repeat(1, 3))
        quats = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype).repeat(num_new, 1)
        opacities = torch.logit(torch.full((num_new, 1), 0.1, device=device, dtype=dtype))
        return {
            "means": new_xyz,
            "features_dc": features_dc,
            "features_rest": features_rest,
            "scales": scales,
            "quats": quats,
            "opacities": opacities,
        }

    @torch.no_grad()
    def insert_object_gaussians(
        self,
        new_xyz: Tensor,
        new_rgb: Tensor,
        object_flag: bool = True,
        instance_id: int = 0,
    ) -> Tensor:
        """Concatenate ``new_xyz`` / ``new_rgb`` Gaussians onto the scene,
        write the identity flags for the inserted range, and refresh the
        optimizers. Returns the inserted index range.

        Used by Phase 0b after CPD/TEASER++ registration:
        ``insert_object_gaussians(culled_pts, culled_colors,
        object_flag=False, instance_id=k)`` writes ``object_instance_ids=k``
        and ``inserted_flags=1`` for the new Gaussians; ``object_flags``
        stays 0 (D0 selection is the dynamic-gs pipeline's job, not ours).
        """
        num_new = new_xyz.shape[0]
        if num_new == 0:
            return torch.zeros((0,), dtype=torch.long, device=self.means.device)

        new_tensors = self._build_new_gaussian_tensors(new_xyz, new_rgb)
        old_num_points = self.num_points
        for name in ["means", "features_dc", "features_rest", "scales", "quats", "opacities"]:
            concatenated = torch.cat(
                [self.gauss_params[name].detach(), new_tensors[name]], dim=0
            )
            self.gauss_params[name] = torch.nn.Parameter(concatenated)

        self._resize_dynamic_buffers(self.num_points)
        if object_flag:
            self.object_flags[old_num_points:] = 1.0
        if instance_id > 0:
            self.object_instance_ids[old_num_points:] = instance_id
        self.inserted_flags[old_num_points:] = 1.0
        self._refresh_gaussian_optimizers(reset_means_optimizer=True)
        return torch.arange(
            old_num_points, self.num_points,
            device=self.means.device, dtype=torch.long,
        )

    @torch.no_grad()
    def delete_gaussian_indices(self, indices: Tensor) -> int:
        """Prune Gaussians at ``indices``. Resizes both ``gauss_params``
        and the four identity buffers, then refreshes optimizers. Returns
        the count actually deleted (after dedup + bounds clipping)."""
        if indices is None or indices.numel() == 0:
            return 0
        device = self.means.device
        num_points = self.num_points
        indices = indices.to(device=device, dtype=torch.long).unique()
        valid = (indices >= 0) & (indices < num_points)
        indices = indices[valid]
        if indices.numel() == 0:
            return 0

        keep = torch.ones(num_points, dtype=torch.bool, device=device)
        keep[indices] = False
        n_deleted = int((~keep).sum().item())

        for name in ["means", "features_dc", "features_rest", "scales", "quats", "opacities"]:
            sliced = self.gauss_params[name].detach()[keep]
            self.gauss_params[name] = torch.nn.Parameter(sliced)

        self._buffers["object_flags"] = self.object_flags[keep]
        self._buffers["sam3d_init_target_flags"] = self.sam3d_init_target_flags[keep]
        self._buffers["object_instance_ids"] = self.object_instance_ids[keep]
        self._buffers["inserted_flags"] = self.inserted_flags[keep]

        self._refresh_gaussian_optimizers(reset_means_optimizer=True)
        return n_deleted

    # ------------------------------------------------------------------
    # Gaussian-subset queries (Phase 0b CPD target + cull/flag slab)
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_spacing(points: np.ndarray, max_samples: int = 50_000) -> float:
        """Median of the per-point mean k-NN distance (k=3). Cheap proxy
        for "typical point spacing" — used by Phase 0b to set the cull
        and flag radii adaptively to the local point density."""
        if len(points) <= 1:
            return 1e-3
        if len(points) > max_samples:
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(len(points), size=max_samples, replace=False)
            points = points[sample_idx]
        neighbor_k = min(3, len(points) - 1)
        distances, _ = k_nearest_sklearn(
            torch.from_numpy(points.astype(np.float32)), neighbor_k
        )
        return float(distances.mean(dim=-1).median().item())

    def _get_object_mask_slab_indices(
        self,
        render_object_mask: Tensor,
        rendered_depth: Tensor,
        depth_tol_m: float = 0.01,
    ) -> Tensor:
        """All Gaussian indices whose projected center is inside the 2D
        object mask AND whose projected depth is within ``depth_tol_m`` of
        the rendered front-surface depth. Phase 0b uses this twice — once
        with a tight tolerance to define the SAM3D-cull set, once with a
        looser tolerance to flag deeper existing object Gaussians."""
        centers_2d, radii = extract_projected_centers_and_radii(self.info, self.num_points)
        mask = render_object_mask[..., 0] if render_object_mask.ndim == 3 else render_object_mask
        depth_image = rendered_depth[..., 0] if rendered_depth.ndim == 3 else rendered_depth
        height, width = mask.shape

        projected_depths = self.info.get("depths")
        if projected_depths is None:
            return torch.zeros((0,), dtype=torch.long, device=self.means.device)
        if projected_depths.ndim > 1:
            projected_depths = projected_depths.reshape(-1)
        projected_depths = projected_depths.float()

        cx = torch.round(centers_2d[:, 0]).long()
        cy = torch.round(centers_2d[:, 1]).long()
        in_bounds = (
            torch.isfinite(centers_2d).all(dim=-1)
            & torch.isfinite(radii)
            & torch.isfinite(projected_depths)
            & (radii > 0)
            & (cx >= 0) & (cx < width)
            & (cy >= 0) & (cy < height)
        )
        idx = torch.nonzero(in_bounds, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return idx

        in_mask = mask[cy[idx], cx[idx]] > 0.5
        sampled = depth_image[cy[idx], cx[idx]]
        near_surface = torch.isfinite(sampled) & (
            (projected_depths[idx] - sampled).abs() <= float(depth_tol_m)
        )
        return idx[in_mask & near_surface]

    def _get_existing_object_subset(
        self,
        render_object_mask: Tensor,
        rendered_depth: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Phase 0b's CPD registration target. Stricter than the slab
        helper: keeps only the frontmost Gaussian per masked pixel,
        thins by depth tolerance, and downsamples to roughly half the
        survivor count. Returns ``(indices, means, colors)``.
        """
        centers_2d, radii = extract_projected_centers_and_radii(self.info, self.num_points)
        mask = render_object_mask[..., 0] if render_object_mask.ndim == 3 else render_object_mask
        depth_image = rendered_depth[..., 0] if rendered_depth.ndim == 3 else rendered_depth
        height, width = mask.shape

        projected_depths = self.info.get("depths")
        if projected_depths is None:
            raise RuntimeError(
                "SAM3D initialization requires projected Gaussian depths in rasterization info."
            )
        if projected_depths.ndim > 1:
            projected_depths = projected_depths.reshape(-1)
        projected_depths = projected_depths.float()

        center_x = torch.round(centers_2d[:, 0]).long()
        center_y = torch.round(centers_2d[:, 1]).long()
        candidate_mask = (
            torch.isfinite(centers_2d).all(dim=-1)
            & torch.isfinite(radii)
            & torch.isfinite(projected_depths)
            & (radii > 0)
            & (center_x >= 0) & (center_x < width)
            & (center_y >= 0) & (center_y < height)
        )
        candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
        if candidate_indices.numel() > 0:
            candidate_mask[candidate_indices] &= (
                mask[center_y[candidate_indices], center_x[candidate_indices]] > 0.5
            )
            candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)

        if candidate_indices.numel() >= 2:
            pixel_ids = (
                center_y[candidate_indices] * width + center_x[candidate_indices]
            ).detach().cpu().numpy()
            candidate_depths = projected_depths[candidate_indices].detach().cpu().numpy()
            order = np.lexsort((candidate_depths, pixel_ids))
            sorted_indices = candidate_indices[
                torch.from_numpy(order).to(candidate_indices.device)
            ]
            sorted_pixel_ids = pixel_ids[order]
            keep = np.zeros(len(sorted_indices), dtype=bool)
            rank_in_pixel = 0
            # Keep only the frontmost Gaussian per masked pixel. Loosening
            # this leaks table/support geometry into the registration target.
            top_k_per_pixel = 1
            for idx in range(len(sorted_indices)):
                if idx == 0 or sorted_pixel_ids[idx] != sorted_pixel_ids[idx - 1]:
                    rank_in_pixel = 0
                else:
                    rank_in_pixel += 1
                keep[idx] = rank_in_pixel < top_k_per_pixel
            candidate_indices = sorted_indices[
                torch.from_numpy(keep).to(candidate_indices.device)
            ]

        if candidate_indices.numel() >= 3:
            sampled_depth = depth_image[
                center_y[candidate_indices], center_x[candidate_indices]
            ]
            candidate_count_before_depth = int(candidate_indices.numel())

            if candidate_indices.numel() > 1:
                nn_k = min(3, candidate_indices.numel() - 1)
                nn_distances, _ = k_nearest_sklearn(
                    self.means[candidate_indices].detach().cpu(), nn_k
                )
                target_spacing = float(nn_distances.mean(dim=-1).median().item())
            else:
                target_spacing = 1e-3

            depth_tolerance = max(0.008, 5.0 * target_spacing)
            desired_min_keep = max(3, int(0.50 * candidate_count_before_depth))
            best_visible = None
            best_visible_count = 0
            for multiplier in (1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
                current_visible = (
                    torch.isfinite(sampled_depth)
                    & (torch.abs(projected_depths[candidate_indices] - sampled_depth)
                       <= multiplier * depth_tolerance)
                )
                current_visible_count = int(current_visible.sum().item())
                if current_visible_count > best_visible_count:
                    best_visible = current_visible
                    best_visible_count = current_visible_count
                if current_visible_count >= desired_min_keep:
                    break
            if best_visible is not None and best_visible_count >= 3:
                candidate_indices = candidate_indices[best_visible]

        if candidate_indices.numel() >= 6:
            keep_count = max(3, candidate_indices.numel() // 2)
            keep_positions = torch.linspace(
                0, candidate_indices.numel() - 1, steps=keep_count,
                device=candidate_indices.device,
            )
            keep_positions = torch.round(keep_positions).long().unique(sorted=True)
            candidate_indices = candidate_indices[keep_positions]

        existing_means = self.means[candidate_indices].detach()
        existing_colors = self.colors[candidate_indices].detach()
        return candidate_indices, existing_means, existing_colors
