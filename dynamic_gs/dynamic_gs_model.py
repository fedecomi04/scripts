from __future__ import annotations

import copy
import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig, get_viewmat
from nerfstudio.utils.math import k_nearest_sklearn
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.spherical_harmonics import RGB2SH

from .utils import (
    ESAM_NUM_PROMPT_POINTS,
    NoRefineStrategy,
    Sam3DInsertionResult,
    build_active_mask,
    build_change_mask,
    build_esam_ti,
    build_sam2_tiny_video_predictor,
    combine_object_masks,
    dilate_binary_mask,
    extract_projected_centers_and_radii,
    load_sam3d_gaussian_ply,
    masked_l1_depth_loss,
    get_sam3d_output_paths,
    prepare_cropped_sam3d_inputs,
    query_esam_mask,
    query_sam2_propagated_mask,
    register_and_fuse_sam3d_object,
    rigid_or_static_loss,
    run_sam3d_single_object_subprocess,
    save_point_cloud,
)

try:
    from gsplat.rendering import rasterization
except ImportError as exc:  # pragma: no cover - import-time dependency guard
    raise ImportError("dynamic-gs requires gsplat>=1.0.0") from exc


@dataclass
class DynamicGSModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: DynamicGSModel)

    depth_lambda: float = 0.4
    active_mask_dilate_radius: int = 0
    output_depth_during_training: bool = True

    change_mask_depth_threshold: float = 0.02
    change_mask_rgb_threshold: float = 0.15
    change_mask_use_rgb: bool = False
    change_mask_blur_kernel_size: int = 5
    change_mask_blur_sigma: float = 1.0
    change_mask_filter_radius: int = 1
    change_mask_min_component_size: int = 64
    rigid_static_lambda: float = 0.1
    rigid_inlier_threshold: float = 1e-4
    use_sam3d_object_init: bool = True
    reuse_sam3d_generated_ply: bool = True
    enable_dynamic_mean_optimization: bool = False
    enable_cotracker_rigid_motion: bool = True
    cotracker_query_point_count: int = 256
    cotracker_min_track_points: int = 12
    cotracker_ransac_iterations: int = 128
    cotracker_ransac_inlier_threshold: float = 0.008
    cotracker_point_refresh_min_distance: float = 8.0
    cotracker_checkpoint_path: str = ""
    cotracker_hub_repo: str = "facebookresearch/co-tracker"
    cotracker_hub_model: str = "cotracker3_offline"
    enable_scene_optimization: bool = True
    scene_opt_refine_every: int = 100
    scene_opt_densify_grad_thresh: float = 0.0002
    scene_opt_cull_alpha_thresh: float = 0.1
    object_mask_dilate_px: int = 1


class DynamicGSModel(SplatfactoModel):
    config: DynamicGSModelConfig

    def __init__(self, config, metadata=None, **kwargs):
        self.phase = "static"  # type: Literal["static", "dynamic"]
        self._base_lrs = {}  # type: Dict[str, float]
        self._initial_scheduler_states = {}  # type: Dict[str, Dict]
        self._dynamic_ready = False
        self._esam_model = None
        self._sam2_video_predictor = None
        self._reference_flagged_indices = None
        self._reference_flagged_means = None
        self._optimizers_wrapper = None
        self._persistent_object_membership_ready = False
        self._scene_opt_hooks = []
        self._grad2d_accum = None
        self._grad2d_count = None
        self._opt_step = 0
        super().__init__(config=config, metadata=metadata, **kwargs)

    def load_state_dict(self, state_dict, **kwargs):  # type: ignore[override]
        state_dict = state_dict.copy()
        if "gauss_params.means" in state_dict:
            num_points = state_dict["gauss_params.means"].shape[0]
        elif "means" in state_dict:
            num_points = state_dict["means"].shape[0]
        else:
            num_points = self.num_points

        if self.object_flags.shape[0] != num_points:
            self._buffers["object_flags"] = torch.zeros(
                num_points,
                1,
                dtype=self.object_flags.dtype,
                device=self.object_flags.device,
            )
            self._buffers["current_active_mask"] = torch.zeros(
                num_points,
                dtype=torch.bool,
                device=self.current_active_mask.device,
            )
            self._buffers["sam3d_init_target_flags"] = torch.zeros(
                num_points,
                1,
                dtype=self.object_flags.dtype,
                device=self.object_flags.device,
            )

        if "object_flags" not in state_dict:
            state_dict["object_flags"] = torch.zeros_like(self.object_flags)
        if "sam3d_init_target_flags" not in state_dict:
            state_dict["sam3d_init_target_flags"] = torch.zeros_like(self.sam3d_init_target_flags)

        super().load_state_dict(state_dict, **kwargs)

    def populate_modules(self):
        super().populate_modules()
        num_points = self.num_points
        self.register_buffer(
            "object_flags",
            torch.zeros(num_points, 1, dtype=self.means.dtype, device=self.means.device),
            persistent=True,
        )
        self.register_buffer(
            "current_active_mask",
            torch.zeros(num_points, dtype=torch.bool, device=self.means.device),
            persistent=False,
        )
        self.register_buffer(
            "sam3d_init_target_flags",
            torch.zeros(num_points, 1, dtype=self.means.dtype, device=self.means.device),
            persistent=True,
        )
        self.register_buffer(
            "change_mask_image",
            torch.zeros(1, 1, 1, dtype=self.means.dtype, device=self.means.device),
            persistent=False,
        )
        self.strategy = NoRefineStrategy()
        self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)
        self.gauss_params["means"].register_hook(self._mask_means_grad)
        self._apply_phase_trainability()

    def step_cb(self, optimizers, step):
        super().step_cb(optimizers, step)
        self._optimizers_wrapper = optimizers
        if not self._base_lrs:
            self._base_lrs = {
                name: optimizer.param_groups[0]["lr"] for name, optimizer in self.optimizers.items()
            }
        if not self._initial_scheduler_states:
            self._initial_scheduler_states = {
                name: copy.deepcopy(scheduler.state_dict()) for name, scheduler in self.schedulers.items()
            }
        self._apply_phase_trainability()
        self._apply_phase_optimizers(reset_means_optimizer=False)

    def get_gaussian_param_groups(self):
        return {
            "means": [self.gauss_params["means"]],
            "features_dc": [self.gauss_params["features_dc"]],
            "features_rest": [self.gauss_params["features_rest"]],
            "opacities": [self.gauss_params["opacities"]],
            "scales": [self.gauss_params["scales"]],
            "quats": [self.gauss_params["quats"]],
        }

    def set_phase(self, phase, reset_means_optimizer=False):
        self.phase = phase
        self._apply_phase_trainability()
        self._apply_phase_optimizers(reset_means_optimizer=reset_means_optimizer)
        if phase == "static":
            self._clear_scene_opt_hooks()
            self.reset_dynamic_state()
        elif phase == "dynamic" and self.config.enable_scene_optimization:
            self._register_scene_opt_hooks()

    def _apply_phase_trainability(self):
        if self.phase == "dynamic" and self.config.enable_scene_optimization:
            for name in ["means", "features_dc", "features_rest", "opacities", "scales", "quats"]:
                self.gauss_params[name].requires_grad_(True)
            return
        static_phase = self.phase == "static"
        self.gauss_params["means"].requires_grad_(not static_phase)
        for name in ["features_dc", "features_rest", "opacities", "scales", "quats"]:
            self.gauss_params[name].requires_grad_(static_phase)

    def _apply_phase_optimizers(self, reset_means_optimizer):
        if not hasattr(self, "optimizers"):
            return

        if self.phase == "dynamic" and self.config.enable_scene_optimization:
            active_groups = {"means", "features_dc", "features_rest", "opacities", "scales", "quats"}
        elif self.phase == "dynamic":
            active_groups = {"means"}
        else:
            active_groups = {"features_dc", "features_rest", "opacities", "scales", "quats"}
        for name, optimizer in self.optimizers.items():
            base_lr = self._base_lrs.get(name, optimizer.param_groups[0]["lr"])
            is_active = name in active_groups
            for group in optimizer.param_groups:
                group["lr"] = base_lr if is_active else 0.0
                group["initial_lr"] = base_lr
            if (not is_active) or (name == "means" and reset_means_optimizer):
                optimizer.state.clear()

        if reset_means_optimizer and "means" in self.schedulers and "means" in self._initial_scheduler_states:
            scheduler = self.schedulers["means"]
            scheduler.load_state_dict(copy.deepcopy(self._initial_scheduler_states["means"]))
            scheduler.base_lrs = [self._base_lrs["means"] for _ in scheduler.base_lrs]
            scheduler._last_lr = [self._base_lrs["means"] for _ in scheduler.base_lrs]

    def _mask_means_grad(self, grad):
        if self.phase != "dynamic":
            return grad
        if self.config.enable_scene_optimization:
            eligible = self._get_eligible_mask()
            if not eligible.any():
                return torch.zeros_like(grad)
            return grad * eligible.to(device=grad.device, dtype=grad.dtype).unsqueeze(-1)
        if self.config.enable_cotracker_rigid_motion or not self.config.enable_dynamic_mean_optimization:
            return torch.zeros_like(grad)
        mask = self.current_active_mask.to(device=grad.device, dtype=grad.dtype).unsqueeze(-1)
        return grad * mask

    def _set_optim_mask(self, mask):
        mask = mask.detach().float()
        if mask.ndim == 2:
            mask = mask[..., None]
        if self.change_mask_image.shape != mask.shape:
            self._buffers["change_mask_image"] = mask.clone()
        else:
            self.change_mask_image.copy_(mask)

    def _get_optim_mask(self, target_shape=None):
        if not self._dynamic_ready:
            return None
        mask = self.change_mask_image.to(self.device)
        if target_shape is not None:
            th, tw = target_shape[:2]
            if mask.shape[0] != th or mask.shape[1] != tw:
                mask = F.interpolate(
                    mask.permute(2, 0, 1).unsqueeze(0), size=(th, tw), mode="nearest",
                ).squeeze(0).permute(1, 2, 0)
        return mask

    # ---- Scene optimization: frame buffer + support count ----

    def _get_eligible_mask(self) -> Tensor:
        """All non-object Gaussians are eligible for scene optimization."""
        return ~(self.object_flags.squeeze(-1) > 0.5)

    def _make_scene_opt_grad_hook(self):
        def hook(grad):
            if grad is None:
                return grad
            eligible = self._get_eligible_mask()
            if not eligible.any():
                return torch.zeros_like(grad)
            return grad * eligible.to(dtype=grad.dtype).view(-1, *([1] * (grad.ndim - 1)))
        return hook

    def _register_scene_opt_hooks(self) -> None:
        self._clear_scene_opt_hooks()
        for name in ["features_dc", "features_rest", "opacities", "scales", "quats"]:
            param = self.gauss_params[name]
            if param.requires_grad:
                h = param.register_hook(self._make_scene_opt_grad_hook())
                self._scene_opt_hooks.append(h)

    def _clear_scene_opt_hooks(self) -> None:
        for h in self._scene_opt_hooks:
            h.remove()
        self._scene_opt_hooks.clear()

    def get_live_rgb(self, batch, background: Tensor | None = None, apply_training_downscale: bool = True) -> Tensor:
        if background is None:
            background = self._get_background_color()
        image = batch["image"]
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        if apply_training_downscale:
            image = self._downscale_if_required(image)
        image = image.to(self.device)
        return self.composite_with_background(image, background)

    @staticmethod
    def _normalize_quaternions(quats: Tensor) -> Tensor:
        return quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _quaternion_multiply(lhs: Tensor, rhs: Tensor) -> Tensor:
        lhs_w, lhs_x, lhs_y, lhs_z = lhs.unbind(dim=-1)
        rhs_w, rhs_x, rhs_y, rhs_z = rhs.unbind(dim=-1)
        return torch.stack(
            [
                lhs_w * rhs_w - lhs_x * rhs_x - lhs_y * rhs_y - lhs_z * rhs_z,
                lhs_w * rhs_x + lhs_x * rhs_w + lhs_y * rhs_z - lhs_z * rhs_y,
                lhs_w * rhs_y - lhs_x * rhs_z + lhs_y * rhs_w + lhs_z * rhs_x,
                lhs_w * rhs_z + lhs_x * rhs_y - lhs_y * rhs_x + lhs_z * rhs_w,
            ],
            dim=-1,
        )

    @staticmethod
    def _rotation_matrix_to_quaternion(rotation: Tensor) -> Tensor:
        if rotation.shape != (3, 3):
            raise ValueError(f"Expected 3x3 rotation matrix, got shape {tuple(rotation.shape)}")
        trace = rotation[0, 0] + rotation[1, 1] + rotation[2, 2]
        if trace > 0:
            scale = torch.sqrt(trace + 1.0) * 2.0
            quat = torch.stack([
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ])
        elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            scale = torch.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quat = torch.stack([
                (rotation[2, 1] - rotation[1, 2]) / scale,
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
            ])
        elif rotation[1, 1] > rotation[2, 2]:
            scale = torch.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quat = torch.stack([
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
            ])
        else:
            scale = torch.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quat = torch.stack([
                (rotation[1, 0] - rotation[0, 1]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
            ])
        return DynamicGSModel._normalize_quaternions(quat)

    @torch.no_grad()
    def apply_rigid_object_transform(self, rotation, translation) -> int:
        object_mask = self.object_flags.squeeze(-1) > 0.5
        object_count = int(object_mask.sum().item())
        if object_count == 0:
            return 0
        rotation_tensor = torch.as_tensor(rotation, device=self.means.device, dtype=self.means.dtype).reshape(3, 3)
        translation_tensor = torch.as_tensor(translation, device=self.means.device, dtype=self.means.dtype).reshape(3)
        if not torch.isfinite(rotation_tensor).all() or not torch.isfinite(translation_tensor).all():
            return 0
        transformed_means = self.means[object_mask] @ rotation_tensor.transpose(0, 1) + translation_tensor[None, :]
        delta_quat = self._rotation_matrix_to_quaternion(rotation_tensor).expand(object_count, -1)
        transformed_quats = self._normalize_quaternions(
            self._quaternion_multiply(delta_quat, self.quats[object_mask])
        )
        self.gauss_params["means"][object_mask] = transformed_means
        self.gauss_params["quats"][object_mask] = transformed_quats
        return object_count

    def _has_persistent_object_membership(self) -> bool:
        if self._persistent_object_membership_ready:
            return True
        return bool(
            torch.any(self.sam3d_init_target_flags > 0.5).item()
            and torch.any(self.object_flags > 0.5).item()
        )

    def reset_dynamic_state(self):
        if not self._has_persistent_object_membership():
            self.object_flags.zero_()
        self.current_active_mask.zero_()
        self.change_mask_image.zero_()
        self._dynamic_ready = False
        self._reference_flagged_indices = None
        self._reference_flagged_means = None
        self._grad2d_accum = None
        self._grad2d_count = None
        self._opt_step = 0

    def _resize_dynamic_buffers(self, num_points: int) -> None:
        object_flags = self.object_flags
        current_active = self.current_active_mask
        sam3d_init_target_flags = self.sam3d_init_target_flags
        if (
            object_flags.shape[0] == num_points
            and current_active.shape[0] == num_points
            and sam3d_init_target_flags.shape[0] == num_points
        ):
            return

        new_object_flags = torch.zeros(num_points, 1, dtype=object_flags.dtype, device=object_flags.device)
        new_current_active = torch.zeros(num_points, dtype=torch.bool, device=current_active.device)
        new_sam3d_init_target_flags = torch.zeros(num_points, 1, dtype=sam3d_init_target_flags.dtype, device=sam3d_init_target_flags.device)
        keep = min(object_flags.shape[0], num_points)
        if keep > 0:
            new_object_flags[:keep] = object_flags[:keep]
            new_current_active[:keep] = current_active[:keep]
            new_sam3d_init_target_flags[:keep] = sam3d_init_target_flags[:keep]
        self._buffers["object_flags"] = new_object_flags
        self._buffers["current_active_mask"] = new_current_active
        self._buffers["sam3d_init_target_flags"] = new_sam3d_init_target_flags

    def _refresh_gaussian_optimizers(self, reset_means_optimizer: bool) -> None:
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

        self.gauss_params["means"].register_hook(self._mask_means_grad)
        if self.phase == "dynamic" and self.config.enable_scene_optimization:
            self._register_scene_opt_hooks()
        self._apply_phase_trainability()
        self._apply_phase_optimizers(reset_means_optimizer=reset_means_optimizer)

    def _build_new_gaussian_tensors(self, new_xyz: Tensor, new_rgb: Tensor) -> Dict[str, Tensor]:
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

    def insert_object_gaussians(self, new_xyz: Tensor, new_rgb: Tensor, object_flag: bool = True) -> Tensor:
        num_new = new_xyz.shape[0]
        if num_new == 0:
            return torch.zeros((0,), dtype=torch.long, device=self.means.device)

        new_tensors = self._build_new_gaussian_tensors(new_xyz, new_rgb)
        old_num_points = self.num_points
        for name in ["means", "features_dc", "features_rest", "scales", "quats", "opacities"]:
            concatenated = torch.cat([self.gauss_params[name].detach(), new_tensors[name]], dim=0)
            self.gauss_params[name] = torch.nn.Parameter(concatenated)

        self._resize_dynamic_buffers(self.num_points)
        if object_flag:
            self.object_flags[old_num_points:] = 1.0
        self._refresh_gaussian_optimizers(reset_means_optimizer=True)
        return torch.arange(old_num_points, self.num_points, device=self.means.device, dtype=torch.long)

    def _get_render_projection_params(self, camera) -> tuple[np.ndarray, np.ndarray, int, int]:
        if self.training:
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)[0].detach().cpu().numpy().astype(np.float32)
        intrinsics = camera.get_intrinsics_matrices()[0].detach().cpu().numpy().astype(np.float32)
        width, height = int(camera.width.item()), int(camera.height.item())
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore[arg-type]
        return viewmat, intrinsics, width, height

    @staticmethod
    def _estimate_spacing(points: np.ndarray, max_samples: int = 50_000) -> float:
        if len(points) <= 1:
            return 1e-3
        if len(points) > max_samples:
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(len(points), size=max_samples, replace=False)
            points = points[sample_idx]
        neighbor_k = min(3, len(points) - 1)
        distances, _ = k_nearest_sklearn(torch.from_numpy(points.astype(np.float32)), neighbor_k)
        return float(distances.mean(dim=-1).median().item())

    @torch.no_grad()
    def _build_persistent_object_membership(
        self,
        fused_object_points: np.ndarray,
        initial_target_points: np.ndarray,
        initial_target_indices: Tensor,
        inserted_indices: Tensor,
    ) -> Dict[str, float]:
        num_points = self.num_points
        device = self.object_flags.device
        persistent_flags = torch.zeros((num_points, 1), dtype=self.object_flags.dtype, device=device)

        initial_target_indices = initial_target_indices.to(device=device, dtype=torch.long)
        inserted_indices = inserted_indices.to(device=device, dtype=torch.long)
        if initial_target_indices.numel() > 0:
            persistent_flags[initial_target_indices] = 1.0
        if inserted_indices.numel() > 0:
            persistent_flags[inserted_indices] = 1.0

        all_indices = torch.arange(num_points, device=device, dtype=torch.long)
        fixed_mask = persistent_flags.squeeze(-1) > 0.5
        candidate_indices = all_indices[~fixed_mask]
        if candidate_indices.numel() == 0:
            self.object_flags.copy_(persistent_flags)
            self._persistent_object_membership_ready = True
            return {
                "persistent_object_count": float(fixed_mask.sum().item()),
                "proxy_radius": 0.0,
                "target_radius": 0.0,
                "near_proxy_count": 0.0,
                "near_target_count": 0.0,
            }

        candidate_points = self.means[candidate_indices].detach().cpu().numpy().astype(np.float32)
        proxy_points = fused_object_points.astype(np.float32)
        target_points = initial_target_points.astype(np.float32)

        from sklearn.neighbors import NearestNeighbors

        proxy_spacing = self._estimate_spacing(proxy_points)
        target_spacing = self._estimate_spacing(target_points)
        proxy_radius = max(0.003, 1.5 * proxy_spacing)
        target_radius = max(0.002, 2.0 * target_spacing)

        near_proxy = np.zeros((len(candidate_points),), dtype=bool)
        if len(proxy_points) > 0:
            proxy_nn = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean").fit(proxy_points)
            proxy_distances, _ = proxy_nn.kneighbors(candidate_points)
            near_proxy = np.isfinite(proxy_distances[:, 0]) & (proxy_distances[:, 0] <= proxy_radius)

        near_target = np.zeros((len(candidate_points),), dtype=bool)
        if len(target_points) > 0:
            target_nn = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean").fit(target_points)
            target_distances, _ = target_nn.kneighbors(candidate_points)
            near_target = np.isfinite(target_distances[:, 0]) & (target_distances[:, 0] <= target_radius)

        keep_candidates = near_proxy | near_target
        if np.any(keep_candidates):
            keep_candidate_indices = candidate_indices[
                torch.from_numpy(keep_candidates).to(device=device, dtype=torch.bool)
            ]
            persistent_flags[keep_candidate_indices] = 1.0

        self.object_flags.copy_(persistent_flags)
        self._persistent_object_membership_ready = True
        return {
            "persistent_object_count": float((persistent_flags.squeeze(-1) > 0.5).sum().item()),
            "proxy_radius": float(proxy_radius),
            "target_radius": float(target_radius),
            "near_proxy_count": float(int(near_proxy.sum())),
            "near_target_count": float(int(near_target.sum())),
        }

    def _get_existing_object_subset(
        self,
        render_object_mask: Tensor,
        rendered_depth: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        centers_2d, radii = extract_projected_centers_and_radii(self.info, self.num_points)
        mask = render_object_mask[..., 0] if render_object_mask.ndim == 3 else render_object_mask
        depth_image = rendered_depth[..., 0] if rendered_depth.ndim == 3 else rendered_depth
        height, width = mask.shape

        projected_depths = self.info.get("depths")
        if projected_depths is None:
            raise RuntimeError("SAM3D initialization requires projected Gaussian depths in rasterization info.")
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
            & (center_x >= 0)
            & (center_x < width)
            & (center_y >= 0)
            & (center_y < height)
        )
        candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
        if candidate_indices.numel() > 0:
            candidate_mask[candidate_indices] &= mask[center_y[candidate_indices], center_x[candidate_indices]] > 0.5
            candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)

        if candidate_indices.numel() >= 2:
            pixel_ids = (center_y[candidate_indices] * width + center_x[candidate_indices]).detach().cpu().numpy()
            candidate_depths = projected_depths[candidate_indices].detach().cpu().numpy()
            order = np.lexsort((candidate_depths, pixel_ids))
            sorted_indices = candidate_indices[torch.from_numpy(order).to(candidate_indices.device)]
            sorted_pixel_ids = pixel_ids[order]
            keep = np.zeros(len(sorted_indices), dtype=bool)
            rank_in_pixel = 0
            # Keep only the frontmost Gaussian per masked pixel for SAM3D
            # initialization. This is intentionally strict to avoid pulling
            # table/support geometry into the registration target.
            top_k_per_pixel = 1
            for idx in range(len(sorted_indices)):
                if idx == 0 or sorted_pixel_ids[idx] != sorted_pixel_ids[idx - 1]:
                    rank_in_pixel = 0
                else:
                    rank_in_pixel += 1
                keep[idx] = rank_in_pixel < top_k_per_pixel
            candidate_indices = sorted_indices[torch.from_numpy(keep).to(candidate_indices.device)]

        if candidate_indices.numel() >= 3:
            sampled_depth = depth_image[center_y[candidate_indices], center_x[candidate_indices]]
            candidate_count_before_depth = int(candidate_indices.numel())

            if candidate_indices.numel() > 1:
                nn_k = min(3, candidate_indices.numel() - 1)
                nn_distances, _ = k_nearest_sklearn(self.means[candidate_indices].detach().cpu(), nn_k)
                target_spacing = float(nn_distances.mean(dim=-1).median().item())
            else:
                target_spacing = 1e-3

            # Keep a thin front shell for SAM3D initialization. A looser gate
            # tends to leak table/support Gaussians into the target subset and
            # inflates the object scale estimate.
            depth_tolerance = max(0.008, 5.0 * target_spacing)
            desired_min_keep = max(3, int(0.50 * candidate_count_before_depth))
            best_visible = None
            best_visible_count = 0
            for multiplier in (1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
                current_visible = (
                    torch.isfinite(sampled_depth)
                    & (torch.abs(projected_depths[candidate_indices] - sampled_depth) <= multiplier * depth_tolerance)
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
            # Thin the SAM3D registration target while keeping the subset
            # spread across the current front-surface ordering.
            keep_count = max(3, candidate_indices.numel() // 2)
            keep_positions = torch.linspace(
                0,
                candidate_indices.numel() - 1,
                steps=keep_count,
                device=candidate_indices.device,
            )
            keep_positions = torch.round(keep_positions).long().unique(sorted=True)
            candidate_indices = candidate_indices[keep_positions]

        existing_means = self.means[candidate_indices].detach()
        existing_colors = self.colors[candidate_indices].detach()
        return candidate_indices, existing_means, existing_colors

    @torch.no_grad()
    def initialize_object_from_sam3d(
        self,
        render_image_path: Path,
        object_mask_path: Path,
        render_object_mask: Tensor,
        rendered_depth: Tensor,
        camera,
        debug_dir: Path,
        frame_name: str,
    ) -> Dict[str, Union[int, float, str]]:
        if not self.config.use_sam3d_object_init:
            return {}

        existing_indices, existing_means, existing_colors = self._get_existing_object_subset(
            render_object_mask,
            rendered_depth,
        )
        if existing_indices.numel() < 3:
            raise RuntimeError("Not enough existing object Gaussians for SAM3D alignment.")

        output_stem = f"{frame_name}_sam3d"
        run_device = torch.device(self.means.device)
        sam3d_outputs = get_sam3d_output_paths(debug_dir, output_stem)
        preferred_sam3d_ply = debug_dir / f"{frame_name}_d0_true_sam3d_raw_output.ply"
        if preferred_sam3d_ply.exists():
            sam3d_outputs["ply_path"] = preferred_sam3d_ply
        existing_indices = existing_indices.detach().cpu()
        existing_means_np = existing_means.detach().cpu().numpy()
        existing_colors_np = existing_colors.detach().cpu().numpy()
        self.sam3d_init_target_flags.zero_()
        if existing_indices.numel() > 0:
            self.sam3d_init_target_flags[existing_indices.to(self.sam3d_init_target_flags.device)] = 1.0

        # --- TIMING: SAM3D generation (subprocess that creates 3D point cloud from RGB + mask) ---
        t_sam3d_gen = time.time()
        reused_cached_sam3d = self.config.reuse_sam3d_generated_ply and sam3d_outputs["ply_path"].exists()
        if reused_cached_sam3d:
            CONSOLE.log(f"[dynamic-gs] reusing cached SAM3D point cloud: {sam3d_outputs['ply_path']}")
        else:
            self.to("cpu")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                sam3d_inputs = prepare_cropped_sam3d_inputs(
                    render_image_path,
                    object_mask_path,
                    debug_dir,
                    output_stem,
                )
                sam3d_outputs = run_sam3d_single_object_subprocess(
                    sam3d_inputs["render_image_path"],
                    sam3d_inputs["object_mask_path"],
                    debug_dir,
                    output_stem,
                    max_side=80,
                )
            finally:
                self.to(run_device)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        sam3d_generation_time = time.time() - t_sam3d_gen

        # --- TIMING: SAM3D insertion (CPD registration, dedup, Gaussian insertion) ---
        t_sam3d_ins = time.time()
        source_points, source_colors = load_sam3d_gaussian_ply(sam3d_outputs["ply_path"])
        insertion_result: Sam3DInsertionResult = register_and_fuse_sam3d_object(
            source_points=source_points,
            source_colors=source_colors,
            target_points=existing_means_np,
            target_colors=existing_colors_np,
            debug_dir=debug_dir,
            output_stem=output_stem,
        )
        aligned_path = debug_dir / f"{output_stem}_aligned_output.ply"
        save_point_cloud(aligned_path, insertion_result.aligned_points, insertion_result.aligned_colors)

        if insertion_result.kept_point_count > 0:
            inserted_indices = self.insert_object_gaussians(
                torch.from_numpy(insertion_result.kept_points),
                torch.from_numpy(insertion_result.kept_colors),
                object_flag=True,
            )
        else:
            inserted_indices = torch.zeros((0,), dtype=torch.long, device=self.means.device)
        fused_path = debug_dir / f"{output_stem}_fused_object_only.ply"
        fused_points = existing_means_np
        fused_colors = existing_colors_np
        if insertion_result.kept_point_count > 0:
            fused_points = np.concatenate([fused_points, insertion_result.kept_points], axis=0)
            fused_colors = np.concatenate([fused_colors, insertion_result.kept_colors], axis=0)

        persistent_membership_stats = self._build_persistent_object_membership(
            fused_object_points=fused_points,
            initial_target_points=existing_means_np,
            initial_target_indices=existing_indices.to(self.means.device),
            inserted_indices=inserted_indices,
        )
        save_point_cloud(
            fused_path,
            fused_points,
            fused_colors,
        )

        log_path = debug_dir / f"{output_stem}_fusion_log.txt"
        log_lines = [
            f"frame_name: {frame_name}",
            f"chosen_scale: {insertion_result.chosen_scale}",
            f"voxel_size: {insertion_result.voxel_size}",
            f"source_point_count: {insertion_result.source_point_count}",
            f"target_point_count: {insertion_result.target_point_count}",
            f"visible_source_point_count: {insertion_result.visible_source_point_count}",
            f"registration_source_point_count: {insertion_result.registration_source_point_count}",
            f"similarity_transform: {insertion_result.similarity_transform.tolist()}",
            f"similarity_correspondence_count: {insertion_result.similarity_correspondence_count}",
            f"similarity_scale: {insertion_result.similarity_scale}",
            f"correspondence_threshold: {insertion_result.correspondence_threshold}",
            f"kept_points_after_dedup: {insertion_result.kept_point_count}",
            f"dedup_threshold: {insertion_result.dedup_threshold}",
            f"persistent_object_count: {int(persistent_membership_stats['persistent_object_count'])}",
            f"proxy_radius: {persistent_membership_stats['proxy_radius']}",
            f"target_radius: {persistent_membership_stats['target_radius']}",
            f"near_proxy_count: {int(persistent_membership_stats['near_proxy_count'])}",
            f"near_target_count: {int(persistent_membership_stats['near_target_count'])}",
            f"reused_cached_sam3d: {reused_cached_sam3d}",
            f"raw_sam3d_output_ply: {sam3d_outputs['ply_path']}",
            f"aligned_sam3d_output_ply: {aligned_path}",
            f"fused_object_only_ply: {fused_path}",
            f"source_reg_ref_ply: {debug_dir / f'{output_stem}_source_reg_ref.ply'}",
            f"target_reg_ref_ply: {debug_dir / f'{output_stem}_target_reg_ref.ply'}",
            f"correspondence_plot_path: {insertion_result.correspondence_plot_path}",
        ]
        log_path.write_text("\n".join(log_lines) + "\n")
        sam3d_insertion_time = time.time() - t_sam3d_ins

        return {
            "chosen_scale": insertion_result.chosen_scale,
            "existing_object_gaussians": int(existing_indices.numel()),
            "sam3d_generated_points": insertion_result.source_point_count,
            "visible_source_points_used_for_scale": insertion_result.visible_source_point_count,
            "kept_points_after_dedup": insertion_result.kept_point_count,
            "persistent_object_count": int(persistent_membership_stats["persistent_object_count"]),
            "dedup_threshold": insertion_result.dedup_threshold,
            "reused_cached_sam3d": int(reused_cached_sam3d),
            "raw_sam3d_output_ply": str(sam3d_outputs["ply_path"]),
            "aligned_sam3d_output_ply": str(aligned_path),
            "fused_object_only_ply": str(fused_path),
            "fusion_log_path": str(log_path),
            "sam3d_generation_time": sam3d_generation_time,
            "sam3d_insertion_time": sam3d_insertion_time,
        }

    @torch.no_grad()
    def refresh_dynamic_state_after_insertion(self, camera, render_object_mask: Tensor, optim_mask: Tensor) -> int:
        was_training = self.training
        try:
            self.eval()
            self.get_outputs(camera.to(self.device))
            centers_2d, radii = extract_projected_centers_and_radii(self.info, self.num_points)
            visible_object = build_active_mask(render_object_mask, centers_2d, radii)
            persistent_object = self.object_flags.squeeze(-1) > 0.5
            active = persistent_object & visible_object
            if not torch.any(active):
                active = persistent_object & torch.isfinite(radii) & (radii > 0)
            self.current_active_mask.copy_(active)
            flagged_indices = torch.nonzero(active, as_tuple=False).squeeze(-1)
            if flagged_indices.numel() >= 3:
                self._reference_flagged_indices = flagged_indices.detach().clone()
                self._reference_flagged_means = self.means[flagged_indices].detach().clone()
            else:
                self._reference_flagged_indices = None
                self._reference_flagged_means = None
            self._set_optim_mask(optim_mask.to(self.device))
            self._dynamic_ready = True
            return int(active.sum().item())
        finally:
            if was_training:
                self.train()

    def _get_gt_depth(self, batch):
        depth = batch["depth_image"]
        if depth.ndim == 2:
            depth = depth[..., None]
        if not torch.is_floating_point(depth):
            depth = depth.float()
        downscale = self._get_downscale_factor()
        if downscale > 1:
            size = (depth.shape[0] // downscale, depth.shape[1] // downscale)
            depth = F.interpolate(
                depth.permute(2, 0, 1).unsqueeze(0),
                size=size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0)
        return depth.to(self.device)

    def _get_batch_mask(self, batch):
        mask = batch.get("mask")
        if mask is None:
            return None
        if mask.ndim == 2:
            mask = mask[..., None]
        mask = mask.float()
        downscale = self._get_downscale_factor()
        if downscale > 1:
            size = (mask.shape[0] // downscale, mask.shape[1] // downscale)
            mask = F.interpolate(
                mask.permute(2, 0, 1).unsqueeze(0),
                size=size,
                mode="nearest",
            ).squeeze(0).permute(1, 2, 0)
        return mask.to(self.device)

    @staticmethod
    def _masked_rgb_l1(pred, gt, mask):
        if mask is None:
            return torch.abs(pred - gt).mean()
        denom = (mask.sum() * pred.shape[-1]).clamp_min(1.0)
        return torch.abs((pred - gt) * mask).sum() / denom

    def _get_esam_model(self):
        if self._esam_model is None:
            self._esam_model = build_esam_ti(torch.device(self.device))
        return self._esam_model

    def _get_sam2_video_predictor(self):
        if self._sam2_video_predictor is None:
            self._sam2_video_predictor = build_sam2_tiny_video_predictor(torch.device(self.device))
        return self._sam2_video_predictor

    @torch.no_grad()
    def prepare_dynamic_update(
        self,
        camera,
        batch,
        previous_rendered_rgb=None,
        previous_render_object_mask=None,
        previous_live_rgb=None,
        previous_live_object_mask=None,
        use_render_sam2=False,
        use_live_sam2=False,
        external_object_mask=None,
    ):
        """Generate the change mask and active Gaussian subset for one dynamic frame.

        When *external_object_mask* is provided (subsequent frames where SAM2
        already ran in the pipeline), internal ESAM/SAM2 calls are skipped and the
        change-mask comparison additionally masks out the object region.
        """

        if "depth_image" not in batch:
            raise ValueError("dynamic_scene must provide depth_image for dynamic-gs phase 2.")

        was_training = self.training
        _substeps = {}
        try:
            self.eval()
            # --- TIMING: D0.1a Forward render (get_outputs in eval mode) ---
            _t = time.time()
            outputs = self.get_outputs(camera.to(self.device))
            _substeps["D0.1a_forward_render"] = time.time() - _t
            if outputs["depth"] is None:
                raise RuntimeError("Static reference render did not produce depth.")

            gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
            gt_depth = self._get_gt_depth(batch)
            valid_mask = self._get_batch_mask(batch)
            live_rgb = gt_rgb

            # When an external object mask is given, exclude the object region
            # from the change mask to avoid false positives from the moved object.
            change_valid_mask = valid_mask
            if external_object_mask is not None:
                ext_mask = external_object_mask.float()
                if ext_mask.ndim == 2:
                    ext_mask = ext_mask[..., None]
                ext_mask = ext_mask.to(self.device)
                if change_valid_mask is not None:
                    change_valid_mask = change_valid_mask * (1.0 - ext_mask)
                else:
                    change_valid_mask = 1.0 - ext_mask

            # --- TIMING: D0.1b Change mask (MSSIM depth+RGB comparison, morphological filtering) ---
            _t = time.time()
            change_mask = build_change_mask(
                outputs["depth"],
                gt_depth,
                pred_rgb=outputs["rgb"],
                gt_rgb=gt_rgb,
                valid_mask=change_valid_mask,
                depth_threshold=self.config.change_mask_depth_threshold,
                rgb_threshold=self.config.change_mask_rgb_threshold,
                use_rgb=self.config.change_mask_use_rgb,
                blur_kernel_size=self.config.change_mask_blur_kernel_size,
                blur_sigma=self.config.change_mask_blur_sigma,
                filter_radius=self.config.change_mask_filter_radius,
                min_component_size=self.config.change_mask_min_component_size,
            )
            if self.config.active_mask_dilate_radius > 0:
                change_mask = dilate_binary_mask(change_mask, self.config.active_mask_dilate_radius)
            _substeps["D0.1b_change_mask"] = time.time() - _t

            # --- Object segmentation ---
            if external_object_mask is not None:
                # Object mask was already computed externally (SAM2 in pipeline)
                render_object_mask = external_object_mask.float()
                if render_object_mask.ndim == 2:
                    render_object_mask = render_object_mask[..., None]
                render_object_mask = render_object_mask.to(self.device)
                live_object_mask = render_object_mask.clone()
                render_object_mask_source = "external"
                live_object_mask_source = "external"
                render_prompt_points = torch.zeros((0, 2), dtype=torch.long, device=self.device)
                live_prompt_points = render_prompt_points
            else:
                # Bootstrap path: use ESAM / SAM2 internally
                render_prompt_points = torch.zeros((0, 2), dtype=torch.long, device=outputs["rgb"].device)
                # --- TIMING: D0.1c ESAM on render (query_esam_mask on rendered image; includes model load on first call) ---
                _t = time.time()
                if use_render_sam2 and previous_rendered_rgb is not None and previous_render_object_mask is not None:
                    render_object_mask = query_sam2_propagated_mask(
                        self._get_sam2_video_predictor(),
                        previous_rendered_rgb,
                        outputs["rgb"],
                        previous_render_object_mask,
                    )
                    render_object_mask_source = "sam2"
                else:
                    render_object_mask, _, render_prompt_points = query_esam_mask(
                        self._get_esam_model(),
                        outputs["rgb"],
                        change_mask,
                        num_points=ESAM_NUM_PROMPT_POINTS,
                    )
                    render_object_mask_source = "esam"
                _substeps["D0.1c_esam_render"] = time.time() - _t

                live_prompt_points = torch.zeros((0, 2), dtype=torch.long, device=outputs["rgb"].device)
                # --- TIMING: D0.1d ESAM on live (query_esam_mask on live/GT image) ---
                _t = time.time()
                if use_live_sam2 and previous_live_rgb is not None and previous_live_object_mask is not None:
                    live_object_mask = query_sam2_propagated_mask(
                        self._get_sam2_video_predictor(),
                        previous_live_rgb,
                        live_rgb,
                        previous_live_object_mask,
                    )
                    live_object_mask_source = "sam2"
                else:
                    live_object_mask, _, live_prompt_points = query_esam_mask(
                        self._get_esam_model(),
                        live_rgb,
                        change_mask,
                        num_points=ESAM_NUM_PROMPT_POINTS,
                    )
                    live_object_mask_source = "esam"
                _substeps["D0.1d_esam_live"] = time.time() - _t

                render_object_mask = (
                    render_object_mask[..., None].float()
                    if render_object_mask.ndim == 2
                    else render_object_mask.float()
                )
                live_object_mask = (
                    live_object_mask[..., None].float()
                    if live_object_mask.ndim == 2
                    else live_object_mask.float()
                )

            flag_mask = render_object_mask
            if not torch.any(flag_mask > 0.5):
                CONSOLE.log(
                    f"[dynamic-gs] {render_object_mask_source.upper()} returned an empty rendered object mask; "
                    "falling back to the change mask for flagging."
                )
                flag_mask = change_mask

            # --- TIMING: D0.1e Gaussian flagging (combine masks, project centers, flag object Gaussians) ---
            _t = time.time()
            optim_mask = combine_object_masks(
                render_object_mask if torch.any(render_object_mask > 0.5) else flag_mask,
                live_object_mask if torch.any(live_object_mask > 0.5) else change_mask,
                valid_mask=valid_mask,
            )
            if not torch.any(optim_mask > 0.5):
                CONSOLE.log("[dynamic-gs] combined optimization mask is empty; falling back to the change mask.")
                optim_mask = change_mask

            centers_2d, radii = extract_projected_centers_and_radii(self.info, self.num_points)
            if self._has_persistent_object_membership():
                visible_object = build_active_mask(
                    render_object_mask if torch.any(render_object_mask > 0.5) else flag_mask,
                    centers_2d,
                    radii,
                )
                persistent_object = self.object_flags.squeeze(-1) > 0.5
                active = persistent_object & visible_object
                if not torch.any(active):
                    CONSOLE.log(
                        "[dynamic-gs] persistent 3D object membership selected no visible Gaussians; "
                        "falling back to visible persistent members."
                    )
                    active = persistent_object & torch.isfinite(radii) & (radii > 0)
            else:
                active = build_active_mask(flag_mask, centers_2d, radii)
                if not torch.any(active):
                    CONSOLE.log(
                        "[dynamic-gs] object-mask-based flagging selected no Gaussians; "
                        "falling back to visible Gaussians."
                    )
                    active = torch.isfinite(radii) & (radii > 0)
                self.object_flags.copy_(active.float()[:, None])

            self.current_active_mask.copy_(active)
            flagged_indices = torch.nonzero(active, as_tuple=False).squeeze(-1)
            if flagged_indices.numel() >= 3:
                self._reference_flagged_indices = flagged_indices.detach().clone()
                self._reference_flagged_means = self.means[flagged_indices].detach().clone()
            else:
                self._reference_flagged_indices = None
                self._reference_flagged_means = None
            # When external mask is provided, the pipeline sets the loss mask to CDN.
            # Only set optim_mask as loss mask during bootstrap (frame 0).
            if external_object_mask is None:
                self._set_optim_mask(optim_mask.to(self.device))
            self._dynamic_ready = True

            _substeps["D0.1e_gaussian_flagging"] = time.time() - _t

            # Compute scene-opt activated mask: non-object Gaussians in the change region
            scene_opt_activated_mask = None
            if self.config.enable_scene_optimization:
                in_change = build_active_mask(change_mask.squeeze(-1), centers_2d, radii)
                not_object = ~(self.object_flags.squeeze(-1) > 0.5)
                scene_opt_activated_mask = in_change & not_object

            return {
                "prepare_dynamic_update_substeps": _substeps,
                "change_mask_pixels": int((change_mask[..., 0] > 0.5).sum().item()),
                "flagged_gaussians": int(active.sum().item()),
                "render_object_mask_pixels": int((render_object_mask[..., 0] > 0.5).sum().item()),
                "live_object_mask_pixels": int((live_object_mask[..., 0] > 0.5).sum().item()),
                "optim_mask_pixels": int((optim_mask[..., 0] > 0.5).sum().item()),
                "render_object_mask_source": render_object_mask_source,
                "live_object_mask_source": live_object_mask_source,
                "render_prompt_point_count": int(render_prompt_points.shape[0]),
                "live_prompt_point_count": int(live_prompt_points.shape[0]),
                "render_object_mask": render_object_mask,
                "live_object_mask": live_object_mask,
                "optim_mask": optim_mask,
                "render_propagation_mask": render_object_mask if torch.any(render_object_mask > 0.5) else flag_mask,
                "live_rgb": live_rgb,
                "live_propagation_mask": live_object_mask if torch.any(live_object_mask > 0.5) else optim_mask,
                "rendered_rgb": outputs["rgb"],
                "rendered_depth": outputs["depth"],
                "change_mask": change_mask,
                "scene_opt_activated_mask": scene_opt_activated_mask,
            }
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def render_object_mask(self, camera: Cameras) -> Tensor:
        """Render only object_flags Gaussians and return a dilated (H,W,1) binary mask."""
        was_training = self.training
        self.train()  # training mode for correct downscale
        try:
            camera = camera.to(self.device)
            optimized_camera_to_world = camera.camera_to_worlds
            camera_scale_fac = self._get_downscale_factor()
            camera.rescale_output_resolution(1 / camera_scale_fac)
            viewmat = get_viewmat(optimized_camera_to_world)
            K = camera.get_intrinsics_matrices().to(self.device)
            width, height = int(camera.width.item()), int(camera.height.item())
            camera.rescale_output_resolution(camera_scale_fac)

            obj_mask = self.object_flags.squeeze(-1) > 0.5
            if not obj_mask.any():
                return torch.zeros(height, width, 1, device=self.device)

            opacities = torch.sigmoid(self.opacities[obj_mask]).squeeze(-1)
            scales = torch.exp(self.scales[obj_mask])
            colors_dc = self.features_dc[obj_mask]
            if self.config.sh_degree > 0:
                colors = torch.cat((colors_dc[:, None, :], self.features_rest[obj_mask]), dim=1)
            else:
                colors = torch.sigmoid(colors_dc).squeeze(1)
            sh_degree = min(self.step // self.config.sh_degree_interval, self.config.sh_degree) if self.config.sh_degree > 0 else None

            _, subset_alpha, _ = rasterization(
                means=self.means[obj_mask],
                quats=self.quats[obj_mask],
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=viewmat,
                Ks=K,
                width=width,
                height=height,
                packed=False,
                near_plane=0.01,
                far_plane=1e10,
                render_mode="RGB",
                sh_degree=sh_degree,
                sparse_grad=False,
                absgrad=False,
                rasterize_mode=self.config.rasterize_mode,
            )
            binary = (subset_alpha.squeeze(0) > 0.01).float()
            if binary.ndim == 2:
                binary = binary[..., None]
            if self.config.object_mask_dilate_px > 0:
                binary = dilate_binary_mask(binary, self.config.object_mask_dilate_px)
            return binary
        finally:
            if not was_training:
                self.eval()

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[Tensor, list]]:
        if not isinstance(camera, Cameras):
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                empty_outputs = self.get_empty_outputs(
                    int(camera.width.item()),
                    int(camera.height.item()),
                    self.background_color,
                )
                empty_outputs["flagged_rgb"] = empty_outputs["rgb"]
                empty_outputs["non_flagged_rgb"] = empty_outputs["rgb"]
                empty_outputs["sam3d_init_target_rgb"] = empty_outputs["rgb"]
                return empty_outputs
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
            object_flags_crop = self.object_flags[crop_ids].squeeze(-1) > 0.5
            sam3d_init_target_flags_crop = self.sam3d_init_target_flags[crop_ids].squeeze(-1) > 0.5
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats
            object_flags_crop = self.object_flags.squeeze(-1) > 0.5
            sam3d_init_target_flags_crop = self.sam3d_init_target_flags.squeeze(-1) > 0.5

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().to(self.device)
        width, height = int(camera.width.item()), int(camera.height.item())
        self.last_size = (height, width)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore[arg-type]

        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError(f"Unknown rasterize_mode: {self.config.rasterize_mode}")

        render_mode = "RGB+ED" if self.config.output_depth_during_training or not self.training else "RGB"
        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)
            sh_degree_to_use = None

        opacities_crop = torch.sigmoid(opacities_crop).squeeze(-1)
        scales_crop = torch.exp(scales_crop)

        render, alpha, info = rasterization(
            means=means_crop,
            quats=quats_crop,
            scales=scales_crop,
            opacities=opacities_crop,
            colors=colors_crop,
            viewmats=viewmat,
            Ks=K,
            width=width,
            height=height,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=False,
            rasterize_mode=self.config.rasterize_mode,
        )
        self.info = info
        alpha = alpha[:, ...]

        background = self._get_background_color()
        rgb = torch.clamp(render[:, ..., :3] + (1 - alpha) * background, 0.0, 1.0)

        def render_subset_rgb(mask: Tensor) -> Tensor:
            if not bool(mask.any()):
                return background[None, None, :].expand(height, width, 3)

            subset_render, subset_alpha, _ = rasterization(
                means=means_crop[mask],
                quats=quats_crop[mask],
                scales=scales_crop[mask],
                opacities=opacities_crop[mask],
                colors=colors_crop[mask],
                viewmats=viewmat,
                Ks=K,
                width=width,
                height=height,
                packed=False,
                near_plane=0.01,
                far_plane=1e10,
                render_mode="RGB",
                sh_degree=sh_degree_to_use,
                sparse_grad=False,
                absgrad=False,
                rasterize_mode=self.config.rasterize_mode,
            )
            subset_alpha = subset_alpha[:, ...]
            return torch.clamp(subset_render[:, ..., :3] + (1 - subset_alpha) * background, 0.0, 1.0).squeeze(0)

        flagged_rgb = render_subset_rgb(object_flags_crop)
        non_flagged_rgb = render_subset_rgb(~object_flags_crop)
        sam3d_init_target_rgb = render_subset_rgb(sam3d_init_target_flags_crop)

        if self.config.use_bilateral_grid and self.training:
            if camera.metadata is not None and "cam_idx" in camera.metadata:
                rgb = self._apply_bilateral_grid(rgb, camera.metadata["cam_idx"], height, width)

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(height, width, 3)

        if self.training and self.phase == "dynamic" and self.config.enable_scene_optimization:
            if self.info["means2d"].requires_grad:
                self.info["means2d"].retain_grad()

        return {
            "rgb": rgb.squeeze(0),
            "depth": depth_im,
            "flagged_rgb": flagged_rgb,
            "non_flagged_rgb": non_flagged_rgb,
            "sam3d_init_target_rgb": sam3d_init_target_rgb,
            "accumulation": alpha.squeeze(0),
            "background": background,
        }

    def get_metrics_dict(self, outputs, batch):
        metrics = super().get_metrics_dict(outputs, batch)
        metrics["active_gaussian_count"] = int(self.current_active_mask.sum().item())
        metrics["object_flag_count"] = int((self.object_flags.squeeze(-1) > 0.5).sum().item())
        metrics["sam3d_init_target_count"] = int((self.sam3d_init_target_flags.squeeze(-1) > 0.5).sum().item())
        metrics["eligible_gaussian_count"] = int(self._get_eligible_mask().sum().item())
        if self._dynamic_ready:
            metrics["optim_mask_pixels"] = int((self.change_mask_image[..., 0] > 0.5).sum().item())
        return metrics

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        if self.phase == "static":
            return super().get_loss_dict(outputs, batch, metrics_dict)

        mask = self._get_optim_mask(target_shape=outputs["rgb"].shape)
        if mask is None:
            raise RuntimeError("Dynamic phase started before optimization-mask generation.")

        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        rgb_loss = self._masked_rgb_l1(outputs["rgb"], gt_img, mask)

        depth_loss = outputs["rgb"].new_tensor(0.0)
        if outputs["depth"] is not None and "depth_image" in batch:
            depth_loss = masked_l1_depth_loss(outputs["depth"], self._get_gt_depth(batch), mask)

        rigid_static_reg = outputs["rgb"].new_tensor(0.0)
        if (
            self.config.rigid_static_lambda > 0
            and self._reference_flagged_indices is not None
            and self._reference_flagged_means is not None
        ):
            rigid_static_reg = rigid_or_static_loss(
                self._reference_flagged_means,
                self.means[self._reference_flagged_indices],
                rigid_inlier_threshold=self.config.rigid_inlier_threshold,
            )

        loss_dict = {
            "main_loss": rgb_loss,
            "depth_loss": self.config.depth_lambda * depth_loss,
            "rigid_static_loss": self.config.rigid_static_lambda * rigid_static_reg,
        }
        if self.training:
            self.camera_optimizer.get_loss_dict(loss_dict)
        return loss_dict

    def step_post_backward(self, step):
        assert step == self.step
        if self.phase != "dynamic" or not self.config.enable_scene_optimization:
            return
        if not self._dynamic_ready:
            return

        means2d = self.info.get("means2d") if self.info else None
        if means2d is not None and means2d.grad is not None:
            with torch.no_grad():
                grad_norms = means2d.grad.detach().squeeze(0).norm(dim=-1)
                eligible = self._get_eligible_mask() & (self.info["radii"].squeeze(0) > 0)
                if self._grad2d_accum is None:
                    self._grad2d_accum = torch.zeros(self.num_points, device=self.device)
                    self._grad2d_count = torch.zeros(self.num_points, device=self.device)
                self._grad2d_accum[eligible] += grad_norms[eligible]
                self._grad2d_count[eligible] += 1.0

        self._opt_step += 1
        if self._opt_step % self.config.scene_opt_refine_every == 0:
            self._refine_eligible()

    @torch.no_grad()
    def _refine_eligible(self) -> None:
        if self._grad2d_accum is None:
            return
        eligible = self._get_eligible_mask()
        if not eligible.any():
            self._grad2d_accum.zero_()
            self._grad2d_count.zero_()
            return

        changed = False

        # Densify: eligible Gaussians with high 2D gradient
        avg_grad = self._grad2d_accum / self._grad2d_count.clamp_min(1.0)
        densify_mask = (avg_grad > self.config.scene_opt_densify_grad_thresh) & eligible
        n_densified = 0
        if densify_mask.any():
            old_n = self.num_points
            for name in ["means", "features_dc", "features_rest", "scales", "quats", "opacities"]:
                new_slice = self.gauss_params[name].detach()[densify_mask]
                self.gauss_params[name] = torch.nn.Parameter(
                    torch.cat([self.gauss_params[name].detach(), new_slice], dim=0)
                )
            n_densified = self.num_points - old_n
            self._resize_dynamic_buffers(self.num_points)
            zeros = torch.zeros(n_densified, device=self.device)
            self._grad2d_accum = torch.cat([self._grad2d_accum[:old_n], zeros])
            self._grad2d_count = torch.cat([self._grad2d_count[:old_n], zeros])
            changed = True

        # Prune: eligible low-opacity Gaussians
        opacities = torch.sigmoid(self.gauss_params["opacities"].detach()).squeeze(-1)
        eligible = self._get_eligible_mask()
        prune_mask = eligible & (opacities < self.config.scene_opt_cull_alpha_thresh)
        n_pruned = 0
        if prune_mask.any():
            keep_mask = ~prune_mask
            n_pruned = int(prune_mask.sum().item())
            for name in ["means", "features_dc", "features_rest", "scales", "quats", "opacities"]:
                self.gauss_params[name] = torch.nn.Parameter(self.gauss_params[name].detach()[keep_mask])
            self._buffers["object_flags"] = self.object_flags[keep_mask]
            self._buffers["current_active_mask"] = self.current_active_mask[keep_mask]
            self._buffers["sam3d_init_target_flags"] = self.sam3d_init_target_flags[keep_mask]
            if self._grad2d_accum is not None:
                self._grad2d_accum = self._grad2d_accum[keep_mask]
                self._grad2d_count = self._grad2d_count[keep_mask]
            if self._reference_flagged_indices is not None:
                new_idx = torch.cumsum(keep_mask.long(), dim=0) - 1
                valid = keep_mask[self._reference_flagged_indices]
                if valid.any():
                    self._reference_flagged_indices = new_idx[self._reference_flagged_indices[valid]]
                    self._reference_flagged_means = self._reference_flagged_means[valid]
                else:
                    self._reference_flagged_indices = None
                    self._reference_flagged_means = None
            changed = True

        if changed:
            self._refresh_gaussian_optimizers(reset_means_optimizer=False)
            CONSOLE.log(
                f"[dynamic-gs] refine step {self._opt_step}: "
                f"+{n_densified} densified, -{n_pruned} pruned, total={self.num_points}"
            )

        self._grad2d_accum.zero_()
        self._grad2d_count.zero_()
