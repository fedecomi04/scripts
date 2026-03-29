from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Literal, Type, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig, get_viewmat
from nerfstudio.utils.rich_utils import CONSOLE

from .utils import (
    NoRefineStrategy,
    build_active_mask,
    build_change_mask,
    dilate_binary_mask,
    extract_projected_centers_and_radii,
    masked_l1_depth_loss,
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


class DynamicGSModel(SplatfactoModel):
    config: DynamicGSModelConfig

    def __init__(self, config, metadata=None, **kwargs):
        self.phase = "static"  # type: Literal["static", "dynamic"]
        self._base_lrs = {}  # type: Dict[str, float]
        self._initial_scheduler_states = {}  # type: Dict[str, Dict]
        self._dynamic_ready = False
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

        if "object_flags" not in state_dict:
            state_dict["object_flags"] = torch.zeros_like(self.object_flags)

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
            self.reset_dynamic_state()

    def _apply_phase_trainability(self):
        static_phase = self.phase == "static"
        self.gauss_params["means"].requires_grad_(not static_phase)
        for name in ["features_dc", "features_rest", "opacities", "scales", "quats"]:
            self.gauss_params[name].requires_grad_(static_phase)

    def _apply_phase_optimizers(self, reset_means_optimizer):
        if not hasattr(self, "optimizers"):
            return

        active_groups = {"means"} if self.phase == "dynamic" else {
            "features_dc",
            "features_rest",
            "opacities",
            "scales",
            "quats",
        }
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
        mask = self.current_active_mask.to(device=grad.device, dtype=grad.dtype).unsqueeze(-1)
        return grad * mask

    def _set_change_mask(self, mask):
        mask = mask.detach().float()
        if mask.ndim == 2:
            mask = mask[..., None]
        if self.change_mask_image.shape != mask.shape:
            self._buffers["change_mask_image"] = mask.clone()
        else:
            self.change_mask_image.copy_(mask)

    def _get_change_mask(self):
        if not self._dynamic_ready:
            return None
        return self._downscale_if_required(self.change_mask_image).to(self.device)

    def reset_dynamic_state(self):
        self.current_active_mask.zero_()
        self.object_flags.zero_()
        self.change_mask_image.zero_()
        self._dynamic_ready = False

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

    @torch.no_grad()
    def prepare_dynamic_update(self, camera, batch):
        """Generate the change mask and active Gaussian subset for one dynamic frame."""

        if "depth_image" not in batch:
            raise ValueError("dynamic_scene must provide depth_image for dynamic-gs phase 2.")

        was_training = self.training
        try:
            self.eval()
            outputs = self.get_outputs(camera.to(self.device))
            if outputs["depth"] is None:
                raise RuntimeError("Static reference render did not produce depth.")

            gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
            gt_depth = self._get_gt_depth(batch)
            valid_mask = self._get_batch_mask(batch)
            change_mask = build_change_mask(
                outputs["depth"],
                gt_depth,
                pred_rgb=outputs["rgb"],
                gt_rgb=gt_rgb,
                valid_mask=valid_mask,
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

            centers_2d, radii = extract_projected_centers_and_radii(self.info, self.num_points)
            active = build_active_mask(change_mask, centers_2d, radii)
            if not torch.any(active):
                CONSOLE.log(
                    "[dynamic-gs] generated change mask flagged no Gaussians; falling back to visible Gaussians."
                )
                active = torch.isfinite(radii) & (radii > 0)

            self.current_active_mask.copy_(active)
            self.object_flags.copy_(active.float()[:, None])
            self._set_change_mask(change_mask.to(self.device))
            self._dynamic_ready = True

            return {
                "change_mask_pixels": int((change_mask[..., 0] > 0.5).sum().item()),
                "flagged_gaussians": int(active.sum().item()),
                "rendered_rgb": outputs["rgb"],
                "change_mask": change_mask,
            }
        finally:
            if was_training:
                self.train()

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
                return self.get_empty_outputs(
                    int(camera.width.item()),
                    int(camera.height.item()),
                    self.background_color,
                )
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

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

        render, alpha, self.info = rasterization(
            means=means_crop,
            quats=quats_crop,
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
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
        alpha = alpha[:, ...]

        background = self._get_background_color()
        rgb = torch.clamp(render[:, ..., :3] + (1 - alpha) * background, 0.0, 1.0)

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

        return {
            "rgb": rgb.squeeze(0),
            "depth": depth_im,
            "accumulation": alpha.squeeze(0),
            "background": background,
        }

    def get_metrics_dict(self, outputs, batch):
        metrics = super().get_metrics_dict(outputs, batch)
        metrics["active_gaussian_count"] = int(self.current_active_mask.sum().item())
        metrics["object_flag_count"] = int((self.object_flags.squeeze(-1) > 0.5).sum().item())
        if self._dynamic_ready:
            metrics["change_mask_pixels"] = int((self.change_mask_image[..., 0] > 0.5).sum().item())
        return metrics

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        if self.phase == "static":
            return super().get_loss_dict(outputs, batch, metrics_dict)

        mask = self._get_change_mask()
        if mask is None:
            raise RuntimeError("Dynamic phase started before change-mask generation.")

        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        rgb_loss = self._masked_rgb_l1(outputs["rgb"], gt_img, mask)

        depth_loss = outputs["rgb"].new_tensor(0.0)
        if outputs["depth"] is not None and "depth_image" in batch:
            depth_loss = masked_l1_depth_loss(outputs["depth"], self._get_gt_depth(batch), mask)

        loss_dict = {
            "main_loss": rgb_loss,
            "depth_loss": self.config.depth_lambda * depth_loss,
        }
        if self.training:
            self.camera_optimizer.get_loss_dict(loss_dict)
        return loss_dict

    def step_post_backward(self, step):
        assert step == self.step
        return None
