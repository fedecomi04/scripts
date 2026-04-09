from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type

from PIL import Image
from nerfstudio.engine.callbacks import TrainingCallbackAttributes
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler
from nerfstudio.utils.rich_utils import CONSOLE

from .dynamic_gs_datamanager import DynamicGSDataManagerConfig
from .dynamic_gs_model import DynamicGSModelConfig


@dataclass
class DynamicGSPipelineConfig(VanillaPipelineConfig):
    _target: Type = field(default_factory=lambda: DynamicGSPipeline)

    datamanager: DynamicGSDataManagerConfig = field(default_factory=DynamicGSDataManagerConfig)
    model: DynamicGSModelConfig = field(default_factory=DynamicGSModelConfig)

    static_num_steps: int = 100
    dynamic_steps_per_frame: int = 10


class DynamicGSPipeline(VanillaPipeline):
    config: DynamicGSPipelineConfig

    def __init__(self, config, device, test_mode="val", world_size=1, local_rank=0, grad_scaler=None):
        self.current_phase = None  # type: Optional[Literal["static", "dynamic"]]
        self.current_dynamic_frame_idx = None  # type: Optional[int]
        self.dynamic_steps_on_current_frame = 0
        self.total_dynamic_frames = 0
        self.total_dynamic_steps = 0
        self._render_sam2_seeded = False
        self._live_sam2_seeded = False
        self._previous_rendered_rgb = None
        self._previous_render_object_mask = None
        self._previous_live_rgb = None
        self._previous_live_object_mask = None
        self._sam3d_inserted = False
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
        self._render_sam2_seeded = False
        self._live_sam2_seeded = False
        self._previous_rendered_rgb = None
        self._previous_render_object_mask = None
        self._previous_live_rgb = None
        self._previous_live_object_mask = None
        self._sam3d_inserted = False

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

    def _total_train_steps(self) -> int:
        return self.config.static_num_steps + self.total_dynamic_steps

    def _phase_for_step(self, step: int) -> Literal["static", "dynamic"]:
        if self.total_dynamic_frames == 0 or step < self.config.static_num_steps:
            return "static"
        return "dynamic"

    def _dynamic_frame_for_step(self, step: int) -> int:
        dynamic_step = max(step - self.config.static_num_steps, 0)
        return min(dynamic_step // self.config.dynamic_steps_per_frame, self.total_dynamic_frames - 1)

    def _prepare_dynamic_frame(self) -> None:
        assert self.current_dynamic_frame_idx is not None
        self.datamanager.set_dynamic_frame_idx(self.current_dynamic_frame_idx)
        self.model.reset_dynamic_state()
        camera, batch = self.datamanager.get_current_dynamic_train_batch()
        stats = self.model.prepare_dynamic_update(
            camera,
            batch,
            previous_rendered_rgb=self._previous_rendered_rgb,
            previous_render_object_mask=self._previous_render_object_mask,
            previous_live_rgb=self._previous_live_rgb,
            previous_live_object_mask=self._previous_live_object_mask,
            use_render_sam2=self._render_sam2_seeded,
            use_live_sam2=self._live_sam2_seeded,
        )

        # Debug tool: keep per-frame rendered reference and computed mask for inspection.
        frame_name = self.datamanager.get_current_dynamic_frame_name()
        debug_dir = self.datamanager.get_dynamic_debug_dir()
        self._save_image(stats["rendered_rgb"], debug_dir / f"{frame_name}_render.png")
        self._save_image(stats["change_mask"], debug_dir / f"{frame_name}_change_mask.png")
        self._save_image(stats["render_object_mask"], debug_dir / f"{frame_name}_render_object_mask.png")
        self._save_image(stats["live_object_mask"], debug_dir / f"{frame_name}_live_object_mask.png")
        self._save_image(stats["optim_mask"], debug_dir / f"{frame_name}_combined_optim_mask.png")

        if self.current_dynamic_frame_idx == 0 and not self._sam3d_inserted and self.model.config.use_sam3d_object_init:
            sam3d_stats = self.model.initialize_object_from_sam3d(
                render_image_path=debug_dir / f"{frame_name}_render.png",
                object_mask_path=debug_dir / f"{frame_name}_live_object_mask.png",
                render_object_mask=stats["render_object_mask"],
                rendered_depth=stats["rendered_depth"],
                camera=camera,
                debug_dir=debug_dir,
                frame_name=frame_name,
            )
            if sam3d_stats:
                self._sam3d_inserted = True
                stats["flagged_gaussians"] = self.model.refresh_dynamic_state_after_insertion(
                    camera,
                    stats["render_object_mask"],
                    stats["optim_mask"],
                )
                CONSOLE.log(
                    "[dynamic-gs] SAM3D object init -> "
                    f"existing={sam3d_stats['existing_object_gaussians']}, "
                    f"scale={sam3d_stats['chosen_scale']:.4f}, "
                    f"generated={sam3d_stats['sam3d_generated_points']}, "
                    f"kept={sam3d_stats['kept_points_after_dedup']}"
                )

        if stats["render_object_mask_pixels"] > 0:
            if stats["render_object_mask_source"] == "esam":
                self._render_sam2_seeded = True
            if self._render_sam2_seeded:
                self._previous_rendered_rgb = stats["rendered_rgb"].detach().clone()
                self._previous_render_object_mask = stats["render_propagation_mask"].detach().clone()

        if stats["live_object_mask_pixels"] > 0:
            if stats["live_object_mask_source"] == "esam":
                self._live_sam2_seeded = True
            if self._live_sam2_seeded:
                self._previous_live_rgb = stats["live_rgb"].detach().clone()
                self._previous_live_object_mask = stats["live_propagation_mask"].detach().clone()

        CONSOLE.log(
            "[dynamic-gs] frame "
            f"{self.current_dynamic_frame_idx + 1}/{self.total_dynamic_frames} -> {frame_name}, "
            f"change pixels={stats['change_mask_pixels']}, "
            f"render {stats['render_object_mask_source']} pixels={stats['render_object_mask_pixels']}, "
            f"live {stats['live_object_mask_source']} pixels={stats['live_object_mask_pixels']}, "
            f"optim pixels={stats['optim_mask_pixels']}, "
            f"flagged gaussians={stats['flagged_gaussians']}"
        )

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
            self.dynamic_steps_on_current_frame = 0
            self._reset_dynamic_segmentation_state()
            return

        frame_idx = self._dynamic_frame_for_step(step)
        frame_changed = frame_idx != self.current_dynamic_frame_idx
        self.current_dynamic_frame_idx = frame_idx
        self.dynamic_steps_on_current_frame = (step - self.config.static_num_steps) % self.config.dynamic_steps_per_frame

        if phase_changed or frame_changed:
            self._prepare_dynamic_frame()

    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes):
        callbacks = super().get_training_callbacks(training_callback_attributes)
        trainer = training_callback_attributes.trainer
        if trainer is not None:
            trainer.config.max_num_iterations = self._total_train_steps()
        return callbacks

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        self._sync_phase(step)
        return super().get_train_loss_dict(step)

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
