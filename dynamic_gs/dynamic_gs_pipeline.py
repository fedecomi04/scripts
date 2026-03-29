from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type

from PIL import Image
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

    static_num_steps: int = 2500
    dynamic_num_steps: int = 500


class DynamicGSPipeline(VanillaPipeline):
    config: DynamicGSPipelineConfig

    def __init__(self, config, device, test_mode="val", world_size=1, local_rank=0, grad_scaler=None):
        self.current_phase = None  # type: Optional[Literal["static", "dynamic"]]
        self._dynamic_prepared = False
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            grad_scaler=grad_scaler,
        )
        self._sync_phase(0)

    @staticmethod
    def _save_debug_image(image, path):
        path = Path(path)
        image_uint8 = image.detach().clamp(0.0, 1.0).mul(255).byte().cpu().numpy()
        Image.fromarray(image_uint8).save(path)

    def _phase_for_step(self, step):
        dynamic_start = self.config.static_num_steps
        dynamic_end = dynamic_start + self.config.dynamic_num_steps
        if step < dynamic_start:
            return "static"
        if step < dynamic_end:
            return "dynamic"
        return "dynamic"

    def _sync_phase(self, step):
        phase = self._phase_for_step(step)
        if phase == self.current_phase:
            return

        if phase == "dynamic" and not self._dynamic_prepared:
            camera, batch = self.datamanager.get_dynamic_train_batch()
            stats = self.model.prepare_dynamic_update(camera, batch)
            debug_path = Path(self.datamanager.config.data) / "change_mask_debug.png"
            self._save_debug_image(stats["debug_image"], debug_path)
            self._dynamic_prepared = True
            CONSOLE.log(
                "[dynamic-gs] generated change mask with "
                f"{stats['change_mask_pixels']} pixels and "
                f"{stats['flagged_gaussians']} flagged gaussians"
            )
            CONSOLE.log(f"[dynamic-gs] saved change-mask debug image to {debug_path}")

        self.current_phase = phase
        self.datamanager.set_phase(phase)
        self.model.set_phase(phase, reset_means_optimizer=phase == "dynamic")
        CONSOLE.log(f"[dynamic-gs] phase -> {phase} at step {step}")

    @profiler.time_function
    def get_train_loss_dict(self, step):
        self._sync_phase(step)
        return super().get_train_loss_dict(step)

    @profiler.time_function
    def get_eval_loss_dict(self, step):
        self._sync_phase(step)
        return super().get_eval_loss_dict(step)

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step):
        self._sync_phase(step)
        return super().get_eval_image_metrics_and_images(step)

    @profiler.time_function
    def get_average_eval_image_metrics(self, step=None, output_path=None, get_std=False):
        if step is not None:
            self._sync_phase(step)
        return super().get_average_eval_image_metrics(step=step, output_path=output_path, get_std=get_std)
