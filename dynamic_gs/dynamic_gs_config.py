from __future__ import annotations

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from .dynamic_gs_datamanager import DynamicGSDataManagerConfig
from .dynamic_gs_model import DynamicGSModelConfig
from .dynamic_gs_pipeline import DynamicGSPipelineConfig
from .dynamic_gs_trainer import NoSaveTrainer

STATIC_NUM_STEPS = 6000
DYNAMIC_STEPS_PER_FRAME = 50   # optimization epochs per dynamic frame
DEFAULT_MAX_NUM_STEPS = STATIC_NUM_STEPS + DYNAMIC_STEPS_PER_FRAME  # updated at runtime


DynamicGS = MethodSpecification(
    config=TrainerConfig(
        _target=NoSaveTrainer,
        method_name="dynamic-gs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_eval_all_images=500,
        steps_per_save=500,
        max_num_iterations=DEFAULT_MAX_NUM_STEPS,
        mixed_precision=False,
        pipeline=DynamicGSPipelineConfig(
            static_num_steps=STATIC_NUM_STEPS,
            dynamic_steps_per_frame=DYNAMIC_STEPS_PER_FRAME,
            datamanager=DynamicGSDataManagerConfig(),
            model=DynamicGSModelConfig(
                camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
                output_depth_during_training=True,
                stop_split_at=0,
                reuse_sam3d_generated_ply=True,
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": None,
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20.0, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Two-phase static+dynamic Gaussian Splatting with masked mean updates and depth supervision.",
)
