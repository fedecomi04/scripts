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
from .dynamic_gs_pipeline_recorded import RecordedDynamicGSPipelineConfig
from .static_gs_model import StaticGSModelConfig
from .static_gs_pipeline import StaticGSPipelineConfig, StaticGSTrainer

STATIC_NUM_STEPS = 1000
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
        vis="tensorboard",
    ),
    description="Two-phase static+dynamic Gaussian Splatting with masked mean updates and depth supervision.",
)


# ----------------------------------------------------------------------------
# static-gs — train static, run Phase 0a/0b at end, write post_fusion_state.pt
# ----------------------------------------------------------------------------
#
# Purpose: produce the warm-cache snapshot that the future dynamic-gs +
# dynamic-gs-live pipelines load to skip static + Phase 0b entirely. No
# tracker, no decoder, no dynamic phase — just train the scene, fuse the
# SAM3D objects, write the .pt.
#
# Optimizer set + LRs intentionally identical to dynamic-gs so the loaded
# snapshot is byte-compatible with the dynamic pipelines that consume it.

StaticGS = MethodSpecification(
    config=TrainerConfig(
        _target=StaticGSTrainer,
        method_name="static-gs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_eval_all_images=500,
        steps_per_save=500,
        max_num_iterations=STATIC_NUM_STEPS,
        mixed_precision=False,
        pipeline=StaticGSPipelineConfig(
            static_num_steps=STATIC_NUM_STEPS,
            datamanager=DynamicGSDataManagerConfig(),
            model=StaticGSModelConfig(
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
        vis="tensorboard",
    ),
    description="Static Gaussian Splatting + Phase 0a/0b fusion; writes post_fusion_state.pt for warm-restart into dynamic-gs.",
)


# ----------------------------------------------------------------------------
# dynamic-gs-v2 — Phase 3 recorded subclass of DynamicGSPipelineBase
# ----------------------------------------------------------------------------
#
# Stage-2 trainer: warm-loads the post_fusion_state.pt produced by static-gs,
# then iterates the recorded dynamic dataset frame-by-frame with the XFeat
# tracker + optional feedforward (rgbd / anysplat) decoder. No dynamic-phase
# optimization. The old monolithic `dynamic-gs` entry-point remains
# registered alongside this so v2 can be smoke-tested incrementally; the
# cutover to v2-as-dynamic-gs happens after the live subclass + dead-code
# deletion ship (Phase 3 stages D + E).
#
# Run with:
#   ns-train dynamic-gs-v2 --data /path/to/dataset/with/post_fusion_state.pt
DEFAULT_DYNAMIC_V2_STEPS = 5000  # arbitrary cap; trainer exits earlier if user hits Ctrl+C

DynamicGSV2 = MethodSpecification(
    config=TrainerConfig(
        _target=NoSaveTrainer,
        method_name="dynamic-gs-v2",
        steps_per_eval_image=1_000_000_000,
        steps_per_eval_batch=0,
        steps_per_eval_all_images=1_000_000_000,
        steps_per_save=1_000_000_000,
        max_num_iterations=DEFAULT_DYNAMIC_V2_STEPS,
        mixed_precision=False,
        pipeline=RecordedDynamicGSPipelineConfig(
            datamanager=DynamicGSDataManagerConfig(),
            model=DynamicGSModelConfig(
                camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
                output_depth_during_training=True,
                stop_split_at=0,
                reuse_sam3d_generated_ply=True,
            ),
        ),
        optimizers={
            # Optimizers exist so Nerfstudio's trainer doesn't complain;
            # all per-step loss is zero in dynamic-gs-v2 so step() is a no-op.
            "means": {
                "optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15),
                "scheduler": None,
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="tensorboard",
    ),
    description="Phase 3 dynamic-gs (recorded): warm-loads static-gs cache, runs XFeat tracker + feedforward decoder. Requires post_fusion_state.pt.",
)
