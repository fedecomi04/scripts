from __future__ import annotations

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from .dynamic_gs_datamanager import DynamicGSDataManagerConfig
from .dynamic_gs_model import DynamicGSModelConfig
from .dynamic_gs_pipeline_live import LiveDynamicGSPipelineConfig
from .dynamic_gs_pipeline_recorded import RecordedDynamicGSPipelineConfig
from .dynamic_gs_trainer import NoSaveTrainer
from .static_gs_model import StaticGSModelConfig
from .static_gs_pipeline import StaticGSPipelineConfig, StaticGSTrainer


# ----------------------------------------------------------------------------
# static-gs — train static, run Phase 0a/0b at end, write post_fusion_state.pt
# ----------------------------------------------------------------------------
#
# Produces the warm-cache snapshot that dynamic-gs / dynamic-gs-live load
# to skip the static phase + Phase 0b fusion entirely. No tracker, no
# decoder, no dynamic phase — just train the scene, fuse the SAM3D
# objects, write the .pt.
#
# Optimizer set + LRs intentionally identical to dynamic-gs so the loaded
# snapshot is byte-compatible with the dynamic pipelines that consume it.

STATIC_NUM_STEPS = 1000

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
            "means":         {"optimizer": AdamOptimizerConfig(lr=1.6e-4,       eps=1e-15), "scheduler": None},
            "features_dc":   {"optimizer": AdamOptimizerConfig(lr=0.0025,       eps=1e-15), "scheduler": None},
            "features_rest": {"optimizer": AdamOptimizerConfig(lr=0.0025/20.0,  eps=1e-15), "scheduler": None},
            "opacities":     {"optimizer": AdamOptimizerConfig(lr=0.05,         eps=1e-15), "scheduler": None},
            "scales":        {"optimizer": AdamOptimizerConfig(lr=0.005,        eps=1e-15), "scheduler": None},
            "quats":         {"optimizer": AdamOptimizerConfig(lr=0.001,        eps=1e-15), "scheduler": None},
            "camera_opt":    {"optimizer": AdamOptimizerConfig(lr=1e-3,         eps=1e-15), "scheduler": None},
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="tensorboard",
    ),
    description="Static Gaussian Splatting + Phase 0a/0b fusion; writes post_fusion_state.pt for warm-restart into dynamic-gs / dynamic-gs-live.",
)


# ----------------------------------------------------------------------------
# Shared optimizer block for dynamic-gs / dynamic-gs-live
# ----------------------------------------------------------------------------
#
# Both dynamic-gs methods are pure tracker+FF runtimes: no per-step gradient
# descent on Gaussian params. The optimizer block exists so Nerfstudio's
# trainer doesn't complain; LRs are 0 so step() is effectively a no-op.

_ZERO_LR_OPTIMIZERS = {
    "means":         {"optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15), "scheduler": None},
    "features_dc":   {"optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15), "scheduler": None},
    "features_rest": {"optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15), "scheduler": None},
    "opacities":     {"optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15), "scheduler": None},
    "scales":        {"optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15), "scheduler": None},
    "quats":         {"optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15), "scheduler": None},
    "camera_opt":    {"optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15), "scheduler": None},
}


# ----------------------------------------------------------------------------
# dynamic-gs — recorded-mode subclass of DynamicGSPipelineBase
# ----------------------------------------------------------------------------
#
# Warm-loads static-gs's post_fusion_state.pt, then iterates the recorded
# dynamic dataset frame-by-frame with the XFeat tracker + optional
# feedforward (rgbd_decode / anysplat_decode) decoder.
#
# Run:
#   ns-train dynamic-gs --data /path/to/dataset/with/post_fusion_state.pt

DEFAULT_DYNAMIC_RECORDED_STEPS = 5000
"""Arbitrary cap; trainer exits earlier when the dataset frames are
exhausted (or on Ctrl+C)."""

DynamicGS = MethodSpecification(
    config=TrainerConfig(
        _target=NoSaveTrainer,
        method_name="dynamic-gs",
        steps_per_eval_image=1_000_000_000,
        steps_per_eval_batch=0,
        steps_per_eval_all_images=1_000_000_000,
        steps_per_save=1_000_000_000,
        max_num_iterations=DEFAULT_DYNAMIC_RECORDED_STEPS,
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
        optimizers=_ZERO_LR_OPTIMIZERS,
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="tensorboard",
    ),
    description="Recorded-mode dynamic-gs: warm-loads static-gs cache, runs XFeat tracker + feedforward decoder on a recorded dataset.",
)


# ----------------------------------------------------------------------------
# dynamic-gs-live — live ROS-fed subclass of DynamicGSPipelineBase
# ----------------------------------------------------------------------------
#
# Same warm cache as dynamic-gs; frame source is the ROS publisher
# subprocess (live_ros_publisher.py in the dynamic_gs_ros conda env)
# feeding shared memory. Runs until 'stop' on stdin or Ctrl+C.
#
# Run:
#   ns-train dynamic-gs-live --data /path/to/dataset/with/post_fusion_state.pt

DEFAULT_DYNAMIC_LIVE_STEPS = 10**9
"""Effectively infinite — live mode terminates on user 'stop' / Ctrl+C
rather than hitting an iteration cap."""

DynamicGSLive = MethodSpecification(
    config=TrainerConfig(
        _target=NoSaveTrainer,
        method_name="dynamic-gs-live",
        steps_per_eval_image=1_000_000_000,
        steps_per_eval_batch=0,
        steps_per_eval_all_images=1_000_000_000,
        steps_per_save=1_000_000_000,
        max_num_iterations=DEFAULT_DYNAMIC_LIVE_STEPS,
        mixed_precision=False,
        pipeline=LiveDynamicGSPipelineConfig(
            datamanager=DynamicGSDataManagerConfig(),
            model=DynamicGSModelConfig(
                camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
                output_depth_during_training=True,
                stop_split_at=0,
                reuse_sam3d_generated_ply=True,
            ),
        ),
        optimizers=_ZERO_LR_OPTIMIZERS,
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="tensorboard",
    ),
    description="Live-ROS dynamic-gs: warm-loads static-gs cache, runs XFeat tracker + feedforward decoder against a ROS-fed live SHM stream (publisher auto-spawned).",
)
