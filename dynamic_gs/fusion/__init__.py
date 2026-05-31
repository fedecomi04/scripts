"""Phase 0 — SAM3 segmentation, SAM3D 3D object generation, and Phase 0b fusion.

This subpackage is the public seam between the pipeline and the fusion stack.
Today it re-exports the underlying ``dynamic_gs.utils.*`` modules so callers
can write ``from dynamic_gs.fusion import run_sam3_subprocess`` instead of
reaching into ``utils``. The implementation files will move out of ``utils/``
into this subpackage when the static-gs pipeline lands (Phase 2).
"""

from ..utils.sam3_segmentation import load_sam3_masks, run_sam3_subprocess
from ..utils.sam3d import (
    get_sam3d_output_paths,
    load_sam3d_pose,
    prepare_cropped_sam3d_inputs,
    resolve_sam3d_pose_path,
    run_sam3d_multi_object_subprocess,
    run_sam3d_single_object,
    run_sam3d_single_object_subprocess,
    sam3d_pose_has_rotation,
)
from ..utils.sam3d_fusion import (
    Sam3DInsertionResult,
    load_sam3d_gaussian_ply,
    load_sam3d_rotation_wxyz,
    reconstruct_mesh_from_gaussian_ply,
    reconstruct_mesh_from_points,
    register_and_fuse_sam3d_object,
    save_point_cloud,
)
from .phase0 import (
    backproject_mask_to_world,
    run_phase0a_sam3_and_sam3d,
    run_phase0b_fusion,
    save_sam3_debug_plots,
)

__all__ = [
    # Phase 0 driver (lifted from pipeline)
    "backproject_mask_to_world",
    "run_phase0a_sam3_and_sam3d",
    "run_phase0b_fusion",
    "save_sam3_debug_plots",
    # SAM3 segmentation
    "load_sam3_masks",
    "run_sam3_subprocess",
    # SAM3D generation
    "get_sam3d_output_paths",
    "load_sam3d_pose",
    "prepare_cropped_sam3d_inputs",
    "resolve_sam3d_pose_path",
    "run_sam3d_multi_object_subprocess",
    "run_sam3d_single_object",
    "run_sam3d_single_object_subprocess",
    "sam3d_pose_has_rotation",
    # SAM3D fusion (CPD + insertion)
    "Sam3DInsertionResult",
    "load_sam3d_gaussian_ply",
    "load_sam3d_rotation_wxyz",
    "reconstruct_mesh_from_gaussian_ply",
    "reconstruct_mesh_from_points",
    "register_and_fuse_sam3d_object",
    "save_point_cloud",
]
