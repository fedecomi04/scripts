"""static_sam3d.py — SAM3D 3D-object generation stage (subprocess wrap).

Drives the SAM3D handle (model_loader.Sam3dHandle) on the anchor + the FastSAM masks
to produce a per-object PLY + rigid-init pose. Residency policy (static_phase.md §2c):
SAM3D runs as a subprocess that dies after inference, 100%-freeing its ~13 GB for the
TSDF seed + splatfacto that follow — so nothing here holds GPU between uses.

Outputs land in segmentation/objects/ (obj_NN_sam3d_raw_output.ply + _pose.json), the
same self-describing folder static_segment owns. The orchestrator times this whole stage
as `trigger.sam3d_infer` (hidden under continued operator motion).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .static_segment import AnchorRef


def generate(anchor: AnchorRef, objects: List[dict], sam3d_handle) -> List[dict]:
    """Run SAM3D on every FastSAM mask against the anchor frame; return the proven
    per-object result dicts ({ply_path, pose_path, ...}; may contain {} for a failed
    object). Inputs come straight from the segmentation/ anchor (rgb + float32-m depth +
    intrinsics) so SAM3D sees exactly the frame the mask belongs to."""
    if not objects:
        return []
    objects_dir = anchor.seg_dir / "objects"
    objects_dir.mkdir(parents=True, exist_ok=True)
    mask_paths = [Path(o["mask_path"]) for o in objects]
    stems = [f"obj_{int(o['object_index']):02d}_sam3d" for o in objects]
    return sam3d_handle.generate(
        render_image_path=anchor.rgb_path,
        object_mask_paths=mask_paths,
        output_dir=objects_dir,
        output_stems=stems,
        depth_path=anchor.depth_path,            # float32 metres TIFF (SAM3D scale = 1.0)
        intrinsics_path=anchor.intrinsics_path)
    # The SAM3D subprocess exits here -> its ~13 GB is fully reclaimed for the TSDF
    # seed + splatfacto that run next on the now-free GPU.
