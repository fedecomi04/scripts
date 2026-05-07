#!/usr/bin/env python3
"""SAM 3D Objects — strategy comparison harness (raw-pose reprojection).

Goal
----
Two practical questions about the SAM 3D Objects (sam-3d-objects, "SAM3D")
inference pipeline this repo uses:

1. When you have several object masks in one image, is it worth using a
   "multi-object" call (one model load + a python loop over masks) vs
   running each mask through a fresh model load?
2. The SAM3D paper says you can supply a point map. The demo code never
   passes one — it lets MoGe estimate depth internally. Since this repo
   has metric depth + intrinsics for every frame, does feeding a real
   point map produce a noticeably better/aligned reconstruction?

This script runs the same set of (image, masks, depth, intrinsics) inputs
through 3 strategies, saves the resulting Gaussian splat PLYs in separate
subfolders, and writes a combined ``scene_with_objects.ply`` per strategy
where each object is placed in the SfM scene **using only SAM3D's
predicted pose** (rotation quaternion + translation + uniform scale). NO
probreg / CPD fusion runs — that lets us judge SAM3D's own pose
prediction directly, with no offline registration covering up errors.

Inputs come from ``static_scene/`` exclusively:

* RGB:    ``static_scene/rgb/<STATIC_FRAME_NAME>.png``  (real photograph)
* depth:  ``static_scene/depth/<STATIC_FRAME_NAME>.tiff`` (metric, mm)
* SfM:    ``static_scene/depth_camera_init_points.ply``  (visual backdrop)
* c2w:    first frame of ``static_scene/transforms.json``

The 3 strategies are:

    S1_cropped_moge      crop tightly around mask, MoGe estimates depth
                         (current production single-object path)
    S2_cropped_pointmap  crop tightly around mask, feed metric pointmap
                         built from real depth + intrinsics
    S4_full_pointmap     no crop (full image), feed full-image metric
                         pointmap built from real depth + intrinsics

The cropped/MoGe baseline is S1; the two pointmap variants test whether
real depth helps SAM3D produce a better-aligned 3D shape. The
full-image-without-pointmap variant ("S3") is removed because giving
SAM3D the entire 800x800 scene with a tiny per-object alpha mask and no
depth signal is strictly worse than at least one of the kept variants
(the model gets less object-relative context than the cropped path AND
no metric-depth supervision like the pointmap paths).

Within each strategy the SAM3D model is loaded once and reused across
all object masks. The model-load wall-clock time is reported separately,
so the cost of "fresh load per object" can be back-calculated as
``model_load_s * num_objects + sum(per_mask_inference_s)``.

How depth / pointmap is fed into the model
------------------------------------------
The official SAM3D demo (``notebook/demo_single_object.ipynb``) only
calls ``inference(image, mask, seed=42)`` and lets MoGe (a monocular
depth model, ``Ruicheng/moge-vitl``) estimate depth internally. The
``Inference.__call__`` signature in ``notebook/inference.py`` actually
also accepts a ``pointmap`` argument — when present, MoGe is bypassed
and SAM3D's preprocessor uses the supplied pointmap directly.

Required pointmap format:

* shape ``(H, W, 3)`` — same H/W as the (already preprocessed) image
  passed to ``inference``. If you pad / resize / crop the image, you
  must apply the same operations to the pointmap (we do this here via
  ``resize_pointmap``).
* axis convention is **PyTorch3D camera frame**, NOT standard CV:
  ``x`` points LEFT, ``y`` points UP, ``z`` points FORWARD. Standard
  CV backprojection produces ``(x-right, y-down, z-forward)``, so the
  conversion from a metric depth image + pinhole intrinsics is::

      x_p3d = -(u - cx) / fx * z
      y_p3d = -(v - cy) / fy * z
      z_p3d =  z

  where ``(u, v)`` are pixel coordinates and ``z`` is metric depth in
  metres. This is exactly what ``_build_pytorch3d_pointmap`` in
  ``dynamic_gs/utils/sam3d.py`` does, and that is the helper we reuse.
* invalid pixels (``z <= 0``) must be ``NaN``: SAM3D's preprocessor
  treats them as holes via ``_clip_pointmap``. Do NOT zero-fill
  invalid depth — that would create a flat plane at the camera origin.
* dtype is ``torch.float32``; the inference call casts internally.

The call site (see ``run_one_object`` below) is:

    inference(image_rgb, mask, seed=SEED, pointmap=pointmap_tensor)

The pipeline-runtime config used here is
``pipeline_runtime_small.yaml`` whose ``_target_`` is
``InferencePipelinePointMap`` — that is the variant of the pipeline
that knows how to consume an externally-supplied pointmap. (The
older ``InferencePipeline`` class would silently ignore the kwarg.)

Run from the `radiance_ros` conda env (the same one used by the rest of
the SAM3D subprocess code in `dynamic_gs.utils.sam3d`)::

    conda activate radiance_ros
    python old/test_sam3d_strategies.py

Edit the ``DATASET_ROOT`` / ``STATIC_FRAME_NAME`` constants below to
target a different dataset.

Outputs per strategy (under ``OUTPUT_ROOT/<strategy_name>/``):

    static0_obj_NN.ply          per-object SAM3D Gaussian output
    static0_obj_NN_pose.json    SAM3D pose decomposition (R, t, scale)
    static0_obj_NN_preview.png  RGB+mask preview fed to the model
    scene_with_objects.ply      SfM scene + each object placed using
                                ONLY SAM3D's predicted pose:
                                  p_world = (p_obj * scale) @ R + t
                                  p_world = p_world @ P3D_TO_NS
                                  p_world = p_world @ c2w_R.T + c2w_t
                                No CPD, no bbox-scale, no centroid match.
                                This is the file to inspect to compare
                                strategies.
"""

from __future__ import annotations

import os

# Ask PyTorch's caching allocator to grow segments on demand instead of
# failing when no single contiguous block is large enough. We saw obj_04
# (the largest mask) need a 5.63 GiB allocation that ~5.54 GiB of free
# memory could not satisfy because it was fragmented across smaller
# segments. ``expandable_segments:True`` is the recommended workaround
# in the OOM message PyTorch printed. MUST be set before ``import torch``
# so the allocator picks it up at CUDA init.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import json
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

# --- Inputs --------------------------------------------------------------

DATASET_ROOT = Path(
    "/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/"
    "dynamic_gs_test_2026-04-15_15-38-26"
)
# Real static-scene RGB photograph (NOT the rendered ``static0_rgb.png``
# under ``dynamic_scene/initialization_debug/``). SAM3D + the pointmap
# both come from the same static frame, so the predicted pose is
# expressed in the static-scene camera frame — exactly the c2w we read
# from ``static_scene/transforms.json[frames[0]]`` below.
STATIC_FRAME_NAME = "arm_00299"
STATIC_RGB_PATH = DATASET_ROOT / f"static_scene/rgb/{STATIC_FRAME_NAME}.png"
STATIC_DEPTH_PATH = DATASET_ROOT / f"static_scene/depth/{STATIC_FRAME_NAME}.tiff"
STATIC_TRANSFORMS_PATH = DATASET_ROOT / "static_scene/transforms.json"
# SfM seed point cloud used to initialize Gaussians for this dataset.
# It is the visual backdrop against which we judge each strategy's
# alignment quality.
SFM_PLY_PATH = DATASET_ROOT / "static_scene/depth_camera_init_points.ply"
# Index into transforms.json["frames"] for the frame static0_rgb.png
# was rendered from. The dataset-collection pipeline sorts frames by
# filename before writing static0, so 0 is correct for this dataset.
STATIC_FRAME_INDEX = 0
# Per-object SAM3 masks (5 in this dataset). Anything matching the
# pattern below is picked up; sort order is alphabetical so obj_00,
# obj_01, ... go through every strategy in the same order.
MASKS_GLOB = "static0_obj_[0-9][0-9]_mask.png"
MASKS_DIR = DATASET_ROOT / "dynamic_scene/initialization_debug"
# Restrict to a subset of object indices. SAM3 picks up some non-object
# regions (table edges, light fixtures, etc.) on this dataset; setting
# this to e.g. {0, 1, 4} drops those. ``None`` means "use every match
# of MASKS_GLOB".
KEEP_OBJECT_INDICES: set[int] | None = {0, 1, 4}

# When True, skip the SAM3D inference loop and reuse the per-object PLY +
# pose JSON files from the previous run (under ``OUTPUT_ROOT/<strategy>``).
# Useful when the goal is just to re-do the scene construction or
# reprojection without paying the ~5+ min SAM3D inference cost again.
REUSE_SAM3D_OUTPUTS: bool = True

# Max valid depth (m) when back-projecting the static depth to scene
# points; anything beyond is treated as "no return" / sky / outlier.
SCENE_MAX_DEPTH_M: float = 5.0

# Pixel stride when back-projecting the per-frame depth into the merged
# scene cloud. With 80 static frames at 800x800 = ~50M points, stride=2
# gives ~12M which is plenty for visual inspection and keeps the PLY
# under ~250 MB. Set to 1 for full density, 4 for a much lighter PLY.
SCENE_PIXEL_STRIDE: int = 2

# Where to write the strategy outputs. Each strategy gets its own subdir.
OUTPUT_ROOT = DATASET_ROOT / "dynamic_scene/sam3d_strategy_test"

# SAM3D inference resolution (matches the rest of the pipeline).
MAX_SIDE = 518

# Random seed for SAM3D (kept constant across strategies so any quality
# difference comes from inputs, not from the diffusion sampler).
SEED = 42

# --- Pull repo-internal helpers ------------------------------------------
# These all live in dynamic_gs.utils.sam3d. Re-using them keeps the test
# in lock-step with what production actually does — if those helpers
# change, the test reflects the change.

REPO_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(REPO_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_SCRIPTS_DIR))

from dynamic_gs.utils.sam3d import (
    _build_pytorch3d_pointmap,
    _import_official_api,
    _load_binary_mask,
    _resize_image_and_mask,
    _save_preview,
    _write_runtime_config,
    prepare_cropped_sam3d_inputs,
)
from dynamic_gs.utils.sam3d_fusion import (
    load_sam3d_gaussian_ply,
    save_point_cloud,
)

# pytorch3d camera frame: x-LEFT, y-UP, z-FORWARD.
# nerfstudio (OpenGL) camera frame: x-RIGHT, y-UP, z-BACKWARD.
# So the conversion flips x and z signs — same constant production fusion
# uses (``SAM3D_P3D_TO_NS_CAMERA`` in ``sam3d_fusion.py``).
SAM3D_P3D_TO_NS_CAMERA = np.diag([-1.0, 1.0, -1.0]).astype(np.float32)

# --- Strategy definition --------------------------------------------------


@dataclass
class StrategyConfig:
    name: str
    crop_input: bool          # crop tightly around mask before SAM3D inference
    use_pointmap: bool        # pass metric pointmap (vs let MoGe estimate)


STRATEGIES: list[StrategyConfig] = [
    StrategyConfig("S4_full_pointmap",     crop_input=False, use_pointmap=True),
]


@dataclass
class ObjectResult:
    object_stem: str
    inference_seconds: float
    num_gaussians: int
    pose_translation: list[float]
    pose_rotation: list[float]
    pose_scale: list[float]
    ply_path: str
    error: Optional[str] = None


@dataclass
class StrategyResult:
    name: str
    crop_input: bool
    use_pointmap: bool
    model_load_seconds: float
    total_inference_seconds: float
    object_results: list[ObjectResult] = field(default_factory=list)
    error: Optional[str] = None
    combined_ply_path: Optional[str] = None
    fusion_seconds: float = 0.0


# --- Helpers --------------------------------------------------------------


def load_intrinsics() -> dict:
    transforms = json.loads(STATIC_TRANSFORMS_PATH.read_text())
    return {
        "fx": float(transforms["fl_x"]),
        "fy": float(transforms["fl_y"]),
        "cx": float(transforms["cx"]),
        "cy": float(transforms["cy"]),
        "width": int(transforms["w"]),
        "height": int(transforms["h"]),
    }


def load_depth_meters(depth_path: Path, scale: float = 1e-3) -> np.ndarray:
    arr = np.array(Image.open(depth_path)).astype(np.float32)
    return arr * float(scale)


def find_masks() -> list[Path]:
    paths = sorted(MASKS_DIR.glob(MASKS_GLOB))
    if not paths:
        raise FileNotFoundError(
            f"No object masks matched {MASKS_GLOB} in {MASKS_DIR}"
        )
    if KEEP_OBJECT_INDICES is not None:
        def _idx(p: Path) -> int:
            # static0_obj_NN_mask.png -> NN
            return int(p.stem.split("_")[2])
        kept = [p for p in paths if _idx(p) in KEEP_OBJECT_INDICES]
        if not kept:
            raise FileNotFoundError(
                f"No mask paths match KEEP_OBJECT_INDICES={KEEP_OBJECT_INDICES}"
            )
        return kept
    return paths


def stem_for_mask(mask_path: Path) -> str:
    # static0_obj_03_mask.png -> static0_obj_03
    return mask_path.stem.replace("_mask", "")


def resize_pointmap(pointmap_full: np.ndarray, target_hw: tuple[int, int]) -> torch.Tensor:
    """Nearest-resize a pytorch3d pointmap to ``target_hw`` and return as torch tensor (HxWx3)."""
    pm = torch.from_numpy(pointmap_full).permute(2, 0, 1).unsqueeze(0)
    pm = torch.nn.functional.interpolate(pm, size=target_hw, mode="nearest")
    return pm.squeeze(0).permute(1, 2, 0).contiguous()


def write_pose_sidecar(pose_path: Path, output: dict) -> tuple[list[float], list[float], list[float]]:
    pose_data: dict[str, list[float]] = {}
    for key in ("translation", "rotation", "scale"):
        value = output.get(key)
        if value is not None:
            pose_data[key] = torch.as_tensor(value).detach().cpu().reshape(-1).tolist()
    pose_path.write_text(json.dumps(pose_data, indent=2) + "\n")
    return (
        pose_data.get("translation", []),
        pose_data.get("rotation", []),
        pose_data.get("scale", []),
    )


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- Scene / reprojection helpers ----------------------------------------
#
# Reprojection here uses ONLY SAM3D's predicted pose (R, t, scale) — no
# probreg / CPD / bbox-scale / centroid alignment. That isolates SAM3D's
# own metric prediction quality, which is the thing we are trying to
# evaluate in this debug harness.


def load_first_frame_c2w() -> np.ndarray:
    """Return the (4, 4) camera-to-world transform for the static frame
    whose RGB and depth feed SAM3D. The 3x3 rotation block + translation
    are used to map the per-object cloud (in the SAM3D pytorch3d camera
    frame, after axis flip) into world coordinates."""
    transforms = json.loads(STATIC_TRANSFORMS_PATH.read_text())
    frame = transforms["frames"][STATIC_FRAME_INDEX]
    c2w = np.asarray(frame["transform_matrix"], dtype=np.float32)
    if c2w.shape == (3, 4):
        bottom = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        c2w = np.concatenate([c2w, bottom[None, :]], axis=0)
    return c2w


def load_sfm_scene_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the static-scene SfM seed PLY and return ``(xyz, rgb)``.

    Colors fall back to mid-grey when the PLY has no RGB attribute; this
    only affects how the merged PLY looks in the viewer."""
    import open3d as o3d  # type: ignore

    cloud = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(cloud.points, dtype=np.float32)
    if cloud.has_colors():
        rgb = np.asarray(cloud.colors, dtype=np.float32)
    else:
        rgb = np.full((xyz.shape[0], 3), 0.5, dtype=np.float32)
    return xyz, rgb


def build_scene_from_all_static_frames(
    transforms_path: Path,
    intrinsics: dict,
    depth_scale: float = 1e-3,
    max_depth_m: float = 5.0,
    stride: int = 1,
    use_frame_masks: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project every frame in ``static_scene/transforms.json`` and
    concatenate into a single world-space cloud.

    Each frame contributes the points its depth image projects to (with
    a stride to keep the cloud compact), filtered to pixels where the
    frame's ``mask_path`` is "keep" (>127). The dataset convention is
    ``mask=255`` for scene/background to keep, ``mask=0`` for the
    robot/gripper to exclude. Colors are sampled from the same frame's
    RGB. Camera convention is Nerfstudio/OpenGL — same as ``c2w`` in
    transforms.json.

    Set ``use_frame_masks=False`` to ignore the mask sidecars.
    """
    transforms = json.loads(Path(transforms_path).read_text())
    frames = transforms.get("frames", [])
    if not frames:
        raise RuntimeError(f"No frames in {transforms_path}")

    base_dir = Path(transforms_path).parent

    fx0 = float(intrinsics["fx"])
    fy0 = float(intrinsics["fy"])
    cx0 = float(intrinsics["cx"])
    cy0 = float(intrinsics["cy"])
    src_W = int(intrinsics["width"])
    src_H = int(intrinsics["height"])

    xyz_chunks: list[np.ndarray] = []
    rgb_chunks: list[np.ndarray] = []
    skipped = 0
    for frame in frames:
        rgb_rel = frame.get("file_path")
        depth_rel = frame.get("depth_file_path")
        c2w_list = frame.get("transform_matrix")
        if rgb_rel is None or depth_rel is None or c2w_list is None:
            skipped += 1
            continue
        rgb_path = (base_dir / rgb_rel).resolve()
        depth_path = (base_dir / depth_rel).resolve()
        if not rgb_path.exists() or not depth_path.exists():
            skipped += 1
            continue

        try:
            depth_raw = np.array(Image.open(depth_path)).astype(np.float32)
            depth_m = depth_raw * float(depth_scale)
            rgb = np.array(Image.open(rgb_path).convert("RGB"))
        except Exception:
            skipped += 1
            continue

        H, W = depth_m.shape[:2]
        if rgb.shape[:2] != (H, W):
            rgb = np.array(Image.fromarray(rgb).resize((W, H), Image.BILINEAR))
        rgb_f = rgb.astype(np.float32) / 255.0

        # Per-frame keep-mask: dataset convention is mask=255 for scene
        # (keep) and mask=0 for the robot/gripper (exclude). Sample the
        # path declared in transforms.json[frames[i]]["mask_path"].
        keep_mask: np.ndarray | None = None
        if use_frame_masks:
            mask_rel = frame.get("mask_path")
            if mask_rel is not None:
                mask_path = (base_dir / mask_rel).resolve()
                if mask_path.exists():
                    try:
                        m_arr = np.array(Image.open(mask_path).convert("L"))
                        if m_arr.shape[:2] != (H, W):
                            m_arr = np.array(
                                Image.fromarray(m_arr).resize((W, H), Image.NEAREST)
                            )
                        keep_mask = m_arr > 127
                    except Exception:
                        keep_mask = None

        c2w = np.asarray(c2w_list, dtype=np.float32)
        if c2w.shape == (3, 4):
            bottom = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            c2w = np.concatenate([c2w, bottom[None, :]], axis=0)

        # Stride pixel sampling to keep the merged cloud small.
        ys_grid = np.arange(0, H, max(1, stride), dtype=np.int32)
        xs_grid = np.arange(0, W, max(1, stride), dtype=np.int32)
        yy, xx = np.meshgrid(ys_grid, xs_grid, indexing="ij")
        ys_flat = yy.reshape(-1)
        xs_flat = xx.reshape(-1)
        z_all = depth_m[ys_flat, xs_flat]
        valid = (z_all > 1e-4) & (z_all < float(max_depth_m))
        if keep_mask is not None:
            valid &= keep_mask[ys_flat, xs_flat]
        if not valid.any():
            continue
        ys = ys_flat[valid]
        xs = xs_flat[valid]
        z = z_all[valid]

        if (H, W) != (src_H, src_W):
            sx = W / float(src_W)
            sy = H / float(src_H)
            fx = fx0 * sx
            fy = fy0 * sy
            cx = cx0 * sx
            cy = cy0 * sy
        else:
            fx, fy, cx, cy = fx0, fy0, cx0, cy0

        x_cam = (xs.astype(np.float32) - cx) / fx * z
        y_cam = -(ys.astype(np.float32) - cy) / fy * z
        z_cam = -z
        pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

        R = c2w[:3, :3].astype(np.float32)
        t = c2w[:3, 3].astype(np.float32)
        pts_world = pts_cam @ R.T + t[None, :]
        xyz_chunks.append(pts_world.astype(np.float32))
        rgb_chunks.append(rgb_f[ys, xs].astype(np.float32))

    if skipped:
        print(f"  [scene-build] skipped {skipped} frames (missing/unreadable)")

    if not xyz_chunks:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    return (
        np.concatenate(xyz_chunks, axis=0).astype(np.float32),
        np.concatenate(rgb_chunks, axis=0).astype(np.float32),
    )


def _dilate_mask(mask_bool: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation of a (H, W) bool mask by ``radius`` pixels using a
    square structuring element. Pure-numpy max-filter via padding shifts
    so we don't pull in scipy as a hard dependency."""
    if radius <= 0:
        return mask_bool
    out = mask_bool.copy()
    for _ in range(int(radius)):
        shifted = np.zeros_like(out)
        shifted[1:, :] |= out[:-1, :]
        shifted[:-1, :] |= out[1:, :]
        shifted[:, 1:] |= out[:, :-1]
        shifted[:, :-1] |= out[:, 1:]
        out |= shifted
    return out


def build_static_scene_pointcloud(
    depth_path: Path,
    rgb_path: Path,
    intrinsics: dict,
    c2w: np.ndarray,
    exclude_masks: list[np.ndarray] | None = None,
    depth_scale: float = 1e-3,
    max_depth_m: float = 5.0,
    dilate_exclude_px: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project ``depth_path`` (uint16, scaled to metres) through
    ``intrinsics`` and ``c2w`` to produce a world-space point cloud,
    colored by sampling ``rgb_path`` at each surviving pixel.

    ``exclude_masks`` is a list of (H, W) bool arrays — pixels covered
    by ANY of them are dropped from the cloud (used to remove the
    graspable-object regions, since those will be filled by SAM3D's
    per-object outputs). Each excluded mask is dilated by
    ``dilate_exclude_px`` before union, to avoid object-edge halos.

    Camera convention is Nerfstudio/OpenGL (x-right, y-up, z-backward),
    matching the c2w stored in ``static_scene/transforms.json``.
    """
    depth_raw = np.array(Image.open(depth_path)).astype(np.float32)
    depth_m = depth_raw * float(depth_scale)
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    H, W = depth_m.shape[:2]
    if rgb.shape[:2] != (H, W):
        rgb = np.array(Image.fromarray(rgb).resize((W, H), Image.BILINEAR))
    rgb_f = rgb.astype(np.float32) / 255.0

    valid = (depth_m > 1e-4) & (depth_m < float(max_depth_m))
    if exclude_masks:
        exclude_union = np.zeros((H, W), dtype=bool)
        for m in exclude_masks:
            if m.shape[:2] != (H, W):
                m_resized = (
                    np.array(
                        Image.fromarray(m.astype(np.uint8) * 255).resize(
                            (W, H), Image.NEAREST
                        )
                    )
                    > 127
                )
                m = m_resized
            exclude_union |= m.astype(bool)
        if dilate_exclude_px > 0:
            exclude_union = _dilate_mask(exclude_union, dilate_exclude_px)
        valid &= ~exclude_union

    ys, xs = np.where(valid)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    z = depth_m[ys, xs]

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    src_W = int(intrinsics["width"])
    src_H = int(intrinsics["height"])
    if (H, W) != (src_H, src_W):
        sx = W / float(src_W)
        sy = H / float(src_H)
        fx *= sx
        fy *= sy
        cx *= sx
        cy *= sy

    # Nerfstudio / OpenGL camera: x right, y up, z backward.
    x_cam = (xs.astype(np.float32) - cx) / fx * z
    y_cam = -(ys.astype(np.float32) - cy) / fy * z
    z_cam = -z
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

    R = c2w[:3, :3].astype(np.float32)
    t = c2w[:3, 3].astype(np.float32)
    pts_world = pts_cam @ R.T + t[None, :]
    return pts_world.astype(np.float32), rgb_f[ys, xs].astype(np.float32)


def _quaternion_wxyz_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    """Right-multiplication-convention rotation matrix for a wxyz quaternion.

    Matches ``_quaternion_wxyz_to_rotation_matrix`` in
    ``dynamic_gs/utils/sam3d_fusion.py`` (``points @ R`` to rotate
    row-vector points), so SAM3D's predicted rotation is applied the
    same way here as in production fusion.
    """
    w, x, y, z = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float32)
    w /= norm
    x /= norm
    y /= norm
    z /= norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def apply_sam3d_pose_to_world(
    source_points: np.ndarray,
    pose: dict,
    c2w: np.ndarray,
) -> np.ndarray:
    """Place SAM3D source points in world coordinates using only the
    predicted pose (no CPD / no bbox-scale / no centroid alignment).

    Steps:
      1. p_camera_p3d = (p_obj * scale) @ R + t   — SAM3D's prediction
      2. p_camera_ns  = p_camera_p3d @ P3D_TO_NS  — pytorch3d → nerfstudio
      3. p_world      = p_camera_ns @ c2w_R.T + c2w_t — camera → world

    Translation/scale are the metric values SAM3D predicts when fed a
    metric pointmap; for MoGe runs they are scale-and-shift invariant
    and the reprojection will look wrong (which is the point of the
    comparison).
    """
    R = _quaternion_wxyz_to_rotmat(pose["rotation"])
    t = np.asarray(pose["translation"], dtype=np.float32).reshape(3)
    scale_arr = np.asarray(pose["scale"], dtype=np.float32).reshape(-1)
    s = float(scale_arr[0]) if scale_arr.size > 0 else 1.0

    p_p3d = (source_points.astype(np.float32) * s) @ R + t[None, :]
    p_ns = p_p3d @ SAM3D_P3D_TO_NS_CAMERA

    c2w_R = np.asarray(c2w[:3, :3], dtype=np.float32)
    c2w_t = np.asarray(c2w[:3, 3], dtype=np.float32)
    return (p_ns @ c2w_R.T + c2w_t[None, :]).astype(np.float32)


def reproject_strategy_into_scene(
    strategy_dir: Path,
    object_results: list["ObjectResult"],
    c2w: np.ndarray,
    scene_xyz: np.ndarray,
    scene_rgb: np.ndarray,
) -> Path | None:
    """For one strategy, place every successful per-object PLY in the
    SfM scene using only SAM3D's predicted pose, and write the union as
    ``<strategy_dir>/scene_with_objects.ply``.
    """
    merged_xyz = [scene_xyz]
    merged_rgb = [scene_rgb]
    placed_count = 0
    for result in object_results:
        if result.error is not None or result.num_gaussians == 0:
            print(f"  [reproject-skip] {result.object_stem}: {result.error or 'empty'}")
            continue
        ply_path = Path(result.ply_path)
        pose_path = strategy_dir / f"{result.object_stem}_pose.json"
        if not ply_path.exists() or not pose_path.exists():
            print(f"  [reproject-skip] {result.object_stem}: missing PLY or pose")
            continue
        try:
            source_xyz, source_rgb = load_sam3d_gaussian_ply(ply_path)
            pose = json.loads(pose_path.read_text())
        except Exception as exc:
            print(f"  [reproject-skip] {result.object_stem}: load failed: {exc}")
            continue
        if "rotation" not in pose or "translation" not in pose or "scale" not in pose:
            print(f"  [reproject-skip] {result.object_stem}: pose JSON missing R/t/scale")
            continue

        try:
            world_xyz = apply_sam3d_pose_to_world(source_xyz, pose, c2w)
        except Exception as exc:
            print(f"  [reproject-err] {result.object_stem}: {exc}")
            continue

        merged_xyz.append(world_xyz)
        merged_rgb.append(source_rgb)
        placed_count += 1
        s_val = float(np.asarray(pose["scale"]).reshape(-1)[0])
        print(
            f"  [reproject-ok] {result.object_stem}: "
            f"N={len(world_xyz)}, scale={s_val:.4f}"
        )

    if placed_count == 0:
        print(f"  no objects placed for {strategy_dir.name}")
        return None

    out_path = strategy_dir / "scene_with_objects.ply"
    save_point_cloud(
        out_path,
        np.concatenate(merged_xyz, axis=0),
        np.concatenate(merged_rgb, axis=0),
    )
    print(f"  combined scene+objects PLY written: {out_path}")
    return out_path


# --- Per-mask inference ---------------------------------------------------


def run_one_object(
    inference,
    cfg: StrategyConfig,
    mask_path: Path,
    full_image_rgb: np.ndarray,
    full_pointmap: np.ndarray | None,
    full_intrinsics: dict,
    out_dir: Path,
) -> ObjectResult:
    """Run SAM3D for one (image, mask) pair under the given strategy."""

    object_stem = stem_for_mask(mask_path)
    ply_path = out_dir / f"{object_stem}.ply"
    pose_path = out_dir / f"{object_stem}_pose.json"
    preview_path = out_dir / f"{object_stem}_preview.png"

    H, W = full_image_rgb.shape[:2]
    full_mask = _load_binary_mask(mask_path, (W, H))
    if int(full_mask.sum()) == 0:
        return ObjectResult(
            object_stem=object_stem,
            inference_seconds=0.0,
            num_gaussians=0,
            pose_translation=[], pose_rotation=[], pose_scale=[],
            ply_path=str(ply_path),
            error="empty mask",
        )

    # Pre-resize source: pull the largest image / mask / pointmap available,
    # so the retry loop can downsize as needed without redoing crop I/O.
    if cfg.crop_input:
        cropped_paths = prepare_cropped_sam3d_inputs(
            render_image_path=STATIC_RGB_PATH,
            object_mask_path=mask_path,
            output_dir=out_dir / "_crops",
            output_stem=object_stem,
            image_dir=out_dir / "_crops",
            depth_path=STATIC_DEPTH_PATH if cfg.use_pointmap else None,
            depth_scale=1e-3,
            camera_intrinsics=full_intrinsics if cfg.use_pointmap else None,
        )
        src_image_rgb = np.array(Image.open(cropped_paths["render_image_path"]).convert("RGB"))
        src_mask = _load_binary_mask(cropped_paths["object_mask_path"], src_image_rgb.shape[:2][::-1])
        if cfg.use_pointmap:
            cropped_intrinsics = json.loads(Path(cropped_paths["intrinsics_path"]).read_text())
            cropped_depth_m = np.array(
                Image.open(cropped_paths["depth_path"])
            ).astype(np.float32)
            src_pointmap: np.ndarray | None = _build_pytorch3d_pointmap(cropped_depth_m, cropped_intrinsics)
        else:
            src_pointmap = None
    else:
        src_image_rgb = full_image_rgb
        src_mask = full_mask
        src_pointmap = full_pointmap if cfg.use_pointmap else None

    # Retry loop: SAM3D's first allocation scales ~quadratically with image
    # side. obj_04 needs ~5.6 GiB at max_side=518 which is just over the
    # ~5.5 GiB free on this 8 GiB GPU once the model is loaded. Dropping
    # to 384 cuts that to ~3.1 GiB; 256 to ~1.4 GiB. The sleep+cleanup
    # helps the allocator coalesce free segments between attempts.
    candidate_sizes: list[int] = [MAX_SIDE]
    for fallback in (384, 256):
        if fallback < MAX_SIDE and fallback not in candidate_sizes:
            candidate_sizes.append(fallback)

    output = None
    inference_seconds = 0.0
    last_oom: Optional[Exception] = None
    used_max_side: Optional[int] = None
    image_rgb = src_image_rgb
    mask = src_mask
    pointmap_for_inference: Optional[torch.Tensor] = None
    for size in candidate_sizes:
        free_gpu()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time.sleep(1.0)

        image_rgb, mask = _resize_image_and_mask(src_image_rgb, src_mask, max_side=size)
        if src_pointmap is not None:
            pointmap_for_inference = resize_pointmap(src_pointmap, image_rgb.shape[:2])
        else:
            pointmap_for_inference = None
        _save_preview(mask, image_rgb, preview_path)

        t0 = time.time()
        try:
            if pointmap_for_inference is not None:
                output = inference(image_rgb, mask, seed=SEED, pointmap=pointmap_for_inference)
            else:
                output = inference(image_rgb, mask, seed=SEED)
            inference_seconds += time.time() - t0
            used_max_side = size
            if size != MAX_SIDE:
                print(
                    f"  [oom-fallback] {object_stem}: succeeded at max_side={size} "
                    f"(default was {MAX_SIDE})"
                )
            break
        except torch.cuda.OutOfMemoryError as exc:
            inference_seconds += time.time() - t0
            last_oom = exc
            print(f"  [oom-retry] {object_stem}: max_side={size} OOM'd, falling back")
            output = None
            free_gpu()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            time.sleep(1.0)
            continue
        except Exception as exc:
            return ObjectResult(
                object_stem=object_stem,
                inference_seconds=inference_seconds + (time.time() - t0),
                num_gaussians=0,
                pose_translation=[], pose_rotation=[], pose_scale=[],
                ply_path=str(ply_path),
                error=f"{type(exc).__name__}: {exc}",
            )

    if output is None:
        return ObjectResult(
            object_stem=object_stem,
            inference_seconds=inference_seconds,
            num_gaussians=0,
            pose_translation=[], pose_rotation=[], pose_scale=[],
            ply_path=str(ply_path),
            error=(
                f"OOM at all candidate sizes {candidate_sizes}: "
                f"{type(last_oom).__name__ if last_oom else 'unknown'}: {last_oom}"
            ),
        )

    if "gs" not in output:
        return ObjectResult(
            object_stem=object_stem,
            inference_seconds=inference_seconds,
            num_gaussians=0,
            pose_translation=[], pose_rotation=[], pose_scale=[],
            ply_path=str(ply_path),
            error="no 'gs' key in output",
        )

    output["gs"].save_ply(str(ply_path))
    num_g = int(output["gs"].get_xyz.shape[0])
    translation, rotation, scale = write_pose_sidecar(pose_path, output)

    # Drop the heavy output dict before the next iteration.
    del output
    free_gpu()

    return ObjectResult(
        object_stem=object_stem,
        inference_seconds=inference_seconds,
        num_gaussians=num_g,
        pose_translation=translation,
        pose_rotation=rotation,
        pose_scale=scale,
        ply_path=str(ply_path),
    )


# --- Reuse-existing-outputs path -----------------------------------------
#
# When ``REUSE_SAM3D_OUTPUTS=True``, skip the SAM3D model load + inference
# loop entirely and just reconstruct ``ObjectResult`` entries from the
# per-object PLY + pose JSON files written by the previous run. Used when
# we want to iterate on the scene-construction or reprojection stages
# without paying the ~5+ min SAM3D cost again.


def gather_existing_strategy_outputs(
    cfg: StrategyConfig,
    mask_paths: list[Path],
) -> StrategyResult:
    out_dir = OUTPUT_ROOT / cfg.name
    print(f"\n=== Reusing existing outputs for {cfg.name} ===")
    if not out_dir.is_dir():
        return StrategyResult(
            name=cfg.name,
            crop_input=cfg.crop_input,
            use_pointmap=cfg.use_pointmap,
            model_load_seconds=0.0,
            total_inference_seconds=0.0,
            error=f"reuse path missing: {out_dir}",
        )

    object_results: list[ObjectResult] = []
    for mask_path in mask_paths:
        object_stem = stem_for_mask(mask_path)
        ply_path = out_dir / f"{object_stem}.ply"
        pose_path = out_dir / f"{object_stem}_pose.json"
        if not ply_path.exists() or not pose_path.exists():
            object_results.append(
                ObjectResult(
                    object_stem=object_stem,
                    inference_seconds=0.0,
                    num_gaussians=0,
                    pose_translation=[], pose_rotation=[], pose_scale=[],
                    ply_path=str(ply_path),
                    error=f"missing PLY or pose for reuse",
                )
            )
            print(f"  [reuse-skip] {object_stem}: missing PLY or pose")
            continue
        try:
            xyz, _rgb = load_sam3d_gaussian_ply(ply_path)
            num_g = int(len(xyz))
        except Exception:
            num_g = 0
        try:
            pose = json.loads(pose_path.read_text())
        except Exception:
            pose = {}
        translation = list(pose.get("translation") or [])
        rotation = list(pose.get("rotation") or [])
        scale = list(pose.get("scale") or [])
        object_results.append(
            ObjectResult(
                object_stem=object_stem,
                inference_seconds=0.0,
                num_gaussians=num_g,
                pose_translation=translation,
                pose_rotation=rotation,
                pose_scale=scale,
                ply_path=str(ply_path),
            )
        )
        print(f"  [reuse-ok] {object_stem}: {num_g} gaussians from cached PLY")

    return StrategyResult(
        name=cfg.name,
        crop_input=cfg.crop_input,
        use_pointmap=cfg.use_pointmap,
        model_load_seconds=0.0,
        total_inference_seconds=0.0,
        object_results=object_results,
    )


# --- Strategy runner ------------------------------------------------------


def run_strategy(
    cfg: StrategyConfig,
    mask_paths: list[Path],
    full_image_rgb: np.ndarray,
    full_pointmap: np.ndarray | None,
    full_intrinsics: dict,
) -> StrategyResult:
    print(f"\n=== Running {cfg.name} (crop={cfg.crop_input}, pointmap={cfg.use_pointmap}) ===")
    out_dir = OUTPUT_ROOT / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    runtime_config_path = _write_runtime_config()
    Inference = _import_official_api()

    free_gpu()
    t_load_0 = time.time()
    try:
        inference = Inference(str(runtime_config_path), compile=False)
    except Exception as exc:
        return StrategyResult(
            name=cfg.name,
            crop_input=cfg.crop_input,
            use_pointmap=cfg.use_pointmap,
            model_load_seconds=time.time() - t_load_0,
            total_inference_seconds=0.0,
            error=f"model load failed: {type(exc).__name__}: {exc}",
        )
    model_load_seconds = time.time() - t_load_0
    print(f"  model loaded in {model_load_seconds:.1f}s")

    object_results: list[ObjectResult] = []
    total_infer = 0.0
    try:
        for mask_path in mask_paths:
            result = run_one_object(
                inference=inference,
                cfg=cfg,
                mask_path=mask_path,
                full_image_rgb=full_image_rgb,
                full_pointmap=full_pointmap,
                full_intrinsics=full_intrinsics,
                out_dir=out_dir,
            )
            total_infer += result.inference_seconds
            object_results.append(result)
            if result.error is None:
                print(
                    f"  [ok] {result.object_stem}: "
                    f"{result.num_gaussians:>6d} gaussians, "
                    f"{result.inference_seconds:.1f}s"
                )
            else:
                print(f"  [err] {result.object_stem}: {result.error}")
    finally:
        del inference
        free_gpu()

    return StrategyResult(
        name=cfg.name,
        crop_input=cfg.crop_input,
        use_pointmap=cfg.use_pointmap,
        model_load_seconds=model_load_seconds,
        total_inference_seconds=total_infer,
        object_results=object_results,
    )


# --- Summary --------------------------------------------------------------


def write_summary(results: list[StrategyResult], path: Path) -> None:
    lines: list[str] = ["SAM3D strategy comparison summary"]
    lines.append("=" * 72)
    lines.append(f"dataset: {DATASET_ROOT}")
    lines.append(f"static_rgb: {STATIC_RGB_PATH.name}")
    lines.append(f"depth: {STATIC_DEPTH_PATH.name}")
    lines.append(f"max_side: {MAX_SIDE}, seed: {SEED}")
    lines.append("")

    # Per-strategy header
    lines.append(
        f"{'strategy':<26}{'load_s':>10}{'infer_s':>10}{'fuse_s':>10}"
        f"{'mean/obj_s':>14}{'objs_ok':>10}"
    )
    for r in results:
        n_ok = sum(1 for o in r.object_results if o.error is None)
        n_total = len(r.object_results)
        mean_inf = (r.total_inference_seconds / n_total) if n_total else 0.0
        lines.append(
            f"{r.name:<26}"
            f"{r.model_load_seconds:>10.1f}"
            f"{r.total_inference_seconds:>10.1f}"
            f"{r.fusion_seconds:>10.1f}"
            f"{mean_inf:>14.1f}"
            f"{n_ok:>5}/{n_total:<4}"
        )
    lines.append("")

    # Combined PLYs (the artifacts to inspect visually)
    lines.append("Combined scene+objects PLYs:")
    for r in results:
        if r.combined_ply_path:
            lines.append(f"  {r.name:<26} {r.combined_ply_path}")
        else:
            lines.append(f"  {r.name:<26} (no combined PLY)")
    lines.append("")

    # Per-object pose comparison: shows scale/translation/rotation as a
    # quick sanity check that the strategies are producing meaningfully
    # different (or same) layouts. Numbers are SAM3D's predicted
    # decomposed pose for each object.
    object_stems = sorted({
        o.object_stem for r in results for o in r.object_results
    })
    for stem in object_stems:
        lines.append(f"--- object {stem} ---")
        for r in results:
            o = next((x for x in r.object_results if x.object_stem == stem), None)
            if o is None:
                continue
            if o.error is not None:
                lines.append(f"  {r.name:<26} ERROR: {o.error}")
                continue
            t = ", ".join(f"{v:+.3f}" for v in o.pose_translation) if o.pose_translation else ""
            s = ", ".join(f"{v:.3f}" for v in o.pose_scale) if o.pose_scale else ""
            lines.append(
                f"  {r.name:<26} N={o.num_gaussians:>6}  "
                f"t=[{t}]  s=[{s}]"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n")
    print(f"\nSummary written to {path}")


# --- Main -----------------------------------------------------------------


def main() -> int:
    if not STATIC_RGB_PATH.exists():
        print(f"ERROR: missing static RGB: {STATIC_RGB_PATH}", file=sys.stderr)
        return 1
    if not STATIC_DEPTH_PATH.exists():
        print(f"ERROR: missing depth: {STATIC_DEPTH_PATH}", file=sys.stderr)
        return 1
    if not STATIC_TRANSFORMS_PATH.exists():
        print(f"ERROR: missing transforms: {STATIC_TRANSFORMS_PATH}", file=sys.stderr)
        return 1

    if not SFM_PLY_PATH.exists():
        print(f"ERROR: missing SfM scene PLY: {SFM_PLY_PATH}", file=sys.stderr)
        return 1

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    intrinsics = load_intrinsics()
    full_image_rgb = np.array(Image.open(STATIC_RGB_PATH).convert("RGB"))
    H, W = full_image_rgb.shape[:2]
    if (W, H) != (intrinsics["width"], intrinsics["height"]):
        print(
            f"WARNING: intrinsics ({intrinsics['width']}x{intrinsics['height']}) "
            f"mismatch RGB ({W}x{H}). Using RGB shape; intrinsics fx/fy still apply."
        )

    # Pointmap is built from STATIC_DEPTH_PATH which resolves to
    # ``static_scene/depth/<STATIC_FRAME_NAME>.tiff``. No dynamic-scene
    # file is ever read for the pointmap. (Dynamic depths in this dataset
    # start at arm_00552.tiff onward — well after STATIC_FRAME_NAME.)
    assert "static_scene" in str(STATIC_DEPTH_PATH), (
        f"STATIC_DEPTH_PATH must be inside static_scene/, got {STATIC_DEPTH_PATH}"
    )
    depth_m = load_depth_meters(STATIC_DEPTH_PATH, scale=1e-3)
    if depth_m.shape != (H, W):
        depth_pil = Image.fromarray(depth_m)
        depth_pil = depth_pil.resize((W, H), Image.NEAREST)
        depth_m = np.array(depth_pil, dtype=np.float32)
    full_pointmap = _build_pytorch3d_pointmap(depth_m, intrinsics)

    c2w = load_first_frame_c2w()

    mask_paths = find_masks()
    print(f"Found {len(mask_paths)} object masks:")
    for p in mask_paths:
        print(f"  - {p.name}")

    # Build the scene point cloud by back-projecting EVERY frame in
    # static_scene/transforms.json. SAM3D's reprojected objects are
    # appended on top in ``reproject_strategy_into_scene``; we don't
    # mask their regions out of the scene cloud because the user wants
    # the full per-frame coverage as the backdrop.
    scene_xyz, scene_rgb = build_scene_from_all_static_frames(
        transforms_path=STATIC_TRANSFORMS_PATH,
        intrinsics=intrinsics,
        depth_scale=1e-3,
        max_depth_m=SCENE_MAX_DEPTH_M,
        stride=SCENE_PIXEL_STRIDE,
    )
    print(
        f"Built scene cloud from {STATIC_TRANSFORMS_PATH.name}: "
        f"{len(scene_xyz)} points (stride={SCENE_PIXEL_STRIDE}, "
        f"max_depth={SCENE_MAX_DEPTH_M}m)"
    )

    results: list[StrategyResult] = []
    for cfg in STRATEGIES:
        try:
            if REUSE_SAM3D_OUTPUTS:
                result = gather_existing_strategy_outputs(cfg=cfg, mask_paths=mask_paths)
            else:
                result = run_strategy(
                    cfg=cfg,
                    mask_paths=mask_paths,
                    full_image_rgb=full_image_rgb,
                    full_pointmap=full_pointmap,
                    full_intrinsics=intrinsics,
                )
        except Exception as exc:
            print(f"  STRATEGY {cfg.name} crashed: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
            result = StrategyResult(
                name=cfg.name,
                crop_input=cfg.crop_input,
                use_pointmap=cfg.use_pointmap,
                model_load_seconds=0.0,
                total_inference_seconds=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )

        # Build the combined "scene + reprojected objects" PLY for this
        # strategy. Done outside run_strategy so a reprojection crash on
        # one strategy cannot lose the inference results we already paid
        # for.
        if result.error is None and result.object_results:
            print(f"  [reproject] placing {cfg.name} objects into scene cloud (SAM3D pose only)")
            t_reproject = time.time()
            try:
                combined_path = reproject_strategy_into_scene(
                    strategy_dir=OUTPUT_ROOT / cfg.name,
                    object_results=result.object_results,
                    c2w=c2w,
                    scene_xyz=scene_xyz,
                    scene_rgb=scene_rgb,
                )
                if combined_path is not None:
                    result.combined_ply_path = str(combined_path)
            except Exception as exc:
                print(f"  [reproject] {cfg.name} crashed: {exc}", file=sys.stderr)
                traceback.print_exc()
            result.fusion_seconds = time.time() - t_reproject

        results.append(result)
        # Persist after each strategy in case a later one crashes.
        (OUTPUT_ROOT / "results.json").write_text(
            json.dumps([asdict(r) for r in results], indent=2) + "\n"
        )

    write_summary(results, OUTPUT_ROOT / "summary.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
