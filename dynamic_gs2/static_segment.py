"""static_segment.py — FastSAM segmentation + the self-describing segmentation/ folder.

Owns the anchor snapshot (the EXACT frame SAM3D will see) and the segmentation
artifact folder (static_phase.md §3): a single dedicated, copy/rename-safe folder
whose manifest carries RELATIVE paths only — so a renamed dataset never breaks the
way the old cached SAM3 JSON did (it held absolute replay_20260612 paths). The depth
is written as float32-metres TIFF and intrinsics as fx/fy/cx/cy/w/h — exactly what
the SAM3D subprocess (static_sam3d.py) consumes, so there is no format mismatch.

Stages (each timed by the orchestrator):
  snapshot_anchor  : freeze rgb+depth+pose+intrinsics from the trigger frame.
  fastsam_segment  : FastSAM+CLIP text segment -> per-object masks (via model_loader).
  write_seg_folder : masks + overlays + manifest (relative paths).
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .frame import Frame, Intrinsics

SEG_DIRNAME = "segmentation"


@dataclass(frozen=True)
class AnchorRef:
    """The frozen reference frame SAM3D + the mask are built on (paths are absolute on
    disk; the manifest stores RELATIVE)."""
    seg_dir: Path
    rgb_path: Path
    depth_path: Path        # float32 metres TIFF (SAM3D-ready)
    pose_path: Path         # c2w 4x4 OpenGL
    intrinsics_path: Path   # fx fy cx cy w h
    intr: Intrinsics


def _seg_dir(data_dir) -> Path:
    return Path(data_dir) / "static_scene" / SEG_DIRNAME


def snapshot_anchor(frame: Frame, intr: Intrinsics, data_dir, *,
                    gripper_black: bool = True) -> AnchorRef:
    """Freeze the trigger frame into segmentation/anchor/ (wiping any stale folder first):
    rgb.png (optionally gripper-blacked, as SAM3D segments it), depth.tiff (float32 m),
    pose.json (c2w), intrinsics.json. Returns the AnchorRef the later stages key off."""
    seg = _seg_dir(data_dir)
    if seg.exists():
        shutil.rmtree(seg)                       # wipe per run — never reuse stale across datasets
    anchor = seg / "anchor"
    anchor.mkdir(parents=True, exist_ok=True)

    rgb = frame.rgb_bgr.copy()
    if gripper_black and frame.mask_keep is not None:
        keep = frame.mask_keep
        keep = keep[..., 0] if keep.ndim == 3 else keep
        rgb[keep == 0] = 0                       # black out the robot exactly as the old anchor did
    rgb_path = anchor / "rgb.png"
    depth_path = anchor / "depth.tiff"
    pose_path = anchor / "pose.json"
    intr_path = anchor / "intrinsics.json"
    cv2.imwrite(str(rgb_path), rgb)
    cv2.imwrite(str(depth_path), frame.depth_m.astype(np.float32))   # metres, SAM3D reads as-is (scale 1)
    pose_path.write_text(json.dumps(np.asarray(frame.c2w_4x4, dtype=np.float64).tolist()))
    intr_path.write_text(json.dumps({
        "fx": float(intr.fx), "fy": float(intr.fy),
        "cx": float(intr.cx), "cy": float(intr.cy),
        "w": int(intr.width), "h": int(intr.height)}))
    return AnchorRef(seg_dir=seg, rgb_path=rgb_path, depth_path=depth_path,
                     pose_path=pose_path, intrinsics_path=intr_path, intr=intr)


def segment(anchor: AnchorRef, fastsam_handle, prompt_text: str, *,
            min_area_ratio: float = 0.002, max_area_ratio: float = 0.25,
            dedup_iou: float = 0.6, max_objects: int = 8,
            min_score: float = 0.0) -> List[dict]:
    """Run FastSAM+CLIP on the anchor rgb, returning the proven per-object dict list
    ({mask_path, score, bbox, mask_area, object_index}). Masks land in segmentation/masks/."""
    masks_dir = anchor.seg_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    return fastsam_handle.segment(
        image_path=anchor.rgb_path, text_prompt=prompt_text,
        output_dir=masks_dir, output_stem="obj",
        min_area_ratio=min_area_ratio, max_area_ratio=max_area_ratio,
        dedup_iou=dedup_iou, max_objects=max_objects, min_score=min_score)


def write_seg_folder(anchor: AnchorRef, objects: List[dict], prompt_text: str) -> Path:
    """Write the per-object overlays + the manifest (RELATIVE paths) to segmentation/.
    The overlay (masks/obj_NN_overlay.png) is the ONE image the UI shows to validate the
    mask. Returns the manifest path. Mask PNGs already live in masks/ (from segment())."""
    seg = anchor.seg_dir
    masks_dir = seg / "masks"
    rgb = cv2.imread(str(anchor.rgb_path), cv2.IMREAD_COLOR)
    entries = []
    for obj in objects:
        i = int(obj["object_index"])
        mask = cv2.imread(str(obj["mask_path"]), cv2.IMREAD_GRAYSCALE)
        overlay_name = f"obj_{i:02d}_overlay.png"
        if mask is not None and rgb is not None:
            ov = rgb.copy()
            red = np.zeros_like(ov); red[..., 2] = 255          # BGR red
            sel = mask > 127
            ov[sel] = (0.5 * ov[sel] + 0.5 * red[sel]).astype(np.uint8)
            cv2.imwrite(str(masks_dir / overlay_name), ov)
        entries.append({
            "object_index": i,
            "mask": f"masks/{Path(obj['mask_path']).name}",   # RELATIVE to seg/
            "overlay": f"masks/{overlay_name}",
            "score": float(obj.get("score", 0.0)),
            "bbox": [float(x) for x in obj.get("bbox", [])],
            "mask_area": int(obj.get("mask_area", 0)),
        })
    manifest = {
        "prompt": prompt_text,
        "n_objects": len(entries),
        "anchor": {"rgb": "anchor/rgb.png", "depth": "anchor/depth.tiff",
                   "pose": "anchor/pose.json", "intrinsics": "anchor/intrinsics.json"},
        "objects": entries,
    }
    manifest_path = seg / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path
    # All manifest paths are RELATIVE to seg/ -> a copied/renamed dataset resolves
    # them against the folder's own location, so it can never FileNotFound.


def overlay_paths(data_dir) -> List[Path]:
    """Resolve the per-object overlay PNGs (the validation surface) from the manifest's
    RELATIVE paths against the folder location. Used by the UI to show the masks."""
    seg = _seg_dir(data_dir)
    mp = seg / "manifest.json"
    if not mp.is_file():
        return []
    man = json.loads(mp.read_text())
    return [seg / o["overlay"] for o in man.get("objects", []) if (seg / o["overlay"]).is_file()]
