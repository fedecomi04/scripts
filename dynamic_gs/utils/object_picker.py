"""SAM3 object loader + overlay for the interactive object picker.

The dynamic-phase object picker shows the operator the exact image that was
given to SAM3 during static init, with each segmented object's mask overlaid
in a distinct color and numbered. The number is the **object id** the operator
selects, which equals the SAM3 mask number + 1 — the same value carried in the
per-Gaussian ``object_instance_ids`` buffer in BOTH init paths:

* ``static-gs`` (SAM3D init): ``instance_id = sam3_object_index + 1`` is assigned
  1:1 in ``fusion/phase0.run_phase0b_fusion``.
* ``static-gs-preseg``: as of the 2026-06-10 id-order fix in
  ``preseg_seed._assign_and_merge`` / ``_propagate``, the SAM2-video ``obj_id``
  (and thus the sidecar id) is seeded as ``sam3_mask_index + 1``.

So ``object_flags = (object_instance_ids == picked_id)`` selects exactly the
clicked object's Gaussians, regardless of which init path produced the scene.

This module only READS the on-disk artifacts each init path already wrote; it
never re-runs SAM3.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class ObjEntry:
    """One selectable object."""

    instance_id: int          # == sam3_mask_index + 1; matches object_instance_ids
    mask: np.ndarray          # (H, W) bool
    score: float              # SAM3 confidence (0 if unknown)
    color_bgr: tuple          # overlay color used for this id


def _hue_color_bgr(i: int, n: int) -> tuple:
    """Evenly-spaced HSV hue -> BGR (same scheme as preseg_seed overlay)."""
    hue = int((179 * i) / max(n, 1)) % 180
    hsv = np.zeros((1, 1, 3), np.uint8)
    hsv[0, 0] = (hue, 220, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def _find_sam3d_artifacts(data_dir: Path):
    """Return (image_path, sam3_results_json) for the SAM3D-init path, or None.

    static-gs writes ``initialization_debug/static0_rgb.png`` +
    ``static0_sam3_results.json``. ``initialization_debug`` is reachable both
    directly under ``dynamic_scene/`` and via the ``dynamic_scene_30fps/`` symlink;
    probe the common locations.
    """
    candidates = [
        data_dir / "dynamic_scene" / "initialization_debug",
        data_dir / "dynamic_scene_30fps" / "initialization_debug",
    ]
    for d in candidates:
        js = d / "static0_sam3_results.json"
        if js.is_file():
            return js
    return None


def _find_preseg_artifacts(data_dir: Path):
    """Return the preseg artifacts dir if present, else None."""
    d = data_dir / "static_scene" / "preseg_artifacts"
    if (d / "_sam3_prompt_00_raw_masks.npz").is_file():
        return d
    return None


def load_sam3_objects(
    data_dir: Path,
) -> Optional[tuple[np.ndarray, list[ObjEntry]]]:
    """Auto-detect the init path and load (image_rgb_uint8, [ObjEntry, ...]).

    Returns ``None`` if neither path's artifacts are on disk (the caller falls
    back to the heuristic / forced id). Entries are sorted by ``instance_id``.
    """
    data_dir = Path(data_dir)

    # --- Preseg path (preferred when present: gives all K SAM3 objects) ---
    preseg = _find_preseg_artifacts(data_dir)
    if preseg is not None:
        img_path = preseg / "_frame0_input.png"
        npz = np.load(preseg / "_sam3_prompt_00_raw_masks.npz")
        masks = np.asarray(npz["masks"], dtype=bool)  # (K, H, W)
        scores = np.asarray(npz["scores"], dtype=np.float32) if "scores" in npz else None
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        img_rgb = img[:, :, ::-1].copy()
        K = masks.shape[0]
        entries = [
            ObjEntry(
                instance_id=j + 1,
                mask=masks[j],
                score=float(scores[j]) if scores is not None else 0.0,
                color_bgr=_hue_color_bgr(j, K),
            )
            for j in range(K)
        ]
        return img_rgb, entries

    # --- SAM3D path ---
    js = _find_sam3d_artifacts(data_dir)
    if js is not None:
        meta = json.loads(js.read_text())
        # Resolve paths relative to the JSON's own dir (the JSON may store
        # absolute paths into a sibling dir; prefer the JSON-local copy).
        base = js.parent
        img_path = base / "static0_rgb.png"
        img = cv2.imread(str(img_path))
        if img is None:
            # Fall back to the path recorded in the JSON.
            ip = Path(meta.get("image_path", ""))
            img = cv2.imread(str(ip))
        if img is None:
            return None
        img_rgb = img[:, :, ::-1].copy()
        objs = meta.get("objects", [])
        K = len(objs)
        entries = []
        for o in objs:
            idx = int(o["object_index"])
            mp = base / Path(o["mask_path"]).name
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is None:
                m = cv2.imread(str(Path(o["mask_path"])), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            entries.append(
                ObjEntry(
                    instance_id=idx + 1,
                    mask=m > 127,
                    score=float(o.get("score", 0.0)),
                    color_bgr=_hue_color_bgr(idx, max(K, 1)),
                )
            )
        if not entries:
            return None
        entries.sort(key=lambda e: e.instance_id)
        return img_rgb, entries

    return None


def render_picker_overlay(
    image_rgb: np.ndarray,
    entries: list[ObjEntry],
    *,
    alpha: float = 0.55,
) -> np.ndarray:
    """Blend each object's mask onto the image in its color + draw the id label.

    Returns an (H, W, 3) uint8 RGB image suitable for ``server.gui.add_image``.
    The number drawn on each object is its ``instance_id`` — the value the
    operator picks in the button-group.
    """
    out = image_rgb.astype(np.float32).copy()
    H, W = out.shape[:2]
    for e in entries:
        m = e.mask
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        color_rgb = np.array(e.color_bgr[::-1], np.float32)  # BGR tuple -> RGB
        out[m] = (1.0 - alpha) * out[m] + alpha * color_rgb[None, :]
    overlay = np.clip(out, 0, 255).astype(np.uint8)
    # Labels drawn after blending so they stay crisp.
    for e in entries:
        m = e.mask
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        ys, xs = np.where(m)
        if len(xs) == 0:
            continue
        cx, cy = int(np.median(xs)), int(np.median(ys))
        txt = str(e.instance_id)
        cv2.putText(overlay, txt, (cx - 8, cy + 8), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(overlay, txt, (cx - 8, cy + 8), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, tuple(int(c) for c in e.color_bgr[::-1]), 2, cv2.LINE_AA)
    return overlay
