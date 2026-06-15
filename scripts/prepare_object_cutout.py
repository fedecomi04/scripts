"""Compose the RGBA object cutout that the World-Tracing object model consumes.

The replay dataset already has the prepared FastSAM object mask + the
segmentation-source RGB of the LAST static frame (static0 == arm_00026). This
reads those, crops to the mask bbox with padding (so the object dominates the
canvas — the model also center-crops internally), and writes an RGBA PNG whose
**alpha channel is the object mask**. That PNG is the input to
``world_tracing_worker.py``.

Defaults target ``replay_20260612_203321``; zero-arg run produces
``<dataset>/world_tracing/obj_rgba.png``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# replay_20260612_203321 was wiped externally on 2026-06-14; 'screwdriver recorded full'
# is the structurally-identical intact screwdriver scene (57 frames, arm_00026, prompt "screwdriver").
_DEFAULT_DS = Path("/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/screwdriver recorded full")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=_DEFAULT_DS)
    ap.add_argument("--rgb", type=Path, default=None,
                    help="Source RGB. Default = dynamic_scene/initialization_debug/static0_rgb.png "
                         "(gripper-blacked; the image the mask was computed on).")
    ap.add_argument("--mask", type=Path, default=None,
                    help="Object mask. Default = dynamic_scene/initialization_debug/static0_obj_00_mask.png")
    ap.add_argument("--pad-frac", type=float, default=0.30, help="Bbox padding as fraction of the longer bbox side")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    dbg = args.dataset / "dynamic_scene" / "initialization_debug"
    rgb_path = args.rgb or (dbg / "static0_rgb.png")
    mask_path = args.mask or (dbg / "static0_obj_00_mask.png")
    out = args.out or (args.dataset / "world_tracing" / "obj_rgba.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    rgb = cv2.cvtColor(cv2.imread(str(rgb_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask_bool = mask > 127
    if mask_bool.shape != rgb.shape[:2]:
        mask_bool = cv2.resize(mask_bool.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        raise SystemExit(f"empty mask: {mask_path}")
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    side = max(y1 - y0, x1 - x0)
    pad = int(round(args.pad_frac * side))
    H, W = rgb.shape[:2]
    cy0 = max(0, y0 - pad); cy1 = min(H, y1 + 1 + pad)
    cx0 = max(0, x0 - pad); cx1 = min(W, x1 + 1 + pad)

    rgb_crop = rgb[cy0:cy1, cx0:cx1]
    alpha_crop = (mask_bool[cy0:cy1, cx0:cx1].astype(np.uint8)) * 255
    rgba = np.dstack([rgb_crop, alpha_crop])

    cv2.imwrite(str(out), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    meta = {
        "source_rgb": str(rgb_path),
        "source_mask": str(mask_path),
        "full_bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "crop_xyxy_in_full": [int(cx0), int(cy0), int(cx1), int(cy1)],
        "crop_hw": [int(cy1 - cy0), int(cx1 - cx0)],
        "mask_px": int(mask_bool.sum()),
        "pad_frac": args.pad_frac,
    }
    (out.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2))
    print(f"[cutout] wrote {out}  crop={meta['crop_hw']}  mask_px={meta['mask_px']}  bbox={meta['full_bbox_xyxy']}")
    print(f"[cutout] meta → {out.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
