"""_diag_render_anchor.py — render the fused scene from the ANCHOR camera, overlay on anchor RGB.

The anchor frame is the ground truth the static fuse was built on. If the rendered fused object
lands on the real object in the anchor RGB, the placement is correct end-to-end (geometry is fine
and any 'wrong place' the operator saw is a viewer/camera-alignment visual artifact). Writes three
PNGs to <data>/static_scene/segmentation/_render_check/:
    anchor_rgb.png   — the segmentation anchor (truth)
    rendered.png     — full fused scene rendered from the anchor camera
    obj_only.png     — ONLY the tracked object (instance_id==d0) rendered from the anchor camera
    overlay.png      — rendered object outline (green) on the anchor RGB

Run from scripts/ in dynamic_gs env (CUDA_HOME/CPATH set):
  python -m dynamic_gs2._diag_render_anchor ../data_teleoperation/datasets/2026-06-20_224240
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from . import config as _C
from . import static_persist
from .pipeline import pick_d0_instance_id


def main(data_dir, device="cuda"):
    data_dir = Path(data_dir)
    seg = data_dir / "static_scene" / "segmentation"
    out = seg / "_render_check"
    out.mkdir(parents=True, exist_ok=True)

    cfg = _C.load_runtime_config()
    cache = static_persist.warm_cache_path(data_dir)
    sm, gset, lock = static_persist.build_loaded_scene(cfg, device, cache, phase="dynamic")
    d0 = pick_d0_instance_id(gset)

    from nerfstudio.cameras.cameras import Cameras, CameraType
    pose = json.loads((seg / "anchor" / "pose.json").read_text())
    intr = json.loads((seg / "anchor" / "intrinsics.json").read_text())
    c2w = torch.tensor(np.asarray(pose, np.float32)[:3, :4]).unsqueeze(0)
    cam = Cameras(camera_to_worlds=c2w, fx=intr["fx"], fy=intr["fy"], cx=intr["cx"], cy=intr["cy"],
                  width=intr["w"], height=intr["h"], camera_type=CameraType.PERSPECTIVE).to(device)
    cam.metadata = {"cam_idx": 0}

    with lock:
        rgb, _, _ = sm.render(cam)
        inst = (gset.snapshot().buffers["object_instance_ids"][:, 0] == d0)
        objmask = sm.render_object_mask(cam, inst)   # (H,W) or (H,W,1) alpha of object-only render

    rend = (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[..., ::-1]   # RGB->BGR
    anchor = cv2.imread(str(seg / "anchor" / "rgb.png"), cv2.IMREAD_COLOR)
    om = objmask.cpu().numpy()
    om = om[..., 0] if om.ndim == 3 else om
    om_bin = (om > 0.3).astype(np.uint8)

    cv2.imwrite(str(out / "anchor_rgb.png"), anchor)
    cv2.imwrite(str(out / "rendered.png"), rend)
    cv2.imwrite(str(out / "obj_only.png"), (om_bin * 255))

    # overlay: green outline of the rendered object on the anchor RGB
    ov = anchor.copy()
    cnts, _ = cv2.findContours(om_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnts, -1, (0, 255, 0), 3)
    cv2.imwrite(str(out / "overlay.png"), ov)

    # quantify: where is the rendered object's mask centroid vs the FastSAM mask centroid?
    man = json.loads((seg / "manifest.json").read_text())
    fmask = cv2.imread(str(seg / man["objects"][0]["mask"]), cv2.IMREAD_GRAYSCALE) > 127
    if fmask.shape != om_bin.shape:
        fmask = cv2.resize(fmask.astype(np.uint8), (om_bin.shape[1], om_bin.shape[0]),
                           interpolation=cv2.INTER_NEAREST) > 0
    def _cxy(b):
        ys, xs = np.where(b)
        return (float(xs.mean()), float(ys.mean())) if xs.size else (np.nan, np.nan)
    rc = _cxy(om_bin > 0); fc = _cxy(fmask)
    print(f"d0={d0}, obj gaussians rendered, anchor {intr['w']}x{intr['h']}")
    print(f"rendered-object mask centroid (px) = {rc}")
    print(f"FastSAM (truth) mask centroid  (px) = {fc}")
    if not (np.isnan(rc[0]) or np.isnan(fc[0])):
        d = np.hypot(rc[0] - fc[0], rc[1] - fc[1])
        print(f">>> centroid offset = {d:.1f} px on a {intr['w']}px-wide frame "
              f"({100*d/intr['w']:.2f}% of width)")
        print(">>> small (<~30 px) = object renders ON the real object = placement correct end-to-end.")
        print(">>> large = the rendered object is genuinely offset from where it was segmented.")
    print(f"\nwrote -> {out}/  (open overlay.png — green outline should hug the real object)")


if __name__ == "__main__":
    main(*sys.argv[1:])
