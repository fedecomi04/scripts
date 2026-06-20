"""_diag_dynamic_handoff.py — does the object MOVE in the dynamic loop (no real motion)?

Warm-loads the fused static_state.pt into the DYNAMIC scene, picks d0, and drives the
loop with STATIC keyframes (the object is NOT moving in these — they're the static sweep).
Reports the tracked-object centroid:
  - before any tick           (== where the static fuse put it; the truth)
  - after seed tick           (ref.capture; should be unchanged)
  - after the first track tick (ref.apply(R,t); if the object jumps here with NO real
                                object motion, the dynamic handoff is displacing it)
Also prints the raw tracker (R,t) so we can see if the first estimate is non-identity.

Run from scripts/ in dynamic_gs env:
  python -m dynamic_gs2._diag_dynamic_handoff ../data_teleoperation/datasets/2026-06-20_224240
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from . import config as _C
from . import static_persist
from .adapters_source import camera_from_frame
from .dynamic_track import ReferenceObjectPose, TrackerInputs, XFeatTracker
from .frame import Frame, Intrinsics
from .pipeline import pick_d0_instance_id


def _obj_centroid(gset, d0):
    snap = gset.snapshot()
    ids = snap.buffers["object_instance_ids"][:, 0]
    return snap.params["means"][ids == d0].mean(0).cpu().numpy()


def _frames_from_static(data_dir):
    """Load the static keyframes as Frames (sorted by file_path, like the anchor pick)."""
    import cv2
    st = Path(data_dir) / "static_scene"
    meta = json.loads((st / "transforms.json").read_text())
    intr = Intrinsics(width=int(meta["w"]), height=int(meta["h"]),
                      fx=meta["fl_x"], fy=meta["fl_y"], cx=meta["cx"], cy=meta["cy"])
    frames = sorted(meta["frames"], key=lambda f: f["file_path"])
    out = []
    for i, f in enumerate(frames):
        rgb = cv2.imread(str(st / f["file_path"].lstrip("./")), cv2.IMREAD_COLOR)
        dp = f.get("depth_file_path") or f["file_path"].replace("rgb", "depth").replace(".png", ".tiff")
        d = cv2.imread(str(st / dp.lstrip("./")), cv2.IMREAD_UNCHANGED)
        depth_m = (d.astype(np.float32) * 1e-3)
        mp = f.get("mask_path")
        m = cv2.imread(str(st / mp.lstrip("./")), cv2.IMREAD_GRAYSCALE) if mp else None
        mask = (m > 0).astype(np.uint8) if m is not None else np.ones(depth_m.shape, np.uint8)
        out.append(Frame(seq=i + 1, stamp_sec=float(i), rgb_bgr=np.ascontiguousarray(rgb),
                         depth_m=np.ascontiguousarray(depth_m), mask_keep=np.ascontiguousarray(mask),
                         c2w_4x4=np.asarray(f["transform_matrix"], np.float64)))
    return out, intr


def main(data_dir, device="cuda"):
    data_dir = Path(data_dir)
    cfg = _C.load_runtime_config()
    cache = static_persist.warm_cache_path(data_dir)
    sm, gset, lock = static_persist.build_loaded_scene(cfg, device, cache, phase="dynamic")
    d0 = pick_d0_instance_id(gset)
    ref = ReferenceObjectPose(d0_instance_id=d0)
    tracker = XFeatTracker(device, cfg.tracker, cfg.pose_filter)

    frames, intr = _frames_from_static(data_dir)
    print(f"d0={d0}, {gset.num_points} gaussians, {len(frames)} static frames, "
          f"obj count={int((gset.snapshot().buffers['object_instance_ids'][:,0]==d0).sum())}")

    c_truth = _obj_centroid(gset, d0)
    print(f"\nobj centroid (TRUTH, fused) = {c_truth}")

    def _inputs(fr):
        rgb = torch.from_numpy(np.ascontiguousarray(fr.rgb_bgr[..., ::-1])).float().to(device) / 255.0
        depth = torch.from_numpy(np.ascontiguousarray(fr.depth_m)).to(device)
        keep = torch.from_numpy(np.ascontiguousarray(fr.mask_keep)).to(device).float()
        cam = camera_from_frame(fr, intr, device)
        with lock:
            inst = (gset.snapshot().buffers["object_instance_ids"][:, 0] == d0)
            objmask = sm.render_object_mask(cam, inst)
        return TrackerInputs(rgb=rgb, depth=depth, camera=cam, keep_mask=keep,
                             object_mask=objmask, stamp_sec=fr.stamp_sec), cam

    # SEED on the LAST static frame (== the anchor frame; the cleanest D0)
    seed_fr = frames[-1]
    snap = gset.snapshot()
    ref.capture(snap)
    inp, cam = _inputs(seed_fr)
    kept = tracker.seed(inp)
    print(f"seed on anchor frame: kept={kept}, ready={tracker.ready}")
    c_after_seed = _obj_centroid(gset, d0)
    print(f"obj centroid after SEED = {c_after_seed}  (|d| from truth = "
          f"{np.linalg.norm(c_after_seed-c_truth)*1000:.2f} mm — should be 0)")

    # FIRST TRACK on the SAME anchor frame (object did NOT move -> must be identity)
    snap = gset.snapshot()
    est = tracker.track(inp)
    R = np.asarray(est.rotation, float); t = np.asarray(est.translation, float).reshape(3)
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print(f"\nFIRST TRACK (same anchor frame, NO real motion):")
    print(f"  success={est.success} inliers={est.inlier_count} |t|={np.linalg.norm(t)*1000:.2f} mm "
          f"rot={ang:.2f} deg")
    if est.success:
        out = ref.apply(est.rotation, est.translation, snap)
        if out is not None:
            gset.write_object_pose(*out)
    c_after_track = _obj_centroid(gset, d0)
    print(f"  obj centroid after TRACK = {c_after_track}  (|d| from truth = "
          f"{np.linalg.norm(c_after_track-c_truth)*1000:.2f} mm)")
    print(f"\n>>> If |t|/rot above are ~0 and centroid drift ~0, the handoff is NOT the bug.")
    print(f">>> If the FIRST track returns a big (R,t) on a no-motion frame, THAT is the displacement.")


if __name__ == "__main__":
    main(*sys.argv[1:])
