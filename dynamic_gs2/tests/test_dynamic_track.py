"""Smoke test for dynamic_gs2.dynamic_track — XFeat WRAP plumbing + ReferenceObjectPose math.

ReferenceObjectPose test is deterministic CPU (always runs). The XFeat seed/track smoke
needs GPU + the screwdriver dataset + xfeat weights; skips gracefully otherwise.
Run (from scripts/): LD_LIBRARY_PATH=$CONDA_PREFIX/lib python -m dynamic_gs2.tests.test_dynamic_track
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from dynamic_gs2 import config as C
from dynamic_gs2.dynamic_track import ReferenceObjectPose, rotation_matrix_to_quaternion
from dynamic_gs2.gaussian_set import GaussianSnapshot

DATASET = Path("/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/screwdriver recorded full")


def _fake_snapshot(means, quats, ids):
    return GaussianSnapshot(
        params={"means": means, "quats": quats},
        buffers={"object_instance_ids": ids.reshape(-1, 1)},
        num_points=means.shape[0], version=0)


def test_reference_pose():
    # 4 gaussians: ids [5,5,9,5] -> tracked id=5 selects rows 0,1,3
    means = torch.tensor([[1., 0, 0], [0, 1., 0], [5., 5, 5], [0, 0, 1.]])
    quats = torch.tensor([[1., 0, 0, 0]]).repeat(4, 1)
    ids = torch.tensor([5, 5, 9, 5])
    snap = _fake_snapshot(means, quats, ids)

    ref = ReferenceObjectPose(d0_instance_id=5)
    assert ref.capture(snap) == 3

    # identity -> subset unchanged
    out = ref.apply(np.eye(3), np.zeros(3), snap)
    assert out is not None
    ms, qs, mask = out
    assert mask.tolist() == [True, True, False, True]
    assert torch.allclose(ms, means[mask], atol=1e-6)

    # 90deg about z + translation: (1,0,0)->(0,1,0)+t
    Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float32)
    t = np.array([0.5, 0.0, 0.0], np.float32)
    ms2, qs2, _ = ref.apply(Rz, t, snap)
    assert torch.allclose(ms2[0], torch.tensor([0.5, 1.0, 0.0]), atol=1e-5), ms2[0]
    assert torch.allclose(qs2.norm(dim=-1), torch.ones(3), atol=1e-5), "quats normalized"

    # count-mismatch (FF appends a tracked-id row -> ref stale) -> None, no crash
    means_grown = torch.cat([means, torch.zeros(1, 3)], 0)
    quats_grown = torch.cat([quats, quats[:1]], 0)
    ids_grown = torch.cat([ids, torch.tensor([5])], 0)   # 4th tracked row appears
    assert ref.apply(np.eye(3), np.zeros(3), _fake_snapshot(means_grown, quats_grown, ids_grown)) is None
    print("[dynamic_track] ReferenceObjectPose OK")


def test_xfeat_smoke():
    tj = DATASET / "dynamic_scene" / "transforms.json"
    if not tj.exists() or not torch.cuda.is_available():
        print("[dynamic_track] xfeat smoke SKIP (no dataset/CUDA)")
        return
    try:
        import cv2
        from dynamic_gs2.dynamic_track import TrackerInputs, XFeatTracker
        from dynamic_gs2.adapters_source import camera_from_frame
        from dynamic_gs2.frame import Frame, Intrinsics
    except Exception as e:
        print(f"[dynamic_track] xfeat smoke SKIP (import: {e})")
        return

    meta = json.loads(tj.read_text())
    intr = Intrinsics(width=meta["w"], height=meta["h"], fx=meta["fl_x"], fy=meta["fl_y"],
                      cx=meta["cx"], cy=meta["cy"])
    frames = sorted(meta["frames"], key=lambda f: f["file_path"])
    device = "cuda"

    def load(i):
        f = frames[i]
        dd = DATASET / "dynamic_scene"
        bgr = cv2.imread(str(dd / f["file_path"].lstrip("./")), cv2.IMREAD_COLOR)
        d = cv2.imread(str(dd / f["depth_file_path"].lstrip("./")), cv2.IMREAD_UNCHANGED).astype(np.float32) * 1e-3
        c2w = np.asarray(f["transform_matrix"], np.float64)
        fr = Frame(seq=i + 1, stamp_sec=float(i), rgb_bgr=bgr, depth_m=d,
                   mask_keep=np.ones(d.shape, np.uint8), c2w_4x4=c2w)
        rgb = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1])).float().to(device) / 255.0
        depth = torch.from_numpy(d).to(device)
        cam = camera_from_frame(fr, intr, torch.device(device))
        return rgb, depth, cam

    cfg = C.load_runtime_config()
    trk = XFeatTracker(device, cfg.tracker, cfg.pose_filter)
    H, W = meta["h"], meta["w"]
    objbox = torch.zeros((H, W), dtype=torch.bool, device=device)
    objbox[H // 3:2 * H // 3, W // 3:2 * W // 3] = True   # center region as the "object"

    rgb0, d0, cam0 = load(0)
    kept = trk.seed(TrackerInputs(rgb=rgb0, depth=d0, camera=cam0, keep_mask=None,
                                  object_mask=objbox, stamp_sec=0.0))
    print(f"[dynamic_track] seed kept {kept} keypoints, ready={trk.ready}")
    assert trk.ready, "D0 seeded"

    rgb1, d1, cam1 = load(2)
    est = trk.track(TrackerInputs(rgb=rgb1, depth=d1, camera=cam1, keep_mask=None,
                                  object_mask=objbox, stamp_sec=2.0))
    assert est.rotation.shape == (3, 3) and est.translation.shape == (3,)
    assert np.isfinite(est.rotation).all() and np.isfinite(est.translation).all()
    assert "xfeat_extract" in est.timings or "input_prep" in est.timings
    print(f"[dynamic_track] track: success={est.success} inliers={est.inlier_count} "
          f"corr={est.correspondence_count} |t|={np.linalg.norm(est.translation)*1000:.1f}mm")


def main():
    test_reference_pose()
    test_xfeat_smoke()
    print("test_dynamic_track OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
