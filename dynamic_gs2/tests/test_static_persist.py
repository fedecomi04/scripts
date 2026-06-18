"""Integration test: load the REAL screwdriver static_state.pt into the new core + render + round-trip.

Needs GPU + the screwdriver dataset. Run (from scripts/):
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib python -m dynamic_gs2.tests.test_static_persist
Skips (exit 0) if the dataset is absent.
"""
import json
import re
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import torch

from dynamic_gs2 import config as C
from dynamic_gs2 import static_persist as P
from dynamic_gs2.gaussian_set import GaussianSet

DATASET = Path("/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/screwdriver recorded full")


def _camera(device):
    from nerfstudio.cameras.cameras import Cameras, CameraType
    meta = json.loads((DATASET / "static_scene" / "transforms.json").read_text())
    frame = sorted(meta["frames"], key=lambda f: int(re.findall(r"\d+", f["file_path"])[-1]))[-1]
    c2w = torch.tensor(np.asarray(frame["transform_matrix"], np.float32)[:3, :4])[None]
    return Cameras(camera_to_worlds=c2w, fx=meta["fl_x"], fy=meta["fl_y"], cx=meta["cx"], cy=meta["cy"],
                   width=int(meta["w"]), height=int(meta["h"]), camera_type=CameraType.PERSPECTIVE).to(device)


def main():
    cache = DATASET / "static_scene" / "static_state.pt"
    if not cache.exists():
        print(f"test_static_persist SKIP (no dataset at {cache})")
        return 0
    assert torch.cuda.is_available(), "needs CUDA"
    device = "cuda"
    cfg = C.load_runtime_config()

    n_expected = int(torch.load(cache, map_location="cpu", weights_only=False)["num_points"])
    sm, gset, lock = P.build_loaded_scene(cfg, device, cache, phase="dynamic")
    assert gset.num_points == n_expected, f"loaded {gset.num_points} != {n_expected}"
    assert sm.gauss_params["means"].shape[0] == n_expected, "model bound to loaded tensors"
    print(f"[static_persist] loaded real scene: {gset.num_points:,} gaussians")

    # identity buffers loaded (object_instance_ids carries the inserted object)
    snap = gset.snapshot()
    n_obj = int((snap.buffers["object_instance_ids"][:, 0] > 0).sum())
    n_ins = int((snap.buffers["inserted_flags"][:, 0] > 0).sum())
    print(f"[static_persist] object_instance_ids>0: {n_obj:,}  inserted_flags>0: {n_ins:,}")

    cam = _camera(device)
    with lock:
        rgb, depth, alpha = sm.render(cam)
    assert torch.isfinite(rgb).all(), "rendered rgb finite"
    assert float(alpha.max()) > 0.3, f"scene rendered (alpha_max={float(alpha.max()):.3f})"
    print(f"[static_persist] render OK: rgb{tuple(rgb.shape)} alpha_max={float(alpha.max()):.3f} "
          f"depth_med={float(depth[alpha[...,0]>0.5].median()) if (alpha[...,0]>0.5).any() else 0:.3f}m")

    # round-trip: save the SSOT, reload into a fresh scene, counts + values match
    with tempfile.TemporaryDirectory() as td:
        out = P.save_warm_cache(gset, td, cfg=cfg)
        assert out.exists()
        sm2, gset2, _ = P.build_loaded_scene(cfg, device, out, phase="dynamic")
        assert gset2.num_points == n_expected, "round-trip count"
        assert torch.allclose(gset2.snapshot().params["means"].cpu(),
                              snap.params["means"].cpu(), atol=1e-5), "round-trip means match"
        print("[static_persist] save->load round-trip OK")

    print("test_static_persist OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
