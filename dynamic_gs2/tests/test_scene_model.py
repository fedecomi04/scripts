"""Integration test for dynamic_gs2.scene_model — WRAP build + render + surgery+rebind+render.

Needs GPU + gsplat + nerfstudio. Run (from scripts/):
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib conda run -n dynamic_gs \
        python -m dynamic_gs2.tests.test_scene_model
(or the wrapper scripts/dynamic_gs2/tests/run_gpu_tests.sh)
"""
import sys
import threading

import torch

from nerfstudio.cameras.cameras import Cameras, CameraType

from dynamic_gs2.scene_model import SceneModel
from dynamic_gs2.gaussian_set import GaussianSet, build_default_gauss_tensors


class _Cfg:
    class _ST:
        sh_degree = 3
    static_train = _ST()


def _camera(device, H=128, W=128, dist=3.0):
    c2w = torch.eye(4)[:3, :4].clone()
    c2w[2, 3] = dist                       # OpenGL: at +z looking toward -z (origin)
    return Cameras(
        camera_to_worlds=c2w[None].to(device),
        fx=torch.tensor([[120.0]]), fy=torch.tensor([[120.0]]),
        cx=torch.tensor([[W / 2]]), cy=torch.tensor([[H / 2]]),
        width=torch.tensor([[W]]), height=torch.tensor([[H]]),
        camera_type=CameraType.PERSPECTIVE,
    )


def main():
    assert torch.cuda.is_available(), "scene_model render needs CUDA"
    device = "cuda"
    torch.manual_seed(0)

    n = 2000
    xyz = (torch.rand(n, 3) - 0.5) * 0.6     # ~tight cloud at origin
    rgb = torch.rand(n, 3)
    sm = SceneModel(_Cfg(), device, seed_xyz=xyz, seed_rgb=rgb, phase="dynamic")
    lock = threading.RLock()
    sm.attach_render_lock(lock)
    assert sm.gauss_params["means"].shape[0] == n
    assert sm.gauss_params["features_rest"].shape[1] == 15   # sh_degree 3 -> 15
    bg = sm._get_background_color()
    assert torch.allclose(bg.cpu(), torch.tensor([0.86, 0.92, 1.0]), atol=1e-4), "Gazebo sky bg (Inv #6)"

    cam = _camera(device)
    with lock:
        rgb_img, depth, alpha = sm.render(cam)
    assert rgb_img.shape[-1] == 3 and rgb_img.shape[0] == 128 and rgb_img.shape[1] == 128
    assert torch.isfinite(rgb_img).all(), "rgb finite"
    assert alpha is not None and float(alpha.max()) > 0.1, "something rendered (alpha>0)"
    print(f"[scene_model] render OK: rgb{tuple(rgb_img.shape)} alpha_max={float(alpha.max()):.3f}")

    # bind GaussianSet, do a real insert through the SSOT, rebind, render again
    gset = GaussianSet(sm, lock)
    gt = build_default_gauss_tensors((torch.rand(300, 3) - 0.5) * 0.6, torch.rand(300, 3),
                                     sh_degree=3, sh_rest_dim=15, device=device, dtype=torch.float32)
    gset.insert(gt, object_flag=0.0, instance_id=5)
    assert gset.num_points == n + 300
    assert sm.gauss_params["means"].shape[0] == n + 300, "model sees the insert (same dict)"
    with lock:
        rgb2, _, alpha2 = sm.render(cam)
    assert torch.isfinite(rgb2).all(), "render after insert+rebind finite"
    print(f"[scene_model] insert+rebind+render OK: N={gset.num_points} alpha_max={float(alpha2.max()):.3f}")

    # object-mask of instance 5 (the inserted tail)
    snap = gset.snapshot()
    inst_mask = (snap.buffers["object_instance_ids"][:, 0] == 5)
    assert int(inst_mask.sum()) == 300
    with lock:
        objmask = sm.render_object_mask(cam, inst_mask)
    assert objmask.shape == (128, 128) and objmask.dtype == torch.bool
    # subset render must not corrupt the full param dict
    assert sm.gauss_params["means"].shape[0] == n + 300, "gauss_params restored after subset render"
    print(f"[scene_model] render_object_mask OK: {int(objmask.sum())} px")

    # cull through the SSOT, render again
    gset.cull(torch.arange(0, 100, device=device))
    with lock:
        rgb3, _, _ = sm.render(cam)
    assert torch.isfinite(rgb3).all()
    print(f"[scene_model] cull+rebind+render OK: N={gset.num_points}")

    # ---- FREE-LIST render equivalence: dead capacity rows must NOT affect the image ----
    # Build a freelist set, insert then cull so live rows < physical capacity (dead tail exists),
    # render it, and compare to a SEPARATE packed model rendered from the SAME live rows. Equal
    # image => the [:count] render-slice correctly excludes the dead rows on REAL gsplat.
    torch.manual_seed(1)
    fx = (torch.rand(1500, 3) - 0.5) * 0.6
    fr = torch.rand(1500, 3)
    sm_f = SceneModel(_Cfg(), device, seed_xyz=fx, seed_rgb=fr, phase="dynamic")
    lock_f = threading.RLock(); sm_f.attach_render_lock(lock_f)
    gset_f = GaussianSet(sm_f, lock_f, freelist=True)
    sm_f.set_count_provider(lambda: gset_f.count)
    add = build_default_gauss_tensors((torch.rand(500, 3) - 0.5) * 0.6, torch.rand(500, 3),
                                      sh_degree=3, sh_rest_dim=15, device=device, dtype=torch.float32)
    gset_f.insert(add, object_flag=1.0, instance_id=9)          # count 2000 (capacity grows x2 on overflow)
    gset_f.cull(torch.arange(0, 400, device=device))            # count 1600 -> dead tail below capacity
    phys = sm_f.gauss_params["means"].shape[0]
    assert gset_f.count == 1600 and phys > 1600, f"dead tail exists (count 1600 < capacity {phys})"
    with lock_f:
        rgb_free, _, _ = sm_f.render(cam)

    # packed reference: a fresh model seeded with EXACTLY the 1600 live rows
    snap_f = gset_f.snapshot()
    sm_p = SceneModel(_Cfg(), device, seed_xyz=snap_f.params["means"], seed_rgb=torch.rand(1600, 3),
                      phase="dynamic")
    lock_p = threading.RLock(); sm_p.attach_render_lock(lock_p)
    with torch.no_grad():
        for k in ("means", "features_dc", "features_rest", "scales", "quats", "opacities"):
            sm_p.gauss_params[k] = torch.nn.Parameter(snap_f.params[k].clone(), requires_grad=False)
    with lock_p:
        rgb_packed, _, _ = sm_p.render(cam)
    max_diff = float((rgb_free - rgb_packed).abs().max())
    assert max_diff < 1e-4, f"free-list render != packed render (dead rows leaked!) max_diff={max_diff}"
    print(f"[scene_model] free-list render == packed render OK: dead=400, max_diff={max_diff:.2e}")

    # ---- virtual-hide (FF Option A): hidden rows vanish from the image; unhide restores ----
    with lock_f:
        rgb_before, _, _ = sm_f.render(cam)
    hide = torch.arange(0, 800, device=device)           # hide a chunk of the live rows
    with lock_f:
        sm_f.set_hidden_indices(hide)
        rgb_hidden, _, _ = sm_f.render(cam)
        sm_f.set_hidden_indices(None)
        rgb_after, _, _ = sm_f.render(cam)
    assert float((rgb_before - rgb_hidden).abs().max()) > 1e-3, "hiding 800 rows must change the image"
    assert float((rgb_before - rgb_after).abs().max()) < 1e-5, "unhide must restore the exact image"
    assert gset_f.count == 1600, "virtual-hide must NOT delete rows from the SSOT"
    print(f"[scene_model] virtual-hide OK: hidden changes image, unhide restores, count unchanged")

    print("test_scene_model OK")


if __name__ == "__main__":
    sys.exit(main())
