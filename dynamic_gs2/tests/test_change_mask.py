"""Tests for dynamic_gs2.change_mask — self-contained single-scale RGB CDN.

Deterministic CPU (no gsplat/render). Validates: no-change->empty, real change fires in the
right region, object/gripper exclusion, and the depth-as-validity hole gate. Run:
    conda run -n dynamic_gs python -m dynamic_gs2.tests.test_change_mask
"""
import sys

import torch

from dynamic_gs2 import config as C
from dynamic_gs2.change_mask import compute_change_mask, resolve_downsample_factor


def _cfg():
    return C.load_runtime_config().change_mask


def main():
    cm = _cfg()
    H = W = 256
    torch.manual_seed(0)
    base = torch.rand(H, W, 3)                      # a textured "scene"
    alpha = torch.ones(H, W, 1)                     # fully covered
    depth = torch.full((H, W, 1), 1.0)              # valid near surface everywhere

    # ---- downsample factor scales with resolution ----
    assert resolve_downsample_factor(base, 150) == max(1, int((H * W) ** 0.5 / 150))

    # ---- 1. identical images -> no change ----
    m = compute_change_mask(rendered_rgb=base, rendered_alpha=alpha, live_rgb=base.clone(),
                            gt_depth=depth, gripper_keep=None, object_mask=None, cfg=cm)
    assert m.shape == (H, W, 1)
    assert float(m.sum()) == 0.0, f"identical images must give empty CDN, got {float(m.sum())}"

    # ---- 2. a big changed patch fires, and ONLY there ----
    live = base.clone()
    live[60:160, 60:160, :] = 1.0 - live[60:160, 60:160, :]   # invert a 100x100 block
    m = compute_change_mask(rendered_rgb=base, rendered_alpha=alpha, live_rgb=live,
                            gt_depth=depth, gripper_keep=None, object_mask=None, cfg=cm)
    fired = m[..., 0] > 0.5
    assert float(fired.sum()) > 0, "real change must fire"
    # centroid of fired pixels lands inside the changed block
    ys, xs = torch.where(fired)
    cy, cx = float(ys.float().mean()), float(xs.float().mean())
    assert 60 <= cy <= 160 and 60 <= cx <= 160, f"change centroid ({cy:.0f},{cx:.0f}) not in the patch"
    # nothing fires far from the patch (top-left corner stays clean)
    assert float(fired[:40, :40].sum()) == 0, "no spurious change far from the real patch"

    # ---- 3. object_mask EXCLUDES its region even though it changed ----
    obj = torch.zeros(H, W)
    obj[60:160, 60:160] = 1.0                       # mark the changed block as the tracked object
    m = compute_change_mask(rendered_rgb=base, rendered_alpha=alpha, live_rgb=live,
                            gt_depth=depth, gripper_keep=None, object_mask=obj, cfg=cm)
    assert float(m.sum()) == 0.0, "change inside the object mask must be excluded"

    # ---- 4. gripper_keep=0 region is excluded ----
    keep = torch.ones(H, W)
    keep[60:160, 60:160] = 0.0                      # gripper covers the changed block
    m = compute_change_mask(rendered_rgb=base, rendered_alpha=alpha, live_rgb=live,
                            gt_depth=depth, gripper_keep=keep, object_mask=None, cfg=cm)
    assert float(m.sum()) == 0.0, "change under gripper (keep=0) must be excluded"

    # ---- 5. depth-as-validity HOLE gate: uncovered pixel kept iff live depth is a near surface ----
    # Make the changed block UNCOVERED (alpha=0). With valid live depth there -> kept (fillable hole).
    alpha_hole = torch.ones(H, W, 1); alpha_hole[60:160, 60:160] = 0.0
    m_hole = compute_change_mask(rendered_rgb=base, rendered_alpha=alpha_hole, live_rgb=live,
                                 gt_depth=depth, gripper_keep=None, object_mask=None, cfg=cm)
    assert float(m_hole.sum()) > 0, "uncovered pixel with valid live depth = fillable hole -> kept"
    # Same uncovered block but live depth = 0 (void / no return) -> dropped.
    depth_void = depth.clone(); depth_void[60:160, 60:160] = 0.0
    m_void = compute_change_mask(rendered_rgb=base, rendered_alpha=alpha_hole, live_rgb=live,
                                 gt_depth=depth_void, gripper_keep=None, object_mask=None, cfg=cm)
    assert float(m_void.sum()) == 0.0, "uncovered + no live depth = genuine void -> dropped"

    # ---- voxel_merge: fuse a dense cluster -> 1 gaussian sized to the cluster EXTENT (no hole) ----
    from dynamic_gs2.dynamic_ff_backends import voxel_merge, _clamp_log_scale
    # 27 tiny (0.1mm) splats packed in a 0.9mm cube + 1 lone splat in a far voxel.
    import itertools
    grid = torch.tensor(list(itertools.product([0.0, 0.0004, 0.0008], repeat=3)))   # 27 pts in <1mm
    lone = torch.tensor([[0.05, 0.05, 0.05]])
    means = torch.cat([grid, lone], 0)
    nN = means.shape[0]
    tiny = float(torch.log(torch.tensor(0.0001)))                # 0.1mm splats
    g = {"means": means, "scales": torch.full((nN, 3), tiny),
         "features_dc": torch.rand(nN, 3), "features_rest": torch.zeros(nN, 15, 3),
         "quats": torch.tensor([[1., 0, 0, 0]]).repeat(nN, 1), "opacities": torch.zeros(nN, 1)}
    out = voxel_merge(g, voxel_m=0.001)
    assert out["means"].shape[0] == 2, f"27-in-a-voxel + 1 lone -> 2 merged, got {out['means'].shape[0]}"
    # the merged cluster splat must be sized to the ~0.9mm cluster extent, NOT the 0.1mm input
    merged_max = torch.exp(out["scales"]).max(dim=1).values
    big = float(merged_max.max())
    assert big > 0.0003, f"merged splat must span the cluster (~0.4mm+ sigma), got {big*1e3:.3f}mm (HOLE!)"
    assert big < 0.002, f"merged splat shouldn't blow up past the voxel, got {big*1e3:.3f}mm"
    # the lone splat passes through ~unchanged (still ~0.1mm)
    assert float(merged_max.min()) < 0.0003, "lone splat in its own voxel stays small"
    # robustness: a COLLINEAR cluster + a NaN-poisoned row must NOT crash eigh and give finite output
    col = torch.stack([torch.linspace(0, 0.0009, 20), torch.zeros(20), torch.zeros(20)], dim=1)  # 1-D line
    nbad = col.shape[0]
    gd = {"means": col.clone(), "scales": torch.full((nbad, 3), tiny),
          "features_dc": torch.zeros(nbad, 3), "features_rest": torch.zeros(nbad, 15, 3),
          "quats": torch.tensor([[1., 0, 0, 0]]).repeat(nbad, 1), "opacities": torch.zeros(nbad, 1)}
    gd["means"][0, 0] = float("nan")                             # poison one row
    od = voxel_merge(gd, voxel_m=0.001)
    assert torch.isfinite(od["means"]).all() and torch.isfinite(od["scales"]).all(), \
        "collinear+NaN cluster must merge to FINITE output (no cusolver crash)"

    # ---- _grow_inplane: 2 largest (surface) axes grow x1.5, smallest (normal) untouched ----
    from dynamic_gs2.dynamic_ff_backends import _grow_inplane, _corner_score
    gi = torch.exp(_grow_inplane(torch.log(torch.tensor([[0.010, 0.008, 0.001]])), 1.5))[0]
    assert abs(float(gi[0]) - 0.015) < 1e-6 and abs(float(gi[1]) - 0.012) < 1e-6, "in-plane x1.5"
    assert abs(float(gi[2]) - 0.001) < 1e-6, "normal axis untouched (no blur)"
    # cornerness = MAX(crease, boundary). On a finite flat grid: the INTERIOR is low (flat, neighbours
    # surround it), but the RIM is high (boundary: surface ends -> neighbour centroid shifts inward).
    g = 24
    gx, gy = torch.meshgrid(torch.linspace(0, 0.1, g), torch.linspace(0, 0.1, g), indexing="ij")
    plane = torch.stack([gx.reshape(-1), gy.reshape(-1), torch.zeros(g * g)], dim=1)
    cs = _corner_score(plane, 8, 0.10, 0.80).reshape(g, g)
    interior = cs[3:-3, 3:-3]
    assert float(interior.max()) < 0.2, f"flat interior stays low, got {float(interior.max()):.2f}"
    rim = torch.cat([cs[0], cs[-1], cs[:, 0], cs[:, -1]])
    assert float(rim.mean()) > float(interior.mean()) + 0.1, "boundary rim scores higher than interior"

    # ---- _dilate_corner_mask: ONE neighbour hop grows the set but does NOT cascade to everything ----
    from dynamic_gs2.dynamic_ff_backends import _dilate_corner_mask
    line = torch.stack([torch.linspace(0, 0.05, 100), torch.zeros(100), torch.zeros(100)], dim=1)  # 100 pts in a row
    seed = torch.zeros(100, dtype=torch.bool); seed[50] = True   # one corner in the middle
    grown = _dilate_corner_mask(line, seed, halo_k=5)
    ng = int(grown.sum())
    assert ng > 1, "halo must grow the single corner to its neighbours"
    assert ng < 100, f"halo must NOT cascade to all points, got {ng}/100"
    assert ng <= 12, f"one 5-NN hop should grow ~the local neighbourhood only, got {ng}"  # bounded
    assert bool(grown[50]), "the original corner stays flagged"

    # ---- _clamp_log_scale: definitive insert-boundary cap (uniform shrink, shape preserved) ----
    ls = torch.log(torch.tensor([[0.20, 0.10, 0.05]]))           # a 20cm rogue splat
    capped = torch.exp(_clamp_log_scale(ls, 0.02))[0]
    assert abs(float(capped[0]) - 0.02) < 1e-6, f"largest axis clamped to 0.02, got {float(capped[0])}"
    assert abs(float(capped[0] / capped[1]) - 2.0) < 1e-4, "uniform shrink preserves aspect (0.20/0.10)"

    # ---- _rotmat_to_quat_wxyz: correct for ALL angles incl >=120deg (trace<=0 branch) ----
    from dynamic_gs2.dynamic_ff_backends import _rotmat_to_quat_wxyz, _quat_to_rotmat
    import math
    def _R(axis, ang):
        ax = torch.tensor(axis, dtype=torch.float64); ax = ax / ax.norm()
        x, y, z = ax; c, s = math.cos(ang), math.sin(ang); cc = 1 - c
        return torch.tensor([[c + x*x*cc, x*y*cc - z*s, x*z*cc + y*s],
                             [y*x*cc + z*s, c + y*y*cc, y*z*cc - x*s],
                             [z*x*cc - y*s, z*y*cc + x*s, c + z*z*cc]], dtype=torch.float64)
    Rs = torch.stack([_R([0.3, 0.5, 0.81], math.radians(a)) for a in (0, 90, 120, 121, 179, 240)])
    Rback = _quat_to_rotmat(_rotmat_to_quat_wxyz(Rs))
    assert float((Rback - Rs).abs().max()) < 1e-5, "rotmat->quat->rotmat must round-trip at ALL angles (>=120 too)"

    # ---- _knn_indices shared: corner_score with a precomputed knn == fresh-built ----
    from dynamic_gs2.dynamic_ff_backends import _knn_indices, _dilate_corner_mask
    pts = torch.rand(200, 3)
    shared = _knn_indices(pts, 30)                                # one tree, query k=30
    cs_shared = _corner_score(pts, 10, 0.10, 0.80, knn=shared)    # sliced to 11
    cs_fresh = _corner_score(pts, 10, 0.10, 0.80)                 # builds its own
    assert torch.allclose(cs_shared, cs_fresh, atol=1e-5), "shared-knn corner_score must equal fresh-built"
    seed = torch.zeros(200, dtype=torch.bool); seed[0] = True
    h_shared = _dilate_corner_mask(pts, seed, 8, knn=shared)
    h_fresh = _dilate_corner_mask(pts, seed, 8)
    assert torch.equal(h_shared, h_fresh), "shared-knn halo must equal fresh-built (first k of larger query)"

    print("test_change_mask OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
