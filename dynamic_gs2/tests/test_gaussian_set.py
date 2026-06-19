"""Tests for dynamic_gs2.gaussian_set — SSOT surgery, snapshot, buffers, reload, helpers.

CPU-only against a fake scene model (no gsplat/nerfstudio). Run:
    conda run -n dynamic_gs python -m dynamic_gs2.tests.test_gaussian_set   (from scripts/)
"""
import sys
import threading

import torch

from dynamic_gs2 import gaussian_set as G
from dynamic_gs2.gaussian_set import GaussianSet, GaussTensors


class FakeSceneModel:
    """Mimics the scene_model seam GaussianSet binds to: a gauss_params dict + rebind()."""

    def __init__(self, n=20, sh_rest_dim=15, device="cpu"):
        self._device = torch.device(device)
        self.rebind_calls = 0
        g = {}
        g["means"] = torch.nn.Parameter(torch.randn(n, 3))
        g["features_dc"] = torch.nn.Parameter(torch.randn(n, 3))
        g["features_rest"] = torch.nn.Parameter(torch.randn(n, sh_rest_dim, 3))
        g["scales"] = torch.nn.Parameter(torch.randn(n, 3))
        g["quats"] = torch.nn.Parameter(torch.randn(n, 4))
        g["opacities"] = torch.nn.Parameter(torch.randn(n, 1))
        self.gauss_params = g

    @property
    def device(self):
        return self._device

    def rebind(self):
        self.rebind_calls += 1


def _mk_tensors(m, sh_rest_dim=15):
    return GaussTensors(
        means=torch.randn(m, 3),
        features_dc=torch.randn(m, 3),
        features_rest=torch.randn(m, sh_rest_dim, 3),
        scales=torch.randn(m, 3),
        quats=torch.randn(m, 4),
        opacities=torch.randn(m, 1),
    )


def main():
    torch.manual_seed(0)
    model = FakeSceneModel(n=20, sh_rest_dim=15)
    gset = GaussianSet(model, threading.RLock())

    # init: buffers allocated at N, zeros, invariant
    assert gset.num_points == 20
    assert gset.sh_rest_dim == 15
    snap0 = gset.snapshot()
    assert snap0.num_points == 20 and len(snap0) == 20
    assert snap0.version == 0
    for name, _ in G.IDENTITY_BUFFER_SPECS:
        assert snap0.buffers[name].shape == (20, 1)
        assert float(snap0.buffers[name].abs().sum()) == 0.0
    assert model.object_flags is gset._buffers["object_flags"]   # accessible as model.<name>

    # insert: count grows, tail identity written, rebind called, requires_grad preserved
    rng = gset.insert(_mk_tensors(5), object_flag=1.0, instance_id=7)
    assert gset.num_points == 25
    assert torch.equal(rng, torch.arange(20, 25))
    assert model.rebind_calls == 1
    assert gset._params()["means"].requires_grad is True
    snap1 = gset.snapshot()
    assert snap1.version == 1
    assert torch.all(snap1.buffers["object_flags"][20:] == 1.0)
    assert torch.all(snap1.buffers["object_flags"][:20] == 0.0)
    assert torch.all(snap1.buffers["object_instance_ids"][20:] == 7)
    assert torch.all(snap1.buffers["inserted_flags"][20:] == 1.0)
    assert torch.all(snap1.buffers["sam3d_init_target_flags"] == 0.0)  # never written (Inv #8)

    # snapshot count is FROZEN across a later insert (point-in-time)
    gset.insert(_mk_tensors(3), object_flag=0.0, instance_id=0)
    assert gset.num_points == 28
    assert snap1.num_points == 25, "old snapshot count must not change"

    # cull with protect_mask: protected rows survive even if in the delete set
    n = gset.num_points
    protect = torch.zeros(n, dtype=torch.bool); protect[0] = True
    to_del = torch.tensor([0, 1, 2, 2, 999, -1])   # dup + oob tolerated
    deleted = gset.cull(to_del, protect_mask=protect)
    assert deleted == 2, f"deleted {deleted} (0 protected, 1&2 real, dups/oob clipped)"
    assert gset.num_points == n - 2
    assert model.rebind_calls == 3

    # write_object_pose: uid-keyed (resolves uid->live row under the lock), in-place, no count change
    n = gset.num_points
    uids = gset.snapshot().buffers["gauss_uid"][:, 0][0:4]   # write to the first 4 rows BY uid
    new_means = torch.full((4, 3), 9.0)
    new_quats = torch.tensor([[1.0, 0, 0, 0]]).repeat(4, 1)
    rows = gset.write_object_pose(new_means, new_quats, uids)
    assert rows == 4 and gset.num_points == n
    assert torch.allclose(gset._params()["means"][0:4], new_means)

    # mismatched subset raises
    try:
        gset.write_object_pose(torch.zeros(3, 3), torch.zeros(3, 4), uids)
        raise AssertionError("expected row-count mismatch")
    except AssertionError as e:
        assert "subset" in str(e) or "!=" in str(e)

    # a uid culled since the snapshot is silently skipped (FF-race safety)
    uids5 = torch.cat([uids, torch.tensor([10_000_000], dtype=uids.dtype)])  # one bogus/absent uid
    rows = gset.write_object_pose(torch.full((5, 3), 1.0), torch.tensor([[1., 0, 0, 0]]).repeat(5, 1), uids5)
    assert rows == 4, f"absent uid skipped, wrote {rows} (expected 4)"

    # set_object_flags / write_instance_ids in place
    fmask = torch.zeros(gset.num_points, dtype=torch.bool); fmask[5:8] = True
    gset.set_object_flags(fmask, 1.0)
    assert torch.all(gset._buffers["object_flags"][5:8] == 1.0)
    gset.write_instance_ids(fmask, 3)
    assert torch.all(gset._buffers["object_instance_ids"][5:8] == 3)

    # reload_from_state_dict: realloc at a new N, buffers rebuilt
    sd = {f"gauss_params.{k}": torch.randn_like(model.gauss_params[k][:12]) for k in G.PARAM_NAMES}
    sd["object_flags"] = torch.ones(12, 1)
    gset.reload_from_state_dict(sd, num_points=12)
    assert gset.num_points == 12
    assert torch.all(gset._buffers["object_flags"] == 1.0)
    assert torch.all(gset._buffers["inserted_flags"] == 0.0)   # absent in sd -> zero-filled

    # state_dict round-trips shape
    out = gset.state_dict()
    assert out["gauss_params.means"].shape == (12, 3)
    assert out["object_flags"].shape == (12, 1)

    # helpers
    logits = torch.tensor([-10.0, 0.0, 10.0]).unsqueeze(1)   # sig ~0, 0.5, ~1
    li = G.low_opacity_indices(logits, 0.05)
    assert li.tolist() == [0]
    ls = torch.log(torch.tensor([[0.5, 0.01, 0.01], [0.005, 0.005, 0.005]]))
    new_ls, keep = G.uniform_shrink_log_scales(ls, max_scale_m=0.05, min_scale_m=0.0)
    shr = torch.exp(new_ls[0])
    assert abs(float(shr.max()) - 0.05) < 1e-5, "largest axis shrunk to cap"
    assert abs(float(shr[0] / shr[1]) - 50.0) < 1e-3, "aspect preserved (0.5/0.01)"
    _, keep2 = G.uniform_shrink_log_scales(ls, max_scale_m=0.0, min_scale_m=0.02)
    assert keep2.tolist() == [True, False], "tiny gaussian culled by min_scale"

    gt = G.build_default_gauss_tensors(torch.randn(8, 3), torch.rand(8, 3),
                                       sh_degree=3, sh_rest_dim=15, device="cpu", dtype=torch.float32)
    gt.validate(15)
    assert gt.means.shape == (8, 3) and gt.features_rest.shape == (8, 15, 3)
    assert gt.quats.shape == (8, 4) and torch.all(gt.quats[:, 0] == 1.0)
    assert abs(float(torch.sigmoid(gt.opacities[0]) - 0.1)) < 1e-4

    # GaussTensors.validate coerces a short features_rest + 1-D opacity
    bad = GaussTensors(means=torch.randn(4, 3), features_dc=torch.randn(4, 1, 3),
                       features_rest=torch.zeros(4, 0), scales=torch.randn(4, 3),
                       quats=torch.randn(4, 4), opacities=torch.randn(4))
    bad.validate(15)
    assert bad.features_dc.shape == (4, 3)   # (n,1,3) coerced to (n,3)
    assert bad.features_rest.shape == (4, 15, 3)
    assert bad.opacities.shape == (4, 1)

    print("test_gaussian_set OK")


if __name__ == "__main__":
    sys.exit(main())
