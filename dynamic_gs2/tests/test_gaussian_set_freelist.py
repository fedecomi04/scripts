"""Tests for the GaussianSet FREE-LIST path (freelist=True, dynamic phase).

Proves the capacity/count + swap-remove cull + tail-write insert behave correctly AND are
content-equivalent to the proven reallocating path under the same op sequence (the strongest
check — the free-list is a perf optimization, so it must be a no-op on observable state).

Swap-remove REORDERS rows, so equivalence is checked order-independently (sort both snapshots
by a stable per-row key built from means+ids). CPU-only fake model. Run:
    conda run -n dynamic_gs python -m dynamic_gs2.tests.test_gaussian_set_freelist
"""
import sys
import threading

import torch

from dynamic_gs2 import gaussian_set as G
from dynamic_gs2.gaussian_set import GaussianSet, GaussTensors
from dynamic_gs2.tests.test_gaussian_set import FakeSceneModel


def _tensors_seeded(m, seed, sh_rest_dim=15):
    """Deterministic insert batch keyed by seed so the two sets get IDENTICAL new rows."""
    g = torch.Generator().manual_seed(seed)
    return GaussTensors(
        means=torch.randn(m, 3, generator=g),
        features_dc=torch.randn(m, 3, generator=g),
        features_rest=torch.randn(m, sh_rest_dim, 3, generator=g),
        scales=torch.randn(m, 3, generator=g),
        quats=torch.randn(m, 4, generator=g),
        opacities=torch.randn(m, 1, generator=g))


def _row_key(snap):
    """Stable per-row fingerprint (means + instance id) for order-independent compare."""
    means = snap.params["means"]
    ids = snap.buffers["object_instance_ids"].float()
    return torch.cat([means, ids], dim=1)


def _assert_same_set(a, b, msg):
    """a,b = snapshots. Same row MULTISET regardless of order."""
    assert a.num_points == b.num_points, f"{msg}: count {a.num_points} != {b.num_points}"
    ka, kb = _row_key(a), _row_key(b)
    # sort rows lexicographically by a scalar hash of the key
    def _sort(k):
        w = torch.tensor([1.0, 1.3, 1.7, 2.1], dtype=k.dtype)[: k.shape[1]]
        order = torch.argsort((k * w).sum(dim=1))
        return k[order]
    assert torch.allclose(_sort(ka), _sort(kb), atol=1e-5), f"{msg}: row sets differ"


def _identical_pair(n=20):
    """Two GaussianSets seeded byte-identically; one freelist, one not."""
    torch.manual_seed(123)
    m1 = FakeSceneModel(n=n, sh_rest_dim=15)
    torch.manual_seed(123)
    m2 = FakeSceneModel(n=n, sh_rest_dim=15)
    base = GaussianSet(m1, threading.RLock(), freelist=False)
    free = GaussianSet(m2, threading.RLock(), freelist=True)
    return base, free


def _cull_by_tag(g, tags_to_del):
    """Delete rows whose means[:,0] tag is in tags_to_del, locating them in g's CURRENT order.
    (Swap-remove REORDERS rows, so a fixed index means different rows in the two sets — but the
    real pipeline always derives cull indices from a fresh snapshot, never a stale index. Culling
    by content models that and makes the two paths comparable.)"""
    tag = g.snapshot().params["means"][:, 0].round().int()
    want = torch.zeros_like(tag, dtype=torch.bool)
    for t in tags_to_del:
        want |= (tag == t)
    g.cull(torch.nonzero(want, as_tuple=False).flatten())


def main():
    # ---- 1. EQUIVALENCE ORACLE: same ops -> same observable state ----
    # Tag every row uniquely in means[:,0] so culls can target the SAME gaussian in both sets
    # regardless of internal row order (swap-remove reorders; the reallocating path doesn't).
    base, free = _identical_pair(20)
    _next_tag = [0]
    def _tagged_insert(m, iid):
        seed = 1000 + m * 7 + iid
        tb = _tensors_seeded(m, seed)
        tf = _tensors_seeded(m, seed)
        tags = torch.arange(_next_tag[0], _next_tag[0] + m, dtype=torch.float32)
        _next_tag[0] += m
        tb.means[:, 0] = tags; tf.means[:, 0] = tags
        base.insert(tb, object_flag=1.0, instance_id=iid)
        free.insert(tf, object_flag=1.0, instance_id=iid)
    # tag the initial 20 rows too (identical in both since same seed)
    with torch.no_grad():
        for g in (base, free):
            g._params()["means"][:, 0] = torch.arange(20, dtype=torch.float32)
    _next_tag[0] = 20
    ops = [("ins", 5, 7), ("ins", 4, 9), ("del", [1, 3]), ("ins", 6, 7),
           ("del", [0, 5, 19]), ("ins", 3, 9), ("del", [2, 21, 24, 25, 27, 28])]
    for op in ops:
        if op[0] == "ins":
            _tagged_insert(op[1], op[2])
        else:
            _cull_by_tag(base, op[1]); _cull_by_tag(free, op[1])
        assert base.num_points == free.num_points, f"count drift after {op}"
    _assert_same_set(base.snapshot(), free.snapshot(), "after op sequence")
    # state_dict exports only live rows, same set
    assert free.state_dict()["gauss_params.means"].shape[0] == free.num_points
    print(f"  [1] equivalence: {free.num_points} live rows match reallocating path")

    # ---- 2. swap-remove keeps the RIGHT rows (content, by unique tag in means[:,0]) ----
    torch.manual_seed(0)
    m = FakeSceneModel(n=0 if False else 6, sh_rest_dim=15)
    g = GaussianSet(m, threading.RLock(), freelist=True)
    # overwrite means[:,0] with 0..5 as a tag, then cull {1,4} -> survivors tags {0,2,3,5}
    with torch.no_grad():
        g._params()["means"][:, 0] = torch.arange(6, dtype=torch.float32)
    g.cull(torch.tensor([1, 4]))
    assert g.num_points == 4
    tags = set(g.snapshot().params["means"][:, 0].round().int().tolist())
    assert tags == {0, 2, 3, 5}, f"swap-remove kept wrong rows: {tags}"
    print("  [2] swap-remove keeps exactly the non-culled rows (by content tag)")

    # ---- 3. capacity grows on overflow; dead rows invisible ----
    torch.manual_seed(0)
    g = GaussianSet(FakeSceneModel(n=10, sh_rest_dim=15), threading.RLock(), freelist=True)
    assert g._capacity == 10 and g._count == 10
    g.cull(torch.tensor([0, 1, 2, 3]))                 # count 6, capacity STILL 10 (no realloc)
    assert g._count == 6 and g._capacity == 10, "cull must not shrink capacity"
    phys_before = g._params()["means"].shape[0]
    g.insert(_tensors_seeded(2, 1), object_flag=1.0, instance_id=1)   # 6+2=8 <= 10: no realloc
    assert g._count == 8 and g._params()["means"].shape[0] == phys_before, "insert into dead region"
    g.insert(_tensors_seeded(20, 2), object_flag=1.0, instance_id=1)  # 8+20=28 > 10: grow
    assert g._count == 28 and g._capacity >= 28, f"capacity grew to {g._capacity}"
    assert g.snapshot().num_points == 28, "snapshot sees only live rows"
    assert g._params()["means"].shape[0] == g._capacity, "invariant: phys len == capacity"
    print(f"  [3] capacity: realloc only on overflow (cap={g._capacity}, count=28)")

    # ---- 4. protect_mask never drops the protected row (free-list path) ----
    torch.manual_seed(0)
    g = GaussianSet(FakeSceneModel(n=8, sh_rest_dim=15), threading.RLock(), freelist=True)
    with torch.no_grad():
        g._params()["means"][:, 0] = torch.arange(8, dtype=torch.float32)
    prot = torch.zeros(8, dtype=torch.bool); prot[2] = True
    deleted = g.cull(torch.tensor([2, 3, 5]), protect_mask=prot)        # 2 protected -> only 3,5 die
    assert deleted == 2 and g.num_points == 6
    tags = set(g.snapshot().params["means"][:, 0].round().int().tolist())
    assert 2 in tags and 3 not in tags and 5 not in tags, f"protect failed: {tags}"
    print("  [4] protect_mask: protected row survives swap-remove cull")

    # ---- 5. write_object_pose / flags / ids index live rows under capacity ----
    torch.manual_seed(0)
    g = GaussianSet(FakeSceneModel(n=8, sh_rest_dim=15), threading.RLock(), freelist=True)
    g.cull(torch.tensor([0, 1]))                        # count 6, capacity 8 (2 dead rows)
    n = g.num_points
    mask = torch.zeros(n, dtype=torch.bool); mask[0:3] = True
    uids3 = g.snapshot().buffers["gauss_uid"][:, 0][0:3]   # write the first 3 LIVE rows BY uid
    g.write_object_pose(torch.full((3, 3), 5.0), torch.tensor([[1., 0, 0, 0]]).repeat(3, 1), uids3)
    assert torch.allclose(g.snapshot().params["means"][0:3], torch.full((3, 3), 5.0))
    g.set_object_flags(mask, 1.0)
    g.write_instance_ids(mask, 42)
    snap = g.snapshot()
    assert torch.all(snap.buffers["object_flags"][0:3] == 1.0)
    assert torch.all(snap.buffers["object_instance_ids"][0:3] == 42)
    assert snap.buffers["object_flags"].shape[0] == 6, "buffers expose only live rows"
    print("  [5] in-place writers index live rows correctly under dead capacity")

    # ---- 6. cull_and_insert == cull-then-insert (atomic surgery, both modes) ----
    for fl in (True, False):
        torch.manual_seed(7)
        ma = FakeSceneModel(n=30, sh_rest_dim=15)
        torch.manual_seed(7)
        mb = FakeSceneModel(n=30, sh_rest_dim=15)
        ga = GaussianSet(ma, threading.RLock(), freelist=fl)   # atomic path
        gb = GaussianSet(mb, threading.RLock(), freelist=fl)   # sequential reference
        for g in (ga, gb):
            with torch.no_grad():
                g._params()["means"][:, 0] = torch.arange(30, dtype=torch.float32)
        ins = _tensors_seeded(8, 555)
        ins.means[:, 0] = torch.arange(100, 108, dtype=torch.float32)   # distinct tags, can't collide with 0..29
        cull = torch.tensor([2, 5, 9, 11])
        prot = torch.zeros(30, dtype=torch.bool); prot[5] = True   # 5 protected -> survives
        # atomic
        nca, _ = ga.cull_and_insert(cull, ins, object_flag=1.0, instance_id=999, protect_mask=prot)
        # sequential reference (same protect)
        ncb = gb.cull(cull, protect_mask=prot)
        gb.insert(ins, object_flag=1.0, instance_id=999)
        assert nca == ncb == 3, f"culled {nca}/{ncb} (expected 3; row 5 protected)"
        assert ga.num_points == gb.num_points == 35
        _assert_same_set(ga.snapshot(), gb.snapshot(), f"cull_and_insert freelist={fl}")
        # protected row 5 still present; culled 2,9,11 gone; inserted id=999 present
        snap = ga.snapshot()
        tags = set(snap.params["means"][:, 0].round().int().tolist())
        assert 5 in tags and 2 not in tags and 9 not in tags and 11 not in tags
        assert int((snap.buffers["object_instance_ids"][:, 0] == 999).sum()) == 8
    print("  [6] cull_and_insert == cull-then-insert (atomic, both modes)")

    # ---- 7. stable gauss_uid: each row's uid travels with its DATA through every surgery ----
    # (Identity tag that makes ReferenceObjectPose reorder-proof — the Fix-2 guarantee.)
    torch.manual_seed(3)
    g = GaussianSet(FakeSceneModel(n=12, sh_rest_dim=15), threading.RLock(), freelist=True)
    with torch.no_grad():
        g._params()["means"][:, 0] = torch.arange(12, dtype=torch.float32)   # tag = uid at start
    snap0 = g.snapshot()
    # uid starts as 0..11, equal to the means[:,0] tag
    assert torch.equal(snap0.buffers["gauss_uid"][:, 0], torch.arange(12)), "uid init = arange"
    # build a tag->uid truth map, then reorder via cull and check uid still tracks the same DATA row
    def tag_to_uid(g):
        s = g.snapshot()
        return {int(t): int(u) for t, u in zip(s.params["means"][:, 0].round().int().tolist(),
                                               s.buffers["gauss_uid"][:, 0].tolist())}
    before = tag_to_uid(g)
    g.cull(torch.tensor([1, 4, 7]))                        # reorders survivors via swap-remove
    after = tag_to_uid(g)
    for tag, uid in after.items():
        assert before[tag] == uid, f"uid for data tag {tag} changed {before[tag]}->{uid} (uid must track data!)"
    # insert assigns FRESH uids (>= old max), never reused
    rng = g.insert(_tensors_seeded(3, 1), object_flag=1.0, instance_id=999)
    new_uids = g.snapshot().buffers["gauss_uid"][:, 0][-3:]
    assert int(new_uids.min()) >= 12, f"new uids must be fresh (>=12), got {new_uids.tolist()}"
    assert len(set(g.snapshot().buffers["gauss_uid"][:, 0].tolist())) == g.num_points, "all uids unique"
    print("  [7] gauss_uid tracks each row's DATA through cull reorder + fresh on insert")

    print("test_gaussian_set_freelist OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
