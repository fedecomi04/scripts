"""Tests for dynamic_gs2.dynamic_feedforward — P0 frozen-dispatch + load-shed + purge.

Deterministic CPU; injects fake cdn_fn/decode_fn (no AnySplat/GPU). Validates the
architecturally-critical NEW logic; the heavy AnySplat decode is validated in the
integration run. Run: conda run -n dynamic_gs python -m dynamic_gs2.tests.test_dynamic_feedforward
"""
import dataclasses
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

from dynamic_gs2 import config as C
from dynamic_gs2.dynamic_feedforward import FeedforwardDispatch, FeedforwardWorker
from dynamic_gs2.gaussian_set import GaussianSet, GaussTensors


class FakeModel:
    def __init__(self, n=100, sh_rest_dim=15):
        self._device = torch.device("cpu")
        self.rebind_calls = 0
        g = {
            "means": torch.nn.Parameter(torch.randn(n, 3)),
            "features_dc": torch.nn.Parameter(torch.randn(n, 3)),
            "features_rest": torch.nn.Parameter(torch.randn(n, sh_rest_dim, 3)),
            "scales": torch.nn.Parameter(torch.randn(n, 3)),
            "quats": torch.nn.Parameter(torch.randn(n, 4)),
            "opacities": torch.nn.Parameter(torch.randn(n, 1)),
        }
        self.gauss_params = g

    @property
    def device(self):
        return self._device

    def rebind(self):
        self.rebind_calls += 1


def _tensors(m, opacity_logit=2.0, sh_rest_dim=15):
    return GaussTensors(
        means=torch.randn(m, 3), features_dc=torch.randn(m, 3),
        features_rest=torch.zeros(m, sh_rest_dim, 3), scales=torch.randn(m, 3),
        quats=torch.tensor([[1., 0, 0, 0]]).repeat(m, 1),
        opacities=torch.full((m, 1), float(opacity_logit)))


def _dispatch(d0=7):
    z = torch.zeros((4, 5))
    return FeedforwardDispatch(seq=1, camera=None, rgb_bgr=np.zeros((4, 5, 3), np.uint8),
                               depth_m=z, object_mask=z, gripper_keep=z,
                               scene_intr={}, d0_instance_id=d0)


def _wait_idle(w, timeout=5.0):
    t0 = time.time()
    while w.in_flight() and time.time() - t0 < timeout:
        time.sleep(0.005)
    w.close()


def main():
    cfg = C.load_runtime_config()
    # cull_before_decode off here: the cull needs a real camera+scene+depth (exercised in the
    # integration/live path); this test validates the worker orchestration with a [True] CDN sentinel.
    ffc = dataclasses.replace(cfg.feedforward, cadence_ticks=10, insert_id=999, cull_before_decode=False)

    # ---- due() gate ----
    g = GaussianSet(FakeModel(100), threading.RLock())
    budget = dataclasses.replace(cfg.budget, live_gaussian_ceiling=10_000_000)
    w = FeedforwardWorker(g, g._lock, ffc, budget,
                          cdn_fn=lambda d, **k: [True], decode_fn=lambda d, r, s: _tensors(20))
    assert w.due(10) and not w.due(5) and not w.due(0)
    w._inflight = True
    assert not w.due(10), "in-flight blocks due"
    w._inflight = False

    # ---- dispatch -> insert happens, on_insert fired ----
    got = {}
    w2 = FeedforwardWorker(g, g._lock, ffc, budget, cdn_fn=lambda d, **k: [True],
                           decode_fn=lambda d, r, s: _tensors(20),
                           on_insert=lambda rng: got.update(n=int(rng.numel())))
    assert w2.dispatch(_dispatch())
    _wait_idle(w2)
    assert g.num_points == 120, f"insert grew scene to {g.num_points}"
    assert got.get("n") == 20 and w2.last_inserted == 20

    # ---- empty CDN -> no insert ----
    g3 = GaussianSet(FakeModel(50), threading.RLock())
    w3 = FeedforwardWorker(g3, g3._lock, ffc, budget,
                           cdn_fn=lambda d, **k: [], decode_fn=lambda d, r, s: _tensors(20))
    w3.dispatch(_dispatch()); _wait_idle(w3)
    assert g3.num_points == 50, "empty CDN inserts nothing"

    # ---- single-in-flight: 2nd dispatch refused while 1st runs ----
    block = threading.Event()
    g4 = GaussianSet(FakeModel(50), threading.RLock())
    w4 = FeedforwardWorker(g4, g4._lock, ffc, budget, cdn_fn=lambda d, **k: [True],
                           decode_fn=lambda d, r, s: (block.wait(2.0), _tensors(5))[1])
    assert w4.dispatch(_dispatch())
    time.sleep(0.05)
    assert not w4.dispatch(_dispatch()), "single-in-flight: 2nd dispatch refused"
    block.set(); _wait_idle(w4)
    assert g4.num_points == 55

    # ---- load-shed ceiling: trim batch to fit ----
    g5 = GaussianSet(FakeModel(95), threading.RLock())
    tight = dataclasses.replace(cfg.budget, live_gaussian_ceiling=100,
                                dynamic_purge_opacity_below=0.0)   # no purge -> pure trim
    w5 = FeedforwardWorker(g5, g5._lock, ffc, tight, cdn_fn=lambda d, **k: [True],
                           decode_fn=lambda d, r, s: _tensors(20))
    w5.dispatch(_dispatch()); _wait_idle(w5)
    assert g5.num_points == 100, f"trimmed to ceiling, got {g5.num_points}"
    assert w5.last_inserted == 5

    # ---- purge protects the tracked object, drops low-opacity FF inserts ----
    g6 = GaussianSet(FakeModel(40), threading.RLock())
    g6.insert(_tensors(10, opacity_logit=5.0), object_flag=1.0, instance_id=7)   # tracked (high opac)
    g6.insert(_tensors(15, opacity_logit=-5.0), object_flag=1.0, instance_id=999)  # FF, low opac
    before = g6.num_points
    pb = dataclasses.replace(cfg.budget, dynamic_purge_opacity_below=0.05)
    w6 = FeedforwardWorker(g6, g6._lock, ffc, pb, cdn_fn=lambda d, **k: [True],
                           decode_fn=lambda d, r, s: _tensors(0))
    culled = w6._purge_ff_inserts(d0_instance_id=7)
    assert culled == 15, f"purged {culled} low-opacity FF inserts (expected 15)"
    assert g6.num_points == before - 15
    ids = g6.snapshot().buffers["object_instance_ids"][:, 0]
    assert int((ids == 7).sum()) == 10, "tracked object (id=7) protected"
    assert int((ids == 999).sum()) == 0, "low-opac FF inserts gone"

    # ---- Option A: deferred cull commits ATOMICALLY with insert (no count dip a render can see) ----
    # Free-list GaussianSet; slow decode keeps FF in-flight; a sampler thread polls num_points and
    # must NEVER observe a count below the start (the old flow dipped to start-culled for ~decode).
    import dataclasses as _dc
    ffc_def = _dc.replace(cfg.feedforward, cadence_ticks=10, insert_id=999, cull_before_decode=True)
    g7 = GaussianSet(FakeModel(200), threading.RLock(), freelist=True)
    start_n = g7.num_points
    CULL_K, INS_M = 30, 12
    hidden_calls = []
    block = threading.Event()
    def _slow_decode(d, r, s):
        block.wait(2.0)                  # hold FF in-flight so the sampler runs during the gap
        return _tensors(INS_M, sh_rest_dim=15)
    w7 = FeedforwardWorker(g7, g7._lock, ffc_def, budget, cdn_fn=lambda d, **k: [True],
                           decode_fn=_slow_decode, set_hidden_fn=lambda idx: hidden_calls.append(idx))
    # fake the projection-culls (no camera/depth on CPU). The replaced-cull (1mm-behind, committed
    # atomically with the insert) needs a real camera; stub it empty so this test isolates atomicity.
    w7._compute_cull_in_front = lambda d, cdn: (
        torch.arange(CULL_K), torch.zeros(g7.num_points, dtype=torch.bool))
    w7._compute_cull_replaced = lambda d, cdn: torch.empty(0, dtype=torch.long)
    final_n = start_n - CULL_K + INS_M   # net -18: this FF culls 30, inserts 12
    seen = set()
    stop = threading.Event()
    def _sampler():
        while not stop.is_set():
            seen.add(g7.num_points)      # record EVERY distinct count observed during the op
    samp = threading.Thread(target=_sampler, daemon=True); samp.start()
    assert w7.dispatch(_dispatch())
    time.sleep(0.1)                      # let it reach the slow decode while sampler watches
    block.set()                          # release decode -> atomic cull_and_insert commits
    _wait_idle(w7)
    stop.set(); samp.join(timeout=1.0)
    assert g7.num_points == final_n, f"final {g7.num_points} != {final_n}"
    # ATOMICITY: the ONLY counts a concurrent reader may observe are start (before) or final (after).
    # The old flow exposed the cull-only transient start-CULL_K=170 (< final 182); the atomic single
    # _count write must make that intermediate UNobservable.
    bad = {c for c in seen if c not in (start_n, final_n)}
    assert not bad, f"reader saw intermediate count(s) {sorted(bad)} — cull_and_insert not atomic!"
    assert (start_n - CULL_K) not in seen, "cull-only dip (170) was observable — not atomic!"
    assert hidden_calls and hidden_calls[-1] is None, "hidden indices must be cleared after the re-CDN"
    assert w7.last_culled == CULL_K and w7.last_inserted == INS_M

    # ---- folded periodic purge: low-opacity FF inserts purged WITHIN the atomic cull_and_insert ----
    # deferred path + purge-every-1; no in-front cull -> the purge indices are the ONLY cull, folded
    # into the same cull_and_insert as the decode insert (one surgery, not insert-then-purge).
    g8 = GaussianSet(FakeModel(50), threading.RLock(), freelist=True)
    g8.insert(_tensors(20, opacity_logit=-5.0, sh_rest_dim=15), object_flag=1.0, instance_id=999)  # low-opac FF
    g8.insert(_tensors(5, opacity_logit=5.0, sh_rest_dim=15), object_flag=1.0, instance_id=7)       # tracked d0
    start8 = g8.num_points                                   # 50 + 20 + 5 = 75
    pb8 = _dc.replace(cfg.budget, dynamic_purge_opacity_below=0.05, dynamic_purge_every_n_ff=1,
                      live_gaussian_ceiling=10_000_000)
    INS8 = 6
    w8 = FeedforwardWorker(g8, g8._lock, ffc_def, pb8, cdn_fn=lambda d, **k: [True],
                           decode_fn=lambda d, r, s: _tensors(INS8, opacity_logit=5.0, sh_rest_dim=15),
                           set_hidden_fn=lambda idx: None)
    w8._compute_cull_in_front = lambda d, cdn: (          # no in-front cull -> purge is the only cull
        torch.empty(0, dtype=torch.long), torch.zeros(g8.num_points, dtype=torch.bool))
    w8._compute_cull_replaced = lambda d, cdn: torch.empty(0, dtype=torch.long)   # needs real camera; stub empty
    w8.dispatch(_dispatch(d0=7)); _wait_idle(w8)
    ids8 = g8.snapshot().buffers["object_instance_ids"][:, 0]
    assert g8.num_points == start8 - 20 + INS8, f"purge+insert folded: {g8.num_points} != {start8-20+INS8}"
    assert int((ids8 == 7).sum()) == 5, "tracked d0 protected through folded purge"
    assert int((ids8 == 999).sum()) == INS8, "old low-opac inserts purged; only the new high-opac insert remains"
    assert w8.last_culled == 20 and w8.last_inserted == INS8

    # ---- _compute_cull_replaced projection: cull eligible gaussians AT-OR-BEHIND the live surface ----
    # Real camera looking down -Z (OpenGL), surface at 1.0 m. Place 4 gaussians on the optical axis
    # (centre pixel, so they land inside the all-ones CDN): in-front 0.9 m, AT 1.0 m, BEHIND 1.1 m
    # (all eligible id=0), plus a tracked-object gaussian (id=7) at 1.1 m that MUST be protected.
    from nerfstudio.cameras.cameras import Cameras, CameraType
    W9 = H9 = 8
    fx9 = fy9 = 8.0
    cam9 = Cameras(camera_to_worlds=torch.eye(4)[:3, :4].unsqueeze(0),
                   fx=fx9, fy=fy9, cx=W9 / 2.0, cy=H9 / 2.0, width=W9, height=H9,
                   camera_type=CameraType.PERSPECTIVE)
    depth9 = torch.full((H9, W9), 1.0)                       # live surface 1 m everywhere
    # The replaced-cull keeps ONLY the thin slab [sensor, sensor + tol(0.5mm)] just behind the surface.
    # OpenGL: camera at origin looking down -Z. A point at world z=-Z is Z metres in front. Surface=1.0m.
    front = torch.tensor([0.0, 0.0, -0.90])                 # 0.90 m  -> in front -> NOT replaced-cull's job
    at    = torch.tensor([0.0, 0.0, -1.0000])               # at surface (slab start) -> culled
    inslab= torch.tensor([0.0, 0.0, -1.0003])               # 0.3mm behind (inside 0.5mm slab) -> culled
    behind= torch.tensor([0.0, 0.0, -1.02])                 # 2cm behind (BEYOND slab) -> KEPT (was the bug)
    deep  = torch.tensor([0.0, 0.0, -1.50])                 # 50cm behind (BEYOND slab) -> KEPT
    trk   = torch.tensor([0.0, 0.0, -1.0003])               # tracked (id=7) in slab -> PROTECTED
    g9 = GaussianSet(FakeModel(0), threading.RLock(), freelist=True)
    def _one(xyz, oid):
        t = _tensors(1, opacity_logit=2.0, sh_rest_dim=15); t.means[:] = xyz
        g9.insert(t, object_flag=(1.0 if oid == 7 else 0.0), instance_id=oid)
    _one(front, 0); _one(at, 0); _one(inslab, 0); _one(behind, 0); _one(deep, 0); _one(trk, 7)
    w9 = FeedforwardWorker(g9, g9._lock, ffc_def, budget, cdn_fn=lambda d, **k: [True],
                           decode_fn=lambda d, r, s: _tensors(0))
    d9 = FeedforwardDispatch(seq=1, camera=cam9, rgb_bgr=np.zeros((H9, W9, 3), np.uint8),
                             depth_m=depth9, object_mask=torch.zeros((H9, W9)),
                             gripper_keep=torch.ones((H9, W9)), scene_intr={}, d0_instance_id=7)
    cdn9 = np.ones((H9, W9), dtype=bool)                     # whole frame is "changed"
    idx9 = w9._compute_cull_replaced(d9, cdn9)
    culled = set(int(i) for i in idx9.tolist())
    assert 1 in culled and 2 in culled, f"AT (row1) + in-slab 0.3mm (row2) must be culled, got {sorted(culled)}"
    assert 0 not in culled, "the in-front gaussian (row0) is NOT the replaced-cull's job"
    assert 3 not in culled, "2cm-behind (row3) is BEYOND the 0.5mm slab -> must survive (this was the over-cull bug)"
    assert 4 not in culled, "DEEP background (row4, 50cm behind) must survive"
    assert 5 not in culled, "tracked object (id=7, row5) must be protected"
    # disabled (tol < 0) -> empty
    w9.cfg = dataclasses.replace(ffc_def, cull_replaced_depth_tol_m=-1.0)
    assert w9._compute_cull_replaced(d9, cdn9).numel() == 0, "cull_replaced_depth_tol_m<0 disables it"

    # ---- P0 invariant: FF worker never reads a live _latest_ attribute ----
    src = Path(__file__).resolve().parents[1].joinpath("dynamic_feedforward.py").read_text()
    assert "_latest_" not in src, "P0 violated: FF reads a live _latest_ attribute"

    print("test_dynamic_feedforward OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
