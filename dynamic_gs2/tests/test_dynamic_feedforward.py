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
                          cdn_fn=lambda d: [True], decode_fn=lambda d, r, s: _tensors(20))
    assert w.due(10) and not w.due(5) and not w.due(0)
    w._inflight = True
    assert not w.due(10), "in-flight blocks due"
    w._inflight = False

    # ---- dispatch -> insert happens, on_insert fired ----
    got = {}
    w2 = FeedforwardWorker(g, g._lock, ffc, budget, cdn_fn=lambda d: [True],
                           decode_fn=lambda d, r, s: _tensors(20),
                           on_insert=lambda rng: got.update(n=int(rng.numel())))
    assert w2.dispatch(_dispatch())
    _wait_idle(w2)
    assert g.num_points == 120, f"insert grew scene to {g.num_points}"
    assert got.get("n") == 20 and w2.last_inserted == 20

    # ---- empty CDN -> no insert ----
    g3 = GaussianSet(FakeModel(50), threading.RLock())
    w3 = FeedforwardWorker(g3, g3._lock, ffc, budget,
                           cdn_fn=lambda d: [], decode_fn=lambda d, r, s: _tensors(20))
    w3.dispatch(_dispatch()); _wait_idle(w3)
    assert g3.num_points == 50, "empty CDN inserts nothing"

    # ---- single-in-flight: 2nd dispatch refused while 1st runs ----
    block = threading.Event()
    g4 = GaussianSet(FakeModel(50), threading.RLock())
    w4 = FeedforwardWorker(g4, g4._lock, ffc, budget, cdn_fn=lambda d: [True],
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
    w5 = FeedforwardWorker(g5, g5._lock, ffc, tight, cdn_fn=lambda d: [True],
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
    w6 = FeedforwardWorker(g6, g6._lock, ffc, pb, cdn_fn=lambda d: [True],
                           decode_fn=lambda d, r, s: _tensors(0))
    culled = w6._purge_ff_inserts(d0_instance_id=7)
    assert culled == 15, f"purged {culled} low-opacity FF inserts (expected 15)"
    assert g6.num_points == before - 15
    ids = g6.snapshot().buffers["object_instance_ids"][:, 0]
    assert int((ids == 7).sum()) == 10, "tracked object (id=7) protected"
    assert int((ids == 999).sum()) == 0, "low-opac FF inserts gone"

    # ---- P0 invariant: FF worker never reads a live _latest_ attribute ----
    src = Path(__file__).resolve().parents[1].joinpath("dynamic_feedforward.py").read_text()
    assert "_latest_" not in src, "P0 violated: FF reads a live _latest_ attribute"

    print("test_dynamic_feedforward OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
