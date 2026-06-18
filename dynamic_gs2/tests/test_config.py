"""Tests for dynamic_gs2.config — defaults, env overrides, #17 sim/real, validation, fingerprint, reload.

Run: python -m dynamic_gs2.tests.test_config   (from scripts/)
"""
import os
import sys

from dynamic_gs2 import config as C


def _clear_dgs_env():
    for k in [k for k in os.environ if k.startswith("DGS_")]:
        del os.environ[k]


def main():
    _clear_dgs_env()

    # defaults
    cfg = C.load_runtime_config()
    assert cfg.source == "sim"
    assert cfg.tracker.top_k == 1024
    assert cfg.pose_filter.accel_sigma == 0.005 and cfg.pose_filter.alpha_sigma == 0.025  # verified tuned
    assert cfg.background_color == (0.86, 0.92, 1.0)
    assert cfg.depth.scene_depth_max_m == cfg.depth.depth_max_m
    assert cfg.sim_noise.enabled is True, "sim source -> noise on"

    # #17: real source auto-disables sim noise
    os.environ["DGS_SOURCE"] = "real"
    creal = C.load_runtime_config()
    assert creal.source == "real" and creal.sim_noise.enabled is False, "real -> noise OFF"
    _clear_dgs_env()

    # env overrides apply
    os.environ["DGS_XFEAT_TOP_K"] = "2048"
    os.environ["DGS_FF_ICP"] = "0"
    os.environ["DGS_TSDF_DEPTH_MAX_M"] = "3.0"
    c2 = C.load_runtime_config()
    assert c2.tracker.top_k == 2048
    assert c2.feedforward.icp_refine is False
    assert c2.depth.depth_max_m == 3.0 and c2.depth.scene_depth_max_m == 3.0  # kept equal
    _clear_dgs_env()

    # fingerprint: stable for same cfg, changes on a knob
    f1 = C.config_fingerprint(C.load_runtime_config())
    f1b = C.config_fingerprint(C.load_runtime_config())
    assert f1 == f1b, "fingerprint must be stable"
    os.environ["DGS_XFEAT_TOP_K"] = "512"
    assert C.config_fingerprint(C.load_runtime_config()) != f1, "fingerprint must change on drift"
    _clear_dgs_env()

    # reload_overrides swaps ONLY the whitelist, atomically (new object), still valid
    base = C.load_runtime_config()
    os.environ["DGS_FF_MAX_SCALE_M"] = "0.01"
    os.environ["DGS_HOLD_WINDOW"] = "20"
    os.environ["DGS_XFEAT_TOP_K"] = "999"          # NOT in whitelist -> must be ignored
    rel = C.reload_overrides(base)
    assert rel is not base
    assert rel.feedforward.max_scale_m == 0.01
    assert rel.tracker.static_hold_window == 20
    assert rel.tracker.top_k == base.tracker.top_k, "non-whitelisted knob must not reload"
    _clear_dgs_env()

    # validation: background tamper raises
    bad = C.dataclasses.replace(C.load_runtime_config(), background_color=(0.0, 0.0, 0.0))
    try:
        C._validate(bad)
        raise AssertionError("expected background invariant violation")
    except ValueError:
        pass

    print("test_config OK")


if __name__ == "__main__":
    sys.exit(main())
