"""static_persist.py — warm-cache .pt save/load + the warm-restart entrypoint.

The dynamic phase does NOT retrain; it warm-loads the post-fusion scene the static
phase produced (static_scene/static_state.pt) into a fresh GaussianSet + SceneModel.
Persistence goes through GaussianSet (the SSOT), NOT nerfstudio model.state_dict()
(rewrite_spec/static_persist.md, D2). Config-fingerprint tagged with loud-on-drift
warning (ARCH #8); legacy caches (no fingerprint) are accepted with a warning.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, Tuple

import torch

from .config import config_fingerprint
from .gaussian_set import PARAM_NAMES, GaussianSet
from .scene_model import SceneModel

DEFAULT_CACHE_NAME = "static_state.pt"


def warm_cache_path(data_dir, filename: str = DEFAULT_CACHE_NAME) -> Path:
    return Path(data_dir) / "static_scene" / filename


def seed_ply_path(data_dir) -> Path:
    return Path(data_dir) / "static_scene" / "depth_camera_init_points.ply"


def save_warm_cache(gset: GaussianSet, data_dir, cfg=None, *, filename: str = DEFAULT_CACHE_NAME) -> Path:
    """Write the SSOT (params + 4 buffers) + num_points + config fingerprint.

    model_state_dict mirrors the old layout (gauss_params.<name> + buffer names) so the
    old viewer/tools can still read it, but the SOURCE is GaussianSet.state_dict()."""
    sd = gset.state_dict()                       # {gauss_params.<name>, <buffer>} on CPU
    blob = {
        "model_state_dict": sd,
        "num_points": int(gset.num_points),
        "config_fingerprint": config_fingerprint(cfg) if cfg is not None else None,
        "layout": "dynamic_gs2.v1",
    }
    path = warm_cache_path(data_dir, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)
    return path


def load_warm_cache(gset: GaussianSet, cache_path, cfg=None) -> int:
    """Reload params+buffers into `gset` from a .pt. Warns (does not fail) on fingerprint
    drift / legacy cache. Returns num_points loaded."""
    blob = torch.load(Path(cache_path), map_location="cpu", weights_only=False)
    sd = blob["model_state_dict"]
    n = int(blob.get("num_points", sd["gauss_params.means"].shape[0]))
    stored_fp = blob.get("config_fingerprint")
    if cfg is not None and stored_fp is not None:
        cur = config_fingerprint(cfg)
        if cur != stored_fp:
            print(f"[static_persist] WARNING config fingerprint drift: cache={stored_fp} current={cur} "
                  f"— loading anyway (geometry intact; tuning knobs differ).")
    elif stored_fp is None:
        print(f"[static_persist] WARNING legacy cache (no fingerprint) at {cache_path} — accepting.")
    gset.reload_from_state_dict(sd, num_points=n)
    return n


def _read_cache_seed(cache_path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pull means + per-point DC color (SH band-0 -> rgb) to seed the inner model at the right N/scale."""
    blob = torch.load(Path(cache_path), map_location="cpu", weights_only=False)
    sd = blob["model_state_dict"]
    means = sd["gauss_params.means"].float()
    fdc = sd["gauss_params.features_dc"].float()
    if fdc.dim() == 3:
        fdc = fdc.reshape(fdc.shape[0], 3)
    rgb = (fdc * 0.28209479177387814 + 0.5).clamp(0, 1)    # SH2RGB band-0
    return means, rgb


def build_loaded_scene(cfg, device, cache_path, *, phase: str = "dynamic",
                       lock: Optional["threading.RLock"] = None
                       ) -> Tuple[SceneModel, GaussianSet, "threading.RLock"]:
    """Warm-restart entrypoint: build SceneModel seeded from the cache, bind a GaussianSet,
    reload the exact tensors, attach the shared render lock. Returns (scene_model, gset, lock)."""
    lock = lock or threading.RLock()
    seed_xyz, seed_rgb = _read_cache_seed(cache_path)
    sm = SceneModel(cfg, device, seed_xyz=seed_xyz, seed_rgb=seed_rgb, phase=phase)
    sm.attach_render_lock(lock)
    freelist = (phase == "dynamic")          # dynamic: LR=0 + no_grad render make the capacity buffer safe
    gset = GaussianSet(sm, lock, freelist=freelist)
    if freelist:
        sm.set_count_provider(lambda: gset.count)   # render [:count] -> dead capacity rows never rasterize
    load_warm_cache(gset, cache_path, cfg=cfg)
    return sm, gset, lock
