"""static_train.py — native 500-step Splatfacto train loop on SceneModel (no ns-train).

Replaces the old `ns-train static-gs` subprocess train. Drives the inner SplatfactoModel's
grad render directly: build a nerfstudio Optimizers from the model's param groups (means LR=0,
Invariant #1), step it under fp16 autocast, with the half-res-first-100 schedule (set via
num_downscales=1 / resolution_schedule=100 on the SceneModel) and NoRefineStrategy (densify OFF,
set in SceneModel). Early-stop on the photometric-loss EMA, matching the old behavior.

The built Optimizers IS assigned to model.optimizers so GaussianSet.rebind() survives the later
opacity-purge cull + Phase-0b inserts (which re-point the optimizer param groups).
"""
from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional

import torch

from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.engine.optimizers import AdamOptimizerConfig

# Standard Splatfacto LRs (matches dynamic_gs/dynamic_gs_config.py static-gs block) — means=0.
_STATIC_LRS = {
    "means": 0.0, "features_dc": 0.0025, "features_rest": 0.0025 / 20.0,
    "opacities": 0.05, "scales": 0.005, "quats": 0.001, "camera_opt": 1e-3,
}


def _build_optimizers(scene_model, means_lr: float) -> Optimizers:
    """Construct a nerfstudio Optimizers over the model's param groups at the static LRs (means
    forced to means_lr=0). Assigns it to model.optimizers so surgery rebind() finds it."""
    groups = scene_model.param_groups()                 # {name: [Parameter]}
    cfg = {}
    for name in groups:
        lr = means_lr if name == "means" else _STATIC_LRS.get(name, 0.0)
        cfg[name] = {"optimizer": AdamOptimizerConfig(lr=lr, eps=1e-15), "scheduler": None}
    opt = Optimizers(cfg, groups)
    scene_model.model.optimizers = opt                  # so rebind() (purge/insert) re-points THIS
    scene_model.model._optimizers_wrapper = opt
    return opt


def _shrink_oversized_scales(scene_model, trigger_m: float, reset_m: float) -> None:
    """During-training scale RESET with hysteresis (port of StaticGSModel._shrink_oversized_scales_cb):
    any gaussian whose largest WORLD axis > trigger_m is uniformly shrunk so its largest axis becomes
    reset_m (all 3 log-scales shifted by the SAME amount — shape preserved, NOT a per-axis clamp). Mid-
    training (not the end) so the optimizer re-covers; densification is off so Splatfacto's prune never
    fires. Bounds the sparse far-band splats that otherwise smear the background (measured up to 1.45 m)."""
    import math
    if trigger_m <= 0.0 or reset_m <= 0.0:
        return
    log_trig, log_reset = math.log(trigger_m), math.log(reset_m)
    with torch.no_grad():
        s = scene_model.model.gauss_params["scales"]            # (N,3) log-scale
        log_max = s.max(dim=1, keepdim=True).values
        shift = torch.where(log_max > log_trig, log_max - log_reset, torch.zeros_like(log_max))
        s.sub_(shift)


def train_static(scene_model, gset, cameras: List, batches: List[dict], *,
                 num_steps: int = 500, means_lr: float = 0.0, mixed_precision: bool = True,
                 early_stop_loss: float = 0.02, early_stop_patience: int = 8,
                 early_stop_min_steps: int = 100, scale_clamp_max_m: float = 0.05,
                 scale_reset_value_m: float = 0.01, scale_clamp_every_n: int = 10,
                 tm=None, log_every_s: float = 2.0) -> int:
    """Fit Splatfacto for up to num_steps on the static keyframes. Returns the step it stopped at.

    cameras[i] = a nerfstudio Cameras(1); batches[i] = {"image","depth_image","mask"} (on device).
    Means stay locked by LR=0 (the ONLY means-lock in static — there is no grad hook). Densify is
    off (NoRefineStrategy). fp16 autocast + GradScaler when mixed_precision."""
    dev = scene_model.device
    opt = _build_optimizers(scene_model, means_lr)
    scene_model.enforce_phase_lr()                      # re-assert means LR=0 on the now-built opt
    scaler = torch.amp.GradScaler("cuda", enabled=bool(mixed_precision))
    n = len(cameras)
    assert n > 0 and len(batches) == n, "need matching cameras + batches"

    ema = None
    hits = 0
    t_log = time.time()
    stopped_at = num_steps
    for step in range(num_steps):
        i = step % n                                    # round-robin the keyframes (shuffle-free, deterministic)
        cam, batch = cameras[i], batches[i]
        scene_model.model.step = step                   # drives the resolution schedule
        scene_model.model.train()
        opt.zero_grad_all()
        ctx = torch.autocast("cuda", dtype=torch.float16) if mixed_precision else nullcontext()
        with ctx:
            out = scene_model.model.get_outputs(cam)    # GRAD path (NOT scene_model.render = no-grad)
            loss = sum(scene_model.get_loss_dict(out, batch).values())
        scaler.scale(loss).backward()
        # Use nerfstudio's multi-optimizer scaler step: it skips optimizers with NO grad (means LR=0,
        # camera_opt off) — calling scaler.step() on a grad-less optimizer repeatedly corrupts the
        # scaler's per-optimizer state and silently skips real steps (measured: 17 dB vs 27 dB).
        opt.optimizer_scaler_step_all(scaler)
        scaler.update()
        if scale_clamp_every_n > 0 and (step + 1) % scale_clamp_every_n == 0:
            _shrink_oversized_scales(scene_model, scale_clamp_max_m, scale_reset_value_m)
        lv = float(loss.detach())
        ema = lv if ema is None else 0.9 * ema + 0.1 * lv
        if time.time() - t_log >= log_every_s:
            print(f"[static-train] step {step:4d} loss={lv:.4f} ema={ema:.4f}", flush=True)
            t_log = time.time()
        if early_stop_loss > 0 and step >= early_stop_min_steps:
            hits = hits + 1 if ema < early_stop_loss else 0
            if hits >= early_stop_patience:
                print(f"[static-train] early-stop: ema={ema:.4f} < {early_stop_loss} at step {step}", flush=True)
                stopped_at = step
                break
    if tm is not None:
        tm.gauge("static_train_steps", stopped_at)
    scene_model.model.step = 30000                       # render-time: all SH bands on (Phase-0b renders)
    scene_model.model.eval()
    return stopped_at
