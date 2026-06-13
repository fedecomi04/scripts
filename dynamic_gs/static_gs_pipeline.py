"""Minimal pipeline for the ``static-gs`` ns-train method.

What it does, end-to-end:

1. ``__init__``  — builds datamanager + model (reusing ``DynamicGSDataManager``
   and ``DynamicGSModel`` as-is), then runs Phase 0a (SAM3 segmentation +
   Fast-SAM3D 3D generation) and stashes the outputs on the instance. SAM3D
   subprocess pays its multi-minute cost here, before static training starts.

2. Standard Nerfstudio training loop  — ``static_num_steps`` iterations of
   Splatfacto. ``means`` LR is zeroed by ``DynamicGSModel.populate_modules``
   so positions stay on the SfM seed during this phase (consistent with the
   legacy pipeline's static behavior).

3. ``AFTER_TRAIN`` callback  — runs Phase 0b (CPD / TEASER++ registration +
   ``insert_object_gaussians`` + instance-id propagation) on the trained
   model, then writes ``<data_root>/static_scene/static_state.pt``.
   The future ``dynamic-gs`` / ``dynamic-gs-live`` pipelines warm-start
   from that file via ``persistence.load_post_fusion_state``.

What it deliberately does NOT do:

* No dynamic phase. No XFeat tracker, no ESAM, no feedforward decoder, no
  scene-opt-during-dynamic, no rigid transform application.
* No convergence-check early-exit yet — train for the full
  ``static_num_steps`` budget. (Hookable later via
  ``change_detection.compute_change_mask`` against the GT render.)
* No FoundationPose construction. Phase 0b builds no FP trackers. The
  manifest carries only registration stats (kept points, cull stats, etc.).

The companion ``StaticGSTrainer`` is a paper-thin subclass of
``NoSaveTrainer`` — only present so the eventual model strip in Phase 2e
has a place to land per-trainer behavior changes without touching the
dynamic trainer.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type

import torch

from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils.rich_utils import CONSOLE

from .dynamic_gs_datamanager import DynamicGSDataManagerConfig
from .dynamic_gs_trainer import NoSaveTrainer
from .fusion import run_phase0a_sam3_and_sam3d, run_phase0b_fusion
from .persistence import save_post_fusion_state
from .static_gs_model import StaticGSModelConfig


# ============================================================================
# Pipeline config + class
# ============================================================================


@dataclass
class StaticGSPipelineConfig(VanillaPipelineConfig):
    """Config for ``static-gs``. Intentionally minimal — no dynamic / live /
    feedforward / tracker / optim-pool / keyframe-filter fields. Those will
    live on the future ``DynamicGSPipelineConfig`` rewrite (Phase 3)."""

    _target: Type = field(default_factory=lambda: StaticGSPipeline)
    datamanager: DynamicGSDataManagerConfig = field(
        default_factory=DynamicGSDataManagerConfig
    )
    model: StaticGSModelConfig = field(default_factory=StaticGSModelConfig)

    static_num_steps: int = 1000
    """Number of Splatfacto training iterations before Phase 0b runs.
    Matches the legacy ``DynamicGSPipelineConfig.static_num_steps``."""

    post_fusion_cache_subpath: str = "static_scene/static_state.pt"
    """Where to write the post-fusion model snapshot, relative to
    ``datamanager.data``. The future dynamic-gs pipeline warm-starts from
    this path via ``persistence.load_post_fusion_state``."""


class StaticGSPipeline(VanillaPipeline):
    """End-to-end static-only pipeline. See module docstring for flow."""

    config: StaticGSPipelineConfig

    def __init__(
        self,
        config: StaticGSPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler=None,
    ):
        self._timing: defaultdict[str, list] = defaultdict(list)
        self._sam3d_generation_outputs: Optional[dict] = None
        self._phase0b_done: bool = False
        self._timing_report_written: bool = False
        self._train_start_t: Optional[float] = None

        # Fresh timing ledger for this run (recorded flow starts here; the live
        # flow's capture stage owns its own ledger render before this resets).
        try:
            from .utils import timing_ledger as _tl
            _tl.reset(config.datamanager.data)
        except Exception:
            pass

        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            grad_scaler=grad_scaler,
        )

        import atexit as _atexit
        _atexit.register(self._write_timing_report)

        # Phase 0a: SAM3 segmentation + Fast-SAM3D 3D generation.
        # Runs once at construction, BEFORE the trainer starts the loop.
        # The SAM3D subprocess moves the model to CPU around its run so
        # small GPUs survive the load; everything is restored on the way out.
        try:
            self._sam3d_generation_outputs = run_phase0a_sam3_and_sam3d(
                model=self.model,
                datamanager=self.datamanager,
                timing=self._timing,
            )
        except Exception as exc:
            CONSOLE.log(
                f"[static-gs] Phase 0a raised; continuing without prefusion: {exc}"
            )
            self._sam3d_generation_outputs = None

        # Eager AnySplat pre-spawn (DGS_EAGER_ANYSPLAT=1, set by bootstrap_live.sh).
        # Phase 0a is done (SAM3D unloaded), so the GPU has room: AnySplat (~3.5 GB)
        # + splatfacto training (~2.5 GB) + Gazebo (~2.6 GB) = ~8.6 GB < 16. Its
        # ~17 s load fully overlaps the static training loop, and the subsequent
        # dynamic-gs-live run ADOPTS the warm worker (no teleop-start load stall).
        # Spawns a detached FIFO worker at <data>/.anysplat_worker.
        if os.environ.get("DGS_EAGER_ANYSPLAT") == "1":
            try:
                from .utils.anysplat_decode import spawn_detached_anysplat_worker
                fifo_dir = Path(self.config.datamanager.data) / ".anysplat_worker"
                _as_pid = spawn_detached_anysplat_worker(fifo_dir)
                CONSOLE.log(
                    f"[static-gs] eager AnySplat worker spawned (pid={_as_pid}) → {fifo_dir} "
                    f"(loads during training; dynamic-gs-live will adopt it)"
                )
            except Exception as exc:
                CONSOLE.log(f"[static-gs] eager AnySplat spawn failed (non-fatal): {exc}")

    # --------------------------------------------------------------
    # Training-time hooks
    # --------------------------------------------------------------

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ):
        """Compose base callbacks with our end-of-training Phase 0b + save."""
        callbacks = super().get_training_callbacks(training_callback_attributes)

        def _stamp_train_start(step: int) -> None:
            if self._train_start_t is None:
                import time as _time
                self._train_start_t = _time.time()

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=_stamp_train_start,
            )
        )
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN],
                update_every_num_iters=1,
                func=self._finalize_static_training,
            )
        )
        return callbacks

    def _finalize_static_training(self, step: int) -> None:
        """End-of-training: Phase 0b fusion + post-fusion cache save."""
        if self._phase0b_done:
            return
        self._phase0b_done = True

        # Record the Splatfacto training loop wall (pure training; phase0b runs
        # after this point) into the ledger.
        try:
            import time as _time
            from .utils import timing_ledger as _tl
            if self._train_start_t is not None:
                _tl.record(self.config.datamanager.data, "static_training",
                           "Splatfacto", "train", self._train_start_t, _time.time())
        except Exception:
            pass

        if not self._sam3d_generation_outputs:
            CONSOLE.log(
                "[static-gs] Phase 0a produced no objects; skipping Phase 0b + cache save"
            )
            return

        CONSOLE.log(
            "[static-gs] training complete — running Phase 0b (fusion) on the trained scene"
        )
        try:
            manifest = run_phase0b_fusion(
                model=self.model,
                datamanager=self.datamanager,
                generation_outputs=self._sam3d_generation_outputs,
                device=self.device,
                timing=self._timing,
            )
        except Exception as exc:
            CONSOLE.log(f"[static-gs] Phase 0b raised; skipping cache save: {exc}")
            return

        if not manifest:
            CONSOLE.log(
                "[static-gs] Phase 0b returned an empty manifest; skipping cache save"
            )
            return

        cache_path = (
            Path(self.config.datamanager.data) / self.config.post_fusion_cache_subpath
        )
        ok = save_post_fusion_state(self.model, cache_path)
        if ok:
            obj_count = int(self.model.object_flags.sum().item())
            inst_count = int((self.model.object_instance_ids > 0).any(dim=-1).sum().item())
            CONSOLE.log(
                f"[static-gs] post-fusion cache written → {cache_path} "
                f"(N={int(self.model.num_points)}, object_flags={obj_count}, "
                f"instance_id>0={inst_count})"
            )
        else:
            CONSOLE.log("[static-gs] post-fusion cache save failed; see prior error")

    # --------------------------------------------------------------
    # Timing report (atexit; mirrors dynamic_gs_pipeline_base._write_timing_report)
    # --------------------------------------------------------------

    def _write_timing_report(self) -> None:
        """Write ``<data_root>/timing_report_static.txt`` from ``self._timing``.

        Idempotent — atexit may fire twice on Ctrl+C. Distinct filename
        from the dynamic-gs report so the two don't overwrite each other
        when both are run on the same dataset."""
        if self._timing_report_written:
            return
        try:
            datamanager = object.__getattribute__(self, "datamanager")
        except AttributeError:
            return
        if datamanager is None or not hasattr(datamanager, "config"):
            return
        timing = self._timing
        if not timing:
            return
        self._timing_report_written = True

        from datetime import datetime

        def _row(key: str, vals) -> str:
            if not vals:
                return f"  {key:<42s}        N/A"
            n = len(vals)
            avg_ms = float(sum(vals)) / n * 1000.0
            total = float(sum(vals))
            mn = min(vals) * 1000.0
            mx = max(vals) * 1000.0
            return (
                f"  {key:<42s} n={n:<6d} avg={avg_ms:>8.1f}ms "
                f"min={mn:>7.1f}ms max={mx:>8.1f}ms total={total:>7.1f}s"
            )

        lines: list[str] = []
        lines.append("=" * 96)
        lines.append("STATIC-GS TIMING REPORT")
        lines.append("=" * 96)
        lines.append(f"Generated:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Data root:   {datamanager.config.data}")
        lines.append(f"Num points:  {getattr(self.model, 'num_points', '?')}")
        lines.append("")

        # Bulleted by-phase load/inference report from the unified timing ledger.
        try:
            from .utils import timing_ledger as _tl
            # If an eager AnySplat worker reported its load, fold it in (loads
            # during training → should show as overlap, not a stall).
            try:
                import json as _json
                _aw = Path(datamanager.config.data) / ".anysplat_worker"
                _ls = 0.0
                for _f in (_aw / "ready.json", _aw / "spawn.json"):
                    if _f.is_file():
                        _d = _json.loads(_f.read_text().splitlines()[0])
                        _ls = max(_ls, float(_d.get("load_seconds", 0.0)))
                if _ls > 0 and getattr(self, "_train_start_t", None):
                    _tl.record(datamanager.config.data, "static_training", "AnySplat",
                               "load", self._train_start_t, self._train_start_t + _ls)
            except Exception:
                pass
            lines.append(_tl.render(datamanager.config.data))
            lines.append("")
        except Exception as _exc:
            lines.append(f"(timing-ledger render failed: {_exc})")
            lines.append("")

        lines.append("DETAIL (per-substep, ms):")
        # Group by prefix (S0. / static_step / etc.) for readability.
        groups: dict[str, list[str]] = {}
        for k in sorted(timing.keys()):
            prefix = k.split(".", 1)[0] if "." in k else k.split("_", 1)[0]
            groups.setdefault(prefix, []).append(k)
        for group_name in sorted(groups.keys()):
            group_keys = groups[group_name]
            lines.append(f"--- {group_name} ({len(group_keys)} key{'s' if len(group_keys)!=1 else ''}) ---")
            for k in group_keys:
                lines.append(_row(k, timing[k]))
            lines.append("")

        out_path = Path(datamanager.config.data) / "timing_report_static.txt"
        try:
            out_path.write_text("\n".join(lines) + "\n")
            CONSOLE.log(f"[static-gs] timing report written → {out_path}")
        except Exception as exc:
            CONSOLE.log(f"[static-gs] failed to write timing report: {exc}")


# ============================================================================
# Trainer (paper-thin subclass — present so the model strip in Phase 2e
# has a landing zone for trainer-side behavior changes)
# ============================================================================


# Early-stop knobs (env-overridable for tuning). The static phase optimizes
# Splatfacto's photometric loss ``main_loss = (1-ssim_lambda)*L1 +
# ssim_lambda*(1-SSIM)`` (see splatfacto.get_loss_dict). Means are frozen and
# densification is off, so the loss is monotone-ish and plateaus quickly; once
# it's below ``STATIC_EARLY_STOP_LOSS`` for ``STATIC_EARLY_STOP_PATIENCE``
# consecutive steps (after a ``STATIC_EARLY_STOP_MIN_STEPS`` warmup so the
# first noisy steps can't trip it), there's nothing left to gain — stop and go
# straight to Phase 0b. Set the loss to 0 (or DGS_STATIC_EARLY_STOP=0) to
# disable and always run the full max_num_iterations budget.
STATIC_EARLY_STOP_ENABLED = os.environ.get("DGS_STATIC_EARLY_STOP", "1") != "0"
# 0.09 fired at ~step 107 on the real 1920x1200 scene, but render PSNR keeps
# climbing well past that (measured ladder, same 459k seed: step107=24.0 dB,
# step500=26.3, step1000=27.3) — the loss-EMA flattens long before the render
# is sharp at high resolution, so 0.09 was undertraining = the "scene is
# blurry" report. 0.02 sits below the EMA reached by ~step 500, so the scene
# trains essentially the full STATIC_NUM_STEPS=500 budget (the +2.3 dB jump)
# while still early-exiting a genuinely-trivial scene that converges harder.
STATIC_EARLY_STOP_LOSS = float(os.environ.get("DGS_STATIC_EARLY_STOP_LOSS", "0.02"))
STATIC_EARLY_STOP_PATIENCE = int(os.environ.get("DGS_STATIC_EARLY_STOP_PATIENCE", "8"))
STATIC_EARLY_STOP_MIN_STEPS = int(os.environ.get("DGS_STATIC_EARLY_STOP_MIN_STEPS", "100"))


class StaticGSTrainer(NoSaveTrainer):
    """Same as ``NoSaveTrainer`` today, plus photometric-loss early-stop so the
    static fit exits as soon as ``main_loss`` plateaus below threshold instead
    of always burning the full step budget. Kept separate so static-only
    trainer tweaks don't touch the dynamic trainer."""

    _early_stop_hits: int = 0

    def train_iteration(self, step):  # type: ignore[override]
        # NoSaveTrainer's dynamic-phase fast path is irrelevant here
        # (static-gs never enters the dynamic phase), so go straight
        # to the standard Trainer.train_iteration.
        from nerfstudio.engine.trainer import Trainer

        loss, loss_dict, metrics_dict = Trainer.train_iteration(self, step)

        main = loss_dict.get("main_loss")
        main_val = float(main.detach()) if main is not None else None
        if main_val is not None:
            import time as _t
            now = _t.time()
            if getattr(self, "_loss_t0", None) is None:
                self._loss_t0 = now
                self._loss_last = -1.0
                self._loss_ema = main_val
            self._loss_ema = 0.9 * self._loss_ema + 0.1 * main_val
            el = now - self._loss_t0
            # log once per wall-second so the curve is "time to convergence"
            if el - self._loss_last >= 1.0:
                mcfg = self.pipeline.model.config
                nd = getattr(mcfg, "num_downscales", 2)
                rs = getattr(mcfg, "resolution_schedule", 100)
                ds = 2 ** max(nd - step // max(rs, 1), 0)
                CONSOLE.log(f"static-gs| t={el:4.1f}s step={step:4d} res=1/{ds} "
                            f"loss={main_val:.4f} ema={self._loss_ema:.4f}")
                self._loss_last = el

        if STATIC_EARLY_STOP_ENABLED and STATIC_EARLY_STOP_LOSS > 0:
            # Fire on the EMA (not the per-image loss, which oscillates ~0.05-0.15
            # and never gives a stable 10-in-a-row near the plateau). The full-res
            # main_loss EMA plateaus at ~0.08 (the 500-step "perfect" value); the
            # threshold is that plateau's lower bound. Requires full-res early
            # (num_downscales=0) or the EMA stays ~0.10 until the schedule reaches
            # full res. See DGS_STATIC_EARLY_STOP_LOSS.
            if main_val is not None and step >= STATIC_EARLY_STOP_MIN_STEPS:
                if self._loss_ema < STATIC_EARLY_STOP_LOSS:
                    self._early_stop_hits += 1
                else:
                    self._early_stop_hits = 0
                if self._early_stop_hits >= STATIC_EARLY_STOP_PATIENCE:
                    CONSOLE.log(
                        f"static-gs| early-stop: loss_ema={self._loss_ema:.4f} < "
                        f"{STATIC_EARLY_STOP_LOSS} for {STATIC_EARLY_STOP_PATIENCE} "
                        f"steps at step {step}; ending static training. "
                        f"(disable via DGS_STATIC_EARLY_STOP=0)"
                    )
                    # The Trainer.train() loop checks self.stop_training at the
                    # top of each step and breaks; the AFTER_TRAIN callback
                    # (_finalize_static_training → Phase 0b) still fires.
                    self.stop_training = True

        return loss, loss_dict, metrics_dict
