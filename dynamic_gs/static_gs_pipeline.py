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
   model, then writes ``<data_root>/static_scene/post_fusion_state.pt``.
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

    post_fusion_cache_subpath: str = "static_scene/post_fusion_state.pt"
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

        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            grad_scaler=grad_scaler,
        )

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

    # --------------------------------------------------------------
    # Training-time hooks
    # --------------------------------------------------------------

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ):
        """Compose base callbacks with our end-of-training Phase 0b + save."""
        callbacks = super().get_training_callbacks(training_callback_attributes)
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


# ============================================================================
# Trainer (paper-thin subclass — present so the model strip in Phase 2e
# has a landing zone for trainer-side behavior changes)
# ============================================================================


class StaticGSTrainer(NoSaveTrainer):
    """Same as ``NoSaveTrainer`` today. Kept as a separate class so future
    static-only trainer tweaks don't touch the dynamic trainer."""

    def train_iteration(self, step):  # type: ignore[override]
        # NoSaveTrainer's dynamic-phase fast path is irrelevant here
        # (static-gs never enters the dynamic phase), so go straight
        # to the standard Trainer.train_iteration.
        from nerfstudio.engine.trainer import Trainer

        return Trainer.train_iteration(self, step)
