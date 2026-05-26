from __future__ import annotations

import time

import torch

from nerfstudio.engine.trainer import Trainer


class NoSaveTrainer(Trainer):
    """Trainer customised for dynamic-gs.

    Two behaviour deltas vs the stock ``nerfstudio.engine.trainer.Trainer``:

    1. ``save_checkpoint`` is a no-op so rapid iteration / live runs don't
       litter ``outputs/`` with multi-GB ``post_fusion_state.pt`` snapshots.
    2. ``train_iteration`` short-circuits the entire backward + optimizer
       + scheduler + grad-scaler pipeline when the pipeline is in the
       dynamic phase AND ``disable_dynamic_optimization=True``. In that
       regime the pipeline's ``get_train_loss_dict`` already returns a
       zero-loss dummy whose only purpose was to keep the trainer happy;
       the trainer was then burning ~25 ms per step doing
       ``grad_scaler.scale(0).backward()``, optimizer no-op steps, and
       scheduler bumps — pure overhead that drove the live tick-to-tick
       gap up to ~27 ms. With the short-circuit the dummy isn't even
       built: we just fire ``get_train_loss_dict`` (which runs the
       tracker + viewer push as a side effect via ``_tracker_tick_live``)
       and return immediately.
    """

    def save_checkpoint(self, step: int) -> None:
        pass

    def train_iteration(self, step):  # type: ignore[override]
        pipeline = self.pipeline
        live_tracking_only = (
            getattr(pipeline.config, "disable_dynamic_optimization", False)
            and getattr(pipeline, "current_phase", None) == "dynamic"
        )
        if not live_tracking_only:
            return super().train_iteration(step)

        # Tracking-only fast path. The pipeline contract for live mode is:
        #   - ``get_train_loss_dict`` fires ``_tracker_tick_live`` and
        #     ``_force_viewer_rerender`` as side effects.
        #   - It returns ``({}, {"main_loss": zero}, {})`` once done.
        # We don't need any of zero_grad / backward / optimizer step /
        # scheduler / grad-scaler — and skipping them collapses the
        # between-tick gap from ~27 ms to <2 ms.
        t = time.time()
        _, loss_dict, metrics_dict = pipeline.get_train_loss_dict(step=step)
        # ``loss`` is just used by the writer for the "Train Loss" scalar;
        # a CPU tensor avoids a CUDA sync on the writer path.
        loss = torch.zeros((), device="cpu")
        # Record the short-circuit cost so it's visible in
        # timing_report.txt as a sanity check.
        if hasattr(pipeline, "_timing"):
            pipeline._timing["TRAIN.tracking_only_step"].append(time.time() - t)
        return loss, loss_dict, metrics_dict
