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

    def _train_complete_viewer(self) -> None:  # type: ignore[override]
        """Called by ``Trainer.train()`` after the loop exits (when
        ``viewer.quit_on_train_completion`` is False, our default). Runs on the
        MAIN thread, BEFORE any teardown/atexit, with the viser-direct render
        daemon thread + websocket server still fully alive — so the scene stays
        interactive (mouse orbit re-renders via on_update→request_render).

        If the pipeline opts into end-of-run keep-alive, block here until the
        operator clicks the in-viewer 'Shutdown viewer' button (or the
        timeout). This replaces the old atexit-based block, which ran during
        interpreter shutdown and left the scene frozen (daemon threads stall
        once finalization starts)."""
        pipeline = self.pipeline
        hook = getattr(pipeline, "block_until_viser_shutdown", None)
        if callable(hook):
            try:
                hook()
                return
            except Exception:
                pass
        # Fall back to the stock NS-viewer keep-alive only if we didn't handle it.
        try:
            super()._train_complete_viewer()
        except Exception:
            pass

    def train_iteration(self, step):  # type: ignore[override]
        pipeline = self.pipeline
        # The dynamic phase is ALWAYS a tracker-only runtime now (invariant #4:
        # every gauss-param LR is 0 and get_train_loss_dict returns a zero loss),
        # so the full NS train step — zero_grad / backward / optimizer / scheduler
        # / AFTER_TRAIN_ITERATION callbacks — is pure waste (~25 ms/tick measured
        # as GAP.trainer_outer_loop). Skip it for the whole dynamic phase. (This
        # gate used to also require disable_dynamic_optimization, which was purged
        # with the scene-opt machinery — leaving the fast path dead until now.)
        live_tracking_only = (getattr(pipeline, "current_phase", None) == "dynamic")
        if not live_tracking_only:
            return super().train_iteration(step)

        # GAP.trainer_outer_loop = wall-clock spent between the PREVIOUS
        # train_iteration return and this entry. This is everything the
        # Nerfstudio outer loop does between iterations: AFTER_TRAIN_ITERATION
        # callbacks (splatfacto step_post_backward), writer scalars, viewer
        # state update, scheduler bumps for non-tracking-only paths. With
        # the rest of between_tick_gap accounted for by pipeline overhead
        # (GAP.pipeline_overhead, computed in the report), this isolates
        # the trainer's cost from the pipeline's cost.
        iter_entry_t = time.time()
        if hasattr(pipeline, "_timing") and hasattr(pipeline, "_last_iter_exit_t"):
            pipeline._timing["GAP.trainer_outer_loop"].append(iter_entry_t - pipeline._last_iter_exit_t)
        # Stash entry time so _tracker_tick_live can compute the
        # pipeline-side prelude (sync_phase, dispatch, etc.) between
        # train_iteration entry and the actual tracker tick.
        pipeline._last_iter_entry_t = iter_entry_t

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
            # GAP.pipeline_postlude = time spent in get_train_loss_dict
            # AFTER _tracker_tick_live finished. This is the pipeline-side
            # tail (timing-summary checks, return of zero loss).
            if hasattr(pipeline, "_last_tick_end_t"):
                pipeline._timing["GAP.pipeline_postlude"].append(time.time() - pipeline._last_tick_end_t)
            pipeline._last_iter_exit_t = time.time()
        return loss, loss_dict, metrics_dict
