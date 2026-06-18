# Code Audit — `dynamic_gs/dynamic_gs_pipeline_recorded.py`

Adversarial audit. File is the recorded-mode subclass of `DynamicGSPipelineBase`. Registered as the `dynamic-gs` ns-train method via `pyproject.toml` → `dynamic_gs.dynamic_gs_config:DynamicGS` → `RecordedDynamicGSPipelineConfig`.

LIVE-PATH NOTE: this module is the RECORDED pipeline. The actual live ROS path is `dynamic_gs_pipeline_live.py` (a sibling subclass). The recorded `_tracker_tick` is single-threaded on the trainer thread; the only background concurrency comes from the shared base's `_dispatch_feedforward_async` → `_feedforward_threaded` and the viser-direct render daemon. Live-vs-recorded thread-safety findings below are scoped to what THIS file touches.

---

## 1) FUNCTION / CLASS MAP

### `RecordedDynamicGSPipelineConfig` (dataclass) — recorded.py:39
Config for the recorded pipeline. Thin extension of `DynamicGSPipelineBaseConfig`; adds D0 anchor ratios + keep-viser-alive knobs.
- **Callers:** referenced at recorded.py:90, recorded.py:94 (type annotations) and instantiated at `dynamic_gs_config.py:174` inside `DynamicGS = MethodSpecification(...)`. ENTRY POINT (nerfstudio method_configs). Not dead.

Fields:
- `_target` (recorded.py:45) — `default_factory=lambda: RecordedDynamicGSPipeline`. Consumed by nerfstudio's `InstantiateConfig.setup`. Entry-point machinery.
- `d0_anchor_x_ratio: float = 0.5` (recorded.py:47) — read at recorded.py:123 (log), recorded.py:290 (`_pick_d0_object`). Read in-module only; no external refs.
- `d0_anchor_y_ratio: float = 0.75` (recorded.py:51) — read at recorded.py:123, recorded.py:291. In-module only.
- `keep_viser_alive_at_end: bool = True` (recorded.py:55) — read at recorded.py:146. In-module only.
- `keep_viser_alive_timeout_s: float = 1800.0` (recorded.py:61) — read at recorded.py:156. In-module only.

### `RecordedDynamicGSPipeline` (class) — recorded.py:71
Recorded-mode dynamic pipeline; subclass of `DynamicGSPipelineBase`.
- **Callers:** `_target` factory at recorded.py:45; referenced in docstrings of `dynamic_gs_pipeline_live.py` and `_base.py`. Instantiated by nerfstudio via the config `_target`. ENTRY POINT. Not dead.

#### `__init__(self, config, device, test_mode="val", world_size=1, local_rank=0, grad_scaler=None)` — recorded.py:92
Sets `_next_frame_to_track=0`, calls `super().__init__`, flips datamanager + model to `"dynamic"` phase, builds `_accepted_dynamic_frames`, logs readiness, sets `_keep_alive_done=False`.
- **Callers:** invoked by nerfstudio config instantiation. ENTRY POINT.

#### `block_until_viser_shutdown(self) -> None` — recorded.py:135
End-of-run main-thread hook: writes the timing report, then (if keep-alive on) blocks viser-direct serving until operator shutdown click or timeout.
- **Callers:** `dynamic_gs_trainer.py:47` (`hook = getattr(pipeline, "block_until_viser_shutdown", None)` then `hook()`). 1 caller. Not dead — invoked by the trainer's `NoSaveTrainer._train_complete_viewer`.

#### `_tracker_tick(self, step: int) -> None` — recorded.py:174
The per-step core. Advances one dataset frame (paced by `dynamic_steps_per_frame`), filters depth, optional interactive picker, D0 bootstrap or motion-estimate, decides FF-due, publishes `_latest_tracker_frame`, pushes viser, fires `_on_tracker_frame`, advances frame idx.
- **Callers:** abstract hook declared at `_base.py:1100`; called by the base scheduler at `_base.py:986` (`self._tracker_tick(step)`). 1 effective caller. Not dead (override of abstract).

#### `_pick_d0_object(self, camera, prefused_instance_ids) -> int` — recorded.py:261
2D anchor pick: projects each prefused instance centroid into the camera, returns the instance id whose centroid is closest to `(d0_anchor_x_ratio*W, d0_anchor_y_ratio*H)`. Honors `d0_force_instance_id` override.
- **Callers:** recorded.py:371 (`_bootstrap_d0`). Also overrides the abstract at `_base.py:1108`; base calls it at `_base.py:1210` (inside `_resolve_d0_instance` path). The live subclass has its own override (live.py:346). Not dead.

#### `_on_tracker_frame(self, camera, batch, cdn, is_first) -> None` — recorded.py:324
Post-tick callback. On first frame stamps the from-scratch timing section; then if `_ff_due_this_tick` dispatches the recurring feedforward async.
- **Callers:** recorded.py:256 (inside `_tracker_tick`). Overrides abstract at `_base.py:1118`. Not dead.

#### `_bootstrap_d0(self, camera, batch) -> None` (`@torch.no_grad()`) — recorded.py:356
First-tick bootstrap: picks moved-object instance, then calls shared `_reseed_tracked_object` (object_flags + reference pose + object mask + XFeat anchor seed). Sets `_d0_selected_instance_id=0` and disables tracking when nothing is picked.
- **Callers:** recorded.py:215 (inside `_tracker_tick`). Live subclass has its own (live.py:480). Not dead.

---

## 2) DEAD-CODE CANDIDATES

### `self._accepted_dynamic_frames` — recorded.py:120 — **HIGH confidence, write-only / dead**
Set to `list(range(n_dyn))` in recorded `__init__`. Grep across `dynamic_gs/` + `scripts/` shows the attribute is **only ever assigned**, never read:
```
dynamic_gs_pipeline_recorded.py:120:  self._accepted_dynamic_frames = list(range(n_dyn))   # write
dynamic_gs_pipeline_live.py:132:      self._accepted_dynamic_frames = []                   # write
dynamic_gs_pipeline_base.py:458:      self._accepted_dynamic_frames: list[int] = []        # write (init)
```
No indexing (`_accepted_dynamic_frames[`), no membership/iteration anywhere. Zero reads in the entire repo. The whole computation `list(range(n_dyn))` plus the `get_num_dynamic_frames` call feeding it (only otherwise needed for the log line, which already calls `get_num_dynamic_frames` separately at recorded.py:176) is dead. NOTE: this is a base-class attribute also written in live; removing it requires touching all three sites. Not invariant-protected.

### Everything else: NOT dead
- All five `*Config` fields are read in-module (grep evidence in §1). `d0_anchor_x_ratio`/`_y_ratio`/`keep_viser_alive_*` having no *external* refs is expected — they configure this module only.
- `block_until_viser_shutdown` — 1 real caller (trainer). Keep.
- All abstract-hook overrides (`_tracker_tick`, `_pick_d0_object`, `_on_tracker_frame`, `_bootstrap_d0`) are dispatched polymorphically by the base; not dead.
- `RecordedDynamicGSPipeline` / `RecordedDynamicGSPipelineConfig` — entry points via `pyproject.toml` method_configs + `dynamic_gs_config.py:174`. Excluded by rule.

---

## 3) DATA-LIFECYCLE

### `.pt` warm-cache (`post_fusion_state.pt`)
Not loaded/saved in THIS file. The recorded `__init__` only calls `super().__init__` (base does the warm load) and then flips phase. No direct `persistence/` touch here. The `set_phase("dynamic")` on model+datamanager (recorded.py:113–116) mutates `means.requires_grad` and the active frame source — this is correct per the comment (without the model switch, the static branch would set `means.requires_grad=False` and break `register_hook` on the next `insert_inpaint_gaussians`). No leak.

### The 4 identity buffers (invariant-protected)
- `object_flags` — written via `_reseed_tracked_object` (called from `_bootstrap_d0`, recorded.py:383). This is the documented D0-selection writer (Invariant #8). Correct.
- `object_instance_ids` — only READ here: `instance_ids_buf = self.model.object_instance_ids` (recorded.py:370), `self.model.means[mask]` keyed off ids (recorded.py:305–306). Read-only consumption; Phase-0b owns writes. Correct (invariant-protected).
- `sam3d_init_target_flags` — not touched here. Invariant-protected placeholder.
- `inserted_flags` — not touched here (written by FF Mode B inside the base/`rgbd_decode`). Correct.
No desync introduced by this file.

### Per-tick batch / depth tensor (`batch["depth_image"]`) — recorded.py:191–196
`get_current_dynamic_train_batch()` returns `(camera, batch)`. Depth is filtered IN PLACE (`batch["depth_image"] = _depth_filter.filter_depth_torch(...)`) once at the source so both tracker and FF consume cleaned depth (parity with live; correct, matches CLAUDE.md depth-filter note). The whole `batch` (rgb + depth + mask, GPU tensors per `cache_images="cpu"` config) is then stored on `self._latest_tracker_frame` (recorded.py:237–243). **Lifecycle concern (LOW):** `_latest_tracker_frame` holds the entire previous batch (camera + cdn + batch dict) until the next tick overwrites it. At 1920×1200 that is one full rgb+depth+mask frame pinned on the heap/GPU at all times. Single-frame retention is bounded (not a growing leak) — the dict is reassigned, not appended — but the FF bg thread reads `self._latest_tracker_frame` asynchronously (see §4), so the new assignment can swap the object the bg thread is mid-read on.

### Timing dict (`self._timing`) — recorded.py:253
`self._timing["DN.4_viser_push"].append(...)`. `_timing` is a base `defaultdict(list)` (_base.py:380); grows unboundedly across the run (one float per tick per key). Expected/by-design for a bounded recorded run; for a very long replay this is a slow monotonic memory growth (LOW). `block_until_viser_shutdown` flushes it to `timing_report.txt` via `_write_timing_report` (guarded by `_timing_report_written`, idempotent — confirmed at _base.py:803/815). No double-write.

### viser-direct server handle — recorded.py:151–166
`srv = getattr(self, "_viser_direct_server", None)`; guarded by `srv.is_closing`. `keep_alive_until_shutdown` + `wait_for_shutdown(timeout)` exist (viser_direct.py:311/334). No explicit free here — teardown is the base/atexit's job. The comment (recorded.py:126–132) documents that this MUST run on the main thread pre-teardown because daemon threads stall during interpreter finalization. Correct rationale; no leak introduced.

### SHM
Not touched in this file (recorded mode reads frames from disk via the datamanager, not SHM). SHM lifecycle lives in `live_shm_reader.py` / the live subclass.

---

## 4) DESIGN SMELLS

### Thread-safety: `_latest_tracker_frame` swap races the FF bg thread — MEDIUM
recorded.py:237 reassigns `self._latest_tracker_frame = {...}` on the trainer thread every tick, while `_on_tracker_frame` (recorded.py:349) hands `self._latest_tracker_frame` to `_dispatch_feedforward_async` which runs `_feedforward_threaded` on a BACKGROUND thread (_base.py:2519). The dispatch passes the dict by reference (`self._latest_tracker_frame`), and the next tick rebinds the attribute to a NEW dict — so the in-flight FF thread keeps a reference to the snapshot it was dispatched with (that specific dict object is not mutated in place after creation, depth is filtered before insertion), which is SAFE for the dict identity. BUT the GPU tensors inside it (`batch`, `camera`) are shared with the model/datamanager and could be reused/overwritten by the next `get_current_dynamic_train_batch`. The decode-once-per-N-ticks gate + "no-op if a prior FF is still in flight" (recorded.py:348 comment) mitigates overlap but does not formally guarantee the bg thread's tensors aren't aliased by a later tick's batch. Worth confirming the datamanager returns fresh tensors per call rather than a reused buffer (out-of-scope: lives in `dynamic_gs_datamanager.py`).

### `cdn` is hard-wired `None` → confusing dead-looking plumbing — MEDIUM
recorded.py:234 sets `cdn = None` unconditionally (the CDN render moved to the FF bg thread). It is then:
- stored in `_latest_tracker_frame["cdn"]` (recorded.py:239),
- passed to `_on_tracker_frame(camera, batch, cdn, is_first)` (recorded.py:256),
- whose signature still types it `Optional[torch.Tensor]` and whose docstring talks about CDN (recorded.py:324–337).
So `_on_tracker_frame`'s `cdn` parameter is **always None in recorded mode** — a threaded-through dead parameter. The `_ff_due_this_tick` flag (recorded.py:230) is the real signal `_on_tracker_frame` uses. The `cdn`-handling comments (recorded.py:219–234) describe machinery that no longer renders here. Leaky/misleading: a reader expects `cdn` to sometimes be a tensor. Recommend documenting "always None in recorded; bg thread renders it" at the param, or dropping the param.

### `_pick_d0_object` is a mild god-method / mixes concerns — LOW
recorded.py:261–322: does force-id override + logging + manual camera-intrinsics scalarization (`_scalar` nested helper) + 3D→2D projection + per-instance loop + verbose per-candidate logging. The hand-rolled OpenGL projection (recorded.py:307–312, note the `-centroid_cam[2]` depth and `-centroid_cam[1]` v-flip) is a leaky abstraction duplicating camera math that nerfstudio `Cameras` could provide. The live subclass (live.py:346) has a parallel `_pick_d0_object` (3D variant) — projection/anchor logic is forked between the two, a duplication risk if conventions change.

### `_scalar` nested helper — LOW
recorded.py:286 defines `_scalar` inside `_pick_d0_object` to unwrap possibly-batched camera scalar tensors. This pattern (camera fields may be `(1,)` tensors or scalars) recurs; defining it locally per-call is minor churn but harmless.

### Swallowed exception in `block_until_viser_shutdown` — LOW
recorded.py:167 `except Exception as exc:` logs and returns — acceptable for a teardown-time best-effort path (don't crash the exit on viser hiccup). The mirror in the trainer (`dynamic_gs_trainer.py` `except Exception: pass`) is fully silent. Acceptable but worth noting the double broad-catch.

### Config docstring admits hard-coding instead of config — LOW
recorded.py:41–43: "all recorded-specific behavior is hard-coded rather than configurable." Honest, not a bug, but flags that the recorded-specific pacing/FF cadence relies on base fields rather than recorded-scoped ones.

### Dead branch unreachability
No live-mode-only branches here (this IS the recorded file). The `sel_status == "seeded"` branch (recorded.py:212) is only reachable when `interactive_object_selection=True` (default False) — reachable but off by default. Not dead, just default-disabled.
