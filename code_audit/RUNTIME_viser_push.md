# RUNTIME trace — viser-direct visualization push path (LIVE dynamic-gs)

Scope: the visualization half of the live pipeline. How `ViserDirectScene`
(`dynamic_gs/utils/viser_direct.py`) is driven from `DynamicGSPipelineBase`
(`dynamic_gs/dynamic_gs_pipeline_base.py`) and `LiveDynamicGSPipeline`
(`dynamic_gs/dynamic_gs_pipeline_live.py`): which threads render, what shared
state they touch, what lock (if any) they hold, the `is_closing` guards, and
the teardown race.

Verified by reading source; line numbers are from the files as they exist now.
Per CLAUDE.md invariant #9 this path is canonical (NS viewer is OFF;
`vis="tensorboard"`). Invariant #4 (all dynamic LRs = 0) means the render thread
only ever *reads* gauss params; the writers are the tracker rigid transform and
the FF bg insert/cull, never an optimizer step.

---

## (a) Annotated call chain

### Threads in play

| Thread | Created at | Role |
|---|---|---|
| **Trainer/tracker thread** (main ns-train loop) | nerfstudio `Trainer.train` → `_tracker_tick` | per-tick: peek SHM, XFeat motion, `apply_rigid_object_transform_from_reference` (model write), then the viser push calls |
| **viser-direct-render thread** (daemon) | `ViserDirectScene.attach_model` → `threading.Thread(target=self._render_loop)` — `viser_direct.py:437` | the only renderer for the browser: per request, `get_outputs` per client, push image |
| **FF bg thread** (daemon, `ff-<mode>`) | `_dispatch_feedforward_async` → `threading.Thread(target=self._feedforward_threaded)` — `dynamic_gs_pipeline_base.py:2513` | CDN render + AnySplat decode + cull + insert (model write); calls `request_render` after each insert |
| **viser internal thread pool** | inside `viser.ViserServer` | runs `on_client_connect` / `on_client_disconnect` / `camera.on_update` / GUI button callbacks |

### Spin-up

1. `DynamicGSPipelineBase.__init__` builds `self._model_lock = threading.RLock()`
   — `dynamic_gs_pipeline_base.py:474`.
2. After `super().__init__()` builds `self.model`, `model.attach_render_lock(self._viser_lock_ctx)`
   — `:519-520` (wires the SAME lock into the NS-viewer render path too, defensive).
3. Warm cache loaded — `:526`.
4. `if config.enable_viser_direct: self._setup_viser_direct()` — `:529-530`.
5. `_setup_viser_direct` — `:1452`:
   - `ViserDirectScene(port=...)` ctor — `viser_direct.py:201` starts the viser server,
     builds GUI (`_build_gui` `:271`), wires `on_client_connect`/`on_client_disconnect`
     (`:274-280`).
   - **Lock swap**: `self._viser_direct_server.model_lock = self._model_lock` — `:1480`.
     The server's own `RLock` (created at `viser_direct.py:208`) is discarded; the
     render thread now shares the pipeline's RLock. This is load-bearing — without it
     the render thread and FF bg thread would lock on two different objects and tear
     gauss_params.
   - `attach_model(self.model, device=self.device)` — `:1481` → `viser_direct.py:421`
     stores `_model`, `_device`, and **starts the render thread** (`:435-442`).

### Per-tick push (tracker thread, live)

`LiveDynamicGSPipeline._tracker_tick` — `dynamic_gs_pipeline_live.py:243`:

- peek SHM, build `camera` + `batch` — `:271-272`.
- `_invalidate_object_mask_cache()` — `:275`.
- D0 bootstrap or `_apply_motion_estimator(camera, batch)` — `:303`.
  - `_apply_motion_estimator` (base `:2162`) acquires `with self._viser_lock_ctx():`
    and calls `model.apply_rigid_object_transform_from_reference(...)` — `:2163-2166`
    (**model write under lock**), then unconditionally `srv.request_render()` — `:2194-2196`.
- `_build_viser_direct_handles(camera)` — `:333` → base `:1496`; legacy stub path,
  sets initial camera once.
- `_push_viser_direct_transforms()` — `:334` → base `:1521`; calls
  `srv.push_tracker_transform(...)` which is a **no-op stub** (`viser_direct.py:506`).
- `_push_viser_camera_feed(camera, batch)` — `:335` → base `:1537`:
  - reads `batch["image"]`, converts to uint8 HWC, `srv.update_camera_feed(rgb_np)`
    (`viser_direct.py:342`: ref-swap under `_feed_lock`).
  - reads `camera.camera_to_worlds`, `srv.update_tracked_camera(c2w)`
    (`viser_direct.py:353`: ref-swap under `_follow_lock`).
- `_force_viewer_rerender()` — `:336` → base `:1614`; no-op unless NS viewer up
  (it isn't, invariant #9).
- `_on_tracker_frame(...)` — `:338` → may `_dispatch_feedforward_async`.

### Render (viser-direct-render thread)

`_render_loop` — `viser_direct.py:624`:
- blocks on `self._render_requested.wait(timeout=1.0)` — `:632`.
- on wake: `_render_requested.clear()` `:637`, `self._render_once()` `:639`.
- `_render_once` — `:658`:
  - early-out if `_stop_event.is_set()` — `:659`.
  - snapshot client list under `_client_state_lock` — `:666-667`.
  - `_refresh_feed_image()` — `:672` (reads `_feed_rgb`/`_feed_dirty` under `_feed_lock`
    `:366-368`; then touches `self.server.gui` handle **outside** the lock `:378-388`).
  - per client: `_apply_follow_pose(client)` `:679` (reads `_follow_c2w` under
    `_follow_lock`, writes `client.camera.*`), `_build_camera_from_viser` `:680`.
  - **`with self.model_lock, torch.no_grad(): outputs = model.get_outputs(camera)`**
    — `:689-690`. This is the only place the render thread touches model params, and
    it holds the shared RLock.
  - `rgb_t.clamp(...).cpu().numpy()` `:695`, `client.scene.set_background_image(...)`
    `:703` (**outside** the lock — correct, render output is a private tensor).

### FF insert → render wake (FF bg thread)

`_anysplat_bg_run` — base `:3209` (and the rgbd path `_run_feedforward` `:2546`):
- model reads/writes wrapped in `with self._viser_lock_ctx():`
  (`:3281-3282` means read, `:3497-3506` insert).
- after insert: `self._viser_direct_register_ff_insert(inserted_ids)` — `:3508`
  → base `:1571`:
  - `if self._viser_direct_server is None: return` `:1579`.
  - **`if getattr(self._viser_direct_server, "is_closing", False): return`** `:1585`
    (the teardown guard — `is_closing` returns `_stop_event.is_set()`, `viser_direct.py:609`).
  - `srv.add_ff_insert_chunk(...)` `:1588` (no-op stub `viser_direct.py:512`).
  - `srv.request_render()` `:1592` → `viser_direct.py:611`: no-op if `_stop_event` set,
    else `_render_requested.set()`.

### Teardown

- `atexit.register(self._cleanup_viser_direct)` — base `:476`.
- `_cleanup_viser_direct` — base `:614`: `srv.flush_pending_ff(model)` (stub),
  `srv.close()` `:626`.
- `ViserDirectScene.close()` — `viser_direct.py:713`: `_stop_event.set()`,
  `render_thread.join(timeout=2.0)`, `server.stop()`.

---

## (b) Shared-state access table

`L` = pipeline `_model_lock` (the shared RLock, == `srv.model_lock`).

| Shared state | Site (file:line) | Thread | Lock held? | Racy? |
|---|---|---|---|---|
| `model.gauss_params` (read for render) | `viser_direct.py:690` `get_outputs` | render | **L (yes)** | no |
| `model.gauss_params["means"]` (read) | `base:3282` | FF bg | L (yes) | no |
| `model.*` insert (write) | `base:3498`, `base:2700` | FF bg | L (yes) | no |
| `model.*` rigid transform (write) | `base:2164` | tracker | L (yes) | no |
| `model.*` cull (write) | `base:2691`, `base:2877` | FF bg | L (yes) | no |
| `model.render_object_mask` (read) | `base:1680` | tracker | L (yes) | no |
| `model.get_outputs` (read) | `base:1699`, `1722` `_render_from_camera` | tracker / FF bg | L (yes, around get_outputs) | no for params; **see hazard H1** |
| **`model.training` flag** (`model.train()`/`model.eval()`) | `base:1696,1701` and `1719,1724`; model.py `1866,2129,2135` | tracker / FF bg | **NO — toggled OUTSIDE L** | **YES — H1** |
| `srv._model` (read) | `viser_direct.py:661` | render | no | no (single writer at attach, ref read) |
| `srv._stop_event` | `viser_direct.py:609,620,629,633,659,714` | all | no (Event = atomic) | no |
| `srv._render_requested` | `viser_direct.py:622,632,637` | all | no (Event) | no |
| `srv._client_state` (dict) | `viser_direct.py:545,577,667,479` | render + viser pool | **`_client_state_lock`** | no |
| `srv._initial_camera_applied` (set) | `viser_direct.py:486,569,578,567` | viser pool + tracker(via set_initial_camera) | **NO** | **YES — H2 (minor)** |
| `srv._feed_rgb` / `_feed_dirty` | `viser_direct.py:350,367` | tracker / render | `_feed_lock` | no |
| `srv._feed_gui_image` (handle) | `viser_direct.py:379-385,370` | render only | n/a | no |
| `srv._follow_c2w` | `viser_direct.py:360,403` | tracker / render | `_follow_lock` | no |
| `srv._feed_toggle.value` (read) | `viser_direct.py:365,400` | render | no (viser handle is thread-safe-ish) | low |
| `client.scene.set_background_image` | `viser_direct.py:541,703` | render + viser pool(connect) | no | low (viser-internal queue) |
| `client.camera.position/.wxyz` (write) | `viser_direct.py:411-412,594-595` | render / viser-pool / tracker | no | low (viser-internal) |
| `srv._render_count` / `_render_error_count` / `_render_window_total_ms` | `viser_direct.py:641,701,704,654-655` | render only | n/a | no |
| `self._viser_direct_server` (ref) | base `:1480,1494,1525,1542,1579,2194,615,629` | tracker / FF bg / atexit | **NO** | **YES — H3 (teardown)** |
| `self._last_motion_estimate` | base `:2169,1527` | tracker only | n/a | no |

---

## (c) Hazards

### H1 — `model.train()/eval()` flipped outside the model lock while the render thread reads `self.training` (RACE) — MEDIUM/HIGH
- Sites: `_render_from_camera` toggles `self.model.train()` at `base:1696` and
  `self.model.eval()` at `base:1701` (finally), **both outside** the
  `with self._viser_lock_ctx()` block at `:1698`. Same pattern in
  `_render_from_camera_at_scale` `:1719/1724`, and inside the model itself
  (`dynamic_gs_model.py:1866,2129,2135`).
- The render thread calls `model.get_outputs(camera)` under the lock
  (`viser_direct.py:690`), and `get_outputs` branches on `self.training`
  (`dynamic_gs_model.py:2219, 2268, 2336, 2400` — selects render_mode `RGB`
  vs `RGB+ED`, applies bilateral grid, applies camera_optimizer, etc.).
- Because the *flag itself* is a plain bool mutated by another thread **without
  the lock**, the render thread can observe `training=True` mid-flip even though
  it holds the lock for the params. Worst observed effect is a wrong-mode render
  (e.g. camera_optimizer pose correction applied during a viewer frame, or
  `RGB+ED` vs `RGB`), not a crash — but it is a genuine cross-thread read of an
  unsynchronized field. With invariant #4 (all LRs 0, camera_opt off in dynamic)
  the *visual* damage is small, which is why it has never been caught.
- Fix: move the `model.train()/eval()` toggles INSIDE the `with self._viser_lock_ctx()`
  block in `_render_from_camera`/`_render_from_camera_at_scale`, OR stop toggling
  training mode entirely for these reads (pass an explicit render-mode arg). The
  render thread should also assert/force a known mode under the lock.

### H2 — `_initial_camera_applied` set mutated from two threads without a lock (RACE, minor) — LOW
- `viser_direct.py:486` (tracker thread via `set_initial_camera`) and `:569`
  (viser pool via `_on_client_connect`) both `.add()`; `:578` (`_on_client_disconnect`)
  `.discard()`; `:567/482` read membership. No lock. The neighbouring
  `_client_state` dict IS protected by `_client_state_lock`, but the
  `_initial_camera_applied` set is read/written outside it (note `set_initial_camera`
  reads `_client_state` under the lock at `:478` but then mutates
  `_initial_camera_applied` outside it at `:486`).
- Worst case: a client gets snapped to the initial camera twice, or a
  connect/`set_initial_camera` interleave double-applies/skips a snap. Cosmetic
  (a one-frame camera jump), never a crash. `set` ops are not atomic across the
  GIL boundary for compound check-then-add.
- Fix: guard all `_initial_camera_applied` access with `_client_state_lock`
  (it is already the natural lock for client lifecycle).

### H3 — `self._viser_direct_server` read without synchronization across teardown (TOCTOU) — LOW/MEDIUM
- `_cleanup_viser_direct` (atexit) sets `self._viser_direct_server = None`
  (`base:629`) while the tracker thread / FF bg thread read it at
  `base:1525,1542,1579,2194`. The reads use a local snapshot pattern in some
  places (`srv = getattr(self, "_viser_direct_server", None); if srv is not None`
  at `:1542,2194,615`) which is safe, but others read `self._viser_direct_server`
  directly and then call a method on it (`:1525,1579,1588,1592`) — between the
  `is None` check and the method call the atexit thread could null it / `close()`
  it. The `is_closing` guard at `:1585` mitigates the FF path (a closing server
  no-ops `request_render`), and the daemon render thread is joined first, so the
  realistic blast radius is a late `request_render`/`update_*` after `close()`.
- The documented failure this guards against ("cannot schedule new futures after
  shutdown", `viser_direct.py:608,619`) is handled by `_stop_event`/`is_closing`,
  so this is mostly defused — but the `self._viser_direct_server` field itself is
  still read-then-use without a snapshot at `:1525,1579`.
- Fix: snapshot `srv = self._viser_direct_server` once at the top of
  `_push_viser_direct_transforms` and `_viser_direct_register_ff_insert` (as the
  other call sites already do) and operate on the local; rely on `is_closing`.

### H4 — `_render_loop`/`_render_once` swallow ALL exceptions; a torch CUDA error inside `get_outputs` is counted and retried forever — LOW
- `_render_loop` catches every exception (`viser_direct.py:640-643`) and only
  prints the first 3 + every 50th. `_render_once` per-client `try/except`
  (`:696-699`) `continue`s. A persistent failure (e.g. model in a bad state,
  device mismatch) silently spins the render thread at up to the request rate
  with no surfaced error after the 3rd. Not a deadlock (lock is released by the
  `with` on exception), but a silent-failure / log-flood hazard.
- Fix: after N consecutive errors, back off (sleep) or stop the render thread and
  log loudly once.

### H5 — per-tick allocations on the push path — LOW (perf, not correctness)
- `_push_viser_camera_feed` (`base:1547-1560`) does `arr.cpu().numpy()` +
  `np.ascontiguousarray(...)` on the full RGB **every tick** even when no client
  is connected (the JPEG encode is deferred to render-side, but the CPU copy is
  not). At 1920×1200 that's a ~6.9 MB host copy per tick on the tracker's
  critical path.
- `_build_camera_from_viser` (`viser_direct.py:144-163`) allocates several small
  tensors per client per render; negligible vs the render itself.
- `_render_once` `set_background_image` re-encodes a full JPEG per client per
  request (`:703`) — inherent to the push-image design, not a bug.
- Fix (optional): in `_push_viser_camera_feed`, skip the `.cpu().numpy()` copy
  when `srv` has zero connected clients (`srv._client_state` empty). Cheap guard,
  removes the per-tick host copy in the common headless-tracker case.

### H6 — `request_render()` collapses bursts to ONE render (by design) — NOT A BUG, noted
- `_render_requested` is a binary `Event`; multiple `request_render` calls between
  two render passes coalesce into a single render (`viser_direct.py:622,637`).
  This is intentional (one render per wake, latest state). Listed so the purge
  does not "fix" it into a queue — the render always pulls *live* model state, so
  coalescing loses nothing.

### H7 — `_refresh_feed_image` touches `self.server.gui` outside `_feed_lock` — LOW
- `viser_direct.py:378-388` mutates `_feed_gui_image.image`/`.visible` and calls
  `server.gui.add_image` outside any lock, from the render thread only. Single
  writer (render thread), so no intra-field race; viser GUI mutation is the only
  concern and viser serializes its own outbound. Benign as long as the render
  thread stays the sole writer of `_feed_gui_image` (it is).

---

## Dead/legacy branches relevant to the purge

These `ViserDirectScene` methods are **no-op stubs** (kept for call-site compat) —
safe to treat as inert when purging, but the call sites still execute them:
- `push_tracker_transform` (`viser_direct.py:506`) — called every tick via
  `_push_viser_direct_transforms` (`base:1531`). The whole
  `_push_viser_direct_transforms` method does nothing but read
  `_last_motion_estimate` and call a no-op; it could be removed along with its
  per-tick call at `live:334`.
- `add_ff_insert_chunk` (`:512`), `maybe_flush_ff_handle` (`:518`),
  `flush_pending_ff` (`:522`), `refresh_static_handle` (`:526`),
  `setup_handles` (`:494`, partially live — it forwards to `attach_model` +
  `set_initial_camera`).
- `_build_viser_direct_handles` (`base:1496`) / `_refresh_viser_direct_after_feedforward`
  (`base:1596`) / `_force_viser_direct_push` (`base:1605`) are thin wrappers over
  the stubs.
- `_force_viewer_rerender` (`base:1614`) is dead under invariant #9 (NS viewer
  off) — it early-returns because `_trainer.viewer_state` is None.

Do NOT remove: the `model_lock` swap (`base:1480`), the `is_closing` guard
(`base:1585`), `request_render` calls after writes (`base:2196,3508`), and the
`update_camera_feed`/`update_tracked_camera` real hooks.
