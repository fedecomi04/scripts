# Code Audit — `dynamic_gs/utils/viser_direct.py`

Module: server-side rasterize + push-image viser viewer for the LIVE dynamic-gs path (viser-direct, port 8081 — the canonical live-viz surface per Design Invariant #9; NS viewer is forbidden). 721 lines.

Reference counts from:
`grep -rn "<sym>" scripts/dynamic_gs scripts/scripts --include=*.py` excluding the definition file.

---

## 1) FUNCTION / CLASS MAP

### Module-level functions

- `_quat_wxyz_to_rotmat_np(q: (4,) ndarray) -> (3,3) ndarray)` — viser_direct.py:61 — normalizes a wxyz quaternion and builds a rotation matrix. **Callers: 0 external.** Used internally at :138 (`_build_camera_from_viser`). Not dead.
- `_rotmat_to_quat_wxyz_np(R: (3,3) ndarray) -> (4,) ndarray)` — viser_direct.py:76 — Shepperd's-method rotation→wxyz quaternion. **Callers: 0 external.** Used internally at :412 (`_apply_follow_pose`) and :593 (`_apply_initial_camera`). Not dead.
- `_build_camera_from_viser(client_camera, W, H, device) -> Cameras` — viser_direct.py:124 — converts a viser client camera (pos/wxyz/fov) into a nerfstudio `Cameras` (applies `_FLIP_YZ`, fy from vertical FOV, fx==fy). **Callers: 0 external.** Used internally at :680 (`_render_once`). Not dead.

### Module-level constant

- `_FLIP_YZ` — viser_direct.py:113 — Y-up/Z-back (nerfstudio/OpenGL) ↔ Y-down/Z-forward (viser/OpenCV) flip matrix, self-inverse. **Callers: 0 external.** Used internally at :140, :410, :592. Not dead.

### class `ViserDirectScene` — viser_direct.py:167

Instantiated once at `dynamic_gs_pipeline_base.py:1473`. **Class refs: 3 (instantiation at :1473, import at :1469, type/log).**

- `__init__(port=8081, render_hz=15.0, render_size=(1920,1080), jpeg_quality=92, opacity_floor=0.05, static_refresh_min_gap_s=-1.0, push_min_gap_s=0.033, ff_coalesce_gap_s=1.0)` — :184 — builds the `viser.ViserServer`, sets up locks/state, wires client connect/disconnect, builds GUI. Called at base:1473 **with only `port=` passed** — every other arg is left at default.
- `_build_gui(self)` — :293 — adds the shared "Tracker view" folder with "Show camera feed" + "Follow tracked frame" checkboxes. Internal-only; called from `__init__` :271. Callers external: 0.
- `keep_alive_until_shutdown(self, banner="Run finished")` — :311 — adds a red "Shutdown viewer" button at end-of-run. **Callers: 1** — `dynamic_gs_pipeline_recorded.py:155`.
- `wait_for_shutdown(self, timeout_s=None) -> bool` — :334 — blocks until the shutdown button fires or timeout. **Callers: 1** — `dynamic_gs_pipeline_recorded.py:162`.
- `update_camera_feed(self, rgb: ndarray)` — :342 — stashes the tracked frame RGB for the side-panel thumbnail (reference swap under lock). **Callers: 1** — `dynamic_gs_pipeline_base.py:1560`.
- `update_tracked_camera(self, camera_to_world: ndarray)` — :353 — stashes the tracked c2w for the "Follow tracked frame" toggle. **Callers: 1** — `dynamic_gs_pipeline_base.py:1567`.
- `_refresh_feed_image(self)` — :362 — render-thread side; lazily adds/updates the GUI feed image in place. Internal-only; called at :672 (`_render_once`). Callers external: 0.
- `_apply_follow_pose(self, client)` — :390 — if follow toggle on, snaps the client camera to the stored tracked c2w (with the `@ _FLIP_YZ` conversion). Internal-only; called at :679 (`_render_once`). Callers external: 0.
- `attach_model(self, model, device=None)` — :421 — stores the model + device and starts the render thread. **Callers: 1** — `dynamic_gs_pipeline_base.py:1481` (also invoked indirectly via `setup_handles`).
- `set_initial_camera(self, c2w_4x4, look_at=None, fov_y_rad=None)` — :444 — records the initial camera pose and snaps already-connected, not-yet-snapped clients. **Callers external: 0**, BUT reached internally via `setup_handles` :504. Not dead (see §2).
- `setup_handles(self, model, tracked_instance_id=None, initial_c2w=None)` — :494 — legacy entry point; now just `attach_model` + `set_initial_camera`. **Callers: 2** — `dynamic_gs_pipeline_base.py:1507` (+ docstring/ref).
- `push_tracker_transform(self, R, t)` — :506 — legacy no-op stub. **Callers: 1** — `dynamic_gs_pipeline_base.py:1531`.
- `add_ff_insert_chunk(self, model, inserted_ids)` — :512 — legacy no-op stub. **Callers: 3** — `dynamic_gs_pipeline_base.py:1588` (+ refs).
- `maybe_flush_ff_handle(self, model, force=False)` — :518 — legacy no-op stub. **Callers external: 0 — NO REFS FOUND.**
- `flush_pending_ff(self, model)` — :522 — legacy no-op stub. **Callers: 1** — `dynamic_gs_pipeline_base.py:622`.
- `refresh_static_handle(self, model)` — :526 — legacy no-op stub. **Callers: 1** — `dynamic_gs_pipeline_base.py:1601`.
- `_on_client_connect(self, client)` — :534 — sets placeholder bg image, registers per-client state, wires `camera.on_update`→`request_render`, snaps initial camera. Internal-only; wired at :276. Callers external: 0.
- `_on_client_disconnect(self, client)` — :575 — pops per-client state + initial-applied set. Internal-only; wired at :280. Callers external: 0.
- `_apply_initial_camera(self, client)` — :581 — snaps one client camera to `_initial_c2w`. Internal-only; called at :485, :568. Callers external: 0.
- `is_closing` (property) — :603 — True once `close()` began teardown; off-thread push guard. **Callers: 2** — `dynamic_gs_pipeline_recorded.py:152` (+ ref).
- `request_render(self)` — :611 — sets the render-requested event (no-op if closing). **Callers: 4** — `dynamic_gs_pipeline_base.py:1592`, :2196 (+ internal :488, :559, :572).
- `_render_loop(self)` — :624 — background render thread; blocks on the event, renders once per request, emits 1 Hz diagnostics. Internal-only; thread target at :438. Callers external: 0.
- `_render_once(self)` — :658 — snapshots clients, refreshes feed, builds camera, renders under `model_lock`, pushes JPEG bg per client. Internal-only; called at :639. Callers external: 0.
- `close(self)` — :713 — sets stop event, joins render thread (2 s), stops the server. `close` grep is 91 (too generic — matches every `.close()` in the repo); the relevant viser-direct teardown call is `_cleanup_viser_direct` at base:614/626.
- attribute `model_lock` (RLock, :208) — **reassigned externally** at `dynamic_gs_pipeline_base.py:1480` to the pipeline-owned `_model_lock`. **9 refs** (mostly inside base for the shared lock).

---

## 2) DEAD-CODE CANDIDATES

- `maybe_flush_ff_handle` — viser_direct.py:518 — **NO REFS FOUND** (0 external, never called internally). Genuine zero-ref. Confidence: **high**. It is a legacy API stub kept "for stability," but unlike its sibling stubs (`push_tracker_transform`, `add_ff_insert_chunk`, `flush_pending_ff`, `refresh_static_handle`) it has no surviving call site. Pure dead code (already a no-op, so removing it changes nothing). Not an entry point / callback / monkeypatch target.

Everything else with "0 external refs" (`_quat_wxyz_to_rotmat_np`, `_rotmat_to_quat_wxyz_np`, `_build_camera_from_viser`, `_FLIP_YZ`, `_build_gui`, `_refresh_feed_image`, `_apply_follow_pose`, `_on_client_connect`, `_on_client_disconnect`, `_apply_initial_camera`, `_render_loop`, `_render_once`, `set_initial_camera`) is **reachable internally** (render loop, GUI build, client-connect decorators, or via `setup_handles`) — NOT dead.

Note: the four other legacy stubs (`push_tracker_transform`, `add_ff_insert_chunk`, `flush_pending_ff`, `refresh_static_handle`) ARE still called from `dynamic_gs_pipeline_base.py` but do nothing. They are not dead code in the grep sense (referenced), but they are dead *logic* — see §4.

---

## 3) DATA-LIFECYCLE

This module touches no `.pt` warm-cache, no SHM, and none of the 4 identity buffers directly. It holds: one `viser.ViserServer` (a process/network resource), one daemon render thread, per-client dict state, GUI image handles, and a borrowed reference to the model (whose GPU tensors it reads under lock).

- **`viser.ViserServer`** — created `__init__` :201, stopped in `close()` :718 (`self.server.stop()` wrapped in bare try/except). Teardown is also driven by `_cleanup_viser_direct` (base:614, `atexit`-registered base:476). Single create / single stop — OK. If `ViserServer(port=...)` raises (port busy), `__init__` aborts and base:1486 catches it → `_viser_direct_server=None` fallback. No leak there.
- **Render thread** (`self._render_thread`, daemon) — started in `attach_model` :437, joined (2 s timeout) in `close()` :716. `attach_model` is re-entrant-safe: it only spawns a new thread if none alive (:435). Joined with timeout — if the render thread is wedged in `model.get_outputs` >2 s, `join` returns and the daemon may still be alive at interpreter exit (daemon=True → killed, acceptable). **Minor:** `attach_model` swaps `self._model` then conditionally starts a thread; if called twice it just swaps the reference — but `setup_handles`→`attach_model` is also called at D0 (base:1507) AFTER the first `attach_model` (base:1481), so `attach_model` runs at least twice. Harmless (model ref swap + thread-already-alive short-circuit).
- **Model reference** (`self._model`, :427) — borrowed, never freed by this module. Correct — the pipeline owns the model lifetime.
- **`model_lock` reassignment** — :208 creates an RLock; base:1480 **overwrites** it with the pipeline's `_model_lock` BEFORE `attach_model` starts the thread (:1481). So the render thread always reads the post-swap shared lock. Ordering is correct *only because* base does the swap before attach. If a future caller calls `attach_model` (starting the thread) before swapping the lock, the render thread would capture the local RLock and race the FF/tracker threads on a different lock. Fragile coupling, not currently a bug (DATA-LIFECYCLE / thread-safety, medium).
- **`_feed_rgb`** (:242) — swapped under `_feed_lock` in `update_camera_feed` (:350), read+cleared-dirty under the same lock in `_refresh_feed_image` (:367). The pipeline passes `np.ascontiguousarray(rgb_np[:,:,:3])` (base:1560), i.e. a fresh copy each tick — no aliasing of a mutating buffer. OK. One full-frame uint8 array retained between ticks (bounded, replaced not accumulated).
- **`_follow_c2w`** (:251) — written under `_follow_lock` (:360), read under same lock (:402). OK.
- **GUI image handle `_feed_gui_image`** (:243) — created lazily once (:380), updated in place after (:384). Single handle, no per-tick allocation of new handles. OK.
- **Per-client state `_client_state`** (:227) — added on connect (:545), popped on disconnect (:577) under `_client_state_lock`. `_initial_camera_applied` set added on connect/snap (:486, :569), discarded on disconnect (:578). Both keyed by `int(client.client_id)`. No leak across connect/disconnect cycles.
- **Per-render GPU→CPU transfer** (:695) — `outputs["rgb"]` is `.clamp().*255 .to(uint8).detach().cpu().numpy()` each render. One full-frame (default 1920×1080×3) host alloc per client per render, transient (handed to viser JPEG encode). Not retained. Acceptable; not a leak. The model `get_outputs` allocates render tensors on GPU per call — owned by torch, freed under `torch.no_grad()`.
- **Shape/format contract:** `_render_once` assumes `outputs["rgb"]` is `(H,W,3)` float in [0,1] (Splatfacto convention, :694). `_build_camera_from_viser` builds a `(1,3,4)` c2w and `(1,1)` intrinsics. If `model.get_outputs` ever returned a different layout, the uint8 cast/`set_background_image` would silently push garbage (no shape assert). Low risk while only Splatfacto-derived models are attached.

No double-load, no missing free of a persistent resource. The only unbounded-ish growth would be the per-client dicts, which are correctly popped on disconnect.

---

## 4) DESIGN SMELLS

- **Dead config args on `__init__`** — viser_direct.py:187-194. `render_hz`, `render_size`, `jpeg_quality`, `opacity_floor`, `static_refresh_min_gap_s`, `push_min_gap_s`, `ff_coalesce_gap_s` are constructor params. The sole call site (base:1473) passes **only `port`**, so all of these are frozen at their defaults forever. Of these:
  - `render_hz` (:203) — stored, **never read** anywhere (the render loop is event-driven via `_render_requested`, not Hz-throttled). Misleading: printed in the startup banner (:283) as if it governs cadence. Dead field. (severity: medium — actively misleads a reader into thinking there's a Hz cap)
  - `opacity_floor`, `static_refresh_min_gap_s`, `push_min_gap_s`, `ff_coalesce_gap_s` (:191-194) — explicitly commented "Legacy kwargs kept so old call sites don't trip — ignored." Truly dead; never assigned to `self`. Honest but pure cruft. (low)
  - `render_size` / `jpeg_quality` ARE read (:204-205, :673, :703) but never overridden — so the hardcoded 1920×1080 render is unconfigurable from the pipeline config even though `viser_direct_port` IS plumbed. Inconsistent. (low)
- **`set_initial_camera` stores `look_at` and `fov_y_rad` that are never consumed** — :472-475 set `self._initial_look_at` and `self._initial_fov_y`, but `_apply_initial_camera` (:581) only reads `_initial_c2w`; the FOV used at render time comes from the live viser client (`_build_camera_from_viser` :137), not `_initial_fov_y`. The docstring (:456) even claims `look_at` defaults are "handled inside `_apply_initial_camera`" — they are NOT. Dead state + misleading docstring. (medium)
- **Five legacy no-op stub methods** — `push_tracker_transform` (:506), `add_ff_insert_chunk` (:512), `maybe_flush_ff_handle` (:518), `flush_pending_ff` (:522), `refresh_static_handle` (:526). Four are still called from `dynamic_gs_pipeline_base.py` (1531/1588/622/1601) and do nothing; the call sites do real work (e.g. base:1531 reads `est.rotation/.translation` then calls a no-op). This is a leaky abstraction: the "API surface kept stable" is dead plumbing the pipeline still threads arguments into. Given the module is being purged tomorrow, both the stubs AND their base call sites are removable. (medium)
- **Broad swallowed exceptions throughout** — every viser interaction is wrapped in bare `except Exception` with at most a print: GUI build (:308), feed update (:386), follow-pose (:413), camera build (:681), get_outputs (:696), set_background_image (:705), client connect (:542, :560), initial camera (:596), server stop (:719). The render-loop top-level (:640) increments `_render_error_count` and rate-limits the print to first-3-then-every-50. This means a *persistent* failure (e.g. model on wrong device, shape mismatch) shows 3 lines then goes quiet — the viewer silently shows the last good frame. For a purged module this is moot, but it is a real "swallowed exception hides a broken render" smell. (low/medium)
- **Thread-safety — `_initial_camera_applied` set mutated without its own lock.** It is read under `_client_state_lock` in `set_initial_camera` (:478-483) but then **mutated outside that lock** at :486 (`.add`), and separately mutated at :569 (`_on_client_connect`, no lock) and :578 (`_on_client_disconnect`, no lock). `_on_client_connect`/`disconnect` run on viser's own server thread while `set_initial_camera` runs on the pipeline thread. A connect racing a `set_initial_camera` could double-snap or skip-snap a client (the `pending` snapshot at :478 can go stale before :486). Cosmetic at worst (a camera snap), not a crash, but it is an unsynchronized cross-thread set. (low)
- **Thread-safety — `model_lock` held across the full `get_outputs`** (:689). Correct for race-freedom (Invariant #9 mandates the shared lock), but it serializes the render thread against the tracker tick and the FF bg insert — the render holds the lock for the entire ~25 ms+ rasterization. This is the documented trade-off (CLAUDE.md notes 26 ms/tick model-lock-wait); not a bug, but the lock granularity is coarse: the lock is needed only to read a consistent `gauss_params` snapshot, yet it is held through the GPU kernel + the implicit sync inside `get_outputs`. (low — known/accepted)
- **`_render_count` / `_render_window_total_ms` updated without a lock** (:701, :704) — only the render thread writes them and the same thread reads them in the diag block (:646-655), so single-threaded by construction. Fine. (informational)
- **Diagnostic counter `_render_count` gates the 1 Hz log on `> 0` (:646)** — so if every render *errors* (count stays 0 while `_render_error_count` climbs), the periodic diagnostic never prints and only the rate-limited error line shows. Minor observability gap. (low)
- **`is_closing` / `request_render` no-op-on-closing guard** (:609, :620) is correct and necessary (prevents `cannot schedule new futures after shutdown` from late FF-bg pushes). Good. (informational, not a smell)

### Invariant-protected / intentional (NOT flagged)
- This module deliberately renders server-side and is the **only** sanctioned live-viz path (Invariant #9 — viser-direct on :8081, never the NS viewer). The shared `model_lock` (base:1480 reassignment + render under it at :689) is the documented concurrency-safety mechanism, not a smell.
- The legacy stubs exist by explicit design ("API surface kept stable", module docstring :33-38) — flagged as removable cruft, not as accidental dead code.
