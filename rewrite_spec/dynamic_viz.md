# `dynamic_viz.py` — the viser-direct live viewer bridge (layer: dynamic)

## (1) RESPONSIBILITY

Run the viser-direct server on its own light, event-driven daemon thread: per render request, read an immutable `GaussianSnapshot` (NEVER the live model under a foreign lock), rasterize each connected client's camera server-side, and push the JPEG as the client background image — the ONE sanctioned live-viz surface (Invariant #9; the NS viewer is forbidden) — plus the side-panel camera-feed thumbnail and the follow-tracked-frame toggle.

---

## (2) PUBLIC INTERFACE (the contract `pipeline.py` calls)

```python
class ViserBridge:
    """Server-side rasterize + push-image viser viewer. Spin up once per live run.
    The pipeline calls attach() once the model + GaussianSet exist, then request_render()
    after every scene mutation (tracker write, FF insert/cull). Read-only of scene state."""

    def __init__(self, cfg: "ViserConfig", *, render_size: tuple[int, int] = (1920, 1080),
                 jpeg_quality: int = 92) -> None:
        """Start the viser.ViserServer on cfg.port, build the GUI ('Tracker view' folder:
        'Show camera feed' + 'Follow tracked frame'), wire client connect/disconnect +
        camera.on_update -> request_render. Raises if viser is not installed AND cfg.enabled."""

    def attach(self, render_fn: "Callable[[Cameras], Tensor]",
               snapshot_fn: "Callable[[], GaussianSnapshot]") -> None:
        """Hand the bridge its two read-only callbacks and start the render thread.
        render_fn(camera) -> (H,W,3) float[0,1] rgb, MUST internally take _model_lock + no_grad
        and return a private tensor (the bridge never touches gauss_params or the lock itself).
        snapshot_fn() -> current GaussianSnapshot (used only for the version-skip optimization).
        Idempotent: re-attach swaps the callbacks; the thread is started at most once."""

    def set_initial_camera(self, c2w_4x4: np.ndarray) -> None:
        """Pose newly-connected clients land on; also snaps already-connected, not-yet-snapped
        clients (at most one snap per client/session — never undoes a user's manual orbit)."""

    def request_render(self) -> None:
        """Wake the render thread for ONE pass (binary Event; bursts coalesce to one render of
        latest state). No-op once closing. Called from every tracker tick + every FF insert site."""

    def update_camera_feed(self, rgb: np.ndarray) -> None:
        """Stash the tracked frame's RGB (H,W,3 uint8) for the side-panel thumbnail (ref-swap
        under a private lock; JPEG encode deferred to the render thread, and only if a client is up)."""

    def update_tracked_camera(self, c2w_4x4: np.ndarray) -> None:
        """Stash the tracked frame's c2w (3x4 or 4x4); the render loop snaps the viewer camera to
        it when 'Follow tracked frame' is on (ref-swap under a private lock)."""

    @property
    def is_closing(self) -> bool:
        """True once close() has begun teardown. Off-thread callers (FF bg) check before pushing
        so they never submit onto a shutting-down executor ('cannot schedule new futures...')."""

    def keep_alive_until_shutdown(self, banner: str = "Run finished") -> None:
        """End-of-run (recorded): add a red 'Shutdown viewer' button so the operator can inspect
        the final scene. Pair with wait_for_shutdown(). Render thread keeps running. Idempotent."""

    def wait_for_shutdown(self, timeout_s: float | None = None) -> bool:
        """Block until the 'Shutdown viewer' button fires or timeout. True=fired, False=timeout.
        Returns immediately if already closing."""

    def close(self) -> None:
        """Set the stop event, join the render thread (timeout), stop the viser server.
        Idempotent; safe under atexit."""
```

Notes:
- The interface drops to: lifecycle (`__init__`/`attach`/`close`), the render trigger (`request_render`), two real per-tick hooks (`update_camera_feed`/`update_tracked_camera`), the initial-camera snap, and the recorded-run keep-alive pair. Everything else in the current file is internal or dead.
- `attach()` takes a **render callback + a snapshot callback**, not the model. The bridge never holds the model, never imports `scene_model`, and never takes `_model_lock` itself — the lock lives inside `render_fn` (owned by the pipeline). This removes the fragile post-hoc `server.model_lock = self._model_lock` swap (see DROPPED / DEPENDS ON).

---

## (3) DEPENDS ON (NEW modules only)

- **`config.py`** — `ViserConfig` (`enabled`, `port`) consumed by the ctor. `render_size`/`jpeg_quality` are ctor args defaulted here (config currently does not carry them — see OQ1).
- **`gaussian_set.py`** — the `GaussianSnapshot` TYPE only (returned by `snapshot_fn`, used for the version-skip optimization). The bridge never calls a `GaussianSet` surgery method and never mutates the set.
- **`frame.py`** — indirectly, for the c2w convention contract: `update_tracked_camera` / `set_initial_camera` consume **OpenGL c2w** (the same `Frame.c2w_4x4` convention) and the bridge applies the documented OpenGL(Y-up,Z-back) ↔ viser(Y-down,Z-forward) `_FLIP_YZ`. No `Frame` object enters the bridge.

It does NOT depend on `pipeline.py` (the orchestrator calls IN), the tracker, the FF dispatcher, the decoders, `scene_model`, or nerfstudio's viewer. `render_fn`/`snapshot_fn` are closures the pipeline injects, so the bridge stays decoupled from where the lock + model live.

---

## (4) CONSUMES / PRODUCES

**CONSUMES:**
- `render_fn(camera) -> (H,W,3) float[0,1] rgb tensor` — the pipeline's locked render closure (Splatfacto/`scene_model.get_outputs`, internally under `_model_lock` + `torch.no_grad`). The bridge calls it on the render thread and treats the returned tensor as private/disposable.
- `snapshot_fn() -> GaussianSnapshot` — for `version`-based render skipping (don't re-render if neither the scene version nor any client camera changed).
- viser client camera state (`position`, `wxyz`, `fov`) per connected client — read each render to build a nerfstudio `Cameras`.
- `update_camera_feed` RGB (H,W,3 uint8) and `update_tracked_camera` c2w (OpenGL, 3x4/4x4) per tick.

**PRODUCES:**
- per-client `client.scene.set_background_image(jpeg)` — the rendered splat view.
- the side-panel `gui.add_image` thumbnail (live camera feed), updated in place.
- viewer-camera snaps (initial-camera + follow-tracked-frame) written to `client.camera.position/.wxyz`.
- `is_closing` flag for off-thread teardown guarding.

No `.pt`, no SHM, no identity buffers, no disk artifacts. Borrows nothing it must free except the viser server + its own render thread.

---

## (5) SOURCE MOVED IN (current `file:symbol` → what it becomes)

| Current (`viser_direct.py`) | Becomes |
|---|---|
| `ViserDirectScene` class | `ViserBridge` (renamed, slimmed; ctor takes `ViserConfig` + render/jpeg sizes, not 8 scalar kwargs) |
| `_quat_wxyz_to_rotmat_np`, `_rotmat_to_quat_wxyz_np`, `_FLIP_YZ`, `_build_camera_from_viser` | kept as private module helpers (the camera-convention math; sole consumer is `_render_once`) |
| `attach_model` + the implicit `model_lock` swap (base:1480) | `attach(render_fn, snapshot_fn)` — lock + model stay behind `render_fn`; the swap disappears (fixes H-viser-swap setup-ordering, RUNTIME_target_architecture §"setup ordering fix") |
| `_render_loop` / `_render_once` | kept (event-driven loop) but `get_outputs`-under-`model_lock` becomes a call to the injected `render_fn`; per-client loop + JPEG push unchanged. Add a `version`-skip early-out (snapshot bounded-work, ARCHITECTURE_PRINCIPLES §3) |
| `request_render` + `is_closing` + `_stop_event`/`_render_requested` Events | kept verbatim (the coalescing trigger + the teardown guard are load-bearing — RUNTIME_target_architecture "KEEP") |
| `_on_client_connect` / `_on_client_disconnect` / `_apply_initial_camera` / `set_initial_camera` | kept; `_initial_camera_applied` set now guarded by `_client_state_lock` (fixes H2 unsynchronized cross-thread set) |
| `update_camera_feed` / `update_tracked_camera` / `_refresh_feed_image` / `_apply_follow_pose` | kept (the two REAL per-tick hooks + their render-side appliers) |
| `_build_gui` + the "Tracker view" folder | kept |
| `keep_alive_until_shutdown` / `wait_for_shutdown` / `_shutdown_*` | kept (recorded-run end-of-run inspect) |
| `close` | kept |

---

## (6) DROPPED (NOT carried, with reason + audit ref)

| Dropped | Reason | Audit ref |
|---|---|---|
| `push_tracker_transform`, `add_ff_insert_chunk`, `maybe_flush_ff_handle`, `flush_pending_ff`, `refresh_static_handle`, `setup_handles` (legacy Path-A no-op stubs) | All no-ops; the render thread reads live scene state each pass so per-tick motion + FF inserts appear automatically. The pipeline call sites are deleted too. `maybe_flush_ff_handle` is already zero-ref. | `viser_direct.md` §2 (dead-code), §4 ("five legacy no-op stubs"); RUNTIME_viser_push "Dead/legacy branches"; RUNTIME_target_architecture "DELETE" |
| ctor kwargs `render_hz`, `opacity_floor`, `static_refresh_min_gap_s`, `push_min_gap_s`, `ff_coalesce_gap_s` | `render_hz` is stored but never read (loop is event-driven, not Hz-throttled) and misleads in the banner; the other four are explicitly "ignored legacy kwargs". | `viser_direct.md` §4 ("dead config args on `__init__`", medium) |
| `_initial_look_at` / `_initial_fov_y` state + the `look_at`/`fov_y_rad` params of `set_initial_camera` | Stored but never consumed — `_apply_initial_camera` reads only `_initial_c2w`; render-time FOV comes from the live viser client, not `_initial_fov_y`. Docstring claim was false. | `viser_direct.md` §4 ("stores look_at and fov that are never consumed", medium) |
| The post-hoc `server.model_lock = pipeline._model_lock` swap (base:1480) + the public `model_lock` attribute | Replaced by the `render_fn` closure carrying the lock internally; the bridge no longer owns or swaps a lock, removing the "thread started with the wrong lock" fragility. | `viser_direct.md` §3 ("model_lock reassignment ... fragile coupling"); RUNTIME_target_architecture §"setup ordering fix" (H viser-swap) |
| Direct `model.get_outputs` + `with self.model_lock` inside the render thread | The bridge must not know the model or the lock; it calls `render_fn`. Single-source-of-truth: all scene reads go through the snapshot/locked-render the pipeline provides. | ARCHITECTURE_PRINCIPLES §1, §9; RUNTIME_target_architecture §"the one lock" |
| FF-video machinery (`feedforward_video_out`) | No writer exists anywhere; never produced an mp4. Not a viz-bridge concern. | CLAUDE.md viser notes ("no writer is implemented — no mp4"); gaussian_set.md §6 |
| `_force_viewer_rerender` coupling / any NS-viewer render-state-machine plumbing | NS viewer is OFF under Invariant #9; the bridge is the only renderer. No `_trainer.viewer_state` hook. | `viser_direct.md` §4; RUNTIME_target_architecture #9, "DELETE" (`_force_viewer_rerender`) |
| "oneshot FF" / per-call splat-handle path (Path A) | Superseded by push-image; the file's own history block documents the WIP-handle flash + 1.0.29 browser crash. | current `viser_direct.py` module docstring (history); MEMORY `project_viser_path_a_status` |

Kept-but-hardened (NOT dropped): `is_closing`/`request_render` no-op-on-closing guards, the `_FLIP_YZ` follow/initial conversion, the per-client `on_update -> request_render` wiring, the placeholder-image-on-connect.

---

## (7) INVARIANTS PRESERVED (CLAUDE.md) + how

- **#9 (live viz is viser-direct ONLY, port 8081, NEVER the NS viewer):** this module IS the sanctioned surface. It renders server-side via the injected `render_fn` and pushes images client-side; it never instantiates or pokes nerfstudio's `ViewerState`/render state machine, and the method configs keep `vis="tensorboard"`. The render thread is **read-only of scene state** — it calls `render_fn` (a locked snapshot render) and a frozen `GaussianSnapshot`; it never mutates `gauss_params` or any identity buffer.
- **#4 (dynamic phase = pure runtime, no gradient descent):** the bridge does zero optimization and zero model mutation — it only reads to render. Nothing here can introduce a backward/optimizer step.
- **#6 (background = Gazebo sky `(0.86,0.92,1.0)`):** unchanged — the background composite lives in `render_fn`/`scene_model`, not here. The bridge only re-encodes whatever rgb the render returns.
- **#8 (identity buffers):** the bridge reads buffers only through the immutable `GaussianSnapshot` (detached) and never writes them — buffer ownership stays with `GaussianSet`/the phases.
- **Single lock discipline (RUNTIME_target_architecture §"the one lock"):** the bridge holds NO scene lock of its own; the only critical section it depends on is inside `render_fn`. This removes a class of "render thread holds the wrong lock" bugs.

---

## (8) THREADING

- **Owns ONE daemon thread:** `viser-direct-render`, started in `attach()`, joined (with timeout) in `close()`. Plus it borrows viser's internal server thread-pool for `on_client_connect`/`disconnect`/`camera.on_update`/GUI-button callbacks (viser-owned).
- **Render thread blocks ONLY on:** its own `_render_requested` Event (with a ~1 s timeout for prompt stop), and — transiently, INSIDE `render_fn` — the pipeline's `_model_lock` (held by the pipeline, not by the bridge). The bridge must NOT block on: SHM, subprocess IPC, the FF slot lock, disk, or the publisher pose/joint lock.
- **Lock discipline:**
  - `_client_state` + `_initial_camera_applied` → guarded by the private `_client_state_lock` (the latter is the H2 fix — both mutated under the same lock now).
  - `_feed_rgb`/`_feed_dirty` → `_feed_lock`; `_follow_c2w` → `_follow_lock` (both ref-swaps, render thread reads under the same lock).
  - `_feed_gui_image` GUI handle → render-thread-only (single writer, no lock needed).
  - `_render_count`/error counters → render-thread-only.
- **Coalescing is intentional:** `request_render` sets a binary Event; multiple requests between two passes collapse to one render of the LATEST state (RUNTIME_viser_push H6 — do not "fix" into a queue). With the snapshot `version`-skip, an idle scene with no camera move does zero work.
- **Bounded work / fault isolation (ARCHITECTURE_PRINCIPLES §3, §7):** per render = N clients × (camera build + one `render_fn` + one JPEG encode), N typically 1. Persistent render errors must NOT spin silently forever — after K consecutive failures the loop backs off (sleep) and logs loudly once, instead of the current first-3-then-every-50 then-quiet pattern (RUNTIME_viser_push H4). The render thread is a daemon, so a wedged `render_fn` is killed at interpreter exit after the join timeout.
- **Teardown ordering:** `close()` sets `_stop_event` (→ `is_closing` True → off-thread `request_render` becomes a no-op, preventing "cannot schedule new futures after shutdown"), joins the render thread, then stops the server. Off-thread callers (FF bg) check `is_closing` before pushing.

---

## (9) OPEN QUESTIONS

1. **`render_size` / `jpeg_quality` in config?** Currently `ViserConfig` carries only `enabled` + `port`; the 1920×1080 render size and JPEG quality are hardcoded ctor defaults. Add them to `ViserConfig` (so the live render resolution is configurable), or keep them as ctor args the pipeline passes? (Leaning: add `render_size`/`jpeg_quality` to `ViserConfig` — the port is already there.)
2. **`render_fn` vs handing in the model+lock.** The spec injects a `render_fn` closure (bridge stays model-agnostic + lock-agnostic). The alternative — pass `scene_model` + `_model_lock` and call `get_outputs` here — is closer to today's code but reintroduces the model coupling and the lock swap. Confirm the closure approach (it is what kills H-viser-swap).
3. **Snapshot version-skip granularity.** The version-skip needs "scene unchanged AND no client camera moved" to be a true no-op. Camera moves already fire `request_render`; should the bridge also track each client's last (pos,wxyz,fov) to skip re-rendering a client whose camera AND the scene both didn't change, or is per-request "render all clients" acceptable (N≈1 in practice)? (Leaning: track per-client last-camera + scene `version`; skip identical.)
4. **Multi-client policy.** Today every connected client gets its own per-pose render each pass (N renders/request). For teleop there is normally one operator. Keep N-client support, or cap to the first client (and reject/mirror extras) to bound worst-case render latency? (ARCHITECTURE_PRINCIPLES §2 — predictable p99.)
5. **Does the recorded pipeline still need `keep_alive_until_shutdown`?** It is the only caller-pair (recorded end-of-run inspect). If the rewrite's recorded path keeps a "hold the viewer open until the operator clicks shutdown" UX, keep it; otherwise drop the shutdown-button subtree too.
6. **Camera-feed thumbnail necessity.** `update_camera_feed`/`_refresh_feed_image` add a per-tick uint8 ref-swap on the tracker hot path (the `.cpu().numpy()` copy is on the tracker side — RUNTIME_viser_push H5). Keep the live thumbnail (useful operator context) but gate the upstream copy on "≥1 client connected", or drop the thumbnail entirely to keep the tracker tick allocation-free? (Leaning: keep, but the pipeline skips the copy when `not bridge.has_clients()` — may need a small `has_clients()` accessor added to the interface.)
