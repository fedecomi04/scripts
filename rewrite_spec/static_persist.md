# `static_persist.py` — module spec (layer: static)

## 1. Responsibility

Own the on-disk static→dynamic handoff artifacts — the warm-cache `.pt` (6 gauss_params + 4 identity buffers + minimal pipeline state) and the seed PLY / Phase-0 init-artifact paths — making the `.pt` a **config-fingerprint-tagged versioned contract** that round-trips losslessly on save and **fails loudly with an actionable message on drift** (wrong fingerprint, schema version, or missing key), never a raw tensor-shape traceback.

## 2. Public interface (the contract other modules call)

```python
WARM_CACHE_NAME: str = "static_state.pt"
LEGACY_WARM_CACHE_NAME: str = "post_fusion_state.pt"   # read-only back-compat fallback
SEED_PLY_NAME: str = "depth_camera_init_points.ply"
SCHEMA_VERSION: int = 1                                 # bumped on any blob-layout change


@dataclass(frozen=True)
class WarmCacheLoadResult:
    """Outcome of a warm-cache load attempt (no exceptions cross the module boundary for
    the common 'cache absent / config drifted' cases — caller decides fall-back-to-train)."""
    success: bool
    num_points: int = 0
    path: Optional[Path] = None          # the file actually read (resolved legacy fallback)
    error: Optional[str] = None          # human-readable, ready to log; set iff success is False
    drift: bool = False                  # True iff failure was a fingerprint/schema mismatch (vs IO/missing)


def warm_cache_path(data_dir: Path) -> Path:
    """Canonical `<data_dir>/static_scene/static_state.pt`. The single place this path is built."""

def warm_cache_exists(data_dir: Path) -> bool:
    """True if either the canonical or the legacy `.pt` is present beside `static_scene/`."""

def save_warm_cache(gaussian_set, fingerprint: str, *, data_dir: Path,
                    pipeline_state: Optional[dict] = None) -> bool:
    """Snapshot the live scene for a future warm-start. Pulls a `GaussianSnapshot` from
    `gaussian_set`, writes {schema_version, fingerprint, num_points, params(6), buffers(4),
    pipeline_state} atomically (tmp + os.replace) to `warm_cache_path(data_dir)`.
    Returns True on success; logs and returns False on any IO error (best-effort snapshot)."""

def load_warm_cache(gaussian_set, expected_fingerprint: str, *, data_dir: Path,
                    strict_fingerprint: bool = True) -> WarmCacheLoadResult:
    """Resolve canonical→legacy path, torch.load, VALIDATE schema_version + fingerprint BEFORE
    touching the model, then hand (state_dict-shaped params/buffers + num_points) to
    `gaussian_set.reload_from_state_dict(...)`. On fingerprint/schema mismatch returns
    success=False, drift=True, error='<what changed>; delete <path> and re-run static-gs'
    WITHOUT mutating `gaussian_set`. Returns the restored pipeline_state via the result-adjacent
    accessor (see `loaded_pipeline_state`)."""

def loaded_pipeline_state(result: WarmCacheLoadResult) -> dict:
    """The `pipeline_state` dict carried by the last successful load for `result` (empty dict if
    none / failed). Kept off the frozen dataclass so the result stays a small value type."""

def read_warm_cache_meta(data_dir: Path) -> Optional[dict]:
    """Peek {schema_version, fingerprint, num_points} WITHOUT loading tensors (for diagnostics /
    `view_static_ckpt`-style tools). None if no cache. Does not validate against any config."""

def seed_ply_path(data_dir: Path) -> Path:
    """Canonical `<data_dir>/static_scene/depth_camera_init_points.ply` (the Splatfacto init seed)."""

def init_artifacts_dir(data_dir: Path) -> Path:
    """`<data_dir>/static_scene/initialization_artifacts/` — per-object SAM3D PLY + pose JSON
    produced by Phase-0; this module owns the path constant, not the writing of its contents."""
```

## 3. Depends on (NEW modules only)

- **`gaussian_set.py`** — `save_warm_cache` reads `gaussian_set.snapshot()` (→ `GaussianSnapshot`: 6 detached params + 4 buffers + num_points); `load_warm_cache` calls `gaussian_set.reload_from_state_dict(state_dict, num_points)` (the SINGLE owner of the Parameter realloc, buffer rebuild, means-grad-hook re-bind, optimizer refresh, length-invariant assert). This module performs **no** tensor surgery itself.
- **`config.py`** — for the `config_fingerprint(cfg) -> str` value passed in as `fingerprint` / `expected_fingerprint`. This module does NOT compute the fingerprint (it receives it) and does NOT read `os.environ`.
- No other NEW-module dependency. The pipeline (god-file) and the static/preseg method paths are *callers*.

## 4. Consumes / produces

**Consumes (inputs at the boundary):**
- A `GaussianSnapshot` (from `gaussian_set.snapshot()`) on save — the 6 params + 4 identity buffers + num_points.
- A `fingerprint: str` (from `config.config_fingerprint`) on save; an `expected_fingerprint: str` on load.
- An optional `pipeline_state: dict` on save (small, JSON-serializable scalars only — see Open Q1): the minimal non-model state the dynamic phase needs that is NOT reconstructable (e.g. `_static_converged_step`). NOT the model.
- The on-disk `.pt` (canonical or legacy) on load.

**Produces (outputs):**
- The `.pt` warm-cache file: a single dict blob
  `{ "schema_version": int, "fingerprint": str, "num_points": int,
     "params": {6 named tensors}, "buffers": {4 named tensors}, "pipeline_state": {...} }`.
  This is the versioned boundary contract (Architecture principle #8).
- A `WarmCacheLoadResult` on load (success/num_points/path/error/drift) — `success=False` is "fall back to standard static + Phase-0b".
- Path constants for the seed PLY and init-artifacts dir (no file content authored here).

## 5. Source moved in (current `file:symbol` → what it becomes)

| Current | Becomes |
|---|---|
| `persistence/post_fusion_cache.py:save_post_fusion_state` | `save_warm_cache` — now reads a `GaussianSnapshot` (not `model.state_dict()`), stamps `schema_version` + `fingerprint`, writes atomically, carries `pipeline_state`. |
| `persistence/post_fusion_cache.py:load_post_fusion_state` | `load_warm_cache` — validate fingerprint/schema FIRST, then delegate the resize/hook-rebind to `gaussian_set.reload_from_state_dict` (NOT inlined here). |
| `persistence/post_fusion_cache.py:PostFusionLoadResult` | `WarmCacheLoadResult` (+ `path`, `drift` fields). |
| `persistence/post_fusion_cache.py:_resolve_cache_path` + `_LEGACY_CACHE_NAME` | `warm_cache_path` / `warm_cache_exists` + the legacy fallback folded INTO `load_warm_cache` (one resolver, not two — see DROPPED). |
| `dynamic_gs_pipeline_base.py:_load_warm_cache_or_die` (the path-resolve + legacy fallback half) | `warm_cache_exists` + `load_warm_cache`; the "or die" (raise on absent) becomes the *caller's* policy on `WarmCacheLoadResult.success`. |
| `post_fusion_cache.py:121-143` Parameter realloc + `model.load_state_dict(strict=False)` + means-hook re-bind | **NOT here** — moved to `gaussian_set.reload_from_state_dict` (per `gaussian_set.md` §5). `static_persist` only feeds it the dict + N. |
| seed-PLY / init-artifact path strings scattered in `static_gs_pipeline.py` / `fusion/phase0.py` (`depth_camera_init_points.ply`, `initialization_artifacts/`) | `seed_ply_path` / `init_artifacts_dir` constants (path ownership centralized; content-writing stays in the fusion/static modules). |

## 6. Dropped (NOT carried, with reason + audit ref)

- **`model.state_dict()` as the save payload.** The blob is now an explicit `{params, buffers}` selection from the snapshot, not the whole nerfstudio module dict. *Reason:* `state_dict()` dragged in viewer-GUI / framework keys that `strict=False` had to silently swallow on load (the exact drift-masking smell). Explicit selection = no silent partial copies. *Audit: persistence_init_misc.md §3 risk #1 (`strict=False` masks shape/key drift; only the 6 gauss_params verified, buffers ride unverified).* 
- **`strict=False` `load_state_dict` semantics for integrity.** Replaced by an up-front fingerprint+schema gate + an explicit length-invariant assert inside `reload_from_state_dict`. *Reason:* `strict=False` let a missing/mis-shaped identity buffer keep a cold-start value with no warning. *Audit: persistence_init_misc.md §3 risk #1; ARCHITECTURE_PRINCIPLES §8 (config-implicit serialized state → loud versioned contract).* 
- **Two independent legacy-fallback implementations** (`_resolve_cache_path` AND base.py `_load_warm_cache_or_die` both re-implementing `static_state.pt → post_fusion_state.pt`). Collapsed to one resolver in this module. *Audit: persistence_init_misc.md §3 risk #2 (duplication/desync, must be kept in lockstep).* 
- **The `post_fusion` naming** (`save_post_fusion_state` / `PostFusionLoadResult` / `_LEGACY_CACHE_NAME` prefix). Renamed to `warm_cache` — one artifact, one name. The file is `static_state.pt` (legacy read-only). *Audit: persistence_init_misc.md §4 ("three different names for one artifact").* 
- **The in-line Parameter realloc + means-grad re-bind code.** Not duplicated here — single-owner is `gaussian_set.reload_from_state_dict`. *Audit: gaussian_set.md §5 (warm-restart resize is one owner); RUNTIME_warmload_lifecycle (b) (hook re-bind is mandatory, lives with the resize).* 
- **FF-video machinery / oneshot-FF / `feedforward_video_out`.** Never persistence concerns; no writer exists. *Audit: gaussian_set.md §6; RUNTIME_target_architecture DELETE list.* 
- **`DynamicKeyframeFilter` and the rest of `keyframe_filter.py`.** Co-located in the audit scope but unrelated to persistence and proven dead (0 instantiations). Not pulled into this module. *Audit: persistence_init_misc.md §2 (HIGH-confidence dead).* 
- **Exception-swallowing on the final-snapshot save being silently dropped to a log.** Kept best-effort (returns False) for the dynamic exit snapshot, BUT the static-gs producer caller must surface `False` (no-warm-cache-produced is currently swallowable). *Reason/Audit: persistence_init_misc.md §3 SAVE-path note (callers don't hard-fail on `ok==False`).* — flagged for the caller, not fixed silently here.

## 7. Invariants preserved (CLAUDE.md) + how

- **#8 (per-object identity buffers).** The blob carries exactly the 4 buffers (`object_flags`, `object_instance_ids` [long], `sam3d_init_target_flags`, `inserted_flags`). `save_warm_cache` writes them verbatim from the snapshot; `load_warm_cache` hands them straight to `reload_from_state_dict`, which asserts `all params+buffers share shape[0]==num_points` — so a buffer can never silently desync from geometry on restore. `sam3d_init_target_flags` is persisted as-is (all-zeros expected; never written here). NO buffer is dropped "because it looks unused."
- **#1 / #4 (means LR=0 static / all gauss LRs=0 dynamic).** This module sets no LRs and re-binds no hook itself; the means-grad hook re-bind + phase optimizer re-apply happen inside `reload_from_state_dict` (gaussian_set). `static_persist` only guarantees the restored tensors are the saved ones — it never resurrects a trainable means. *(RUNTIME_warmload_lifecycle: the re-bind is mandatory and owned by the resize site.)*
- **#5 (`outputs/` suppressed).** All artifacts written under `<data_dir>/static_scene/` via `warm_cache_path` / `seed_ply_path` / `init_artifacts_dir`. This module never writes into `outputs/`.
- **#9 / threading.** `save_warm_cache` reads through `gaussian_set.snapshot()` (the atomic, locked read-view) rather than poking `model.gauss_params` directly, so a save that overlaps an in-flight FF resize cannot tear. (See §8.)
- **Architecture principle #8 (versioned contract).** The fingerprint+schema_version gate is the literal implementation of "stamp a config fingerprint into the `.pt`; on mismatch raise a clear 'config changed, delete the cache' error." Drift → `WarmCacheLoadResult(success=False, drift=True, error=...)` BEFORE any model mutation.

## 8. Threading

- **Lives on the main/trainer thread only.** `save_warm_cache` is called at the static→dynamic transition (Phase-0b end, single-threaded) and at the dynamic-exit final snapshot (atexit/LIFO, after the FF bg slot is drained per RUNTIME_warmload_lifecycle H3). `load_warm_cache` is called once during `pipeline.__init__`, before any worker/render/viser thread is spawned (RUNTIME_warmload_lifecycle (b): the realloc is single-threaded by construction).
- **Lock discipline:** this module holds NO lock of its own. It reaches shared Gaussian state ONLY through `gaussian_set.snapshot()` (save) and `gaussian_set.reload_from_state_dict()` (load) — both of which take the one `_model_lock` internally. So even the (rare) exit-time save that could overlap a final FF insert is safe via the snapshot's brief locked detach. The module never indexes `model.gauss_params` / identity buffers directly.
- **May block on:** disk IO (torch.save/torch.load) and the brief `_model_lock` acquisition inside `snapshot()`/`reload_from_state_dict`. **May NOT block on:** the FF slot lock, SHM, subprocess IPC, or any network — none are touched.
- **Atomic write:** save uses tmp-file + `os.replace` so a concurrent `read_warm_cache_meta` / a crashed save never yields a half-written `.pt` (mirrors the publisher's atomic frame-write discipline; ARCHITECTURE_PRINCIPLES §6/§8).

## 9. Open questions for the human

1. **`pipeline_state` scope.** What exactly belongs in the persisted `pipeline_state` dict beyond the model? Candidates the current code sets *after* a load (`_sam3d_inserted`, `_static_converged_step`, `_step_offset=10_000`, `_filter_depth_at_ff`). `_step_offset` is a fixed constant (don't persist); the others may be re-derivable. Confirm the minimal set, or keep `pipeline_state={}` and let the dynamic pipeline set all of it post-load (the current post_fusion_cache.py behavior — "the caller is responsible").
2. **Fingerprint strictness + back-compat.** Legacy `post_fusion_state.pt` files predate any fingerprint. Should `load_warm_cache` (a) treat a *missing* fingerprint as drift (force re-train), (b) accept-with-warning when reading the legacy name, or (c) gate on `strict_fingerprint`? The signature exposes `strict_fingerprint=True`; pick the legacy default.
3. **Fingerprint ownership confirmation.** This spec puts fingerprint *validation* in `static_persist` and fingerprint *computation* in `config.config_fingerprint` (matching config.md §2b and gaussian_set.md Open-Q6's "leaning the latter"). Confirm `gaussian_set.reload_from_state_dict` stays fingerprint-agnostic (pure resize), and this module is the sole gate.
4. **What the fingerprint must cover.** Minimum to prevent a silent bad load: `sh_degree` (→ features_rest width), background color, camera-optimizer mode, and the SH-rest dim. Is the existing `config_fingerprint(cfg)` already inclusive of these, or does `static_persist` need a narrower "tensor-shape-affecting subset" fingerprint distinct from the full config hash (so a benign tuning-knob change doesn't needlessly invalidate a valid cache)?
5. **Seed PLY / init-artifacts read API.** This spec only centralizes the *paths*. Do any NEW callers need a `load_seed_ply()` / `load_init_artifacts()` reader here, or do the fusion/datamanager modules keep their own Open3D/JSON readers and just consume the path constants? (Leaning: paths only; readers stay where the data is consumed.)
6. **Drift recovery automation.** On `drift=True`, should the module ever auto-delete the stale `.pt` and signal "re-run static", or strictly leave deletion to the human (the message says "delete the cache")? Auto-delete is convenient but destroys a possibly-recoverable artifact; the spec assumes message-only.
