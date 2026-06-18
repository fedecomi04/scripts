# `gaussian_set.py` — module spec (layer: core)

## 1. Responsibility

The single source of truth for live scene state — the 6 `gauss_params` plus the 4 identity buffers — owned by one object, mutated ONLY through a small locked surgery API (subset / grow / insert / cull / rigid-write + optimizer refresh, with a length invariant asserted on every mutation under one RLock), and read ONLY through an immutable atomic `snapshot()`; this kills the H-CROP unlocked-read race, the static↔dynamic surgery drift, and the four-buffer foot-gun in one place.

## 2. Public interface (the contract other modules call)

```python
PARAM_NAMES: tuple[str, ...] = ("means", "features_dc", "features_rest", "scales", "quats", "opacities")
# Fixed processing order for every param loop. Never reorder.

IDENTITY_BUFFER_SPECS: tuple[tuple[str, torch.dtype], ...] = (
    ("object_flags", torch.float32),
    ("sam3d_init_target_flags", torch.float32),
    ("object_instance_ids", torch.long),
    ("inserted_flags", torch.float32),
)
# The 4 persistent identity buffers, in fixed order, with their dtypes.
# object_instance_ids is the ONLY long buffer. (current_active_mask is intentionally NOT here — see DROPPED.)


@dataclass(frozen=True)
class GaussianSnapshot:
    """Immutable, detached read-view handed to reader threads (viser, tracker pose-read, diag)."""
    params: dict[str, torch.Tensor]      # 6 detached tensors, PARAM_NAMES order, shared-storage read-only
    buffers: dict[str, torch.Tensor]     # 4 detached identity buffers
    num_points: int
    version: int                          # monotonic; bumped on every mutating call
    def __len__(self) -> int: ...         # == num_points


class GaussianSet:
    """Owns the 6 params + 4 buffers + their optimizer wiring; the ONE chokepoint for all surgery."""

    def __init__(self, model, lock: "threading.RLock") -> None:
        """Bind to the live splat `model` (provides gauss_params, optimizers, _optimizers_wrapper,
        device/dtype, the means-grad hook + phase-apply callbacks) and the shared `_model_lock`.
        Allocates/validates the 4 identity buffers at the model's current num_points."""

    # ---- reads ----
    @property
    def num_points(self) -> int: ...                          # current Gaussian count (lock-free read of one int)
    @property
    def device(self) -> torch.device: ...
    def snapshot(self) -> GaussianSnapshot: ...               # atomic detached read-view; reader threads use ONLY this
    def version(self) -> int: ...                             # monotonic mutation counter

    # ---- surgery (all acquire the lock internally; all assert the length invariant at exit) ----
    def cull(self, indices: torch.Tensor, *, protect_mask: torch.Tensor | None = None) -> int:
        """Delete Gaussians at `indices` (deduped, bounds-clipped). `protect_mask` (bool, len==num_points)
        forces-keep — callers pass `object_instance_ids == d0_id` so the tracked object is never dropped.
        Slices all 6 params + all 4 buffers in lockstep, refreshes optimizers, bumps version. Returns n deleted."""

    def insert(self, tensors: GaussTensors, *, object_flag: float, instance_id: int) -> torch.Tensor:
        """Append pre-built Gaussians (the FF/Phase-0b decoders produce `tensors`; this module never decodes).
        Concats all 6 params (requires_grad preserved across the no_grad resize), grows the 4 buffers,
        writes object_flags=object_flag / object_instance_ids=instance_id / inserted_flags=1 on the new tail,
        refreshes optimizers, bumps version. Returns the inserted index range (arange[old:new])."""

    def write_object_pose(self, means_subset: torch.Tensor, quats_subset: torch.Tensor,
                          object_mask: torch.Tensor) -> int:
        """In-place value write (NO count change) of means/quats for `object_mask` rows — the tracker rigid
        transform. Asserts mask row-count == subset row-count, finite. Bumps version. Returns rows written."""

    def set_object_flags(self, mask: torch.Tensor, value: float) -> None:
        """D0 selection: write object_flags[mask]=value in place (no count change). Bumps version."""

    def write_instance_ids(self, mask: torch.Tensor, instance_id: int) -> None:
        """Phase-0b membership: object_instance_ids[mask]=instance_id in place. Bumps version."""

    # ---- escape hatch for framework load (warm-cache) ----
    def reload_from_state_dict(self, state_dict: dict, num_points: int) -> None:
        """Reallocate the 6 params at `num_points`, rebuild + zero-fill the 4 buffers, re-bind the means-grad
        hook, refresh optimizers. Used by the .pt warm-cache loader. Asserts invariant at exit, bumps version."""


@dataclass
class GaussTensors:
    """The decoder→insert contract: 6 tensors, already in scene frame / log-scale / logit-opacity.
    Lives here so decoders and Phase-0b build the same shape. `validate()` checks dims + SH-rest width."""
    means: torch.Tensor; features_dc: torch.Tensor; features_rest: torch.Tensor
    scales: torch.Tensor; quats: torch.Tensor; opacities: torch.Tensor
    def validate(self, sh_rest_dim: int) -> "GaussTensors": ...   # coerce features_rest width, unsqueeze opacities


# Free helpers (the deduped surgery primitives — internal, but exported for the two insert builders + tests)
def build_default_gauss_tensors(new_xyz, new_rgb, *, sh_degree: int, sh_rest_dim: int,
                                device, dtype) -> GaussTensors: ...
    # kNN-spacing log-scale seed, RGB2SH/logit features_dc, zero features_rest, identity quats, logit(0.1) opacity.
def activated_opacity(opacity_logits): ...                     # sigmoid(logit); torch/numpy backend-dispatched
def low_opacity_indices(opacity_logits, thr: float) -> torch.Tensor: ...   # purge-set helper
def uniform_shrink_log_scales(log_scales, max_scale_m, *, min_scale_m=0.0): ...  # (log_scales, keep_mask)
```

## 3. Depends on (NEW modules only)

- **`scene_model.py`** (the splat-model module of the rewrite) — `GaussianSet` is constructed around it and calls back into it for: `gauss_params` dict, `optimizers` / `_optimizers_wrapper`, `_mask_means_grad`, `_apply_phase_trainability`, `_apply_phase_optimizers`, `device`/`dtype`, `config.sh_degree`. `GaussianSet` owns *count/identity surgery*; `scene_model` owns *render + phase/LR policy*. (If the rewrite folds the model into the set, this dependency disappears; keep the seam thin.)
- **the shared `_model_lock`** — created by `pipeline.py` and passed into the ctor. Not a module, a handle. `GaussianSet` does not create it (the viser server and tracker must share the same instance).
- No other NEW-module dependency. Decoders (`feedforward`/`anysplat`/`rgbd`), Phase-0b fusion, and the warm-cache loader are *callers*, not dependencies.

## 4. Consumes / produces

**Consumes (inputs at the boundary):**
- `GaussTensors` (6 tensors, scene-frame, log-scale, logit-opacity) from FF decoders and Phase-0b — the insert contract.
- `indices` / `protect_mask` (long / bool, len==num_points) for `cull`.
- `means_subset` / `quats_subset` + `object_mask` for the tracker rigid write.
- a `state_dict` + `num_points` from the `.pt` warm-cache for `reload_from_state_dict`.

**Produces (outputs):**
- `GaussianSnapshot` (immutable detached params+buffers+num_points+version) — the ONLY read path for any non-owning thread.
- the inserted index range (long arange) from `insert`.
- mutation `version` counter for cheap change-detection by readers.
- the underlying `model.gauss_params` + 4 registered buffers stay the framework-visible state (so nerfstudio save/load + the renderer keep working) — but every *count-or-identity* mutation funnels through this class.

## 5. Source moved in (current `file:symbol` → what it becomes)

| Current | Becomes |
|---|---|
| `dynamic_gs_model.py:delete_gaussian_indices` + `static_gs_model.py:delete_gaussian_indices` (twin copies, DUP Pattern A) | one `GaussianSet.cull` (+ `protect_mask` param folds in the "never drop object_flags==1" TODO) |
| `dynamic_gs_model.py:insert_inpaint_gaussians` + `insert_object_gaussians` + `static_gs_model.py:insert_object_gaussians` (DUP Pattern B) | one `GaussianSet.insert(GaussTensors, object_flag=, instance_id=)`; the FF tail (id=999) and Phase-0b tail (id=k) are just different args |
| `dynamic_gs_model.py:_build_new_gaussian_tensors` + `static_gs_model.py:_build_new_gaussian_tensors` (DUP Pattern C, byte-identical) | `build_default_gauss_tensors` free helper |
| `_refresh_gaussian_optimizers` (both models) + `_resize_dynamic_buffers` (both, DUP Pattern D) | internal `GaussianSet._refresh_optimizers` + `_resize_buffers` driven by `IDENTITY_BUFFER_SPECS` (data, not duplicated code); the dynamic-only tail (means-grad re-hook + phase apply) called back into `scene_model` |
| `dynamic_gs_model.py:apply_rigid_object_transform_from_reference` (the value-write half, `model:1006-1007`) | `GaussianSet.write_object_pose`. The reference-pose snapshot + R/t→quat math stays in `scene_model`/tracker; only the locked in-place write moves here |
| `object_flags.copy_` D0 write (`base:1165`) | `GaussianSet.set_object_flags` |
| `_propagate_instance_membership` instance write (kept for roadmap) | `GaussianSet.write_instance_ids` (the *write*; the KNN-propagation logic stays in Phase-0b) |
| `static_gs_model.py:load_state_dict` buffer-resize + `persistence/post_fusion_cache.py:120-143` reload | `GaussianSet.reload_from_state_dict` (single owner of the warm-restart resize + hook re-bind + invariant assert) |
| `static_gs_pipeline.py:_finalize_static_training` opacity purge (sigmoid threshold, DUP Pattern E) | `low_opacity_indices` + `cull` (caller composes; logic named once) |
| `static_gs_model.py:_shrink_oversized_scales_cb` log-scale shrink + `anysplat_decode.py:791-803` (DUP Pattern F) | `uniform_shrink_log_scales` free helper (callers stay where they are; the algorithm is shared) |
| the four `register_buffer(...)` blocks in both `populate_modules` | one `IDENTITY_BUFFER_SPECS`-driven allocation inside `GaussianSet.__init__` (still registered on the model so the framework sees them) |

## 6. Dropped (NOT carried, with reason + audit ref)

- **`current_active_mask` (5th buffer)** — non-persistent, only resized/sliced to keep length==num_points; its writers (`prepare_dynamic_update`, `refresh_dynamic_state_after_insertion`) are all dead. Drop it from the lockstep set entirely. *Audit: DUP_tensor_surgery §A note; dynamic_model.md §2 (`refresh_dynamic_state_after_insertion`, `prepare_dynamic_update` = dead, high conf); RUNTIME_state_mutation_map §H9.*
- **`change_mask_image` buffer** — only read by the dynamic `get_loss_dict` branch, which is loss-bypassed in live; not identity state. Stays on the model if the recorded-loss path needs it, but it is NOT part of `GaussianSet`. *Audit: dynamic_model.md §2 (`_get_optim_mask`/`_set_optim_mask` live-dead).*
- **`self.info` cross-thread reads** for cull-target selection (`extract_projected_centers_and_radii`) — the SAM3D-init slab + `_get_existing_object_subset` paths. Not moved in; the AnySplat default proves cull works via direct projection without the shared mutable `self.info`. *Audit: RUNTIME_target_architecture "DELETE" list; RUNTIME_hazards self.info HIGH; dynamic_model.md §3 self.info.*
- **`reset_means_optimizer` parameter** — never read; both call sites pass `True`, body clears all state regardless. The internal refresh takes no such arg. *Audit: static_model_pipeline.md §4 "Unused parameter".*
- **The legacy `apply_rigid_object_transform` (absolute, `model:922-939`)** — no live caller; superseded by `_from_reference`. Only the `_from_reference` value-write becomes `write_object_pose`. *Audit: dynamic_model.md §2 (high); RUNTIME_hazards H4.*
- **The whole SAM3D-init helper subtree on the DynamicGSModel side** (`_get_existing_object_subset`, `_get_object_mask_slab_indices`, `_build_persistent_object_membership`, `initialize_object_from_sam3d`, `insert_object_gaussians` dynamic copy) — unreachable on the dynamic class (Phase-0b runs on the static model's twins). The shared *surgery* primitives are unified here; the SAM3D-specific query/registration logic stays in Phase-0b (`fusion/phase0.py` + the static model), not in `GaussianSet`. *Audit: dynamic_model.md §2 "Dead-in-dynamic-model", §4.*
- **FF-video machinery / oneshot FF path** — not state surgery; lives in the FF dispatcher, not here. `feedforward_video_out` has no writer. *Audit: CLAUDE.md viser notes (no mp4 writer); RUNTIME_target_architecture (single recurring FF path).*
- **Per-tick wasted subset renders** (`flagged_rgb`/`non_flagged_rgb`/`non_inserted_rgb`) — render concern, not surgery; not in scope. *Audit: dynamic_model.md §4 "Wasted per-tick compute".*

## 7. Invariants preserved (CLAUDE.md) + how

- **#1 (static means LR=0) / #4 (dynamic ALL gauss LRs=0):** `GaussianSet` never sets LRs. After any resize it calls back into `scene_model._refresh_optimizers` tail which re-binds the means-grad hook (`_mask_means_grad`) + re-applies `_apply_phase_optimizers`, so the lr=0 / zero-grad policy survives every insert/cull (today these are re-applied inside `_refresh_gaussian_optimizers`). The hook re-bind on the freshly-allocated `means` Parameter is mandatory — kept.
- **#8 (per-buffer phase ownership):** `GaussianSet` carries exactly the 4 identity buffers in `IDENTITY_BUFFER_SPECS`. `sam3d_init_target_flags` is allocated and sliced/grown in lockstep but has **no value-writer** — there is intentionally no public method that writes it (matches "never written at runtime; all-zeros is correct"). `object_instance_ids` written only via `insert`/`write_instance_ids` (Phase-0b/FF tail), `inserted_flags` only via `insert` tail, `object_flags` only via `insert` tail + `set_object_flags` (D0). The owners stay the same callers; `GaussianSet` just makes the lockstep impossible to forget (the exit assert `all params+buffers share shape[0]`).
- **#9 (viser-direct read-only):** readers (viser render thread) call `snapshot()` and never mutate — the snapshot is detached/frozen. The render itself still happens on `scene_model` under the same `_model_lock`; `GaussianSet` does not render.
- **#6 (background) / #2 (camera-opt off):** untouched — not state-surgery concerns; remain in `scene_model`.
- **Decoders produce tensors only:** enforced by the `insert(GaussTensors, ...)` signature — there is no path for a decoder to mutate the set except by handing over built tensors.

## 8. Threading

- **Owns the one `_model_lock` (RLock) for all mutation.** Every surgery method (`cull`/`insert`/`write_object_pose`/`set_object_flags`/`write_instance_ids`/`reload_from_state_dict`) acquires it internally with `with self._lock:` and asserts the length invariant before releasing. Callers no longer have to remember to wrap — this fixes H-CROP / H-PICK / train-eval unlocked accesses by making the lock non-optional.
- **Writer threads:** tracker/main thread (`write_object_pose`, `set_object_flags`) and the FF-bg single-in-flight thread (`insert`, `cull`). The RLock is re-entrant so a FF cull→insert sequence under an outer lock is fine. **Must NOT hold the lock across the ~270 ms AnySplat subprocess call** — the FF caller builds `GaussTensors` *outside* the lock and only calls `insert`/`cull` (each a short locked critical section); `GaussianSet` never blocks on a subprocess, network, or disk.
- **Reader threads:** viser render + any diag/tracker pose-read take `snapshot()`. `snapshot()` acquires the lock only briefly to detach + bundle references (O(1), no copy), so readers never see a torn resize. `num_points`/`version` are single-int reads (lock-free is acceptable; document as eventually-consistent).
- **May block on:** the `_model_lock` only (bounded — held only for the O(N) tensor cat/slice + optimizer re-point, the documented per-FF churn). **May NOT block on:** the FF slot lock, SHM, subprocess IPC, the publisher pose/joint lock — those are other modules' concerns.
- **Lock discipline rule restated:** there is exactly ONE implementation of param+buffer subset/grow, with a length-invariant assert, behind one lock (ARCHITECTURE_PRINCIPLES §1, §2). No second copy may exist.

## 9. Open questions for the human

1. **Model fold-in:** should `GaussianSet` *be* the splat model (owning `gauss_params` directly), or wrap the nerfstudio `SplatfactoModel` (which must keep `gauss_params` for its renderer + framework save/load)? The spec assumes wrap-and-callback to minimize the diff; folding would remove the `scene_model` dependency but couples surgery to nerfstudio's Parameter registration. Which?
2. **Snapshot sharing vs copy:** `snapshot()` shares detached storage (cheap, but a reader holding an old snapshot keeps freed tensors alive across an insert/cull — they're reallocated, so the old storage is GC'd when the snapshot drops). Is share-and-let-GC acceptable, or do you want an explicit `version`-guard ("snapshot stale → re-fetch") protocol for the viser thread?
3. **Static vs dynamic phase:** do we keep a single `GaussianSet` used by BOTH `static-gs` (Phase-0b inserts, no rigid write) and `dynamic-gs-live` (rigid + FF), with the phase-specific optimizer tail injected via `scene_model`? Or two thin subclasses? One class is the point of the rewrite (kills the static↔dynamic drift) — confirm.
4. **Bounded growth ownership:** the FF-growth cap (periodic purge, never drop `object_flags==1`) — does the *policy* (when/how-much to purge) live in the FF dispatcher calling `cull(low_opacity_indices(...), protect_mask=tracked)`, with `GaussianSet` only providing the primitives? The spec assumes yes (mechanism here, policy in the caller). Confirm.
5. **`protect_mask` semantics:** should `cull` *silently* drop protected indices from the delete set, or *raise* if a caller tries to delete a protected Gaussian? Silent-drop is safer for the FF purge; raise catches bugs. Pick.
6. **Warm-cache config fingerprint:** ARCHITECTURE_PRINCIPLES §8 wants the `.pt` config-tagged with a loud mismatch error. Is stamping/validating that fingerprint a `GaussianSet.reload_from_state_dict` responsibility, or the warm-cache module's (with `GaussianSet` only doing the resize)? Leaning the latter.
