# scene_model.py — render/train behavior over GaussianSet (WRAP)

> Added 2026-06-18. Resolves the blocking gap D2/#4. **WRAP, confirmed:** `GaussianSet`
> owns the tensors (pure, lockable state); `SceneModel` is the thin adapter that *renders
> and trains* those tensors via nerfstudio's `SplatfactoModel` — composition, not
> inheritance. This is the one module that touches nerfstudio's model internals, so every
> other module is shielded from them.

## 1. RESPONSIBILITY
Provide rendering + (static-phase) training **on the tensors owned by `GaussianSet`**, by
holding a nerfstudio `SplatfactoModel` whose `gauss_params` **are** GaussianSet's tensors
(same Python objects — no copy). When surgery replaces tensors, rebind. Nothing else in
the pipeline imports `SplatfactoModel`.

## 2. PUBLIC INTERFACE
```python
class SceneModel:
    def __init__(self, gaussians: GaussianSet, cfg: Config, device): ...
        # builds an inner SplatfactoModel; binds its gauss_params to `gaussians`' tensors;
        # sets background = Gazebo sky (Inv #6); means LR policy per phase (Inv #1/#4).

    # --- render (both phases; dynamic is render-only) ---
    def render(self, camera) -> tuple[Tensor, Tensor, Tensor]: ...
        # -> (rgb, depth, alpha) via gsplat. MUST be called under render_lock.
    def render_object_mask(self, camera, instance_id: int) -> Tensor: ...
        # silhouette of one tracked instance (subset render). Under render_lock.

    # --- binding (called by GaussianSet's locked surgery) ---
    def rebind(self) -> None: ...
        # re-point the inner model's gauss_params at GaussianSet's (possibly new) tensors
        # AND refresh optimizer state to match. The thin binding layer WRAP costs;
        # GaussianSet.subset/grow/insert call this inside the lock.

    # --- static-phase training only ---
    def param_groups(self) -> dict: ...               # for the trainer's optimizers
    def get_loss_dict(self, outputs, batch) -> dict: ...# L1+SSIM; honors mask_provider
    def set_mask_provider(self, fn) -> None: ...        # static loss-mask (#15: callable, not monkeypatch)
    def enforce_phase_lr(self) -> None: ...             # means LR=0 (Inv #1) / all-zero dynamic (Inv #4)

    # --- thread safety ---
    def attach_render_lock(self, lock: RLock) -> None: ...
        # the ONE _model_lock; render()/render_object_mask hold it; FF/tracker share it.

    @property
    def device(self): ...
    @property
    def dtype(self): ...
```

## 3. DEPENDS ON
`gaussian_set` (the tensors it binds + renders), `config` (sh_degree, background, cam-opt mode, phase). External: nerfstudio `SplatfactoModel` + `gsplat` (the ONLY module allowed to import them).

## 4. CONSUMES / PRODUCES
Consumes: a `GaussianSet`, a `Cameras`, (static) a GT batch + mask_provider.
Produces: rendered `(rgb, depth, alpha)`; (static) a loss dict. Mutates **nothing** in GaussianSet directly — surgery goes through GaussianSet; SceneModel only *rebinds* to it.

## 5. SOURCE MOVED IN (current → here)
| Current | Becomes |
|---|---|
| `dynamic_gs_model.py` / `static_gs_model.py` **`SplatfactoModel` subclassing + populate_modules + get_outputs** | the inner wrapped `SplatfactoModel` + `render()`. The subclass IS-A relationship is dropped; the render path is kept. |
| `DynamicGSModel.render_object_mask` (+ the every-Nth-tick mask cache) | `render_object_mask()`. |
| `*_model._mask_means_grad` / means-LR=0 / `_ZERO_LR_OPTIMIZERS` enforcement | `enforce_phase_lr()`. |
| `*_model.populate_modules` background `(0.86,0.92,1.0)` | set in `__init__`. |
| optimizer-refresh tail of the current delete/insert surgery | `rebind()` (called BY GaussianSet surgery, not duplicated). |
| `attach_render_lock` hook | `attach_render_lock()`. |
| `StaticGSModel.get_loss_dict` depth/scene/gripper mask AND | `get_loss_dict()` + `set_mask_provider()` (#15). |

## 6. DROPPED
- **All nerfstudio-VIEWER integration** baked into the current models — `ViewerDropdown`, `viewer_object_selector`, `_viewer_keep_mask`, `_refresh_viewer_object_options`, and the "Nerfstudio viewer render thread" coupling. Reason: Invariant #9 — you use viser-direct (client-side WebGL), NOT the NS viewer. This was the only thing that genuinely tied the model to *being* a SplatfactoModel; it's vestigial. (git: present since v0 `0a23676`, never a reasoned choice.)
- `initialize_object_from_sam3d` (238 lines, no caller — 00_DEAD_CODE).
- `prepare_dynamic_update` + the ESAM/optim-mask/combine_object_masks/depth-score chain it transitively pulls (dead — 00_DEAD_CODE).
- `enable_dynamic_mean_optimization` toggle (dead scene-opt era).

## 7. INVARIANTS PRESERVED
- **#1** static means LR = 0 (`enforce_phase_lr`, static phase). **#4** dynamic all-LR = 0 (same, dynamic phase). **#6** background = Gazebo sky (`__init__`). **#9** no NS viewer — render is called only by viser-direct's `render_fn` + the CDN; this module exposes `render`, it does NOT start a viewer.
- **#8** SceneModel NEVER resizes tensors or touches the 4 identity buffers — only `rebind()`s to whatever GaussianSet currently holds. All length-changing surgery is GaussianSet's, under the lock.

## 8. THREADING
`render()` / `render_object_mask()` run on whatever thread calls them (tracker main for the object-mask + CDN; FF bg for its CDN render) and **MUST hold `_model_lock`** — they read the live tensors. `rebind()` runs *inside* GaussianSet's already-held surgery lock (no re-entrancy issue: `_model_lock` is an RLock). SceneModel holds no lock of its own; it borrows the shared one.

## 9. KEY SUBTLETY (the WRAP binding contract)
The inner `SplatfactoModel.gauss_params[k]` must be the **same tensor object** as `GaussianSet`'s `k` — so in-place mutation (the per-tick `means[mask] = …`, your supervisor's perf point) is preserved with zero copy. The ONLY time they can diverge is when surgery *replaces* a tensor (subset/grow create new tensors); `rebind()` re-points the model + optimizer in the same locked critical section. So WRAP's whole cost is one `rebind()` call already adjacent to the optimizer-refresh the surgery does anyway.

## 10. OPEN QUESTIONS
- Does the static trainer drive `SceneModel` through nerfstudio's `Trainer` (so `param_groups`/`get_loss_dict` must match nerfstudio's Model API), or does `static_fit` call SceneModel directly? (ties to #13/#14 — trainer glue.) Recommend: SceneModel exposes a nerfstudio-compatible facade so the vanilla Trainer works without a custom Trainer subclass.
