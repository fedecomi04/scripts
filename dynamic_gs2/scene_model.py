"""scene_model.py — render/train behavior over GaussianSet (WRAP, not BE).

The ONE module that touches nerfstudio's SplatfactoModel. It holds an inner
SplatfactoModel whose gauss_params ARE GaussianSet's tensors (same Python objects,
no copy); GaussianSet does all count/identity surgery and then calls rebind() to
re-point the optimizer + re-assert the phase LR policy. Every other module is
shielded from nerfstudio model internals. (rewrite_spec/scene_model.md, D2.)
"""
from __future__ import annotations

import threading
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from nerfstudio.data.scene_box import SceneBox
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig

# Try to disable densification (static phase mutates count surgically, not via
# Splatfacto's clone/split/prune). NoRefineStrategy lives in the old package; we
# re-import it if present, else fall back to a no-op post-backward.
try:
    from dynamic_gs.utils.no_refine_strategy import NoRefineStrategy  # type: ignore
    _HAVE_NOREFINE = True
except Exception:                                                      # pragma: no cover
    NoRefineStrategy = None
    _HAVE_NOREFINE = False

_GAZEBO_SKY = (0.86, 0.92, 1.0)          # Invariant #6
_PARAM_NAMES = ("means", "features_dc", "features_rest", "scales", "quats", "opacities")


class SceneModel:
    """Thin render/train adapter around a wrapped SplatfactoModel."""

    def __init__(self, cfg, device, *, seed_xyz: torch.Tensor, seed_rgb: torch.Tensor,
                 phase: str = "dynamic", num_train_data: int = 1,
                 aabb_scale: float = 4.0):
        """cfg = RuntimeConfig (reads cfg.sh_degree via static_train). phase in {static,dynamic}.
        seed_rgb is uint8 [0,255] or float [0,1] -> normalized to [0,255] for Splatfacto."""
        self.cfg = cfg
        self.phase = phase
        self._device = torch.device(device)
        self._render_lock: Optional[threading.RLock] = None
        self._mask_provider: Optional[Callable] = None
        self._count_provider: Optional[Callable] = None   # () -> live row count (free-list); None = render full tensor
        self._hidden_idx: Optional[torch.Tensor] = None    # rows to VIRTUALLY hide from the render (deferred cull)
        self._means_grad_handle = None

        sh_degree = int(getattr(getattr(cfg, "static_train", cfg), "sh_degree", 3))
        sxyz = torch.as_tensor(seed_xyz, dtype=torch.float32)
        srgb = torch.as_tensor(seed_rgb, dtype=torch.float32)
        if srgb.max() <= 1.0 + 1e-6:
            srgb = srgb * 255.0
        ext = sxyz.abs().max().item() if sxyz.numel() else 1.0
        a = max(ext * 1.0, 1.0) * aabb_scale
        scene_box = SceneBox(aabb=torch.tensor([[-a, -a, -a], [a, a, a]], dtype=torch.float32))

        model_cfg = SplatfactoModelConfig(sh_degree=sh_degree, random_init=False)
        self.model: SplatfactoModel = model_cfg.setup(
            scene_box=scene_box, num_train_data=num_train_data,
            metadata={"seed_points": (sxyz, srgb)},
            seed_points=(sxyz, srgb), device=self._device,
        )
        self.model.to(self._device)
        self.model.set_background(torch.tensor(_GAZEBO_SKY, dtype=torch.float32, device=self._device))
        # render-time: activate all SH bands; surgical count changes only.
        self.model.step = 30000
        self.model.crop_box = None
        if _HAVE_NOREFINE:
            self.model.strategy = NoRefineStrategy()
            self.model.strategy_state = self.model.strategy.initialize_state(scene_scale=1.0)
        self._bind_means_grad_hook()
        self.enforce_phase_lr()

    # ---- the WRAP binding: gauss_params ARE the SSOT tensors ----
    @property
    def gauss_params(self) -> Dict[str, torch.Tensor]:
        return self.model.gauss_params

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self.model.gauss_params["means"].dtype

    def attach_render_lock(self, lock: "threading.RLock") -> None:
        self._render_lock = lock

    def set_count_provider(self, fn: Optional[Callable]) -> None:
        """Free-list: fn() -> live row count. When set, the render feeds gsplat a [:count] VIEW
        of each gauss_param so the over-allocated dead rows [count:] never rasterize. None (default)
        = render the full tensor (static phase / no free-list)."""
        self._count_provider = fn

    def set_hidden_indices(self, idx: Optional[torch.Tensor]) -> None:
        """Deferred-cull (FF Option A): rows whose OPACITY the render zeroes so they don't show,
        WITHOUT deleting them from the SSOT. Lets the re-CDN render see the scene as-if-culled while
        the real cull is held back to commit atomically with the FF insert (no cull-without-insert
        flicker). Cheap: clones only the (N,1) opacity tensor + zeroes k rows. None = nothing hidden.
        MUST be set/cleared under the render lock around the render that should honor it."""
        self._hidden_idx = idx

    # ---- background-color helper kept for callers that read it ----
    def _get_background_color(self):
        return self.model.background_color.to(self._device)

    # ---- means-grad hook + phase LR (Inv #1 static means LR=0 / #4 dynamic all=0) ----
    def _bind_means_grad_hook(self) -> None:
        """Dynamic phase: hard-zero the means gradient so no optimizer step can move
        seed positions even if a stray backward fires. Static phase relies on LR=0."""
        if self._means_grad_handle is not None:
            try:
                self._means_grad_handle.remove()
            except Exception:
                pass
            self._means_grad_handle = None
        if self.phase == "dynamic":
            means = self.model.gauss_params["means"]
            if means.requires_grad:
                self._means_grad_handle = means.register_hook(lambda g: torch.zeros_like(g))

    def enforce_phase_lr(self) -> None:
        opt = getattr(self.model, "optimizers", None)
        if opt is None:
            return
        groups = ("means",) if self.phase == "static" else _PARAM_NAMES
        for name in groups:
            o = opt.optimizers.get(name) if hasattr(opt, "optimizers") else opt.get(name)
            if o is not None:
                for pg in o.param_groups:
                    pg["lr"] = 0.0

    def rebind(self) -> None:
        """Re-point the inner model's optimizers at the (possibly new) gauss_params
        Parameter objects + re-bind the means-grad hook + re-assert phase LR. Called
        BY GaussianSet surgery, inside the shared lock."""
        opt = getattr(self.model, "optimizers", None)
        wrapper = getattr(self.model, "_optimizers_wrapper", None)
        if opt is not None and hasattr(opt, "optimizers"):
            for name, o in opt.optimizers.items():
                if name in self.model.gauss_params:
                    o.param_groups[0]["params"] = [self.model.gauss_params[name]]
                    o.state.clear()
        if wrapper is not None and hasattr(wrapper, "parameters"):
            for name in self.model.gauss_params:
                if name in wrapper.parameters:
                    wrapper.parameters[name] = [self.model.gauss_params[name]]
        self._bind_means_grad_hook()
        self.enforce_phase_lr()

    # ---- render ----
    def _render_param_override(self, restrict_idx: Optional[torch.Tensor] = None):
        """Return an nn.ParameterDict to render IN PLACE OF the full gauss_params, or None to
        render the full tensor. Combines three cheap, copy-free-ish transforms:
          - free-list [:count] slice (VIEWS, no copy) so dead capacity rows never rasterize;
          - deferred-cull virtual-hide: clone ONLY the (N,1) opacity + set hidden rows to -30
            (sigmoid->0) so they vanish from the image without being deleted from the SSOT;
          - restrict_idx: GATHER only those (live) rows so the rasterizer projects+sorts ONLY that
            subset (e.g. gaussians inside the re-CDN's first-region bbox — saves the per-gaussian
            projection on everything outside the changed region). A small copy of the gathered subset.
        gauss_params is an nn.Module child, so the override must be an nn.ParameterDict."""
        full = self.model.gauss_params
        phys = full["means"].shape[0]
        c = int(self._count_provider()) if self._count_provider is not None else phys
        hide = self._hidden_idx
        if c >= phys and (hide is None or hide.numel() == 0) and restrict_idx is None:
            return None                                  # nothing to slice, hide, or restrict
        out = {k: full[k][:c] for k in _PARAM_NAMES}     # [:c] views (c==phys -> full view)
        if hide is not None and hide.numel() > 0:
            opa = out["opacities"].clone()               # clone ONLY opacity (N,1) — tiny
            h = hide[(hide >= 0) & (hide < opa.shape[0])]
            opa[h] = -30.0                               # sigmoid(-30) ~ 0 -> hidden, not deleted
            out["opacities"] = opa
        if restrict_idx is not None:                     # gather the in-region subset (after the [:c] slice)
            ri = restrict_idx[(restrict_idx >= 0) & (restrict_idx < c)].to(out["means"].device)
            out = {k: out[k][ri] for k in _PARAM_NAMES}
        return nn.ParameterDict({k: nn.Parameter(out[k], requires_grad=False) for k in _PARAM_NAMES})

    def means_in_bbox_idx(self, camera, bbox, pad: int = 0) -> torch.Tensor:
        """Indices (into the LIVE [:count] rows) of gaussians whose MEAN projects into the pixel
        bbox=(left,top,right,bottom). Used to restrict the re-CDN render to the changed region.
        Cheap: one project of means (no rasterize). Returns a long tensor on the model device."""
        full = self.model.gauss_params
        c = int(self._count_provider()) if self._count_provider is not None else full["means"].shape[0]
        means = full["means"][:c]
        cam = camera.to(self._device)
        c2w = cam.camera_to_worlds
        c2w = c2w[0] if c2w.dim() == 3 else c2w
        R, t = c2w[:3, :3].to(means.dtype), c2w[:3, 3].to(means.dtype)
        def _s(x):
            return float(x.reshape(-1)[0]) if torch.is_tensor(x) else float(x)
        fx, fy, cx, cy = _s(cam.fx), _s(cam.fy), _s(cam.cx), _s(cam.cy)
        mc = (means - t[None, :]) @ R                    # world->camera (OpenGL c2w)
        z = -mc[:, 2]
        front = z > 1e-6
        zz = torch.where(front, z, torch.ones_like(z))
        u = fx * (mc[:, 0] / zz) + cx
        v = fy * (-mc[:, 1] / zz) + cy
        l, tp, r, b = bbox
        inb = front & (u >= l - pad) & (u < r + pad) & (v >= tp - pad) & (v < b + pad)
        return torch.nonzero(inb, as_tuple=False).flatten()

    def _render(self, camera, restrict_idx: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        cam = camera.to(self._device)
        if cam.camera_to_worlds.dim() == 2:
            cam = cam.reshape((1,))
        if restrict_idx is not None and restrict_idx.numel() == 0:
            return self._empty_outputs(cam)              # no gaussians in the region -> all-background
        self.model.eval()
        override = self._render_param_override(restrict_idx)
        saved = None
        if override is not None:
            saved = self.model.gauss_params
            self.model.gauss_params = override           # [:count] views + opacity-hidden + restrict-gather
        try:
            with torch.no_grad():
                return self.model.get_outputs(cam)
        finally:
            if saved is not None:
                self.model.gauss_params = saved

    def _empty_outputs(self, camera) -> Dict[str, torch.Tensor]:
        """Background-filled render with ZERO gaussians (rgb=bg, alpha=0, depth=0) — used when a
        region-restricted render selects no gaussians, so gsplat is never called with an empty tensor."""
        H = int(camera.height.reshape(-1)[0]); W = int(camera.width.reshape(-1)[0])
        bg = self._get_background_color().reshape(1, 1, 3)
        return {"rgb": bg.expand(H, W, 3).clone(),
                "depth": torch.zeros(H, W, 1, device=self._device),
                "accumulation": torch.zeros(H, W, 1, device=self._device)}

    def render(self, camera, restrict_idx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> (rgb HxWx3, depth HxWx1, alpha HxWx1). MUST be called under render_lock.
        restrict_idx (optional): render ONLY those live-row gaussians (re-CDN region restriction)."""
        out = self._render(camera, restrict_idx)
        rgb = out["rgb"]
        depth = out.get("depth")
        alpha = out.get("accumulation", out.get("alpha"))
        return rgb, depth, alpha

    def render_object_mask(self, camera, instance_mask: torch.Tensor,
                           alpha_thr: float = 0.5) -> torch.Tensor:
        """Silhouette (bool HxW) of the gaussians selected by `instance_mask` (bool, len==N).
        Renders only the subset by temporarily swapping gauss_params, reading alpha.
        MUST be called under render_lock. Returns all-False if the subset is empty."""
        m = torch.as_tensor(instance_mask, device=self._device).bool().flatten()
        H = int(camera.height.item()) if hasattr(camera.height, "item") else int(camera.height)
        W = int(camera.width.item()) if hasattr(camera.width, "item") else int(camera.width)
        if m.sum() == 0:
            return torch.zeros((H, W), dtype=torch.bool, device=self._device)
        saved = self.model.gauss_params
        # Free-list: the mask is len==count but saved[k] is the capacity tensor — slice [:c] first.
        c = int(self._count_provider()) if self._count_provider is not None else saved["means"].shape[0]
        try:
            self.model.gauss_params = nn.ParameterDict({
                k: nn.Parameter(saved[k][:c][m].detach(), requires_grad=False) for k in _PARAM_NAMES
            }).to(self._device)
            out = self._render(camera)
            alpha = out.get("accumulation", out.get("alpha"))
        finally:
            self.model.gauss_params = saved
        return (alpha[..., 0] > alpha_thr) if alpha is not None else \
            torch.zeros((H, W), dtype=torch.bool, device=self._device)

    # ---- static-phase training facade ----
    def param_groups(self) -> Dict[str, list]:
        return self.model.get_param_groups()

    def set_mask_provider(self, fn: Callable) -> None:
        """fn(batch) -> bool/float mask tensor (or None). ANDed into the loss mask (#15)."""
        self._mask_provider = fn

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> dict:
        if self._mask_provider is not None:
            extra = self._mask_provider(batch)
            if extra is not None:
                e = extra.to(self._device).float()
                if e.ndim == 2:
                    e = e[..., None]
                existing = batch.get("mask")
                combined = e if existing is None else existing.to(self._device).float() * e
                batch = {**batch, "mask": combined}
        return self.model.get_loss_dict(outputs, batch, metrics_dict)
