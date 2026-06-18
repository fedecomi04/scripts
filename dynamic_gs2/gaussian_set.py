"""gaussian_set.py — the single source of truth for live scene state.

Owns the 6 gauss_params + 4 identity buffers. EVERY count-or-identity mutation
funnels through one locked surgery API (cull / insert / write_object_pose /
set_object_flags / write_instance_ids / reload_from_state_dict); EVERY non-owning
thread reads through an immutable atomic snapshot(). This is the chokepoint that
kills the H-CROP unlocked-read race, the static<->dynamic surgery drift, and the
four-buffer foot-gun (see rewrite_spec/gaussian_set.md).

WRAP seam (D2): GaussianSet does the tensor surgery, then calls model.rebind()
ONCE — scene_model owns optimizer refresh / means-grad re-hook / phase-LR policy
(it knows nerfstudio internals; GaussianSet does not). GaussianSet never sets LRs
and never renders.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Optional

import torch

# Fixed processing order for every param loop. NEVER reorder (optimizer groups,
# state_dict keys, and the inner SplatfactoModel all key off this).
PARAM_NAMES = ("means", "features_dc", "features_rest", "scales", "quats", "opacities")

# The 4 persistent identity buffers, fixed order + dtype. object_instance_ids is
# the ONLY long buffer. current_active_mask is intentionally absent (DROPPED — all
# its writers are dead; see spec §6).
IDENTITY_BUFFER_SPECS = (
    ("object_flags", torch.float32),
    ("sam3d_init_target_flags", torch.float32),
    ("object_instance_ids", torch.long),
    ("inserted_flags", torch.float32),
)
_BUFFER_NAMES = tuple(n for n, _ in IDENTITY_BUFFER_SPECS)

_RGB2SH_C0 = 0.28209479177387814   # SH band-0 constant


# ----------------------------------------------------------------- read view
@dataclass(frozen=True)
class GaussianSnapshot:
    """Immutable detached read-view for reader threads (viser version-check, tracker
    crop-bbox, FF ICP target, diag).

    COUNT and `version` are frozen atomically (captured under the lock). Param tensor
    STORAGE is shared-detached (O(1), no copy) — surgery REPLACES tensors so an old
    snapshot keeps its old complete tensors (safe), while an in-place value write
    (write_object_pose / set_object_flags) by the owner is eventually visible on the
    shared storage. That is acceptable by design: the authoritative render holds the
    lock, and in-place writes touch only the tracked object's rows. (spec open-Q#2.)
    """
    params: Dict[str, torch.Tensor]
    buffers: Dict[str, torch.Tensor]
    num_points: int
    version: int

    def __len__(self) -> int:
        return self.num_points


# ----------------------------------------------------------- insert contract
@dataclass
class GaussTensors:
    """Decoder->insert contract: 6 tensors already in scene frame / log-scale /
    logit-opacity. Defined here so FF decoders and Phase-0b build the same shape."""
    means: torch.Tensor
    features_dc: torch.Tensor
    features_rest: torch.Tensor
    scales: torch.Tensor
    quats: torch.Tensor
    opacities: torch.Tensor

    def validate(self, sh_rest_dim: int) -> "GaussTensors":
        n = self.means.shape[0]
        assert self.means.shape[1:] == (3,), f"means {self.means.shape}"
        assert self.scales.shape == (n, 3), f"scales {self.scales.shape}"
        assert self.quats.shape == (n, 4), f"quats {self.quats.shape}"
        # features_dc: (n,3) canonical — matches nerfstudio SplatfactoModel + the old insert path
        if self.features_dc.dim() == 3 and self.features_dc.shape[1] == 1:
            self.features_dc = self.features_dc.reshape(n, 3)
        assert self.features_dc.shape == (n, 3), f"features_dc {self.features_dc.shape}"
        # features_rest: (n, sh_rest_dim, 3); coerce empty/short width
        fr = self.features_rest
        if fr.numel() == 0 or fr.shape[1] != sh_rest_dim:
            fr = torch.zeros((n, sh_rest_dim, 3), dtype=self.means.dtype, device=self.means.device)
        self.features_rest = fr
        # opacities: (n,1)
        if self.opacities.dim() == 1:
            self.opacities = self.opacities.unsqueeze(1)
        assert self.opacities.shape == (n, 1), f"opacities {self.opacities.shape}"
        return self

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {k: getattr(self, k) for k in PARAM_NAMES}


# ----------------------------------------------------------------- the SSOT
class GaussianSet:
    """Owns the 6 params + 4 buffers + the ONE chokepoint for all surgery."""

    def __init__(self, model, lock: "threading.RLock") -> None:
        self._model = model
        self._lock = lock
        self._version = 0
        # Adopt the model's gauss_params as the SSOT (same Parameter objects; no copy).
        n = self._params()["means"].shape[0]
        dev = self.device
        self._buffers: Dict[str, torch.Tensor] = {}
        for name, dt in IDENTITY_BUFFER_SPECS:
            shape = (n, 1)
            self._buffers[name] = torch.zeros(shape, dtype=dt, device=dev)
            # GaussianSet (not the nn.Module) is the SSOT for identity buffers: persistence
            # goes through state_dict()/reload_from_state_dict() here, NOT model.state_dict()
            # (static_persist drops model.state_dict()). setattr is only a convenience accessor
            # (model.<name>); buffers are allocated on the model's device, so no .to() skew.
            setattr(self._model, name, self._buffers[name])
        self._assert_invariant()

    # ----- internal access to the model's live param dict (mutable) -----
    def _params(self) -> Dict[str, torch.Tensor]:
        return self._model.gauss_params

    @property
    def device(self) -> torch.device:
        return self._model.device

    @property
    def num_points(self) -> int:
        return self._params()["means"].shape[0]

    def version(self) -> int:
        return self._version

    @property
    def sh_rest_dim(self) -> int:
        return int(self._params()["features_rest"].shape[1])

    # ----- invariant -----
    def _assert_invariant(self) -> None:
        n = self._params()["means"].shape[0]
        for k in PARAM_NAMES:
            assert self._params()[k].shape[0] == n, f"param {k} len {self._params()[k].shape[0]} != {n}"
        for k in _BUFFER_NAMES:
            assert self._buffers[k].shape[0] == n, f"buffer {k} len {self._buffers[k].shape[0]} != {n}"

    def _set_param(self, name: str, tensor: torch.Tensor) -> None:
        """Replace a param in the model's dict, preserving requires_grad as a leaf Parameter."""
        p = torch.nn.Parameter(tensor.contiguous(), requires_grad=self._params()[name].requires_grad)
        self._params()[name] = p

    def _sync_buffer_attr(self) -> None:
        for k in _BUFFER_NAMES:
            setattr(self._model, k, self._buffers[k])

    def _bump(self) -> None:
        self._version += 1

    # ----- reads -----
    def snapshot(self) -> GaussianSnapshot:
        with self._lock:
            params = {k: self._params()[k].detach() for k in PARAM_NAMES}
            buffers = {k: self._buffers[k].detach() for k in _BUFFER_NAMES}
            return GaussianSnapshot(params=params, buffers=buffers,
                                    num_points=params["means"].shape[0],
                                    version=self._version)

    # ----- surgery (all lock + assert invariant at exit) -----
    def cull(self, indices: torch.Tensor, *, protect_mask: Optional[torch.Tensor] = None) -> int:
        with self._lock:
            n = self.num_points
            dev = self.device
            idx = torch.as_tensor(indices, device=dev).long().flatten()
            if idx.numel() == 0:
                return 0
            idx = idx[(idx >= 0) & (idx < n)].unique()
            if protect_mask is not None:
                pm = torch.as_tensor(protect_mask, device=dev).bool().flatten()
                assert pm.shape[0] == n, "protect_mask len mismatch"
                keep_protected = pm[idx]
                idx = idx[~keep_protected]            # silent-drop protected (spec open-Q#5: safer for FF purge)
            if idx.numel() == 0:
                return 0
            keep = torch.ones(n, dtype=torch.bool, device=dev)
            keep[idx] = False
            for k in PARAM_NAMES:
                self._set_param(k, self._params()[k].detach()[keep])
            for k in _BUFFER_NAMES:
                self._buffers[k] = self._buffers[k][keep]
            self._sync_buffer_attr()
            self._model.rebind()
            self._assert_invariant()
            self._bump()
            return int(idx.numel())

    def insert(self, tensors: GaussTensors, *, object_flag: float, instance_id: int) -> torch.Tensor:
        with self._lock:
            old = self.num_points
            dev, dt = self.device, self._params()["means"].dtype
            t = tensors.validate(self.sh_rest_dim)
            add = t.as_dict()
            m = add["means"].shape[0]
            for k in PARAM_NAMES:
                cur = self._params()[k].detach()
                new = add[k].to(device=dev, dtype=cur.dtype)
                self._set_param(k, torch.cat([cur, new], dim=0))
            # grow buffers + write identity on the new tail
            for name, bdt in IDENTITY_BUFFER_SPECS:
                tail = torch.zeros((m, 1), dtype=bdt, device=dev)
                if name == "object_flags":
                    tail.fill_(float(object_flag))
                elif name == "object_instance_ids":
                    tail.fill_(int(instance_id))
                elif name == "inserted_flags":
                    tail.fill_(1.0)
                # sam3d_init_target_flags stays 0 (no value-writer, Inv #8)
                self._buffers[name] = torch.cat([self._buffers[name], tail], dim=0)
            self._sync_buffer_attr()
            self._model.rebind()
            self._assert_invariant()
            self._bump()
            return torch.arange(old, old + m, device=dev)

    def write_object_pose(self, means_subset: torch.Tensor, quats_subset: torch.Tensor,
                          object_mask: torch.Tensor) -> int:
        with self._lock:
            dev = self.device
            mask = torch.as_tensor(object_mask, device=dev).bool().flatten()
            assert mask.shape[0] == self.num_points, "object_mask len mismatch"
            rows = int(mask.sum().item())
            assert means_subset.shape[0] == rows, f"means subset {means_subset.shape[0]} != {rows}"
            assert quats_subset.shape[0] == rows, f"quats subset {quats_subset.shape[0]} != {rows}"
            assert torch.isfinite(means_subset).all() and torch.isfinite(quats_subset).all(), "non-finite pose"
            with torch.no_grad():
                mp, qp = self._params()["means"], self._params()["quats"]
                mp[mask] = means_subset.to(device=mp.device, dtype=mp.dtype)   # device too (adversarial review: cross-device guard)
                qp[mask] = quats_subset.to(device=qp.device, dtype=qp.dtype)
            self._bump()
            return rows

    def set_object_flags(self, mask: torch.Tensor, value: float) -> None:
        with self._lock:
            m = torch.as_tensor(mask, device=self.device).bool().flatten()
            assert m.shape[0] == self.num_points, "mask len mismatch"
            self._buffers["object_flags"][m] = float(value)
            self._bump()

    def write_instance_ids(self, mask: torch.Tensor, instance_id: int) -> None:
        with self._lock:
            m = torch.as_tensor(mask, device=self.device).bool().flatten()
            assert m.shape[0] == self.num_points, "mask len mismatch"
            self._buffers["object_instance_ids"][m] = int(instance_id)
            self._bump()

    # ----- warm-cache reload -----
    def reload_from_state_dict(self, state_dict: dict, num_points: int) -> None:
        """Reallocate the 6 params at num_points + rebuild the 4 buffers, then rebind.

        `state_dict` carries gauss_params.<name> and (optionally) identity-buffer tensors.
        Buffers absent from the dict are zero-filled (legacy caches)."""
        with self._lock:
            dev = self.device
            for k in PARAM_NAMES:
                key = k if k in state_dict else f"gauss_params.{k}"
                t = state_dict[key].to(dev)
                assert t.shape[0] == num_points, f"{k} reload len {t.shape[0]} != {num_points}"
                self._set_param(k, t)
            for name, bdt in IDENTITY_BUFFER_SPECS:
                if name in state_dict:
                    b = state_dict[name].to(device=dev, dtype=bdt)
                    if b.dim() == 1:
                        b = b.unsqueeze(1)
                    assert b.shape[0] == num_points, f"{name} reload len mismatch"
                    self._buffers[name] = b
                else:
                    self._buffers[name] = torch.zeros((num_points, 1), dtype=bdt, device=dev)
            self._sync_buffer_attr()
            self._model.rebind()
            self._assert_invariant()
            self._bump()

    def state_dict(self) -> dict:
        """Detached SSOT export for the warm-cache writer (params + buffers)."""
        with self._lock:
            out = {f"gauss_params.{k}": self._params()[k].detach().cpu() for k in PARAM_NAMES}
            for k in _BUFFER_NAMES:
                out[k] = self._buffers[k].detach().cpu()
            return out


# --------------------------------------------------------------- free helpers
def activated_opacity(opacity_logits):
    if isinstance(opacity_logits, torch.Tensor):
        return torch.sigmoid(opacity_logits)
    import numpy as np
    return 1.0 / (1.0 + np.exp(-opacity_logits))


def low_opacity_indices(opacity_logits: torch.Tensor, thr: float) -> torch.Tensor:
    a = activated_opacity(opacity_logits).flatten()
    return torch.nonzero(a < thr, as_tuple=False).flatten()


def uniform_shrink_log_scales(log_scales: torch.Tensor, max_scale_m: float, *, min_scale_m: float = 0.0):
    """Uniformly shrink (preserve shape) gaussians whose LARGEST axis exceeds max_scale_m;
    return (new_log_scales, keep_mask) where keep_mask drops those whose largest axis < min_scale_m.
    Matches the static + FF-insert scale-hygiene (spec §5 DUP Pattern F)."""
    scales = torch.exp(log_scales)
    largest, _ = scales.max(dim=1)
    out = log_scales.clone()
    if max_scale_m and max_scale_m > 0:
        over = largest > max_scale_m
        if over.any():
            factor = (max_scale_m / largest[over]).log()      # add in log-space => uniform divide
            out[over] = out[over] + factor.unsqueeze(1)
    keep = torch.ones(log_scales.shape[0], dtype=torch.bool, device=log_scales.device)
    if min_scale_m and min_scale_m > 0:
        keep = largest >= min_scale_m
    return out, keep


def _knn_mean_dist(xyz: torch.Tensor, k: int = 3) -> torch.Tensor:
    n = xyz.shape[0]
    if n <= 1:
        return torch.full((n,), 0.01, device=xyz.device, dtype=xyz.dtype)
    kk = min(k + 1, n)
    d = torch.cdist(xyz, xyz)                       # (n,n)
    knn, _ = torch.topk(d, kk, dim=1, largest=False)
    return knn[:, 1:].mean(dim=1).clamp_min(1e-6)   # exclude self (col 0)


def build_default_gauss_tensors(new_xyz, new_rgb, *, sh_degree: int, sh_rest_dim: int,
                                device, dtype) -> GaussTensors:
    """kNN-spacing log-scale seed, RGB2SH features_dc, zero features_rest, identity quats,
    logit(0.1) opacity. Used for the seed + tests; FF/Phase-0b build their own GaussTensors."""
    xyz = torch.as_tensor(new_xyz, device=device, dtype=dtype)
    rgb = torch.as_tensor(new_rgb, device=device, dtype=dtype)
    n = xyz.shape[0]
    spacing = _knn_mean_dist(xyz)
    log_scales = torch.log(spacing).unsqueeze(1).repeat(1, 3)
    features_dc = ((rgb - 0.5) / _RGB2SH_C0).reshape(n, 3)
    features_rest = torch.zeros((n, sh_rest_dim, 3), device=device, dtype=dtype)
    quats = torch.zeros((n, 4), device=device, dtype=dtype); quats[:, 0] = 1.0
    opacities = torch.full((n, 1), -2.1972246, device=device, dtype=dtype)   # logit(0.1)
    return GaussTensors(means=xyz, features_dc=features_dc, features_rest=features_rest,
                        scales=log_scales, quats=quats, opacities=opacities)
