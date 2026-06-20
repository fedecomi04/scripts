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

FREE-LIST (dynamic phase only): the gauss_params tensors are OVER-ALLOCATED to a
capacity; the `_count` live rows are kept packed at [0:_count]. cull does swap-remove
(move tail live rows into the deleted holes, _count -= k = O(deleted), no realloc);
insert writes into the [_count:_count+m] dead region (O(m), no realloc) and only
reallocs (grows capacity x2) on overflow. EVERY count/identity read slices [:_count]
so the dead capacity rows [_count:] are invisible to readers, persistence, masks. The
ONLY consumer that sees the full capacity tensor is the render — scene_model.render
hands gsplat a [:_count] VIEW so dead rows never rasterize (render stays O(count)).
Enabled when freelist=True (dynamic phase: LR=0 + no_grad render make the capacity
Parameter safe); static phase keeps the exact reallocating path (it trains WITH grad).

STABLE GAUSSIAN ID (`gauss_uid` buffer): because swap-remove / insert REORDER rows, no
consumer may rely on row position being stable across surgeries. Every row carries a unique
`gauss_uid` (int64) assigned at insert/load and carried through every move. The tracked-object
correspondence (ReferenceObjectPose) is keyed by THIS uid, not by row index — so reordering is
harmless.
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

    def __init__(self, model, lock: "threading.RLock", *, freelist: bool = False) -> None:
        self._model = model
        self._lock = lock
        self._version = 0
        # Free-list: dynamic phase over-allocates + keeps a live `_count` (swap-remove cull /
        # tail-write insert = O(changed)). Static phase keeps the exact reallocating path.
        self._freelist = bool(freelist)
        # Adopt the model's gauss_params as the SSOT (same Parameter objects; no copy).
        n = self._params()["means"].shape[0]
        self._count = n              # live row count; == shape[0] when NOT freelist
        self._capacity = n           # allocated rows; == _count until an insert grows it
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
        # Stable per-row uid (int64): unique + carried through every reorder/insert so the tracked
        # object can be matched by identity, not row position (ReferenceObjectPose). NOT persisted
        # (rebuilt fresh each load — the reference is captured at runtime). Lives alongside the
        # 6 params + 4 buffers and is moved in lockstep by every surgery (_uid_all() helpers).
        self._uid = torch.arange(n, dtype=torch.long, device=dev).reshape(n, 1)
        self._next_uid = n
        self._assert_invariant()

    # ----- internal access to the model's live param dict (mutable) -----
    def _params(self) -> Dict[str, torch.Tensor]:
        return self._model.gauss_params

    @property
    def device(self) -> torch.device:
        return self._model.device

    @property
    def num_points(self) -> int:
        return self._count       # live rows only; dead capacity rows [_count:] are invisible

    @property
    def count(self) -> int:
        """Live row count — scene_model slices gauss_params[:count] for the render."""
        return self._count

    def version(self) -> int:
        return self._version

    @property
    def sh_rest_dim(self) -> int:
        return int(self._params()["features_rest"].shape[1])

    # ----- invariant -----
    def _assert_invariant(self) -> None:
        # All param + buffer tensors share ONE physical length (= capacity); _count <= capacity;
        # live rows are packed at [0:_count]. (When not freelist, _count == capacity == shape[0].)
        cap = self._params()["means"].shape[0]
        assert self._capacity == cap, f"capacity {self._capacity} != means physical len {cap}"
        assert 0 <= self._count <= cap, f"count {self._count} out of [0,{cap}]"
        assert self._uid.shape[0] == cap, f"uid len {self._uid.shape[0]} != capacity {cap}"
        for k in PARAM_NAMES:
            assert self._params()[k].shape[0] == cap, f"param {k} len {self._params()[k].shape[0]} != {cap}"
        for k in _BUFFER_NAMES:
            assert self._buffers[k].shape[0] == cap, f"buffer {k} len {self._buffers[k].shape[0]} != {cap}"

    def _set_param(self, name: str, tensor: torch.Tensor) -> None:
        """Replace a param in the model's dict, preserving requires_grad as a leaf Parameter."""
        p = torch.nn.Parameter(tensor.contiguous(), requires_grad=self._params()[name].requires_grad)
        self._params()[name] = p

    def _sync_buffer_attr(self) -> None:
        for k in _BUFFER_NAMES:
            setattr(self._model, k, self._buffers[k])

    def _bump(self) -> None:
        self._version += 1

    def enable_freelist(self) -> None:
        """Flip static (exact-realloc) -> dynamic (free-list) IN PLACE, for the single-process
        static->dynamic hand-off (no scene reload, keeps the warm models). Safe ONLY when there are
        no dead rows yet (_count == _capacity), which is exactly the post-static state: the next insert
        then begins over-allocating from here, identical to a fresh reload_from_state_dict + freelist."""
        with self._lock:
            assert self._count == self._capacity, \
                f"enable_freelist requires packed rows (_count {self._count} == _capacity {self._capacity})"
            self._freelist = True
            self._assert_invariant()

    # ----- reads -----
    def snapshot(self) -> GaussianSnapshot:
        with self._lock:
            c = self._count
            params = {k: self._params()[k].detach()[:c] for k in PARAM_NAMES}
            buffers = {k: self._buffers[k].detach()[:c] for k in _BUFFER_NAMES}
            buffers["gauss_uid"] = self._uid.detach()[:c]   # stable per-row id (identity-keyed match)
            return GaussianSnapshot(params=params, buffers=buffers,
                                    num_points=c, version=self._version)

    # ----- surgery (all lock + assert invariant at exit) -----
    def cull(self, indices: torch.Tensor, *, protect_mask: Optional[torch.Tensor] = None) -> int:
        with self._lock:
            n = self._count
            dev = self.device
            idx = torch.as_tensor(indices, device=dev).long().flatten()
            if idx.numel() == 0:
                return 0
            idx = idx[(idx >= 0) & (idx < n)].unique()
            if protect_mask is not None and idx.numel():
                pm = torch.as_tensor(protect_mask, device=dev).bool().flatten()
                assert pm.shape[0] == n, "protect_mask len mismatch"
                idx = idx[~pm[idx]]                   # silent-drop protected (spec open-Q#5: safer for FF purge)
            if idx.numel() == 0:
                return 0
            k = int(idx.numel())
            if self._freelist:
                self._cull_swap_remove(idx, n, k)     # O(deleted): no realloc
            else:
                keep = torch.ones(n, dtype=torch.bool, device=dev)
                keep[idx] = False
                for name in PARAM_NAMES:
                    self._set_param(name, self._params()[name].detach()[keep])
                for name in _BUFFER_NAMES:
                    self._buffers[name] = self._buffers[name][keep]
                self._uid = self._uid[keep]           # carry stable uids through the cull
                self._capacity = self._count = n - k
            self._sync_buffer_attr()
            self._model.rebind()
            self._assert_invariant()
            self._bump()
            return k

    def _cull_swap_remove(self, idx: torch.Tensor, n: int, k: int) -> None:
        """Swap-remove: move the live rows from the tail [n-k:n] into the holes left by `idx`
        in the kept region [0:n-k], then drop _count by k. Copies <= k rows (O(deleted)),
        never reallocates. Live rows stay packed at [0:new_count]; dead rows [new_count:] are
        left as stale data (invisible to every reader, which slices [:_count]). REORDERS rows —
        safe because consumers match by gauss_uid (carried below), never by row index."""
        new_n = n - k
        dev = self.device
        del_mask = torch.zeros(n, dtype=torch.bool, device=dev)
        del_mask[idx] = True
        holes = idx[idx < new_n]                       # deleted positions that survive the truncation
        tail = torch.arange(new_n, n, device=dev)      # the k rows about to fall off the end
        tail_keep = tail[~del_mask[tail]]              # of those, the ones still LIVE -> move them down
        # |holes| == |tail_keep| by construction (every deleted-in-kept-region hole is backfilled
        # by a live tail row; deleted tail rows just vanish with the truncation).
        if holes.numel() > 0:
            for name in PARAM_NAMES:
                p = self._params()[name]
                with torch.no_grad():
                    p[holes] = p[tail_keep]
            for name in _BUFFER_NAMES:
                b = self._buffers[name]
                b[holes] = b[tail_keep]
            self._uid[holes] = self._uid[tail_keep]    # carry stable uids through the swap-remove
        self._count = new_n

    def insert(self, tensors: GaussTensors, *, object_flag: float, instance_id: int) -> torch.Tensor:
        with self._lock:
            old = self._count
            dev = self.device
            t = tensors.validate(self.sh_rest_dim)
            add = t.as_dict()
            m = add["means"].shape[0]
            if m == 0:
                return torch.arange(old, old, device=dev)
            id_vals = {"object_flags": float(object_flag),
                       "object_instance_ids": int(instance_id),
                       "inserted_flags": 1.0}   # sam3d_init_target_flags stays 0 (Inv #8)
            if self._freelist:
                self._insert_tail_write(add, m, id_vals)     # grow-if-needed + write [count:count+m]
            else:
                for name in PARAM_NAMES:
                    cur = self._params()[name].detach()
                    new = add[name].to(device=dev, dtype=cur.dtype)
                    self._set_param(name, torch.cat([cur, new], dim=0))
                for name, bdt in IDENTITY_BUFFER_SPECS:
                    tail = torch.full((m, 1), id_vals.get(name, 0.0), dtype=bdt, device=dev)
                    self._buffers[name] = torch.cat([self._buffers[name], tail], dim=0)
                self._uid = torch.cat([self._uid, self._fresh_uids(m)], dim=0)
                self._capacity = self._count = old + m
            self._sync_buffer_attr()
            self._model.rebind()
            self._assert_invariant()
            self._bump()
            return torch.arange(old, old + m, device=dev)

    def _fresh_uids(self, m: int) -> torch.Tensor:
        """m new globally-unique row ids (monotonic counter, never reused). Shape (m,1) int64."""
        u = torch.arange(self._next_uid, self._next_uid + m, dtype=torch.long,
                         device=self.device).reshape(m, 1)
        self._next_uid += m
        return u

    def _insert_tail_write(self, add: Dict[str, torch.Tensor], m: int, id_vals: dict) -> None:
        """Free-list insert core (NO lock/rebind — caller owns those): grow capacity if needed,
        then write the m new rows into the [count:count+m] dead region + their identity. O(m)."""
        old = self._count
        dev = self.device
        if old + m > self._capacity:
            self._grow_capacity(old + m)                     # realloc x2 (amortized O(1))
        for name in PARAM_NAMES:
            p = self._params()[name]
            with torch.no_grad():
                p[old:old + m] = add[name].to(device=dev, dtype=p.dtype)
        for name, bdt in IDENTITY_BUFFER_SPECS:
            self._buffers[name][old:old + m] = id_vals.get(name, 0.0)
        self._uid[old:old + m] = self._fresh_uids(m)         # assign stable ids to the new rows
        self._count = old + m

    def _grow_capacity(self, need: int) -> None:
        """Reallocate params + buffers to at least `need` rows (grow x2, amortized O(1) inserts).
        Copies the live [0:_count] rows into the front of the new storage; rows [_count:] are
        uninitialized (params: stale/zero — overwritten by the insert; buffers: zero-filled)."""
        new_cap = max(int(need), self._capacity * 2, 1)
        dev = self.device
        c = self._count
        for name in PARAM_NAMES:
            cur = self._params()[name].detach()
            big = torch.zeros((new_cap, *cur.shape[1:]), dtype=cur.dtype, device=dev)
            big[:c] = cur[:c]
            self._set_param(name, big)
        for name, bdt in IDENTITY_BUFFER_SPECS:
            cur = self._buffers[name]
            big = torch.zeros((new_cap, 1), dtype=bdt, device=dev)
            big[:c] = cur[:c]
            self._buffers[name] = big
        big_uid = torch.zeros((new_cap, 1), dtype=torch.long, device=dev)
        big_uid[:c] = self._uid[:c]
        self._uid = big_uid
        self._capacity = new_cap

    def cull_and_insert(self, cull_indices: torch.Tensor, tensors: GaussTensors, *,
                        object_flag: float, instance_id: int,
                        protect_mask: Optional[torch.Tensor] = None):
        """ATOMIC cull + insert in ONE locked surgery + ONE rebind. Used by the FF Option-A flow so
        nothing ever observes a culled-but-not-yet-filled scene (no flicker). _count is written EXACTLY
        ONCE at the end — even a LOCKLESS num_points poll never sees the cull dip (the render is atomic
        regardless, since it holds the lock for the whole op). Returns (n_culled, inserted_index_range,
        in POST-cull coordinates). Free-list: row moves run with _count pinned at its old value."""
        with self._lock:
            dev = self.device
            n = self._count
            idx = torch.as_tensor(cull_indices, device=dev).long().flatten()
            idx = idx[(idx >= 0) & (idx < n)].unique() if idx.numel() else idx
            if protect_mask is not None and idx.numel():
                pm = torch.as_tensor(protect_mask, device=dev).bool().flatten()
                assert pm.shape[0] == n, "protect_mask len mismatch"
                idx = idx[~pm[idx]]
            k = int(idx.numel())
            t = tensors.validate(self.sh_rest_dim)
            add = t.as_dict()
            m = add["means"].shape[0]
            id_vals = {"object_flags": float(object_flag),
                       "object_instance_ids": int(instance_id), "inserted_flags": 1.0}
            new_count = n - k
            if self._freelist:
                # 1) swap-remove DATA moves only — _count stays pinned at n until the final write.
                if k > 0:
                    del_mask = torch.zeros(n, dtype=torch.bool, device=dev)
                    del_mask[idx] = True
                    holes = idx[idx < new_count]
                    tail = torch.arange(new_count, n, device=dev)
                    tail_keep = tail[~del_mask[tail]]
                    if holes.numel() > 0:
                        for name in PARAM_NAMES:
                            with torch.no_grad():
                                self._params()[name][holes] = self._params()[name][tail_keep]
                        for name in _BUFFER_NAMES:
                            self._buffers[name][holes] = self._buffers[name][tail_keep]
                        self._uid[holes] = self._uid[tail_keep]   # carry uids through swap-remove
                # 2) grow if needed, then write the inserts at [new_count : new_count+m].
                if new_count + m > self._capacity:
                    self._grow_capacity(new_count + m)
                if m > 0:
                    for name in PARAM_NAMES:
                        with torch.no_grad():
                            self._params()[name][new_count:new_count + m] = add[name].to(dev, self._params()[name].dtype)
                    for name, bdt in IDENTITY_BUFFER_SPECS:
                        self._buffers[name][new_count:new_count + m] = id_vals.get(name, 0.0)
                    self._uid[new_count:new_count + m] = self._fresh_uids(m)   # uids for the new rows
                self._count = new_count + m                    # SINGLE count write (atomic, no dip)
            else:
                if k > 0:
                    keep = torch.ones(n, dtype=torch.bool, device=dev)
                    keep[idx] = False
                    for name in PARAM_NAMES:
                        self._set_param(name, self._params()[name].detach()[keep])
                    for name in _BUFFER_NAMES:
                        self._buffers[name] = self._buffers[name][keep]
                    self._uid = self._uid[keep]
                if m > 0:
                    for name in PARAM_NAMES:
                        cur = self._params()[name].detach()
                        self._set_param(name, torch.cat([cur, add[name].to(device=dev, dtype=cur.dtype)], dim=0))
                    for name, bdt in IDENTITY_BUFFER_SPECS:
                        tail = torch.full((m, 1), id_vals.get(name, 0.0), dtype=bdt, device=dev)
                        self._buffers[name] = torch.cat([self._buffers[name], tail], dim=0)
                    self._uid = torch.cat([self._uid, self._fresh_uids(m)], dim=0)
                self._capacity = self._count = new_count + m
            self._sync_buffer_attr()
            self._model.rebind()                              # ONE rebind for the whole atomic op
            self._assert_invariant()
            self._bump()
            return k, torch.arange(new_count, new_count + m, device=dev)

    def write_object_pose(self, means_subset: torch.Tensor, quats_subset: torch.Tensor,
                          subset_uid: torch.Tensor) -> int:
        """Write the tracked object's transformed pose onto its LIVE rows, matched by stable gauss_uid.
        subset_uid[i] is the uid the means_subset[i]/quats_subset[i] row belongs to. We resolve uid->row
        HERE under the lock against the live _uid, so any FF cull/insert/reorder that ran since the caller
        snapshotted is harmless: a culled uid is simply skipped, and reordering can't misplace the pose."""
        with self._lock:
            dev = self.device
            n = self.num_points
            want = torch.as_tensor(subset_uid, device=dev).flatten()
            assert means_subset.shape[0] == want.shape[0], f"means subset {means_subset.shape[0]} != uids {want.shape[0]}"
            assert quats_subset.shape[0] == want.shape[0], f"quats subset {quats_subset.shape[0]} != uids {want.shape[0]}"
            assert torch.isfinite(means_subset).all() and torch.isfinite(quats_subset).all(), "non-finite pose"
            live_uid = self._uid[:n, 0]
            # row[i] = live index whose uid == want[i]; -1 if that uid was culled since the snapshot.
            order = torch.argsort(live_uid)
            pos = torch.searchsorted(live_uid[order], want).clamp(max=n - 1)
            rows = order[pos]
            valid = live_uid[rows] == want                       # drop uids no longer present (FF-culled)
            if not bool(valid.all()):
                rows, means_subset, quats_subset = rows[valid], means_subset[valid], quats_subset[valid]
            with torch.no_grad():
                mp, qp = self._params()["means"][:n], self._params()["quats"][:n]
                mp[rows] = means_subset.to(device=mp.device, dtype=mp.dtype)
                qp[rows] = quats_subset.to(device=qp.device, dtype=qp.dtype)
            self._bump()
            return int(rows.shape[0])

    def set_object_flags(self, mask: torch.Tensor, value: float) -> None:
        with self._lock:
            c = self._count
            m = torch.as_tensor(mask, device=self.device).bool().flatten()
            assert m.shape[0] == c, "mask len mismatch"
            self._buffers["object_flags"][:c][m] = float(value)   # [:c] view -> writes the live rows
            self._bump()

    def write_instance_ids(self, mask: torch.Tensor, instance_id: int) -> None:
        with self._lock:
            c = self._count
            m = torch.as_tensor(mask, device=self.device).bool().flatten()
            assert m.shape[0] == c, "mask len mismatch"
            self._buffers["object_instance_ids"][:c][m] = int(instance_id)   # [:c] view -> live rows
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
            # Reload packs exactly num_points live rows at full capacity (no dead tail yet;
            # the first freelist insert will over-allocate from here).
            self._capacity = self._count = num_points
            self._uid = torch.arange(num_points, dtype=torch.long, device=dev).reshape(num_points, 1)
            self._next_uid = num_points
            self._sync_buffer_attr()
            self._model.rebind()
            self._assert_invariant()
            self._bump()

    def state_dict(self) -> dict:
        """Detached SSOT export for the warm-cache writer (params + buffers). Exports only the
        [:count] live rows — dead capacity rows are never persisted."""
        with self._lock:
            c = self._count
            out = {f"gauss_params.{k}": self._params()[k].detach()[:c].cpu() for k in PARAM_NAMES}
            for k in _BUFFER_NAMES:
                out[k] = self._buffers[k].detach()[:c].cpu()
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
