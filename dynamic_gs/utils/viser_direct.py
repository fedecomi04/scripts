"""Viser-direct visualization for live tracking (Path A — hybrid).

Bypasses Nerfstudio's server-side rasterize-and-push viewer path by
maintaining two ``GaussianSplatHandle`` objects in a standalone Viser
server:

  - ``static_handle``  : all non-tracked Gaussians (``object_flags == 0``).
                          Holds the original scene + any feedforward
                          inserts. Re-uploaded whenever the count changes
                          (after each feedforward call).
  - ``tracked_handle`` : the moved-object Gaussians (``object_flags == 1``,
                          excluding feedforward-instance 999). Uploaded
                          ONCE at the static→dynamic boundary in the D0
                          reference pose. Each tracker tick pushes
                          ``handle.position`` + ``handle.wxyz`` with the
                          world-frame rigid transform that the motion
                          estimator returned.

Browser-side does the WebGL splat rasterization; the training GPU never
serves a viewer frame, so the live tick rate is no longer capped by
viewer rerender contention.

Use with ``--vis=tensorboard`` (no Nerfstudio viewer). Open the printed
``http://<host>:<port>`` in a browser to see the scene.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

try:
    import viser  # type: ignore
    _HAS_VISER = True
except Exception:
    viser = None  # type: ignore
    _HAS_VISER = False


# SH C0 coefficient. Splatfacto stores RGB color as ``features_dc`` in
# SH-DC space; convert back to base RGB with ``0.5 + dc * SH_C0``.
_SH_C0 = 0.28209479177387814


def _quat_wxyz_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Normalized wxyz quaternion → (..., 3, 3) rotation matrix."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(-1)
    R = torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],     dim=-1),
            torch.stack([2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],     dim=-1),
            torch.stack([2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)], dim=-1),
        ],
        dim=-2,
    )
    return R


def _rotmat_to_wxyz_np(R: np.ndarray) -> np.ndarray:
    """(3, 3) numpy rotation → (4,) wxyz numpy. Shepperd's method."""
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float32)


def _build_splat_arrays(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales_log: torch.Tensor,
    features_dc: torch.Tensor,
    opacities_logit: torch.Tensor,
) -> dict:
    """Convert the live model param tensors for a Gaussian subset into the
    numpy arrays viser's ``_add_gaussian_splats`` wants:
      centers (N, 3), covariances (N, 3, 3), rgbs (N, 3), opacities (N,).
    Everything moved to CPU/float32/numpy in a single batch.
    """
    with torch.no_grad():
        scales = torch.exp(scales_log)  # log → linear
        R = _quat_wxyz_to_rotmat(quats)  # (N, 3, 3)
        # cov = R @ diag(s^2) @ R.T = (R * s[None,:]) @ (R * s[None,:]).T
        M = R * scales.unsqueeze(-2)  # multiply column j of R by s_j
        covariances = M @ M.transpose(-1, -2)
        rgbs = (0.5 + features_dc * _SH_C0).clamp(0.0, 1.0)
        # viser expects opacities as (N, 1).
        if opacities_logit.ndim == 1:
            opacities = torch.sigmoid(opacities_logit).unsqueeze(-1)
        else:
            opacities = torch.sigmoid(opacities_logit)
    return {
        "centers":     means.detach().cpu().float().numpy(),
        "covariances": covariances.detach().cpu().float().numpy(),
        "rgbs":        rgbs.detach().cpu().float().numpy(),
        "opacities":   opacities.detach().cpu().float().numpy(),
    }


class ViserDirectScene:
    """Wraps a standalone viser server + two splat handles for live mode.

    Use ``--vis=tensorboard`` so Nerfstudio's viewer is OFF; this server
    handles all visualization. The training GPU does not render for the
    viewer — the browser does, via WebGL splatting.
    """

    def __init__(self, port: int = 8081, opacity_floor: float = 0.05,
                 static_refresh_min_gap_s: float = -1.0,
                 push_min_gap_s: float = 0.033):
        """``opacity_floor`` drops splats with sigmoid(opacities) below
        the threshold. The prior viser dump-tool symptom (white browser)
        was 95% near-zero-opacity splats overwhelming Chrome WebGL.
        Default 0.05 typically halves the splat count with no visible
        difference; set to 0.0 to keep everything."""
        if not _HAS_VISER:
            raise RuntimeError(
                "viser is not installed in the dynamic_gs env. "
                "pip install viser, or disable enable_viser_direct."
            )
        self.server = viser.ViserServer(port=port)
        self.port = port
        self.opacity_floor = float(opacity_floor)
        self.static_refresh_min_gap_s = float(static_refresh_min_gap_s)
        self._last_static_refresh_t = 0.0
        self._pending_refresh_count = 0
        # Per-tick transform push throttle. At 25+ Hz tracker with chunked
        # tracked-handles, the per-tick push fires many handle.position +
        # .wxyz writes in rapid succession, which hits a known race in
        # websockets.legacy.protocol._drain_helper (AssertionError on
        # `waiter is None or waiter.cancelled()`). Throttling to ~30 Hz
        # wall-clock (default 33 ms gap) has no visual cost but stops the
        # drain race. Set to 0 to disable throttling.
        self.push_min_gap_s = float(push_min_gap_s)
        self._last_push_t = 0.0
        # Always-on world axes — diagnostic: if axes show in the browser
        # but splats don't, the camera is correct and splat-data conversion
        # is wrong; if even axes don't show, the camera or the WebGL bundle
        # is broken.
        try:
            self.server.scene.world_axes.visible = True
        except Exception:
            pass
        # Lazy: handles created when ``setup_handles`` is called (after
        # Phase 0b fusion finalises the scene).
        self.static_handle = None        # all non-tracked-object Gaussians
        self.tracked_handle = None       # the (single) moved object (chunks)
        self.tracked_root = None         # parent FrameHandle for tracked chunks
        self.tracked_instance_id: Optional[int] = None
        self._tracked_count = 0
        self._static_count = 0
        # Per-FF-call insert handles: each FF call adds ONE new small
        # handle here, never re-uploads prior ones. Capped at
        # ``_ff_handle_cap`` — when exceeded, the oldest handle is
        # removed to keep the scene-graph bounded.
        self.ff_handles: list = []
        self._ff_handle_cap = 500
        self._ff_call_counter = 0

    # ------------------------------------------------------------------
    # Handle setup
    # ------------------------------------------------------------------
    def setup_handles(self, model, tracked_instance_id: Optional[int] = None,
                       initial_c2w: Optional[np.ndarray] = None) -> None:
        """Build the (static, tracked) handles from the current model state.

        Splits ``model.gauss_params`` by ``object_flags`` + ``object_instance_ids``:
          - ``tracked_handle`` gets Gaussians where ``object_flags > 0.5`` AND
            ``object_instance_ids != 999`` (real Phase-0b objects, not FF inserts).
            If ``tracked_instance_id`` is given, restrict further to that ID.
          - ``static_handle`` gets everything else (background + FF inserts).
        """
        with torch.no_grad():
            flags = model.object_flags.squeeze(-1) > 0.5
            inst = model.object_instance_ids.squeeze(-1)
            tracked_mask = flags & (inst != 999)
            if tracked_instance_id is not None:
                tracked_mask = tracked_mask & (inst == int(tracked_instance_id))
            self.tracked_instance_id = tracked_instance_id
            static_mask = ~tracked_mask
            mm_all = model.means.detach().cpu().float()
            bbox_min = mm_all.min(dim=0).values.tolist()
            bbox_max = mm_all.max(dim=0).values.tolist()
            if int(tracked_mask.sum().item()) > 0:
                target = model.means[tracked_mask].detach().cpu().float().median(dim=0).values
            else:
                target = mm_all.median(dim=0).values
        target_np = target.numpy().astype(float)

        # Compute camera pose FIRST (needed for the distance diagnostic).
        if initial_c2w is not None:
            # Nerfstudio / OpenGL convention: +X right, +Y up, -Z forward.
            c2w = np.asarray(initial_c2w, dtype=np.float32)
            if c2w.shape == (3, 4):
                c2w = np.concatenate([c2w, np.array([[0, 0, 0, 1]], dtype=np.float32)], axis=0)
            elif c2w.shape != (4, 4):
                raise ValueError(f"initial_c2w must be 4x4 or 3x4, got {c2w.shape}")
            cam_pos = c2w[:3, 3].astype(np.float32)
            forward = -c2w[:3, 2].astype(np.float32)
            up = c2w[:3, 1].astype(np.float32)
            look_at = (cam_pos + forward).astype(np.float32)
            cam_source = "live_c2w"
        else:
            cam_pos = (target_np + np.array([1.5, 0.0, 0.5])).astype(np.float32)
            look_at = target_np.astype(np.float32)
            up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            cam_source = "fallback_target_median"

        # Build handles (cam_pos passed in so distance-from-camera
        # diagnostic prints inside _add_handle).
        self.static_handle, n_static_kept, n_static_total = self._add_handle(
            "/static_scene", model, static_mask, cam_pos=cam_pos,
        )
        self._static_count = n_static_kept
        if int(tracked_mask.sum().item()) > 0:
            # Parent frame for all tracked-object chunks. Per-tick transform
            # pushes update THIS frame's wxyz+position; viser propagates the
            # transform to all children via scene-graph hierarchy. This way
            # we send 2 attr writes per tick instead of 2 × N_chunks ×
            # N_clients (which triggered the websockets _drain_helper race).
            self.tracked_root = self.server.scene.add_frame(
                name="/tracked_object_root",
                show_axes=False,
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(0.0, 0.0, 0.0),
            )
            # Chunk names are CHILDREN of the parent frame.
            self.tracked_handle, n_tr_kept, n_tr_total = self._add_handle(
                "/tracked_object_root/splats", model, tracked_mask, cam_pos=cam_pos,
            )
            self._tracked_count = n_tr_kept
        else:
            self.tracked_handle = None
            self.tracked_root = None
            self._tracked_count = 0
            n_tr_kept, n_tr_total = 0, 0

        # Diagnostic: a big red sphere at the look-at point + axes. If the
        # browser shows the sphere/axes but no splats, splat rendering is
        # broken (browser WebGL splat extension). If even the sphere is
        # missing, browser cache or camera frustum issue.
        try:
            self.server.scene.add_icosphere(
                name="/diag_sphere_at_lookat",
                radius=0.05,
                color=(255, 0, 0),
                position=look_at,
            )
            self.server.scene.add_frame(
                name="/diag_lookat_axes", position=look_at, axes_length=0.3, axes_radius=0.005,
            )
        except Exception as exc:
            print(f"[viser-direct] diag sphere add failed: {exc}", flush=True)

        @self.server.on_client_connect
        def _on_connect(client):
            client.camera.position = cam_pos
            client.camera.look_at = look_at
            client.camera.up_direction = up

        print(
            f"[viser-direct] handles built: "
            f"static={n_static_kept}/{n_static_total} splats, "
            f"tracked={n_tr_kept}/{n_tr_total} splats "
            f"(instance_id={tracked_instance_id}, opacity_floor={self.opacity_floor})\n"
            f"[viser-direct] scene bbox: min={[round(v,3) for v in bbox_min]} "
            f"max={[round(v,3) for v in bbox_max]}\n"
            f"[viser-direct] camera pre-set on connect ({cam_source}): "
            f"look_at={[round(float(v),3) for v in look_at]} "
            f"position={[round(float(v),3) for v in cam_pos]}",
            flush=True,
        )

    def _add_handle(self, name: str, model, mask: torch.Tensor, cam_pos=None,
                    max_per_handle: int = 80_000):
        """Slice ``model.gauss_params`` by ``mask``, apply the opacity
        floor, drop non-PD covariances, regularize, and upload as one
        or more splat handles. When the splat count exceeds
        ``max_per_handle``, the data is chunked into multiple handles
        ``<name>/chunk_00``, ``<name>/chunk_01``, ... — each handle is
        one WebSocket message, so chunking is what keeps individual
        uploads under Chrome's ~16 MB frame limit. At 64 B/splat
        (centers 12 + cov 36 + rgbs 12 + opac 4), 80k splats ≈ 5 MB.

        Returns ``(handle_or_list, n_kept, n_total)``. For chunked
        uploads the first element is a list of handles."""
        with torch.no_grad():
            n_total = int(mask.sum().item())
            keep = mask.clone()
            if self.opacity_floor > 0.0 and n_total > 0:
                opc = torch.sigmoid(model.opacities.squeeze(-1) if model.opacities.ndim > 1 else model.opacities)
                keep = keep & (opc > self.opacity_floor)
            n_after_op = int(keep.sum().item())
        arrs = _build_splat_arrays(
            means          = model.means[keep],
            quats          = model.quats[keep],
            scales_log     = model.scales[keep],
            features_dc    = model.features_dc[keep],
            opacities_logit= model.opacities[keep],
        )
        centers, covs, rgbs, opacs = arrs["centers"], arrs["covariances"], arrs["rgbs"], arrs["opacities"]

        # ---- Diagnostics: opacity ----
        opc_flat = opacs.reshape(-1)
        print(
            f"[viser-direct/{name}] opacity range = [{opc_flat.min():.4f}, "
            f"{opc_flat.max():.4f}] mean={opc_flat.mean():.4f} "
            f"(>0.5 frac: {(opc_flat > 0.5).mean()*100:.1f}%)",
            flush=True,
        )
        # ---- Diagnostics: RGB ----
        print(
            f"[viser-direct/{name}] rgb range per ch = R[{rgbs[:,0].min():.3f},{rgbs[:,0].max():.3f}] "
            f"G[{rgbs[:,1].min():.3f},{rgbs[:,1].max():.3f}] "
            f"B[{rgbs[:,2].min():.3f},{rgbs[:,2].max():.3f}] "
            f"mean={rgbs.mean(axis=0).round(3).tolist()}",
            flush=True,
        )

        # ---- Diagnostics + filter: covariance positive-definiteness ----
        # viser's WebGL splatter does a Cholesky decomp per splat. Degenerate
        # (≈0 smallest eigenvalue) covariances either silently skip OR break
        # the whole shader pipeline (white screen). Match what the working
        # `view_splats_viser.py` does: drop non-PD, regularize the rest.
        eig_min = np.linalg.eigvalsh(covs.astype(np.float64)).min(axis=-1)
        pd_keep = eig_min > 1e-9
        n_drop_pd = int((~pd_keep).sum())
        if n_drop_pd > 0:
            print(f"[viser-direct/{name}] dropping {n_drop_pd} non-PD covariances "
                  f"(min eig <= 1e-9; smallest seen: {eig_min.min():.2e})", flush=True)
            centers, covs, rgbs, opacs = centers[pd_keep], covs[pd_keep], rgbs[pd_keep], opacs[pd_keep]
        # Always regularize a tiny epsilon on the diagonal to avoid float32
        # numerical edge cases inside the shader's Cholesky.
        covs = covs + np.eye(3, dtype=covs.dtype)[None] * 1e-7

        # ---- Diagnostics: camera-to-splat distances ----
        if cam_pos is not None and centers.shape[0] > 0:
            d = np.linalg.norm(centers - np.asarray(cam_pos, dtype=np.float32)[None, :], axis=-1)
            print(
                f"[viser-direct/{name}] dist(splat → camera) "
                f"min={d.min():.3f}m p10={np.percentile(d,10):.3f} "
                f"median={np.median(d):.3f} p90={np.percentile(d,90):.3f} "
                f"max={d.max():.3f}m  "
                f"(any in front (d>0.05)? {int((d > 0.05).sum())}/{len(d)})",
                flush=True,
            )

        n_kept = centers.shape[0]
        if n_kept <= max_per_handle:
            print(f"[viser-direct/{name}] uploading {n_kept} splats (single)", flush=True)
            handle = self.server.scene._add_gaussian_splats(
                name=name,
                centers=centers,
                covariances=covs,
                rgbs=rgbs,
                opacities=opacs,
            )
            return handle, n_kept, n_total
        # Chunked upload: each handle is one WebSocket message under Chrome's
        # ~16 MB frame ceiling. Static-scene handles only — tracked-object
        # handle must stay single so a single per-tick transform applies to
        # all of it (~7 MB at 200k splats, fits comfortably).
        n_chunks = (n_kept + max_per_handle - 1) // max_per_handle
        chunk_size_mb = max_per_handle * 64 / (1024 * 1024)
        print(
            f"[viser-direct/{name}] uploading {n_kept} splats in {n_chunks} chunks "
            f"of ≤{max_per_handle} (~{chunk_size_mb:.1f} MB/chunk)",
            flush=True,
        )
        handles = []
        for i in range(n_chunks):
            start = i * max_per_handle
            end = min(start + max_per_handle, n_kept)
            chunk_name = f"{name}/chunk_{i:02d}"
            h = self.server.scene._add_gaussian_splats(
                name=chunk_name,
                centers=centers[start:end],
                covariances=covs[start:end],
                rgbs=rgbs[start:end],
                opacities=opacs[start:end],
            )
            handles.append(h)
        return handles, n_kept, n_total

    # ------------------------------------------------------------------
    # Per-tick transform push (cheap — handle stays uploaded; browser
    # re-rasterizes with the new transform on its own GPU).
    # ------------------------------------------------------------------
    def push_tracker_transform(self, R, t) -> None:
        """Push the latest world-frame rigid (R, t) to the tracked-object
        parent frame. Children (the splat chunks) inherit the transform
        via the viser scene graph, so we only emit 2 attribute writes
        per tick regardless of chunk count or client count. Throttled
        to ``push_min_gap_s``."""
        if self.tracked_root is None:
            return
        if self.push_min_gap_s > 0.0:
            import time as _time
            now = _time.time()
            if (now - self._last_push_t) < self.push_min_gap_s:
                return
            self._last_push_t = now
        R_np = R.detach().cpu().numpy() if isinstance(R, torch.Tensor) else np.asarray(R)
        t_np = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
        wxyz = _rotmat_to_wxyz_np(R_np[:3, :3])
        pos = t_np.astype(np.float32).reshape(3)
        self.tracked_root.wxyz = wxyz
        self.tracked_root.position = pos

    # ------------------------------------------------------------------
    # Static-handle refresh (call after FF inserts/deletes)
    # ------------------------------------------------------------------
    def refresh_static_handle(self, model) -> None:
        """Re-upload the static splat handle from the current model state.
        DISABLED by default (``static_refresh_min_gap_s < 0``) — the
        remove+re-add cycle for ~428k splats causes the whole static
        scene to flicker every refresh, which is jarring. When disabled,
        the browser holds the D0 snapshot of the static scene; FF inserts
        don't appear in viser (but the tracked-object handle still moves
        smoothly per tick — that's the whole point of Path A).

        Set ``static_refresh_min_gap_s > 0`` to enable throttled refresh.
        """
        if self.static_handle is None:
            return
        if self.static_refresh_min_gap_s <= 0:
            return  # disabled
        import time as _time
        self._pending_refresh_count += 1
        now = _time.time()
        if (now - self._last_static_refresh_t) < self.static_refresh_min_gap_s:
            return
        with torch.no_grad():
            flags = model.object_flags.squeeze(-1) > 0.5
            inst = model.object_instance_ids.squeeze(-1)
            tracked_mask = flags & (inst != 999)
            if self.tracked_instance_id is not None:
                tracked_mask = tracked_mask & (inst == int(self.tracked_instance_id))
            static_mask = ~tracked_mask
        # Replace the handle(s) entirely — viser doesn't support per-splat
        # update on this version (0.2.7). ``static_handle`` may be a list
        # of chunk-handles when the splat count exceeds the 16 MB
        # per-message limit (see ``_add_handle``).
        try:
            old = self.static_handle
            if isinstance(old, list):
                for h in old:
                    try: h.remove()
                    except Exception: pass
            else:
                old.remove()
        except Exception:
            pass
        self.static_handle, n_kept, _ = self._add_handle(
            "/static_scene", model, static_mask,
        )
        self._static_count = n_kept
        coalesced = self._pending_refresh_count
        self._pending_refresh_count = 0
        self._last_static_refresh_t = now
        print(
            f"[viser-direct] static refresh ({n_kept} splats, "
            f"coalesced {coalesced} FF calls)",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Incremental FF-insert visualization
    # ------------------------------------------------------------------
    def add_ff_insert_chunk(self, model, inserted_ids) -> None:
        """Upload JUST the splats freshly inserted by this FF call as a
        new standalone splat handle. The handle is appended to
        ``ff_handles``; prior handles are never re-uploaded. Per-call
        upload is tens to hundreds of KB on the wire.

        ``inserted_ids``: 1-D tensor returned by
        ``model.insert_inpaint_gaussians`` (the index range of the new
        splats in the now-resized ``gauss_params`` tensors).
        """
        if inserted_ids is None:
            return
        try:
            ids = inserted_ids.detach().cpu().long() if isinstance(inserted_ids, torch.Tensor) else torch.as_tensor(inserted_ids, dtype=torch.long)
        except Exception:
            return
        if ids.numel() == 0:
            return
        N = int(model.means.shape[0])
        mask = torch.zeros(N, dtype=torch.bool, device=model.means.device)
        valid = ids[(ids >= 0) & (ids < N)]
        if valid.numel() == 0:
            return
        mask[valid] = True
        chunk_name = f"/ff_inserts/call_{self._ff_call_counter:04d}"
        self._ff_call_counter += 1
        try:
            handle_or_list, n_kept, _ = self._add_handle(chunk_name, model, mask)
        except Exception as exc:
            print(f"[viser-direct] add_ff_insert_chunk failed: {exc}", flush=True)
            return
        if isinstance(handle_or_list, list):
            self.ff_handles.extend(handle_or_list)
        else:
            self.ff_handles.append(handle_or_list)
        # Cap scene-graph size — remove the oldest handles past the cap.
        while len(self.ff_handles) > self._ff_handle_cap:
            old = self.ff_handles.pop(0)
            try:
                old.remove()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def close(self) -> None:
        try:
            self.server.stop()
        except Exception:
            pass
