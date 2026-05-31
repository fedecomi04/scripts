"""AnySplat feedforward decode — subprocess dispatcher + canonical→world alignment.

The actual model runs in a sibling conda env (``anysplat_dynamic_gs``) via
``scripts/anysplat_worker.py`` because AnySplat's pinned wheels are
incompatible with the main ``dynamic_gs`` env's torch 2.11+cu128 stack.
We pay one subprocess cold-spawn per FF call (~10 s on warm HF cache).

This module is the in-env half:
    * spawn the worker on N image paths, parse the .npz
    * Umeyama 7-DoF similarity: canonical camera centres → known scene c2w centres
    * Apply (s, R, t) to gaussian means + scale scales by ``s`` + rotate quats by ``R``
    * Project the transformed gaussians through the known scene camera and keep
      only those whose 2D footprint lands inside a (resized) CDN component mask
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False

# Resolve once: third_party/AnySplat lives next to dynamic_gs/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ANYSPLAT_REPO = _REPO_ROOT / "third_party" / "AnySplat"
_WORKER_SCRIPT = _REPO_ROOT / "anysplat_worker.py"


class PersistentAnysplatWorker:
    """Long-lived AnySplat subprocess. Load model once, then issue many inferences.

    Spawn pays the model-load cost (~9 s warm cache). Each subsequent ``inference(...)``
    sends one JSON line over stdin and reads one JSON line from stdout; cost = pure
    GPU inference (~0.6 s for K=1) + IPC + disk write.
    """

    def __init__(self, conda_env: str = "anysplat_dynamic_gs", startup_timeout_s: float = 60.0):
        import json as _json
        if not _WORKER_SCRIPT.exists():
            raise FileNotFoundError(f"AnySplat worker script not found: {_WORKER_SCRIPT}")

        env = os.environ.copy()
        env_prefix = Path.home() / "miniconda3" / "envs" / conda_env
        env["LD_LIBRARY_PATH"] = (str(env_prefix / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")).rstrip(":")

        cmd = [
            "conda", "run", "-n", conda_env, "--no-capture-output",
            "python", str(_WORKER_SCRIPT),
            "--persistent",
            "--anysplat-repo", str(_ANYSPLAT_REPO),
        ]
        self._proc = subprocess.Popen(
            cmd, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        # Wait for the "ready" sentinel on stdout (model loaded into VRAM).
        t0 = time.time()
        while True:
            if time.time() - t0 > startup_timeout_s:
                self._proc.kill()
                raise TimeoutError(f"AnySplat persistent worker startup exceeded {startup_timeout_s}s")
            if self._proc.poll() is not None:
                stderr_tail = (self._proc.stderr.read() or "")[-1500:] if self._proc.stderr else ""
                raise RuntimeError(f"AnySplat worker exited during startup: {stderr_tail}")
            line = self._proc.stdout.readline()
            if not line:
                continue
            try:
                msg = _json.loads(line.strip())
            except Exception:
                continue
            if msg.get("status") == "ready":
                self._load_s = time.time() - t0
                return

    @property
    def load_seconds(self) -> float:
        return getattr(self, "_load_s", 0.0)

    def inference(self, image_paths: list[Path], output_npz: Path, timeout_s: float = 60.0) -> dict:
        """Run one inference. Returns a dict with the output .npz path plus per-phase
        timing breakdown:

            {"output": Path, "t_ipc_send_ms": ..., "t_ipc_wait_ms": ...,
             "t_images_load_ms": ..., "t_forward_ms": ..., "t_convert_ms": ...,
             "t_npz_save_ms": ...}

        ``ipc_send`` = stdin write+flush. ``ipc_wait`` = round-trip from end-of-send
        to response readline (this includes the entire worker-side cost plus
        stdout buffering / readline blocking). Worker-side phases are reported
        verbatim from the worker's ``_run_one``. They will not sum to ``ipc_wait``
        because the JSON write itself and Python overhead are unmeasured.
        Raises on worker error."""
        import json as _json
        if self._proc.poll() is not None:
            raise RuntimeError("AnySplat persistent worker is no longer running")
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        req = {"images": [str(p) for p in image_paths], "output": str(output_npz)}

        t_send0 = time.time()
        self._proc.stdin.write(_json.dumps(req) + "\n")
        self._proc.stdin.flush()
        t_ipc_send_ms = (time.time() - t_send0) * 1000.0

        t_wait0 = time.time()
        t0 = t_wait0
        while True:
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"AnySplat inference exceeded {timeout_s}s")
            line = self._proc.stdout.readline()
            if not line:
                if self._proc.poll() is not None:
                    stderr_tail = (self._proc.stderr.read() or "")[-1500:] if self._proc.stderr else ""
                    raise RuntimeError(f"AnySplat worker died: {stderr_tail}")
                continue
            try:
                resp = _json.loads(line.strip())
            except Exception:
                continue
            if resp.get("status") == "ok":
                t_ipc_wait_ms = (time.time() - t_wait0) * 1000.0
                return {
                    "output": Path(resp["output"]),
                    "t_ipc_send_ms": t_ipc_send_ms,
                    "t_ipc_wait_ms": t_ipc_wait_ms,
                    "t_images_load_ms": float(resp.get("t_images_load_ms", 0.0)),
                    "t_forward_ms": float(resp.get("t_forward_ms", 0.0)),
                    "t_convert_ms": float(resp.get("t_convert_ms", 0.0)),
                    "t_npz_save_ms": float(resp.get("t_npz_save_ms", 0.0)),
                }
            if resp.get("status") == "error":
                raise RuntimeError(f"AnySplat worker error: {resp.get('msg')}")

    def close(self) -> None:
        if self._proc.poll() is not None:
            return
        try:
            import json as _json
            self._proc.stdin.write(_json.dumps({"cmd": "quit"}) + "\n")
            self._proc.stdin.flush()
            self._proc.wait(timeout=5.0)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass


def run_anysplat_subprocess(
    image_paths: list[Path],
    output_npz: Path,
    *,
    conda_env: str = "anysplat_dynamic_gs",
    timeout_s: float = 300.0,
) -> Path:
    """Spawn the AnySplat worker. Returns the output .npz path on success."""

    if not _WORKER_SCRIPT.exists():
        raise FileNotFoundError(f"AnySplat worker script not found: {_WORKER_SCRIPT}")
    if not _ANYSPLAT_REPO.exists():
        raise FileNotFoundError(f"AnySplat repo not found: {_ANYSPLAT_REPO}")

    output_npz.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "conda", "run", "-n", conda_env, "--no-capture-output",
        "python", str(_WORKER_SCRIPT),
        "--output", str(output_npz),
        "--anysplat-repo", str(_ANYSPLAT_REPO),
    ]
    for p in image_paths:
        cmd.extend(["--image", str(p)])

    # The worker needs $CONDA_PREFIX/lib on LD_LIBRARY_PATH for its native
    # torch_scatter build. ``conda run`` sets CONDA_PREFIX; we prepend the
    # env's lib dir explicitly so torch_scatter_cuda.so resolves.
    env = os.environ.copy()
    env_prefix = Path.home() / "miniconda3" / "envs" / conda_env
    extra_ld = str(env_prefix / "lib")
    env["LD_LIBRARY_PATH"] = (extra_ld + ":" + env.get("LD_LIBRARY_PATH", "")).rstrip(":")

    t0 = time.time()
    result = subprocess.run(
        cmd, env=env, timeout=timeout_s,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(
            f"AnySplat worker failed (exit {result.returncode}, {elapsed:.1f}s):\n"
            f"{result.stdout[-2000:]}"
        )
    if not output_npz.exists():
        raise RuntimeError(f"AnySplat worker did not produce {output_npz}")
    return output_npz


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Umeyama 1991 — closed-form 7-DoF similarity src → dst.

    src, dst : (K, 3) point sets in correspondence. Returns (s, R, t) such that
    ``s * R @ src.T + t.reshape(3,1) ≈ dst.T``. Needs K >= 3 for a unique solution
    (K=2 is degenerate but works with reduced rank handling).
    """

    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.shape[1] != 3:
        raise ValueError(f"src/dst must be (K, 3), got src={src.shape} dst={dst.shape}")
    K = src.shape[0]

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    var_src = (src_c ** 2).sum() / K
    cov = (dst_c.T @ src_c) / K
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    s = (D * np.diag(S)).sum() / max(var_src, 1e-12)
    t = mu_dst - s * R @ mu_src
    return float(s), R.astype(np.float32), t.astype(np.float32)


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Batched wxyz quaternion → (..., 3, 3) rotmat. q assumed unit-norm."""

    q = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.stack([
        1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w),
        2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),
        2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y),
    ], axis=-1).reshape(*q.shape[:-1], 3, 3)
    return R.astype(np.float32)


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Batched (..., 3, 3) rotmat → wxyz quat. Shepperd's method."""

    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    trace = m00 + m11 + m22
    out = np.zeros(R.shape[:-2] + (4,), dtype=np.float32)

    # Use the largest of {trace, m00, m11, m22} to avoid numerical issues.
    cond_t = trace > 0
    s = np.sqrt(np.where(cond_t, trace + 1.0, 1.0)) * 2.0
    out[..., 0] = np.where(cond_t, 0.25 * s, out[..., 0])
    out[..., 1] = np.where(cond_t, (m21 - m12) / s, out[..., 1])
    out[..., 2] = np.where(cond_t, (m02 - m20) / s, out[..., 2])
    out[..., 3] = np.where(cond_t, (m10 - m01) / s, out[..., 3])
    # Fallback branches for non-positive trace
    rest = ~cond_t
    if rest.any():
        idx = np.argmax(np.stack([m00, m11, m22], axis=-1), axis=-1)
        for j in range(3):
            sel = rest & (idx == j)
            if not sel.any():
                continue
            i = j
            k = (j + 1) % 3
            l = (j + 2) % 3
            Rii = R[..., i, i]; Rkk = R[..., k, k]; Rll = R[..., l, l]
            Rki = R[..., k, i]; Rik = R[..., i, k]
            Rlk = R[..., l, k]; Rkl = R[..., k, l]
            Rli = R[..., l, i]; Ril = R[..., i, l]
            s_j = np.sqrt(np.where(sel, Rii - Rkk - Rll + 1.0, 1.0)) * 2.0
            qx = np.zeros_like(s_j); qy = np.zeros_like(s_j); qz = np.zeros_like(s_j); qw = np.zeros_like(s_j)
            qw = np.where(sel, (Rlk - Rkl) / s_j, 0)
            comps = [0, 0, 0]
            comps[i] = 0.25 * s_j
            comps[k] = (Rik + Rki) / s_j
            comps[l] = (Ril + Rli) / s_j
            out[..., 0] = np.where(sel, qw, out[..., 0])
            out[..., 1] = np.where(sel, comps[0], out[..., 1])
            out[..., 2] = np.where(sel, comps[1], out[..., 2])
            out[..., 3] = np.where(sel, comps[2], out[..., 3])
    return out / np.linalg.norm(out, axis=-1, keepdims=True).clip(min=1e-12)


def apply_similarity_to_gaussians(
    *,
    means_canonical: np.ndarray,  # (N, 3)
    log_scales: np.ndarray,       # (N, 3)
    quats_wxyz: np.ndarray,       # (N, 4)
    similarity_s: float,
    similarity_R: np.ndarray,     # (3, 3)
    similarity_t: np.ndarray,     # (3,)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply (s, R, t) similarity to a gaussian set. Returns (means_world, log_scales_world, quats_world)."""

    means_world = (similarity_s * (similarity_R @ means_canonical.T)).T + similarity_t  # (N, 3)
    log_scales_world = log_scales + np.log(max(similarity_s, 1e-12)).astype(np.float32)
    Rmat_g = quat_wxyz_to_rotmat(quats_wxyz)             # (N, 3, 3)
    Rmat_g_world = similarity_R @ Rmat_g                  # (N, 3, 3)
    quats_world = rotmat_to_quat_wxyz(Rmat_g_world)       # (N, 4)
    return means_world.astype(np.float32), log_scales_world.astype(np.float32), quats_world


def _world_to_image_opengl(
    xyz_world: torch.Tensor,    # (N, 3)
    c2w: torch.Tensor,          # (4, 4) OpenGL convention (Nerfstudio)
    fx: float, fy: float, cx: float, cy: float, width: int, height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project world xyz to pixel (u, v) using Nerfstudio OpenGL convention.

    Returns (uv (N, 2), valid (N,) bool). A point is valid iff in front of the
    camera (z_cam < 0 in OpenGL) and the projected (u, v) lies inside the image.
    """

    R_wc = c2w[:3, :3].T  # world → camera rotation = R_c2w^T
    t_wc = -R_wc @ c2w[:3, 3]
    p_cam = (R_wc @ xyz_world.T).T + t_wc  # (N, 3)
    x_c, y_c, z_c = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]
    in_front = z_c < 0  # OpenGL: forward = -z
    u = fx * (x_c / (-z_c).clamp(min=1e-9)) + cx
    v = fy * (-y_c / (-z_c).clamp(min=1e-9)) + cy   # OpenGL: y up → image v down
    uv = torch.stack([u, v], dim=-1)
    in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return uv, in_front & in_image


def reproject_anysplat_to_scene(
    *,
    means_canonical: np.ndarray,        # (N, 3) AnySplat canonical positions
    log_scales: np.ndarray,             # (N, 3) log-scale (canonical)
    quats_wxyz: np.ndarray,             # (N, 4)
    opacity_logits: np.ndarray,         # (N,)
    features_dc: np.ndarray,            # (N, 3)
    features_rest: np.ndarray,          # (N, 15, 3)
    pred_c2w_0: np.ndarray,             # (4, 4) AnySplat's predicted camera 0
    pred_K_norm: np.ndarray,            # (3, 3) AnySplat's predicted intrinsics (normalized, *W or *H)
    pred_image_hw: tuple[int, int],     # (H_any, W_any) AnySplat input image resolution (typically 448, 448)
    sensor_depth_m: np.ndarray,         # (H, W) sensor depth in metres — will be resized to pred_image_hw
    scene_c2w: np.ndarray,              # (4, 4) known scene camera-to-world (OpenGL convention)
    scene_intr: dict,                   # {fl_x, fl_y, cx, cy, w, h} from scene transforms.json
    opacity_min: float = 0.05,
    drop_no_sensor_depth: bool = False,
    background_rgb: tuple[float, float, float] = (0.86, 0.92, 1.0),
    background_tol: float = 0.08,
    global_s_fallback: Optional[float] = None,
    component_mask: Optional[np.ndarray] = None,   # (H, W) bool, restrict insertion to this mask (any resolution)
) -> dict:
    """Canonical AnySplat → scene reprojection (memory: anysplat-reprojection-method).

    Each gaussian is placed at the world point obtained by back-projecting its pred-pixel
    through the SCENE's intrinsics + scene_c2w with sensor depth (per-pixel) — NOT through
    AnySplat's predicted intrinsics. This avoids the ~28% lateral-error pattern from
    AnySplat's wrong focal-length prediction.

    Returns a dict with keys: xyz, features_dc, features_rest, opacities (LOGITS, shape (N, 1)),
    scales (LOG-SCALES, shape (N, 3)), quats (WXYZ, shape (N, 4)) — ready for
    ``model.insert_inpaint_gaussians``.
    """

    H_any, W_any = pred_image_hw
    if not _HAS_CV2:
        raise RuntimeError("cv2 required for sensor depth resize")

    # --- Opacity filter ---
    opac = 1.0 / (1.0 + np.exp(-opacity_logits))
    keep = opac >= opacity_min
    means_canonical = means_canonical[keep]; log_scales = log_scales[keep]; quats_wxyz = quats_wxyz[keep]
    opacity_logits = opacity_logits[keep]; features_dc = features_dc[keep]; features_rest = features_rest[keep]

    # --- Background-color filter ---
    SH_C0 = 0.28209479177387814
    rgb_pred = features_dc * SH_C0 + 0.5
    keep_bg = ~np.all(np.abs(rgb_pred - np.asarray(background_rgb, dtype=np.float32)) <= background_tol, axis=-1)
    means_canonical = means_canonical[keep_bg]; log_scales = log_scales[keep_bg]; quats_wxyz = quats_wxyz[keep_bg]
    opacity_logits = opacity_logits[keep_bg]; features_dc = features_dc[keep_bg]; features_rest = features_rest[keep_bg]
    N = means_canonical.shape[0]
    if N == 0:
        return {"xyz": np.empty((0, 3), dtype=np.float32)}

    # --- Pred-camera position + pred-pixel for each gaussian ---
    pred_c2w_0 = pred_c2w_0.astype(np.float64)
    R_pred = pred_c2w_0[:3, :3]; t_pred = pred_c2w_0[:3, 3]
    p_cam_cv = ((means_canonical - t_pred) @ R_pred).astype(np.float64)
    z_cam = p_cam_cv[:, 2]

    fx_p = pred_K_norm[0, 0] * W_any; fy_p = pred_K_norm[1, 1] * H_any
    cx_p = pred_K_norm[0, 2] * W_any; cy_p = pred_K_norm[1, 2] * H_any
    safe_z = np.where(z_cam > 1e-6, z_cam, 1.0)
    u = fx_p * p_cam_cv[:, 0] / safe_z + cx_p
    v = fy_p * p_cam_cv[:, 1] / safe_z + cy_p
    in_image = (z_cam > 1e-6) & (u >= 0) & (u < W_any) & (v >= 0) & (v < H_any)
    u_idx = np.clip(u.astype(np.int64), 0, W_any - 1)
    v_idx = np.clip(v.astype(np.int64), 0, H_any - 1)

    # --- Sensor depth at pred resolution + per-gauss lookup ---
    sensor_resized = cv2.resize(sensor_depth_m.astype(np.float32), (W_any, H_any), interpolation=cv2.INTER_NEAREST)
    sensor_per_gauss = np.where(in_image, sensor_resized[v_idx, u_idx], 0.0).astype(np.float64)
    valid_sensor = in_image & (sensor_per_gauss > 0.01)

    if drop_no_sensor_depth:
        keep_d = valid_sensor
        means_canonical = means_canonical[keep_d]; log_scales = log_scales[keep_d]; quats_wxyz = quats_wxyz[keep_d]
        opacity_logits = opacity_logits[keep_d]; features_dc = features_dc[keep_d]; features_rest = features_rest[keep_d]
        u = u[keep_d]; v = v[keep_d]; z_cam = z_cam[keep_d]
        d_per_gauss = sensor_per_gauss[keep_d]
        N = means_canonical.shape[0]
    else:
        if global_s_fallback is None:
            global_s_fallback = float(np.median(sensor_per_gauss[valid_sensor] / z_cam[valid_sensor])) if valid_sensor.any() else 1.0
        d_per_gauss = np.where(valid_sensor, sensor_per_gauss, z_cam * global_s_fallback)
    if N == 0:
        return {"xyz": np.empty((0, 3), dtype=np.float32)}

    # --- Optional CDN component mask filter (resized to pred resolution) ---
    if component_mask is not None:
        mask_np = component_mask if isinstance(component_mask, np.ndarray) else np.asarray(component_mask)
        if mask_np.ndim == 3: mask_np = mask_np[..., 0]
        mask_resized = cv2.resize(mask_np.astype(np.uint8), (W_any, H_any), interpolation=cv2.INTER_NEAREST).astype(bool)
        u_idx2 = np.clip(u.astype(np.int64), 0, W_any - 1)
        v_idx2 = np.clip(v.astype(np.int64), 0, H_any - 1)
        keep_c = mask_resized[v_idx2, u_idx2]
        means_canonical = means_canonical[keep_c]; log_scales = log_scales[keep_c]; quats_wxyz = quats_wxyz[keep_c]
        opacity_logits = opacity_logits[keep_c]; features_dc = features_dc[keep_c]; features_rest = features_rest[keep_c]
        u = u[keep_c]; v = v[keep_c]; z_cam = z_cam[keep_c]; d_per_gauss = d_per_gauss[keep_c]
        N = means_canonical.shape[0]
        if N == 0:
            return {"xyz": np.empty((0, 3), dtype=np.float32)}

    # --- Back-project (u, v, d) through SCENE intrinsics, OpenGL convention ---
    fx_s = scene_intr["fl_x"] * W_any / scene_intr["w"]
    fy_s = scene_intr["fl_y"] * H_any / scene_intr["h"]
    cx_s = scene_intr["cx"]   * W_any / scene_intr["w"]
    cy_s = scene_intr["cy"]   * H_any / scene_intr["h"]
    p_cam_gl = np.stack([
        d_per_gauss * (u - cx_s) / fx_s,
        -d_per_gauss * (v - cy_s) / fy_s,
        -d_per_gauss,
    ], axis=-1)
    R_scene = scene_c2w[:3, :3]; t_scene = scene_c2w[:3, 3]
    means_world = (R_scene @ p_cam_gl.T).T + t_scene

    # --- Per-gauss scale (image-space footprint preservation) ---
    safe_z2 = np.where(z_cam > 1e-6, z_cam, 1.0)
    s_per_gauss = d_per_gauss / safe_z2
    log_scales_world = log_scales + np.log(np.clip(s_per_gauss, 1e-9, None))[:, None].astype(np.float32)

    # --- Rotation: canonical-CV → scene-GL basis ---
    M_rot = R_scene @ np.diag([1.0, -1.0, -1.0]) @ R_pred.T
    Rg_can = quat_wxyz_to_rotmat(quats_wxyz).astype(np.float64)
    Rg_world = (M_rot[None, :, :] @ Rg_can).astype(np.float32)
    quats_world = rotmat_to_quat_wxyz(Rg_world).astype(np.float32)

    return {
        "xyz": means_world.astype(np.float32),
        "features_dc": features_dc.astype(np.float32),
        "features_rest": features_rest.astype(np.float32),
        "opacities": opacity_logits.reshape(-1, 1).astype(np.float32),
        "scales": log_scales_world.astype(np.float32),
        "quats": quats_world.astype(np.float32),
    }


def filter_gaussians_by_component_mask(
    *,
    means_world: torch.Tensor,    # (N, 3) on device
    target_camera,                # Nerfstudio single-frame Cameras
    component_mask: torch.Tensor, # (H, W) or (H, W, 1) bool/0-1, at SCENE resolution
) -> torch.Tensor:
    """Return a bool mask (N,) — gaussians whose 2D projection falls inside the component."""

    def _scalar(x):
        if isinstance(x, torch.Tensor):
            return float(x.detach().cpu().reshape(-1)[0].item())
        return float(x)

    fx = _scalar(target_camera.fx); fy = _scalar(target_camera.fy)
    cx = _scalar(target_camera.cx); cy = _scalar(target_camera.cy)
    width = int(_scalar(target_camera.width))
    height = int(_scalar(target_camera.height))
    c2w = target_camera.camera_to_worlds
    if c2w.ndim == 3:
        c2w = c2w[0]
    if c2w.shape == (3, 4):
        bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=c2w.device, dtype=c2w.dtype)
        c2w = torch.cat([c2w, bottom], dim=0)

    uv, valid = _world_to_image_opengl(means_world, c2w, fx, fy, cx, cy, width, height)

    if component_mask.dim() == 3:
        component_mask = component_mask[..., 0]
    comp = component_mask.to(dtype=torch.bool, device=means_world.device)

    out = torch.zeros(means_world.shape[0], dtype=torch.bool, device=means_world.device)
    if valid.any():
        u_idx = uv[valid, 0].long().clamp(0, width - 1)
        v_idx = uv[valid, 1].long().clamp(0, height - 1)
        out[valid] = comp[v_idx, u_idx]
    return out
