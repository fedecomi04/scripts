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

        env_prefix = Path.home() / "miniconda3" / "envs" / conda_env
        env_python = env_prefix / "bin" / "python"
        if not env_python.exists():
            raise FileNotFoundError(
                f"AnySplat env python not found at {env_python}. "
                f"Expected env '{conda_env}' under {env_prefix.parent}."
            )

        # Invoke the env's python directly instead of going through ``conda
        # run`` — same pattern as live_shm_reader._spawn_publisher. This
        # avoids depending on ``conda`` being on PATH (Bash tool environments
        # often don't have it) and skips the ~0.5 s ``conda run`` wrapper
        # overhead. The env's lib dir is prepended to LD_LIBRARY_PATH so the
        # worker's native extensions (torch_scatter_cuda.so etc.) resolve.
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = (str(env_prefix / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")).rstrip(":")
        env["PYTHONUNBUFFERED"] = "1"

        cmd = [
            str(env_python), "-u", str(_WORKER_SCRIPT),
            "--persistent",
            "--anysplat-repo", str(_ANYSPLAT_REPO),
        ]
        self._proc = subprocess.Popen(
            cmd, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        # Generalized IPC handles — pipe mode here; ``adopt()`` builds an
        # instance whose handles are FIFOs and whose ``_proc`` is None.
        self._send_f = self._proc.stdin
        self._recv_f = self._proc.stdout
        self._adopted_pid: int | None = None
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

    def _alive(self) -> bool:
        if self._proc is not None:
            return self._proc.poll() is None
        if self._adopted_pid is not None:
            return _pid_is_anysplat_worker(self._adopted_pid)
        return False

    @classmethod
    def adopt(cls, fifo_dir: Path, wait_ready_timeout_s: float = 60.0) -> "PersistentAnysplatWorker | None":
        """Adopt a worker pre-spawned by :func:`spawn_detached_anysplat_worker`
        in an earlier process (e.g. live capture pre-spawned it right after
        SAM3D so its ~17 s model load overlaps static training).

        Returns None (caller should spawn fresh) when: no spawn record exists,
        the spawned pid died, or ready.json doesn't appear within
        ``wait_ready_timeout_s`` (clock starts only if the pid is alive but
        still loading). Never blocks when there is nothing to wait for.
        """
        import json as _json
        fifo_dir = Path(fifo_dir)
        spawn_file = fifo_dir / "spawn.json"
        ready_file = fifo_dir / "ready.json"
        if not spawn_file.exists():
            return None
        try:
            spawn_info = _json.loads(spawn_file.read_text())
            pid = int(spawn_info["pid"])
        except Exception:
            return None

        # Wait for ready.json while (and only while) the worker pid is alive.
        t0 = time.time()
        while not ready_file.exists():
            if not _pid_is_anysplat_worker(pid):
                return None  # worker died during load (or pid recycled)
            if time.time() - t0 > wait_ready_timeout_s:
                return None
            time.sleep(0.25)

        try:
            ready = _json.loads(ready_file.read_text())
            if int(ready.get("pid", -1)) != pid:
                return None  # stale ready.json from an older worker
        except Exception:
            return None
        if not _pid_is_anysplat_worker(pid):
            return None

        # Connect. Open-order protocol mirrors the worker: cmd first, then
        # res. The worker blocks in open(cmd, 'r'); our open(cmd, 'w')
        # completes the rendezvous, then both sides open res.
        cmd_fifo = fifo_dir / "cmd.fifo"
        res_fifo = fifo_dir / "res.fifo"
        try:
            send_f = open(cmd_fifo, "w", buffering=1)
            recv_f = open(res_fifo, "r")
        except OSError:
            return None

        w = cls.__new__(cls)
        w._proc = None
        w._send_f = send_f
        w._recv_f = recv_f
        w._adopted_pid = pid
        w._load_s = float(ready.get("load_seconds", 0.0))
        return w

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
        if not self._alive():
            raise RuntimeError("AnySplat persistent worker is no longer running")
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        req = {"images": [str(p) for p in image_paths], "output": str(output_npz)}

        t_send0 = time.time()
        self._send_f.write(_json.dumps(req) + "\n")
        self._send_f.flush()
        t_ipc_send_ms = (time.time() - t_send0) * 1000.0

        t_wait0 = time.time()
        t0 = t_wait0
        while True:
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"AnySplat inference exceeded {timeout_s}s")
            line = self._recv_f.readline()
            if not line:
                if not self._alive():
                    stderr_tail = ""
                    if self._proc is not None and self._proc.stderr:
                        stderr_tail = (self._proc.stderr.read() or "")[-1500:]
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
        if not self._alive():
            return
        try:
            import json as _json
            self._send_f.write(_json.dumps({"cmd": "quit"}) + "\n")
            self._send_f.flush()
            if self._proc is not None:
                self._proc.wait(timeout=5.0)
            elif self._adopted_pid is not None:
                # Detached worker: give the quit a moment to land, then
                # targeted-kill the verified pid if it lingers (NEVER a
                # pattern kill — verify cmdline first via _alive()).
                t0 = time.time()
                while time.time() - t0 < 5.0 and self._alive():
                    time.sleep(0.2)
                if self._alive():
                    os.kill(self._adopted_pid, 9)
        except Exception:
            try:
                if self._proc is not None:
                    self._proc.kill()
                elif self._adopted_pid is not None and self._alive():
                    os.kill(self._adopted_pid, 9)
            except Exception:
                pass
        finally:
            for f in (self._send_f, self._recv_f):
                try:
                    f.close()
                except Exception:
                    pass


def _pid_is_anysplat_worker(pid: int) -> bool:
    """True iff ``pid`` is alive AND its cmdline is our anysplat worker
    script. The cmdline check prevents acting on a recycled pid."""
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode()
    except OSError:
        return False
    return "anysplat_worker.py" in cmdline


def spawn_detached_anysplat_worker(
    fifo_dir: Path,
    conda_env: str = "anysplat_dynamic_gs",
) -> int:
    """Fire-and-forget spawn of the FIFO-mode AnySplat worker.

    Called by the live capture session right after SAM3D finishes so the
    worker's model load (measured 16.9 s / 3.5 GB VRAM on this machine)
    overlaps the operator sweep + static training instead of stalling the
    dynamic pipeline's startup. The dynamic pipeline adopts it via
    :meth:`PersistentAnysplatWorker.adopt`.

    Returns the worker pid immediately; the load completes in the background
    and ``<fifo_dir>/ready.json`` appears when it's done. Any previous worker
    recorded in this dir is closed first (verified by cmdline, then killed).
    """
    fifo_dir = Path(fifo_dir)
    fifo_dir.mkdir(parents=True, exist_ok=True)

    import json as _json
    # Replace a stale/previous worker for this dataset dir.
    spawn_file = fifo_dir / "spawn.json"
    if spawn_file.exists():
        try:
            old_pid = int(_json.loads(spawn_file.read_text()).get("pid", -1))
            if _pid_is_anysplat_worker(old_pid):
                os.kill(old_pid, 9)
        except Exception:
            pass
    for name in ("ready.json", "spawn.json"):
        try:
            (fifo_dir / name).unlink()
        except OSError:
            pass
    for name in ("cmd.fifo", "res.fifo"):
        p = fifo_dir / name
        if not p.exists():
            os.mkfifo(p)

    env_prefix = Path.home() / "miniconda3" / "envs" / conda_env
    env_python = env_prefix / "bin" / "python"
    if not env_python.exists():
        raise FileNotFoundError(f"AnySplat env python not found at {env_python}")
    if not _WORKER_SCRIPT.exists():
        raise FileNotFoundError(f"AnySplat worker script not found: {_WORKER_SCRIPT}")

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (str(env_prefix / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")).rstrip(":")
    env["PYTHONUNBUFFERED"] = "1"

    log_f = open(fifo_dir / "worker.log", "a")
    proc = subprocess.Popen(
        [str(env_python), "-u", str(_WORKER_SCRIPT),
         "--fifo-dir", str(fifo_dir),
         "--anysplat-repo", str(_ANYSPLAT_REPO)],
        env=env,
        stdin=subprocess.DEVNULL, stdout=log_f, stderr=log_f,
        start_new_session=True,  # survives the spawning process
    )
    log_f.close()  # child holds its own fd
    spawn_file.write_text(_json.dumps({"pid": proc.pid, "ts": time.time()}))
    return proc.pid


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

    env_prefix = Path.home() / "miniconda3" / "envs" / conda_env
    env_python = env_prefix / "bin" / "python"
    if not env_python.exists():
        raise FileNotFoundError(f"AnySplat env python not found at {env_python}")

    cmd = [
        str(env_python), "-u", str(_WORKER_SCRIPT),
        "--output", str(output_npz),
        "--anysplat-repo", str(_ANYSPLAT_REPO),
    ]
    for p in image_paths:
        cmd.extend(["--image", str(p)])

    # The worker needs $CONDA_PREFIX/lib on LD_LIBRARY_PATH for its native
    # torch_scatter build. Invoking the env's python directly bypasses
    # ``conda run`` (avoids depending on conda being on PATH and saves ~0.5 s).
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (str(env_prefix / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")).rstrip(":")
    env["PYTHONUNBUFFERED"] = "1"

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


def icp_refine_scene_c2w(
    *,
    sensor_depth_m: np.ndarray,        # (H, W) sensor depth in metres
    scene_c2w: np.ndarray,             # (4, 4) initial scene c2w (OpenGL)
    scene_intr: dict,                  # {fl_x, fl_y, cx, cy, w, h}
    target_xyz_gpu: "torch.Tensor",    # (M, 3) GPU tensor, frustum-culled world points
    max_iters: int = 30,
    max_dist_m: float = 0.02,
    stride: int = 4,
    min_pts: int = 1000,
) -> tuple[np.ndarray, dict]:
    """GPU point-to-plane ICP of the live sensor cloud against a caller-supplied
    (frustum-culled) target tensor on GPU. Component-agnostic.

    Source = sensor depth back-projected through scene intrinsics + scene_c2w.
    Target = ``target_xyz_gpu``.
    Both go to Open3D's CUDA tensor API. Returns (refined_c2w, info_dict).
    """
    info: dict = {"ran": False, "n_src": 0, "n_tgt": int(target_xyz_gpu.shape[0])}
    if target_xyz_gpu.shape[0] < min_pts:
        return scene_c2w.astype(np.float64), info

    import open3d as o3d
    import open3d.core as o3c

    # --- Build source cloud on GPU ---
    dev = target_xyz_gpu.device
    H_s, W_s = sensor_depth_m.shape[:2]
    fx = float(scene_intr["fl_x"]); fy = float(scene_intr["fl_y"])
    cx = float(scene_intr["cx"]);   cy = float(scene_intr["cy"])

    depth_t = torch.from_numpy(sensor_depth_m.astype(np.float32)).to(dev)
    depth_t = depth_t[::stride, ::stride]
    H_sub, W_sub = depth_t.shape
    vv_t, uu_t = torch.meshgrid(
        torch.arange(0, H_sub, device=dev, dtype=torch.float32) * stride,
        torch.arange(0, W_sub, device=dev, dtype=torch.float32) * stride,
        indexing="ij",
    )
    valid_t = depth_t > 0.01
    if int(valid_t.sum().item()) < min_pts:
        return scene_c2w.astype(np.float64), info
    d_v = depth_t[valid_t]
    u_v = uu_t[valid_t]
    v_v = vv_t[valid_t]
    # OpenGL camera-frame (y up, z back)
    p_cam_t = torch.stack([
        d_v * (u_v - cx) / fx,
        -d_v * (v_v - cy) / fy,
        -d_v,
    ], dim=-1)
    c2w_t = torch.as_tensor(scene_c2w.astype(np.float32), device=dev)
    R_t = c2w_t[:3, :3]; t_t = c2w_t[:3, 3]
    src_world_t = p_cam_t @ R_t.T + t_t  # (N_src, 3) on GPU
    info["n_src"] = int(src_world_t.shape[0])

    # --- Hand off to Open3D Tensor API on CUDA ---
    cuda_dev = o3c.Device("CUDA:0") if "cuda" in str(dev) else o3c.Device("CPU:0")
    src_t = o3d.t.geometry.PointCloud(cuda_dev)
    src_t.point.positions = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(src_world_t.contiguous()))

    tgt_t = o3d.t.geometry.PointCloud(cuda_dev)
    tgt_t.point.positions = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(target_xyz_gpu.contiguous().to(torch.float32)))
    # Estimate normals on the target (point-to-plane needs them)
    tgt_t.estimate_normals(max_nn=30, radius=0.02)

    init = o3c.Tensor(np.eye(4, dtype=np.float64), o3c.Dtype.Float64, cuda_dev)
    reg = o3d.t.pipelines.registration.icp(
        source=src_t,
        target=tgt_t,
        max_correspondence_distance=float(max_dist_m),
        init_source_to_target=init,
        estimation_method=o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iters)),
    )
    T = reg.transformation.cpu().numpy().astype(np.float64)
    info.update({
        "ran": True,
        "fitness": float(reg.fitness),
        "inlier_rmse": float(reg.inlier_rmse),
    })
    refined = (T @ scene_c2w).astype(np.float64)
    return refined, info


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
    voxel_dedup_m: Optional[float] = None,         # if set, pick ONE representative per voxel of this size (metres). 0.002 matches static-phase TSDF.
    scale_multiplier: float = 1.0,                 # additionally enlarges each gaussian's three log-scales by log(scale_multiplier).
    scene_crop: Optional[tuple] = None,            # (left, top, size): the SQUARE scene sub-window that was fed to AnySplat. When set, the pred-crop pixel maps back via this window (NOT process_image's full-frame center-crop).
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
    # --- Map each pred-crop pixel (u, v) BACK to the full scene pixel ---
    # AnySplat's process_image() aspect-preserve-resizes so the SHORTER side =
    # W_any (= H_any = 448), then CENTER-CROPS to (W_any, H_any). Invert that here
    # so the depth lookup + back-projection below use TRUE scene pixels. The old
    # full-frame-squash mapping (`scene_intr * W_any / scene_w`) only matched a
    # SQUARE scene (the 800x800 era); at 1920x1200 it used the wrong x-scale
    # (448/1920 instead of 448/1200) and dropped the center-crop offset, so every
    # insert was mapped sideways -> the "ghost" copies offset next to objects.
    if scene_crop is not None:
        # The caller pre-cropped the scene to a SQUARE window (left, top, size)
        # and fed THAT to AnySplat (process_image then just resizes size->W_any,
        # no further crop). Invert: scene_x = u * size/W_any + left.
        cw_left, cw_top, cw_size = int(scene_crop[0]), int(scene_crop[1]), int(scene_crop[2])
        u_scene = u * (float(cw_size) / W_any) + cw_left
        v_scene = v * (float(cw_size) / H_any) + cw_top
    else:
        sc_w = float(scene_intr["w"]); sc_h = float(scene_intr["h"])
        crop_scale = float(W_any) / min(sc_w, sc_h)
        new_w = W_any if sc_w <= sc_h else int(sc_w * crop_scale)
        new_h = H_any if sc_h <= sc_w else int(sc_h * crop_scale)
        crop_left = (new_w - W_any) // 2
        crop_top  = (new_h - H_any) // 2
        u_scene = (u + crop_left) / crop_scale
        v_scene = (v + crop_top)  / crop_scale

    # --- Sensor depth at FULL scene resolution + per-gauss lookup ---
    Hs, Ws = sensor_depth_m.shape[:2]
    sensor_full = sensor_depth_m.astype(np.float32)
    us_idx = np.clip(np.round(u_scene).astype(np.int64), 0, Ws - 1)
    vs_idx = np.clip(np.round(v_scene).astype(np.int64), 0, Hs - 1)
    sensor_per_gauss = np.where(in_image, sensor_full[vs_idx, us_idx], 0.0).astype(np.float64)
    valid_sensor = in_image & (sensor_per_gauss > 0.01)

    if drop_no_sensor_depth:
        keep_d = valid_sensor
        means_canonical = means_canonical[keep_d]; log_scales = log_scales[keep_d]; quats_wxyz = quats_wxyz[keep_d]
        opacity_logits = opacity_logits[keep_d]; features_dc = features_dc[keep_d]; features_rest = features_rest[keep_d]
        u_scene = u_scene[keep_d]; v_scene = v_scene[keep_d]; z_cam = z_cam[keep_d]
        d_per_gauss = sensor_per_gauss[keep_d]
        N = means_canonical.shape[0]
    else:
        if global_s_fallback is None:
            global_s_fallback = float(np.median(sensor_per_gauss[valid_sensor] / z_cam[valid_sensor])) if valid_sensor.any() else 1.0
        d_per_gauss = np.where(valid_sensor, sensor_per_gauss, z_cam * global_s_fallback)
    if N == 0:
        return {"xyz": np.empty((0, 3), dtype=np.float32)}

    # --- Optional CDN component mask filter (indexed at SCENE resolution) ---
    if component_mask is not None:
        mask_np = component_mask if isinstance(component_mask, np.ndarray) else np.asarray(component_mask)
        if mask_np.ndim == 3: mask_np = mask_np[..., 0]
        mask_bool = mask_np.astype(np.uint8) > 0
        if mask_bool.shape[:2] != (Hs, Ws):
            mask_bool = cv2.resize(mask_bool.astype(np.uint8), (Ws, Hs), interpolation=cv2.INTER_NEAREST).astype(bool)
        usc = np.clip(np.round(u_scene).astype(np.int64), 0, Ws - 1)
        vsc = np.clip(np.round(v_scene).astype(np.int64), 0, Hs - 1)
        keep_c = mask_bool[vsc, usc]
        means_canonical = means_canonical[keep_c]; log_scales = log_scales[keep_c]; quats_wxyz = quats_wxyz[keep_c]
        opacity_logits = opacity_logits[keep_c]; features_dc = features_dc[keep_c]; features_rest = features_rest[keep_c]
        u_scene = u_scene[keep_c]; v_scene = v_scene[keep_c]; z_cam = z_cam[keep_c]; d_per_gauss = d_per_gauss[keep_c]
        N = means_canonical.shape[0]
        if N == 0:
            return {"xyz": np.empty((0, 3), dtype=np.float32)}

    # --- Back-project (u_scene, v_scene, d) through FULL SCENE intrinsics ---
    fx_s = float(scene_intr["fl_x"]); fy_s = float(scene_intr["fl_y"])
    cx_s = float(scene_intr["cx"]);   cy_s = float(scene_intr["cy"])
    p_cam_gl = np.stack([
        d_per_gauss * (u_scene - cx_s) / fx_s,
        -d_per_gauss * (v_scene - cy_s) / fy_s,
        -d_per_gauss,
    ], axis=-1)
    R_scene = scene_c2w[:3, :3]; t_scene = scene_c2w[:3, 3]
    means_world = (R_scene @ p_cam_gl.T).T + t_scene

    # --- Per-gauss scale (image-space footprint preservation) ---
    safe_z2 = np.where(z_cam > 1e-6, z_cam, 1.0)
    s_per_gauss = d_per_gauss / safe_z2
    log_scales_world = log_scales + np.log(np.clip(s_per_gauss, 1e-9, None))[:, None].astype(np.float32)
    if scale_multiplier != 1.0:
        log_scales_world = log_scales_world + np.float32(np.log(max(float(scale_multiplier), 1e-12)))

    # --- Rotation: canonical-CV → scene-GL basis ---
    M_rot = R_scene @ np.diag([1.0, -1.0, -1.0]) @ R_pred.T
    Rg_can = quat_wxyz_to_rotmat(quats_wxyz).astype(np.float64)
    Rg_world = (M_rot[None, :, :] @ Rg_can).astype(np.float32)
    quats_world = rotmat_to_quat_wxyz(Rg_world).astype(np.float32)

    # --- Optional voxel dedup: pick ONE representative per voxel (no averaging) ---
    # np.unique on the integer voxel index returns the FIRST occurrence per
    # voxel, which is what we want — no quaternion averaging, no logit
    # averaging, no SH averaging.  Simplest correct reduction.
    if voxel_dedup_m is not None and voxel_dedup_m > 0.0 and means_world.shape[0] > 0:
        voxel_idx = np.floor(means_world / float(voxel_dedup_m)).astype(np.int64)
        # np.unique on rows of shape (N, 3); return_index gives the FIRST row
        # per unique voxel.  Sorting cost is O(N log N) — cheap for N ~ 1e5.
        _, keep_idx = np.unique(voxel_idx, axis=0, return_index=True)
        keep_idx.sort()  # preserve original order so any later debug print stays stable
        means_world = means_world[keep_idx]
        log_scales_world = log_scales_world[keep_idx]
        quats_world = quats_world[keep_idx]
        features_dc = features_dc[keep_idx]
        features_rest = features_rest[keep_idx]
        opacity_logits = opacity_logits[keep_idx]

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
