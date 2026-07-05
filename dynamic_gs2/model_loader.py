"""model_loader.py — the ONE place every "load model X" lives (static_phase.md §5).

A uniform prewarm registry so the static orchestrator never `import`s a model
directly: every heavy model is a handle with the same async-prewarm interface
(prewarm = load on a bg thread, non-blocking; wait_ready = block until loaded;
release = free VRAM). The residency policy per model (static_phase.md §2c) is
encoded here, not scattered across the stage modules.

PREWARM IS REAL (not stubs): FastSAM + SAM3D share ONE persistent SamWorkerClient
(the proven dynamic_gs.utils.sam_worker, spawn-once load-on-demand JSON-over-pipe,
CLAUDE.md Invariant #7). prewarm() spawns the worker + loads the model on a bg thread
DURING the sweep, so the trigger pays only inference (FastSAM ~1 s, SAM3D ~10 s) instead
of the ~10 s + ~48 s cold import+checkpoint-load that subprocess-per-call paid. If the
worker can't spawn/load, each handle FALLS BACK to the old subprocess path (never worse).

  - FastSAM   : ~0.85 GB, loaded in the shared worker during the sweep, unloaded after segment.
  - SAM3D     : ~7.3 GB trimmed, loaded in the SAME worker during the sweep (co-resident with
                FastSAM + Gazebo fits 16 GB); inference at the trigger; unloaded after, freeing
                VRAM for the TSDF seed + splatfacto.
  - AnySplat  : persistent worker (reuses dynamic_ff_backends.AnysplatHandle); prewarmed at
                sweep-start so its ~17 s load overlaps the sweep, not the train.
  - XFeat+LG  : tiny in-process; built on a bg thread for the dynamic loop.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Optional


# ----------------------------------------------------------------- timing sink
# Optional global TimingLedger so the bg PREWARM threads can record their actual model-LOAD wall
# time (prewarm() is non-blocking -> a tm.stage() around the call only times the dispatch, not the
# load). run_static calls set_timing_ledger(tm); each _go() records "load.<model>" via record_ms.
_TM = None


def set_timing_ledger(tm) -> None:
    global _TM
    _TM = tm


def _record_load(name: str, seconds: float) -> None:
    if _TM is not None:
        try:
            _TM.record_ms(f"load.{name}", seconds * 1000.0)
        except Exception:
            pass


# ----------------------------------------------------------------- shared SAM worker
class _SharedSamWorker:
    """One persistent SamWorkerClient shared by FastSamHandle + Sam3dHandle, so FastSAM and
    SAM3D load into the SAME process (one spawn, concurrent loads) during the sweep. Thread-safe
    lazy spawn; tolerant of a failed spawn (callers fall back to the per-call subprocess)."""

    def __init__(self, conda_env: str = "sam3_dynamic_gs"):
        self._conda_env = conda_env
        self._client = None
        self._spawn_failed = False
        self._lock = threading.Lock()

    def client(self):
        """Spawn-once the worker (blocking ~1.3 s) and return it, or None if spawn failed
        (caller then uses the subprocess fallback). Safe to call from multiple threads."""
        with self._lock:
            if self._client is not None or self._spawn_failed:
                return self._client
            try:
                import os
                # SAM3D's third_party/Fast-SAM3D/notebook/inference.py does
                # `os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]` at import — but the mode scripts
                # launches via bare env-python so CONDA_PREFIX is unset -> KeyError in the worker.
                # The worker runs in the sam3 env, so set CONDA_PREFIX to that env's prefix; the
                # SamWorkerClient spawns with os.environ.copy(), so the worker inherits it. (Matches
                # the proven per-call SAM3D subprocess, which pins CONDA_PREFIX itself.)
                _sam_prefix = os.path.expanduser(f"~/miniconda3/envs/{self._conda_env}")
                os.environ.setdefault("CONDA_PREFIX", _sam_prefix)
                from .sam_worker import SamWorkerClient
                self._client = SamWorkerClient(conda_env=self._conda_env)
            except Exception as e:
                print(f"[model_loader] SAM worker spawn failed ({e}); using subprocess fallback", flush=True)
                self._spawn_failed = True
            return self._client

    def close(self):
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None


# ----------------------------------------------------------------- FastSAM
class FastSamHandle:
    """FastSAM+CLIP text segmenter. prewarm() loads FastSAM into the shared persistent worker
    DURING the sweep (so segment() at the trigger is warm ~1 s, not a ~10 s cold subprocess).
    Falls back to run_fastsam_subprocess if the worker is unavailable."""

    def __init__(self, worker: _SharedSamWorker, conda_env: str = "sam3_dynamic_gs"):
        self._worker = worker
        self.conda_env = conda_env
        self._thread: Optional[threading.Thread] = None
        self._loaded = threading.Event()

    def prewarm(self) -> None:
        """Spawn the shared worker + load FastSAM on a bg thread (non-blocking). Idempotent."""
        if self._thread is not None:
            return
        def _go():
            try:
                c = self._worker.client()
                if c is not None:
                    s = c.load_fastsam()
                    _record_load("fastsam", s)
                    print(f"[model_loader] FastSAM pre-loaded in worker ({s:.1f}s)", flush=True)
            except Exception as e:
                print(f"[model_loader] FastSAM prewarm failed (will subprocess lazily): {e}", flush=True)
            finally:
                self._loaded.set()
        self._thread = threading.Thread(target=_go, name="fastsam-prewarm", daemon=True)
        self._thread.start()

    def wait_ready(self) -> None:
        if self._thread is not None:
            self._loaded.wait()

    def segment(self, image_path: Path, text_prompt: str, output_dir: Path,
                output_stem: str, **filter_kwargs) -> List[dict]:
        """Warm inference via the shared worker; subprocess fallback if no worker."""
        self.wait_ready()
        c = self._worker.client()
        if c is not None:
            try:
                return c.fastsam_infer(image_path=Path(image_path), text_prompt=text_prompt,
                                       output_dir=Path(output_dir), output_stem=output_stem,
                                       **filter_kwargs)
            except Exception as e:
                print(f"[model_loader] FastSAM worker infer failed ({e}); subprocess fallback", flush=True)
        from .fastsam_segmentation import run_fastsam_subprocess
        return run_fastsam_subprocess(
            image_path=Path(image_path), text_prompt=text_prompt,
            output_dir=Path(output_dir), output_stem=output_stem,
            sam3_conda_env=self.conda_env, **filter_kwargs)

    def release(self) -> None:
        c = self._worker.client()
        if c is not None:
            try:
                c.unload_fastsam()
            except Exception:
                pass


# ----------------------------------------------------------------- SAM3D
class Sam3dHandle:
    """SAM3D 3D-object generation. prewarm() loads SAM3D into the SAME shared worker DURING the
    sweep (so generate() at the trigger pays only ~10 s inference, not the ~11 s import + ~36 s
    checkpoint-load a cold subprocess paid). Falls back to run_sam3d_multi_object_subprocess."""

    def __init__(self, worker: _SharedSamWorker, conda_env: str = "sam3_dynamic_gs",
                 no_trim: bool = False):
        self._worker = worker
        self.conda_env = conda_env
        self.no_trim = no_trim
        self._thread: Optional[threading.Thread] = None
        self._loaded = threading.Event()

    def prewarm(self) -> None:
        """Spawn the shared worker (if needed) + load SAM3D on a bg thread (non-blocking).
        SAM3D's ~48 s import+load now overlaps the operator sweep instead of the trigger."""
        if self._thread is not None:
            return
        def _go():
            try:
                c = self._worker.client()
                if c is not None:
                    s = c.load_sam3d()
                    _record_load("sam3d", s)
                    print(f"[model_loader] SAM3D pre-loaded in worker ({s:.1f}s)", flush=True)
            except Exception as e:
                print(f"[model_loader] SAM3D prewarm failed (will subprocess lazily): {e}", flush=True)
            finally:
                self._loaded.set()
        self._thread = threading.Thread(target=_go, name="sam3d-prewarm", daemon=True)
        self._thread.start()

    def wait_ready(self) -> None:
        if self._thread is not None:
            self._loaded.wait()

    def generate(self, *, render_image_path: Path, object_mask_paths: List[Path],
                 output_dir: Path, output_stems: List[str],
                 depth_path: Optional[Path] = None,
                 intrinsics_path: Optional[Path] = None) -> List[dict]:
        """Warm inference via the shared worker; subprocess fallback if no worker."""
        self.wait_ready()
        c = self._worker.client()
        if c is not None:
            try:
                return c.sam3d_infer(
                    render_image_path=Path(render_image_path),
                    object_mask_paths=[Path(p) for p in object_mask_paths],
                    output_dir=Path(output_dir), output_stems=list(output_stems),
                    image_dir=Path(output_dir),
                    depth_path=Path(depth_path) if depth_path else None,
                    intrinsics_path=Path(intrinsics_path) if intrinsics_path else None)
            except Exception as e:
                print(f"[model_loader] SAM3D worker infer failed ({e}); subprocess fallback", flush=True)
        from .sam3d import run_sam3d_multi_object_subprocess
        return run_sam3d_multi_object_subprocess(
            render_image_path=Path(render_image_path),
            object_mask_paths=[Path(p) for p in object_mask_paths],
            output_dir=Path(output_dir), output_stems=list(output_stems),
            image_dir=Path(output_dir),
            depth_path=Path(depth_path) if depth_path else None,
            intrinsics_path=Path(intrinsics_path) if intrinsics_path else None)

    def release(self) -> None:
        """Unload SAM3D from the worker, freeing its ~7.3 GB for the TSDF seed + splatfacto."""
        c = self._worker.client()
        if c is not None:
            try:
                c.unload_sam3d()
            except Exception:
                pass


# ----------------------------------------------------------------- AnySplat
class AnysplatPrewarmHandle:
    """Persistent AnySplat worker for the DYNAMIC phase. prewarm() spawns+loads it on a bg
    thread; started at SWEEP-START so its ~17 s load overlaps the sweep (not the train, where
    it previously raced the gsplat compile + timed out). Wraps dynamic_ff_backends.AnysplatHandle."""

    def __init__(self, device, conda_env: str = "anysplat_dynamic_gs"):
        from .dynamic_ff_backends import AnysplatHandle
        self._h = AnysplatHandle(device, conda_env=conda_env)

    def prewarm(self) -> None:
        self._h.prewarm()                 # spawn+load+self-warm on a bg thread (non-blocking)
        # Record the actual load wall-time (the subprocess load is opaque to us): a tiny watcher thread
        # blocks on wait_ready and stamps load.anysplat when the worker reports ready.
        t0 = time.time()
        def _watch():
            try:
                self._h.wait_ready()
                _record_load("anysplat", time.time() - t0)
            except Exception:
                pass
        threading.Thread(target=_watch, name="anysplat-load-timer", daemon=True).start()

    def wait_ready(self) -> None:
        self._h.wait_ready()              # block until the worker has finished loading

    def handle(self):
        return self._h                    # hand the live AnysplatHandle to the dynamic loop

    def release(self) -> None:
        self._h.close()
    # AnySplat is a subprocess; "GPU-resident once up". The dynamic loop adopts handle()
    # directly — no CPU-park API (it's spawned-or-not).


# ----------------------------------------------------------------- XFeat + LighterGlue
class XFeatHandle:
    """XFeat+LighterGlue tracker for the DYNAMIC phase. Tiny; built in-process on a bg
    thread during the static phase so the dynamic loop starts warm. Wraps the proven
    dynamic_gs2.dynamic_track.XFeatTracker."""

    def __init__(self, device, tracker_cfg, pose_filter_cfg):
        self._device = device
        self._tracker_cfg = tracker_cfg
        self._pose_filter_cfg = pose_filter_cfg
        self._tracker = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def prewarm(self) -> None:
        if self._thread is not None or self._tracker is not None:
            return
        def _go():
            try:
                t0 = time.time()
                self._build()
                _record_load("xfeat", time.time() - t0)
            except Exception as e:
                print(f"[model_loader] XFeat prewarm failed (will build lazily): {e}", flush=True)
        self._thread = threading.Thread(target=_go, name="xfeat-prewarm", daemon=True)
        self._thread.start()

    def _build(self):
        with self._lock:
            if self._tracker is None:
                from .dynamic_track import XFeatTracker
                self._tracker = XFeatTracker(self._device, self._tracker_cfg, self._pose_filter_cfg)

    def wait_ready(self) -> None:
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._tracker is None:
            self._build()

    def tracker(self):
        self.wait_ready()
        return self._tracker

    def release(self) -> None:
        ...
    # XFeat is tiny + reused every dynamic tick; it stays resident for the live loop.


# ----------------------------------------------------------------- registry
def build_registry(cfg, device, *, want_anysplat: bool = True,
                   want_xfeat: bool = True) -> dict:
    """Construct every model handle the static phase needs, in ONE place. Returns a dict
    keyed by name; the orchestrator calls prewarm()/wait_ready()/generate() at the schedule
    points. FastSAM + SAM3D share one persistent SamWorkerClient (key '_sam_worker' so the
    orchestrator can close it after fusion). AnySplat + XFeat are the dynamic-phase models."""
    sam_worker = _SharedSamWorker()
    reg: dict = {
        "_sam_worker": sam_worker,
        "fastsam": FastSamHandle(sam_worker),
        "sam3d": Sam3dHandle(sam_worker, no_trim=bool(cfg.segmentation.sam3d_no_trim)),
    }
    if want_anysplat:
        reg["anysplat"] = AnysplatPrewarmHandle(device)
    if want_xfeat:
        reg["xfeat"] = XFeatHandle(device, cfg.tracker, cfg.pose_filter)
    return reg
