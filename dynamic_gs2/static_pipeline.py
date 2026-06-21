"""static_pipeline.py — the static-phase orchestrator (the §1 schedule + state machine).

Owns the SWEEP -> TRIGGER -> SAM3D -> TRAIN -> FUSE -> DONE schedule, the red-box trigger
UI, the per-bulletpoint timing (one timed stage per §1 line, rendered FF-report-style to
timing_report_static.txt), and the prewarm wiring. Each heavy stage is a single-responsibility
module (static_seed / static_segment / static_sam3d / static_fuse) and a model handle in the
prewarm registry (model_loader); this file only SEQUENCES them and hands the GaussianSet the
exported warm-cache. (rewrite_spec/static_phase.md §1/§4.)

ONE source-agnostic entry point:
  run_static(source_kind=...) : the deep stages (segment / SAM3D / seed / train / Phase-0b fuse)
      are identical no matter where the frames come from; ONLY the front end differs.
        live_bridge / ros1 : SWEEP the SHM stream into the red-box trigger UI + record static_scene/.
        replay (recorded)  : NO UI, NO sweep — static_scene/ is already on disk; anchor = the last
            keyframe (or trigger_frame). Prints phase boundaries instead of the viser checklist.
      Returns the dynamic_gs2 static_state.pt path, OR (path, (sm,gset,lock), registry) when
      return_scene=True (the single-process hand-off into the dynamic loop).

DEAD-TIME headline is measured from the 'sam3d_done' event to the last stage end.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import timing as _T
from .frame import Frame, Intrinsics


# ----------------------------------------------------------------- recorded path 
def _anchor_from_disk(data_dir, cfg, trigger_frame=None):
    """Build (Frame, Intrinsics) from a recorded static keyframe — the frame Phase-0a segments +
    SAM3D sees. DEFAULT = the LAST keyframe (the operator's final head-on view), exactly as the old
    phase0a used cached_train[-1]. trigger_frame=N instead picks the N-th keyframe (1-based, capture
    order) — the recorded analogue of pressing the live trigger on a specific frame.

    Frames are sorted by file_path STRING (NOT trailing integer) to match nerfstudio's dataparser.
    The capture convention prefixes the pre-motion lead-in frames with 'aa_' SO THEY SORT FIRST,
    leaving the last SWEEP keyframe (arm_*) as the anchor. A trailing-int sort breaks that
    (aa_lead_00029 would sort after arm_00026) -> a near/oblique lead-in -> undersized fused object."""
    import json, cv2
    from .frame import Frame, Intrinsics
    st = Path(data_dir) / "static_scene"
    meta = json.loads((st / "transforms.json").read_text())
    frames = sorted(meta["frames"], key=lambda f: f["file_path"])
    if trigger_frame is None:
        fr = frames[-1]
    else:
        fr = frames[max(0, min(len(frames) - 1, int(trigger_frame) - 1))]   # 1-based, clamped
    rgb = cv2.imread(str(st / fr["file_path"].lstrip("./")), cv2.IMREAD_COLOR)
    dp = fr.get("depth_file_path") or fr["file_path"].replace("rgb", "depth").replace(".png", ".tiff")
    d = cv2.imread(str(st / dp.lstrip("./")), cv2.IMREAD_UNCHANGED)
    depth_m = (d.astype(np.float32) * 1e-3) if d is not None else np.zeros((meta["h"], meta["w"]), np.float32)
    mp = fr.get("mask_path")
    m = cv2.imread(str(st / mp.lstrip("./")), cv2.IMREAD_GRAYSCALE) if mp else None
    mask = (m > 0).astype(np.uint8) if m is not None else np.ones(depth_m.shape, np.uint8)
    intr = Intrinsics(width=int(meta["w"]), height=int(meta["h"]), fx=meta["fl_x"], fy=meta["fl_y"],
                      cx=meta["cx"], cy=meta["cy"])
    frame = Frame(seq=1, stamp_sec=0.0, rgb_bgr=rgb, depth_m=depth_m, mask_keep=mask,
                  c2w_4x4=np.asarray(fr["transform_matrix"], dtype=np.float64))
    return frame, intr


# ----------------------------------------------------------------- deferred TSDF seed
def _build_seed_deferred(static_dir) -> None:
    """Build depth_camera_init_points.ply via the CPU TSDF in a SUBPROCESS at 3 mm voxel.

    CPU-ONLY (the live seed is the incremental CPU SweepSeedBuilder; this is the recorded-path
    builder + the live fallback, kept consistent on CPU). A/B on identical data showed the CPU
    seed fits AS WELL OR BETTER than the old GPU build (train ema 0.0099 vs 0.0186, PSNR 15.75 vs
    15.06 dB), and CPU can't OOM the shared 16 GB GPU that SAM3D/train need.

    WHY still a subprocess + 3 mm (NOT an in-process fuse_recorded_dataset call):
      * 3 mm matches the SweepSeedBuilder + the dataparser's expected seed density.
      * Open3D teardown can abort the process on some paths; a subprocess contains that so it can't
        take down the whole pipeline.
    """
    import subprocess as _sp
    import sys as _sys
    static_dir = Path(static_dir)
    env = dict(os.environ)
    env.setdefault("DGS_TSDF_VOXEL_M", "0.003")          # 3 mm — matches SweepSeedBuilder.SEED_VOXEL_M
    env["DGS_FUSION_DEVICE"] = "cpu"                      # CPU ALWAYS (can't OOM the GPU; better fit)
    r = _sp.run([_sys.executable, "-m", "dynamic_gs.utils.online_fusion", str(static_dir)],
                env=env, capture_output=True, text=True, timeout=900)
    if r.returncode != 0 or not (static_dir / "depth_camera_init_points.ply").exists():
        raise RuntimeError(f"CPU TSDF build failed rc={r.returncode}: {(r.stderr or '')[-400:]}")
    print(f"[static] CPU TSDF seed built -> {static_dir / 'depth_camera_init_points.ply'}", flush=True)


# ----------------------------------------------------------------- gsplat backward warm
def _warm_gsplat_backward(cfg, device) -> None:
    """Run ONE throwaway gsplat rasterization + backward so the gsplat CUDA kernels (proj fwd/bwd,
    fully_fused_projection, rasterize) JIT-COMPILE here (under the sweep) instead of on the first
    training step. The compile is a one-time ~minutes nvcc build cached in ~/.cache/torch_extensions;
    triggering it on a tiny scene during the sweep keeps it off the critical path. Idempotent + cheap
    once cached (~ms). Runs on a daemon thread so it never blocks the sweep loop."""
    import threading as _th
    def _go():
        try:
            import torch
            import gsplat
            d = device
            N = 16
            means = torch.randn(N, 3, device=d, requires_grad=True)
            quats = torch.randn(N, 4, device=d, requires_grad=True)
            scales = (torch.rand(N, 3, device=d) * 0.01).requires_grad_(True)
            opac = torch.rand(N, device=d, requires_grad=True)
            colors = torch.rand(N, 3, device=d, requires_grad=True)
            viewmat = torch.eye(4, device=d)[None]
            K = torch.tensor([[[100.0, 0, 32], [0, 100.0, 32], [0, 0, 1]]], device=d)
            means_c = means + torch.tensor([0.0, 0.0, 2.0], device=d)   # in front of camera (+z)
            out, _alpha, _info = gsplat.rasterization(
                means_c, quats, scales, opac, colors, viewmat, K, width=64, height=64)
            out.sum().backward()         # forces the SLOW backward kernels to compile
            torch.cuda.synchronize()
            print("[static] gsplat backward warmed (CUDA kernels compiled off the critical path)", flush=True)
        except Exception as e:
            print(f"[static] gsplat warm skipped ({e}); will compile lazily on first train step", flush=True)
    _th.Thread(target=_go, name="gsplat-warm", daemon=True).start()


# ----------------------------------------------------------------- live schedule
# Pipeline steps shown in the checklist, in schedule order. (key, label).
_STEP_LIST = [
    ("capture", "Initial capture"),
    ("segment", "Segmentation"),
    ("generate", "Object generation"),
    ("train", "Scene training"),
    ("fuse", "Fusion"),
    ("realtime", "Realtime pipeline"),
]


class _RedBoxUI:
    """Minimal viser red-box trigger UI: shows the live camera feed with a centered red box and
    a 'Trigger (object in box)' button. Owns a bare viser server (NO gaussian render — the static
    sweep has no scene yet) and pushes the boxed feed directly as the client background image
    (server-side push-image, Invariant #9). Headless fallback: bare Enter on stdin also triggers.

    Also renders a STEP CHECKLIST panel (initial capture / segmentation / object generation /
    scene training / fusion / realtime): grey=not started, yellow=ongoing, green=done+elapsed.
    The orchestrator drives it via begin(key)/done(key); the current step's label is the feed banner."""

    def __init__(self, cfg, box_px: int = 300):
        self._box = int(box_px)
        self._fired = threading.Event()
        self._server = None
        self._draw_box = True             # box during sweep; drop it once triggered
        self._feed_ring = None            # SHM ring -> background feed thread pulls latest frames
        self._stop_feed = threading.Event()
        self._feed_thread = None
        # step checklist state: key -> {"label", "state" in {todo,doing,done}, "t0", "elapsed"}
        self._steps = {k: {"label": lbl, "state": "todo", "t0": None, "elapsed": 0.0}
                       for k, lbl in _STEP_LIST}
        self._steps_lock = threading.Lock()
        self._gui_steps = None
        try:
            import viser
            self._server = viser.ViserServer(port=int(getattr(cfg.viser, "port", 8081)))
            btn = self._server.gui.add_button("Trigger (object in box)")
            btn.on_click(lambda *_a: self._fired.set())
            # markdown checklist (live-updated): one ⚪/🟡/🟢 line per pipeline step. This is the ONLY
            # phase indicator now — no separate status-text widget, no on-feed banner (operator request).
            try:
                self._gui_steps = self._server.gui.add_markdown(self._render_steps_md())
            except Exception:
                self._gui_steps = None
            print(f"[static] red-box UI at http://localhost:{int(getattr(cfg.viser,'port',8081))} "
                  f"(press the Trigger button — or Enter in this terminal — when the object fills the box)")
        except Exception as e:
            print(f"[static] viser UI unavailable ({e}); use Enter to trigger.", flush=True)
        threading.Thread(target=self._stdin_wait, daemon=True).start()

    def _stdin_wait(self) -> None:
        try:
            input()                       # bare Enter triggers (headless / no browser)
            self._fired.set()
        except Exception:
            pass

    # ---- step checklist ----
    def _render_steps_md(self) -> str:
        """Render the checklist as markdown: 🟢 done (+elapsed) / 🟡 ongoing / ⚪ not started."""
        icon = {"todo": "⚪", "doing": "🟡", "done": "🟢"}
        lines = ["**Pipeline**"]
        for k, _lbl in _STEP_LIST:
            s = self._steps[k]
            suffix = ""
            if s["state"] == "done":
                suffix = f"  _({s['elapsed']:.1f}s)_"
            elif s["state"] == "doing" and s["t0"] is not None:
                suffix = f"  _({time.monotonic() - s['t0']:.0f}s…)_"
            lines.append(f"{icon[s['state']]} {s['label']}{suffix}")
        return "\n\n".join(lines)

    def _push_steps(self) -> None:
        if self._gui_steps is not None:
            try:
                self._gui_steps.content = self._render_steps_md()
            except Exception:
                pass

    def begin(self, key: str, keep: tuple = ()) -> None:
        """Mark a step ongoing (yellow) + start its clock; auto-finish any OTHER doing step
        except those in `keep` (e.g. keep 'capture' yellow while 'generate' runs)."""
        with self._steps_lock:
            for k, s in self._steps.items():     # close any still-running prior step (unless kept)
                if s["state"] == "doing" and k != key and k not in keep:
                    s["state"] = "done"
                    s["elapsed"] = (time.monotonic() - s["t0"]) if s["t0"] else s["elapsed"]
            s = self._steps.get(key)
            if s is not None and s["state"] != "done":
                s["state"], s["t0"] = "doing", time.monotonic()
        self._push_steps()

    def done(self, key: str) -> None:
        """Mark a step done (green) + freeze its elapsed time."""
        with self._steps_lock:
            s = self._steps.get(key)
            if s is not None and s["state"] != "done":
                s["elapsed"] = (time.monotonic() - s["t0"]) if s["t0"] else 0.0
                s["state"] = "done"
        self._push_steps()

    def drop_box(self) -> None:
        """Stop drawing the centered red box (called at trigger). The checklist now conveys phase."""
        self._draw_box = False

    def start_feed(self, ring) -> None:
        """Spawn a daemon that continuously pushes the FRESHEST SHM frame to the viewer, so the feed
        stays real-time during EVERYTHING (sweep + SAM3D + train) — not just while the main thread
        calls show(). Independent of the pipeline thread = never looks frozen."""
        self._feed_ring = ring
        self._feed_thread = threading.Thread(target=self._feed_loop, name="redbox-feed", daemon=True)
        self._feed_thread.start()

    def _feed_loop(self) -> None:
        last = -1
        n = 0
        while not self._stop_feed.is_set():
            try:
                fr = self._feed_ring.peek_latest() if self._feed_ring is not None else None
                if fr is not None and int(fr.seq) != last:
                    last = int(fr.seq)
                    self.show(fr.rgb_bgr)
                else:
                    self._stop_feed.wait(0.03)
                n += 1
                if n % 15 == 0:           # ~0.5s: tick the ongoing step's elapsed counter in the panel
                    self._push_steps()
            except Exception:
                self._stop_feed.wait(0.1)

    def show(self, rgb_bgr: np.ndarray) -> None:
        """Draw the centered red box (during sweep only) on the feed (BGR->RGB) + push it as every
        client's background image (one atomic set_background_image per client, Invariant #9).
        NO text banner — the step checklist panel conveys the current phase (operator request)."""
        if self._server is None:
            return
        import cv2
        img = rgb_bgr.copy()
        H, W = img.shape[:2]
        if self._draw_box:
            s = min(self._box, H, W)
            l, t = (W - s) // 2, (H - s) // 2
            cv2.rectangle(img, (l, t), (l + s, t + s), (0, 0, 255), 3)
        rgb = np.ascontiguousarray(img[..., ::-1])      # BGR -> RGB for viser
        try:
            for _cid, client in self._server.get_clients().items():
                client.scene.set_background_image(rgb, format="jpeg")
        except Exception:
            pass

    def fired(self) -> bool:
        return self._fired.is_set()

    def close(self) -> None:
        self._stop_feed.set()
        if self._feed_thread is not None:
            self._feed_thread.join(timeout=2.0)
        try:
            self._server is not None and self._server.stop()
        except Exception:
            pass


class _NullUI:
    """No-op stand-in for the red-box UI on RECORDED runs — no viser server, no live feed, no
    trigger button. The pipeline phase is conveyed by the [static] print lines instead (operator
    request). Every _RedBoxUI method the shared back-end calls is a safe no-op here."""
    def begin(self, *a, **k): pass
    def done(self, *a, **k): pass
    def drop_box(self): pass
    def start_feed(self, *a, **k): pass
    def show(self, *a, **k): pass
    def fired(self) -> bool: return False
    def close(self): pass


def run_static(data_dir, cfg, device, *, prompt_text: str = "", source_kind: str = "live_bridge",
               trigger_frame: Optional[int] = None, box_px: int = 300,
               prewarm_dynamic: bool = True, ff_enabled: bool = True, return_scene: bool = False):
    """The ONE static-phase orchestrator — source-agnostic. The deep stages (segment / SAM3D / seed /
    train / Phase-0b fuse) are identical no matter where the frames come from; ONLY the front end differs:

      live_bridge / ros1 : SWEEP the SHM stream into the red-box trigger UI + record static_scene/,
          anchor = the frame on the operator's Trigger (button / Enter).
      replay (recorded)  : NO UI, NO sweep — static_scene/ is already on disk; anchor = the LAST
          keyframe (or trigger_frame), reusing the existing seed PLY. Prints phase boundaries.

    Returns the dynamic_gs2 static_state.pt path, OR (path, (sm,gset,lock), registry) when
    return_scene=True (the single-process hand-off into the dynamic loop)."""
    from . import model_loader, static_segment, static_sam3d, static_fuse
    from .adapters_source import open_source, ShmRing
    from .static_capture import StaticRecorder        # writes the recorded static_scene/ dataset
    from .static_seed_stream import SweepSeedBuilder   # incremental CPU TSDF seed during the live sweep
    tm = _T.new_ledger()
    model_loader.set_timing_ledger(tm)   # so bg prewarm threads record load.{sam3d,fastsam,anysplat,xfeat}
    data_dir = Path(data_dir)
    prompt = prompt_text or cfg.segmentation.prompt_text
    is_live = source_kind in ("live_bridge", "ros1")

    # AnySplat is prewarmed for the dynamic loop's feedforward ONLY — gate on ff_enabled so a --no-ff run
    # never spawns its ~17 s / ~3.5 GB worker (which the FF-off dynamic loop would never adopt or free).
    reg = model_loader.build_registry(cfg, device,
                                      want_anysplat=(prewarm_dynamic and ff_enabled),
                                      want_xfeat=prewarm_dynamic)
    # SAM3D is the ONLY thing that loads during the sweep — it gets the GPU/CPU to itself so its ~25 s
    # load fully hides under the operator sweep. The dynamic-phase models (AnySplat + XFeat/LighterGlue)
    # and the gsplat-backward warm are deferred to AFTER SAM3D (the free slot, where they overlap the
    # seed-build + train): loading them during the sweep contends with the SAM3D load and stretches it
    # ~25 s -> ~40-49 s (measured). FastSAM loads lazily right before segment.
    with tm.stage("sweep.sam3d_load"):
        reg["sam3d"].prewarm()

    # ---- FRONT END: obtain the anchor Frame (+ intrinsics). live = sweep SHM; recorded = read disk ----
    src = None
    seed_builder = None             # live: incremental CPU TSDF seed (built during the sweep)
    if is_live:
        # Optional sweep-publisher throttle (DGS_PUB_SWEEP_HZ, default 0 = OFF = full rate, unchanged for
        # fast machines). On a 16 GB / shared GPU, capping the SWEEP publisher to ~10 Hz cuts its per-frame
        # decode + mask-GL + SHM contention ~3x so the concurrent SAM3D load hides under the sweep instead
        # of being stretched. The DYNAMIC publisher (run_live) is spawned separately at FULL rate.
        try:
            _sweep_hz = float(os.environ.get("DGS_PUB_SWEEP_HZ", "0"))
        except ValueError:
            _sweep_hz = 0.0
        src = open_source(source_kind, data_dir=data_dir, shm_name=cfg.shm_name, attach=True,
                          max_hz=_sweep_hz)
        ring = ShmRing(cfg.shm_name)
        intr = ring.intrinsics()
        ui = _RedBoxUI(cfg, box_px=box_px)
        ui.start_feed(ring)             # background thread keeps the LIVE feed flowing through EVERYTHING
        ui.begin("capture")
        recorder = StaticRecorder(data_dir, intr)
        # Incremental CPU TSDF seed: fuse each kept keyframe AS IT ARRIVES during the sweep on a bg
        # CPU thread, so the seed is ready (modulo a ~2 s finalize) at trigger instead of paying a
        # ~12 s cold GPU subprocess after SAM3D. Pure CPU -> no contention with the SAM3D GPU load.
        seed_builder = SweepSeedBuilder(intr, tm=tm)
        try:
            seed_builder.start()
        except Exception as _e:
            print(f"[static] WARNING incremental seed builder failed to start ({_e}); "
                  f"will fall back to GPU build at trigger", flush=True)
            seed_builder = None
        # NOTE: AnySplat is NOT prewarmed here. Measured 2026-06-21: loading AnySplat (~3.4 GB) during
        # the sweep, concurrent with the SAM3D load (~8.6 GB) + Gazebo, OOMs the 16 GB card (45 MiB free
        # -> SAM3D load aborts -> run dies). AnySplat MUST wait for sam_worker_close to free SAM3D's VRAM
        # (see after.* block below). This is a hard VRAM ceiling, not a speed choice -> the ~10 s handoff
        # tail is the cost of the 16 GB budget. (To hide it, need a bigger card or a smaller AnySplat.)
        # SWEEP: RECORD frames until the operator triggers.
        anchor_frame: Optional[Frame] = None
        last_seq = -1
        while not ui.fired():
            fr = ring.peek_latest()
            if fr is None or int(fr.seq) == last_seq:
                time.sleep(0.005); continue
            last_seq = int(fr.seq)
            if recorder.add(fr) and seed_builder is not None:
                seed_builder.submit(fr)        # kept keyframe -> fuse it on the bg thread
            anchor_frame = fr
        tm.event("triggered")
        print(f"[static] sweep ended: recorder kept {recorder.num_kept} keyframes", flush=True)
        if recorder.num_kept == 0:
            print("[static] WARNING: 0 keyframes recorded — the seed build WILL fail. Check the live "
                  "SHM feed (publisher producing frames with valid rgb/depth?).", flush=True)
        recorder.finalize()             # writes static_scene/{rgb,depth,masks,transforms.json}
        ui.drop_box()
        # Sweep is over: no more frames are consumed except the UI's live RGB feed. Touch the rgb-only
        # sentinel so the static publisher drops its GPU-heavy robot-mask GL render + depth decode and
        # caps to ~10 Hz, freeing the GPU for the concurrent TSDF-seed build + Splatfacto train. The
        # sentinel lives in LIVE_ROOT, which the SEPARATE dynamic publisher rmtree's at its own spawn,
        # so dynamic always comes back full mask+depth+rate.
        try:
            from dynamic_gs.utils.live_shm_reader import LIVE_ROOT as _LIVE_ROOT
            (Path(_LIVE_ROOT) / ".rgb_only").touch()
        except Exception as _e:
            print(f"[static] WARNING: could not set rgb-only sentinel ({_e})", flush=True)
    else:
        # RECORDED: static_scene/ is already on disk. Anchor = last keyframe (or trigger_frame). We read
        # it off disk (file_path string-sort) rather than re-streaming through ReplaySource: ReplaySource
        # orders by trailing integer, which would pick an 'aa_' lead-in frame as "last" instead of the
        # final sweep keyframe -> undersized fused object (see _anchor_from_disk).
        ui = _NullUI()
        anchor_sel = "last keyframe" if trigger_frame is None else f"keyframe #{trigger_frame}"
        print(f"[static] recorded source: static_scene/ on disk, anchor = {anchor_sel}, no UI", flush=True)
        anchor_frame, intr = _anchor_from_disk(data_dir, cfg, trigger_frame)

    # ---- SHARED back end (identical for live + recorded): segment -> SAM3D -> seed -> train -> fuse ----
    ui.begin("segment", keep=("capture",))
    print("[static] segmentation: start", flush=True)
    with tm.stage("trigger.snapshot_anchor"):
        anchor = static_segment.snapshot_anchor(anchor_frame, intr, data_dir)
    with tm.stage("trigger.fastsam_segment"):
        # Load FastSAM into the shared worker NOW (right before segment, NOT at sweep-start: its 0.85 GB
        # resident during SAM3D's load peak OOMs a 16 GB card). Warm worker infer ~2.5 s vs the ~40 s
        # cold subprocess the worker falls back to when FastSAM was never loaded.
        reg["fastsam"].prewarm(); reg["fastsam"].wait_ready()
        objects = static_segment.segment(anchor, reg["fastsam"], prompt)
    with tm.stage("trigger.write_seg_folder"):
        static_segment.write_seg_folder(anchor, objects, prompt)
    tm.event("objects_found", n=len(objects))
    print(f"[static] segmentation: done — {len(objects)} object(s)", flush=True)
    if objects:                          # valid segmentation -> unload FastSAM (~0.85 GB) from the shared
        reg["fastsam"].release()         # worker NOW, so it isn't held idle through the ~10 s SAM3D infer
    ui.done("segment")
    ui.begin("generate", keep=("capture",))

    print("[static] object generation (SAM3D): start", flush=True)
    with tm.stage("trigger.sam3d_infer"):
        sam3d = static_sam3d.generate(anchor, objects, reg["sam3d"])
    tm.event("sam3d_done")             # dead-time origin
    print("[static] object generation (SAM3D): done", flush=True)
    ui.done("generate")
    ui.done("capture")                  # capture finishes AFTER SAM3D (live: the object is now reconstructed)

    # Free the SAM worker (FastSAM + SAM3D VRAM) BEFORE the seed + train — they can't co-reside with the
    # VoxelBlockGrid at 1200p; closing the worker process 100%-frees its CUDA context. Then spawn AnySplat
    # into the freed slot (its ~17 s load overlaps the seed-build + train).
    with tm.stage("after.sam_worker_close"):
        reg["_sam_worker"].close()

    # END OF SAM3D = the free slot: NOW load AnySplat (~3.4 GB) + XFeat/LighterGlue + warm the gsplat
    # backward. AnySplat can ONLY load here — its VRAM doesn't fit alongside the SAM3D load (OOM, see
    # the sweep note above). Its ~17 s load overlaps the seed-finalize + train; whatever doesn't finish
    # by the train end is the visible end.wake_dynamic handoff (load.anysplat shows the real load cost).
    if prewarm_dynamic:
        with tm.stage("after.anysplat_spawn"):
            reg.get("anysplat") and reg["anysplat"].prewarm()
        with tm.stage("after.dyn_models_prewarm"):
            reg.get("xfeat") and reg["xfeat"].prewarm()
    with tm.stage("after.gsplat_warm"):
        _warm_gsplat_backward(cfg, device)

    # Seed: build the TSDF init PLY if it isn't already on disk (a fresh live capture always builds it;
    # a recorded dataset reuses the seed it was captured with).
    ui.begin("train")
    seed_ply = data_dir / "static_scene" / "depth_camera_init_points.ply"
    if seed_ply.exists() and seed_builder is None:
        # Recorded run (or a re-run): reuse the seed already on disk. (A LIVE run always has a
        # seed_builder, whose finalize() WRITES this PLY below — so don't short-circuit on it.)
        print(f"[static] reusing existing seed -> {seed_ply.name}", flush=True)
    else:
        built = None
        if seed_builder is not None:
            # LIVE: the incremental CPU builder fused every keyframe during the sweep (+ all of the
            # SAM3D wait); finalize() is just the ~2 s extract + downsample + write. No GPU, no cold
            # subprocess. Falls through to the GPU build only if it fused nothing / errored.
            print("[static] finalizing incremental CPU TSDF seed", flush=True)
            with tm.stage("after.tsdf_integrate"):
                built = seed_builder.finalize(data_dir / "static_scene")
        if built is None:
            print("[static] building TSDF seed (GPU subprocess fallback)", flush=True)
            with tm.stage("after.tsdf_integrate"):
                _build_seed_deferred(data_dir / "static_scene")

    print("[static] scene training: start", flush=True)
    with tm.stage("after.splatfacto_train"):
        # on_fuse fires at the train->Phase-0b boundary (both live inside train_fuse_and_export).
        def _on_fuse():
            ui.done("train"); ui.begin("fuse")
            print("[static] fusion (Phase-0b): start", flush=True)
        out, sm, gset, lock = static_fuse.train_fuse_and_export(
            data_dir, cfg, device, anchor=anchor, sam3_objects=objects, sam3d_results=sam3d,
            tm=tm, return_scene=True, on_fuse=_on_fuse)
    tm.event("static_state_written")
    print("[static] scene training + fusion: done", flush=True)
    ui.done("fuse")
    ui.begin("realtime")

    if prewarm_dynamic:
        with tm.stage("end.wake_dynamic"):
            reg.get("anysplat") and reg["anysplat"].wait_ready()
            reg.get("xfeat") and reg["xfeat"].wait_ready()

    if src is not None:
        src.close()
    ui.close()
    _write_report(tm, data_dir, out)
    if return_scene:
        return out, (sm, gset, lock), reg     # single-process hand-off: scene + warm model registry
    return out


# ----------------------------------------------------------------- report
def _write_report(tm, data_dir: Path, out_pt: Path) -> None:
    """Render the static-phase timing report (schedule order + overlap timeline + dead time)."""
    try:
        path = Path(data_dir) / "timing_report_static.txt"
        tm.write_static(path, dead_time_after="sam3d_done")
        print(f"[static] timing report -> {path}", flush=True)
        print(f"[static] dynamic_gs2 warm-cache -> {out_pt}", flush=True)
    except Exception as e:
        print(f"[static] timing report failed: {e}", flush=True)


# ----------------------------------------------------------------- CLI
def _main():
    import argparse
    from . import config as _C
    ap = argparse.ArgumentParser(description="dynamic_gs2 static-phase runner")
    ap.add_argument("--mode", choices=["recorded", "live"], default="recorded",
                    help="recorded = old static_scene/ on disk (unattended); live = sweep+red-box UI (needs sim)")
    ap.add_argument("--data", required=True, help="dataset dir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt", default="", help="object text prompt (else cfg/DGS_SAM3_PROMPT)")
    ap.add_argument("--source", default="live_bridge", help="live source kind (live_bridge|ros1)")
    ap.add_argument("--box-px", type=int, default=500, help="live: red-box side in px")
    ap.add_argument("--no-prewarm", dest="prewarm", action="store_false", default=True,
                    help="DEFAULT prewarms AnySplat+XFeat DURING the static train (hidden under it) so "
                         "go-live is instant = the fastest end-to-end path; --no-prewarm only to debug")
    args = ap.parse_args()
    cfg = _C.load_runtime_config()
    if args.mode == "recorded":
        run_static_from_recorded(args.data, cfg, args.device,
                                 prompt_text=args.prompt, prewarm_dynamic=args.prewarm)
    else:
        run_static_live(args.data, cfg, args.device, prompt_text=args.prompt,
                        source_kind=args.source, box_px=args.box_px)


if __name__ == "__main__":
    _main()
