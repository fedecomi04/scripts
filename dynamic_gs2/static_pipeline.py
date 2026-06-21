"""static_pipeline.py — the static-phase orchestrator (the §1 schedule + state machine).

Owns the SWEEP -> TRIGGER -> SAM3D -> TRAIN -> FUSE -> DONE schedule, the red-box trigger
UI, the per-bulletpoint timing (one timed stage per §1 line, rendered FF-report-style to
timing_report_static.txt), and the prewarm wiring. Each heavy stage is a single-responsibility
module (static_seed / static_segment / static_sam3d / static_fuse) and a model handle in the
prewarm registry (model_loader); this file only SEQUENCES them and hands the GaussianSet the
exported warm-cache. (rewrite_spec/static_phase.md §1/§4.)

Two entry points:
  run_static_from_recorded : the unattended-validatable path — a recorded static_scene/ already
      on disk. Runs old static-gs (train+phase0a+phase0b) -> converts to dynamic_gs2
      static_state.pt -> prewarms the dynamic models. Times the after-SAM3D stages.
  run_static_live          : the full live schedule with the red-box UI (capture sweep -> trigger
      -> segment -> SAM3D -> build seed -> write dataset -> old static-gs -> convert -> prewarm).
      Needs the sim; OPERATOR-validated (the UI/SHM/capture pieces can't be unattended-tested).

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
def _last_static_frame(data_dir, cfg):
    """Build (Frame, Intrinsics) from the LAST static keyframe (the operator's final head-on view) — the frame
    Phase-0a segments + SAM3D sees, exactly as the old phase0a used cached_train[-1]."""
    import json, cv2
    from .frame import Frame, Intrinsics
    st = Path(data_dir) / "static_scene"
    meta = json.loads((st / "transforms.json").read_text())
    # Sort by file_path STRING (NOT trailing integer) to match nerfstudio's dataparser, which is what
    # the old Phase-0a's cached_train[-1] used. The capture convention renames the pre-motion lead-in
    # frames with an 'aa_' prefix SO THEY SORT FIRST, leaving the last SWEEP keyframe (arm_*) as the
    # anchor. A trailing-int sort breaks that: aa_lead_00029 (29) wrongly sorts after arm_00026 (26),
    # picking a near/oblique lead-in frame -> the object back-projects small -> undersized fused object.
    fr = sorted(meta["frames"], key=lambda f: f["file_path"])[-1]
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


def run_static_from_recorded(data_dir, cfg, device, *, prompt_text: str = "",
                             prewarm_dynamic: bool = True) -> Path:
    """Recorded/from-scratch static phase, FULLY NATIVE (no ns-train subprocess): the static_scene/
    dataset is already on disk. Phase-0a (FastSAM segment + SAM3D on the last keyframe) -> native
    train (static_train) + native Phase-0b (static_phase0b) + warm-cache (static_fuse) -> prewarm the
    dynamic models. Returns the dynamic_gs2 static_state.pt path. Unattended-validated."""
    from . import model_loader, static_fuse, static_segment, static_sam3d
    tm = _T.new_ledger()
    data_dir = Path(data_dir)
    prompt = prompt_text or cfg.segmentation.prompt_text

    reg = model_loader.build_registry(cfg, device,
                                      want_anysplat=prewarm_dynamic, want_xfeat=prewarm_dynamic)

    # Phase-0a — segment + SAM3D on the last static keyframe (the head-on view the masks belong to).
    frame, intr = _last_static_frame(data_dir, cfg)
    with tm.stage("trigger.snapshot_anchor"):
        anchor = static_segment.snapshot_anchor(frame, intr, data_dir)
    with tm.stage("trigger.fastsam_segment"):
        objects = static_segment.segment(anchor, reg["fastsam"], prompt)
    with tm.stage("trigger.write_seg_folder"):
        static_segment.write_seg_folder(anchor, objects, prompt)
    tm.event("objects_found", n=len(objects))
    with tm.stage("trigger.sam3d_infer"):
        sam3d = static_sam3d.generate(anchor, objects, reg["sam3d"])
    tm.event("sam3d_done")          # dead-time origin: everything after this is operator-visible wait
    reg["_sam_worker"].close()      # free FastSAM+SAM3D VRAM before the train (same as the live path)

    # AnySplat spawn overlaps the native train (the §1 free slot), prewarm XFeat too (for go-live).
    if prewarm_dynamic:
        with tm.stage("after.anysplat_spawn"):
            reg.get("anysplat") and reg["anysplat"].prewarm()
        with tm.stage("sweep.dyn_models_prewarm"):
            reg.get("xfeat") and reg["xfeat"].prewarm()

    # Native train + Phase-0b + export (static_train + static_phase0b internals).
    with tm.stage("after.splatfacto_train"):
        out = static_fuse.train_fuse_and_export(
            data_dir, cfg, device, anchor=anchor, sam3_objects=objects, sam3d_results=sam3d, tm=tm)
    tm.event("static_state_written")

    if prewarm_dynamic:
        with tm.stage("end.wake_dynamic"):
            reg.get("anysplat") and reg["anysplat"].wait_ready()
            reg.get("xfeat") and reg["xfeat"].wait_ready()

    _write_report(tm, data_dir, out)
    return out


# ----------------------------------------------------------------- deferred TSDF seed
def _build_seed_deferred(static_dir) -> None:
    """Build depth_camera_init_points.ply via the GPU TSDF in a SUBPROCESS at 3 mm voxel,
    with a CPU fallback — the proven live_session DGS_LIVE_DEFER_TSDF=1 pattern.

    WHY a subprocess + 3 mm (NOT an in-process fuse_recorded_dataset call):
      * At 1920x1200/110deg the GPU VoxelBlockGrid hashmap at the 2 mm DEFAULT OOMs 16 GB even
        with ~12 GB free; DGS_TSDF_VOXEL_M=0.003 fits.
      * An Open3D CUDA OOM POISONS its allocator cache and ABORTS the process at teardown
        (std::runtime_error 'Block ... should have been recorded') — in-process that kills the
        WHOLE pipeline even though work elsewhere succeeded. In a subprocess the abort is
        contained and we fall back to the CPU build cleanly. (This was the 2026-06-20 crash:
        the in-process 2 mm call OOM'd and took the live run down before go-live.)
    """
    import subprocess as _sp
    import sys as _sys
    static_dir = Path(static_dir)
    env = dict(os.environ)
    env.setdefault("DGS_TSDF_VOXEL_M", "0.003")          # 3 mm fits 1200p; 2 mm OOMs the VBG hashmap
    try:
        r = _sp.run([_sys.executable, "-m", "dynamic_gs.utils.online_fusion", str(static_dir)],
                    env=env, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"GPU TSDF subprocess rc={r.returncode}: {(r.stderr or '')[-400:]}")
        seed = static_dir / "depth_camera_init_points.ply"
        if not seed.exists():
            raise RuntimeError("GPU TSDF subprocess wrote no seed PLY")
        print(f"[static] deferred GPU TSDF seed built -> {seed}", flush=True)
    except Exception as exc:
        print(f"[static] WARNING deferred GPU TSDF failed ({exc}); trying CPU build", flush=True)
        env["DGS_FUSION_DEVICE"] = "cpu"                  # CPU ScalableTSDFVolume can't OOM the GPU
        r = _sp.run([_sys.executable, "-m", "dynamic_gs.utils.online_fusion", str(static_dir)],
                    env=env, capture_output=True, text=True, timeout=900)
        if r.returncode != 0 or not (static_dir / "depth_camera_init_points.ply").exists():
            raise RuntimeError(f"CPU TSDF fallback also failed rc={r.returncode}: {(r.stderr or '')[-400:]}")
        print("[static] CPU TSDF seed built (GPU fallback)", flush=True)


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


def run_static_live(data_dir, cfg, device, *, prompt_text: str = "",
                    source_kind: str = "live_bridge", box_px: int = 500,
                    return_scene: bool = False):
    """Full live static schedule (OPERATOR-validated — needs the sim).

    SWEEP: stream live frames into the red-box UI + record them to static_scene/ + ICP-live the
      seed; prewarm FastSAM/SAM3D/XFeat. TRIGGER (button/Enter): snapshot anchor -> FastSAM
      segment -> SAM3D infer. AFTER: finalize seed, then hand to the proven old static-gs on the
      recorded dataset -> convert -> prewarm AnySplat under train. Returns the dynamic_gs2 .pt, OR
      (path, (sm,gset,lock), registry) when return_scene=True (the single-process hand-off)."""
    from . import model_loader, static_segment, static_sam3d, static_fuse
    from .adapters_source import open_source, ShmRing
    from .static_capture import StaticRecorder        # writes the recorded static_scene/ dataset
    tm = _T.new_ledger()
    data_dir = Path(data_dir)
    prompt = prompt_text or cfg.segmentation.prompt_text

    reg = model_loader.build_registry(cfg, device)
    # Prewarms that fit the SWEEP VRAM budget kick off NOW (non-blocking bg threads), so their load
    # overlaps the operator sweep instead of the trigger. Hidden here: FastSAM ~10 s + SAM3D ~48 s
    # (both in the shared worker), XFeat ~2 s, gsplat-backward compile. AnySplat is NOT prewarmed here
    # ON PURPOSE — its ~3.5 GB on top of SAM3D's 11.7 GB peak + Gazebo 2.6 OOMs a 16 GB card; it spawns
    # AFTER the SAM worker is closed (post-SAM3D), overlapping the seed-build + train (the free slot).
    # VRAM: FastSAM (0.85 GB) resident + SAM3D's ~11.7 GB LOAD-PEAK + Gazebo 2.6 OOMs a 16 GB card.
    # So during the sweep load SAM3D ALONE (full headroom, the 40 s cost hidden under the sweep);
    # FastSAM is loaded at the TRIGGER right before segment (only ~2.8 s, and it's needed there first
    # anyway — segment runs before SAM3D infer). This guarantees FastSAM-ready-before-use AND gives
    # SAM3D its full load headroom (the FastSAM-first-resident attempt OOM'd SAM3D's load, 2026-06-21).
    with tm.stage("sweep.sam3d_load"):
        reg["sam3d"].prewarm()         # load SAM3D async (~40 s) — overlaps the sweep, no FastSAM contention
    with tm.stage("sweep.dyn_models_prewarm"):
        reg["xfeat"].prewarm()
    with tm.stage("sweep.gsplat_warm"):
        _warm_gsplat_backward(cfg, device)   # one throwaway grad step -> the gsplat CUDA backward compiles
                                             # HERE (under the sweep), not on the first training step

    src = open_source(source_kind, data_dir=data_dir, shm_name=cfg.shm_name, attach=True)
    ring = ShmRing(cfg.shm_name)
    intr = ring.intrinsics()
    ui = _RedBoxUI(cfg, box_px=box_px)
    ui.start_feed(ring)                 # background thread keeps the LIVE feed flowing through EVERYTHING
    ui.begin("capture")                 # checklist: capture is the first step (runs through the sweep)
    recorder = StaticRecorder(data_dir, intr)

    # SWEEP: RECORD frames only until the operator triggers (the UI feed-thread shows them). The GPU-TSDF
    # seed is DEFERRED (built once after SAM3D unloads, §2a/§2c): GPU-TSDF's VoxelBlockGrid is ~8 GB at
    # 1200p and CANNOT coexist on the GPU with SAM3D / FastSAM / the publisher / Gazebo — running it live
    # per-frame here OOMs (the exact failure DGS_LIVE_DEFER_TSDF=1 prevents). The seed build is ~1.5 s.
    anchor_frame: Optional[Frame] = None
    last_seq = -1
    while not ui.fired():
        fr = ring.peek_latest()
        if fr is None or int(fr.seq) == last_seq:
            time.sleep(0.005); continue
        last_seq = int(fr.seq)
        recorder.add(fr)
        anchor_frame = fr
    tm.event("triggered")
    print(f"[static] sweep ended: recorder kept {recorder.num_kept} keyframes", flush=True)
    if recorder.num_kept == 0:
        print("[static] WARNING: 0 keyframes recorded — the seed build WILL fail. Check the live "
              "SHM feed (publisher producing frames with valid rgb/depth?).", flush=True)
    # 'capture' stays ONGOING (yellow) through segmentation + SAM3D and turns done only after SAM3D —
    # capture isn't "finished" until the object the operator framed has been segmented + reconstructed.
    ui.drop_box()                                            # box gone at trigger; capture step still yellow
    ui.begin("segment", keep=("capture",))                  # keep 'capture' yellow through segment + SAM3D

    # TRIGGER: freeze anchor, segment, SAM3D (the operator keeps moving — the live feed keeps flowing).
    with tm.stage("trigger.snapshot_anchor"):
        anchor = static_segment.snapshot_anchor(anchor_frame, intr, data_dir)
    # Load FastSAM NOW (after SAM3D's load-peak is done; ~2.8 s) into the worker that already holds
    # SAM3D resident — 7.3 + 0.85 + Gazebo fits 16 GB (only SAM3D's transient LOAD peak didn't).
    with tm.stage("trigger.fastsam_segment"):
        reg["fastsam"].prewarm(); reg["fastsam"].wait_ready()
        objects = static_segment.segment(anchor, reg["fastsam"], prompt)
    with tm.stage("trigger.write_seg_folder"):
        static_segment.write_seg_folder(anchor, objects, prompt)
    tm.event("objects_found", n=len(objects))
    ui.done("segment")
    ui.begin("generate", keep=("capture",))                 # keep 'capture' yellow alongside 'generate'
    with tm.stage("trigger.sam3d_infer"):
        sam3d = static_sam3d.generate(anchor, objects, reg["sam3d"])
    tm.event("sam3d_done")             # dead-time origin
    ui.done("generate")
    ui.done("capture")                                       # capture finishes AFTER SAM3D (operator request)

    # AFTER SAM3D: free the shared SAM worker's VRAM (FastSAM + SAM3D ~8 GB) BEFORE the GPU-TSDF seed +
    # splatfacto — they can't co-reside with the VoxelBlockGrid at 1200p. Closing the worker process
    # 100%-frees its CUDA context (more reliable than unload_* which leaves cached allocator VRAM).
    with tm.stage("after.sam_worker_close"):
        reg["_sam_worker"].close()

    # NOW spawn AnySplat (the GPU is free of SAM3D): its ~17 s load overlaps the seed-build + train
    # below (the proven 'free slot'). Done HERE (not at sweep-start) because AnySplat's ~3.5 GB on top
    # of SAM3D's 11.7 GB peak + Gazebo would OOM during the sweep.
    with tm.stage("after.anysplat_spawn"):
        reg["anysplat"].prewarm()

    # finalize the recorded dataset, then build the seed in ONE GPU pass (the proven
    # DGS_LIVE_DEFER_TSDF=1 path), then native train + Phase-0b.
    recorder.finalize()                # writes static_scene/{rgb,depth,masks,transforms.json}
    ui.begin("train")                  # 'Scene training' covers seed-build (below) + splatfacto train
    with tm.stage("after.tsdf_integrate"):
        _build_seed_deferred(data_dir / "static_scene")      # -> depth_camera_init_points.ply
    with tm.stage("after.splatfacto_train"):
        # on_fuse: the native fuse calls this between train + Phase-0b, so the checklist flips
        # train->done / fuse->doing at the real boundary (train + fuse live inside one function).
        def _on_fuse():
            ui.done("train"); ui.begin("fuse")
        out, sm, gset, lock = static_fuse.train_fuse_and_export(
            data_dir, cfg, device, anchor=anchor, sam3_objects=objects, sam3d_results=sam3d,
            tm=tm, return_scene=True, on_fuse=_on_fuse)
    ui.done("fuse")
    tm.event("static_state_written")
    ui.begin("realtime")
    with tm.stage("end.wake_dynamic"):
        reg["anysplat"].wait_ready(); reg["xfeat"].wait_ready()

    src.close(); ui.close()
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
