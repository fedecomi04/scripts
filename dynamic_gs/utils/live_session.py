"""Pre-training interactive session for `ns-train dynamic-gs --live`.

Runs entirely on the main thread before nerfstudio's pipeline `__init__`
takes over:

    1. Wipe LIVE_ROOT, start ROS subscriber.
    2. Wait for first synced tuple → print "ready!".
    3. Prompt for SAM3 text (default if blank).
    4. Wait for first Enter ("move robot in front of object").
    5. Capture next frame as SAM3 anchor; start writing every subsequent
       synced tuple to LIVE_ROOT/static_scene/.
    6. SAM3 subprocess (blocking, ~1-2s) on the anchor → print summary.
    7. SAM3D subprocess (background thread, ~100s) on SAM3 masks. The
       ROS subscriber keeps writing frames during this window.
    8. Wait for second Enter ("done capturing static views"). After
       Enter, also wait for the SAM3D thread to finish — static
       optimisation cannot share the GPU with SAM3D.
    9. Stop recording. Append a stub frame to LIVE_ROOT/dynamic_scene/
       (the dataparser refuses an empty dynamic dataset). Build the
       SfM init PLY for static_scene/.

Returns LIVE_ROOT so the pipeline can point its dataparser at it.
"""

from __future__ import annotations

import atexit
import gc
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from .live_shm_reader import (
    LIVE_ROOT,
    LiveFrame,
    LiveShmSubscriber,
)
from .sam3_segmentation import run_sam3_subprocess
from .fastsam_segmentation import run_fastsam_subprocess
from .sam3d import run_sam3d_multi_object_subprocess, sam3d_pose_has_rotation
from .sam_worker import SamWorkerClient
from . import timing_ledger as _tl

INIT_CLOUD_NAME = "depth_camera_init_points.ply"


def _has_complete_recording_cache(live_root: Path) -> bool:
    """Return True iff a previous live session left behind enough on disk
    to skip the recording + SAM3 + SAM3D workflow on this run.

    Checks the four marker files the pipeline downstream will read:
    static_scene/transforms.json (camera frames), static_scene/<PLY>
    (SfM init seed), dynamic_scene/initialization_debug/static0_sam3_results.json
    (SAM3 cache), and at least one valid SAM3D PLY+pose pair under
    dynamic_scene/initialization_artifacts/. If any are missing we
    fall back to the full interactive flow.
    """
    static = live_root / "static_scene"
    dyn = live_root / "dynamic_scene"
    debug = dyn / "initialization_debug"
    artifact = dyn / "initialization_artifacts"
    if not (static / "transforms.json").exists():
        return False
    if not (static / INIT_CLOUD_NAME).exists():
        return False
    if not (dyn / "transforms.json").exists():
        return False
    if not (debug / "static0_sam3_results.json").exists():
        return False
    for ply in sorted(artifact.glob("static0_obj_*_sam3d_raw_output.ply")):
        pose = ply.with_name(ply.name.replace("_raw_output.ply", "_pose.json"))
        if pose.exists() and sam3d_pose_has_rotation(pose):
            return True
    return False
    # PROBLEM: this looks at file existence, not file content integrity.
    # If a previous run was killed mid-write (e.g. SAM3 wrote the JSON
    # but SAM3D crashed before the PLY landed) we recover automatically
    # because we only return True when at least one valid pose+PLY pair
    # is found. But the matching SAM3 mask count vs SAM3D PLY count
    # is NOT verified — a stale partial cache could pass this check
    # and trip up Phase 0b fusion. Delete LIVE_ROOT to force a fresh
    # capture.


# Module-level flag + sub reference so the unpause-on-exit guard only
# fires when we actually paused. Set True by ``pause_gazebo_physics``;
# cleared by ``unpause_gazebo_physics``. The atexit hook reads this to
# decide whether to attempt a final unpause.
#
# The publisher subprocess (in the ROS env) owns the actual rospy
# ServiceProxy calls; we route through it via the LiveShmSubscriber
# control pipe. That keeps the dynamic_gs env free of rospy.
_GAZEBO_PHYSICS_PAUSED = False
_PAUSE_SUB: Optional["LiveShmSubscriber"] = None


def pause_gazebo_physics(sub: "LiveShmSubscriber") -> bool:
    """Pause Gazebo via the publisher subprocess.

    Used to free CPU/GPU contention from Gazebo for the duration of
    the SAM3D subprocess + init PLY build. Safe to call when Gazebo
    isn't running — returns False without raising.
    """
    global _GAZEBO_PHYSICS_PAUSED, _PAUSE_SUB
    ok = bool(sub.pause_gazebo_physics())
    if ok:
        _GAZEBO_PHYSICS_PAUSED = True
        _PAUSE_SUB = sub
        print("[live] Gazebo physics paused", flush=True)
    else:
        print("[live] could not pause Gazebo", flush=True)
    return ok


def unpause_gazebo_physics(sub: "LiveShmSubscriber") -> bool:
    """Unpause Gazebo. Idempotent."""
    global _GAZEBO_PHYSICS_PAUSED
    _GAZEBO_PHYSICS_PAUSED = False
    ok = bool(sub.unpause_gazebo_physics())
    if ok:
        print("[live] Gazebo physics unpaused", flush=True)
    else:
        print("[live] could not unpause Gazebo", flush=True)
    return ok


def _atexit_unpause() -> None:
    """Final safety net: if the process exits while Gazebo is still
    paused — for example because SAM3D crashed or the user Ctrl+C'd
    during PCD build — unpause so we don't leave the simulator frozen
    for the next operator.
    """
    if _GAZEBO_PHYSICS_PAUSED and _PAUSE_SUB is not None:
        try:
            _PAUSE_SUB.unpause_gazebo_physics()
        except Exception:
            pass


atexit.register(_atexit_unpause)

# Hardcoded SAM3 defaults (kept here, not in the model config, because
# the live workflow expects to override them via the user prompt).
DEFAULT_SAM3_PROMPT = "oobject table"
SAM3_CONDA_ENV = "sam3_dynamic_gs"
# Segmentation backend: "fastsam" (default — ~0.85 GB, co-resides with SAM3D so
# SAM3D can load during the capture tail) or "sam3". Env-overridable so
# bootstrap_live.sh can switch without code edits.
SEGMENTATION_BACKEND = os.environ.get("DGS_SEGMENTATION_BACKEND", "fastsam").strip().lower()
SAM3_CANDIDATE_MIN_AREA_RATIO = 0.002
SAM3_CANDIDATE_MAX_AREA_RATIO = 0.25
SAM3_CANDIDATE_DEDUP_IOU = 0.6
SAM3_CANDIDATE_MAX_OBJECTS = 8
SAM3_CONFIDENCE_THRESHOLD = 0.3
SAM3_MIN_SCORE = 0.2


def _wipe_live_root() -> None:
    if LIVE_ROOT.exists():
        shutil.rmtree(LIVE_ROOT)
    (LIVE_ROOT / "static_scene").mkdir(parents=True, exist_ok=True)
    (LIVE_ROOT / "dynamic_scene").mkdir(parents=True, exist_ok=True)


def _save_anchor_for_sam3(anchor: LiveFrame, debug_dir: Path) -> Path:
    """Write the SAM3 input image: anchor RGB with the robot mask
    applied so the gripper doesn't drift into SAM3's text-prompt
    response. Mirrors the masking step in the recorded-mode
    `_run_sam3_and_sam3d_generation`.
    """
    debug_dir.mkdir(parents=True, exist_ok=True)
    rgb_rgb = cv2.cvtColor(anchor.rgb_bgr, cv2.COLOR_BGR2RGB).copy()
    keep = anchor.mask_keep > 0
    if keep.shape != rgb_rgb.shape[:2]:
        keep = np.array(
            Image.fromarray(keep.astype(np.uint8) * 255).resize(
                (rgb_rgb.shape[1], rgb_rgb.shape[0]), Image.NEAREST
            )
        ) > 127
    rgb_rgb[~keep] = 0
    out_path = debug_dir / "static0_rgb.png"
    Image.fromarray(rgb_rgb).save(out_path)
    return out_path
    # PROBLEM: SAM3 has no inherent gripper exclusion, so an
    # incorrectly-aligned URDF mask will let the gripper bleed into
    # SAM3's "graspable object" response. The mask used here is the
    # same one persisted to disk, so a bad mask is visible end-to-end.


def _save_anchor_intrinsics_and_depth(anchor: LiveFrame, intrinsics, artifact_dir: Path) -> tuple[Path, Path]:
    """Write the depth + intrinsics sidecars SAM3D expects when running
    full-image (no-crop) inference with a metric pointmap.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # Publisher already converted depth to float32 metres; just save it.
    depth_m = anchor.depth_m.astype(np.float32)
    depth_path = artifact_dir / "static0_full_depth_meters.tiff"
    Image.fromarray(depth_m).save(depth_path)
    intrinsics_path = artifact_dir / "static0_full_intrinsics.json"
    intrinsics_path.write_text(
        json.dumps(
            {
                "fx": float(intrinsics.fx),
                "fy": float(intrinsics.fy),
                "cx": float(intrinsics.cx),
                "cy": float(intrinsics.cy),
                "width": int(intrinsics.width),
                "height": int(intrinsics.height),
            },
            indent=2,
        )
        + "\n"
    )
    return depth_path, intrinsics_path
    # PROBLEM: depth is float-meters in a TIFF. If anything downstream
    # decides to read it as uint16 mm (the recorded layout's depth
    # convention), values will be off by 1000x silently. SAM3D itself
    # reads it correctly from the same path used today.


def _prompt_user(prompt_text: str) -> str:
    """Blocking input() on the main thread.

    Used twice: once for the SAM3 text prompt and once for the second
    Enter that ends static-view capture. Kept dead simple — no thread,
    no signal handling.

    Headless mode: if stdin is not a TTY (e.g. ns-train launched via
    nohup / detached), input() raises EOFError immediately. Rather than
    returning "" right away (which would collapse the static-capture
    window to ~0 frames), sleep AUTONOMOUS_PROMPT_HOLDOFF_S so the ROS
    publisher has time to accumulate keyframes. This lets the
    autonomous TAPIR test run end-to-end without an operator.
    """
    try:
        return input(prompt_text)
    except EOFError:
        holdoff = float(os.environ.get("AUTONOMOUS_PROMPT_HOLDOFF_S", "15"))
        if holdoff > 0:
            print(f"[live] (non-interactive: holding off {holdoff:.0f}s for headless capture)",
                  flush=True)
            time.sleep(holdoff)
        return ""
    # PROBLEM: blocks the main thread. ROS callbacks still run on
    # rospy's threadpool, but anything else on the main thread (e.g. a
    # progress spinner) is frozen. Acceptable: we only call this twice
    # in the whole pre-training session.


def _spawn_sam3d_in_thread(
    anchor_rgb_path: Path,
    sam3_objects: list,
    artifact_dir: Path,
    debug_dir: Path,
    depth_path: Path,
    intrinsics_path: Path,
) -> tuple[threading.Thread, dict]:
    """Run run_sam3d_multi_object_subprocess on a background thread.

    Returns (thread, result_slot). `result_slot` is mutated in-place by
    the thread: on success `result_slot["results"]` holds the per-object
    output dicts; on failure `result_slot["error"]` holds the exception.
    """
    result_slot: dict = {"results": None, "error": None, "finished": False}
    output_stems = [f"static0_obj_{i:02d}_sam3d" for i in range(len(sam3_objects))]
    mask_paths = [Path(obj["mask_path"]) for obj in sam3_objects]

    def _worker():
        try:
            result_slot["results"] = run_sam3d_multi_object_subprocess(
                render_image_path=anchor_rgb_path,
                object_mask_paths=mask_paths,
                output_dir=artifact_dir,
                output_stems=output_stems,
                image_dir=debug_dir,
                max_side=518,
                depth_path=depth_path,
                intrinsics_path=intrinsics_path,
            )
        except Exception as exc:
            result_slot["error"] = exc
        finally:
            result_slot["finished"] = True

    thread = threading.Thread(target=_worker, name="sam3d_subprocess", daemon=True)
    thread.start()
    return thread, result_slot
    # PROBLEM: SAM3D subprocess uses the same GPU as the eventual
    # static training. We block on this thread BEFORE static training
    # starts (in the cutover gate), so the GPU never has both running.
    # If the calling code forgets that gate, you'll get OOM the moment
    # the trainer's first forward pass collides with SAM3D.


def _register_seed_ply_path(static_dir: Path) -> None:
    """Write ``ply_file_path`` into static_scene/transforms.json so the
    dataparser inits Splatfacto from the seed PLY (Design Invariant #1: means
    locked on the TSDF seed). ``build_tsdf_seed`` writes the PLY but NOT this
    key — without it nerfstudio silently falls back to random init. The
    concurrent runner's stop_and_finalize writes it; the deferred batch path
    must do it here. Atomic write, idempotent."""
    tp = static_dir / "transforms.json"
    if not tp.exists():
        return
    try:
        meta = json.loads(tp.read_text())
    except Exception:
        return
    if meta.get("ply_file_path") == INIT_CLOUD_NAME:
        return
    meta["ply_file_path"] = INIT_CLOUD_NAME
    tmp = tp.with_name(f".{tp.name}.tmp")
    tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, tp)


def _seed_dynamic_scene_stub(static_dir: Path, dynamic_dir: Path) -> None:
    """Nerfstudio refuses an empty `dynamic_scene/`. Symlink the first
    static frame into `dynamic_scene/` so the dataparser builds.

    Live mode never reads from this stub; `_tracker_tick_live` ignores
    `_next_frame_to_track` and pulls every frame straight from the ROS
    subscriber. The stub exists only to satisfy the constructor's
    "must contain at least one training frame" check.
    """
    dynamic_dir.mkdir(parents=True, exist_ok=True)
    (dynamic_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (dynamic_dir / "depth").mkdir(parents=True, exist_ok=True)
    (dynamic_dir / "masks").mkdir(parents=True, exist_ok=True)

    static_meta = json.loads((static_dir / "transforms.json").read_text())
    if not static_meta.get("frames"):
        raise RuntimeError("static_scene has no recorded frames; cannot seed dynamic stub")
    first = static_meta["frames"][0]
    rgb_name = Path(first["file_path"]).name
    depth_name = Path(first["depth_file_path"]).name
    mask_name = Path(first["mask_path"]).name
    for sub, name in [("rgb", rgb_name), ("depth", depth_name), ("masks", mask_name)]:
        src = (static_dir / sub / name).resolve()
        dst = (dynamic_dir / sub / name)
        if dst.exists():
            dst.unlink()
        try:
            dst.symlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
    dyn_meta = {k: v for k, v in static_meta.items() if k not in {"frames", "ply_file_path"}}
    dyn_meta["frames"] = [
        {
            "file_path": first["file_path"],
            "depth_file_path": first["depth_file_path"],
            "mask_path": first["mask_path"],
            "transform_matrix": first["transform_matrix"],
        }
    ]
    (dynamic_dir / "transforms.json").write_text(json.dumps(dyn_meta, indent=2) + "\n")
    # PROBLEM: the symlinked stub depends on the static frame staying
    # on disk. We never delete static frames during the run, but a
    # post-run cleanup that wipes static_scene/ would invalidate the
    # symlinks. Punted — `live/` is a transient working dir.


def run_live_capture_session(sam3_prompt_text: Optional[str] = None) -> Path:
    """Drive the full interactive pre-training session.

    Args:
        sam3_prompt_text: SAM3 text prompt. If None or empty, falls
            back to ``DGS_SAM3_PROMPT`` env var, then to
            ``DEFAULT_SAM3_PROMPT``. **Never asked interactively** —
            the bootstrap script passes it on the command line so the
            user is not re-prompted.

    Flow (per the user's spec, 2026-05-31):
        1. Wipe LIVE_ROOT, start ROS subscriber, wait for first frame.
        2. **Start recording immediately.** Tell the user to move the
           arm toward the object of interest and press Enter when in
           front of it.
        3. Single Enter → capture the latest streamed frame as the
           SAM3 anchor. SAM3 runs **blocking** on that anchor (~1-2s).
           Recording keeps streaming in parallel.
        4. **If SAM3 returns 0 masks**: do NOT crash. Print
           "re-aim and press Enter to retry, or 'q' to abort", loop
           on Enter, capture a fresh latest frame, re-run SAM3 with
           the SAME prompt.
        5. Once SAM3 ≥ 1 mask: continue to Fast-SAM3D + sweep window
           (recording keeps running). Second Enter ends sweep.
        6. ICP+TSDF refinement, seed dynamic_scene stub, return.

    Returns LIVE_ROOT.

    Warm path: if LIVE_ROOT already holds a complete recording +
    SAM3 + SAM3D cache from a previous session, capture is skipped.
    """
    static_dir = LIVE_ROOT / "static_scene"
    dynamic_dir = LIVE_ROOT / "dynamic_scene"
    debug_dir = dynamic_dir / "initialization_debug"
    artifact_dir = dynamic_dir / "initialization_artifacts"

    # Resolve the SAM3 prompt without ever asking the user — argv > env > default.
    sam3_text = (sam3_prompt_text or os.environ.get("DGS_SAM3_PROMPT") or DEFAULT_SAM3_PROMPT).strip()
    if not sam3_text:
        sam3_text = DEFAULT_SAM3_PROMPT

    if _has_complete_recording_cache(LIVE_ROOT):
        # Prefer the new name; fall back to legacy post_fusion_state.pt.
        post_fusion_cache = LIVE_ROOT / "static_scene" / "static_state.pt"
        if not post_fusion_cache.is_file():
            legacy = LIVE_ROOT / "static_scene" / "post_fusion_state.pt"
            if legacy.is_file():
                post_fusion_cache = legacy
        tier = "Tier 2 (static-state cache)" if post_fusion_cache.is_file() else "Tier 1 (SAM3+SAM3D only)"
        print(
            f"\n==> [live] reusing cached recording + SAM3 + SAM3D from {LIVE_ROOT}\n"
            f"    cache level: {tier} — delete this folder to force a fresh recording.",
            flush=True,
        )
        sub = LiveShmSubscriber(wipe_live_root=False)
        sub.wait_for_first_frame(timeout_s=90.0)
        if post_fusion_cache.is_file():
            print("[live] ROS publisher ready; jumping straight to dynamic phase "
                  "(static + Phase 0b will be loaded from snapshot)", flush=True)
        else:
            print("[live] ROS publisher ready; jumping straight to static training "
                  "(no post-fusion snapshot — Phase 0b will run after static)", flush=True)
        return LIVE_ROOT

    # Publisher subprocess wipes LIVE_ROOT and recreates the
    # static_scene/ + dynamic_scene/ skeletons. We then mkdir the
    # debug/artifact subdirs that the SAM3/SAM3D workers write to.
    print(
        "[live] launching ROS publisher subprocess, "
        "waiting for first synced (rgb, depth, pose) tuple...",
        flush=True,
    )
    sub = LiveShmSubscriber(wipe_live_root=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    sub.wait_for_first_frame(timeout_s=90.0)
    print(f"[live] ready! SAM3 prompt: {sam3_text!r}", flush=True)

    # ------------------------------------------------------------------
    # Step 2: start recording immediately. The user moves the arm
    # toward the object while frames stream to disk; when they press
    # Enter, we grab whatever the latest streamed frame is and feed it
    # to SAM3. No "press Enter to start" — recording is automatic.
    # ------------------------------------------------------------------
    # `capture_anchor` returns the latest synced tuple and is the
    # canonical "this is the current view" snapshot; `start_recording`
    # then begins flushing every subsequent synced tuple to disk.
    # For the live spec we want to begin recording RIGHT NOW so the
    # operator's approach motion is captured, even before they press
    # Enter. The anchor we feed to SAM3 will be a FRESH snapshot taken
    # at the moment of Enter, not this bootstrap anchor.
    bootstrap_anchor = sub.capture_anchor()
    sub.start_recording(bootstrap_anchor)
    # Arm concurrent ICP+TSDF fusion. From here until `stop_recording`,
    # every newly-written keyframe is consumed by a worker thread and
    # fused into the TSDF volume. This covers the SAM3 retry loop +
    # SAM3D wait + post-SAM3D sweep — all wall-clock that was wasted
    # before. At the second Enter, `stop_and_finalize()` blocks only
    # for the drain tail + ~0.6 s `finalize()`, then writes the seed
    # PLY directly. Replaces the old build_static_init_pointcloud +
    # build_tsdf_seed two-pass post-capture step.
    from .fusion_runner import ConcurrentFusionRunner
    fusion_runner = ConcurrentFusionRunner(static_dir, sub.intrinsics)
    # DEFER the concurrent TSDF (default ON): building the TSDF volume during
    # capture grows it to ~7 GB, which collides with SAM3D's ~12 GB load at
    # Enter (+ Gazebo 2.6) → OOM on a 16 GB GPU. Instead, skip concurrent
    # fusion; after SAM3D unloads (Enter), build the seed with the batch
    # ICP+TSDF pass (build_tsdf_seed) when the GPU is free of SAM3D. The
    # constructor is kept (allocates zero GPU) so _finalize_safe stays valid
    # (its stop_and_finalize() is a no-op when .start() was never called).
    _defer_tsdf = os.environ.get("DGS_LIVE_DEFER_TSDF", "1") == "1"
    if not _defer_tsdf:
        fusion_runner.start()
    else:
        print("[live] concurrent TSDF DEFERRED (DGS_LIVE_DEFER_TSDF=1) — seed "
              "built by batch ICP+TSDF after SAM3D unloads (no VRAM collision)",
              flush=True)

    # Fresh timing ledger — capture is the true start of the live flow. The
    # capture/segmentation/3d-gen/fusion rows recorded here are rendered to
    # timing_report_capture.txt at the end of this function; the stage-2
    # ns-train static-gs run resets the ledger again for its own report.
    _t_capture_start = time.time()
    try:
        _tl.reset(LIVE_ROOT)
    except Exception:
        pass

    # Spawn the persistent SAM3+SAM3D worker and kick off SAM3's 6 s weight
    # load on a background thread. Capture wallclock is operator-controlled, so
    # by the time the operator centers the camera and hits Enter the segmenter
    # is already on the GPU.
    #
    # SAM3D-during-capture preload: ENABLED by default WHEN the TSDF is deferred
    # (the default). The preload hides SAM3D's ~28 s model-load behind the
    # operator sweep, so only the ~10 s infer is exposed at Enter (measured
    # 2026-06-11 vs the replay: dead-time-to-teleop-ready 76 s → 44 s).
    # History: the preload OOM'd ONLY because the CONCURRENT TSDF held ~1.5–7 GB
    # during capture (12 GB SAM3D load peak + 7 GB TSDF + 0.85 FastSAM + 2.6
    # Gazebo > 16 GB). With DGS_LIVE_DEFER_TSDF=1 there is NO concurrent TSDF, so
    # the measured capture-time peak is 12.65 GB (3.2 GB headroom) — safe. It is
    # therefore gated OFF when deferred-TSDF is OFF (the old OOM condition), and a
    # failed load is best-effort (falls back to the at-Enter load). Override with
    # DGS_SAM3D_LOAD_DURING_CAPTURE=0/1.
    sam_worker: SamWorkerClient | None = None
    _sam3_load_thread: threading.Thread | None = None
    _sam3_load_err: dict = {"err": None, "seconds": 0.0}
    # SAM3D-during-capture preload state (fastsam backend only; SAM3 4.5 + SAM3D
    # don't co-fit). Best-effort: any failure leaves loaded=False and the Enter
    # path loads SAM3D then (the proven sequential fallback).
    _sam3d_preload: dict = {"loaded": False, "seconds": 0.0, "err": None}
    _preload_sam3d = (SEGMENTATION_BACKEND == "fastsam"
                      and os.environ.get("DGS_SAM3D_LOAD_DURING_CAPTURE",
                                         "1" if _defer_tsdf else "0") == "1")
    try:
        sam_worker = SamWorkerClient(conda_env=SAM3_CONDA_ENV)
        print(f"[live] SAM worker spawned ({sam_worker.spawn_seconds:.2f}s)", flush=True)

        def _bg_load_sam3() -> None:
            try:
                t0 = time.time()
                if SEGMENTATION_BACKEND == "fastsam":
                    sam_worker.load_fastsam()
                else:
                    sam_worker.load_sam3(confidence_threshold=SAM3_CONFIDENCE_THRESHOLD)
                _sam3_load_err["seconds"] = time.time() - t0
                _tl.record(LIVE_ROOT, "capture",
                           "FastSAM+CLIP" if SEGMENTATION_BACKEND == "fastsam" else "SAM3",
                           "load", t0, time.time())
                # Preload SAM3D during capture so its model-load hides behind the
                # sweep. Best-effort — on OOM/any error we fall back to loading it
                # at Enter (see the load_sam3d call site).
                if _preload_sam3d:
                    try:
                        ts = time.time()
                        sam_worker.load_sam3d()
                        _sam3d_preload["loaded"] = True
                        _sam3d_preload["seconds"] = time.time() - ts
                        _tl.record(LIVE_ROOT, "capture", "SAM3D (trim)", "load", ts, time.time())
                        print(f"[live] SAM3D preloaded during capture "
                              f"({_sam3d_preload['seconds']:.1f}s) — its load now hides behind the sweep",
                              flush=True)
                    except Exception as exc:
                        _sam3d_preload["err"] = exc
                        print(f"[live] SAM3D preload-during-capture failed ({exc}); "
                              f"will load at Enter instead", flush=True)
            except Exception as exc:
                _sam3_load_err["err"] = exc

        _sam3_load_thread = threading.Thread(
            target=_bg_load_sam3, name="sam3_bg_load", daemon=True,
        )
        _sam3_load_thread.start()
    except Exception as exc:
        print(f"[live] WARNING: persistent SAM worker spawn failed ({exc}); "
              f"falling back to per-call subprocess", flush=True)
        sam_worker = None

    print("[live] recording started — move the arm toward the object of interest.\n"
          "       press ENTER when the camera is centered on it.", flush=True)

    # ------------------------------------------------------------------
    # Step 3 + 4: Enter → capture latest frame → SAM3 blocking → retry
    # on zero masks. Loop until SAM3 returns ≥ 1 mask (or 'q' to abort).
    # ------------------------------------------------------------------
    sam3_objects: list = []
    sam3_duration: float = 0.0
    anchor: LiveFrame = bootstrap_anchor  # placeholder, overwritten on Enter
    anchor_rgb_path: Path = debug_dir / "static0_rgb.png"
    depth_path: Path = artifact_dir / "static0_full_depth_meters.tiff"
    intrinsics_path: Path = artifact_dir / "static0_full_intrinsics.json"

    # M2: any exit path through this block (SAM3 abort, exception, success)
    # must still call fusion_runner.stop_and_finalize so the user gets a seed
    # PLY they can resume from. Set by SAM3-abort or any exception below.
    _finalize_done = {"value": False}

    def _finalize_safe(reason: str) -> None:
        if _finalize_done["value"]:
            return
        _finalize_done["value"] = True
        print(f"[live] finalizing fusion ({reason})...", flush=True)
        _t_fin = time.time()
        try:
            fusion_runner.stop_and_finalize()
            _tl.record(LIVE_ROOT, "pointcloud_fusion", "TSDF finalize+seed", "fusion",
                       _t_fin, time.time())
            # Per-frame add_frame breakdown (concurrent path only). In the
            # deferred-TSDF default the worker never ran, so per_frame_add_stats
            # returns None → no row (the batch GPU seed has its own row). The
            # mean fills the timed slot; p90/max/n/fail go in the name string
            # since the ledger has no extra-fields mechanism.
            _pf = fusion_runner.per_frame_add_stats()
            if _pf is not None:
                _mean_s = _pf["mean_ms"] / 1000.0
                _t_pf = time.time()
                _tl.record(
                    LIVE_ROOT, "pointcloud_fusion",
                    (f"TSDF per-frame add_frame (mean; p90={_pf['p90_ms']:.0f}ms "
                     f"max={_pf['max_ms']:.0f}ms n={_pf['n']} fail={_pf['fail']})"),
                    "fusion", _t_pf - _mean_s, _t_pf,
                )
        except Exception as exc:
            print(f"[live] WARNING: fusion finalize on '{reason}' failed: {exc}", flush=True)

    # Counts SAM3 attempts that returned 0 objects. Each failed attempt's
    # input RGB + (empty/garbage) segmentation overview is preserved under
    # `failed_segmentations/attempt_NN_*` so the operator can inspect them
    # afterwards and judge whether it's a threshold issue. Without this the
    # next attempt overwrites `static0_rgb.png` and the failure is lost.
    _failed_seg_dir = debug_dir / "failed_segmentations"
    _seg_attempt = 0

    while True:
        reply = _prompt_user("").strip().lower()
        if reply in ("q", "quit", "abort"):
            sub.stop_recording()
            if sam_worker is not None:
                try:
                    sam_worker.close()
                except Exception:
                    pass
            _finalize_safe("SAM3-abort")
            raise RuntimeError("user aborted SAM3 retry loop")

        # Grab the latest streamed frame as the SAM3 anchor. We do NOT
        # call `start_recording` again — recording is already running
        # and the publisher keeps streaming during SAM3.
        anchor = sub.capture_anchor()
        anchor_rgb_path = _save_anchor_for_sam3(anchor, debug_dir)
        depth_path, intrinsics_path = _save_anchor_intrinsics_and_depth(
            anchor, sub.intrinsics, artifact_dir
        )
        print(f"[live] SAM3 input frame captured (seq={anchor.seq})", flush=True)

        # Free any CUDA reservation held by the parent before the
        # SAM3 subprocess starts its first cuBLAS call.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[live] running SAM3 (blocking; prompt: {sam3_text!r})...", flush=True)
        t_sam3 = time.time()
        try:
            if sam_worker is not None:
                # Block on the background load (almost always already done
                # by the time the user pressed Enter).
                if _sam3_load_thread is not None and _sam3_load_thread.is_alive():
                    _sam3_load_thread.join()
                if _sam3_load_err["err"] is not None:
                    raise _sam3_load_err["err"]
                if SEGMENTATION_BACKEND == "fastsam":
                    sam3_objects = sam_worker.fastsam_infer(
                        image_path=anchor_rgb_path,
                        text_prompt=sam3_text,
                        output_dir=debug_dir,
                        output_stem="static0",
                        min_area_ratio=SAM3_CANDIDATE_MIN_AREA_RATIO,
                        max_area_ratio=SAM3_CANDIDATE_MAX_AREA_RATIO,
                        dedup_iou=SAM3_CANDIDATE_DEDUP_IOU,
                        max_objects=SAM3_CANDIDATE_MAX_OBJECTS,
                        min_score=SAM3_MIN_SCORE,
                    ) or []
                else:
                    sam3_objects = sam_worker.sam3_infer(
                        image_path=anchor_rgb_path,
                        text_prompt=sam3_text,
                        output_dir=debug_dir,
                        output_stem="static0",
                        min_area_ratio=SAM3_CANDIDATE_MIN_AREA_RATIO,
                        max_area_ratio=SAM3_CANDIDATE_MAX_AREA_RATIO,
                        dedup_iou=SAM3_CANDIDATE_DEDUP_IOU,
                        max_objects=SAM3_CANDIDATE_MAX_OBJECTS,
                        min_score=SAM3_MIN_SCORE,
                    ) or []
            else:
                _seg_subprocess = (run_fastsam_subprocess if SEGMENTATION_BACKEND == "fastsam"
                                   else run_sam3_subprocess)
                _seg_kwargs = dict(
                    image_path=anchor_rgb_path,
                    text_prompt=sam3_text,
                    output_dir=debug_dir,
                    output_stem="static0",
                    sam3_conda_env=SAM3_CONDA_ENV,
                    min_area_ratio=SAM3_CANDIDATE_MIN_AREA_RATIO,
                    max_area_ratio=SAM3_CANDIDATE_MAX_AREA_RATIO,
                    dedup_iou=SAM3_CANDIDATE_DEDUP_IOU,
                    max_objects=SAM3_CANDIDATE_MAX_OBJECTS,
                    min_score=SAM3_MIN_SCORE,
                )
                if SEGMENTATION_BACKEND != "fastsam":
                    _seg_kwargs["confidence_threshold"] = SAM3_CONFIDENCE_THRESHOLD
                sam3_objects = _seg_subprocess(**_seg_kwargs) or []
        except Exception as exc:
            sam3_objects = []
            print(f"[live] SAM3 raised: {exc}", flush=True)
        sam3_duration = time.time() - t_sam3
        if sam3_objects:
            # Segmenter is already warm (loaded during capture) so this is ~infer.
            _tl.record(LIVE_ROOT, "segmentation",
                       "FastSAM" if SEGMENTATION_BACKEND == "fastsam" else "SAM3",
                       "infer", t_sam3, t_sam3 + sam3_duration)

        if sam3_objects:
            print(f"[live] SAM3: found {len(sam3_objects)} graspable masks "
                  f"({sam3_duration:.1f}s)", flush=True)
            break

        # Preserve this failed attempt before the next retry overwrites the
        # debug files. Copy the input RGB + the (empty) segmentation overview
        # into failed_segmentations/attempt_NN_* so it can be reviewed later
        # (e.g. to decide if SAM3_MIN_SCORE / fastsam_conf is too strict).
        _seg_attempt += 1
        try:
            _failed_seg_dir.mkdir(parents=True, exist_ok=True)
            for _src_name, _dst_suffix in (
                ("static0_rgb.png", "input_rgb.png"),
                ("static0_sam3_overview.png", "overview.png"),
            ):
                _src = debug_dir / _src_name
                if _src.exists():
                    shutil.copy2(_src, _failed_seg_dir / f"attempt_{_seg_attempt:02d}_{_dst_suffix}")
            print(f"[live] saved failed segmentation attempt {_seg_attempt} -> "
                  f"{_failed_seg_dir}/attempt_{_seg_attempt:02d}_*", flush=True)
        except Exception as _exc:
            print(f"[live] WARNING: could not save failed segmentation attempt: {_exc}", flush=True)

        print(f"[live] SAM3 found 0 objects (took {sam3_duration:.1f}s).\n"
              f"       re-aim the camera and press ENTER to retry, or 'q' to abort.",
              flush=True)

    # The single Enter ends static-view capture: the sweep happened BEFORE
    # it, and the frame just captured is the SAM3D anchor. Stop recording
    # NOW (no post-SAM3D sweep stage) so capture is over before Gazebo is
    # paused for the SAM3D compute window.
    _t_capture_end = time.time()
    sub.stop_recording()
    n_static = sub.num_recorded_frames()
    print(f"[live] recording stopped after {n_static} keyframes", flush=True)

    # Operator sweep duration: wall-clock the user spent moving the arm and
    # capturing static keyframes — from recording-armed (_t_capture_start, just
    # after start_recording) to the single Enter that ends static-view capture
    # (the SAM3 retry loop broke on success → _t_capture_end captured above).
    # This is operator-controlled, not compute, so it's the slowest segment by
    # design; the row makes that explicit in the report.
    _tl.record(LIVE_ROOT, "capture",
               f"operator sweep ({n_static} keyframes)", "infer",
               _t_capture_start, _t_capture_end)

    # Pause Gazebo physics from here through the end of the init-PLY
    # build. The window covers SAM3D subprocess and the depth-back-
    # projection PLY assembly — both compete with gzserver for CPU/GPU.
    pause_gazebo_physics(sub)
    try:
        # Persist the measured SAM3 duration to a sidecar JSON. The
        # pipeline reads this in ``_run_sam3_and_sam3d_generation`` and
        # re-injects it into ``self._timing["S0.1_sam3_segmentation"]``
        # so the timing report shows the real subprocess wall-clock.
        live_timings_path = artifact_dir / "live_sam3_timings.json"
        live_timings_path.write_text(json.dumps({
            "S0.1_sam3_segmentation": float(sam3_duration),
        }, indent=2) + "\n")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Swap the segmenter out and SAM3D in. Measured: SAM3 4.5 + SAM3D-trim
        # 11.7 = 16.2 GB (over); FastSAM 1.9 + SAM3D-trim 11.7 = 13.6 GB (fits),
        # but we still unload the segmenter first to keep headroom for Gazebo.
        if sam_worker is not None:
            try:
                if SEGMENTATION_BACKEND == "fastsam":
                    sam_worker.unload_fastsam()
                    print("[live] FastSAM unloaded from worker", flush=True)
                else:
                    sam_worker.unload_sam3()
                    print("[live] SAM3 unloaded from worker", flush=True)
            except Exception as exc:
                print(f"[live] WARNING: segmenter unload failed: {exc}", flush=True)

        print(f"[live] running SAM3D on {len(sam3_objects)} object(s)", flush=True)
        sam3d_results: list = [{} for _ in sam3_objects]
        try:
            if sam_worker is not None:
                t_load = time.time()
                if _sam3d_preload["loaded"]:
                    # Already resident from the capture-window preload — the
                    # ~23 s model-load is fully hidden; nothing to wait for.
                    print(f"[live] SAM3D already resident (preloaded during capture, "
                          f"{_sam3d_preload['seconds']:.1f}s) — skipping load", flush=True)
                else:
                    sam_worker.load_sam3d()
                    print(f"[live] SAM3D model loaded ({time.time()-t_load:.1f}s)", flush=True)
                    # Exposed load (preload didn't happen) — record it.
                    _tl.record(LIVE_ROOT, "object_3d_gen", "SAM3D model", "load", t_load, time.time())
                t_infer = time.time()
                # Crop each object to a square around its mask BEFORE SAM3D,
                # instead of feeding the full frame (which SAM3D resizes to
                # max_side=518 — squashing a 1920x1200 scene so a small object
                # becomes a few pixels -> garbage 3D). prepare_cropped_sam3d_inputs
                # makes a tight crop (object bbox + 32 px context, min 300 px
                # square) and writes crop-shifted intrinsics + cropped depth so
                # the metric pointmap stays correct. The object then fills the
                # 518 frame -> real detail. The worker takes ONE shared render/
                # depth, so we crop per-object and call it once per object.
                from .sam3d import prepare_cropped_sam3d_inputs
                cam_intr = json.loads(Path(intrinsics_path).read_text())
                worker_results = []
                for _i, obj in enumerate(sam3_objects):
                    _stem = f"static0_obj_{_i:02d}_sam3d"
                    try:
                        _crop = prepare_cropped_sam3d_inputs(
                            render_image_path=anchor_rgb_path,
                            object_mask_path=Path(obj["mask_path"]),
                            output_dir=artifact_dir,
                            output_stem=_stem,
                            image_dir=debug_dir,
                            depth_path=depth_path,
                            depth_scale=1.0,           # depth tiff is float32 METERS
                            camera_intrinsics=cam_intr,
                        )
                    except Exception as _exc:
                        print(f"[live] SAM3D crop failed for obj {_i} ({_exc}); "
                              f"falling back to full-frame", flush=True)
                        worker_results.extend(sam_worker.sam3d_infer(
                            render_image_path=anchor_rgb_path,
                            object_mask_paths=[Path(obj["mask_path"])],
                            output_stems=[_stem],
                            output_dir=artifact_dir, image_dir=debug_dir,
                            max_side=518, depth_path=depth_path,
                            intrinsics_path=intrinsics_path))
                        continue
                    _r = sam_worker.sam3d_infer(
                        render_image_path=_crop["render_image_path"],
                        object_mask_paths=[_crop["object_mask_path"]],
                        output_stems=[_stem],
                        output_dir=artifact_dir,
                        image_dir=debug_dir,
                        max_side=518,
                        depth_path=_crop.get("depth_path"),
                        intrinsics_path=_crop.get("intrinsics_path"),
                    )
                    worker_results.extend(_r)
                _tl.record(LIVE_ROOT, "object_3d_gen", "SAM3D", "infer", t_infer, time.time())
                print(f"[live] SAM3D inference {time.time()-t_infer:.1f}s "
                      f"({len(worker_results)} masks, cropped per-object)", flush=True)
                # Reconstruct the downstream-compatible per-object dict shape:
                # {ply_path, pose_path, preview_path, ...} when ok, else {}.
                from .sam3d import get_sam3d_output_paths, resolve_sam3d_pose_path
                for i, r in enumerate(worker_results):
                    if not isinstance(r, dict) or r.get("status") != "ok":
                        sam3d_results[i] = {}
                        continue
                    stem = f"static0_obj_{i:02d}_sam3d"
                    paths = get_sam3d_output_paths(artifact_dir, stem, image_dir=debug_dir)
                    resolved = resolve_sam3d_pose_path(paths["ply_path"], paths["pose_path"])
                    if paths["ply_path"].exists() and sam3d_pose_has_rotation(resolved):
                        if resolved is not None:
                            paths["pose_path"] = resolved
                        sam3d_results[i] = paths
                    else:
                        sam3d_results[i] = {}
                # SAM3D fully done — unload before TSDF finalize's 12.3 GB peak.
                try:
                    sam_worker.unload_sam3d()
                except Exception as exc:
                    print(f"[live] WARNING: SAM3D unload failed: {exc}", flush=True)
            else:
                sam3d_thread, sam3d_slot = _spawn_sam3d_in_thread(
                    anchor_rgb_path=anchor_rgb_path,
                    sam3_objects=sam3_objects,
                    artifact_dir=artifact_dir,
                    debug_dir=debug_dir,
                    depth_path=depth_path,
                    intrinsics_path=intrinsics_path,
                )
                last_log = time.time()
                while not sam3d_slot["finished"]:
                    sam3d_thread.join(timeout=5.0)
                    if not sam3d_slot["finished"] and (time.time() - last_log) >= 5.0:
                        print("[live] still waiting for SAM3D...", flush=True)
                        last_log = time.time()
                if sam3d_slot["error"] is not None:
                    raise sam3d_slot["error"]
                sam3d_results = sam3d_slot["results"] or [{} for _ in sam3_objects]
        except Exception as exc:
            print(f"[live] SAM3D failed: {exc}", flush=True)
            sam3d_results = [{} for _ in sam3_objects]
        n_ok = sum(1 for r in sam3d_results if r)
        for i, r in enumerate(sam3d_results):
            print(f"[live] SAM3D obj {i}: {'ok' if r else 'failed'}", flush=True)
        print(f"[live] SAM3D done: {n_ok}/{len(sam3_objects)} objects ready", flush=True)

        # Eager AnySplat pre-spawn (DGS_EAGER_ANYSPLAT=1, set by
        # bootstrap_live.sh): fire-and-forget a detached FIFO-mode worker NOW
        # so its model load (~17 s, ~3.5 GB VRAM) overlaps the sweep + static
        # training. Stage 3 (dynamic-gs-live) adopts it instead of paying the
        # load at go-live time. SAM3D just unloaded, so its VRAM slot is free.
        # In deferred-TSDF mode the spawn moves to AFTER the TSDF build:
        # its ~3.5 GB load otherwise overlaps the GPU TSDF and (measured at
        # 1920x1200) tips it into OOM. The load still hides behind static
        # training + Phase 0b, so nothing is exposed either way.
        if os.environ.get("DGS_EAGER_ANYSPLAT") == "1" and not _defer_tsdf:
            try:
                from .anysplat_decode import spawn_detached_anysplat_worker
                _as_pid = spawn_detached_anysplat_worker(LIVE_ROOT / ".anysplat_worker")
                print(f"[live] AnySplat worker pre-spawned (pid={_as_pid}); "
                      f"model loads in background, dynamic-gs-live will adopt it",
                      flush=True)
            except Exception as exc:
                print(f"[live] WARNING: eager AnySplat pre-spawn failed: {exc} "
                      f"(go-live will load it as usual)", flush=True)

        # Recording already stopped at the single Enter (above) — there is
        # no post-SAM3D sweep. Gazebo stays paused through the TSDF seed
        # build; the finally-block unpause is the idempotent safety net.
        print("[live] SAM3D complete — building static seed.", flush=True)

        # Concurrent ICP+TSDF fusion has been running on a worker thread
        # since `start_recording`; here we just drain the tail + run
        # `finalize()` (~0.6 s) and the seed PLY lands at
        # `<static>/depth_camera_init_points.ply` with transforms.json
        # `ply_file_path` updated. Replaces the legacy
        # `build_static_init_pointcloud` (naive back-projection) +
        # `rgbd_fusion_init.build_tsdf_seed` (post-pass ICP+TSDF refine)
        # — both passes are subsumed by the streaming runner.
        if _defer_tsdf:
            # Free the SAM worker's whole CUDA context + allocator cache BEFORE
            # the TSDF: even with both models unloaded the worker process holds
            # GBs of cached VRAM, which at 1920x1200 pushed free memory below
            # the GPU-TSDF threshold (-> 127s CPU fallback). The finally-block
            # close below stays as a no-op safety net.
            if sam_worker is not None:
                try:
                    sam_worker.close()
                    print("[live] SAM worker closed before deferred TSDF (frees VRAM)", flush=True)
                except Exception as exc:
                    print(f"[live] WARNING: early SAM worker close failed: {exc}", flush=True)
                sam_worker = None
            # Deferred path: SAM3D + FastSAM are unloaded, so the GPU is free
            # for the batch ICP+TSDF. fuse_recorded_dataset runs the GPU
            # OnlineFusion (auto-CUDA, ~5s vs ~48s for the CPU build_tsdf_seed)
            # AND writes ply_file_path into transforms.json itself, so Splatfacto
            # inits from the seed. No concurrent VBG ever collided with SAM3D.
            # We then bump the seed PLY mtime PAST the transforms.json rewrite so
            # bootstrap stage 1.5's _output_is_fresh check sees the seed as fresh
            # and skips the (previously ~45s) redundant rebuild.
            try:
                # SUBPROCESS isolation: a GPU OOM in Open3D poisons its CUDA
                # memory cache and ABORTS the process at teardown (measured at
                # 1920x1200) — in-process the whole capture dies even though
                # the work succeeded. In a subprocess the abort is contained
                # and we fall back to CPU cleanly. DGS_TSDF_VOXEL_M=0.003 for
                # the GPU attempt: at 1200p/110deg the 2 mm VoxelBlockGrid
                # hashmap OOMs 16 GB even with ~12 GB free; 3 mm fits.
                import subprocess as _sp
                t_fin = time.time()
                _env = dict(os.environ)
                _env.setdefault("DGS_TSDF_VOXEL_M", "0.003")
                _r = _sp.run(
                    [sys.executable, "-m", "dynamic_gs.utils.online_fusion", str(static_dir)],
                    env=_env, capture_output=True, text=True, timeout=300,
                )
                if _r.returncode != 0:
                    raise RuntimeError(
                        f"GPU TSDF subprocess rc={_r.returncode}: "
                        f"{(_r.stderr or '')[-300:]}")
                seed_ply = static_dir / INIT_CLOUD_NAME
                if not seed_ply.exists():
                    raise RuntimeError("GPU TSDF subprocess wrote no seed PLY")
                seed_ply.touch()
                _tl.record(LIVE_ROOT, "pointcloud_fusion", "TSDF batch seed (GPU)",
                           "fusion", t_fin, time.time())
                print("[live] deferred GPU TSDF seed built + ply_file_path registered",
                      flush=True)
            except Exception as exc:
                print(f"[live] WARNING: deferred GPU TSDF failed ({exc}); "
                      f"trying CPU build_tsdf_seed", flush=True)
                try:
                    from .rgbd_fusion_init import build_tsdf_seed
                    cpu_ply = build_tsdf_seed(LIVE_ROOT, force=True, verbose=True)
                    _register_seed_ply_path(static_dir)
                    cpu_ply.touch()
                except Exception as exc2:
                    print(f"[live] WARNING: CPU TSDF also failed ({exc2}); "
                          f"falling back to naive back-projected seed", flush=True)
                    try:
                        sub.build_static_init_pointcloud()
                    except Exception as exc3:
                        print(f"[live] WARNING: naive seed fallback also failed ({exc3})", flush=True)
            if os.environ.get("DGS_EAGER_ANYSPLAT") == "1":
                try:
                    from .anysplat_decode import spawn_detached_anysplat_worker
                    _as_pid = spawn_detached_anysplat_worker(LIVE_ROOT / ".anysplat_worker")
                    print(f"[live] AnySplat worker pre-spawned post-TSDF (pid={_as_pid})",
                          flush=True)
                except Exception as exc:
                    print(f"[live] WARNING: eager AnySplat pre-spawn failed: {exc}", flush=True)
        else:
            try:
                _finalize_safe("happy-path")
            except Exception as exc:
                print(f"[live] WARNING: concurrent fusion finalize failed ({exc}); "
                      f"falling back to legacy naive seed + post-pass refine", flush=True)
                sub.build_static_init_pointcloud()
                try:
                    from .rgbd_fusion_init import build_tsdf_seed
                    build_tsdf_seed(LIVE_ROOT, force=True, verbose=True)
                except Exception as exc2:
                    print(f"[live] WARNING: legacy fallback also failed ({exc2}); "
                          f"seed will be naive only", flush=True)
    finally:
        # Close the persistent SAM worker BEFORE finalize_safe so its
        # remaining VRAM (if any) is freed before TSDF's 12.3 GB peak.
        if sam_worker is not None:
            try:
                sam_worker.close()
            except Exception as exc:
                print(f"[live] WARNING: SAM worker close failed: {exc}", flush=True)
        # M2: even if SAM3D or the sweep failed, make sure we left a seed
        # PLY so the user can resume from a static_scene that has one.
        _finalize_safe("finally-block")
        # PCD build done (or aborted); resume the simulator before
        # static training starts.
        unpause_gazebo_physics(sub)

    _seed_dynamic_scene_stub(static_dir, dynamic_dir)

    # Render the capture-phase by-phase load/inference report. Stage-2 ns-train
    # static-gs resets the ledger for its own report, so capture its timings now.
    try:
        report = _tl.render(LIVE_ROOT)
        (LIVE_ROOT / "timing_report_capture.txt").write_text(report + "\n")
        print(f"[live] capture timing report → {LIVE_ROOT / 'timing_report_capture.txt'}", flush=True)
    except Exception as exc:
        print(f"[live] capture timing report failed: {exc}", flush=True)

    print("[live] static capture complete; starting static training", flush=True)
    return LIVE_ROOT
    # PROBLEM: if the user closes stdin (e.g. `nohup ns-train ... < /dev/null`),
    # both _prompt_user calls return immediately with "" — the workflow
    # collapses and we record almost nothing. Live mode is by design an
    # interactive flow, not a batch one.
