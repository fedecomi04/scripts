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
    fusion_runner.start()

    # Spawn the persistent SAM3+SAM3D worker and kick off SAM3's 6 s weight
    # load on a background thread. Capture wallclock is typically ≥30 s
    # (operator-controlled), so by the time the operator centers the camera
    # and hits Enter, SAM3 is already on the GPU. VRAM during capture is
    # ~3 GB (TSDF integrate) + 4.5 GB (SAM3 resident) = 7.5 GB on 16 GB.
    # Safe.
    sam_worker: SamWorkerClient | None = None
    _sam3_load_thread: threading.Thread | None = None
    _sam3_load_err: dict = {"err": None, "seconds": 0.0}
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
        try:
            fusion_runner.stop_and_finalize()
        except Exception as exc:
            print(f"[live] WARNING: fusion finalize on '{reason}' failed: {exc}", flush=True)

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
            print(f"[live] SAM3: found {len(sam3_objects)} graspable masks "
                  f"({sam3_duration:.1f}s)", flush=True)
            break

        print(f"[live] SAM3 found 0 objects (took {sam3_duration:.1f}s).\n"
              f"       re-aim the camera and press ENTER to retry, or 'q' to abort.",
              flush=True)

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
                sam_worker.load_sam3d()
                print(f"[live] SAM3D model loaded ({time.time()-t_load:.1f}s)", flush=True)
                t_infer = time.time()
                worker_results = sam_worker.sam3d_infer(
                    render_image_path=anchor_rgb_path,
                    object_mask_paths=[Path(obj["mask_path"]) for obj in sam3_objects],
                    output_stems=[f"static0_obj_{i:02d}_sam3d" for i in range(len(sam3_objects))],
                    output_dir=artifact_dir,
                    image_dir=debug_dir,
                    max_side=518,
                    depth_path=depth_path,
                    intrinsics_path=intrinsics_path,
                )
                print(f"[live] SAM3D inference {time.time()-t_infer:.1f}s "
                      f"({len(worker_results)} masks)", flush=True)
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
        if os.environ.get("DGS_EAGER_ANYSPLAT") == "1":
            try:
                from .anysplat_decode import spawn_detached_anysplat_worker
                _as_pid = spawn_detached_anysplat_worker(LIVE_ROOT / ".anysplat_worker")
                print(f"[live] AnySplat worker pre-spawned (pid={_as_pid}); "
                      f"model loads in background, dynamic-gs-live will adopt it",
                      flush=True)
            except Exception as exc:
                print(f"[live] WARNING: eager AnySplat pre-spawn failed: {exc} "
                      f"(go-live will load it as usual)", flush=True)

        # Sweep window: recording is still running. Let the operator
        # sweep additional views of the scene to give the static
        # optimiser more coverage. Press Enter to end the sweep.
        print("[live] SAM3D complete — sweep the scene to capture more views.\n"
              "       press ENTER when done capturing static views.", flush=True)
        _prompt_user("")
        sub.stop_recording()
        n_static = sub.num_recorded_frames()
        print(f"[live] recording stopped after {n_static} keyframes", flush=True)

        # Concurrent ICP+TSDF fusion has been running on a worker thread
        # since `start_recording`; here we just drain the tail + run
        # `finalize()` (~0.6 s) and the seed PLY lands at
        # `<static>/depth_camera_init_points.ply` with transforms.json
        # `ply_file_path` updated. Replaces the legacy
        # `build_static_init_pointcloud` (naive back-projection) +
        # `rgbd_fusion_init.build_tsdf_seed` (post-pass ICP+TSDF refine)
        # — both passes are subsumed by the streaming runner.
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

    print("[live] static capture complete; starting static training", flush=True)
    return LIVE_ROOT
    # PROBLEM: if the user closes stdin (e.g. `nohup ns-train ... < /dev/null`),
    # both _prompt_user calls return immediately with "" — the workflow
    # collapses and we record almost nothing. Live mode is by design an
    # interactive flow, not a batch one.
