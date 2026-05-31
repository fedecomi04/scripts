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
from .sam3d import run_sam3d_multi_object_subprocess, sam3d_pose_has_rotation

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
DEFAULT_SAM3_PROMPT = "the can of coke on the table"
SAM3_CONDA_ENV = "sam3_dynamic_gs"
SAM3_CANDIDATE_MIN_AREA_RATIO = 0.002
SAM3_CANDIDATE_MAX_AREA_RATIO = 0.25
SAM3_CANDIDATE_DEDUP_IOU = 0.6
SAM3_CANDIDATE_MAX_OBJECTS = 8
SAM3_CONFIDENCE_THRESHOLD = 0.3
SAM3_MIN_SCORE = 0.44


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


def run_live_capture_session() -> Path:
    """Drive the full interactive pre-training session.

    Returns LIVE_ROOT (which now contains a populated static_scene/,
    a stub dynamic_scene/, the SAM3+SAM3D cache, and the SfM init
    PLY).

    Warm path: if LIVE_ROOT already holds a complete recording +
    SAM3 + SAM3D cache from a previous session, the interactive
    capture is skipped entirely. The ROS publisher still spawns
    (we need live frames for the dynamic loop), but no prompts,
    no recording, no SAM3, no SAM3D — the pipeline downstream will
    reuse the on-disk artifacts via `sam3_reuse_cached=True`.
    """
    static_dir = LIVE_ROOT / "static_scene"
    dynamic_dir = LIVE_ROOT / "dynamic_scene"
    debug_dir = dynamic_dir / "initialization_debug"
    artifact_dir = dynamic_dir / "initialization_artifacts"

    if _has_complete_recording_cache(LIVE_ROOT):
        post_fusion_cache = LIVE_ROOT / "static_scene" / "post_fusion_state.pt"
        tier = "Tier 2 (post-fusion state)" if post_fusion_cache.is_file() else "Tier 1 (SAM3+SAM3D only)"
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
        "[live] launching ROS publisher subprocess (radiance_ros_4060), "
        "waiting for first synced (rgb, depth, pose) tuple...",
        flush=True,
    )
    # Subscriber spawns the publisher in the ROS env and blocks until
    # the publisher reports "ready" on stdout. The publisher itself
    # waits for /camera_info before signalling ready, so by the time
    # this returns we have intrinsics.
    sub = LiveShmSubscriber(wipe_live_root=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    sub.wait_for_first_frame(timeout_s=90.0)
    print("ready!", flush=True)

    user_prompt = _prompt_user(
        f"press enter for default prompt ('{DEFAULT_SAM3_PROMPT}'), "
        f"or write a specific prompt for SAM3: "
    ).strip()
    sam3_text = user_prompt if user_prompt else DEFAULT_SAM3_PROMPT

    _prompt_user(
        "move the robot in front of the object that will be manipulated. "
        "press enter when you are in front of it: "
    )

    anchor = sub.capture_anchor()
    sub.start_recording(anchor)
    anchor_rgb_path = _save_anchor_for_sam3(anchor, debug_dir)
    depth_path, intrinsics_path = _save_anchor_intrinsics_and_depth(
        anchor, sub.intrinsics, artifact_dir
    )
    print(f"[live] SAM3 input frame captured (seq={anchor.seq})", flush=True)

    # Free any CUDA reservation held by the parent (torch import,
    # nerfstudio trainer setup, pyrender EGL warmup) so the SAM3
    # subprocess's first cuBLAS call has clean VRAM.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Launch SAM3 in a background thread so the operator can press
    # enter to stop recording at any time — even before SAM3 finishes.
    # Recording (rospy callback thread) keeps writing keyframes to
    # disk in parallel, gated by the ORB-SLAM filter so we don't
    # bloat the dataset with near-duplicate views. GPU contention
    # between SAM3 and the URDF mask renderer is accepted; URDF
    # render slows down but doesn't fail under shared VRAM.
    print(f"[live] recording started; SAM3 running in background (prompt: {sam3_text!r}). "
          "press enter when you are done capturing static views.", flush=True)
    sam3_slot = {"objects": None, "error": None, "finished": False}

    def _run_sam3_worker():
        # Measured here (not in the pipeline) because in live mode the
        # pipeline's ``_run_sam3_and_sam3d_generation`` only re-checks
        # the cached SAM3 results and so its S0.1 timer reports ~0s.
        # The duration captured here is written to a sidecar JSON below
        # and re-injected into self._timing["S0.1_sam3_segmentation"]
        # by the pipeline.
        t_sam3 = time.time()
        try:
            sam3_slot["objects"] = run_sam3_subprocess(
                image_path=anchor_rgb_path,
                text_prompt=sam3_text,
                output_dir=debug_dir,
                output_stem="static0",
                sam3_conda_env=SAM3_CONDA_ENV,
                min_area_ratio=SAM3_CANDIDATE_MIN_AREA_RATIO,
                max_area_ratio=SAM3_CANDIDATE_MAX_AREA_RATIO,
                dedup_iou=SAM3_CANDIDATE_DEDUP_IOU,
                max_objects=SAM3_CANDIDATE_MAX_OBJECTS,
                confidence_threshold=SAM3_CONFIDENCE_THRESHOLD,
                min_score=SAM3_MIN_SCORE,
            )
        except Exception as exc:
            sam3_slot["error"] = exc
        finally:
            sam3_slot["duration"] = time.time() - t_sam3
            sam3_slot["finished"] = True

    sam3_thread = threading.Thread(target=_run_sam3_worker, name="sam3_subprocess", daemon=True)
    sam3_thread.start()

    _prompt_user("")  # Enter to stop recording.
    sub.stop_recording()
    n_static = sub.num_recorded_frames()
    print(f"[live] recording stopped after {n_static} keyframes", flush=True)

    # Pause Gazebo physics from here through the end of the init-PLY
    # build. The window covers SAM3-finish, SAM3D subprocess, and the
    # depth-back-projection PLY assembly — all of which compete with
    # gzserver for CPU/GPU. The atexit guard above unpauses if we die
    # in this stretch; the try/finally below covers normal raises.
    pause_gazebo_physics(sub)
    try:
        if not sam3_slot["finished"]:
            print("[live] waiting for SAM3 to finish...", flush=True)
            sam3_thread.join()
        if sam3_slot["error"] is not None:
            raise sam3_slot["error"]
        sam3_objects = sam3_slot["objects"] or []
        if not sam3_objects:
            raise RuntimeError("SAM3 found 0 objects — adjust the prompt and retry")
        print(f"[live] SAM3: found {len(sam3_objects)} graspable masks "
              f"({sam3_slot['duration']:.1f}s)", flush=True)
        # Persist the measured SAM3 duration to a sidecar JSON. The
        # pipeline reads this in ``_run_sam3_and_sam3d_generation`` and
        # re-injects it into ``self._timing["S0.1_sam3_segmentation"]``
        # so the timing report shows the real subprocess wall-clock,
        # not the pipeline's near-zero cached-check.
        live_timings_path = artifact_dir / "live_sam3_timings.json"
        live_timings_path.write_text(json.dumps({
            "S0.1_sam3_segmentation": float(sam3_slot["duration"]),
        }, indent=2) + "\n")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[live] running SAM3D (background, ~100s for ~4 objects)", flush=True)
        sam3d_thread, sam3d_slot = _spawn_sam3d_in_thread(
            anchor_rgb_path=anchor_rgb_path,
            sam3_objects=sam3_objects,
            artifact_dir=artifact_dir,
            debug_dir=debug_dir,
            depth_path=depth_path,
            intrinsics_path=intrinsics_path,
        )

        # Static training cannot share the GPU with SAM3D, so we block here.
        last_log = time.time()
        while not sam3d_slot["finished"]:
            sam3d_thread.join(timeout=5.0)
            if not sam3d_slot["finished"] and (time.time() - last_log) >= 5.0:
                print("[live] still waiting for SAM3D...", flush=True)
                last_log = time.time()
        if sam3d_slot["error"] is not None:
            raise sam3d_slot["error"]

        sam3d_results = sam3d_slot["results"] or [{} for _ in sam3_objects]
        n_ok = sum(1 for r in sam3d_results if r)
        for i, r in enumerate(sam3d_results):
            print(f"[live] SAM3D obj {i}: {'ok' if r else 'failed'}", flush=True)
        print(f"[live] SAM3D done: {n_ok}/{len(sam3_objects)} objects ready", flush=True)

        print(f"[live] building init PLY from {n_static} static views...", flush=True)
        sub.build_static_init_pointcloud()
        # Refine the just-written naive concat seed with ICP + TSDF fusion.
        # Same output path (transforms.json's ply_file_path), same world
        # frame -- Splatfacto's load_3D_points=True picks it up unchanged
        # and gets a denoised, real-RGB, adaptive-density cloud instead of
        # the ~N-times-overlaid naive back-projection.
        try:
            from .rgbd_fusion_init import build_tsdf_seed
            print("[live] refining init PLY via ICP + TSDF fusion...", flush=True)
            build_tsdf_seed(LIVE_ROOT, force=True, verbose=True)
        except Exception as exc:
            print(f"[live] WARNING: ICP+TSDF refinement failed ({exc}); "
                  f"falling back to the naive seed", flush=True)
    finally:
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
