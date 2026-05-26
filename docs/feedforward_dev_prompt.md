# Cloud Code Prompt — InstaInpaint feasibility study for dynamic-gs

## Your task

**Research + design-doc only. Do not modify the runtime pipeline.**

Investigate whether the **InstaInpaint** feed-forward 3DGS inpainting model
(Wang et al. 2025) can replace the per-frame iterative optimization loop in
the dynamic-gs pipeline's dynamic phase.

**Concrete usage:** every time the change-mask CDN flags pixels (i.e. the
object moved and uncovered scene regions that weren't previously visible),
**call InstaInpaint to fill / replace the Gaussians in that region** instead
of running the 50-step photometric optimization loop. CDN is the inpaint
mask; the existing scene Gaussians in that region are candidates for
deletion / replacement by the feed-forward model's output. Object Gaussians
themselves are *never* touched by InstaInpaint — they continue to move
rigidly under XFeat tracking.

Output a single design document
(`docs/feedforward_dev_design.md`) covering the questions below. A small
read-only probe script that loads a few frames from the recorded dataset
to sanity-check assumptions is acceptable, but do **not** wire InstaInpaint
into `dynamic_gs_pipeline.py` / `dynamic_gs_model.py` — that's a follow-up.

## Background: what dynamic-gs does today

This repo is a two-phase Gaussian-Splatting pipeline built on top of
Nerfstudio. See [CLAUDE.md](../CLAUDE.md) for the full architecture; the
short version:

- **Phase 1 (static):** 4000 steps of Splatfacto-style optimization on a
  static SfM scene. No densification (`NoRefineStrategy`).
- **Phase 0b (boundary):** A SAM3D-generated Gaussian cloud for the moved
  object is fused into the scene; FoundationPose trackers are pre-built.
- **Phase 2 (dynamic):** Two decoupled loops sharing one Gaussian scene:
  1. **Tracker loop** (every `tracker_tick_every_steps=3` steps) — runs
     the **XFeat + LighterGlue + Kabsch-RANSAC** tracker on a new frame,
     applies a rigid 6-DoF transform to flagged object Gaussians via
     `apply_rigid_object_transform_from_reference(R, t)`. The XFeat tracker
     is the *only* supported tracker after the 2026-05-26 purge — see
     `dynamic_gs/utils/xfeat_motion.py` and `tracker_common.py`.
  2. **Optim loop** — for each frame the keyframe filter accepts AND whose
     change-mask CDN clears `optim_pool_min_change_pixels=500`, push
     `(camera, cdn)` onto an `OptimPool` (capacity 15, FIFO). Optim picks
     pool entries round-robin; each step runs one masked photometric +
     depth + rigid-reg loss step on the scene-Gaussian subset whose 2D
     footprint overlaps CDN AND that is not flagged as object. Evict on
     `epochs_used >= 50` OR `last_loss < initial_loss × 0.3`. **This is
     the loop InstaInpaint is being evaluated to replace** — the
     scene-Gaussian "fill in what the object uncovered" job.

The change mask CDN (computed in `_compute_change_mask`) is a per-pixel
MS-SSIM dissimilarity between the rendered scene (with object at its
current pose) and the live image, with the gripper and projected object
mask excluded. **CDN is precisely the "regions that need to be inpainted"
signal that InstaInpaint takes as input** — so the integration shape is
not coincidental.

The tracker was developed and tuned exclusively on **live data at ~10 Hz
wall-clock** (gazebo sim publishing at 5 Hz nominal, achievable rate
limited by tracker cost). It has **not been tested on recorded data**.

## What InstaInpaint provides

- Paper: <https://arxiv.org/pdf/2506.10980>
- Project page: <https://dhmbb2.github.io/InstaInpaint_page/>
- Code: <https://github.com/dhmbb2/InstaInpaint>

A **feed-forward** model that takes a posed Gaussian scene + a mask of
"region to inpaint" + (a small set of?) reference views, and produces
new Gaussians filling the hole in **a single forward pass** — no
per-scene optimization. Replaces what is currently a ~50-step iterative
photometric optimization in our pipeline.

The exact inputs, outputs, and assumptions are what you need to map out
in the design doc.

## Recorded dataset for evaluation

`/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/dynamic_gs_test_2026-03-28_19-49-45_w_background/`

Layout:
```
static_scene/    Nerfstudio transforms.json + rgb/ + depth/ + SfM points
dynamic_scene/   transforms.json + rgb/ + depth/ + masks/  (61 frames @ 5 Hz)
```

61 dynamic frames = ~12 s of teleop capture with a robot arm + manipulated
object on a tabletop. Depth is uint16 PNG with scale `1e-3` m/unit.
Background was kept (not masked out) — the per-frame `masks/` are
gripper-exclusion only.

Already has `phase0_*.log`, `timing_report*.txt`, `foundation_pose_debug.mp4`,
`pipeline_presentation.png` from prior runs — useful priors on what the
scene looks like and how the existing pipeline behaves on it.

## The pacing concern (important)

The XFeat tracker is fast enough that on **recorded** data (no wall-clock
gating from the camera) it will rip through all 61 frames in ~3-4 seconds
of GPU time, moving the object far in a single optim-loop iteration. The
optim pool was designed assuming the tracker is bounded by camera arrival
rate (~5-10 Hz). If feed-forward inpainting is also fast, the system
becomes a tight loop with no "settling" time — possibly fine, possibly
catastrophic for quality.

**Mitigation:** Throttle the recorded-frame feeder to **~5 Hz wall-clock**
(matching the live capture rate the tracker was tuned for) by injecting a
`time.sleep` or equivalent in the dynamic-frame iterator. This is the
required fallback. **Do not over-engineer** an iterative tick-by-tick mimic
of live mode — the goal is just to test whether the feed-forward model
can replace the optimization step on regions revealed by object motion.

Document the throttling design in the doc — where the sleep goes, what
loop it gates, how to disable it for benchmarking.

## Deliverable: `docs/feedforward_dev_design.md`

Answer these in order. Be concrete — cite file paths, function names,
config fields, line numbers.

### 1. InstaInpaint capability inventory
- Read the paper + repo README + key source files. Summarize in <300 words:
  what inputs it requires (Gaussian format? camera count? mask format?
  metric vs normalized scene?), what it outputs (replacement Gaussians?
  new Gaussians to append?), what model weights it needs, what GPU /
  VRAM budget, expected per-call latency.
- Identify any **hard incompatibilities** with our scene format
  (Splatfacto Gaussian param layout: `means`, `features_dc`,
  `features_rest`, `opacities`, `scales`, `quats`; SH degree; world-frame
  metric coords with `auto_scale_poses=False`).

### 2. Integration shape
- Where would InstaInpaint slot in? Map it onto the current optim-pool
  code path (`OptimPool`, `_prepare_frame_n`, `_compute_change_mask` in
  `dynamic_gs/dynamic_gs_pipeline.py`).
- What replaces the 50-step per-frame loss optimization?
- How does CDN map to InstaInpaint's mask input? (Pixel-space mask vs.
  3D Gaussian flagging — do we need to back-project CDN to identify
  which Gaussians to delete/replace, or does InstaInpaint take a 2D mask
  + camera and figure it out?)
- Does InstaInpaint *delete* existing Gaussians in the change region
  before filling, or *add* new ones alongside? How do we reconcile with
  the existing `scene_opt_active_mask` gating logic?
- Reference-view question: if InstaInpaint needs N reference views, what
  N, and where do they come from? (Static phase rendered views? Recent
  dynamic frames from the optim pool? Both?)

### 3. Recorded-data test plan
- Step-by-step plan for the minimal end-to-end test (which the user, not
  you, will run after reviewing your doc):
  1. Load dataset, run static phase to completion.
  2. Run dynamic phase with the **throttled** feeder.
  3. For the first M frames where CDN > threshold, call InstaInpaint
     instead of the optim loop.
  4. Render comparison views (with optimization vs. with InstaInpaint vs.
     ground-truth live frame) and dump to `docs/feedforward_dev_results/`.
- What metrics to look at: PSNR on the CDN region, visual sanity of
  the rendered comparison panel, Gaussian count delta per frame, total
  wall-clock for dynamic phase end-to-end.

### 4. Throttling implementation
- Exactly which function/loop to gate, what target rate, how to make it
  configurable (env var or pipeline config flag). The throttle should
  affect *recorded-data runs only* — live mode must remain unchanged.
- Identify the current dynamic-frame iterator in
  `dynamic_gs_pipeline.py` / `dynamic_gs_datamanager.py` (search for
  `_dynamic_frame_for_step` and `tracker_tick_every_steps`).

### 5. Risks + open questions
- 3 sentences max each. What might kill the experiment outright? What's
  the most likely "works but quality regression" failure mode? What
  assumptions about InstaInpaint do you have that the doc-readers (the
  user) should verify before committing to implementation?

### 6. Recommended next step
- Either "proceed to implementation, here's the file-level plan", or
  "stop, here's why this won't work". Pick one and defend it in <200 words.

## Constraints

- **No edits** to `dynamic_gs_pipeline.py`, `dynamic_gs_model.py`,
  `dynamic_gs_datamanager.py`, or any file under `dynamic_gs/utils/`
  (except creating new files for a read-only probe script if useful).
- **No new commits to runtime code**. You may commit `docs/` additions.
- Do not clone InstaInpaint into `third_party/` yet — just read the
  GitHub repo. Cloning is a deliberate decision for the next phase.
- Read CLAUDE.md fully before starting. The notes under "Dynamic-GS
  Cleanup" (sections dated 2026-05-26) document hard-won XFeat tracker
  config sweet spots — do not propose changes that contradict them
  without explicit justification.
- If you discover that the recorded dataset is missing a file the existing
  pipeline needs, stop and surface that as a blocker in the design doc
  rather than fabricating it.

## What "done" looks like

A single `docs/feedforward_dev_design.md` (~1500-2500 words) on the
`feedforward_dev` branch, committed with a clean message. Plus optionally
a `docs/feedforward_dev_probe.py` read-only script if it informed your
analysis. No runtime code touched. User reviews the doc and decides
whether to greenlight the implementation phase.
