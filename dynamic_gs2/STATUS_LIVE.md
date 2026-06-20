# dynamic_gs2 — LIVE pipeline STATUS + next steps (2026-06-20)

Single-source-of-truth after the live-bring-up session. **Read this first** before touching the
static/live path again. The repo has many audit/refactor md files now — this is the only one that
reflects the CURRENT live reality.

## What works NOW (validated live on Gazebo, 2026-06-20)
`dynamic_gs2/live.sh <data_dir> [prompt]` = the WHOLE pipeline in ONE command, ONE process:
live sweep (red-box UI) → FastSAM segment → SAM3D → deferred GPU-TSDF seed → native fp32 train →
native Phase-0b fuse → **re-phase scene static→dynamic IN PLACE** (`SceneModel.set_phase` +
`GaussianSet.enable_freelist` + count-provider) → **hand the warm AnySplat+XFeat straight into the
live tracking loop** (no reload, no ~17s re-spawn) → live track + FF. Ran end-to-end:
`FULL-LIVE: re-phased ... LIVE: d0_instance_id=1, scene=900266`. AnySplat decode confirmed live
(insert-scale-cap message fired). GPU peak ~11GB (<16.3 w/ Gazebo). Entry: `pipeline.run_full_live`
(`--mode full`). Fused `static_state.pt` saved at `datasets/2026-06-20_224240/static_scene/`.

## THE REMAINING BUG (next session #1 priority) — object inserted in WRONG PLACE
Operator: "the object was fused well but the INSERTION was wrong — it's literally not where it
should be, its means moved." **Capture the POSE of the frame segmentation/SAM3D ran on, and do NOT
re-place the object against a LATER/different pose.**

### VERDICT 2026-06-20: GEOMETRY IS CORRECT — it was a VIEWER bug. FIXED.
Three independent measurements on the saved `2026-06-20_224240/static_state.pt` all agree the object
is placed right. The most decisive: **`_diag_render_anchor.py`** renders the fused scene from the
anchor camera and overlays it on the anchor RGB → the rendered object's mask centroid is **1.4 px**
off the FastSAM truth mask (0.07% of a 1920px frame); `overlay.png`'s green outline HUGS the real
screwdriver. End-to-end placement is correct.

**ROOT CAUSE (found): the LIVE viser never set its initial camera.** `run_live` created the
`ViserBridge` but (unlike `run_view_recorded`) never called `set_initial_camera`, and "Follow tracked
frame" defaulted OFF. So the viewer opened at viser's ARBITRARY default pose (origin/+z), while the
metric scene lives at world `~[-0.4,0.2,0.5]` in the robot-base frame — the correctly-placed object
rendered off to the side / tiny / off-screen relative to the camera-feed thumbnail = "in the wrong
place." Nothing wrong with the means.

**FIX (applied 2026-06-20):**
- `pipeline.run_live`: on the first live frame, `bridge.set_initial_camera(fr.c2w_4x4)` — the viewer
  opens aligned to the live camera.
- `ViserBridge(follow_default=True)` for the live path: "Follow tracked frame" defaults ON, so the
  rendered view tracks the live camera and the rendered object overlays the camera-feed object.
  Operator unchecks it to free-orbit. (`dynamic_viz.py` got a `follow_default` ctor arg.)
- `DynamicLoop.step` now also dumps the SEED tick under `DGS_TRACK_CMP=1` (tick `000000_*`), so the
  decisive no-motion frame is captured for verification.

**Re-confirm on the next live run:** the object should now render on the real object immediately.
If still off, set `DGS_TRACK_CMP=1` and overlay `_track_debug/000000_1_live.png` vs `000000_2_rend.png`
— they should match (rules in geometry vs the secondary live-world-frame-drift hypothesis below).

### DATA from 2026-06-20 (both original prime suspects RULED OUT — measured, not theorized)
Ran two diagnostics on the saved `2026-06-20_224240/static_scene/static_state.pt`
(`dynamic_gs2/_diag_insert_pose.py`, `dynamic_gs2/_diag_dynamic_handoff.py`):
- **Static fuse places the object CORRECTLY.** Fused object (instance_id=1, 24114 gaussians)
  centroid `[-0.436, 0.186, 0.537]`; back-projected anchor-mask target centroid `[-0.426, 0.189,
  0.543]` → **|A−B| = 12.3 mm** (expected: the fused object includes the SAM3D-reconstructed
  occluded BACK that the anchor depth can't see). Size ratio 1.565× = same reason (complete object
  vs front-only back-projection). NOT a gross displacement, NOT undersized. ✓
- **Dynamic handoff does NOT move the object.** Warm-loaded the .pt into the dynamic scene, seeded
  the tracker on the anchor frame, ran the first track on the SAME no-motion frame:
  `success=True, inliers=68, |t|=0.00 mm, rot=0.01°`; object centroid drift after SEED and after
  TRACK = **0.00 mm**. The tracker returns clean identity; `ReferenceObjectPose.capture/apply`
  keep the object exactly where the fuse put it. ✓
- No recenter / rescale / orientation / camera-opt offset anywhere in dynamic_gs2 load or render
  (grepped): rendered scene + live camera poses share ONE metric world frame.

### So the displacement is NOT on the static→fuse→handoff path. Remaining live-only hypotheses:
1. **Visual, not geometric.** The static red-box UI shows the LIVE CAMERA FEED, never the rendered
   fused scene — so the operator never actually saw the fused object during static. The first sight
   of the rendered object is the dynamic viser. If the viser camera being orbited isn't aligned with
   the live feed, the object can LOOK misplaced vs the camera-feed thumbnail while being correct in
   world. CHECK by capturing rendered-vs-live pairs (below).
2. **Live world-frame drift between static sweep and dynamic loop.** Both use the same ROS publisher
   FK in one session, so they SHOULD share an origin — but if the sim/world was reset, or the FK
   re-zeroed, between the sweep and go-live, the fused object (anchor-frame coords) would render
   offset from the live camera. (Static-pose-plugin reset note in CLAUDE.md is adjacent evidence
   this CAN happen.) Only a live run can confirm.

### Next diagnostic (needs a live run — the saved artifacts are exhausted)
Run `live.sh` (or `resume_live.sh` on the saved .pt) with **`DGS_TRACK_CMP=1`** set — `DynamicLoop`
already dumps per-tick `NNNNNN_1_live.png` (live feed) + `NNNNNN_2_rend.png` (rendered scene) to
`dynamic_scene/_track_debug/`. Overlay tick 0's pair: if the rendered object sits on the real object
→ it's the viser-camera-alignment visual artifact (#1, fix the viewer, not the geometry). If the
rendered object is genuinely offset in world from the live object → live world-frame drift (#2),
check the publisher FK/world reset between sweep and go-live.

Diagnostics kept at `dynamic_gs2/_diag_insert_pose.py` + `dynamic_gs2/_diag_dynamic_handoff.py`
(re-runnable on any saved static_state.pt).

## Other operator feedback from the live run (next steps, priority order)
2. **Live feed is SLOW (~2-4 fps), laggy.** The red-box UI `_feed_loop` (static_pipeline.py) pushes
   full-res `set_background_image` per frame via JPEG over websocket — too heavy at 1200p. Fix:
   downscale the feed (e.g. to ~640px) + lower JPEG quality + cap push rate. Same likely applies to
   the dynamic ViserBridge feed. Profile the encode+push cost.
3. **Step-list UI panel** (operator wants this): a small viser GUI list of the pipeline steps —
   `initial capture, segmentation, object generation, scene training, fusion, realtime` — each
   GREY (not started) / YELLOW (ongoing) / GREEN (done, with elapsed time). Replace the single
   `set_status` text banner (static_pipeline `_RedBoxUI.set_status`) with this colored checklist.
   The "???" the operator saw = the status banner wasn't granular; this checklist fixes it.
4. **SIGTERM doesn't save** `post_dynamic_state.pt` + FF report (only KeyboardInterrupt does). Add a
   SIGTERM handler in `run_live`/`run_full_live` that runs the same finally block. (Minor.)

## What's GOOD / settled (don't re-litigate)
- **AnySplat works very well** (operator). NDP works perfectly (operator) — DO NOT touch NDP /
  `register_and_fuse_sam3d_object` / `base_scale`. The earlier under-scale was the `aa_lead` wrong
  anchor frame (fixed) + the last-frame clipping the screwdriver tip (capture limitation, accepted).
- **fp16 BROKE the static fit** (17 vs 26 dB → blurry → starved tracker). static_fuse uses fp32. Do
  NOT re-enable fp16 for the static train. (Also reverted the old static-gs mixed_precision=True.)
- **FastSAM promote-to-container fix** (dynamic_gs/utils/fastsam_segmentation.py): keeps the FULL
  object mask (shaft+handle), not the high-CLIP sub-part. Default on. Keep.
- **Deferred GPU-TSDF seed**: sweep RECORDS only; seed built once after SAM3D via
  `fuse_recorded_dataset` (GPU-TSDF + SAM3D can't coexist = OOM). Keep.
- Re-phase-in-place validated bit-identical to `build_loaded_scene(phase='dynamic')`.

## 7 live-path bugs fixed THIS session (all were never-run-before code)
intrinsics property call · live GPU-TSDF OOM (deferred seed) · NaN-safe depth cast · Phase-0b
missing `initialization_debug/` dir (mkdir) · frozen UI (background feed thread + status) ·
dVRK core-pin + gsplat JIT env on dynamic_gs2 scripts · the `--mode full` warm handoff itself.

## Which md files to read for context (ignore the rest)
- THIS file (STATUS_LIVE.md) — current live reality + next steps.
- `dynamic_gs2/STATUS.md` — the original rewrite status (architecture: WRAP, GaussianSet SSOT,
  scene_model, free-list/uid invariant).
- `rewrite_spec/static_phase.md` — the static-phase schedule + the §2a GPU-TSDF/SAM3D residency
  constraints (why the seed is deferred).
- Memory: `project_dgs2_native_phase0b.md` (native Phase-0b + fp16-breaks-static-fit),
  `project_dgs2_static_phase_built.md` (static-phase module map),
  `project_dgs2_pinned_prefix_invariant.md` (uid-keyed tracked-object correspondence — RELEVANT to
  the insertion-pose bug: the object is matched by gauss_uid + ReferenceObjectPose, key that math).
- IGNORE the `scripts/code_audit/` purge-prep package + the many one-off audit md's for this bug —
  they predate the live work.

## Key files for the insertion-pose bug (next session)
- `dynamic_gs2/dynamic_track.py` — `ReferenceObjectPose` (R,t from d0 ref → object subset) + XFeatTracker
- `dynamic_gs2/pipeline.py` — `DynamicLoop.step` (how the tracker transform is applied to the object
  per tick), `run_full_live` (the handoff), `run_live` (d0 pick + ref-pose build at first frame)
- `dynamic_gs2/static_phase0b.py` — `load_anchor_for_fusion` + the fuse (where the object world pose
  is set) — confirm its world frame == the dynamic loop's frame-0 camera frame.
- `dynamic_gs2/static_segment.py` — `snapshot_anchor` (writes anchor pose.json)
