# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" â "Write tests for invalid inputs, then make them pass"
- "Fix the bug" â "Write a test that reproduces it, then make it pass"
- "Refactor X" â "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] â verify: [check]
2. [Step] â verify: [check]
3. [Step] â verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

<!-- ============================================================ -->
<!-- BEGIN: CLEANUP NOTES (working section, prepend new items here) -->
<!-- ============================================================ -->

# Dynamic-GS Cleanup

Working section for the in-progress cleanup of the dynamic-gs pipeline. All new cleanup-related notes go here, above the separator below. Older project documentation stays untouched after the separator.

## Keeping this file accurate (MANDATORY)

This file is loaded as authoritative instructions every session — a stale claim actively misleads. Two rules:

1. **If you change code that this file references, update this file in the SAME change.** That includes: config defaults / field values (`xfeat_top_k`, `STATIC_NUM_STEPS`, `segmentation_backend`, …), the lines that enforce a Design Invariant, default-flag states, and any symbol/file/module name mentioned here. The diff is not done until the doc matches the code.
2. **Reference code by symbol name, NOT line number.** Write `` [`dynamic_gs_config.py`](dynamic_gs/dynamic_gs_config.py) (`_ZERO_LR_OPTIMIZERS`) `` — never `:138` / `#L138`. Line numbers drift on any unrelated edit above them; symbol names only break when the symbol is actually renamed (and then rule 1 applies). The exception is the vendored-nerfstudio reference trace far below, which points into a pinned dependency.

Dated session notes (`### … (YYYY-MM-DD)`) are historical records of *measurements taken at that date* — do NOT rewrite their numbers (that fabricates). If a dated note's conclusion was later reverted, prepend a `> **SUPERSEDED (date):**` banner stating current reality instead of editing the body.

## Goal

Do a whole cleanup of the dynamic-gs pipeline (model, pipeline, datamanager, config, utils). Open scope — entries below will refine what "cleanup" covers.

## Design Invariants (NON-NEGOTIABLE — DO NOT VIOLATE)

These are hard rules the pipeline depends on. If a change appears to require breaking one of them, **stop and flag it explicitly** — do not silently violate. Each rule has a stated reason so you can judge edge cases.

1. **Static phase: `means` LR = 0.** Gaussian positions stay locked on the TSDF-fused seed (`depth_camera_init_points.ply`). Only `features_dc`, `features_rest`, `opacities`, `scales`, `quats` train. **Why:** the TSDF seed is geometrically correct (ICP-fused depth); allowing means to drift under photometric loss in 500 steps (`STATIC_NUM_STEPS=500`) produces visibly blurry / smeared output. Verified by: the `means` optimizer in [`dynamic_gs_config.py`](dynamic_gs/dynamic_gs_config.py) (`lr=0.0`). *History: was `1.6e-4` until 2026-06-02; the docstring claimed "effectively 0 because densification is off" but Adam moves means via `.grad` regardless of densification — that claim was wrong. Resolved by setting LR to 0.0 explicitly.*

2. **Static phase: `camera_optimizer.mode = "off"`.** Camera poses are NOT optimized during static training. **Why:** `transforms.json` contains ICP-refined poses (see invariant #3), so the residual error is sub-mm — there is nothing for camera-opt to fix. Leaving it on at LR=1e-3 over 500 steps drifts cameras by visible amounts (degrees / cm), smearing the scene. Verified by: the static-gs `camera_optimizer` in [`dynamic_gs_config.py`](dynamic_gs/dynamic_gs_config.py) (`mode="off"`). *History: was `"SO3xR3"` until 2026-06-02; user explicitly observed cameras moving "by a LOT" in the viewer during run #2 and reported the prior fixed-means + fixed-poses runs converged "insanely good" in 500 epochs.*

3. **`<data>/static_scene/transforms.json` contains ICP-refined poses, not raw URDF FK.** The raw URDF capture is preserved at `<data>/static_scene/transforms_urdf_backup.json`. **Why:** the TSDF seed PLY lives in the ICP-refined frame; if training cameras stay in the raw FK frame, there's a systematic 1–4 mm misalignment across the trajectory that camera-opt would have to undo (but per invariant #2, camera-opt is off). Tool: [`scripts/rewrite_transforms_with_icp.py`](rewrite_transforms_with_icp.py) (back up + ICP rewrite + new top-level `pose_source: "icp_refined_from_urdf_v1"` flag in transforms.json). Idempotent — re-runs detect the existing backup and refuse to overwrite it. **Drift on new_env (measured):** median 0.96 mm / 0.053°, max 3.94 mm / 0.41° over 68 frames. *Future: integrate this write-back directly into `ConcurrentFusionRunner.stop_and_finalize` so capture itself produces ICP-refined transforms.json (no post-pass needed).*

4. **Dynamic phase: ALL gauss-param LRs = 0.** Only the XFeat tracker's rigid transform + feedforward decode insertions mutate the scene during dynamic phase. **Why:** the dynamic phase is a runtime, not a training loop. Per-step gradient descent on Gaussian params would fight against the tracker. Enforced by [`_ZERO_LR_OPTIMIZERS`](dynamic_gs/dynamic_gs_config.py) used by both `dynamic-gs` and `dynamic-gs-live`.

5. **`outputs/` is suppressed across all runs.** Nerfstudio's default `outputs/<exp>/<method>/<timestamp>/` directory tree is not used. All artifacts live under `<data_dir>/`. **Why:** dataset dir is self-contained, portable, and survives output-tree cleanups. Enforced by three monkeypatches in [`dynamic_gs/__init__.py`](dynamic_gs/__init__.py) targeting `ExperimentConfig.save_config`, `Trainer.train`'s `dataparser_transforms.json` write, and `writer.setup_event_writer`'s tensorboard branch. Note: when `--vis viewer` is enabled, Nerfstudio's `ViewerState.__init__` still calls `mkdir(exist_ok=True)` on `outputs/<run>` — pre-create the parent before launching.

6. **Background color = Gazebo sky `(0.86, 0.92, 1.0)`.** Used by both `StaticGSModel` and `DynamicGSModel` (and the viser viewer default). **Why:** the simulator renders against this sky; training/inference against any other background introduces a constant photometric bias the renderer would have to compensate for via opacity tweaks at silhouettes.

7. **Persistent SAM3 + Fast-SAM3D worker is the canonical path for SAM3 / SAM3D during live capture.** [`SamWorkerClient`](dynamic_gs/utils/sam_worker.py) (spawn-once, load-on-demand, JSON-over-pipe). **Why:** measured 9.4 s/call savings on SAM3 cold-start + 22 s/call on SAM3D (when reused). The legacy per-call subprocess paths in [`sam3_segmentation.run_sam3_subprocess`](dynamic_gs/utils/sam3_segmentation.py) / [`sam3d.run_sam3d_multi_object_subprocess`](dynamic_gs/utils/sam3d.py) are still fallbacks but the live flow auto-spawns the worker at `fusion_runner.start()` time. Future: also wire into `fusion/phase0.py` (currently still uses the legacy per-call path for recorded datasets).

8. **Per-object identity buffers are owned by specific pipeline phases:**
   - `object_instance_ids` — written by Phase 0b fusion only.
   - `inserted_flags` — written by Phase 0b (SAM3D inserts) and by `rgbd_decode.insert_inpaint_gaussians` (FF Mode B inserts).
   - `sam3d_init_target_flags` — written by Phase 0b only.
   - `object_flags` — written by the dynamic-gs pipeline's D0 selection on the first dynamic frame, **never by the static pipeline**. `object_flags=0` in `post_fusion_state.pt` is the correct/expected state.

9. **Live visualization uses viser-direct, NEVER Nerfstudio's viewer.** Always connect to **`http://localhost:8081`** (the viser-direct port), never `:7007` (the NS viewer). Do NOT pass `--vis viewer` on the CLI. **Why:** the NS viewer does server-side gsplat rasterization in a render-state-machine thread:
   - **Speed:** every camera move triggers a server render. Even with the model lock plumbed in, the NS viewer is ~10× slower than viser-direct's client-side WebGL splatting (~3 fps server-side vs 15–25 fps client-side at 1920×1080 on this scene).
   - **Concurrency hazards:** the NS render thread reads `gauss_params` while the FF bg thread is mid-`insert_inpaint_gaussians`. Even with the shared `_model_lock` + `attach_render_lock` hook plumbed into `DynamicGSModel.get_outputs_for_camera`, a post-training rasterization race can deadlock the render thread (observed 2026-06-02). Viser-direct sidesteps the entire surface — it pushes splat handles to the browser and the browser does the rasterization; the server never calls `get_outputs_for_camera` for visualization.
   - **How enforced:** every method config in [`dynamic_gs_config.py`](dynamic_gs/dynamic_gs_config.py) sets `vis="tensorboard"` (NS viewer OFF; the tensorboard writer is suppressed by `_suppress_nerfstudio_output_writes` in [`dynamic_gs/__init__.py`](dynamic_gs/__init__.py)). `DynamicGSPipelineBaseConfig.enable_viser_direct: bool = True` is the default in [`dynamic_gs_pipeline_base.py`](dynamic_gs/dynamic_gs_pipeline_base.py), so viser-direct spins up automatically on port 8081 (configurable via `viser_direct_port`).
   - **History:** commits [`703cb9b feat(viser-direct): Path A hybrid visualization for live tracking`](https://github.com), [`92b11a5 feat(viser-direct): visualize FF inserts via per-call splat handles`](https://github.com). The Path-A handle-based path is the canonical one; a server-side push-image fallback exists in [`viser_pushimg_baseline`](memory/project_viser_pushimg_baseline.md) for environments where client-side WebGL can't run, but it's not the default.

## Notes

### AnySplat reprojection was square-only — broke at 1920×1200 (2026-06-13)

**Symptom:** at 1200p the FF produced a "ghost" second copy of static objects offset sideways (e.g. a second droid next to the real one), comet-tail smears, the gripper appearing in the scene, and runaway insert accumulation (scene ballooned 459k→1.6M gaussians as misplaced inserts re-triggered CDN every tick). At 800×800 the SAME code worked perfectly.

**Root cause:** AnySplat's [`process_image`](third_party/AnySplat/src/utils/image.py) does **aspect-preserving resize (shorter side → 448) + CENTER-CROP to 448×448** — for 1920×1200 it resizes to 716×448 then crops 134 px off each side (AnySplat only sees the centre horizontal slice). But [`reproject_anysplat_to_scene`](dynamic_gs/utils/anysplat_decode.py) assumed a full-frame **anisotropic squash**: it scaled scene intrinsics by `W_any/scene_w` (= 448/1920) for x and resized sensor depth with a plain `cv2.resize(..., (448,448))`. Those two agree **only when the scene is square** (the 800×800 era: resize→448, no crop, `448/800` correct on both axes). At 1920×1200 the x-scale was wrong (used /1920 instead of /1200) **and** the centre-crop offset (−134 px) was missing → every insert mapped sideways = ghosts; the component (CDN/gripper) mask was indexed at the wrong pixels too → gripper leaked in.

**Fix:** `reproject_anysplat_to_scene` now **inverts the resize+center-crop**. It maps each AnySplat pred-crop pixel `(u,v)` back to the true scene pixel via `crop_scale = 448/min(W,H)`, `crop_left/top = (new−448)//2`, then samples **full-resolution** sensor depth at `(u_scene,v_scene)`, indexes the component mask at **scene** resolution, and back-projects through the **full** scene intrinsics. Algebraically identical to the old path at 800×800 (crop_scale=448/800, crop_left=0). Recorded-1200p result vs the broken run: per-call inserts 20k–66k → 200–1.8k, scene 1.6M → ~497k, ghost/comet-tail/gripper gone (frames 131/137). **No code regression caused this — the reproject was always square-only; the ZED-X move to 1920×1200 exposed the latent assumption.** The hardcoded `H_any,W_any = 448,448` in [`_anysplat_bg_run`](dynamic_gs/dynamic_gs_pipeline_base.py) is CORRECT (AnySplat does output a 448 crop) — the bug was the coordinate math around it, not the 448.

**Follow-up — adaptive crop (same session):** AnySplat's fixed center-crop only sees the centre slice, so change near the image edges was never fed to it. [`_anysplat_crop_windows`](dynamic_gs/dynamic_gs_pipeline_base.py) now picks the square crop to **ENCOMPASS the change mask** (`size = max(bbox_w,bbox_h) + 2·50 px` at the change's natural scale — NOT forced to 448; AnySplat up/down-samples internally) centred on the change, clamped to image bounds. When the change bbox is **wider than the image short side** (one square can't cover it) it returns **two** horizontally-tiled windows (capped at 2); otherwise one (don't run AnySplat twice when unneeded). `_anysplat_bg_run` decodes per window (ICP runs ONCE, before the loop), each filtered by the FULL change mask + union-deduped so overlaps don't double-insert; `reproject_anysplat_to_scene(scene_crop=(left,top,size))` maps the 448 pixels back through the chosen window (any size). Recorded-1200p: `windows=1` normally, `windows=2` fired on wide multi-component frames, scene plateaued ~663k (vs 497k center-crop — the extra is legit edge/wide-change coverage the center-crop skipped), no runaway.

**Follow-up — ordered `_ff_debug` dump + object-mask %-scale (same session):**
* The per-FF-call dump now writes a fixed numbered sequence so files sort/view in raw→clean pipeline order: `<stem>_1_gripper_mask`, `_2_object_mask`, `_3_real`, `_4_rendered` (PRE-cull render = what the RAW CDN scored against), `_5_rerendered_after_cull` (POST-cull render = what the CLEAN CDN scored against), `_6_raw_mask`, `_7_clean_mask`. Compare 4↔5 for the cull's effect and 6↔7 for the resulting mask change. **Caveat:** `_4_rendered` is captured at FF-dispatch (a few ticks AFTER the tick that produced `_6_raw_mask`, since the FF runs on a bg thread while the main loop keeps ticking + moving the tracked object), so 4↔6 is *approximate*, not a pixel-exact pair — to make it exact you'd stash the tick render in the `TrackerFrame` and thread it through (not done; the render delta is imperceptible by eye so it wasn't worth the plumbing). Diagnostic finding via this dump: the big raw→clean removal that's "neither object nor gripper" is the `_feedforward_cull_then_reclean_cdn` **re-render** (raw = tick CDN, clean = a fresh `_compute_tick_cdn` on the later/post-cull scene), NOT the object-footprint subtract (measured: object-subtract explained only ~122 of ~123k removed px on one frame).
* **Object-mask %-scale** (`feedforward_object_mask_scale: float = 1.02`): `_feedforward_clean_cdn` now enlarges the subtracted object footprint by +2% about its OWN centroid (`_scale_mask_about_centroid`, cv2 warpAffine) before the existing `feedforward_object_mask_dilate_px` dilation. Scales with object size (unlike fixed-px), so it swallows the thin rendered-vs-live **misplacement ring** that otherwise gets flagged as change → the FF would insert a flat copy of the object onto the tracked 3D object. 1.0 disables.

### Real-1200p end-to-end + tracker/CDN/insert overhaul (2026-06-12 → 13)

One day of fixes, each measured on the recorded screwdriver scene (800×800 `recording_15fps_2026-06-11_115107`, then the REAL 1920×1200 `replay_20260612_203321`). Commits `4402133..d1f6e7e`. Final validation on the real-1200p episode: **312/312 ticks, 0 tracking failures, peak 22.7 cm / 33°, end 20.6 cm (object stays on the plate), tail jiggle 0.7 mm / 0.2°/tick**.

* **Tracker** (see also the updated KF note below): per-tick object-mask match filter re-enabled (`xfeat_object_mask_filter`, stops background-pinning once the object is grasped); subsequent anchors unified to the D0 process (full-image keypoints POST-filtered — the old pre-masked re-extract corrupted descriptors; failures 87→22); scale-ratio anchor gate ([`SCALE_GATE_RATIO`](dynamic_gs/utils/xfeat_motion.py) 1.3× on cam↔centroid distance, `DGS_XFEAT_SCALE_SELECT=1` enables scale-aware selection); **static-hold** (`xfeat_static_hold*`: trend-gated median over 10 ticks, gates 12 mm/4°) kills stationary shake.
* **CDN**: MS-SSIM pyramid weights now COARSE-heavy `(0.15,0.30,0.55)` (`_rgb_msssim_score pyramid_weights`) — the full-res band read the soft-render-vs-sharp-live mismatch as change everywhere (static false inserts 14 939→934 gauss / 20 calls). `_apply_cleanup_recipe` cleanup-empty now returns EMPTY (the old raw-mask fallback fed noise specks to the FF every quiet tick → compounding insert loop, 47→849/call at 1200p).
* **FF inserts**: AnySplat voxel dedup OFF (both `feedforward_anysplat_voxel_dedup_m` and `_far_m` = 0 — code runs dedup if EITHER >0) + `feedforward_anysplat_scale_multiplier=2.0` (dense+2× = smooth; thinned+2× = blurry+CDN churn; dense+1× = gritty, tuned by eye). Open item: motion-phase insert volume at 1200p is still large (the validation run grew 459 k→1.29 M gauss; no OOM, but heavy).
* **Static-render blur was UNDERTRAINING, not seed density** (user confirmed the TSDF seed is correct → NoRefineStrategy/no-densification is right). The early-stop loss EMA flattens long before the render is sharp at 1200p: it fired at step 107 while render PSNR keeps climbing (same 459k seed: step107=24.0 dB, step500=26.3, step1000=27.3). Fixed by lowering `STATIC_EARLY_STOP_LOSS` 0.09→0.02 ([`static_gs_pipeline.py`](dynamic_gs/static_gs_pipeline.py)) so the scene trains ~the full 500-step budget. (`replay_20260612_203321` static_state.pt is the 1000-step build, 27.3 dB.) The min-view ~14 dB floor is a sweep coverage gap, not steps.
* **VRAM at 1200p (16 GB card)**: datamanager inner config now `cache_images="cpu"` (a ~300-frame 1200p episode is ~11 GB of rgb+depth+mask on GPU); [`fusion/phase0.py`](dynamic_gs/fusion/phase0.py) post-segmentation restore no longer force-moves caches to CUDA (was the measured 11.33 GB OOM).
* **FastSAM auto-threshold** ([`select_kept_indices`](dynamic_gs/utils/fastsam_segmentation.py)): replaced the hardcoded `min_score=0.2` with largest log-ratio-gap ("how many") + raw-cosine presence gate ("whether any"); masks are now **connected-component-split before CLIP scoring** (a single FastSAM mask spanning two objects — screwdriver+Android — was the instance-contamination root cause). `margin_min=0.04` (0.05 rejected the real screwdriver at margin 0.042 when the gripper blackout hides the handle — the capture flow always ends hovering over the object).
* **Real-1200p dataset** `replay_20260612_203321`: static = 57 keyframes (27 replay-sweep + 30 pose-deduped from the manual recording's pre-motion lead-in, renamed `aa_*` so the segmentation anchor stays the last SWEEP frame — frames sort by file_path and the anchor is the last frame); dynamic = manual teleop pickup, 643 frames recorded via `capture_only.py --no-fusion` (RTF ~1 with nothing else on GPU), trimmed to start at frame 330 (`transforms_full_backup.json` = untrimmed, `transforms_313_trimmed.json` = the active cut). Replay-harness gotchas: full-stack replay tanks RTF (GPU contention — capture with recorder only); the publisher needs `/dynaarm_arm/joint_states_full` (the merger needs the CONTROLLERS launch: load `controllers/joint_state_controller` + start via `switch_controller` if the spawner races params).

### Segmentation: FastSAM replaces SAM3 as default + SAM3D gaussian-only VRAM trim (2026-06-11)

Replaced SAM3 with **FastSAM + CLIP** as the default text-prompted segmenter (`segmentation_backend: Literal["sam3","fastsam"] = "fastsam"` on BOTH `StaticGSModelConfig` and `DynamicGSModelConfig`, kept in sync). Goal: shrink the segmentation footprint so SAM3D can load earlier/co-reside, cutting time-to-teleop-ready.

* **Why FastSAM:** measured resident **854 MiB** (peak 1930) vs SAM3's **3772** (peak 4522) — ~4.4× lighter; load **2.4 s** warm (vs SAM3 8.4 s); infer ~1 s. [`utils/fastsam_segmentation.py`](dynamic_gs/utils/fastsam_segmentation.py) `FastSamTextSegmenter`: FastSAM-x (class-agnostic masks) + CLIP `ViT-B-32-quickgelu`/openai scoring, keeping ALL candidates (not top-1) so SAM3's area/border/dedup/max_objects filters + the byte-identical output contract (`{mask_path,score,bbox,mask_area,object_index}` + raw NPZ) apply unchanged. Use the `-quickgelu` CLIP variant for OpenAI weights or match quality silently degrades.
* **Quality gate** ([`scripts/compare_sam3_fastsam.py`](scripts/compare_sam3_fastsam.py)): screwdriver on recording_15fps, gripper-blacked last frame → top-1 IoU **0.79** (near-identical bbox; FastSAM mask ~25 % looser, the whole IoU gap). PASS (≥0.75). Looser mask → SAM3D gets a bigger region → more fused gaussians (54 559 vs SAM3's 11 780 on this scene); tighten via `fastsam_conf` if it grabs background.
* **Worker:** FastSAM lives in the same `sam3_dynamic_gs` `SamWorkerClient` as SAM3/SAM3D (`load_fastsam`/`unload_fastsam`/`fastsam_infer`/`fastsam_infer_raw`). All `load_*`/`*_infer` responses now carry `gpu_resident_mb`/`gpu_peak_mb` (permanent instrumentation). Subprocess fallback + CLI: `run_fastsam_subprocess`. Phase0a + live_session branch on the backend; `sam3_reuse_cached` is now backend-aware (a SAM3 cache won't be served to a FastSAM run). Deps: `ultralytics` + `open-clip-torch` installed `--no-deps` into the sam3 env (torch untouched).
* **SAM3D gaussian-only trim** ([`apply_sam3d_gaussian_trim`](dynamic_gs/utils/sam3d.py), on by default; `DGS_SAM3D_NO_TRIM=1` disables): fp16 the two diffusion generators + both DINOv2 condition embedders (SAFE — forward already runs under `autocast(float16)`, so fp32 weights were cast per-op anyway) + move never-invoked modules (`slat_decoder_mesh`, `ss_encoder`, `slat_decoder_gs_4`) to CPU. Measured **resident 11698 → 7273 MiB (−4.4 GB)**, peak 12536 → 11707, gs 58944 → 58912, chamfer 4.66 mm on a ~1 m object (and it's NDP-registered onto real depth in Phase 0b anyway). 7.3 GB resident lets SAM3D LOAD during live capture alongside Gazebo (~2.6) + TSDF integrate (~3); 11.7 peak co-resides with splatfacto (~1.1).
* **Measured VRAM table (RTX 5070 Ti, 15842 MiB; supersedes the stale eyeballed comments in [`sam_worker.py`](dynamic_gs/utils/sam_worker.py)):** SAM3 3.8/4.5 · SAM3D 12.0/13.0 (trimmed 7.3/11.7) · FastSAM 0.85/1.9 · splatfacto step **1.1** (NOT 5–8). Tools: [`scripts/measure_vram.py`](scripts/measure_vram.py), [`scripts/sam3d_trim_probe.py`](scripts/sam3d_trim_probe.py).
* **Orchestration:** eager-AnySplat now also fires in recorded `static-gs` (`__init__`, gated `DGS_EAGER_ANYSPLAT`) → loads during training, adopted by dynamic-gs-live. **SAM3D‖splatfacto concurrent is deliberately NOT enabled** — trimmed peak 11.7 + training 2.5 + Gazebo 2.6 = 16.8 > 15.8 (OOM when sim is up). Safe realized parallelism: FastSAM‖TSDF-integrate during capture + AnySplat‖splatfacto. Also fixed: SAM3D subprocess now pins `CONDA_PREFIX` itself (was crashing when ns-train launched via bare env python).

### Dynamic-phase tracker freeze root cause: cudnn.benchmark autotune (2026-06-11)

The per-tick object freeze (~0.5 s hitches, multi-second worst case) was **`torch.backends.cudnn.benchmark = True`** — set globally by `nerfstudio/scripts/train.py:71` — combined with the XFeat crop (`_crop_for_xfeat`) presenting a **new conv input shape almost every tick**. Benchmark mode runs an exhaustive cudnn conv-algorithm autotune per new shape (100s of ms to seconds).

* **Fix:** `DynamicGSPipelineBase.__init__` unconditionally sets `torch.backends.cudnn.benchmark = False` for the dynamic phase. Static training (fixed 800×800 shapes) keeps nerfstudio's default. gsplat (custom CUDA) and LighterGlue (cublas matmul) are unaffected by the flag.
* **Measured (new_env, 192 frames, viser on + connected client, FF rgbd):** `DN.3c_xfeat_extract` avg 754.4 ms / max 5711.7 ms → **avg 14.3 ms / max 29.7 ms** (53×); `DN.3_estimate_total` 763.4 → 22.2 ms — back at the documented 17–30 Hz sweet spot.
* **Ruled out by measurement first** (don't re-suspect these): viser server-side rendering (viser-OFF run was *slower*, 685 ms avg) and GPU queue backlog (new `DN.3c0_gpu_queue_wait` sync-split key measured 0.1 ms avg — queue was empty; the extract call itself was slow).
* New permanent timing keys: `DN.3c0_gpu_queue_wait` (sync before extract — splits queue-wait from compute) and `DN.2_cdn_render` (per-tick CDN render wall; `DGS_DIAG_SYNC=1` makes it sync inside the timer for true GPU cost).

### Interactive object picker + preseg id-order fix (2026-06-11)

Operator now picks the tracked object in viser instead of the anchor heuristic, and can switch objects mid-run.

* **Picker** (`interactive_object_selection: bool = False` on `DynamicGSPipelineBaseConfig`): at D0 a viser GUI folder shows the SAM3 input image with every object's mask colored + numbered ([`utils/object_picker.py`](dynamic_gs/utils/object_picker.py) auto-detects preseg vs SAM3D artifacts), an **`add_dropdown`** of object ids (NOT `add_button_group` — button groups are momentary, their `.value` doesn't persist the click) and a **Done** button. The tick **blocks** (`_wait_for_selection`, 0.25 s Event slices) until Done or `object_selection_timeout_s` — deliberately, so the trainer's step counter doesn't race to `max_num_iterations` while the panel is open (a recorded run otherwise burns all 5000 steps in ~6 s and the panel vanishes). The viser render thread is independent, so the browser stays live during the block. A persistent **"Change object"** button (+ bare-Enter on stdin in live mode) sets `_reselect_requested`; the tick reopens the picker; old object freezes at its last pose. Selection funnels through the shared `_reseed_tracked_object(new_id, camera, batch)` on the base (reset surface + object_flags + `capture_reference_object_pose` + XFeat re-seed) — also used by both subclasses' `_bootstrap_d0`. Headless/timeout fallback: `d0_force_instance_id` (promoted to the base config) → per-subclass heuristic.
* **Invariant the picker relies on: SAM3 mask number i ⇔ `object_instance_ids == i+1`, in BOTH init paths.** SAM3D path already did this (`phase0.py` `instance_id = obj_idx + 1`). The preseg path **violated it** (the AMG coverage-merge compacted ids via `sorted(groups)` + positional `enumerate`, silently renumbering whenever a SAM3 mask had no AMG coverage — on new_env this merged the two androids into one id). Fixed in [`preseg_seed.py`](dynamic_gs/utils/preseg_seed.py): `_assign_and_merge` returns `(sam3_index, mask)` pairs and `_propagate` seeds SAM2-video `obj_id = sam3_index + 1`. Additionally a SAM3 mask with **no** AMG coverage now falls back to the raw SAM3 mask instead of being dropped (the AMG step is stochastic; a run lost 2 of 9 objects before this). **Datasets labeled before 2026-06-11 carry stale ids — re-run `static-gs-preseg` (`--pipeline.reuse-sidecar-if-present False`) before trusting picker ids.** new_env regenerated + verified (9 ids, androids 8/9 separate, projection purity 99–100 %).
* Teardown noise: `RuntimeError: cannot schedule new futures after shutdown` at "Training Finished" is viser's own inbound-websocket handler racing interpreter exit when a browser tab is still open — cosmetic. Our outbound pushes are guarded (`ViserDirectScene.is_closing` + guards in `request_render` / `_render_once` / `_viser_direct_register_ff_insert`). Note `feedforward_video_out` is currently declared but **no writer is implemented** — no mp4 is produced by the dynamic pipelines.

### Anchor keyframe gate: relative camera↔object orientation (2026-06-11)

The XFeat multi-anchor pool now captures a new keyframe (fresh object features) when the object's orientation **as seen from the camera** moves >`ROTATION_GATE_DEG` (22.5°) from every existing anchor — **not** the object's absolute world rotation as before. Relative orientation = `R_rel = R_cam_world^T @ R_object_world` ([`_relative_object_rotation`](dynamic_gs/utils/xfeat_motion.py)); the gate ([`_min_anchor_relative_distance_deg`](dynamic_gs/utils/xfeat_motion.py)) and anchor selection ([`_select_nearest_anchor_by_rotation`](dynamic_gs/utils/xfeat_motion.py)) both compare relative-to-relative.

* **Why:** XFeat/LighterGlue match on *appearance from a viewpoint*. The old absolute-object gate never fired when the **camera** moved while the object was still (or both moved together), so no fresh features were captured even though the view changed → match degradation / tracking loss. The relative gate fires on object-only, camera-only, or combined motion.
* Each `_Anchor` now stores `camera_rotation` (`camera_to_world[:3,:3]` at capture); the c2w was already passed to `_build_anchor` (used only for back-projection) — now retained. The output pose contract is **unchanged**: `_cumulative_*` (object D0→current world pose), the Kabsch composition, the Kalman filter, and the returned `MotionEstimate` are untouched — this is purely an internal keyframing-policy change.
* **Verified:** unit cases — object +30°/cam still → 30°; **cam +30°/object still → 30°** (was ~0° before, the bug); both +30° together → 0° (no spurious anchor). Recorded + live runs grow the pool via `[xfeat-anchor] … rel-view rot from nearest = …` as the arm camera orbits a static object; extract steady ~13–15 ms.

### Viser tracker-view features (2026-06-11)

Two additive controls in viser-direct (`ViserDirectScene`, shared by recorded + live; a **"Tracker view"** GUI folder), pushed each tick from the base via `_push_viser_camera_feed(camera, batch)`:

* **Live camera-feed thumbnail** ("Show camera feed", on by default): `gui.add_image` side-panel showing the current tracked frame's RGB (`batch["image"]`), refreshed in place each render (~ms JPEG encode, render-side only when a client is connected).
* **Follow-tracked-frame toggle** ("Follow tracked frame", off by default): snaps the viewer camera to the tracked frame's c2w each render so the splat view matches the feed. MUST use the same `R_viser = R_nerf @ _FLIP_YZ` nerfstudio→viser conversion as `_apply_initial_camera` — omitting the flip misaligns every frame except the initial one (the bug that surfaced + was fixed on first use).

### Tracker pose Kalman filter (2026-06-10)

*Validated 2026-06-11: user confirmed tracking looks clearly better with the filter ON (less stationary jiggle) than the raw-pose run (`xfeat_pose_filter_enabled False`); ON stays the default.*


Output-side SE(3) constant-velocity error-state Kalman filter on the XFeat/RANSAC pose, targeting stationary-object jiggle (root cause per the 2026-05-26 notes: per-tick match-set variance → different Kabsch subsets).

* [`tracker_common.PoseKalmanFilter`](dynamic_gs/utils/tracker_common.py) — 12-state ESKF (pos, vel, rot-err, ang-vel), `cv2.Rodrigues` exp/log, ~31 µs/call.
* **Filters ONLY the `MotionEstimate.rotation/translation` returned to the pipeline.** The tracker's internal `_cumulative_*` pose (anchor selection, anchor creation, next-tick prediction) stays raw — smoothing lag can never destabilize tracking. Failure ticks hold the last filtered pose (cosmetic; pipeline only applies on `success`).
* Knobs: `xfeat_pose_filter_*` in `DynamicGSModelConfig` (enabled by default). **Defaults retuned 2026-06-12** to the measured optimum: `accel_sigma=0.02`, `alpha_sigma=0.1`, meas sigma **20 mm / 10°** (U-curve on the static tail: 3 mm/0.5° → 3.4 mm/4.2° jiggle; 20/10 → 0.8/0.9 best; 40/20 → 8.3/4.0 WORSE — drift/snap sawtooth; do not raise past 20/10). The old 0.5° meas sigma over-trusted rotation 8× vs the actual ~3–4°/tick wander at low inliers. Snap gate env-tunable via `DGS_KF_SNAP_TRANS_M`/`DGS_KF_SNAP_ROT_DEG`. A **static-hold** (`xfeat_static_hold*`, trend-gated median over 10 ticks) sits after the KF and pins genuinely stationary objects (the KF plateau alone leaves ~2.4 mm/2.5°).
* **Synthetic bench** (3 mm / 0.5° measurement noise, 20 Hz): stationary jitter 5.22→2.48 mm, 0.844→0.406° (~2.1×); smooth motion (0.2 m/s, 30°/s) tracks at 2.27 mm mean (better than raw — velocity state models it); worst-case instantaneous 5 cm step settles to <5 mm in 4 ticks (~200 ms). Lower `accel_sigma` = more smoothing + slower step response.
* **Innovation gate (snap-reset):** per-tick pose jumps > 5 cm or > 10° (filter ctor defaults `snap_trans_m` / `snap_rot_rad`) cannot come from continuous motion — they are reacquisitions (object left the view and came back moved) or anchor-pool discontinuities. The filter snaps to the measurement and restarts the velocity estimate instead of smoothing through (which would overshoot — the step kicks the velocity state). Tracking-loss gaps are also safe by construction: failure ticks never call `filter()`, so no velocity extrapolation during the gap, and the reacquire `dt` is clamped to 50 ms.
* **Reacquire bench** (after 5 s gap): unmoved → noise floor in 1 tick; moved 2 cm/3° (below gate) → ~4 ticks; moved 30 cm/45° (gate fires) → noise floor in 1 tick. The actual reacquisition risk is UPSTREAM, not the KF: if the object rotated far from every stored anchor viewpoint while unseen, LighterGlue has no appearance match and tracking stays lost (new anchors are only created on success).
* Not yet validated on a live run — if jiggle persists at default settings, drop `accel_sigma` to 0.02 / `alpha_sigma` to 0.1 (~2.5×, 7-tick settle) before suspecting a different root cause (e.g. anchor-switch pose jumps, which the KF spreads over a few ticks rather than removes — an 8 mm jump takes ~6 ticks to converge).

### Phase 0b registration: NDP non-rigid + post-fusion cull (2026-06-10)

Replaced the rigid-only CPD/TEASER++ similarity fit as the **default** Phase-0b SAM3D registration with **NDP (Neural Deformation Pyramid)** non-rigid registration. CPD and TEASER++ are **not removed** — both remain selectable fallbacks via config.

* **Why:** the SAM3D model is a complete-but-approximate object; the masked-depth back-projection is a partial-but-metrically-accurate scan of the same object. A single rigid+scale fit can't conform the approximate model to the real geometry. NDP non-rigidly deforms the complete cloud onto the accurate partial scan.
* **Backend switch:** `sam3d_registration_backend: Literal["ndp", "cpd", "teaser"] = "ndp"` in BOTH [`StaticGSModelConfig`](dynamic_gs/static_gs_model.py) and [`DynamicGSModelConfig`](dynamic_gs/dynamic_gs_model.py) (kept byte-for-byte in sync). The `"ndp"` branch in [`register_and_fuse_sam3d_object`](dynamic_gs/utils/sam3d_fusion.py) reuses the existing rigid init (SAM3D-rotation + bbox-scale + centroid translate) as NDP's initialization, then non-rigidly deforms onto the target; `aligned_points`/`kept_points` become the warped cloud (NDP is non-linear → `similarity_transform` stays identity, `canonical_to_world_4x4` carries only the rigid-init approximation — fine since FoundationPose is no longer wired in).
* **Vendored NDP:** [`dynamic_gs/utils/ndp/`](dynamic_gs/utils/ndp/) (`nets.py` + `rigid_body.py`, pure-torch, **no pytorch3d**) + the wrapper [`dynamic_gs/utils/ndp_register.py`](dynamic_gs/utils/ndp_register.py) (`deform_source_to_target`). Runs in-process on GPU in the main `dynamic_gs` env. No checkpoint (no-learned optimization): model construct ≈ **3.8 ms**, solve ≈ **2.0 s/object** (hierarchical Sim3, 9 levels, 500 iters/level, 6000-pt subsample, full-cloud warp). Config defaults mirror `DeformationPyramid/shape_transfer.py`. The no-learned NDP hyperparameters (m=9, Sim3 motion, w_reg=0) were tuned by eye on the banana and screwdriver scenes to land on these values.
* **Post-fusion cull (two stages, in [`run_phase0b_fusion`](dynamic_gs/fusion/phase0.py)):** the inserted SAM3D points are culled against the trusted real surface so the accurate scan owns the visible side and SAM3D only fills the occluded back:
  1. **Proximity de-dup** (existing, tuned, ON by default): cull points within `tau = max(spacing(E)·1.3, 3 mm)` of the existing visible-surface Gaussians. NOTE: 3D-distance based → on **thin** parts it also removes the occluded back (the back sits within `tau` of the front). Known tradeoff.
  2. **In-front (occlusion) cull** (NEW, band = 0): [`cull_points_in_front`](dynamic_gs/fusion/phase0.py) drops any inserted point closer to the camera than the back-projected real front surface (depth-buffer test, inverse of `backproject_mask_to_world`, 2 px dilation). Keeps everything at-or-behind the front, including thin-part backs.
* **Prototype + tuning bench:** [`scripts/experiments/nonrigid_bench/`](experiments/nonrigid_bench/) — standalone NDP-vs-(SPARE/SyNoRiM) benchmark on a single SAM3 → SAM3D → back-project pair, with `02_run_ndp.py` (the reference deform+cull driver) and `view_results.py`. Cull knobs there: `NRB_CULL_PROXIMITY` (default on), `NRB_CULL_BAND_M` (default 0), `NRB_CULL_STRENGTH`/`NRB_CULL_TAU_FLOOR_M`.
* **Verified:** NDP branch through `register_and_fuse_sam3d_object` (65 888 kept pts, 2.0 s) + `cull_points_in_front` with a real `Cameras` (removed in-front pts). Full `ns-train static-gs` end-to-end run not yet re-run on this change.

### Static-phase init seed: online ICP+TSDF fusion (2026-06-01)

Replaced the naive per-frame back-projection + post-pass refine with **one streaming pass** that runs concurrent with capture.

* New shared utility: [`dynamic_gs/utils/online_fusion.py`](dynamic_gs/utils/online_fusion.py) — `OnlineFusion` class wrapping Open3D `ScalableTSDFVolume` + point-to-plane ICP. Verbatim port of `experiments/icp_fusion_mvp/online_fusion.py`.
* Concurrent runner: [`dynamic_gs/utils/fusion_runner.py`](dynamic_gs/utils/fusion_runner.py) — `ConcurrentFusionRunner` polls `static_scene/transforms.json` on a watcher thread, enqueues each newly-written keyframe to a worker thread that calls `add_frame`. On Enter: drain queue + `finalize()` → `depth_camera_init_points.ply`. Wired into both `capture_only.py` and `live_session.py`.
* **GPU port**: Open3D 0.19 tensor pipelines on this sm_120 GPU work. **Measured** 38.8× total speedup vs CPU (CPU 630 ms/frame → GPU 16 ms/frame at 800×800, 2.5 mm voxel, 30-frame avg). The previous note that "Open3D GPU TSDF is broken on this sm_120 GPU" is outdated and removed. The bench script lives in [`scripts/bench_gpu_fusion.py`](scripts/bench_gpu_fusion.py).
* **GPU now the default (2026-06-01)**: `OnlineFusion` was split into `_CpuOnlineFusion` (legacy `ScalableTSDFVolume` + `registration_icp`) and `_GpuOnlineFusion` (`o3d.t.pipelines.slam.Model` VoxelBlockGrid + `multi_scale_icp` on CUDA). The public `OnlineFusion(...)` constructor auto-selects GPU when `o3d.core.cuda.is_available()`; set `DGS_FUSION_DEVICE=cpu` to force the fallback. Measured on validate_run_1 (71 frames, 800×800, full pipeline including `finalize` + `adaptive_downsample` + PLY write): **CPU 73.9 s → GPU 7.7 s (9.6× total)**; per-frame `add_frame` 17.6 ms mean (24 ms p90). Initial block count lowered to 8 k (was 40 k in the bench) to avoid OOM when other GPU workloads share the device.
* **TSDF voxel**: was 2.5 mm CPU. New default **1.5 mm GPU** ([`online_fusion.py`](dynamic_gs/utils/online_fusion.py) `TSDF_VOXEL_M = 0.0015`). 1.0 mm OOMs at 16 GB GPU (Open3D hashmap rehash overhead, not actual block memory). 1.5 mm → 10.1 M points on validate_run_1, 5 ms integrate.
* **Profiler** [`scripts/profile_fusion.py`](scripts/profile_fusion.py): per-substep breakdown on CPU; integrate dominated (320 ms / 51 %), ICP only 127 ms / 20 %. Drove the GPU decision.

### Static-phase seed downsampling: near/far split (2026-06-01)

Output of the TSDF fusion is too uniformly dense for the Splatfacto seed. Tried per-point feature scoring; abandoned. Final strategy is simple binary near/far split.

* **Tried + dropped**: percentile-rank feature score (curv + color gradient), with multiple variants — depth-multiplicative penalty, depth-linear-floor penalty, depth-as-effective-threshold-multiplier. All abandoned. Reason: feature scoring on TSDF surfaces couldn't separate real edges from grazing-wall noise; curvature (`λ₀/Σλ`) is unstable at grazing angles because the smallest eigenvalue of the local cov spikes from depth noise on slanted thin-band TSDF surfaces. Raising the curv threshold killed real subtle edges before killing the wall ellipses. Color-gradient turned out to be a small fraction (~5 %) of the signal because TSDF integration smooths small color deltas.
* **Shipped (initial form)**: [`scripts/adaptive_downsample.py`](scripts/adaptive_downsample.py). Keep all points within **1.0 m of the last camera pose** at native 1.5 mm density. Voxel-downsample the rest to **5 mm**. On validate_run_1: 10.15 M → 1.14 M (8.9× reduction, near zone = 3.4 % of points).
* **Auto-wired (2026-06-01)**: the same logic now lives in [`dynamic_gs/utils/online_fusion.py`](dynamic_gs/utils/online_fusion.py) as `adaptive_downsample(pc, last_cam_xyz, ...)`, invoked between `OnlineFusion.finalize()` and the PLY write in BOTH:
  * [`ConcurrentFusionRunner.stop_and_finalize`](dynamic_gs/utils/fusion_runner.py) — used by `capture_only.py` + `live_session.py` (last camera pose read from the freshly-written `transforms.json`).
  * [`fuse_recorded_dataset`](dynamic_gs/utils/online_fusion.py) — used by the recorded-data post-pass.
* Hyperparameters as module constants `NEAR_RADIUS_M = 1.0`, `FAR_VOXEL_M = 0.005` in `online_fusion.py`. Standalone script kept as-is for ad-hoc runs against existing PLYs. Re-verified end-to-end on validate_run_1: 13.77 M → 1.50 M (9.1× reduction).
* **GPU port (2026-06-01)**: `adaptive_downsample` now uses `o3d.t.geometry.PointCloud` on CUDA when available; auto-falls back to CPU. Measured **7.1× speedup** on validate_run_1 (CPU 4437 ms → GPU 624 ms, 13.77 M-point cloud). Gotcha: Open3D 0.19's CUDA reduction kernel rejects `Tensor.sum(dim=1)` on large Float32 tensors ("Unsupported data type"); workaround is manual `sq[:,0]+sq[:,1]+sq[:,2]` elementwise add (lowers to safe ops). Result point counts differ ~6 % between CPU and GPU paths due to different voxel tie-breaking — both correctly enforce the 5 mm grid.
* Reference camera pose is the **last** frame's `transform_matrix` (the operator's final viewpoint), not the first.
* Feature-analysis tooling kept for future work: [`scripts/analyze_feature_score.py`](scripts/analyze_feature_score.py) (rank + absolute-threshold modes with cached `.npy` sidecars). Not on any code path.

### Capture-only recorder + bootstrap rewrites (2026-05-31 → 2026-06-01)

* New: [`scripts/capture_only.sh`](scripts/capture_only.sh) / [`scripts/capture_only.py`](scripts/capture_only.py) — record-only flow, no SAM3 / SAM3D / training. Static phase = publisher dedup recorder + concurrent fusion. Dynamic phase = reader-side 30 fps SHM polling, no dedup. Two Enter presses (switch phase / stop). Default dir = timestamped subdir under `data_teleoperation/datasets/`.
* [`bootstrap_live.sh`](scripts/bootstrap_live.sh) capture flow rewritten: prompt via CLI (no second interactive ask), record-from-launch with on-screen "press Enter when centered", SAM3-zero-mask retry loop instead of crash.
* Static keyframe dedup default lowered to **2 cm + 20° OR-rule** in `live_shm_reader.py` (was 1 cm; raised again because 1 cm produced too many near-duplicates on slow sweeps).
* `outputs/` wiped + suppressed: [`dynamic_gs/__init__.py`](dynamic_gs/__init__.py) monkeypatches three nerfstudio write-sites (`ExperimentConfig.save_config`, `Trainer.train`'s `dataparser_transforms.json` write, `writer.setup_event_writer` tensorboard branch). All artifacts now live under the dataset dir.

### Tracker purge — XFeat only (2026-05-26)

The 5-tracker dispatch (cotracker / tapir / tapnext / klt / xfeat) is gone. XFeat is now the only supported tracker. Surviving runtime files:

- `dynamic_gs/utils/xfeat_motion.py` — the tracker
- `dynamic_gs/utils/tracker_common.py` — shared `MotionEstimate` dataclass + Kabsch/RANSAC helpers (extracted from the old `cotracker_motion.py` because XFeat depended on those static methods)
- `dynamic_gs/utils/_purged/` — `cotracker_motion.py`, `tapir_motion.py`, `tapnext_motion.py`, `klt_motion.py`, `live_ros_subscriber.py`. Kept for reference, never imported by the runtime, their imports are intentionally broken.
- `dynamic_gs/utils/sam2.py` — deleted entirely.

Pipeline + model config changes:
- Model: 50+ legacy fields removed (`cotracker_*`, `tapir_*`, `tapnext_*`, `klt_*`, `dynamic_tracker` Literal). Only `xfeat_*` fields remain.
- Pipeline: `_TRACKER_LABELS`, `tracker_kind`/`dynamic_tracker` reads, and the 5-branch `_initialize_motion_estimator` dispatch all collapsed to single XFeat constructor.
- `utils/__init__.py` keeps `CoTrackerMotionEstimate` as an alias for `tracker_common.MotionEstimate` (a couple of debug-viz call sites import the old name).
- `enable_cotracker_rigid_motion: bool = True` kept as the master tracker switch despite the legacy name (config-compat).

To revive a purged tracker: port its `CoTrackerMotionEstimator._foo` static-method calls to `tracker_common.foo`, move the file back to `dynamic_gs/utils/`, re-add the dispatch branch in `_initialize_motion_estimator`. See [_purged/__init__.py](dynamic_gs/utils/_purged/__init__.py) for the recipe.

### Per-tick object mask removal — XFeat (2026-05-26)

> **SUPERSEDED (2026-06-12):** the per-tick object mask was later RE-ADDED, and full-frame extraction was dropped. Current code:
> - `xfeat_object_mask_filter: bool = True` (in [`dynamic_gs_model.py`](dynamic_gs/dynamic_gs_model.py), default ON) renders the tracked instance's mask each tick and uses it as a **post-match** filter — matches landing outside the object's predicted footprint are dropped before RANSAC. Without it the pose pins to static background once the object is grasped + lifted ("stops moving"). Set to `False` to restore the gripper-keep-only behavior described below.
> - Extraction no longer runs on the full natural frame: `xfeat_crop_to_object_bbox=True` (in [`dynamic_gs_model.py`](dynamic_gs/dynamic_gs_model.py)) crops rgb+depth+camera to the object's projected bbox padded by `xfeat_crop_padding_px=300` (raised from 60 on 2026-06-12 — the 60 px box clipped the CNN receptive field of boundary keypoints, destabilizing descriptors).
> The 2026-05-26 measurements below are historical (top_k was 300 then; now 3000).

`render_object_mask + erode + dilate + pre-mask` was firing every tick for XFeat (~4 ms/tick) and produced **zero measurable quality benefit**. Removed entirely; XFeat now extracts on the full natural frame and uses **only `gripper_keep`** for post-match filtering.

Net effect at 800×800, `xfeat_top_k=300`:
- `DN.3j_object_mask_render`: 4ms → 0ms
- `DN.3_tracker_motion`: 34 → 21 ms (-38%)
- Effective rate: 12 Hz → 17.7 Hz
- Inliers under motion: 100-160 with min ~19-30 (well above `min_track_points=12`)

Anchor pool quality holds — new anchors filter on `gripper_keep` only (instead of `gripper_keep ∩ object_mask`), but LighterGlue's attention rejects background pairs and RANSAC catches anything that slips through. KLT path (still in `_purged/`) **does** need the mask structurally because it samples FAST keypoints from inside it.

### XFeat config sweet spots — known shake / RANSAC failure modes (2026-05-26)

> **UPDATE (2026-06-12):** `xfeat_top_k` default is now **3000** (in [`dynamic_gs_model.py`](dynamic_gs/dynamic_gs_model.py)), raised from 300 in `06d2c47` (2026-06-03). The 300-era reasoning below still holds as an argument against going *too low* (match-set variance → shake) — it does not endorse 300 specifically. `xfeat_lighterglue_depth_confidence=-1.0` and `xfeat_ransac_iterations=32` are unchanged and current.

These three settings together produced a stable tracker (17-30 Hz, inliers/correspondences ≈ 1.0). Deviations caused regressions that took hours to diagnose:

- **`xfeat_top_k=300`** — DO NOT drop below 200. At `top_k=100` the post-depth-filter survivor set varies tick-to-tick → different ~30-50 point Kabsch subsets → visible object shake on stationary scenes. The "tick_total went UP" symptom (28→31ms) is a secondary clue — fewer keypoints didn't even help speed because XFeat extract is image-bound, not K-bound.
- **`xfeat_lighterglue_depth_confidence=-1.0`** — DISABLED. Enabling early-exit (e.g. 0.95) picks a different transformer layer each tick → different match set → noisy Kabsch → shake. Same root cause as the `top_k=100` shake: match-set variance.
- **`xfeat_ransac_iterations=32`** — was 128, no quality loss at 32 with `~64-100` candidate correspondences. Saved ~4ms.

Inlier diagnostic now in `[tracker-rate]`: `inliers=X/Y avg (min Z, n=N/F)`. Watch `min` — if it consistently drops below 15 under fast motion, the mask removal would need to be revisited for anchor creation.

### train_lock disable was a mistake — DO NOT redo (2026-05-26)

Attempted to disable Nerfstudio's `train_lock` in `NoSaveTrainer.setup` to let renders and tracker ticks run concurrently. Worked in isolation (`LIVE.between_tick_gap` 27 → 2 ms) but caused **GPU contention** between the splat renderer and XFeat/LighterGlue kernels:

| | Baseline (lock on) | Lock off + render every tick | Lock off + render every 3rd tick |
|---|---|---|---|
| outside-tick | 27 ms | 2 ms | 4 ms |
| tracker_motion | 28 ms | 77 ms | 25 ms |
| effective rate | 17 Hz | 10 Hz | 30 Hz |

Net throughput unchanged at every-tick-render — the lock was hiding render cost behind serialization, not wasting cycles. Only worked at N=3 throttle but visual rate dropped to ~10 Hz and the user perceived it as "not updating." Reverted everything. The achievable lock-disable win requires bypassing server-side rendering entirely (supervisor's "push transforms directly to Viser via `GaussianSplatHandle.position`/`.wxyz`" — browser does WebGL splat) — that's a 1-2 day rewrite not yet attempted.

Files to NOT re-touch unless attempting the full Viser-direct rewrite:
- `NoSaveTrainer.setup` — currently clean (no override of `train_lock` or `_update_viewer_state`)
- Pipeline `live_render_kick_every_n_ticks` config exists with default N=1 (inert) for future experiments

### Killing ns-train safely — `kill <bash_pid>` is wrong (2026-05-26)

`nohup ns-train ... > log 2>&1 &` returns the **bash wrapper PID**, not the Python process. `kill $BASHPID` kills only the wrapper; the Python process keeps running, holding GPU memory and CUDA streams. Three of these zombies stacked up during this session and caused a 2× tracker slowdown ([tracker-rate] dropped from ~18 Hz to ~10 Hz) before I noticed.

Correct sequence:
```bash
PYPID=$(ps -ef | grep "ns-train.*dynamic-gs" | grep -v grep | awk '{print $2}')
kill -9 $PYPID
pkill -9 -f "live_ros_publisher"
# Then ALWAYS verify:
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

If `nvidia-smi` shows any leftover python pids holding VRAM, kill those by PID directly. Any unexplained tracker slowdown should immediately trigger a zombie check.

### Live mode timing-gap decomposition (2026-05-26)

Added per-step instrumentation to localise `LIVE.between_tick_gap`:

- `GAP.trainer_outer_loop` — wall-clock between `train_iteration` return and next entry. Captures Nerfstudio's `AFTER_TRAIN_ITERATION` callbacks, writer scalars, `_update_viewer_state`, eval check, `write_out_storage`. Typically **~24-27 ms** in live tracking-only mode and is the dominant chunk.
- `GAP.pipeline_prelude` — `train_iteration` entry → `_tracker_tick_live` start (= `_sync_phase` + pipeline dispatch). Typically <0.1 ms.
- `GAP.pipeline_postlude` — `_tracker_tick_live` exit → `get_train_loss_dict` return (= zero-loss dummy + timing-summary check). Typically <0.2 ms.

If `between_tick_gap` is large, ~100 % of it lives in `GAP.trainer_outer_loop`. The pipeline-side contributes essentially zero overhead in tracking-only mode. To shrink the gap further would require overriding `Trainer.train()` (the whole outer loop) — not just `train_iteration`.

### Live tracker 30 Hz fix (2026-05-26)

Root cause: gazebo RTF=0.32 (sim ran at 32 % real-time) → camera produced ~10 wall-clock fps; `rostopic hz` reported "30 Hz" using sim time and misled the diagnosis for hours.

Sim-side fixes (the ones that solved it):
- Replaced triangle-mesh `<collision>` with cylinder/box primitives on 6 world objects (coke_can, banana, fidget_spinner, side_plate, wooden_box, bolt); originals kept as XML comments.
- Dropped ODE `<iters>` 500 → 50 in [empty_world.world](../../dev/teleop/catkin_ws/src/active_camera_arm_control/active_camera_arm_gazebo/worlds/dynamic_gaussian_splat/empty_world.world).
- Capped pose-plugin `<updateRate>` 0.0 → 250 Hz in both dynaarm URDFs (was firing every 1000 Hz world tick).

Publisher-side improvements kept as permanent wins even though they did NOT fix the root cause:
- RGB `/compressed` (JPEG) + depth `/compressedDepth` (PNG-16UC1) transports, with auto-launched C++ `image_transport republish` for depth and a decoder that skips the 12-byte ConfigHeader.
- Worker-thread architecture in [live_ros_publisher.py](dynamic_gs/utils/live_ros_publisher.py): `_on_synced` just enqueues into `queue.Queue(maxsize=4)` with drop-oldest; `_worker_loop` drains and does cv_bridge + pose interp + mask render + shm write.
- 50 Hz throttle (`_POSE_JOINT_MIN_DT_SEC = 0.02`) on pose/joint callbacks (we only need bracketing samples per ~33 ms image stamp).

Diagnostic dead-ends — removed; do not re-add:
- E-core CPU pinning (`taskset -c 20-23`) — made the publisher slower because E-cores are slower per clock than P-cores.
- `DGS_PUB_NO_POSE_JOINT` / `DGS_PUB_SKIP_POSE_JOINT` env vars — tested the wrong hypothesis (pose/joint GIL contention was never the cause).
- `_count_rgb_cb` / `_count_depth_cb` + `_maybe_log_rates` — measured per-topic inbound Hz; answer was always "10 Hz" so the cap was upstream of rospy.
- `rospy.AnyMsg` + manual `struct.unpack` bypass — pointless once wall-vs-sim clock measurement revealed RTF was the real issue.

Confirmation tool: [/tmp/test_wall_rate.py](file:///tmp/test_wall_rate.py) subscribes to `/clock`, `/image_raw`, `/image_raw/compressed` via `rospy.AnyMsg` and computes rate from `time.time()`; `/clock` wall-rate ÷ physics target gives true RTF. RTF < 1.0 ⇒ no rospy tuning can help.

### Splatfacto per-iteration sequence (with code references)

Reference trace of what Nerfstudio + Splatfacto actually do per training step. Useful as the baseline to compare the dynamic-gs custom phase/optim logic against.

- Outer loop sets `self.step = step` — [trainer.py:247](../nerfstudio/nerfstudio/engine/trainer.py#L247)
- Fire `BEFORE_TRAIN_ITERATION` callbacks — [trainer.py:260-263](../nerfstudio/nerfstudio/engine/trainer.py#L260-L263)
- `step_cb` stashes step, optimizers, schedulers onto the model — [splatfacto.py:407-410](../nerfstudio/nerfstudio/models/splatfacto.py#L407-L410)
- Call `train_iteration(step)` — [trainer.py:266](../nerfstudio/nerfstudio/engine/trainer.py#L266)
- Zero gradients on this step's active param groups — [trainer.py:497](../nerfstudio/nerfstudio/engine/trainer.py#L497)
- `pipeline.get_train_loss_dict(step)` called — [trainer.py:502](../nerfstudio/nerfstudio/engine/trainer.py#L502)
- Pipeline calls `model.get_outputs(camera)` — [splatfacto.py:485](../nerfstudio/nerfstudio/models/splatfacto.py#L485)
  - Apply learned camera-pose correction — [splatfacto.py:501](../nerfstudio/nerfstudio/models/splatfacto.py#L501)
  - Build view matrix + intrinsics K — [splatfacto.py:534-535](../nerfstudio/nerfstudio/models/splatfacto.py#L534-L535)
  - Call `gsplat.rasterization(...)` → `render, alpha, self.info` — [splatfacto.py:555-575](../nerfstudio/nerfstudio/models/splatfacto.py#L555-L575)
  - `strategy.step_pre_backward(...)` (registers `means2d` to retain its gradient) — [splatfacto.py:577-579](../nerfstudio/nerfstudio/models/splatfacto.py#L577-L579)
  - Composite rendered RGB with background — [splatfacto.py:583](../nerfstudio/nerfstudio/models/splatfacto.py#L583)
- Pipeline calls `model.get_loss_dict(outputs, batch)` — [splatfacto.py:652](../nerfstudio/nerfstudio/models/splatfacto.py#L652)
  - Composite GT image with background — [splatfacto.py:660](../nerfstudio/nerfstudio/models/splatfacto.py#L660)
  - Compute L1 = `mean(|gt − pred|)` — [splatfacto.py:673](../nerfstudio/nerfstudio/models/splatfacto.py#L673)
  - Compute `1 − SSIM(gt, pred)` — [splatfacto.py:674](../nerfstudio/nerfstudio/models/splatfacto.py#L674)
  - Combine: `(1 − ssim_lambda)·L1 + ssim_lambda·(1 − SSIM)` — [splatfacto.py:689](../nerfstudio/nerfstudio/models/splatfacto.py#L689)
- Sum loss_dict into scalar `loss` — [trainer.py:503](../nerfstudio/nerfstudio/engine/trainer.py#L503)
- `grad_scaler.scale(loss).backward()` fills `.grad` on every param tensor and on `info["means2d"]` — [trainer.py:504](../nerfstudio/nerfstudio/engine/trainer.py#L504)
- `optimizer_scaler_step_some` → Adam step on each active param group — [trainer.py:510](../nerfstudio/nerfstudio/engine/trainer.py#L510)
- `scheduler_step_all` → schedulers decay LRs for next step — [trainer.py:527](../nerfstudio/nerfstudio/engine/trainer.py#L527)
- `train_iteration` returns to outer loop
- Fire `AFTER_TRAIN_ITERATION` callbacks — [trainer.py:269-272](../nerfstudio/nerfstudio/engine/trainer.py#L269-L272)
- `step_post_backward` dispatcher runs — [splatfacto.py:365](../nerfstudio/nerfstudio/models/splatfacto.py#L365)
- Delegates to `strategy.step_post_backward(...)` — [splatfacto.py:367-374](../nerfstudio/nerfstudio/models/splatfacto.py#L367-L374)
  - If outside refinement window or wrong step → return early
  - Else read `info["means2d"].grad`, decide clone/split/prune, mutate `gauss_params` + each Adam's `m`,`v` state in lockstep
  - Every `reset_alpha_every × refine_every` steps: reset all opacities low
- Loop to next step

### Splatfacto `get_outputs(camera)` — the render function

Pure forward render: given one camera, returns the rendered image (plus depth/alpha). Does **not** compute loss, does **not** call backward, does **not** modify Gaussians.

1. **Apply pose correction** — `camera_optimizer.apply_to_camera(camera)` adds the learned 6D offset to the dataset c2w (training only, if camera-opt is on) — [splatfacto.py:501](../nerfstudio/nerfstudio/models/splatfacto.py#L501).
2. **Pick which Gaussians to render** — all of them, unless a `crop_box` is set (viewer feature) — [splatfacto.py:506-528](../nerfstudio/nerfstudio/models/splatfacto.py#L506-L528).
3. **Build camera matrices** — `viewmat` from corrected c2w, intrinsics `K` — [splatfacto.py:534-535](../nerfstudio/nerfstudio/models/splatfacto.py#L534-L535).
4. **Pick render mode** — `"RGB+ED"` if depth is needed this step, else `"RGB"` — [splatfacto.py:544-547](../nerfstudio/nerfstudio/models/splatfacto.py#L544-L547).
5. **Active SH degree** — `min(step // sh_degree_interval, max_sh_degree)` — coarse-to-fine color schedule — [splatfacto.py:549-553](../nerfstudio/nerfstudio/models/splatfacto.py#L549-L553).
6. **Call `gsplat.rasterization(...)`** — the differentiable splatting kernel; inputs the 7 param tensors + camera matrices; returns `render`, `alpha`, `self.info` — [splatfacto.py:555-575](../nerfstudio/nerfstudio/models/splatfacto.py#L555-L575).
7. **`strategy.step_pre_backward(...)`** — registers `means2d` so its gradient is retained through backward (needed by densification) — [splatfacto.py:577-579](../nerfstudio/nerfstudio/models/splatfacto.py#L577-L579).
8. **Composite with background** — `rgb = render + (1 − alpha) · background`, clamp [0,1] — [splatfacto.py:582-584](../nerfstudio/nerfstudio/models/splatfacto.py#L582-L584).
9. **Apply bilateral grid** — only if enabled and training; per-image color correction — [splatfacto.py:587-589](../nerfstudio/nerfstudio/models/splatfacto.py#L587-L589).
10. **Extract depth** — mask out empty regions (alpha = 0) — [splatfacto.py:591-595](../nerfstudio/nerfstudio/models/splatfacto.py#L591-L595).
11. **Return** `{"rgb", "depth", "accumulation" (= alpha), "background"}` — [splatfacto.py:600-604](../nerfstudio/nerfstudio/models/splatfacto.py#L600-L604).

**`alpha`** is per-pixel accumulated opacity in `[0, 1]`: `alpha = 1 − Π(1 − αᵢ)` over all Gaussians touching that pixel. Used to composite the background and to mask depth.

### Splatfacto `get_loss_dict(outputs, batch, metrics_dict)` — the loss function

Takes the rendered output and GT batch, returns a dict of scalar losses that the trainer sums and backprops.

1. **Composite GT with background** — same background as the render, so they're compared on equal footing — [splatfacto.py:660](../nerfstudio/nerfstudio/models/splatfacto.py#L660).
2. **Apply mask if present** — if `batch["mask"]` exists, multiply both GT and pred so masked pixels contribute zero — [splatfacto.py:665-671](../nerfstudio/nerfstudio/models/splatfacto.py#L665-L671).
3. **L1 loss** — `mean(|gt − pred|)` — [splatfacto.py:673](../nerfstudio/nerfstudio/models/splatfacto.py#L673).
4. **SSIM loss** — `1 − SSIM(gt, pred)`; windowed structural similarity — [splatfacto.py:674](../nerfstudio/nerfstudio/models/splatfacto.py#L674).
5. **Combine** — `main_loss = (1 − ssim_lambda)·L1 + ssim_lambda·(1 − SSIM)`, default `ssim_lambda = 0.2` — [splatfacto.py:689](../nerfstudio/nerfstudio/models/splatfacto.py#L689).
6. **Scale regularization** (optional, only if `use_scale_regularization=True`, every 10 steps) — penalizes Gaussians with large max/min scale ratio (PhysGaussian) — [splatfacto.py:675-686](../nerfstudio/nerfstudio/models/splatfacto.py#L675-L686).
7. **MCMC regularizers** (only if `strategy="mcmc"`) — L1 on opacity and exp(scale) — [splatfacto.py:693-702](../nerfstudio/nerfstudio/models/splatfacto.py#L693-L702).
8. **Camera optimizer loss** (training only) — regularization on learned pose offsets — [splatfacto.py:704-706](../nerfstudio/nerfstudio/models/splatfacto.py#L704-L706).

Returns: `{"main_loss", "scale_reg", possibly "mcmc_opacity_reg", "mcmc_scale_reg", "camera_opt_*"}`.

### Splatfacto optional features (opt-in via config)

| Feature | Flag | Default | What it does |
|---|---|---|---|
| **Bilateral grid** | `use_bilateral_grid` | `False` | Per-image learnable color correction (exposure/WB drift) |
| **Camera-pose optimization** | `camera_optimizer.mode` | `"off"` | Learnable 6D pose offset per training image (`"SO3xR3"` or `"SE3"`) |
| **Antialiased rasterization** | `rasterize_mode` | `"classic"` | `"antialiased"` adjusts opacity to keep splats consistent across resolutions; reduces aliasing |
| **Scale regularization** | `use_scale_regularization` | `False` | Penalizes spiky/elongated Gaussians (max/min scale ratio > `max_gauss_ratio`); from PhysGaussian |
| **Absolute-gradient densification** | `use_absgrad` | `True` | Uses absolute screen-space grad instead of signed; densifies more aggressively |
| **MCMC strategy** | `strategy` | `"default"` | `"mcmc"` swaps clone/split/prune for Langevin-dynamics sampling; adds opacity + scale L1 regs |
| **Random init** | `random_init` | `False` | Init Gaussians in a random cube instead of SfM points |
| **Output depth during training** | `output_depth_during_training` | `False` | Render depth every train step (slower, enables depth losses) |
| **Color-corrected metrics** | `color_corrected_metrics` | `False` | Histogram match before PSNR/SSIM — fair comparison under color drift |
| **Background color** | `background_color` | `"random"` | `"random"` / `"black"` / `"white"`; randomization prevents memorizing a fixed bg |
| **Max Gaussian cap** | `max_gs_num` | `1_000_000` | Hard cap; densification stops past this |
| **SH degree schedule** | `sh_degree` + `sh_degree_interval` | `3`, every `1000` steps | Activates one extra SH band per interval — coarse-to-fine on color |
| **Resolution schedule** | `num_downscales` + `resolution_schedule` | `2`, every `3000` steps | Start at 1/4 res, double up to full — coarse-to-fine on image res |

In dynamic-gs, `camera_optimizer.mode="SO3xR3"` is overridden on in `DynamicGSModelConfig`.

### Live-mode viewer refresh fix (2026-05-25)

> **PRE-REWRITE (2026-06-12 note):** this describes code in the since-deleted monolith `dynamic_gs_pipeline.py` (5329 LOC, removed in the 2026-05-30→06-01 rewrite). `_tracker_tick_live` and `_apply_cotracker_motion` no longer exist; the logic moved into the split `dynamic_gs_pipeline_{base,live,recorded}.py`. Kept for the design reasoning, not as a current code map.

**Problem.** In live tracking-only mode the tracker mutated object Gaussian means at ~8 Hz, but the viser viewer only repainted them at ~0.5–2 Hz (camera-move was smooth, object motion was not). Two throttles on the render path were the cause:

1. **`viewer.update_scene(step)`'s step-count gate** at [viewer.py:520](../nerfstudio/nerfstudio/viewer/viewer.py#L520) — fires a `"step"` render action only every `render_freq = train_util * vis_time / (train_time - train_util * train_time)` train iterations. With `train_time` very small (tracker-tick-only step), `render_freq` blows up to ~70+ iters/render → ~3–4 actions/sec.
2. **`RenderStateMachine.action()` filter** at [render_state_machine.py:99](../nerfstudio/nerfstudio/viewer/render_state_machine.py#L99) — ignores `"step"` when `self.state == "low_move"`. Viser camera `on_update` events (fired by the browser even on passive interaction) keep the state in `low_move`. So even the few `"step"` actions that did fire were dropped → state never promoted to `"high"` → `_calculate_image_res` returned the `vis_rays_per_sec / target_fps` fallback (~60 px) instead of `max_res`.

**Why `"rerender"` is the wrong action.** The state transitions in [render_state_machine.py:73-78](../nerfstudio/nerfstudio/viewer/render_state_machine.py#L73-L78):
- `low_static + step → high` (promotes)
- `high + rerender → low_static` (DEMOTES — `"rerender"` means "restart at low res then re-promote", not "scene changed material")

**Fix.** Don't go through `action()` at all. Directly push a high-res render on every fresh tracker tick:

```python
# dynamic_gs/dynamic_gs_pipeline.py — _force_viewer_rerender helper
sm.state = "low_static"                              # force out of low_move
sm.next_action = RenderAction("step", camera_state)   # queue high-res step
sm.render_trigger.set()                               # wake render thread
```

Called from `_tracker_tick_live` immediately after `_apply_cotracker_motion` returns. Zero modifications to nerfstudio core code — the back-reference to the trainer (and thus `trainer.viewer_state.render_statemachines`) is acquired via `training_callback_attributes.trainer` in `get_training_callbacks`. *(All three symbols were in the deleted monolith — see the banner above.)*

**Net effect.** Visual rate becomes whatever `_render_img` itself can do (limited by render cost + `train_lock` contention), instead of being throttled to 0.5–2 Hz. `outside-tick` in the `[tracker-rate]` line grows correspondingly — that's real render work now, not dedup-spinning.

**What this means for the rewrite.** The viewer / trainer / tracker integration is currently three independent throttles fighting each other (`update_scene`'s step gate, `action()`'s state filter, `train_lock`). In a clean rewrite, consider:
- A direct "scene mutated" signal from the tracker to the render thread that bypasses the state machine entirely (single threading.Event per client).
- Either eliminate `train_lock` in tracking-only mode (the trainer doesn't mutate during a tracker tick) or use RWLock so renders can run concurrently with read-only tracker ticks.
- The render state machine's `low_move/low_static/high` heuristic is designed for interactive NeRF training. For live tracking it's mostly noise — a fixed `high` state with `max_res` would be simpler and equally correct.

<!-- ============================================================ -->
<!-- END: CLEANUP NOTES — existing project documentation below     -->
<!-- ============================================================ -->

---

## Project Overview

**dynamic-gs** is a static + dynamic Gaussian Splatting system for robotic teleoperation, integrated with [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio). The static phase fits a Splatfacto scene; the dynamic phase tracks objects via XFeat and (optionally) feedforward-decodes newly revealed surfaces. Designed for live RGB-D streams from a single arm-mounted camera.

The codebase was rewritten 2026-05-30 → 2026-06-01. The historical monolith (`dynamic_gs_pipeline.py`, 5329 LOC) was deleted; capabilities are now split across three thin pipelines, the dynamic logic into a shared base.

## Installation

```bash
pip install -e .   # from scripts/
```

ns-train auto-discovers our methods via the `nerfstudio.method_configs` entry-point in `pyproject.toml`. Method names registered: `static-gs`, `static-gs-preseg`, `dynamic-gs`, `dynamic-gs-live`.

## Conda Environments

| Env | Python | torch | sm_120 native | Role |
|---|---|---|---|---|
| `dynamic_gs` | 3.12 | 2.11+cu128 | ✅ | Main env: hosts all four ns-train methods, XFeat tracker, Open3D 0.19 (TSDF on GPU), nerfstudio, gsplat. |
| `sam3_dynamic_gs` | 3.12 | 2.11+cu128 | ✅ | SAM3 + Fast-SAM3D subprocess env. Invoked via `conda run -n sam3_dynamic_gs python ...` from `utils/sam3d.py` and `utils/sam3_segmentation.py`. |
| `dynamic_gs_ros` | 3.8 | none | n/a | Minimal ROS Noetic env for the live publisher subprocess. ROS bindings come from `/opt/ros/noetic/lib/python3/dist-packages` via `source /opt/ros/noetic/setup.bash`. The publisher spawn wrapper sets `PYTHONNOUSERSITE=1` — without it, user-local pyrender shadows the env's. |
| `anysplat_dynamic_gs` | 3.12 | — | ✅ | AnySplat feedforward decoder env (persistent worker; see `utils/anysplat_decode.py`). |

## Running

Three top-level scripts cover the common flows. Defaults are chosen so most invocations are zero-argument.

```bash
# Capture-only: record a fresh static + dynamic dataset, no training
scripts/capture_only.sh
# default: data → datasets/<YYYY-MM-DD_HHMMSS>/, 2cm/20° dedup static, 30 fps dynamic

# Full pipeline: capture + train static + go live
scripts/bootstrap_live.sh <data_dir> [sam3_prompt]

# Resume on a pre-trained dataset (skip capture + static training)
scripts/resume_live.sh <data_dir>
# requires <data_dir>/static_scene/post_fusion_state.pt
```

Direct method invocations also work:
```bash
ns-train static-gs        --data <data_dir> --pipeline.model.sam3_prompt_text "..."
ns-train static-gs-preseg --data <data_dir> --pipeline.text-prompts "..."   # per-Gaussian SAM IDs, no SAM3D/CPD
ns-train dynamic-gs       --data <data_dir>   # recorded dataset
ns-train dynamic-gs-live  --data <data_dir>   # live SHM stream
```

`outputs/` is intentionally empty across runs — see [`dynamic_gs/__init__.py`](dynamic_gs/__init__.py) for the three monkeypatches that suppress nerfstudio's `config.yml` / `dataparser_transforms.json` / tensorboard writes. All artifacts live under the dataset dir.

## High-Level Architecture

### Pipelines (4 modules)

```
DynamicGSPipelineBase            (dynamic_gs_pipeline_base.py)
├─ RecordedDynamicGSPipeline    (dynamic_gs_pipeline_recorded.py)
└─ LiveDynamicGSPipeline        (dynamic_gs_pipeline_live.py)

StaticGSPipeline                 (static_gs_pipeline.py)
```

* **`StaticGSPipeline`** — fits Splatfacto on the static dataset, then runs Phase 0a (SAM3 + Fast-SAM3D, in `fusion/phase0.py`) and Phase 0b (NDP non-rigid registration by default — CPD/TEASER++ still selectable — + insertion + post-fusion cull). Writes the warm-cache snapshot `<data>/static_scene/post_fusion_state.pt`.
* **`DynamicGSPipelineBase`** — shared dynamic-phase logic: XFeat tracker tick, feedforward dispatcher (`rgbd_decode` or `anysplat_decode`), viser-direct push, persistent per-object identity buffers.
* **`RecordedDynamicGSPipeline`** — feeds the base from a recorded `dynamic_scene/`.
* **`LiveDynamicGSPipeline`** — feeds the base from a `LiveShmSubscriber` polling the ROS publisher's shared memory.

### Models (2 modules)

* **`StaticGSModel`** (`static_gs_model.py`) — straight subclass of `SplatfactoModel` + the four persistent buffers (`object_flags`, `object_instance_ids`, `sam3d_init_target_flags`, `inserted_flags`). Uses `NoRefineStrategy` so densification is OFF during static training. Means LR is zeroed so seed positions stay fixed.
* **`DynamicGSModel`** (`dynamic_gs_model.py`) — superset used by recorded + live dynamic pipelines. Adds `render_object_mask`, the rigid-transform helpers, ESAM lazy load, a `means`-gradient zeroing hook (enforces invariant #4), and the dynamic-phase config knobs. *(The scene-optimization machinery — `enable_scene_optimization`, `scene_opt_*`, `scene_opt_active_mask` — was purged in `06d2c47`; the dynamic phase does no per-step gradient descent.)*

### Data + persistence

* **Datamanager**: [`DynamicGSDataManager`](dynamic_gs/dynamic_gs_datamanager.py) wraps two `FullImageDatamanager`s (`static_scene/`, `dynamic_scene/`). Live mode pulls frames from SHM via [`LiveShmSubscriber`](dynamic_gs/utils/live_shm_reader.py).
* **Persistence**: [`dynamic_gs/persistence/`](dynamic_gs/persistence/) — `save_post_fusion_state` writes `gauss_params.*` + all persistent buffers; `load_post_fusion_state` warm-restarts the dynamic pipelines from a static-gs snapshot.
* **Fusion phase 0**: [`dynamic_gs/fusion/phase0.py`](dynamic_gs/fusion/phase0.py) — `run_phase0a_sam3_and_sam3d` + `run_phase0b_fusion`.

### Static-phase seed pipeline (2026-06-01)

The PLY at `<data>/static_scene/depth_camera_init_points.ply` is what Splatfacto inits Gaussians from. Today:

1. **During capture**, [`utils/fusion_runner.py`](dynamic_gs/utils/fusion_runner.py) runs an `OnlineFusion` worker thread that watches `transforms.json` and integrates each new keyframe (`add_frame`). GPU TSDF + ICP at 1.5 mm voxel, ~16 ms/frame at 800×800.
2. **On capture stop**, `stop_and_finalize()` drains the queue, calls `finalize()`, and writes the PLY (~0.6 s).
3. **Optional adaptive downsample**: [`scripts/adaptive_downsample.py`](scripts/adaptive_downsample.py) keeps the <1 m near-zone at full density, voxel-downsamples the rest to 5 mm. Not yet auto-wired into the bootstrap flow; run manually if seed size matters.

### Utility modules ([`dynamic_gs/utils/`](dynamic_gs/utils/))

| Module | Role |
|---|---|
| `online_fusion.py` | `OnlineFusion` class: TSDF + ICP per frame, GPU. Drives the init seed. |
| `fusion_runner.py` | `ConcurrentFusionRunner` — watcher + worker for streaming fusion. |
| `rgbd_fusion_init.py` | Legacy offline post-pass refinement; only used as a fallback. |
| `xfeat_motion.py` | The XFeat-only dynamic tracker (5-tracker dispatch purged 2026-05-26). |
| `tracker_common.py` | Kabsch + RANSAC helpers + `MotionEstimate` dataclass shared by the tracker. |
| `live_ros_publisher.py` | The ROS publisher run inside `dynamic_gs_ros` (subprocess). Owns the SHM. |
| `live_shm_reader.py` | Reader-side wrapper: spawns the publisher, polls SHM, gives `peek_latest()`. |
| `live_session.py` | The bootstrap-time interactive capture flow (SAM3 retry loop, SAM3D, fusion). |
| `keyframe_filter.py` | ORB-SLAM-style greedy 2 cm/20° pose dedup; shared between recorded + live. |
| `sam3_segmentation.py`, `sam3d.py`, `sam3d_fusion.py` | SAM3 + Fast-SAM3D subprocess wrappers + Phase-0b registration & fusion (NDP default; CPD/TEASER++ fallbacks). |
| `ndp_register.py`, `ndp/` | Vendored NDP non-rigid deformation (`deform_source_to_target`) — the default Phase-0b backend. Pure-torch, no pytorch3d, in-process GPU. |
| `esam.py` | ESAM interactive object-mask query (D0 bootstrap). |
| ~~`optim_pool.py`~~ | **REMOVED** — dynamic-phase per-step optimization was dropped (the dynamic phase is a pure tracker+FF runtime; see invariant #4). File no longer exists; `OptimPool`/`optim_pool_*` are dead references. |
| `active_mask.py` | `build_change_mask`, `select_top_n_components_filtered`, projection helpers. |
| `viser_direct.py` | Standalone viser server pushed by the tracker; bypasses ns-viewer state machine. |
| `rgbd_decode.py` | Feedforward Mode A/B: direct RGB-D back-projection into frozen Gaussians. |
| `anysplat_decode.py` | Feedforward via the AnySplat persistent subprocess. |
| `depth_loss.py`, `rigid_regularization.py`, `no_refine_strategy.py` | Smaller pieces used by the model. |

`utils/_purged/` holds dead trackers (cotracker / tapir / tapnext / klt) and an old live-subscriber for reference only — never imported by the runtime.

### Three-phase training (overview)

**Phase 0 (Static)** — `static-gs`. Splatfacto fit on the SfM/TSDF seed for `static_num_steps` (default 500, `STATIC_NUM_STEPS` in `dynamic_gs_config.py`). Densification OFF, means LR = 0, camera-pose optimizer = `off` (NOT `SO3xR3` — see invariant #2). At end: Phase 0a SAM3 + Fast-SAM3D, then Phase 0b NDP non-rigid registration (default; CPD/TEASER++ selectable) + insertion + post-fusion cull (proximity de-dup + in-front occlusion). Writes `post_fusion_state.pt`.

**Phase 1 (Dynamic)** — `dynamic-gs` / `dynamic-gs-live`. Warm-load from `.pt`. Per tracker tick: XFeat motion estimation → `apply_rigid_object_transform_from_reference` → viser-direct push. Optionally feedforward-decode CDN regions (rgbd or anysplat) into the scene.

The legacy "Phase 0 split" (object insertion AFTER static training) is preserved by `static-gs`'s `_finalize_static_training` AFTER_TRAIN callback. The dynamic pipelines load the post-fusion snapshot directly and skip retraining.

### Per-object identity buffers

| Buffer | Type | Set by | Purpose |
|---|---|---|---|
| `object_instance_ids` | long (N,1) | Phase 0b fusion | Multi-object identity, 1..K |
| `object_flags` | float (N,1) | D0 selection | Active dynamic object (0/1) |
| `sam3d_init_target_flags` | float (N,1) | Phase 0b | Marks SAM3D-inserted Gaussians |
| `inserted_flags` | float (N,1) | rgbd_decode | Feedforward Mode B inserts |

`object_instance_ids` only carries IDs for Fast-SAM3D-inserted Gaussians today. Future #1 (top of roadmap) will give every TSDF-seeded Gaussian a real ID via per-frame SAM2 propagation.

### Optimizer groups + LRs

Standard Splatfacto 7 groups: `means`, `features_dc`, `features_rest`, `opacities`, `scales`, `quats`, `camera_opt`.

* Static phase (`static-gs`): all groups active. `means` LR = 1.6e-4 in the seed config but EFFECTIVELY 0 because the means optimizer's only update is via densification (which is OFF).
* Dynamic phase (`dynamic-gs` + `dynamic-gs-live`): all LRs zeroed (`_ZERO_LR_OPTIMIZERS` in `dynamic_gs_config.py`). The trainer's optimizer step is a no-op; mutations come from `apply_rigid_object_transform_from_reference` and feedforward inserts.

## Data Format

```
<data_dir>/
├── static_scene/
│   ├── rgb/                          (BGR PNG)
│   ├── depth/                        (uint16 mm TIFF)
│   ├── masks/                        (uint8 robot-exclusion mask)
│   ├── transforms.json               (Nerfstudio-formatted)
│   ├── depth_camera_init_points.ply  (TSDF-fused seed)
│   └── post_fusion_state.pt          (warm-cache after static-gs)
├── dynamic_scene/
│   ├── rgb/  depth/  masks/  transforms.json
│   ├── initialization_debug/         (SAM3 anchor + debug images)
│   └── initialization_artifacts/     (per-object SAM3D PLY + pose JSON)
└── timing_report*.txt
```

Dataparser settings: `orientation_method="none"`, `center_method="none"`, `auto_scale_poses=False` — poses are kept in metric units, not recentered.

## ROS Data Collection

The live publisher subprocess is auto-spawned by `LiveShmSubscriber`. It runs the URDF FK + frame sync + atomic frame writes; the reader-side process never imports rospy.

Required: `dynaarm_with_gripper_for_gazebo_only_no_wrist_collision.urdf` must define `camera_pose_link` as a `<link>` and load the `libactive_camera_arm_link_pose_publisher.so` Gazebo plugin (publishes `/dynaarm_arm/dynaarm_arm/camera1/gazebo_pose`). See the historic 2026-05-04 version in `~/.config/Code/User/History/-45f4ea38/KHwu.urdf` for the canonical content.

`urdf/dynamic_gaussian_splat/` and `worlds/dynamic_gaussian_splat/` symlinks are required under the catkin workspace — the publisher expects them at those paths.

## Third-Party Dependencies (`third_party/`)

* **`sam-3d-objects/`** — SAM3D model for single-view 3D object reconstruction. Multi-object path via `utils/sam3d.run_sam3d_multi_object`.
* **SAM3** ([facebookresearch/sam3](https://github.com/facebookresearch/sam3)) — text-prompted segmentation. Invoked via `conda run -n sam3_dynamic_gs python`.
* **ESAM** — interactive segmentation, D0 bootstrap.
* **AnySplat** — feedforward decoder, persistent subprocess in `anysplat_dynamic_gs`.
* **NDP (Neural Deformation Pyramid)** — vendored in `utils/ndp/` (`nets.py` + `rigid_body.py`); the default non-rigid Phase-0b backend via `utils/ndp_register.py`. Upstream: github.com/rabbityl/DeformationPyramid (no-learned path, no checkpoint).
* **PROBREG / Open3D** — `utils/sam3d_fusion.py` CPD fallback; TEASER++ the rigid alternative (both still selectable, no longer the default).
* **XFeat / LighterGlue** — vendored under `dynamic_gs/utils/xfeat_motion.py`'s dependencies.
* **FoundationPose** — `third_party/FoundationPose/` kept on disk but no longer wired into the runtime (XFeat purge 2026-05-26).

## Background + Camera Conventions

* Camera poses are OpenGL c2w in `transforms.json`. `OnlineFusion` and FoundationPose helpers convert to OpenCV internally via `diag(1, -1, -1, 1)`.
* Depth is uint16 millimetres on disk (`depth_unit_scale_factor = 1e-3` in the dataparser). The publisher converts to float32 metres at the SHM boundary.
* Simulator background: Gazebo sky color `(0.86, 0.92, 1.0)` is set as the model's render background — not the dataparser `mask_color`. Defined in `StaticGSModel.populate_modules()` and `DynamicGSModel.populate_modules()` and as the viewer default in `nerfstudio/viewer/control_panel.py`.

## Open Roadmap

(Detailed in [`memory/project_multi_object_roadmap.md`](../../../home/mrc-cuhk/.claude/projects/-home-mrc-cuhk-Documents-dynamic-gaussian-splat-scripts/memory/project_multi_object_roadmap.md))

1. **Per-Gaussian SAM IDs** — port from `experiments/icp_fusion_mvp/`. Every Gaussian gets a real instance ID at the source, not just Fast-SAM3D inserts.
2. **Auto-pick by gripper TCP** — D0 picker uses closest-point to gripper, not 3D centroid to camera.
3. **Multi-object Fast-SAM3D** — `sam3_prompt_text` becomes a `list[str]`; multi-mask insertion with distinct instance IDs.
4. **Multi-object switching tracker** — track whichever instance is currently moving; swap on detected motion change.

Also pending: sub-0.05 opacity Gaussian purge at end of static phase (~26 % reduction, no visible change); Phase 0b CPD vs TEASER++ comparison.

## Timing Reference

Per-substep numbers live in `<data_dir>/timing_report.txt` after each run. Don't trust any number quoted here without verifying against a recent report — historical estimates have been wildly off (see memory entry [`feedback_no_timing_guesses.md`](../../../home/mrc-cuhk/.claude/projects/-home-mrc-cuhk-Documents-dynamic-gaussian-splat-scripts/memory/feedback_no_timing_guesses.md)).

Most recent measurements (validate_run_1, 800×800, 71 frames, 2026-06-01):
* Online fusion (GPU): mean 16 ms/frame, p90 21 ms — see `scripts/bench_gpu_fusion.py`.
* Static training (Splatfacto, 1000 steps, no densify): under 20 s.
* XFeat tick: 17–30 Hz steady, ~21 ms/tick at `xfeat_top_k=300`.
