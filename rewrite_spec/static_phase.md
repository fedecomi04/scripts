# Static phase — architecture & requirements (dynamic_gs2 rewrite)

Status: **BUILT** (2026-06-20). The §4 module layout is implemented under `dynamic_gs2/`
(`static_pipeline.py` orchestrator + `static_seed/segment/sam3d/capture/fuse.py` stages +
`model_loader.py` prewarm registry + a static-phase report in `timing.py`). Entry:
`dynamic_gs2/static.sh <data_dir> [prompt] [--live]` (or `-m dynamic_gs2.static_pipeline`).

**SUPERSEDED by the NATIVE rewrite (2026-06-20, later same day).** The first cut wrapped the old
`ns-train static-gs` as a subprocess + converted its `.pt`. That has been REPLACED by a fully
NATIVE in-process static phase (operator request: "implement a fully new version"). The subprocess
+ `convert_old_state_to_dgs2` are DELETED. Now:

* **NATIVE train** (`static_train.py`): a 500-step Splatfacto fit driven directly on `SceneModel`
  (build a nerfstudio `Optimizers` from the model param groups, means LR=0, fp16 autocast,
  num_downscales=1/resolution_schedule=100, NoRefineStrategy, during-train scale-reset, early-stop).
* **NATIVE Phase-0b** (`static_phase0b.py`): the 8 old StaticGSModel methods reimplemented on
  GaussianSet/SceneModel — the two projected-gaussian queries read gsplat's `info` via the NEW
  `SceneModel.get_outputs_with_info` + the PROVEN `extract_projected_centers_and_radii`; spacing/
  insert/instance-id-flag run on the SSOT; register + cull + back-project + NDP + SAM3D-load are
  WRAPPED unchanged. `static_fuse.train_fuse_and_export` = seed PLY → SceneModel → train → opacity
  purge → Phase-0b → `save_warm_cache`.

**VALIDATED (2026-06-20):**
* **Per-method unit A/B** (old StaticGSModel vs native, SAME render+mask): existing-object-subset,
  both slab queries, estimate_spacing all **bit-identical (Jaccard 1.0, exact spacing)**.
* **Isolation full-Phase-0b A/B** (SAME old scene + SAME SAM3D PLY): native insert count **41804 ==
  old 41804 exactly**, fused-object centroid within **3.4 mm** (residual = NDP seeded-stochastic).
* **Adversarial multi-agent review** (4 lenses × verify): 13/13 flagged risks HANDLED; found + fixed
  ONE real bug — Splatfacto's `background_color="random"` made the native train composite against a
  RANDOM background during training (Invariant #6); now `_get_background_color` is overridden to the
  fixed Gazebo sky in train AND eval (matches the old StaticGSModel override). Also wired the dead
  scale-clamp config (max scale 10.2cm → 5.0cm) + a freelist=False guard in Phase-0b.
* **Full from-scratch native** `--mode recorded`: fresh FastSAM+SAM3D + native train + native Phase-0b
  → valid `dynamic_gs2.v1` cache (N=372290, 30600 object / 30302 inserted, scales bounded ≤5cm) that
  the dynamic phase warm-loads + tracks. Post-SAM3D dead time **19.8 s** (per-§1-stage timing now
  visible in-process, vs the old wrap's opaque 236s bar). Live red-box UI + live SHM = operator-only.

The new dynamic_gs2 modules own ORCHESTRATION / timing / red-box UI / SHM capture / segmentation
folder / prewarm AND now the train + Phase-0b fusion natively; only the pure register/cull/backproject/
NDP/SAM3D math is WRAPPED. Timing → `timing_report_static_dgs2.txt`.

This spec defines the dynamic_gs2-style static phase: a single SHM-fed black box, parallel to
the existing dynamic_gs2 dynamic phase, replacing the old two-process (`live_session` +
`ns-train static-gs`) path.

Goal: **minimise the user's PERCEIVED dead time** by overlapping every heavy load/inference
with the operator's continued motion, and unify recorded + live behind ONE SHM source
(same contract as the dynamic phase — `frame.py` / `shm_channel.py` / `adapters_source.py`).

---

## 0. Hard constraints (measured / code-verified — do NOT design against these)

| Fact | Source | Implication for the schedule |
|---|---|---|
| SAM3D infer resident **~13.1 GB sustained** (~6s) | measured 2026-06-20 | Nothing else big (splatfacto/AnySplat) can be GPU-resident *during* SAM3D. |
| Card **16303 MiB**; Gazebo **2.5 GB** concurrent | measured | Budget during SAM3D = 16.3 − 13.1 − 2.5 ≈ **0.7 GB**. Phase-0b blip 13.7 GB → 16.2 w/ Gazebo ≈ OOM edge. |
| GPU TSDF **~21.9 ms/frame mean** (1200p, measured §2a); CPU ~630 ms/frame | §2a / online_fusion.py | 57 frames: GPU seed wall **1.5 s** incl finalize. **TSDF MUST be GPU.** |
| **GPU TSDF needs ~8.1 GB resident** (VoxelBlockGrid, 1200p/2mm) — measured §2a | §2a | CANNOT coexist with SAM3D (7-12GB) on GPU. SEQUENCE them. But seed is 1.5s → run serially after SAM3D unloads. |
| `add_frame` already does **ICP → then integrate** at the refined pose | online_fusion.py `add_frame` L280–305 | ICP-before-TSDF is correct as-is. Reuse verbatim. |
| FastSAM ~0.85 GB; SAM3D ~7.3 GB trimmed (peak 11.7) ; splatfacto train ~2.5 GB; AnySplat ~3.5 GB; NDP ~0 (no ckpt, ~4ms construct) | CLAUDE.md VRAM table | AnySplat (3.5) + splatfacto train (2.5) + Gazebo (2.5) = 8.5 GB → CAN co-reside. SAM3D + anything big cannot. |
| SAM3D load: operator says **≤30s** (CLAUDE.md's 61s is @1200p, likely stale) | TBD | **MEASURE cleanly before finalizing the schedule** (load vs infer split). |

**The one irreducible serial cost:** SAM3D-infer (~13 GB, alone) → unload → splatfacto. They
cannot overlap. The win is not removing it but **hiding it under continued operator motion.**

---

## 1. FINAL SCHEDULE (canonical — measured 2026-06-20; ‖ = parallel)

**SWEEP** (operator moving; UI shows live frames + 500px red box):
- ‖ record frames → disk
- ‖ **ICP live** per frame → keep refined poses in a CPU list  *(28 ms/frame, 3.4 GB — coexists w/ FastSAM+Gazebo)*
- ‖ FastSAM load (~0.85 GB)  `[load time UNMEASURED]`
- ‖ SAM3D **build on GPU (16 s) → park to CPU (0.8 s)**
- ‖ XFeat + LighterGlue → **load to CPU** (tiny, in-process; for dynamic). NOT AnySplat here — its
  3.5 GB worker would collide with SAM3D's 12 GB GPU build; spawn it after SAM3D (see below).

**RED-BOX TRIGGER** (object fills box; KEEP recording):
- snapshot anchor (rgb+depth+pose+intr), immutable
- FastSAM segment(anchor) → mask → unload → write `segmentation/` folder + `overlay.png`
- **pause ICP** (just keep the pose list — free; no grid to evict)
- SAM3D **wake CPU→GPU (2.3 s) → infer (9.3 s) → unload/park**  ← operator keeps moving → hidden

**AFTER SAM3D** (GPU free):
- stop_recording (UI keeps showing frames)
- ‖ **TSDF integrate-only** (live poses, ICP already done) → finalize → seed PLY  *(~1.3 s, 8 GB)*
- ‖ **splatfacto LOAD** (instantiate + →GPU)  `[load UNMEASURED]`  *(8 + 2.5 + 2.5 Gazebo ≈ 13 GB, fits)*
- ↳ **splatfacto TRAIN** 500 steps  *(~17 s, 2.5 GB; starts after finalize ~0.3 s; 40 ms/iter to investigate)*
  - ‖ **NDP** (~0 GB): reproject mask→depth target + register SAM3D PLY  *during training*
  - ‖ **spawn AnySplat worker** (~17 s, 3.5 GB) — overlaps training, hidden; GPU-resident by hand-off.
    Fits: AnySplat 3.5 + splatfacto-train 2.5 + Gazebo 2.5 = 8.5 GB. (Here, NOT during sweep — SAM3D's
    12 GB build would collide. This is the free slot.)

**END / HAND-OFF to dynamic:**
- opacity purge → fuse NDP object into scene → write `static_state.pt`
- **wake the dynamic models to GPU** (they were CPU-loaded during the sweep): AnySplat worker
  `prewarm()`→`wait_ready()` (subprocess resident; no CPU-park API — it's spawned-or-not), XFeat +
  LighterGlue `.cuda()`. Done while/after `static_state.pt` writes so the dynamic loop starts warm.

**Post-SAM3D dead time** ≈ TSDF 1.3 s + train 17 s + fuse 4 s ≈ **~22 s** (all else hidden under the sweep).
NOTE: AnySplat has NO CPU-park/GPU-wake — `AnysplatHandle` either has a spawned worker or not
(`prewarm`/`wait_ready`/`close`). So "load to CPU during sweep" for AnySplat = spawn the worker early
so its ~17s load overlaps the sweep; it stays GPU-resident once up (it's the persistent subprocess).
**Only hard ordering:** train-after-finalize (seed feeds init, ~0.3 s wait) and NDP-after-SAM3D-PLY (done).
**Open numbers:** splatfacto load time; the 40 ms/iter cause (means LR=0 yet slow); VRAM peak of the TSDF‖splatfacto-load overlap.

## 1-OLD. (superseded by §1 above) earlier dead-time-minimal sketch

```
PHASE                          GPU resident            hidden under...
─────────────────────────────────────────────────────────────────────────────
launch + SHM attach            —                       (instant)
sweep begins (record frames)   —                       operator sweeping
  ├─ FastSAM preload (0.85GB)  FastSAM                  ← sweep
  └─ SAM3D preload (7.3GB)     FastSAM + SAM3D          ← sweep   [fits: 8.2GB]
  └─ (NO GPU-TSDF during sweep — it needs ~8.1GB, collides w/ SAM3D wake; §2a)
     frames just recorded to disk; seed built later (only 1.5s, §2a)
─────────────────────────────────────────────────────────────────────────────
RED-BOX TRIGGER (object fills box) — user presses / clicks; KEEP RECORDING
  ├─ snapshot ANCHOR frame (rgb+depth+pose+intr) — frozen, immutable
  ├─ FastSAM segment(anchor, prompt) → masks → UNLOAD FastSAM (free 0.85GB)
  ├─ SAM3D wake CPU→GPU (~2.3s) → infer(anchor) → object PLY  ← user STILL MOVING (hidden!)
  └─    (SAM3D alone ~13GB; nothing else big resident here)
─────────────────────────────────────────────────────────────────────────────
SAM3D done → UNLOAD/park SAM3D (free GPU)  ← GPU now wide open
  ├─ stop_recording (but UI KEEPS showing live frames — perceived "still working")
  ├─ GPU-TSDF seed build (all frames, ~1.5s, ~8.1GB; SAM3D no longer resident)
  ├─ splatfacto LOAD + TRAIN (~2.5GB, ~17s)
  │     └─ AnySplat preload  (3.5GB)  ‖ in parallel  [2.5+3.5+2.5 Gazebo = 8.5GB ✓]
  │     └─ XFeat+LighterGlue preload (tiny) ‖ in parallel
  │     └─ NDP construct (~0, no ckpt) ‖
  └─ end of train → opacity purge → Phase-0b: reproject mask→depth target,
        NDP-register SAM3D PLY onto it, insert into trained scene
─────────────────────────────────────────────────────────────────────────────
write static_state.pt → HAND OFF to dynamic phase (AnySplat/XFeat already warm)
```

**Perceived dead time** ≈ (SAM3D infer not yet hidden) + (splatfacto train + Phase-0b after
the user is told "done moving"). If the user keeps sweeping until SAM3D returns, the only
visible wait is splatfacto-train (~17s) + fuse (~4s) ≈ **~20s**, with AnySplat/XFeat already
warm so go-live is instant.

---

## 2a. GPU TSDF+ICP cost — MEASURED 2026-06-20 (corrects the "~0.1 GB TSDF" guess)

Real 57 static_scene frames @1920×1200, OnlineFusion GPU path (ICP→integrate, 2mm voxel):
- **per-frame add_frame: mean 21.9 ms / p50 17.8 / p90 27.0** (max 134 ms = first-frame warmup)
- **finalize: 288 ms**; **total seed-build wall: 1.5 s** → 4.56M-pt seed
- **VRAM: ctor +626 MiB; ramps to a PLATEAU of ~8910 MiB used = +8.1 GB over baseline, then flat.**

**CORRECTION to §1/§5: GPU-TSDF needs ~8.1 GB, NOT ~0.1 GB.** The VoxelBlockGrid hashmap at
1200p/2mm grows to ~8.9 GB resident. CONSEQUENCES:
- **GPU-TSDF CANNOT coexist with SAM3D on GPU** (8.1 + 7–12 + 2.5 Gazebo ≫ 16.3). This is exactly
  why production sets `DGS_LIVE_DEFER_TSDF=1` — now confirmed with a number.
- **But the seed build is only 1.5 s total** → it does NOT need to overlap anything. Run it SERIALLY
  in the gap **after SAM3D unloads, before splatfacto loads**, on the freed GPU. ~1.5 s dead time, no
  collision. (CPU TSDF would be ~630 ms/frame ≈ 36 s — never use it.)

**Revised sweep policy:** SAM3D parked on CPU during the sweep (0 GPU); frames recorded to disk;
**no GPU-TSDF during the sweep** (collides with SAM3D wake at trigger AND it's only 1.5 s anyway).
After SAM3D infer+unload → 1.5 s GPU-TSDF seed → splatfacto.

## 2c. Staging-strategy benchmark (SSD→ready) + residency cycle — measured 2026-06-20

Goal restated (operator): NOT to speed up SAM3D inference, but to keep VRAM FREE between uses so
ICP/TSDF/other models run, waking a model to GPU only for its inference.

Benchmark (env load excluded; build = DINO+graph+ckpt; "move" = →GPU; VRAM resident after):
| strategy | build | move | total | VRAM resident |
|---|---|---|---|---|
| A baseline `device=cuda` | 15.8s | — | **15.8s** | 12.2 GB |
| B CPU-build → `.cuda()` | 27.3s | 0.8s | 28.1s | 7.2 GB |
| C CPU-build → pinned `.cuda()` | 26.9s | 2.3s | 29.1s | 7.2 GB |
| D cuda-build + fp16 trim | 15.7s + 0.3 | — | **16.0s** | **7.6 GB** |
| E CPU-build + trim + pinned | 27.1s | 1.6s | 29.5s | **5.3 GB** |

**KEY: CPU-build is ~11s SLOWER (27s vs 16s)** — DINO+graph build runs faster with CUDA present.
So "force the build onto CPU" LOSES. The right pattern is **build on GPU once (16s), then PARK on
CPU** between uses:

Residency cycle (build-on-GPU → `.to(cpu)` → `.to(cuda)` for inference → back):
  build(GPU) 16.5s → **PARK→CPU 0.78s** → **WAKE→GPU(pinned) 2.3s** → PARK→CPU 0.78s
**⚠ naive park freed only 12.2→6.5 GB, NOT to 0** — ~6.5 GB stays resident because DINO/MoGe
submodels + raw cuda buffers aren't in the nn.Module set a simple mover walks. To free the FULL
~12 GB you must move EVERY submodule (reuse `apply_sam3d_gaussian_trim`'s complete module list) OR
run SAM3D in a SUBPROCESS that you KILL after inference (100% free, guaranteed; cost = 16s rebuild/call).

**Decision per phase:**
- **Static (SAM3D used ONCE):** subprocess-per-call is fine — pay 16s build once, process death 100%-frees
  VRAM for splatfacto. Park-on-CPU buys nothing for a single use.
- **Dynamic (AnySplat/XFeat reused every FF cycle):** park-on-CPU / persistent-worker wins — wake ~2-3s,
  amortized over many calls. (AnySplat already a persistent worker.)
- **The disk→CPU read (4.6s) is only worth prewarming if you'll wake-from-CPU repeatedly.** For a single
  SAM3D use it's dominated by the 16s build, which you pay regardless.

## 2b. SAM3D load anatomy — measured 2026-06-20 (the prewarm budget, honest)

SAM3D end-to-end **43.2s** on the real screwdriver mask/img/depth (1200p anchor → SAM3D infers at 324×518):
`import+config 7.1s | model_load (Inference ctor) 22.3s | infer 9.3s | + subprocess spawn`. VRAM peak **12.8 GB**.
Output: 86,336-pt PLY (correct screwdriver). Decomposing the 22.3s ctor:

| sub-component of the 22.3s ctor | time | hideable under the sweep? |
|---|---|---|
| disk→CPU read of the 2 big ckpts (ss_gen 6.7GB + slat_gen 4.9GB ≈ 11.9GB) | **4.6 s** | ✅ prewarm to CPU |
| CPU→GPU move (naive 1.6s / **pinned 0.54s**) | **~0.5–1.6 s** | at-trigger (pin during sweep) |
| **the rest: DINO load (×4) + model build + CUDA first-kernel init** | **~16 s** | ⚠ MOSTLY hideable ONLY if the whole `Inference` object is constructed on CPU during the sweep; the CUDA-context/first-kernel init part is irreducible-at-trigger (must touch GPU). |

**Honest correction to the earlier optimism:** prewarming just the *checkpoint files* to CPU saves only **~5 s** of the 22.3 s. The big ~16 s is DINO+build+CUDA-init, NOT file I/O. To hide it you must prewarm the **entire `Inference` model on CPU** during the sweep (needs SAM3D to support CPU construction — UNVERIFIED; the ctor currently builds on cuda). If it can't build on CPU, the floor at the trigger is ~16–22 s of ctor that only continued operator motion can mask. **infer (9.3s) hides cleanly** under post-trigger motion regardless.

## 2. RESOLVED — CPU-stage the weights; GPU-TSDF runs collision-free (measured 2026-06-20)

**Decision: preload SAM3D (and every heavy model) weights to CPU RAM during the sweep, and
`.cuda()` them on demand at the trigger.** This keeps the sweep-time VRAM free for GPU-TSDF, so
there is NO sweep-time collision to design around. The host→device copy is cheap:

| blob (CPU→GPU) | naive `.cuda()` (pageable) | pinned `.cuda(non_blocking)` |
|---|---|---|
| 7.3 GB (trimmed SAM3D) | **1.05 s** (6.9 GB/s) | **0.35 s** (20.7 GB/s) |
| 11 GB (fp32 SAM3D)     | **1.97 s** (5.6 GB/s) | **0.52 s** (21 GB/s) |

`pin_memory()` itself costs ~1.5–4.6 s but runs DURING the sweep (free time), so only the
~0.35–0.5 s copy lands on the critical path at the trigger — hidden under continued motion.
(Measured with a synthetic 400-tensor state_dict; the TRANSFER is real, but the real SAM3D's
first-forward CUDA warmup is separate and still paid once on `.cuda()` — measure end-to-end in §7.)

**Consequence:** the generalized residency model = weights live on CPU until needed, `.cuda()`
(pinned) right before compute, back to CPU / freed after. Turns "resident the whole time" into
"resident only while computing" — the whole game on a 16 GB card shared with Gazebo. This is the
basis for the prewarm registry (§5): `prewarm()` = load-to-CPU (+pin), `activate()` = `.cuda()`,
`release()` = free VRAM (weights stay on CPU for re-activation).

---

## 3. The segmentation/SAM3D artifact folder (user requirement: "no mistakes can happen")

A SINGLE dedicated, self-describing folder per run, absolute-path-free internally so a
copied/renamed dataset never breaks (the exact bug that killed the from-scratch re-run:
cached SAM3 JSON held `replay_20260612` absolute paths → FileNotFoundError).

```
<data_dir>/static_scene/segmentation/        ← THE folder (new, canonical)
  anchor/
    rgb.png            (the EXACT frame SAM3D saw — gripper-blacked as segmented)
    depth.tiff         (uint16 mm, same frame)
    pose.json          (c2w 4x4, OpenGL)
    intrinsics.json    (fx fy cx cy w h)
  masks/
    obj_00.png         (uint8 mask, per object)
    obj_00_overlay.png (mask ON anchor rgb — the ONE image to show the user to validate)
    ...
  objects/
    obj_00_sam3d.ply   (raw SAM3D output, object frame)
    obj_00_pose.json   (SAM3D→world rigid init)
  manifest.json        (prompt, N objects, scores, bboxes; RELATIVE paths only)
```

Rules:
- **All paths inside `manifest.json` are RELATIVE to the segmentation/ folder.** Reloading
  resolves against the folder's own location → copy/rename safe by construction.
- The overlay (`masks/obj_NN_overlay.png`) is the validation surface: to show the user the
  mask, the UI just loads that one PNG. No re-render needed.
- Folder is **wiped + rewritten per run** (no stale reuse across datasets). A warm-start that
  reuses cached SAM3D is opt-in and validates the folder belongs to THIS dataset (fingerprint).

---

## 4. Module layout — "smart way to not mix all files together"

Mirror the dynamic phase's separation of concerns. The static phase becomes a small set of
single-responsibility modules under `dynamic_gs2/`, NOT one god-file:

```
dynamic_gs2/
  static_pipeline.py     orchestrator: the §1 schedule + the ONE model_lock. Owns the
                         state machine (SWEEP → TRIGGERED → SAM3D → TRAIN → FUSE → DONE).
  static_seed.py         GPU TSDF seed: wraps the proven online_fusion.OnlineFusion
                         (ICP→integrate). Consumes the SAME SHM frames as the tracker.
  static_segment.py      FastSAM/SAM3 segment → the segmentation/ folder (§3). Owns the
                         anchor snapshot + mask + overlay write. Backend-agnostic.
  static_sam3d.py        SAM3D generate (subprocess wrap) → objects/*.ply. CPU/GPU residency
                         policy (§2) lives here.
  static_fuse.py         Phase-0b: mask→depth reproject target + NDP register + insert.
                         Wraps ndp_register + the proven fusion math.
  model_loader.py        ★ the prewarm registry (§5) — ALL "load model X" lives here, ONE place.
  (reuse) gaussian_set.py / scene_model.py / static_persist.py  — already exist, unchanged.
  (reuse) adapters_source.py / frame.py / shm_channel.py — the ONE SHM ingest, shared w/ dynamic.
```

Each `static_*.py` is a pure stage: takes immutable inputs, returns artifacts, touches the
GaussianSet only via its locked surgery API. The orchestrator sequences them per the schedule.

---

## 5. Prewarm registry — "preload XFeat + LighterGlue + AnySplat without mixing files"

ONE `model_loader.py` with a uniform async-prewarm interface, so every "load model X" is a
registry entry, not scattered `import`s. Pattern (already proven by `AnysplatHandle.prewarm()`):

```python
class ModelHandle(Protocol):
    def prewarm(self) -> None: ...      # load weights to CPU RAM (+pin) on a bg thread, non-blocking
    def activate(self) -> None: ...     # .cuda() the weights (pinned ~0.35-0.5s, §2); call before first use
    def release(self) -> None: ...      # free VRAM; weights stay on CPU for cheap re-activate
    def wait_ready(self) -> None: ...   # block until prewarm (CPU load) done

REGISTRY = {
  "fastsam":  FastSamHandle,    # load on sweep, unload after segment
  "sam3d":    Sam3dHandle,      # load on sweep (or CPU, §2), unload after infer
  "splatfacto": SceneTrainHandle,
  "anysplat": AnysplatHandle,   # preload under splatfacto train (for dynamic)
  "xfeat":    XFeatHandle,      # preload under splatfacto train (for dynamic)  ← tiny
}
```

The orchestrator calls `prewarm()` at the schedule point, `wait_ready()` at first use,
`unload()` at the handoff. XFeat+LighterGlue+AnySplat all `prewarm()` together at "SAM3D done"
so they load under splatfacto training and the dynamic phase starts warm. Keeps load policy in
ONE file; the stage modules never `import` a model directly.

---

## 6. Warm-start contract — "run dynamic-only by prewarming with static data"

**What the dynamic phase needs to warm-start = EXACTLY ONE FILE: `static_scene/static_state.pt`.**
(Code-verified: `dynamic_gs2/static_persist.build_loaded_scene` reads gauss_params + the 4
identity buffers + num_points from the .pt and seeds the inner SplatfactoModel from THAT. It
does NOT read the seed PLY, rgb/depth/masks, or transforms.json at warm-start.)

So the two run modes:

| Mode | Needs | Produces |
|---|---|---|
| **FULL (static+dynamic)** | live SHM stream (or recorded SHM) | static_state.pt → then dynamic |
| **DYNAMIC-ONLY (warm-start)** | `static_scene/static_state.pt` ONLY | live tracking + FF |

`static_state.pt` payload (the contract — keep stable):
```
{ "model_state_dict": { "gauss_params.means|features_dc|features_rest|opacities|scales|quats",
                        "object_flags","object_instance_ids","inserted_flags","gauss_uid" },
  "num_points": int, "config_fingerprint": str, "layout": "dynamic_gs2.v1" }
```

**Answer to "is it the same data structure as what would be generated?"** — YES for warm-start:
the dynamic-only path consumes the identical `static_state.pt` the full static phase emits. You
do NOT need the rgb/depth/masks/transforms/seed-PLY/segmentation folder to GO LIVE — those are
*inputs that produced* the .pt, not warm-start inputs. Keep them on disk only if you want to
RE-FUSE or re-train; for pure dynamic-only, the single .pt is sufficient (and is exactly what
today's validated dynamic_gs2 already loads).

Recommended on-disk layout (full = superset; dynamic-only = just the .pt):
```
<data_dir>/static_scene/
  static_state.pt              ← REQUIRED for dynamic-only (the whole contract)
  depth_camera_init_points.ply ← input to FULL only (seed); not needed to go live
  rgb/ depth/ masks/ transforms.json   ← input to FULL only
  segmentation/                ← input/validation for FULL only (§3)
```

---

## 7. Next step BEFORE implementing: clean measurement pass

The schedule's overlap decisions (§2) and the "≤30s SAM3D" claim need real numbers. Measure,
each isolated, no Gazebo, restore originals after:
- SAM3D **load** time vs **infer** time (separately) + resident-while-idle vs peak-while-infer.
- FastSAM load; splatfacto load; AnySplat load; XFeat+LighterGlue load; NDP construct.
- GPU-TSDF per-frame at the real capture resolution (1200p), and whether it co-resides with
  SAM3D-preloaded (§2 decision).
Output: a numbers table that fills the schedule's "hidden under…" column for real.

---

## 8. Non-goals / explicitly deferred
- Multi-object tracking selection (roadmap, separate).
- Static-from-scratch SEGMENTATION robustness across all objects (separate; FastSAM+CLIP cliff).
- Touching AnySplat/SAM3D model internals for VRAM (risky; out of scope for the schedule).
