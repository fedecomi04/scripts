# Prompt: new static method `static-gs-preseg` (SAM3→SAM2 per-Gaussian IDs, fuse-before-train)

## What I want, in one sentence

Add a **third static-init method** alongside `static-gs` that produces a **byte-for-byte
equivalent `post_fusion_state.pt`** (a trained Splatfacto static scene whose Gaussians carry
per-point object instance IDs), but builds it via the SAM3→SAM2 per-Gaussian-ID seeding path
and a **reordered flow where object fusion happens BEFORE static training**, not after — and is
heavily optimized for wall-clock by preloading all models at script launch.

This is a *parallel* method, not a replacement. The existing `static-gs` (SAM3 + Fast-SAM3D +
CPD/TEASER fusion AFTER training) stays exactly as-is and remains the default.

---

## Hard output contract (NON-NEGOTIABLE)

The new method's artifact must be a drop-in replacement for the existing one. The dynamic
pipelines (`dynamic-gs`, `dynamic-gs-live`) warm-load from `post_fusion_state.pt` via
`persistence.load_post_fusion_state` and must not know or care which static method produced it.
That means the new method must write a `post_fusion_state.pt` with the **same schema**:

- `gauss_params.{means,features_dc,features_rest,opacities,scales,quats}` — a trained static scene
- `object_instance_ids` (long, (N,1)) — per-Gaussian instance id, 1..K, 0 = background. In this
  method every Gaussian gets a real id at the source (from the SAM2 label transfer), not just
  SAM3D-inserted ones — this is exactly roadmap item #1 ("per-Gaussian SAM IDs").
- `object_flags`, `sam3d_init_target_flags`, `inserted_flags` — must EXIST with the same dtype/shape
  so the saver/loader schema matches. This method has no SAM3D insertion concept at all, so
  `sam3d_init_target_flags` and `inserted_flags` are all-zero, and `object_flags=0` at static end is
  the correct/expected state (D0 selection sets it later in the dynamic pipeline). All-zero is fine —
  the contract is schema-identical, not value-identical, for these three.
- `num_points` matching the gaussian tensors.

Verify the contract by loading the new method's output with the UNMODIFIED
`load_post_fusion_state` and confirming the dynamic pipeline starts without shape/key errors.

The current saver is `persistence/post_fusion_cache.py::save_post_fusion_state(model, cache_path)`
and the model's `load_state_dict` override keys off `object_flags.shape[0]`. Reuse the SAME saver.

---

## The SAM3→SAM2 per-Gaussian-ID approach — DO NOT REDESIGN THIS

The 2D-segmentation + per-Gaussian-ID-transfer method is ALREADY DESIGNED AND VALIDATED by another
chat that has the full context. **Do not reinvent, retune, or "improve" it.** Treat it as a
black-box component you are wiring in. The canonical references:

- Memory note `project_sam2_sam3_boundary_finding.md` (full recipe + the why).
- Throwaway-but-validated implementation: `experiments/sam3_seed_sam2_mvp/amg_merge_propagate.py`
  (SAM3 grouping + SAM2-AMG tight seeds + SAM2-video propagation → merged per-object masks across
  all frames) and `experiments/sam3_seed_sam2_mvp/color_precise_cloud.py` (occlusion-voted 3D
  label transfer: votes each cloud point into every camera, reads the SAM2 label where
  depth-consistent, with the fixed vote rule `ids = where((best>=2) & (best > votes[:,0]), best, 0)`).

Recipe summary you must preserve verbatim (details in the memory note):
1. SAM2-AMG on frame 0 → edge-tight masks (drop blobs > 60% frame so the table stays background).
2. SAM3 on frame 0 (`"objects"`, ct=0.40) used ONLY as grouping targets.
3. Assign each AMG mask to its best SAM3 instance by coverage > 0.8; union per instance; DROP
   AMG masks backed by no SAM3 object (kills table/background fragments).
4. SAM2-video propagate the merged seeds → tight borders, merged IDs across all frames.
5. 3D label transfer via occlusion-voted projection (the fixed vote rule above; tint object
   points, keep TSDF RGB; no erosion, no table-as-object hack).

**Coordinate ONLY on the integration seam, not the internals.** If you find a bug inside the SAM
code, STOP and flag it — do not patch it. The other chat owns that code.

---

## This method's structure: label-the-seed, then train ONCE

Context only (NOT something you modify) — the separate `static-gs` method, for contrast, trains
first and assembles objects last: `__init__` runs SAM3+Fast-SAM3D generation, Nerfstudio trains
Splatfacto on the TSDF seed PLY, then an `AFTER_TRAIN` callback runs registration+insertion. That
ordering is irrelevant to `static-gs-preseg` and you are not touching `static_gs_pipeline.py`.

`static-gs-preseg` has its OWN, simpler structure: **label the seed cloud first, then train once.**
Concretely:

- Build ONE seed file that is a point cloud with, per point: **xyz, rgb, and an object instance id
  (0 = background)**. This is the union of (a) the TSDF/back-projection background cloud and
  (b) the per-object points, each carrying its instance id from the SAM3→SAM2 transfer above.
- Train Splatfacto ONCE on that combined seed. Because every seed point already has an id, the
  trained Gaussians inherit `object_instance_ids` directly from their seed point — no
  post-training registration/insertion pass is needed.
- Save `post_fusion_state.pt` from the trained model with the id buffers populated.

Net effect: this method's full sequence is just segment → build-combined-seed → train → save. The
id assignment lives at the seed, where it's cheap and exact (every point labeled at the source),
with no 3D registration anywhere.

**What "the objects' Gaussians" are in this method:**
The object Gaussians are **the per-object points carved out of the existing TSDF / precise cloud
by the SAM2 masks** — i.e. the existing background cloud, with each point tagged by its object
instance id (0 = background) via the occlusion-voted 3D label transfer in
`experiments/sam3_seed_sam2_mvp/color_precise_cloud.py`. The combined seed is one cloud
(xyz + rgb + id); training it once yields Gaussians that inherit the ids from their seed point.
With ids assigned at the seed there is nothing left to register after training.

**This method is fully independent of SAM3D / Fast-SAM3D / CPD / TEASER.** Those belong to the
SEPARATE, independently-selectable `static-gs` method and have NOTHING to do with `static-gs-preseg`.
`static-gs-preseg` does not use, import, modify, "replace", or "drop" any of that code — it simply
never involved it. When the user picks this method they get the SAM3→SAM2 seeding path; when they
pick `static-gs` they get the SAM3D+registration path. Two parallel, unrelated methods that both
happen to emit a schema-compatible `post_fusion_state.pt`. Do not frame `static-gs-preseg` as a
modification of `static-gs`.

---

## The one real mechanism you must ADD: carry per-point ids from seed → Gaussians

I checked the codebase so you don't have to discover this the hard way. The seed→Gaussian path
is vanilla Splatfacto: Nerfstudio's dataparser loads `load_3D_points=True` and gives the model
ONLY xyz + rgb (`dynamic_gs_datamanager.py:79`). `StaticGSModel.populate_modules()` calls
`super().populate_modules()` then registers `object_instance_ids` (and the other 3 buffers) as
**all-zeros sized to num_points** (`static_gs_model.py:152-170`). Today those ids are written ONLY
later by `insert_object_gaussians` on the SAM3D path. **So Nerfstudio has NO native channel to
carry a per-point instance id from the seed cloud into the Gaussian model — that is the single new
mechanism this method introduces, and the crux of the work.**

You must bridge it. The model already KNOWS how to build Gaussians from a seed and already HAS the
`object_instance_ids` buffer — you only need to populate that buffer from the seed's id column at
init, in seed-point order, so each Gaussian inherits its seed point's id. Concretely (pick the
cleanest; you decide):
- Carry the id array alongside the cloud out-of-band (the seed PLY can hold a per-vertex
  `instance_id` field; read it where the model first materializes `means`, and write
  `object_instance_ids` from it BEFORE training rather than leaving zeros), OR
- Have the new pipeline load the labeled seed itself (not via the stock dataparser), set
  `means`/colors AND `object_instance_ids` together, bypassing the xyz+rgb-only dataparser channel.

Because static-phase means LR = 0 (Invariant #1) and densification is OFF (NoRefineStrategy), the
Gaussian set is FIXED to the seed for the whole run — Gaussian i stays seed point i. That is what
makes the seed→id inheritance exact and stable: no clone/split/prune reorders or invents points, so
the id column never needs re-association after training. Confirm this assumption holds in your
implementation (if anything can change the point count between seed and save, the ids break).

---

## Time optimization: preload all models at script launch

The biggest cost today is cold model loads / subprocess spawns paid lazily mid-run (SAM3 ~9.4s
cold, SAM3D ~22s, per the persistent-worker note in CLAUDE.md invariant #7; SAM2 has its own load).
For this method I want **all required models preloaded as soon as the launch script starts**, so by
the time segmentation is needed the weights are already resident. This must apply to BOTH entry
paths that will use this method:

- Live: `scripts/bootstrap_live.sh` / `resume_live.sh`
- Recorded: whatever the recorded `static-gs-preseg` invocation is.

Use the persistent-worker pattern (`SamWorkerClient` in `dynamic_gs/utils/sam_worker.py`, the
canonical SAM3/SAM3D worker per CLAUDE.md invariant #7) — extend or mirror it so SAM2 is also a
spawn-once, load-on-demand persistent worker. Spawn the workers at `fusion_runner.start()` /
script-launch time, NOT at first-call time. Measure and report the before/after wall-clock so we
know the preload actually paid off (don't guess — see memory `feedback_no_timing_guesses`).

---

## Registration / method scaffolding specifics

- New ns-train method name: `static-gs-preseg` (or propose a better name). Register it via the
  `nerfstudio.method_configs` entry-point in `pyproject.toml`, same mechanism as the existing three.
- New pipeline class, e.g. `StaticPresegPipeline`, modeled on `static_gs_pipeline.py` but with the
  reordered flow. Keep `StaticGSPipeline` untouched.
- Honor the static-phase Design Invariants in CLAUDE.md: means LR = 0 (#1), camera_optimizer off
  (#2), ICP-refined transforms.json (#3), `outputs/` suppressed (#5), Gazebo-sky background (#6).
  These apply to the new method identically — it's still a static Splatfacto fit on metric poses.
- `outputs/` stays empty; all artifacts under `<data_dir>/static_scene/`.

---

## What to deliver

1. A one-paragraph plan stating the exact combined-seed schema (field names/dtypes of the
   xyz+rgb+id cloud) and the precise file/function where the preload workers get spawned in the
   live and recorded entry paths. The object-Gaussian definition is already fixed (per-object
   points carved from the TSDF/precise cloud by SAM2 masks — see above); do not re-litigate it.
   No sign-off gate — proceed straight to implementation unless the seed schema forces a contract
   conflict, in which case flag it.
2. The new method + pipeline + entry-script preload wiring.
3. A verification that the new method's `post_fusion_state.pt` loads cleanly through the
   UNMODIFIED `load_post_fusion_state` and the dynamic pipeline starts.
4. **Run it end-to-end on the dataset we've been using and report timing + VRAM.** Dataset:
   `/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/dynamic_gs_test_2026-03-28_19-49-45_w_background`
   (its `static_scene/` already has rgb/depth/masks/transforms.json + the TSDF seed PLY). Run via
   `conda run -n dynamic_gs ns-train static-gs-preseg --data <that dir>` (mirror the `static-gs`
   invocation in `scripts/bootstrap_live.sh:94`). Report:
   - total wall-clock, broken down: model preload, SAM3→SAM2 segmentation + id transfer,
     combined-seed build, Splatfacto training, save.
   - peak VRAM during the run (`nvidia-smi --query-compute-apps=pid,used_memory --format=csv` polled,
     or `torch.cuda.max_memory_allocated()`).
   - before/after wall-clock for the preload optimization specifically (lazy-load vs preloaded).
   - how many objects / instance ids ended up in `object_instance_ids` and the final Gaussian count.
   MEASURE, don't estimate (per memory `feedback_no_timing_guesses`).

## Autonomy

Implement and test this WITHOUT pausing for sign-off. The design is fully specified above:
method name, the object-Gaussian definition, the seed→id mechanism, the preload requirement, the
output contract, and the test command + dataset are all fixed. Proceed straight through
implement → run on the dataset → report. Only STOP and ask if (a) you find a bug INSIDE the
SAM3/SAM2 code (the other chat owns it — flag, don't patch), or (b) the seed→id mechanism would
force a change to the `post_fusion_state.pt` schema or to `load_post_fusion_state` (that breaks the
output contract — flag it). Everything else: decide and proceed.

Do not commit; leave changes in the working tree for review. Do not touch the SAM3/SAM2 internals —
coordinate with the other chat on that seam only.
