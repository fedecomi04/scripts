# commands.md — quick reference

Run everything from `scripts/`:
```bash
cd /home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts
```

Env pythons (for the diagnostic scripts below):
- main:   `/home/mrc-cuhk/miniconda3/envs/dynamic_gs/bin/python`
- sam3:   `/home/mrc-cuhk/miniconda3/envs/sam3_dynamic_gs/bin/python`

Viser live view: **http://localhost:8081** (NOT :7007).

> **Current pipeline = `dynamic_gs2/`** (the clean rewrite, built + validated). The old
> `dynamic_gs/` package is the frozen ground-truth baseline — its `ns-train` methods still
> run but new work goes through `dynamic_gs2/`. Single source of truth for live status:
> [`dynamic_gs2/STATUS_LIVE.md`](dynamic_gs2/STATUS_LIVE.md).

---

## 1. Run the pipeline (dynamic_gs2)

**FOUR modes, nothing more.** The only axes are: full (run static) vs warm-start (load `static_state.pt`),
and live (camera) vs recorded (replay through SHM). The core pipeline is source-agnostic — these scripts
just pick `--source {live_bridge,replay}` and whether the static phase runs.

```bash
# 1) FULL LIVE — whole pipeline in ONE process (needs Gazebo/ROS + dVRK up):
#    live sweep (red-box UI) -> segment -> SAM3D -> seed -> train -> Phase-0b fuse
#    -> re-phase static->dynamic IN PLACE -> live track (+ FF).
dynamic_gs2/full_live.sh <data_dir> [prompt]

# 2) FULL RECORDED — whole pipeline on a recorded dataset (no sim, no UI):
#    static reuses static_scene/ on disk (anchor = last keyframe, or --trigger-frame N),
#    dynamic replays dynamic_scene/ through SHM. Prints phase boundaries.
dynamic_gs2/full_recorded.sh <data_dir> [prompt] [--trigger-frame N] [--no-ff]

# 3) WARM LIVE — skip static; warm-load static_state.pt + models, track on the LIVE camera:
dynamic_gs2/warm_live.sh <data_dir> [--no-ff]

# 4) WARM RECORDED — skip static; warm-load static_state.pt, replay dynamic_scene/ (no sim).
#    The fast dev path to see the dynamic phase without retraining:
dynamic_gs2/warm_recorded.sh <data_dir> [transforms_name] [--no-ff] [--fps N] [--loop]
```

Validation / dev tools (NOT pipeline entrypoints):
```bash
# Visual: replay a recorded dataset through the pipeline with the viser viewer up.
dynamic_gs2/view_dynamic.sh "<dataset>" transforms_313_trimmed.json   # --once / --ff

# Recorded A/B (unattended-validated, FF off): old-vs-new per-tick trace verdict.
dynamic_gs2/replay_ab.sh "<dataset>" transforms_313_trimmed.json

# Side-by-side [live | new-render] mp4 of the tracking:
python -m dynamic_gs2.visualize "<dataset>" --transforms transforms_313_trimmed.json
#   -> <dataset>/dynamic_gs2_viz.mp4
```

`<data_dir>` lives under `data_teleoperation/datasets/`.

---

## 2. Old pipeline (`dynamic_gs/` — frozen baseline)

Still runnable; kept as the ground-truth reference the rewrite is verified against.
```bash
scripts/capture_only.sh   [data_dir]              # record static+dynamic, no training
scripts/bootstrap_live.sh <data_dir> "<prompt>"   # capture -> train static -> go live
scripts/resume_live.sh    <data_dir>              # resume a trained scene

# Direct ns-train method calls (the scripts wrap these):
ns-train static-gs        --data <data_dir> --pipeline.model.sam3_prompt_text "<prompt>"
ns-train static-gs-preseg --data <data_dir> --pipeline.text-prompts "<prompt>"   # per-Gaussian IDs
ns-train dynamic-gs       --data <data_dir>   # recorded dynamic dataset
ns-train dynamic-gs-live  --data <data_dir>   # live SHM stream
```

---

## 3. Save the feedforward (CDN) debug frames

Prefix any old-pipeline live launch with `DGS_FF_DEBUG=1`:
```bash
DGS_FF_DEBUG=1 scripts/resume_live.sh   <data_dir>
DGS_FF_DEBUG=1 scripts/bootstrap_live.sh <data_dir> "<prompt>"
```
Recorded mode (no dVRK/Gazebo needed — same churn, easiest to inspect):
```bash
ns-train dynamic-gs --data <data_dir> --pipeline.save-debug-images=True
```
Frames → `<data_dir>/dynamic_scene/_ff_debug/call_NNNN_frame_MMMMMM_K_NAME.png`
(7 per FF call: `1_gripper_mask 2_object_mask 3_real 4_rendered
5_rerendered_after_cull 6_raw_mask 7_clean_mask`). Off by default (adds renders
on the FF thread → slower tracker). `_4`↔`_5` = cull effect; `_6`↔`_7` = mask cleanup.

---

## 4. Inspect the SAM3D insert / registration

Canonical anchor (the exact frame SAM3D used) after a **live** run:
`<data_dir>/static_scene/anchor_ref/` — `overlay.png` is the guaranteed-correct
mask-on-image; `rgb.png / mask_NN.png / depth.tiff / intrinsics.json / c2w.json`
are what phase-0 reads. If `overlay.png` looks right but the fused object is off,
the bug is downstream of the mask (registration/cull), not the mask itself.

Run with the main env python; PLYs land in the dataset's
`dynamic_scene/initialization_debug/` (open in SuperSplat / any PLY viewer):
```bash
PY=/home/mrc-cuhk/miniconda3/envs/dynamic_gs/bin/python
DS=/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets

# Is the inserted object offset from the scene? (writes overlay PLY + cm offsets)
$PY scripts/diag_insert_offset.py "$DS/<dataset>"

# How much table contaminates the registration target? (table % + bias)
$PY scripts/diag_target_contamination.py "$DS/<dataset>"

# Border-leak / near-surface filter preview (kept=green, dropped=red PLY)
$PY scripts/diag_near_surface.py "$DS/<dataset>"

# Validate the anchor-as-final-keyframe fix on a dataset (helper + alignment)
$PY scripts/diag_validate_fix.py "$DS/<dataset>"
```
(Other `scripts/diag_*.py` exist for deeper investigation — pose drift, mask
alignment, etc.)

Re-run FastSAM on a saved frame (sam3 env), e.g. to check a mask:
```bash
/home/mrc-cuhk/miniconda3/envs/sam3_dynamic_gs/bin/python \
  dynamic_gs/utils/fastsam_segmentation.py \
  --image <data_dir>/dynamic_scene/initialization_debug/static0_rgb.png \
  --text-prompt "banana" --output-dir /tmp/repro --output-stem repro --imgsz 1024
```

---

## 5. Stop / GPU / safety

```bash
# Stop a running ns-train (kill the PYTHON pid, NOT the bash wrapper):
PYPID=$(ps -ef | grep "ns-train.*dynamic-gs" | grep -v grep | awk '{print $2}')
kill -9 $PYPID
# then ALWAYS verify nothing is holding VRAM:
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader

nvidia-smi                      # full GPU status
```
- NEVER `pkill -9 -f python` — it kills Gazebo / roscore / the dVRK.
- The launch scripts auto-pin the pipeline off CPU cores 0–3 and lock the dVRK
  onto them (needs sudo once; pass `DGS_NO_CPU_PIN=1` to skip).

---

## 6. Useful env vars

| var | effect |
|---|---|
| `DGS_FF_DEBUG=1` | save FF/CDN debug frames on a live run (section 3) |
| `DGS_FF_ICP=0` | disable AnySplat FF ICP refine (inserts placed via raw live pose). Default is ON (looks better — less seam with the static scene; not rigorously A/B'd). `=1` forces on. Runs off the tracker thread either way. |
| `DGS_FF_MAX_SCALE_M=0.03` | clamp each FF-inserted gaussian's per-axis world scale (m). Stops one oversized insert from smearing the scene; lower = tighter. 0 disables. (dynamic_gs2 default 0.02; old dynamic_gs/ 0.05.) |
| `DGS_FF_MIN_SCALE_M=0.0005` | drop FF-inserted gaussians whose largest axis < this (m) — culls sub-mm specks. Default 0.0 (off). |
| `DGS_NO_CPU_PIN=1` | skip the dVRK CPU isolation |
| `DGS_LIVE_DEFER_TSDF=0` | use the concurrent TSDF fuser instead of the deferred batch seed |
| `DGS_FUSION_DEVICE=cpu` | force CPU TSDF (default auto-GPU) |
| `DGS_TSDF_DEPTH_MAX_M=6.0` | raise the 2 m TSDF integration cap (default 2.0). **OOM risk** — raise `DGS_TSDF_VOXEL_M` too and watch `nvidia-smi`. |
| `DGS_TSDF_VOXEL_M=0.004` | coarser TSDF voxel (default 0.002 = 2 mm); needed if you raise the depth cap on a 16 GB GPU. |
| `DGS_EAGER_ANYSPLAT=1` | preload AnySplat during capture/training (set by bootstrap) |

Tracker live A/B knobs (no relaunch): `DGS_HOLD_WINDOW`, `DGS_HOLD_TRANS_MM`,
`DGS_HOLD_ROT_DEG`.
