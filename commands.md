# commands.md — quick reference

Run everything from `scripts/`:
```bash
cd /home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts
```

Env pythons (for the diagnostic scripts below):
- main:   `/home/mrc-cuhk/miniconda3/envs/dynamic_gs/bin/python`
- sam3:   `/home/mrc-cuhk/miniconda3/envs/sam3_dynamic_gs/bin/python`

Viser live view: **http://localhost:8081** (NOT :7007).

---

## 1. Run the pipeline

```bash
# Capture only (record static+dynamic dataset, no training).
# Default data dir = datasets/<timestamp>/
scripts/capture_only.sh [data_dir]

# Full: capture -> train static -> go live. Prompt = bare noun ("banana", no "the").
scripts/bootstrap_live.sh <data_dir> "<prompt>"

# Resume an already-trained scene (skips capture + training; needs
# <data_dir>/static_scene/static_state.pt).
scripts/resume_live.sh <data_dir>
```

Direct method calls (the scripts wrap these):
```bash
ns-train static-gs        --data <data_dir> --pipeline.model.sam3_prompt_text "<prompt>"
ns-train static-gs-preseg --data <data_dir> --pipeline.text-prompts "<prompt>"
ns-train dynamic-gs       --data <data_dir>   # recorded dynamic dataset
ns-train dynamic-gs-live  --data <data_dir>   # live SHM stream
```

`<data_dir>` lives under `data_teleoperation/datasets/`. A bare name (e.g.
`my_scene`) is resolved there automatically by capture_only.sh.

---

## 2. Save the feedforward (CDN) debug frames

Prefix any live launch with `DGS_FF_DEBUG=1`:
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

## 3. Inspect the SAM3D insert / registration (the current work)

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

## 4. Stop / GPU / safety

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

## 5. Useful env vars

| var | effect |
|---|---|
| `DGS_FF_DEBUG=1` | save FF/CDN debug frames on a live run (section 2) |
| `DGS_FF_ICP=0` | disable AnySplat FF ICP refine (inserts placed via raw live pose). Default is ON (looks better — less seam with the static scene; not rigorously A/B'd). `=1` forces on. Runs off the tracker thread either way. |
| `DGS_FF_MAX_SCALE_M=0.03` | clamp each FF-inserted gaussian's per-axis world scale (m). Default 0.05 (5 cm). Stops one oversized insert from smearing the scene; lower = tighter. 0 disables. |
| `DGS_FF_MIN_SCALE_M=0.0005` | drop FF-inserted gaussians whose largest axis < this (m) — culls sub-mm specks. Default 0.0 (off). |
| `DGS_NO_CPU_PIN=1` | skip the dVRK CPU isolation |
| `DGS_LIVE_DEFER_TSDF=0` | use the concurrent TSDF fuser instead of the deferred batch seed |
| `DGS_FUSION_DEVICE=cpu` | force CPU TSDF (default auto-GPU) |
| `DGS_TSDF_DEPTH_MAX_M=6.0` | raise the 3 m TSDF integration cap (default 3.0). **OOM risk** — raise `DGS_TSDF_VOXEL_M` too and watch `nvidia-smi`. |
| `DGS_TSDF_VOXEL_M=0.004` | coarser TSDF voxel (default 0.002 = 2 mm); needed if you raise the depth cap on a 16 GB GPU. |
| `DGS_EAGER_ANYSPLAT=1` | preload AnySplat during capture/training (set by bootstrap) |

Tracker live A/B knobs (no relaunch): `DGS_HOLD_WINDOW`, `DGS_HOLD_TRANS_MM`,
`DGS_HOLD_ROT_DEG`.
