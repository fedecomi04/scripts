# PROMPT — Switch the Gazebo sim camera to ZED X (2.2 mm wide), phased

## Goal

Replace the current Kinect-style square sim camera with a **StereoLabs ZED X, 2.2 mm wide lens** model.
Do it in **two phases**: ship the light 16:10 resolution first and **commit it as a known-good fallback**,
then raise to the true ZED X resolution — so if the full resolution is too heavy (RTF/VRAM), we revert to
the committed checkpoint instead of re-deriving it.

ZED X 2.2 mm reference specs (datasheet v1.4): sensor 1/2.6", array 1928×1208, **3 µm square pixels**,
global shutter, **16:10** aspect. Native outputs (per eye): **1920×1200** (1200p), 1920×1080 (crop),
**960×600** (binning 2×2). FOV (2.2 mm): **110°(H) × 80°(V)** × 120°(D). Depth range 0.3–20 m.

## Target file

`~/dev/teleop/catkin_ws/src/active_camera_arm_control/active_camera_arm_examples/dynaarm_description/urdf/dynamic_gaussian_splat/dynaarm_with_gripper_for_gazebo_only_no_wrist_collision.urdf`

Camera sensor block: lines **99–152** (`<gazebo reference="camera_link">` → `depth` sensor →
`libgazebo_ros_openni_kinect.so`). This is the active URDF (the `dynamic_gaussian_splat/` subdir copy is the
canonical one per CLAUDE.md; the publisher reads intrinsics from `/camera_info`, which Gazebo derives from
these values). **Note:** this file is tracked by the `active_camera_arm_control` git repo
(`~/dev/teleop/catkin_ws/src/active_camera_arm_control`), **not** the `dynamic_gaussian_splat/scripts` repo —
the Phase-1 checkpoint commit lands there.

The two edits in both phases (only `<width>`/`<height>` differ between phases):

```xml
<horizontal_fov>1.9198622</horizontal_fov>   <!-- line 106: 110° = 110*pi/180 (was 1.3962634 = 80°) -->
<image>
  <width>...</width>     <!-- line 108 -->
  <height>...</height>   <!-- line 109 -->
  <format>R8G8B8</format>
</image>
```

`<horizontal_fov>` is **identical in both phases** (FOV is resolution-independent). Gazebo derives the pinhole
from `horizontal_fov` + aspect (square pixels → fx = fy, cx = W/2, cy = H/2).

---

## Phase 1 — 960×600 (do first, then COMMIT)

The compute-matched 16:10 target. It is a real ZED X output mode (binning 2×2), and at **576k px it is ~the
same pixel count as today's 800×800 (640k px)** — so performance should be essentially unchanged while the
camera becomes a faithful ZED X with the correct aspect ratio.

```xml
<width>960</width>
<height>600</height>
<horizontal_fov>1.9198622</horizontal_fov>
```

Implied intrinsics: fx = fy ≈ **336.1 px**, cx = 480, cy = 300, VFOV ≈ 83.5°.

**Validate Phase 1 (goal: confirm performance ≈ current 800×800):**
- [ ] Re-launch Gazebo; confirm `/dynaarm_arm/camera1/camera_info` reports W=960, H=600, fx≈336.
- [ ] `capture_only.sh`; confirm `static_scene/transforms.json` has the new `w`/`h`/`fl_x`/`fl_y`/`cx`/`cy`.
- [ ] Fresh `timing_report.txt`; **Gazebo RTF via `/clock` wall-rate**; `nvidia-smi` VRAM peak. Compare to the
      most recent 800×800 baselines — expect parity (this is the whole point of Phase 1).

**Then COMMIT the URDF as the fallback checkpoint** (in the `active_camera_arm_control` repo). If on its
default branch, branch first. Suggested message:

```
feat(sim): ZED X 2.2mm wide camera — 960x600 16:10 (binning-2x2 mode), 110° HFOV

Compute-matched checkpoint (~640k→576k px) before stepping up to full 1200p.
Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

This commit is the safety net: if Phase 2's full resolution is too heavy, `git checkout`/`revert` back to it.

---

## Phase 2 — 1920×1200 (the true ZED X 1200p; revert to Phase 1 if too heavy)

Faithful native ZED X 1200p, **3.6× the pixels** of Phase 1 (2.30 M vs 0.576 M).

```xml
<width>1920</width>
<height>1200</height>
<horizontal_fov>1.9198622</horizontal_fov>
```

Implied intrinsics: fx = fy ≈ **672.2 px**, cx = 960, cy = 600, VFOV ≈ 83.5° (same camera as Phase 1, finer
sampling — only `s = 2×` more pixels per axis).

**This is where the real work is** — the pipeline code is resolution-agnostic (verified: live SHM sizes from
`/camera_info`, transforms.json is intrinsics-driven, nerfstudio downscale pinned to 1, only literal `800`s in
`dynamic_gs/` are comments), but every **tuned value and measured baseline** assumes the old resolution and must
be re-validated at 3.6× the pixels:

1. **Re-measure all timings** — do NOT reuse old numbers (`memory/feedback_no_timing_guesses.md`): fusion
   ms/frame, static training wallclock, XFeat tick rate, AnySplat/FF latencies.
2. **Gazebo RTF** — more pixels in the depth-camera plugin lowers real-time-factor → live fps. The 30 Hz live
   path was RTF-bottlenecked and hard-won (CLAUDE.md 2026-05-26). Measure RTF; **this is the most likely thing
   that forces a revert to Phase 1.**
3. **VRAM headroom** — the CLAUDE.md VRAM table is all @800×800. Re-check tight OOM cases (SAM3D-trim peak +
   training + Gazebo). Larger render/decode buffers shrink the margin.
4. **XFeat tuning** — extract is image-bound (CLAUDE.md); revisit `xfeat_top_k=300` and the per-tick budget.
   cudnn.benchmark stays forced off (good).
5. **Aspect 1:1 → 16:10** — already crossed in Phase 1, so no new work here; the known paths (masks via
   `cv2.resize`, depth back-project via intrinsics) are aspect-safe.

**Decision gate:** if RTF or VRAM regress unacceptably, revert the URDF to the Phase-1 commit (960×600) and
stay there. Otherwise commit Phase 2 on top.

---

## Notes

- Optional realism (either phase): depth `<clip><near>` 0.01 → 0.3 (ZED X min range); `<far>` already 20.
- Datasheet lists 80° V but the pinhole derives ≈83.5°; the gap is lens distortion the datasheet's "max FOV"
  includes. Leave `distortionK1..T2` at 0.0 unless modeling distortion is an explicit goal.
- If line numbers have shifted from a prior edit, the `<width>`/`<height>`/`<horizontal_fov>` tags are
  unambiguous regardless.

## Out of scope (separate task)

Camera **position** change (`camera_link` mount pose). Track it independently from this resolution/FOV change.
