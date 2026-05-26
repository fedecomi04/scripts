# InstaInpaint feasibility study for dynamic-gs

Status: research + design only. No runtime code modified.
Branch: `feedforward_dev`.
Recorded test dataset (verified present):
`/home/mrc-cuhk/Documents/dynamic_gaussian_splat/data_teleoperation/datasets/dynamic_gs_test_2026-03-28_19-49-45_w_background/` — 61 dynamic frames at 5 Hz, depth uint16 (`1e-3` m/unit), per-frame gripper masks.

Sources read:
- Paper: <https://arxiv.org/pdf/2506.10980>
- Project page: <https://dhmbb2.github.io/InstaInpaint_page/>
- GitHub: <https://github.com/dhmbb2/InstaInpaint> (shallow clone inspected in `/tmp/InstaInpaint/`)
- Runtime files: `dynamic_gs/dynamic_gs_pipeline.py`, `dynamic_gs/dynamic_gs_model.py`, `dynamic_gs/utils/optim_pool.py`, `dynamic_gs/utils/xfeat_motion.py`, `CLAUDE.md`.

---

## 1. InstaInpaint capability inventory

**InstaInpaint is NOT a Gaussian-scene editor.** Reading
`instainpaint/evaluate.py:658-746` (the `test_one_iteration` function) and
`instainpaint/models/gaussian_decoder.py:38-90` makes this unambiguous: it is
a **per-pixel Large Reconstruction Model (LRM)** in the
LGM / PixelSplat / LRM family. It produces a brand-new feed-forward Gaussian
scene from N posed RGB images plus 2D inpainting masks. It does **not**
consume any existing `.ply`, does **not** look at any pre-existing Gaussian
parameters, and has no concept of "delete the Gaussian at world position X".

Concretely, per `evaluate.py:683-728`:

| Input | Shape / type | Source in repo |
|---|---|---|
| `rgb_input` | `(B, N=4, 3, 512, 904)` BF16 | dataset image loader |
| `instance_masks_input` | `(B, N=4, 1, H, W)` BF16, 1 = inpaint here | precalculated SAM2 masks |
| `rays_o_input` / `rays_d_input` / `rays_d_un_input` | Plücker rays from posed cameras | `cameras_input` |
| `cameras_input` / `cameras_output` | extrinsics + intrinsics | dataset |

Reference-image semantics: the **first** of the 4 input views has its mask
forcibly zeroed (`evaluate.py:690-691` — "remove the mask of the reference
image"). So the model takes 1 clean reference + 3 mask-only views and
predicts the inpaint content guided by the reference appearance.

Output (`gaussian_decoder.py:42-46, 86-90`, then `evaluate.py:728-746`):

| Field | Per-pixel head | Shape |
|---|---|---|
| `depth` | sigmoid, then `(1-d)·near + d·far`, range `[0, 500]` units | `(B, N, 1, H, W)` |
| `rgb` | direct RGB (no SH) | `(B, N, 3, H, W)` |
| `opacity` | sigmoid with bias -2 | `(B, N, 1, H, W)` |
| `scale` | `exp(scale + scale_bias=-2.3)`, clamped to `[1e-4, 0.3]` | `(B, N, 3, H, W)` |
| `rotation` | quaternion | `(B, N, 4, H, W)` |
| `xyz` | computed downstream as `rays_o + rays_d_un * depth` | `(B, N, 3, H, W)` |

So the output is exactly `N × H × W` Gaussians (`4 × 512 × 904 ≈ 1.85 M`
per call) laid out as one Gaussian per input pixel. **No SH at all** —
`color_space="rgb"` (`gaussian_decoder.py:28`), single RGB head. Our
Splatfacto scene uses SH degree 3 (16 coefficients per channel).

Model weights: ViT-style transformer encoder/decoder, ~24-block depth,
1024-dim embeddings, patch=8. Checkpoint
`exp_ins+random+3d_121_multimask.pth` on HuggingFace (size not advertised
in README; the architecture suggests ~1-3 GB).

Hard deps: torch 2.5.1 + cu124 + flash-attn 2.8.3 + gsplat. This is
incompatible with `dynamic_gs`'s env (torch 2.11+cu128). InstaInpaint
must run in its **own conda env**, mirroring how SAM3 and SAM3D are
already isolated (`utils/sam3.py`, `utils/sam3d.py` both shell out via
`conda run -n sam3_dynamic_gs ...`).

Latency: claimed 0.4 s/forward at 512×904, 4 input views, on a single
modern GPU. VRAM not stated; the 24-layer transformer on 8128-token
sequence (4·512·904/64) plus flash-attn comfortably fits in a 16 GB GPU.

Training domain: DL3DV (in-the-wild indoor + outdoor) and Spin-NeRF
(indoor scenes). No claim of robotic / tabletop or close-range
generalization. No metric-scale guarantee — depth output is in
"near/far" sigmoid coordinates, with near/far set per scene.

**Hard incompatibilities with our scene format:**

1. **Output is per-pixel Gaussians, not "patches to glue into existing scene".** The model emits a complete `1.85 M`-Gaussian scene for the masked views. We would have to (a) discard everything outside the mask, (b) merge what survives into our existing Splatfacto scene. The merge is non-trivial because:
2. **No SH coefficients.** Output is `rgb` only. To live in our Splatfacto scene we must back-convert: write `rgb` into `features_dc` (the DC SH band) via `RGB2SH(rgb)`, zero all `features_rest`. Acceptable but means the inpaint patch has flat shading.
3. **Coordinate frame.** InstaInpaint's `xyz` is in the world frame implied by `cameras_input`. As long as we pass the same world-frame c2w that Splatfacto uses (the optimized ones from the camera optimizer — see Phase 0b notes in `CLAUDE.md`), the output xyz lands in our world frame. `auto_scale_poses=False` is exactly what we want here — InstaInpaint's `depth_far=500` clamp easily covers the ~1-3 m tabletop range, but the small absolute scale (gauges in meters) may sit awkwardly within a transformer trained on much larger DL3DV scenes — likely the biggest open question (see §5).
4. **No "scene-conditioning" path.** The model cannot see what is *already there* — it always produces a fresh prediction. Our existing scene Gaussians around the hole edges therefore won't influence its output, and there is no mechanism for the inpaint patch to texture-match the surrounding wood-grain / specular highlights other than via the reference view.

---

## 2. Integration shape

### Where it would slot in

CDN computation stays unchanged
([`_compute_change_mask`, dynamic_gs_pipeline.py:1767](../dynamic_gs/dynamic_gs_pipeline.py#L1767)).
The integration point is the **pool consumer**, not the producer.

Current shape, simplified
([`_dynamic_get_train_loss_dict`, dynamic_gs_pipeline.py:3090-3208](../dynamic_gs/dynamic_gs_pipeline.py#L3090)):

```
tracker_tick  -> apply (R,t) to object Gaussians          (XFeat, ~50 ms)
push (camera, cdn) -> OptimPool (cap 15)                  (when CDN > 500 px)
pool.pick_round_robin() -> one masked photometric step    (~58 ms, ×50 = 2.9 s)
evict on epoch budget or 0.3× initial loss
```

Proposed feed-forward shape:

```
tracker_tick  -> apply (R,t) to object Gaussians          (XFeat, unchanged)
push (camera, cdn) -> CdnInpaintQueue                     (when CDN > 500 px)
queue.pop_one() ->
    1. select 4 reference views (1 clean + 3 with our CDN as input mask)
    2. resize all to 512×904, build Plücker rays
    3. conda-run InstaInpaint subprocess -> gaussians (xyz, rgb, opacity, scale, quat)
    4. crop output to projected-CDN pixels of view 0      (discard the other 3·H·W gauss.)
    5. delete the existing scene Gaussians whose 2D footprint lies wholly inside CDN
       AND whose flag `object_flags == 0` (do NOT touch the moved object)
    6. insert the cropped feed-forward gauss with SH=0 (rgb -> features_dc, zero features_rest)
    7. mark inserted gauss as scene (object_flags=0) and exempt from scene-opt
       gradient hooks for K subsequent frames (let them settle visually)
```

### What replaces the 50-step optim

The 50 inner steps disappear. The replacement is **one InstaInpaint
forward + one Gaussian-list mutation**. The expected wall time of
`0.4 s + IPC` is roughly comparable to today's `50 × 58 ms = 2.9 s`,
i.e. ~7× speedup if the IPC stays under ~100 ms, smaller (~1×) if we
have to pay subprocess spawn each call.

### CDN → InstaInpaint mask

CDN is a 2D pixel mask (same shape as live RGB). InstaInpaint's
`instance_masks_input` is also 2D pixel. So **the mapping is direct** for
view 0 (the live frame at capture time). For the other 3 reference views,
we need to **back-project** the CDN region to 3D (using
`rays_o + rays_d * depth_now` from the current rendered depth) and
re-project into each reference view to get its 2D inpaint mask. The
back-projection helper already exists in the model
(`_backproject_mask_to_world` is referenced in CLAUDE.md's Phase 0b
narrative).

### Delete or add?

The cleanest answer is **delete-then-add**:
1. Find scene Gaussians (`object_flags == 0`) whose projected 2D centers
   in the inpaint camera fall inside CDN. (`extract_projected_centers_and_radii`
   already exists in the model; cited in the D0.1e flagging path.)
2. Mark them for removal *after* the InstaInpaint forward returns.
3. Append the feed-forward Gaussians.

This avoids the "two layers of color at the same XYZ" problem that pure-add
would cause, and reconciles cleanly with `scene_opt_active_mask`: the
freshly inserted Gaussians get `object_flags=0`, are inside the change
region, and their next forward sets `scene_opt_active_mask=True` for them
automatically. No new flag needed.

### Reference views — the design's biggest open variable

InstaInpaint takes 4: 1 clean reference + 3 inpaint views. Three sensible
sources of references in our context:

- **Reference (clean):** the **rendered static scene** at the inpaint camera
  pose. This was the entire point of Phase 1 — the static scene IS what
  belongs in the un-occluded background. Render it on demand. The model
  has never seen "renders of Gaussian scenes" at training time, but it
  has seen plenty of clean indoor images; the appearance domain gap
  should be manageable.
- **3 inpaint views:** the most recent 3 frames from the optim pool with
  CDN > 500 px. They already carry the "this region is dynamic, infer what
  goes here" signal. Their masks ARE the per-frame CDNs.
- Cameras: their stored `(camera, c2w)` from the pool entry.

This means we do not need to retain the OptimPool as an *optimization*
pool, but we still need it as a **short-term ring buffer of recent
posed frames + their CDNs** to feed InstaInpaint. Rename rather than
remove.

---

## 3. Recorded-data test plan

The user, not the agent, runs this.

1. **Setup.** Clone `InstaInpaint` into `third_party/InstaInpaint/`, create
   `instainpaint` conda env per the upstream README (torch 2.5.1 + cu124 +
   flash-attn 2.8.3, ~10 GB of deps). Download
   `exp_ins+random+3d_121_multimask.pth` (~1-3 GB) to
   `third_party/InstaInpaint/checkpoints/`.

2. **Probe.** Run a standalone subprocess test (separate from `ns-train`):
   pick 4 frames from the recorded dataset's `dynamic_scene/rgb/`, fabricate
   a square CDN-style mask in the center of 3 of them, call InstaInpaint's
   `evaluate.py`-style inference on them, and dump the predicted Gaussians
   to PLY. **Inspect manually** in `splat-viewer` or Open3D. The two
   things to check are (a) does the predicted Gaussian cloud sit at
   roughly the right metric scale relative to our static-scene PLY, and
   (b) is the inpaint content visually plausible. If either fails,
   stop — don't proceed to the pipeline integration.

3. **Static phase.** Run `ns-train dynamic-gs` for the 4000 static steps
   normally. **Save the trained Gaussian state** before the dynamic
   boundary (the current pipeline does not checkpoint, so add a one-off
   `torch.save(model.state_dict())` to the boundary callback for this
   experiment, or use `ns-export` after a stop-and-resume).

4. **Throttled feeder + InstaInpaint dynamic phase.** Run the dynamic
   phase with both (a) the throttle (§4) and (b) an
   `enable_feedforward_inpaint=True` flag that routes pool entries to
   InstaInpaint instead of the optim step. For the first M=10 frames
   only, write per-frame:
   - `render_pre.png` (RDN before inpaint)
   - `render_post.png` (RDN after inpaint patch insertion)
   - `live.png` (ground-truth dynamic frame)
   - `cdn.png` (the mask)
   - `ply_pre.ply` / `ply_post.ply` (Gaussian state delta)
   to `docs/feedforward_dev_results/frame_NN/`.

5. **Metrics.** Compare against the existing 50-step baseline on the same
   recorded dataset:
   - **PSNR / SSIM on the CDN region only** (mask both pred and live with
     CDN before computing — small absolute differences off-mask should
     not dominate).
   - **Gaussian count delta** per frame (InstaInpaint emits ~few-k
     per-view Gaussians for a small CDN region; bound the bloat).
   - **Wall-clock** of the dynamic phase end-to-end (`time.time()` around
     the dynamic loop).
   - **Visual sanity:** the `pipeline_presentation.png` generator already
     in `scripts/` can be re-run to produce a final render comparison.

---

## 4. Throttling implementation

Target rate: **5 Hz wall-clock**, matching what the live tracker was tuned
against (CLAUDE.md "Live tracker 30 Hz fix" section explicitly calls out
the 5 Hz nominal camera rate). Higher is wasteful (object motion exceeds
tracker delta tolerance); lower starves the optim pool.

**Where to gate.** The tracker tick is what drives recorded-data progression
([`_dynamic_get_train_loss_dict`, dynamic_gs_pipeline.py:3122-3128](../dynamic_gs/dynamic_gs_pipeline.py#L3122)):

```python
elif (
    self._dynamic_step_counter % cadence == 0
    and self._next_frame_to_track < self.total_dynamic_frames
):
    self._tracker_tick(self._next_frame_to_track)
    self._next_frame_to_track += 1
```

The minimal-surface-area change is a sleep gate inside `_tracker_tick`
(recorded path only), modeled after the existing 50 Hz throttle pattern
in `live_ros_publisher.py` (`_POSE_JOINT_MIN_DT_SEC = 0.02`). Pseudocode:

```python
# Top of _tracker_tick(self, frame_idx):
if not self.config.live and self.config.recorded_feeder_min_dt_sec > 0:
    now = time.monotonic()
    if self._last_recorded_tick_wall is not None:
        elapsed = now - self._last_recorded_tick_wall
        slack = self.config.recorded_feeder_min_dt_sec - elapsed
        if slack > 0:
            time.sleep(slack)
    self._last_recorded_tick_wall = time.monotonic()
```

**Config.** Add to `DynamicGSPipelineConfig`:

```python
recorded_feeder_min_dt_sec: float = 0.20   # 5 Hz; 0.0 to disable
```

Override via env var for benchmarking: `DGS_RECORDED_FEEDER_HZ=0` skips
the gate entirely. Implement by reading the env var in `__init__` after
config is materialized.

**Live mode unaffected** because `self.config.live` short-circuits.
**Tracking-only mode unaffected** because `disable_dynamic_optimization`
still routes through the same tracker tick — but the throttle is the
same regardless of whether the optim path runs.

This is a 6-line addition to `_tracker_tick` plus 2 lines of config —
**out of scope for this design doc commit** but documented here for the
implementation phase.

---

## 5. Risks + open questions

**Could kill outright:**
- *Metric-scale generalization.* InstaInpaint was trained on DL3DV/Spin-NeRF
  scenes where scene-scale is dataset-normalized via `near/far`. Our scene
  is metric meters with `auto_scale_poses=False`, and a tabletop occupies
  ~0.5-1.5 m from the camera. The fixed `depth_far=500` clamp in
  `gaussian_decoder.py:16` is geometric, not perceptual — if the
  transformer's notion of "distant background pixels" expects depth
  values in the tens-of-units range, our 1 m depths may collapse onto
  a near-saturated sigmoid. The probe in §3 step 2 is *the* gate.

**Likely "works but quality regression":**
- *Texture/appearance discontinuity at CDN boundary.* The inpaint patch
  comes from a model that has never seen our scene's Gaussians. SH=0
  flat shading + per-pixel Gaussians from 4 ref views will not match
  Splatfacto's SH=3 anisotropic shading around the hole. Visible seams
  in side-by-side views are the expected first-cut failure mode. The
  current optim loop, by contrast, blends seamlessly because it optimizes
  the same SH parameters that bound the surrounding scene.

**Assumptions to verify before committing:**
- That `evaluate.py`'s `test_one_iteration` is callable as a Python API
  (not just a script). It is — the function takes a `batch` dict and
  returns predictions — but the surrounding `SpinNerfDataset` does
  significant preprocessing (Plücker ray construction, FOV→intrinsic
  conversion, centralized cropping). Replicating that for our cameras
  is the bulk of the wrapper work.
- That `gsplat`-format outputs (xyz, opacity, scale, quat) match our
  Splatfacto buffer ordering. They almost certainly do (both are
  3D-position + scale-3 + quat-4 + opacity-1), but the quat convention
  (`wxyz` vs `xyzw`) and scale activation (`exp` vs `log`) must be
  checked.
- That the upstream env (torch 2.5.1) coexists with our env on the same
  machine without LD conflicts. The SAM3 / SAM3D precedent says yes via
  conda isolation.

---

## 6. Recommended next step

**Proceed to a small, gated implementation — but run the §3 step 2
probe first as a hard gate.** Do not skip it.

Reasoning: InstaInpaint's architectural shape (per-pixel LRM, 4 input
views, 2D mask, 0.4 s forward) maps cleanly onto our pool-consumer
contract. The integration risk is concentrated in two places that are
both empirically testable in <1 day:

1. *Does it produce metrically-sensible Gaussians for our 1 m tabletop
   given 4 of our recorded frames as input?* — answered by the probe.
2. *Does the patched scene render coherently after deleting +
   reinserting Gaussians?* — answered by the §3 M=10 frame comparison.

If both gates pass, the implementation is mechanically simple:
- New `dynamic_gs/utils/instainpaint_subprocess.py` mirroring the
  `sam3d.py` subprocess pattern.
- New conda env `instainpaint_dynamic_gs`.
- ~150-line `_inpaint_one_pool_entry` in pipeline replacing the
  `_dynamic_get_train_loss_dict` pool-pick branch behind a config flag.
- 6-line throttle in `_tracker_tick`.

If the metric-scale probe fails, **stop**. Retraining InstaInpaint on
robotic-scale data is multi-week work and out of scope for "feasibility
study". A weaker fallback would be a smaller cost-reduction on the
existing optim loop (e.g. drop `optim_pool_max_epochs` from 50 → 10) —
worth keeping as Plan B but explicitly not what this experiment is
testing.

The XFeat tracker sweet-spot configuration documented in CLAUDE.md
(2026-05-26 entries on `xfeat_top_k=300`, depth_confidence=-1.0,
ransac_iterations=32) is **untouched** by anything proposed here —
InstaInpaint replaces only the optim half of the dynamic loop, not the
tracker half.
