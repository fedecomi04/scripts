# Post-submission change checklist

Changes to make to the pipeline AFTER the ICRA submission is out. Collected while
re-reading the report. Each item = what's wrong / desired + where the code lives.

## 1. Object selection = user click, fall back to closest (NOT most-Gaussians)

**Current (paper eq 8):** the tracked object is the non-zero `object_instance_ids`
with the largest Gaussian count — `pick_d0_instance_id` in
`dynamic_gs2/pipeline.py:32` (`torch.unique(...).counts.argmax()`). This is a weak
criterion with multiple objects.

**Desired:**
1. In the DYNAMIC pipeline UI, let the operator **click on the RGB frame**.
2. Reproject that clicked pixel to a Gaussian (back-project through live depth +
   the current camera pose, or a per-pixel instance-id render).
3. If the hit Gaussian has a non-zero `object_instance_ids`, select **that**
   instance id as D0.
4. Fallback (no click / no hit) → **closest object** (e.g. nearest instance
   centroid to the camera or gripper TCP), NOT most-Gaussians.

**Where:**
- selector: `dynamic_gs2/pipeline.py::pick_d0_instance_id` (replace / add a
  click-driven + closest-object path). Callers: `run_live`, `run_recorded_trace`,
  `run_view_recorded`, plus the `DynamicLoop.__init__` `d0_id <= 0` guard.
- click surface: `dynamic_gs2/dynamic_viz.py` (viser bridge — add a click
  handler on the pushed RGB/render; viser client cameras + scene click events).
- instance-id-at-pixel: render an instance-id buffer, or back-project the clicked
  pixel via `scene_model` + live depth and look up the nearest Gaussian's
  `object_instance_ids`.
- NOTE: this overlaps the multi-object roadmap "gripper-TCP picker" (roadmap #2).

## 2. Background-color cull can drop genuinely background-colored FF geometry

**Current:** in the feed-forward reproject, AnySplat-decoded Gaussians whose
predicted color is within `0.08` per channel (AND across all 3) of the sky color
`(0.86, 0.92, 1.0)` are culled — `dynamic_gs2/anysplat_decode.py:678`
(`background_rgb=(0.86,0.92,1.0)`, `background_tol=0.08`, not config-overridden).
Aimed at AnySplat hallucinating sky-colored splats at silhouette edges. Only
touches FF inserts inside the CDN change region — NOT the static object or the
tracked object.

**Problem:** a genuinely sky-colored surface newly revealed during manipulation
(pale blue-white patch behind a lifted object) gets wrongly culled → the region
stays a hole. Also brittle to a different scene background (the sim sky is fixed
and known; real captures are not).

**Desired (post-submission):** make the cull object/geometry-aware instead of a
blanket color gate. Options:
- Skip the bg-color cull where the FF pixel has a VALID live sensor depth in
  range (a real revealed surface returns depth; a hallucinated sky splat does
  not) — i.e. only cull sky-colored splats that also lack sensor support.
- Or make `background_rgb` / `background_tol` config-driven and default the cull
  OFF for real (non-sim) captures.

**Where:** `dynamic_gs2/anysplat_decode.py::reproject_anysplat_to_scene` (the
`keep_bg` mask, ~line 678) + expose `background_tol`/`background_rgb` via
`dynamic_gs2/config.py` (`FeedforwardConfig`), passed from
`dynamic_gs2/dynamic_ff_backends.py:303` (currently uses the function defaults).

NOTE: paper line describing this cull was REMOVED (2026-07-06), so no paper
change needed — this is code-only.

## 3. Change the default viewer layout

**Current defaults** (`dynamic_gs2/dynamic_viz.py`):
- `DEFAULT_VIEW_MODE = "1 render"` (line 210)
- Render 1 (bottom-left) default source = `"Top"` (line 255)
- Render 2 (top-centre) default source = `"Top"` (line 259)
- Panel layout (`_compose`, docstring lines 11-13): bottom-left = render 1,
  bottom-right = real camera feed, top-centre = render 2 (only in "2 render" mode).

So today the default is a SINGLE bird's-eye render (bottom-left) + camera feed
(bottom-right); the top panel is empty unless the user switches to "2 render".

**Desired default:**
- Top of screen (top-centre) = **Top view** (bird's-eye).
- Bottom-left = **rendered same-camera view** (render source `"Cam"`).
- Bottom-right = **real camera feed**.

i.e. default to **"2 render"** mode with Render 1 = `"Cam"` (bottom-left) and
Render 2 = `"Top"` (top-centre).

**Where:** `dynamic_gs2/dynamic_viz.py`:
- `DEFAULT_VIEW_MODE = "2 render"` (line 210)
- `self._gui_r1_src.value = "Cam"` (line 255)
- `self._gui_r2_src.value = "Top"` (line 259 — already Top)
- sanity-check `_compose` / `_sync_gui_visibility` handle "2 render" as the
  initial mode (r2 panel + its controls visible from the start).

## 4. (add more here as found while reading the report)
