# FF insert density — what was tried and why it was reverted (2026-06-19)

**Problem being chased:** AnySplat inserts looked bad in the live FF viewer — two distinct symptoms:
1. **Holes on flat surfaces** (the table), especially the table's narrow *width* side: viewed from a
   grazing angle, the flat splats (thin discs oriented for the top-down capture view) present
   near edge-on → you see between them → holes → CDN re-flags → FF refills → churn.
2. **Edge/corner leak:** flat-surface splats grown/extended bleed past the object silhouette.

**Outcome:** every density-shaping attempt below made it **WORSE or no better** than plain raw
inserts. Reverted to **raw inserts** (`scale_multiplier=1.0`, `max_scale_m=0.02` cap on gross
outliers only, NO grow / NO voxel-dedup / NO cornerness). Next idea to try instead:
**cull existing scene gaussians NEAR each insert** (voxel-hash, O(N+M)) so raw inserts replace the
old surface rather than stacking on it — NOT scale-shaping the inserts.

## What was tested (in order) and why each failed

1. **`scale_multiplier` (×2.0 → uniform scale-up of every insert).** Original old-pipeline default.
   Inflated edge/background splats → giant blobs leaking past the change region → CDN re-flags →
   insert↔cull churn, scene ballooned 458k→1.28M. Set to **1.0** (no inflation). Helped the churn
   but left holes. KEPT at 1.0.

2. **Uniform in-plane grow (grow the 2 largest axes only, ×1.5 → ×4).** Idea: widen flat discs to
   fill gaps without inflating the normal (no blur). At ×1.5–2.5 still left many holes on the flat
   width-side; pushing to ×4 just made edges leak again (a wide disc near a silhouette bleeds past
   it). Uniform factor can't satisfy "fill flat gaps" AND "don't leak at edges" at once.

3. **Voxel-dedup thinning (3mm → 2mm → 1mm).** Idea: thin the dense AnySplat cloud so fewer/bigger
   splats. Thinning + uniform grow still holey at 2–3mm; at 1mm it's nearly no thinning (dense tiny
   splats = the original hole problem). No setting fixed the grazing-angle holes — because the hole
   is a *splat-orientation* problem (edge-on discs), not purely a density one.

4. **2-tier adaptive (flat vs non-flat by the splat's OWN aspect ratio).** flat → coarse voxel +
   big grow; non-flat → fine voxel + small grow. Failed two ways: (a) per-splat aspect ratio has
   **no spatial awareness** — a flat splat sitting next to a corner is still "flat" → grows big →
   leaks into the corner; (b) flat regions still had holes. Operator: "still many holes, and corners
   still leak."

5. **Continuous kNN-PCA cornerness (the most elaborate).** Per-point surface-variation
   (`l0/(l0+l1+l2)`) from k=8 neighbours → spatially-aware `corner_score ∈ [0,1]` (a flat point NEAR
   a corner scores high because its NEIGHBOURHOOD is non-planar = a free "halo"). LERP both the grow
   factor (flat 2.5 ↔ corner 1.5) and voxel (flat 3mm ↔ corner 1mm) by the score. Two crashes had to
   be fixed before it even ran a full episode:
   - **`torch.cdist` N×N (~1.9 GB at N~20k) → AnySplat-worker CUDA OOM** → FF stopped inserting
     "halfway." Fixed by swapping to **scipy cKDTree (CPU, O(N log N), 13ms typical)**.
   - **`eigvalsh` on degenerate/coincident neighbourhoods → cusolver NaN crash** (AnySplat emits
     duplicate back-projected points). Fixed with cov sanitize + ridge ε + CPU fallback + a
     graceful "treat batch as flat".
   - `corner_var_scale` calibrated 0.05 → 0.10 (measured: surf_var median 0.003, p90 0.05, max 0.11;
     0.05 over-flagged the top ~12% of FLAT table as corner).
   After all that it ran stably (0 OOM, 0 NaN, FF stable to tick 214) — but the **visual result was
   WORSE, "much worse than before."** The whole approach of shaping insert scales by geometry did not
   produce a good-looking surface. Reverted.

## The lesson

Shaping the *inserted splats'* scale (uniform, in-plane, or adaptive-by-cornerness) does **not**
fix the look — it trades holes for leaks or vice-versa, and adds real complexity + crash surface
(O(N²) memory, eigvalsh NaN).

## Measured insert density (DGS_DENSITY_DEBUG=1, real screwdriver run)

Per FF insert the cloud is HIGHLY variable:
- **Flat-patch inserts:** bbox ~160×100×**2 mm** (a thin slab), nn-spacing median **0.35 mm**,
  splat size **0.27 mm**. Voxel-dedup survivors: **1mm→24% (4× cut), 2mm→6%, 3mm→3%.**
- **Room-scale inserts:** bbox ~2.4×4.8×1.7 **m**, nn-spacing 0.8–8 mm (very non-uniform), splats
  already 1–3.5 mm. Survivors: **1mm→89–98% (barely thins), 3mm→44–68%.**

Key facts this proves:
- Density varies ~10× between consecutive inserts → no single fixed voxel is right for all.
- Splats (0.27 mm on flat) are MUCH smaller than any useful voxel → **dedup ALWAYS opens sub-voxel
  holes unless survivors are grown to ~voxel size** (and growing leaks at edges). That tension is
  fundamental, not a tuning miss.
- Empty/sparse regions are untouched by dedup (0–1 pt/voxel) — dedup only removes, never fills.

## Current state (2026-06-19)

**RAW plain inserts look BEST** (full coverage; cost is gaussian count, e.g. 458k→~943k/episode = a
load-shed concern, not a visual one). 3mm voxel = holes or leaks. This is the shipped default.

## Still to try (untested)

1. **Cull existing scene gaussians where the patch lands** — insert raw, but first remove the OLD
   scene geometry the insert overlaps (voxel-hash nearby-cull, protect d0), so new+old don't stack.
   Attacks the overlap/stacking ugliness WITHOUT thinning the insert (no holes from dedup).
2. **1mm voxel-dedup inserts + slight grow (~×1.5)** — the dense flat inserts (0.35mm spacing) 4×-cut
   at 1mm; a small ×1.5 grow on survivors (0.27→0.4mm, still ≤1mm) to close the sub-voxel gap.
   Milder than the ×2.5–4 grows that leaked; may thin a lot with acceptable holes. Untested at 1mm
   WITHOUT the aggressive grow.
