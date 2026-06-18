# Code Audit — Segmenters

Files audited:
- `dynamic_gs/utils/fastsam_segmentation.py`
- `dynamic_gs/utils/sam3_segmentation.py`
- `dynamic_gs/utils/esam.py`

Grep base for ref counts:
`grep -rn "<name>" dynamic_gs scripts --include=*.py` (callers exclude the symbol's own definition file).

---

## 1) FUNCTION/CLASS MAP

### fastsam_segmentation.py

- `_resolve_env_python(conda_env)` — fastsam_segmentation.py:41 — Returns the sam3-env python binary Path or None. — 1 caller: `run_fastsam_subprocess` (:540).
- `_compute_iou(mask_a, mask_b)` — fastsam_segmentation.py:49 — IoU of two binary masks. — **NO REFS FOUND** outside this file (dedup here uses an inline IoU at :411, not this helper). Note: an identically-named copy in sam3_segmentation.py IS used; this fastsam copy is a duplicate.
- `_touches_n_borders(mask, n=2)` — fastsam_segmentation.py:55 — True if mask touches >= n image borders. — 1 caller in-file (`infer` :353). No external refs (sam_worker uses the sam3_segmentation copy).
- `_default_weights_path(weights)` — fastsam_segmentation.py:68 — Resolve bare weights name to a cache path. — 1 caller in-file (`FastSamTextSegmenter.__init__` :159). No external refs.
- `select_kept_indices(probs, cosines, ...)` — fastsam_segmentation.py:79 — Auto-threshold: how-many (log-prob cliff) + whether-any (cosine presence gate). — 1 caller in-file (`infer` :369). Externally referenced only in a comment (live_session.py:167). Code refs: in-file only.
- `FastSamTextSegmenter` (class) — fastsam_segmentation.py:138 — Persistent FastSAM+CLIP segmenter (construct once, infer warm). — 3 callers: sam_worker.py:591, scripts/compare_sam3_fastsam.py:93, scripts/measure_vram.py:219.
- `FastSamTextSegmenter.__init__(weights, clip_model, ...)` — :142 — Lazy-loads torch/ultralytics/open_clip, builds FastSAM + CLIP. — via class instantiation (3 sites above).
- `FastSamTextSegmenter._run_fastsam(image_path, conf, iou, imgsz)` — :170 — One FastSAM forward → (masks N,H,W bool, boxes N,4). — in-file callers (`infer` :336, `infer_raw` :494) + scripts/compare_sam3_fastsam.py:96.
- `FastSamTextSegmenter._clip_scores(image_rgb, masks, text_prompt)` — :191 — Per-mask (softmax_prob, raw_cosine) vs prompt via CLIP on whited-bg crops. — in-file (`infer` :343, `infer_raw` :495) + scripts/compare_sam3_fastsam.py:98.
- `FastSamTextSegmenter._split_into_components(masks, min_area_px)` — :238 — Split each instance mask into connected components so CLIP scores each object. — 1 caller in-file (`infer` :341). No external refs.
- `FastSamTextSegmenter._save_components_debug(...)` — :267 — Best-effort debug overlay + JSON sidecar of top-K components. — 1 caller in-file (`infer` :435). No external refs.
- `FastSamTextSegmenter.infer(...)` — :314 — Full pipeline: fastsam → split → CLIP → filter/threshold → dedup → write masks + `_sam3_results.json`. — sam_worker.py:622, scripts/measure_vram.py:229, in-file `_main` :623.
- `FastSamTextSegmenter.infer_raw(...)` — :478 — Write all masks ≥ min_score to `_raw_masks.npz` (no other filters). — sam_worker.py:645.
- `run_fastsam_subprocess(image_path, text_prompt, output_dir, output_stem, ...)` — :526 — Launch fastsam infer in sam3 env subprocess; returns SAM3-shaped objects. — live_session.py:811, phase0.py:557 (imported phase0.py:42, live_session.py:49).
- `_parse_args()` — :588 — CLI args for the worker. — 1 caller in-file (`_main` :615).
- `_main()` — :614 — CLI entry: build segmenter, run infer, write timing sidecar. — `__main__` block :648 (subprocess entry point).

### sam3_segmentation.py

- `_resolve_env_python(conda_env)` — sam3_segmentation.py:38 — Env python resolver. — 1 caller in-file (`run_sam3_subprocess` :268).
- `_compute_iou(mask_a, mask_b)` — sam3_segmentation.py:50 — IoU of two binary masks. — in-file (`run_sam3_segmentation` :189) + sam_worker.py:173/226 (`_seg._compute_iou`).
- `_touches_n_borders(mask, n=2)` — sam3_segmentation.py:59 — Border-touch test. — in-file (:165) + sam_worker.py:174/215.
- `load_sam3_masks(results_json_path)` — sam3_segmentation.py:74 — Load `objects` list from the worker summary JSON. — utils/__init__.py:25, fusion/__init__.py:10, phase0.py:41/551, in-file (`run_sam3_subprocess` :315).
- `run_sam3_segmentation(image_path, text_prompt, ...)` — sam3_segmentation.py:86 — The SAM3 worker: build model, infer, filter, dedup, write masks + summary, free GPU. — 1 caller in-file (`_main` :341). Reachable only via the subprocess CLI; NOT imported elsewhere (sam_worker reimplements inline — see sam_worker.py:167 comment).
- `run_sam3_subprocess(image_path, text_prompt, ...)` — sam3_segmentation.py:250 — Launch SAM3 worker in sam3 env subprocess. — utils/__init__.py:25/72, fusion/__init__.py:10/45, phase0.py:41/575, live_session.py:48/812.
- `_parse_args()` — sam3_segmentation.py:323 — CLI args. — 1 caller in-file (`_main` :339).
- `_main()` — sam3_segmentation.py:338 — CLI entry → `run_sam3_segmentation`. — `__main__` block :359 (subprocess entry point).

### esam.py

- `_to_mask_numpy(mask)` — esam.py:28 — Tensor mask → bool numpy (>0.5), drops trailing singleton. — 1 caller in-file (`compute_prompt_interior` :36). No external refs.
- `compute_prompt_interior(mask, keep_ratio)` — esam.py:35 — Bbox-restricted EDT → interior keep-mask + distance map. — in-file: `_run_esam_query` :140, `query_esam_mask_pair` :299.
- `sample_interior_points(inner_mask, distance_map, num_points)` — esam.py:65 — Farthest-point-style interior point selection for the prompt. — in-file: `_run_esam_query` :141, `query_esam_mask_pair` :300.
- `ensure_esam_checkpoint(checkpoint_path)` — esam.py:97 — Download the ESAM ViT-Tiny checkpoint if absent. — 1 caller in-file (`build_esam_ti` :112). No external refs.
- `build_esam_ti(device, checkpoint_path)` — esam.py:105 — Build EfficientSAM-Tiny, move to device, eval. — utils/__init__.py:17/50, dynamic_gs_model.py:30/1911.
- `_select_esam_mask(predicted_logits, predicted_iou, prompt_region)` — esam.py:121 — Pick best candidate mask (by predicted IoU, else prompt overlap). — 1 caller in-file (`_run_esam_query` :152). No external refs.
- `_run_esam_query(model, image_tensor, prompt_region, ...)` — esam.py:133 — Single-image ESAM forward → (mask, inner_mask, points). — 1 caller in-file (`query_esam_mask` :202). No external refs.
- `query_esam_mask(model, rendered_rgb, change_mask, ...)` — esam.py:156 — Single-image public ESAM query (down/upsample wrapper). — Imported (utils/__init__.py:18/67, dynamic_gs_model.py:40) but **never CALLED** (no `query_esam_mask(` anywhere; only the `_pair` variant is invoked at dynamic_gs_model.py:2016).
- `query_esam_mask_pair(model, rendered_rgb_a, rendered_rgb_b, change_mask, ...)` — esam.py:236 — Batched 2-image ESAM (shared prompt) → both results. — dynamic_gs_model.py:2016 (imported :41, __init__.py:19/68).
- `_pick(idx)` — esam.py:319 (nested in `query_esam_mask_pair`) — Pick best mask for batch index idx. — 2 in-function calls (:328/:329). Duplicates `_select_esam_mask` logic inline.

---

## 2) DEAD-CODE CANDIDATES

Entry points excluded (`_main`/`__main__`, subprocess CLI workers, `*_subprocess` launchers, public re-exports via `utils/__init__.py` are treated as borderline, not auto-dead).

| Symbol | file:line | Ref evidence | Confidence |
|---|---|---|---|
| `query_esam_mask` (single-image) | esam.py:156 | Imported in dynamic_gs_model.py:40 and re-exported utils/__init__.py:18, but `grep "query_esam_mask("` → only the def. The live path calls `query_esam_mask_pair`. | medium (it is imported, so an import-cleanup linter wouldn't flag it; but it is genuinely never invoked — the pair variant superseded it and duplicates its body) |
| `_run_esam_query` | esam.py:133 | Only caller is `query_esam_mask` (:202), itself uncalled. Transitively dead. | medium |
| `_select_esam_mask` | esam.py:121 | Only caller is `_run_esam_query` (:152), transitively dead; logic re-implemented inline as `_pick` in the live `_pair` path. | medium |
| `_compute_iou` (fastsam copy) | fastsam_segmentation.py:49 | `grep _compute_iou` shows only the sam3_segmentation copy is consumed (sam_worker.py:173/226). This fastsam definition has 0 references — fastsam's `infer` dedup uses an inline IoU (:411), not this helper. | high |

Notes:
- `run_sam3_segmentation` (sam3_segmentation.py:86) is NOT dead — it is the body of the SAM3 subprocess CLI (`_main` :341). sam_worker reimplements the same logic inline (sam_worker.py:167 comment) rather than importing it, so the import-graph shows 0 cross-module refs, but the `__main__` subprocess path keeps it live.
- `infer_raw` (fastsam, :478) is reachable via `sam_worker.fastsam_infer_raw` → `worker.fastsam_infer_raw` RPC, but no in-repo CLIENT calls `fastsam_infer_raw` (`grep` of the client method returns nothing outside sam_worker.py). The SAM3 equivalent `sam3_infer_raw` IS called (preseg_seed.py:451). So the fastsam raw path is a wired-but-currently-unexercised RPC — flagged under smells, not dead (worker-dispatch entry point).

---

## 3) DATA-LIFECYCLE

These three modules carry NO `.pt` warm-cache, NO SHM, and NONE of the 4 identity buffers (`object_flags`/`object_instance_ids`/`sam3d_init_target_flags`/`inserted_flags` — all invariant-protected, owned by phase0/model, not touched here). Lifecycle surface is models, GPU tensors, file artifacts, and subprocess handles.

**fastsam_segmentation.py**
- Model load: `FastSamTextSegmenter.__init__` (:151-166) loads FastSAM + CLIP onto GPU. NO explicit `del`/`empty_cache` / `unload` method on the class — relies on the holder (`SamWorkerClient.unload_fastsam`, sam_worker.py:597-599 sets `self.fastsam_seg = None`) for the free. Inside the standalone CLI `_main` the process exits, so no leak there. **Persistent-worker path: freeing is delegated to sam_worker; the class itself has no teardown — fine as long as the holder always nulls it.**
- File artifacts written by `infer`: per-object `*_obj_NN_mask.png` (:444), `*_sam3_results.json` summary (:457), debug `*_fastsam_components.png` + `.json` (:293/:308). `infer_raw` writes `*_raw_masks.npz` (:513). Files overwrite same-stem on re-run (idempotent by stem). No cleanup of stale higher-index masks if a re-run yields fewer objects — old `_obj_05_mask.png` from a prior run can linger (LOW; downstream reads from the summary JSON `objects` list, not by globbing).
- GPU tensors in `_clip_scores` (:223-232) are scoped under `torch.no_grad()` and converted to numpy before return — no retained graph.

**sam3_segmentation.py**
- Model load + EXPLICIT free: `run_sam3_segmentation` builds the model (:119) and at :232-240 does `del model, processor; gc.collect(); torch.cuda.empty_cache()`. Clean. (This is the subprocess path; process exit would free anyway, but explicit is correct.)
- File artifacts: `*_obj_NN_mask.png` (:203) + `*_sam3_results.json` (:228). Same stale-higher-index caveat as fastsam (LOW).
- Subprocess handle: `run_sam3_subprocess` uses `subprocess.run(..., capture_output=True)` (:295) — blocking, fully reaped, no orphan. Same for `run_fastsam_subprocess`.

**esam.py**
- Model load: `build_esam_ti` (:105) downloads (`ensure_esam_checkpoint` :97, urlretrieve to `~/.cache/efficient_sam/`) and builds on device. Cached on the model side (`dynamic_gs_model._esam_model`, lazy at :1911); never explicitly freed during a run — lives for the dynamic phase (expected, single-load).
- GPU tensors: all forwards under `torch.no_grad()` (:149, :314). Interpolations + masks are per-call, no accumulation.

**Save/load FORMAT contract (cross-module) — flag:**
- FastSAM and SAM3 both emit `<stem>_sam3_results.json` with an `objects` list of `{mask_path, score, bbox, mask_area, object_index}`, consumed by `load_sam3_masks` (sam3_segmentation.py:74) + phase0. **Shape/semantics mismatch risks:**
  - `score`: SAM3 writes the model confidence (sam3 :168); FastSAM writes a survivor-softmax PROB (fastsam :384, `surv_score`) — different scales/meaning under the same key. Any downstream that thresholds on `score` numerically behaves differently per backend. (MEDIUM)
  - `bbox`: SAM3 stores the model box (xyxy float, sam3 :172); FastSAM stores `boxes[i]` which after `_split_into_components` is the COMPONENT bbox as `[x0,y0,x1,y1]` int (fastsam :262). Format aligns (xyxy) but provenance differs. (LOW)
- `infer_raw` NPZ keys `{masks(bool), scores(f32), boxes(f32)}` match between fastsam (:513) and the preseg expectation (preseg_seed.py path). OK.

No double-loads, no SHM, no identity-buffer desync in these files.

---

## 4) DESIGN SMELLS

- **God function: `FastSamTextSegmenter.infer`** (fastsam_segmentation.py:314-476, ~160 lines) — does fastsam inference, component split, CLIP scoring, area/border filter, auto-threshold, legacy floor, containment dedup, debug dump, mask writes, AND summary-JSON authoring. Multiple concerns; the dedup and threshold blocks are independently testable but inlined. (MEDIUM)
- **Duplicated helpers across modules** — `_compute_iou` and `_touches_n_borders` exist verbatim in BOTH fastsam_segmentation.py (:49/:55) and sam3_segmentation.py (:50/:59). sam_worker imports the sam3 copies (sam_worker.py:173-174); the fastsam copies are redundant (fastsam `_compute_iou` is fully dead — see §2). (MEDIUM)
- **Duplicated ESAM mask-pick logic** — `_select_esam_mask` (esam.py:121) and the nested `_pick` (esam.py:319) implement the same argmax-IoU-else-overlap selection. The single-image path (`query_esam_mask`/`_run_esam_query`/`_select_esam_mask`) is entirely dead while the batched `_pair` path re-implements it inline. The dead single-image trio + duplicate `_pick` is ~80 lines of parallel maintenance. (MEDIUM)
- **`score` key overloaded across backends** (see §3) — same JSON field carries SAM3 confidence vs FastSAM survivor-softmax-prob; "byte-identical contract" is structurally true but semantically leaky. Misleading for any numeric `score` consumer. (MEDIUM)
- **Default `min_score` divergence** — `run_sam3_segmentation` default `min_score=0.2` (sam3 :96) but its CLI `_parse_args` default is `0.44` (sam3 :334). So calling the function directly vs via subprocess yields different filtering. FastSAM keeps `0.2` consistent (fastsam :324/:598). The 0.2/0.44 split is an easy footgun. (MEDIUM)
- **Wired-but-unexercised RPC** — `FastSamTextSegmenter.infer_raw` → `sam_worker.fastsam_infer_raw` has no in-repo client caller (only the SAM3 `sam3_infer_raw` is used by preseg_seed.py:451). Dead-ish plumbing kept for backend parity. (LOW)
- **Swallowed exceptions** — `_save_components_debug` JSON sidecar wrapped in bare `except Exception: pass` (fastsam :309); the debug-dump call wrapped in `except Exception as _exc: print(...)` (fastsam :438); timing sidecar `except Exception: pass` (fastsam :637). All debug/best-effort, acceptable, but the silent `pass` at :309 hides real bugs in the sidecar writer. (LOW)
- **Backward-compat dead-ish field: `min_score` AND-guard** (fastsam :374-375) — with `auto_threshold=True` (the default) the auto gate "already owns the decision" per the comment, yet the `min_score=0.2` floor still re-filters on the survivor-softmax prob, which can silently drop a valid lone object whose re-softmax prob < 0.2. Config field that subtly fights the auto path. (MEDIUM)
- **Params threaded through many layers** — the `auto_*` / `fastsam_*` / `min_area_ratio` set is threaded `infer` → `select_kept_indices`, and separately enumerated three times: `infer` signature (:314), `run_fastsam_subprocess._cli_keys` whitelist (:554), and `_parse_args` (:594). Adding one knob requires editing all three or it is silently dropped by the subprocess whitelist. (MEDIUM)
- **`_resolve_env_python` duplicated** in fastsam_segmentation.py:41 and sam3_segmentation.py:38 (and per comments in sam3d.py / sam_worker.py / anysplat_decode.py) — same 4-line resolver copy-pasted across the codebase. (LOW)
