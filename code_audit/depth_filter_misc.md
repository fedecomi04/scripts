# Code Audit — depth_filter.py / object_picker.py / timing_ledger.py

Repo: `/home/mrc-cuhk/Documents/dynamic_gaussian_splat/scripts`
Auditor pass: adversarial. LIVE-PATH priority on `depth_filter.py`.
All caller counts from:
`grep -rn "<name>" dynamic_gs scripts --include=*.py`

---

## 1) FUNCTION / CLASS MAP

### `dynamic_gs/utils/depth_filter.py`

- **module-level env knobs** — `depth_filter.py:29-33` — `_MED_K`, `_BI_D`, `_BI_SIGMA_COLOR_M`, `_BI_SIGMA_SPACE` read **once at import** from `DGS_DEPTH_*`. Used by both `filter_depth` and `filter_depth_torch`. **2 internal use sites each.** NOTE: import-time read means a per-run env A/B requires a fresh process (cannot toggle mid-run); contrast with `enabled()` which reads env each call.

- `enabled() -> bool` — `depth_filter.py:36` — returns `DGS_DEPTH_FILTER != "0"`; the global on/off gate, read every call. **Callers (4): `dynamic_gs_pipeline_recorded.py:195`, `dynamic_gs_pipeline_base.py:3248`, `live_ros_publisher.py:1129`, plus self-calls inside `filter_depth`/`filter_depth_torch` (`:47`, `:99`).**

- `filter_depth(depth_m: np.ndarray) -> np.ndarray` — `depth_filter.py:41` — CPU cv2 median(5)+weight-corrected bilateral on a metres float32 depth map; returns a new array, holes held out. **Callers (1): `live_ros_publisher.py:1133`** (loaded via importlib as `_DEPTH_FILTER` in the ROS publisher subprocess). Also docstring-referenced in `dynamic_gs_pipeline_base.py:330`.

- `filter_depth_torch(depth_t, *, median=True, bilateral=True)` — `depth_filter.py:87` — GPU torch median and/or bilateral; matches the cv2 path within depth-quant. **Callers (3): `dynamic_gs_pipeline_recorded.py:196`, `dynamic_gs_pipeline_base.py:3253`, `scripts/compare_depth_filters_zed.py:36`.** All three production/diagnostic call sites pass **no median/bilateral kwargs** (full filter); only the bench (`compare_depth_filters_zed.py:36`) ever exercises the split — see Smell DF-1.

### `dynamic_gs/utils/object_picker.py`

- `class ObjEntry` (dataclass: `instance_id`, `mask`, `score`, `color_bgr`) — `object_picker.py:33` — one selectable object for the picker overlay. **Constructed internally `:101`, `:139`; type-annotation use `dynamic_gs_pipeline_base.py:435` (comment only — `self._sam3_objects` holds `[ObjEntry]`).** No external `import ObjEntry`.

- `_hue_color_bgr(i, n) -> tuple` — `object_picker.py:42` — evenly-spaced HSV→BGR overlay color. **Callers (2, internal): `:105`, `:143`.**

- `_find_sam3d_artifacts(data_dir) -> Path|None` — `object_picker.py:51` — locate `static0_sam3_results.json` for the SAM3D init path. **Callers (1, internal): `:112`.**

- `_find_preseg_artifacts(data_dir) -> Path|None` — `object_picker.py:70` — locate the preseg `_sam3_prompt_00_raw_masks.npz` dir. **Callers (1, internal): `:89`.**

- `load_sam3_objects(data_dir) -> (rgb_uint8, [ObjEntry]) | None` — `object_picker.py:78` — auto-detect init path (preseg preferred, then SAM3D), load image + per-object masks/scores. **Callers (1): `dynamic_gs_pipeline_base.py:1243`** (imported `:1238`).

- `render_picker_overlay(image_rgb, entries, *, alpha=0.55) -> np.ndarray` — `object_picker.py:154` — blend masks + draw id labels for `server.gui.add_image`. **Callers (1): `dynamic_gs_pipeline_base.py:1247`** (imported `:1240`).

### `dynamic_gs/utils/timing_ledger.py`

- **constants** `LEDGER_FILENAME` (`:28`), `PHASE_ORDER` (`:32`), `PHASE_LABELS` (`:42`) — used **only inside `render()`/`ledger_path()`**. **NO external REFS FOUND** (`grep PHASE_ORDER|PHASE_LABELS|LEDGER_FILENAME` outside the module = 0 hits).

- `ledger_path(data_root) -> Path` — `timing_ledger.py:56` — JSONL path under data root. **Callers: internal only (`record:79`, `reset:63`, `render:108`).** NO external REFS FOUND.

- `reset(data_root)` — `timing_ledger.py:60` — delete the ledger at run start. **Callers (4): `static_gs_pipeline.py:122`, `dynamic_gs_pipeline_base.py:390`, `live_session.py:619`** (via `_tl.reset`).

- `record(data_root, phase, op, kind, t_start, t_end, **meta)` — `timing_ledger.py:68` — append one timing row; swallows all write errors. **Callers (many, ~20): `static_gs_pipeline.py:215,341`; `dynamic_gs_pipeline_base.py:569,2020,2927`; `fusion/phase0.py:614,615,618,712,713,714,1156`; `live_session.py:661,673,720,731,834,881,955,1004,1132`.**

- `timed(data_root, phase, op, kind, **meta)` (contextmanager) — `timing_ledger.py:86` — `with`-wraps a block and records its wall as work. **NO REFS FOUND** outside the module (`_tl.timed` / `.timed(` = 0 external hits). Every call site uses the raw `record()` two-stamp form instead. See Dead-code DL-1.

- `_fmt(seconds) -> str` — `timing_ledger.py:96` — ms/s pretty-print. **Callers (internal, many in `render`): `:183,186,189,191,195`.**

- `render(data_root, extra=None) -> str` — `timing_ledger.py:100` — render the by-phase load/infer report. **Callers (4): `static_gs_pipeline.py:345`, `dynamic_gs_pipeline_base.py:892`, `live_session.py:1192`** (via `_tl.render`).

---

## 2) DEAD-CODE CANDIDATES

- **DL-1 `timed()` contextmanager** — `timing_ledger.py:86` — **confidence: medium.** Zero external references: `grep "_tl.timed\|\.timed("` over `dynamic_gs`+`scripts` = 0 hits outside the module. Every timing site uses `_tl.record(...)` with explicit `t0`/`time.time()` stamps instead. It is a clean, advertised public helper (the module docstring/examples reference `with timed(...)`), so it may be intended ergonomic API — but as of this tree it is unused. Low risk to drop; flagged, not removed.

- **DF-1 `filter_depth_torch` `median=`/`bilateral=` keyword params** — `depth_filter.py:87` — **NOT dead as a function, but the parameters are dead-on-arrival.** All 3 production/bench callers (`dynamic_gs_pipeline_recorded.py:196`, `dynamic_gs_pipeline_base.py:3253`, `compare_depth_filters_zed.py:36`) — the two pipeline callers pass **no kwargs** (default full filter); only the bench passes them. The docstring's described split ("tracker runs median-only … FF adds only bilateral") has **no production caller** that uses it. CLAUDE.md explicitly confirms: "`filter_depth_torch(median=, bilateral=)` allows a per-stage split (unused — all callers run the full filter)." Confidence: high that the split is currently unexercised in the live/recorded paths.

- **No other dead candidates.** `_hue_color_bgr`, `_find_sam3d_artifacts`, `_find_preseg_artifacts`, `ObjEntry`, `ledger_path`, `_fmt`, the env-knob constants are all reachable internally; the public `enabled`/`filter_depth`/`filter_depth_torch`/`load_sam3_objects`/`render_picker_overlay`/`reset`/`record`/`render` all have external callers. `PHASE_ORDER`/`PHASE_LABELS`/`LEDGER_FILENAME` are internal config consumed by `render()` (not dead, just not externally referenced).

---

## 3) DATA-LIFECYCLE

### depth_filter.py — per-tick GPU/heap allocation (LIVE-PATH, HOT)
- **No persistent state, no `.pt`, no SHM, no file/process handles owned here.** Pure functions. Good — no leaks possible at this layer.
- **DL-A (per-tick GPU churn).** `filter_depth_torch` (`:87`) is called every FF tick (`dynamic_gs_pipeline_base.py:3253`, live only) and every recorded tracker tick (`dynamic_gs_pipeline_recorded.py:196`). Each call allocates multiple full-frame intermediates: `F.unfold` → `(1, k*k, H*W)` then `.sort()` (median, `:124-125`) and `(1, d*d, H*W)` for bilateral (`:142-143`), plus `rng`, `w`, `num`, `den`. At 1920×1200, d=5 the bilateral unfold tensor is `1×25×2.3M ≈ 57.6M floats ≈ 230 MB`, sort/median similar. These are transient (freed when the function returns) — **not a leak** — but they are sizeable per-tick allocations on the **shared** GPU that the tracker tick, the FF bg thread, and the viser render all contend for. On the live FF bg thread this is "free" wall-time (off the tracker), but the VRAM spike co-resides with the tracker's own render + the growing scene. Worth noting given the documented 1200p VRAM pressure (459k→1.29M gaussians). No fix needed for purge, but flag the allocation size.
- **DL-B (thread-safety — benign).** `filter_depth`/`filter_depth_torch` read the module-level `_MED_K`/`_BI_D`/… constants (set once at import) and call `enabled()` (env read). No shared mutable state, no in-place mutation of caller buffers (`copy=True` at `:51`; `.float()`/`.view` produce new tensors at `:106-108`). Safe to call concurrently from the tracker thread, the FF bg thread, and the publisher process. The only cross-thread concern is the **input tensor aliasing** the model's depth — but both call sites assign the result back (`batch["depth_image"] = …`), and the function never mutates the input in place, so a concurrent reader of the old `batch["depth_image"]` sees a consistent array. OK.
- **DL-C (publisher-side, separate process).** `live_ros_publisher.py:211` loads this module via `importlib` as `_DEPTH_FILTER` once at module import in the `dynamic_gs_ros` (py3.8) env; `filter_depth` is then called in `_write_frame_to_disk` (`:1133`) on the publisher worker thread. RAW depth is preserved to `depth_raw/` (`:1130-1132`) and the filtered uint16 written to `depth/`. No handle leak (cv2.imwrite). One concern: the filter runs **inside the frame-write critical path** on the publisher worker — at ~60 ms/frame CPU it can backpressure the write queue under fast sweeps, but that is the publisher's design, not a lifecycle bug.

### object_picker.py — read-only artifact loading
- **DL-D.** `load_sam3_objects` (`:78`) only **reads** on-disk artifacts (npz via `np.load`, PNGs via `cv2.imread`, JSON via `read_text`). `np.load(...)` at `:92` returns an `NpzFile` whose handle is **not explicitly closed** — for an in-memory `.npz` (default `mmap_mode=None`) the arrays are materialized eagerly (`np.asarray(...)` at `:93-94`), so the file handle is released at GC; not a true leak but not closed deterministically. Called at most a handful of times (cached: `self._sam3_objects is None` guard at `dynamic_gs_pipeline_base.py:1236`), so impact is nil. The result is cached on the pipeline (`self._sam3_objects`) — masks are full-frame bool arrays kept resident for the run; for many objects at 1200p that is a few MB, acceptable.
- **Identity-buffer coupling (invariant-protected).** The module's whole contract is `instance_id == sam3_mask_index + 1 == object_instance_ids` (docstring `:6-16`), feeding `object_flags = (object_instance_ids == picked_id)` in `_reseed_tracked_object`. This is invariant-protected (Per-object identity buffers; `object_instance_ids` written by Phase-0b only, `object_flags` by D0 selection). The picker only **reads** ids to drive selection; it does not write any of the 4 buffers. No desync introduced here. **Reachability:** the entire picker path is gated by `config.interactive_object_selection` (default `False`, `dynamic_gs_pipeline_base.py:152`) — in a default live run this code is **never reached**.

### timing_ledger.py — append-only JSONL, cross-process
- **DL-E (lifecycle correct).** State is one append-only JSONL under the data root. `reset()` (`:60`) unlinks at run start (3 callers, one per process entry). `record()` (`:68`) opens/append/closes per row via `with open(...)` (`:79`) — no handle leak; **all write errors swallowed** (`except Exception: pass`, `:81-82`) by design ("timing must never break the pipeline"). `render()` (`:100`) reads the whole file with `read_text().splitlines()`. No save/load shape mismatch — rows are self-describing dicts; `render` defends missing keys with `.get(...)` (`:124-125`, `:160`) and `try/except` around `json.loads` (`:115-118`).
- **DL-F (cross-process append race — benign-by-design).** Three processes (live_session, ns-train static-gs, dynamic-gs-live) append to the **same** JSONL concurrently (CLAUDE.md: "Rows from ALL processes … land in one JSONL"). POSIX `O_APPEND` writes under the OS write-buffer size are atomic, and each `json.dumps(row)+"\n"` is small, so interleaving is line-safe in practice; `render` already skips unparseable lines (`:115-118`). Not a correctness bug, but it relies on small-write atomicity, not on a lock. Worth a one-line note if this ever writes very large `meta`.
- **DL-G.** No GPU tensors, no SHM, no `.pt` touched in this module.

---

## 4) DESIGN SMELLS

- **DF-1 (dead/misleading API surface) — `filter_depth_torch` per-stage split.** `depth_filter.py:87-98`. The `median=`/`bilateral=` params + the long docstring describing a tracker-median-only / FF-bilateral-only policy are not used by any production caller (all pass full filter). This is a leaky abstraction: a future reader will assume the documented split is wired up. Severity low (CLAUDE.md already flags it), but it is a maintenance trap on a module being purged. Recommend either wiring the split (median-only tracker) or deleting the params + docstring claim.

- **DF-2 (cv2-vs-torch parity is a hand-maintained invariant).** `depth_filter.py:117-128` carries a long comment justifying replicating `cv2.medianBlur`'s "lower-median over the full window including stored 0s" behavior so the GPU path matches the A/B'd CPU path bit-for-near-bit. This is a genuine duplicated-logic hazard: the two implementations (`filter_depth` cv2 at `:64-74`, `filter_depth_torch` at `:124-150`) must stay numerically aligned but share no code; a tweak to one silently diverges from the operator-validated A/B. No automated parity check exists (only the ad-hoc `compare_depth_filters*.py` scripts). Severity medium given the live path.

- **DF-3 (import-time env capture vs per-call gate).** `depth_filter.py:29-33` read `_MED_K`/`_BI_D`/sigmas **once at import**, while `enabled()` (`:36`) reads `DGS_DEPTH_FILTER` **per call**. Inconsistent: the master toggle is hot-swappable but the kernel sizes/sigmas are frozen at import. The module docstring says "Knobs are env-overridable for A/B (no relaunch)" — that is only true for the on/off toggle, not for the kernel/sigma knobs, which require a process restart. Misleading doc. Severity low–medium.

- **DF-4 (silent fallback swallows a real misconfig).** `object_picker.py:118-124` — in the SAM3D path, if `static0_rgb.png` fails to load it silently falls back to `meta["image_path"]`, and if that also fails returns `None`. Combined with the caller's broad `except Exception` (`dynamic_gs_pipeline_base.py:1249`) that downgrades to the heuristic fallback id, a genuinely broken/missing artifact set produces only a single log line, never an error — the operator may pick "the wrong object" without knowing the picker degraded. Severity low (picker is off by default).

- **DF-5 (duplicated mask-resize in render_picker_overlay).** `object_picker.py:170-171` and `:178-179` recompute the identical `m.shape != (H,W)` resize twice per entry (once for blending, once for label placement). Minor duplication; cheap since picker fires rarely. Severity low.

- **DF-6 (god-function `render`).** `timing_ledger.py:100-200` — one 100-line function does file read, JSON parse, phase ordering, per-(op,kind) aggregation, extra-folding, wall/idle computation, and string formatting. Hard to test in pieces. Acceptable for a report renderer but it is the heaviest function in the three files. Severity low.

- **DF-7 (no dead *Config fields here).** `depth_filter_enabled: bool = True` is declared on the pipeline config (`dynamic_gs_pipeline_base.py:327`) and **is** read (`dynamic_gs_pipeline_recorded.py:195`). The base FF-site branch (`dynamic_gs_pipeline_base.py:3248`) is gated by the instance attr `_filter_depth_at_ff`, set `True` only in live (`dynamic_gs_pipeline_live.py:117`) and `False` in base/recorded (`:499`) — so the **FF-site filter branch is unreachable in recorded mode** by design (recorded filters at the batch source instead, `:196`). Not a bug, but a branch that is dead in recorded and live in live — note for the purge so it isn't mistaken for unconditional. Severity low.
