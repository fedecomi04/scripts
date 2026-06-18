# Duplication audit — Subprocess / worker invocation & atomic-write boilerplate

Scope: cross-env subprocess spawns (conda-env-python-direct), env-var hardening
(`CONDA_PREFIX` / `PYTHONNOUSERSITE` / `LD_LIBRARY_PATH`), the segmentation
subprocess driver shape, and atomic `transforms.json` writes (tmp + `os.replace`).

Method: grepped `dynamic_gs/` + `scripts/` for `os.replace`, `CONDA_PREFIX`,
`PYTHONNOUSERSITE`, `conda run`, `subprocess.run/Popen`, `_resolve_env_python`,
and the depth/rgb/mask disk-write idioms, then read each hit to confirm the
logic is genuinely the same (not a name coincidence). Conventions respected:
depth is uint16 mm on disk (`depth_unit_scale_factor = 1e-3`), RGB written BGR
via cv2, masks uint8 keep — any consolidating helper must preserve those.

---

## Pattern 1 — Atomic `transforms.json` rewrite (tmp + write_text + os.replace)

The exact 3-line idiom
`tmp = tp.with_name(f".{tp.name}.tmp"); tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8"); os.replace(tmp, tp)`
is copy-pasted verbatim across 7 sites. It writes the Nerfstudio `transforms.json`
metadata dict atomically so the dataparser/fusion watcher never reads a torn file.

Confirmed identical sites (all write `meta` dict → `transforms.json`):
- `dynamic_gs/utils/fusion_runner.py:335-337` (finalize, sets `ply_file_path`)
- `dynamic_gs/utils/online_fusion.py:627-630` (seed finalize, sets `ply_file_path`; locally re-imports `os as _os`)
- `dynamic_gs/utils/live_session.py:295-297` (`_append_anchor_as_static_keyframe`)
- `dynamic_gs/utils/live_session.py:444-446` (`ply_file_path` point-at-seed rewrite)
- `dynamic_gs/utils/live_ros_publisher.py:1148-1150` (`_write_frame_to_disk` per-frame append, under `_record_lock`)
- `dynamic_gs/utils/live_ros_publisher.py:1225-1227` (`build_static_init_pointcloud`, sets `ply_file_path`)
- `scripts/capture_only.py:125-127` (recorded dynamic transforms append)
- `scripts/test_concurrent_fusion.py:76-78` (test fixture — lower priority but identical)

Risk: **correctness.** Every site independently re-implements an atomicity
contract. If one is edited to drop the `.tmp` indirection (or use a non-atomic
`json.dump(open(...))`), the fusion watcher in `fusion_runner.py` (which polls
`transforms.json` on a thread) can read a half-written file → bad/empty seed.
They can also silently drift on the `indent=2 + "\n"` / encoding formatting,
making byte-comparison tooling (e.g. anchor-keyframe round-trip checks) flaky.

Proposed helper: `atomic_write_transforms(path: Path, meta: dict) -> None`
in a new `dynamic_gs/utils/io_atomic.py` (must be import-light — `live_ros_publisher`
runs in the minimal ROS py3.8 env and imports it; keep it stdlib-only: `os`, `json`,
`pathlib`). Note `live_ros_publisher` is loaded as a standalone script in
`dynamic_gs_ros`, so the helper module it imports must not pull `nerfstudio`/torch.

Est. LOC saved: ~14 (2 lines net saved per site × 7).

---

## Pattern 2 — Cross-env conda subprocess env hardening (`LD_LIBRARY_PATH` + `PYTHONNOUSERSITE` [+ CONDA_PREFIX/CUDA_HOME])

The "build the subprocess env that points at a sibling conda env's native libs"
block is re-implemented 4 times with the same core
(`env = os.environ.copy(); env["LD_LIBRARY_PATH"] = (str(<prefix>/lib) + ":" + env.get("LD_LIBRARY_PATH","")).rstrip(":"); env["PYTHONNOUSERSITE"] = "1"`)
plus per-site extras (`CONDA_PREFIX`, `CUDA_HOME`, `PYTHONUNBUFFERED`):

- `dynamic_gs/utils/sam3d.py:54-63` (`_sam3d_subprocess_env`: + `CONDA_PREFIX` + `CUDA_HOME`)
- `dynamic_gs/utils/sam3_segmentation.py:289-293` (inline in `run_sam3_subprocess`)
- `dynamic_gs/utils/fastsam_segmentation.py:564-569` (inline in `run_fastsam_subprocess`: + `setdefault CONDA_PREFIX`)
- `dynamic_gs/utils/sam_worker.py:753-759` (`SamWorkerClient.__init__`: + `PYTHONUNBUFFERED` + `PYTHONPATH`)
- `dynamic_gs/utils/anysplat_decode.py:374-376` (`run_anysplat_subprocess`: + `PYTHONUNBUFFERED`, no NOUSERSITE)

The comments at each site explicitly cross-reference each other ("mirroring
sam3_segmentation / sam_worker", "Same trick as anysplat_decode", "same hardening
as sam_worker / anysplat") — confirming these are deliberate copies of one recipe.

Risk: **correctness / maintenance.** The CLAUDE.md ROS-env note shows this class
of bug is real (user-local site-packages shadowing the env). If the recipe needs
a fix (e.g. a new env var to fully isolate native libs), it must be applied in 5
places; missing one reintroduces the shadowing/`KeyError CONDA_PREFIX` crash that
`sam3d.py` documents.

Proposed helper:
`conda_subprocess_env(env_prefix: Path, *, cuda_home=False, conda_prefix=False, pythonpath: list[Path]|None=None, unbuffered=False) -> dict[str,str]`
in a new `dynamic_gs/utils/conda_env.py` (stdlib-only). Each call site picks the
flags it needs (sam3d → cuda_home+conda_prefix; sam_worker → pythonpath+unbuffered).

Est. LOC saved: ~15 (the boilerplate collapses to one call per site; net ~3/site).

---

## Pattern 3 — `_resolve_env_python(conda_env)` duplicated definition

Byte-for-byte the same function defined twice:
- `dynamic_gs/utils/sam3_segmentation.py:38-42`
- `dynamic_gs/utils/fastsam_segmentation.py:41-43`

Both: `py = _CONDA_ROOT / "envs" / conda_env / "bin" / "python"; return py if py.exists() else None`.
`sam3d.py:32-41` (`_resolve_sam3d_python`) and `sam_worker.py:742-743` /
`anysplat_decode.py:358-361` are the same `<prefix>/envs/<env>/bin/python` resolution
with slightly different return contracts (str vs Path vs raise-on-missing).

Risk: **maintenance.** Hardcoded `_CONDA_ROOT = /home/mrc-cuhk/miniconda3` (and
`Path.home()/"miniconda3"` in anysplat) is repeated; a conda-root move requires
editing every copy. Low correctness risk since they all resolve to the same path
today.

Proposed helper: `resolve_env_python(conda_env: str) -> Path | None` in the same
new `dynamic_gs/utils/conda_env.py` (single source for `_CONDA_ROOT`). Callers
that want the raise-on-missing behavior wrap it.

Est. LOC saved: ~6.

---

## Pattern 4 — Segmentation subprocess driver (SAM3 / FastSAM) near-identical

`run_sam3_subprocess` (`sam3_segmentation.py:258-315`) and `run_fastsam_subprocess`
(`fastsam_segmentation.py:529-585`) are structurally the same driver: resolve env
python → build `[python, __file__]` (else `conda run --no-capture-output` fallback)
→ append `--image/--text-prompt/--output-dir/--output-stem` + forward `filter_kwargs`
as `--k-ebab` flags → build hardened `sub_env` (Pattern 2) → `subprocess.run(cwd=parents[2],
capture_output=True, text=True)` → raise with STDOUT/STDERR on nonzero → read the
SAME `f"{output_stem}_sam3_results.json"` summary. They even share the output-contract
docstring ("`{mask_path, score, bbox, mask_area, object_index}`").

The two differ only in: which keys are forwarded (fastsam filters to `_cli_keys`),
and the final parse (`load_sam3_masks` vs `json.loads(...).get("objects", [])`).

Risk: **maintenance.** This is what backs CLAUDE.md's "Phase0a + live_session
branch on the backend; `sam3_reuse_cached` is backend-aware" — the two drivers must
stay in lockstep on the CLI flag contract and the results-JSON filename. They are
already documented as the byte-identical-output-contract pair.

Proposed helper:
`run_segmentation_subprocess(worker_file, image_path, text_prompt, output_dir, output_stem, conda_env, *, allowed_cli_keys=None, **filter_kwargs) -> Path`
returning the summary JSON path, in `dynamic_gs/utils/seg_subprocess.py`. Each backend
keeps its own thin wrapper that calls it then parses the summary its own way.

Est. LOC saved: ~35 (the ~45-line driver body collapses to one shared call + a
~5-line wrapper each).

---

## NOT duplication (verified, excluded)

- **`SamWorkerClient.load_*/unload_*/*_infer` wrappers** (`sam_worker.py:132-660`
  worker side, `825-964` client side): these are thin typed methods over a single
  shared `_request(cmd, **kwargs)` (`sam_worker.py:794`). That is the consolidated
  form already — not cross-file duplication. The CLAUDE.md invariant #7 names this
  the canonical path; leave it.
- **ROS publisher spawn** (`live_shm_reader.py:210-224`): bash-wrapped + ROS-sourced
  + deliberately STRIPS `LD_LIBRARY_PATH/CPATH/CUDA_HOME` (opposite of Pattern 2,
  for documented libstdc++ reasons). Related theme, genuinely different logic — do
  not fold into Pattern 2's helper.
