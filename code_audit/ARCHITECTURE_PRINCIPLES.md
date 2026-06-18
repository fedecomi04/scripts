# Live-Path Architecture Principles — purge north star

Scoped to the dynamic-gs **live path** (publisher → SHM → tracker tick → CDN → FF-bg → viser).
Use this to judge every purge edit: *does it move the system toward these properties, or just delete lines on top of the current broken design?* The goal of the purge is not "fewer lines" — it is **predictable latency + correct concurrency**. Fewer lines is a side effect.

> **The reframe:** "speed" is the wrong target. The target is **bounded, predictable latency**. Hz (throughput) is the metric that matters *least* for teleop. Optimize the hierarchy below top-down.

## Priority hierarchy (resolve conflicts in this order)

1. **Concurrency correctness** — no torn reads / desync of shared state across the 3 threads.
2. **Bounded, predictable latency** — worst-case (p99) and a load-shedding policy, not mean Hz.
3. **Throughput (Hz)** — only after 1 & 2 hold.
4. **Steady-state memory / resource lifecycle** — no unbounded growth, no leaks.
5. **Fault isolation** — one component dying never silently kills or hangs the pipeline.

You spent effort on #3 while #1 and #2 were broken (the audit proved it). Re-architect, don't just trim.

---

## The principles (each maps to a real audit finding)

### 1. Concurrency by design, not by locking
- **Rule:** the render/FF threads must never touch state a writer is mutating. Prefer **atomic snapshot swap (double-buffer)** over guarding shared mutable tensors with locks.
- **Your reality:** H-CROP race (`_object_crop_bbox` reads `means`/`object_instance_ids` unlocked while FF-bg resizes them → hard crash); the 4 identity buffers + 6 gauss_params hand-resized in ~6 places.
- **Purge action:** route **all** gauss_params/buffer mutation through **one locked `gaussian_surgery` chokepoint** (`_subset_all_buffers`/`_grow_all_buffers` under `_model_lock`), then have viser/tracker read an **immutable snapshot** swapped atomically. See `RUNTIME_target_architecture.md`.
- [ ] Every `gauss_params`/identity-buffer access is under `_model_lock` OR reads a snapshot. No exceptions.

### 2. Make illegal states unrepresentable
- **Rule:** if four arrays must stay equal-length, they should be *one type* that can't be resized partially.
- **Your reality:** `gaussian_surgery` already **drifted** between StaticGSModel and DynamicGSModel; a single forgotten buffer silently desyncs identity from geometry (Inv #8 foot-gun).
- **Purge action:** one `GaussianSet` surgery API with an internal `assert all params+buffers share shape[0]`. Drift becomes impossible, not "remembered."
- [ ] There is exactly ONE implementation of param+buffer subset/grow, with a length invariant assert.

### 3. Bounded work per frame + explicit load-shedding
- **Rule:** decide *in advance* what to drop/degrade when behind. Never let work grow without a cap.
- **Your reality:** FF inserts grow 589k→3M gaussians (unbounded); good counter-example = publisher `queue(maxsize=4, drop-oldest)`.
- **Purge action:** cap cumulative FF gaussians (periodic dynamic-phase purge, never dropping `object_flags==1`); skip FF / coarsen CDN when a tick runs long. "If behind, do less" must be code, not hope.
- [ ] There is a hard ceiling on live gaussian count and a "skip-FF-if-behind" path.

### 4. Latency tail, measured continuously
- **Rule:** track p99/worst-case per stage, always-on, cheap. The mean lies.
- **Your reality:** cudnn.benchmark freeze (754 ms behind a 14 ms mean); `DGS_FF_DEBUG` I/O cost 140 ms/tick on the hot path.
- **Purge action:** keep the timing ledger emitting p90/p99 (not just mean); heavy tracing stays opt-in and **off the hot path**.
- [ ] No I/O, logging, or allocation on the tracker hot path. Debug paths are opt-in and off-thread.

### 5. Time is event-time, not `now()`
- **Rule:** reason about capture timestamps + explicit clock domains, never wall-clock `now()` for fusion/filtering.
- **Your reality:** KF wall-clock-dt detuning bug; the sim-time vs wall-time (RTF) confusion.
- **Purge action:** every LiveFrame carries a capture timestamp; any dt-dependent math (KF if re-enabled) uses it; document the clock domain at each boundary.
- [ ] No dt/velocity computation uses `time.time()` deltas; all use frame capture timestamps.

### 6. Steady-state resources + guaranteed release
- **Rule:** pre-allocate, reuse, run flat. Every resource has one owner and a release path **including on exception**.
- **Your reality:** `/dev/shm` AnySplat file leak (`pipeline_base.py:640` cleans a filename never written); `_anysplat_slot_lock` leaks forever if `Thread.start()` raises; gaussian growth.
- **Purge action:** `try/finally` around every lock/SHM/subprocess/temp-file; fix the tmpfs cleanup to target real filenames; bound all pools.
- [ ] Every lock/SHM/subprocess/temp-file acquire has a `finally` release. No per-frame allocation.

### 7. Fault isolation & supervision
- **Rule:** a subprocess/thread crash degrades, never hangs or silently dies.
- **Your reality:** FF can go silently dead for the rest of a run; `NameError phase0.py:661` aborts the live anchor_ref path.
- **Purge action:** timeouts on every subprocess/worker call; FF-thread exceptions logged + recovered (not swallowed); fail-safe defaults (no FF this tick > crash).
- [ ] No bare `except: pass`; worker calls time out; a dead FF thread is detected and reported.

### 8. Explicit, versioned contracts at boundaries
- **Rule:** the seams (SHM layout, `.pt` warm-cache, transforms.json) are contracts — validate + version them.
- **Your reality:** `.pt` cache is config-implicit → opaque tensor-shape traceback on drift; watcher-thread torn-read on atomic transforms.json writes.
- **Purge action:** stamp a config fingerprint into the `.pt`; on mismatch raise a clear "config changed, delete the cache" error; atomic write + complete-read discipline on transforms.json.
- [ ] The `.pt` snapshot is config-tagged and fails loudly+clearly on mismatch.

### 9. Single source of truth + narrow module seams
- **Rule:** one owner per piece of state; the 3 threads talk through narrow interfaces, ideally a dataflow graph of bounded queues.
- **Your reality:** `pipeline_base.py` is a 3553-LOC god-module tangling tracker/FF/viser/timing/lifecycle.
- **Purge action:** split into `feedforward_dispatcher` / `viser_bridge` / tracker core, each single-threaded internally, communicating via the snapshot + bounded queues.
- [ ] Each thread's logic lives in its own module with a documented interface to shared state.

---

## Every purge edit must pass
- [ ] Does it touch shared state? → it goes through the ONE locked surgery chokepoint or reads a snapshot.
- [ ] Does it add hot-path work? → no (move off-thread / opt-in).
- [ ] Does it acquire a resource? → `finally` release exists.
- [ ] Does it grow unbounded? → there's a cap + shed policy.
- [ ] Does it break a CLAUDE.md invariant (see `00_PURGE_PLAN.md` DO-NOT-TOUCH appendix)? → stop.
- [ ] Re-run the pipeline against tag `design-freeze-2026-06-17` → identical/better behavior.

## Delete-on-sight anti-patterns (found in this codebase)
- Two copies of the same critical surgery (→ one chokepoint).
- Unlocked read of a tensor another thread resizes (→ snapshot/lock).
- `cleanup()` that targets a path the writer never produced (→ leak).
- Lock/handle acquired without `finally` (→ leak on exception).
- dt from `time.time()` (→ event-time).
- Unbounded accumulation with no cap (→ load-shed).
- Config-implicit serialized state (→ versioned contract).
