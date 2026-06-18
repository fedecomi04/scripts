# Adversarial re-verification of RUNTIME_shm_to_batch.md hazards

Stance: SKEPTIC — for each hazard I tried to DISPROVE it against the live code
(`vis="tensorboard"` / viser-direct only per invariant #9, KF kept ON, AnySplat FF
default, replay-recording OFF by default). Source untouched. Every verdict carries
file:line evidence. "refuted" only where I can show unreachable/guarded/dead;
"confirmed" only where I can show real+reachable+unguarded; "uncertain" where I
genuinely cannot prove either side.

Files read: `live_ros_publisher.py`, `live_shm_reader.py`, `dynamic_gs_pipeline_live.py`,
`dynamic_gs_pipeline_base.py`, `dynamic_gs_config.py`, CLAUDE.md.

---

## H1 — SHM leak on publisher crash (orig LOW)

**Claim:** `shutdown()` never `unlink()`s; only the next launch reclaims; a SIGKILL
or a sequence of differently-named crashed runs leaks `/dev/shm/dgs_live_shm`.

**Refutation attempt:** Tried to find a signal/atexit unlink that the audit missed.
There isn't one. `shutdown()` explicitly documents NOT unlinking (`live_ros_publisher.py:1295-1299`)
and the publisher signal handler `_on_signal` calls `pub.shutdown()` then `os._exit(128+signum)`
(`:1456-1460`) — `os._exit` skips atexit and never unlinks either. Reclaim is solely
the next launch's stale-unlink (`:594-597`). So the leak window described is real.
But it is bounded: same `--shm-name` (the only name the live flow uses, `DEFAULT_SHM_NAME`,
`live_shm_reader.py:52`) is reclaimed every run, and the comment's reasoning (don't unlink
while the reader may hold the fd) is sound. Benign-by-design.

**Verdict: confirmed** (as a benign, by-design leak — exactly as the audit framed it).
**Revised severity: low.** Reachable in live: partial (only on abnormal exit; normal
exit also leaks but is reclaimed next run).

---

## H2 — seqlock has no cross-process write barrier (orig MEDIUM)

**Claim:** payload stores (`:1021-1025`) precede the `latest_seq` publish (`:1027`) with
no memory fence; correct on x86 by strong store ordering, architecture-dependent; the
5-retry cap (`peek_latest` `live_shm_reader.py:386-414`) can silently return None under saturation.

**Refutation attempt:** Tried to show the seqlock fully closes the race. It does close
the *tearing* race: writer tags `slot["seq"]=seq` first (`:1020`), reader rejects if
`slot.seq != s_pre` pre-copy (`live_shm_reader.py:393`) and re-checks `s_post==s_pre AND
slot.seq==s_pre` post-copy (`:406`), and slot reuse every NUM_SLOTS=4 (`:1015`) guarantees
the next writer bumps `slot.seq` before touching payload. So no torn frame is ever
returned — the functional safety claim is solid. What I could NOT refute: (a) the absence
of an explicit fence (the stores are plain numpy slice assigns + `struct.pack_into` under
the GIL, no `__sync`/atomic) — correct on x86 only, as claimed; (b) the 5-retry → `return None`
(`:414`) silently skips a tick under saturation. Both are real but the target is x86
(this repo is sm_120 / x86-64 throughout), so the practical risk is the silent skip, which
at 30 Hz with 4 slots is rare.

**Verdict: confirmed** but real-but-rare; the data-corruption part is refuted on the
actual x86 target (lock-free + correct), only the silent-skip + portability caveat survive.
**Revised severity: low** (downgraded from MEDIUM: no corruption on the deployed arch;
worst case is an observably-rare dropped tick).

---

## H3 — pose/joint history lists mutated cross-thread without a lock (orig MEDIUM)

**Claim:** `_on_joint_state` (`:751-763`) and `_on_gazebo_pose` (`:765-781`) do
`list.insert` on the rospy receive thread(s); `_interpolate_c2w` (`:783-820`) reads them
on the worker thread; no lock; a half-updated (times grown, matrices not) pair → IndexError
or time/matrix mismatch.

**Refutation attempt:** I tried hardest to refute this as "single rospy thread serializes
the callbacks." It does NOT hold here: joint and pose are TWO independent `rospy.Subscriber`
objects (`:644-648`), and rospy gives each subscription its own receive thread by default
(no `MultiThreadedSpinner`, but per-subscriber threads are the rospy default). So
`_on_joint_state` and `_on_gazebo_pose` genuinely run on different threads concurrently,
and the worker thread (`_process_synced_pair` → `_interpolate_c2w` `:975`/`:783`) reads
`_gazebo_pose_times_sec`/`_gazebo_pose_matrices` and `_joint_state_*` (the latter also
captured-by-reference into `RobotMaskGenerator` `:814-815`) with NO lock on any of the four
lists. The two-statement `times.insert` then `mats.insert` (`:780-781`) is interruptible
between the two inserts → a reader can see `len(times) != len(mats)` or a time at index i
not matching the matrix at index i. `_state_lock` (`:548`) and `_record_lock` guard *other*
state, never these lists. I could not find any guard. The "hasn't blown up" is luck
(GIL granularity + the 20 ms throttle `:745` makes the window small), not correctness.

**Verdict: confirmed** — real, reachable in the default live path, unguarded.
**Revised severity: medium** (kept; this is the one true unguarded cross-thread mutable
state that can corrupt rather than drift, matching the audit's own "Notes for the purge").

---

## H4 — reader `close()` races an in-flight `peek_latest` (orig LOW)

**Claim:** `close()` sets `_slot_views=[]` then `_shm.close()` (`live_shm_reader.py:587-589`);
`peek_latest` indexes `_slot_views[idx]` (`:391`) and reads memoryviews; `close()` also
runs from the SIGINT/SIGTERM handler (`dynamic_gs_pipeline_live.py:170-184`) which can fire
mid-`peek_latest` → IndexError or freed-memoryview segfault. There is no `_closed` check at
the top of `peek_latest`.

**Refutation attempt:** Tried to show the handler can't fire concurrently with peek.
Two facts limit but do not eliminate the race: (1) `peek_latest` is only ever called from
the trainer MAIN thread (`dynamic_gs_pipeline_live.py:250`, `_tracker_tick`) — grep shows no
FF/VIS caller — so the *application* threading is single-reader; (2) the signal handler
`_on_signal` is installed via `signal.signal` (`:186-187`), and Python delivers signals on
the MAIN thread between bytecodes. So if SIGINT arrives WHILE the main thread is inside
`peek_latest`, Python runs `_on_signal` → `_cleanup_live_subscriber` → `sub.close()` re-entrantly
on the same main thread stack, sets `_slot_views=[]` and closes `_shm`, then returns into the
interrupted `peek_latest` which then does `np.array(slot[...])` over a closed buffer
(`:398-402`) or `_slot_views[idx]` on the now-empty list. So the race the audit describes is
a re-entrant-via-signal race, not a two-threads race — but it IS reachable and there is no
`_closed`/`[]`-guard inside `peek_latest` (confirmed: `:386-414` has no such check; the
`self._closed` flag exists at `:286` but is never consulted by peek). Real, but only in the
exact Ctrl-C-lands-inside-the-copy window. After `close()` returns the next tick is gated by
`if self._shm_sub is None: return` (`dynamic_gs_pipeline_live.py:247-248`, set by
`_cleanup_live_subscriber` `:205`), so it's a one-shot teardown-only hazard.

**Verdict: confirmed** real-but-rare (teardown-only, signal-reentrancy window).
**Revised severity: low** (kept). Reachable: partial (only during SIGINT/SIGTERM mid-copy).

---

## H5 — `_latest_tracker_frame` handoff: FF reads pose live, not from captured frame (orig LOW/MEDIUM)

**Claim:** MAIN writes a fresh `_latest_tracker_frame` dict each tick
(`dynamic_gs_pipeline_live.py:323-329`); FF is dispatched with the *captured* dict
(`:432` → `_dispatch_feedforward_async` `base:2513-2514`), but `_scene_c2w_for_frame`
(live override `:458-472`) reads `self._latest_tracker_frame["camera"]` — the LIVE attribute
— so a newer tick's pose can be used to place inserts the CDN judged on the older frame.

**Refutation attempt:** Tried to show FF uses the captured frame end-to-end. It does for
the CDN (`_feedforward_threaded` passes `target_frame.get("camera")`/`("batch")` into
`_compute_tick_cdn` `base:2531-2536`), and `_anysplat_slot_lock` serializes FF so only one
is ever in flight (`base:2509`). But the insert-placement pose comes from
`_scene_c2w_for_frame(frame_idx)` (`base:3125`), and the LIVE override
(`dynamic_gs_pipeline_live.py:465`) reads `self._latest_tracker_frame["camera"]`, NOT the
`target_frame` threaded into the FF thread. Since the MAIN thread keeps ticking and
overwriting `self._latest_tracker_frame` while the FF bg thread runs (the whole point of
moving FF off-thread), `_scene_c2w_for_frame` can read a pose from a tick *after* the one the
CDN scored. This is exactly the "ICP places inserts in a different frame than the CDN judges"
seam CLAUDE.md documents. Not a crash (atomic ref read), a correctness drift. I could not
refute it — the live override genuinely ignores its `frame_idx` arg and reads live state.

**Verdict: confirmed** (correctness drift, not a crash; reachable in default AnySplat FF).
**Revised severity: low** (the audit's LOW/MEDIUM → low: it's drift on top of an already-
documented ICP/frame seam, and the operator keeps ICP ON having judged the net result better;
no crash, bounded by the single-in-flight FF lock).

---

## H6 — `_on_synced` drop-oldest is a non-atomic get/put sequence (orig LOW)

**Claim:** on a full queue, `get_nowait()` then `put_nowait()` (`:858-869`), each in its own
try/except; interleaving receive callbacks could exceed maxsize or drop two.

**Refutation attempt:** `_on_synced` is the SINGLE registered sync callback
(`self._sync.registerCallback(self._on_synced)` `:677`) of ONE
`ApproximateTimeSynchronizer` over two `message_filters.Subscriber`s. The sync's callback
is invoked from whichever filter thread completes the match, but there is exactly one sync
and its dispatch is effectively serialized per match. Even if two fired, `queue.Queue`
get/put are individually thread-safe and the freshest-frame tracking semantics make a
transient over/under-fill harmless (the worker just drains newest-available). So the only
residual is the cosmetic one the audit already flagged. Could not elevate it.

**Verdict: confirmed** (benign, exactly as framed).
**Revised severity: low** (kept). Reachable: partial (needs multi-threaded rospy sync
dispatch, which the default single-sync setup makes unlikely).

---

## H7 — replay counters updated unlocked from two threads (orig LOW)

**Claim:** `_replay_count`/`_replay_dropped` written from the replay-writer thread (`:959`)
and the worker thread (`:1036`) without a lock; affects only finalize metadata; OFF by default.

**Refutation attempt:** Confirmed OFF by default: the `--record-replay` arg is only added
when `DGS_RECORD_REPLAY` env is set (`live_shm_reader.py:199-201`), and `_replay_dir`
guards every replay site (`:1032`, `:966`). So in the default live config this code path is
**not taken** — the hazard is unreachable unless an operator opts in. When opted in, the
unlocked counter is genuinely racy but purely cosmetic (a possibly-off-by-N count in
`replay_meta.json`).

**Verdict: refuted-as-unreachable in the default live config** (dead unless
`DGS_RECORD_REPLAY` is set); cosmetic-only even when enabled.
**Revised severity: low.** Reachable in live: no (default).

---

## H8 — stale doc claim "live depth filtered at the batch source" (orig DOC)

**Claim:** CLAUDE.md says the depth filter runs ONCE at the batch source incl. live
`_batch_from_live_frame`; the code keeps RAW depth there for live and filters only at the FF site.

**Refutation attempt:** Tried to show the batch is filtered for live. It is NOT:
`_batch_from_live_frame` (`dynamic_gs_pipeline_live.py:530-543`) builds `depth_image`
directly from `frame.depth_m` with an explicit comment "RAW depth for the tracker (live)…
batch['depth_image'] stays raw here" (`:532-537`). The filter is applied only at the FF
site, gated by `self._filter_depth_at_ff=True` set in the live ctor (`:117`) and consumed at
`base:3248-3254`. The recorded subclass filters in `_tracker_tick` and sets
`_filter_depth_at_ff=False` (`base:499`). So the live tracker DOES consume raw depth — the
CLAUDE.md "median+bilateral on ALL depth … dynamic tracker batch['depth_image']… consume the
cleaned depth" statement is inaccurate for the LIVE path (it holds for recorded). Pure doc
inconsistency; no runtime hazard; a purge "removing the duplicate batch-source filter" would
find nothing to remove (correct, as the audit warns).

**Verdict: confirmed** (documentation inconsistency, code is self-consistent).
**Revised severity: none** (doc-only; no code defect). Reachable: n/a.

---

## H9 — `_obj_mask_cache` None-check outside the lock (orig LOW)

**Claim:** `if self._obj_mask_cache is None:` (`base:1666`) is checked before acquiring the
model lock; only the render (`:1680`) is locked; MAIN invalidates (`live:275`) while FF may
read → at worst two threads both render (wasted work) or FF reads a slightly-newer mask; no
tearing because each assignment is a whole-tensor ref swap.

**Refutation attempt:** Confirmed precisely: the None-check at `:1666` is outside the
`lock.acquire()` at `:1677`; the render at `:1680` IS under the model lock (which guards the
torn-`gauss_params` race during FF insert per invariant #9). So the tearing-during-render
hazard is already guarded; the only unguarded thing is the check-then-act on the Python
attribute, whose worst case is a redundant render or a one-tick-stale mask. `_obj_mask_cache`
assignment (`:1680`/`:1686`) is a whole-object ref rebind (atomic under GIL) → no torn tensor.
Nothing to elevate.

**Verdict: confirmed** (benign, exactly as framed).
**Revised severity: low** (kept). Reachable: yes, but harmless.

---

## H10 — env-strip on publisher spawn is load-bearing (orig: NOT a bug; flag for purge safety)

**Claim:** `live_shm_reader.py:225-226` pops `LD_LIBRARY_PATH/CPATH/LIBRARY_PATH/CUDA_HOME`
before spawning the ROS-env publisher; removing it makes rospy/pyrender load the wrong
libstdc++/CUDA and crash at import. Do not purge.

**Refutation attempt:** Confirmed verbatim: lines 223-226 build `env=dict(os.environ)`,
set `PYTHONUNBUFFERED`, then `for _var in ("LD_LIBRARY_PATH","CPATH","LIBRARY_PATH","CUDA_HOME"): env.pop(_var, None)`,
with a long load-bearing comment (`:217-222`) explaining the cross-env libstdc++/cuda
incompatibility. This is intentional, correct, and NOT dead. Removing it would break the
publisher subprocess import in the `dynamic_gs_ros` py3.8 env.

**Verdict: refuted** (this is correct, intentional, load-bearing code — not a hazard).
**Revised severity: none.** Reachable: yes, and required.

---

## Summary table

| Hazard | Orig | Reachable (live) | Already guarded | Verdict | Revised |
|---|---|---|---|---|---|
| H1 SHM leak on crash | LOW | partial | n/a (by design) | confirmed (benign) | low |
| H2 seqlock no fence | MED | yes | seqlock (tearing only) | confirmed (no corruption on x86) | low |
| H3 pose/joint lists unlocked | MED | yes | no | confirmed | medium |
| H4 close() vs peek_latest | LOW | partial (signal reentrancy) | no `_closed` guard in peek | confirmed (teardown-only) | low |
| H5 FF reads pose live not captured | LOW/MED | yes | slot-lock (CDN only) | confirmed (drift) | low |
| H6 drop-oldest non-atomic | LOW | partial | Queue thread-safe | confirmed (benign) | low |
| H7 replay counters unlocked | LOW | no (DGS_RECORD_REPLAY) | n/a | refuted (unreachable default) | low |
| H8 stale depth-filter doc | DOC | n/a | n/a | confirmed (doc-only) | none |
| H9 obj_mask None-check unlocked | LOW | yes | lock on render | confirmed (benign) | low |
| H10 env-strip load-bearing | n/a | yes (required) | n/a | refuted (correct code) | none |

**The one to fix:** H3 (pose/joint history lists). It is the only unguarded cross-thread
mutable state that can corrupt (IndexError / time-matrix mismatch → wrong interpolated pose)
rather than merely drift, and it is reachable on the default live path because joint and pose
arrive on two independent rospy Subscriber threads (`live_ros_publisher.py:644-648`).
Everything else is benign, teardown-only, off-by-default, x86-safe, or doc-only.
