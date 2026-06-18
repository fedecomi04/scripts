# 00 — Review Guide (read this first)

How to review the `code_audit/` package and run the purge. Written 2026-06-18 while the
context was fresh, so the next conversation (and you) start with the approach already
decided instead of re-deriving it.

> **Mental model:** the audit is a *map*, not the territory, and it was written by AI
> agents — so it shares the AI blind spots listed below. Treat it as a **prioritized
> review queue**, not a verified TODO. Every DELETE gets a confirm-read; every REFACTOR
> gets your judgment.

---

## 1. What to read, in what ORDER

Read by **dataflow, not file order.** Build the live mental model first, then triage.

1. `ARCHITECTURE_PRINCIPLES.md` — what to optimize *for* (the north star). 5 min.
2. `RUNTIME_target_architecture.md` — the intended clean 3-thread design. 5 min.
3. `RUNTIME_<path>.md` in execution order: `shm_to_batch` → `tracker_tick` → `ff_dispatch` → `viser_push` → `state_mutation_map` → `warmload_lifecycle`. This is the live tick end-to-end. Follow the actual source alongside each.
4. `RUNTIME_hazards.md` + `00_DEAD_CODE.md` — the verified-ish findings.
5. `00_PURGE_PLAN.md` — the master checklist; tick `DECISION:` slots as you go.
6. `DUP_consolidation_plan.md` — the ~660 LOC + the drifted `gaussian_surgery`.
7. `<module>.md` ×24 — the per-function index. Don't read cover-to-cover; use as a lookup: read the one-liner, jump to the function, decide.

**For the code itself:** open at `_tracker_tick` and follow the data. The `<module>.md`
function-maps are your index into the source — not a substitute for reading it.

---

## 2. TRUST TIERS — what to trust vs verify

| Finding type | Trust | Your action |
|---|---|---|
| **grep-verified dead code** (`00_DEAD_CODE.md`, 39 symbols) | HIGH | 2-min confirm-read (watch for dynamic dispatch / getattr / entry points), then DELETE |
| **LOC / duplication counts** | HIGH | mechanical; trust |
| **god-module / cohesion smells** | MED-HIGH | real, but the *how to split* is your call |
| **races / leaks / lifecycle hazards** (H-CROP, /dev/shm, slot-lock) | MED — *hypotheses from code structure, NOT observed* | read the lines; ideally reproduce/instrument before trusting the severity |
| **the `NameError phase0.py:661`** | HIGH (it's a plain undefined-name on a real branch) | confirm the branch is reachable in your live flow, then fix |
| **design/refactor recommendations** | LOW-MED | these are judgment calls where your domain knowledge >> the audit's; you own them |
| **any "this is unused / pointless"** | SUSPECT | could be Chesterton's fence — see blind spots |

---

## 3. AI BLIND SPOTS — where the audit (and I) fail to see something is off

The audit was written by agents with these exact limits. Do **not** let it be the final word in categories 1–5:

1. **Intent — "why does this exist."** AI sees what code does, never why. Anything load-bearing for a reason in your head / a paper / a hardware quirk / a future plan reads as deletable. (`sam3d_init_target_flags` would look dead without CLAUDE.md.) **#1 failure mode = Chesterton's fence.** → *You* adjudicate intent.
2. **Runtime truth — races, contention, real timing.** The hazards are *inferred from code structure*, never *observed firing*. (cf. CLAUDE.md: "two confident freeze hypotheses were wrong; only instrumented A/B found cudnn.benchmark.") → *Measurement* adjudicates, not reading.
3. **Perceptual / numerical correctness.** AI can't tell if depth looks right, the tracker jitters, a blob is misplaced, or a coordinate sign is flipped (a sign error looks fine on the page). → *The viewer / A/B* adjudicates.
4. **Performance reality.** AI timing guesses here have been 2–5× wrong. Real bottlenecks are counterintuitive (SVO-vs-live fps; 140ms debug-I/O). → Trust *numbers*, not claims.
5. **Hardware/env coupling.** VRAM, sm_120, Jetson SDK version, RTF — known only when told; a proposal may OOM or not run.

**Division of labor:** AI maps & catalogs (breadth — all 28k LOC, grep, cross-ref). **You** adjudicate intent. **The runtime** (measurement, viewer, A/B) adjudicates correctness.

---

## 4. PURGE EXECUTION DISCIPLINE

- **Baseline:** everything diffs against tag `design-freeze-2026-06-17` (working ZED pipeline). After each batch of edits, re-run the pipeline against it → identical-or-better behavior, or revert.
- **The gate (from `ARCHITECTURE_PRINCIPLES.md`):** every edit passes — touches shared state? → one locked surgery chokepoint or a snapshot. Adds hot-path work? → no. Acquires a resource? → `finally`. Grows unbounded? → capped. Breaks a CLAUDE.md invariant? → stop.
- **Order (from `DUP_consolidation_plan.md`):** safest first — stdlib-only helpers (R5 io/subprocess → R4 camera_conventions → R3 rotations), then depth/mask (R2), then the model tensor-surgery (R1 `gaussian_surgery`) **last, behind a hard verification gate** (equal-length asserts on all params+buffers, `.pt` round-trip, LRs stay 0, `sam3d_init_target_flags` stays zero).
- **Highest-leverage single move:** the locked `gaussian_surgery` chokepoint + snapshot reads fixes the H-CROP race + the static↔dynamic drift + the buffer foot-gun **together.** Do this as the anchor refactor, not as 3 separate patches.
- **DO NOT TOUCH:** the CLAUDE.md invariant appendix in `00_PURGE_PLAN.md` (means LR=0, cam-opt off, _ZERO_LR_OPTIMIZERS, viser-direct, the 4 buffers' existence, the TSDF trunc fix, the cotracker-named master switch, the outputs/ monkeypatches).

---

## 5. HANDOFF — start the purge in a FRESH conversation

This conversation is compacting (fidelity already degrading). Everything needed is
self-contained: the `code_audit/` files + `CLAUDE.md` + the memory entry
`project_code_audit_purge_prep` (auto-loads, points at all of it).

Open a new conversation with roughly:
> "Read `scripts/code_audit/00_REVIEW_GUIDE.md` and `CLAUDE.md`. We're purging the
> dynamic-gs LIVE path. Start with the `gaussian_surgery` chokepoint refactor; work
> against tag `design-freeze-2026-06-17`."

Tick your `DECISION:` slots in `00_PURGE_PLAN.md` first (or as you go) so the executor
acts on *your* calls, not the audit's defaults.
