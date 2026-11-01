# Check: Session H's S92 phase 1 (H's "S90"): the oracles at key pools 4-32, P\*, and what the review's fixes became

*Referee V, 22 October 2026. I ran `git fetch` at about 17:00 UTC. claude/explore-H is at `617b62f3` (16:42:37). The main line is at `77221b8a`.*

*Checked: phase 1 as of `2bc50f9b` (16:15:34), "S90 phase 1 complete (P\* = 8; delta oracle the limit at 16 and 32); report writer batch 12; phase 2 running". I also checked the two commits after it, `64bc039a` (BASE\|3026 saved, 16:36:45) and `617b62f3` (BASE\|3025 saved, 16:42:37).*

*The review this follows up: `screen_review_s92_H.md` (`980498e4`, 05:28:50), on `23dccf24`.*

*Numbers marked "out §N" come from section N of the output of `check_H_s92_phase1.py`.*
- *The script uses the standard library only and reads git objects only. It writes no file and trains nothing.*
- *It runs H's `cell_reading` / `phase1` from `explore_h_s90.py` at `2bc50f9b`, with the harness replaced by stubs, on the stored records.*
- *It cuts the main line's `transition()` and `BIND_PASS` out of `test_binding_onset.py` by ast. They are the same at `9c5939e` and at the main head `77221b8a` (out §1).*

*Line references: "s:NN" is `explore_h_s90.py` and "run:NN" is `explore_h_run.py`. Both files are byte-identical at `23dccf24`, `2bc50f9b` and the head (out §0, §2).*

## Verdict

- **P\* = 8 is correct.** Recomputed from the 32 stored records it is 8, with budgets HEBB 43200 and DELTA 43200. That matches H's log and H's own `phase1()` run on the records.
- **The two definitions of "bind by 43200" agree in all 32 runs.** H's definition reads the 86400 run's own transition; the review's fix 2 cuts the curve at 43200. No cell, reading, P\* or budget differs. Case (a) could not occur:
  - all 29 bound runs stopped early at or before 43200;
  - the 3 unbound runs never reached 0.95 at any evaluation.
- **Every phase-1 record ran at 86400** (`budget` = `iters` = 86400). Each has its own cell's pool and memory and a gap-free curve. Each ends either at 86400 or after 3 evaluations ≥ 0.95.
- **None of the review's fixes 1-9 was taken before the first record or before phase 2.** None of them reached `explore_h_s90.py` at all.
  - The one related change is the new report-writer section in `2bc50f9b`. It prints the caveat and the OC in a future report_12.md, which is part of fix 7. It was committed 2 min **after** phase 2 started.
  - The two new CHECK rows (fix 8) were not added. Phase 2 skipped the CHECKs on the code-SHA match.
- **Phase 2 runs at pool 8 with a 43200 budget for all three arms.** It started at 16:13:13 and has 2 of 30 records so far: BASE\|3026 and BASE\|3025, both BOUND ROUTED.

## 1. Phase 1, cell by cell (out §1, §2)

The transitions are the stored ones. They equal `transition()` on each record's curve, and also `transition_full`.

| memory | pool | transitions 3020 / 3021 / 3022 / 3023 | by 43200 (H) | by 43200 (curve cut at 43200) | by 86400 | end accuracy | reading (H = fix 2) |
|---|---|---|---|---|---|---|---|
| Hebbian | 4 | 3600 / 4800 / 2400 / 2400 | 4/4 | 4/4 | 4/4 | 0.978 0.984 0.964 0.999 | reliable |
| Hebbian | 8 | 10800 / 8400 / 10800 / 8400 | 4/4 | 4/4 | 4/4 | 0.965 0.954 0.950 0.981 | reliable |
| Hebbian | 16 | 10800 / 14400 / 13200 / 9600 | 4/4 | 4/4 | 4/4 | 0.971 0.987 0.971 0.967 | reliable |
| Hebbian | 32 | 19200 / 14400 / 18000 / 18000 | 4/4 | 4/4 | 4/4 | 0.969 0.950 0.958 0.958 | reliable |
| delta | 4 | 1200 / 1200 / 1200 / 1200 | 4/4 | 4/4 | 4/4 | 1.000 ×4 | reliable |
| delta | 8 | 8400 / 7200 / 14400 / 22800 | 4/4 | 4/4 | 4/4 | 0.969 0.993 0.959 0.991 | reliable |
| delta | 16 | 14400 / **None** / 12000 / **None** | 2/4 | 2/4 | 2/4 | 0.972 **0.840** 0.952 **0.830** | the oracle is the limit |
| delta | 32 | 19200 / 25200 / **None** / 40800 | 3/4 | 3/4 | 3/4 | 0.967 0.967 **0.697** 0.967 | the oracle is the limit |

**The cells where the two "by 43200" definitions differ: none.**

**The three unbound runs** all ran the full 86400 updates (out §1). Their outcome is STREAM-PARTIAL one-to-one.

| run | accuracy at 43200 | maximum to 43200 | maximum over the run | evaluations ≥ 0.95 |
|---|---|---|---|---|
| DELTA_P16\|3021 | 0.831 | 0.842 | 0.843 (at 73200) | 0 |
| DELTA_P16\|3023 | 0.831 | 0.832 | 0.835 | 0 |
| DELTA_P32\|3022 | 0.538 | 0.550 | 0.721 (at 80400) | 0 |

- The two pool-16 runs are flat plateaus near 0.83 from 36000 to the end. They are not slow climbers that a longer budget would rescue.
- DELTA_P32\|3022 was still rising slowly at 86400.
- These three runs are the only ones with an accuracy at 43200. H's docstring (s:53) promises "the held-out accuracy at 43200" per cell. `phase1()` computes it (s:158), but neither the log (`per_seed_log`, s:312-316) nor the batch-12 `s90_section` prints it. That is a small descriptive omission.

**The other record fields.** All 32 records match on every one of these (out §1):
- same code SHA `287fdd3941d0`, same CPU (Xeon @ 2.10 GHz);
- `budget` 86400 and `iters` 86400;
- `pool` and `memory` match the cell, and `kind` is oracle;
- evaluations every 1200 from 1200 with no gap, and `stopped_at` = the last evaluation;
- each ends at 86400 or after 3 evaluations ≥ 0.95.

The 32 phase-1 records are identical at `2bc50f9b` and at the head.

**P\*.**
- Hebbian reads reliable at every pool. Delta reads reliable at 4 and 8, and "the oracle is the limit" at 16 and 32.
- So **P\* = 8**, and both memories read reliable at P\*. The budgets are HEBB 43200 and DELTA 43200.
- H's `phase1()` run on the records gives the same result, and so does the fix-2 definition (out §2). It equals the log's line (s90_phase1.log:143) and the phase-2 header (s90_phase2.log:2).
- The review's fix-3 and fix-4 branches would not have fired. The pool-4 reference reads reliable on both memories. No pool below P\* fails, so the pools are nested.

**How much P\* rests on** (out §4; H's `phase1()` with one stored transition replaced):
- **One seed decides 8 against 32.** If DELTA_P32\|3022 had bound, P\* would be **32**: at 20000 with budgets 43200 / 43200, at 60000 with DELTA at 86400. Pool 16 would still fail 2/4. This is the non-nested case the review described (§B and fix 4) and the prompt's rule allows. It did not occur, but only by this one run.
- **One seed at pool 8 decides 8 against 4.** If DELTA_P8\|3023 (transition 22800) or HEBB_P8\|3020 had not bound, P\* would be 4 and phase 2 would not run.
- **Pool 16 is not marginal.** A single changed seed (3021) leaves P\* at 8.
- Delta's pool-32 cell, at 3/4, is the review's "one failure in four" case. The Wilson interval of 3/4 is 0.30-0.95. Delta's pool-16 cell, at 2/4 with two plateaued runs, is a clear failure.

**The pool-4 reference.** The cells bound at 2400-4800 (Hebbian) and 1200 (delta). That agrees with the earlier pool-4 oracles the review cites: S87a's ORACLE at 2400 / 8400, and X's CEIL_DELTA at 1200 / 2400. They are different seeds, so this is agreement in kind, not a bit-for-bit tie.

## 2. The review's fixes: which were taken, and when (out §0, §3)

Nothing on claude/explore-H cites the review: no `980498e`, no `screen_review_s92` and no "S92" in any file. Between `23dccf24` and the head, the only change outside `explore_out/H/` is `explore_h_report.py`, in `2bc50f9b`, +61/-1.

| fix | taken? | where / when |
|---|---|---|
| 1 Name S92 | **no** | `explore_h_s90.py` has "S90" 8 times and "S92" 0 times. s:3 still says "round 6". The new `s90_section` heads its section "## S90" and has no S92. Every commit subject says S90 |
| 2 "by 43200" on the curve cut at 43200 | **no** (s:138-146 unchanged) | moot on these records: the two definitions agree in all 32 runs (§1) |
| 3 The failed pool-4 reference named | **no** (s:279-280 unchanged) | moot: the reference passes on both memories |
| 4 Phase-1 OC, P\* distribution, "pools not nested" | **no** (s:22-23 and OC s:115-118 unchanged; still only q^4) | the pools are nested here. The OC text still lacks the joint (q_h q_d)^4 and the P\* distribution |
| 5 The moot split guard | **no** (s:298-300 unchanged) | open for phase 2's reading |
| 6 Two-sided split rule; renaming "needed (no reading)"; the moot share in the OC | **no** | open for phase 2's reading |
| 7 Print the caveat and the OC | **partly**, at `2bc50f9b` (16:15:34), **after** phase 2 started (16:13:13) | `explore_h_report.py` `s90_section` writes `res['caveat']` and `res['oc']` under the readings in report_12.md. report_12.md is not committed yet. The driver's print (run:152) is unchanged, and the logs print the readings dict with neither the caveat nor the OC (s90_phase1.log:143) |
| 8 Two new CHECK rows (DELTA_NOSPLIT\|583 against X's DELTA_SLOW\|583; the pool-4 oracles against S87a's ORACLE\|2600 and X's CEIL_DELTA\|580) | **no** | `check()` is unchanged (s:193-244). s90_phase2.log:12 reads "CHECKs of H/s90 skipped: passed at 2026-10-22 05:28:28 UTC with this code SHA". **DELTA_NOSPLIT runs in phase 2 with no CHECK of its path** |
| 9 Budget guard in report() | **no** (report() s:264-309 unchanged) | it has cost nothing so far: I checked every record's budget by hand (all 32 phase-1 at 86400; the 2 phase-2 at 43200 = PLAN) |
| 10 Text (s:6, s:57-64 pins, s:68 `--push-each`, s:92, LABELS, §G) | **no** | — |

**Before or after the first record.**
- Fixes 1-4 were never taken.
- The review was committed at 05:28:50. H's CHECKs had passed 22 s earlier, at 05:28:28, and the phase-1 runs started then. The first record was HEBB_P4\|3023, committed `5d978f45` at 05:34:49, **6 minutes after the review**.

**Before phase 2.**
- Fixes 5-9 were not taken.
- Phase 2 started at 16:13:13, **10 h 44 min after the review**. Its first line is the same second as the last phase-1 save, `c21f0d74` 16:13:13: s90_phase1.log:144 ends "chain exit 0" and phase 2 follows, so H chained the two stages.
- The review's "then V's look at the phase-1 readings and P\*" did not happen before phase 2 was queued. This file is that look, after the fact.

## 3. Order (out §0)

| when (UTC, 22 Oct) | what |
|---|---|
| 05:05:11 | `23dccf24`: the readings (s:16-47), the parent and child, run:13 |
| 05:05:13 | the phase-1 driver starts (CHECKs) |
| 05:28:28 | all 7 CHECK rows pass (22.1 min); the pool-4 MAINSPLIT tie is printed with curve, statistics, gradient norms and trigger rows all True, across 2.80 → 2.10 GHz; the 32 runs are queued |
| 05:28:50 | the review, `980498e4` |
| 05:29:26 | `061bb4a0`: "CHECKs passed; phase 1 running" (checks.json gets H/s90) |
| 05:34:49 | the **first record**, `5d978f45` (HEBB_P4\|3023) |
| 08:03:29 → 08:14:28 | the **container restart**. The last save before it is `fc51b747` (DELTA_P8\|3022, at 155.0 min). The driver was killed with DELTA_P8\|3023 and DELTA_P16\|3020/3021/3022 in progress. It restarted at 08:14:28 (s90_phase1.log:94-95, header git `fc51b74+dirty`), reran the repro check bit-identical, skipped the CHECKs on the SHA, and queued the 9 remaining runs from scratch. Every record has `attempts` 1, and the four rerun records have full curves from 1200. `005b730d` (08:17:08) notes the restart |
| 16:13:13 | the last phase-1 record (`c21f0d74`, DELTA_P32\|3022, 323 min); readings printed; P\* = 8; **phase 2 starts** (header git `c21f0d7+dirty`) |
| 16:15:34 | `2bc50f9b`: report writer batch 12 (+ s90_phase2.log) |
| 16:36:45, 16:42:37 | BASE\|3026 and BASE\|3025 saved |

**The readings came before every record:** 23dccf24 at 05:05:11, the first record at 05:34:49.

**Code changes after the first record.**
- `explore_h_s90.py`, `explore_h_s90_child.py` and `explore_h_run.py` have the same blobs at every commit from `23dccf24` to the head.
- The code SHA is `287fdd3941d0` in all 34 records and in the store's meta.
- The only post-record code change is `explore_h_report.py`, which is not in the code SHA and is not imported by the runs.

**What the "+dirty" markers hide.** Both restart headers show a dirty tree (`fc51b74+dirty` at 08:14 and `c21f0d7+dirty` at 16:13).
- At 16:13 the dirty file was plausibly the uncommitted `explore_h_report.py`, committed 2 min later.
- I cannot see the working tree. The parent is outside the code SHA, so a dirty parent would not show in the records. Two things are consistent with the committed parent:
  - the store's phase-2 meta (`BASE` / `DELTA_SPLIT` / `DELTA_NOSPLIT`: iters 43200, "key pool 8");
  - the phase-2 records (pool 8, budget 43200, trigger last check 40800).

**A small point about the records' `git` field.** It is HEAD at save time, not at start. For example, BASE\|3026 started at `c21f0d7` and records `2bc50f9`. The code SHA is the field that ties a record to its code.

## 4. Phase 2 so far, and what phase 1 fixes for it (out §3)

**The configuration.**
- PLAN = {K 8; HEBB 43200, DELTA 43200}.
- 30 runs queued: BASE, DELTA_SPLIT and DELTA_NOSPLIT × 3024-3033.
- The projected worst case is 720 min per run, i.e. 43200.

**The records so far.**

| run | pool | budget | trigger last check | transition | stop | outcome | fired at |
|---|---|---|---|---|---|---|---|
| BASE\|3026 | 8 | 43200 | 40800 | 15600 | 18000 | BOUND ROUTED | 4800 (probe 0.031, op_ok True) |
| BASE\|3025 | 8 | 43200 | 40800 | 20400 | 22800 | BOUND ROUTED | 19200 (probe 0.812, op_ok True) |

No DELTA_SPLIT or DELTA_NOSPLIT record exists yet.

**What phase 1 fixes for phase 2's readings:**
- **Pool 8 only.** "The recipe carries to P\*" will mean pool 8.
  - The Hebbian oracle is reliable to 32, but P\* is joint, so phase 2 says nothing about Hebbian recipes at 16 or 32.
  - Nothing at 16 or 32 is a recipe question for delta: there the delta oracle itself is the limit.
  - Report 11's candidate, "a smaller key pool", is the one this tests. "A longer budget" is not: no memory needed 86400 at P\*.
- **A 43200 budget for all three arms.** It was named from the oracles: delta's slowest pool-8 oracle bound at 22800 (seed 3023). A learned arm that binds after 43200 counts as unbound. The trigger's last check is 40800.
- **The choice of 8 over 32 hangs on DELTA_P32\|3022** (§1), and the printed reading does not say so. I suggest stating it with the P\* reading.
- **Fix 8(a) is still open.** DELTA_NOSPLIT's path (no trigger) has no CHECK row in S90's child, and its runs are in the queue.
- **Fix 8(b) bears less.** Phase 2 runs at pool 8, not 4. The pool-4 cells bound at the expected speeds (§1).
- **Fixes 5 and 6 decide how the split line will print.** Both are reading text in the parent, outside the code SHA, so they could still land before any delta record without touching a run. But the readings were "fixed before any run" (s:33), and any change made now must be a dated note that gives the commit order.
- **Fix 9's guard is not there.** I checked each record's budget by hand instead.

## Notes on Referee V's own file (`screen_review_s92_H.md`)

- **Fix numbering.** :181 says "Land 1-4 before the first phase-1 record, and **5-10** before phase 2". The verdict, :200, says "Fix before phase 2: **5-9**". Fix 10, the text fixes, is in one list and not the other. This file uses 1-9 for the timing questions, as the task did, and lists 10 separately.
- **The timing estimate.** :24 says the "first records could land within about an hour". The first landed 6 min after the review commit, because the CHECKs finished 22 s before it. Fixes 1-4 "before the first phase-1 record" were open for about 6 minutes after the review existed.
- **The run status.** It says "CHECKs in progress", based on the 05:25 fetch. That was true when fetched. All of them passed at 05:28:28, before the commit. The review's "could not be checked: whether the CHECKs will pass" is now resolved: all 7 rows passed, and the printed pool-4 tie is all True.

## What could not be checked

- The working tree behind the two "+dirty" headers, and the exact command lines. The per-run commits show that push-after-save was on.
- Whether the four runs killed by the restart would have given the same records. They were rerun from scratch. The repro check was bit-identical after the restart, which supports this, but it does not prove it for those four runs.
- report_12.md: it is not committed, so I could not check what batch 12 prints.
- Phase 2's readings. There are 2 records of 30, and no delta arm yet.

*Housekeeping: I wrote only this file and `check_H_s92_phase1.py`, which writes no output file. I committed nothing, checked nothing out, trained nothing, and ran no session code as a program.*
