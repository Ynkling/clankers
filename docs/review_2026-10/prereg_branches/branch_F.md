<!-- Appendix to docs/review_2026-10/preregistration.md. Produced for the referee by a delegated read-only audit of branch F (git log/show/diff and JSON via git show only); the referee spot-checked its load-bearing claims against git (see preregistration.md). -->

# Pre-registration audit, Session F (origin/claude/explore-F)

Scope: S53, S54 (a)-(d), S59, S68, S69, S70. Read-only; only `git log/show/diff/merge-base` and python3 on JSON from `git show`.
Timezones: every git timestamp below is the committer date (`%cI`) and every one carries `+00:00` (UTC). Times in run
logs and checks files (`started ...`, filenames `checks_<screen>_YYYYMMDD_HHMMSS.json`, runtime `measured`) carry no zone;
they come from `time.strftime` (local time; explore_f_common.py l.197). The queue lines in the same logs print `date` with
"UTC" (for example `=== S70 phase 1 Fri Oct  9 14:19:53 UTC 2026`, one second before `started 2026-10-09 14:19:54`), so the
container's local time was UTC. They are written as "Z (log)" below.
Limits of git: committer dates are self-reported and the push time is not recorded. "+dirty" in a run's provenance means
the working tree had uncommitted changes, and git does not say what they were (the outputs under explore_out/F are the
likely cause, but that cannot be proved). No results store has a `meta` entry (`"meta": {}` at every commit) and no record
has a start-time field. Records carry only `secs`/`secs_wall` and `git` (the HEAD when the record was written; for example
the S70 oracle records say 9899520, which was committed after that segment started at a1112da).

## How the checks were done
- `git log [-p] [--follow] origin/claude/explore-F -- explore_f_<name>.py` and `git log --all -- explore_f_<name>.py`:
  **every one of the 7 screen files has exactly one commit, the commit that added it.** `git diff <spec> origin/claude/explore-F -- <file>` is
  empty for all 7. No branch rewrites the files.
- Report code: explore_f_reports.py: bb906e5, then 9ddda81, which only adds lines (no `-` lines). explore_f_reports2.py: only a1112da.
  Child files: only explore_f_window_delta_child.py and explore_f_window_delta_scale_child.py change afterwards, in da4f065
  (2026-10-08T05:03:51Z, the fast evaluation path, before any S54 run). That diff has no reading, threshold, arm or seed lines.
- Each spec commit is an ancestor of the first commit that holds results for its screen (`git merge-base --is-ancestor`), and the branch has no merges.
- Stores: python3 over `git show <c>:explore_out/F/<name>_results.json` at every commit that touches the store.

---------------------------------------------------------------------------------------------------------------------
## S53 explore_f_delta_controls.py

| item | commit / file | time |
|---|---|---|
| readings first committed (file added) | bb906e5 "Session F S53 delta_controls: specification and readings (committed before any run), ..." | 2026-10-08T04:14:49Z |
| run start (s53.log header, provenance `git bb906e5`, clean) | s53.log | started 2026-10-08 04:14:55Z (log) |
| first checks file | checks_s53_20261008_041457.json (provenance started 04:14:59, git bb906e5); added in 91a3d26 | 04:14:57Z (filename) |
| timings measured / first run start | s53_runtime.json `measured`; log `start 0.0 min ORACLE_DELTA_P8 seed 300` | 2026-10-08 04:16:09Z (log) |
| first store commit with run records | 91a3d26 (29 records; the earliest records have `git` bb906e5) | 2026-10-08T05:07:28Z |
| later changes to reading lines | none (single commit; checked as above) | — |
| report statement | report_1.md (93e0aa9, 2026-10-08T06:18:11Z) | matches |

Readings as first committed (bb906e5, lines 34-48, verbatim):
```
VALIDITY (fixed before any run): a layout (P = 4) is VALID FOR THE DELTA READING if ORACLE_DELTA binds on >= 2 of seeds
300-302 there (if the delta memory cannot bind with perfect routing, a single delta channel's failure says nothing).
ORACLE_HEBB's validity on each layout (>= 2 of 300-302) is printed alongside.

READINGS (fixed before any run):
  R1 "delta makes the single channel fail" if SINGLE_DELTA BOUND <= 1/10 on both layouts AND SINGLE_HEBB BOUND >= 3/10
     on at least one layout, both layouts valid for the delta reading. If a layout is not valid: R1 is UNTESTED.
     Otherwise: neither.
  R2 "delta raises the oracle" if, at P = 8 within 48000 updates, ORACLE_DELTA binds on >= 2 of the 3 seeds on which
     ORACLE_HEBB does not bind (seeds 300-302, paired). Otherwise: neither.
  Descriptive, fixed now: SINGLE_DELTA's final held-out accuracy by write order (explore_f_tasks.write_order_acc on the
  2048 held-out sequences): over queries whose (stream, key) was the LAST of its key's two writes, and over the others;
  for the others, the fraction answered with the last-written stream's value; and by query stream. "keeps exactly the
  last-written stream" if acc_last >= 0.9, acc_notlast <= 0.1 and (last-written value among the others) >= 0.8, on a
  layout's median run; "does not" otherwise. Printed for every arm.
```
Arms and seeds (l.19, code l.70-71): `ARMS (seeds 300-309 at P = 4 on each layout, 24000 updates; the P = 8 pair on seeds 300-302, 48000 updates):` / `ITERS, ITERS_P8 = 24000, 48000` / `SEEDS, SEEDS_P8 = tuple(range(300, 310)), (300, 301, 302)`.

Report_1: "The rule needed SINGLE_DELTA ≤ 1/10 on both layouts and SINGLE_HEBB ≥ 3/10 on one." and R2 "(0 seeds where only
delta binds; rule ≥ 2)". These match the docstring. The write-order reading is reported as "does **not** keep only the
last-written stream", which uses the same rule.
Note: the readings commit came 6 s before the logged run start, so nothing in git separates "commit" from "run". The order (commit, then run)
is supported by the run's provenance recording HEAD = bb906e5 with a clean tree.

---------------------------------------------------------------------------------------------------------------------
## S54 explore_f_window_delta.py ((a), (d)) and explore_f_window_delta_scale.py ((b), (c))

| item | commit / file | time |
|---|---|---|
| readings first committed (both files added) | 9ddda81 "Session F S54 (window gate on delta channels, (a)-(d)) and S59 (decay): specifications and readings, committed before any run" | 2026-10-08T04:35:17Z |
| first S54 harness start (checks only; no `start` lines in this segment) | s54.log `=== queue: S54 start Thu Oct  8 06:15:14 UTC 2026`, started 06:15:16, git 147e5cf+dirty | 2026-10-08 06:15Z (log) |
| first checks file | checks_s54_20261008_061520.json (added in 93e0aa9) | 06:15:20Z (filename) |
| restart (checks only) | s54.log `S54 (re)start Thu Oct  8 07:51:08 UTC 2026`; checks_s54_20261008_075116.json | 07:51Z (log) |
| first training runs started (C_HEBB 310-313; none of these were stored, because the container was restarted) | s54.log `=== segment Thu Oct  8 08:38:26 UTC 2026`, then 147 s of timings, then `start 0.0 min ... C_HEBB seed 310` | ≈08:41Z (log) |
| next restart (C_DELTA 310-313 started) | `S54 (re)start Thu Oct  8 08:53:55 UTC 2026`; checks_s54_20261008_085400.json; s54_runtime measured 08:53:04 | 08:53Z (log) |
| first store commit with records, (b)/(c) | 0268500: window_delta_scale_results.json, 1 record (C_DELTA\|312) | 2026-10-08T14:16:38Z |
| first store commit with records, (a)/(d) | c6a1e56: window_delta_results.json, 4 records (D_DELTA 330/331, D_ORACLE_DELTA 330/331) | 2026-10-09T03:21:00Z |
| later changes to reading lines | none in either file (single commit each) | — |
| report statement | report_2.md (a82c62a, 2026-10-09T08:04:31Z) | matches the docstring; differs from the prompt for (a), see below |

Readings as first committed, explore_f_window_delta.py (9ddda81, lines 35-45, verbatim):
```
VALIDITY (fixed before any run): a memory's arm is VALID if its perfect-gate arm binds on at least 1 of its 2 seeds within
24000 updates; otherwise its reading is UNTESTED.

READINGS (fixed before any run)
 (a) on DISCOVERED, A_DELTA (40 seeds) against the recorded Hebbian LOCAL3_SLOW 31/40 (unpaired, another CPU-independent
     seed block; counts only, one-sided Fisher printed): "delta not worse" if A_DELTA >= 29/40 (31 - 2: the bound);
     "delta better" if A_DELTA >= 35/40 (31 + 4); "delta worse" if A_DELTA <= 28/40. (Both "not worse" and "better"
     hold at >= 35.)
 (d) descriptive: per memory, BOUND, DISCOVERED and ROUTED*@end counts, failure classes (expected: KEY splits),
     transitions; paired D_DELTA vs D_HEBB (one-sided McNemar both ways, printed). The BASELINE for Session G is
     D_HEBB's and D_DELTA's DISCOVERED counts (with BOUND and ROUTED*@end alongside).
```
The same commit sets the matching code constants (l.69-71): `A_SEEDS, D_SEEDS = tuple(range(310, 350)), tuple(range(330, 350))` /
`REC_LOCAL3_SLOW = 31                       # batch 16's S43 LOCAL3_SLOW DISCOVERED on 160-199 (Hebbian), of 40` /
`NOT_WORSE, BETTER, WORSE_MAX = 29, 35, 28`.

Readings as first committed, explore_f_window_delta_scale.py (9ddda81, lines 26-32, verbatim):
```
VALIDITY (fixed before any run): per configuration and memory, VALID if the perfect gate binds on at least 1 of its 2
seeds within the configuration's budget; a reading needs both memories VALID, else UNTESTED.

READINGS (fixed before any run), per configuration, paired over its seeds, b = DELTA only, c = HEBB only (BOUND ROUTED):
   "delta not worse"  if c − b <= 2  (DELTA >= HEBB − 2; a bound, no p-value);
   "delta better"     if b >= 4 and c = 0;
   otherwise "delta worse" (c − b > 2). One-sided exact McNemar printed both ways.
```
Code (l.56-58): `B_ITERS, C_ITERS = 28800, 43200` / `B_SEEDS, C_SEEDS = tuple(range(310, 330)), tuple(range(310, 320))` / `NOT_WORSE_D, BETTER_B = 2, 4`.

**Where the S54(a) ">= 35" came from.** The original prompt (docs/reading/parallel_sessions_prompts.md, ad8a92e,
2026-10-08T03:50:57Z, Session F block) says: `Readings: (a)-(c) "delta not worse" if DELTA ≥ HEBB − 2 discordant pairs (a bound, no p-value);` /
`"delta better" if ≥ 4 vs 0 paired. (d) descriptive: counts, failure classes (expect KEY splits).` The prompt also makes
(a) unpaired: "vs the recorded LOCAL3_SLOW on Hebbian (counts only, different CPU)". The unpaired threshold
"delta better if A_DELTA >= 35/40 (31 + 4)" was first written in **9ddda81, 2026-10-08T04:35:17Z**, in the docstring of
explore_f_window_delta.py (l.41) and as `BETTER = 35` (l.71). `git log --all -S'31 + 4'` and
`-S'NOT_WORSE, BETTER, WORSE_MAX'` return only 9ddda81. The later edits of the prompts document (6f6e583, 8be3f98) do not
touch the S54 lines. The threshold therefore predates every S54 run: the first training start was ≈08:41Z on 10-08, and the
first A_DELTA record came later still. It does depart from the prompt's paired rule, but the departure is disclosed in
the docstring as written before any run ("unpaired ... counts only").
The prompt's paired rule "≥ 4 vs 0" is kept for (b)/(c) as "b >= 4 and c = 0".

Report_2 statements: table "**delta better** (≥ 35)", and "(a) "delta better": 36/40 ≥ 35 (and "not worse", ≥ 29)". Both
match the docstring. "(b) ... c − b = −1 ≤ 2 ("better" needs b ≥ 4)": the report leaves out the "and c = 0" part, which
changes nothing here (b = 1). "(c) "delta worse": c − b = 6 > 2" matches. "(d) UNTESTED ... Neither perfect gate binds
RandHeaderTask ... within 24000" matches the VALIDITY rule (>= 1 of 2 seeds). The report then proposes a lighter
RandHeaderTask, which became S70 (a new screen; see below).

---------------------------------------------------------------------------------------------------------------------
## S59 explore_f_decay.py

| item | commit / file | time |
|---|---|---|
| readings first committed (file added) | 9ddda81 (same commit as S54) | 2026-10-08T04:35:17Z |
| run start | s59.log `=== queue: S59 oracle arms Fri Oct  9 05:42:42 UTC 2026`; started 05:42:44, git b77e8bb+dirty | 2026-10-09 05:42Z (log) |
| first checks file | checks_s59_20261009_054246.json (added in e900359) | 05:42:46Z (filename) |
| timings measured | s59_runtime.json `measured` | 2026-10-09 05:44:12Z (log) |
| first store commit with records | e900359: decay_results.json, 23 records (ORACLE_* arms) | 2026-10-09T06:02:22Z |
| later changes to reading lines | none (single commit) | — |
| report statement | report_2.md (a82c62a) | matches |

Readings as first committed (9ddda81, lines 27-35, verbatim):
```
VALIDITY (fixed before any run): a decay's arms are VALID if its ORACLE binds on >= 2 of seeds 350-352 within 48000;
the reading for ROUTED or GDN needs that decay and FIXED both VALID, else it is UNTESTED for that decay.

READING (fixed before any run): per seed and decay d, the oracle gap gap_d = (final held-out accuracy of ORACLE_d) −
(final held-out accuracy of SINGLE_d) (rec["acc"], the last evaluation). "the horizon knob matters" if for ROUTED or for
GDN, gap_d − gap_FIXED >= 0.15 on >= 7 of the 10 seeds. Otherwise "it does not". If it matters, the winning decay is the
one with more such seeds (a tie: the larger median of gap_d − gap_FIXED), and D_DELTA_<winner> runs on seeds 330-339
(descriptive: DISCOVERED, BOUND, ROUTED*@end and failure classes beside S54(d)'s D_DELTA on the same seeds). If it does
not matter, the rerun is not made.
```
Code (l.58-61): `ITERS, D_ITERS = 48000, 24000` / `SEEDS, D_SEEDS = tuple(range(350, 360)), tuple(range(330, 340))` / `VALID_SEEDS = (350, 351, 352)` / `GAP, N_SEEDS = 0.15, 7`.
Report_2: "The rule needed gap_d − gap_FIXED ≥ 0.15 on ≥ 7/10; ROUTED 0/10, GDN 1/10 ... The conditional rerun under
S54(d)'s gate was not made." This matches. (The prompt's reading is the same rule: "≥ 0.15 on ≥ 7/10 seeds".)

---------------------------------------------------------------------------------------------------------------------
## S68 / S69 / S70: explore_f_premise.py, explore_f_hebb_decay.py, explore_f_rh_lite.py (spec a1112da)

| item | S68 premise | S69 hebb_decay | S70 rh_lite |
|---|---|---|---|
| readings first committed | a1112da 2026-10-09T14:19:50Z | a1112da 2026-10-09T14:19:50Z | a1112da 2026-10-09T14:19:50Z |
| run segment start | followup.log `=== S68, S69, S70 phase 2 Fri Oct  9 14:24:25 UTC 2026`, started 14:24:27, git 9899520+dirty | same segment | phase 1: `=== S70 phase 1 Fri Oct  9 14:19:53 UTC 2026`, started 14:19:54, **git a1112da (clean)** |
| first checks file | checks_followup_20261009_142433.json (added 1335f0b) | same | checks_s70o_20261009_141958.json (added 9899520, 2026-10-09T14:21:34Z) |
| first run start | `start 0.0 min ... S68_ORACLE_S8 / S68_SINGLE_S8` after timings (followup_runtime measured 14:26:31Z (log)) | `start 262.2 min ... S69_ORACLE seed 350` (offset in the same segment, so about 18:49Z; approximate) | phase 1 `start 0.0 min ... S70_ORACLE_DELTA seed 330` after timings (s70o_runtime measured 14:21:11Z (log)); phase 2 `start 485.5 min ... S70_HEBB seed 330` (about 22:32Z; approximate) |
| first store commit with records | 1335f0b 2026-10-09T14:42:55Z (2 records, S68_ORACLE_S8 300/301) | da5e67e 2026-10-09T19:19:31Z (2 records, S69_ORACLE 350/351) | 1335f0b 2026-10-09T14:42:55Z (6 records, S70_ORACLE_* 330-332); rh_lite_budget.json `{"phase1": "bound", "oracle_24k": {"HEBB": 3, "DELTA": 3}, "budget": 24000}` in the same commit |
| later changes to reading lines | none | none | none |
| report statement | report_3.md (9a6c2bf, 2026-10-09T23:33:58Z): matches (incomplete, see below) | matches | matches |

S68 readings as first committed (a1112da, lines 19-24, verbatim):
```
VALIDITY (fixed before any run): S = s is VALID if S68_ORACLE_S{s} binds on >= 1 of its 2 seeds within 43200; a reading
needs both S = 4 and S = 8 VALID, else UNTESTED.
READINGS (fixed before any run):
  "context-tagged keys scale"                          if S68_SINGLE_S8 BOUND >= 7/10;
  "the partition's advantage is in the number of streams"  if S68_SINGLE_S8 <= 3/10 while S68_SINGLE_S4 >= 7/10;
  otherwise "neither".
```
Arms (l.12, 14): `S68_SINGLE_S4, S68_SINGLE_S8   seeds 300-309: ...` / `S68_ORACLE_S4, S68_ORACLE_S8   seeds 300-301 (validity): ...`; code `LR, ITERS = 1e-3, 43200` / `SEEDS, OSEEDS = tuple(range(300, 310)), (300, 301)`.
Report_3: ""Context-tagged keys scale" needed ≥ 7/10 at S = 8. "The partition's advantage is in the number of streams"
needed ≥ 7/10 at S = 4, and it bound only 2/10." The report leaves out the "S68_SINGLE_S8 <= 3/10" part of the second
reading. The verdict is the same either way (S8 = 0/10 and S4 = 2/10 give "neither").

S69 readings as first committed (a1112da, lines 21-25, verbatim):
```
VALIDITY (fixed before any run): S69_ORACLE binds on >= 1 of its 2 seeds within 48000, else UNTESTED.
READINGS (fixed before any run):
  "the forget gate needs the delta rule"  if ROUTED <= FIXED + 1 and GDN <= FIXED + 1 (BOUND counts);
  "the forget gate alone suffices"        if ROUTED >= 8/10 or GDN >= 8/10;
  otherwise "neither". Printed beside S59's delta counts (FIXED 3/10, ROUTED 9/10, GDN 8/10) on the same seeds.
```
Arms (l.13, 19): `ARMS (seeds 350-359 — S59's, ...` / `S69_ORACLE  the perfect gate, k = 2, ... seeds 350-351 (validity).`; code `LR, ITERS = 1e-3, 48000`, `SEEDS = tuple(range(350, 360))`.
Report_3: "S69 "the forget gate needs the delta rule". ... the Hebbian channel binds 0 / 0 / 0", and in the table
"valid (≥ 1/2)" for the oracle at 1/2. The report does not restate the "<= FIXED + 1" rule, but 0 <= 0 + 1 satisfies it.

S70 reading as first committed (a1112da, lines 11-15 and 23-24, verbatim):
```
PHASE 1, ORACLE FIRST (seeds 330-332, Adam 1e-3, 24000): S70_ORACLE_HEBB (the perfect gate, Hebbian) and S70_ORACLE_DELTA
  (the perfect gate, delta β = 1, L2 keys). An oracle arm BINDS if >= 2 of its 3 seeds bind. If either does not bind,
  both are run once more at double the budget (S70_ORACLE_HEBB_2X, S70_ORACLE_DELTA_2X, 48000, the same seeds); if either
  still does not bind, S70 is UNTESTED and stops. The budget of phase 2 is the one at which both bound (24000 or 48000).
  The decision is written to explore_out/F/rh_lite_budget.json by decide() before phase 2 is queued.
READING (descriptive, fixed now): the BASELINE for Session G is S70_HEBB's DISCOVERED count (BOUND and ROUTED*@end
  alongside) at the phase-2 budget; S70_DELTA printed beside it. No other reading.
```
Arms (l.16): `PHASE 2 (seeds 330-349): S70_HEBB, S70_DELTA — LOCAL3 + SLOW ...`; code `OSEEDS, SEEDS = (330, 331, 332), tuple(range(330, 350))`.
Report_3: "**Session G's BASELINE: S70_HEBB DISCOVERED 0/20, BOUND 1/20, ROUTED*@end 0/20, KEY 19.**" with S70_DELTA beside it.
This matches the descriptive reading, and the budget file records phase 1 bound at 24000 (3/3 each).

Context: all three follow-up specs were written after S54 and S59 were reported (a82c62a, 2026-10-09T08:04:31Z), and
their docstrings cite those results. This is disclosed. S70 replaces S54(d), which was UNTESTED, with a new and lighter
task; S54(d)'s own docstring was never changed. The rule was kept for S70 phase 1, which started 3 s after the spec
commit with a clean tree at a1112da.

---------------------------------------------------------------------------------------------------------------------
## Summary
- All 7 screen files have one commit each, which carries the readings. **Nothing in any reading line (thresholds, readings, arms, seeds,
  outcomes, validity) changed after any run.** The report code (explore_f_reports*.py) did not change after the runs either.
- Every spec commit comes before its screen's first logged run start and before its first stored record. The gaps are:
  S53 6 s (04:14:49Z commit, 04:14:55Z start); S54 1 h 40 min to the first checks-only start and about 4 h to the first training run;
  S59 about 25 h; S70 3 s; S68/S69 about 4.5 min.
- S54(a)'s ">= 35/40 (31 + 4)" was written in 9ddda81 (2026-10-08T04:35:17Z), before any S54 run. It replaces the prompt's
  paired "≥ 4 vs 0", which cannot apply to an unpaired comparison with recorded runs. The docstring states the departure.
- The reports state the readings consistently with the docstrings. Small omissions, none of which changes a verdict:
  report_2 (b) leaves out "and c = 0"; report_3 S68 leaves out "S8 <= 3/10"; report_3 S69 does not restate "<= FIXED + 1".
- Cannot be determined from git: the real push or commit times (committer dates are self-reported), and what the "+dirty"
  working-tree changes were at run time (S54, S59, S68/S69 segments). Because the stores have no start-time fields, the
  run start times above come from the logs, not the stores.
