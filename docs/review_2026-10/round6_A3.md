# Plan item A3: `test_delta_recipe`'s L half and pooled block, R's Phase 5, I's report 3

*Referee V, 13 October 2026.* The round plan is `docs/reading/plan_2026-10-13.md` (main line, `6912e0b`), item A3.

**What was checked.**

| what | record | check | rows | MATCH | PARTIAL | DISAGREE | not record-checkable |
|---|---|---|---|---|---|---|---|
| `test_delta_recipe`, L half and pooled | main line `b03d353` (and X's log, `7d7661d`) | `check_delta_recipe_L.py` / `.md` | 66 | 55 | 7 | 1 | 3 |
| R's Phase 5 | `claude/replicate-R` `31c39fc` (protocol `550a736`) | `check_R_phase5.py` / `.md` | 38 | 32 | 2 | 0 | 4 |
| I's report 3 (`test_window_gate` on the GTX 1660 SUPER) | `claude/explore-I` `3d7e68e` (note `30807ec`, records `148c161`) | `check_I_report3.py` / `.md` | 65 | 64 | 0 | 0 | 1 |
| Amendment 2's statement, against X's log | main line `7d7661d` | `check_amendment2_log.py` / `_out.txt` | | exact | | | |

**Rules followed.**
- Nothing outside `docs/review_2026-10/` was changed.
- Nothing was trained, and no session code was run. The test functions were extracted by ast and run on the records.

## 1. `test_delta_recipe`: L and pooled

**Every pre-registered number reproduces.** That covers every count, pair, p-value, interval, band, CHECK and label in RESULT (L) and RESULT (POOLED, PRIMARY). The test's own `report()` regenerates L's in-run report (368 lines) and the `--report --also` file (412 lines) with 0 differing lines.

**The pooled result** (X + L, both parts VALID):

| claim | result | verdict |
|---|---|---|
| D1 | DELTA_SPLIT 35/40 against HEBB_SPLIT 35/40; 5 vs 5 (X 4 vs 1, L 1 vs 4); d = 0; difference +0.000, Newcombe [−0.162, +0.162] | "delta not shown worse by more than 2 pairs" |
| D2 | DELTA_SPLIT 35/40 against DELTA_SLOW 38/40; 0 vs 3; p = 1 | NOT SHOWN. DELTA_SLOW is ≥ 36/40, so NO ROOM |
| D4 | HEBB_SPLIT 35/40 against HEBB_SLOW 17/40; 19 vs 1; p = 2.0e-5 | SHOWN |
| D3 (secondary) | 30 shared binds: delta earlier on 24, later on 5, 1 tie; p = 0.00027 | |
| readings | "delta not shown worse by more than 2 pairs"; "delta: no room; Hebbian: helps" | |

The pooled D1 and D4 pairs, the D2 no-room count and D4's p were also recomputed here independently with `rlib`.

**L's run.**
- **What the records show.**
  - Seeds 1580–1599 (perfect gates 1580–1581), on the i7-12650H, with `meta.git` `8ae660d`.
  - The run code is the same as X's by ast. Only report code and the CHECKs 159/162 reference paths differ between `a581027` and `8ae660d`.
  - 84 of 84 runs, and no resume.
- **The two starts.** Both are headed "start 1", and the first left no run. The clock is UTC−6, from the longrun json's epoch.

**One false statement in RESULT (L).** It says the first start "printed its projection". It printed only the PROJECTION heading (`results/L/delta_recipe_L.log`:1252-1254). It was lost during the projection's timing, not after it.

**Loose statements:**
- **RESULT (L), "runs from 15:10 after the verification".** 15:10:00 (local) is `meta.started`, the end of verification. The first runs began about 19 min later, after the projection.
- **RESULT (L), "1125.8 min of runs".** That figure includes the ~19 min projection; the runs alone took about 1106.8 min.
- **RESULT (L) does not mention the environment change between the starts.** The torch build and worker count changed: torch 2.14.0+cu130 with 6 workers, then torch 2.14.1 with 5. X ran 2.14.0+cu130, so the pool spans two torch builds.
- **"D3 delta earlier 24 of 29".** This is in `b03d353`'s message and the plan's line 12. It is 24 of 30 shared binds; 29 is the untied count.
- **`results/README.md`, L's row.** It gives the commit as `a581027`; the table's rule is `meta.git`, which is `8ae660d`.
- **RESULT (X), "line for line".** X's log now shows that the in-run and regenerated reports differ on six lines: the claims header, the D1 interval's name, and the probe-blind lines. "Line for line" is true of the claims, bands and CHECKs 171/174, as `rev9/check_delta_recipe_X.md` said.
- **Not record-checkable: that the first start was lost to an OOM.** The records show a reboot (a new boot id) and no run line.

**A pre-registered label to read with care.** On delta, D2's printed b and c are 0 vs 3: the split arm lost three seeds the no-trigger arm bound (L 1581, 1582, 1591; two-sided p = 0.25). The pre-registered label is "no room", and it stands. The direction is against the split on delta.

**Amendment 2, closed against X's log.**
- At 15:36:08 (`55dac50`) and 15:37:52 (`2225f00`), exactly the amendment's 24 runs had finished. The next, DELTA_SPLIT 599, finished at 15:38:19.
- The process start is recorded at 02:12:34 UTC.
- No report was printed before the amendment.

Details are in the addendum to `test_check_delta_recipe_amendment2.md`. The reconstruction in `rev9/check_delta_recipe_X.md` is confirmed in every point.

**The plan's open item A2 is done.** X's log was committed at `7d7661d` (20:35:34 UTC), 8 minutes after the plan (`6912e0b`), which still lists it as open.

## 2. R's Phase 5

**Every number reproduces**, with R's outcome functions and with the main line's, which agree:

| arm | count | Wilson 95% | the main line pooled | "consistent" range | label |
|---|---|---|---|---|---|
| COPY | 18/20 | [0.699, 0.972] | 18/20 | 14..19 | consistent |
| RESET | 9/20 | [0.258, 0.658] | 6/20 | 3..10 | consistent |
| DELTA1 | 0/20 | [0, 0.161] | 0/20 | 0..3 | consistent |

- **Pairs.** COPY vs RESET is 9 vs 0 (p = 0.002); COPY vs DELTA1 is 18 vs 0 (p = 3.8e-6). The main line has 12 vs 0 and 18 vs 0.
- **The oracles** bound 2/2.

**Pre-registration is clean.**
- The protocol (`550a736`, 22:47:46 UTC on 12 October) precedes every record.
- Nothing of the protocol changed afterwards; the only diff to `phase5.py` is its RESULT block.
- Seeds 940–959 were unused anywhere before.
- **The start cannot be fixed.** A replay of the run's times puts it no later than 22:49:48, but a start a few minutes before the protocol commit cannot be excluded from the records (Phase 4 had the same pattern).

**The rule's power.** The "consistent" rule (x/20 inside the main line's pooled Wilson interval) separates little at n = 20. COPY reads "consistent" with probability 0.61 at a true rate of 0.70, and 0.64 at 0.95. R reads nothing beyond it, which is right.

**Partial statements:**
- **`replicate_report.md`:218-221, "From the written specification alone, an independent implementation reproduces …".**
  - COPY is R's Phase 4 recipe, which took the probe and the 32 body key positions from the main line's code (DIFF.md D-1, D-2). RESET uses the same trigger.
  - In Phase 3 R read `test_window_gate.py` at `404e3a0`, which holds the reset arm and operation.
  - The claim holds at most for DELTA1. The prompt itself made COPY the Phase 4 recipe, so this is overstated wording, not a hidden change.
- **`replicate_report.md`:214, "per-stream accuracy roughly even within a run".** The spread is 0.25 or more on 4 of the 20 DELTA1 runs (950, 951, 954, 959).

**Minor:**
- U-50 attributes the docstring's phrase to the spec.
- "(all at 1200)" (:197) is right for CEIL_DELTA_D8 only.
- ORACLE_ADAM 940 and 941 have identical evaluations at 1200 and 2400; the cause is not recorded, and it does not affect validity.

## 3. I's report 3: `test_window_gate` on the GTX 1660 SUPER

**Every number matches the records**, recomputed with the main line's own functions and independently with `rlib`; this file also re-checked them directly:

| arm | count |
|---|---|
| the recipe (WIN3_SPLIT_D8) | 19/20 |
| no split (WIN3_SLOW_D8) | 6/20 |
| recurrent (HINGE_D8) | 0/20 |
| CEIL_C (validity) | 2/2, so VALID |

- **Pairs.** 19 vs 0 (split vs recurrent), 13 vs 0 (split vs no split) and 6 vs 0 (no split vs recurrent).
- **p-values.** Exact one-sided McNemar gives 1.9e-6, 1.2e-4 and 0.016. These are the referee's; report 3 prints none, as its note fixed.

**The order is clean, and needs no clock.**
- Every record carries code SHA `05bbaaaac7a3`, recomputed from git objects. That code first exists at the note's commit `30807ec`.
- The note never changed after it.
- The descriptions generator changed three times, only in its print lines, and the report discloses each.

**The code that ran is the main line's own test code.**
- All 29 root modules are byte-identical to the main line's.
- They run through a thin device wrapper, `explore_gpu/mainline.py` and `device.py`, which builds on CPU and moves to CUDA.
- Report 3 disclaims bit-equivalence with the CPU runs ("outcomes, not curves").

**Seeds 2540–2559 are fresh** on every branch and in all history.

**The one not-record-checkable row** is report 3's account of which commit X's runs used. It changes no number.

## Errors in the referee's own earlier files, found here

All are dated in the files concerned.
- **`rev9/sixth_pass.md` §3 and `rev9/check_R_v6.md`:11 and row 8.** `550a736` also changes `replicate.py` and `test_replicate.py`, and Phase 5 was already running at 23:14.
- **No error found** in `rev9/check_delta_recipe_X.md`, `check_S75.md` or `check_S75_v6.md`. X's log confirms the X reconstruction in every point.
