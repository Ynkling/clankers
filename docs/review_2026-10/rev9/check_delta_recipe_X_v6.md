# test_delta_recipe on X (`8ae660d`) and the sixth draft's lines that cite it: referee check, v6

*Referee V. Generated in full by `python3 docs/review_2026-10/rev9/check_delta_recipe_X_v6.py > docs/review_2026-10/rev9/check_delta_recipe_X_v6.md` (standard library only; git objects only; nothing run or trained). The v6 copy of `check_delta_recipe_X.py`: every record recomputation is kept (sections 1-3 of the log below), every draft row is anchored by a verbatim fragment located in tex6, and rows were added for every changed sentence that cites the test.*

## Inputs and branch status

- **Draft:** `claude/rev9-draft` at `42c4557` (2026-10-12T23:14:04+00:00), `multichannel_hebbian_report_v9.tex` = tex6. Lines changed from `c5e4508` (tex5), in place: [36, 41, 54, 58, 201, 205, 256, 271, 349, 360, 368, 382, 441, 458, 477, 478, 495, 499, 516, 518]; the test is cited on [36, 41, 54, 256, 349, 360, 472, 476, 496, 499].
- **Main line at the draft's time** (`origin/claude/bdh-growth-hebbian-inference-w90069`, 23:14:04 UTC): head `80bfc1d`. Commits after the record commit `8ae660d`: `80bfc1d` 2026-10-12T22:51:55+00:00 results/L/longrun: split_copy_L's second start (json, exit 0), as the user s; `bbb600c` 2026-10-12T22:51:19+00:00 test_split_copy.py docstring (RESULT (L)): the three CHECKs printed rather t; `0754d61` 2026-10-12T22:35:26+00:00 docs/reading: round-5 prompts for sessions V, G, H, F, R and I (12 October, . `results/X/delta_recipe_results.json` and `test_delta_recipe.py` unchanged since `8ae660d`: True. **L's half has not been pushed** (`results/L/delta_recipe_results.json` at `80bfc1d`: False), and **X's log has not been pushed**: no path on any ref matching a delta log, `results/X/longrun/*` or `results/L/delta*` after `git fetch --all` (paths found: ['explore_out/G/s56_delta.log']; `explore_out/G/s56_delta.log` is session G's S56, unrelated).
- **Records:** `results/X/delta_recipe_results.json` at `8ae660d`: 84 records, all ok; meta.git a581027; meta.started '2026-10-11 05:27:31' (no zone; UTC by §3).

## Counts

| scope | rows | MATCH | PARTIAL | DISAGREE | NOT RECORD-CHECKABLE |
|---|---|---|---|---|---|
| RESULT (X) | 21 | 18 | 2 | 0 | 1 |
| commit messages | 2 | 0 | 2 | 0 | 0 |
| draft (tex6) | 35 | 30 | 2 | 0 | 3 |
| **total** | **58** | **48** | **6** | **0** | **4** |

check_delta_recipe_X (tex5) had 47 rows: MATCH 35, PARTIAL 8, DISAGREE 2, NOT RECORD-CHECKABLE 2.

## Every non-MATCH row

| line | statement | recomputed | status | note |
|---|---|---|---|---|
| RESULT | run started 02:12 UTC through longrun.py; runs began at 05:27 after the verification | meta.started 2026-10-11 05:27:31 (UTC by inference, .md §3); 02:12 is not in the records (test commit 02:12:29); runs were submitted after the projection, >= 05:48 | **PARTIAL** | 05:27:31 is when verification ended and the projection began; the first run started >= 21 min later. 02:12 rests on the RESULT and Amendment 2 alone (no log) |
| RESULT | the --report figures' claims, bands and checks equal the in-run report's line for line | regenerated both from the records: 18 lines identical, 3 only in A2's, 3 only in the in-run (the probe-blind lines, which Amendment 2 (f) changed; the claims and bands identical) | **PARTIAL** | True of claims and bands and of CHECKs 171/174; the probe-blind lines differ by construction (A2 (f)); the in-run line counts all checks incl. 2400. Without the log the in-run report is a reconstruction |
| msg 8ae660d | commit message: 'D3 delta earlier on 10 of 14, p = 0.090' | earlier 10 of 15 shared binds (14 untied), p = 0.0898 | **PARTIAL** | 14 is the untied count; the RESULT and the draft say 10 of 15 |
| msg 902c39a | draft commit title: 'D2 no room' | the test printed D2 NOT SHOWN; NO ROOM is pooled only and was not printed | **PARTIAL** | as tex5:256's 'leaving no room' |
| RESULT | Verification: CHECKs 130, 158, 159, 162, 161 (1.19e-7), 173 | not in the records (no log) | **NOT RECORD-CHECKABLE** |  |
| tex6:36 | abstract: '... and runs on the other' | no L file or log on any ref; no L record on any ref; session I's window_gate pre-run note (30807ec, explore-I, 22:44 UTC 12 Oct) says that run 'moved here from the RTX 3070, which ran out of host memory running it beside test_delta_recipe' (which machine holds the 3070 is not recorded); the round-5 prompts (0754d61) say L is running it | **NOT RECORD-CHECKABLE** | consistent with session I's note; not a record |
| tex6:36 | abstract: 'the split helps the Hebbian write, 10 vs 1' | D4 HEBB_SPLIT only 10 (s580, s583, s585, s587, s588, s591, s594, s596, s597, s598), HEBB_SLOW only 1 (s595); p = 0.0059; per machine the test prints 'SHOWN' (secondary); 'helps' is a pooled label (READINGS (pooled), split_label: 'helps' (SHOWN), 'no room', 'not shown'), not printed on X | **PARTIAL** | the pair is right; 'helps' is the pre-registered pooled label for D4, which waits for L, while tex6:256 says 'The pooled labels and readings wait for L' -- the same objection this check raised against 'no room' for D2 on X alone. fifth_pass.md §2 item 1's suggested fix used this word ('the split helps the Hebbian write, 10 vs 1'): the referee's own fix carried the pooled label onto one machine. 'shown on X (10 vs 1)' would match |
| tex6:36, tex6:41, tex6:54, tex6:360, tex6:476, tex6:496 | L's status across the draft: 'runs on the other' (36), 'L's half is next' and 'has not run' (41), '(X; L running)' (54), 'L's half is running' (360), next step 'L's half' (476), 'L to run' (496) | no L record on any ref; session I's window_gate pre-run note (30807ec, explore-I, 22:44 UTC 12 Oct) says that run 'moved here from the RTX 3070, which ran out of host memory running it beside test_delta_recipe' (which machine holds the 3070 is not recorded); the round-5 prompts (0754d61) say L is running it | **PARTIAL** | not record-checkable either way, but the draft is inconsistent with itself: three lines say L is running, three that it is next or has not run (tex6:41 was rewritten in this version and kept 'is next' and 'has not run') |
| tex6:54 | index row, first column: '(X; L running)' [changed: tex5 '(specified)'] | X recorded (8ae660d); L: no L record on any ref; session I's window_gate pre-run note (30807ec, explore-I, 22:44 UTC 12 Oct) says that run 'moved here from the RTX 3070, which ran out of host memory running it beside test_delta_recipe' (which machine holds the 3070 is not recorded); the round-5 prompts (0754d61) say L is running it | **NOT RECORD-CHECKABLE** | the X half right; 'L running' not in any record |
| tex6:256 | a difference two screens of ten seeds had not seen | other sessions' screens (H's S79 and the like); not in this record | **NOT RECORD-CHECKABLE** |  |

## Table

| line | statement | recomputed | status | note |
|---|---|---|---|---|
| RESULT | test commit a581027; 84/84 runs, none failed; one start | meta.git a581027; 84 records, all ok; no meta.starts (resume_meta would add one on a restart) | MATCH |  |
| RESULT | run started 02:12 UTC through longrun.py; runs began at 05:27 after the verification | meta.started 2026-10-11 05:27:31 (UTC by inference, .md §3); 02:12 is not in the records (test commit 02:12:29); runs were submitted after the projection, >= 05:48 | PARTIAL | 05:27:31 is when verification ended and the projection began; the first run started >= 21 min later. 02:12 rests on the RESULT and Amendment 2 alone (no log) |
| RESULT | 29.2 h wall clock (projection 45.3 h worst case) | projected_wall_h 45.28; reconstructed 02:12:29 -> end >= 10-12 07:20 = >= 29.13 h | MATCH | Not in the records. Consistent: 02:12:29 + 29.2 h = 07:24; the reconstructed end is >= 07:20 and the result commit 07:25:51. The test's own report prints wall time after verification (>= 25.9 h in-run; 0.0 min under --report), so 29.2 h is from the longrun start, not a printed figure |
| RESULT | Validity: CEIL_HEBB 2/2 (3600, 3600); CEIL_DELTA 2/2 (2400, 1200); both parts VALID | CEIL_HEBB 2/2 [3600, 3600]; CEIL_DELTA 2/2 [2400, 1200]; validity {'HEBB': (2, True), 'DELTA': (2, True)} | MATCH |  |
| RESULT | HEBB_SPLIT 16/20 (median transition 17400) | HEBB_SPLIT 16/20, median transition 17400 | MATCH |  |
| RESULT | HEBB_SLOW 7/20 | HEBB_SLOW 7/20, median transition 18000 | MATCH |  |
| RESULT | DELTA_SPLIT 19/20 (13200) | DELTA_SPLIT 19/20, median transition 13200 | MATCH |  |
| RESULT | DELTA_SLOW 19/20 (14400 | DELTA_SLOW 19/20, median transition 14400 | MATCH |  |
| RESULT | DELTA_SLOW per seed the same outcome as DELTA_SPLIT, both failing only on s594, MERGED | seeds with different outcomes none; MERGED [594], [594] | MATCH |  |
| RESULT | every BOUND ROUTED run has overlap <= 0.1 | {'HEBB_SPLIT': 16, 'HEBB_SLOW': 7, 'DELTA_SPLIT': 19, 'DELTA_SLOW': 19} vs counts; largest 0.0341 | MATCH |  |
| RESULT | D1: DELTA only 4, HEBB only 1, d = -3; +0.150, 95% [-0.082, +0.374] | DELTA only 4 [582, 586, 590, 595], HEBB only 1 [594], d = -3; newcombe10 +0.150 [-0.082, +0.374] (test's function: same to 0e+00) | MATCH | Per machine no label is printed (correct per spec); pooled, d = -3 would read 'delta not shown worse; ahead by >= 3 pairs (untested)' |
| RESULT | D2: 0 vs 0, p = 1, NOT SHOWN (DELTA_SLOW 19/20) | 0 vs 0, p = 1; code: NOT SHOWN, dropped by CHECK 174 [] | MATCH |  |
| RESULT | D4: 10 vs 1, p = 0.0059, SHOWN | 10 vs 1, p = 0.005859; code SHOWN | MATCH | p = 12/2048 |
| RESULT | D3: 15 shared binds, earlier 10, later 4, tied 1; p = 0.090; medians 13200 vs 16800 | 15 shared, earlier 10, later 4, tied 1; p = 0.0898; medians 13200 vs 16800 | MATCH |  |
| RESULT | Bands: HEBB_SPLIT MAJORITY [0.584, 0.919]; HEBB_SLOW MINORITY [0.181, 0.567]; DELTA_SPLIT, DELTA_SLOW RELIABLE [0.764, 0.991] | HEBB_SPLIT MAJORITY [0.584, 0.919]; HEBB_SLOW MINORITY [0.181, 0.567]; DELTA_SPLIT RELIABLE [0.764, 0.991]; DELTA_SLOW RELIABLE [0.764, 0.991] | MATCH |  |
| RESULT | Firings: Hebbian 39 in 20 runs, every target ok; delta 31 in 20 runs, 10 ok | HEBB_SPLIT 39 firings, 39 ok, 19 runs fired; DELTA_SPLIT 31, 10 ok, 18 runs fired | MATCH | 'in 20 runs' is the runs counted, as the code prints it, not the runs that fired |
| RESULT | Probe-blind: HEBB_SPLIT 1 of 181, alone 3; DELTA_SPLIT 1 of 142, alone 5 | HEBB_SPLIT 1 of 181, alone 3; DELTA_SPLIT 1 of 142, alone 5 (own and the code's) | MATCH |  |
| RESULT | CHECK 171 PASS (80 records); CHECK 174 PASS (40 pairs) | code: 'CHECK 171 fail_class_v2 vs v1 on every record of this test: 80 records, disagreements 0  -> PASS'; CHECK 174 40 pairs, broken none (own: none) | MATCH |  |
| RESULT | the --report figures' claims, bands and checks equal the in-run report's line for line | regenerated both from the records: 18 lines identical, 3 only in A2's, 3 only in the in-run (the probe-blind lines, which Amendment 2 (f) changed; the claims and bands identical) | PARTIAL | True of claims and bands and of CHECKs 171/174; the probe-blind lines differ by construction (A2 (f)); the in-run line counts all checks incl. 2400. Without the log the in-run report is a reconstruction |
| RESULT | the figures are Amendment 2's report regenerated with --report (2225f00) from the same results file | the RESULT's probe-blind figures (1 of 181, alone 3; 1 of 142, alone 5) are what only 2225f00's probe_blind prints (a581027's prints 1 of 201, 1 of 162, no 'alone'); 8ae660d's report functions = 2225f00's (AST) | MATCH | That both reports are kept in the log cannot be checked: no log is on any ref |
| msg 8ae660d | commit message: 'D3 delta earlier on 10 of 14, p = 0.090' | earlier 10 of 15 shared binds (14 untied), p = 0.0898 | PARTIAL | 14 is the untied count; the RESULT and the draft say 10 of 15 |
| msg 902c39a | draft commit title: 'D2 no room' | the test printed D2 NOT SHOWN; NO ROOM is pooled only and was not printed | PARTIAL | as tex5:256's 'leaving no room' |
| RESULT | Verification: CHECKs 130, 158, 159, 162, 161 (1.19e-7), 173 | not in the records (no log) | NOT RECORD-CHECKABLE |  |
| tex6:36 | abstract: 'its pre-registered test is complete on one machine (...) and runs on the other' [changed: tex5 'is the next main-line test, specified and reviewed'] | X: 84/84 records ok at 8ae660d; L: no results file on any ref at 80bfc1d (main line 80bfc1d) | MATCH | the 'one machine' half; 'runs on the other' is the next row |
| tex6:36 | abstract: '... and runs on the other' | no L file or log on any ref; no L record on any ref; session I's window_gate pre-run note (30807ec, explore-I, 22:44 UTC 12 Oct) says that run 'moved here from the RTX 3070, which ran out of host memory running it beside test_delta_recipe' (which machine holds the 3070 is not recorded); the round-5 prompts (0754d61) say L is running it | NOT RECORD-CHECKABLE | consistent with session I's note; not a record |
| tex6:36 | abstract: 'the delta recipe 19/20 against the Hebbian 16/20' (D1's arms; no pair count given) | DELTA_SPLIT 19/20, HEBB_SPLIT 16/20; discordant DELTA only 4 (s582, s586, s590, s595), HEBB only 1 (s594), d = -3 | MATCH | the 4 vs 1 is not stated in the abstract (it is at tex6:256 as d = -3) |
| tex6:36 | abstract: 'without a trigger 19/20 against 7/20, 12 vs 0' [changed at 42c4557: c916996 said '10 vs 1'] | DELTA_SLOW 19/20, HEBB_SLOW 7/20; DELTA_SLOW only 12 (s580, s582, s583, s585, s586, s587, s588, s590, s591, s596, s597, s598), HEBB_SLOW only 0; both 7 | MATCH | printed nowhere by the test (no claim compares the two no-trigger arms): the referee's pairing, as check_onetoone_v5 gave it; the fifth pass's §2 item 1, resolved |
| tex6:36 | abstract: 'the split helps the Hebbian write, 10 vs 1' | D4 HEBB_SPLIT only 10 (s580, s583, s585, s587, s588, s591, s594, s596, s597, s598), HEBB_SLOW only 1 (s595); p = 0.0059; per machine the test prints 'SHOWN' (secondary); 'helps' is a pooled label (READINGS (pooled), split_label: 'helps' (SHOWN), 'no room', 'not shown'), not printed on X | PARTIAL | the pair is right; 'helps' is the pre-registered pooled label for D4, which waits for L, while tex6:256 says 'The pooled labels and readings wait for L' -- the same objection this check raised against 'no room' for D2 on X alone. fifth_pass.md §2 item 1's suggested fix used this word ('the split helps the Hebbian write, 10 vs 1'): the referee's own fix carried the pooled label onto one machine. 'shown on X (10 vs 1)' would match |
| tex6:41 | Revision 9 paragraph: the test file checked against the amended spec before any arm's run | the check 6f2788d committed 2026-10-11 03:01:36 UTC; meta.started (verification end) 2026-10-11 05:27:31 UTC; first runs submitted >= 05:48:43 (§3) | MATCH | the referee's fourth-pass item 6, withdrawn at e733dca |
| tex6:41 | '(the check preceded the runs by two and a half hours; X's verification took three)' | check -> meta.started 2.43 h; check -> first submitted run >= 2.79 h; test commit a581027 (02:12:29) -> meta.started 3.25 h | MATCH | approximate: 2.43 h to the end of the verification, at least 2.79 h to the first run (the projection ran in between); the verification's 3.25 h counts from the test commit, the process start (02:12) being stated only by Amendment 2 and the RESULT |
| tex6:41 | test_delta_recipe ... complete on X (a second amendment ... committed during X's run ...); L's half is next | 84/84 records ok; 55dac50 15:36:08 between meta.started and the reconstructed end; no L file at 80bfc1d | MATCH |  |
| tex6:41 | test_delta_recipe's L half has not run | results/L/delta_recipe_results.json at 80bfc1d: False | MATCH | as of the records; no L record on any ref; session I's window_gate pre-run note (30807ec, explore-I, 22:44 UTC 12 Oct) says that run 'moved here from the RTX 3070, which ran out of host memory running it beside test_delta_recipe' (which machine holds the 3070 is not recorded); the round-5 prompts (0754d61) say L is running it |
| tex6:36, tex6:41, tex6:54, tex6:360, tex6:476, tex6:496 | L's status across the draft: 'runs on the other' (36), 'L's half is next' and 'has not run' (41), '(X; L running)' (54), 'L's half is running' (360), next step 'L's half' (476), 'L to run' (496) | no L record on any ref; session I's window_gate pre-run note (30807ec, explore-I, 22:44 UTC 12 Oct) says that run 'moved here from the RTX 3070, which ran out of host memory running it beside test_delta_recipe' (which machine holds the 3070 is not recorded); the round-5 prompts (0754d61) say L is running it | PARTIAL | not record-checkable either way, but the draft is inconsistent with itself: three lines say L is running, three that it is next or has not run (tex6:41 was rewritten in this version and kept 'is next' and 'has not run') |
| tex6:54 | index row: on X (secondary): D4 SHOWN 10 vs 1, D2 0 vs 0 with DELTA_SLOW 19/20, D1 d = -3; pooled waits for L | D4 10 vs 1 SHOWN; D2 0 vs 0, DELTA_SLOW 19/20; d = -3 | MATCH |  |
| tex6:54 | index row, first column: '(X; L running)' [changed: tex5 '(specified)'] | X recorded (8ae660d); L: no L record on any ref; session I's window_gate pre-run note (30807ec, explore-I, 22:44 UTC 12 Oct) says that run 'moved here from the RTX 3070, which ran out of host memory running it beside test_delta_recipe' (which machine holds the 3070 is not recorded); the round-5 prompts (0754d61) say L is running it | NOT RECORD-CHECKABLE | the X half right; 'L running' not in any record |
| tex6:256 | 'with the test file checked against the amended spec before any arm's run; run on X' [changed: tex5 'checked against the amended spec; running on X'] | 6f2788d 03:01:36 < meta.started 05:27:31 < first runs >= 05:48:43 (11 Oct, UTC) | MATCH | 'run on X' replaces 'running' (check_delta_recipe_X's tense note, resolved) |
| tex6:256 | Amendment 2's statement: 24 run lines read, 'none of them a claim, a reading or a Hebbian learned-gate run; two were the Hebbian perfect gate's' [changed: tex5 '... or a Hebbian run'] | Amendment 2's 24 = CEIL_HEBB 580, 581, CEIL_DELTA 580, 581, DELTA_SPLIT 580-598, DELTA_SLOW 580 (24 runs; §3: the set finished at 15:36:08 in 0.999 of the accepted draws, otherwise the same plus the boundary case DELTA_SPLIT 599); Hebbian perfect gates among them: 2; HEBB_SPLIT/HEBB_SLOW finished by then: 0 (first >= 10-11 23:55) | MATCH | check_delta_recipe_X's DISAGREE, resolved |
| tex6:256 | D1's first label: 'delta not shown worse; ahead by >= 3 pairs (untested)' (d <= -3; Amendment 2's wording, the first draft having read 'delta binds more often') [changed: tex5 quoted the first draft only] | D1_LABELS[0] at 8ae660d = 'delta not shown worse; ahead by >= 3 pairs (untested)'; at a581027 (before Amendment 2) = 'delta binds more often' | MATCH | check_delta_recipe_X's DISAGREE, resolved |
| tex6:256 | a ten-point deficit passes it 22-29% of the time at 40 pairs | D1_OC: P(d <= 2) 0.29/0.22 at DELTA 0.85 vs HEBB 0.95 | MATCH | Unchanged text; quoted from the test's D1_OC |
| tex6:256 | X's half is complete (seeds 580-599, 84 runs, 29.2 h; both perfect gates 2/2; CHECK 174 dropped no pair) | seeds 580-599, 84 runs; CEIL 2/2 and 2/2; dropped D2 [], D4 [] | MATCH | 29.2 h: Not in the records. Consistent: 02:12:29 + 29.2 h = 07:24; the reconstructed end is >= 07:20 and the result commit 07:25:51. The test's own report prints wall time after verification (>= 25.9 h in-run; 0.0 min under --report), so 29.2 h is from the longrun start, not a printed figure |
| tex6:256 | BR: DELTA_SPLIT 19/20, DELTA_SLOW 19/20 (same outcome every seed, both failing on one, a merge), HEBB_SPLIT 16/20, HEBB_SLOW 7/20 | HEBB_SPLIT 16/20, HEBB_SLOW 7/20, DELTA_SPLIT 19/20, DELTA_SLOW 19/20; differing seeds none; MERGED [594] | MATCH |  |
| tex6:256 | D1 d = -3 (delta ahead by three pairs, the first label's case; +0.15, 95% [-0.08, +0.37]) [', the first label's case' added] | d = -3 (4 vs 1); +0.150 [-0.082, +0.374]; d <= -3 -> D1_LABELS[0]: True | MATCH | per machine; the label itself is pooled and is not printed on X (the sentence does not say it was) |
| tex6:256 | D2 0 vs 0, NOT SHOWN on X, DELTA_SLOW 19/20 leaving at most one pair of room ('no room' is the pooled label, >= 36/40, and waits for L) [changed: tex5 'leaving no room'] | 0 vs 0, p = 1, NOT SHOWN; DELTA_SLOW 19/20 -> at most 1 DELTA_SPLIT-only pair; NO_ROOM = 36 (of 40, pooled); pooled no room iff L's DELTA_SLOW >= 17/20 | MATCH | check_delta_recipe_X's PARTIAL, resolved |
| tex6:256 | D4 SHOWN, 10 vs 1, p = 0.0059 | 10 vs 1, p = 0.0059 | MATCH |  |
| tex6:256 | D3 delta sooner on 10 of 15 shared binds, later on 4, p = 0.09 | 10 of 15, later 4, tie 1; p = 0.0898 | MATCH | Record commit 8ae660d's message says 'earlier on 10 of 14' (14 = the untied); the draft follows the RESULT |
| tex6:256 | every Hebbian firing on a labelled target (39 of 39), the delta firings on 10 of 31 | 39/39; 10/31 | MATCH |  |
| tex6:256 | the probe blocked 3 Hebbian and 5 delta checks alone | blocked by the probe alone: HEBB_SPLIT 3, DELTA_SPLIT 5 | MATCH | The other count (probe >= 0.95 while held-out < 0.95: 1 and 1) is not given |
| tex6:256 | on delta channels the split did not change a single seed's outcome, with the no-trigger arm already binding routed 19/20 where its Hebbian counterpart binds 7/20 [changed: tex5 'because ...'] | DELTA pairs: outcomes identical on all 20 seeds; DELTA_SLOW 19/20, HEBB_SLOW 7/20; DELTA_SPLIT fired in 18 of 20 runs (31 firings) | MATCH | descriptive, labelled 'pending L'; the unmarked inference ('because') removed: check_delta_recipe_X's PARTIAL, resolved |
| tex6:256 | the exact-copy split on Hebbian channels, 16/20 here, sits below the noisy copy's 18/20 of test_split_copy on other seeds | HEBB_SPLIT 16/20 (580-599); split_copy WIN3_SPLIT_D8 X 10/10 + L 8/10 = 18/20 (seeds 560-569, 1560-1569) | MATCH | split_copy's split also zeroes Adam's state (copy + noise + reset); the exact copy here does not, so two things differ |
| tex6:256 | a difference two screens of ten seeds had not seen | other sessions' screens (H's S79 and the like); not in this record | NOT RECORD-CHECKABLE |  |
| tex6:256 | The pooled labels and readings wait for L. | print_claims prints labels only pooled; no L file at 80bfc1d | MATCH |  |
| tex6:476 | next: test_delta_recipe's L half (X's is complete) | X 84/84; no L file at 80bfc1d | MATCH |  |
| tex6:496 | provenance: spec 05256c5, A1 32c121b / test a581027, A2 55dac50, 2225f00 / 8ae660d (X); X complete; L to run; 84 | meta.git a581027; record commit 8ae660d; 84 records; 55dac50 and 2225f00 committed during the run (between 05:27:31 and >= 10-12 07:20) | MATCH | 'L to run' vs 'L running' elsewhere: the consistency row above |
| tex6:499 | caption: X's recorded start times are UTC | meta.started 2026-10-11 05:27:31: UTC is the only zone consistent with the 02:12 start and 29.2 h (other zones shift the end by whole hours, §3) | MATCH | Inferred, not recorded: X's times carry no zone |
| tex6:472 | conclusion: the write rule is open ... on screens | cites no X result; X's D1 (d = -3) does not contradict it | MATCH | unchanged; no mention that the pre-registered test has run on X |
| tex6:360 | assessment: 'A pre-registered test is specified and complete on X; L's half is running.' [changed: tex5 'is specified.'] | X 84/84 at 8ae660d; L: no L record on any ref; session I's window_gate pre-run note (30807ec, explore-I, 22:44 UTC 12 Oct) says that run 'moved here from the RTX 3070, which ran out of host memory running it beside test_delta_recipe' (which machine holds the 3070 is not recorded); the round-5 prompts (0754d61) say L is running it | MATCH | check_delta_recipe_X's PARTIAL (stale), resolved; 'L's half is running' is not record-checkable (consistency row above) |
| tex6:349 | referee section: the passes found 'a pre-amendment label quoted for D1, a pair of discordant counts attached to the wrong arms in the abstract' | tex5:256 had "labelled ``delta binds more often'' ($d \leq -3$)": True; c916996:36 had 'without a trigger $19/20$ against $7/20$, 10 vs 1': True | MATCH |  |
| tex6:349 | referee section: 'the fifth pass found every number of every pre-registered claim reproducing, and withdrew one of its own earlier items (the test-file check did precede every run of test_delta_recipe)' | fifth_pass.md (e384a87): 'every number of a claim reproduces from its record' True; 'fourth_pass.md §1 item 6 is withdrawn' True; the check 6f2788d 03:01 before meta.started 05:27 | MATCH | the fifth pass's verdict was on c916996, before 78b7aaf and 42c4557 |

## Recomputation log (sections 0-3, unchanged from check_delta_recipe_X)

```
====================================================================================================
0. Inputs
====================================================================================================
  results/X/delta_recipe_results.json at 8ae660d: 84 records; meta.git a581027, cpu Intel(R) Xeon(R) Processor @ 2.10GHz, workers 4, started '2026-10-11 05:27:31' (no zone), torch 2.14.0+cu130, 'starts' (resumes) in meta: False
  projected_wall_h 45.28, projected_rest_h 45.28
  a log of X's delta_recipe run on any ref (any path with 'delta' and 'log'/'longrun', or results/X/longrun/*): ['explore_out/G/s56_delta.log']
  results/X/*delta* at 8ae660d: ['results/X/delta_recipe_results.json']
  a581027 2026-10-11T02:12:29+00:00 test_delta_recipe.py: the memory rule at eight streams (spec 05256c5 + Amendment 1 32c121b)
  55dac50 2026-10-11T15:36:08+00:00 specs/test_delta_recipe.md: Amendment 2 (during X's run; report rules only)
  2225f00 2026-10-11T15:37:52+00:00 test_delta_recipe: Amendment 2's report rules; CHECKs 159/162 read vendored records
  8ae660d 2026-10-12T07:25:51+00:00 Record test_delta_recipe on X: D2 0 vs 0 (DELTA_SLOW 19/20), D4 SHOWN 10 vs 1, D1 d = -3
  902c39a 2026-10-12T07:28:22+00:00 Revision 9 draft: test_delta_recipe's X half (DELTA_SPLIT 19/20, DELTA_SLOW 19/20, HEBB_SPLIT 16/20, HEBB_SLOW 7/20;
  42c4557 2026-10-12T23:14:04+00:00 Revision 9 draft, sixth version: every item of Referee V's fifth pass (e384a87) — the abstract's discordant pairs on
  test_delta_recipe.py: top-level defs/assignments that differ between 2225f00 and 8ae660d: none; module docstring differs: True
  ... and between a581027 and 2225f00: ['D1_LABELS', 'REFS', 'SPLIT_READINGS', 'broken_pairs', 'check_174', 'diff_str', 'pool2', 'print_claims', 'probe_blind', 'record_at', 'report', 'split_reading', 'tally', 'verify']

====================================================================================================
1. The test's own report, regenerated from the records (functions extracted from the sources at 8ae660d)
====================================================================================================
  test_window_gate constants: FIRST 4800, ACC_THR 0.95, RISE 0.02, MAX_SPLITS 3, GAP 4800, VALID_N 2; ROUTED_ACC 0.9; bands 0.9/0.5

  -- Amendment 2 (--report under 2225f00): validity, counts, claims, bands, checks (verbatim lines of the regenerated report) --
   |  HEBB: CEIL_HEBB bound 2/2 (s580 BOUND at 3600, s581 BOUND at 3600)  -> VALID
   |  DELTA: CEIL_DELTA bound 2/2 (s580 BOUND at 2400, s581 BOUND at 1200)  -> VALID
   |  HEBB_SPLIT   LOCAL3 + SLOW + the exact-copy split, Hebbian                 BOUND ROUTED 16/20   overlap <= 0.1 16   MERGED  3   completed 20/20   transition 17400 [10800, 32400]
   |  HEBB_SLOW    LOCAL3 + SLOW, no trigger, Hebbian (Amendment 1)              BOUND ROUTED  7/20   overlap <= 0.1  7   MERGED 13   completed 20/20   transition 18000 [10800, 27600]
   |  DELTA_SPLIT  LOCAL3 + SLOW + the exact-copy split, delta (beta 1, L2 keys) BOUND ROUTED 19/20   overlap <= 0.1 19   MERGED  1   completed 20/20   transition 13200 [6000, 37200]
   |  DELTA_SLOW   LOCAL3 + SLOW, no trigger, delta (beta 1, L2 keys)            BOUND ROUTED 19/20   overlap <= 0.1 19   MERGED  1   completed 20/20   transition 14400 [6000, 25200]
   |PER MACHINE (SECONDARY): machine X
   |  D1  DELTA_SPLIT vs HEBB_SPLIT (BOUND ROUTED): 19/20 vs 16/20; DELTA_SPLIT only 4, HEBB_SPLIT only 1; d = HEBB only - DELTA only = -3
   |      rate difference DELTA_SPLIT - HEBB_SPLIT +0.150, 95% [-0.082, +0.374] (the hybrid score interval with continuity-corrected phi; method 10 as Fagerland, Lydersen and Laake 2014 give it)
   |     (per machine: the rate difference only, no label)
   |  D2  DELTA_SPLIT beats DELTA_SLOW (BOUND ROUTED): 19/20 vs 19/20; DELTA_SPLIT only 0, DELTA_SLOW only 0; p = 1
   |     *** D2: NOT SHOWN ***
   |  D4  HEBB_SPLIT beats HEBB_SLOW (BOUND ROUTED): 16/20 vs 7/20; HEBB_SPLIT only 10, HEBB_SLOW only 1; p = 0.005859
   |     *** D4: SHOWN ***
   |  D3  (secondary) on the 15 seeds both split arms are BOUND ROUTED: median transition DELTA_SPLIT 13200, HEBB_SPLIT 16800; DELTA earlier on 10, later on 4, tied 1; one-sided sign test p = 0.08978 (DELTA earlier)
   |  band HEBB_SPLIT   (BOUND ROUTED): 16/20 -> MAJORITY   Wilson 95% [0.584, 0.919]  (RELIABLE >= 18, MAJORITY >= 10)
   |  band HEBB_SLOW    (BOUND ROUTED): 7/20 -> MINORITY   Wilson 95% [0.181, 0.567]  (RELIABLE >= 18, MAJORITY >= 10)
   |  band DELTA_SPLIT  (BOUND ROUTED): 19/20 -> RELIABLE   Wilson 95% [0.764, 0.991]  (RELIABLE >= 18, MAJORITY >= 10)
   |  band DELTA_SLOW   (BOUND ROUTED): 19/20 -> RELIABLE   Wilson 95% [0.764, 0.991]  (RELIABLE >= 18, MAJORITY >= 10)
   |  (the readings are pooled only)
   |
   |  CHECK 171 fail_class_v2 vs v1 on every record of this test: 80 records, disagreements 0  -> PASS
   |  CHECK 174 each split arm's curve equals its no-trigger arm's on the same seed through the first firing: 40 pairs, PASS
   |  probe-blind checks, HEBB_SPLIT (checks at update >= 4800): 1 of 181 read >= 0.95 on the probe while the held-out accuracy was < 0.95; blocked by the probe alone 3
   |  probe-blind checks, DELTA_SPLIT (checks at update >= 4800): 1 of 142 read >= 0.95 on the probe while the held-out accuracy was < 0.95; blocked by the probe alone 5
   |  HEBB_SPLIT: 39 firings in 20 runs (per run [3, 1, 3, 1, 2, 3, 3, 2, 2, 0, 3, 3, 1, 1, 1, 3, 2, 2, 2, 1]); targets labelled ok 39/39
   |  DELTA_SPLIT: 31 firings in 20 runs (per run [0, 1, 1, 2, 3, 2, 2, 2, 2, 1, 1, 2, 1, 2, 3, 3, 1, 1, 1, 0]); targets labelled ok 10/31

  -- in-run (a581027): validity, counts, claims, bands, checks (verbatim lines of the regenerated report) --
   |  HEBB: CEIL_HEBB bound 2/2 (s580 BOUND at 3600, s581 BOUND at 3600)  -> VALID
   |  DELTA: CEIL_DELTA bound 2/2 (s580 BOUND at 2400, s581 BOUND at 1200)  -> VALID
   |  HEBB_SPLIT   LOCAL3 + SLOW + the exact-copy split, Hebbian                 BOUND ROUTED 16/20   overlap <= 0.1 16   MERGED  3   completed 20/20   transition 17400 [10800, 32400]
   |  HEBB_SLOW    LOCAL3 + SLOW, no trigger, Hebbian (Amendment 1)              BOUND ROUTED  7/20   overlap <= 0.1  7   MERGED 13   completed 20/20   transition 18000 [10800, 27600]
   |  DELTA_SPLIT  LOCAL3 + SLOW + the exact-copy split, delta (beta 1, L2 keys) BOUND ROUTED 19/20   overlap <= 0.1 19   MERGED  1   completed 20/20   transition 13200 [6000, 37200]
   |  DELTA_SLOW   LOCAL3 + SLOW, no trigger, delta (beta 1, L2 keys)            BOUND ROUTED 19/20   overlap <= 0.1 19   MERGED  1   completed 20/20   transition 14400 [6000, 25200]
   |PER MACHINE (SECONDARY): machine X
   |  D1  DELTA_SPLIT vs HEBB_SPLIT (BOUND ROUTED): 19/20 vs 16/20; DELTA_SPLIT only 4, HEBB_SPLIT only 1; d = HEBB only - DELTA only = -3
   |      rate difference DELTA_SPLIT - HEBB_SPLIT +0.150, 95% [-0.082, +0.374] (Newcombe, paired)
   |     (per machine: the rate difference only, no label)
   |  D2  DELTA_SPLIT beats DELTA_SLOW (BOUND ROUTED): 19/20 vs 19/20; DELTA_SPLIT only 0, DELTA_SLOW only 0; p = 1
   |     *** D2: NOT SHOWN ***
   |  D4  HEBB_SPLIT beats HEBB_SLOW (BOUND ROUTED): 16/20 vs 7/20; HEBB_SPLIT only 10, HEBB_SLOW only 1; p = 0.005859
   |     *** D4: SHOWN ***
   |  D3  (secondary) on the 15 seeds both split arms are BOUND ROUTED: median transition DELTA_SPLIT 13200, HEBB_SPLIT 16800; DELTA earlier on 10, later on 4, tied 1; one-sided sign test p = 0.08978 (DELTA earlier)
   |  band HEBB_SPLIT   (BOUND ROUTED): 16/20 -> MAJORITY   Wilson 95% [0.584, 0.919]  (RELIABLE >= 18, MAJORITY >= 10)
   |  band HEBB_SLOW    (BOUND ROUTED): 7/20 -> MINORITY   Wilson 95% [0.181, 0.567]  (RELIABLE >= 18, MAJORITY >= 10)
   |  band DELTA_SPLIT  (BOUND ROUTED): 19/20 -> RELIABLE   Wilson 95% [0.764, 0.991]  (RELIABLE >= 18, MAJORITY >= 10)
   |  band DELTA_SLOW   (BOUND ROUTED): 19/20 -> RELIABLE   Wilson 95% [0.764, 0.991]  (RELIABLE >= 18, MAJORITY >= 10)
   |  (the readings are pooled only)
   |
   |  CHECK 171 fail_class_v2 vs v1 on every record of this test: 80 records, disagreements 0  -> PASS
   |  CHECK 174 each split arm's curve equals its no-trigger arm's on the same seed through the first firing: 40 pairs, PASS
   |  probe-blind checks, HEBB_SPLIT: 1 of 201 checks read >= 0.95 on the probe while the held-out accuracy was < 0.95
   |  probe-blind checks, DELTA_SPLIT: 1 of 162 checks read >= 0.95 on the probe while the held-out accuracy was < 0.95
   |  HEBB_SPLIT: 39 firings in 20 runs (per run [3, 1, 3, 1, 2, 3, 3, 2, 2, 0, 3, 3, 1, 1, 1, 3, 2, 2, 2, 1]); targets labelled ok 39/39
   |  DELTA_SPLIT: 31 firings in 20 runs (per run [0, 1, 1, 2, 3, 2, 2, 2, 2, 1, 1, 2, 1, 2, 3, 3, 1, 1, 1, 0]); targets labelled ok 10/31

  in-run vs Amendment 2 report, claims/bands/checks block: 18 lines identical; only in Amendment 2's 3; only in the in-run 3
   in-run only  |      rate difference DELTA_SPLIT - HEBB_SPLIT +0.150, 95% [-0.082, +0.374] (Newcombe, paired)
   in-run only  |  probe-blind checks, HEBB_SPLIT: 1 of 201 checks read >= 0.95 on the probe while the held-out accuracy was < 0.95
   in-run only  |  probe-blind checks, DELTA_SPLIT: 1 of 162 checks read >= 0.95 on the probe while the held-out accuracy was < 0.95
   A2 only      |      rate difference DELTA_SPLIT - HEBB_SPLIT +0.150, 95% [-0.082, +0.374] (the hybrid score interval with continuity-corrected phi; method 10 as Fagerland, Lydersen and Laake 2014 give it)
   A2 only      |  probe-blind checks, HEBB_SPLIT (checks at update >= 4800): 1 of 181 read >= 0.95 on the probe while the held-out accuracy was < 0.95; blocked by the probe alone 3
   A2 only      |  probe-blind checks, DELTA_SPLIT (checks at update >= 4800): 1 of 142 read >= 0.95 on the probe while the held-out accuracy was < 0.95; blocked by the probe alone 5
  whole reports: 369 vs 369 lines; lines that differ (position by position): 6
  claims header A2: 'PRE-REGISTERED CLAIMS (Amendments 1-2: D1 labelled by d with a paired interval; D2, D4 exact one-sided McNemar, SHOWN if'
  claims header run: 'PRE-REGISTERED CLAIMS (Amendment 1: D1 labelled by d with a paired interval; D2, D4 exact one-sided McNemar, SHOWN if p '

====================================================================================================
2. Independent recomputation (rlib outcomes; own statistics)
====================================================================================================
  rlib transition (first eval >= 0.95 held to the end) = record's transition on all 84: True; rlib outcome = the test's success() on all 84: True; stream_gate ties decided by ch_map: 0
  CEIL_HEBB     2/2  median transition 3600  not bound(-routed): 
  CEIL_DELTA    2/2  median transition 1800  not bound(-routed): 
  HEBB_SPLIT   16/20  median transition 17400  not bound(-routed): s582 MERGED (2 share), s586 STREAM-PARTIAL one-to-one, s590 MERGED (2 share), s595 MERGED (2 share)
  HEBB_SLOW     7/20  median transition 18000  not bound(-routed): s580 MERGED (2 share), s582 MERGED (3 share), s583 MERGED (2 share), s585 MERGED (3 share), s586 MERGED (2 share), s587 MERGED (2 share), s588 MERGED (2 share), s590 MERGED (2 share), s591 MERGED (2 share), s594 MERGED (2 share), s596 MERGED (2 share), s597 MERGED (3 share), s598 MERGED (2 share)
  DELTA_SPLIT  19/20  median transition 13200  not bound(-routed): s594 MERGED (2 share)
  DELTA_SLOW   19/20  median transition 14400  not bound(-routed): s594 MERGED (2 share)
  validity: CEIL_HEBB 2/2 [3600, 3600]; CEIL_DELTA 2/2 [2400, 1200]
  BOUND ROUTED with overlap <= 0.1: {'HEBB_SPLIT': 16, 'HEBB_SLOW': 7, 'DELTA_SPLIT': 19, 'DELTA_SLOW': 19} (largest overlap over BR runs 0.0341); MERGED: {'HEBB_SPLIT': [582, 590, 595], 'HEBB_SLOW': [580, 582, 583, 585, 586, 587, 588, 590, 591, 594, 596, 597, 598], 'DELTA_SPLIT': [594], 'DELTA_SLOW': [594]}
  DELTA_SPLIT vs DELTA_SLOW: seeds with a different outcome string: none; different BR: none
  D1: both 15, DELTA_SPLIT only 4 [582, 586, 590, 595], HEBB_SPLIT only 1 [594], neither 0; d = HEBB only - DELTA only = -3; Amendment 2 label (pooled only; here per machine) would be 'delta not shown worse; ahead by >= 3 pairs (untested)'
      DELTA - HEBB: referee's newcombe10 +0.1500 [-0.0823, +0.3745]; test's newcombe_paired +0.1500 [-0.0823, +0.3745]; largest difference 0.0e+00
      one-sided McNemar DELTA over HEBB (printed nowhere, for the label's evidence): p = 0.1875
  D2: DELTA_SPLIT 19/20 vs DELTA_SLOW 19/20; DELTA_SPLIT only 0 [], DELTA_SLOW only 0 []; one-sided exact McNemar p = 1.000000 -> NOT SHOWN
  D4: HEBB_SPLIT 16/20 vs HEBB_SLOW 7/20; HEBB_SPLIT only 10 [580, 583, 585, 587, 588, 591, 594, 596, 597, 598], HEBB_SLOW only 1 [595]; one-sided exact McNemar p = 0.005859 -> SHOWN
  no-room rule (pooled >= 36/40): X's DELTA_SLOW 19/20, HEBB_SLOW 7/20. Per machine the code prints no NO ROOM (print_claims: 'pooled and d["old"] >= NO_ROOM'); pooled, delta is NO ROOM iff L's DELTA_SLOW >= 17/20 (on the pairs CHECK 174 keeps); Hebbian cannot be (max 27/40)
  on X alone at most 1 DELTA_SPLIT-only pair was possible: best p = 0.50, so D2 could not be SHOWN on X whatever the split did
  D3: 15 seeds both split arms BR; delta earlier 10, later 4, tied 1; one-sided sign p = 0.0898; medians DELTA 13200, HEBB 16800
      rows: s580 21600/6000, s581 16800/10800, s583 15600/13200, s584 18000/26400, s585 32400/13200, s587 15600/26400, s588 20400/10800, s589 10800/10800, s591 22800/20400, s592 12000/13200, s593 14400/22800, s596 27600/21600, s597 15600/12000, s598 18000/14400, s599 15600/12000
  band HEBB_SPLIT   16/20 MAJORITY  Wilson [0.584, 0.919]
  band HEBB_SLOW    7/20 MINORITY  Wilson [0.181, 0.567]
  band DELTA_SPLIT  19/20 RELIABLE  Wilson [0.764, 0.991]
  band DELTA_SLOW   19/20 RELIABLE  Wilson [0.764, 0.991]
  CHECK 174 (own): 40 pairs, broken none; split runs that never fired (whole curve compared): {'DELTA_SPLIT': [580, 599], 'HEBB_SPLIT': [589]}
      of those, whole curves identical to the no-trigger arm's: {'DELTA_SPLIT': [580, 599], 'HEBB_SPLIT': [589]}
  probe-blind (own; checks >= 4800): HEBB_SPLIT 1 of 181, blocked by the probe alone 3 (all check rows incl. 2400: 201); DELTA_SPLIT 1 of 142, blocked by the probe alone 5 (all check rows incl. 2400: 162)
  firings HEBB_SPLIT: 39 in 20 runs (19 runs fired), targets labelled ok 39/39
  firings DELTA_SPLIT: 31 in 20 runs (18 runs fired), targets labelled ok 10/31
  CPU labels in the 84 records: {'Intel(R) Xeon(R) Processor @ 2.10GHz': 84}; sum of secs 101.11 h

====================================================================================================
3. Amendment 2's timing, reconstructed from the records (no log on any ref)
====================================================================================================
  meta.started 2026-10-11 05:27:31 (UTC); projection lower bound 1272 s -> first runs submitted >= 05:48:43; last run ends >= 2026-10-12 07:20:06 (makespan 25.523 h); result commit 2026-10-12 07:25:51: slack <= 345 s
  from the test's commit (02:12:29) to the earliest end: 29.13 h; to the result commit 29.22 h; verification + start 195 min
  the test's own 'total wall clock after verification' would be >= 25.88 h; under --report it prints 0.0 min
  finish times, no overhead (each a lower bound), around the two commits:
   # 1 CEIL_HEBB    581  start 05:48:43  end 05:56:12  ( -34795 s from 55dac50)
   # 2 CEIL_HEBB    580  start 05:48:43  end 05:56:12  ( -34795 s from 55dac50)
   # 3 CEIL_DELTA   581  start 05:48:43  end 06:03:14  ( -34373 s from 55dac50)
   # 4 CEIL_DELTA   580  start 05:48:43  end 06:07:56  ( -34091 s from 55dac50)
   #20 DELTA_SPLIT  596  start 11:31:25  end 14:13:08  (  -4980 s from 55dac50)
   #21 DELTA_SPLIT  594  start 10:26:26  end 14:30:37  (  -3931 s from 55dac50)
   #22 DELTA_SPLIT  598  start 13:31:17  end 14:59:27  (  -2200 s from 55dac50)
   #23 DELTA_SPLIT  595  start 11:30:40  end 15:10:13  (  -1555 s from 55dac50)
   #24 DELTA_SLOW   580  start 14:30:37  end 15:19:57  (   -971 s from 55dac50)
   #25 DELTA_SPLIT  599  start 14:13:08  end 15:36:04  (     -4 s from 55dac50)
   #26 DELTA_SLOW   581  start 14:59:27  end 16:18:54  (  +2566 s from 55dac50)
   #27 DELTA_SLOW   582  start 15:10:13  end 16:31:31  (  +3324 s from 55dac50)
   #28 DELTA_SLOW   583  start 15:19:57  end 16:45:49  (  +4181 s from 55dac50)
  finished at 15:36:08, no overhead: 25: CEIL_DELTA|580, CEIL_DELTA|581, CEIL_HEBB|580, CEIL_HEBB|581, DELTA_SLOW|580, DELTA_SPLIT|580, DELTA_SPLIT|581, DELTA_SPLIT|582, DELTA_SPLIT|583, DELTA_SPLIT|584, DELTA_SPLIT|585, DELTA_SPLIT|586, DELTA_SPLIT|587, DELTA_SPLIT|588, DELTA_SPLIT|589, DELTA_SPLIT|590, DELTA_SPLIT|591, DELTA_SPLIT|592, DELTA_SPLIT|593, DELTA_SPLIT|594, DELTA_SPLIT|595, DELTA_SPLIT|596, DELTA_SPLIT|597, DELTA_SPLIT|598, DELTA_SPLIT|599
  Monte Carlo: projection +U(0, 360 s), per-run overhead U(0, omax), omax U(0, 8 s); 3371 of 4000 draws end before the result commit
   at 15:36:08: 24 runs: 0.999   = the amendment
   at 15:36:08: 25 runs (vs the amendment: + [('DELTA_SPLIT', 599)], - []): 0.001
   at 15:37:52: 24 runs: 0.692   = the amendment
   at 15:37:52: 25 runs (vs the amendment: + [('DELTA_SPLIT', 599)], - []): 0.308
  DELTA_SPLIT|599's end relative to 15:36:08 over the accepted draws: min -3 s, median +163 s, max +338 s
  DELTA_SLOW|580's end relative to 15:36:08: min -970 s, max -628 s
  DELTA_SLOW|581's end relative to 15:36:08: min +2567 s, max +2909 s
  DELTA_SPLIT|599's worker chain (no overhead): DELTA_SPLIT|599 <- DELTA_SPLIT|596 <- DELTA_SPLIT|592 <- DELTA_SPLIT|587 <- DELTA_SPLIT|583 <- CEIL_DELTA|580 (6 runs, + the projection): >= 4 s of unrecorded time on it puts its end after 15:36:08, >= 108 s after 15:37:52
  zone of meta.started from git and records alone: UTC-0.10 h <= offset <= UTC+3.25 h, i.e. whole-hour zones UTC, +1, +2, +3; with the RESULT's 02:12 UTC start and 29.2 h only UTC fits (end >= 07:20 local vs 07:24 UTC)
  earliest Hebbian learned-gate run end (lower bound) 2026-10-11 23:55; second DELTA_SLOW end 16:18:54 (lower bound)
   if meta.started were UTC-1, 15:36:08 UTC would be 14:36:08 in its clock: 21 runs finished (no overhead)
   if meta.started were UTC+1, 15:36:08 UTC would be 16:36:08 in its clock: 27 runs finished (no overhead)
   if meta.started were UTC+2, 15:36:08 UTC would be 17:36:08 in its clock: 30 runs finished (no overhead)
   if meta.started were UTC+3, 15:36:08 UTC would be 18:36:08 in its clock: 32 runs finished (no overhead)

====================================================================================================
4. RESULT (X) block and the draft
====================================================================================================
  test_split_copy WIN3_SPLIT_D8 (noisy copy + Adam reset): X 10/10 seeds 560-569, L 8/10 seeds 1560-1569
```

## What could not be checked

- X's log (on no ref): the per-run lines and timestamps, the verification CHECKs (130, 158, 159, 161, 162, 172, 173), that both reports are kept, the 02:12 process start.
- L's half: not on any ref at the draft's time; 'L running' / 'runs on the other' rest on session I's note and the prompts.
- The time zone of meta.started (inferred, §3).
