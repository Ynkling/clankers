<!-- Appendix to docs/review_2026-10/preregistration.md. Produced for the referee by a delegated read-only audit of branch E (git log/show/diff and JSON via git show only); the referee spot-checked its load-bearing claims against git (see preregistration.md). -->

# Pre-registration audit, branch origin/claude/outside-ideas (session E): batches 16-19

Read-only audit. I used only `git log`, `git show`, `git diff` and `python3` on JSON read through `git show`. I did not check anything out, run anything or modify anything.

**Timezones.** Every commit time below is the committer date from `%cI`, and every one is `+00:00` (UTC). Author dates equal committer dates on every commit I checked. The store field `meta.provenance.started` has **no timezone**: it is written by `time.strftime("%Y-%m-%d %H:%M:%S")` (explore_common.py:97), which uses the container's local time. Each `started` value falls 2-10 s after a UTC commit made just before it (for example 14:53:07 against 0b63873 at 14:53:04Z), so it is very probably UTC. The field itself cannot prove that.

**Per-record times.** No run record has a wall-clock timestamp. Records carry `secs`/`secs_wall` (durations) and `git`. `git` is `ec.git_head()`, set in `rec.update(...)` *after* `run_job` returns (explore_common16.py:180, explore_common19.py:158), so it is HEAD when the run finished, not when it started. "+dirty" means tracked files were modified at that moment, and git cannot show which. Exact run start times therefore **cannot be determined from git**. The bound I give is: the run had finished by the store commit that first contains it.

**Method for item 3.**
- I ran `git log -p origin/claude/outside-ideas -- <file>` for each screen file.
- I listed every commit touching `explore_*.py` since 2026-10-06. The last commit before each batch's `started` is shown below, and none comes after it.
- I compared the SHA-1 of each docstring at its first commit with the one at the branch tip. All are identical except explore_keysplits.py, whose only change came before the dry run.
- I checked that no non-`explore_*` file changed on the branch since 2026-10-06. Only `explore_*.py` and `explore_out/` did.
- I checked that each store's `meta.arms` (arm list + seeds), `meta.code_shas`, `meta.cut` and `meta.provenance` are byte-identical in every commit of the store.
- I scanned the README history from 09c1192 on for `-`/`+` lines in the verdict tables.
- The branch has no merge commits.

---

## Batch 16: S43 (window_recipe) and S44 (window_scale)

| item | S43 `explore_window_recipe.py` | S44 `explore_window_scale.py` |
|---|---|---|
| Readings first committed | 93bdfe6 2026-10-06T13:48:18+00:00 "Batch 16 code: S43 window_recipe, S44 window_scale (+ SPLIT arm), ..." (the file's only commit) | same commit 93bdfe6 (the file's only commit) |
| Labelled dry run | started `2026-10-06 13:48:26` (no tz), git 93bdfe6; first dry store commit c3ec737 2026-10-06T14:16:16+00:00; labelled complete 4c67121 2026-10-06T14:52:56+00:00. Store dry_window_recipe_results.json: meta.dry true, 2 records, all dry | same dry run; dry_window_scale_results.json: 7 records, all dry |
| Real batch provenance | `started` `2026-10-06 14:53:07` (no tz), git `0b63873+dirty` (0b63873 = 2026-10-06T14:53:04+00:00), CPU Xeon 2.80GHz; empty store skeleton 82e897b 2026-10-06T15:10:56+00:00 | same `started`/git; skeleton 82e897b |
| First commit with a non-dry record | **8e64157 2026-10-06T19:45:33+00:00**, `LOCAL3_SLOW\|160` (record git 6d3f20d+dirty = 19:44:00Z, CPU 2.10GHz, secs_wall 102) | **2e7966b 2026-10-06T15:48:20+00:00**, `LOCAL3_SLOW_D8_M\|261` (record git 3d950d9 = 15:23:19Z, 2.80GHz, secs_wall 2231) |
| Code SHA, dry vs real | 46867e792241 in both | 01767493e840 in both |
| Reading changes after first run | **none** | **none** |
| README verdict section | added in 09c1192 2026-10-07T11:06:22+00:00; its lines were never edited afterwards (only `+` lines) | same |

S43 reading lines as first committed (93bdfe6), quoted verbatim:
```
READINGS (DISCOVERED = bound and final VAL cos < 0.5; fixed before any run)
- LOCAL3_SLOW vs SLOW_HINGE over 160-199: "the window gate replaces the recurrent gate and the hinge" if LOCAL3_SLOW
  >= 36/40 and (SLOW_HINGE only) - (LOCAL3_SLOW only) <= 1; "it does not" if LOCAL3_SLOW <= 30/40; otherwise neither.
- vs SLOW_MEM (24/40): printed with McNemar (exact, two-sided). Also printed: LOCAL3 over 160-199 (batch 1's records
  + the new runs) and LOCAL3_SLOW vs LOCAL3.
DIAGNOSTICS: failure classes (prediction: no POSITION; KEY failures fall from 4/20 to <= 1/20 on 160-179);
```
Code constant in the same commit: `REPLACE_N, REPLACE_D, NOT_N = 36, 1, 30`. Seeds: `OLD = tuple(range(160, 180))`, `NEW = tuple(range(180, 200))`.

S44 reading lines as first committed (93bdfe6), quoted verbatim:
```
READINGS (BOUND ROUTED at 28800; fixed before any run):
(a) "carries to four streams" if either arm has >= 16/20 and is not worse than its reference by more than 1 discordant
    pair (reference only - arm only <= 1); otherwise "not shown".
(b) "breaks the eight-stream stall" if either of LOCAL3_SLOW_D8_A and LOCAL3_SLOW_D8_M has >= 3/10; "does not" if both
    have 0/10; otherwise neither. An arm cut by the runtime rule drops out of "either"/"both".
    LOCAL3_SLOW_D8_A_SPLIT: its own line with the same thresholds (>= 3/10, 0/10), and McNemar against LOCAL3_SLOW_D8_A;
    it does not enter (b)'s "either arm".
```
Code constants: `CARRY_N, CARRY_D = 16, 1`, `BREAK_N, NOT_N = 3, 0`, `SA = tuple(range(240, 260))`, `SB = tuple(range(260, 270))`.

**Harness commits to batch 16 after the first real run.** Neither touches a reading, threshold, arm list, seed or outcome; both are packing and job order only.
- 14f9019 2026-10-06T23:13:49+00:00 (explore_batch16.py: sizes unobserved arms by the CPU's observed/projected ratio)
- ed6f8ab 2026-10-07T00:48:37+00:00 (explore_common16.py: `-    todo = sorted(jobs, key=lambda j: -dur(j))` / `+    todo = sorted(jobs, key=lambda j: (j.get("opt") != "muon", -dur(j)))`)

The arm cut (`LOCAL3_SLOW16_M`, `LOCAL3_SLOW_D8_A_SPLIT`, `MUON_HINGE16_REF`) is already in `meta.cut` at the skeleton 82e897b, before any record, and never changes.

**Item 4, README against the docstring:**
- S43, README: `"neither" (replaces: >= 36/40; does not: <= 30/40)`. This matches the docstring's 36/40 and 30/40. The README omits the second condition of "replaces", `(SLOW_HINGE only) - (LOCAL3_SLOW only) <= 1`. The omission does not matter here because 31/40 < 36.
- S44: the README gives the readings by name (`"carries to four streams"`, `(b) "breaks the eight-stream stall"`) and does not restate the numbers. The reported counts satisfy the docstring thresholds: 20/20 ≥ 16 with reference-only 0 ≤ 1, and 3/10 ≥ 3.
- S44, discrepancy in the reference count. The docstring background says `paired with X's HINGE4k16 (17/20 BOUND)`, but the README table gives `X HINGE4k16 15/20` under BOUND ROUTED. The two use different outcome labels (BOUND vs BOUND ROUTED). The reading's result is the same either way (reference-only 0).

---

## Batch 17: S48 (window_d8, W_SPLIT)

| item | S48 `explore_window_d8.py` |
|---|---|
| Readings first committed | 8ca286e 2026-10-07T14:23:39+00:00 "Batch 17 code: S48 window_d8, S49 window_fail, driver" |
| Other commits to the file | f91f08b 2026-10-07T14:37:39+00:00 (a CHECK fix after dry run 1 stopped at a CHECK); 2531480 2026-10-07T14:58:24+00:00 (report label only: `-        d = rp.compare(rs, pr, a["seeds"], arm, plab.split(" ")[0])` / `+        d = rp.compare(rs, pr, a["seeds"], arm, "B16_D8_M" if plab.startswith("batch 16") else plab.split(" ")[0])`). Both precede the real `started`. The child file also changed in 25e5691 (14:24:08Z) and f91f08b, both before the dry run 2 / real run |
| Labelled dry run | attempt 1 stopped at a CHECK (`batch17_dry_check_failed.log`, fixed in f91f08b); attempt 2 started `2026-10-07 14:37:43` (no tz), git f91f08b; first dry store commit f59e0d5 2026-10-07T14:49:29+00:00; labelled complete 2531480 2026-10-07T14:58:24+00:00 (8 records, all dry) |
| Real batch provenance | `started` `2026-10-07 14:58:34` (no tz; 10 s after 2531480), git `810d2d3+dirty` (810d2d3 = 14:58:31Z), 2.10GHz; skeleton 0bf8926 2026-10-07T15:10:20+00:00 |
| First commit with a non-dry record | **aa917ac 2026-10-07T15:36:08+00:00**, `W_LONG\|261` (record git 10385b1 = 15:24:24Z, secs_wall 1548) |
| Code SHA, dry vs real | 375b30d487df in both, so 2531480 did not change run code |
| Reading changes after first run | **none** (no commit touches the file after 14:58:24Z) |
| README verdict section | added in f6d06e6 2026-10-07T21:49:55+00:00 and never edited afterwards |

Reading line as first committed (8ca286e), quoted verbatim:
```
READING per arm (fixed before any run): "improves" if BOUND ROUTED >= 6/10; "does not" if <= 3/10; otherwise neither.
McNemar (exact, two-sided) against the paired arm is printed with it. An arm cut by the runtime rule has no reading.
```
Code constants: `IMPROVES_N, NOT_N = 6, 3` and `SB = tuple(range(260, 270))`. The W_SPLIT arm is defined as `S37's KEYMASS plateau trigger on: checks every 2400 from 4800 (2400 = reference), at most 3 splits per run (>= 4800 apart, S37's gap), 28800 updates.`

**Item 4:** the README says `Reading per arm: "improves" if >= 6/10, "does not" if <= 3/10.` This matches the docstring.

---

## Batch 18: S50 (window_recipe_ablation, W_RESET) and S52 (two_stream_keysplits, W2_KEYHINGE)

| item | S50 `explore_recipe_ablation.py` | S52 `explore_keysplits.py` |
|---|---|---|
| Readings first committed | 669c3e9 2026-10-08T01:08:28+00:00 "Batch 18 code: S50 ..., S51 ..., S52 ..., driver" (the file's only commit) | same 669c3e9 |
| Other commits to the file | none | 7ef4e15 2026-10-08T01:17:14+00:00 "S52 CHECK fixes before the dry run". It edits only the docstring's CHECKS/VALIDITY paragraph and the check code; no READINGS line. The commit message says "No arm or reading changes", and the diff confirms it |
| Labelled dry run | started `2026-10-08 01:17:17` (no tz), git 7ef4e15; first dry store commit a4429e5 2026-10-08T01:34:36+00:00; labelled complete 17fe9ab 2026-10-08T01:40:52+00:00 (5 records, all dry) | same dry run (2 records, all dry) |
| Real batch provenance | `started` `2026-10-08 01:41:02` (no tz), git `2afa02f+dirty` (2afa02f = 01:41:00Z), 2.10GHz; skeleton 4857653 2026-10-08T01:53:53+00:00 | same |
| First commit with a non-dry record | **91b8ba3 2026-10-08T02:07:49+00:00**, `W_SPLIT_M\|261` (record git 4857653+dirty = 01:53:53Z, secs_wall 834) | **91f1401 2026-10-08T03:24:19+00:00**, `W2_K4\|160` (record git ef87d80 = 03:23:17Z, secs_wall 76) |
| Code SHA, dry vs real | 300b8e0ba54a in both | bd2ad99c5016 in both |
| Reading changes after first run | **none** | **none** |
| README verdict section | added in ac2d3ba 2026-10-08T06:37:08+00:00 and never edited afterwards | same |

S50 reading lines as first committed (669c3e9), quoted verbatim:
```
READINGS (fixed before any run; BOUND ROUTED counts; exact McNemar two-sided against W_SPLIT printed; the rule is on counts):
  W_RESET         "the copy does the work" if <= 4/10; "the reset does the work" if >= 8/10; otherwise neither.
  W_SPLIT_NOSLOW  "the slow phase is unnecessary at eight streams" if >= 8/10; "needed" if <= 5/10; otherwise neither.
  W_SPLIT_W4      "width 3 suffices" if within 1 of 9/10 (8-10/10); otherwise printed, and which way (<= 7/10: width 4
                  does worse).
  W_SPLIT_M       "the recipe carries to Muon" if >= 7/10 and (W_SPLIT_M - reference) >= 4; otherwise not shown.
An arm cut by the runtime rule has no reading.
```
Arms: W_RESET, W_SPLIT_NOSLOW, W_SPLIT_W4, W_SPLIT_M (+ the W_M_REF reference), on `seeds 260-269` (`SB = tuple(range(260, 270))`). `meta.cut` = `["W_SPLIT_NOSLOW", "W_SPLIT_W4"]` from the skeleton 4857653 on, before any record.

S52 reading lines as first committed (669c3e9), quoted verbatim:
```
READINGS (fixed before any run; N = the arm's seeds, 40, or 20 if the runtime rule cuts W2_KEYHINGE to 160-179):
  per arm: "removes the key splits" if successes >= ceil(0.9 N) (36/40; 18/20) and KEY failures <= floor(N / 40) (1/40;
  0/20); "does not" if successes <= floor(31 N / 40) (31/40; 15/20); otherwise neither. Successes: W2_K4 BOUND ROUTED,
  W2_KEYHINGE DISCOVERED. KEY failures: unbound runs whose failure class is KEY (W2_K4: test_scale_axes.fail_class_k on the
  pooled eta^2; W2_KEYHINGE: test_router_layout.fail_class, S43's).
```
W2_KEYHINGE arm, verbatim: `k = 2, plus the hinge's KEY term only: 1.0 x relu(eta2_key - 0.2) on the read gate at key positions, weight 1 on updates 1-2400 and 0 after`. Seeds: `SEEDS = tuple(range(160, 200))`.

The pre-dry-run diff in 7ef4e15 (CHECKS paragraph only, quoted in part):
```
-the perfect gate (streams 0, 1 -> channels 0, 1) binds and routes on seeds 160-161 (validity); the paired records are all
-present.
+records are all present; VALIDITY: with k = 4 the perfect gate (streams 0, 1 -> channels 0, 1) binds and routes on seeds
+160-161. Run as this layout's own recorded validity arm, test_short_conv's ceiling_conv (the perfect gate + conv "layer",
```

**Item 4:**
- S50: the README gives W_SPLIT_M as `"the recipe carries to Muon" (>= 7/10 and +7 >= 4; ...)`, which matches. For W_RESET it gives only `"the copy does the work"`, with no numbers. The count 2/10 ≤ 4/10 is consistent with the docstring.
- S52: the README gives W2_K4 as `"neither" (36 >= 36 but KEY 2 > 1)`, consistent with ceil(0.9·40)=36 and floor(40/40)=1. W2_KEYHINGE has no numbers in the README: `"removes the key splits"` (36/40, KEY 1).
- S52 caveat, disclosed in the README itself: three W2_KEYHINGE failures (163, 182, 186) sit at eta² by key exactly 0.500. They are labelled STREAM-PARTIAL, and labelled KEY they would make the reading "neither". The docstring delegates the KEY class to `test_router_layout.fail_class` and does not state how a value exactly on 0.5 is handled. Its BACKGROUND notes that such runs exist. The README attributes the labelling to "main's classifier". I did not check whether that is the same function the docstring names.

---

## Batch 19: S64 (scale_width) and S65 (scale_depth)

| item | S64 `explore_scale_width.py` | S65 `explore_scale_depth.py` |
|---|---|---|
| Readings first committed | 617681d 2026-10-09T06:17:08+00:00 "Batch 19 work in progress: S64-S67 parents, ... (EXPLORATORY, not run)" (the file's only commit). `git log -S 'width-sensitive'` finds no earlier commit | same 617681d (`-S 'depth-sensitive'` finds no earlier commit) |
| Other commits to the file | none | 8cff21e 2026-10-09T08:20:12+00:00 (report-only change to the RES6 diagnostic table: `-    c19.arm_table("RES6", runs, SEEDS, ...)` / `+    seeds = me.ARMS["RES6"]["seeds"]` ...). It came 10 s before `started`, and the docstring is unchanged |
| Labelled dry run | attempt 1 ran out of memory in the projection (`batch19_dry_oom_failed.log`; fixed in 2b1dac4 2026-10-09T06:51:20+00:00, model/child files only); attempt 2 started `2026-10-09 06:51:30` (no tz), git 2b1dac4; first dry store commit ffd172f 2026-10-09T07:31:03+00:00; labelled 8cff21e 2026-10-09T08:20:12+00:00 (10 records, all dry) | same dry run (9 records, all dry, RES6 included) |
| Real batch provenance | `started` `2026-10-09 08:20:22` (no tz), git `c4aa26a+dirty` (c4aa26a = 08:20:20Z), 2.80GHz; skeleton 542eb33 2026-10-09T09:01:39+00:00 | same |
| First commit with a non-dry record | **7f1a71d 2026-10-09T09:20:22+00:00**, `ORC_D128\|270` and `ORC_D128\|271` (record git 542eb33+dirty, secs_wall 1122 each) | **e44e794 2026-10-09T14:06:26+00:00**, `ORC_L4\|280` (record git b3323b5 = 14:01:21Z, secs_wall 758) |
| Code SHA, dry vs real | 17e56410a1db in both | 17e56410a1db in both |
| Reading changes after first run | **none** | **none** |
| README verdict section | added in 9397c05 2026-10-09T16:42:03+00:00 while the batch was still running (S66/S67 continue to 6ecaa8d 2026-10-10T03:36:55Z); never edited afterwards | same |

S64 reading lines as first committed (617681d), quoted verbatim:
```
ORACLE RULE (fixed before any run; the user's S67 rule applied here too): a size's learned arm runs once its oracle has
BOUND on a seed; if neither oracle seed binds within 28800 the size is UNTESTED (not run, no part in the reading).
PAIRING: by seed (the same batches; the initial parameters differ with the size).
OUTCOME: BOUND ROUTED (test_stream_recipe.outcome: transition not None, one-to-one stream -> channel map and every stream's
held-out accuracy >= 0.9 at the end); transition time; failure classes.
READING (the user's, fixed before any run; on counts, with d = arm's BOUND ROUTED - N256's over the paired seeds = arm-only
minus N256-only discordant pairs; exact McNemar two-sided printed):
  "scale-free"       if every tested size has d >= -1;
  "width-sensitive"  if any tested size has d <= -4;
  otherwise neither. Transition medians (over BOUND ROUTED runs) per size are printed.
```
Arms: `N256 ... N512 ... N1024 ... D64 ... D128` + `ORC_<size>`, `ARMS (seeds 270-279; the oracle on 270-271)`. Code constants: `FREE_D, SENS_D = -1, -4`, `SEEDS = tuple(range(270, 280))`, `ORACLE_SEEDS = (270, 271)`.

S65 reading lines as first committed (617681d), quoted verbatim:
```
  RES6 (diagnostic, printed, no reading): run only if 6 layers fails, which is fixed here as: L6 is tested and its BOUND
  ROUTED count is below L3's by 2 or more (it misses the "scale-free" margin of 1). ...
READING (the user's "same readings", on counts, d = arm's BOUND ROUTED - L3's over the paired seeds; exact McNemar
printed): "scale-free" if every tested depth has d >= -1; "depth-sensitive" if any tested depth has d <= -4; otherwise
neither. Transition medians per depth printed.
```
Arms: `L2, L3 (the reference, the current model), L4, L6; ORC_L<n>`, `ARMS (seeds 280-289; the oracle on 280-281)`. Code constants: `FAIL_D = -2`, `SEEDS = tuple(range(280, 290))`, `DEPTHS = (2, 3, 4, 6)`.

**Item 4:**
- S64/S65: the README gives `**"scale-free"** (every d >= -1)` for both, which matches the docstring. It does not restate the `d <= -4` band.
- RES6: the README gives `its condition (L6 - L3 <= -2) did not hold (d = -1)`, which matches "below L3's by 2 or more" / `FAIL_D = -2`.
- Phase-2 tie-break (S66/S67 territory, outside S64/S65's readings): the README's own note says the L2/L4 tie was resolved by arm order, and "That last step is not in the pre-registered rule".

---

## Bottom line

For all seven screens, the reading/threshold lines were committed before the batch's labelled dry run and before the first non-dry record:

| screen | readings committed | first non-dry record committed | lead |
|---|---|---|---|
| S43 | 93bdfe6 13:48:18Z on 10-06 | 8e64157 19:45:33Z on 10-06 | ~6.0 h |
| S44 | 93bdfe6 13:48:18Z on 10-06 | 2e7966b 15:48:20Z on 10-06 | ~2.0 h |
| S48 | 8ca286e 14:23:39Z on 10-07 | aa917ac 15:36:08Z on 10-07 | ~1.2 h |
| S50 | 669c3e9 01:08:28Z on 10-08 | 91b8ba3 02:07:49Z on 10-08 | ~1.0 h |
| S52 | 669c3e9 01:08:28Z on 10-08 | 91f1401 03:24:19Z on 10-08 | ~2.3 h |
| S64 | 617681d 06:17:08Z on 10-09 | 7f1a71d 09:20:22Z on 10-09 | ~3.1 h |
| S65 | 617681d 06:17:08Z on 10-09 | e44e794 14:06:26Z on 10-09 | ~7.8 h |

No reading, threshold, arm list, seed range or outcome line in these files changed after the first non-dry run. The only post-run commits to batch code were two batch-16 harness commits (packing and job order).

The README verdict thresholds that are quoted match the docstrings. Some thresholds are not quoted in the README: S43's discordant-pair condition, S44's numbers, S50 W_RESET's numbers, and the `<= -4` bands.

Limits of what git can show:
- Exact run start times. Records have no timestamp, and `git` is HEAD at the end of the run.
- The timezone of `started`. It is very probably UTC, inferred from adjacent commits.
- The contents of the "+dirty" working trees at run time.
