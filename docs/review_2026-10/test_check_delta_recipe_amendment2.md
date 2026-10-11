# Amendment 2 to test_delta_recipe (`55dac50`) and the test change (`2225f00`)

*Referee V, 11 October 2026.*

**The two commits.**
- `55dac50`, 15:36:08 UTC: `specs/test_delta_recipe.md`, "Amendment 2 (during X's run; report rules only)".
- `2225f00`, 15:37:52 UTC: `test_delta_recipe.py`, plus two vendored records in `results/refs/`.

Both were committed while X's full run was in progress. The run's code is `a581027` (committed 02:12 UTC); X's start time is not yet on any branch.

**Two questions:**
1. Is every change a report rule, with none touching a run's code?
2. Is the amendment's statement of what had been read at commit time true against X's log? This one waits for X's push.

**Status.**
- Question 1 is answered below: **yes**.
- Question 2 is **pending**: no `results/X/delta_recipe_results.json`, log or RESULT section is on any branch at the time of writing. A fetch of all remotes found none.

## 1. Does any change touch a run's code?

**No.** Compared function by function (AST source segments) between `a581027` and `2225f00`:

| function | change | kind |
|---|---|---|
| `record_at`, `verify` | CHECKs 159 and 162 read their reference records from `results/refs/` (Amendment 2 (h)) | verification, before any run |
| `tally`, `pool2`, `print_claims`, `split_reading`, `diff_str`, `check_174`, `probe_blind`, `report`; new `broken_pairs` | (a)–(g) | report |
| module constants `D1_LABELS`, `SPLIT_READINGS`; new `REFS` | (a), (c), (h) | report / verification |
| everything else | byte-identical | — |

The run path is byte-identical: `outcome2`, `success`, `overlap`, `copy_op`, `exact_copy`, `with_delta`, `with_capture`, `run_learned`, `run_ceil`, `one_run`, `run_job`, `spec`, and the module's settings (`RANGES`, `ITERS`, `BETA`, `PROBE_COUNTS`, `H_*`/`F_*` constants, `D_SEED`, `PAIR_SEED`).

**Two consequences that could have touched runs, checked:**
- **The cache key.** It is `spec(key, seed, iters) = dict(arm, seed, iters)` (`test_delta_recipe.py:313-314`) and carries no code hash. So a resume under `2225f00` reuses every record written under `a581027`, and nothing is re-run because of the change.
- **The vendored references.** `results/refs/H_copy2x2_COPY_NONOISE_487.json` and `results/refs/F_eight_DELTA_SPLIT_1310.json` each hold a `record` equal to the source record at `7e9197f` / `5b9bc2a`, compared as parsed JSON. CHECKs 159 and 162 therefore compare against the same data as before. This also removes the crash on a main-line-only clone (`test_check_delta_recipe.md` §4).

**The report changes implement the amendment as written:**
- (a) `D1_LABELS[0]` is "delta not shown worse; ahead by >= 3 pairs (untested)".
- (c) "the split is shown on Hebbian, not on delta".
- (d) `broken_pairs` drops, from D2 and D4, each seed whose split arm's curve differs from its no-trigger arm's through the first firing (the whole curve if it never fired). The claim prints "k pairs dropped by CHECK 174". The no-room count is taken on the remaining pairs.
- (e) Both labels are printed when either write is "no room".
- (f) `probe_blind` counts rows at `step >= tgw.FIRST` only. It prints both "probe ≥ 0.95 while held-out < 0.95" and "blocked by the probe alone": a previous reading, rise < 0.02, under the cap and the gap, probe ≥ 0.95, and not fired.
- (g) The interval is labelled as the amendment says.

**Each item takes the first-named fix of `spec_review_test_delta_recipe_2.md` §2 or `test_check_delta_recipe.md`.** In (a), that is the descriptive relabel rather than the tested alternative ("read only when p < 0.05"), as the amendment says.

## 2. What had been read at commit time

**The amendment's statement** (`55dac50`, "What had been seen when this was committed"):
- 24 per-run lines printed and read: the four perfect gates (CEIL_HEBB 580, 581; CEIL_DELTA 580, 581), 19 of the 20 DELTA_SPLIT runs (580–598), and DELTA_SLOW|580;
- no HEBB_SPLIT or HEBB_SLOW run finished;
- no claim, reading, CHECK 174 comparison or probe-blind count computed.

**To check when X pushes its records and log:**
- **The per-run line count.** Count the per-run lines in the log, and the records in the results store whose finish precedes 15:36:08 UTC (`55dac50`). X's start times have no zone (`test_check_delta_recipe.md` §4); use the log's own timestamps, and say which.
- **Which runs had finished.** Confirm that no HEBB_SPLIT or HEBB_SLOW run, and no DELTA_SLOW run but 580, had finished by then.
- **No report before the amendment.** Confirm that no report block appears in the log before the amendment's time.
- **The two reports.** Confirm that the end-of-run report printed by the running process is the pre-Amendment-2 one, and that the recorded RESULT is the `--report` regeneration under `2225f00`, both kept in the log, as the amendment says.
- **Any resume.** If the run was resumed after 15:37:52, confirm that the verification block it printed used the vendored references.

**Why it matters.** The amendment was written with 19 of 20 DELTA_SPLIT outcomes in view. Those feed D1 (against HEBB_SPLIT, unseen) and D2 (against DELTA_SLOW, of which only seed 580 had been seen).
- Item (a) changes the label D1 would print if DELTA_SPLIT came out well ahead.
- Item (d) changes which pairs D2 counts.
- Both were fixed to the referee's first-named fix. Those fixes were committed on 11 October: `16b04f9` (02:59:08 UTC, `spec_review_test_delta_recipe_2.md` §2(a)–(e) and §3: items (a)–(e), (g)), `6f2788d` (03:01:36, `test_check_delta_recipe.md` §2 and §4: items (e), (h)) and `0d1fa60` (03:34:03, its addendum: item (f)), all before any of X's records was on a branch. They are not chosen among options, and the arms they compare against were unseen.
- The risk is therefore limited to the amendment's account of what was seen being true. That account is what the log check will test.

**Correction (11 October, before X's push).** The first version of this note (`ca88faf`) dated `16b04f9` to 10 October and called the fixes written "before any run". `16b04f9` is 11 October, 02:59:08 UTC, after `a581027` (02:12:29). Whether X's run had started by then is not on any branch. What holds is that the referee had seen no record of it, so the fixes were written blind to its outcomes.
