# Screen review: Session H's round-7 screen (the prompt's S92; H's "S90"): the oracles at key pools 4-32 to 86,400 updates, then the recipe at P*

*Referee V, 22 October 2026, 05:30 UTC (`git fetch` at 05:25: claude/explore-H is still at `117ded72`). Reviewed on claude/explore-H:*
- *`23dccf24` (05:05:11), the screen: `explore_h_s90.py`, `explore_h_s90_child.py`, and a 5-line change to `explore_h_run.py`;*
- *`117ded72` (05:05:38), the CHECK log in progress (`explore_out/H/s90_phase1.log`, 7 lines).*

*The prompt is the main line's `docs/reading/parallel_sessions_prompts.md` at `5e51a294`, "## Session H — round 7 (S92 …)" (prompts:1316-1337).*

*Line references:*
- *"s:NN" is a line of `explore_h_s90.py`, "ch:NN" of `explore_h_s90_child.py`, "run:NN" of `explore_h_run.py`, all at `23dccf24`.*
- *"out §N" is a section of `screen_checks_s92_H_out.txt`, written by `screen_checks_s92_H.py`. That script uses stdlib only, reads git, trains nothing and imports no torch. It execs `explore_h_s90.py` from source with the harness stubbed, runs `cell_reading` / `phase1` / `report` / `payload` on synthetic records, and cuts the main line's `transition()` and `outcome` out by ast.*

*The earlier reviews this builds on: `screen_review_s89bc_H.md` (S89b/S89c) and `check_H_report11.md` (report 11). S89b's gate was ORACLE_DELTA 2/2, and it bound 1/2 (2913 at 24000; 2912 at 0.875). That screen came out UNTESTED. None of the ten fixes in the S89b/S89c review was taken, and neither were the S89 text fixes (§G).*

## Run status

| what | state at `117ded72` (05:05:38; unchanged at 05:25) |
|---|---|
| readings | committed at `23dccf24` (05:05:11), **2 s before** the driver started (05:05:13) |
| CHECKs | **in progress.** The log has the banner and the code SHA `287fdd3941d0` (55 modules, git `23dccf2`, 2.10 GHz), and nothing more. No repro result, no CHECK row, and no `H/s90` in `checks.json` (out §0). S89b's CHECKs took 38.9 min |
| records | **0.** There is no `s90_results.json` and no "saved" commit. The log name says the stage is `H_S90_STAGE=oracle` (phase 1 only), but the command line is not recorded |
| phase 2 | not queued. configure() queues it only when the stage is not `oracle`, phase 1 is complete, and P\* > 4 (s:167-190) |

**The readings come before every record.** Phase 1's first records could land within about an hour: the K = 4 oracles bound at 2400-8400 before, which means early stops near 5000-11000 updates. **The phase-1 text fixes below (1-4) should land before the first phase-1 record.** They are all in the parent, `explore_h_s90.py`, which is not among the 55 modules of the code SHA (out §0). They therefore change no CHECK and no run.

**Verdict: fit to run phase 1; fix before phase 2.**
- The budget reaches every run, and the seeds are free. The CHECK targets exist, are on this CPU where they are asserted, and are unchanged since they were first committed.
- Every phase-1 and phase-2 outcome gets exactly one name under the docstring's rules (0 disagreements over 625 phase-1 patterns, 6,561 cell combinations and 19,448 phase-2 compositions).
- Six defects need fixing:
  - the name collides with G's S90;
  - "by 43200" is read from an 86400 curve, and on a curve that dips after 43200 that is not what a 43200 run gives;
  - a failed pool-4 reference prints under the P\* = 4 name;
  - "the split is not needed" prints when nothing binds;
  - the caveat and the OC are not printed;
  - two CHECK gaps (the pool-4 path and DELTA_NOSPLIT).

---

## A. Name and seeds

**The name.** The prompt calls this screen **S92** (prompts:1316; V's line, prompts:1367: "H's S92 file before its learned arms"; STATUS.md:156: "H runs S92"). H calls it **S90** everywhere: s:3, s:41, s:278/280/292 (the readings key "S90"), ch:3, run:13 and both commit subjects.
- **It collides.** G's **S90**, the repeat hinge on lite (`explore_g_s90.py`, `b57c1867` 05:03:55; prompts:1280-1282; STATUS.md:154), started two minutes earlier and is running. F's own "S90-F" (`explore_f_s90_ctx.py`, `run_s90.sh`, `s90.log`) is a third "S90". The prompt names F's screen nothing, so F's label is not wrong, but it adds to the ambiguity (out §1).
- **There is no data collision.** The stores are namespaced (`H/s90` → `explore_out/H/s90_results.json`; G's and F's live under their own `explore_out/`). The collision is in citations: "S90 reads …" would be ambiguous in STATUS.md and in V's files.
- s:3 also says "**round 6**". This is round 7.
- s:92 SOURCE cites "S89c: the channel is the limit" bare. `check_H_report11.md` §D asked for that name to be quoted with "neither schedule change reaches 4/10; memory and budget not varied".

**Seeds** (out §2).
- H's block is 3020-3059. H uses 3020-3023 (phase 1) and 3024-3033 (phase 2).
- No JSON/JSONL record holds a 3020-3059 key, `seed` or `seeds` entry: none of the 369 blobs on the 13 heads and none of the 2,558 blobs in history.
- In text, the numbers appear only in the prompt file and in `explore_h_s90.py`.
- G uses 2700-2720 and F uses 3060-3072.
- The CHECK seeds (2900, 2913, 2902, 583, 540) are recorded seeds, re-run to tie to their records, and are not stored.
- **Free.** s:6 claims a scan of 3020-3039. It holds, and so does the whole block.

## B. Phase 1: readings and P\*

**The readings** (s:16-21, `cell_reading` s:138-146). I exec'd them on all 625 patterns of four seeds over five transition classes: ≤ 43200, exactly 43200, 44400-86400, exactly 86400, none.
- Every pattern gets one name: reliable 16, needs the longer budget 240, the oracle is the limit 369.
- **0 disagreements** with the prompt's text.
- A cell with fewer than 4 ok records reads None, which prints as "incomplete". P\* is computed only when all 8 cells are complete (s:160-162). An incomplete phase 1 prints "INCOMPLETE (phase 1), no reading" (out §3).

**Defect B1: "by 43200" is read off the 86400 run's transition.** s:142 tests `t <= SHORT` on the stored transition. The main line's transition (test_binding_onset.transition, equal at the main head and `9c5939e`, out §6) is "the first evaluation ≥ 0.95 that holds **to the end of the run**". On an 86400 run that end is 86400, or the early stop.
- A curve at ≥ 0.95 at 42000 and 43200 that dips to 0.94 at 44400 shows the gap:
  - on the 86400 run, H reads it **not** bound by 43200 (transition None; or 50400 if it re-binds there);
  - a 43200-budget run of the same trajectory has transition **42000**;
  - so does the same curve cut at 43200 (out §3, cases a / a').
- The error runs one way only. H's "by 43200" implies the cut curve binds, but not the converse. So H can name "needs the longer budget" (and then a phase-2 budget of 86400) or "the oracle is the limit" where the prompt's reading, "4/4 bind by 43200", applies.
- The case needs a curve near 0.95 at the end of the window. It is rare for sharp binders, but not for S89b's 2912-like slow climbers.
- **The fix is parent-only.** Every record holds its curve, and the run is the same through 43200 at either budget: `tscur.run_sc`'s schedule only sets the total, and the early stop is the same.

**P\*** (s:20-21, s:161-163).
- **The rule.** P\* is the largest pool at which both memories read "reliable" or "needs the longer budget". There is one P\* per outcome and no ties. The budget for each memory is 43200 if it reads reliable at P\*, else 86400.
- **Every combination.** I exec'd `phase1()` on all 3⁸ = 6,561 combinations of the eight cell readings. P\* was 32 in 2,916, 16 in 1,620, 8 in 900, 4 in 500 and None in 625. All four budget pairs occur (out §3).
- **Non-monotone outcomes take P\* literally.** In 4,460 of the 5,436 combinations with P\* > 4, a smaller pool fails on some memory. Example: delta fails at 16, but both memories pass at 32, so P\* = 32 (out §3). That is what the prompt says, so it is not a defect. But the reading prints only the per-pool list, and it does not say that the pools are not nested.
- **"Reliable at 16 but not at 8"** gives P\* = 16, or higher.

**Defect B2: P\* = None prints under the P\* = 4 name.** s:279-280 prints "no pool above 4 has a reliable oracle (P\* = None)" when even pool 4 fails (out §3: every cell failing prints exactly this).
- Pool 4 is "the main line's; the reference" (prompts:1322). A pool-4 failure means the reference oracle did not bind 4/4 on these seeds at 86400. At pool 4 the same gates bound at 2400 / 8400 (H's S87a ORACLE 2600/2601, 2.10 GHz) and at 1200 / 2400 (X's CEIL_DELTA). Such a failure is a validity problem, not a pool result.
- **It also happens with P\* > 4.** In 3,020 combinations pool 4 itself fails on some memory, yet phase 2 would run at P\* > 4.
- **P\* = 4** is handled as the prompt says: phase 2 does not run, and the screen reads "no pool above 4 …" (s:41, s:279-280; out §3). The word "reliable" in that name also covers "needs the longer budget". The name should say "binds 4/4 by 86400".

**Operating characteristics of the 4/4 rule** (s:22-23; out §5).
- **One cell.** q⁴ is 0.96 / 0.81 / 0.66 / 0.41 at q = 0.99 / 0.95 / 0.9 / 0.8, as H states. So a cell reads "the oracle is the limit" with probability 0.04 / 0.19 / 0.34 / 0.59. The Wilson interval of 3/4 is 0.30-0.95.
- **Both memories.** (q_h q_d)⁴ is 0.92 / 0.66 / 0.43 / 0.17. H does not state this.
- **P\* itself is noisy.** With q = 0.95 at every pool, P\* = 32 only 0.66 of the time, and it is 4 or None 0.04 of the time. With q = 0.9 everywhere, P\* is 4 or None 0.19 of the time. With 1.0 / 0.95 / 0.9 / 0.7 at 4 / 8 / 16 / 32, P\* is 16 (0.41), 8 (0.36) or 4 (0.18).
- **So "the oracle is the limit at p" names one failure in four.** And P\* is the largest pool that happened to pass, not the largest reliable one. Neither the joint OC nor the P\* distribution is printed.

## C. Phase 2: readings

**The rules.** s:33-41 and s:294-300 hold three per-memory names (≥ 7 carries, ≤ 3 does not, otherwise none) and one split name: "not needed" if (SPLIT only − NOSPLIT only) ≤ 1, else "the split is needed on delta (no reading)". The pair direction is correct: `hc.paired(x = NOSPLIT, y = SPLIT)` gives b = NOSPLIT only, c = SPLIT only and p_rev = P(SPLIT > NOSPLIT).

**Every composition.** I exec'd report() on all 19,448 compositions of 10 seeds into the 8 per-seed patterns of (BASE, DELTA_SPLIT, DELTA_NOSPLIT), at P\* = 16 with budgets HEBB 43200 and DELTA 86400.
- **0 disagreements** with the docstring.
- Per memory: carries 5,216, does not carry 5,216, no reading 9,016.
- Split: not needed 13,350, "needed (no reading)" 6,098.
- With 9 of 10 seeds stored, the reading is "INCOMPLETE (phase 2), no reading" (out §4).
- **The outcome is the main line's.** `b16.finish` sets `rec["outcome"] = tsr.outcome(rec)` on the tree at `9c5939e`, and `hc.br` reads "BOUND ROUTED". `test_stream_recipe.outcome` and what it calls have equal syntax trees at the main head `5e51a294` and at `9c5939e` (out §6).

**Defect C1: "the split is not needed" is not moot-guarded.** With all 30 runs unbound, report() prints "the recipe does not carry to P\*" for both memories, and also "**the split is not needed on delta at P\*** (0, 0; p 1)" (out §4).
- This happens in 1,304 of the compositions where both delta arms are ≤ 3/10.
- **By the OC:** at SPLIT = NOSPLIT = 0.1, the moot print comes 0.86 / 0.93 / 0.99 of the time (ρ = 0 / 0.5 / 1). At 0.3 it comes 0.38 / 0.47 / 0.65.
- **S89 fix 3** ("moot carries no name") is the precedent.

**Defect C2: "within 1 pair" is read one-sided, and the negative branch carries a name.**
- **One-sided.** H's rule also prints "not needed" when NOSPLIT is ahead by ≥ 2 pairs (6,098 compositions). At SPLIT 0.5 / NOSPLIT 0.9 it prints with probability 1.00, where a two-sided "within 1 pair" prints 0.05-0.09.
  - As an answer to "is the split needed", one-sided is defensible, and s:38-39 writes out the formula.
  - But the prompt's phrase reads two-sided, and the print does not separate "the split costs" from "the split does not matter".
- **The negative branch.** "The split is needed on delta (no reading)" puts a claim before "no reading". At 2 vs 0 the p value is 0.25.

**Phase-2 OC** (s:42-47; out §5).
- **The 7/10 and 3/10 lines** reproduce exactly: P(≥ 7/10) = 0.001 / 0.011 / 0.172 / 0.382 / 0.650 / 0.879 / 0.987 and P(≤ 3/10) = 0.879 / 0.650 / 0.172 / 0.055 / 0.011 / 0.001 / 0.000 at 0.2 … 0.9.
- **The split rows** reproduce under V's model: 0.9/0.9 0.88/0.95/1.00; 0.9/0.7 0.39/0.38/0.38; 0.9/0.5 0.09/0.07/0.05; 0.7/0.7 0.77/0.86/1.00; 0.7/0.4 0.24/0.20/0.15.
- **The moot share is missing.** It is the 0.86-0.99 at low rates above.

## D. CHECKs (s:57-64, s:193-244, ch:83-105)

| prompt CHECK | implemented | pin / target | status |
|---|---|---|---|
| pool 4 reproduces WIN3_SPLIT_D8\|540 through 4800 (printed if the CPU differs) | s:207, s:235-239: **S89's child**, arm MAINSPLIT (the main split op), K = 4, seed 540, 4800; printed, not asserted | read at `628eb7c` (s89.MAINREF), identical at the main head. Record on **2.80 GHz**; transition 20400; fired [4800, 12000, 19200] (out §7) | **PARTLY.** Printed as the prompt allows. But it runs S89's child, not S90's: neither ORACLE_HEBB nor BASE nor any K = 4 path of `explore_h_s90_child` is tied to a pool-4 record. PoolTask(4) = TASK48 bit for bit is a row of **S89's** checks(), which passed under S89 and which S90 does not rerun. `explore_h_s89_child.py` is unchanged since then (out §7), so the chain holds, but no S90 row closes it |
| S89's ORACLE\|2900 and ORACLE_DELTA\|2913 through 2400 | s:200-201, s:220-223: S90's child, K = 32, iters 2400; curve, statistics and gradient norms asserted | `s89.store_run` reads the **working-tree** store, not a pinned ref. ORACLE\|2900 was first committed `4324d2eb` (19 Oct 18:39:40) and ORACLE_DELTA\|2913 `d0095a8c` (21 Oct 08:21:30); both are identical at `23dccf24`, on 2.10 GHz (out §7) | **IMPLEMENTED, not yet run.** Pin by citing the two commits |
| the delta recipe's code = test_delta_recipe's by ast | ch:83-97: with_delta's inner `builder`, `copy_exact` against `copy_op` bodies, and BETA, at `a581027` | `with_delta`, `copy_op` and `BETA` are equal at the main head and `a581027` (out §6) | **IMPLEMENTED, not yet run.** The ast covers the converter and the copy only. The trigger and LOCAL3 + SLOW are tied by behaviour, by DS583 (next row) |
| (H's addition) BASE\|2902, K = 32, budget 86400, = S89's BASE\|2902 through 4800; last check 84000 | s:202, s:224-229 | first committed `7939dd4c`, identical at the head; fired [4800, 12000, 28800] | good: it asserts the trigger's 84000 |
| (H's addition) DELTA_SPLIT\|583, K = 4, = X's through 4800 | s:203, s:230-234 | `8ae660d` (s89b.DELTA_REC, pinned), 2.10 GHz; fired [4800, 12000] | good |

**Gaps.**
- **D1. DELTA_NOSPLIT has no CHECK.** Its path is ch:35 `nosplit` → ch:61 no trigger. S89b checked its own child's NOSPLIT against X's DELTA_SLOW\|583 (2.10 GHz, fired []); S90's child has its own `run()`.
- **D2. The pool-4 oracle (the reference cell) has no S90 row.** Both targets are on this CPU: H's S87a ORACLE\|2600 (transition 2400) and X's CEIL_DELTA\|580 (2400).

**The CHECK log** shows only the banner and the code SHA: no repro result and no row. Nothing has passed yet.

## E. The budget reaches every run

The S89H defect (ITERS never reaching the child) is absent here (out §8).
- **The path.** The driver's `_job` calls `mod.run_job(arm, seed)` (`explore_common16._job`). `run_job` sends `payload()` (s:126-130), and payload() exec'd gives:
  - phase-1 cells: `iters 86400` with the cell's K (HEBB_P4: K 4; DELTA_P32: K 32);
  - phase 2 (PLAN K 16, HEBB 43200, DELTA 86400): BASE `iters 43200, K 16`; DELTA_SPLIT and DELTA_NOSPLIT `iters 86400, K 16`.
- **The child.**
  - It passes `p["iters"]` to `b16.path` (ch:51, ch:66), and that becomes `sched = dict(tscur.REAL, total=iters)` (`explore_muon_scale_child.path`).
  - `total` defaults to `iters` (ch:42), so the trigger's last check is `total - 2400`, i.e. 40800 or 84000 (ch:63, ch:68).
  - `budget = total` is stored in every record (ch:44).
- **PLAN reaches the workers.** configure() fills it in the parent before `run_pool` forks its workers (run:96, fork context).
- **configure() avoids S89b's bug.** It reads its own `code_sha` and `cpu_model()` (s:174-176), not `c16.STATE`, which run:102-110 sets only after configure() (the S89b gate-lookup bug, check_H_report11 §C).
- **The hazard: no guard in report().** Nothing asserts that a stored record ran at its intended budget. A record's `budget` field, and the last curve step, or early stop, give that check for free (fix 9).
- **A second hazard: the code SHA.** configure() finds phase-1 records only under the current code SHA (s:179). Any change to one of the 55 modules before phase 2 makes phase 1 invisible:
  - in a non-oracle stage nothing would be queued;
  - in the oracle stage all 32 runs would be redone.
- **So every fix must stay in the parent.** All of mine do.

## F. Labels, printing, push

**Labels.**
- BASE and DELTA_SPLIT use `target="key"` (ch:54), the key positions 3j+1: the main line's labelled recipe, as the prompt asks.
- The oracles are the perfect gate (stream labels).
- `op_ok`, the pairs' labels and `map_before` are descriptive.
- The decision to fire reads the probe's accuracy.
- The caveat (s:48-52) says the keys stay positional. I checked its frequencies: 33/(98K), i.e. 0.0105 / 0.021 / 0.042 / 0.084.
- **No label-free claim is made, so none can leak.** As in S89b (fix 4, not taken), no line in the file states which statistics read labels.

**Printing.** The prompt asks for the caveat with every reading and the OC of the 7/10 and 3/10 lines. s:42 and s:48 say so too.
- The driver prints `res['readings']` only (run:152). CAVEAT and OC are in `res` (s:272) but in no reading string, and `explore_h_report.py` has no s90 section (out §9).
- **So neither is printed. This is S89b/S89c fix 2 again.**
- What is printed: `per_seed_log` (run:150-151; s:312-322) prints each cell's transitions and end accuracies, and every phase-2 firing with its probe value, the pair, `op_ok` and the map before it, which meets "every firing and the map before it". `configure()` prints the cell readings and P\* (s:189).

**Push after every saved run; no pull.**
- `push_after` (run:72-83) commits `explore_out/H` and pushes after every `save_run`, in the background under `flock`, with 4 retries. There is no `pull` (run:77-78; out §10). **Implemented.**
- But it is **opt-in**: `--push-each` is `store_true` (run:91). The S90 docstring's STAGES line (s:68) does not say to pass it, and the log does not record the command line.
- Push failures go to `/dev/null` (run:81), so a failed push leaves no trace in the log.
- There is no "saved" commit yet, which is consistent with 0 records.

## G. Earlier fixes still not taken (at `23dccf24`)

- README:636 "the four bound learned runs" (S89 fix 1).
- report_10.md:83 and notes:45 "seven of the nine" (S89 fix 3).
- `explore_h_s89.py:90` and report_10.md:60 cite the review as "3c772c3 / dcd549d", which is **G's** review (S89 fix 2).
- None of the S89c / S89b fixes 1-5. Report 11's printed S89c name went out without the §D quotation, and s:92 now repeats it.

---

## Fixes

All of these are in the parent `explore_h_s90.py` (outside code SHA `287fdd3941d0`), or in `explore_h_report.py` / the README. None reruns a CHECK or a run. **Land 1-4 before the first phase-1 record, and 5-10 before phase 2 is queued.** If any lands after a record exists, say so in a dated note with the commit order.

1. **Name.** s:3 → "Session H, round 7, screen S92 (the round-7 prompt's name; the module, store and log keep the key `s90`, which G's S90 and F's S90-F do not share): …". Rename the readings key "S90" → "S92" (s:278, s:280, s:292) and the run:13 help text "s90 (S90)" → "s90 (S92)". Use "S92" in report_12.md and the README. Keep `NAME = "H/s90"`: changing it would move the store and the `checks.json` key.
2. **"By 43200" as a 43200 run reads it** (s:17-18, s:138-146). In the docstring: "'bind by 43200' = the main line's transition() on the run's curve cut at 43200, which is what a 43200-budget run of the same trajectory gives; 'bind by 86400' = the run's own transition". In code, pass the records to `cell_reading` and use `tr43 = tbo.transition([c for c in r['curve'] if c[0] <= SHORT])` for the first test. Print any cell where the two definitions differ.
3. **The reference** (s:20-21, s:41, s:279-280). Add before P\*: "'the reference fails' if either memory at pool 4 reads 'the oracle is the limit': no reading about the pool; phase 2 does not run". Then change the P\* = 4 name to "no pool above 4 binds 4/4 by 86400 on both memories (P\* = 4)". Print P\* = None and P\* = 4 under these two different names.
4. **Phase-1 OC and monotonicity** (after s:23): "Both memories pass a pool with probability (q_h q_d)^4 (0.66 at 0.95 each, 0.43 at 0.9). P\* is the largest pool that passed, not the largest reliable one: with q = 0.95 at every pool, P\* = 32 with probability 0.66 and 4 or None 0.04. With 1.0 / 0.95 / 0.9 / 0.7 at 4 / 8 / 16 / 32, P\* is 16 (0.41), 8 (0.36) or 4 (0.18). 'The oracle is the limit at p' can mean 3 of 4 (Wilson 0.30-0.95); each cell is printed with its k/4. If a pool below P\* fails on either memory, the reading prints 'pools not nested: <list>'."
5. **Moot split** (s:38-40, s:298-300): "If DELTA_SPLIT <= 3/10 and DELTA_NOSPLIT <= 3/10: 'delta split: moot (neither delta arm binds), counts only'". In code, put that test before `c["c"] - c["b"] <= 1`.
6. **The split's other branches** (s:38-40, s:299). Either keep the one-sided rule and add "'the split costs on delta at P\*' if DELTA_NOSPLIT only − DELTA_SPLIT only >= 2 (printed with p)", or make the rule two-sided as the prompt reads. Rename "the split is needed on delta (no reading)" → "no reading applies (DELTA_SPLIT ahead by >= 2 pairs; p printed)". Add the moot share to the OC: "at SPLIT = NOSPLIT = 0.1 the moot branch takes 0.86-0.99; at 0.3, 0.38-0.65".
7. **Print what the docstring says is printed** (run:152, or the reading strings). Print `res['caveat']` and `res['oc']` after the readings, and add an s90 section to `explore_h_report.py` that prints both under every reading in report_12.md.
8. **Two CHECK rows, before phase 2** (s:200-203). (a) "DELTA_NOSPLIT\|583 (K 4, this child) = X's DELTA_SLOW\|583 (8ae660d) through 4800". (b) "ORACLE_HEBB\|2600 (K 4) = S87a's ORACLE\|2600 through 2400, and ORACLE_DELTA\|580 (K 4) = X's CEIL_DELTA\|580 through 2400". Both are on this CPU, and both should be asserted. These are parent edits to check(), so rerun the CHECKs once by hand (the SHA skip at run:119 would not). Optionally also print S90's BASE at K 4, seed 540, against WIN3_SPLIT_D8\|540 through 3600, before its first firing.
9. **The budget guard** (in report(), before the readings). Return "INCOMPLETE (budget mismatch)" unless every phase-1 record has `budget == 86400` and every phase-2 record has `budget == PLAN budget[mem]`. Each record must also end at its budget or at a 3-evaluation early stop.
10. **Text.**
    - s:6: "no 3020-3039" → "no 3020-3059 (the block) in any JSON record on any head or in history (Referee V, 22 October)".
    - s:57-64: cite the record pins: ORACLE\|2900 `4324d2eb`, ORACLE_DELTA\|2913 `d0095a8c`, BASE\|2902 `7939dd4c`, all read from the working-tree store.
    - s:68: "run with --push-each".
    - s:92: "S89c: the pre-fixed reading 'the channel is the limit' (neither schedule change reaches 4/10; memory and budget not varied)".
    - Add a "LABELS:" line as in S89b fix 4.
    - Apply §G's three S89 text fixes.

**Verdict: fit to run phase 1** (the CHECKs may finish and the 32 oracle runs may start). Fixes 1-4 should land before the first phase-1 record; they cost nothing. **Fix before phase 2:** 5-9, with the two CHECK rows (fix 8) passing before any learned arm is queued, then V's look at the phase-1 readings and P\*.

## What could not be checked

- **Whether the CHECKs will pass.** None has printed.
- **The command line**, i.e. the stage and `--push-each`.
- **Whether a case-(a) curve occurs.** That needs records.
- **The behaviour of PoolTask(8) and PoolTask(16).** The child CHECK prints their structure and frequencies; I checked the frequencies only by formula.

Housekeeping: only this file, `screen_checks_s92_H.py` and `screen_checks_s92_H_out.txt` were written. Nothing was committed or checked out, and no session code was run as a program.

**Corrections (22 October, after phase 1).**
- **The fix ranges disagree.** Line 181 says fixes "5-10 before phase 2"; the verdict says 5-9. The verdict's 5-9 is right; fix 10 is text.
- **The timing estimate was wrong.** The review estimated the first phase-1 records "within about an hour". They landed at 05:34:49, 6 minutes after this review was committed (05:28:50).
- **The CHECK status was stale when committed.** The CHECKs had passed at 05:28:28, 22 s before the commit, so "CHECKs in progress" was already out of date.

H took none of fixes 1-9. Phase 2 started at 16:13:13, chained from phase 1. On phase 1's records, fixes 2 and 3 would have changed nothing (`check_H_s92_phase1.md`).
