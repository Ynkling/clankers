# Screen review: Session H's S89 fixes, S89b (delta memory, 32-key pool) and S89c (trigger schedule, 32-key pool)

*Referee V, 20 October 2026, 22:20 UTC (`git fetch` at 22:20 shows nothing newer than `cf67855d`). Reviewed on claude/explore-H after `13fe610`:*
- *`2a98deff` (21:38:45), S89's fixes;*
- *`c81398ca` (21:41:43), S89c;*
- *`bccb7cbd` (21:46:05), S89b;*
- *`555d6816` / `cf67855d` (21:46:32 / 22:00:07), S89c's CHECK log.*

*Line references:*
- *"s89:NN", "c:NN", "cc:NN", "b:NN", "bc:NN" are lines of `explore_h_s89.py`, `explore_h_s89c.py`, `explore_h_s89c_child.py`, `explore_h_s89b.py` and `explore_h_s89b_child.py` at the head `cf67855d`. Each file is byte-identical to the commit that introduced it.*
- *"r10:NN" is a line of `explore_out/H/report_10.md` at the head.*
- *"out §N" is a section of `screen_checks_s89bc_H_out.txt`, written by `screen_checks_s89bc_H.py`. That script uses stdlib and numpy only, reads git, trains nothing and imports no torch. It execs H's `report()` functions from source with the store loaders stubbed, and cuts `rule_s89c` / `rule_s89b` out of `explore_h_oc.py` by ast.*

## Run status

| screen | readings commit | CHECKs | records | learned arms |
|---|---|---|---|---|
| S89c | `c81398ca` 21:41:43 | started 21:42:42 at `c81398c`. All 5 rows passed at 21:59:18 (`checks.json`: git `555d681+dirty`, code SHA `4dda56358910`) | **0** stored at the head. 30 runs were queued on 4 workers, and BASE 2924-2927 started at about 22:00 | running (all three arms are learned arms; there is no gate stage) |
| S89b | `bccb7cbd` 21:46:05 | not run (no `H/s89b` in `checks.json`; no `s89b` log or store on any head) | 0 | **not started**. They are queued only after ORACLE_DELTA binds 2/2 *and* S89c is complete (b:166-170), which is hours away. ORACLE_DELTA's stage could start at any time |

**The readings precede every record in both screens.**
- S89c's readings were committed 59 s before its CHECK began, and no record exists yet.
- S89b's readings come before any CHECK or run.
- S89b's commit (21:46:05) is 3 s earlier than my `check_H_report10.md` (`adecd3db`, 21:46:08). Its nine requirements were therefore not available to H; §B checks them anyway.

**Verdicts.**
- **S89 fixes (`2a98deff`):** 7 of 8 resolved, 1 partly. No record and no child code changed. The "four runs" miscount is gone from report 10 but **still in README:636**. The fix commit has two new text errors: it cites the wrong review commits, and it introduces one new wrong count. **Fix the text; nothing else.**
- **S89c:** **sound to run; fix the text before the first record.** The readings give one name each and the CHECKs tie to S89's records bit for bit. But "the channel is the limit" names a cause the screen does not test, and it also prints when the BASE rerun itself binds. The 12000 schedule was chosen after S89's curves, and the docstring does not say so. The caveat and the OC are never printed. All the fixes are parent-only, outside the code SHA, and can land now.
- **S89b:** **fix first (before ORACLE_DELTA's CHECKs).** The readings are correct and the seeds are free. But it has no feasibility prior for its gate, at a rate the pool cut from 0.87 to 0.1 on Hebbian memory. Its expected print is "moot" by its own OC. The schedule it may inherit from S89c has no CHECK on delta, and the caveat, OC, fix-5 lines and schedule are never printed.

---

## A. S89: the eight fixes (`2a98deff`)

**What changed.** Only these files changed: `explore_h_s89.py`, `explore_h_report.py`, report_10.md, report_10_notes.md and the README (out §0).
- **Records.** `s89_results.json`, `s89.log`, `s89_oracle.log` and `s89_runs.md` have the same blobs at `13fe610`, at `2a98deff` and at the head. The `H/s89` entry in `checks.json` is unchanged.
- **Code SHA.** It covers 51 modules, 46 of them files on the branch. None changed between `13fe610` and the head. Neither `explore_h_s89` nor `explore_h_report` is among them (out §1).
- **So no run, CHECK record or child code changed.** The commit's claim is true.

**report() at `2a98deff`, exec'd.**
- **On S89's 42 stored records** it prints the same reading: "the labelled recipe itself fails here (BASE 1/10); no other reading".
  - Fix-5 lines: BASE 30 / 1 / 27, ALLMASS 30 / 1 / 28, REPEATMASS 30 / 1 / 28 (firings / one-to-one before / op_ok).
  - Fix-2 note: "repeat set = 95.7% ...; same pair as ALLMASS at 28 of 30 firings".
  - All of these match `check_H_report10.md`.
- **On synthetic outcomes**: 6,767 compositions (one for each of the 1,772 predicate patterns, plus 5,000 at random), each with and without firings, for 13,534 calls.
  - 1,244 were moot, and **0 moot trigger readings carry a name**.
  - Every non-moot trigger reading carries exactly one name (out §1).

| fix | status | where (head) |
|---|---|---|
| 1 what the screen tests; caveat | **RESOLVED** | s89:3-5 restated; s89:60-64 and `CAVEAT` s89:133. Printed in the report section (`explore_h_report.py:487`, r10:25). The console line `explore_h_run.py:150` prints only the readings, which is enough for S89 because report 10 carries the caveat |
| 2 REPEATMASS | **RESOLVED** | s89:24-28; `repeat_note` s89:302-306, s89:330-331. Deviation: the "as ALLMASS" note is a *prefix* to the reading, not a replacement for it (out §1 example). Acceptable. `explore_h_report.py:490` hard-codes "8 of 10 seeds" (true, from `check_H_report10`) |
| 3 moot carries no name | **RESOLVED** | s89:51-53, s89:329. Verified over every predicate pattern (above) |
| 4 operating characteristics; p with every reading | **RESOLVED** | s89:65-70, `OC` s89:137; p values in s89:318, s89:326-327, with the directions correct (`hc.paired`: p = x > y, p_rev = y > x) |
| 5 one-to-one before, op_ok | **RESOLVED** | s89:73-75, s89:299; r10:27 |
| 6 pin MAINREF | **RESOLVED** | s89:132 `628eb7c`, which holds `WIN3_SPLIT_D8|540` (out §1) |
| 7 citation 50dc8e4 | **RESOLVED** | s89:11-12 ("the same at F's head 52e2ea7"). That was true at 21:38. F's head is now `53578896`, which adds only `_GateStub`, so there is still no grouped-layout class (out §1) |
| 8 MAINSPLIT rationale | **RESOLVED** | s89:87-88 |
| (`check_H_report10`) "four runs" for 7 runs on 4 seeds | **PARTLY** | Gone from report_10.md and the notes: the paragraph was rewritten, not corrected. **README:636 still reads "the four bound learned runs at 36000-40800".** |

**New errors in `2a98deff`.**
1. **Wrong review commits.** The REVISION note (s89:90-91), r10:60 and report_10_notes.md:22 cite the review as "claude/review-V 3c772c3 / dcd549d". Those are **G's** review: `3c772c3` is "interim: S89 G screen review in progress" and `dcd549d` is `screen_review_s89_G.md` (out §0). H's review is `ee5217b` (18:40:23 UTC, 19 Oct), with its correction in `adecd3db`.
2. **One new descriptive count is wrong.** r10:83 and notes:45 say "In seven of the nine the last evaluation (43200) is the curve's maximum".
   - Recomputed: **4 of 9** (2904, 2906, 2907, 2909).
   - On 2902, 2903, 2908 and 2911 the maximum is at 42000, 0.001-0.009 above the last.
   - On 2905 it is at 37200, equal to the last to three decimals (out §1).
   - The other new figures reproduce. The perfect gate is at 0.21-0.41 at 1200 and 0.71-0.77 at 4800, and first reaches ≥ 0.90 at 8400 and 9600. BASE first reaches 0.3 at 12000-22800 and 0.5 at 14400-26400. The nine failures end at 0.595-0.852 with 2-6 streams below 0.9, 7 of them (2 share) and 2 (3 share). 2910 reaches 0.8 at 21600. ALLMASS's failures end at 0.454-0.841, and its 2910 ends at 0.454 with 8/8 streams below 0.9.

**Fixes (S89 text only; no code-SHA file).**
1. README:636: "the four bound learned runs at 36000-40800" → "the seven bound learned runs (on four seeds: BASE 2910; ALLMASS and REPEATMASS 2904, 2908, 2909) at 36000-40800".
2. s89:90-91, r10:60, notes:22: "claude/review-V 3c772c3 / dcd549d" → "claude/review-V ee5217b (pushed 18:40:23 UTC on 19 October, within the 58 s in which the learned arms were launched; correction adecd3db)".
3. r10:83-84, notes:45-46: "In seven of the nine the last evaluation (43200) is the curve's maximum: they were still rising when the budget ended" → "In four of the nine the last evaluation (43200) is the curve's maximum; in the other five the maximum (at 42000, or 37200 for 2905) is at most 0.009 above it. They were level or still rising slowly when the budget ended."

**Verdict A: fix the text (3 edits). The S89 result and its records stand.**

---

## B. S89b (`bccb7cbd`): delta memory on the 32-key pool

**Design.**
- Arms: ORACLE_DELTA on 2912-2913 (the validity gate), then DELTA_SPLIT (key positions), DELTA_NOSPLIT and DELTA_ALLMASS, paired on 2914-2923.
- Gate: DELTA_SPLIT ≥ 7/10. Readings are as S89's with fix 3, for one trigger arm.
- The trigger schedule is S89's, unless S89c reads "the schedule was the limit" (b:20-24).

### B1. Readings (b:36-47, b:247-291; out §3)

report() was exec'd on all 19,448 compositions of the 10 seeds into the 8 per-seed patterns of the three learned arms, under four ORACLE_DELTA states.

- **ORACLE 2/2.** fails 14,232; not needed 1,668; no reading 2,514; needs key positions 751; survives 283.
  - **0 disagreements** with H's `explore_h_oc.rule_s89b` (the rule its OC tables use) and with my reading of the docstring.
  - Moot prints "DELTA_ALLMASS: moot (...); <counts>" with no name.
  - "survives" and "needs key positions" are disjoint (c − b ≤ 1 against c ≥ 4, b = 0).
- **ORACLE 1/2 or 0/2.** UNTESTED in every case.
- **One ORACLE run stored.** INCOMPLETE in every case.
- **The outcome** is the main line's: `b16.finish` → `tsr.outcome` on the tree served at `9c5939e` (bc:102). That is the same path as S89's, and `check_H_report10` §A confirmed its syntax tree against the main head.
- **`s89c_schedule`** was exec'd on synthetic S89c comparisons. It takes the arm that meets the rule with more "V only" pairs, BASE_LATE at a tie, and {} when neither arm meets the rule, as b:22 says (out §3).

### B2. Seeds (out §4)

- 2912-2959 appear in no JSON / JSONL record on any of the 13 heads (348 blobs) or anywhere in history (2,356 blobs). The one exception is S89c's own meta (2924-2933).
- Text mentions on the heads are H's files, my review files, and numbers in other files that are not seeds (none of them is a record key, "seed" field or "seeds" list).
- S89's 2900-2911 are disjoint.
- The CHECK seed 2959 is outside the screen seeds.
- **Free.** b:6-7 cites my 19 October scan; it still holds.

### B3. CHECKs (b:67-76, b:182-225)

**The pool-4 ties are pinned.**
- They read X's records at `8ae660d`, "Record test_delta_recipe on X". Those records are on 2.10 GHz, the same CPU as here.
- `DELTA_SPLIT|583` fired at [4800, 12000], so the CHECK's expected `fired == [4800]` through 7200 is consistent.
- `DELTA_SLOW|583` and `CEIL_DELTA|580` (transition 2400) exist (out §3).

**At pool 32 nothing can be tied to a record.** The threshold-0 equality on seed 2959 is what can be asked.

**Gap: the conditional schedule.** If S89c reads "the schedule was the limit", the DELTA arms run with `first = 12000` or `cap = 6` (bc:90). No S89b CHECK exercises that, because every check job runs the default schedule (b:184-188). S89c's mechanics CHECK covers it on the Hebbian path only.

**Not yet run.** The CHECKs must pass under the code SHA of the learned arms.

### B4. Operating characteristics (out §5)

My own enumeration reproduces every number in b:55-61. They are all at rates taken from pool 4:

| true DELTA_SPLIT rate | 0.3 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|
| P(pass the gate, ≥ 7/10) | 0.011 | 0.172 | 0.382 | 0.650 | 0.879 | 0.987 |

**Feasibility.** On Hebbian memory the same pool moved the labelled recipe from 26/30 (S86 + S87a, pool 4) to **1/10** (Wilson 0.02-0.40). The perfect gate bound about 2.5× later.
- No pool-32 delta rate is known. The docstring's OC assumes the pool-4 rates (SPLIT = NOSPLIT = 0.9).
- At SPLIT = NOSPLIT = ALLMASS = 0.7 the print is "fails" 0.35 and "not needed" 0.43-0.65.
- At 0.5 it is "fails" 0.83.

**What the screen can answer.**
- **At the docstring's own expected rates, the trigger reading is moot with probability 0.87-0.99.** That is stated (b:39-43).
- A trigger reading is reachable only if DELTA_SPLIT ≥ 7/10 *and* DELTA_NOSPLIT falls ≥ 2 pairs below it, e.g. SPLIT 0.9, NOSPLIT 0.5. There, "survives" is 0.44-0.61 at ALLMASS 0.9.
- So DELTA_ALLMASS's ten runs most likely buy no reading. The screen's realistic question is "does delta memory need the split at pool 32 (and does it bind at all)?".

### B5. Labels (bc:41-42, bc:89-99)

- DELTA_SPLIT reads key positions 3j+1. These are the labelled ceiling, as BASE was in S89 (`make_trigger_s89`, target "key").
- DELTA_ALLMASS reads the read gate only (target "all"; S89 review §6).
- ORACLE_DELTA uses the perfect gate (stream labels).
- `op_ok`, `key_ok`, `all_ok` and `map_before` use routing_k's stream labels and are descriptive.
- The schedule choice reads S89c's BOUND-ROUTED counts. That is an adaptive design step, not a label reaching the trigger.
- **No leak.** But the docstring does not say this in one place (requirement 8).

### B6. Against `check_H_report10.md`'s nine requirements

| # | requirement | status |
|---|---|---|
| 1 | name the change | **MET**: delta memory (b:3-5, b:14-16, CHANGE b:102) |
| 2 | feasibility prior for the gate | **NOT MET**: there is no expected pool-32 rate, and the OC is only at the pool-4 rates (above) |
| 3 | one name each; moot without a name | **MET** (B1) |
| 4 | the main line's outcome | **MET** (B1) |
| 5 | seeds named and free; CHECK seed outside them | **MET** (B2) |
| 6 | CHECKs tie to a recorded run; pinned ref; old-setting CHECK if the schedule changes | **PARTLY**: pool-4 ties to `8ae660d` are pinned. There is no CHECK for the S89c-chosen schedule on delta (B3) |
| 7 | OC printed with the readings | **PARTLY**: OC is in the docstring and `res["oc"]`, but nothing prints it. `explore_h_run.py:150` prints `res['readings']` only, and `explore_h_report.py` has no s89b section. The same holds for the CAVEAT ("printed with every reading", b:48), the fix-5 `trigger_lines` (b:64, b:261) and the schedule ("printed with the readings", b:23-24, b:289) |
| 8 | which statistics read labels | **PARTLY**: true by the code (B5), but not stated |
| 9 | REPEATMASS dropped or redefined | **MET**: dropped |

### B7. Smaller points

- b:12 "F's head 52e2ea7": F's head is now `53578896`, still with no grouped-layout class.
- ORACLE_DELTA's form is "test_delta_recipe's CEIL_DELTA form" in b:26, but "test_split_copy.b_recipe(True)'s form" in bc:13. The CHECK against `CEIL_DELTA|580` decides which; make the two texts agree.
- `summarize` collects `schedules` per arm (b:236), but report() does not require one schedule across the learned runs.

### Fixes (S89b; all in the parent `explore_h_s89b.py` or `explore_h_run.py`, outside the child's code SHA)

1. **Feasibility** (requirement 2). Add after b:61:
   > "EXPECTED RATE (stated before any run): unknown at pool 32. At pool 4 delta bound without the split (DELTA_SLOW 19/20, 8ae660d); on Hebbian memory the 32-key pool moved the labelled recipe from 26/30 to 1/10 (S89) and the perfect gate's binding about 2.5x later. P(DELTA_SPLIT >= 7/10) = 0.99 / 0.88 / 0.65 / 0.38 / 0.17 / 0.01 at a true 0.9 / 0.8 / 0.7 / 0.6 / 0.5 / 0.3. At the pool-4 rates the expected print is 'the split is not needed here' with DELTA_ALLMASS moot (0.87-0.99); a trigger reading needs DELTA_SPLIT >= 7/10 and DELTA_NOSPLIT at least 2 pairs below it. If the gate fails, the screen says only that delta memory at pool 32 does not bind by 43200 under this recipe."

   Optional, and the cheapest guard: read ORACLE_DELTA's transitions against S89's Hebbian 10800/16800 before queuing the learned arms, and say in the docstring what transition would make 43200 too short.
2. **Print what the docstring says is printed.** In `explore_h_run.py:150` (or in report()'s reading strings), print `res.get('caveat')`, `res.get('oc')`, `res.get('trigger_lines')` and `res.get('schedule')` after the readings, for S89b and S89c alike.
3. **The inherited schedule** (b:20-24). Add to `check()` a job for each non-empty `SCHEDULE`: "pool 32, seed 2959, threshold 0, DELTA_SPLIT and DELTA_ALLMASS with the chosen schedule equal DELTA_NOSPLIT bit for bit through 4800 (or through `first` + 2400 for BASE_LATE)". In report(), return "INCOMPLETE (mixed schedules)" unless `A['DELTA_SPLIT']['schedules']` and `A['DELTA_ALLMASS']['schedules']` each hold exactly one value, and the two are equal.
4. **Labels** (requirement 8). Add after b:32:
   > "LABELS: DELTA_SPLIT's target reads the key positions 3j+1 (the generator's layout; the labelled ceiling); DELTA_ALLMASS's reads the read gate only; ORACLE_DELTA is the perfect gate (stream labels). key_ok / all_ok / op_ok / map_before use routing_k's stream labels and are descriptive; the decision to fire reads the probe's accuracy (the training target). The schedule is chosen from S89c's BOUND-ROUTED counts before any learned run."
5. b:12: "at F's head 52e2ea7" → "at 52e2ea7 and at 53578896 (F's head on 20 October)". Make b:26 and bc:13 name the same ORACLE form.

**Verdict B: fix first.** Fixes 1-5 are text, report path or CHECK additions. None of them touches `explore_h_s89b_child.py`, so they cost no rerun of anything. Fix 3 adds a CHECK row, so the CHECKs must run (or rerun) after it. Land them before S89b's CHECKs run; ORACLE_DELTA may then start.

---

## C. S89c (`c81398ca`): the trigger's schedule on the 32-key pool

**Design.**
- Arms: BASE (S89's rerun), BASE_LATE (first eligible check 12000) and BASE_CAP6 (cap 6), paired on 2924-2933.
- Validity is borrowed from S89's perfect gate (2/2).
- Readings (c:21-27): "the schedule was the limit" (either new arm beats BASE 4 vs 0); "the channel is the limit" (neither new arm reaches 4/10); otherwise "no reading applies".

### C1. Readings (out §2)

report() was exec'd on all 19,448 compositions.
- **Names:** schedule 2,273; channel 1,467; no reading 15,708.
- **0 disagreements** with `rule_s89c` and with my reading.
- Every outcome gets exactly one name. The first two names are disjoint because 4 vs 0 needs ≥ 4/10.
- An incomplete store gives "INCOMPLETE, no reading". There is no moot reading.
- The outcome is the main line's (cc:50 `b16.finish`).

**Defect 1: "the channel is the limit" does not look at the BASE rerun.**
- It prints whenever both new arms are below 4/10, whatever BASE does.
- It prints with BASE ≥ 4/10 in 1,031 of the 1,467 compositions, and with BASE ≥ 7/10 in 436 of them. There the screen's own control has bound, and the pool-32 failure that S89c sets out to explain has not replicated.
- P(BASE rerun ≥ 4/10) is 0.013 at a true 0.1, 0.12 at 0.2 and 0.35 at 0.3. At a true 0.3, a non-replication is a real possibility.

**Defect 2: the name claims a cause the screen does not vary.**
- "The channel" is the Hebbian memory, which S89b then replaces.
- But report 10 (r10:120-122) and its notes name *the budget and the cap* as the candidates. Report 10's own new paragraph says the BASE failures were "slow climbs cut off by the budget" (r10:75).
- The budget is not varied here. "Neither schedule change helps" leaves the budget and the memory both standing.
- The OC shows the hazard. At BASE 0.1 and both new arms at 0.3, a tripling of the rate, "the channel is the limit" prints 0.42-0.65 (c:39-40 says so, which is honest, but the name stays).

### C2. Operating characteristics (out §5)

My enumeration reproduces every number in c:31-41:

| true BASE / new arms | schedule | channel | none (ρ = 0 / 0.5 / 1) |
|---|---|---|---|
| 0.1 / 0.1 | 0.01 / 0.00 / 0.00 | 0.97 / 0.98 / 0.99 | 0.02 / 0.02 / 0.01 |
| 0.1 / 0.3 | 0.27 / 0.23 / 0.12 | 0.42 / 0.50 / 0.65 | 0.31 / 0.27 / 0.23 |
| 0.1 / 0.5 | 0.64 / 0.69 / 0.62 | 0.03 / 0.07 / 0.17 | 0.33 / 0.23 / 0.21 |
| 0.2 / 0.5 | 0.38 / 0.45 / 0.35 | 0.03 / 0.07 / 0.17 | 0.59 / 0.47 / 0.48 |
| 0.3 / 0.5 | 0.21 / 0.24 / 0.12 | 0.03 / 0.07 / 0.17 | 0.76 / 0.68 / 0.71 |
| 0.8 / 0.3 | 0 | 0.42 / 0.50 / 0.65 | 0.58 / 0.50 / 0.35 |

- **There is no BASE gate.** A gate would be wrong here, because BASE is expected to fail; but then the premise needs its own reading (fix 1).
- **At S89's Wilson range for BASE (0.02-0.40)**, "the schedule was the limit" needs a new-arm rate of about 0.5 or more to be likely.
- **The last row:** a BASE that replicates at 0.8 with both schedule changes harmful (0.3) prints "the channel is the limit" up to 0.65 of the time.

### C3. Post-hoc design

**Both constants were chosen after S89's records.**
- **12000.** `2a98deff` (21:38:45) added r10:79, "held-out accuracy first reaches 0.3 at 12000-22800". S89c came 3 minutes later with 12000 as the new first check. 12000 is exactly the earliest update at which any S89 BASE run reached 0.3 (per seed: 12000, 13200, 13200, 14400, 15600, 16800, 18000, 18000, 21600, 22800).
- **Firings.** 15 of S89 BASE's 30 firings were before 12000, and 17 were at probe < 0.2.
- **Cap 6.** It answers S89's 29-31 cap blocks per arm and S79's CAP6.

**What 12000 means for BASE_LATE.**
- Through 12000 BASE_LATE equals NOSPLIT (the CHECK asserts this).
- On S89's NOSPLIT runs, held-out accuracy rose by < 0.02 from 9600 to 12000 on only 2 of 10 seeds. So BASE_LATE will usually not fire at 12000 but at a later flat check.
- This is the intended effect, and it is measured on held-out accuracy, not on the 64-sequence probe.

**This is a legitimate post-hoc design.** The seeds are fresh (2924-2933), there is a concurrent BASE rerun, and the readings were fixed before any record. The screen tests *these two constants*; it is not a re-reading of S89.

**What the docstring says.** c:3-6 and SOURCE (c:70) say the change is motivated by S89. **They do not say that 12000 was read off S89's curves.** That belongs in the docstring, because a "schedule was the limit" at 12000 will invite the inference "the trigger must wait until the probe reads 0.3", and 12000 was tuned to that.

### C4. CHECKs (cc:57-85, c:100-125; `s89c.log`)

All 5 rows passed:
- the mechanics on S48's harness at pool 4;
- BASE_LATE's rows equal BASE's through 4800;
- BASE|2902 and BASE_CAP6|2902 equal S89's BASE|2902 bit for bit through 12000, both firing at [4800, 12000] like the record;
- BASE_LATE|2902 equals S89's NOSPLIT|2902 through 12000.

**The records.** S89's BASE|2902 was first committed at `7939dd4` (19 Oct 19:39) and NOSPLIT|2902 at `a8816ce`. Both are identical at the head, under code SHA `6e0a80a55be3` on the same CPU (out §2).
- **So the S89c code is tied bit for bit to two recorded runs.** BASE_CAP6's behaviour beyond the third firing is tied only by the pool-4 mechanics, which is all that can be asked.

**Minor points.**
- The CHECK reads the records through `s89.store_run` (the working-tree store), not a pinned ref. That is harmless, because the records are unchanged since they were first committed.
- The mechanics CHECK uses screen seed 2924 (cc:65), at pool 4 and not stored. That is harmless; next time use 2959.
- The CHECK ran from `555d681+dirty`. The parent files (`explore_h_s89c.py`, `explore_h_s89.py`, `explore_h_oc.py`) are not in the code SHA, but their committed text is unchanged at the head.

**Validity is borrowed from S89's perfect gate.** That is sound: none of the 46 branch modules of S89's code SHA changed (§A), and `explore_h_s89_child` is among them. `valid` is hard-coded True (c:166).

### C5. Labels and seeds

- **Labels.** All three arms use the key-position target (labelled, as S89's BASE), and the schedule constants read nothing. The decision to fire reads the probe's accuracy, and op_ok and the map are descriptive. **No label-free claim is made, so none can leak.**
- **Seeds.** 2924-2933 are free (B2). S89's 2902 is used only as a CHECK seed, against its stored records.

### C6. Printing

As in S89b, the CAVEAT ("printed with every reading", c:28) and the OC are placed in `res` (c:147) but never printed. `explore_h_run.py:150` prints the readings only, and there is no s89c report section.

### Fixes (S89c; parent `explore_h_s89c.py` and `explore_h_run.py` only, outside code SHA `4dda56358910`; land before the first S89c record)

1. **A premise reading first** (c:21, c:153-165). Insert before "the schedule was the limit":
   > "'S89's failure does not replicate here' BASE >= 4/10. Then each new arm is printed with its counts and p, with no other name."

   In code, before `beat = ...`: `if A['BASE']['br'] >= 4: s = "S89's failure does not replicate here"`. This removes the 1,031 compositions where "the channel is the limit" now prints with BASE ≥ 4. It costs at most 0.013 at a true 0.1 and 0.35 at 0.3.
2. **Rename the second reading** (c:23, c:162, `explore_h_oc.py` rule_s89c):
   > "'neither schedule change reaches 4/10' (BASE < 4/10 and both new arms < 4/10). The budget (43200; S89's BASE runs were level or still rising at the end) and the memory both remain; this does not say the channel is the limit."
3. **Say the design is post hoc.** Add after c:6:
   > "POST-HOC CHOICE (stated before any run): both constants were chosen after S89's records. 12000 is the earliest update at which any S89 BASE run reached held-out accuracy 0.3 (12000-22800 across seeds; report 10); 15 of S89 BASE's 30 firings came before 12000, 17 at probe < 0.2. Cap 6 answers S89's 29-31 cap blocks per arm (S79's CAP6). Fresh seeds and a concurrent BASE rerun make this a test of these two constants; a 'schedule was the limit' at 12000 does not by itself say the trigger should wait for a probe reading of 0.3."
4. **Print the caveat and the OC with the reading.** The same edit as S89b's fix 2, in `explore_h_run.py:150`.
5. (Record only.) Add to c:44-49: "the S89 records compared are BASE|2902 (first committed 7939dd4) and NOSPLIT|2902 (a8816ce), code SHA 6e0a80a55be3, read from the working-tree store; the mechanics CHECK's seed 2924 is an S89c seed, at pool 4 and not stored."

**Verdict C: sound to run (the runs already started may continue). Fix the readings' text before the first record:**
- Fixes 1-2 change only the names that a rule over the stored counts prints. They are parent-only, so no run or CHECK changes. If they land after a record exists, say so, with the commit order.
- Fixes 3-5 are text and printing.

---

## What could not be checked

- **What was uncommitted when the S89c CHECKs ran** (`555d681+dirty`). Only the parent files fall outside the code SHA.
- **Whether S89b's CHECKs will pass**, i.e. whether the delta conversion and S89's trigger reproduce X's `DELTA_SPLIT|583` and `CEIL_DELTA|580`. They have not run.
- **Whether the container has already started ORACLE_DELTA.** Nothing on any head shows it, and the 4 workers are occupied by S89c.
- **The probe-based rise at 12000 for BASE_LATE.** I used the held-out curve of S89's NOSPLIT as a proxy.

Housekeeping: only this file, `screen_checks_s89bc_H.py` and `screen_checks_s89bc_H_out.txt` were written. Nothing was committed or checked out, and no session code was run as a program.
