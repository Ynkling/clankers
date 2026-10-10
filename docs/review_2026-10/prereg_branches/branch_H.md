<!-- Appendix to docs/review_2026-10/preregistration.md. Produced for the referee by a delegated read-only audit of branch H (git log/show/diff and JSON via git show only); the referee spot-checked its load-bearing claims against git (see preregistration.md). -->

# Pre-registration audit: Session H (origin/claude/explore-H), screens S58, S58b, S60, S61

Method: read-only `git log --format='%h %cI %aI %s'`, `git log -p`, `git show`, `git diff`, and `python3` on store JSON
read through `git show`. **Timezones:** every commit time below is the committer date `%cI`, and all are `+00:00` (UTC).
For every commit cited, the author date equals the committer date. The stores' `meta.provenance.started` strings
(e.g. `2026-10-08 04:20:21`) **carry no zone**. They fit UTC: S58's is 3 s after commit 705000d, and the logs and reports
mark the same kind of harness time as "UTC". No run record holds a start or time field. I searched every record's keys for
start, time, date and when, and found none.
Prompt source: `origin/claude/bdh-growth-hebbian-inference-w90069:docs/reading/parallel_sessions_prompts.md`. Its Session
H block was added in ad8a92e (2026-10-08T03:50:57+00:00) and is unchanged in the Session H part since then.

**Limits of git evidence (applies to every screen).** Commit times are set by the committer and nothing signs them. git
cannot show when a remote branch was pushed, or whether it was force-pushed or rewritten. The order "readings committed,
then runs" therefore rests on (a) commit times and parentage, (b) the ancestry of each store's first commit, and (c) the
`git` field in the run records and logs, all of which the same session wrote.

## Summary table

| Screen | Readings first committed (docstring) | Invocation `started` (store meta, no tz) | First store commit with >= 1 run record | Store complete | Changes to reading lines after first run |
|---|---|---|---|---|---|
| S58 collapse | bbfea60 2026-10-08T04:14:03Z | 2026-10-08 04:20:21 (git 705000d) | c06f911 2026-10-08T04:29:26Z (4 SPLIT runs, git 69d01dc) | 9200cac 2026-10-08T10:42:30Z (242/242) | none |
| S58b eight | 69d01dc 2026-10-08T04:21:59Z (rule + readings) | 2026-10-08 10:43:16 (git d9baf09) | 4bf95bf 2026-10-08T11:30:40Z (4 SPLIT runs) | 8cd03b6 2026-10-08T16:31:26Z (32/32) | none |
| S60 stability | 705000d 2026-10-08T04:20:18Z | 2026-10-08 10:43:16 (git d9baf09) | b6ec6b4 2026-10-08T16:16:23Z (10 REF runs; earliest record git 93ac87d+dirty, 93ac87d = 16:01:20Z) | cc3a767 2026-10-08T16:40:29Z (44/44) | none |
| S61 copy2x2 | f630ef2 2026-10-08T20:11:01Z | 2026-10-08 20:32:11 (git 3e4212d) | 145b69c 2026-10-08T21:26:08Z (3 SPLIT runs) | 493f95f 2026-10-09T05:42:16Z (52/52) | none |

**How "none" was checked** (the same for every screen):
1. `git log -p origin/claude/explore-H -- <file>` lists only: collapse bbfea60, 705000d; eight 69d01dc, d9baf09;
   stability 705000d; copy2x2 f630ef2. Every one is earlier than that screen's first run record.
2. `git diff <first-run commit> origin/claude/explore-H -- <file>` is empty (0 lines), and `git log <first-run>..tip -- <file>`
   is empty, for all four parent files and all four `_child.py` files.
3. The reading functions in the code are also frozen. S58's `report()` and S58b's `select()` and `configure()` live in
   these files. Report writer explore_h_report.py was added at 9aec3db (04:40:43Z, after S58 began) and edited at 7cbcc12
   (2026-10-09T16:53:10Z). It only formats the output: a grep finds none of the reading names or thresholds in it.
4. Recount from the final stores: every BOUND ROUTED count and every discordant pair quoted in the reports matches the
   records.

---

## S58 explore_h_collapse.py

**1. Readings as first committed, bbfea60 (2026-10-08T04:14:03Z), verbatim:**
```
READINGS (fixed now; BOUND ROUTED on the 40 paired seeds; "X only" = seeds where X is BOUND ROUTED and the other arm is
not; every comparison printed with the exact one-sided McNemar p in both directions and both arms' Wilson intervals and
bands):
  R1 "noise replaces the split"  if (SPLIT only - GUMBEL_W only) <= 1  AND  (GUMBEL_W only >= 4 and NONE only = 0 in
                                 GUMBEL_W vs NONE). Otherwise "does not apply".
  R2 "the split is the reset"    if (SPLIT only - RESET only) <= 1. Otherwise "does not apply". Qualifier fixed now: if SPLIT
                                 and NONE differ on <= 1 discordant pair in SPLIT's favour (SPLIT only - NONE only <= 1), the
                                 split has no measurable effect here and R2 is printed as "uninformative (the split itself
                                 does nothing at this configuration)" next to its rule result.
  R3 "the balance loss helps"    if SWITCH only >= 4 and NONE only = 0 (SWITCH vs NONE). Otherwise "does not apply".
  GUMBEL_RW has no reading; its counts and comparisons with NONE and SPLIT are printed (descriptive).
VALIDITY (fixed now): the screen is VALID only if ORACLE binds on both seeds (420, 421) within 28800; otherwise every
reading below is UNTESTED (not a negative).
```
The same commit also states this expectation: "EXPECTATION STATED BEFORE ANY RUN: since NONE's recipe bound 20/20 at this
configuration in batch 16, the "> NONE by >= 4 vs 0" readings can only apply if NONE fails on >= 4 of these 40 seeds".

**Comparison with the prompt.** The prompt reads: "Readings fixed now: "noise replaces the split" if GUMBEL_W ≥ SPLIT − 1
discordant pairs and GUMBEL_W > NONE by ≥ 4 vs 0; "the split is the reset" if RESET ≥ SPLIT − 1 pair; "the balance loss
helps" if SWITCH > NONE by ≥ 4 vs 0."
- R1, R2 and R3 match the prompt in substance. The docstring writes them as counts of discordant pairs.
- The docstring adds R2's "uninformative" qualifier (not in the prompt), an oracle validity rule (as the preamble
  requires) and "Otherwise does not apply" outcomes.
- Arms and seeds (420-459) match the prompt. The docstring adds an ORACLE arm on seeds 420-421.

**2. First run.**
- The first CHECK pass ran at git bbfea60 and failed (log `explore_out/H/s58_check_failed.log`, committed in 705000d).
- 705000d (04:20:18Z) changed only the CHECKS paragraph. Its message says "Arms and readings unchanged", and the diff
  confirms this. The removed lines are:
  ```
  -LOCAL3_SLOW16_A|240 (curve and every statistic): NONE through 2400; GUMBEL_W and GUMBEL_RW at noise scale 0 through 2400;
  -SWITCH at alpha 0 through 2400; SPLIT and RESET with the trigger's threshold at 0 (never fires) through 6000.
  ```
  They were replaced by the GUMBEL_W "TWO_NODE control" CHECK. This was before the first run and is not a reading line.
- Store meta: `provenance.started` = "2026-10-08 04:20:21", `git` = "705000d".
- In s58.log the CHECKs pass in 4.2 min and then "242 runs queued". report_1 gives "CHECKs: all passed (2026-10-08
  04:25:12 UTC, git 69d01dc)". The first run finishes 4.1 min after the queue starts, at about 04:29.
- The first store commit with records is **c06f911, 2026-10-08T04:29:26Z**: 4 records (SPLIT seeds 420-423), each with
  `git` = "69d01dc". The store is complete (242/242) at 9200cac, 10:42:30Z.

**3. Changes after the first run:** none. No commit touches explore_h_collapse.py after 705000d (04:20:18Z).

**4. Reports.**
- report_1.md (9200cac) gives R1-R3 with the docstring's wording, thresholds and R2 qualifier; results match the
  records: R1 SPLIT only 3, GUMBEL_W only 1, GUMBEL_W vs NONE 2 vs 3 → does not apply; R2 SPLIT vs RESET 0 vs 0 → holds
  by rule, qualifier uninformative (SPLIT vs NONE 1 vs 0); R3 SWITCH vs NONE 1 vs 3 → does not apply.
- The explore_out/README.md row (cc3a767) agrees.
- One addition beyond the readings: report_1's "Unexpected" bullet "The split's one gain is the reset's". It rests on one
  seed (453), and the report itself says "One seed is not evidence". It is not a pre-registered reading.

---

## S58b explore_h_eight.py (selection rule for "the two best arms")

**1. Rule and readings as first committed, 69d01dc (2026-10-08T04:21:59Z), verbatim:**
```
ARM SELECTION (fixed now, before S58's results exist; applied once S58 is complete and stored in this screen's store):
the two arms of S58 with the most BOUND ROUTED among {RESET, GUMBEL_W, GUMBEL_RW, SWITCH} (the devices; SPLIT is the
comparator, NONE the no-device baseline); ties broken by the earlier median transition time, then by fewer MERGED runs,
then by that listed order. If S58 is UNTESTED (its oracle failed), S58b does not run.
...
VALIDITY (fixed now): VALID only if ORACLE binds both seeds within 43200; otherwise UNTESTED.
READING (fixed now): for each selected arm X, "X carries to eight streams" if (SPLIT only - X only) <= 1 on the 10 paired
seeds (BOUND ROUTED). The screen's reading: "carries to eight streams" if it holds for the better of the two arms at eight
streams (more BOUND ROUTED; if tied, if it holds for either), else "does not carry". Exact one-sided McNemar both ways,
Wilson intervals and bands printed.
```
The same commit puts the rule in code: `CANDIDATES = ("RESET", "GUMBEL_W", "GUMBEL_RW", "SWITCH")`, and `select()` sorts
by key `(-br, trans_med, merged, listed order)` and keeps `[:2]`.

**Timing of the rule versus S58's results.**
- Rule committed: 04:21:59Z.
- S58's invocation began at 04:20:21 (at 705000d), but its 4.2-min CHECK pass was still running.
- The first S58 screen run started at about 04:25. The first S58 result was committed at 04:29:26Z (c06f911).

So the rule precedes every S58 result, by about 7 min to the first result and about 6 h 20 min to S58's completion
(10:42:30Z). It does not precede the start of the S58 process.

The chosen arms come after S58's results, as the rule intends:
- d9baf09 (10:43:12Z, after S58 completed) changed only harness code in `configure()`. The two added lines are:
  ```
  +        c16.STATE["shas"].setdefault(s58.NAME, ec.load_store(s58.NAME)["meta"]["code_shas"][-1])
  +        c16.STATE["batch_cpu"] = c16.STATE["batch_cpu"] or ec.cpu_model()
  ```
  The rule, the candidate list and the readings did not change.
- The selection `['RESET','GUMBEL_W']` was first stored in `eight_results.json` meta at **6f67a6b
  (2026-10-08T10:45:38Z)**. Its `selection_basis` is RESET br 39, GUMBEL_W 37, SWITCH 36, GUMBEL_RW 5, which is S58's final
  counts. The rule applied correctly: the top two by BOUND ROUTED, with no tie-break needed.

**Comparison with the prompt.** The prompt reads: "the two best arms of S58 vs SPLIT at eight streams. Reading: "carries to
eight streams" if the best arm ≥ SPLIT − 1 pair."
- The prompt does not define "best". The docstring fixes it as the most BOUND ROUTED, among the devices only.
- The docstring leaves out NONE. NONE (38/40) outranked GUMBEL_W (37) and SWITCH (36) in S58.
- The prompt's "the best arm" is read as the better arm *at eight streams* (if tied, either arm). The prompt leaves this
  open: it could also mean S58's best arm.
- The docstring adds the validity rule and the ORACLE arm on seeds 460-461.
- Seeds 460-469 match the prompt.

**2. First run.**
- `provenance.started` = "2026-10-08 10:43:16", git d9baf09. This was one joint invocation with S60. Its CHECKs passed in
  6.7 min.
- The first commit with records is **4bf95bf, 2026-10-08T11:30:40Z** (SPLIT 460-463).
- The store is complete (32/32) at 8cd03b6, 16:31:26Z.

**3. Changes after the first run:** none. No commit touches explore_h_eight.py after d9baf09, which is before the first
run.

**4. Reports.** report_2.md (cc3a767) gives the readings with the docstring's rule:
"RESET does not carry (SPLIT only 8 - RESET only 0 [<= 1])", "GUMBEL_W does not carry (SPLIT only 10 - GUMBEL_W
only 0 [<= 1])", "does not carry (better arm: RESET)". These match the docstring and the records.

report_2's "Unexpected" bullet "At eight streams the row copy is the active part of the split, not the reset" is an
inference from SPLIT vs RESET. No S58b reading covers it, and copy without reset was not tested until S61.

---

## S60 explore_h_stability.py

**1. Readings as first committed, 705000d (2026-10-08T04:20:18Z), verbatim:**
```
VALIDITY (fixed now): VALID only if ORACLE_CONV binds both seeds within 24000; otherwise everything below is UNTESTED.
READINGS (descriptive; fixed now; per arm vs REF on the 10 paired seeds, with exact one-sided McNemar both ways):
  D1 DISCOVERED counts and transition times (median, range); a device "changes discovery" only if its discordant pairs
     with REF are >= 3 vs 0 either way (printed as "more" / "fewer"), else "no change at n = 10".
  D2 saturation: at each evaluation, "saturated" = the gate's max probability > 0.99 at > 90% of the probe's positions.
     Per arm: the fraction of evaluations saturated BEFORE the transition (evaluations at steps < transition, bound runs)
     and AT/AFTER it, and over the whole run for unbound runs; "saturates before binding" counts the bound runs with any
     saturated evaluation before their transition. A device "delays saturation" if its pooled before-transition saturated
     fraction is lower than REF's by >= 0.2 (absolute).
  D3 EMA_EVAL and the BOUND asymmetry: per arm, the runs that count as BOUND on a single final evaluation (transition =
     the last evaluation and that evaluation is the budget's, 24000) and the runs whose held-out accuracy fell below 0.95
     after first reaching it ("flips", evaluations after the first >= 0.95). "The asymmetry shrinks under EMA" if
     EMA_EVAL's single-final-evaluation bindings plus flips are fewer than REF's (both counted on the 10 seeds); with
     REF's typical 0-1 such runs this is expected to be uninformative at n = 10, which is then stated.
```

**Comparison with the prompt.** The prompt defines S60 as purely descriptive: "Descriptive: DISCOVERED counts, time to
transition, how often the gate softmax saturates (max probability > 0.99 at > 90% of positions) before vs after the
transition."
- The docstring keeps the saturation definition and adds its own thresholds: D1 ">= 3 vs 0" and D2 ">= 0.2".
- D3 turns the prompt's question about the EMA asymmetry into a rule.
- TEMP_FLOOR becomes a *learned* τ with a 0.5 floor ("the user's choice of 8 October 2026"). The prompt has
  "temperature max(τ, 0.5)" with no learning stated.
- The docstring adds the validity arms ORACLE_CONV and ORACLE_PLAIN.
- Seeds 470-479 match the prompt.

**2. First run.**
- `provenance.started` = "2026-10-08 10:43:16" (git d9baf09), the same invocation as S58b. In s58b_s60.log the first S60
  run starts at 311.9 min, about 16:03 UTC.
- The first commit with records is **b6ec6b4, 2026-10-08T16:16:23Z** (REF 470-479). Its earliest records carry git
  "93ac87d+dirty"; 93ac87d is dated 16:01:20Z.
- The store is complete (44/44) at cc3a767, 16:40:29Z.

The S60 child was changed in 69d01dc (04:21:59Z, "S60 group CHECK corrected"). The change was to a CHECK row only, and it
came before the run.

**3. Changes after the first run:** none. explore_h_stability.py has a single commit, 705000d.

**4. Reports.** report_2.md gives D1, D2 and D3 in the docstring's terms:
"D1 no change at n = 10", "D2 does not delay saturation", "D3 does not shrink ... uninformative: REF has none at
n = 10". These match.

The README row's phrase "saturation is never reached before binding" is a descriptive summary. It is consistent with the
reported 0/… saturated before-transition evaluations.

---

## S61 explore_h_copy2x2.py

**1. Readings as first committed, f630ef2 (2026-10-08T20:11:01Z), verbatim:**
```
VALIDITY (fixed now): VALID only if ORACLE binds both seeds within 43200; otherwise every reading is UNTESTED.
READINGS (fixed now; BOUND ROUTED on the 10 paired seeds; "X only" = X bound routed and the other arm not; exact one-sided
McNemar both ways, Wilson intervals and bands printed):
  R1 "the copy alone suffices"  if (SPLIT only - COPY only) <= 1 in COPY vs SPLIT.
  R2 "the noise matters"        if COPY only >= 4 and COPY_NONOISE only = 0 in COPY vs COPY_NONOISE.
  R3 "the reset adds nothing"   if |SPLIT only - COPY only| <= 1 (SPLIT vs COPY within one discordant pair either way).
  Printed alongside (no reading): RESET vs SPLIT (S58b's comparison on new seeds), every arm vs NONE.
```

**Comparison with the prompt.**
- **S61 is not in the main-line prompt file.** `git log --all -S'copy2x2'` and `-S'the copy alone'` find these strings only
  on explore-H (f630ef2 onward) and on review-V (ca9882e). The main-line file has no Session H follow-up block. So no
  external pre-specification can be checked from git; the docstring is the only pre-registration.
- **The audit brief's reading "the noise does not matter" is not a pre-registered reading.** The docstring's R2 is the
  opposite claim, "the noise matters". Its only other outcome is "does not apply"; there is no "does not matter" outcome.
  The phrase "the noise does not matter (0 vs 0)" first appears in explore_out/README.md at **7e9197f
  (2026-10-09T17:49:16Z)**, after the run.
- R1 and R3 test almost the same thing. R3 (|SPLIT only − COPY only| ≤ 1) implies R1, so R3 adds only the bound in the
  other direction.

**2. First run.**
- The first invocation at f630ef2+dirty passed S61's CHECKs ("2026-10-08 20:23:29 UTC"). S63's CHECK failed in that pass
  (s61_s63_check_failed.log). 3e4212d (20:32:07Z) fixed explore_h_hardness.py only.
- The relaunch has `provenance.started` = "2026-10-08 20:32:11", git 3e4212d. In s61_s63.log: "CHECKs of H/copy2x2
  skipped: passed at 2026-10-08 20:23:29 UTC with this code SHA".
- The first commit with records is **145b69c, 2026-10-08T21:26:08Z** (SPLIT 480-482). The earlier store commit 05d8cb6
  (20:41:06Z) holds 0 runs.
- The store is complete (52/52) at 493f95f, 2026-10-09T05:42:16Z.

**3. Changes after the first run:** none. explore_h_copy2x2.py has a single commit, f630ef2. Its blob c656548 is
identical at the tip.

**4. Reports.**
- report_3.md (7e9197f) matches the docstring:
  - "R1: the copy alone suffices (SPLIT only 0 - COPY only 0 = 0 [<= 1])"
  - "R2: does not apply (COPY vs COPY_NONOISE 0 vs 0 [>= 4 vs 0])"
  - "R3: the reset adds nothing (|SPLIT only 0 - COPY only 0| = 0 [<= 1])"
  - It also says "R2 "the noise matters" does not [apply]".

  The counts match the records: SPLIT, COPY and COPY_NONOISE 9/10 each with 0 discordant pairs; RESET 4; NONE 3.
- **Mismatches:**
  - The README row (7e9197f, line 566) gives R2 as "the noise does not matter (0 vs 0)".
  - report_3's "Unexpected" heading reads "The split is the copy, and the copy needs no noise".

  Both turn R2's "does not apply" (no evidence for the noise at n = 10) into a positive null claim that was not
  pre-registered. Neither is supported as a pre-registered reading.

---

## Bottom line

For S58, S58b, S60 and S61, git shows the readings committed before the first run record (S58: 15 min; S58b's
selection rule: 7 min before S58's first result; S60: about 12 h; S61: 75 min).

No reading line (thresholds, readings, arms, seeds or outcomes) changes after any first run. The only pre-run edits were S58's CHECK paragraph (705000d) and S58b's
harness lines (d9baf09).

Where restatements depart from the pre-registration:
1. S61 has no main-line prompt. "The noise does not matter" appears only after the run (README, report_3 narrative); the
   pre-registered R2 was "the noise matters" → does not apply.
2. S58b's docstring fixes "best" as devices only, so NONE is excluded, and reads "the best arm" as the best at eight
   streams. The prompt leaves both open.
3. S60's docstring adds thresholds (D1, D2, D3) and a learned τ that the prompt does not state.

Git cannot establish push times or rule out rewritten history.
