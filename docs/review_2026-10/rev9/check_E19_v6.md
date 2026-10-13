# Batch 19 (Session E) against the sixth draft: check_E19_v6

*Referee V, sixth pass.*

**What was checked.** The draft is `multichannel_hebbian_report_v9.tex` at `42c4557` (tex6:NNN), 568 lines, committed 2026-10-12 23:14:04 UTC. It is compared with the fifth version `c5e4508` (tex5:NNN). The changes replace lines in place, so tex5:N is tex6:N.

**What changed for batch 19.**
- **tex6:368.** "whose one-pair rule misses a drop from $0.8$ to $0.6$ more often than not" now reads "… about two times in five". This is v5's suggested fix, verbatim.
- **tex6:41.** "every session's queue is empty" is removed. The line now reads "E's batch~20 (S67's unbound runs continued to 86{,}400 updates) and the sessions' fifth round are running" (`b5fd2f2`, 22:53:57).
- **tex6:349.** The account of the referee's passes now lists "a one-pair power stated the wrong way round" among the slips each pass found.
- No batch 19 number changed.

**Records.** `origin/claude/outside-ideas`.
- **At `42c4557`'s commit time (23:14:04 UTC on 12 October), E's head was `46cb8d0`** (22:25:40): "batch 20 progress, 14 runs saved". Batch 20 was **running, not finished**:
  - the README row says "**IN PROGRESS: 14 of 19 runs saved.**";
  - `s67_continue_results.json` holds 14 of 19 runs;
  - C_P8_N256 272 and C_P16_N256 276–279 were unsaved;
  - no completion line is in `batch20.log`.
- E's next commit, `0b15ced` (23:18:04, "segment 13 end, 14 runs saved"), is still in progress. So is E's head now, `5245ec1` (00:41:55 on 13 October, "16 runs saved").
- The five batch 19 stores are byte-identical from `e3f5809` to `46cb8d0` and to `5245ec1` (§8). The records are therefore v5's.

**Script.** `check_E19_v6.py`, adapted from `check_E19_v5.py`. It uses only the standard library and `rlib`, prints its output and writes nothing.
- §0–§7 are v5's recomputations. Their output is identical to v5's apart from the tex6 labels.
- §8 is rewritten. It gives E's branch at the sixth version's commit time (`git log --until`), with the fifth's for comparison, and adds:
  - batch 20's store and README row at that head;
  - the set of S67 runs batch 20 continues, checked against S67's unbound runs;
  - where the changed wording came from.
- §9 locates the wording.
- §10 prints the rows. Each row is anchored on a fragment of its tex6 line (0 anchors missing), and the counts follow.

**Re-anchoring.** Simply swapping `c5e4508` for `42c4557` in `check_E19_v5.py` leaves 2 anchors missing:
- v5 row 1, tex6:41 "every session's queue is empty";
- v5 row 2, tex6:368 "more often than not".

Both are re-anchored on the new text.

**Method.** Bound routed is recomputed with `rlib.bound_routed`, and failure classes with `rlib.outcome_k`. McNemar is exact and two-sided, as E prints it. The power figures use the independence approximation, as `stats_checks.py` §7 does; §7 also gives the maximal-correlation bound.

**Counts.** 32 rows: **0 DISAGREE**, 0 PARTIAL, 0 NOT RECORD-CHECKABLE and 32 MATCH. In v5 they were 2, 0, 0 and 29 over 31 rows.
- **Both v5 DISAGREEs are now MATCH.**
  - **tex6:368.** "about two times in five" matches P(reads "carries" | 0.6 against 0.8) = 0.398 (independence) and 0.376 (maximal correlation). It now agrees with tex6:237's "about 40%".
  - **tex6:41.** At 23:14:04 batch 20 was running, at 14 of 19 runs (above).
    - "S67's unbound runs continued to 86,400 updates" is exact. The 18 runs batch 20 continues to 86,400 are exactly S67's 18 unbound runs: P16_N256 270–279, P16_D128 271, 272, 273, 275, 276 and 279, P8_N256 272, and the D=128 oracle's seed 270. The nineteenth job is a reproduction check of P8_N256 272 to 44,400.
    - The README's own phrase, "every unbound S67 learned run", leaves out the oracle. The draft's wording is the more accurate of the two.
    - "batches 1--19, complete" is true.
    - The draft calls it "batch~20", not E's "S68", so the clash with F's S68 that v5 noted does not arise.
- **One row is added**, for tex6:349's "a one-pair power stated the wrong way round", and it is MATCH. That text first appeared at `e0c5549` (the fifth version's own new text) and was found by the fifth pass (`check_E19_v5` row 2; `fifth_pass.md`:69).
- **Row 31 (tex6:477) has a new note.** The draft now names batch 20 at tex6:41. tex6:477's "a main-line test with a longer budget" is still not stale, because batch 20 is exploratory.

## §7: the one-pair rule against a reference at 0.8 (10 seeds, "carries" if d ≥ −1), unchanged

| arm's true rate | P(reads "carries") = P(misses the drop), independence | max. correlation (arm's binds ⊂ reference's) |
|---|---|---|
| 0.75 | 0.706 | 0.914 |
| 0.70 | 0.602 | 0.736 |
| 0.65 | 0.498 | 0.544 |
| **0.60** | **0.398** | **0.376** |
| 0.55 | 0.306 | 0.244 |
| 0.50 | 0.226 | 0.149 |

## Rows (DISAGREE first)

| # | tex6 line | item | draft says | recomputed | status | note |
|---|---|---|---|---|---|---|
| 1 | 41 | E's status at 42c4557 (v5 row 1, DISAGREE; changed text) | 'E (... batches 1--19, complete)'; 'E's batch 20 (S67's unbound runs continued to 86,400 updates) and the sessions' fifth round are running' | E's head at 42c4557's commit time: 46cb8d0 (2026-10-12T22:25:40), README 'IN PROGRESS: 14 of 19 runs saved'; store 14 of 19, unsaved [('C_P16_N256', 276), ('C_P16_N256', 277), ('C_P16_N256', 278), ('C_P16_N256', 279), ('C_P8_N256', 272)]; the 18 runs continued to 86,400 are exactly S67's 18 unbound runs (True), plus a reproduction check to 44,400; next E commit 0b15ced (2026-10-12T23:18:04) still in progress | MATCH | Batch 20 was running, not finished, at 23:14:04 UTC (and is still in progress at E's current head). 'S67's unbound runs' is exact (it includes the D=128 oracle's unbound seed, which the README's 'every unbound S67 learned run' omits). The draft does not use E's name S68 for it, so the collision with F's S68 (v5 note) does not arise. 'batches 1--19, complete' is true. |
| 2 | 41 | E's CPU | batch 19 on 2.80 GHz | 186 records, cpu {'Intel(R) Xeon(R) Processor @ 2.80GHz'} | MATCH |  |
| 3 | 57 | index row (fourth pass §2, tex4:57) | '(one-pair rule); the oracle (2 seeds) binds 16 keys at N = 256, the learned recipe 0/10' | oracle 2/2; P16_N256 0/10 | MATCH |  |
| 4 | 57 | index row, scope | scale-free to N=1024, D=128, 4 layers at four streams | S64, S65 'scale-free'; E's verdict 'promising to 4 layers' | MATCH | The rule's reading also covers 6 layers (d = -1). |
| 5 | 235 | S65 'bound routed' (fourth pass §2, tex4:233) | 'bound routed 10/10, 10/10 and 9/10 against 10/10 for three ... (median 4800)' | L2 10, L4 10, L6 9 (bound 10), L3 10; medians {'L2': 3600.0, 'L3': 3600.0, 'L4': 3600.0, 'L6': 4800} | MATCH |  |
| 6 | 235 | S64 | 10/10 each vs 9/10; medians 3600 | {'N256': 9, 'N512': 10, 'N1024': 10, 'D64': 10, 'D128': 10}; medians {'N256': 3600, 'N512': 3600.0, 'N1024': 3600.0, 'D64': 3600.0, 'D128': 3600.0} | MATCH | All four d = +1 rest on seed 273. |
| 7 | 235 | S64/S65 oracles | oracle binds everywhere | S64 10/10, S65 8/8 (§1, §2) | MATCH |  |
| 8 | 235 | S64/S65 readings | scale-free, every d >= -1 | S64 +1 x4; S65 0, 0, -1; RES6 not run | MATCH |  |
| 9 | 235 | transition sentence (unchanged) | as stated | — | MATCH | S66 and S67 also ran N256 arms; 'at the larger widths' is loose (v4 row 15). |
| 10 | 237 | S66's power (fourth pass §2, tex4:235) | 'reads "carries" about 40% of the time ... 0.6 against 0.8' | 0.398 | MATCH | The power caveat now covers eight streams. |
| 11 | 237 | the two-layer arm's tie-break (fourth pass §2, tex4:235) | 'through a tie with four that the arms' order broke, a step outside the pre-registered rule' | phase-2 decision 2026-10-09 16:40:29 UTC: best_depth 2; L2 and L4 10/10 and 10/10, medians 3600.0 and 3600.0, equal parameters; README:657-665 | MATCH | E's further warning (README:699, 'Its 10/10 reads only for 2 layers, not as "fewer layers is better"') is not relayed; the draft makes no such claim. |
| 12 | 237 | the D=128 oracle (fourth pass §2, tex4:235) | 'and at D = 128 on one of two ... the oracle gives no capacity reason ...' | ORC_P16_N256 2/2 at [24000, 27600]; ORC_P16_D128 1/2 at [12000]; README S67 'the oracle gives no capacity reason' | MATCH |  |
| 13 | 237 | completion | 186 of 186 | 186 runs; first draft 9bfa85a 10-10 22:16, e3f5809 10-11 04:25 | MATCH |  |
| 14 | 237 | S66 configuration and reference | k=16, conv, 43,200, 290-299, reference 8/10 | N256_L3 8/10 | MATCH |  |
| 15 | 237 | S66 counts | 8/10 (2 vs 2, d = 0); 10/10 (2 vs 0, p = 0.5, d = +2) | D128 8/10 (2 vs 2, d = +0); L2 10/10 (2 vs 0, two-sided p = 0.5) | MATCH |  |
| 16 | 237 | S66 transitions and probe | 15,600 vs 21,600; higher probe at first check | medians {'N256_L3': 21600.0, 'D128_L3': 15600.0, 'N256_L2': 15600.0}; probe at 4800 medians {'N256_L3': 0.094, 'D128_L3': 0.188, 'N256_L2': 0.312} | MATCH | Descriptions, not readings. |
| 17 | 237 | S66 failures | all failures 3 splits, merged | [('N256_L3', 290, 'MERGED (2 share)', 3), ('N256_L3', 296, 'MERGED (3 share)', 3), ('D128_L3', 293, 'MERGED (3 share)', 3), ('D128_L3', 299, 'MERGED (2 share)', 3)] | MATCH |  |
| 18 | 237 | S67 P=8 | 9/10, 10/10; median 7200 | 9/10, 10/10; medians 7200, 7200.0 | MATCH |  |
| 19 | 237 | S67 P=16 | 0/10 vs 4/10; 4 vs 0; p = 0.125 | 0/10 vs 4/10; 4 vs 0; two-sided p = 0.125 | MATCH |  |
| 20 | 237 | S67 failure classes | most one-to-one unbound | {('P16_N256', 'STREAM-PARTIAL one-to-one'): 6, ('P16_N256', 'MERGED (2 share)'): 3, ('P16_N256', 'MERGED (3 share)'): 1, ('P16_D128', 'STREAM-PARTIAL one-to-one'): 6} | MATCH | 12 of 16. |
| 21 | 237 | S66 schedule note | none at or before 4800 / 2400 | earliest transition 10800 | MATCH |  |
| 22 | 237 | E's lead | as README | README 'Handed back' | MATCH | Text, not records. |
| 23 | 349 | the account of the passes, batch 19's item (changed text) | 'each pass found the previous version's own slips, nearly all in the text added since the pass before (among them ... a one-pair power stated the wrong way round ...)' | tex5:368's 'more often than not' first committed at e0c5549 (the fifth version's own text) and found by the fifth pass (check_E19_v5 row 2, DISAGREE; fifth_pass.md:69) | MATCH | Text against the referee's files, not records. |
| 24 | 368 | S66's power in the verdict paragraph (v5 row 2, DISAGREE; changed text) | 'whose one-pair rule misses a drop from 0.8 to 0.6 about two times in five' | P(reads 'carries' \| 0.6 vs 0.8) = 0.398 (independence), 0.376 (max. correlation) | MATCH | v5's suggested wording, verbatim; now agrees with tex6:237's 'about 40%'. The v5 DISAGREE is resolved. |
| 25 | 368 | S67 in the verdict paragraph | oracle, two seeds; learned 0/10 vs 4/10 | oracle 2/2; 0/10 vs 4/10 | MATCH |  |
| 26 | 368 | four-stream power (unchanged) | 39% at 0.7 vs 0.9 | 0.387 | MATCH |  |
| 27 | 368 | S66 in the verdict paragraph | carry to D=128 and two layers | D128_L3 d = +0, N256_L2 d = +2: both 'carries' | MATCH |  |
| 28 | 443 | scale ranges (unchanged) | 2-6, to 128, to 1024 | S64, S65 | MATCH |  |
| 29 | 454 | 'What Phase VI adds' (fourth pass §2, tex4:452) | 'its constants read "scale-free" and "carries" by one-pair rules ..., to four times the width and to four applications of the layer at four streams ...' | S64 d = +1 x4; S65 L4 d = 0; S66 D128 d = +0, L2 d = +2 | MATCH | The readings are named as one-pair readings; 'twice the depth' (L6) replaced by four applications (E's verdict). |
| 30 | 468 | staged path (fourth pass §2, tex4:464) | 'eight streams at the larger widths (screened once in S66)' | S66 D128_L3 8/10, N256_L2 10/10 | MATCH |  |
| 31 | 477 | next steps (unchanged) | as stated | E's batch 20 is an exploratory continuation of S67 to 86,400, in progress at 42c4557 (46cb8d0) | MATCH | Not stale: batch 20 is not the main-line test. The draft now names batch 20 as running (tex6:41). |
| 32 | 512 | provenance | e3f5809; 186 | 186; stores unchanged from e3f5809 to E's head at 42c4557's time (46cb8d0) and now: True | MATCH |  |
**Readings against the docstrings.** Unchanged from `check_E19_v4.md` and v5:
- every READING block is byte-identical from `617681d` to `e3f5809`;
- every rule predates its first record;
- the phase-2 decision (stored 2026-10-09 16:40:29 UTC) predates every S66 and S67 record.

**The referee's own earlier files.** No error found in `check_E19_v5.md`:
- Its row 1 put E's head at `c5e4508`'s time (19:14:19) at `a876751` (17:49:11, "10 of 19 runs saved"). §8 reproduces that.
- Its §7 table reproduces here.

## Could not check

- **E's advice and the README's prose (row 22).** These are text, so they were compared with the README, not with the records.
- **The power figures.** They are approximations under stated dependence assumptions; see §7.
- **Batch 20's results.**
  - They are in progress, not a result, and the draft cites only the batch's status, so its runs were not recomputed.
  - Only the status, the saved count and the run set it continues were checked.
- **The rest of tex6:41's status.** "The sessions' fifth round are running" is checked here only as far as E is concerned. F, G, H and I all committed between 22:53 and 23:13 on 12 October; their checks cover them.

<details><summary>Full output of <code>python3 docs/review_2026-10/rev9/check_E19_v6.py</code></summary>

```
== 0. Stores at e3f5809: run counts, ok, dry, CPU, code SHA; stored outcome vs recomputed
  scale_width: 60 runs; not ok 0; dry 0; cpu {'Intel(R) Xeon(R) Processor @ 2.80GHz'}; code {'17e56410a1db'}; BOUND ROUTED stored-vs-recomputed mismatches []
  scale_depth: 48 runs; not ok 0; dry 0; cpu {'Intel(R) Xeon(R) Processor @ 2.80GHz'}; code {'17e56410a1db'}; BOUND ROUTED stored-vs-recomputed mismatches []
  scale_eight: 30 runs; not ok 0; dry 0; cpu {'Intel(R) Xeon(R) Processor @ 2.80GHz'}; code {'17e56410a1db'}; BOUND ROUTED stored-vs-recomputed mismatches []
  capacity: 48 runs; not ok 0; dry 0; cpu {'Intel(R) Xeon(R) Processor @ 2.80GHz'}; code {'17e56410a1db'}; BOUND ROUTED stored-vs-recomputed mismatches []
  total 186 (draft tex6:237 and tex6:512: 186)
  rlib ties (rounded stream_gate) so far: []

== 1. S64 width (seeds 270-279, oracle 270-271; reading: 'scale-free' if every tested d >= -1, 'width-sensitive' if any d <= -4)
  oracle N256: bound 2/2 at [1200, 1200]
  oracle N512: bound 2/2 at [1200, 1200]
  oracle N1024: bound 2/2 at [1200, 1200]
  oracle D64: bound 2/2 at [1200, 1200]
  oracle D128: bound 2/2 at [1200, 1200]
  N256 (reference): 9/10 (RELIABLE, Wilson [0.60, 0.98]); transition median 3600; [3600, 3600, 3600, 3600, 3600, 3600, 3600, 4800, 4800]
  N512: 10/10 (RELIABLE, Wilson [0.72, 1.00]); transition median 3600.0; [3600, 3600, 3600, 3600, 3600, 3600, 3600, 4800, 6000, 12000]; vs N256 1 / 0, d = +1, two-sided p = 1 (one-sided 0.5)
  N1024: 10/10 (RELIABLE, Wilson [0.72, 1.00]); transition median 3600.0; [2400, 2400, 2400, 2400, 3600, 3600, 3600, 3600, 3600, 6000]; vs N256 1 / 0, d = +1, two-sided p = 1 (one-sided 0.5)
  D64: 10/10 (RELIABLE, Wilson [0.72, 1.00]); transition median 3600.0; [2400, 2400, 2400, 3600, 3600, 3600, 3600, 3600, 3600, 4800]; vs N256 1 / 0, d = +1, two-sided p = 1 (one-sided 0.5)
  D128: 10/10 (RELIABLE, Wilson [0.72, 1.00]); transition median 3600.0; [2400, 2400, 2400, 3600, 3600, 3600, 3600, 3600, 3600, 6000]; vs N256 1 / 0, d = +1, two-sided p = 1 (one-sided 0.5)
  ref failures: [(273, 'MERGED (2 share)')]
  reading: scale-free

== 2. S65 depth (seeds 280-289, oracle 280-281; same rule vs L3; RES6 only if L6 - L3 <= -2)
  oracle L2: bound 2/2 at [1200, 1200]
  oracle L3: bound 2/2 at [1200, 1200]
  oracle L4: bound 2/2 at [1200, 1200]
  oracle L6: bound 2/2 at [2400, 2400]
  L3 (reference): 10/10 (RELIABLE, Wilson [0.72, 1.00]); transition median 3600.0; [3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 6000]
  L2: 10/10 (RELIABLE, Wilson [0.72, 1.00]); transition median 3600.0; [2400, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600]; vs L3 0 / 0, d = +0, two-sided p = 1
     failures []
  L4: 10/10 (RELIABLE, Wilson [0.72, 1.00]); transition median 3600.0; [3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 4800, 6000]; vs L3 0 / 0, d = +0, two-sided p = 1
     failures []
  L6: 9/10 (RELIABLE, Wilson [0.60, 0.98]); transition median 4800; [3600, 3600, 4800, 4800, 4800, 4800, 4800, 7200, 25200]; vs L3 0 / 1, d = -1, two-sided p = 1
     failures [(283, 'BOUND NOT routed')]
  reading: scale-free; RES6 condition (L6 - L3 <= -2): False; RES6 records: 0

== 3. S66 eight streams (seeds 290-299; reading per arm: 'carries' if d >= -1 vs N256_L3)
  N256_L3: 8/10 (MAJORITY, Wilson [0.49, 0.94]); transition median 21600.0; [13200, 15600, 15600, 20400, 22800, 22800, 25200, 30000]
     failures (seed, class, splits, final acc): [(290, 'MERGED (2 share)', 3, 0.63), (296, 'MERGED (3 share)', 3, 0.49)]
     splits per run [3, 2, 1, 2, 2, 2, 3, 2, 1, 3], total 21
     probe accuracy at the first check (4800): median 0.094; [0.11, 0.09, 0.09, 0.23, 0.09, 0.14, 0.14, 0.08, 0.06, 0.08]
     transitions (all bound runs) [13200, 15600, 15600, 20400, 22800, 22800, 25200, 30000]; recorded 'transition' field agrees: True
  D128_L3: 8/10 (MAJORITY, Wilson [0.49, 0.94]); transition median 15600.0; [10800, 12000, 14400, 15600, 15600, 19200, 21600, 27600]; vs N256_L3 2 / 2, d = +0, two-sided p = 1 (one-sided 0.688) -> carries
     failures (seed, class, splits, final acc): [(293, 'MERGED (3 share)', 3, 0.49), (299, 'MERGED (2 share)', 3, 0.88)]
     splits per run [1, 2, 3, 3, 0, 2, 3, 3, 1, 3], total 21
     probe accuracy at the first check (4800): median 0.188; [0.22, 0.27, 0.42, 0.11, 0.47, 0.12, 0.16, 0.14, 0.61, 0.14]
     transitions (all bound runs) [10800, 12000, 14400, 15600, 15600, 19200, 21600, 27600]; recorded 'transition' field agrees: True
  N256_L2: 10/10 (RELIABLE, Wilson [0.72, 1.00]); transition median 15600.0; [13200, 14400, 15600, 15600, 15600, 15600, 20400, 20400, 24000, 25200]; vs N256_L3 2 / 0, d = +2, two-sided p = 0.5 (one-sided 0.25) -> carries
     failures (seed, class, splits, final acc): []
     splits per run [1, 1, 2, 3, 1, 3, 2, 3, 1, 2], total 19
     probe accuracy at the first check (4800): median 0.312; [0.23, 0.66, 0.22, 0.12, 0.28, 0.53, 0.34, 0.17, 0.67, 0.77]
     transitions (all bound runs) [13200, 14400, 15600, 15600, 15600, 15600, 20400, 20400, 24000, 25200]; recorded 'transition' field agrees: True
  earliest transition in S66: 10800; any at or before 4800: False; at or before 2400: False
  docstring ALSO REPORTED rule: 'if an arm's median transition is at or before 4800, the schedule is size-dependent'

== 4. S67 capacity (oracle 270-271, learned 270-279; reading on the P=16 oracles only)
  oracle P=8 N256: bound 2/2; [(270, 2400, 1.0), (271, 2400, 1.0)]
  oracle P=8 D128: bound 2/2; [(270, 2400, 1.0), (271, 2400, 1.0)]
  oracle P=16 N256: bound 2/2; [(270, 27600, 0.98), (271, 24000, 0.99)]
  oracle P=16 D128: bound 1/2; [(270, None, 0.32), (271, 12000, 0.97)]
  reading: not capacity-bound at N=256
  P8_N256: 9/10 (RELIABLE, Wilson [0.60, 0.98]); transition median 7200; [6000, 6000, 7200, 7200, 7200, 8400, 8400, 10800, 37200]
     failures: [(272, 'MERGED (2 share)', 0.74)]
     failure classes {'MERGED (2 share)': 1}
  P8_D128: 10/10 (RELIABLE, Wilson [0.72, 1.00]); transition median 7200.0; [6000, 6000, 6000, 6000, 7200, 7200, 7200, 7200, 7200, 15600]
     failures: []
     failure classes {}
  P=8: D128 only 1 / N256 only 0, two-sided p = 1 (one-sided 0.5); printed, no reading
  P16_N256: 0/10 (NEVER, Wilson [0.00, 0.28]); transition median None; []
     failures: [(270, 'STREAM-PARTIAL one-to-one', 0.15), (271, 'MERGED (2 share)', 0.11), (272, 'STREAM-PARTIAL one-to-one', 0.14), (273, 'STREAM-PARTIAL one-to-one', 0.59), (274, 'STREAM-PARTIAL one-to-one', 0.72), (275, 'STREAM-PARTIAL one-to-one', 0.67), (276, 'STREAM-PARTIAL one-to-one', 0.29), (277, 'MERGED (2 share)', 0.16), (278, 'MERGED (3 share)', 0.18), (279, 'MERGED (2 share)', 0.12)]
     failure classes {'STREAM-PARTIAL one-to-one': 6, 'MERGED (2 share)': 3, 'MERGED (3 share)': 1}
  P16_D128: 4/10 (MINORITY, Wilson [0.17, 0.69]); transition median 31800.0; [19200, 26400, 37200, 38400]
     failures: [(271, 'STREAM-PARTIAL one-to-one', 0.29), (272, 'STREAM-PARTIAL one-to-one', 0.4), (273, 'STREAM-PARTIAL one-to-one', 0.28), (275, 'STREAM-PARTIAL one-to-one', 0.31), (276, 'STREAM-PARTIAL one-to-one', 0.27), (279, 'STREAM-PARTIAL one-to-one', 0.31)]
     failure classes {'STREAM-PARTIAL one-to-one': 6}
  P=16: D128 only 4 / N256 only 0, two-sided p = 0.125 (one-sided 0.0625); printed, no reading

== 5. Operating characteristic of 'd >= -1 over 10 seeds' (independence approximation, as stats_checks.py §7)
  reference 0.9, arm 0.7: P(reads 'scale-free' / 'carries') = 0.39
  reference 0.9, arm 0.6: P(reads 'scale-free' / 'carries') = 0.20
  reference 0.8, arm 0.8: P(reads 'scale-free' / 'carries') = 0.80
  reference 0.8, arm 0.6: P(reads 'scale-free' / 'carries') = 0.40
  reference 0.8, arm 0.5: P(reads 'scale-free' / 'carries') = 0.23
  reference 0.8, arm 0.4: P(reads 'scale-free' / 'carries') = 0.11

== 6. Readings commit vs first record
  docstrings first committed: 617681d 2026-10-09 06:17:08 +0000
  READING blocks unchanged 617681d..e3f5809: True
  first scale_width record: ('7f1a71d 2026-10-09 09:20:22 +0000', ['ORC_D128|270|17e56410a1db|Intel(R) Xeon(R) Processor @ 2.80GHz'])
  first scale_depth record: ('e44e794 2026-10-09 14:06:26 +0000', ['ORC_L4|280|17e56410a1db|Intel(R) Xeon(R) Processor @ 2.80GHz'])
  first scale_eight record: ('1bbf799 2026-10-09 17:21:19 +0000', ['N256_L3|292|17e56410a1db|Intel(R) Xeon(R) Processor @ 2.80GHz'])
  first capacity record: ('9ee5263 2026-10-10 01:56:39 +0000', ['ORC_P16_D128|271|17e56410a1db|Intel(R) Xeon(R) Processor @ 2.80GHz'])
  phase-2 decision stored 2026-10-09 16:40:29 UTC: best_width D128, best_depth 2

== 7. 'carries' (d >= -1, 10 seeds) against a reference at 0.8: P(reads 'carries') = P(misses the drop)
  arm 0.75 vs 0.8: independence 0.706; arm's binds a subset of the reference's (max. correlation) 0.914
  arm 0.7 vs 0.8: independence 0.602; arm's binds a subset of the reference's (max. correlation) 0.736
  arm 0.65 vs 0.8: independence 0.498; arm's binds a subset of the reference's (max. correlation) 0.544
  arm 0.6 vs 0.8: independence 0.398; arm's binds a subset of the reference's (max. correlation) 0.376
  arm 0.55 vs 0.8: independence 0.306; arm's binds a subset of the reference's (max. correlation) 0.244
  arm 0.5 vs 0.8: independence 0.226; arm's binds a subset of the reference's (max. correlation) 0.149
  P(carries) first falls below 0.5 at an arm rate of 0.65 (independence): a drop to 0.6 is missed 0.40 of the time, i.e. caught 0.60, not 'missed more often than not'
  four streams, 0.7 vs 0.9: P(reads 'scale-free') = 0.387 (tex6:368 '39%')

== 8. E's branch at the sixth version's commit (42c4557), and at the fifth's (c5e4508) for comparison
  42c4557 at 2026-10-12T23:14:04+00:00; E's last commit before it: 46cb8d0 2026-10-12T22:25:40+00:00 explore_out: batch 20 progress, 14 runs saved (EXPLORATORY, not a result)
  c5e4508 at 2026-10-12T19:14:19+00:00; E's last commit before it: a876751 2026-10-12T17:49:11+00:00 explore_out: batch 20 progress, 10 runs saved (EXPLORATORY, not a result)
  E's first commit after 42c4557: 0b15ced 2026-10-12T23:18:04+00:00 explore_out: batch 20 segment 13 end, 14 runs saved (EXPLORATORY, not a result); E's head now: 5245ec1 2026-10-13T00:41:55+00:00 explore_out: batch 20 progress, 16 runs saved (EXPLORATORY, not a result)
  commits after e3f5809 up to 46cb8d0: 65; first 4acf915 2026-10-11T21:27:31+00:00 Batch 20 (S68 continue): code and readings fixed before any run (EXPLORATORY, not a result)
  batch 20's first segment: 404d3bd 2026-10-11T22:04:45+00:00 explore_out: batch 20 segment 1 started, CHECK pass running (EXPLORATORY, not a result)
  README at a876751, batch 20 row status: IN PROGRESS: 10 of 19 runs saved.
  README at 46cb8d0, batch 20 row status: IN PROGRESS: 14 of 19 runs saved.
  s67_continue store at 46cb8d0: 14 runs saved of 19 specified; started 2026-10-11 22:00:03
  not yet saved at 46cb8d0: [('C_P16_N256', 276), ('C_P16_N256', 277), ('C_P16_N256', 278), ('C_P16_N256', 279), ('C_P8_N256', 272)]
  batch20.log at 46cb8d0: 'batch complete'/'ALL RUNS SAVED'-like line: none
  S67 runs not bound routed (learned) or not bound (oracle): 18: [('ORC_P16_D128', 270), ('P16_D128', 271), ('P16_D128', 272), ('P16_D128', 273), ('P16_D128', 275), ('P16_D128', 276), ('P16_D128', 279), ('P16_N256', 270), ('P16_N256', 271), ('P16_N256', 272), ('P16_N256', 273), ('P16_N256', 274), ('P16_N256', 275), ('P16_N256', 276), ('P16_N256', 277), ('P16_N256', 278), ('P16_N256', 279), ('P8_N256', 272)]
  batch 20 arms continued to 86,400: 18 runs; equal to S67's unbound set: True; other arms: [('RCHK_P8_N256', 44400, [272])]
  batch 19 stores byte-identical e3f5809 .. 46cb8d0 and .. origin head (5245ec1): True
  'every session's queue is empty' removed at b5fd2f2 2026-10-12T22:53:57+00:00; "E's batch~20" introduced at b5fd2f2 2026-10-12T22:53:57+00:00
  fifth_pass.md lines on the one-pair power: [(69, "- tex5:368, the one-pair rule's power stated the wrong way round;"), (182, '| 368 | "about two times in five" | RESOLVED (0.398) |')]; check_E19_v5.md DISAGREE rows on it: [56]
  'more often than not' first in the draft at e0c5549 2026-10-11T15:44:42+00:00

== 9. Where the sixth version says it (tex6; tex5 for comparison)
  'batch~?19': tex6 [41, 368, 477]; tex5 [41, 368, 477]
  'S66': tex6 [237, 368, 468, 512]; tex5 [237, 368, 468, 512]
  'S67': tex6 [41, 237, 368, 512]; tex5 [237, 368, 512]
  'one-pair rule': tex6 [57, 368, 454]; tex5 [57, 368, 454]
  '40\\\\%': tex6 [237]; tex5 [237]
  'more often than not': tex6 []; tex5 [368]
  'one of two': tex6 [237, 265]; tex5 [237, 265]
  'best depth': tex6 [237]; tex5 [237]
  'bound routed \\$10/10\\$, \\$10/10\\$ and \\$9/10\\$': tex6 [235]; tex5 [235]
  'screened once in S66': tex6 [468]; tex5 [468]
  'e3f5809': tex6 [512]; tex5 [512]
  'queue is empty': tex6 []; tex5 [41]
  'batches 1--19': tex6 [41]; tex5 [41]
  'capacity reason': tex6 [237]; tex5 [237]
  'four applications of': tex6 [454]; tex5 [454]
  'about two times in five': tex6 [368]; tex5 []
  'batch~20': tex6 [41]; tex5 []
  '86\\{,\\}400': tex6 [41]; tex5 []
  'fifth round': tex6 [41]; tex5 []
  'one-pair power stated the wrong way round': tex6 [349]; tex5 []

== 10. Rows (each anchored on a fragment of its tex6 line) and counts
   1 | tex6:41 | anchor ok | MATCH | E's status at 42c4557 (v5 row 1, DISAGREE; changed text) | recomputed: E's head at 42c4557's commit time: 46cb8d0 (2026-10-12T22:25:40), README 'IN PROGRESS: 14 of 19 runs saved'; store 14 of 19, unsaved [('C_P16_N256', 276), ('C_P16_N256', 277), ('C_P16_N256', 278), ('C_P16_N256', 279), ('C_P8_N256', 272)]; the 18 runs continued to 86,400 are exactly S67's 18 unbound runs (True), plus a reproduction check to 44,400; next E commit 0b15ced (2026-10-12T23:18:04) still in progress | Batch 20 was running, not finished, at 23:14:04 UTC (and is still in progress at E's current head). 'S67's unbound runs' is exact (it includes the D=128 oracle's unbound seed, which the README's 'every unbound S67 learned run' omits). The draft does not use E's name S68 for it, so the collision with F's S68 (v5 note) does not arise. 'batches 1--19, complete' is true.
   2 | tex6:41 | anchor ok | MATCH | E's CPU | recomputed: 186 records, cpu {'Intel(R) Xeon(R) Processor @ 2.80GHz'}
   3 | tex6:57 | anchor ok | MATCH | index row (fourth pass §2, tex4:57) | recomputed: oracle 2/2; P16_N256 0/10
   4 | tex6:57 | anchor ok | MATCH | index row, scope | recomputed: S64, S65 'scale-free'; E's verdict 'promising to 4 layers' | The rule's reading also covers 6 layers (d = -1).
   5 | tex6:235 | anchor ok | MATCH | S65 'bound routed' (fourth pass §2, tex4:233) | recomputed: L2 10, L4 10, L6 9 (bound 10), L3 10; medians {'L2': 3600.0, 'L3': 3600.0, 'L4': 3600.0, 'L6': 4800}
   6 | tex6:235 | anchor ok | MATCH | S64 | recomputed: {'N256': 9, 'N512': 10, 'N1024': 10, 'D64': 10, 'D128': 10}; medians {'N256': 3600, 'N512': 3600.0, 'N1024': 3600.0, 'D64': 3600.0, 'D128': 3600.0} | All four d = +1 rest on seed 273.
   7 | tex6:235 | anchor ok | MATCH | S64/S65 oracles | recomputed: S64 10/10, S65 8/8 (§1, §2)
   8 | tex6:235 | anchor ok | MATCH | S64/S65 readings | recomputed: S64 +1 x4; S65 0, 0, -1; RES6 not run
   9 | tex6:235 | anchor ok | MATCH | transition sentence (unchanged) | recomputed: — | S66 and S67 also ran N256 arms; 'at the larger widths' is loose (v4 row 15).
  10 | tex6:237 | anchor ok | MATCH | S66's power (fourth pass §2, tex4:235) | recomputed: 0.398 | The power caveat now covers eight streams.
  11 | tex6:237 | anchor ok | MATCH | the two-layer arm's tie-break (fourth pass §2, tex4:235) | recomputed: phase-2 decision 2026-10-09 16:40:29 UTC: best_depth 2; L2 and L4 10/10 and 10/10, medians 3600.0 and 3600.0, equal parameters; README:657-665 | E's further warning (README:699, 'Its 10/10 reads only for 2 layers, not as "fewer layers is better"') is not relayed; the draft makes no such claim.
  12 | tex6:237 | anchor ok | MATCH | the D=128 oracle (fourth pass §2, tex4:235) | recomputed: ORC_P16_N256 2/2 at [24000, 27600]; ORC_P16_D128 1/2 at [12000]; README S67 'the oracle gives no capacity reason'
  13 | tex6:237 | anchor ok | MATCH | completion | recomputed: 186 runs; first draft 9bfa85a 10-10 22:16, e3f5809 10-11 04:25
  14 | tex6:237 | anchor ok | MATCH | S66 configuration and reference | recomputed: N256_L3 8/10
  15 | tex6:237 | anchor ok | MATCH | S66 counts | recomputed: D128 8/10 (2 vs 2, d = +0); L2 10/10 (2 vs 0, two-sided p = 0.5)
  16 | tex6:237 | anchor ok | MATCH | S66 transitions and probe | recomputed: medians {'N256_L3': 21600.0, 'D128_L3': 15600.0, 'N256_L2': 15600.0}; probe at 4800 medians {'N256_L3': 0.094, 'D128_L3': 0.188, 'N256_L2': 0.312} | Descriptions, not readings.
  17 | tex6:237 | anchor ok | MATCH | S66 failures | recomputed: [('N256_L3', 290, 'MERGED (2 share)', 3), ('N256_L3', 296, 'MERGED (3 share)', 3), ('D128_L3', 293, 'MERGED (3 share)', 3), ('D128_L3', 299, 'MERGED (2 share)', 3)]
  18 | tex6:237 | anchor ok | MATCH | S67 P=8 | recomputed: 9/10, 10/10; medians 7200, 7200.0
  19 | tex6:237 | anchor ok | MATCH | S67 P=16 | recomputed: 0/10 vs 4/10; 4 vs 0; two-sided p = 0.125
  20 | tex6:237 | anchor ok | MATCH | S67 failure classes | recomputed: {('P16_N256', 'STREAM-PARTIAL one-to-one'): 6, ('P16_N256', 'MERGED (2 share)'): 3, ('P16_N256', 'MERGED (3 share)'): 1, ('P16_D128', 'STREAM-PARTIAL one-to-one'): 6} | 12 of 16.
  21 | tex6:237 | anchor ok | MATCH | S66 schedule note | recomputed: earliest transition 10800
  22 | tex6:237 | anchor ok | MATCH | E's lead | recomputed: README 'Handed back' | Text, not records.
  23 | tex6:349 | anchor ok | MATCH | the account of the passes, batch 19's item (changed text) | recomputed: tex5:368's 'more often than not' first committed at e0c5549 (the fifth version's own text) and found by the fifth pass (check_E19_v5 row 2, DISAGREE; fifth_pass.md:69) | Text against the referee's files, not records.
  24 | tex6:368 | anchor ok | MATCH | S66's power in the verdict paragraph (v5 row 2, DISAGREE; changed text) | recomputed: P(reads 'carries' | 0.6 vs 0.8) = 0.398 (independence), 0.376 (max. correlation) | v5's suggested wording, verbatim; now agrees with tex6:237's 'about 40%'. The v5 DISAGREE is resolved.
  25 | tex6:368 | anchor ok | MATCH | S67 in the verdict paragraph | recomputed: oracle 2/2; 0/10 vs 4/10
  26 | tex6:368 | anchor ok | MATCH | four-stream power (unchanged) | recomputed: 0.387
  27 | tex6:368 | anchor ok | MATCH | S66 in the verdict paragraph | recomputed: D128_L3 d = +0, N256_L2 d = +2: both 'carries'
  28 | tex6:443 | anchor ok | MATCH | scale ranges (unchanged) | recomputed: S64, S65
  29 | tex6:454 | anchor ok | MATCH | 'What Phase VI adds' (fourth pass §2, tex4:452) | recomputed: S64 d = +1 x4; S65 L4 d = 0; S66 D128 d = +0, L2 d = +2 | The readings are named as one-pair readings; 'twice the depth' (L6) replaced by four applications (E's verdict).
  30 | tex6:468 | anchor ok | MATCH | staged path (fourth pass §2, tex4:464) | recomputed: S66 D128_L3 8/10, N256_L2 10/10
  31 | tex6:477 | anchor ok | MATCH | next steps (unchanged) | recomputed: E's batch 20 is an exploratory continuation of S67 to 86,400, in progress at 42c4557 (46cb8d0) | Not stale: batch 20 is not the main-line test. The draft now names batch 20 as running (tex6:41).
  32 | tex6:512 | anchor ok | MATCH | provenance | recomputed: 186; stores unchanged from e3f5809 to E's head at 42c4557's time (46cb8d0) and now: True

  COUNTS: 0 DISAGREE, 0 PARTIAL, 0 NOT RECORD-CHECKABLE, 32 MATCH (32 rows); anchors missing: 0
```

</details>
