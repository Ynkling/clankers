# S75b: S75 at S = 4 on a configuration where the bar is reachable (DRAFT spec, 2026-10-12, EXPLORATORY)

**Status: draft for the user.** Nothing here has run, and no S75b code exists yet. The spec follows the follow-up rule
committed in `explore_out/I/gpu/s75_S4_prerun_note_2026-10-12.md` (5bc6c2e):
- fresh seeds;
- every constant derived only from ORACLE and SINGLE quantities (S75's ORACLE 830–831 and SINGLE 830–839, the
  ideal-learner computations, and a new 2-seed ORACLE pilot of the exact configuration), never from S75's WIDE_SPLIT,
  REG3 or WIDE_SLOW outcomes;
- the configuration changed rather than the bar;
- the split guarded at k = S.

**Provenance: the design commit was built on a tree without S75's learned arms. Its blindness rests on the author's
statement; every change after the merge 142f47e is not blind by git (Appendix B).**
- **The design.** The configuration, the predictions, the pilot rules, the guard, the arms and the seeds were committed
  at 1e31409. Its parent is 5bc6c2e, whose tree holds only the ORACLE 830–831 and SINGLE 830–839 records.
- **What git can and cannot show.** On the author's account (git cannot show it: the reflog is not pushed), the
  learned-arm records (b649751) had been fetched into the local object store at
  14:10 UTC, before that commit. On the author's statement they were neither merged nor opened then; a parent shows
  only what the tree held. They were merged at 142f47e (14:16; the server's push times: b649751 13:53:11, 142f47e
  14:16:16, Appendix C).
- **Later edits.** The cost section (ee97443) and the revision 114cd08 were made after that merge, with b649751's
  records and `s75_S4_summary.md` in the tree. They are not blind by git (Appendix B).
  - The review behind 114cd08 was run in this session.
  - That it, and the edits, were made without opening S75's learned-arm files is the author's statement.
  - The same session committed the S75 write-up (03bffdf) 130 s after 114cd08.
  - On the author's statement, the edits draw on:
    - the two CUDA CHECKs' K9 benchmarks;
    - CPU profiles and memory measurements at L = 1536;
    - ideal-learner computations;
    - the 5bc6c2e ORACLE and SINGLE records and the pre-run note (5bc6c2e);
    - the review behind 114cd08.
- **The revision.** It changed one constant, the pilot proceed rule, from "both runs ≥ 0.912" to the rule below; the
  reason is given there. It made run decisions that had been open fixed, among them WIDE_SLOW unconditional, which
  Appendix B.1 counts with the proceed rule as its two design changes. Both are **disclosed, not blind by git**
  (Appendix C.5). It also corrected numbers. The diff from 1e31409 shows every change.

## Question

The question is S75's at S = 4: does the channel partition emerge label-free at four sources under the dense loss? S75
could not answer it: its ORACLE stopped at 0.892 and 0.880, under the 0.9 bar (UNTESTED).

## Configuration B

ICMC with S = 4, V = 4, blocks of 6–14 words, NB = 32 blocks per source, M = 352 words per source, L = 1536 (the model
reads 1535). The model is unchanged (BDH, N = 256, D = 32, 3 layers, decay 0.95, conv width 4). The rule changes the
configuration, not the model, so the decay is kept, although the pre-run note names it in S75's shortfall. B = 32, Adam
1e-3, evaluation every 1200 updates on 1024 held-out sequences, the probe and the routing statistics are as in S75. Each
run is 24,000 updates.

**Why B.** It keeps S75's block structure and doubles each source's history.
- Block lengths, S = 4 and the random interleaving are unchanged, so the gaps between a source's blocks are distributed
  as in S75. That gap is what made the first word of a block hard in S75: bin 1 was 0.719 and 0.712 against a bound of
  0.943.
- The longer history raises the Bayes bound from 0.927 to 0.963.
- Bin shares among the loss positions are unchanged: 0.088 / 0.091 / 0.182 / 0.638, against 0.086 / 0.091 / 0.183 /
  0.640 in S75.

**Ideal learners** (`explore_i_task.ideal_set_acc`, 2 × 128 sequences; the windowed learner on 48):

| learner | S75 (L = 768) | B (L = 1536) |
|---|---|---|
| own, the Bayes bound | 0.927 | 0.963 (the pre-run note's 0.964 came from another sample) |
| own, window W = 220 / 250 tokens | 0.897 / 0.908 | 0.927 / 0.939 |
| own_fade (decay 0.95, 58-token horizon) | 0.679 | 0.691 |
| mixed (sources pooled) | 0.496 | 0.517 |

**Predicted ORACLE at B.**
- *Per-bin transfer.* Take S75's ORACLE shortfall below the bound in each bin, apply it to B's bound in that bin, and
  weight by B's bin shares. This gives 0.928 (seed 830) and 0.916 (seed 831); mean 0.922. The share of the
  faded-to-bound gap that S75's ORACLE closed, applied to B, gives 0.919.
- *Fixed reach, the pessimistic case.* S75's ORACLE bins 2+ match a hard-window per-source learner with W ≈ 220
  (seed 831) to 250 (seed 830). Suppose the model's reach is fixed in tokens. Carrying that window to B, and keeping
  S75's bin 1, gives 0.909 (831-like) and 0.920 (830-like); mean 0.914. B still helps here, because more of each
  source's evidence falls inside the window.
- *Risks.* The weaker seed may land near the proceed line, and training at L = 1536 may differ in ways no ideal learner
  captures. That is why B is piloted before the screen.

**Predicted SINGLE at B.** The mixed learner (0.517) plus S75's SINGLE excess over it (median 0.538 − 0.496 = +0.043)
gives about 0.56, far below ORACLE − 0.2.

**Configurations not chosen.**
- A (NB = 24, L = 1152) is predicted at about 0.91, too close to the bar.
- Longer blocks (12–28 or 16–32 words, L of about 1200–1350) are predicted at about 0.92. They change the gap
  distribution, so their predictions rest on a weaker transfer.
- Longer sequences than B would cost about (L / 768)² in attention: 9× S75's cost at L = 2304.

## Stage 1: ORACLE pilot of B

**The pilot.** The arm ORACLE_PILOT (the perfect gate, k = 4, ADAM), seeds 840 and 841, 24,000 updates. It is its own
arm with its own records. It never enters VALID, GAP or any stage-2 aggregate, and it never substitutes for a failed
stage-2 ORACLE run.

**Proceed rule (fixed now).** Both rules read `end.set`: the overall SET of the final evaluation, at 24,000 updates on
the 1024 held-out sequences, unrounded. Go to stage 2 if **both**:
1. the mean of the two pilot runs is ≥ 0.912, which is 0.9 plus S75's ORACLE seed spread (0.892 − 0.880 = 0.012),
   rounded up from the unrounded 0.9 + 0.01144 = 0.9114 (Appendix C; the constant stays 0.912);
2. each pilot run is ≥ 0.9.

**Why this rule.** The design commit 1e31409 had "both runs ≥ 0.912". The review showed that under the fixed-reach prediction
that rule stops the screen about 64% of the time, even though such a screen would usually be VALID. Under the per-bin
prediction it stops about 20% of the time. Under this rule the stop rates are about 0.05 (per-bin) and 0.40 (fixed
reach), with a seed sd of about 0.008. When the screen proceeds, both of its ORACLE runs are expected to clear 0.9 with
probability about 0.92–0.99. The rates for sd 0.004–0.012, the sd's interval from one pair, and seed-specific means
are in Appendix C. This rule is the revision's (114cd08): disclosed, not blind by git.

**Stop rule.** If the pilot fails, S75b stops.
- The report says that B's pilot did not clear the proceed rule. It says "S = 4 is not usable at B" only if a pilot run
  ended below 0.9 or diverged (Appendix B.3, a later change). This label is set by this rule alone; it is not S74's
  usability statistic, which uses ORACLE's mean (Appendix C).
- No further configuration is tried under this spec.
- Any successor is a new spec under the pre-run note's follow-up rule. It may use S75b's pilot only as a stated ORACLE
  input, and it records whether its author had read S75's learned-arm records.

**SPLIT_THR := min(end.set of ORACLE_PILOT 840, 841) − 0.03**, unrounded. This is S75's own rule (lowest ORACLE final
SET − 0.03), applied to the same configuration's pilot instead of being carried over from S = 2.

**Mechanics.** The driver is committed before stage 1. At the start of stage 2 it applies the proceed rule and computes
SPLIT_THR from the ORACLE_PILOT records under its own code SHA. It refuses stage 2 unless both records exist and pass.
It writes SPLIT_THR and the stage-1 commit into stage 2's meta, and the stage-1 record is committed before stage 2 runs.

## Stage 2: the screen (seeds 850–859)

| arm | seeds | what it is | role |
|---|---|---|---|
| ORACLE | 850–851 | perfect gate, k = 4, ADAM | validity |
| SINGLE | 850–859 | one channel, ADAM | the interference baseline |
| WIDE_SPLIT_G | 850–859 | WIDE window gate (width 16), k = 4, SLOW, split with the guard | read |
| REG3 | 850–859 | Session G's register gate at all three layers, k = 4, SLOW, as S75 | read |
| WIDE_SLOW | 850–859 | WIDE_SPLIT_G without the split | printed, never read; runs unconditionally (since 114cd08: disclosed, not blind by git) |

**The split trigger** is S75's, transcribed. It checks every 2400 updates from 2400 to 21,600 on the 64-sequence probe.
The check at 2400 is the reference: it records the probe SET and is never eligible. From 4800 on, a check is eligible
when probe SET < SPLIT_THR and the rise since the previous check is < 0.02. At most 3 fires, at least 4800 apart. The
guard adds one rule; checks are decided in this order:
1. A check that is not eligible is recorded as in S75.
2. An eligible check that is capped or gapped is recorded as blocked by `cap` or `gap`, as in S75: `cap` when both hold.
3. **An eligible check that is not capped or gapped, and whose source → channel map at word positions is one-to-one, is
   recorded as blocked by `one-to-one`.** The fire count and the last-fire step are unchanged; `prev` is updated as for
   every check. The test is `routing_layer`'s `one_to_one` field alone, on the trigger's 64-sequence probe at the
   check's step, at the first gate layer: half of the test ROUTED uses, which also needs η² ≥ 0.9 (Appendix C.3).
4. Otherwise (the map is not one-to-one) the split fires, also when the least-used channel `c0` is in the map
   (`on_target` false), which overwrites a live channel (Appendix C.3).

Every row also records the map's η², margin and `one_to_one`. With four sources on four channels there is nothing to
separate, and a fire would overwrite a live channel: the hazard S75's note declared. An eligible check that is capped
or gapped on a one-to-one map is recorded as `cap` or `gap`, with its `one_to_one` in the row (Appendix C.2).

**Fixed now for stage 2.**
- Every arm runs all its seeds, whatever ORACLE 850–851 show and in whatever order runs finish.
- If the result is not VALID, it is UNTESTED. ROUTED counts (with Wilson intervals), GAP, η², margin and the probe are
  then printed as descriptions only, with no "carries" or "does not carry" label. They are never re-read under a later
  bar, and the pilot runs are never pooled with 850–851.

**Outcomes, VALID and READING (S75's, unchanged).**
- ROUTED := at every gate layer, η² by source at word positions ≥ 0.9 and a one-to-one map, at the end.
- GAP := (SET(arm) − SET(SINGLE, same seed)) / (mean SET(ORACLE 850–851) − SET(SINGLE, same seed)), overall and by bin.
- VALID := both ORACLE runs 850–851 end with SET ≥ 0.9.
- The reading is "carries to four sources" if the better of WIDE_SPLIT_G and REG3 is ROUTED on ≥ 8/10, and "does not
  carry" otherwise. It is UNTESTED if not VALID.
- **Printed:** ROUTED with Wilson interval and band; GAP and GAP by bin; SET; η² and margin per layer; the probe per
  layer; REG3's β; S74's usability count; every split check (fired, or blocked by gap, cap or one-to-one), each
  eligible one with its map's `one_to_one`; WIDE_SLOW beside WIDE_SPLIT_G, paired by seed.

**Machines and run mechanics (fixed now).**
- **Allocation.** The 1660 runs the pilot, then ORACLE 850–851, SINGLE and WIDE_SLOW. The 3070 runs REG3 and
  WIDE_SPLIT_G.
- **Fallback.** If the 3070 is unavailable when stage 2 starts, every arm runs on the 1660 and nothing is dropped.
- **Runs per batch.** R = 1 for every arm on both cards. S75's default of 2 on 8 GB would overflow at L = 1536, and K9
  shows R = 2 gains nothing per run.
- **REG3's register mode, by rule.** The scan runs where the L = 1536 benchmark, including one evaluation, shows peak
  reserved memory plus context below the card's total. Otherwise the loop runs. On the numbers below that means the
  scan on the 3070 and the loop on the 1660.
- **CPU workers.** `--stats-workers` is fixed per machine from the benchmark's host memory before stage 1.

## Where each constant comes from

| constant | value | from (an allowed input) |
|---|---|---|
| S, V, block lengths | 4, 4, 6–14 | S75 (unchanged; registered before any S75 run) |
| NB, M, L | 32, 352, 1536 | the ideal learners at B; the ORACLE predictions from S75's ORACLE per-bin values; first named in the pre-run note (5bc6c2e note l.116–118), before any learned-arm record was in git |
| model | BDH N = 256, D = 32, 3 layers, decay 0.95, conv width 4 | S75, unchanged |
| VALID bar | 0.9 | S74's registered bar, unchanged |
| pilot proceed rule | mean ≥ 0.912 and both ≥ 0.9 | S75's ORACLE seed spread (0.9114, rounded up); the stop rates under the ORACLE predictions; loosened in 114cd08: disclosed, not blind by git |
| SPLIT_THR | min(pilot) − 0.03 | S75's rule; the pilot's ORACLE |
| split trigger | every 2400 from 2400 (the reference, never eligible) to 21,600; eligible from 4800; rise 0.02; ≤ 3 fires, ≥ 4800 apart; 64-sequence probe | S75 (explore_i_common), unchanged |
| the guard | no fire on a one-to-one map, with the precedence above | the pre-run note offered a guard or a spare channel (5bc6c2e l.119, before any learned-arm record was in git); 1e31409 chose the guard (on a tree without the learned arms; blind on the author's statement); its precedence was written in 114cd08, not blind by git |
| updates, B, lr, evaluation, probe | as S75 | S75, unchanged |
| arms, outcomes, reading | as S75, plus the guard; WIDE_SLOW printed | S75's registered design |
| run mechanics | R = 1; register by rule; arm → machine as above; workers per machine | fixed by rule in this spec |
| task CHECK | `check_task` at n = 4096 | its default; S75's n = 512 fails at B on a tolerance (verified) |
| seeds | pilot 840–841; screen 850–859 | fresh, within Session I's 800–859 |

S75's WIDE_SPLIT, REG3 and WIDE_SLOW records feed nothing in this table, on the author's statement. The referee found
no constant derived from them; the rows written in 114cd08 are not blind by git (Appendix B.1). If the user changes S75b after reading S75's
descriptions, that change is not blind, and S75b's record must say so.

## Cost

**Method.**
- *Time.* A CPU profile of one update at L = 1536, scaled by the factor by which each card's K9 at L = 768 exceeded the
  profile's estimate: 1.10–1.25 on the 1660. The 3070's K9 measures 0.77–0.79× the 1660's time per update. REG3's time
  is K9's L = 768 measurement (0.0939 s on the 1660, 0.0740 s on the 3070) × 3.5–4; the profiler ran out of CPU memory
  on REG3. That factor has no source (the other arms' measured ratios are 3.24–3.35); the L = 1536 benchmark measures
  REG3's time on each card (Appendix C).
- *GPU memory.* Live tensor bytes per run at R = 1, plus the gate tensor (32 × 1 × T × T float32) that an evaluation
  leaves on a gated run's view: 0.07 GB at L = 768, 0.28 GB at L = 1536. That tensor is why S75's ORACLE records peak at
  0.69 GB against K9's update-only 0.62.
- *Usable memory.* The 1660 has 5.61 GB in total and about 5.5 usable. The 3070 has 7.71 GB in total and about 7.0
  usable, since its display holds about 0.75 GB.

| arm | GPU peak GB (×1.15) | 1660 s / update, h per run | fits the 1660 | 3070 h per run |
|---|---|---|---|---|
| ORACLE | 2.19 (2.5) | 0.22, 1.5 | yes | about 1.15 |
| SINGLE | 1.63 (1.9) | 0.20, 1.3 | yes | about 1.0 |
| WIDE_SPLIT_G, WIDE_SLOW | 3.33 (3.8) | 0.24, 1.6 | yes | about 1.25 |
| REG3, scan | 4.80 (5.5) | 0.33–0.38, 2.2–2.5 | no (5.5–5.8 with context) | 1.7–2.0 (fits) |
| REG3, loop | 3.96 (4.6) | 0.55–0.8, 3.7–5.3 (about 27.6k extra kernel launches) | yes | – |

**CPU side.** At L = 1536 the routing statistics take about 21 s and 4.3 GB of peak RAM per 64-sequence chunk, against
2 s and 1.4 GB at L = 768.
- Each evaluation runs them on 256 sequences per gate layer in a stats worker: about 1.4 min per layer, about 4 min for
  REG3's three.
- Each split check runs them on 64 sequences in the main process.
- With 2–3 workers overlapping, host peaks of about 9–13 GB are possible, and neither machine's RAM is recorded. That
  range is an unsourced extrapolation. The benchmark records host memory, and `--stats-workers` is set from it.

**Totals.**
- Stage 1: about 3 h on the 1660.
- Stage 2 on the 3070: REG3 (scan) about 17–20 h plus WIDE_SPLIT_G about 12.5 h, so about 30–32 h.
- Stage 2 on the 1660: ORACLE, SINGLE and WIDE_SLOW, about 32 h.
- Wall time with both cards: about 32–33 h (1.4 days).
- Everything on the 1660 (the fallback, with REG3 on the loop): about 85–101 h, without the 3 h pilot (88–105 h with
  it).
- For scale, K9 projected S75's core arms at 16.1 h on the 1660 (training only). Per run, S75b costs about 3.3× that,
  and REG3 3.5–4×.

## Code S75b needs (not written; written only after the user approves this draft)

- **A new driver, `explore_gpu/s75b.py`.** It must not change any module S75's code imports, so that S75's SHA stays
  5fb120a92d69.
  - It holds configuration B, the two stages with their rules, the seeds, the machine allocation, R = 1 and
    `--register`.
  - It holds the guarded trigger, a transcription of `explore_i_common.make_split`. The trigger is installed by
    assigning `ic.make_split` in the S75b process, because `run_batched_icmc` looks it up at run time (run.py:514), so
    neither run.py nor explore_i_common is edited. It calls `ic.logits_chunked`, `ic.model_gates`, `ic.routing_layer`
    and `ic.split_op` through the module, so the CUDA patches in eval_scope still apply.
- **CHECKs before any run.**
  - The task CHECK at B, `check_task` at n = 4096.
  - The guard's mechanics, with threshold 1.01 and rise 1.0:
    - an eligible check on a one-to-one map is blocked as `one-to-one` and leaves n = 0 and last = None;
    - a run blocked at 4800 whose map is merged at 7200 fires at 7200;
    - cap and gap rows match `make_split`'s on the same map sequence.
  - A lock-step (Judge) of batched against sequential at L = 1536, for each S75b arm and for REG3 in every register mode
    it may run in, 20 updates. The CUDA CHECKs validated L = 768 only.
  - A K9-style benchmark at L = 1536 on each card: each arm at R = 1, REG3 in both modes. It runs one full evaluation
    and one split check before reading peak device memory and host memory.

## Order

1. Write the driver and its CPU CHECKs, then commit.
2. Run the CUDA CHECKs and the benchmark at L = 1536 on both cards; fix `--register` and `--stats-workers` by their
   rules, then commit.
3. Run stage 1 (the pilot) and commit its record.
4. Apply the proceed rule: stop and report, or run stage 2.
5. Write the report, counts table first.

## Appendix (2026-10-13): changes after approval at 114cd08, NOT BLIND

This spec was approved at 114cd08. Everything below was decided while writing the driver (`explore_gpu/s75b.py`) and
after reviewing it, which was after S75's learned-arm records had been read, **so none of it is blind**. None of it
changes a design constant: configuration B, the seeds, the arms, the outcomes, the bars, the proceed rule, SPLIT_THR's
rule and the reading are as approved above. These items are the first six of the driver's `CHANGES_SINCE_SPEC`. It is written into each
card's benchmark file, the stage-1 record, stage 2's meta and the report header.

1. **The pilot's card.** The pilot may run on the RTX 3070 if it is free before the GTX 1660 (the user, 12 October). The
   stage-1 record names the card.
2. **Host memory.** The routing margin runs in chunks of 16 sequences instead of 64. CHECK B3 shows the same value up
   to float64 summation order, a relative difference of 2.8e-16. The evaluation statistics' worker pool is the driver's
   own StatsPool.
3. **The `--stats-workers` formula is the driver's.** The spec says only "from the benchmark's host memory".
   - **What the benchmark measures:** each process's own peak RSS (VmHWM) for the main process, one statistics worker
     and the data process; the tensors one evaluation hands to the statistics; and the seconds per update and per
     evaluation's statistics.
   - **The budget** is the smaller of two figures:
     - min(the machine's RAM, the user's figure; MemTotal) − the co-running job's measured peak (`--beside`) − 1.5 GB;
     - MemAvailable at the benchmark's start − the co-running job's possible regrowth to its peak − 0.5 GB.
   - **Workers:**
     1. The smallest *w* ≤ 3 that keeps up with the evaluations and fits.
     2. Else the largest *w* that fits. The evaluations it falls behind on are held in the main process and are counted.
     3. Else 0 (the statistics run in line), if that fits.
     4. Else the benchmark refuses.
4. **Records from two cards.** Records of one arm and seed from two cards are not resolved by the driver. Stage 1 and 2
   refuse; the report prints the duplicates and marks the reading incomplete.
5. **Order, enforced by the driver.**
   - Stage 1 needs this card's benchmark and CUDA CHECK output committed.
   - Stage 2 needs the stage-1 record, every results file, and this card's benchmark and CUDA CHECK output
     committed.
   - (Withdrawn in Appendix B.3.4: the pilot card's benchmark may be rerun after the stage-1 record. Each run record
     carries the benchmark it ran under.)
6. **The lock-step CHECK (B7)** runs at R = 1, as the runs do, with each seed alone, batch 8 on CUDA, and REG3 in both
   register modes on both cards. The R = 2, batch-32 lock-step does not fit either card at L = 1536: the batched graph
   and the float64 arbitration share the card. A control with W_g's batched gradient doubled must fail.

## Appendix B (2026-10-13, before the pilot): provenance as git shows it, and rules for failures. NOT BLIND

Referee V's review of this draft (`docs/review_2026-10/spec_review_s75b_draft.md`, 3f7d84b) found two gaps.
- **Provenance.** The draft states its blindness as fact, but git shows only what each tree held.
- **Outcomes.** Four outcomes have no rule.

Both are settled here before the pilot. The rules were written, reviewed adversarially in this session four times,
and revised after each review (B.4). Everything in this appendix was written with S75's learned-arm records in the tree and read, so **none of
it is blind**. None of it changes a design constant (configuration, seeds, arms, bars, the proceed rule, SPLIT_THR's
rule). The failure rules extend the outcomes, VALID and the reading to failed runs.

The driver carries the same text, in `SPEC_PROVENANCE`, in its docstring's FAILURES section, and in the last items of
`CHANGES_SINCE_SPEC`.
- SPEC_PROVENANCE is written into the benchmark files, the results files' meta, the stage-1 record, stage 2's meta and
  the report.
- CHANGES_SINCE_SPEC goes to the same places.
- Per-run records carry neither.

### B.1 Provenance labels

**Blind by git, as far as git can show:**
- **1e31409, the design.** Its parent is 5bc6c2e, whose tree holds only S75's ORACLE 830–831 and SINGLE 830–839 records.
- **What git cannot show.**
  - That the design was written blind is the author's statement. b649751, which holds the WIDE_SPLIT and REG3 records,
    was on the remote at 13:53:11 UTC, 23 minutes before 1e31409.
  - 1e31409 also adds `explore_gpu/s75_writeup.py`, the generator of the learned-arm descriptions. It is unchanged
    since and was later run on the learned-arm records. Git cannot show whether it was run on b649751's records before
    the commit.
- **Earlier evidence.** Configuration B itself was already named in the pre-run note (5bc6c2e, note l.116–118), 13 h
  before any learned-arm record reached git. WIDE_SPLIT 830–832 had already finished on the user's machine about 2
  minutes before that note (report_2.md); git holds no learned-arm record before b649751.

**Not blind by git.** These were made after the merge 142f47e (14:16), which brought b649751's records and
`s75_S4_summary.md` into the tree:
- **ee97443, the cost section.**
- **114cd08, the revision.** The same session committed the S75 write-up (03bffdf) 130 s later.
  - **Design changes:**
    - the proceed rule was loosened, from "both ≥ 0.912" to "mean ≥ 0.912 and each ≥ 0.9";
    - WIDE_SLOW was made unconditional.
  - **Corrections and operational choices:**
    - the bound 0.963 and the gap share;
    - the windowed learner and the fixed-reach case;
    - the stop wording and the guard's precedence;
    - the allocation, R = 1, the register rule and the workers;
    - n = 4096, the memory and the 3070's factor.
  - **The other rules in 15c5e4f..114cd08:**
    - ORACLE_PILOT as its own arm, never substituting for ORACLE 850–851;
    - every arm runs all its seeds;
    - not VALID means descriptions only, never re-read, and the pilot is never pooled;
    - the successor rule;
    - the trigger's transcription and precedence;
    - the CHECK list and the Order.

  The review judged neither design change to be a function of a learned-arm quantity:
  - the proceed rule follows from ORACLE-only inputs;
  - the pre-run note had already given WIDE_SLOW its role.

  Its literal-number check found no learned-arm number in any version, but that check does not catch rounded or
  transformed numbers. Neither change is blind by git.
- **Everything after 114cd08:** the driver, Appendix A and this appendix.

**This appendix's edits above it.**
- The heading of the Provenance section; its "What git can and cannot show", "Later edits" and "The revision"
  bullets.
- The qualifier after the constants table.
- Appendix A's pointer to `CHANGES_SINCE_SPEC`, and its item 5's last line (withdrawn).
- "The design commit 1e31409" in "Why this rule".
- The stop rule's pointer to B.3.
- The guard's row of the constants table.
- report_2.md's closing paragraph, corrected the same way.

The stage-1 record labels 1e31409 "built on a tree without the learned arms (git); blind on the author's statement
only".

### B.2 Failed runs

**Two kinds of failure:**
- **Diverged.** A run whose evaluation logits are not all finite. The driver checks every evaluation, and the split
  trigger's probe, before anything else, on CPU and on CUDA. Without that check, a non-finite run fails first at the
  residual replay's assert, whose message names no NaN. The record's non-finite-loss check and a NaN or inf reaching a
  statistic also count.
  - A divergence is an **outcome**. It is never retried and never rerun, under any code, on any card.
  - It stays the seed's outcome even if a later run of that seed completed.
- **Operational.** Any other failure: out of memory, a crash, a driver error.
  - A process that ends without a record (killed, interrupted) leaves an open start marker. That counts as an
    operational failure of its invocation.

**What happens after an operational failure:**
- **The invocation stops** at the first operational failure that survives the driver's one retry. Only that seed is
  charged, not the seeds after it.
- **Rerun.** An operational failure is rerun on the same seed, never a fresh one. It runs on whichever card runs that
  arm, and the records keep each seed's card.
- **Halt.** A seed with operational failures in two invocations is **halted**.
  - The driver does not run it again until the user clears it (`--retry-halted`) or the code changes before stage 2.
  - A clear names exactly the invocations it clears. No rule compares times across machines (the server runs on UTC,
    the laptop on UTC−6). Every time S75b writes is UTC, marked with its zone.
  - A halt waits on the user. It is never a STOP, never final, and says nothing about B.

Every failure entry is kept, with its kind, error, invocation and time, and the report prints them for every code.

### B.3 The four uncovered outcomes

1. **A failed pilot run.**
   - **A value STOP** is decided as soon as a pilot run ends below 0.9 or diverges, whatever the other seed.
     - Wording: "B's pilot did not clear the proceed rule …: S = 4 is not usable at B" (for a divergence, "pilot run(s)
       … diverged … and so did not reach 0.9").
     - No other configuration is tried.
   - **An operational failure** is rerun on the same seed (`--stage 1` again). The stage-1 record says INCOMPLETE.
   - **A halted seed** leaves the decision to the user (HALTED).
   - **The record.** Stage 1 writes its record before the pilot's first run, marked running, and rewrites it when the
     invocation ends, however it ends short of a hard kill. Stage 2 refuses a record still marked running.
     - The record lists every stage-1 invocation under its code, from every card's results file (card, benchmark,
       workers, why it stopped), and each seed's card.
     - The pilot runs on one card. Stage 1's start is pushed (the results file and the running record), and a stage-1
       invocation refuses while another card's stage 1, as pushed, has not ended (One invocation per card, below).
       Two pilots started before either start was pushed are caught afterwards, as seeds with outcomes from two cards.
       A pilot may still be split across the cards, for example after one card fails.
     - Stage 2 reads the pilot from every card's committed results file.
2. **A code change.**
   - **Before stage 2 starts.**
     - The driver's code, every module in its code SHA, is frozen from stage 1.
     - A change that cannot be avoided (a bug that stops a stage) is committed with a dated note before any further run.
       The first stage 1 under the new code refuses without `--code-change-note PATH` naming that committed note, and
       records it. The note is needed whenever another code has a stage-1 record, current or archived, or any pilot
       entries, including a code whose stage 1 was killed before it wrote anything else.
     - The CUDA CHECKs, the benchmarks and the pilot (840–841) are then rerun under the new code.
     - **A value STOP under any code is final.** It is recomputed from every code's pilot records, not only from the
       record files.
     - Stage 2 needs every complete stage-1 decision, under every code, to be PROCEED. SPLIT_THR comes from the pilot
       under the code stage 2 runs.
     - Earlier stage-1 records are kept as `s75b_stage1_record_<sha>.*`, printed, and never pooled.
     - A superseded code is never used again, so a revert cannot re-adopt an earlier pilot. A code is superseded when
       its record was archived, or when it has pilot entries while the record is under another code. Since stage 1
       writes its record before its first run, a later code supersedes the earlier one as soon as its stage 1 starts,
       even if that stage 1 is then killed.
     - A code change re-draws ok pilot runs of 0.9 or more whose values have been seen. That is the price of the bug's
       fix; the dated note records why the change could not be avoided.
   - **After stage 2 starts.** Once stage 2 has a record or a failure under one code, the code is frozen for good.
     - Stage 1 and stage 2 refuse under any other code.
     - A change halts the screen for the user, and no stage-2 outcome is re-drawn.
     - The report reads only the stage-2 code; a second stage-2 code means no reading. The report also gives no
       reading after a value STOP under any code, or if the stage-2 code was superseded after stage 2 started.
     - The other machine sees the freeze once stage 2's results file is pushed. It is pushed as soon as stage 2 starts.
       If that push is rejected and the upstream holds a code change, the user is told at once: the change is reverted
       before the other machine runs any stage, and then the start is pushed.
   - **While a stage runs.** The data and statistics workers that each run spawns import the code from disk, so a pull
     or an edit during a stage would reach the runs. The driver compares the code on disk with the invocation's code
     before and after every run.
     - A change found before a run stops the invocation. Nothing is charged.
     - A change found after a run keeps that run's record, marked `code_changed`, and stops the invocation. The screen
       halts: both stages refuse, and the report gives no reading, until the user decides.
     - The decision is recorded by re-invoking the stage, on the card that ran those records and with the code on disk
       the invocation's again, with `--code-changed-decision keep` (the records count) or `rerun` (they are set aside,
       kept and printed, never counted, and their seeds are rerun). The report prints the decision.
     - So code is not pulled or edited while a stage runs.
3. **A stage-2 ORACLE run missing or failed** (and the other stage-2 arms).
   - **ORACLE.**
     - A diverged ORACLE 850 or 851 run counts as below 0.9: NOT VALID, so the reading is UNTESTED.
     - A halted ORACLE seed leaves validity incomplete until the user decides.
     - Any other missing ORACLE run is rerun.
   - **The pilot never substitutes for an ORACLE run**, as 114cd08 says (not blind by git).
   - **Learned arms.** A diverged run is not ROUTED, and its seed stays in n (n stays 10).
   - **SINGLE.** A diverged run gives no GAP for its seed.
   - **A halted read-arm seed** leaves the reading incomplete until the user decides. A halted SINGLE seed gives no GAP
     for it, and a halted WIDE_SLOW seed no paired row.
   - **A seed drawn on two cards**, with an outcome (an ok record or a divergence) from each, leaves the reading
     incomplete until the user decides which draw counts. Stages refuse meanwhile.
4. **The L = 1536 benchmark.**
   - **When a card is usable.** Usability is per arm. An arm runs on a card iff the card's latest benchmark under this
     code:
     - finished;
     - passed every case the arm uses (REG3 in the register mode the benchmark chose);
     - got workers from its host-memory rule.

     The pilot needs only the ORACLE case. A stage runs the arms its card allows and names the ones it refuses; a stage
     with no allowed arm refuses.
   - **Reruns.** A benchmark may be rerun under the same code at any time, even after stage 1, since it measures and
     decides no outcome. Each earlier file is kept in `bench_archive/`. A benchmark that crashes leaves its file "in
     progress", which counts as failed. Each run record carries the benchmark it ran under (commit and time) and its
     workers.
   - **Fallbacks.** If one card is unusable or unavailable, the other runs everything: `--machine fallback` on the
     1660, `--machine fallback-3070` on the 3070. Both need `--reason`, which is written into stage 2's meta and
     records.
     - If the other card's benchmark refuses only some of its arms, the fallback runs just those, with `--arms`
       (recorded).
     - A fallback never runs arms that the other card's unfinished invocation, as pushed, runs (One invocation per
       card, below).
   - **No usable card.**
     - If no card's benchmark allows the pilot, S75b cannot start.
     - If no card's benchmark allows a stage-2 arm, that arm cannot run. Without ORACLE, validity stays incomplete;
       without a read arm, the reading does. Without SINGLE there is no GAP, and without WIDE_SLOW no paired rows.
     - The report says which. No configuration, R or batch is changed to make a card fit; the user decides.

**Sync.** With commits required, stage 1 and stage 2 refuse unless:
- a `git fetch` succeeds, and the checkout is neither behind nor ahead of its upstream;
- every S75b JSON file in `explore_out/I/gpu/s75b/` is committed.

Stage 2 checks this before anything else, so a checkout that is behind is told to pull. The stage-1 record and the
results files are committed and pushed whatever the decision, so the finality rules on either machine see the other's
state. `--device cpu` is for the CHECKs only.

**One invocation per card.**
- **On one card.** A stage or a benchmark holds two locks on its card for its whole run, one in the checkout and one
  per machine, so a second invocation on that card refuses, from any checkout on the machine. The operating system
  releases the locks when the process ends, however it ends.
- **Across the cards.** Each results file records its card's last invocation, its arms, and whether it ended. Both
  stages push their start. A stage refuses while another card's last invocation, as pushed, has not ended and runs any
  of the same arms.
  - Stage 1's arm is the pilot, so the pilot runs on one card.
  - A killed process never records its end. Its card's results file is first committed and pushed from its own
    machine, because it holds the runs that finished before the kill. Only then does `--other-card-idle` state that no
    S75b invocation runs on that card. The statement is recorded, and that card's open markers then count as killed.
  - If that machine cannot be reached, the user decides: a fallback would draw its unpushed runs again.
- A seed with outcomes from two cards (an ok record or a divergence on each, for example after a start that was not
  pushed) is still caught: stages refuse and the report marks the reading incomplete.

### B.4 How the rules were checked

The first version of these rules (403538b) was reviewed adversarially. The review found real faults:
- **A real divergence was classed operational.** It failed at the replay assert, whose message names no NaN. Fixed by
  the finiteness check; CHECK B9 now poisons real runs, on CPU and on CUDA.
- **One environmental fault was charged to every seed after it.** Fixed: the invocation stops.
- **"Cannot run" was final, which locked out a code fix.** It is now a halt.
- **A pilot already below 0.9 could be re-drawn by a code change.** Fixed: a value STOP is decided early and
  recomputed from every code.
- **A code change could re-draw stage-2 outcomes.** Fixed: the freeze.
- **Killed processes went uncounted, and so did crashed benchmarks.** Fixed: start markers and progress files.
- **Finality read only local files.** Fixed: the sync check.
- **The text and the code disagreed in places.**

The revision (4437f75) was reviewed again. That review found:
- **A pilot split across the cards could never reach stage 2.** Fixed: stage 2 reads every committed results file.
- **A failed `git fetch` passed the sync check silently**, and unpushed commits were not refused. Fixed.
- **Halt clears compared times across machines.** Fixed: a clear names its invocations.
- **A reverted code could re-adopt an earlier pilot.** Fixed: superseded codes are refused.
- **A code change before stage 2 had no machine check.** Fixed: `--code-change-note`.
- **The report could read after a value STOP.** Fixed.
- **A REG3 failure on both cards blocked WIDE_SPLIT_G too.** Fixed: usability per arm.
- **A benchmark rerun after stage 1 blocked stage 2**, and a no-op stage 1 rewrote the record. Fixed: reruns are allowed,
  the record keeps every invocation, and each run record carries its benchmark.
- **Text and coverage gaps.**

The second revision (f4cf7a8) was reviewed a third time. That review found:
- **Two invocations on one card could overwrite each other's results file**, drawing seeds twice and losing records
  (high). Fixed: one invocation per card, by a lock.
- **A killed later pilot could be undone.** A code counted as superseded only once its record was archived, and the
  record was written only at the end of stage 1. Fixed: the record is written before the first run.
- **The code-change note was not required when the earlier code left only pilot entries.** Fixed.
- **A fallback could rerun a seed still running on the other card**, since that seed's open marker counted as failed.
  Fixed: each card records its last invocation, stages refuse overlapping arms until it has ended, and `--arms` and
  `--other-card-idle` are recorded.
- **A pull during a stage reached the runs unchecked**, through the workers each run spawns. Fixed: the code is compared
  before and after every run.
- **Four times were written in local time**, against this appendix's text. Fixed: UTC with its zone.
- **The report read a stage-2 code that a later pilot had superseded.** Fixed.
- **Smaller faults.**
  - Stage 2 checked the record before the sync.
  - The stage-2 summary after an abort was empty.
  - The stage-1 record left out killed invocations.
  - The text still said "the pilot results file" and "revised once".
- **Coverage.** 18 mutants of the driver's rules passed every CHECK. A CHECK that raised SystemExit ended the CHECK
  process and left the previous run's "ALL CHECKS PASSED" on disk. Fixed: a CHECK for each, a failing CHECK no longer
  ends the run, and the output file is rewritten at the start.

The third revision (5f6f840) was reviewed a fourth time. That review found:
- **A fallback with `--other-card-idle` for a card that died re-drew that card's unpushed runs** (high). Fixed in the
  procedure, the refusal text and this appendix: the dead card's results file is pushed from its own machine first, and
  if that machine cannot be reached, the user decides.
- **Stage 1 never pushed its start**, so the pilot's one-card rule was not enforced while a pilot ran. Fixed: stage 1
  asks for a push at its start, and the text says "as pushed".
- **The rejected-push procedure hid the freeze when the upstream held a code change.** Fixed: the user is told at
  once, and the change is reverted first.
- **A `code_changed` record had no recorded way out.** Fixed: `--code-changed-decision keep|rerun`.
- **The report's "no card allows" line said the reading was incomplete for any arm.** Fixed: per arm.
- **A seed diverged on one card and ok on the other was not caught as drawn twice.** Fixed: an outcome is an ok record
  or a divergence.
- **Smaller faults.**
  - The lock covered one checkout only; it now also holds per machine.
  - Stage 2 named a running record before a superseded code.
  - The report said "incomplete" after a value STOP under its own code.
  - Stage 1's refusal on a failed ORACLE case named a stage-2 fallback.
  - `--only` runs of the CHECKs printed "ALL CHECKS PASSED".
  - The README's commit lists left out archived files.
  - Coverage: the real `code_now`, stage 1's end mark and the real `dirty_code` had no CHECK.

CHECK B4 covers the decision rules. B5 covers the refusals, the run mechanics on a fake trainer, the third and fourth
reviews' points, the report on synthetic stores, and the sync check on temporary repositories. B9 covers a real
divergence.

Mutation passes (`explore_gpu/s75b_mutants.py`; results in `s75b_mutants_cpu_output.txt`). Each mutant breaks one rule of the driver and is run against B4 and B5. After the third revision, 35 mutants were run at 5f6f840: the third review's 18, which had passed every CHECK, and 17 that break the new rules. After the fourth revision, those and 13 more for its fixes, 48 in all, were run at 3cef609. Every mutant is caught in both passes, and the unmutated driver passes.

Not checked by a CHECK:
- `main()`'s argument wiring (`--arms`, `--other-card-idle`, `--code-changed-decision`, the benchmark's lock);
- the cards' real `git push`, and whether the operator pushes each start;
- a lock held by another process: the CHECK holds it in the same process;
- a hard kill: the CHECKs interrupt with KeyboardInterrupt and read the record during a run.

**What is not covered here.** The review's other points were left open for the user on 13 October. Appendix C (22
October) takes them all except a rule for runtimes over the estimate, which stays open. None of them changes what the
driver does at the pilot.

## Appendix C (2026-10-22, before the pilot): Referee V's review, item by item. NOT BLIND

Referee V reviewed this draft at 114cd08 in `docs/review_2026-10/spec_review_s75b_draft.md` (3f7d84b, on the main line;
read at 77221b8). On 13 October, Appendix B took the review's fix-list item 6 (the four uncovered outcomes) and parts of
items 1, 2, 7 and 14. This appendix takes every item of the fix list, in the review's order.

**How to read it.**
- "l.N" is a line of this file at 3cef609, the driver's commit after the fourth review. "114cd08 l.N" is a line the
  review cites. "s75b.py l.N" and "s75b_checks.py l.N" are lines of `explore_gpu/` at 3cef609.
- Where 3cef609 had already taken an item, the line is given. Where it is taken now, the edit is listed in C.7.
- Everything here was written with S75's learned-arm records in the tree and read, so **none of it is blind**.

**What does not change.**
- No constant changes: not configuration B, the seeds, the arms, the bars, the proceed rule, SPLIT_THR's rule, the
  trigger's constants, or the guard and its precedence.
- Nothing in the driver's code SHA changes. It is ee04da11a164 at this commit, as at 3cef609. So the CUDA CHECKs and
  the benchmarks run on 3cef609 stay valid, and no code-change note is needed.
- Two files are added outside the code SHA (the driver imports neither): `explore_gpu/s75b_rates.py` (items 4–5) and
  `explore_gpu/s75b_triggers.py` (items 9–10). Both describe and decide nothing.

### C.1 The four outcomes the pilot rules did not name (fix item 6)

The review asked which consequence follows each of four outcomes (§2, "Outcomes not covered" 1, 3, 4 and 6), and offered
alternatives: stop, retry the same seed, or a fresh seed; UNTESTED or a rerun. Each outcome gets exactly one name here,
and the name fixes what follows. Nothing is left to choose when it happens.

**A run whose evaluation logits are not finite is not a failed run under these names.** It is a value, DIVERGED (B.2,
l.371–376), which counts as below 0.9. So it falls under the rules that read values, which already name it:
- in the pilot, the stop rule: a value STOP, "S = 4 is not usable at B" (l.103–105, B.3 l.397–400);
- in stage 2, VALID: NOT VALID, so the reading is UNTESTED (B.3 l.451).

B.3's item 1 (l.396–402) lists three bullets under "A failed pilot run". Under the names here they are: the value STOP
(the stop rule, as above), PILOT RERUN, and PILOT RERUN's limit, HALTED (below). The rules are the same; only the
grouping differs.

| | the review's outcome | its one name | what follows, and only that | taken at 3cef609 | what the record or the report prints |
|---|---|---|---|---|---|
| 1 | a pilot run that fails: out of memory, a crash, a kill | **PILOT RERUN** | The same seed is run again: re-invoke `--stage 1`. Never a fresh seed. S75b does not stop on it. | B.2 l.377–390; B.3 l.401 | the stage-1 record: INCOMPLETE, "rerun --stage 1 (the same seeds)" (s75b.py l.947, l.1708); the seed's status: failed |
| 2 | the code SHA changes between the stages | **PILOT RERUN UNDER THE NEW CODE** | 840–841 are run again under the new code, never fresh seeds. First come a committed, dated note (`--code-change-note`), the CUDA CHECKs and the benchmarks under the new code. The old code is superseded. Its stage-1 decision stands beside the new one: a value STOP under it is final, and stage 2 needs both to PROCEED. | B.3 l.412–430 | stage 2 under the new code refuses, "both pilot records under this code SHA are needed" (s75b.py l.1781); the earlier record is kept as `s75b_stage1_record_<sha>.*` |
| 3 | a stage-2 ORACLE run (850 or 851) is missing or fails | **ORACLE RERUN** | The same seed is run again: re-invoke `--stage 2` on the card that runs ORACLE. VALID waits for it. Never the pilot in its place (l.88–90), never a fresh seed. | B.3 l.449–454 | VALIDITY "… -> incomplete" (s75b.py l.1978–1981); the stage-2 log, "an operational failure is rerun by re-invoking --stage 2" (s75b.py l.1850–1855) |
| 4 | the L = 1536 benchmark fails: a case fails, the run crashes, or no workers fit | **ARM REFUSED** | Per arm and per card: the arm does not run on that card. It runs on the other card if that card's benchmark allows it (`--machine fallback` or `fallback-3070`, with `--arms` and `--reason`). If neither card allows it, it does not run: for the pilot, S75b cannot start. No configuration, R or batch is changed. | B.3 l.461–485 | stage 2's log, "REFUSED by this card's benchmark" (s75b.py l.1813); stage 1, "this card's benchmark allows none of ['ORACLE_PILOT']" (s75b.py l.1397); the report, "BENCHMARKS", "ARMS NO CARD'S BENCHMARK ALLOWS", "S75b CANNOT START" (s75b.py l.1901) |

**The rerun's limit is not a fifth outcome.**
- A seed whose operational failures fall in two invocations is HALTED (B.2 l.386–390). The driver stops rerunning it,
  and the user decides: `--retry-halted` (recorded), or a code fix under PILOT RERUN UNDER THE NEW CODE.
- The halt applies to PILOT RERUN and ORACLE RERUN alike. It is never a STOP, and it says nothing about B.

### C.2 The two cases where two rules fire (fix item 10; review §3, points 1 and 2)

**1. Cap and gap together.**
- The row is labelled `cap`. `make_split` sets `blocked_by="cap" if n >= max else "gap"` (explore_i_common l.544), and
  the transcription has the same line (s75b.py l.641).
- The review counts this on 10 of S75's 39 blocked rows.
- The trigger's rule 2 now says so (C.7).

**2. Eligible, capped or gapped, and the map one-to-one.**
- Rule 2 comes first, so the row is labelled `cap` or `gap`. Its one-to-one state is kept in the row's `map_one_to_one`
  field (s75b.py l.636).
- Two counts follow, both printed:
  - The report's table prints "splits fired / blocked one-to-one" for WIDE_SPLIT_G (s75b.py l.1984, l.2007). "Blocked
    one-to-one" counts rule 3's blocks only.
  - Its listing, "WIDE_SPLIT_G's split checks (every eligible check)" (s75b.py l.2057), prints each eligible check with
    its map's `one_to_one` and its outcome. So the eligible checks on one-to-one maps that cap or gap blocked appear
    there, row by row.
- The review asked for a second total. `python -m explore_gpu.s75b_triggers --counts` prints it after stage 2, from the
  records, beside the fires on a map whose `c0` was live (C.3). It does this outside the driver: a total in the report
  would change the code SHA, and no rule's text needs it there.
- **The guard CHECK for both cases has been in CHECK B2 since the driver's first commit** (8cb8dce; s75b_checks.py
  l.156–214, the expected labels at l.182–183).
  - The check at 9600 is eligible, gapped and one-to-one, and is labelled `gap`.
  - The check at 21,600 is eligible, capped, gapped and one-to-one, and is labelled `cap`.
  - Every row's `map_one_to_one` is asserted as scripted.

### C.3 The guard: half of ROUTED's test (fix items 11 and 12; review §3, points 3 and 4)

**What the guard reads.** It reads `routing_layer`'s `one_to_one` alone (s75b.py l.632–642): the map from each source
to the channel with the highest mean gate at word positions (explore_i_common l.383–385).
- It reads it on the trigger's 64-sequence probe at the check's step, at the first gate layer. WIDE_SPLIT_G has one
  shared gate, so that is its only gate layer.
- It reads no η².
- ROUTED (l.147) needs η² ≥ 0.9 **and** a one-to-one map, at every gate layer, at the end, on 256 held-out sequences.
- So a one-to-one map with low η² blocks a fire. Such a map is a source-blind gate whose per-source argmaxes happen to
  differ.

**This is the rule as committed, and it stays.**
- The guard is "no fire on a one-to-one map". It was offered by the pre-run note (5bc6c2e l.119) and chosen at 1e31409.
  Its precedence was written in 114cd08.
- An η² condition, or reading the evaluation instead of the probe, would change the guard after S75's learned arms
  were read. Such a change would not be blind: the review records η² ≤ 0.0004 and no one-to-one map in S75's learned
  arms. It would also change a design constant of the approved spec, which this appendix does not do.
- The body's "the same test ROUTED uses" (l.134) is replaced by what the guard reads (C.7).
- The driver's docstring already says "routing_layer's own one_to_one", without the ROUTED claim (s75b.py l.36–39), so
  no code changes.

**Rule 4 fires when `c0` is live (item 12).**
- Rule 4 fires on any eligible check that is not capped or gapped and whose map is not one-to-one. That holds also when
  the least-used channel `c0` is in the map (`on_target` false), which overwrites a live channel (S75's declared hazard).
- This stays as committed. Blocking it would be a design change made after the learned arms were known: in S75, 30 of
  30 fires were on target. It would not be blind.
- Every row records `on_target` (s75b.py l.635), and `s75b_triggers --counts` totals the fires off target.

### C.4 The check schedule (fix item 13; review §3, point 5; 114cd08 l.119 and l.169)

**The draft said** "every 2400 updates from 4800" (l.127, l.177).

**The trigger in fact runs at every multiple of 2400 from 2400 to 21,600** (explore_i_common l.104 and l.529: SPLIT_REF
= 2400, and `last` is the run's updates − 2400).
- The check at 2400 is the reference. It records the probe SET as `prev` and is never eligible: its `prev` is unset, and
  eligibility needs step ≥ SPLIT_FIRST = 4800.
- Eligibility starts at 4800. S75's rows start at 2400.

**The transcription keeps the 2400 check.**
- s75b.py l.624 is `make_split`'s l.529.
- CHECK B2 shows the guarded trigger's rows equal to `make_split`'s on every check from 2400 when every map is merged
  (s75b_checks.py l.188–214).
- `s75b_triggers --check` replays the rule from 2400 and reproduces S75's pre-run note table.

**The trigger paragraph and the constants table's trigger row are corrected** (C.7).

### C.5 The two design changes after the merge: disclosed, not blind by git (fix item 14; review §4)

1. **The proceed rule was loosened.** 1e31409 l.59 had "both ≥ 0.912"; l.92–96 have "mean ≥ 0.912 and each ≥ 0.9".
2. **WIDE_SLOW was made unconditional.** ee97443 l.137 offered "without WIDE_SLOW … about 16 h"; l.125 and l.141 run it
   always.

Both came in 114cd08, after the merge 142f47e. Their label is **disclosed, not blind by git.**
- *Disclosed:* the spec states each change, its commit and its reason (l.31–34, l.97–101, B.1 l.326–352). This was
  already so at 3cef609.
- *Not blind by git:* 114cd08's tree held b649751's learned-arm records and `s75_S4_summary.md`.
- The review judged neither change a function of a learned-arm quantity:
  - the proceed rule follows from ORACLE-only inputs;
  - the pre-run note had already given WIDE_SLOW its role (5bc6c2e note l.102–104).

  That is the review's judgement, with the author's statement. Git cannot show it.
- The label is now written where each change appears: "The revision" (l.31–34), "Why this rule" (l.97), the WIDE_SLOW
  row of the stage-2 table (l.125), and the proceed-rule row of the constants table (l.175).
- The driver already lists both changes as not blind by git (`SPEC_PROVENANCE`, s75b.py l.246–251), so no code changes.

### C.6 The other items of the fix list, in order

1. **"Blind" as the author's word; the server's push times.**
   - Taken at 3cef609: the Provenance heading (l.12–13), the guard's row (l.178), and B.1 l.314–321 (b649751 on the
     remote at 13:53:11).
   - Taken now: 142f47e's push at 14:16:16 is added beside "14:16" (l.19).
2. **The revisions were made with the records in the tree; whose statement "no learned-arm file opened" is; 03bffdf.**
   - Taken at 3cef609: "Later edits" (l.20–30), and report_2.md corrected (B.1 l.363).
3. **B's first statement, before b649751.**
   - Taken at 3cef609 in B.1 l.322–324 ("Earlier evidence": 5bc6c2e note l.116–118).
   - Taken now in the constants table's NB, M, L row (l.172).
4. **0.912, rounded.**
   - Taken now. The unrounded spread is 0.89154 − 0.88010 = 0.01144, so 0.9 + spread = 0.9114. The rule's 0.912 rounds
     each value first.
   - The constant stays 0.912. The rule's first clause now says so (l.94).
5. **The stop and VALID rates over σ, σ's interval, and seed-specific means.**
   - Taken now, with `python -m explore_gpu.s75b_rates`. The model: two independent normal runs with seed sd σ. The
     review's figures reproduce to three decimals.
   - σ = spread/√2 = 0.00809, from one pair (1 df). Its 95% interval is 0.0036 to 0.258.

   | ORACLE prediction at B | σ | P(stop), 1e31409's rule | P(stop), the rule | P(stage-2 VALID) |
   |---|---|---|---|---|
   | per-bin, common mean 0.922 | 0.004 / 0.008 / 0.012 | 0.012 / 0.200 / 0.364 | 0.000 / 0.040 / 0.139 | 1.000 / 0.994 / 0.934 |
   | per-bin, seed means 0.928 / 0.916 | 0.004 / 0.008 / 0.012 | 0.159 / 0.324 / 0.427 | 0.000 / 0.051 / 0.163 | 1.000 / 0.977 / 0.900 |
   | fixed reach, common mean 0.914 | 0.004 / 0.008 / 0.012 | 0.522 / 0.642 / 0.679 | 0.240 / 0.366 / 0.433 | 1.000 / 0.921 / 0.771 |
   | fixed reach, seed means 0.920 / 0.909 | 0.004 / 0.008 / 0.012 | 0.779 / 0.702 / 0.700 | 0.190 / 0.353 / 0.434 | 0.988 / 0.864 / 0.736 |

   - At a true common mean of 0.905 (σ 0.008) the rule proceeds with probability 0.106 and VALID follows with 0.539. At
     0.910 these are 0.358 and 0.800.
   - "About 0.05 and 0.40" in "Why this rule" (l.99) are the per-bin and fixed-reach rows at σ 0.008: 0.040 and 0.366.
6. **The four outcomes.** C.1.
7. **The pilot on the 3070.**
   - Taken at 3cef609 in Appendix A item 1 (l.265–266) and B.3 l.403–410: the pilot may run on either card, on one card
     only, and the record names it.
   - In the plan of 22 October the pilot runs on the 1660, as the allocation (l.157) says.
8. **"Not usable" and S74's statistic.**
   - Taken now. The words "S74's usability bar" are dropped from the stop rule (l.105). The label "S = 4 is not usable at
     B" is set by this spec's rule alone: a pilot run below 0.9, or diverged. That is the driver's `stop_words`
     (s75b.py l.964–972), so no code changes.
   - S74's own statistic uses ORACLE's mean and a SINGLE half (`s75_s4.py` l.43, l.270–271). The report prints it for
     stage 2 as "S74's usability count" (s75b.py l.2039), a description.
   - So the review's example, a pilot of 0.915 and 0.898, is a value STOP labelled "not usable at B" by this rule,
     whatever S74's statistic says.
9. **A replay of the trigger on the pilot's curves.**
   - Taken now, as a description. `python -m explore_gpu.s75b_triggers --replay` replays S75's trigger rule at SPLIT_THR
     on the ORACLE_PILOT curves after stage 1, and on ORACLE 850–851 after stage 2. This is the pre-run note's hazard
     table for B. `--check` reproduces that note's table on S75's ORACLE curves.
   - It reads the 1024-sequence evaluation curve as a stand-in for the 64-sequence probe, and applies no guard. An
     ORACLE's map is one-to-one, so the guard would block every eligible check.
   - It decides nothing: SPLIT_THR is the rule's. As the review notes, proceeding selects high pilot values, so SPLIT_THR
     is biased upward.
10. **The two cases where two rules fire, and the guard CHECK.** C.2.
11. **"The same test ROUTED uses."** C.3.
12. **Rule 4 with `c0` live.** C.3.
13. **The 2400 check.** C.4.
14. **The two design changes.** C.5.
15. **The cost sources.**
    - Taken now. REG3's ×3.5–4 has no source: the profiler ran out of CPU memory on REG3, and the other arms' measured
      ratios are 3.24–3.35. The 9–13 GB host peak is an unsourced extrapolation (l.193–196, l.215–216).
    - The L = 1536 benchmark now measures both on each card: seconds per update per arm, and every process's host
      memory. `--stats-workers` is set from it (Appendix A item 3).
    - "85–101 h" excludes the 3 h pilot; with it, 88–105 h (l.223).
    - **Not taken: a rule for runtimes over the estimate** (review §2, outcome 6, second half). It would need a new
      constant, a factor. The benchmark decides usability by memory and workers, not by time; each run's time is
      recorded, and no rule reads it. It stays open for the user.

The review's §1 points 1–3 are fix items 1–3. Its point 4, the literal-number check, needs nothing.

**The review could not check some numbers, and they stay unchecked by a referee.** These are the ideal-learner numbers
(0.963, 0.927 / 0.939, 0.691, 0.517, the per-bin and fixed-reach transfers) and the CPU-profile figures. No run was made
for this appendix.

### C.7 This appendix's edits above it (text only)

- **Provenance:** 142f47e's push time (l.19); "disclosed, not blind by git" in "The revision" (l.31–34).
- **The pilot:**
  - the proceed rule's first clause: 0.912 rounded up from 0.9114 (l.94);
  - "Why this rule": the pointer to C.6 item 5, and the label (l.97–101);
  - the stop rule without "S74's usability bar" (l.105).
- **Stage 2:**
  - the WIDE_SLOW row's label (l.125);
  - the trigger paragraph: the 2400 reference check; `cap` when cap and gap both hold; what the guard reads; rule 4
    with `c0` live; the eligible checks on one-to-one maps blocked by cap or gap (l.127–138);
  - the printed list: each eligible check with its map's `one_to_one` (l.152).
- **The constants table:** the NB, M, L row (l.172), the proceed-rule row (l.175) and the trigger row (l.177).
- **Cost:** REG3's factor and the host peak marked unsourced (l.193–196, l.215–216); the totals with and without the
  pilot (l.223).
- **B.4:** "What is not covered here" points to this appendix.
