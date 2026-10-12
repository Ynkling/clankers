# S75b: S75 at S = 4 on a configuration where the bar is reachable (DRAFT spec, 2026-10-12, EXPLORATORY)

**Status: draft for the user.** Nothing here has run, and no S75b code exists yet. The spec follows the follow-up rule
committed in `explore_out/I/gpu/s75_S4_prerun_note_2026-10-12.md` (5bc6c2e):
- fresh seeds;
- every constant derived only from ORACLE and SINGLE quantities (S75's ORACLE 830–831 and SINGLE 830–839, the
  ideal-learner computations, and a new 2-seed ORACLE pilot of the exact configuration), never from S75's WIDE_SPLIT,
  REG3 or WIDE_SLOW outcomes;
- the configuration changed rather than the bar;
- the split guarded at k = S.

**Provenance: blind to S75's learned arms.**
- **The design.** The configuration, the predictions, the pilot rules, the guard, the arms and the seeds were committed
  at 1e31409. Its parent is 5bc6c2e, whose tree holds only the ORACLE 830–831 and SINGLE 830–839 records.
- **What git can and cannot show.** The learned-arm records (b649751) had been fetched into the local object store at
  14:10 UTC, before that commit. On the author's statement they were neither merged nor opened then; a parent shows
  only what the tree held. They were merged at 142f47e (14:16).
- **Later edits.** The cost section (ee97443) and the revisions from a blind review on 2026-10-12 were made after that
  merge, without opening S75's learned-arm files. They use only:
  - the two CUDA CHECKs' K9 benchmarks;
  - CPU profiles and memory measurements at L = 1536;
  - ideal-learner computations;
  - the 5bc6c2e ORACLE and SINGLE records.
- **The revision.** It changed one constant, the pilot proceed rule, from "both runs ≥ 0.912" to the rule below; the
  reason is given there. It made run decisions that had been open fixed, and it corrected numbers. The diff from
  1e31409 shows every change.

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
1. the mean of the two pilot runs is ≥ 0.912, which is 0.9 plus S75's ORACLE seed spread (0.892 − 0.880 = 0.012);
2. each pilot run is ≥ 0.9.

**Why this rule.** The blind commit had "both runs ≥ 0.912". The review showed that under the fixed-reach prediction
that rule stops the screen about 64% of the time, even though such a screen would usually be VALID. Under the per-bin
prediction it stops about 20% of the time. Under this rule the stop rates are about 0.05 (per-bin) and 0.40 (fixed
reach), with a seed sd of about 0.008. When the screen proceeds, both of its ORACLE runs are expected to clear 0.9 with
probability about 0.92–0.99.

**Stop rule.** If the pilot fails, S75b stops.
- The report says that B's pilot did not clear the proceed rule. It says "S = 4 is not usable at B" only if a pilot run
  ended below 0.9, S74's usability bar.
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
| WIDE_SLOW | 850–859 | WIDE_SPLIT_G without the split | printed, never read; runs unconditionally |

**The split trigger** is S75's, transcribed. It checks every 2400 updates from 4800 to 21,600 on the 64-sequence probe.
A check is eligible when probe SET < SPLIT_THR and the rise since the previous check is < 0.02. At most 3 fires, at least
4800 apart. The guard adds one rule; checks are decided in this order:
1. A check that is not eligible is recorded as in S75.
2. An eligible check that is capped or gapped is recorded as blocked by `cap` or `gap`, as in S75.
3. **An eligible check that is not capped or gapped, and whose source → channel map at word positions is one-to-one, is
   recorded as blocked by `one-to-one`.** The fire count and the last-fire step are unchanged; `prev` is updated as for
   every check. The test is `routing_layer`'s own `one_to_one` field, the same test ROUTED uses.
4. Otherwise the split fires.

Every row also records the map's η², margin and `one_to_one`. With four sources on four channels there is nothing to
separate, and a fire would overwrite a live channel: the hazard S75's note declared.

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
  layer; REG3's β; S74's usability count; every split check (fired, or blocked by gap, cap or one-to-one); WIDE_SLOW
  beside WIDE_SPLIT_G, paired by seed.

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
| NB, M, L | 32, 352, 1536 | the ideal learners at B; the ORACLE predictions from S75's ORACLE per-bin values |
| model | BDH N = 256, D = 32, 3 layers, decay 0.95, conv width 4 | S75, unchanged |
| VALID bar | 0.9 | S74's registered bar, unchanged |
| pilot proceed rule | mean ≥ 0.912 and both ≥ 0.9 | S75's ORACLE seed spread; the stop rates under the ORACLE predictions |
| SPLIT_THR | min(pilot) − 0.03 | S75's rule; the pilot's ORACLE |
| split trigger | every 2400 from 4800 to 21,600; rise 0.02; ≤ 3 fires, ≥ 4800 apart; 64-sequence probe | S75 (explore_i_common), unchanged |
| the guard | no fire on a one-to-one map, with the precedence above | the pre-run note's commitment (blind to the learned arms) |
| updates, B, lr, evaluation, probe | as S75 | S75, unchanged |
| arms, outcomes, reading | as S75, plus the guard; WIDE_SLOW printed | S75's registered design |
| run mechanics | R = 1; register by rule; arm → machine as above; workers per machine | fixed by rule in this spec |
| task CHECK | `check_task` at n = 4096 | its default; S75's n = 512 fails at B on a tolerance (verified) |
| seeds | pilot 840–841; screen 850–859 | fresh, within Session I's 800–859 |

S75's WIDE_SPLIT, REG3 and WIDE_SLOW records feed nothing in this table. If the user changes S75b after reading S75's
descriptions, that change is not blind, and S75b's record must say so.

## Cost

**Method.**
- *Time.* A CPU profile of one update at L = 1536, scaled by the factor by which each card's K9 at L = 768 exceeded the
  profile's estimate: 1.10–1.25 on the 1660. The 3070's K9 measures 0.77–0.79× the 1660's time per update. REG3's time
  is K9's L = 768 measurement (0.0939 s on the 1660, 0.0740 s on the 3070) × 3.5–4; the profiler ran out of CPU memory
  on REG3.
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
- With 2–3 workers overlapping, host peaks of about 9–13 GB are possible, and neither machine's RAM is recorded. The
  benchmark records host memory, and `--stats-workers` is set from it.

**Totals.**
- Stage 1: about 3 h on the 1660.
- Stage 2 on the 3070: REG3 (scan) about 17–20 h plus WIDE_SPLIT_G about 12.5 h, so about 30–32 h.
- Stage 2 on the 1660: ORACLE, SINGLE and WIDE_SLOW, about 32 h.
- Wall time with both cards: about 32–33 h (1.4 days).
- Everything on the 1660 (the fallback, with REG3 on the loop): about 85–101 h.
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
