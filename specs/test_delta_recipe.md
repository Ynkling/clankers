# test_delta_recipe.py — specification (DRAFT for the referee's review)

Fixed before the test is written (2026-10-10). Both machines run it, each on its own seeds. No run of an arm starts
until Session V's review of this file (docs/review_2026-10/spec_review_test_delta_recipe.md) has been read and every
ambiguity it finds is resolved by an amendment committed here, dated, before the first run.

## The user's specification (verbatim)

> test_delta_recipe.py — the memory rule at eight streams (pull first; spec committed to specs/
> before the test is written; both machines, disjoint seeds, X 580-599, L 1580-1599; --machine X|L;
> --also pools). Not started until V's spec review (docs/review_2026-10/spec_review_test_delta_recipe.md)
> has been read and any ambiguity it finds is resolved by an amendment committed before the first run.
>
> Background (exploratory, F's S76 on 1300-1319): LOCAL3 + SLOW + the exact-copy split on delta
> channels with β fixed at 1 and L2-normalised keys bound routed 9/10 against 10/10 Hebbian on the
> same seeds, sooner on 8 of 9; the delta perfect gate binds at 1200. S54(c)'s 3/10 used a learned β.
>
> Configuration: S=8, P=4, k=16, conv, 43200 updates, all eight streams from update 1, run_sc with D8
> (test_window_gate's Part C/D path), Adam 1e-3, SLOW. The split: H's exact-copy form (W_g[c0] :=
> W_g[c*], no noise, no Adam reset; KEYMASS trigger, checks every 2400 from 4800 on the fixed
> 64-sequence probe, fire if probe accuracy < 0.95 and rose < 0.02, ≤ 3 splits ≥ 4800 apart) — stated
> in full in the docstring with the probe's composition (Generator(12345); its per-stream query counts
> printed) and the body-key positions used for the mass. Memories: HEBB (the Hebbian write);
> DELTA (explore_delta_mem.py from claude/explore-F at 9a6c2bf, copied unchanged: β = 1 fixed, L2 keys,
> tied write, decay 0.95). Arms, 20 seeds per machine, paired by seed (same batches, same initial
> parameters where the two models share them): HEBB_SPLIT, DELTA_SPLIT, DELTA_SLOW (no trigger, for the
> split's effect on delta). Validity: CEIL_HEBB and CEIL_DELTA (the perfect gate, single-rate Adam 1e-3
> as in test_window_gate's CEIL_C) on 580-581 / 1580-1581; a part is UNTESTED, not a negative, if its
> ceiling does not bind both seeds.
> Outcome: BOUND ROUTED (bound; one-to-one stream-to-channel map by largest mean gate mass at value
> positions; every stream ≥ 0.9), with the transition; also printed: the overlap-based map (largest
> pairwise overlap of the streams' channel sets, from the referee's recount), MERGED counts, every
> split with its label.
> Claims, pooled primary, per machine secondary, exact one-sided McNemar on BOUND ROUTED:
>   D1 DELTA_SPLIT is not worse than HEBB_SPLIT by more than 2 discordant pairs pooled (a bound, no
>      p-value): d = (HEBB only) − (DELTA only); HOLDS if d ≤ 2.
>   D2 DELTA_SPLIT beats DELTA_SLOW (the split is an ingredient on delta channels too).
>   D3 DELTA_SPLIT's median transition is earlier than HEBB_SPLIT's on the seeds both bind (printed with
>      a sign test; secondary).
> Readings, fixed now: "the memory rule is open" if D1 HOLDS and D2 is SHOWN pooled (the two writes are
> interchangeable at eight streams under this recipe); "the Hebbian write stays" if D1 FAILS; "the
> split is Hebbian-specific" if D2 is NOT SHOWN while D1 HOLDS.
> CHECKs: the inherited chain; HEBB_SPLIT at a Part D seed equals this machine's test_window_gate
> record through the first split except for the split's noise and reset (state the first differing
> step), and equals H's COPY_NONOISE|487 bit for bit through 7200 on a matching CPU; DELTA at β = 0
> equals HEBB to 1e-7; DELTA_SPLIT|1310 equals F's S76 record through 2400 on a matching CPU; the probe's
> per-stream counts printed; with the threshold at 0 both split arms equal their no-trigger forms.
> Runtime projection before the first run (delta is about 1.5× Hebbian here); nothing cut; results row
> in results/README.md; push.

## Background (exploratory; numbers from the branches' own records)

- F's S76 (claude/explore-F, report 5 at 5b9bc2a, records explore_out/F/eight_results.json; Xeon @ 2.10GHz): Phase 2 on
  seeds 1310-1319, paired. HEBB_SPLIT BOUND ROUTED 10/10, DELTA_SPLIT 9/10 (1313 ended MERGED); Hebbian only 1, delta only
  0. On the 9 seeds both bound, delta bound sooner on 8 (median 14400 vs 24000). Phase 1 (1300-1303): the delta perfect
  gate (β 1, L2 keys) bound at 1200 on 4/4, the Hebbian one at 2400-6000.
- On delta, 6 of F's 12 splits fired on a target not labelled ok (c* held < 2 streams or c0 was not empty); every
  Hebbian split was labelled ok (22/22).
- S54(c) (delta with a learned β and S48's noisy split with an Adam reset): 3/10 against 9/10 Hebbian.
- Session H's S61 (claude/explore-H at 7e9197f, seeds 480-489): COPY_NONOISE (the exact copy, no reset) BOUND ROUTED 9/10,
  as SPLIT 9/10.
- test_split_copy (X, 560-569): the noisy split with the reset 10/10, the reset alone 3/10, no trigger 5/10.

## Interpretation, fixed here before any code

**The path.** test_window_gate's Part C/D path: `test_stream_curriculum.run_sc` with D8's settings (`cur` False), on
`task_for(4, 8)` (S = 8, P = 4, n_vals 16, n_q 1, L = 99), conv 'layer' K = 4, batch 32, 43200 updates, evaluation every
1200 on 2048 held-out sequences, run_one's early stop (three evaluations >= 0.95).

**The recipe of the learned-gate arms** (test_window_gate's WIN3_SLOW_D8, imported):
- The gate is LOCAL3 (test_window_gate.to_local).
- Adam 1e-3 on the gate group (W_in, W_g, the window); every other trainable parameter at 1e-4 for updates 1-2400 and
  1e-3 after the evaluation at 2400 (SLOW). W_h is frozen.
- No hinge.

**The memories.**
- HEBB: the model's own Hebbian memory.
- DELTA: `explore_delta_mem.py` with β = 1 fixed, L2-normalised keys, tied write, decay 0.95. It is applied after the
  LOCAL3 conversion (F's S76 order), through `explore_f_delta_fast.to_delta_fast(m, seed=seed, beta=1.0)`, F's fused
  evaluation of the same rule (equal to the parallel form to 1e-5 relative by F's CHECKs).
- Both files are already on the main line, byte for byte from 9a6c2bf (test_split_copy, CHECK 152). F's S76 records
  were made with the fused form, so CHECK 162 reproduces them only through it.
- The delta memory adds no parameter at β fixed. HEBB_SPLIT, DELTA_SPLIT and DELTA_SLOW at a seed therefore start from
  identical parameters and draw identical training batches (run_one's generator is seeded by the seed alone). CHECK 166
  shows both.

**The split** (H's exact-copy form; the docstring states it in full):
- The trigger is test_window_gate's (batch 17's KEYMASS rule), with H's operation in place of `split_op`.
- **Checks.** At updates 2400, 4800, ..., 40800 (2400 is the reference only; the first eligible check is 4800).
- **Probe.** The trigger evaluates the model on a fixed probe: `task.make_batch(64, torch.Generator().manual_seed(12345))`.
  That is 64 sequences, one query each, with per-stream query counts 5, 5, 6, 8, 10, 12, 5, 13 for streams 0-7 (computed
  before this spec; printed at every start).
- **Firing rule.** A check fires if the probe accuracy is < 0.95 and rose < 0.02 since the previous check; at most 3
  operations, >= 4800 updates apart.
- **Target.** c* (the busiest channel) and c0 (the idlest) are the argmax and argmin over channels of the read gate's mean
  mass. The mass is taken over the 64 probe sequences and the 32 body-key positions 3j + 1, j = 0..31, of the input
  (the query's key at 97 is not included). The read gate is computed in eval mode at τ_end.
- **Operation.** W_g[c0] := W_g[c*] exactly: no noise, c*'s row unchanged, W_g's Adam state untouched.
- **Label.** Each firing is labelled ok if, on the labelled stream -> channel map just before it (`test_scale_axes.routing_k`
  on the run's statistics probe, `probe_batch(task, seed)`), c* holds >= 2 streams and c0 none.

**Arms (20 seeds per machine).**
- HEBB_SPLIT: HEBB + the recipe + the split.
- DELTA_SPLIT: DELTA + the recipe + the split.
- DELTA_SLOW: DELTA + the recipe, no trigger.
- Validity, on the first two seeds:
  - CEIL_HEBB: test_window_gate's CEIL_C, the perfect gate (SC8_ceil, k = 16, ctx_pad 16, no curriculum).
  - CEIL_DELTA: the same model converted to DELTA.
  - Both use single-rate Adam 1e-3 on every parameter.
- The two VALIDITY parts:
  - HEBB is VALID on a machine if CEIL_HEBB binds both of its seeds.
  - DELTA is VALID if CEIL_DELTA does.
  - A claim is UNTESTED on a machine where a part it uses is not VALID (not a negative). D1 and D3 use both parts; D2
    uses DELTA.
  - A pooled claim is UNTESTED unless its parts are VALID on both machines.

**Outcome.** BOUND ROUTED = `test_stream_recipe.outcome(r) == "BOUND ROUTED"`: a transition held to the end, a
one-to-one stream -> channel map (each stream's channel = the argmax of its mean read-gate mass at the body's value
positions on the run's statistics probe `probe_batch(task, seed)`, `test_scale_axes.routing_k`), and every stream's held-out accuracy >= 0.9 at the end.
Reported with the transition. Printed, not claims:
- **The overlap.** Per run, the largest pairwise overlap max_{i<j} sum_c min(g_i[c], g_j[c]) of the streams'
  value-position gates (the referee's recount, docs/review_2026-10/recompute.py `overlap`, on the stored `stream_gate`).
- **The strict count.** "BOUND ROUTED and overlap <= 0.1" per arm.
- **MERGED counts and failure classes.** Unbound k = 16 runs, by `fail_class_v2.outcome_v2`.
- **Every firing.** Its update, c* -> c0, the label, and the map before and after.

## Claims (fixed now)

Seed pairs; b = first arm only, c = second arm only (BOUND ROUTED).

- **D1** DELTA_SPLIT is not worse than HEBB_SPLIT by more than 2 discordant pairs pooled.
  - d = (HEBB_SPLIT only) − (DELTA_SPLIT only) = c − b (first arm DELTA_SPLIT).
  - HOLDS if d <= 2, FAILS if d > 2.
  - A bound: no p-value. Per machine, the same d is printed.
- **D2** DELTA_SPLIT beats DELTA_SLOW: exact one-sided McNemar, SHOWN if p < 0.05, else NOT SHOWN.
- **D3** (secondary, printed). On the seeds where both DELTA_SPLIT and HEBB_SPLIT are BOUND ROUTED:
  - the median transition of each arm;
  - the per-seed signs of (HEBB transition − DELTA transition), ties dropped;
  - the exact one-sided sign test p for "DELTA earlier": P(X >= positives), X ~ Bin(positives + negatives, 1/2).
  - "Earlier" is stated as the medians and the sign test show it; it enters no reading.

Pooled (PRIMARY): b and c summed over both machines' pairs. Per machine (SECONDARY): the same on each machine's 20
pairs. Bands with Wilson 95% intervals for every arm, per machine and pooled (RELIABLE >= ceil(0.9 n), MAJORITY >=
ceil(0.5 n), MINORITY >= 1, else NEVER).

**Readings** (pooled only), with every combination named:

| D1 (pooled) | D2 (pooled) | reading |
|---|---|---|
| HOLDS | SHOWN | "the memory rule is open" (the two writes are interchangeable at eight streams under this recipe) |
| HOLDS | NOT SHOWN | "the split is Hebbian-specific" |
| FAILS | any | "the Hebbian write stays" |
| UNTESTED | any | no reading |
| HOLDS | UNTESTED | no reading (cannot occur: D1 needs DELTA VALID on both machines, which is all D2 needs) |

## CHECKs (all before any run of an arm; numbered from 158)

- **The inherited chain.** test_split_copy's verification on this machine's files: its CHECKs 145-157, and through it
  test_window_gate's 135-144 and everything below. test_muon_recipe's CHECK 130 is asserted when this machine's
  test_early_recipe file was written on this CPU and printed otherwise (test_split_copy's knob).
- **158** HEBB_SPLIT at a Part D seed vs this machine's test_window_gate record (WIN3_SPLIT_D8: the noisy split with the
  reset).
  - The seed is the first of the machine's Part D range on which the recorded run fired by 7200: X 540 (first split at
    4800), L 1541 (4800).
  - Through the first split's check, the curve, gradient norms, statistics and the trigger rows up to the firing (step,
    accuracy, previous accuracy, map before, c*, c0, fired) are equal.
  - After it they differ (the operation is not the same); the first differing evaluation is stated.
  - Asserted on the record's CPU, printed elsewhere (X's file was written on a Xeon @ 2.80GHz; Adam runs have
    reproduced across 2.10/2.80GHz, but the CPU differs).
- **159** HEBB_SPLIT|487 equals H's COPY_NONOISE|487 (claude/explore-H at 7e9197f, explore_out/H/copy2x2_results.json,
  Xeon @ 2.10GHz) bit for bit through 7200: curve, statistics, gradient norms, and the trigger rows including the first
  split at 4800. Asserted on a matching CPU, printed otherwise.
- **160** The operation: W_g[c0] equals W_g[c*] exactly after it; c*'s and every other row are unchanged; W_g's Adam
  state is untouched (H's check).
- **161** DELTA at β = 0 (erase off, write 1, raw keys) against HEBB on the same model. **OPEN**, see below.
- **162** DELTA_SPLIT|1310 equals F's S76 DELTA_SPLIT|1310 (explore_out/F/eight_results.json at 5b9bc2a, Xeon @ 2.10GHz)
  through 2400: curve, statistics, gradient norms. Asserted on a matching CPU, printed otherwise.
- **163** The probe: 64 sequences from Generator(12345); its per-stream query counts printed and equal to 5, 5, 6, 8, 10,
  12, 5, 13; the mass positions are the 32 body keys.
- **164** With the threshold at 0 (never eligible), both split arms equal their no-trigger forms bit for bit through 6000
  (seed 1):
  - HEBB_SPLIT equals test_window_gate's WIN3_SLOW_D8;
  - DELTA_SPLIT equals DELTA_SLOW;
  - the trigger checks are at 2400 and 4800, none fired.
- **165** The trigger's mechanics with the exact copy (CHECK 142's harness on Part C's LOCAL3 model, seed 260, always
  eligible): fires at 4800, 9600, 14400, blocked by the gap and the cap as in 142; after each firing W_g[c0] = W_g[c*]
  exactly; W_g's Adam state unchanged.
- **166** The pairing at a seed (580):
  - HEBB_SPLIT, DELTA_SPLIT and DELTA_SLOW have identical initial parameters (the delta memory adds none) and identical
    first three training batches;
  - CEIL_DELTA's initial parameters equal CEIL_HEBB's and its gate is one-hot on each position's stream;
  - DELTA's memory is a FastDeltaMemory with β 1, L2 keys, tied write, decay 0.95.
- **167** The recipe: through each arm's real path (forward stubbed) at updates 1, 2400 and 2401:
  - the learned-gate arms have two groups, the gate at 1e-3 and the rest at 1e-4 then 1e-3, with W_h in no group;
  - the perfect gates have one group at 1e-3.
- **168** The seeds: every arm on the machine's range (X 580-599, L 1580-1599), the perfect gates on its first two;
  disjoint from every earlier main-line seed (test_split_copy's included) and from the other machine's range. With
  --also: the other file's machine, CPU and seeds.
- **169** A worker's run is bit-identical to the same run here (DELTA_SPLIT and HEBB_SPLIT, seed 1, 1200 steps).
- **170** (a resume) Records cached by an earlier start reproduce under this code through 2400.
- **171** (in the report) fail_class_v2's outcome equals v1's on every record of this test. Any disagreement is printed
  with η² by key and margin; at eight streams the two definitions differ by design where both exceed their thresholds.

## Runtime

Projection before the first run, with pool-load timing: each arm's real path for 1200 steps as simultaneous copies on
the workers, the median. Printed for all runs and for the runs still to do (`longrun.projection`). Nothing is cut.
Perfect gates first, then longest first. Runs go through `longrun.py` and are cached in delta_recipe_results.json
(gitignored, written atomically after each run).

## Open points for the referee's review (found while writing this; to be settled by the amendment)

1. **CHECK 161's tolerance.** Measured before this spec (Xeon @ 2.10GHz), the delta memory at β = 0 against the
   Hebbian memory, as max |Δ logits| on 16 sequences:
   - the perfect-gate model (CEIL_DELTA's): exactly 0 (impl parallel and fast);
   - the LOCAL3 k = 16 model (the learned-gate arms'): 1.19e-7, 1.34e-7, 1.49e-7 at seeds 580, 581, 1310 (impl parallel
     and fast; impl "hebb" bit-identical).
   - The differences are float32 summation order: the delta form sums per channel, the Hebbian form multiplies by
     G = Σ g_r g_wᵀ first. F's own tolerance was 1e-5.
   - As written (1e-7) the CHECK fails on the learned-gate model. Proposed: 1e-7 on the perfect-gate model, 1e-6 on the
     LOCAL3 model, and "hebb" bit-identical on both.
2. **The delta cost.** Delta is not about 1.5× Hebbian here. F's S76 timings: 0.248 s/step for DELTA_SPLIT against
   0.079 for HEBB_SPLIT by pool-load timing (3.1×), and 0.26-0.38 s against 0.09-0.10 s per step in its recorded runs
   (about 3.4×); the delta perfect gate costs 3.2× the Hebbian one (test_split_copy: 284 vs 89 ms).
   - Projection from F's rates on 4 workers: about 25 h realistic (DELTA_SLOW's unbound runs dominate) and about 45 h
     worst case (every run to 43200).
3. **D3's seeds.** "The seeds both bind" is read as both BOUND ROUTED (the outcome), not merely bound.
4. **The probe's composition.** One query per sequence and 5-13 queries per stream. The trigger's accuracy is
   dominated by streams 7, 5 and 4, and a 64-query probe can read >= 0.95 while the held-out accuracy is lower
   (Session R). The trigger is kept as specified; the counts are printed.
5. **Power.** 40 pairs pooled. D1's margin of 2 discordant pairs is a bound on the difference, not a test.

## Amendment 1 (2026-10-10, after Referee V's review, before any run of an arm)

Source: docs/review_2026-10/spec_review_test_delta_recipe.md (claude/review-V at fad1489; copied to the main line
byte for byte with this amendment). The user chose two of the resolutions below: add HEBB_SLOW with a no-room rule,
and rename D1's readings with a printed interval rather than add seeds. Every other item takes the review's own fix.
Where this amendment and the draft above differ, this amendment holds.

**Arms.** Add **HEBB_SLOW**: HEBB + the recipe, no trigger (test_window_gate's WIN3_SLOW_D8 on these seeds), 20 seeds
per machine, part HEBB. The split's effect is then measured on both writes on the same seeds. Arms: HEBB_SPLIT,
HEBB_SLOW, DELTA_SPLIT, DELTA_SLOW, CEIL_HEBB, CEIL_DELTA.

**Claims.**
- **D1** (computed as before: d = HEBB_SPLIT only − DELTA_SPLIT only, pooled) is reported with one of three labels,
  none of them "interchangeable":
  - d <= −3: "delta binds more often";
  - −2 <= d <= 2: "delta not shown worse by more than 2 pairs";
  - d >= 3: "delta not shown non-inferior".
  - Printed beside d: the paired difference of BOUND ROUTED rates (DELTA_SPLIT − HEBB_SPLIT) with a 95% interval
    (Newcombe's hybrid score interval for paired proportions, 1998 method 10).
  - Also printed, quoted from the review: P(d <= 2) at 40 pairs with HEBB_SPLIT at 0.95 is 0.29 / 0.22
    (independent / nested outcomes) when DELTA_SPLIT is 0.85, and 0.11 / 0.05 when it is 0.80. The label is a bound,
    not a non-inferiority test.
  - Per machine: the rate difference and its interval only, no label.
- **D2** DELTA_SPLIT beats DELTA_SLOW (exact one-sided McNemar, SHOWN if p < 0.05).
- **D4** (new) HEBB_SPLIT beats HEBB_SLOW (exact one-sided McNemar, SHOWN if p < 0.05).
- **No-room rule.** If a write's no-trigger arm binds routed on >= 36 of the pooled 40 seeds, that write's split claim
  (D2 for delta, D4 for Hebbian) is "NO ROOM": printed, not read.
- **D3** unchanged (secondary; population: both split arms BOUND ROUTED).
- Validity: D1, D3 use HEBB and DELTA; D2 uses DELTA; D4 uses HEBB.

**Readings** (pooled only; each is printed as two parts):
- The memory rule: D1's label (above).
- The split, per write W (delta: D2, DELTA_SLOW; Hebbian: D4, HEBB_SLOW):
  - "helps" if its claim is SHOWN;
  - "no room" if W_SLOW binds routed on >= 36/40;
  - "not shown" otherwise.
- Named combinations of the two split labels:

  | delta | Hebbian | reading |
  |---|---|---|
  | helps | helps | "the split helps on both writes" |
  | not shown | helps | "the split is Hebbian-specific" |
  | helps | not shown | "the split helps on delta" |
  | not shown | not shown | "the split's effect is not shown on either write" |
  | no room on either write | | that write's label only |

- A claim that is UNTESTED (a part not VALID) gives no label.

**Outcome definitions.**
- The stream -> channel map (`test_scale_axes.routing_k`) takes, for each stream, the argmax of its mean read-gate mass
  at the body's value positions, on unrounded values (torch.argmax: ties go to the lowest channel index).
- The trigger's label uses the same function on the same probe (`probe_batch(task, seed)`) as the end outcome, at the
  check.
- The stored `stream_gate` is rounded to 4 decimals; the overlap is computed from it, as the referee's was.
- The overlap function is copied into the test. CHECK 172 shows it gives the referee's values: it compares with
  `overlap` in docs/review_2026-10/recompute.py at 0fdb363 (extracted from that commit and run) on every
  WIN3_SPLIT_D8 record of this machine's test_window_gate file.

**CHECKs** (changes and additions):
- **161** tolerance: 1e-7 on the perfect-gate model, 1e-6 on the LOCAL3 model (7x the measured 1.49e-7); impl "hebb"
  bit-identical on both.
- **173** (new, CPU-independent) the arms' own memory: at β = 1 with L2 keys, the fused form (impl "fast", which the arms
  run) against the parallel form, on the LOCAL3 and the perfect-gate models (seed 580, 16 training sequences). Logits to
  1e-5 relative and every parameter gradient to 1e-4 relative (F's tolerances). Asserted on every machine.
- **158** is asserted only where this machine's CPU equals the CPU that wrote the record. X's test_window_gate file was
  written on a Xeon @ 2.80GHz and this container is a Xeon @ 2.10GHz, so on X it is printed, not asserted. On L it is
  asserted if L runs this test on the i7 that wrote L's file.
- **164** also runs HEBB_SLOW: HEBB_SPLIT at threshold 0 = HEBB_SLOW = test_window_gate's WIN3_SLOW_D8 (seed 1, 6000).
- **174** (new, in the report) for every seed, each split arm's curve equals its no-trigger arm's curve on the same seed
  through the split arm's first firing (the whole run if it never fired). Pairs: DELTA_SPLIT with DELTA_SLOW,
  HEBB_SPLIT with HEBB_SLOW. Printed PASS or FAIL with the failing seeds; McNemar's pairing rests on it.
- **Probe-blind checks** (printed, in the report):
  - For every trigger check of every split arm: the probe accuracy beside the held-out accuracy at the same update.
  - The count of checks where the probe read >= 0.95 (so the rule could not fire) while the held-out accuracy was < 0.95.
- 172, 173 and 174 are numbered after 171.

**Runtime.** Unchanged rule. The review computes DELTA_SPLIT at 2.2x HEBB_SPLIT per step from F's recorded secs/iters;
the projection measures it before the first run and prints the ratio.
