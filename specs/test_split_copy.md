# test_split_copy.py — specification

Fixed before the test is written and before any run (2026-10-10). Both machines run it, each on its own seeds.

## The user's specification (verbatim)

> test_split_copy.py — Part C's configuration throughout: S=8, P=4, k=16, conv, 43200 updates, all
> eight streams from update 1, run_sc with D8 (test_window_gate's Part C/D path). Seeds X 560-569,
> L 1560-1569; validity oracles on the first two seeds per machine.
> - Part A, the split's ingredient: WIN3_SPLIT_D8 (test_window_gate's Part D arm, unchanged: the
>   KEYMASS trigger, the row copy with noise 0.1 std, W_g's Adam state zeroed, ≤ 3 splits ≥ 4800
>   apart), WIN3_RESET_D8 (its reset-only control, unchanged: the same trigger, Adam state zeroed, W_g
>   untouched; now on all ten seeds), WIN3_SLOW_D8 (no trigger). Outcome BOUND ROUTED with the
>   transition. Validity: CEIL_C (the perfect gate, Hebbian) on 560-561 / 1560-1561.
> - Part B, the premise: SINGLE_DELTA_D8 — k=1, Session F's explore_delta_mem.py copied unchanged
>   from claude/explore-F into the main line (β=1 fixed, L2-normalised keys, decay 0.95, tied write,
>   no gate), on the same path and budget (F's S68 arm: 0/10 at S=8 on seeds 300-309);
>   SINGLE_HEBB_D8 — k=1 Hebbian, the project's original single channel, same path. Outcome BOUND
>   (k=1 has no routing) with the transition. Validity: CEIL_DELTA_D8 — the delta channels (k=16,
>   β=1) under the perfect gate on 560-561 / 1560-1561 (F's bound 2/2 at 1200); if it does not
>   bind on a machine, Part B is UNTESTED there, not a negative.
> - Claims, pooled over both machines primary, per machine secondary, exact one-sided McNemar:
>   C1 WIN3_SPLIT_D8 beats WIN3_RESET_D8 (BOUND ROUTED).
>   C2 WIN3_RESET_D8 is not better than WIN3_SLOW_D8 by more than 2 discordant pairs pooled (a
>      bound, printed as such, no p-value).
>   C3 WIN3_SPLIT_D8 (BOUND ROUTED) beats SINGLE_DELTA_D8 (BOUND).
>   Printed, not claims: SINGLE_DELTA_D8 vs SINGLE_HEBB_D8; every split's update, c* → c0 and
>   whether the target is labelled ok (c* ≥ 2 streams, c0 none, on the labelled map before it);
>   the control's resets; CEIL_DELTA_D8's transitions; final accuracy of the k=1 arms by stream.
>   Bands with Wilson intervals for every arm.
> - Readings, fixed now: "the copy, not the reset" if C1 is SHOWN pooled and C2 HOLDS; "the
>   partition is necessary at eight streams against a single delta channel" if C3 is SHOWN pooled,
>   SINGLE_DELTA_D8 is below MAJORITY pooled and CEIL_DELTA_D8 bound on both machines.
> - CHECKs: the inherited chain; WIN3_SPLIT_D8 and WIN3_RESET_D8 at a Part D seed equal
>   test_window_gate's records bit for bit through 7200 (asserted on a matching CPU, printed
>   otherwise); CHECKs 142 and 143 rerun (the trigger's mechanics; threshold 0 equals WIN3_SLOW_D8);
>   the delta memory at β=0 equals the Hebbian model's logits to 1e-7 (F's CHECK) and SINGLE_DELTA_D8
>   at seed 300 equals F's S68 record (explore_out/F/premise_results.json) through 2400 on a
>   matching CPU, printed otherwise; fail_class_v2 agrees with v1 on every Part A record (no
>   two-stream records here, so any disagreement is a bug).
> - Runtime projection before the first run; nothing cut; results row in results/README.md; push.
>
> (Both machines, disjoint seeds, --machine X|L; --also pools. New tests use fail_class_v2. Long runs: detached
> process plus a watcher, resume from cached records, a runtime projection before the first run.)

## Background

- test_window_gate Part D (X 540-559, L 1540-1559; pooled): WIN3_SPLIT_D8 BOUND ROUTED 38/40; G4 (the split vs no
  trigger, WIN3_SLOW_D8) SHOWN 23 vs 0. The reset-only control WIN3_RESET_D8 ran on each machine's first five seeds
  (printed, not a claim): bound 0/5 on X, 3/5 on L. Every split's target was labelled ok.
- The split does two things at once: it copies W_g's row c* onto c0 (with noise) and zeroes W_g's Adam state. Part A
  separates them on paired seeds: the copy and the reset (WIN3_SPLIT_D8), the reset alone (WIN3_RESET_D8), neither
  (WIN3_SLOW_D8).
- Session F's S68 (claude/explore-F, EXPLORATORY): one delta-rule channel (β = 1, L2 keys, tied write, decay 0.95) on
  the grouped eight-stream task with conv, Adam 1e-3 throughout, 43200 updates, bound 0/10 on seeds 300-309; the
  perfect gate (k = 8) on the same delta memory bound 2/2 at 1200. Part B asks the same on the main line, on this
  test's seeds and path, paired with the window gate's split.

## Interpretation, fixed here before any code

**The path.** Every arm runs `test_stream_curriculum.run_sc` with D8's settings (`cur` False: all eight streams from
update 1), as test_window_gate's Part C and Part D do: `test_channel_binding.task_for(4, 8)` (S = 8, P = 4,
n_vals = 16, n_q = 1, L = 99), conv 'layer' K = 4, batch 32, 43200 updates, evaluation every 1200 on 2048
held-out sequences, run_one's early stop (three evaluations >= 0.95), run_sc's statistics.

**Part A** imports test_window_gate's arms and `one_run` unchanged: WIN3_SPLIT_D8, WIN3_RESET_D8 and WIN3_SLOW_D8
(LOCAL3 window gate + SLOW, no hinge; the trigger as in test_window_gate) and CEIL_C (SC8_ceil's perfect gate,
k = 16, no curriculum, Hebbian, Adam 1e-3 throughout). WIN3_RESET_D8 runs on all ten seeds.

**Part B.**
- `explore_delta_mem.py` is copied byte for byte from claude/explore-F at 9a6c2bf. F's S68 runs used its subclass
  `explore_f_delta_fast.FastDeltaMemory` (the same delta rule in a fused form, equal to the parallel form to 1e-5
  relative by F's CHECKs; impl "fast"), through `explore_f_delta_fast.to_delta_fast(m, seed=seed, beta=1.0)`.
  `explore_delta_mem.py` alone would not reproduce F's record, so `explore_f_delta_fast.py` is copied byte for byte
  from the same commit too. Nothing else from F is copied.
- SINGLE_DELTA_D8: test_short_conv's B_conv (k = 1, gate 'none', conv 'layer' K = 4) on task_for(4, 8), built by
  run_sc's builder, converted by `to_delta_fast(m, seed=seed, beta=1.0)`: β = 1 fixed, L2 keys, tied write, decay
  0.95, no gate; no parameter added. This is F's S68_SINGLE_S8 arm.
- SINGLE_HEBB_D8: the same k = 1 model, unconverted (Hebbian).
- CEIL_DELTA_D8: CEIL_C's model (the perfect gate, k = 16, ctx_pad 16) converted by `to_delta_fast(m, seed=seed,
  beta=1.0)`. F's oracle was k = 8 (one channel per stream); this one is k = 16 like CEIL_C, with eight channels
  unused.
- The recipe of all three is F's S68 recipe: Adam 1e-3 on every parameter throughout. SLOW does not apply (k = 1 has
  no gate; the perfect gate has no parameters), and F's record (CHECK 150) was made with this recipe.

**Outcomes.** Part A: BOUND ROUTED (`test_stream_recipe.outcome(r) == "BOUND ROUTED"`), reported with the transition.
Part B: BOUND (`test_conv_lr.bound_r`: a transition held to the end), with the transition. Failure classes of
unbound k = 16 runs: `fail_class_v2.outcome_v2` (diagnostic only).

**Validity** (test_window_gate's rule, VALID_N = 2): Part A is VALID on a machine if CEIL_C binds on both of its seeds;
Part B is VALID on a machine if CEIL_DELTA_D8 binds on both of its seeds. A claim is UNTESTED on a machine where a
part it uses is not VALID (Part B NOT VALID is UNTESTED, not a negative). C3 uses both parts.

**Seeds.** X 560-569, L 1560-1569, every arm; the perfect gates on the first two (560-561, 1560-1561). Disjoint from
every earlier main-line seed (test_window_gate's ranges included) and from the other machine's. Pairs are by seed.

## Claims (fixed now)

Exact one-sided McNemar on seed pairs (b = first arm only, c = second arm only), SHOWN if p < 0.05. Pooled (PRIMARY):
b and c summed over both machines' pairs; a pooled claim is UNTESTED unless its parts are VALID on both machines
(test_window_gate's pooling). Per machine (SECONDARY): the same tests on each machine's ten pairs.

- **C1** WIN3_SPLIT_D8 beats WIN3_RESET_D8 (BOUND ROUTED).
- **C2** WIN3_RESET_D8 is not better than WIN3_SLOW_D8 by more than 2 discordant pairs pooled: d = (RESET only) −
  (SLOW only) <= 2 HOLDS, else FAILS. A bound, printed as such, no p-value. Per machine the same d is printed.
- **C3** WIN3_SPLIT_D8 (BOUND ROUTED) beats SINGLE_DELTA_D8 (BOUND).

**Readings** (pooled only):
- "the copy, not the reset" if C1 is SHOWN pooled and C2 HOLDS (pooled).
- "the partition is necessary at eight streams against a single delta channel" if C3 is SHOWN pooled,
  SINGLE_DELTA_D8 is below MAJORITY pooled (bound on fewer than ceil(0.5 n) of the pooled n), and CEIL_DELTA_D8 bound
  on both machines (Part B VALID on X and on L).
- Otherwise the reading does not apply. Nothing else is read.

**Printed, not claims:** SINGLE_DELTA_D8 vs SINGLE_HEBB_D8 (BOUND; paired counts, one-sided p both ways, labelled
descriptive); every split's update, c* -> c0 and whether its target is labelled ok (c* holds >= 2 streams and c0 none
on the labelled map before it); the control's resets (updates); CEIL_DELTA_D8's and CEIL_C's transitions; the final
held-out accuracy of the k = 1 arms by stream; bands with Wilson 95% intervals for every arm, per machine and
pooled (RELIABLE >= ceil(0.9 n), MAJORITY >= ceil(0.5 n), MINORITY >= 1, else NEVER).

## CHECKs (all before any run unless stated; numbered from 145)

- **The inherited chain:** test_window_gate's verification (its CHECKs 135-144, and below them test_muon_recipe's,
  test_early_recipe's, ... down to test_multilayer_binding's), on this machine's own earlier results files.
  - One knob, needed on a host whose CPU differs from the one that wrote this machine's test_early_recipe file (this
    container is a Xeon @ 2.10GHz; X's file was written on a Xeon @ 2.80GHz; Muon's bf16 Newton-Schulz does not
    reproduce across them): test_muon_recipe's CHECK 130 is asserted when the file was written on this CPU and printed
    otherwise. The keyword `early_assert` defaults to True (as recorded); test_window_gate.verify passes it through.
  - test_window_gate's own CHECKs 135 and 141 compare with records written on a Xeon @ 2.10GHz; CHECK 138 and the
    Adam-path CHECKs below compare across 2.10/2.80GHz, where Adam runs have reproduced bit for bit.
- **145** the knob is inert: `test_muon_recipe.verify` and `test_window_gate.verify` at the knob's default have the
  same syntax tree as at the commit before the knob (the parameter and the branch it selects removed).
- **146** WIN3_SPLIT_D8 and WIN3_RESET_D8 at a Part D seed equal this machine's test_window_gate records bit for bit
  through 7200: curve, gradient norms, every statistic both records hold, and the trigger's rows. The seed is the first of
  the machine's Part D range on which both recorded arms fired by 7200 (X 540, L 1541). Asserted when this machine's
  CPU equals the records' CPU, printed otherwise.
- **147** CHECK 142 rerun: the trigger's mechanics (fires at 4800, 9600, 14400 when always eligible; blocked by the gap
  at 7200 and 12000 and by the cap after; W_g's Adam state zeroed; the split moves W_g's rows, the reset does not).
- **148** CHECK 143 rerun: with the threshold at 0 the two Part D arms equal WIN3_SLOW_D8 (seed 1) bit for bit through
  6000.
- **149** the delta memory at β = 0 (erase off, write 1, raw keys) gives the Hebbian model's logits to 1e-7 on
  SINGLE's k = 1 model and on CEIL_DELTA's perfect-gate k = 16 model, impl parallel and fast; impl "hebb" bit-identical
  (F's C2, at this test's models).
- **150** SINGLE_DELTA_D8 at seed 300 on this test's path equals F's S68_SINGLE_S8|300 (explore_out/F/
  premise_results.json at 9a6c2bf; written on a Xeon @ 2.10GHz) through 2400: curve, gradient norms and every statistic
  both records hold. Asserted on a matching CPU, printed otherwise.
- **151** fail_class_v2 against v1:
  - (a) before the runs: on every Part C/D record of this machine's test_window_gate file, `outcome_v2` equals
    `test_stream_recipe.outcome` (asserted).
  - (b) in the report: on every Part A record of this test, both labels are printed and compared.
  - The user's note says any disagreement is a bug. One fact recorded since: v1 and v2 also differ at eight streams
    wherever eta^2 by key >= 0.5 and margin >= 0.25 (recorded: test_recipe_scope's SC8_H s264 on X and L; see
    docs/fail_class_v2.md). So a disagreement in (b) prints the record's eta^2 by key and margin, which tells that case
    from a bug.
  - The CHECK reports PASS or FAIL. It does not stop the report (no claim uses the failure classes).
- **152** the copied modules equal F's files at 9a6c2bf byte for byte; F's fast_checks (fast = parallel in float64 and
  at the k = 16 size in float32) pass here.
- **153** the Part B models: SINGLE_DELTA_D8 is B_conv on task_for(4, 8) (k = 1, gate 'none', conv 'layer' K = 4) with a
  FastDeltaMemory (β = 1, L2 keys, tied write, decay 0.95), no parameter added; SINGLE_HEBB_D8's initial parameters are
  SINGLE_DELTA_D8's; CEIL_DELTA_D8's initial parameters are CEIL_C's and its gate is one-hot on each position's stream.
- **154** the Part B recipe: one Adam group at 1e-3 holding every trainable parameter at updates 1, 2400 and 2401,
  through each arm's real path (forward stubbed).
- **155** the seeds: every arm on the machine's range, the perfect gates on its first two; disjoint from every earlier
  main-line seed and from the other machine's range; with --also, the other file's machine, CPU and seeds.
- **156** a worker's run is bit-identical to the same run here (SINGLE_DELTA_D8 and WIN3_SPLIT_D8, seed 1, 1200 steps).
- **157** (a resume) records cached by an earlier start reproduce under this code through 2400 (the first cached
  WIN3_SLOW_D8 and SINGLE_DELTA_D8 records).

## Runtime

Projection before the first run, with pool-load timing: each arm's real path for 1200 steps as simultaneous copies on
the workers, the median. The worst case runs every learned-gate and k = 1 run to 43200; the projection is printed for
all runs and for the runs still to do (`longrun.projection`). Nothing is cut. Order: perfect gates first, then longest
first. Runs are cached in split_copy_results.json (gitignored, written atomically after every run); a restart resumes
from the cached records (`longrun.resume_meta`), and long runs go through `longrun.py`.

## Output

Per-seed raw values before any aggregate; validity; counts; the claims pooled (with `--also`, the other machine's
complete file) and per machine; bands; readings; the printed comparisons; diagnostics; curves. RESULT sections go in
the docstring, the JSON to results/<machine>/, and a row to results/README.md.
