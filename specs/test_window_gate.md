# test_window_gate.py — specification

Fixed before the test is written and before any run. Both machines run it, each on its own seeds.

## The user's specification (verbatim)

> test_window_gate.py (pull first; spec committed to specs/ before writing the test). Both
> machines, disjoint seeds (X 500-559, L 1500-1559). Gate: batch 1's LOCAL3 (width-3 causal
> conv of the embeddings, h = tanh(W_in u), g = softmax(W_g h); explore_local_gate.py at
> batch 16's 93bdfe6), no hinge. Optimizer: Adam 1e-3 with SLOW (other groups 1e-4 for
> 1-2400) — batch 16's LOCAL3_SLOW — and the plain single-rate variant as a second arm at S=2.
> - Part A (S=2, P=4, k=2, no conv, 24000; seeds 500-539 / 1500-1539): WIN3, WIN3_SLOW,
>   and the current recipe HINGE0 (SLOW + hinge) as the comparison. Outcome DISCOVERED.
> - Part B (S=4, P=4, k=16, conv, 28800; 540-559 / 1540-1559): WIN3_SLOW16 vs WIN16_A
>   (test_early_recipe's Adam + WINDOW). Outcome BOUND ROUTED.
> - Part C (S=8, P=4, k=16, conv, 43200; 540-559 / 1540-1559): WIN3_SLOW_D8 vs HINGE_D8
>   (test_slow_start's recipe). Outcome BOUND ROUTED, with the transition.
> Claims, pooled over both machines (primary; exact McNemar one-sided unless stated), then per
> machine: G1 WIN3_SLOW16 beats WIN16_A (Part B). G2 WIN3_SLOW_D8 beats HINGE_D8 (Part C).
> G3 WIN3_SLOW is not worse than HINGE0 by more than 2 discordant pairs pooled (Part A;
> printed as a bound, not a p-value). Bands with Wilson intervals for every arm. Validity: the
> perfect gate at each configuration on two seeds per machine. CHECKs: the inherited chain;
> WIN3_SLOW at seed 160 equals batch 16's LOCAL3_SLOW|160 bit for bit through 2400 on X's
> CPU; the gate cannot see token t-3; disjoint seeds and differing CPUs with --also. Runtime
> projection; never cut Parts B or C.

## Background (batch 16, EXPLORATORY, claude/outside-ideas; verdicts in explore_out/README.md at 092f355)

- S43 LOCAL3_SLOW (S=2, P=4, k=2, no conv; seeds 160-199): DISCOVERED 31/40 (STREAM-PARTIAL 5, KEY 4, no
  POSITION) vs S13's SLOW_HINGE 34/40 (5 vs 8, p = 0.58): inconclusive. LOCAL3 alone over 160-199: 33/40.
- S44(a) LOCAL3_SLOW16_A (S=4, P=4, k=16, conv; 240-259): BOUND ROUTED 20/20 vs X's HINGE4k16 15/20 (5 vs 0,
  p = 0.0625): "carries to four streams".
- S44(b) LOCAL3_SLOW_D8_A (S=8, P=4, k=16, conv; 260-269; 28800 steps): BOUND ROUTED 3/10 (transitions
  16800-21600) vs X's HINGE_D8 0/10 (3 vs 0, p = 0.25): "breaks the eight-stream stall".

## Interpretation, fixed here before any code

**The gate.** LOCAL3 is copied from `explore_b16_gates.py` at 93bdfe6 (`LocalBDH`, `window_init`, `to_local`),
which is batch 1's `explore_local_gate.LocalGateBDH` line for line: u_t = sum_{j<3} w_j v_{t-j} (depthwise, causal,
zero-padded; the window drawn from a generator seeded seed + 31337, uniform +-1/sqrt(3), no bias), h_t =
tanh(W_in u_t), g_t = softmax(W_g h_t), read gate = write gate. The run path's model is built as before and
converted in place; every existing parameter is unchanged, the window is added, W_h stays in the model unused,
frozen (requires_grad False) and in no optimizer group.

**Recipes (all Adam, lr 1e-3 = test_slow_start.LR; no Muon in this test).**
- WIN3: LOCAL3, onset_run's single Adam at 1e-3 throughout (batch 1's LOCAL3), no hinge.
- WIN3_SLOW, WIN3_SLOW16, WIN3_SLOW_D8: LOCAL3 + SLOW, no hinge (batch 16's LOCAL3_SLOW / LOCAL3_SLOW16_A /
  LOCAL3_SLOW_D8_A): two groups over the trainable parameters, the gate group (W_in, W_g, gate_conv.conv_w) at
  1e-3 throughout, every other trainable parameter at 1e-4 for updates 1-2400 and 1e-3 after (the switch right
  after the evaluation at 2400, as SLOW). test_slow_start's extra statistics are recorded (make_recipe with no
  hinge and no SLOW supplies them; this test's own groups and switch are composed on top, as test_early_recipe
  composed MUON). No existing module changes.
- HINGE0, HINGE_D8: test_slow_start's HINGE recipe (make_recipe(True, TAU)), recurrent gate, on its own run
  paths (part_run "0" and "C"); HINGE_D8 at 43200 steps instead of 28800.
- WIN16_A: test_early_recipe's arm, unchanged (ADAM + WINDOW, firing diagnostics), through its one_run.

**Run paths and seeds.** Part A: test_short_conv's arm A path (24000 steps). Part B: test_stream_recipe.run_attempt
with A4k16 (28800). Part C: test_stream_curriculum.run_sc with D8, all 8 streams from step 1, total 43200. X: Part A
500-539, Parts B and C 540-559. L: Part A 1500-1539, Parts B and C 1540-1559. A required flag --machine X|L.

**Outcomes.** Part A: DISCOVERED (bound and final VAL cos < 0.5; test_slow_start.success "0"). Parts B and C:
BOUND ROUTED = test_stream_recipe.outcome(record) == "BOUND ROUTED" (bound, one-to-one map, every stream >= 0.9);
the transition is printed for every bound run.

**Validity.** Per part, the perfect gate on the part's first two seeds of this machine (X 500-501, 540-541,
540-541; L 1500-1501, 1540-1541, 1540-1541), Adam 1e-3 single rate, on the part's run path: Part A
test_short_conv's perfect gate on Part A's run_one call; Part B test_stream_recipe's ceiling4k16; Part C
test_stream_curriculum's SC8_ceil with the curriculum off (cur False, the D8 path), 43200 steps. VALID if it binds
(transition not None) on both seeds; otherwise that part's claims are UNTESTED on that machine, and pooled if the
part is not valid on either machine.

**Claims (exact McNemar, one-sided, first arm higher; SHOWN if p < 0.05).**
- Pooled (primary): computed when --also gives the other machine's complete file (its meta names the other
  machine; every run its meta schedules is recorded and finished); b and c summed over both machines' pairs.
  - G1: WIN3_SLOW16 beats WIN16_A (BOUND ROUTED).
  - G2: WIN3_SLOW_D8 beats HINGE_D8 (BOUND ROUTED).
  - G3: d = (HINGE0 only) - (WIN3_SLOW only) on DISCOVERED, pooled pairs. HOLDS if d <= 2, FAILS if d > 2. A
    bound, printed without a p-value.
- Per machine (secondary): G1, G2, G3 on that machine's pairs with the same rules.
- Bands, per machine and pooled, every arm (WIN3, WIN3_SLOW, HINGE0, WIN3_SLOW16, WIN16_A, WIN3_SLOW_D8,
  HINGE_D8): RELIABLE >= 90%, MAJORITY >= 50%, MINORITY >= 1 run, NEVER 0 (test_stream_curriculum.band_n), with
  the Wilson 95% interval (test_stream_recipe.wilson).
- Printed, not claims: WIN3_SLOW vs WIN3 (McNemar, both directions); median transitions for every arm.

**Diagnostics (not part of the verdict).** Failure classes (k=2: test_router_layout.fail_class via
test_slow_start.tag; k=16: test_stream_recipe.outcome); merged runs; the stream -> channel map with distinct
channels at 4800, 9600, 19200 and the end (Parts B and C); WIN16_A's firings and hinge/task ratios
(test_early_recipe's); HINGE0's and HINGE_D8's hinge counts; X and L side by side with --also.

**CHECKs (after the inherited chain: test_muon_recipe's verify, through CHECK 134).**
- 135 WIN3_SLOW at seed 160 equals batch 16's LOCAL3_SLOW|160 (explore_out/window_recipe_results.json on
  claude/outside-ideas at 092f355; written on a Xeon @ 2.10GHz) bit for bit through 2400: curve, gradient norms
  and every statistic both records hold. Also WIN3 at seed 160 against batch 1's LOCAL3|160
  (explore_out/local_gate_results.json) through 2400. Asserted on machine X ("on X's CPU"); on L printed, not
  asserted.
- 136 The gate (Parts A, B and C models): at t it ignores token t-3 and sees token t-2; the conversion keeps every
  parameter of the run path's model bitwise and adds only the window, equal to batch 1's for the seed; W_h
  frozen; with the window at [1, 0, 0] and W_h = 0 the gate equals the recurrent gate with W_h = 0.
- 137 Groups and lrs through the real paths (forward and evaluation stubbed, 2401 updates): every WIN3_SLOW arm
  has two groups covering every trainable parameter once, gate group (W_in, W_g, gate_conv.conv_w), lrs
  [1e-3, 1e-4] at updates 1 and 2400, [1e-3, 1e-3] at 2401, W_h in no group and unchanged; WIN3 one group at
  1e-3 at 1, 2400 and 2401.
- 138 HINGE_D8 at seed 260 on this test's 43200-step Part C path equals this machine's recorded test_slow_start
  HINGE_D8|260 (--slow) bit for bit through 2400.
- 139 Seeds: this machine's seeds are its ranges, the perfect gates on each part's first two seeds; disjoint from
  the other machine's ranges and from every earlier main-line seed (test_muon_recipe's 400-479 and 1400-1479
  included); with --also: the other file's meta names the other machine, its seeds lie in the other machine's
  ranges, its CPU label differs and no seed overlaps.
- 140 A worker's run is bit-identical to the same run here (WIN3_SLOW_D8 seed 1 and WIN3 seed 1, 1200 steps).

**Runtime.** Projection with pool-load timing (each arm's real path for 1200 steps, as simultaneous copies on the
workers, the median per-step time; worst case = every run to its last step), printed. Nothing is cut: Parts B
and C are never cut, and no cut rule was given for Part A. Perfect gates first, then longest first; finished
runs cached; a resume keeps the stored schedule.

**Output.** Per-seed raw values first, then validity, counts, claims (pooled with --also, then per machine),
bands, diagnostics, curves. Results: window_gate_results.json (gitignored; each machine's copy in
results/X or results/L). A report-only mode (--report) recomputes the report from a finished file.

## Amendment of 8 October 2026 (the full run had started at 2f98f06: added as Part D)

### The user's amendment (verbatim)

> Amendment to test_window_gate (before any run, or as Part D if it has started):
> - Part C/D (S=8, P=4, k=16, conv, 28800; seeds 540-559 on X, 1540-1559 on L): add
>   WIN3_SPLIT_D8 = WIN3_SLOW_D8 + batch 17's W_SPLIT trigger (S37's KEYMASS; checks every 2400
>   from 4800 on the 64-sequence probe; fire if probe acc < 0.95 and rose < 0.02; at most 3
>   splits, >= 4800 apart; copy W_g[c*] to W_g[c0] + noise 0.1 std; reset W_g's Adam state).
> - Claims (pooled, primary; per machine secondary): G2 becomes WIN3_SPLIT_D8 beats HINGE_D8
>   (exact McNemar one-sided). Add G4: WIN3_SPLIT_D8 beats WIN3_SLOW_D8. Bands with Wilson for
>   both window arms; the reading "eight streams bind without labels or restarts" if
>   WIN3_SPLIT_D8 is RELIABLE pooled (>= 36/40) and G2 and G4 are SHOWN.
> - Diagnostics: every split's update and labelled target; maps at each check; a reset-only
>   control on 5 seeds per machine (the Adam-state reset at the same checks, no copy), printed
>   not claimed. Never cut the split arm.

The user's answers (AskUserQuestion, before any Part D code): Part D runs **43200** steps, matching Part C, so G2 and
G4 pair on equal budgets; the original G2 (WIN3_SLOW_D8 beats HINGE_D8) is **printed, not a claim**.

### Interpretation, fixed here before any Part D code

**Part D** (S=8, P=4, k=16, conv; Part C's path, run_sc with D8, 43200 steps; Part C's seeds; validity is Part C's
CEIL_C, same configuration and seeds):
- WIN3_SPLIT_D8: WIN3_SLOW_D8 + batch 17's W_SPLIT trigger (explore_window_d8_child.make_trigger at 992e320, which
  is S37's explore_split_target_child.make_trigger with the cap as a parameter; copied with S36's probe_data, split_op
  and with_trigger and S37's read_gate, select and targeting). Checks after the evaluations at 2400 (reference only),
  4800, 7200, ... through 40800 (total - 2400), after the run path's check and the lr switch, on the 64-sequence probe
  (generator seeded 12345): fire if the probe accuracy is < 0.95 and it rose < 0.02 since the previous check; at most
  3 splits per run, none within 4800 updates of the previous one (a check that meets the rule but is capped or gapped is
  logged as blocked, with the reason). The split: c* = the channel with the largest mean read-gate mass over the probe's
  key positions, c0 = the smallest; w = W_g[c*]; W_g[c0] = w + n2, W_g[c*] = w + n1, n1, n2 ~ N(0, (0.1 std(w))^2) from a
  generator seeded 12_000_000 + 1000 x seed + the split count; then W_g's Adam state (exp_avg, exp_avg_sq, step) zeroed
  in place. onset_run's Adam is built through a capturing factory (the same constructor and arguments).
- WIN3_RESET_D8 (the reset-only control): WIN3_SLOW_D8 + the same trigger rule on its own run (same probe, checks, cap
  3 and gap); where the rule fires, W_g's Adam state is zeroed and W_g is not touched (no copy, no noise). Seeds: Part
  C's first five (X 540-544, L 1540-1544). Printed, not claimed.
- Labelled target: a split is labelled ok if c* holds >= 2 streams and c0 none on the labelled map before it
  (test_scale_axes.routing_k on the run's probe batch; S37's targeting).

**Claims** (pooled primary, per machine secondary, as before):
- G1 unchanged (WIN3_SLOW16 beats WIN16_A).
- G2: WIN3_SPLIT_D8 beats HINGE_D8 (BOUND ROUTED, exact McNemar one-sided).
- G3 unchanged (the bound, Part A).
- G4: WIN3_SPLIT_D8 beats WIN3_SLOW_D8 (BOUND ROUTED, exact McNemar one-sided).
- Printed, not a claim: the original G2, WIN3_SLOW_D8 vs HINGE_D8 (McNemar); WIN3_RESET_D8 vs WIN3_SLOW_D8 on its
  five seeds per machine.
- Bands with Wilson for every arm, the two window arms of Part D included.
- Reading (pooled): "eight streams bind without labels or restarts" if WIN3_SPLIT_D8 is RELIABLE pooled (>= 36/40) and
  G2 and G4 are SHOWN (pooled); otherwise the reading does not apply (printed with what fails).
- Validity: Part D's claims (G2, G4) are UNTESTED where Part C is not valid (its CEIL_C).

**Diagnostics** (not part of the verdict): every split's update, c* and c0 and whether the target is labelled ok;
blocked checks and why; the labelled map at every check (and after each split); the reset-only control's resets.

**CHECKs** (added after 140):
- 141 WIN3_SPLIT_D8 at seed 263 equals batch 17's W_SPLIT|263 (explore_out/window_d8_results.json on
  claude/outside-ideas at 992e320; written on a Xeon @ 2.10GHz; its first split at 4800) bit for bit through 7200: the
  curve, gradient norms, every statistic both records hold, and the trigger's rows at 2400, 4800 and 7200 (the fields
  both hold). Asserted on machine X, printed on L.
- 142 The trigger's mechanics on a model after one Adam step, the trigger called at each check with no training in
  between and the threshold above 1 (every check from 4800 eligible): fires at 4800, 9600 and 14400, blocked by the gap
  at 7200 and 12000 and by the cap from 16800; each split leaves W_g's rows c* and c0 apart (noise) and zeroes W_g's Adam
  state; the reset-only variant at the same checks zeroes the state and leaves W_g unchanged.
- 143 With the threshold at 0 (never eligible), WIN3_SPLIT_D8 and WIN3_RESET_D8 at seed 1 equal WIN3_SLOW_D8 bit for
  bit through 6000 (the capture, the probe and the measurements are inert).
- 139 extended: WIN3_SPLIT_D8 on Part C's range; WIN3_RESET_D8 on its first five seeds.

**Runtime and the resume.** Never cut the split arm (nothing is cut). On X the run started at 2f98f06 finishes Parts A-C
untouched; the amended test then resumes the same results file at its own commit: the full verification again (with
141-143), the projection, and only Part D's runs (the cached Parts A-C records are kept). The meta keeps the first
start's commit and time and records the amendment's start (commit, time, arms). L runs the amended test from the start.
