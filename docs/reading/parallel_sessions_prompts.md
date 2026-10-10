# Prompts for three parallel exploratory sessions (F, G, H)

Common rules are in the first block; paste it at the top of each session's prompt, then the session's own block. Each session works on its own branch off `claude/outside-ideas`, its own seed block, its own output directory, and creates new files only. The one shared-code exception is Session F's `explore_delta_mem.py`, which F pushes before anything else and G later pulls.

---

## Common preamble (paste into all three)

```
You are joining the Ynkling/clankers research project (a fork of Pathway's BDH with k memory channels
and a learned gate; read README.md, docs/audit_2026-10.md, docs/reading/distant_cues_2026-10.md and
explore_out/README.md first, in that order, then the batch 16-18 explore files on claude/outside-ideas).
You are one of several exploratory sessions running in parallel. Rules, all fixed:

- Branch: create claude/explore-<SESSION> from origin/claude/outside-ideas. Pull outside-ideas before
  starting; never push to it or to the main line (claude/bdh-growth-hebbian-inference-w90069).
- Files: new files only, prefixed explore_<session>_*.py; never edit explore_common*.py, any test_*.py,
  or another session's files. Outputs under explore_out/<SESSION>/; verdict sections appended to
  explore_out/README.md under a heading "Session <SESSION>" only.
- Labels: everything is EXPLORATORY, not a result. 1 thread per run (torch.set_num_threads(1)). Record
  the CPU model and the code SHA in every run's cache key; retry runs with ok = False; never cut an arm a
  reading depends on. Pair bit for bit only within this container's CPU; against recorded runs from
  other CPUs, compare counts only.
- Seeds: use ONLY your assigned block. Main line uses 500-559 (X) and 1500-1559 (L); the side session
  (E) uses 160-299. Do not touch those.
- Readings: for every screen, write the readings (what count means what) into the file's docstring
  BEFORE any run, commit, then run. Outcomes and statistics exactly as the harness defines them
  (BOUND, DISCOVERED, BOUND ROUTED, failure classes KEY/POSITION/STREAM-PARTIAL/OTHER/MERGED, routing
  margin, η² thresholds 0.5; exact one-sided McNemar on paired seeds, Fisher otherwise, Wilson
  intervals with every band). Routing is measured at VAL positions as well as KEY positions
  (audit §3.2).
- Validity: every screen includes the perfect (oracle) gate on at least 2 seeds; if the oracle does
  not bind within budget the screen is UNTESTED, not a negative.
- Report: a single markdown report per batch (explore_out/<SESSION>/report_<n>.md) with: per arm the
  counts, bands, Wilson intervals, failure classes, transition times; the pre-fixed readings and which
  one applies; CHECK results; runtime; anything unexpected. The user relays this report by hand, so
  keep it under ~3 pages and put the counts table first.
- CHECKs: before any screen, reproduce one recorded run bit for bit on this CPU with your new code path
  disabled (e.g. β = 0 for a delta memory must equal the current Hebbian memory), and assert it.
- Do not edit the task definitions in test_binding_capacity.py; subclass (as explore_far_cue.HeaderTask
  does).
```

---

## Session F — delta-rule channels (branch claude/explore-F, seeds 300-359, output explore_out/F/)

```
Your subject: replace the Hebbian channel write by a (gated) delta-rule write, and measure what it
changes. Background: docs/reading/distant_cues_2026-10.md §4.1, §4.6; docs/reading/2412.06464.md
(Gated DeltaNet, eq. 10 and the code's α parameterisation); docs/reading/2501.12352.md §5.1-5.3.

TASK 0 (first, push within the first hours; other sessions depend on it):
explore_delta_mem.py — a drop-in channel memory in STATE form. Per channel c, per layer, state
S^c ∈ R^{d×n} (values × keys, matching BDH's ρ ∈ R^{n×d} transposed; d = 32, n = 256):
    write at t:  S^c ← S^c · ( α^c_t ( I − g^c_t β_t x̂_t x̂_tᵀ ) ) + g^c_t β_t v_t x̂_tᵀ
                 implemented as S^c ← α^c_t ( S^c − g^c_t β_t (S^c x̂_t) x̂_tᵀ ) + g^c_t β_t v_t x̂_tᵀ
    read at t:   o_t = Σ_c g^c_t S^c x̂_t      (strictly causal: read BEFORE the write at t, as tril(-1))
with x̂_t = x_t/‖x_t‖₂ when normalize=True, else x_t; v_t the write vector the current model writes;
α^c_t = decay (0.95) unless a decay module is supplied; β_t = σ(w_b·v_t + b_b) ∈ (0,1) per token
(init b_b so β ≈ 0.5), or a fixed β passed as a float. The gate g_t is whatever gate module is
supplied (oracle, recurrent, LOCAL3). Sequential over tokens is fine at L ≤ 99.
CHECKs (assert, both before anything else and in every batch): with β = 0, normalize=False, the
outputs equal the current parallel-score model's outputs to 1e-5 and a recorded run (batch 16
LOCAL3_SLOW seed 160, first 2400 updates) is reproduced bit for bit on this CPU; with the oracle gate
every cross-stream contribution is exactly zero; with β = 1, normalize=True, a single channel after
writing (k, v1) then (k, v2) returns v2 at key k with ‖S k̂ − v2‖ < 1e-5.
Commit and push explore_delta_mem.py + its CHECK script before S53.

S53 explore_f_delta_controls (seeds 300-309; P=4, S=2, n_vals=16, n_q=1; both layouts: grouped and
explore_far_cue.HeaderTask; 24000 updates, Adam 1e-3, eval every 1200, early stop after 3 ≥ 0.95):
 arms  SINGLE_HEBB (k=1, β=0, current memory)      SINGLE_DELTA (k=1, β=1 fixed, L2 keys)
       SINGLE_DELTA_L (k=1, learned β)             ORACLE_HEBB (perfect gate, k=S)
       ORACLE_DELTA (perfect gate, β=1, L2 keys)
 plus ORACLE_DELTA vs ORACLE_HEBB on the header layout at P=8, S=2 (seeds 300-302, budget 48000).
 Readings fixed now: "delta makes the single channel fail" if SINGLE_DELTA BOUND ≤ 1/10 on both
 layouts where SINGLE_HEBB ≥ 3/10 on at least one; "delta raises the oracle" if ORACLE_DELTA at P=8
 binds within 48000 on ≥ 2/3 seeds where ORACLE_HEBB does not; otherwise neither. Also report
 SINGLE_DELTA's final eval accuracy by stream (does it keep exactly the last-written stream?).

S54 explore_f_window_delta (seeds 310-349): the current best recipe on delta channels.
 (a) S=2, P=4, k=2, no conv, 24000, seeds 310-349: LOCAL3 + SLOW on DELTA (learned β, L2 keys) vs the
     recorded LOCAL3_SLOW on Hebbian (counts only, different CPU). Outcome DISCOVERED.
 (b) S=4, P=4, k=16, conv, 28800, seeds 310-329: LOCAL3 + SLOW + KEYMASS split on DELTA vs the same on
     HEBB run here (paired). Outcome BOUND ROUTED.
 (c) S=8, P=4, k=16, conv, 43200, seeds 310-319: same two arms. Outcome BOUND ROUTED with transition.
 (d) The randomised header layout (new subclass RandHeaderTask: block length P_b drawn per block from
     {2,3,4,5,6} with the same keys appearing once per stream across the sequence — define it exactly
     in the docstring, with CHECKs that every (stream, key) appears once and a single CTX per block),
     S=2, k=2, seeds 330-349, 24000: LOCAL3 + SLOW on DELTA and on HEBB, plus oracle 2 seeds each. This
     is the BASELINE Session G must beat; publish its counts in the report under that name.
 Readings: (a)-(c) "delta not worse" if DELTA ≥ HEBB − 2 discordant pairs (a bound, no p-value);
 "delta better" if ≥ 4 vs 0 paired. (d) descriptive: counts, failure classes (expect KEY splits).

S59 explore_f_decay (seeds 350-359; header layout P=8, S=2, k=2, oracle gate first, 48000):
 arms FIXED (0.95) vs ROUTED (α^c_t = exp(a_t g^c_t), a_t = −softplus(w·v_t)·exp(Δ), Δ init −2.6 so
 that g=1 gives 0.95; w init 0) vs GDN (α^c_t = exp(−A_c softplus(w_a^c·v_t + b_Δ^c)), A_c ~ U(0,16),
 b_Δ = softplus⁻¹(dt), dt log-uniform [0.001, 0.1], no weight decay on A, b_Δ), all on DELTA channels.
 Reading: "the horizon knob matters" if the oracle gap (oracle minus SINGLE_DELTA final accuracy)
 under ROUTED or GDN exceeds FIXED's by ≥ 0.15 on ≥ 7/10 seeds; then rerun the winning decay under
 S54(d)'s gate on seeds 330-339 (descriptive).

Order: TASK 0 → S53 → S54 → S59. Push after each. Report after S53 (short) and after S54+S59.
```

---

## Session G — distant cues (branch claude/explore-G, seeds 360-419, output explore_out/G/)

```
Your subject: make the header-layout cue reach the gate. Background: docs/reading/distant_cues_2026-10.md
§3, §4.2, §4.3, §4.5; docs/reading/2603.20997.md §5; docs/reading/1609.01704.md §5.2-5.3;
docs/reading/2605.10970.md §4-5; docs/reading/2203.11560.md §5. Session F is publishing
explore_delta_mem.py on claude/explore-F; pull it when it appears (git fetch origin claude/explore-F;
copy the file, do not merge the branch). Start with the parts that do not need it.

S55 explore_g_probe (seeds 360-369; no training of new models): on the recorded batch 2 / batch 16
header-layout runs you can reproduce here (far_L3 and LOCAL3_SLOW on HeaderTask, P=4, S=2) and on fresh
LOCAL3_SLOW runs on RandHeaderTask (define it exactly as Session F's S54(d) does — copy its definition
from their docstring once pushed, or write the same one), fit a logistic-regression probe (sklearn,
held-out 20%) from the residual stream entering layer ℓ (ℓ = 1, 2, 3) at K and at V positions to the
true stream, on 2000 probe sequences. Also probe the raw embedding and the final layer. Report
accuracy per (layer, position type) with Wilson intervals. Do NOT use cosine similarity between
stream-mates as a diagnostic. Readings: "the cue is present at layer ℓ" if held-out accuracy ≥ 0.9 at
both K and V; "absent" if ≤ 0.6 at every layer > 1 (then the fix is upstream and S56's register is
mandatory, not optional).

S56 explore_g_register (seeds 370-409; RandHeaderTask S=2, P_b ∈ {2..6}, k=2, no conv, 24000; then
S=2 fixed P=8 for a distance check):
 Register (delta form, differentiable):
    R_t = R_{t−1}(1 − β^R_t) + β^R_t W_R v_t,   β^R_t = σ(w_R·v_t + b_R),  b_R init logit(0.1), w_R small
    R_0 = 0; R_t ∈ R^{16}. Causal: the gate at t reads R_{t−1} (the register written by tokens < t) —
    state this and CHECK it.
 Gate: layer 1 keeps LOCAL3; layers 2-3 get g^{(ℓ)}_t = softmax(W_g tanh(A [ LN(r^{(ℓ)}_t) ; v_t ;
    R_{t−1} ])) with A initialised so that u ≈ W_in v_t at step 0 (the residual and register paths
    start as perturbations); shared W_g across layers 2-3. Variant NOSG: gradient flows from the gate
    into the stack; variant SG: stop-gradient on r^{(ℓ)} into the stack (audit §3.3 risk). Recipe:
    SLOW (gate groups 1e-3, rest 1e-4 for 1-2400), no hinge.
 Arms (seeds 370-389): REG_NUDGE (5% labelled nudge on the stream split, as explore_batch4's far_nudge,
    for validity), REG (no nudge, SG), REG_NOSG (no nudge), RESGATE_ONLY (per-layer residual gate
    without the register), and the BASELINE LOCAL3+SLOW counts from Session F's S54(d) (run them here
    too if F's are not yet published). Memory: Hebbian first; repeat REG and REG_NUDGE on DELTA
    channels (seeds 390-409) once explore_delta_mem.py is available. Oracle gate 2 seeds per
    configuration.
 Readings fixed now: "the register carries the cue" if REG_NUDGE BOUND ROUTED ≥ 9/10 (validity);
    "discovered" if REG (either variant) beats BASELINE by ≥ 3 bound runs with KEY splits ≤ 2;
    "architecture insufficient" if REG_NUDGE < 6/10 (then report β^R's values at CTX vs K/V tokens:
    is it the write decision or the read path?). Always report β^R at CTX, K and V positions
    (mean, by step), the routing margin at VAL positions, and the CTX/KEY/VAL × stream contingency
    table of g_t at layers 2 and 3.

S57 explore_g_latch (seeds 410-419) — run ONLY if S56's REG fails with β^R settling in 0.2-0.8 at CTX
 tokens (a leaky compromise): HM-RNN latch, u_t = w_z·v_t + b_z, z̃ = max(0, min(1, (a u + 1)/2)),
 z_t = 1[z̃ > 0.5] forward, straight-through backward with the hard-sigmoid derivative, r_t = z_t W_r v_t
 + (1 − z_t) r_{t−1}; a annealed linearly 1 → 5 over the last two thirds of the budget; b_z = 0, w_z
 small so |u| < 1 at init (dead-STE guard, CHECK it at step 0); optional firing-rate prior
 λ(mean_t z_t − 1/(2P̄+1))², λ = 0.1, as a second arm. Same arms and readings as S56's REG/REG_NUDGE.

Order: S55 → S56 (Hebbian) → S56 (delta, when available) → S57 if triggered. Push after each. One
report after S55 (one page), one after S56/S57.
```

---

## Session H — anti-collapse and recipe (branch claude/explore-H, seeds 420-479, output explore_out/H/)

```
Your subject: cheaper, continuous alternatives to the plateau-triggered KEYMASS split, and two
stability checks, on the EXISTING Hebbian code (no dependency on other sessions). Background:
docs/reading/distant_cues_2026-10.md §4.7; docs/reading/2607.25357.md §5.2 (Gumbel);
docs/reading/2502.13685.md §5.3 (Switch loss, Fig. 9 collapse); docs/reading/2201.00042.md §5.2
(kWTA); docs/reading/2510.04871.md §5.5 (stable-max, EMA).

S58 explore_h_collapse (seeds 420-459; S=4, P=4, k=16, conv, 28800; LOCAL3 gate, SLOW; outcome BOUND
 ROUTED and transition time; all arms paired on the same seeds and batches):
   SPLIT        the current KEYMASS split (reference, run here)
   RESET        the split's optimizer reset only, no row copy (the missing control, audit §2.3)
   GUMBEL_W     write gate g^w = softmax(z + n), n ~ Gumbel(0,1) i.i.d. per token and channel, training
                only, no annealing; read gate g^r = softmax(z) (noise on the write side only); no split
   GUMBEL_RW    noise on both write and read gates; no split
   SWITCH       no split; L_aux = 1e-3 · k · Σ_c f_c P_c with P_c = mean_t g_t[c], f_c = fraction of
                tokens with argmax g_t = c, over the batch
   NONE         neither split nor noise nor loss
 Readings fixed now: "noise replaces the split" if GUMBEL_W ≥ SPLIT − 1 discordant pairs and
 GUMBEL_W > NONE by ≥ 4 vs 0; "the split is the reset" if RESET ≥ SPLIT − 1 pair; "the balance loss
 helps" if SWITCH > NONE by ≥ 4 vs 0. Report merges (MERGED class) per arm and the time of the
 transition; also the fraction of runs still flat at 14400 that bind by 28800 (the plateau-vs-dead
 question from the routing paper).

S58b explore_h_eight (seeds 460-469; S=8, P=4, k=16, conv, 43200): the two best arms of S58 vs SPLIT
 at eight streams. Reading: "carries to eight streams" if the best arm ≥ SPLIT − 1 pair.

S60 explore_h_stability (seeds 470-479; S=2, P=4, k=2, no conv, 24000; LOCAL3 + SLOW):
   TEMP_FLOOR   gate softmax at temperature max(τ, 0.5) (logits scaled), vs τ = 1 reference
   STABLEMAX    stable-max in place of softmax on the gate logits (Prieto et al. 2501.04697)
   EMA_EVAL     evaluate with an EMA (0.999) of the weights, training unchanged (does the BOUND
                asymmetry — a run reaching budget counting on one final eval — shrink?)
 Descriptive: DISCOVERED counts, time to transition, how often the gate softmax saturates (max
 probability > 0.99 at > 90% of positions) before vs after the transition.

Order: S58 → S58b → S60. Push after each. One report after S58, one after S58b+S60.
```

---

## Session G — follow-up (S71-S73; seeds 700-759; branch claude/explore-G, output explore_out/G/)

Written 2026-10-10 after G's report 2 (S55-S57) and F's report 3 (S68-S70). The common rules above still apply.

```
Follow-up to your S55-S57. Three screens on RandHeaderTask-lite, which Session F has now published
with its baseline. Pull claude/explore-F (git fetch origin claude/explore-F; copy files, do not merge
the branch): copy explore_f_tasks.py unchanged, and re-copy explore_delta_mem.py if its SHA differs
from the one you have (then rerun your "β = 0 equals Hebbian" CHECK). Read explore_out/F/report_3.md
and rh_lite_tables.md.

Context since your round.
- RandHeaderTask-lite = explore_f_tasks.RandHeaderTask(P=4, nb=2, lmin=1, lmax=3), S=2, n_vals=16,
  n_q=1, L=23, k=2, no conv. Both perfect gates bind it by update 1200 on 5 of 6 seeds (3/3 each at
  24000). LOCAL3+SLOW does not: Hebbian DISCOVERED 0/20, BOUND 1/20, ROUTED*@end 0/20, KEY 19 (seeds
  330-349); delta β=1: 0/20, 6/20, 0/20, KEY 14. F's outcomes: DISCOVERED = bound and final VAL cos
  < 0.5; ROUTED*@end = margin ≥ 0.9 and η² by stream at KEY > 0.9 at the end, both from
  explore_f_tasks.rand_header_routing_stats.
- F's S68: one delta channel binds the grouped layout 2/10 at S=4 and 0/10 at S=8 (the gated Hebbian
  recipe 19/20, 9/10). S69: a single Hebbian channel does not learn the reset at CTX (0/10 under
  every decay). So the partition is what binds at four and eight streams; the forget gate needs
  the delta rule.

My reading of your round, which the screens below test.
(1) REG_NUDGE on Hebbian memory had a clean latch (β^R 0.96 at CTX) and perfect routing at layers
    2-3, and sat at 0.32 on BOTH streams in every run, so the far block's decay is not the cause.
    ORACLE23 0/3 against ORACLE_H's bound seed 371 is one paired seed. My spec kept layer 1 on
    LOCAL3; that choice is what S71 and S72 test.
(2) REG_D's "discovered" was by the letter of the rule (KEY splits counted on unbound runs), not
    its intent. The readings below use routed binding, so a key-split binding cannot count.
(3) S57's latch ran on the memory where routing did not pay; it gets its run on delta memory.
(4) The fixed P=8 layout let gates split by pair index (η² by index 0.6-0.74 in your baselines);
    the lite task removes most of that, and its Hebbian oracle is not at the edge.
(5) Your S55 probe becomes an in-run readout on every arm, so "does the stack carry the cue" is
    answered for the delta memory too (F's one-seed observation of stream-tagged keys at layer 3).

Common to S71-S73.
- Task: RandHeaderTask-lite exactly as F's S70; run F's check_rand_header as a CHECK. Budget 24000,
  Adam, k=2, no conv. Seeds 700-759 (if any appears in explore_out/ or results/, say so and shift the
  block by 100).
- Outcomes per run: BOUND (your rule); ROUTED*@end and DISCOVERED (F's, computed with
  rand_header_routing_stats so they pair with F's baseline); ROUTED-BOUND := BOUND ∧ ROUTED*@end.
  The readings use ROUTED-BOUND only; print your BOUND ROUTED beside it. Failure classes as before.
- In-run probe: at the final evaluation of every arm and seed, fit the S55 probe (logistic
  regression, held-out 20%, 2000 sequences) from the residual entering layers 2 and 3 and from the
  final layer, at K and at V positions, to the stream; save the accuracies in the record.
- Also save held-out accuracy by (query stream, block index of the target pair, pair index within
  its block) at the final evaluation; report it for every arm that sits on a plateau.
- Report as before: β^R (or the latch's z) at CTX/KEY/VAL by step, the routing margin at VAL per
  layer, the CTX/KEY/VAL × stream contingency of g_t at every gated layer.
- Statistics: paired by seed, exact one-sided McNemar on ROUTED-BOUND; "A beats B" means p < 0.05
  (at 20 seeds: 5 vs 0, 7 vs 1, 9 vs 2 or better). Wilson 95% and bands with every count. Validity:
  a family is VALID if its perfect gate binds ≥ 8/10 on seeds 700-709 (F's bound by 1200).
- Specs and readings committed before any run; everything EXPLORATORY, not a result. Detached runs
  with a watcher and the 10-minute commit loop; cached runs reused after a restart. About six hours
  on four workers in all.

S71 explore_g_layers (seeds 700-709): which layers must be routed. Perfect gates on Hebbian memory:
 ORACLE_ALL (all three layers; your ORACLE_H), ORACLE_23 (perfect at layers 2-3, LOCAL3 at layer 1;
 your ORACLE23), ORACLE_1 (perfect at layer 1, LOCAL3 at layers 2-3). On delta memory (β=1, L2 keys,
 decay 0.95): ORACLE_ALL_D, ORACLE_23_D. 50 runs. CHECKs: each oracle is one-hot on the stream at its
 perfect layers and LOCAL3 at the others; ORACLE_ALL equals your ORACLE_H code path. Readings (BOUND,
 paired): "layer 1 is the block" if ORACLE_ALL beats ORACLE_23; "layer 1 is not needed" if
 ORACLE_23 ≥ ORACLE_ALL − 1 pair; "layer 1 alone suffices" if ORACLE_1 ≥ ORACLE_ALL − 1 pair;
 "delta tolerates an unrouted layer 1" if ORACLE_23_D ≥ ORACLE_ALL_D − 1 pair. ORACLE_ALL and
 ORACLE_ALL_D on 700-709 are also the validity oracles of S72 and S73.

S73 explore_g_register3_delta (delta memory, β=1 fixed, L2 keys, decay 0.95; seeds 700-719, the
 10-seed arms on 700-709): the discovery devices, on the memory where routing pays. REG3 = your S56
 RegBDH with ONE gate shared by layers 1-3, g^(ℓ)_t = softmax(W_g tanh(A [LN(r^(ℓ)_t); window_t;
 R_{t−1}])), A initialised so u ≈ W_in window_t at step 0 at every layer (at layer 1 r^(1) =
 LN(embedding), kept for a uniform code path); register, SG and the SLOW groups as S56's REG.
 Arms: BASELINE_D (LOCAL3+SLOW; F's S70_DELTA is the reference), REG3_D (no nudge), REG3_NUDGE_D
 (the 5% labelled nudge, validity; 10 seeds), RESGATE_ONLY_D (the per-layer residual gate without
 the register, at all three layers; 10 seeds: can the gate read the stream tag the delta memory
 writes into its own keys?), LATCH3_D (S57's HM-RNN latch, slope annealing 1→5, no prior, feeding
 the gate at all three layers; 10 seeds), LATCH3_NUDGE_D (10 seeds), SINGLE_D (k=1 delta channel,
 fixed decay, no gate, F's single-channel path as in S53/S68; 10 seeds: is the gate needed on this
 task). 90 runs. No forget gate anywhere: with two blocks per stream, clearing at a CTX token
 erases that stream's earlier block. CHECKs: S56's 12 and S57's latch checks at layer 1 (causality,
 dead-STE guard); REG3 with the layer-1 path disabled equals your S56 code path bit for bit on a
 recorded run through 2400; with every new path disabled equals LOCAL3; delta β=0 equals Hebbian;
 SINGLE_D equals F's single-channel path.
 Readings (ROUTED-BOUND, paired on the shared seeds):
   "the register carries the cue" if REG3_NUDGE_D ≥ 9/10;
   "discovered (register)" if REG3_D beats BASELINE_D;
   "discovered (residual gate)" if RESGATE_ONLY_D beats BASELINE_D;
   "discovered (latch)" if LATCH3_D beats BASELINE_D;
   "the latch holds on delta memory" if LATCH3_NUDGE_D's z at the end is ≥ 0.9 at CTX and ≤ 0.1 at
     KEY and at VAL on ≥ 8/10 runs;
   "the delta stack carries the cue" if BASELINE_D's probe at the residual entering layer 3 is
     ≥ 0.9 at K and at V on ≥ 10/20 runs (descriptive: the same for every arm, and whether the
     runs whose probe reads the stream are the ones that route);
   "the gate is not needed on this task" if SINGLE_D BOUND ≥ 8/10 (descriptive; print
     write_order_acc).

S72 explore_g_register3 (Hebbian memory; seeds 700-719): the register gate at all three layers,
 on the main line's memory. Arms: BASELINE (LOCAL3+SLOW; F's S70_HEBB is the reference), REG3 (no
 nudge, SG), REG3_NUDGE (validity). 60 runs. CHECKs as S73's, on the Hebbian path. Readings
 (ROUTED-BOUND): "the register carries the cue at all three layers" if REG3_NUDGE ≥ 17/20;
 "architecture insufficient" if REG3_NUDGE < 12/20 (then: does the per-(stream, block, pair)
 accuracy pattern match S56's 0.32 plateau?); "discovered" if REG3 beats BASELINE; "not
 discovered" otherwise.

Order: S71 → S73 → S72. Push after each. One report after S71 (one page), one after S73+S72
(counts table first, under three pages), and a "Session G follow-up" section in
explore_out/README.md.
```

---

# Prompts for three further sessions (R, V, I), 2026-10-10

R replicates the headline result from the report alone; V referees the report against the records; I asks whether the
partition emerges under a dense next-token loss. R and V have their own rules (R must not see the code; V changes
nothing). I is an exploratory session like F/G/H: paste the common preamble above, then I's block.

## Session R — clean-room replication (own repository; pushes an orphan branch claude/replicate-R; seeds 900-939)

```
You are replicating a result from a technical report without access to the authors' code. Attached:
multichannel_hebbian_report_v8.pdf (Revision 8, draft). The report describes a research fork of
Pathway's BDH (github.com/pathwaycom/bdh, public) with k memory channels and a learned gate, and a
pre-registered result: a recipe called window gate + SLOW + SPLIT binds eight streams with one
channel per stream in 38/40 runs (Section 5, Table 1) against 15/40 for the same recipe without the
split. Your job is to find out whether the report is enough to reproduce that, and what it leaves
out.

Isolation rules; they are the experiment:
- Work in a fresh directory. Clone ONLY github.com/pathwaycom/bdh. Do NOT clone, fetch, browse,
  web-search or otherwise look at github.com/Ynkling/clankers or any of its branches, and do not
  search the web for the project, until Phase 3 says so. If at any point you realise you have seen
  any of its code, say so in the report: the replication is then void and you stop.
- Your only sources are the PDF and Pathway's repository. Wherever the report does not determine a
  detail (a batch size, an initialisation, an evaluation interval, what "conv" is, how the probe is
  drawn, ...), choose something reasonable and LOG IT in UNSTATED.md (what, the choice, why) before
  running. That file is a deliverable in its own right.
- Push as an orphan branch: git init; git remote add origin <the clankers URL the user gives you>;
  commit; git push origin HEAD:refs/heads/claude/replicate-R. Never fetch from that remote before
  Phase 3. Python 3.11+, torch; 1 thread per run (torch.set_num_threads(1)), 4 workers.

Phase 1, implement from the report alone: the task (the grouped layout, S=8 streams, P=4 keys, 16
value tokens, one query; Section 3), the model (BDH with k=16 channels and the gate; N, D, layers,
decay and "conv" as the report states them; Sections 1, 3, 4), the outcomes (bound, the transition,
bound routed with the stream-to-channel map at value positions; Section 3), the recipes WIN3_SLOW_D8
(window gate + SLOW, no split) and WIN3_SPLIT_D8 (with SPLIT; Sections 3-5), and the perfect gate
(one-hot on the stream) as the validity oracle. Unit-test what the report makes checkable: the gate
cannot see token t−3; the split's mechanics (fires on a plateau, copies the busiest row onto the
idlest with the stated noise, zeroes the Adam state, at most three, at least 4800 apart); the
outcome definitions on hand-made gates.
Phase 2, run: ORACLE on seeds 900-901 (validity: both must bind), WIN3_SLOW_D8 and WIN3_SPLIT_D8 on
seeds 900-919, at the budget the report gives for eight streams. Record per run: bound, transition,
bound routed, the map, every split (update, c*, c0), final accuracy per stream, wall time. Fix the
readings in the code's docstring and commit BEFORE running: "replicated" if WIN3_SPLIT_D8 bound
routed ≥ 18/20 and beats WIN3_SLOW_D8 by exact one-sided McNemar p < 0.05 on the paired seeds;
"partially replicated" if ≥ 12/20; "not replicated" otherwise. Wilson 95% intervals on every count.
Push the records before Phase 3.
Phase 3, only after Phase 2's records are pushed: fetch the main line
(claude/bdh-growth-hebbian-inference-w90069), read README.md, test_window_gate.py and what it imports,
and write DIFF.md: every discrepancy between your implementation and theirs, each labelled "stated
in the report (I misread it)", "unstated" or "contradicts the report". If Phase 2 read "partially"
or "not replicated", rerun WIN3_SPLIT_D8 on seeds 900-909 with ONE discrepancy fixed at a time, the
ones you judge likeliest to matter, at most four reruns, and report which one moves the count.
Report: replicate_report.md, under four pages, counts table first, then UNSTATED.md and DIFF.md
summarised, then what the report should add for the next reader. Label everything "independent
replication, EXPLORATORY, not a result". The user relays nothing by hand: push, and say when it is
up.
```

## Session V — referee (branch claude/review-V off the main line; writes only under docs/review_2026-10/)

```
You are the referee. Clone Ynkling/clankers, branch claude/bdh-growth-hebbian-inference-w90069 (the
main line), and create claude/review-V from it. Read multichannel_hebbian_report_v8.pdf and its
.tex, README.md, docs/audit_2026-10.md, docs/revision7.md, specs/, the docstrings of every test_*.py
the report's provenance tables cite, results/README.md and the JSON records under results/X and
results/L, explore_out/README.md, and docs/reading/. Then write an adversarial review.
Rules: you change nothing outside docs/review_2026-10/ (the review, your scripts, their outputs).
You run nothing but your own checking scripts and, at most, one recorded run's bit-for-bit
reproduction per main-line test if you want to verify the reproduction chain (1 thread; record the
CPU; the records name theirs). You do not fix what you find; you report it, with file, line and
number for every point. You are not asked to be kind or harsh; you are asked to be right, and to
say what you could not check and why. Commit and push after each deliverable.

Deliverables, in docs/review_2026-10/:
1. recompute.py, recompute.md: every count, discordant pair, p-value, Wilson interval and band in the
   report's Table 1, the copy/reset table and the Phase VI provenance tables, recomputed from the JSON
   records (never from docstrings), with a column "matches the report". Then every number in
   Sections 5-11 that a record can support. Disagreements first.
2. preregistration.md: for each main-line test the report cites, the git history: commit and
   timestamp of its spec in specs/ (or of its docstring's readings), of the test's first commit, of
   its first run (meta.started in the results), and of the results commit; any change to a
   docstring's claims or readings after the first run (git log -p), quoted verbatim. The same, as
   far as the branches allow, for the exploratory "readings committed before any run" claims on
   claude/outside-ideas and claude/explore-F/G/H: at least three screens per branch.
3. claims.md: every sentence of the abstract, Sections 5, 6 and 11-13 and the conclusion that states
   a result, with a verdict: SUPPORTED (by which record), OVERSTATED (how), UNSUPPORTED, or
   UNVERIFIABLE (what would be needed). Include the report's corrections to Revision 7 and whether
   each correction is itself supported.
4. statistics.md: the choices (one-sided McNemar on discordant pairs, pooling two machines' disjoint
   seeds, "bound" claims without p-values, the "~60 claims per machine" framing, the bands), what a
   statistician referee would object to, what analysis would answer each objection; recompute any
   p-value you dispute.
5. literature.md: for each of the report's references, whether the citation resolves (arXiv id,
   title, authors) and whether the report's one-line use of it is faithful: check against the notes
   in docs/reading/ and, where arxiv.org is reachable from your container, the abstract. Flag any
   characterisation that goes beyond what the note supports, the BDH-CQ reconstruction above all.
6. review.md, under five pages: the contribution as you understand it; its three strongest points;
   every substantive weakness, each with the evidence that would resolve it; the five fixes that
   matter most, ranked; the questions you could not answer from the repository; and a
   recommendation as a venue would want it (accept / minor / major / reject) with the reason.
```

## Session I — a dense loss (branch claude/explore-I off outside-ideas; seeds 800-859; output explore_out/I/)

Paste the common preamble first, then:

```
Your subject: does the channel partition emerge under a dense next-token loss on interleaved
sources? The project's binding task puts its loss at one query per sequence. Revision 8, Section 15,
says language supplies a denser loss, and Section 10 and Session G's reports (explore_out/G/) show
the distant-cue failures under the sparse loss: the stack does not carry the block's context token,
and the loss gives no pressure to. Background: README.md Sections 3-5, 10 and 15;
docs/reading/distant_cues_2026-10.md §3; explore_out/G/report_2.md; for the task family, Bietti et
al. 2023, "Birth of a Transformer: a memory viewpoint" (in-context bigrams), and Edelman et al.
2024, "The evolution of statistical induction heads: in-context learning Markov chains" (read the
abstracts if arxiv.org is reachable; the task is defined fully here).

Task ICMC (in-context Markov chains with sources), explore_i_task.py; a new task class, not a
subclass of BindTask, keeping the harness conventions (fixed L; make_batch(B, gen); stream_labels
giving every position its source; a loss mask):
- Vocabulary: S source tokens SRC_s, then V = 16 word tokens.
- Per sequence, each source s draws its own table T_s: for every word w, a set of 3 successors drawn
  uniformly without replacement from the 16 words; the next word is uniform over the 3. Different
  sources' sets are drawn independently; the tables are fresh per sequence, so there is nothing to
  memorise in the weights: the sets must be learned in context.
- Layout: NB blocks per source, each [SRC_s, w_1, ..., w_n] with n uniform in {LMIN..LMAX}; the S·NB
  blocks in uniformly random order; a block continues its source's chain from the last word of that
  source's previous block (the first word of a source's first block is uniform). Fixed L: draw the
  lengths as compositions, as explore_f_tasks.RandHeaderTask does (copy that file unchanged if useful).
- Loss: cross-entropy at every position whose target is a word with a known same-source predecessor,
  i.e. masked at SRC tokens, at the first word of each source's first block and at the prediction OF
  each SRC token (switches are unpredictable).
- What the memory must do: at word w of source s the right successor set is T_s[w], held in context
  only; a memory that mixes sources holds the union over sources, up to 3S words. The cue is the
  block's SRC token, 1 to LMAX tokens back: the header layout's distant cue, now with a loss at every
  position.
- Metrics per run (held-out 1024 sequences, at every evaluation): the masked loss; SET accuracy (the
  fraction of masked positions whose predicted top-3 set equals T_s[w]: the clean readout of "knows
  the right set"); both by position within block (1, 2, 3-4, 5+); routing: η² by source of the gate
  at word positions, the source-to-channel map at word positions (as the bound-routed map), the
  margin; and the S55 probe (logistic regression from the residual entering each layer to the
  source, at word positions, held-out 20%).
Model: the project's BDH (N=256, D=32, 3 layers, decay 0.95, conv as in the eight-stream tests)
with k channels; gates: the LOCAL3 window gate, a WIDE window gate (width LMAX+2, same construction,
so it can see the SRC token at every word of a block), the perfect gate, and, copied unchanged from
claude/explore-G if present, Session G's register gate at all three layers (REG3; copy
explore_g_regmodel.py and what it needs). Recipes: SLOW; the KEYMASS split with "key positions" =
all word positions (there are no keys; say so in the docstring).

S74 explore_i_pilot (seeds 800-803; constants only, no readings): make the task usable. Arms at
S=2 and at S=4, 12000 updates: ORACLE (perfect gate, k=S), SINGLE (k=1), NOMIX (S=1 source per
sequence at the same L: the floor without interference). Start at L=192, NB=3, LMIN..LMAX = 6..14,
B=32. Usable at a configuration if ORACLE's SET accuracy at the end ≥ 0.9 and SINGLE's ≤ ORACLE − 0.2
on ≥ 3/4 seeds; adjust V, LMIN..LMAX, NB, L, B (never the model) until it is, logging every change
in the pilot's log; measure the wall time per run. Then commit the constants and S75's readings
before any S75 run. If nothing is usable within a day of compute, report and stop.

S75 explore_i_dense (S=2 on seeds 810-829; the 10-seed arms on 810-819; 24000 updates, or what the
pilot needed): ORACLE (810-811, validity), SINGLE (20), WIN3_SPLIT (k=S; 20), WIDE_SPLIT (k=S; 20),
REG3 (no nudge; 20), WIDE_SLOW (no split; 10), REG3_NUDGE (5% labelled nudge; 10). If the pilot's
runs took over 30 minutes each, every arm runs on 10 seeds and the thresholds below become 8/10,
≤ 2/10 and 5 vs 0. Outcomes: ROUTED := η² by source at word positions ≥ 0.9 and a one-to-one
source-to-channel map at the end; GAP := (SET(arm) − SET(SINGLE)) / (SET(ORACLE) − SET(SINGLE)) on
the paired seed, the fraction of the interference gap closed. Exact one-sided McNemar on ROUTED.
Readings fixed now:
  "the dense loss supplies the cue" if WIDE_SPLIT is ROUTED on ≥ 15/20 with median GAP ≥ 0.8 (a
    gate that can see the cue finds it label-free under this loss);
  "the partition emerges with the three-token window" if WIN3_SPLIT is ROUTED on ≥ 15/20;
  "the dense loss does not supply the cue" if every learned-gate arm is ROUTED on ≤ 5/20 (then
    report η² by source at block positions 1-2 against 5+, and the probe: does the stack carry the
    source here where the sparse-loss stacks did not?);
  "the register is discovered under a dense loss" if REG3 beats WIN3_SPLIT on ROUTED (p < 0.05);
  "the split matters here" if WIDE_SPLIT beats WIDE_SLOW on ROUTED on the shared seeds.
  Descriptive: GAP by position within block for every arm (the first words after a switch are where
  a three-token window sees the cue and a mixed memory cannot).
S75 at S=4 (seeds 830-839): ORACLE (2), SINGLE, and the two learned arms with the highest ROUTED
counts at S=2 (decided by that count; say which). Reading: "carries to four sources" if the best
arm is ROUTED on ≥ 8/10.
CHECKs: the task (every target is its source's chain; the mask; the compositions and the block
order; stream_labels; SET accuracy of the true sets is 1.0); the model with k=1 on the grouped task
equals a recorded run bit for bit with the new task and gates disabled; the WIDE gate sees token
t−(LMAX+1) and not t−(LMAX+2); the split's mechanics on this task; the probe's synthetic check.
Order: S74 → S75 (S=2) → S75 (S=4). Push after each. One report after S74 (one page: the constants
and the pilot counts), one after S75 (counts table first, under three pages), and a "Session I"
section in explore_out/README.md.
```

---

## Session F — follow-up 2 (S76-S77; seeds 1000-1059; branch claude/explore-F, output explore_out/F/)

Written 2026-10-10 after F's report 3, G's report 2 and H's report 3. The common rules apply.

```
Follow-up to your S68-S70. Two screens. Context since your round: Session G (explore_out/G/report_2.md)
found that on the P=8 header layout the delta memory binds where the Hebbian one does not (its
oracle 2/2 vs 3 of 6; a nudged register routes 7/10 on delta, 0/10 on Hebbian), so delta memory is
the live route for distant cues; but your S54(c) has the eight-stream recipe stalling on delta
channels (3/10 vs 9/10) with a slow delta oracle (22800-28800 against 2400-4800). Session H
(explore_out/H/report_3.md) found the split is an exact row copy: no noise, no Adam reset needed.
Your S68 settled the premise (one delta channel 2/10 at S=4, 0/10 at S=8). Seeds 1000-1059 (if any
appears in explore_out/ or results/, say so and shift the block by 100).

S77 explore_f_probe (seeds 300-309 and 350-359, reruns; cheap, run first): put your one-seed
 observation of stream-tagged keys on ten seeds. Rerun S53's single delta channel (P=4 header,
 300-309) and S59's GDN single channel (P=8 header, 350-359) to their records (bit for bit on this
 CPU; assert it) and, at the end of each run, fit Session G's S55 probe (logistic regression,
 held-out 20%, 2000 sequences, from the residual entering layers 1, 2 and 3 and from the final
 layer, at K and at V positions, to the stream; copy explore_g_probe_child.py's probe from
 claude/explore-G unchanged if it can be used as is, otherwise reimplement to that specification
 and pass G's synthetic check: 0.998 on an informative feature, ~0.5 on a random one). Also the
 same on the Hebbian single channel of S69 (350-359), which bound nothing. Readings fixed now:
 "the delta stack tags its keys by stream" if, among the bound delta runs, the probe at the
 residual entering layer 3 is ≥ 0.9 at K and at V on ≥ 8 of them while layer 1's is ≤ 0.6; "the tag
 is where binding is" if the bound runs' layer-3 probe exceeds the unbound runs' by ≥ 0.2 (median);
 descriptive otherwise, with the probe by layer for every arm. One page in report_4.md.

S76 explore_f_eight (S=8, P=4, k=16, conv, 43200; test_window_gate's Part C/D path): why the delta
 recipe stalls at eight streams, and whether it can be made not to.
 Phase 1, the oracle (seeds 1000-1003): the perfect gate on delta channels with β ∈ {1, 0.25} ×
 keys ∈ {L2-normalised (yours), raw (the Hebbian model's keys)}, four arms, and the Hebbian perfect
 gate as the reference. Readings: "the slowness is the overwrite" if both β=0.25 arms bind within
 2× the Hebbian oracle's transition on ≥ 3/4 seeds and the β=1 arms do not; "the slowness is the key
 normalisation" if the raw-key arms do and the L2 arms do not; "neither" otherwise (then report
 the oracles' accuracy curves and where they sit at 12000 and 24000). Pick the delta variant with the
 earliest median transition as DELTA* for Phase 2 and say which; commit the choice before Phase 2.
 Phase 2, the recipe (seeds 1010-1019): LOCAL3 + SLOW + the split in H's simplified form (exact
 copy of the busiest row onto the idlest at the KEYMASS trigger, no noise, no Adam reset; the rest
 of the trigger as test_window_gate's Part D) on DELTA* channels (DELTA_SPLIT) and on Hebbian
 channels (HEBB_SPLIT, the reference; H's S61 COPY_NONOISE bound 9/10 on 480-489), paired. Outcome
 BOUND ROUTED with the transition. Readings: "the delta recipe keeps eight streams" if DELTA_SPLIT ≥
 HEBB_SPLIT − 1 discordant pair; "the delta recipe loses eight streams" if HEBB_SPLIT beats
 DELTA_SPLIT by ≥ 4 vs 0; otherwise "inconclusive". Report the splits (update, target labelled ok)
 and the MERGED counts per arm. CHECKs: your delta checks (β=0 equals Hebbian; the recorded
 reproduction with the new path disabled); the simplified split's mechanics equal H's
 COPY_NONOISE on one recorded run through its first split (copy explore_h_copy2x2_child's split
 op if that is simpler than rewriting it; cite the SHA).
Order: S77 → S76. Push after each. report_4.md after S77 (one page), report_5.md after S76
(counts first, under three pages), and the README section.
```

## Session H — follow-up 2 (S78-S79; seeds 1100-1159; branch claude/explore-H, output explore_out/H/)

Written 2026-10-10 after H's report 3. The common rules apply.

```
Follow-up to your S61-S63. Two screens on the split, now that you have shown it is an exact row copy.
Context: the main line has a classifier fix, fail_class_v2 (commit ba8901a on
claude/bdh-growth-hebbian-inference-w90069: the key rule ahead of the margin rule; copy the module
unchanged, do not merge the branch), and is running test_split_copy (the copy vs the reset at eight
streams, pre-registered). The two-stream configuration (S=2, P=4, k=2, no conv) is the project's
hardest relative to its size: the main line's WIN3_SLOW discovers 69/80 and its failures are key
splits, which the KEYMASS split was never applied to because at k=S no channel is idle. Seeds
1100-1159 (if any appears in explore_out/ or results/, say so and shift the block by 100).

S78 explore_h_two (S=2, P=4, k=2, no conv, 24000; test_window_gate's Part A path; seeds 1100-1119):
 the split at k=S=2. Arms: REF (WIN3_SLOW, the main-line arm); SPLIT2 (the KEYMASS trigger with c* =
 the larger key-position read mass and c0 = the other row; W_g[c0] := W_g[c*] exactly, no noise, no
 Adam reset; checks every 2400 from 4800, cap 3, gap 4800); SPLIT2_NOISE (the main line's form:
 noise 0.1 std on both rows and the Adam reset; does symmetry need breaking at k=2?); K4 (k=4, the
 two-spare-channel recipe of batch 18's S52 W2_K4, copied from claude/outside-ideas, under this
 budget). Outcome DISCOVERED for the k=2 arms; for K4 the outcome S52 used (copy its definition and
 say so). Failure classes under v1 AND v2, both reported. Validity: the perfect gate on 1100-1101.
 Readings fixed now: "the split carries to two streams" if SPLIT2 beats REF on DISCOVERED (exact
 one-sided McNemar p < 0.05 on 20 pairs: 5 vs 0, 7 vs 1, 9 vs 2 or better); "noise is needed at
 k=2" if SPLIT2_NOISE beats SPLIT2 by ≥ 4 vs 0; "spare channels are the better device" if K4 beats
 SPLIT2 by ≥ 4 vs 0 on the respective outcomes; "neither device" if no arm beats REF. Diagnostics:
 for every split, η² by key and by stream at KEY and at VAL before and at the next check after it
 (did the copy break the key split, and what formed instead?); the v1/v2 disagreements with their
 η² and margin.

S79 explore_h_constants (S=8, P=4, k=16, conv, 43200; Part C/D path; seeds 1120-1129): which of the
 split's constants matter, with the simplified copy as the base. The report states them without
 having varied them, and an independent replication (Session R) will have to guess the ones it does
 not state. BASE: exact copy, no noise, no reset; checks every 2400 from 4800; fire if probe acc
 < 0.95 and rose < 0.02; 64-sequence probe; cap 3; gap 4800. One change at a time: INT1200 (checks
 every 1200 from 2400, gap 2400), CAP1, CAP6 (gap 2400 so it can fire), PROBE16, THR05 (rose < 0.05).
 Outcome BOUND ROUTED with the transition; record every firing. Readings fixed now, per variant:
 "robust" if the variant ≥ BASE − 1 discordant pair; "sensitive" if BASE beats the variant by ≥ 4
 vs 0; "inconclusive" otherwise; and "the extra splits are harmless" if CAP6's bound-routed count
 ≥ BASE − 1 with its extra firings labelled ok (c* ≥ 2 streams, c0 none). Descriptive: transitions
 by arm, and for INT1200 whether the first split comes earlier.
CHECKs: the copied modules' SHAs; SPLIT2 on one seed with the threshold at 0 equals REF bit for bit;
the trigger's mechanics at k=2 (c0 is a busy row); S52's W2_K4 record reproduced through 2400 on
this CPU with the copied code; fail_class_v2 equals v1 on every run except where η² by key ≥ 0.5
and the margin rule fired (list them).
Order: S78 → S79. Push after each. report_4.md after S78 (counts first, two pages), report_5.md
after S79, and the README section.
```
