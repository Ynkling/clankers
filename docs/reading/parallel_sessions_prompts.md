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
