# test_register_gate.py — specification (DRAFT, from Session G, for the user and X to adopt)

Drafted 2026-10-11 on claude/explore-G from screens S71-S83 (EXPLORATORY). Revision 1 (2026-10-21): Referee V's review
(`docs/review_2026-10/spec_review_register_draft.md` on the main line), fixes 1-10, applied as written; the changes are
listed at the end. To be committed to `specs/` on the main line, reviewed, and amended (dated) before the test is written
and before any run. Both machines run it, each on its own seeds.

## The specification (proposed wording)

> test_register_gate.py — REG3 with the key term routes and binds a distant-cue stream (pull first; spec committed to
> specs/ before the test is written; both machines, disjoint seeds, X 600-619, L 1600-1619; --machine X|L; --also pools).
>
> Task: RandHeaderTask-lite (Session F's explore_f_tasks.RandHeaderTask(P=4, nb=2, lmin=1, lmax=3), S=2, n_vals=16,
> n_q=1; copied byte for byte from claude/explore-F at 9a6c2bf), k = 2, no conv, Hebbian memory (decay 0.95), 24000
> updates, evaluation every 1200 on 2048 held-out sequences, early stop after three evaluations >= 0.95.
> Arms (20 seeds per machine, paired by seed: the same batches; the same initial parameters wherever the models share
> them):
>   BASELINE          LOCAL3 + SLOW (test_window_gate's WIN3_SLOW on this task)
>   REG3              the register gate shared by the three layers (below), SLOW, no term
>   REG3_KEYHINGE     REG3 + the key term for updates 1-2400
>   LOCAL3_KEYHINGE   BASELINE + the key term on the window gate for updates 1-2400 (no register, no residual input)
>   RESGATE_KEYHINGE  REG3 without the register (reg = False) + the key term for updates 1-2400
> Validity, on the first two seeds per machine: CEIL (the perfect gate at all three layers, single-rate Adam 1e-3) and
> REG3_NUDGE (REG3 + the 5% labelled nudge). VALID: CEIL binds 2/2 and REG3_NUDGE is ROUTED-BOUND on at least 1 of 2.
> Outcome: ROUTED-BOUND = test_register_gate.routed_bound (copied from Session G's explore_g_rhl_parent.py at 076005d):
> bound, and the routing at the end, margin >= 0.9 and eta^2 by stream at KEY > 0.9.
> Claims, pooled over both machines primary, per machine secondary, exact one-sided McNemar on ROUTED-BOUND:
>   C1 REG3_KEYHINGE beats BASELINE.  C2 REG3_KEYHINGE beats LOCAL3_KEYHINGE.  C3 REG3_KEYHINGE beats REG3.
>   C4 REG3_KEYHINGE beats RESGATE_KEYHINGE.
>   R  REG3_KEYHINGE ROUTED-BOUND >= 18/20 on a machine: MET / NOT MET / UNTESTED per machine, with Wilson and
>      Clopper-Pearson intervals; the pooled count beside it.

## Background (claude/explore-G, EXPLORATORY; reports 3-8 in explore_out/G/)

- S71 (seeds 700-709): on Hebbian memory the routing that matters is layer 1's.
  - The perfect gate at layers 2-3 with LOCAL3 at layer 1 binds 0/10, against 10/10 for all three layers.
  - The perfect gate at layer 1 alone binds 9/10.
- S72 (700-719): the register gate shared by all three layers (REG3) is ROUTED-BOUND 20/20 with the 5% nudge. Label-free
  it is on 4/20, against BASELINE 0/20 (F's S70_HEBB on 330-349: 0/20 routed). The failures lock into a key split
  before the register's write at CTX appears.
- S82 (2100-2119), against BASELINE 0/20 and REG3 2/20:
  - REG3_KEYHINGE: 20/20, transitions 3600-6000.
  - REG3 with the gate group at 1e-4 for updates 1-1200: 0/20.
  - REG3 without the window input: 5/20.
- S83 (2100-2119, paired with S82):
  - LOCAL3_KEYHINGE: 0/20.
  - A KL-to-uniform term at key positions: 1/20.
  - The same at every body position: 0/20.
  - The key term for updates 1-1200: 18/20.
- S87 (2700-2719): a frequency hinge (the key term at the positions of the most frequent token types, grouped by token
  id; it reads no labels) 17/20 against the key term's 20/20 (3 vs 0, p 0.125); 10/10 on delta memory. On this task its
  frequent types are the 2 CTX and 4 key types, so it reads about what the key term reads, plus the stream tokens.
- S89 (2842-2861, delta memory, F's round-6 pool tasks; report 8): on RandHeaderTask-pool (keys from 32 types) REG3
  alone routes and binds 16/20, the key term 20/20, a label-free repeat hinge (the term at every token type that repeats
  in the sequence) 20/20; on -allpool (contexts from 16 types) every arm, the key term included, 0/20. These are delta
  results on other tasks; the claims below stay Hebbian on lite.

## Interpretation, fixed here before any code

**The task.** `explore_f_tasks.py` copied byte for byte (RandHeaderTask, body_positions, rand_header_routing_stats,
check_rand_header, write_order_acc). Each stream's 4 keys come in 2 blocks of 1-3 pairs, each block headed by one CTX
token, and the 4 blocks are in random order. L = 23, qpos 21, vocabulary 22. No position determines a token's stream.

**The models** (one new module, `register_gate.py`, written from these definitions and CHECKed against Session G's
classes on every machine, below):
- LOCAL3 is test_window_gate's (batch 1's window gate, W_h frozen and in no group).
- REG3 starts from the LOCAL3 model and adds parameters from a generator seeded seed + 56000, in this order:
  - A_r ~ N(0, 0.01²) (32 × 32);
  - A_R ~ N(0, 0.01²) (32 × 16);
  - W_R ~ N(0, 0.1²) (16 × 32);
  - w_R ~ N(0, 0.01²) (32);
  - b_R = logit(0.1);
  - A_v = W_in and W_g2 = W_g as copies (no draws).
- The register: R_t = R_{t−1}(1 − β_t) + β_t W_R v_t, with β_t = σ(w_R·v_t + b_R), R_{−1} = 0, and v_t the raw
  embedding. The gate at t reads R_{t−1}.
- One gate is shared by the three layers: g^(ℓ)_t = softmax(W_g2 tanh(A_r LN(r^(ℓ)_t) + A_v window_t + A_R R_{t−1})).
  - r^(ℓ) is the residual entering layer ℓ, with LN(embedding) at layer 1, under a stop-gradient into the stack.
  - window_t is LOCAL3's causal width-3 convolution of the raw embeddings.
  - Read gate = write gate. Each layer's memory uses its own G^(ℓ) = g^(ℓ) g^(ℓ)ᵀ.
- LOCAL3's W_in and W_g stay in the model, unused by REG3.
- RESGATE_KEYHINGE is REG3 with reg = False (no register: no A_R term, and none of W_R, w_R, b_R; Session G's S73
  RESGATE_ONLY_D construction on Hebbian memory), plus the key term. It differs from REG3_KEYHINGE by the register only,
  and from LOCAL3_KEYHINGE by the residual input only.

**The key term.** On training forwards of updates 1-2400 only, the loss gains λ · mean_ℓ relu(η²_key(g^(ℓ)) − τ), with
λ = 1.0 and τ = 0.2 (batch 18's W2_KEYHINGE constants).
- η²_key(g) is the η² of the gate vector at the body's KEY positions, read per sequence, grouped by the key's id (token
  − S). It is pooled over channels: between-group over total sum of squares, + 1e-12 (explore_b18_gates.eta2_pool).
- For LOCAL3_KEYHINGE the mean is over one gate, the window gate used at all three layers.
- The term enters through an identity-forward / unit-backward function on the logits (explore_aux_gate._Inject). After
  update 2400 it is not computed.

**What the key term reads (the task-structure caveat).** The key term reads which tokens are keys and which key each is.
It reads no stream label, no CTX token and no block boundary. But it is task structure: it tells the gate, during the
first 2400 updates, that its decision must not depend on key identity at key positions. A language-model version would
have to choose which token classes play the role of keys, and such a choice is a prior about the data, not a property of
the architecture. S83 found that the two weaker priors tried do not reach ROUTED-BOUND: a KL term on key positions, and
the same everywhere. Both hold the gate at uniform and starve the register (Session G's reading of S83's traces). The
claims below are about this term on this task and nothing wider.

**Recipes (all Adam, lr 1e-3).**
- SLOW (BASELINE, REG3, REG3_KEYHINGE, LOCAL3_KEYHINGE, RESGATE_KEYHINGE, REG3_NUDGE):
  - The gate group at 1e-3 throughout: LOCAL3's W_in, W_g and window, plus every REG3 parameter the arm has (A_v, A_r,
    W_g2, A_R, W_R, w_R, b_R).
  - Every other trainable parameter at 1e-4 for updates 1-2400 and 1e-3 after (the switch right after the evaluation at
    2400).
- CEIL: Adam 1e-3 on every parameter throughout.
- REG3_NUDGE: REG3 + Session G's nudge, applied inside make_model: 300 Adam steps at lr 0.01 on the REG3 parameters
  toward 0.475 + 0.05 · onehot(stream) at every position of all three layers' gates (MSE), batches of 64 from a
  generator seeded seed + 55555.

**Outcome (a function, fixed now).** ROUTED-BOUND(r) = `test_register_gate.routed_bound(r)`, copied from
`explore_g_rhl_parent.py` at `076005d` (`bound` :39, `routed_star` :43, `routed_bound` :52, thresholds :35). It holds when:
- `r["transition"] is not None`, where the transition is run_one's `test_binding_onset.transition(curve)` (BIND_PASS 0.95);
- `r["end"]["margin"] >= 0.9`;
- `r["end"]["eta_key_by_stream"] > 0.9`.

A missing statistic counts as 0. The end statistics are `explore_f_tasks.rand_header_routing_stats(model, task, probe)`,
where:
- the probe is run_one's statistics probe `probe_batch(task, seed)` (test_multilayer_binding.py:352: its own generator,
  seed + 99,000, PROBE_B sequences);
- the gate is the one the model returns (layer 3's for the REG3 arms, the one gate for LOCAL3).

Printed beside it, not claims:
- BOUND;
- DISCOVERED (BOUND and VAL cos < 0.5; the name is reserved for this outcome);
- ROUTED AT EVERY LAYER: bound, the VAL stream → channel map one-to-one at every layer, the routing margin ≥ 0.9 at
  every layer, and every stream's accuracy ≥ 0.9 (printed, not a claim);
- failure classes of unbound runs by fail_class_v2;
- the transition.

**Validity.** A machine is VALID if:
- CEIL binds on both of its validity seeds (test_window_gate's rule, VALID_N = 2); and
- REG3_NUDGE is ROUTED-BOUND on at least one of them. One routed nudged run shows that the machine's REG3 code path can
  carry the stream. S82's rule (`explore_g_devices.py:120`) required only the oracle.

Both counts are printed. A claim is UNTESTED on a machine that is not VALID. A pooled claim is UNTESTED unless both
machines are VALID.

**Seeds.** X 600-619 and L 1600-1619, every arm; CEIL and REG3_NUDGE on 600-601 / 1600-1601. On 2026-10-11 no run
record on any branch used them (checked: the `runs` keys and `seed` fields of every JSON file on every branch, including
the main line's results/X and results/L). Every specification on the main line uses lower blocks; the highest is
test_delta_recipe's 580-599 / 1580-1599. Pairs are by seed. Re-checked by the referee on 2026-10-12: all 12 remote branch
heads, 308 JSON/JSONL blobs, every `ARM|seed` key, `seed` field and `seeds` list. No hit in 600-619 or 1600-1619. The
integer keys 600 and 1600 that occur are update steps.

## Claims (fixed now)

Exact one-sided McNemar on seed pairs (b = first arm only, c = second arm only), SHOWN if p < 0.05.
- Pooled (PRIMARY): b and c summed over both machines' pairs.
- Per machine (SECONDARY): the same test on each machine's twenty pairs.

- **C1** REG3_KEYHINGE beats BASELINE (ROUTED-BOUND).
- **C2** REG3_KEYHINGE beats LOCAL3_KEYHINGE (ROUTED-BOUND).
- **C3** REG3_KEYHINGE beats REG3 (ROUTED-BOUND).
- **C4** REG3_KEYHINGE beats RESGATE_KEYHINGE (ROUTED-BOUND).
- **R** REG3_KEYHINGE ROUTED-BOUND ≥ 18/20 on a machine (the RELIABLE band at n = 20).
  - **Per machine:** MET, NOT MET or UNTESTED (machine not VALID). Printed with the count, its Wilson and
    Clopper-Pearson 95% intervals (at 18/20: [0.699, 0.972] and [0.683, 0.988]), and the band.
  - **Pooled (printed beside it):** the count with its Wilson interval and band (RELIABLE ≥ 36/40).
  - **The machines compared:** Fisher's exact test, two-sided, printed and not read.
  - **Printed with R, fixed now:** P(≥ 18/20 on both machines) is 0.04, 0.16, 0.46 and 0.85 at true rates 0.80, 0.85,
    0.90 and 0.95.

**Multiplicity.** C1-C4 are a family of four one-sided tests at 0.05 each, with no correction. The readings are
intersection-union rules: every claim a reading names must be SHOWN, so each reading holds at 0.05. Holm-adjusted
p-values over C1-C4 are printed beside the raw ones. For the registry, this test adds 4 pooled tests, 8 per-machine tests
and one rate rule per machine.

**Readings** (pooled; UNTESTED if either machine is not VALID; each combination of C1-C4 has exactly one name). Every
reading is printed with the caveat line below.
- C1 not SHOWN: "REG3 with the key term not shown to route and bind more often than the baseline", with both arms' pooled
  counts.
- C1 SHOWN, C2 not SHOWN (C3, C4 either way): "REG3 with the key term routes and binds more often than the baseline; not
  shown more often than the key term on the window gate."
- C1 and C2 SHOWN, C3 not SHOWN (C4 either way): "REG3 with the key term routes and binds more often than the baseline
  and than the key term on the window gate; not shown more often than REG3 without the term."
- C1, C2 and C3 SHOWN, C4 not SHOWN: "REG3 with the key term routes and binds more often than the baseline, than the key
  term on the window gate, and than REG3 without the term; not shown more often than the same gate without the register."
- C1, C2, C3 and C4 SHOWN: "REG3 with the key term routes and binds more often than the baseline, than the key term on
  the window gate, than REG3 without the term, and than the same gate without the register."

Caveat line: "The key term reads which tokens are keys and each key's identity, during updates 1-2400 only. REG3 differs
from the window gate by the register and the residual input together; C4 isolates the register."

"Discovers" is not used: the report reserves DISCOVERED (bound and VAL cos < 0.5), which is printed beside ROUTED-BOUND.

The rate reading is one of:
- "R MET on both machines";
- "R MET on X only";
- "R MET on L only";
- "R MET on neither";
- "R UNTESTED on <machine>", with the other machine's verdict printed.

None of these is read as a difference between the machines.

**Printed, not claims:**
- β^R at CTX / KEY / VAL at every evaluation.
- η²_key per layer at updates 1, 300, 600, 1200 and 2400, and the term's firings.
- The routing margin at VAL per layer, and the CTX / KEY / VAL × stream contingency of the gate at every layer.
- Held-out accuracy by (query stream, block, pair index) for unbound runs.
- REG3 vs BASELINE; RESGATE_KEYHINGE vs LOCAL3_KEYHINGE (the residual input, given the term).
- Bands with Wilson 95% intervals for every arm, per machine and pooled (RELIABLE >= ceil(0.9 n), MAJORITY >= ceil(0.5 n),
  MINORITY >= 1, else NEVER).
- The count of runs where ROUTED* on the returned gate differs from ROUTED* at every layer, listed.
- fail_class_v2 against v1 on every unbound run, with η² by key and the margin of each disagreement.

## CHECKs (all before any run; numbered from the main line's next free number)

- **The inherited chain** (test_window_gate's verification and below, on this machine's own results files).
- **The task:** explore_f_tasks.check_rand_header (every (stream, key) once; S distinct values per key; one CTX per block
  at its start; the block-length frequencies; the block order varies; stream_labels; L, qpos, select), plus L = 23,
  qpos 21, vocabulary 22.
- **The outcome function:**
  - ROUTED-BOUND is True on a constructed record that meets every condition. It is False when any one condition fails
    (no transition; margin 0.89; η² by stream 0.90).
  - On every record of `explore_out/G/s82_devices_results.json` and `s83_results.json` at `076005d` (vendored in
    `results/refs/` with commit, path and key), `routed_bound` gives Session G's counts: BASELINE 0/20, REG3 2/20,
    REG3_KEYHINGE 20/20, REG3_SLOWGATE 0/20, REG3_NOWIN 5/20, LOCAL3_KEYHINGE 0/20, REG3_KEYHINGE_POS 1/20, REG3_FLAT
    0/20, REG3_KEYHINGE_1200 18/20, REG3_NUDGE 2/2 and 2/2. The recorded `transition` equals
    `test_binding_onset.transition(curve)` on all 186 records. (Session G re-scored these counts with its own
    `routed_bound` on 2026-10-21: all equal.)
- **The models:**
  - With A_r = A_R = 0, REG3's gate at every layer equals LOCAL3's gate and its logits equal BASELINE's, bit for bit.
  - REG3's initial parameters other than the new ones equal BASELINE's; RESGATE_KEYHINGE's equal REG3's minus the
    register's.
  - β^R at initialisation lies in (0.09, 0.11) at every position of 16 sequences (Session G's CHECK,
    `explore_g_rhl_child.py:466-468`).
  - Strict causality: changing token t changes no logit and no gate at positions < t. Changing token t's register write
    changes no gate at <= t at any layer, and changes every layer's at t + 1.
  - The stop-gradient: the gates send no gradient into the stack.
  - The SLOW groups cover every trainable tensor once, with lrs [1e-3, 1e-4] at update 1 and 2400 and [1e-3, 1e-3] at
    2401.
- **The model against Session G's code, CPU-independent** (asserted on every machine):
  - At seed 600, on 16 training sequences, with the parameters `register_gate.py` builds: REG3's and REG3_KEYHINGE's
    logits, every layer's gate, β^R and every parameter gradient, against the same from `explore_g_rhl_child.Reg3BDH` and
    `explore_g_devices_child.KeyHingeBDH`; LOCAL3_KEYHINGE's against `explore_g_s83_child.TermLocal`;
    RESGATE_KEYHINGE's against `Reg3BDH` with reg = False plus `KeyHingeBDH`'s term. G's classes are extracted from
    `076005d` and run on the same parameters.
  - Tolerance: bit for bit, or, where the operation order differs, to 1e-6 relative, with the measured difference
    printed.
  - The S82 reproduction below is asserted only on a matching CPU; this CHECK is the link on L. (Alternative: copy G's
    model and term code byte for byte, as test_split_copy copied F's delta files (its CHECK 152), and CHECK the syntax
    trees.)
- **The key term:**
  - Its gradient through the injection equals autograd of λ · mean_ℓ relu(η²_key − τ) computed directly (with τ lowered
    so every term is active).
  - η²_key at RandHeaderTask's KEY positions equals a hand computation.
  - The schedule: weight 1 at updates 1 and 2400, 0 at 2401 and 2402, counting training forwards only.
  - Its gradient reaches the gate's parameters and the embedding (through the gate's input) and nothing in the stack.
- **Each device with its switch off equals its base bit for bit through 2400** (curve and every statistic):
  REG3_KEYHINGE at λ = 0 equals REG3; LOCAL3_KEYHINGE at λ = 0 equals BASELINE; RESGATE_KEYHINGE at λ = 0 equals REG3
  with reg = False.
- **The reproduction:** BASELINE, REG3, REG3_KEYHINGE and REG3_NUDGE (S82, seed 2100), LOCAL3_KEYHINGE (S83's
  `s83_results.json`, seed 2100) and CEIL against ORACLE_ALL|2100 equal Session G's records through 2400 bit for bit
  (curve and every statistic both records hold). Each reference is read from a vendored copy in `results/refs/` with its
  source commit (`076005d`), path and key, and where the commit is present the CHECK prints whether the copy equals it
  (as test_delta_recipe Amendment 2(h)). Asserted on a matching CPU (Intel Xeon @ 2.10GHz), printed otherwise.
- **The nudge:** it moves only the REG3 parameters. After it, the stream lean |mean p₀ at stream-0 positions − mean p₀
  at stream-1 positions| on the probe is > 0.02 at every layer (`explore_g_rhl_child.py:514`).
- **The perfect gate:** one-hot on the stream at every position and layer. The query's logits are bit-identical when
  every VAL token of the other stream changes.
- **Seeds.** Every arm runs on the machine's range (X 600-619, L 1600-1619); CEIL and REG3_NUDGE run on its first two
  seeds. The ranges are disjoint from every main-line seed and from the other machine's range. With --also, the other
  file's machine, CPU and seeds are printed.
- **Pairing.** At seed 600, BASELINE, REG3, REG3_KEYHINGE, LOCAL3_KEYHINGE and RESGATE_KEYHINGE draw identical first
  three training batches, and their shared initial parameters are identical. REG3_NUDGE's nudge draws only from its own
  generator (seed + 55555).
- **Workers.** A worker's run is bit-identical to the same run in the parent (REG3_KEYHINGE, seed 1, 1200 updates).
  **Resume.** Records cached by an earlier start reproduce under this code through 2400.

## Runtime (projection to be printed before the first run)

About 104 runs per machine:
- 5 arms × 20, plus 2 × 2 validity runs.
- Session G's timings on a 4-worker Xeon @ 2.10GHz (the referee's medians): REG3_KEYHINGE 2.2 min (2.1-3.0); BASELINE
  6.3 (3.5-6.5); REG3 unbound 8.5-8.7; LOCAL3_KEYHINGE 4.6 (2.1-4.7); validity runs 1.0-1.9. RESGATE_KEYHINGE is not
  screened: between 2.2 (bound) and 8.7 (unbound).
- About 8-10 CPU-hours per machine, so about 2-2.5 h wall on 4 workers.

Nothing is cut. The results row goes in results/README.md; push.

## Revision 1 (2026-10-21): what changed, per Referee V's fixes

1. Outcome: named function `test_register_gate.routed_bound`, copied from G's code; the probe named; a CHECK against
   G's recorded counts on vendored S82/S83 records.
2. Model equivalence: a CPU-independent CHECK against G's classes, asserted on every machine (copy-and-AST alternative
   stated).
3. Validity: REG3_NUDGE ROUTED-BOUND on at least 1 of 2 (was 2 of 2).
4. Readings: reworded without "discover", one name per combination, a caveat line on every reading; S83's mechanism
   labelled as G's reading.
5. Rate claim R: MET / NOT MET / UNTESTED per machine with Wilson and Clopper-Pearson intervals, the pooled count, a
   Fisher comparison printed and not read, and the operating characteristics fixed.
6. Multiplicity: the family named, no correction, Holm printed.
7. CHECKs: β^R tolerance; the stream lean defined; the reproduction widened and its references vendored in
   results/refs/; seeds, pairing, workers and resume CHECKs; fail_class v2 vs v1 and the per-layer ROUTED* agreement in
   the report.
8. Seeds: the referee's re-check recorded.
9. The printed outcome renamed ROUTED AT EVERY LAYER.
10. The optional arm adopted: RESGATE_KEYHINGE and C4, so that a claim can speak of the register alone (about 1-2
    CPU-hours per machine). Without C4 the C2 wording stands as "the register and the residual input together".
