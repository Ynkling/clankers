# test_register_gate.py — specification (DRAFT, from Session G, for the user and X to adopt)

Drafted 2026-10-11 on claude/explore-G from screens S71-S83 (EXPLORATORY). To be committed to `specs/` on the main line,
reviewed, and amended (dated) before the test is written and before any run. Both machines run it, each on its own seeds.

## The specification (proposed wording)

> test_register_gate.py — a gate that discovers the stream from a distant cue (pull first; spec committed to specs/
> before the test is written; both machines, disjoint seeds, X 600-619, L 1600-1619; --machine X|L; --also pools).
>
> Task: RandHeaderTask-lite (Session F's explore_f_tasks.RandHeaderTask(P=4, nb=2, lmin=1, lmax=3), S=2, n_vals=16,
> n_q=1; copied byte for byte from claude/explore-F at 9a6c2bf), k = 2, no conv, Hebbian memory (decay 0.95), 24000
> updates, evaluation every 1200 on 2048 held-out sequences, early stop after three evaluations >= 0.95.
> Arms (20 seeds per machine, paired by seed: the same batches; the same initial parameters wherever the models share
> them):
>   BASELINE        LOCAL3 + SLOW (test_window_gate's WIN3_SLOW on this task)
>   REG3            the register gate shared by the three layers (below), SLOW, no term
>   REG3_KEYHINGE   REG3 + the key term for updates 1-2400
>   LOCAL3_KEYHINGE BASELINE + the key term on the window gate for updates 1-2400 (no register)
> Validity, on the first two seeds per machine: CEIL (the perfect gate at all three layers, single-rate Adam 1e-3) and
> REG3_NUDGE (REG3 + the 5% labelled nudge).
> Outcome: ROUTED-BOUND (bound, and the routing at the end: margin >= 0.9 and eta^2 by stream at KEY > 0.9).
> Claims, pooled over both machines primary, per machine secondary, exact one-sided McNemar on ROUTED-BOUND:
>   C1 REG3_KEYHINGE beats BASELINE.  C2 REG3_KEYHINGE beats LOCAL3_KEYHINGE.  C3 REG3_KEYHINGE beats REG3.
>   R  REG3_KEYHINGE ROUTED-BOUND >= 18/20 on each machine (a rate claim, with its Wilson interval).

## Background (claude/explore-G, EXPLORATORY; reports 3-6 in explore_out/G/)

- S71 (seeds 700-709): on Hebbian memory the routing that matters is layer 1's.
  - The perfect gate at layers 2-3 with LOCAL3 at layer 1 binds 0/10, against 10/10 for all three layers.
  - The perfect gate at layer 1 alone binds 9/10.
- S72 (700-719): the register gate shared by all three layers (REG3) binds routed 20/20 with the 5% nudge. Label-free it
  does so on 4/20, against BASELINE 0/20 (F's S70_HEBB on 330-349: 0/20 routed). The failures lock into a key split
  before the register's write at CTX appears.
- S82 (2100-2119), against BASELINE 0/20 and REG3 2/20:
  - REG3_KEYHINGE: 20/20, transitions 3600-6000.
  - REG3 with the gate group at 1e-4 for updates 1-1200: 0/20.
  - REG3 without the window input: 5/20.
- S83 (2100-2119, paired with S82):
  - LOCAL3_KEYHINGE: 0/20 ("the register is needed").
  - A KL-to-uniform term at key positions: 1/20 ("the identity term is needed").
  - The same at every body position: 0/20.
  - The key term for updates 1-1200: 18/20.

## Interpretation, fixed here before any code

**The task.** `explore_f_tasks.py` copied byte for byte (RandHeaderTask, body_positions, rand_header_routing_stats,
check_rand_header, write_order_acc). Each stream's 4 keys come in 2 blocks of 1-3 pairs, each block headed by one CTX
token, and the 4 blocks are in random order. L = 23, qpos 21, vocabulary 22. No position determines a token's stream.

**The models** (one new module, `register_gate.py`, written from these definitions and CHECKed against Session G's
records):
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
the architecture. S83 found that the two weaker priors tried do not discover: a KL term on key positions, and the same
everywhere. Both hold the gate at uniform and starve the register. The claims below are about this term on this task and
nothing wider.

**Recipes (all Adam, lr 1e-3).**
- SLOW (BASELINE, REG3, REG3_KEYHINGE, LOCAL3_KEYHINGE, REG3_NUDGE):
  - The gate group at 1e-3 throughout: LOCAL3's W_in, W_g and window, plus every REG3 parameter (A_v, A_r, W_g2, A_R,
    W_R, w_R, b_R).
  - Every other trainable parameter at 1e-4 for updates 1-2400 and 1e-3 after (the switch right after the evaluation at
    2400).
- CEIL: Adam 1e-3 on every parameter throughout.
- REG3_NUDGE: REG3 + Session G's nudge, applied inside make_model: 300 Adam steps at lr 0.01 on the REG3 parameters
  toward 0.475 + 0.05 · onehot(stream) at every position of all three layers' gates (MSE), batches of 64 from a
  generator seeded seed + 55555.

**Outcome (a function, fixed now).** ROUTED-BOUND(r) = BOUND(r) and margin_end(r) >= 0.9 and eta_key_by_stream_end(r)
> 0.9.
- BOUND: a transition held to the end (test_binding_onset.transition is not None).
- margin_end and eta_key_by_stream_end: explore_f_tasks.rand_header_routing_stats on the model's returned gate at the
  end. That gate is the last layer's for REG3 and the one gate for LOCAL3.
- Printed beside it, not claims:
  - BOUND;
  - DISCOVERED (BOUND and VAL cos < 0.5);
  - BOUND ROUTED with a margin guard (the VAL stream → channel map one-to-one at every layer, the routing margin at every
    layer >= 0.9, and every stream's accuracy >= 0.9);
  - failure classes of unbound runs by fail_class_v2;
  - the transition.

**Validity.** A machine is VALID if CEIL binds and REG3_NUDGE is ROUTED-BOUND on both of its validity seeds. A claim is
UNTESTED on a machine that is not VALID. A pooled claim is UNTESTED unless both machines are VALID.

**Seeds.** X 600-619 and L 1600-1619, every arm; CEIL and REG3_NUDGE on 600-601 / 1600-1601. On 2026-10-11 no run
record on any branch used them (checked: the `runs` keys and `seed` fields of every JSON file on every branch, including
the main line's results/X and results/L). Every specification on the main line uses lower blocks; the highest is
test_delta_recipe's 580-599 / 1580-1599. Pairs are by seed.

## Claims (fixed now)

Exact one-sided McNemar on seed pairs (b = first arm only, c = second arm only), SHOWN if p < 0.05.
- Pooled (PRIMARY): b and c summed over both machines' pairs.
- Per machine (SECONDARY): the same test on each machine's twenty pairs.

- **C1** REG3_KEYHINGE beats BASELINE (ROUTED-BOUND).
- **C2** REG3_KEYHINGE beats LOCAL3_KEYHINGE (ROUTED-BOUND).
- **C3** REG3_KEYHINGE beats REG3 (ROUTED-BOUND).
- **R** REG3_KEYHINGE ROUTED-BOUND >= 18/20 on a machine. Printed per machine with its Wilson 95% interval. Pooled: HOLDS
  if it holds on both machines.

**Readings** (pooled; each outcome has exactly one name; UNTESTED if any machine is not VALID):
- "the register, released by the key term, discovers the stream" — C1, C2 and C3 all SHOWN.
- "the term and the register discover it; the register alone is not shown worse" — C1 and C2 SHOWN, C3 not SHOWN.
- "discovered, but the register is not shown to be needed" — C1 SHOWN, C2 not SHOWN.
- "not replicated" — C1 not SHOWN.

The rate claim has its own reading:
- "reliable on both machines" — R holds on X and on L.
- "reliable on one machine" — R holds on exactly one.
- "not reliable" — R holds on neither.

**Printed, not claims:**
- β^R at CTX / KEY / VAL at every evaluation.
- η²_key per layer at updates 1, 300, 600, 1200 and 2400, and the term's firings.
- The routing margin at VAL per layer, and the CTX / KEY / VAL × stream contingency of the gate at every layer.
- Held-out accuracy by (query stream, block, pair index) for unbound runs.
- REG3 vs BASELINE.
- Bands with Wilson 95% intervals for every arm, per machine and pooled (RELIABLE >= ceil(0.9 n), MAJORITY >= ceil(0.5 n),
  MINORITY >= 1, else NEVER).

## CHECKs (all before any run; numbered from the main line's next free number)

- **The inherited chain** (test_window_gate's verification and below, on this machine's own results files).
- **The task:** explore_f_tasks.check_rand_header (every (stream, key) once; S distinct values per key; one CTX per block
  at its start; the block-length frequencies; the block order varies; stream_labels; L, qpos, select), plus L = 23,
  qpos 21, vocabulary 22.
- **The outcome function:** ROUTED-BOUND is True on a constructed record that meets every condition. It is False when
  any one condition fails (no transition; margin 0.89; η² by stream 0.90).
- **The models:**
  - With A_r = A_R = 0, REG3's gate at every layer equals LOCAL3's gate and its logits equal BASELINE's, bit for bit.
  - REG3's initial parameters other than the new ones equal BASELINE's.
  - β^R ≈ 0.1 at initialisation.
  - Strict causality: changing token t changes no logit and no gate at positions < t. Changing token t's register write
    changes no gate at <= t at any layer, and changes layer 1's at t + 1.
  - The stop-gradient: the gates send no gradient into the stack.
  - The SLOW groups cover every trainable tensor once, with lrs [1e-3, 1e-4] at update 1 and 2400 and [1e-3, 1e-3] at
    2401.
- **The key term:**
  - Its gradient through the injection equals autograd of λ · mean_ℓ relu(η²_key − τ) computed directly (with τ lowered
    so every term is active).
  - η²_key at RandHeaderTask's KEY positions equals a hand computation.
  - The schedule: weight 1 at updates 1 and 2400, 0 at 2401 and 2402, counting training forwards only.
  - Its gradient reaches the gate's parameters and the embedding (through the gate's input) and nothing in the stack.
- **Each device with its switch off equals its base bit for bit through 2400** (curve and every statistic):
  REG3_KEYHINGE at λ = 0 equals REG3; LOCAL3_KEYHINGE at λ = 0 equals BASELINE.
- **The S82 reproduction:** BASELINE, REG3 and REG3_KEYHINGE at seed 2100 equal Session G's S82 records
  (claude/explore-G, explore_out/G/s82_devices_results.json; written on an Intel Xeon @ 2.10GHz) through 2400 bit for
  bit: curve and every statistic both records hold. Asserted on a matching CPU, printed otherwise. Session G's S83 did
  exactly this under its own code (s83_repro.json: 6/6 reproduce).
- **The nudge:** it moves only the REG3 parameters and leaves a stream lean > 0.02 at every layer.
- **The perfect gate:** one-hot on the stream at every position and layer. The query's logits are bit-identical when
  every VAL token of the other stream changes.

## Runtime (projection to be printed before the first run)

About 84 runs per machine:
- 4 arms × 20, plus 2 × 2 validity runs.
- Session G's timings on a 4-worker Xeon @ 2.10GHz: an unbound Hebbian run on this task takes 6-8 min at 1 thread; a
  REG3_KEYHINGE run stops at about 3600-6000 (2-3 min).
- About 7-8 CPU-hours per machine, so about 2-2.5 h wall on 4 workers.

Nothing is cut. The results row goes in results/README.md; push.
