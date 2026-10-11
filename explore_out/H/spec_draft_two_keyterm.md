# test_two_keyterm.py — specification (DRAFT, from Session H; for the user and X to adopt)

A draft written on claude/explore-H (EXPLORATORY branch) in the form of specs/test_split_copy.md. Nothing here has been
run on the main line. Once adopted, it is fixed before the test is written and before any run; both machines run it, each
on its own seeds.

## The user's specification (verbatim)

> Then write explore_out/H/spec_draft_two_keyterm.md: the main-line specification of your S85 proposal,
> in the form of specs/test_split_copy.md. S=2, P=4, k=2, no conv, 24000; WIN3_SLOW with and without the
> key term on updates 1-2400 (S52's W2_KEYHINGE); the perfect gate as validity; outcome DISCOVERED under
> fail_class_v2; 20 seeds per machine (propose blocks; check every branch and the main line's seed
> list); claims per machine secondary and pooled primary: K1 WIN3_SLOW_KEYHINGE beats WIN3_SLOW (exact
> one-sided McNemar); K2 its KEY failures, a printed count; a rate claim >= 18/20 per machine; readings
> named so that each outcome has exactly one name; the task-structure caveat (the term reads key
> positions and key identities) in the spec's words; CHECKs: the term with its switch off equals
> WIN3_SLOW bit for bit through 2400, the S52 and S85 reproductions, the v2 classifier on the recorded
> two-stream runs.

## Background

- **test_window_gate Part A** (X 500-539, L 1500-1539): WIN3_SLOW (LOCAL3 window gate + SLOW, no hinge) discovered
  X 33/40, L 36/40 (69/80 pooled). Its failures are key splits: under fail_class_v2, X KEY 5 + STREAM-PARTIAL 2
  (docs/fail_class_v2.md).
- **Batch 18's S52 W2_KEYHINGE** (claude/outside-ideas, EXPLORATORY; seeds 160-199, Xeon @ 2.10GHz): WIN3_SLOW + the key
  term on updates 1-2400 discovered 36/40, against the same path's LOCAL3_SLOW 31/40 (7 vs 2 discordant, p = 0.18). One
  KEY failure under v1. Three runs sat at the two-of-four pattern, eta^2 by key ~0.5: 163 (KEY under v2), 182 and 186
  (STREAM-PARTIAL under v2, 1.4e-4 and 1.5e-5 below 0.5).
- **Session H's S78** (claude/explore-H, EXPLORATORY; seeds 1300-1319): repair by the KEYMASS copy does not carry to
  k = S = 2. The copy makes the gate uniform, and the same key split re-forms by the next check. REF (WIN3_SLOW) 15/20,
  SPLIT2 16/20, SPLIT2_NOISE 17/20, K4 (two spare channels) 15/20.
- **Session H's S85** (seeds 2310-2329, Xeon @ 2.10GHz): prevention.
  - DISCOVERED: WIN3_SLOW 15/20, WIN3_SLOW + key term (S52's W2_KEYHINGE, unchanged) 20/20 (5 vs 0, one-sided p =
    0.031), the gate held at 1e-4 for updates 1-1200 14/20, both together 19/20.
  - KEY failures under v2: 4, 0, 4, 1.
  - With the term on, eta^2 by key at KEY positions was below the term's threshold 0.2 at every measurement from update
    25 to update 2400 in all 20 runs (maximum 0.195). Where a key split tried to form after the window, the run
    discovered anyway.
  - Pooled over S52 and S85 (different seeds, the same code): 12 vs 2 discordant.
- Every number above is exploratory. This test is the pre-registered check of it, on fresh seeds on both machines.

## Interpretation, fixed here before any code

**The path.** Every learned arm runs test_window_gate's Part A path unchanged: test_short_conv's TASK (S = 2, P = 4,
grouped, n_q = 1, no convolution), 24000 updates, evaluation every 1200 on the 2048 held-out queries, run_one's early
stop (three evaluations >= 0.95), test_window_gate's statistics. The model is LOCAL3: test_window_gate's LocalBDH, the
width-3 window drawn from a generator seeded seed + 31337, W_h frozen and unused.

**The recipe.** SLOW, as test_window_gate's WIN3_SLOW:
- the gate group (W_in, W_g, gate_conv.conv_w) at 1e-3 throughout;
- every other trainable parameter at 1e-4 for updates 1-2400 and 1e-3 after, switched right after the evaluation at 2400.
- No hinge, no split, no other change.

**The arms.**
- WIN3_SLOW: test_window_gate's arm, imported unchanged.
- WIN3_SLOW_KEYHINGE: WIN3_SLOW + the key term (S52's W2_KEYHINGE), and nothing else.
- CEIL_A: test_window_gate's perfect gate on Part A's run_one call (test_short_conv's ceiling_conv, one Adam at 1e-3).
  The plain perfect gate without the conv layer does not bind this layout (batch 18, S60, S78, S85), so ceiling_conv is
  the validity arm, as test_window_gate's.

**The key term.** It is copied byte for byte (syntax trees CHECKed) from explore_keysplits_child.attach_key_hinge,
explore_b18_gates.eta2_key / eta2_pool and explore_aux_gate._Inject on claude/outside-ideas, the code of S52's runs.
- A forward hook on the model adds L_key to the loss on every training forward (training mode, gradients on), through
  the identity-on-logits injection node, the same node as test_slow_start.Inject:
  L_key(u) = w(u) * relu(eta2_key(g) - 0.2),   w(u) = 1.0 for u <= 2400, 0 for u > 2400.
- u is the hook's count of training forwards (one per update).
- After update 2400 the hook returns at once and does nothing.
- eta2_key(g) is the eta^2 of the read gate g at the body's KEY positions 3j+1 (j = 0 .. S*P - 1; the query's key is
  excluded), grouped by the key identity there (token - S). It is between-group over total sum of squares, both summed
  over the k = 2 channels, with 1e-12 added to the total. It is differentiable through g, so its gradient reaches the
  gate's parameters and the embedding.
- The switch: "the term off" is w(u) = 0 at every update, i.e. a window of 0 updates. The hook then does nothing on every
  forward.

**Outcome.** DISCOVERED (explore_common.discovered: bound — a transition held to the end — and final VAL cos < 0.5), the
Part A outcome. Failure classes of the unbound runs under fail_class_v2.fail_class_v2 (primary). v1
(test_router_layout.fail_class) is printed beside it.

**The task-structure caveat** (stated with every claim and in the result):
- The key term is not label-free in the way the window gate is. To compute eta^2 by key it reads the task's layout
  (which positions hold keys, 3j+1) and the key identities (the key token at each of those positions).
- It reads no stream label and no answer, and it acts only during updates 1-2400.
- A positive result therefore says this: holding key-identity structure out of the gate early lets the stream routing
  form. It does not say the gate discovers the partition from the task's loss alone.
- A language model would need a substitute for "key positions" and "key identities", for example token identity at
  every position. That substitute is untested.

**Validity** (test_window_gate's rule, VALID_N = 2). The test is VALID on a machine if CEIL_A binds on both of its first
two seeds within 24000. A claim is UNTESTED on a machine where it is not VALID. A pooled claim is UNTESTED unless both
machines are VALID. UNTESTED is not a negative.

**Seeds.** X 620-639, L 1620-1639 (20 each), every arm; CEIL_A on the first two (620-621, 1620-1621). Pairs are by seed.
- Proposed after a scan of every branch's explore_out/, results/, specs/ and docs/ for seed keys ("ARM|seed"),
  "seed": n fields, "seed(s) n" and ranges n-m.
- The main line's seed list: test_window_gate X 500-559 / L 1500-1559; test_split_copy X 560-569 / L 1560-1569;
  test_delta_recipe X 580-599 / L 1580-1599; test_muon_recipe 400-479 / 1400-1479; restart and attempt seeds 1000-1259
  and 2200-2259 in results/.
- The prompts propose X 600-619 / L 1600-1619 for Session G's register specification.
- 620-639 and 1620-1639 appear nowhere as seeds. The only text matches are source-line ranges in explore-I's
  gpu/audit.json.
- Re-scan before adoption.

## Claims (fixed now)

Exact one-sided McNemar on seed pairs (b = WIN3_SLOW_KEYHINGE only, c = WIN3_SLOW only), SHOWN if p < 0.05.
- Pooled (PRIMARY): b and c summed over both machines' pairs.
- Per machine (SECONDARY): the same test on each machine's 20 pairs.

- **K1** WIN3_SLOW_KEYHINGE beats WIN3_SLOW on DISCOVERED.
- **K2** WIN3_SLOW_KEYHINGE's KEY failures under fail_class_v2: a printed count, per machine and pooled, beside
  WIN3_SLOW's. Not a test and no p-value.
- **K3** (the rate claim) WIN3_SLOW_KEYHINGE discovers >= 18 of 20 on each machine: MET or NOT MET per machine, with the
  Wilson 95% interval. Pooled, it is MET only if it is MET on both machines.

**Readings** (pooled; each outcome has exactly one name):
- **"the key term makes two streams reliable"**: K1 SHOWN pooled, and K3 MET on both machines.
- **"the key term helps, but not reliably"**: K1 SHOWN pooled, and K3 NOT MET on at least one machine.
- **"reliable, but not shown better than WIN3_SLOW"**: K1 not SHOWN pooled, and K3 MET on both machines.
- **"not shown"**: K1 not SHOWN pooled, and K3 NOT MET on at least one machine.
- **"UNTESTED"**: either machine NOT VALID. In that case no other reading is given.

Each reading is printed with the task-structure caveat. K2 is printed under every reading, so a reader can see whether
the key splits are gone or only rarer.

**Printed, not claims:**
- the per-machine K1 (secondary);
- bands with Wilson 95% intervals for both arms, per machine and pooled (RELIABLE >= ceil(0.9 n), MAJORITY >=
  ceil(0.5 n), MINORITY >= 1, else NEVER);
- failure classes under v2 and v1 with eta^2 by key and the margin of every unbound run;
- the key term's firings per run (training forwards with eta2_key > 0.2), and eta2_key at updates 1, 300, 600, 1200 and
  2400;
- median transitions per arm;
- the update at which the read gate's eta^2 by key at KEY positions first exceeds 0.5 and whether it does at the end
  (measured on the run's probe after updates 0, 25, ..., 1200, 1300, ..., 2400 and every 1200 after; eval mode, no
  gradient, training untouched). A near-uniform gate can exceed 0.5 at update 0, so that value is printed but not read
  as a formed split (audit 2.1).

## CHECKs (all before any run unless stated; numbered from the main line's next free number, T1-T11 here)

- **The inherited chain:** test_window_gate's verification on this machine's own earlier results files, with
  test_split_copy's knob for the Muon CHECK where the CPU differs.
- **T1** the copied functions (attach_key_hinge, eta2_key, eta2_pool, _Inject) have the same syntax trees as their
  sources on claude/outside-ideas at a fixed commit (named when adopted). _Inject's forward and backward equal
  test_slow_start.Inject's.
- **T2 (switch off)** WIN3_SLOW_KEYHINGE with the term off (window 0) equals WIN3_SLOW bit for bit through 2400 on the
  machine's first seed: curve, gradient norms and every statistic. The hook is registered and counts forwards, but adds
  nothing.
- **T3 (S52)** WIN3_SLOW_KEYHINGE at seed 160 equals S52's W2_KEYHINGE|160 (claude/outside-ideas,
  explore_out/two_stream_keysplits_results.json; written on a Xeon @ 2.10GHz) through 2400: curve, statistics, and the
  term's firing count and eta2_key at updates 1, 300, 600, 1200 and 2400.
  - Asserted on a matching CPU, printed otherwise.
  - On X (Xeon), Adam runs have reproduced across 2.10/2.80GHz. L (i7-12650H) prints it.
- **T4 (S85)** WIN3_SLOW_KEYHINGE at seed 2310 equals S85's KEYHINGE|2310 (claude/explore-H,
  explore_out/H/two_b_results.json; Xeon @ 2.10GHz) through 2400, the same fields. Asserted on a matching CPU, printed
  otherwise.
- **T5** WIN3_SLOW at seed 160 equals S43's LOCAL3_SLOW|160 (claude/outside-ideas) through 2400. This is
  test_window_gate's CHECK 135, rerun here.
- **T6 (the window)** the term's weight is 1.0 at updates 1 and 2400 and 0 at 2401, logged through the real path.
- **T7 (the gradient)** on one training batch the term's gradient is nonzero on the gate's parameters (W_in, W_g,
  gate_conv.conv_w) and the embedding, and zero on every other parameter.
- **T8 (the statistic)** eta2_pool with test_slow_start.eta2_hinge's groupings equals eta2_hinge to 1e-6
  (explore_b18_gates.check_vs_slow_start). eta2_key at k = 2 equals routing_stats' channel-0 eta^2 by key.
- **T9 (the classifier)** fail_class_v2 against v1 on every recorded unbound two-stream run of this machine's results
  files (test_window_gate Part A, test_slow_start part 0, test_recipe_scope, the others in docs/fail_class_v2.md). The two
  labels differ exactly on the runs with eta^2 by key >= 0.5 and margin >= 0.25, and those runs are listed. Asserted.
  - In the report, the same property is checked on this test's runs and printed.
- **T10 (seeds)** every arm on the machine's range, CEIL_A on its first two. Disjoint from every earlier main-line seed
  and from the other machine's range. With --also: the other file's machine, CPU and seeds.
- **T11** a worker's run is bit-identical to the same run here (WIN3_SLOW_KEYHINGE, seed 1, 1200 steps). A resume
  reproduces the first cached record of each arm through 2400.

## Runtime

Projection before the first run with pool-load timing, as test_window_gate's. Expected per machine: 42 runs on the
two-stream path, about 5 minutes each with early stop (S85: 82 runs in 52 minutes on 4 workers). Nothing is cut. Order:
CEIL_A first, then longest first. Runs are cached in two_keyterm_results.json (gitignored, written atomically after every
run), and a restart resumes from the cached records.

## Output

- Per-seed raw values first: outcome, transition, VAL cos, v2 and v1 classes, eta^2 by key and margin, the term's firings.
- Then: validity; counts; K1 pooled and per machine; K2; K3 per machine with Wilson; the reading with the caveat; bands;
  the printed diagnostics.
- RESULT sections go in the docstring, the JSON to results/<machine>/, and a row to results/README.md.
