# Screen review: Session G's S89 (`claude/explore-G` at `7ab98c1`)

*Referee V, 19 October 2026. Written before any S89 run. Reviewed: `explore_g_s89.py` (parent, cited as "s89:NN"), `explore_g_s89_child.py` ("child:NN") and the copy of `explore_f_tasks.py` that G vendored at `7ab98c1`, against F's module at `claude/explore-F` `50dc8e4` and G's round-6 prompt (`docs/reading/parallel_sessions_prompts.md` on the main line at `7c4ceb8`, l.1158-1196). "out §N" refers to a section of `screen_checks_s89_G_out.txt`, which `screen_checks_s89_G.py` writes. That script reads git objects only and trains nothing. torch is not installed here, so G's torch functions were not executed. The generator, `repeat_mask` and the largest-ratio cut were re-implemented line for line in numpy and run on 4096 sequences drawn from the same distribution, then checked against the tables F printed in `explore_out/F/s89.log`.*

**Verdict: fix first.** At `7ab98c1` the screen cannot run on either round-6 task, and the code that would compute its readings does not exist. The commit message says only "F's vendored module, its commit and the cuts" are still to come. Much more is missing:
- the parent has no code at all;
- the vendored module is the old one;
- the ceiling arm, the oracle on -allpool, and every statistic that ROUTED-BOUND needs are hard-wired to the lite vocabulary.

On -pool and -allpool these crash: the statistics at update 1200, and the key term at update 1. Until that code is committed and CHECKed, the readings in s89:39-52 are prose that no committed function computes. There is also a design problem: about a third to a half of the plausible outcomes at learned rates of 0.7-0.8 get no name for the frequency arm. None of this needs a run to fix.

**What checks out:**
- **The two hinge statistics read token ids only.** `frequent()` (child:73-77) takes `make_batch(...)[0][:, :-1]`, so the query stream and the target are both dropped. `hinge_mask` (child:80-86), `repeat_mask` (child:62-67) and the non-key branch of `term_value` (child:92-96) read no task attribute (out §2). Labels reach only the ceiling arm (`dc.key_eta`) and the statistics, which take no gradient.
- **The frequency arm is S87's.** `Term89.forward` differs from S87's `Term87.forward` only by the removed TEMP branch, a fixed LAM and a task name (ast diff, out §2). S87's `REG3_FREQHINGE|2700` record is reproduced through both run paths (child:229-234). That record exists (out §7).
- **Each switch is REG3 when off, by construction.**
  - Term89 with `t_on = False` never wraps `layer_gate`.
  - The key hinge at `lam = 0` never computes its term.
  - Each is CHECKed through 2400 (child:255-260).
- **The seeds are free.** 2840-2899 appear in no JSON/JSONL record at any of the 13 remote branch tips, nor in any JSON/JSONL/log/txt blob in the history (out §6).
- **G's own arithmetic about the repeat hinge is right**, and more honest than the prompt. The prompt says the repeat hinge selects "the keys and nothing else". It actually selects 5 CTX + 9 KEY + 2.67 VAL positions per sequence, and 48% of those are key-term positions (out §3, which agrees with child:22-26).

## 1. Does every outcome get exactly one name? (s89:39-52)

G committed no reading function: `explore_g_s89.py` is one docstring and nothing else (out §2). I implemented the prose as written and enumerated every outcome of the three arms that each question involves (BASELINE, REG3_KEYHINGE and the arm): all 888,030 count vectors over the 8 per-seed patterns at n = 20, and all 19,448 at n = 10 (out §4).

**No overlaps.**
- "survives" and "needs frequent keys" exclude each other. "needs" requires p ≤ 0.0625, which means (KH only) − (arm only) ≥ 4.
- "survives" and "clamps" exclude each other. Being discovered needs b − c ≥ 5; "clamps" needs b − c ≤ 1.
- "needs" and "clamps" can apply together: 32,397 classes at n = 20, when the commitment is below BASELINE's. That is two names at once, but each is informative.
- "the labelled hinge itself fails here" correctly suppresses everything else.

**Gaps.** Each gap is an outcome in which the question gets no name.

| gap | where | size (n = 20, KH discovered) |
|---|---|---|
| G1: frequency arm discovered but 2-3 pairs short of KH (or more, with p > 0.0625): neither "survives" nor "needs" | s89:45-46 | 10,663 classes |
| G1': frequency arm not discovered and ≥ 2 pairs short without p ≤ 0.0625 | s89:45-46 | 24,518 |
| G1'': frequency arm within 1 pair of KH but not discovered | s89:45 | 3,669 |
| G2: the repeat arm has no counterpart to "needs frequent keys", so it is named only if it replaces KH or clamps | s89:47 | 89,565 unnamed, vs 38,850 for the frequency arm |
| G3: "no reading applies" is task-level (s89:50), so when any other arm's reading applies, an unnamed arm stays unnamed | s89:50 | — |

S87's own record is a point in G1 (out §4):
- KH 20/20, FREQ 17/20, BASELINE 0/20;
- KH only 3, FREQ only 0, p = 0.125.

Under S89's rules that outcome would be unnamed whenever the repeat hinge "replaces" or anything "clamps". At the plausible rates in §5, G1 has probability 0.34-0.48.

**Ambiguities:**
- (a) "clamps (per arm)" (s89:48-49) does not say which arms carry the reading. REG3 and REG3_KEYHINGE can both meet its condition, and REG3 has no hinge. S87 applied it to the new arms only (`explore_g_s87.py:107-113`).
- (b) Nothing says what happens when a run is missing or crashed. `hp.paired` (`explore_g_rhl_parent.py:72-76`) silently intersects the seed sets, so n shrinks without a word.
- (c) **"the labelled hinge itself fails here" is a misnomer when BASELINE binds.** If BASELINE binds 16/20 or more, KH cannot be "discovered" even at 20/20, because 4 vs 0 gives p = 0.0625 (out §4). The screen would then report that the labelled hinge fails when in fact the task does not need it. H's screen has the reading this lacks: "the split is not needed here".

## 2. Is the outcome a named function, imported or reimplemented?

**ROUTED-BOUND.**
- It exists in G's committed code as `explore_g_rhl_parent.routed_bound` (`explore_g_rhl_parent.py:52`, with `routed_star` at :43 and `outcome()` at :61).
- `paired`, `beats` and `not_worse_1` are at :72, :79 and :84.
- S87's parent uses them (`explore_g_s87.py:100`).
- S89 imports none of them, because its parent has no code.
- The main line has no `routed_bound` outside V's review scripts (out §2).

So the outcome is a named function on G's branch only, and S89 neither imports nor reimplements it: it has no outcome code at all.

**`routed_star` and the new tasks.** `routed_star` reads `end["margin"]` and `end["eta_key_by_stream"]`. On the new tasks those must come from F's `pool_routing_stats`, which differs from `rand_header_routing_stats` in four lines only (keys by `task.key_mask`, key = token − key0, query stream from `stream_labels`; ast diff by hand). The S89 run path, however, still uses `rc.stats_fn` → `STATS_F = ft.rand_header_stats` (`explore_g_rhl_child.py:87`, :253-261, child:198). That calls the module-level `ft.body_positions`, whose assert (`explore_f_tasks.py:99-100`) holds in 0 of 4096 sequences on either new task (out §3).

**"discovered".** In S89 it means "beats BASELINE, p < 0.05" (s89:40). `explore_common.discovered` (`explore_common.py:241`) and `explore_g_rhl_parent.discovered` (:48) mean "bound and val_cos < 0.5". This is the clash my review of G's register draft already raised (`spec_review_register_draft.md` item 4).

**The hinge code.** The frequency term is S87's, re-typed with the edits listed in the out §2 ast diff, and tied to S87 by the reproduction. The key term is imported (`dc.key_eta`, child:91). `repeat_mask` is new.

## 3. Seeds (s89:32-37)

Out §6 checks:
- every JSON/JSONL file at the 13 remote branch tips, by full parse and by regex on `|seed|` keys, `"seed":` and `"seeds": [...]`;
- every JSON/JSONL/log/txt blob reachable from any remote branch;
- the `.py`/`.md`/`.sh` files at every tip, for 28xx seed ranges.

**No record uses 2840-2899.** The only mentions are the round-6 prompt and G's own docstring.

The tips at the final pass were the main line `ed9ca2b`, explore-F `8a24e4a`, explore-G `7ab98c1`, explore-H `42d86ae`, and the others as listed in out §6. The history scan covered 4,623 blobs, 2.44 GB.

Two non-issues:
- The CHECK runs use seed 2842 (child:253). That is an arm seed, but the runs are deterministic, so this is harmless.
- F's validity pilots and baselines use 2800-2811, which is F's block.

## 4. Do the CHECKs tie the new code to a recorded run? Off-switches?

**The tie on lite.** CHECK (1) (child:229-234) reruns `REG3_FREQHINGE|2700` through 2400 and compares it with the record:
- record: `explore_out/G/s87_results.json`, key `REG3_FREQHINGE|2700|f205db31dfed|Intel(R) Xeon(R) Processor @ 2.10GHz`;
- store sha `f205db31…`, provenance `3919ac1+dirty`, last committed `57c25c0`;
- curve [[1200, 0.2241], [2400, 0.2256]], stats at 0/1200/2400.

It does so through S87's path and through S89's path on lite. That ties Term89 to a recorded run on lite. It does not:
- tie S89's BASELINE, REG3_KEYHINGE or ORACLE_ALL paths, which are built in S89's own `run()` (child:183-214) rather than through `dc.run` as in S87, to any record;
- tie any -pool or -allpool code path to anything.

**The record lookup is CPU-bound.** `s87_record` builds the key with `ec.cpu_model()` (child:221). On any CPU other than the 2.10 GHz Xeon the CHECK raises `KeyError` instead of printing. F's container is a 2.80 GHz Xeon, and F reproduced G's S87 `BASELINE|2700` on it bit for bit (`explore_out/F/s89.log`), so a cross-CPU reproduction is feasible and should be printed and asserted, not crash.

**On the new tasks the CHECKs cannot run:**
- `TASKS` holds `lite` only (child:54), so `task_of('…pool')` raises `KeyError`.
- With the tasks added, CHECK (3) trains through `rc.stats_fn`, which crashes at the evaluation at 1200.
- CHECK (4) calls `selected()`, which calls `ft.body_positions(t, x)` (child:170) and fails its assert.

Even a fixed `selected()` would mislabel roles: child:174 uses `x < S` and `S ≤ x < S + P`. On -allpool its "ctx" catches 0.65 of the 5 CTX tokens and its "key" catches 1.22 CTX tokens and 0 keys (out §3).

**The off-switches pass where the on-arm cannot run.** REG3_KEYHINGE's off-switch passes trivially on the new tasks because `key_eta` is never called at `lam = 0`. The on-arm, by contrast, fails at update 1:
- `ft.body_positions` asserts;
- if it did not, `kid = tokens − task.S` reaches 31 (-pool) or 45 (-allpool) against `one_hot(.., task.P = 4)` (`explore_g_devices_child.py:124, 128`; out §3).

No CHECK runs the on-arm of the ceiling on a new task.

**What is absent:**
- **The vendored-blob CHECK** (s89:55-56; prompt l.1163). `checks()` contains no comparison of blob, hash or commit (out §2). At `7ab98c1`, `explore_f_tasks.py` is blob `de47463…` (14,101 bytes), which equals `a199d5e`'s. F's `50dc8e4` (and F's tip) is blob `ffe249b…` (35,605 bytes). G's copy is an exact byte prefix of F's (out §1): F's "appended only" holds, but the vendored copy is not F's module, and `RandHeaderTaskPool`, `RandHeaderTaskAllPool`, `pool_stats` and `largest_ratio_cut` are absent from it.
- **The perfect-gate gate** (s89:33-34, 60-61). It exists as `oracle_check` (child:300-302), with no rule that calls it.

**The oracle on -allpool.** `rc.make_model("ORACLE_ALL")` builds lite's ceiling arm, which uses `perfect_gate_general(tokens, ctx_tokens)` (`explore_g_rhl_child.py:84, 149`). -allpool sets `ctx_tokens` to 16 types, and F documents that the built-in gate then "fails loudly" (`explore_f_tasks.py@50dc8e4:283-285`). The validity runs on -allpool need `task.perfect_gate`, as F's pilots use.

## 5. The "survives" rule: operating characteristics (out §5)

The setup is exact, by multinomial over the 8 per-seed patterns. BASELINE rate pB is 0 (S87's lite BASELINE was 0/20) or 0.1. The ceiling KH rate is 0.9 or 0.85. The learned arm's rate is 0.9, 0.8, 0.7 or 0.5.
- "independent" means the three arms are independent per seed.
- "paired" means KH and the arm are comonotone (one arm's successes nested in the other's, as S87's FREQ in KH), with BASELINE independent.

n = 20, pK = 0.9, pB = 0 (pB = 0.1 changes "discovered" only at pA = 0.5: 0.79 instead of 0.99):

| learned rate | P(discovered) | P(survives) ind. / paired | P(needs frequent keys) ind. / paired | P(no name) ind. / paired |
|---|---|---|---|---|
| 0.9 | 1.000 | 0.79 / 1.00 | 0.02 / 0.00 | 0.19 / 0.00 |
| 0.8 | 1.000 | 0.41 / 0.39 | 0.16 / 0.13 | 0.43 / 0.48 |
| 0.7 | 1.000 | 0.15 / 0.07 | 0.41 / 0.59 | 0.44 / 0.34 |
| 0.5 | 0.994 | 0.007 / 0.001 | 0.86 / 0.98 | 0.13 / 0.015 |

At pK = 0.85, P(survives) is 0.89 / 1.00, 0.58 / 0.74, 0.28 / 0.18 and 0.02 / 0.00 for the four rates. At n = 10 (pK 0.9, pB 0), P(survives) is 0.88-1.00, 0.63-0.74, 0.38-0.39 and 0.05-0.09, and "discovered" falls to 0.62 at 0.5.

**What the rule can say.** With BASELINE near 0, "discovered" is certain for any arm that binds 5 or more of 20, so it adds nothing. The rule rests entirely on "within 1 pair". That separates "at the ceiling" from "half the ceiling":
- an arm at 0.5 survives less than 1% of the time;
- an arm at the ceiling survives 79-100% of the time.

**What it cannot say.** It cannot separate 0.9 from 0.8. An arm one tenth below the ceiling survives about 40% of the time, at 0.85 vs 0.8 up to 74%. An arm at 0.7 still survives 7-28%. Between 0.7 and 0.8 the most likely outcome is no name at all (§1, G1).

**At n = 10** (the delta arms, descriptive): an arm at 0.8 "survives" 63-74% of the time, so a 10-seed "within one pair" says almost nothing.

**Recommendation.** State these operating characteristics in the docstring. Name the middle band (fix 4), so that the S87-like result "discovered, two or three pairs short" is a reading and not a silence.

## 6. Does anything read a label it claims not to?

**Call paths (out §2):**
- **Frequency hinge:** `Term89.forward` → `term_value(kind, live, tokens, t_task)` → `hinge_mask` → `frequent(task_name)` → `s87.freq_types(task.make_batch(4096, Generator(87_000))[0][:, :-1])`. Then `(tokens.unsqueeze(-1) == sel).any(-1)` on the forward's input, and `eta2_pool(g[mask], tokens[mask], tokens.max()+1)`.
- **Repeat hinge:** `repeat_mask(tokens)` on the forward's input.
- **Task attributes:** no task attribute is read on either path. The task object is used only for its generator, with output [0] (tokens) taken and the target column dropped.
- **Labelled code:** `dc.key_eta` (ceiling only) and the statistics (`rc.stats_fn`, `probe_rh`, `acc_by_cell`, `write_order_acc`, `gc.stream_acc`), none of which takes a gradient.

**No leak found.**

**The weakness is in the CHECK, not the design.** The CHECK "token ids only" (child:236-247; S87's `explore_g_s87_child.py:83`) asserts only that the argument is a LongTensor and refuses a float. Every label tensor in this code base is a LongTensor: `make_batch`'s query stream, `stream_labels` and `key_index`. The assert therefore cannot detect the leak it is named for; only the call path does. Fix 7 makes the CHECK able to fail.

**Coincidence with the key positions** (4096 sequences, out §3; the per-role frequency ranges and both cuts agree with F's `s89.log` within sampling noise):

| task | frequency cut (largest ratio) | positions per sequence | of the key term's 8 | repeat hinge positions | of the key term's 8 |
|---|---|---|---|---|---|
| lite | 2 CTX + 4 KEY types | 14 (5 CTX, 9 KEY) | 8 | 16.67 (5 CTX, 9 KEY, 2.67 VAL) | 8 |
| -pool | 2 CTX types | 5 (CTX only) | **0** | 16.67, identical position by position to lite's | 8 |
| -allpool | **16 VAL types** | 8 (VAL only) | **0** | 16.67, identical | 8 |

**The repeat hinge does not change across tasks.** F's relabelling is one-to-one within each role and each sequence, so `repeat_mask` returns exactly lite's mask on both new tasks, at every position (out §3: True). On every task the repeat hinge covers all 8 key-term positions, and they are 48% of what it selects. It is the S87 lite hinge plus the chance value repeats, regrouped by pool ids. That is fine, but the report should say so: it is not "the keys and nothing else" (prompt l.1176).

**On -allpool the frequency hinge is a value hinge.** G's docstring (s89:24-27) puts -allpool CTX at "0.028 per type" and expects the cut to select "the 16 CTX and 16 VAL types". 5 CTX tokens over 16 types is 5/22/16 = 0.0142, not 0.028. The table is VAL 0.0227 > CTX 0.0142 > KEY 0.0128, and the largest ratio (1.60) falls after the 16 value types (out §3; F's log: "largest-ratio cut 16 types ['VAL']"). So on -allpool REG3_FREQHINGE penalises the gate's dependence on value identity at the 8 value positions. That device is neither "nothing" nor the "positional residue" the prompt expected. On -pool it is the CTX part of S87's hinge alone, grouping by CTX id, which there is the stream.

**Consequences:**
- "the frequency hinge needs frequent keys" on -allpool would credit or blame keys for an arm that touches none.
- The across-task drop (s89:51-52) for FREQHINGE compares a CTX hinge with a VAL hinge.

## 7. Anything else that makes the screen uninterpretable

1. **The rest of the run path is lite-only** (`explore_g_rhl_child.py`, `explore_g_common.py`):
   - `roles` (:186-191) → beta^R by role is wrong;
   - `layer_stats` (:218) and `probe_rh` (:286) assert;
   - `acc_by_cell` (:326) → `found` stays `None`, a `TypeError`;
   - `gc.stream_acc` (`explore_g_common.py:287`) and `ft.write_order_acc` compare CTX token ids with stream indices, which is silently wrong on -allpool.

   F's module now provides `pool_stats`, `task.body_positions`, `task.key_index` and `task.perfect_gate`. None is wired in.
2. **REG3 is run but read nowhere.** It costs 20 runs per task (s89:32), and no reading uses it, except possibly "clamps" (ambiguity (a) in §1). Either give it a reading (KH beats REG3: "the term is needed here") or state that it is descriptive.
3. **The delta rule** (s89:35-38):
   - It does not say what happens when "the labelled hinge itself fails here" but an arm is discovered.
   - `ARMS` has `REG3_KEYHINGE_D` (child:140), which S87 ran and the docstring now omits.
4. **The validity condition has no code.** "A task runs only if F's pilots read 'the task is valid' or 'needs the longer budget'" (s89:13-15) is the right condition. The `s89_tasks.json` it promises does not exist, and no code reads it.
5. **Budget.** About 5 h per task (s89:62) assumes S87's per-run times. The -pool and -allpool vocabularies are 50 and 64 rather than 22; the per-step cost is about the same (F's timings: 16.4 vs 15.4 ms/step for NOMEM). Not an issue.

## Fixes (replacement text), most important first

1. **Commit the parent's run and reading code before any S89 run**, in `explore_g_s89.py` after the docstring. Use S87's `explore_g_s87.py` structure: `ARMS` with the seed tuples 2840-2841 / 2842-2861 / 2862-2871, the validity gate, `report()` and `main()`. Replace s89:64 with:
   > "Run: python explore_g_s89.py [--report-only]. The readings are computed by report() from explore_g_rhl_parent.routed_bound, paired, beats and not_worse_1 (imported, not re-implemented); a CHECK prints report()'s readings on S87's records re-labelled as one task (BASELINE, REG3_KEYHINGE, REG3_FREQHINGE from s87_results.json) and asserts 'the frequency hinge needs frequent keys' does not apply and the frequency arm's middle-band reading (fix 4) does, as S87's report 7 read them."

2. **Vendor F's module and assert it.**
   - Replace `explore_f_tasks.py` with the blob at `claude/explore-F` `50dc8e4` (`ffe249bb644423479e869cff11d2d723ac35450b`).
   - Add to `checks()`, as its first row: `subprocess.check_output(["git", "rev-parse", "50dc8e4:explore_f_tasks.py"])` equals `git hash-object explore_f_tasks.py`, plus a second row stating that the file's first 14,101 bytes equal `a199d5e`'s blob `de47463…`.
   - Replace child:54 with: `TASKS = {"lite": rc.RHL, "pool": ft.RandHeaderTaskPool(), "allpool": ft.RandHeaderTaskAllPool()}`.

3. **Make the ceiling, the oracle and the statistics task-aware, and CHECK them on lite against records.** Replace s89:19 with:
   > "REG3_KEYHINGE the key term with the true key positions and identities: on the round-6 tasks positions task.body_positions(x) and ids task.key_index(x) grouped over task.k_pool types (explore_f_tasks@50dc8e4:354-369); on lite dc.key_eta unchanged. CHECKs: on lite the task-aware key term equals dc.key_eta bit for bit on a batch, and S87's REG3_KEYHINGE|2700 and BASELINE|2700 reproduce through 2400 through this screen's run(); on each new task the key term's positions equal task.key_mask's 8 body positions per sequence and one training forward of the on-arm runs."

   Also:
   - use `ft.pool_stats(c2.ROUTE_AT)` as the base of `stats_fn` on the new tasks;
   - make `roles`, `layer_stats`, `probe_rh`, `acc_by_cell`, `stream_acc` and `write_order_acc` read `task.role_masks`, `task.body_positions` and `task.stream_labels` (stream index, not CTX id);
   - add a CHECK that on each new task the untrained ORACLE_ALL model's `stats_fn(.., "end")` gives `eta_key_by_stream` = 1 and margin >= 0.9, so that ROUTED* holds for the perfect gate (the gate is fixed, so this needs no training);
   - make ORACLE_ALL use `task.perfect_gate` on -allpool;
   - reproduce one of F's recorded pool baselines (`BASELINE`, -pool, seed 2802, through 2400) through this screen's run path once F records it, before the learned arms.

4. **Close the naming gaps.** Replace s89:45-50 with:
   > "the frequency hinge survives — REG3_FREQHINGE discovered and (KH only) − (FREQ only) <= 1; the frequency hinge is weaker than the key hinge — REG3_FREQHINGE discovered and (KH only) − (FREQ only) >= 2 and not 'needs frequent keys'; the frequency hinge needs frequent keys — p(KH > FREQ) <= 0.0625 [on -allpool: 'needs the key positions', since the cut there selects values]; the frequency hinge does nothing here — REG3_FREQHINGE not discovered and not 'needs'; the same four for REG3_REPEATHINGE ('replaces / is weaker than / needs key identities / does nothing'); clamps — for REG3_FREQHINGE and REG3_REPEATHINGE only. Every learned arm gets exactly one of these names per task; 'no reading applies' is removed."

5. **Name the case where the task does not need the hinge.** Insert before s89:42:
   > "first: 'the task does not need the hinge' if BASELINE binds (ROUTED-BOUND) on >= n − 4 of the n paired seeds (16 of 20: REG3_KEYHINGE can then win at most 4 vs 0, p 0.0625, and cannot be discovered); then no other reading on that task."

   Then "the labelled hinge itself fails here" applies only when BASELINE binds n − 5 or fewer. Add:
   > "A run that does not complete is rerun once; if it fails again the task's readings are UNTESTED."

6. **Correct the frequency arithmetic and the across-task line.** Replace s89:24-27 from "-allpool CTX 0.028" with:
   > "-allpool CTX 0.0142 per type (5 of 22 tokens over 16 types), VAL 0.0227, KEY 0.0128: the largest ratio (1.60) cuts after the 16 VALUE types, so on -allpool REG3_FREQHINGE is a value hinge (8 VAL positions per sequence, none of the key term's), and on -pool a CTX hinge (5 CTX positions, none of the key term's); F's 50dc8e4 table agrees."

   Replace s89:51-52 with:
   > "ACROSS TASKS (descriptive): the drop of BASELINE, REG3, REG3_KEYHINGE and REG3_REPEATHINGE from -pool to -allpool on the same seed numbers; REG3_FREQHINGE is not compared across tasks (a CTX hinge on -pool, a VAL hinge on -allpool)."

7. **Make the "token ids only" CHECK able to fail.** Replace the S87-inherited row (child:236-247) with an assert inside `frequent()` and `hinge_mask()`: the argument has shape `(n, task.L - 1)`, its values are in `[0, task.vocab)`, and `torch.equal(sample, ft.freq_sample(task))` in `frequent()`. Then add a CHECK row that `frequent()` raises when it is handed `task.stream_labels(x)` or `task.key_index(x)`.

8. **Make the reproduction CPU-tolerant and printed.** Replace child:221 with a lookup by the prefix `f"{arm}|{seed}|{st['meta']['sha'][:12]}|"` that returns the single matching record and prints its CPU next to `ec.cpu_model()`. The CHECK still asserts bit for bit, as F's cross-CPU reproduction of `BASELINE|2700` shows is feasible.

9. **Say which reading REG3 serves and which delta arms run.** Add to s89:32:
   > "REG3 is descriptive (the hinge's contribution: KH vs REG3 printed)."

   Replace s89:36-37 with:
   > "with BASELINE_D, REG3_D and REG3_KEYHINGE_D; no delta runs on a task whose first reading is 'fails here' or 'does not need the hinge'."

10. **State the operating characteristics** (§5) in the docstring:
    > "At 20 paired seeds with BASELINE ~0 and KH ~0.9, 'survives' fires with probability ~0.8-1.0 for an arm at the ceiling, ~0.4 at 0.8, 0.07-0.15 at 0.7, < 0.01 at 0.5 (V's screen_checks_s89_G_out.txt §5): it separates the ceiling from half of it, not from 0.8."

Commit reviewed: `claude/explore-G` `7ab98c1` (vendored module compared with `claude/explore-F` `50dc8e4`; prompt at main line `7c4ceb8`). The checks are in `screen_checks_s89_G.py`, with output in `screen_checks_s89_G_out.txt`.
