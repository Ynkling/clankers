# Session G — report 8 (S89: the frequency and repeat hinges where keys are not frequent; delta memory). EXPLORATORY, not a result

Branch `claude/explore-G`; Intel Xeon @ 2.10GHz, torch 2.14.0+cu130, 1 thread per run; code SHA 5d4a9b1dba33. Tasks: F's
RandHeaderTask-pool and -allpool (`explore_f_tasks.py` vendored from claude/explore-F b0c6332, blob e24d5e10, asserted).
Delta memory (beta 1, L2 keys), k = 2, no conv, 24000, SLOW, REG3 model. Seeds 2840-2841 (perfect gate) and 2842-2861
(arms), paired across arms and tasks. Spec, readings and CHECKs committed at 003a370 / db14e9b before any run (after
Referee V's "fix first" on 7ab98c1). 204/204 runs ok, none retried; 25.7 CPU-hours. The container restarted four times
during the screen; each resume re-ran all 106 CHECKs (6 × 106/106 ok) under the same code SHA and continued from the
cached runs.

## Counts (ROUTED-BOUND; Wilson 95%; band). POOL: pooled η²; ALLPOOL: within-sequence η² (the pooled form gives the same counts)

| arm (delta) | POOL ROUTED-BOUND | POOL BOUND | ALLPOOL ROUTED-BOUND | ALLPOOL BOUND | lite (S87, 2720-2729) |
|---|---|---|---|---|---|
| ORACLE_D (perfect gate; validity) | 2/2 (2400) | 2/2 | 2/2 (3600) | 2/2 | – |
| BASELINE_D | **0/20** [0.00, 0.16] NEVER | 14/20 (unrouted, margin ~0.45) | **0/20** NEVER | 0/20 | 0/10 |
| REG3_D (descriptive) | **16/20** [0.58, 0.92] MAJORITY | 16/20 | **0/20** NEVER | 0/20 | 0/10 |
| REG3_KEYHINGE_D (key positions + ids) | **20/20** [0.84, 1.00] RELIABLE | 20/20 | **0/20** NEVER | 2/20 (key split) | 10/10 |
| REG3_FREQHINGE_D (pool: a CTX hinge; allpool: a value hinge) | **7/20** [0.18, 0.57] MINORITY | 8/20 | **0/20** NEVER | 2/20 (key split) | 10/10 |
| REG3_REPEATHINGE_D (types repeated in the sequence) | **20/20** [0.84, 1.00] RELIABLE | 20/20 | **0/20** NEVER | 2/20 (key split) | – |

Transitions (median, range): pool KH 6000 (4800-8400), REPEAT 6000 (4800-10800), REG3 4800 (3600-9600), FREQ 7800
(6000-22800). Unbound classes: pool REG3 KEY 4; FREQ KEY 6, OTHER 3, STREAM-PARTIAL 3; BASELINE STREAM-PARTIAL 6. Allpool:
KEY 12-19 per arm, MERGED 2-5.

## Readings (pre-fixed; computed by explore_g_s89.readings)

| task | first reading | REG3_FREQHINGE_D | REG3_REPEATHINGE_D |
|---|---|---|---|
| POOL | KH discovered (20 / 0 vs BASELINE, p 9.5e-7) | **"the frequency hinge needs frequent keys"** (discovered, 7 / 0, p 0.0078; KH only 13, FREQ only 0, p(KH >) 1.2e-4) | **"the repeat hinge replaces the key hinge"** (20 / 0 vs BASELINE; KH only 0, REPEAT only 0) |
| ALLPOOL | **"the labelled hinge itself fails here"** (KH 0/20 = BASELINE 0/20) | no reading (suppressed) | no reading (suppressed) |

- "clamps": does not apply to either hinge arm on POOL (arm only 7 and 20 vs BASELINE only 0).
- REG3 (descriptive): KH vs REG3_D on POOL 4 / 0, p 0.0625; REPEAT vs REG3_D 4 / 0, p 0.0625. REG3_D vs FREQ 10 / 1,
  p 0.0059.
- Across tasks (descriptive, same seed numbers): BASELINE 0 → 0; REG3 16 → 0 (16 / 0, p 1.5e-5); KH 20 → 0 (p 9.5e-7);
  REPEAT 20 → 0 (p 9.5e-7). FREQ is not compared: it is a CTX hinge on POOL and a value hinge on ALLPOOL.
- **ALLPOOL caveat (pre-fixed).** A gate that reads token identity can route at most 0.533 of ALLPOOL sequences by a fixed
  CTX-type → channel map; the rest needs a comparison of the two contexts within the sequence. So this null does not
  separate "keys are not frequent" from "contexts have no fixed identity".

## What happened (exploratory)

- **POOL: renaming the keys makes the task easier for the register gate, not harder.** REG3 with no hinge routes and binds
  16/20 here, against 0/10 on lite (S87) and 2/20 on Hebbian lite (S82). The key hinge and the repeat hinge reach 20/20,
  four pairs above REG3 (p 0.0625 each). One reading consistent with this, not tested: with 4 keys drawn from 32 types,
  each key type is rare, so a gate that splits by key identity gains less than on lite. Both hinges' runs look alike:
  the term fires 100-120 times in 1-2400 and holds the commitment at ~0.1 to 2400, β^R at CTX rises from 0.14 (2400) to
  0.5-0.65, and all three layers route (margin 1.00, contingency 1.0/0.0 at KEY and VAL, ~0.5 at CTX).
- **The repeat hinge does the key hinge's work on POOL without labels.** It selects every repeated type in the sequence:
  5 CTX + 9 KEY + ~2.6 VAL positions, all 8 key-term positions among them. It reads no frequency table and no role. Its
  record is position for position the key hinge's (20/20, same transitions). Given REG3's 16/20, the hinge (either one)
  adds 4 pairs.
- **The frequency hinge hurts on POOL.** The cut there selects only the 2 CTX types, so the term penalises the gate's
  dependence on CTX identity at the CTX positions, that is, on the stream at the stream tokens. It routes and binds 7/20,
  below REG3's 16/20 (10 / 1). The commitment stays low past the window (0.08 at 2400, 0.21 at 3600, against ~0.97 for
  KH and REPEAT), β^R at CTX separates late (0.22 at 4800, 0.48 at the end), and 12 runs plateau at ~0.35. This is what
  S87's SELFKL did on lite: removing the stream signal where the register must learn to write.
- **ALLPOOL: no arm routes, and the failures show the cap.** Every learned arm key-splits (η² by key at KEY 0.95-1.0) or
  settles on a type-level map: REG3 5 runs, KH 5, REPEAT 2 end with 0.48-0.54 of the probe's sequences routed (margin
  ~0.5, held-out 0.72-0.92). That is the 0.533 cap of a fixed CTX-type → channel map; no run learned the relative routing
  the perfect gate is handed. The 2 BOUND runs per hinge arm bind late (18000-22800) under a key split with margin 0:
  delta memory binding without routing (descriptive).
- **POOL BASELINE binds without routing.** 14/20 BOUND, 0 ROUTED: margin ~0.45, the window gate's VAL contingency
  0.70/0.30. The delta memory completes a partial routing. ROUTED-BOUND excludes these by construction.
- Plateaus: unbound ALLPOOL runs sit at 0.39-0.49 held-out; F's report 9 puts a no-memory model with a 10-token reach at
  ~0.4 on these tasks, so accuracy there is not evidence of memory use.

## CHECKs (106, asserted before every segment, all ok)

- The vendored module (blob = F's at b0c6332; first 35,605 bytes = 50dc8e4's, 14,101 = a199d5e's); F's check_pool_task
  and check_inv_stats on both tasks; routed_bound (pooled / within-sequence view) = F's pool_outcome on a grid.
- Token ids only: frequent() equals F's largest_ratio_cut on the task's sample and raises on stream labels, key ids, a
  float, a narrower tensor, another sample; repeat positions identical on the three tasks; pool's cut CTX only, allpool's
  VAL only.
- The task-aware key term = dc.key_eta on lite bit for bit; its positions = task.key_mask's 8 on the new tasks; the
  task-aware statistics = the lite functions on lite; the perfect gate = explore_g_rhl_child's ORACLE_ALL_D on lite and
  pool (parameters, logits, gates); the untrained perfect gate reads ROUTED* on all three tasks.
- S87's REG3_FREQHINGE_D, REG3_KEYHINGE_D, BASELINE_D, REG3_D (2720) and REG3_FREQHINGE, REG3_KEYHINGE, BASELINE (2700)
  reproduce bit for bit through 2400 through this screen's run() (same CPU).
- Per new task: each hinge off = REG3_D through 2400 bit for bit; every arm end to end; gradients reach the gate only;
  schedules; SLOW lrs (the perfect gate's gate group empty).
- The readings: S87's records re-labelled as one task read "weaker than the key hinge" (Hebbian) and "survives" (delta);
  the four names exclusive and exhaustive over every count vector at n = 10 and 20.
- F's BASELINE -pool 2802 could not be reproduced: F ran no baseline.

## For the main line

- **The repeat hinge is the label-free candidate, on POOL only.** It matches the key hinge where the frequency hinge
  cannot reach the keys. But POOL's REG3 alone is 16/20, so this task shows the hinge's margin over no hinge is 4 pairs,
  not 20. On lite (where REG3_D is 0/10) the repeat hinge was not run; that is the test that would separate "replaces
  the key hinge" from "the task barely needs one".
- **ALLPOOL is not a test of hinges.** Nothing here routes it, the labelled ceiling included, and the failures sit at the
  type-level cap. A gate for ALLPOOL must compare the contexts within the sequence; the register does not learn to here.
- **Not run:** any Hebbian learned arm. F's convolution pilot (3dd5b74 / 73035d3) bound the Hebbian perfect gate + conv
  on POOL 2/2 and ALLPOOL 1/2 ("partly"); this branch's register models have no conv path. A Hebbian POOL screen would
  need one, with its own CHECKs.
