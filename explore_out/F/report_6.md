# Session F — report 6 (S80 the delta recipe at two and four streams). EXPLORATORY, not a result

Branch `claude/explore-F`, readings committed at ebb381c before any run. Code SHAs: (a) d7c6a0326eae (this branch's
modules, S43's path), (b) 678c244056e9 (main-line modules at 9c5939e). Intel Xeon @ 2.10GHz, torch 2.14.0+cu130, 1 thread
per run. 48/48 runs ok, none retried. DELTA* = β 1, L2 keys (S76's choice). Seeds 2000-2019: the block appears in no
record on any branch. Full tables: `configs_tables.md`.

## Counts (Wilson 95%; band)

| configuration | arm | outcome | count | transitions (median; all) |
|---|---|---|---|---|
| (a) S=2, k=2, WIN3_SLOW, no split; seeds 2000-2009 | HEBB_A | DISCOVERED | 10/10 [0.72, 1.00] RELIABLE | 2400 (2400 ×6, 3600 ×3, 6000) |
| | DELTA_A | DISCOVERED | 10/10 [0.72, 1.00] RELIABLE | **1200** (all ten at the first evaluation) |
| (b) S=4, k=16, WIN3_SLOW16 + exact-copy split; 2010-2019 | HEBB_B | BOUND ROUTED | 10/10 [0.72, 1.00] RELIABLE | 3600 (all ten) |
| | DELTA_B | BOUND ROUTED | 10/10 [0.72, 1.00] RELIABLE | 3600 (all ten) |

Validity: the delta perfect gate bound on both seeds in each configuration, at 1200 every time (O_DELTA_A 2000-2001,
O_DELTA_B 2010-2011). The Hebbian perfect gate: (a) 2400 and 1200; (b) 1200 and 1200. Both configurations are VALID.

Paired: (a) delta only 0, Hebbian only 0; (b) 0 and 0. One-sided McNemar p = 1 both ways in both configurations.

## Readings (fixed in the docstring before any run) and which applies

- **(a) two streams: "the delta recipe keeps it"** (c − b = 0 ≤ 1). Main-line reference, counts only: WIN3_SLOW 69/80.
- **(b) four streams: "the delta recipe keeps it"** (c − b = 0 ≤ 1). Main-line reference, counts only: WIN3_SLOW16 38/40.
- **Caveat:** both arms are at the ceiling in both configurations.
  - At the main line's rates (0.86 and 0.95), a Hebbian arm loses about 1.4 and 0.5 runs in 10.
  - So these screens can only detect a large loss (the "loses it" rule, 4 vs 0), and none happened.
  - "Keeps it" here means no seed failed on delta. It does not show equivalence at the main line's power.

## Transitions and speed

- **(a)** Delta binds at 1200 on all ten seeds, Hebbian at 2400-6000. Delta is sooner on 10 of 10 pairs. The delta perfect
  gate also binds at 1200, so the learned gate costs delta nothing at two streams.
- **(b)** Both arms bind at 3600 on every seed. At the 2400 evaluation, the end of the slow phase, delta is ahead on 9 of
  10 pairs (median accuracy 0.56 vs 0.32).
- **Splits in (b):** none fired in either arm. Every run had bound before the first eligible check at 4800. So (b) ran
  WIN3_SLOW16's path bit for bit on both memories, as the CHECK shows a run without a firing does. The trigger question
  of S81 does not arise at four streams on these seeds.

## Failure classes (v1 vs v2)

- There were no failures, so no run needed classing. v1 and v2 agree trivially on S80.
- Classifier CHECK at report time, over all 367 stored Session F records (236 two-stream, 131 multichannel): v2 equals v1
  except on one record, S54(b)'s B_HEBB|323 (bound; η² by key 0.676, margin 0.362; STREAM-PARTIAL → KEY). That is
  exactly the case the rule allows (η² by key ≥ 0.5 with the margin rule firing). No violations.

## The probe at two streams (descriptive; does the delta stack carry the stream here?)

S77's probe (G's S55 logistic regression) at the end of every (a) run. It reads the stream of each body triple at its KEY
and VAL positions; chance is 0.5. Medians, KEY / VAL:

| arm | entering L1 | entering L2 | entering L3 | final | L3 at init |
|---|---|---|---|---|---|
| HEBB_A | 0.50 / 0.50 | 0.56 / 0.51 | 0.57 / 0.52 | 0.57 / 0.56 | 0.79 / 0.68 |
| DELTA_A | 0.50 / 0.50 | **0.99** / 0.69 | **1.00** / 0.58 | 1.00 / 0.62 | 0.95 / 0.72 |
| O_DELTA_A (perfect gate) | 0.50 / 0.50 | 0.91 / 0.55 | 0.98 / 0.52 | 0.99 / 0.58 | 0.98 / 0.81 |

- **At KEY positions, yes.** On all ten DELTA_A runs the stream is linearly readable from layer 2 on (L3 0.98-1.00). On
  HEBB_A it is near chance (0.52-0.60).
- **At VAL positions, no.** DELTA_A 0.54-0.65 at L3.
- S77's criterion (L3 ≥ 0.9 at both KEY and VAL) holds on 0 of 10 runs in each arm.
- The delta perfect gate shows the same pattern, so it belongs to the memory, not to the learned gate.
- Training raises the KEY signal on delta (0.95 at init → 1.00) and lowers it on Hebbian (0.79 → 0.57).
- **Post hoc (not tested here):** in the grouped layout the stream's CTX token sits just before its KEY. The delta read at
  a KEY may be dominated by that most recent write. S77's header layouts had the CTX far from the keys, and there the delta
  channel showed no such signal (L3 0.67 at KEY). So this looks like adjacency, not a stream tag carried to the value's
  write.

## CHECKs (all passed, asserted before any run)

- explore_f_delta_checks 26/26 (β = 0 equals the Hebbian memory; the recorded LOCAL3_SLOW|160 reproduced with the delta
  path disabled).
- (a):
  - HEBB_A and DELTA_A with the delta path disabled each reproduce batch 16's LOCAL3_SLOW|160 bit for bit through 2400:
    curve, statistics, decoder and gradient norms.
  - DELTA_A's model is HEBB_A's with only the memory changed (same 9 parameters, none added).
  - Groups and lrs at updates 1, 2400 and 2401.
  - G's residual loop gives each k = 2 model's own logits bit for bit.
  - The probe's positions, labels and split; G's synthetic probe check.
  - The perfect gate on delta is one-hot.
- (b):
  - HEBB_B and DELTA_B with the delta path disabled each reproduce batch 16's LOCAL3_SLOW16_A|242 bit for bit through 2400.
  - HEBB_B with the threshold at 0 equals the path without a trigger bit for bit through 7200.
  - H's copy op at k = 16.
  - The classifier copy is byte for byte ba8901a's fail_class_v2.py.
  - The v1/v2 rule on the stored records, as above.
  - The perfect gate on delta at k = 16 is one-hot.

## Runtime

4.1 h of run time (a 0.55 h, b 3.57 h), 19:24-20:41 UTC on 10 October (about 1.3 h wall on 4 workers, CHECKs included).
S81 started at 20:41.

## Anything unexpected

- The ceiling: 40/40 learned-gate runs succeeded, against the main line's 69/80 and 38/40 for the Hebbian recipes. These
  seeds may be easy, or 10 seeds are simply too few to see a failure.
- In (b) the delta runs took about 3× the Hebbian runs' wall time (about 810 s vs 267 s per run) for the same 8400 updates.
- The 1300-1359 block I shifted S76 into is also used by Session H's S78-S79 (claude/explore-H, explore_h_two: seeds
  1300-1329). H shifted to it by the same rule as I did, and both sessions found it free when they searched. The arms and
  paths differ, so no record is affected; only S76's Phase 1 seeds 1300-1303 and H's coincide.
