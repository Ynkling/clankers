# Session H — report 12 (EXPLORATORY, not a result)

## S92 (explore_h_s90; the round-7 prompt's name, the key s90 kept) — the oracles at key pools 4 / 8 / 16 / 32 and 86400 updates, then the recipe at P* (S=8, P=4, k=16, conv)

### Phase 1: the perfect gates (seeds 3020-3023 per cell; budget 86400, early stop)

| memory | pool | transitions (3020-3023) | bound by 43200 (cut curve) | bound by 86400 | held-out accuracy at the end | reading |
|---|---|---|---|---|---|---|
| HEBB | 4 | 3600, 4800, 2400, 2400 | 4/4 | 4/4 | 0.978, 0.984, 0.964, 0.999 | reliable |
| HEBB | 8 | 10800, 8400, 10800, 8400 | 4/4 | 4/4 | 0.965, 0.954, 0.950, 0.981 | reliable |
| HEBB | 16 | 10800, 14400, 13200, 9600 | 4/4 | 4/4 | 0.971, 0.987, 0.971, 0.967 | reliable |
| HEBB | 32 | 19200, 14400, 18000, 18000 | 4/4 | 4/4 | 0.969, 0.950, 0.958, 0.958 | reliable |
| DELTA | 4 | 1200, 1200, 1200, 1200 | 4/4 | 4/4 | 1.000, 1.000, 1.000, 1.000 | reliable |
| DELTA | 8 | 8400, 7200, 14400, 22800 | 4/4 | 4/4 | 0.969, 0.993, 0.959, 0.991 | reliable |
| DELTA | 16 | 14400, None, 12000, None | 2/4 | 2/4 | 0.972, 0.840, 0.952, 0.830 | the oracle is the limit |
| DELTA | 32 | 19200, 25200, None, 40800 | 3/4 | 3/4 | 0.967, 0.967, 0.697, 0.967 | the oracle is the limit |

**P* = 8**; budgets {'HEBB': 43200, 'DELTA': 43200}. Runs where the 43200 reading by the cut curve differs from the run's own transition: differ in no run.

### Phase 2: the recipe at pool 8 (seeds 3024-3033, paired)

| arm | budget | BOUND ROUTED | band | Wilson 95% | failures v1 | failures v2 | transitions | splits | own pair ok | at probe < 0.2 | map 1:1 before | cap-blocked checks | held-out accuracy at the end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BASE | 43200 | 6/10 | MAJORITY | 0.31-0.83 | MERGED 4 | MERGED (2 share) 3, MERGED (3 share) 1 | 15600, 18000, 19200, 20400, 25200, 30000 | 23 | 23/23 | 10 | 0 | 15 | 0.48, 0.5, 0.75, 0.88, 0.98, 1.0, 1.0, 1.0, 1.0, 1.0 |
| DELTA_SPLIT | 43200 | 10/10 | RELIABLE | 0.72-1.00 | — | — | 24000, 24000, 26400, 26400, 28800, 30000, 33600, 34800, 40800, 40800 | 22 | 7/22 | 13 | 9 | 6 | 0.95, 0.96, 0.96, 0.97, 0.97, 0.97, 0.97, 0.98, 0.98, 0.99 |
| DELTA_NOSPLIT | 43200 | 7/10 | MAJORITY | 0.40-0.89 | STREAM-PARTIAL 3 | STREAM-PARTIAL one-to-one 3 | 18000, 18000, 19200, 25200, 28800, 30000, 33600 | -- | -- | -- | -- | -- | 0.79, 0.85, 0.95, 0.96, 0.97, 0.97, 0.98, 0.98, 0.99, 0.99 |

- DELTA_NOSPLIT vs DELTA_SPLIT: 7/10 vs 10/10; DELTA_NOSPLIT only 0, DELTA_SPLIT only 3; p(DELTA_NOSPLIT>DELTA_SPLIT) 1, p(DELTA_SPLIT>DELTA_NOSPLIT) 0.125

- DELTA_SPLIT vs BASE: 10/10 vs 6/10; DELTA_SPLIT only 4, BASE only 0; p(DELTA_SPLIT>BASE) 0.0625, p(BASE>DELTA_SPLIT) 1

**Readings (fixed in explore_h_s90's docstring before any run):**

- phase 1 HEBB: **pool 4: reliable (4/4 by 43200, 4/4 by 86400); pool 8: reliable (4/4 by 43200, 4/4 by 86400); pool 16: reliable (4/4 by 43200, 4/4 by 86400); pool 32: reliable (4/4 by 43200, 4/4 by 86400)**
- phase 1 DELTA: **pool 4: reliable (4/4 by 43200, 4/4 by 86400); pool 8: reliable (4/4 by 43200, 4/4 by 86400); pool 16: the oracle is the limit (2/4 by 43200, 2/4 by 86400); pool 32: the oracle is the limit (3/4 by 43200, 3/4 by 86400)**
- phase 1, 43200 by the cut curve vs the run's transition: **differ in no run**
- P*: **P* = 8; budgets {'HEBB': 43200, 'DELTA': 43200}; the passing pools are nested**
- Hebbian (BASE): **no reading applies (BASE 6/10 at pool 8, budget 43200)**
- delta (DELTA_SPLIT): **the recipe carries to P* (DELTA_SPLIT 10/10 at pool 8, budget 43200)**
- delta split: **no reading applies (DELTA_SPLIT ahead by >= 2 pairs) (DELTA_SPLIT 10/10, DELTA_NOSPLIT 7/10; DELTA_SPLIT only 3, DELTA_NOSPLIT only 0; p(SPLIT>NOSPLIT) 0.125, p(NOSPLIT>SPLIT) 1)**
- caveat (with every reading): **the pool makes key types rare across the corpus (at K = 32, 0.0105 per type; 16: 0.021; 8: 0.042; 4: 0.084); it does not move the keys: they stay at 3j+1, are the most repeated types within a sequence (8 each; CTX 4, VAL 1-4), and each sits next to its stream's CTX token, which the window gate reads. The key-position mask is the main line's, unchanged. A reading is about rare key types on a layout with positional keys, not about unmarked key positions.**
- operating characteristics: **operating characteristics (exact): P(>= 7/10) = 0.001, 0.011, 0.172, 0.382, 0.650, 0.879, 0.987 and P(<= 3/10) = 0.879, 0.650, 0.172, 0.055, 0.011, 0.001, 0.000 at a true rate of 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9; 'split not needed' (V's model, rho 0 / 0.5 / 1) at SPLIT / NOSPLIT 0.9 / 0.9: 0.88/0.95/1.00, 0.9 / 0.7: 0.39/0.38/0.38, 0.9 / 0.5: 0.09/0.07/0.05, 0.7 / 0.7: 0.77/0.86/1.00; the moot branch 0.86/0.93/0.99 at SPLIT = NOSPLIT = 0.1 and 0.38/0.47/0.65 at 0.3; phase 1: a cell passes 4/4 with probability q^4 (0.81 at q = 0.95), both memories (q_h q_d)^4 (0.66 at 0.95 each); P* is the largest pool that passed, not the largest reliable one (with q = 0.95 everywhere, P* = 32 with probability 0.66); 'the oracle is the limit' can mean 3 of 4 (Wilson 0.30-0.95).**

*Caveat, with every reading:* the pool makes key types rare across the corpus (at K = 32, 0.0105 per type; 16: 0.021; 8: 0.042; 4: 0.084); it does not move the keys: they stay at 3j+1, are the most repeated types within a sequence (8 each; CTX 4, VAL 1-4), and each sits next to its stream's CTX token, which the window gate reads. The key-position mask is the main line's, unchanged. A reading is about rare key types on a layout with positional keys, not about unmarked key positions.

*The rules' operating characteristics (exact): P(>= 7/10) = 0.001, 0.011, 0.172, 0.382, 0.650, 0.879, 0.987 and P(<= 3/10) = 0.879, 0.650, 0.172, 0.055, 0.011, 0.001, 0.000 at a true rate of 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9; 'split not needed' (V's model, rho 0 / 0.5 / 1) at SPLIT / NOSPLIT 0.9 / 0.9: 0.88/0.95/1.00, 0.9 / 0.7: 0.39/0.38/0.38, 0.9 / 0.5: 0.09/0.07/0.05, 0.7 / 0.7: 0.77/0.86/1.00; the moot branch 0.86/0.93/0.99 at SPLIT = NOSPLIT = 0.1 and 0.38/0.47/0.65 at 0.3; phase 1: a cell passes 4/4 with probability q^4 (0.81 at q = 0.95), both memories (q_h q_d)^4 (0.66 at 0.95 each); P* is the largest pool that passed, not the largest reliable one (with q = 0.95 everywhere, P* = 32 with probability 0.66); 'the oracle is the limit' can mean 3 of 4 (Wilson 0.30-0.95).*

**Runtime:** 62 runs, 88.9 CPU-hours on Intel(R) Xeon(R) Processor @ 2.10GHz; code SHA 287fdd3941d0. **CHECKs:** all passed (2026-10-22 05:28:28 UTC, git 117ded7+dirty, 22.1 min)

**Name.** This screen is the round-7 prompt's **S92**. Its files, store and logs keep the key "s90"
(explore_h_s90.py, explore_out/H/s90_*). That key is not shared with G's S90 or F's S90-F; Referee V's fix 1.

**Which readings apply (each with the caveat and the operating characteristics above).**
- **Phase 1.**
  - Hebbian + conv perfect gate: "the oracle is reliable" at every pool, 4/4 bound by 43200 at 4, 8, 16 and 32.
    Transitions: 2400-4800, 8400-10800, 9600-14400 and 14400-19200.
  - Delta perfect gate: "reliable" at 4 (all at 1200) and 8 (7200-22800). "The oracle is the limit" at 16 (2/4: 12000
    and 14400; 3021 and 3023 ended at 0.84 and 0.83 at 86400) and at 32 (3/4: 19200, 25200, 40800; 3022 ended at
    0.70).
  - The 43200 reading by the cut curve agrees with the runs' own transitions in all 32 runs: every bound run stopped
    early within 2400 updates of its transition. The reference (pool 4) passes on both memories, and the passing pools
    are nested.
  - **P\* = 8**, with a budget of 43200 for both memories.
- **Phase 2 (pool 8, seeds 3024-3033).**
  - Hebbian: **"no reading applies"** (BASE 6/10).
  - Delta: **"the recipe carries to P\*"** (DELTA_SPLIT 10/10).
  - Delta split: **"no reading applies (DELTA_SPLIT ahead by >= 2 pairs)"**. DELTA_NOSPLIT bound 7/10; DELTA_SPLIT
    only 3, DELTA_NOSPLIT only 0, p = 0.125. So the split is not shown to be unneeded on delta at pool 8, and its
    advantage is not shown at 10 pairs either.

**Referee V's review and when its fixes landed.**
- V's screen review (claude/review-V 980498e, screen_review_s92_H.md: "fit to run phase 1, fix before phase 2") was
  written at 05:30 UTC on 22 October, before phase 1's first record. I read it at 17:00 UTC.
- By then phase 1 was complete (P\* = 8) and phase 2's first record had been saved (BASE|3026 at 16:36:45 UTC).
- **No fix was applied before phase 2's first record. All were applied after it**, in the parent and the report path
  (03f98e5):
  - fix 1: the name;
  - fix 2: "by 43200" from the curve cut at 43200;
  - fix 3: the reference and P\* = 4 names;
  - fix 4: the phase-1 OC and the nesting check;
  - fixes 5-6: the moot and cost branches of the delta-split reading, and the renamed "no reading applies" branch;
  - fix 7: the caveat and OC printed with the readings;
  - fix 9: the budget guard;
  - fix 10: the text pins, the seed-scan text, --push-each and the S89 text fixes.
- Fix 8's three CHECK rows ran by hand at 17:02 UTC, after phase 2 had started (24cc15d).
- No fix touched the child (code SHA 287fdd3941d0 throughout), so no run or record changed. Fix 2 changed no cell, and
  the phase-2 plan (pool 8, 43200 and 43200) is what the corrected phase1() gives too. The driver recomputed it that
  way when it resumed phase 2 after a container restart.

**CHECKs.**
- Before any run (22.1 min), all passed:
  - the delta recipe's converter and exact copy are test_delta_recipe's by syntax tree;
  - PoolTask(8) and PoolTask(16) are well formed;
  - S89's ORACLE|2900 and S89b's ORACLE_DELTA|2913 through 2400;
  - S89's BASE|2902 at an 86400 budget through 4800 (last check 84000);
  - X's DELTA_SPLIT|583 at pool 4 through 4800.
  - Printed, not asserted: the main line's X record WIN3_SPLIT_D8|540 (628eb7c, a 2.80 GHz CPU) through 4800, equal on
    curve, statistics, gradient norms and trigger rows.
- After phase 2 started (fix 8), all passed bit for bit on this CPU:
  - DELTA_NOSPLIT|583 at pool 4 = X's DELTA_SLOW|583 through 4800;
  - ORACLE_HEBB|2600 at pool 4 = S87a's ORACLE|2600 through 2400;
  - ORACLE_DELTA|580 at pool 4 = X's CEIL_DELTA|580 through 2400.

**Runtime and incidents.** 62 runs, 88.9 CPU-hours on a Xeon @ 2.10 GHz, 4 workers. No run failed.
- The container restarted once during each phase. The runs in flight were lost and rerun from scratch on the same CPU
  model and code SHA; no saved run changed.
- The driver did not pull (report 11's fix), and no log line was lost.

**What phase 2 shows (descriptive).**
- **Hebbian at pool 8 sits between pool 4 and pool 32.** BASE 6/10 here, against 17/20 and 9/10 at pool 4 (S87a, S86;
  exact copy) and 1/10 and 2/10 at pool 32 (S89, S89c).
  - All 23 of its firings were on labelled targets, and none fired on a one-to-one map.
  - The four failures ended MERGED (three with 2 streams sharing, one with 3) at held-out accuracy 0.48-0.88. Each had
    used its cap of 3 by 19200-26400, and 10 of the 23 firings came at probe < 0.2.
  - This is the pattern of S89: splits spent early, a late merge with none left.
- **Delta at pool 8 binds every seed with the split, and 7 without it.**
  - The three NOSPLIT failures (3027, 3028, 3029) are not merges. Each ended STREAM-PARTIAL with a one-to-one map:
    every stream on its own channel, held-out accuracy 0.85, 0.950 (rounded; the 0.95 bar not held to the end) and 0.79. Two were still at
    their maximum at the end.
  - That is the shape of the delta perfect gate's failures at pools 16 and 32: routing in place, recall short of the
    bar.
  - With the split, the same three seeds bound at 40800, 30000 and 24000.
- **The split works differently on delta.**
  - Only 7 of DELTA_SPLIT's 22 firings were on labelled targets.
  - Nine fired on a map that was already one-to-one, at probe 0.23-0.94, eight of them at 0.77 or more and late in the run: the case S75b's guard would
    stop.
  - Every one of those runs bound. On delta channels a copy onto a channel that already holds a stream did not cost a
    run here, as in test_delta_recipe on X (10 of 31 firings labelled).
  - Why the split helps the three seeds above is not separated by this screen: firings on routed maps, flat-start
    copies, or noise.
- **Delta against Hebbian, both with the split, at pool 8:** 10/10 against 6/10, DELTA_SPLIT only 4, BASE only 0, p =
  0.0625. This is printed, not a reading; the screen does not compare the memories.

**For the main line (exploratory).**
- On this layout with positional keys, both perfect gates bind reliably up to 8 key types.
- The delta perfect gate becomes the limit from 16 key types, at 86400 updates. The Hebbian + conv perfect gate stays
  reliable at 32.
- At pool 8 the delta recipe with the split carries (10/10). The Hebbian recipe does not decide (6/10).
- So the delta memory's own capacity, not the gate, bounds the delta side of this task from 16 key types. On the
  Hebbian side the recipe falls well before the perfect gate does (6/10 at 8, 1-2/10 at 32 with a reliable oracle), so
  the learned gate and its schedule are the limit there.

The two-stream spec's revision per V's review is next.
