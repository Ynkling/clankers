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
