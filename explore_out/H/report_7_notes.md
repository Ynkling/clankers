**Which readings apply.** VALID: ceiling_conv bound 2/2, both at 1200.
- **KEYHINGE: "prevention works".** It discovered 20/20 against REF's 15/20 (5 vs 0, one-sided p = 0.031) and has no KEY
  failures (REF 4 under v2), so "the key split only moves" does not apply to it.
- **SLOWGATE: prevention not shown, and "the key split only moves".** It discovered 14/20 (0 vs 1), with 4 KEY failures
  under v2, the same as REF.
- **SLOWGATE_KEYHINGE: prevention not shown.** It discovered 19/20, 5 vs 1 (p = 0.109). It has 1 KEY failure, so the key
  split does not only move.

**What the timing shows**
- **The "update 0" onsets are the initial gate, not a formed split.** At initialization the gate is near-uniform, and its
  tiny variations can be structured by key: eta^2 by key at update 0 was 0.70 on 2313, 0.84 on 2320 and > 0.5 on 6
  seeds in all. That is the audit's 2.1 item: no near-uniform guard. In REF these early values decayed by update 600 on
  the seeds that discovered. A real formed split is one that is still > 0.5 at the end (the * in the table), or one
  that is first > 0.5 after update 0: REF's 25-125 on 2310, 2323 and similar.
- **SLOWGATE does not stop the split forming; it only delays it a little.** With the gate at 1e-4, REF's real splits on 2310
  and 2323 still formed: at 275 and 575 (REF: 125 and 125). A third formed on 2321 at 275, a seed REF discovered. The
  gate's learning rate is not what lets the split form early.
- **The hinge stops the split during its window, and its runs stay clean.** In every KEYHINGE run eta^2 by key fell to
  0.00-0.12 by update 25 and stayed below 0.07 to update 2400. The term fired on 1-37 batches per run.
- **After the window, a split can try to form.** It reached 0.21-0.64 at 3600 on some seeds (2317: 0.64; 2321 > 0.5). In
  KEYHINGE every such run recovered and discovered by 4800-8400: the stream routing had already formed.
- **SLOWGATE + KEYHINGE fails exactly where the two collide.** On seed 2317 the slow gate held the gate back while the
  hinge's window (updates 1-2400) passed. eta^2 by key then rose after the window (0.30 at 3600, 0.77 at 4800) and the
  run ended KEY. KEYHINGE alone discovered it. Delaying the gate pushes its learning past the hinge's protection.
- **Price of the hinge: later transitions.** Median transition: KEYHINGE 4800, SLOWGATE_KEYHINGE 6000, REF 2400.

**Classifier.** v2 differs from v1 on one S85 run, SLOWGATE 2323 (STREAM-PARTIAL -> KEY at eta^2 by key 0.5000, margin
0.500), exactly as predicted by eta^2 by key >= 0.5 with margin >= 0.25. KEYHINGE has no failures under either.

**Take-away (exploratory).** The two-stream key split is prevented, not repaired: a term that holds eta^2 by key down
during the first 2400 updates gives 20/20 here (S52: 36/40 on seeds 160-199, with three boundary cases). Holding the gate
back does not prevent the split, and combined with the hinge it moves the split past the hinge's window. A candidate for a
pre-registered test: WIN3_SLOW + the key term (S52's W2_KEYHINGE) against WIN3_SLOW, on disjoint seeds, under v2.

**Seeds.** 2310-2329, the specification's 2210-2229 shifted by 100 (see report 6).
