# Session H — report 7 (EXPLORATORY, not a result)

## S85 explore_h_two_b — S=2, P=4, k=2, no conv, 24000, WIN3_SLOW path, seeds 2310-2329

| arm | DISCOVERED | band | Wilson 95% | bound, not discovered | unbound: classes v1 | unbound: classes v2 | transition median (all) | key split formed (eta^2 by key > 0.5 at KEY), median update [runs] | > 0.5 at the end |
|---|---|---|---|---|---|---|---|---|---|
| REF | 15/20 | MAJORITY | 0.53-0.89 | 0 | KEY 4, STREAM-PARTIAL 1 | KEY 4, STREAM-PARTIAL 1 | 2400 (2400, 2400, 2400, 2400, 2400, 2400, 2400, 2400, 3600, 3600, 3600, 3600, 3600, 4800, 4800) | 25 [12/20] | 4 |
| SLOWGATE | 14/20 | MAJORITY | 0.48-0.85 | 0 | KEY 3, STREAM-PARTIAL 3 | KEY 4, STREAM-PARTIAL 2 | 3600 (2400, 2400, 2400, 2400, 2400, 2400, 3600, 3600, 3600, 3600, 3600, 3600, 4800, 4800) | 0 [8/20] | 3 |
| KEYHINGE | 20/20 | RELIABLE | 0.84-1.00 | 0 | — | — | 4800 (3600, 3600, 3600, 4800, 4800, 4800, 4800, 4800, 4800, 4800, 4800, 4800, 4800, 4800, 6000, 6000, 6000, 6000, 7200, 8400) | 0 [7/20] | 0 |
| SLOWGATE_KEYHINGE | 19/20 | RELIABLE | 0.76-0.99 | 0 | KEY 1 | KEY 1 | 6000 (2400, 3600, 3600, 4800, 4800, 4800, 4800, 6000, 6000, 6000, 6000, 6000, 6000, 7200, 7200, 7200, 8400, 8400, 9600) | 0 [7/20] | 1 |

ORACLE (ceiling_conv, seeds 2310-2311): bound 2/2, transitions {2310: 1200, 2311: 1200} → VALID.

- SLOWGATE vs REF: 14/20 vs 15/20; SLOWGATE only 0, REF only 1; p(SLOWGATE>REF) 1, p(REF>SLOWGATE) 0.5
- KEYHINGE vs REF: 20/20 vs 15/20; KEYHINGE only 5, REF only 0; p(KEYHINGE>REF) 0.0312, p(REF>KEYHINGE) 1
- SLOWGATE_KEYHINGE vs REF: 19/20 vs 15/20; SLOWGATE_KEYHINGE only 5, REF only 1; p(SLOWGATE_KEYHINGE>REF) 0.109, p(REF>SLOWGATE_KEYHINGE) 0.984

**Readings (fixed in explore_h_two_b's docstring before any run):**

- SLOWGATE: **prevention not shown (SLOWGATE vs REF 0 vs 1, one-sided p = 1); the key split only moves (KEY failures v2: SLOWGATE 4, REF 4)**
- KEYHINGE: **prevention works (KEYHINGE vs REF 5 vs 0, one-sided p = 0.0312); the key split does not only move (KEY failures v2: KEYHINGE 0, REF 4)**
- SLOWGATE_KEYHINGE: **prevention not shown (SLOWGATE_KEYHINGE vs REF 5 vs 1, one-sided p = 0.109); the key split does not only move (KEY failures v2: SLOWGATE_KEYHINGE 1, REF 4)**

**When the key split forms** (first measured update with eta^2 by key at KEY positions > 0.5; '-' never; * = still > 0.5 at the end), per seed:

| seed | REF | SLOWGATE | KEYHINGE | SLOWGATE_KEYHINGE |
|---|---|---|---|---|
| 2310 | 125* | 275* | - D | - D |
| 2311 | 25 D | - D | - D | - D |
| 2312 | - D | - D | - D | - D |
| 2313 | 0 D | 0 D | 0 D | 0 D |
| 2314 | - D | - D | - D | - D |
| 2315 | 25 D | - D | - D | - D |
| 2316 | 50 D | - D | - D | - D |
| 2317 | - D | - D | 3600 D | 4800* |
| 2318 | 0* | 0 | 0 D | 0 D |
| 2319 | 50 D | - D | - D | - D |
| 2320 | 0 D | 0 D | 0 D | 0 D |
| 2321 | - D | 275* | 3600 D | - D |
| 2322 | 25 D | - D | - D | - D |
| 2323 | 125* | 575 | - D | 3600 D |
| 2324 | - D | - D | - D | - D |
| 2325 | - D | - D | - D | - D |
| 2326 | 0 | 0 | 0 D | 0 D |
| 2327 | - D | - D | - D | - D |
| 2328 | 0* | 0* | 0 D | 0 D |
| 2329 | - D | - D | - D | - D |
(D = DISCOVERED.)

**v1 / v2** on S85's unbound runs: differ on 1; exactly the runs with eta^2 by key >= 0.5 and margin >= 0.25: True. SLOWGATE|2323 STREAM-PARTIAL -> KEY (key 0.5000, margin 0.500)

**Runtime:** 82 runs, 3.5 CPU-hours on Intel(R) Xeon(R) Processor @ 2.10GHz; code SHA 957abe999f8d. **CHECKs:** all passed (2026-10-11 01:36:48 UTC, git 96e1e90+dirty, 1.2 min)

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
