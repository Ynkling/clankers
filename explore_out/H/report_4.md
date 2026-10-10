# Session H — report 4 (EXPLORATORY, not a result)

## S78 explore_h_two — S=2, P=4, k=2 (K4: k=4), no conv, 24000, WIN3_SLOW path, seeds 1300-1319

| arm | outcome | count | band | Wilson 95% | bound, not successful | unbound: classes v1 | unbound: classes v2 | transition median (all) | splits fired (runs) |
|---|---|---|---|---|---|---|---|---|---|
| REF | DISCOVERED | 15/20 | MAJORITY | 0.53-0.89 | 0 | KEY 3, STREAM-PARTIAL 2 | KEY 4, STREAM-PARTIAL 1 | 2400 (2400, 2400, 2400, 2400, 2400, 2400, 2400, 2400, 3600, 3600, 3600, 3600, 3600, 3600, 4800) | — |
| SPLIT2 | DISCOVERED | 16/20 | MAJORITY | 0.58-0.92 | 0 | KEY 3, STREAM-PARTIAL 1 | KEY 4 | 3000 (2400, 2400, 2400, 2400, 2400, 2400, 2400, 2400, 3600, 3600, 3600, 3600, 3600, 3600, 4800, 6000) | 12 (5) |
| SPLIT2_NOISE | DISCOVERED | 17/20 | MAJORITY | 0.64-0.95 | 0 | KEY 1, STREAM-PARTIAL 2 | KEY 3 | 3600 (2400, 2400, 2400, 2400, 2400, 2400, 2400, 2400, 3600, 3600, 3600, 3600, 3600, 3600, 4800, 6000, 6000) | 10 (5) |
| K4 | BOUND ROUTED | 15/20 | MAJORITY | 0.53-0.89 | 0 | STREAM-PARTIAL one-to-one 3, non-stream KEY 2 | STREAM-PARTIAL one-to-one 3, non-stream KEY 2 | 2400 (2400, 2400, 2400, 2400, 2400, 2400, 2400, 2400, 2400, 2400, 2400, 3600, 3600, 3600, 4800) | — |

ORACLE (ceiling_conv, seeds 1300-1301): bound 2/2, transitions {1300: 1200, 1301: 1200} → VALID.

| x vs y | x | y | x only | y only | p(x>y) | p(y>x) |
|---|---|---|---|---|---|---|
| SPLIT2 vs REF | 16/20 | 15/20 | 1 | 0 | 0.5 | 1 |
| SPLIT2_NOISE vs REF | 17/20 | 15/20 | 2 | 0 | 0.25 | 1 |
| K4 vs REF | 15/20 | 15/20 | 3 | 3 | 0.656 | 0.656 |
| SPLIT2_NOISE vs SPLIT2 | 17/20 | 16/20 | 1 | 0 | 0.5 | 1 |
| K4 vs SPLIT2 | 15/20 | 16/20 | 2 | 3 | 0.812 | 0.5 |

**Readings (fixed in explore_h_two's docstring before any run):**

- R1: **does not apply (SPLIT2 vs REF 1 vs 0, one-sided p = 0.5)**
- R2: **does not apply (SPLIT2_NOISE vs SPLIT2 1 vs 0)**
- R3: **does not apply (K4 vs SPLIT2 2 vs 3)**
- R4: **neither device (SPLIT2 vs REF 1 vs 0, p = 0.5; SPLIT2_NOISE vs REF 2 vs 0, p = 0.25; K4 vs REF 3 vs 3, p = 0.656)**

**Splits** (22 firings). eta^2 of the read gate at KEY positions by key / by stream, and at VAL by key / by stream, at the firing check -> right after the copy -> at the next check; run's end tag:

| arm | seed | update | KEY key/stream before | after | next | VAL key/stream before | after | next | end |
|---|---|---|---|---|---|---|---|---|---|
| SPLIT2 | 1309 | 4800 | 0.20/0.60 | 0.00/0.00 | 0.00/1.00 | 0.20/0.60 | 0.00/0.00 | 0.00/1.00 | DISCOVERED |
| SPLIT2 | 1311 | 7200 | 1.00/0.00 | 0.00/0.00 | 1.00/0.00 | 1.00/0.00 | 0.00/0.00 | 1.00/0.00 | KEY |
| SPLIT2 | 1311 | 14400 | 1.00/0.00 | 0.00/0.00 | 1.00/0.00 | 1.00/0.00 | 0.00/0.00 | 1.00/0.00 | KEY |
| SPLIT2 | 1311 | 21600 | 1.00/0.00 | 0.00/0.00 | -- | 1.00/0.00 | 0.00/0.00 | -- | KEY |
| SPLIT2 | 1313 | 14400 | 1.00/0.00 | 0.00/0.00 | 1.00/0.00 | 1.00/0.00 | 0.00/0.00 | 1.00/0.00 | KEY |
| SPLIT2 | 1313 | 21600 | 1.00/0.00 | 0.00/0.00 | -- | 1.00/0.00 | 0.00/0.00 | -- | KEY |
| SPLIT2 | 1315 | 7200 | 1.00/0.00 | 0.00/0.00 | 1.00/0.00 | 1.00/0.00 | 0.00/0.00 | 1.00/0.00 | KEY |
| SPLIT2 | 1315 | 14400 | 1.00/0.00 | 0.00/0.00 | 1.00/0.00 | 1.00/0.00 | 0.00/0.00 | 1.00/0.00 | KEY |
| SPLIT2 | 1315 | 19200 | 1.00/0.00 | 0.00/0.00 | 1.00/0.00 | 1.00/0.00 | 0.00/0.00 | 1.00/0.00 | KEY |
| SPLIT2 | 1317 | 4800 | 0.50/0.25 | 0.00/0.00 | 0.51/0.25 | 0.50/0.25 | 0.00/0.00 | 0.51/0.25 | STREAM-PARTIAL |
| SPLIT2 | 1317 | 12000 | 0.50/0.25 | 0.00/0.00 | 0.51/0.24 | 0.50/0.25 | 0.00/0.00 | 0.51/0.24 | STREAM-PARTIAL |
| SPLIT2 | 1317 | 19200 | 0.50/0.25 | 0.00/0.00 | 0.50/0.25 | 0.50/0.25 | 0.00/0.00 | 0.50/0.25 | STREAM-PARTIAL |
| SPLIT2_NOISE | 1309 | 4800 | 0.20/0.60 | 0.43/0.55 | 0.00/1.00 | 0.20/0.60 | 0.52/0.46 | 0.00/1.00 | DISCOVERED |
| SPLIT2_NOISE | 1311 | 7200 | 1.00/0.00 | 0.89/0.01 | 0.75/0.07 | 1.00/0.00 | 0.67/0.02 | 0.76/0.06 | STREAM-PARTIAL |
| SPLIT2_NOISE | 1311 | 12000 | 0.74/0.07 | 0.81/0.16 | 0.74/0.07 | 0.75/0.06 | 0.85/0.12 | 0.75/0.07 | STREAM-PARTIAL |
| SPLIT2_NOISE | 1311 | 19200 | 0.73/0.07 | 0.79/0.10 | 0.74/0.07 | 0.74/0.07 | 0.81/0.09 | 0.75/0.06 | STREAM-PARTIAL |
| SPLIT2_NOISE | 1313 | 14400 | 1.00/0.00 | 0.76/0.12 | 0.56/0.23 | 1.00/0.00 | 0.83/0.02 | 0.59/0.21 | STREAM-PARTIAL |
| SPLIT2_NOISE | 1313 | 19200 | 0.50/0.25 | 0.61/0.30 | 0.51/0.25 | 0.53/0.23 | 0.80/0.14 | 0.53/0.24 | STREAM-PARTIAL |
| SPLIT2_NOISE | 1315 | 7200 | 1.00/0.00 | 0.98/0.00 | 1.00/0.00 | 1.00/0.00 | 0.90/0.00 | 1.00/0.00 | KEY |
| SPLIT2_NOISE | 1315 | 12000 | 1.00/0.00 | 0.95/0.01 | 1.00/0.00 | 1.00/0.00 | 0.93/0.00 | 1.00/0.00 | KEY |
| SPLIT2_NOISE | 1315 | 16800 | 1.00/0.00 | 1.00/0.00 | 1.00/0.00 | 1.00/0.00 | 1.00/0.00 | 1.00/0.00 | KEY |
| SPLIT2_NOISE | 1317 | 4800 | 0.50/0.25 | 0.71/0.28 | 0.00/1.00 | 0.50/0.25 | 0.68/0.30 | 0.00/1.00 | DISCOVERED |

**v1 / v2** on S78's unbound runs: differ on 4; exactly the runs with eta^2 by key >= 0.5 and margin >= 0.25: True. REF|1317 STREAM-PARTIAL -> KEY (key 0.5000, margin 0.562); SPLIT2_NOISE|1311 STREAM-PARTIAL -> KEY (key 0.7365, margin 0.267); SPLIT2_NOISE|1313 STREAM-PARTIAL -> KEY (key 0.5068, margin 0.449); SPLIT2|1317 STREAM-PARTIAL -> KEY (key 0.5021, margin 0.552)

**Runtime:** 82 runs, 3.2 CPU-hours on Intel(R) Xeon(R) Processor @ 2.10GHz; code SHA 5af849d1455f. **CHECKs:** all passed (2026-10-10 04:11:34 UTC, git 3997ae2, 1.6 min)

**Which reading applies: R4 "neither device".** No arm beats REF. SPLIT2 vs REF is 1 vs 0, SPLIT2_NOISE vs REF 2 vs 0, and K4
vs REF 3 vs 3; the best one-sided p is 0.25. R1, R2 and R3 do not apply. All four arms sit in the MAJORITY band (15-17/20),
with overlapping Wilson intervals.

**What the copy does at k = 2** (from the split table)
- **The exact copy erases the gate's partition, and the same split comes back.** After W_g[c0] := W_g[c*] the two rows are
  equal, so the gate is uniform: every eta^2 is 0.00 right after the copy. By the next check, 2400 updates later, the KEY
  split had re-formed exactly (KEY/stream eta^2 1.00/0.00) after every firing on seeds 1311, 1313 and 1315 (8 of 8
  firings). On 1317 the half-key pattern (0.50/0.25) re-formed after all 3 firings.
- **Why the same split returns.** The copy leaves W_g's Adam moments untouched. At k = 2 the two identical rows differ only
  in those moments, and the moments still point along the old split. This is our reading of the mechanism; we did not
  measure the moments. The one rescue is 1309, where the gate was not yet a key split at the firing (0.20/0.60): after the
  copy the streams separated (0.00/1.00) and the run discovered.
- **Noise + reset (the main line's form) breaks it more often, but not cleanly.** The same 5 runs fired in both arms. It
  rescued 1309 and 1317 (both DISCOVERED). On 1311 and 1313 it turned a full key split into a partial one (key eta^2
  0.5-0.75, v1 STREAM-PARTIAL, v2 KEY). On 1315 the key split held through 3 firings. So at k = 2 symmetry-breaking helps a
  little (+2 vs REF, +1 vs SPLIT2), well short of R2's 4 vs 0.
- **Spare channels (K4) are not better on these seeds.** K4 is 15/20, the same count as REF, with 3 vs 3 discordant. Its
  failures are STREAM-PARTIAL one-to-one 3 and non-stream KEY 2 under both v1 and v2. S52's 36/40 on seeds 160-199 is the
  same arm on other seeds.
- **REF here: 15/20.** The main line's WIN3_SLOW discovered 69/80 (86%). 15/20 (Wilson 0.53-0.89) is consistent with that.
  The failures are key splits, as the main line reports (v1 KEY 3 + STREAM-PARTIAL 2; v2 KEY 4 + 1).

**Classifier.** On S78's 17 unbound runs v2 differs from v1 on 4: REF 1317, SPLIT2 1317, SPLIT2_NOISE 1311 and 1313. Each is
STREAM-PARTIAL -> KEY, and they are exactly the runs with eta^2 by key >= 0.5 and margin >= 0.25 (the property holds). On
this branch's 29 recorded unbound two-stream runs (the pre-run CHECK), v2 differed on 6, again exactly the predicted ones.
Two of them are S52's W2_K4 MERGED runs (169, 196), which v2 calls non-stream KEY.

**Seeds and setup.** The block moved from 1100-1159 to 1300-1359:
- 1100-1149 are test_router_reliability's trial seeds in results/X and results/L.
- 1200-1259 hold restart-attempt seeds in stream_recipe and curriculum_confirm, and appear in explore-G's logs.
- 1300-1359 appears nowhere in explore_out/ or results/ on any branch.

The other choices:
- REF = S43's LOCAL3_SLOW path, which test_window_gate's CHECK 135 equates with WIN3_SLOW. It reproduced S43's record bit
  for bit through 2400.
- K4 = S52's W2_K4 code, identical to claude/outside-ideas (same blob). It reproduced its record bit for bit through 2400.

The first driver attempt crashed before any run, on a leftover line in my CHECK code; the log is kept as s78_check_crashed.log.

**Take-away (exploratory).** At k = S the KEYMASS split has no idle channel to copy onto. An exact copy only resets the gate
to uniform, and the same key split comes back. The copy works at eight streams because there c0 is empty, so the copy
creates a near-tie between a crowded channel and a free one. Two-stream key splits need another device.
