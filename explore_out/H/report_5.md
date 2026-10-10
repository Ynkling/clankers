# Session H — report 5 (EXPLORATORY, not a result)

## S79 explore_h_constants — S=8, P=4, k=16, conv, 43200, LOCAL3 + SLOW + exact-copy split, seeds 1320-1329

| arm | BOUND ROUTED | band | Wilson 95% | failures v1 | failures v2 | transition median (all) | splits (max/run) | labelled ok |
|---|---|---|---|---|---|---|---|---|
| BASE | 9/10 | RELIABLE | 0.60-0.98 | MERGED 1 | MERGED (2 share) 1 | 24000 (15600, 16800, 18000, 18000, 24000, 24000, 27600, 27600, 28800) | 21 (3) | 21/21 |
| INT1200 | 7/10 | MAJORITY | 0.40-0.89 | MERGED 3 | MERGED (2 share) 2, MERGED (3 share) 1 | 14400 (9600, 10800, 13200, 14400, 18000, 24000, 32400) | 28 (3) | 28/28 |
| CAP1 | 5/10 | MAJORITY | 0.24-0.76 | MERGED 5 | MERGED (2 share) 5 | 21600 (15600, 18000, 21600, 22800, 43200) | 9 (1) | 9/9 |
| CAP6 | 9/10 | RELIABLE | 0.60-0.98 | MERGED 1 | MERGED (2 share) 1 | 18000 (15600, 16800, 18000, 18000, 18000, 24000, 24000, 27600, 30000) | 29 (6) | 28/29 |
| PROBE16 | 10/10 | RELIABLE | 0.72-1.00 | — | — | 18600 (13200, 15600, 15600, 16800, 18000, 19200, 19200, 25200, 25200, 27600) | 18 (3) | 18/18 |
| THR05 | 10/10 | RELIABLE | 0.72-1.00 | — | — | 19200 (14400, 15600, 15600, 16800, 18000, 20400, 25200, 25200, 27600, 28800) | 24 (3) | 24/24 |

ORACLE (ceil8_D8, seeds 1320-1321): bound 2/2, transitions {1320: 3600, 1321: 3600} → VALID.

- INT1200 vs BASE: 7/10 vs 9/10; INT1200 only 0, BASE only 2; p(INT1200>BASE) 1, p(BASE>INT1200) 0.25
- CAP1 vs BASE: 5/10 vs 9/10; CAP1 only 0, BASE only 4; p(CAP1>BASE) 1, p(BASE>CAP1) 0.0625
- CAP6 vs BASE: 9/10 vs 9/10; CAP6 only 0, BASE only 0; p(CAP6>BASE) 1, p(BASE>CAP6) 1
- PROBE16 vs BASE: 10/10 vs 9/10; PROBE16 only 1, BASE only 0; p(PROBE16>BASE) 0.5, p(BASE>PROBE16) 1
- THR05 vs BASE: 10/10 vs 9/10; THR05 only 1, BASE only 0; p(THR05>BASE) 0.5, p(BASE>THR05) 1

**Readings (fixed in explore_h_constants' docstring before any run):**

- INT1200: **inconclusive (BASE only 2, INT1200 only 0)**
- CAP1: **sensitive (BASE only 4, CAP1 only 0)**
- CAP6: **robust (BASE only 0, CAP6 only 0)**
- PROBE16: **robust (BASE only 0, PROBE16 only 1)**
- THR05: **robust (BASE only 0, THR05 only 1)**
- CAP6_extra: **does not apply (CAP6 9 vs BASE 9; extra firings 7, labelled ok 6)**

INT1200's first split vs BASE's (per seed, update or none): 1320: 2400/19200, 1321: 2400/4800, 1322: 2400/7200, 1323: 2400/12000, 1324: 2400/16800, 1325: 9600/None, 1326: 3600/12000, 1327: 3600/4800, 1328: 2400/16800, 1329: 3600/12000 — earlier on 10 of 10.
CAP6's firings beyond the third: [(1320, 31200, True), (1320, 33600, True), (1320, 40800, True), (1321, 26400, True), (1321, 28800, False), (1323, 26400, True), (1327, 14400, True)]. v1/v2 outcome differences: .

**Runtime:** 62 runs, 33.8 CPU-hours on Intel(R) Xeon(R) Processor @ 2.10GHz; code SHA 39143d0c9c2e. **CHECKs:** all passed (2026-10-10 05:16:35 UTC, git e6db83b+dirty, 14.4 min)

**Which readings apply.** VALID: the oracle bound 2/2, both at 3600.
- **CAP1 is "sensitive"**: BASE only 4, CAP1 only 0 (one-sided p = 0.0625).
- **INT1200 is "inconclusive"**: BASE only 2, INT1200 only 0.
- **CAP6, PROBE16 and THR05 are "robust"**: 0 vs 0, 0 vs 1, 0 vs 1.
- **"The extra splits are harmless" does not apply.** CAP6 matches BASE (9 vs 9), but 1 of its 7 extra firings was not
  labelled ok (seed 1321 at 28800, a run that bound anyway at 30000). The rule required every extra firing to be ok.

**What matters, and why** (from the firing records)
- **The cap matters.** A split spent at the start is wasted:
  - With one split allowed, 4 of CAP1's 5 failures fired it at 4800-12000, while held-out accuracy was still 0.09-0.50.
    That is the flat start before any stream has bound, not a merge plateau.
  - BASE, on the same seeds, still had two splits left for the real merge later.
  - Every CAP1 split was aimed at a crowded channel (9/9 labelled ok); it was simply spent too early.
  - The fifth failure (1320) also failed under BASE, with all three splits.
- **The check interval works through the same mechanism.**
  - INT1200's first split came earlier than BASE's on 10/10 seeds, mostly at 2400-3600, at chance accuracy (0.03-0.09).
  - Its three failures (1320, 1321, 1327) had used all three splits by 7200-10800, at accuracy <= 0.47, and then merged.
  - When INT1200 succeeds it binds earlier (median transition 14400 vs 24000).
  - So the interval trades speed for the risk of exhausting the cap during the flat start.
- **The other constants do not matter here.** A 16-sequence probe (18 firings, all labelled ok) and rise < 0.05 (24, all ok)
  bound 10/10. Six splits instead of three cost nothing.
- **What a replication must state:** the cap, the first eligible check and the check interval, which together fix how many
  splits remain after the flat start. Probe size and the rise threshold can be stated loosely.

**Classifier.** v1 and v2 agree on every S79 run (all failures are MERGED (2-3 share)). No eight-stream failure here had
eta^2 by key >= 0.5 with the margin rule firing.

**Seeds.** S79 ran on 1320-1329, the specification's 1120-1129 shifted with S78's block (see report 4).
