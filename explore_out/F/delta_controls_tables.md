### S53 delta_controls — counts (BOUND; Wilson 95%; band)

| arm | layout | BOUND | transitions: median (all) | final acc median | write order: acc last / not-last / last-written value among not-last (medians) |
|---|---|---|---|---|---|
| SINGLE_HEBB_G | grouped | 0/10 [0.00, 0.28] NEVER of 10 | -- | 0.49 | 0.53 / 0.45 / 0.55 |
| SINGLE_HEBB_H | header | 1/10 [0.02, 0.40] MINORITY of 10 | 24000 (24000) | 0.40 | 0.37 / 0.44 / 0.04 |
| SINGLE_DELTA_G | grouped | 4/10 [0.17, 0.69] MINORITY of 10 | 3600 (2400, 3600, 3600, 4800) | 0.52 | 0.53 / 0.53 / 0.44 |
| SINGLE_DELTA_H | header | 8/10 [0.49, 0.94] MAJORITY of 10 | 2400 (1200, 2400, 2400, 2400, 2400, 2400, 2400, 2400) | 1.00 | 0.99 / 1.00 / 0.00 |
| SINGLE_DELTA_L_G | grouped | 0/10 [0.00, 0.28] NEVER of 10 | -- | 0.22 | 0.22 / 0.22 / 0.25 |
| SINGLE_DELTA_L_H | header | 0/10 [0.00, 0.28] NEVER of 10 | -- | 0.32 | 0.31 / 0.32 / 0.07 |
| ORACLE_HEBB_G | grouped | 10/10 [0.72, 1.00] RELIABLE of 10 | 1200 (1200, 1200, 1200, 1200, 1200, 1200, 2400, 2400, 2400, 6000) | 1.00 | 1.00 / 1.00 / 0.00 |
| ORACLE_HEBB_H | header | 10/10 [0.72, 1.00] RELIABLE of 10 | 1200 (1200, 1200, 1200, 1200, 1200, 1200, 1200, 2400, 2400, 4800) | 1.00 | 1.00 / 1.00 / 0.00 |
| ORACLE_DELTA_G | grouped | 10/10 [0.72, 1.00] RELIABLE of 10 | 1200 (1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200) | 1.00 | 1.00 / 1.00 / 0.00 |
| ORACLE_DELTA_H | header | 10/10 [0.72, 1.00] RELIABLE of 10 | 1200 (1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200) | 1.00 | 1.00 / 1.00 / 0.00 |
| ORACLE_HEBB_P8 | header8 | 3/3 [0.44, 1.00] RELIABLE of 3 | 18000 (14400, 18000, 24000) | 1.00 | 1.00 / 1.00 / 0.00 |
| ORACLE_DELTA_P8 | header8 | 3/3 [0.44, 1.00] RELIABLE of 3 | 4800 (2400, 4800, 6000) | 1.00 | 1.00 / 1.00 / 0.00 |

Validity (perfect gate binds on >= 2 of seeds 300-302):

- grouped: ORACLE_DELTA 3/3 -> VALID (the delta reading needs this)
- grouped: ORACLE_HEBB 3/3 -> VALID (printed)
- header: ORACLE_DELTA 3/3 -> VALID (the delta reading needs this)
- header: ORACLE_HEBB 3/3 -> VALID (printed)

Paired (same seeds and batches; exact one-sided McNemar):

- grouped: SINGLE_HEBB 0/10 vs SINGLE_DELTA 4/10: SINGLE_HEBB only 0, SINGLE_DELTA only 4, p(SINGLE_HEBB > SINGLE_DELTA) = 1
- grouped: SINGLE_HEBB 0/10 vs SINGLE_DELTA_L 0/10: SINGLE_HEBB only 0, SINGLE_DELTA_L only 0, p(SINGLE_HEBB > SINGLE_DELTA_L) = 1
- grouped: ORACLE_DELTA 10/10 vs ORACLE_HEBB 10/10: ORACLE_DELTA only 0, ORACLE_HEBB only 0, p(ORACLE_DELTA > ORACLE_HEBB) = 1
- header: SINGLE_HEBB 1/10 vs SINGLE_DELTA 8/10: SINGLE_HEBB only 0, SINGLE_DELTA only 7, p(SINGLE_HEBB > SINGLE_DELTA) = 1
- header: SINGLE_HEBB 1/10 vs SINGLE_DELTA_L 0/10: SINGLE_HEBB only 1, SINGLE_DELTA_L only 0, p(SINGLE_HEBB > SINGLE_DELTA_L) = 0.5
- header: ORACLE_DELTA 10/10 vs ORACLE_HEBB 10/10: ORACLE_DELTA only 0, ORACLE_HEBB only 0, p(ORACLE_DELTA > ORACLE_HEBB) = 1
- header P=8 (48000): ORACLE_DELTA 3/3 vs ORACLE_HEBB 3/3: delta only 0, hebb only 0, p = 1

READING R1: **neither** (SINGLE_DELTA 4/10 grouped, 8/10 header; SINGLE_HEBB 0/10, 1/10; rule: SINGLE_DELTA <= 1/10 on both and SINGLE_HEBB >= 3/10 on one, both layouts valid)

READING R2: **neither** (seeds where ORACLE_DELTA binds and ORACLE_HEBB does not: 0 of 3; rule >= 2)

Write order (descriptive; median run per layout; held-out 2048):

- grouped SINGLE_DELTA: acc last 0.53, not-last 0.53, last-written value among not-last 0.44; by query stream 0.52 / 0.52 -> does not
- grouped SINGLE_DELTA_L: acc last 0.22, not-last 0.22, last-written value among not-last 0.25; by query stream 0.21 / 0.22
- grouped SINGLE_HEBB: acc last 0.53, not-last 0.45, last-written value among not-last 0.55; by query stream 0.50 / 0.48
- header SINGLE_DELTA: acc last 0.99, not-last 1.00, last-written value among not-last 0.00; by query stream 1.00 / 1.00 -> does not
- header SINGLE_DELTA_L: acc last 0.31, not-last 0.32, last-written value among not-last 0.07; by query stream 0.32 / 0.32
- header SINGLE_HEBB: acc last 0.37, not-last 0.44, last-written value among not-last 0.04; by query stream 0.40 / 0.41

Per seed (BOUND transition or final acc; write order last/not-last):

| seed | SINGLE_HEBB_G | SINGLE_HEBB_H | SINGLE_DELTA_G | SINGLE_DELTA_H | SINGLE_DELTA_L_G | SINGLE_DELTA_L_H | ORACLE_HEBB_G | ORACLE_HEBB_H | ORACLE_DELTA_G | ORACLE_DELTA_H |
|---|---|---|---|---|---|---|---|---|---|---|
| 300 | 0.49 (0.54/0.44) | 0.46 (0.40/0.54) | 0.50 (0.54/0.45) | B@2400 (0.99/1.00) | 0.22 (0.22/0.23) | 0.31 (0.32/0.30) | B@2400 (1.00/1.00) | B@4800 (1.00/1.00) | B@1200 (1.00/1.00) | B@1200 (1.00/1.00) |
| 301 | 0.50 (0.51/0.48) | 0.44 (0.35/0.53) | 0.49 (0.44/0.54) | B@2400 (0.98/0.99) | 0.21 (0.22/0.21) | 0.32 (0.31/0.32) | B@1200 (1.00/1.00) | B@1200 (1.00/1.00) | B@1200 (1.00/1.00) | B@1200 (1.00/1.00) |
| 302 | 0.50 (0.51/0.49) | 0.34 (0.35/0.32) | 0.50 (0.49/0.51) | 0.33 (0.34/0.32) | 0.21 (0.20/0.22) | 0.33 (0.34/0.31) | B@2400 (1.00/1.00) | B@2400 (1.00/1.00) | B@1200 (1.00/1.00) | B@1200 (1.00/1.00) |
| 303 | 0.49 (0.56/0.43) | 0.33 (0.33/0.34) | 0.39 (0.39/0.38) | 0.55 (0.55/0.55) | 0.23 (0.25/0.21) | 0.31 (0.32/0.31) | B@1200 (1.00/1.00) | B@1200 (1.00/1.00) | B@1200 (1.00/1.00) | B@1200 (1.00/1.00) |
| 304 | 0.50 (0.58/0.42) | B@24000 (0.95/1.00) | 0.52 (0.52/0.53) | B@2400 (0.99/0.99) | 0.22 (0.22/0.22) | 0.33 (0.32/0.33) | B@1200 (1.00/1.00) | B@1200 (1.00/1.00) | B@1200 (1.00/1.00) | B@1200 (1.00/1.00) |
| 305 | 0.48 (0.46/0.51) | 0.37 (0.41/0.32) | B@4800 (1.00/1.00) | B@2400 (1.00/1.00) | 0.22 (0.22/0.22) | 0.33 (0.34/0.33) | B@1200 (1.00/1.00) | B@1200 (1.00/1.00) | B@1200 (1.00/1.00) | B@1200 (1.00/1.00) |
| 306 | 0.49 (0.57/0.42) | 0.94 (0.88/0.99) | B@3600 (1.00/1.00) | B@2400 (1.00/1.00) | 0.22 (0.22/0.21) | 0.32 (0.31/0.34) | B@1200 (1.00/1.00) | B@1200 (1.00/1.00) | B@1200 (1.00/1.00) | B@1200 (1.00/1.00) |
| 307 | 0.48 (0.54/0.43) | 0.32 (0.32/0.32) | B@2400 (1.00/1.00) | B@1200 (1.00/1.00) | 0.22 (0.21/0.23) | 0.31 (0.31/0.32) | B@2400 (1.00/1.00) | B@2400 (1.00/1.00) | B@1200 (1.00/1.00) | B@1200 (1.00/1.00) |
| 308 | 0.49 (0.52/0.47) | 0.82 (0.72/0.93) | 0.51 (0.52/0.49) | B@2400 (1.00/1.00) | 0.22 (0.22/0.22) | 0.22 (0.23/0.21) | B@1200 (1.00/1.00) | B@1200 (1.00/1.00) | B@1200 (1.00/1.00) | B@1200 (1.00/1.00) |
| 309 | 0.49 (0.53/0.46) | 0.31 (0.33/0.29) | B@3600 (1.00/1.00) | B@2400 (0.99/1.00) | 0.21 (0.21/0.20) | 0.32 (0.31/0.33) | B@6000 (1.00/1.00) | B@1200 (1.00/1.00) | B@1200 (1.00/1.00) | B@1200 (1.00/1.00) |

P = 8 header per seed: 300: HEBB B@24000, DELTA B@6000; 301: HEBB B@14400, DELTA B@2400; 302: HEBB B@18000, DELTA B@4800

SINGLE_DELTA_L: mean β at CTX / KEY / VAL per layer at the end (median over runs):

- grouped: layer 1 0.30/0.00/0.14; layer 2 0.56/0.00/0.12; layer 3 0.57/0.01/0.16
- header: layer 1 0.65/0.00/0.35; layer 2 0.49/0.00/0.10; layer 3 0.64/0.00/0.03

Oracle routing check (end, median): ORACLE_HEBB_G margin 1.00; ORACLE_DELTA_G margin 1.00; ORACLE_HEBB_H margin 1.00; ORACLE_DELTA_H margin 1.00

Runs: 106 ok (failed records 0), 7.67 h of run time; CPU ['Intel(R) Xeon(R) Processor @ 2.10GHz']; git ['147e5cf', '147e5cf+dirty', '91a3d26', '91a3d26+dirty', '93ddb60', '9ddda81', '9ddda81+dirty', 'bb906e5', 'bb906e5+dirty', 'bd1c954', 'bd1c954+dirty', 'da4f065']; code SHA 429221e60279
