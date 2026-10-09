### S59 decay — counts (BOUND; Wilson 95%; band)

| arm | n | BOUND | transitions: median (all) | final acc median | write order acc last / not-last (median) |
|---|---|---|---|---|---|
| ORACLE_FIXED | 10/10 | 10/10 [0.72, 1.00] RELIABLE | 3600 (2400, 2400, 2400, 2400, 3600, 3600, 3600, 4800, 4800, 7200) | 1.00 | 1.00 / 1.00 |
| ORACLE_ROUTED | 10/10 | 10/10 [0.72, 1.00] RELIABLE | 1200 (1200, 1200, 1200, 1200, 1200, 1200, 2400, 2400, 2400, 2400) | 1.00 | 1.00 / 1.00 |
| ORACLE_GDN | 10/10 | 10/10 [0.72, 1.00] RELIABLE | 2400 (1200, 1200, 2400, 2400, 2400, 2400, 2400, 3600, 3600, 4800) | 1.00 | 1.00 / 1.00 |
| SINGLE_FIXED | 10/10 | 3/10 [0.11, 0.60] MINORITY | 14400 (8400, 14400, 18000) | 0.42 | 0.42 / 0.43 |
| SINGLE_ROUTED | 10/10 | 9/10 [0.60, 0.98] RELIABLE | 10800 (4800, 7200, 7200, 8400, 10800, 14400, 15600, 16800, 27600) | 1.00 | 1.00 / 1.00 |
| SINGLE_GDN | 10/10 | 8/10 [0.49, 0.94] MAJORITY | 16200 (6000, 7200, 8400, 14400, 18000, 18000, 25200, 43200) | 0.99 | 1.00 / 0.99 |

Validity (ORACLE binds on >= 2 of 350-352):

- FIXED: 3/3 -> VALID
- ROUTED: 3/3 -> VALID
- GDN: 3/3 -> VALID

Oracle gap per seed (ORACLE acc − SINGLE acc), and gap_d − gap_FIXED:

| seed | FIXED | ROUTED | GDN | ROUTED − FIXED | GDN − FIXED |
|---|---|---|---|---|---|
| 350 | 0.720 | 0.009 | 0.005 | -0.711 | -0.715 |
| 351 | 0.354 | 0.001 | 0.008 | -0.353 | -0.346 |
| 352 | -0.007 | 0.000 | 0.010 | 0.007 | 0.017 |
| 353 | 0.758 | 0.805 | 0.002 | 0.047 | -0.755 |
| 354 | 0.569 | 0.002 | 0.494 | -0.567 | -0.075 |
| 355 | 0.024 | 0.001 | 0.445 | -0.023 | 0.420 |
| 356 | 0.821 | 0.001 | -0.009 | -0.820 | -0.830 |
| 357 | 0.000 | 0.006 | 0.011 | 0.006 | 0.011 |
| 358 | 0.577 | 0.002 | 0.002 | -0.575 | -0.575 |
| 359 | 0.709 | 0.000 | 0.000 | -0.709 | -0.709 |

READING: **it does not** (seeds with gap_d − gap_FIXED >= 0.15: ROUTED 0, GDN 1 of 10; rule >= 7; medians of gap_d − gap_FIXED: ROUTED -0.460, GDN -0.460)

Learned decay at the end (median α per layer at CTX/KEY/VAL):

- ORACLE_FIXED: L1 other 0.950/0.950/0.950, own 0.950/0.950/0.950; L2 other 0.950/0.950/0.950, own 0.950/0.950/0.950; L3 other 0.950/0.950/0.950, own 0.950/0.950/0.950
- ORACLE_ROUTED: L1 other 1.000/1.000/1.000, own 0.932/0.993/0.976; L2 other 1.000/1.000/1.000, own 0.953/0.999/0.993; L3 other 1.000/1.000/1.000, own 0.954/0.999/0.996
- ORACLE_GDN: L1 other 0.990/0.995/0.988, own 0.988/0.995/0.988; L2 other 0.993/0.999/0.996, own 0.992/0.999/0.996; L3 other 0.993/0.999/0.997, own 0.992/0.999/0.997
- SINGLE_FIXED: L1 ch0 0.950/0.950/0.950; L2 ch0 0.950/0.950/0.950; L3 ch0 0.950/0.950/0.950
- SINGLE_ROUTED: L1 ch0 0.256/0.998/0.998; L2 ch0 0.484/1.000/0.998; L3 ch0 0.743/1.000/1.000
- SINGLE_GDN: L1 ch0 0.179/0.998/0.997; L2 ch0 0.612/1.000/0.998; L3 ch0 0.723/0.999/0.999

Runs: 60 ok (failed records 0), 8.45 h of run time; CPU ['Intel(R) Xeon(R) Processor @ 2.10GHz']; git ['2f1323d', '2f1323d+dirty', '4f486ad', '4f486ad+dirty', '6f2794b', '6f2794b+dirty', '93dfa18', '93dfa18+dirty', 'a6a5997', 'a6a5997+dirty', 'b77e8bb+dirty', 'e900359', 'e900359+dirty']; code SHA 55653f2b2a4e
