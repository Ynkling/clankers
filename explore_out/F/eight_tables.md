### S76 eight streams — Phase 1, the oracle (seeds 1300-1303; BOUND; Wilson 95%)

| arm | BOUND | per seed: transition (or final acc / DIVERGED) | acc @12000 / @24000 (median) | meets the 2x mark |
|---|---|---|---|---|
| O_HEBB | 4/4 [0.51, 1.00] RELIABLE | 1300 B@2400; 1301 B@3600; 1302 B@2400; 1303 B@6000 | 0.99 / 0.99 | -- |
| O_B1_L2 | 4/4 [0.51, 1.00] RELIABLE | 1300 B@1200; 1301 B@1200; 1302 B@1200; 1303 B@1200 | 1.00 / 1.00 | 4/4 |
| O_B025_L2 | 4/4 [0.51, 1.00] RELIABLE | 1300 B@2400; 1301 B@2400; 1302 B@2400; 1303 B@2400 | 1.00 / 1.00 | 4/4 |
| O_B1_RAW | 3/4 [0.30, 0.95] MAJORITY | 1300 DIVERGED; 1301 B@16800; 1302 B@4800; 1303 B@2400 | 1.00 / 1.00 | 2/4 |
| O_B025_RAW | 4/4 [0.51, 1.00] RELIABLE | 1300 B@7200; 1301 B@1200; 1302 B@2400; 1303 B@1200 | 1.00 / 1.00 | 3/4 |

READING Phase 1: **neither** (seeds meeting the 2x mark: O_B1_L2 4/4, O_B025_L2 4/4, O_B1_RAW 2/4, O_B025_RAW 3/4; within 2x = >= 3/4)

DELTA* = {'arm': 'O_B1_L2', 'beta': 1.0, 'normalize': True, 'key': ['1200.0', '-4', '1200.0', '0']}

Accuracy curves (held-out, every 2400):

- O_HEBB 1300: 0.98 1.00
- O_HEBB 1301: 0.88 1.00
- O_HEBB 1302: 0.97 0.99
- O_HEBB 1303: 0.77 0.93 0.99
- O_B1_L2 1300: 1.00
- O_B1_L2 1301: 1.00
- O_B1_L2 1302: 1.00
- O_B1_L2 1303: 1.00
- O_B025_L2 1300: 1.00 1.00
- O_B025_L2 1301: 1.00 1.00
- O_B025_L2 1302: 1.00 0.95
- O_B025_L2 1303: 1.00 1.00
- O_B1_RAW 1301: 0.56 0.74 0.87 0.94 0.98 0.76 0.99 0.99
- O_B1_RAW 1302: 0.82 0.98 1.00
- O_B1_RAW 1303: 1.00 1.00
- O_B025_RAW 1300: 0.76 0.92 0.99 1.00
- O_B025_RAW 1301: 1.00
- O_B025_RAW 1302: 0.98 1.00
- O_B025_RAW 1303: 1.00

### S76 Phase 2, the recipe (seeds 1310-1319; BOUND ROUTED; Wilson 95%)

| arm | n | BOUND ROUTED | BOUND | outcomes | MERGED | splits (total; runs; targets ok) | transitions of BOUND ROUTED |
|---|---|---|---|---|---|---|---|
| HEBB_SPLIT | 10/10 | 10/10 [0.72, 1.00] RELIABLE | 10/10 [0.72, 1.00] RELIABLE | {'BOUND ROUTED': 10} | 0 | 22; 10; 22/22 | 23400.0 (13200, 20400, 22800, 22800, 22800, 24000, 25200, 28800, 36000, 40800) |
| DELTA_SPLIT | 10/10 | 9/10 [0.60, 0.98] RELIABLE | 9/10 [0.60, 0.98] RELIABLE | {'BOUND ROUTED': 9, 'MERGED (2 share)': 1} | 1 | 12; 9; 6/12 | 14400 (9600, 12000, 12000, 13200, 14400, 14400, 15600, 20400, 20400) |

READING Phase 2: **the delta recipe keeps eight streams** (BOUND ROUTED DELTA_SPLIT 9/10 vs HEBB_SPLIT 10/10; delta only 0, hebb only 1; p(hebb > delta) = 0.5, p(delta > hebb) = 1; rules: keeps c − b <= 1, loses c >= 4 and b = 0)

Splits per run (update c*->c0 ok?):

- HEBB_SPLIT: 1310 BOUND ROUTED@13200 [4800 12->8 ok, 9600 12->6 ok]; 1311 BOUND ROUTED@22800 [16800 3->1 ok, 21600 15->8 ok]; 1312 BOUND ROUTED@20400 [19200 4->14 ok]; 1313 BOUND ROUTED@22800 [4800 4->6 ok, 12000 6->11 ok, 19200 5->0 ok]; 1314 BOUND ROUTED@22800 [16800 15->11 ok, 21600 0->3 ok]; 1315 BOUND ROUTED@36000 [4800 6->13 ok, 9600 6->4 ok, 14400 1->5 ok]; 1316 BOUND ROUTED@40800 [9600 3->9 ok, 16800 3->10 ok, 26400 10->11 ok]; 1317 BOUND ROUTED@24000 [16800 11->12 ok, 21600 11->2 ok]; 1318 BOUND ROUTED@25200 [14400 2->6 ok, 19200 10->11 ok]; 1319 BOUND ROUTED@28800 [19200 1->15 ok, 24000 15->4 ok]
- DELTA_SPLIT: 1310 BOUND ROUTED@20400 [14400 12->6 NOT ok]; 1311 BOUND ROUTED@20400 [7200 14->5 NOT ok]; 1312 BOUND ROUTED@14400 [7200 2->10 NOT ok]; 1313 MERGED (2 share) 0.62 [4800 8->9 ok, 19200 1->13 ok, 28800 3->14 ok]; 1314 BOUND ROUTED@12000 [4800 5->6 ok, 9600 6->10 NOT ok]; 1315 BOUND ROUTED@13200 [4800 3->2 ok]; 1316 BOUND ROUTED@12000 [4800 5->8 NOT ok]; 1317 BOUND ROUTED@14400 [12000 12->7 NOT ok]; 1318 BOUND ROUTED@9600 []; 1319 BOUND ROUTED@15600 [4800 5->13 ok]

Runs: 39 ok (failed or diverged records 1), 31.90 h of run time; CPU ['Intel(R) Xeon(R) Processor @ 2.10GHz']; code SHA 2336ceff7760
