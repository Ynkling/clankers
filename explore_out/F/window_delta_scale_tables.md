### S54 (b), (c) — counts (BOUND ROUTED; Wilson 95%; band)

| arm | n | BOUND ROUTED | BOUND | outcomes | splits fired (total; runs) | transitions of BOUND ROUTED: median (all) | final acc median |
|---|---|---|---|---|---|---|---|
| B_DELTA | 20/20 | 20/20 [0.84, 1.00] RELIABLE | 20/20 [0.84, 1.00] RELIABLE | {'BOUND ROUTED': 20} | 5; 3 | 3600.0 (2400, 2400, 2400, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 8400, 9600, 10800, 22800) | 1.00 |
| B_HEBB | 20/20 | 19/20 [0.76, 0.99] RELIABLE | 20/20 [0.84, 1.00] RELIABLE | {'BOUND ROUTED': 19, 'BOUND NOT routed': 1} | 2; 1 | 3600 (3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 4800, 4800, 14400) | 1.00 |
| C_DELTA | 10/10 | 3/10 [0.11, 0.60] MINORITY | 3/10 [0.11, 0.60] MINORITY | {'STREAM-PARTIAL one-to-one': 6, 'BOUND ROUTED': 3, 'MERGED (2 share)': 1} | 28; 10 | 20400 (18000, 20400, 38400) | 0.43 |
| C_HEBB | 10/10 | 9/10 [0.60, 0.98] RELIABLE | 9/10 [0.60, 0.98] RELIABLE | {'BOUND ROUTED': 9, 'MERGED (2 share)': 1} | 26; 10 | 21600 (18000, 18000, 20400, 20400, 21600, 22800, 24000, 27600, 32400) | 1.00 |
| B_ORACLE_DELTA | 2/2 | 2/2 [0.34, 1.00] RELIABLE | 2/2 [0.34, 1.00] RELIABLE | -- | 0; 0 | 1200.0 (1200, 1200) | 1.00 |
| B_ORACLE_HEBB | 2/2 | 2/2 [0.34, 1.00] RELIABLE | 2/2 [0.34, 1.00] RELIABLE | -- | 0; 0 | 1200.0 (1200, 1200) | 1.00 |
| C_ORACLE_DELTA | 2/2 | 2/2 [0.34, 1.00] RELIABLE | 2/2 [0.34, 1.00] RELIABLE | -- | 0; 0 | 25800.0 (22800, 28800) | 1.00 |
| C_ORACLE_HEBB | 2/2 | 2/2 [0.34, 1.00] RELIABLE | 2/2 [0.34, 1.00] RELIABLE | -- | 0; 0 | 3600.0 (2400, 4800) | 0.99 |

Validity (perfect gate binds on >= 1 of its 2 seeds):

- B_ORACLE_DELTA: 2/2 -> VALID (transitions [1200, 1200])
- B_ORACLE_HEBB: 2/2 -> VALID (transitions [1200, 1200])
- C_ORACLE_DELTA: 2/2 -> VALID (transitions [28800, 22800])
- C_ORACLE_HEBB: 2/2 -> VALID (transitions [4800, 2400])

READING (b): **delta not worse** (BOUND ROUTED DELTA 20/20 vs HEBB 19/20; delta only 1, hebb only 0; one-sided McNemar p(delta > hebb) = 0.5, p(hebb > delta) = 1; rules: not worse c − b <= 2, better b >= 4 and c = 0)

READING (c): **delta worse** (BOUND ROUTED DELTA 3/10 vs HEBB 9/10; delta only 0, hebb only 6; one-sided McNemar p(delta > hebb) = 1, p(hebb > delta) = 0.0156; rules: not worse c − b <= 2, better b >= 4 and c = 0)

Per seed (outcome@transition, splits, end map):

- B_DELTA: 310 BOUND ROUTED@3600 s0 1+1+1+1; 311 BOUND ROUTED@3600 s0 1+1+1+1; 312 BOUND ROUTED@2400 s0 1+1+1+1; 313 BOUND ROUTED@3600 s0 1+1+1+1; 314 BOUND ROUTED@3600 s0 1+1+1+1; 315 BOUND ROUTED@3600 s0 1+1+1+1; 316 BOUND ROUTED@22800 s3 1+1+1+1; 317 BOUND ROUTED@3600 s0 1+1+1+1; 318 BOUND ROUTED@2400 s0 1+1+1+1; 319 BOUND ROUTED@2400 s0 1+1+1+1; 320 BOUND ROUTED@10800 s1 1+1+1+1; 321 BOUND ROUTED@3600 s0 1+1+1+1; 322 BOUND ROUTED@3600 s0 1+1+1+1; 323 BOUND ROUTED@9600 s0 1+1+1+1; 324 BOUND ROUTED@8400 s1 1+1+1+1; 325 BOUND ROUTED@3600 s0 1+1+1+1; 326 BOUND ROUTED@3600 s0 1+1+1+1; 327 BOUND ROUTED@3600 s0 1+1+1+1; 328 BOUND ROUTED@3600 s0 1+1+1+1; 329 BOUND ROUTED@3600 s0 1+1+1+1
- B_HEBB: 310 BOUND ROUTED@3600 s0 1+1+1+1; 311 BOUND ROUTED@3600 s0 1+1+1+1; 312 BOUND ROUTED@3600 s0 1+1+1+1; 313 BOUND ROUTED@3600 s0 1+1+1+1; 314 BOUND ROUTED@4800 s0 1+1+1+1; 315 BOUND ROUTED@3600 s0 1+1+1+1; 316 BOUND ROUTED@14400 s2 1+1+1+1; 317 BOUND ROUTED@3600 s0 1+1+1+1; 318 BOUND ROUTED@3600 s0 1+1+1+1; 319 BOUND ROUTED@3600 s0 1+1+1+1; 320 BOUND ROUTED@3600 s0 1+1+1+1; 321 BOUND ROUTED@3600 s0 1+1+1+1; 322 BOUND ROUTED@3600 s0 1+1+1+1; 323 BOUND NOT routed@9600 s0 3+1; 324 BOUND ROUTED@3600 s0 1+1+1+1; 325 BOUND ROUTED@3600 s0 1+1+1+1; 326 BOUND ROUTED@3600 s0 1+1+1+1; 327 BOUND ROUTED@3600 s0 1+1+1+1; 328 BOUND ROUTED@3600 s0 1+1+1+1; 329 BOUND ROUTED@4800 s0 1+1+1+1
- C_DELTA: 310 STREAM-PARTIAL one-to-one 0.33 s3 1+1+1+1+1+1+1+1; 311 STREAM-PARTIAL one-to-one 0.50 s3 1+1+1+1+1+1+1+1; 312 STREAM-PARTIAL one-to-one 0.43 s3 1+1+1+1+1+1+1+1; 313 STREAM-PARTIAL one-to-one 0.32 s3 1+1+1+1+1+1+1+1; 314 STREAM-PARTIAL one-to-one 0.32 s3 1+1+1+1+1+1+1+1; 315 BOUND ROUTED@20400 s3 1+1+1+1+1+1+1+1; 316 BOUND ROUTED@18000 s1 1+1+1+1+1+1+1+1; 317 BOUND ROUTED@38400 s3 1+1+1+1+1+1+1+1; 318 MERGED (2 share) 0.43 s3 2+1+1+1+1+1+1; 319 STREAM-PARTIAL one-to-one 0.34 s3 1+1+1+1+1+1+1+1
- C_HEBB: 310 BOUND ROUTED@24000 s3 1+1+1+1+1+1+1+1; 311 BOUND ROUTED@18000 s2 1+1+1+1+1+1+1+1; 312 BOUND ROUTED@21600 s3 1+1+1+1+1+1+1+1; 313 BOUND ROUTED@32400 s3 1+1+1+1+1+1+1+1; 314 MERGED (2 share) 0.88 s3 2+1+1+1+1+1+1; 315 BOUND ROUTED@27600 s3 1+1+1+1+1+1+1+1; 316 BOUND ROUTED@22800 s3 1+1+1+1+1+1+1+1; 317 BOUND ROUTED@20400 s3 1+1+1+1+1+1+1+1; 318 BOUND ROUTED@18000 s2 1+1+1+1+1+1+1+1; 319 BOUND ROUTED@20400 s1 1+1+1+1+1+1+1+1

B_DELTA gate-state decodability at KEY / VAL (median): 1200 0.80/0.78; 2400 0.91/0.90; 4800 1.00/1.00; 9600 0.92/0.94; end 1.00/1.00

B_HEBB gate-state decodability at KEY / VAL (median): 1200 0.84/0.88; 2400 0.98/1.00; 4800 1.00/1.00; 9600 0.54/0.52; end 1.00/1.00

C_DELTA gate-state decodability at KEY / VAL (median): 1200 0.85/0.84; 2400 0.69/0.69; 4800 0.67/0.75; 9600 0.79/0.86; end 1.00/1.00

C_HEBB gate-state decodability at KEY / VAL (median): 1200 0.85/0.83; 2400 0.54/0.57; 4800 0.48/0.59; 9600 0.85/0.86; end 1.00/1.00

Learned β at CTX/KEY/VAL per layer, end (median): B_DELTA L1 0.51/0.34/0.59; L2 0.59/0.11/0.64; L3 0.60/0.09/0.65; C_DELTA L1 0.03/0.01/0.44; L2 0.01/0.02/0.20; L3 0.01/0.06/0.24

Runs: 68 ok (failed records 0), 71.58 h of run time; CPU ['Intel(R) Xeon(R) Processor @ 2.10GHz']; git ['007c33b', '007c33b+dirty', '0268500', '0268500+dirty', '0442f35', '092753c', '130a278', '1af0eae', '1af0eae+dirty', '21fae31', '3083fad', '3083fad+dirty', '313aa2f', '313aa2f+dirty', '3459b4d', '365b008', '365b008+dirty', '439a406', '439a406+dirty', '4fd5694', '4fd5694+dirty', '5532430', '98cad4c', '98cad4c+dirty', 'af53cb4', 'c7359e5', 'c7359e5+dirty', 'c87b266', 'c87b266+dirty', 'd7ae05c', 'f241dc2', 'f241dc2+dirty', 'f2db36e', 'f2db36e+dirty', 'f5dfb37', 'f5dfb37+dirty']; code SHA c8612c854acc
