### S88b the horizon (one delta channel, RandHeaderTask-lite at 1x and 2x cue-to-key distance, 24000; seeds 2420-2429; BOUND; Wilson 95%)

Validity (the perfect gate binds on both seeds at each length):

- O_FIXED_1X: 2420 bound @1200; 2421 bound @1200
- O_FIXED_2X: 2420 bound @1200; 2421 bound @1200

| arm | n | BOUND | transitions (median; all) | final accuracy (median) | accuracy by cue-to-key distance at the end (median over runs) |
|---|---|---|---|---|---|
| FIXED_1X | 10/10 | 9/10 [0.60, 0.98] RELIABLE | 6000 (2400, 2400, 2400, 3600, 6000, 6000, 7200, 16800, 21600) | 1.00 | 1: 1.00, 3: 1.00, 5: 1.00 |
| ROUTED_1X | 10/10 | 6/10 [0.31, 0.83] MAJORITY | 16200 (1200, 2400, 15600, 16800, 16800, 20400) | 1.00 | 1: 1.00, 3: 0.99, 5: 0.99 |
| FIXED_2X | 10/10 | 4/10 [0.17, 0.69] MINORITY | 14400 (4800, 14400, 14400, 22800) | 0.51 | 2: 0.50, 6: 0.51, 10: 0.53 |
| ROUTED_2X | 10/10 | 5/10 [0.24, 0.76] MAJORITY | 13200 (2400, 3600, 13200, 16800, 20400) | 0.75 | 2: 0.76, 6: 0.76, 10: 0.75 |

Within a length (b = ROUTED alone, c = FIXED alone):

- 1X: ROUTED 6/10 vs FIXED 9/10: b 0, c 3; p(ROUTED > FIXED) = 1, p(FIXED > ROUTED) = 0.125
- 2X: ROUTED 5/10 vs FIXED 4/10: b 3, c 2; p(ROUTED > FIXED) = 0.5, p(FIXED > ROUTED) = 0.812

Across lengths (same seed numbers): drop_FIXED = 5, drop_ROUTED = 1

READING S88b: **untested (at 1x FIXED 9/10, ROUTED 6/10; rule >= 8/10 each)** (rules: untested if at 1x either arm < 8/10; extends if at 2x b >= 4 and c = 0; the decay's if drop_FIXED >= 4 and |drop_ROUTED − drop_FIXED| <= 1)

The learned decay at the end (ROUTED; median over runs; α at CTX / KEY / VAL / FILL per layer) and Δ:

- ROUTED_1X: L1 ctx 0.889 key 0.999 val 0.987; L2 ctx 0.977 key 1.000 val 0.999; L3 ctx 0.977 key 1.000 val 0.998; Δ median -3.007 (init -2.604), |w| median 3.100
- ROUTED_2X: L1 ctx 0.997 fill 1.000 key 0.999 val 0.978; L2 ctx 0.999 fill 1.000 key 1.000 val 1.000; L3 ctx 0.999 fill 1.000 key 0.999 val 1.000; Δ median -3.048 (init -2.604), |w| median 2.089

Per run (B = bound @transition, else final accuracy; write order: accuracy when the queried pair was the last write of its key / not):

- FIXED_1X: 2420 B@2400 (1.00/1.00); 2421 B@3600 (1.00/1.00); 2422 B@6000 (1.00/1.00); 2423 B@2400 (1.00/0.99); 2424 B@6000 (1.00/1.00); 2425 B@21600 (1.00/1.00); 2426 0.50 (0.48/0.52); 2427 B@16800 (1.00/1.00); 2428 B@2400 (1.00/1.00); 2429 B@7200 (1.00/1.00)
- ROUTED_1X: 2420 0.48 (0.47/0.50); 2421 0.48 (0.47/0.50); 2422 B@16800 (1.00/1.00); 2423 B@2400 (1.00/1.00); 2424 B@16800 (1.00/1.00); 2425 B@15600 (1.00/0.99); 2426 0.51 (0.52/0.49); 2427 0.51 (0.56/0.46); 2428 B@20400 (1.00/1.00); 2429 B@1200 (1.00/1.00)
- FIXED_2X: 2420 B@14400 (1.00/1.00); 2421 0.48 (0.47/0.49); 2422 0.51 (0.51/0.50); 2423 B@22800 (1.00/0.99); 2424 0.50 (0.50/0.51); 2425 B@14400 (1.00/1.00); 2426 0.49 (0.51/0.47); 2427 B@4800 (1.00/1.00); 2428 0.51 (0.56/0.46); 2429 0.49 (0.45/0.53)
- ROUTED_2X: 2420 B@2400 (1.00/1.00); 2421 B@13200 (1.00/1.00); 2422 B@16800 (1.00/1.00); 2423 0.50 (0.52/0.47); 2424 0.50 (0.51/0.49); 2425 0.51 (0.56/0.45); 2426 0.48 (0.52/0.44); 2427 B@3600 (1.00/1.00); 2428 0.51 (0.55/0.46); 2429 B@20400 (1.00/1.00)

F/horizon: 44 runs ok (failed records 0), 8.73 h of run time; CPU ['Intel(R) Xeon(R) Processor @ 2.80GHz']; code SHA e9c41e74d945
