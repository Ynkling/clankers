### S81 the split's trigger on delta memory (S = 8, P = 4, k = 16, conv, 43200; DELTA*; seeds 2020-2029; BOUND ROUTED; Wilson 95%)

Validity (the delta perfect gate binds on both seeds):

- O_DELTA_C: 2020 bound @6000; 2021 bound @1200

| arm | n | BOUND ROUTED | BOUND | outcomes (v1) | firings (total; runs; labelled ok) | firings at a disagreement | transitions of BOUND ROUTED |
|---|---|---|---|---|---|---|---|
| BASE | 10/10 | 9/10 [0.60, 0.98] RELIABLE | 9/10 [0.60, 0.98] RELIABLE | {'BOUND ROUTED': 9, 'MERGED (2 share)': 1} | 19; 9; 10/19 (53%) | 7 | 15600 (7200, 9600, 12000, 13200, 15600, 20400, 20400, 22800, 32400) |
| GUARD | 10/10 | 7/10 [0.40, 0.89] MAJORITY | 7/10 [0.40, 0.89] MAJORITY | {'MERGED (2 share)': 3, 'BOUND ROUTED': 7} | 6; 3; 6/6 (100%) | 0 | 13200 (7200, 10800, 12000, 13200, 13200, 14400, 15600) |
| EVAL | 10/10 | 10/10 [0.72, 1.00] RELIABLE | 10/10 [0.72, 1.00] RELIABLE | {'BOUND ROUTED': 10} | 13; 9; 8/13 (62%) | 5 | 15600 (7200, 9600, 13200, 14400, 15600, 15600, 16800, 19200, 22800, 31200) |
| GUARD_EVAL | 10/10 | 8/10 [0.49, 0.94] MAJORITY | 8/10 [0.49, 0.94] MAJORITY | {'MERGED (2 share)': 2, 'BOUND ROUTED': 8} | 9; 3; 9/9 (100%) | 4 | 13200 (7200, 10800, 12000, 13200, 13200, 14400, 15600, 31200) |

Checks (all arms; printed, not readings): disagreement = the rule's eligibility read on the probe differs from it read on the evaluation (with the arm's guard).

| arm | checks | guarded (eligible but < 0.5) | disagreements: probe yes / eval no | probe no / eval yes | probe >= 0.95 while eval < 0.95 | eval >= 0.95 while probe < 0.95 |
|---|---|---|---|---|---|---|
| BASE | 87 | 0 | 8 | 12 | 2 | 0 |
| GUARD | 92 | 13 | 1 | 9 | 5 | 0 |
| EVAL | 77 | 0 | 9 | 8 | 0 | 0 |
| GUARD_EVAL | 89 | 18 | 1 | 8 | 4 | 0 |

Paired against BASE (BOUND ROUTED; b = the arm alone, c = BASE alone; exact one-sided McNemar):

- GUARD 7/10 vs BASE 9/10: b 0, c 2; p(arm > BASE) = 1, p(BASE > arm) = 0.25
- EVAL 10/10 vs BASE 9/10: b 1, c 0; p(arm > BASE) = 0.5, p(BASE > arm) = 1
- GUARD_EVAL 8/10 vs BASE 9/10: b 1, c 2; p(arm > BASE) = 0.875, p(BASE > arm) = 0.5

READINGS S81: **guard: no reading applies; evaluation: no reading applies** (GUARD firings labelled ok 6/6, rule >= 90%; EVAL firings at a disagreement 5, rule 0; BASE labelled ok 10/19 (S76: 6/12); GUARD_EVAL labelled ok 9/9, at a disagreement 4)

Firings per run (update c*->c0, labelled ok?, probe / evaluation reading, D = at a disagreement):

- BASE: 2020 BOUND ROUTED@12000 [4800 8->15 ok 0.06/0.06, 9600 13->1 ok 0.11/0.12 D]; 2021 BOUND ROUTED@9600 [4800 10->3 NOT ok 0.06/0.07]; 2022 BOUND ROUTED@20400 [4800 10->2 ok 0.06/0.06, 12000 2->6 NOT ok 0.06/0.10 D]; 2023 BOUND ROUTED@7200 [4800 0->15 ok 0.09/0.10 D]; 2024 BOUND ROUTED@22800 [4800 0->5 NOT ok 0.08/0.06, 9600 7->14 ok 0.03/0.10 D, 19200 4->13 ok 0.66/0.68 D]; 2025 BOUND ROUTED@15600 [4800 4->7 ok 0.06/0.07, 9600 12->7 NOT ok 0.08/0.13 D]; 2026 BOUND ROUTED@20400 [4800 7->4 NOT ok 0.05/0.06, 9600 5->3 NOT ok 0.05/0.15 D]; 2027 BOUND ROUTED@32400 [7200 12->3 NOT ok 0.05/0.07, 21600 15->11 NOT ok 0.81/0.77, 26400 14->9 NOT ok 0.81/0.77]; 2028 BOUND ROUTED@13200 []; 2029 MERGED (2 share) 0.88 [14400 11->13 ok 0.61/0.69, 24000 13->5 ok 0.77/0.87, 28800 11->4 ok 0.78/0.87]
- GUARD: 2020 MERGED (2 share) 0.75 [31200 12->14 ok 0.78/0.73, 36000 7->9 ok 0.78/0.74]; 2021 BOUND ROUTED@10800 []; 2022 BOUND ROUTED@15600 []; 2023 BOUND ROUTED@7200 []; 2024 BOUND ROUTED@14400 []; 2025 MERGED (2 share) 0.93 [31200 2->12 ok 0.94/0.87]; 2026 BOUND ROUTED@12000 []; 2027 BOUND ROUTED@13200 []; 2028 BOUND ROUTED@13200 []; 2029 MERGED (2 share) 0.88 [14400 11->13 ok 0.61/0.69, 24000 13->5 ok 0.77/0.87, 28800 11->4 ok 0.78/0.87]
- EVAL: 2020 BOUND ROUTED@14400 [4800 8->15 ok 0.06/0.06]; 2021 BOUND ROUTED@9600 [4800 10->3 NOT ok 0.06/0.07]; 2022 BOUND ROUTED@19200 [4800 10->2 ok 0.06/0.06, 9600 15->0 ok 0.12/0.08 D]; 2023 BOUND ROUTED@7200 []; 2024 BOUND ROUTED@16800 [4800 0->5 NOT ok 0.08/0.06]; 2025 BOUND ROUTED@15600 [4800 4->7 ok 0.06/0.07]; 2026 BOUND ROUTED@22800 [4800 7->4 NOT ok 0.05/0.06, 19200 6->2 NOT ok 0.92/0.84 D]; 2027 BOUND ROUTED@13200 [4800 3->13 ok 0.09/0.06 D]; 2028 BOUND ROUTED@15600 [4800 12->11 NOT ok 0.12/0.06 D]; 2029 BOUND ROUTED@31200 [14400 11->13 ok 0.61/0.69, 21600 11->5 ok 0.78/0.87 D, 26400 11->4 ok 0.75/0.86]
- GUARD_EVAL: 2020 MERGED (2 share) 0.74 [31200 12->14 ok 0.78/0.73, 36000 7->9 ok 0.78/0.74, 40800 9->2 ok 0.84/0.74 D]; 2021 BOUND ROUTED@10800 []; 2022 BOUND ROUTED@15600 []; 2023 BOUND ROUTED@7200 []; 2024 BOUND ROUTED@14400 []; 2025 MERGED (2 share) 0.92 [28800 2->12 ok 0.95/0.87 D, 33600 2->6 ok 0.97/0.90 D, 38400 2->13 ok 0.94/0.91]; 2026 BOUND ROUTED@12000 []; 2027 BOUND ROUTED@13200 []; 2028 BOUND ROUTED@13200 []; 2029 BOUND ROUTED@31200 [14400 11->13 ok 0.61/0.69, 21600 11->5 ok 0.78/0.87 D, 26400 11->4 ok 0.75/0.86]

F/trigger: 42 runs ok (failed or diverged records 0), 69.17 h of run time; CPU ['Intel(R) Xeon(R) Processor @ 2.10GHz']; code SHA bdd8269d38af
