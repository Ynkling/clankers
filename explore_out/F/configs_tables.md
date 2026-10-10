### S80 (a) two streams (S = 2, P = 4, k = 2, no conv, 24000; WIN3_SLOW; seeds 2000-2009; DISCOVERED; Wilson 95%)

Validity (the delta perfect gate binds on both seeds):

- O_DELTA_A: 2000 bound @1200; 2001 bound @1200
- O_HEBB_A: 2000 bound @2400; 2001 bound @1200

| arm | n | DISCOVERED | BOUND | transitions of DISCOVERED: median (all) | not discovered: bound (VAL cos) / unbound |
|---|---|---|---|---|---|
| HEBB_A | 10/10 | 10/10 [0.72, 1.00] RELIABLE | 10/10 [0.72, 1.00] RELIABLE | 2400 (2400, 2400, 2400, 2400, 2400, 2400, 3600, 3600, 3600, 6000) | none |
| DELTA_A | 10/10 | 10/10 [0.72, 1.00] RELIABLE | 10/10 [0.72, 1.00] RELIABLE | 1200 (1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200) | none |

READING (a): **the delta recipe keeps it** (DISCOVERED DELTA_A 10/10 vs HEBB_A 10/10; delta only 0, hebb only 0; p(hebb > delta) = 1, p(delta > hebb) = 1; rules: keeps c − b <= 1, loses c >= 4 and b = 0). Reference (counts only, other CPUs and seeds): WIN3_SLOW pooled 69/80.

Failure classes of the unbound runs (v2 = fail_class_v2, ba8901a; v1 = test_router_layout.fail_class):

- HEBB_A: none
- DELTA_A: none
- v1 and v2 differ on: none

Probe at the end of every run (S77's: G's S55 logistic regression on the residual; stream of the body triple; held-out 400 sequences; chance 0.5):

| arm | n | emb K / V | L1 K / V | L2 K / V | L3 K / V | final K / V | L3 >= 0.9 at K and V | at init: L3 K / V |
|---|---|---|---|---|---|---|---|---|
| HEBB_A | 10 | 0.50 / 0.50 | 0.50 / 0.50 | 0.56 / 0.51 | 0.57 / 0.52 | 0.57 / 0.56 | 0/10 | 0.79 / 0.68 |
| DELTA_A | 10 | 0.50 / 0.50 | 0.50 / 0.50 | 0.99 / 0.69 | 1.00 / 0.58 | 1.00 / 0.62 | 0/10 | 0.95 / 0.72 |
| O_HEBB_A | 2 | 0.50 / 0.50 | 0.50 / 0.50 | 0.64 / 0.51 | 0.67 / 0.56 | 0.66 / 0.64 | 0/2 | 1.00 / 0.98 |
| O_DELTA_A | 2 | 0.50 / 0.50 | 0.50 / 0.50 | 0.91 / 0.55 | 0.98 / 0.52 | 0.99 / 0.58 | 0/2 | 0.98 / 0.81 |

Per seed, L3 K / V (HEBB_A | DELTA_A; D = discovered):

- 2000: 0.59 / 0.52 D | 1.00 / 0.65 D
- 2001: 0.59 / 0.57 D | 1.00 / 0.57 D
- 2002: 0.54 / 0.53 D | 1.00 / 0.63 D
- 2003: 0.52 / 0.54 D | 0.99 / 0.56 D
- 2004: 0.54 / 0.53 D | 0.99 / 0.55 D
- 2005: 0.57 / 0.49 D | 1.00 / 0.58 D
- 2006: 0.57 / 0.50 D | 1.00 / 0.60 D
- 2007: 0.55 / 0.53 D | 1.00 / 0.62 D
- 2008: 0.60 / 0.52 D | 0.98 / 0.54 D
- 2009: 0.58 / 0.52 D | 1.00 / 0.58 D

### S80 (b) four streams (S = 4, P = 4, k = 16, conv, 28800; WIN3_SLOW16 + exact-copy split; seeds 2010-2019; BOUND ROUTED; Wilson 95%)

Validity:

- O_DELTA_B: 2010 bound @1200; 2011 bound @1200
- O_HEBB_B: 2010 bound @1200; 2011 bound @1200

| arm | n | BOUND ROUTED | BOUND | outcomes (v1) | splits (total; runs; targets ok) | transitions of BOUND ROUTED |
|---|---|---|---|---|---|---|
| HEBB_B | 10/10 | 10/10 [0.72, 1.00] RELIABLE | 10/10 [0.72, 1.00] RELIABLE | {'BOUND ROUTED': 10} | 0; 0; 0/0 | 3600 (3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600) |
| DELTA_B | 10/10 | 10/10 [0.72, 1.00] RELIABLE | 10/10 [0.72, 1.00] RELIABLE | {'BOUND ROUTED': 10} | 0; 0; 0/0 | 3600 (3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600) |

READING (b): **the delta recipe keeps it** (BOUND ROUTED DELTA_B 10/10 vs HEBB_B 10/10; delta only 0, hebb only 0; p(hebb > delta) = 1, p(delta > hebb) = 1). Reference (counts only): WIN3_SLOW16 pooled 38/40 (no split).

Per run (outcome v1 [v2 where it differs] @transition or final acc; splits update c*->c0 ok?):

- HEBB_B: 2010 BOUND ROUTED@3600 []; 2011 BOUND ROUTED@3600 []; 2012 BOUND ROUTED@3600 []; 2013 BOUND ROUTED@3600 []; 2014 BOUND ROUTED@3600 []; 2015 BOUND ROUTED@3600 []; 2016 BOUND ROUTED@3600 []; 2017 BOUND ROUTED@3600 []; 2018 BOUND ROUTED@3600 []; 2019 BOUND ROUTED@3600 []
- DELTA_B: 2010 BOUND ROUTED@3600 []; 2011 BOUND ROUTED@3600 []; 2012 BOUND ROUTED@3600 []; 2013 BOUND ROUTED@3600 []; 2014 BOUND ROUTED@3600 []; 2015 BOUND ROUTED@3600 []; 2016 BOUND ROUTED@3600 []; 2017 BOUND ROUTED@3600 []; 2018 BOUND ROUTED@3600 []; 2019 BOUND ROUTED@3600 []

Classifier CHECK at report time: v2 equals v1 on 367 stored Session F records (236 two-stream, 131 multichannel) except where eta2 by key >= 0.5 and the margin rule fired: 1 differ: window_delta_scale:B_HEBB|323 (bound) STREAM-PARTIAL -> KEY (eta2 by key 0.6762, margin 0.362); violations none (ok)

F/configs_a: 24 runs ok (failed records 0), 0.55 h of run time; CPU ['Intel(R) Xeon(R) Processor @ 2.10GHz']; code SHA d7c6a0326eae

F/configs_b: 24 runs ok (failed records 0), 3.57 h of run time; CPU ['Intel(R) Xeon(R) Processor @ 2.10GHz']; code SHA 678c244056e9
