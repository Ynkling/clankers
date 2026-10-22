### S90-F the context-pool family (seeds 3060-3071)

Token-type frequencies over S87's sample (4096 sequences, torch.Generator(87_000), input tokens), both cuts, and the cap of a fixed type -> channel map (the fraction of sequences whose D contexts land on D distinct channels; exact):

| task | CTX types | KEY types | VAL types | largest-ratio cut | half-mass cut | cap at k = D | k = 2D | k = 16 |
|---|---|---|---|---|---|---|---|---|
| CTXPOOL (D = 2) | 16 (0.0135-0.0154) | 4 (0.1017-0.1027) | 16 (0.0220-0.0236) | 4 KEY | 4 KEY, 4 VAL (mass 0.502) | 64/120 = 0.533 | 0.800 | 1.000 |
| DOCS3 (D = 3) | 16 (0.0126-0.0145) | 4 (0.1013-0.1018) | 16 (0.0229-0.0242) | 4 KEY | 4 KEY, 4 VAL (mass 0.502) | 150/560 = 0.268 | 0.664 | 1.000 |
| DOCS4 (D = 4) | 16 (0.0128-0.0139) | 4 (0.1010-0.1014) | 16 (0.0233-0.0244) | 4 KEY | 4 KEY, 4 VAL (mass 0.502) | 256/1820 = 0.141 | 0.615 | 1.000 |

Phase 0, no-memory solvability (held-out accuracy at the end, seeds 3060 / 3061; rule: stopped if > 0.6), with P(the target pair inside the reach) and the positional bound P + (1 - P)/(n_vals - S + 1):

| task | NOMEM (reach 4) | bound | NOMEM_SELF (reach 10) | bound | stopped |
|---|---|---|---|---|---|
| CTXPOOL | 0.168 / 0.166 | P 0.129, bound 0.187 | 0.366 / 0.365 | P 0.382, bound 0.423 | False |
| DOCS3 | 0.123 / 0.123 | P 0.091, bound 0.156 | 0.190 / 0.200 | P 0.257, bound 0.310 | False |
| DOCS4 | 0.110 / 0.110 | P 0.069, bound 0.141 | 0.165 / 0.121 | P 0.187, bound 0.250 | False |

Phase 1, validity pilots (the perfect gate handed the per-sequence index, k = S, SLOW; seeds 3060 / 3061; transition or final accuracy), and the readings per task and memory:

| task | memory | 24000 | 48000 (fallback) | reading |
|---|---|---|---|---|
| CTXPOOL | HEBBCONV | @2400; @4800 | not run | **the task is valid** |
| CTXPOOL | DELTA | @1200; @2400 | not run | **the task is valid** |
| DOCS3 | HEBBCONV | @3600; @3600 | not run | **the task is valid** |
| DOCS3 | DELTA | @1200; @2400 | not run | **the task is valid** |
| DOCS4 | HEBBCONV | @3600; @3600 | not run | **the task is valid** |
| DOCS4 | DELTA | @1200; @2400 | not run | **the task is valid** |

The type-level perfect gate (balanced map of the 16 types to k = S channels; 24000; descriptive): final accuracy; end fraction of probe sequences separated / the cap; end margin; sep_key_inv; VAL cos within sequences:

| task | memory | 3060 | 3061 | separated (cap) | margin | sep_key_inv | VAL cos (within) |
|---|---|---|---|---|---|---|---|
| CTXPOOL | HEBBCONV | unbound 0.77 | unbound 0.76 | 0.53 (0.533) | 0.53 | 0.53 | 0.47 |
| CTXPOOL | DELTA | @9600 | @10800 | 0.53 (0.533) | 0.53 | 0.53 | 0.47 |
| DOCS3 | HEBBCONV | unbound 0.73 | unbound 0.72 | 0.32 (0.268) | 0.73 | 0.73 | 0.27 |
| DOCS3 | DELTA | @8400 | @8400 | 0.32 (0.268) | 0.73 | 0.73 | 0.27 |
| DOCS4 | HEBBCONV | unbound 0.81 | unbound 0.73 | 0.14 (0.141) | 0.80 | 0.80 | 0.20 |
| DOCS4 | DELTA | @9600 | @10800 | 0.14 (0.141) | 0.80 | 0.80 | 0.20 |

Phase 2, the screen (one channel, no gate; seeds 3062-3071 paired; at each (task, memory)'s budget):

| task | arm | budget | BOUND | transitions (median; all) | unbound: final acc | write order: last / not last |
|---|---|---|---|---|---|---|
| CTXPOOL | SINGLE_HEBBCONV_CTXPOOL | 24000 | 0/10 [0.00, 0.28] NEVER | -- | 3062 0.48, 3063 0.50, 3064 0.49, 3065 0.51, 3066 0.49, 3067 0.49, 3068 0.50, 3069 0.51, 3070 0.51, 3071 0.49 | 0.52 / 0.48 |
| CTXPOOL | SINGLE_DELTA_CTXPOOL | 24000 | 10/10 [0.72, 1.00] RELIABLE | 9600 (7200, 8400, 8400, 8400, 9600, 9600, 9600, 10800, 14400, 15600) | none | 0.99 / 0.99 |
| DOCS3 | SINGLE_HEBBCONV_DOCS3 | 24000 | 0/10 [0.00, 0.28] NEVER | -- | 3062 0.34, 3063 0.33, 3064 0.32, 3065 0.35, 3066 0.35, 3067 0.33, 3068 0.34, 3069 0.35, 3070 0.34, 3071 0.33 | 0.35 / 0.33 |
| DOCS3 | SINGLE_DELTA_DOCS3 | 24000 | 0/10 [0.00, 0.28] NEVER | -- | 3062 0.43, 3063 0.34, 3064 0.32, 3065 0.81, 3066 0.33, 3067 0.33, 3068 0.30, 3069 0.32, 3070 0.32, 3071 0.32 | 0.43 / 0.36 |
| DOCS4 | SINGLE_HEBBCONV_DOCS4 | 24000 | 0/10 [0.00, 0.28] NEVER | -- | 3062 0.25, 3063 0.24, 3064 0.24, 3065 0.25, 3066 0.24, 3067 0.24, 3068 0.26, 3069 0.25, 3070 0.24, 3071 0.25 | 0.24 / 0.25 |
| DOCS4 | SINGLE_DELTA_DOCS4 | 24000 | 0/10 [0.00, 0.28] NEVER | -- | 3062 0.22, 3063 0.23, 3064 0.24, 3065 0.24, 3066 0.20, 3067 0.24, 3068 0.22, 3069 0.25, 3070 0.22, 3071 0.24 | 0.25 / 0.22 |

- CTXPOOL: SINGLE_DELTA 10/10 vs SINGLE_HEBBCONV 0/10: delta only 10, Hebbian + conv only 0; exact one-sided McNemar p (delta > Hebbian) = 0.0009766, (Hebbian > delta) = 1
- DOCS3: SINGLE_DELTA 0/10 vs SINGLE_HEBBCONV 0/10: delta only 0, Hebbian + conv only 0; exact one-sided McNemar p (delta > Hebbian) = 1, (Hebbian > delta) = 1
- DOCS4: SINGLE_DELTA 0/10 vs SINGLE_HEBBCONV 0/10: delta only 0, Hebbian + conv only 0; exact one-sided McNemar p (delta > Hebbian) = 1, (Hebbian > delta) = 1

Per run (phase 2): seed outcome (B@t bound at t; else final accuracy):

- SINGLE_HEBBCONV_CTXPOOL: 3062 0.48; 3063 0.50; 3064 0.49; 3065 0.51; 3066 0.49; 3067 0.49; 3068 0.50; 3069 0.51; 3070 0.51; 3071 0.49
- SINGLE_DELTA_CTXPOOL: 3062 B@8400; 3063 B@8400; 3064 B@9600; 3065 B@9600; 3066 B@9600; 3067 B@7200; 3068 B@8400; 3069 B@14400; 3070 B@10800; 3071 B@15600
- SINGLE_HEBBCONV_DOCS3: 3062 0.34; 3063 0.33; 3064 0.32; 3065 0.35; 3066 0.35; 3067 0.33; 3068 0.34; 3069 0.35; 3070 0.34; 3071 0.33
- SINGLE_DELTA_DOCS3: 3062 0.43; 3063 0.34; 3064 0.32; 3065 0.81; 3066 0.33; 3067 0.33; 3068 0.30; 3069 0.32; 3070 0.32; 3071 0.32
- SINGLE_HEBBCONV_DOCS4: 3062 0.25; 3063 0.24; 3064 0.24; 3065 0.25; 3066 0.24; 3067 0.24; 3068 0.26; 3069 0.25; 3070 0.24; 3071 0.25
- SINGLE_DELTA_DOCS4: 3062 0.22; 3063 0.23; 3064 0.24; 3065 0.24; 3066 0.20; 3067 0.24; 3068 0.22; 3069 0.25; 3070 0.22; 3071 0.24

READINGS per task and memory: **CTXPOOL HEBBCONV: the task is valid**; **CTXPOOL DELTA: the task is valid**; **DOCS3 HEBBCONV: the task is valid**; **DOCS3 DELTA: the task is valid**; **DOCS4 HEBBCONV: the task is valid**; **DOCS4 DELTA: the task is valid**

SCREEN READING (CTXPOOL, one delta channel): **one channel binds without a context identity**

F/s90_ctx: 96 runs ok (failed records 0), 11.48 h of run time; CPUs ['Intel(R) Xeon(R) Processor @ 2.10GHz']; code SHA 809fde1f5168; decision {'phase0': {'CTXPOOL': {'acc': {'NOMEM': [0.16845703125, 0.166015625], 'NOMEM_SELF': [0.36572265625, 0.365234375]}, 'stopped': False}, 'DOCS3': {'acc': {'NOMEM': [0.123046875, 0.12255859375], 'NOMEM_SELF': [0.1904296875, 0.2001953125]}, 'stopped': False}, 'DOCS4': {'acc': {'NOMEM': [0.1103515625, 0.1103515625], 'NOMEM_SELF': [0.16455078125, 0.12060546875]}, 'stopped': False}}, 'budget': {'CTXPOOL': {'HEBBCONV': 24000, 'DELTA': 24000}, 'DOCS3': {'HEBBCONV': 24000, 'DELTA': 24000}, 'DOCS4': {'HEBBCONV': 24000, 'DELTA': 24000}}, 'pilots': {'CTXPOOL': {'HEBBCONV': {'at24': True, 'double': False}, 'DELTA': {'at24': True, 'double': False}}, 'DOCS3': {'HEBBCONV': {'at24': True, 'double': False}, 'DELTA': {'at24': True, 'double': False}}, 'DOCS4': {'HEBBCONV': {'at24': True, 'double': False}, 'DELTA': {'at24': True, 'double': False}}}}
