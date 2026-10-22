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
| CTXPOOL | 0.168 / 0.166 | P 0.129, bound 0.187 | 0.366 / 0.365 | P 0.382, bound 0.423 | None |
| DOCS3 | 0.123 / 0.123 | P 0.091, bound 0.156 | 0.190 / 0.200 | P 0.257, bound 0.310 | None |
| DOCS4 | 0.110 / 0.110 | P 0.069, bound 0.141 | 0.165 / 0.121 | P 0.187, bound 0.250 | None |

Phase 1, validity pilots (the perfect gate handed the per-sequence index, k = S, SLOW; seeds 3060 / 3061; transition or final accuracy), and the readings per task and memory:

| task | memory | 24000 | 48000 (fallback) | reading |
|---|---|---|---|---|
| CTXPOOL | HEBBCONV | -- | not run | **INCOMPLETE, no reading** |
| CTXPOOL | DELTA | -- | not run | **INCOMPLETE, no reading** |
| DOCS3 | HEBBCONV | -- | not run | **INCOMPLETE, no reading** |
| DOCS3 | DELTA | -- | not run | **INCOMPLETE, no reading** |
| DOCS4 | HEBBCONV | -- | not run | **INCOMPLETE, no reading** |
| DOCS4 | DELTA | -- | not run | **INCOMPLETE, no reading** |

The type-level perfect gate (balanced map of the 16 types to k = S channels; 24000; descriptive): final accuracy; end fraction of probe sequences separated / the cap; end margin; sep_key_inv; VAL cos within sequences:

| task | memory | 3060 | 3061 | separated (cap) | margin | sep_key_inv | VAL cos (within) |
|---|---|---|---|---|---|---|---|

Phase 2, the screen (one channel, no gate; seeds 3062-3071 paired; at each (task, memory)'s budget):

| task | arm | budget | BOUND | transitions (median; all) | unbound: final acc | write order: last / not last |
|---|---|---|---|---|---|---|


Per run (phase 2): seed outcome (B@t bound at t; else final accuracy):


READINGS per task and memory: **CTXPOOL HEBBCONV: INCOMPLETE, no reading**; **CTXPOOL DELTA: INCOMPLETE, no reading**; **DOCS3 HEBBCONV: INCOMPLETE, no reading**; **DOCS3 DELTA: INCOMPLETE, no reading**; **DOCS4 HEBBCONV: INCOMPLETE, no reading**; **DOCS4 DELTA: INCOMPLETE, no reading**

SCREEN READING (CTXPOOL, one delta channel): **INCOMPLETE, no reading**

F/s90_ctx: 12 runs ok (failed records 0), 1.10 h of run time; CPUs ['Intel(R) Xeon(R) Processor @ 2.10GHz']; code SHA 809fde1f5168; decision {}
