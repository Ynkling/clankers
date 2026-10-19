### S89-F the lite task without marked keys (seeds 2800-2811)

Token-type frequencies over S87's sample (4096 sequences, torch.Generator(87_000), input tokens) and the two cuts:

| task | CTX types (frequency) | KEY types | VAL types | largest-ratio cut selects | half-mass cut selects |
|---|---|---|---|---|---|
| LITE | 2 (0.1136-0.1137) | 4 (0.1017-0.1027) | 16 (0.0220-0.0236) | 2 CTX, 4 KEY | 2 CTX, 3 KEY (mass 0.535) |
| POOL | 2 (0.1136-0.1137) | 32 (0.0120-0.0138) | 16 (0.0220-0.0236) | 2 CTX | 2 CTX, 12 VAL (mass 0.502) |
| ALLPOOL | 16 (0.0128-0.0151) | 32 (0.0120-0.0138) | 16 (0.0220-0.0236) | 16 VAL | 10 CTX, 16 VAL (mass 0.509) |

Phase 0, positional solvability (no memory; held-out accuracy at the end, seeds 2800 / 2801; rule: stopped if > 0.6):

| task | NOMEM (zero read) | NOMEM_SELF (own V only) | stopped |
|---|---|---|---|
| LITE | 0.166 / 0.168 | 0.407 / 0.374 | (reference) |
| POOL | 0.167 / 0.168 | 0.401 / 0.398 | None |
| ALLPOOL | 0.169 / 0.169 | 0.381 / 0.372 | None |

Phase 1, validity pilots (the perfect gate, SLOW; transition per seed, or final accuracy if unbound):


READINGS (per task): **POOL: INCOMPLETE, no reading**; **ALLPOOL: INCOMPLETE, no reading**

Phase 2, baselines (seeds 2802-2811; no reading; ROUTED*@end = end margin >= 0.9 and η² by stream at KEY > 0.9):

| task | arm | n | BOUND | ROUTED-BOUND | DISCOVERED | transitions (median; all) | unbound: final acc / class |
|---|---|---|---|---|---|---|---|

Per run (BASELINE: end margin / η² by stream at KEY / η² by key at KEY; B = bound):


F/s89_tasks: 12 runs ok (failed records 0), 1.57 h of run time; CPU ['Intel(R) Xeon(R) Processor @ 2.80GHz']; code SHA ea9728c636c6; decision {}
