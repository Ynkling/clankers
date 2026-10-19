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
| POOL | 0.167 / 0.168 | 0.401 / 0.398 | False |
| ALLPOOL | 0.169 / 0.169 | 0.381 / 0.372 | False |

Phase 1, validity pilots (the perfect gate, SLOW; transition per seed, or final accuracy if unbound):

- O_HEBB_POOL: 2800 unbound 0.62; 2801 unbound 0.48
- O_DELTA_POOL: 2800 bound @3600; 2801 bound @2400
- O_HEBB_POOL_48K: 2800 unbound 0.72; 2801 unbound 0.51
- O_DELTA_POOL_48K: 2800 bound @3600; 2801 bound @2400
- O_HEBB_ALLPOOL: 2800 unbound 0.33; 2801 unbound 0.50
- O_DELTA_ALLPOOL: 2800 bound @3600; 2801 bound @3600
- O_HEBB_ALLPOOL_48K: 2800 unbound 0.32; 2801 unbound 0.48
- O_DELTA_ALLPOOL_48K: 2800 bound @3600; 2801 bound @3600

READINGS (per task): **POOL: untested**; **ALLPOOL: untested**

Phase 2, baselines (seeds 2802-2811; no reading; ROUTED*@end = end margin >= 0.9 and η² by stream at KEY > 0.9):

| task | arm | n | BOUND | ROUTED-BOUND | DISCOVERED | transitions (median; all) | unbound: final acc / class |
|---|---|---|---|---|---|---|---|

Per run (BASELINE: end margin / η² by stream at KEY / η² by key at KEY; B = bound):


F/s89_tasks: 28 runs ok (failed records 0), 4.05 h of run time; CPU ['Intel(R) Xeon(R) Processor @ 2.80GHz']; code SHA ea9728c636c6; decision {'phase0': {'POOL': {'acc': {'NOMEM': [0.1669921875, 0.16796875], 'NOMEM_SELF': [0.40087890625, 0.3984375]}, 'stopped': False}, 'ALLPOOL': {'acc': {'NOMEM': [0.16943359375, 0.1689453125], 'NOMEM_SELF': [0.38134765625, 0.37158203125]}, 'stopped': False}}, 'budget': {'POOL': 'UNTESTED', 'ALLPOOL': 'UNTESTED'}, 'pilots': {'POOL': {'at24': False, 'double': True}, 'ALLPOOL': {'at24': False, 'double': True}}}
