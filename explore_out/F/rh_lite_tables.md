### S70 RandHeaderTask-lite — counts (Wilson 95%; band)

Phase 1 (oracle first): {'phase1': 'bound', 'oracle_24k': {'HEBB': 3, 'DELTA': 3}, 'budget': 24000}

| arm | n | BOUND | DISCOVERED | ROUTED*@end | failures (unbound) | transitions: median (all) | final acc median |
|---|---|---|---|---|---|---|---|
| S70_ORACLE_HEBB | 3/3 | 3/3 [0.44, 1.00] RELIABLE | -- | -- | -- | 1200 (1200, 1200, 8400) | 1.00 |
| S70_ORACLE_DELTA | 3/3 | 3/3 [0.44, 1.00] RELIABLE | -- | -- | -- | 1200 (1200, 1200, 1200) | 1.00 |
| S70_HEBB | 20/20 | 1/20 [0.01, 0.24] MINORITY | 0/20 [0.00, 0.16] NEVER | 0/20 [0.00, 0.16] NEVER | {'KEY': 19} | 13200 (13200) | 0.50 |
| S70_DELTA | 20/20 | 6/20 [0.15, 0.52] MINORITY | 0/20 [0.00, 0.16] NEVER | 0/20 [0.00, 0.16] NEVER | {'KEY': 14} | 5400 (3600, 3600, 4800, 6000, 7200, 16800) | 0.87 |

**BASELINE for Session G (S70_HEBB, LOCAL3 + SLOW, Hebbian, RandHeaderTask-lite, 24000 updates, seeds 330-349): DISCOVERED 0/20 [0.00, 0.16] NEVER; BOUND 1/20 [0.01, 0.24] MINORITY; ROUTED*@end 0/20 [0.00, 0.16] NEVER; failures {'KEY': 19}**
- paired DISCOVERED: DELTA 0/20 vs HEBB 0/20: delta only 0, hebb only 0; p(delta > hebb) = 1, p(hebb > delta) = 1
- paired BOUND: DELTA 6/20 vs HEBB 1/20: delta only 5, hebb only 0; p(delta > hebb) = 0.0312, p(hebb > delta) = 1
- paired ROUTED*@end: DELTA 0/20 vs HEBB 0/20: delta only 0, hebb only 0; p(delta > hebb) = 1, p(hebb > delta) = 1

η² of the read gate at the end (median; by stream / key / half / index), KEY and VAL; margin:

- S70_HEBB: KEY 0.00/1.00/0.00/0.00; VAL 0.00/1.00/0.00/0.00; margin 0.00
- S70_DELTA: KEY 0.00/1.00/0.00/0.01; VAL 0.00/1.00/0.00/0.00; margin 0.00

Per seed: S70_HEBB 330 KEY 0.50, 331 KEY 0.52, 332 KEY 0.51, 333 KEY 0.50, 334 KEY 0.47, 335 BOUND unrouted@13200, 336 KEY 0.52, 337 KEY 0.50, 338 KEY 0.48, 339 KEY 0.52, 340 KEY 0.50, 341 KEY 0.50, 342 KEY 0.49, 343 KEY 0.51, 344 KEY 0.49, 345 KEY 0.51, 346 KEY 0.47, 347 KEY 0.50, 348 KEY 0.50, 349 KEY 0.49; S70_DELTA 330 KEY 0.87, 331 KEY 0.74, 332 BOUND unrouted@6000, 333 KEY 0.86, 334 BOUND unrouted@7200, 335 BOUND unrouted@4800, 336 KEY 0.77, 337 BOUND unrouted@3600, 338 BOUND unrouted@16800, 339 KEY 0.88, 340 KEY 0.75, 341 KEY 0.86, 342 KEY 0.88, 343 KEY 0.76, 344 KEY 0.74, 345 KEY 0.75, 346 KEY 0.87, 347 KEY 0.88, 348 KEY 0.76, 349 BOUND unrouted@3600

Runs: 46 ok (failed records 0), 5.01 h of run time; CPU ['Intel(R) Xeon(R) Processor @ 2.10GHz']; code SHA ff908bdbc0b2
