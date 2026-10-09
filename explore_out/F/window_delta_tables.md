### S54 (a), (d) — counts (Wilson 95%; band)

| arm | n | BOUND | DISCOVERED | ROUTED*@end | failures (unbound) | transitions: median (all) | final acc median |
|---|---|---|---|---|---|---|---|
| A_DELTA | 40/40 | 36/40 [0.77, 0.96] RELIABLE | 36/40 [0.77, 0.96] RELIABLE | 36/40 [0.77, 0.96] RELIABLE | {'KEY': 4} | 1200 (1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 2400, 2400, 2400, 2400) | 1.00 |
| A_ORACLE_DELTA | 2/2 | 2/2 [0.34, 1.00] RELIABLE | -- | -- | -- | 1200 (1200, 1200) | 1.00 |
| A_ORACLE_HEBB | 2/2 | 2/2 [0.34, 1.00] RELIABLE | -- | -- | -- | 1200 (1200, 1200) | 1.00 |
| D_DELTA | 20/20 | 0/20 [0.00, 0.16] NEVER | 0/20 [0.00, 0.16] NEVER | 0/20 [0.00, 0.16] NEVER | {'KEY': 16, 'OTHER': 4} | -- | 0.22 |
| D_HEBB | 20/20 | 0/20 [0.00, 0.16] NEVER | 0/20 [0.00, 0.16] NEVER | 0/20 [0.00, 0.16] NEVER | {'OTHER': 10, 'KEY': 10} | -- | 0.27 |
| D_ORACLE_DELTA | 2/2 | 0/2 [0.00, 0.66] NEVER | -- | -- | -- | -- | 0.24 |
| D_ORACLE_HEBB | 2/2 | 0/2 [0.00, 0.66] NEVER | -- | -- | -- | -- | 0.68 |

Validity (perfect gate binds on >= 1 of its 2 seeds):

- A_ORACLE_DELTA: 2/2 -> VALID (transitions [1200, 1200])
- A_ORACLE_HEBB: 2/2 -> VALID (transitions [1200, 1200])
- D_ORACLE_DELTA: 0/2 -> NOT VALID (transitions [None, None])
- D_ORACLE_HEBB: 0/2 -> NOT VALID (transitions [None, None])

READING (a): **delta better (and not worse)** (A_DELTA DISCOVERED 36/40 vs recorded Hebbian LOCAL3_SLOW 31/40 on 160-199, counts only; rules: not worse >= 29, better >= 35; one-sided Fisher p(delta > hebb) = 0.112, p(hebb > delta) = 0.967)

(d) BASELINE (RandHeaderTask, LOCAL3 + SLOW, k = 2, 24000):

- BASELINE D_HEBB: DISCOVERED 0/20 [0.00, 0.16] NEVER; BOUND 0/20 [0.00, 0.16] NEVER; ROUTED*@end 0/20 [0.00, 0.16] NEVER; failures {'OTHER': 10, 'KEY': 10}; validity False
- BASELINE D_DELTA: DISCOVERED 0/20 [0.00, 0.16] NEVER; BOUND 0/20 [0.00, 0.16] NEVER; ROUTED*@end 0/20 [0.00, 0.16] NEVER; failures {'KEY': 16, 'OTHER': 4}; validity False
- paired DISCOVERED: D_DELTA 0/20 vs D_HEBB 0/20: delta only 0, hebb only 0; p(delta > hebb) = 1, p(hebb > delta) = 1
- paired BOUND: D_DELTA 0/20 vs D_HEBB 0/20: delta only 0, hebb only 0; p(delta > hebb) = 1, p(hebb > delta) = 1
- paired ROUTED*@end: D_DELTA 0/20 vs D_HEBB 0/20: delta only 0, hebb only 0; p(delta > hebb) = 1, p(hebb > delta) = 1

η² of the read gate at the end (median over runs; by stream / key / half / index), KEY and VAL positions; margin:

- A_DELTA: KEY 1.00/0.00/0.00/0.00; VAL 1.00/0.00/0.00/0.00; margin 0.99
- D_DELTA: KEY 0.00/0.98/0.00/0.00; VAL 0.00/1.00/0.00/0.00; margin 0.00
- D_HEBB: KEY 0.00/0.38/0.00/0.05; VAL 0.00/1.00/0.00/0.00; margin 0.00

Learned β at CTX/KEY/VAL per layer, end (median): A_DELTA L1 0.82/0.76/0.74; L2 0.84/0.21/0.68; L3 0.85/0.12/0.69; D_DELTA L1 0.02/0.00/0.04; L2 0.02/0.00/0.03; L3 0.02/0.00/0.03

A_DELTA gate-state decodability at KEY / VAL (median; chance 0.5): 1200 1.00/1.00; 2400 1.00/1.00; 4800 0.75/0.75; end 1.00/1.00

Per seed (outcome tag; transition):

- A_DELTA: 310 DISCOVERED@1200, 311 DISCOVERED@1200, 312 KEY 0.63, 313 DISCOVERED@1200, 314 DISCOVERED@1200, 315 DISCOVERED@1200, 316 DISCOVERED@1200, 317 DISCOVERED@2400, 318 DISCOVERED@1200, 319 DISCOVERED@1200, 320 DISCOVERED@1200, 321 DISCOVERED@1200, 322 DISCOVERED@1200, 323 DISCOVERED@1200, 324 DISCOVERED@1200, 325 DISCOVERED@1200, 326 DISCOVERED@1200, 327 DISCOVERED@1200, 328 DISCOVERED@2400, 329 DISCOVERED@1200, 330 DISCOVERED@1200, 331 KEY 0.88, 332 DISCOVERED@1200, 333 KEY 0.48, 334 DISCOVERED@1200, 335 DISCOVERED@1200, 336 DISCOVERED@1200, 337 DISCOVERED@1200, 338 DISCOVERED@2400, 339 DISCOVERED@1200, 340 DISCOVERED@1200, 341 KEY 0.50, 342 DISCOVERED@1200, 343 DISCOVERED@2400, 344 DISCOVERED@1200, 345 DISCOVERED@1200, 346 DISCOVERED@1200, 347 DISCOVERED@1200, 348 DISCOVERED@1200, 349 DISCOVERED@1200
- D_DELTA: 330 KEY 0.20, 331 KEY 0.23, 332 OTHER 0.22, 333 OTHER 0.23, 334 KEY 0.22, 335 KEY 0.23, 336 KEY 0.22, 337 KEY 0.23, 338 KEY 0.22, 339 OTHER 0.24, 340 KEY 0.21, 341 KEY 0.20, 342 KEY 0.23, 343 KEY 0.22, 344 KEY 0.21, 345 KEY 0.23, 346 KEY 0.23, 347 OTHER 0.21, 348 KEY 0.22, 349 KEY 0.23
- D_HEBB: 330 OTHER 0.21, 331 OTHER 0.27, 332 KEY 0.32, 333 OTHER 0.32, 334 OTHER 0.27, 335 KEY 0.27, 336 OTHER 0.25, 337 KEY 0.29, 338 KEY 0.24, 339 OTHER 0.32, 340 KEY 0.27, 341 KEY 0.27, 342 KEY 0.27, 343 KEY 0.24, 344 OTHER 0.31, 345 KEY 0.25, 346 OTHER 0.24, 347 OTHER 0.32, 348 OTHER 0.33, 349 KEY 0.27

Runs: 88 ok (failed records 0), 10.76 h of run time; CPU ['Intel(R) Xeon(R) Processor @ 2.10GHz']; git ['16c6277', '16c6277+dirty', '1af0eae+dirty', '1f0835f', '1f0835f+dirty', '2a93d6f', '2a93d6f+dirty', '3643818', '3643818+dirty', '5c1f180', '5c1f180+dirty', 'b77e8bb', 'b77e8bb+dirty', 'c6a1e56', 'c6a1e56+dirty']; code SHA 30feac285d87
