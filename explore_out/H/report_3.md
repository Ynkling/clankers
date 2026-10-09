# Session H — report 3 (EXPLORATORY, not a result)

## S61 explore_h_copy2x2 — S=8, P=4, k=16, conv, 43200, LOCAL3 + SLOW + one plateau trigger, seeds 480-489

| arm | BOUND ROUTED | band | Wilson 95% | BOUND (any) | failure classes | transition median (all BR) | ops fired |
|---|---|---|---|---|---|---|---|
| SPLIT | 9/10 | RELIABLE | 0.60-0.98 | 9/10 | MERGED 1 | 19200 (13200, 15600, 16800, 18000, 19200, 22800, 22800, 22800, 25200) | 18 |
| RESET | 4/10 | MINORITY | 0.17-0.69 | 4/10 | MERGED 6 | 22800 (16800, 20400, 25200, 38400) | 26 |
| COPY | 9/10 | RELIABLE | 0.60-0.98 | 9/10 | MERGED 1 | 21600 (15600, 19200, 19200, 20400, 21600, 24000, 25200, 25200, 27600) | 20 |
| COPY_NONOISE | 9/10 | RELIABLE | 0.60-0.98 | 9/10 | MERGED 1 | 20400 (12000, 15600, 15600, 20400, 20400, 22800, 25200, 25200, 31200) | 19 |
| NONE | 3/10 | MINORITY | 0.11-0.60 | 3/10 | MERGED 7 | 27600 (19200, 27600, 32400) | 0 |

ORACLE (ceil8_D8, seeds 480-481): bound 2/2, transitions {480: 4800, 481: 3600} → VALID.

| x vs y | x | y | x only | y only | p(x>y) | p(y>x) |
|---|---|---|---|---|---|---|
| COPY vs SPLIT | 9/10 | 9/10 | 0 | 0 | 1 | 1 |
| COPY vs COPY_NONOISE | 9/10 | 9/10 | 0 | 0 | 1 | 1 |
| RESET vs SPLIT | 4/10 | 9/10 | 0 | 5 | 1 | 0.0312 |
| SPLIT vs NONE | 9/10 | 3/10 | 6 | 0 | 0.0156 | 1 |
| COPY vs NONE | 9/10 | 3/10 | 6 | 0 | 0.0156 | 1 |
| COPY_NONOISE vs NONE | 9/10 | 3/10 | 6 | 0 | 0.0156 | 1 |
| RESET vs NONE | 4/10 | 3/10 | 2 | 1 | 0.5 | 0.875 |

**Readings (fixed before any run):**

- R1: **the copy alone suffices (SPLIT only 0 - COPY only 0 = 0 [<= 1])**
- R2: **does not apply (COPY vs COPY_NONOISE 0 vs 0 [>= 4 vs 0])**
- R3: **the reset adds nothing (|SPLIT only 0 - COPY only 0| = 0 [<= 1])**

## S63 explore_h_hardness — S=8, P=4, k=16, conv, 43200, LOCAL3 + SLOW + KEYMASS, seeds 490-499

| arm | BOUND ROUTED | band | Wilson 95% | BOUND (any) | failure classes | transition median (all BR) | ops fired | earlier than REF (both BR) |
|---|---|---|---|---|---|---|---|---|
| REF | 10/10 | RELIABLE | 0.72-1.00 | 10/10 | — | 20400 (13200, 14400, 19200, 20400, 20400, 20400, 22800, 25200, 27600, 27600) | 20 | — |
| ANNEAL | 9/10 | RELIABLE | 0.60-0.98 | 9/10 | MERGED 1 | 18000 (15600, 15600, 15600, 18000, 18000, 20400, 20400, 27600, 32400) | 23 | 5/10 |
| HARD_W | 0/10 | NEVER | 0.00-0.28 | 0/10 | KEY 3, MERGED 4, OTHER 3 | -- () | 30 | 0/10 |

ORACLE (ceil8_D8, seeds 490-491): bound 2/2, transitions {490: 3600, 491: 4800} → VALID.

- ANNEAL vs REF: 9/10 vs 10/10; ANNEAL only 0, REF only 1; p(ANNEAL>REF) 1, p(REF>ANNEAL) 0.5
- HARD_W vs REF: 0/10 vs 10/10; HARD_W only 0, REF only 10; p(HARD_W>REF) 1, p(REF>HARD_W) 0.000977

**Readings (fixed before any run):**

- ANNEAL: **does not help (ANNEAL vs REF 0 vs 1 [>= 4 vs 0: False]; earlier on 5/10 with BR 9 vs 10 [>= 8 and no loss: False])**
- HARD_W: **does not help (HARD_W vs REF 0 vs 10 [>= 4 vs 0: False]; earlier on 0/10 with BR 0 vs 10 [>= 8 and no loss: False])**

## S62 explore_h_query_gate — the query-KEY gate vs the body maps, reruns of S58 / S58b's bound runs

Per run at its stop: query map (stream -> argmax of the read gate at the query KEY position) = body VAL map, = body KEY map, both, or neither. margin_val = query gate mass on its stream's VAL-map channel minus mean mass on the other streams' VAL-map channels.

| screen.arm | bound runs | reproduced bit for bit | BOUND ROUTED: both / VAL only / KEY only / neither | BOUND NOT routed: both / VAL only / KEY only / neither | margin_val (median) | harness margin (median) |
|---|---|---|---|---|---|---|
| collapse.GUMBEL_RW | 6 | 6/6 | 5 / 0 / 0 / 0 | 1 / 0 / 0 / 0 | 0.99 | 0.982 |
| collapse.GUMBEL_W | 38 | 38/38 | 32 / 1 / 4 / 0 | 0 / 0 / 0 / 1 | 0.738 | 0.699 |
| collapse.NONE | 40 | 40/40 | 36 / 0 / 2 / 0 | 1 / 0 / 0 / 1 | 0.886 | 0.869 |
| collapse.RESET | 40 | 40/40 | 36 / 1 / 1 / 1 | 1 / 0 / 0 / 0 | 0.886 | 0.862 |
| collapse.SPLIT | 40 | 40/40 | 36 / 0 / 1 / 2 | 1 / 0 / 0 / 0 | 0.886 | 0.866 |
| collapse.SWITCH | 40 | 40/40 | 2 / 0 / 24 / 10 | 0 / 0 / 0 / 4 | 0.411 | 0.46 |
| eight.RESET | 2 | 2/2 | 2 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0.959 | 0.934 |
| eight.SPLIT | 10 | 10/10 | 10 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0.959 | 0.94 |

**H/copy2x2:** 52 runs, 33.4 CPU-hours on Intel(R) Xeon(R) Processor @ 2.10GHz; code SHA 0b17627cb1b8; CHECKs all passed (2026-10-08 20:23:29 UTC, git f630ef2)

**H/hardness:** 32 runs, 24.5 CPU-hours on Intel(R) Xeon(R) Processor @ 2.10GHz; code SHA 8240313e6822; CHECKs all passed (2026-10-08 20:41:04 UTC, git 3e4212d)

**H/query_gate:** 216 runs, 26.3 CPU-hours on Intel(R) Xeon(R) Processor @ 2.10GHz; code SHA fb7f8b77db6a; CHECKs all passed (2026-10-08 20:23:32 UTC, git f630ef2+dirty)

**Which readings apply.** All three screens are VALID: each oracle bound both of its seeds within 4800 updates.
- S61: R1 "the copy alone suffices" and R3 "the reset adds nothing" apply. R2 "the noise matters" does not: COPY,
  COPY_NONOISE and SPLIT bind the same 9 seeds and miss the same one (483, MERGED 2 share).
- S63: "hardness helps" does not apply to either arm.
- S62 is descriptive.

**Unexpected / worth a look**
- **The split is the copy, and the copy needs no noise.** Every S61 operation was aimed at a channel holding >= 2 streams
  and copied onto an empty one (targets labelled correct in all 83 firings across the four trigger arms). The exact copy
  (two identical rows, symmetry broken only by the rows' different Adam moments and the data) works as well as the noisy one.
  RESET alone is no better than no trigger (4/10 vs 3/10; 2 vs 1). Together with S58b (SPLIT 10/10, RESET 2/10), the
  KEYMASS split's mechanism is a row copy onto an idle channel. A simplified split (exact copy, no reset, no noise) is the
  natural form for a pre-registered test.
- **A hard write gate destroys routing at eight streams.** HARD_W bound 0/10 (KEY 3, MERGED 4, OTHER 3), although every run
  had the same KEYMASS operations available (30 fired). With the write one-hot from update 2400, the gate gets gradient
  only through the read side, and the stream labelling of the writes never forms.
- **Annealing is neutral.** ANNEAL bound 9/10 against REF's 10/10. Its median transition is 18000 against 20400, earlier on
  5 of the 9 seeds both bound. Not "earlier on >= 8".
- **S62: under the Switch loss the query routes like the body KEY positions, not like the writes.** In 34 of SWITCH's 36
  BOUND ROUTED runs the query map differs from the body VAL map. It equals the body KEY map in 24 and neither map in 10;
  only 2 have the query map equal to the VAL map. These runs still answer at >= 0.9 per stream, because the query's read
  gate keeps enough mass on its stream's VAL channel (margin_val median 0.41). The harness's BOUND ROUTED looks at the VAL
  map only, so it counts them as routed. In every other arm the query map equals both body maps in 32-36 of the bound runs
  (eight streams: 12/12). The few "KEY only" or "neither" cases elsewhere (1-4 per arm) are bound runs, so routing at a
  softer read is enough to answer there too. margin_val tracks the harness margin within about 0.05 in every arm.
- **CHECK failure before any screen run**: S63's REF has no update counter, and its first CHECK pass wrongly required one.
  All three CHECK runs equalled the record bit for bit. Fixed and rerun; log in `s61_s63_check_failed.log`, which also holds
  S61's and S62's passing CHECK output.

**Runtime:** 300 runs, 84 CPU-hours, wall clock 21.1 h on 4 workers (one invocation).
