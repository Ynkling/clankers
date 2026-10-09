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
