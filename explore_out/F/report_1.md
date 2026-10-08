# Session F — report 1 (S53 delta_controls). EXPLORATORY, not a result

Branch `claude/explore-F`, code SHA 429221e60279 (screen) on Intel Xeon @ 2.10GHz, torch 2.14.0+cu130, 1 thread per run.
Seeds 300-309 (P=4) and 300-302 (P=8). 106/106 runs ok, none retried.

## Counts (BOUND; Wilson 95%; band)

| arm (k, memory) | grouped P=4 | header P=4 | transitions (median) G / H |
|---|---|---|---|
| SINGLE_HEBB (k=1, Hebbian) | 0/10 [0.00, 0.28] NEVER | 1/10 [0.02, 0.40] MINORITY | -- / 24000 |
| SINGLE_DELTA (k=1, β=1, L2 keys) | **4/10** [0.17, 0.69] MINORITY | **8/10** [0.49, 0.94] MAJORITY | 3600 / 2400 |
| SINGLE_DELTA_L (k=1, learned β) | 0/10 [0.00, 0.28] NEVER | 0/10 [0.00, 0.28] NEVER | -- / -- |
| ORACLE_HEBB (perfect gate, k=2) | 10/10 [0.72, 1.00] RELIABLE | 10/10 [0.72, 1.00] RELIABLE | 1200 / 1200 |
| ORACLE_DELTA (perfect gate, β=1) | 10/10 [0.72, 1.00] RELIABLE | 10/10 [0.72, 1.00] RELIABLE | 1200 / 1200 (all ten at 1200) |

| header P=8, 48000 updates | seed 300 | 301 | 302 | BOUND |
|---|---|---|---|---|
| ORACLE_HEBB_P8 | 24000 | 14400 | 18000 | 3/3 [0.44, 1.00] |
| ORACLE_DELTA_P8 | 6000 | 2400 | 4800 | 3/3 [0.44, 1.00] |

Paired (one-sided exact McNemar): SINGLE_DELTA vs SINGLE_HEBB grouped 4 vs 0 (p = 0.0625), header 7 vs 0 (p = 0.0078);
SINGLE_DELTA_L vs SINGLE_HEBB header 0 vs 1. Failure classes do not apply (k = 1 has no gate; the oracle arms all bound).

## Readings (fixed in the docstring before any run)

- Validity: ORACLE_DELTA bound 3/3 of seeds 300-302 on both layouts (VALID); ORACLE_HEBB 3/3 on both.
- **R1 "delta makes the single channel fail": does not apply → "neither".** The rule needed SINGLE_DELTA ≤ 1/10 on both
  layouts and SINGLE_HEBB ≥ 3/10 on one. The opposite happened: the single delta channel binds *more* often than the
  Hebbian one (8/10 vs 1/10 on the header layout, 4/10 vs 0/10 grouped).
- **R2 "delta raises the oracle": does not apply → "neither".** The delta oracle binds the P=8 header layout on 3/3, but
  so does the Hebbian oracle within 48000 (0 seeds where only delta binds; rule ≥ 2). Delta binds about 4× sooner
  (median 4800 vs 18000).
- Write order (descriptive): SINGLE_DELTA does **not** keep only the last-written stream. Its bound runs answer both
  streams: acc 0.99 on last-written pairs and 1.00 on the others (median, header); unbound runs sit near 0.50 on both
  (grouped), i.e. it does not fail by overwriting.

## What this means (post hoc, labelled as such)

The prediction of distant_cues §4.1 ("a single delta channel provably fails on all but the last-written stream") assumes
the two writes of a key use the same key vector. In BDH the key is the layer's learned sparse code at the VAL position,
which can carry the stream. Post-hoc diagnostic (`explore_f_s53_keydiag.py`, `s53_keydiag.json`; seed 307 rerun, curves
reproduce the records): the median cosine between the two writes' normalised keys is 0.90 / 0.92 / **0.16** at layers 1 /
2 / 3 in the trained single delta channel (Hebbian single channel: 0.44 / 0.70 / 0.83). The delta model learns
near-orthogonal stream-tagged keys at layer 3; the erase then leaves the other stream's binding intact, and the
least-squares read separates them. **Consequence for the program:** with learned keys a delta single channel is *not* a
"provably fails" negative control, and "bound ⇒ routed" does not hold by construction (S54/G should keep reporting
routing, not just BOUND).

## Anything unexpected

- The plain Hebbian perfect gate (no conv) binds the grouped two-stream layout 10/10 here (1200-6000). Batch 18's dry
  run saw it fail on seeds 160-161 (0.54, 0.33 after 9600); seed-dependent.
- The Hebbian perfect gate binds the P=8 header layout by 24000 on 3/3 (S28 saw 0/3 at 24000 on 160-162: 0.87, 0.86,
  0.55) — S28's "validity failed" was close to the budget edge.
- Learned β (σ(w·v + b), 0.5 at init) never binds as a single channel (0/20, accuracy ≈ 0.22 grouped, 0.32 header). It
  learns β ≈ 0.00 at KEY positions in every layer, 0.03-0.35 at VAL, 0.3-0.65 at CTX: it mostly stops writing.
- The delta oracle binds every P=4 run at the first evaluation (1200).

## CHECKs

`explore_f_delta_checks.py` passed in full (26/26) before TASK 0 was pushed, at both S53 segment starts and at S54's
start (`checks_*.json`). Among them: β = 0 (erase off, write 1) equals the Hebbian model to ≤ 1.3e-7 in the logits on four
architectures; with the new path disabled (impl "hebb") batch 16's LOCAL3_SLOW|160 reproduces bit for bit through 2400;
the oracle's cross-stream contributions are exactly 0 (fixed and routed decay, both layouts, both implementations);
β = 1 overwrite returns v2 to 1.6e-6; parallel = sequential to 4e-16 (gradients 2e-15). The screen's own CHECKs (models
as stated; paired arms share every initial parameter) passed.

## Runtime

7.67 h of run time, 2.0 h wall on 4 workers (04:15-06:15 UTC), two segments.

## Notes

- RandHeaderTask (for S54(d) and Session G) is in `explore_f_tasks.py`: P = 8 keys per stream in 2 blocks per stream,
  each block uniform on 2..6 pairs, 4 blocks in random order, one CTX per block, L = 39 (my resolution of the
  specification; every (stream, key) once).
- S54's delta arms use `explore_f_delta_fast.py` (same numbers as the parallel form; 2.3× faster at k = 16).
- S54 started 06:15 UTC; S59 follows.
