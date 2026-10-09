# Session F — report 2 (S54 window gate on delta channels; S59 decay). EXPLORATORY, not a result

Branch `claude/explore-F`; Intel Xeon @ 2.10GHz, torch 2.14.0+cu130, 1 thread per run. Code SHAs: S54(a,d) 30feac285d87,
S54(b,c) c8612c854acc, S59 55653f2b2a4e. 216/216 runs ok, none retried. DELTA = learned β, L2 keys, tied write, decay
0.95 (S54); β = 1 (S59). Full tables: `window_delta_tables.md`, `window_delta_scale_tables.md`, `decay_tables.md`.

## Counts (Wilson 95%; band)

| screen / arm | outcome | DELTA | HEBB | paired b / c (one-sided McNemar) | reading |
|---|---|---|---|---|---|
| S54(a) S=2, P=4, k=2, no conv, 310-349 | DISCOVERED | **36/40** [0.77, 0.96] RELIABLE (failures KEY 4) | recorded LOCAL3_SLOW 31/40 (160-199; counts only) | unpaired; Fisher p(delta > hebb) = 0.11 | **delta better** (≥ 35) |
| S54(b) S=4, P=4, k=16, conv, 310-329 | BOUND ROUTED | 20/20 [0.84, 1.00] RELIABLE | 19/20 [0.76, 0.99] RELIABLE | 1 / 0, p = 0.5 | **delta not worse** |
| S54(c) S=8, P=4, k=16, conv, 310-319 | BOUND ROUTED | 3/10 [0.11, 0.60] MINORITY | 9/10 [0.60, 0.98] RELIABLE | 0 / 6, p(hebb > delta) = 0.016 | **delta worse** |
| S54(d) RandHeaderTask, k=2, 330-349 | DISCOVERED / BOUND | 0/20 / 0/20 [0.00, 0.16] (KEY 16, OTHER 4) | 0/20 / 0/20 (KEY 10, OTHER 10) | 0 / 0 | **UNTESTED** (oracles 0/2) |
| S59 header P=8, 350-359, ORACLE (k=2) | BOUND | FIXED 10/10, ROUTED 10/10, GDN 10/10 | — | — | valid (3/3 each) |
| S59 header P=8, 350-359, SINGLE (k=1) | BOUND | FIXED 3/10 [0.11, 0.60], ROUTED **9/10** [0.60, 0.98], GDN **8/10** [0.49, 0.94] | — | — | **it does not** |

Validity arms (perfect gate, 2 seeds each): (a) DELTA 2/2, HEBB 2/2 at 1200; (b) 2/2, 2/2 at 1200; (c) DELTA 2/2 at
22800 / 28800, HEBB 2/2 at 2400 / 4800; **(d) DELTA 0/2 (final acc 0.24), HEBB 0/2 (0.68)**; S59 FIXED / ROUTED / GDN 3/3.

Transitions (median): (a) DELTA 1200 (32 of 36 at 1200); (b) DELTA 3600, HEBB 3600; (c) DELTA 20400 (18000, 20400, 38400),
HEBB 21600 (18000-32400); S59 ORACLE FIXED 3600, ROUTED 1200, GDN 2400; SINGLE FIXED 14400, ROUTED 10800, GDN 16200.
Failure classes (unbound): (c) DELTA STREAM-PARTIAL one-to-one 6, MERGED 1 (all 8 streams on 8 channels in 9 of 10 at the
end, accuracy 0.32-0.50); HEBB MERGED 1. Splits fired: (c) DELTA 28 in 10 runs, HEBB 26 in 10; (b) DELTA 5, HEBB 2.

## Readings (fixed in the docstrings before any run) and which applies

- **(a) "delta better"**: 36/40 ≥ 35 (and "not worse", ≥ 29). Counts against another seed block and another CPU's
  records; the Fisher p is 0.11, so this is a bound, not a test.
- **(b) "delta not worse"**: c − b = −1 ≤ 2 ("better" needs b ≥ 4).
- **(c) "delta worse"**: c − b = 6 > 2. On delta the S=8 runs mostly reach a one-to-one map (8 streams on 8 channels)
  but stall at accuracy 0.3-0.5; the delta oracle itself is slow here (22800 / 28800, against 2400 / 4800 for Hebbian).
  The limit at eight streams on delta looks like the memory/optimisation, not the routing.
- **(d) UNTESTED, so there is no BASELINE for Session G.** Neither perfect gate binds RandHeaderTask (P = 8 keys per
  stream, 16 pairs) within 24000. My choice of P = 8 made the task too heavy for this budget (the P = 8 header layout's
  Hebbian oracle needs 14400-24000 alone, S53). Session G should not compare against these counts. Options: a lighter
  RandHeaderTask (e.g. P = 4 per stream in 2 blocks, lengths 2 or 2+2), or a longer budget.
- **S59 "the horizon knob matters": does not apply → "it does not".** The rule needed gap_d − gap_FIXED ≥ 0.15 on ≥ 7/10;
  ROUTED 0/10, GDN 1/10. The effect goes the other way: a learned decay makes the *single* channel bind (9/10, 8/10 vs
  3/10), so the oracle gap shrinks (medians −0.46). The conditional rerun under S54(d)'s gate was not made.

## What the decays learned (post hoc, descriptive)

The single delta channel with a learned decay forgets hard at CTX tokens and holds otherwise: median α at CTX / KEY / VAL
in layer 1 is 0.26 / 1.00 / 1.00 (ROUTED) and 0.18 / 1.00 / 1.00 (GDN), rising to 0.72-0.74 at CTX in layer 3. With the
perfect gate, routed decay keeps the unwritten channel at α = 1.000 exactly and the written one at 0.93-0.99. In the
single channel it is "clear (partly) at the context switch", Gated DeltaNet's α_t → 0 behaviour. Like S53's stream-tagged
keys, this is another way a single delta channel solves a task meant to need the partition.

## Anything unexpected

- **Learned β fails.** It collapses towards "do not write" when it is the only change: S53's SINGLE_DELTA_L 0/20, and
  D_DELTA's β ≈ 0.02 at every position. With the window gate on the grouped layout ((a), (b)) it stays at 0.1-0.8 and
  works.
- At S=8 the delta perfect gate binds slowly (22800 / 28800), far behind the Hebbian (2400 / 4800). At S=4 they are equal.
- **Runtime.** The delta memory at k = 16 costs about 3× the Hebbian per step even with the fused kernel
  (`explore_f_delta_fast.py`, same numbers). Under four concurrent runs it ran at 0.4 s/step, about 2× the measured timing.
- **Lost runs and container restarts.** The container was reclaimed twice while the session was idle with only detached
  processes running (07:49 and 08:38 UTC on 8 October). In-flight S54 runs were lost and rerun from the start (S54's
  first two starts saved nothing). From then on a harness-tracked keepalive kept the container alive, with no further loss.

## CHECKs

`explore_f_delta_checks.py` passed in full (26/26) at every batch start, 8 times. The screens' own CHECKs all passed at
every start. Among them:
- (b) and (c)'s Hebbian arms, and their delta arms with the new path disabled, reproduce batch 17's W_SPLIT|260 and
  batch 16's LOCAL3_SLOW16_A|242 bit for bit through 2400 (curve, statistics, decoder).
- The fast kernel equals the parallel form (fp64 forward 0, gradients 9e-16; model gradients 1e-5 relative).
- RandHeaderTask's structural CHECKs pass.
- The groups, the lrs and the decay initialisations are as stated.

## Runtime

S54: 82.4 h of run time; S54 started 08:54 UTC on 8 October (after the restarts) and ended 05:42 UTC on 9 October
(20.8 h wall). S59: 8.4 h of run time, 05:42-08:03 UTC (2.4 h wall).
