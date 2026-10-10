# Session F — report 5 (S76 eight streams on delta channels). EXPLORATORY, not a result

Branch `claude/explore-F`, code SHA 2336ceff7760 (main-line modules at 9c5939e: test_window_gate's Part C/D path, run_sc
with D8; S=8, P=4, k=16, conv, 43200), Intel Xeon @ 2.10GHz, torch 2.14.0+cu130, 1 thread per run. **Seeds: the assigned
block 1000-1059 appears in recorded runs (results/X/router_reliability_results.json), and so do 1100-1159 and 1200-1259.
Shifted by 100 three times to 1300-1359**: Phase 1 1300-1303, Phase 2 1310-1319. 40 runs: 39 ok and 1 diverged
(O_B1_RAW|1300, the same non-finite loss on all 3 attempts).

## Counts (Wilson 95%; band)

**Phase 1, the perfect gate (seeds 1300-1303), BOUND and transition per seed:**

| arm | BOUND | 1300 | 1301 | 1302 | 1303 | seeds within 2x of Hebbian |
|---|---|---|---|---|---|---|
| O_HEBB (reference) | 4/4 [0.51, 1.00] | 2400 | 3600 | 2400 | 6000 | — |
| O_B1_L2 (β 1, L2 keys) | 4/4 [0.51, 1.00] | **1200** | **1200** | **1200** | **1200** | 4/4 |
| O_B025_L2 (β 0.25, L2) | 4/4 [0.51, 1.00] | 2400 | 2400 | 2400 | 2400 | 4/4 |
| O_B1_RAW (β 1, raw keys) | 3/4 [0.30, 0.95] | DIVERGED | 16800 | 4800 | 2400 | 2/4 |
| O_B025_RAW (β 0.25, raw) | 4/4 [0.51, 1.00] | 7200 | 1200 | 2400 | 1200 | 3/4 |

**Phase 2, the recipe (seeds 1310-1319; LOCAL3 + SLOW + H's exact-copy split; paired):**

| arm | BOUND ROUTED | outcomes | MERGED | transitions of BOUND ROUTED: median (all) | splits (total; runs; targets labelled ok) |
|---|---|---|---|---|---|
| HEBB_SPLIT | **10/10** [0.72, 1.00] RELIABLE | BOUND ROUTED 10 | 0 | 23400 (13200, 20400, 22800 ×3, 24000, 25200, 28800, 36000, 40800) | 22; 10; 22/22 |
| DELTA_SPLIT (DELTA* = β 1, L2) | **9/10** [0.60, 0.98] RELIABLE | BOUND ROUTED 9, MERGED (2 share) 1 | 1 (1313) | **14400** (9600, 12000, 12000, 13200, 14400, 14400, 15600, 20400, 20400) | 12; 9; 6/12 |

Paired: delta only 0, Hebbian only 1 (1313); one-sided McNemar p(hebb > delta) = 0.5. On the 9 seeds both bind, delta binds
sooner on 8 (median 14400 vs 24000).

## Readings (fixed in the docstring before any run) and which applies

- **Phase 1 "neither".** "The slowness is the overwrite" needed the β = 0.25 arms within 2× and the β = 1 arms not;
  "the slowness is the key normalisation" needed the raw arms within 2× and the L2 arms not. Instead every L2 arm meets
  the mark on 4/4, with β = 1 the fastest, binding all four seeds at the first evaluation (1200), sooner than the Hebbian
  oracle. With the perfect gate the delta memory is *not* slow at eight streams when β is fixed. Raw keys are the
  unstable variant: one divergence and one late bind at β = 1.
- **DELTA\*** = O_B1_L2 (β = 1, L2 keys; median transition 1200), chosen by the fixed rule and committed (1db6d28) before
  Phase 2.
- **Phase 2 "the delta recipe keeps eight streams"**: c − b = 1 − 0 = 1 ≤ 1. (S54(c) was "delta worse", 3/10 vs 9/10.)

## What changed since S54(c) (post hoc; not separated by this screen)

S54(c) differed in two things: learned β (σ(w·v + b), 0.5 at init) instead of β = 1, and S48's noisy split with an Adam
reset instead of H's exact copy. Its delta oracle bound at 22800 / 28800, and this one at 1200. The oracle comparison
holds the routing fixed, so learned β is the likely cause of S54(c)'s slow delta memory. It matches the learned-β
collapses seen in S53 and S54(d). The split operation is not separated here: H's S61 showed the exact copy is enough on
Hebbian channels, and both arms here use it.

## Splits and merges

Every Hebbian split targeted a merged channel onto an empty one (22/22 labelled ok; 1-3 per run). On delta, 6 of 12
splits fired with the target not labelled ok (c* held < 2 streams or c0 was not empty at the check), mostly as single
splits in runs that then bound within 4800-8400 updates. The delta runs' plateaus trigger the split before the map has
merged. One delta run (1313) used all three splits on correctly labelled targets and ended MERGED (2+2+2+1+1) at 0.62.
1318 bound at 9600 without any split.

## Anything unexpected

- With β fixed at 1 the delta perfect gate is faster than the Hebbian one at eight streams (1200 vs 2400-6000). S54's
  "slow delta oracle" was a learned-β artefact.
- Raw keys are numerically fragile for the delta rule (β ‖x‖² > 2 makes I − β x xᵀ expanding); the divergence guard stopped
  O_B1_RAW|1300 at its first non-finite evaluation, and all 3 harness attempts diverged identically.

## CHECKs

explore_f_delta_checks passed 26/26 at both phases: β = 0 equals the Hebbian memory, and the recorded LOCAL3_SLOW|160
reproduces with the delta path disabled. The screen's CHECKs:
- HEBB_SPLIT on seed 487 reproduces H's COPY_NONOISE|487 (claude/explore-H 7e9197f, 2.10GHz) bit for bit through 7200:
  curve, statistics, and trigger rows including the first split at 4800.
- In Phase 2, DELTA_SPLIT with its delta path disabled does too.
- copy_nonoise_op copies the row exactly and leaves the other rows and the Adam state untouched (H's check).
- The oracle variants are as stated, and β = 0 with raw keys equals the Hebbian oracle exactly.

H's split and trigger are copied from claude/explore-H at 7e9197f.

## Runtime

31.9 h of run time. Phase 1 05:58-08:01 UTC and Phase 2 08:01-15:19 UTC on 10 October (9.3 h wall on 4 workers). No
restarts.
