# Session F — report 7 (S81 the split's trigger on delta memory). EXPLORATORY, not a result

Branch `claude/explore-F`. Readings committed at 451cde9 before any stored run. Code SHA bdd8269d38af (main-line modules at
9c5939e). Intel Xeon @ 2.10GHz, torch 2.14.0+cu130, 1 thread per run. 42/42 runs ok, none retried, none diverged.

Setup: S = 8, P = 4, k = 16, conv, 43200; S76's DELTA_SPLIT path on DELTA* (β 1, L2 keys); seeds 2020-2029, paired.
Full tables, with every firing: `trigger_tables.md`.

## Counts (Wilson 95%; band)

| arm | BOUND ROUTED | vs BASE: arm only / BASE only | firings (total; runs) | labelled ok | firings at a disagreement | transitions (median; all) |
|---|---|---|---|---|---|---|
| BASE (64-sequence probe) | 9/10 [0.60, 0.98] RELIABLE | — | 19; 9 | 10/19 (53%) | 7 | 15600 (7200-32400) |
| GUARD (probe, no split below 0.5) | 7/10 [0.40, 0.89] MAJORITY | 0 / 2 (2020, 2025); p = 0.25 | 6; 3 | **6/6 (100%)** | 0 | 13200 (7200-15600) |
| EVAL (2048-sequence evaluation) | **10/10** [0.72, 1.00] RELIABLE | 1 (2029) / 0; p = 0.5 | 13; 9 | 8/13 (62%) | 5 | 15600 (7200-31200) |
| GUARD_EVAL (both) | 8/10 [0.49, 0.94] MAJORITY | 1 (2029) / 2 (2020, 2025) | 9; 3 | 9/9 (100%) | 4 | 13200 (7200-31200) |

- **Validity:** the delta perfect gate bound on 2020 (at 6000) and 2021 (at 1200). VALID.
- **Failures:** every unbound run is MERGED (2 share), the same under v1 and v2. They are BASE 2029; GUARD 2020, 2025,
  2029; GUARD_EVAL 2020, 2025.
- **Disagreement:** the rule's decision differs depending on whether it reads the probe or the evaluation.

## Readings (fixed in the docstring before any run) and which applies

- **"The guard aims the split": does not apply.**
  - Its first condition holds: GUARD's firings are labelled ok on 6 of 6.
  - Its second fails: BASE alone binds on 2 seeds and GUARD alone on none, so c − b = 2 > 1.
- **"The guard costs binding": does not apply** (c = 2 < 4).
- **"The evaluation is the better probe": does not apply.**
  - Its first condition holds: EVAL 10/10 vs BASE 9/10, b = 1, c = 0.
  - Its second fails: 5 of EVAL's 13 firings came at a check where the probe would have decided otherwise.
- **"The evaluation costs binding": does not apply** (c = 0).
- **So: no reading applies to either device.** Together they are not "both cost binding" either (GUARD_EVAL c = 2).
- **The disagreement clause is my definition.** I fixed it before the runs as "the rule's eligibility read on the probe
  differs from it read on the evaluation". Under the simpler rule Session H's S84 uses (EVAL ≥ BASE − 1 pair, with no
  disagreement clause), EVAL would read "the evaluation is the better probe".
  - Of EVAL's 5 such firings, 3 were in the flat start (2022 at 9600, 2027 and 2028 at 4800).
  - In each of those, the probe's rise on 64 sequences differed from the evaluation's.
  - The other 2 came later (2026 at 19200, 2029 at 21600).

## What happened (post hoc; one screen, ten seeds)

- **On delta channels, the flat-start splits are not wasted.**
  - BASE made 13 of its 19 firings while accuracy was still below 0.5 (at 4800-12000), 6 of them labelled ok. All 8 runs
    that split in the flat start bound.
  - The guard removed exactly those firings: 13 eligible checks were held back in GUARD and 18 in GUARD_EVAL.
  - On the two seeds where BASE's 4800 split was labelled ok and GUARD then failed (2020, 2025), the guarded runs merged
    later. Their late splits (31200-40800, all labelled ok) did not separate the merged streams.
  - This is not H's flat-start mechanism. That was a split spent early under a cap of one, leaving none for the later
    merge. Here three splits remain, and an early copy may simply prevent a merge from forming.
  - "Labelled ok" in the flat start is read on a map that exists before anything has bound.
- **R's seed-914 failure appears, in GUARD 2025.** At 5 checks between 28800 and 40800 the probe read 0.95-0.98 while
  the evaluation was at 0.87-0.92, so the rule could not fire. It fired only once, at 31200, where the probe read 0.94.
  The run ended MERGED at 0.93.
  - In BASE it occurred at 2 checks (2025 at 14400, 2027 at 31200), both shortly before those runs bound.
  - With the evaluation as the reading (GUARD_EVAL 2025), the rule did fire there: three splits, all labelled ok. The run
    still merged (0.92).
- **EVAL's one win (2029).** BASE and GUARD fired at 14400, 24000 and 28800 and merged. EVAL fired its second split at
  21600: the evaluation was flat (0.868 to 0.870), while the probe rose from 0.75 to 0.78 (≥ 0.02, so the probe would not
  have fired). EVAL then bound at 31200, as did GUARD_EVAL, which follows the same path.
- **BASE's labelled-ok rate:** 10/19 (53%), as in S76 (6/12). The guard lifts it to 100% by leaving only the post-flat-start
  firings, but on these seeds it gives up the early ones that helped.

## CHECKs (all passed, asserted before any run)

- explore_f_delta_checks 26/26 (β = 0 equals the Hebbian memory; the recorded LOCAL3_SLOW|160 reproduced with the delta
  path disabled).
- The child's checks:
  - BASE on seed 1315 reproduces S76's DELTA_SPLIT|1315 bit for bit through 7200 (its first split at 4800 + one check):
    curve, statistics and trigger rows.
  - Each device with its condition disabled equals BASE bit for bit through the same 7200, curve, statistics and trigger
    rows: GUARD with the guard at 0, GUARD with no guard, EVAL reading the probe, GUARD_EVAL with both disabled.
  - EVAL's reading at every check is the curve's held-out accuracy exactly, and the captured evaluations equal the curve.
  - The trigger's mechanics on an untrained model (BASE fires at 4800, 9600 and 14400; GUARD never fires; GUARD_EVAL on
    readings 0.3 then 0.6 fires at 7200, 12000 and 16800).
  - The perfect gate on DELTA* is one-hot.
  - DELTA* is as committed.
- **Disclosure:** before the readings' commit (but with the readings already written in the docstring), I ran one
  2400-update smoke test of GUARD_EVAL on seed 2029 to exercise the evaluation capture. It is not stored and no number here
  uses it.

## Runtime

69.2 h of run time. 20:41 UTC 10 October to 17:37 UTC 11 October: about 21 h wall on 4 workers, including 55 min of
CHECKs. The longest runs were the unbound ones, which ran to 43200 (up to 6.0 h each).

## Anything unexpected

- The direction of the guard's effect. H's flat-start reading predicted that guarding would remove wasted splits. On delta
  channels the removed splits were mostly useful, and the guard's only effect on outcomes was to lose 2 seeds.
- How often the probe and the evaluation disagree in the flat start. At accuracies of 0.05-0.15 the 64-sequence probe's
  rise test is mostly noise.
  - BASE and EVAL had 20 and 17 disagreeing checks, 12 of each in the flat start.
  - The guarded arms had 10 and 9, none in the flat start, where the guard makes both readings ineligible.
