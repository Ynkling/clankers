# Session H — report 9 (EXPLORATORY, not a result)

## (1) S87a explore_h_allmass2 — S=8, P=4, k=16, conv, 43200, Hebbian, exact copy, seeds 2600-2619 (as S86)

| arm | BOUND ROUTED | band | Wilson 95% | failures v1 | failures v2 | transition median (all) | splits | key-label ok | all-label ok | own pair ok | pairs equal | map 1:1 before a split |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BASE | 17/20 | MAJORITY | 0.64-0.95 | MERGED 3 | MERGED (2 share) 2, MERGED (3 share) 1 | 18000 (9600, 10800, 15600, 15600, 15600, 16800, 18000, 18000, 18000, 18000, 19200, 20400, 20400, 22800, 22800, 28800, 30000) | 41 | 41/41 | 39/41 | 41/41 | 17/41 | 0 |
| ALLMASS | 16/20 | MAJORITY | 0.58-0.92 | MERGED 4 | MERGED (2 share) 3, MERGED (3 share) 1 | 18600 (10800, 15600, 15600, 16800, 18000, 18000, 18000, 18000, 19200, 20400, 20400, 20400, 21600, 24000, 28800, 30000) | 42 | 42/42 | 40/42 | 40/42 | 15/42 | 0 |

ORACLE (ceil8_D8, seeds 2600-2601): bound 2/2, transitions {2600: 2400, 2601: 8400} → VALID.

- ALLMASS vs BASE (20 new pairs): 16/20 vs 17/20; ALLMASS only 1, BASE only 2; p(ALLMASS>BASE) 0.875, p(BASE>ALLMASS) 0.5
- pooled with S86 (30 pairs, no reading): ALLMASS 24/30, BASE 26/30; ALLMASS only 1, BASE only 3; p(ALLMASS>BASE) 0.938, p(BASE>ALLMASS) 0.312

**Readings (fixed in explore_h_allmass2's docstring before any run):**

- keeps: **keeps it (BASE only 2, ALLMASS only 1)**
- costs: **no cost by the rule (BASE only 2, ALLMASS only 1)**
- aims: **aims (ALLMASS firings labelled ok 40/42)**

ALLMASS firings whose own pair was not a labelled target (seed, update, key c*->c0, all c*->c0, key_ok, all_ok): [(2611, 14400, 7, 14, 13, 3, True, False), (2618, 19200, 6, 15, 5, 15, True, False)]; BASE's: none. v1/v2 differences: none.

## (2) The one-to-one guard on the main line's split (no runs)

```
(a) the 50 runs: the main line at 0282955 (S86's set)
  file           runs firings key ok all ok map 1:1 before runs BR at end
  window_gate X    20      47     47     46              0             18
  window_gate L    20      38     38     30              0             20
  split_copy X     10      22     22     21              0             10
  total            50     107    107     97              0             48
  REPRODUCES S86's count (50 runs, 107 firings, key pair 107, all-position pair 97): True
(b) now: the main line at 0754d61 (L's split_copy file added)
  file           runs firings key ok all ok map 1:1 before runs BR at end
  window_gate X    20      47     47     46              0             18
  window_gate L    20      38     38     30              0             20
  split_copy X     10      22     22     21              0             10
  split_copy L     10      19     19     18              0              8
  total            60     126    126    115              0             56
  field: checks[].map_before (routing_k's VAL-position map on the run's probe, before the operation)
  firings with a one-to-one map before: (a) 0 of 107; (b) 0 of 126
```

## (3) S87c explore_h_dose — S=2, P=4, k=2, no conv, 24000, WIN3_SLOW + the key term, seeds 2630-2639

| arm | lam | window | DISCOVERED | band | Wilson 95% | unbound: v1 | unbound: v2 | transition median (all) | max eta^2 by key (25..window) | key split > 0.5 at the end | firings (per run) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| KEYHINGE | 1.0 | 2400 | 10/10 | RELIABLE | 0.72-1.00 | — | — | 4800 (3600, 4800, 4800, 4800, 4800, 4800, 6000, 6000, 7200, 7200) | 0.130 | 0 | 2, 1, 1, 3, 5, 3, 2, 3, 4, 7 |
| LAM03 | 0.3 | 2400 | 10/10 | RELIABLE | 0.72-1.00 | — | — | 4800 (3600, 3600, 3600, 4800, 4800, 4800, 4800, 4800, 4800, 6000) | 0.153 | 0 | 4, 1, 1, 3, 5, 3, 2, 3, 4, 7 |
| LAM3 | 3.0 | 2400 | 10/10 | RELIABLE | 0.72-1.00 | — | — | 6600 (4800, 4800, 4800, 6000, 6000, 7200, 7200, 8400, 9600, 9600) | 0.138 | 0 | 2, 1, 1, 3, 5, 3, 2, 3, 4, 7 |
| WIN1200 | 1.0 | 1200 | 10/10 | RELIABLE | 0.72-1.00 | — | — | 4800 (3600, 4800, 4800, 4800, 4800, 4800, 6000, 6000, 7200, 7200) | 0.130 | 0 | 2, 1, 1, 3, 5, 3, 2, 3, 4, 7 |

ORACLE (ceiling_conv, seeds 2630-2631): bound 2/2, transitions {2630: 1200, 2631: 1200} → VALID.

- LAM03 vs KEYHINGE: 10/10 vs 10/10; LAM03 only 0, KEYHINGE only 0; p(LAM03>KEYHINGE) 1, p(KEYHINGE>LAM03) 1
- LAM3 vs KEYHINGE: 10/10 vs 10/10; LAM3 only 0, KEYHINGE only 0; p(LAM3>KEYHINGE) 1, p(KEYHINGE>LAM3) 1
- WIN1200 vs KEYHINGE: 10/10 vs 10/10; WIN1200 only 0, KEYHINGE only 0; p(WIN1200>KEYHINGE) 1, p(KEYHINGE>WIN1200) 1

**Readings (fixed in explore_h_dose's docstring before any run):**

- robust: **robust (LAM03 only 0, KEYHINGE only 0; LAM3 only 0, KEYHINGE only 0; WIN1200 only 0, KEYHINGE only 0)**
- amend: **no variant beats the drafted constants by >= 4 vs 0: the draft stands**

v1 / v2 on S87c's unbound runs: differ on 0, exactly the predicted ones: True.

**H/allmass2:** 42 runs, 28.2 CPU-hours on Intel(R) Xeon(R) Processor @ 2.10GHz; code SHA 898867f69ad4; CHECKs all passed (2026-10-12 22:57:03 UTC, git f4fd535)

**H/dose:** 42 runs, 2.3 CPU-hours on Intel(R) Xeon(R) Processor @ 2.10GHz; code SHA 65d3498e4511; CHECKs all passed (2026-10-12 23:00:36 UTC, git f4fd535+dirty)

**Which readings apply.** Both screens are VALID: the S=8 oracle bound 2/2 (at 2400 and 8400), the S=2 oracle 2/2 (at
1200 and 1200). No run failed and none was retried.
- **(1) "Keeps it" applies** on the 20 new pairs: BASE only 2, ALLMASS only 1. **No cost by the rule.** **"Aims" applies**:
  40 of ALLMASS's 42 firings were labelled ok on its own pair (95%).
  Pooled with S86 (30 pairs, printed, no reading): ALLMASS 24/30, BASE 26/30; ALLMASS only 1, BASE only 3.
- **(2) The guard is moot on the main line.** Field: `checks[].map_before`. Over the 50 recorded runs, 0 of 107 firings had
  a one-to-one map before the fire, and 0 of 126 counting L's split_copy runs now on the main line. The recount reproduced
  S86's 107 / 97 first. Per the request, the guarded-trigger screen on 2620-2629 was not run; those seeds are unused.
  - In this session's own eight-stream runs, 2 of 498 firings had a one-to-one map before: S86 seed 2334 at 26400, in
    BASE and EVAL. Neither was labelled ok, the map did not change, and both runs ended BOUND ROUTED.
- **(3) "Robust" applies**: all four arms are 10/10 DISCOVERED, with no discordant pair. The amendment rule does not fire,
  so **the draft's constants stand** (weight 1.0, window 2400).

**What the firings show (S87a)**
- **The losses are not mistargets.** On both BASE-only pairs (2610, 2618), ALLMASS chose a different but still labelled
  pair at an early firing (2610 at 12000: 2 -> 11 against BASE's 5 -> 11; 2618 at 4800: 6 -> 13 against 7 -> 13). From
  there the runs diverge.
  - On 2610 every ALLMASS firing was labelled ok, and the run merged.
  - On 2618 the third firing (19200, 5 -> 15) was not labelled ok, and the run merged. Whether that firing decided it
    cannot be told from one run.
  - The ALLMASS-only pair (2614) also went that way: at 4800 it chose 12 -> 9 where BASE chose 12 -> 14, and ALLMASS bound
    at 30000 while BASE merged.
- **ALLMASS's other mistarget, on 2611, cost nothing.** It fired at 14400 (13 -> 3) and the run still bound, at 20400.
- **The two pairs agree less often than at the main line.** The key-position and all-position pairs were equal at 15-17 of
  ~41 firings per arm. The main line has 46/107 and S86 8/20.
- **The flat start again.** 12-13 of each arm's ~41 firings came at held-out accuracy < 0.2. 6-7 runs per arm used the
  full cap of 3.
- **Two seeds merged under both rules,** 2600 and 2613. Every one of their firings was labelled ok.

**What the dose shows (S87c)**
- **The screen detects only a worse variant.** The drafted arm is at the ceiling (10/10), so no variant could beat it, and
  "robust" here means "none is worse within 10 seeds". No term-free REF arm was run: S85's REF was 15/20 on 2310-2329.
- **The term is mostly idle.** It fired 1-7 times per run, in nearly identical numbers across the arms (LAM03 on 2630:
  4 against 2). The maximum η² by key at KEY positions over updates 25 to the window was 0.13-0.15 in every arm, and at
  the end no run of any arm held a key split above 0.5.
- **The weight moves the transition.** LAM3's median is 6600 (4800-9600) against KEYHINGE's 4800 (3600-7200). LAM03 is
  4800 (3600-6000). WIN1200 is identical to KEYHINGE (same transitions, same firings). This is descriptive: all bound
  anyway.
- **WIN1200 = KEYHINGE run for run** is consistent with the term idling after 1200 on these seeds.

**Classifier.** v1 and v2 agree on every S87a and S87c run.

**Reading for the main line (exploratory).** Over 30 pairs, the all-position target is within two pairs of the key-position
one (24 vs 26). Its losses come from choosing a different labelled pair early, not from a wrong one. The guard question is
empty at k = 16. The key term's constants are not sensitive within 0.3x-3x and window 1200-2400 at two streams; the spec
draft is unchanged.
