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
