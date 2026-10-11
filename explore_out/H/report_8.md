# Session H — report 8 (EXPLORATORY, not a result)

## S86 explore_h_allmass — S=8, P=4, k=16, conv, 43200, LOCAL3 + SLOW + exact-copy split, Hebbian, seeds 2330-2339

**First, the main line's 50 split runs** (explore_h_allmass_counts.py, committed before the screen):

```
main line at 0282955; arm WIN3_SPLIT_D8; firings = checks with fired True
file            runs firings key pair ok all pair ok pairs equal both ok
window_gate X     20      47          47          46          23      46
window_gate L     20      38          38          30          16      30
split_copy X      10      22          22          21           7      21
total             50     107         107          97          46      97
```

| arm | BOUND ROUTED | band | Wilson 95% | failures v1 | failures v2 | transition median (all) | splits | key-label ok | all-label ok | own pair ok | pairs equal | accuracy read at the splits |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BASE | 9/10 | RELIABLE | 0.60-0.98 | MERGED 1 | MERGED (2 share) 1 | 19200 (13200, 14400, 15600, 18000, 19200, 21600, 27600, 28800, 28800) | 20 | 19/20 | 17/20 | 19/20 | 8/20 | 0.03, 0.08, 0.08, 0.11, 0.11, 0.19, 0.55, 0.64, 0.67, 0.67, 0.7, 0.73, 0.73, 0.78, 0.8, 0.84, 0.88, 0.88, 0.89, 0.89 |
| ALLMASS | 8/10 | MAJORITY | 0.49-0.94 | MERGED 2 | MERGED (2 share) 2 | 19800 (13200, 14400, 15600, 18000, 21600, 24000, 25200, 26400) | 21 | 21/21 | 20/21 | 20/21 | 8/21 | 0.03, 0.08, 0.08, 0.11, 0.11, 0.19, 0.55, 0.64, 0.67, 0.7, 0.72, 0.73, 0.78, 0.8, 0.81, 0.83, 0.84, 0.86, 0.88, 0.89, 0.92 |
| EVAL | 9/10 | RELIABLE | 0.60-0.98 | MERGED 1 | MERGED (2 share) 1 | 21600 (13200, 14400, 15600, 16800, 21600, 24000, 28800, 30000, 34800) | 20 | 19/20 | 16/20 | 19/20 | 8/20 | 0.08, 0.09, 0.1, 0.11, 0.13, 0.57, 0.59, 0.62, 0.68, 0.74, 0.75, 0.86, 0.86, 0.87, 0.87, 0.88, 0.88, 0.88, 0.88, 0.91 |

ORACLE (ceil8_D8, seeds 2330-2331): bound 2/2, transitions {2330: 6000, 2331: 3600} → VALID.

- ALLMASS vs BASE: 8/10 vs 9/10; ALLMASS only 0, BASE only 1; p(ALLMASS>BASE) 1, p(BASE>ALLMASS) 0.5
- EVAL vs BASE: 9/10 vs 9/10; EVAL only 0, BASE only 0; p(EVAL>BASE) 1, p(BASE>EVAL) 1
- (descriptive) EVAL vs BASE pooled over S84 and S86: 19/20 vs 18/20; EVAL only 1, BASE only 0; p(EVAL>BASE) 0.5, p(BASE>EVAL) 1

**Readings (fixed in explore_h_allmass' docstring before any run):**

- keeps: **all-position targeting keeps it (BASE only 1, ALLMASS only 0)**
- aims: **aims (ALLMASS firings labelled ok 20/21)**
- costs: **no cost by the rule (BASE only 1, ALLMASS only 0)**
- eval: **the evaluation is the better probe (BASE only 0, EVAL only 0)**

ALLMASS firings whose own (all-position) pair was not a labelled target (seed, update, key c*->c0, all c*->c0, key_ok, all_ok): [(2334, 19200, 7, 14, 1, 6, True, False)].
v1/v2 outcome differences: none.

**Runtime:** 32 runs, 17.4 CPU-hours on Intel(R) Xeon(R) Processor @ 2.10GHz; code SHA 898867f69ad4. **CHECKs:** all passed (2026-10-11 03:05:09 UTC, git 50da719, 7.8 min)

**Which readings apply.** VALID: the oracle bound 2/2, at 6000 and 3600.
- **"All-position targeting keeps it" applies**: BASE only 1, ALLMASS only 0.
- **"Aims" applies**: 20 of ALLMASS's 21 firings were labelled ok on its own pair (95%).
- **No cost by the rule.**
- **"The evaluation is the better probe" applies**: EVAL vs BASE is 0 vs 0. Pooled over S84 and S86 (20 pairs,
  descriptive) EVAL is 19/20 against BASE's 18/20: EVAL only 1, BASE only 0, one-sided p = 0.5.

**What the firings show**
- **The one discordant pair is the one all-position mistarget.**
  - On seed 2334 the first eligible check (19200, held-out accuracy 0.80) had different pairs: the key-position pair
    7 -> 14 was labelled ok, the all-position pair 1 -> 6 was not.
  - ALLMASS copied 1 -> 6, fired twice more (both labelled ok) and ended MERGED (2 share).
  - BASE copied 7 -> 14 and bound routed at 28800 (with one later split that was not labelled ok, at 26400).
  - EVAL did the same as BASE.
  - So, as in the main line's records (97/107 against 107/107), the all-position pair is right most of the time and
    occasionally wrong. Here the one wrong choice cost the run.
- **The pairs mostly differ, and still both are usually labelled targets.** In every arm the key-position and
  all-position pairs were equal at only 8 of ~20 firings (main line: 46/107). Labels: ALLMASS's own pair 20/21; BASE's
  key pair 19/20 (all pair 17/20); EVAL's key pair 19/20 (all pair 16/20).
- **Seed 2331 merged under all three rules,** after three labelled-ok splits each. It is not a targeting failure.
- **The flat start appears again.** Each arm fired 5-6 of its ~20 splits at accuracy 0.03-0.19. Those splits cost nothing
  on these seeds, as in S84.

**Reading for the main line (exploratory).** A target chosen by the all-position read-gate mass, which reads no task
structure, kept 8 of BASE's 9 bound runs here; across 10 seeds that is within one pair. The main line's records predict
about 1 mistarget in 10 firings. Here there was 1 in 21, and it decided one run. The key-position rule's advantage is one
mistarget fewer, at the price of reading the key positions.

**Classifier.** v1 and v2 agree on every S86 run.

**Report code.** The first report pass crashed after all runs were saved: a dict merge with integer seed keys in the
descriptive pooling. That was fixed in the parent's report code only; no run's code SHA changed.
