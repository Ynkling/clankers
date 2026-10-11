# Session H — report 6 (EXPLORATORY, not a result)

## S84 explore_h_guard — S=8, P=4, k=16, conv, 43200, LOCAL3 + SLOW + exact-copy split, Hebbian, seeds 2300-2309

| arm | BOUND ROUTED | band | Wilson 95% | failures v1 | failures v2 | transition median (all) | splits | labelled ok | accuracy read at the splits | guard-blocked checks |
|---|---|---|---|---|---|---|---|---|---|---|
| BASE | 9/10 | RELIABLE | 0.60-0.98 | MERGED 1 | MERGED (3 share) 1 | 16800 (15600, 15600, 15600, 16800, 16800, 19200, 20400, 25200, 27600) | 24 | 24/24 | 0.06, 0.06, 0.08, 0.09, 0.11, 0.11, 0.16, 0.42, 0.45, 0.47, 0.5, 0.61, 0.62, 0.69, 0.69, 0.72, 0.73, 0.77, 0.78, 0.8, 0.83, 0.86, 0.88, 0.88 | 0 |
| GUARD | 9/10 | RELIABLE | 0.60-0.98 | MERGED 1 | MERGED (3 share) 1 | 18000 (15600, 15600, 15600, 15600, 18000, 20400, 27600, 30000, 30000) | 17 | 17/17 | 0.5, 0.62, 0.62, 0.73, 0.77, 0.77, 0.8, 0.8, 0.8, 0.81, 0.83, 0.83, 0.84, 0.86, 0.88, 0.88, 0.91 | 15 |
| GUARD_INT1200 | 9/10 | RELIABLE | 0.60-0.98 | MERGED 1 | MERGED (3 share) 1 | 15600 (10800, 12000, 13200, 15600, 15600, 16800, 16800, 21600, 27600) | 18 | 18/18 | 0.5, 0.52, 0.59, 0.59, 0.67, 0.67, 0.75, 0.75, 0.8, 0.8, 0.81, 0.83, 0.83, 0.84, 0.86, 0.89, 0.91, 0.94 | 50 |
| EVAL | 10/10 | RELIABLE | 0.72-1.00 | — | — | 18000 (15600, 15600, 15600, 15600, 15600, 20400, 22800, 25200, 25200, 27600) | 19 | 19/19 | 0.07, 0.11, 0.45, 0.59, 0.6, 0.6, 0.74, 0.74, 0.74, 0.74, 0.74, 0.75, 0.75, 0.86, 0.86, 0.87, 0.87, 0.87, 0.88 | 0 |
| EVAL_GUARD_INT1200 | 9/10 | RELIABLE | 0.60-0.98 | MERGED 1 | MERGED (3 share) 1 | 15600 (12000, 13200, 13200, 15600, 15600, 15600, 16800, 24000, 25200) | 18 | 18/18 | 0.58, 0.61, 0.62, 0.71, 0.73, 0.74, 0.74, 0.75, 0.75, 0.75, 0.86, 0.86, 0.87, 0.87, 0.87, 0.87, 0.87, 0.87 | 56 |

ORACLE (ceil8_D8, seeds 2300-2301): bound 2/2, transitions {2300: 3600, 2301: 3600} → VALID.

- GUARD vs BASE: 9/10 vs 9/10; GUARD only 0, BASE only 0; p(GUARD>BASE) 1, p(BASE>GUARD) 1
- GUARD_INT1200 vs BASE: 9/10 vs 9/10; GUARD_INT1200 only 0, BASE only 0; p(GUARD_INT1200>BASE) 1, p(BASE>GUARD_INT1200) 1
- EVAL vs BASE: 10/10 vs 9/10; EVAL only 1, BASE only 0; p(EVAL>BASE) 0.5, p(BASE>EVAL) 1
- EVAL_GUARD_INT1200 vs BASE: 9/10 vs 9/10; EVAL_GUARD_INT1200 only 0, BASE only 0; p(EVAL_GUARD_INT1200>BASE) 1, p(BASE>EVAL_GUARD_INT1200) 1

**Readings (fixed in explore_h_guard's docstring before any run):**

- guard aims: **the guard aims the split (labelled ok 17/17; BASE only 0, GUARD only 0)**
- faster: **does not apply (BASE only 0, GUARD_INT1200 only 0; median transition 15600 vs BASE 16800)**
- eval: **the evaluation is the better probe (BASE only 0, EVAL only 1)**
- cost GUARD: **no cost by the rule (BASE only 0, GUARD only 0)**
- cost GUARD_INT1200: **no cost by the rule (BASE only 0, GUARD_INT1200 only 0)**
- cost EVAL: **no cost by the rule (BASE only 0, EVAL only 1)**
- cost EVAL_GUARD_INT1200: **no cost by the rule (BASE only 0, EVAL_GUARD_INT1200 only 0)**

First split per seed (update or none): BASE 2300:4800 2301:4800 2302:9600 2303:4800 2304:4800 2305:14400 2306:9600 2307:14400 2308:12000 2309:4800; GUARD 2300:14400 2301:16800 2302:9600 2303:14400 2304:14400 2305:14400 2306:None 2307:19200 2308:19200 2309:16800; GUARD_INT1200 2300:9600 2301:9600 2302:8400 2303:13200 2304:13200 2305:13200 2306:None 2307:19200 2308:9600 2309:9600; EVAL 2300:14400 2301:14400 2302:9600 2303:4800 2304:14400 2305:14400 2306:4800 2307:14400 2308:12000 2309:12000; EVAL_GUARD_INT1200 2300:9600 2301:12000 2302:8400 2303:12000 2304:13200 2305:13200 2306:None 2307:13200 2308:18000 2309:9600
v1/v2 outcome differences: none.

**Runtime:** 52 runs, 25.0 CPU-hours on Intel(R) Xeon(R) Processor @ 2.10GHz; code SHA 708d5adde607. **CHECKs:** all passed (2026-10-10 19:15:40 UTC, git 834581a, 9.3 min)

**Which readings apply.** VALID: the oracle bound 2/2, both at 3600.
- **"The guard aims the split" applies.** GUARD's 17 splits were all labelled ok, and GUARD vs BASE is 0 vs 0.
- **"The evaluation is the better probe" applies.** EVAL bound routed 10/10 against BASE's 9; BASE only 0, EVAL only 1.
- **"Faster without the cost" does not apply.** GUARD_INT1200 matches BASE (0 vs 0), but its median transition is 15600
  against BASE's 16800, only 1200 earlier where 2400 was required.
- **No arm "costs binding".**

**What happened, and the limits of these 10 seeds**
- **These seeds hardly tested the flat start.** BASE bound 9/10 here, although it fired 7 of its 24 splits at accuracy
  0.06-0.16 and 4 more at 0.42-0.50. In S79 such early splits cost CAP1 and INT1200 their runs. Here BASE kept enough
  splits for the real merge, so the guard could not show a gain in count. What it changed is when the splits fire: its
  17 splits all came at accuracy 0.50-0.91, and the guard blocked 15 checks. INT1200 went from S79's 7/10 (other seeds)
  to 9/10 with the guard, but that is across seeds, so counts only.
- **"Labelled ok" does not catch flat-start splits.** BASE's splits at chance accuracy were all labelled ok: early on, the
  crowded channel holds >= 2 streams and an empty one exists, so the label is trivially met. The guard's 95% criterion
  therefore cannot fail on aim, only on count. The accuracy read at each split (in the table) is the column that shows
  what the guard changed.
- **The one failure in every guarded arm is a run the guard locked out.** On seed 2306 the probe accuracy rose only
  slowly; at the end the held-out accuracy was 0.38 in all three guarded arms (0.50 for BASE). The guard blocked every
  split in GUARD, GUARD_INT1200 and EVAL_GUARD_INT1200: no split fired, and the run ended MERGED (3 share). BASE fired on
  2306 (at 9600, 24000 and 28800) and also ended MERGED (3 share), at accuracy 0.50. EVAL bound it (transition 25200)
  with a single split at 4800, at a held-out accuracy of 0.07.
- **Why EVAL caught seed 2306.** The 64-sequence probe read 0.000 then 0.047 at 2400 and 4800, a rise of 0.047, so BASE
  did not fire at 4800. The 2048-query evaluation read 0.066 then 0.068, a rise of 0.002, so EVAL fired at 4800. On this
  seed the probe's noise decided which check fired. That is one seed; it is the only discordant pair in the screen.
- **Faster checks with the guard bind sooner, but by less than the rule needs.** Median transitions: GUARD_INT1200 and
  EVAL_GUARD_INT1200 15600, BASE 16800, GUARD and EVAL 18000. The guard delays the first split (to 9600-19200, from
  BASE's 4800-14400), which offsets part of INT1200's speed.

**Classifier.** v1 and v2 agree on every S84 run (the failures are all MERGED (3 share)).

**For pairing with Session F's S81 (delta channels):** the arm names, rules and readings are as specified. The guard
reads the same accuracy as each arm's rule. EVAL's KEYMASS target is still chosen on the 64-sequence probe.
