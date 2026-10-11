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
