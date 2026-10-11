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
