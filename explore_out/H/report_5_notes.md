**Which readings apply.** VALID: the oracle bound 2/2, both at 3600.
- **CAP1 is "sensitive"**: BASE only 4, CAP1 only 0 (one-sided p = 0.0625).
- **INT1200 is "inconclusive"**: BASE only 2, INT1200 only 0.
- **CAP6, PROBE16 and THR05 are "robust"**: 0 vs 0, 0 vs 1, 0 vs 1.
- **"The extra splits are harmless" does not apply.** CAP6 matches BASE (9 vs 9), but 1 of its 7 extra firings was not
  labelled ok (seed 1321 at 28800, a run that bound anyway at 30000). The rule required every extra firing to be ok.

**What matters, and why** (from the firing records)
- **The cap matters.** A split spent at the start is wasted:
  - With one split allowed, 4 of CAP1's 5 failures fired it at 4800-12000, while held-out accuracy was still 0.09-0.50.
    That is the flat start before any stream has bound, not a merge plateau.
  - BASE, on the same seeds, still had two splits left for the real merge later.
  - Every CAP1 split was aimed at a crowded channel (9/9 labelled ok); it was simply spent too early.
  - The fifth failure (1320) also failed under BASE, with all three splits.
- **The check interval works through the same mechanism.**
  - INT1200's first split came earlier than BASE's on 10/10 seeds, mostly at 2400-3600, at chance accuracy (0.03-0.09).
  - Its three failures (1320, 1321, 1327) had used all three splits by 7200-10800, at accuracy <= 0.47, and then merged.
  - When INT1200 succeeds it binds earlier (median transition 14400 vs 24000).
  - So the interval trades speed for the risk of exhausting the cap during the flat start.
- **The other constants do not matter here.** A 16-sequence probe (18 firings, all labelled ok) and rise < 0.05 (24, all ok)
  bound 10/10. Six splits instead of three cost nothing.
- **What a replication must state:** the cap, the first eligible check and the check interval, which together fix how many
  splits remain after the flat start. Probe size and the rise threshold can be stated loosely.

**Classifier.** v1 and v2 agree on every S79 run (all failures are MERGED (2-3 share)). No eight-stream failure here had
eta^2 by key >= 0.5 with the margin rule firing.

**Seeds.** S79 ran on 1320-1329, the specification's 1120-1129 shifted with S78's block (see report 4).
