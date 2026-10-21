# Session H — report 11 (EXPLORATORY, not a result)

## S89c explore_h_s89c — the trigger's schedule on the 32-key pool task (S=8, k=16, conv, 43200, Hebbian, exact copy); seeds 2924-2933 paired

| arm | BOUND ROUTED | band | Wilson 95% | failures v1 | failures v2 | transitions | splits | own pair ok | at probe < 0.2 | map 1:1 before | cap-blocked checks | last firing per run | held-out accuracy at the end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BASE | 2/10 | MINORITY | 0.06-0.51 | MERGED 8 | MERGED (2 share) 8 | 32400, 43200 | 30 | 25/30 | 16 | 2 | 24 | 14400, 21600, 24000, 24000, 28800, 28800, 31200, 33600, 33600, 36000 | 0.58, 0.64, 0.71, 0.71, 0.71, 0.75, 0.87, 0.87, 0.95, 0.96 |
| BASE_LATE | 1/10 | MINORITY | 0.02-0.40 | MERGED 9 | MERGED (2 share) 7, MERGED (3 share) 1, MERGED (4 share) 1 | 40800 | 29 | 29/29 | 6 | 0 | 15 | 21600, 28800, 28800, 31200, 31200, 31200, 33600, 38400, 40800, 40800 | 0.57, 0.58, 0.61, 0.69, 0.72, 0.73, 0.73, 0.75, 0.82, 0.96 |
| BASE_CAP6 | 2/10 | MINORITY | 0.06-0.51 | MERGED 6, STREAM-PARTIAL 2 | MERGED (2 share) 6, STREAM-PARTIAL one-to-one 2 | 32400, 43200 | 46 | 39/46 | 16 | 4 | 1 | 28800, 31200, 36000, 38400, 38400, 40800, 40800, 40800, 40800, 40800 | 0.59, 0.66, 0.69, 0.7, 0.71, 0.76, 0.77, 0.89, 0.95, 0.96 |

Validity: S89's perfect gate on the same task and code path, 2/2 (at 16800 and 10800).

- BASE_LATE vs BASE: 1/10 vs 2/10; BASE_LATE only 0, BASE only 1; p(BASE_LATE>BASE) 1, p(BASE>BASE_LATE) 0.5
- BASE_CAP6 vs BASE: 2/10 vs 2/10; BASE_CAP6 only 0, BASE only 0; p(BASE_CAP6>BASE) 1, p(BASE>BASE_CAP6) 1
- BASE_LATE vs BASE_CAP6: 1/10 vs 2/10; BASE_LATE only 0, BASE_CAP6 only 1; p(BASE_LATE>BASE_CAP6) 1, p(BASE_CAP6>BASE_LATE) 0.5

**Reading (fixed in explore_h_s89c's docstring before any run):**

- S89c: **the channel is the limit (BASE_LATE 1/10 (BASE only 1, BASE_LATE only 0, p(BASE_LATE>BASE) 1); BASE_CAP6 2/10 (BASE only 0, BASE_CAP6 only 0, p(BASE_CAP6>BASE) 1); BASE 2/10)**

*Caveat, with every reading:* the pool makes key types rare across the corpus (0.0105 per type against 0.084 at pool size 4); it does not move the keys: they stay at 3j+1, are the most repeated types within a sequence (8 each; CTX 4, VAL 1-4), and each sits next to its stream's CTX token, which the window gate reads. BASE's key-position mask is the main line's, unchanged. A reading here is about rare key types on a layout with positional keys, not about unmarked key positions.

*The rule's operating characteristics (explore_h_oc.py, exact, BASE 0.1, rho 0 / 0.5 / 1): both new arms at 0.3: schedule 0.27/0.23/0.12, channel 0.42/0.50/0.65; at 0.5: schedule 0.64/0.69/0.62, channel 0.03/0.07/0.17; at 0.7: schedule 0.87/0.93/0.95; one arm at 0.5, the other at 0.1: schedule 0.47/0.55/0.62, channel 0.17. 'The channel is the limit' reads a gain below about +0.2 (and half the time at +0.2); 'the schedule was the limit' needs about +0.4 to be likely; 4 vs 0 is one-sided p = 0.0625.*

## S89b explore_h_s89b — the split on delta memory (beta 1, L2 keys) on the 32-key pool task (S=8, k=16, conv, 43200); the delta perfect gate on 2912-2913, learned arms on 2914-2923

| run | outcome | transition | held-out accuracy at 1200 / 12000 / 24000 / end | streams at the end | runtime |
|---|---|---|---|---|---|
| ORACLE_DELTA 2912 | STREAM-PARTIAL one-to-one | None | 0.31 / 0.49 / 0.59 / 0.875 | 0.85-0.91, map [0, 1, 2, 3, 4, 5, 6, 7] | 161 min |
| ORACLE_DELTA 2913 | BOUND ROUTED | 24000 | 0.29 / 0.88 / 0.96 / 0.964 | 0.95-0.97, map [0, 1, 2, 3, 4, 5, 6, 7] | 99 min |

Validity: ORACLE_DELTA bound 1/2 ({2912: None, 2913: 24000}) → UNTESTED. Learned-arm runs: 0 (queued only after the perfect gate binds both seeds).

**Reading (fixed in explore_h_s89b's docstring before any run):**

- S89b: **UNTESTED (ORACLE_DELTA bound 1/2); no learned arm runs**

*Caveat, with every reading:* the pool makes key types rare across the corpus (0.0105 per type against 0.084 at pool size 4); it does not move the keys: they stay at 3j+1, are the most repeated types within a sequence (8 each; CTX 4, VAL 1-4), and each sits next to its stream's CTX token, which the window gate reads. BASE's key-position mask is the main line's, unchanged. A reading here is about rare key types on a layout with positional keys, not about unmarked key positions.

**H/s89c:** 30 runs, 25.8 CPU-hours on Intel(R) Xeon(R) Processor @ 2.10GHz; code SHA 4dda56358910; CHECKs all passed (2026-10-20 21:59:18 UTC, git 555d681+dirty, 15.8 min)

**H/s89b:** 2 runs, 4.3 CPU-hours on Intel(R) Xeon(R) Processor @ 2.10GHz; code SHA 9cb518b59c14; CHECKs all passed (2026-10-21 06:42:32 UTC, git 568a12f+dirty, 38.9 min)

**Which readings apply.**
- **S89c: "the channel is the limit."** Neither schedule change reached 4/10: BASE_LATE 1/10, BASE_CAP6 2/10, against the
  BASE rerun's 2/10. No pair favours a new arm (BASE_LATE only 0, BASE_CAP6 only 0). By the operating characteristics
  fixed before the run, this reads a schedule effect below about +0.2 in the binding rate. It does not show that the
  schedule cannot matter. Validity is S89's perfect gate on the same task and code path (2/2).
- **S89b: UNTESTED.** The delta perfect gate bound 1 of its 2 seeds, so by the rule fixed before any run no learned arm
  ran. Nothing about the trigger on delta memory is learned here.

**CHECKs.**
- S89c (15.8 min): the trigger's mechanics on S48's harness; BASE and BASE_CAP6 reproduce S89's BASE|2902 bit for bit
  through 12000; BASE_LATE reproduces S89's NOSPLIT|2902 through 12000.
- S89b (38.9 min): the delta modules served at a581027 (F's 9a6c2bf byte for byte); F's fast_checks; the conversion adds
  no parameter. At pool size 4, three runs reproduce X's test_delta_recipe records bit for bit on this CPU (a Xeon @
  2.10GHz): DELTA_SPLIT|583 through 7200, across its firing at 4800; DELTA_SLOW|583 through 4800; CEIL_DELTA|580 through
  2400. At pool size 32 the trigger arms at threshold 0 equal DELTA_NOSPLIT through 4800.

**Runtime and incidents.**
- S89c: 30 runs, 25.8 CPU-hours. S89b: 2 runs, 4.3 CPU-hours. No run failed.
- The container restarted twice during S89c. The runs in flight were lost and rerun on the same CPU model with the same
  code SHA; no saved run changed.
- **A logging defect, now fixed.** The push after each saved run used `git pull --rebase` with autostash, which rewrites
  the working tree. The running driver's log was rewritten under it, and the driver's later writes went to an unlinked
  copy. S89c's log lost the lines of its last seven runs and its final summary. The store holds every run, and the
  summary was regenerated into the log with --report-only, with a note. The driver no longer pulls (only this session
  pushes to this branch). S89b's validity stage was restarted under the fixed driver before it had run anything.
- **A gate-code defect in S89b, fixed after the validity stage.** Its lookup of S89c's result read S89c as incomplete:
  the harness starts STATE["batch_cpu"] at None, and setdefault left it there. Had the perfect gate passed, the learned
  arms would have stayed unqueued; they would not have run on a wrong schedule. The perfect gate's 1/2 makes the screen
  UNTESTED either way. The fix and a dated note are in explore_h_s89b.py.

**What S89c shows (descriptive).**
- **A later first check does not reach past the flat start.** At 12000 the probe still read 0.09-0.38 in every
  BASE_LATE run, and 6 of its 29 firings came at probe < 0.2. Without the early splits its held-out accuracy reached 0.3
  later: 12000-22800, against the BASE rerun's 12000-18000.
- **More splits do not convert the merges.**
  - BASE_CAP6 is BASE until a run's fourth firing. On the two seeds BASE bound (2924, 2933) it never fired a fourth time,
    and those runs are identical.
  - On the other eight the extra firings came at 24000-40800 (46 in all, 39 labelled ok). They ended 6 MERGED and 2
    STREAM-PARTIAL one-to-one, at held-out accuracy 0.59-0.89.
  - Only one check in ten runs was blocked by the cap of 6.
- **The failures are slow climbs, as in S89.** In about half the failures per arm (4 of 8, 5 of 9, 4 of 8) the last
  evaluation is the curve's maximum. The BASE rerun's two binds came at 32400 and 43200.

**What S89b's perfect gate shows (descriptive).** The delta perfect gate, which routes each stream to its own channel by
construction, is slow on the pool task.
- Seed 2913 reached 0.88 at 12000 and bound at 24000.
- Seed 2912 sat at 0.30 until about 9600, then climbed to 0.875 by 43200, still rising. Every stream was at 0.85-0.91
  and the map was one-to-one.
- On the main line's 4-key task the same gate binds at 1200-2400 (X's CEIL_DELTA 2400 and 1200).
- So, on our reading, what is slow on delta memory at 32 key types is recall within a stream's own channel, not
  routing.

**What follows (the user's plan; nothing here is run).** "The schedule was the limit" did not apply. By the plan, the
all-position trigger question does not reopen on this task with a new schedule. S89b's delta arms are untested, because
the delta perfect gate does not reliably bind the pool task within 43200. Both perfect gates bind late (Hebbian 10800 and
16800; delta 24000 and not by 43200), and about half the learned failures are still rising at the end. On our reading,
the budget and the size of the key vocabulary, not the trigger, are what this task tests first. Candidates, none
screened:
- a longer budget for every arm, the perfect gates included;
- a smaller pool (8 or 16 key types), between the main line's 4 and 32.

The two-stream spec draft's revision per the referee's review is still waiting.
