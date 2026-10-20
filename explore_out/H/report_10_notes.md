**Which reading applies.** VALID: the perfect gate bound 2/2 (at 16800 and 10800; it ran first and gated the learned
arms). **"The labelled recipe itself fails here" applies:** BASE bound routed 1/10, under the 7/10 the rule required, so
no other reading applies. The trigger readings and "the split is not needed here" are not read. The comparisons above
are printed only: ALLMASS and REPEATMASS 3/10, NOSPLIT 0/10; ALLMASS vs NOSPLIT 3 vs 0, p = 0.125.

**CHECKs.** All passed before any run (20 min):
- At pool size 4 the task equals the main line's batches bit for bit.
- BASE|2600 and ALLMASS|2600 reproduce S87a's records through 9600, including the firing at 4800.
- NOSPLIT|260 reproduces batch 16's record through 6000, and ORACLE|2600 reproduces S87a's through 2400.
- The pool-32 trigger arms at threshold 0 equal NOSPLIT through 4800.
- The exact copy is S61's copy_nonoise_op, both as syntax trees and bit for bit.
- The repeat statistic reads token ids only.
- Printed, not asserted, because the record is from a 2.80 GHz Xeon: at pool size 4 with the main line's split op, X's
  WIN3_SPLIT_D8|540 reproduced through 4800 in every compared field (curve, statistics, gradient norms, the trigger rows
  at 2400 and 4800).

**Runtime.** 42 runs, 36.6 CPU-hours on a Xeon @ 2.10 GHz, on 4 workers.
- The container restarted once mid-batch (after 26 saved runs). The 4 runs in flight were lost and rerun from scratch on
  the same CPU model with the same code SHA. No saved run changed.
- No run failed.

**What happened (descriptive).**
- **The pool task is slower for every arm, the perfect gate included.** The perfect gate bound at 10800 and 16800,
  against 2400 and 8400 on the main line's task (S87a's oracle seeds 2600-2601; medians 13800 against 5400).
  - The four runs that bound routed (BASE 2910; ALLMASS and REPEATMASS 2904, 2908, 2909) bound at 36000-40800.
  - On the main line's task the same recipe bound at 9600-30000, median 18000 (S87a BASE).
  - Three of the four bound within 2400-6000 updates of the 43200 budget.
- **The trigger spent its cap before the merges formed.**
  - Every triggered run fired at 4800, in the flat start; 17 of each arm's 30 firings came at probe accuracy < 0.2.
  - Every run had used all 3 splits by 14400-36000.
  - The rule then asked to fire at 29-31 later checks per arm, each blocked by the cap.
  - The runs ended merged: 2 streams sharing in most, held-out accuracy 0.45-0.85.
  - This is the failure mode S79 found with a cap of one (splits spent in the flat start are not there for the late
    merge), here at the cap of three.
- **The aim was not the problem.** At its firings each arm's own pair was a labelled target 27-28 times of 30 (c* holding
  >= 2 streams, c0 none). The key-position pair was a labelled target at 27 of 30 in every arm. The labels come from the
  generator's stream tokens.
- **The split helps, but not enough to bind.**
  - NOSPLIT bound 0/10 and ended lower: held-out accuracy at 24000 was 0.15-0.71, against 0.14-0.89 in the triggered
    arms; at the end it was 0.19-0.85.
  - Five NOSPLIT runs ended with 3 streams on one channel, against 1-2 in each triggered arm.
- **REPEATMASS is ALLMASS on this layout, as the docstring said before any run.**
  - The repeat set is 96% of input positions: every CTX and key position, 88% of the values, the query's CTX and key.
    The key positions are 34% of it.
  - The repeat pair equalled the all-position pair at 28 of 30 firings. On 8 of the 10 seeds the two arms' runs are bit
    for bit identical.
  - On 2902 and 2903 they diverged at the third firing (21600: a different c*), and both arms merged on both seeds.
  - So this screen says nothing about a repeat statistic that isolates the keys. On the grouped layout the stream tokens
    and most values repeat too.
- **The key pool did what it was built to do.**
  - Key types fell to frequency 0.010 each, below the values (0.020) and stream tokens (0.042); at pool size 4 they are
    0.084.
  - Within a sequence nothing changed: each key still fills one group of 8 adjacent triples at the positions 3j+1.
  - So the slowdown comes from the key identities varying across sequences (32 keys to bind across sequences instead of
    4), not from the layout.

**Reading for the round (exploratory).** On this task the screen cannot say whether the all-position trigger survives:
the labelled recipe it is measured against does not bind at the main line's budget and cap. The perfect gate's later
binding (about 2.5x by the medians) and the late transitions point to the budget and the cap, not the trigger's target, as what fails first.
Neither was varied here. Options, none screened:
- a longer budget for every arm;
- S84's guard (no split while the probe reads < 0.5, which keeps the cap for the merges);
- a larger cap (S79's cap of six bound 9/10 on the main line's task).

The two-stream spec draft's revision was to wait until this screen is read; it has been read, and the revision is not
started (it is the next item if wanted).
