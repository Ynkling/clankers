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

**Referee V's review.** V reviewed the screen at d4a9a7d (claude/review-V 3c772c3 / dcd549d,
docs/review_2026-10/screen_review_s89_H.md: "fix first", text and the report path only) before the learned arms ran. I
read it after they had finished. Fixes 1-6, and the text of 7 and 8, are now in explore_h_s89.py's docstring and
report():
- (1) the screen restated as rare key types on a layout with fixed key positions, with the caveat printed with every
  reading;
- (2) REPEATMASS printed beside ALLMASS, not as a second finding;
- (3) a moot reading carries no name;
- (4) the rule's operating characteristics;
- (5) one-to-one-before and op_ok with each trigger arm;
- (6) the main-line record read at 628eb7c.

That file is not in the child's code SHA, so no run, CHECK or record changed. The reading is the same under the fixed
text, and fix 3's case does not arise.

**What happened (descriptive).**
- **The nine BASE failures are slow climbs cut off by the budget, not stalls.**
  - The perfect gate on the pool task climbs from the start: 0.21-0.41 at 1200, 0.71-0.77 at 4800, 0.90-0.91 by
    8400-9600. Its formal binds at 10800 and 16800 come after a stretch at 0.90-0.96.
  - The learned BASE runs stay near chance far longer: held-out accuracy first reaches 0.3 at 12000-22800 and 0.5 at
    14400-26400.
  - The nine failures end at 0.60-0.85 with 2-6 streams below 0.9, two streams sharing a channel in seven and three
    in two.
  - In seven of the nine the last evaluation (43200) is the curve's maximum: they were still rising when the budget
    ended.
  - The one BASE bind (2910) reached 0.8 at 21600 and bound at 36000.
- **ALLMASS on the same seeds (all 10 runs saved).**
  - 3/10 bound routed (2904, 2908, 2909, at 37200-40800). BASE bound none of these three.
  - ALLMASS failed on BASE's bind 2910: held-out accuracy 0.45 at the end, every stream below 0.9.
  - Its seven failures end at 0.45-0.84, the same shape as BASE's.
  - Both arms first fire at 4800 on every seed.
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
  - On our reading, the slowdown comes from the key identities varying across sequences (32 key types to learn instead
    of 4), not from the layout.

**Reading for the round (exploratory).** On this task the screen cannot say whether the all-position trigger survives:
the labelled recipe it is measured against does not bind at the main line's budget and cap. The perfect gate's later
binding (about 2.5x by the medians) and the late transitions point to the budget and the cap, not the trigger's target, as what fails first.
Neither was varied here. S89c, next, varies two of them on the Hebbian recipe: the first eligible check at 12000 instead
of 4800 (BASE_LATE) and a cap of 6 instead of 3 (BASE_CAP6), against a BASE rerun on seeds 2924-2933. S89b (delta
memory, seeds 2912-2923) follows. The two-stream spec draft's revision waits until this screen is read, which is now
done, but it is not started: the user's order puts S89c and S89b first.
