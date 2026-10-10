# Session I — report 1 (S74 pilot: the ICMC task). EXPLORATORY, not a result

Branch `claude/explore-I`. Intel Xeon @ 2.10GHz, torch 2.14.0+cu130, 1 thread per run, 4 workers. Seeds 800-803.
Task `explore_i_task.ICMCTask`: S sources, V words, per sequence a fresh table of 3 successors per word and source,
NB blocks per source of LMIN..LMAX words, the blocks in random order, the loss at every word with a same-source
predecessor. Model: BDH, N = 256, D = 32, 3 layers, decay 0.95, conv "layer" width 4 (unchanged). Arms: ORACLE (perfect
gate, k = S), SINGLE (k = 1), NOMIX (one source at the same L), Adam 1e-3, 12000 updates, B = 32. Usable: ORACLE's
final SET accuracy ≥ 0.9 and SINGLE's ≤ ORACLE's − 0.2 on ≥ 3/4 seeds. Full log: `s74_pilot_log.md`.

## Counts

| config | S, V, words/block, NB, L | Bayes bound | ORACLE ≥ 0.9 | SINGLE ≤ ORACLE − 0.2 | usable | min/run |
|---|---|---|---|---|---|---|
| P0 | 2, 16, 6..14, 3, 192 | — | — | — | infeasible (≤ 90 tokens fit) | — |
| C1 | 2, 4, 6..14, 16, 384 | 0.932 | 4/4 (0.910-0.918) | 2/4 (0.661-0.746) | **no** (2/4) | 106-111 |
| C2 | 2, 4, 3..7, 40, 480 | 0.936 | 4/4 (0.916-0.918) | 1/4 (0.654-0.863) | **no** (1/4) | 155-166 |

NOMIX (descriptive): C1 stopped after the decision; C2 __NOMIX__.
S = 4: not run. The exact bound (uniform tables: given the observed successors O, every completing set is equally
likely) is 0.82 at L = 288 and needs L ≈ 768 for 0.93, ~10 h per 12000-update run here: excluded by the bound.

## What the pilot found

1. **The 0.9 bar forces V = 4 and long sequences.** SET accuracy needs the whole successor set, so positions where a
   word's three successors have not all been seen in its source cannot be right. The bound is < 0.9 for every V ≥ 5 at
   L ≤ 288 and for every S = 4 configuration below L ≈ 700 (table in the log). The models sit close to it: ORACLE ran
   0.015-0.02 under the bound at L = 384 and 480; on a short smoke test both arms matched the ideal learners within 0.01.
2. **Under the dense loss, one channel learns to carry the source.** In C2, three SINGLE runs made a step transition
   (seed 803 at update 2400, 800 at 9600, 801 at 10800): at one evaluation SET rose by 0.10-0.21 and a logistic probe
   from the residual entering layer 3 to the source went from 0.52-0.58 to **0.98** (layer 2: 0.87-0.91). The gap to
   ORACLE then closed by 60-80%. Seed 802 stayed at the mixed learner's level (0.65, probe 0.51). In C1 (longer
   blocks) no probe moved (≤ 0.54) and two seeds still drifted to 0.74 (the recent memory is mostly the current source).
   ORACLE's stack never carried the source (probe ≤ 0.53): with the partition given, nothing asks it to.
3. **So the screen's control is not a control at S = 2.** More interference (short blocks) drives the single channel to
   tag its memory by source; less interference (long blocks) leaves it close by recency. The remaining knob that would
   keep SINGLE down at 12000 updates is a smaller batch, which would only push the transition past the pilot and into
   S75's 24000-update budget. Not tried. **Nothing usable; stopped** after ~9.5 h of the pilot's day. S75 not run.

## Relevance to the session's question (descriptive, not a reading)

The sparse-loss stacks of Section 10 and Session G did not carry the block's context token (probes ≤ 0.6). Here, with a
loss at every word, a single channel's stack carries the source at layers 2-3 in 3/4 runs of C2 within 12000 updates,
with no gate and no labels. The dense loss does put pressure on the stack to carry the cue. Whether a gate reading that
stack (REG3) or a wide window would then partition the memory is S75's question and remains open.

## CHECKs (asserted before each configuration; all ok)

The task (chains, mask, compositions against the exact DP marginals, block order, stream labels, SET = 1.0 on the true
sets); the reproduction: the new loop with the grouped task, k = 1 and the query loss equals X's recorded B_conv|160 bit
for bit through 2400 (curve and gradient norms); LOCAL3 sees t−2 and not t−3, WIDE sees t−(LMAX+1) and not t−(LMAX+2);
REG3's convolution loop equals Session G's no-conv loop at the identity conv, its parameters equal `to_reg`'s, strict
causality, the register's one-step delay; the perfect gate one-hot; the split's mechanics (fires 4800/9600/14400, gap-
and cap-blocked as specified, `split_op` equal to test_window_gate's; at threshold 0 identical to no split through 7200);
the probe's synthetic check (1.00 informative, 0.49 random); the SLOW groups.

## Options for the user (none taken)

(a) Accept a weaker control at S = 2: e.g. "usable" as the pilot stands for ORACLE, and read S75's GAP against SINGLE
as it is (a single channel that tags by source is the competitor the partition must beat). (b) Use NOMIX or the
mixed ideal learner as the interference floor instead of SINGLE. (c) Make S = 4 affordable (a GPU, or fewer seeds at
L ≈ 768), where four sources may be harder for one channel to tag. (d) A SET criterion relative to the Bayes bound would
admit V ≥ 5 and shorter sequences.
