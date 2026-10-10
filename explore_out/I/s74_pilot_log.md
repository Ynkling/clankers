# S74 pilot log (Session I) — EXPLORATORY, not a result

Every change to the task's constants, with its reason, in order. Model never changed (BDH N=256, D=32, 3 layers,
decay 0.95, conv "layer" width 4). CPU: Intel(R) Xeon(R) Processor @ 2.10GHz; torch 2.14.0+cu130; 1 thread per run.

## 2026-10-10 — before any training run

**P0. The starting point is infeasible.** L = 192, NB = 3, LMIN..LMAX = 6..14, V = 16, B = 32. At a fixed length every
source has M = (L − S·NB)/S words in NB blocks of 6..14 words, so 18 ≤ M ≤ 42: at S = 2 a sequence holds at most
2·(3 + 42) = 90 tokens (S = 4: 180). L = 192 cannot be filled. Not run.

**P1. The Bayes bound on SET accuracy.** Tables are uniform and independent across words and sources, so given the
successors O of word w already observed in its source, every 3-set containing O is equally likely: no predictor's
expected SET accuracy at that position exceeds 1/C(V − |O|, 3 − |O|), and averaging over positions gives an exact upper
bound (explore_i_task.ideal_set_acc, "own"). The same computation with the counts pooled over sources ("mixed") is what
a memory that mixes sources can know; "_fade" variants weigh an observation d tokens back by 0.95^d (the model's decay)
and count it only above 0.05 (one observation ≈ 58 tokens). Monte Carlo over 32-48 sequences (± ~0.01).

| S | V | NB | words/block | M | L | own (Bayes bound) | mixed | own_fade | mixed_fade |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 16 | 8 | 6..14 | 88 | 192 | 0.355 | 0.170 | 0.181 | 0.126 |
| 2 | 8 | 8 | 6..14 | 88 | 192 | 0.603 | 0.295 | 0.426 | 0.278 |
| 2 | 6 | 8 | 6..14 | 88 | 192 | 0.719 | 0.390 | 0.589 | 0.398 |
| 2 | 5 | 8 | 6..14 | 88 | 192 | 0.784 | 0.470 | 0.680 | 0.509 |
| 2 | 5 | 13 | 6..14 | 131 | 288 | 0.846 | 0.501 | 0.709 | 0.499 |
| 2 | 4 | 4 | 6..14 | 44 | 96 | 0.720 | 0.538 | 0.703 | 0.574 |
| 2 | 4 | 8 | 6..14 | 88 | 192 | 0.855 | 0.593 | 0.800 | 0.626 |
| 2 | 4 | 13 | 6..14 | 131 | 288 | 0.902 | 0.622 | 0.829 | 0.633 |
| 2 | 4 | 16 | 6..14 | 176 | 384 | 0.932 | 0.650 | 0.851 | 0.667 |
| 2 | 4 | 20 | 6..14 | 220 | 480 | 0.942 | 0.658 | 0.850 | 0.671 |
| 2 | 4 | 9 | 10..20 | 135 | 288 | 0.905 | 0.633 | 0.830 | 0.679 |
| 2 | 4 | 6 | 16..32 | 138 | 288 | 0.908 | 0.626 | 0.841 | 0.718 |
| 2 | 4 | 4 | 24..48 | 140 | 288 | 0.910 | 0.652 | 0.841 | 0.782 |
| 4 | 16 | 4 | 6..14 | 44 | 192 | 0.143 | 0.037 | 0.067 | 0.045 |
| 4 | 5 | 7 | 6..14 | 65 | 288 | 0.697 | 0.292 | 0.468 | 0.304 |
| 4 | 4 | 4 | 6..14 | 44 | 192 | 0.721 | 0.440 | 0.606 | 0.473 |
| 4 | 4 | 7 | 6..14 | 65 | 288 | 0.803 | 0.461 | 0.629 | 0.478 |
| 4 | 4 | 3 | 16..32 | 69 | 288 | 0.820 | 0.476 | 0.696 | 0.600 |
| 1 | 4 | 16 | 6..14 | 176 | 192 | 0.926 | 0.926 | 0.919 | 0.919 |
| 1 | 4 | 32 | 6..14 | 352 | 384 | 0.962 | 0.962 | 0.952 | 0.952 |

Reading of the table (design, not a result): V ≥ 5 is excluded at every L ≤ 288 (bound < 0.9: no model can be usable);
V = 4 (3 successors of 4: the set is "all words but one") is the only vocabulary with a bound above 0.9 within
L ≤ 480 at S = 2, from L ≈ 288 (0.90) and with a margin from L = 384 (0.93). Longer blocks raise the faded ceiling a
little and shrink the faded gap (the recent memory is then mostly the current source). At S = 4 the bound at a given L
is S = 2's at half the L: ≥ 0.93 needs L ≈ 768.

**Timing** (one process, 1 thread, LOCAL3 k = 2 model, B = 32): L = 96 0.064 s/update; 192 0.168; 288 0.34; 384 0.54;
576 1.38. A 12000-update run at L = 768 would take ~10 h before evaluation and pool contention: S = 4 cannot be
piloted within the day at any configuration whose bound allows usability, so **S = 4 is not run** (decided by the bound).

**Smoke test** (not a pilot run; seed 800, 3600 updates, S = 2, V = 4, NB = 4, M = 44, L = 96): ORACLE SET 0.693
(bound 0.720, faded 0.703), SINGLE 0.539 (mixed 0.538): the models sit on the ideal learners. The source probe read
0.49-0.51 at every layer in both (the stack did not carry the source in 3600 updates).

**C1** (first pilot configuration): S = 2, V = 4, LMIN..LMAX = 6..14, NB = 16, M = 176, L = 384, B = 32; NOMIX S = 1,
NB = 32, M = 352. Reason: the shortest L with a margin of the bound above 0.9 (0.93); the faded learner predicts ORACLE
≈ 0.85, so C1 tests whether the model's memory reaches past the faded horizon. Usable only if it does.
