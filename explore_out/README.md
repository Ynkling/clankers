# explore_out (branch claude/explore-I)

This branch holds only Session I's outputs; the other sessions' sections are on claude/outside-ideas and
claude/explore-F/G/H.

## Session I

EXPLORATORY, not a result. Subject: does the channel partition emerge under a dense next-token loss on interleaved
sources (ICMC: in-context Markov chains with sources)? Code: explore_i_task.py, explore_i_common.py, explore_i_checks.py,
explore_i_pilot.py (S74), explore_i_dense.py (S75, a draft with placeholders, not run). Seeds used: 800-803.

- **S74 pilot (report_1.md, s74_pilot_log.md): nothing usable; stopped.** The usability bar (ORACLE SET ≥ 0.9) forces
  V = 4 and L ≥ 384 at S = 2 (exact Bayes bound) and puts S = 4 out of reach (L ≈ 768). C1 (blocks 6..14, L = 384):
  ORACLE 4/4 ≥ 0.9, SINGLE ≤ ORACLE − 0.2 on 2/4. C2 (blocks 3..7, L = 480): ORACLE 4/4, SINGLE on 1/4, because three
  single-channel runs learned to carry the source in the stack (probe from the residual entering layer 3: 0.52-0.58 →
  0.98 at one evaluation, SET +0.10-0.21). Under a dense loss the stack does carry the cue, where the sparse-loss stacks
  did not.
- **S75: not run** (no usable configuration). Options for the user are listed at the end of report_1.md.
