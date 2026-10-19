#!/usr/bin/env python
"""
explore_g_s89.py — EXPLORATORY, not a result. Session G, screen S89 (round 6): does S87's frequency hinge do anything on a
task where keys are not a frequent class, and does a within-sequence repeat hinge replace the key hinge? Everything below
was fixed and committed before any S89 run.

BACKGROUND. S87 (RandHeaderTask-lite, seeds 2700-2719): REG3_FREQHINGE discovered 17/20 (17 vs 0 against BASELINE), two
pairs short of "replaces the key hinge" (REG3_KEYHINGE 20/20); on that task the largest-ratio cut selected the 2 CTX and
the 4 key types. Round 6 (the user's prompt of 19 October): F builds RandHeaderTask-pool (the 4 keys of each sequence
drawn from a pool of 32 key types) and RandHeaderTask-allpool (the 2 CTX tokens drawn from a pool of 16 too).

TASKS (vendored before any run): explore_f_tasks.py byte for byte from claude/explore-F at the commit F names (a CHECK
asserts the blob equals F's at that commit). A task runs only if F's pilots read "the task is valid" (24000 updates) or
"needs the longer budget" (then at 48000, every arm); F's commit, readings and the budget per task are written to
explore_out/G/s89_tasks.json before the first run and printed.
MODEL, RECIPE, HINGES, RECORDS: explore_g_s89_child (S87's model and recipe: Hebbian, k = 2, no conv, SLOW, REG3 at all
three layers; every hinge LAM 1.0 x mean over layers of relu(eta2 by token id at its positions - 0.2), updates 1-2400 only,
with a switch that makes it REG3 bit for bit when off).
  REG3_KEYHINGE    the key term with the true key positions and identities (still in the generator): the labelled ceiling.
  REG3_FREQHINGE   S87's frequency hinge exactly, the largest-ratio cut recomputed on the task's own sample (printed).
                   The largest-ratio rule always selects at least one type; if on a task it selects no position of a
                   sequence the term is 0 there. The expectation (the user's, to be confirmed by the printed table): the
                   two CTX types on -pool; on -allpool "nothing or the positional residue". G's arithmetic before the
                   tables exist (22 input tokens per sequence: 5 CTX, 9 KEY, 8 VAL): -pool CTX 0.114 per type, VAL 0.023,
                   KEY 0.013, so the largest ratio (about 5) cuts after the CTX types; -allpool CTX 0.028 per type (16
                   types), VAL 0.023, KEY 0.013, so the largest ratio (about 1.8) cuts after the VALUE types and the cut
                   selects the 16 CTX and 16 VAL types, not nothing. Which applies is printed from the sample.
  REG3_REPEATHINGE the positions of token types occurring more than once in the input sequence (within-sequence; no
                   table, no label). On the lite task it selects every CTX (5), every KEY (9) and 2.63 VAL positions by
                   chance per sequence (16.6; the key term's 8 are 48% of them); at the CTX positions grouping by token
                   id is grouping by stream. The counts per role on each new task are printed (a CHECK).
ARMS per valid task, seeds 2842-2861 (20, paired): BASELINE (LOCAL3 + SLOW), REG3, REG3_KEYHINGE, REG3_FREQHINGE,
  REG3_REPEATHINGE; the perfect gate (ORACLE_ALL, this branch's code path) on 2840-2841, run and required to bind 2/2
  BEFORE any learned arm of that task (else the task is UNTESTED here and its learned arms are not run).
DELTA (beta 1, L2 keys; seeds 2862-2871), per valid task: the least-reading arm discovered on Hebbian memory, in the order
  REG3_REPEATHINGE (a within-sequence statistic), REG3_FREQHINGE (a corpus frequency table); with BASELINE_D and REG3_D;
  descriptive, as in S87. None discovered on a task -> no delta runs on it. The choice is written to
  explore_out/G/s89_delta.json before the delta runs are queued.
READINGS (fixed now; ROUTED-BOUND = BOUND and F's ROUTED*@end; exact one-sided McNemar on seed pairs, b = the first arm
only; "discovered" = beats BASELINE with p < 0.05; "within 1 pair of REG3_KEYHINGE" = (REG3_KEYHINGE only) - (arm only)
<= 1; "beats by >= 4 vs 0" = one-sided p <= p(4 vs 0) = 0.0625, as in S82-S87), per task:
  first:  "the labelled hinge itself fails here"   if REG3_KEYHINGE is not discovered; then NO OTHER READING on that task,
                                                   and the report says what that means for the register spec.
  else, each that applies:
          "the frequency hinge survives"            REG3_FREQHINGE discovered and within 1 pair of REG3_KEYHINGE
          "the frequency hinge needs frequent keys" REG3_KEYHINGE beats REG3_FREQHINGE by >= 4 vs 0
          "the repeat hinge replaces the key hinge" REG3_REPEATHINGE discovered and within 1 pair of REG3_KEYHINGE
          "clamps" (per arm)                        (arm only) - (BASELINE only) <= 1 AND the arm's median commitment at
                                                    2400 (gate_stats_k's spread) below BASELINE's; printed for every arm
          "no reading applies"                      if none of the above applies on that task.
  ACROSS TASKS (descriptive, not a reading): each arm's ROUTED-BOUND drop from -pool to -allpool on the same seed numbers
  (the paired 2x2 per arm).
ALSO REPORTED: the hinges' selected positions per role and the frequency tables; firings and values; beta^R by role and
step; the margin per layer; the contingency tables; the probe; the plateaus by (stream, block, pair).
CHECKS (explore_g_s89_child.checks, asserted before any run, and X's arm A bit for bit): the vendored task file equals F's
blob at F's commit; S87's REG3_FREQHINGE|2700 reproduces bit for bit through 2400 under the current code (S87's path and this
screen's on the lite task); the statistics receive token ids only (one parameter, a LongTensor asserted, a float tensor
refused) and repeat_mask by hand; per task: each hinge with its switch off equals REG3 bit for bit through 2400, what each
hinge selects (counts per role, the key term's positions covered), each term's gradient reaches the gate's parameters (and
the embedding through its input) and not the stack, the repeat hinge's schedule over real training forwards; the perfect
gate 2/2 per task before any learned arm (run first).
BUDGET: at S87's rate about 5 h per task on 4 workers.

Run:  python explore_g_s89.py [--report-only]
"""
