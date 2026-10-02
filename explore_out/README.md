# explore_out — EXPLORATORY, not a result

Outputs of the outside-ideas screens on branch `claude/outside-ideas` (code: `explore_*.py`).
Nothing here is a result. A screen that comes out promising is handed back to the main branch
for a pre-registered test.

| file | what |
|---|---|
| `batch1_aborted_threads.log` | first launch of batch 1: stopped by the reproduction check. It ran in torch's default 4 threads and did not match X; at 1 thread it matches X bit for bit. |
| `batch1.log` | batch 1 (S1 local_gate, S2 kwta, S3 aux_gate), git 8544207, 1 thread per run; complete (63/63 runs, 229 min), per-seed reports and verdicts at the end. |
| `batch2_dry.log`, `dry_<screen>_results.json` | batch 2's labelled DRY RUN (git 52ce3b6): every CHECK, 1 seed per arm, runs capped at 3600 steps. It exercises the code; its numbers are not for interpretation. |
| `batch2.log` (S4 kwta_warm, S5 slow_mem, S6 far_cue, ref_a2400) | batch 2, screen code of git 6f529b5 (unchanged since; driver flags added at 9697dec), 1 thread per run; complete (83/83 runs). Run in 6 segments: each after the first stopped by the session's background time limit and resumed (RESUME lines; repro check bit-identical to X in every segment; the full CHECK suite in the first segment, skipped later with the diff stat because the screen code was unchanged). Reports, verdicts and SUMMARY at the end. |
| `batch3_dry.log`, `dry_<screen>_results.json` (S7 slow_mem_ext, S8 far_gates, S9 near_check) | batch 3's labelled DRY RUN (git 3e70887): every CHECK, 1 seed per arm, runs capped at 3600 steps. It exercises the code; its numbers are not for interpretation. |
| `batch3.log` (S7 slow_mem_ext, S8 far_gates, S9 near_check) | batch 3, code of git 3e70887 (the full CHECK pass recorded at 82ee219 in `batch3_checks.json`), 1 thread per run; complete (70/70 runs). Run in 5 segments: each after the first stopped by the session's background time limit and resumed (RESUME lines; repro check bit-identical to X in every segment; later segments skipped the rest of the CHECK suite with the diff stat, the screen code being unchanged). Reports, verdicts and SUMMARY at the end. |
| `batch4_dry.log`, `dry_<screen>_results.json` (S10 far_express, S11 gate_reset, S12 slow_pos) | batch 4's labelled DRY RUN (git fdd1a46): every CHECK, 1 seed per arm, runs capped at 3600 steps. It exercises the code; its numbers are not for interpretation (S10's readings use counts of 10 seeds). |
| `batch4.log` (S10 far_express, S11 gate_reset, S12 slow_pos) | batch 4, code of git fdd1a46 (the full CHECK pass recorded at 6ac992e in `batch4_checks.json`), 1 thread per run; complete (50/50 runs). Run in 3 segments: each after the first stopped by the session's background time limit and resumed (repro check bit-identical to X in every segment; later segments skipped the rest of the CHECK suite, the screen code being unchanged). S10 (a)'s fits are in `far_express_fits.json`. Reports, verdicts and SUMMARY at the end. |
| `batch5_dry.log`, `dry_<screen>_results.json` (S13 slow_hinge, S14 stall_probe, S15 far_nudge_slow) | batch 5's labelled DRY RUN (git 98b2441): every CHECK, 1 seed per arm, runs capped at 3600 steps (S14: reruns capped at 3600, continuations 1200). It exercises the code; its numbers are not for interpretation. |
| `batch5.log` (S13 slow_hinge, S14 stall_probe, S15 far_nudge_slow) | batch 5, code of git 98b2441 (the full CHECK pass recorded at edf3152 in `batch5_checks.json`), 1 thread per run; complete (59/59 runs). The first segment was interrupted by a container restart during its CHECKs (no run started) and re-run in full; then 3 segments, each after the first stopped by the session's background time limit and resumed (repro check bit-identical to X in every segment). Reports, readings, verdicts and SUMMARY at the end. |
| `batch6_dry.log`, `dry_<screen>_results.json` (S16 kick_control, S17 far_slow_hinge, S18 stuck_memory) | batch 6's labelled DRY RUN (git 8407985): every CHECK, 1 seed per arm, runs capped at 3600 steps (S18: reruns capped at 3600, continuations 1200). It exercises the code; its numbers are not for interpretation. |
| `batch6.log` (S16 kick_control, S17 far_slow_hinge, S18 stuck_memory) | batch 6, code of git 8407985 (the full CHECK pass recorded at 254c6cf in `batch6_checks.json`), 1 thread per run; complete (38/38 runs) in one segment (78.9 min of runs after 9.1 min of CHECKs; repro check bit-identical to X). Reports, readings, verdicts and SUMMARY at the end. |
| `batch7_dry_check_failed.log` | batch 7's first labelled DRY RUN (git 7dadd6c): stopped by a CHECK before any run. S19's eta2_key matched the routing statistics' eta_key_by_key only to 2.4e-5 (tolerance 1e-5): float32 cancellation between two group means near 0.5 at init. The penalty now centres the values first (same quantity; agreement <= 1.4e-7); the dry run was repeated. |
| `batch7_dry.log`, `dry_<screen>_results.json` (S19 token_hinge, S20 kick_untimed, S21 merge_kick) | batch 7's labelled DRY RUN (git aa6c4ff): every CHECK passed, 1 seed per arm, runs capped at 3600 steps (S21: rerun to 2400, continuations 1200). It exercises the code; its numbers are not for interpretation. After it, S20's summary line was corrected to print each group's run count (it printed /20); nothing else changed. |
| `batch7.log` (S19 token_hinge, S20 kick_untimed, S21 merge_kick) | batch 7, code of git cea4756, 1 thread per run. **IN PROGRESS: 59 of 78 runs saved (merge_kick 8, token_hinge 30, kick_untimed 21).** Run in segments, each resumed after the session's background time limit stops it; queue order merge_kick, token_hinge, kick_untimed; pushed after every saved run. No report or verdict until the SUMMARY at the end of the log. |
| `<screen>_results.json` | every run's record (run_one's fields plus `lr`, `secs_wall`); `meta.provenance` holds CPU, torch, git, threads. |

## Batch 1 verdicts (screen rule in `explore_batch1.py`, fixed before any run)

Seeds 160-179, P=4, lr 1e-3 (test_channel_binding.SUB_LR), paired with X's recorded arm A
(test_short_conv), which this container reproduces bit for bit at 1 thread. Outcome DISCOVERED.

| screen | DISCOVERED | X arm A | candidate only / A only | McNemar p | verdict |
|---|---|---|---|---|---|
| S1 local_gate (LOCAL3) | 15/20 | 7/20 | 12 / 4 | 0.077 | promising |
| S2 kwta (KWTA16; ceiling 2/3) | 2/20 | 7/20 | 1 / 6 | 0.125 | not |
| S3 aux_gate (AUX1) | 1/20 | 7/20 | 1 / 7 | 0.070 | not |

## Batch 2 verdicts (batch 1's screen rule, unchanged; fixed before any run)

Seeds 160-179, P=4, lr 1e-3 (test_channel_binding.SUB_LR), paired with X's recorded arm A (test_short_conv).
ROUTED* = margin >= 0.9 and eta_key_by_stream > 0.9; X's arm A at 2400 from the re-run (explore_ref_a2400,
20/20 reproduce X's record exactly).

| screen | DISCOVERED | X arm A | cand. only / A only | McNemar p | ROUTED*@1200 (vs A) | verdict |
|---|---|---|---|---|---|---|
| S4 kwta_warm (KWTA_WARM) | 7/20 | 7/20 | 5 / 5 | 1 | 15/20 vs 6/20 (12 / 3, p = 0.035) | not |
| S5 slow_mem (SLOW_MEM) | 12/20 | 7/20 | 6 / 1 | 0.125 | 12/20 vs 6/20 (7 / 1, p = 0.070) | inconclusive; reading (b) a race |

S6 far_cue (descriptive; header layout): far_ceil bound 3/3 (valid); far_A bound 0/10, far_L3 bound 2/10
(both unrouted, key-split gates); far_L3 - far_A = +2: neither reading applies.

## Batch 3 verdicts (rules fixed before any run)

S7 (batch 1's rule, the 20 new seeds 180-199, paired with X's recorded arm A):

| screen | DISCOVERED | X arm A | cand. only / A only | McNemar p | verdict |
|---|---|---|---|---|---|
| S7 slow_mem_ext (SLOW_MEM, seeds 180-199) | 12/20 | 5/20 | 9 / 2 | 0.065 | promising |
| pool 160-199 (S5 + S7; not a verdict) | 24/40 | 12/40 | 15 / 3 | 0.0075 | ROUTED*@1200 23/40 vs 11/40 (15 / 3, p = 0.0075) |

S8 (header layout, seeds 160-169; rule per arm: promising >= 4/10, not <= 1/10; batch 2's far_A 0/10):
far_EMA 0/10 (all 10 KEY splits), far_SEL 0/10 (all 10 KEY splits), far_A_slow 0/10 (8 POSITION): not, not, not.

S9 (grouped layout, seeds 160-169, descriptive; X's arm A 4/10): EMA_near 0/10, SEL_near 0/10 (KEY splits
9 and 8): both "does not keep the near case", so S8's two new gates fail on the standard task as well.

## Batch 4 verdicts (rules fixed before any run)

S10 far_express (header layout): (a) CHECK 45's fit, argmax match: arm A's gate 1.000, EMA 1.000, SEL 1.000
(all fit), LOCAL3 0.765 (does not fit; the control). (b) far_nudge (A_nudge on the header layout) DISCOVERED 5/10
vs batch 2's far_A 0/10 (5 vs 0, p = 0.0625). Reading: neither applies (a gate fits; far_nudge 5/10, between 3
and 7).

| screen | DISCOVERED | X arm A | cand. only / A only | McNemar p | verdict |
|---|---|---|---|---|---|
| S11 gate_reset (RESET, seeds 160-179) | 10/20 | 7/20 | 3 / 0 | 0.25 | inconclusive |
| S12 slow_pos (SLOW_POS, seeds 160-179) | 15/20 | 7/20 | 9 / 1 | 0.0215 | promising |

S11: 27 resets over 20 runs; outcome by resets: 0 resets 7/9, 1 reset 3/3, 3 resets 0/8 (the re-drawn gates
ended in key splits). S12 vs batch 2's SLOW_MEM on the same seeds: 15/20 vs 12/20 (6 vs 3, p = 0.51); failures
SLOW_POS OTHER 5 (no POSITION, no KEY) vs SLOW_MEM POSITION 7, OTHER 1.

## Batch 5 verdicts (rules fixed before any run)

| screen | DISCOVERED | reference | cand. only / ref only | McNemar p | verdict |
|---|---|---|---|---|---|
| S13 slow_hinge (SLOW_HINGE, seeds 160-199) | 34/40 | SLOW_MEM 24/40 | 10 / 0 | 0.002 | promising; reading "noise floor" |
| S13 vs X's arm A (40 seeds; not the rule) | 34/40 | X A 12/40 | 23 / 1 | 3e-06 | |
| S13 vs SLOW_POS (160-179; not the rule) | 18/20 | SLOW_POS 15/20 | 4 / 1 | 0.375 | |
| S15 far_nudge_slow (seeds 160-169) | 10/10 | far_nudge 5/10 | 5 / 0 | 0.0625 | promising (>= 8/10) |

S13: ROUTED*@1200 on 160-179 15/20 (SLOW_POS 0/20); no uncommitted failures; failures OTHER 4, KEY 2 (SLOW_MEM
POSITION 11). Post hoc: on the 20 seeds where the hinge never fired, SLOW_HINGE's curves equal SLOW_MEM's; where it
fired (often on 1-3 of ~5000 batches), SLOW_MEM's 11 POSITION failures became 9 DISCOVERED and 2 OTHER.

S14 stall_probe (diagnostic): control far_nudge 160 SWAP-FIXES; header stalls: far_nudge 161 MEMORY-STUCK, far_nudge
164, 165 and far_A 165, 166 NEITHER (SWAP and both continuations stay at 0.52-0.55; far_A's gates match the perfect
gate everywhere); blocked 120, 130, 136 NEITHER (0.32-0.35). S15: 0 stalls (far_nudge 3).

CORRECTION to S14 (found while writing batch 6): S14's continuation loaded the run's optimizer with `opt.load_state_dict(opt0.state_dict())`, which shares the state tensors (exp_avg, exp_avg_sq, step) with the run's optimizer. The perfect-gate continuation ran first and was correct; the learned-gate continuation (the control) then started from the end-state weights with the Adam state the perfect-gate continuation left. SWAP, the perfect-gate continuations and the NEITHER and SWAP-FIXES readings are unaffected; far_nudge 161's MEMORY-STUCK depends on the control. Batch 6's S18 deep-copies the state and redoes the control on the 8 stalled runs (an extra).

## Batch 6 verdicts (rules fixed before any run)

| screen | outcome | reference | cand. only / ref only | McNemar p | verdict / reading |
|---|---|---|---|---|---|
| S16 kick_control (KICK, the 20 seeds where S13's hinge fired) | DISCOVERED 12/20 | SLOW_HINGE 14/20 | 1 / 3 | 0.625 | "a kick of that size suffices" (rule: >= 12/20; on the boundary) |
| S16 vs SLOW_MEM (not the rule) | 12/20 | SLOW_MEM 4/20 | 9 / 1 | 0.0215 | |
| S17 far_slow_hinge (header layout, seeds 160-169) | DISCOVERED 0/10 | far_A_slow 0/10, far_A 0/10 | 0 / 0 | 1 | not |

S16: ROUTED*@1200 KICK 4/20 vs SLOW_HINGE 8/20 (0 / 4, p = 0.125); failures KICK KEY 3, POSITION 3, OTHER 1,
STREAM-PARTIAL 1 (SLOW_HINGE OTHER 4, KEY 2; SLOW_MEM POSITION 11, KEY 3, OTHER 2). The three KICK POSITION failures
(173, 190, 196) are SLOW_HINGE discoveries; 196 is SLOW_MEM's only discovery that KICK lost.
S17: 9 KEY splits and 1 bound by a key split (166); far_A_slow's 8 POSITION failures are gone, replaced by KEY
splits. The hinge fired on 65 of 224400 training batches (1-22 per run).

S18 stuck_memory (diagnostic): no run reads "recency" or "primacy". With the learned gate the stalled header runs
answer 0.51-0.57 in either block (blocked: 0.32-0.36) and 0.50-0.57 at each key position (blocked: 0.31-0.39),
except far_nudge 161, which answers the first key of each block 1.000 and the others 0.36-0.45; 6-8% of wrong
answers are the other stream's value (1/15 = 0.067 for a uniformly drawn wrong value). 24000 perfect-gate updates:
far_nudge 161 "slow" (1.00 from +3600); the other 7 "basin" (header 0.51-0.56, blocked 0.32-0.35 throughout). Extra: S14's learned-gate control redone with the
run's own optimizer state leaves every S14 reading unchanged (far_nudge 161 MEMORY-STUCK; the rest NEITHER).
