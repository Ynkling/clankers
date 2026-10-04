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
| `batch7.log` (S19 token_hinge, S20 kick_untimed, S21 merge_kick) | batch 7, code of git cea4756 (the full CHECK pass recorded at 9a51dbe in `batch7_checks.json`), 1 thread per run; complete (78/78 runs). The first segment ran the full CHECK suite (15.8 min) and saved 4 runs; the container was then restarted (RESTART note; the 4 runs in flight were re-run); the second segment saved 69 and was stopped by the session's background time limit; the third saved the last 5. Repro check bit-identical to X in every segment; later segments skipped the rest of the CHECK suite, the screen code being unchanged. The first segment ran on a Xeon @ 2.10GHz, the later two (and both dry runs) on a Xeon @ 2.80GHz; the repro check was bit-identical to X on both, and every S21 rerun reproduced its recorded curve on either. (The RESUME line after the RESTART note carries the driver's fixed text, 'stopped by the session's background time limit'; that stop was the container restart.) S21's main-branch modules and records were read from 520fe51 (git show), not added to this branch. Reports, readings, verdicts and SUMMARY at the end. |
| `batch8_dry.log`, `dry_<screen>_results.json` (S22 far_tok_check, S23 far_single, S24 merge_split) | batch 8's labelled DRY RUN (git d15eb62, Intel Xeon @ 2.10GHz): the repro check bit-identical, every CHECK passed (17.1 min), 1 seed per arm, runs capped at 3600 steps (S24: rerun to 2400, continuations 1200). It exercises the code; its numbers are not for interpretation. After it, S24's report and the driver's S24 line take the continuation steps from the run records (they printed 9600 to 19200 in the dry run, where the continuations ran 2400 to 3600); print only, nothing else changed. |
| `batch8.log` (S22 far_tok_check, S23 far_single, S24 merge_split) | batch 8, code of git 572a00d (the full CHECK pass recorded at 8f9dec4 in `batch8_checks.json`; the banners' git f5a4d1b+dirty and 756d6f3+dirty are the watcher's explore_out commits on top of it, the screen code unchanged), 1 thread per run; complete (78/78 runs). The first segment ran the full CHECK suite (16.5 min) and saved 34 runs before the session's background time limit stopped it; the second (RESUME) saved the other 44. Repro check bit-identical to X in both segments; the second skipped the rest of the CHECK suite, the screen code being unchanged. Both segments ran on a Xeon @ 2.10GHz. S24's main-branch modules and records were read from 520fe51 (git show), not added to this branch. Reports, readings and SUMMARY at the end. |
| `batch9_dry_check_failed.log` | batch 9's first labelled DRY RUN (git 0a318be, Intel Xeon @ 2.10GHz): the repro check bit-identical; every CHECK of S26, S28 and S27 passed, and S25's first five; stopped by S25's last CHECK (run last on purpose): the Newton-Schulz output of torch's Muon (coefficients 3.4445, -4.775, 2.0315, 5 steps, bf16) on the 32 x 256 Gaussian probe has singular values in [0.6835, 1.0490], below the pre-registered [0.7, 1.3] (these coefficients map 1 to 0.70 and 1.05 to 0.68 by design; Keller Jordan's note gives S' ~ Uniform(0.5, 1.5)). No run was made. The range is the spec's; it awaits a decision. |
| `batch9_dry.log`, `dry_<screen>_results.json`, `dry_muon_recipe_lrsel_results.json` (S25 muon_recipe, S26 gate_langevin, S27 zloss_merge, S28 far_p8) | batch 9's labelled DRY RUN, second attempt (git a2b0e06, Intel Xeon @ 2.10GHz): the repro check bit-identical, every CHECK passed (18.4 min; S25's singular-value range [0.5, 1.5], widened from the pre-registered [0.7, 1.3] by the user's decision after the first attempt), S25's lr selection shortened (seed 160, 3600 steps; Muon lr 0.005 bound, 0.01 and 0.02 did not), 1 seed per arm, runs capped at 3600 steps (S27: rerun to 2400, continuation 1200). It exercises the code; its numbers are not for interpretation. After it, S28's report prints a note when the validity arm FAR8_CEIL binds fewer than 2/3 (print only; the reading is unchanged). |
| `batch9.log`, `muon_recipe_lrsel_results.json` (S25 muon_recipe, S26 gate_langevin, S27 zloss_merge, S28 far_p8) | batch 9, code of git d0c31ce (the full CHECK pass recorded at 49834e4 in `batch9_checks.json`; the banners' git 49834e4+dirty, c789cd8+dirty and 5326564+dirty are the watcher's explore_out commits on top of it, the screen code unchanged), 1 thread per run; complete (117/117 runs: 6 S25 lr-selection runs and 111 screen runs). Segment 1 ran the full CHECK suite (17.6 min) and the lr selection (Muon lr 0.005) and saved 60 before the background time limit; segment 2 (RESUME) saved 24 before the container was restarted (RESTART note; the S26 runs in flight were lost and re-run); segment 3 (RESUME) saved the last 33. Repro check bit-identical to X in every segment; later segments skipped the rest of the CHECK suite, the screen code being unchanged, and reused the stored lr selection. All on a Xeon @ 2.10GHz. S27's main-branch modules and records were read from 520fe51 (git show), not added to this branch. Reports, readings and SUMMARY at the end. |
| `batch10_dry.log`, `dry_<screen>_results.json`, `dry_muon_scale_valid_results.json` (S32 muon_slow, S33 muon_scale) | batch 10's labelled DRY RUN (git 3305b5f, Intel Xeon @ 2.10GHz): the repro check bit-identical, every CHECK passed (6.0 min; S33's in the child process on main's modules at 9c5939e, 4.5 min), S33's validity shortened (1 seed, 3600 steps: (a) Muon lr 0.005 bound at 1200; (b) 0.005 unbound at 0.893, 0.0025 bound at 3600), 1 seed per arm, runs capped at 3600 steps. It exercises the code; its numbers are not for interpretation. After it, before any real run: S33's projection timing changed. It had timed each arm alone as the difference of a 120- and a 20-step run, which underestimated the pool's per-step time (A_SLOW 22 ms projected vs about 35-40 ms per step in the dry run's pool) and gave negative values when the arms were timed at once; it now takes the median interval between optimizer steps, one child per arm running at once (32.5 / 33.7 ms for (a), 70.8 / 71.5 ms for (b) on the check). X's per-seed channel columns now use the Muon arms' rule for runs that stopped early. Runs, CHECKs and readings unchanged. |
| `batch10.log`, `muon_scale_valid_results.json` (S32 muon_slow, S33 muon_scale) | batch 10, code of git d16ec61 (the full CHECK pass recorded at cc87382 in `batch10_checks.json`; the banners' git cc87382+dirty, 334e1f1+dirty, 4321801+dirty and c877544+dirty are the watcher's explore_out commits on top of it, the screen code unchanged), 1 thread per run; complete (124/124 runs: 4 S33 validity runs, 80 S32 runs, 40 S33 runs). Segment 1 ran the full CHECK suite (6.0 min), S33's validity (Muon lr 0.005 bound both seeds in both configurations) and the projection (S33 worst case 5.06 h on 4 workers: no cut, stored), and saved 84 runs before the background time limit; segments 2-4 (RESUME) saved 8, 14 and 14 (runs in flight at each stop were lost and re-run). Repro check bit-identical to X in every segment; later segments skipped the rest of the CHECK suite, the screen code being unchanged, and reused the stored validity choice and cut decision. All on a Xeon @ 2.10GHz. S33 ran every run in a child process on the main line's modules at 9c5939e (`explore_main9c`, git show, read-only; nothing added to this branch). Reports, readings and SUMMARY at the end. |
| `batch11_dry.log`, `dry_<screen>_results.json`, `dry_muon_k16_valid_results.json` (S34 hinge_window, S35 muon_k16) | batch 11's labelled DRY RUN (git 05ff5f9, Intel Xeon @ 2.10GHz): the repro check bit-identical, every CHECK passed (5.9 min; S35's in the child process on main's modules at 9c5939e), S35's validity shortened (seed 240, 3600 steps: Muon lr 0.005 bound at 1200), the projection with pool-load timing (19.1 / 16.3 ms per step for S34's Muon / Adam arms, 37.1 ms for S35), 1 seed per arm, runs capped at 3600 steps. It exercises the code; its numbers are not for interpretation (S34's 'identical curves 0/1' is the 3600-step cap: the window runs equal the full-hinge records through 3600). After it, before any real run, report formatting only: the gradient ratios printed to 4 significant digits with the count of firings at ratio >= 100, and S35's outcome columns widened. Runs, CHECKs and readings unchanged. |
| `batch11.log`, `muon_k16_valid_results.json` (S34 hinge_window, S35 muon_k16) | batch 11, code of git 9f47868 (the full CHECK pass recorded at 55e35c7 in `batch11_checks.json`; the banner's git 55e35c7+dirty is the watcher's explore_out commit on top of it, the screen code unchanged), 1 thread per run; complete (102/102 runs: 2 S35 validity runs, 80 S34 runs, 20 S35 runs) in one segment (91.6 min of runs): the repro check bit-identical to X, the full CHECK suite (6.2 min), S35's validity (Muon lr 0.005 bound both seeds at 1200), the projection (batch worst case 4.34 h on 4 workers: no cut, stored). All on a Xeon @ 2.10GHz. S35 ran every run in a child process on the main line's modules at 9c5939e (`explore_main9c`, read-only). Reports, readings and SUMMARY at the end. |
| `batch12_dry_check_failed.log` (S36 split_plateau) | batch 12's first labelled DRY RUN (git 9e32a46): stopped at the CHECKs. One CHECK failed, in the check's own code: the synthetic-gate test compared the noise scale (0.1 x std(W_g[c*]), computed in float32 by the operation) with a float64 recomputation at a tolerance of 1e-12 (difference 3e-9); the rows themselves were exact (verified separately: W_g[c0], W_g[c*] and the other 14 rows bit for bit). Every other CHECK passed, among them the three bit-for-bit equalities through 7200 with the threshold at 0. Fixed in f714c03 (the scale recomputed in float32; rows and scale reported separately) before any real run. |
| `batch12_dry.log`, `dry_split_plateau_results.json` (S36 split_plateau) | batch 12's labelled DRY RUN, second attempt (git f714c03, Intel Xeon @ 2.10GHz): the repro check bit-identical, every CHECK passed (14.0 min, child processes on main's modules at 9c5939e), the projection with pool-load timing (49.7 ms per step for (a), 102.6 / 98.2 ms for SPLIT_D8_M / SPLIT_D8_A), 1 seed per arm, runs capped at 3600 steps (no check reaches 4800, so no split; the operation is exercised by the CHECKs). It exercises the code; its numbers are not for interpretation. After it, before any real run: the driver submits the runs in waves of 4 (b) runs then 4 (a) runs (submission order only), so a 2-hour segment ends on a wave of short runs rather than losing a wave of 43200-step runs in flight. Runs, CHECKs and readings unchanged. |
| `batch12.log`, `split_plateau_runtime_results.json` (S36 split_plateau) | batch 12, code of git faa1926 (the full CHECK pass recorded at fcb4a27 in `batch12_checks.json`; the banners' git fcb4a27+dirty, ae40635+dirty, e9e9c4e+dirty, e3d76a1+dirty and 6739017+dirty are the watcher's explore_out commits on top of it, the screen code unchanged), 1 thread per run; complete (30/30 runs) in 5 segments: segment 1 ran the full CHECK suite (13.6 min) and the projection (7.79 h on 4 workers: no cut, stored) and saved 6 runs; segments 2-5 (RESUME) saved 8, 8, 4 and 4, the runs submitted in waves of 4 (b) then 4 (a) runs (runs in flight at a stop were lost and re-run). Repro check bit-identical to X in every segment; later segments skipped the rest of the CHECK suite, the screen code being unchanged, and reused the stored cut decision. All on a Xeon @ 2.10GHz; every run in a child process on the main line's modules at 9c5939e (`explore_main9c`, read-only). Reports, readings and SUMMARY at the end (the per-check masses are in the log). |
| `batch13_dry.log`, `dry_<screen>_results.json` (S37 split_target, S38 gate_state) | batch 13's labelled DRY RUN (git 346adbf, Intel Xeon @ 2.10GHz): the repro check bit-identical, every CHECK passed (10.7 min, child processes on main's modules at 9c5939e: the trigger-off equalities through 7200, KEYMASS on the synthetic 2+1+1 gate, the four reruns through 2400 with the recomputed read gate exact, the decoder 1.0 / 0.1275 on the synthetic sets), the projection with pool-load timing, 1 seed per arm, runs capped at 3600 steps (no check reaches 4800, so no split and no S38 measurement at 4800). It exercises the code; its numbers are not for interpretation. |
| `batch13.log` (S37 split_target, S38 gate_state) | batch 13, code of git 346adbf, 1 thread per run. **IN PROGRESS: 4 of 50 runs saved (split_target 4, gate_state 0).** Run in segments, each resumed after the session's background time limit stops it; S37's runs first; pushed after every saved run. No report or reading until the SUMMARY at the end of the log. |
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

## Batch 7 verdicts (rules fixed before any run)

| screen | outcome | reference | cand. only / ref only | McNemar p | verdict / reading |
|---|---|---|---|---|---|
| S19 token_hinge FAR_TOK (header layout, seeds 160-169) | DISCOVERED 6/10 | S17 FAR_SLOW_HINGE 0/10 | 6 / 0 | 0.031 | promising (>= 4/10) |
| S19 token_hinge NEAR_TOK (grouped, seeds 160-179) | DISCOVERED 19/20 | S13 SLOW_HINGE 18/20 | 2 / 1 | 1 | "keeps the near case" (>= 16/20) |
| S20 kick_untimed (KICK120, seeds 160-199) | DISCOVERED 27/40 | SLOW_MEM 24/40 | 6 / 3 | 0.51 | inconclusive; reading "neither" (easy 18/20, hard 9/20) |
| S21 merge_kick (8 merged S=4 runs) | split under KICKS 0/8 | CONTROL 0/8 | | | "they do not" (<= 1/8) |

S19 FAR_TOK: of the 6 DISCOVERED (bound, VAL cos < 0.5), 3 route by stream at keys and values (ROUTED* at the end:
163, 164, 168; 168 reached 0.956 at the last evaluation, 24000); 166 and 167 separate the streams at value positions only
(eta^2 by stream at keys 0.00-0.01, at values 0.97-0.99; margin 0.04 and 0.77); 160 partly (0.37 / 0.54, margin 0.55).
Counting full stream routing alone, FAR_TOK is 3/10 (S17 0/10). 3 more bound without stream routing (161, 162, 165);
1 OTHER. The first label-free bindings with stream routing on the header layout in this branch. The key term fired on
5-10 batches per run. NEAR_TOK: ROUTED*@1200 0/20 vs SLOW_HINGE 15/20 (0 / 15, p = 6e-05): the key term fired on 1-9 batches
in every run (SLOW_HINGE's hinge never fired on 10 of these 20 seeds) and the routing came later (transitions 3600-6000).

S20: vs SLOW_HINGE 27/40 vs 34/40 (0 / 7, p = 0.016); vs S16's KICK on its 20 seeds 9/20 vs 12/20 (0 / 3, p = 0.25); the
untimed kick lost 2 easy seeds (186 POSITION, 191 KEY) and ROUTED*@1200 on the easy seeds fell to 12/20 (SLOW_MEM 20/20).
Adam's second moment of the gate tensors jumped x1e7-1e9 at the kick and stayed x7e6 (W_g) at update 2400.

S21: every in-job check passed (each rerun reproduced its record to 9600; CONTROL reproduced it to 19200; identical batches;
separate optimizer states; 16 kicks per run at 100x the batch gradient norm). The stream -> channel maps under KICKS stayed
those of CONTROL at every evaluation (one unmerged stream of SR 248 moved channel). CONTROL 0/8 is by construction (the
runs were chosen as merged to the end and CONTROL reproduces the record).

## Batch 8 verdicts (rules fixed before any run)

| screen | outcome | reference | cand. only / ref only | McNemar p | reading / verdict |
|---|---|---|---|---|---|
| S22 FAR_TOK_NEW (header, seeds 170-179) | DISCOVERED 4/10, ROUTED*@end 4/10 (BOUND 7/10) | S19 FAR_TOK on 160-169: 6/10, 3/10 | | | "replicates" (>= 4/10 and >= 2/10); pooled 20 seeds DISCOVERED 10/20, ROUTED*@end 7/20 |
| S22 FAR_TOK_LATE vs FAR_TOK (header, 160-179) | ROUTED*@end 3/20 (DISCOVERED 3/20, BOUND 9/20) | FAR_TOK 7/20 (10/20, 16/20) | 2 / 6 (0 / 7, 1 / 8) | 0.29 (0.016, 0.039) | "it costs" (c - b = 4) |
| S22 NEAR_TOK_LATE (grouped, 160-179) | ROUTED*@1200 15/20, DISCOVERED 18/20 | S19 NEAR_TOK 0/20, 19/20; S13 SLOW_HINGE 15/20, 18/20 | vs SLOW_HINGE 0 / 0, 0 / 0 | 1 | "routing returns" (>= 10/20 and >= 16/20) |
| S23 far_single (k = 1, header, 160-169) | BOUND: B_FAR 3/10, B_FAR_SLOW 7/10 | batch 3 far_A_slow 0/10, S19 FAR_TOK 9/10 | B_FAR_SLOW vs B_FAR 5 / 1 | 0.22 | "a single channel binds the header layout" (>= 3/10) |
| S24 merge_split (8 merged S=4 runs) | split by 19200: SPLIT 8/8 | NOISE 1/8, CONTROL 0/8 | | | "splitting un-merges" (SPLIT >= 4/8, NOISE <= 1/8) |
| S24 gate state at 9600 | merged ratio >= 0.5 x non-merged in 0/8 | | | | rule not met |

S22: the key term was above TAU on the first training batches of all 50 runs (eta2_key after 0 updates 0.05-0.96, median
0.58); in FAR_TOK_NEW it fired at update 1 in 9/10 runs. FAR_TOK_LATE and NEAR_TOK_LATE are FAR_SLOW_HINGE and SLOW_HINGE bit
for bit through update 2400 (CHECKed), so their ROUTED*@1200 are those arms' by construction. In FAR_TOK_LATE, 17 of 20 runs
were in a key split at 2400 (eta_key_by_key 0.60-1.00; S17's FAR_SLOW_HINGE ended KEY 9/10); the late key term then fired
3-5000 times (0 in 175, routed by stream at 2400) and 10/20 ended OTHER near 0.5 (FAR_TOK 3/20). In NEAR_TOK_LATE the key term fired after 2400 in 2 runs only (167,
171, OTHER in both arms); outcome tags and transitions equal SLOW_HINGE's on all 20 seeds: "routing returns" because the late
key term is silent on the grouped layout, not because it helps.

S23: B_FAR bound late (161, 162, 167; transitions 20400-24000), B_FAR_SLOW at 6000-19200 (160-166). On the header layout a
single channel binds without any partition, so a header-layout BOUND is not evidence of routing (k = 1 cannot be DISCOVERED:
VAL cos is 1 by construction). On the same seeds and batches the slow schedule binds 7/10 with k = 1 and 0/10 with arm A's
2-channel gate (batch 3's far_A_slow, POSITION 8/10). FAR_TOK's 3 'bound with every eta^2 near 0' runs (S19: 161, 162, 165)
are consistent with single-channel binding.

S24: every in-job check passed (reruns to 9600 and CONTROL to 19200 reproduce the records; identical batches; separate
optimizer states; W_g's Adam step restarted at 1; rows as specified; the run's state unchanged; recomputed gate exact). c* was
the shared channel in 8/8; c0 the spare channel in 7/8 (SR 248: stream 3's channel, the spare not being the least-used over all
positions). SPLIT split 1200-8400 updates after the operation, every run to 1.000 with each stream on its own channel; NOISE
split 223 only (at 12000), CONTROL none. At 9600 the merged streams' gate states were nearly identical (ratio 0.005-0.041 vs
0.70-1.41 for the non-merged pair; the pair's mean read gates at values within 0.01 of each other); after SPLIT their ratio was 1.13-1.54 at
19200 (CONTROL 0.012-0.028). Reading: the copy makes the shared and the idle channel's logits nearly tie (row distance 1.96-3.32
-> 0.24-0.40), so the small differences in h decide the channel and the task gradient can grow them; noise of the same size
without the tie (row distance unchanged) did not, in 7 of 8.

## Batch 9 verdicts (rules fixed before any run; S25's singular-value CHECK range widened by the user after the first dry run)

| screen | outcome | reference | cand. only / ref only | McNemar p | reading / verdict |
|---|---|---|---|---|---|
| S25 MUON_HINGE (Muon lr 0.005, seeds 160-179) | DISCOVERED 20/20 (ROUTED*@1200 20/20) | S13 SLOW_HINGE 18/20 (15/20) | 2 / 0 (5 / 0) | 0.5 (0.0625) | "the recipe survives Muon" (>= 15/20) |
| S25 MUON_A (Muon lr 0.005, seeds 160-179) | DISCOVERED 9/20 | X arm A 7/20 | 4 / 2 | 0.69 | (descriptive) |
| S26 gate_langevin (seeds 160-199) | DISCOVERED 27/40 (ROUTED*@1200 23/40) | SLOW_MEM 24/40 (23/40) | 4 / 1 | 0.375 | inconclusive |
| S26 vs SLOW_HINGE | 27/40 | SLOW_HINGE 34/40 | 0 / 7 | 0.016 | (printed) |
| S27 zloss_merge (8 merged S=4 runs) | split by 19200: 2/8 (225, 226) | S24 SPLIT 8/8, NOISE 1/8, CONTROL 0/8 | | | "neither" (2 is between 1 and 4) |
| S28 far_p8 (header layout, P=8) | FAR8_CEIL BOUND 0/3; B_FAR8_SLOW 0/10 | | | | "it does not" by the rule, but the validity arm failed: no answer |
| S28 FAR_TOK8 (seeds 160-169) | DISCOVERED 0/10, ROUTED*@end 2/10 | | | | "not promising" |

S25: the lr selection (perfect gate under Muon, 6000 steps) bound both seeds at 0.005 only (0.01: one seed stalled at
0.76; 0.02: both stalled). Under Muon the HINGE recipe routed by 1200 in all 20 runs (transition 1200-2400 in 19, 10800 in
166, where the hinge fired 281 times; it fired 0-4 times in the others, never in 6). Muon alone (MUON_A) did not fix arm A:
its 11 failures are all position splits (X's arm A: 8 POSITION, 4 KEY, 1 OTHER). The gate's update norm per step over
updates 1-2400 was 2.3e-2 to 3.2e-2 (per-run medians), max 2.9e-2 to 3.6e-2. Every MUON run used Muon lr 0.005; weight
decay 0; lm_head under Muon (a 2-D weight), the embedding under Adam.

S25 note (added with batch 10, from `muon_recipe_results.json`): hinge firings per MUON_HINGE seed (training batches
with a term above TAU): 160 2, 161 1, 162 2, 163 2, 164 1, 165 0, 166 281, 167 4, 168 0, 169 0, 170 0, 171 1, 172 1,
173 3, 174 0, 175 1, 176 2, 177 1, 178 1, 179 0 — 0-4 on 19 seeds (never on 165, 168, 169, 170, 174, 179), 281 on 166,
which was already stream-routed at 1200. At the first evaluation (1200), eta2 of the read gate at key positions by stream
was 1.00 on all 20 MUON_HINGE seeds (by index 0.001-0.009). Every MUON_A run was decided by 1200: stream-routed (eta2 by
stream 1.00 on 161, 163-166, 170, 175, 176, 178) or position-split (eta2 by index 0.98-1.00 on the other 11). The gate's
update norm, max/median per seed over updates 1-2400: MUON_HINGE 1.149-1.243, MUON_A 1.105-1.278 (below 1.15 on MUON_A
169, 172, 174, 177, 179 and MUON_HINGE 175, so batch 10's "1.15-1.3 on every seed" is approximate).

S26: LANGEVIN gained 175, 176, 177, 180 (SLOW_MEM: position splits; here DISCOVERED late, transitions 4800-9600) and lost
186 (DISCOVERED -> KEY). Below SLOW_HINGE on every seed where they differ (0 vs 7).

S27: every in-job check passed (reruns reproduce to 9600; the continuation's batches equal S24's recorded CONTROL batches;
optimizer state equal and separate). The z-loss did what it is built to do: the probe's mean (logsumexp)^2 fell from
23-46 at 9600 to 0.002-0.18 by 19200 in every run, all channels' mean logits pushed down together (to -1.9 to -5.0); the
merged pair stayed on one channel in 6 of 8 (accuracy 0.73-0.77), and 226 (at 10800) and 225 (at 16800) split to 1.000.

S28: the perfect gate did not bind the P=8 header layout in 24000 steps (final accuracy 0.871, 0.858, 0.547), so the
validity arm failed and the screen cannot say whether P=8 needs the partition. The single channel with the slow schedule
stayed at 0.20-0.24 (10/10). FAR_TOK8 routed by stream in 163 and 167 (VAL cos 0.00, margin 1.00) and reached 0.79-0.81
by 24000, close to the perfect gate's 0.86-0.87: on this task the memory, not the routing, is the limit within the budget.

## Batch 10 verdicts (rules fixed before any run; S33's projection timing changed after the dry run, before any real run)

| screen | outcome | reference | cand. only / ref only | McNemar p | reading / verdict |
|---|---|---|---|---|---|
| S32 MUON_SLOW vs MUON_HINGE (Muon lr 0.005, seeds 160-199) | MUON_HINGE DISCOVERED 39/40 (ROUTED*@1200 39/40) | MUON_SLOW 31/40 (29/40) | 8 / 0 (10 / 0) | 0.0078 (0.002) | "the hinge matters under Muon" (b - c >= 4, p < 0.1) |
| S32 MUON_SLOW vs SLOW_MEM (Adam), 160-199 | MUON_SLOW 31/40 (ROUTED*@1200 29/40) | SLOW_MEM 24/40 (23/40) | 10 / 3 (9 / 3) | 0.092 (0.15) | (printed) |
| S32 MUON_HINGE vs SLOW_HINGE (Adam), 160-199 | 39/40 | SLOW_HINGE 34/40 | 6 / 1 | 0.125 | (printed) |
| S33 (a) S=4, P=4, k=4, conv (seeds 240-249) | BOUND: MUON_SLOW 3/10, MUON_HINGE 4/10 | X A4k4 4/10 | 3 / 4; 2 / 2 | 1; 1 | "not promising at four streams" (better arm 4/10; >= 8/10 needed) |
| S33 (b) S=8, P=4, k=16, conv (seeds 260-269) | BOUND: MUON_SLOW 0/10, MUON_HINGE 0/10; median distinct channels at the end 1 and 2 | X HINGE_D8 0/10, median 2 | 0 / 0 | 1 | "does not break the coarse split" |

S32: every MUON_SLOW failure but one was a position split (8 POSITION, 1 KEY), and the hinge rescued all 8 (162, 166, 167,
173, 177, 194, 196, 197); 191 is KEY in both arms. On each rescued seed the hinge first fired at update 50-168, before any
run was routed: ROUTED* was 0/40 at 300 and 600 in both arms, and 38/40 (MUON_HINGE) vs 27/40 (MUON_SLOW) at 900. Firings
fell in updates 1-300 (47), 301-600 (9), 601-1200 (none) and later (283: 277 on 166, 3 on 191, 1 each on 172, 180, 190).
The 11 seeds where the hinge never fired were DISCOVERED in both arms. The re-runs of S25's 20 MUON_HINGE runs reproduced
S25's curves 20/20. Without the hinge, Muon still beats Adam's SLOW_MEM (31/40 vs 24/40, 10 vs 3, p = 0.092).

S33: the perfect gate bound under Muon lr 0.005 in both configurations ((a) at 1200, (b) at 4800 and 6000). (a): Muon did
not stop the merges. Two streams shared a channel at the end in 7/10 runs of each arm (6 MERGED, 2 share), as in X's A4k4
(merged or position-split in 6/10). The main line's hinge (pooled over channels) fired 0-5 times per run here. (b): no run
bound or exceeded 0.39 accuracy. Without the hinge, Muon put all 8 streams on one channel in 7/10 runs (median 1 channel at
the end, below X's HINGE_D8 at 2). With the hinge (0-250 firings per run) the median was 2, as X's: the coarse split is
unchanged. S31 was to matter if S33 (a) still merges under Muon; it does.

## Batch 11 verdicts (rules fixed before any run; S31 dropped, S29 and S30 deferred)

| screen | outcome | reference | full only / window only (ref only / cand. only) | McNemar p | reading / verdict |
|---|---|---|---|---|---|
| S34 MUON_H2400 (Muon lr 0.005, seeds 160-199) | DISCOVERED 38/40 | MUON_HINGE (full hinge) 39/40 | 1 / 0 | 1 | "an early window suffices" (b - c <= 1) |
| S34 MUON_H2400 vs MUON_SLOW | 38/40 | MUON_SLOW 31/40 | (0 / 7) | 0.016 | (printed) |
| S34 ADAM_H2400 (seeds 160-199) | DISCOVERED 34/40 | SLOW_HINGE (full hinge) 34/40 | 0 / 0 | 1 | "an early window suffices" (b - c <= 1) |
| S34 ADAM_H2400 vs SLOW_MEM | 34/40 | SLOW_MEM 24/40 | (0 / 10) | 0.002 | (printed) |
| S34 firing diagnostics | Muon: median hinge/task gradient ratio 850 (>= 100 at 37/58 firings), W_g momentum cosine >= 0.5 for a median 59 updates | Adam: median ratio 726 (28/48) | | | "directional kick" (median ratio >= 100, median >= 50 updates) |
| S35 MUON_HINGE16 (S=4, P=4, k=16, Muon lr 0.005, seeds 240-259) | BOUND 17/20 (15 at 2400, 2 at 3600) | X HINGE4k16 17/20 (3600-21600, median 7200) | 3 / 3 | 1 | "the Muon recipe carries to four streams at k=16" (>= 16/20, b - c <= 1) |
| S35 vs X A4k16 | 17/20 | X A4k16 12/20 | (1 / 6) | 0.125 | (printed) |

S34: under Muon the window and the full hinge differ on one seed: 166, where the full hinge fired 277 times after 2400 and
bound at 10800; with the window the run ended OTHER, its hinge terms above TAU on 5295 updates after 2400 (weight 0). 36/40
window runs have curves identical to the full-hinge runs (the full hinge never fired after 2400 there). Under Adam the
DISCOVERED outcomes are identical (32/40 identical curves); the 4 seeds where SLOW_HINGE fired late (167, 171, 193, 197: 34-
1938 firings, all ending OTHER) failed with the window too, as POSITION (167, 193, 197) and KEY (171), the terms above TAU on
21214-21509 updates after 2400 in the three position splits. Firing timing, which S13 did not log: under Adam 25 of 48 firings
fell in updates 1-300, 19 in 1201-2400 (Muon: 47 and 2 of 58); on the 10 seeds the window rescued from SLOW_MEM the first firing
was at update 49-155. At the firings the hinge's gradient on the gate was a median ~800x the task's (range 0.15x to 2e5x);
under Muon, W_g's momentum stayed aligned with a firing's hinge gradient (cosine >= 0.5) for a median 59 updates (0 to > 200).

S35: at k=16 the Muon recipe binds as often as the main line's HINGE (17/20 each; 3 vs 3) and earlier: 15 of 17 at 2400, 2
at 3600 (X's HINGE4k16: 3600-21600, median 7200). All 17 are BOUND ROUTED with one stream per channel from 4800; the 3
failures (245, 252, 259) are merges of two streams held from 4800 to the end. The hinge fired 33 times in 16 runs, all by
update 1200 (X's HINGE4k16: 380 firings in 18 runs over its windows). The perfect gate bound at Muon lr 0.005 on both seeds at
1200. Faster binding at an equal rate is a candidate for a pre-registered test on the main branch.

## Batch 12 verdicts (rules fixed before any run; one CHECK's own tolerance fixed after the first dry run)

| arm | outcome | paired reference | cand. only / ref only | McNemar p | reading / verdict |
|---|---|---|---|---|---|
| S36 SPLIT4k4_M (S=4, k=4, Muon, seeds 240-249) | BOUND 5/10 | S33 A_HINGE 4/10 | 2 / 1 | 1 | "it does not" (<= 5/10) |
| S36 SPLIT_D8_M (S=8, k=16, Muon, 260-269, 43200 steps) | BOUND 0/10 (at 28800 0/10) | S33 D8_HINGE 0/10 | 0 / 0 | 1 | "it does not" (>= 3/10 needed) |
| S36 SPLIT_D8_A (S=8, k=16, Adam, 260-269, 43200 steps) | BOUND 0/10 (at 28800 0/10) | X HINGE_D8 0/10 | 0 / 0 | 1 | "it does not" (>= 3/10 needed) |

S36 (a): the trigger fired at 31 of 62 checks in 7 of 10 runs. Two runs were rescued by their first split: 240 (split at
4800, 2+1+1 -> 1+1+1+1 by 7200, bound at 7200) and 247 (split at 7200, bound at 8400); one was lost: 245 (A_HINGE bound, not
routed, at 27600; with the trigger 6 splits and two merged pairs, 2+2, at the end). In the 5 runs that stayed merged the
trigger fired 5-7 times each; the splits moved streams between the 2+1+1 and 2+2 patterns without separating a merged pair,
and in 3 of them (241, 243, 249) the held-out accuracy fell from ~0.75 to 0.35-0.51 within 2400 updates of a split (HARM)
before returning to the plateau.

S36 (b): the trigger fired at 98 and 97 of 160 checks (every run split 7-12 times); no run bound by 43200 under Muon or Adam,
the best final accuracy was 0.49 in each arm, and no split was followed by a HARM drop (the accuracy was already near chance).
The streams stayed on 1-3 channels (median at the end: SPLIT_D8_M 3, SPLIT_D8_A 2, as X's HINGE_D8 2); a split's new
channel rarely held a stream by the next check: copying the busiest row to the idlest does not break the coarse split.
Verdict: not promising in either configuration.
