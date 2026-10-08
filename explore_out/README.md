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
| `batch13.log`, `split_target_runtime_results.json` (S37 split_target, S38 gate_state) | batch 13, code of git c2b30dc (the full CHECK pass recorded at b2981bd in `batch13_checks.json`; the banners' git b2981bd+dirty and b981809+dirty are the watcher's explore_out commits on top of it, the screen code unchanged), 1 thread per run; complete (50/50 runs: 30 S37, 20 S38) in 2 segments: segment 1 ran the full CHECK suite (11.3 min) and the projection (4.14 h on 4 workers: no cut, stored) and saved 33 runs; segment 2 (RESUME) saved the last 17. Repro check bit-identical to X in both segments; segment 2 skipped the rest of the CHECK suite, the screen code being unchanged. Every S38 rerun reproduced its recorded curve through 9600 (20/20). All on a Xeon @ 2.10GHz; every run in a child process on the main line's modules at 9c5939e (`explore_main9c`, read-only). Reports, readings and SUMMARY at the end. |
| `batch14_dry.log`, `dry_<screen>_results.json` (S39 gate_memory, S40 gate_prev) | batch 14's labelled DRY RUN (git 93c2755, Intel Xeon @ 2.10GHz, torch 2.14.0): the repro check bit-identical, every CHECK passed (4.8 min, child processes on main's modules at 9c5939e: S40's shared parameters bitwise equal at update 0, W_prev (32, 32) std 0.1008 vs W_in's 0.0985; the gate input at key positions decodes the stream at 1.000 with GATE_PREV and 0.125 without; the optimizer groups cover every parameter once with W_prev in the gate group; with W_prev fixed at 0 and excluded, D8_PREV_M = S33's D8_HINGE|260 and D8_PREV_A = X's HINGE_D8|260 bit for bit through 1200; S39's four reruns with every measurement reproduce their records through 1200, the recomputed read gate exact, the decoder 1.0 / 0.1275 on the synthetic sets; the served modules), the projection with pool-load timing, 1 seed per arm, runs capped at 2400 steps (S40 probes at 1200 and 2400 only). It exercises the code; its numbers are not for interpretation. After it, one report-only change: S40's table also prints the decodability of the gate input W_in v_t + W_prev v_(t-1) at key positions (already recorded by the run). |
| `batch14.log`, `gate_prev_runtime_results.json` (S39 gate_memory, S40 gate_prev) | batch 14, code of git 07ccfb7 (the screen code is unchanged since; the full CHECK pass recorded at 0ce53bf in `batch14_checks.json`; the banners' git 7105374+dirty, 01753bb+dirty and d3621c3+dirty are the watcher's explore_out commits on top of it), 1 thread per run; complete (40/40 runs: 20 S40, 20 S39) in 3 segments: segment 1 ran the full CHECK suite (4.5 min) and the projection (4.20 h on 4 workers: no cut, stored) and saved 24 runs (4 S40, all 20 S39); segment 2 (RESUME) saved 8 and segment 3 (RESUME) the last 8, all S40 (the 4 S40 runs in flight at each stop were re-run from the start). Repro check bit-identical to X in all three segments; segments 2 and 3 skipped the rest of the CHECK suite, the screen code being unchanged. Every S39 rerun reproduced its recorded curve through 2400 (20/20). All on a Xeon @ 2.10GHz; every run in a child process on the main line's modules at 9c5939e (`explore_main9c`, read-only). Reports, readings, labels and SUMMARY at the end. |
| `batch15_dry_check_failed.log` (S41 gate_cap, S42 wh_spectrum) | batch 15's first labelled DRY RUN (git d57f621), STOPPED by a CHECK: on this container's CPU (Intel Xeon @ 2.80GHz; every earlier batch ran on a Xeon @ 2.10GHz) the Muon runs no longer reproduce their records. The repro check (X's arm A, Adam) was bit-identical and every other CHECK passed (CAP keeps sigma_max(W_h) <= 0.5 + 1e-6, sigma_max(W_h) 1.08 at init so the cap acts from update 1; NOREC's W_h stays 0 with no update and every other initial value is D8_PREV_M's; the groups cover every trainable parameter once; the decoder check), but CAP_M and PREV_CAP_M with the cap at infinity did not equal S33's D8_HINGE|260 and S40's D8_PREV_M|260 through 1200 (identical statistics at update 0, different at 1200). Diagnosed after the stop (scratchpad runs, not in the log): S33's plain D8_HINGE recipe without any batch 15 code gives the same curve as the capped-at-infinity run (held-out 0.0674 at 1200 vs the record's 0.0703), so the cap is not the cause; the Adam configurations reproduce bit for bit here (X's HINGE_D8|260 and S40's D8_PREV_A|260 through 1200). The Muon optimizer's Newton-Schulz step runs in bfloat16, whose CPU kernels depend on the instruction set, so Muon trajectories are machine-specific. No batch 15 run was made. |
| `batch15_dry.log`, `dry_<screen>_results.json` (S41 gate_cap, S42 wh_spectrum) | batch 15's second labelled DRY RUN (git 2a50422, Intel Xeon @ 2.80GHz, torch 2.14.0), after the user's decision on the machine change (fresh Muon references REF_HINGE_M and REF_PREV_M on this CPU, a Muon repro check per segment, S42's D8_HINGE_M checked against REF_HINGE_M): the repro check bit-identical; the Muon fingerprint taken (REF_HINGE_M|260 through 1200; not stored in a dry run); every CHECK passed (8.8 min): with the cap at infinity CAP_M and PREV_CAP_M equal fresh REF_HINGE_M|260 and REF_PREV_M|260 bit for bit through 1200 (curve, statistics, every diagnostic) and PREV_CAP_A (Adam) equals S40's recorded D8_PREV_A|260; the REF runs' statistics at update 0 equal the 2.10GHz records'; the cap keeps sigma_max(W_h) <= 0.5 + 1e-6 (1.08 at init: it acts from update 1); NOREC keeps W_h at 0 with no update; the groups cover every trainable parameter once; S42's D8_HINGE_M|260 equals REF_HINGE_M|260 through 1200 and HINGE_D8_A|260 equals X's record through 1600, every 50 updates measured; rho and sigma_max agree with numpy; the decoder check; the served modules. Then the projection with pool-load timing (about 130-143 ms per D8 step on this CPU, against about 96 ms on the 2.10GHz one), 1 seed per arm, runs capped at 2400 steps. It exercises the code; its numbers are not for interpretation. After it, cosmetic changes only: S42's pairing line, and the per-run log lines (sigma_max at init to 4 decimals; S42's Muon runs show their 2.10GHz-record comparison). |
| `batch15.log`, `gate_cap_runtime_2p10_results.json` (S41 gate_cap, S42 wh_spectrum) | batch 15, complete (45/45 runs: 35 S41, 10 S42) in 6 segments. Segment 1 (git 1432fe0 + 35d785e watcher commits) ran on a Xeon @ 2.80GHz under the user's first machine decision; of its 14 runs, the six Adam runs (PREV_CAP_A|260, S42's HINGE_D8_A 260-264) are kept and the eight Muon runs are in the `_2p80` stores, unused (see that row). Segments 2-6 ran on the Xeon @ 2.10GHz of batches 1-14 under the user's second decision (the pre-registered pairing with the recorded runs; code of git 90c3a28, unchanged since; the full CHECK pass recorded at 5ce369b in `batch15_checks.json`): segment 2 (RE-PLAN) ran the full CHECK suite (6.5 min) and the projection (9.50 h on 4 workers > 9 h: PREV_CAP_A cut to 260-264, stored; 8.42 h after the cut) and saved 9 runs; segments 3-6 (RESUME) saved 8, 8, 8 and 6. The repro check was bit-identical to X in all six segments, and the Muon record check (S33's D8_HINGE|260 through 1200 against its record) in all five 2.10GHz segments; segments 3-6 skipped the rest of the CHECK suite, the screen code being unchanged. Every S42 rerun reproduced its record through 1600 (10/10) and repeated S39's measurements exactly at the 10 shared updates. Every run in a child process on the main line's modules at 9c5939e. Reports, readings, labels and SUMMARY at the end. |
| `gate_cap_2p80_results.json`, `wh_spectrum_2p80_results.json`, `gate_cap_muon_repro_results.json`, `gate_cap_runtime_results.json` (batch 15, 2.80GHz, unused) | batch 15 segment 1 (in `batch15.log`, git 1432fe0) ran on a Xeon @ 2.80GHz under the user's first decision (fresh Muon references REF_HINGE_M / REF_PREV_M on that CPU, a per-segment fingerprint; projection 17.24 h, PREV_CAP_A cut to 260-264) and saved 14 runs before the background time limit stopped it. Segment 2's container had the Xeon @ 2.10GHz of batches 1-14 again: there S33's D8_HINGE|260 reproduces its record through 1200 and PREV_CAP_A|260 (Adam) equals the 2.80GHz run exactly (curve, statistics, diagnostics at 600 and 1200), checked before any change. By the user's second decision batch 15 returned to its pre-registered pairing with the recorded runs on the 2.10GHz CPU: the REF arms dropped; segment 1's eight Muon runs (CAP_M|260, PREV_CAP_M|260, PREV_NOREC_M|260, S42's D8_HINGE_M 260-264) moved to these `_2p80` stores, unused; its six Adam runs (PREV_CAP_A|260, S42's HINGE_D8_A 260-264) kept; the 2.80GHz fingerprint and runtime decision kept here as history; the runtime rule applied again on the 2.10GHz CPU (`gate_cap_runtime_2p10_results.json`); a per-segment check that S33's D8_HINGE|260 reproduces its record through 1200 (another CPU stops the batch). No third dry run: the changes restore the pre-registered pairing (exercised by the first dry run), the report code was exercised with --report-only, and the full CHECK suite runs again in segment 2. |
| `batch16_dry.log`, `dry_<screen>_results.json` (S43 window_recipe, S44 window_scale, S45 reservoir_gate, S46 stream_knobs, S47 write_decoders) | batch 16's labelled DRY RUN (code of git 93bdfe6, Intel Xeon @ 2.80GHz, torch 2.14.0+cu130, 1 thread per run): each screen's code SHA computed; the repro check bit-identical to X; the projection with pool-load timing (D8 steps 135-200 ms, S=4 k=16 68-100 ms, S43 38 ms): the full batch 44.33 h worst case on 4 workers, 28.28 h after the four listed cuts (S46 LR3E4, S45 RES_D8_M, S44's SPLIT, S44(a)'s Muon arm with its reference), printed and not stored or applied in a dry run; every CHECK passed (40 lines, 17.2 min), among them LOCAL3|160 with the decoder reproducing batch 1's record through 2400, the generic Adam recipe reproducing X's HINGE_D8|260 through 1200, the SPLIT arm with its threshold at 0 equal to LOCAL3_SLOW_D8_A bit for bit through 6000, the window gate's causality, decay 0.98 entering the decay mask only, the reservoir's rho 0.5 with no update to W_h, the decoder's synthetic checks; then 21 runs (1 seed per arm, runs capped at 2400 steps), none failed. It exercises the code; its numbers are not for interpretation. After it: S47's reproduction is compared through the rerun's own length (the dry run compared its 2400-step reruns through 4800: the addendum at the end of the log shows the corrected column from a report-only pass: HINGE_D8_A|260 and D8_PREV_A|260 reproduce their 2.10GHz records, the fresh Muon reruns equal S44's fresh references on this CPU); the driver reuses the stored projection timings on resume and packs a segment by the arms' observed run durations (harness only; no run's code SHA changes). |
| `batch16.log`, `batch16_checks.json`, `batch16_runtime_results.json`, `<screen>_results.json` (S43 window_recipe, S44 window_scale, S45 reservoir_gate, S46 stream_knobs, S47 write_decoders) | batch 16 (docs/audit_2026-10.md Appendix A + the user's two changes), complete: 165/165 runs, none failed or retried, in 15 segments (each RESUME banner says 'stopped by the background time limit'; that wording is generic: most segments ended themselves when no remaining run fitted their budget, and three were stopped by container restarts, noted in the log). Code of git 93bdfe6; the run code SHAs (window_recipe 46867e792241, window_scale 01767493e840, reservoir_gate f2c00d3f4a59, stream_knobs ea36a6e5d44b, write_decoders d6ccc68892a7) never changed, so segments 2-15 skipped the CHECK suite after segment 1's full pass (5.7 min). Segment 1 applied the runtime rule (projection 51.12 h on 4 workers > 10 h: cut S46 LR3E4, S45 RES_D8_M, S44 LOCAL3_SLOW_D8_A_SPLIT, S44 LOCAL3_SLOW16_M with its reference MUON_HINGE16_REF, re-projecting after each: 32.82 h, still over 10 h, so the never-cut arms ran on). Machines: the batch's CPU (the full CHECK pass) was a Xeon @ 2.80GHz; three container restarts (segments 2, 6, 7: their in-flight runs lost; segment 6's 4 saved runs kept); segments 3-7 had a Xeon @ 2.10GHz and ran Adam jobs only, by the rule fixed before any run. Every Muon run (LOCAL3_SLOW_D8_M, D8_HINGE_M_REF and S47's three Muon configurations, 35 runs) ran on 2.80GHz. Adam runs on 2.10GHz: S43 60, LOCAL3_SLOW16_A 18, RES_D8_A 4, S46 LR3E3 5, WH_SLOW 5, DECAY98 4, S47 HINGE_D8_A 5, D8_PREV_A 4; every other Adam run on 2.80GHz. The repro check (X's arm A, seed 160, 2400 steps) was bit-identical in every segment on both CPUs; S47's Adam reruns reproduce X's HINGE_D8 records and S40's D8_PREV_A records through 4800 (10/10), and its Muon reruns equal S44's fresh references on 2.80GHz through 4800. Harness changes during the batch (no run's code SHA changed): the stored projection timings reused on resume; segment packing by observed run durations, then by the CPU's observed/projected ratio; Muon jobs started first. Every run in a child process (S43 on this branch's modules, S44-S47 on the main line's at 9c5939e), 1 thread per run. Reports, readings and SUMMARY at the end; verdicts below. |
| `batch17_dry.log`, `batch17_dry_check_failed.log`, `dry_<screen>_results.json` (S48 window_d8, S49 window_fail) | batch 17's labelled DRY RUN (Intel Xeon @ 2.10GHz, torch 2.14.0+cu130, 1 thread per run). The first attempt stopped at a CHECK (`batch17_dry_check_failed.log`, with an addendum): in the threshold-0 check of W_SPLIT, the trigger's last check was the 6000-update check run's length minus 2400, so only the reference check ran and the condition 'at least two checks' failed; fixed (the check passes the full run's length for the trigger's horizon; the arms' runs are unchanged) and rerun. Second attempt (code of git f91f08b): each screen's code SHA computed; the repro check bit-identical to X; the projection with pool-load timing (eight-stream steps 68-74 ms, S49 13.5 ms): the full batch 12.83 h worst case on 4 workers, 11.32 h after cutting W_WIDTH4, 9.89 h after W_NOSLOW, printed and not stored or applied (the first attempt's timing gave 10.27 h at that point and would also have cut W_SPLIT_M with W_M_REF: the decision sits near 10 h and is taken once, from the first real segment's timing); every CHECK passed (9.0 min), among them W_SPLIT|260 at threshold 0 equal bit for bit through 6000 to batch 16's LOCAL3_SLOW_D8_A|260 (run on 2.80GHz), W_SPLIT_M|260 at threshold 0 equal to W_M_REF through 3600, the trigger at cap 2 equal to S37's row for row and at cap 3 firing at 4800, 9600, 14400, the width-4 window's causality and initialisation, decay 0.98 entering the decay mask only, W_NOSLOW / W_WIDTH4 / W_D98's lrs, S49's dense measurements inert (LOCAL3_SLOW|163 through 2400), and S49's seeds equal to S43's nine unbound seeds; then 9 runs (1 seed per arm, capped at 2400 steps), none failed; S49's rerun of 163 reproduces S43's record through 2400. After it, report-only changes (no run code SHA changed): the Muon reference's comparison label, and S49 marks runs whose eta^2 by key sits on the 0.5 threshold (a gate that splits the streams on two keys and puts the other two on different channels has exactly 0.5). It exercises the code; its numbers are not for interpretation. |
| `batch17.log`, `batch17_checks.json`, `batch17_runtime_results.json`, `window_d8_results.json`, `window_fail_results.json` (S48 window_d8, S49 window_fail) | batch 17 (the user's specification of 7 October 2026), complete: 49/49 runs, none failed or retried, in 4 segments (each RESUME banner's wording is generic: every segment ended itself when no remaining run fitted its budget; segment 3 started about 35 min late because the wait loop meant to start it matched its own command line, which affected no run). Code of git 2531480 (the last commit touching explore_*.py); the run code SHAs (window_d8 375b30d487df, window_fail 3d097ab8b5cc) never changed, so segments 2-4 skipped the CHECK suite after segment 1's full pass (8.9 min). Every run on the batch's CPU, a Xeon @ 2.10GHz, torch 2.14.0+cu130, 1 thread per run; the repro check (X's arm A, seed 160, 2400 steps) bit-identical in every segment. Segment 1 applied the runtime rule (projection 13.78 h on 4 workers > 10 h: cut W_WIDTH4 -> 12.17 h, W_NOSLOW -> 10.63 h, W_SPLIT_M with its reference W_M_REF -> 7.35 h, re-projecting after each). The Adam arms pair with batch 16's LOCAL3_SLOW_D8_A records (run on 2.80GHz): W_SPLIT equals them bit for bit through its first split on 10/10 seeds and W_LONG through 28800 on 10/10; W_SPLIT_D98 equals W_D98 through its first split on 10/10. S49's nine reruns reproduce S43's records bit for bit. GitHub refused pushes for about 15 min during segment 1 (HTTP 500); the commits went up when it recovered. Reports, readings and SUMMARY at the end of the log; verdicts below. |
| `batch18_dry.log`, `dry_<screen>_results.json` (S50 window_recipe_ablation, S51 split_k4, S52 two_stream_keysplits) | batch 18's labelled DRY RUN (code of git 7ef4e15, Intel Xeon @ 2.10GHz, torch 2.14.0+cu130, 1 thread per run). Before it, the CHECK functions were run once on their own: S52's validity check as first written (with k = 4 the perfect gate binds on seeds 160-161) could not pass — the plain perfect gate (no convolution) does not bind on the grouped two-stream layout at k = 2 or k = 4 (bit-identical curves; seed 160 at 0.54, 161 at 0.33 after 9600) — so it is run with this layout's recorded validity arm, test_short_conv's ceiling_conv (the perfect gate + conv 'layer'), padded to k = 4, which binds and routes on both seeds at 1200; a CHECK prints the plain gate's outcome and that the padding is inert. Also fixed then: the copy check named the wrong main file for stream_acc (it is test_scale_axes'), and eta2_key against S19's centred key_penalty has a float32 tolerance of 1e-4 (2.7e-5 seen). No arm or reading changed. The dry run: each screen's code SHA computed; the repro check bit-identical to X; S50's Muon reference decided (batch 16's LOCAL3_SLOW_D8_M|260, made on 2.80GHz, does not reproduce here through 1200: equal accuracy, loss 2.99112 vs 2.99034 — so W_M_REF runs as W_SPLIT_M's pair); the projection with pool-load timing (eight-stream steps 60-65 ms, S=4 k=4 26-27 ms, S=2 12.5 ms): 38.25 h of runs, makespan 9.58 h, the packing simulation 6 segments; the rule would cut W_SPLIT_W4 (5), W4k4 (5), W_SPLIT_NOSLOW (4: fits), printed and not stored or applied; every CHECK passed (13.5 min), among them W_RESET|261 equal to batch 16's LOCAL3_SLOW_D8_A|261 through its first reset and different after, W_SPLIT_NOSLOW and W_SPLIT_W4 at threshold 0 equal to S48's no-split recipes through 3600, W_SPLIT_M at threshold 0 equal to W_M_REF through 3600, S37's HINGE4k4_A|240 reproducing its record through 1200, W4k4_SPLIT at threshold 0 equal to W4k4 through 3600, the key term's weight 1 at update 2400 and 0 at 2401, W2_KEYHINGE at weight 0 equal to LOCAL3_SLOW through 1200, the copied main functions' syntax trees, and the key term equal to test_slow_start's eta2 code (7e-9); then 9 runs (1 seed per arm, capped at 2400 steps), none failed. It exercises the code; its numbers are not for interpretation. |
| `batch18.log` (S50 window_recipe_ablation, S51 split_k4, S52 two_stream_keysplits) | batch 18 (the user's specification of 8 October 2026), code of git 7ef4e15, 1 thread per run, on Intel(R) Xeon(R) Processor @ 2.10GHz. **IN PROGRESS: 75 of 120 runs saved (window_recipe_ablation 28/30, split_k4 4/10, two_stream_keysplits 43/80); cut by the runtime rule: W_SPLIT_W4, W4k4, W_SPLIT_NOSLOW).** Run in segments, each resumed after the session's background time limit stops it; pushed after every saved run. No report, reading or label until the SUMMARY at the end of the log. |
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

## Batch 13 verdicts (rules fixed before any run)

| screen / arm | outcome | paired reference | cand. only / ref only | McNemar p | reading / verdict |
|---|---|---|---|---|---|
| S37 SPLITK_M (S=4, k=4, Muon, seeds 240-249) | BOUND 6/10 | S33 A_HINGE 4/10; S36 SPLIT4k4_M 5/10 | 3 / 1; 1 / 0 | 0.625; 1 | "neither" (6 is between 5 and 8) |
| S37 HINGE4k4_A (Adam, main's HINGE recipe at k=4, run here) | BOUND 7/10 (5 routed, 2 not routed) | X A4k4 4/10 | 5 / 2 | 0.45 | (baseline, printed) |
| S37 SPLITK_A (Adam) | BOUND 9/10 (6 routed, 3 not routed) | HINGE4k4_A 7/10 | 2 / 0 | 0.5 | "the targeted split fixes four-stream merges" (>= 8/10) |
| S37 targeting (fired splits; checks with a shared and an empty channel) | KEYMASS: Muon 10/11, 49/51; Adam 7/8, 30/40 | all-position rule at the same checks: Muon 5/11, 29/51; Adam 6/8, 34/40 | | | (printed) |
| S38 D8_HINGE_M (Muon, S=8, k=16, 260-264) | stream from h at key positions, median at 4800: 0.13 (chance 0.125) | gate input 0.12 | | | "the eight-stream gate state does not carry the stream" |
| S38 HINGE_D8_A (Adam, S=8, k=16, 260-264) | median at 4800: 0.14 | gate input 0.12 | | | "the eight-stream gate state does not carry the stream" |
| S38 A_HINGE_M (Muon, S=4, k=4, 240-244) | median at 4800: 0.84 (chance 0.25) | gate input 0.25 | | | (printed alongside) |
| S38 HINGE16_M (Muon, S=4, k=16, 240-244) | median at 4800: 1.00 | gate input 0.25 | | | (printed alongside) |

S37, Muon: KEYMASS targeted 10 of 11 splits correctly (S36's all-position rule would have targeted 5 of them). The first split
rescued 3 runs (240 and 241 at 4800, 247 at 7200, the last one mistargeted onto a singleton channel and bound anyway), as S36
rescued 240 and 247; 241 is new. The 4 runs that stayed merged (243, 249: 2+1+1; 244, 245: two pairs, 2+2) took both their
splits, correctly targeted, from 7200 on, and the map did not change; 245 was lost against A_HINGE (bound, not routed, at 27600
there). No HARM in either arm (the cap kept the splits to 2 per run).

S37, Adam: the reading holds by the rule, but most of it is the baseline: main's HINGE recipe at k=4 binds 7/10 here without
any split (X's plain A4k4: 4/10). The split added 246 (bound, not routed) and 247 (routed); on 241 and 245 both arms bound
(SPLITK_A at 12000 and 8400, HINGE4k4_A at 10800 and 8400), and 249 failed in both. Three of SPLITK_A's 9 are bound but not
routed (243, 246, 248; 248 with all four streams on one channel). Under Adam the all-position rule targeted slightly better than
KEYMASS over all merged checks (34/40 vs 30/40).

S38: at S=8 the gate state h at the key positions carries almost nothing about the stream, under Muon and Adam alike (median
0.13 and 0.14 at 4800, chance 0.125), although the stream's CTX token is the token just before each key. Exceptions: D8_HINGE_M
263 (about 0.5 throughout), HINGE_D8_A 260 (rising to 0.54 by 9600), 261 (0.47-0.56 to 4800, chance at 9600) and 262 (0.81 at
1200, chance from 2400). At S=4 the same decoder reads the stream from h at 0.84 (k=4, Muon; 0.63-0.85 at 4800 in the four
merged runs) and 1.00 (k=16, Muon) at 4800. The gate input (the key's embedding) is at chance in every configuration, as it
should be. At S=8 h at a block's first key decodes slightly above its last key (median 0.19 vs 0.13 and 0.14 at 4800);
at S=4 they decode alike. Every rerun reproduced its recorded curve through 9600.

## Batch 14 verdicts (rules fixed before any run)

| screen / arm | outcome | paired reference | cand. only / ref only | McNemar p | reading / label / verdict |
|---|---|---|---|---|---|
| S40 D8_PREV_M (S33's D8_HINGE + GATE_PREV, Muon; S=8, P=4, k=16, seeds 260-269) | BOUND ROUTED 0/10 (9 MERGED, 3-4 share; 1 non-stream OTHER) | S33 D8_HINGE 0/10 (7 non-stream OTHER, 3 MERGED) | 0 / 0 | 1 | "it does not" (0/10); verdict: not |
| S40 D8_PREV_A (X's HINGE_D8 + GATE_PREV, Adam; seeds 260-269) | BOUND ROUTED 0/10 (7 MERGED, 2-6 share; 3 non-stream OTHER) | X HINGE_D8 0/10 (6 non-stream OTHER, 4 MERGED) | 0 / 0 | 1 | "it does not" (0/10); verdict: not |
| S39 D8_HINGE_M (S33's D8_HINGE, Muon, S=8, 260-264) | median stream decodability of h at key positions: 0.91 at update 0, 0.39 at 400, 0.18 at 600, 0.13 at 2400 (chance 0.125) | | | | "formed, then lost" |
| S39 D8_SLOW_M (S33's D8_SLOW, no hinge) | 0.91 at 0, 0.51 at 400, 0.26 at 600, 0.15 at 1200, 0.14 at 2400 | | | | "formed, then lost" |
| S39 HINGE_D8_A (X's HINGE_D8, Adam) | 0.91 at 0, 0.79 at 400, 0.50 at 600-800, 0.24 at 1200, 0.14 at 2400 | | | | "formed, then lost" |
| S39 A_HINGE_M (S33's A_HINGE, Muon, S=4, k=4, 240-244) | 0.95 at 0, lowest 0.54 at 300, 0.83 at 2400 (chance 0.25) | | | | "partial" (reference) |

S39: at S=8 the stream is in the gate state at the key positions from the start: with the initial weights (W_in, W_h at 0.1
scale) h at a key still carries the stream token one step back (median 0.91 at update 0, every configuration; S=4: 0.95). So
the "formed" half of the label is met by the initialization, not by training: training does not build this memory, it
removes it. The loss coincides with the recurrent term outgrowing the input term before the tanh: the median norm of
W_h h_(t-1) at key positions goes from 0.04 to 0.20 at 400 and 1.91 at 600 under Muon with the hinge (2.28 at 600 without
it), and from 0.06 at 400 to 0.32 at 1200 and 3.66 at 1600 under Adam, reaching 5-7 by 2400, while W_in v_t stays at 0.06-0.34
(at S=4, k=4 W_h h_(t-1) reaches 2.27 at 2400 while W_in v_t at the stream-token positions grows to 0.55, and the stream stays
decodable). By 2400 h at the stream-token position itself no longer decodes its own token (median 0.14-0.16 at S=8), so the
gate state has stopped reflecting its input, not just the previous one. It is mostly not saturation: the median fraction of h units with |h| > 0.95 at key
positions at 2400 is 0.05-0.15 (one run higher: D8_SLOW_M 262, 0.57). The read gate stays close to uniform at key positions at S=8 (entropy 2.0-2.7 nats of log 16 = 2.77;
S=4: 0.06 of 1.39). The hinge only brings the loss forward: D8_HINGE_M and D8_SLOW_M measure identically through 200 in all five
seeds (through 400 in 260 and 261), then 0.39 vs 0.51 at 400 and 0.18 vs 0.26 at 600, and alike from 1200 (0.15). The exceptions match
S38: D8_HINGE_M 263 stays at 0.39-0.73 from 300 (0.48 at 2400); D8_SLOW_M 264 drops to chance at 600 and comes back to 0.70 by 2400;
HINGE_D8_A 261 keeps 0.50-0.55 from 1200, 262 holds 0.81-0.94 to 1200 and is at chance from 1600.

S40: GATE_PREV puts the stream back into the gate state, but not one stream per channel. The gate input
W_in v_t + W_prev v_(t-1) at key positions decodes the stream at 0.97-1.00 (median) throughout under Muon and at 0.75-0.94
under Adam (265 loses it: 0.12-0.53), and h at key positions decodes it at a median 0.53 / 0.64 / 0.56 at 2400 / 4800 / 9600
under Muon (S38's D8_HINGE: 0.13 at 4800) and 0.57 / 0.34 / 0.29 / 0.53 at 1200-9600 under Adam. The failure moved from
non-stream to merged: the 8 streams end on 3-4 channels in 9/10 Muon runs (maps 3+3+2, 4+2+2, 3+2+2+1, 4+3+1; 267 5+3) and on
1-5 channels under Adam (266: all eight on one), where S33's D8_HINGE and X's HINGE_D8 ended non-stream OTHER in 7/10 and 6/10. No run bound in either
arm, so both readings are "it does not". The hinge fired in 0-103 of 28800 training batches per run. From the saved records
(rec["acc"], not part of the report's readings): the final held-out accuracy is higher than the paired run's in 9/10 Muon
pairs (median 0.36 vs 0.14) and 6/10 Adam pairs (0.30 vs 0.20).

Taken together (exploratory): at S=8 the gate loses the one-token memory early because its recurrent drive swamps the input,
and handing it the previous token directly (S40) restores the stream in the gate state and turns non-stream runs into merged
ones, but does not get eight streams onto eight channels by 28800 under either optimizer. Nothing here is handed back as
promising.

## Batch 15 verdicts (rules fixed before any run; the machine decisions are the user's, see the batch 15 rows)

| screen / arm | outcome | paired reference | cand. only / ref only | McNemar p | reading / label / verdict |
|---|---|---|---|---|---|
| S41 CAP_M (S33's D8_HINGE + CAP, Muon; S=8, P=4, k=16, seeds 260-269) | BOUND ROUTED 0/10 (7 MERGED, 4-8 share; 3 non-stream) | S33 D8_HINGE 0/10 | 0 / 0 | 1 | "it does not"; verdict: not |
| S41 PREV_CAP_M (S40's D8_PREV_M + CAP) | BOUND ROUTED 0/10 (10 MERGED: 5 with 2 share, 3 with 3, 1 with 4, 1 with 5) | S40 D8_PREV_M 0/10 | 0 / 0 | 1 | "it does not"; verdict: not (see below) |
| S41 PREV_NOREC_M (S40's D8_PREV_M, W_h = 0 frozen) | BOUND ROUTED 0/10 (10 non-stream KEY) | S40 D8_PREV_M 0/10 | 0 / 0 | 1 | "it does not"; verdict: not |
| S41 PREV_CAP_A (S40's D8_PREV_A + CAP, Adam; cut to 260-264) | BOUND ROUTED 0/5 (5 MERGED, 3-5 share) | S40 D8_PREV_A 0/5 | 0 / 0 | 1 | "it does not" (as of 5); verdict: not |
| S42 D8_HINGE_M (S33's D8_HINGE reruns to 1600, Muon, 260-264) | rho(W_h) > 1 first at 450 / 550 / 300 / 350 / 350; \|W_h h\|@key > 1 first at 500 / 550 / 400 / 1400 / 350 | | | | "gain crossing" (4/5 within 150 updates) |
| S42 HINGE_D8_A (X's HINGE_D8 reruns to 1600, Adam, 260-264) | rho > 1 first at -- / -- / 1400 / 400 / 600; \|W_h h\|@key > 1 first at -- / -- / 1400 / 400 / 600 | | | | "not" (3/5; 260 and 261 cross neither by 1600) |

S42: at initialization sigma_max(W_h) is 1.08 and rho(W_h) 0.58 (medians). In every run where |W_h h_(t-1)| at key positions
crossed 1 by 1600 (8 of 10), the spectral radius had crossed 1 at or before it, at the same 50-update measurement in 5 runs and 50
or 100 updates earlier in 2; the exception is Muon 263, where rho crossed at 350 and |W_h h|@key only at 1400 (S39: its stream
decodability stays near 0.5). Under Adam, 260 and 261 cross neither by 1600 (max rho 0.99 and 0.88), which is why Adam's label is
"not": a later crossing, not a reversed order. So the take-off of the recurrent term that removes the stream from h (S39) coincides
with the recurrent matrix becoming expanding.

S41: none of the four arms bound a run, so every reading is "it does not". What the arms did:
- CAP holds sigma_max(W_h) at 0.5 (the cap acts from update 1, since sigma_max is about 1.08 at init, and on a median of 24,700
  (CAP_M) and 26,700 (PREV_CAP_M) of 28,800 updates under Muon, 12,400 under Adam). Alone (CAP_M) it does not keep the stream in
  h: the median decodability at key positions is 0.25-0.44 (uncapped S38: 0.13), and |W_in v_t| itself grows late (median 7.1 at
  the end); 7 of 10 end MERGED,
  final held-out accuracy median 0.19 vs 0.14 for the paired D8_HINGE (higher in 7/10).
- With the look-back (PREV_CAP_M) the stream stays in h at key positions throughout (median decodability 0.93-1.00 from 600 to the
  end; S40's uncapped D8_PREV_M: 0.53-0.64), the value-position map spreads to 4-5 channels by 4800 (median), and at the end the
  8 streams sit on 3 to 7 channels (routing_k's map; 7 channels, i.e. one shared pair, in 2 runs; 5 channels in 4). By the end
  |W_prev v_(t-1)| dominates the pre-activation (median 15.8) and h is saturated at key positions (0.94 of units at |h| > 0.95).
  Final held-out
  accuracy is higher than the paired D8_PREV_M in 10/10 pairs (median 0.61 vs 0.37; 0.87 in 260 and 261). This is the closest the
  eight-stream configuration has come in these screens, but no run reached the bind criterion, and the pre-registered reading is 0/10.
- PREV_NOREC_M's input (v_t, v_(t-1)) cannot see the stream token at VAL positions (CTX is two back), so it cannot label the
  writes; its 10/10 KEY splits are a property of that window, not evidence about recurrence. The numbers as measured: every run
  ends non-stream KEY; the gate input's decodability of the stream at key positions falls from a median 0.56 at 600 to 0.19 at
  4800; |W_in v_t| at key positions grows past |W_prev v_(t-1)| (3.5 vs 1.6 at the end); final held-out accuracy is below the
  paired D8_PREV_M in 9/10.
- Under Adam (PREV_CAP_A, 5 runs) the cap keeps h's decodability at 0.60-0.94 (median), 5/5 MERGED with 3-5 sharing, accuracy as
  the paired runs (higher in 2/5).

Nothing is handed back as promising by the pre-registered readings. Printed for the record (not a reading): PREV_CAP_M's final
held-out accuracy and channel counts are the best of any eight-stream arm screened so far.

## Batch 16 verdicts (docs/audit_2026-10.md Appendix A with the user's two changes; rules fixed before any run)

| screen / arm | outcome | paired reference | arm only / ref only | McNemar p | reading / verdict |
|---|---|---|---|---|---|
| S43 LOCAL3_SLOW (S=2, P=4, k=2, no conv; 160-199) | DISCOVERED 31/40 (failures: STREAM-PARTIAL 5, KEY 4, no POSITION) | S13 SLOW_HINGE 34/40 | 5 / 8 | 0.581 | "neither" (replaces: >= 36/40; does not: <= 30/40); verdict: inconclusive |
| S43, printed | LOCAL3_SLOW 31/40 | S5/S7 SLOW_MEM 24/40 | 13 / 6 | 0.167 | LOCAL3 alone over 160-199: 33/40 (batch 1's 15/20 + the new 18/20) |
| S44(a) LOCAL3_SLOW16_A (S=4, P=4, k=16, conv; 240-259) | BOUND ROUTED 20/20 (all one-to-one by 4800; transitions 3600-10800) | X HINGE4k16 15/20 | 5 / 0 | 0.0625 | "carries to four streams"; verdict: promising |
| S44(b) LOCAL3_SLOW_D8_A (S=8, P=4, k=16, conv; 260-269) | BOUND ROUTED 3/10 (261, 263, 264; transitions 16800-21600) | X HINGE_D8 0/10 | 3 / 0 | 0.25 | (b) "breaks the eight-stream stall"; verdict: promising |
| S44(b) LOCAL3_SLOW_D8_M (Muon 0.005) | BOUND ROUTED 3/10 (261, 264, 267; transitions 7200-9600) | D8_HINGE_M_REF (S33's D8_HINGE, fresh, 2.80GHz) 0/10 | 3 / 0 | 0.25 | (same reading) |
| S44 LOCAL3_SLOW_D8_A_SPLIT, LOCAL3_SLOW16_M | cut by the runtime rule (with MUON_HINGE16_REF) | | | | no reading |
| S45 RES_D8_A (W_h frozen at rho 0.5; 260-269) | BOUND ROUTED 0/10 (MERGED 6, non-stream OTHER 4) | X HINGE_D8 0/10 | 0 / 0 | 1 | "does not" (RES_D8_M cut); verdict: not |
| S46 (descriptive; 260-264) | bound: LR3E3 0/5, WH_SLOW 0/5, DECAY98 0/5, PERFECT98 5/5 (all at 1200) | X HINGE_D8 0/5 | | | median final accuracy LR3E3 0.10, WH_SLOW 0.48, DECAY98 0.63 (X HINGE_D8 0.25); LR3E4 cut |
| S47 (descriptive) | median decodability of h at CTX / KEY / VAL at 4800 below | each rerun's record | | | Adam reruns reproduce 10/10 |

S44, what the window gate does at four and eight streams (observations, not readings):
- The window gate's state carries the stream at the write positions as well as the read: at S=4 (Adam) the median
  decodability of h at KEY / VAL is 0.73 / 0.81 at 1200 and 1.00 / 1.00 at 4800 (the two runs still training at 9600:
  0.69-0.76); at S=8 under Muon 0.93-1.00 at both
  from 1200 to 9600; at S=8 under Adam it dips to 0.55-0.67 at 2400-4800 and recovers to 0.87 / 0.88 at 9600. The recurrent
  reference D8_HINGE_M_REF stays at 0.14-0.28 (median) at both positions.
- These are the first eight-stream runs in these screens to bind routed (every recurrent arm in S38-S42 and S45 bound 0/10).
  The bound Adam runs transitioned late (16800-21600); the seven unbound Adam runs ended MERGED with 2 or 3 streams sharing,
  on 6 or 7 channels, at held-out accuracy 0.73-0.88. Under Muon the three bound runs were one-to-one from 4800; the seven
  unbound ones held a fixed 2+2+1+1+1+1 (or 2+2+2+2 / 2+1+1+1+1+1+1) map from 4800 to the end, accuracy 0.50-0.86.
- No hinge in any LOCAL3 arm; the references have main's hinge.

S43: LOCAL3 with slow memory bound 31/40, between the two pre-registered bands; no POSITION failure (prediction met), KEY
failures 2/20 on 160-179 (prediction <= 1: not met); every discovered run decodes the stream at KEY and VAL at 1.00 from 1200.

S45: the reservoir's state decodes the stream at CTX (median 0.81-1.00) but at KEY 0.43-0.72 and VAL 0.25-0.38: a frozen
recurrence of spectral radius 0.5 does not carry the stream two tokens; rho and sigma_max stayed constant (0.500000).

S46: with the recurrent gate no knob bound at eight streams; decay 0.98 and W_h at lr/10 raised the median final accuracy
(0.63, 0.48 against X's 0.25), lr 3e-3 lowered it (0.10) while rho(W_h) jumped to 3-7. The perfect gate binds at decay 0.98
(5/5 at 1200): the decay is valid.

S47 (closes the gap behind Revision 7, Section 11): median decodability of h at CTX / KEY / VAL —

| configuration | update 0 | 1200 | 2400 | 4800 |
|---|---|---|---|---|
| X's HINGE_D8 (Adam, S=8) | 1.00 / 0.91 / 0.47 | 0.45 / 0.23 / 0.20 | 0.17 / 0.15 / 0.14 | 0.25 / 0.14 / 0.13 |
| S33's D8_HINGE (Muon, S=8) | 1.00 / 0.91 / 0.47 | 0.26 / 0.28 / 0.26 | 0.47 / 0.57 / 0.67 | 0.47 / 0.49 / 0.49 |
| S33's A_HINGE (Muon, S=4, k=4) | 1.00 / 0.95 / 0.66 | 0.81 / 0.91 / 0.90 | 0.87 / 0.83 / 0.84 | 0.81 / 0.81 / 0.80 |
| S35's MUON_HINGE16 (S=4, k=16) | 1.00 / 0.95 / 0.66 | 0.98 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| S40's D8_PREV_A (GATE_PREV, Adam) | 0.99 / 1.00 / 0.69 | 0.53 / 0.58 / 0.40 | 0.37 / 0.46 / 0.34 | 0.42 / 0.53 / 0.48 |

At initialization the recurrent state decodes the stream at VAL (two back) well below KEY (0.47-0.69 against 0.91-1.00);
after training VAL is within 0.1 of KEY in the four plain recurrent configurations, and 0.05-0.18 below it under GATE_PREV. GATE_PREV's input decodes the stream at KEY (median
0.81-1.00) and at chance at VAL (0.12-0.13), as the audit said; its state h decodes VAL at 0.34-0.48.

Handed back as promising for a pre-registered test on the main branch: S44, the window gate (LOCAL3) with slow memory and no
hinge, at four streams (20/20 against 15/20) and at eight (3/10 under Adam and 3/10 under Muon, where every recurrent arm bound
0/10). Inconclusive: S43. Not: S45. Descriptive: S46, S47.

## Batch 17 verdicts (the user's specification of 7 October 2026; rules fixed before any run)

All S48 arms: S=8, P=4, k=16, conv, seeds 260-269, Adam, BOUND ROUTED; paired with batch 16's LOCAL3_SLOW_D8_A (3/10: 261,
263, 264). Reading per arm: "improves" if >= 6/10, "does not" if <= 3/10.

| screen / arm | BOUND ROUTED | paired reference | arm only / ref only | McNemar p | reading / verdict |
|---|---|---|---|---|---|
| S48 W_SPLIT (KEYMASS trigger, <= 3 splits) | 9/10 (all but 267; transitions 14400-22800, median 16800) | LOCAL3_SLOW_D8_A 3/10 | 6 / 0 | 0.0312 | "improves"; verdict: promising |
| S48 W_SPLIT_D98 (trigger + decay 0.98) | 8/10 (265, 268 bound NOT routed; routed transitions 8400-20400, median 14400) | LOCAL3_SLOW_D8_A 3/10 | 5 / 0 | 0.0625 | "improves"; verdict: promising (the splits, not the decay: see W_D98) |
| S48 W_D98 (decay 0.98) | 1/10 (260; 265, 268 bound NOT routed) | LOCAL3_SLOW_D8_A 3/10 | 1 / 3 | 0.625 | "does not"; verdict: not |
| S48 W_LONG (43200 updates) | 4/10 (261, 263, 264 as batch 16, + 267 at 31200) | LOCAL3_SLOW_D8_A 3/10 | 1 / 0 | 1 | "neither"; verdict: not (the merges hold) |
| S48 W_NOSLOW, W_WIDTH4, W_SPLIT_M | cut by the runtime rule (W_SPLIT_M with W_M_REF) | | | | no reading |
| S49 (descriptive) | S43's nine unbound LOCAL3_SLOW runs, reproduced 9/9 | each S43 record | | | below |

S48, what the splits did (observations, not readings):
- 19 splits fired in W_SPLIT (1-3 per run). Every target was labelled ok: c* held 2-5 streams and c0 none. In 6 of the 9
  bound runs the transition came at the evaluation right after the last split (1200 updates later), in the other three
  2400-4800 later. The one unbound run (267) used all three splits and ended MERGED (2 share), accuracy 0.88.
- W_SPLIT's maps before its first split are LOCAL3_SLOW_D8_A's (bit for bit). The arms differ only by the splits.
- Decay 0.98 alone settled the maps early (one partition, mostly 2+2+1+1+1+1 or 2+1+1+1+1+1+1, from 4800-12000 to the
  end). Two runs reached accuracy 1.0 without routing (BOUND NOT routed at 9600 and 12000). It lost all three seeds the
  reference bound (261, 263, 264), while its state decoded the stream at KEY / VAL at 0.93 / 0.95 (median) already at 4800. The window
  gate carries the stream there; the merge is not a decodability problem. With the trigger on the same runs (W_SPLIT_D98)
  the splits broke the held merge on all seven seeds where W_D98 ended MERGED; the two not-routed runs fired no split (accuracy 1.0 before a plateau
  check could fire) and are W_D98's runs bit for bit.
- W_LONG: through 28800 every run is LOCAL3_SLOW_D8_A's; the extra 14400 updates bound one more (267 at 31200). The other
  six ended MERGED (2 share) at accuracy 0.75-0.88; five held one partition from 12000-21600 to 43200, and 260 went from
  3+1+1+1+1+1 to 2+1+1+1+1+1+1 at 36000.

S49, LOCAL3_SLOW's nine unbound two-stream runs (S43). LOCAL3's read gate equals its write gate, so the KEY and VAL tables
are the same function of the window; they agree on every run.
- KEY runs (169, 173, 186, 192): the key split formed by update 600 in all four (held from 300, 0, 200, 150). At 173 eta^2
  by key was 0.55 at initialization. At the end three gates route by key alone (one key on one channel, three on the other;
  margin 0); 173 splits the streams on one key. The per-key pattern is the same at update 600 and at the end in all four.
- STREAM-PARTIAL runs (163, 183, 185, 188, 197): at VAL the gate is hard (no soft gates). It separates the two streams on
  some keys and shares a channel on the others:
  - on 3 of 4 keys: 185, 188;
  - on 2 of 4: 163, 183, with the other two keys on different channels, which puts eta^2 by key at exactly 0.5, the KEY
    threshold, so their KEY-split label follows sampling noise;
  - on 2 of 4: 197, with the other two keys both on channel 1.
  Their patterns were fixed by update 600-1200, except 197. Each passed through a key-dominated gate early (first crossings
  at 25-225); 183, 185 and 188 left it by 400-600, 163 fell back to the 0.5 threshold by 800. 197 kept a full key split (eta^2 1.0 from 800) through at least 12000
  and became stream-partial only by the end.
- So at two streams the window gate's key splits form within the first 600 updates, while the non-gate parameters are
  still at 1e-4. Where the streams do separate, they separate key by key and stop partway.

Handed back as promising for a pre-registered test on the main branch: S48's W_SPLIT, the window gate (LOCAL3) with slow
memory, no hinge and S37's KEYMASS split trigger (<= 3 splits) at eight streams: 9/10 BOUND ROUTED against 3/10 for the same
seeds without the trigger (6 / 0 discordant, p = 0.031), every split targeted as labelled. W_SPLIT_D98 (8/10) adds nothing
the splits do not. Not: W_D98, W_LONG. Cut: W_NOSLOW, W_WIDTH4, W_SPLIT_M. Descriptive: S49.
