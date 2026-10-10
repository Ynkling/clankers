# Results from each machine

Copies of the gitignored `*_results.json` files, so the other machine can pool them (`--also`).
One directory per machine: `X` is the cloud container, `L` the other machine.
Commit = the git head each run was started from (its `meta.git`); date = its `meta.started`.

| file | machine (CPU) | commit | date |
|---|---|---|---|
| X/channel_binding_results.json | X: Intel(R) Xeon(R) Processor @ 2.10GHz | 93b227c | 2026-09-25 14:08 |
| X/router_discovery_results.json | X: Intel(R) Xeon(R) Processor @ 2.10GHz | 4422dc4 | 2026-09-25 18:31 |
| X/router_curriculum_results.json | X: Intel(R) Xeon(R) Processor @ 2.10GHz | 1604209 | 2026-09-25 21:23 |
| X/router_confirm_results.json | X: Intel(R) Xeon(R) Processor @ 2.10GHz | 2271b55 | 2026-09-25 23:24 |
| X/readout_path_results.json | X: Intel(R) Xeon(R) Processor @ 2.10GHz | 58c625c | 2026-09-26 03:09 |
| X/router_reliability_results.json | X: Intel(R) Xeon(R) Processor @ 2.10GHz | 001c63e | 2026-09-26 18:42 |
| X/router_layout_results.json | X: Intel(R) Xeon(R) Processor @ 2.10GHz | 24987ec | 2026-09-26 23:40 |
| X/short_conv_results.json | X: Intel(R) Xeon(R) Processor @ 2.10GHz | f1cc9ab | 2026-09-27 05:14 |
| X/conv_lr_results.json | X: Intel(R) Xeon(R) Processor @ 2.10GHz | 3c69afd | 2026-09-27 17:43 |
| X/p_scaling_results.json | X: Intel(R) Xeon(R) Processor @ 2.10GHz | 745564f | 2026-09-27 23:46 |
| X/load_curriculum_results.json | X: Intel(R) Xeon(R) Processor @ 2.10GHz | b7c18f0 | 2026-09-28 07:23 |
| X/curriculum_confirm_results.json | X: Intel(R) Xeon(R) Processor @ 2.10GHz | c5ec279 | 2026-09-28 15:59 |
| X/scale_axes_results.json | X: Intel(R) Xeon(R) Processor @ 2.10GHz | 248c482 | 2026-09-28 23:14 |
| X/stream_channels_results.json | X: Intel(R) Xeon(R) Processor @ 2.10GHz | e5a24ae | 2026-09-29 18:36 |
| X/stream_recipe_results.json | X: Intel(R) Xeon(R) Processor @ 2.10GHz | 092937b | 2026-09-30 08:55 |
| X/stream_curriculum_results.json | X: Intel(R) Xeon(R) Processor @ 2.80GHz | ffdf0aa | 2026-10-01 00:53 |
| X/slow_start_results.json | X: Intel(R) Xeon(R) Processor @ 2.80GHz | 9c5939e | 2026-10-02 05:21 |
| X/recipe_scope_results.json | X: Intel(R) Xeon(R) Processor @ 2.80GHz | a429af9 | 2026-10-03 02:50 |
| X/early_recipe_results.json | X: Intel(R) Xeon(R) Processor @ 2.80GHz | 9f25dc8 | 2026-10-04 06:26 |
| X/muon_recipe_results.json | X: Intel(R) Xeon(R) Processor @ 2.80GHz | 74f5907 | 2026-10-06 06:19 |
| X/window_gate_results.json | X: Intel(R) Xeon(R) Processor @ 2.80GHz | 886a668 (first start 2f98f06: its six perfect-gate runs kept) | 2026-10-07 23:30 |
| L/router_discovery_results.json | L: 12th Gen Intel(R) Core(TM) i7-12650H | 4422dc4 (run at a9b3112) | 2026-09-25 12:42 |
| L/router_confirm_results.json | L: 12th Gen Intel(R) Core(TM) i7-12650H | 2271b55 (run at 5ba7c1d) | 2026-09-25 22:31 |
| L/router_curriculum_results.json | L: 12th Gen Intel(R) Core(TM) i7-12650H | 1604209 (run at 5ba7c1d) | 2026-09-26 07:27 |
| L/readout_path_results.json | L: 12th Gen Intel(R) Core(TM) i7-12650H | 58c625c (run at 5ba7c1d) | 2026-09-26 09:17 |
| L/router_reliability_results.json | L: 12th Gen Intel(R) Core(TM) i7-12650H | 001c63e (run at 403605d) | 2026-09-26 12:27 |
| L/router_layout_results.json | L: 12th Gen Intel(R) Core(TM) i7-12650H | 24987ec (run at 99992ec) | 2026-09-26 17:52 |
| L/short_conv_results.json | L: 12th Gen Intel(R) Core(TM) i7-12650H | f1cc9ab (run at 4610878) | 2026-09-26 22:42 |
| L/conv_lr_results.json | L: 12th Gen Intel(R) Core(TM) i7-12650H | 3c69afd (run at ce62073) | 2026-09-27 12:02 |
| L/p_scaling_results.json | L: 12th Gen Intel(R) Core(TM) i7-12650H | 745564f (run at 3b5e041) | 2026-09-27 21:34 |
| L/load_curriculum_results.json | L: 12th Gen Intel(R) Core(TM) i7-12650H | b7c18f0 (run at 98f3bd3) | 2026-09-28 00:08 |
| L/curriculum_confirm_results.json | L: 12th Gen Intel(R) Core(TM) i7-12650H | c5ec279 | 2026-09-28 11:59 |
| L/scale_axes_results.json | L: 12th Gen Intel(R) Core(TM) i7-12650H | 248c482 | 2026-09-28 20:39 |
| L/stream_channels_results.json | L: 12th Gen Intel(R) Core(TM) i7-12650H | e5a24ae | 2026-09-29 12:11 |
| L/stream_recipe_results.json | L: 12th Gen Intel(R) Core(TM) i7-12650H | 092937b | 2026-09-29 22:57 |
| L/stream_curriculum_results.json | L: 12th Gen Intel(R) Core(TM) i7-12650H | ffdf0aa (run at 631fd62) | 2026-09-30 15:27 |
| L/slow_start_results.json | L: 12th Gen Intel(R) Core(TM) i7-12650H | 9c5939e | 2026-10-01 21:19 |
| L/recipe_scope_results.json | L: 12th Gen Intel(R) Core(TM) i7-12650H | a429af9 (run at c69f1e6) | 2026-10-03 13:02 |
| L/early_recipe_results.json | L: 12th Gen Intel(R) Core(TM) i7-12650H | 9f25dc8 (run at 9437e01) | 2026-10-04 18:38 |
| L/muon_recipe_results.json | L: 12th Gen Intel(R) Core(TM) i7-12650H | 74f5907 | 2026-10-05 23:08 |
| L/window_gate_results.json | L: 12th Gen Intel(R) Core(TM) i7-12650H | 886a668 | 2026-10-08 11:33 |

X's `meta.started` times are UTC; L's are the laptop's local time (UTC−6). L's Phase III–IV files were committed on 10 October from the laptop's working copies (the test scripts write them next to themselves, gitignored); `L/channel_binding_results.json` is being rerun and will follow. A 5 October commit on the side branch `claude/bdh-repo-curl-obgdqk` (`664729f`) had placed copies of X's files under `results/L/`; those were not L's and were replaced.
