# Check (v6): S75 at four sources (session I, GTX 1660 SUPER) against the sixth Revision 9 draft and session I's write-up

Referee V. Generated in full by `docs/review_2026-10/rev9/check_S75_v6.py` (`python3 -I docs/review_2026-10/rev9/check_S75_v6.py > docs/review_2026-10/rev9/check_S75_v6.md` from the repository root after `git fetch`). It reads every file with `git show`, runs no session code and trains nothing. It is the v6 copy of `check_S75.py`: every record recomputation is kept, every draft row is anchored by a verbatim fragment located in tex6, and rows were added for every changed sentence in its scope.

**Pins.** Draft: `claude/rev9-draft` at `42c4557` (2026-10-12T23:14:04+00:00), `multichannel_hebbian_report_v9.tex` (= tex6, 568 lines; identical in length to `c5e4508` = tex5, the changes being in-place line replacements). Lines changed tex5 -> tex6: [36, 41, 54, 58, 201, 205, 256, 271, 349, 360, 368, 382, 441, 458, 477, 478, 495, 499, 516, 518]; of the lines this check reads ([36, 61, 456, 458, 462, 464, 472, 478, 516]), changed: [36, 458, 478, 516]. Session I: `origin/claude/explore-I` at `9034b83` (2026-10-12T22:53:23+00:00), its head when `42c4557` was committed (current tip `2fd4c7e`, 2026-10-13T00:43:46+00:00). The results file `explore_out/I/gpu/s75_S4_NVIDIA_GeForce_GTX_1660_SUPER_results.json` is byte-identical at `4c3984a` and `9034b83` (41 records; WIDE_SLOW seeds [830, 831, 832, 833, 834, 835, 836, 837, 838]); at the tip it holds 41 records and WIDE_SLOW [830, 831, 832, 833, 834, 835, 836, 837, 838]. report_2.md and s75_S4_descriptions.md are read at `9034b83` (last changed `559b942`); the pre-run note at `5bc6c2e` (unchanged); S74's records at `a2588da` (unchanged).

**Branch status at 42c4557's time (23:14:04 UTC).** Session I's tenth WIDE_SLOW seed (839) **has not landed**: the S75 results file last changed at `4c3984a` (2026-10-12T19:12:59+00:00), and neither `559b942` (report 2 section 6, 'seed 839 pending'), `30807ec` (window_gate pre-run note for the 1660), `8cb8dce` / `9034b83` (S75b driver and CHECKs) nor the later `2fd4c7e` adds it. So the WIDE_SLOW-vs-WIDE_SPLIT pairing stays at nine seeds: ROUTED neither on all nine, |SET difference| max 0.00325 (831), median 0.00064; the end set, eta^2 and one_to_one recomputations for 839 cannot be made.

**Conventions.** As check_S75: MATCH reproduces at the stated precision; a stated bound ('at most', 'within', 'no higher than') is MATCH only if the unrounded value satisfies it, PARTIAL if only after rounding; chance for the source probe is the ORACLE band 0.2367-0.2671, 'above chance' > 0.27; eta^2 = routing[L].eta.

## Counts

103 rows: **MATCH 82, PARTIAL 13, DISAGREE 1, NOT RECORD-CHECKABLE 7**. Draft rows (tex6 and the caller's paraphrases): 49 (MATCH 44, PARTIAL 3, DISAGREE 0, NOT RECORD-CHECKABLE 2); session I's files and the S75 rule: 54 (MATCH 38, PARTIAL 10, DISAGREE 1, NOT RECORD-CHECKABLE 5). check_S75 (tex5, 4c3984a) had 91 rows: MATCH 67, PARTIAL 16, DISAGREE 1, NOT RECORD-CHECKABLE 7.

### The fifth pass's S75 items at tex6

| fifth-pass item | tex6 | row | now |
|---|---|---|---|
| SINGLE's rise 'at most 0.03' (0.0329) | tex6:458 | 32 | **MATCH** |
| 'had lost it by 3600' (REG3 839 0.291) | tex6:458 | 38 | **MATCH** |
| 'within 0.02' (0.0235) | tex6:458 | 29 | **MATCH** |
| REG3 '0.535-0.540' (nine seeds; 839 0.588) | tex6:458 | 21 | **MATCH** |
| 'late rise of about 0.06 ... 0.29 and 0.28' | tex6:458 | 34 | **MATCH** |
| 'late rise ... took the probe to 0.29 and 0.28' (the peak 0.340) | tex6:458 | 35 | **PARTIAL** |
| the 'three window-gate runs' parenthesis | tex6:458 | 16 | **MATCH** |
| next step tex5:478 ('the window gate without the split ... first') | tex6:478 | 47 | **MATCH** |
| I's provenance row (4c3984a; 32 runs) | tex6:516 | 48 | **MATCH** |

### Every non-MATCH row

| # | where | statement | recomputed | status |
|---|---|---|---|---|
| 14 | tex6:458 | bound of 0.927 | not in the records | **NOT RECORD-CHECKABLE** |
| 35 | tex6:458 | ...took the probe no higher than 0.34 (0.29 and 0.28 at the end) [changed: tex5 said 'took the probe to 0.29 and 0.28'] | SINGLE 838: peak 0.3403 (L3, 8400), end max 0.2892; REG3 839: peak from 3600 0.2913 (3600), from 4800 0.2890 (7200), end max 0.2778 (its early hold: 0.3793 at 1200, 0.3339 at 2400) | **PARTIAL** |
| 39 | caller's paraphrase (check_S75) | chance in every arm from 3600 | ORACLE max 0.2671 (seed 831, 4800, final; evaluations > 0.27: 0); SINGLE max 0.3403 (seed 838, 8400, L3; evaluations > 0.27: 16); WIDE_SPLIT max 0.2676 (seed 831, 8400, final; evaluations > 0.27: 0); REG3 max 0.2913 (seed 839, 3600, L2; evaluations > 0.27: 17); WIDE_SLOW max 0.2647 (seed 831, 24000, final; evaluations > 0.27: 0) | **PARTIAL** |
| 41 | tex6:458 | decay-faded ideal learner 0.680 at four sources against 0.840 at two | not in the records | **NOT RECORD-CHECKABLE** |
| 46 | caller's paraphrase (check_S75) | WIDE_SLOW's nine seeds within 0.003 of WIDE_SPLIT (SET) | 830: -0.00057, 831: +0.00325, 832: +0.00064, 833: -0.00037, 834: +0.00049, 835: -0.00122, 836: -0.00112, 837: +0.00069, 838: -0.00006 | **PARTIAL** |
| 54 | rule | pre-run note 5bc6c2e (00:46:28 UTC 12 Oct) committed before any WIDE_SPLIT/REG3 run | not establishable from the records (see the timeline) | **NOT RECORD-CHECKABLE** |
| 60 | explore_out/I/gpu/s75_S4_prerun_note_2026-10-12.md:61-62 | a uniform loss of 0.017-0.036 on bins 2+ | per-bin bounds are not in the records | **NOT RECORD-CHECKABLE** |
| 67 | explore_out/I/report_2.md:24 | regenerating the summary reproduces the pushed copy exactly | not run (session code not executed) | **NOT RECORD-CHECKABLE** |
| 70 | explore_out/I/report_2.md:45 | rows of the source x channel table differ by at most 0.02 in any run | at the end max 0.0230; at any evaluation max 0.0415 | **PARTIAL** |
| 71 | explore_out/I/report_2.md:45 | the median margin is 0.03 | WIDE_SPLIT 0.0254; REG3 0.0227 | **PARTIAL** |
| 72 | explore_out/I/report_2.md:46 | REG3's third-layer gate puts about 0.7 on one channel, identically for every source | end L3 largest channel mass per seed 0.301-0.948 (median 0.664) | **PARTIAL** |
| 75 | explore_out/I/report_2.md:52-53 | the source probe reads chance (0.25 to 0.28) at every layer in every learned run, as in SINGLE | at the end 0.247-0.278; at 1200 the L2 probe reads 0.352-0.612 | **PARTIAL** |
| 83 | explore_out/I/report_2.md:22 | WIDE_SLOW is running on the 1660 now | 9/10 WIDE_SLOW records at 9034b83 (since 4c3984a); 839 pending | **PARTIAL** |
| 86 | explore_out/I/report_2.md:72 | eta^2 0.00; the rows of the source x channel table differ by at most 0.02 | eta max any evaluation 0.00057; row spread at the end max 0.0219, at any evaluation 0.0297 | **PARTIAL** |
| 88 | explore_out/I/report_2.md:74 | SET is 0.530-0.536, SINGLE's level, and the source probe is at chance (0.24-0.26) at every layer | SET 0.5300-0.5360; probe at the end 0.2430-0.2647; from 3600 0.2410-0.2647; at 1200 (L2) 0.3966-0.5796 | **PARTIAL** |
| 89 | explore_out/I/report_2.md:76 | paired by seed with WIDE_SPLIT: neither arm ROUTED on any of the 9 shared seeds; per seed SET differs by at most 0.003 (median absolute difference 0.0006) | ROUTED pairs [(False, False)]; \|difference\| max 0.00325 (seed 831), median 0.00064 | **PARTIAL** |
| 91 | explore_out/README.md:42 | Correction to the pre-run note: three WIDE_SPLIT runs had already finished on the 1660 when it was committed | no per-run time in the records (timeline section) | **NOT RECORD-CHECKABLE** |
| 92 | explore_out/I/report_2.md:29-31 | from the durations and the start time (18:04 on 10-11), WIDE_SPLIT 830-832 ran 23:11 (10-11) to 00:44 (10-12); the note was committed at 00:46 UTC | under one uninterrupted invocation: WIDE_SPLIT 830 starts 23:11:35, 832 ends 00:44:50; 5bc6c2e 00:46:28 | **PARTIAL** |
| 93 | explore_out/I/report_2.md:31 | the other seven WIDE_SPLIT runs and all ten REG3 runs came after it | under the same premises WIDE_SPLIT 833 started 00:44:50, 98.0 s before the note's commit | **DISAGREE** |
| 94 | explore_out/I/report_2.md:32-33 | the 1660's timestamps can only be UTC (ORACLE and SINGLE committed 23:42 UTC, so they ended before) | SINGLE 839 ends 23:11:35 machine time; 86652b1 authored 2026-10-11T23:42:23+00:00 | **PARTIAL** |
| 96 | explore_out/I/report_2.md:20-21 | deviation: the note assigned WIDE_SPLIT and REG3 to the RTX 3070; the 1660 ran them instead | not in the records | **NOT RECORD-CHECKABLE** |

## Rows

| # | line or file:line | statement | recomputed | status | note |
|---|---|---|---|---|---|
| 1 | tex6:36, tex6:61, tex6:458 | perfect gate ended at 0.880 and 0.892 (bar 0.9) | ORACLE 830 0.8915, 831 0.8801 | MATCH | [record] end.set; both < 0.9 -> not VALID |
| 2 | tex6:61, tex6:36, tex6:472, tex6:462 | window gate with split and register gate route 0/10 each; neither learned gate routed (ten seeds each) | ROUTED WIDE_SPLIT 0/10, REG3 0/10 | MATCH | [record] end.routing[L].eta >= 0.9 and one_to_one at every gate layer (explore_i_dense.routed) |
| 3 | tex6:61, tex6:36, tex6:472, tex6:462 | one channel learns to carry the source on none of ten (at four sources) | SINGLE probe max over layers: end 0.257-0.289; any evaluation 0.261-0.340 (838: 0.340) | MATCH | [record probe[L].acc] descriptive; against 0.96-0.98 at two sources (S74, below); the largest value, SINGLE 838's 0.340 (L3, 8400), is far from carrying |
| 4 | tex6:456 | pilot: perfect gate, one channel, single-source control at two sources, 12,000 updates, four seeds | iters [12000]; arms ['C1_ORACLE', 'C1_SINGLE', 'C2_NOMIX', 'C2_ORACLE', 'C2_SINGLE']; seeds [800, 801, 802, 803] | MATCH | [S74 record] C1 has no NOMIX record (stopped after the decision, report_1 l.18) |
| 5 | tex6:456 | three of four runs of the shorter-block configuration (C2: blocks of 3-7 words) | C2 SINGLE runs whose L3 probe reaches >= 0.9: 3/4 ([800, 801, 803]) | MATCH | [S74 record]  |
| 6 | tex6:456 | a step transition raised the set accuracy by 0.09-0.22 | 0.0872-0.2169 (800: +0.1080, 801: +0.0872, 803: +0.2169) | MATCH | [S74 record] transition = the evaluation with the largest one-evaluation rise of the L3 probe (as recompute_I) |
| 7 | tex6:456 | probe from the residual entering layer 3 from 0.54-0.58 to 0.96-0.98 | before 0.5377-0.5797; at the transition 0.9568-0.9803 | MATCH | [S74 record] probe['L3'].acc |
| 8 | tex6:456 | closing 44-83% of each seed's gap to the perfect gate | 800: 55%, 801: 44%, 803: 83% | MATCH | [S74 record] 1 - (ORACLE end - SINGLE end) / (ORACLE - SINGLE at the evaluation before the transition), same seed; report_1 l.32 says 60-80% (another definition; recompute_I row 7) |
| 9 | tex6:456 | the perfect gate's own stack never carried the source | S74 ORACLE probe max 0.5263 | MATCH | [S74 record] every layer, every evaluation, C1 and C2 |
| 10 | tex6:456 | four sources need sequences of about 770 tokens; S = 4, L = 768; 24,000 updates | config {'S': 4, 'V': 4, 'NB': 16, 'lmin': 6, 'lmax': 14, 'M': 176}; iters 24000; L = S*(NB+M) = 768 | MATCH | [record] meta.config, meta.iters; every record iters = {24000} |
| 11 | tex6:456 | S75: 32 runs, about 16 hours of training on that card | 32 registered runs; train_secs 16.34 h, wall secs 16.97 h | MATCH | [record] ORACLE 2 + SINGLE 10 + WIDE_SPLIT 10 + REG3 10; all R = {1} |
| 12 | tex6:456 | perfect gate on two seeds; both must end at a set accuracy of at least 0.9; ten seeds each | ORACLE seeds [830, 831]; others SINGLE 830-839 (n=10); WIDE_SPLIT 830-839 (n=10); REG3 830-839 (n=10) | MATCH | [record + code] VALID rule: s75_s4.py docstring OUTCOMES and report() (valid = both ORACLE end set >= 0.9) |
| 13 | tex6:456 | window gate at width 16 (the longest block plus two) | lmax 14 + 2 = 16 | MATCH | [code + record meta] explore_gpu/run.py make_model: ic.to_window(m, seed, spec.get('width') or task.lmax + 2); S75_ARMS WIDE_SPLIT has no width |
| 14 | tex6:458 | bound of 0.927 | not in the records | NOT RECORD-CHECKABLE | [session I note] pre-run note l.53 (explore_i_task.ideal_set_acc, 144-512 sequences); the draft quotes it correctly |
| 15 | tex6:458 | pre-run note committed after the perfect-gate and one-channel runs and before any learned-arm record had been pushed | results file at 5bc6c2e: 12 records, arms ['ORACLE', 'SINGLE']; first commit with a learned-arm record: b649751 (2026-10-12T13:53:09+00:00) | MATCH | [git] 5bc6c2e 2026-10-12T00:46:28+00:00; parent 86652b1 = 86652b1 |
| 16 | tex6:458 | (whether learned-arm runs had started on the card by then is not recorded) [changed: tex5 said three window-gate runs had finished] | no record carries a per-run start or end time; meta.started is written once (the first invocation's start); git gives commit times only | MATCH | [record + git] the fifth pass's fix, adopted. The timeline section below still shows what the records allow: under one uninterrupted invocation and a UTC clock, WIDE_SPLIT 832 ends 98 s before 5bc6c2e and 833 is running; without that premise (contradicted by 86652b1's per-GPU summary) 0-3 had finished. Session I's own explore_out/README.md at 9034b83 still says three had (row below) |
| 17 | tex6:458 | follow-up fixed: fresh seeds from 840; 32 blocks per source at L = 1536, bound 0.964; a bar from perfect-gate and one-channel quantities; a split that does not fire on a one-to-one map | note l.111-119: 840 and up; NB = 32, M = 352, L = 1536, bound 0.964; ORACLE/SINGLE inputs only; guard or spare | MATCH | [session I note] not record quantities; the S75b draft (1e31409/114cd08) gives that configuration's bound as 0.963 |
| 18 | tex6:458 | eta^2 by source below 0.003 at every evaluation of every gate layer in all twenty runs | WIDE_SPLIT max 0.00058 (seed 839, step 1200, L1); REG3 max 0.00225 (seed 832, step 1200, L3) | MATCH | [record] curve[i].routing[L].eta over 200 + 200 evaluations (REG3: 3 layers); the restricted field eta_pos12 reaches 0.0366 (REG3) / 0.0030 (WIDE_SPLIT), not the registered quantity |
| 19 | tex6:458 | no map was one-to-one at any evaluation | WIDE_SPLIT 0 one-to-one of 200 (evaluation x layer); REG3 0 one-to-one of 600 (evaluation x layer) | MATCH | [record] curve[i].routing[L].one_to_one; also 0 in the 64-sequence map_before of all 69 eligible checks |
| 20 | tex6:458 | window gate ended 0.529-0.536 | 0.5293-0.5365 -> 0.529-0.536 | MATCH | [record] end.set, ten seeds |
| 21 | tex6:458 | register gate ended 0.535-0.540 on nine seeds with one at 0.588 [changed: tex5 gave 0.535-0.540 for all ten] | nine seeds (without 839) 0.5352-0.5398; 839 0.5881 | MATCH | [record] the fifth pass's item, resolved |
| 22 | tex6:458 | 0.534-0.540 on nine seeds of one channel with one at 0.595 [', with one at 0.595' added] | SINGLE without 838: 0.5344-0.5403; 838 0.5954 | MATCH | [record]  |
| 23 | tex6:458 | 0.880-0.892 for the perfect gate | 0.8801-0.8915 | MATCH | [record]  |
| 24 | tex6:458 | the split fired its three permitted times in every window-gate run | splits field [3]; fired checks per run [3] | MATCH | [record] record splits, checks[].fired |
| 25 | tex6:458 | at the first three checks at which the trigger allowed a fire (7200, 12,000 and 16,800) in nine of ten | fires at 7200/12000/16800 on 9/10; seed 830 at [9600, 14400, 19200] (first eligible check 9600; at 7200 its probe rose 0.0206 >= 0.02) | MATCH | [record checks] the fires are always the earliest eligible checks the 4800 gap leaves |
| 26 | tex6:458 | the trigger's probe read 0.518-0.542 at the fires | 0.5185-0.5417 | MATCH | [record] checks[].acc (fired) |
| 27 | tex6:458 | threshold 0.88, set from the pilot's perfect gate | meta.split_thr 0.88 | MATCH | [record + code] docstring: C1's lowest ORACLE final SET 0.910 - 0.03 |
| 28 | tex6:458 | the mass over the four channels was uneven, 0.05-0.38 at the fires | checks[].masses 0.0517-0.3792; evaluation at the fire step 0.0509-0.3781 | MATCH | [record]  |
| 29 | tex6:458 | identical across sources to within 0.024 (at the fires) [changed: tex5 said 0.02] | largest per-channel difference between sources at the 30 fire-step evaluations 0.0235; 12/30 exceed 0.020, 0/30 exceed 0.024 | MATCH | [record] routing.L1.src_gate at the evaluation of the fire step (the checks hold only pooled masses); the fifth pass's item, resolved |
| 30 | tex6:458 | margin at most 0.04 | at the fires max 0.0294; any evaluation of any learned arm max 0.0360 | MATCH | [record] routing[L].margin |
| 31 | tex6:458 | the hazard (a split on a routed run overwrites a live channel) did not arise, no run having routed | fires on a one-to-one map_before: 0; runs ROUTED at the evaluation of a fire and not at the end: 0; evaluations ROUTED in any learned run: 0 | MATCH | [record]  |
| 32 | tex6:458 | one channel: on nine of ten seeds its accuracy rose by at most 0.033 after 4800 updates [changed: tex5 said 0.03] | nine seeds: end - SET(4800) 0.0119-0.0329; max after 4800 - SET(4800) 0.0159-0.0329; 838: 0.0911 | MATCH | [record curve[].set] both readings of 'rose' hold unrounded (largest 0.0329, seed 839); the fifth pass's item, resolved |
| 33 | tex6:458 | ...and a probe of the source from the residual stayed at chance (nine one-channel seeds) | nine SINGLE seeds, every layer and evaluation: max 0.2699; ORACLE (the chance band) 0.2367-0.2671 | MATCH | [record probe[L].acc] the nine-seed maximum (SINGLE 830, L3, 1200) is 0.003 above the ORACLE band; from 3600 on their maximum is 0.2687 |
| 34 | tex6:458 | on the tenth, as on one register seed, a late rise (0.09 and 0.08 after 4800) [changed: tex5 said 'of about 0.06'] | SET(end) - SET(4800): SINGLE 838 +0.0911, REG3 839 +0.0837 | MATCH | [record curve[].set] for reference, the excess over the other seeds' median end: SINGLE 838 +0.0573, REG3 839 +0.0494; the fifth pass's item, resolved |
| 35 | tex6:458 | ...took the probe no higher than 0.34 (0.29 and 0.28 at the end) [changed: tex5 said 'took the probe to 0.29 and 0.28'] | SINGLE 838: peak 0.3403 (L3, 8400), end max 0.2892; REG3 839: peak from 3600 0.2913 (3600), from 4800 0.2890 (7200), end max 0.2778 (its early hold: 0.3793 at 1200, 0.3339 at 2400) | PARTIAL | [record] the bound holds only after rounding: SINGLE 838's peak is 0.3403 > 0.34 (stated bounds are MATCH only unrounded, as everywhere in this check); 0.29 / 0.28 are the end values, as now stated. REG3 839's probe never returned to the chance band after 1200 (3600: 0.291, 7200: 0.289), so for it 'took the probe' describes a level held since the early transient more than a late rise; 'no higher than 0.34' holds for it from 2400 on |
| 36 | tex6:458 | probe from the residual entering layer 2 read 0.35-0.61 at 1200 updates in every gated run | WIDE_SPLIT 0.3971-0.5802; REG3 0.3524-0.6118; WIDE_SLOW 0.3966-0.5796 | MATCH | [record] combined 0.3524-0.6118; probe['L2'] is the residual entering layer 2 (explore_i_common.residuals) |
| 37 | tex6:458 | ...chance in the perfect-gate and one-channel runs (at 1200) | ORACLE 0.2542-0.2646; SINGLE 0.2496-0.2650 | MATCH | [record]  |
| 38 | tex6:458 | ...and all but one (a register run, at 0.29) had lost it by 3600 [changed: tex5 said all had] | WIDE_SPLIT max 0.2612; REG3 max 0.2913; WIDE_SLOW max 0.2646; above 0.27 at 3600: [('REG3', 839, 0.2913)] of 29 gated runs | MATCH | [record] the next highest at 3600 is REG3 832 at 0.2679, 0.0008 above the ORACLE band's top (0.2671) and below 0.27, so the count of one depends on the 0.27 line; the fifth pass's item, resolved |
| 39 | caller's paraphrase (check_S75) | chance in every arm from 3600 | ORACLE max 0.2671 (seed 831, 4800, final; evaluations > 0.27: 0); SINGLE max 0.3403 (seed 838, 8400, L3; evaluations > 0.27: 16); WIDE_SPLIT max 0.2676 (seed 831, 8400, final; evaluations > 0.27: 0); REG3 max 0.2913 (seed 839, 3600, L2; evaluations > 0.27: 17); WIDE_SLOW max 0.2647 (seed 831, 24000, final; evaluations > 0.27: 0) | PARTIAL | [record] holds except SINGLE 838 (above 0.27 at 16 of 18 evaluations from 3600, from 6000 on) and REG3 839 (above 0.27 at 17 of 18); tex6:458 now names both |
| 40 | tex6:458, tex6:464 | half of the perfect gate's shortfall at a block's first word | bin weights [0.0857, 0.0914, 0.1829, 0.64] (max residual 4.8e-08); bin-1 share of (0.927 - SET) with the note's bin-1 bound 0.944: 830 0.54, 831 0.42 | MATCH | [record + session I note] the two bounds are the note's (not records); set_1 830 0.7192, 831 0.7121 |
| 41 | tex6:458 | decay-faded ideal learner 0.680 at four sources against 0.840 at two | not in the records | NOT RECORD-CHECKABLE | [session I note] pre-run note l.55-56; quoted correctly |
| 42 | tex6:458 | WIDE_SLOW: nine seeds run so far (the tenth is queued) | WIDE_SLOW records at 9034b83 (I's head at 42c4557's commit time): seeds [830, 831, 832, 833, 834, 835, 836, 837, 838]; at the current tip 2fd4c7e: [830, 831, 832, 833, 834, 835, 836, 837, 838] | MATCH | [record + git] 839 has not landed (results file last changed at 4c3984a); 'queued' is not record-checkable (I's 4c3984a and 559b942 say 'pending') |
| 43 | tex6:458 | 0.530-0.536 against 0.529-0.536 | WIDE_SLOW 0.5300-0.5360; WIDE_SPLIT same nine seeds 0.5293-0.5365 | MATCH | [record]  |
| 44 | tex6:458 | WIDE_SLOW eta^2 below 0.001 at every evaluation, no one-to-one map at any | max eta 0.00057; one-to-one 0 of 180 | MATCH | [record] eta_pos12 reaches 0.0030 |
| 45 | tex6:458 | matched the split arm seed for seed (nine seeds) | ROUTED pairs (split, slow): [(False, False)] on 9 seeds; \|SET difference\| max 0.00325 (seed 831) | MATCH | [record]  |
| 46 | caller's paraphrase (check_S75) | WIDE_SLOW's nine seeds within 0.003 of WIDE_SPLIT (SET) | 830: -0.00057, 831: +0.00325, 832: +0.00064, 833: -0.00037, 834: +0.00049, 835: -0.00122, 836: -0.00112, 837: +0.00069, 838: -0.00006 | PARTIAL | [record] seed 831: 0.53274 vs 0.53598 = +0.00325 > 0.003 unrounded (0.533 vs 0.536 at three decimals); the draft's own text says only 'seed for seed'; report_2 l.76 now says it (row below) |
| 47 | tex6:478 | Next: the dense-loss task at four sources again, in session I's committed follow-up: a configuration piloted so that the perfect gate reaches its bar, fresh seeds, and a split that does not fire on a one-to-one map [changed: tex5 ended '; the window gate without the split on the present configuration first'] | stale item removed: present in tex5 True, in tex6 False; WIDE_SLOW 9/10 recorded at 9034b83, its tenth still pending and described at tex6:458 as queued; the follow-up as the pre-run note fixes it (l.111-119: fresh seeds from 840, a configuration where 0.9 is reachable, a split guard) and as S75b's driver (8cb8dce, WIP) implements it (a 2-seed ORACLE pilot with proceed/stop rules, a guarded trigger) | MATCH | [git + session I note] the fifth pass's item, resolved |
| 48 | tex6:516 | provenance: records 86652b1, b649751, 4c3984a; pre-run note 5bc6c2e; report 2 03bffdf, 559b942; commit 559b942; runs 'pilot; 41' [changed: tex5 stopped at 03bffdf, 'pilot; 32'] | the results file changed at ['4c3984a', 'b649751', '86652b1']; 41 S75 records at 9034b83; report_2.md changed at ['559b942', '03bffdf']; I's commits after 559b942 up to 9034b83: ['9034b83 S75b CHECKs: interim CPU output (partial runs; the full run follows)', '8cb8dce S75b driver (explore_gpu/s75b.py) and its CHECKs (s75b_checks.py), WIP: B1-B6 pass on CPU; B7-B8 running', '30807ec window_gate on the GTX 1660 SUPER: pre-run note (before the run), its description generator, and one thread per mainline worker on every device'] | MATCH | [git + record] the commits after 559b942 are window_gate's pre-run note and S75b's driver and CHECKs, not S75 material; the fifth pass's item, resolved |
| 49 | rule (s75_s4.py READING, report()) | VALID := both ORACLE runs end with SET >= 0.9; reading UNTESTED if not VALID; else 'carries' iff the better of WIDE_SPLIT and REG3 is ROUTED on >= 8/10 | VALID False (0.8915, 0.8801); done True; ROUTED {'WIDE_SPLIT': 0, 'REG3': 0}; verdict UNTESTED | MATCH | [record + code] summary prints 'READING (pre-fixed): UNTESTED'; had VALID held, 0/10 < 8/10 would read 'does not carry' |
| 50 | rule | ROUTED 0/10 [0.00, 0.28] NEVER (both read arms); ORACLE 2/2 [0.34, 1.00] RELIABLE | Wilson 0/10 [0.00, 0.28] NEVER; 2/2 [0.34, 1.00] RELIABLE | MATCH | [record] explore_i_common.wilson / band (z = 1.96) |
| 51 | rule (printed, not read) | S74 usability at S = 4: ORACLE mean 0.886 (>= 0.9: False); SINGLE <= ORACLE - 0.2 on 10/10 | ORACLE mean 0.8858; 10/10; smallest margin 0.2904 (838) | MATCH | [record] note l.23: 'smallest margin 0.29' |
| 52 | rule | readings, outcomes and constants fixed before the first run and unchanged since | READING / OUTCOMES text and 7 constants identical at 5d9dcf9 and 9034b83: True | MATCH | [git + record] 5d9dcf9 2026-10-10T20:31:55+00:00; first record: meta.started 2026-10-11 18:04:26 (machine-local clock; holds for any zone, the gap is ~21.5 h) |
| 53 | rule / provenance tex6:516 | code SHA 5fb120a92d69 (driver 5f03929), unchanged through the note and every record commit | records' sha ['5fb120a92d69']; recomputed: 5d9dcf9 cb0494b55a7f, ab65d9b c581e7d7076c, 9faa37b a39b6f08bbd5, 5f03929 5fb120a92d69, 86652b1 5fb120a92d69, 5bc6c2e 5fb120a92d69, b649751 5fb120a92d69, 03bffdf 5fb120a92d69, 9034b83 5fb120a92d69 | MATCH | [git + record] SHA-1 over meta.modules' sources as explore_gpu/run.code_sha does; first commit with that SHA 5f03929 (2026-10-10T22:26:44+00:00); the note's 'as committed in 5d9dcf9' holds for the rules, the code is 5f03929's |
| 54 | rule | pre-run note 5bc6c2e (00:46:28 UTC 12 Oct) committed before any WIDE_SPLIT/REG3 run | not establishable from the records (see the timeline) | NOT RECORD-CHECKABLE | [record + git] git establishes only 'before any learned-arm record was pushed' (first: b649751 13:53:09 UTC) |
| 55 | rule | descriptions generator (s75_writeup.py) and S75b draft committed blind (1e31409, parent 5bc6c2e) | 1e31409 parent 5bc6c2e; 1e31409 2026-10-12T14:16:14+00:00, b649751 2026-10-12T13:53:09+00:00 | MATCH | [git] parentage as stated; note that 1e31409 was committed 23 min after b649751 reached the branch (merged in 142f47e); what to describe had been fixed in 5bc6c2e |
| 56 | explore_out/I/gpu/s75_S4_prerun_note_2026-10-12.md:22 | no ORACLE evaluation ever reached 0.9 (bests 0.892 at 24k, 0.881 at 21.6k) | 830: 0.8915 at 24000, 831: 0.8805 at 21600 | MATCH | [record]  |
| 57 | explore_out/I/gpu/s75_S4_prerun_note_2026-10-12.md:23 | SINGLE 0.534-0.540 on 9 seeds, 0.595 on 838; <= ORACLE - 0.2 on 10/10 (smallest margin 0.29) | 0.5344-0.5403; 838 0.5954; 10/10; margin 0.2904 | MATCH | [record]  |
| 58 | explore_out/I/gpu/s75_S4_prerun_note_2026-10-12.md:59-61 | ORACLE ends 0.035 and 0.047 under the bound; bin 1 0.719 and 0.712; beats the faded learner by ~0.20 | 0.927 - SET: 0.0355, 0.0469; set_1 0.7192, 0.7121; mean - 0.680 = 0.206 | MATCH | [record + note] bounds and the faded learner are the note's constants |
| 59 | explore_out/I/gpu/s75_S4_prerun_note_2026-10-12.md:63-64 | with bin 1 at its bound seed 831 would reach about 0.900 | 831: 0.9000 | MATCH | [record + note] using the recomputed bin-1 weight and the note's bin-1 bound 0.944 |
| 60 | explore_out/I/gpu/s75_S4_prerun_note_2026-10-12.md:61-62 | a uniform loss of 0.017-0.036 on bins 2+ | per-bin bounds are not in the records | NOT RECORD-CHECKABLE | [note]  |
| 61 | explore_out/I/gpu/s75_S4_prerun_note_2026-10-12.md:73-77 | replay: 830 fires at 7200 (0.88); 831 at 4800, 9600, 14400 (0.88), 4800 (0.87), 7200 (0.86); none at 0.85; S = 2 ORACLEs (8 runs) never | S=4: {0.88: {830: [7200], 831: [4800, 9600, 14400]}, 0.87: {830: [], 831: [4800]}, 0.86: {830: [], 831: [7200]}, 0.85: {830: [], 831: []}}; S=2 fires per threshold {0.88: 0, 0.87: 0, 0.86: 0, 0.85: 0} | MATCH | [record (S75 + S74)] replayed on the 1024-sequence evaluation curves as the note did; S=2 at 12000 updates, checks through 9600 |
| 62 | explore_out/I/report_2.md:5-8 | table: median SET (range) ORACLE 0.886 (0.880, 0.892); SINGLE 0.538 (0.534-0.595); WIDE_SPLIT 0.534 (0.529-0.536); REG3 0.539 (0.535-0.588) | ORACLE 0.886 (0.880-0.892); SINGLE 0.538 (0.534-0.595); WIDE_SPLIT 0.534 (0.529-0.536); REG3 0.539 (0.535-0.588) | MATCH | [record]  |
| 63 | explore_out/I/report_2.md:5-8 | median GAP ORACLE 1.00, SINGLE 0, WIDE_SPLIT -0.01, REG3 0.00 | ORACLE +1.0001; SINGLE +0.0000; WIDE_SPLIT -0.0111; REG3 +0.0021 | MATCH | [record] GAP as explore_i_dense.gap, mean ORACLE |
| 64 | explore_out/I/report_2.md:7-8 | WIDE_SPLIT 30 splits (3 per run); eta^2 at the end 0.00 per gate layer | splits 30; end eta max WIDE_SPLIT 0.00033, REG3 0.00035 | MATCH | [record]  |
| 65 | explore_out/I/report_2.md:14, explore_out/I/report_2.md:51, explore_out/I/report_2.md:83 | the interference gap it divides by is 0.35; routing is worth 0.35 SET | mean ORACLE - SINGLE per seed 0.290-0.351; median 0.348 | MATCH | [record] 0.29 on seed 838 |
| 66 | explore_out/I/report_2.md:18 | all four registered arms on the 1660, R = 1, code SHA 5fb120a92d69 | hw ['NVIDIA GeForce GTX 1660 SUPER']; R [1] | MATCH | [record]  |
| 67 | explore_out/I/report_2.md:24 | regenerating the summary reproduces the pushed copy exactly | not run (session code not executed) | NOT RECORD-CHECKABLE | [record] the summary's numbers that report_2 quotes are recomputed in these rows; s75_S4_summary.md and the per-GPU summary at 4c3984a are byte-identical and omit WIDE_SLOW (both written before 4c3984a) |
| 68 | explore_out/I/report_2.md:41-42 | in all 20 learned runs, at each of the 20 evaluations, eta^2 0.00 at every gate layer; no map one-to-one; no run ROUTED at any point | max eta 0.00225; one-to-one 0; evaluations per run [20] | MATCH | [record]  |
| 69 | explore_out/I/report_2.md:43-45 | WIDE_SPLIT 830 ended with gate mass 0.16 / 0.29 / 0.38 / 0.16 | 0.1637 / 0.2929 / 0.3817 / 0.1617 | MATCH | [record] end.routing.L1.mass |
| 70 | explore_out/I/report_2.md:45 | rows of the source x channel table differ by at most 0.02 in any run | at the end max 0.0230; at any evaluation max 0.0415 | PARTIAL | [record] holds at the end after rounding to two decimals; unrounded 0.023, and 0.042 at some evaluation (REG3) |
| 71 | explore_out/I/report_2.md:45 | the median margin is 0.03 | WIDE_SPLIT 0.0254; REG3 0.0227 | PARTIAL | [record] 0.03 for WIDE_SPLIT; REG3's is 0.02 (the summary prints both) |
| 72 | explore_out/I/report_2.md:46 | REG3's third-layer gate puts about 0.7 on one channel, identically for every source | end L3 largest channel mass per seed 0.301-0.948 (median 0.664) | PARTIAL | [record] the median is about 0.7, the range is 0.30-0.95; per-source maxima within 0.020 |
| 73 | explore_out/I/report_2.md:48-49 | WIDE_SPLIT 0.529-0.536; REG3 0.535-0.540 on nine seeds; 839 0.588 (GAP 0.14) | 0.529-0.536; 0.535-0.540; 0.5881 (GAP 0.1431) | MATCH | [record]  |
| 74 | explore_out/I/report_2.md:50 | seed 838's negative GAP (-0.21 and -0.19) comes from SINGLE 838 (0.595) | WIDE_SPLIT -0.2060, REG3 -0.1939; SINGLE 838 0.5954 | MATCH | [record]  |
| 75 | explore_out/I/report_2.md:52-53 | the source probe reads chance (0.25 to 0.28) at every layer in every learned run, as in SINGLE | at the end 0.247-0.278; at 1200 the L2 probe reads 0.352-0.612 | PARTIAL | [record] true of the end only; at 1200 every learned run's residual carried the source (per-run max over L2/L3/final 0.352-0.612) |
| 76 | explore_out/I/report_2.md:55-57 | beta high at SRC (median 0.80, range 0.44-0.90), low at words (median 0.15) | src median 0.8032, range 0.4350-0.9018; word median 0.1514 | MATCH | [record] record beta_src / beta_word |
| 77 | explore_out/I/report_2.md:59 | 30 fires, 3 per run, at 7200, 12,000 and 16,800 (seed 830: 9600, 14,400, 19,200) | 30 fires; 9/10 at 7200/12000/16800; 830 [9600, 14400, 19200] | MATCH | [record]  |
| 78 | explore_out/I/report_2.md:60-61 | every fire on a map that was not one-to-one, on_target true, at probe SET 0.518-0.542 | one-to-one map_before 0/30; on_target 30/30; 0.518-0.542 | MATCH | [record]  |
| 79 | explore_out/I/report_2.md:62 | fires on a one-to-one map 0; runs ROUTED before a fire and not at the end 0 | 0; 0 | MATCH | [record]  |
| 80 | explore_out/I/report_2.md:64 | by the next check the map had merged again | one-to-one at the check after a fire: 0/30 | MATCH | [record] checks[i+1].map_before (the last fire of each run has a following check) |
| 81 | explore_out/I/report_2.md:87-88 | S75b design commit 1e31409 has parent 5bc6c2e, which holds only the ORACLE and SINGLE records | parent 5bc6c2e; records at 5bc6c2e: ['ORACLE', 'SINGLE'] | MATCH | [git]  |
| 82 | explore_out/I/report_2.md:9 | table row (559b942): WIDE_SLOW 830-838 (839 pending); 0/9 [0.00, 0.30] NEVER; median SET 0.534 (0.530-0.536); eta^2 at the end 0.00; no split | seeds [830, 831, 832, 833, 834, 835, 836, 837, 838]; ROUTED 0/9, Wilson [0.00, 0.30] NEVER; median 0.5340 (0.5300-0.5360); end eta max 0.00029; splits [None] | MATCH | [record]  |
| 83 | explore_out/I/report_2.md:22 | WIDE_SLOW is running on the 1660 now | 9/10 WIDE_SLOW records at 9034b83 (since 4c3984a); 839 pending | PARTIAL | [record + git] stale since 4c3984a; 559b942 updated the table row and section 6 but left this line (section 6 says 839 is pending) |
| 84 | explore_out/I/report_2.md:68-69 | section 6: seeds 830-838 (4c3984a; seed 839 is pending) | WIDE_SLOW seeds at 9034b83 [830, 831, 832, 833, 834, 835, 836, 837, 838]; results file last changed at 4c3984a | MATCH | [record + git]  |
| 85 | explore_out/I/report_2.md:70 | ROUTED 0/9 at the end, and never at any of the 20 evaluations of any run | end 0/9; evaluations ROUTED 0 of 180 ([20] per run) | MATCH | [record]  |
| 86 | explore_out/I/report_2.md:72 | eta^2 0.00; the rows of the source x channel table differ by at most 0.02 | eta max any evaluation 0.00057; row spread at the end max 0.0219, at any evaluation 0.0297 | PARTIAL | [record] holds at the end after rounding to two decimals only (0.0219); 0.0297 at some evaluation; the same pattern as section 1's WIDE_SPLIT statement (row above) |
| 87 | explore_out/I/report_2.md:72 | the margin is 0.01-0.03 | end margins 0.0102-0.0300; any evaluation 0.0091-0.0324 | MATCH | [record] at the end (as section 1's 'median margin'); over every evaluation the range is wider |
| 88 | explore_out/I/report_2.md:74 | SET is 0.530-0.536, SINGLE's level, and the source probe is at chance (0.24-0.26) at every layer | SET 0.5300-0.5360; probe at the end 0.2430-0.2647; from 3600 0.2410-0.2647; at 1200 (L2) 0.3966-0.5796 | PARTIAL | [record] true at the end (and from 3600); at 1200 every WIDE_SLOW run's residual carried the source, as in the other gated arms (tex6:458 says so) -- the same omission as report_2's section 3 (row above) |
| 89 | explore_out/I/report_2.md:76 | paired by seed with WIDE_SPLIT: neither arm ROUTED on any of the 9 shared seeds; per seed SET differs by at most 0.003 (median absolute difference 0.0006) | ROUTED pairs [(False, False)]; \|difference\| max 0.00325 (seed 831), median 0.00064 | PARTIAL | [record] holds after rounding only: seed 831 differs by 0.00325 (0.53274 vs 0.53598, 0.533 vs 0.536 at three decimals); the median reproduces |
| 90 | explore_out/I/report_2.md:77 | as a description: the window gate stayed source-blind with the split and without it; the 30 fires neither helped nor harmed routing that never formed | ROUTED at any evaluation: WIDE_SPLIT 0, WIDE_SLOW 0 | MATCH | [record] descriptive, labelled so |
| 91 | explore_out/README.md:42 | Correction to the pre-run note: three WIDE_SPLIT runs had already finished on the 1660 when it was committed | no per-run time in the records (timeline section) | NOT RECORD-CHECKABLE | [session I] session I's index at 9034b83 (unchanged by 559b942) still states report_2's reconstruction as fact; the draft (tex6:458) no longer does |
| 92 | explore_out/I/report_2.md:29-31 | from the durations and the start time (18:04 on 10-11), WIDE_SPLIT 830-832 ran 23:11 (10-11) to 00:44 (10-12); the note was committed at 00:46 UTC | under one uninterrupted invocation: WIDE_SPLIT 830 starts 23:11:35, 832 ends 00:44:50; 5bc6c2e 00:46:28 | PARTIAL | [record + git] the arithmetic reproduces (record secs, meta.started); the premise of one uninterrupted invocation is not in the records and is contradicted by 86652b1's per-GPU summary (see the timeline section) |
| 93 | explore_out/I/report_2.md:31 | the other seven WIDE_SPLIT runs and all ten REG3 runs came after it | under the same premises WIDE_SPLIT 833 started 00:44:50, 98.0 s before the note's commit | DISAGREE | [record + git] by report_2's own reconstruction a fourth run (833) was in progress at 00:46:28; the statement should read 'three had finished and a fourth had started' |
| 94 | explore_out/I/report_2.md:32-33 | the 1660's timestamps can only be UTC (ORACLE and SINGLE committed 23:42 UTC, so they ended before) | SINGLE 839 ends 23:11:35 machine time; 86652b1 authored 2026-10-11T23:42:23+00:00 | PARTIAL | [record + git] the commit bounds the clock only from one side: machine time - UTC >= -31 min; a clock ahead of UTC is not excluded (UTC is plausible: meta.started 18:04:26 is 31 s before 203a5ad's 18:04:57 UTC) |
| 95 | explore_out/I/report_2.md:34 | no learned-arm record was available until b649751 (13:53 UTC on 10-12) | first commit with a learned-arm record: b649751 2026-10-12T13:53:09+00:00 | MATCH | [git]  |
| 96 | explore_out/I/report_2.md:20-21 | deviation: the note assigned WIDE_SPLIT and REG3 to the RTX 3070; the 1660 ran them instead | not in the records | NOT RECORD-CHECKABLE | [report_2] inconsistent with l.29-31: runs that started at 23:11, before the note, were not run 'instead' of the note's assignment; one of the two accounts is wrong |
| 97 | explore_out/I/gpu/s75_S4_descriptions.md:9-48 | section 1: per-run ROUTED trajectories (WIDE_SPLIT 10, REG3 10, WIDE_SLOW 9 runs x 20 evaluations), end mark, end eta^2 and each block's 'k/n ROUTED at the end' | 29 lines reproduce, 0 differ []; per block (lines, header reproduces): {'WIDE_SPLIT': [10, True], 'REG3': [10, True], 'WIDE_SLOW': [9, True]} | MATCH | [record] regenerated at 559b942 (file history to 9034b83: ['559b942', '03bffdf']); the WIDE_SLOW block is new |
| 98 | explore_out/I/gpu/s75_S4_descriptions.md:37 | WIDE_SLOW: 0/9 ROUTED at the end [was 'no records yet' at 03bffdf] | 9 WIDE_SLOW records at 9034b83, ROUTED 0 | MATCH | [record + git] check_S75's stale-WIDE_SLOW PARTIAL, resolved by 559b942 |
| 99 | explore_out/I/gpu/s75_S4_descriptions.md:56-124 | section 2: every eligible check of the split trigger (seed, step, probe SET (prev), map before, one-to-one, on_target, fired / blocked (gap / cap), routed@step / next / end) | 69 rows reproduce of 69 eligible checks in the records; differing [] | MATCH | [record] status recomputed independently (cap: 3 earlier fires; gap: < 4800 since the last) and compared with blocked_by |
| 100 | explore_out/I/gpu/s75_S4_descriptions.md:128-130 | fires 30 over 10 runs; on a one-to-one map 0; routed before a fire and not at the end 0 []; runs with at least one fire 10/10 | 30; 0; 0; 10/10 | MATCH | [record]  |
| 101 | explore_out/I/gpu/s75_S4_descriptions.md:136-145 | section 4: WIDE_SPLIT's columns (routed@end, fires, eta^2, SET) and WIDE_SLOW's (routed@end, eta^2, SET) per seed; 839 pending | 10/10 rows reproduce (9 with WIDE_SLOW values, pending [839]); differing [] | MATCH | [record] WIDE_SLOW's three columns new at 559b942; SET at three decimals, so 831 reads 0.533 vs 0.536 (difference 0.00325) |
| 102 | explore_out/I/gpu/s75_S4_descriptions.md:147 | paired seeds with both records 9; ROUTED in both 0, WIDE_SPLIT only 0, WIDE_SLOW only 0, neither 9; pending for 1 of 10 seeds | 9 pairs; both 0, split only 0, slow only 0, neither 9; pending 1 | MATCH | [record]  |
| 103 | explore_out/I/gpu/s75_S4_descriptions.md:3 | generated under code SHA 5fb120a92d69; the registered result is UNTESTED | 5fb120a92d69; UNTESTED | MATCH | [record]  |

## Timeline (unchanged from check_S75; printed for the tex6:458 parenthesis)

```
machine clock of meta.started, ONE UNINTERRUPTED INVOCATION ASSUMED (a premise git contradicts); record secs:
  record order is the arms' order: True
  ORACLE     830 10-11 18:04:26 -> 10-11 18:31:38 ( 1632.4 s)
  ORACLE     831 10-11 18:31:38 -> 10-11 18:59:04 ( 1646.3 s)
  SINGLE     830 10-11 18:59:04 -> 10-11 19:24:13 ( 1508.4 s)
  SINGLE     831 10-11 19:24:13 -> 10-11 19:49:22 ( 1509.7 s)
  SINGLE     832 10-11 19:49:22 -> 10-11 20:14:32 ( 1509.7 s)
  SINGLE     833 10-11 20:14:32 -> 10-11 20:39:51 ( 1519.3 s)
  SINGLE     834 10-11 20:39:51 -> 10-11 21:05:09 ( 1517.2 s)
  SINGLE     835 10-11 21:05:09 -> 10-11 21:30:27 ( 1517.9 s)
  SINGLE     836 10-11 21:30:27 -> 10-11 21:55:43 ( 1516.1 s)
  SINGLE     837 10-11 21:55:43 -> 10-11 22:21:00 ( 1516.9 s)
  SINGLE     838 10-11 22:21:00 -> 10-11 22:46:16 ( 1516.6 s)
  SINGLE     839 10-11 22:46:16 -> 10-11 23:11:35 ( 1519.2 s)
  WIDE_SPLIT 830 10-11 23:11:35 -> 10-11 23:42:39 ( 1863.6 s)
  WIDE_SPLIT 831 10-11 23:42:39 -> 10-12 00:13:45 ( 1866.5 s)
  WIDE_SPLIT 832 10-12 00:13:45 -> 10-12 00:44:50 ( 1864.1 s)
  WIDE_SPLIT 833 10-12 00:44:50 -> 10-12 01:15:56 ( 1866.6 s) <- straddles 5bc6c2e
  WIDE_SPLIT 834 10-12 01:15:56 -> 10-12 01:47:01 ( 1865.2 s)
  WIDE_SPLIT 835 10-12 01:47:01 -> 10-12 02:18:06 ( 1865.1 s)
  WIDE_SPLIT 836 10-12 02:18:06 -> 10-12 02:49:17 ( 1870.7 s)
  WIDE_SPLIT 837 10-12 02:49:17 -> 10-12 03:20:42 ( 1885.2 s)
  WIDE_SPLIT 838 10-12 03:20:42 -> 10-12 03:52:06 ( 1883.4 s)
  WIDE_SPLIT 839 10-12 03:52:06 -> 10-12 04:23:11 ( 1865.4 s)
  REG3       830 10-12 04:23:11 -> 10-12 05:03:06 ( 2395.4 s)
  REG3       831 10-12 05:03:06 -> 10-12 05:43:16 ( 2409.6 s)
  REG3       832 10-12 05:43:16 -> 10-12 06:23:18 ( 2402.4 s)
  REG3       833 10-12 06:23:18 -> 10-12 07:03:18 ( 2399.6 s)
  REG3       834 10-12 07:03:18 -> 10-12 07:43:10 ( 2392.1 s)
  REG3       835 10-12 07:43:10 -> 10-12 08:23:08 ( 2397.7 s)
  REG3       836 10-12 08:23:08 -> 10-12 09:02:53 ( 2384.6 s)
  REG3       837 10-12 09:02:53 -> 10-12 09:42:57 ( 2404.7 s)
  REG3       838 10-12 09:42:57 -> 10-12 10:23:02 ( 2404.8 s)
  REG3       839 10-12 10:23:02 -> 10-12 11:02:52 ( 2389.7 s)
  WIDE_SLOW  830 10-12 11:02:52 -> 10-12 11:33:37 ( 1844.9 s)
  WIDE_SLOW  831 10-12 11:33:37 -> 10-12 12:04:25 ( 1848.7 s)
  WIDE_SLOW  832 10-12 12:04:25 -> 10-12 12:35:29 ( 1863.4 s)
  WIDE_SLOW  833 10-12 12:35:29 -> 10-12 13:06:31 ( 1862.1 s)
  WIDE_SLOW  834 10-12 13:06:31 -> 10-12 13:37:44 ( 1872.9 s)
  WIDE_SLOW  835 10-12 13:37:44 -> 10-12 14:08:36 ( 1852.2 s)
  WIDE_SLOW  836 10-12 14:08:36 -> 10-12 14:39:28 ( 1852.1 s)
  WIDE_SLOW  837 10-12 14:39:28 -> 10-12 15:10:22 ( 1853.9 s)
  WIDE_SLOW  838 10-12 15:10:22 -> 10-12 15:41:15 ( 1853.2 s)
  WIDE_SPLIT + REG3 wall time 42676 s = 11.85 h; b649751 at 2026-10-12 13:53:09; latest possible start of WIDE_SPLIT 830 if contiguous to REG3 839: 10-12 02:01:52
  WIDE_SLOW 830-838 wall time 16703 s; 4c3984a at 2026-10-12 19:12:59; latest start 14:34:35; the uninterrupted reconstruction would end 838 at 10-12 15:41:15 and 839 by 16:12:15, yet 839 was pending at 19:12, still at 9034b83 (22:53) and at 42c4557 (23:14): WIDE_SLOW ran in a later invocation, so gaps between invocations are real and unrecorded
  meta.task_check.secs at 86652b1 / b649751 / 4c3984a: 1.0 / 1.0 / 0.9
  86652b1 per-GPU summary rows: | WIDE_SPLIT | - | | | | | | |; | REG3 | - | | | | | | |; READING (pre-fixed): incomplete  (ROUTED {'WIDE_SPLIT': 0, 'REG3': 0}; the better arm needs >= 8/10)
CHANCE BAND (ORACLE probe, all layers and evaluations): 0.2367-0.2671
```

Under one uninterrupted invocation and a UTC clock, WIDE_SPLIT 832 ends 98 s before `5bc6c2e` and 833 is running; without that premise the records allow 0-3 finished WIDE_SPLIT runs at the note's commit. tex6:458's 'whether learned-arm runs had started on the card by then is not recorded' is what the records support.

## What could not be checked

- WIDE_SLOW seed 839: not recorded on any ref at 42c4557's time (nor at the current tip); 'queued' (tex6:458) and 'pending' (session I) are not record-checkable.
- Per-run start and end times of S75 (none in the records); the 1660's clock zone.
- The bounds 0.927 / 0.964 / 0.963, the faded learner 0.680 / 0.840 and the per-bin bounds: session I's note, not records.
- Regenerating session I's summary (session code not run).
