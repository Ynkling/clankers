# Check: S75 at four sources (session I, GTX 1660 SUPER) against the Revision 9 draft and session I's write-up

Referee V. Generated rows: `docs/review_2026-10/rev9/check_S75.py` (run `python3 -I docs/review_2026-10/rev9/check_S75.py`
from the repository root after `git fetch`; it reads every file with `git show`, runs no session code, and prints the
table and the timeline below).

**Pins.** Records: `explore_out/I/gpu/s75_S4_NVIDIA_GeForce_GTX_1660_SUPER_results.json` at `origin/claude/explore-I`
`4c3984a` (41 records, all code SHA `5fb120a92d69`: ORACLE 830-831, SINGLE / WIDE_SPLIT / REG3 830-839, WIDE_SLOW
830-838), and the same file at `86652b1`, `5bc6c2e`, `b649751`. Session I's files: `explore_out/I/report_2.md` (`03bffdf`,
unchanged at `4c3984a`), `gpu/s75_S4_prerun_note_2026-10-12.md` (`5bc6c2e`, unchanged), `gpu/s75_S4_descriptions.md`
(`03bffdf`, unchanged), `gpu/s75_S4_summary.md` and the per-GPU summary (`b649751`), the S75 driver
`explore_gpu/s75_s4.py` (last changed `5f03929`) with `explore_i_dense.py` / `explore_i_common.py` / `explore_gpu/run.py`,
the descriptions generator `explore_gpu/s75_writeup.py` (`1e31409`), and S74's `explore_out/I/s74_pilot_results.json`
(identical at `a2588da` and `4c3984a`). Draft: `claude/rev9-draft` at `c5e4508`, `multichannel_hebbian_report_v9.tex`
(= `tex5`, identical to the scratchpad copy `v9e.tex`); passages tex5:36 (abstract), tex5:61 (index row), tex5:456 (A
dense loss), tex5:458 (Four sources (S75)), tex5:462 and tex5:464 (limits), tex5:472 (conclusion), tex5:478 (next steps),
tex5:516 (provenance row); draft history `aab7f7c`, `771f4ab`, `18c5fbc`, `c5e4508`.

**Definitions** (from the code at the pins). eta^2 = `routing[L]["eta"]` (the registered ROUTED quantity,
`explore_i_dense.routed`); ROUTED = eta^2 >= 0.9 and `one_to_one` at every gate layer, at the end. The probe "from the
residual entering layer l" = `probe["L<l>"]["acc"]` (`explore_i_common.residuals`: `feats["L<l>"]` is the residual
entering layer l). GAP = (SET(arm) - SET(SINGLE, same seed)) / (mean ORACLE SET - SET(SINGLE, same seed)). Chance for the
probe: 0.25 (four sources); the empirical band is the ORACLE runs' range over every layer and evaluation,
0.2367-0.2671 (their stacks have no reason to carry the source), and "above chance" below means > 0.27.
A stated bound ("at most", "within", "below") is MATCH only if the unrounded value satisfies it; when it holds only after
rounding to the stated decimals the row is PARTIAL and the note gives the exact value.

## Counts

91 rows: **MATCH 67, PARTIAL 16, DISAGREE 1, NOT RECORD-CHECKABLE 7.**
By source: the draft passages (rows 1-47, of which rows 38 and 45 check two quantities the caller attributed to the
draft's authors), the S75 rule (48-54), the pre-run note (55-60), report_2.md (61-85), the descriptions (86-91).

### Every non-MATCH row

| # | where | statement | recomputed | status |
|---|---|---|---|---|
| 16 | tex5:458 | "three window-gate runs had by then finished on the card" | records hold no per-run time; reconstruction needs two unrecorded premises, one contradicted by git (see Timeline); 0-3 possible | NOT RECORD-CHECKABLE |
| 21 | tex5:458 | register gate ended 0.535-0.540 | ten seeds 0.5352-0.5881; 0.535-0.540 only on nine (REG3 839 = 0.5881), and the sentence does not say "nine" | PARTIAL |
| 29 | tex5:458 | gate rows identical across sources "to within 0.02" at the fires | max per-channel difference between sources 0.0235; 12 of 30 fire-step evaluations exceed 0.020 | PARTIAL |
| 32 | tex5:458 | one channel rose "by at most 0.03 after 4800" on nine seeds | end - SET(4800): 0.0119-0.0329 (830 0.0301, 831 0.0302, 839 0.0329 exceed 0.030); max-after-4800 rise up to 0.0329 | PARTIAL |
| 34 | tex5:458 | "a late rise of about 0.06 took the probe to 0.29 and 0.28" (SINGLE 838, REG3 839) | rise after 4800 0.091 / 0.084 (0.057 / 0.049 above the other seeds' median); 0.289 / 0.278 are end values; SINGLE 838 peaked 0.340 (L3, 8400); REG3 839 was above chance from 1200 on (0.291 at 3600, 0.289 at 7200), so the rise did not take it there | PARTIAL |
| 37 | tex5:458 | the residual "had lost [the source] by 3600" (every gated run) | at 3600: WIDE_SPLIT max 0.2612, WIDE_SLOW 0.2646, REG3 0.2913 (seed 839); 29 of 30 gated runs at chance | PARTIAL |
| 38 | caller | chance in every arm from 3600 | ORACLE <= 0.2671, WIDE_SPLIT <= 0.2676, WIDE_SLOW <= 0.2647; SINGLE 838 up to 0.3403 (above 0.27 at 16 of 18 evaluations, from 6000); REG3 839 up to 0.2913 (17 of 18) | PARTIAL |
| 45 | caller | WIDE_SLOW within 0.003 of WIDE_SPLIT on nine seeds | seed 831: 0.53274 vs 0.53598 = +0.00325; the other eight within 0.00122 (the draft text itself says only "seed for seed", row 44 MATCH) | PARTIAL |
| 46 | tex5:478 | Next: "the window gate without the split ... first" | 9 of 10 WIDE_SLOW runs already recorded (4c3984a) and described at tex5:458 | PARTIAL (stale) |
| 47 | tex5:516 | provenance row: records 86652b1, b649751; note 5bc6c2e; report 2 03bffdf; commit 03bffdf; runs "pilot; 32" | tex5:458 also uses the 9 WIDE_SLOW records of 4c3984a (19:12:59 UTC), not listed; commit column and run count stop before them | PARTIAL |
| 14 | tex5:458 | bound 0.927 | not a record (pre-run note l.53); quoted correctly | NOT RECORD-CHECKABLE |
| 40 | tex5:458 | faded learner 0.680 vs 0.840 | not a record (note l.55-56); quoted correctly | NOT RECORD-CHECKABLE |
| 53 | rule | note 5bc6c2e committed before any WIDE_SPLIT/REG3 run | not establishable (Timeline); git shows only "before any learned-arm record was pushed" | NOT RECORD-CHECKABLE |
| 59 | note:61-62 | uniform loss 0.017-0.036 on bins 2+ | per-bin bounds not in the records | NOT RECORD-CHECKABLE |
| 66 | report_2:24 | regenerating the summary reproduces the pushed copy | session code not run; its quoted numbers recomputed (rows 61-65); the summary omits WIDE_SLOW | NOT RECORD-CHECKABLE |
| 69 | report_2:45 | source x channel rows differ by at most 0.02 in any run | at the end 0.0230; at some evaluation 0.0415 (REG3) | PARTIAL |
| 70 | report_2:45 | the median margin is 0.03 | WIDE_SPLIT 0.0254, REG3 0.0227 (0.02) | PARTIAL |
| 71 | report_2:46 | REG3's third-layer gate puts about 0.7 on one channel | end L3 largest channel mass 0.301-0.948 per seed, median 0.664 | PARTIAL |
| 74 | report_2:52-53 | probe chance (0.25-0.28) at every layer in every learned run | true at the end (0.247-0.278); at 1200 every learned run read 0.352-0.612 (max over L2/L3/final) | PARTIAL |
| 81 | report_2:29-31 | WIDE_SPLIT 830-832 ran 23:11 to 00:44 | arithmetic reproduces (23:11:35 - 00:44:50) under one uninterrupted invocation; that premise is not recorded and git contradicts it | PARTIAL |
| 82 | report_2:31 | "the other seven WIDE_SPLIT runs and all ten REG3 runs came after it" | under report_2's own premises WIDE_SPLIT 833 started 00:44:50, 98 s before the commit (00:46:28) | **DISAGREE** |
| 83 | report_2:32-33 | the 1660's timestamps "can only be UTC" | the 23:42 UTC commit bounds the clock from one side only (machine - UTC >= -31 min); a clock ahead of UTC is not excluded | PARTIAL |
| 85 | report_2:20-21 | the 1660 ran WIDE_SPLIT/REG3 "instead" of the 3070 (a deviation from the note) | not in the records; inconsistent with l.29-31 (runs that began at 23:11, before the note, were not a deviation from it) | NOT RECORD-CHECKABLE |
| 87 | descriptions:37, :124-135 | WIDE_SLOW "no records yet" / "pending" 10 of 10 | 9 records at 4c3984a; the file was not regenerated | PARTIAL (stale) |

## The quantities the draft's authors computed themselves (item 1)

| quantity (tex5:458) | record field | reproduces? |
|---|---|---|
| eta^2 below 0.003 at every evaluation of every gate layer (20 runs) | `curve[i].routing[L].eta` | yes: max 0.00058 (WIDE_SPLIT 839, 1200), 0.00225 (REG3 832, L3, 1200). The restricted `eta_pos12` reaches 0.0366 (REG3), not the registered quantity |
| no one-to-one map at any evaluation | `curve[i].routing[L].one_to_one` | yes: 0 of 200 (WIDE_SPLIT) and 0 of 600 (REG3) evaluation x layer; also 0 in the 69 eligible checks' `map_before` |
| probe from the residual entering layer 2 at 1200: 0.35-0.61 gated, chance in ORACLE and SINGLE | `curve[0].probe.L2.acc` (step 1200) | yes: WIDE_SPLIT 0.3971-0.5802, REG3 0.3524-0.6118, WIDE_SLOW 0.3966-0.5796; ORACLE 0.2542-0.2646, SINGLE 0.2496-0.2650 |
| chance in every arm from 3600 ("had lost it by 3600") | `curve[i].probe[L].acc` | partly: holds for ORACLE, WIDE_SPLIT, WIDE_SLOW and 9 + 9 seeds of SINGLE / REG3; not for REG3 839 (0.291 at 3600) or SINGLE 838 (0.340 at 8400) |
| SINGLE's rise of at most 0.03 after 4800 on nine seeds | `curve[i].set` | after rounding only: 0.0119-0.0329 |
| the fires' masses 0.05-0.38 | `checks[].masses` (fired), `routing.L1.mass` at the fire step | yes: 0.0517-0.3792 (checks), 0.0509-0.3781 (evaluations) |
| ...identical across sources to within 0.02 | `routing.L1.src_gate` at the fire step (the checks hold no per-source table) | after rounding only: 0.0235 max, 12/30 above 0.020 |
| ...margin at most 0.04 | `routing[L].margin` | yes: 0.0294 at the fires, 0.0360 at any evaluation of any learned arm |
| WIDE_SLOW's nine seeds within 0.003 of WIDE_SPLIT | `end.set` paired by seed | 8 of 9; seed 831 differs by 0.00325. ROUTED matches seed for seed (0 and 0 on all nine) |

Nothing in this list was impossible to recompute: every quantity is a record field.

## The S75 rule (item 2)

- **Inputs and verdict reproduce.** VALID needs both ORACLE runs at end SET >= 0.9: 0.8915 (830), 0.8801 (831), so not
  VALID; both read arms complete (10 + 10, ORACLE 2); ROUTED WIDE_SPLIT 0/10, REG3 0/10 (Wilson [0.00, 0.28], NEVER);
  the rule gives **UNTESTED**, as the summary prints. Had VALID held, 0/10 < 8/10 would have read "does not carry".
  Printed beside it: ORACLE mean 0.8858; SINGLE <= ORACLE - 0.2 on 10/10, smallest margin 0.2904 (seed 838).
- **Order.** The READING and OUTCOMES text and the seven constants (ITERS, SPLIT_THR, SEEDS, ORACLE_SEEDS, READ_ARMS,
  EXTENDED, HI) are identical at `5d9dcf9` (2026-10-10 20:31:55 UTC) and `4c3984a`. The code SHA in every record,
  5fb120a92d69, recomputed here as `explore_gpu/run.code_sha` defines it (SHA-1 over the sources of `meta.modules`),
  first appears at `5f03929` (2026-10-10 22:26:44 UTC) and is unchanged at `86652b1`, `5bc6c2e`, `b649751`, `03bffdf` and
  `4c3984a`; the earlier commits `5d9dcf9`, `ab65d9b`, `9faa37b` hash differently (work in progress). The first run
  started at `meta.started` 2026-10-11 18:04:26 (machine clock): 19.6 h after `5f03929` on a UTC clock, and after it under any time zone (offsets span -12 to +14 h). So the
  rule and the code were committed before the records they read. (The note's "stay as committed in 5d9dcf9" is right for
  the rules; the code the records ran is `5f03929`'s.)
- **The note before the learned arms.** Git establishes that `5bc6c2e` (00:46:28 UTC 12 Oct) was committed when the
  branch held only the 12 ORACLE/SINGLE records, and that the first learned-arm record arrived in `b649751` (13:53:09
  UTC). Whether any WIDE_SPLIT/REG3 *run* preceded the commit is not in the records (next section).
- **The note's declared hazard** (replay of the trigger on the ORACLE curves) reproduces exactly: at 0.88, 830 fires at
  7200 and 831 at 4800 / 9600 / 14400; at 0.87, 831 at 4800; at 0.86, 831 at 7200; none at 0.85; the eight S = 2
  ORACLE runs never fire (row 60).
- The descriptions generator and the S75b draft (`1e31409`) have parent `5bc6c2e`, as stated; `1e31409` itself was
  committed 23 minutes after `b649751` reached the branch (14:16:14 vs 13:53:09 UTC). What the descriptions contain had
  been fixed in `5bc6c2e`.

## Timeline (item 3: report_2.md's correction, the draft's parenthesis)

**What the records hold.** No record has a start or end time. They hold `meta.started` = "2026-10-11 18:04:26"
(`time.strftime`, the machine's local clock, no zone; written once by `setdefault`, so it is the first invocation's start
only) and per record `secs` (wall time of the `run_batched_icmc` call) and `train_secs`. The order of records in the file
is the order they were saved, and it is the arms' order (ORACLE, SINGLE, WIDE_SPLIT, REG3, WIDE_SLOW).

**report_2's reconstruction reproduces arithmetically.** Chaining `secs` from 18:04:26 with no gaps:

| run | start | end |
|---|---|---|
| SINGLE 839 | 22:46:16 | 23:11:35 |
| WIDE_SPLIT 830 | 23:11:35 | 23:42:39 |
| WIDE_SPLIT 831 | 23:42:39 | 00:13:45 |
| WIDE_SPLIT 832 | 00:13:45 | 00:44:50 |
| WIDE_SPLIT 833 | **00:44:50** | 01:15:56 |
| ... REG3 839 | 10:23:02 | 11:02:52 |

(the full table is printed by the script). Note commit `5bc6c2e`: 00:46:28 UTC.

**But it rests on two premises that are not in the records.**
1. *One uninterrupted invocation from ORACLE 830 to REG3 839.* The git history argues against it. `86652b1` (23:42 UTC)
   contains `s75_S4_NVIDIA_GeForce_GTX_1660_SUPER_summary.md`, and in the driver at the code SHA only `drive()` writes that
   per-GPU file, and only after its loop over the arms ends. That copy lists WIDE_SPLIT and REG3 as "-" and the reading
   as "incomplete". An invocation running all four arms could not have written it at 23:11. So the ORACLE/SINGLE
   invocation ended after SINGLE 839 (it was restricted to those arms), unless someone called `report(summary=...)` by
   hand. WIDE_SPLIT/REG3 then ran in a later invocation whose start is not recorded (`meta.started` is not updated, and
   `task_check.secs` reads 1.0 in both files). The WIDE_SLOW invocation shows that such gaps are real: chained without
   gaps, WIDE_SLOW 839 would have ended by 16:12, yet it was still pending at `4c3984a` (19:12:59).
   report_2's own l.20-21 ("the 1660 ran them instead", a deviation from the note's assignment to the 3070) also implies
   that the learned arms started after the note was read, which contradicts its l.29-31.
2. *A UTC machine clock.* The argument at report_2 l.32-33 bounds the clock from one side only: SINGLE 839 ends at
   23:11:35 machine time, before `86652b1`'s 23:42:23 UTC, so the clock is at most 31 minutes behind UTC. A clock
   ahead of UTC is not excluded, and under premise 1 it would mean more runs before the note, not fewer. UTC is
   plausible: `meta.started` is 31 s before `203a5ad` (the CUDA check commit, 18:04:57 UTC). The user's laptop
   commit `13de308` carries -06:00, which this bound rules out for the 1660's clock.

**What the records do allow.** If the WIDE_SPLIT/REG3 invocation ran its 20 runs back to back (42,676 s) and
`b649751` was pushed when they finished, it started between 23:11:35 (10-11) and 02:01:52 (10-12) on a UTC clock. So 0, 1,
2 or 3 WIDE_SPLIT runs had finished at 00:46:28, possibly with one more in progress.

**Verdict.**
- *report_2 l.29-31* ("WIDE_SPLIT 830, 831 and 832 ran from 23:11 to 00:44"): PARTIAL. The arithmetic is right
  (23:11:35 to 00:44:50). It is stated as fact but is a reconstruction under an unstated no-gap premise that git
  contradicts.
- *report_2 l.31* ("the other seven WIDE_SPLIT runs and all ten REG3 runs came after it"): **DISAGREE** on its own
  premises. WIDE_SPLIT 833 would have started at 00:44:50, 98 s before the commit, so the correct wording would be
  "three had finished and a fourth had begun". The 98 s margin is also smaller than the unrecorded per-boundary
  overheads summed over 15 runs plus start-up, so even "three finished" is not secure under those premises.
- *The draft* (tex5:458, from `18c5fbc`: "three window-gate runs had by then finished on the card, unseen by the
  session"; `18c5fbc`'s commit message says "three window-gate runs preceded it"): NOT RECORD-CHECKABLE. It repeats
  report_2's reconstruction as a fact. The defensible wording is: the note was committed before any learned-arm record
  was pushed (git); whether learned-arm runs had already started on the card is not recorded, and a reconstruction that
  assumes one uninterrupted invocation and a UTC clock puts up to three finished and a fourth running.
- *"before any learned-arm record had been pushed"* and *"no learned-arm record was available until b649751"*: MATCH (git).

## The descriptions (s75_S4_descriptions.md, 03bffdf)

- Section 1: all 20 trajectory lines (20 evaluations each), the end marks and the end eta^2 strings reproduce from
  `curve[i].routing` (row 86).
- Section 2: all 69 eligible checks (`checks[].eligible`) reproduce: probe SET and previous value, `map_before`,
  one-to-one, `on_target`, and fired / blocked (gap) / blocked (cap), which I recomputed independently (cap = three
  earlier fires, gap = under 4800 since the last fire) and which agree with `blocked_by`. routed@step, @next and @end also
  reproduce. No row is missing or extra (row 88).
- Section 3 counts (30 fires; 0 on a one-to-one map; 0 runs routed before a fire and not at the end; 10/10 runs with a
  fire) reproduce (row 89).
- Section 4: WIDE_SPLIT's four columns reproduce for all ten seeds (row 90). The WIDE_SLOW side ("no records yet",
  "pending" for 10 of 10) is stale at `4c3984a`, which holds nine WIDE_SLOW records, and the file was not regenerated
  (row 87). Recomputed here instead: WIDE_SLOW 830-838 are ROUTED on none (same as WIDE_SPLIT, so the pairs are 9
  neither, 0 discordant). End eta^2 is 0.00 on all nine, and SET is 0.5300-0.5360.

## What could not be checked, and why

- Every per-run start or end time, and so whether any learned-arm run preceded `5bc6c2e`: the records hold only
  durations and one machine-local start stamp (Timeline).
- The Bayes bounds (0.927 at S = 4; the bin-1 bound 0.944; 0.964 / 0.963 for NB = 32, L = 1536) and the decay-faded
  learner (0.680 vs 0.840). These are Monte-Carlo constants of `explore_i_task`, not records, and computing them would
  mean running session code. The draft and report quote the note's values correctly. The bin weights, by contrast, are
  fixed by the fixed evaluation set and were recovered exactly from the records (0.0857 / 0.0914 / 0.1829 / 0.6400,
  residual 5e-8). With them, the note's "about half of the shortfall in bin 1" (0.54 and 0.42) and "831 would reach about
  0.900" (0.9000) reproduce, given its bounds.
- The note's ORACLE projections (48k / 72k / 96k) and the per-bin "uniform loss 0.017-0.036 on bins 2+": these are
  fits and bounds, not records.
- report_2 l.24 (regenerating the summary reproduces the pushed copy): it needs session I's code run. Its numbers that
  report_2 quotes were recomputed instead. Note that `s75_S4_summary.md` and the per-GPU summary at `4c3984a` are
  byte-identical and do not include WIDE_SLOW.
- "the tenth is queued" (tex5:458): only the absence of WIDE_SLOW 839 is recordable; `4c3984a`'s message says
  "pending".

## Rows

Counts: 91 rows; MATCH 67, PARTIAL 16, DISAGREE 1, NOT RECORD-CHECKABLE 7

| # | line or file:line | statement | recomputed | status | note |
|---|---|---|---|---|---|
| 1 | tex5:36, tex5:61, tex5:458 | perfect gate ended at 0.880 and 0.892 (bar 0.9) | ORACLE 830 0.8915, 831 0.8801 | MATCH | [record] end.set; both < 0.9 -> not VALID |
| 2 | tex5:61, tex5:36, tex5:462, tex5:472 | window gate with split and register gate route 0/10 each; neither learned gate routed (ten seeds each) | ROUTED WIDE_SPLIT 0/10, REG3 0/10 | MATCH | [record] end.routing[L].eta >= 0.9 and one_to_one at every gate layer (explore_i_dense.routed) |
| 3 | tex5:61, tex5:36, tex5:462, tex5:472 | one channel learns to carry the source on none of ten (at four sources) | SINGLE probe max over layers: end 0.257-0.289; any evaluation 0.261-0.340 (838: 0.340) | MATCH | [record probe[L].acc] descriptive; against 0.96-0.98 at two sources (S74, below); the largest value, SINGLE 838's 0.340 (L3, 8400), is far from carrying |
| 4 | tex5:456 | pilot: perfect gate, one channel, single-source control at two sources, 12,000 updates, four seeds | iters [12000]; arms ['C1_ORACLE', 'C1_SINGLE', 'C2_NOMIX', 'C2_ORACLE', 'C2_SINGLE']; seeds [800, 801, 802, 803] | MATCH | [S74 record] C1 has no NOMIX record (stopped after the decision, report_1 l.18) |
| 5 | tex5:456 | three of four runs of the shorter-block configuration (C2: blocks of 3-7 words) | C2 SINGLE runs whose L3 probe reaches >= 0.9: 3/4 ([800, 801, 803]) | MATCH | [S74 record]  |
| 6 | tex5:456 | a step transition raised the set accuracy by 0.09-0.22 | 0.0872-0.2169 (800: +0.1080, 801: +0.0872, 803: +0.2169) | MATCH | [S74 record] transition = the evaluation with the largest one-evaluation rise of the L3 probe (as recompute_I) |
| 7 | tex5:456 | probe from the residual entering layer 3 from 0.54-0.58 to 0.96-0.98 | before 0.5377-0.5797; at the transition 0.9568-0.9803 | MATCH | [S74 record] probe['L3'].acc |
| 8 | tex5:456 | closing 44-83% of each seed's gap to the perfect gate | 800: 55%, 801: 44%, 803: 83% | MATCH | [S74 record] 1 - (ORACLE end - SINGLE end) / (ORACLE - SINGLE at the evaluation before the transition), same seed; report_1 l.32 says 60-80% (another definition; recompute_I row 7) |
| 9 | tex5:456 | the perfect gate's own stack never carried the source | S74 ORACLE probe max 0.5263 | MATCH | [S74 record] every layer, every evaluation, C1 and C2 |
| 10 | tex5:456 | four sources need sequences of about 770 tokens; S = 4, L = 768; 24,000 updates | config {'S': 4, 'V': 4, 'NB': 16, 'lmin': 6, 'lmax': 14, 'M': 176}; iters 24000; L = S*(NB+M) = 768 | MATCH | [record] meta.config, meta.iters; every record iters = {24000} |
| 11 | tex5:456 | S75: 32 runs, about 16 hours of training on that card | 32 registered runs; train_secs 16.34 h, wall secs 16.97 h | MATCH | [record] ORACLE 2 + SINGLE 10 + WIDE_SPLIT 10 + REG3 10; all R = {1} |
| 12 | tex5:456 | perfect gate on two seeds; both must end at a set accuracy of at least 0.9; ten seeds each | ORACLE seeds [830, 831]; others SINGLE 830-839 (n=10); WIDE_SPLIT 830-839 (n=10); REG3 830-839 (n=10) | MATCH | [record + code] VALID rule: s75_s4.py docstring OUTCOMES and report() (valid = both ORACLE end set >= 0.9) |
| 13 | tex5:456 | window gate at width 16 (the longest block plus two) | lmax 14 + 2 = 16 | MATCH | [code + record meta] explore_gpu/run.py make_model: ic.to_window(m, seed, spec.get('width') or task.lmax + 2); S75_ARMS WIDE_SPLIT has no width |
| 14 | tex5:458 | bound of 0.927 | not in the records | NOT RECORD-CHECKABLE | [session I note] pre-run note l.53 (explore_i_task.ideal_set_acc, 144-512 sequences); the draft quotes it correctly |
| 15 | tex5:458 | pre-run note committed after the perfect-gate and one-channel runs and before any learned-arm record had been pushed | results file at 5bc6c2e: 12 records, arms ['ORACLE', 'SINGLE']; first commit with a learned-arm record: b649751 (2026-10-12T13:53:09+00:00) | MATCH | [git] 5bc6c2e 2026-10-12T00:46:28+00:00; parent 86652b1 = 86652b1 |
| 16 | tex5:458 | (three window-gate runs had by then finished on the card, unseen by the session) | no per-run timestamp in the records; see the timeline section | NOT RECORD-CHECKABLE | [record + git] a reconstruction from meta.started + secs that assumes (a) one uninterrupted invocation from ORACLE 830 to REG3 839 and (b) a UTC clock; under them WIDE_SPLIT 832 ends 00:44:50, 98 s before 5bc6c2e (00:46:28), and 833 is already running; (a) is contradicted by 86652b1's per-GPU summary (written by drive() only when an invocation ends, it lists no WIDE_SPLIT/REG3 record); without (a) the records allow 0-3 finished runs |
| 17 | tex5:458 | follow-up fixed: fresh seeds from 840; 32 blocks per source at L = 1536, bound 0.964; a bar from perfect-gate and one-channel quantities; a split that does not fire on a one-to-one map | note l.111-119: 840 and up; NB = 32, M = 352, L = 1536, bound 0.964; ORACLE/SINGLE inputs only; guard or spare | MATCH | [session I note] not record quantities; the S75b draft (1e31409/114cd08) gives that configuration's bound as 0.963 |
| 18 | tex5:458 | eta^2 by source below 0.003 at every evaluation of every gate layer in all twenty runs | WIDE_SPLIT max 0.00058 (seed 839, step 1200, L1); REG3 max 0.00225 (seed 832, step 1200, L3) | MATCH | [record] curve[i].routing[L].eta over 200 + 200 evaluations (REG3: 3 layers); the restricted field eta_pos12 reaches 0.0366 (REG3) / 0.0030 (WIDE_SPLIT), not the registered quantity |
| 19 | tex5:458 | no map was one-to-one at any evaluation | WIDE_SPLIT 0 one-to-one of 200 (evaluation x layer); REG3 0 one-to-one of 600 (evaluation x layer) | MATCH | [record] curve[i].routing[L].one_to_one; also 0 in the 64-sequence map_before of all 69 eligible checks |
| 20 | tex5:458 | window gate ended 0.529-0.536 | 0.5293-0.5365 -> 0.529-0.536 | MATCH | [record] end.set, ten seeds |
| 21 | tex5:458 | register gate ended 0.535-0.540 | ten seeds 0.5352-0.5881; nine seeds (without 839) 0.5352-0.5398 | PARTIAL | [record] holds on nine seeds; REG3 839 ended 0.5881 (the sentence does not say 'nine'; 839 is mentioned later only as 'one register seed') |
| 22 | tex5:458 | 0.534-0.540 on nine seeds of one channel | SINGLE without 838: 0.5344-0.5403 | MATCH | [record] 838 ended 0.5954 |
| 23 | tex5:458 | 0.880-0.892 for the perfect gate | 0.8801-0.8915 | MATCH | [record]  |
| 24 | tex5:458 | the split fired its three permitted times in every window-gate run | splits field [3]; fired checks per run [3] | MATCH | [record] record splits, checks[].fired |
| 25 | tex5:458 | at the first three checks at which the trigger allowed a fire (7200, 12,000 and 16,800) in nine of ten | fires at 7200/12000/16800 on 9/10; seed 830 at [9600, 14400, 19200] (first eligible check 9600; at 7200 its probe rose 0.0206 >= 0.02) | MATCH | [record checks] the fires are always the earliest eligible checks the 4800 gap leaves |
| 26 | tex5:458 | the trigger's probe read 0.518-0.542 at the fires | 0.5185-0.5417 | MATCH | [record] checks[].acc (fired) |
| 27 | tex5:458 | threshold 0.88, set from the pilot's perfect gate | meta.split_thr 0.88 | MATCH | [record + code] docstring: C1's lowest ORACLE final SET 0.910 - 0.03 |
| 28 | tex5:458 | the mass over the four channels was uneven, 0.05-0.38 at the fires | checks[].masses 0.0517-0.3792; evaluation at the fire step 0.0509-0.3781 | MATCH | [record]  |
| 29 | tex5:458 | identical across sources to within 0.02 (at the fires) | largest per-channel difference between sources at the 30 fire-step evaluations 0.0235; 12/30 exceed 0.020 | PARTIAL | [record] routing.L1.src_gate at the evaluation of the fire step (the checks hold only pooled masses); holds after rounding to two decimals only |
| 30 | tex5:458 | margin at most 0.04 | at the fires max 0.0294; any evaluation of any learned arm max 0.0360 | MATCH | [record] routing[L].margin |
| 31 | tex5:458 | the hazard (a split on a routed run overwrites a live channel) did not arise, no run having routed | fires on a one-to-one map_before: 0; runs ROUTED at the evaluation of a fire and not at the end: 0; evaluations ROUTED in any learned run: 0 | MATCH | [record]  |
| 32 | tex5:458 | one channel: on nine of ten seeds its accuracy rose by at most 0.03 after 4800 updates | nine seeds: end - SET(4800) 0.0119-0.0329; max after 4800 - SET(4800) 0.0159-0.0329; 838: 0.0911 | PARTIAL | [record curve[].set] holds at two decimals; unrounded, end-rise exceeds 0.030 on 830 (0.0301), 831 (0.0302), 839 (0.0329); the max-after-4800 rise on 830 (0.0301), 831 (0.0319), 834 (0.0305), 839 (0.0329) |
| 33 | tex5:458 | ...and a probe of the source from the residual stayed at chance (nine one-channel seeds) | nine SINGLE seeds, every layer and evaluation: max 0.2699; ORACLE (the chance band) 0.2367-0.2671 | MATCH | [record probe[L].acc] the nine-seed maximum (SINGLE 830, L3, 1200) is 0.003 above the ORACLE band; from 3600 on their maximum is 0.2687 |
| 34 | tex5:458 | on the tenth, as on one register seed, a late rise of about 0.06 took the probe to 0.29 and 0.28 | SINGLE 838: SET 4800->end +0.0911, end 0.5954 vs nine-seed median 0.5381 (+0.0573); end probe max 0.2892 (final), peak 0.3403 (L3, 8400). REG3 839: SET 4800->end +0.0837, end 0.5881 vs median of the other nine 0.5387; end probe max 0.2778 (L3) | PARTIAL | [record] 0.29 / 0.28 are the end values (max over layers); 'about 0.06' is the excess over the other seeds (0.057 / 0.049), the rise after 4800 is 0.091 / 0.084; SINGLE 838's probe peaked at 0.340 at 8400 before settling; REG3 839's probe never returned to the chance band after 1200 (3600: 0.291, 7200: 0.289), so the rise did not 'take' it to 0.28 |
| 35 | tex5:458 | probe from the residual entering layer 2 read 0.35-0.61 at 1200 updates in every gated run | WIDE_SPLIT 0.3971-0.5802; REG3 0.3524-0.6118; WIDE_SLOW 0.3966-0.5796 | MATCH | [record] combined 0.3524-0.6118; probe['L2'] is the residual entering layer 2 (explore_i_common.residuals) |
| 36 | tex5:458 | ...chance in the perfect-gate and one-channel runs (at 1200) | ORACLE 0.2542-0.2646; SINGLE 0.2496-0.2650 | MATCH | [record]  |
| 37 | tex5:458 | ...and had lost it by 3600 | WIDE_SPLIT max 0.2612; REG3 max 0.2913; WIDE_SLOW max 0.2646 | PARTIAL | [record] above the ORACLE band (> 0.27) at 3600: [('REG3', 839, 0.2913)]; 29 of 30 gated runs (19 of 20 registered) at chance |
| 38 | caller's paraphrase | chance in every arm from 3600 | ORACLE max 0.2671 (seed 831, 4800, final; evaluations > 0.27: 0); SINGLE max 0.3403 (seed 838, 8400, L3; evaluations > 0.27: 16); WIDE_SPLIT max 0.2676 (seed 831, 8400, final; evaluations > 0.27: 0); REG3 max 0.2913 (seed 839, 3600, L2; evaluations > 0.27: 17); WIDE_SLOW max 0.2647 (seed 831, 24000, final; evaluations > 0.27: 0) | PARTIAL | [record] holds except SINGLE 838 (above 0.27 at 16 of 18 evaluations from 3600, from 6000 on) and REG3 839 (above 0.27 at 17 of 18); the draft itself names those two seeds, but with their end values only |
| 39 | tex5:458, tex5:464 | half of the perfect gate's shortfall at a block's first word | bin weights [0.0857, 0.0914, 0.1829, 0.64] (max residual 4.8e-08); bin-1 share of (0.927 - SET) with the note's bin-1 bound 0.944: 830 0.54, 831 0.42 | MATCH | [record + session I note] the two bounds are the note's (not records); set_1 830 0.7192, 831 0.7121 |
| 40 | tex5:458 | decay-faded ideal learner 0.680 at four sources against 0.840 at two | not in the records | NOT RECORD-CHECKABLE | [session I note] pre-run note l.55-56; quoted correctly |
| 41 | tex5:458 | WIDE_SLOW: nine seeds run so far (the tenth is queued) | WIDE_SLOW records at 4c3984a: seeds [830, 831, 832, 833, 834, 835, 836, 837, 838] | MATCH | [record + git] 839 absent; 'queued' is not record-checkable (4c3984a's message says 'pending') |
| 42 | tex5:458 | 0.530-0.536 against 0.529-0.536 | WIDE_SLOW 0.5300-0.5360; WIDE_SPLIT same nine seeds 0.5293-0.5365 | MATCH | [record]  |
| 43 | tex5:458 | WIDE_SLOW eta^2 below 0.001 at every evaluation, no one-to-one map at any | max eta 0.00057; one-to-one 0 of 180 | MATCH | [record] eta_pos12 reaches 0.0030 |
| 44 | tex5:458 | matched the split arm seed for seed (nine seeds) | ROUTED pairs (split, slow): [(False, False)] on 9 seeds; \|SET difference\| max 0.00325 (seed 831) | MATCH | [record]  |
| 45 | caller's paraphrase | WIDE_SLOW's nine seeds within 0.003 of WIDE_SPLIT (SET) | 830: -0.00057, 831: +0.00325, 832: +0.00064, 833: -0.00037, 834: +0.00049, 835: -0.00122, 836: -0.00112, 837: +0.00069, 838: -0.00006 | PARTIAL | [record] seed 831: 0.53274 vs 0.53598 = +0.00325 > 0.003 unrounded (0.533 vs 0.536 at three decimals); the draft's own text says only 'seed for seed' |
| 46 | tex5:478 | Next: ... the window gate without the split on the present configuration first | 9/10 WIDE_SLOW runs recorded at 4c3984a and described in tex5:458 | PARTIAL | [record + git] stale at c5e4508: only seed 839 remains |
| 47 | tex5:516 | provenance: records 86652b1, b649751; pre-run note 5bc6c2e; report 2 03bffdf; commit 03bffdf; runs 'pilot; 32' | tex5:458 also reads the WIDE_SLOW records of 4c3984a (9 runs; 2026-10-12T19:12:59+00:00) | PARTIAL | [git] 4c3984a is not listed, the commit column stops at 03bffdf and the count omits the 9 WIDE_SLOW runs; the other commits exist with the stated content (driver 5f03929 = the code SHA's tree, below) |
| 48 | rule (s75_s4.py READING, report()) | VALID := both ORACLE runs end with SET >= 0.9; reading UNTESTED if not VALID; else 'carries' iff the better of WIDE_SPLIT and REG3 is ROUTED on >= 8/10 | VALID False (0.8915, 0.8801); done True; ROUTED {'WIDE_SPLIT': 0, 'REG3': 0}; verdict UNTESTED | MATCH | [record + code] summary prints 'READING (pre-fixed): UNTESTED'; had VALID held, 0/10 < 8/10 would read 'does not carry' |
| 49 | rule | ROUTED 0/10 [0.00, 0.28] NEVER (both read arms); ORACLE 2/2 [0.34, 1.00] RELIABLE | Wilson 0/10 [0.00, 0.28] NEVER; 2/2 [0.34, 1.00] RELIABLE | MATCH | [record] explore_i_common.wilson / band (z = 1.96) |
| 50 | rule (printed, not read) | S74 usability at S = 4: ORACLE mean 0.886 (>= 0.9: False); SINGLE <= ORACLE - 0.2 on 10/10 | ORACLE mean 0.8858; 10/10; smallest margin 0.2904 (838) | MATCH | [record] note l.23: 'smallest margin 0.29' |
| 51 | rule | readings, outcomes and constants fixed before the first run and unchanged since | READING / OUTCOMES text and 7 constants identical at 5d9dcf9 and 4c3984a: True | MATCH | [git + record] 5d9dcf9 2026-10-10T20:31:55+00:00; first record: meta.started 2026-10-11 18:04:26 (machine-local clock; holds for any zone, the gap is ~21.5 h) |
| 52 | rule / provenance tex5:516 | code SHA 5fb120a92d69 (driver 5f03929), unchanged through the note and every record commit | records' sha ['5fb120a92d69']; recomputed: 5d9dcf9 cb0494b55a7f, ab65d9b c581e7d7076c, 9faa37b a39b6f08bbd5, 5f03929 5fb120a92d69, 86652b1 5fb120a92d69, 5bc6c2e 5fb120a92d69, b649751 5fb120a92d69, 03bffdf 5fb120a92d69, 4c3984a 5fb120a92d69 | MATCH | [git + record] SHA-1 over meta.modules' sources as explore_gpu/run.code_sha does; first commit with that SHA 5f03929 (2026-10-10T22:26:44+00:00); the note's 'as committed in 5d9dcf9' holds for the rules, the code is 5f03929's |
| 53 | rule | pre-run note 5bc6c2e (00:46:28 UTC 12 Oct) committed before any WIDE_SPLIT/REG3 run | not establishable from the records (see the timeline) | NOT RECORD-CHECKABLE | [record + git] git establishes only 'before any learned-arm record was pushed' (first: b649751 13:53:09 UTC) |
| 54 | rule | descriptions generator (s75_writeup.py) and S75b draft committed blind (1e31409, parent 5bc6c2e) | 1e31409 parent 5bc6c2e; 1e31409 2026-10-12T14:16:14+00:00, b649751 2026-10-12T13:53:09+00:00 | MATCH | [git] parentage as stated; note that 1e31409 was committed 23 min after b649751 reached the branch (merged in 142f47e); what to describe had been fixed in 5bc6c2e |
| 55 | explore_out/I/gpu/s75_S4_prerun_note_2026-10-12.md:22 | no ORACLE evaluation ever reached 0.9 (bests 0.892 at 24k, 0.881 at 21.6k) | 830: 0.8915 at 24000, 831: 0.8805 at 21600 | MATCH | [record]  |
| 56 | explore_out/I/gpu/s75_S4_prerun_note_2026-10-12.md:23 | SINGLE 0.534-0.540 on 9 seeds, 0.595 on 838; <= ORACLE - 0.2 on 10/10 (smallest margin 0.29) | 0.5344-0.5403; 838 0.5954; 10/10; margin 0.2904 | MATCH | [record]  |
| 57 | explore_out/I/gpu/s75_S4_prerun_note_2026-10-12.md:59-61 | ORACLE ends 0.035 and 0.047 under the bound; bin 1 0.719 and 0.712; beats the faded learner by ~0.20 | 0.927 - SET: 0.0355, 0.0469; set_1 0.7192, 0.7121; mean - 0.680 = 0.206 | MATCH | [record + note] bounds and the faded learner are the note's constants |
| 58 | explore_out/I/gpu/s75_S4_prerun_note_2026-10-12.md:63-64 | with bin 1 at its bound seed 831 would reach about 0.900 | 831: 0.9000 | MATCH | [record + note] using the recomputed bin-1 weight and the note's bin-1 bound 0.944 |
| 59 | explore_out/I/gpu/s75_S4_prerun_note_2026-10-12.md:61-62 | a uniform loss of 0.017-0.036 on bins 2+ | per-bin bounds are not in the records | NOT RECORD-CHECKABLE | [note]  |
| 60 | explore_out/I/gpu/s75_S4_prerun_note_2026-10-12.md:73-77 | replay: 830 fires at 7200 (0.88); 831 at 4800, 9600, 14400 (0.88), 4800 (0.87), 7200 (0.86); none at 0.85; S = 2 ORACLEs (8 runs) never | S=4: {0.88: {830: [7200], 831: [4800, 9600, 14400]}, 0.87: {830: [], 831: [4800]}, 0.86: {830: [], 831: [7200]}, 0.85: {830: [], 831: []}}; S=2 fires per threshold {0.88: 0, 0.87: 0, 0.86: 0, 0.85: 0} | MATCH | [record (S75 + S74)] replayed on the 1024-sequence evaluation curves as the note did; S=2 at 12000 updates, checks through 9600 |
| 61 | explore_out/I/report_2.md:5-8 | table: median SET (range) ORACLE 0.886 (0.880, 0.892); SINGLE 0.538 (0.534-0.595); WIDE_SPLIT 0.534 (0.529-0.536); REG3 0.539 (0.535-0.588) | ORACLE 0.886 (0.880-0.892); SINGLE 0.538 (0.534-0.595); WIDE_SPLIT 0.534 (0.529-0.536); REG3 0.539 (0.535-0.588) | MATCH | [record]  |
| 62 | explore_out/I/report_2.md:5-8 | median GAP ORACLE 1.00, SINGLE 0, WIDE_SPLIT -0.01, REG3 0.00 | ORACLE +1.0001; SINGLE +0.0000; WIDE_SPLIT -0.0111; REG3 +0.0021 | MATCH | [record] GAP as explore_i_dense.gap, mean ORACLE |
| 63 | explore_out/I/report_2.md:7-8 | WIDE_SPLIT 30 splits (3 per run); eta^2 at the end 0.00 per gate layer | splits 30; end eta max WIDE_SPLIT 0.00033, REG3 0.00035 | MATCH | [record]  |
| 64 | explore_out/I/report_2.md:14, :51, :75 | the interference gap it divides by is 0.35; routing is worth 0.35 SET | mean ORACLE - SINGLE per seed 0.290-0.351; median 0.348 | MATCH | [record] 0.29 on seed 838 |
| 65 | explore_out/I/report_2.md:18 | all four registered arms on the 1660, R = 1, code SHA 5fb120a92d69 | hw ['NVIDIA GeForce GTX 1660 SUPER']; R [1] | MATCH | [record]  |
| 66 | explore_out/I/report_2.md:24 | regenerating the summary reproduces the pushed copy exactly | not run (session code not executed) | NOT RECORD-CHECKABLE | [record] the summary's numbers that report_2 quotes are recomputed in these rows; s75_S4_summary.md and the per-GPU summary at 4c3984a are byte-identical and omit WIDE_SLOW (both written before 4c3984a) |
| 67 | explore_out/I/report_2.md:41-42 | in all 20 learned runs, at each of the 20 evaluations, eta^2 0.00 at every gate layer; no map one-to-one; no run ROUTED at any point | max eta 0.00225; one-to-one 0; evaluations per run [20] | MATCH | [record]  |
| 68 | explore_out/I/report_2.md:43-45 | WIDE_SPLIT 830 ended with gate mass 0.16 / 0.29 / 0.38 / 0.16 | 0.1637 / 0.2929 / 0.3817 / 0.1617 | MATCH | [record] end.routing.L1.mass |
| 69 | explore_out/I/report_2.md:45 | rows of the source x channel table differ by at most 0.02 in any run | at the end max 0.0230; at any evaluation max 0.0415 | PARTIAL | [record] holds at the end after rounding to two decimals; unrounded 0.023, and 0.042 at some evaluation (REG3) |
| 70 | explore_out/I/report_2.md:45 | the median margin is 0.03 | WIDE_SPLIT 0.0254; REG3 0.0227 | PARTIAL | [record] 0.03 for WIDE_SPLIT; REG3's is 0.02 (the summary prints both) |
| 71 | explore_out/I/report_2.md:46 | REG3's third-layer gate puts about 0.7 on one channel, identically for every source | end L3 largest channel mass per seed 0.301-0.948 (median 0.664) | PARTIAL | [record] the median is about 0.7, the range is 0.30-0.95; per-source maxima within 0.020 |
| 72 | explore_out/I/report_2.md:48-49 | WIDE_SPLIT 0.529-0.536; REG3 0.535-0.540 on nine seeds; 839 0.588 (GAP 0.14) | 0.529-0.536; 0.535-0.540; 0.5881 (GAP 0.1431) | MATCH | [record]  |
| 73 | explore_out/I/report_2.md:50 | seed 838's negative GAP (-0.21 and -0.19) comes from SINGLE 838 (0.595) | WIDE_SPLIT -0.2060, REG3 -0.1939; SINGLE 838 0.5954 | MATCH | [record]  |
| 74 | explore_out/I/report_2.md:52-53 | the source probe reads chance (0.25 to 0.28) at every layer in every learned run, as in SINGLE | at the end 0.247-0.278; at 1200 the L2 probe reads 0.352-0.612 | PARTIAL | [record] true of the end only; at 1200 every learned run's residual carried the source (per-run max over L2/L3/final 0.352-0.612) |
| 75 | explore_out/I/report_2.md:55-57 | beta high at SRC (median 0.80, range 0.44-0.90), low at words (median 0.15) | src median 0.8032, range 0.4350-0.9018; word median 0.1514 | MATCH | [record] record beta_src / beta_word |
| 76 | explore_out/I/report_2.md:59 | 30 fires, 3 per run, at 7200, 12,000 and 16,800 (seed 830: 9600, 14,400, 19,200) | 30 fires; 9/10 at 7200/12000/16800; 830 [9600, 14400, 19200] | MATCH | [record]  |
| 77 | explore_out/I/report_2.md:60-61 | every fire on a map that was not one-to-one, on_target true, at probe SET 0.518-0.542 | one-to-one map_before 0/30; on_target 30/30; 0.518-0.542 | MATCH | [record]  |
| 78 | explore_out/I/report_2.md:62 | fires on a one-to-one map 0; runs ROUTED before a fire and not at the end 0 | 0; 0 | MATCH | [record]  |
| 79 | explore_out/I/report_2.md:64-65 | by the next check the map had merged again | one-to-one at the check after a fire: 0/30 | MATCH | [record] checks[i+1].map_before (the last fire of each run has a following check) |
| 80 | explore_out/I/report_2.md:78-80 | S75b design commit 1e31409 has parent 5bc6c2e, which holds only the ORACLE and SINGLE records | parent 5bc6c2e; records at 5bc6c2e: ['ORACLE', 'SINGLE'] | MATCH | [git]  |
| 81 | explore_out/I/report_2.md:29-31 | from the durations and the start time (18:04 on 10-11), WIDE_SPLIT 830-832 ran 23:11 (10-11) to 00:44 (10-12); the note was committed at 00:46 UTC | under one uninterrupted invocation: WIDE_SPLIT 830 starts 23:11:35, 832 ends 00:44:50; 5bc6c2e 00:46:28 | PARTIAL | [record + git] the arithmetic reproduces (record secs, meta.started); the premise of one uninterrupted invocation is not in the records and is contradicted by 86652b1's per-GPU summary (see the timeline section) |
| 82 | explore_out/I/report_2.md:31 | the other seven WIDE_SPLIT runs and all ten REG3 runs came after it | under the same premises WIDE_SPLIT 833 started 00:44:50, 98.0 s before the note's commit | DISAGREE | [record + git] by report_2's own reconstruction a fourth run (833) was in progress at 00:46:28; the statement should read 'three had finished and a fourth had started' |
| 83 | explore_out/I/report_2.md:32-33 | the 1660's timestamps can only be UTC (ORACLE and SINGLE committed 23:42 UTC, so they ended before) | SINGLE 839 ends 23:11:35 machine time; 86652b1 authored 2026-10-11T23:42:23+00:00 | PARTIAL | [record + git] the commit bounds the clock only from one side: machine time - UTC >= -31 min; a clock ahead of UTC is not excluded (UTC is plausible: meta.started 18:04:26 is 31 s before 203a5ad's 18:04:57 UTC) |
| 84 | explore_out/I/report_2.md:34 | no learned-arm record was available until b649751 (13:53 UTC on 10-12) | first commit with a learned-arm record: b649751 2026-10-12T13:53:09+00:00 | MATCH | [git]  |
| 85 | explore_out/I/report_2.md:20-21 | deviation: the note assigned WIDE_SPLIT and REG3 to the RTX 3070; the 1660 ran them instead | not in the records | NOT RECORD-CHECKABLE | [report_2] inconsistent with l.29-31: runs that started at 23:11, before the note, were not run 'instead' of the note's assignment; one of the two accounts is wrong |
| 86 | explore_out/I/gpu/s75_S4_descriptions.md:9-35 | section 1: per-run ROUTED trajectories (20 runs x 20 evaluations), end mark and end eta^2 | 20 lines reproduce, 0 differ [] | MATCH | [record]  |
| 87 | explore_out/I/gpu/s75_S4_descriptions.md:37, :124-135 | WIDE_SLOW: no records yet / pending for 10 of 10 seeds | 9 WIDE_SLOW records at 4c3984a | PARTIAL | [record + git] stale: generated at 03bffdf, before 4c3984a; not regenerated (s75_writeup promises to regenerate when they land) |
| 88 | explore_out/I/gpu/s75_S4_descriptions.md:42-112 | section 2: every eligible check of the split trigger (seed, step, probe SET (prev), map before, one-to-one, on_target, fired / blocked (gap / cap), routed@step / next / end) | 69 rows reproduce of 69 eligible checks in the records; differing [] | MATCH | [record] status recomputed independently (cap: 3 earlier fires; gap: < 4800 since the last) and compared with blocked_by |
| 89 | explore_out/I/gpu/s75_S4_descriptions.md:116-118 | fires 30 over 10 runs; on a one-to-one map 0; routed before a fire and not at the end 0 []; runs with at least one fire 10/10 | 30; 0; 0; 10/10 | MATCH | [record]  |
| 90 | explore_out/I/gpu/s75_S4_descriptions.md:124-133 | section 4: WIDE_SPLIT's columns (routed@end, fires, eta^2, SET) per seed | 10/10 reproduce; differing [] | MATCH | [record]  |
| 91 | explore_out/I/gpu/s75_S4_descriptions.md:3 | generated under code SHA 5fb120a92d69; the registered result is UNTESTED | 5fb120a92d69; UNTESTED | MATCH | [record]  |

```
TIMELINE (machine clock of meta.started, one uninterrupted invocation assumed; record secs):
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
  WIDE_SLOW 830-838 wall time 16703 s; 4c3984a at 2026-10-12 19:12:59; latest start 14:34:35; the uninterrupted reconstruction would end 838 at 10-12 15:41:15 and 839 by 16:12:15, yet 839 was pending at 19:12: WIDE_SLOW ran in a later invocation, so gaps between invocations are real and unrecorded
  meta.task_check.secs at 86652b1 / b649751 / 4c3984a: 1.0 / 1.0 / 0.9
  86652b1 per-GPU summary rows: | WIDE_SPLIT | - | | | | | | |; | REG3 | - | | | | | | |; READING (pre-fixed): incomplete  (ROUTED {'WIDE_SPLIT': 0, 'REG3': 0}; the better arm needs >= 8/10)
CHANCE BAND (ORACLE probe, all layers and evaluations): 0.2367-0.2671
```
