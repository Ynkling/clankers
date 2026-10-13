# Registry EXPL_E: notes (session E, branch origin/claude/outside-ideas, batches 1-19)

Read-only extraction. Source: `git show 5dfc2b5:explore_out/README.md` ("## Batch N verdicts" sections). 5dfc2b5 = 2026-10-09T19:49:07Z, the last README commit before 2026-10-09 19:55 UTC. TSV: `/tmp/claude-0/registry_EXPL_E.tsv` (76 rows + header, 14 columns).

## How rows were chosen
- One TSV row per verdict-table row that carries a reading or verdict. Skipped and counted below: rows marked "(printed)", "(descriptive)", "(baseline, printed)", "not a verdict", "not the rule", "reference / no reading", "cut by the runtime rule", oracle-validity rows, and "not run".
- Verdicts given only in prose (no table row) are not in the TSV. They are listed per batch below.
- Batch 19: the per-arm rows (d = +1, 0, -1) are included because they hold the counts and p-values. The two "S64 reading" / "S65 reading" rows are also included, because they hold the reading. The per-arm rows' reading column says they feed the screen reading.
- All 110 table data rows were accounted for: 76 included; 29 verdict-table rows skipped; 5 rows of the descriptive S47 decodability sub-table (batch 16) skipped.

## Columns
- test_used: wherever a p is printed it is exact McNemar, two-sided: `test_router_confirm.mcnemar_exact`, 2·P(Bin(b+c, ½) ≤ min). This is called through `explore_common`, `explore_common2` and `explore_common16`. Every TSV p was recomputed from its discordant pair and matches the README (no mismatches). Rows with no p are decided by a count threshold or by a decoder median. Their test_used and sided columns say so ("n/a").
- readings_commit: the add commit (`--diff-filter=A`) of the screen's `explore_<name>.py`. Each quoted reading name was checked against the docstring or rule text at that commit, and every one is present there. For batches 1-5 the screen rule also sits in `explore_batchN.py`, which was added in the same commit.
- Docstrings were edited after the add commit only in these places. None of the edits changed a reading's text or threshold.
  - merge_kick (S21, 7dadd6c): wording of a CHECK.
  - muon_recipe (S25, a2b0e06): a CHECK range widened from [0.7, 1.3] to [0.5, 1.5] by the user before any run.
  - gate_cap / wh_spectrum (S41/S42, 2a50422 then 90c3a28): the paired references were changed to fresh 2.80GHz REF arms and then back to the recorded 2.10GHz runs.
  - keysplits (S52, 7ef4e15): the validity check was rewritten.
- verdict_commit: the first commit that added "## Batch N verdicts" (`git log -S`). In every batch the verdict table at that commit is identical to the table at 5dfc2b5. Prose under the tables was edited later in batches 5, 6, 8, 9, 13, 15, 16 and 17. Batch 15's PREV_NOREC_M interpretation "So W_h is needed..." was later replaced with "its 10/10 KEY splits are a property of that window, not evidence about recurrence".
- Every readings_commit is dated before its verdict_commit.

## Batch 19 after 5dfc2b5 (branch head bf964ec, 2026-10-10T04:07Z)
- No verdict rows were added to README. The README at head differs from 5dfc2b5 only in the batch19.log progress row: 118 runs saved became 145 (scale_eight 30/30, capacity 7/48).
- `explore_out/batch19.log` at head does contain completed S66 readings. They first appeared at b119c76 (2026-10-10T01:39:58Z, "segment 11 end"); before that they printed as INCOMPLETE. They are not in the README and are not in the TSV:
  - S66 scale_eight D128_L3: BOUND ROUTED 8/10 vs N256_L3 8/10, discordant 2 / 2, p = 1, d = +0, reading "carries" (rule d >= -1).
  - S66 scale_eight N256_L2: BOUND ROUTED 10/10 vs N256_L3 8/10, discordant 2 / 0, p = 0.5, d = +2, reading "carries".
  - S66's reading is in `explore_scale_eight.py` at 617681d (2026-10-09T06:17:08Z).
  - S67 capacity: INCOMPLETE, no reading.

## Per batch (TSV rows / rows with a p-value / skipped table rows / prose-only verdicts)
| batch | rows | with p | skipped table rows | prose-only verdicts |
|---|---|---|---|---|
| 1 | 3 | 3 | 0 | - |
| 2 | 2 | 2 | 0 | S6 far_cue "neither reading applies" |
| 3 | 1 | 1 | 1 (pool 160-199, "not a verdict") | S8 far_EMA / far_SEL / far_A_slow: not, not, not (0/10 each); S9 descriptive ("does not keep the near case") |
| 4 | 2 | 2 | 0 | S10 far_express "neither applies" ((b) 5/10 vs 0/10, p = 0.0625) |
| 5 | 2 | 2 | 2 ("not the rule") | S14 per-run diagnostic readings (SWAP-FIXES / MEMORY-STUCK / NEITHER) |
| 6 | 2 | 2 | 1 ("not the rule") | S18 per-run diagnostic readings ("slow" / "basin") |
| 7 | 4 | 3 | 0 | - |
| 8 | 6 | 3 | 0 | - |
| 9 | 5 | 2 | 2 (descriptive; printed) | - |
| 10 | 3 | 3 | 2 (printed) | - |
| 11 | 4 | 3 | 3 (printed) | - |
| 12 | 3 | 3 | 0 | - |
| 13 | 4 | 2 | 4 (baseline / printed / printed alongside x2) | - |
| 14 | 6 | 2 | 0 | - |
| 15 | 6 | 4 | 0 | - |
| 16 | 5 | 5 | 4 (1 printed, 1 cut by the runtime rule, S46 and S47 descriptive) + the 5-row S47 sub-table | - |
| 17 | 4 | 4 | 2 (1 cut, S49 descriptive) | - |
| 18 | 5 | 5 | 3 (W_M_REF reference, 2 cut) | - |
| 19 | 9 | 9 | 5 (2 oracle-validity, 2 references, RES6 not run) | - |
| total | 76 | 60 | 29 (+5) | |

"With p" counts rows whose p column is not empty. Many of these p-values are 1 with 0 / 0 discordant pairs (both arms 0/10), mostly in batches 12, 14, 15 and 16.

## Readings whose name claims more than the comparison can show (factual)
- **B2 S5 "(b) a race".** The rule maps ROUTED*@1200 up versus X arm A (7 / 1, p = 0.070) to the mechanism "a race". The comparison shows only earlier routing. On DISCOVERED the screen is inconclusive (6 / 1, p = 0.125).
- **B5 S13 "noise floor".** The reading is two unpaired thresholds on seeds 160-179: ROUTED*@1200 >= 8/20 and at most 2 uncommitted failures. No noise-matched control is compared.
- **B6 S16 "a kick of that size suffices".**
  - It is met exactly on the rule's boundary (12/20).
  - Against SLOW_HINGE it is 1 / 3 (p = 0.625), which cannot show equivalence. ROUTED*@1200 was 4/20 vs 8/20.
  - Batch 7's S20 untimed kick was below SLOW_HINGE (0 / 7, p = 0.016).
- **B7 S19 NEAR_TOK "keeps the near case".** DISCOVERED is 19/20 vs 18/20, but ROUTED*@1200 is 0/20 vs SLOW_HINGE's 15/20 (0 / 15, p = 6e-05).
- **B7 S19 FAR_TOK "promising".** It is 6/10 DISCOVERED. By the README's own count, 3/10 route fully by stream. Batch 8's S23 then showed that a single channel binds the header layout, so a header BOUND is not evidence of routing.
- **B8 S22 NEAR_TOK_LATE "routing returns".** The arm is bit-identical to SLOW_HINGE through update 2400, so its ROUTED*@1200 is SLOW_HINGE's by construction. Its outcomes equal SLOW_HINGE's on all 20 seeds (0 / 0). The README says so: the late key term "is silent... not because it helps".
- **B8 S22 FAR_TOK_NEW "replicates".** This is a threshold on new seeds 170-179 against counts on 160-169. It is unpaired and has no test.
- **B8 S22 FAR_TOK_LATE "it costs".** The rule is c - b = 4 on ROUTED*@end, and that measure's p is 0.29.
- **B11 S34 "an early window suffices" (both optimizers).** This is equivalence read from b - c <= 1 (1 / 0 and 0 / 0). On Adam, 32/40 curves are identical by construction.
- **B11 S34 "directional kick".** The reading is defined on gradient ratio and momentum persistence, with no random-direction control in this screen. Batch 6's S16 random-direction control had read "a kick of that size suffices".
- **B13 S37 SPLITK_A "the targeted split fixes four-stream merges".**
  - 9/10 vs HINGE4k4_A 7/10, 2 / 0, p = 0.5.
  - The README says "most of it is the baseline".
  - 3 of the 9 are bound but not routed.
- **B14 S39 "formed, then lost" (three arms).** Stream decodability is 0.91 at update 0. The README says the "formed" half is met by the initialisation, not by training.
- **B16 S44(b) "breaks the eight-stream stall" (verdict promising).**
  - 3/10 vs 0/10 in each optimizer, 3 / 0, p = 0.25.
  - The candidate (window gate, no hinge) and the references (recurrent gate with main's hinge) differ in more than one factor.
- **B16 S44(a) "carries to four streams".**
  - 20/20 vs X HINGE4k16 15/20 (5 / 0, p = 0.0625).
  - The screen docstring names the reference as "X's HINGE4k16 (17/20 BOUND)". The table uses 15/20 BOUND ROUTED. Batch 11's S35 row gives X HINGE4k16 17/20 BOUND.
- **B17 S48 W_SPLIT_D98 "improves... (the splits, not the decay)".** The attribution rests on W_D98 (1/10). There is no paired W_SPLIT_D98-vs-W_SPLIT comparison.
- **B17 S48 W_LONG.** The reading is "neither", but the verdict is recorded as "not".
- **B18 S50 W_RESET "the copy does the work".**
  - The arm is a reset-only control: it applies the optimizer-state reset without the row copy.
  - It shows that the reset alone does not reproduce W_SPLIT (1 / 8, p = 0.039).
  - There is no copy-without-reset arm, and the split also adds noise to both rows.
  - The README's observation goes further ("so W_SPLIT's gain comes from the copy"). The comparison does not test that.
- **B18 S51 "the split fixes four-stream merges at k=4".**
  - 8/10 vs 5/10, 5 / 2, p = 0.453.
  - 7 of the 8 bound runs fired no split and are the no-split W4k4's runs bit for bit. The no-split W4k4 arm was cut.
  - The README's own verdict says "not shown for the split".
- **B18 S52 W2_KEYHINGE "removes the key splits".**
  - 36/40 vs 31/40, 7 / 2, p = 0.18; vs SLOW_HINGE 6 / 4, p = 0.754.
  - The KEY count of 1 depends on 3 runs sitting exactly at the 0.5 eta^2 threshold. Labelled KEY, they would give "neither".
  - Two new failures appeared, and every failure formed after the key term had switched off.
- **B19 S64 / S65 "scale-free".**
  - The margin is a count margin (d >= -1) with n = 10 and a reference at 9/10 or 10/10. The README notes it rules out only a large drop.
  - All four S64 d = +1 values rest on one seed (273).
  - In S65 the layers share weights, so "depth" is repeated application with an unchanged parameter count. L6 was slower (transition median 4800).
  - The L2-vs-L4 phase-2 tie-break by arm order is not in the pre-registered rule (the README says so).
- **Validity caveat, not an overclaim:** B9 S28 far_p8 reads "it does not" by the rule while its validity arm failed. The README records "no answer".
