# Revision 9 draft, fifth version: fifth pass

*Referee V, 12 October 2026.*

**What was checked.**
- **The fifth version.** `multichannel_hebbian_report_v9.tex` and `.pdf` on `claude/rev9-draft` at `c5e4508`, committed 19:14:19 UTC, 568 lines, cited as `tex5:NNN`.
- **Four later commits.** The branch moved during this pass to `c916996` (22:59:26): `cc379b2`, `b5fd2f2`, `dbd9a7e` and `c916996`. They take in this pass's interim commits and change 14 lines in place, so `tex5:N` is the same line at `c916996`.
- **The verdict is on `c916996`**, the head when this file was written.

**How.**
- **The v4 checks, re-run as v5 against `c5e4508`:** `check_R_v5`, `check_E19_v5`, `check_G_v5`, `check_H_v5`, `check_F_v5` (with F's report 7, S81, in full), `check_split_copy_v5` and `check_appendixB_v5`.
- **The material added since `c66cc76`, checked against its records:**
  - `check_split_copy_L` (L and pooled, `ffb0771`);
  - `check_delta_recipe_X` (X, `8ae660d`, with Amendment 2's timing);
  - `check_S75` (S75 at four sources and session I's report 2).
- **The item-by-item verdicts:** `resolution_R_E5.md`, `resolution_G_H5.md` and `resolution_F_sc_appB5.md`.
- **The four later commits were checked here, line by line:**
  - `check_onetoone_v5.py` recomputes the new firing counts and the abstract's pairs from the records.
  - `check_pdf_v5.py` compares the PDF with the tex.

**Rules followed.**
- Nothing outside `docs/review_2026-10/` was changed, and nothing was trained.
- The Appendix B check built models read-only from `git archive 8ae660d`.

## Verdict

**Not ready to merge.** It is closer than any earlier version.
- At `c916996` every number of a claim reproduces from its record, and so does every pre-registered verdict, interval and reading.
- Every item `fourth_pass.md` raised on R, batch 19, G, H and F is resolved or needs no change.
- **What remains:**
  - **One false statement**, in the abstract (§2 item 1).
  - **One statement of fact the records cannot support** (§2 item 2).
  - **About ten loose statements** (§3). Six are in the S75 paragraph, which none of the four later commits touched.

None is a count, a pair, a p-value or an interval of a pre-registered claim.

**What should clear it.** One more version, checked against the branch heads at its commit.

**Corrections to the referee's own files, made in this pass:**
- **`fourth_pass.md` §1 item 6 is withdrawn.** X's training runs began after 05:27:31, so the check at 03:01 preceded every run. The fourth pass's nine false statements were eight.
- **`fourth_pass.md` also had wrong scope for G's KL terms.** Its "while the terms are on" is wrong for the position-only term, and the draft took it over at tex5:271 (fixed at `b5fd2f2`).
- **`fourth_pass.md` missed R's second interval error.**
- **`test_check_delta_recipe.md` and `test_check_delta_recipe_amendment2.md`:** the dated corrections about X's start were themselves corrected.

All of these are dated corrections in those files.

**The prompt's own WIDE_SLOW figure.** The prompt gives "WIDE_SLOW's nine seeds within 0.003 of WIDE_SPLIT". That is not true unrounded: seed 831 differs by 0.00325 (0.53274 against 0.53598). The draft itself says only "seed for seed", which holds.

## 1. The checks against `c5e4508`

| check | rows | DISAGREE | PARTIAL | not record-checkable | MATCH |
|---|---|---|---|---|---|
| `check_R_v5` | 39 | 0 | 1 | 2 | 36 |
| `check_E19_v5` | 31 | 2 | 0 | 0 | 29 |
| `check_G_v5` | 159 | 1 | 2 | 2 | 154 |
| `check_H_v5` | 116 | 0 | 1 | 2 | 113 |
| `check_F_v5` (live rows) | 182 | 2 | 9 | 6 | 165 |
| `check_F_v5`, S81 (report 7) | 40 | 0 | 0 | 0 | 40 |
| `check_split_copy_v5` §1 | 45 | 0 | 1 | 0 | 44 |
| `check_split_copy_v5` §4 | 20 | 1 | 4 | 0 | 15 |
| `check_appendixB_v5` (FALSE / PARTIAL / UNVERIFIABLE / TRUE) | 106 | 0 | 0 | 4 | 102 |
| `check_split_copy_L` (L and pooled) | 62 | 2 | 4 | 1 | 55 |
| `check_delta_recipe_X` | 47 | 2 | 8 | 2 | 35 |
| `check_S75` | 91 | 1 | 16 | 7 | 67 |

**Some DISAGREEs are counted twice.** The checks overlap on shared lines. tex5:41's "every session's queue is empty" is counted in E19 and F. tex5:256's "none … a Hebbian run" is counted in delta_recipe_X, F and split_copy §4.

**Distinct items wrong at `c5e4508`.** Six false statements:
- tex5:368, the one-pair rule's power stated the wrong way round;
- tex5:41, the queue;
- tex5:271, the KL bound and its scope;
- tex5:256, the D1 label;
- tex5:256, "none … a Hebbian run";
- tex5:499, the torch version.

There were also the PARTIALs of the table, some counted in more than one check.

**Of those, after `cc379b2`–`c916996`:**
- All six false statements are fixed.
- Every PARTIAL on a line those commits changed is fixed or reduced (§4). The S75 PARTIALs were not touched, and they and the rest are in §3.
- One new false statement was introduced (§2 item 1).

**What reproduces.** These are recomputed from their records; every pre-registered number matches.

- **`test_split_copy` on L and pooled (`ffb0771`).**
  - C1 pooled 18/20 vs 6/20, 12 vs 0, p = 0.000244, SHOWN; on X 7 vs 0, on L 5 vs 0 (p = 0.031).
  - C2 pooled 6/20 vs 8/20, 0 vs 2, HOLDS.
  - C3 pooled 18 vs 0, p = 3.8e-6, SHOWN.
  - The bands, and both readings, apply.
  - The report regenerated with the test's own functions equals the log's printed report line for line (log:2447-2774, 0 differing lines).
- **`test_delta_recipe` on X (`8ae660d`).**
  - BOUND ROUTED: DELTA_SPLIT 19/20, DELTA_SLOW 19/20, HEBB_SPLIT 16/20, HEBB_SLOW 7/20.
  - D1: d = −3; DELTA − HEBB +0.150, Newcombe [−0.082, +0.374], equal to an independent implementation.
  - D2: 0 vs 0, NOT SHOWN.
  - D4: 10 vs 1, p = 0.00586, SHOWN.
  - D3: 10 earlier, 4 later, 1 tie, p = 0.090.
  - CHECKs 171 and 174 pass, with no pair broken.
  - The RESULT is the `--report` regeneration under `2225f00`.
- **Amendment 2's timing.** No log is on any ref, so finish times were reconstructed from `meta.started`, each run's `secs` and the scheduler.
  - At 15:36:08 the finished runs are exactly the amendment's 24.
  - DELTA_SPLIT 599 is within seconds either way.
  - No HEBB_SPLIT or HEBB_SLOW run had finished.
  - The process was never resumed.

  Details are in `check_delta_recipe_X.md` and the addendum to `../test_check_delta_recipe_amendment2.md`.
- **F's S81 (report 7, `d5851b1`).** All 40 rows match.
  - BASE 9/10; GUARD 7/10 (0 vs 2); EVAL 10/10 (1 vs 0); GUARD_EVAL 8/10.
  - No reading applies.
  - The readings were committed at `451cde9`, before the first S81 record.
- **S75 (`4c3984a`).**
  - ORACLE ended at 0.8915 and 0.8801, below 0.9, so the screen is not VALID and its reading is UNTESTED.
  - ROUTED is 0/10 on both read arms.
  - η² by source is below 0.003 at every evaluation of every gate layer (largest 0.00225), and no map is one-to-one at any evaluation.
  - The probe at 1200 is 0.35–0.61 in the gated arms and at chance in ORACLE and SINGLE.
  - The fires' masses are 0.05–0.38, with margins at most 0.04.
  - The descriptions file's 20 trajectory lines and 69 split-check rows reproduce.
  - The rules and the code predate the records.
- **The PDF.** `check_pdf_v5`: of the six-word runs found only in `c5e4508`'s tex, 95.0% are in its PDF; of those found only in `c66cc76`'s, 13.0% are. The PDF was built from the tex it was committed with. The remaining misses are table and caption layout.

## 2. What is false or unsupported at `c916996`

**1. The abstract (tex5:36), new at `dbd9a7e`.**
- **The text:** "its pre-registered test is complete on one machine (the delta recipe $19/20$ against the Hebbian $16/20$, and without a trigger $19/20$ against $7/20$, 10 vs 1)".
- **The error:** the pair "10 vs 1" belongs to D4, the Hebbian split against no trigger (16/20 vs 7/20). Here it stands after the no-trigger arms, delta 19/20 against Hebbian 7/20, whose pairs are 12 vs 0. The split arms' pairs are 4 vs 1. (`check_onetoone_v5_out.txt`.)
- *Fix:* "without a trigger $19/20$ against $7/20$ (12 vs 0); the split helps the Hebbian write, 10 vs 1".

**2. tex5:458, S75: "(three window-gate runs had by then finished on the card, unseen by the session, …)".**
- **Not in the records.** No record carries a start or end time. The count comes from report 2's chaining of run durations with no gaps between invocations.
- **Contradicted by git.**
  - The summary committed at `86652b1` is written only when an invocation ends, and it lists the learned arms as not run. So the ORACLE/SINGLE invocation ended before the learned arms started.
  - Gaps between invocations did occur: chained without gaps, WIDE_SLOW 839 would have finished by 16:12, yet it was pending at 19:12.
- **What the records allow.** The learned arms started between 23:11:35 and 02:01:52, so 0 to 3 WIDE_SPLIT runs had finished at the note's commit (00:46:28).
- *Fix:* "committed before any learned-arm record was pushed; whether runs had started on the card is not recorded".

**Session I's report 2 itself (task 4).** Its "the other seven WIDE_SPLIT runs … came after it" (report_2:31) fails even on its own assumptions: WIDE_SPLIT 833 starts at 00:44:50, 98 s before the note. Full detail is in `check_S75.md`, "Timeline".

## 3. Loose statements at `c916996`

**S75 (tex5:458), unchanged since `c5e4508`.** All values are from `check_S75`.
- **"its accuracy rose by at most $0.03$ after 4800 updates" (SINGLE, nine seeds).** The rises are 0.0119–0.0329; seeds 830 (0.0301), 831 (0.0302) and 839 (0.0329) exceed 0.03. Say "at most $0.033$".
- **"and had lost it by 3600" (the residual's source probe, every gated run).** True of 29 of 30 runs; REG3 839 reads 0.291 at 3600 and stays above 0.27 at 17 of 18 evaluations.
- **"identical across sources to within $0.02$".** The largest difference is 0.0235; 12 of 30 fires exceed 0.020.
- **"$0.535$--$0.540$" (REG3's end).** This holds on nine seeds; seed 839 ended at 0.588. The sentence says "nine seeds" only for one channel.
- **"a late rise of about $0.06$ took the probe to $0.29$ and $0.28$".**
  - The rises after 4800 are 0.091 (SINGLE 838) and 0.084 (REG3 839). The 0.06 is their excess over the other seeds.
  - 0.29 and 0.28 are end values; SINGLE 838 peaked at 0.340.

**S75's place in the rest of the draft:**
- **tex5:478, next steps.** "the window gate without the split on the pre[sent configuration first]" is stale: 9 of 10 WIDE_SLOW runs are recorded (`4c3984a`) and described at tex5:458.
- **tex5:516, I's provenance row.** It stops at `03bffdf` with 32 runs. tex5:458 uses the nine WIDE_SLOW records of `4c3984a` (41 records in all).

**Other lines:**
- **tex5:201, L's `test_split_copy` CHECKs.** "every CHECK passed, those comparing with L's own records asserted and the three comparing with Xeon records printed, as on X".
  - The three cross-CPU CHECKs (135, 141, 150) print "DIFFERS" on L (log:2271-2342), as designed. That is not "passed".
  - On X, CHECKs 135 and 141 were asserted.
  - *Fix:* "the three comparing with Xeon records printed, and differ across CPUs, as the specification expects".
- **tex5:477, next steps.** "session~F's trigger on delta channels (S81)" is stale: S81 is complete and folded in at tex5:205. If a main-line test is meant, say so.
- **tex5:518, V's provenance row.** It ends at `0d1fa60`. The referee's later commits before the draft are the fourth pass (`907eb2c`), its interim files, and this pass's interim commits, which the draft cites in its commit messages but not in the row.
- **tex5:349, "Every item is addressed in this version".** True of every earlier pass's items. As a sentence about this version it is not yet true while §2 stands.

**What needs no change:**
- **tex5:54, the index's "(X; L running)".** L's run is not record-checkable; nothing on any branch shows it.
- **tex5:330, Phase 4's start before the amended rule.** Not record-checkable, unchanged, no fix needed.
- **The two omissions at tex5:530 and Appendix B items 6–8.**

## 4. The four later commits, line by line

Each changed line of `c5e4508..c916996` was compared with the record it cites.

| line | change | status |
|---|---|---|
| 36 | replication's subject scoped to its first twenty seeds | RESOLVED (`check_R_v5`'s tex5:36 PARTIAL) |
| 36 | memory rule: "complete on one machine (…)" | **NEW ERROR**, §2 item 1 |
| 41 | "E's batch 20 … and the sessions' fifth round are running" | RESOLVED. E `a876751`, G `3c89f87`, H `5b5deec`/`f4fd535` |
| 54 | index: "(X; L running)" | RESOLVED. The status is right on X; L is not record-checkable |
| 58 | F's cells: "lost two seeds", "none (no reading applies)" | RESOLVED. GUARD 0 vs 2; no S81 reading applies |
| 201 | L's CHECKs | PARTLY, §3 |
| 205 | the flat-start account "on session F's post-hoc account"; the 914 mode's other occurrences (GUARD_EVAL 2025 at four checks, BASE at two) | RESOLVED |
| 256 | C3 "pooled and on each machine"; "run on X"; the two Hebbian perfect-gate lines among the 24; D1's first label as Amendment 2 worded it; D2 NOT SHOWN on X with "no room" waiting for L; "with" for "because" | RESOLVED. Every number against `check_delta_recipe_X` |
| 271 | the KL "from update 300 through 1200 (at most $1.4\times10^{-5}$; the flat term's through 2400)"; the spec's expectation of the opposite failure | RESOLVED. `term.value_at`: seed 2109 at 1200 is 1.3e-5, 1.4e-5, 9e-6 |
| 360 | "specified and complete on X …; L's half is running" | RESOLVED |
| 368 | "about two times in five" | RESOLVED (0.398) |
| 382 | C1 "12 vs 0 pooled, 7 vs 0 on X and 5 vs 0 on L" | RESOLVED |
| 441 | 126 firings in 60 runs; all-position pair labelled at 115 (46 of 47, 30 of 38, 21 of 22, 18 of 19); coincide at 54; 11 in 126; none on a one-to-one map | RESOLVED. `check_onetoone_v5`, independently of H's `5b5deec`: 60 / 126 / 115 / 54 / 0 |
| 495 | split_copy row: L's docstring corrected at `80bfc1d` | RESOLVED. `80bfc1d` (main line, 22:51:55) commits L's second-start json and corrects "exit 0" |
| 499 | caption: torch 2.14.1 for L's `channel_binding` rerun and `split_copy` run | RESOLVED (`split_copy` meta.torch 2.14.1) |

## 5. The rows of `fourth_pass.md`

| fourth pass | `c5e4508` | `c916996` |
|---|---|---|
| §1.1 tex4:328, the interval "includes 0.95" | RESOLVED, tex5:330 ("reaches 0.945, just below 0.95") | — |
| §1.2 tex4:36, "never saw the code" | RESOLVED, tex5:36; PARTIAL residue | RESOLVED |
| §1.3 tex4:354, "from the text alone" | RESOLVED, tex5:356 | — |
| §1.4 tex4:203, "occupied channel" | RESOLVED, tex5:205 | — |
| §1.5 tex4:341 and 459, stale | RESOLVED, tex5:343 and 463 | — |
| §1.6 tex4:41, "before any run" | withdrawn (the referee's error) | — |
| §1.7 tex4:54, "not run" | RESOLVED; label "(specified)" left | RESOLVED |
| §1.8 tex4:254, "not run" | PARTLY ("running on X") | RESOLVED |
| §1.9 tex4:330, the parenthesis | RESOLVED, tex5:332 | — |
| §2 R: 914 inference; 150 vs 52; "near 0.70"; 13 of 192; "920–939 alone" | RESOLVED (tex5:330, 393, 207, 356) | — |
| §2 R: the two not-record-checkable rows | NOT (none needed) | — |
| §2 E: power at eight streams | NEW ERROR at tex5:368 | RESOLVED |
| §2 E: D128 oracle 1/2; L2 tie-break; tex4:57; 452; 233; 464 | RESOLVED (tex5:237, 57, 454, 235, 468) | — |
| §2 G: "bit for bit" at one seed | RESOLVED, tex5:271 | — |
| §2 G: "never leaves its initial value" | NEW ERROR at tex5:271 (taken from this referee's wrong scope) | RESOLVED |
| §2 G: the ratio-term clause unmarked | RESOLVED, tex5:271 | — |
| §2 G: "positions only, or nothing" / "needs key identities" | PARTLY, tex5:441 | RESOLVED |
| §2 G: "proposed to follow" | NOT (none needed) | — |
| §2 H: "the same rate"; "all body positions" | RESOLVED, tex5:441 and 205 | RESOLVED (126 firings) |
| §2 test status: tex4:472, 492 | RESOLVED, tex5:476 and 496 | — |
| tex4:347, "Every item is addressed" | PARTLY | PARTLY (§3) |
| §2a, sessions' reports (R's interval; G's report 6; F's report 6) | NOT (outside the draft; the draft carries none) | — |

## 6. Errors in the sessions' own files (not the draft)

These do not block the draft; they are listed so their authors can correct them.

- **R, `replicate_report.md`.**
  - Line 138: "includes 0.95" (35/40 is [0.739, 0.945]).
  - Line 27: 17/20's interval "contains 0.95" (it is [0.640, 0.948]).
- **G, `report_6.md`.**
  - Lines 38–39: "η²_key 0.18 at update 1" is the term's value; η² is about 0.42.
  - Line 44: "0.0000 from update 300" is false at 2400 for the position-only term.
  - Lines 52–53: the 1200-window failures split at 3600, after the slow phase.
- **F, `report_6.md`:100-101.** It gives seeds 1300–1329 for H's `two_results.json`, which holds 1300–1319.
- **I.**
  - `report_2.md`:31, the timeline (§2 item 2).
  - Lines 29–33: the no-gap and "can only be UTC" arguments.
  - Lines 45–52: three bounds that hold only after rounding.
  - `s75_S4_descriptions.md`:37 and 124–135: WIDE_SLOW is still "pending", though nine records exist.
- **The main line, `test_split_copy.py` at `ffb0771`, doc:125.**
  - "every inherited and own CHECK passed, asserted": three cross-CPU CHECKs were printed and DIFFER on L.
  - doc:122's "exit 0" is now supported by `80bfc1d`. doc:125 is unchanged there.

## What could not be checked

- **X's `test_delta_recipe` log.** It is on no ref. Without it, these cannot be checked:
  - the process start at 02:12;
  - the absence of a manual `--report` before the amendment;
  - that both reports are kept in the log;
  - the verification CHECKs.
- **Per-run times for S75.**
- **Running session code.** The CHECKs of S81, S83 and S86, and the regeneration of I's summary, were read, not re-run.
- **L's working tree** at its `test_split_copy` run (no file hash is recorded).
- **Push times**, beyond the GitHub activity that the S75b review read.
- **Anything after `c916996`.**

## Corrections (13 October, sixth pass)

1. **§4, the row for line 41 ("RESOLVED. E `a876751`, G `3c89f87`, H `5b5deec`/`f4fd535`").** The cited G and H commits do not show a run:
   - `3c89f87` is G's S87 specification; G's start is `02b3eed` (22:48:43).
   - `5b5deec` is H's completed one-to-one recount, and `f4fd535` is H's code and readings.
   - H's first record of a run is `b17517a` (23:00:50), after `c916996` (22:59:26).

   "Running" was right in substance (H's CHECKs ran about 22:47–23:00), but the commits cited do not show it. (`check_G_v6`, `check_H_v6`.)
2. **`check_H_v5.py` is not reproducible as committed.** Its provenance row and its "after the draft" facts read the moving origin head. Re-run after H's round-5 stores were added, that row flips from MATCH to DISAGREE. `check_H_v6.py` pins `f4fd535` and reproduces all 116 v5 statuses. `check_G_v5.py`'s header reads the moving head too, but only in text.
3. **A label slip, no line changes.** `check_onetoone_v5.py`'s docstring calls `c916996`'s line "tex6:441"; in this file's numbering it is tex5:441. Tex6 now means `42c4557`; the line number is the same.
4. **§3, tex5:201: "the three comparing with Xeon records".** This pass's proposed fix undercounts, and the draft and the main line's docstring (`bbb600c`) both took it over.
   - On L, the inherited CHECKs 122 and 123 also print, not asserted, a comparison with a record written on the 2.10 GHz host, and both are unequal (log:2191, 2195).
   - CHECK 112 ("informative, not asserted", log:2125) compares with E's screen records, whose CPU is not recorded.
   - `check_split_copy_L.py` searched for the phrase "printed, not asserted" and missed all three. (`check_split_copy_L_v6`.)
   - The log's own verdict is "ALL VERIFICATION CHECKS PASSED", 24 times. This file's "that is not 'passed'" is therefore a reading of the word, not of the log. The exact wording is "the cross-CPU comparisons printed and unequal, as the specification provides".
5. **§2 item 1's fix carries a pooled label onto one machine.** The fix was "the split helps the Hebbian write, 10 vs 1". "Helps" is D4's pooled label, which X alone cannot give; that is the same objection this pass raised to "no room". The exact wording is "shown on X (10 vs 1)".
6. **§3, "True of 29 of 30 runs"** (the residual's source probe by 3600). There are 29 gated runs (10 + 10 + 9), so it should read "28 of 29". `check_S75.py`:347 hard-codes the same slip.
