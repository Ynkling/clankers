# check_R_v6: session R's Phase 4 in the Revision 9 draft, sixth version

This check covers the draft at `42c4557` on `claude/rev9-draft`, committed 2026-10-12 23:14:04 UTC (tex6:NNN is `multichannel_hebbian_report_v9.tex` there, 568 lines). It is compared with the fifth version at `c5e4508` (tex5:NNN). The changes `c5e4508..42c4557` replace lines in place, so tex5:N is tex6:N.

R's records were read with `git show` from `claude/replicate-R`:
- `1f8be04`, the final Phase 4 commit;
- `0364e28`, the interim records;
- `464be8d`, the pre-registration.

**R's branch when `42c4557` was committed.** The head was **`550a736`** (22:47:46 UTC on 12 October, 26 min before the draft): "Phase 5 protocol: copy vs reset vs one delta channel (seeds 940–959)" (§12b).
- That is the only commit after `1f8be04`. It adds `phase5.py` and 21 entries to `UNSTATED.md` (U-50..U-70). There is no Phase 5 record, and `phase4.py` and `records/phase4.jsonl` are byte-identical to `1f8be04`.
- It is still R's head now.

The script is `rev9/check_R_v6.py`, adapted from `check_R_v5.py`. It uses only the standard library, writes nothing, and its output is quoted as §N below and given in full at the end.
- §0–§12 are v5's record recomputations, unchanged. Their output is identical to v5's apart from three things: the tex6 labels, one added commit line (`42c4557`), and §12's origin-head line, which now reads `550a736 (NEWER: see 12b)`.
- §12b is new. It gives R's branch at the draft's commit time (`git log --until=2026-10-12T23:14:04Z`) and the numbers behind the changed wording.
- §13 locates the wording.
- §14 prints the rows. Each row is anchored on a fragment of its tex6 line (0 anchors missing), and the counts follow.

Bound routed is recomputed with `replication_check.py`'s `r_transition` and `r_bound_routed`. The script extracts them with `ast`, so that module never runs.

**Re-anchoring.** Simply swapping `c5e4508` for `42c4557` in `check_R_v5.py` leaves 2 anchors missing, both on tex6:36: v5 row 1 (the subject "from the report alone") and v5 row 5 ("had not seen the code when it ran its first twenty seeds"). Both are re-anchored on the new text.
- v5 row 8 (tex6:41) is re-anchored on the whole new status clause. Its old fragment "R's Phase~4 are complete" still matched, but the clause around it changed.

## What changed for R between the fifth and sixth versions

- **tex6:36 (the abstract).** It now reads: "A clean-room replication, from the report alone for its first twenty seeds, by a session that had not seen the code when it ran them, bound eight streams $17/20$ …, and $18/20$ on twenty further seeds after taking the split's probe and key positions from our code". This is v5 row 1's suggested wording, verbatim.
- **tex6:41 (the status).** "every session's queue is empty" is removed (`b5fd2f2`, 22:53:57). It now reads: "… R's Phase~4 are complete, E's batch~20 (S67's unbound runs continued to 86{,}400 updates) and the sessions' fifth round are running".
- **tex6:349 (the account of the referee's passes).** It now says each pass "found the previous version's own slips, nearly all in the text added since the pass before (among them a miscount of R's labels, the uncommitted Phase~4 reading, …)", and that "the fifth pass found every number of every pre-registered claim reproducing".
- No R number changed.

## Recomputed (unchanged from v5)

| item | recomputed | draft (tex6) |
|---|---|---|
| Phase 2 split / no split | 17/20 [0.6396, 0.9476] / 7/20; 10 vs 0, p = 0.00098 | 17/20 [0.640, 0.948]; 7/20; 10 vs 0 (36, 330) |
| against the report | Fisher two-sided p = 0.322; X + L + R 55/60 [0.819, 0.964] | 0.32; 55/60 [0.819, 0.964] (330) |
| Phase 4 920–939 | 18/20 [0.699, 0.972] vs 5/20, 13 vs 0; ORACLE_ADAM 2/2 at 2400 | 18/20; 2/2 at 2400 (36, 330) |
| 40 pooled | **35/40 [0.7389, 0.9454]** vs 12/40; 23 vs 0, p = 1.19e-7; "consistent with the report" | 35/40 [0.739, 0.945]; 12/40; 23 vs 0 (36, 330) |
| blocks | fixed probe 3 of 263; per-seed probe 13 of 282 (Phase 2 13/192, R2 0/90); 914: 12 blocked, 11 eligible | 3 of 263; 13 of 192 (13 of 282) (207, 330) |
| first rule's power | P("a lower rate") 0.113 at 0.70 (1 in 8.8) | "one time in nine" (330, 393) |
| amended (b) | x ≤ 27/32: size 0.020, power 0.536 at 0.85 | 0.020, 0.54 (330) |
| 150 vs 52 | two arms: 141 per arm two-sided, 111 one-sided; one sample: n = 52, x ≤ 46, size 0.045, power 0.812 | "about 150 … as two arms"; 46/52 (330, 430) |
| UNSTATED | at `1f8be04`: stated 4, in part 3, [A] 20 + [B] 12 + [C] 2 = 34, process 3; total 44. At `550a736`: 65 rows (the 44 unchanged + U-50..U-70 for Phase 5) | 44 (34 guesses) (36, 332) |
| CPU | no CPU/host/device field; Phase 4 `wall_s` 37.2 h | "R's records carry no CPU model" (41) |

## Rows: DISAGREE first

| # | tex6 line | item | draft says | recomputed | status | note |
|---|---|---|---|---|---|---|
| 1 | 330 | pre-registration | 'pre-registered before any run (464be8d)' | no phase4.jsonl at 464be8d; rule and code unchanged to 1f8be04; replayed start no later than 19:02:43 UTC vs 464be8d 19:01:33 | NOT RECORD-CHECKABLE | Unchanged from v4 row 10: the records carry no timestamps. |
| 2 | 330 | the amended reading's history | written after the review; no commit until db22857, after Phase 4 had started; not used | db22857 10-11 03:01 after 0364e28 10-11 00:02 (21 records); no amended arm, seed or rule in R's records | NOT RECORD-CHECKABLE | Unchanged from v4 row 11: 'written after the referee's review' rests on chat; the rest is checkable and matches. |
| 3 | 25 | title | '17/20 in a replication from the text alone' | 17/20, Phase 2 | MATCH |  |
| 4 | 36 | the abstract's subject 'from the report alone' (v5 row 1, PARTIAL; changed text) | 'A clean-room replication, from the report alone for its first twenty seeds, by a session that had not seen the code when it ran them, bound ... 17/20 ..., and 18/20 on twenty further seeds after taking the split's probe and key positions from our code (35/40 ...)' | 17/20 = Phase 2, seeds 900-919 (de8df0f 13:55), before DIFF.md (00990ea 14:01); 18/20 = Phase 4, 920-939, D-1 and D-2 from our code; 35/40 pooled over both | MATCH | v5 row 1's suggested wording, verbatim: 'from the report alone' now governs the first twenty seeds only, and the 18/20 and 35/40 are said to use our code. The v5 PARTIAL is resolved. |
| 5 | 36 | the abstract's 'had not seen the code' (fourth pass §1 item 2; changed text) | 'by a session that had not seen the code when it ran them' ('them' = its first twenty seeds) | Phase 2 records de8df0f (13:55) before DIFF.md 00990ea (14:01); Phase 4 pre-registered at 464be8d (19:01) | MATCH | Same claim as tex5:36 ('when it ran its first twenty seeds'), the antecedent moved into the subject. |
| 6 | 36 | abstract counts | 17/20 against 7/20 (10 vs 0); 18/20; 35/40 against 12/40 pooled, 23 vs 0 | P2 split 17/20, no split 7/20; P4 18/20; pooled 35/40 vs 12/40, 23 vs 0 | MATCH |  |
| 7 | 36 | UNSTATED count | 44 details (34 guesses) | 44 entries at 1f8be04; [A] 20 + [B] 12 + [C] 2; at R's head when 42c4557 was committed (550a736) UNSTATED.md has 65 rows, the 44 unchanged plus 21 for Phase 5 (U-50..U-70) | MATCH | See §10 for the label tally. The 44 are the entries through Phase 4, the replication the draft reports (provenance 1f8be04); the Phase 5 entries were logged after it and belong to a phase the draft does not report. |
| 8 | 41 | R's status at 42c4557 (changed text: 'every session's queue is empty' removed) | '... and R's Phase 4 are complete, E's batch 20 (...) and the sessions' fifth round are running' | 42 of 42 jobs, one '42 jobs to run' header; R's head at 42c4557's commit time 550a736 (2026-10-12T22:47:46): 1 commit after 1f8be04, the Phase 5 protocol (phase5.py, UNSTATED.md), no Phase 5 record; phase4 records and phase4.py unchanged | MATCH | R's Phase 4 is complete. R's Phase 5 protocol (copy vs reset vs one delta channel, 940-959) was committed 26 min before the draft, 'before any Phase 5 run', and is not mentioned; whether it was running at 23:14 is not recorded. Not an error: the draft reports Phase 4, and 'the sessions' fifth round' may be read to cover it. The removed 'every session's queue is empty' (v5 row 8's note) is gone. |
| 9 | 41 | CPU | 'R's records carry no CPU model' | no CPU, host or device field in phase2/3/4 (§8) | MATCH |  |
| 10 | 62 | index row ('the main line's protocol', fourth pass §2 / v4 row 4) | '18/20 with the split's probe and key positions as the main line specifies them' | WIN3_SPLIT_D8_ML = D-1 (fixed probe) + D-2 (32 body keys), phase4.py:5-9 | MATCH | R's own wording ('as the main line specifies it'). |
| 11 | 207 | fixed-probe count | 3 of 263, never decided a run | R1 + Phase 4: 3 of 263, 1 eligible (905, bound routed) | MATCH |  |
| 12 | 207 | per-seed count and '914' (fourth pass §2, tex4:205) | '13 of 282 and, on R's inference, it decided one run' | Phase 2 + R2: 13 of 282; 914: 12 blocked, 11 eligible, bound routed False | MATCH | Pooled the same way as the fixed-probe 263 (both include a 900-909 rerun); 'on R's inference' marks the attribution. |
| 13 | 207 | 914's readings (unchanged text) | probe 0.953-1.000 from 16,800, evaluation 0.871-0.887 | probe 0.953-1.000, evaluation 0.871-0.887 | MATCH |  |
| 14 | 330 | Phase 2 (unchanged text) | 17/20 [0.640, 0.948]; 7/20; 10 vs 0, p = 0.00098 | 17/20 [0.640, 0.948]; 7/20; 10 vs 0, p = 0.00098 | MATCH |  |
| 15 | 330 | against the report (unchanged) | Fisher two-sided p = 0.32; 55/60 [0.819, 0.964] | p = 0.322; 55/60 [0.819, 0.964] | MATCH |  |
| 16 | 330 | Phase 2 bar | fails a true 0.90 recipe a third of the time | P(Bin(20, 0.9) <= 17) = 0.323 | MATCH |  |
| 17 | 330 | Phase 4 design (v4 rows 4, 18) | 'the split with its probe and key positions as the main line specifies them (...; its other choices, among them the map on the held-out sequences, R's own)' | D-1, D-2 adopted; D-4..D-10 R's own (D-10: map on 2048 held-out sequences) | MATCH |  |
| 18 | 330 | validity | 2/2 at 2400 | ORACLE_ADAM 920, 921 transitions [2400, 2400] | MATCH |  |
| 19 | 330 | pooled count | 35/40 [0.739, 0.945]; 12/40; 23 vs 0; p = 1.2e-7 | 35/40 [0.7389, 0.9454]; 12/40; 23 vs 0; p = 1.19e-07 | MATCH |  |
| 20 | 330 | reading | 'consistent with the report'; main line 38/40 vs 15/40, 23 vs 0 | 'consistent with the report'; main line 38/40 vs 15/40, 23 vs 0 | MATCH |  |
| 21 | 330 | 920-939 alone (fourth pass §2 '920-939 alone'; v4 rows 5, 21) | '18/20 ..., which phase4.py prints as secondary without a reading' | 18/20 vs 5/20, 13 vs 0; phase4.py:19-20 'printed beside it, as a secondary count', :125 prints it, no reading | MATCH |  |
| 22 | 330 | failures | two streams sharing a channel after three splits | 920: 3 splits, shared [0]; 933: 3 splits, shared [4]; X 555, 556 each 3 fired splits and a shared channel (§5) | MATCH |  |
| 23 | 330 | the interval vs 0.95 (fourth pass §1 item 1) | 'the interval reaches 0.945, just below 0.95 (R's report says it includes 0.95; ... unaffected)' | Wilson(35, 40) upper 0.9454 < 0.95; replicate_report.md:138 'includes 0.95' | MATCH | The fourth pass's fix, and R's own error now attributed to R. |
| 24 | 330 | lower bound | excludes rates below 0.74 | lower 0.7389 | MATCH |  |
| 25 | 330 | 150 vs 52 (fourth pass §2) | 'as two arms against each other ... about 150 seeds per arm (against a fixed rate ..., the 52 of Section sec:method ...)' | two-sample normal approx: 141 per arm (two-sided), 111 (one-sided); R says 'on the order of 150' (replicate_report.md:139); one-sample n = 52, x <= 46, size 0.045, power 0.812 (tex6:430) | MATCH | R's figure is labelled R's; the two questions are now told apart. |
| 26 | 330 | the first rule's power | 0.70 about one time in nine | 0.113 = 1 in 8.8 | MATCH |  |
| 27 | 330 | the amended reading's numbers | x <= 27/32, size 0.020, power 0.54 | size 0.020, power 0.536 | MATCH |  |
| 28 | 330 | per-seed count, 'decided' (fourth pass §2, tex4:328) | '13 of 192 (13 of 282 with R's second rerun, which had none) and, on R's inference, decided one run in twenty (914); the rerun that would test that inference, ... was not run' | Phase 2 13/192, R2 0/90, pooled 13/282; amended part (a) (914, 915) has no record | MATCH | 'under the main line's protocol' here is amended part (a)'s own wording (db22857:866). |
| 29 | 330 | fixed-probe count | 3 of 263; never decided a run | 3 of 263, 1 eligible | MATCH |  |
| 30 | 330 | R's advice | as stated | replicate_report.md Phase 4, 'For the report' | MATCH |  |
| 31 | 332 | UNSTATED labels and the parenthesis (fourth pass §1 item 9) | 4 outright, 3 in part, 34 guessed (parenthesis after it): 20, 12, 2; 3 process | stated 4, in part 3, [A] 20, [B] 12, [C] 2, process 3 (total 44) | MATCH | The parenthesis now follows the guesses; U-6, U-11, U-16, U-23 are guessed entries. |
| 32 | 349 | R's branch (unchanged text) | R's branch carries the first rule | phase4.py byte-identical 464be8d..1f8be04 (§0) and to 550a736, R's head at 42c4557's commit time (§12b) | MATCH |  |
| 33 | 349 | the account of the passes, R's items (changed text) | 'each pass found the previous version's own slips, nearly all in the text added since the pass before (among them a miscount of R's labels, the uncommitted Phase 4 reading, ...)' | both R items first appear in the second version 6c37339 ('on its 44 entries' 'nine'; 'amended Phase 4 reading'), absent from 9bfa85a, and are second_pass.md §1 items 1 and 2 | MATCH | Text against the referee's files, not records. |
| 34 | 349 | the fifth pass's finding (new text) | 'the fifth pass found every number of every pre-registered claim reproducing' | fifth_pass.md:28 'every number of a claim reproduces from its record, and so does every pre-registered verdict, interval and reading'; for R, check_R_v5 had no DISAGREE and its record rows all MATCH (and do here) | MATCH | Text against the referee's files; for R the pre-registered numbers are Phase 2's and Phase 4's, all reproduced in §1-§4. |
| 35 | 356 | 'replicated from the text alone' (fourth pass §1 item 3; v4 rows 3-5) | 'replicated, on our assessment, in session R: 17/20 from the text alone ("partially replicated" by R's bar), then 18/20 with the split's probe and key positions taken from our code, 35/40 pooled ("consistent with the report"' | Phase 2 reading 'partially replicated'; Phase 4 reading 'consistent with the report'; 18/20 used D-1, D-2 | MATCH | 'from the text alone' now covers 17/20 only; 'replicated' marked as the authors' assessment; R's two readings quoted. |
| 36 | 393 | what the first rule could detect (fourth pass §2, tex4:391) | 'only rates near 0.70 or below, and 0.70 itself one time in nine' | P('a lower rate') 0.113 at 0.70, 0.238 at 0.65, 0.404 at 0.60, 0.041 at 0.75 | MATCH | The review's (01bee1f:148) 'or below' restored. |
| 37 | 393 | the amended reading's commit (unchanged) | as stated | db22857's message says the same; phase4.py unchanged | MATCH |  |
| 38 | 430 | the 52 tex6:330 cites (unchanged) | x <= 46/52, size 0.045, power 0.81 | x <= 46, size 0.045, power 0.812 | MATCH |  |
| 39 | 472 | conclusion | 17 of 20 from the text alone | 17/20 | MATCH |  |
| 40 | 477 | next steps | as stated | the amended Phase 4 was not run | MATCH |  |
| 41 | 517 | provenance | commits; 42; 20; 42 | runs 42; 20; 42; R's head at 42c4557's commit time 550a736 (Phase 5 protocol only) | MATCH | 1f8be04 is still the commit of everything the draft reports from R; 550a736 adds no record. |
**Counts: 0 DISAGREE, 0 PARTIAL, 2 NOT RECORD-CHECKABLE, 39 MATCH (41 rows).** In v5 they were 0 DISAGREE, 1 PARTIAL, 2 NOT RECORD-CHECKABLE and 36 MATCH (39 rows).
- **v5's one PARTIAL (tex5:36, the subject "from the report alone") is now MATCH.** The sixth version takes the suggested wording verbatim.
- **Two rows are added for changed text**, both MATCH: tex6:349's R items, and tex6:349's "the fifth pass found every number of every pre-registered claim reproducing" (`fifth_pass.md`:28, 83).
- **The two NOT RECORD-CHECKABLE rows are v4's rows 10 and 11, unchanged.** They are "pre-registered before any run" and the amended reading "written after the referee's review".

**tex6:41, R's status.** "R's Phase~4 are complete" is true at 23:14:04:
- the records hold 42 of 42 jobs under one "42 jobs to run" header;
- R's head, `550a736`, adds no Phase 4 record.

`550a736` is R's Phase 5 protocol, committed 26 minutes before the draft "before any Phase 5 run". The draft does not mention it. Whether Phase 5 was running at 23:14 is not recorded: there is no `records/phase5*` at `550a736` or now. This is not an error, because the draft reports only Phases 2–4. "The sessions' fifth round are running" can be read to include it.

**tex6:36's replication sentence.** Every part of it is MATCH:
- the subject, which now restricts "from the report alone" to the first twenty seeds;
- "had not seen the code when it ran them": Phase 2's records are at `de8df0f` (13:55), before DIFF.md at `00990ea` (14:01);
- 17/20 against 7/20, 10 vs 0;
- 18/20 "after taking the split's probe and key positions from our code" (D-1, D-2);
- 35/40 against 12/40, 23 vs 0;
- 44 details, 34 guesses.

The "44" counts `UNSTATED.md` as of `1f8be04`. At R's head that file has 65 rows. The 21 added rows are Phase 5's guesses, and the draft does not report Phase 5, so the 44 stands (row 7's note).

## Also found

- **In the referee's own files (minor).** `check_R_v5.md`:5 and :56/:236, and `resolution_R_E5.md`:22, say that R's head is "still `1f8be04`; nothing is newer". All of them were committed in `3f7d84b` at 22:50:37 UTC on 12 October, but R had pushed `550a736` at 22:47:46, 2 min 51 s earlier.
  - The claim was true at the fifth version's own time (`c5e4508`, 19:14), which is what v5 row 8 checks.
  - It was stale as a statement about R's branch when it was committed. It should have read "at `c5e4508`'s time".
- **Carried from v5, unchanged on R's branch.** `replicate_report.md`:138 ("includes 0.95") and :27 ("contains the report's 38/40 (0.95)") are both R's errors: Wilson upper 0.9454 and 0.9476 are both below 0.95. The draft does not repeat either one.

## Not checked

- Whether R was ever given the amended prompt in chat, and the actual start time of Phase 4. Neither is recorded (see v4).
- R's Phase 5 (`550a736`). It is a protocol with no records, and the draft does not cite it.
- The other parts of tex6:41's status ("F's S81, G's S83, H's S86 … complete"; "the sessions' fifth round are running"). These belong to the F/G/H/I checks; E's batch 20 is in `check_E19_v6`.
- The PDF.

<details><summary>Full output of <code>python3 docs/review_2026-10/rev9/check_R_v6.py</code></summary>

```

== 0. Which rule: phase4.py at 464be8d (pre-registration), 0364e28 (interim records), 1f8be04 (final)
  464be8d -> 0364e28: design/readings text identical: True; code identical: True
  464be8d -> 1f8be04: design/readings text identical: True; code identical: True
  pre-registered RESULT block at 464be8d: '(not yet run)'
  rule (phase4.py docstring): 'consistent' = Wilson(split, 40 pooled) overlaps [0.835, 0.986] AND McNemar p < 0.05; 'a lower rate' = upper < 0.835; else 'neither'
  phase4.jsonl at 464be8d exists: False
  amended-prompt arms in replicate.py at 1f8be04: COPY_ONLY False ; RESET_ONLY False
  amended prompt db22857:858-884, first line: ## Session R — Phase 4, amended (replaces the Phase 4 prompt above; given to the user in chat on 10 October af
  amended prompt lines with 914/915, 27/32, COPY_ONLY: [866, 869, 873, 875]
  R's branch mentions of the amended prompt (amended|27/32|COPY_ONLY|RESET_ONLY|914 and 915): none
  464be8d 2026-10-10T19:01:33+00:00  Pre-register Phase 4: 40-seed count with the main line's split
  01bee1f 2026-10-10T19:08:20+00:00  Referee V: replication_review.md, review of Session R's replication
  0364e28 2026-10-11T00:02:24+00:00  Phase 4 interim records
  db22857 2026-10-11T03:01:26+00:00  docs/reading: session R's amended Phase 4 prompt, committed (it had been given only in cha
  1f8be04 2026-10-11T04:26:27+00:00  Phase 4 complete: 40-seed count 35/40 vs 12/40 (23 vs 0): consistent with the report
  9657edb 2026-10-11T04:29:29+00:00  Revision 9 draft: fold in session R's Phase 4 (35/40 vs 12/40 pooled, 23 vs 0; 18/20 on fr
  a5f67a2 2026-10-11T03:15:04+00:00  Revision 9 draft: the referee's consolidated second pass, remaining items (R's 914 probe 0
  c66cc76 2026-10-11T07:52:07+00:00  Revision 9 draft: S86 in the index, the running list and H's provenance row
  c5e4508 2026-10-12T19:14:19+00:00  Revision 9 draft: S75's WIDE_SLOW pairing, nine seeds (matches WIDE_SPLIT seed for seed; t
  42c4557 2026-10-12T23:14:04+00:00  Revision 9 draft, sixth version: every item of Referee V's fifth pass (e384a87) — the abst

== 1. Records: integrity and per-arm counts (R's outcome recomputed)
  lines: phase2 42, phase3 20, phase4 42 (interim 21)
  interim records are the first 21 final records unchanged: True
  phase2/phase3 unchanged since 100dced: True True
  phase4.log: 43 lines; 'jobs to run' headers: ['42 jobs to run']
  phase2: duplicate (arm, seed): 0; stored flags/transition disagreeing with the recount: []
  phase3: duplicate (arm, seed): 0; stored flags/transition disagreeing with the recount: []
  phase4: duplicate (arm, seed): 0; stored flags/transition disagreeing with the recount: []
  P2 ORACLE seeds 900-901: bound routed 2/2 Wilson [0.342, 1.000]
  P2 WIN3_SLOW_D8 seeds 900-919: bound routed 7/20 Wilson [0.181, 0.567]
  P2 WIN3_SPLIT_D8 seeds 900-919: bound routed 17/20 Wilson [0.640, 0.948]
  P3 SPLIT_R1_PROBE seeds 900-909: bound routed 9/10 Wilson [0.596, 0.982]
  P3 SPLIT_R2_BODYKEYS seeds 900-909: bound routed 9/10 Wilson [0.596, 0.982]
  P4 ORACLE_ADAM seeds 920-921: bound routed 2/2 Wilson [0.342, 1.000]
  P4 WIN3_SLOW_D8 seeds 920-939: bound routed 5/20 Wilson [0.112, 0.469]
  P4 WIN3_SPLIT_D8_ML seeds 920-939: bound routed 18/20 Wilson [0.699, 0.972]
  Phase 4 arms present: ['ORACLE_ADAM', 'WIN3_SLOW_D8', 'WIN3_SPLIT_D8_ML'] seeds 920 - 939
  any Phase 4 record on 914/915 or 940-951: []

== 2. Validity (ORACLE_ADAM, 920-921)
  ORACLE_ADAM 920: bound True, transition 2400, bound routed True, updates 4800
  ORACLE_ADAM 921: bound True, transition 2400, bound routed True, updates 4800

== 3. The 40-seed pooled count (Phase 2 900-919 WIN3_SPLIT_D8 + Phase 4 920-939 WIN3_SPLIT_D8_ML) and the reading
  pairs 40 (900-939); split 35/40 Wilson [0.7389, 0.9454]; no split 12/40 [0.1807, 0.4543]; discordant 23 vs 0; one-sided exact p = 1.192e-07 (2^-23 = 1.192e-07)
  point estimate 0.875; Wilson upper 0.94540 < 0.95 -> the interval EXCLUDES 0.95; lower 0.7389
  overlap with [0.835, 0.986]: True; READING (phase4.py rule): 'consistent with the report'
  of the 35: from Phase 2 (R's own probe/key positions) 17/20, from Phase 4 (main line's split) 18/20

== 4. 920-939 alone (secondary, printed with no reading in phase4.py)
  WIN3_SPLIT_D8_ML 18/20 Wilson [0.699, 0.972]; WIN3_SLOW_D8 5/20 [0.112, 0.469]; pairs 13 vs 0, p = 0.0001221
  exact one-sided against 0.95 on the 20 fresh seeds: P(Bin(20, 0.95) <= 18) = 0.264 (descriptive)
  ML failure 920: splits 3 at [4800, 12000, 16800]; map [0, 14, 2, 5, 11, 0, 13, 15]; shared channels [0]; per-stream min 0.48
  ML failure 933: splits 3 at [4800, 9600, 19200]; map [4, 9, 14, 5, 1, 8, 15, 4]; shared channels [4]; per-stream min 0.49
  Phase 4 splits 44, on target by labels 44; latest split update 24000 (main line's last check 40,800: D-5 inert)
  WIN3_SLOW_D8 failures over 40: 28; of them with a shared channel: 28
  median transition, bound routed over the 40: split 20400, no split 22800.0

== 5. The main line's pooled count (X 540-559 + L 1540-1559, window_gate) and X's failures
  main line: split 38/40 Wilson [0.835, 0.986]; no split 15/40; 23 vs 0
  X 555: fired splits 3; map [13, 11, 4, 1, 11, 0, 5, 9]; shared [11]
  X 556: fired splits 3; map [13, 0, 10, 8, 13, 6, 9, 7]; shared [13]

== 6. Probe-versus-evaluation blocks, as phase4.py's docstring defines them
  Phase 2 WIN3_SPLIT_D8 900-919 (per-seed probe): 192 checks, blocked 13 (12 otherwise eligible), runs [906, 914] (not bound routed [914]); converse 0; checks not a multiple of 2400: 0
  R1 SPLIT_R1_PROBE 900-909 (fixed probe): 91 checks, blocked 1 (1 otherwise eligible), runs [905] (not bound routed []); converse 0; checks not a multiple of 2400: 0
  R2 SPLIT_R2_BODYKEYS 900-909 (per-seed probe, body keys): 90 checks, blocked 0 (0 otherwise eligible), runs [] (not bound routed []); converse 0; checks not a multiple of 2400: 0
  Phase 4 WIN3_SPLIT_D8_ML 920-939 (fixed probe): 172 checks, blocked 2 (0 otherwise eligible), runs [] (not bound routed []); converse 0; checks not a multiple of 2400: 0
  fixed probe pooled (R1 + Phase 4): 3 of 263 checks, 1 eligible
  per-seed probe pooled (Phase 2 + R2): 13 of 282 checks; Phase 2 alone 13 of 192 as the draft quotes
  seed 914 (Phase 2): splits [14400] ; blocked checks (u, probe, eval): [(16800, 0.96875, 0.872), (19200, 0.984375, 0.882), (21600, 0.96875, 0.877), (24000, 1.0, 0.873), (26400, 0.953125, 0.884), (28800, 0.953125, 0.878), (31200, 0.953125, 0.884), (33600, 0.953125, 0.887), (36000, 0.953125, 0.88), (38400, 0.953125, 0.871), (40800, 0.96875, 0.878), (43200, 0.96875, 0.875)]
  Phase 2 seed 906: blocked 1, eligible 1, bound routed True
  Phase 2 seed 914: blocked 12, eligible 11, bound routed False
  914 rerun under the main line's protocol (amended part (a)): none in any record

== 7. Calibration: what the pre-registered rule could detect; the amended rule's numbers
  'a lower rate' needs Wilson upper < 0.835: x <= 28/40; with Phase 2's 17/20 pooled, <= 11 of 20 new
  true 0.95: P('a lower rate' | 17/20 kept) = 0.000; P(upper >= 0.835, i.e. not 'lower') = 1.000
  true 0.85: P('a lower rate' | 17/20 kept) = 0.001; P(upper >= 0.835, i.e. not 'lower') = 0.999
  true 0.8: P('a lower rate' | 17/20 kept) = 0.010; P(upper >= 0.835, i.e. not 'lower') = 0.990
  true 0.75: P('a lower rate' | 17/20 kept) = 0.041; P(upper >= 0.835, i.e. not 'lower') = 0.959
  true 0.7: P('a lower rate' | 17/20 kept) = 0.113; P(upper >= 0.835, i.e. not 'lower') = 0.887
  true 0.65: P('a lower rate' | 17/20 kept) = 0.238; P(upper >= 0.835, i.e. not 'lower') = 0.762
  amended (b): 'below 0.95' if x <= 27/32: size 0.020; power at 0.85 0.536, at 0.80 0.796
  one-sample, 0.85 vs fixed 0.95 (alpha 0.05 one-sided, power 0.8): n = 52, x <= 46
  two-sample (two arms), 0.85 vs 0.95, alpha 0.05 two-sided, power 0.8: about 141 per arm (normal approximation) -- R's 'on the order of 150 per arm'
  two-sample (two arms), 0.85 vs 0.95, alpha 0.05 one-sided, power 0.8: about 111 per arm (normal approximation) -- R's 'on the order of 150 per arm'

== 8. CPU and wall time in R's records
  phase2: CPU-like fields none; torch ['2.14.1+cu130']; sum wall_s 37.4 h (runs 42)
  phase3: CPU-like fields none; torch ['2.14.1+cu130']; sum wall_s 15.2 h (runs 20)
  phase4: CPU-like fields none; torch ['2.14.1+cu130']; sum wall_s 37.2 h (runs 42)
  R's prose: replicate.py:36 'PyTorch 2.14.1 (CPU), 37.4 h' (Phase 2); replicate_report.md:109 '4 workers x 1 thread'

== 9. When Phase 4 started (simulated from wall_s; 4 workers, jobs in replicate.py's order, imap_unordered)
  simulated completion order equals the record order: True
  21st completion at +4.71 h, 22nd at +5.09 h, last at +9.40 h
  start no later than 2026-10-10T19:20:00.773368+00:00 (interim commit) and 2026-10-10T19:02:43.586629+00:00 (final commit); if the interim commit preceded the 22nd record, after 2026-10-10T18:56:50.830896+00:00
  pre-registration 2026-10-10T19:01:33+00:00; referee's replication review 01bee1f 2026-10-10T19:08:20+00:00

== 10. UNSTATED.md labels (tex6:36, tex6:332)
  [(none: process)]: 3 ['U-42', 'U-43', 'U-44']
  [A]: 20 
  [A in code, stated in report]: 1 ['U-3']
  [B]: 12 
  [C]: 2 ['U-10', 'U-11']
  [stated]: 4 ['U-4', 'U-8', 'U-9', 'U-40']
  [stated + A]: 2 ['U-13', 'U-25']
  total 44

== 11. Provenance row tex6:517
  de8df0f: Phase 2 complete: WIN3_SPLIT_D8 17/20 vs WIN3_SLOW_D8 7/20 (10 vs 0)
  00990ea: DIFF.md: discrepancies against the main line (Phase 3)
  100dced: Phase 3 reruns and replicate_report.md
  runs: Phase 2 42, Phase 3 20, Phase 4 42

== 12. Numbers the fifth version's new wording rests on
  origin/claude/replicate-R head: 550a736 (NEWER: see 12b)
  Wilson(35, 40) upper 0.94540: rounds to 0.945; '0.945, just below 0.95': True
  Wilson(17, 20) = [0.6396, 0.9476] (upper < 0.95); Wilson(55, 60) = [0.819, 0.964]
  Fisher two-sided 17/20 vs 38/40: p = 0.322
  P(Bin(20, 0.90) >= 18) = 0.677 (fails a true 0.90 recipe 0.323, 'one time in three'); P(Bin(20, 0.95) <= 17) = 0.0755
  P('a lower rate' | 0.70) = 0.113 = 1 in 8.8 ('one time in nine'); at 0.65 0.238, 0.60 0.404, 0.75 0.041
  one-sample n = 52 against 0.95: x <= 46, size 0.045, power at 0.85 0.812; '46/52' at tex6 [430]; section label sec:method at tex6 [410]
  R2 (the second rerun) blocked 0 of 90; Phase 2 13 of 192; per-seed pooled 13 of 282 = 192 + 90; 914: 12 blocked, 11 eligible
  914 from 16,800 on: probe 0.953-1.000 at 12 checks; evaluation at those updates 0.871-0.887; final map [5, 15, 3, 5, 13, 6, 0, 8]
  replicate_report.md lines on the interval: [(27, '- 17/20 sits one run below the threshold fixed before the run, and its interval contains t'), (138, '- The interval excludes rates below 0.74 and includes 0.95.')]
  de8df0f 2026-10-10T13:55:49+00:00 Phase 2 complete: WIN3_SPLIT_D8 17/20 vs WIN3_SLOW_D8 7/20 (10 vs 0)
  00990ea 2026-10-10T14:01:24+00:00 DIFF.md: discrepancies against the main line (Phase 3)
  464be8d 2026-10-10T19:01:33+00:00 Pre-register Phase 4: 40-seed count with the main line's split
  phase4.py:19-20 and :47: ['changed one of the two and moved nothing (9/10 -> 9/10 both). The 920-939 count alone, which is', "entirely under the main line's specification, is printed beside it, as a secondary count.", "Secondary, 920-939 only (all under the main line's split): WIN3_SPLIT_D8_ML 18/20 [0.699, 0.972];"]
  amended prompt part (a): ["(a) The failing seeds: rerun WIN3_SPLIT_D8 on 914 and 915 with the main line's protocol."]

== 12b. R's branch at the sixth version's commit time, and the sixth version's changed wording
  42c4557 committed 2026-10-12T23:14:04+00:00; R's head then: 550a736 2026-10-12T22:47:46+00:00 Phase 5 protocol: copy vs reset vs one delta channel (seeds 940-959)
  R's head now: 550a736 2026-10-12T22:47:46+00:00
  commits on R after 1f8be04 up to that head: 1: ['550a736 2026-10-12T22:47:46+00:00 Phase 5 protocol: copy vs reset vs one delta channel (seeds 940-959)']
  phase5 files at 550a736: ['phase5.py']; any Phase 5 record (records/phase5*): []
  phase4 records unchanged 1f8be04..550a736: True; phase4.py unchanged: True
  UNSTATED.md rows at 550a736: 65 (44 at 1f8be04 unchanged: True; new 21: U-50..U-70)
  draft's R provenance row cites: ['de8df0f', '00990ea', '100dced', '464be8d', '1f8be04', '1f8be04']
  tex6:36 subject: 'A clean-room replication, from the report alone for its first twenty seeds, by a session that had not seen the code when it ran them': True
  'from the report alone' governs only the first twenty seeds (Phase 2, 900-919, 20 split runs, records de8df0f 10-10 13:55; DIFF.md 00990ea 10-10 14:01)
  9bfa85a (first draft): 'on its 44 entries' 0; 'amended Phase~4 reading' 0
  6c37339 (second version): 'on its 44 entries' 1; 'amended Phase~4 reading' 1
  second_pass.md items: [(68, '**1. tex2:327: the 44 entries.** "By R\'s own labels on its 44 entries, the text ... was am'), (73, '**2. tex2:325, 344, 388, 470: the Phase 4 reading.** "The amended Phase 4 reading was issu')]

== 13. Where the sixth version says it (tex6 line numbers; tex5 for comparison)
  '35/40': tex6 [36, 62, 289, 330, 356]; tex5 [36, 62, 289, 330, 356]
  'includes \\$0\\.95\\$': tex6 [330]; tex5 [330]
  'reaches \\$0\\.945\\$': tex6 [330]; tex5 [330]
  'one time in nine': tex6 [330, 393, 430]; tex5 [330, 393, 430]
  '3 of 263': tex6 [207, 330]; tex5 [207, 330]
  '13 of 192': tex6 [330]; tex5 [330]
  '13 of 282': tex6 [207, 330]; tex5 [207, 330]
  'never saw the code': tex6 []; tex5 []
  'had not seen the code': tex6 [36]; tex5 [36]
  "main line's protocol": tex6 [330]; tex5 [330]
  "split's probe and key positions": tex6 [36, 62, 356]; tex5 [36, 62, 356]
  "on R's inference": tex6 [207, 330]; tex5 [207, 330]
  '150 seeds': tex6 [330]; tex5 [330]
  'as two arms against each other': tex6 [330]; tex5 [330]
  '464be8d': tex6 [330, 517]; tex5 [330, 517]
  'db22857': tex6 [330, 349, 393]; tex5 [330, 349, 393]
  'from the text alone': tex6 [25, 62, 356, 472]; tex5 [25, 62, 356, 472]
  "R's records carry no CPU": tex6 [41]; tex5 [41]
  'carries the first rule': tex6 [349, 393]; tex5 [349, 393]
  'queue is empty': tex6 []; tex5 [41]
  'from the report alone for its first twenty seeds': tex6 [36]; tex5 []
  'when it ran them': tex6 [36]; tex5 []
  'fifth round': tex6 [41]; tex5 []
  'batch~20': tex6 [41]; tex5 []
  'Phase~5': tex6 []; tex5 []
  "a miscount of R's labels": tex6 [349]; tex5 [349]
  'the fifth pass found': tex6 [349]; tex5 []

== 14. Rows (each anchored on a fragment of its tex6 line) and counts
   1 | tex6:330 | anchor ok | NOT RECORD-CHECKABLE | pre-registration | recomputed: no phase4.jsonl at 464be8d; rule and code unchanged to 1f8be04; replayed start no later than 19:02:43 UTC vs 464be8d 19:01:33 | Unchanged from v4 row 10: the records carry no timestamps.
   2 | tex6:330 | anchor ok | NOT RECORD-CHECKABLE | the amended reading's history | recomputed: db22857 10-11 03:01 after 0364e28 10-11 00:02 (21 records); no amended arm, seed or rule in R's records | Unchanged from v4 row 11: 'written after the referee's review' rests on chat; the rest is checkable and matches.
   3 | tex6:25 | anchor ok | MATCH | title | recomputed: 17/20, Phase 2
   4 | tex6:36 | anchor ok | MATCH | the abstract's subject 'from the report alone' (v5 row 1, PARTIAL; changed text) | recomputed: 17/20 = Phase 2, seeds 900-919 (de8df0f 13:55), before DIFF.md (00990ea 14:01); 18/20 = Phase 4, 920-939, D-1 and D-2 from our code; 35/40 pooled over both | v5 row 1's suggested wording, verbatim: 'from the report alone' now governs the first twenty seeds only, and the 18/20 and 35/40 are said to use our code. The v5 PARTIAL is resolved.
   5 | tex6:36 | anchor ok | MATCH | the abstract's 'had not seen the code' (fourth pass §1 item 2; changed text) | recomputed: Phase 2 records de8df0f (13:55) before DIFF.md 00990ea (14:01); Phase 4 pre-registered at 464be8d (19:01) | Same claim as tex5:36 ('when it ran its first twenty seeds'), the antecedent moved into the subject.
   6 | tex6:36 | anchor ok | MATCH | abstract counts | recomputed: P2 split 17/20, no split 7/20; P4 18/20; pooled 35/40 vs 12/40, 23 vs 0
   7 | tex6:36 | anchor ok | MATCH | UNSTATED count | recomputed: 44 entries at 1f8be04; [A] 20 + [B] 12 + [C] 2; at R's head when 42c4557 was committed (550a736) UNSTATED.md has 65 rows, the 44 unchanged plus 21 for Phase 5 (U-50..U-70) | See §10 for the label tally. The 44 are the entries through Phase 4, the replication the draft reports (provenance 1f8be04); the Phase 5 entries were logged after it and belong to a phase the draft does not report.
   8 | tex6:41 | anchor ok | MATCH | R's status at 42c4557 (changed text: 'every session's queue is empty' removed) | recomputed: 42 of 42 jobs, one '42 jobs to run' header; R's head at 42c4557's commit time 550a736 (2026-10-12T22:47:46): 1 commit after 1f8be04, the Phase 5 protocol (phase5.py, UNSTATED.md), no Phase 5 record; phase4 records and phase4.py unchanged | R's Phase 4 is complete. R's Phase 5 protocol (copy vs reset vs one delta channel, 940-959) was committed 26 min before the draft, 'before any Phase 5 run', and is not mentioned; whether it was running at 23:14 is not recorded. Not an error: the draft reports Phase 4, and 'the sessions' fifth round' may be read to cover it. The removed 'every session's queue is empty' (v5 row 8's note) is gone.
   9 | tex6:41 | anchor ok | MATCH | CPU | recomputed: no CPU, host or device field in phase2/3/4 (§8)
  10 | tex6:62 | anchor ok | MATCH | index row ('the main line's protocol', fourth pass §2 / v4 row 4) | recomputed: WIN3_SPLIT_D8_ML = D-1 (fixed probe) + D-2 (32 body keys), phase4.py:5-9 | R's own wording ('as the main line specifies it').
  11 | tex6:207 | anchor ok | MATCH | fixed-probe count | recomputed: R1 + Phase 4: 3 of 263, 1 eligible (905, bound routed)
  12 | tex6:207 | anchor ok | MATCH | per-seed count and '914' (fourth pass §2, tex4:205) | recomputed: Phase 2 + R2: 13 of 282; 914: 12 blocked, 11 eligible, bound routed False | Pooled the same way as the fixed-probe 263 (both include a 900-909 rerun); 'on R's inference' marks the attribution.
  13 | tex6:207 | anchor ok | MATCH | 914's readings (unchanged text) | recomputed: probe 0.953-1.000, evaluation 0.871-0.887
  14 | tex6:330 | anchor ok | MATCH | Phase 2 (unchanged text) | recomputed: 17/20 [0.640, 0.948]; 7/20; 10 vs 0, p = 0.00098
  15 | tex6:330 | anchor ok | MATCH | against the report (unchanged) | recomputed: p = 0.322; 55/60 [0.819, 0.964]
  16 | tex6:330 | anchor ok | MATCH | Phase 2 bar | recomputed: P(Bin(20, 0.9) <= 17) = 0.323
  17 | tex6:330 | anchor ok | MATCH | Phase 4 design (v4 rows 4, 18) | recomputed: D-1, D-2 adopted; D-4..D-10 R's own (D-10: map on 2048 held-out sequences)
  18 | tex6:330 | anchor ok | MATCH | validity | recomputed: ORACLE_ADAM 920, 921 transitions [2400, 2400]
  19 | tex6:330 | anchor ok | MATCH | pooled count | recomputed: 35/40 [0.7389, 0.9454]; 12/40; 23 vs 0; p = 1.19e-07
  20 | tex6:330 | anchor ok | MATCH | reading | recomputed: 'consistent with the report'; main line 38/40 vs 15/40, 23 vs 0
  21 | tex6:330 | anchor ok | MATCH | 920-939 alone (fourth pass §2 '920-939 alone'; v4 rows 5, 21) | recomputed: 18/20 vs 5/20, 13 vs 0; phase4.py:19-20 'printed beside it, as a secondary count', :125 prints it, no reading
  22 | tex6:330 | anchor ok | MATCH | failures | recomputed: 920: 3 splits, shared [0]; 933: 3 splits, shared [4]; X 555, 556 each 3 fired splits and a shared channel (§5)
  23 | tex6:330 | anchor ok | MATCH | the interval vs 0.95 (fourth pass §1 item 1) | recomputed: Wilson(35, 40) upper 0.9454 < 0.95; replicate_report.md:138 'includes 0.95' | The fourth pass's fix, and R's own error now attributed to R.
  24 | tex6:330 | anchor ok | MATCH | lower bound | recomputed: lower 0.7389
  25 | tex6:330 | anchor ok | MATCH | 150 vs 52 (fourth pass §2) | recomputed: two-sample normal approx: 141 per arm (two-sided), 111 (one-sided); R says 'on the order of 150' (replicate_report.md:139); one-sample n = 52, x <= 46, size 0.045, power 0.812 (tex6:430) | R's figure is labelled R's; the two questions are now told apart.
  26 | tex6:330 | anchor ok | MATCH | the first rule's power | recomputed: 0.113 = 1 in 8.8
  27 | tex6:330 | anchor ok | MATCH | the amended reading's numbers | recomputed: size 0.020, power 0.536
  28 | tex6:330 | anchor ok | MATCH | per-seed count, 'decided' (fourth pass §2, tex4:328) | recomputed: Phase 2 13/192, R2 0/90, pooled 13/282; amended part (a) (914, 915) has no record | 'under the main line's protocol' here is amended part (a)'s own wording (db22857:866).
  29 | tex6:330 | anchor ok | MATCH | fixed-probe count | recomputed: 3 of 263, 1 eligible
  30 | tex6:330 | anchor ok | MATCH | R's advice | recomputed: replicate_report.md Phase 4, 'For the report'
  31 | tex6:332 | anchor ok | MATCH | UNSTATED labels and the parenthesis (fourth pass §1 item 9) | recomputed: stated 4, in part 3, [A] 20, [B] 12, [C] 2, process 3 (total 44) | The parenthesis now follows the guesses; U-6, U-11, U-16, U-23 are guessed entries.
  32 | tex6:349 | anchor ok | MATCH | R's branch (unchanged text) | recomputed: phase4.py byte-identical 464be8d..1f8be04 (§0) and to 550a736, R's head at 42c4557's commit time (§12b)
  33 | tex6:349 | anchor ok | MATCH | the account of the passes, R's items (changed text) | recomputed: both R items first appear in the second version 6c37339 ('on its 44 entries' 'nine'; 'amended Phase 4 reading'), absent from 9bfa85a, and are second_pass.md §1 items 1 and 2 | Text against the referee's files, not records.
  34 | tex6:349 | anchor ok | MATCH | the fifth pass's finding (new text) | recomputed: fifth_pass.md:28 'every number of a claim reproduces from its record, and so does every pre-registered verdict, interval and reading'; for R, check_R_v5 had no DISAGREE and its record rows all MATCH (and do here) | Text against the referee's files; for R the pre-registered numbers are Phase 2's and Phase 4's, all reproduced in §1-§4.
  35 | tex6:356 | anchor ok | MATCH | 'replicated from the text alone' (fourth pass §1 item 3; v4 rows 3-5) | recomputed: Phase 2 reading 'partially replicated'; Phase 4 reading 'consistent with the report'; 18/20 used D-1, D-2 | 'from the text alone' now covers 17/20 only; 'replicated' marked as the authors' assessment; R's two readings quoted.
  36 | tex6:393 | anchor ok | MATCH | what the first rule could detect (fourth pass §2, tex4:391) | recomputed: P('a lower rate') 0.113 at 0.70, 0.238 at 0.65, 0.404 at 0.60, 0.041 at 0.75 | The review's (01bee1f:148) 'or below' restored.
  37 | tex6:393 | anchor ok | MATCH | the amended reading's commit (unchanged) | recomputed: db22857's message says the same; phase4.py unchanged
  38 | tex6:430 | anchor ok | MATCH | the 52 tex6:330 cites (unchanged) | recomputed: x <= 46, size 0.045, power 0.812
  39 | tex6:472 | anchor ok | MATCH | conclusion | recomputed: 17/20
  40 | tex6:477 | anchor ok | MATCH | next steps | recomputed: the amended Phase 4 was not run
  41 | tex6:517 | anchor ok | MATCH | provenance | recomputed: runs 42; 20; 42; R's head at 42c4557's commit time 550a736 (Phase 5 protocol only) | 1f8be04 is still the commit of everything the draft reports from R; 550a736 adds no record.

  COUNTS: 0 DISAGREE, 0 PARTIAL, 2 NOT RECORD-CHECKABLE, 39 MATCH (41 rows); anchors missing: 0
```

</details>
