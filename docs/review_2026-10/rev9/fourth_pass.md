# Revision 9 draft, fourth version (`c66cc76`): fourth pass

*Referee V, 11 October 2026.*

**What was checked.** `multichannel_hebbian_report_v9.tex` on `claude/rev9-draft` at `c66cc76`, committed 07:52:07 UTC, 564 lines. `tex4:NNN` is a line of its tex.

**The diff.** `a5f67a2..c66cc76` is 10 commits (03:20–07:52), with 39 lines added and 37 removed. Line mapping: tex3:N is tex4:N up to 232, N+2 from 234 to 304, and N+3 from 305 on. The version takes in:
- every item of `third_pass.md` and of `resolution_F3/G3/H3/V3.md`;
- R's Phase 4 (`1f8be04`);
- E's batch 19 (`e3f5809`);
- G's S83 (`076005d`);
- H's S86 (`9b72bf6`).

**How.**
- Every changed or added line was checked against the record it cites. The four new reports were checked against their stores, as S82, S84 and S85 were:
  - R's Phase 4: `check_R_v4`;
  - E's batch 19: `check_E19_v4`;
  - G's S83: `check_G_v4`;
  - H's S86: `check_H_v4`.
- The v3 checks were re-run as v4: `check_F_v4`, `check_split_copy_v4`, `check_appendixB_v4`, `check_G_v4` and `check_H_v4`. Each re-anchors the rewritten rows and keeps every record recomputation.
- The item-by-item verdicts are in `resolution_F_sc_appB4.md`, `resolution_G4.md`, `resolution_H4.md` and `resolution_V4.md`. R's and E's new text has no earlier items; its verdicts are the rows of `check_R_v4.md` and `check_E19_v4.md`.

**Rules followed.**
- Nothing outside `docs/review_2026-10/` was changed, and no training was run.
- The Appendix B check built models read-only from `git archive db22857`.
- Records were read from the branch heads named above.

**The draft moved on during this pass.** `claude/rev9-draft` now has five commits after `c66cc76` (`e3c08e5` to `fabf505`, 15:38:32 to 15:54:51 UTC). They take in this pass's committed files and are **not checked here**.

## Verdict

**Not ready to merge.** Every number of a claim reproduces, and every item the third pass left open is resolved, except the one that needs no fix. What remains is nine false statements and about twenty loose ones. None of them is a count, pair, p-value or interval of a claim.

**Where the nine false statements come from:**
- **Five were written into this version's new text.**
  - Three are in R's Phase 4 sentences: §1 items 1–3.
  - One is in S86's account: item 4.
  - One is the misplaced parenthesis (item 9). It sits inside a sentence that is otherwise now right.
- **One is old text made stale by this version** (item 5, two lines). S86 screened what they still call unscreened.
- **Three are unchanged statements about `test_delta_recipe`** (items 6–8).
  - The main line's Amendment 2 (`55dac50`, 15:36:08 UTC) shows them false, and it postdates this version. The draft's author may not have known that X's run had started.
  - The referee's own earlier files carried the same error; they are corrected (`test_check_delta_recipe.md`, correction of 11 October).

**What should clear it.** One more version, checked against the branch heads and the main line at the time of its commit.

**The fourth-pass checks:**

| check against `c66cc76` | rows | DISAGREE | PARTIAL | not record-checkable | MATCH |
|---|---|---|---|---|---|
| `check_R_v4` (R's Phase 4) | 29 | 3 | 6 | 2 | 18 |
| `check_E19_v4` (batch 19) | 30 | 0 | 8 | 0 | 22 |
| `check_G_v4` (G with S83) | 155 | 0 | 4 | 1 | 150 |
| `check_H_v4` (H with S86) | 112 | 3 | 2 | 1 | 106 |
| `check_F_v4` | 170 | 3 | 3 | 6 | 158 |
| `check_split_copy_v4` §1 | 45 | 0 | 0 | 0 | 45 |
| `check_split_copy_v4` §4 (status rows) | 15 | 3 | 2 | 0 | 10 |
| `check_appendixB_v4` (FALSE / PARTIAL / UNVERIFIABLE / TRUE) | 106 | 0 | 1 | 4 | 101 |

The three DISAGREEs of `check_F_v4` and those of `check_split_copy_v4` §4 are the same three lines (tex4:41, 54 and 254). Each check reaches them through its own scope.

**The items open from the third pass:**

| resolution file | items | RESOLVED | PARTLY | NOT | NEW ERROR |
|---|---|---|---|---|---|
| `resolution_V4.md` | 5 | 3 | 0 | 0 | 1 (tex4:330) + 1 pending (tex4:347, below) |
| `resolution_H4.md` (distinct items) | 6 | 6 | 0 | 0 | 3 (tex4:203; tex4:341, 459 stale) |
| `resolution_G4.md` (with overlaps) | 19 | 18 | 0 | 1 (not record-checkable; no fix needed) | 0 |
| `resolution_F_sc_appB4.md` | 39 | 27 | 4 | 6 | 1 (tex4:330, the same) |

`resolution_F_sc_appB4.md`'s NOT rows are:
- report 6, lines 100–101, which is outside the draft;
- three Appendix B items that need no change (tex4:122, 534, 538);
- two omissions at tex4:526.

**tex4:347, "Every item is addressed in this version".** True of the third pass's items, all of which are addressed. As a sentence about the version it is not yet true while §1 stands.

## 1. False statements

**1. tex4:328: "the interval includes $0.95$".**
- Wilson for 35 of 40 (z = 1.959964) is [0.7389, 0.9454], so it excludes 0.95.
- R's `replicate_report.md` (Phase 4, "What the 40 seeds can and cannot say") makes the same error, and the draft repeats it as R's statement.
- The pre-registered reading is unaffected: it tests overlap with [0.835, 0.986].
- *Fix:* "the interval reaches 0.945 and excludes rates below 0.74". (`check_R_v4`; the bounds were recomputed here.)

**2. tex4:36: "a clean-room replication from the report alone, by a session that never saw the code, bound … $18/20$ … [and] $35/40$".**
- R read the main line's code in Phase 3, before Phase 4 ran.
- The 18/20 and the 35/40 are Phase 4 counts.
- *Fix:* "that had not seen the code when it ran its first twenty seeds". (`check_R_v4`)

**3. tex4:354: "replicated from the text alone ($17/20$, then $18/20$ under the main line's protocol, $35/40$ pooled)".**
- The parenthesis puts Phase 4's counts under "from the text alone".
- Two further problems in the same sentence:
  - "the main line's protocol": R adopted only D-1, D-2 and D-3 (`phase4.py`:5: "the split as the main line specifies it");
  - "replicated": R's mechanical readings were "partially replicated" (Phase 2) and "consistent with the report" (Phase 4).
- *Fix:* keep "from the text alone" for 17/20, and say "the main line's split" (`check_R_v4`).

**4. tex4:203: "the all-position pair copied onto an occupied channel".**
- At seed 2334, update 19200, the pair was 1→6.
- Neither channel 1 nor channel 6 held a stream. The copy put an unused channel onto an unused one, which left the shared channel shared.
- *Fix:* say so (`check_H_v4`; re-read here from `9b72bf6`'s store).

**5. tex4:341 and tex4:459: stale since this version.**
- tex4:341: "a trigger that reads all positions is not yet screened".
- tex4:459: "a trigger on all positions is the next screen".
- S86, folded in at tex4:203 and 439, screened it (8/10 against 9/10, 0 vs 1). (`check_H_v4`)

**6. tex4:41: "its test file checked against the amended spec before any run".**
- Amendment 2 states that X's full run started at 02:12 UTC on 11 October with `a581027`.
- The referee's check `6f2788d` is dated 03:01:36, 49 minutes later.
- The start rests on the amendment's text alone; no log is on any branch.
- *Fix:* "checked against the amended spec (after X's run had started)". (`check_F_v4`, `check_split_copy_v4` §4)

**7. tex4:54: the index row's "not run".**
- At 07:52 X's run had been going for 5 h 40 min, per `55dac50`.
- *Fix:* "running on X since 02:12 (`a581027`); no result".

**8. tex4:254: "(… with the test file checked against the amended spec; not run)".** The same as items 6 and 7.

**9. tex4:330: the parenthesis "(the initialisation scales, the parameter dtype, the evaluation's sampling and the trigger's details among them)".**
- It names guessed entries (U-6, U-11, U-16, U-23) but follows "3 entries of process" (U-42 to U-44).
- The counts 4 + 3 + 34 + 3 = 44 and 20 + 12 + 2 = 34 are right.
- *Fix:* move the parenthesis after "34 to be guessed". (`resolution_V4.md`, corrected; `resolution_F_sc_appB4.md`)

## 2. Loose, overstated or unmarked statements

Each is a sentence that is true in part. The count it rests on is right.

**R's Phase 4 (`check_R_v4`):**
- **tex4:328, "decided one run in twenty (914)".** This is R's inference. The test of it, amended part (a) (914 under the main line's protocol), was not run. tex4:328 says as much elsewhere.
- **tex4:328, "about 150 seeds per arm".** This answers a two-arm question. tex4:428's 52 answers a fixed-rate one. Say which, side by side.
- **tex4:391, "would have detected only rates near $0.70$".** The referee's review (`01bee1f`:148) says "near 0.70 or below". The rule detects 0.70 about one time in nine (tex4:328, tex4:428).
- **tex4:205, "13 of 192".** It is unlabelled; tex4:328 labels the same count "(Phase 2)". With per-seed probes pooled the same way as the fixed-probe count, the figure would be 13 of 282.
- **Phase 4's "920–939 alone" 18/20.** `phase4.py` prints it as secondary with no reading. Where the draft calls it a replication, mark the word as the authors'.
- **The two not record-checkable rows (tex4:328).**
  - Phase 4's start before the amended rule: the records carry no timestamps. The bound puts it by 19:02:43 on 10 October, before the referee's review at 19:08:20.
  - The other row is checkable apart from its timing.

**Batch 19 (`check_E19_v4`):**
- **tex4:235 and tex4:366.** S66's one-pair rule ("carries" if d ≥ −1, ten seeds) reads "carries" 40% of the time when the larger model is at 0.6 against 0.8. tex4:233's caveat ("rules out a large drop, not a small one") is given for four streams only.
- **tex4:235.**
  - The D=128 oracle bound P=16 only 1/2 seeds, which bears on the D=128 lead the draft passes on.
  - The L2 arm was chosen by a tie-break the rule does not contain. E discloses it and warns against "fewer layers is better"; the draft does not.
- **tex4:57, "not capacity-bound at $N = 256$ for 16 keys".** This is the oracle on 2 seeds. The learned recipe binds 0/10 there.
- **tex4:452, "constants held … to four times the width and twice the depth at four streams and to $D = 128$ and two layers at eight".**
  - It restates one-pair screen readings as the method's case.
  - "Twice the depth" is L6, at d = −1 with one failure (283) and slower. E's own verdict is "promising to 4 layers" (README:634), as tex4:57 says.
- **tex4:233, "bound $10/10$" for 2, 4 and 6 applications.** The reading is on bound routed, and L6 is 9/10 routed. Say "bound routed" throughout.
- **tex4:464.** "(i) On CPUs: … eight streams at the larger widths" is partly done. tex4:473 was updated and this line was not.

**G's S83 (`check_G_v4`):**
- **tex4:269, "each S82 reference reproduced bit for bit through 2400 first".** The reproduction was one seed (2100) per reference arm, as `explore_g_s83.py`:12-15 specifies.
- **tex4:269, "both clamp the gate to exactly uniform and the register's write never leaves its initial value".**
  - It is true while the terms are on: the KL is ≤ 1.4e-5 at 300–1200 in every run, and the write at the cue is 0.098–0.112 at 1200.
  - "Never" is not true. At the end of the run the write at the cue is 0.155–0.484 with the position-only term (20 of 20 more than 0.03 from 0.10) and 0.053–0.887 with the flat term (19 of 20). These were recomputed here from `076005d`'s `s83_results.json`, `end.beta.CTX`.
  - The following clause ("where the ratio term … free to commit by stream") is report 6's exploratory account, and it is unmarked.
- **tex4:439, "a term reading positions only, or nothing, fails ($1/20$, $0/20$)" and "which S83 found needs key identities".**
  - The counts are right.
  - S83 tried one form, an absolute KL from uniform. Its own child file warned in advance that a null "may be a weakness of this form" (`explore_g_s83_child.py`:21-23).
  - *Fix:* scope both clauses to that form.
- **tex4:269, "proposed to follow `test_delta_recipe`".** Not record-checkable. It needs no fix if it is meant as the draft's plan (tex4:472).

**H's S86 (`check_H_v4`):**
- **tex4:439, "1 mistarget in 21 at the same rate".** The main line's all-position pair missed 10 of 107 (0.093); S86 missed 1 of 21 (0.048). The two are consistent (P = 0.40), but not the same.
- **tex4:203, "all body positions".** The trigger reads every input position, 98 of them, including the query's stream and key.

**`test_delta_recipe`'s status (`check_F_v4`, `check_split_copy_v4` §4):**
- **tex4:472.** The next-steps list puts `test_delta_recipe` after L's half of `test_split_copy`, but X's half was already running.
- **tex4:492.** The commits are right. The X column's "not run" is not the state.

## 2a. Errors in the sessions' reports, not in the draft

These do not block the draft; they are listed so the reports can be corrected.

- **R's `replicate_report.md`:** the interval statement of §1 item 1.
- **G's report 6:**
  - "η²_key 0.18 at update 1" is the term's value; η² itself is about 0.42.
  - Its 1200-window failures (seeds 2103, 2112) split at 3600, after the slow phase, not during it.
- **F's report 6, lines 100–101:** the seed range attributed to H's store (`check_F_v4`).

## 3. What reproduces

- **R's Phase 4.**
  - 35/40 vs 12/40, 23 vs 0, p = 1.19e-7, "consistent with the report".
  - 920–939: 18/20 vs 5/20, 13 vs 0.
  - ORACLE_ADAM 2/2.
  - `phase4.py` is byte-identical at `464be8d`, `0364e28` and `1f8be04`: R ran its first rule, and no part of the amended reading. The draft now says so (tex4:328, 391).
- **Batch 19.** Every count and reading reproduces. The rules predate the first records (`617681d`).
- **G's S83.**
  - The readings were committed at 03:10:39, before the first records at `15b280b` (03:30:49).
  - 82 records, all ok.
  - Counts 0/20, 1/20, 0/20, 18/20 and 2/2.
  - "The register is needed": 20 vs 0, p = 9.5e-7.
  - "The identity term is needed": 19 vs 0, p = 1.9e-6.
  - The other two readings do not apply, by their rules.
- **H's S86.**
  - BASE 9/10, ALLMASS 8/10 (0 vs 1, "keeps it"), "aims" 20/21, EVAL 9/10.
  - The readings were fixed at `50da719`, before the first record (`71fc5f2`).
  - `9b72bf6`'s fix is report-only.
- **The provenance tables (tex4:486–518).** All 66 commit hashes exist and predate `c66cc76`. Each session's last cited commit is its branch's last before 07:52.
- **Appendix B.** No false item.
- **`test_split_copy`.** All 45 fragments match. L's half is still not on any branch; the only L artefact is the failed start at `84a7e57`.

## What could not be checked

- **X's start time for `test_delta_recipe`.** It is checked against Amendment 2's text only; see `../test_check_delta_recipe_amendment2.md`.
- **Whether L restarted** in a process that has committed nothing.
- **Push times.**
- **The re-runs.** S83's bit-for-bit reproduction and its 61 CHECKs, and S86's CHECKs, were read from their files, not re-run.
- **The PDF.**
- **The five draft commits after `c66cc76`.**

## Corrections (12 October, fifth pass)

1. **§2, the tex4:269 bullet on "never leaves its initial value".**
   - **What this pass said.** It called the statement true "while the terms are on" and gave the KL bound "at 300–1200".
   - **Why that is wrong for the position-only term.** Both KL terms act through update 2400: their weight is 1 at 2400 and 0 at 2401, in all 40 runs. At 2400 the position-only term's KL is above 1e-5 in 13 of 20 runs (maximum 0.068), and its write has moved more than 0.03 in 14 of 20.
   - **What else was wrong.** The bound "≤ 1.4e-5" holds from update 300 through 1200 only. At update 1, every run is at 1.2e-4 to 7.9e-4. At 1200, seed 2109 reads 1.3e-5, 1.4e-5 and 9e-6 (`term.value_at`, re-read here).
   - **The exact scope.** "from update 300 through 1200 (at most 1.4×10⁻⁵; the flat term's through 2400)".
   - **Consequence.** The fifth version adopted this pass's wrong scope as "while they act" (tex5:271). See `resolution_G_H5.md`.
   - The same wording is in `check_G_v4.md` (row tex4:269) and `resolution_G4.md` (row 269 and §3 item 2). Their numbers are right; their scope is not.

2. **§2a missed a second interval error in R's `replicate_report.md`.** At line 27 it says 17/20's interval "contains the report's 38/40 (0.95)". Wilson for 17/20 is [0.6396, 0.9476], which does not contain 0.95. The draft does not repeat it. (Found by `check_R_v5`.)

3. **§1 item 6 (tex4:41, "checked against the amended spec before any run") is withdrawn. It was not false.**
   - The test's process started at 02:12 UTC, but it spent the time until 05:27:31 on verification. `meta.started` is stamped after `verify()` returns (`test_delta_recipe.py`:1321-1326 at `a581027`). The first run began at or after 05:48:43, after the projection (`check_delta_recipe_X.md`).
   - The referee's check `6f2788d` (03:01:36) and second review `16b04f9` (02:59:08) therefore preceded every run.
   - The same error is in `check_F_v4` row P3, `check_split_copy_v4` §4 (tex4:41) and `resolution_F_sc_appB4` §5.
   - **Items 7 and 8 stand ("not run" at 07:52 was false), but on less than this pass said.** Runs had been going for about 2 h, not 5 h 40 min.
   - **The count changes.** The verdict's "nine false statements" were eight. (Found by `check_F_v5`.)
