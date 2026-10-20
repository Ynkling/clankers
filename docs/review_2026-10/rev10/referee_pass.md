# Revision 10: the referee's pass

*Referee V, 19 October 2026.*

**What was checked.**
- **The revision.** `multichannel_hebbian_report_v10.tex` and `.pdf` on the main line at `585b666`, the merge of `claude/rev10-draft`, whose tex and PDF are `d2b7abd`'s. The text is 591 lines, cited as `tex10:NNN`.
- **The scope.** The text added since Revision 9: `diff` of `c918ff7`'s `multichannel_hebbian_report_v9.tex` against v10, 59 lines added or replaced.
- **The method** is the Revision 9 passes'. Every count, pair, p-value, interval, band and reading was recomputed from the records, with the main line's or the screen's own outcome functions (cut out by ast) and independently with `rlib`. Each screen's pre-registration order, seeds and code identity were checked against git.
- **The other session's passes**, `first_pass.md` and `second_pass.md` with their `recompute_*.py`, were then audited.

**Rules followed.** Nothing outside `docs/review_2026-10/` was changed, nothing was trained, and no session code or other session's script was run.

| check | scope | rows | MATCH | PARTIAL | DISAGREE | not record-checkable |
|---|---|---|---|---|---|---|
| `check_rev10_A` | `test_delta_recipe` X/L/pooled, I's GPU run, R's Phase 5, S75 (seed 839), S75b, the referee history | 82 | 72 | 5 | 1 | 4 |
| `check_rev10_FG` | F's S88 (`a199d5e`), G's S87 (`e46799a`), and the text on them | 107 | 83 | 15 | 6 | 3 |
| `check_rev10_HE` | H's round 5 (`347bd25`), E's batches 20–21 (`419368a`), and the text on them | 39 | 24 | 10 | 3 | 2 |
| `audit_passes` | the two non-referee passes: coverage, follow-through, scripts, method | — | — | — | — | — |
| `check_pdf_v10` | the PDF against the tex | 92.0% of v10-only runs in the PDF, 0.3% of v9-only | | | | |

## Verdict

**Revision 10 does not stand as merged.**

**What holds.**
- Every number of every pre-registered claim reproduces from the records: `test_delta_recipe` X, L and pooled; the GPU run; R's Phase 5.
- So does every count, pair, p-value and reading of the fifth round's screens.
- Every screen fixed its readings before its first record.
- The PDF shows the tex.

**What must change.**
- **Nine statements, in sixteen places, are false against the records or git (§1).** None is a number of a claim. They are:
  - an account of who checked what (§1.1–1.3);
  - a run count made stale by S75's tenth seed (§1.4);
  - a sentence made stale by H's S87a on three lines (§1.5);
  - two descriptions of F's and G's screens (§1.6, §1.7);
  - four provenance rows that stop before the fifth round (§1.8).
- **About fifteen loose ones should change (§2).**

**The two non-referee passes were not an adequate substitute for a referee pass (§3).** Their numbers are right wherever they checked, and their definitions match the main line's.
- **No pass ran on the merged version.**
- **One error the first pass found is still in the text.**
- **Whole kinds of check were not done:**
  - the screens' pre-registration order, seeds and code identity;
  - the provenance rows;
  - the carried-over text;
  - the PDF.
- **Five of the nine false statements were missed** (§1.4–1.8). Two more (§1.1, §1.9) were flagged but not followed through, and two (§1.2, §1.3) came in the merged version, which no pass read.
- **Their independence from the draft's author cannot be shown from git.**

## 1. False statements (must change)

**1. tex10:495: "A referee recomputed the numbers of Revisions~8 and~9 and of this revision's additions from the records and found them to reproduce".**
- **What the referee checked before this pass.** Of this revision's additions, the referee (V) had checked only `test_delta_recipe`'s L half and pooled block, I's GPU run and R's Phase 5 (`round6_A3.md`).
- **It contradicts its own paragraph.** The same paragraph later says the fifth round's screens are not yet refereed.
- **The first pass flagged it** (row 12.1, DISAGREE). The second pass rated the line "MATCH as worded" (row 32) without checking that the old words had gone, and they had not (`audit_passes.md` §2).
- *Fix:* name what each check covered.

**2. tex10:44 and 495: "every number recomputed" and "the fifth round's screens have been recomputed there" (`docs/review_2026-10/rev10/`).**
- On the main line, `rev10/` holds recompute scripts for `test_delta_recipe`, the GPU run and H only.
- F's S88, G's S87, E's batches 20–21 and S75's seed 839 have none.
- 13 of the first pass's MATCH rows rest only on a session's report or README (rows 2.4, 4.2, 4.5, 6.3–6.5, 6.10, 7.5–7.8, 8.4, 10.3).
- These sentences came in `d2b7abd`, which no pass read.
- *Fix:* say what was recomputed, or cite this pass's checks.

**3. tex10:44 and 495: the checks by "a session that did not write" the draft.** Not shown by git, and against what it does show.
- Every draft commit (`ea3afef`, `6e75f69`, `d2b7abd`) and the commit that adds the pass files (`d2b7abd`) carry the same `Claude-Session` trailer, `session_01D7HpBudv8or6aFXGqt2fyj`.
- The pass files first appear in git together with the third version, so git cannot show which came first.
- *Fix:* state what git shows; at most "checked by a separate pass whose files are committed with the third version".

**4. tex10:481: "all but one of the 29 (a register run, at $0.29$) had lost it by 3600".**
- With S75's tenth WIDE_SLOW seed (839, record at `4371824`) there are 30 gated runs.
- WIDE_SLOW 839 held the source at 1200 (layer-2 probe 0.389) and had lost it by 3600 (0.254). The one exception is still REG3 839 (0.291). Recomputed here: 30 runs, 1 above 0.27 at 3600.
- The same paragraph updated "nine seeds" to "all ten" and missed this count.
- *Fix:* "of the 30".

**5. tex10:358, 464 and 486: "a trigger that reads all positions was screened once (S86)".** Stale since H's S87a, which tex10:217 reports: 20 more seeds (2600–2619), ALLMASS 16/20 against BASE 17/20, 1 vs 2, and pooled with S86 24/30 against 26/30.
- `rev9/resolution_G_H5.md` had listed all four lines, but only tex10:217 was updated.
- *Fix:* "screened twice (S86, S87a: 24/30 against 26/30)" in all three places.

**6. tex10:266, F's S88: "$\beta$ falls from 1 to below $0.1$ in every layer within about 1200 updates of its learning rate reaching $10^{-3}$ (by 3600 in the first arm, 3600--8400 in the slow one)".** The sentence contradicts its own parenthesis.
- DELTA_LEARNED's rate is $10^{-3}$ from update 1, and its layer medians first fall below 0.1 at 3600 (10 of 10).
- In the slow arm the collapse comes 1200, 1200, 1200, 1200, 2400, 2400, 3600, 3600, 3600 and 6000 updates after the switch: within 1200 in 4 of 10.
- `report_8.md`:138 says the same and contradicts its own lines 52–54.
- *Fix:* "by 3600 in the first arm, and 1200–6000 updates after the switch in the slow one".

**7. tex10:283, G's S87: SELFKL described as "a penalty on the gate's own distance from uniform at every position".**
- The code computes $\mathrm{KL}(g_t \| \bar g)$, where $\bar g$ is the sequence's own mean gate over the 20 body positions (`explore_g_s87_child.py`:33-35, 116-122).
- Distance from uniform is S83's POS/FLAT term. The table row at tex10:322 ("self-distance") is right.
- *Fix:* "a penalty on each position's gate's distance from the sequence's own mean gate".

**8. The provenance table stops before the fifth round for four sessions.** The caption says the commit column is each session's last report.

| line | row | what it gives | what is missing |
|---|---|---|---|
| tex10:537 | F | ends at S81 (`d5851b1`, 42 runs) | S88: report 8, `a199d5e`, 78 runs (34 + 44) |
| tex10:538 | G | ends at S83 (`076005d`, 82 runs) | S87: report 7, `e46799a`, spec `3c89f87`, 142 runs |
| tex10:536 | H | | S87a and S87c (report 9, `347bd25`) and their 84 runs |
| tex10:535 | E | no row for batches 20 (19 runs) and 21 (20 runs) | the rows themselves |

Neither non-referee pass checked these rows.

**9. tex10:70, V's index row: "the three drafted specifications are not ready".**
- V reviewed G's register draft, H's two-stream draft and I's S75b. tex10:499's "three drafted" are G's, H's and E's, and E's was not reviewed by V before this revision (V's `keyload_options_review.md`, `11a78ee`, came after the merge).
- *Fix:* "the three specifications it reviewed".

## 2. Loose or stale statements (should change)

**Who did what:**
- **tex10:264.** "With a GPU job beside it" is stated as fact. The records show a reboot and no run line, and tex10:426 rightly attributes the account to the author. Attribute it here too.
- **tex10:424.** "Each is corrected where it stands, with the date." `76851eb` corrects RESULT (L) and the README row in place, but neither text carries a date.

**H's key term:**
- **tex10:67 and 217, "the key term's constants are robust".** S87c's drafted arm was at the ceiling (10/10), so the screen could only show a worse variant. A variant with true rate 0.8 still reads "robust" 38% of the time (0.9: 74%). H's report and spec note say so; the draft should.

**E's batches 20–21:**
- **tex10:245, "two validity seeds would both bind with probability about 0.13".** That is for two seeds drawn from these ten (4·3/90). The drafted test's validity seeds are fresh: 0.16 per machine at 4/10, and 0.026 for both machines (`keyload_options_review.md`).
- **tex10:245, "the slow start only delays".** No SLOW oracle run bound by 40,800, and SLOW ended lower on the 7 seeds that bound or came close. E's README says SLOW hurt.
- **tex10:245, "S69".** E's batch-21 screen shares the number with F's S69 (tex10:256).
- **tex10:44, "fifth round (E's batches 20 and 21)".** Batch 20's prompt precedes the round-5 header, which does not name E. This is a label only.
- **tex10:245, batch 20's $N = 256$ change "$-0.02$ to $+0.41$".** The records give −0.0146 to +0.4136, which rounds to −0.01 (minor).

**F's S88:**
- **tex10:266 "never diverges by stream" and "order 1 only at the first layer's VAL"; tex10:377 "collapses to about zero everywhere but the first layer's value positions".**
  - At the end no run diverges by stream, as the screen defines it.
  - Layer-1 VAL is ≥ 0.5 in only 13 of 20 runs.
  - VAL is ≥ 0.5 at layers 2–3 in 8 run-layers (up to 1.85), and CTX is 0.72 in one run (2403, layer 3).
- **tex10:487, limits, memory horizon.** Not updated for S88b: untested by its rule (ROUTED 6/10 at 1×), with its failures stream confusions at 0.5.

**G's S87:**
- **tex10:485, limits item 1, "the hinge reads key identities".** Not updated for S87's label-free frequency hinge (17/20).
- **Its caveat must travel with it.** On this task the frequency cut is exactly "not a value token". All 8 key-term positions are among its 14, which are 5 context and 9 key positions, as report 7 and tex10:283 say.

**S75b:**
- **tex10:481, "about three days for the second stage" on the 3070.** The spec's own per-run hours sum to 54–57 h (2.3–2.4 days). "Three days" is in `STATUS.md` only.

**Carried over from Revision 9 with the wrong referent.** "This revision's first draft" and "this revision's checks" at tex10:42, 245, 364 (commit `9bfa85a`), 443 and 541, and four places in the README. These mean Revision 9's; neither pass read them, being scoped to added text.

**After the merge (not an error at `585b666`).**
- tex10:268, "the two screens close F's programme": F opened round 6 at `8ba99bb` (18:02 UTC, 19 October).
- The round-6 screens generally postdate this revision.

## 3. The two non-referee passes

**What they did right.**
- Every number they marked MATCH that this pass recomputed reproduces, with the same definitions as the main line:
  - BOUND ROUTED as `ok` + transition + one-to-one + every stream ≥ 0.9;
  - exact one-sided McNemar;
  - Wilson;
  - Newcombe's method 10 with continuity-corrected φ;
  - NO_ROOM 36;
  - the D3 sign test;
  - CHECK 174 and the probe-alone count re-implemented.
- Their recomputations agree with the referee's earlier `check_delta_recipe_L.md`, `check_I_report3.md` and `rev9/check_onetoone_v5_out.txt` on every overlapping number.
- They found real errors, and the second version fixed most of them. Among the first pass's items taken correctly (`audit_passes.md` §2):
  - the "one of three seeds" oracle;
  - "within two" pairs;
  - the four merged batch-20 runs;
  - the $N = 256$ creep.

  The items not taken are in the next list.

**Why they were not an adequate substitute.**
1. **No pass on the merged version.** `d2b7abd` changed tex10:44, 481, 495 and 500 after the second pass and was merged 58 s later. Two of those changes are false (§1.2, §1.3).
2. **No follow-through check.** The second pass checked that added words were right, not that flagged words were gone. tex10:495 kept the sentence the first pass marked DISAGREE (§1.1), and tex10:70 (§1.9) and tex10:264 were not taken.
3. **Whole kinds of check were not done.**
   - Not checked: any screen's pre-registration order (readings before records), seed freshness, or code identity (hashes, ast). This pass checked all of these, and all hold.
   - Not checked: the provenance rows (§1.8), the carried-over text, the README, `STATUS.md` or the PDF.
   - Outcomes were read from recorded `transition` and `end.one_to_one` fields rather than re-derived. That cost nothing here, since `rlib` agrees on every record, but it is not the "independent outcome function" the README describes.
4. **They missed five false statements.** These are §1.4–1.8: 29 against 30 gated runs, "screened once", β's timing, SELFKL's description, and the provenance rows.
5. **13 MATCH rows rest on a session's report, not a record** (§1.2).
6. **The files' form.**
   - The three scripts carry placeholder paths and cannot run from the repository.
   - `recompute_H_out.txt` is spliced: it holds a traceback from a crashing version, then part of a rerun. Its "BASE … own pair ok 39" is wrong; the true figure is 41/41, though the draft does not use it.
   - The second pass cites files not in the repository (`worddiff2.txt`, `rev10_first_pass.md`).
   - Both passes' stated counts are off by one:
     - the first pass reads 92/17/11/3 by its first verdicts, against its stated 91/17/11/4;
     - the second pass has 32 rows, not 31;
     - the first pass's "seven distinct errors" leaves out its own 4.3 and 5.18.
7. **Independence is not shown by git** (§1.3).

**In short.** As a check of the numbers the passes did well. As a substitute for a referee pass they fall short on four counts:
- they did not read the merged text;
- they did not verify their own items were taken;
- they skipped pre-registration, seed and provenance checks;
- their independence from the draft cannot be shown.

## 4. For the record

These do not bear on the draft:
- **S75's tenth seed.** WIDE_SLOW 839 matches WIDE_SPLIT 839: SET 0.5333 against 0.5338, neither ROUTED, end η² 0.00029 both, no one-to-one map at any of 20 evaluations. Over ten seeds, 0 of 200 evaluations is one-to-one. tex10:68 and 481's "all ten" hold.
- **The sessions' own report errors.**
  - F's `report_8.md`:138 (the same as §1.6), and its :55–:74 descriptions (`check_rev10_FG.md`).
  - G's `report_7.md`:20 says BASELINE_D "3/10 (key splits)": only 2721 is a key split. 2726 and 2728 are half-routed: margin 0.47, η² by key 0.00.
  - G's `report_7.md`:53, "η² to 0 by 300": the record holds relu(η² − 0.2), not η² itself.
  - I's `report_2.md` §6, "SET differs by at most 0.003": seed 831 differs by 0.00325.
- **E's batch 20 resume check.** The pre-registered 44,400 resume check printed DIFFERS, though every measured field agrees. The draft does not mention it.

## What could not be checked

- **The re-runs.** The bit-for-bit CHECKs and reproductions of S87, S87a/c, S88 and batches 20–21 were read from their files and logs, not re-run.
- **What is in no record.** The server's departure (`STATUS.md` only); the author of the passes beyond what git shows; and the S88a seed-7 smoke run (disclosed, not stored).
- **Anything after `585b666`.** That includes the main line's round-6 prompts (`7c4ceb8`), which change no tex.

## Corrections (20 October, after Revision 10.1)

Found by `revision_10_1_check.md`.
1. **The verdict's list of the nine statements (lines 32–37) names eight.** It omits §1.9, V's index row. The draft's 10.1 copied the list with the same omission (tex101:46, 370).
2. **§2's carried-over referents undercount.**
   - The tex list misses tex10:404 ("This revision's …") and the heading at tex10:358.
   - The README count is wrong: it has five lines with "this revision's first draft", plus :493 and :546, not four. `audit_passes.md`:15 and :88 have the same omission.
3. **§1.8 missed two stale provenance rows.**
   - I's row cites `559b942` and 41 runs, but S75 has had 42 runs since `4371824` (seed 839), and report 2 was updated later.
   - R's row stops at Phase 4 (`1f8be04`); Phase 5 (`31c39fc`, 64 records) is cited nowhere. `check_rev10_A` missed them too.
4. **tex10:443, "the referee now reads … every draft before it is merged", was never flagged.** It was already false for Revision 10, which was merged without a referee pass.

No number in `check_rev10_A`, `check_rev10_FG` or `check_rev10_HE` is wrong.
