# Revision 10.2: the referee's check before merge

*Referee V, 21 October 2026.*

**What was checked.**
- **The draft.** Revision 10.2 on `claude/rev10-draft` at `33e0036` (20 October, 21:53:32 UTC; not merged). Its tex is 596 lines, cited as `tex102:NNN`.
- **The diff from Revision 10.1** (`846d0c8`), as changed lines:
  - `multichannel_hebbian_report_v10.tex`: 14 lines replaced in place;
  - `STATUS.md`, `docs/review_2026-10/rev10/README.md`, `tools/readme/parts/header.md`, and the rebuilt `README.md` and `CITATION.cff`;
  - the PDF.
- **The rest of the diff.** It merges this branch up to `0e9e625` (the referee's files) and a one-character change to `tools/readme/build_readme.py`: its pattern now accepts "Revisions". No other file changed.
- **Each changed line** was checked against the records, the branches at `33e0036`'s commit time, and the referee's files.

**Rules followed.** Nothing outside `docs/review_2026-10/` was changed, and nothing was run but `rev10/check_pdf_v10.py`.

## Verdict

**Ready to merge after three one-line edits (§3).** Each is a statement about a branch or a commit, and none is a number of a claim. Every item of `revision_10_1_check.md`'s verdict is resolved.
- The edits can be checked against this file by their diff alone.
- With them, the draft's three statements that 10.2 "was passed by the referee before its merge" (tex102:46, 370 and 447) become true on its merge. The same sentence is in `STATUS.md` and the README header.

**The PDF shows the tex** (`check_pdf_v10.py 33e0036 846d0c8`):
- 94.5% of the six-word runs found only in 10.2's tex are in its PDF;
- 0.9% of those found only in 10.1's tex are (one run);
- of the whole tex, 6062 of 6370 runs are in the PDF.

## 1. The items of `revision_10_1_check.md`'s verdict

| item | 10.1 | 10.2 | status |
|---|---|---|---|
| tex101:381, the learned β's exceptions | "except at some value positions" | tex102:381 "except at some value positions and one context position" | RESOLVED (DELTA_LEARNED 2403, layer 3, 0.72) |
| tex101:485, S75b's second stage | "about two and a half days" | tex102:485 "54--57 hours for the second stage by the specification's own run times" | RESOLVED (the spec's 54.3–57.3 h) |
| STATUS.md:103, the same | "about three days" | "54–57 hours on one card by the specification's run times" | RESOLVED |
| carried referents, tex101:42 | "this revision's first draft … corrected here" | tex102:42 "Revision~9's first draft … corrected and listed" | RESOLVED. It was not wrong in the Revision 9 paragraph (`referee_pass.md` correction 5), and is now unambiguous |
| carried referents, tex101:408 | "This revision's first draft repeated the failure" | tex102:408 "Revision~9's first draft" | RESOLVED |
| carried referents, the README (:64, :378, :497) | "this revision's first draft" | none left in `README.md` at `33e0036`. The remaining "this revision" at README:342 is in the Revision 10 section and means it | RESOLVED |
| tex101:270, "closed session F's fourth round" | | tex102:270 "The two screens closed S88; session~F's branch holds eight reports." | RESOLVED for the round label. **NEW ERROR** for the count (§2.1) |
| tex101:46, "every claim"; the list of nine; "corrects them … and nothing else" | | tex102:46: "every pre-registered claim"; "and an index row"; "corrected them and most of the looser statements" | RESOLVED |
| tex101:44, the "fifth round (E's batches 20 and 21 …)" label | NOT in 10.1 | tex102:44 "the fifth round (F's S88, G's S87, H's S87, I's last S75 arm) and E's batches~20 and~21, which ran beside it" | RESOLVED |
| tex101:546, V's row cites `2c250bd` (NEW ERROR in 10.1) | | tex102:546 cites `480a2de` | RESOLVED. The row's other additions are PARTLY right (§2.2) |
| tex101:370, eight of nine; the S75b length "the author's account" (NEW ERROR in 10.1) | | tex102:370 adds "an index row …"; "restated from the specification's run times" | RESOLVED |
| residual, tex101:544, I's row (41 runs, `559b942`) | | tex102:544: `4371824` (the tenth seed), `403538b`; last commit `403538b`; "pilot; 42" | RESOLVED. 42 S75 records; `403538b` (13 Oct 21:54) is I's last commit touching `report_2.md` before `33e0036`. The row still omits report 3 (§2.3) |
| residual, tex101:545, R's row (stops at Phase 4) | | tex102:545 adds "Phase 5, `test_split_copy` on fresh seeds (protocol `550a736`; report `31c39fc`)", last commit `31c39fc`, runs "42; 20; 42; 64" | RESOLVED (64 Phase 5 records, `check_R_phase5.md`) |
| residual, tex101:408 | | as above | RESOLVED |
| residual, tex101:447, "the referee now reads … every draft before it is merged" | | tex102:447 "The referee now reads every main-line specification before its first run; Revisions~10 and~10.1 were merged before it read them and corrected after, and from Revision~10.2 on a revision is merged only after the referee's pass" | RESOLVED. True of 10 and 10.1 by git; of 10.2 on its merge after this file |
| outside, STATUS.md:75–77 | "a session that had not written it, every number recomputed" | "a checking agent run from the author's own session (git cannot show their independence …), which recomputed `test_delta_recipe`, the GPU run and H's screens and read the other screens' reports" | RESOLVED. It matches `audit_passes.md`. It adds "taken in Revision 10.2, the first revision since 9 passed by the referee before its merge": true on merge after this file |
| outside, `rev10/README.md` | "by a session that did not write it … every number recomputed" | "a checking agent run from the author's own session … its independence cannot be shown … where they recomputed (`test_delta_recipe`, the GPU run, H's screens) and read the other screens' reports" | RESOLVED. One clause is not record-checkable: "the agent had no view of the drafting" (§3, optional) |
| outside, the README header | "the current refereed report" | `tools/readme/parts/header.md`, README:3 "Revision 10.2 is the current report, passed by the referee before its merge" | RESOLVED on merge after this file. The header's date still reads "Status (19 October 2026)" for a 20 October revision (§3, optional) |

## 2. What the changed lines get wrong

**1. tex102:270 (and the README's mirror): "session~F's branch holds eight reports".**
- F's `report_9.md` was first committed at `ebe5bcb0` (19 October, 19:31:23 UTC).
- At `33e0036`'s time F's branch (`530240c`) holds `report_1.md` to `report_9.md`, plus report 9's addendum (`2279ca1`, `5357889`).
- "Eight" was true at 10.1's commit (18:33:26, 19 October) and stale at 10.2's.
- *Fix:* "The two screens closed S88 (report 8); session~F's round-6 report 9 is not in this revision."

**2. tex102:546, V's row: two citations are interim commits.**
- **"Revision~10.1's check (`0e9e625`)".** The check is `70a7992` (`revision_10_1_check.md`). `0e9e625` is an interim commit (the re-run scripts in progress) that came after it. As the row's last commit, `0e9e625` is right: it is V's last commit before `33e0036`.
- **The screen reviews and report checks** list `3ae815b`, `3c772c3`, `dcd549d`, `adecd3d` and `e3749da`.
  - `3c772c3` is an interim commit holding a draft of G's review, which is `dcd549d`.
  - H's screen review, `ee5217b`, is missing.
  - `e3749da` is the interim commit that holds F's report-9 check in its final form; it is right as a citation, but should be named as such.
- *Fix:* "Revision~10.1's check (`70a7992`); the round-6 screen reviews (`3ae815b` F, `ee5217b` H, `dcd549d` G) and report checks (`adecd3d` H report 10, `e3749da` F report 9)", with the last commit `0e9e625`.

**3. tex102:544, I's row: report 3 is cited nowhere in the provenance tables.**
- tex102:183 reports I's GPU run of `test_window_gate`: "session~I, report 3"; 62 records; note `30807ec`, records `148c161`, report `3d7e68e`.
- Neither table cites these commits. The referee's `rev10/referee_pass.md` §1.8 and `revision_10_1_check.md` also missed this; it is noted here as an error of those files too.
- *Fix:* add to I's row "`test_window_gate` on the GTX 1660 SUPER (note `30807ec`; records `148c161`; report 3, `3d7e68e`)". Set the last commit to I's last report commit before `33e0036` (`3d7e68e`, or `403538b` if report 2's later edit is meant), and the run counts to "pilot; 42; 62".

## 3. Edits

**Required before merge.**
1. tex102:270: fix §2.1.
2. tex102:546: fix §2.2.
3. tex102:544: fix §2.3.

Each needs the same change in the README's mirror (rebuilt by `tools/readme/build_readme.py`).

**Optional.**
- **The README header's date.** `tools/readme/parts/header.md`, "Status (19 October 2026)" should be 20 October, or the merge date.
- **`rev10/README.md`, "the agent had no view of the drafting".** Not record-checkable; write "the author states the agent had no view of the drafting".
- **`rev10/README.md`, unchanged lines.**
  - "an independent outcome function" overstates: the scripts read the recorded `transition` and `end.one_to_one`, as `audit_passes.md` §4 found.
  - The first pass's counts (91/17/11/4) are off by one by its own first verdicts (92/17/11/3), as `audit_passes.md` found.

None of these touches the tex's claims.

## Corrections to the referee's own files

- **The I report-3 omission.** `rev10/referee_pass.md` §1.8, `rev10/check_rev10_A.md` and `rev10/revision_10_1_check.md` did not notice that I's report 3 is cited in no provenance row (§2.3).
- **The two corrections to tex10:42 disagree.** `revision_10_1_check.md` lists tex101:42's "this revision's first draft" as a residual (PARTLY), and `referee_pass.md` correction 5 says it was not wrong. The second is right: the line sits in the Revision 9 paragraph. 10.2's "Revision~9's" serves either reading.
