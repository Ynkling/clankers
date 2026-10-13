# Revision 9 draft, seventh version (`084327e`): seventh pass

*Referee V, 13 October 2026.*

**What was checked.**
- **The seventh version.** `multichannel_hebbian_report_v9.tex` and `.pdf` on `claude/rev9-draft` at `084327e` (13 October, 01:08:04 UTC, 568 lines), cited as `tex7:NNN`.
- **The diff.** `125809e..084327e` is two commits:
  - `bb0e7b0` (01:07:41), the seventh version;
  - `084327e` (01:08:04), "the index row's L status shortened to fit its column … wording only".
- **The two items.** Those of `sixth_pass.md` §2: L's status in all six statements, and the 44 entries.

## Verdict

**Ready to merge.**
- The diff makes exactly the two changes asked for and nothing else. Six tex lines are replaced in place, and the PDF is rebuilt.
- After it, the draft states L's half of `test_delta_recipe` the same way everywhere, and that statement is true against every branch.
- With the sixth pass, that leaves no item open on the draft. Every number of every claim reproduces, as the sixth pass found, and no line those numbers stand on changed here.

## 1. The diff, line by line

| line | `125809e` | `084327e` | status |
|---|---|---|---|
| tex7:36 | "… 10 vs 1) and runs on the other" | "… 10 vs 1) and not yet recorded on the other" | RESOLVED |
| tex7:41 (first) | "L's half is next" | "L's half is not yet recorded" | RESOLVED |
| tex7:41 (second) | "`test_delta_recipe`'s L half has not run" | "… L half is not yet recorded" | RESOLVED |
| tex7:54 | "(X; L running)" | "(X; L unrecorded)" | RESOLVED. The wording is shortened for the column (`084327e`); it means the same |
| tex7:360 | "L's half is running." | "L's half is not yet recorded." | RESOLVED |
| tex7:496 | "X complete; L to run" | "X complete; L not yet recorded" | RESOLVED |
| tex7:528 | "session~R's `UNSTATED.md` (44 entries)" | "(44 entries for Phases~2--4)" | RESOLVED. `UNSTATED.md` holds 44 `U-` rows at `1f8be04` (Phase 4, the draft's provenance) and 65 at `550a736` (Phase 5's U-50 to U-70 added) |

No other line of the tex changed: `git diff --stat` shows 6 insertions and 6 deletions in the tex, plus the PDF.

**L's status is now the same everywhere.** A search of tex7 for every L-status phrase finds these six statements and no others. Their old forms ("running", "next", "has not run", "to run", "runs on the other") are gone.
- tex7:476's next steps ("`test_delta_recipe`'s L half (X's is complete), then …") lists it as next work, which is consistent.

**"Not yet recorded" is true.** As of 01:08:04 no `results/L/*delta*` file is on any branch. The statement needs no claim about whether L is running.

## 2. The PDF

- **Built with the tex.** The PDF was rebuilt in `084327e`, with a creation date of 01:08:04 UTC.
- **Old wording gone.** `check_pdf_v5.py 084327e 125809e` (`check_pdf_v7_out.txt`): none of the 28 six-word runs found only in `125809e`'s tex is in the new PDF.
- **New wording present.**
  - 24 of the 38 runs found only in the new tex are in the PDF. The rest fall in the two table rows (tex7:54, 496), where pdftotext breaks word order.
  - Checked directly, the PDF's text has "not yet recorded" 5 times and "unrecorded" once, as the tex does, plus the Phases 2–4 scope.
  - "runs on the other", "L running", "L to run" and "has not run" appear in neither.

## 3. Status lines at `084327e`'s commit (01:08:04 UTC)

The draft's other status statements still hold against each branch's last commit before 01:08:04:
- **Session I:** `370d6a1` (S75b CHECKs). The S75 file still has WIDE_SLOW 830–838, so "the nine seeds run so far (the tenth is queued)" holds.
- **E:** `16f9245`, batch 20 progress with 17 runs saved, still in progress. "E's batch~20 … running" holds.
- **G, H and F:** `7749e04`, `a25b454` and `fd7b373`, all progress commits. G's last report is still 6, H's 8 and F's 7, as the provenance rows cite.
- **R:** `550a736`, unchanged since the sixth pass.

## 4. For the record

- **The main line's docstring.** The main line corrected `test_split_copy.py`'s RESULT (L) docstring once more at `c3ad769` (01:03:50, before this version). It now names all six printed cross-CPU comparisons (CHECKs 112, 122, 123, 135, 141 and 150) and "ALL VERIFICATION CHECKS PASSED, 24 times", after the referee's correction at `515a414`. That matches L's log (`check_split_copy_v6` §6, `fifth_pass.md` correction 4). It changes no code.
- **The split_copy provenance row.** The draft's row (tex7:495) still cites `80bfc1d` for the docstring correction. That is true but no longer the last such commit; adding `c3ad769` is optional and changes no statement.
- **Open outside the draft.** These do not bear on merging:
  - X's `test_delta_recipe` log, L's half and WIDE_SLOW 839 have not landed.
  - The sessions' own report errors listed in `fifth_pass.md` §6 stand in their files.
