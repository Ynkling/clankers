# Fourth pass, the referee's own rows

*Referee V, 11 October 2026.*

**What was checked.** The fourth version, `c66cc76` (claude/rev9-draft, 07:52:07 UTC, 564 lines), cited as `tex4:NNN`. The rows are the open items of `resolution_V3.md` and of `third_pass.md` §2 that are the referee's. R's Phase 4 statements are in `check_R_v4.md`.

| item (third pass) | fourth version | status |
|---|---|---|
| tex3:327: R's labels, 20 + 12 + 2 ≠ 37; [C] misdescribed | tex4:330: "stated four outright and three only in part (the gate's input among the four) and left 34 to be guessed, 20 on which it was silent, 12 on which it was ambiguous and 2 carried over from what the report states for another component, with 3 entries of process (the initialisation scales, the parameter dtype, the evaluation's sampling and the trigger's details among them)" | RESOLVED. 4 + 3 + 34 + 3 = 44; 20 + 12 + 2 = 34; [C] described as R defines it. Placement: the parenthesis names guessed details but follows "3 entries of process", so it reads as if it named the process entries. Move it after "34 to be guessed" |
| tex3:36: "the 44 details it had to guess" | tex4:36: "the 44 details it logged (34 of them guesses) are stated in an appendix" | RESOLVED |
| tex3:325, 344, 388: "this version committed it" | tex4:328, 347, 391: "existed in no commit until `db22857`" | RESOLVED. `db22857`, main line, 03:01:26. tex4:328's "after Phase 4 had started, and was not used" is checked in `check_R_v4.md` |
| tex3:511: the V row's commits | tex4:514: "... delta spec reviews (fad1489, 16b04f9); the test check (6f2788d); this revision's checks (e8fbb25, 66334b8, 0d1fa60) & 0d1fa60" | RESOLVED. All four commits exist on claude/review-V and are what the row says. This pass's verdict (`fourth_pass.md`) postdates the version, as it must |
| tex3:344: "Every item is addressed in this version" | tex4:347, the same sentence | PARTLY, pending this pass (`fourth_pass.md`) |

**Correction (11 October, fourth pass).** The tex4:330 row should read **NEW ERROR (minor)**, not RESOLVED with a placement note, as `resolution_F_sc_appB4.md` labels it. The parenthesis names guessed entries (U-6, U-11, U-16, U-23) but stands after "3 entries of process" (U-42 to U-44), so the sentence as written names the wrong entries. The counts 4 + 3 + 34 + 3 = 44 are right; the fix is unchanged (move the parenthesis after "34 to be guessed").
