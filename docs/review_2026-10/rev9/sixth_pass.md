# Revision 9 draft, sixth version (`42c4557`): sixth pass

*Referee V, 13 October 2026.*

**What was checked.**
- **The sixth version.** `multichannel_hebbian_report_v9.tex` and `.pdf` on `claude/rev9-draft` at `42c4557` (12 October, 23:14:04 UTC, 568 lines), cited as `tex6:NNN`.
- **The diff.** `c5e4508..42c4557` replaces 24 lines in place, in six commits: `cc379b2`, `b5fd2f2`, `dbd9a7e`, `c916996`, `78b7aaf` and `42c4557`.
- **Two later commits.** The branch has two commits after `42c4557` (`47886dd`, `125809e`, to 01:03:27 on 13 October). Their changed lines are checked in §4.

**How.**
- **Every v5 check re-run as v6 against `42c4557`.** Each script reads the branch heads at 23:14:04 and re-anchors every changed fragment; no row is unanchored. Each keeps every record recomputation:
  - `check_R_v6` and `check_E19_v6`;
  - `check_G_v6` and `check_H_v6`;
  - `check_F_v6`, `check_split_copy_v6` and `check_appendixB_v6`;
  - `check_S75_v6`, `check_split_copy_L_v6` and `check_delta_recipe_X_v6`.
- **The PDF:** `check_pdf_v6_out.txt`.

**Rules followed.**
- Nothing outside `docs/review_2026-10/` was changed, and nothing was trained.
- The Appendix B check built models read-only from `git archive 80bfc1d`.

## Verdict

**Not ready to merge.** One item stands between this version and merging: the draft contradicts itself on L's half of `test_delta_recipe` (§2 item 1).

**What holds.**
- Every number of every claim reproduces from the records: counts, pairs, p-values, intervals, verdicts and readings.
- No row of any check DISAGREEs with the draft.
- Every item of `fifth_pass.md` §2 and §3 is resolved, except three that are only partly resolved, and two of those trace to fixes I proposed in the fifth pass (§1).
- The status lines are right against the branch heads at 23:14:04 (§3).
- The PDF shows the tex: 94.1% of the six-word runs found only in `42c4557` are in its PDF, and none of those found only in `c5e4508`.

**At `125809e`, the head now.** The three partly-resolved items are resolved (§4). What remains there:
- the L-status contradiction;
- one wording point at tex6:528.

**What would make it ready.** Make the L-status statements agree (six statements on five lines). If tex6:528 is also changed, the next version is "ready to merge" with no further check of numbers: neither change touches a number.

## 1. `fifth_pass.md` §2 and §3, item by item

| fifth pass | `42c4557` | status |
|---|---|---|
| §2.1, the abstract's "10 vs 1" on the no-trigger arms | tex6:36: "without a trigger $19/20$ against $7/20$, 12 vs 0; the split helps the Hebbian write, 10 vs 1" | **PARTLY.** The pairs are right (`check_delta_recipe_X_v6`). "Helps" is D4's pooled label, which X alone cannot give; it was the referee's own proposed fix (`fifth_pass.md` correction 5). RESOLVED at `125809e` ("shown on that machine") |
| §2.2, S75's "three window-gate runs had by then finished" | tex6:458: "(whether learned-arm runs had started on the card by then is not recorded)" | **RESOLVED** |
| §2, I's `report_2.md`:31 (outside the draft) | unchanged at I's head `9034b83` | **NOT.** Still contradicted on its own premises. I's `explore_out/README.md`:42 also still says three had finished |
| §3, SINGLE's rise "at most 0.03" | tex6:458: "at most $0.033$" | **RESOLVED** (0.0329) |
| §3, "had lost it by 3600" | tex6:458: "all but one (a register run, at $0.29$) had lost it by 3600" | **RESOLVED.** Only REG3 839 (0.2913) is above 0.27 of the 29 gated runs. The next highest, REG3 832 at 0.2679, is 0.0008 above ORACLE's band, so the count of one depends on the 0.27 line |
| §3, "within 0.02" | tex6:458: "within $0.024$" | **RESOLVED** (0.0235) |
| §3, REG3 "0.535–0.540" | tex6:458: "on nine seeds with one at $0.588$"; one channel "with one at $0.595$" | **RESOLVED** (0.5352–0.5398, 0.5881; SINGLE 838 0.5954) |
| §3, "a late rise of about 0.06 … 0.29 and 0.28" | tex6:458: "a late rise ($0.09$ and $0.08$ after 4800) took the probe no higher than $0.34$ ($0.29$ and $0.28$ at the end)" | **PARTLY.** The rises 0.0911 and 0.0837 and the ends 0.2892 and 0.2778 match. SINGLE 838 peaked at 0.3403 (L3, update 8400), above 0.34 unrounded. RESOLVED at `47886dd` ("no higher than $0.35$"). *Minor:* REG3 839's probe never returned to chance after 1200 (0.379 at 1200, 0.291 at 3600), so for it the "late rise" is a level held since the early transient |
| §3, tex5:478, next step "the window gate without the split … first" | tex6:478: removed | **RESOLVED** |
| §3, tex5:516, I's provenance row | tex6:516: `86652b1`, `b649751`, `4c3984a`; `559b942`; 41 runs | **RESOLVED.** 41 records; `559b942` is I's last S75 commit (report 2, §6) |
| §3, tex5:201, L's CHECKs | tex6:201: "every CHECK comparing with L's own records asserted and passed, the three comparing with Xeon records printed and differing across CPUs, as the specification expects" | **PARTLY.** It says "the three", which the referee proposed (`fifth_pass.md` correction 4). It undercounts: CHECKs 122 and 123 (log:2191, 2195) also print an unequal comparison with 2.10 GHz records, and CHECK 112 (log:2125) one with records whose CPU is not recorded. RESOLVED at `125809e` ("the cross-CPU comparisons printed and unequal, as the specification provides") |
| §3, tex5:477, S81 as a next step | removed | **RESOLVED** |
| §3, tex5:518, V's row | tex6:518: `907eb2c`, `e384a87`; last `e384a87` | **RESOLVED.** `e384a87` (23:11:16) is V's last commit before the draft |
| §3, tex5:349, "Every item is addressed in this version" | tex6:349: "Every item is addressed, and each later version was checked again (… `second_pass.md` to `fifth_pass.md` …): each pass found the previous version's own slips, nearly all in the text added since the pass before (…); the fifth pass found every number of every pre-registered claim reproducing, and withdrew one of its own earlier items" | **RESOLVED.** Every slip it lists is in the pass files; the withdrawn item is `fourth_pass.md` §1.6 |

**The check counts against `42c4557`** (DISAGREE / PARTIAL / not record-checkable / MATCH):

| check | rows | D | P | NRC | M | v5 |
|---|---|---|---|---|---|---|
| `check_R_v6` | 41 | 0 | 0 | 2 | 39 | 0 / 1 / 2 / 36 |
| `check_E19_v6` | 32 | 0 | 0 | 0 | 32 | 2 / 0 / 0 / 29 |
| `check_G_v6` | 159 | 0 | 0 | 1 | 158 | 1 / 2 / 2 / 154 |
| `check_H_v6` | 118 | 0 | 1 | 1 | 116 | 0 / 1 / 2 / 113 |
| `check_F_v6` (live rows) | 186 | 0 | 3 | 6 | 177 | 2 / 9 / 6 / 165 |
| `check_F_v6`, report 7 | 40 | 0 | 0 | 0 | 40 | 0 / 0 / 0 / 40 |
| `check_split_copy_v6` §1 | 50 | 0 | 1 | 0 | 49 | 0 / 1 / 0 / 44 |
| `check_split_copy_v6` §4 | 26 | 0 | 2 | 0 | 24 | 1 / 4 / 0 / 15 |
| `check_split_copy_v6` §6 (the docstring) | 10 | 0 | 1 | 1 | 8 | new |
| `check_appendixB_v6` (FALSE / PARTIAL / UNVERIFIABLE / TRUE) | 107 | 0 | 1 | 4 | 102 | 0 / 0 / 4 / 102 |
| `check_S75_v6` (all rows) | 103 | 1 | 13 | 7 | 82 | 1 / 16 / 7 / 67 |
| `check_S75_v6`, the draft's rows only | 49 | 0 | 3 | 2 | 44 | |
| `check_split_copy_L_v6` | 64 | 0 | 2 | 0 | 62 | 2 / 4 / 1 / 55 |
| `check_delta_recipe_X_v6` | 58 | 0 | 6 | 4 | 48 | 2 / 8 / 2 / 35 |

**The non-MATCH rows that are not about the draft.**
- **S75's one DISAGREE is in session I's `report_2.md`:31**, not the draft. Most of S75's PARTIALs are also in I's own files:
  - `report_2.md`:22 (stale);
  - :72 and :76 (bounds that hold only after rounding);
  - :74 (true at the end only);
  - :29–33 and :45–53.
- **`check_F_v6`'s carried rows** are F's report 6 and other sessions' counts, not the draft.

**The draft's own non-MATCH rows** are those in §1, §2 and §4.

## 2. What remains at `42c4557`

**1. L's half of `test_delta_recipe`: the draft says both "running" and "not run".** This needs a fix.

| line | says |
|---|---|
| tex6:36 | "runs on the other" |
| tex6:54 | "(X; L running)" |
| tex6:360 | "L's half is running" |
| tex6:41 | "L's half is next" and "`test_delta_recipe`'s L half has not run" (the line was rewritten in this version) |
| tex6:496 | "X complete; L to run" |

- **What the record shows.** No L file for the test is on any branch. The main line's round-5 prompts (`0754d61`, 22:35:26, `parallel_sessions_prompts.md`:918) say "L runs `test_delta_recipe`".
- **The fix.** Choose one wording and use it in all five lines, for example "L's half is running (no record yet)" or "L's half is next".

**2. tex6:528, "session~R's `UNSTATED.md` (44 entries)".** Wording.
- R's file at `550a736` (22:47:46, before the draft) holds 65 entries: Phase 5's U-50 to U-70 were added.
- The 44 are the Phase 2 list, which is what Appendix B compiles.
- *Fix:* "(44 entries for Phases 2–4)".

**3. tex6:271, "the register test proposed to follow `test_delta_recipe`".** Not record-checkable, as before, and needs no fix.

**Not errors:**
- G and H both number their new screens S87 (G's S87; H's S87a and S87c). The draft will need to tell them apart when it reports them.
- tex6:41's "the sessions' fifth round" uses H's and the prompts' term; G's records do not use it.

## 3. Branch heads at `42c4557`'s commit (23:14:04 UTC, 12 October)

| branch | last commit | state | the draft |
|---|---|---|---|
| main line | `80bfc1d` (22:51:55) | `bbb600c`, `80bfc1d`: `test_split_copy.py` docstring and L's second-start longrun files; no L `test_delta_recipe` file; no X log | right; L's delta status contradictory (§2.1) |
| E (`outside-ideas`) | `46cb8d0` (22:25:40) | batch 20 IN PROGRESS, 14 of 19 runs saved; the next commit `0b15ced` (23:18) still 14 | tex6:41 "E's batch~20 … running": MATCH |
| F | `cdbb886` (23:07:46) | S88 readings fixed (`40087cb`) and queue; no report after `d5851b1`; F's stores unchanged | right (tex6:514) |
| G | `464f7e4` (23:13:33) | S87 started (`02b3eed`); 4 of 102 runs, no label-free arm recorded, no report 7 | right (tex6:41, 515, 476) |
| H | `8aa677c` (23:01:08) | round 5: the one-to-one recount `5b5deec` (complete), S87a/S87c code and readings `f4fd535`, 4 runs started (`b17517a`), 0 records | "running" right. tex6:441 uses `5b5deec`'s count, which H's row did not cite (PARTIAL in `check_H_v6`); resolved at `47886dd` |
| I | `9034b83` (22:53:23) | **WIDE_SLOW 839 has not landed.** The S75 file is byte-identical from `4c3984a` to I's tip `2fd4c7e` (41 records; WIDE_SLOW 830–838); `559b942` says "839 pending"; I is running S75b and `window_gate` on the 1660 | "the nine seeds run so far (the tenth is queued)": MATCH |
| R | `550a736` (22:47:46) | Phase 5 protocol (`phase5.py`, U-50–U-70), no records | "R's Phase~4 [is] complete": MATCH. Phase 5 is not in the draft, which reports Phases 2–4. See §2.2 for the 44 |
| V | `e384a87` (23:11:16) | the fifth pass | right (tex6:518) |

**WIDE_SLOW's pairing on nine seeds.** Neither arm routes on any of the nine. The largest difference in the end set is 0.00325 (seed 831); the median is 0.00064. The draft says "seed for seed" and gives the ranges, both right. 839 cannot be computed yet.

## 4. `47886dd` and `125809e` (after `42c4557`)

| line | change | status |
|---|---|---|
| 36 | "the split on the Hebbian write shown on that machine, 10 vs 1" | RESOLVED (D4 SHOWN on X, p = 0.0059) |
| 201 | "every CHECK asserted that could be, the cross-CPU comparisons printed and unequal, as the specification provides" | RESOLVED (CHECKs 112, 122, 123, 135, 141, 150) |
| 458 | "no higher than $0.35$"; "all but one of the 29" | RESOLVED (0.3403; 29 gated runs) |
| 513 | H's row adds "the one-to-one recount over the main line's 126 firings (`5b5deec`, round 5)", last commit `5b5deec` | RESOLVED (0 of 126 recomputed here, `check_onetoone_v5`) |

What remains at `125809e` is §2 items 1 and 2.

## 5. For the record: the main line's `test_split_copy.py` docstring

The RESULT (L) block was corrected after `check_split_copy_L` (`7b48cda`):
- **`bbb600c`** (22:51:19) made two changes:
  - "one start" became "one start for the runs", with the counter's reset and both headers reading "start 1";
  - "every inherited and own CHECK passed, asserted" became an account of CHECKs 135, 141 and 150, printed and DIFFERing on L.
- **`80bfc1d`** (22:51:55) committed L's second-start longrun json and `.exit` file. The `.exit` holds `0`, and the json's start, boot and git match the log's second header. The docstring now cites them for "exit 0" and the UTC−6 offset.

Both commits change only the docstring; the code is identical at `ffb0771`, `bbb600c` and `80bfc1d`.

**What the corrected text gets right**, against L's log:
- the exit code;
- the offset;
- the two headers;
- 24 "ALL VERIFICATION CHECKS PASSED" and no failure;
- CHECKs 135, 141 and 150, each printed and DIFFERS;
- CHECKs 130 and 146, IDENTICAL and asserted;
- CHECK 151: 0 disagreements.

**One gap**, the same one this pass found in the draft. "The three that compare with 2.10GHz Xeon records" leaves out CHECKs 122 and 123 (log:2191, 2195), which also print an unequal comparison with 2.10 GHz records, and CHECK 112 (log:2125). The count of three came from the referee's own `check_split_copy_L`, which searched for one phrase (`fifth_pass.md` correction 4). (`check_split_copy_v6` §6.)

## Corrections to the referee's own files, made in this pass

All are dated in `fifth_pass.md` and `resolution_R_E5.md`.
- **R's head.** `resolution_R_E5.md` and `check_R_v5.md` called R's head unchanged at `1f8be04`. It was stale by the time they were committed (R pushed `550a736` at 22:47:46).
- **The commits cited for "running".** `fifth_pass.md` §4 cited G's and H's round-5 commits as showing runs; they were a specification, a recount, and code with readings.
- **`check_H_v5.py`** is not reproducible as committed, because it read a moving head.
- **"The three" cross-CPU CHECKs** were six.
- **"Helps"** is a pooled label.
- **"29 of 30" gated runs** should be 28 of 29.

## What could not be checked

- **X's `test_delta_recipe` log.** It is still on no ref.
- **L's `test_delta_recipe` run.** No record exists, so which of §2.1's two states is true cannot be decided from git.
- **WIDE_SLOW 839.** It has not landed.
- **Re-running session code.** S81's, S83's and S86's CHECKs, and I's summary, were read, not re-run.
- **Anything after `125809e`.**

**Correction (13 October).** §3 lists R's `550a736` as "Phase 5 protocol (`phase5.py`, U-50–U-70), no records", and `check_R_v6.md` (:11, row 8) says the same.
- **What it leaves out.** The commit also changes `replicate.py` (+60 −15: delta memory, the reset operation, the three arms, the Phase 5 job list) and `test_replicate.py` (+95). No number changes.
- **"No records" was true of what had been pushed.** A replay of the run's `wall_s` puts Phase 5's start no later than 22:49:48, so it was already running at 23:14 (`check_R_phase5.md`).
