# Revision 9 draft, third version: resolution of the Session F items

*Referee V, third pass, 11 October 2026.*

**What was compared:**
- Third draft: `multichannel_hebbian_report_v9.tex` at `a5f67a2` (03:15:04 UTC). It is cited as `tex3:NNN`.
- Second draft: `f67bc07`, cited as `tex2:NNN`. The two files have the same 561 lines, and all 40 changed lines are one-for-one replacements, so `tex2:N` and `tex3:N` name the same place.
- Records: `origin/claude/explore-F` at `1218a24` (03:04:36 UTC), the last F commit before the third draft. Since `793cbf2` it adds only S81 progress (`s81.log`, `trigger_results.json`). Every store the draft uses is byte-identical to the one at the report commit it names.

**Evidence:** `check_F_v3.py` / `check_F_v3.md`. Row ids:
- `N..` are the second pass's rows;
- `M1`–`M18` are this pass's rows, one for each new or changed F sentence.

Quotes are verbatim from the third draft's tex.

**Status labels:** as in `resolution_F.md`.
- **RESOLVED**: the new text is right against the records.
- **PARTLY**: part of the item is fixed and part stands.
- **NOT**: the text is unchanged, or still wrong.
- **NEW ERROR**: the revision introduced a wrong statement.

## Tally

| scope | items | RESOLVED | PARTLY | NOT | NEW ERROR |
|---|---|---|---|---|---|
| `second_pass.md`, Session F items (§1 items 5, 6, 10; §2 row tex2:252; §3 tex2:355, 338, 357) | 7 | 7 | 0 | 0 | — |
| `resolution_F.md`, every item not RESOLVED there (rows 2, 23; §3 B5, C3 tex2:336, raw keys, tex:349 stall; new errors 1, 2; new PARTIALs tex2:338, 357) | 10 | 9 | 0 | 1 (report 6, outside the draft) | — |
| `check_F_v2.md` PARTIAL rows (N13, N25, N26, N28, N33, N34, N36, report_6.md:100-101) | 8 | 7 | 0 | 1 (the same report 6 row) | — |
| other changed lines in the diff that concern F, the delta write, the premise or the horizon (M2, M4–M8, M12, M17, M18; tex3:344 ×2, 511) | 12 | 11 | 1 (tex3:511, provenance) | 0 | 0 |

`check_F_v3.md` against `a5f67a2`: **168 rows: DISAGREE 0, PARTIAL 1, NOT RECORD-CHECKABLE 6, MATCH 161.**
- The second draft had 162 rows: DISAGREE 0, PARTIAL 8, NOT RECORD-CHECKABLE 7, MATCH 147.
- 12 rows were removed or superseded in v3. All 12 are replaced by M rows, and every M row is MATCH.
- The one PARTIAL left is report 6, not the draft.

**Nothing in the third draft about Session F is false against the records.** Two small wording points remain, neither an error:
- tex3:438's "screened once" (M15);
- tex3:511's V row, which does not cite the commits that hold the second pass and the test check.

## 1. `second_pass.md`, Session F items

| item | second draft | status | third draft and evidence |
|---|---|---|---|
| §1 item 5: the six delta splits | tex2:246, 344 | **RESOLVED** | **tex3:246**: "Six of the twelve splits on delta channels were not on labelled targets: at each the map had already merged; in five of the six the split copied a channel that held no stream or one, and at the sixth (seed 1314, update 9600) it copied a merged channel onto an occupied one; mostly single splits in runs that then bound." <br>**Records** (`eight_results.json`, `DELTA_SPLIT` checks with `key_ok` false; `map_before`, `key_cs`, `key_c0`): <br>• merged before: 6 of 6; <br>• source c\* held 0–1 streams at 5 of 6 (1310@14400, 1311@7200, 1312@7200, 1316@4800, 1317@12000); <br>• at 1314@9600 the map was [6, 5, 4, 6, 10, 13, 2, 2]: c\* = 6 holds streams 0 and 3, and c0 = 10 holds stream 4. So it copied a merged channel onto an occupied one; that is the only occupied destination; <br>• single-split runs: 5 of 6; runs bound: 6 of 6 (M1, M2). <br>**tex3:344**: "(it had; five of the six copied the wrong channel)" (M11). |
| §1 item 6: the forget gate and the horizon | tex2:438, 457 | **RESOLVED** | **tex3:438**: "the one device screened against it, a learned forget gate, was screened once at a fixed length: it raised one delta channel's binding ($9/10$ and $8/10$ against $3/10$) and not a Hebbian one's, and its pre-fixed reading on the horizon read ``it does not''". <br>**tex3:457**: "a learned forget gate raises one delta channel's binding at a fixed length and not a Hebbian one's, its reading on the horizon itself read ``it does not''". <br>**Records:** <br>• S59 ROUTED 9/10, GDN 8/10 against FIXED 3/10. <br>• S69 (Hebbian) 0/10, 0/10 against 0/10, on the same seeds 350–359 and the same task. <br>• S59's reading recomputes to "it does not" (`explore_f_decay.py:30-34`: "the horizon knob matters" needs gap_d − gap_FIXED ≥ 0.15 on ≥ 7/10). Instead gap_d − gap_FIXED is negative on 7/10 for each decay, median −0.46 for both. <br>• L = 37 (`explore_f_decay.py:8-9`). The conditional rerun at another length (D_DELTA_\*) was not made: no such arm is in `decay_results.json` (M15, M16). <br>**Wording point, not an error:** "screened once" covers two screens, S59 on delta and S69 on Hebbian, at one length. The sentence gives both outcomes, so it reads as "at one fixed length". "Screened at one fixed length (L = 37)" would be exact. |
| §1 item 10: S68 "supported the reading" | tex2:336 | **RESOLVED** | **tex3:336**: "(session~F's premise test was on its branch 82 minutes before the commit, and its counts, $0/10$ at eight streams, bore on the claim)". <br>**Records:** S68_SINGLE_S8 BOUND 0/10. All ten of its records were first at `719ac78` (18:33:27 UTC, 9 Oct), 82.2 min before Revision 8's draft `4a3eebc` (19:55:37). <br>The claim it bears on is the corrected one, "no single-channel run had been made at eight streams". S68's readings ("neither") are no longer invoked (M9). |
| §2: the raw-key mechanism | tex2:252 | **RESOLVED** | **tex3:252**: "Raw keys are excluded by their numerical fragility ($\beta\|x\|^2 > 2$ makes $I - \beta x x^\top$ expanding, which is session~F's post hoc reading of the one divergence in eight raw-key oracle runs)". <br>**Records:** 8 raw-key oracle runs (O_B1_RAW 4, O_B025_RAW 4). One diverged: O_B1_RAW\|1300. The mechanism is `report_5.md:62-63`, under "## Anything unexpected" (`report_5.md:58`), i.e. stated after the run. Report 5 does not use the words "post hoc"; the label is the draft's, and fair (M3). |
| §3: the stall as a cause | tex2:355 | **RESOLVED** | **tex3:355**: "the stall Revision~8 reported came from a screen with a learned write strength (and a different split and seeds; the attribution is post hoc)". <br>**Records:** <br>• S54(c) C_DELTA: β = 'learned', seeds 310–319, the noisy split with a reset (`explore_f_window_delta_scale.py:3-5, 12`); <br>• S76 DELTA_SPLIT: β = 1, seeds 1310–1319, the exact copy (`explore_f_eight.py:30-31`); <br>• no S76 arm varies β (M13). |
| §3: Gated DeltaNet's write at 2, 4, 8 | tex2:338 | **RESOLVED** | **tex3:338**: "one delta channel and the delta write with its decay and write strength fixed are now run at two, four and eight streams on the grouped layout, and Gated DeltaNet's learned forget gate on the eight-key header layout". <br>**One delta channel:** S53 SINGLE_DELTA_G 4/10 (layout field "grouped"), S68 2/10 and 0/10 (grouped, `explore_f_premise.py:7-9`), all with β 1 and decay "fixed". <br>**The delta write:** <br>• S80(a) DELTA_A: `delta_cfg` β 1, decay "fixed"; <br>• S80(b) DELTA_B: `delta` β 1, L2; <br>• S76 DELTA_SPLIT: β 1; <br>• decay 0.95 for S80(b) and S76 per `explore_f_configs.py:7-8` and `explore_f_eight.py:17`. <br>**GDN decay:** every arm with one in F's stores (decay ORACLE_GDN and SINGLE_GDN, hebb_decay S69_GDN, probe S59_GDN and S69_GDN) is on HeaderTask(8, S=2), L = 37 (M10). |
| §3: C3 "the same counts" | tex2:357 | **RESOLVED** | **tex3:357**: "session~F's S68 had $0/10$ at eight streams too ($2/10$ at four), but its own reading of that name did not apply". <br>**Records:** S68 S8 0/10, S4 2/10, reading "neither". C3's 0/10 against 10/10 is MATCH in `check_split_copy_v2.md` (M14). |

## 2. `resolution_F.md`, items not RESOLVED there

| `resolution_F.md` item | then | now | third draft |
|---|---|---|---|
| §1 row 2 (= B5): "at each ... not the merged one" | PARTLY, with NEW ERROR 1 | **RESOLVED** | tex3:246, as in §1 item 5 above (M1). The fix is word for word the one suggested. |
| §2 row 23: report 6's "1300–1359 also used by H's S78–S79 (1300–1329)" | NOT (outside the draft) | **NOT** (outside the draft) | `report_6.md:100-101` is unchanged at `1218a24`. H's `two_results.json` holds 1300–1319, and 1320–1329 are in `constants_results.json`. The draft does not repeat it. |
| §3 B5 | PARTLY | **RESOLVED** | as row 2 |
| §3 C3: tex2:336 "supported the reading" | NOT | **RESOLVED** | tex3:336 (M9). The wording follows the suggestion. |
| §3 §3: the raw-key fragility, unmarked | NOT | **RESOLVED** | tex3:252 (M3) |
| §3 §3: tex:349, "the stall ... came from a screen with a learned write strength" | PARTLY | **RESOLVED** | tex3:355 (M13). The wording follows the suggestion. |
| §4 NEW ERROR 1: tex2:246 / 344 | NEW ERROR | **RESOLVED** | tex3:246 (M1), tex3:344 (M11) |
| §4 NEW ERROR 2: tex2:438 / 457, "worked" against the horizon | NEW ERROR | **RESOLVED** | tex3:438 (M15), tex3:457 (M16). The suggested reading-first wording was taken, apart from "screened once" (§1 above). |
| §4 other PARTIAL: tex2:338 (N25) | PARTIAL | **RESOLVED** | tex3:338 (M10) |
| §4 other PARTIAL: tex2:357 (N28; row 21's wording point) | PARTIAL | **RESOLVED** | tex3:357 (M14). The wording follows the suggestion. |

## 3. `check_F_v2.md` PARTIAL rows

| row | tex2 | v3 state | replaced by | status |
|---|---|---|---|---|
| N13 | 246 | REMOVED in v3 | M1, tex3:246 | MATCH |
| N36 | 336 | REMOVED in v3 | M9, tex3:336 | MATCH |
| N25 | 338 | REMOVED in v3 | M10, tex3:338 | MATCH |
| N26 | 344 | REMOVED in v3 | M11, tex3:344 (and M12 for its first half, "it rose in two screens": rose S54(a), S54(b); fell S53, S54(d); mixed S54(c)) | MATCH |
| N28 | 357 | REMOVED in v3 | M14, tex3:357 | MATCH |
| N33 | 438 | REMOVED in v3 | M15, tex3:438 | MATCH (wording point "once") |
| N34 | 457 | REMOVED in v3 | M16, tex3:457 | MATCH |
| report_6.md:100-101 | — | unchanged | — | PARTIAL (outside the draft) |

The NOT RECORD-CHECKABLE rows are unchanged:
- N19 at tex3:250: G's in-run probe.
- tex3:252's 14/20 against 1/20, and S71: session G's.
- Report 6's main-line references, CHECKs and wall times.

## 4. Other changed lines that concern Session F, the delta write, the premise or the horizon

| tex3 | new text (excerpt) | status | evidence |
|---|---|---|---|
| 41 | "a fourth (\texttt{test\_delta\_recipe}), specified, reviewed by the referee, amended once and written, its test file checked against the amended spec before any run" | **RESOLVED** (MATCH) | The commits, in this order and all before `a5f67a2` (M4): <br>• spec `05256c5` (20:39:30, 10 Oct); <br>• review `fad1489` (20:43:31); <br>• Amendment 1 `32c121b` (20:47:35); <br>• test `a581027` (02:12:29, 11 Oct; on the main line, an ancestor of `0282955`); <br>• the test checked against the amended spec, `6f2788d` (03:01:36; `test_check_delta_recipe.md`). <br>No delta_recipe result file exists on any branch. |
| 41 | "E's batch~19 (S66--S67) and F's S81 are running" | **RESOLVED** (MATCH) | At `1218a24` (03:04:36): 12 S81 records (GUARD 9, GUARD_EVAL 3), outcomes BOUND ROUTED 10 and MERGED 2; no "S81 done"; no report 7. No S81 number is quoted in tex3 (lines 41, 203, 252, 470, 507) (M18). |
| 54 | "spec reviewed by the referee and amended; the test written and checked against the amended spec; not run" | **RESOLVED** (MATCH) | as M4 (M5) |
| 252 | "(specified in \texttt{specs/}, reviewed by the referee before any run, amended once on its review, and written, with the test file checked against the amended spec; not run)" | **RESOLVED** (MATCH) | as M4 (M6) |
| 489 | "spec \texttt{05256c5}, Amendment~1 \texttt{32c121b} / test \texttt{a581027} / not run" | **RESOLVED** (MATCH) | tex2:489's "not written" (second pass §1 item 4) is fixed (M7) |
| 427 | "A test should print them side by side, with the count of checks the probe blocked, as \texttt{test\_delta\_recipe} now does." | **MATCH** | `test_delta_recipe.py@a581027`: `probe_blind()` at :1009-1022, printed at :1178 ("probe-blind checks ... read >= ACC_THR on the probe while the held-out ..."). Code, not records: the test has not run (M8). |
| 458 | "One channel of either write rule is run at two, four and eight streams" | **RESOLVED** (MATCH) | Second pass §3, the V item "at every size", is fixed. <br>• One delta channel: 4/10, 2/10 and 0/10 (S53, S68). <br>• One Hebbian channel: F's records have two streams only (S53 SINGLE_HEBB_G 0/10). The four- and eight-stream cells are the main line's 0/20 and 0/10 (tex3:279-280), not checked here (M17). |
| 344 | "and three ranges stated for every run that held for the median run" | **RESOLVED** | The first pass listed three: B6 (F's S80 1.00 on all ten), B7 and B8. B6 is fixed at tex3:250 ("$0.98$--$1.00$ at layer~3, median $1.00$", row N18 MATCH). |
| 344 | "Every item is addressed in this version, which the referee's second pass checked again (\texttt{rev9/resolution\_*.md}; it found the second version's own slips ... corrected in turn)" | **RESOLVED for F** | Every Session F item of the second pass is resolved here (§§1–3). I did not re-check other sessions' items for this sentence. |
| 511 | V row: "delta spec reviews (\texttt{fad1489}, \texttt{16b04f9}); this revision's checks (\texttt{e8fbb25}, \texttt{16b04f9})" | **PARTLY** (provenance, not F's records) | tex3:41, 54 and 252 rest on the test check `6f2788d` (03:01:36), and tex3:344 on the second pass's resolution files: <br>• `fc873d2`, 03:06:56, `check_F_v2` / `resolution_F`; <br>• `66334b8`, 03:13:44, `second_pass.md`. <br>All three precede `a5f67a2` (03:15:04). None is cited, and the V row's last commit is still `16b04f9`. |

Unchanged F text on changed lines was re-located and still matches:
- tex3:36: "not shown worse ... $10/10$, $10/10$ and $9/10$ against $10/10$ ... binds sooner at two and eight" (N1).
- tex3:384: "(it keeps them $9/10$ against $10/10$ with the write strength fixed)" (N31), and "(it shrank it, on the eight-key header layout)", MATCH (gap_d − gap_FIXED median −0.46).
- tex3:465: "the delta rule is not shown worse than the Hebbian one on screens at every size tried" (N35).

The other changed lines are outside Session F's records (G, H, R, L, the appendix):
- 60, 199, 203, 205, 207, 228;
- 261, 263, 265, 316;
- 325, 327, 334;
- 388, 390, 436, 440, 455, 470, 488, 510, 525.

## What I could not check

- **Session G's numbers inside F's sentences:**
  - tex3:252's 14/20 against 1/20, and S71;
  - the randomised-layout in-run probe at tex3:250 and tex3:359.
- **The main-line counts beside F's:**
  - C3, taken from `check_split_copy_v2.md`;
  - the one-Hebbian-channel cells at four and eight streams (tex3:279-280);
  - 69/80 and 38/40.
- **test_delta_recipe.** It is checked from git history and its source only: it has no run.
- **The checks file** behind 2×10⁻⁷ and "2400 updates", as before.
- **Whether S81 is still running** in a process that has committed nothing since `1218a24`.
- **The PDF** was not compared with the tex.
