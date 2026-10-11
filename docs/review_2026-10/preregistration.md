# Pre-registration: what was fixed, when, and what changed after the runs

*Referee's deliverable 2. All commit times are git committer dates (`%cI`), all `+00:00` unless stated. Run start times are the results files' `meta.started`, written with `time.strftime` in the machine's local time and carrying no timezone. Commit times are self-reported: git cannot prove when a commit was made or pushed, or that a branch was never rewritten. The order "fixed, then run" below therefore rests on commit times, ancestry and the records' own `git` fields, all written by the same sessions.*

## Summary

- **Main line.** Both Phase VI tests the report cites, and the Phase V test behind the Revision 7 claim it corrects, had their claims and readings committed before their first run on either machine. No line of any of the three docstrings was removed or changed after a first run: every later commit only appended (amendment, results). The test code changed only as the amendment describes.
- **The headline claim was added mid-test, openly.** test_window_gate's spec and first start registered G2 as *WIN3_SLOW_D8 beats HINGE_D8*. X's run started at 23:30 on 7 October. The amendment that adds the split arm, makes G2 *WIN3_SPLIT_D8 beats HINGE_D8*, adds G4 and adds the eight-stream reading was committed at 00:48 on 8 October, 78 minutes later. It came after batch 17's screen result (9/10, verdict committed 21:50 on 7 October) and before any learned-gate record or any Part D run on either machine. That is a legitimate confirmatory design, and the spec says so ("the full run had started at 2f98f06: added as Part D"). The report's phrase "the test's specification was ... amended, before any Part D run" (tex 136) is accurate. The abstract's "its reading, fixed before the run" is accurate for Part D, not for the test as first started.
- **L's start times only make sense in local time at UTC−6.** Read as UTC, L's `muon_recipe` start (2026-10-05 23:08:48) predates the commit of the code it names (74f5907, 03:07 UTC on 6 October). Read at UTC−6, the offset of the human's own commits from L (for example 631fd62, 7a85d2a), all three of L's starts fall after their test commits and leave enough time for the summed run time. No record states L's timezone; this is an inference.
- **Exploratory branches (E, F, G, H; 21 screens audited).** Every screen's reading lines were committed before its first run record, and none changed afterwards. Two qualifications: (i) session H's S61, whose result the brief and the README describe as "the noise does not matter", pre-registered the opposite reading ("the noise matters"), which did not apply; "does not matter" was written after the run. (ii) Several exploratory screens departed from the prompt's readings in their first commit, before any run: S54(a)'s "≥ 35/40", S55's and S56's designs. Those are disclosed departures, not post-hoc changes.
- **Not pre-registered, and presented in the report as findings:** the copy/reset pooling "19/20 against 4/20" (tex 191, labelled "printed, not claimed", correctly); the abstract's "the optimizer reset ... adds nothing" (no pre-registered comparison of copy-without-reset was cited; the one that exists, S61, is uncited); and the report's own Table 4.

## 1. Main-line tests

### 1.1 test_window_gate (the main result)

| event | commit / time (UTC) | evidence |
|---|---|---|
| spec, readings G1–G3 | `9a769c7` 2026-10-07 14:11:38 | `specs/test_window_gate.md`, "Fixed before the test is written and before any run" |
| test written (docstring = spec) | `2f98f06` 2026-10-07 19:36:39 | test_window_gate.py:1-108 at 2f98f06 |
| batch 17's W_SPLIT verdict (the screen behind Part D) | `992e320` 2026-10-07 21:50:22 (claude/outside-ideas) | explore_out/README.md, batch 17 |
| **X first start** (Parts A–C, at 2f98f06) | `meta.started` 2026-10-07 23:30:25 (X's clock; UTC by every other check) | results/X/window_gate_results.json meta; only the six perfect-gate records survive from it (test_window_gate.py:140-141) |
| amendment: Part D, new G2, G4, the reading | `d05f0ca` 2026-10-08 00:48:06 | specs/test_window_gate.md "Amendment of 8 October 2026" |
| test code for Part D | `886a668` 2026-10-08 07:09:26 | |
| X resume, all 225 learned-gate runs | `meta.starts[0]` 2026-10-08 11:48:32 at 886a668 | X's meta |
| **L start**, everything at once | `meta.started` 2026-10-08 11:33:10 L-local (≈ 17:33 UTC at UTC−6) at 886a668 | L's meta |
| L's file recorded | `1cd20a1` 2026-10-09 05:20:05 | |
| X result and pooled claims recorded | `628eb7c` 2026-10-09 18:42:10 | |
| report Revision 8 | `4a3eebc` 2026-10-09 19:55:37 | |

**Changes to the docstring's claims or readings after the first run: none.** `git diff 2f98f06 886a668 -- test_window_gate.py` removes no docstring line; it adds the amendment (docstring lines 101–136 at 886a668). `git diff 886a668 628eb7c` adds the RESULT sections and removes nothing. The code diff 2f98f06 → 886a668 changes `CLAIMS` exactly as the amendment says (quoted verbatim):

```
-CLAIMS = (("G1", "WIN3_SLOW16", "WIN16_A", "B"), ("G2", "WIN3_SLOW_D8", "HINGE_D8", "C"))
+CLAIMS = (("G1", "WIN3_SLOW16", "WIN16_A", "B"), ("G2", "WIN3_SPLIT_D8", "HINGE_D8", "C"),
+          ("G4", "WIN3_SPLIT_D8", "WIN3_SLOW_D8", "C"))
```

and refactors `one_run` so the window-gate arms share a path with the split arms. CHECK 144, which was to show that the old code's Part A–C records reproduce under the new code, had no learned-gate record to compare (test_window_gate.py:143), so the claim that the refactor is inert for Parts A–C rests on CHECK 143 (threshold 0: the split arm equals WIN3_SLOW_D8 through 6000) and on my reproduction below.

**What the amendment changed, verbatim** (spec, user's text): *"G2 becomes WIN3_SPLIT_D8 beats HINGE_D8 (exact McNemar one-sided). Add G4: WIN3_SPLIT_D8 beats WIN3_SLOW_D8. ... the reading 'eight streams bind without labels or restarts' if WIN3_SPLIT_D8 is RELIABLE pooled (>= 36/40) and G2 and G4 are SHOWN."* And, from the answers recorded before any Part D code: *"Part D runs 43200 steps, matching Part C ... the original G2 (WIN3_SLOW_D8 beats HINGE_D8) is printed, not a claim."* The user's text said 28800 steps; 43200 was chosen before Part D code was written. The original G2 is printed in the docstring's results (15 vs 0, p = 3.1e-5), so demoting it hid nothing.

**What I could not determine.** Whether anyone saw an in-flight Part C learned-gate run before 00:48. The six perfect-gate records are the only records from the first start, and the scheduler ran the longest runs (Part C, about two hours each at X's speed) first after the perfect gates, so a completed learned-gate outcome within 78 minutes is unlikely; the log of that start is not in the repository.

### 1.2 test_muon_recipe

| event | commit / time | evidence |
|---|---|---|
| spec in `specs/` | **none**: `specs/` was created on 7 October; this test predates it | `git log --all -- specs/` |
| test written, claims M1–M3 and the reading in the docstring | `74f5907` 2026-10-06 03:07:00 | |
| **L start** | 2026-10-05 23:08:48 L-local (≈ 05:08 UTC at UTC−6; *before* the commit if read as UTC) at 74f5907 | L's meta |
| **X start** | 2026-10-06 06:19:21 at 74f5907 | X's meta |
| X result | `3e2d171` 2026-10-06 11:45:58 | |
| L file and pooled claims | `2a6ba3d` 2026-10-06 13:07:25 | |

Changes after the first run: none; `3e2d171` and `2a6ba3d` add lines only (git diff shows no removed line). Table 1's caption says main-line specifications "are committed to `specs/` before the test is written" (tex 62, `tab:index` caption); for test_muon_recipe the docstring is the only pre-registration, which is adequate but not what the caption says.

### 1.3 test_early_recipe (Revision 7's E2, which Revision 8 corrects)

Written `9f25dc8` 2026-10-04 03:30:57; X started 06:26:22 at 9f25dc8; X result `9437e01` 09:07:51; L started 2026-10-04 18:38:18 L-local (≈ 00:38 UTC on 5 October at UTC−6) at 9437e01, i.e. after X's result was in the docstring; L result `00ac0f2` 2026-10-05 02:30:26. Docstring changes after the first run: none (the later diffs remove only code in `one_run` and `lr_job` for test_muon_recipe's new knobs, CHECK 129). L ran with X's result visible; the readings were already fixed, so this is not a pre-registration problem, but it means L's run was not blind.

### 1.4 L's clock

| test | L `started` | as UTC−6 | test commit | L's record commit | summed run time ÷ 6 workers | projection |
|---|---|---|---|---|---|---|
| early_recipe | 10-04 18:38 | 10-05 00:38 | 10-04 03:30 | 10-05 02:30 | 1.5 h | 4.6 h |
| muon_recipe | 10-05 23:08 | 10-06 05:08 | 10-06 03:07 | 10-06 13:07 | 2.5 h | 7.2 h |
| window_gate | 10-08 11:33 | 10-08 17:33 | 10-08 07:09 | 10-09 05:20 | 11.0 h | 17.3 h |

Read as UTC, the muon_recipe start precedes its code's commit. At UTC−6 every start follows its commit, and every gap to the record commit exceeds the summed run time divided by the six workers. I conclude L's clock is local UTC−6. A results-file field recording the timezone (or UTC times) would remove the question.

### 1.5 Reproduction of the chain (my one run per test)

`repro.py` ran one recorded run of each Phase VI test through the test's own `run_job`, at 1 thread, torch 2.14.0+cu130, on **Intel(R) Xeon(R) Processor @ 2.10GHz** (this container). The records were made on X's **Xeon @ 2.80GHz**.

| test | arm \| seed | result | time |
|---|---|---|---|
| window_gate | WIN3_SLOW \| 500 (X) | curve, transition and all 167 shared fields equal | 65 s |
| muon_recipe | WIN8_A \| 460 (X) | curve, transition and 300 of 302 shared fields equal; the 2 that differ are host fingerprints (`host/ns_fp`, `host/onednn_bf16`) | 206 s |

So the recorded runs come from the committed code (at this branch's head, identical to the main line for these files), and Adam runs reproduce across X's two Xeon models. Output: `repro_out.jsonl`. I did not run a split-arm or eight-stream run (one run per test was the budget).

## 2. Exploratory branches: "readings committed before any run"

Delegated read-only audits produced the four appendices (`prereg_branches/branch_E.md`, `_F.md`, `_G.md`, `_H.md`; each quotes the reading lines verbatim as first committed). I re-checked against git: S48's reading line at `8ca286e` (explore_window_d8.py:30) and its file history (8ca286e → f91f08b → 2531480, all before the batch's start); S50's reading at `669c3e9` (explore_recipe_ablation.py:38); F's "≥ 35/40" at `9ddda81` (explore_f_window_delta.py:41, :71); that every G screen file has exactly one commit; S58b's selection rule at `69d01dc` (explore_h_eight.py:21-23) before S58's first record; S61's readings at `f630ef2` (explore_h_copy2x2.py:28-30).

| branch | screen | readings committed | first run record committed | reading lines changed afterwards |
|---|---|---|---|---|
| E | S43 | `93bdfe6` 10-06 13:48 | `8e64157` 10-06 19:45 | none |
| E | S44 | `93bdfe6` 10-06 13:48 | `2e7966b` 10-06 15:48 | none |
| E | S48 | `8ca286e` 10-07 14:23 | `aa917ac` 10-07 15:36 | none |
| E | S50 | `669c3e9` 10-08 01:08 | `91b8ba3` 10-08 02:07 | none |
| E | S52 | `669c3e9` 10-08 01:08 | `91f1401` 10-08 03:24 | none |
| E | S64 / S65 | `617681d` 10-09 06:17 | `7f1a71d` 09:20 / `e44e794` 14:06 | none |
| F | S53 | `bb906e5` 10-08 04:14:49 | run started 04:14:55 (log); `91a3d26` 05:07 | none |
| F | S54 | `9ddda81` 10-08 04:35 | `0268500` 10-08 14:16 | none |
| F | S59 | `9ddda81` 10-08 04:35 | `e900359` 10-09 06:02 | none |
| F | S68 / S69 / S70 | `a1112da` 10-09 14:19:50 | `1335f0b` 10-09 14:42 | none |
| G | S55 | `e4d8107` 10-09 14:37 | `4da0c21` 10-09 14:49 | none |
| G | S56 | `a7bbee0` 10-09 14:49 | `7a2fcc9` 10-09 15:38 | none |
| G | S71 / S72 / S73 | `9114843` 10-10 00:23 / `26c4ab1` 00:25 | `191d177` 00:29 / `d212293` 03:40 / `57308ad` 01:15 | none |
| H | S58 | `bbfea60` 10-08 04:14 | `c06f911` 10-08 04:29 | none |
| H | S58b (and its arm-selection rule) | `69d01dc` 10-08 04:21 | `4bf95bf` 10-08 11:30 | none |
| H | S60 | `705000d` 10-08 04:20 | `b6ec6b4` 10-08 16:16 | none |
| H | S61 | `f630ef2` 10-08 20:11 | `145b69c` 10-08 21:26 | none |

**Qualifications, each verified:**

1. **S61's R2 says the opposite of what is now reported.** Pre-registered at `f630ef2` (explore_h_copy2x2.py:29), verbatim: `R2 "the noise matters"        if COPY only >= 4 and COPY_NONOISE only = 0 in COPY vs COPY_NONOISE.` The result (0 vs 0) gives "does not apply". The README row on the branch (`7e9197f`, after the run) reads "the copy alone suffices; **the noise does not matter (0 vs 0)**; the reset adds nothing"; report_3's heading says "the copy needs no noise". "Does not matter" is a post-hoc reading of a 0 vs 0 result on ten seeds. S61 is not cited in Revision 8, but its README wording is what a reader of the branch finds, and the follow-up spec `specs/test_split_copy.md` builds on it.
2. **Pre-run departures from the prompts (disclosed, not post hoc).** F's S54(a) replaced the prompt's paired rule ("≥ 4 vs 0") by "delta better if A_DELTA ≥ 35/40 (31 + 4)" because (a) is unpaired; G's S55 moved from RandHeaderTask without training to fresh training on the P = 8 header, citing a user instruction of 9 October that is not in git; G's S56 used seeds 370-379 instead of 370-389 and counts "KEY failures" among unbound runs only, which decides its delta R2 (three of REG_D's six bound runs bound by key splits; `branch_G.md`).
3. **S58b's arm selection** was fixed before S58's results (`69d01dc` 04:21:59 against the first S58 record at 04:29:26). It defines "best" among devices only, excluding NONE (38/40, which would have ranked second). The rule is pre-registered; the report's sentence "a rule fixed beforehand then carried the two best-scoring arms" (tex 193) is accurate if "arms" is read as "devices".
4. **S50's reading names claim more than the design tests.** Pre-registered at `669c3e9`: `W_RESET "the copy does the work" if <= 4/10`. A reset-only arm failing shows the reset alone is not sufficient; it cannot show the copy alone is. The reading's name travels into the report's "the copy, not the reset" (claims.md).
5. **What git cannot show** (all branches): run start times (records carry durations and the HEAD at the end of the run, not timestamps; the stores' `started` has no timezone, consistent with UTC by adjacency to commits); the content of "+dirty" working trees at run time (G's stores hash every imported module and match the committed sources; E's, F's and H's code SHAs are the same in the dry and real stores).

## 3. What the report presents that was not pre-registered

| statement (tex) | status |
|---|---|
| "The two machines ran disjoint seeds, so the pooled test is a test" (35) | pre-registered as primary in both test docstrings: fine |
| "Three exploratory looks ... agree that the split's row copy is its active part and that the optimizer reset bundled with it adds nothing" (37) | the three looks are pre-registered screens with readings about the reset alone; "adds nothing" is not a reading of any of them (statistics.md §6b) |
| "Pooled over the two ten-seed screens ... 19/20 against 4/20" (191) | post hoc; labelled "printed, not claimed": fine |
| the reset-only control "matched ... seed for seed on L" (164) | a printed comparison, pre-specified as printed: fine |
| "a learned forget gate ... partly clears at the context token" (224) | post-hoc diagnostic, labelled "post hoc": fine |
| Table 4 (`tab:synthesis`, 237-247) | a summary of pre-registered counts and one exploratory count (marked): fine |
| "the recipe's constants survive a fourfold change of width and a doubling of depth" (35) | the pre-registered reading is "scale-free (every d ≥ −1)" at four streams; the abstract's wording adds generality the reading does not test (statistics.md §3) |
