# Registry D: notes

Repo /home/user/clankers, branch claude/review-V (HEAD 0fdb363). I only read the repo; nothing in it was modified. Sources: the module docstrings at HEAD (ast.get_docstring); `git log` and `git show` of every commit that touched each test, specs/test_window_gate.md and README.md/docs/revision7.md; the `verdict` and `meta` fields of results/{X,L}/*.json. Dates are committer dates (%cI). The registry is in /tmp/claude-0/registry_D.tsv: 143 rows plus a header, with 18 tab-separated columns.

## Rows per test (claim id x machine)

| test | phase | claim ids | machines |
|---|---|---|---|
| test_stream_curriculum | V | VALID (gate), C1 (McNemar), C2 (band of SC8_R), C3 (Fisher), READ | X, L |
| test_slow_start | V | PAIRING (gate, CHECK 110), H0, HA, HB, S0, H0S, BAND:HINGE0/HINGE8/HINGE4k16, READ-H0, READ-HAHB, READ-C | X, L |
| test_recipe_scope | V | PAIRING (gate, CHECK 118), Q1-Q5, BAND:SC8_H, READ-alone, READ-both, READ-Q5 | X, L |
| test_early_recipe | V | VALID-A, VALID-B, E1, E2, BAND:WIN_M/WIN16_A/WIN16_M, READ-E1, READ-E2 | X, L |
| test_muon_recipe | VI | VALID-A/B/C (X, L only), M1-M3, BAND:WIN_M/WIN16_M/WIN8_M | X, L, pooled |
| test_muon_recipe | VI | READ | pooled only (pre-registered as pooled) |
| test_window_gate | VI | VALID-A/B/C (X, L only), G1, G2 (amended), G3 (bound), G4, BAND for all 9 arms | X, L, pooled |
| test_window_gate | VI | READ | pooled only (pre-registered as pooled) |

- **No pooled rows for Phase V.** The four Phase V tests pre-registered verdicts per machine only. Their `--also` gives pooled counts that are explicitly "descriptive". test_early_recipe's pooled block (test_early_recipe.py:201) is labelled "Pooled, DESCRIPTIVE (not a claim, no p)", so I excluded it.
- **Pooled validity has no rows.** For Phase VI it is derived from the per-machine rows: a pooled claim is UNTESTED only if its part is invalid on both machines.

**Excluded as "printed, not a claim", descriptive or diagnostic:**
- every DIAGNOSTICS block;
- the stream_curriculum Fisher test on all SC8 vs all D8 (printed alongside C1; its value is noted in the C1 rows);
- early_recipe's BOUND WIN16_M vs WIN16_A and its median transitions;
- muon_recipe's BOUND comparisons and median transitions;
- window_gate's original G2 (WIN3_SLOW_D8 vs HINGE_D8), WIN3_RESET_D8 vs WIN3_SLOW_D8, WIN3_SLOW vs WIN3, and its median transitions;
- slow_start's CHECK 112, which is informative only.

## Results missing from the docstrings, and where I found them

- **test_stream_curriculum, test_slow_start and test_recipe_scope have only an X RESULT section.**
  - **Where L's verdicts are.** They are in docs/revision7.md:
    - stream_curriculum: line 188;
    - slow_start: Table 4 at line 216, counts at lines 220-228, readings at line 230;
    - recipe_scope: Table 5 at line 236, counts at lines 240-242.
  - **When they were first written.** All of them first appear at **6dd4248** (2026-10-05T21:08:32+00:00), in README.md as Revision 7. That text was moved unchanged to docs/revision7.md in 4a3eebc. Revision 7 says (docs/revision7.md:81 and :502) that these L verdicts were regenerated from L's results files with each test's report(). Those files were committed only later, at d0e6c85 (2026-10-06T03:58:09Z). That commit's message re-lists the L verdicts.
  - **Earlier mentions of H0 and Q2.** The background of test_early_recipe at 9f25dc8 (2026-10-04T03:30:57Z; test_early_recipe.py:23) already says "test_slow_start H0 SHOWN on X and L; test_recipe_scope Q2 SHOWN on X and L". I therefore set recorded_commit = 9f25dc8 for H0-L and Q2-L. That line gives no counts.
  - **slow_start HA on L.** It was reported at 6dd4248, before L's scale_axes file (its pairing partner) was in the repo. It was recomputed from the committed files only at 42af095 (2026-10-06T13:13:27Z): 5 vs 0, p = 0.031, SHOWN, which matches.
  - **Recomputed L counts.** For stream_curriculum L I also used `verdict` in results/L/stream_curriculum_results.json. The C3 group sizes (0 routed vs 27 not routed, 5 of the 27 bound) and the Fisher p (0.177) appear only there. The L reading row cites `verdict.readings` and its recorded_commit is d0e6c85, because Revision 7 states C1 NOT SHOWN but does not quote the reading text.
  - **Pairing gates on L.** The L pairing gates (slow_start CHECK 110, recipe_scope CHECK 118) are not printed in any committed text. `meta.pairs` in the L JSON files is all true. recorded_commit is d0e6c85.
- **test_window_gate L.** The docstring's L section (lines 176-190) was written at 628eb7c, together with X and pooled. L's verdict was first written in commit message **1cd20a1** (2026-10-09T05:20:05Z): "all four claims SHOWN/HOLD on L; pooled awaits X". I used 1cd20a1 for G1-G4 on L. I used 628eb7c for L's validity and bands, which 1cd20a1 does not mention.
- **Wilson intervals.** window_gate's L band lines print counts only, so the TSV says "Wilson not printed". muon_recipe prints Wilson intervals for every machine.

## Rules that changed after the first run

**test_window_gate, G2 (and the new G4).** X's full run started at 2f98f06 at 2026-10-07 23:30:25 (meta.started; X's clock appears to be UTC). The amendment came after that start: spec d05f0ca at 2026-10-08T00:48:06Z, code 886a668 at 2026-10-08T07:09:26Z.
- **Before the amendment.** The spec at 9a769c7 said: `  - G2: WIN3_SLOW_D8 beats HINGE_D8 (BOUND ROUTED).`
- **Diff in specs/test_window_gate.md at d05f0ca:**
  - `+> - Claims (pooled, primary; per machine secondary): G2 becomes WIN3_SPLIT_D8 beats HINGE_D8`
  - `+- G2: WIN3_SPLIT_D8 beats HINGE_D8 (BOUND ROUTED, exact McNemar one-sided).`
  - `+- Printed, not a claim: the original G2, WIN3_SLOW_D8 vs HINGE_D8 (McNemar); WIN3_RESET_D8 vs WIN3_SLOW_D8 on its`
  - `+G4 pair on equal budgets; the original G2 (WIN3_SLOW_D8 beats HINGE_D8) is **printed, not a claim**.`
- **Diff in test_window_gate.py at 886a668:**
  - `-CLAIMS = (("G1", "WIN3_SLOW16", "WIN16_A", "B"), ("G2", "WIN3_SLOW_D8", "HINGE_D8", "C"))`
  - `+CLAIMS = (("G1", "WIN3_SLOW16", "WIN16_A", "B"), ("G2", "WIN3_SPLIT_D8", "HINGE_D8", "C"),`
  - `+          ("G4", "WIN3_SPLIT_D8", "WIN3_SLOW_D8", "C"))`
  - `+           ("the original G2", "WIN3_SLOW_D8", "HINGE_D8", "C"),`
  - In the docstring: `+- Claims: G2 becomes WIN3_SPLIT_D8 beats HINGE_D8 (BOUND ROUTED); G4: WIN3_SPLIT_D8 beats WIN3_SLOW_D8 (BOUND ROUTED);`
- **Mitigation.** The docstring (lines 140-144) says the first start recorded only the six perfect-gate runs before a container restart killed it, and that no learned-gate outcome existed when the rule changed. L ran the amended test from the start at 886a668.
- **How I filled the columns.** For G2-new, G4 and the WIN3_SPLIT_D8 and WIN3_RESET_D8 bands, fixed_commit = d05f0ca (the spec, which is earlier); the test commit is 886a668. G1, G3 and the seven Part A-C bands are fixed at spec 9a769c7, before test commit 2f98f06.
- **Other changes in the amendment.**
  - The user's verbatim amendment said Part C/D runs 28800 steps. The user's answer, recorded in the same commit, made it 43200.
  - For reference only, since the original G2 is not a claim: it would also have passed (X 6 vs 0, p = 0.016; pooled 15 vs 0, p = 3.1e-5). The switch did not rescue a failing claim.

**All other tests: no rule changed.**
- I diffed the docstring at every commit that touched each file. After the test commit, the only changes are appended RESULT sections:
  - stream_curriculum: 520fe51;
  - slow_start: b8c6007;
  - recipe_scope: c69f1e6;
  - early_recipe: 9437e01 and 00ac0f2;
  - muon_recipe: 3e2d171 and 2a6ba3d.
- Later code edits added inert run-path knobs, each guarded by a CHECK:
  - stream_curriculum, 9c5939e: the `recipe=` knob in run_sc;
  - slow_start, a429af9: `stage=`;
  - slow_start, 9f25dc8: window, diag and stat;
  - early_recipe, 74f5907: one_run(path=) and lr_job(arm=, path=).
- A grep of those diffs for McNemar, band, SHOWN, claim and report logic found no change to claim logic.

## Ambiguities and caveats

1. **WIN3_RESET_D8 band.** The amendment says the control is "Printed, not claimed", and the user's verbatim text says "Bands with Wilson for both window arms". The fixed interpretation (spec line 167) says "Bands with Wilson for every arm, the two window arms of Part D included". Its band is printed for every machine (X NEVER 0/5, L MAJORITY 3/5, pooled MINORITY 3/10). I kept the rows and flagged them in the statement. Drop them if you read the control as not claimed.
2. **slow_start Part C reading.** The Part C arm is labelled "(descriptive)" in ARMS, but its reading has a fixed rule (at most 2 of 10 collapsed) inside the CLAIMS section. I kept it as a reading row.
3. **The window_gate reading on X.** It is pooled-only. The X section (lines 159-160) only notes that its conditions hold on X, so there is no X or L row.
4. **Cut runs reduce the pairs.**
   - early_recipe E2 on X: drop rule, 12 pairs.
   - muon_recipe M3 on X: 10 pairs, and 30 pooled.
   - stream_curriculum C1 on X: D8 cut to 6 seeds.
   - These cuts are pre-registered drop rules, applied as printed.
5. **stream_curriculum C1 on L.** The paired counts "SC8 2/10 vs D8 0/10" are derived from the discordant pairs (2 vs 0) and D8 0/10. Revision 7 prints only SC8 4/20 and D8 0/10.
6. **p-values not printed.**
   - recipe_scope Q4 on L: Revision 7 gives only "2 vs 2". The exact one-sided p is 0.6875.
   - L's Phase V p-values are as Revision 7 prints them, rounded: for example H0 "2×10⁻⁴" (exact 2.44e-4).
7. **Clocks.** L's meta start times appear to be local time (UTC-6). On that reading every L run started after its test commit. Read as UTC, slow_start (21:19 on 10-01, commit 23:28Z) and muon_recipe (23:08 on 10-05, commit 03:07Z on 10-06) would precede their commits. The evidence for UTC-6 is that the human commits in the repo are at -06:00 and every start fits under that offset.
8. **L's code commits.** L's stream_curriculum ran at 631fd62, recipe_scope at c69f1e6 and early_recipe at 9437e01. Each equals its test commit apart from the docstring, as stated in d0e6c85, 00ac0f2 and the docstrings. I did not re-diff the code.
9. **UNKNOWN.** No row is UNKNOWN.

## Consistency check (not a claim)

I recomputed every printed McNemar p from its discordant counts as an exact one-sided binomial (44 values). All match to the printed rounding. The arithmetic of each band against its rule, and of G3's d, also checks out.
