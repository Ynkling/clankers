# Registry C: notes

Repo /home/user/clankers at HEAD 0fdb363 (claude/review-V). Nothing in the repo was modified. Output: /tmp/claude-0/registry_C.tsv, 84 rows: 42 claim ids, each with an X row and an L row. Of these, 50 are claim, validity or band rows and 34 are reading rows.

## Claims per test (each one has an X row and an L row)

| test | phase | ids | fixed (first in the test file) | X RESULT commit |
|---|---|---|---|---|
| test_load_curriculum.py | IV | VALID, C1, C2, C3, C4, RD1-RD4 | b7c18f0 | d79626e |
| test_curriculum_confirm.py | IV | VALID, K1, K2, K3, RD1-RD3 | c5ec279 | 72e4e07 |
| test_scale_axes.py | IV | VALID A, VALID B, M1, S1, S2, RD1-RD4 | 248c482 | 3f4222f |
| test_stream_channels.py | IV | VALID, K8, K16, MG, band_A4k8, band_A4k16, RD1-RD3 | e5a24ae | 4ea2693 |
| test_stream_recipe.py | V | VALID, R1, R2, R3, R4, RD1-RD3 | 092937b | 18d5ae1 |

- RDn are the "Readings, printed verbatim where they apply", numbered in the order the docstring lists them. They are conditional statements, not tests.
- band_A4k8 and band_A4k16 are listed inside stream_channels' CLAIMS section but have no claim id. I named them myself.
- The fixed commits were checked with `git log --all -S'<claim text>'` on each file. Every claim's rule first appears in the file's add commit.
  - None of the five files is touched by the import commit 6c90a19, and no spec file in specs/ precedes them.
- Excluded:
  - DIAGNOSTICS and post-hoc lines.
  - The descriptive arms: CUR_A_lo; curriculum_confirm Part 2 (P=16); stream_recipe Part 2 (S=8).
  - "CUR_lo's band printed alongside" (K3), which I folded into K3's counts cell.
  - The Part 2 drop and skip rules. On X, stream_recipe's Part 2 was DROPPED (14.04 h > 12 h, test_stream_recipe.py:158). On L it ran (A8k16_R 0/5, ceiling8k16 2/2).

## Rules changed after the first run: none

- I ran `git diff <add commit> HEAD` on each file. The docstrings only gained a RESULT section: the earlier placeholder line "(Results are recorded at the bottom of this docstring after the run.)" was replaced by the RESULT text. No CLAIMS, VALIDITY or READINGS line changed.
- Later code changes:
  - test_load_curriculum.py: c5ec279 added the `stats_at` / `arm` knobs (`cur_stats_fn(t_switch, also=())`, `run_cur(..., stats_at=())`, `run_trial(..., arm=None, stats_at=())`). The same commit moved CHECK 70's body into `generator_check()`.
  - test_scale_axes.py: 9c5939e added `run_job(sp, recipe=None)`.
  - test_stream_recipe.py: 9c5939e added `run_attempt(..., recipe=None)`.
  - All three changes are inert at their defaults, and none touches verdict or report logic.
- Follow-up commits 1062bb6 and d045d25 edited only Diagnostics lines.
- L ran load_curriculum at merge 98f3bd3. Its test file has the same blob as b7c18f0 (19013e7).

## Where L's verdicts come from (none of the five docstrings has an L RESULT section)

### 1. The three tests with no L file in the repo

- **What is missing:** load_curriculum, curriculum_confirm and stream_channels have no L results file on the main line.
- **The off-main-line copies are not L's data:**
  - Commit 664729f ("Added machine L's test results", 2026-10-05) is not an ancestor of HEAD. It is only on origin/claude/bdh-repo-curl-obgdqk and origin/claude/relaxed-edison-he00cc.
  - Its results/L/{load_curriculum, curriculum_confirm, stream_channels}_results.json are byte-identical to results/X's blobs (0c59ddc, 308d788, 6cecda2): same Xeon CPU, same meta.started, same verdicts as X.
  - Its scale_axes and stream_recipe "L" files are also X's blobs.
  - These are mislabelled copies of X's data and must not be used as L evidence.
- **Where the L verdicts actually are:** the BACKGROUND section of the next test's docstring, and later docs/revision6.md (the Revision 6 text was first committed to README in 631fd62, 2026-09-30T14:00:19-06:00, and archived in 6dd4248):
  - load_curriculum (L): test_curriculum_confirm.py:16-26 (c5ec279). The C1 p = 0.058 is there too.
  - curriculum_confirm (L): test_scale_axes.py:16-19 (248c482). K2's p = 2x10^-4 appears only in docs/revision6.md:416 (631fd62).
  - stream_channels (L): test_stream_recipe.py:17 (092937b). The p-values 0.019 / 0.011 / 0.027 and the 10 vs 2 / 9 vs 1 discordant pairs appear only in docs/revision6.md:479 (631fd62).
- **L numbers that were never recorded:** load_curriculum C2's CUR_A count on seeds 160-179 and both p-values (C2, C3); curriculum_confirm K1's p. Three entries in the TSV are derived rather than recorded:
  - C3 on L: pooled 52/59 vs 2/20 minus X gives 25/30 vs 1/11.
  - MG on L: docs Table 7 minus X gives 8/20 vs 15/20. A recomputed one-sided Fisher gives 0.0268, matching the printed 0.027.
  - VALID on L for stream_channels: pooled 6/6 minus X gives 3/3.
- **Recomputed checks (exact, by me), all matching the printed values:**
  - C1 on L: McNemar 14 vs 6 gives 0.0577.
  - K2 on L: Fisher 17/20 vs 3/15 gives 1.66e-4.
  - K8 and K16 on L: 0.0193 and 0.0107.
  - S2 on L: 0.1088.
  - M1 on L: 5.58e-4.
  - K1 on L, never printed anywhere: 0.52.

### 2. scale_axes (L)

- The verdicts were first recorded in test_stream_channels.py:14-17 (e5a24ae).
- L's own file, results/L/scale_axes_results.json (CPU i7-12650H, meta.git 248c482), was committed in 42af095.
- Its `verdict` field supplies M1's McNemar (5 vs 0, p = 0.0625), Fisher p = 0.000558, validA and validB = true, and the readings ["M1 SHOWN", "S1 MINORITY"].

### 3. stream_recipe (L)

- The verdicts were first recorded in test_stream_curriculum.py:16-20 at ffdf0aa (2026-09-30T19:32Z). This is about 30 minutes before the README update 631fd62 (20:00Z), which carries the same verdicts.
- L's file (i7-12650H, meta.git 092937b) was committed in d0e6c85. Its commit message gives "VALID; R1 RELIABLE (19/20); R2 PRECISE (19/19); R3 SHOWN (p = 0.0002); R4 NOT SHOWN (8 vs 4, p = 0.19)".
- The JSON supplies Wilson [0.832, 1.0], R3's McNemar 11 vs 0 (p = 4.9e-4), Fisher p = 2.16e-4, and R4 p = 0.194.
- docs/revision7.md:172 and :180 (Table 2) repeat these. Revision 7 lists stream_recipe as Phase V; Revision 6 lists it under Phase IV as "running".
- L's VALID: ceiling 3/3 first appears in the Revision 7 table (6dd4248, 2026-10-05). The word "VALID" for L first appears in the d0e6c85 commit message.

## UNKNOWN recorded commits

There are 10 L reading rows: RD rows for load_curriculum, curriculum_confirm and stream_channels.
- L's report output for these tests is not in the repo, and no document prints which reading applied on L.
- The verdict cell instead states whether L's recorded claim verdicts meet the condition. That is the case for load_curriculum RD3, curriculum_confirm RD2 and stream_channels RD1. The other rows are marked not applicable.

## Ambiguities and points for the referee

- **"Not recorded as printed" in the validity rows:** for L's validity rows in Phase IV, only the ceiling counts are recorded (docs tables). L's VALID line and its pairing CHECKs (72 for load_curriculum, 85 for stream_channels) and gate fits were never written down. That L's claims were reported implies VALID, but nothing prints it.
- **Two different "L" verdict sources disagree:** for load_curriculum, curriculum_confirm and stream_channels, the off-main-line "L" JSONs at 664729f (really X's files) say X's verdicts. For load_curriculum and stream_channels these contradict the text records of L: C1 NOT SHOWN, C4 MAJORITY, and K8/K16/MG SHOWN. I used the text records.
- **Timezone of L's meta.started is unknown:**
  - L's scale_axes says 2026-09-28 20:39, against 248c482 at 22:05Z. This is consistent only if L's clock is local -06:00 (L's merge commits carry -06:00), i.e. 02:39Z on 09-29.
  - L's stream_recipe says 2026-09-29 22:57, which is 04:57Z if -06:00, after 092937b at 03:37Z.
  - If those times are UTC, L's scale_axes run would predate its test commit. Please confirm L's clock.
- **X's runs start after their fixed commits:** load_curriculum 07:23 vs 05:45; curriculum_confirm 15:59 vs 14:56; scale_axes 23:14 vs 22:05; stream_channels 18:36 (09-29) vs 17:28; stream_recipe 08:55 (09-30) vs 03:37.
- **R2 is a threshold rule, not a test:**
  - It is classed "bound": a 0.9 precision threshold on a point estimate, with the Wilson CI printed but not part of the rule.
  - R1, C4, K3, S1 and the stream_channels bands are classed "band".
- **"One-sided" in the sided column:** C2, K1, K2, M1 and R3 carry a second test "printed alongside": McNemar, two-sided for C2, K1, K2 and M1, one-sided for R3. Only the primary test decides the verdict; the alongside p-value is in the discordant cell.
