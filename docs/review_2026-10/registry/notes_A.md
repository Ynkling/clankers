# Registry A: notes

Repo /home/user/clankers, branch claude/review-V (HEAD 0fdb363). Nothing in the repo was modified. All dates are committer dates converted to UTC.
TSV: /tmp/claude-0/registry_A.tsv, 28 rows (5 tests: 15 X rows, 13 L rows).

## General

- **6c90a19 is a shallow-clone boundary, not an import.** `git rev-parse --is-shallow-repository` returns true, `.git/shallow` lists 6c90a19, and its recorded parent 9bb19f8 is not in the object store. The true history of the five files is reachable through `--all`, via origin/claude/bdh-repo-curl-obgdqk and relaxed-edison-he00cc. Each file at HEAD is byte-identical to its last pre-boundary commit (66754b2, 29f1f9e, 2271b55, d5e3db6, 8f28abb).
- **Where the RESULT sections are.** At HEAD the test docstrings hold only machine X's results. No docstring has a pooled section, and no file has an L RESULT section. L's results come from other places:
  - docs/revision6.md, the archived Revision 6 report. It was first written as README.md at 631fd62 (Ynkling, 2026-09-30T20:00:19Z), then moved to docs/ at 6dd4248.
  - The BACKGROUND sections of later test docstrings (test_router_discovery 4422dc4, test_readout_path 58c625c).
  - results/ has no L file for any of these five tests. results/X/ has X's files.
  - For L rows, docstring_ref points to that secondary source.
- **Phase.**
  - channel_binding, router_discovery, router_curriculum and router_confirm are Phase III. Evidence: docs/revision6.md:154-160 is the Phase III (Revision 5) summary covering the perfect gate 10/10, readout arms, the 5% nudge and the confirmation. docs/revision6.md:795 says "router_curriculum, router_confirm (Phase III, on L)". docs/review_2026-10/revision_8_1_check.md:54 says "Phase III arms named B (router_discovery, router_curriculum, router_confirm)".
  - readout_path is Phase IV: docs/revision6.md:84-88, "Table 1. Phase IV".
  - No commit message names a phase.
- **Claim ids.** Only router_confirm (C1-C3) and readout_path (R1-R3) give ids in the docstring. For the other three I took the names from the docstring's own section headings:
  - channel_binding: GATE0, GATE1, VERDICT.
  - router_discovery and router_curriculum: INVALID (the validity rule) and VERDICT.
  - router_confirm's INVALID rule is also a row.
- **Rule changes after the first run: none, in any of the five files.**
  - In each file, the only docstring change between the test commit and the result commit is that the line `-(Results are recorded at the bottom of this docstring after the run.)` is replaced by the RESULT block.
  - Later changes add text or code but alter no claim rule:
    - 29f1f9e (router_discovery) adds a POST-HOC block and a `load_legacy_pair(sha=None)` refactor.
    - 2271b55 changes router_curriculum code (`run_spec` accepts `arm_def`, `time_arm` accepts a dict) after X's run. This is not a rule change. L's run at 5ba7c1d used this later code.
  - I checked the code thresholds (FOUND_K, PARTIAL_K = 8, 3; ALPHA; RELIABLE_K/MAJORITY_K) and they match the docstrings.
- **L's run commits** (docs/revision6.md Table 10 and git):
  - channel_binding ran at 7cf7d5a, a merge on L. Its test_channel_binding.py is identical to 93b227c.
  - router_curriculum, router_confirm and readout_path ran at 5ba7c1d, a merge on L. There, test_readout_path.py is identical to 58c625c, and router_confirm/router_curriculum are identical to d5e3db6.
  - router_discovery's L run commit is not recorded.
- **p-values recomputed** with math.comb (one-sided Fisher). All printed values reproduce:
  - X: C1 0.004318, C3 0.8686; R1 and R2 7.233e-08, R3 1.740e-08.
  - L: C1 0.00195, C3 0.7498; R1 2.67e-08, R2 4.14e-10, R3 9.35e-08. These match the rounded figures 0.002, 0.75, 3e-8, 4e-10 and 9e-8.
- **Excluded as diagnostic / not a claim:**
  - channel_binding: the DIAGNOSTICS block.
  - router_discovery: DIAGNOSTICS and POST-HOC.
  - router_curriculum: DIAGNOSTICS, including the "accuracy at T near 0.5" precondition.
  - router_confirm: the paired McNemar, labelled diagnostic.
  - readout_path: the recovery Fisher tests, McNemar, gradient norms and the rest of its diagnostics.
  - Diagnostic McNemar figures are quoted in the discordant column only as context, marked "not the claim".

## Per test

### test_channel_binding.py (III)

- Ids: GATE0, GATE1, VERDICT.
- Rule fixed at 93b227c (2026-09-25T14:05:41Z).
- X recorded at 66754b2 (2026-09-25T16:10:21Z).
- Machines: X and L.
  - L has no RESULT section. Its NOT SUPPORTED verdict and pooled counts are stated in test_router_discovery.py:15-18 (4422dc4): "NOT SUPPORTED on both machines at P=4, the only config that passed Gates 0 and 1 on both"; pooled perfect gate 10/10, controls 0/50, learned channels 1/20, the one success being L's A_ro seed 4.
  - My L counts (perfect gate 5/5, learned 1/10) are derived by subtracting X.
- Ambiguous:
  1. On X, Gate 0 at P=8 was 2/3. On L, the P=8 outcome and which gate failed it are not stated.
  2. X's RESULT says "(The other machine bound P=8 5/5.)". Five seeds does not match Gate 0's 3 seeds. This probably refers to test_binding_recipe on L, not to this test's gate.
  3. The verdict is a margin rule with no p-value. I classed it as "bound".
  4. P=16 was dropped by the logistics projection rule (> 8 h), not by a gate.

### test_router_discovery.py (III)

- Ids: INVALID, VERDICT.
- Rule fixed at 4422dc4 (2026-09-25T18:27:13Z).
- X recorded at 357062a (2026-09-25T19:59:39Z).
- Machines: X and L.
  - **L's result is missing.** No L verdict is printed anywhere.
  - Only these appear, at 58c625c (test_readout_path.py:16-18): L's A_ro 1/10, and "2/20 on A_ro (X and L combined)" for the nudge. The second implies L's A_ro_nudge was 0/10, which means INVALID on L, but that verdict is never printed.
  - 29f1f9e's POST-HOC says "on both machines A_ro's gate hardened".
  - L rows carry verdict "NOT PRINTED".
- Ambiguous:
  - The printed X verdict gives a cause, "does not reproduce on this machine". The POST-HOC note at 29f1f9e (2026-09-25T21:18:18Z) says that cause is wrong. The verdict itself is unchanged; this is not a rule change.

### test_router_curriculum.py (III)

- Ids: INVALID, VERDICT.
- Rule fixed at 1604209 (2026-09-25T21:18:18Z).
- X recorded at 6e0f0da (2026-09-25T22:45:07Z).
- L recorded at 631fd62 (2026-09-30T20:00:19Z), README.md:160 then; now docs/revision6.md:162: "PARTIAL, as on X (A_late_ln 6/10, A_ln 6/10, A_late 5/10, A 3/10, nudge 10/10)".
- L's validity outcome is not printed separately.

### test_router_confirm.py (III)

- Ids: INVALID, C1, C2, C3.
- Rule fixed at 2271b55 (2026-09-25T23:16:20Z).
- X recorded at d5e3db6 (2026-09-26T01:53:39Z).
- L recorded at 631fd62 (2026-09-30T20:00:19Z), now docs/revision6.md:163-166. L's C2 count of 21/40 is taken from the C1 line.
- Ambiguous:
  - A and A_ln are paired by seed, but C3 is pre-registered as an unpaired Fisher test. The paired McNemar (X: 5 vs 9, two-sided p = 0.424) is diagnostic only.
- Earlier, cruder trace of L's counts: test_router_reliability.py:19-20 (001c63e, 2026-09-26T18:13:14Z) has "L 45/80" for arm A. That equals 3 + 23 + 19.

### test_readout_path.py (IV)

- Ids: R1, R2, R3. R1 also gates the interpretation of R2 and R3.
- Rule fixed at 58c625c (2026-09-26T03:00:29Z).
- X recorded at 8f28abb (2026-09-26T05:33:17Z).
- L recorded at 631fd62 (2026-09-30T20:00:19Z), now docs/revision6.md:235-238 (counts) and 241-243 (verdicts), with p printed only to one significant figure.
  - An earlier qualitative statement exists at 001c63e (test_router_reliability.py:24-25): "test_readout_path confirmed, on both machines, that the cause is the readout's gradient into the gate." That covers R1 and R2, without numbers.
- Ambiguous:
  - The arms are paired by seed, but the claims are unpaired Fisher tests. X's McNemar figures (21/1, 21/1, 22/1) are diagnostic.

## Machines that did not run a test

- Only X and L appear. Machine E, and any other machine, ran none of these five tests.
- No "pooled" claim exists in these five tests. router_curriculum and router_confirm offer `--also` pooling as descriptive only.
