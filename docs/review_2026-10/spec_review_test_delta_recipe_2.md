# Spec review 2: specs/test_delta_recipe.md, Amendment 1 (`32c121b`)

*Referee V, 11 October 2026. Reviews the amendment that answers `spec_review_test_delta_recipe.md` (`fad1489`). "A1:NN" is a line of the amendment section (line 1 = "## Amendment 1"). Numbers come from `spec_checks_delta2.py` (output in `spec_checks_delta2_out.txt`).*

**Verdict.** The amendment takes every fix, and in two places the user chose a stronger one than I proposed: HEBB_SLOW, and labels with a printed interval. The no-room threshold is exactly the right number. The Newcombe interval is implemented correctly (see 3). Three things are left open, none of them blocking:
- **The d ≤ −3 label.** "Delta binds more often" is the mirror of the "Hebbian write stays" problem.
- **"Hebbian-specific" still names a contrast that no test makes.**
- **CHECK 174's failure has no consequence stated.**

## 1. Each fix, as meant?

| first review (`fad1489`) | Amendment 1 | taken as meant? |
|---|---|---|
| Reading 1: "interchangeable" from a d ≤ 2 bound (the S61 pattern). Rename; print the operating characteristics and a paired interval; add a row for d ≤ −3 | Three labels (A1:13-17): "delta binds more often" (d ≤ −3), "delta not shown worse by more than 2 pairs" (−2 ≤ d ≤ 2), "delta not shown non-inferior" (d ≥ 3). Newcombe's interval printed beside d (A1:18-19). The review's operating characteristics quoted (A1:20-22): 0.29/0.22 at 0.85 and 0.11/0.05 at 0.80, matching `spec_checks_delta_out.txt` §2 (0.287/0.223, 0.109/0.049) | **Yes**, with one asymmetry left (§2a) |
| Reading 2: "Hebbian-specific" cannot separate a high DELTA_SLOW from a failing split; condition on DELTA_SLOW or add HEBB_SLOW | Both. HEBB_SLOW added, with claim D4 (A1:8-10, 25). No-room rule at ≥ 36 of the pooled 40 (A1:26-27) | **Yes.** The threshold is exact (§2b). The name "Hebbian-specific" survives (§2c) |
| Reading 3: "the Hebbian write stays" on 3 vs 0 (p = 0.125) | Now "delta not shown non-inferior" (A1:17) | **Yes** |
| Per-machine d with the pooled margin | "Per machine: the rate difference and its interval only, no label" (A1:23) | **Yes** |
| Ties in the map; the label to use the outcome's map | A1:50-53: `torch.argmax`, lowest index; the trigger's label uses `routing_k` on `probe_batch(task, seed)`, as the outcome does | **Yes** |
| Overlap imported from a file under review; copy it and CHECK it | Copied into the test; CHECK 172 compares it with `recompute.py` at `0fdb363` on every WIN3_SPLIT_D8 record of the machine's `window_gate` file (A1:54-57). `overlap` is at `recompute.py:215` there, in the same form as the copy (`test_delta_recipe.py:198-204`) | **Yes** |
| CHECK 161 tolerance (open point 1) | 1e-7 on the perfect-gate model, 1e-6 on LOCAL3, "hebb" bit-identical (A1:60-61) | **Yes** |
| A CPU-independent CHECK of the arms' own memory (β = 1, L2 keys), since 161 tests β = 0 and 162 only on a 2.10 GHz Xeon | **173**: fused form against parallel form on the LOCAL3 and perfect-gate models, logits to 1e-5 relative and every gradient to 1e-4 relative, on every machine (A1:62-64) | **Yes**, and stronger (gradients too) |
| 158 asserted only on the record's CPU: say so | A1:65-67 says it in full: printed, not asserted, on X | **Yes** |
| Per-seed check that DELTA_SPLIT = DELTA_SLOW up to the first firing | **174** in the report, for both pairs, PASS/FAIL with seeds (A1:69-71); 164 extended to HEBB_SLOW (A1:68) | **Yes**, but FAIL's consequence is unstated (§2d) |
| Probe-blind checks counted and printed | A1:72-74: every check's probe beside the held-out accuracy, and the count of probe ≥ 0.95 with held-out < 0.95 | **Yes** |
| D3's population | "both split arms BOUND ROUTED" (A1:28) | **Yes** |
| Runtime | quotes the review's 2.2× and measures it (A1:77-78) | **Yes** |

## 2. What is left open

**(a) "Delta binds more often" at d ≤ −3 is as unsupported as the old "the Hebbian write stays" at d ≥ 3.**
- **The evidence the label can rest on.** d = −3 can be 0 vs 3, one-sided p = 0.125.
- **How often it fires with no difference.** Under equal rates of 0.95 with independent outcomes, it fires 9.5% of the time. At equal rates of 0.90 it fires 17%; "not shown non-inferior" fires equally often then.
- **How often it fires with a real difference.** With HEBB at 0.90 and DELTA at 0.95 it fires only 41% (independent) or 32% (nested).
- **The fix.** Name it like its mirror: "delta not shown worse; ahead by ≥ 3 pairs (untested)". Or let it read only when the one-sided McNemar for DELTA over HEBB gives p < 0.05, which needs 5 vs 0 or better. The printed interval already carries the uncertainty; the label should not outrun it.

**(b) The no-room threshold (A1:26).**
- **It is exactly where SHOWN becomes impossible.** A no-trigger arm at s of 40 leaves at most 40 − s split-only pairs:
  - at s = 36, the best possible outcome is 4 vs 0, p = 0.0625, so SHOWN is impossible;
  - at s = 35, 5 vs 0 gives p = 0.031.

  The rule therefore labels as "no room" precisely the counts at which no result could be SHOWN. It does not discard a result that could have been read. 36 is also the RELIABLE band at n = 40.
- **What it does not cover: the region just below.** With the split arm at 0.95, D2/D4 come out:

  | no-trigger arm's rate | helps | no room | not shown |
  |---|---|---|---|
  | 0.80 | 0.52 (nested 0.74) | 0.08 | 0.41 (nested 0.19) |
  | 0.85 | 0.28 (0.37) | 0.26 | 0.46 (0.37) |
  | 0.90 | 0.09 (0.05) | 0.63 | 0.28 (0.32) |

  So at a no-trigger rate of 0.85–0.90, a third to a half of outcomes read "not shown", although the split may help. "Not shown" there means "no power", not "no effect".
- **The fix.** Print the no-trigger count beside every "not shown" label, as well as the McNemar's b and c.

**(c) The name "the split is Hebbian-specific" (A1:42).**
- **When it fires.** It reads "not shown" on delta and "helps" on Hebbian.
- **What it is.** A difference between two significance verdicts, not a test that the split's effect differs between the writes (an interaction).
- **When it misleads.** With DELTA_SLOW at 0.80–0.85, "not shown" on delta comes out 37–46% of the time even if the split helps equally on both. The name asserts specificity that no test examines.
- **The fix.** Rename it "the split is shown on Hebbian, not on delta". Or add an interaction comparison: for example, Fisher on (b, c) of D2 against (b, c) of D4, printed and not read.

**(d) CHECK 174's FAIL.** The amendment prints PASS or FAIL "with the failing seeds; McNemar's pairing rests on it" (A1:69-71), but says nothing about what a FAIL does.
- **The proposed rule.** A seed whose split arm diverges from its no-trigger arm before the first firing has broken pairing. Its pair should be dropped from D2/D4, and the claim marked accordingly, with the count printed.
- **Why it matters.** It must be fixed now, before any run, or it will be decided after seeing which seeds fail.

**(e) The readings table's last row (A1:45).**
- **What it says.** "No room on either write | | that write's label only".
- **Two readings of it.** It can mean "no room on one of them" or "on both". If both, "that write's label" is undefined.
- **What the code does.** It prints both labels, "delta: …; Hebbian: …" (`test_delta_recipe.py:938-941`).
- **The fix.** Say that: "if either write is no room, both labels are printed and no combination is named".

**(f) Unchanged and accepted.** These are the user's choices, made knowingly:
- D1 remains a bound without an error rate.
- 40 pooled pairs.

The amendment prints the bound's operating characteristics beside it, which is what was asked.

## 3. The Newcombe interval

The amendment specifies "Newcombe's hybrid score interval for paired proportions, 1998 method 10" (A1:18-19). The test implements it at `test_delta_recipe.py:895-913`:
- Wilson limits for each margin (`test_stream_recipe.wilson`, z = 1.959963984540054, `test_stream_recipe.py:332-339`);
- the correlation φ̂ corrected for continuity when ad > bc, max(ad − bc − n/2, 0) over √((a+b)(c+d)(a+c)(b+d)), as Fagerland, Lydersen and Laake (2014) give method 10;
- θ − √((p₁−l₁)² − 2φ̂(p₁−l₁)(u₂−p₂) + (u₂−p₂)²) and θ + √((u₁−p₁)² − 2φ̂(u₁−p₁)(p₂−l₂) + (p₂−l₂)²);
- limits clipped to [−1, 1].

**Checked against an independent implementation.** I wrote the formula independently (`spec_checks_delta2.py` §4) and compared it with the test's function, extracted from `a581027` and run: on all 105,993 2×2 tables with n = 20, 40 and 80, the largest difference is 0. Illustrations at n = 40:
- 2 vs 0 discordant (36 both): +0.050, [−0.050, +0.167];
- 0 vs 3: −0.075, [−0.182, +0.025].

**Its direction.** θ = (b − c)/n with b = the first ("new") arm only. D1's interval is to be DELTA_SPLIT − HEBB_SPLIT, so DELTA_SPLIT must be passed as the first arm. That is a task-3 item (`test_check_delta_recipe.md`).

**What I could not check.** The method number, against Newcombe (1998) itself; the paper is not in the repository and arxiv.org is blocked here. The formula follows the standard description of the hybrid score interval with a continuity-corrected φ̂. If "method 10" in the 1998 paper is the uncorrected variant, then the corrected one is method 11. The implementation would be unchanged, and only the citation would need its number checked.

## What I could not check

- **The new CHECKs' code.** CHECKs 172–174 are checked in `test_check_delta_recipe.md` against `test_delta_recipe.py` as committed, not run here.
- **Newcombe's paper.** See §3.
