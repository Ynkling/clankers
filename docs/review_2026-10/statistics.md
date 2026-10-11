# Statistics: the choices, the objections, and the analyses that would answer them

*Referee's deliverable 4. Every number below is printed by `stats_checks.py` (output in `stats_checks_out.txt`) or `recompute.py`, from the records. "tex N" is a line of `multichannel_hebbian_report_v8.tex`.*

## Verdict in one paragraph

The headline does not depend on any contestable statistical choice. G2 (38 vs 0) and G4 (23 vs 0) are significant one-sided or two-sided, paired or unpaired, and survive Bonferroni over 120 claims (G4: 1.4e-5). The weaker results are where the choices matter: G1 (13 vs 2, p = 0.0037) would not survive control of family-wise error over the project's own count of about sixty claims per machine (Bonferroni 0.22); G3's "bound" has poor operating characteristics and is compatible with the window recipe being 8 points worse; and the report's reading of the Muon result ("a seed effect") is not what the data say. I dispute no p-value the report prints: all reproduce (recompute.md). I dispute three inferences drawn from them.

## 1. Exact one-sided McNemar on discordant pairs (tex 122, 160)

**The choice.** Arms are paired by seed (same initialisation of the shared parameters and same batches), only discordant pairs count, and the test is P(Bin(b + c, ½) ≥ b), with the direction fixed in the specification.

**What a statistician would object to.**
- *Direction.* One-sided tests are defensible only when the direction is fixed before the data; here it was (spec `9a769c7`, amendment `d05f0ca`, both before any learned-gate run; preregistration.md). No objection, but the report should print the two-sided values beside the one-sided ones, since readers will compare them with two-sided screen p-values printed elsewhere in the same report (tex 129, 131, 180). Two-sided: G1 0.0074, G2 7.3e-12, G4 2.4e-7, M1 0.002. Every SHOWN claim stays below 0.05.
- *Pairing across different architectures.* For G2 the pair is a recurrent-gate model and a window-gate model; the seed fixes the batches and the shared stack, not the gate. The pairing then buys little, and the conditional test discards concordant information. That is harmless here: the unpaired one-sided Fisher tests give G1 0.0016, G2 8.0e-21, G4 2.1e-8 (stats_checks §1). The conclusions do not rest on the pairing.
- *What the test is about.* It is a test of these two arms of this code on this task with this budget. A significant McNemar says that, conditional on a seed disagreeing, the split arm wins; it does not estimate how much better the recipe is in a population of tasks or models. The report's wording ("binds eight streams ... in 38/40 runs") is scoped correctly; the abstract's subtitle-level framing ("solves context-conditional binding") is not a statistical claim and should not borrow the p-value's authority.

**Analysis that answers it.** Print two-sided p-values alongside one-sided ones; report the paired difference in success rates with a confidence interval (for G4: (23 − 0)/40 = 0.575 with an exact conditional interval), not only the p-value.

## 2. Pooling two machines' disjoint seeds (tex 40, 122, 291)

**The choice.** b and c are summed over both machines' pairs, and the pooled test is primary.

**Is it a valid test?** Yes. Within each machine a discordant pair is equally likely to go either way under the null, whatever the machine; summing conditionally independent binomials with p = ½ gives Bin(b + c, ½). This is the exact stratified McNemar, and "the pooled test is a test" (abstract) is right.

**What a statistician would object to.**
- *Pooling is not replication.* The report's multiplicity defence (tex 122) puts "replicated on both machines" and "pooled over disjoint seeds" on one footing. They are not: a pooled test can be driven by one machine. The heterogeneity tests are the missing step. For G1, G3 and G4 the two machines agree (Fisher on the discordant splits p = 1.00, 0.67, 1.00). For M2 they do not: X 7 vs 1, L 5 vs 7, p = 0.070; WIN16_M binds by 4800 on 15/20 seeds on X against 9/20 on L (p = 0.105). The report notices the disagreement but neither tests nor explains it.
- *"Machine" confounds CPU and seed block.* X ran 500–559, L 1500–1559. A machine difference is indistinguishable from a seed-block difference. Nothing suggests either matters for the window gate; for Muon, X's 2.80 GHz and L's i7 differ in the bf16 Newton–Schulz kernel (`host` fields in the records), so a numerics × optimiser interaction is plausible and testable.
- *Two machines are two levels of a fixed factor, not a sample of machines.* "Two machines" adds robustness to numerics, not generality.

**Analysis that answers it.** Report per-machine estimates with a heterogeneity test for every pooled claim (as above). For Muon, swap seed blocks between machines (X runs 1440–1459, L runs 440–459) to separate CPU from seeds.

## 3. "Bound" claims without p-values (G3; S54 "delta not worse"; S64/S65 "scale-free")

**G3** (tex 160): "WIN3_SLOW not worse than HINGE0 by more than two discordant pairs" holds with d = 9 − 11 = −2.

**Objections.**
- *The margin is a count, not a rate.* Two pairs is 5% of 40 pairs per machine and 2.5% of 80 pooled. A fixed count margin gets stricter as n grows and was not justified.
- *Its operating characteristics are poor in both directions* (stats_checks §4). With 20 discordant pairs, if the two recipes are exactly equal the rule FAILS 25% of the time; if WIN3_SLOW is truly 5 points worse it HOLDS 40% of the time; 10 points worse, 11%. At exact equality the failure rate rises with the number of discordant pairs (17%, 25%, 32% at 10, 20, 40).
- *What the data allow.* The paired difference is +0.025, and the 95% interval obtained from the discordant split is [−0.079, +0.121]. The data are compatible with the window recipe being eight points worse at two streams. "HOLDS" is a mechanical outcome of the rule, not evidence of non-inferiority at any stated margin.

**Analysis that answers it.** A pre-specified non-inferiority margin on the rate scale (for example 5 points), tested with an exact or Tango interval for the paired difference; with 80 pairs and 20 discordant, the lower bound −0.079 would fail a 5-point margin.

**S64/S65 "scale-free" (d ≥ −1 over 10 paired seeds).** The report says the screen "rules out a large drop, not a small one" (tex 214), which is fair. The abstract says the constants "survive a fourfold change of width and a doubling of depth without retuning", which is not. Under an independence approximation (stats_checks §7), with the reference at 0.9 and the larger model at 0.7, the rule reads "scale-free" 39% of the time; at 0.6, 20%. Every d = +1 rests on one seed (273) that the reference lost.

**S54(a) "delta better"** (F, tex 224): 36/40 against a recorded 31/40 on other seeds and another CPU, unpaired, one-sided Fisher p = 0.11. The rule (≥ 35/40) was fixed before the run (preregistration.md), but a reading called "better" at p = 0.11 is a threshold on a point estimate. The report's own sentence ("a bound by the screen's rule") is accurate; F's report labels it "delta better" in bold.

## 4. "About sixty pre-registered one-sided claims per machine" (tex 122)

**The framing.** Sixty one-sided claims at α = 0.05 give about three false positives under the global null; trust results replicated on both machines or pooled.

**Objections.**
- *The number is not checkable from the report.* No list of the sixty claims exists in the repository, so neither the count nor which ones passed can be verified. I did not reconstruct it.
- *Replication on both machines is a strong filter; pooling is not the same filter.* Under the global null a given claim passes on both machines with probability 0.0025 (0.15 expected among sixty). A pooled test passes with probability 0.05.
- *Family-wise control kills G1.* Bonferroni over 60: G1 0.22, M1 0.059; over 120: G1 0.44, M1 0.12. G2 and G4 survive any correction (2.2e-10 and 7.2e-6 over 60). G1 is in the abstract ("at four streams ... 38/40 against 27/40 ... p = 0.004"). Under a false-discovery-rate view G1 would likely survive, but the report should say which error rate it is controlling.
- *The global null is not the relevant null.* Claims chosen after promising screens have high prior probability; the expected number of false positives is lower than three. That argument cuts in the report's favour, but the report does not make it.

**Analysis that answers it.** Publish the list of claims with p-values; state the error rate controlled; apply Holm (FWER) or Benjamini–Hochberg (FDR) to the whole list and report which headline claims survive. Within test_window_gate alone, Holm over G1, G2, G4 keeps all three (G1 at 0.0037).

## 5. Bands (RELIABLE ≥ 90%, MAJORITY ≥ 50%, with Wilson intervals; tex 122, 160)

**Objections.**
- *Bands are thresholds on point estimates.* RELIABLE at 18/20 (X's WIN3_SPLIT_D8) has a Wilson lower bound of 0.70; at 38/40, 0.835. A reader takes "RELIABLE" to mean "≥ 90% with confidence"; it means "≥ 90% observed". The report does print the intervals, which is good practice.
- *The readings use bands inconsistently.* test_muon_recipe's reading required a Wilson lower bound ≥ 0.80 as well as the band; test_window_gate's reading required only the band. The headline would pass either rule (lower bound 0.835), so nothing changes, but a single rule should be used.
- *Wilson vs exact.* Wilson can be anticonservative near 1; the exact one-sided 95% lower bound for 38/40 is 0.851 (stats_checks §5), so the conclusion is unchanged.

**Analysis that answers it.** Define bands on the lower confidence bound (e.g. RELIABLE = one-sided 95% lower bound ≥ 0.8), or drop the bands and report intervals.

## 6. Two inferences I dispute

**(a) "The one-machine result was a seed effect" (tex 267) and "did not replicate on the other machine's seeds" (tex 200).** Revision 7's E2 was SHOWN on **L** (7 vs 0, p = 0.008) and pointed the same way on X (5 vs 1). In the Muon-recipe test, on fresh seeds, X confirmed it (7 vs 1, p = 0.035) and L reversed (5 vs 7). So the result failed on its *own* machine's fresh seeds and was confirmed on the *other* machine's, the opposite of the report's sentence. Across the four looks the counts are 24 vs 9; the two E2 looks share seeds, so they cannot be pooled as independent; M2 pooled with L's E2 (disjoint seeds) is 19 vs 8, one-sided p = 0.026 (descriptive, not pre-registered). And M2's pooled test, with 20 discordant pairs, had power 0.42 against a 70/30 split of discordant pairs and 0.62 against 75/25 (stats_checks §3). "Not shown" is the correct verdict; "a seed effect" is a conclusion the data do not support, and the heterogeneity between machines (p = 0.070) is the more interesting finding.

**(b) "The copy, not the reset" and "the optimizer reset ... adds nothing" (abstract, tex 37; tex 191, 330).** The three cited looks each compare the split (copy + reset) with the reset alone; they show the reset alone is not enough (pooled S50 + S58b, 16 vs 1, one-sided p = 1.4e-4; the main test's control seeds 7 vs 0, p = 0.0078). None of them has a copy-without-reset arm, so none can show the reset adds nothing. The only arm that does is session H's S61 (`copy2x2`, uncited; COPY 9/10 = SPLIT 9/10 on ten seeds), whose resolution is a handful of discordant pairs. The first half of the claim is supported; the second rests on one uncited ten-seed screen.

## 7. Smaller points

- *Outcome definition.* "Bound" accepts a run that reaches the budget on one evaluation ≥ 0.95 (tex 108). One of the 38 headline successes (X, seed 545) rests on two evaluations at the budget; none on one. The count is robust; the report's "11 of 1028" is stale (recompute.md).
- *"One channel per stream" is an argmax.* In 18 of the 38 headline successes some stream's largest channel holds under 0.9 of its gate mass; with "no two streams overlap by more than 0.1" as the outcome, G1 becomes 16 vs 2 and G4 stays 23 vs 0 (recompute.md, sensitivity section). The claims survive; the wording should say "disjoint channel sets".
- *Mixed sidedness.* The copy/reset table prints a two-sided p (S50) beside a one-sided one (S58b) in the same column (tex 180–181). Use one convention per table.
- *Screens at n = 10.* Readings such as "improves if ≥ 6/10" have wide intervals (9/10: Wilson [0.60, 0.98]). The report labels screens exploratory; the abstract's "Three exploratory looks ... agree" should keep the n beside each look.
