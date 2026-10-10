# Review of Session R's replication (claude/replicate-R)

*Referee V, 10 October 2026.*

**What was read.**
- `replicate_report.md`, `DIFF.md`, `UNSTATED.md`, `replicate.py`'s docstring, `records/phase2.jsonl` and `records/phase3.jsonl`, all at `100dced`. That is the commit holding the report.
- `phase4.py`'s docstring at the branch head `464be8d`, committed 10 October 19:01 UTC. It pre-registers a 40-seed Phase 4 that has not been run.

**How the numbers were checked.** Every number below is recomputed by `replication_check.py`; its output is in `replication_check_out.txt`, sections 1–5. R's records hold each run's evaluation curve, map, per-stream accuracy, splits and probe readings, so R's counts can be checked without trusting its stored flags. One computation used torch: regenerating the main line's split probe (§4 of the output), with no training.

**Line references.**
- `rr:` is `replicate_report.md` at `100dced`.
- `D:` is `DIFF.md` at `100dced`.
- `p4:` is `phase4.py` at `464be8d`.
- Main-line files are cited at `7cc87e6`.

## Summary

- **R's numbers are right.** Every count, discordant pair, p-value, interval, split label and transition median reproduces from its records, and no stored flag disagrees with the recount.
- **"Partially replicated" is the correct mechanical reading** of 17/20 under R's pre-registered rule. That rule could not separate the outcomes it was meant to separate. The defensible statement is R's own (rr:30): 17/20 is consistent with 38/40, and 20 seeds cannot tell 0.85 from 0.95.
- **R's Phase 4 reading does not repair this.** By pooling the 17/20 already seen, its "consistent" verdict needs only 12 of 20 new seeds and is close to settled before it runs.
- **The seven additions.** Six are correct and needed; the seventh is already done in the PDF except for its seed-range point. Addition 5's "guessed right" is wrong for the vocabulary.
- **The additions are not sufficient.** Three more are needed. One is a defect in the main line's own probe: its 64 queries are unbalanced across streams, the same for every seed.
- **The isolation disclosure** does not bear on the result, but it shows that isolation rested on self-report.
- **A second replication** should be isolated by construction, sized to tell 0.85 from 0.95 (52 seeds), and should test the mechanism as well as the count.

## 1. Is "partially replicated" the right reading of 17/20?

**By R's rule, yes.**
- **The rule.** "Replicated" if x ≥ 18 and the one-sided McNemar p < 0.05; "partially replicated" if x ≥ 12; otherwise "not replicated" (`replicate.py:27-29`).
- **The rule was fixed before any run.** It is in the docstring at `786028e` (04:10 UTC); the first Phase 2 records are at `e327f42` (05:41 UTC). Later commits added only the RESULT block (`replicate.py:39-43`); no reading line changed.
- **The recount** (output §1):
  - WIN3_SPLIT_D8 17/20, Wilson [0.640, 0.948].
  - WIN3_SLOW_D8 7/20, [0.181, 0.567].
  - Discordant pairs 10 vs 0, p = 0.00098.
  - Oracle 2/2.
  - 45 splits (1–3 per run), 44 on target.
  - Median transitions 20,400 / 22,800.
  - Phase 3 R1 9/10, R2 9/10.
- **The reading.** 17 < 18 gives "partially replicated" (`replicate.py:43`).

**The rule was badly calibrated at both cuts** (output §2).
- **The upper cut (≥ 18/20) is a point threshold that often fails good recipes.** It fails 7.5% of the time at a true rate of 0.95, 32% at 0.90 and 60% at 0.85. R computes the first two itself (rr:97).
- **The middle band (12–17) is close to certain for any rate from 0.6 to 0.9.** At 0.85, P(x ≥ 12) = 0.999. It carries almost no information.

**What the data say.**
- **Against the main line.** 17/20 against the main line's 38/40 gives Fisher two-sided p = 0.32: no evidence of a lower rate. Pooled X + L + R: 55/60, Wilson [0.819, 0.964].
- **The same direction as the main test:**
  - every discordant pair favours the split (10 vs 0 here; 23 vs 0 there);
  - the no-split arm (7/20 against 15/40);
  - the failure mode: every failure is a merge of two or three streams. The maps for seeds 907, 914 and 915 are in output §1.

**Where R's conclusion goes beyond its evidence.** rr:30 says the shortfall is "consistent with seed-level variation, not with an identified implementation difference."
- **The reruns cannot support the second half for the failure that matters.** The reruns used seeds 900–909 (rr:23-24). Seed 914 is R's one failure that a protocol difference plausibly explains (rr:88; D:27):
  - Its probe read 0.953–0.984 at all 12 checks from 16,800 to 43,200.
  - Its 2048-query evaluation sat at 0.871–0.887 at the same updates.
  - The split rule therefore never fired again.
- **D-1 (the probe) was not rerun on seed 914.** Seed 907, which was rerun, failed again under both changes.
- **So the reruns show that D-1 and D-2 do not move the outcome on 900–909.** They do not show that D-1 did not cost seed 914.
- **The probe was blind more often than reported.** R's records show it blind on 13 checks, at seeds 906 (1 check, at 24,000; the run bound later) and 914 (12). rr:88 mentions only 914.

**Verdict.** "Partially replicated" is the correct mechanical reading and the wrong headline. The report should say: "17/20, consistent with the report's 38/40 (Fisher p = 0.32); the replication's bar was not calibrated to tell 0.85 from 0.95."

**R's Phase 4 does not fix the calibration** (p4:14-31; output §5).
- **Its rule.** "Consistent with the report" if the Wilson interval of the 40 seeds overlaps [0.835, 0.986] and McNemar p < 0.05; "a lower rate" if the interval's upper bound is below 0.835.
- **The threshold.** The upper bound reaches 0.835 at x ≥ 29/40. The 40 include Phase 2's 900–919 at 17/20 (p4:14-17), so the new seeds 920–939 need only 12/20.
- **What that rule can detect.** The probability of the reading "a lower rate" is:
  - 0.001 at a true rate of 0.85;
  - 0.041 at 0.75;
  - 0.113 at 0.70;
  - 0.238 at 0.65.
- **Most of the outcome is already fixed.** The reading is settled by data already seen, and the decision to extend to 40 seeds was taken after seeing a shortfall.
- **The cleaner figure** is the 920–939 count alone, all under the main line's specification. R prints it only as secondary (p4:19-20). It should be the primary count, compared with 38/40 by Fisher.
- **A test against 0.835 is weak at this size even on 40 fresh seeds.** "Lower if x ≤ 28" has size 0.024 and power only 0.29 at 0.75 and 0.56 at 0.70.

## 2. Are R's seven additions to the report correct and sufficient?

| # | addition (rr line) | correct? | evidence | needed? |
|---|---|---|---|---|
| 1 | The validity arm trains with single-rate Adam, not SLOW (rr:85) | yes | `specs/test_window_gate.md:65-66` ("Adam 1e-3 single rate"); D:29 explains R's later oracle transitions (6000 and 10,800 against 3600–6000) | yes; tex:138 names the validity arm ("the perfect gate on two seeds per part") but not its optimizer |
| 2 | The probe: the same 64 sequences for every seed (generator 12345); only the query scored; read at 2400 as the first reference; last check at total − 2400 (rr:86) | yes | `test_window_gate.py:299` (`PROBE_N, PROBE_SEED = 64, 12345`), `:301` (`REF_AT = 2400`), `:449-450`, `:577` (`sched["iters_c"] - EVERY`) | yes; tex:120 says only "a fixed 64-sequence probe" |
| 3 | The split's key mass uses the body's 32 keys, not the query key (rr:87) | yes | `test_window_gate.py:465-466` (`j = arange(S*P)`, `gr[:, 3*j+1]`) | yes; tex:120 says "key positions" |
| 4 | A 64-query probe can read ≥ 0.95 while the model sits at 0.88; say how often checks were blocked (rr:88) | yes, and it applies to the main line | Main line (output §3): **6 of 393** split checks read the probe ≥ 0.95 with the evaluation at 0.865–0.88 (X seed 552; L seeds 1550, 1553 ×2, 1554 ×2). All four runs bound later (transitions 27,600, 18,000, 22,800, 25,200), so no main-line failure is attributable to it. The reset arm: 0 of 145. R: 13 checks (above) | yes |
| 5 | The architecture choices a reader must guess, "each of which I had to guess (and guessed right)" (rr:89-96) | **as a recommendation, yes; the parenthesis, no** | The main line's vocabulary has 28 tokens with no `?`; R's has 29 tokens, with a `?` and an input of 99 (D:47, D-11). R did not guess the vocabulary right. D-11 to D-16 are inert for the forward pass (rr:81), and the mechanical check permutes the vocabulary (D:10). The other six items match DIFF.md's account | yes, all seven |
| 6 | A rate with its uncertainty, and the count an outside replication should expect (rr:97) | yes | P(≥ 18 of 20) = 0.925 at 0.95 and 0.677 at 0.90; P(≥ 17 of 20) = 0.98 at 0.95 (output §2) | yes |
| 7 | README 8.1's errata belong in the PDF (rr:98) | **already done**, except the seed ranges | The PDF was rebuilt with the 8.1 text at `404e3a0`; R read the `4a3eebc` PDF. The per-part seed ranges (X: Part A 500–539, Parts B–D 540–559; L: 1500–1539, 1540–1559; `specs/test_window_gate.md:65-66` gives the validity seeds) are in neither tex nor PDF: tex:138 gives only "X 500--559, L 1500--1559" and the per-part seed counts | the seed ranges, yes |

**The additions are not sufficient. Three more are needed.**

**(a) The main line's probe is unbalanced across streams, and the imbalance is the same for every seed.**
- **The composition.** Regenerated from `test_window_gate.py:449-450` on `TASK48` (output §4), the fixed probe's 64 queries fall 5, 5, 6, 8, 10, 12, 5 and 13 on streams 0–7.
- **What a merge costs each measure.** Suppose two streams merge and each is answered at about half accuracy.
  - The balanced 2048-query evaluation loses about 2 × (1/8) × 0.5 = 0.125.
  - The probe loses (n_i + n_j)/64 × 0.5. That is 0.078 for streams 0 and 1 (10 queries), 0.078 for 0 and 6, and 0.195 for 5 and 7 (25 queries).
- **Which way the error runs.** A merge of under-sampled streams is under-weighted by the probe. That is the direction that blocks a split, seed 914's mode.
- **It is not noise.** Because one probe serves every seed, the bias is systematic. The streams that merged at main-line seeds 552, 1550, 1553 and 1554 are not in the records' check entries, so I could not test whether they were the under-sampled ones.
- **What the report should do.** State the probe's composition, or the test should use a balanced probe (eight queries per stream).

**(b) The outcome's boundary rules.** A replicator needs all three to score its runs the way the main line does.
- **"Bound" can rest on the last evaluation alone.** It is the first evaluation ≥ 0.95 that holds to the end, so a run can bind on the last evaluation or two. X 545 binds on two evaluations at the budget.
- **The map's argmax resolves ties on unrounded values.**
- **The map comes from a 256-sequence probe** (`Generator(seed+99000)`), not from the 2048 held-out sequences (D:41, D-10).

**(c) A record format.** R's records cannot be scored by the main line's outcome code, nor the main line's by R's.
- **What R stores.** R stores the map but not the per-stream gate mass, and its failure labels are a map-based simplification.
- **What the report should publish.** The per-run fields an outside scorer needs:
  - the curve, `stream_gate` and `stream_acc`;
  - each check's probe and evaluation accuracy;
  - each split's (update, c*, c0, map before).

## 3. Does the isolation disclosure matter?

R reports (rr:108) that a filename search printed six names from the clankers checkout that held the PDF: `bdh_multichannel.py`, the v7 and v8 `.tex` files, and older report PDFs. R says it opened none of them.

**For the result, no.**
- **The names say nothing about the protocol.** Six filenames carry no information about the choices that decide the outcome, and the v8 `.tex` contains the same text as the PDF R was given.
- **R's choices differ from the main line exactly where reading the code would have given the answer:**
  - the probe's generator (D-1);
  - the key positions (D-2);
  - the oracle's optimizer (D-3);
  - the order of operations at an evaluation (D-4);
  - the initialisation order and the absence of W_h (D-8);
  - the map's probe (D-10);
  - the vocabulary (D-11).

  A reader of `test_window_gate.py` would have copied at least the probe.
- **R's code before Phase 3 carries no trace of the main line's code** (`786028e`, `de8df0f`).
  - It holds none of the main line's distinctive constants (31337, 12345, 12_000_000, 424242) or names (`LOCAL3`, `run_sc`, `TASK48`, `MultiBDH`).
  - Its one shared term, "KEYMASS", is in the report.

**For the method, yes.**
- **Isolation rested on self-report.** The whole checkout was on the replicator's disk, and only the replicator's account says it was not read.
- **R's matches cannot settle it either way.** R matched the main line on the other six of addition 5's seven choices. That is evidence about how determinate the report is, but it would look the same if the code had been read.

## 4. What I would require of a second replication

1. **Isolation by construction.** A container that never holds the clankers repository:
   - only the Revision 9 PDF and Pathway's public `bdh.py`;
   - the network restricted to package indexes;
   - a log of the files opened.

   It should be run by a different implementer from R.
2. **A sample size and reading fixed in advance, with power stated** (output §5).
   - **52 fresh seeds**, none pooled with any earlier run.
   - **The reading:** "below the report's 0.95" if x ≤ 46/52, an exact one-sided test of size 0.045 with power 0.81 at a true rate of 0.85. "Consistent" otherwise.
   - **The comparison with the main line,** stated beside the reading: Fisher against 38/40.
   - **The split against the no-split arm,** by McNemar on the same seeds.
   - **Why this rule.** An interval-overlap rule like Phase 4's detects only rates near 0.70 or below. If 52 seeds are not affordable, say in advance what the chosen size can detect.
3. **The mechanism as well as the count.** Run the no-split, reset-only and copy-only arms on the same seeds. Revision 8's second claim is "the copy, not the reset", which the main line is now testing in `specs/test_split_copy.md`. A replication of the count alone does not replicate the paper.
4. **The protocol as Revision 9 states it.** Every remaining unstated choice should be logged before running, as R did in UNSTATED.md (44 items). The checklist is UNSTATED.md plus (a)–(c) above. Any choice that differs from the main line's is decided by the report, not by the code.
5. **Records in the published format** ((c) above), so that the referee can score them with the main line's own outcome code. That includes the blind-check count and the per-stream gate masses.
6. **A different machine class.** A GPU, if Session I's package passes its CUDA CHECK, in float32 with a stated tolerance. The claim is about a recipe, not about one CPU's numerics.
7. **Seeds disjoint from every block in use,** registered before the run. For any shortfall, a pre-specified rerun that applies the main line's choices to the **failing** seeds (here 914, 915), not to a convenient block.

## What I could not check

- **R's runtime and environment** (37.4 h over 4 workers × 1 thread, rr:40). Taken from its records' fields, not reproduced.
- **R's account of what it saw before Phase 3.** Checked only through its code's content (§3). The session's tool log is not in the repository.
- **DIFF.md's mechanical check** (R's weights in the main line's model; gates to 7×10⁻⁹, logits to 7×10⁻⁸). It needs R's weights, which are not committed.
- **Which streams merged at the six blind main-line checks.** The check entries do not store the map at the check.
- **Phase 4's outcome.** It has not been run (p4, "RESULT ... (not yet run)").
