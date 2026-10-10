# Spec review: specs/test_delta_recipe.md (draft `05256c5`)

*Referee V, 10 October 2026, before any run. Lines are the draft's (`specs/test_delta_recipe.md` at `05256c5`). Numbers come from `spec_checks_delta.py` (output in `spec_checks_delta_out.txt`).*

**Verdict.** The path, the arms, the pairing and the CHECK chain are sound. The facts the draft cites check out against the records:
- CHECK 158's seeds first split at 4800: X 540 and L 1541.
- H's COPY_NONOISE|487 split at 4800 on a 2.10 GHz Xeon.
- F's S76 is 10/10 against 9/10.
- `recompute.py:215 overlap` exists on the main line.
- Seeds 580–599 and 1580–1599 appear in no record on any branch.

The **readings** are not ready. Two repeat the S61 pattern (no difference read as "no effect"), and D2's power is unknown because F's S76 had no DELTA_SLOW arm. Amend the following before the first run.

## Readings that can be read two ways

1. **"The memory rule is open (the two writes are interchangeable)"** (l.40-41, 150) rests on D1 HOLDS: d ≤ 2, a bound with no error rate. This is S61's "the noise does not matter" written in advance.

   **Power of the bound.** At 40 pairs, with HEBB_SPLIT at 0.95, D1 HOLDS with probability:

   | true DELTA_SPLIT rate | P(D1 HOLDS), outcomes independent across arms | outcomes nested |
   |---|---|---|
   | 0.85 (10 points worse) | 0.29 | 0.22 |
   | 0.80 | 0.11 | 0.05 |

   **No threshold on d fixes this at n = 40.**
   - With d ≤ −1, a 10-point deficit passes 4% of the time, but equal recipes pass only 40% of the time.
   - At 80 pairs, d ≤ 1 gives 4% and 71%.

   **Also, d ≤ 2 includes a delta recipe that is much *better*** (d ≪ 0), which "interchangeable" misdescribes.

   *Amend:*
   - Rename the reading "delta not shown worse by more than 2 pairs", and print its operating characteristics (the table above).
   - Print the paired difference with a 95% interval (Newcombe/Tango) beside d.
   - Add a row for d ≤ −3: "delta binds more often".
   - Or fix a margin with a stated error rate and the n it needs.
2. **"The split is Hebbian-specific"** (l.42, 151) when D2 is NOT SHOWN. NOT SHOWN has two causes that this test cannot tell apart:
   - **(a) DELTA_SLOW already binds.** Delta binds early: its perfect gate binds at 1200, and S76's delta binds a median of 9600 updates sooner. With DELTA_SLOW at 0.85, D2 is SHOWN with probability only 0.05–0.09.
   - **(b) The split fails on delta.**

   Only (b) is "Hebbian-specific". The test also has no HEBB_SLOW arm, so "helps on Hebbian" is imported from test_window_gate's other seeds.

   *Amend:* make the reading conditional on DELTA_SLOW's count (e.g., if DELTA_SLOW ≥ 36/40, D2 is "no room", not a reading). Or add HEBB_SLOW (Hebbian runs cost about ⅓ to ½ of delta's) so the split's effect on both writes is measured on the same seeds.
3. **"The Hebbian write stays"** (l.41, 152) when D1 FAILS. d = 3 can be 3 vs 0, one-sided p = 0.125, or 4 vs 1, p = 0.19. A strong reading should not rest on a difference no test would call significant. Equal rates of 0.95 FAIL 9.5% of the time.

   *Amend:* "delta not shown non-inferior", with the same printed interval.
4. **Per-machine d** (l.134) uses the same margin of 2 on 20 pairs. That is twice as strict per pair; P(HOLDS | 0.85) is 0.62–0.68 per machine against 0.22–0.29 pooled. Print the rate difference per machine, not a HOLD/FAIL.

**Outcome definitions** (the REG_D lesson).
- BOUND ROUTED is fixed by function (l.117-119), which is right.
- D3's population is settled as "both BOUND ROUTED" (open point 3); accept it.

Still open:
- **Argmax ties.** State that the stream → channel map's ties are resolved on unrounded values, and that the label (l.99-100) uses the same map as the outcome.
- **The overlap function.** The strict count imports `overlap` from a file under review (`docs/review_2026-10/recompute.py`). Copy the function into the test, or pin the commit, and add a CHECK that it reproduces the referee's value on one stored record.

## Missing or weak CHECKs

- **161 tests a different configuration from the arm.**
  - CHECK 161 sets β = 0, erase off and raw keys; the arms use β = 1 and L2 keys. It validates the code path, not the arm's memory.
  - CHECK 162 (F's record) covers the arm's configuration, but it is asserted only on a 2.10 GHz Xeon, so L never asserts it.
  - **Add** a CPU-independent CHECK: the fused form against the parallel form at β = 1 with L2 keys, to F's 1e-5 relative tolerance, on each machine.
  - **On the open tolerance (point 1):** accept 1e-6 on LOCAL3, which is 7× the measured 1.49e-7, and require the "hebb" implementation to be bit-identical.
- **158 is asserted only on the record's CPU (a 2.80 GHz Xeon).** In practice it is printed, not asserted, on both machines unless X runs on 2.80 GHz. Say so in the spec.
- **Pairing between DELTA_SPLIT and DELTA_SLOW.** D2's McNemar assumes the two arms are identical up to the first firing. CHECK 164 shows this on seed 1 only. Add a report-time check that every DELTA_SPLIT curve equals DELTA_SLOW's on the same seed up to the first firing.
- **Probe-blind checks.** Print every check's probe accuracy beside the held-out accuracy, and count the checks blocked by the probe. This is Session R's seed-914 mode: the main line had 6 of 393 such checks; R had 13. The probe's per-stream imbalance (5–13 queries) is the same for both arms, but which streams merge may differ between them.

## Power at the planned seeds

- **D2 depends entirely on DELTA_SLOW's rate,** and there are no data on it. With DELTA_SPLIT at 0.90, P(SHOWN) over 40 pairs is:

  | true DELTA_SLOW rate | P(D2 SHOWN), outcomes independent across arms | outcomes nested |
  |---|---|---|
  | 0.50 | 0.99 | 1.00 |
  | 0.65 | 0.80 | 0.98 |
  | 0.75 | 0.43 | 0.74 |
  | 0.80 | 0.23 | 0.37 |
  | 0.85 | 0.09 | 0.05 |

  For comparison, the Hebbian no-split arm bound 15/40 in test_window_gate.
- **D1:** as in reading 1 above.
- **D3** (secondary): adequate for a sign test on about 34 shared binds.
- **Runtime (open point 2):** F's records give DELTA_SPLIT 0.127 s/step against HEBB_SPLIT 0.059 (medians of secs/iters, about 2.2×). The draft's 3.1–3.4× is from pool-load timing, so its 25–45 h projection is conservative.

## What I could not check

- **The bit-for-bit claims** of CHECKs 159, 162 and 164, and the β = 0 logits differences of open point 1. Both need forward passes or runs, which this round's rules do not allow.
- **CHECK 159's probe.** H's trigger lineage uses the same probe as test_window_gate: 64 sequences from a generator seeded 12345 (`explore_split_child.py:46` at `7e9197f`, inherited by S48 and S61). So the probes agree on paper; only a run can show they agree bit for bit.
- **DELTA_SLOW's likely rate.** No arm in any record runs delta at eight streams without the split.
