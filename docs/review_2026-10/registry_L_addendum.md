# Addendum to claims_registry.md: L's Phase III–IV records

*Referee V, 10 October 2026.*

**Why this addendum.** `claims_registry.md:23-26` says L's Phase III–IV records do not exist in the repository. That was true when it was written. Since then the main line has added them:
- `9a53302` (16:30 UTC): twelve files that L ran on 25–29 September;
- `9c53e16` (18:46 UTC): channel_binding, rerun on L.

Each new file names L's CPU, the i7-12650H; none is a copy of X's file. The earlier copies were the `664729f` files, byte-identical to X's.

**How this was checked.** `registry_L_addendum.py` compares every L row of `registry/main_*.tsv` with the verdict and p the new file stores; its output is in `registry_L_addendum_out.txt`. Re-running `registry.py` on the merged tree changes its check 2 from "compared 55, agree 55" to "compared 90, agree 89" (`registry_checks.txt`). The one difference is `short_conv` K2, which Revision 6 prints as 0.03 for the stored 0.0339. That is the rounding already listed under check 1.

## What the records settle

- **Second-hand verdicts.** All 17 second-hand (SECONDARY) L rows for which the file stores a verdict agree with it. Among them are router_layout L1 and L2, router_reliability K4, K8 and RESTART, and short_conv K1–K3. No second-hand verdict was wrong.
- **Primary verdicts.** All 21 primary L rows for which the file stores a verdict agree with it.
- **New p-values.** Eleven L hypothesis tests that had no printed p now have one from the file:
  - conv_lr LR1 0.0064, LR2 0.59, LR4 0.11;
  - p_scaling M1-P8 2.9×10⁻⁶, P2 0.76, P3 0.60;
  - load_curriculum C2 0.0017, C3 2.5×10⁻⁵;
  - router_layout L1 0.97, L2 0.5;
  - curriculum_confirm K1 0.52.
- **Still no p on L.** Two hypothesis tests have no p anywhere: stream_curriculum C3 and recipe_scope Q4.

## Corrections to claims_registry.md

1. **`claims_registry.md:23-26` (and `registry/summary.md:23-26`).** "L's records for Phases III–IV do not exist" is superseded. They exist from `9a53302` and `9c53e16` on. L's Phase III–IV verdicts no longer rest only on second-hand text.
2. **L's family under correction** (the table at `claims_registry.md`, "Which error rate the count supports", row L). It now reads:

   | family | tests | with a p | p < 0.05 | Holm, FWER 0.05 | BH, FDR 0.05 | positives that depend on the unadjusted α |
   |---|---|---|---|---|---|---|
   | L, Phases III–VI | 53 | **51** (was 40) | 32 | **13** (was 11) | **27** (was 23) | short_conv K2 (0.034), stream_channels MG (0.027), slow_start HA (0.031), H0S (0.031), S0 (0.046) |

   The four positives the registry listed as having no printed p (conv_lr LR1, p_scaling M1-P8, load_curriculum C2, C3) all survive BH with their stored p.
3. **The sentence on L** ("On L 37 are primary records, 16 are second-hand, and only 40 have a printed p") should read: 37 primary in a docstring or results file, 16 confirmed by the results files that arrived on 10 October, and 51 with a p.

## What this does not settle

- **The 32 L rows the registry marks NOT RECORDED.** These are bands, validity gates and readings. The new files store no verdict key for them. Their per-run records would allow recomputing them, but I have not done so here; that is a recount of 15 tests' arms under their own docstrings.
- **The registry's own TSVs.** They are not edited, so the tables in `claims_registry.md` still show the old labels. This addendum supersedes the points above.
