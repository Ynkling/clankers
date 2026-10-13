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
3. **The sentence on L** (`claims_registry.md:38`, "On L 37 are primary records, 16 are second-hand, and only 40 have a printed p") should read: 37 primary in a docstring or results file, 16 confirmed by the results files that arrived on 10 October, and 51 with a p.

## What this does not settle

- **The 32 L rows the registry marks NOT RECORDED.** These are bands, validity gates and readings. They are recounted in the next section.
- **The registry's own TSVs.** They are not edited, so the tables in `claims_registry.md` still show the old labels. This addendum supersedes the points above.

## The 32 NOT RECORDED L rows, recounted from the records

**Where they come from.** These are the 32 rows of `registry/main_*.tsv` whose L verdict was "NOT RECORDED", "NOT PRINTED" or "not printed …". They span 11 tests (Phases III–IV):
- 13 validity gates;
- 9 bands;
- 10 conditional readings.

**What each row was checked against.** `registry_L_recount.py` recounts each row from the per-run records in `results/L` under the test's own rule; its output is in `registry_L_recount_out.txt`. Each recount is compared with two things:
- what the results file itself stores for that row, under keys the earlier comparison did not match: `band_*`, `valid`, `validA`/`validB`, `reading(s)`, `gate0`/`gate1`, `counts`;
- the counts the registry had derived from later text.

**The rules used.**
- BOUND means the record has a transition (`test_short_conv.py:70`); DISCOVERED means BOUND and a final VAL cos below 0.5 (`test_router_discovery.py:68`).
- Bands are on BOUND, except `router_layout`'s, which its code computes on DISCOVERED (`test_router_layout.py:630`).
- Each record's stored transition was recomputed from its curve (the first evaluation ≥ 0.95 that holds to the end): 0 differences.

**Result.**
- All **32 of 32 agree** with the stored field.
- Every count the registry had derived from later text (`docs/revision6.md`, the next test's BACKGROUND, or pooled minus X) **matches** the record count.
- No validity gate fails, no band changes, and each conditional reading applies exactly where the registry said it would.

| test | rows | recount (from `results/L`) |
|---|---|---|
| channel_binding | GATE0, GATE1 | S=1 binds 3/3 at P=4 and 3/3 at P=8 (Gate 0 passes at both). Gate 1: P=4 ceiling 5/5 with fit argmax 1.00, **VALID**; P=8 ceiling **3/5**, **EXCLUDED** (seeds 1 and 4 end at 0.943 and 0.940; seeds 0 and 3 bind only at the last evaluation, 38,400). New to the registry, which had "P=8 outcome on L not stated". On X, P=8 failed Gate 0 instead. Either way P=4 is the only scored configuration on both machines, as the registry says. |
| router_discovery | INVALID, VERDICT | A_ro_nudge discovers **0/10**, so **INVALID**, and no candidate is interpreted (A_ro_lr10 0/10, A_ro_ln 0/10, A_ro_noise 1/10; baseline A_ro 1/10). This confirms the registry's derived "implies INVALID" and fills in the candidate counts. |
| router_curriculum | INVALID | A_nudge discovers 10/10: the positive control is valid |
| router_layout | 3 bands | A_grouped 28/40, A_blocked 21/40, A_shuffled 29/40: all **MAJORITY** (BOUND and DISCOVERED give the same counts) |
| short_conv | VALID, 3 bands, 3 readings | ceiling_conv 5/5, VALID. B_conv 5/40 MINORITY, A_conv 28/40 MAJORITY, A 14/40 MINORITY. K1 NOT SHOWN, so the K1− reading applies and the K1+ readings do not |
| conv_lr | VALID, 2 bands, 4 readings | ceiling_conv4 5/5 with the pairing CHECK stored true, VALID. B_conv4 16/40 MINORITY, A_conv4 28/40 MAJORITY. LR3 SHOWN and LR4 NOT SHOWN, so the "LR3+LR4−" reading applies; "A_conv4 RELIABLE" does not apply |
| p_scaling | VALID, 4 readings | ceiling_conv8 8/8, VALID. P1 SHOWN and P2 NOT SHOWN, so "P1+P2−" applies; P3 NOT SHOWN, so "P3+" does not |
| load_curriculum | VALID | CUR_ceiling 5/5 with the pairing CHECK stored true: VALID |
| curriculum_confirm | VALID | CUR_ceiling_lo 5/5: VALID |
| scale_axes | VALID A, VALID B | ceiling8 3/3, VALID. ceiling4k4 5/5 with k=4 fit argmax 1.000, VALID |
| stream_channels | VALID | ceiling4k8 3/3, pairing CHECK stored true, k=8 and k=16 fit argmax 1.0: VALID |

**What changes in `claims_registry.md`.**
- **The 32 L rows** read as above, sourced from `results/L`. None is "not recorded" any longer.
- **Counts.** L now has a recorded outcome for every Phase III–IV row except the CHECK-based parts of four validity gates. Those parts are the conv-inert-at-init CHECK 54 and the pairing CHECKs 59, 64, 72 and 85. Each is stored only as a boolean in the file (`valid`, `pairing`), and the CHECK's own comparison is not in any record.
- **One addition, channel_binding P=8 on L.** It passes Gate 0 and fails Gate 1. Revision 9's account of P=8 should say that it was excluded on X by Gate 0 and on L by Gate 1.

**Caveats.**
- **channel_binding's L file is a rerun.** It was run on L on 10 October (`meta.started` 10:28, `meta.git` `886a668`; added at `9c53e16`, whose message says the test code is unchanged since `93b227c`). It is not the September run that L's NOT SUPPORTED verdict was first reported from. The two agree where both can be compared: in both, learned A_ro binds only on seed 4, at 3600 (VAL cos 0.0009).
- **The readings branch on verdicts, not on run counts.** Which reading applies is derived from the stored K/LR/P verdicts, which agree with every L verdict row (above). Their p-values were not recomputed from the per-run records here.
