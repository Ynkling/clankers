# Spec review: the two-stream key-term test draft (Session H, `explore_out/H/spec_draft_two_keyterm.md` at `9b72bf6`)

*Referee V, 12 October 2026, before the main line adopts the draft and before any run. "l.NN" is a line of the draft at `9b72bf6`; the draft is unchanged at explore-H's head, `5b5deec`. "out §N" is a section of `spec_checks_register_keyterm_out.txt`, written by `spec_checks_register_keyterm.py` (stdlib only; it reads git and nothing else). The project's rules are `CONTRIBUTING.md` at `a76c4ee`. The form follows `specs/test_split_copy.md`, as the user asked, and `specs/test_delta_recipe.md` with its amendments.*

**Verdict.** The draft's structure is sound:
- two arms paired by seed, and a perfect-gate oracle;
- one hypothesis test, with readings that give every outcome exactly one name (out §6);
- the task-structure caveat, printed with every reading;
- the key term copied byte for byte with a syntax-tree CHECK, so L has a link to the screens' code that does not depend on the CPU (T1, T2).

Its background reproduces from the records (out §3), with one p-value that is two-sided where the test is one-sided (l.26). The seeds are free (out §1).

**What the test cannot do as drafted: K1 has too little power.**
- **The baseline is high.** It is the same WIN3_SLOW arm that discovered 69/80 in test_window_gate Part A (X 33/40, L 36/40). There is little room above it.
- **The likely chance of SHOWN is low.** At the main line's rate (0.8625) and a key-term rate of 0.93-0.95, P(K1 SHOWN) over 40 pooled pairs is 0.15-0.27. At the screens' own pooled discordance (S52 + S85: 12 vs 2 of 60) it is 0.62.
- **A third of outcomes cannot show it at all.** In 34% of outcomes at 0.8625 (63% at 0.90), WIN3_SLOW binds at least 36 of 40, which leaves at most 4 discordant pairs, and K1 then *cannot* be SHOWN.

So "not shown" would mostly mean "no power", the S61 pattern again.

**The other items:**
- The outcome is named by an exploratory function (`explore_common.discovered`, l.75), not the main line's.
- The rate claim K3 has the same weaknesses as the register draft's R. Its readings call a sampling miss "not reliably".
- T1's commit is left "named when adopted" (l.143).
- K2 is listed as a claim but is a count, which the readings invite the reader to judge without a rule.

The numbered fixes at the end are replacement text.

## 1. Arms (l.44-59)

**The arms:**
- WIN3_SLOW: test_window_gate's arm, imported.
- WIN3_SLOW_KEYHINGE: the same plus S52's key term, and nothing else.
- CEIL_A, for validity.

This is the smallest design that answers the question, and it matches the user's prompt (l.9-18). That makes 42 runs per machine.

**Pairing.** The arms are paired through the identical initial parameters and batches that T2 implies: with the term off, the run is bit for bit WIN3_SLOW through 2400.

**Runtime** (out §3). The draft's "about 5 minutes each" (l.175) overstates S85's runs:

| S85 arm | minutes per run, median (range) |
|---|---|
| KEYHINGE | 2.0 (1.7-2.9) |
| REF | 1.6 (1.3-6.3) |

So the test needs about 1.5-2 CPU-hours per machine. Doubling the seeds (fix 1) is cheap.

## 2. Seeds (l.92-101)

**Scan.** I scanned the same 12 branch heads, 308 JSON/JSONL blobs and 28,791 seed values as for the register draft (out §1, listed in `spec_review_register_draft.md` §2):
- **Records.** No record uses X 620-639 or L 1620-1639.
- **Integer keys.** The only integer dictionary key in the block is 625, under `dense`: an update step in the S52/S78/S85 records' dense η² traces.
- **Text.** Mentions are only the proposal itself (l.92, 98-99) and explore-H's README (`explore_out/README.md:610-611`). My text scan covered `*.py`, `*.md`, `*.tex` and `*.txt`, and found none in explore-I. The draft's "source-line ranges in explore-I's gpu/audit.json" (l.99-100) is a JSON text match, which my record-level parse confirms is no seed.

**Neighbours** (out §1):
- G's register draft: X 600-619, L 1600-1619.
- Free: X 640-699 and L 1640-1699, so fix 1's extension to 620-659 / 1620-1659 is free.
- Round-5 blocks (F 2400-2459, H 2600-2659, G 2700-2759, R 940-959, I 2540-2559 and 840-859) are disjoint.

**The seed list in l.95-97 is right.** test_window_gate is X 500-559 / L 1500-1559, test_split_copy X 560-569 / L 1560-1569, test_delta_recipe X 580-599 / L 1580-1599 (L not yet recorded).

**T10 enforces it.**

## 3. Validity (l.57-59, 88-90)

**The rule.** CEIL_A must bind both of the machine's first two seeds within 24000 updates (test_window_gate's VALID_N = 2).

**The oracle.** CEIL_A is `test_short_conv`'s `ceiling_conv`: the perfect gate plus the conv layer, with one Adam at 1e-3.

**Its record.** Bound at 1200 in every look:
- test_short_conv, 5/5;
- test_window_gate's CEIL_A, 2/2 on each machine;
- S78, 2/2;
- S85, 2/2 (out §3).

**What it validates.** CEIL_A has a conv layer and the arms do not. It therefore validates the task, the memory and the path, not the no-conv arms' capacity. The rule is inherited unchanged from test_window_gate Part A (`test_window_gate.py:55`) and the draft gives the reason (l.58-59). Fix 9 asks for one sentence saying what the part validates.

**One part covers every claim.** That is right here: nothing in the design depends on a learned oracle.

## 4. Outcome: DISCOVERED (l.75-77)

**The draft names `explore_common.discovered`**: bound, and the final VAL cos below 0.5. That module is on the exploratory branches only. The main line has 2 `explore_*` files, and this is not one of them.

**The main line's definition of the same outcome:**
- `test_router_reliability.run_one` records `rec["discovered"] = bool(rec["transition"] is not None and rec["val_cos"] < DISC_COS)` (`test_router_reliability.py:338`);
- `DISC_COS` = 0.5 (`test_router_discovery.py:173`);
- test_window_gate reads it through `success(key, r)` for Part A arms (`test_window_gate.py:319-325`).

Fix 3 names these at the main line's commit.

**Failure classes.** `fail_class_v2.fail_class_v2` (main line, `fail_class_v2.py:40`) and v1 `test_router_layout.fail_class` are named correctly.

**One caution for K2: the KEY class sits on a threshold** (η² by key ≥ 0.5). Records near it:
- X's WIN3_SLOW s522 is at 0.4999999999784 and s531 at 0.5000018 (`docs/fail_class_v2.md:82, 93`).
- S52's 182 and 186 are 1.4e-4 and 1.5e-5 below 0.5 (l.27-28).

KEY counts can change by one on a tenth-of-a-millionth. Fix 6 prints η² by key with every classified run.

## 5. Claims and their multiplicity (l.103-113)

**What the draft claims:**
- **K1** is the test's one hypothesis test: exact one-sided McNemar, pooled primary and per machine secondary.
- **K2** is a printed count. It is listed among the claims but has no test.
- **K3** is a rule on counts.

**Multiplicity.** Within the test there is one primary p-value, so no correction is needed.

**Across tests: the same key term twice.** If the main line also adopts the register draft (`test_register_gate`), both tests use the same key term with S52's constants (λ = 1.0, τ = 0.2, updates 1-2400). The blobs are identical on every branch (out §2).

A sentence in the report about "the key term" would then rest on two families. State it now (fix 7) so that the report does not merge their verdicts silently.

**Background p-value.** l.26 gives S52's "7 vs 2 discordant, p = 0.18". That is the *two-sided* exact McNemar, as in batch 18's README (`explore_out/README.md:563` on outside-ideas). The one-sided p, the test's own form, is 0.090 (out §3). Fix 10.

## 6. The rate claim K3: its interval and operating characteristics (l.112-113)

**The interval and the odds.** These are the same numbers as for the register draft (out §4):
- At the threshold, 18/20 has Wilson [0.699, 0.972] and Clopper-Pearson [0.683, 0.988]. MET supports a rate of at least about 0.70.
- P(≥ 18/20 on both machines) by exact binomial:

  | true rate | on both | on exactly one |
  |---|---|---|
  | 0.80 | 0.042 | 0.327 |
  | 0.85 | 0.164 | 0.482 |
  | 0.90 | 0.458 | 0.437 |
  | 0.93 | 0.704 | 0.270 |
  | 0.95 | 0.855 | 0.140 |

**What the screens predict.**
- The key term discovered 56/60 in S52 + S85.
- The predictive P(MET on both) is 0.59 with a uniform prior and 0.64 with Jeffreys; 0.62-0.67 if both machines share one rate (out §4).
- So even if the screens transfer exactly, K3 fails somewhere about 4 times in 10.

**The rule is not monotone.**
- "Pooled: MET only if MET on both machines" (l.113) is not monotone in the pooled count: 20/20 with 17/20 fails, 18/20 with 18/20 passes.
- CONTRIBUTING asks for pooled-primary claims. Print the pooled count with its Wilson interval and band beside the per-machine verdicts (fix 5).

**The UNTESTED case is covered.** "Either machine NOT VALID … no other reading is given" (l.120) means no reading reads a K3 verdict from one valid machine.

## 7. McNemar power for K1 (out §5)

**Rates in the records** (all one arm, by test_window_gate's CHECK 135):

| source | WIN3_SLOW discovered |
|---|---|
| main line, Part A | 69/80 = 0.8625 |
| S43 / batch 16 (160-199) | 31/40 |
| S78 REF | 15/20 |
| S85 REF | 15/20 |
| all records | 130/160 = 0.8125 |

The key term discovered 56/60 = 0.933 (S52 36/40, S85 20/20).

**P(K1 SHOWN)**, independent / nested outcomes, at 40 pooled pairs (20 per machine) and at 20 pairs (one machine):

| WIN3_SLOW | key term 0.933 | key term 0.95 | key term 1.00 |
|---|---|---|---|
| 0.75 | 0.626 / 0.880; 0.267 / 0.297 | 0.731 / 0.924; 0.329 / 0.370 | 0.984; 0.585 |
| 0.80 | 0.400 / 0.629; 0.144 / 0.117 | 0.516 / 0.737; 0.184 / 0.170 | 0.924; 0.370 |
| 0.8125 | 0.343 / 0.537; 0.118 / 0.084 | 0.455 / 0.660; 0.153 / 0.130 | 0.893; 0.316 |
| 0.8625 | **0.146 / 0.148**; 0.042 / 0.011 | **0.220 / 0.270**; 0.056 / 0.026 | 0.660; 0.130 |
| 0.90 | 0.052 / 0.010; 0.013 / 0.000 | 0.088 / 0.048; 0.017 / 0.003 | 0.371; 0.043 |

**At the screens' own discordance:**

| screen | 40 pairs | 20 pairs | 80 pairs |
|---|---|---|---|
| S52 + S85 (12 vs 2 of 60) | 0.616 | 0.232 | 0.933 |
| S52 alone (7 vs 2) | 0.372 | — | 0.723 |
| S85 alone (5 vs 0) | 0.984 | — | — |

**Pooled pairs needed for 80% power:**

| scenario | pairs |
|---|---|
| the screens' discordance | 56 |
| 0.95 vs 0.80, independent | 68 |
| 0.95 vs 0.8625, nested | 76 |
| 0.95 vs 0.8625, independent | 152 |

**With 40 seeds per machine (80 pooled pairs)**, P(K1 SHOWN), independent / nested:

| key term vs WIN3_SLOW | P(K1 SHOWN) |
|---|---|
| 0.933 vs 0.8125 | 0.68 / 0.97 |
| 0.95 vs 0.8125 | 0.82 / 0.99 |
| 0.933 vs 0.8625 | 0.34 / 0.67 |
| 0.95 vs 0.8625 | 0.51 / 0.84 |

**The no-room region.**
- If WIN3_SLOW discovers on 36 or more of the pooled 40, then b ≤ 4 and p ≥ 0.0625, so K1 cannot be SHOWN.
- P(WIN3_SLOW ≥ 36/40) is 0.076 at 0.80, 0.107 at 0.8125, 0.340 at 0.8625 and 0.629 at 0.90.
- At 80 pairs the bound is ≥ 76/80.

The draft has no no-room rule. Its reading "reliable, but not shown better than WIN3_SLOW" covers the no-room case and the low-power case together.

**Why the per-machine K1 should carry its power.** Per machine, K1 (secondary) is SHOWN with probability at most 0.33 at plausible rates. Print it with that operating characteristic, not as a verdict on a machine.

## 8. The readings: exhaustiveness and wording (l.115-123; enumerated in out §6)

| K1 (pooled) | K3 on X | K3 on L | reading printed (draft) |
|---|---|---|---|
| SHOWN | MET | MET | "the key term makes two streams reliable" |
| SHOWN | MET / NOT, NOT / MET, NOT / NOT (3 rows) | | "the key term helps, but not reliably" |
| not SHOWN | MET | MET | "reliable, but not shown better than WIN3_SLOW" |
| not SHOWN | not both MET (3 rows) | | "not shown" |
| either machine not VALID | | | "UNTESTED" only |

**Coverage.** Every combination gets exactly one name, with no gap or overlap.

**A good property.** "Makes two streams reliable" cannot fire while WIN3_SLOW is itself at least 18/20 on both machines. That would mean at least 36/40 pooled, hence b ≤ 4 and p ≥ 0.0625. The reading never credits the term with a reliability the baseline already had.

**What the names get wrong:**

**(a) "Not reliably" names a sampling miss.**
- Given K1 SHOWN, "helps, but not reliably" fires whenever K3 fails on either machine.
- That happens with probability 0.54 at a true rate of 0.90, 0.27 at 0.933, 0.15 at 0.95 and 0.03 at 0.975.
- At the screens' 0.933, more than a quarter of the cases where the term works as screened would be printed as "not reliably". Name the count, not a property.

**(b) "Reliable, but not shown better than WIN3_SLOW" lumps no room with no power.** It mixes:
- the no-room case (the baseline at 36/40 or more, so K1 is impossible);
- the low-power case.

Amendment 2(b) of test_delta_recipe settled the same problem by printing the baseline's count and b, c with every label. Fix 2 adds a NO ROOM label.

**(c) "Two streams" overreaches.** The subject of every reading generalises beyond this configuration: S = 2, P = 4, k = 2, no conv, grouped layout, the CTX token one position before every key. The readings should name it.

**(d) K2's reading has no rule.** "K2 is printed under every reading, so a reader can see whether the key splits are gone or only rarer" (l.122-123) invites a judgement without a rule. Either drop the purpose clause or fix a descriptive rule now, as S85's "the key split only moves" was (fix 6).

## 9. The CHECKs (l.138-170)

| CHECK | can it fail? | before any run? | note |
|---|---|---|---|
| inherited chain (l.140-141) | yes | yes | |
| T1 copied functions, syntax trees (l.142-144) | yes | yes | the commit is "named when adopted": name it now. The three files are the same blobs at S52's start commit `2afa02f` (outside-ideas), S85's start `37c1fbf`, and every branch head that has them: `explore_keysplits_child.py` `f05e8f9d3b`, `explore_b18_gates.py` `a61a577388`, `explore_aux_gate.py` `ffae342178` (out §2) |
| T2 switch off (l.145-147) | yes | yes | CPU-independent: the link between the arms |
| T3 S52 reproduction (l.148-152) | yes, on a matching CPU | yes | the reference names no commit. The record exists with the fields T3 reads: `hinge.fired` 2; `hinge.eta_at` at 1, 300, 600, 1200 and 2400; CPU 2.10 GHz (out §3). Last commit of the file: `f671ab3`. Vendor it (fix 4) |
| T4 S85 reproduction (l.153-155) | yes, on a matching CPU | yes | key `KEYHINGE\|2310\|957abe999f8d\|Intel(R) Xeon(R) Processor @ 2.10GHz` exists with a hinge record (out §3); file last changed at `1e09ecd`. Vendor it |
| T5 = test_window_gate CHECK 135 (l.156-157) | yes | yes | |
| T6 the window (l.158) | yes | yes | |
| T7 the gradient (l.159-160) | yes | yes | |
| T8 the statistic (l.161-162) | partly | yes | "eta2_key at k = 2 equals routing_stats' channel-0 eta² by key" has no tolerance. At k = 2, channel 1 = 1 − channel 0, so the pooled ratio equals channel 0's up to the 1e-12 in the denominator and float order. State 1e-6 |
| T9 the classifier (l.163-166) | yes | yes | "the others in docs/fail_class_v2.md" is open-ended. Name the population: `fail_class_v2.reclassify` over `docs/fail_class_v2.md`'s population, on this machine's files |
| T10 seeds (l.167-168) | yes | yes | |
| T11 workers, resume (l.169-170) | yes | yes | |

**On L, T3 and T4 only print.** L is an i7-12650H (`results/L/split_copy_results.json`: 54/54). L's link to the screens is T1 (the code is the same) and T2 (the arm with its switch off is WIN3_SLOW). That is enough: unlike the register draft, nothing in H's arm is rewritten.

**Missing: the key positions.**
- The term reads fixed positions 3j + 1 (l.68-69). This is right for the grouped TASK: `test_binding_capacity.BindTask` has L = 3(SP + n_q) = 27 and qpos = 3·SP + 1 = 25.
- G's term finds keys per sequence instead.
- A one-line CHECK that every 3j + 1 (j < SP) holds a key token (S ≤ token < S + P), on the probe and a training batch, makes the claim falsifiable (fix 8).

## 10. The task-structure caveat (l.79-86)

**What the caveat says** (l.79-86):
- the term reads "which positions hold keys, 3j+1" and "the key identities";
- it reads no stream label and no answer;
- a positive result "does not say the gate discovers the partition from the task's loss alone";
- a language model's substitute is untested.

**Assessment.** It is stated in the spec's own words, and l.122 prints it with every reading. This is the right form, and better than the register draft's (which is in no reading).

**One addition.** In this layout the stream's CTX token is at 3j, the position just before every key. So the window gate sees the cue directly, and the term's knowledge of "3j + 1" is knowledge of the layout's period of 3. Add it to the caveat (fix 9) so that the result is not read as a distant-cue result.

## 11. What the test leaves open

1. **Power.** As drafted, the likeliest outcome at the main line's baseline rate is "not shown", or no room (§7). Fix 1 or fix 2 is needed for the test to be informative.
2. **Whether the key splits are gone or rarer.** K2 is a count with no test.
   - A one-sided McNemar on "KEY failure", WIN3_SLOW over the key term, would have little power: the base rate is 4-5 per 20-40.
   - The count is threshold-sensitive (§4).
3. **Generality.** Only the near-cue grouped layout at S = 2. The register draft carries the term to a distant cue. Neither tests more streams.
4. **The constants.** λ = 1.0, τ = 0.2 and the 2400-update window come from S52.
   - H's round-5 item (3) screens 0.3×, 3× and a 1200 window on seeds 2630-2639. It may amend the draft's constant "only if a variant beats it by ≥ 4 vs 0" (prompts, Session H round 5).
   - The constants must be fixed at adoption. Any later change is a dated amendment before the first run.
5. **Speed.** The key term delays the transition (S85 median 4800 against 2400). This is printed, not claimed.
6. **A label-free substitute.** The caveat says it is untested.

## Fixes, in order (each is replacement text for the adoption; the first-named fix is first)

1. **Seeds and power** (replaces l.92 and adds a line under K1):

   > **Seeds.** X 620-659, L 1620-1659 (40 each), every arm; CEIL_A on the first two seeds (620-621, 1620-1621). Pairs are by seed. (Referee scan, 12 October: no record on the 12 branch heads uses them.)

   Under K1:

   > Printed with K1, fixed now: P(K1 SHOWN) over the 80 pooled pairs is 0.93 at the screens' discordance (S52 + S85: 12 vs 2 of 60). With the key term at 0.95 it is 0.82 / 0.99 against WIN3_SLOW at 0.8125, and 0.51 / 0.84 at the main line's Part A rate of 0.8625 (independent / nested outcomes).

   If 20 seeds per machine are kept, print the 40-pair figures instead: 0.62; 0.46 / 0.66; 0.22 / 0.27.

   With 40 per machine, K3 becomes "≥ 36/40 on a machine" (the RELIABLE band). P(on both machines) is 0.40, 0.76, 0.91 and 0.99 at true rates 0.90, 0.933, 0.95 and 0.975.

2. **No-room rule** (new, after K1):

   > **No room.** If WIN3_SLOW is DISCOVERED on N − 4 or more of the N pooled pairs (36 of 40; 76 of 80), K1 cannot reach p < 0.05 (b ≤ 4 gives p ≥ 0.0625). K1 is then printed "NO ROOM (WIN3_SLOW k/N)", not SHOWN / NOT SHOWN. Every K1 label is printed with WIN3_SLOW's pooled count and McNemar's b and c.

3. **Outcome** (replaces l.75):

   > **Outcome.** DISCOVERED = `test_window_gate.success(key, r)` for a Part A key, at the main line's `0754d61` (`test_window_gate.py:319-325`). It reads run_one's recorded `rec["discovered"]` = the transition is not None and the final VAL cos < `test_router_discovery.DISC_COS` = 0.5 (`test_router_reliability.py:338`; `test_router_discovery.py:173`).
   >
   > CHECK: it gives test_window_gate's recorded Part A counts on this machine's file (WIN3_SLOW X 33/40, L 36/40).

4. **T1, T3 and T4 references** (replaces l.143 and the source clauses of l.148-155):

   > **T1** at claude/outside-ideas `2afa02f` (S52's start commit). Blobs: `explore_keysplits_child.py` `f05e8f9d3b`, `explore_b18_gates.py` `a61a577388`, `explore_aux_gate.py` `ffae342178`; identical at S85's start, `37c1fbf`.
   >
   > **T3 and T4** read their reference records from vendored copies in `results/refs/`, each with its source commit, path and key, as test_delta_recipe's Amendment 2(h):
   > - `W2_KEYHINGE|160|bd2ad99c5016|Intel(R) Xeon(R) Processor @ 2.10GHz` from outside-ideas `f671ab3`, `explore_out/two_stream_keysplits_results.json`;
   > - `KEYHINGE|2310|957abe999f8d|Intel(R) Xeon(R) Processor @ 2.10GHz` from explore-H `1e09ecd`, `explore_out/H/two_b_results.json`.
   >
   > Where the source commit is present, each prints whether the copy equals the commit's record.

5. **Rate claim K3** (replaces l.112-113):

   > - **K3** (the rate claim) WIN3_SLOW_KEYHINGE discovers ≥ 18 of 20 (≥ 36 of 40 if fix 1 is taken) on a machine.
   >   - **Per machine:** MET, NOT MET or UNTESTED. Printed with the count, its Wilson and Clopper-Pearson 95% intervals (at 18/20: [0.699, 0.972] and [0.683, 0.988]), and the band.
   >   - **Pooled (printed beside it):** the count with its Wilson interval and band.
   >   - **The machines compared:** Fisher's exact test, two-sided, printed and not read.
   >   - **Printed, fixed now:** P(MET on both machines) at true rates 0.80, 0.85, 0.90 and 0.95 is 0.04, 0.16, 0.46 and 0.85 (at 40 per machine, for ≥ 36/40: 0.006, 0.069, 0.40 and 0.91).

6. **K2** (replaces l.110-111 and the second sentence of l.122-123):

   > - **K2** (printed, not a claim): WIN3_SLOW_KEYHINGE's and WIN3_SLOW's KEY failures under `fail_class_v2.fail_class_v2`, per machine and pooled.
   >   - Each classified run is printed with η² by key (7 decimals) and its margin.
   >   - The count of runs within 0.001 of η² by key = 0.5 is printed beside it.
   >   - No reading uses K2.

7. **Multiplicity** (new paragraph after l.107):

   > K1 is this test's one hypothesis test, one-sided at 0.05, pooled primary and per machine secondary. K3 is a rule on counts; K2 a printed count. test_register_gate (if adopted) tests the same key term, with S52's constants, at a distant cue. A statement about "the key term" that draws on both tests names both verdicts; neither test's verdict is read for the other.

8. **CHECKs**:
   - T8 gains "to 1e-6".
   - T9's population becomes "`fail_class_v2.reclassify` over the population of `docs/fail_class_v2.md` on this machine's files".
   - New **T12 (key positions):** on the probe and on one training batch, every position 3j + 1, j = 0 … S·P − 1, holds a key token (S ≤ token < S + P), and position 3j holds a CTX token; L = 27 and qpos = 25.

9. **Caveat and validity wording.**
   - Add to l.79-86: "In this layout every key is preceded by its stream's CTX token (positions 3j, 3j + 1), so the window gate sees the cue directly, and 'which positions hold keys' is the layout's period of 3. A positive K1 is a near-cue result."
   - Add to l.88-90: "CEIL_A has the conv layer and the arms do not. The part validates the task, the memory and the path, not the arms' capacity (test_window_gate's rule)."

10. **Readings** (replaces l.115-120; every reading is printed with the configuration "S = 2, P = 4, k = 2, no conv, grouped layout", the caveat, and K2):

    > - K1 SHOWN, K3 MET on both machines: "the key term discovers more often than WIN3_SLOW, and ≥ 18/20 on both machines".
    > - K1 SHOWN, K3 NOT MET on at least one machine: "the key term discovers more often than WIN3_SLOW; ≥ 18/20 not met on <X / L / either>".
    > - K1 NO ROOM: "no room: WIN3_SLOW discovers k of N; K1 cannot be shown", with K3 per machine.
    > - K1 NOT SHOWN, K3 MET on both machines: "≥ 18/20 on both machines; not shown more often than WIN3_SLOW (b, c)".
    > - K1 NOT SHOWN, K3 NOT MET on at least one machine: "not shown (b, c; WIN3_SLOW k/N)".
    > - Either machine NOT VALID: "UNTESTED", and no other reading is given.

    Background, l.26: "(7 vs 2 discordant, two-sided p = 0.18; one-sided 0.090)".

    Runtime, l.175: "about 2 minutes each with early stop (S85: KEYHINGE 1.7-2.9 min, REF 1.3-6.3)".
