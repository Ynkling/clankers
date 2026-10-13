# Spec review: the register test draft (Session G, `explore_out/G/spec_draft_register.md` at `076005d`)

*Referee V, 12 October 2026. Written before the main line adopts the draft into `specs/` and before any run. "l.NN" is a line of the draft at `076005d`. "out §N" is a section of `spec_checks_register_keyterm_out.txt`, written by `spec_checks_register_keyterm.py` (stdlib only; it reads git and nothing else). The project's rules are in `CONTRIBUTING.md` at `a76c4ee`, "Before a run" 1-4. The form follows `specs/test_delta_recipe.md` with its Amendments 1 and 2, on the main line at `0754d61`.*

**Verdict.** The draft is not ready to adopt as written. None of the problems needs a run to fix.

**What checks out:**
- **The background.** Every count in l.29-43 matches G's records (out §2): S71, S72, S82 and S83, re-scored here with ROUTED-BOUND written out from G's definition.
- **The seeds.** No record on any branch uses X 600-619 or L 1600-1619 (out §1).
- **The task file.** `explore_f_tasks.py` is the same blob at `9a6c2bf` and `076005d`.
- **C1-C3 have power near 1** at the screens' rates (§7).
- **The CHECK list is long**, and almost every item can fail.

**What has to change:**
1. **The outcome has a definition but no named function** (l.96-100). No CHECK ties it to G's recorded verdicts, and CONTRIBUTING "Before a run" 4 asks for one.
2. **The model is written fresh** (`register_gate.py`, l.51) rather than copied. Its only link to G's records is the S82 reproduction, which is asserted only on a matching CPU (l.174-177). L (i7-12650H) therefore never asserts that it runs G's model.
3. **The rate claim R** (l.126-127, 135-138):
   - Its threshold of 18/20 supports a rate of at least 0.70, not "reliable".
   - "Reliable on one machine" fires 44% of the time when both machines share a true rate of 0.90.
   - The rate readings have no UNTESTED row.
4. **The readings:**
   - They say "discovers" and "discovered" for ROUTED-BOUND, a name the report reserves for another outcome (v9 draft tex l.114).
   - Reading 2 credits the key term when C3, the test of the term, is not SHOWN.
   - The readings credit "the register" for C2, but C2's comparator lacks both the register and the residual input.
5. **Validity** requires REG3_NUDGE on 2 of 2 seeds (l.109). That is stricter than any of G's screens, and at a nudge rate of 0.95 it makes a machine UNTESTED 10% of the time.
6. **Missing CHECKs:** seeds, pairing of batches, worker identity, resume, and vendored references.
7. **The task-structure caveat** is stated well (l.77-83) but appears in no reading.

The numbered fixes at the end are written as replacement text.

## 1. Arms (l.14-21, 85-94)

**The arms.** Four learned arms, paired by seed, plus two validity arms: 4 × 20 + 2 × 2 = 84 runs per machine (l.184-185, correct). They match report 5's proposal and the round-4 prompt.

**The three contrasts:**

| claim | contrast | what it tests |
|---|---|---|
| C1 | REG3_KEYHINGE vs BASELINE | term + REG3 gate, against neither |
| C2 | REG3_KEYHINGE vs LOCAL3_KEYHINGE | the REG3 gate, given the term |
| C3 | REG3_KEYHINGE vs REG3 | the term, given the REG3 gate |

**The REG3 gate differs from LOCAL3's in two inputs:**
- A_r LN(r^(ℓ)), the residual;
- A_R R_{t−1}, the register (l.63).

The draft's own CHECK says that with A_r = A_R = 0 the two gates are equal (l.158). So C2 tests the register and the residual input together. No arm isolates the register. G has the code path for one: S73's RESGATE_ONLY_D, "Reg3BDH without the register (reg = False)", in the `explore_g_rhl_child.py` docstring at `076005d`. See §8 and fix 10.

**Report 6's POS arm** is rightly left out: 1/20, not discovered.

**Runtime** (out §2). Medians, with ranges, on G's 2.10 GHz Xeon:

| arm | minutes per run |
|---|---|
| REG3_KEYHINGE | 2.2 (2.1-3.0) |
| BASELINE | 6.3 (3.5-6.5) |
| REG3, unbound | 8.5-8.7 |
| LOCAL3_KEYHINGE | 4.6 (2.1-4.7) |
| validity runs | 1.0-1.9 |

This gives about 7 CPU-hours per machine, which agrees with l.186-188. The draft's "an unbound Hebbian run … takes 6-8 min" understates REG3's unbound runs (8.5-8.7 min) and S82's slow-gate runs (up to 9.8). The projection is printed before the run, so this matters little.

## 2. Seeds (l.112-115)

**What I scanned.** Every JSON and JSONL blob at the head of all 12 remote branches (out §1):

| branch | head |
|---|---|
| main line | `0754d61` |
| bdh-repo-curl-obgdqk | `a3e0a97` |
| explore-F | `d5851b1` |
| explore-G | `076005d` |
| explore-H | `5b5deec` |
| explore-I | `559b942` |
| outside-ideas | `46cb8d0` |
| relaxed-edison-he00cc | `3c2fb70` |
| replicate-R | `1f8be04` |
| rev9-draft | `c5e4508` |
| review-V | `b25a4f6` |
| feature/per-channel-plasticity | `4af9d50` |

That is 308 distinct blobs (966 branch-paths). From them I read 28,791 seed values: every key of the form `ARM|seed[|…]`, every `seed` field and every `seeds` list.

**Result.** No record uses any of X 600-619 or L 1600-1619.
- **Integer dictionary keys** equal to 600 and 1600 do occur, but only as update steps or sample indices. Their parent keys are `by_step`, `eta_at`, `dense`, `diag`, `mem`, `spec`, `rho`, `value_at`, `early_evals` and `exp_avg_sq`, plus tolerance-keyed dictionaries in explore-I's `explore_gpu/checks_*_output.json`. Example: `spec` under `D8_HINGE_M|260` is keyed by update 0, 100, …
- **Text** (`*.py`, `*.md`, `*.tex`, `*.txt` on every head). The blocks are mentioned only by:
  - the proposals themselves;
  - the main line's round-5 prompts (`docs/reading/parallel_sessions_prompts.md:791, 954`);
  - the referee's earlier check (`rev9/check_G_v4.*`);
  - two source-line ranges (`registry_L_recount.py:163`, `test_check_delta_recipe.md:29`), which are false matches.

**Neighbouring seeds** (out §1):
- In use: X 500-569 and 580-599 (580-599 is `test_delta_recipe` on X, recorded at `8ae660d`); L 1500-1569.
- Reserved but not yet recorded: L 1580-1599 (`test_delta_recipe`'s L half).
- Free: 640-699 and 1640-1699. Seed 700 is G's S71-S73.

**Round 5.** The blocks assigned for round 5 (prompts, "Round 5": F 2400-2459, H 2600-2659, G 2700-2759, R 940-959, I 2540-2559 and 840-859) are disjoint from these. So is H's draft (X 620-639, L 1620-1639).

**Derived generators.** They do not meet:
- REG3's parameters, seed + 56000: X 56600-56619, L 57600-57619.
- The nudge, seed + 55555: 56155-56174, 57155-57174.
- The window, seed + 31337.
- The probe, seed + 99000.

**Verdict on l.112-115.** The statement is true at today's heads. The draft has no CHECK that enforces it, however (main line 168, H's T10). Fix 7(d).

## 3. Validity (l.20-21, 109-110)

**The rule.** One part covers every claim. A machine is VALID if:
- CEIL binds both validity seeds; and
- REG3_NUDGE is ROUTED-BOUND on both.

**CEIL is safe.** S71 ORACLE_ALL bound 10/10; S82 2/2 at 1200 (out §2).

**The REG3_NUDGE half is the risk.** 24 of 24 nudged runs are recorded as routed (S72 20/20, S82 2/2, S83 2/2). The Clopper-Pearson 95% lower bound on that rate is 0.858. Two of two passing is then a random event (out §7):

| nudge rate | P(machine VALID) | P(both machines VALID) |
|---|---|---|
| 0.99 | 0.980 | 0.961 |
| 0.95 | 0.902 | 0.815 |
| 0.90 | 0.810 | 0.656 |
| 0.858 | 0.735 | 0.541 |

**G's own rules were weaker:**
- S82 coded VALID as "ORACLE ≥ 1 of 2" and only printed the nudge (`explore_g_devices.py:120`).
- S83 printed the nudge (`explore_g_s83.py:22`).

**What a failed nudge would do.** It would make the machine's *positive* C1 UNTESTED too, and so would discard the result the test exists to read.

**What each check shows:**
- CEIL shows that the task, memory and path can bind on that machine.
- A single routed nudged run shows that the machine's REG3 code path can carry the stream.
- A second success adds nothing to validity. At a rate below 1, it only adds risk.

Fix 3 requires "≥ 1 of 2". At the 0.858 bound that passes 0.980 per machine.

## 4. Outcome: ROUTED-BOUND (l.22, 96-107)

**The definition matches G's code exactly** (`explore_g_rhl_parent.py` at `076005d`):

| line | name | definition |
|---|---|---|
| :35 | ROUTE_MARGIN, ROUTE_ETA | 0.9, 0.9 |
| :39 | `bound` | the recorded transition is not None |
| :43 | `routed_star` | end margin ≥ 0.9 and end `eta_key_by_stream` > 0.9; a missing statistic counts as 0 |
| :52 | `routed_bound` | `bound` and `routed_star` |

I wrote this out from the definition and re-scored every record (out §2). It gives every count G reports:

| screen | arm | ROUTED-BOUND |
|---|---|---|
| S82 | BASELINE | 0 |
| S82 | REG3 | 2 |
| S82 | REG3_KEYHINGE | 20 |
| S82 | REG3_SLOWGATE | 0 |
| S82 | REG3_NOWIN | 5 |
| S83 | LOCAL3_KEYHINGE | 0 |
| S83 | REG3_KEYHINGE_POS | 1 |
| S83 | REG3_FLAT | 0 |
| S83 | REG3_KEYHINGE_1200 | 18 |
| S72 | BASELINE | 0 |
| S72 | REG3 | 4 |
| S72 | REG3_NUDGE | 20 |
| S82, S83 | REG3_NUDGE | 2 + 2 |

**What checks out:**
- `test_binding_onset.transition`, which l.98 names, exists on the main line (`test_binding_onset.py:247`, BIND_PASS 0.95, `:113`).
- The recorded `transition` equals `transition(curve)` on all 186 S82/S83 records (out §2).

**(a) Not fixed by a named function.**
- The draft says "a function, fixed now" (l.96) but names none. CONTRIBUTING "Before a run" 1 asks for "the outcome fixed by a named function".
- The outcome CHECK (l.155-156) tests constructed records only. Rule 4 asks that "its outcome functions agree with the recorded verdicts".

Fix 1 names the function, copies it from `explore_g_rhl_parent.py`, and CHECKs it against G's counts above.

**(b) The thresholds are not knife-edge on the record.**
- Of the 60 bound Hebbian records in S82/S83, none has an end margin or η² by stream in [0.75, 0.98).
- The margins are 1.00 (52 records) or 0.44-0.51 (8 records). The 8 are the key-split bindings of BASELINE (3: 2106, 2108, 2116) and LOCAL3_KEYHINGE (5: 2100, 2103, 2104, 2106, 2116).

So what ROUTED-BOUND does on this record is exclude key-split bindings. BOUND alone would count LOCAL3_KEYHINGE 5/20 and BASELINE 3/20.

**(c) The layer.**
- The outcome reads the gate the model returns: layer 3 for REG3, the one gate for LOCAL3 (l.99-100).
- S71, which the draft cites (l.29-31), found that layer 1's routing is the one that matters.
- On the record the choice does not matter. All 52 bound records that are ROUTED* on the returned gate are ROUTED* at every layer, and the 8 that are not are ROUTED* at no layer (out §2).

Keep the definition. Print any disagreement in the report (fix 7(g)).

**(d) The probe is not named.**
- `rand_header_routing_stats(model, task, probe)` reads run_one's statistics probe, `probe_batch(task, seed)` (`test_router_reliability.py:303`). That probe is `test_multilayer_binding.py:352`: its own generator, seed + 99,000, PROBE_B sequences.
- The draft names neither. A different probe gives a different margin. Fix 1.

**(e) Three names for three outcomes, differing by word order:**
- ROUTED-BOUND, the claim;
- the main line's BOUND ROUTED (`test_stream_recipe.outcome`);
- the draft's printed "BOUND ROUTED with a margin guard" (l.104-105).

The printed one is not G's "BOUND ROUTED (mine)" either. That one (`explore_g_rhl_parent.py:56`) has no margin guard, as report 5's footnote says. Fix 9 renames the printed one.

## 5. Claims and their multiplicity (l.23-25, 117-127)

**The family.**
- Three pooled exact one-sided McNemar tests (C1-C3), primary.
- Six per-machine McNemar tests, secondary.
- A rate rule R per machine.

The draft names no family and states no correction. Fix 6 adds the statement.

**The readings need no correction.** Each reading requires every claim it names to be SHOWN, an intersection-union rule, so it holds at 0.05 as it stands. A sentence quoted from one claim is per-comparison; report 6's "the register is needed" is an example.

**Holm over C1-C3 costs almost nothing at the screens' effects** (out §5):
- With c = 0, b ≥ 6 shows at 0.05/3, against b ≥ 5 at 0.05.
- The power of C3 at 0.05/3 stays at 0.990 with REG3_KEYHINGE at 0.80 against REG3 at 0.30, and 0.897 at 0.70.
- In S82/S83 the pairs were 20 vs 0, 20 vs 0 and 18 vs 0 (p = 9.5e-7, 9.5e-7 and 3.8e-6).

**For the claims registry** (`claims_registry.md`), the test adds 3 pooled tests, 6 per-machine tests and 2 rate rules.

## 6. The rate claim R: its interval and operating characteristics (l.25, 126-127, 135-138)

**The interval at the threshold** (out §4):

| count | Wilson 95% | Clopper-Pearson 95% |
|---|---|---|
| 18/20 | [0.699, 0.972] | [0.683, 0.988] |
| 19/20 | [0.764, 0.991] | [0.751, 0.999] |
| 20/20 | [0.839, 1.000] | [0.832, 1.000] |
| 36/40 | [0.769, 0.960] | [0.763, 0.972] |
| 38/40 | [0.835, 0.986] | [0.831, 0.994] |
| 40/40 | [0.912, 1.000] | [0.912, 1.000] |

So "R holds on a machine" supports a rate of at least about 0.70. No per-machine count supports 0.9; even 20/20 supports only 0.84.

**Operating characteristics, by exact binomial** (equal true rates, machines independent):

| true rate | ≥ 18/20 on a machine | on both | on exactly one | pooled ≥ 36/40 |
|---|---|---|---|---|
| 0.80 | 0.206 | 0.042 | 0.327 | 0.076 |
| 0.85 | 0.405 | 0.164 | 0.482 | 0.263 |
| 0.90 | 0.677 | 0.458 | 0.437 | 0.629 |
| 0.95 | 0.925 | 0.855 | 0.140 | 0.952 |

**What the screens predict.**
- From S82's 20/20, the predictive P(≥ 18/20 on both machines) is 0.80 with a uniform prior and 0.91 with Jeffreys.
- If S83's 1200-update arm (18/20) is counted as evidence about the same rate, it is 0.65-0.75 (out §4).
- So "reliable on both machines" is likely, but far from certain, even if the screens transfer.

**The problems:**
- **"Reliable on one machine" is mostly noise.** At equal true rates of 0.85-0.90 it fires 44-48% of the time. The reading suggests a difference between machines that nothing tests.
- **"Pooled: HOLDS if it holds on both machines" (l.126-127) is not monotone in the pooled count.** X 20/20 with L 17/20 is 37/40 and fails; 18/20 with 18/20 is 36/40 and passes.
- **CONTRIBUTING "Before a run" 1** wants claims "pooled over both machines primary, per machine secondary". R as drafted is per machine only. The user's prompt asks for "≥ 18/20 per machine", so keep it, and add the pooled count beside it.
- **The rate readings (l.135-138) have no row for an UNTESTED machine.** Of the 9 combinations of {holds, fails, UNTESTED} per machine, 5 print nothing defined (out §6).

## 7. McNemar power at the screens' effect sizes (out §5)

REG3_KEYHINGE at rate p1 against a comparator at p2. Each cell gives P(SHOWN) as "40 pooled pairs; 20 pairs per machine", each as outcomes independent / nested.

| p1 | p2 = 0.00 | p2 = 0.10 | p2 = 0.16 | p2 = 0.30 |
|---|---|---|---|---|
| 0.50 | 1.000 / 1.000; 0.994 / 0.994 | 0.989 / 1.000; 0.794 / 0.949 | 0.929 / 1.000; 0.611 / 0.863 | 0.489 / 0.924; 0.233 / 0.370 |
| 0.70 | 1.000 / 1.000; 1.000 / 1.000 | 1.000 / 1.000; 0.987 / 1.000 | 1.000 / 1.000; 0.950 / 0.998 | 0.965 / 1.000; 0.718 / 0.949 |
| 0.80 | 1.000 / 1.000; 1.000 / 1.000 | 1.000 / 1.000; 0.999 / 1.000 | 1.000 / 1.000; 0.993 / 1.000 | 0.998 / 1.000; 0.906 / 0.994 |
| 0.90 | 1.000 / 1.000; 1.000 / 1.000 | 1.000 / 1.000; 1.000 / 1.000 | 1.000 / 1.000; 1.000 / 1.000 | 1.000 / 1.000; 0.987 / 1.000 |

**The comparators' recorded rates.** BASELINE 0/20 (Wilson upper 0.16); LOCAL3_KEYHINGE 0/20 (0.16); REG3 2/20 (0.30), and 4/20 in S72.

**Reading the table:**
- C1 and C2 are near-certain whenever REG3_KEYHINGE is above 0.5.
- C3 is at least 0.965 pooled whenever REG3_KEYHINGE is at least 0.7.

The test's power is not in question. What carries the uncertainty is R (§6) and what C2 is allowed to say (§8).

## 8. The readings: exhaustiveness and wording (l.129-138; enumerated in out §6)

| C1 | C2 | C3 | reading printed (draft) |
|---|---|---|---|
| S | S | S | "the register, released by the key term, discovers the stream" |
| S | S | N | "the term and the register discover it; the register alone is not shown worse" |
| S | N | S | "discovered, but the register is not shown to be needed" |
| S | N | N | "discovered, but the register is not shown to be needed" |
| N | any | any (4 rows) | "not replicated" |

**Coverage.** Each of the 8 combinations gets exactly one name, so there is no gap or overlap among C1-C3. UNTESTED is handled ("if any machine is not VALID"). The R readings have the gap described in §6.

**What the names get wrong:**

**(a) "Discovers / discovered."**
- The test's outcome is ROUTED-BOUND. DISCOVERED (bound and VAL cos < 0.5) is printed beside it as a different outcome (l.103).
- The report reserves the name: the v9 draft tex l.114 says "the name DISCOVERED is reserved for the outcome above, since one screen's reading named 'discovered' was fixed on ROUTED-BOUND and would have come out differently".
- These readings would repeat that.

**(b) Reading 2 credits the term when its test fails.**
- It fires when C3 (REG3_KEYHINGE vs REG3, the test of the key term) is not SHOWN.
- Yet it says "the term and the register discover it". That credits the term in exactly the case where its contribution was not shown.

**(c) "The register" for C2.**
- LOCAL3_KEYHINGE lacks both the register and the residual input (§1), so C2 cannot speak of the register alone.
- Readings 1 and 3 do.

**(d) "Not replicated" also covers a case it misdescribes.**
- It is printed whenever C1 is not SHOWN. That includes REG3_KEYHINGE at 40/40 against a BASELINE at 36/40 (4 vs 0, p = 0.0625).
- The rate reading would then print "reliable on both machines" beside it.
- The case is implausible (BASELINE has been 0/20 three times), but the name should fit every outcome it covers.

**(e) "Released by"** names a mechanism, and the test measures none. β^R and η²_key are printed, not claimed (l.141-142).

## 9. The CHECKs (l.149-180)

| CHECK (draft) | can it fail? | before any run? | note |
|---|---|---|---|
| inherited chain (l.151) | yes | yes | |
| the task (l.152-154) | yes | yes | `explore_f_tasks.py` is blob `de47463` at `9a6c2bf` and `076005d` (out §2): "byte for byte" holds |
| the outcome function (l.155-156) | yes | yes | constructed record only; fix 1 adds recorded verdicts |
| A_r = A_R = 0 gives LOCAL3, bit for bit (l.158) | yes | yes | |
| initial parameters equal BASELINE's (l.159) | yes | yes | |
| "β^R ≈ 0.1 at initialisation" (l.160) | **no**: "≈" has no tolerance | yes | G's own CHECK asserts 0.09 < β^R < 0.11 (`explore_g_rhl_child.py:466-468`) |
| causality (l.161-162) | yes | yes | every layer's gate at t + 1 reads R_t, so "layer 1's" is a weaker form than G's CHECK (d) |
| stop-gradient (l.163) | yes | yes | |
| SLOW groups (l.164-165) | yes | yes | |
| key term: gradient, η² by hand, schedule, reach (l.166-171) | yes | yes | |
| switch off equals base through 2400 (l.172-173) | yes | yes | |
| S82 reproduction (l.174-177) | yes, on a matching CPU only | yes | see below |
| the nudge's "stream lean > 0.02" (l.178) | **no**: "lean" is not defined | yes | G defines it as \|mean p₀ at stream-0 positions − mean p₀ at stream-1 positions\| per layer (`explore_g_rhl_child.py:514`) |
| the perfect gate (l.179-180) | yes | yes | |

**The S82 reproduction is the only link between the new module and G's records:**
- X runs on a 2.10 GHz Xeon (`results/X/delta_recipe_results.json`: 84/84 runs), so it asserts there.
- L runs on an i7-12650H (`results/L/split_copy_results.json`: 54/54), so it only prints there.
- The draft writes the model fresh (l.51). `test_split_copy` copied F's delta files byte for byte for exactly this reason (its CHECK 152).
- A CPU-independent equality against G's classes, extracted from `076005d` and run on every machine, closes the gap (fix 2).
- The reproduction also covers only BASELINE, REG3 and REG3_KEYHINGE. LOCAL3_KEYHINGE (S83's `TermLocal`), REG3_NUDGE and CEIL are new code with recorded references too.
- Its references live on an exploratory branch. `test_delta_recipe` Amendment 2(h) vendored such references into `results/refs/`.

**Missing CHECKs:**
- **Seeds** (main line 168, H's T10).
- **Pairing:** identical first training batches across the four learned arms, and the nudge drawing only from its own generator.
- **Worker identity and resume** (main line 169-170).
- **In the report:**
  - fail_class_v2 against v1 on this test's unbound runs (main line 171);
  - the per-layer ROUTED* agreement of §4(c).

## 10. The task-structure caveat (l.77-83)

**What the caveat says** (l.77-83):
- The term "reads which tokens are keys and which key each is".
- It "reads no stream label, no CTX token and no block boundary".
- It is "task structure … a prior about the data, not a property of the architecture".
- "The claims below are about this term on this task and nothing wider."

**Assessment.** This is accurate and well put. The term finds key positions by token class (`body_positions`: S ≤ token < S + P) and groups them by token − S (`explore_g_devices_child.key_eta`).

**What is missing:**
- **It is in no reading and no claim.** A reading lifted out of the spec, as reading 1's "the register … discovers the stream", carries none of it. Fix 4 adds a caveat line to every reading.
- **It interprets S83 inside the spec.** "Both hold the gate at uniform and starve the register" (l.82-83) is G's account of S83's traces. Label it as G's reading of S83 (fix 4).

## 11. What the test leaves open

1. **Whether the register is what is needed.** C2 removes the register and the residual input together. An arm with the register switched off (reg = False, as in S73's RESGATE_ONLY_D) plus the key term would separate them (fix 10, optional).
2. **Whether a term that reads less would do.** The test fixes the key term, which reads key identities. G's round-5 S87 (seeds 2700-2759) screens hinges that read no labels. Any replacement would need its own test.
3. **The window.** 2400 updates is fixed. S83's 1200-update arm was one pair short (18/20, "one pair short of 'the short window suffices'").
4. **Delta memory.** S82's REG3_KEYHINGE_D 10/10 and other configurations (report 5's four-stream suggestion) are not in the test.
5. **The rate.** At the threshold, R supports at least 0.70. The test cannot show a rate of 0.9 on either machine (§6).
6. **Timing.** Transitions (S82: 3600 in 18 of 20) and the β^R separation are printed, not claimed.

## Fixes, in order (each is replacement text for the adoption; the first-named fix is first)

1. **Outcome** (replaces l.96-100, keeping l.101-107):

   > **Outcome (a function, fixed now).** ROUTED-BOUND(r) = `test_register_gate.routed_bound(r)`, copied from `explore_g_rhl_parent.py` at `076005d` (`bound` :39, `routed_star` :43, `routed_bound` :52, thresholds :35). It holds when:
   > - `r["transition"] is not None`, where the transition is run_one's `test_binding_onset.transition(curve)` (BIND_PASS 0.95);
   > - `r["end"]["margin"] >= 0.9`;
   > - `r["end"]["eta_key_by_stream"] > 0.9`.
   >
   > A missing statistic counts as 0. The end statistics are `explore_f_tasks.rand_header_routing_stats(model, task, probe)`, where:
   > - the probe is run_one's statistics probe `probe_batch(task, seed)` (generator seed + 99,000);
   > - the gate is the one the model returns (layer 3's for the REG3 arms, the one gate for LOCAL3).
   >
   > **CHECK (before any run).** On every record of `explore_out/G/s82_devices_results.json` and `s83_results.json` at `076005d` (vendored in `results/refs/` with commit, path and key), `routed_bound` gives Session G's counts: BASELINE 0/20, REG3 2/20, REG3_KEYHINGE 20/20, REG3_SLOWGATE 0/20, REG3_NOWIN 5/20, LOCAL3_KEYHINGE 0/20, REG3_KEYHINGE_POS 1/20, REG3_FLAT 0/20, REG3_KEYHINGE_1200 18/20, REG3_NUDGE 2/2 and 2/2. The recorded `transition` equals `test_binding_onset.transition(curve)` on all 186 records.

2. **Model equivalence on every machine** (new CHECK, after l.165):

   > **The model against Session G's code, CPU-independent.**
   > - The comparison: at seed 600, on 16 training sequences, with the parameters `register_gate.py` builds:
   >   - REG3's and REG3_KEYHINGE's logits, every layer's gate, β^R and every parameter gradient, against the same from `explore_g_rhl_child.Reg3BDH` and `explore_g_devices_child.KeyHingeBDH`;
   >   - LOCAL3_KEYHINGE's, against `explore_g_s83_child.TermLocal`;
   >   - G's classes extracted from `076005d` and run on the same parameters.
   > - Tolerance: bit for bit, or, where the operation order differs, to 1e-6 relative, with the measured difference printed.
   > - Asserted on every machine.
   > - The S82 reproduction below is asserted only on a matching CPU; this CHECK is the link on L.

   Alternative: copy G's model and term code byte for byte, as `test_split_copy` copied F's delta files (its CHECK 152), and CHECK the syntax trees.

3. **Validity** (replaces l.109-110):

   > **Validity.** A machine is VALID if:
   > - CEIL binds on both of its validity seeds (test_window_gate's rule, VALID_N = 2); and
   > - REG3_NUDGE is ROUTED-BOUND on at least one of them. One routed nudged run shows that the machine's REG3 code path can carry the stream. S82's rule (`explore_g_devices.py:120`) required only the oracle.
   >
   > Both counts are printed. A claim is UNTESTED on a machine that is not VALID. A pooled claim is UNTESTED unless both machines are VALID.

4. **Readings** (replaces l.129-133):

   > **Readings** (pooled; UNTESTED if either machine is not VALID; each combination of C1-C3 has exactly one name). Every reading is printed with the caveat line below.
   > - C1, C2 and C3 SHOWN: "REG3 with the key term routes and binds more often than the baseline, than the key term on the window gate, and than REG3 without the term."
   > - C1 and C2 SHOWN, C3 not SHOWN: "REG3 with the key term routes and binds more often than the baseline and than the key term on the window gate; not shown more often than REG3 without the term."
   > - C1 SHOWN, C2 not SHOWN (C3 either way): "REG3 with the key term routes and binds more often than the baseline; not shown more often than the key term on the window gate."
   > - C1 not SHOWN: "REG3 with the key term not shown to route and bind more often than the baseline", with both arms' pooled counts.
   >
   > Caveat line: "The key term reads which tokens are keys and each key's identity, during updates 1-2400 only. REG3 differs from the window gate by the register and the residual input together."
   >
   > "Discovers" is not used: the report reserves DISCOVERED (bound and VAL cos < 0.5), which is printed beside ROUTED-BOUND.

   In l.82-83, after "starve the register", add "(Session G's reading of S83's traces)".

5. **Rate claim** (replaces l.126-127 and l.135-138):

   > - **R** REG3_KEYHINGE ROUTED-BOUND ≥ 18/20 on a machine (the RELIABLE band at n = 20).
   >   - **Per machine:** MET, NOT MET or UNTESTED (machine not VALID). Printed with the count, its Wilson and Clopper-Pearson 95% intervals (at 18/20: [0.699, 0.972] and [0.683, 0.988]), and the band.
   >   - **Pooled (printed beside it):** the count with its Wilson interval and band (RELIABLE ≥ 36/40).
   >   - **The machines compared:** Fisher's exact test, two-sided, printed and not read.
   >   - **Printed with R, fixed now:** P(≥ 18/20 on both machines) is 0.04, 0.16, 0.46 and 0.85 at true rates 0.80, 0.85, 0.90 and 0.95.
   >
   > The rate reading is one of:
   > - "R MET on both machines";
   > - "R MET on X only";
   > - "R MET on L only";
   > - "R MET on neither";
   > - "R UNTESTED on <machine>", with the other machine's verdict printed.
   >
   > None of these is read as a difference between the machines.

6. **Multiplicity** (new paragraph after l.121):

   > C1-C3 are a family of three one-sided tests at 0.05 each, with no correction. The readings are intersection-union rules: every claim a reading names must be SHOWN, so each reading holds at 0.05. Holm-adjusted p-values over C1-C3 are printed beside the raw ones. For the registry, this test adds 3 pooled tests, 6 per-machine tests and one rate rule per machine.

7. **CHECKs** (changes and additions):
   - (a) l.160 becomes "β^R at initialisation lies in (0.09, 0.11) at every position of 16 sequences (Session G's CHECK, `explore_g_rhl_child.py:466-468`)".
   - (b) l.178 becomes "The nudge moves only the REG3 parameters. After it, the stream lean \|mean p₀ at stream-0 positions − mean p₀ at stream-1 positions\| on the probe is > 0.02 at every layer (`explore_g_rhl_child.py:514`)."
   - (c) l.174-177: the reproduction covers BASELINE, REG3, REG3_KEYHINGE and REG3_NUDGE (S82, seed 2100), LOCAL3_KEYHINGE (S83's `s83_results.json`, seed 2100) and CEIL against ORACLE_ALL|2100. Each reference is read from a vendored copy in `results/refs/` with its source commit (`076005d`), path and key, and where the commit is present the CHECK prints whether the copy equals it (as test_delta_recipe Amendment 2(h)).
   - (d) New: "**Seeds.** Every arm runs on the machine's range (X 600-619, L 1600-1619); CEIL and REG3_NUDGE run on its first two seeds. The ranges are disjoint from every main-line seed and from the other machine's range. With --also, the other file's machine, CPU and seeds are printed. (Referee scan, 12 October: no record on the 12 branch heads uses them.)"
   - (e) New: "**Pairing.** At seed 600, BASELINE, REG3, REG3_KEYHINGE and LOCAL3_KEYHINGE draw identical first three training batches, and their shared initial parameters are identical. REG3_NUDGE's nudge draws only from its own generator (seed + 55555)."
   - (f) New: "**Workers.** A worker's run is bit-identical to the same run in the parent (REG3_KEYHINGE, seed 1, 1200 updates). **Resume.** Records cached by an earlier start reproduce under this code through 2400."
   - (g) New (in the report):
     - the count of runs where ROUTED* on the returned gate differs from ROUTED* at every layer, listed;
     - fail_class_v2 against v1 on every unbound run, with η² by key and the margin of each disagreement.

8. **Seeds paragraph** (l.112-115): add "Re-checked by the referee on 2026-10-12: all 12 remote branch heads, 308 JSON/JSONL blobs, every `ARM|seed` key, `seed` field and `seeds` list. No hit in 600-619 or 1600-1619. The integer keys 600 and 1600 that occur are update steps."

9. **Printed outcome name** (l.104-105): "BOUND ROUTED with a margin guard" becomes "ROUTED AT EVERY LAYER: bound, the VAL stream → channel map one-to-one at every layer, the routing margin ≥ 0.9 at every layer, and every stream's accuracy ≥ 0.9 (printed, not a claim)".

10. **Optional, if C2 is to speak of the register.** Add an arm:

    > RESGATE_KEYHINGE: REG3 without the register (`reg = False`, as S73's RESGATE_ONLY_D), plus the key term for updates 1-2400, 20 seeds per machine; and **C4** REG3_KEYHINGE beats RESGATE_KEYHINGE.

    This adds about 1-2 CPU-hours per machine. Without it, fix 4's wording stands.
