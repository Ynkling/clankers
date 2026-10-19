# Referee note: the five ways forward for the key-load draft after batch 21

*Referee V, 19 October 2026. Before any choice and before any run. The options are the user's to choose. This note says what a referee would require before each one ran. It does not choose.*

*Sources and how they are cited:*
- *"draft l.NN": `explore_out/spec_draft_keyload.md` at `419368a` on claude/outside-ideas.*
- *"README l.NN": `explore_out/README.md` at `419368a`.*
- *"b21 l.NN": `explore_out/batch21.log` at `419368a`.*
- *"out l.NN": `keyload_options_checks_out.txt`, written by `keyload_options_checks.py`. The script is stdlib only, reads git objects and nothing else, and trains nothing. It recounts from `oracle_d128_results.json` and `s67_continue_results.json` at `419368a` and `capacity_results.json` at `e3f5809`.*
- *"CONTRIBUTING l.NN": the main line's `CONTRIBUTING.md` at `7c4ceb8`.*
- *"register review l.NN": `spec_review_register_draft.md` in this folder.*

## What holds for every option

**1. The draft's 0.13 understates the problem.**
- 0.13 = 4·3/90 is the chance that two seeds drawn *from S69's ten* both bind (draft l.12-13; out l.47).
- The test's validity seeds are fresh (660-661 and 1660-1661, draft l.121), so the right model is binomial at the observed rate: 0.16 per machine.
- The pooled primary needs both widths VALID on *both* machines (draft l.114-115), so its chance is 0.026 for D128 alone (out l.52).
- Counting D32 as well, it is 0.010-0.026 (out l.74).

**2. The D32 oracle is unscreened at this budget.**
- Its P = 16 validity rests on S67's two seeds (24,000 and 27,600 at a 43,200 budget; out l.34). S69 did not run it (README l.803).
- At 2/2 the rate is not pinned down: Wilson lower bound 0.342.
- VALID_N = 2 on both machines then has chance 0.014 at that bound, 0.43 under a uniform predictive and 0.60 under Jeffreys (out l.66-69).
- Any option that keeps CEIL_D32 at VALID_N = 2 needs a D32 oracle screen at 40,800 before the test, sized as in option 1.

**3. On these records, the D128 oracle does not predict the learned arm, seed by seed.**
- Of the four seeds whose oracle bound, two had their learned run bound (S67). Of the six whose oracle did not bind, two had their learned run bound. Fisher p = 1 (out l.37-39).
- Learned 270 bound at 38,400. Its oracle was still unbound at 86,400 (out l.28, l.40).
- Under SLOW the oracle bound 0/10 and the learned gate 4/10 (out l.16, l.41; README l.796-798).
- So a failed D128 oracle is mostly a draw from a roughly 0.4 process that is unrelated to the claim. It does not show that the path is broken (draft l.17). A validity rule must be sized so that such a draw rarely discards the test.

**4. Batch 21's own rule rejected VALID_N = 1 at this count.**
- The pre-registered spec reading was "b_A ≥ 9 keep; else b_S ≥ 9 SLOW; else max ≥ 5 VALID_N = 1; else no D128 validity check" (b21 l.1431). With b_A = 4 and b_S = 0 it printed "no D128 validity check at 40,800" (b21 l.1436).
- Any option that relaxes validity is therefore an amendment made after seeing S69. It must be dated and must say so (CONTRIBUTING l.27-28, l.41-42).

**5. The learned budget is not reopened by any option.**
- 40,800 came from S68's pre-registered rule (README l.750-752).
- Changing it because of S69, which measured a different arm, would be choosing the budget after seeing results.
- Option 1 changes the oracle's budget only. That is allowed only under a rule fixed before a fresh screen.

**6. Some requirements apply to every option that runs.** Keep:
- the inherited chain;
- CHECKs 174-182 (draft l.172-196);
- the K1/K2 thresholds, which re-derive correctly: 13/40 at α = 0.043 with power 0.87 at 0.4, and 8/20 at α = 0.032 with power 0.58 (out l.101-105; draft l.139-145).

Every changed validity arm needs a reference record in CHECK 174, vendored into `results/refs/` with its commit, path and key (register review l.438).

## The options

**Option 1. A longer budget for the D128 oracle only (draft l.20-21).**

*What it can reach.*
- On S69's own seeds the ceiling is 9/10, because 270 was still unbound at 0.376 at 86,400 (out l.28, l.95-96).
- The plausible gain is the three close seeds. They are 275, 277 and 279, at 0.91-0.92, which reached 0.94-0.95 between 28,800 and 38,400 but never held it (out l.11-15; README l.799-801). That would make the rate about 0.7.
- At 0.7, VALID_N = 2 passes 0.49 per machine and 0.24 on both machines (out l.55). Even at 0.9 it passes only 0.66 on both (out l.56).
- So option 1 alone does not rescue VALID_N = 2. To pass on both machines with chance ≥ 0.8, it needs either a rate near 0.95 (0.815, out l.57) or option 3's ≥ 1-of-n rule (0.83 at 0.7 with 1 of 2, out l.55).

*Required before the test: a screen on fresh seeds.*
- The seeds must not be 270-279: their curves were seen when this option was written. They must be disjoint from 660-679 and 1660-1679 and from every block in draft l.121-131.
- At least 10 D128 oracle seeds (single Adam, as in CEIL_D128), run to a cap fixed now (e.g. 86,400, S68's length).
- Its reading must be fixed in the docstring before any run (CONTRIBUTING l.29-31), and should reuse S69's bands (reliable ≥ 9/10, b21 l.1415). At n = 10 that rule passes 0.914 at a true rate of 0.95, 0.736 at 0.9 and 0.149 at 0.7. At n = 20 with ≥ 18/20 it passes 0.925, 0.677 and 0.035 (out l.84-86). Ten seeds tell 0.95 from 0.7; twenty are needed to tell 0.9 from 0.8 at all.
- The oracle's budget must come from a rule written before that screen. S68's form would do: the latest transition among the screen's bound oracles, plus 2400, rounded up to a multiple of 2400 (README l.750-751).
- The same screen at D32 (note 2).

*Required in the test.*
- The validity rule is restated with its pass probability at the screen's rate printed beside it, as register review l.420 did for R.
- Cost: the D128 oracle is 6.0 h at 40,800 worst case (b21 l.12), so roughly 12.7 h at 86,400 per validity seed.
- CHECK 182 is needed (checkpoints), with resumes recorded cumulatively. Batch 20's 44,400 check failed on exactly that (README l.753-757; draft l.194-196).
- CHECK 174's CEIL_D128 reference can be extended past 2400 against S68's regenerated ORC_P16_D128|270, which equalled S67 at 43,200 (README l.737).

*What the claim can say.* "The D128 path binds P = 16 with the routing given within <B> updates on this machine."

*What it cannot say.*
- It cannot say the path binds within the learned arms' 40,800.
- A K1 or K2 that is NOT SHOWN therefore cannot rule out "D128 needs longer", which S68 already printed ("86,400 is not enough", README l.751-752).
- The NOT SHOWN rows (draft l.153-155) must keep "at this budget". The "UNTESTED at this budget" row (draft l.151) must name the oracle budget and the learned budget separately.

**Option 2. A different D128 validity check: the D128 oracle at P = 8 (draft l.22-23).**

*The record and what it gives.*
- S67's P = 8 oracles bound 2/2 at both widths, all at 2400 (out l.35-36). That is cheap and fast, but 2/2 is all there is.
- At the 2/2 Wilson lower bound (0.342), VALID_N = 2 passes 0.014 on both machines (out l.69). Under a Jeffreys predictive it passes 0.60 (out l.68).
- So a screen comes first: at least 10 fresh seeds of the P = 8 oracle at D128, *and* at D32 (note 2), each run to the test's budget. If they bind at 2400 as S67's did, each run stops at about 4800-7200.
- Even 10/10 supports only a rate ≥ 0.722 (out l.89). At that bound VALID_N = 2 passes 0.27 on both machines and ≥ 1 of 2 passes 0.85 (out l.116).
- So the referee would pair option 2 with ≥ 1 of 2. A single bound P = 8 oracle already shows what this check can show: the D128 code path, model and memory bind on this machine (register review l.130-135; test_window_gate.md l.65-70 on the main line).

*Symmetry.* Both widths' validity must move to P = 8. If D128 is checked at P = 8 and D32 at P = 16, the arms are certified for different things, and K1 compares them across the difference.

*CHECKs and task.*
- CHECK 174's oracle references change to ORC_P8_D128|270 and ORC_P8_N256|270 at `e3f5809`. Through 2400 these cover the transition itself (out l.35-36).
- CHECK 178 must also register and check `task_for(8, 4)` as "P8S4", the task S67's oracles ran.

*What the claim can say.* The draft says it: this validates the D128 model and path, not P = 16 capacity (draft l.22-23).

*What it cannot say.*
- A NOT SHOWN K1 or K2 cannot be separated from "the perfect gate itself binds only about 4/10 at P = 16, D = 128 in 40,800" (out l.5).
- The readings must carry that line. A reading that appeals to capacity is not available; S67 already found the oracle gives no capacity reason for D128's lead (README l.717-718).

*A variant the referee would prefer.* The positive control could run the arms' own recipe instead of the oracle's single Adam: KL_D128 at P = 4, which bound 10/10 at D128 with transitions about 3600 (README l.631, l.644-645). The oracle's recipe is not the arms' recipe; under SLOW the oracle never bound (out l.16). This variant needs the same screen, rule and CHECKs.

**Option 3. VALID_N = 1 at D128 with the recorded recipe (draft l.24).**

*It is an amendment after S69.*
- S69 pre-registered VALID_N = 1 only for max(b_A, b_S) ≥ 5, and the count was 4 (b21 l.1431). So this is an amendment made after S69, dated and saying so (note 4).
- The referee's own precedent supports the form: "a second success adds nothing to validity; at a rate below 1 it only adds risk" (register review l.133-135). But that precedent stood on a rate ≥ 0.858, not 0.4.

*Pass probabilities.*
- The draft's 0.67 is again hypergeometric (out l.47). On fresh seeds, ≥ 1 of 2 at 0.4 passes 0.64 per machine and 0.41 on both (out l.52).
- Across S69's Wilson interval [0.17, 0.69] it passes 0.095-0.814 on both machines (out l.50, l.53).
- With D32 at VALID_N = 2 it drops to 0.17-0.41 (out l.75).
- So, at the point estimate, the pooled primary would be UNTESTED more often than not.

*What the referee would require.*
- Pre-register more validity seeds, not one fewer success. ≥ 1 of 4 passes 0.76 on both machines at 0.4; ≥ 1 of 6 passes 0.91 (out l.52).
- Fix n from a lower bound, not the point estimate. At S69's Wilson lower bound, 0.9 on both machines needs 17 seeds per machine (out l.59).
- A fresh 30-40-seed oracle screen at 40,800 would bring that to 10-11, if the rate stays at 0.4 (out l.121-124). At 6.0 h per oracle run (b21 l.12) that costs about 180-240 h of runs before the test. The test then adds 6-11 oracles per machine.
- CHECK 179 must list the n validity seeds (the first n of each block).
- Validity failures are unrelated to the learned outcome (note 3). A failed machine therefore discards data without biasing it, so the expected cost can be stated in advance: P(pooled UNTESTED) is printed beside the rule.

*What the claim can say.* "On each VALID machine the D128 path bound P = 16 with the routing given at least once in <n> seeds within 40,800."

*What it cannot say.* That the D128 oracle is reliable. The readings must not call a machine's UNTESTED a negative (draft l.114-115).

**Option 4. Drop the D128 oracle requirement; K1/K2 conditional on the learned arms alone (draft l.25).**

*What it buys and what it removes.*
- The D128 factor in the pass probability becomes 1. Pooled validity then rests on D32 alone: 0.41-0.82 on both machines for a D32 oracle rate of 0.8-0.95 (out l.78). So the D32 screen (note 2) is still required, or D32's oracle is dropped too and the spec says so.
- The cost is interpretability. Without a D128 positive control:
  - a SHOWN K1 or K2 is self-validating (D128 bound, so the path works);
  - every NOT SHOWN row loses its meaning. "No key-load lead at this budget" (draft l.155) and "the width effect is not shown" (l.154) cannot be told apart from "the D128 path is broken on this machine".

*CHECKs.* The CHECKs must then carry what validity carried, on every machine.
- CHECK 174 is asserted only on a 2.80 or 2.10 GHz Xeon and is printed on L (draft l.179-180). On L, nothing would tie the D128 arm to a recorded run.
- The referee would require a CPU-independent CHECK of the D128 model against the batch-19 builder. That means `explore_b19_model.builder_of`, extracted from claude/outside-ideas: logits, gate and every parameter gradient at one seed, bit for bit or to 1e-6 relative with the difference printed, as register review fix 2 (l.379-390).

*Readings and claims.*
- The readings must be re-enumerated with no D128 UNTESTED row, and every NOT SHOWN row must carry "no positive control at D128".
- The claim cannot be read as evidence about width when NOT SHOWN.
- K1's power is near 1 if D32 stays near 0 (0.924 at D128 0.2 against D32 0.0). It falls to 0.23 at 0.2 against 0.1 and 0.64 at 0.3 against 0.1 (out l.106-111). So a NOT SHOWN K1 is likely at modest rates, and it would be uninterpretable.

*Cheapest repair.* Add option 2's learned positive control (KL_D128 at P = 4, ≥ 1 of 2). Pre-registered now, it would give the NOT SHOWN rows their meaning back.

**Option 5. Drop the D128 arm and the test until a D128 configuration has a reliable oracle (draft l.26).**

*What must be recorded instead of a run.*
- The S67 lead stays "inconclusive, weaker than handed back" (README l.772-776). Its 4/10 against 0/10, p = 0.125 (README l.688), is cited nowhere as a width effect.
- The draft stays labelled NOT READY FOR V (draft l.8).
- Seeds X 660-679 and L 1660-1679 (draft l.121) are either released or recorded as held, so that no other session takes them in the meantime.

*Before the test is reopened.*
- "Reliable" must be defined before the screen that looks for it. S69's band (≥ 9/10, b21 l.1415) has the operating characteristics in out l.84.
- If several configurations are screened (budget, oracle lr, P), selection inflates the pass rate. At a true rate of 0.7, the best of 3 configurations passes ≥ 9/10 with chance 0.384 against 0.149 for one (out l.93).
- So the chosen configuration needs its own confirmation screen on fresh seeds, of at least 10.
- If the configuration changes the recipe the learned arms run, S68's budget rule must be re-run on that recipe. The 40,800 does not carry over (README l.750-751).
- The oracle's recipe must be stated relative to the arms'. A "reliable" oracle on a recipe the arms do not use, such as single Adam against SLOW (out l.16, l.41), validates the model, not the arms' path.

*What it claims.* Nothing about width at P = 16. A D32-only or D128-only rate test is not this test and would need its own spec.

## Summary

| option | must come first | validity pass P, both machines (D128 factor; D32 not included) | claim can say / cannot say |
|---|---|---|---|
| 1 longer oracle budget | fresh-seed D128 oracle screen, ≥ 10 seeds, budget rule fixed first; D32 screen | VALID_N 2: 0.24 at 0.7, 0.66 at 0.9; with ≥ 1 of 2: 0.83 at 0.7 (out l.55-56) | path binds with routing given within B; not within the arms' 40,800 |
| 2 oracle at P = 8 | ≥ 10-seed P = 8 oracle screen at both widths; both widths moved | after 10/10: 0.27 (2 of 2) or 0.85 (≥ 1 of 2) (out l.116) | model and path work; not P = 16 capacity |
| 3 VALID_N = 1 | amendment dated after S69; n fixed from a lower bound (screen of 30-40 seeds) | 1 of 2: 0.41 (0.095-0.81); 1 of 6: 0.91 (out l.50-53) | the path bound at least once in n; not that the oracle is reliable |
| 4 no D128 oracle | CPU-independent D128 model CHECK; D32 screen; re-enumerated readings | 1 (D32: 0.41-0.82, out l.78) | SHOWN is self-validating; NOT SHOWN is uninterpretable unless a learned positive control is added |
| 5 drop the test | reliability rule and confirmation screen fixed before any configuration search | none | nothing about width at P = 16 |
