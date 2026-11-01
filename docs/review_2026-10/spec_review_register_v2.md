# Spec review 2: the register test draft, revision 1 (Session G, `explore_out/G/spec_draft_register.md` at `c98648e6`)

*Referee V, 21 October 2026. Before the main line adopts the draft and before any run.*

**Conventions:**
- "r1 l.NN" is a line of the draft at `c98648e6` (claude/explore-G, 2026-10-21 09:47:17 UTC).
- "r0 l.NN" is a line of the draft at `076005d`.
- "V1 §N / fix N" is a section or fix of `spec_review_register_draft.md`, the first review, on the draft at `076005d`.
- "out §N" is a section of `spec_checks_register_v2_out.txt`, written by `spec_checks_register_v2.py`. That script is stdlib only and reads git and nothing else.
- The diff `076005d → c98648e6` changes only this file (+178 / −67; out §2).

**Verdict: ready for X's adoption after the fixes at the end.** None of them needs a run.

**The tally:**
- **Fixes 1-10:** all are applied as written. Counting fix 7's seven parts separately, that is 16 items: 15 RESOLVED and 1 PARTLY (fix 5: the rate reading has no row for "both machines UNTESTED", a gap that was in my own replacement text).
- **The first review's other findings:** 13 items. 12 are RESOLVED. One is NOT: the term that reads less (V1 §11.2). The main line's plan later asked for it as an arm, and revision 1 does not have it.

**What is still wrong, in the order the fixes take:**
1. **The plan's arm is missing.**
   - The main line's plan (`plan_2026-10-13.md:48-50`, 2026-10-13) asks for S87's frequency hinge as an arm. So does STATUS.md (:87, "S87's frequency hinge belongs in it as an arm").
   - Each of G's last two handbacks names a label-free hinge for this draft: the README's S87 section asks for FREQHINGE, and its S89 section asks for the repeat hinge.
   - Revision 1 has neither arm and says nothing about it.
   - Without one, the test cannot answer the question the main line gives it, "a distant cue discovered label-free" (STATUS.md:84-85). Its only label-free arms are its comparators.
   - Whatever is decided now is decided after S89's records (2026-10-20 22:16 to 2026-10-21 09:46 UTC). The draft has to date the decision.
2. **The task-structure caveat** covers the key term only. If a label-free hinge is added, the caveat must say in counts how far that hinge coincides with the keys on lite. On lite it contains the key term's positions and grouping exactly (§3.3).
3. **The S89 background bullet** (r1 l.53-56) leaves out the frequency hinge's 7/20 on POOL.
4. **RESGATE_KEYHINGE** (fix 10, adopted) has four gaps:
   - no CHECK that it differs from LOCAL3_KEYHINGE "by the residual input only";
   - no named reference construction (there is no Hebbian reg = False record anywhere);
   - C4's power is not stated, and it depends entirely on an unscreened rate;
   - the comparator's name in the readings does not say it has the term.
5. **Small edits:**
   - the test's file name differs from the plan's;
   - "186 records" is the Hebbian subset;
   - G's re-score is cited but not committed;
   - the task file's blob is not pinned, now that G's branch carries F's longer module;
   - the both-UNTESTED rate row is missing;
   - the lean CHECK's sample is not named;
   - one printed name clashes with another;
   - the change list omits the background edits.

## 1. Fixes 1-10, item by item

| fix | asked (V1) | revision 1 (file:line, quoted) | state |
|---|---|---|---|
| 1 | name the outcome function, copied from `explore_g_rhl_parent.py` at `076005d`; name the probe; CHECK it against G's recorded counts | r1 l.112-113: "ROUTED-BOUND(r) = `test_register_gate.routed_bound(r)`, copied from `explore_g_rhl_parent.py` at `076005d` (`bound` :39, `routed_star` :43, `routed_bound` :52, thresholds :35)"; l.120-121: "the probe is run_one's statistics probe `probe_batch(task, seed)` (test_multilayer_binding.py:352 ... seed + 99,000, PROBE_B sequences)"; l.217-222: the CHECK with all eleven counts | **RESOLVED** (see notes) |
| 2 | a CPU-independent model CHECK against G's classes, on every machine | r1 l.234-244: "**The model against Session G's code, CPU-independent** (asserted on every machine) ... against ... `Reg3BDH` and ... `KeyHingeBDH`; LOCAL3_KEYHINGE's against `explore_g_s83_child.TermLocal`; RESGATE_KEYHINGE's against `Reg3BDH` with reg = False plus `KeyHingeBDH`'s term ... Tolerance: bit for bit, or ... 1e-6 relative" | **RESOLVED** (the new arm's reference is under fix 10) |
| 3 | VALID: CEIL 2/2 and REG3_NUDGE ≥ 1 of 2 | r1 l.24: "VALID: CEIL binds 2/2 and REG3_NUDGE is ROUTED-BOUND on at least 1 of 2"; l.132-138 (the replacement text verbatim) | **RESOLVED** |
| 4 | readings: "routes and binds", no "discover", one name per combination, a caveat line on every reading, REG3 described as register plus residual input, S83's account labelled | r1 l.170-186: five names over C1-C4; l.183-184: caveat line "...REG3 differs from the window gate by the register and the residual input together; C4 isolates the register."; l.186: "'Discovers' is not used"; l.98: "(Session G's reading of S83's traces)"; title l.10: "REG3 with the key term routes and binds a distant-cue stream" | **RESOLVED** (16 of 16 combinations have exactly one name, out §6; wording note under fix 10) |
| 5 | R: MET / NOT MET / UNTESTED per machine, Wilson and Clopper-Pearson, the pooled count, Fisher printed, the operating characteristics fixed, a rate reading for every case | r1 l.157-163 (verbatim); l.188-195: "R MET on both machines" ... "R UNTESTED on <machine>, with the other machine's verdict printed" | **PARTLY**: 8 of 9 cases are named. The case of both machines UNTESTED has no defined text: "the other machine's verdict" is itself UNTESTED (out §6). The gap was in my replacement text. |
| 6 | multiplicity paragraph; Holm printed; registry counts | r1 l.165-168: "C1-C4 are a family of four one-sided tests at 0.05 each, with no correction ... Holm-adjusted p-values over C1-C4 are printed ... 4 pooled tests, 8 per-machine tests and one rate rule per machine" | **RESOLVED** (extended to C4) |
| 7(a) | β^R in (0.09, 0.11) | r1 l.227-228 | **RESOLVED** |
| 7(b) | the stream lean defined | r1 l.259-260: "the stream lean \|mean p₀ at stream-0 positions − mean p₀ at stream-1 positions\| on the probe is > 0.02 at every layer (`explore_g_rhl_child.py:514`)" | **RESOLVED**. Note: G's CHECK computes the lean on 256 sequences of its own (`_x(256, 9)`, `explore_g_rhl_child.py:509-514`), not on run_one's probe. Either sample can fail the CHECK; the spec should name which (fix 11). |
| 7(c) | reproduction widened; references vendored in `results/refs/` with commit, path and key | r1 l.254-258 | **RESOLVED**. Every reference exists: S82 BASELINE, REG3, REG3_KEYHINGE and REG3_NUDGE at 2100, ORACLE_ALL\|2100 (recipe `adam`, i.e. CEIL's single rate) and S83 LOCAL3_KEYHINGE\|2100 (out §3). |
| 7(d) | a seeds CHECK | r1 l.263-265 | **RESOLVED** |
| 7(e) | a pairing CHECK | r1 l.266-268 (with RESGATE_KEYHINGE added) | **RESOLVED** |
| 7(f) | worker and resume CHECKs | r1 l.269-270 | **RESOLVED** |
| 7(g) | per-layer ROUTED* agreement and fail_class v2 vs v1 in the report | r1 l.205-206 | **RESOLVED** |
| 8 | the referee's seed re-check recorded | r1 l.143-145 (verbatim) | **RESOLVED** |
| 9 | "BOUND ROUTED with a margin guard" renamed | r1 l.127-128: "ROUTED AT EVERY LAYER: bound, the VAL stream → channel map one-to-one at every layer, the routing margin ≥ 0.9 at every layer, and every stream's accuracy ≥ 0.9 (printed, not a claim)" | **RESOLVED**. The name now sits next to l.205's "ROUTED* at every layer", a different statistic (fix 12). |
| 10 | optional: RESGATE_KEYHINGE (reg = False + key term) and C4 | r1 l.22, l.81-83, l.102-104, l.156, l.178-184, l.202, l.225-226, l.238, l.252-253, l.266, l.277-278 | **RESOLVED** (adopted). Four new gaps follow (§2.2). |

**Notes on fix 1.**
- **The function.** `explore_g_rhl_parent.py` is the same blob (`0e1b03ad69`) at `076005d`, `c98648e6` and the branch head. Its line numbers are exactly the draft's: :35, :39, :43, :52. Out §2 quotes the function.
- **Re-scored here** (out §3), `routed_bound` gives every count in the CHECK (r1 l.218-220): 11 of 11 MATCH. The recorded transition equals `test_binding_onset.transition(curve)` on all 218 records in the two files.
- **Not yet in any file.** `test_register_gate.py` (and `register_gate.py`, `test_register.py`, `specs/test_register*.md`) exists on none of the 13 remote heads (out §2). That is expected before adoption. The definition being copied, though, exists only on claude/explore-G (and in my own check scripts).
- **Three residues:**
  - The plan names the files `specs/test_register.md` and `test_register.py` (`plan_2026-10-13.md:50`, STATUS.md:88). The draft names `test_register_gate.py` (r1 l.1, 10, 25, 112). The named function's module depends on which name X uses.
  - The two files hold 218 records: 186 Hebbian and 32 delta (`_D`). The CHECK says "On every record" (r1 l.217) and "on all 186 records" (l.221). The counts it lists are the Hebbian arms', and the transition equality holds on all 218.
  - "(Session G re-scored these counts with its own `routed_bound` on 2026-10-21: all equal.)" (r1 l.221-222) is not in any commit: `c98648e6` changes only the draft (out §2). My re-score agrees with it, so the sentence is true. It should cite something that can be read.

## 2. The first review's other findings, and what is new

### 2.1 The other findings of the first review

| V1 | finding | revision 1 | state |
|---|---|---|---|
| §1 | the runtime understated REG3's unbound runs (8.5-8.7 min) | r1 l.276-279: "(the referee's medians): REG3_KEYHINGE 2.2 min (2.1-3.0); BASELINE 6.3 ...; REG3 unbound 8.5-8.7 ... About 8-10 CPU-hours per machine" (it adds up: 7.2 h for the four arms, plus 0.7-2.9 h for RESGATE_KEYHINGE) | **RESOLVED** |
| §6 | 18/20 supports a rate of at least 0.70, not "reliable" | r1 l.157: "(the RELIABLE band at n = 20)", intervals printed, readings say "R MET" | **RESOLVED** |
| §6 | "reliable on one machine" is mostly noise | r1 l.161 Fisher "printed and not read"; l.195 "None of these is read as a difference between the machines" | **RESOLVED** |
| §6 | the pooled rule was not monotone | r1 l.160: the pooled count printed beside it | **RESOLVED** |
| §8(a) | "discovers" | gone (r1 l.186) | **RESOLVED** |
| §8(b) | reading 2 credits the term when C3 fails | r1 l.176-177: "...not shown more often than REG3 without the term" | **RESOLVED** |
| §8(c) | "the register" for C2 | r1 l.174-175 says "the key term on the window gate"; the caveat names both inputs | **RESOLVED** |
| §8(d) | "not replicated" misdescribed one case | r1 l.172-173: "not shown to route and bind more often than the baseline", with both counts | **RESOLVED** |
| §8(e) | "released by" names a mechanism | gone | **RESOLVED** |
| §9 | causality: "layer 1's at t + 1" was weaker than G's CHECK | r1 l.229-230: "changes every layer's at t + 1" | **RESOLVED** |
| §5 | a reading quoted from one claim is per-comparison (report 6's "the register is needed") | r0 l.40-41's quoted readings are removed (r1 l.46-47) | **RESOLVED** |
| §11.4 | delta memory not in the test | r1 l.55-56: "These are delta results on other tasks; the claims below stay Hebbian on lite." | **RESOLVED** (stated) |
| §11.2 | "whether a term that reads less would do ... any replacement would need its own test" | background only (r1 l.50-56). The main line's plan of the next day asked for S87's frequency hinge as an arm (`plan_2026-10-13.md:48-50`: "its own claim against BASELINE; a pre-fixed one-pair comparison against REG3_KEYHINGE; the frequency cut's coincidence with the key positions in the caveat"); STATUS.md:87 says the same | **NOT** (see §3) |

### 2.2 What revision 1 adds that was not asked for

| new in r1 | lines | test | finding |
|---|---|---|---|
| S87 background | l.50-52 | against `s87_results.json` at `8e9e3187` (out §3) | **MATCH**: FREQ 17/20, KH 20/20, 3 vs 0 (p 0.125), FREQ_D 10/10. "it reads about what the key term reads, plus the stream tokens": it also reads the query's key (14 positions = 8 + 5 CTX + 1; report 7:51-53). |
| S89 background | l.53-56 | against `s89_results.json` (out §3) | Counts **MATCH**: POOL REG3_D 16/20, KH 20/20, REPEAT 20/20; ALLPOOL every learned arm 0/20. The task sizes are right (`K_POOL, N_CTX_POOL = 32, 16`, `explore_f_tasks.py:302` at `c98648e6`), and so are the seeds (2842-2861). It **omits** two results: the frequency hinge's POOL result (7/20, "needs frequent keys", KH 13 vs 0, p 1.2e-4) and BASELINE_D 0/20. The omitted one is the S89 result that bears on the arm the plan asked for (fix 3). |
| The S83 bullet's quoted readings removed | l.46-47 | | Consistent with V1 §5. Not listed in the change list (l.283-301). |
| RESGATE_KEYHINGE's definition | l.81-83 | `explore_g_regmodel.to_reg` (:119-139 at `076005d`): reg = False draws A_r first and skips A_R, W_R, w_R, b_R | So "RESGATE_KEYHINGE's [initial parameters] equal REG3's minus the register's" (l.225-226) is true by construction, and the CHECK can fail if the order changes. "Differs ... from LOCAL3_KEYHINGE by the residual input only" (l.82-83) has **no CHECK**. The analogue of l.224 (A_r = A_R = 0 gives LOCAL3) is missing for A_r = 0 (fix 4). The "only" also covers a per-layer gate, which comes with the residual input. |
| RESGATE_KEYHINGE's reference | l.238 | records on every head | **No Hebbian reg = False record exists** (out §3). G's code has `RESGATE_ONLY_D` only on delta (`explore_g_rhl_child.py:102`). Its 10 runs are 0/10 ROUTED-BOUND and 5/10 BOUND (S73). The reference "`Reg3BDH` with reg = False plus `KeyHingeBDH`'s term" can be built: `make_reg3(task, seed, "hebb", reg=False)` (:137) followed by `explore_g_devices_child.make_model`'s class swap (:151-159). But no G entry point does this, so the spec must name the construction (fix 5). |
| C4's power | l.156 | out §5 | RESGATE_KEYHINGE is unscreened, so C4's power is whatever its rate makes it. With REG3_KEYHINGE at 1.0, 40 pairs: 1.000 / 0.984 / 0.737 / 0.371 at RESGATE_KEYHINGE's true rate 0.50 / 0.75 / 0.85 / 0.90. With KH at 0.95: 1.000 / 0.924 / 0.371 / 0.048. Holm's first step over four needs b ≥ 7 with c = 0 (against b ≥ 5 at 0.05). C4 not SHOWN is a weak statement unless the rate is printed beside it (fix 6). |
| The C4 readings' comparator | l.179, l.181 | wording | "the same gate without the register" does not say that the comparator has the key term (fix 7). |
| The task file on G's branch | l.14, l.60 | blobs (out §2) | The draft says "copied byte for byte from claude/explore-F at 9a6c2bf" (blob `de47463`). Since S89, claude/explore-G carries F's `b0c6332` module instead (blob `e24d5e1`, 49,923 bytes). Its first 14,101 bytes are `de47463` and it redefines no name, so lite's code is unchanged. An adopter copying "G's" module would get the larger file, and the "byte for byte" CHECK (r1 l.211-213) would then fail or be weakened. Pin the blob (fix 9). |
| Seeds | l.140-145 | full scan, today's 13 heads (out §1) | **Free.** I read 359 JSON/JSONL blobs (1,081 branch-paths) and 35,196 seed values, with no hit in 600-619 or 1600-1619. The integer keys 600 and 1600 occur only as update steps and tolerances (`by_step`, `eta_at`, `spec`, `1e-06`, ...). G's S87 (2700-2729) and S89 (2840-2861) are disjoint. Every text mention of the block refers to this draft. Seeds are unchanged from r0. |
| Delta memory, the window, the thresholds | | diff | Unchanged. No delta arm was added. |

## 3. The plan's label-free arm, the caveat, and what was chosen after S89

### 3.1 What was asked, and when

| date (UTC) | source | what it says |
|---|---|---|
| 2026-10-12 23:00 | V1 (`723827b`) | §11.2: a term that reads less is left open. No fix asks for it. |
| 2026-10-13 02:25 | G report 7 (`e46799a`) and README:711 | "Handed back: FREQHINGE as a label-free variant for the main-line spec draft, with that caveat" |
| 2026-10-13 20:27 | main line `plan_2026-10-13.md:48-50` (`6912e0b`) | "G revises its spec draft with V's fixes and S87's FREQHINGE as an arm (its own claim against BASELINE; a pre-fixed one-pair comparison against REG3_KEYHINGE; the frequency cut's coincidence with the key positions in the caveat)" |
| 2026-10-19 / 20 | round-6 prompt (`parallel_sessions_prompts.md`, Session G, round 6) | "Then, and only then, the register spec's revision per the referee's review, as a separate commit." (no word on the arm) |
| 2026-10-20 21:53 | STATUS.md:84-89 (main line) | "the project's open question, a distant cue discovered label-free ... S87's frequency hinge belongs in it as an arm" |
| 2026-10-20 22:16 → 2026-10-21 09:46 | S89's records (`0bd95b2` … `8e9e318`, 88 commits); report 8 at 09:46:54 | POOL (delta): FREQ 7/20 "needs frequent keys"; REPEAT 20/20 "replaces the key hinge"; README: "Handed back: the repeat hinge as the label-free candidate" |
| 2026-10-21 09:47:17 | revision 1 (`c98648e6`), 23 s after report 8 | neither hinge; RESGATE_KEYHINGE and C4 (asked 12 October, before S87 and S89) |

**What was chosen after S89.** Nothing in revision 1's arms, seeds, thresholds or claims was chosen on S89's numbers:
- RESGATE_KEYHINGE, C4, the validity rule and the rate rule were all asked for on 12 October.
- The S89 text is background, and selective (§2.2).

**What is undecided, and is now post-S89 whichever way it goes:**
- Leaving FREQHINGE out departs from a plan made before S89. If that is deliberate, it is a choice made after seeing FREQ fail on POOL.
- Adding REPEATHINGE would be a choice made on S89's POOL records.

Either is admissible for a confirmatory test on fresh seeds. The spec must say which was made, when, and on what (fix 1).

### 3.2 The arms the plan names, measured

All figures are exact one-sided McNemar at the screens' paired discordance (out §3, §5), with 40 pooled and 20 per-machine pairs.

| claim | screen evidence (pairs) | P(SHOWN), 40 pairs | at Holm's 0.0125 | 20 pairs | P("within 1 pair"), 40 / 20 |
|---|---|---|---|---|---|
| C1 KH vs BASELINE | S82 20 vs 0; S87 20 vs 0 | 1.000 | 1.000 | 1.000 | — |
| C2 KH vs LOCAL3_KEYHINGE | S83 20 vs 0 | 1.000 | 1.000 | 1.000 | — |
| C3 KH vs REG3 | S82 18 vs 0 (S89 POOL delta: 4 vs 0) | 1.000 (0.924) | 1.000 (0.714) | 1.000 (0.370) | — |
| C4 KH vs RESGATE_KEYHINGE | none on Hebbian; S73 delta RESGATE_ONLY_D 0/10 without the term | see §2.2 | | | — |
| F1 FREQ vs BASELINE | S87 17 vs 0 | 1.000 | 1.000 | 1.000 | — |
| F2 KH vs FREQ | S87 3 vs 0 (S89 POOL delta: 13 vs 0) | 0.737 | 0.393 | 0.170 | 0.012 / 0.176 |
| RP KH vs REPEAT | S89 POOL delta 0 vs 0; never run on lite or on Hebbian memory | no lite evidence | | | |

**The plan's "pre-fixed one-pair comparison" does not scale to 40 pairs.** With KH at 1.0, P(KH − FREQ ≤ 1 pooled) at FREQ's true rate is:

| FREQ's true rate | pooled (40 pairs) | per machine (20 pairs) |
|---|---|---|
| 0.85 | 0.012 | 0.176 |
| 0.90 | 0.080 | 0.392 |
| 0.95 | 0.399 | 0.736 |
| 0.975 | 0.736 | 0.912 |

So on 40 pooled pairs it would fail even for a hinge that binds 95% of the time. If it is kept, apply it per machine, as S87 did on 20, and print its operating characteristics. The test of the difference is F2, a McNemar claim: 0.74 power at S87's effect.

### 3.3 The task-structure caveat on lite, in counts (out §7)

The key term reads key positions and identities. A frequency or repeat hinge reads token statistics: a corpus table, or the within-sequence counts. On lite the input positions are 0-21: 5 CTX (four block heads and the query's), 9 KEY (8 body and the query's) and 8 VAL.

| hinge | positions per sequence | grouping | contains the key term's 8? |
|---|---|---|---|
| key term | 8 (body KEY) | key id | — |
| frequency (S87: the largest-ratio cut, types {0..5}) | 14 = 8 KEY + the query's KEY + 5 CTX | token id: the key id at KEY, the stream at CTX | yes, 8 of 8, same grouping (S87's CHECK) |
| repeat (S89's `repeat_mask`) | 14 (every CTX and KEY type repeats on lite) + the body VALs whose value recurs: P = 1 − (7/8)³ = 0.330 each, mean 2.64; none in 15.7% of sequences (simulation of `make_batch:69`'s draw, 200,000 sequences) | token id | yes, 8 of 8, same grouping |

**So on the spec's task, a label-free hinge reads a superset of what the key term reads.**
- It is given no labels, but where the key term's positions are, it reads the same positions and grouping.
- It adds the stream itself at the context tokens. That is the part S89 found harmful when it is all the cut selects (POOL: 7/20).

A result for either hinge on lite therefore cannot separate "a token statistic suffices" from "the key positions suffice". Its caveat has to say so in these counts. Tasks where the two differ (POOL, ALLPOOL) were screened only on delta memory, and there the repeat hinge's lead over REG3 alone is 4 pairs.

### 3.4 The readings are exhaustive; the CHECKs can fail

- **C1-C4.** Each of the 16 combinations gets exactly one name (out §6).
- **R.** 8 of 9 machine states are named; (UNTESTED, UNTESTED) is not (fix 10).
- **The CHECKs.** Every CHECK revision 1 adds can fail before a run, except the parenthesis at l.221-222, which is a statement about an uncommitted re-score and not a CHECK.
- **Missing:** the A_r = 0 equivalence for RESGATE_KEYHINGE (fix 4).

### 3.5 The rate claim's operating characteristics (out §4)

Exact binomial, equal true rates on the two machines, machines independent:

| true rate | MET on a machine | on both | on exactly one | on neither | pooled ≥ 36/40 |
|---|---|---|---|---|---|
| 0.80 | 0.206 | 0.042 | 0.327 | 0.630 | 0.076 |
| 0.85 | 0.405 | 0.164 | 0.482 | 0.354 | 0.263 |
| 0.90 | 0.677 | 0.458 | 0.437 | 0.104 | 0.629 |
| 0.95 | 0.925 | 0.855 | 0.140 | 0.006 | 0.952 |

The "both" column is r1 l.162-163's "0.04, 0.16, 0.46 and 0.85" (correct). The intervals at 18/20 are right as printed: Wilson [0.699, 0.972], Clopper-Pearson [0.683, 0.988].

## Remaining fixes, in order (replacement text; the first-named first)

1. **The plan's label-free arm** (after r1 l.22, l.156-163 and l.170-195). Choose one of three. The choice belongs to the user and X; the referee needs only that it is written and dated.

   **(A), the plan's.** Add the arm:

   > REG3_FREQHINGE  REG3 + Session G's S87 frequency hinge for updates 1-2400: the key term's λ, τ and injection at the positions of the frequent token types, grouped by token id. The types are `explore_g_s87_child.freq_types` at `e46799a`: the largest-ratio cut of the token-type frequencies of the task's own sample, {0, …, 5} on lite. It receives token ids only.

   Add the claims, as their own family, after R:

   > - **F1** REG3_FREQHINGE beats BASELINE (ROUTED-BOUND).
   > - **F2** REG3_KEYHINGE beats REG3_FREQHINGE (ROUTED-BOUND).
   > - **F3** (pre-fixed, printed with its operating characteristics; not a test) "within one pair" per machine: REG3_KEYHINGE only − REG3_FREQHINGE only ≤ 1 on that machine's 20 pairs. At REG3_KEYHINGE 1.0, P(within one pair on a machine) is 0.18, 0.39, 0.74 and 0.91 at REG3_FREQHINGE's true rate 0.85, 0.90, 0.95 and 0.975. It is not applied pooled, where it would hold with 0.01-0.74.
   >
   > F1-F2 are a family of two at 0.05 with Holm printed; the registry gains 2 pooled and 4 per-machine tests. Printed with them, fixed now: at S87's effect (3 vs 0 of 20), P(F2 SHOWN) is 0.74 pooled and 0.17 per machine.
   >
   > **Readings** (pooled; one name per combination of F1 and F2; each printed with the caveat line):
   > - F1 not SHOWN: "the frequency hinge not shown to route and bind more often than the baseline".
   > - F1 and F2 SHOWN: "the frequency hinge routes and binds more often than the baseline; the key term routes and binds more often than the frequency hinge".
   > - F1 SHOWN, F2 not SHOWN: "the frequency hinge routes and binds more often than the baseline; the key term not shown to route and bind more often than it", with F3 printed per machine.

   Add the CHECKs:

   > - `freq_types` receives a LongTensor of token ids and raises on a float tensor.
   > - Its cut on lite's sample is {0, …, 5}.
   > - The coincidence counts are 14 positions, 8 key-term positions, and 8 of 8 among them; all three are printed.
   > - At λ = 0 the hinge equals REG3 bit for bit through 2400.
   > - S87's REG3_FREQHINGE|2700 reproduces through 2400 from a vendored copy, asserted on a matching CPU.
   > - The model CHECK of r1 l.234 covers `explore_g_s87_child`'s term.

   Runtime: add 20 runs per machine, at S87's 2.0 min median (1.9-7.7).

   **(A′), optional, beside (A).** Add REG3_REPEATHINGE (`explore_g_s89_child.repeat_mask` at `8e9e3187`; grouped by token id) with the same three claims against it. State: "chosen on 2026-10-21 after S89's POOL records (delta, another task); never run on lite or on Hebbian memory".

   **(B), none.** Insert after r1 l.31:

   > No label-free hinge is an arm. The test does not answer whether a distant cue is discovered label-free: its label-free arms are BASELINE and REG3, its comparators. Decided <date>, after S89's records (2026-10-20 22:16 to 2026-10-21 09:46 UTC). The main line's plan of 2026-10-13 (`plan_2026-10-13.md:48-50`) had asked for S87's frequency hinge as an arm.

   **In every case,** add to the header (r1 l.3-6):

   > Revision 1 was written after S89's records (2026-10-20 22:16 to 2026-10-21 09:46 UTC). Its arms, seeds, thresholds and claims were fixed by the referee's review of 2026-10-12. Only the background's S87 and S89 bullets, and <the choice above>, came after.

2. **The caveat** (after r1 l.99), with (A) or (A′):

   > On lite the frequency cut selects the 2 context and 4 key types: 14 positions per sequence, the key term's 8 among them, plus the query's key and the 5 context tokens. Grouping by token id there is grouping by key id at the keys and by stream at the context tokens. The repeat hinge selects the same 14 plus the body values that recur (2.64 per sequence on average; none in 16% of sequences). Neither is given a label, but on this task each reads the key term's positions and grouping exactly. A result for either cannot separate a token statistic from the key positions. Session G's S89 separated them only on delta memory and other tasks.

   Add to the caveat line (r1 l.183-184): "The frequency (repeat) hinge reads no labels; on this task it reads the key term's positions and identities, plus the context tokens."

3. **The S89 bullet** (r1 l.53-56) becomes:

   > S89 (2842-2861, delta memory, F's round-6 pool tasks; report 8). On RandHeaderTask-pool (keys from 32 types):
   > - BASELINE 0/20;
   > - REG3 alone 16/20 (descriptive in S89);
   > - the key term 20/20;
   > - the frequency hinge 7/20 (its cut selects only the 2 CTX types there; "needs frequent keys", 13 vs 0 against the key term);
   > - a label-free repeat hinge (the term at every token type that repeats in the sequence) 20/20.
   >
   > On -allpool (contexts from 16 types) every learned arm, the key term included, is 0/20 (the perfect gate 2/2). These are delta results on other tasks; the claims below stay Hebbian on lite.

4. **RESGATE_KEYHINGE: the residual input only** (new CHECK after r1 l.226):

   > With A_r = 0, RESGATE_KEYHINGE's gate at every layer equals LOCAL3_KEYHINGE's window gate, and its logits and its key term equal LOCAL3_KEYHINGE's, bit for bit, on one forward of 16 sequences.

   In r1 l.82-83, "by the residual input only" becomes "by the residual input only (with it, a gate that differs by layer; the CHECK above asserts the rest)".

5. **RESGATE_KEYHINGE's reference** (r1 l.238). "RESGATE_KEYHINGE's against `Reg3BDH` with reg = False plus `KeyHingeBDH`'s term" becomes:

   > RESGATE_KEYHINGE's against `explore_g_rhl_child.make_reg3(task, seed, "hebb", reg=False)` (:137), given `KeyHingeBDH`'s class and term state as `explore_g_devices_child.make_model` does (:151-159).

   Add: "No Hebbian record of this construction exists. S73's RESGATE_ONLY_D is delta and has no term: 0/10 ROUTED-BOUND, 5/10 BOUND."

6. **C4's power, printed with C4** (after r1 l.156):

   > RESGATE_KEYHINGE is unscreened. At REG3_KEYHINGE 1.0, P(C4 SHOWN, 40 pairs) is 1.00, 0.98, 0.74 and 0.37 at RESGATE_KEYHINGE's true rate 0.50, 0.75, 0.85 and 0.90. C4 not SHOWN is printed with both arms' pooled counts.

7. **The C4 comparator's name** (r1 l.179 and l.181): "than the same gate without the register" becomes "than the same gate and term without the register".

8. **The outcome** (r1 l.1, 10, 25, 112, 217, 221-222):
   - **The name.** Use the plan's file name, "`test_register.routed_bound`" in `test_register.py` (`plan_2026-10-13.md:50`; STATUS.md:88). Or keep `test_register_gate` and have X record the name at adoption. Either way the function is "copied from `explore_g_rhl_parent.py:35-53` at `076005d` (blob `0e1b03a`)".
   - **The CHECK's scope.** r1 l.217 "On every record of" becomes "On the 186 Hebbian records of". r1 l.221 "on all 186 records" becomes "on all 218 records (186 Hebbian, 32 delta)".
   - **The re-score.** r1 l.221-222 becomes "(Re-scored by the referee on 2026-10-21, `docs/review_2026-10/spec_checks_register_v2_out.txt` §3: all equal.)"

9. **The task file** (r1 l.14 and l.60): after "copied byte for byte from claude/explore-F at 9a6c2bf", add:

    > (blob `de47463`. claude/explore-G now carries F's `b0c6332` module, blob `e24d5e1`, whose first 14,101 bytes are `de47463` and which redefines no name. Vendor `de47463`, or `e24d5e1` with a CHECK that its first 14,101 bytes are `de47463`.)

10. **The rate reading** (after r1 l.193): add "- 'R UNTESTED on both machines'."

11. **The lean's sample** (r1 l.259-260): "on the probe" becomes "on run_one's statistics probe (Session G's CHECK used 256 sequences of its own, `explore_g_rhl_child.py:509-514`)".

12. **The name clash** (r1 l.205): "The count of runs where ROUTED* on the returned gate differs from ROUTED* at every layer" becomes "The count of runs where ROUTED* (margin ≥ 0.9 and η² by stream > 0.9) on the returned gate differs from ROUTED* on each of layers 1-3 separately, listed", so that it is not read as ROUTED AT EVERY LAYER (l.127).

13. **The change list** (after r1 l.301):

    > 11. Background: S87 and S89 added (written after S89's records, 2026-10-21); report 6's quoted readings removed from the S83 bullet.
    >
    > Seeds re-checked by the referee on 2026-10-21: 13 heads, 359 JSON/JSONL blobs, 35,196 seed values, no hit (`spec_checks_register_v2_out.txt` §1).

**Verdict for X's adoption: ready after these fixes.**
- Fix 1 is a decision, not an edit. Without (A) or (B) in writing, X would adopt a test that the main line's STATUS.md describes as answering a question the test cannot answer.
- Fixes 2-13 are text.
- Nothing needs a run, and nothing in fixes 1-10 of the first review needs re-doing.

**Corrections (22 October, `spec_review_register_v3.md`).**
- **The rate-claim names don't give every state exactly one name.** This review's fix 10 text, read literally, gives 5 of the 9 machine states two names. They are exactly-one only if UNTESTED takes precedence, and that should have been stated.
- **Fix 4's CHECK clause can't do its job.** "its key term equal … bit for bit" either passes trivially (the term is 0 at initialisation) or fails spuriously (float32 means). It should compare each layer's η²_key bit for bit, and the term to 1e-6 relative, with τ lowered so the term is active.
