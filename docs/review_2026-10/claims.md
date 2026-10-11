# Claims: every result-stating sentence of the abstract, Sections 5, 6, 11–13 and the conclusion

*Referee's deliverable 3. Verdicts: **SUPPORTED** (by which record), **OVERSTATED** (how), **UNSUPPORTED**, **UNVERIFIABLE** (what would be needed). "rec" = recompute.md row, recomputed from the JSON records; "stats" = statistics.md; "lit" = literature.md; "prereg" = preregistration.md. Quotes are abridged at "…"; tex line numbers refer to `multichannel_hebbian_report_v8.tex`. Records: main line at 4a3eebc; E 5dfc2b5, F a2e55ed, G 514d17f, H 7e9197f (the last commits before the report's commit at 2026-10-09 19:55 UTC).*

## Tally

116 result-stating sentences or table rows, each counted once under its most severe verdict:

| verdict | count |
|---|---|
| SUPPORTED (one of them only as interpretation) | 77 |
| OVERSTATED (one of them "partial") | 20 |
| UNSUPPORTED (false when committed, or not supported by the evidence cited) | 7 |
| UNVERIFIABLE | 10 |
| stale (true when drafted, overtaken by records on the branches before the report was committed) | 2 |

The seven unsupported: "no single-channel run has yet been made" at eight streams, stated three times (A24, 11.19, 12.12); "the optimizer reset ... adds nothing" from looks that never remove the reset (A12); Revision 7's Muon result as "a seed effect" (R4); routing for HINGE4k16 "recorded only on X" (12.3); and "every grouped-layout configuration tried binds" given the partition, which the report's own Section 1 contradicts (12.4).

## Abstract (tex 35, 37)

| # | sentence | verdict | evidence |
|---|---|---|---|
| A1 | "no learned-gate run trained on all eight from the start had bound" (Revision 7) | SUPPORTED | results/X/slow_start HINGE_D8 0/10; Rev. 7 Table 8; batch 13–16 recurrent arms 0/10 (README_outside-ideas batch 15–16) |
| A2 | "screens traced the failure to the gate's recurrent state, which stops carrying the context before any routing forms" | SUPPORTED (exploratory, 5–10 seeds) | S47 decodability of h for X's HINGE_D8: 0.47 at VAL at update 0, 0.14 by 2400 (write_decoders store; README batch 16 table) |
| A3 | the window gate "was screened in the first exploratory batch, beat the recurrent gate there, and was dropped" | OVERSTATED | 15/20 vs 7/20, 12 vs 4, two-sided p = 0.077: "promising" by the screen's rule, not a win; the screen's own verdict word was "promising" (rec, S4 history) |
| A4 | "Combined with slow memory … and a plateau-triggered channel split … it binds eight streams with one channel per stream in 38/40 runs of a pre-registered test" | SUPPORTED; wording OVERSTATED | 38/40 recomputed (rec T2). "One channel per stream" is an argmax: in 18 of 38 runs some stream's largest channel holds < 0.9 of its gate mass; streams occupy disjoint channel *sets* (overlap ≤ 0.047) (rec, sensitivity) |
| A5 | "against 0/40 for the recurrent-gate recipe (38 vs 0, p = 3.6e-12) and 15/40 for the window gate without the split (23 vs 0, p = 1.2e-7)" | SUPPORTED | rec T2 caption rows |
| A6 | "The two machines ran disjoint seeds, so the pooled test is a test." | SUPPORTED | exact stratified McNemar is valid (stats §2); seeds disjoint by meta (X 500–559, L 1500–1559) |
| A7 | "The recipe uses no labels, no restarts and no hinge" | OVERSTATED by omission | no stream labels; but the split reads the task's key positions (tex 118, 166). Section 5 says so; the abstract does not |
| A8 | "its reading, fixed before the run, was 'eight streams bind without labels or restarts'" | SUPPORTED with a qualification | fixed by the amendment `d05f0ca` before any Part D run, 78 min after X's first start of the test (prereg §1.1) |
| A9 | "At four streams … 38/40 against 27/40 … (13 vs 2, p = 0.004)" | SUPPORTED; fragile | rec; not shown on X alone (5 vs 1, p = 0.11); Bonferroni over the project's ~60 claims gives 0.22 (stats §4) |
| A10 | "at two streams it is within the pre-registered bound of the previous recipe, and better by two pairs" | OVERSTATED | the rule holds (d = −2), but the paired difference +0.025 has a 95% interval [−0.079, +0.121]; "better by two pairs" is within noise (stats §3) |
| A11 | "Three exploratory looks … agree that the split's row copy is its active part" | SUPPORTED (exploratory) | split vs reset only: S50 8 vs 1, S58b 8 vs 0; the main test's control seeds 7 vs 0 (rec T3; stats §6b) |
| A12 | "… and that the optimizer reset bundled with it adds nothing" | UNSUPPORTED by the cited looks | none of the three has a copy-without-reset arm. The only one that does, H's S61 (COPY 9/10 = SPLIT 9/10, 10 seeds), is uncited (rec, S61 note; stats §6b) |
| A13 | "Gumbel noise on the read gate collapses routing" | OVERSTATED | the arm put noise on read *and* write gates (GUMBEL_RW, 5/40, 33 merged); there is no read-only arm. Attributing the collapse to the read side is an inference from RW vs W (37/40) |
| A14 | "and on the write gate worsens eight-stream merges" | OVERSTATED | S58b has no no-intervention arm at eight streams: GUMBEL_W 0/10 with up to six sharing is worse than the split (10 vs 0); "worsens" against no device is a cross-seed comparison |
| A15 | "a load-balancing loss changed the shape of the routing at four streams without changing its count" | SUPPORTED | SWITCH 36/40 vs NONE 38/40; key map ≠ value map in 34 of 36; margin 0.87 → 0.46 (rec S6 H) |
| A16 | "the recipe's constants survive a fourfold change of width and a doubling of depth without retuning, on ten seeds per size" | OVERSTATED | counts reproduce (rec S10) but the rule (d ≥ −1, 10 seeds) reads "scale-free" 39% of the time when the larger model's rate is 0.7 against 0.9 (stats §3); every d = +1 rests on seed 273; "depth" is applications of one shared layer, and six layers were slower |
| A17 | "Muon is not an ingredient: the recipe binds eight streams at 9/10 under either optimizer" | SUPPORTED (exploratory) | S48 W_SPLIT 9/10, S50 W_SPLIT_M 9/10, 1 vs 1 (rec S7) |
| A18 | "a pooled test of Muon's four-stream speed, shown on one machine in Revision 7, was not shown" | SUPPORTED | M2 12 vs 8, p = 0.25 (rec S7) |
| A19 | "None of twelve papers reviewed carries an inferred context cue across distance without labels" | OVERSTATED (contradicted by the project's own note) | HM-RNN infers boundaries from the likelihood alone and holds the segment state across distance (`docs/reading/1609.01704.md:144,148`); lit row 15 |
| A20 | "our own distant-cue task remains unsolved" | SUPPORTED | F S54(d) UNTESTED (oracles 0/2, 0/2); G S55: cue absent from the trained window stack (G report_1, `6cd36ff`) |
| A21 | "a single memory channel with a delta-rule write binds the two-stream header layout" | SUPPORTED (exploratory) | S53 8/10 vs Hebbian 1/10; S59 9/10, 8/10 at P = 8 with learned decay (rec S11 F) |
| A22 | "by routes that need no partition (context-tagged keys at a fixed decay; a learned forget gate that partly clears at the context token)" | OVERSTATED | both mechanisms are post-hoc diagnostics; the keys from one rerun (seed 307, F report_1), the decays from F's S59 tables; neither was a pre-registered measurement |
| A23 | "so the partition's claim is about the number of contexts" | SUPPORTED as inference — by a result the report omits | F's S68 (on the branch before the report): one delta channel 2/10 at four streams, 0/10 at eight |
| A24 | "at eight streams the gated recipe binds 38/40, and no single-channel run has yet been made there" | **UNSUPPORTED** (false when committed) | F's `premise_results.json` held ten S=8 single-delta-channel runs (0 bound) by 18:33 UTC, 82 min before the report's commit; the first two by 15:51 (rec S11 F) |
| A25 | "We correct several of our own statements, among them that the gate 'needs only one token of memory' and that Muon 'binds four streams sooner'" | SUPPORTED for the first; the second correction itself overstated | see 13.1 below |

## Section 5, the main result (tex 136–166)

| # | sentence | verdict | evidence |
|---|---|---|---|
| 5.1 | "The test's specification was committed before the test was written … and amended, before any Part D run, to add the split" | SUPPORTED | `9a769c7` 14:11 < `2f98f06` 19:36; amendment `d05f0ca` 00:48 before `886a668` and any Part D record (prereg §1.1) |
| 5.2 | "Both machines ran it on their own seeds (X 500–559, L 1500–1559)" | SUPPORTED | meta.seeds of both files |
| 5.3 | Parts A–D: configurations, budgets, arms, seed counts | SUPPORTED | meta.sched (24000, 28800, 43200), meta.seeds; arm counts 40/20/20/5/2 per machine |
| 5.4 | "The perfect gate on two seeds per part … bound on every one (Part C at 4800 and 3600–6000)" | SUPPORTED | CEIL_A/B 1200 ×4 each; CEIL_C X 4800, 4800; L 6000, 3600 (rec) |
| 5.5 | "Every CHECK passed on X, including bit-for-bit reproductions of batch 1's window gate, batch 16's LOCAL3_SLOW and batch 17's W_SPLIT (its split at 4800 included)" | UNVERIFIABLE | X's log is not in the repository; the docstring says so (test_window_gate.py:142-143). My reproductions (one run per test on a 2.10 GHz Xeon) match X's 2.80 GHz records bit for bit, which makes the claim plausible (prereg §1.5). Needed: the verification log |
| 5.6 | "L's file carries the metadata the test writes only after its verification passes" | SUPPORTED | in `main()` the `store["meta"]` assignment follows `verify(...)` (test_window_gate.py, main) |
| 5.7 | "Nothing was cut" | SUPPORTED | meta.drop = "" on both; every scheduled record present (231/231) |
| 5.8 | "X ran its 225 learned-gate records in 30.9 h of training (its six perfect-gate records came from an earlier start …), L its 231 records in one start (projected 17.3 h)" | counts and projection SUPPORTED; 30.9 h UNVERIFIABLE | meta.starts, record counts, projected_wall_h 17.30; wall time is not recorded in the JSON |
| 5.9 | Table 2's counts, pairs, p-values, bands and Wilson intervals | SUPPORTED | 39 rows, all MATCH (rec T2) |
| 5.10 | "Per machine every claim is SHOWN or HOLDS on both, except G1 on X" | SUPPORTED | rec |
| 5.11 | "The pre-registered reading … applies." | SUPPORTED | 38/40 ≥ 36, G2 and G4 SHOWN pooled |
| 5.12 | "The recurrent-gate recipe bound none of 40 … (15 collapsed, 17 non-stream, 8 merged)" | SUPPORTED | the three categories partition the 40 runs; collapsed runs are all non-stream (rec diag 164) |
| 5.13 | "The window gate alone bound 15/40, and every one of its 25 failures was a merge of two, three or four streams" | SUPPORTED | shares {2: 17, 3: 7, 4: 1} (rec) |
| 5.14 | "The split raised that to 38/40: the trigger fired 85 splits in 40 runs (none to three per run), every target labelled on target" | SUPPORTED | 85, 0–3, 85/85 (rec) |
| 5.15 | "and brought the median time to bind forward (X 21,600 against 27,600; L 18,600 against 32,400)" | SUPPORTED as description | medians over bound runs; different sets of runs (38 vs 15), so not a paired comparison |
| 5.16 | "The two failures (X's seeds 555 and 556) ended with two streams sharing a channel" | SUPPORTED | rec |
| 5.17 | "The reset-only control matched the window gate without the split seed for seed on L (3/5 and 3/5, 0 vs 0) and bound 0/5 against 1/5 on X: on these seeds the row copy, not the Adam reset, made the difference" | SUPPORTED, and understated | on the same ten seeds the split arm bound 10/10: split vs reset 7 vs 0, one-sided p = 0.0078 — the most direct same-test comparison, left out of the report's table (rec T3 row 182) |
| 5.18 | WIN3_SLOW16's two failures; WIN16_A's thirteen (4 position, 3 merges, 4 unrouted binds, 2 other) | SUPPORTED | rec |
| 5.19 | "At two streams no arm failed by position; WIN3's failures were key splits (12 of 15), WIN3_SLOW's … (6 and 5 of 11), HINGE0's … (10 of 13), with two OTHER and one partial" | SUPPORTED | rec |
| 5.20 | "Slow memory on the window gate at two streams is printed, not claimed: 69/80 against 65/80 (7 vs 3, p = 0.17)" | SUPPORTED | rec |
| 5.21 | "a pre-registered, two-machine, disjoint-seed test of a recipe that uses no stream labels, no restarts and no hinge, and that finds an eight-way partition … with one channel per stream in 38 of 40 runs where the previous recipe found none" | SUPPORTED; "one channel per stream" OVERSTATED (as A4) | |
| 5.22 | "The split needs to know where the key positions are …; the window gate needs the cue within two tokens; and the task is the grouped layout" | SUPPORTED | tex 118; the window's width |
| 5.23 | "The audit's over-prediction for two streams … was not met on X (33/40) and KEY failures were reduced, not removed; what was claimed, G3, holds" | SUPPORTED, selectively | the audit's prediction was for seeds 160–199 (31/40 there); on the main test L reached 36/40, which the sentence omits (it appears at tex 276) |

## Section 6, the split (tex 171–195)

| # | sentence | verdict | evidence |
|---|---|---|---|
| 6.1 | "Revision 7's screens had no reset-only control" | SUPPORTED | audit §2.3 |
| 6.2 | "Phase VI ran four looks at it, on disjoint seeds and with two different implementations of the reset" | SUPPORTED | S50 (replayed), S58b and the control (live), S58; seeds 260–269, 460–469, 540–544/1540–1544, 420–459 |
| 6.3 | Table 3, rows 1–3 | SUPPORTED | rec T3 |
| 6.4 | Table 3, row 4: S58 "39/40, 39/40, 0 vs 0" | SUPPORTED, with a caveat | RESET\|453 counts as routed only through an argmax over a near-tie (two streams put 0.500 and 0.498 of their mass on one channel) (rec T3 row 183) |
| 6.5 | caption: reset-only fired under the live rule … on targets labelled on target …; where a no-intervention arm exists, bound no more often than it | SUPPORTED | S58b reset 30/30 on target; L's control 10/10; S50 reset 2/10 vs no split 3/10; control 3/10 vs 4/10 |
| 6.6 | "Pooled over the two ten-seed screens … 19/20 against 4/20" | SUPPORTED (printed, not claimed, as stated) | rec |
| 6.7 | "The copy puts the idle channel's gate logits in a near-tie with the busy channel's, so the next gradient step can separate two merged streams; a reset leaves the idle channel where it was" | UNVERIFIABLE as mechanism | the records show the near-tie (row distance after a split 0.69–0.90 against 3–5 before, X 555) but not the separation step; at X 555's third split the merged pair moved together to the new channel rather than separating (map 13,_,13 → 4,_,4). Needed: per-step maps after each split |
| 6.8 | "'The copy, not the reset' remains printed, not claimed" | SUPPORTED | spec `specs/test_split_copy.md` (`fb3acb7`, after the report) |
| 6.9 | Gumbel on read and write: "collapsed routing even at four streams (33 of 40 runs merged)" | SUPPORTED | rec |
| 6.10 | "consistent with our read being routed where Raven's is dense" | interpretation, fair | lit row 13 |
| 6.11 | Gumbel write-only: 37/40, median 6000 vs 3600, lower margin; at eight 0/10, up to six sharing, 10 vs 0 against the split | SUPPORTED (counts); "made the merges worse" OVERSTATED (A14) | rec |
| 6.12 | Switch loss: 36/40, 34 of 36 maps differ, margin 0.87 → 0.46; "not carried to eight streams" | SUPPORTED | rec; S58b's selection rule picked devices by count (prereg §2) |
| 6.13 | two streams: "a learned gate temperature with a floor ran to the floor in every run, and a stable-max softmax changed nothing" | counts SUPPORTED, "ran to the floor" UNVERIFIABLE by me | S60: REF 8, TEMP_FLOOR 7, STABLEMAX 7 of 10 (rec); I did not locate the learned temperature in the records. Needed: τ at the end per run |
| 6.14 | "the gate wants to be sharper, not softer, and … a saturated gate marked a committed wrong partition, not a plateau before binding" | UNVERIFIABLE (interpretation of S60 diagnostics) | H report_2; not record-checked |
| 6.15 | S51: "the trigger once fired on a correctly routed map, found no empty channel, overwrote a live one and merged the run; in seven of its eight bound runs it never fired" | SUPPORTED | seed 245 at 14400; 7 of 8 (rec S6 guard) |

## Section 11, distant cues and the memory rule (tex 220–228)

| # | sentence | verdict | evidence |
|---|---|---|---|
| 11.1 | "Twelve papers … were read in full" | SUPPORTED | twelve notes in `docs/reading/`, each recording a full read |
| 11.2 | "None of the twelve carries an inferred context cue across distance without labels" | OVERSTATED | A19; lit row 15 |
| 11.3 | the content-routing study: "0.9–29% … 98–99.7%" | PARTIAL | 0.9% is Fourier *mixing*; raw embeddings 1.2%; the paper's distant object is the target, not the cue (lit row 14) |
| 11.4 | Flesch et al.: "a slow integrator must receive only the cue, slow integration merges interleaved contexts, … no pressure to use a constant cue" | first clause OVERSTATED, the rest SUPPORTED | lit row 9 |
| 11.5 | "Those are the failures the papers predict." | OVERSTATED | the papers do not study these gates or tasks; the predictions are the project's analogies (the notes mark them "[inference]") |
| 11.6 | BDH-CQ quotes ("linear correction rules on S", …, "remain proprietary") | UNVERIFIABLE | no reading note of 2608.09888; known only via "the brief"; arXiv unreachable here (lit row 2) |
| 11.7 | the reconstruction: affine-in-S, read-before-write rules = the delta-rule family; the loop matches TRM / recurrent depth, "of which [recdepth] explains why the frozen memory must enter every step" | OVERSTATED | the quoted sentence calls linear attention (which does not read S) a realisation of "linear correction rules"; Geiping et al.'s argument concerns the input embedding, not a memory (lit, BDH-CQ section) |
| 11.8 | "The 97.4% Sudoku figure … is from Pathway's blog …, not comparable to the Sudoku-Extreme numbers of [trm]" | SUPPORTED by the synthesis | `distant_cues_2026-10.md:83`; the blog itself not checked |
| 11.9 | session F's rule; "at β = 0 its logits agree with the Hebbian model's to 1e-7, and with the new path disabled it reproduces a recorded run bit for bit" | SUPPORTED | F checks_task0.json: logits max \|diff\| 1.2e-7 to 1.8e-7; "logits bit-identical" with the path disabled |
| 11.10 | "We had predicted that a single delta channel would be the negative control the project lacks" | SUPPORTED | `distant_cues_2026-10.md` §0.4 ("provably fails") |
| 11.11 | "with β = 1 and the fixed decay, on the two-stream header layout at P = 4 a single delta channel bound 8/10 (Hebbian 1/10), and on the grouped layout 4/10 against 0/10" | SUPPORTED | rec S11 F |
| 11.12 | "a post-hoc look at one rerun (seed 307) found the keys at the third layer tagged by stream and nearly orthogonal across streams" | SUPPORTED as labelled (one seed, post hoc) | F report_1; `s53_keydiag.json` |
| 11.13 | learned forget gate: "9/10 and 8/10 against 3/10 with the fixed decay" | SUPPORTED | rec |
| 11.14 | "median α of 0.18–0.26 at CTX in the first layer, 0.72–0.74 in the third, 1.00 at keys and values" | UNVERIFIABLE by me (post hoc; labelled) | F report_2; needed: the decay diagnostics file per run |
| 11.15 | "delta … ahead at two streams (36/40 against the recorded 31/40 …; Fisher p = 0.11, a bound by the screen's rule), equal at four (20/20 against 19/20) and worse at eight (3/10 against 9/10, 0 vs 6, p = 0.016)" | SUPPORTED | rec; "ahead" at p = 0.11 is fairly hedged |
| 11.16 | "even the delta perfect gate takes 22,800–28,800 updates against 2400–4800 for Hebbian, at about three times the compute per step" | SUPPORTED (transitions); compute UNVERIFIABLE by me | rec; compute ratio from F report_2 |
| 11.17 | "A learned write strength β collapsed toward 'do not write' whenever it was the only change" | SUPPORTED (count); β values from report | SINGLE_DELTA_L 0/20 (rec); β ≈ 0 from F report_1 |
| 11.18 | "At two streams the memory has routes to conflicting bindings that need no partition" | SUPPORTED (header 8/10; grouped 4/10) | at grouped two streams the single delta channel's 4/10 is far below the gated recipe's 69/80: the partition still matters there |
| 11.19 | "at eight streams … the same recipe on delta channels stalls at 3/10, and a single delta channel has not yet been run. That premise test … is the next screen." | **UNSUPPORTED** | S68 had run (rec S11 F; A24) |
| 11.20 | "If the one-seed observation of stream-tagged keys … holds up, the context is reaching depth in the header layout …; Session G's first task is to probe it directly." | stale | G's S55 probe had reported at `6cd36ff` (15:25 UTC): the cue is absent (held-out probe ≤ 0.6 at layers 2, 3 and final) from the trained *Hebbian window* stacks. It does not test F's delta channel, but it is the available evidence and it cuts against "the context is reaching depth" for the main-line model |
| 11.21 | "A header layout with random block lengths and eight keys per stream exceeded the oracle's budget (neither perfect gate bound within 24,000 updates)" | SUPPORTED | S54(d) oracles 0/2, 0/2 (rec) |
| 11.22 | "a lighter version with four keys per stream is the next step" | stale | F's S70 (RandHeaderTask-lite) was running: six oracle records on the branch at a2e55ed |

## Section 12, where the mechanism stands (tex 237–254)

| # | sentence | verdict | evidence |
|---|---|---|---|
| 12.1 | Table 4, rows of `test_window_gate` | SUPPORTED | rec T4 |
| 12.2 | Table 4, "one channel: 0 of >200" at S=2, no conv | UNVERIFIABLE | results/ holds 60 such runs (0 bound); the rest predate results/ (rec) |
| 12.3 | footnote: "Revision 7's full-hinge arm bound 35/40 on shared seeds, with routing recorded only on X (15/20)" | count SUPPORTED; "routing recorded only on X" **UNSUPPORTED** | HINGE4k16 bound X 17/20 + L 18/20 = 35/40; L's records carry `stream_gate` and give 16/20 bound routed (results/L/slow_start_results.json) |
| 12.4 | "Given the partition, every grouped-layout configuration tried binds" | **UNSUPPORTED** as written | the report's own Section 1 (tex 98) says the perfect gate without the convolution failed on four seeds at two streams. Those runs used a 7,552-parameter (N = 64) model (rec S1 conv); with the N = 256 model the perfect gate bound everywhere it was run. One of the two sentences has to change |
| 12.5 | "one Hebbian channel does not, where it was run (two streams at four and eight keys; four streams with four channels)" | OVERSTATED | with the convolution at lr 4e-3 one channel binds 18/40 and 10/20 at two streams, four keys (results/X/conv_lr); Revision 6 reported 34/80 (rec S12) |
| 12.6 | "A gate that reads three tokens, trained with the memory held slow …, and with a channel split when binding stalls, finds the partition at two, four and eight streams without labels or restarts" | SUPPORTED | 69/80, 38/40, 38/40 (the split only at eight) |
| 12.7 | "Slow memory keeps the memory from exploiting a non-stream split first" | OVERSTATED | at two streams slow memory on the window gate is 7 vs 3, p = 0.17, not shown; KEY failures 12 → 6 is the supporting evidence |
| 12.8 | "the window gate cannot count positions, so position splits are gone" | SUPPORTED | no POSITION failure in any window arm (rec); by construction |
| 12.9 | "the split breaks the merges that remain at eight streams by giving an idle channel the busy channel's logits" | SUPPORTED (effect); mechanism as 6.7 | G4; split vs reset |
| 12.10 | "The recurrent gate's pathologies … were obstacles of that gate" | SUPPORTED as interpretation | HINGE_D8 0/40 vs window arms |
| 12.11 | "A single delta-rule channel binds the two-stream header layout (8/10 …; 9/10 and 8/10 … at eight keys) and the grouped two-stream layout on 4/10" | SUPPORTED | rec |
| 12.12 | "The partition's advantage, so far, is in the number of contexts; one channel at eight streams is untested." | first clause SUPPORTED (by S68, uncited); second **UNSUPPORTED** | A24 |

## Section 13, retrospective, and the corrections to Revision 7 (tex 259–284)

### 13.1 Revision 7's claims, and whether each correction is itself supported

| # | Revision 7 claim → Revision 8 verdict | is the correction supported? | evidence |
|---|---|---|---|
| R1 | "At eight streams the gate forgets the context" → Superseded: true of the recurrent gate; the window gate "carries the context at every write position by construction and binds 38/40" | SUPPORTED | 38/40; "by construction" means the CTX token is inside the window at VAL positions; the window gate's *state* decoded the stream at only 0.55–0.67 at 2400–4800 under Adam (rec S4 b16), so "carries" is about the input, not the state |
| R2 | Rev. 7's recipe "binds without restarts at two streams and at four with sixteen channels" → Stands; replaced by a recipe that "equals it at two streams (within the bound), beats its early-window form at four (G1) and beats it at eight (G2)" | "equals it at two streams" OVERSTATED; the rest SUPPORTED | G3's interval [−0.079, +0.121] (stats §3); G1, G2 (rec) |
| R3 | "The gate needs only one token of memory" → Corrected: one at the read, two at the write | SUPPORTED | qpos at the query KEY with CTX one back, writes at VAL with CTX two back (audit §3.2, test_binding_capacity.py:218); S47 decoders measured VAL (README batch 16) |
| R4 | "Under Muon the recipe binds four streams sooner" → Not shown on disjoint seeds pooled; "the machines disagreed. The one-machine result was a seed effect." | "not shown" SUPPORTED; "a seed effect" **UNSUPPORTED**; tex 200's "did not replicate on the other machine's seeds" inverted | Rev. 7's E2 was shown on L, confirmed on X's fresh seeds (7 vs 1), reversed on L's (5 vs 7); M2's power was 0.42–0.62 against plausible effects; machine heterogeneity p = 0.070 (stats §6a) |
| R5 | "A cap on the gate's recurrence is being screened" → Resolved: not; "No capped arm bound (0/10, 0/10, 0/10 and 0/5); the cap acted from update 1" | SUPPORTED | rec S8 (one of the four arms, PREV_NOREC_M, removes the recurrence rather than capping it) |
| R6 | "The row copy breaks merges (S36/S37, which scored BOUND)" → Supported with bound routed and a reset-only control, in three exploratory looks; printed, not claimed | SUPPORTED | split vs reset in three looks (rec T3); the comparison isolates the copy given the reset |

### 13.2 Errors made during Phase VI (tex 276–284)

| # | sentence | verdict | evidence |
|---|---|---|---|
| E1 | predicted a single delta channel could not bind conflicting streams (it binds the header layout 8/10) | SUPPORTED | rec |
| E2 | predicted Gumbel noise could replace the split (on the read gate it collapsed routing; on the write gate it worsened eight-stream merges) | SUPPORTED as an error; wording as A13–A14 | |
| E3 | predicted a balance loss would attack merges ("where it was run there were no merges to attack") | SUPPORTED | S58 NONE's two failures are bound-not-routed, no merges (rec) |
| E4 | predicted a learned forget gate would widen the oracle gap (it shrank it, on the eight-key header layout) | SUPPORTED | F report_2: ROUTED 0/10, GDN 1/10 meeting the widening rule; medians −0.46 |
| E5 | the audit's ≥ 36/40 with no key failures (31/40 on 160–199; 33 and 36 of 40 on the main test's seeds, key failures reduced) | SUPPORTED | rec |
| E6 | "Session F's randomized header layout did not bind under the perfect gate within its budget …; the specification fixed the block lengths and left the key load open" | SUPPORTED | rec; prompt text (S54(d)) |
| E7 | "Three batch-18 runs sit at exactly η² = 0.5 … The reading that depended on them is withdrawn until the classifier is fixed" | SUPPORTED (one is 0.4999) | rec S9; `fail_class_v2.py` came after the report (`ba8901a`) |
| E8 | Attribution of the 97.4% Sudoku figure | SUPPORTED by the synthesis | lit row 7 |
| E9 | "Container restarts cost in-flight runs on X …, on F (twice …) and on E …; Every resumed segment began by reproducing a recorded run bit for bit; Batch 18's Muon reference did not reproduce across hosts and was rerun fresh" | partly UNVERIFIABLE | X's first start: consistent with the meta (six records); F: F report_2; E: batch logs on the branch, not read in full by me; batch 18's Muon reference: README batch 18 |

## Conclusion (tex 330)

| # | sentence | verdict | evidence |
|---|---|---|---|
| C1 | "Multi-channel Hebbian plasticity in multilayer BDH solves context-conditional binding by partitioning memory" | OVERSTATED | it solves the grouped synthetic binding task when the cue is within the window; distant cues are unsolved (tex 228, 303); at two streams a single delta channel binds without a partition (11.18) |
| C2 | "and a label-free gate finds the partition at eight streams: a window gate, slow memory for 2400 updates, and a channel split when binding stalls, in 38 of 40 runs on two machines with disjoint seeds, where the previous recipe bound none" | SUPPORTED | rec T2; "label-free" in the stream sense (the split reads key positions) |
| C3 | "In every look so far the split's row copy, not its reset, is what works" | SUPPORTED for "not the reset alone"; see A12 | |
| C4 | "The obstacle Revision 7 named, the gate's recurrence, was removed rather than fixed." | SUPPORTED as description | |
