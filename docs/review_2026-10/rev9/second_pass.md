# Revision 9 draft, second version (`f67bc07`): second pass

*Referee V, 11 October 2026.*

**What was checked.** `multichannel_hebbian_report_v9.tex` / `.pdf` on `claude/rev9-draft` at `f67bc07`, committed 02:43:57 UTC. `tex2:NNN` is a line of its tex; `tex:NNN` is the first draft (`9bfa85a`).

**How.** For every row of `revision_9_check.md` and of the five first-pass detail files, this pass says whether the second version resolves it. Each first-pass script was re-run as a v2 that finds its quotes in the new tex by text search. The three reports folded in after the first pass were checked against their reports and the records:
- G's report 5 (S82);
- H's reports 6 and 7 (S84, S85).

**Files, all in `rev9/`:**

| file | content |
|---|---|
| `check_split_copy_v2`, `check_H_v2`, `check_F_v2`, `check_G_v2`, `check_appendixB_v2` (`.py` with `.md` output) | the v2 checks |
| `resolution_split_copy_appB.md`, `resolution_H.md`, `resolution_F.md`, `resolution_G.md`, `resolution_V.md` | row-by-row RESOLVED / PARTLY / NOT / NEW ERROR, with the tex2 line |
| `count_allpos.py` | the limits' 97 of 107 |

**Rules followed.** Nothing outside `docs/review_2026-10/` was changed, and no training was run. The Appendix B v2 built the Part C model and CEIL_C at seed 540, read-only.

## Verdict

**Not ready to merge.** The second version fixes the great majority of the first pass's findings:
- the probe records;
- S82, S85 and R's Phase 4 state;
- "all reproduce";
- the 70 L rows;
- the CPU, and R's 10⁻⁸ (now 7×10⁻⁸);
- the delta "parity";
- the title and scope;
- the β collapse;
- the S80 and S71 ranges;
- S61's arms and firings;
- Appendix B's forward pass and its six missing entries.

Its tables and every claim's count, pair, p-value and interval still reproduce. The limits' new 97 of 107 reproduces exactly, and so do the folded-in S82, S84 and S85 counts and readings.

What remains:
- **11 statements that are false against the records or contradict the draft itself**, some new in this version (§1). Two of them the draft inherited from my own documents, now corrected here.
- **About ten unmarked descriptions, against tex2:342's claim that all are marked** (§2).
- **A short list of wording fixes** (§3).

None of these touches a claim's number. All of them are statements of fact.

| check against `f67bc07` | rows | DISAGREE | PARTIAL | not record-checkable | MATCH |
|---|---|---|---|---|---|
| `test_split_copy` (X) and its new sentences | 45 | 0 | 1 | 1 | 43 |
| session H, with S84 and S85 | 92 | 3 | 9 | 1 | 79 |
| session F | 162 | 0 | 8 | 7 | 147 |
| session G, with S82 | 137 | 1 | 14 | 1 | 121 |
| **total** | **436** | **4** | **32** | **10** | **390** |
| Appendix B against the code (atomic items) | 101 | 1 FALSE | 2 | 4 unverifiable | 94 TRUE |

Resolution of the first pass's items:

| scope | items | RESOLVED | PARTLY | NOT | NEW ERROR in this version |
|---|---|---|---|---|---|
| split_copy and Appendix B | 42 | 37 | 2 | 3 (none an error) | 1 |
| session H | 54 | 51 | 1 | 2 (one item, counted twice) | 3 in the new S85 sentence |
| session F | 54 | 48 | 3 | 3 | 2 |
| session G | 42 | 33 | 5 | 2 (one item, counted twice) | 2 (one item, counted twice) |
| the referee's own rows | 23 | 16 | 5 | 1 (A5, the Phase 4 "correction") | 1 among the 23 (tex2:327); plus, outside them: 4 more (tex2:325/344/388/470, 488, 489, 510), tex2:427 left standing, and the 914 / 393 slips inherited from the referee's documents |

The first pass's 22 disagreements are all RESOLVED, except B5. B5 is PARTLY: the draft over-generalised it; see §1 item 5.

## 1. False against the records, or contradicting the draft itself (must change)

**1. tex2:327: the 44 entries.** "By R's own labels on its 44 entries, the text ... was ambiguous or silent on **nine**".
- R's labels (UNSTATED.md): 20 [A], 12 [B], 2 [C], 2 [stated + A], 1 [A in code], 4 [stated], 3 unlabelled.
- So **37 of 44** are entries where the text was silent, ambiguous or carried over.
- The first example the draft gives, "the gate's input", is U-9, which R labels [stated].

**2. tex2:325, 344, 388, 470: the Phase 4 reading.** "The amended Phase 4 reading was issued to session R as a prompt"; "session R's Phase 4 under the amended reading".
- No amended reading exists in any commit.
- The only Phase 4 prompt is the overlap rule (`docs/reading/parallel_sessions_prompts.md:731-742`, unchanged at the main line's head `0282955`).
- If a prompt was sent outside the repository, cite it; otherwise write "proposed".

**3. tex2:427: the probe readings.** "a probe whose readings the records do not keep".
- This contradicts the corrected tex2:205 and 390.
- The records keep every check's probe reading, the evaluation at the same step and the per-stream masses at every evaluation.

**4. The provenance tables:**
- **tex2:488.** split_copy's L half is "running", but tex2:41, 53, 195 and 469 say it has not run. No L file exists on any branch.
- **tex2:489.** delta_recipe is "not written", but `test_delta_recipe.py` was committed at `a581027`, 02:12:29, 31 minutes before this version.
- **tex2:510.** R's "Phase 4 running", but tex2:41 says it has not started, and no record exists.

**5. tex2:246 (and 344): the six delta splits.** "at each [of the six] ... the split copied a channel that was not the merged one" / "they copied the wrong channel". True of five.
- At S76 seed 1314, update 9600, the map was [6, 5, 4, 6, 10, 13, 2, 2]. The split copied channel 6, which held streams 0 and 3, a merged channel, onto channel 10, which held stream 4.
- It failed the label because its target was not empty.
- The sentence's own parenthesis says "five of the six".

**6. tex2:438 (and 457): the forget gate.** The learned forget gate is "the one device screened against [the memory horizon]" and "worked on the delta write".
- S59 is the horizon screen (`explore_f_decay.py:3-4`). Its pre-fixed reading "the horizon knob matters" came out not to apply: gap_d − gap_FIXED was negative on 7 of 10 seeds per decay, median −0.46.
- It held the length fixed (L = 37).
- The 9/10 and 8/10 against 3/10 are binding counts on the header layout, not a horizon result.

**7. tex2:261: the S56 plateau.** S56 REG_NUDGE "plateaus at 0.32–0.44".
- The records give 0.20–0.44: seed 378 is at 0.20 (streams 0.19/0.21) and seed 379 at 0.31.
- 0.32–0.44 is `report_2.md:86`'s range, not the records'.

**8. tex2:228: the S85 descriptive sentence**, copied from report 7, with three numbers the records do not support:
- **η² by key "falls to 0.00–0.12 by update 25 and stays below 0.07 to 2400".**
  - Under the key term, update 25 reads 0.001–0.195 (seeds 2326 0.168, 2327 0.195).
  - Over updates 50–2400, 7 of 20 runs reach ≥ 0.07; the largest is 2327 at 0.165.
- **"0.21–0.64 at 3600 on some seeds".** The runs at ≥ 0.2 span 0.21–0.88 (seed 2321).
- **"organised by key on six of the twenty seeds (η² up to 0.84 at update 0)".** Five exceed 0.5: 2313 0.70, 2318 0.64, 2320 0.84, 2326 0.56, 2328 0.76. Six only comes out at an unstated cutoff of about 0.47 (2319, 0.476). The classifier's threshold is 0.5. Report 7's own table shows five.

**9. tex2:205 (and 390): R's seed 914 and the probe-blind count.** These are inherited from the referee's documents, both corrected on 11 October.
- **Seed 914.** R's probe read **0.953–1.000** from 16,800 on (1.000 at 24,000), not 0.953–0.984. `replication_review.md` gave 0.984, and the first pass marked the draft's copy MATCH.
- **The count.** "6 of the 393 checks" counts 393 *probe readings*: 353 checks from 4800, plus 40 reference readings at 2400. All six are at ≥ 14,400, so it is **6 of 353 checks**.

**10. tex2:336: the premise test.** "session F's premise test ... supported the reading". Unchanged from tex:332, and missed by both earlier passes.
- The 82 minutes reproduce.
- But S68's two pre-fixed readings both came out "neither", which tex2:357 now says itself.

**11. The Appendix B FALSE item.** It is item 9's 914 range (`check_appendixB_v2.md` row 1).

## 2. Unmarked descriptions; tex2:342 says "those are now marked"

tex2:342's sentence is not true of these places, which state as findings what no pre-fixed reading decided:

| tex2 | statement | why it is a description |
|---|---|---|
| 207 | "it changed the routing's shape, not its count" (S58 balance loss) | the count part is 36/40 against the no-device arm's 38/40 (1 vs 3 discordant) |
| 203 | S84: "the probe's noise decided whether the first check fired"; "the base kept enough splits for its merges" | interpretations, unmarked |
| 252 | the raw-key fragility mechanism (β‖x‖² > 2 makes the update expanding) | report 5's post hoc reading of one diverged run in eight |
| 384 | "(it needed layer 1)" | unchanged from tex:378. The interpretation is marked "ours" at tex2:261, not here, and it crosses tasks and gate inputs |
| 36, 265, 316, 455 | "key splits that form first" | an order read from the traces, not a reading. The records agree in all 28 key-split failures, but it is unmarked |
| 265, 384 | SLOWGATE "worse than nothing", "because it delays the write" | S82's "the device hurts" did not apply (0 vs 0 against the baseline, 0 vs 2 against REG3, p = 0.25); the cause is interpretation |
| 440 | "the one repair device that works at eight streams cannot work here" | stated as fact; tex2:209 marks the same as "our reading"; S78 measured only "neither device" |
| 436, 465, 228 | the key term "prevents the key split at two streams", "that nothing repairs"; the conclusion's "found on every seed" | no "on a screen" marker. S85 is 5 vs 0, p = 0.031, on 20 seeds; S82 is a screen; "nothing repairs" generalises two forms of one device (S78) |

## 3. Wording fixes (PARTIAL; each holds under one reading, not the one the sentence names)

**Session F:**
- **tex2:355.** "the stall ... came from a screen with a learned write strength" reads as a cause, which tex2:246 calls post hoc.
- **tex2:338.** "Gated DeltaNet's write ... at two, four and eight streams": that write ran with α fixed at 0.95 and β = 1. Gated DeltaNet's learned gate ran only on the P = 8 header layout.
- **tex2:357.** "the same counts": C3 is at eight streams only, so only S68's 0/10 is the same count.

**Session G:**
- **tex2:263.** The label "discovered (register)" is S73's; S72's reading is "discovered" / "not discovered".
- **tex2:263.** "Not separated above keys and values" holds in 27 of 28 key-split runs (REG3|712: CTX 0.29 against 0.19/0.18).
- **tex2:263.** The half-routed baseline's "0.86–1.00" is at keys; VAL is 0.79–0.99.
- **tex2:265.** "binds 10/10 against 0/10 and 2/10" on delta is ROUTED-BOUND; phase 2 is descriptive by the docstring and is not said to be. On BOUND the references are 2/10 and 4/10. Also, "the key split does not return" is observed only to the run's stop (6000–8400).
- **tex2:36.** "reads the task's key positions and identities as the split does": the split reads key positions only. tex2:265 words it right.
- **tex2:455.** "a fifth of seeds" leaves out S82's 2/20 on fresh seeds.

**Session H:**
- **tex2:60.** "the flat-start guard aims the splits": the pre-fixed criterion (labelled ok) cannot fail at the flat start. All 24 of the base arm's splits were labelled ok, 7 of them at accuracy 0.06–0.16, as report 6 says.
- **tex2:228.** "the slow gate ... with the same four key failures": the same number on different seeds. The baseline's are 2310, 2318, 2323, 2328; the slow gate's are 2310, 2321, 2323, 2328, and its 2323 is KEY only under v2.
- **tex2:228.** "at every 25 updates" holds to 1200 only.
- **tex2:228.** "the baseline's real splits form at 25–125 and the slow gate moves them to 275–575": the ranges recompute, but only two seeds are common to both. Five of the baseline's seven onsets do not occur under the slow gate at all.
- **S84's readings.** EVAL_GUARD_INT1200 (9/10, 0 vs 0) and its "costs binding" reading are not stated.

**Session V (the referee's own rows):**
- **tex2:327.** Two of the "eleven" matched guesses the draft lists, the 0.02 initialisation and the map, are not among DIFF.md's eleven bullets.
- **tex2:330.** "reports 3–6": the 808 rows cover F 3–5, G 3–4, H 4–5 and I 1. F's report 6 was checked in `rev9/check_F`.
- **tex2:344.** "five ranges": the first pass listed three (B6–B8).
- **tex2:458.** "One channel of either write rule is run at every size": at S = 2, 4 and 8, not at the batch-19 sizes. tex2:334 words it right.
- **tex2:511.** The V row of the batches table stops at `0e6c189`, before the check (`e8fbb25`) and the spec review (`fad1489`) that the draft cites.

**Main line:**
- **tex2:36.** "the 44 details ... stated in an appendix": the held-out set's generator seed (`EVAL_SEED = 424242`, `test_binding_onset.py:116`) is not given.
- **tex2:199.** It does not say that CHECK 130 "differs" (printed, not asserted).

## 4. The new material

- **S82** (G's report 5, `8973d48`). Every count and reading reproduces:
  - KEYHINGE 20/20 ROUTED-BOUND against BASELINE 0/20 and REG3 2/20; SLOWGATE 0/20; NOWIN 5/20.
  - Delta: 10/10 against 0/10 and 2/10. Validity: 2/2 and 2/2.
  - "discovered" applies to KEYHINGE (20 vs 0) and to NOWIN (5 vs 0, p = 0.031); "helps" applies to KEYHINGE only (18 vs 0).
  - The best-device rule recomputes, and `s82_best.json` was committed before the first phase-2 record.
  - The readings (`83699a2`) predate the first record.
  - The draft says the hinge reads key positions and key identities and no stream label.
- **S84** (report 6, `37c1fbf`). Every count matches:
  - guard 9/10 against 9/10;
  - 17 splits at 0.50–0.91 against the base's 7 of 24 at 0.06–0.16;
  - seed 2306 never split in a guarded arm;
  - the fast check with the guard 9/10, a median 1200 earlier;
  - the evaluation trigger 10/10, 1 vs 0.
- **S85** (report 7, `1e09ecd`). The counts match: the key term 20/20 against 15/20 (5 vs 0, p = 0.031), no key failure against four; both devices 19/20. The descriptive sentence does not match (§1, item 8).
- **Readings before runs.** The readings of S84 and S85 (`834581a`) precede their first records.
- **The limits' count.** 97 of 107 reproduces exactly (`count_allpos.py`):
  - 50 runs and 107 firings;
  - the all-position pair labelled at 97 (46/47, 30/38, 21/22), the key-position pair at 107, the two coinciding at 46;
  - the stored `all_ok` and `key_ok` equal the labels recomputed from `map_before` at all 214.

## What could not be checked

- The PDF, page by page.
- Whether L's split_copy half, F's S81 or R's Phase 4 are running in processes that have committed nothing. Git shows only commits.
- That seeds 2300–2329 are fresh; this comes from H's own scan.
- The tex2:228 statement "the stream routing having formed first", which would need η² by stream.
- tex2:390's "nobody read the two against each other until the referee did".
- The local-time offsets of tex2:492; the records carry no zone.
