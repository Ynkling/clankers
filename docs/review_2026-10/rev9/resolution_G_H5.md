# Sessions G and H: fourth draft to fifth draft (referee, fifth pass)

*Referee V, 12 October 2026.*

**What is compared.**
- Fourth draft: `c66cc76` (2026-10-11T07:52:07Z), cited tex4:NNN (564 lines).
- Fifth draft: `c5e4508` on `claude/rev9-draft` (2026-10-12T19:14:19Z), `multichannel_hebbian_report_v9.tex`, cited tex5:NNN (568 lines).
- **Line mapping.** Two `test_split_copy` L rows were inserted after tex4:192, and tex4:452 became tex5:454–456 (session I's S75 paragraph). So tex4:N is tex5:N up to 192, tex5:N+2 from 193 to 452 (tex4:203 → 205, 269 → 271, 341 → 343, 439 → 441) and tex5:N+4 from 453 on (459 → 463, 472 → 476, 509 → 513, 511 → 515).
- **The commits that answer the fourth pass for G and H:**
  - `e92e1c6` (15:50:35Z): H/S86, at tex4:203, 341, 439 and 459;
  - `fabf505` (15:54:51Z): G/S83, at tex4:269 and 439;
  - `fd2fa89` (15:58:36Z): G's ratio-term account marked, at tex4:269.

  The later commits up to `c5e4508` touch G or H text only at tex5:41 ("every session's queue is empty"), tex5:205 (F's S81 inserted into H's S79/S84/S86 paragraph, with a three-screen sum over S84, S86 and S81) and tex5:441 (the 50-run count, which this version's own L fold-in overtakes). Session I's S75 paragraph (tex5:456) names G's register gate.

**Records.**
- G: `claude/explore-G` at `076005d` (report 6), the head at the draft's commit; no G commit between 05:08 on 11 October and the draft.
- H: `claude/explore-H` at `9b72bf6` (report 8), the head at the draft's commit; no H commit between 07:49 on 11 October and the draft.
- **Both branches have moved since the draft (fetched 22:5x UTC).** Neither move changes a record the draft cites:

  | branch | commits after the draft | what they are |
  |---|---|---|
  | G | `3c89f87` (22:48:17), `02b3eed` (22:48:43), `f632291` (22:53:29) | S87 specified, started, progress (`s87.log`, `s87_repro.json`) |
  | H | `5b5deec` (22:40:01) | "round 5 (2)": a recount over the main line's records, no runs |
  | H | `f4fd535` (22:45:48) | S87a (ALLMASS vs BASE on 20 more seeds) and S87c, code and readings |

- Both checks read each branch as it stood at `c5e4508` (the last commit at or before 19:14:19), and assert that this is the pinned commit.
- `claude/rev9-draft` itself has moved to `cc379b2` (22:52:37), which is **not checked here**.

**Evidence.** Both scripts are in this directory; each writes its md and nothing else.
- **`check_G_v5.py` → `check_G_v5.md`.**
  - A copy of `check_G_v4.py`. Every first- to fourth-pass row keeps its recomputation. Fourth-pass rows are asserted on tex4 and then located in tex5 by text search; a row whose fragment is gone is listed as REMOVED4.
  - Rows `new` (row5) cover every changed G sentence.
  - New recomputations from `s83_results.json` at `076005d`: the KL term at every recorded update (1, 300, 600, 1200, 2400; per layer and the layer mean); the register's β^R at every evaluation; and the term's weights at 2400 and 2401. It also reads the specification's caveat (`explore_g_s83_child.py`:21–23 at `b2f9907`) and report 6's own account.
- **`check_H_v5.py` → `check_H_v5.md`.**
  - A copy of `check_H_v4.py`. Every row of the fourth pass is located in tex5 by its tex4 fragment, and a rewritten sentence must be re-anchored (OV5) or the script stops.
  - New rows (NEW5) are recomputed from S86's store, the main line's split records including L's `split_copy` file (`ffb0771`), and F's S81 store (`d5851b1`, for one sum only).

**Statuses.**
- **RESOLVED**: the sentence is now true against the records.
- **PARTLY**: the error named is fixed, but part of the item still does not hold.
- **NOT**: unchanged, or the same problem remains.
- **NEW ERROR**: the replacement text is itself contradicted by the records.

## Tally

**The checks.**

| check against `c5e4508` | rows | DISAGREE | PARTIAL | NOT RECORD-CHECKABLE | MATCH |
|---|---|---|---|---|---|
| `check_G_v5` (G with S83) | 159 | 1 | 2 | 2 | 154 |
| `check_G_v5`, carried from passes 1–4 (68 + 48 + 16 + 18) | 150 | 0 | 0 | 1 | 149 |
| `check_G_v5`, new for the fifth version | 9 | 1 | 2 | 1 | 5 |
| `check_H_v5` (H with S86) | 116 | 0 | 1 | 2 | 113 |
| `check_H_v5`, carried (5 re-anchored + 107 unchanged) | 112 | 0 | 0 | 1 | 111 |
| `check_H_v5`, new for the fifth version (2 new sentences + 2 new checks of unchanged ones) | 4 | 0 | 1 | 1 | 2 |

- **REMOVED in `check_G_v5`:** 5 fourth-pass rows (4 PARTIAL, 1 MATCH), all on rewritten S83 sentences. Each has a replacement row. No first-, second- or third-pass row is newly removed.
- **REMOVED in `check_H_v5`:** 3 rows, gone since the second draft.
- **H's fourth-pass statuses → fifth:** DISAGREE → MATCH 3, PARTIAL → MATCH 2, MATCH → MATCH 106, NOT RECORD-CHECKABLE → same 1.

**The fourth pass's items in this scope.**

| source | items | RESOLVED | PARTLY | NOT | NEW ERROR |
|---|---|---|---|---|---|
| `fourth_pass.md` §1 items 4–5 (tex4:203; tex4:341, 459) | 3 | 3 | 0 | 0 | 0 |
| §2, G bullets (tex4:269 ×5, tex4:439 ×2)¹ | 7 | 4 | 1 | 1 (not record-checkable; no fix needed) | 1 |
| §2, H bullets (tex4:439, tex4:203) | 2 | 2 | 0 | 0 | 0 |
| §2a, G's report 6 (outside the draft) | 2 | 0 | 0 | 2 (report unchanged; the draft carries neither) | 0 |
| **all** | **14** | **9** | **1** | **3** | **1** |

¹ The tex4:269 bullet "both clamp the gate to exactly uniform and the register's write never leaves its initial value" has two halves, counted apart: the write (RESOLVED) and the gate (rewritten into a NEW ERROR).

**New in the fifth version, outside the items:**
- **Two PARTIALs about the specification's caveat:** tex5:271 "which S83's specification had allowed in advance", and its echo at tex5:441, which is the PARTLY item above.
- **One sentence made loose by this version's own fold-in:** tex5:441, "the 50 recorded split runs at eight streams".
- **One not-record-checkable statement now out of date for both sessions:** tex5:41, "every session's queue is empty".
- **One comma splice:** tex5:441.

## 1. `fourth_pass.md` §1 items 4 and 5 (session H)

| item | tex4 | tex5 now | evidence | status |
|---|---|---|---|---|
| §1.4 | 203: "the all-position pair copied onto an occupied channel" | 205: "the all-position pair copied a channel that held no stream onto another that held none, leaving the shared channel as it was, after which the run merged while the key-position rule's run bound" | S86 at `9b72bf6`, seed 2334, update 19200: the all-position pair 1→6, and channels 1 and 6 hold 0 streams each. Map before [10, 7, 15, 0, 4, 7, 3, 9] = map after. The one shared channel is 7 (streams 1 and 5). (`check_H_v5`, row tex5:205) | **RESOLVED** (`e92e1c6`) |
| §1.5 | 341: "a trigger that reads all positions is not yet screened" | 343: "a trigger that reads all positions was screened once, session~H's S86, $8/10$ against $9/10$" | ALLMASS 8/10, BASE 9/10, "keeps it" | **RESOLVED** (`e92e1c6`) |
| §1.5 | 459: "a trigger on all positions is the next screen" | 463: "a trigger on all positions was screened once (S86) and kept binding within one pair" | BASE only 1, ALLMASS only 0; 1 − 0 ≤ 1 | **RESOLVED** (`e92e1c6`) |

No line of tex5 still says "not yet screened" or "is the next screen" of the all-position trigger.

## 2. `fourth_pass.md` §2: the G and H bullets

| bullet | tex5 now | evidence | status |
|---|---|---|---|
| tex4:269 "each S82 reference reproduced bit for bit through 2400 first" | 271: "(S83; each S82 reference arm reproduced bit for bit through 2400 at one seed first)" | `s83_repro.json`: all six reference arms True (03:16:03 UTC), before the first S83 record (`15b280b`, 03:30:49). `explore_g_s83.py`:12–15 reruns each arm at seed 2100. | **RESOLVED** (`fabf505`) |
| tex4:269 "the register's write never leaves its initial value" | 271: "and the register's write stays at its initial value through 1200, moving only afterwards" | max over runs of max \|β^R − 0.10\| over CTX, KEY and VAL:<br>- POS: 0.0001 at 0, 0.0120 at 1200, 0.311 at 2400 (more than 0.03 in 14/20).<br>- FLAT: 0.0071 at 1200, 0.006 at 2400 (0/20), and moved by the end in 20/20. | **RESOLVED** (`fabf505`) |
| tex4:269 "both clamp the gate to exactly uniform" (same bullet) | 271: "both hold the gate at uniform while they act (its KL from uniform at most $10^{-5}$ through update 1200 in every run)" | §4 below, the first new finding. The stated bound is exceeded (POS 1.4e-5 at 1200, every run at update 1). "While they act" is false for POS, whose term acts through 2400, where its KL reaches 0.068 in 13/20 runs. | **NEW ERROR** (`check_G_v5`: DISAGREE) |
| tex4:269 the ratio-term clause, unmarked | 271: "where, on session~G's account, the ratio term removes only the share of the gate's variation organised by key and leaves it free to commit by stream" | report_6.md:49, under "What happened (exploratory)" | **RESOLVED** (`fd2fa89`) |
| tex4:439 "a term reading positions only, or nothing, fails" | 441: "session~G's S83 found that its KL terms reading positions only, or nothing, fail ($1/20$, $0/20$) by clamping the gate, a weakness its specification had allowed to the form" | POS 1/20, FLAT 0/20, so the scope ("its KL terms") is fixed. But the appositive says the specification allowed *the clamp*. The caveat (`explore_g_s83_child.py`:21–23) allowed a null result as the form's weakness, and named the opposite mechanism: the KL's "gradient vanishes at the uniform gate". Report 6 (l.47): "My spec's caveat expected the opposite failure: the absolute KL term too weak near uniform. It was too strong." | **PARTLY** (`check_G_v5`: PARTIAL) |
| tex4:439 "which S83 found needs key identities" | 441: "the hinge's analogue, which S83's one position-only term could not replace, is untested" | "The identity term is needed" applies (19 vs 0, p = 1.9e-6), with POS at 1/20. This is the wording the fourth pass proposed. | **RESOLVED** (`fabf505`) |
| tex4:269 "proposed to follow `test_delta_recipe`" | 271: unchanged; 476: "`test_delta_recipe`'s L half (X's is complete), then the hinged-register test" | No record states the order. `specs/` at `c5e4508` holds `test_delta_recipe.md`, `test_split_copy.md` and `test_window_gate.md` only. | **NOT** (not record-checkable; the draft's plan, no fix needed) |
| tex4:439 "1 mistarget in 21 at the same rate" (H) | 441: "S86 then found 1 mistarget in 21 firings, consistent with that rate (10 in 107)" | 1/21 = 0.048 against 10/107 = 0.093; P(≤ 1 of 21) = 0.40. With L's split_copy, 11/126 = 0.087 and P = 0.44. | **RESOLVED** (`e92e1c6`). Removing the "and" left a comma splice: "would have gone differently, S86 then found". |
| tex4:203 "all body positions" (H) | 205: "the mean read-gate mass over every input position" | `select` averages over every position of `data[:, :-1]`: 96 body positions and the query's stream and key, 98 in all | **RESOLVED** (`e92e1c6`) |

## 3. `fourth_pass.md` §2a: G's report 6 (not the draft)

| item | now | status |
|---|---|---|
| report_6.md:38–39 "η²_key 0.18 at update 1": this is the term's value, and η² is about 0.42 | `report_6.md` unchanged (no G commit touches it after `076005d`). The draft gives no number (tex5:271 "on our reading of G's traces, the hinge has nothing to release there"). | **NOT** (outside the draft; no draft impact) |
| report_6.md:52–53: the 1200-window failures 2103 and 2112 split "while the stack was still at its slow rate" | Unchanged. In the records, η² by key is still near 0 at 2400 and first reaches 0.5 at 3600, after the slow phase. The draft does not repeat the claim (tex5:271 says only "one pair short"). | **NOT** (outside the draft) |
| **New, not in §2a:** report_6.md:44 "their value is 0.0000 from update 300" | True to four decimals through 1200 for both terms, and for FLAT at 2400. Not for POS at update 2400, while its term still acts: 12 of 20 runs are at or above 0.00005, with a maximum of 0.068. | report error, for G to correct; the draft does not quote it |

## 4. NEW ERROR, and the new PARTIALs

**tex5:271, "both hold the gate at uniform while they act (its KL from uniform at most $10^{-5}$ through update 1200 in every run)"** — `check_G_v5`: DISAGREE.

The records (KL(g ‖ U) per layer and per run at each recorded update; the term is the layer mean):
- **Update 1**, which "through update 1200" includes: 1.2e-4 to 7.9e-4, in 40 of 40 runs.
- **Updates 300 and 600:** at most 2e-6.
- **Update 1200:**
  - POS seed 2109 has layers 1.3e-5, 1.4e-5 and 9e-6, a layer mean of 1.2e-5.
  - POS seed 2118 has 1.2e-5 in one layer.
  - FLAT is at most 5e-6.
- **Update 2400:** both terms are still on, with weight 1 at 2400 and 0 at 2401 in all 40 runs. The draft itself says "updates 1--2400" two sentences earlier.
  - POS's KL is above 1e-5 in 13 of 20 runs, with a maximum of 0.068.
  - POS's write has moved more than 0.03 in 14 of 20 runs.
  - FLAT stays at most 2e-6.

The clamp is real and the magnitudes are small. But both the stated bound and "while they act" are contradicted.

*Exact wording:* "both held the gate at uniform from update 300 through 1200 (its KL from uniform at most $1.4\times10^{-5}$ in every run; the flat term's through 2400)".

The fourth pass's own proposed wording caused this; see §7.

**tex5:271, "and the one form tried of each device that reads less fails, by clamping the gate, which S83's specification had allowed in advance as a weakness of the form rather than of reading less"** — PARTIAL.
- **What holds.** "The one form tried of each" and the counts (POS 1/20, FLAT 0/20) are right.
- **The antecedent of "which" is ambiguous.** Read as "the failure", the clause is true: the caveat says "A null result for pos / flat may be a weakness of this form, not of 'reading positions'". Read as "clamping the gate", it is false: the caveat expected the term too weak (its gradient vanishing at uniform), and report 6 says it was "too strong".
- **"By clamping the gate" is report 6's account,** stated without a marker. The clamp itself is in the records.
- *Exact:* "…, which S83's specification had allowed in advance as a weakness of the form (though it expected the term too weak, not too strong)".

**tex5:441, "a weakness its specification had allowed to the form"** — PARTIAL. The same point, and here the appositive attaches unambiguously to "clamping the gate". This is the PARTLY item of §2.

**tex5:441, "at the 107 firings of the 50 recorded split runs at eight streams (`window_gate` X and L, `split_copy` X)"** — PARTIAL (`check_H_v5`, new check; the clause is unchanged since tex4:439).
- **Why it is now loose.** This version folds in `test_split_copy`'s L half (`ffb0771`, committed 02:39:51 on 12 October, before the draft; tex5:193–201). So the recorded split runs at eight streams are now 60, not 50.
- **Recomputed** from the four files with the screen's label rule:

  | firings | all-position pair labelled | key pair labelled | same pair | mistargets |
  |---|---|---|---|---|
  | 126 | 115 (L: 18 of 19) | 126 | 54 | 11 of 126 = 0.087 |

  S86's 1 in 21 remains consistent with that rate (P = 0.44).
- The count is right for the three files the sentence names. H's own recount after the draft (`5b5deec`, `onetoone_counts.txt`) gets the same 60 / 126 / 126 / 115.
- *Fix:* "of the 50 split runs recorded before `test_split_copy`'s L half", or the 60-run count.

**tex5:41, "every session's queue is empty"** — NOT RECORD-CHECKABLE (both checks). A queue is not a record.
- At the draft's commit, neither G nor H had committed for 35–38 hours.
- After it, both opened round 5:
  - H: `5b5deec` (22:40:01, titled "round 5 (2)") and `f4fd535` (22:45:48, S87a/S87c readings);
  - G: `3c89f87` (22:48:17, S87 readings), then S87 started.
- So the sentence is now out of date for G and H. Whether their round 5 had been set before 19:14 cannot be told from git.

## 5. Every changed line of `git diff c66cc76 c5e4508` touching G or H

| tex5 | change | rows | status |
|---|---|---|---|
| 41 | "F's S81, G's S83, H's S86 and R's Phase~4 are complete, every session's queue is empty" | G/H "complete" carried (MATCH); queue (both) | MATCH; NOT RECORD-CHECKABLE |
| 205 | `e92e1c6`: "every input position"; "copied a channel that held no stream onto another that held none, leaving the shared channel as it was" | OV5 ×2 | MATCH, MATCH |
| 205 | F's S81 inserted into H's S79/S84/S86 paragraph; "Over the three screens of the evaluation as the trigger's probe (S84, S86, S81) it is $29/30$ against $27/30$, 2 vs 0, descriptive; it has not cost a run" | NEW5 | MATCH. S84 EVAL 10/10 vs BASE 9/10, 1 vs 0 (`37c1fbf`); S86 9/10 vs 9/10, 0 vs 0 (`9b72bf6`); S81 10/10 vs 9/10, 1 vs 0 (`d5851b1`, rlib = stored outcome). BASE alone binds on no seed. F's other S81 statements are F's and are not checked here. |
| 207 | "13 of 282 and, on R's inference," (in H's probe paragraph) | — | session R's numbers; not checked here |
| 271 | `fabf505`, `fd2fa89`: five changed clauses | row5 ×5 | MATCH 3, DISAGREE 1, PARTIAL 1 |
| 343 | `e92e1c6`: stale clause replaced | OV5 | MATCH |
| 441 | `fabf505` (G ×2), `e92e1c6` (H's rate) | row5 ×2, OV5, plus a new check of the unchanged 50-run count | PARTIAL, MATCH, MATCH, PARTIAL |
| 456 | session I's S75: "session~G's register gate at all three layers, ten seeds each" | row5 | MATCH for the name only: `explore_gpu/s75_s4.py`:23 at `03bffdf` says "REG3 Session G's register gate at all three layers + SLOW". S75's results are I's. |
| 462 | "The hinge reads key identities; ... at four sources on the one GPU screen ... neither learned gate routed" | — | S75's REG3 result is session I's record; not checked here |
| 463 | `e92e1c6`: stale clause replaced | OV5 | MATCH |
| 476 | "`test_delta_recipe`'s L half (X's is complete), then the hinged-register test ..." | G's parenthesis carried (MATCH) | the order is the draft's plan (as above) |
| 513, 515 | unchanged (H and G provenance) | carried, plus a new check of H's pin at the draft | MATCH: `9b72bf6` and `076005d` were the heads at 19:14:19 |

Nothing else in the diff touches G or H.

## 6. H's report 8 (S86) in tex5: coverage

- **Every tex5 line that names S86 or the all-position pair has a row in `check_H_v5`:** tex5:41, 60, 205 (×8 rows plus the three-screen sum), 343, 441 (×5), 463 and 513 (×2). `check_H_v5.md` lists no other line ("other lines naming S86 ...: []").
- Report 8's four readings are as the docstring fixed them at `50da719`, and all are recomputed. The draft quotes "keeps it" and "aims"; "no cost by the rule" and "the evaluation is the better probe" are not named, which is harmless.
- **Not covered, because the draft does not use them:**
  - report 8's "The flat start appears again" (report_8.md:60: 5–6 of ~20 splits per arm at accuracy 0.03–0.19);
  - `report_8_notes.md`.
- **H's new S87a** (`f4fd535`: ALLMASS vs BASE on 20 more seeds) postdates the draft. When it reports, it will bear on tex5:205, 343, 441 and 463.

## 7. Errors in the referee's own earlier files

The proposed wording "while the terms are on" is wrong for POS.
- **Where it was written:**
  - `fourth_pass.md`:153: "It is true while the terms are on: the KL is ≤ 1.4e-5 at 300–1200 in every run";
  - `check_G_v4.md` row tex4:269 ("Exact: 'both hold the gate at uniform while they are on, ...'");
  - `resolution_G4.md` row 269 ("True while the terms hold" and the same "Exact") and §3 item 2 ("It holds only while the terms are on: through 1200, and through 2400 for FLAT").
- **What the records show.** Both terms are on through update 2400. POS's KL at 2400 is above 1e-5 in 13 of 20 runs (max 0.068), which `resolution_G4.md` itself records in the same row.
- **Effect.** The draft adopted the wording as "while they act", and that is this version's NEW ERROR.
- The numbers in those files (≤ 1.4e-5 at 300–1200; 0.098–0.112 at 1200) are right.

`fourth_pass.md` §2a also missed report 6's "0.0000 from update 300" (§3 above). That is an omission, not a false statement.

## What could not be checked

- **"Every session's queue is empty"** (tex5:41): no record holds a queue.
- **S83 and S86 files read, not rerun:** S83's bit-for-bit phase-0 reproduction and its 61 CHECKs, S86's CHECKs, and the run code SHAs (24268c2446f8, 898867f69ad4).
- **Seed freshness.** That seeds 2300–2339 are fresh rests on the docstrings' scans. G's seed scan for 600–619 / 1600–1619 covers the branch heads fetched at run time.
- **Other sessions' parts of shared lines:**
  - F's S81 beyond the two arms summed at tex5:205;
  - R's probe counts at tex5:207;
  - I's S75 results at tex5:456–462;
  - the main line's `test_split_copy` L rows (tex5:193–201, checked in `check_split_copy_L`).
- **Push times.** Commit times are committer-set.
- **The PDF.**
- **The draft's sixth commit `cc379b2`** (22:52:37), and the G and H commits after the draft (§ What is compared).
