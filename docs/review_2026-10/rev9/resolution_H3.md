# Revision 9, third draft: how the Session H findings of the second pass were resolved

*Referee V, third pass, 11 October 2026.*

**What was compared.**
- Second draft: `f67bc07`, cited `tex2:NNN`.
- Third draft: `a5f67a2` on `origin/claude/rev9-draft`, cited `tex3:NNN`. The Session H changes are in `4950cdb`; `ca2bc99`, `a1ee526` and `a5f67a2` touch other sessions' sentences on the same lines. `git diff f67bc07 a5f67a2 -- multichannel_hebbian_report_v9.tex` changes 40 lines (80 diff lines). None of them adds or removes a line, so line numbers are the same in both versions.

**Evidence.**
- `rev9/check_H_v3.py` and `rev9/check_H_v3.md` are the third-pass run. The script is a copy of `check_H_v2.py`, so every v2 assertion and recomputation still runs. It then finds every row in tex3 by text search. Any row whose sentence was rewritten has to be re-anchored on the new wording, and its number is checked again against the record. If a rewritten row has no re-anchor, the script stops.
- **Result:** 95 rows. DISAGREE 0, PARTIAL 4, NOT RECORD-CHECKABLE 1, MATCH 90. Three rows are REMOVED, all of them already gone in tex2.
- The records were read with `git show` on `origin/claude/explore-H` at the commits the draft names: report 6 `37c1fbf` (S84), report 7 `1e09ecd` (S85), and the earlier reports as before.
- H's branch head is now `dba7cd7`, five commits after `1e09ecd` (see F2). Every H store the draft cites is byte-identical at the new head.

**Statuses.**
- **RESOLVED**: tex3 now agrees with the records.
- **PARTLY**: part of the item is fixed.
- **NOT**: unchanged.
- **NEW ERROR**: an error in text changed in tex3.
- **FOUND**: an error in unchanged text that this pass could check for the first time.

## Tally

| scope | items | RESOLVED | PARTLY | NOT | NEW ERROR |
|---|---|---|---|---|---|
| second_pass.md §1 item 8 (tex2:228, three numbers) | 3 | 3 | 0 | 0 | 0 |
| second_pass.md §2, Session H rows (tex2:207, 203, 440, 436/465/228) | 4 | 4 | 0 | 0 | 0 |
| second_pass.md §3, Session H (tex2:60, 228 ×3, S84's readings) | 5 | 5 | 0 | 0 | 0 |
| resolution_H.md, items not RESOLVED (tex2:207 NOT; tex2:342 PARTLY) | 2 | 1 | 1 | 0 | 0 |
| resolution_H.md §6 / check_H_v2.md new material: DISAGREE rows | 3 | 3 (= §1 item 8) | 0 | 0 | 0 |
| resolution_H.md §6 / check_H_v2.md new material: PARTIAL rows | 9 | 8 | 1 (tex3:384) | 0 | 0 |
| check_H_v2.md: S84 readings-vs-docstring PARTIAL; unmarked descriptions marked NEW (S84 ×2, S85 2317, tex2:440); hinge.fired side note | 6 | 5 | 0 | 1 (the 2317 account) | 0 |
| other changed lines about Session H (tex3:36, 316, 384, 436, 455, 465, 470) | 7 | — | — | — | 0 |
| found in this pass (unchanged text / branch moved) | 3 | — | — | — | — (1 FOUND, 1 provenance PARTLY, 1 note) |

**Deduplicated.** The rows above overlap: §1 item 8 is the same as v2's three DISAGREE rows, and §2 and §3 repeat v2's PARTIAL rows. Counted once each, there are **18 distinct second-pass items**:
- **15 RESOLVED**;
- **2 PARTLY**: tex3:384's "the onsets move later", and tex3:342's "Each is marked";
- **1 NOT**: tex3:228's account of seed 2317, still unmarked;
- **0 NEW ERROR**: no changed sentence about Session H introduces an error.

**This pass found two more**, plus one note:
- **FOUND (F1).** tex3:228's "the stream routing having formed first" is not borne out by the dense cells. The text is unchanged, and no earlier pass could check it.
- **PARTLY (F2).** tex3:506 cites report 7 at `1e09ecd`, but report 7 was corrected at `de11b2a`, and the draft's own S85 sentence follows that correction. This is provenance only.
- **Note (F3).** S86 is not mentioned (tex3:436).

## 1. second_pass.md §1 item 8: tex2:228's three numbers (= check_H_v2 DISAGREE rows)

| item | tex3:228 now | record (two_b_results.json @ 1e09ecd) | status |
|---|---|---|---|
| "η² by key falls to 0.00–0.12 by update 25 and stays below 0.07 to 2400" | "under the key term $\eta^2$ by key stays below the term's threshold of $0.2$ at every measurement from update 25 to 2400 (maximum $0.195$ over the 20 runs; the term fired on 1--11 batches per run)" | Each KEYHINGE run's maximum over the dense cells at updates 25–2400 is 0.005–**0.195**; the 0.195 is seed 2327 at update 25. The hinge's τ is 0.2 (`explore_h_two_b.py` l.17, `relu(eta2_key - 0.2)`; recorded `hinge.tau` = 0.2 in all 20 runs). `hinge.fired` is **1–11**. It counts the training forward passes in updates 1–2400 whose own batch η² exceeded τ (`attach_key_hinge`, `explore_keysplits_child.py` l.174–200 at `1e09ecd`), so it is consistent with every probe measurement staying below 0.2. Report 7 was corrected to the same words at `de11b2a`. | **RESOLVED** |
| "0.21–0.64 at 3600 on some seeds" | "a split can start after the window ($0.21$--$0.88$ at 3600 on seven seeds) and every such run recovers" | At 3600 the KEYHINGE runs at or above 0.2 are 2313 0.21, 2317 0.64, 2318 0.30, 2319 0.35, 2321 **0.88**, 2323 0.25 and 2325 0.22. That is **seven** runs; the next value below is 0.136. All 20 KEYHINGE runs discovered. The corrected report 7 still says "0.21-0.64 ... on some seeds"; the draft follows the records. | **RESOLVED** |
| "organised by key on six of the twenty seeds" | "organised by key on five of the twenty seeds ($\eta^2$ up to $0.84$ at update 0)" | Above 0.5 at update 0 (identical in the four arms): 2313 0.70, 2318 0.64, 2320 0.84, 2326 0.56 and 2328 0.76, so **five** seeds. The next value is 0.476. REF's discovering seeds 2313 and 2320 are at 0.00 by 600. Report 7 (l.64) still says "> 0.5 on 6 seeds". | **RESOLVED** |

## 2. second_pass.md §2: unmarked descriptions (Session H rows)

| tex2 | item | tex3 now | status |
|---|---|---|---|
| 207 | S58: "it changed the routing's shape, not its count" (also resolution_H.md's only NOT) | tex3:207: "the routing margin fell from $0.87$ to $0.46$; **on our reading** it changed the routing's shape, not its count, and it was not carried to eight streams". This recomputes: SWITCH 36/40 against NONE 38/40 (1 vs 3), with the key- and value-position maps differing in 34 of the 36. | **RESOLVED** |
| 203 | S84: "the probe's noise decided whether the first check fired"; "the base kept enough splits for its merges" | tex3:203: "so, **on our reading**, the probe's noise decided whether the first check fired"; "since, **on our reading**, the base kept enough splits for its merges after spending some". The record behind the first is the seed-2306 rises at 4800. The probe went 0.000 → 0.047, a rise of at least 0.02, so it did not fire. The evaluation went 0.066 → 0.068, a rise below 0.02, so it fired. | **RESOLVED** |
| 440 | "the one repair device that works at eight streams cannot work here" | tex3:440: "did not work here (S78, ``neither device'')". This recomputes: S78's exact copy is 1 vs 0 and the noisy copy with reset 2 vs 0, so R4 reads "neither device". | **RESOLVED** |
| 436, 465, 228 | No screen marker; "nothing repairs"; "found on every seed" | tex3:436: "which **on screens** prevents the key split at two streams and for the register at a distance". tex3:465: "is found on every seed **of a screen** ... which **on a screen** also prevents the two-stream key split that **the copy did not repair**". tex3:228: "prevents the split that **the copy did not repair (S78)**". The records: S85 KEYHINGE 20/20 against 15/20 (5 vs 0, p = 0.031); S78 "neither device"; the same key split back after 8 of 8 firings on the KEY-ending seeds. | **RESOLVED** |

## 3. second_pass.md §3: wording fixes (Session H)

| tex2 | item | tex3 now | record | status |
|---|---|---|---|---|
| 60 | "the flat-start guard aims the splits": the criterion cannot fail at the flat start | tex3:60: "the flat-start guard moves the splits to higher probe accuracies without changing the count" | GUARD's 17 splits fire at a probe accuracy of 0.500–0.906 (median 0.797). BASE's 24 fire at 0.062–0.875 (median 0.617; 10 below 0.5, 7 below 0.2). Both bind 9/10, 0 vs 0. The pre-fixed name "the guard aims the split" now appears nowhere in the draft; it applies by its rule, so this is harmless. | **RESOLVED** |
| 228 | "with the same four key failures" | tex3:228: "the slow gate discovers $14/20$ (0 vs 1) with four key failures of its own, three on the baseline's seeds" | Under v2 (the paragraph's "key-first classifier"), SLOWGATE's KEY seeds are 2310, 2321, 2323 and 2328, and REF's are 2310, 2318, 2323 and 2328. Three are shared. Under v1 SLOWGATE has three; 2323 is KEY only under v2, at η² = 0.5000005. "Prevention not shown" (0 vs 1) and "only moves" (4 ≥ 3) recompute. | **RESOLVED** |
| 228 | "at every 25 updates" holds to 1200 only | tex3:228: "every 25 updates to 1200 and every 100 to 2400" | Dense cells: 49 at a spacing of 25 over 0–1200, 13 at 100 over 1200–2400, then every 1200 (`explore_h_two_b.py` l.28–30) | **RESOLVED** |
| 228 | onsets "25–125 → 275–575" read as the same splits | tex3:228: "the baseline's onsets after update 0 lie at updates 25--125 (seven seeds) and the slow gate's at 275--575 (three seeds, two of them seeds the baseline also split and one it discovered), so, on our reading, the gate's learning rate is not what lets them form" | REF's onsets: 2310 125, 2311 25, 2315 25, 2316 50, 2319 50, 2322 25, 2323 125. SLOWGATE's: 2310 275, 2321 275, 2323 575. The seeds in both are 2310 and 2323, both REF KEY failures. 2321 is new, and REF discovered it. The inference is marked. | **RESOLVED** (tex3:384, see §4) |
| — | S84: EVAL_GUARD_INT1200 and "costs binding" not stated | tex3:203: "the guard with fast checks and the evaluation together bind $9/10$ (0 vs 0), and no arm read ``costs binding''" | EVAL_GUARD_INT1200 (source eval, guard 0.5, every 1200) bound routed 9/10, 0 vs 0. "Costs binding" (BASE only ≥ 4 and the arm only = 0) is False for GUARD, GUARD_INT1200, EVAL and EVAL_GUARD_INT1200. The median transition, 15600, is not given; nothing requires it. | **RESOLVED** |

## 4. resolution_H.md items not RESOLVED, and check_H_v2.md's new-material rows

| source | item | tex3 now | status |
|---|---|---|---|
| resolution_H §4/§5 (NOT) | tex2:207 "shape, not count" | §2 above | **RESOLVED** |
| resolution_H §4 (PARTLY) | tex2:342 "Each is marked where it appears ... those are now marked" | tex3:342 is unchanged. The places resolution_H named (S58 at tex3:207; the S84 interpretations at tex3:203) are now marked. Still unmarked: (a) tex3:228's account of 2317 (next row but one). (b) tex3:209's recommendation "the trigger should not fire when its target already carries key mass", which resolution_H §4 noted but did not count. "Each is marked" is therefore still not quite true; the one substantive case is (a). | **PARTLY** |
| check_H_v2 PARTIAL, tex2:384 | "it moves it from update 25--125 to 275--575 and prevents nothing" | tex3:384: "would prevent the two-stream key split (the onsets move later and the splits still form; $14/20$ against $15/20$)" | **PARTLY** (minor) |
| | | As a statement about the two sets it holds: all three slow-gate onsets (275, 275, 575) come after all seven baseline onsets (25–125). The splits do still form (four KEY failures). "Move" still suggests the same splits shifted, which is true on 2 seeds (2310: 125 → 275; 2323: 125 → 575). Five of the baseline's seven onsets (25–50, on seeds both arms discovered) have no counterpart. tex3:228 gives the overlap and this summary does not. A fix: "the onsets that occur come later". | |
| check_H_v2 DESC (NEW, not marked); resolution_H §6 | tex2:228 "the one failure a seed on which the slow gate held the gate back until the hinge's window had passed and the split then formed" | tex3:228, unchanged and unmarked. The record does not support the account as worded. SLOWGATE_KEYHINGE\|2317: η² by key 0.20 at update 0, 0.06–0.12 over 25–575, ≤ 0.05 from 600 to 2400, 0.30 at 3600, 0.77 at 4800; routing margin 0.00 at 1800 and 2400. But the slow gate's hold ends at update 1200, inside the window, and **the slow gate alone routed 2317 by 1800** (margin 0.99). Under the key term alone, 2317 was **also** unrouted at 2400 (margin 0.00) and **also** split after the window (η² by key 0.64 at 3600), yet recovered. What the record shows is the combination failing on 2317, not the slow gate holding the gate past the window. It should be marked as the session's reading, or replaced by the numbers. | **NOT** |
| check_H_v2 READ, S84 (PARTIAL) | "aims"; the fifth arm and "costs binding" missing | §3 above | **RESOLVED** |
| check_H_v2 DESC, tex2:228 (marked, three numbers wrong) | the "Descriptively" sentence | §1 above. Its remaining clause is F1 below. | **RESOLVED** (numbers) |
| resolution_H §6 side note | `hinge.fired` 1–11 against report 7's "1–37 batches"; unit not traced | tex3:228: "the term fired on 1--11 batches per run". The unit is now traced (§1 row 1): training batches with batch η² by key > 0.2 in updates 1–2400. Report 7 was corrected to "1-11 training batches" at `de11b2a`. | **RESOLVED** |

The other v2 PARTIAL rows (tex2:60, 228 ×4, 436, 440, 465) are covered in §2 and §3: all RESOLVED.

**Residual remarks from rows that resolution_H counted as RESOLVED.** None of these is counted in the tally, and none changed in tex3:
- tex3:207 still calls S63's temperature anneal (1 → 0.25) "slope annealing".
- S58b's reading name "does not carry" is not quoted.
- tex3:209's "ended in a key split" is v1's three seeds (1317 is KEY under v2).
- tex3:226's eight-stream near-tie is S24's measurement, not an H record.

## 5. Other changed lines that concern Session H

| tex3 | change | status |
|---|---|---|
| 36 | The changed clauses ("key splits, formed in the records before any write at the cue", "found $20/20$ on fresh seeds", "as the split reads key positions") are session G's. H's clause, "Two-stream key splits resist repair and are prevented by the hinge's key term on a screen ($20/20$ against $15/20$)", is unchanged. | MATCH (check row) |
| 316 | The change ("a key split that in the records forms before the register's write") is G's half. H's half, "the hinge's key term prevents on a screen (S85, $20/20$ against $15/20$)", is unchanged. | MATCH |
| 384 | S85's half is §4 above (PARTLY). The slow gate against the register ("it binds $0/20$; on our reading it slows the write too") is G's S82, not checked here. | see §4 |
| 436 | "on screens" was added (§2). The all-position count 97 of 107 is unchanged; it agrees with `rev9/count_allpos.py` and with H's own count (F3). | RESOLVED |
| 455 | Session G only (S72/S73, S82's 2/20) | not H |
| 465 | §2 | RESOLVED |
| 470 | H's item, "Session~H's proposed test of the key term at two streams", is unchanged. The change is R's "now committed". H's branch now also carries `spec_draft_two_keyterm.md` (`de11b2a`), the draft specification of that test. It is not cited; nothing requires it. | MATCH |

No changed sentence about Session H introduces an error (**NEW ERROR: 0**).

## 6. Found in this pass

**F1. tex3:228 (unchanged since tex2:228): "every such run recovers, the stream routing having formed first". FOUND (PARTIAL in check_H_v3).**

The dense cells carry η² by stream and the routing margin, so this clause can be checked now. The second pass listed it as uncheckable. Look at the seven KEYHINGE runs whose η² by key is ≥ 0.2 at 3600, at the last dense cell before that (2400):
- the routing margin is **0.00 on all seven** (|margin| ≤ 1e-5);
- the channel map is one-to-one on **2 of 7** (2318, 2319);
- accuracy is **0.24–0.27**;
- η² by stream at value positions is 0.000–0.339;
- η² by stream at key positions is 0.06–0.77, and ≥ 0.5 on only 4 of 7. It falls to 0.001–0.26 at 3600.

Routing with margin ≥ 0.25 first appears at **4800** (2317, 2319, 2321, 2323, 2325) or **6000** (2313, 2318). That is *after* the key rise at 3600, not before it. So "formed first" holds only in a weak sense, as partial stream structure at key positions on four seeds, and that structure itself dropped when the key split rose.

The clause comes from report 7 ("the stream routing had already formed"); the draft marks the sentence "Descriptively". Suggested text: "and every such run recovers by 4800–6000". If the draft keeps the mechanism, it should say "on our reading".

**F2. tex3:506: provenance of report 7 and H's branch commit. PARTLY (minor; check_H_v3 PARTIAL ×2).**
- tex3:506 still cites "report 7, `1e09ecd`" and the branch commit `1e09ecd`.
- `report_7.md` was rewritten at `de11b2a` (2026-10-11T03:07:17Z, eight minutes before `a5f67a2`). The rewrite is the correction of the key-term window sentence to "below the term's threshold 0.2 ... maximum 0.195 ... 1-11 training batches". tex3:228 now uses exactly that wording, and `ca2bc99`'s message cites "H's report-7 correction".
- H's head is now `dba7cd7`. The commits after `1e09ecd` are:
  - `e0140ad`, `50da719`, `d514b74` and `dba7cd7`: S86;
  - `de11b2a`: the report-7 correction and the spec draft.
- Every store the draft cites is byte-identical at `dba7cd7`, so no number is affected.
- A fix: cite report 7 as corrected at `de11b2a`, or say that the window sentence follows the records rather than report 7.

**F3. tex3:436: "a trigger that reads all positions has not been screened". Note, not an error.**
- S86 (`explore_h_allmass.py`, pre-fixed readings at `50da719`, 2026-10-11T02:56:33Z, before both `d425b71` and `a5f67a2`) is exactly that screen: ALLMASS against BASE and EVAL on seeds 2330–2339.
- It is running and has no result, so the sentence is true as of the draft. But the draft mentions S86 neither here nor in the next steps (tex3:470).
- H's preliminary count (`explore_out/H/allmass_counts.txt`, `e0140ad`) gives 107 firings: key pair labelled ok at 107, all-position pair at 97 (46/47, 30/38, 21/22), coinciding at 46. This equals tex3:436 and `rev9/count_allpos.py`.

## What could not be checked

- That seeds 2300–2329 are fresh. This is taken from the scan recorded at `explore_h_guard.py` l.9–12.
- Whether H's S85 CHECK, run on a dirty tree (`96e1e90+dirty`), used exactly the committed code.
- Records that are not Session H's:
  - session G's S82 (tex3:228's comparison, tex3:265, tex3:384's register half, tex3:455);
  - session F's S81 (tex3:203);
  - S24's near-tie (tex3:226).
- S86's progress records (`dba7cd7`). They are not cited by the draft and were not read beyond the count file.
