# Revision 9, fourth draft: the Session H items of the third pass, and S86

*Referee V, fourth pass, 11 October 2026.*

**What was compared.**
- Third draft: `a5f67a2`, cited `tex3:NNN` (561 lines).
- Fourth draft: `c66cc76` on `origin/claude/rev9-draft` (07:52:07 UTC), cited `tex4:NNN` (564 lines). Line numbers shift by 0 to 3: `tex3:233` became `tex4:233-235`, and one line was inserted after `tex3:304`.
- The Session H changes are in `a469c1f` (the third-pass items), `73021d3` (S86 folded in) and `c66cc76` (S86 in the index, the running list and H's provenance row).

**Evidence.**
- `rev9/check_H_v4.py` and `rev9/check_H_v4.md` are this pass's run.
  - The script is a copy of `check_H_v3.py`. Every first-, second- and third-pass recomputation and anchor still runs and is asserted.
  - Each third-pass row is then found in tex4 by text search on its third-draft fragment. A rewritten sentence must be re-anchored on its new wording (OV4), or the script stops.
  - S86 is recomputed from `explore_out/H/allmass_results.json` at `9b72bf6`. The script uses `rlib` and the screen's own label rule.
- **Result:** 112 rows. DISAGREE 3, PARTIAL 2, NOT RECORD-CHECKABLE 1, MATCH 106. Three rows are REMOVED; they have been gone since the second draft.
  - The 95 third-pass rows: 8 re-anchored (all MATCH), 87 unchanged (86 MATCH, 1 not record-checkable). From third pass to fourth: PARTIAL → MATCH 4, MATCH → MATCH 90.
  - The 17 rows new in tex4 (15 on S86 sentences, 2 on sentences S86 makes stale): DISAGREE 3, PARTIAL 2, MATCH 12.
- H's branch head is `9b72bf6` (07:49:11, three minutes before the draft). Nothing was committed after it. Every H store the draft cites is byte-identical there.

**Statuses.**
- **RESOLVED**: tex4 now agrees with the records.
- **PARTLY**: part of the item is fixed.
- **NOT**: unchanged.
- **NEW ERROR**: an error in text that this version changed, or that this version's own changes made false.

## Tally

| scope | items | RESOLVED | PARTLY | NOT | NEW ERROR |
|---|---|---|---|---|---|
| resolution_H3.md, items not RESOLVED (tex3:384 PARTLY; tex3:342 PARTLY; tex3:228 2317 account NOT) | 3 | 3 | 0 | 0 | 0 |
| resolution_H3.md §6 found (F1 "routing formed first"; F2 tex3:506 provenance; F3 S86 not mentioned) | 3 | 3 | 0 | 0 | 2 (stale, from F3) |
| third_pass.md §1 item 2 (= F1 + the 2317 account) and §2 H items (tex3:342 with tex3:209 and tex3:259; 384; 506; 436) | (same items) | — | — | — | — |
| S86 as folded in (new sentences at tex4:41, 60, 203, 439, 509) | 15 rows | 12 MATCH | 2 PARTIAL | — | 1 DISAGREE |

**Deduplicated.**
- **6 distinct open H items**, all **RESOLVED**: tex4:387, 345, 228 (×2), 509 and 203/439.
- **3 NEW ERRORS**:
  - tex4:203: "copied onto an occupied channel" (new text);
  - tex4:341 and tex4:459: "not yet screened" and "the next screen". These sentences are unchanged, but this version's own S86 additions made them false.
- **2 new PARTIALs**, both minor: tex4:439, "at the same rate"; tex4:203, "all body positions".

## 1. The open items, one by one

| item (source) | tex4 now | evidence | status |
|---|---|---|---|
| tex3:384, "the onsets move later" (resolution_H3 §4, PARTLY; third_pass §2) | tex4:387: "(it has four key failures of its own and $14/20$ against $15/20$; its three onsets after update 0 come later than the baseline's seven, on two shared seeds)" | SLOWGATE's KEY failures under v2 are 2310, 2321, 2323 and 2328 (four). DISCOVERED is 14/20 against 15/20. Onsets after update 0: SLOWGATE 275, 275, 575 (three); REF 25, 25, 25, 50, 50, 125, 125 (seven). Every slow-gate onset is later than every baseline onset. The seeds with an onset in both arms are 2310 and 2323. | **RESOLVED** |
| tex3:342, "Each is marked ... those are now marked" (resolution_H3 §4, PARTLY; third_pass §2) | tex4:345 is unchanged, and is now true of every place the passes named | (a) tex4:209: "and, **our recommendation**, the trigger should not fire when its target already carries key mass". (b) tex4:228: the 2317 account (next row). (c) tex4:261, S55's three statements: "(where, **on our reading**, the decayed sum of the block still carries the header)"; "(**our reading:** the gate reads only the window, so nothing asks the stack to keep it)"; "... **is our reading of it**". All are asserted in check_H_v4. | **RESOLVED** |
| tex3:228, the account of seed 2317 (resolution_H3 §4, NOT; third_pass §1 item 2) | tex4:228: "the one failure (seed 2317) a run that each device alone discovered: under both together the gate was still unrouted at 2400 and the split formed after the window ($\eta^2$ by key $0.30$ at 3600, $0.77$ at 4800) without recovering, which session~H reads as the slow gate pushing the gate's learning past the hinge's window, though its hold ends at 1200 and the record shows only the combination failing" | `two_b_results.json` @ `1e09ecd`: SLOWGATE\|2317 and KEYHINGE\|2317 both DISCOVERED. SLOWGATE_KEYHINGE\|2317 has routing margin 0.000 at 2400, η² by key ≤ 0.200 up to 2400, 0.30 at 3600 and 0.77 at 4800. It ends KEY and is not discovered. The hold covers updates 1–1200 (`explore_h_two_b.py` docstring). The account is now marked as H's, and qualified. | **RESOLVED** |
| F1: tex3:228, "the stream routing having formed first" (third_pass §1 item 2) | tex4:228: "every such run recovers, routing by stream appearing at 4800 or 6000, after the rise" | In the seven KEYHINGE runs with η² by key ≥ 0.2 at 3600, the routing margin at 2400 is 0.00 in all seven. The first dense cell with margin ≥ 0.25 is 4800 on 2317, 2319, 2321, 2323 and 2325, and 6000 on 2313 and 2318. All seven are DISCOVERED. | **RESOLVED** |
| F2: tex3:506, H's provenance (third_pass §2) | tex4:509: "S85 prevention at two streams (report 7, `1e09ecd`, corrected `de11b2a`); S86 all-position targeting (report 8, `9b72bf6`) & `9b72bf6` & 318; 300; 82; 62; 52; 82; 32" | Commits touching `report_7.md`: `de11b2a`, `1e09ecd`. `report_8.md`: `9b72bf6`. The head is `9b72bf6`, with no commit after it before the draft. The run counts are as before, plus S86's 32 records (BASE, ALLMASS and EVAL with 10 each, ORACLE 2). | **RESOLVED** |
| F3: tex3:436, S86 not mentioned (third_pass §2) | tex4:203 (paragraph), 439 (limits), 60 (index), 41 (running list: "H's S86 ... complete"), 509 (provenance) | §2 below. But two sentences the version did not touch still describe the screen as not yet done (§3). | **RESOLVED**, with 2 NEW ERRORS (stale) |

**Residual remarks from earlier passes.** These were not counted, and they are unchanged in tex4:
- tex4:207 still calls S63's temperature anneal "slope annealing".
- S58b's reading name "does not carry" is not quoted.
- tex4:209's "ended in a key split" counts v1's three seeds.
- tex4:226's eight-stream near-tie is S24's measurement, not an H record.

## 2. S86 (`explore_h_allmass`), against its records

**Files and timing.**
- **Screen.** `explore_h_allmass.py` (parent) and `explore_h_allmass_child.py` (child) were both first committed at `50da719` (02:56:33 UTC), "code and pre-fixed readings".
- **Preliminary count.** `explore_h_allmass_counts.py` and `explore_out/H/allmass_counts.txt` were committed at `e0140ad` (02:53:36).
- **Store.** `explore_out/H/allmass_results.json` has 32 records at `9b72bf6`, which is the head.
- **Log, report and notes.** `explore_out/H/s86.log`, `report_8.md` and `report_8_notes.md`.
- **Order.**
  - The CHECKs passed at 03:05:09 on `50da719`.
  - The store's provenance reads "git 50da719, started 02:56:36".
  - The first commit whose store holds a run is `71fc5f2` (03:26:35, one run). The readings therefore precede the first record.
  - The parent was touched by two commits, `50da719` and `9b72bf6`. Its docstring, readings included, is byte-identical in both.

**The "report code fix" in `9b72bf6`.**
- **What it changes.** One line of `report()` (l.171) changes from `dict(e84, **B["EVAL"])` to `{**e84, **B["EVAL"]}`, and likewise for BASE.
- **Why.** The old form raised `TypeError: keywords must be strings` (integer seed keys) after all 32 runs were saved. The traceback is at the end of `s86.log`.
- **Effect on the result.** None:
  - The new form builds the same merged map: the seeds are disjoint, and both merge orders give the same map (checked).
  - The line feeds only the descriptive pooling. No reading's input or count depends on it.
  - The run code SHA `898867f69ad4` covers 49 modules, which include `explore_h_allmass_child` and exclude the parent. So no run's SHA changed, as report 8 says.
- **Conclusion.** It changed no reading and no count.

**Configuration.** Every arm's `trig_cfg` agrees with the docstring:
- every 2400, first check 4800, reference 2400, cap 3, gap 4800, rise 0.02, threshold 0.95, probe 64, no guard;
- BASE: target key, source probe; ALLMASS: target all, source probe; EVAL: target key, source eval;
- last check 40800.

**Validity.** The ORACLE bound 2330 at 6000 and 2331 at 3600, so the screen is **VALID**.

| arm | BOUND ROUTED (rlib = recorded) | vs BASE (arm only, BASE only) | failures (v1 = v2) | firings | own pair labelled | key pair labelled | all-position pair labelled | same pair | both labelled |
|---|---|---|---|---|---|---|---|---|---|
| BASE | 9/10 | — | MERGED (2 share) 1 (2331) | 20 | 19 | 19 | 17 | 8 | 17 |
| ALLMASS | 8/10 | 0, 1 (p = 0.50) | MERGED (2 share) 2 (2331, 2334) | 21 | 20 | 21 | 20 | 8 | 20 |
| EVAL | 9/10 | 0, 0 (p = 1) | MERGED (2 share) 1 (2331) | 20 | 19 | 19 | 16 | 8 | 16 |

- Every label was recomputed from `map_before` with the screen's rule: c\* holds ≥ 2 streams and c0 holds none. Each equals the stored `key_ok`, `all_ok` and `op_ok`.
- Transitions recompute, and v2 recomputes.
- Pooled with S84 (`guard_results.json` @ `37c1fbf`: EVAL 10/10, BASE 9/10), EVAL is 19/20 against BASE's 18/20: EVAL only 1, BASE only 0, one-sided p = 0.50.

**The readings, each under its rule (docstring at `50da719`):**

| reading | rule | recomputed | applies |
|---|---|---|---|
| all-position targeting keeps it | BASE only − ALLMASS only ≤ 1 | 1 − 0 = 1 | yes |
| aims | ≥ 90% of ALLMASS's firings labelled ok on its own pair | 20/21 = 0.952 | yes |
| costs binding (ALLMASS) | BASE only ≥ 4 and ALLMASS only = 0 | 1, 0 | no ("no cost by the rule") |
| the evaluation is the better probe | BASE only − EVAL only ≤ 1 | 0 − 0 = 0 | yes |

These match `report_8.md` word for word. They also match the draft commit's summary in `73021d3`: "keeps it, 8/10 vs 9/10, 0 vs 1; aims 20/21; the one mistarget cost the run; EVAL 9/10 vs 9/10, pooled 19/20 vs 18/20".

**The one mistarget: seed 2334, update 19200.**
- The probe read 0.797.
- **The two pairs.** The key-position pair was 7→14: channel 7 holds streams 1 and 5, and channel 14 holds none, so it is labelled. The all-position pair was 1→6: channel 1 holds **no** stream and channel 6 holds **no** stream, so it is not labelled.
- **The map.** Before the split: [10, 7, 15, 0, 4, 7, 3, 9]. After it: the same.
- **The paired runs.** ALLMASS\|2334 and BASE\|2334 have identical curves through 19200, and first differ at 20400.
- **After the split.** ALLMASS fired twice more on channel 7 (7→13 at 24000, 7→2 at 28800, both labelled), and ended MERGED (2 share). BASE bound routed at 28800, after one off-target split of its own at 26400 (key pair 0→6, where channel 0 holds one stream). EVAL did the same as BASE.

### The S86 sentences of tex4

| tex4 | sentence | status | evidence |
|---|---|---|---|
| 203 | "Session~H then screened what the trigger reads (S86, seeds 2330--2339; the oracle $2/2$)" | MATCH | seeds and ORACLE as above; freshness not rescanned |
| 203 | "chosen by the mean read-gate mass over all body positions, which reads no task structure" | **PARTIAL** (minor) | `explore_split_target_child.select` (l.63-68, the same as `test_window_gate.select` l.463-468) averages the read gate over every position of `data[:, :-1]`. On TASK48 (S = 8, P = 4, n_q = 1, length 99) that is the 96 body positions plus the query's stream and key: 98 positions. Say "all positions" or "every input position". "Reads no task structure" holds. |
| 203 | "binds routed $8/10$ against the key-position rule's $9/10$ (0 vs 1; ``keeps it'')" | MATCH | 8/10 against 9/10; ALLMASS only 0, BASE only 1; "keeps it" applies |
| 203 | "with 20 of its 21 firings on labelled targets (``aims'')" | MATCH | 20/21 = 0.952 ≥ 0.9. Not stated, and not needed: BASE's own key pair was labelled at 19/20 and EVAL's at 19/20 (both miss at 2334, 26400). |
| 203 | "the one discordant seed is the one mistarget, at a check where the two rules chose different pairs and the all-position pair copied onto an occupied channel" | **DISAGREE** (NEW ERROR) | The seed and the differing pairs are right: 7→14 against 1→6. But the target channel 6 held no stream, so it was not occupied. The pair is unlabelled because its **source**, channel 1, held no stream either. It copied an unused channel onto another unused one, and left the shared channel 7 (streams 1 and 5) as it was; the map after the split equals the map before. Fix: "the all-position pair copied a channel that held no stream, leaving the shared one". Report 8 says only that the pair "was not" labelled ok. |
| 203 | "after which the run merged while the key-position rule's run bound" | MATCH | MERGED (2 share) against BOUND ROUTED at 28800. The runs are identical to 19200. |
| 203 | "The two rules chose the same pair at only 8 of about 20 firings per arm, and both pairs were labelled targets at most of them, as in the main line's records" | MATCH | Same pair 8/20, 8/21, 8/20. Both labelled 17/20, 20/21, 16/20. Main line: same 46/107, both 97/107. |
| 203 | "The evaluation as the trigger's probe binds $9/10$ against $9/10$ here (0 vs 0), and $19/20$ against $18/20$ pooled with S84 (1 vs 0, descriptive)" | MATCH | As above. The applying reading "the evaluation is the better probe" is not named for S86 (it is for S84); harmless. |
| 60 | "a trigger reading all positions keeps binding within one pair" | MATCH | "keeps it", 1 − 0 |
| 41 | "H's S61--S86"; "G's S83, H's S86 and R's Phase~4 are complete" | MATCH (H's half) | 30/30 learned runs and 2/2 oracle runs at `9b72bf6`, 07:49:11, before `c66cc76` at 07:52:07 |
| 439 | "a trigger that reads all positions was screened once (session~H's S86): it keeps binding within one pair of the key-position rule ($8/10$ against $9/10$) and its one mistarget in 21 firings cost that run" | MATCH | "Cost that run" is report 8's wording. The paired runs share every update to 19200, so the copy there is the first difference between a bound run and a merged one. That makes it a paired fact, not only an interpretation. |
| 439 | the main line's count: "at 97 (21 of 22, 46 of 47 and 30 of 38), the key-position pair at all 107; the two pairs coincide at 46" | MATCH | Recomputed again from window_gate X, window_gate L (main line) and split_copy X (`592c574`): 50 runs, 107 firings, all-position 97 (21, 46, 30), key 107, same 46 (the same as `rev9/count_allpos.py` and H's `allmass_counts.txt`) |
| 439 | "and S86 then found 1 mistarget in 21 at the same rate" | **PARTIAL** | The main line's all-position rate is 10/107 = 0.093. S86's is 1/21 = 0.048, half that. So it is not "the same rate". It is consistent with the main line's rate: P(≤ 1 of 21 at 0.093) = 0.40. Report 8 says "predict about 1 mistarget in 10 firings. Here there was 1 in 21". Fix: "at a rate consistent with it (1 in 21 against 10 in 107)". The sentence also runs two "and" clauses together. |
| 439 | "A split that reads no task structure is therefore possible at a small cost in aim" | MATCH (note) | The cost in aim appears in the counterfactual labels: on the main line 97 against 107; over all S86 firings, the all-position pair at 53/61 against the key pair at 59/61. It does **not** appear in the S86 arms' own aim, 20/21 against the key rule's 19/20. This is an unmarked synthesis of one ten-seed screen; the paragraph does name it as a screen. |
| 509 | "S86 all-position targeting (report 8, `9b72bf6`) & `9b72bf6` & ...; 32" | MATCH | report_8 at `9b72bf6`; 32 records. The readings commit `50da719` is not cited, as with the rows for the earlier reports. |

## 3. NEW ERRORS: sentences S86 makes false (unchanged text)

| tex4 | sentence | why it is now false |
|---|---|---|
| 341 (= tex3:338) | the referee's requests: "the split's dependence on key positions (stated everywhere it is claimed; **a trigger that reads all positions is not yet screened**)" | S86 screened it, and the draft reports it at tex4:203, 439 and 60. Replace with "screened once, session H's S86, $8/10$ against $9/10$". |
| 459 (= tex3:456) | limitations list: "The key positions come from the task; **a trigger on all positions is the next screen.**" | The same. It contradicts tex4:439's "was screened once". |

## What could not be checked

- That seeds 2330–2339 (and 2300–2329) are fresh. This rests on the docstrings' scans and was not rescanned.
- Whether the S86 runs used exactly the committed code. Many record `+dirty` trees while the progress commits were being made (`s86.log`). The single run code SHA `898867f69ad4` is the only evidence.
- G's S83 clause in tex4:439 ("a term reading positions only, or nothing, fails ($1/20$, $0/20$)"), and the S81, S82 and S83 material elsewhere. These are not H records.
- The PDF, page by page.
- Commit times are set by the committer, so push order cannot be established from git.
