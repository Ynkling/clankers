# Revision 9 draft, third version (`a5f67a2`): third pass

*Referee V, 11 October 2026.*

**What was checked.** `multichannel_hebbian_report_v9.tex` / `.pdf` on `claude/rev9-draft` at `a5f67a2`, committed 03:15:04 UTC. `tex3:NNN` is a line of its tex. The second and third versions have the same 561 lines, and every change is a line replaced in place, so `tex2:N` and `tex3:N` are the same place.

**How.**
- The tex was diffed `f67bc07..a5f67a2`: 40 lines replaced, in five commits, one per scope of `second_pass.md`.
- Every changed line was checked against the record it cites.
- The v2 checks were re-run as v3 (`check_split_copy_v3`, `check_H_v3`, `check_F_v3`, `check_G_v3`, `check_appendixB_v3`). Each re-anchors the rewritten rows on the new wording, re-checks the number, and keeps every record recomputation.
- The item-by-item verdicts are in `resolution_split_copy_appB3.md`, `resolution_H3.md`, `resolution_F3.md`, `resolution_G3.md` and `resolution_V3.md`.

**Rules followed.** Nothing outside `docs/review_2026-10/` was changed, and no training was run. The Appendix B check built the Part C model and CEIL_C and drew the held-out batch, read-only.

## Verdict

**Not ready to merge.** It is close.

**What reproduces.**
- Every number this version changed reproduces from its record, with the single exception of the tex3:261 parenthesis (§1 item 3).
- Every count, pair, p-value and interval of a claim reproduces.
- Appendix B now has no false item.

**What remains.**
- **Three statements false against records already pushed** (§1). Two of them the draft took over from the referee's own second pass; that is said in each place.
- **A handful of unmarked or over-general sentences, and two omissions** (§2, §3).

All are statements, not numbers of a claim. One more version, checked against the branch heads at the time of its commit, should clear them.

| check against `a5f67a2` | rows | DISAGREE | PARTIAL | not record-checkable | MATCH |
|---|---|---|---|---|---|
| `test_split_copy` | 45 | 0 | 0 | 0 | 45 |
| provenance and main-line rows (`check_split_copy_v3` §4) | 11 | 3 | 5 | 0 | 3 |
| session H, with S84 and S85 | 95 | 0 | 4 | 1 | 90 |
| session F | 168 | 0 | 1 (report 6, outside the draft) | 6 | 161 |
| session G, with S82 | 141 | 1 | 6 | 1 | 133 |
| **total** | **460** | **4** | **16** | **8** | **432** |
| Appendix B against the code (atomic items) | 105 | 0 FALSE | 3 | 4 unverifiable | 98 TRUE |

Second pass → third:

| | DISAGREE | PARTIAL |
|---|---|---|
| second pass | 4 | 32 |
| third pass | 4 | 16 |

Three of the four remaining disagreements are the same R Phase 4 error, at three places (§1 item 1).

| resolution of the second pass's items | items | RESOLVED | PARTLY | NOT | NEW ERROR |
|---|---|---|---|---|---|
| split_copy, Appendix B and the main line | 42 | 26 | 7 | 6 (5 need no change: 3 claims about need or prescriptions, 2 bit-for-bit omissions) | 3 |
| session H | 18 | 15 | 2 | 1 | 0, plus 1 found in unchanged text |
| session F | 37 (with overlaps) | 34 | 1 | 2 (report 6 only, counted twice) | 0 |
| session G | 36 (with overlaps) | 28 | 4 | 0 | 4 (one finding, counted four times) |
| the referee's own rows | 16 | 10 (and 1 MATCH) | 4 | 0 | 1 (tex3:510, the referee's own) |

## 1. False against the records (must change)

**1. R's Phase 4 has run, in part.** The draft says otherwise at tex3:41 ("R's Phase 4 has not started"), tex3:325 ("no Phase 4 record exists") and tex3:510 ("Phase 4 rule (464be8d), not started", branch commit `464be8d`).
- **What R's branch holds.** Its head is `0364e28`, "Phase 4 interim records", 00:02:24 UTC on 11 October, three hours before this version. `records/phase4.jsonl` there holds 21 runs under the *first* Phase 4 rule:
  - WIN3_SPLIT_D8_ML, the main line's split: 9/10 bound routed on seeds 920–929;
  - WIN3_SLOW_D8: 4/9;
  - ORACLE_ADAM: 2/2.
- **Whose error.** The second version's "Phase 4 running" (tex2:510) was the true statement. The referee's second pass wrongly called it a contradiction with "no records". The commit was already in the referee's remote-tracking ref (reflog `origin/claude/replicate-R@{0}`), and the branch was described from memory. `second_pass.md` §1 item 4 now carries a dated correction.
- **The fix.**
  - Say Phase 4 started under the first rule and has 21 interim records.
  - Give the counts, or none, but not "not started".
  - Note that the amended reading's "fresh" seeds 920–951 (tex3:470; `db22857:docs/reading/parallel_sessions_prompts.md:868`) include the ten already run under the first rule.

**2. tex3:228: "the stream routing having formed first".** This is S85, about the key term's late key splits. It is unchanged text, newly checkable because the dense records carry the routing margin.
- **The records.** In all seven key-term runs whose η² by key reaches ≥ 0.2 at 3600 (seeds 2313, 2317, 2318, 2319, 2321, 2323, 2325):
  - the routing margin at 2400 is 0.00 in every one;
  - the map is one-to-one in only 2 of 7.
- **When routing appears.** Routing with margin ≥ 0.25 first appears at 4800 (five runs) or 6000 (two), after the key rise.
- **The same sentence's account of seed 2317** ("the slow gate held the gate back until the hinge's window had passed") is unmarked, and not borne out:
  - the slow gate's hold ends at update 1200, inside the window;
  - the slow gate alone had 2317 routed by 1800 (margin 0.99);
  - under the key term alone, 2317 was unrouted at 2400 and also split after the window, yet recovered.

**3. tex3:261: "the run with the weakest latch lowest".** The S56 range 0.20–0.44 is right; the parenthesis is not.
- The lowest run is 378: accuracy 0.198, end β^R at the context token 0.77.
- The weakest latch is 377: 0.49, accuracy 0.318.
- The parenthesis came from the referee's second-pass note (`check_G_v2.md`, "378, the weakest latch"), which was wrong.
- Drop it, or write "the lower of the two runs without a clean latch".

## 2. Statements to qualify or correct (PARTLY)

**R and the provenance tables:**
- **tex3:327 and 36: R's labels.**
  - "left 37 to be guessed, 20 ... silent, 12 ... ambiguous and 2 carried over": 20 + 12 + 2 = 34. The other three are R's entries stated only in part (2 [stated + A], 1 [A in code]), which the "seven outright or in part" already counts.
  - Exact: 4 stated, 3 stated in part, 34 guessed, 3 process.
  - "Carried over from R's own earlier choices" misdescribes [C], which means "the report states it for another component, and I carried it over".
  - tex3:36's "the 44 details it had to guess" should agree with it (34, or 37 with the partials).
  - The referee's second pass wrote "37 of 44 ... silent, ambiguous or carried over", lumping the partials in, and is the source of the 37.
- **tex3:325, 344, 388: "this version committed it".** The amended Phase 4 reading was committed by `db22857` on the main line, at 03:01:26. `a5f67a2` changes only the tex and the PDF. Its numbers check: size 0.0204; power 0.536 at 0.85 and 0.796 at 0.80. That it was "given in chat on 10 October" is the commit's own statement; no record shows it.
- **tex3:511: the V row's last commit.** It gives `16b04f9`. It should be `66334b8` (`second_pass.md`), with `6f2788d` (the test check) and the session second-pass commits (`fc873d2`, `dccc5cb`, `4f3233a`).
- **tex3:506: H's row.** It cites report 7 and H's branch at `1e09ecd`. `report_7.md` was corrected at `de11b2a`, 03:07:17, and tex3:228's window sentence follows that correction. Every store the draft cites is byte-identical, so no number changes.
- **tex3:41, 508: S83 is not mentioned.** Session G's S83 was specified at `b2f9907` (03:10:39) and started at `1973126` (03:11:06), before this version. Its log exists and it has no records yet. The running list and G's provenance row omit it, where F's row says "S81 running".
- **tex3:436: S86 is not mentioned.** "A trigger that reads all positions has not been screened" is true of results. But H's S86 is that screen: readings fixed at `50da719`, running.

**The descriptions.** tex3:342 says "those are now marked". That is still not true of:
- three tex3:259 statements, unchanged since the first draft:
  - "where the decayed sum of the block still carries the header";
  - "the gate reads only the window, nothing asks the stack to keep it";
  - "a gate over the residual stream therefore had nothing to read";
- the 2317 account (§1 item 2);
- tex3:209's recommendation, "the trigger should not fire when its target already carries key mass".

**Over-general or imprecise:**
- **tex3:265.**
  - "The runs that fail to discover end in a key split" holds for 28 of 32 (S72/S73) and 16 of 18 (S82's REG3): "most of".
  - "the write at the cue never separating (1 of 20)": seed 2113 separates from 2400 on (margin 0.75), so "separating in only 1 of 20". This wording came from the referee's `resolution_G.md`.
- **tex3:465.** The conclusion's "found label-free on a fifth of seeds" leaves out S82's 2/20 on fresh seeds (Hebbian) and 2/10 (delta), which tex3:455 and 267 now state.
- **tex3:384.** "the onsets move later and the splits still form" holds of the two sets of onsets. Only two seeds have an onset in both arms (2310: 125 → 275; 2323: 125 → 575).
- **tex3:438.** "screened once at a fixed length": two screens (S59 delta, S69 Hebbian) at one length (L = 37).
- **tex3:427.**
  - "the count of checks the probe blocked, as test_delta_recipe now does": the test counts probe readings ≥ 0.95 with held-out < 0.95, not checks the probe blocked.
  - Its denominator includes the 2400 reference rows (`test_window_gate.py:520-546`); see `test_check_delta_recipe.md`'s addendum.
  - The amended spec asked for "the count of checks where the probe read >= 0.95 ... while the held-out accuracy was < 0.95", so the test is right by its spec. The draft's paraphrase is not.

## 3. What this version fixed (checked against the records)

**Session F.**
- The six delta splits: five copied a channel holding 0–1 streams; 1314@9600 copied the merged channel 6 onto channel 10, which held stream 4.
- S59's horizon reading "it does not": the gap difference is negative on 7/10 per decay, median −0.46, at L = 37.
- S68: "its counts bore on the claim", with readings "neither".
- The raw-key mechanism and the stall's cause are marked post hoc.
- GDN's learned gate is placed on the P = 8 header layout.

**Session G.**
- S56's range is 0.20–0.44.
- "It needed layer 1" is marked as our reading of S71.
- "Formed in the records": the order holds in 28/28 S72/S73 key-split failures and 16/16 S82 REG3 key splits.
- SLOWGATE's "worse than nothing" is gone and its cause is marked.
- The S72/S73 labels.
- The half-routed range is "also at keys".
- S82's delta phase is described as descriptive: ROUTED-BOUND 10/0/2, BOUND 10/2/4.
- "Does not return" is observed for 2400 updates after each transition.
- What the hinge reads.
- S82's 2/20 at tex3:455.

**Session H.**
- S85:
  - "stays below the term's threshold of 0.2 ... maximum 0.195";
  - 0.21–0.88 on seven seeds;
  - five seeds organised by key at initialisation;
  - the slow gate's four key failures, three on the baseline's seeds;
  - the trace's sampling, every 25 to 1200 and every 100 to 2400;
  - the onsets by seed.
- S84's fifth arm, EVAL_GUARD_INT1200, at 9/10, 0 vs 0; no arm read "costs binding".
- The guard "moves the splits to higher probe accuracies": median 0.797 against the base's 0.617.
- "Shape, not count" is marked.
- "Did not work here (S78, 'neither device')".
- The screen markers at tex3:436 and 465.

**Main line and provenance.**
- tex3:427 says the records keep the probe readings, consistently with tex3:205, 344, 382 and 390.
- R's 914 is 0.953–1.000.
- 6 of 353 checks.
- CHECK 130 is "differing".
- Appendix B's `Generator(424242)`, confirmed by drawing the batch: 2048 sequences.
- split_copy's L half is "not run". True: L's one start failed its verification (CHECK 155, `--also` given a directory) with 0 of 54 runs, per the start log in `84a7e57` on `claude/bdh-repo-curl-obgdqk`, 01:58:54. The draft does not mention the attempt.
- delta_recipe's row: spec, Amendment 1, test `a581027`, not run.

**The referee's own rows.**
- tex3:327's eleven guesses and the D-10 map.
- tex3:334's "F 3–5, G 3–4, H 4–5, I 1".
- tex3:458's "two, four and eight streams".
- tex3:344's "three ranges" and "five of the six".
- tex3:465's screen markers.

## Corrections to the referee's own earlier documents, made in this pass

- **`second_pass.md` §1 item 4, the tex2:510 bullet.** It was wrong: R's Phase 4 records existed. A dated correction is added there.
- **`resolution_V3.md`, row tex2:510.** It is now NEW ERROR, and attributed to the referee.
- **`test_check_delta_recipe.md`.** An addendum on the probe-blind count's denominator (it includes the 2400 reference rows) and on its label.
- **Errors in the referee's second-pass files that the draft adopted:**
  - `check_G_v2.md`'s "378, the weakest latch" (§1 item 3);
  - `resolution_G.md`'s "never separating" (§2);
  - the "37" lumping in `resolution_V.md` (§2).

  Each is named in the paragraph that corrects it here.

## What could not be checked

- The PDF, page by page.
- Whether R had the amended prompt before starting Phase 4; whether L restarted after its failed start; whether S81, S83, S86 or R's Phase 4 are running in processes that have committed nothing.
- When `84a7e57` (L's start log) was pushed, so whether the draft's author could have seen it at 03:15.
- tex3:390 and 427's "nobody read the two against each other".
- That X's start times are UTC (L's UTC−6 is confirmed by its start log's epoch).
- That seeds 2300–2329 are fresh (H's own scan), and whether S85's CHECK, run on a dirty tree, used exactly the committed code.
