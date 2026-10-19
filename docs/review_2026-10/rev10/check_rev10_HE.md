# Revision 10 on Session H's round 5 and Session E's batches 20-21: check_rev10_HE

Referee V. Draft: `585b666:multichannel_hebbian_report_v10.tex` (byte-identical to the merged third version `d2b7abd`), cited tex10:NNN. Records: `origin/claude/explore-H` at `347bd25` (report 9) and `origin/claude/outside-ideas` at `419368a` (batch 21 complete), both still the branch heads. The script is `check_rev10_HE.py`. It uses the standard library and `rlib` only, reads git objects, trains nothing and runs no session code. It prints its output; nothing is written. Every number in the draft is found by building its text from the recomputed value, so no draft number is an input.

The referee had not checked these screens before. `check_H_v6` saw only S87's readings commit and 0 records.

## Counts

**39 rows: 24 MATCH, 10 PARTIAL, 3 DISAGREE, 2 NOT RECORD-CHECKABLE.**

- Every count, discordant pair, p-value and reading reproduces from the records for:
  - S87a, S87c and the one-to-one recount (H);
  - batches 20 and 21 (E).
- The three DISAGREEs are one stale statement, repeated on three unchanged lines. Each still says the all-position trigger was "screened once (S86)", but this revision reports S87a.
- The PARTIALs are about scope, labels, provenance and one probability model. None is about a count.

**Pre-registration order: clean in all three screens.**

| screen | readings committed | first record | runs | code SHA recomputed from the committed sources |
|---|---|---|---|---|
| H S87a | `f4fd535` 2026-10-12T22:45:48 | `3d4deea` 23:30:52 (the store first appears at `b17517a` 23:00:50 with meta only) | 42 | 898867f69ad4 = store |
| H S87c | `f4fd535` 22:45:48 | `c315b21` 2026-10-13T06:01:43 | 42 | 65d3498e4511 = store |
| E batch 20 (S68) | `4acf915` 2026-10-11T21:27:31 | `b6cc8a4` 23:10:58 (the dry store starts at `1b18ac4` 21:45:07) | 19 | 78c8b0a643d9 = store (sources at `0fa79a3`) |
| E batch 21 (S69) | `b93b73c` 2026-10-13T20:40:35 | `5bcdf63` 2026-10-14T00:15:48 (dry from `9cdd1b3` 21:49:22) | 20 | f7a606d7399a = store |

**What changed after the readings commits.**
- H:
  - `explore_h_allmass2.py`, `explore_h_dose.py` and `explore_h_dose_child.py` have one commit each (`f4fd535`).
  - No commit after it touches any module in either run's code-SHA list.
  - The only `.py` file changed is `explore_h_report.py`, the report writer, which no child imports.
- E:
  - The READINGS docstrings of `explore_continue.py` and `explore_oracle_d128.py` are identical at the head.
  - The later diffs are report code only: `report()` lookups, a nan guard in `both_two`, and a CHECK comparison of lists coming back from JSON.

**The dose CHECK's "f4fd535+dirty".** Both screens' CHECKs ran in one invocation of `explore_h_run.py`. The run's git was `f4fd535`, clean, per s87.log. The sequence:

1. The allmass2 CHECKs passed. `main()` then rewrote `explore_out/H/checks.json` at 22:57:03. That file is tracked at `f4fd535` and had no H/allmass2 entry there.
2. `git_head()` uses `git status --porcelain --untracked-files=no`, so every later stamp in that process carries "+dirty". That includes the dose CHECK's at 23:00:36.
3. checks.json was committed with both entries at `b17517a` (23:00:50).

Is this the only possible cause? An edit to `explore_h_report.py` (committed at `8aa677c`, 23:01:08) cannot be excluded either. It does not matter for the runs: both code SHAs, recomputed here from `f4fd535`'s committed sources (and main's `9c5939e` for the eight served modules), equal the stores' SHAs exactly.

Of the 40 distinct git stamps on the records, 24 are clean. A clean stamp means no tracked file, the run modules included, differed from the commit at that moment.

**Seeds.**
- H: 2600-2659 is the block the user's round-5 prompt assigns to H. On every remote branch, a seed in 2600-2659 appears in a record key or seed field under explore_out/, results/ or specs/ only in H's own `allmass2_results.json` and `dose_results.json`. The guarded-trigger seeds, 2620-2629, are unused.
- E: batches 20 and 21 reuse S67's seeds 270-279 by design. Batch 20 continues S67's runs. Batch 21 pairs with S67's oracle and reproduces it.

## Session H (round 5)

Recomputed (`rlib.bound_routed` and `outcome_k`; the labels are recomputed from `map_before` and agree with the stored flags everywhere):

- **S87a, VALID.** ORACLE 2/2, at 2400 and 8400.
  - ALLMASS bound routed 16/20, BASE 17/20. ALLMASS only 1 (2614), BASE only 2 (2610, 2618). One-sided p is 0.875 for ALLMASS > BASE and 0.5 for BASE > ALLMASS.
  - ALLMASS firings 42: key_ok 42, all_ok 40 (own pair 40). The two misses are 2611 at 14400 and 2618 at 19200.
  - BASE firings 41: key_ok 41, all_ok 39, own (key) pair 41.
  - The two rules chose the same pair at 15 (ALLMASS) and 17 (BASE) firings. No firing had a one-to-one map before it.
  - Readings: "keeps it" (2 vs 1), "aims" (0.952), "costs binding" does not apply. The stored and recomputed classes agree, and v1 = v2 on every run.
- **Pooled with S86** (store unchanged since `9b72bf6`; S86 alone: BASE 9/10, ALLMASS 8/10, 0 vs 1). ALLMASS 24/30, BASE 26/30, 1 vs 3. One-sided p is 0.9375 for ALLMASS > BASE and 0.3125 for BASE > ALLMASS.
- **The recount.** WIN3_SPLIT_D8 records with checks; every WIN3_SPLIT_D8 run on the main line has checks.
  - At `0282955`: 50 runs, 107 firings, key 107, all-position 97, same pair 46, one-to-one before 0.
  - At `0754d61` and `585b666`: 60 runs, 126 firings, key 126, all-position 115 (46/47, 30/38, 21/22, 18/19), same pair 54, one-to-one before 0.
  - H's `onetoone_counts.txt` reproduces.
- **S87c, VALID.** ORACLE 2/2, at 1200 and 1200.
  - KEYHINGE, LAM03, LAM3 and WIN1200 are each DISCOVERED 10/10, with no discordant pair. The reading is "robust", and the amendment rule does not fire.
  - Median transitions: KEYHINGE 4800, LAM03 4800, LAM3 6600, WIN1200 4800.
  - WIN1200 equals KEYHINGE run for run (curves and firings).
  - The maximum key η² over updates 25 to the window is 0.130, 0.153, 0.138 and 0.130. No run ends with a key split above 0.5.
  - **Power:** a variant whose true rate is 0.8, against a drafted arm at 10/10, reads "within 1 pair" 38% of the time; at 0.9, 74%.

| # | tex10 | item | draft | records | status | note |
|---|---|---|---|---|---|---|
| H1 | 358 | stale | "a trigger that reads all positions was screened once, session~H's S86, $8/10$ against $9/10$" | screened twice: S86 and S87a (16/20 vs 17/20; pooled 24/30 vs 26/30), which tex10:67 and tex10:217 report | **DISAGREE** | unchanged line, false since this revision folds in S87a |
| H2 | 464 | stale | "a trigger that reads all positions was screened once (session~H's S86): it keeps binding within one pair ... ($8/10$ against $9/10$) and its one mistarget in 21 firings cost that run" | as H1 | **DISAGREE** | unchanged line. In S87a, ALLMASS's two mistargets were 2611, which still bound, and 2618, which merged |
| H3 | 486 | stale next step | "a trigger on all positions was screened once (S86) and kept binding within one pair" | as H1 | **DISAGREE** | unchanged line |
| H4 | 67 | index: "the key term's constants are robust" | as quoted | "robust" on 10/10 × 4; the drafted arm at the ceiling | PARTIAL | the reading is the printed one. The index gives no scope: the screen could detect only a worse variant, and H's report 9 and its dated spec note both say so. A variant at 0.8 passes 38% of the time |
| H5 | 217 | S87c | "all discovered $10/10$ at two streams (S87c; ``robust'', and the drafted constants stand)", under "closed three small questions" | as above | PARTIAL | the counts and both readings MATCH. "Closed" overstates a ceiling screen that could show only a worse variant (see the power above), a caveat H states and the draft leaves out |
| H6 | 536 | provenance (tab:batches), H's row | "... the one-to-one recount ... (`5b5deec`, round 5) & `5b5deec` & 318; ...; 32" | S87a 42 + S87c 42 runs, report 9 at `347bd25` | PARTIAL | the draft uses S87a and S87c (tex10:67, 217), but the row cites neither of them, nor report 9. Its commit column is not H's last report, and its run counts leave out 84 |
| H7 | 217 | "and the key-position rule stays" | as quoted | no reading names it | NOT RECORD-CHECKABLE | a design decision, not a reading |
| H8 | 44 | H's S87 in the fifth round; "the fifth round's screens have not been refereed" | as quoted | true at `585b666` | MATCH | |
| H9 | 67 | index | "twenty new seeds ($16/20$ against $17/20$; $24/30$ against $26/30$ over two screens)" | as recomputed | MATCH | |
| H10 | 67 | index | "no main-line firing had a one-to-one map before it" | 0 of 126 | MATCH | |
| H11 | 217 | S87a | "bound $16/20$ against the key-position rule's $17/20$ (S87a, seeds 2600--2619; 1 vs 2), $24/30$ against $26/30$ pooled with S86" | as recomputed; oracle VALID | MATCH | |
| H12 | 217 | S87a firings | "all 42 of its firings on a target the key-position labels name and 40 on one the all-position labels name" | 42, 42, 40 | MATCH | |
| H13 | 217 | S87a readings | "``keeps it'' and ``aims'' apply, ``costs binding'' does not (printed ``no cost by the rule'')" | as recomputed | MATCH | |
| H14 | 217 | the guard | "none of the 126 firings in the 60 recorded split-arm runs ... had a one-to-one map before it" | 60 runs, 126 firings, 0 | MATCH | the counterfactual "would have changed nothing" follows: the guard blocks only on a one-to-one map |
| H15 | 464 | counterfactual labels (recheck) | "at 115 (46 of 47, 30 of 38, 21 of 22 and 18 of 19), the key-position pair at all 126; ... coincide at 54" | as recomputed | MATCH | |
| H16 | 44 | "G's S87, H's S87" | as quoted | both sessions numbered their round-5 screens S87 | MATCH | the draft disambiguates by session and by suffix; noted only |

Report 9's descriptive claims, checked though the draft does not use them:
- flat start: 13 (BASE) and 12 (ALLMASS) firings at accuracy < 0.2;
- runs at the cap of 3: 6 and 7;
- the same pair at 17/41 and 15/42;
- 2600 and 2613 merged under both arms;
- LAM03 on 2630 fired 4 times against KEYHINGE's 2;
- WIN1200 equals KEYHINGE run for run.

All of these reproduce. Report 9's "2 of 498 firings" in H's own eight-stream runs was not recounted (see the last section).

## Session E (batches 20 and 21)

Recomputed. Outcomes use `rlib.outcome_k`, and the reproduction check reads the S67 records at `e3f5809`.

**Batch 20 (S68).** S67's unbound P = 16 learned runs:
- D128: 6, all one-to-one (271, 272, 273, 275, 276, 279).
- N256: 10, of which 6 are one-to-one (270, 272-276) and 4 merged (271, 277-279).

All of them were continued, together with P8_N256 272 and the D128 oracle on 270, plus RCHK.

**Reproduction.** All 18 regenerated runs equal their S67 records at 43,200 in:
- the curve through 43,200;
- the statistics;
- the early curve;
- the weights' SHA-1 against S67's `w_sha`.

**Outcomes at 86,400.**
- The one-to-one runs bound 0/6 at D128 and 0/6 at N256: "stuck" and "stuck", joint "neither". The merged runs bound 0/4.
- N256 279 went from MERGED to one-to-one, unbound.
- The D128 oracle on 270 is unbound at 0.376.

**Accuracy change from 43,200 to 86,400.**
- D128: +0.018 to +0.102 unrounded, 0.02 to 0.10 from rounded endpoints.
- N256 one-to-one: −0.015 to +0.414 unrounded, −0.02 to +0.41 from rounded endpoints. The draft and E's README use the rounded endpoints.

**Budget.**
- Still rising (≥ 0.02 from 76,800 to 86,400): D128 271 and 272, N256 276.
- The bound-routed P = 16 learned transitions (S67 and S68) are 19200, 26400, 37200 and 38400, so the budget is 40,800, printed with "86,400 is not enough".

**The 44,400 resume check.** The pre-registered check printed DIFFERS, because the record's `resumed` list is [61200]. Every measured field agrees at 44,400: the curve, the statistics, and the weights' SHA at 43,200 and 44,400. That is E's addendum, confirmed here. The draft does not mention the check.

**Batch 21 (S69).**
- ORC_A_D128 bound 4/10: 271, 273, 274 and 278, at 12000, 19200, 27600 and 31200.
  - Close but unbound: 275, 277 and 279, at 0.911, 0.906 and 0.921.
  - Stuck: 270, 272 and 276, at 0.318, 0.287 and 0.296.
- ORC_S_D128 bound 0/10. Final accuracies are 0.330-0.356, with every stream between 0.31 and 0.38.
- Readings: "unreliable" for both arms. The spec reading is rule 4, "no D128 validity check at 40,800".
  - b(b−1)/90 = 0.133.
  - Wilson interval for A: [0.17, 0.69].
  - A only 4, S only 0; McNemar two-sided p = 0.125.
- Reproduction against S67: ORC_A_D128 on 270 and 271 equal S67's records through 40,800, and 271 matches the final weights' SHA as well.
- The spec draft's status section reads "NOT READY FOR V" and lists five options, none screened and none adopted.

| # | tex10 | item | draft | records | status | note |
|---|---|---|---|---|---|---|
| E1 | 245 | "two validity seeds would both bind with probability about $0.13$" | as quoted | 4·3/90 = 0.133 is for two seeds drawn from these ten. The draft's validity seeds are fresh (660-661, 1660-1661): 0.4² = 0.16 per machine, 0.026 on both | PARTIAL | the number is E's printed figure, but it uses the wrong model for the draft's seeds. V's `keyload_options_review.md` item 1, written after Rev 10, says the same |
| E2 | 245 | "with the routing given and no gate to protect, the slow start only delays" | as quoted | no SLOW run went past 40,800. SLOW ended below single Adam on the 7 seeds that bound or came close, and above it on the 3 stuck ones (+0.03, +0.04, +0.04) | PARTIAL | an unmarked interpretation. The records show SLOW at 0.33-0.36 everywhere at 40,800, not a delay. E's README says SLOW "did not help ... it hurt" |
| E3 | 245 | "batch~21 screened that first (S69, ..." | as quoted | tex10:256 and tab:batches use S69 for F's Hebbian-decay screen; E's S68 also repeats F's number | PARTIAL | two different screens carry one name in the same report |
| E4 | 535 | tab:batches, E | only "15--18 (E)" and "19 (E)" | batch 20 (19 runs, `0fa79a3`/`f5d3893`) and batch 21 (20 runs, `419368a`) have no row | PARTIAL | their results are used at tex10:64, 245 and 455 |
| E5 | 44 | "the exploratory sessions' fifth round (E's batches~20 and~21, ...)" | as quoted | in the prompts file, batch 20's prompt comes before the "Round 5" header, which names V, G, H, F, R and I. Batch 21 was a chat request on 13 Oct | PARTIAL | labelling only |
| E6 | 44, 495 | "every number recomputed from the records" / "the fifth round's screens have been recomputed there" (`docs/review_2026-10/rev10/`) | as quoted | on main, rev10/ has recompute_H.py for H. No script there recomputes E's batches 20-21: the first-pass and second-pass rows cite E's README, and two cite the JSON stores | PARTIAL (2 rows) | true for H, not for E. This check now recomputes E's numbers, and all of them reproduce |
| E7 | 455 | "ten seeds of it cost less than a review" | as quoted | batch 21: 20 runs, about 30 h wall | NOT RECORD-CHECKABLE | |
| E8 | 64 | index | "stuck ($0/12$ by 86{,}400); the oracle $4/10$ and $0/10$ at 40{,}800: the drafted key-load test has no validity check" | as recomputed | MATCH | |
| E9 | 245 | which runs; reproduction | "(six one-to-one at $D = 128$; six one-to-one and four merged at $N = 256$) was regenerated from update 0, reproduced its record at 43{,}200 bit for bit, and continued to 86{,}400 updates" | 18/18 reproduce | MATCH | |
| E10 | 245 | counts and readings | "the one-to-one groups $0/6$ and $0/6$, ... ``stuck'' at both widths and ``neither'' jointly ...; the merged four $0/4$, descriptive" | as recomputed | MATCH | |
| E11 | 245 | accuracy changes | "$0.02$--$0.10$ ... $-0.02$ to $+0.41$" | as above | MATCH | from rounded endpoints; unrounded, the N256 low end is −0.015 (seed 274) |
| E12 | 245 | oracle 270 | "still unbound at 86{,}400" | 0.376 | MATCH | |
| E13 | 245 | budget | "gave 40{,}800 (the latest transition of any bound sixteen-key run, 38{,}400, plus 2400), and printed beside it ``86{,}400 is not enough''" | as recomputed | MATCH | the rule takes BOUND ROUTED *learned* runs. Including the oracles (12000, 24000, 27600) would not change the maximum |
| E14 | 245 | "rested on an oracle that had bound one of its two seeds (three runs, the failed seed having been continued)" | as quoted | S67 270 unbound, 271 at 12000; 270 continued | MATCH | |
| E15 | 245 | S69 order and design | "(S69, seeds 270--279, 40{,}800 updates, readings fixed before any run)" | readings `b93b73c`, first record `5bcdf63` | MATCH | |
| E16 | 245 | S69 counts | "$4/10$ ... ($0.91$--$0.92$ ... three stuck near $0.30$) and $0/10$ under SLOW (every seed at $0.33$--$0.36$" | as recomputed | MATCH | |
| E17 | 245 | reading | "``no $D = 128$ validity check at 40{,}800''" | rule 4 (b_A 4, b_S 0) | MATCH | |
| E18 | 245 | status note | "five ways forward, none screened and none adopted" | 5 options, as listed in the draft | MATCH | |
| E19 | 245 | E's verdict | "``inconclusive, weaker than handed back''" (4/10 vs 0/10, $p = 0.125$) | README, batch 20 verdicts; McNemar 0.125 | MATCH | |
| E20 | 366 | "committed after those reviews and then failed its own validity screen ...; it has not been reviewed" | as quoted | draft `f5d3893` 13 Oct 02:54; V's spec reviews `3f7d84b` 12 Oct 22:50 and `723827b` 23:00; V's keyload note `11a78ee` 19 Oct 17:51, after Rev 10 (16:15) | MATCH | |
| E21 | 455 | "screened on ten, it binds four under the recorded recipe and none under the other" | as quoted | 4/10, 0/10 | MATCH | |
| E22 | 499 | key-load test: "a validity check it does not have at its budget (Section~\ref{sec:referee})" | as quoted | E's own status. sec:referee (tex10:366) points on to sec:scale | MATCH | the finding is E's (batch 21), not a review |

Not mentioned in the draft, from E's records:
- S68's pre-registered 44,400 resume check printed DIFFERS. It failed on a bookkeeping condition; the measured fields agree.
- Batch 21's segment 4 ran three single-Adam runs on a 2.10 GHz Xeon: 270, 272 and 274. Most segments ran on 2.80 GHz.

## Other observations

- **Main's `docs/review_2026-10/rev10/recompute_H_out.txt`** (another session; read, not run) has two problems:
  - The committed output contains an IndexError traceback (recompute_H.py l.54), followed by a rerun.
  - Its "BASE ... own pair ok 39" is wrong: BASE's own pair is the key pair, labelled at 41/41. The script falls back to `all_ok` when there is no `own_ok` field. The draft does not use this number.
- **The referee's earlier files:** in this scope, no error was found in V's earlier files.
  - `check_H_v6`'s round-5 notes (0 records at `8aa677c`; the dose CHECK's "f4fd535+dirty") are confirmed. The dirty flag is now explained.
  - `resolution_G_H5.md` predicted that S87a would bear on tex5:205, 343, 441 and 463. Those lines are tex10:217, 358, 464 and 486; only tex10:217 was updated, which gives rows H1-H3.

## What could not be checked

- **Report 9's "2 of 498 firings had a one-to-one map before"** in H's own eight-stream runs. Which stores make up the 498 is not stated, and the draft does not use the number.
- **Whether every run used exactly the committed code.**
  - The evidence:
    - The code SHA is computed once per invocation, at its start. For H, E batch 20 and E batch 21 it recomputes exactly from the committed sources.
    - No commit after the readings touches a run module.
    - Clean git stamps occur during H's runs.
  - What a SHA cannot exclude: a child imports from disk, so an uncommitted edit made and undone between those clean stamps would not show in it.
- **A review's cost** (tex10:455). It is not recorded anywhere.
- **"and the key-position rule stays"** (tex10:217). It is a decision, not a record.
