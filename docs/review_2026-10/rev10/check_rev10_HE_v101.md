# Revision 10.1 on Session H's round 5 and Session E's batches 20-21: check_rev10_HE re-run against `846d0c8`

Referee V, 20 October 2026.

**What was re-run.** `check_rev10_HE.py`, written against `585b666` (tex10, 591 lines), is re-pointed as `check_rev10_HE_v101.py` at Revision 10.1, `846d0c8:multichannel_hebbian_report_v10.tex` (596 lines), cited tex101:NNN.

**Records.** The records are unchanged: `explore-H` at `347bd25` (report 9) and `outside-ideas` at `419368a` (batch 21 complete). The script uses the standard library and `rlib` only, reads git objects, trains nothing and runs no session code. It prints its output; this file is that output with a summary.

**How the re-run differs from the original.**
- **Anchors.** Every row is anchored on its tex10 line at `585b666`, as before, and on the line that line became at `846d0c8` (difflib's line map). Fragments that `846d0c8` rewrote are re-anchored on their new wording and judged against it. Some draft numbers are rebuilt from recomputed values, and those values are unchanged; one of them (−0.01) now uses the unrounded differences, as `846d0c8` does. tex10:495's "the fifth round's screens have been recomputed there" no longer exists, so it is reported **REMOVED**, with a new row for its replacement (tex101:499).
- **Pins.** Branches are pinned to what the original read, from this clone's reflog at its run (19 Oct, before 18:01:47 UTC): `explore-H` `347bd25`, `outside-ideas` `419368a`, the main line `7c4ceb8` (for the prompts file), and the seed grep's 13 remote branches (`BRANCH_PINS`).
- **Recomputations.** Every record recomputation is kept. The one-to-one recount is also run at `846d0c8`, and it equals the one at `585b666`.
- **The original's statuses.** These are the original's 39 rows (H1–H16 and E1–E22; E6 covers two rows). They are given row by row, and the script asserts that they total 24/10/3/2.

## Counts

| | rows | MATCH | PARTIAL | DISAGREE | NOT RECORD-CHECKABLE | REMOVED |
|---|---|---|---|---|---|---|
| original, against `585b666` | 39 | 24 | 10 | 3 | 2 | |
| the same 39 rows, against `846d0c8` | 39 | 35 | 1 | 0 | 2 | 1 |
| new rows (text `846d0c8` changed or added) | 5 | 5 | 0 | 0 | 0 | |
| **all, against `846d0c8`** | **44** | **40** | **1** | **0** | **2** | **1** |

**Every count, discordant pair, p-value and reading still reproduces from the records:**
- S87a, S87c and the recount: 16/20 against 17/20, 1 vs 2; 24/30 against 26/30; 42/42/40 firings; "robust"; 0 of 126 firings on a one-to-one map.
- Batches 20 and 21: 0/6 and 0/6; 18/18 reproduce at 43,200; a budget of 40,800; 4/10 and 0/10; rule 4.

**The pre-registration order is clean in all three screens, unchanged:**

| screen | readings committed | first record | code SHA recomputed from the committed sources |
|---|---|---|---|
| H S87a | `f4fd535` 2026-10-12T22:45:48 | `3d4deea` 23:30:52 | 898867f69ad4 = store |
| H S87c | `f4fd535` | `c315b21` 2026-10-13T06:01:43 | 65d3498e4511 = store |
| E batch 20 (S68) | `4acf915` 2026-10-11T21:27:31 | `b6cc8a4` 23:10:58 | 78c8b0a643d9 = store |
| E batch 21 (S69) | `b93b73c` 2026-10-13T20:40:35 | `5bcdf63` 2026-10-14T00:15:48 | f7a606d7399a = store |

## Status changes, `585b666` → `846d0c8`

| # | tex10 → tex101 | was | now | the records |
|---|---|---|---|---|
| H1, H2, H3 | 358 → 360, 464 → 468, 486 → 490 | DISAGREE | MATCH | **"screened twice (S86, S87a)"**: S86 8/10 vs 9/10 (0 vs 1), S87a 16/20 vs 17/20 (1 vs 2), pooled 24/30 vs 26/30.<br>tex101:468's new count, "of its three mistargets in 63 firings two cost their run", is recomputed:<br>• firings 21 + 42;<br>• mistargets S86 2334 at 19200 (unbound), S87a 2611 at 14400 (bound) and 2618 at 19200 (merged);<br>• two of the three runs lost |
| H4 | 67 → 69 | PARTIAL | MATCH | **"robust on a ceiling screen"** |
| H5 | 217 → 219 | PARTIAL | MATCH | **"the drafted arm was at the ceiling … a variant at a true rate of 0.8 still reads robust 38% of the time"**: P = 0.376 |
| H6 | 536 → 541 | PARTIAL | MATCH | H's provenance row: **S87a, S87c and the recount (report 9, `347bd25`), commit `347bd25`, runs …; 84** (42 + 42) |
| E1 | 245 → 247 | PARTIAL | MATCH | **"two validity seeds drawn from these ten … about 0.13 (two fresh seeds on one machine about 0.16 at that rate, on both machines 0.03)"**: 0.133, 0.16, 0.0256 |
| E2 | 245 → 247 | PARTIAL | MATCH | **"the slow start hurt: no SLOW run bound and SLOW ended lower on every seed that bound or came close under the other recipe"**: 0/10; SLOW lower on 7 of 7 |
| E3 | 245 → 247 | PARTIAL | MATCH | **"(E's S69, not F's;"**. E's S68 is named only in the provenance table (tex101:540), beside F's S68 (tex101:542), each under its session's label |
| E4 | 535 → 540 (new row) | PARTIAL | MATCH | **"20–21 (E) & S68 … (readings `4acf915`; complete `f5d3893`); S69 … (readings `b93b73c`; complete `419368a`) & `419368a` & 19; 20"**: 19 and 20 runs; `f5d3893` is "Batch 20 complete" and `419368a` is batch 21's completion |
| E6 (44) | 44 → 44 | PARTIAL | MATCH | **"which recomputed `test_delta_recipe`, the GPU run and H's screens and read the other screens' reports"**: the scripts at `585b666` are recompute_delta, _gpu and _H; none is for E |
| E6 (495) | 495 → — | PARTIAL | **REMOVED** | replaced at tex101:499: "after the merge it checked the rest, the fifth round's screens included, with the same outcome" (new row, MATCH) |
| H8 | 44 → 44 | MATCH | MATCH (re-anchored) | "after the merge, of the whole revision and of the fifth round's screens (`rev10/referee_pass.md`": present at `846d0c8` |
| E11 | 245 → 247 | MATCH | MATCH (re-anchored) | "$-0.01$ to $+0.41$": the unrounded N256 low end, −0.0146 (seed 274), rounded once. The original's "−0.02" came from rounded endpoints, as E's README does, and the referee's pass noted −0.01 |
| E15 | 245 → 247 | MATCH | MATCH (re-anchored) | "(E's S69, not F's; seeds 270–279, 40,800 updates, readings fixed before any run)" |

One row is still PARTIAL:
- **E5, tex101:44, "the exploratory sessions' fifth round (E's batches 20 and 21, …)".** Batch 20's prompt comes before the "Round 5" header, which does not name E. This is labelling only, and `846d0c8` did not take it.

The two NOT RECORD-CHECKABLE rows are unchanged:
- **H7, tex101:219, "and the key-position rule stays".** It is a design decision.
- **E7, tex101:459, "ten seeds of it cost less than a review".** A review's cost is not recorded.

## New rows (text `846d0c8` added on H and E), all MATCH

- **tex101:247, "The batch completed after Revision 9's first draft (186 of 186 runs)".** Batch 19 was complete at `e3f5809` (2026-10-11T04:25:06, "batch 19 complete (186/186 runs)"), after Revision 9's first draft `9bfa85a` (2026-10-10T22:16:07). This is the referent fix.
- **tex101:370, 10.1's account.** It says "``screened once'' … on three lines after S87a" and "provenance rows for E … and H". It lists the loose items as taken: the dose screen's ceiling, the fresh-seed probabilities, "the slow start hurt", and E's S69.
- **tex101:499.** "After the merge it checked the rest, the fifth round's screens included, with the same outcome." `check_rev10_HE.md` is at `846d0c8`, and every H and E number reproduces.

## Pinned branches, and heads that moved before `846d0c8` (18:33:26 UTC)

| branch | pinned (the original read) | first-parent commits after the pin, by 18:33:26 | the draft against them |
|---|---|---|---|
| `explore-H` | `347bd25` | `d4a9a7d` 17:54:28 (S89 readings, round 6), `7820b12` 17:56:24, `0dd88bf` 18:16:29, `42d86ae` 18:32:42 | the draft has no H status line beyond the fifth round; "Session H's fifth round … closed three small questions" stays right |
| `outside-ideas` | `419368a` | none | "five ways forward, none screened and none adopted": still right |
| `explore-F` | `a199d5e` | 9 (S89, round 6, from `50dc8e4` 17:54:31) | see `check_rev10_FG_v101.md` (tex101:270) |
| `explore-G` | `e46799a` | `7ab98c1` 17:54:12 (S89 readings) | no G status line in this scope |
| `review-V` | `6ab7649` | 8, through `9fda38e` | tex101:44, 499: right |
| main line, I, R, `rev9-draft`, others | as pinned | none | — |

The seed grep, re-run on the pinned trees, finds seeds 2600–2659 only in H's two S87 stores. H's S89 uses 2900–2959.

## What could not be checked

As in the original:
- report 9's "2 of 498 firings";
- whether every run used exactly the committed code (the code SHAs match, and clean stamps occur);
- a review's cost;
- "the key-position rule stays".

## Errors in the referee's own files

1. **`check_rev10_HE.md` says that `347bd25` and `419368a` were "both still the branch heads". That was not so for `explore-H`.**
   - The clone's remote-tracking ref had last been fetched on 13 Oct.
   - On the remote, H had moved to `d4a9a7d` (committed 17:54:28) and `7820b12` (17:56:24). Both arrived with the next fetch, at 18:02:41, 54 s after the HE file's commit (18:01:47).
   - The original's seed grep "on every remote branch" therefore read stale F, G and H trees (`a199d5e`, `e46799a`, `347bd25`).
   - Neither changes a result: H's S89, F's and G's new work use other seed blocks, and every H number is from `347bd25`'s stores. The claim itself is false.
2. **The original is not reproducible as committed.** It reads refs, not commits, through `MAINL` and every `git branch -r` tip, and it prints "branch head now" against them. The comparison pattern of the A and FG scripts, 7-digit `%h` against literals, depends on the clone's abbreviation length. The v101 script pins every ref and sets `core.abbrev=7`.

The original's numbers are unchanged.

## Rows (generated)

| # | where | item | draft | records | status | v585b666 → v846d0c8 | note |
|---|---|---|---|---|---|---|---|
| 1 | tex101:44 | H: fifth round includes H's S87; after the merge the referee checked the fifth round's screens (rev10/referee_pass.md) | and, after the merge, of the whole revision and of the fifth round's screens (\texttt{rev10/referee\_pass.md} | S87a/S87c/recount at 347bd25; V's branch had no S87 check before Rev 10 (check_H_v6: 0 records); at 846d0c8 check_rev10_HE.md and referee_pass.md in rev10/: True | **MATCH** | MATCH (unchanged); re-anchored (tex10:44 rewritten) | the original row was MATCH for 'not refereed' (true at 585b666); 846d0c8 records the referee's check, which exists |
| 2 | tex101:69 | H index: all-position trigger | twenty new seeds ($16/20$ against $17/20$; $24/30$ against $26/30$ over two screens) | S87a 16/20 vs 17/20; pooled 24/30 vs 26/30 | **MATCH** | MATCH (unchanged); line changed elsewhere (tex10:67), fragment intact |  |
| 3 | tex101:69 | H index: no main-line firing on a one-to-one map | no main-line firing had a one-to-one map before it | 0 of 126 at 585b666 | **MATCH** | MATCH (unchanged); line changed elsewhere (tex10:67), fragment intact |  |
| 4 | tex101:69 | H index: 'the key term's constants are robust on a ceiling screen' | the key term's constants are robust on a ceiling screen | S87c 'robust' True; every arm {'KEYHINGE': 10, 'LAM03': 10, 'LAM3': 10, 'WIN1200': 10}; drafted arm at the ceiling (10/10) | **MATCH** | PARTIAL → MATCH; re-anchored (tex10:67 rewritten) | 846d0c8 adds the scope the original row asked for (the drafted arm was at 10/10, so the screen could show only a worse variant; a variant at a true 0.8 reads within one pair 38% of the time, at 0.9 74%) |
| 5 | tex101:219 | S87a counts, pairs, pooled | bound $16/20$ against the key-position rule's $17/20$ (S87a, seeds 2600--2619; 1 vs 2), $24/30$ against $26/30$ pooled with S86 | ALLMASS 16/20, BASE 17/20, 1 vs 2; pooled 24/30 vs 26/30 (1 vs 3); ORACLE 2/2 {2600: (True, 2400), 2601: (True, 8400)} VALID | **MATCH** | MATCH (unchanged); line changed elsewhere (tex10:217), fragment intact |  |
| 6 | tex101:219 | S87a ALLMASS firings labelled | all 42 of its firings on a target the key-position labels name and 40 on one the all-position labels name | firings 42, key_ok 42 (map_before 42), all_ok 40 (map_before 40) | **MATCH** | MATCH (unchanged); line changed elsewhere (tex10:217), fragment intact |  |
| 7 | tex101:219 | S87a readings | its readings ``keeps it'' and ``aims'' apply, ``costs binding'' does not (printed ``no cost by the rule'') | keeps it True (2 vs 1); aims 40/42 >= 0.9; costs binding False | **MATCH** | MATCH (unchanged); line changed elsewhere (tex10:217), fragment intact |  |
| 8 | tex101:219 | S87a: 'and the key-position rule stays' | and the key-position rule stays | no reading or record names it; report 9's main-line reading: within two pairs (24 vs 26) | **NOT RECORD-CHECKABLE** | NOT RECORD-CHECKABLE (unchanged); line changed elsewhere (tex10:217), fragment intact | a design decision, not a reading; nothing in H's records decides it |
| 9 | tex101:219 | one-to-one guard: 'would have changed nothing' | none of the 126 firings in the 60 recorded split-arm runs of \texttt{test\_window\_gate} | WIN3_SPLIT_D8 with checks: 60 runs (all 60), 126 firings, one-to-one before 0; H's recount (a) 0/107 at 0282955, (b) 0/126 at 0754d61 reproduce | **MATCH** | MATCH (unchanged); line changed elsewhere (tex10:217), fragment intact |  |
| 10 | tex101:219 | S87c counts and readings, with the ceiling and its power | all discovered $10/10$ at two streams (S87c; ``robust'', and the drafted constants stand; the drafted arm was at the ceiling, so the screen could only have shown a worse variant, and a variant at a true rate of $0.8$ still reads ``robust'' $38\%$ of the time) | {'KEYHINGE': 10, 'LAM03': 10, 'LAM3': 10, 'WIN1200': 10}; ORACLE {2630: (True, 1200), 2631: (True, 1200)} VALID; V only/K only {'LAM03': (0, 0), 'LAM3': (0, 0), 'WIN1200': (0, 0)}; robust True; amendment none; P(reads 'robust' \| true 0.8, KEYHINGE 10/10) 0.376 | **MATCH** | PARTIAL → MATCH; re-anchored (tex10:217 rewritten) | counts and readings MATCH; 846d0c8 now states the ceiling and the power ('a variant at a true rate of 0.8 still reads robust 38% of the time'), the caveat the original row found missing; LAM3's median transition is later (6600.0 vs 4800.0), descriptive |
| 11 | tex101:360 | 'screened twice, session H's S86 and S87a, 24/30 against 26/30' | a trigger that reads all positions was screened twice, session~H's S86 and S87a, $24/30$ against $26/30$ | screened twice: S86 (8/10 vs 9/10) and S87a (16/20 vs 17/20); pooled 24/30 vs 26/30 | **MATCH** | DISAGREE → MATCH; re-anchored (tex10:358 rewritten) | 846d0c8 takes the referee's fix. The paragraph's heading now reads 'what Revision 9 did with it', while this item reports S87a, which came after Revision 9: the item is the request's current state |
| 12 | tex101:468 | 'screened twice (S86 and S87a): within one pair on each screen ...; of its three mistargets in 63 firings two cost their run' | a trigger that reads all positions was screened twice (session~H's S86 and S87a): it keeps binding within one pair of the key-position rule on each screen ($8/10$ against $9/10$; $16/20$ against $17/20$; $24/30$ against $26/30$ pooled), and of its three mistargets in 63 firings two cost their run | S86: ALLMASS only 0, BASE only 1; S87a: 1 vs 2; pooled 24/30 vs 26/30; firings 21 + 42 = 63; mistargets 1 + 2 (seed, step, run bound routed): S86 [(2334, 19200, False)], S87a [(2611, 14400, True), (2618, 19200, False)]; runs lost after a mistarget 2 | **MATCH** | DISAGREE → MATCH; re-anchored (tex10:464 rewritten) | 846d0c8 takes the referee's fix and recounts the mistargets over both screens; 'within one pair on each screen' holds for S86 and S87a (the pooled 1 vs 3 is two pairs, and is given as a count, not as a screen) |
| 13 | tex101:490 | 'screened twice (S86, S87a) and kept binding within one pair on each' | a trigger on all positions was screened twice (S86, S87a) and kept binding within one pair on each | S86 BASE only - ALLMASS only = 1; S87a 1 | **MATCH** | DISAGREE → MATCH; re-anchored (tex10:486 rewritten) | 846d0c8 takes the referee's fix |
| 14 | tex101:468 | main line's counterfactual labels (recheck) | at 115 (46 of 47, 30 of 38, 21 of 22 and 18 of 19), the key-position pair at all 126; the two pairs coincide at 54 | labels recomputed from map_before == flags: all True, key True | **MATCH** | MATCH (unchanged); line changed elsewhere (tex10:464), fragment intact |  |
| 15 | tex101:541 | provenance (tab:batches), H's row: S87a, S87c and the recount (report 9, 347bd25), commit 347bd25, runs ...; 84 | S87a the all-position trigger on 20 more seeds, S87c the key term's dose, and the one-to-one recount over the main line's 126 firings (report 9, \texttt{347bd25}) & \texttt{347bd25} & 318; 300; 82; 62; 52; 82; 32; 84\\ | S87a 42 + S87c 42 runs, report 9 at 347bd25 (explore_out/H/report_9.md there: True) | **MATCH** | PARTIAL → MATCH; re-anchored (tex10:536 rewritten) | 846d0c8 adds S87a and S87c, report 9's commit and the 84 runs |
| 16 | tex101:44 | screen-number collision G's S87 / H's S87 | G's S87, H's S87 | both sessions numbered round 5 'S87' (G 2700-2729; H S87a/S87c) | **MATCH** | MATCH (unchanged); line changed elsewhere (tex10:44), fragment intact | the draft disambiguates by session and by suffix (S87a, S87c); noted only |
| 17 | tex101:66 | E index row | stuck ($0/12$ by 86{,}400); the oracle $4/10$ and $0/10$ at 40{,}800 | one-to-one 0/6 + 0/6; ORC_A 4/10, ORC_S 0/10; spec reading 'no D128 validity check at 40,800' | **MATCH** | MATCH (unchanged) |  |
| 18 | tex101:44 | E: 'fifth round (E's batches 20 and 21 ...)' | E's batches~20 and~21 | prompts file: batch 20's prompt sits before the 'Round 5' header, which names V, G, H, F, R and I, not E; batch 21 was a chat request (13 Oct) | **PARTIAL** | PARTIAL (unchanged); line changed elsewhere (tex10:44), fragment intact | labelling only: batch 20 was specified on 11 October, before round 5 |
| 19 | tex101:247 | S68: which runs; reproduced at 43,200 | (six one-to-one at $D = 128$; six one-to-one and four merged at $N = 256$) was regenerated from update 0, reproduced its record at 43{,}200 bit for bit, and continued to 86{,}400 updates | S67 unbound: D128 6 (6 one-to-one), N256 10 (6 one-to-one, 4 merged); all continued True; reproduction failing none of 18 (curve, stats, early curve, weights SHA-1) | **MATCH** | MATCH (unchanged); line changed elsewhere (tex10:245), fragment intact |  |
| 20 | tex101:247 | S68 counts and readings | none bound (the one-to-one groups $0/6$ and $0/6$, on which the pre-fixed readings ``stuck'' at both widths and ``neither'' jointly are read; the merged four $0/4$, descriptive) | D128 0/6 'stuck', N256 0/6 'stuck', joint 'neither', merged 0/4 | **MATCH** | MATCH (unchanged); line changed elsewhere (tex10:245), fragment intact |  |
| 21 | tex101:247 | S68 accuracy changes (unrounded differences, rounded once) | the $D = 128$ one-to-one runs creeping by $0.02$--$0.10$ over the second 43{,}200 updates and the $N = 256$ ones moving by $-0.01$ to $+0.41$ without binding | unrounded: D128 [0.102, 0.093, 0.045, 0.094, 0.018, 0.066]; N256 [0.004, 0.003, 0.07, -0.015, 0.122, 0.414] | **MATCH** | MATCH (unchanged); re-anchored (tex10:245 rewritten) | 846d0c8 now rounds the unrounded differences once: N256 -0.0146 -> -0.01 (seed 274), as the referee's pass noted; the D128 range +0.018 to +0.102 reads 0.02-0.10 either way |
| 22 | tex101:247 | S68 D128 oracle on 270 | the $D = 128$ perfect gate on the seed it had failed (270) was still unbound at 86{,}400 | bound False; acc 0.376 at 86,400; S67 oracle D128: 271 at 12000, 270 unbound | **MATCH** | MATCH (unchanged); line changed elsewhere (tex10:245), fragment intact |  |
| 23 | tex101:247 | budget rule | gave 40{,}800 (the latest transition of any bound sixteen-key run, 38{,}400, plus 2400), and printed beside it ``86{,}400 is not enough'' | BR P=16 learned transitions [19200, 26400, 37200, 38400] -> 40800; still rising ['C_P16_D128\|271', 'C_P16_D128\|272', 'C_P16_N256\|276'] | **MATCH** | MATCH (unchanged); line changed elsewhere (tex10:245), fragment intact | the rule is on BOUND ROUTED learned runs (the draft's 'any bound sixteen-key run' would also take the oracles: 12000, 24000, 27600; same maximum) |
| 24 | tex101:247 | key-load draft rested on a 1-of-2 oracle | rested on an oracle that had bound one of its two seeds (three runs, the failed seed having been continued) | S67 ORC_P16_D128 [(270, None), (271, 12000)]; S68 continued 270 (unbound) | **MATCH** | MATCH (unchanged); line changed elsewhere (tex10:245), fragment intact |  |
| 25 | tex101:247 | S69 order and design | S69, not F's; seeds 270--279, 40{,}800 updates, readings fixed before any run) | readings b93b73c 2026-10-13T20:40:35+00:00; first record ('5bcdf63', '2026-10-14T00:15:48+00:00', 1); docstring unchanged True | **MATCH** | MATCH (unchanged); re-anchored (tex10:245 rewritten) |  |
| 26 | tex101:247 | S69 counts | binds $4/10$ under the recorded single-Adam recipe (three more seeds ended at $0.91$--$0.92$, under the $0.95$ bar, and three stuck near $0.30$) and $0/10$ under SLOW (every seed at $0.33$--$0.36$ | A bound [271, 273, 274, 278] at [12000, 19200, 27600, 31200]; close [275, 277, 279], stuck [270, 272, 276] ([0.318, 0.287, 0.296]); S 0/10 | **MATCH** | MATCH (unchanged); line changed elsewhere (tex10:245), fragment intact |  |
| 27 | tex101:247 | S69: 'the slow start hurt: no SLOW run bound and SLOW ended lower on every seed that bound or came close under the other recipe' | the slow start hurt: no SLOW run bound and SLOW ended lower on every seed that bound or came close under the other recipe | SLOW bound 0/10; SLOW final below single Adam's on the 7 seeds that bound or came close: 7 of 7; above it on the 3 stuck ones ({270: 0.03, 272: 0.04, 276: 0.04}) | **MATCH** | PARTIAL → MATCH; re-anchored (tex10:245 rewritten) | 846d0c8 replaces the interpretation with what the records show, as E's README reads it ('it hurt') |
| 28 | tex101:247 | S69 pre-registered reading | The pre-registered reading is ``no $D = 128$ validity check at 40{,}800'' | rule 4 of the docstring: b_A 4, b_S 0 -> 'no D128 validity check at 40,800' | **MATCH** | MATCH (unchanged); line changed elsewhere (tex10:245), fragment intact |  |
| 29 | tex101:247 | S69: 'two validity seeds drawn from these ten ... about 0.13 (two fresh seeds on one machine about 0.16 at that rate, on both machines 0.03)' | two validity seeds drawn from these ten would both bind with probability about $0.13$ (two fresh seeds on one machine about $0.16$ at that rate, on both machines $0.03$) | b(b-1)/90 = 0.133 (two of these ten seeds, without replacement); fresh seeds at 4/10: 0.16 per machine, 0.0256 on both | **MATCH** | PARTIAL → MATCH; re-anchored (tex10:245 rewritten) | 846d0c8 states both models, as the referee's pass and keyload_options_review.md item 1 asked |
| 30 | tex101:247 | status note: five ways forward | Session~E's status note lists five ways forward, none screened and none adopted | spec_draft_keyload.md at 419368a: 'NOT READY FOR V'; 5 options; 'None is screened and none is adopted here' | **MATCH** | MATCH (unchanged); line changed elsewhere (tex10:245), fragment intact |  |
| 31 | tex101:247 | E's verdict on the lead | is ``inconclusive, weaker than handed back'' | README Batch 20 verdicts: 'Verdict on the batch-19 lead ... inconclusive, weaker than handed back'; 4 vs 0, two-sided p 0.125 | **MATCH** | MATCH (unchanged); line changed elsewhere (tex10:245), fragment intact |  |
| 32 | tex101:247 | batch 21 named 'E's S69, not F's' | batch~21 screened that first (E's S69, not F's; | tex101:258 and tab:batches use S69 for session F's Hebbian-decay screen; E numbered its batch-20/21 screens S68/S69, both F's numbers; lines of tex101 naming S68: [42, 256, 266, 303, 383, 540, 542] | **MATCH** | PARTIAL → MATCH; re-anchored (tex10:245 rewritten) | 846d0c8 qualifies E's S69 at its one use in the text. E's S68 is named only in the provenance table (tex101:540), whose rows are labelled by session, beside F's S68 (tex101:542) |
| 33 | tex101:368 | key-load draft: after the reviews; not reviewed | Session~E's key-load draft was committed after those reviews and then failed its own validity screen | draft first committed f5d3893 2026-10-13T02:54:10; V's spec reviews 3f7d84b 2026-10-12T22:50:37 and 723827b 2026-10-12T23:00:53; V's keyload_options_review.md is 11a78ee 2026-10-19T17:51:52, after Rev 10 (16:15:41) | **MATCH** | MATCH (unchanged) |  |
| 34 | tex101:459 | next-step paragraph | screened on ten, it binds four under the recorded recipe and none under the other | 4/10, 0/10 | **MATCH** | MATCH (unchanged) |  |
| 35 | tex101:459 | 'ten seeds of it cost less than a review' | ten seeds of it cost less than a review | batch 21: 20 runs, about 30 h wall (README); a review's cost is not recorded | **NOT RECORD-CHECKABLE** | NOT RECORD-CHECKABLE (unchanged) |  |
| 36 | tex101:503 | key-load test needs revision; cross-reference | the key-load test a validity check it does not have at its budget (Section~\ref{sec:referee}) | E's own status ('NOT READY FOR V'); sec:referee (tex10:366) only points on to sec:scale; V has not reviewed the draft | **MATCH** | MATCH (unchanged) | the finding is E's (batch 21), not a referee review; tex10:366 says so |
| 37 | tex10:495 REMOVED (replaced at tex101:499) | 'the fifth round's screens have been recomputed there' (rev10/) | the fifth round's screens have been recomputed there | main's docs/review_2026-10/rev10/: recompute_H.py covers S87a, S87c and the recount (its committed output holds an IndexError traceback, then a rerun); no script recomputes E's batches 20-21: first_pass/second_pass rows cite E's README, two cite the JSON stores | **REMOVED** | PARTIAL → REMOVED | was PARTIAL (true for H, not for E). 846d0c8 deletes the sentence; its replacement is the new tex101:499 row below |
| 38 | tex101:44 | 'which recomputed test_delta_recipe, the GPU run and H's screens and read the other screens' reports' (rev10/) | which recomputed \texttt{test\_delta\_recipe}, the GPU run and H's screens and read the other screens' reports | 585b666 rev10/ scripts: ['docs/review_2026-10/rev10/recompute_H.py', 'docs/review_2026-10/rev10/recompute_delta.py', 'docs/review_2026-10/rev10/recompute_gpu.py']; none for E's batches 20-21 | **MATCH** | PARTIAL → MATCH; re-anchored (tex10:44 rewritten) | 846d0c8 says what the checking agent recomputed (H among them) and that it read the other screens' reports (E's README among them) |
| 39 | tex101:540 | provenance (tab:batches): E, batches 20-21 (S68 readings 4acf915, complete f5d3893; S69 readings b93b73c, complete 419368a), commit 419368a, 19; 20 | 20--21 (E) & S68 the sixteen-key runs continued to 86{,}400 (readings \texttt{4acf915}; complete \texttt{f5d3893}); S69 the $D = 128$ oracle's validity (readings \texttt{b93b73c}; complete \texttt{419368a}) & \texttt{419368a} & 19; 20 | batch 20: readings 4acf915, 19 runs, f5d3893 'Batch 20 complete (EXPLORATORY, not a re'; batch 21: readings b93b73c, 20 runs, complete at 419368a (last store commit bb1821b) | **MATCH** | PARTIAL → MATCH; re-anchored (tex10:535 rewritten) | 846d0c8 adds the row (tex101:540). E's S68 and S69 sit in the same table as F's S68 and S69 (tex101:542), each under its session's label |
| 40 | tex101:247 | 'The batch completed after Revision 9's first draft (186 of 186 runs)' | The batch completed after Revision~9's first draft (186 of 186 runs) | batch 19 complete at e3f5809 2026-10-11T04:25:06+00:00 ('explore_out: batch 19 complete (186/186 runs)'); Revision 9's first draft 9bfa85a 2026-10-10T22:16:07+00:00 | **MATCH** | new: MATCH | 846d0c8 names the revision ('this revision's' in tex10:245 meant Revision 9's) |
| 41 | tex101:370 | 10.1's account: 'screened once' on three lines after S87a | ``screened once'' for the all-position trigger on three lines after S87a screened it again | tex10:358, 464, 486 -> tex101:360, 468, 490, rows above | **MATCH** | new: MATCH |  |
| 42 | tex101:370 | 10.1's account: provenance rows for E and H | and provenance rows for E, F, G and H that stopped before the fifth round | tex101:540 (E, batches 20-21) and tex101:541 (H, report 9), rows above | **MATCH** | new: MATCH |  |
| 43 | tex101:370 | 10.1's list of loose items taken (H and E) | the dose screen's ceiling; the key-load validity probabilities for fresh seeds; ``the slow start hurt''; E's S69 distinguished from F's | tex101:69 and 219 (ceiling), 247 (0.16/0.03; 'hurt'; E's S69), rows above | **MATCH** | new: MATCH |  |
| 44 | tex101:499 | conclusion: after the merge the referee checked the fifth round's screens (H's and E's here), the numbers right | and after the merge it checked the rest, the fifth round's screens included, with the same outcome | check_rev10_HE at 846d0c8: True; every count, pair, p and reading of S87a, S87c, the recount and batches 20-21 reproduces (sections H1-E2) | **MATCH** | new: MATCH | replaces the removed tex10:495 sentence |

rows (original, 585b666) 39: DISAGREE 3, PARTIAL 10, NOT RECORD-CHECKABLE 2, MATCH 24
rows (the same 39 against 846d0c8): DISAGREE 0, PARTIAL 1, NOT RECORD-CHECKABLE 2, MATCH 35, REMOVED 1
rows (846d0c8, all 44): DISAGREE 0, PARTIAL 1, NOT RECORD-CHECKABLE 2, MATCH 40, REMOVED 1
status changes: row 4: PARTIAL → MATCH; re-anchored (tex10:67 rewritten); row 10: PARTIAL → MATCH; re-anchored (tex10:217 rewritten); row 11: DISAGREE → MATCH; re-anchored (tex10:358 rewritten); row 12: DISAGREE → MATCH; re-anchored (tex10:464 rewritten); row 13: DISAGREE → MATCH; re-anchored (tex10:486 rewritten); row 15: PARTIAL → MATCH; re-anchored (tex10:536 rewritten); row 27: PARTIAL → MATCH; re-anchored (tex10:245 rewritten); row 29: PARTIAL → MATCH; re-anchored (tex10:245 rewritten); row 32: PARTIAL → MATCH; re-anchored (tex10:245 rewritten); row 37: PARTIAL → REMOVED; row 38: PARTIAL → MATCH; re-anchored (tex10:44 rewritten); row 39: PARTIAL → MATCH; re-anchored (tex10:535 rewritten)
order: H readings before records True; E batch 20 True; E batch 21 True; H seeds only in H's S87 stores True; code SHAs recomputed H True, E b20 True, b21 True
