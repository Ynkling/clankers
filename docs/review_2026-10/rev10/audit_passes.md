# Revision 10: audit of the two passes made over its draft

*Referee V, 19 October 2026. The passes audited are `docs/review_2026-10/rev10/first_pass.md` (on draft `ea3afef`) and `second_pass.md` (on `6e75f69`), with `recompute_delta.py`, `recompute_gpu.py` and `recompute_H.py` and their `_out.txt` files and the directory's `README.md`. All of them were committed at `d2b7abd`, the third version, which is the tex merged at `585b666`. Baselines: Revision 9's final tex `c918ff7`; the v9→v10 diff of the final text. Every number below is printed by `audit_passes.py` in this directory (`python3 -I docs/review_2026-10/rev10/audit_passes.py`). The script uses the standard library and `rlib.py`, reads git objects only, writes nothing, runs no session code and trains nothing. The other session's `recompute_*.py` scripts were read, not run. Where this audit needed one of their computations, it re-implemented it with the referee's definitions.*

Line numbers are tex line numbers. The three drafts have 591 lines each and every change between them replaced a line in place, so a line number means the same line in `ea3afef`, `6e75f69` and `d2b7abd`. No shift needs adjusting.

## Findings, most serious first

1. **One first-pass DISAGREE was not taken, and the second pass passed it.** First-pass row 12.1 and item 1 found that tex:495's "A referee recomputed the numbers of Revisions~8 and~9 and of this revision's additions from the records and found them to reproduce" is false: V never checked the fifth round's screens. That sentence is still in the final text, word for word. The second version added a sentence after it ("... the fifth round's screens await the referee"), and the third version reworded that addition ("... have been recomputed there and not yet refereed by session~V"). So the merged conclusion now contradicts itself within one paragraph. Second-pass row 32 rated tex:495 "MATCH as worded", and its §2 lists item 1 as "resolved | 44, 495". The second pass checked the words that were added. It did not check that the old words had gone.
2. **No pass ran on the merged text.** The third version `d2b7abd` changed tex:44, 481, 495 and 500, and the merge followed 58 s later (16:14:43 → 16:15:41 UTC). Two of those changes are new claims about the checks themselves, and neither holds as written:
   - tex:44, "every number recomputed from the records". By its own source column, the first pass checked 13 MATCH rows only against a session report or README (2.4, 4.2, 4.5, 6.3, 6.4, 6.5, 6.10, 7.5–7.8, 8.4, 10.3). Its 4 NOT RECORD-CHECKABLE rows are not recomputations either. The second pass recomputed nothing; it reuses the first pass's outputs.
   - tex:495, "the fifth round's screens have been recomputed there". `rev10/` holds a script for H's S87a/S87c only. No committed script reads the records of F's S88a/S88b, G's S87, E's batches 20–21 or S75's tenth seed (section J). The first pass's item 1 sends the reader to "`recompute_H_out.txt`, `recompute_delta_out.txt` and §§6–7" for those screens. `recompute_delta_out.txt` contains nothing about them.

   The counts themselves reproduce (section E below), so this is a statement about what was done, not about the results.
3. **Text carried over from Revision 9 whose "this revision" now points at the wrong revision.** None of it is in the added text, so neither pass's scope reached it. tex:42 (in the paragraph now titled "Revision~9") reads "The referee also checked this revision's first draft against the records, and the statements it found wrong are corrected here". The same carry-over appears at tex:245 ("completed after this revision's first draft"), tex:364 ("What it found in this revision's first draft ... commit `9bfa85a`", which is Revision 9's draft, although the paragraph was edited in v10), tex:443 and tex:541 ("this revision's checks (`e8fbb25`, …)"). In Revision 10 a reader takes "this revision" to mean Revision 10. The README has the same text (four occurrences of "this revision's first draft").
4. **Smaller follow-through gaps.**
   - tex:70, the index row for V: "the three drafted specifications are not ready". This is the count that item 9 corrected at tex:44, 366 and 499: four were drafted and V reviewed three. Row 1.21 flagged it ("see 1.10"), but item 9 did not list tex:70.
   - tex:264 still says, as fact, "lost ... to a reboot during the projection's timing, with a GPU job beside it". Row 5.3 rated the GPU job NOT RECORD-CHECKABLE, and tex:426 now attributes it to the author.
   - tex:481: the S75b review passes are "on session~I's branch". Row 11.3 asked for them to be called session I's own reviews.
5. **The passes' own bookkeeping.**
   - The first pass has 123 rows. Counting each row by the first verdict it names gives 92/17/11/3; its stated totals are 91/17/11/4. Rows 2.7 and 5.3 carry two verdicts each.
   - The second pass has 32 rows, not the "31" it states. It has 1 PARTIAL, not 3. Rows 4 and 32 read "MATCH ...; NOT RECORD-CHECKABLE".
   - The first pass's "seven distinct errors" names six kinds of error. Rows 4.3 (a range) and 5.18 ("above") are further distinct errors, and both were DISAGREE.
   - Both passes cite tex:499 for the S75b bullet; it is tex:500.
   - The second pass cites `rev10_first_pass.md` and `worddiff2.txt`, which are not in the repository.
   - The three scripts read placeholder paths ("the main-line clone/results", "branch claude/explore-H/...") and cannot run from the repository.
   - `recompute_H_out.txt` is a splice. It begins with the tail of one run that ends in a `Traceback` (`IndexError` at the line building `arms`), then repeats lines from a second run of an edited script, the committed one with `arms = ["WIN3_SPLIT_D8"]` hard-coded. No single run of the committed script produced the committed output.
6. **Independence and order of the passes cannot be shown from the record.**
   - Every draft commit, and the commit that adds the pass files, carries the authoring session's trailer (`session_01D7HpBudv8or6aFXGqt2fyj`).
   - The pass files arrive only in `d2b7abd`, in the same commit as the third version. Git therefore cannot show that the first pass preceded the second version, or the second pass the third.
   - The drafts are 21 min apart (`ea3afef` 15:49:58 → `6e75f69` 16:11:01) and then 3 min 42 s apart (→ `d2b7abd` 16:14:43).
   - "By a session that did not write them" (tex:44, 495; README) is NOT RECORD-CHECKABLE. In Revision 9 each referee pass was committed on V's own branch, with its own time, before the version that took it.

What the passes got right: every number they marked MATCH that this audit recomputed reproduces, both with the referee's definitions and against the referee's own earlier checks (section 4). Of the 39 follow-through checks in section D, 36 find the corrected wording in place; the three that do not are tex:495 (finding 1), tex:264 and tex:481 (finding 4). Of the second pass's three items, two were taken correctly and the third (the citation) was taken with an overclaim (finding 2).

---

## 1. Coverage of the added text (v9 → final v10)

Method: the added words of each changed line are found with a word diff of `c918ff7` against `d2b7abd`. Every number in them is extracted: counts, fractions, decimals, ranges, intervals, p-values, hashes, and the number words (one … twenty, half, third, fifth). Each pass row is mapped to the tex lines it cites, taken from the line numbers in its statement column or else from its section heading. A token counts as **covered** when a row citing the same line contains it, and as **weak** when it appears only in a row citing another line. Lines 14, 46, 48–52, 56 and 74 are LaTeX only (the longtable). Lines 53–55 are table rows whose added text holds no number.

| tex line (added text) | distinct number tokens | covered on the line | weak | rows citing the line |
|---|---|---|---|---|
| 27 (author line, "Revision 10") | 1 | 0 | 1 | none |
| 37 abstract | 19 | 18 | 1 | P1 1.1–1.6; P2 1 |
| 42 Revision 9 paragraph | 4 | 4 | 0 | P1 1.7, 1.8 |
| 44 Revision 10 paragraph | 17 | 15 | 2 | P1 1.9–1.11; P2 2–5 |
| 60, 64–70 index rows | 36 | 35 | 1 | P1 1.4, 1.5, 1.6, 1.12–1.21; P2 6, 7 |
| 181 headline on a GPU | 24 | 21 | 3 | P1 2.1–2.9; P2 8, 9 |
| 217 H's fifth round | 20 | 18 | 2 | P1 3.1–3.6; P2 10, 11 |
| 245 batches 20–21 | 36 | 34 | 2 | P1 4.1–4.11; P2 12–17 |
| 264 `test_delta_recipe` | 52 | 50 | 2 | P1 5.1–5.19; P2 18 |
| 266, 268 F's S88a/S88b | 46 | 46 | 0 | P1 6.1–6.10 |
| 283, 322, 328 G's S87 | 33 | 32 | 0 | P1 7.1–7.8; P2 19 |
| **301** caption of the grouped-layout table | 5 | **0** | 5 | **none** |
| 336, 347 write rule; R's Phase 5 | 35 | 33 | 2 | P1 8.1–8.9; P2 20 |
| 364, 366 the referee section | 27 | 27 | 0 | P1 1.10, 9.1–9.9; P2 21–24 |
| 377 withdrawn claim | 13 | 6 | **7** | P1 1.4, 6.11, 10.1 ("as §5"); P2 25 |
| 424, 426 corrections; process | 2 | 2 | 0 | P1 10.2–10.5; P2 26, 27 |
| **431** lessons count | 3 | **0** | 3 | **none** |
| 455, 457 two lessons | 10 | 5 | 5 | P1 10.6, 10.7; P2 28 |
| 472 limits | 5 | 5 | 0 | P1 1.6, 10.8; P2 29 |
| 481 S75 / S75b | 14 | 13 | 1 | P1 11.1–11.6; P2 30, 31 |
| 495 conclusion | 14 | 13 | 1 | P1 1.5, 1.6, 6.11, 12.1, 12.2; P2 32 |
| 499 next steps, specifications | 3 | 2 | 1 | P1 1.10, 11.7 (meant 500), 12.3 |
| **500** S75b bullet (final wording) | 1 | 0 | 1 | **none on the final wording** |
| **501, 502** next steps | 1 | 0 | 1 | **none** (12.3 names them in a word) |
| 519, 522 appendix | 13 | 12 | 1 | P1 12.4, 12.5 |

### Statements no row covers

Each item says whether the statement holds, as recomputed here.

| # | tex | statement | status here |
|---|---|---|---|
| U1 | 301 | caption: "at eight streams the pre-registered `test_delta_recipe` (pooled over X and L, 40 pairs) has the delta recipe $35/40$ against the Hebbian recipe $35/40$ and, without a trigger, $38/40$ against $17/40$" | holds (section E: rlib 35, 35, 38, 17) |
| U2 | 431 | "Revision~9 added seven, ... and Revision~10 two" | holds (two new `\paragraph` lessons at 455 and 457) |
| U3 | 44 | (third version) "this revision's own additions were checked in two passes ..., every number recomputed from the records, `docs/review_2026-10/rev10/`" | **does not hold** as written (finding 2) |
| U4 | 495 | (third version) "the fifth round's screens have been recomputed there and not yet refereed by session~V" | **partly**: a script exists for H only (section J) |
| U5 | 481 | (third version) "no learned-arm record was in git until `b649751` at 13:53 UTC" | holds (`b649751` 13:53:09 UTC; the earlier commit of the results file, `86652b1`, holds 0 learned-arm keys; section J) |
| U6 | 500 | (third version) "specified and reviewed, its final driver checked on the CPU; the CUDA CHECKs to rerun on the final driver and the referee's review points to take, then the pilot, on the one remaining GPU" | consistent with tex:481 and first-pass row 11.4; no pass read this wording |
| U7 | 499 | "The register test is the nearest to running." | a judgement; no row |
| U8 | 501, 502 | "A calibrated reading for any further replication phase, and a replication isolated by construction"; "BDH-CQ and Triadic Linear Attention read in full before Revision~11" | plans; covered only by the word "replication; BDH-CQ" in 12.3 |
| U9 | 377 | "binds $7/10$ against the fixed one's $10/10$ (no reading applies)"; the pooled block repeated | holds (F: rlib 10/10, 7/10); row 10.1 says only "as §5" |
| U10 | 457 | "either half alone would have been reported as a three-pair lead" | follows from X $d = -3$, L $d = +3$ (row 10.7) |
| U11 | 217 | "Session~H's fifth round, on fresh seeds" | not checked by any pass; holds over every branch tip (section G: 2600–2619 and 2630–2639 appear in no other record file) |
| U12 | 245, 266 (and 283, 268) | "readings fixed before any run" / "pre-fixed" for batch 21, S88a, S88b, S87 | not checked by any pass beyond naming `40087cb` (6.1); holds by git (section F) |
| U13 | 42, 245, 364, 443, 541 | "this revision's first draft" / "this revision's checks", carried over from Revision 9 | **wrong referent in v10** (finding 3); outside both passes' scope |

---

## 2. Follow-through on the final text (`d2b7abd`)

Every first-pass PARTIAL or DISAGREE and every second-pass item. "Old" is the `ea3afef` wording, unless a row says it is `6e75f69`'s. "Final" is `d2b7abd`'s.

| item (rows) | tex | old | final | taken? |
|---|---|---|---|---|
| P1-1 (1.9) | 44 | "the referee's checks of all of it" | "the referee's checks of the first three of these (…; the fifth round's screens have not been refereed; …)" | yes |
| **P1-1 (12.1)** | **495** | "A referee recomputed the numbers of Revisions~8 and~9 and of this revision's additions from the records and found them to reproduce" | **unchanged**; a later sentence now says the screens are "not yet refereed by session~V" | **NO: contradiction left; the second pass marked it resolved** |
| P1-2 (10.4) | 424 | "session~I's pre-run note for S75 said no window-gate run preceded it (three had finished on the card; …)" | item removed from "Statements corrected after Revision~9" | yes |
| P1-2 (11.6) / P2-1 (31) | 481 | "three of the ten window-gate runs had finished on the card before the note was committed, though no learned-arm record was readable anywhere until thirteen hours later" | "the card's timeline puts up to three before it, which the records cannot fix (the referee: between none and three); no learned-arm record was in git until `b649751` at 13:53 UTC, and session~I states that the note's decisions were made blind. The note itself was not edited; its correction is in report 2." | yes (the `6e75f69` "readable anywhere" was fixed at `d2b7abd`) |
| P1-3 (4.6, 10.6) | 245, 455 | "one of three seeds" | "one of its two seeds (three runs, the failed seed having been continued)"; 455 "one of its two seeds" | yes |
| P1-4 (4.1) | 245 | "(six at each width)" | "(six one-to-one at $D = 128$; six one-to-one and four merged at $N = 256$)" | yes |
| P1-4 (4.3) | 245 | "the one-to-one runs creeping by $0.02$--$0.10$" | "... and the $N = 256$ ones moving by $-0.02$ to $+0.41$" | yes. Precision note: the recorded changes are −0.0146 to +0.4136 (section I), i.e. −0.01 to +0.41 to two places; "−0.02" is the difference of the README's rounded endpoints (0.72→0.70) |
| P1 (4.2) | 245 | "($0/6$ and $0/6$; …)" | "… the merged four $0/4$, descriptive" | yes |
| P1-5 (5.17, 5.18) | 264 | "$35/40$ here, is within two of the noisy copy's $38/40$ … and above its $18/20$" | "binds $35/40$ here ($0.875$), against the noisy copy's $38/40$ … and $18/20$ …; … the two copies have not been compared on the same seeds" | yes |
| P1-6 (7.2) | 283 | "built in the first updates" | "computed once before training from a sample of 4096 training sequences" | yes |
| P1-7 (1.6, 1.20, 10.8) | 37, 44, 69, 472, 495 | "from its specification" | "its delta arm from the specification alone" (472: "… and the other two with its earlier recipe") | yes |
| P1-8 (11.4) | 481 | "its driver written and checked on the CPU and on both cards" | "its final driver checked on the CPU (the cards' CUDA CHECKs were run on an earlier driver and are to be rerun), the referee's review points still to take" | yes |
| P1-8 (11.7) / P2-3 | 500 (cited as 499) | "specified, reviewed and checked, with its pilot to run on the one remaining GPU" | "specified and reviewed, its final driver checked on the CPU; the CUDA CHECKs to rerun on the final driver and the referee's review points to take, then the pilot, on the one remaining GPU" | yes, in the third version; checked by no pass |
| P1-9 (1.10) | 44, 366, 499 | "the three main-line specifications" | "three of the four specifications the sessions drafted (three main-line tests and one exploratory screen)" | yes |
| P1 (1.21) | **70** | "the three drafted specifications are not ready" | **unchanged** | **no** (not listed in item 9) |
| P1-10 (3.4) | 217 | "``keeps it'', ``no cost'' and ``aims'' all apply" | "``keeps it'' and ``aims'' apply, ``costs binding'' does not (printed ``no cost by the rule'')" | yes |
| P1-10 (1.16) | 67 | "within one pair ($24/30$ against $26/30$ …)" | "within one pair on twenty new seeds ($16/20$ against $17/20$; $24/30$ against $26/30$ over two screens)" | yes |
| P1-11 (9.2, 9.3) | 366 | "(66 rows: one disagreement …; the pooled D1, D2 and D4 pairs recomputed independently)" | "(66 rows: 55 match, 7 partial, 3 not record-checkable and one disagreement …; the pooled D1 and D4 pairs, the D2 no-room count and D4's $p$ recomputed independently)" | yes |
| P1-12 (9.1) | 364 | "The sixth and seventh passes found one item, …" | "The sixth pass left two items, … and the scope of R's 44 unstated details; the seventh version fixed both …" | yes |
| P1-13 (4.10) | 245 | "the one lead …, stands as it was" | "Session~E's own verdict on the lead …, is ``inconclusive, weaker than handed back''" | yes |
| P1-14 (8.2) | 347 | "R was given `specs/test_split_copy.md`" | "… and the test's docstring" | yes |
| P1-14 (6.11) | 377 | "collapses to zero" | "collapses to about zero everywhere but the first layer's value positions" | yes (495 keeps "collapses on a screen", which is not wrong) |
| P1-14 (2.7) | 181 | "the recipe binds $57/60$, the no-split arm $21/60$ and the recurrent gate $0/60$" | "report 3 states that the GPU run is never pooled with it" | yes |
| P1-14 (2.9) | 181 | "which left the project after this run" | "… after this run and S75b's CUDA CHECKs" | yes |
| P1-14 (10.5) | 426 | "to the OOM killer with a GPU job beside it on a 15\,GB laptop" | "to a reboot that the author attributes to the OOM killer, with a GPU job beside it" | yes |
| P1 (5.3) | **264** | "lost before any run to a reboot …, with a GPU job beside it" | **unchanged** | **no**: as fact here, attributed at 426 (not an item) |
| P1-14 (3.5) | 217 | "the 60 recorded runs" | "the 60 recorded split-arm runs" | yes |
| P1-14 (9.8) | 366 | "not reviewed, having failed its own validity screen first" | "was committed after those reviews and then failed its own validity screen …; it has not been reviewed" | yes |
| P1 (11.3) | 481 | "revised through further review passes on session~I's branch" | unchanged | partly (not an item) |
| P2-2 (4, 32) | 44, 495 | (`6e75f69`) "checked once by a session …"; "the fifth round's screens await the referee" | "checked in two passes … every number recomputed from the records, `docs/review_2026-10/rev10/`"; "have been recomputed there and not yet refereed by session~V" | **taken, with an overclaim** (finding 2) |

---

## 3. What changed between the version a pass checked and the final text

- **`ea3afef` → `6e75f69`: 19 lines changed**: 37, 44, 67, 69, 181, 217, 245, 264, 283, 347, 364, 366, 377, 424, 426, 455, 472, 481, 495. This matches the second pass's "19 changed lines", and every one is cited by at least one second-pass row. The second pass did not check that wording it had asked to remove was gone. tex:495's first sentence was left in place (finding 1).
- **`6e75f69` → `d2b7abd`: 4 lines changed**: 44, 481, 495, 500. No pass ran on `d2b7abd`. The word diffs:
  - 44: [-refereed, and-] {+refereed;+} … [-once-] {+in two passes+} … {+, every number recomputed from the records, `docs/review_2026-10/rev10/`+}
  - 481: [-readable anywhere-] {+in git+} … {+`b649751` at+}
  - 495: {+, in two passes,+} … {+(`docs/review_2026-10/rev10/`)+} … [-await the referee.-] {+have been recomputed there and not yet refereed by session~V.+}
  - 500: [-specified, reviewed and checked, with its pilot to run-] {+specified and reviewed, its final driver checked on the CPU; the CUDA CHECKs to rerun on the final driver and the referee's review points to take, then the pilot,+}
- **Outside the tex.** The README was rebuilt at `d2b7abd` (68 lines of diff), and `STATUS.md` (`f342d1d`) and `CITATION.cff` changed in the same revision. No pass read any of them. The README carries all of the tex's remaining problems: "of this revision's additions" (1), "every number recomputed" (1), and "this revision's first draft" (4). Its other corrected phrases ("one of three seeds", "within two of", "57/60", "readable anywhere" and so on) are gone. The README's "no cost" and "collapses to zero" occur once each; their context was not audited here.

---

## 4. The recompute scripts

**Definitions they use**, all copied in each script:
- bound = `ok` and recorded `transition` not None;
- routed = recorded `end.one_to_one` and min(`end.stream_acc`) ≥ 0.9;
- BOUND ROUTED = both;
- McNemar = exact one-sided binomial tail;
- Wilson at z = 1.95996;
- Newcombe's method 10 with the continuity-corrected φ (`max(ad−bc−n/2, 0)`);
- bands at ⌈0.9n⌉ and ⌈0.5n⌉;
- D1's labels by d = c − b, NO_ROOM 36;
- D3 as a one-sided sign test on seeds where both split arms are BOUND ROUTED;
- CHECK 174 re-implemented as equal curves up to the first firing;
- "probe alone" re-implemented from the trigger's constants (rise 0.02, threshold 0.95, gap 4800, cap 3, first check 4800), using the check's `acc` field.

Every one of these matches the main line: `test_stream_recipe.routed`/`outcome`, `fail_class_v2.outcome_v2`, and `test_delta_recipe`'s bands and NO_ROOM.

**Divergence from the referee's method.** "An independent outcome function" (first pass, Method) overstates it. The scripts take the recorded `transition` and the recorded `end.one_to_one`, which is the code's own map (argmax of the mean read-gate mass at value positions). The referee's `rlib` recomputes the transition from the curve ("first evaluation ≥ 0.95 that holds to the end") and the map from `end.stream_gate`. Section E compares the two on every record. They agree on every record of `test_delta_recipe` X and L, the GPU run, H's S87a and S86, and F's S88a: 0 transitions differ and 0 one-to-one flags differ. The divergence costs nothing here, but the scripts are not independent of the code at the outcome level. Other notes:
- `recompute_H`'s guard count reads `checks[].map_before` and counts a map as one-to-one when its 8 entries are distinct, as the referee's `check_onetoone_v5.py` does. It counts the exact-copy arm only. The reset arm (`WIN3_RESET_D8`, 78 firings over 30 runs) also has 0 firings on a one-to-one map, which the tex's "split-arm" scoping leaves out correctly.
- No bug changes a number.
- Defects of form: the placeholder paths make all three unrunnable as committed, and `recompute_H_out.txt` is spliced from two runs, one of them a crashing earlier version (section H).
- There is no committed script for F, G, E or S75 (section J).

**Agreement with the referee's own numbers.**

| quantity | recompute_* | referee | this audit (rlib) |
|---|---|---|---|
| delta pooled counts | 35, 17, 35, 38 | check_delta_recipe_L.md: same | 35, 17, 35, 38 |
| D1 / D2 / D4 | 5 vs 5, d 0, +0.000 [−0.162, +0.162]; 0 vs 3, p 1; 19 vs 1, p 2.003e−5 | same (newcombe10 to 0e+00) | 5 vs 5; 0 vs 3; 19 vs 1, p 2e−5 |
| D3 | 30; 24/5/1; p 0.00027; medians 13200 vs 18600 | same (146596/2^29) | same |
| per machine D1 | X −3 (+0.150 [−0.082, +0.374]); L +3 (−0.150 [−0.374, +0.082]) | L row: same; X: rev9 X check | — |
| bands (Wilson) | [0.835, 0.986]; [0.739, 0.945] ×2; [0.285, 0.578] | same | same |
| probe-blind / alone | X HEBB 1 of 181, 3; DELTA 1 of 142, 5; L HEBB 3 of 170, 2; DELTA 6 of 149, 9 | L: same; X: the regenerated X log, same (the in-run X report's "1 of 201 / 1 of 162" is the pre-Amendment-2 count) | — |
| cost three | L 1581, 1582 (B-notR 36000), 1591; two-sided 0.25 | same | — |
| GPU | 19/20, 6/20, 0/20; 19, 13, 6 vs 0; 1.91e−6, 1.22e−4, 0.0156; CEIL_C 6000, 4800; code sha 05bbaaaac7a3; secs 15.03 h | check_I_report3.md: same (15.025 h) | 19, 6, 0 |
| guard | 60 runs, 126 firings (47, 38, 22, 19), 0 one-to-one | rev9/check_onetoone_v5_out.txt: same | same |
| H S87a / S86 | 16/20, 17/20, 1 vs 2; 8/10, 9/10, 0 vs 1; 42 firings, key 42, all 40 | not refereed | same |
| F S88a, S88b; G S87; E S69; S75 seed 839 | no script | not refereed | F 10/7/3 and 9, 6, 4, 5 (perfect gates 2/2 each); G 0, 20, 17, 0, 3, 10/10 (delta), ORACLE 2/2; E 4/10 (271, 273, 274, 278) and 0/10; S67 oracle D128 one of two seeds (270 None, 271 12000); S75 WIDE_SLOW 0.5300–0.5360, WIDE_SPLIT 0.5293–0.5365, max η² 0.00061 / 0.00058, 0 one-to-one |

Spot checks of rows that cite only a report (section I) reproduce from the records:
- learned-β perfect gate 30000 and 21600, against 1200 for β = 1;
- learned-arm failures one-to-one at 0.315–0.779, plus one merge (2405);
- S88b unbound runs ending at 0.477–0.511;
- batch 20 D128 changes +0.018 to +0.102.

---

## 5. Method: what the passes did not do

Compared with the referee's checks in Revision 9 (`revision_9_check.md`, `rev9/*`):

1. **No pass on the version that was merged.** Revision 9's seventh pass passed the final text. Revision 10 merged a version (`d2b7abd`) with four changed lines that no pass read, two of them new claims about the checks (section 3).
2. **No check that removed wording was removed.** The `resolution_*.md` files of Revision 9 tracked each item to its final wording. The second pass checked the added words only, which is how finding 1 got through.
3. **Pre-registration order of the screens not checked.** Only row 6.1 names the readings commit (`40087cb`), and no row compares a readings commit with the first record. This audit did (section F), and every order holds by git:
   - S88a/S88b readings `40087cb` 12 Oct 23:07, first records `e8278ee` 13 Oct 01:48 and `dfc71c5` 14 Oct 17:21;
   - G S87 `3c89f87` 22:48:17, first record 23:08:31;
   - H S87a/S87c `f4fd535` 22:45:48, first record 23:00:50;
   - batch 20 `4acf915` 11 Oct 21:27, first record 22:16;
   - batch 21 `b93b73c` 13 Oct 20:40 (last edit `9cdd1b3` 21:49), first record 22:13.

   No script changed after its first record, and each screen's records carry one code hash. Clock bounds were not checked, here or by the passes. The referee bounded the session clocks by git ancestry in `check_I_report3.md`.
4. **Seed freshness not checked** (tex:217 "on fresh seeds"). The referee scanned all history for the GPU and R seeds. This audit scanned every remote branch tip (section G): 2400–2409, 2420–2429, 2600–2619, 2630–2639 and 2700–2729 appear in no record file but the screen's own. All history was not scanned.
5. **Code identity not checked for the screens.** The referee recomputed code hashes by each session's definition and compared modules by AST. The passes did neither. Each screen's records carry a single code hash, but no one has tied that hash to the code at the readings commit.
6. **Outcomes not re-derived from raw fields.** The scripts read the recorded `transition` and `end.one_to_one` (section 4).
7. **Not reproducible as committed.** The scripts use placeholder paths and one output is spliced. Five screens have no script. The second pass cites files that are not in the repository. Revision 9's referee scripts read git objects at pinned commits and regenerate their `.md`/`_out.txt` in full.
8. **Scope limited to added text.** Carried-over sentences whose referent changed with the new revision were not read (finding 3). The README, `STATUS.md`, `CITATION.cff` and the PDF were not checked. The referee's `check_pdf_v10.py` now covers the PDF.
9. **Report-only rows.** 13 MATCH rows rest on a session report or README, not a record (section A). Section I spot-checks four of them and they reproduce. The tex's "every number recomputed from the records" does not describe them.
10. **Independence not recorded** (finding 6).

---

## Appendix: the passes' own counts (section A)

| file | rows | MATCH | PARTIAL | DISAGREE | NRC | stated |
|---|---|---|---|---|---|---|
| first_pass.md | 123 | 92 (2.7, 5.3 also NRC) | 17 | 11 | 3 | 123: 91 / 17 / 11 / 4 |
| second_pass.md | 32 | 31 (4, 32 also NRC) | 1 | 0 | 0 | "31": 27 / 3 / 0 / 1 |
