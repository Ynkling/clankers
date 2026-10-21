# Revision 10.1 against the referee's pass over Revision 10

*Referee V, 20 October 2026. Generated with `check_rev10_1.py` in this directory (`python3 -I docs/review_2026-10/rev10/check_rev10_1.py`, output in `check_rev10_1_out.txt`). The script uses the standard library, `pdftotext`/`pdfinfo` and `rlib.py`. It reads git objects only, trains nothing, runs no session code and writes nothing but stdout. Section numbers below (§0–§6) refer to the script's output.*

**What was checked.**
- **The revision.** `multichannel_hebbian_report_v10.tex` and `.pdf` at `846d0c8` (19 Oct 18:33:26 UTC, 596 lines), cited as `tex101:NNN`.
  - It was merged to the main line at `ed9ca2b6` (18:33:46) and to the default branch at `264e227` (18:33:50).
  - The tex, PDF, README, STATUS.md and CITATION.cff blobs at `264e227` equal `846d0c8`'s (§0).
- **The baseline.** Revision 10 as merged, `585b666` (591 lines), cited as `tex10:NNN`.
- **The pass.** `referee_pass.md` §1 and §2, with `check_rev10_A/FG/HE.md` and `audit_passes.md`.
  - The copies merged into `846d0c8` are byte-identical to this branch's (§0).
  - The copy merged is `480a2de`'s (§4).
- **The diff.** 22 changed hunks (§1). Most are lines replaced in place. Two paragraphs were inserted: the Revision 10.1 paragraph after tex101:44 and the section-15 paragraph at tex101:369–370. The provenance block went from three lines to four (tex10:536–538 → tex101:540–543).
- **The method.**
  - **The wording.** For every item, the script asserts three things: the old wording is on its tex10 line, it is gone from the whole of tex101, and the new wording is on its tex101 line (§2). All 40 anchors were found.
  - **The numbers.** Every new number was recomputed from the records (§3). Every commit hash in the changed provenance rows was resolved, dated and placed on its branch, and each session's branch was read at `846d0c8`'s commit time (§4).
  - **The PDF.** It was checked by `check_pdf_v10.py`'s method, re-implemented in the script (§5).

## Counts

| | entries | RESOLVED | PARTLY | NOT | NEW ERROR |
|---|---|---|---|---|---|
| §1, false statements | 9 items, 16 places | **9 (16 places)** | 0 | 0 | 0 |
| §2, loose statements | 15 actionable bullets (and one with no action) | 10 | 4 | 1 | 0 |
| added text (tex101:44–46, 72, 360, 369–370, 447, 468, 499, 540–546) | 12 rows | 8 | 1 | — | **2** (minor) |
| outside any item | — | — | — | — | 4 residuals in unchanged text, 3 outside the tex |

**Every new number reproduces from the records.** That covers 30 gated runs; 24/30 against 26/30; 3 mistargets in 63 firings; 38%; 0.16 and 0.03; −0.01; 13 of 20 and 8 run-layers; 3600 and 1200–6000; 228 rows; and the run counts 19, 20, 84, 78 and 142. Every hash exists, predates `846d0c8`, and is what its row says it is. **The PDF shows the tex.**

## Verdict

**Revision 10.1 stands on §1.** Every false statement is corrected in all sixteen places, with numbers that reproduce.

What remains is small, and none of it is a number of a claim.

**Carried from §2:**
- tex101:42 still reads "this revision's first draft".
- tex101:44's "fifth round (E's batches 20 and 21)" label is unchanged.
- tex101:381 leaves out one exception.
- "About two and a half days" is high: the specification's run times give 2.26–2.39 days.
- "F's fourth round" conflicts with the text's own numbering.

**Two minor new errors:**
- tex101:546 cites the superseded pass `2c250bd`, which says "eleven false statements".
- tex101:370 lists eight of the nine statements.

**Four residuals in unchanged text that the pass did not list:**
- the I and R provenance rows (tex101:544, 545);
- tex101:408's "This revision's first draft";
- tex101:447's "the referee now reads … every draft before it is merged", which Revision 10.1's own 20-second merge contradicts.

**Outside the tex:**
- STATUS.md:76 keeps both §1.2 and §1.3 claims.
- STATUS.md:103 keeps "three days".
- `rev10/README.md` keeps "by a session that did not write it".

A Revision 10.2 (or an erratum) would close all of these in about a dozen line edits.

## 1. referee_pass.md §1 (false statements)

| item | old (tex10) | new (tex101) | status | note |
|---|---|---|---|---|
| 1.1 | 495: "A referee recomputed the numbers of Revisions~8 and~9 and of this revision's additions from the records and found them to reproduce" | 499: "… of Revisions~8 and~9 from the records and found them to reproduce …; of this revision's additions it had checked `test_delta_recipe`'s L half and pooled block, the GPU run and R's Phase~5 before the draft, and after the merge it checked the rest, the fifth round's screens included, with the same outcome (the numbers right, nine statements false, corrected in Revision~10.1 …)" | **RESOLVED** | The contradicting last sentence ("… not yet refereed by session~V") is deleted. The three checks named are `round6_A3.md`'s. |
| 1.2 (44) | "every number recomputed from the records, `docs/review_2026-10/rev10/`" | "which recomputed `test_delta_recipe`, the GPU run and H's screens and read the other screens' reports, `rev10/first_pass.md`, `second_pass.md`" | **RESOLVED** | It matches the scripts in `rev10/` (delta, GPU, H only). Both cited files exist at `846d0c8`. |
| 1.2 (495) | "the fifth round's screens have been recomputed there" | deleted | **RESOLVED** | |
| 1.3 (44) | "checked in two passes by a session that did not write them" | "checked in two passes by a separate checking agent run from the author's own session, whose files git cannot order before the third version" | **RESOLVED** | It now states what git shows. "Separate agent" is still the author's account, not shown by git (minor). |
| 1.3 (495) | "in two passes, by a session that did not write them" | deleted | **RESOLVED** | |
| 1.4 | 481: "all but one of the 29 (a register run, at $0.29$)" | 485: "all but one of the 30 (a register run, at $0.29$)" | **RESOLVED** | Recomputed at `4371824`: 30 gated runs (WIDE_SPLIT 10, REG3 10, WIDE_SLOW 10). One run is above 0.27 at 3600: REG3 839 at 0.291. The L2 probe at 1200 reads 0.352–0.612. |
| 1.5 (358) | "screened once, session~H's S86, $8/10$ against $9/10$" | 360: "screened twice, session~H's S86 and S87a, $24/30$ against $26/30$" | **RESOLVED** | S86: 8/10 vs 9/10 (0 vs 1). S87a: 16/20 vs 17/20 (1 vs 2). Pooled: 24/30 vs 26/30. |
| 1.5 (464) | "screened once (session~H's S86): it keeps binding within one pair … ($8/10$ against $9/10$) and its one mistarget in 21 firings cost that run" | 468: "screened twice (session~H's S86 and S87a): it keeps binding within one pair of the key-position rule on each screen ($8/10$ against $9/10$; $16/20$ against $17/20$; $24/30$ against $26/30$ pooled), and of its three mistargets in 63 firings two cost their run" | **RESOLVED** | ALLMASS has 21 + 42 = 63 firings, with the own pair missed 1 + 2 = 3 times. **S86 2334** (19200) failed (MERGED) where BASE bound. **S87a 2611** (14400) still bound. **S87a 2618** (19200) failed (MERGED) where BASE bound. "Cost" is attribution, as before. |
| 1.5 (486) | "screened once (S86) and kept binding within one pair." | 490: "screened twice (S86, S87a) and kept binding within one pair on each." | **RESOLVED** | |
| 1.6 | 266: "within about 1200 updates of its learning rate reaching $10^{-3}$ (by 3600 in the first arm, 3600--8400 in the slow one …)" | 268: "below $0.1$ in every layer's median, by 3600 in the first arm (whose rate is $10^{-3}$ from the start) and 1200--6000 updates after the switch in the slow one" | **RESOLVED** | DELTA_LEARNED crosses at 3600 in 10/10 runs. SLOW crosses at switch + 1200 ×4, 2400 ×2, 3600 ×3 and 6000. Minor: "in every learned run" also covers the learned perfect gate (6000, 8400), which neither clause names. |
| 1.7 | 283: "a penalty on the gate's own distance from uniform at every position" | 285: "a penalty on each position's gate's distance from the sequence's own mean gate" | **RESOLVED** | Matches KL(g_t ‖ ḡ) (`check_rev10_FG.md`). |
| 1.8 E | 535: no row for batches 20–21 | 540: "20--21 (E) & S68 … (readings `4acf915`; complete `f5d3893`); S69 … (readings `b93b73c`; complete `419368a`) & `419368a` & 19; 20" | **RESOLVED** | The stores hold 19 and 20 runs. `4acf915` is the batch 20 readings; `f5d3893` is "Batch 20 complete"; `b93b73c` is the batch 21 readings; `419368a` is "Batch 21 complete" and is the branch head at `846d0c8`'s time. |
| 1.8 H | 536: "… (`5b5deec`, round 5) & `5b5deec` & …; 32" | 541: "S87a …, S87c …, and the one-to-one recount … (report 9, `347bd25`) & `347bd25` & …; 32; 84" | **RESOLVED** | 42 + 42 = 84. `347bd25` is report 9 and was H's last report at `846d0c8`'s time; everything after it on H is S89 progress. There are 8 groups and 8 counts. |
| 1.8 F | 537: "… (report 7, `d5851b1`) & `d5851b1` & …; 42" | 542: "…; S88a the write strength, S88b the horizon (report 8, `a199d5e`; readings `40087cb`) & `a199d5e` & …; 42; 78" | **RESOLVED** | 34 + 44 = 78. `a199d5e` is F's last report; F's later commits (`50dc8e4`, `8ba99bb`, S89 progress) are not reports. |
| 1.8 G | 538: "… (report 6, `076005d`; spec `b2f9907`) & `076005d` & …; 82" | 543: "…; S87 hinges that read no task labels (report 7, `e46799a`; spec `3c89f87`) & `e46799a` & …; 82; 142" | **RESOLVED** | 142 runs. `e46799a` is G's last report; `7ab98c1` is the S89 readings. |
| 1.9 | 70: "the three drafted specifications are not ready" | 72: "the three specifications it reviewed are not ready" | **RESOLVED** | |

## 2. referee_pass.md §2 (loose statements)

| bullet | old (tex10) | new (tex101) | status | note |
|---|---|---|---|---|
| L's lost start | 264: "to a reboot during the projection's timing, with a GPU job beside it" | 266: "…, which the author attributes to a GPU job running beside it" | **RESOLVED** | Now attributed. Minor: the author's account (`test_delta_recipe.py`:133, tex101:430) is "the OOM killer with a GPU job beside it", not the GPU job itself. |
| "with the date" | 424: "Each is corrected where it stands, with the date." | 428: "Each is corrected where it stands (`76851eb`)." | **RESOLVED** | |
| "robust" | 67: "the key term's constants are robust;" and 217: "(S87c; ``robust'', and the drafted constants stand)." | 69: "robust on a ceiling screen;" and 219: "… the drafted arm was at the ceiling, so the screen could only have shown a worse variant, and a variant at a true rate of $0.8$ still reads ``robust'' $38\%$ of the time" | **RESOLVED** | P(≥ 9/10 at 0.8) = 0.376; at 0.9 it is 0.736. |
| validity probability | 245: "would both bind with probability about $0.13$," | 247: "two validity seeds drawn from these ten would both bind with probability about $0.13$ (two fresh seeds on one machine about $0.16$ at that rate, on both machines $0.03$)" | **RESOLVED** | 4·3/90 = 0.133; 0.4² = 0.160; 0.4⁴ = 0.0256. |
| "only delays" | 245: "the slow start only delays" | 247: "the slow start hurt: no SLOW run bound and SLOW ended lower on every seed that bound or came close under the other recipe" | **RESOLVED** | ORC_S is 0/10. On the 7 seeds that bound or reached ≥ 0.9 under single Adam, SLOW ended 0.55–0.66 lower. |
| S69 | 245: "screened that first (S69," | 247: "(E's S69, not F's;" | **RESOLVED** | Minor, new: the new provenance row (tex101:540) names E's "S68" and "S69" one row above F's "S68 premise, S69 Hebbian decay" (tex101:542), without "E's". |
| fifth-round label | 44: "fifth round (E's batches~20 and~21, …" | 44: unchanged | **NOT** | Label only. Batch 20's prompt (prompts:883) precedes the round-5 header (prompts:918). |
| −0.02 | 245: "$-0.02$ to $+0.41$" | 247: "$-0.01$ to $+0.41$" | **RESOLVED** | The records give −0.0146 (seed 274) to +0.4136 (276). |
| F's S88 description | 266: "never diverges by stream, … stays of order 1 only at the first layer's value positions"; 377: "collapses to about zero everywhere but the first layer's value positions" | 268: "at the end it does not diverge by stream in any run, and where it stays of order 1 it is mostly at value positions (the first layer's in 13 of 20 runs, a later layer's in 8 run-layers, and the context position in one)"; 381: "collapses to about zero except at some value positions" | **PARTLY** | 268 is exact: 0 runs diverge at the end; layer-1 VAL ≥ 0.5 in 13/20; layers 2–3 VAL ≥ 0.5 in 8 run-layers; CTX 0.72 in one (2403, L3). 381 leaves out that CTX exception, which is not "about zero" and not a value position. |
| memory horizon | 487: "its reading on the horizon itself read ``it does not'';" | 491: "… and the direct horizon screen (S88b) is untested by its rule, with its failures stream confusions at $0.5$ rather than faded memory;" | **RESOLVED** | |
| limits item 1 | 485: "The hinge reads key identities;" | 489: "The hinge reads key identities, and the one label-free hinge found so far, on a frequency table of token ids, discovers $17/20$ on a task where that table selects the keys and the contexts (S87);" | **RESOLVED** | |
| its caveat travels | — | 489, as above ("where that table selects the keys and the contexts") | **RESOLVED** | |
| S75b "three days" | 481: "about three days for the second stage," | 485: "about two and a half days for the second stage by the specification's own run times," | **PARTLY** | The spec's Cost table gives 2×1.15 + 10×1.0 + 10×1.25 + 10×1.25 + 10×(1.7–2.0) = 54.3–57.3 h, which is **2.26–2.39 days**, not 2.5 (60 h). STATUS.md:103 still says "about three days", and tex101:370 calls this "the author's account … stated as such". |
| carried-over referents | 42, 245, 364, 443, 541 "this revision's first draft / checks"; README, four places | 247, 366, 447, 546: "Revision~9's"; **42 unchanged**; README:64 and :378 unchanged | **PARTLY** | 4 of 5 tex places done, and tex101:42 still reads "The referee also checked this revision's first draft against the records". README:64 (= tex 42) and README:378 (= tex 408, below) remain, plus README:497 ("its passes over this revision's draft (`revision_9_check.md`, …)"). |
| "close F's programme" (after the merge) | 268: "The two screens close session~F's programme" | 270: "The two screens closed session~F's fourth round" | **PARTLY** | The staleness is fixed: F's branch held eight reports at `846d0c8`'s time; the rest is S89 progress. But "fourth round" conflicts with tex101:44 ("the exploratory sessions' fifth round (… F's S88 …)") and with prompts:1029 ("Session F — round 5 (S88 …)"). It is F's fourth only if one counts the rounds F took part in (1, 2, 3, 5), which the text never does. |
| round-6 screens postdate | — | — | (no action) | |

## 3. The text 10.1 added

| tex101 | statement | against | status | note |
|---|---|---|---|---|
| 27 | author line "Revision 10.1" | — | OK | Also in CITATION.cff, STATUS.md's first paragraph and the README header. |
| 44 | "the referee's checks of the first three of these before the draft (`round6_A3.md`) and, after the merge, of the whole revision and of the fifth round's screens (`rev10/referee_pass.md`; …)" | referee_pass.md:7: "The scope: the text added since Revision 9" | OK (loose) | "Of the whole revision" overstates the pass's scope, which was the added text plus the carried-over referents. tex101:499's "it checked the rest" is exact. |
| 46 | the 10.1 paragraph | referee_pass.md:23–37 | **PARTLY** | **"found every number of every claim to reproduce":** the pass says "every pre-registered claim" (tex101:370 has it right). **The list of nine** ("an account of who had checked what, a run count and a sentence …, two descriptions …, four provenance rows") omits §1.9, as referee_pass.md:33–37 does (see the last section). **"corrects them and the looser statements the pass listed, and nothing else":** 1 §2 bullet is not taken and 4 only partly. |
| 72 | V's index row: "after the merge, Revision~10 itself: the numbers right, nine statements false (10.1)" | pass | OK | |
| 360 | heading "What it asked for, and what Revision~9 did with it" (was "this revision does") | — | OK (loose) | This is a referent fix the pass did not list. The paragraph now cites S87a, which is Revision 10 work, under "what Revision 9 did". |
| 369–370 | "What it found in Revision~10 as merged": 228 rows; every number of every pre-registered claim and every count, pair, p-value and reading of the screens reproduce; readings before records; the PDF shows the tex; nine statements in sixteen places; what the passes missed (five), flagged (two) and could not see (two) | referee_pass.md; 82 + 107 + 39 = 228 | **NEW ERROR** (minor) | Two errors. (a) The enumeration of "nine statements" names eight; the index row (§1.9) is missing. (b) The looser-ones list says "the author's account … of the S75b stage's length stated as such", but tex101:485 now cites "the specification's own run times", not the author. Everything else in the paragraph matches the pass. |
| 447 | "… and to Revision~10, whose draft a separate checking agent had passed twice and whose merged text the referee then found to hold nine false statements of its own, none a number. A checking agent that recomputes the numbers is not a referee: it did not read the merged text, did not verify that its own items had been taken, and skipped the pre-registration, seed and provenance checks." | referee_pass.md §3 "In short" | OK | See the next section for the sentence after it. |
| 468 | new numbers | records | OK | See 1.5 (464). |
| 499 | see 1.1 | | OK | |
| 540–543 | E, H, F, G rows | git, records | OK | See 1.8. All 29 distinct hashes on these rows exist, predate `846d0c8` and are on their session's branch. Commit columns are each session's last report at 18:33:26. |
| 546 | V row: "…; round 6's A3 (`0e70640`); Revision~10's pass (`2c250bd`) & `2c250bd` & 2 reproductions" | git | **NEW ERROR** (minor) | `0e70640` is right: it adds `round6_A3.md`, 13 Oct. `2c250bd` (18:07:36) is the first version of the pass, which reads "**Eleven** statements are false". `480a2de`, 18 s later, rewrote it to "Nine statements, in sixteen places". `480a2de`'s copy is the one merged in `846d0c8` and the one tex101:46/370 quote. V's last report at `846d0c8`'s time was `3ae815b` (`screen_review_s89_F.md`, also merged in `846d0c8`). The commit column should be `480a2de` at least. |
| 270 | "closed session~F's fourth round" | | see §2 | |

## 4. Not covered by any item

In unchanged text, missed by the referee's pass and still standing in 10.1:

| tex101 (tex10) | statement | records | note |
|---|---|---|---|
| 544 (539) | I row: "… report 2, `03bffdf`, `559b942`) & `559b942` & pilot; 41" | S75's store has 41 runs at `559b942` and **42** at `4371824` (seed 839). report 2 was updated after `559b942`, at `d552c6c` (13 Oct 06:14) and `403538b` (13 Oct 21:54). | The same tenth-seed staleness as §1.4, in a fifth provenance row. |
| 545 (540) | R row stops at Phase 4: "… report `1f8be04`) & `1f8be04` & 42; 20; 42" | R's Phase 5 report `31c39fc` (13 Oct 18:41, R's last report at `846d0c8`'s time) holds 64 records (20 + 20 + 20 + 4 perfect gates). `31c39fc` is cited nowhere in tex101; only the protocol `550a736` is (tex101:349). | The caption makes the commit column "the last report's". |
| 408 (404) | "This revision's first draft repeated the failure in kind: the referee's check of it found nine statements false …" | It means Revision 9's draft (the same sentence is in `c918ff7` at v9:387). | A carried-over referent with a capital T. README:378 carries it too. |
| 447 (443) | "The referee now reads every main-line specification before its first run and every draft before it is merged" | Revision 10 was merged without a referee pass (the sentence just before it says so). Revision 10.1 was committed at 18:33:26, merged at `ed9ca2b6` 18:33:46 and `264e227` 18:33:50, with no referee pass; this check is after the merge. | It contradicts its own paragraph and the revision it is in. |

Outside the tex (the diff's README, STATUS.md and CITATION changes):

| file | statement | note |
|---|---|---|
| STATUS.md:75–77 (unchanged lines) | "Its draft was checked in two passes by a session that had not written it, every number recomputed from the records (`docs/review_2026-10/rev10/`): every number of every claim reproduced" | These are the §1.2 and §1.3 statements the tex corrected. STATUS.md:78–80 (new) then reports V's pass. |
| STATUS.md:103 | "stage 2 would take about three days on one card" | Against tex101:485's "two and a half" and the spec's 2.26–2.39 days. |
| `docs/review_2026-10/rev10/README.md` (unchanged) | "Two passes over the Revision 10 draft by a session that did not write it … with every number recomputed from the records" | tex101:44 cites this directory's `first_pass.md`. The index does not list the referee's files beside it. |
| README header (`tools/readme/parts/header.md`) | "Revision 10.1 is the current refereed report" | 10.1 itself was merged 20 s after it was committed, before any referee read it. |

The rest of the diff is in order:
- CITATION.cff: the revision number.
- `tools/readme/build_readme.py`: expects three revision paragraphs.
- README: rebuilt from the tex; it mirrors the tex's corrections and residuals.

## 5. The PDF

The PDF check is `check_pdf_v10.py`'s method with REF `846d0c8`, PREV `585b666`, re-implemented in `check_rev10_1.py` §5. Its first lines:

```
846d0c8: 6012/6314 six-word runs of the tex found in the PDF text (95.22%)
lines with a run not found: 101
six-word runs only in 846d0c8's tex: 810, found in the PDF: 757 (93.5%)
six-word runs only in 585b666's tex: 255, found in the PDF: 1 (0.4%)
```

**The PDF shows its tex.**
- **The one old-only run found** is "memory reach the two screens close". It is a substring artefact: "close" inside the new "closed".
- **The new paragraphs are in the PDF text:** tex101:46 has 70/75 runs, tex101:370 has 251/262, and "Revision 10.1" appears 5 times.
- **The misses are layout, not stale text.** The table rows tex101:540–546 split by column under pdftotext. Their cells, including "19; 20", "…; 84", "…; 78", "…; 142" and "Revision 10's pass (2c250bd)", are present.
- **The PDF was built from this tex.** pdfinfo gives CreationDate Mon Oct 19 18:33:05 2026 UTC, 21 s before `846d0c8`, and 47 pages (Revision 10: 46 pages, 16:14:29).

## Errors in the referee's own files

1. **referee_pass.md:32–37.** It says "Nine statements … They are:" and then lists §1.1–1.8 only; the index row (§1.9) is missing. tex101:46 and :370 copied the omission.
2. **referee_pass.md:137 and audit_passes.md:15 and :88 (U13).**
   - **tex10:404 is missing from the carried-over list.** It reads "This revision's first draft repeated …", with a capital T; the list was evidently found by a case-sensitive search.
   - **tex10:358's heading "what this revision does with it" is missing too.** 10.1 fixed it anyway.
   - **The README count is low.** "Four places in the README" is the case-sensitive count. `585b666`'s README had five lines with "this revision's first draft" (64, 240, 336, 374, 412), plus :493 ("this revision's draft") and :546 ("this revision's checks").
3. **referee_pass.md:92–101 (§1.8).**
   - **"Four provenance rows" misses two.** The I row (tex10:539: `559b942`, 41 runs against 42 since `4371824`) is stale in the same way as §1.4. The R row (tex10:540: stops at Phase 4; `31c39fc` cited nowhere) is stale too.
   - **check_rev10_A missed both.** Its provenance rows (25, 36) checked tex10:519 and 522 only.
4. **tex10:443's "every draft before it is merged" was not flagged.** It was already false for Revision 10.

No number in `check_rev10_A/FG/HE.md` was found wrong. Every value this check recomputed equals theirs:
- 30 gated runs;
- 0.291;
- 21 + 42 firings;
- −0.0146 and +0.4136;
- 13/20 and 8 run-layers;
- 3600 and +1200…+6000;
- 54.3–57.3 h;
- 228 rows.

## What could not be checked

- **Whether a mistarget "cost" its run** (tex101:468). This is attribution. The records show only that the S86 2334 and S87a 2618 ALLMASS runs failed (MERGED) where BASE bound, and that 2611 bound.
- **Whether the "separate checking agent" was separate.** Git shows one session for the author and the passes (referee_pass.md §1.3).
- **The PDF beyond its text.** Figures, table layout and typesetting were not compared; pdftotext order breaks in tables.
- **README.md beyond pattern searches.** Its other rebuilt sections were not audited line by line.
- **Nothing was re-run.** Every count comes from the committed records, with the referee's `rlib`. S75b's hours are the spec's own estimates.

## The Revision 10 checks re-run against 10.1 (`check_rev10_*_v101`)

All 228 original rows are re-anchored on both texts. Every record recomputation gives the same numbers as against `585b666`.

| check | against `585b666` | the same rows against `846d0c8` | all rows against `846d0c8` (D / P / NRC / M) |
|---|---|---|---|
| A | 82: 1 / 5 / 4 / 72 | 0 / 2 / 2 / 78 | 104: 4 / 7 / 2 / 91 |
| FG | 107: 6 / 15 / 3 / 83 | 2 / 10 / 3 / 91, 1 removed | 112: 2 / 10 / 3 / 96, 1 removed |
| HE | 39: 3 / 10 / 2 / 24 | 0 / 1 / 2 / 35, 1 removed | 44: 0 / 1 / 2 / 40, 1 removed |

**The DISAGREEs against `846d0c8`.**
- **tex101:447**, the referee reading every draft before it is merged.
- **tex101:544–545**, I's and R's provenance rows.
- **STATUS.md:76**, not the tex.
- **FG's two are in the sessions' own reports**, not the draft: F's `report_8.md`:138 and G's `report_7.md`:20.

These agree with this file's residuals above. The re-run grades tex101:485 PARTIAL (2.26–2.39 days against "about two and a half") where this file grades it PARTLY.
