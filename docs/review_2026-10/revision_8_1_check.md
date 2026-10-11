# Revision 8.1: what was corrected, whether it is right, and what else changed

*Referee V, 10 October 2026. Diff: `git diff 4a3eebc 8fb968b -- multichannel_hebbian_report_v8.tex` (one commit, 8fb968b, touches the tex; 61 lines). Line numbers are 8.1's (`8fb968b`). Numbers were recomputed from the records with `recompute.py` and the checks quoted below; nothing was run. The PDF at 8fb968b was compared with the tex by `pdftotext` on every corrected sentence: they agree.*

## Verdict

The six substantive errors (W1 a–c, W2, W3's abstract sentence, the Muon reversal) are corrected, and every number the authors took from my files is right. One of those numbers is right by coincidence: the "0/60" single-channel count I supplied was wrongly composed. Details are under "A number I supplied", and `recompute.py` is fixed.

8.1 leaves four kinds of problem:

- **The premise test is still listed as future work** in two places (tex 328, 337).
- **New content was added under a note that says "nothing else" changed.** The note itself contradicts the new citations: it says S61 and S70 "will be in Revision 9" while the body already cites them.
- **Two new mistakes:**
  - "S69–S70 add 102 runs" (102 is S68–S70; S69–S70 is 78).
  - "Phase III's `conv_lr`" (conv_lr is Phase IV).
- **One older error, missed in my first review, persists:** Table 4 says one Hebbian channel was never run at four streams on the grouped layout; `test_scale_axes`'s B4s did exactly that, 0/20.

Most of W8 and part of W7 are untouched. The 8.1 note does not claim them; it defers "structural requests" to Revision 9.

## 1. My weaknesses W1, W2, W3, W7, W8, W9

| item | made? | correct? | where (8.1 line) | notes |
|---|---|---|---|---|
| W1a premise test "not run" | yes (abstract 37, §11 228, §12 257) | **yes** | S68: one delta channel 2/10 at S = 4, 0/10 at S = 8; seeds 300–309; 43,200 updates; both delta perfect gates at 1200; gated Hebbian 19/20 and 9/10 on other seeds. All match `premise_results.json` and `window_delta_scale_results.json` | **Not made in two places:** §16's staged path still lists "the premise test (one delta channel at four and eight streams)" as step (i) (tex 328); the conclusion's next steps still list "session F's premise test" (tex 337) |
| W1b Muon reversal | yes (§7 202, §13 269) | **yes** | E2 L 7 vs 0, X 5 vs 1 (cut arm); M2 X 7 vs 1, L 5 vs 7; "a seed effect, which the data do not establish" | the README's Summary still says "Muon does not 'bind four streams sooner'" (README "Corrections" paragraph), an assertion of absence the corrected §7 no longer makes |
| W1c HINGE4k16 routing on L | yes (Table 4 caption 248) | **yes** | X 15/20, L 16/20 routed; 35/40 bound | |
| W2 the convolution argument | yes (§1 94) | **yes** | N = 256 perfect gate without conv: 4/4 (CEIL_A) and 10/10 (F's ORACLE_HEBB_G); the failing runs used the N = 64, 7,552-parameter ceiling, against 25,984 | §12 (252) still says "Given the partition, every grouped-layout configuration tried binds". The N = 64 perfect gate on the grouped layout is a configuration tried that did not bind; "with the N = 256 model" would make it true |
| W3 "the reset adds nothing" | yes (abstract 37, §6 193) | **yes**, with one wording slip | S61: COPY_NONOISE (exact copy, no reset, no noise) 9/10, COPY (copy + noise, no reset) 9/10, SPLIT 9/10, RESET 4/10, NONE 3/10 — all match `copy2x2_results.json` | (i) "run after these looks" (193) holds for S50 and S58b (complete by 16:31 UTC on 8 October) but not for the main test: S61 ran from 20:32 on 8 October to 05:42 on 9 October, while L's window_gate run (17:33 UTC on 8 October to before 05:20 on 9 October) and X's Part D (11:48 on 8 October to about 18:40 on 9 October) were in progress; the records carry no per-run times, so whether the control's own runs finished before S61 cannot be said; (ii) S61's own pre-registered readings are not mentioned, nor that its "the noise matters" reading did not apply (preregistration.md §2) |
| W7 "one channel per stream" | partly (abstract 35; §3 112) | **yes** for what was added | the §3 sentence quotes 0.047 and 18 of 38 correctly | the abstract keeps "with one channel per stream" and adds "(the streams on disjoint sets of channels ...)", which contradicts it in a parenthesis; §5 (168) keeps "with one channel per stream in 38 of 40 runs"; the overlap-based G1 (16 vs 2) is not reported |
| W8 literature | partly | the two fixes are right; two new phrases are unsupported | **fixed:** "none of twelve ... under a sparse end-of-sequence loss" with the HM-RNN exception (37, 222); BDH-CQ now "one reading among several, since the quoted sentence counts plain linear attention ... among such rules" (224); "papers' analyses, read as analogies" (222) | **new, unsupported:** TLA's form now "(from its abstract; the paper is not among the twelve read in full)" (78), but the repository records no reading of TLA's abstract (the audit's "verified by search", `audit_2026-10.md:99`, is the only trace); BDH-CQ's reading is now "an inference from the paper's public text", but no note reads the paper (the notes work from "the brief"). **Not made:** the routing study's "0.9–29%" (the 0.9% is Fourier mixing; 222); Flesch's "must receive only the cue" (222); "[recdepth] explains why the frozen memory must enter every step" (224); "Gumbel noise works for a dense read" (301, against Raven's mixed ablation); the Nested Learning title, "Architectures" (398), against the note's "Architecture" |
| W9 provenance and bookkeeping | yes (40, 64, 110, Table 6 371–377) | mostly | batch 16: "60 of batch 16's 165 runs ... on 2.80 GHz" ✓; batch 15's missing per-run CPU ✓; 14 of 1869 ✓; specs/ sentence ✓; branch heads pinned (7e9197f, a2e55ed, 514d17f) ✓; batch 19 118 at 5dfc2b5 ✓; L's local time UTC−6 ✓ | **new errors in Table 6:** "S69–S70 add 102 runs (9a6c2bf)": S68 24 + S69 32 + S70 46 = 102, so S69–S70 add 78. The caption says "later screens are noted with their run counts", but S61 (in 7e9197f), S68 (in a2e55ed) and 67 of S56's records (in 514d17f) were already in the pinned commits, not later. The single-evaluation sentence (110) says the audit's 11 of 1028 "counted X's files only". The numerator reproduces on X's files; the denominator does not (924 bound records there), so "X's files only" is an inference. Write "the audit's 11 reproduces on X's files; its 1028 does not" |

## 2. recompute.md's DISAGREE and PARTIAL rows

| row (Rev. 8 line) | 8.1 | correct? |
|---|---|---|
| abstract: no single-channel run at eight streams (35) | corrected (37, 228, 257) | yes; see W1a for the two places left |
| batch 16's CPU (40) | corrected (40) | yes: 60 of 165 at 2.80 GHz, recomputed |
| §1 convolution, seed-dependent (98) | corrected (94) | yes |
| Muon's E2 machine (200) | corrected (202, 269) | yes |
| §12 one Hebbian channel (250) | corrected (252) | 0/60 without the conv and 18/40, 10/20 with it are right as numbers. The caption calls conv_lr "Phase III's"; it is a Phase IV test (`docs/revision6.md:92`) |
| Table 4 footnote, routing only on X (246) | corrected (248) | yes |
| 11 of 1028 (108, PARTIAL) | corrected (110) | 14 of 1869 is right; the parenthesis over-attributes (W9) |
| batch 1's five failures (127, PARTIAL) | corrected (129) | yes: 3 KEY + 2 STREAM-PARTIAL under the margin-first rule; 4 KEY + 1 under the key-first rule (seed 177: margin 0.254, η² by key 0.733) |
| batch 1 "bound faster" (127, PARTIAL) | corrected (129) | yes: means 2720 vs 3429, equal medians |
| decodability "from 4800" (129, PARTIAL) | corrected (131) | yes: 0.70–0.72 for the two runs at 9600 |
| PREV_CAP_M "0.61 against 0.37" (204, PARTIAL) | **not corrected** (207) | the paired median is 0.3645, which rounds to 0.36 |
| batch 19 "116 so far" (365, PARTIAL) | corrected (371) | yes: 118 at 5dfc2b5 |
| batch 15 CPU (UNVERIFIABLE) | noted (40) | yes |
| Table 4 "0 of >200" (UNVERIFIABLE) | replaced by "0/60^c" (243, 248) | right as a number; see the next section |

**A number I supplied, and an older error both of us missed.**

- **The composition of 0/60.** My first `recompute.py` counted only records carrying `k = 1`. That missed the Phase III arms named B (router_discovery, router_curriculum, router_confirm), whose records carry no `k`, and it included test_router_layout's blocked and shuffled layouts (30 runs). The corrected count, grouped layout only, is still 0/60: short_conv B 20, router_discovery B, router_curriculum B and router_confirm B 10 each, and conv_lr B4 10, the last at lr 4e-3. The blocked and shuffled layouts add 0/30. So the authors' "0/60" and "every such run in `results/`" are right if read as the grouped layout. `recompute.py` and `recompute.md` are corrected and say so.
- **Not run at four streams?** Table 4's caption (248, also in Rev. 8 at 246) says "One Hebbian channel has not been run at four or eight streams on the grouped layout". `test_scale_axes`'s B4s is a single Hebbian channel with the convolution at S = 4 on the grouped layout: 0/10 on X and 0/10 on L, every run at final accuracy 0.23–0.27. Revision 6 claimed it (S2, "the gate beats one channel at S=4", SHOWN). The first draft's §12 even cited it ("four streams with four channels"); 8.1 deleted that clause and kept the caption. Recorded as a DISAGREE row in recompute.md (tex 248). It strengthens the paper.

## 3. What changed that should not have, or contradicts itself

1. **"This revision corrects them and nothing else"** (42), yet 8.1 adds results that did not exist when Revision 8 was committed:
   - S70's counts (230): F's report 3 is `9a6c2bf`, 23:33 UTC on 9 October, after the report's 19:55.
   - The S56–S57 run count (Table 6).
   - S61's counts in the abstract and §6. These existed before Revision 8, so citing them is a correction of an omission.

   Adding S70 is reasonable, but it is new content, and the note should say so.
2. **The Revision 8 paragraph contradicts the body.** It says "F's S69–S70, G's S56–S57 and H's S61–S63 are on their branches and will be in Revision 9" (40), while the abstract and §6 cite S61 and §11 (230) cites S70's results.
3. **The two new literature phrases** ("from its abstract" for TLA; "from the paper's public text" for BDH-CQ) assert readings the repository does not record (W8 row).
4. **"Phase III's conv_lr"** (248, 252 via the caption): conv_lr is Phase IV.
5. **Table 6 counts and caption** (371–377): see the W9 row.

Nothing else in the tex changed: the diff has 13 hunks (`git diff -U3`), all accounted for above or in the tables.

## 4. The README (regenerated at 8fb968b)

- **Summary against the abstract.** The content matches, split into the paragraphs "A window gate", "The split's mechanism", "Scale, Muon", "Distant cues and the memory rule" and "Corrections". Every corrected number (38/40, 15/40, 13 vs 2, S61, 0/10 and 2/10, the scoped "none of twelve") is the same. One difference: the Corrections paragraph keeps "Muon does not 'bind four streams sooner'", stronger than the abstract's "we correct ... that Muon 'binds four streams sooner'" and than §7's "pooled it is not shown".
- **Tables against the records.** Table 2 (main) and Table 3 (copy/reset) match the records cell for cell; the copy/reset table's "—" for the control's split cell is unchanged, as in the tex (on those seeds the split arm bound 10/10, 7 vs 0 against the reset). Table 4 matches, with the B4s caveat above. Table 6's batch labels are restored, and its runs column matches the records (45, 165, 49, 120, 118, 318, 322, 22). **Table 5 still drops X's run counts**: the "runs, X; L" column reads "; 166" and "; 231" where the tex and the records give "156; 166" and "231; 231". This conversion bug was already present at 4a3eebc.
- **The upstream section** (Pathway's README, kept verbatim) still tabulates "Pathway BDH | Sudoku Extreme Accuracy | 97.4%" (README "Performance Comparison"). The report itself says this figure is not a Sudoku-Extreme number (§11, 224). A one-line note under the upstream table would stop a reader taking it as the project's claim.

## 5. What the 8.1 note says it corrected but did not, in full

| the note says corrected | not done |
|---|---|
| "the premise test had been run (Sections far and synthesis)" | §16 (328) and the conclusion (337) still list it as a next step |
| "the convolution argument compared two different models (Section window)" | §12 (252) "every grouped-layout configuration tried binds" not qualified |
| "'one channel per stream' is by largest mass (Section task)" | the abstract (35) and §5 (168) still say "one channel per stream" |
| "the one-channel column ... misstated (Section synthesis)" | the caption's "not run at four ... streams" (248) is wrong (B4s, 0/20); "Phase III's conv_lr" is Phase IV |
| "the branch heads were misstated (the appendix)" | heads pinned, but the "add 102 runs" figure and "later screens" wording are wrong (W9) |
| "two literature sentences were overstated (Section far)" | the two named sentences are fixed; the same section's routing range, Flesch clause and recurrent-depth clause (222, 224) are not, nor §14's "Gumbel noise works for a dense read" (301) |

Not claimed by the note and still open (for Revision 9, as the note says):

- the PREV_CAP_M rounding (207);
- G3's "better by two pairs" and R2's "equals it at two streams" (statistics.md §3);
- the abstract's "survive a fourfold change of width and a doubling of depth";
- the conclusion's "solves context-conditional binding";
- the intervals and the claim list (see `claims_registry.md`).
