# Fifth pass: R's Phase 4 and batch 19, item by item

*Referee V, 12 October 2026.*

## What was checked

**The draft.** The fifth version is `c5e4508` (`claude/rev9-draft`, committed 2026-10-12 19:14:19 UTC, 568 lines), cited as `tex5:NNN`. The fourth version is `c66cc76` (`tex4:NNN`). Line mapping: tex4:N is tex5:N up to 192, N+2 from 193 to 451, and N+4 from 453 on (tex4:452 became tex5:454–456).

**The rows.** These are every row of `fourth_pass.md` in this scope:
- §1 items 1–3;
- the R bullets and the batch 19 bullets of §2;
- §2a's R item.

Every line of `git diff c66cc76 c5e4508 -- multichannel_hebbian_report_v9.tex` that touches R or batch 19 was also checked.

**The commits.** The referee-fix commits are `da96f69` (R) and `e0c5549` (batch 19). Three other commits touched these lines:
- `3b06af2` added tex5:41's "every session's queue is empty";
- `fabf505` moved the parenthesis at tex5:332;
- `ef6d690` and `aab7f7c` changed other parts of tex5:36, 41 and 468.

**The records.**
- R: `origin/claude/replicate-R`, still at `1f8be04`; nothing is newer.
- E: `origin/claude/outside-ideas`, now at `46cb8d0`. Its 65 commits after `e3f5809` are all batch 20, and the batch 19 stores are byte-identical.

**The checks.** Numbers are from `check_R_v5.py` and `check_E19_v5.py`, with their tables in `check_R_v5.md` and `check_E19_v5.md`. They use `rlib.py` and `replication_check.py`'s `r_transition` and `r_bound_routed`, as v4 did.

## Tally

| scope | items | RESOLVED | PARTLY | NOT | NEW ERROR |
|---|---|---|---|---|---|
| §1 items 1–3 (R) | 3 | 3 | 0 | 0 | 0 |
| §2, R bullets (the not record-checkable pair counted as two) | 7 | 5 | 0 | 2 (no change, none needed) | 0 |
| §2, batch 19 bullets (tex4:235 and tex4:366 counted apart) | 8 | 7 | 0 | 0 | 1 (tex5:368) |
| §2a, R's report | 1 | 0 | 0 | 1 (R's branch unchanged; the draft now names it) | 0 |
| **fourth-pass items** | **19** | **15** | **0** | **3** | **1** |
| diff lines outside the items | — | — | — | — | 1 (tex5:41) + 1 residual PARTIAL (tex5:36, minor) |

**Checks against `c5e4508`:**

| check | rows | DISAGREE | PARTIAL | NOT RECORD-CHECKABLE | MATCH |
|---|---|---|---|---|---|
| `check_R_v5` | 39 | 0 | 1 | 2 | 36 |
| `check_E19_v5` | 31 | 2 | 0 | 0 | 29 |

## §1: false statements (R)

| item (fourth pass) | fifth version | status |
|---|---|---|
| 1. tex4:328: "the interval includes $0.95$" | tex5:330: "the point estimate $0.875$ sits below $0.95$, the interval reaches $0.945$, just below $0.95$ (R's report says it includes $0.95$; the pre-registered reading, which tests overlap with the report's interval, is unaffected), it excludes rates below $0.74$" | **RESOLVED.** Wilson(35, 40) = [0.7389, 0.9454]. 0.945 is right, and so is 0.74. R's error is now attributed to R (`replicate_report.md`:138). |
| 2. tex4:36: "by a session that never saw the code" | tex5:36: "A clean-room replication from the report alone, by a session that had not seen the code when it ran its first twenty seeds, bound eight streams $17/20$ against $7/20$ without the split (10 vs 0), and $18/20$ on twenty further seeds after taking the split's probe and key positions from our code ($35/40$ against $12/40$ pooled, 23 vs 0)" | **RESOLVED.** This is the fourth pass's wording, verbatim. Phase 2's records (`de8df0f`, 13:55:49) precede `DIFF.md` (`00990ea`, 14:01:24). The counts are right: 17/20, 7/20, 10 vs 0, 18/20, 35/40, 12/40, 23 vs 0. *Residual, minor:* the subject "a clean-room replication from the report alone" still governs the 18/20, which the sentence itself says used our code. That is `check_R_v5` row 1, PARTIAL. The referee's own fix left this phrase in place. |
| 3. tex4:354: "replicated from the text alone ($17/20$, then $18/20$ under the main line's protocol, $35/40$ pooled)" | tex5:356: "\textbf{Stands}, and was replicated, on our assessment, in session~R: $17/20$ from the text alone (``partially replicated'' by R's bar), then $18/20$ with the split's probe and key positions taken from our code, $35/40$ pooled (``consistent with the report''; …)" | **RESOLVED** on all three points: "from the text alone" now covers 17/20 only; "the main line's protocol" became the two choices R took; and "replicated" is marked as the authors' assessment, beside R's two mechanical readings ("partially replicated" in Phase 2 and "consistent with the report" in Phase 4, both recomputed). |

## §2: loose statements, R's Phase 4

| item (fourth pass) | fifth version | status |
|---|---|---|
| tex4:328: "decided one run in twenty (914)" | tex5:330: "with one probe drawn per seed (Phase~2) it occurred at 13 of 192 (13 of 282 with R's second rerun, which had none) and, on R's inference, decided one run in twenty (914); the rerun that would test that inference, 914 under the main line's protocol, was not run." tex5:207: "at 13 of 282 and, on R's inference, it decided one run" | **RESOLVED.** 914 has 12 blocked checks, 11 of them eligible, and did not end bound routed. Amended part (a), 914 and 915 "with the main line's protocol" (`db22857`:866, whose own wording the draft keeps), has no record. |
| tex4:328: "about 150 seeds per arm" | tex5:330: "separating $0.85$ from $0.95$ as two arms against each other would need about 150 seeds per arm (against a fixed rate on fresh seeds, the 52 of Section~\ref{sec:method} answer a different question)" | **RESOLVED.** The two-arm normal approximation gives 141 per arm two-sided and 111 one-sided; the 150 is R's figure ("on the order of 150", `replicate_report.md`:139). The one-sample figure is n = 52 (x ≤ 46, size 0.045, power 0.812). tex5:430 states "$x \leq 46/52$ … size $0.045$ and power $0.81$", inside `sec:method` (label at tex5:410). |
| tex4:391: "would have detected only rates near $0.70$" | tex5:393: "would have detected only rates near $0.70$ or below, and $0.70$ itself one time in nine" | **RESOLVED.** P("a lower rate" given 17/20 kept) is 0.113 at 0.70 (1 in 8.8), 0.238 at 0.65, 0.404 at 0.60 and 0.041 at 0.75. |
| tex4:205: "13 of 192" (unlabelled) | tex5:207: "with one probe drawn per seed at 13 of 282" | **RESOLVED.** Phase 2 is 13 of 192 and R2 is 0 of 90, so 13 of 282. That is pooled the same way as the fixed-probe 3 of 263 (R1 1/91 + Phase 4 2/172). tex5:330 gives both figures with the label "(Phase~2)". |
| Phase 4's "920–939 alone" 18/20 | tex5:330: "On the fresh seeds alone, with those two choices, the split arm bound $18/20$, the count Phase~2's bar had asked for, which \texttt{phase4.py} prints as secondary without a reading"; tex5:356: "replicated, on our assessment" | **RESOLVED.** 18/20 against 5/20, 13 vs 0. `phase4.py`:19–20 says "printed beside it, as a secondary count", and `phase4.py`:125 prints it with no reading. |
| Not record-checkable (a): "under a rule pre-registered before any run (\texttt{464be8d})" | tex5:330, unchanged | **NOT (no change; none needed).** The records carry no timestamps. The replayed schedule puts the start no later than 19:02:43 UTC on 10 October; `464be8d` is 19:01:33 and the referee's review `01bee1f` is 19:08:20. |
| Not record-checkable (b): "was written after the referee's review but existed in no commit until \texttt{db22857}, after Phase~4 had started, and was not used" | tex5:330, unchanged | **NOT (no change; none needed).** "Written after the referee's review" rests on chat. The rest checks out: `db22857` (11 Oct 03:01) is after `0364e28` (00:02, 21 records), and no amended arm, seed or rule appears in R's records. |

## §2: loose statements, batch 19

| item (fourth pass) | fifth version | status |
|---|---|---|
| tex4:235: S66's one-pair rule has no power caveat | tex5:237: "The rule, ``carries'' at $d \geq -1$ on ten seeds against a reference at $8/10$, reads ``carries'' about $40\%$ of the time when the larger model's true rate is $0.6$ against $0.8$, so the screen rules out a large drop and not a moderate one." | **RESOLVED.** P(carries) = 0.398 under independence and 0.376 under maximal correlation. |
| tex4:366: the same, in the verdict paragraph | tex5:368: "Eight streams carry to $D = 128$ and two layers on a ten-seed screen (S66), whose one-pair rule misses a drop from $0.8$ to $0.6$ more often than not" | **NEW ERROR.** The rule misses that drop (reads "carries") **0.40** of the time, or 0.38 under maximal correlation. It catches the drop 0.60 of the time. The sentence states the reverse, and contradicts tex5:237's own "about 40%". P(miss) exceeds 0.5 only for an arm at 0.66 or above (at 0.65 it is 0.498). *Fix:* "misses a drop from 0.8 to 0.6 about two times in five". |
| tex4:235: the D=128 oracle bound P=16 only 1/2 | tex5:237: "the perfect gate binds $P = 16$ at $N = 256$ on both of its seeds, late (24{,}000 and 27{,}600), and at $D = 128$ on one of two, so the reading ``not capacity-bound at $N = 256$'', which is about the oracle, applies and the oracle gives no capacity reason for the wider model's learned advantage" | **RESOLVED.** ORC_P16_N256 bound 2/2 at 24000 and 27600; ORC_P16_D128 bound 1/2 at 12000. The last clause is E's README ("So the oracle gives no capacity reason for D128's learned advantage", S67 observations). |
| tex4:235: the L2 tie-break | tex5:237: "two layers was S65's ``best depth'' through a tie with four that the arms' order broke, a step outside the pre-registered rule, which session~E discloses" | **RESOLVED.** L2 and L4 were 10/10 each, with median 3600 and 27,872 parameters each. The decision was stored 2026-10-09 16:40:29 UTC, `best_depth` 2. README:659–661 says "not in the pre-registered rule". E's further warning (README:699, "not as 'fewer layers is better'") is not relayed, but the draft makes no such claim. |
| tex4:57: "not capacity-bound at $N = 256$ for 16 keys" | tex5:57: "carries to $D = 128$ and 2 layers at eight (one-pair rule); the oracle (2 seeds) binds 16 keys at $N = 256$, the learned recipe $0/10$" | **RESOLVED.** Oracle 2/2; P16_N256 0/10. |
| tex4:452: "constants held … twice the depth" | tex5:454: "its constants read ``scale-free'' and ``carries'' by one-pair rules on ten-seed screens, to four times the width and to four applications of the layer at four streams and to $D = 128$ and two layers at eight" | **RESOLVED.** S64 d = +1 four times; S65 L4 d = 0; S66 D128 d = 0 and L2 d = +2. "Four applications" is E's own "promising to 4 layers" (README:634). |
| tex4:233: "bound $10/10$, $10/10$ and $9/10$" | tex5:235: "2, 4 and 6 applications of the shared layer bound routed $10/10$, $10/10$ and $9/10$" | **RESOLVED.** Bound routed: L2 10, L4 10, L6 9. L6 is bound 10/10, and seed 283 is bound but not routed. |
| tex4:464: "eight streams at the larger widths" | tex5:468: "eight streams at the larger widths (screened once in S66)." | **RESOLVED.** |

## §2a: R's report (not the draft)

| item | now | status |
|---|---|---|
| `replicate_report.md`: "The interval excludes rates below 0.74 and includes 0.95" (line 138) | R's branch is unchanged at `1f8be04`. The draft now names the error as R's and corrects it (tex5:330). | **NOT** in R's report. It needs no draft change. |
| **New, missed by the fourth pass:** `replicate_report.md`:27, Phase 3 reading: "17/20 … its interval contains the report's 38/40 (0.95)" | Wilson(17, 20) = [0.6396, **0.9476**]; 0.9476 < 0.95. The interval overlaps the report's [0.835, 0.986] but does not contain 0.95. | An error in R's report of the same kind as line 138. The draft does not repeat it: tex5:330's "17/20 is consistent with 38/40" is supported by Fisher p = 0.322. |

## Diff lines touching R or batch 19, beyond the items above

| tex5 line (commit) | new wording | check |
|---|---|---|
| 41 (`3b06af2`) | "F's S81, G's S83, H's S86 and R's Phase~4 are complete, every session's queue is empty" | **NEW ERROR (stale).** At `c5e4508` (19:14 on 12 October), E's batch 20 had been running since 2026-10-11 22:04 UTC (`404d3bd`). Its last commit before the draft, `a876751` (17:49), reads "IN PROGRESS: 10 of 19 runs saved". Batch 20 regenerates every unbound S67 learned run and continues it to 86,400 updates, which is the draft's own "clearest lead" (tex5:237, 477). The clause was true when written (`3b06af2`, 17:43 on 11 October; batch 20's code is `4acf915`, 21:27). It has been false in the six draft commits since `ef6d690`. "E (… batches 1--19, complete)" is true. R's queue is empty. *Fix:* "E's batch 20 (S67 continued) is running". Batch 20 names its screen "S68", which collides with F's S68 (tex5:41, 362, 514). |
| 41 | "R's Phase~4 are complete"; "R's records carry no CPU model"; "batch~19 on 2.80\,GHz" | MATCH. 42/42 jobs. No CPU field in R's records. 186 batch 19 records, all on "Intel(R) Xeon(R) Processor @ 2.80GHz". |
| 62 (`da96f69`) | "$18/20$ with the split's probe and key positions as the main line specifies them" | MATCH. D-1 and D-2 (`phase4.py`:5–9). This resolves v4 row 4 here as well. |
| 330 (`da96f69`) | "A fourth phase ran the split with its probe and key positions as the main line specifies them (the fixed probe, the body's key positions; its other choices, among them the map on the held-out sequences, R's own)" | MATCH. D-10 is the map on the 2048 held-out sequences, R's own. D-4 to D-9 are R's own too. |
| 332 (`fabf505`) | "left 34 to be guessed (the initialisation scales, … among them): 20 …, 12 … and 2 …, with 3 further entries of process" | MATCH. This is the fourth pass's §1 item 9, resolved: the parenthesis now follows the guesses. U-6, U-16 and U-23 are [A] entries and U-11 is a [C] entry, all among the 34 guesses; the process entries are U-42 to U-44. 4 + 3 + 34 + 3 = 44. |
| 36, 41, 468, 472 (`ef6d690`, `aab7f7c`, `902c39a`) | `test_split_copy`, `test_delta_recipe`, S75 | Outside this scope; the R and batch 19 parts of these lines are unchanged and MATCH (`check_R_v5` rows 4, 6, 37; `check_E19_v5` row 29). |
| 205 (`3b06af2`) | "The seed-914 mode appeared in one guarded run" (F's S81) | Not checked: it is F's record. It names R's failure mode and does not cite R's records. |

## Errors in the referee's own earlier files

- **`check_R_v4.md` and `fourth_pass.md` §2a** list only `replicate_report.md`:138. They miss the same error at line 27 (17/20's interval "contains … 0.95"; its upper bound is 0.9476).
- **`check_R_v4.py` §6** stores the per-arm block totals in `res[name[:2]]`, and the key is "Ph" for both Phase 2 and Phase 4. `res` is never read afterwards, so v4's output and tables are unaffected. `check_R_v5.py` computes Phase 2's 13/192 directly.
- No other error was found in `check_R_v4.md`, `check_E19_v4.md` or `fourth_pass.md` in this scope.

## Could not check

- Phase 4's start time and the chat history of the amended prompt (the two NOT rows).
- Batch 20's records: in progress, not a result, and not cited by the draft.
- The PDF.
