# Batch 19 (Session E) against the fourth draft: check_E19_v4

Referee V, fourth pass. Draft: `multichannel_hebbian_report_v9.tex` at `c66cc76` (tex4:NNN), compared with the third version `a5f67a2` (tex3:NNN). Batch 19 is folded in by `98ddebf`. Records: `origin/claude/outside-ideas` at `e3f5809`, in the files `explore_out/{scale_width,scale_depth,scale_eight,capacity}_results.json` and `batch19_runtime_results.json` (the phase-2 decision). README: `explore_out/README.md` at `e3f5809`, lines 612-713 ("Batch 19 verdicts"). Screen docstrings: `explore_scale_width.py` (S64), `explore_scale_depth.py` (S65), `explore_scale_eight.py` (S66) and `explore_capacity.py` (S67). Script: `check_E19_v4.py`, with its output in `check_E19_v4_out.txt`. It uses only the standard library and `rlib`. Bound routed is recomputed with `rlib.bound_routed`, and failure classes with `rlib.outcome_k`. McNemar is exact and two-sided, as E's docstrings print it.

**Counts.** 30 rows: 0 DISAGREE, 8 PARTIAL, 22 MATCH and 0 NOT RECORD-CHECKABLE.

- Every number in the draft about batch 19 reproduces from the records. That covers every count, discordant pair, p-value and median transition, plus the run total (186).
- Every reading reproduces under its docstring rule.
- The PARTIALs are about scope and power, not about numbers:
  - S66's "carries" uses the same one-pair rule as "scale-free", against a reference at 8/10. It reads "carries" 40% of the time at a true 0.6 against 0.8, and 23% at 0.5. The draft gives the power caveat for four streams only.
  - The index row gives S67's oracle reading without saying it is the oracle's.
  - The draft leaves out two facts: the D=128 oracle bound only 1/2, and the phase-2 depth was chosen by a tie-break outside the rule.
  - L6 is described as "bound 9/10".
  - One next step is stale.

## Rows (DISAGREE first; there are none)

| # | tex4 line | item | draft says | recomputed | status | note |
|---|---|---|---|---|---|---|
| 1 | tex4:235 | S66 readings, power | D=128 binds routed 8/10 "(2 vs 2; 'carries', d = 0)"; L2 binds 10/10 "(2 vs 0, p = 0.5; 'carries', d = +2)" | Counts as stated. The rule "carries if d >= -1" (explore_scale_eight.py READING; CARRY_D = -1) on 10 seeds, independence approximation as in stats_checks.py §7: P(carries) = 0.80 at 0.8 vs 0.8, **0.40 at 0.6 vs 0.8**, 0.23 at 0.5, 0.11 at 0.4 | PARTIAL | tex4:233 says "with ten seeds the screen rules out a large drop, not a small one", but only for four streams. S66 uses the same one-pair margin against a weaker reference (8/10), and the draft gives it no such sentence. |
| 2 | tex4:366 | verdict paragraph, S66 | "Eight streams carry to $D = 128$ and two layers on a ten-seed screen (S66)" | as row 1 | PARTIAL | The referee's 39% note in the same paragraph is about the four-stream rule. At eight streams the same rule reads "carries" 40% of the time when the larger model is at 0.6 against 0.8. The sentence says "ten-seed screen" but not what such a screen can miss. |
| 3 | tex4:452 | "What Phase VI adds", added clause | "its constants held, on ten-seed screens, to four times the width and twice the depth at four streams and to $D = 128$ and two layers at eight" | S64: d = +1 ×4. S65: L2 0, L4 0, L6 −1 (median 4800). S66: D128 d = 0 (8/10), L2 d = +2 | PARTIAL | (a) "Constants held" restates two mechanical one-pair readings (rows 1-2) as the case for the method. (b) "Twice the depth" is L6, which is inside the margin at d = −1, has its one failure (283), and is slower. E's verdict is "promising to 4 layers" (README:634), and tex4:57 also says "4 layers". (c) The sentence says "an eight-way partition ... at four streams". At four streams the partition is four-way. |
| 4 | tex4:57 | index row, result | "not capacity-bound at $N = 256$ for 16 keys" | The reading is on the oracle (explore_capacity.py READING): ORC_P16_N256 bound 2/2 (27600, 24000). The **learned** recipe at P16_N256 binds 0/10 (6 STREAM-PARTIAL one-to-one, 4 MERGED) | PARTIAL | The row does not say "the oracle (2 seeds)". Read alone, it suggests the recipe handles 16 keys at N = 256, where the recipe binds 0/10. tex4:235 and tex4:366 do give the learned 0/10. |
| 5 | tex4:235 | S67 oracles | "the perfect gate binds $P = 16$ at $N = 256$ on both of its seeds, late (24,000 and 27,600)" | Correct as stated. Left out: ORC_P16_D128 bound **1/2** (271 at 12000; 270 unbound, final accuracy 0.32) | PARTIAL | The reading itself does not depend on D128's oracle. The omitted fact does bear on the lead the draft passes on (P=16 at D=128, 4/10 vs 0/10). E's README says "the oracle gives no capacity reason for D128's learned advantage" (S67 observations), and the draft does not relay this. |
| 6 | tex4:235 (and tex4:57, 366, 452) | S66's two-layer arm | "$10/10$ with two applications of the layer" | Phase-2 decision stored at 2026-10-09 16:40:29 UTC: best_depth 2. In S65, L2 and L4 tie on every term of the rule (10/10, median 3600, 27872 parameters). E's code then kept the arms' order, a step that is **not in the pre-registered rule** (README:657-665) | PARTIAL | E disclosed this and warns that the 10/10 "reads only for 2 layers, not as 'fewer layers is better'". The draft does not say how the arm was chosen. |
| 7 | tex4:233 | S65 counts (wording) | "2, 4 and 6 applications of the shared layer bound $10/10$, $10/10$ and $9/10$" | L6 is **bound** 10/10 and **bound routed** 9/10. Seed 283 is BOUND NOT routed: transition 6000, final accuracy 1.00, map [1, 4, 1, 3] | PARTIAL | The reading is on bound routed, so the counts are right only if "bound" means bound routed. The earlier part of the sentence says "bound routed". This is old text from Revision 8; only "(E)" and "follow." changed in this line. |
| 8 | tex4:464 | staged path (unchanged line) | "(i) On CPUs: ...; eight streams at the larger widths." | S66 has screened eight streams at D=128 (8/10, carries) and at 2 layers | PARTIAL | The step is now partly done. tex4:473 was updated to "batch 19's lead"; this line was not. |
| 9 | tex4:41 | E's status | "batches 1--19, complete"; "batch~19 on 2.80\,GHz" | 186/186 records: ok, not dry, cpu "Intel(R) Xeon(R) Processor @ 2.80GHz", code 17e56410a1db | MATCH | |
| 10 | tex4:57 | index row, scope and other results | "scale-free to $N = 1024$, $D = 128$, 4 layers at four streams; carries to $D = 128$ and 2 layers at eight" | S64 and S65 both read "scale-free". S66: D128_L3 and N256_L2 both read "carries" | MATCH | "4 layers" follows E's verdict. The rule's reading also covers 6 layers (d = −1). |
| 11 | tex4:233 | S64 | N512, N1024, D64, D128 bound routed 10/10 each against 9/10, "all with median transition 3600" | Each 10/10; N256 9/10 (273 MERGED, 2 share). Each 1 vs 0, two-sided p = 1. Medians 3600 everywhere | MATCH | All four d = +1 rest on seed 273. |
| 12 | tex4:233 | S65 | 10/10, 10/10 against 10/10 for three; six layers slower (median 4800) | L2 and L4 10/10 (0 vs 0); L6 9/10 (0 vs 1, p = 1); medians 3600, 3600, 3600 and 4800 | MATCH | Wording is covered in row 7. |
| 13 | tex4:233 | oracles | "the oracle binding at every size and depth" | S64 oracles 10/10 (all at 1200); S65 oracles 8/8 (L6 at 2400, others at 1200) | MATCH | |
| 14 | tex4:233 | readings | "Both readings were 'scale-free' by their pre-fixed rule (every $d \geq -1$), at four streams only" | S64 scale-free (d +1 ×4); S65 scale-free (0, 0, −1); RES6 not run (−1 > −2) | MATCH | The rule is in the docstrings at 617681d. |
| 15 | tex4:233 | sentence changed in v4 | "Eight streams and larger key loads at the larger widths follow." | — | MATCH | The next paragraph also covers N256_L2 and N=256, so "at the larger widths" is loose. |
| 16 | tex4:235 | completion | "The batch completed after this revision's first draft (186 of 186 runs)" | 60 + 48 + 30 + 48 = 186. First draft 9bfa85a, 2026-10-10 22:16; e3f5809, 2026-10-11 04:25 | MATCH | |
| 17 | tex4:235 | S66 configuration | $k = 16$, the convolution, 43,200 updates, seeds 290--299, "every constant unchanged" | explore_scale_eight.py: ARM D8, k=16, conv, ITERS 43200, SEEDS 290-299, LOCAL3 + SLOW + KEYMASS as in batches 17-18; records' sched string identical across arms | MATCH | |
| 18 | tex4:235 | S66 reference | "the $N = 256$, three-layer reference at $8/10$" | N256_L3 8/10 (290 MERGED, 2 share; 296 MERGED, 3 share) | MATCH | |
| 19 | tex4:235 | S66 D128 | 8/10; 2 vs 2; d = 0; "carries" | 8/10; 2 / 2; d = 0; p = 1; carries | MATCH | |
| 20 | tex4:235 | S66 L2 | 10/10; 2 vs 0; p = 0.5; d = +2; "carries" | 10/10; 2 / 0; two-sided p = 0.5 (one-sided 0.25); carries | MATCH | The p-value is two-sided, as E prints it. |
| 21 | tex4:235 | S66 transitions and probe | earlier median transition (15,600 against 21,600); higher probe accuracy at the first check | Medians: 21600 (reference), 15600, 15600. Probe at 4800: median 0.094, 0.188, 0.312 | MATCH | These are E's observations, not readings. Neither has a test. The draft states them without marking them as descriptions. |
| 22 | tex4:235 | S66 failures | "every failure in the three arms had spent its three splits and ended merged" | 4 failures (ref 290, 296; D128 293, 299), each with splits = 3 and class MERGED | MATCH | Split totals 21, 21 and 19. |
| 23 | tex4:235 | S67 reading | "so the reading 'not capacity-bound at $N = 256$' applies" | ORC_P16_N256 2/2 ≥ 1 → "not capacity-bound at N=256" (cap_reading) | MATCH | The reading rests on 2 oracle seeds. |
| 24 | tex4:235 | S67 P=8 | 9/10, 10/10; median 7200 | P8_N256 9/10 (272 MERGED, 2 share; 277 at 37200), median 7200. P8_D128 10/10, median 7200. 1 / 0, p = 1 | MATCH | The P=8 oracles are 4/4, all at 2400. |
| 25 | tex4:235 | S67 P=16 | 0/10 at N=256 against 4/10 at D=128; 4 vs 0; p = 0.125; "printed, no reading" | 0/10 and 4/10 (transitions 19200, 26400, 37200, 38400; median 31800); 4 / 0; two-sided p = 0.125 (one-sided 0.0625) | MATCH | The docstring prints the learned arms with no reading, and the draft marks them so. |
| 26 | tex4:235 | S67 failure classes | "most of the failures at both widths ending one-to-one and unbound, which the screen cannot tell from slow" | N256: 6 STREAM-PARTIAL one-to-one, 4 MERGED. D128: 6 of 6 STREAM-PARTIAL one-to-one. 12 of 16 overall | MATCH | |
| 27 | tex4:235 | S66 schedule note | "no run in S66 had its transition at or before the first check or the learning-rate switch, so the trigger's absolute schedule can stand", "as a description" | Earliest transition 10800 (D128_L3 294); none ≤ 4800, none ≤ 2400 | MATCH | The docstring's ALSO REPORTED rule is set on the arm *median* (≤ 4800 means size-dependent). The draft's per-run statement is stronger and still holds. |
| 28 | tex4:235 and 473 | E's lead | P=16 at D=128 against D=32, "clearest lead, weak"; longer budget; pre-register how a one-to-one unbound run counts | README "Handed back" block | MATCH | This is text, not records. |
| 29 | tex4:366 | S67 in the verdict paragraph | "$N = 256$ is not capacity-bound at sixteen keys per stream (S67), where the learned recipe binds $0/10$ against $4/10$ at $D = 128$" | as rows 23 and 25 | MATCH | |
| 30 | tex4:508 | provenance | "S64 width, S65 depth, S66 eight streams, S67 capacity (complete, \texttt{e3f5809})" & \texttt{e3f5809} & 186 | e3f5809 is the completion commit. 186 runs = S64 60 + S65 48 + S66 30 + S67 48 (oracles included). No failures or retries in the stores | MATCH | |

Also checked: the scale ranges in tex4:441 ("two to six screened", "to 128 screened", "to 1024 screened") MATCH S64 and S65. Only the forget-gate wording in that line changed in v4. The conclusion (tex4:468) and the methodological findings (tex4:430-445) do not mention batch 19.

## Readings vs docstring (readings commit vs first record)

| screen | rule (docstring, fixed at 617681d, 2026-10-09 06:17:08) | first record | recomputed reading | README | draft |
|---|---|---|---|---|---|
| S64 width | "scale-free" if every tested d >= −1; "width-sensitive" if any d <= −4 (FREE_D, SENS_D = −1, −4) | 7f1a71d, 2026-10-09 09:20 (ORC_D128 270). The segment-1 commit 542eb33 (09:01) has empty stores | scale-free (+1, +1, +1, +1) | scale-free | scale-free (tex4:233) |
| S65 depth | the same rule against L3; RES6 only if L6 − L3 <= −2 | e44e794, 2026-10-09 14:06 | scale-free (0, 0, −1); RES6 not run | same | same |
| phase-2 choice | best_candidate: most bound routed → earlier median → larger model | decision stored 2026-10-09 16:40:29 UTC, committed 1840bcc 16:40:36 with S66/S67 stores empty | D128 (larger model than N1024). L2 = L4 on every term; order of arms decided (not in the rule) | disclosed | not mentioned (row 6) |
| S66 eight | "carries" if d >= −1 per arm, else "does not carry"; McNemar two-sided printed | 1bbf799, 2026-10-09 17:21 (N256_L3 292) | D128_L3 carries (d = 0); N256_L2 carries (d = +2) | same | same (tex4:235) |
| S67 capacity | "capacity-bound at N=256" iff ORC_P16_N256 0/2 and ORC_P16_W >= 1/2; else "not capacity-bound" (N256 ≥ 1) or "neither"; learned arms printed, no reading | 9ee5263, 2026-10-10 01:56 (ORC_P16_D128 271) | not capacity-bound at N=256 (N256 2/2, D128 1/2) | same | same; learned arms marked "printed, no reading" |

- The READING blocks are byte-identical from 617681d to e3f5809. Commit 8cff21e changed only report code: the RES6 table's seeds and the format of the S67 count strings.
- Every rule predates its first record. The phase-2 decision predates every S66 and S67 record.

## Could not check

- Nothing numeric was left unchecked.
- E's advice and the README's prose (row 28) are text, so they were compared with the README rather than with the records.
- The power figures use the independence approximation (stats_checks.py §7). The true paired correlation would change them.
