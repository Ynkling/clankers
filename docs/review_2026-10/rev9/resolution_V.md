# Second pass, the referee's own rows

*Referee V, 11 October 2026. Second draft: `f67bc07` (claude/rev9-draft, committed 02:43:57 UTC). `tex2:NNN` is its line. Rows are those of `check_V.md` and items A1, A4–A9, B9–B11, C2 and section 4 of `revision_9_check.md`.*

## Resolution of the first-pass rows

| first pass | item | second draft | status |
|---|---|---|---|
| tex:36, 63 (A6) | "all reproduce" | tex2:36 "140 of 156 reproduce; the seven that disagreed were statements"; tex2:63 "140 of 156 numbers reproduce, the rest statements" | RESOLVED |
| tex:41 (A7) | every reading of every screen re-derived | tex2:41 "re-derived every pre-registered reading of the sessions' later reports ... and checked that the earlier ones were committed before their first run"; tex2:330 the same | RESOLVED. Note: tex2:330 says the passes covered "the exploratory sessions' reports 3–6"; the 808 rows cover F 3–5, G 3–4, H 4–5 and I 1, and report 6 (F's S80) was checked in rev9/check_F, not in the 808 |
| tex:330 (A9) | "a third CPU model" | tex2:330 "on a 2.10 GHz Xeon, the model of X's slower host" | RESOLVED |
| tex:36 (B9) | "to 10⁻⁸" | tex2:36, 327: "7×10⁻⁹ in the gates and 7×10⁻⁸ in the logits" | RESOLVED |
| tex:36, 323 | "the 44 details ... are now stated"; "Appendix B states all of them" | tex2:36 "stated in an appendix"; tex2:327 lists the six added entries; Appendix B (tex2:521–535) now has float32, the forward pass to the logits, `Generator(seed + 31337)`, W_h drawn and frozen, the perfect gate's model, the tie rules (`torch.argmax`, lowest index) and `torch.std` with Bessel's correction | RESOLVED (for the item-by-item check against the code, see `check_appendixB_v2.md`) |
| tex:323 (B10) | "all determined by the text" | tex2:327 "By R's own labels on its 44 entries, the text determined most of the architecture ... and was ambiguous or silent on **nine**" | **NEW ERROR.** R's labels (UNSTATED.md): 20 [A], 12 [B], 2 [C], 2 [stated + A], 1 [A in code, stated in report], 4 [stated], 3 unlabelled (process). So **37 of 44** are entries where the text was silent, ambiguous or carried over, not nine. Of the four examples, "the gate's input" is U-9, labelled [stated] |
| tex:323 | "nothing ... contradicts ... nothing it had misread" without qualifier | tex2:327 "in the paths of our code it read (not the CHECK chain, the report printing or the hinge paths)" | RESOLVED |
| tex:323 | "Nine of the 44 guesses were material and all nine matched" | tex2:327 "R lists eleven guesses that matched our choices, among them: ..." | PARTLY. DIFF.md's list of matched points has 11 bullets. Two of the draft's examples are not on it: the 0.02 embedding initialisation (in replicate_report.md:89-96 and D-8) and the map (filed in DIFF.md as D-10, a difference). The draft says the map "matched in statistic only", which is right |
| tex:321 (A4) | Phase 4 "calibrated ... running" | tex2:325 "A fourth phase was proposed ... R's branch still carries the first pre-registered rule ... detects a true rate of 0.70 about one time in nine, **the amended reading has not been adopted there**, and no Phase 4 record exists" | PARTLY. The facts about R's branch are right. But no amended reading exists in any commit: `docs/reading/parallel_sessions_prompts.md:731-742` at the main line's head `0282955` still carries the overlap rule, and no other file holds a Phase 4 reading |
| tex:382 (A5) | "Both were corrected" | tex2:388 "The referee's review corrected both in principle; the amended Phase 4 reading **was issued to session R as a prompt**"; tex2:344 "(the prompt was issued; R's branch had not adopted it)" | **NOT.** The same unsupported claim in a new form. The only Phase 4 prompt on record is the uncalibrated one (`parallel_sessions_prompts.md:731-742`, `7cc87e6`, unchanged at `0282955`). If a corrected prompt was sent outside the repository, the draft should say so and cite it; as written, it states as fact what no record shows |
| tex:419 | "detects only rates near 0.70" | tex2:425 "detects a true rate of 0.70 about one time in nine and rarely any rate above 0.65" | RESOLVED |
| tex:205, 384 (A1) | records do not store the probe | tex2:205 and tex2:390 correct (6 of 393; 2 blocked by the probe alone, both fired next and bound; none in the reset arm) | PARTLY: **tex2:427 ("Store what the trigger saw") still says "The split's decisions depend on a probe whose readings the records do not keep"**, contradicting tex2:205 and 390. Only the per-check per-stream mass is missing from the records (`stream_gate` is stored at every evaluation, which includes every check step), so "belong in the record" is already met |
| tex:325 | R's CPU | tex2:41 "(R's records carry no CPU model)"; tex2:331 the same | RESOLVED |
| tex:128 | "controls the FDR" | tex2:128 "assessed under Benjamini–Hochberg at 0.05 after the fact", L's five named | RESOLVED |
| tex:336 (A8) | 90 L rows | tex2:340 "all 70 L rows" | RESOLVED |
| tex:334 | "at every size" | tex2:334 "at two, four and eight streams on the grouped layout; no external model ... has been trained on the task" | RESOLVED there. **NOT** at tex2:458 (Baselines): "One channel of either write rule is run at every size" |
| tex:338 | "each is marked"; "noise helps a little" listed | tex2:342 drops "noise helps a little" and says the unmarked places are now marked | PARTLY: depends on the session checks (`resolution_F/G/H.md`) |
| tex:445 | "60–80%"; "9–13 h" | tex2:451 "closing 44–83% of each seed's gap"; "estimates ... from a CPU profile, at 9–13 hours on an RTX 3070 Laptop or 14–21 on a GTX 1660 SUPER" | RESOLVED |
| tex:36 (C2) | "settled" | tex2:36 "Exploratory screens, which are not claims, answered three questions ... each on ten to twenty paired seeds" | RESOLVED |
| tex:36, 314, 349, 459 (C1) | delta parity | tex2:36, 252, 355: "not shown worse" with the counts; tex2:252 "one-sided bounds at the ceiling and not parity"; tex2:465 "not shown worse ... on screens at every size tried" | RESOLVED in the places listed (the session-level check is in `resolution_F.md`) |
| tex:23–25, 428 | title; "The title says so" | tex2:23–25 "A Learned Partition of Working Memory on a Synthetic Binding Task with Local Cues"; subtitle "a channel split aimed at the task's key positions ... of the grouped layout ... and 17/20 in a replication from the text"; tex2:432 "The title and subtitle say so" | RESOLVED |
| tex:459 | "Every number here was recomputed" | tex2:465 "A referee recomputed the numbers of Revision 8 and of this revision's first draft from the records and found them to reproduce; the statements it found wrong in each draft are corrected and listed" | PARTLY. The new numbers of this version (S82 report 5, S84, S85, the 97 of 107, the Amendment 1 description) were not in either pass until this one |
| tex:551 | "Jegou"; not read in full | tex2:451 "(the task family of \cite{bietti, edelman}, cited for it; neither paper has been read in full)"; tex2:557 `J\'egou` | RESOLVED |

## New errors, or items made stale by records pushed before this draft

| tex2 | draft | record | status |
|---|---|---|---|
| 327 | the text "ambiguous or silent on nine" of R's 44 entries | 37 of 44 carry [A], [B] or [C] (above) | NEW ERROR |
| 325, 344, 388, 470 | an amended Phase 4 reading "was issued to session R as a prompt"; "session R's Phase 4 under the amended reading" | no such prompt or reading in any commit; the only Phase 4 prompt (`parallel_sessions_prompts.md:731-742`) is the overlap rule | NEW ERROR (unsupported) |
| 427 | "a probe whose readings the records do not keep" | the records keep them (tex2:205, 390) | NOT (A1 left standing here) |
| 488 (prov. table) | split_copy L: "running" | tex2:41 "L's half ... has not run"; no `results/L/split_copy_results.json` at the main line's head | NEW ERROR (internal contradiction) |
| 489 (prov. table) | delta_recipe: "spec in specs/ (this revision); not written" | `test_delta_recipe.py` committed at `a581027`, 02:12:29 UTC, 31 minutes before this draft | NEW ERROR (stale) |
| 510 (batches table, R row) | "Phase 4 running" | tex2:41 "R's Phase 4 has not started"; no records | NEW ERROR (internal contradiction) |
| 511 (batches table, V row) | V's work up to `0e6c189` | tex2:344 cites `revision_9_check.md` (`e8fbb25`) and tex2:252 the delta spec review (`fad1489`); neither is in the row | PARTLY |
| 344 | "five ranges stated for every run that held for the median run" | revision_9_check.md lists three such ranges (B6, B7, B8) | PARTLY |
| 344 | "Every item is corrected in this version" | see this file and `resolution_*.md`: items NOT or NEW ERROR remain | NOT (as of this check) |
| 436 | the all-position count | `count_allpos.py`: 50 runs, 107 firings; all-position labelled 97 (window_gate X 46/47, L 30/38, split_copy X 21/22), key-position 107, pairs coincide 46; the stored `all_ok` / `key_ok` equal the labels recomputed from `map_before` at all 214 | MATCH |
