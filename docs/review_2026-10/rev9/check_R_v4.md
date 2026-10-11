# check_R_v4: session R's Phase 4 in the Revision 9 draft, fourth version

This check covers the draft at `c66cc76` on `claude/rev9-draft` (tex4:NNN is `multichannel_hebbian_report_v9.tex` there, 564 lines), compared with the third version at `a5f67a2` (tex3).

R's records were read with `git show` from `claude/replicate-R`:
- `1f8be04`, the final Phase 4 commit;
- `0364e28`, the interim records;
- `464be8d`, the pre-registration.

The script is `rev9/check_R_v4.py`. It uses only the standard library, writes nothing, and its output is quoted as §N below. It recomputes R's outcome with `replication_check.py`'s `r_transition` and `r_bound_routed`, which it extracts with `ast` so that the module never runs. Bound routed means three things together:
- a transition from evaluations that hold ≥ 0.95 to the end;
- eight distinct channels in the map;
- every `per_stream_acc` ≥ 0.9.

## Which rule was run

R ran and read Phase 4 under the **first rule**. That rule, from `phase4.py`'s docstring at `464be8d`, has three readings:
- **"consistent with the report"**: the split arm is pooled over 40 seeds (Phase 2's WIN3_SPLIT_D8 on 900–919 plus WIN3_SPLIT_D8_ML on 920–939), its Wilson interval overlaps [0.835, 0.986], and the exact one-sided McNemar test against WIN3_SLOW_D8 gives p < 0.05;
- **"a lower rate"**: the interval's upper bound is below 0.835;
- **"neither"**: otherwise.

The rule did not change at any point (§0):
- Between `464be8d`, `0364e28` and `1f8be04`, the design and readings text of `phase4.py` and all of its code are byte-identical. Only the RESULT block changed, from "(not yet run)" to the numbers.
- `464be8d` has no `records/phase4.jsonl`.
- The interim 21 records are the first 21 final records, unchanged.
- The log has a single "42 jobs to run" header.
- Phase 2 and Phase 3 records are unchanged since `100dced`.

**No part of the amended prompt was run.** The amended prompt is `db22857`, `docs/reading/parallel_sessions_prompts.md:858-884`. Against each of its parts:
- No record exists on 914 or 915 under the main line's protocol (part (a)).
- No seeds 940–951 were run, and the arm used on 920–939 is WIN3_SPLIT_D8_ML pooled with Phase 2, not a 32-seed fresh count (part (b)).
- There is no COPY_ONLY or RESET_ONLY arm in `replicate.py`, and no mechanism arms (part (c)).
- Nothing on R's branch mentions the amended prompt.

**Timing (§0, §9).** The records carry no timestamps. However, the run order in `replicate.py` with four workers and `imap_unordered`, replayed with the recorded `wall_s`, reproduces the record order exactly. Its makespan is 9.40 h. Measured back from the final commit (04:26:27), Phase 4 therefore started **no later than 19:02:43 UTC on 10 October**:
- that is within 70 s of the pre-registration commit `464be8d` (19:01:33);
- it is before the referee's replication review `01bee1f` (19:08:20).

So the amended prompt, written after that review, could not have preceded the runs. "Pre-registered before any run" is consistent with git but holds only if R launched within about a minute of its commit and committed the report within about a minute of the last run. The records cannot confirm it.

## Recomputed

| item | recomputed (R's outcome) | R / draft |
|---|---|---|
| ORACLE_ADAM 920, 921 | both bound, transition 2400 → validity PASS | 2/2 at 2400 |
| Phase 2 WIN3_SPLIT_D8 900–919 | 17/20 [0.640, 0.948] | 17/20 |
| Phase 2 WIN3_SLOW_D8 900–919 | 7/20 [0.181, 0.567] | 7/20 |
| R1 / R2 (900–909) | 9/10, 9/10 | 9/10, 9/10 |
| Phase 4 WIN3_SPLIT_D8_ML 920–939 | **18/20** [0.699, 0.972] | 18/20 |
| Phase 4 WIN3_SLOW_D8 920–939 | 5/20 [0.112, 0.469]; 13 vs 0, p = 1.22e-4 | 5/20 |
| 40 pooled, split | **35/40** [0.7389, **0.9454**] (17 from Phase 2, 18 from Phase 4) | 35/40 [0.739, 0.945] |
| 40 pooled, no split | 12/40 [0.181, 0.454] | 12/40 |
| discordant pairs | **23 vs 0**, p = 1.19e-7 (= 2^-23) | 23 vs 0, 1.2e-7 |
| reading (phase4.py rule) | overlap True, p < 0.05 → **"consistent with the report"** | same |
| stored flags vs recount | 0 disagreements in phase2, phase3 and phase4 | — |
| ML failures | 920 (splits at 4800, 12000, 16800; streams 0 and 5 share ch 0); 933 (4800, 9600, 19200; streams 0 and 7 share ch 4) | 920, 933 |
| Phase 4 splits | 44, all on target by the labels; latest at 24000 (D-5 inert) | 44 on target |
| main line (X + L) | 38/40 [0.835, 0.986], 15/40, 23 vs 0; X 555 and 556 each fired 3 splits and ended with a shared channel | 38/40, 15/40, 23 vs 0 |
| blocks, Phase 2 (per-seed) | 13 of 192 checks, 12 eligible (906: 1/1, bound routed; 914: 12/11, not bound routed) | 13/192, 12 |
| blocks, R1 (fixed) | 1 of 91, 1 eligible (905, bound routed) | same |
| blocks, R2 (per-seed, body keys) | 0 of 90 | same |
| blocks, Phase 4 (fixed) | 2 of 172, 0 eligible; converse 0 everywhere | same |
| fixed probe pooled / per-seed pooled | 3 of 263 / **13 of 282** (Phase 2 alone 13 of 192) | 3/263 / 13/192 |
| P("a lower rate" given 17/20 kept) | 0.95: 0.000; 0.85: 0.001; 0.80: 0.010; **0.75: 0.041**; 0.70: 0.113; 0.65: 0.238 | — |
| amended (b), x ≤ 27/32 | size 0.020, power 0.536 at 0.85, 0.796 at 0.80 | 0.020, 0.54 |
| CPU in R's records | no CPU, host or device field in any phase; torch "2.14.1+cu130"; summed `wall_s` 37.4 h (P2), 15.2 h (P3), **37.2 h (P4)**, 4 workers × 1 thread | "no CPU model" |

## Rows: DISAGREE first

| # | tex4 line | item | draft says | recomputed | status | note |
|---|---|---|---|---|---|---|
| 1 | 328 | Wilson interval vs 0.95 | "R's own statement …: the point estimate 0.875 sits below 0.95, **the interval includes 0.95** and excludes rates below 0.74" | Wilson(35, 40) = [0.7389, **0.9454**]: the upper bound is below 0.95, so the interval **excludes** 0.95 (§3) | **DISAGREE** | R's `replicate_report.md` (Phase 4, "What the 40 seeds can and cannot say") makes the same error; the draft repeats it as R's statement. The pre-registered reading is unaffected because it tests overlap with [0.835, 0.986]. The sentence should say that the interval reaches 0.945, just below the report's 0.95. |
| 2 | 36 | the abstract's "never saw the code" | "A clean-room replication from the report alone, **by a session that never saw the code**, bound … 17/20 … and **18/20 on twenty further seeds once it had the main line's protocol** (35/40 against 12/40 pooled, 23 vs 0)" | R read the main line's code in Phase 3 (`DIFF.md` lists `test_window_gate.py` and its imports; tex4:326 says "only then fetched the main line and diffed"). That was before Phase 4 (`464be8d`), and Phase 4's arm exists because of that diff. | **DISAGREE** | The 18/20 and the 35/40 were not produced by a session that never saw the code. The 7e-9/7e-8 agreement in the same sentence also came from Phase 3. Suggested wording: "that had not seen the code when it ran its first twenty seeds". |
| 3 | 354 | "from the text alone" | "Stands, and was replicated **from the text alone** (17/20, then 18/20 under the main line's protocol, 35/40 pooled)" | Only 17/20 (Phase 2) was from the text alone. The 18/20 used WIN3_SPLIT_D8_ML (D-1 and D-2 taken from the main line's code), and the 35/40 pools both. | **DISAGREE** | The parenthesis puts Phase 4 counts under "from the text alone". |
| 4 | 36, 62, 354 | "the main line's protocol" | "18/20 … once it had the main line's protocol" (36); "18/20 under the main line's protocol" (62, 354); the 9657edb commit message says "18/20 on fresh seeds under the main line's protocol" | WIN3_SPLIT_D8_ML set the main line's choice for D-1 (fixed probe, Generator(12345)) and D-2 (32 body keys), and ORACLE_ADAM did the same for D-3. Still R's own: D-4 (order of trigger and early stop), D-5 (last check; inert, latest split 24000), and D-6 to D-10 (generators, initialisation, noise draws, and the stream-to-channel map on 2048 held-out sequences rather than a 256-sequence probe) | PARTIAL | R's own words are "the split as the main line specifies it" and "main line's specification" (`phase4.py`:5, `replicate_report.md`). "The main line's split" would be exact. |
| 5 | 354 | "replicated" | "Stands, and was replicated" | R's mechanical readings were "partially replicated" in Phase 2 (17/20 < 18/20; tex4:328's own words) and "consistent with the report" in Phase 4. Neither reads "replicated". | PARTIAL | 920–939 alone would meet Phase 2's bar (18/20; 13 vs 0, p = 1.2e-4), but `phase4.py` prints that count as secondary, with no reading. The word is an author's assessment and should be marked as one. |
| 6 | 205 | per-seed versus fixed probe | "with the fixed probe the mode occurred at 3 of 263 checks and never decided a run, with one probe drawn per seed at 13 of 192" | The fixed-probe 263 pools R1 (91) with Phase 4 (172). The per-seed 192 is Phase 2 only and leaves out R2, also per-seed (0 of 90). Pooled the same way, per-seed is **13 of 282**. | PARTIAL | tex4:328 labels its 13 of 192 "(Phase 2)", which is exact. tex4:205 does not. |
| 7 | 205, 328 | "decided one run in twenty (914)" | the mode "decided one run in twenty (914)" | 914 has 12 blocked checks (11 eligible) from 16,800 to 43,200, probe 0.953–1.000, evaluation 0.871–0.887, one split at 14,400, and ended not bound routed. 906 had 1 eligible block and bound routed. | PARTIAL | That the block decided 914 is R's inference. The test of it, amended part (a) (914 under the main line's protocol), was not run, and tex4:328 itself says the reruns "were not run on the one failure a protocol difference plausibly explains". One of the 11 eligible checks (43,200) lies past the main line's last check (D-5). |
| 8 | 328 | "about 150 seeds per arm" | "separating 0.85 from 0.95 would need about 150 seeds per arm" (R's statement) | Two-arm comparison, power 0.8: about 141 per arm two-sided and about 111 one-sided (normal approximation). One-sample against a fixed 0.95: **52** (x ≤ 46; size 0.045, power 0.81), the draft's own figure at tex4:428. | PARTIAL | The quote is accurate, but beside tex4:428 the reader gets 150 and 52 without being told that they answer different questions (a new arm against a new arm, against fresh seeds tested at a fixed rate). |
| 9 | 391 | what the first rule could detect | "would have detected only rates near 0.70" | P("a lower rate") is 0.113 at 0.70, 0.238 at 0.65 and 0.041 at 0.75. | PARTIAL | The referee's review (`01bee1f`, line 148) says "only rates near 0.70 **or below**". tex4:328 and tex4:428 say "about one time in nine" at 0.70, which is exact. Line 391 drops "or below" and implies that 0.70 is detected. |
| 10 | 328 | pre-registration | "under a rule pre-registered before any run (`464be8d`)" | git agrees: no Phase 4 record at `464be8d`, and the readings and code are unchanged through `1f8be04`. The replayed schedule puts the start no later than 19:02:43 UTC, within 70 s of `464be8d` (19:01:33). | NOT RECORD-CHECKABLE | The records carry no timestamps. The claim holds only for an immediate launch. The same bound puts the start before the referee's review (19:08:20). |
| 11 | 328 | the amended reading's history | "written after the referee's review but existed in no commit until `db22857`" | Chat is not recorded. The prompt's own header (`db22857`:858) says "given to the user in chat on 10 October after the referee's replication review". | NOT RECORD-CHECKABLE | The rest of the sentence is checkable; see row 22. |
| 12 | 25 | title | "17/20 in a replication from the text alone" | 17/20, Phase 2 | MATCH | |
| 13 | 36 | abstract counts | 17/20 against 7/20 (10 vs 0); 18/20; 35/40 against 12/40 pooled, 23 vs 0 | 17/20, 7/20, 10 vs 0; 18/20; 35/40, 12/40, 23 vs 0 | MATCH | The numbers are right; see rows 2 and 4 for the wording. |
| 14 | 36, 330 | UNSTATED labels | "44 details it logged (34 of them guesses)"; "four outright and three only in part (the gate's input among the four) … 34 … 20 … 12 … 2 … 3 entries of process" | stated: U-4, U-8, U-9, U-40 (4; U-9 is the gate input); in part: U-3, U-13, U-25 (3); [A] 20, [B] 12, [C] 2 (34); process: U-42, U-43, U-44 (3); total 44 | MATCH | U-3 is labelled "[A in code, stated in report]". Counting it "in part" is a classification choice. |
| 15 | 41 | Phase 4 complete | "G's S83, H's S86 and R's Phase 4 are complete" | 42 of 42 jobs, one launch, no restart header | MATCH | It is complete under the first rule; the amended parts were never run. |
| 16 | 41 | CPU | "R's records carry no CPU model" | no CPU, host or device field in phase2, phase3 or phase4; torch "2.14.1+cu130"; Phase 4 `wall_s` 37.2 h | MATCH | Phase 4 adds no CPU record either. The prose says CPU, 4 × 1 thread (`replicate_report.md`:109). |
| 17 | 62 | index row | "17/20 against 7/20 (10 vs 0) from the text alone; 18/20 under the main line's protocol; 35/40 against 12/40 pooled (23 vs 0), 'consistent with the report'" | as recomputed; the reading follows from the `phase4.py` rule | MATCH | See row 4 on "protocol". |
| 18 | 328 | design | "the split as the main line specifies it (the fixed probe, the body's key positions) on twenty fresh seeds (920–939) beside the no-split arm, with the single-rate oracle as validity (2/2 at 2400)" | WIN3_SPLIT_D8_ML (D-1 and D-2), WIN3_SLOW_D8 920–939, ORACLE_ADAM 920 and 921 bound at 2400 | MATCH | |
| 19 | 328 | the pooled count | "pooled with Phase 2, the split arm bound routed 35/40 (Wilson [0.739, 0.945]) against 12/40 …, 23 vs 0 (one-sided p = 1.2e-7)" | 35/40 [0.7389, 0.9454]; 12/40; 23 vs 0; p = 1.19e-7 | MATCH | The draft says the 40 pool Phase 2's seeds. It implies, without stating, that those 20 used R's own probe and key positions (`phase4.py`:16-20 says so outright). |
| 20 | 328 | the reading and the report | "'consistent with the report' applies (the intervals overlap; the report's own pooled count is 38/40 against 15/40, also 23 vs 0)" | overlap True, p < 0.05; main line 38/40 [0.835, 0.986], 15/40, 23 vs 0 | MATCH | |
| 21 | 328 | fresh seeds and failures | "On the fresh seeds alone, under the main line's protocol, the split arm bound 18/20, the count Phase 2's bar had asked for; both failures were two streams sharing a channel after three splits, as X's were" | 18/20; 920 and 933 each fired 3 splits and ended with a shared channel; X 555 and 556 the same | MATCH | `phase4.py` attaches no reading to this count; the draft claims none. See row 4 on "protocol". |
| 22 | 328 | the first rule and the amended reading | "The rule itself, an interval-overlap test pooled with Phase 2, detects a true rate of 0.70 about one time in nine, which the referee had pointed out; an amended reading … (32 fresh seeds, 'below 0.95' at x ≤ 27, size 0.020, power 0.54 at 0.85) … existed in no commit until `db22857`, after Phase 4 had started, and was not used" | P = 0.113 at 0.70 (the review, `01bee1f`:64-70, gives 0.041 at 0.75 and 0.113 at 0.70); size 0.020, power 0.536; `db22857` at 03:01 is after `0364e28` (00:02, 21 records); no amended arm, seed or rule in any record or in `phase4.py` | MATCH | The draft says plainly which rule ran and that the amended one did not produce the result. Phase 4 had in fact started before the review itself (row 10). |
| 23 | 328 | probe counts | "3 of 263 checks and never decided a run; … (Phase 2) … 13 of 192 and decided one run in twenty (914)" | 3/263, 1 eligible, in 905 (bound routed); 13/192, 12 eligible | MATCH | The counts match; see row 7 on "decided". |
| 24 | 328 | R's advice | "specify the probe as one fixed draw, say that 64 queries can over-read a merged model, and gate the trigger on more queries" | `replicate_report.md` Phase 4, "For the report" | MATCH | |
| 25 | 347 | R's branch | "the amended reading existed in no commit until `db22857`; R's branch carries the first rule" | `phase4.py` at `1f8be04` carries the first rule unchanged | MATCH | |
| 26 | 391 | the amended reading's commit | "existed in no commit until `db22857`, after the referee's second pass had found no record of it, and R's branch carries the first rule" | `db22857`'s commit message says the same; `phase4.py` is unchanged | MATCH | This paragraph does not say that Phase 4 ran and was read under the first rule; tex4:328 does. See row 9. |
| 27 | 468 | conclusion | "in 17 of 20 in a replication from the text alone" | 17/20 | MATCH | Phase 4 is not mentioned here, which is consistent. |
| 28 | 473 | next steps | "a calibrated reading for any further replication phase" (replaces tex3's "session R's Phase 4 under the amended reading") | the amended Phase 4 was not run | MATCH | |
| 29 | 513 | provenance | "Phase 2 (`de8df0f`), Phase 3 diff and reruns (`00990ea`, `100dced`); Phase 4, the 40-seed count (rule `464be8d`; report `1f8be04`) & `1f8be04` & 42; 20; 42" | the commits are those; records have 42, 20 and 42 lines | MATCH | |

**Counts: 3 DISAGREE, 6 PARTIAL, 2 NOT RECORD-CHECKABLE, 18 MATCH (29 rows).**

## Not checked

- Whether R was ever given the amended prompt in chat. Chat is unrecorded; R's branch never mentions it.
- The actual start time of Phase 4. The records have no timestamps, so §9's bound comes from a replayed schedule.
- R's wall-clock environment and CPU model: none are recorded.
- tex4:430's "as `test_delta_recipe` prints", which is outside R's scope.
