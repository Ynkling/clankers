# check_R_phase5: session R's Phase 5 (copy vs reset vs one delta channel, seeds 940–959)

This checks R's Phase 5 on `claude/replicate-R`:
- the protocol `550a736` (2026-10-12 22:47:46 UTC);
- the interim records `29ff87a` (2026-10-13 04:18:51);
- the report commit `31c39fc` (2026-10-13 18:41:38), which is R's head now.

The report is `replicate_report.md`:166–223 at `31c39fc` (cited as `rr:N`), together with the RESULT block of `phase5.py`:46–56 (cited as `ph:N`).

The script is `docs/review_2026-10/check_R_phase5.py`. It uses only the standard library, prints, and writes nothing. It reads everything with `git show` and runs no code from R or the main line. Its output is quoted as §N and given in full at the end.

The outcome was recomputed two ways, and the two give the same counts on every arm (18, 9 and 0):
- **R's definition:** `replication_check.py`'s `r_transition` and `r_bound_routed`, taken by `ast` as in `check_R_v6`. DELTA1 counts as bound when `r_transition` is not None.
- **The main line's definition:** `test_stream_recipe.outcome`, through `rlib.py` (`bound` and `bound_routed`). It was applied to R's records through an adapter (curve = evals; R's map as a one-hot stream gate) and to the main line's `test_split_copy` records (X `592c574`, L `ffb0771`).

## Verdict

- **Counts:** 38 rows: **MATCH 32, PARTIAL 2, DISAGREE 0, NOT RECORD-CHECKABLE 4**.
- **Every number in R's Phase 5 report reproduces from the records:**
  - every count, interval, discordant pair and p-value;
  - the "consistent" label on each arm;
  - every diagnostic count;
  - every number R gives for the main line.
- **R's stored flags agree with the recomputation:** no stored `bound`, `bound_routed` or `transition` differs from the recomputed value (§3).
- **Pre-registration: the commit order is clean.**
  - The protocol (`phase5.py` above its RESULT block, `replicate.py`, `analyze.py`, `phase4.py`, `UNSTATED.md`) was committed before any Phase 5 record and is byte-identical at the report commit. The only change is the RESULT block, filled in from `(not yet run)`.
  - The seeds 940–959 had not been used anywhere before.
  - **Not provable from git:** that the protocol was committed before the first *run*. The records carry no timestamps. A replay of `wall_s` on 4 workers puts the start **no later than 22:49:48 UTC, 2.0 min after the protocol commit**. A start a few minutes before 22:47:46 cannot be excluded (row 2).
- **The two PARTIAL rows:**
  - (row 35) "From the written specification alone" is true of the numbers but not of the COPY and RESET arms: COPY is R's Phase 4 recipe, which took the main line's probe and key positions from the main line's code, and RESET uses the same trigger.
  - (row 32) "Per-stream accuracy is roughly even within a run" fails on 4 of 20 DELTA1 runs.

## Pre-registration order (§0, §1)

| commit | time (UTC) | what | files |
|---|---|---|---|
| `550a736` | 10-12 22:47:46 | protocol | `phase5.py` (+102), `UNSTATED.md` (+51: U-50..U-70), `replicate.py` (+60 −15: `delta_attend`, `memory="delta"`, `Split(op="reset")`, arms RESET_D8_ML / DELTA1 / ORACLE_DELTA, the `--phase 5` job list), `test_replicate.py` (+95) |
| `29ff87a` | 10-13 04:18:51 (+5.52 h) | interim records | `records/phase5.jsonl` (19), `records/phase5.log` (header "64 jobs to run" + 19) |
| `31c39fc` | 10-13 18:41:38 (+19.90 h) | report | `phase5.py` (+10 −2, the RESULT block only), `phase5.jsonl` (+45), `phase5.log` (+45), `phase5_table.md`, `replicate_report.md` (+59) |

**What git confirms:**
- There is no `records/phase5*` at `550a736`.
- `phase5.py` at `29ff87a` equals `550a736`.
- `phase5.py` at `31c39fc` equals `550a736` above "RESULT (" and in all code below the docstring.
- `replicate.py`, `analyze.py`, `phase4.py`, `test_replicate.py`, `UNSTATED.md`, `DIFF.md` and `.gitignore` are identical at all three commits.
- `records/phase2/3/4.jsonl` are identical from `1f8be04` to `31c39fc` ("Phases 2–4 records are not touched", ph:14).
- The 19 interim lines are byte for byte the first 19 lines of the final file. The log has one header, so there was one start.
- The log's order equals the jsonl's order.

**Replay** (`wall_s`, 4 workers, jobs in `replicate.py`'s phase-5 order, each job to the first free worker, as `Pool.imap_unordered` does):
- The replayed completion order equals the records' order.
- The 19th completion falls at +4.69 h, the 20th at +5.59 h and the 64th at +19.86 h. The sum of `wall_s` is 78.35 h.
- **Latest start:**
  - The interim commit holding 19 records allows a start as late as 23:37:19.
  - The report commit holding all 64 allows one no later than **22:49:48**, which is 2.0 min after the protocol commit.
- **Lower bound, under assumptions:** if the interim commit was made before the 20th record existed and there was no overhead beyond `wall_s`, the run started after 22:43:25.
- **Start at the protocol commit:** that puts the last record (RESET_D8_ML 959) at 18:39:35 or later, 2.0 min before the report commit.
- **Reading:** this is consistent with R's statement and resembles Phase 4, whose replayed start was 70 s after `464be8d`. But real per-job overhead (process spawn, model build, all outside `wall_s`) moves the latest start earlier. Git therefore cannot order the first run and the protocol commit to within a few minutes.

**Seeds 940–959 (§2): MATCH.**
- **R's earlier records:** phase 2 used seeds 900–919, phase 3 900–909 and phase 4 920–939. The pilot log is seed 1.
- **Main line `test_split_copy`:** X 560–569, L 1560–1569.
- **Every branch tip:** all 12 remote branches were scanned, both now and as of 550a736's time, parsing 330 distinct json/jsonl blobs.
  - At 550a736's time: no record carries a seed in 940–959.
  - Now: the only hit is R's own `records/phase5.jsonl`.
- **History up to 550a736:** a pickaxe search over all refs finds 5 commits. None is a record:
  - R's protocol;
  - the round-5 prompt that assigned "R 940-959";
  - three false positives, two of them in the referee's own files.
- **A side note:** seeds 940–951 were named for R in the Phase 4 amendment (`db22857`, part (b): "920-951"). R never ran that amendment, and its Phase 4 records stop at 939.

## Rows (non-MATCH first)

| # | R file:line | statement | recomputed | status | note |
|---|---|---|---|---|---|
| 35 | rr:218–221 | "From the written specification alone, an independent implementation reproduces: the copy arm's count exactly; the reset arm within the main line's interval; the single delta channel's 0/20" | Counts: COPY 18/20 (main line 18/20), RESET 9/20 (inside 3..10), DELTA1 0/20, all MATCH. But COPY = WIN3_SPLIT_D8_ML, R's Phase 4 recipe (rr:185; ph:15–18). `phase4.py`:6–9 takes "the split as the main line specifies it": the probe `Generator(12345)` and the 32 body key positions, which DIFF.md D-1/D-2 record as read from the main line's code. RESET uses the same trigger. R read `test_window_gate.py` at `404e3a0` in Phase 3 (DIFF.md); it holds WIN3_RESET_D8 (:278) and `reset_op` (:510–511, "W_g's optimizer state zeroed in place; W_g untouched"). | **PARTIAL** | True of the numbers. "From the written specification alone" holds at most for DELTA1. The user's Phase 5 prompt (`0754d61`) itself sets COPY to the Phase 4 recipe, so this is a wording overstatement, not a hidden change. Suggested wording: "from the specification, with the split's probe and key positions as taken from the main line's code in Phase 4". |
| 32 | rr:214 | "Per-stream accuracy is roughly even within a run (for example 0.72–0.79 on seed 942)" | 942: 0.717–0.795. The per-run spread (max − min stream) has median 0.10, range 0.057–0.355. Spread is ≥ 0.25 on 4 of 20 runs: 950 (0.05–0.34), 951 (0.23–0.52), 954 (0.24–0.59), 959 (0.25–0.50) | **PARTIAL** | 942's range matches. "Roughly even" holds for 16 of 20 runs. It is a diagnostic, not a claim. |
| 2 | rr:174; ph:11 | protocol "committed before the first run" | The replay gives a start no later than 22:49:48, 2.0 min after `550a736`. The 64 runs fill 19.86 of the 19.90 h between the protocol and report commits | NOT RECORD-CHECKABLE | Consistent but tight. The records carry no timestamps, so a start a few minutes before 22:47:46 cannot be excluded. |
| 3 | ph:11; UNSTATED.md:93, :136 (U-70) | "fixed before any Phase 5 run"; "every entry written before any Phase 5 run"; no pilot runs | All protocol files are identical from `550a736` to `31c39fc`; only the RESULT block was filled | NOT RECORD-CHECKABLE | The executions U-70 discloses (unit tests, a 2400-update regression of 920, a timing) leave no record. |
| 5 | rr:172–173 | "from its specification and docstring at `ffb0771` only"; "Not read: no code below that docstring, and nothing from Session F's delta-memory modules" | R's branch is an orphan: it has no merge-base with the main line, and its tree never holds a main-line file (`550a736`, `29ff87a`, `31c39fc`). In Phase 3, though, R read the main line at `404e3a0` (DIFF.md), whose tree already held `test_split_copy.py`, `specs/test_split_copy.md`, `explore_delta_mem.py` and `explore_f_delta_fast.py` | NOT RECORD-CHECKABLE | What R opened is not in git (§5 below). |
| 38 | ph:15–18; rr:185 | COPY = "my Phase 4 recipe unchanged" | `replicate.py` changed from `1f8be04` to `550a736`: `Split.apply` was split into `_copy` behind `op="copy"` (the default), and arms were added. In the diff, the copy path's operations and their order are unchanged | NOT RECORD-CHECKABLE | No Phase 5 record reruns a Phase 4 run. U-70's 920 regression left no record. |
| 1 | rr:174; git log | the protocol was committed (`550a736`) before the first record and was not changed after it | See the order table above | MATCH | |
| 4 | ph:13; `550a736` title | seeds 940–959, not used before | Not used anywhere before (§2) | MATCH | 940–951 were assigned to R in `db22857` but never run. |
| 6 | ph:14 | "Phases 2-4 records are not touched" | phase2/3/4 `.jsonl` identical from `1f8be04` to `31c39fc` | MATCH | |
| 7 | rr:174 | the "Phase 4, amended" text states a different rule (≤ 27/32 on fresh seeds) | `db22857` part (b): "x <= 27/32" on 920–951 | MATCH | |
| 8 | rr:175–179 | guesses are U-50 to U-70; the four that matter most | 21 rows U-50..U-70 (labels [A] 12, [B] 4, [S] 3, [C] 1, [S/A] 1). RESET = U-50, raw queries / L2 keys = U-52, decay = U-53, width = U-58 | MATCH | U-50 (and rr:176) call "W_g's Adam state zeroed, W_g untouched" the spec's wording. That exact phrase is the docstring's. The spec has "Adam state zeroed, W_g untouched" (spec:12) and "W_g's Adam state zeroed" (spec:138). The code R read in Phase 3 (`test_window_gate.py`:510–511 at `404e3a0`) zeroes W_g's state only, as R chose. |
| 9 | rr:194; ph:49 | COPY 18/20 [0.699, 0.972] | 18/20 [0.6990, 0.9721] | MATCH | |
| 10 | rr:195; ph:50 | RESET 9/20 [0.258, 0.658] | 9/20 [0.2582, 0.6579] | MATCH | Successes: 940, 941, 942, 943, 946, 948, 949, 950, 956 |
| 11 | rr:196; ph:51 | DELTA1 0/20 [0, 0.161] | 0/20 [0, 0.1611] | MATCH | |
| 12 | rr:194–196; ph:49–51; `31c39fc` title | each arm "consistent" | COPY 18 in 14..19; RESET 9 in 3..10; DELTA1 0 in 0..3 | MATCH | |
| 13 | ph:32–34 | consistent iff 14..19 / 3..10 / 0..3 | Same sets, both from the printed endpoints and from the unrounded Wilson intervals | MATCH | The rule is ph:29–31: x/20 inside the main line's pooled Wilson 95% interval, endpoints included. |
| 14 | rr:195 | "0.45 is inside" | 0.145 ≤ 0.45 ≤ 0.519 | MATCH | |
| 15 | rr:197; ph:48 | ORACLE_ADAM 2/2 (6000, 6000); ORACLE_DELTA 2/2 (1200, 1200) | [6000, 6000]; [1200, 1200] | MATCH | Not in the report: ORACLE_ADAM 940 and 941 have identical evaluations at 1200 (0.64893) and 2400 (0.79346), then differ. The cause is not recorded, and it does not affect the validity reading. |
| 16 | rr:197 | main line "CEIL_C 4/4; CEIL_DELTA_D8 4/4 (all at 1200)" | CEIL_C 4/4 at [4800, 3600, 4800, 3600]; CEIL_DELTA_D8 4/4 at 1200 ×4 | MATCH | "(all at 1200)" is true only if it attaches to CEIL_DELTA_D8, as its position suggests. CEIL_C bound at 4800 and 3600 on each machine. |
| 17 | rr:194; ph:6, :32 | main line pooled WIN3_SPLIT_D8 18/20 [0.699, 0.972] | X 10/10 + L 8/10 = 18/20 [0.6990, 0.9721] | MATCH | |
| 18 | rr:195; ph:6, :33 | main line pooled WIN3_RESET_D8 6/20 [0.145, 0.519] | X 3/10 + L 3/10 = 6/20 [0.1455, 0.5190] | MATCH | |
| 19 | rr:196; ph:7, :34 | main line pooled SINGLE_DELTA_D8 0/20 [0, 0.161] | 0/20 [0, 0.1611] | MATCH | |
| 20 | rr:200; ph:52 | COPY vs RESET 9 vs 0 (p = 0.0020) | 9 vs 0, p = 0.001953 | MATCH | |
| 21 | rr:201; ph:52–53 | COPY vs DELTA1 18 vs 0 (p = 3.8×10⁻⁶) | 18 vs 0, p = 3.81e-6 | MATCH | |
| 22 | rr:200; ph:6 | "The main line had 12 vs 0" | pooled 12 vs 0 (p = 0.000244) | MATCH | |
| 23 | rr:201; ph:7 | "The main line had 18 vs 0" | pooled 18 vs 0 (p = 3.81e-6) | MATCH | |
| 24 | rr:204 | "No seed went the other way" | RESET-only pairs: 0 | MATCH | |
| 25 | rr:205–207 | 9 shared binds; 941 and 943 trigger never fired, arms bit-identical; on the other seven the reset fired | 9; no firing on 941, 943, where every recorded field but `wall_s` is equal; RESET fired on the other 7 | MATCH | |
| 26 | rr:208 | RESET 0.45 against 0.30, still inside | 9/20 vs 6/20 | MATCH | |
| 27 | rr:209 | every failure a merge: 11 RESET failures, COPY 945 and 947 (three share), RESET failed on both | COPY max share {945: 3, 947: 3}. RESET: 11 failures with max share 2 or 3, and it failed on 945 and 947 | MATCH | "Merge" is R's map-based label (U-35, DIFF.md D-14). It is not the main line's η²-based `fail_class`, and R records no η². |
| 28 | rr:210; ph:53 | 37 copies and 46 resets, every target labelled ok | COPY 37 (37 ok), RESET 46 (46 ok). The latest firing was at 31200, so D-5/U-63 is inert here too. | MATCH | |
| 29 | rr:211; ph:53–54 | blocks: 1 of 175 in COPY (958, which bound); 2 of 246 in RESET (neither eligible); they decided no run | COPY 1 of 175, on 958 (eligible, bound routed); RESET 2 of 246, on 952 and 957, neither eligible | MATCH | |
| 30 | rr:213; ph:54–55 | DELTA1 never bound; final 0.09–0.78, median 0.29 | 0/20; 0.0864–0.7778, median 0.2927 | MATCH | |
| 31 | rr:213 | "the main line reports 0.08–0.76" | SINGLE_DELTA_D8 final accuracy, pooled 0.0776–0.7559 (X 0.08–0.76, L 0.13–0.55) | MATCH | That range is X's; L's lies inside it. |
| 33 | rr:215 | the perfect gate on delta channels bound at 1200 on both seeds, "so the failure is not in the delta write" | ORACLE_DELTA [1200, 1200] | MATCH | The count matches. The "so" is R's inference: the oracle is k = 16 with a one-hot gate, not k = 1. |
| 34 | rr:216; ph:55 | 78.3 h; 4 workers × 1 thread; each DELTA1 run about 2 h | sum of `wall_s` 78.35 h; DELTA1 runs 119–133 min | MATCH | The worker count is not in the records, but a 4-worker replay reproduces the order exactly. |
| 36 | rr:223 | the copy's advantage holds in direction (9 vs 0 against 12 vs 0); it is smaller because RESET bound more often; with 20 seeds the two are not distinguishable | COPY 18 = 18; RESET 9 vs 6. Fisher two-sided p: 9/20 vs 12/20 discordant 0.53; RESET 9/20 vs 6/20 0.51 | MATCH | "Not distinguishable" is a reading, and these tests support it. |
| 37 | ph:6–7 | main line C1 18/20 vs 6/20 (12 vs 0); C3 18/20 vs 0/20 (18 vs 0) | as stated | MATCH | |

The row numbers are the script's §7 numbers. The table lists the non-MATCH rows first.

## The reading rule (§3b)

The rule (ph:29–35, U-66) is: "consistent" if x/20 lies in the main line's pooled Wilson 95% interval for that arm, endpoints included.
- It is the rule the user's Phase 5 prompt (`0754d61`) asked for.
- R correctly notes that the "Phase 4, amended" text it was told to take the rule from actually states a different rule (≤ 27/32).
- R's printed endpoints are the main line's printed bands. Recomputed from the records (X `592c574` + L `ffb0771`), they are [0.6990, 0.9721], [0.1455, 0.5190] and [0, 0.1611]. Rounding changes no integer threshold.

How often the rule reads "consistent" (context only, not a row):

| arm | true rate | P("consistent") |
|---|---|---|
| COPY | 0.70 | 0.61 |
| COPY | 0.90 (the main line's 18/20) | 0.88 |
| COPY | 0.95 | 0.64 (20/20 reads "outside", as ph:35 says) |
| RESET | 0.15 | 0.60 |
| RESET | 0.30 | 0.95 |
| RESET | 0.50 | 0.59 |

The rule therefore separates little at n = 20. R reads nothing beyond it (rr:223), which is right.

## "From the spec and docstring alone": what git can show (§5)

**What git shows about R's branch:**
- It is an orphan: its root is `786028e`, and it has no merge-base with the main line.
- At `550a736` (19 files), `29ff87a` and `31c39fc` its tree holds none of the following: `test_split_copy.py`, the spec, any main-line test or results file, or F's `explore_*` modules.
- So the branch never *carried* the main line's code or results.

**What git cannot show:** what R read. R had read access to the main line throughout. DIFF.md (Phase 3, `00990ea`) says R read the main line at `404e3a0` (05:09 UTC, 10 October):
- `README.md`;
- `test_window_gate.py`;
- the D8 path's imports, including `test_stream_recipe` (`outcome`, `routed`).

**What follows for Phase 5:**
- `test_window_gate.py` at `404e3a0` implements WIN3_RESET_D8 (:278, kind "reset") and `reset_op` (:510–511). `test_split_copy`'s Part A "imports test_window_gate's arms … unchanged" (spec:66). R had therefore read the code of both main-line arms it replicates as COPY and RESET, and in Phase 4 it adopted the probe and key positions from that code (D-1, D-2).
- The tree at `404e3a0` already held `test_split_copy.py` (`1f52cd5`, 05:06:12, three minutes earlier), `specs/test_split_copy.md` and both delta modules. DIFF.md does not list them among the files read, and git cannot show whether they were opened.

**For the claim:**
- "No code below that docstring" (rr:173) is a statement about `test_split_copy.py` and is not record-checkable.
- "From the written specification alone, an independent implementation" (rr:218) does not hold for COPY or RESET: their trigger comes from the main line's code, read in Phase 3 and adopted in Phase 4. This is row 35, PARTIAL.
- It can hold for DELTA1, as far as git shows.

## UNSTATED.md at the report commit (§6)

- **Rows:** 65 rows of the form `| U-NN`, with 65 distinct ids in the range 1–70.
  - 44 rows are the Phase 2–4 entries, unchanged since `1f8be04`.
  - **21** rows are Phase 5's U-50..U-70, contiguous.
  - The file is identical at `550a736` and `31c39fc`.
- **Labels of the 21:** [A] 12, [B] 4, [S] 3, [C] 1, [S/A] 1.
- **The report's statement (rr:175):** "every detail I had to guess is in UNSTATED.md, U-50 to U-70" holds as a statement of where the guesses are. Not all 21 are guesses: 3 are [S] (stated) and 1 is [S/A].
- **The four highlighted guesses** are U-50, U-52, U-53 and U-58, as row 8 says.

## In the referee's own files

- **`rev9/check_R_v6.md`:11** says `550a736` "adds `phase5.py` and 21 entries to `UNSTATED.md`". Row 8 (:58) and `sixth_pass.md`:123 likewise list it as "(`phase5.py`, UNSTATED.md)" / "(`phase5.py`, U-50–U-70)". All three omit that the same commit changes `replicate.py` (+60 −15: the delta memory, the reset op, three arms, the phase-5 job list) and `test_replicate.py` (+95). The omission is minor: no number changes, and "no Phase 5 record" stays true.
- **`check_R_v6.md`:101** says that whether Phase 5 was running at 23:14 "is not recorded". That is still literally true. The replay now shows the run had started by 22:49:48, so it was running at the sixth draft's commit (23:14:04) unless it started after 22:49:48 at a cost the records do not show.
- No other referee statement about R's Phase 5 was found. The "65 rows at `550a736`" in `check_R_v6.md`:44/:57, `check_appendixB_v6.md`:22 and `seventh_pass.md`:29 reproduces.

## What could not be checked

- **The start of the run relative to the protocol commit.** The records carry no timestamps. The replay bounds the start at ≤ 22:49:48 (and ≥ 22:43:25 only under a no-overhead assumption).
- **What R read.** This covers "no code below that docstring" and "nothing from Session F's delta-memory modules"; DIFF.md's Phase 3 reading list is R's own statement.
- **U-70's pre-protocol executions** (unit tests, the 920 regression, the timing).
- **That COPY is bit-for-bit the Phase 4 recipe.** The `replicate.py` diff keeps the copy path's operations, but no record reruns a Phase 4 run.
- **The user's chat statements** that R cites ("the reading rule you stated", "the gate group" in the message). The committed prompt `0754d61` matches both.
- **The ORACLE_ADAM 940/941 coincidence** (identical evaluations at 1200 and 2400): its cause is not in the records.
- **Workers and threads.** They are not in the records, though the 4-worker replay reproduces the completion order exactly.

## Script output (`python3 docs/review_2026-10/check_R_phase5.py`)

```
OUTPUT_PLACEHOLDER
```
