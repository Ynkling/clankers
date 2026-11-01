# Check of Session F, report 10 (S90-F), claude/explore-F at `d9783bb1`

*Referee V, 22 October 2026. This check covers `explore_out/F/report_10.md` and the README section "Session F, round 7" (README l.675–696) at `d9783bb1`. They are checked against `s90_ctx_results.json` (96 records), `s90_decision.json`, `s90_ctx_tables.md` and `s90.log` at the stage commits a4d14c24 (phase 0), 79f00415 (phase 1), d1072a46 (phase1b), 4c9612f6 (phase 2) and b502f1c3 (final), and against the code at every commit from `9017b823^` to `d9783bb1` (25 commits). It also checks F's report-9 erratum `a0dc3c34`. The pre-run review it follows up is `screen_review_s90_F.md`, committed at `ecbaf771`, 05:52:28.*

*Every recomputation is in `check_F_report10.py`, which reads git objects only and prints only. One exception: it also reads this clone's reflog, for one push time. It execs F's own functions, extracted by ast:*
- *`decision`, `_arms`, `outcomes`, `decide`, `reading_task` and `reading_screen` from `explore_f_s90_ctx.py` at `73749e8a`, on in-memory stubs;*
- *`wilson`, `band`, `count_str`, `mcnemar_greater` and `paired` from `explore_f_common.py`;*
- *`balanced_map`, `_esym`, `_partitions` and `type_map_cap` from `explore_f_tasks.py` at `9017b823`;*
- *`rlib.transition`, `bound`, `wilson`, `band`, `mcnemar1` and `fisher1`.*

*"§X" refers to a section of that script's output. No torch is installed here, so the generators, the probes and the CHECK contents are not recomputed. Those rows rest on the log's CHECK lines, or on the numpy replica in `screen_checks_s90_F_out.txt`.*

## Verdict

**The numbers hold, and every reading is mechanical.**
- Every count, Wilson interval, band, transition, median, range, McNemar p, accuracy and cap in report 10 and the README section matches the records or the exact combinatorics.
- `decide()` from `73749e8a`, replayed stage by stage on the store at each stage commit, reproduces the committed decision file at all five stages (§C).
- `reading_task` gives "the task is valid" for all six task × memory pairs. `reading_screen` gives "one channel binds without a context identity": SINGLE_DELTA_CTXPOOL binds 10/10.
- The 27-module code SHA recomputes to `809fde1f5168dcb5…` from the blobs at `73749e8a`. It is constant to the tip and equals all 96 records.

**The order is clean.**
- The readings were committed at `73749e8a` (05:13:59). Phase 0 started 22 s later (05:14:21) on a clean tree ("git 73749e8").
- No code or rule file changed between `73749e8a` and the tip. Each has one blob (§A).

**Non-MATCH rows: 9 PARTIAL, 0 DISAGREE. All are wording or labelling; no reading is affected.**
- **Mislabelled statistics, 2 rows:**
  - "among its misses" is the share among the queries that do not ask the last write. Among misses the share is 1.00 / 0.51 / 0.34.
  - "0.36 and 0.22 mean" are the not-last accuracies. The mean final accuracy is 0.38 / 0.23.
- **Imprecise numbers or claims, 3 rows:**
  - -docs3 3065 "climbs to 0.79" ends at 0.81.
  - The type-level gate's separated fraction "equals the cap". On -docs3 it is 0.32 against 0.268: the probe for seed 3060 is 97/256, z = +4.0.
  - The write-order split is "flat wherever a run binds". The bound type-level delta runs favour the last write by 0.02–0.05.
- **Comparison or causal wording, 4 rows:**
  - -ctxpool's 10/10 against lite's 8/10 crosses recipes: Adam throughout against SLOW.
  - "the failure is the missing partition, not the memory".
  - "a delta channel's capacity is two documents".
  - "the partition … is needed at D ≥ 3". F's own type-level delta binds at D = 3 and 4 while separating 0.14–0.32 of sequences.

**The hold.**
- **What happened.** Phase 2 (the single-channel screen) ran past the referee's hold.
  - The review was pushed at 05:52:30. The "phase1b" stage, which queued the 60 SINGLE runs, began at 06:07:39, 15 min later. The first SINGLE run started between 06:07:43 and 06:11:15.
  - F changed nothing. No code or rule blob moved, and every commit until `d9783bb1` is the queue's own.
  - Report 10 does not mention the review, the hold or any of its fixes.
- **Effect on readings: none.**
  - The readings were fixed before any run, and `decide()` reproduces them.
  - Had fixes 1–12 landed, none would have changed a reading.
  - Fix 2(a) touches only the "≤ 3/10" branch.
  - Fix 4 is a printout.
  - Fix 12's assertion holds on every record.
- **What 10/10 does not show.** It does not show that -ctxpool is "no worse" than lite. The lite references ran under one Adam at 1e-3, and S90 ran under SLOW (§3).

**The cap table is exact.** F prints seven entries plus 1 at k = 16 (0.533, 0.268, 0.141, 0.800, 0.664, 0.615, 1). Each equals F's `type_map_cap`, an independent maximum over all partitions counted subset by subset, and a simulation (§J). Not printed: at k = 2 (G's gates), the cap is 0 at D = 3 and 4.

**The erratum (`a0dc3c34`) is correct and complete for report 9.** The records show the fault directly: every S89 -allpool record has `n_last/n` = 142/2048 = 0.069 under the old function. On -pool it is 1032/2048 = 0.504, the true last-write share at S = 2. Two of the referee's own files matched the withdrawn numbers (§6).

## Counts

| | rows | MATCH | PARTIAL | DISAGREE | NOT RECORD-CHECKABLE |
|---|---|---|---|---|---|
| Report 10 | 62 | 50 | 9 | 0 | 3 |
| README section | 12 | 12 | 0 | 0 | 0 |
| **Total** | **74** | **62** | **9** | **0** | **3** |

## 1. Report 10, statement by statement

### Header

| # | statement (report_10.md l.) | source | recomputed | verdict | note |
|---|---|---|---|---|---|
| 1 | l.4–5: module `explore_f_tasks.py` at 9017b82 (`9017b823e69b…`); append-only, CHECKed | git; log | full hash `9017b823e69bde2ddbe7502c61fefc8fe530be2b`; blob `cfe3fe276f7c…`, one blob 73749e8a..tip (§A); log CHECK ok in all 5 segments | MATCH | The CHECK label still says "48457 bytes", which is characters (review fix 9). The report does not repeat it. |
| 2 | l.4: "the commit G vendors" | remotes | G's tip still holds blob `e24d5e1081` (b0c6332's); no other remote branch holds `cfe3fe27` (§A) | NOT RECORD-CHECKABLE | A statement of intent. |
| 3 | l.6: screen and readings committed at 73749e8 before any run | git, log | 73749e8a 05:13:59; phase 0 began 05:14:21, "git 73749e8" (clean) | MATCH | Phases 1, 1b, 2 and final started "+dirty", because the queue writes tracked logs. The child is covered by the SHA. The parent's `decide()` is not, but it reproduces every decision file (§C). |
| 4 | l.7: Xeon @ 2.10GHz every run; torch 2.14.0+cu130; 1 thread; SHA 809fde1f5168 | records; log | one CPU, one torch, one SHA over 96 records; SHA recomputed from 27 blobs at 73749e8a (§A, §B) | MATCH | |
| 5 | l.8: 96/96 ok, none retried, none diverged | records | 96 ok; attempts {1}; no NaN in any curve | MATCH | 5 phase-0 records carry the harness's `collapsed` flag (final accuracy < 0.15): NOMEM_DOCS3 ×2 (0.123), NOMEM_DOCS4 ×2 (0.110) and NOMEM_SELF_DOCS4 3061 (0.121). That is no-memory accuracy near chance, not divergence. |
| 6 | l.9: seeds 3060–3071, free on all 13 branches | records; 13 remote tips | seeds {3060…3071}. The only hits on any tip are F's own `s90.log` and `s90_ctx_results.json` (§M). The history scan is in screen review §2.2 | MATCH | |

### Counts first

| # | statement | source | recomputed | verdict | note |
|---|---|---|---|---|---|
| 7 | l.19: -ctxpool delta **10/10** [0.72, 1.00] RELIABLE; median 9600 (7200–15600) | records | 10/10; F's `count_str` and rlib both give [0.72, 1.00] RELIABLE; transitions 7200, 8400×3, 9600×3, 10800, 14400, 15600; median 9600; the record transition equals `rlib.transition` on every run (§E) | MATCH | |
| 8 | l.19: -ctxpool Hebbian + conv 0/10 NEVER; final 0.48–0.51 | records | 0.4834–0.5137 | MATCH | |
| 9 | l.19: paired 10 / 0, one-sided McNemar p = 0.00098 | F's `paired`; rlib | b 10, c 0; p = 1/1024 = 0.000977 | MATCH | |
| 10 | l.20: -docs3 delta 0/10; final 0.30–0.43, one at 0.81 | records | 0.3037–0.4331; 3065 0.8071 | MATCH | |
| 11 | l.20: -docs3 Hebbian + conv 0/10; 0.32–0.35 | records | 0.3159–0.3511 | MATCH | |
| 12 | l.21: -docs4 delta 0/10; 0.20–0.25 | records | 0.2041–0.2466 | MATCH | |
| 13 | l.21: -docs4 Hebbian + conv 0/10; 0.24–0.26 | records | 0.2354–0.2554 | MATCH | |
| 14 | l.23–24: SCREEN READING "one channel binds without a context identity" | `reading_screen` exec'd | that string; `outcomes` gives all 10 True | MATCH | Mechanical. |
| 15 | l.30–32: pilots HC (2400, 4800), (3600, 3600), (3600, 3600); delta (1200, 2400) ×3 | records | as stated; record = rlib on all 12 (§D) | MATCH | |
| 16 | l.30–32: "the task is valid" ×6 | `decide`, `reading_task` | budget 24000 for all 6, so "the task is valid" ×6; the decision file is reproduced at every stage (§C) | MATCH | |
| 17 | l.34: no 48000 fallback; each pair 2/2 [0.34, 1.00] RELIABLE | records; ARMS | no `_48K` record or arm; `count_str(2, 2)` | MATCH | |
| 18 | l.39–41: NOMEM 0.17/0.17, 0.12/0.12, 0.11/0.11; NOMEM_SELF 0.37/0.37, 0.19/0.20, 0.16/0.12 | records | 0.1685/0.1660, 0.1230/0.1226, 0.1104/0.1104; 0.3657/0.3652, 0.1904/0.2002, 0.1646/0.1206 (§I) | MATCH | |
| 19 | l.39–41: positional bounds 0.19 / 0.16 / 0.14 (reach 4); 0.42 / 0.31 / 0.25 (reach 10) | tables.md, log CHECK | 0.187 / 0.156 / 0.141; 0.423 / 0.310 / 0.250. The referee's replica of F's form: 0.180 / 0.152 / 0.134 and 0.415 / 0.308 / 0.250 | MATCH | F's approximate form (review fix 7). The generous form at reach 10 is 0.503 / 0.365 / 0.291. Every accuracy is below either form. |
| 20 | l.43: the stop rule (0.6) could not have fired | as row 19 | the largest bound of either form is 0.503 | MATCH | F took S89 fix 5. |

### The tasks and the cap

| # | statement | source | recomputed | verdict | note |
|---|---|---|---|---|---|
| 21 | l.47–49: lite with the contexts renamed; the index by first appearance (CHECKed) | log CHECK; module | `check_ctx_task` ok on every task in all 5 segments | MATCH | From the log only (torch). The replica agrees (screen review §1.2). |
| 22 | l.52: KEY 0.101–0.103, VAL 0.022–0.024, CTX 0.013–0.015 | tables.md | KEY 0.1010–0.1027, VAL 0.0220–0.0244, CTX 0.0126–0.0154 | MATCH | |
| 23 | l.53–54: largest-ratio cut 4 KEY; half-mass 4 KEY + 4 VAL; no cut selects a context | tables.md | as stated (mass 0.502 ×3) | MATCH | Which 4 VAL types is sampling noise (review §1.2). |
| 24 | l.61–63: k = D: 64/120 = 0.533, 150/560 = 0.268, 256/1820 = 0.141 | `type_map_cap`; independent | identical (§J; section 4 below) | MATCH | |
| 25 | l.61–63: k = 2D: 0.800 / 0.664 / 0.615 | as 24 | 96/120, 372/560, 1120/1820 | MATCH | |
| 26 | l.61–63: k = 16: 1 | as 24 | 120/120, 560/560, 1820/1820 | MATCH | |
| 27 | l.56–57: "exact over every split of the 16 types; the balanced split is best" | as 24 | the maximum over all partitions is reached by the balanced sizes at every (D, k) | MATCH | |
| 28 | l.65: the cap is a property of k = D; it vanishes with one channel per type | as 24 | k = 16 gives 1 | MATCH | |

### What the runs show

| # | statement | source | recomputed | verdict | note |
|---|---|---|---|---|---|
| 29 | l.70–71: one delta channel binds the two-document task on every seed, without routing or a fixed context identity | records | 10/10; k = 1, no gate (records `k`, `memory`, `conv`) | MATCH | Under SLOW, on 3062–3071. |
| 30 | l.72–73: "On lite's header layout one delta channel bound 8/10 (Revision 10 §11). -ctxpool's 10/10 is no worse once the contexts lose their identity." | delta_controls_results.json; horizon_results.json | S53 SINGLE_DELTA_H 8/10 (seeds 300–309), sched "Adam 0.001 throughout". S88b FIXED_1X 9/10 (2420–2429), "Adam 1e-3 throughout". S90 SINGLE: "SLOW (rest 1e-4 to 2400)" (§O, §B) | **PARTIAL** | Both numbers match. The comparison crosses schedule, seeds and session; no SLOW lite arm exists (review fix 2). One-sided Fisher 10/10 against 8/10 gives p = 0.24 even on equal footing. So "no worse once the contexts lose their identity" attributes to the contexts what the design cannot isolate. |
| 31 | l.74–75: -docs3 3065 "climbs to 0.79 and is still rising at 24000" | curve | 0.7944 at 22800, **0.8071 at 24000** (the last evaluation); rising | **PARTIAL** | The final value is 0.81, as l.20 says. 0.79 is the penultimate evaluation. |
| 32 | l.77–78: Hebbian + conv ends at 0.48–0.51, 0.32–0.35, 0.24–0.26 | records | as rows 8, 11, 13 | MATCH | |
| 33 | l.79–80: "Among its misses, the last-written value's share is also about 1/D (0.52, 0.34, 0.26)" | `write_order` | mean `lastval_notlast` = 0.520 / 0.342 / 0.255 ✓. That field is P(prediction = the last-written value \| the query does **not** ask the last write) over all such queries, correct ones included (tasks l.1070). Among misses the share is 0.520/(1 − 0.480) = **1.00**, 0.342/0.671 = **0.51** and 0.255/0.754 = **0.34**, i.e. about 1/(D − 1) | **PARTIAL** | The numbers are right; the label is wrong. The conclusion (no preference for the last write; values chosen about uniformly among the key's D) is supported by the correct reading: P(last) ≈ P(correct) ≈ 1/D among not-last queries. |
| 34 | l.81–82: failing delta at D = 3, 4 "(0.36 and 0.22 mean; last-written share 0.34 and 0.26)" | records | mean final accuracy **0.382 / 0.230** (0.335 on -docs3 without 3065). **0.361 / 0.223 are the mean not-last accuracies** (the tables' "write order" column). Last-written share 0.336 / 0.256 ✓ | **PARTIAL** | Mislabelled. "The same level" still holds (≈ 1/D). |
| 35 | l.84: type-level Hebbian + conv unbound 0.77 / 0.76, 0.73 / 0.72, 0.81 / 0.73 | records | 0.7686 / 0.7573, 0.7329 / 0.7246, 0.8130 / 0.7256 | MATCH | |
| 36 | l.85: type-level delta binds all three (8400–10800), final 0.96–1.00 | records | 9600 / 10800, 8400 / 8400, 9600 / 10800; 0.9600–0.9966 | MATCH | |
| 37 | l.86–87: "Within each sequence its statistics equal the cap: separated 0.53 / 0.32 / 0.14 against 0.533 / 0.268 / 0.141 (the probe's 256 sequences)" | records `end` | Means 0.531 / **0.318** / 0.137. Per probe: 140, 132 / **97**, 66 / 33, 37 of 256. The fraction is constant over evaluations (the map is fixed). -docs3 seed 3060's probe is 97/256 = 0.379 against 0.268: z = +4.0, P(X ≥ 97) = 6.9·10⁻⁵ under a uniform context draw (§F) | **PARTIAL** | At D = 2 and 4 it is within noise. At D = 3 it is not equal. What the CHECK asserts, and what holds, is that `seq_separated_frac` equals the balanced map's rate on that probe (log: "separates 0.3789 of the probe's 256 sequences: seq_separated_frac 0.3789 equals it"). Why one probe sits 4 SD above the cap cannot be checked without torch (section 7). |
| 38 | l.88–91: post-hoc account: Hebbian 0.767 / 0.732 / 0.728; delta 1.000 / 0.952 / 0.946; observed as listed | exact enumeration (§G) | E[1/m] over all D-subsets and query documents under the balanced map: 0.7667 / 0.7321 / 0.7280. Delta, with all queries answered for m ≤ 2 and 1/m for m ≥ 3: 1.0000 / 0.9524 / 0.9456. Observed: as rows 35, 36 | MATCH | The delta rule for m ≥ 3 is not stated (l.89). The numbers require 1/m (2/m would give 0.976 / 0.973). The m distribution is 0.533 / 0.467 at D = 2; 0.488 / 0.440 / 0.071 at D = 3; 0.484 / 0.435 / 0.079 / 0.002 at D = 4. Hebbian -docs4 3060 (0.813) is 0.085 above the account. |
| 39 | l.93–94: delta above the prediction at D = 3, 4, "so a delta channel holds a third document on some queries" | as 38 | 0.961, 0.968 > 0.952; 0.966, 0.960 > 0.946 | MATCH | Labelled post hoc. The eval batch's own m distribution is not recorded, so the 0.01–0.02 excess is not separable from it. |
| 40 | l.95–96: every perfect-gate pilot ends with margin 1.00 and separated 1.00 | records `end` | margin, `seq_separated_frac` and `sep_key_inv` 1.0000 on all 12 (§D) | MATCH | |
| 41 | l.97–98: "The corrected write-order split … is flat wherever a run binds. Example: one delta channel on -ctxpool, 0.99 / 0.99" | records | Single delta -ctxpool 0.9925 / 0.9939 ✓; the pilots 1.000 / 1.000. But the **bound type-level delta runs** have last / not last of 0.995 / 0.973, 0.986 / 0.950, 0.995 / 0.955, 1.000 / 0.955 and 0.998 / 0.947: a last-write advantage of 0.02–0.05 (§H) | **PARTIAL** | Not flat on 5 of 6 bound type-level runs. On a channel shared by two documents, delta favours the last write. That bears on row 59's "capacity two documents". |

### Design decisions

| # | statement | source | recomputed | verdict | note |
|---|---|---|---|---|---|
| 42 | l.102–103: lite's keys kept; one draw renames the contexts | tasks l.909–915 | `make_batch` draws lite's batch, then one `torch.rand(B, 16).argsort` | MATCH | |
| 43 | l.104: index by first appearance; at S = 2 -allpool's convention (CHECKed) | log | ok | MATCH | |
| 44 | l.105–111: within-sequence statistics; η² not used at S > 2; `key_sep` / `key_part` NaN | tasks l.868–877, l.1015–1030; records | as stated; the type-level records have `key_sep` NaN | MATCH | The pooled `sep`, `key_cos`, `val_cos` and `discovered` are still stored (review fix 6). The report reads none of them. |
| 45 | l.112–113: the fallback is per task and memory; not needed | s90 l.45–47, l.136–138 | as stated; no `_48K` arm | MATCH | |
| 46 | l.114–115: the screen is read against the pilot; not rerun on 3062–3071 | records | no O_ record on 3062–3071 | MATCH | |
| 47 | l.116–118: the screen ran in the "phase1b" stage; budgets from phase 1's decision | `_arms` on the decision at 79f00415 | it yields the 6 SINGLE arms at 24000, so they were queued at the phase1b import; phase 2 and final queued 0 (§C) | MATCH | Exactly what the review's fix 1 was about. The report does not mention the hold. |

### CHECKs

| # | statement | source | recomputed | verdict | note |
|---|---|---|---|---|---|
| 48 | l.122: explore_f_delta_checks 26/26 | log | "26 ok" in each of the 5 segments | MATCH | |
| 49 | l.123: "S90-F's 107 CHECKs" | log | 107 ok and 0 FAIL in each of the 5 segments (§K) | MATCH | |
| 50 | l.134–137: three ties bit for bit through 2400 with the new tasks' code disabled | log; screen review §2.3 | printed curves equal; the references equal the records at 52e2ea7 / 3e3b1bf (review) | MATCH | The new-task path (child l.92–107, `STATS_CTX`) is tied by none (fix 3). l.134–135 says so. |
| 51 | l.138–142: models CHECKs (one-hot at k = D; k = 1; reaches; lrs) | log; records | ok ×5; the records' `group_names` [[], rest], with the conv in rest on the HC arms | MATCH | |
| 52 | l.143–144: no SLOW single-channel record exists to tie to | review §2.3 | none found | MATCH | |
| 53 | l.145–146: disclosure of hand-run CHECKs before the commit | — | — | NOT RECORD-CHECKABLE | |

### Erratum

| # | statement | source | recomputed | verdict | note |
|---|---|---|---|---|---|
| 54 | l.150–151: `write_order_acc` compared the query's CTX token with the stream index; invalid on -allpool; 485 of 512 differ | tasks l.158–190; S89 records; replica | Old code: `qs = inp[:, qpos-1]` (the CTX token), `is_last = last_stream == qs`. Every S89 -allpool record has `n_last/n` = **142/2048 = 0.069**; every -pool record 1032/2048 = 0.504. The corrected mask on -ctxpool gives 0.504. Replica: P(type ≠ index) = 15/16, so 480 ± 5.5 of 512; 485 is consistent (§L) | MATCH | |
| 55 | l.151–152: `write_order_acc_idx` is the corrected form; every S90 run uses it | records | `n_last/n` on the screen 0.504 / 0.317 / 0.254 ≈ 1/D, as the last-write share should be | MATCH | |
| 56 | l.152–153: G's and H's -allpool splits are affected | — | not examined here | NOT RECORD-CHECKABLE | G's tip holds blob `e24d5e10`, with the old function. |

### Runtime

| # | statement | source | recomputed | verdict | note |
|---|---|---|---|---|---|
| 57 | l.157: 11.48 h on 4 workers, 05:14–08:27 | records; log | Σ `secs_wall` = 11.477 h; log 05:14:21 – "done" 08:27:45 | MATCH | |
| 58 | l.161–163: phase 0 05:14–05:34 (12); phase 1 05:34–06:07 (24); screen 06:07–08:22 (60) | log; stores | stage headers 05:14:21 / 05:34:08 / 06:07:39 / 08:22:04; +12 / +24 / +60 records at a4d14c24 / 79f00415 / d1072a46 | MATCH | |

### Anything unexpected

| # | statement | source | recomputed | verdict | note |
|---|---|---|---|---|---|
| 59 | l.167–169 (and heading l.69): the type-level delta "separates only 14% of -docs4 sequences and still reaches 0.96. A delta channel's capacity is two documents, not one" | records | 0.137 separated; 0.966 / 0.960 ✓ | **PARTIAL** | The numbers match. "Capacity two documents" is the post-hoc account (row 38) stated as a finding, on 2 seeds per cell. The same runs show a 0.04–0.05 last-write advantage (row 41), so two documents on one delta channel are not held equally. |
| 60 | l.170–171: Hebbian + conv fails "exactly at 1/D … so the failure is the missing partition, not the memory" | records | 0.4834–0.5137, 0.3159–0.3511, 0.2354–0.2554 (≈ 1/D, not exactly) | **PARTIAL** | A causal reading of an unpaired comparison: the pilots are 2 seeds on 3060–3061 at k = D; the screen is 10 seeds on 3062–3071 at k = 1. |
| 61 | l.173: -ctxpool valid at 24000; -allpool needs 48000 on Hebbian + conv | records; report 9 addendum `3e3b1bfe` | as stated | MATCH | |
| 62 | l.174–175: "on delta the partition is not needed at D = 2. It is needed at D ≥ 3 (0/10 at both)." | records | single delta 0/10 at D = 3, 4 ✓; type-level delta 2/2 at D = 3, 4 while separating 0.32 / 0.14 (rows 36, 37) | **PARTIAL** | One channel does not suffice at D ≥ 3, but a per-sequence partition is not needed either: a fixed map at k = D binds. F says so itself at l.167–169. |

## 2. README section ("Session F, round 7", README l.675–696 at d9783bb1)

| # | statement | recomputed | verdict | note |
|---|---|---|---|---|
| M1 | Module `explore_f_tasks.py` at 9017b82 | as row 1 | MATCH | |
| M2 | No-memory: conv into the memory 0.37 / 0.19–0.20 / 0.12–0.16, below 0.42 / 0.31 / 0.25; none stopped | as rows 18–20 | MATCH | |
| M3 | Pilots 2/2 on every task and memory (HC 2400–4800; delta 1200–2400); "the task is valid" ×6 | as rows 15–16 | MATCH | |
| M4 | Type-level: delta 2/2 every task (8400–10800); HC 0/2 (0.72–0.81); descriptive | as rows 35–36 | MATCH | |
| M5 | Screen: -ctxpool delta 10/10 (median 9600), HC 0/10 (0.48–0.51; 10 vs 0, p = 0.001); -docs 0/10 ×4; "one channel binds without a context identity" | as rows 7–14 | MATCH | p 0.00098 rounds to 0.001. |
| M6 | -ctxpool separates the contexts' identity from the keys' frequency | design (row 42) | MATCH | |
| M7 | Valid on both memories at 24000 | row 16 | MATCH | |
| M8 | One delta channel binds without any routing; the missing identity is no obstacle at two documents | row 29 | MATCH | Supported by 10/10 under SLOW. Unlike row 30, it makes no cross-recipe comparison. |
| M9 | With 3 or 4 documents one channel fails on both memories; failures near 1/D; values retrieved but not told apart | rows 10–13, 33 | MATCH | One -docs3 delta seed is at 0.81. |
| M10 | Post hoc: 1/m if Hebbian, up to two documents if delta | row 38 | MATCH | Labelled post hoc here. |
| M11 | The cap at k = D is 0.533 / 0.268 / 0.141 | row 24 | MATCH | |
| M12 | ERRATUM: `write_order_acc` invalid on -allpool; report 9's -allpool splits withdrawn | row 54 | MATCH | |

## 3. The hold: phase 2 ran past the review

**Timeline** (UTC, 22 October; §N). Commit times are from git. The push time of `ecbaf771` is from this clone's reflog ("update by push"). F's push times are not in git.

| time | event |
|---|---|
| 05:13:59 | `73749e8a`: screen, readings, `run_s90.sh` (5 stages, no stop) |
| 05:14:21 | phase 0 starts (git 73749e8, clean) |
| 05:34:06 / 05:34:08 | phase 0 decision (`a4d14c24`); the phase 1 stage starts. The log projects "worst case 4.07 h of runs, makespan 1.02 h" |
| 05:52:28 / **05:52:30** | `ecbaf771` screen_review_s90_F.md committed / **pushed**. Fix 1: hold phase 2 behind `explore_out/F/s90_phase2_go` |
| 06:07:37 | `79f00415`: phase 1 decision; budget 24000 for all 6 (task, memory) pairs, so `_arms()` now yields the 6 SINGLE arms |
| **06:07:39** | the **phase1b** stage starts (git 79f0041+dirty). CHECKs begin 06:07:43; "60 runs queued"; the first run is SINGLE_DELTA_DOCS4 3062. Its CHECK file is committed at 06:11:15 (`906216b8`), so the first SINGLE run started between 06:07:43 and 06:11:15 |
| 06:30:31 | first SINGLE records committed (`4f3ea642`, 4 runs) |
| 08:22:01 | all 60 SINGLE records (`d1072a46`) |
| 08:30:00 | `d9783bb1` report 10 |

- **Margin.** Phase 2 started 15 min 9 s after the review was pushed, and 33 min after phase 1 began. That is not the 1.02 h the review used, because 1.02 h was the worst-case projection (section 6).
- **F changed nothing.**
  - The code and rule files (`explore_f_s90_ctx.py`, `_child.py`, `explore_f_reports7.py`, `explore_f_tasks.py`, `run_s90.sh`, the helpers) have one blob each from `73749e8a` to the tip (§A).
  - Every commit from `a0dc3c34` (05:14:45) to `b502f1c3` is the queue's own ("explore_out/F: S90 …", data and logs only).
  - No `s90_phase2_go` file exists.
  - Report 10 and the README never mention the review, the hold, or fixes 1–12. The one "V's" is S89's point (b), l.112.
  - Whether F's session saw the review before 06:07 is not in git.

**Did running past the hold change a reading? No.**
- The readings are fixed at `73749e8a` and applied mechanically. `decide()`, `reading_task` and `reading_screen` reproduce every stage's decision and both readings (§C).
- The hold was for review, not for a rule change. Fix by fix, had it landed before 06:07:

| fix | would it have changed a reading? |
|---|---|
| 1 (go-file) | No. It changes when the runs start. |
| 2 (lite control under SLOW) | **No.** 2(a) rewrites only the "≤ 3/10" branch; the "≥ 7/10" branch fired. 2(b) is a docstring sentence. Both lite references ran under one Adam: S53 SINGLE_DELTA_H 8/10 "Adam 0.001 throughout", S88b FIXED_1X 9/10 "Adam 1e-3 throughout". All 60 S90 SINGLE records are "SLOW (rest 1e-4 to 2400)" (§O, §B). |
| 3 (tie the new-task path) | No reading. It would only have detected a code fault on that path, which nothing here can test (section 7). |
| 4 (operating characteristics) | No; a printout. P(≥ 7/10) = 0.987 / 0.879 / 0.650 / 0.172 at p = 0.9 / 0.8 / 0.7 / 0.5. P(10/10) = 0.35 / 0.11 / 0.028 / 0.001 (§P). |
| 5, 6, 8 (statistics) | No. S90 reads BOUND only. |
| 7, 9, 11 | Wording. |
| 10 (STOPPED) | Moot: nothing stopped. |
| 12 (assert the budget) | No. `rec["iters"] == ARMS[arm]["iters"]` holds on all 96 records (§C). |

**What the 10/10 does and does not show, given the schedule confound.**
- **What it shows:**
  - Under SLOW, at 24000, on seeds 3062–3071, one ungated delta channel binds -ctxpool on every seed: Wilson [0.72, 1.00], and 10/10 has probability ≤ 0.028 if the per-seed rate were ≤ 0.7.
  - The contexts' missing fixed identity did not stop it.
  - The confound cannot explain a 10/10 away. SLOW could only have lowered binding (E's S69), and the reading fired anyway.
- **What it does not show:**
  - (i) That -ctxpool is "no worse" than lite (l.72–73). Lite was never run under SLOW. The references differ in schedule, seeds and session, and 10/10 against 8/10 would not separate them anyway (Fisher p = 0.24).
  - (ii) Anything about the schedule itself. There is no Adam arm on -ctxpool.
  - (iii) Routing. The arm has no gate.
  - (iv) Anything for Hebbian + conv (0/10) or for D ≥ 3 (0/10 on both) beyond those counts.
- **The confound mattered only for the outcome that did not occur.** Under "≤ 3/10" it would have left the reading unattributable. The procedural point stands: the learned arms ran without the review the prompt asks for ("reviewed by V before the learned arms", prompt l.1276 at `5e51a294`). The user never decided fix 2 before the runs.

## 4. The cap table

**Method.** The cap is the fraction of the C(16, D) context subsets that a fixed map of the 16 CTX types to k channels sends to D distinct channels. Three ways (§J):
1. F's `type_map_cap`, exec'd from `9017b823`.
2. An independent enumeration: every partition of 16 into at most k parts (k ≤ 8), with the subsets **counted directly**, not by e_D.
3. The balanced map counted directly, and simulated on 50,000 uniform draws (numpy).

| D | C(16, D) | k = 2 | k = D (printed) | k = 2D (printed) | k = 16 (printed) | best split at k = D / 2D |
|---|---|---|---|---|---|---|
| 2 | 120 | 64 (0.533) | 64/120 = **0.5333** ✓ (sim 0.535) | 96/120 = **0.8000** ✓ (sim 0.798) | 120 = **1** ✓ | (8, 8) / (4, 4, 4, 4) |
| 3 | 560 | **0** | 150/560 = **0.2679** ✓ (sim 0.265) | 372/560 = **0.6643** ✓ (sim 0.664) | 560 = **1** ✓ | (6, 5, 5) / (3, 3, 3, 3, 2, 2) |
| 4 | 1820 | **0** | 256/1820 = **0.1407** ✓ (sim 0.142) | 1120/1820 = **0.6154** ✓ (sim 0.614) | 1820 = **1** ✓ | (4, 4, 4, 4) / (2 × 8) |

- **No entry is wrong.** All seven printed values (0.533, 0.268, 0.141, 0.800, 0.664, 0.615, 1), and the fractions 64/120, 150/560 and 256/1820 in report l.61–63, tables.md and README l.694, agree with all three methods.
- The balanced map (`t·k // 16`) attains the maximum at every (D, k). At D = 3, k = 6 its sizes are [3, 3, 2, 3, 3, 2], the same multiset as the best split.
- **Not printed anywhere in report 10:** at k = 2 (G's two-channel gates) the cap is 0 at D = 3 and 4.
- **Empirical rates** printed by F's CHECK on 4096 sequences: 0.5269 / 0.2742 / 0.1360. All are within 1 SD of the cap.
- **The probes (row 37):** 140, 132 / 97, 66 / 33, 37 of 256. Only -docs3 3060 is out of line (z = +4.0).

## 5. Order and the code SHA

**Order.** The readings commit `73749e8a` (05:13:59) precedes the first run (05:14:21) and the first record commit, `a4d14c24` (05:34:06; phase 0 records). The module `9017b823` (05:08:53) precedes both.

**No change after records.**
- One blob each from `73749e8a` to the tip:
  - `explore_f_s90_ctx.py` `cadbaba54e`;
  - `explore_f_s90_ctx_child.py` `bfb78389e8`;
  - `explore_f_reports7.py` `8b3c65cf5d`;
  - `explore_f_tasks.py` `cfe3fe276f`;
  - `run_s90.sh` `8fe3219914`;
  - `explore_f_common.py`, `explore_common16.py`, `explore_f_reports{,3,4}.py` and `explore_f_s89_tasks_child.py`.
- The commits after `73749e8a` change only `explore_out/F/` data and logs, `report_9.md` (the erratum, `a0dc3c34`, 05:14:45, before any record), `report_10.md` and the README.

**The code SHA** `809fde1f5168dcb53a4737e207af9cb8c7f4ed7a`:
- recomputed from the 27 blobs the log lists, at every commit from `73749e8a` to `d9783bb1`: constant;
- equal to all 96 records' `code_sha`;
- at `9017b823` it is undefined, because the child file did not yet exist.

**Stage replay** (§C). At each stage commit the replay shows two things:
- The arms with new records are a subset of `_arms()` at the stage's import.
- `decide()` equals the committed decision file.

| stage | commit | records | new arms |
|---|---|---|---|
| phase0 | a4d14c24 | 12 | NOMEM ×6 |
| phase1 | 79f00415 | 36 | O_ ×6, TL_ ×6 |
| phase1b | d1072a46 | 96 | SINGLE ×6 |
| phase2 | 4c9612f6 | 96 | none |
| final | b502f1c3 | 96 | none |

**The erratum `a0dc3c34`: correct and complete.**
- **The fault.** The old function's last-write mask is wrong on -allpool. The records show it without any model: the mask depends on the data only, and it marks 142 of 2048 queries (6.9%) as last-write on every -allpool record, where the true share at S = 2 is about 0.50 (-pool 1032/2048; -ctxpool corrected 0.504).
- **Coverage.** The erratum's list matches every -allpool write-order number in report 9:
  - l.204 "0.86 / 0.87";
  - l.312 "0.95 / 0.97";
  - `s89_post_tables.md`'s six -allpool rows;
  - the -allpool part of l.66 "differs by at most 0.06".

  `s89_tasks_tables.md` and the README's round-6 section print no write-order split.
- **What it leaves standing.** "On -pool it stands" is right. The two functions agree on -pool (CHECK). The -pool figures in check_F_report9 row 30 (0.060 at 24000; **0.082** for POOL_48K 2801) stand. The overall accuracies and transitions are unaffected, as the erratum says: `acc` does not use the mask.

## 6. Errors in the referee's own files

1. **`screen_review_s90_F.md`, verdict item 1 and fix 1: "starts at about 06:37 UTC".** The review took the log's *worst-case* makespan (1.02 h) as the expected one, and dropped "worst case" from the quotation. Phase 1 took 33 min, and the stage began at 06:07:39: 30 min earlier than stated, and 15 min after the review was pushed. The urgency was right; the stated deadline was too late.
2. **`screen_review_s89_F_v2.md` row 10 ("r9 l.204: last-write / not 0.86 / 0.87 … MATCH") and `check_F_report9.md` row 30 (and §E3 of its script).** These matched -allpool write-order numbers (0.86 / 0.87; ALLPOOL_48K 2801 "0.064") that came from the invalid function. The records' `n_last/n` = 0.069 on every -allpool record showed it, and the referee did not catch it. Corrections:
   - Row 10 of the v2 review should read "MATCH to the record; the field is invalid on -allpool (F erratum a0dc3c34)".
   - check_F_report9 row 30's -allpool figure is void. Its PARTIAL verdict stands on the -pool figure (0.082).
3. **`screen_review_s90_F.md` §2.6** is confirmed by the records: S53 SINGLE_DELTA_H 8/10 and S88b FIXED_1X 9/10, both under Adam throughout. **§1.3** is confirmed exactly (section 4). No other error found.

## 7. What could not be checked

- **Needs torch:**
  - the frequency tables (only the referee's numpy replica, which agrees);
  - the exact positional bounds (the replica agrees to about 0.01);
  - every CHECK's content (rows 21, 43, 48–51 rest on the log);
  - the probes themselves.

  In particular, **-docs3 seed 3060's probe (97/256 separated against the cap's 68.6 expected, z = +4.0)** cannot be explained here. A uniform `argsort` draw gives P = 6.9·10⁻⁵. The CHECK confirms only that `seq_separated_frac` equals the balanced map's rate on that probe.
- **The new-task code path** (child l.92–107, `TypeGateK`, `STATS_CTX`): no tie to any record exists (review fix 3). Nothing here can test it.
- **F's push times.** Whether F's session saw the review before 06:07 is not in git.
- **G's and H's -allpool write-order splits** (row 56), and whether G vendors `cfe3fe27` (row 2).
- **The hand-run CHECKs before `73749e8a`** (row 53).
