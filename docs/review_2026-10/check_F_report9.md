# Check of Session F, report 9 (S89-F), claude/explore-F at `52e2ea7`

*Referee V, 20 October 2026. This check covers `explore_out/F/report_9.md` and the README section "Session F, round 6" at `52e2ea7`, checked against `s89_tasks_tables.md`, `s89_decision.json`, `s89_tasks_results.json` (28 records), `s89_diag_results.json` (6 records), `s89.log`, `s89diag.log`, and the code at every commit from `8ba99bb` to `52e2ea7` (26 commits). The pre-run review it follows up is `screen_review_s89_F.md`, committed at `3ae815b`, 18:26:30. Every recomputation is in `check_F_report9.py`, which reads git objects only and prints only. It also execs F's own functions, extracted by ast:*
- *`decide`, `_arms`, `_bound` and `decision` from `explore_f_s89_tasks.py` at `8ba99bb`, on in-memory stubs;*
- *the reading expression from `explore_f_reports6.py` l.73–74;*
- *`wilson`, `band` and `count_str` from `explore_f_common.py`;*
- *`rlib.transition`, `rlib.bound`, `rlib.wilson` and `rlib.band`.*

*"§X" refers to a section of that script's output. No torch is installed here, so nothing that needs the generators or a model is recomputed. Those rows are checked against the log's CHECK lines, or marked as not checkable.*

## Verdict

**The numbers hold.**
- Every count, rate, transition, accuracy, Wilson interval and band in report 9 and the README section matches the records.
- Both "untested" readings follow mechanically from the rules fixed at `8ba99bb`. `decide()` from `8ba99bb`, run stage by stage on the store at each stage's commit, reproduces the committed decision file at all four stages. The reading expression then maps `UNTESTED` to "untested".
- The code SHAs recompute from git objects: `ea9728c636c6…` over 26 modules at `8ba99bb`, and `d4af41b711b3…` over 27 at `4639d16`.

**The order is clean.**
- No code or rule file changed anywhere in `8ba99bb..52e2ea7`: each has a single blob.
- The post-hoc diagnostic's code was committed after the final decision and all 28 records it diagnoses, and before its own runs.

**Non-MATCH rows: 7 PARTIAL, 0 DISAGREE.**
- 4 are small imprecisions:
  - the write-order bound;
  - the 48K plateau range;
  - "13771 bytes", which is characters;
  - the diagnostic's run time.
- 3 are one wording issue: the report and the README read the post-hoc diagnostic as establishing a cause ("The recipe is not the cause", "the key pool is the cause"). The diagnostic's own docstring says "Descriptive … No reading", and it has 2 seeds per cell.

**The referee's pre-run fixes: F took none of them.**
- No code changed, and the report and README mention none of them. Neither the ALLPOOL statistics invariant to the channel labelling nor the 8/15 routing cap appears anywhere.
- No change landed after any learned-arm record. There are no learned-arm records: phase 2 queued 0 runs.
- So the ALLPOOL labelling and routing-cap problems touch no reading of F's. They are still present in the module G and H are to vendor (`50dc8e4`, blob `ffe249bb64`, unchanged).

## Counts

| | rows | MATCH | PARTIAL | DISAGREE | NOT RECORD-CHECKABLE |
|---|---|---|---|---|---|
| Report 9 | 62 | 52 | 6 | 0 | 4 |
| README section | 10 | 9 | 1 | 0 | 0 |
| **Total** | **72** | **61** | **7** | **0** | **4** |

## 1. Report 9, statement by statement

### Header and setup

| # | statement | source | recomputed | verdict | note |
|---|---|---|---|---|---|
| 1 | Task module `explore_f_tasks.py` at 50dc8e4 (`50dc8e460b2c89ab6946f03b628598d7d6223d5e`) | git | full hash `50dc8e460b2c89ab6946f03b628598d7d6223d5e`; blob `ffe249bb64`, the same at every commit 8ba99bb..52e2ea7 (§B) | MATCH | |
| 2 | "It was pushed before any S89 run" | git, log | commit 17:54:31; first run started 18:02:17 (s89.log l.1) | MATCH | Commit order only; the push time is not in git. |
| 3 | "This is the commit G and H vendor" | git | no commit on any remote branch other than F's 50dc8e4 holds blob `ffe249bb64` (`git log --remotes --find-object`); G's tip (7ab98c1) still has `de4746371d` (= a199d5e) | NOT RECORD-CHECKABLE | A statement of intent. |
| 4 | Screen and readings committed at 8ba99bb before any run | git, log | 8ba99bb 18:02:14; phase 0 started 18:02:17, "git 8ba99bb+dirty" (s89.log l.4) | MATCH | The dirty start (screen review §7.6) is not mentioned. The child code is the committed code (row 5). The parent's `decide()` is not in the SHA, but the committed `decide()` reproduces every decision file (row 14). |
| 5 | Xeon @ 2.80GHz; torch 2.14.0+cu130; 1 thread; code SHA ea9728c636c6 | records | 26-module SHA from blobs at 8ba99bb = `ea9728c636c64a20def4540e0601e391f51de2c6` = every record's `code_sha`; identical at all 26 commits; every record has cpu 2.80GHz and torch 2.14.0+cu130 (§A) | MATCH | Threads: log l.4 "torch threads 1". |
| 6 | 28/28 ok, none retried, none diverged | records | 28 ok; attempts {1}; collapsed {False}; no NaN in any curve (§E) | MATCH | |
| 7 | Seeds 2800–2801; 2802–2811 not used | records | seeds {2800, 2801} in both stores | MATCH | |
| 8 | "The block 2800–2839 appears in no record on any branch" | 13 remote tips | the only hits are F's own S89 files on explore-F: s89.log (56 lines), s89_tasks_results.json (56), s89_diag_results.json (12), s89diag.log (12) (§I) | MATCH | True of everything other than these runs, as in screen review §3. |

### Validity pilots (Counts first)

| # | statement | source | recomputed | verdict | note |
|---|---|---|---|---|---|
| 9 | POOL Hebbian: 24000 0/2 [0.00, 0.66] NEVER, unbound 0.62 / 0.48; 48000 0/2, unbound 0.72 / 0.51 | records | 0.6226 / 0.4824; 48K 0.7173 / 0.5146; transition None (the record's and `rlib.transition(curve)`) | MATCH | |
| 10 | POOL delta: 2/2 [0.34, 1.00] RELIABLE, 3600 / 2400 at both budgets | records | 3600 / 2400 in record and recomputed; 48K identical | MATCH | |
| 11 | ALLPOOL Hebbian: 0/2, 0.33 / 0.50; 48K 0.32 / 0.48 | records | 0.3306 / 0.4976; 0.3193 / 0.4756 | MATCH | |
| 12 | ALLPOOL delta: 2/2, 3600 / 3600 at both | records | 3600 / 3600 | MATCH | |
| 13 | Wilson and band strings | F's `count_str`; rlib | 0/2 [0.00, 0.66] NEVER; 2/2 [0.34, 1.00] RELIABLE by both | MATCH | |
| 14 | READINGS: POOL "untested"; ALLPOOL "untested" | decision file; `decide()` | `decide()` from 8ba99bb, re-importing `_arms()` per stage as `run_s89.sh` does, on the stores at 368d26c / e036951 / 15b550d / dd09246, equals the decision file at each commit; final budget UNTESTED for both; the rep6 expression gives "untested" (§D) | MATCH | Mechanical. Phase 0 not stopped (max 0.401 ≤ 0.6). 24K: Hebbian 0/2, so at24 False and double True. 48K: Hebbian 0/2, so UNTESTED. |
| 15 | "Delta does on both tasks; Hebbian does on neither, at either budget" | records | delta 8/8; Hebbian 0/8 | MATCH | |
| 16 | "No baseline ran: phase 2 is conditional on a task reading 'valid' or 'needs the longer budget'" | 8ba99bb `_arms` l.127–131; log | phase-2 arms exist only if `budget[t]` is an int; ARMS at the final decision holds no BASELINE/SINGLE arm; log "0 runs queued" | MATCH | |

### Phase 0

| # | statement | source | recomputed | verdict | note |
|---|---|---|---|---|---|
| 17 | LITE NOMEM 0.166 / 0.168; NOMEM_SELF 0.407 / 0.374 | records | 0.16553 / 0.16797; 0.40723 / 0.37354 | MATCH | |
| 18 | POOL 0.167 / 0.168; 0.401 / 0.398 | records, decision | 0.16699 / 0.16797; 0.40088 / 0.39844 | MATCH | |
| 19 | ALLPOOL 0.169 / 0.169; 0.381 / 0.372 | records, decision | 0.16943 / 0.16895; 0.38135 / 0.37158 | MATCH | |
| 20 | "No task was stopped" | decision | stopped False for both; max 0.401 | MATCH | Screen review §7.3 showed the rule could not fire: the Bayes bound is 0.41–0.42. The report does not say so (fix 5). |

### The cuts

| # | statement | source | recomputed | verdict | note |
|---|---|---|---|---|---|
| 21 | LITE: CTX 2 (0.114), KEY 4 (0.102–0.103), VAL 16 (0.022–0.024); cuts 2 CTX + 4 KEY; 2 CTX + 3 KEY (0.535) | tables.md, log CHECK | 0.1136–0.1137; 0.1017–0.1027; 0.0220–0.0236; same cuts and mass | MATCH | The frequencies come from torch's generator and are not recomputed here. The referee's numpy replica (screen_checks_s89_F_out.txt §E) gives the same cuts. |
| 22 | POOL: KEY 32 (0.012–0.014); largest-ratio cut "the 2 CTX only"; half-mass 2 CTX + 12 VAL (0.502) | tables.md, log | 0.0120–0.0138; same | MATCH | |
| 23 | ALLPOOL: CTX 16 (0.013–0.015); largest-ratio cut "the 16 VAL (ratio 1.46)"; half-mass 16 VAL + 10 CTX (0.509) | tables.md, log | CTX 0.0128–0.0151; 0.0220 / 0.0151 = 1.457 | MATCH | |
| 24 | "The 2 CTX types hold 0.227 of the mass … adds 12 of the 16 value types" | tables.md | 0.1136 + 0.1137 = 0.2273; 12 VAL | MATCH | |
| 25 | "Every type is below 1/8 (max 0.0236)"; values are the commonest types | log CHECK | max 0.0236 (VAL); CTX 0.0151, KEY 0.0138 | MATCH | |
| 26 | What MQAR proper adds | prose | | NOT RECORD-CHECKABLE | |

### What the failure looks like

| # | statement | source | recomputed | verdict | note |
|---|---|---|---|---|---|
| 27 | At the end of every Hebbian pilot, margin 1.00 and η² by stream at KEY 1.00 | records `end` | 1.0000 / 1.0000 on all 8 Hebbian pilots | MATCH | On ALLPOOL, η² is pooled under the first-appearance index that the perfect gate is itself given. It is 1 by construction there (screen review §7.1), so it confirms the gate, not a routing finding. |
| 28 | "The curves sit at 0.29–0.34, then some step to about 0.48–0.50" | curves | before the step, evaluations from 2400 on lie in 0.297–0.346; at 1200, 0.271 (ALLPOOL 2801) and 0.286 (ALLPOOL 2800). Steps: POOL 2801 to 0.508 at 19200, then 0.48–0.50; ALLPOOL 2801 to 0.48 at 13200, then 0.465–0.498 | MATCH | Approximate at the first evaluation. |
| 29 | "POOL 2800 alone climbs on, to 0.72 at 32400, and holds 0.69–0.72 to 48000" | curve (48K) | 0.720 at 32400 ✓; 32400–48000 range **0.691–0.728** (0.7275 at 40800) | **PARTIAL** | The top of the range is 0.73, not 0.72. Minor. |
| 30 | "Whether or not the queried pair was its key's last write, accuracy differs by at most 0.06 (POOL 2800 at 24000: 0.65 vs 0.59)" | `write_order` | Example: 0.652 vs 0.593 (0.060) ✓. Over the four 24K Hebbian pilots, max 0.060 ✓. Over the 48K Hebbian pilots: **0.082** (POOL_48K 2801: 0.555 vs 0.473) and **0.064** (ALLPOOL_48K 2801: 0.535 vs 0.471) | **PARTIAL** | Holds at 24000 only. The paragraph discusses the 48K curves and gives no budget. |
| 31 | "The delta perfect gate binds 8/8, at 2400–3600" | records | 8/8; transitions {2400, 3600} | MATCH | |
| 32 | "S70's Hebbian perfect gate bound 3/3 (1200, 1200, 8400) under one Adam at 1e-3" | rh_lite_results.json @52e2ea7 | S70_ORACLE_HEBB 330: 8400, 331: 1200, 332: 1200; sched "Adam 1e-3 throughout" | MATCH | |
| 33 | The pilots differ from S70's oracle "in two ways": the key pool and SLOW | | | NOT RECORD-CHECKABLE | Also different: the seeds (2800–2801 against 330–332) and the vocabulary size, which is part of the pool. |

### Post-hoc diagnostic

| # | statement | source | recomputed | verdict | note |
|---|---|---|---|---|---|
| 34 | "written and committed at 4639d16 after the readings came in; changes no reading" | git | 4639d16 at 19:16:29. UNTESTED was committed at 15b550d, 19:15:09, with all 28 records. The decision file is unchanged after that (§J) | MATCH | |
| 35 | Three arms, 2800–2801, 24000, "the S89 models unchanged" | diag child; records | The child calls `s89.arm_def` / `run_kw` / `make_model`. The Adam arms drop `run_kw` and `check`, giving `Adam(model.parameters(), 1e-3)` (test_binding_onset l.209–210); `group_names` None. The SLOW arm has `group_names` [[], [rest]]. 24000 updates, 2 seeds | MATCH | |
| 36 | D_HEBB_LITE_SLOW 2/2 [0.34, 1.00] RELIABLE, bound @2400 / @2400 | records | transitions 2400 / 2400 (record and recomputed); `count_str` as stated | MATCH | |
| 37 | D_HEBB_POOL_ADAM 0/2 NEVER, unbound 0.32 / 0.33 | records | 0.3203 / 0.3291 | MATCH | |
| 38 | D_HEBB_ALLPOOL_ADAM 0/2 NEVER, unbound 0.31 / 0.31 | records | 0.3135 / 0.3145 | MATCH | |
| 39 | Under Adam it "stays at 0.29–0.34 for all 24000 updates on both pool tasks"; routing perfect throughout (margin 1.00, η² 1.00) | curves, stats | curve ranges 0.290–0.334, 0.298–0.338, 0.292–0.333, 0.301–0.335; margin and η² 1.0 at every evaluation | MATCH | |
| 40 | Adam "never reaches the step to about 0.5 that SLOW gave on 3 of 4 runs at 24000 (all but ALLPOOL 2800)" | curves | SLOW 24K maxima: POOL 0.623 / 0.508, ALLPOOL 0.331 / 0.498. Adam maxima ≤ 0.338 | MATCH | |
| 41 | "SLOW is not what stops it … The key pool is." "So both tasks read untested because of how the Hebbian channel handles 4 keys drawn from 32 types. The recipe is not the cause." | diag docstring; records | Numbers as rows 36–40 | **PARTIAL** | The numbers match. The wording draws a causal conclusion from a post-hoc diagnostic with 2 seeds per cell, whose own docstring says "Descriptive … No reading" (explore_f_s89_diag.py l.12). The section header does say "post-hoc" and "changes no reading". |
| 42 | "6/6 runs ok, 0.68 h of run time, 19:18–19:31 UTC. explore_f_delta_checks passed 26/26 before the runs. The diagnostic adds no CHECK of its own" | records, s89diag.log | 6/6 ok, attempts 1; log 19:18:25–19:31:26; 26 ok. Run time: **0.689 h** by `secs_wall`, which is what `store_meta` uses for the 4.05 h figure; 0.684 h by `secs` | **PARTIAL** | The run time is 0.69 h on the basis the report uses elsewhere. The diagnostic child's `checks()` returns one hard-coded True (explore_f_s89_diag_child.py l.36–37). No S89 CHECK covered the perfect gate on LITE under SLOW, or the Adam path without param groups. |

### Design decisions

| # | statement | source | recomputed | verdict | note |
|---|---|---|---|---|---|
| 43 | The pool tasks are lite's batches with the identities renamed (CHECKed) | log CHECK | "the batch is lite's batch with the identities renamed … True": ok, at every segment | MATCH | From the log only. |
| 44 | ALLPOOL's stream index is by first appearance; Hebbian logits under `task.perfect_gate` equal the built-in gate's bit for bit | log CHECK | ok, at every segment | MATCH | From the log only. |
| 45 | NOMEM sees 4 tokens, NOMEM_SELF 10, CHECKed with randomised conv weights; the rule stops if either exceeds 0.6 | log; s89 l.41–42 | ok, at every segment; rule as stated | MATCH | |
| 46 | The pilots used SLOW; the docstring calls SLOW "the stricter pilot" | s89 l.46–48 | as stated | MATCH | |
| 47 | Under SLOW every parameter is at 1e-4 for updates 1–2400 and 1e-3 after | log lrs CHECK; records | O_DELTA_ALLPOOL: groups [] / 5 rest; lrs [0.001, 0.0001] at 1 and 2400, [0.001, 0.001] at 2401. Pilot records: `group_names` [[], [rest]] | MATCH | |
| 48 | 48K curves equal the 24K runs' bit for bit through 24000 ("verified") | records | All 8 pairs. Curve, stats and gradient norms are identical over the common prefix (Hebbian to 24000; delta early-stopped at 4800 / 6000 in both runs) (§E4) | MATCH | |
| 49 | (i) The first batch at seed 2700 equals G's vendored copy at e46799a | log CHECK | ok, at every segment. G's e46799a copy is blob `de4746371d`, the same as a199d5e | MATCH | From the log only; needs torch. |
| 50 | (ii) S70's path on seed 2700 reproduces G's S87 BASELINE\|2700 bit for bit through 2400 (curve, gradient norms at 1, 10, 100) | log; G record @e46799a | The printed run curve equals the printed reference, which equals the record's `curve[:2]` at e46799a, in all 4 segments (§L). Gradient norms are not printed | MATCH | The curve is checked here; the gradient norms only through the CHECK's own True. This is S70's path, not the screen's (screen review fix 3). |

### CHECKs

| # | statement | source | recomputed | verdict | note |
|---|---|---|---|---|---|
| 51 | explore_f_delta_checks 26/26 at the start of every segment | logs | 26 ok in phase0 / 1 / 1b / 2 and the diagnostic | MATCH | |
| 52 | S70_DELTA\|330 (a 2.10GHz record) reproduces bit for bit through 2400 | log; record @61ba46e | The printed reference equals the record's `curve[:2]` at 61ba46e, and the run equals the reference, in all 4 segments | MATCH | The CHECK reads the working tree (child l.255). |
| 53 | "explore_f_tasks.py's first **13771 bytes** … equal a199d5e's file" | git | a199d5e's file is **14,101 bytes = 13,771 characters**. 50dc8e4's first 14,101 bytes equal it | **PARTIAL** | The CHECK is right. It tests `new.startswith(old)` on the decoded text, so it covers the whole old file. But its label prints `len(str)` as bytes (child l.214), and the report copies the label. |
| 54 | Each new task, 4096 sequences: renamed; key types distinct and from the pool; every key exactly twice; query determined; one CTX per block; labels and the perfect gate | log CHECK | all ok, at every segment | MATCH | From the log only. |
| 55 | POOL max KEY 0.0138 < min VAL 0.0220; ALLPOOL no type above 1/8 (max 0.0236) | log CHECK | as stated | MATCH | |
| 56 | The frequency tables; LITE's equals G's published S87 table and cut | log CHECK | ok | MATCH | |
| 57 | Models: the perfect gates one-hot and equal to `task.perfect_gate`; NOMEM reaches 4 and 10; NOMEM's read zero; groups and lrs right at 1, 2400, 2401 | log CHECK | 33 child CHECK lines ok and 0 FAIL in each of the 4 segments | MATCH | The lrs are CHECKed on 3 arms: BASELINE_POOL, SINGLE_DELTA_ALLPOOL, O_DELTA_ALLPOOL. |
| 58 | Smoke runs (B = 1024 / 2048; 300 updates on seed 99), not stored | | | NOT RECORD-CHECKABLE | |

### Runtime

| # | statement | source | recomputed | verdict | note |
|---|---|---|---|---|---|
| 59 | 4.05 h of run time on 4 workers | records | Σ `secs_wall` = 4.047 h (Σ `secs` 4.02 h) | MATCH | |
| 60 | phase 0: 12 runs, 18:02–18:29; phase 1: 8, 18:30–18:47; phase 1b: 8, 18:47–19:15; phase 2: none queued, 19:15–19:18 | s89.log; git | Stage starts 18:02:17 / 18:29:59 / 18:47:49 / 19:15:11; done 19:18:14. Commits 368d26c 18:29:56, e036951 18:47:47, 15b550d 19:15:09 | MATCH | |

### Anything unexpected

| # | statement | source | recomputed | verdict | note |
|---|---|---|---|---|---|
| 61 | "Renaming lite's 4 fixed keys to 4 of 32 is enough to cause it. Per the diagnostic this holds under both recipes, and SLOW binds lite." | diag | numbers as rows 36–40 | **PARTIAL** | The same post-hoc-as-result wording as row 41. |
| 62 | Largest-ratio cut on ALLPOOL selects the values. NOMEM_SELF reaches 0.37–0.41 on all three tasks, below the 0.6 stop. "Any accuracy near 0.4 … not evidence of memory use" | records | 0.372–0.407 | MATCH | Close to the referee's Bayes bounds of 0.414–0.424 (out §E). That the rule could not fire is still not said. |

## 2. README section ("Session F, round 6: S89-F …", +20 lines at 52e2ea7)

| # | statement | recomputed | verdict | note |
|---|---|---|---|---|
| M1 | Task module at 50dc8e4 | as row 1 | MATCH | |
| M2 | Phase 0: NOMEM 0.17 on every task; NOMEM_SELF lite 0.41 / 0.37, -pool 0.40 / 0.40, -allpool 0.38 / 0.37; neither stopped | 0.166–0.169; 0.407 / 0.374, 0.401 / 0.398, 0.381 / 0.372 | MATCH | |
| M3 | -pool: delta 2/2 (3600, 2400); Hebbian 0/2 at 24000 (0.62, 0.48) and at 48000 (0.72, 0.51); **untested** | as rows 9, 10, 14 | MATCH | |
| M4 | -allpool: delta 2/2 (3600, 3600); Hebbian 0/2 (0.33, 0.50) and (0.32, 0.48); **untested** | as rows 11, 12, 14 | MATCH | |
| M5 | Baselines "not run: no task read valid" | 0 phase-2 runs | MATCH | Strictly, "no task read valid or needs the longer budget". |
| M6 | Post hoc: lite 2/2 (2400, 2400); -pool 0/2 (0.32, 0.33); -allpool 0/2 (0.31, 0.31); reading "none (descriptive)" | as rows 36–38 | MATCH | |
| M7 | Delta binds both at 2400–3600; Hebbian neither, at 24000 or 48000; routing perfect (margin 1.00, η² 1.00) | as rows 15, 27, 31 | MATCH | |
| M8 | "Post hoc, the key pool is the cause, not SLOW … stays at about 0.32 on both pool tasks under S70's one-Adam oracle recipe" | final 0.313–0.329; range 0.290–0.338 | **PARTIAL** | Labelled post hoc, but stated as a cause (as row 41). |
| M9 | Cuts: -pool largest-ratio cut 2 CTX, half-mass adds 12 VAL; -allpool both cuts select 16 VAL first; no type above 1/8 | as rows 22–25 | MATCH | |
| M10 | About 0.4 reachable without memory on all three tasks by a model seeing the last 10 tokens | as row 62 | MATCH | |

## 3. Pre-registration order

**Commits from 8ba99bb to 52e2ea7** (19 October, UTC; §J). The "records" column counts runs in `s89_tasks_results.json`.

| commit | time | content | records | decision file |
|---|---|---|---|---|
| 8ba99bb | 18:02:14 | `explore_f_s89_tasks.py` (docstring: phases, rules, readings; `decide`), `_child.py`, `explore_f_reports6.py` (reading expression) | 0 | — |
| (run) | 18:02:17 | phase 0 starts, "git 8ba99bb+dirty" | | |
| 698a20f | 18:02:47 | run_s89.sh, keepalive.sh, s89.log | 0 | — |
| 0378b3a | 18:05:38 | delta-checks json, log, runtime | 0 | — |
| 40ea56e | 18:25:20 | first phase-0 records | 8 | — |
| *3ae815b (review-V)* | *18:26:30* | *screen_review_s89_F.md* | | |
| 368d26c | 18:29:56 | phase 0 complete; decision `phase0` | 12 | phase0 (not stopped ×2) |
| 644bccb … a84013d (11 commits) | 18:30–18:55 | pilot records and logs | 13–24 | + pilots (at e036951, 18:47:47: at24 False, double True ×2) |
| 15b550d | 19:15:09 | 48K records; budget UNTESTED ×2 | 28 | final |
| 4639d16 | 19:16:29 | `explore_f_s89_diag.py`, `_child.py` (post hoc) | 28 | unchanged |
| dd09246, 09f28e3 | 19:18:12, 19:18:15 | phase 2 (0 runs), logs | 28 | unchanged |
| (diag runs) | 19:18:25–19:31:26 | 6 runs, "git 09f28e3+dirty", code d4af41b711b3 | | |
| ebe5bcb | 19:31:23 | diag results and log; report_9.md first version (with `DIAG_PLACEHOLDER`) | 28 | unchanged |
| 52e2ea7 | 19:32:25 | report_9.md filled in; README section | 28 | unchanged |

**Rules fixed before records.**
- `explore_f_s89_tasks.py`, `explore_f_s89_tasks_child.py`, `explore_f_tasks.py`, `explore_f_reports6.py`, `explore_f_common.py`, `explore_common16.py` and `explore_f_reports.py` each have **one blob** across all 26 commits (§B).
- The 26-module code SHA is `ea9728c636c6` at every commit and equals every record's.
- So no code or rule changed after any record it reads.
- The readings, the phase rules and the reading expression were committed 3 s before the first run started, and 23 min before the first record was committed.

**Caveat: the dirty tree.** Phase 0 started on a dirty tree. The SHA proves that the child's 26 modules were the committed ones. The parent's `decide()` is not in the SHA, but the committed `decide()` reproduces the decision file exactly at all four stages (§D). The report does not mention the dirty start.

**The diagnostic.**
- **Code and the records it diagnoses.** Its code (`4639d16`, 19:16:29) was committed **after** the final decision and all 28 records (`15b550d`, 19:15:09; the 48K Hebbian runs ended about 19:11 by the log).
- **Code and its own runs.** The code was committed **before** its runs, which started 19:18:25. Its 27-module SHA, recomputed at `4639d16`, is `d4af41b711b3e9f1…`, equal to all 6 records. Both files have one blob from 4639d16 to 52e2ea7.
- **Overlap with phase 2.** The phase-2 stage (19:15:11–19:18:14) was running when 4639d16 was committed. It queued 0 runs and does not import the diagnostic modules.
- **Labelled post hoc everywhere?** Yes:
  - the commit message of 4639d16 ("post-hoc … written after the pilots read untested");
  - both docstrings ("POST HOC");
  - the report's section header;
  - the README row ("post hoc … none (descriptive)") and summary ("Post hoc, …").
  - The data commit ebe5bcb's message says only "S89 diagnostic runs".
- **Read as a result?** In wording, yes, in three places, though no reading changes:
  - report "So both tasks read untested because of how the Hebbian channel handles 4 keys drawn from 32 types. The recipe is not the cause." (row 41);
  - report "Per the diagnostic this holds under both recipes" (row 61);
  - README "the key pool is the cause, not SLOW" (M8).

  The diagnostic's own docstring says "Descriptive … No reading", and each cell has 2 seeds.

**Order verdict: clean.** No code or rule changed after a record it reads, and the post-hoc diagnostic was coded after the records it diagnoses and before its own runs. The only deviations are in wording: the causal phrasing above and the unmentioned dirty start.

## 4. The referee's pre-run fixes (screen_review_s89_F.md, 3ae815b, 18:26:30)

The review was committed during phase 0. That was before any pilot, and before any learned arm, of which none ran. No code file changed after 8ba99bb (§B). Report 9 and the README section contain none of the following: "0.533", "8/15", "64/120", "invariant", "_inv", "Bayes", "0.42", "dirty", "p⁴", "unpaired", "v2", "referee" (§K).

| fix | taken? | commit | after learned-arm records? | note |
|---|---|---|---|---|
| 1. ALLPOOL statistics invariant to a per-sequence channel swap (`eta_key_by_stream_inv`, `val_cos_inv`) | **No** | — | n/a (no learned-arm records) | No record carries `_inv` fields, and `explore_f_tasks.py` is unchanged. The only ALLPOOL routing statistics are the perfect-gate pilots' η² 1.00, which is 1 by construction under the first-appearance index (row 27). No reading was affected. The module G and H are to vendor still has the convention. |
| 2. ALLPOOL routing cap: 64/120 = 8/15 = 0.533 for any fixed type-to-channel map. Docstring sentence, or a type-level perfect gate pilot | **No** | — | n/a | Not mentioned in report 9 or the README. The report says only "the model's built-in perfect gate … does not apply here". No type-level gate was run. ALLPOOL reads untested on the Hebbian perfect gate, so the cap did not decide a reading. A null on ALLPOOL in G or H would still be confounded. |
| 3. Tie the screen's own path to a record; pin `G_REF`; read S70's record from a commit; "first 1, 10 and 100 batches" | **No** | — | n/a | Child l.51 still has `G_REF = "origin/claude/explore-G"`. No CHECK runs `explore_f_s89_tasks_child.run` against a record. Child l.255 reads the working tree. The log still prints "(functions of the first batch)". Report 9 describes (ii) correctly as "S70's own path". |
| 4. The baselines' role ("unpaired descriptive rates") | No | — | n/a | Moot: phase 2 queued 0 runs. |
| 5. Phase 0's meaning (Bayes bound 0.19 / 0.42; "not stopped" is guaranteed) | **No** (partly in spirit) | — | n/a | The report still calls phase 0 "positional solvability" and says "No task was stopped". Row 62's "any accuracy near 0.4 … not evidence of memory use" is close to the point, but gives neither the bound nor the fact that the rule could not fire. |
| 6. Failed records named | No | — | n/a | Moot: 28/28 and 6/6 ok. |
| 7. Assert that the learned arms' gates read no labels | No | — | n/a | Moot: no learned arm ran. |
| 8. `fail_class_v2`; NaN `key_sep`/`key_part` on pool tasks; `routed_end` None guard | No | — | n/a | Moot: no BASELINE ran. The pilot records still store lite-named `key_sep`/`key_part`, and the report does not use them. |
| 9. For G and H: the repeat statistic does not select only the keys | not F's | — | — | |
| 10. Operating characteristics (P(valid) = p⁴) | No | — | — | |
| §7.6: report the dirty start | No | — | — | See row 4. |

## 5. Seeds, SHAs and mechanics: summary

- **Seeds.** 2800–2801 only, in both stores. Across 13 remote branch tips, 2800–2839 occurs only in F's own S89/diagnostic files (§I).
- **SHAs.** S89 `ea9728c636c64a20def4540e0601e391f51de2c6`, recomputed from the 26 blobs at 8ba99bb and constant to 52e2ea7. Diagnostic `d4af41b711b3e9f1cf45f27535564b6203b4a5bf`, recomputed from the 27 blobs at 4639d16. The modules hash as `"branch"`, because MAIN is False, so no main-line blob enters.
- **Mechanics.** The "untested" verdicts are mechanical (row 14): F's own `decide()` and reading expression, run on the records, give exactly the committed decision file and readings at every stage.

## 6. Errors in the referee's own files

None found.
- `screen_review_s89_F.md` l.123 says G's vendored copy "is that file" (a199d5e's): confirmed, both are blob `de4746371d`.
- The tallies in §1 add up: 512 + 480 + 7,200 + 16,384 = 24,576 = 2 × 12,288.
- The P(valid) table is right: 0.95⁴ = 0.815, 0.9⁴ = 0.656, 0.8⁴ = 0.410.
- One wording point. The review's header says the tip "has since moved to `0378b3a`". By the review's commit time (18:26:30), F had committed `40ea56e` (18:25:20, the first 8 phase-0 records). This is consistent with "written while phase 0 was running", and it does not affect any finding.

## 7. What could not be checked

- **Needs torch.** Anything that uses the generators or a model:
  - the frequency tables themselves (only the referee's numpy replica, which agrees on every cut);
  - the CHECKs that rename the tasks, check the perfect-gate logits, check the NOMEM reaches, and compare the first batch;
  - the gradient-norm part of the G87 CHECK.

  These rows rest on the log's CHECK lines (rows 43, 44, 49, 54).
- **The push time of 50dc8e4** relative to the runs. Only the commit time is in git.
- **The parent's working-tree content at run time.** "8ba99bb+dirty" is not covered by the code SHA. It is only shown to be consistent through the `decide()` recomputation.
- **The smoke runs (row 58)**, and whether G and H have vendored 50dc8e4 (row 3).
- **The diagnostic's Adam path and the perfect gate on LITE.** No CHECK covers either. Their learning-rate groups are inferred from the code and from `group_names` in the records.

**Correction (22 October).** Row 30's -allpool write-order figure (0.064) is withdrawn with F's erratum `a0dc3c34`: `write_order_acc` read the CTX token as the stream index on -allpool. The records' 0.069 share of "last write" queries on -allpool would have shown the field was wrong. Row 30 stays PARTIAL on the -pool figure (0.082). `screen_review_s89_F_v2.md` row 10 (0.86 / 0.87) quoted the same withdrawn field.
