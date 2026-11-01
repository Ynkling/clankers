# Check: Session G's report 11 (S91b) at `claude/explore-G` `330290dd`

*Referee V, 23 October 2026. What this covers:*
- *`explore_out/G/report_11.md` ("r11"), the README's S91b section (`explore_out/README.md` l.797-814 at `330290dd`), and the commit headline of `330290dd`;*
- *the S91b records (`s91b_results.json`, 136 runs; `s91b.log`; `s91b_summary.json`; `s91b_validity.json`; the three `checks_s91b_*.json`), with S89H's and S89's stores for the k = 2 counterparts and F's report 10 for the -ctxpool single-channel figure;*
- *every commit from `b981c3fa` (S91b's code) to `330290dd`: 73 commits, both ends included.*

*It was read against `check_G_report10.md`, `screen_review_s90_F.md` and `check_F_report10.md` (F's module `9017b82`, the caps, DISCOVERED-at-the-cap, the swap-invariant statistics). No referee review of S91b took place before the run.*

*The numbers come from `check_G_report11.py` ("chk §N"). The script reads git objects only, writes nothing and prints only. Like `check_G_report10.py`, it execs G's modules from blobs statement by statement, with stub modules, and needs no torch. From G it takes `explore_g_s91b` at `9949e495` (the version that ran), `explore_g_s89h`, `explore_g_s89`, `explore_g_rhl_parent`, `explore_g_common`'s store and statistics helpers and `test_router_layout.fail_class`. From F it takes `explore_f_tasks.pool_outcome` and `type_map_cap`. It then runs G's own `report()` on the committed records, sends the summary write to memory, and compares the output with the log line by line. `rlib` supplies the referee's own `transition`, `bound`, `fail_class`, `wilson` and `fisher1`.*

## Verdict

**The numbers are right, and the reading is the committed rule's mechanical output.**
- **Reproduction.** G's `report()` reproduces the log from the count table onward: 364 of 364 lines, 0 differing. Its summary equals `s91b_summary.json` (chk §3).
- **The reading.** An independent recomputation with rlib's BOUND and the screen's ROUTED_k gives "the channels do not suffice" on both memories (chk §4).
- **The budget reached every run, this time through the harness.** The fix to `explore_g_common.py:164` (V's fix 8) went in at `4314e35c` (17:23:21), before S91b's code. All 136 records carry iters 24000, the arm's budget.
- **The vendored module** is F's `9017b82` blob `cfe3fe27`, and its first 49,923 bytes are `e24d5e1`'s.

**Order: clean.**
- The readings, the outcome and the rule were committed at `b981c3fa` (17:33:17), before any record. The first records were committed at 17:58:32.
- `9949e495` (17:43:04) came after a CHECK pass and before any run. It changed only `run_checks()` and a docstring line in the parent. The parent is not in the code SHA. Every reading function is ast-identical across the change.
- The code SHA `a0134e14a262` recomputes from its 36 rows, and every row matches its source at `9949e495`.

**Rows: 55 checked. 45 MATCH, 10 PARTIAL, 0 DISAGREE.**
- None of the PARTIAL rows changes a count or a reading.
- They are about:
  - the failure-class column (two of G's own classifiers disagree);
  - "about 6 channels" (`eff_ch` stands in for the type map's channel count);
  - "near 0 on both maps" for delta;
  - CHECK claims that are asserts or statements rather than CHECK rows;
  - "the closest arm".

**The name "the channels do not suffice" is accurate as the rule's label, but it is ambiguous, and its natural reading as a capacity claim is contradicted by the screen itself.**
- PTYPE, a type-keyed gate at k = 16, binds 2/2 and separates every sequence. So the channels suffice for the memory and the task.
- What 0/10 shows is that no learned gate found such a map within 24000 SLOW updates. That is a learning negative at this budget, not a capacity limit.
- Each arm's 0/10 excludes a true rate above 0.26 (one-sided 95% Clopper-Pearson).
- r11's body says this correctly ("The channel cap is not what limits these arms", "not whether the channels exist"). The label and the headline do not. §4 gives the wording.

## 1. Statements against the records

### 1a. r11 header and run (chk §0, §0b, §0c, §1, §7)

| # | statement | recomputed | verdict | note |
|---|---|---|---|---|
| A1 | Xeon 2.10GHz, torch 2.14.0+cu130, 1 thread; code SHA a0134e14a262 | 136/136 keys and meta; sha1(json.dumps(sha_rows)) = `a0134e14a262…` | MATCH | |
| A2 | `explore_f_tasks.py` vendored from F's 9017b82, blob cfe3fe2, append-only over b0c6332, asserted | G's blob = F's `9017b823` blob `cfe3fe276f7c`, unchanged over the range; 73,009 bytes; starts with `e24d5e1`'s 49,923 bytes; CHECK row ok | MATCH | |
| A3 | k = 16, REG3, SLOW, 24000; seeds 2992-2993 / 2994-3003 | k 16, recipe slow and iters 24000 on 136/136; seeds as stated | MATCH | |
| A4 | spec, readings and CHECKs committed at b981c3f before any run | `b981c3fa` 17:33:17; first records `949936bc` 17:58:32 | MATCH | §2 |
| A5 | one CHECK row superseded at the first CHECK pass, before any run (9949e49) | first pass `checks_s91b_20261022_174236`: 146/147 ok, git b981c3f; the failing row is S89's "is F's b0c6332 blob"; `9949e495` relabels it; no store before `1224d417` | MATCH | §2 |
| A6 | 136/136 runs ok, none retried | ok 136, attempt 1 on 136 | MATCH | |
| A7 | every record carries iters 24000, the arm's budget (136/136, printed) | 136/136 = meta ARMS iters; 116 unbound runs stop and end their curve at 24000; log "records: iters == ARMS[arm]['iters'] on 136/136 (ok)" | MATCH | |
| A8 | worker restart after 84 of 120 learned; the resume re-ran all 147 CHECKs (ok) under the same SHA | 100 records (16 + 84) at `9660733a`; `checks_s91b_20261022_224522` 147/147, git 9660733+dirty; 36 queued after the restart; SHA unchanged | MATCH | |
| A9 | replaces S91's remaining Hebbian half, which was not rerun | `s91_results.json` and `s91.log` unchanged `cbca3e36..330290dd`; the user's instruction is quoted at s91b:10-11 | MATCH | The instruction itself is not in the record. |

### 1b. Counts and readings (chk §2, §2b, §3, §4)

ROUTED_k is the screen's ROUTED-BOUND (`rbk`, s91b:91-93): BOUND, end margin ≥ 0.9, end `seq_separated_frac` > 0.9.

| # | statement | recomputed | verdict | note |
|---|---|---|---|---|
| B1 | PIDX 2/2 (3600, 2400), (1200, 1200), (7200, 7200), (3600, 3600) | ctxpool H 2992 @3600, 2993 @2400; ctxpool D 1200 ×2; allpool H 7200 ×2; allpool D 3600 ×2; ROUTED_k 2/2 in each; rlib transition = record on 136/136 | MATCH | |
| B2 | PTYPE 2/2, TYPE-ROUTED 2 (four cells) | ROUTED_k 2/2 and TYPE-ROUTED 2 in each | MATCH | |
| B3 | REG3 0/10 [0.00, 0.28] (1 BOUND) on -ctxpool H; 0/10 elsewhere | 0/10 ×4; BOUND 1 (2995 @16800, margin 0.04) | MATCH | |
| B4 | REG3_KEYHINGE 0/10 ×4; -allpool H 2 BOUND | 0/10 ×4; allpool H BOUND 3000 @15600, 3003 @24000 (separated 0.72, 0.87) | MATCH | 3003 binds at the last evaluation. |
| B5 | REG3_REPEATHINGE 0/10 ×4; -ctxpool H 1 BOUND | 0/10 ×4; 3001 @12000 (margin 0.28, separated 0) | MATCH | |
| B6 | k = 2 counterparts 0/20 / 0/20 | `counterpart()` 0/20 on all six arm × memory pairs | MATCH | Those ran -allpool at k = 2 and 24000 (`check_G_report9`), so two variables differ from -ctxpool at k = 16. Moot at 0/10. |
| B7 | no learned run is TYPE-ROUTED | 0/120 (no learned run is ROUTED_k) | MATCH | |
| B8 | the k = 2 forms beside ROUTED_k | learned arms 0 under every form: the pooled `routed_bound`, `routed_bound(inv_view)`, F's `pool_outcome` (pooled or within-sequence) | (descriptive) | At k = 16, F's within-sequence form reads PTYPE 0/2. It reads channel 0 only, as s91b:29-30 says. |
| C1 | Hebbian + conv: PIDX 2/2, PTYPE 2/2, every arm 0/10 → "the channels do not suffice" | G's `readings()` and the independent rule (rlib BOUND) agree | MATCH | §4 |
| C2 | delta: the same | the same | MATCH | |
| C3 | "routes by type" needs ≥ 7/10 and discovered vs 0/20; Fisher p = 1 for each | p(0/10 vs 0/20) = 1 (G and rlib); for scale, 7/10 vs 0/20 gives 5.9e-5 | MATCH | |
| C4 | -allpool every learned arm 0/10 on both memories; 0 / 0 against -ctxpool everywhere | paired (0, 0) on all six | MATCH | |

Not counted as a row: B8 (descriptive).

### 1c. "What happened" (chk §2, §2b, §5)

| # | statement | recomputed | verdict | note |
|---|---|---|---|---|
| D1 | the two perfect gates are one ceiling by construction; trained identically seed for seed | curves equal on 8/8 (task × memory × seed) pairs; transitions and accuracies equal | MATCH | Hebbian: G = g gᵀ is identical, so equal by construction. Delta: `set_gates(g, g)` receives permuted channels, so equality there is observed, not by construction. |
| D2 | PTYPE TYPE-ROUTED 2/2, type map 1.00, 15.8 effective channels | type_map_k 1.000; eff_ch 15.77-15.81 (ctxpool), 15.59-15.88 (allpool) | MATCH | |
| D3 | REG3_KEYHINGE (H, -ctxpool): type map 0.97, ~6 channels, separated 0.20 / 0.84, margin 0.67, held-out 0.89 | 0.966; eff_ch 5.95 (4.5-7.6); 0.199 / 0.836; 0.666; acc 0.890 | MATCH | |
| D4 | its unbound class STREAM-PARTIAL 10/10 | `fail_class` (= `rec["fail"]`, = rlib) STREAM-PARTIAL 10; the log's count table (`hp.outcome`) **MERGED 8, STREAM-PARTIAL 2** | PARTIAL | G's two classifiers disagree, and r11 does not say which it uses. `hp.outcome` calls STREAM-PARTIAL "MERGED" when the pooled L3 map is not one to one (rhl_parent l.67-68). Under a type-keyed gate the pooled stream → channel map is not meaningful (e.g. 2994 map [12, 12]). So `fail_class` is the better reading here, but the printed log says MERGED. |
| D5 | REG3_REPEATHINGE: 0.53, ~7, 0.00 / 0.88, 0.50, 0.89 | 0.528; 7.23; 0.000 / 0.875; 0.499; 0.888 | MATCH | |
| D6 | its unbound class STREAM-PARTIAL 8 | `fail_class` STREAM-PARTIAL 8, OTHER 1 (and 1 BOUND); the log's table MERGED 7, STREAM-PARTIAL 1, OTHER 1 | PARTIAL | As D4. |
| D7 | "The key hinge's gate became a fixed function of the context type … But it put the 16 types on about 6 channels, too few and not confidently enough to separate most sequences" | type_map_k > 0.9 on 10/10 (0.90-1.00). eff_ch 4.5-7.6, but eff_ch = exp(entropy) of the overall mean gate over all positions (test_router_reliability l.262), not the number of channels the type map uses. A hard balanced map on m channels caps separation at 0.850 / 0.883 / 0.908 for m = 5 / 6 / 7 (F's `type_map_cap`). Observed separation: median 0.20, best 0.84 (3002, eff_ch 5.85) and 0.73 (2999, eff_ch 4.45) | PARTIAL | "Keys on type": supported. "Not confidently enough": supported, and the main shortfall, since median separation 0.20 is far below a 6-channel cap of 0.88. "Too few": an inference from a proxy, and it could bind only for the best 2-3 runs. Not measured. |
| D8 | REG3 alone: type map 0.02, separated 0.00, POSITION 4 and OTHER 2 among its classes | 0.018; 0.000 (max 0.000); POSITION 4, OTHER 2, STREAM-PARTIAL 2 (log: MERGED 2), KEY 1 | MATCH | |
| D9 | "On delta every arm stays near 0 on both maps" | separated: max 0.000 on -ctxpool, 0.004 on -allpool. type_map_k medians 0.000 / **0.228** / 0.065 (ctxpool REG3 / KH / RH), 0.008 / **0.239** / 0.006 (allpool); KH_D max 0.72 (2998), one RH_D run > 0.9 | PARTIAL | The delta key hinge's type map is about 0.23, not near 0. |
| D10 | delta failures STREAM-PARTIAL / OTHER / KEY | by `fail_class`: ctxpool OTHER 9 + KEY 1; SP 9 + KEY 1; SP 5 + OTHER 4 + KEY 1. allpool OTHER 10; SP 6 + OTHER 2 + KEY 2; OTHER 6 + KEY 4 | MATCH | The log's table prints SP as MERGED (D4). |
| D11 | held-out 0.50-0.77 on -ctxpool (delta) | medians 0.502 / 0.767 / 0.626 | MATCH | Per run, 0.49-0.93 (KH_D\|2998 0.928). |
| D12 | F's single delta channel binds -ctxpool 10/10 without routing | `check_F_report10` row 7: 10/10 | MATCH | |
| D13 | "The channel cap is not what limits these arms. At k = 16 a fixed type map could route every sequence (cap 1), and the ceiling shows the memory would bind with it" | `type_map_cap(16, 2, 16)` = 1; PTYPE 2/2 on all four cells | MATCH | |
| D14 | "they spread over 6-14 effective channels" | learned-arm eff_ch medians 5.93-14.01 | MATCH | |
| D15 | "but do not commit by type at 0.9 (Hebbian), or do not key on type at all (delta)" | Hebbian: no run ≥ 0.9 on margin or separated, but REG3_H does not key on type either (0.02). Delta: 1 of 60 runs has type_map_k > 0.9, and KH_D median 0.23 | PARTIAL | Holds for the hinged Hebbian arms. "At all" overstates delta's key hinge (D9). |
| D16 | budget note: the Hebbian key hinge still at 0.89 held-out and STREAM-PARTIAL at 24000; a longer budget not tested | median separation 0.152 at 18000 → 0.199 at 24000, max 0.227 → 0.836; median acc 0.862 → 0.890; rising in 3002 (0.19 → 0.84), 2999 (0.23 → 0.73), 2996 (0 → 0.39) | MATCH | The records support the caveat. Some runs were still moving at the cap. |

### 1d. r11's CHECKs paragraph (chk §7)

| # | statement | recomputed | verdict | note |
|---|---|---|---|---|
| E1 | 147, all ok, asserted before each segment | the two passes before runs: 147/147 (`175257`, git 9949e49+dirty) and 147/147 (`224522`); the first pass 146/147 aborted before any run | MATCH | |
| E2 | the vendored blob and its 49,923-byte prefix; F's check_ctx_task and check_inv_stats_k; cap 1 at k = 16, 64/120 at k = 2 | rows ok; V recomputed the cap with F's `type_map_cap` | MATCH | |
| E3 | at k = 2 the builder is S89H's and S89's; allpool.REG3_REPEATHINGE_<M>\|2842 reproduce bit for bit through 2400 | rows ok | MATCH | |
| E4 | at k = 16: untrained perfect gates ROUTED_k, PTYPE's type map 1.00 and PIDX's 0.04-0.05; the k-general form agrees with the k = 2 η² form; hinge off = REG3 through 2400; end to end | rows ok (PIDX 0.051 ctxpool, 0.039 allpool) | MATCH | |
| E5 | "The harness: each queued payload carries its arm's iters, and all 136 records agree. This is the fix …" | `explore_g_common.py:156-159` assert + print and l.170 payload (since `4314e35c`); log 16/16, 120/120, 36/36, 0/0; `iters_check` 136/136 | PARTIAL | This is a run-time assert and a report line, not one of the 147 rows. All arms ask for 24000, which is also the child's default (`rc.ITERS`, child l.158), so the record agreement cannot tell the fix from the defect. The payload assert can. |
| E6 | S89's CHECK set, its blob row superseded | as A5 | MATCH | |
| E7 | readings() hand cases; counterparts 0/20 | rows ok | MATCH | |
| E8 | seeds 2992-3019 in no record on any of the 13 branch heads | V now: 13 heads, no other store holds such a seed | PARTIAL | True now. No CHECK row or log line records a pre-run check (as `check_G_report10` I9). |

### 1e. "For the main line", README, headline (chk §2, §5, §6)

| # | statement | recomputed | verdict | note |
|---|---|---|---|---|
| F1 | "does not, by itself, make it route by type at 24000, on either memory: 0/10 …; both perfect gates bind 2/2 and a type-keyed perfect gate routes every sequence" | as B, C, D2 | MATCH | The accurate statement (§4). |
| F2 | "The closest arm is the Hebbian key hinge … (type map 0.97) … separated 0.20, held-out 0.89, all runs STREAM-PARTIAL" | Best arm by median: yes. The closest single runs are ctxpool.REG3_REPEATHINGE_H\|3003 (margin 0.838, separated 0.875, type map 1.00) and allpool.REG3_KEYHINGE_H\|3003 (0.837, 0.867, bound) | PARTIAL | Arm-level, correct. "All STREAM-PARTIAL" as D4. |
| F3 | the next question is how the gate is pushed to use its channels, not whether the channels exist | consistent with PTYPE 2/2 and 0/10 | MATCH | |
| F4 | "the register gate with the key hinge does not use them unaided on a two-document task with pooled contexts" | 0/10 per arm at 24000, SLOW, one seed block | PARTIAL | Needs the qualifier "at 24000 updates; 0/10 excludes a rate above 0.26". |
| G1 | README: setup; "Readings committed at b981c3f before any run. 136/136 runs ok (iters 24000 on 136/136)" | as A3-A7 | MATCH | |
| G2 | README table (2/2 / 2/2; 0/10 ×9) | as B | MATCH | |
| G3 | README KH cell: type map 0.97, ~6 channels, separated 0.20 | as D3 | MATCH | |
| G4 | README reading on both memories, with its rule | as C1, C2 | MATCH | |
| G5 | README: "folds the 16 types onto about 6 channels without committing (all STREAM-PARTIAL, held-out 0.89)" | as D4, D7 | PARTIAL | |
| G6 | README: the perfect gates coincide by construction | as D1 | MATCH | |
| H1 | `330290dd` headline: k = 16 on both tasks; every learned arm 0/10 on both memories while both perfect gates bind 2/2; "the channels do not suffice"; KH keys on context type but uses ~6 channels | numbers as B, C, D3 | MATCH | On the name, §4. On "uses ~6 channels", D7. |

**Totals: 55 rows.**
- MATCH 45.
- PARTIAL 10: D4, D6, D7, D9, D15, E5, E8, F2, F4, G5.
- DISAGREE 0.
- NOT RECORD-CHECKABLE 0.

## 2. Order (chk §0, §0c, §7)

| time (UTC) | event | S91b records |
|---|---|---|
| 17:23:21 | `4314e35c`: `explore_g_common.run_screen` sends `iters=screen.ARMS[arm]["iters"]` and asserts it (V's fix 8); `iters_check` added | – |
| **17:33:17** | **`b981c3fa`**: parent, child, vendored module. Readings (s91b:35-46, `readings()` l.140-171), ROUTED_k (l.91-93), validity rule, CHECKs | – |
| 17:33:21-17:42:36 | first CHECK pass (git b981c3f, clean): 146/147. S89's blob row fails. Assert, no run (log l.155-163) | – |
| 17:38:20 | `736686fb`: log only | – |
| **17:43:04** | **`9949e495`**: `run_checks()` relabels S89's blob row "(S89, superseded …)" with ok = True; docstring l.50-51 | – |
| 17:43:26-17:52:57 | second CHECK pass (git 9949e49+dirty): 147/147 | – |
| 17:53:28 | `1224d417`: store created | 0 |
| 17:58:32 | `949936bc`: first records (perfect gates) | 8 |
| 18:04:20 / 18:08:40 | `s91b_validity.json` written (all four cells valid, SHA `a0134e14a262`) / committed `fe065059` | 16 |
| 22:35:43 | `9660733a`, before the worker restart | 100 |
| 22:36:31-22:45:22 | third CHECK pass (git 9660733+dirty): 147/147 | 100 |
| 00:36:51 | `02e63bd1`: 136 records, summary, report() in the log | 136 |
| 00:50:50 | `330290dd`: report 11 and README only | 136 |

**What `9949e495` changed.**
- **Its target and timing.** It changed only the parent `explore_g_s91b.py`: `run_checks()` and one docstring bullet (chk §0 diff). It came after the first CHECK pass and before any run.
- **The reading functions.** By ast source, `rbk`, `type_routed`, `_sha`, `_valid`, `_arms`, `counterpart`, `describe`, `readings`, `check_readings`, `report` and `main` are identical at `b981c3fa` and `9949e495`. Only `run_checks` differs.
- **The code SHA.** The parent is not in `sha_rows`, as in S90 and S91. So the change cannot touch the SHA. The 36 rows each equal sha1 of the module's source at `9949e495`, and their hash is the meta SHA.
- **What the relabel does.** It does not test anything: it sets ok = True on any S89 row whose name starts with "explore_f_tasks.py is F's b0c6332 blob". The row it replaces asserted the old blob `e24d5e1` and could not pass once the module was `9017b82`.
  - S91b's own row (child l.194-195) asserts the stronger property: blob `cfe3fe27` and the `e24d5e1` byte prefix. It was already present and passing at `b981c3fa`.
  - `screen_review_s90_F` §3(c) asked G to replace the equality CHECK with a prefix CHECK. G added the prefix CHECK in S91b's child but left S89's own equality row in the CHECK set, and the first pass caught it.
- **Disclosure.** r11 l.10-12 discloses the change, and the log marks the relaunch.

**What could differ under "+dirty".** The two runs that produced records started on dirty trees (9949e49+dirty, 9660733+dirty). The code SHA covers every module the child imports, and it is constant on all 136 records, the meta and the validity file. The parent is not covered. Its committed version regenerates the log exactly (§1b C1), which is the best available evidence that the parent that ran was the committed one.

**Verdict on order.**
- Everything that decides a reading was fixed at `b981c3fa`, 25 minutes before the first record, and was not changed afterwards.
- The validity file was written with the SHA before any learned run.
- After the first record, only records, the log, the summary, the validity file, a CHECK file, r11 and the README changed.
- S91b's design is post-F-report-10 and post-S91: F's cap table and the user's instruction (s91b:7-11). No S91b record existed when it was fixed.
- There was no referee pre-run review.

## 3. The budget

- **The harness defect is fixed.** `explore_g_common.py` at `b981c3fa` (blob `e4c622e67a`, last changed `4314e35c`) builds every payload as `dict(arm=arm, seed=s, iters=screen.ARMS[arm]["iters"])`. It asserts this (l.156-157), prints it (l.158-159) and sends it (l.170).
  - The log prints the CHECK for each segment: 16/16, 120/120, 36/36 and 0/0.
  - `report()` prints `iters_check` 136/136.
- **What the records show.** Every arm asks for 24000 (s91b:110, 117), and every record has iters 24000. Every unbound run stops at 24000 with its last evaluation at 24000.
- **What the records cannot show.** Here the intended budget equals the child's default (`p.get("iters", rc.ITERS)`, child l.158, `ITERS = 24000`). The records alone therefore cannot distinguish the fix from the old defect. The payload assert can, and it passed.
- **No CHECK row covers the budget.** None of the 147 rows of any CHECK file mentions iters. The check lives in `run_screen` and `report()`.

## 4. "The channels do not suffice"

**The committed rule** (b981c3fa, s91b:40-41 and `readings()` l.140-171), per memory:
- READING = "the channels do not suffice" if:
  - no learned arm is "routes" (ROUTED_k ≥ 7/10 on -ctxpool and one-sided Fisher p < 0.05 against its k = 2 counterpart's 0/20), and
  - every learned arm has ROUTED_k ≤ 3/10 on -ctxpool (l.167), and
  - `nb == (2, 2)`: PIDX and PTYPE each BOUND 2/2 on -ctxpool (l.145, l.167; BOUND, not ROUTED_k).
- UNTESTED comes first: if PIDX < 2/2 or a learned seed is missing (l.147-153).
- ROUTED_k (`rbk`, l.91-93) is BOUND and end margin ≥ 0.9 and end `seq_separated_frac` > 0.9. Both statistics are within-sequence and invariant to relabelling the channels.

**The inputs.**

| memory | PIDX / PTYPE BOUND (-ctxpool) | REG3 | REG3_KEYHINGE | REG3_REPEATHINGE | reading |
|---|---|---|---|---|---|
| Hebbian + conv | 2/2, 2/2 | 0/10 | 0/10 | 0/10 | the channels do not suffice |
| delta | 2/2, 2/2 | 0/10 | 0/10 | 0/10 | the channels do not suffice |

G's `readings()` and V's independent version (rlib's BOUND from the curves) give the same result (chk §4). **The name is the rule's mechanical output.** It was fixed before the runs and is not a post-hoc label.

**What the name claims, and what the screen shows.**
- **The channels themselves suffice.** At k = 16 the fixed type → channel cap is 1 (F's table; `type_map_cap(16, 2, 16)` = 1, CHECKed).
  - PTYPE, a gate that puts each type on its own channel, binds 2/2 on all four cells. It reads margin 1.00, separated 1.00 and type map 1.00, and trains bit for bit like PIDX.
  - So the memory and the k = 16 layout can bind the task when handed a type-level routing. In the capacity sense, the channels do suffice.
- **What failed is learning.** The 0/10s show that no learned gate found a separating routing within 24000 SLOW updates under these three objectives:
  - 0 of 120 learned runs reach margin ≥ 0.9;
  - 0 of 120 reach separated > 0.9.
  - The closest are ctxpool.REG3_REPEATHINGE_H\|3003 (0.838 / 0.875), allpool.REG3_KEYHINGE_H\|3003 (0.837 / 0.867, bound at the last evaluation) and ctxpool.REG3_KEYHINGE_H\|3002 (0.817 / 0.836).
- **Did a learned arm reach a type-level map? In part.**
  - 19 of 120 learned runs have type_map_k > 0.9: ctxpool KH_H 10/10, allpool KH_H 5, ctxpool RH_H 2, allpool RH_H 1, ctxpool RH_D 1.
  - But type_map_k is η² of the KEY gate by context type, pooled. It says the gate is a function of the type, not that two types get distinct confident channels.
  - None of those 19 separates > 0.9 of sequences. On -ctxpool, the Hebbian key hinge's median is 0.20 separated and 0.67 margin.
  - So it keys on type (supported), but it does not route. A few runs come within 0.03-0.08 of both lines, and on several runs separation was still rising at 24000 (D16).
- **The rule fires on a rate up to about 0.3.**
  - Per arm, 0/10 excludes a true ROUTED_k rate above 0.259 (one-sided 95% Clopper-Pearson). The two-sided 95% bound is 0.309, and Wilson's is 0.28.
  - The rule's threshold is ≤ 3/10. At a true rate of 0.3 it reads "do not suffice" with probability 0.65, and at 0.4 with probability 0.38 (chk §6).
  - Pooling the three arms (0/30) is not one rate.
  - All this is at 24000 updates, SLOW, one seed block of 10, and two-document -ctxpool.

**Verdict on the name.**
- The label is mechanical and correct as the rule's name.
- As English, "the channels do not suffice" reads as a statement about the channels: that k = 16 is not enough to route. The screen refutes that reading through PTYPE.
- What the records support is "having the channels does not suffice for these gates to learn the routing at this budget".
- r11's body says exactly this: "does not, by itself, make it route by type", "The channel cap is not what limits these arms", "not whether the channels exist". The headline, the README line and the readings table carry only the bare label.

**Recommended wording** (README and r11 readings table; the rule's string can stay as committed, with a gloss):
> "the channels do not suffice" (the pre-fixed name; read: giving the register gate a channel per context type does not, by itself, let it learn to route at 24000 updates). It is a learning negative, not a capacity limit: a type-keyed perfect gate at the same k = 16 binds 2/2 and separates every sequence. Each learned arm is 0/10 on -ctxpool, which excludes a ROUTED_k rate above 0.26 per arm (one-sided 95%). The Hebbian key hinge's gate becomes a function of the context type (type map > 0.9 on 10/10) but separates a median 0.20 of sequences (best 0.84), and some runs were still rising at 24000.

For the headline, instead of "uses ~6 channels":
> "keys on context type but does not commit (separated 0.20 median; its overall mean gate has ~6 effective channels)".

## 5. What could not be checked

- **The training itself.** Nothing was rerun. CHECK rows were read from the committed JSON, not re-executed.
- **The relabelled CHECK.** That S89's other 61 CHECK rows still pass under the new module was not re-executed. They are recorded ok in all three files.
- **How many channels the key hinge's type map uses.** The records carry `eff_ch` (the overall mean gate) and the pooled L1-L3 stream maps, not the per-type argmax channel. "6 channels" can be neither confirmed nor refuted (D7).
- **Why delta matches channel-permuted gates bit for bit** (PIDX vs PTYPE, D1). Observed on 4/4 delta pairs. The mechanism is not checked.
- **The "+dirty" trees** of the two record-producing launches. The covered modules are pinned by the SHA. The parent is pinned only by the exact log reproduction.
- **That the seed-collision check ran before the runs** (E8).

## 6. Errors in the referee's own files

- **`screen_review_s90_F.md` §3 (scope, not error).** It says that at S = 2 "G's within-sequence ROUTED-BOUND (`pool_outcome(rec, True)`, tasks l.725-737) applies unchanged". That holds at k = 2, which was G's gate when it was written. It does not carry to k = 16: the within-sequence η² there reads channel 0 only, and F's `pool_outcome(rec, True)` gives PTYPE 0/2 on every cell (chk §2). G saw this and used `seq_separated_frac` (s91b:27-30), as that review's §3(d) and §1.4 recommend. A reader of §3 alone could take the wrong form.
- **`check_G_report10.md` §4 and §3 fix 8.** At `cbca3e36` it says "Fix 8 was not applied". That was true then. G applied it at `4314e35c` (17:23:21), before S91b. Not an error, but the status has changed.
- No other error was found in `check_G_report10.md`, `screen_review_s90_F.md` or `check_F_report10.md` against these records:
  - The caps used here (0.533 at k = 2, 1 at k = 16) equal F's `type_map_cap`.
  - `check_F_report10`'s -ctxpool single delta 10/10 is the figure r11 cites.
- **The task brief's figure.** "Clopper-Pearson upper bound 0.26" is the one-sided 95% bound (0.259). The two-sided 95% bound is 0.309.
