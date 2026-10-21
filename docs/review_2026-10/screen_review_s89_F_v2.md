# Screen review v2: Session F's S89-F revision and follow-up pilots (claude/explore-F `b0c6332`..`5357889`)

*Referee V, 20 October 2026. This follows up `screen_review_s89_F.md` (on review-V at `3ae815b`, written against `8ba99bb`) and `check_F_report9.md`.*

*What it reviews:*
- *`b0c6332` (21:47:22): the appended `explore_f_tasks.py`.*
- *`3dd5b74` (21:47:35): the S89 revision and the follow-up screen `explore_f_s89_post` / `_child`, whose readings were fixed in this commit.*
- *`4c8a343` (21:48:12): the queue script.*
- *The progress commits `b036d91`, `530240c`, `e929827` and `209a271`, and the records at `73035d3` (22:02:28).*
- *`2279ca1` (22:04:19): the report_9 addendum, the README section and a 6-line change to `explore_f_s89_tasks.py`.*
- *`5357889` (22:04:32): the wording correction.*

*Line references:*
- *"tasks l.N" is `explore_f_tasks.py` at `b0c6332`.*
- *"s89 l.N", "child l.N", "rep6 l.N", "post l.N" and "pchild l.N" are `explore_f_s89_tasks.py`, `explore_f_s89_tasks_child.py`, `explore_f_reports6.py`, `explore_f_s89_post.py` and `explore_f_s89_post_child.py`, all at `3dd5b74`.*
- *"r9 l.N" is `explore_out/F/report_9.md` at `5357889`, and "README l.N" is `explore_out/README.md` at `5357889`.*

*"out §X" is a section of `screen_checks_s89_F_v2_out.txt`, written by `screen_checks_s89_F_v2.py`. That script:*
- *reads git objects only and trains nothing;*
- *execs F's `pool_outcome` (tasks l.725–737) and `reading` (post l.106–111), extracted by ast;*
- *replicates `eta2_rows`, `inv_from_gate`, the pooled `pool_routing_stats` / `role_cosines` and the generators line by line in numpy, because no torch is installed here.*

## Verdict

**Fixes 1–3 are resolved, and the module is fit for G's use with two caveats to print. The follow-up's order is clean and its reading is mechanical. The report and README need small corrections. Nothing blocks G.**

**The module change.**
- `b0c6332`'s first 35,605 bytes are `50dc8e4`'s file byte for byte. The append is 14,318 bytes (226 lines). It redefines no earlier name and rebinds no earlier constant (out §A).
- G vendored this blob (`e24d5e1`) at `db14e9b` (22:06:25). H does not vendor `explore_f_tasks.py` at all: its task is the grouped layout (`explore_h_s89.py` l.11).

**The swap-invariant statistics do what fix 1 asked** (out §B, 20,000 replica sequences per task).
- On ALLPOOL, a gate that separates every sequence but takes its channels from the context type scores within-sequence η² 1.000, cos 0.000 and margin 1.000, so it reads ROUTED. The pooled forms give 0.0000 / 1.0000.
- None of the seven source-blind gates reads ROUTED or DISCOVERED in either form, on either task. The highest within-sequence η² is 0.448 (position half).
- On POOL, both forms are identical for every gate that is a fixed function of the context type.

**Two properties to print.**
- **(i) DISCOVERED in the within-sequence form is passed by a gate at the cap.** On ALLPOOL the per-sequence VAL cosine is 0 in a separated sequence and 1 otherwise, so `val_cos_inv` = 1 − (separated fraction). The balanced type-level gate, at the 0.533 cap, reads 0.464 < 0.5, so it would read DISCOVERED if it bound. F's own O_TYPE_HEBB_ALLPOOL|2800 has `val_cos_inv` 0.488.
- **(ii) On POOL, the pooled form also requires a fixed context → channel map**, as the reading uses it (rep6 l.92–94). A gate that routes every sequence with per-sequence channels reads ROUTED on ALLPOOL and not on POOL.

**The follow-up pilots.**
- The hypothesis and readings were committed at 21:47:35. The queue started at 21:47:47 with CHECKs only. The first run started after the timings at 21:50:29, and the first record was committed at 21:52:53.
- The 27-module code SHA, recomputed from blobs, is `267d08456c38` at every commit from `3dd5b74` to `5357889`, the same as all 6 records. Every code file has one blob over the range, except the S89 parent (below).
- "partly" is the rule's mechanical output for 3/4.
- Every count, transition, Wilson interval and band matches the records.

**Non-MATCH rows: 2 DISAGREE, 7 PARTIAL out of 37** (§3.2).
- The two DISAGREEs are one rounding: the type-level 2801 final is 0.2749, printed "0.28" in both r9 and the README.
- The PARTIALs:
  - two prose ranges;
  - "bytes" for characters;
  - "G and H vendor";
  - "fixes 4–10 concern phase 2";
  - the causal sentence on 3 discordant seeds (sign test p = 0.125, across two CPUs), in r9 and the README.

**Fixes 4, 5, 7 and 10 are not taken, and 6 and 8 only partly.** Fixes 5 and 10 are not moot: phase 0 and the validity rule ran.

## 1. The ten fixes of `screen_review_s89_F.md`, item by item

| # | fix | status | where | note |
|---|---|---|---|---|
| 1 | ALLPOOL statistics invariant to a per-sequence channel swap | **RESOLVED** (residuals R1–R3) | tasks l.613–640 (comment), l.663–737; s89 l.86–96 (the text proposed, near verbatim); child l.64 (`STATS_POOL = pool_stats_inv`), l.179–181 (`outcome_pooled` / `outcome_inv`); rep6 l.22–24, l.92–94; CHECK in the log (`s89post.log` at 73035d3, 15 `check_inv_stats` lines, all ok) | **R1:** s89 l.93 says POOL's forms are "recorded and printed", but `report_s89` prints only the pooled form on POOL (rep6 l.92–94). **R2:** `check_inv_stats` (tasks l.765–836) has no negative control, i.e. no source-blind gate that must *not* score routed (§2.3). **R3:** DISCOVERED in the within-sequence form at the cap (§2.4) |
| 2 | ALLPOOL's routing cap: (a) state it and run the type-level gate as a fifth pilot | **RESOLVED** (option a) | s89 l.39–46 (the proposed text); tasks l.641–650, l.740–750; post l.19–25; records `O_TYPE_HEBB_ALLPOOL\|2800/2801` | The pilot ran on Hebbian, where the relative gate is itself unbound. F said beforehand (post l.22–25) that it "cannot separate the cap from the memory". The cap is exact: max a(16 − a)/120 = 64/120 (out §B) |
| 3 | Tie the screen's own path to a record; pin `G_REF`; S70 from a commit; "first 1, 10, 100 batches" | **RESOLVED** | child l.65–69 (`G_REF = G_SHA = e46799a`, `S70_REF = 61ba46e`); l.274–291 (own path, curve + gradient norms + every statistic at 0/1200/2400); l.295; s89 l.103–108; log: "19 fields … equal True: ok" | Extra CHECK: the revised path reproduces `O_HEBB_POOL\|2800` and `O_HEBB_ALLPOOL\|2800` through 2400 (child l.303–319; ok). The 19 fields include `step` and `k`, so 17 are statistics |
| 4 | The baselines' role ("unpaired descriptive rates") | **NOT** | s89 l.6 "records the baselines G and H will pair against on their own seeds"; l.79–80 "the references G and H pair against on their own seeds" | Moot in effect, since S89's queue is closed (s89 l.10–11). The sentence is still false |
| 5 | Phase 0's meaning (the Bayes bound 0.19 / 0.42; the rule could not fire) | **NOT** | s89 l.51–64 unchanged; "POSITIONAL SOLVABILITY" kept | **Not moot**: phase 0 ran and report 9 reads it. r9 l.256 lists fix 5 among those that "concern phase 2", which is wrong |
| 6 | Name the failed-record case | **PARTLY** | taken in the new screen: post l.40–41, `reading()` l.106–111, rep6 `report_s89_post` (counts failures after `MAX_ATTEMPTS`); not in S89 (s89 l.73–77; `decide` unchanged) | Exec'd over all 15 complete cases plus one incomplete case: every case gets one name (out §C) |
| 7 | Assert that the learned arms' gates read no labels (`ctx_tokens` inert) | **NOT** | child "models" part unchanged | Moot for F, since no learned arm ran. For G it is G's own CHECK |
| 8 | `fail_class_v2`; NaN `key_sep` / `key_part` on the pool tasks; `routed_end` None guard | **PARTLY** | Guard resolved: rep6 l.22–24 delegates to `pool_outcome`, whose `mg is not None and eta is not None` (tasks l.734) returns False on a missing field (out §C). v2: s89 l.97 still promises it; child l.176 computes v1 only. `key_sep` / `key_part` are still stored under lite's ids | The new records show it: `O_TYPE_HEBB_ALLPOOL` end `key_part` 1.0, `key_sep` 0.0 (out §D), read on context types 2–5 |
| 9 | (For G and H, the prompt) the repeat statistic selects keys + contexts + colliding values | n/a to F | — | Not F's to fix |
| 10 | Print the readings' operating characteristics | **NOT** | s89 l.73–77 (P(valid) = p⁴ not printed); post l.36–39 (no characteristic for the new 4-run rule) | The new rule's characteristic (out §C): at p = 0.75 "partly" is the modal outcome (0.42). At p = 0.95 it still occurs 17% of the time |
| §7.6 | Disclose the dirty start | **PARTLY** | r9 l.275–276 discloses the follow-up's (`3dd5b74+dirty`; `4c8a343` adds only `keepalive.sh`, `run_s89post.sh` and `s89post.log`) | S89's own dirty start (`8ba99bb+dirty`) is still unmentioned |

## 2. The module change (`b0c6332`)

### 2.1 Append-only

| check | result (out §A) |
|---|---|
| `50dc8e4` file | 35,605 bytes = 34,603 characters, blob `ffe249bb64` |
| `b0c6332` file | 49,923 bytes = 48,457 characters, blob `e24d5e1081` |
| `b0c6332[:35605] == 50dc8e4` | **True** |
| appended | 14,318 bytes = 13,854 characters, 226 lines |
| appended top-level names | `eta2_rows`, `_positions`, `inv_from_gate`, `pool_routing_stats_inv`, `pool_stats_inv`, `pool_outcome`, `type_level_gate`, `_GateStub`, `check_inv_stats`, `BALANCED_MAP` (plus `ROUTED_MARGIN`, `ROUTED_ETA`) |
| redefinitions of a 50dc8e4 name | none |
| blob at 3dd5b74 … 5357889 | `e24d5e1081` at every commit |
| G's tip | `e24d5e1081`, vendored at `db14e9b` 22:06:25 |
| H's tip | no `explore_f_tasks.py` (H's grouped layout has its own pool) |

F's CHECK (child l.236–238) tests `new.startswith(v1)` on decoded text, which is correct. But its label prints `len(str)` as "bytes": "first 34603 bytes", "13854 bytes". r9 l.260 copies that label, as report 9 did with "13771".

### 2.2 The synthetic gates (out §B; numpy replica, 20,000 sequences per task)

ROUTED = margin ≥ 0.9 and η² > 0.9. "DISC" = VAL cos < 0.5 (if bound).

| gate | task | margin | η² pooled | η² within | cos pooled | cos within | ROUTED pooled / within | DISC pooled / within |
|---|---|---|---|---|---|---|---|---|
| relative (`task.perfect_gate`) | POOL | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | yes / yes | yes / yes |
| relative | ALLPOOL | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | yes / yes | yes / yes |
| relative, swapped in a random half | POOL | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | **no / yes** | **no / yes** |
| relative, swapped in a random half | ALLPOOL | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | **no / yes** | **no / yes** |
| **type-ordered** (lower CTX id → ch 0) | POOL | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | yes / yes | yes / yes |
| **type-ordered** | ALLPOOL | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | **no / yes** | **no / yes** |
| balanced type-level map | POOL (= the fixed map) | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | yes / yes | yes / yes |
| balanced type-level map | ALLPOOL | 0.536 | 0.000 | 0.536 | 1.000 | **0.464** | no / no | no / **yes** |
| BLIND uniform 0.5 | both | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | no / no | no / no |
| BLIND position half (t < 10 → ch 0) | ALLPOOL | 0.004 | 0.168 | 0.448 | 0.713 | 0.521 | no / no | no / no |
| BLIND block parity (# CTX seen mod 2) | ALLPOOL | −0.004 | 0.108 | 0.387 | 0.806 | 0.579 | no / no | no / no |
| BLIND latest key-type parity | both | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | no / no | no / no |
| BLIND one channel per sequence | both | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | no / no | no / no |
| BLIND iid random one-hot | ALLPOOL | 0.003 | 0.000 | 0.141 | 1.000 | 0.829 | no / no | no / no |
| BLIND best position-only soft gate | ALLPOOL | 0.000 | 0.253 | 0.349 | 0.872 | 0.858 | no / no | no / no |
| relative in 85 / 90 / 95% of sequences, uniform elsewhere | ALLPOOL | 0.851 / 0.898 / 0.950 | = margin | = margin | 0.160 / 0.107 / 0.052 | 0.149 / 0.102 / 0.050 | no, no, yes in both forms | yes in both |
| relative 0.8/0.2 + a per-sequence offset U(−0.25, 0.25) | POOL | 0.354 | 0.816 | 1.000 | 0.477 | 0.519 | no / no | **yes / no** |
| relative 0.8/0.2 + iid N(0, 0.10) | POOL | 0.357 | 0.903 | 0.924 | 0.474 | 0.476 | no / no | yes / yes |

**What the tests ask, answered.**
- **A gate that routes every sequence perfectly but labels channels by context type scores as routed.** The within-sequence form gives η² 1.000 and cos 0.000 on both tasks, so ROUTED is read on ALLPOOL. The pooled form reads 0.000 and 1.000 there.
- **A source-blind gate does not.** The highest within-sequence η² of the seven blind gates (on either task) is 0.448, against the 0.9 threshold. Their margins lie between −0.004 and 0.011.
  - Under the null, the per-sequence η² on 8 points has mean 0.1429 (theory 1/7).
  - P(η²_seq > 0.9) = 0.0011, so `seq_routed_frac` is near 0 for any blind gate.
- **On POOL the two forms are equal where the labels are consistent.** Every gate that is a fixed function of the context type gives identical numbers, to four decimals, in both forms. This includes the soft map 0.8 / 0.2: η² 1.000 / 1.000 and cos 0.4706 / 0.4706.
- **For a gate that is not a deterministic function of the context, the forms differ**, as s89 l.93's "they agree for a fixed CTX → channel map" implies.
  - Per-position noise reads 0.02–0.06 higher within-sequence. This is small-sample bias.
  - A per-sequence offset reads η² 1.000 within-sequence against 0.816 pooled. It also flips DISC: cos 0.519 within against 0.477 pooled.
  - In ROUTED the margin binds first. A one-hot gate with a |N(0, 0.05)| leak has margin 0.846 with η² 0.996 / 0.997. So the η² bias did not flip ROUTED for any gate tried.
- **The type-level gate's statistics.**
  - Within-sequence η², margin and `seq_routed_frac` each equal the separated fraction exactly, and `val_cos_inv` = 1 − that fraction.
  - The records confirm this exactly: 131/256 for 2800 and 126/256 for 2801 (out §D).
  - The replica gives a separated fraction of 0.5356 against the cap of 0.5333. F's probe values sit at z = −0.69 and −1.32.

### 2.3 Residual: the CHECK has no negative control

`check_inv_stats` asserts positives only:
- the relative, swapped and type-ordered gates must score 1 / 0;
- the pooled forms must read the labelling;
- the type-level gate must equal its separated fraction;
- a random gate is used only to test the vectorisation (tasks l.828–835).

Nothing asserts that a gate blind to the source scores *below* 0.9. The statistic passes such a test here (§2.2), but the CHECK should carry it.

### 2.4 Residual: DISCOVERED in the within-sequence form at the cap

Under the within-sequence form, DISCOVERED reads `val_cos_inv` < 0.5 (tasks l.732, l.736). That is the mean over sequences of a cosine which is 0 where the gate separates the streams and 1 where it puts both on one channel. So DISCOVERED holds as soon as more than half the sequences are separated.
- The balanced type-level gate separates 0.533 in expectation, so a token-identity gate that has learned a fixed map reads DISCOVERED (if bound) while routing about half of ALLPOOL.
- In the pooled form on lite or POOL, the comparable gate (relative in a fraction f of sequences, uniform elsewhere) needs f > 0.577.
- ROUTED*@end (η² > 0.9) is unaffected.
- G reads ROUTED-BOUND and only prints DISCOVERED (`explore_g_s89.py` l.56–63), and F's phase 2 is closed. So no reading is affected yet, but the printed column misleads on ALLPOOL.

## 3. The follow-up pilots (`explore_f_s89_post`)

### 3.1 Order and code

| when (UTC, 20 Oct) | what | evidence |
|---|---|---|
| 21:47:22 | `b0c6332`: module append | git |
| **21:47:35** | **`3dd5b74`: hypothesis, arms, readings (post l.19–46, `reading` l.106–111), child, revised S89 CHECKs** | git. `O_HEBB_CONV` first appears in any commit here (`git log --all -S`) |
| 21:47:47 | queue start; CHECKs (delta 26 ok; S89 52 ok; post 6 ok; 0 FAIL) | s89post.log; `checks_s89post_*.json` started 21:47:51, git `3dd5b74+dirty` |
| 21:48:12 | `4c8a343`: `run_s89post.sh`, `keepalive.sh`, the log; no code | git |
| 21:50:29 | timings measured, then the 6 runs start | `s89post_runtime.json` |
| 21:52:53 | first record (`O_HEBB_CONV_POOL\|2800`) committed (`530240c`) | git |
| 21:55:21 / 21:57:02 / 22:02:28 | 3, 5, 6 records | git |
| 22:04:19 | `2279ca1`: r9 addendum, README, and `explore_f_s89_tasks.py` `pin()` gains `RECORDED_CPU` (l.188, l.191–196) | git |
| 22:04:32 | `5357889`: wording correction | git |

**The code.**
- The 27-module code SHA recomputed from blobs is `267d08456c382675…` at `3dd5b74`, `b036d91`, `530240c`, `e929827`, `209a271`, `73035d3` and `5357889`. All 6 records carry it.
- Every code file has a single blob over `3dd5b74..5357889` except `explore_f_s89_tasks.py` (out §E).
- **The one change after records:** `explore_f_s89_tasks.py` at `2279ca1`.
  - It is the S89 *parent*, which is not among the 27 modules.
  - The change lets `pin()` find S89's 2.80 GHz records from a 2.10 GHz container.
  - It touches no follow-up record or reading, and S89's decision file is never revised.
  - It is a reporting fix after records and should be named in the addendum.

**Hand runs before the commit.** r9 l.273–274 says each CHECK part was run by hand before the commits.
- The follow-up child's CHECK trains each conv arm for 2401 updates on seed 2800 with `tbo.evaluate` and `tbo.logits_of` stubbed (pchild l.88–108), so it produces no accuracy.
- The S89 "inv" part trains only the no-conv pilots.
- No information about the conv arms' binding was available before the readings were fixed.

**Seeds.**
- 2800–2801 are the S89 pilots' seeds, reused deliberately so that each arm shares its pilot's batches and initial parameters (CHECKed).
- Across all 13 remote tips (json, jsonl, log), the block 2800–2839 occurs only in `explore_out/F/` (6 files on explore-F, 0 elsewhere; out §F). **Seeds free.**

**The CHECKs tie the code to recorded runs.**
- S89's path reproduces three recorded runs on this CPU through 2400:
  - G's `BASELINE|2700` at `e46799a`;
  - `O_HEBB_POOL|2800` and `O_HEBB_ALLPOOL|2800` at `52e2ea7`.
- Each conv arm equals its pilot bit for bit at initialisation: logits, shared parameters, and one extra parameter, `short_conv.conv_w` (pchild l.130–141).
- **The conv forward path (`forward_conv`) is not tied to any recorded conv run.** Equality at the identity initialisation does not exercise lags 1–3. H's `ORACLE|2900` (conv, recorded) or the main line's `ceiling_conv` through 2400 would tie it. Minor, since `test_short_conv`'s own CHECKs 53–55 cover the conv.

### 3.2 Every number in the addendum and the README against the records

The records are `s89_post_results.json` at `73035d3`. The S89 pilots are read at `52e2ea7`, and H's oracle at `1b5dfed` (H's last record commit before 21:47:35).

| # | statement | recomputed (out §D) | verdict |
|---|---|---|---|
| 1 | r9 l.183–185: on 2.10 GHz; two 2.80 GHz pilots reproduce through 2400 | cpu {2.10 GHz} on all 6; log CHECK lines ok | MATCH |
| 2 | r9 l.191: CONV_POOL 2/2 [0.34, 1.00] RELIABLE, @4800, @13200; S89 0.62 / 0.48 | transitions 4800 / 13200 (record = `rlib.transition`); Wilson, band as stated; S89 0.6226 / 0.4824 | MATCH |
| 3 | r9 l.192: CONV_ALLPOOL 1/2 [0.09, 0.91] MAJORITY, @6000, unbound 0.87; S89 0.33 / 0.50 | 6000; 0.8721; 0.3306 / 0.4976 | MATCH |
| 4 | r9 l.193: 3/4 [0.30, 0.95] MAJORITY; 0/4 | 3/4 [0.301, 0.954] MAJORITY; S89 Hebbian 24K 0/4 | MATCH |
| 5 | r9 l.194: TYPE 0/2 [0.00, 0.66] NEVER; "unbound 0.28 \| unbound 0.28" | 0.2783 / **0.2749**, so 2801 is 0.27 | **DISAGREE** (rounding) |
| 6 | r9 l.196–197: "partly"; readings at 3dd5b74 before the runs | `reading(3, 0, 4)` = "partly"; order §3.1 | MATCH |
| 7 | r9 l.200: every conv run margin 1.00, η² 1.00 both forms, VAL cos 0.00 both | all 4: 1.0 / 1.0 / 1.0 / 0.0 / 0.0 | MATCH |
| 8 | r9 l.201: bound runs reach 0.96–0.99; 0.32–0.72 without the conv at 24000 and 48000 | 0.9834, 0.9624, 0.9878; S89 Hebbian finals 0.3193–0.7173 | MATCH |
| 9 | r9 l.202–203: ALLPOOL 2801 "rises 0.67 → 0.78 by 7200, holds about 0.77 to 20400, then climbs to 0.87" | 0.669 at 4800, **0.727 at 7200**; 0.78 first at 16800 (0.782); 0.760–0.795 over 9600–20400; 0.853, 0.861, 0.872 | **PARTIAL** |
| 10 | r9 l.204: last-write / not 0.86 / 0.87 | 0.8592 / 0.8730 | MATCH |
| 11 | r9 l.205–207: the conv is the only difference; no-op at init (CHECKed) | pchild l.130–141; log ok ×2 | MATCH (the CPU also differs, disclosed at r9 l.183–185) |
| 12 | r9 l.208–209: the conv "turns a Hebbian channel that cannot use 4 keys out of 32 into one that binds on 3 of 4 seeds" | 3 discordant (task, seed) pairs, 0 the other way: exact sign test p = 0.125; Fisher 3/4 vs 0/4 p = 0.071; two CPUs | **PARTIAL** (causal wording on a post-hoc 3-vs-0 that the rule itself reads only "partly") |
| 13 | r9 l.210: H's Hebbian perfect gate with conv 2/2 on the grouped layout, pool 32 | `ORACLE\|2900` 16800, `\|2901` 10800; 43200, "one Adam 1e-3", k 16, pool 32, conv taps recorded | MATCH |
| 14 | r9 l.211: delta binds without the conv, 2400–3600 | 3600, 2400, 3600, 3600 | MATCH |
| 15 | r9 l.212–214: the conv candidate account | consistent with the conv's definition (`test_short_conv` l.43–52) | NOT RECORD-CHECKABLE |
| 16 | r9 l.217: TYPE "stay at 0.25–0.28 for all 24000 updates" | all evaluations 0.2207–0.2808 and 0.2168–0.2910; from 2400 on, 0.2397–0.2808 and 0.2588–0.2910 | **PARTIAL** |
| 17 | r9 l.219–221: η² 0.51 / 0.49 = separated fraction; margin 0.51 / 0.49; VAL cos 0.49 / 0.51 | 0.5117 / 0.4922 (= 131/256, 126/256; margin = η²_inv = `seq_routed_frac` = 1 − cos_inv exactly); 0.4883 / 0.5078 | MATCH |
| 18 | r9 l.222: pooled η² 0.00 / 0.01, VAL cos 1.00 / 0.99 | 0.0001 / 0.0061; 0.9997 / 0.9879 | MATCH |
| 19 | r9 l.223–224: cannot separate the cap from the memory; stated before the run | post l.22–25 at 3dd5b74 | MATCH |
| 20 | r9 l.233–239: synthetic-gate CHECK (0.0000 / 0.0015; 1.000 / 0.997; 0.5117 on 256; max \|diff\| 0) | log lines, all ok | MATCH |
| 21 | r9 l.240: the S89 pilots' two forms coincide (1.00 / 0.00 at 2400) | log "added at 2400: η² 1.0, cos 0.0" ×2 | MATCH |
| 22 | r9 l.242: cap 64/120 = 0.533 | max a(16 − a)/120 = 0.5333 | MATCH |
| 23 | r9 l.247–250: own path, curve, gradient norms, "all 19 statistic fields" | log: "19 fields … equal True" (17 statistics + `step`, `k`) | MATCH |
| 24 | r9 l.251–253: S70 at 61ba46e; two S89 pilots reproduced | child l.67, l.295; log ok | MATCH |
| 25 | r9 l.256–257: "fixes 4–10 concern phase 2 … or G's and H's arms" | fix 5 concerns phase 0 (ran); fix 6 every phase; fix 10 the phase-1 validity rule (ran) | **PARTIAL** |
| 26 | r9 l.259–260: "The module for G and H … its first 34603 bytes are 50dc8e4's file" | 35,605 **bytes** (34,603 characters); H does not vendor it | **PARTIAL** |
| 27 | r9 l.266–272: CHECKs 26/26; S89 52; child 6; all passed | 26 ok; 52; 6; 0 FAIL | MATCH |
| 28 | r9 l.273–274: CHECK parts run by hand before the commits | — | NOT RECORD-CHECKABLE (no information leak possible: §3.1) |
| 29 | r9 l.275–276: dirty tree = queue script + keepalive | `4c8a343` adds only those and the log; SHA at 3dd5b74 = records | MATCH |
| 30 | r9 l.277: 0.50 h of run time, 21:47–22:02, SHA 267d08456c38 | Σ `secs_wall` 1,783 s = 0.495 h; log 21:47:47–22:02:31 | MATCH |
| 31 | r9 l.282–288: what this leaves | as rows 2–4 | MATCH |
| M1 | README l.659: conv row | as rows 2–6 | MATCH |
| M2 | README l.660: "0/2 (0.28, 0.28)"; η² 0.51 / 0.49; pooled 0.00 / 0.01 | 2801 final 0.2749 | **DISAGREE** (rounding) |
| M3 | README l.663–666: fixes 1–3 taken | §1 | MATCH |
| M4 | README l.667: "G and H vendor explore_f_tasks.py at b0c6332" | G yes (`db14e9b`); H no (`explore_h_s89.py` l.11) | **PARTIAL** |
| M5 | README l.668–670: "lets the Hebbian channel use pooled keys on 3 of 4 seeds … the only difference" | as row 12 | **PARTIAL** (wording) |
| M6 | README l.671–673 | as rows 17–19 | MATCH |

**Totals: 37 rows. 26 MATCH, 7 PARTIAL, 2 DISAGREE, 2 NOT RECORD-CHECKABLE.**

The tables file `s89_post_tables.md` at `73035d3` is generated by `report_s89_post` and agrees with the records. It prints three decimals: 0.278 / 0.275.

The hypothesis (post l.30–35) cites Revision 10 §11's "0/60 without it". That figure is there: `multichannel_hebbian_report_v10.tex` l.256, in §11 "The memory rule". But the same sentence goes on, "and a minority with it at a higher learning rate (18/40, 10/20)", and Revision 7 has one channel with the conv at 10⁻³ binding 7/80. The quote is selective, not false. It concerns one channel, whereas the pilots use the perfect gate.

### 3.3 The reading "partly" against the rule

- **The rule.** The rule (post l.36–41, `reading` l.106–111) maps 4/4 to "restores", 3/4 to "partly" and ≤ 2 to "it does not". A run failed after `MAX_ATTEMPTS` counts as not bound. Fewer than 4 resolved runs reads "INCOMPLETE".
- **Exhaustive check.** Exec'd on all 15 complete (bound, failed) cases and on one incomplete case, every case gets exactly one name (out §C).
- **This run.** The records give 3 bound (POOL 2800 @4800, POOL 2801 @13200, ALLPOOL 2800 @6000) and 1 unbound (ALLPOOL 2801, 0.8721), with 0 failures. That reads **"partly", mechanically**. `outcome_pooled` / `outcome_inv` stored in every record equal `pool_outcome` recomputed (out §D).
- **What "partly" can say.** With per-run binding probability p:

  | p | P("restores") | P("partly") | P("does not") |
  |---|---|---|---|
  | 0.95 | 0.815 | 0.171 | 0.014 |
  | 0.90 | 0.656 | 0.292 | 0.052 |
  | 0.75 | 0.316 | 0.422 | 0.262 |
  | 0.50 | 0.062 | 0.250 | 0.688 |

  So "partly" is consistent with anything from p ≈ 0.5 to 0.95: Wilson [0.30, 0.95]. Against the S89 pilots on the same (task, seed), the conv arms win 3 vs 0 (sign test p = 0.125). The pilots ran on another CPU, though the path reproduces across the two through 2400.
- **What the addendum may say.** The descriptive counts are right. The causal sentences (rows 12, M5) say more than this.

### 3.4 The conv candidate's wording correction at `5357889`

| | text |
|---|---|
| `2279ca1` (13 s earlier) | "One candidate is that a key's write vector can carry its own value (KEY, VAL adjacent within 4 tokens), which a Hebbian outer product of single-token codes cannot." |
| `5357889` (r9 l.212–214) | "One candidate is the short-convolution account from associative recall (H3, Based): with the conv, the write at a VAL position carries its KEY one token back in the same layer. So the key-value association needs no earlier layer to copy the key forward." |

**Why the change was needed.** The convolution is *strictly causal*: position t sees t − 3 … t (`test_short_conv` l.43–46, CHECK 55). In the layout, every VAL is at its KEY's position + 1 (tasks l.369, `kpos + 1`).
- So the write at a KEY position cannot contain the value that follows it, and the first wording had the direction wrong.
- The convolution's output feeds Q = K and V (`test_short_conv` l.50–52). The write at the VAL position therefore carries KEY (lag 1) and VAL, which is the corrected account.
- Dropping "which a Hebbian outer product of single-token codes cannot" also removes a claim that the delta memory contradicts, since delta binds without the conv.

**Its standing.** The correction is prose, made after the records, about a candidate explicitly "not separated here". It changes no number or reading. It could be made checkable: the records do not store the learned conv weights by lag, whereas H's records do (`conv_absmean`). The lag-1 weight's mean |w| would test it (fix 6 below).

## 4. Is the revised screen, and the module, fit for G's and H's use?

**For G: yes.** G vendored the exact blob (`e24d5e1`) at `db14e9b`, two minutes after F's addendum. G's outcome is ROUTED-BOUND with the within-sequence η² on ALLPOOL and the pooled one on POOL, the other form printed (`explore_g_s89.py` l.56–61). §2.2 shows this reads routing, not the labelling, on ALLPOOL.

G should print three things:
- **(i)** DISCOVERED (cos) on ALLPOOL is passed by a gate at the 0.533 cap (§2.4). Print `seq_routed_frac` beside it.
- **(ii)** On POOL the pooled form also requires one fixed context → channel map across sequences. A gate that routes with per-sequence channels reads ROUTED on ALLPOOL and not on POOL, so a POOL → ALLPOOL comparison should be made in one form (the within-sequence form is defined on both).
- **(iii)** `key_sep` / `key_part` in these records read context types 2–5 on ALLPOOL and pool keys 0–3 on POOL (old fix 8). Do not tabulate them.

Old fix 7 (`task.ctx_tokens` is a role label on ALLPOOL) is G's own CHECK to make.

**For H: not applicable.** H's S89 task is the grouped layout at S = 8 with its own key pool. `explore_h_s89.py` l.11 notes that `explore_f_tasks` has no grouped-layout pool variant, and H's tip holds no `explore_f_tasks.py`. The README's "G and H vendor" (l.667) should say G.

**For F's own screen.** S89 is closed, so its stale docstring lines (fixes 4, 5, 8-v2, 10) change no reading. They are still false or missing text in a file the module's users read.

## Fixes (most important first)

1. **Correct the causal sentences and the rounding (r9, README).**
   - Replace r9 l.208–209 with:
     > With the convolution the Hebbian perfect gate bound 3 of 4 runs where the S89 pilots on the same (task, seed) bound 0 of 4 (3 vs 0, exact sign test p = 0.125; post hoc, and on another CPU). The reading fixed beforehand is "partly".
   - Replace README l.668–669's first two sentences with:
     > Post hoc, with a trainable convolution on the memory's input the Hebbian perfect gate bound 3/4 (against 0/4 for the S89 pilots on the same seeds; sign test p = 0.125); reading "partly".
   - In r9 l.194 and README l.660, replace "0.28 / 0.28" with "0.28 / 0.27".
2. **Print the within-sequence DISCOVERED's behaviour at the cap.** Add to the module comment after tasks l.640, and to s89 l.92:
   > On ALLPOOL val_cos_inv = 1 − (the fraction of probe sequences whose streams the gate separates) for a gate that is one-hot per stream, so DISCOVERED (val_cos_inv < 0.5) holds for any gate separating more than half the sequences — including a fixed type → channel map at the 0.533 cap (balanced map: 0.464). Read DISCOVERED on ALLPOOL only beside seq_routed_frac; ROUTED*@end (η² > 0.9) is not affected.
3. **Add a negative control to `check_inv_stats`** (tasks l.765–836):
   > CHECK: gates blind to the source — a position-only gate (channel 0 for t < 10), a block-parity gate (number of CTX tokens seen, mod 2), the latest key type's parity — score within-sequence η² < 0.5 and margin < 0.1 on both tasks (replica: ≤ 0.448 and ≤ 0.011).
4. **State which form POOL prints.** Either print both forms in `report_s89` on POOL (rep6 l.92–94), as s89 l.93 promises, or change s89 l.93 to:
   > On POOL the pooled form is read and printed (it also requires one fixed CTX → channel map across sequences); the within-sequence form is recorded.
5. **Correct "not taken" and the module paragraph.**
   - Replace r9 l.256–257 with:
     > Not taken: fixes 4 and 7 (phase 2 and the learned arms, which did not run); 5 (phase 0's Bayes bound — phase 0 ran, and its 'not stopped' was guaranteed: bound ≈ 0.19 / 0.42 against 0.6); 6 for S89 (taken in explore_f_s89_post); 8's v2 classes and lite-id statistics (the None guard is taken through pool_outcome); 10 (P(valid) = p⁴, 0.66 at p = 0.9).
   - Replace r9 l.259–260's "first 34603 bytes" with "first 35,605 bytes (34,603 characters)", and "for G and H" with "for G (H's task is the grouped layout and does not use it)". Make the same change in README l.667 and in the CHECK label (child l.237: print `len(v1.encode())`).
6. **Make the conv account checkable.**
   - Add to `explore_f_s89_post_child.run`'s record `conv_absmean = m.short_conv.conv_w.abs().mean(0).tolist()` (per lag), as H's records carry.
   - Print it in `report_s89_post`. The candidate predicts that lag 1 grows above lags 2–3.
   - For any rerun, add a CHECK that the conv path reproduces `O_HEBB_CONV_POOL|2800` at `73035d3` through 2400. This ties `forward_conv` beyond the identity initialisation.
7. **Name the after-record code change.** Add to r9's disclosure:
   > explore_f_s89_tasks.py (the S89 parent, not in the follow-up's code SHA) changed at 2279ca1, after the follow-up's records: pin() also admits S89's recording CPU (2.80 GHz), so S89's report reads its records from a 2.10 GHz container. No follow-up record or reading depends on it.
8. **Fix the prose ranges.**
   - r9 l.202: "rises to 0.67 at 4800 and 0.73 at 7200, holds 0.76–0.80 from 9600 to 20400, then climbs to 0.87 at 24000".
   - r9 l.217: "stay at 0.22–0.29 for all 24000 updates (0.24–0.29 from 2400)".
9. **Remove the stale S89 docstring lines** (they change no reading).
   - s89 l.6 and l.79–80: "unpaired descriptive rates on F's seeds (not run)".
   - After l.64: the Bayes-bound paragraph of the first review's fix 5.
   - l.97: delete "and v2".
   - l.21–22: "the S89 runs are on a Xeon @ 2.80GHz; the revision's CHECKs and the follow-up ran on a Xeon @ 2.10GHz".
   - Disclose S89's own dirty start (`8ba99bb+dirty`).
10. **Print the new rule's operating characteristic** beside its reading (post l.36–39):
    > With 4 runs, P("restores") = p⁴, P("partly") = 4p³(1 − p): at p = 0.75 the modal reading is "partly" (0.42); at p = 0.95 "partly" still occurs 17% of the time. 3/4 has Wilson [0.30, 0.95].
11. **(Hypothesis wording)** In post l.32–33, quote Revision 10 §11 whole: "one Hebbian channel binds the grouped layout 0/60 without the convolution and a minority with it at a higher learning rate (18/40, 10/20)".

**Reviewed:** claude/explore-F `b0c6332b`, `3dd5b746`, `4c8a343d`, the progress commits `b036d91`, `530240c`, `e929827`, `209a271`, and `73035d37`, `2279ca1f`, `53578896`. References: G at `e46799a` / `db14e9b`, H at `1b5dfed`, rev10-draft at its tip. Script and output: `docs/review_2026-10/screen_checks_s89_F_v2.py`, `screen_checks_s89_F_v2_out.txt`.
