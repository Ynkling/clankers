# Check: Session G's report 9 (S89H) at `claude/explore-G` `996e38b7`

*Referee V, 22 October 2026. What this covers:*
- *`explore_out/G/report_9.md` ("r9 l.N"), the README's S89H section (`explore_out/README.md` l.735-753), the records `explore_out/G/s89h_results.json` (204 runs), `s89h.log`, `s89h_validity.json`, `s89h_summary.json`, the two `checks_s89h_*.json` and the vendored `refs/f_s89_post_conv_records.json`, all at `996e38b7`;*
- *`explore_g_s89h.py` ("s89h:N") and `explore_g_s89h_child.py` ("h-child:N") at every one of the 61 commits from `d5d48eeb` (the readings commit) to `996e38b7`, with the modules they import;*
- *S89's delta records (`s89_results.json`), S87's and S82's lite records, and F's store `129215e:explore_out/F/s89_post_results.json`.*

*It was read against `check_G_report8.md` and `screen_review_s89_G_v2.md` (fix 1 / reading 2b, l.290: if REG3_KEYHINGE does not beat REG3 (no hinge), "the register does not need the hinge here").*

*The numbers come from `check_G_report9.py` ("chk §N"). It reads git objects only, writes nothing and prints only. The script:*
- *execs G's parent `explore_g_s89h` and `explore_g_s89` at `d5d48eeb` statement by statement, with `explore_g_rhl_parent`, `explore_g_common`'s statistics, `test_conv_lr.mcnemar_greater`, `test_router_confirm.fisher_greater` and `test_router_layout`'s `fail_class`, all from blobs, as `check_G_report8.py` does;*
- *runs G's own `report()` on the committed records, with its one file write sent to memory, and compares the output with the log line by line;*
- *cross-checks the counts with `rlib.py`.*

*torch was not needed. No S89H screen review by V exists; S89H ran unreviewed.*

## Verdict

**The POOL numbers are right and the POOL readings are the committed code's mechanical output. Here, unlike S89, they mean what they say. The ALLPOOL budget in the report is wrong: every ALLPOOL run, the perfect gate included, ran 24000 updates, not 48000.**

- **Reproduction.** G's `report()` from the `d5d48eeb` blob, run on the committed records, reproduces the log from the count table onward with 0 differing lines (390 lines; the log adds only `SCREEN_DONE`). Its summary equals the committed `s89h_summary.json` (chk §2).
- **Cross-checks.** Every ROUTED-BOUND and BOUND count in r9 and the README agrees with `rlib` and with the other outcome form (chk §3).
- **The code did not change.** No `.py` file changed between `d5d48eeb` and `996e38b7`, and the vendored F file has one blob. All 204 records carry code SHA `0f1f20cb2fa8`, attempt 1, ok, and the same CPU.
- **The budget defect.** The parent sets `ITERS = dict(pool=24000, allpool=48000)` (s89h:123), but the value never reaches the child:
  - `gc.run_screen` calls `run_child(screen.CHILD, "run", dict(arm=arm, seed=s))` with no `iters` (`explore_g_common.py:164`). `ITERS` is used only to sort the queue and to fill the meta.
  - The child takes `p.get("iters", rc.ITERS)` (h-child:209), and `rc.ITERS = 24000` (`explore_g_rhl_child.py:81`).
  - The records say so: `iters` is 24000 on all 102 ALLPOOL records, the largest `stopped_at` is 24000, and all 100 unbound ALLPOOL runs stop at 24000 (chk §1). Only the meta and `s89h_validity.json` say 48000, because both are copied from `ITERS`.
  - Consequences:
    - "ALLPOOL 48000" is wrong wherever it appears: r9 title l.1, the table header l.11, l.57 ("budgets differ (24000 / 48000)"), l.83 ("at twice the budget"), README l.737 and the commit headline.
    - The user's instruction quoted at s89h:11-12, "ALLPOOL at 48,000", was not carried out.
    - ALLPOOL ran at S89's budget, so the ALLPOOL memory comparison is at equal budget.
    - No CHECK covers the budget; the end-to-end CHECKs pass `iters` explicitly.

**Rows: 71 checked. 54 MATCH, 7 PARTIAL, 8 DISAGREE, 2 NOT RECORD-CHECKABLE.** Six of the eight DISAGREE rows are the 48000 budget.

**Reading 2b (§3). On Hebbian + conv POOL, every hinge beats the no-hinge register. 2b does not apply, and the names are not descriptive-only.**
- KH vs REG3_H: 9/0, p 0.00195 (two-sided 0.0039).
- FREQ vs REG3_H: 9/0, the same p.
- REPEAT vs REG3_H: 10/0, p 0.00098 (two-sided 0.0020).

An inert hinge carrying REG3_H's records would read "needs key identities" (9/0). A hinge drawn independently at REG3_H's rate (0.45) reads "replaces" with probability 0.0003. So "replaces the key hinge" here does exclude a hinge that adds nothing over the register, which on delta (4/0, p 0.0625) it did not. On ALLPOOL every hinge-vs-REG3 pair is 0/0.

**The cross-memory table (§4).** Every cell recomputes from both stores. The comparisons are paired: seed-matched one-sided McNemar on the same 20 seeds, with p given in the observed direction. The memory and the conv change together, and the shared batches and initial parameters are asserted, not record-checkable.

**ALLPOOL (§5).** It is 0/100 BOUND for the learned arms. The hinge arms stop at or near the type-level cap; BASELINE_H and REG3_H do not reach it (sequences routed ≤ 0.02). As in S89, the null cannot separate "the hinge fails" from "no type-level gate can be ROUTED here", and here it is also a null at 24000.

## 1. Statements against the records

### 1a. r9 header (l.1-9)

| # | statement | source | recomputed | verdict | note |
|---|---|---|---|---|---|
| A1 | "POOL 24000, ALLPOOL 48000" | r9 l.1 | ALLPOOL records: `iters` 24000 on 102/102, max `stopped_at` 24000 | **DISAGREE** | See the verdict. |
| A2 | code SHA 0f1f20cb2fa8; Xeon 2.10GHz; torch 2.14.0+cu130; 1 thread | l.3 | 204/204 keys and `sha`; 4/4 log banners; store provenance | MATCH | |
| A3 | Hebbian + conv 'layer', K 4, k 2, SLOW, S89's hinges unchanged | l.3-5 | records: conv 'layer', k 2, recipe slow on 204/204; CHECK rows K 4; the child imports S89's child for the hinges (h-child:45) | MATCH | The decay 0.95 is in the docstring, not in the records. |
| A4 | seeds 2840-2841 and 2842-2861, paired across arms, tasks and with S89's delta | l.5-6 | identical seed sets in every arm, task and S89 cell | MATCH | Pairing is by seed. Shared batches are not record-checkable (§6). |
| A5 | spec, readings and CHECKs committed at d5d48ee before any run | l.6-7 | `d5d48eeb` 22:01:48; first CHECK file 22:09:49 (git d5d48ee); first record commit `1f6ea2e6` 22:16:58 | MATCH | |
| A6 | 204/204 ok, none retried | l.7-8 | ok 204; `attempt` = 1 on all 204 | MATCH | |
| A7 | 19.2 CPU-hours | l.8 | Σ `secs_wall` 19.19 h; Σ `secs` 18.85 h | PARTIAL | As in r8 A6: the figure is summed wall time. |
| A8 | one restart, after 116 learned runs; the resume re-ran all 139 CHECKs, same SHA, cached runs kept | l.8-9 | 1 `=== RESTART` (01:31); 116 progress lines before it; 2 CHECK files × 139 ok; 84 queued on resume; both banners SHA `0f1f20cb2fa8` | MATCH | |

### 1b. Counts (r9 l.11-24; chk §3)

| # | statement | recomputed (G's `routed_bound`, `inv_view` on ALLPOOL / rlib) | verdict | note |
|---|---|---|---|---|
| B1 | header "ALLPOOL (48000)"; "the other form gives the same counts" | 24000; the other form gives identical counts on every arm | **DISAGREE** | The budget. The counts part matches. |
| B2 | PERFECT_H 2/2 (6000, 7200) POOL, 2/2 (9600, 6000) ALLPOOL; ORACLE_D 2/2, 2/2 | identical | MATCH | ALLPOOL's perfect gate stopped at 12000 / 8400. |
| B3 | BASELINE_H 0/20 NEVER (4 bound unrouted); ALLPOOL 0/20; delta 0/20, 0/20; lite 0/20 | 0 (BOUND 4); 0 (BOUND 0); 0, 0; S87 0/20 | MATCH | |
| B4 | REG3_H 9/20 [0.26, 0.66] MINORITY; 0/20; delta 16/20, 0/20; S82 2/20 | identical | MATCH | |
| B5 | REG3_KEYHINGE_H 18/20 [0.70, 0.97]; 0/20; delta 20, 0; lite 20/20 | identical | MATCH | BOUND 19. Seed 2852 binds at 6000 with margin **0.877** < 0.9; seed 2857 is STREAM-PARTIAL. |
| B6 | REG3_FREQHINGE_H 18/20; 0/20; delta 7, 0; lite 17/20 | identical (fails on 2843, 2848) | MATCH | |
| B7 | REG3_REPEATHINGE_H 19/20 [0.76, 0.99]; 0/20; delta 20, 0 | identical (fails on 2857) | MATCH | |
| B8 | transitions: KH 6000 (4800-9600), FREQ 6600 (6000-20400), REPEAT 6000 (4800-19200), REG3 6000 (4800-9600) | identical (over BOUND runs) | MATCH | |
| B9 | POOL unbound: REG3 OTHER 11; BASELINE STREAM-PARTIAL 10, OTHER 5 | identical, plus BASELINE KEY 1, KH STREAM-PARTIAL 1, FREQ STREAM-PARTIAL 2, REPEAT STREAM-PARTIAL 1 | PARTIAL | Incomplete. |
| B10 | ALLPOOL: MERGED 11-16 per hinge arm; BASELINE and REG3 OTHER 17-19 | MERGED 16 / 15 / 11; OTHER 17 / 19 | PARTIAL | Not mentioned: REPEAT POSITION 5 (and KEY 2, OTHER 1, STREAM-PARTIAL 1); BASELINE KEY 3; REG3 POSITION 1; KH OTHER 1, POSITION 1, STREAM-PARTIAL 2; FREQ KEY 2, STREAM-PARTIAL 3. |

### 1c. Readings (r9 l.26-43; chk §4)

| # | statement | recomputed | verdict | note |
|---|---|---|---|---|
| C1 | POOL: KH discovered 18/0, p 3.8e-6 | 18/0, p 3.81e-6 | MATCH | |
| C2 | 2b: "the hinge is needed over the register" (KH only 9, REG3 only 0, p 0.002) | 9/0, p 0.00195; `readings()` 2b | MATCH | Pre-registered (§2). |
| C3 | FREQ "survives" (KH only 2, FREQ only 2; vs REG3 9/0) | 2/2, p 0.688; vs REG3 9/0, p 0.00195 | MATCH | On POOL this is a CTX hinge (§3). |
| C4 | REPEAT "replaces the key hinge" (KH only 0, REPEAT only 1; vs REG3 10/0, p 0.001) | 0/1; 10/0, p 0.000977 | MATCH | |
| C5 | question: "holds, and the hinge is needed here" | identical (s89h:225-226) | MATCH | |
| C6 | ALLPOOL: "the labelled hinge itself fails here" (0/20 = 0/20); "cannot be asked here" | 0/0, p 1 | MATCH | Mechanical, and at 24000 (§5). |
| C7 | clamps does not apply (arm only 18 / 19 vs BASELINE only 0) | identical; commitment 0.172 / 0.925 vs 0.708 | MATCH | |
| C8 | caveat 2: "REG3_H does not beat REG3_FREQHINGE_H (0 / 9) … On delta it did: 1 / 10, p 0.006" | S89H: REG3 only 0, FREQ only 9. Delta: REG3_D only **10**, FREQ only **1**, p 0.0059 | PARTIAL | The numbers are right but the two pairs are printed in opposite orders. Read in the first pair's order, "1 / 10" says the reverse of the record. |
| C9 | S89 delta read through the same function: KH vs REG3_D 4/0, p 0.0625, "the register does not need the hinge here", "holds (descriptive …)"; ALLPOOL "fails here" | identical (log; s89h:286-301); dated "after the fact" in the code (s89h:63-65) | MATCH | |
| C10 | ALLPOOL caveats: cap 0.533; type-level perfect gate on Hebbian without conv only, O_TYPE_HEBB_ALLPOOL 0/2 | printed (s89h:302-305); F `129215e`: O_TYPE_HEBB_ALLPOOL 2800 and 2801 unbound at 24000 | MATCH | |

### 1d. Across memories and tasks (r9 l.45-66; chk §5)

| # | statement | recomputed | verdict | note |
|---|---|---|---|---|
| D1 | REG3 9/20 vs 16/20, 0/7, p 0.008 (delta higher) | 0/7, p(D >) 0.00781 | MATCH | |
| D2 | KH 18/20 vs 20/20, 0/2, 0.25 | 0/2, p(D >) 0.25 | MATCH | |
| D3 | REPEAT 19/20 vs 20/20, 0/1, 0.5 | 0/1, p(D >) 0.5 | MATCH | |
| D4 | FREQ 18/20 vs 7/20, 11/0, 5e-4 (Hebbian higher) | 11/0, p(H >) 4.88e-4 | MATCH | |
| D5 | BASELINE 0/20 vs 0/20 | 0/0 | MATCH | |
| D6 | same seeds; the memory and the conv differ together | seed sets equal; both slow, k 2, 24000 on POOL; conv 'layer' vs none | MATCH | |
| D7 | POOL → ALLPOOL: REG3 9→0, p 0.002; KH 18→0, 3.8e-6; REPEAT 19→0, 1.9e-6; BASELINE 0→0 | 9/0, 0.00195; 18/0, 3.81e-6; 19/0, 1.91e-6; 0/0 | MATCH | |
| D8 | "the budgets differ (24000 / 48000)" | 24000 on both tasks | **DISAGREE** | The POOL → ALLPOOL drop is at equal budget. |

### 1e. What happened (r9 l.68-89; chk §6, §7)

| # | statement | recomputed | verdict | note |
|---|---|---|---|---|
| E1 | REG3 9/20 vs 16/20; failures end with β^R 0.10 → 0.03 | the 11 failures (all OTHER): CTX β^R median 0.100 (0) → 0.110 (1200) → 0.093 (2400) → 0.065 (4800) → 0.031 (end) | MATCH | |
| E2 | KH and REPEAT beat REG3 by 9-10 pairs (p 0.001-0.002); delta could not show it (4 vs 0) | 9/0, 10/0; delta 4/0, 4/0 | MATCH | |
| E3 | REPEAT commitment ~0.08 at 1200, 0.92 at 2400 | 0.0797, 0.9247 | MATCH | |
| E4 | REPEAT β^R at CTX 0.28 (2400), ~0.6 (4800), 0.79 (end) | 0.284, 0.611, 0.790 | MATCH | |
| E5 | layer 1 routes at KEY and VAL (0.99 / 0.01) | diagnostics, identical | MATCH | |
| E6 | FREQ 18/20 vs 7/20; commitment 0.17 (2400), 0.96 (3600); β^R CTX 0.41 at 4800 | 0.1720, 0.9602; 0.415 | MATCH | |
| E7 | "ALLPOOL: still no arm routes, at twice the budget" | 24000, S89's budget | **DISAGREE** | |
| E8 | hinge arms at the cap: within-sequence η² 0.41-0.58 (median); 0.33-0.57 of sequences routed in 11-17 runs per arm | medians 0.513 / 0.578 / 0.409; runs with `seq_routed_frac` ≥ 0.30: 17 / 17 / 11, spanning 0.328-0.574 | MATCH | 13 runs exceed 64/120 (§5). |
| E9 | MERGED failures | 42 MERGED among the 60 hinge runs | MATCH | |
| E10 | BASELINE and REG3 commit early (0.93-0.95 at 2400) to OTHER plateaus | 0.9531, 0.9322; OTHER 17 and 19 | MATCH | |
| E11 | no run learned the relative routing | 0/100 bound; max margin 0.574, max η²_inv 0.613 | MATCH | |
| E12 | POOL BASELINE binds 4/20 unrouted (margin ~0.46); delta 14/20 | margins 0.448-0.473; delta 14 | PARTIAL | G's own "BOUND ROUTED (mine)" column reads 4/20 for this arm. "Unrouted" holds under ROUTED* only, as in r8 D21. |

### 1f. CHECKs (r9 l.91-112)

| # | statement | recomputed | verdict | note |
|---|---|---|---|---|
| F1 | 139 CHECKs, asserted before each segment, all ok | 2 files × 139 ok; 278 `CHECK S89H:` log lines, 0 FAIL | MATCH | |
| F2 | "S89's full set (93 rows: …)" | **71** rows prefixed "(S89)"; 73 counting the two re-run name enumerations; S89H's own rows 68 | **DISAGREE** | |
| F3 | conv the only extra parameter, shared equal, identity init bit for bit, every arm, three tasks | 18 rows ok (`short_conv.conv_w`) | MATCH | |
| F4 | conv off: REG3_H / BASELINE_H = rhl_child's | 2 rows ok | MATCH | |
| F5 | conv insertion = `forward_conv`; strictly causal | 4 rows ok | MATCH | |
| F6 | F's O_HEBB_CONV_POOL / ALLPOOL\|2800 reproduce through 2400 (vendored from 129215e) | rows ok; the 4 vendored records equal F's store (blob `6e9865974c0f`) value for value | MATCH | |
| F7 | S87's REG3_KEYHINGE, REG3_FREQHINGE, BASELINE\|2700 with the conv off | 3 rows ok | MATCH | |
| F8 | hinge off = REG3_H; end to end; "each term's gradient reaches the gate only (not the stack, not the conv)" | rows ok; the reached list includes `embed.weight` and `gate_conv.conv_w` ("the stack and the conv: none") | PARTIAL | The embedding is reached, as in r8 E6. |
| F9 | the lrs (conv in the rest group; perfect gate's gate group empty) | rows ok | MATCH | |
| F10 | hand cases for 2b, the hurt rule and the question; four names exclusive (n 10, 20); `inv_view` DISCOVERED | 8 + 2 + 1 rows ok | MATCH | |

### 1g. "For the main line" (r9 l.114-122)

| # | statement | recomputed | verdict | note |
|---|---|---|---|---|
| G1 | REPEAT replaces KH on POOL (19 vs 18), both needed over the register (9/20) | 0/1; 10/0 and 9/0 | MATCH | §3 |
| G2 | the result S89 could not give on delta | delta KH vs REG3_D 4/0, p 0.0625 | MATCH | |
| G3 | "the strongest label-free result so far" | – | NOT RECORD-CHECKABLE | |
| G4 | the spec draft is Hebbian lite without conv; F's pilots show the Hebbian perfect gate needs the conv on pooled keys | F's conv records only (POOL 2/2; ALLPOOL 1/2 at 24000, 2/2 at 48000) | NOT RECORD-CHECKABLE | The no-conv failure is in F's report 9, not re-read. |
| G5 | ALLPOOL 0/20 every arm on both memories, "with the type-level cap visible in the failures" | 0/20 everywhere; the cap shows in the hinge arms only | PARTIAL | BASELINE_H and REG3_H route 0.00-0.02 of sequences (§5). |

### 1h. README (l.735-753) and the commit message

| # | statement | recomputed | verdict | note |
|---|---|---|---|---|
| H1 | "-allpool (48000)" | 24000 | **DISAGREE** | l.737 |
| H2 | seeds; perfect gate 2/2 on both; readings at d5d48ee before any run; 204/204 ok | as A4-A6, B2 | MATCH | |
| H3 | table counts (POOL, ALLPOOL, S89 delta POOL) | identical | MATCH | |
| H4 | table readings (2b 9/0, p 0.002; "fails here"; "survives" 2/2, vs REG3 9/0; "replaces" 0/1, vs REG3 10/0) | identical | MATCH | |
| H5 | ALLPOOL "cannot be asked here" (**every arm** at the type-level routing cap) | BASELINE_H and REG3_H: `seq_routed_frac` 0.000-0.020, KEY / OTHER / POSITION; REPEAT 9/20 below 0.30 | **DISAGREE** | r9 l.83 limits it to the hinge arms. |
| H6 | S89 delta POOL through the same function: "holds (descriptive …)" | log | MATCH | |
| I1 | commit `996e38b7` headline: POOL counts, the names, "REG3 9/20 so the hinge is needed" | as B, C | MATCH | |
| I2 | headline: "ALLPOOL every arm 0/20 **at 48000**" | 24000 | **DISAGREE** | |

**Totals: 71 rows.**
- MATCH 54.
- PARTIAL 7: A7, B9, B10, C8, E12, F8, G5.
- DISAGREE 8: A1, B1, D8, E7, F2, H1, H5, I2.
- NOT RECORD-CHECKABLE 2: G3, G4.

## 2. Pre-registration order (chk §0)

| time (UTC) | commit / event | S89H records (learned) |
|---|---|---|
| 10-20 22:26:05 | V's `screen_review_s89_G_v2.md` (review-V `c42ace84`), fix 1 = reading 2b; S89 at 3 learned records | – |
| 10-21 09:42:14 | `d745d869`: S89's last progress commit (203 records) | – |
| 09:46:54 | `8e9e3187`: report 8 (S89 complete, 204) | – |
| 09:47:17 | `c98648e6`: register spec draft revision (no `.py`) | – |
| **22:01:48** | **`d5d48eeb`: S89H parent, child and vendored F records** | 0 |
| 22:04:49 | V's `check_G_report8.md` (review-V `b2169dc0`) | 0 |
| 22:06:52 | `a5100a01`: log only | 0 |
| 22:09:49 / 22:11:55 | first CHECK pass (git d5d48ee, 139/139) / committed (`d877f157`) | 0 |
| 22:12:52 / 22:16:58 | `s89h_validity.json` written / **first records** (`1f6ea2e6`: the 4 PERFECT_H) | 4 (0) |
| 22:22:00 | `865addfc`: **first learned records** (ALLPOOL, seed 2842: BASELINE_H, REG3_H, KH, FREQ) | 8 (4) |
| 10-22 01:31 | worker restart at `cd7a00bc` (120 records, 116 learned); CHECKs re-run 01:40:17 | |
| 03:07:32 | `d60e8668`: last progress (201) | |
| 03:10:42 | `report()` rerun (store provenance `d60e866+dirty`) | 204 |
| 03:12:24 | `996e38b7`: report 9 + README | 204 (200) |

**The code.** Over all 61 commits there is one blob each for:
- `explore_g_s89h.py` (`c31538f1`) and `explore_g_s89h_child.py` (`1cd42e77`);
- `explore_g_s89.py` and `explore_g_s89_child.py` (S89's `c76877d6` and `cd9998ab`, the same as at S89's `db14e9b9`);
- `explore_g_rhl_parent.py`, `explore_g_rhl_child.py`, `explore_g_common.py`, `explore_f_tasks.py`, `explore_common16.py`, `explore_common.py`, `test_short_conv.py` and `test_multilayer_binding.py`;
- the vendored F records (`ef613de7`).

`git diff --name-only d5d48eeb 996e38b7` lists no `.py` file. So nothing in the run code, the reading code or the report code changed after the readings commit, before or after the first record.

**The code SHA.** It is constant: `0f1f20cb2fa8` on all 204 record keys and `sha` fields, on all 4 log banners, in the meta and in the validity file. The 35-module SHA itself was not recomputed (§6).

**The design is post-S89.** S89H's design was fixed in `d5d48eeb`, before any S89H record: the configuration (Hebbian decay 0.95 + conv 'layer'), the arms (S89's five plus PERFECT_H, with the hinges imported unchanged from S89's child), the seeds, the readings, 2b, the hurt rule and the question. But `d5d48eeb` came **after all 204 S89 records and after report 8** (12 h 15 min later). The docstring says so itself:
- "S89 (report 8) ran on delta memory only" (s89h:7);
- it quotes S89's per-arm delta outcome, "POOL REPEAT 'replaces the key hinge' with KH vs REG3_D 4 / 0, p 0.0625" (s89h:63-65).

So S89H was designed knowing S89's results on the same seeds. The choice of Hebbian + conv rests on F's conv pilots and on the user's instruction of 21 October (s89h:8-12). It was not tuned on S89H records, since none existed.

V's `check_G_report8.md` landed 3 minutes after `d5d48eeb`, so S89H could not use it. That explains why s89h:14-15 repeats the v2 review's inert-hinge mechanism ("would read 'replaces'"), which `check_G_report8` §3 corrected.

**The registered design and the run differ in one respect: the ALLPOOL budget** (verdict). The readings were applied as registered, to records made at 24000.

## 3. Reading 2b (chk §4)

**Where it is defined.**
- In committed code, `explore_g_s89h.py` at `d5d48eeb` (10-21 22:01:48), before any S89H record:
  - docstring s89h:42-45;
  - `need = hp.beats(KH, REG, rb)` (one-sided exact McNemar, p < 0.05) and the printed 2b at s89h:207-209;
  - the hurt rule at s89h:215-222;
  - the question at s89h:225-226;
  - hand cases at s89h:236-255 (CHECK rows 0-7, ok).
- It is V's v2 fix 1 (`screen_review_s89_G_v2.md` l.290) as written. G applies it to S89's delta records only "after the fact (descriptive …)" (s89h:63-65, 298-301).
- The blob is unchanged through `996e38b7`.

**Recomputed:**

| screen, task | pair (hinge vs REG3, no hinge) | hinge only / REG3 only | one-sided p (hinge >) | two-sided p |
|---|---|---|---|---|
| S89H POOL (Hebbian + conv) | KEYHINGE vs REG3_H | 9 / 0 | 0.00195 | 0.0039 |
| | FREQHINGE vs REG3_H | 9 / 0 | 0.00195 | 0.0039 |
| | REPEATHINGE vs REG3_H | 10 / 0 | 0.00098 | 0.0020 |
| S89H ALLPOOL | each hinge vs REG3_H | 0 / 0 | 1 | 1 |
| S89 delta POOL | KEYHINGE_D vs REG3_D | 4 / 0 | 0.0625 | 0.125 |
| | REPEATHINGE_D vs REG3_D | 4 / 0 | 0.0625 | 0.125 |
| | FREQHINGE_D vs REG3_D | 1 / 10 | 1 (REG3_D >: 0.0059) | 0.0117 |
| S89 delta ALLPOOL | each | 0 / 0 | 1 | 1 |
| lite, Hebbian no conv (S82, 2100-2119, slow) | REG3_KEYHINGE vs REG3 (20/20 vs 2/20) | 18 / 0 | 3.8e-6 | 7.6e-6 |
| lite, delta (S82, 2100-2109) | REG3_KEYHINGE_D vs REG3_D (10/10 vs 2/10) | 8 / 0 | 0.0039 | 0.0078 |
| lite, delta (S87, 2720-2729) | KEYHINGE_D / FREQHINGE_D vs REG3_D (10/10 vs 0/10) | 10 / 0 | 0.00098 | 0.0020 |

S87's Hebbian lite seeds (2700-2719), the "lite" column of r9's table, have no REG3 arm, so 2b cannot be computed on them. On lite, the comparison exists only in S82, on other seeds.

**Does any hinge beat the no-hinge register?**
- **On Hebbian + conv POOL, yes: all three, at p ≤ 0.002 one-sided (≤ 0.004 two-sided).** REG3_H's 9 binding seeds are a subset of each hinge arm's. Its 11 failures are all OTHER, with β^R at CTX shutting from 0.10 to 0.03.
- **On ALLPOOL, no.** Nothing binds, so 2b is never reached ("fails here" comes first).
- **On delta POOL, no** (4/0, p 0.0625). FREQ is below the register there.
- **On lite (S82, S87), yes**, the register being 0-2 of 10-20.

**What "replaces the key hinge" means here.** The rule (S89's `arm_name`, s89:196-210) gives "replaces" because:
- REPEAT is discovered (19/0 vs BASELINE_H);
- p(KH > REPEAT) = 1 > 0.0625;
- KH only − REPEAT only = 0 − 1 ≤ 1.

Against the register, the counterfactuals that made the delta name hollow now fail:
- **A hinge with no effect** (REG3_H's records seed for seed) reads **"needs key identities"** (KH only 9, arm only 0, p 0.00195).
- **A hinge drawn iid at REG3_H's rate**, with KH's 18/20 and BASELINE's 0/20 fixed as observed, gets these names:

| arm rate | P("replaces") | P("weaker") | P("needs") |
|---|---|---|---|
| 0.45 (REG3_H observed) | 0.0003 | 0.035 | 0.965 |
| 0.66 (REG3_H Wilson upper) | 0.052 | 0.456 | 0.492 |
| 0.8 | 0.411 | 0.516 | 0.073 |
| 0.9 | 0.867 | 0.131 | 0.002 |

So on Hebbian + conv POOL, "replaces the key hinge" is a statement about a hinge that does work the register does not do: 10 seed pairs over REG3_H, and none below KH.

Two limits remain:
- "Replaces" is a count rule (b − c ≤ 1), not an equivalence test. Given KH 18/20, an arm whose true rate is 0.8 reads "replaces" 41% of the time.
- KH's 18 (not 19) rests on seed 2852, which binds at 6000 with margin 0.877, just under ROUTED's 0.9.

The register's own rate also fell from 16/20 (delta) to 9/20 here, with the memory and the conv changed together. "The hinge is needed" is therefore a statement about this configuration.

**What the frequency hinge's "survives" means here.** On POOL the frequency cut is the 2 CTX types, so the term penalises the stream at the 5 CTX positions and touches no key position (V's v2 caveat 2, printed). So "survives" (KH only 2, FREQ only 2; FREQ vs REG3_H 9/0) says that a CTX/stream hinge matches the key hinge and beats the register on Hebbian + conv. It says nothing about key frequency.
- The delta pattern does not recur. There, REG3_D beat FREQ 10/1, p 0.0059; here REG3_H only 0 against FREQ only 9, so "the term hurts the register" correctly does not print.
- FREQ's commitment stays low through 2400 (0.17) and commits by 3600 (0.96), as r9 says.
- r9 l.81 says why the memories differ is not tested. That is right: the S89H and S89 arms differ in the memory and the conv together.

**Verdict on 2b.**
- On Hebbian + conv POOL, 2b correctly does not apply; every hinge beats REG3_H, and both names mean what they say.
- On ALLPOOL, 2b is not reached.
- Through the same function, S89's delta POOL gives "the register does not need the hinge here", as `check_G_report8` §3 found.

## 4. The cross-memory table (chk §5)

Each cell is recomputed from `s89h_results.json` and `s89_results.json` at `996e38b7`, on the same 20 seeds 2842-2861. H = Hebbian + conv, D = delta.

| task.arm | H | D | H only / D only | both / neither | one-sided McNemar | two-sided | unpaired Fisher (one-sided) | r9 |
|---|---|---|---|---|---|---|---|---|
| POOL REG3 | 9/20 | 16/20 | 0 / 7 | 9 / 4 | p(D >) 0.0078 | 0.0156 | 0.024 | 0 / 7, 0.008 ✓ |
| POOL KH | 18/20 | 20/20 | 0 / 2 | 18 / 0 | p(D >) 0.25 | 0.5 | 0.24 | 0 / 2, 0.25 ✓ |
| POOL REPEAT | 19/20 | 20/20 | 0 / 1 | 19 / 0 | p(D >) 0.5 | 1 | 0.5 | 0 / 1, 0.5 ✓ |
| POOL FREQ | 18/20 | 7/20 | 11 / 0 | 7 / 2 | p(H >) 4.9e-4 | 9.8e-4 | 3.9e-4 | 11 / 0, 5e-4 ✓ |
| POOL BASELINE | 0/20 | 0/20 | 0 / 0 | 0 / 20 | 1 | 1 | 1 | – ✓ |
| ALLPOOL, all five arms | 0/20 | 0/20 | 0 / 0 | 0 / 20 | 1 | 1 | 1 | (log only) ✓ |

**The pairing.**
- The comparisons are **paired**: `hp.paired` matches records by seed, and the test is the exact one-sided McNemar.
- r9 prints the p in the observed direction. It labels the direction only for REG3 and FREQ; the KH and REPEAT p values are p(delta >).
- The seed sets are identical in every cell. Both screens use slow, k 2 and 24000 updates on both tasks (ALLPOOL included, by the defect).
- The pairs differ in the memory and the conv together, as r9 says.
- That paired seeds share batches and initial parameters is asserted in the docstring (s89h:25-27). The records cannot show it. The CHECKs tie S89H's run() with the conv off to S87's Hebbian records, not to S89's delta records.

**No cell differs.** The only cross-memory difference beyond chance at 0.05 two-sided is FREQ (delta below) and REG3 (Hebbian below). KH and REPEAT agree across memories.

## 5. ALLPOOL under S89H (chk §1, §6)

**The budget.** It was 24000 for all 102 records (verdict), so this ALLPOOL is S89's budget on another memory, not "twice the budget". F's own 48000 pilot shows the budget can matter for the Hebbian + conv perfect gate: O_HEBB_CONV_ALLPOOL_48K|2801 binds only at 28800. S89H's PERFECT_H seeds 2840 and 2841 bound within 24000 (9600, 6000), so validity passed at the budget actually run.

**The counts.** ROUTED-BOUND is 0/20 for every learned arm in both outcome forms, and BOUND is 0/20 for every learned arm. This differs from S89, which had 6 late key-split binds. By class:

| arm | classes | `seq_routed_frac` median (range) | η²_inv median | runs ≥ 0.30 | commitment at 2400 | median acc |
|---|---|---|---|---|---|---|
| BASELINE_H | OTHER 17, KEY 3 | 0.000 (0.000-0.000) | 0.038 | 0 | 0.953 | 0.373 |
| REG3_H | OTHER 19, POSITION 1 | 0.000 (0.000-0.020) | 0.092 | 0 | 0.932 | 0.350 |
| REG3_KEYHINGE_H | MERGED 16, STREAM-PARTIAL 2, OTHER 1, POSITION 1 | 0.463 (0.031-0.566) | 0.513 | 17 | 0.173 | 0.522 |
| REG3_FREQHINGE_H | MERGED 15, STREAM-PARTIAL 3, KEY 2 | 0.529 (0.000-0.574) | 0.578 | 17 | 0.073 | 0.581 |
| REG3_REPEATHINGE_H | MERGED 11, POSITION 5, KEY 2, OTHER 1, STREAM-PARTIAL 1 | 0.369 (0.000-0.547) | 0.409 | 11 | 0.204 | 0.424 |

Over the 100 learned runs: MERGED 42, OTHER 38, KEY 7, POSITION 7, STREAM-PARTIAL 6. In S89 (delta) it was KEY 73.

**The cap.** A fixed CTX-type → channel map separates at most 64/120 = 0.533 of sequences in expectation (F's `explore_f_tasks.py` b0c6332 l.641-646). ROUTED needs margin ≥ 0.9 and η²_inv > 0.9.
- **No learned run comes near ROUTED:** max margin 0.574, max `seq_routed_frac` 0.574, max η²_inv 0.613.
- **13 hinge runs exceed 0.533:** KH 4, FREQ 8, REPEAT 1, up to 147/256 = 0.574. Every value is a multiple of 1/256, the probe's size.
- **The 0.533 cap is a population figure.** On a 256-sequence probe, a type-level map's own score varies: the balanced map's CHECK score on its probe is 0.5117. So these values do not show a within-sequence comparison. The probe's tokens are not in the records, so the in-sample maximum over type maps was not computed (§6).
- **DISCOVERED (within-sequence form):** 25 runs have `val_cos_inv` < 0.5 (KH 7, FREQ 14, REPEAT 4). None is bound, so none is DISCOVERED.

**The perfect gate** is `task.perfect_gate`, the relative (first-appearance) index (h-child:152-157, CHECK row 100). Its pooled and within-sequence η² are both 1.000. As in S89, validity says the Hebbian + conv memory can use a relative routing that is handed to it, not that a gate under test can compute one. The type-level perfect gate was run by F on Hebbian without the conv only: O_TYPE_HEBB_ALLPOOL 0/2 at 24000 (F's store), printed as a caveat.

**What the null separates.**
- As in `check_G_report8` §4, it does **not** separate "the hinge fails" from "no gate these arms learn can be ROUTED here". The best any hinge arm reaches is the type-level cap.
- On this memory there is a descriptive difference S89 did not show: the hinge arms reach the type-level map and BASELINE_H and REG3_H do not. Runs with `seq_routed_frac` ≥ 0.30 number 17 / 17 / 11 for the hinge arms against 0 / 0. The no-hinge arms commit early (0.93-0.95 at 2400) to OTHER plateaus. So the hinge does move the gate on ALLPOOL, but only as far as the cap.
- "The labelled hinge itself fails here" is the correct mechanical output (KH 0/20 = BASELINE 0/20). As a statement about the hinge it is uninformative, and it is a statement about 24000, not 48000.
- The README's "every arm at the type-level routing cap" is wrong for BASELINE_H and REG3_H (H5).

## 6. What could not be checked

- The training itself: no rerun, and the child's torch code was not executed. CHECK rows were read from the log and the JSON files, not re-executed.
- The 35-module code SHA was not recomputed: `explore_common16.code_sha` imports the child under torch. Instead, no `.py` file changed over the range, and every record, banner, the meta and the validity file carry the same SHA.
- That paired seeds share batches and initial parameters across S89H and S89 (D6, §4), and that the decay is 0.95: neither is in the records.
- The in-sample maximum over fixed type maps on the 256-sequence probe (§5): it needs the probe's tokens.
- G3 ("strongest label-free result so far") and G4 (the spec draft; F's no-conv Hebbian failure on pooled keys, in F's report 9).

**Errors in the referee's own files.** No new error was found. One inherited point: the inert-hinge mechanism of `screen_review_s89_G_v2.md` l.45 and l.269 (corrected in `check_G_report8.md` §3 and the v2 review l.341) is repeated in S89H's docstring, s89h:14-15. Here it is moot: REG3_H at 9/20 makes the inert hinge read "needs" (§3). No screen review of S89H was made before it ran. The budget defect (`explore_g_common.py:164` drops the arm's `iters`) was not caught by any review, including the S89 ones, where it had no effect because both tasks ran at 24000.
