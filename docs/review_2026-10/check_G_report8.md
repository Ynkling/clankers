# Check: Session G's report 8 (S89) at `claude/explore-G` `8e9e3187`

*Referee V, 21 October 2026. What this covers:*
- *`explore_out/G/report_8.md` ("r8 l.N"), the README's S89 section (`explore_out/README.md` l.713-733), the records `explore_out/G/s89_results.json` (204 runs), `s89.log`, `s89_validity.json`, `s89_summary.json` and the six `checks_s89_*.json`, all at `8e9e3187`;*
- *`explore_g_s89.py` ("s89:N") and `explore_g_s89_child.py` ("child:N") at every one of the 95 commits from `db14e9b9` to `8e9e3187`.*

*It was read against `screen_review_s89_G.md` (with its dated correction), `screen_review_s89_G_v2.md` (fixes 1-10) and `screen_review_s89_F_v2.md` (the routing cap).*

*The numbers come from `check_G_report8.py` ("chk §N"), which reads git objects only and writes nothing. The script:*
- *execs G's parent at `db14e9b9` statement by statement, together with `explore_g_rhl_parent`, `explore_g_common`'s statistics, `test_conv_lr.mcnemar_greater`, `test_router_confirm.fisher_greater` and `test_router_layout.fail_class`, all from blobs;*
- *runs G's own `report()` on the committed records, with its one file write sent to memory, and compares the output with the log line by line;*
- *cross-checks the counts with `rlib.py`, which recomputes the transition from the curve.*

*torch was not needed.*

## Verdict

**The numbers are right and the names are the committed code's mechanical output. Two of the names do not mean what they say.**
- **Reproduction.** G's `report()` from the `db14e9b9` blob, run on the committed records, reproduces the log from "OUTCOME FORM" onward with 0 differing lines (368 lines; the log adds only `SCREEN_DONE`). Its summary equals the committed `s89_summary.json` (chk §2).
- **Cross-checks.** Every ROUTED-BOUND and BOUND count in r8 and the README agrees with `rlib` (chk §3).
- **The code did not change.** No `.py` file changed between `db14e9b9` and `8e9e3187`. All 204 records carry code SHA `5d4a9b1dba33`, attempt 1, ok, and the same CPU.
- **The v2 review's fixes were not taken.** None of fixes 1-10 entered the code. Fix 1's REG3_D comparisons appear in r8 only as hand-computed prose, undated and attributed to `explore_g_s89.readings`.

**Rows: 70 checked. 54 MATCH, 11 PARTIAL, 3 DISAGREE, 2 NOT RECORD-CHECKABLE.**

**"The repeat hinge replaces the key hinge"** is what the committed rule outputs. The rule's literal content holds: KH only 0 and REPEAT only 0 on 20 pairs, and REPEAT beats BASELINE_D 20/0. But REG3_D, the register with no hinge, binds POOL 16/20, and neither hinge beats it at 0.05: each is 4/0, p 0.0625. So what the repeat hinge "replaces" is a hinge whose own contribution over no hinge is not established. Under V's v2 fix 1 the reading would be "the register does not need the hinge here", with the names read as descriptive. The name is not vacuous: an exactly inert hinge would read "needs key identities". But the name overstates (§3).

**The ALLPOOL null cannot tell "the hinge fails" from "no gate these arms learned can reach ROUTED here".**
- It is 0/20 ROUTED-BOUND for every learned arm; BOUND is 0, 0, 2, 2 and 2/20 (key split, margin ≈ 0).
- 12 runs end exactly at the type-level cap: separated fraction 0.477-0.535 against 64/120 = 0.533.
- The validity gate is the handed relative-index gate.

"The labelled hinge itself fails here" is the mechanical name. r8's own "ALLPOOL is not a test of hinges" is the correct reading (§4).

**"The frequency hinge needs frequent keys" on POOL is a misnomer.** On POOL the arm touches no key position. It is a CTX/stream hinge, and it is worse than no hinge: REG3_D only 10, FREQ only 1, p 0.0059 (§5).

## 1. Statements against the records

### 1a. r8 header (l.3-9)

| # | statement | source | recomputed | verdict | note |
|---|---|---|---|---|---|
| A1 | code SHA 5d4a9b1dba33 | r8 l.3 | 204/204 keys and `sha` fields; 8/8 log banners | MATCH | |
| A2 | `explore_f_tasks.py` from F's b0c6332, blob e24d5e10, asserted | l.4 | blob `e24d5e1081` at all 95 commits; log CHECK ok | MATCH | |
| A3 | seeds 2840-2841 (perfect gate) and 2842-2861, paired | l.5-6 | meta arms, records | MATCH | |
| A4 | spec, readings and CHECKs committed at 003a370 / db14e9b before any run | l.6 | `db14e9b9` 22:06:25; first CHECK file 22:14:46; first record commit 22:16:41 | MATCH | |
| A5 | 204/204 runs ok, none retried | l.7 | ok 204; `attempt` = 1 on all 204 | MATCH | |
| A6 | 25.7 CPU-hours | l.7 | Σ `secs_wall` 25.72 h; Σ `secs` 25.30 h | PARTIAL | 25.7 is the summed per-run wall time. At 1 thread that is close to CPU time, but the records' `secs` field gives 25.3. |
| A7 | "The container restarted four times" | l.7 | **5** `=== RESTART` markers in `s89.log` (22:59, 01:22, 05:03, 06:34, 08:01) | **DISAGREE** | r8's own "6 × 106/106" implies 5 restarts (1 + 5 passes). The commit messages name only three. |
| A8 | each resume re-ran all 106 CHECKs (6 × 106/106 ok) | l.8 | 6 `checks_s89_*.json`, 106 rows each, all ok; 636 `CHECK S89:` log lines, 0 FAIL | MATCH | |

### 1b. Counts (r8 l.11-24; chk §3)

| # | statement | recomputed (G's `routed_bound`, `inv_view` on allpool / rlib) | verdict | note |
|---|---|---|---|---|
| B1 | "the pooled form gives the same counts" (ALLPOOL); POOL pooled | other form: identical counts on every arm, both tasks | MATCH | |
| B2 | ORACLE_D 2/2 (2400) POOL, 2/2 (3600) ALLPOOL | 2/2 @2400, 2/2 @3600; η² pooled = within = 1.000, margin 1.000 | MATCH | |
| B3 | BASELINE_D POOL 0/20 [0.00, 0.16] NEVER, BOUND 14/20 (unrouted, margin ~0.45); ALLPOOL 0/20, 0/20; lite 0/10 | 0 / 14 (median margin 0.446); 0 / 0; S87 0/10 | MATCH | |
| B4 | REG3_D POOL 16/20 [0.58, 0.92] MAJORITY, 16; ALLPOOL 0, 0; lite 0/10 | 16 / 16; 0 / 0; 0/10 | MATCH | |
| B5 | REG3_KEYHINGE_D 20/20 [0.84, 1.00], 20; ALLPOOL 0, 2/20 (key split); lite 10/10 | 20 / 20; 0 / 2 (η² by key 0.977, 0.997; margin −0.004, 0.002); 10/10 | MATCH | |
| B6 | REG3_FREQHINGE_D 7/20 [0.18, 0.57] MINORITY, 8/20; ALLPOOL 0, 2/20; lite 10/10 | 7 / 8; 0 / 2; 10/10 | MATCH | |
| B7 | REG3_REPEATHINGE_D 20/20, 20; ALLPOOL 0, 2/20; lite – | 20 / 20; 0 / 2; not run in S87 (no such arm in `s87_results.json`) | MATCH | |
| B8 | transitions: KH 6000 (4800-8400), REPEAT 6000 (4800-10800), REG3 4800 (3600-9600), FREQ 7800 (6000-22800) | identical | MATCH | |
| B9 | POOL unbound: REG3 KEY 4; FREQ KEY 6, OTHER 3, STREAM-PARTIAL 3; BASELINE STREAM-PARTIAL 6 | identical | MATCH | |
| B10 | ALLPOOL unbound "KEY 12-19 per arm, MERGED 2-5" | KEY 19 / 12 / 12 / 17 / 13 (B / R / KH / FQ / RP); MERGED 0 / 4 / 5 / 0 / 2; also OTHER 1 / 3 / 1 / 1 / 3 and STREAM-PARTIAL 0 / 1 / 0 / 0 / 0 | PARTIAL | MERGED appears in 3 of 5 arms. The 9 OTHER runs and the 1 STREAM-PARTIAL run are not mentioned. |

### 1c. Readings (r8 l.26-40; chk §4, §8)

| # | statement | recomputed | verdict | note |
|---|---|---|---|---|
| C1 | POOL: KH discovered, 20/0 vs BASELINE, p 9.5e-7 | 20/0, p 9.54e-7 | MATCH | |
| C2 | FREQ "needs frequent keys": discovered 7/0 p 0.0078; KH only 13, FREQ only 0, p 1.2e-4 | 7/0 p 0.00781; 13/0 p 1.22e-4; `arm_name` → names[2] | MATCH | Mechanical. On the meaning, see §5. |
| C3 | REPEAT "replaces the key hinge": 20/0 vs BASELINE; KH only 0, REPEAT only 0 | identical; `arm_name` → names[0] | MATCH | Mechanical. On the meaning, see §3. |
| C4 | ALLPOOL "the labelled hinge itself fails here" (KH 0/20 = BASELINE 0/20) | 0/0, p 1 | MATCH | Mechanical. On the meaning, see §4. |
| C5 | clamps does not apply (arm only 7 and 20 vs BASELINE only 0) | identical; commitment at 2400: 0.0828 / 0.1034 vs 0.8353 | MATCH | |
| C6 | KH vs REG3_D on POOL 4/0, p 0.0625 | 4/0, 0.0625 (printed by `readings()`, s89:237-239) | MATCH | |
| C7 | REPEAT vs REG3_D 4/0, p 0.0625 | 4/0, 0.0625 | PARTIAL | The number is right, but no committed code computes it: it is not in the log or in `s89_summary.json`. It sits under "Readings (pre-fixed; computed by explore_g_s89.readings)", undated, and is not attributed to V's v2 fix 1. |
| C8 | REG3_D vs FREQ 10/1, p 0.0059 | 10/1, 0.00586 | PARTIAL | As C7. Under fix 1 this would have printed "and the term hurts the register here". |
| C9 | across tasks: BASELINE 0→0; REG3 16→0 (16/0, p 1.5e-5); KH 20→0 (9.5e-7); REPEAT 20→0 (9.5e-7); FREQ not compared | identical | MATCH | |
| C10 | ALLPOOL caveat (pre-fixed) | s89:26-31 and 351-353 at `db14e9b9`; printed in the log | MATCH | |

### 1d. What happened (r8 l.42-67)

| # | statement | recomputed | verdict | note |
|---|---|---|---|---|
| D1 | REG3 16/20, against 0/10 on lite (S87) and 2/20 on Hebbian lite (S82) | 16/20; S87 REG3_D 0/10; `s82_devices_results.json` REG3 2/20 | MATCH | |
| D2 | KH and REPEAT 20/20, four pairs above REG3, p 0.0625 each | 4/0 and 4/0 | MATCH | |
| D3 | the term fires 100-120 times in 1-2400 | median 101.5 (77-155) KH, 116.5 (81-181) RP | MATCH | Medians. |
| D4 | commitment ~0.1 to 2400 | 0.0786 / 0.1034 | MATCH | |
| D5 | β^R at CTX from 0.14 (2400) to 0.5-0.65 | KH 0.14 → 0.51 (4800) → 0.64; RP 0.14 → 0.52 → **0.66** | PARTIAL | Rounding |
| D6 | all three layers route: margin 1.00, contingency 1.0/0.0 at KEY and VAL, ~0.5 at CTX | log diagnostics, identical | MATCH | Per run, CTX separation by stream at L1 has median 0.25 / 0.23 (chk §7). |
| D7 | the repeat hinge selects 5 CTX + 9 KEY + ~2.6 VAL positions, all 8 key-term positions | V's replica gives 16.67 with 2.67 VAL (v2 §2.8); G's CHECKs passed | NOT RECORD-CHECKABLE | |
| D8 | "Its record is position for position the key hinge's (20/20, **same transitions**)" | outcome 20/20 on both; same transition on **10/20** seeds; ranges 4800-8400 vs 4800-10800; identical final acc on 0/20 | **DISAGREE** | Only the per-seed outcome pattern agrees. |
| D9 | given REG3's 16/20, the hinge adds 4 pairs | 4/0 each | MATCH | |
| D10 | FREQ's cut on POOL selects only the 2 CTX types | log CHECK: "pool: types [0, 1] (['CTX'])" | MATCH | |
| D11 | FREQ 7/20, below REG3's 16/20 (10/1) | 10/1 | MATCH | |
| D12 | FREQ commitment 0.08 at 2400 and 0.21 at 3600, against ~0.97 for KH and REPEAT | 0.0828, 0.2124; 0.9698 / 0.9735 | MATCH | |
| D13 | FREQ β^R at CTX 0.22 at 4800, 0.48 at the end | identical | MATCH | |
| D14 | 12 runs plateau at ~0.35 | 12 unbound, median 0.35 | MATCH | |
| D15 | "what S87's SELFKL did on lite" | S87 not re-checked here | NOT RECORD-CHECKABLE | |
| D16 | ALLPOOL: "every learned arm key-splits (η² by key at KEY **0.95-1.0**) or settles on a type-level map" | of 100 learned runs: 73 KEY (η² by key 0.546-1.000, **17 below 0.95**), 6 bound under a key split, 12 at the cap (11 MERGED, 1 STREAM-PARTIAL), **9 OTHER** (η² by key 0.07-0.41) | PARTIAL | |
| D17 | REG3 5, KH 5, REPEAT 2 end with 0.48-0.54 of the probe's sequences routed, margin ~0.5 | 5 / 5 / 2; `seq_routed_frac` 0.477-0.535; margin 0.475-0.535 | MATCH | |
| D18 | those runs' held-out accuracy 0.72-0.92 | **0.570-0.916** (REG3_D 2858 at 0.570; the other 11 are 0.716-0.916) | **DISAGREE** | |
| D19 | no run learned the relative routing | max margin 0.535, max η²_inv 0.582 | MATCH | |
| D20 | the 2 BOUND runs per hinge arm bind late (18000-22800) under a key split with margin 0 | 18000-22800; η² by key 0.977-0.997; margin −0.007 to 0.006 | MATCH | |
| D21 | POOL BASELINE binds without routing: 14 BOUND, 0 ROUTED, margin ~0.45, VAL contingency 0.70/0.30 | 14 / 0 / 0.446 / 0.70 / 0.30 | PARTIAL | G's own "BOUND ROUTED (mine)" column in the same log reads **13/20**. "Without routing" holds under F's ROUTED* only. |
| D22 | unbound ALLPOOL runs sit at 0.39-0.49 held-out | unbound medians 0.392 / 0.434 / 0.479 / 0.409 / 0.445 | PARTIAL | 0.49 is REPEAT's all-run median, which includes the bound runs. |
| D23 | F's report 9: no memory, 10-token reach, ~0.4 | F r9 l.33-34, 171: 0.37-0.41 | MATCH | |

### 1e. CHECKs and "For the main line" (r8 l.69-97)

| # | statement | recomputed | verdict | note |
|---|---|---|---|---|
| E1 | 106 CHECKs, asserted before every segment, all ok | 6 × 106 ok | MATCH | |
| E2 | the vendored module and its prefixes; F's `check_pool_task` and `check_inv_stats`; `routed_bound` = `pool_outcome` | log rows ok | MATCH | |
| E3 | token ids only; the repeat positions; pool's cut CTX only, allpool's VAL only | log rows ok (allpool types 48-63, ['VAL']) | MATCH | |
| E4 | the key term, the statistics and the perfect gate CHECKs; the untrained perfect gate ROUTED* on all three tasks | log rows ok | MATCH | |
| E5 | S87's 7 records reproduce bit for bit through 2400 (same CPU) | log rows ok | MATCH | |
| E6 | "gradients reach the gate only" | log: the reached parameters include `embed.weight` | PARTIAL | s89:107-108 says "(and the embedding through its input)"; r8 drops that. |
| E7 | the reading CHECKs ("weaker" / "survives"; exhaustive) | log rows ok | MATCH | |
| E8 | F's BASELINE -pool 2802 not reproducible | s89:113; log | MATCH | |
| F1 | the margin over no hinge is 4 pairs, not 20; the repeat hinge not run on lite | 4/0; no REPEAT arm in S87 | MATCH | |
| F2 | ALLPOOL: "nothing here routes it, the labelled ceiling included, and the failures sit at the type-level cap" | 0/100 learned ROUTED-BOUND; at the cap 12 of 100 (73 key split, 9 OTHER) | PARTIAL | Most failures are key splits, not cap runs. |
| F3 | F's conv pilot: POOL 2/2, ALLPOOL 1/2, "partly" | V's v2 §2.1 (F's stores) | MATCH | F's stores were not re-read here. |

### 1f. README (l.713-733) and the commit message

| # | statement | recomputed | verdict | note |
|---|---|---|---|---|
| G1 | seeds; perfect gate 2/2 on both tasks; committed before any run; 204/204 ok | as A3-A5, B2 | MATCH | Cites only V's first "fix first", not the v2 review (§2). |
| G2 | table counts (POOL 0 / 16 / 20 / 7 / 20; ALLPOOL 0 × 5) | identical | MATCH | |
| G3 | table readings | as C1-C4 | MATCH | Mechanical. |
| G4 | either hinge adds 4 pairs (p 0.0625) | 4/0 each | MATCH | |
| G5 | the frequency hinge "lowers it to 7/20" | 7/20; REG3_D only 10, FREQ only 1, p 0.0059 | MATCH | |
| G6 | "On ALLPOOL every arm key-splits or stops at the type-level routing cap (0.48-0.54 …; 0.533)" | as D16, D17 | PARTIAL | 9 OTHER runs. |
| G7 | handed back; no Hebbian learned arm | – | MATCH | |
| H1 | commit `8e9e3187` headline: counts and names | as B and C | MATCH | Mechanical names. |

**Totals: 70 rows. MATCH 54, PARTIAL 11 (A6, B10, C7, C8, D5, D16, D21, D22, E6, F2, G6), DISAGREE 3 (A7, D8, D18), NOT RECORD-CHECKABLE 2 (D7, D15).**

## 2. Pre-registration order (chk §0)

| time (UTC) | commit / event | records (learned) |
|---|---|---|
| 10-20 22:03:56 | `003a370e` fix-first code | 0 |
| 22:06:25 | `db14e9b9`: final code (parent blob `c76877d6`, child `cd9998ab`) | 0 |
| 22:14:46 | first CHECK pass (`checks_s89_20261020_221446.json`, git `db14e9b`, 106/106) | 0 |
| 22:16:41 | `0bd95b21`: first records (ORACLE_D) | 2 (0) |
| 22:17:05 / 22:19:56 | `s89_validity.json` written / committed (`6bddedc5`) | 4 (0) |
| 22:20:21 | `eb4d3fa4`: **first learned record**, `pool.REG3_D|2842` ROUTED-BOUND | 5 (1) |
| 22:21:43 | `67f3897c` | 7 (3) |
| 22:26:05 | V's `screen_review_s89_G_v2.md` (review-V `c42ace84`) | G's branch at 7 (3) |
| 10-21 09:42:14 | `d745d869`: last progress | |
| 09:44:52 | `report()` rerun (store provenance `d745d86+dirty`) | 204 |
| 09:46:54 | `8e9e3187`: report 8 + README | 204 (200) |

**The code.** Over all 95 commits there is one blob each for:
- `explore_g_s89.py`, `explore_g_s89_child.py`, `explore_g_rhl_parent.py`, `explore_g_common.py`, `explore_f_tasks.py`, `explore_common16.py`, `explore_common.py`;
- and `git diff --name-only db14e9b9 8e9e3187` lists no `.py` file at all.

So there was no change to the parent's reading code or to the child after the first learned record, or at any time in the run.

**The code SHA.** It is constant: `5d4a9b1dba33` on all 204 record keys and `sha` fields, on all 8 log banners, and in the store meta.

**The restarts.** There were five, each followed by 106/106 CHECKs (A7).

**V's v2 fixes, all parent-side and all landable without a rerun.** None was implemented in code. In the text:

| fix | status in r8 / README |
|---|---|
| 1 REG3_D reading ("register does not need the hinge here"; arm vs REG3_D; "term hurts the register") | **Not in code.** r8 l.34-35 adds REPEAT vs REG3_D and REG3_D vs FREQ by hand under the "computed by explore_g_s89.readings" heading, undated, with no mention of the v2 review. l.90-92 concede the 4-pair margin. The 2b reading itself (KH vs REG3_D, p 0.0625 ≥ 0.05, so it would apply) is not printed. |
| 2 the frequency hinge on -pool is a CTX/stream hinge | **Prose only** (r8 l.54-58). The name "needs frequent keys" is kept in r8 l.30, README l.725 and the commit headline. |
| 3 the repeat hinge's pressure differs across tasks | Not stated. |
| 4 DISCOVERED in the within-sequence form on -allpool | Not taken. `count_table`'s DISCOVERED column on allpool is pooled. Moot here: no bound run is in the cap band (§4). |
| 5 operating characteristics at KH ≈ 1.0 | Not taken (s89:86-89 unchanged). |
| 6 validity keyed to the SHA | Not taken. Moot: no child change. |
| 7 delta-only scope; the re-plan's location | Partly: the title says "delta memory" and l.95 says "Not run: any Hebbian learned arm". The re-plan's location is not cited. |
| 8 a missing REG3_D does not void the task | Moot (no missing run). |
| 9 "paired and beats" | Not taken. |
| 10 F's type-level gate on delta untested | Not stated. |

**Readings added after the records?** None in code. The two hand-computed REG3_D comparisons (C7, C8) are post hoc and not dated as such. Their numbers are right.

## 3. "The repeat hinge replaces the key hinge" (chk §4, §5, §9)

**The rule as committed** (`arm_name`, s89:196-210 at `db14e9b9`, the code at the first record). Let p = p(KH > arm), the one-sided exact McNemar on the 20 paired seeds.
- If p ≤ 0.0625: "needs key identities".
- Otherwise, if the arm is discovered (it beats BASELINE_D, p < 0.05) and (KH only) − (arm only) ≤ 1: "replaces the key hinge".
- Otherwise, if discovered: "is weaker".
- Otherwise: "does nothing here".

**Its inputs, recomputed:**
- REPEAT vs BASELINE_D: 20/0, p 9.54e-7, so discovered.
- KH vs REPEAT: 0/0, p 1, and 0 − 0 ≤ 1.
- So the name is "replaces". The rule as committed supports the name in its literal sense: on POOL the repeat hinge is indistinguishable from the key hinge, with 0 discordant seeds.

**REG3_D binds the same task.**
- POOL: 16/20. It fails on 2843, 2853, 2855 and 2861, all KEY class, and both KH and REPEAT bind all four.
- ALLPOOL: 0/20, like every arm.
- On lite delta (S87): 0/10, against KH 10/10.

**The pairs:**

| pair | first only / second only | one-sided p | two-sided p |
|---|---|---|---|
| REPEATHINGE vs REG3_D | 4 / 0 | 0.0625 | 0.125 |
| KEYHINGE vs REG3_D | 4 / 0 | 0.0625 | 0.125 |
| REG3_D vs FREQHINGE | 10 / 1 | 0.0059 | 0.012 |
| REG3_D vs BASELINE_D | 16 / 0 | 1.5e-5 | |

Unpaired, Fisher 20/20 vs 16/20 gives one-sided p 0.053.

**What "replaces" means here.** The key hinge's own contribution over no hinge is 4/0, p 0.0625, and is not established at 0.05. So "replaces the key hinge" says the repeat hinge matches a hinge that is itself not shown to be needed on this task. V's fix 1 (2b) would print "the register does not need the hinge here" and read the names as descriptive. r8 l.89-92 says this in prose ("margin over no hinge is 4 pairs"); the README, the table and the commit headline keep the bare name.

**Would an inert hinge also read "replaces"?** Not as these records stand. This is a correction to my own v2 review.
- **A hinge with no effect at all:** its records are REG3_D's. Run through `arm_name` it reads **"the repeat hinge needs key identities"**, because 4/0 gives p 0.0625 ≤ 0.0625.
- **A hinge that perturbs training but adds nothing:** model it as an independent draw at REG3_D's rate, with KH 20/20 and BASELINE 0/20. Exactly:

| REG3_D rate | P("replaces") | P("weaker") | P("needs") |
|---|---|---|---|
| 0.8 (observed) | 0.069 | 0.342 | 0.589 |
| 0.85 | 0.176 | | |
| 0.9 | 0.392 | | |
| 0.92 (Wilson upper) | 0.51 | | |

So the name is not what an inert hinge would most likely get. It does, however, rest on the same 4 seeds that separate KH from REG3_D.

**Errors in the referee's own files.**
- `screen_review_s89_G_v2.md` l.45 and l.269 say that an inert hinge "reads 'survives' or 'replaces the key hinge'". The correction at `screen_review_s89_G.md` l.238 says the same. That holds only if REG3_D binds (nearly) every seed. At REG3_D's observed 16/20, an exactly inert hinge reads "needs", and a perturbed inert one reads "replaces" with probability 0.07. My own v2 §2.10 table (pA 0.8: survives 0.069, needs 0.589) already says this. The point of fix 1 stands: the names do not test the hinge against the register. But the stated mechanism was overstated.
- v2 §2.8's "routed solution" is the perfect gate. The learned routed runs on POOL (REG3_D, KH, REPEAT) end with only partial CTX separation by stream: median |Δg| 0.23-0.25 at L1 (chk §7). So the "five times lite's pressure" figure (hinge 0.80) describes the perfect gate, not the solutions these arms learned.

**Verdict on "replaces".** It is supported by the committed rule, and literally true as KH = REPEAT (0/0). It is not supported as a claim that the repeat hinge does necessary work. Neither hinge beats REG3_D at 0.05 (4/0, p 0.0625), so on POOL "replaces the key hinge" cannot be told apart from "both hinges add at most a few seeds to a register that mostly binds alone". The test r8 l.91-92 names, the repeat hinge on lite where REG3_D is 0/10, is the one that would separate them.

## 4. The ALLPOOL null (chk §3, §6)

**What was null.**
- ROUTED-BOUND is 0/20 for every learned arm, in both the within-sequence and the pooled form.
- BOUND is BASELINE 0, REG3 0, KH 2, FREQ 2, REPEAT 2. All 6 bound runs bind late (18000-22800) under a key split (η² by key 0.98-1.00, margin −0.007 to 0.006, `seq_routed_frac` 0).
- Of the 100 learned runs: 73 KEY, 12 at the cap (`seq_routed_frac` and margin 0.477-0.535: REG3 5, KH 5, REPEAT 2; unbound, acc 0.57-0.92), 9 OTHER and 6 bound unrouted.
- The perfect gate binds 2/2 (3600).

**The cap.**
- A fixed CTX-type → channel map separates at most 64/120 = 0.533 of sequences (F's `b0c6332` comment; my F v2 review out §B).
- For such a gate the margin, η²_inv and `seq_routed_frac` each equal the separated fraction.
- ROUTED needs margin ≥ 0.9 and η²_inv > 0.9, so **no gate that is a function of the current context type can be ROUTED on ALLPOOL**. The 12 cap-band runs sit exactly there: max margin 0.535, max η²_inv 0.582.

**DISCOVERED.**
- In the within-sequence form, `val_cos_inv` = 1 − separated fraction. Six cap-band runs have `val_cos_inv` 0.465-0.496 < 0.5, so they would read DISCOVERED but not ROUTED **if they bound**.
- None is bound, so "DISCOVERED but not ROUTED at the cap" did not occur.
- G's DISCOVERED column on ALLPOOL uses the pooled `val_cos` anyway: `inv_view` (s89:156-160) does not swap it, and v2 fix 4 was not taken. There it reads 0.99-1.00 on these runs.

**The perfect gate on ALLPOOL** is the relative-index one, not the type-level one:
- `task.perfect_gate` is the one-hot of `stream_labels`, the stream index by first appearance (`explore_f_tasks.py` b0c6332 l.371-376, 401-409);
- G uses it at child:242-243 and :647;
- its pooled and within-sequence η² are both 1.000, because the relative index is the first-appearance labelling.

So validity says the delta memory can use a relative routing that is handed to it. It does not say that any gate under test can compute one. No type-level or learned-relative gate was run on delta: F's type-level pilot was Hebbian only, which is v2 fix 10, not stated in r8.

**Could a learned arm have been ROUTED-BOUND?**
- Not through a fixed type map (the cap).
- Only through a within-sequence comparison: the relative index, or e.g. "lower CTX id → ch 0", which reads ROUTED in the within-sequence form (my F v2 review §2.2).
- G's docstring (s89:28) says the register could carry the first context. Nothing here shows that the REG3 model can learn it in 24000 updates, and nothing shows that it cannot.
- The key hinge supplies no stream information. It only penalises key identity at KEY, and only through 2400. It therefore cannot push a gate past the cap.
- So ROUTED was reachable in principle, but by nothing these arms were built to learn. In practice the best any arm reached was the cap.

**Verdict on the null.** The ALLPOOL null does **not** distinguish "the hinge fails" from "the task's cap makes ROUTED unreachable for these gates".
- "The labelled hinge itself fails here" is the correct mechanical output: KH 0/20 = BASELINE 0/20.
- As a statement about the hinge it is uninformative: the labelled ceiling, the register alone and every hinge stop at a key split or exactly at the type-level cap.
- The pre-fixed caveat (s89:26-31) and r8 l.93-94 ("ALLPOOL is not a test of hinges") say so.
- The commit headline and the README table carry the name without that qualifier.

## 5. "The frequency hinge needs frequent keys" on POOL (fix 2)

**The rule fired mechanically:** p(KH > FREQ) = 1.2e-4 ≤ 0.0625 (KH only 13, FREQ only 0).

**But on POOL the cut is the 2 CTX types** (log CHECK: types [0, 1], ['CTX']). The term therefore:
- penalises η² of the gate by CTX id at the 5 CTX positions, and on POOL CTX id is the stream index;
- touches none of the key term's 8 positions;
- costs the perfect gate a hinge value of 0.80 (V's v2 §2.8 replica).

The name says the hinge's success depends on keys being frequent. On POOL the arm is not a key hinge at all, so the name cannot be about keys.

**What the records show is that the term hurts the register.**
- REG3_D only 10, FREQ only 1, p 0.0059. Under v2 fix 1 this prints "and the term hurts the register here".
- 6 of FREQ's 7 bound seeds are also REG3_D seeds; seed 2861 is the one FREQ-only seed.
- FREQ's commitment stays low past the window (0.21 at 3600, against ~0.97 for KH and REPEAT), and 12 runs plateau at 0.35.

r8 l.54-58 gives exactly this account ("the term penalises … the stream at the stream tokens"). The S87 SELFKL analogy was not checked here. The account is right; the pre-fixed name kept in the table, the README and the headline is not.

**One more point the records show (chk §7).** The learned routed solutions on POOL (REG3_D, KH, REPEAT) end with only partial CTX separation by stream: median 0.23-0.25. So the frequency term's pressure acts on the path to routing (updates 1-2400), not on a CTX split the final solution needs. Several FREQ runs that do bind end with full CTX separation (1.0 on 2842 and 2856).

**Verdict.** The name "needs frequent keys" is not supported on POOL; "the frequency hinge hurts the register here (it penalises the stream at CTX)" is. The 7/20 is right.

## 6. What could not be checked

- The training itself: no rerun; the child's torch code was not executed.
- D7, the repeat hinge's positions, which come from V's numpy replica, not from records.
- D15, the S87 SELFKL analogy.
- F's conv pilot numbers (F3), taken from my v2 review §2.1.
- The 34-module code SHA was not recomputed from `explore_common16.code_sha`. Instead, no `.py` file changed over the range, and every record, banner and the meta carry the same SHA.
- The CHECK rows were read from the log and the JSON files, not re-executed.
