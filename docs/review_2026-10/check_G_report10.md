# Check: Session G's report 10 (S90, S91) at `claude/explore-G` `cbca3e36`

*Referee V, 22 October 2026. What this covers:*
- *`explore_out/G/report_10.md` ("r10"), the README's S90 and S91 sections (`explore_out/README.md` l.755-793 at `cbca3e36`), and the commit headlines of `2c1913b2` and `cbca3e36`;*
- *the S90 records (`s90_results.json`, 40 runs; `s90.log`; `s90_summary.json`; `checks_s90_20261022_050720.json`) and S87's paired records (`s87_results.json`), read through G's own `records()`;*
- *the S91 records (`s91_results.json`, 44 runs; `s91.log`; `s91_validity.json`; `s91_summary.json`; `checks_s91_20261022_061400.json`), with S89H's store and F's `129215e:explore_out/F/s89_post_results.json` for the perfect-gate comparisons;*
- *the spec draft's revision 2 (`6fc55b17:explore_out/G/spec_draft_register.md`, "spec2:N") where it states S90's decision;*
- *every commit from `b57c1867` (S90's code) to `cbca3e36`: 33 commits.*

*It was read against `check_G_report9.md` (the budget defect) and `screen_review_s90_G.md` (fixes 1-8).*

*The numbers come from `check_G_report10.py` ("chk §N"). The script reads git objects only, writes nothing and prints only. Like `check_G_report9.py`, it execs G's parent modules from blobs statement by statement, with stub modules, and torch is not needed. The modules are `explore_g_s90` at `b57c1867`, `explore_g_s91` at `60b1932a`, `explore_g_s89h`, `explore_g_s89`, `explore_g_rhl_parent`, `explore_g_common`'s store helpers and `test_conv_lr.mcnemar_greater`. The script then runs G's own `report()` of each screen on the committed records, sends the file write to memory, and compares the output with the log line by line. G's S91 `_valid()` reads the committed `s91_validity.json` through the same SHA test.*

## Verdict

**S90: the numbers are right. Both regenerated reports equal their logs, and the spec decision is what the committed rule gives on the full records.**

- **Reproduction.** G's S90 `report()` reproduces the log from the count table onward with 0 differing lines (84 of 84). Its summary equals `s90_summary.json` (chk §2).
- **The spec decision.** "the repeat hinge falls short: the spec carries the key hinge only" is the committed rule's output (s90:99-101) on 40/40 records. V's fixed rule (fixes 1 and 2) gives the same here: the question reads "does not hold", not UNTESTED, and REG3 completed 20/20.
- **No change after the records.** G changed no line of code after `b57c1867`. The spec decision was printed once, at `2c1913b2`, with all 40 records in.
- **The fixes.** Of `screen_review_s90_G.md`'s 8 fixes, none was taken in code. Three were taken in part, in text written after the records (§3).

**S91: complete, unreviewed, and its budget reached every run, but not through fix 8.**

- **No pre-run review.** S91's file `60b1932a` ran to completion (44 runs) with no V review. There is no S91 file on `claude/review-V`.
- **Fix 8 was not applied.** `explore_g_common.py` is blob `a398eb7972` from `b57c1867` to `cbca3e36`, and l.164 still sends `dict(arm=arm, seed=s)`.
- **The budget still reached every run.** S91's own child takes `p.get("iters", 48000 if mem == "H" else 24000)` (`explore_g_s91_child.py:229`), so once again the intended budget equals the default. Every S91 record carries its intended iters:
  - PERFECT_H: 48000 on 2/2. Seed 2961 ran to step 48000 (chk §3).
  - every delta arm: 24000.
- **The harness defect is still latent, and no CHECK covers the budget.** Neither CHECK file (81 and 122 rows) has a row on iters.

**Rows: 78 checked. 55 MATCH, 18 PARTIAL, 4 DISAGREE, 1 NOT RECORD-CHECKABLE.**

The 4 DISAGREE rows:
- "84 runs ok": r10 and the README say 84, the store holds 44, and r10's own "4 validity + 40 delta" is 44.
- Two carried from report 9's budget defect, neither corrected: spec revision 2 still says S89H's "ALLPOOL every arm 0/20 at 48000", and the README's S89H section still says "-allpool (48000)".

**Readings.** On both screens the readings are the committed code's mechanical output:
- S90: REPEAT "is weaker than the key hinge" (3/0, p 0.125); 2b "the hinge is needed over the register" (16/0); "does not hold"; "falls short".
- S91: Hebbian UNTESTED (perfect gate 1/2); delta "no device here", every pair 0/0.

Two of S91's descriptive sentences overreach:
- "The bound runs … unrouted (margin ~0, key split)" is wrong for NOVELTY|2990. That run has margin 0.43, η² by stream 0.56, by key 0.02 and type_map 0.998: a bound type-level map.
- "the match removes the type-level solution" rests on 0/10 against 2/10 (one-sided Fisher p 0.24).

## 1. Statements against the records

### 1a. r10, S90 (chk §1, §1b, §2, §6)

| # | statement | recomputed | verdict | note |
|---|---|---|---|---|
| A1 | lite, Hebbian, k 2, no conv, 24000, SLOW, REG3; seeds 2700-2719 | 40/40 S90 records: task lite, iters 24000, k 2, recipe slow; S87's 60: iters 24000 | MATCH | |
| A2 | Xeon 2.10GHz, torch 2.14.0+cu130, 1 thread | store provenance, log banner | MATCH | |
| A3 | "S90 committed at b57c186 … before any of its runs" | `b57c1867` 05:03:55; CHECK file 05:07:20; first record `5c04a858` 05:13:57 | MATCH | |
| A4 | "paired with S87's records, because this CPU reproduces them bit for bit" | CHECK rows: KH, FQ, BASELINE\|2700 and ORACLE_ALL\|2700 equal **through 2400** | PARTIAL | Fix 4 not taken. |
| A5 | 40/40 new runs ok, none retried; 53 min | ok 40, attempt 1 on 40; "S90 runs done in 52.6 min" | MATCH | |
| B1 | BASELINE (S87) 0/20 [0.00, 0.16] NEVER | identical; KEY 17, STREAM-PARTIAL 3 | MATCH | |
| B2 | REG3 4/20 [0.08, 0.42] MINORITY; KH only 16, REG3 only 0, p 1.5e-5 | 4/20 (2708, 2712, 2716, 2719); 16/0, p 1.526e-5 | MATCH | |
| B3 | KH (S87) 20/20 [0.84, 1.00]; vs REG3 16/0 | identical | MATCH | |
| B4 | FREQ (S87) 17/20 [0.64, 0.95]; 3/0 p 0.125; vs REG3 13/0; "is weaker" | identical (fails 2700, 2709, 2713); 13/0 p 1.22e-4 | MATCH | |
| B5 | REPEAT 17/20 [0.64, 0.95]; 3/0 p 0.125; 13/0 p 1.2e-4; "is weaker" | identical (fails 2700, 2713, 2717); two-sided p 0.25 | MATCH | |
| C1 | first: discovered (20/0) | 20/0, p 9.54e-7 | MATCH | |
| C2 | 2b: the hinge is needed over the register (16/0) | identical | MATCH | |
| C3 | question "does not hold" | identical (`s90_summary.json`) | MATCH | |
| C4 | for the spec "the repeat hinge falls short"; key hinge only (rev 2, 6fc55b1) | the committed rule's output; V's fixed rule gives the same | MATCH | §2 |
| D1 | REPEAT commitment 0.08 at 1200, 0.94 at 2400 | 0.0809, 0.9376 (ROUTED-BOUND runs only: 0.0802, 0.9387) | MATCH | G's medians are over all 20 runs, failures included. |
| D2 | β^R at CTX 0.30 at 2400 | 0.30 (bound only 0.30) | MATCH | |
| D3 | layer 1 routes at KEY and VAL (1.0 / 0.0) | contingency L1 KEY, VAL [[0,1],[1,0]] | MATCH | |
| D4 | "look like the key hinge's" | KH (S87): 0.060 / 0.934, β^R CTX 0.32, L1 the same | MATCH | Not printed by G. Recomputed here (chk §6). |
| D5 | failures 2700, 2713, 2717 key-split; FREQ's 2700, 2709, 2713 | KEY 3 and KEY 3 (G's table) | MATCH | |
| D6 | "FREQ against REPEAT is 1 / 1, so the two hinges are statistically the same arm here" | 1/1, p 0.75 either way | PARTIAL | No equivalence test exists. 1/1 is "no difference shown", not "the same arm". |
| D7 | REG3 4/20 (S82 2/20); its failures key-split, β^R flat ~0.23 | KEY 15 and STREAM-PARTIAL 1 of the 16 failures (2704: margin 0.49, β^R CTX 0.90); median β^R 0.22-0.24 at CTX from 1200 to 24000 | PARTIAL | 15 of 16, not all |
| D8 | "within one pair" fires 39% at a true rate of 0.9 (V's §2.10) | 0.392 | MATCH | |
| D9 | S89H POOL REPEAT 19/20 vs KH 18/20 | `check_G_report9` B5, B7 | MATCH | |
| E1 | CHECKs 81, all ok | 81 rows, 81 ok, git b57c186 | MATCH | |
| E2 | S87's three arms at 2700 and ORACLE_ALL\|2700 "reproduce bit for bit" | rows say "through 2400" | PARTIAL | As A4. |
| E3 | repeat set 16.63 = 4 CTX + 8 KEY + 2.63 VAL + 2 QUERY; the key positions inside it, grouping one to one | CHECK rows; `screen_review_s90_G` §1i | MATCH | |
| E4 | switch off = REG3\|2700 through 2400 bit for bit | row ok | MATCH | |
| E5 | "The term's gradient reaches the gate only." | the row's list includes `embed.weight` and `gate_conv.conv_w` | PARTIAL | Fix 5 not taken. The same as r9 F8. |
| E6 | S89's static CHECKs; readings() hand cases | rows ok | MATCH | |
| E7 | "A first start of S90 was launched by mistake before its commit and was stopped during its CHECK phase … the log was deleted" | no trace in git; the first committed log's banner is `git b57c186` (clean) | NOT RECORD-CHECKABLE | Disclosed by G. Consistent with the record. |

### 1b. r10, S91 (chk §3, §3b, §3c, §4, §5, §7)

| # | statement | recomputed | verdict | note |
|---|---|---|---|---|
| F1 | PERFECT_H 1/2 (2960 @16800; 2961 unbound, 0.9375 at 48000, still rising) | 2960 transition 16800 (stop 19200); 2961 iters 48000, curve to 48000, last 4 evaluations 0.900 → 0.906 → 0.910 → 0.929 → 0.9375 | MATCH | |
| F2 | ORACLE_D 2/2 (3600, 3600) | identical | MATCH | |
| F3 | control 0/10 (KEY 7, MERGED 2); routed 0.00 / 0.53; type map 2/10 | G's table: KEY 7, MERGED 2, **OTHER 1**; 0.00 / 0.527; 2/10 | PARTIAL | Incomplete class list. |
| F4 | MATCHGATE 0/10 (2 BOUND unrouted; KEY 8); 0.00 / 0.00; 0/10 | identical | MATCH | |
| F5 | MATCHGATE_NOHINGE 0/10 (1 BOUND unrouted; KEY 8); 0.00 / 0.00; 0/10 | KEY 8, **OTHER 1**; otherwise identical | PARTIAL | Incomplete class list. |
| F6 | NOVELTY 0/10 (2 BOUND unrouted; KEY 4, MERGED 2); 0.00 / 0.54; 3/10 | KEY 4, MERGED 2, **OTHER 2**; 0.539; 3/10 | PARTIAL | Incomplete class list. |
| F7 | "The 80 Hebbian learned runs were not queued" | no `_H` learned arm in the store or in meta `arms` | MATCH | |
| F8 | "84 runs ok (4 validity + 40 delta), none retried" | **44** records, ok 44, attempt 1 | **DISAGREE** | 4 + 40 = 44. |
| F9 | 140 min | "S91 runs done in 139.5 min" (from the start, CHECKs included) | MATCH | |
| G1 | Hebbian + conv UNTESTED: the perfect gate did not bind 2/2 within 48000; not a negative | `s91_validity.json` H valid false (bound 1, n 2), written 06:26:10 before any learned run; the readings print UNTESTED | MATCH | Pre-registered (s91:27-30). |
| G2 | 2961 "at 0.90-0.94 for the last 8 evaluations" | last 8: 0.904, **0.893**, **0.899**, 0.900, 0.906, 0.910, 0.929, 0.9375; no evaluation ≥ 0.95 | PARTIAL | The range is 0.89-0.94. Binding needs three evaluations ≥ 0.95 (2960: 16800, 18000, 19200). |
| G3 | S89H, same configuration: 2/2 at 6000 and 9600 on 2840-2841 | S89H allpool.PERFECT_H 2840 @9600, 2841 @6000 | MATCH | S89H's ALLPOOL ran at 24000 (r9 defect); both bound before it. |
| G4 | F's pilots: 2/2 at 6000 and 28800 | F `129215e` O_HEBB_CONV_ALLPOOL_48K 2800 @6000, 2801 @28800 (48000) | MATCH | At 24000, F's 2801 was unbound. |
| G5 | delta: no new arm beats the control (all 0 / 0); "no device here" | all pairs 0/0, p 1; readings ['no device here'] | MATCH | |
| G6 | no run above 0.533 except one NOVELTY at 0.54 of 256 | NOVELTY\|2989 138/256 = 0.539; control max 135/256 = 0.527 | MATCH | |
| G7 | type-level maps: control 2/10, NOVELTY 3/10, MATCHGATE 0/10 | identical (type_map 1.000, 1.000; 1.000, 0.998, 1.000; all ≤ 0.008) | MATCH | |
| H1 | \|a_M\| from 0 to ~0.21 at the median (max 0.93) | 0.2090 (max 0.9252) | MATCH | |
| H2 | MATCHGATE key-splits (KEY 8 of 10); none at a type-level map | KEY 8; both bound runs η² by key 0.95 and 0.98; type_map ≤ 0.008 on all 10 | MATCH | |
| H3 | "The match removes the type-level solution without supplying the relative one" | MATCHGATE 0/10 vs control 2/10 type maps; one-sided Fisher p 0.237 | PARTIAL | A causal reading of n = 10 counts. |
| H4 | hinge firings: median 512 under MATCHGATE vs control's 232 | 512.0, 232.0 | MATCH | The mechanism ("pushes apart … pushes back") is not tested. |
| H5 | NOVELTY \|w_N\| 0.25; the control's picture | 0.2500; KEY / MERGED / type maps 3/10 | MATCH | |
| H6 | "The bound runs bind late (18000-22800) and are unrouted (margin ~0, key split)" | 5 bound: 4 have margin ≈ 0 and η² by key 0.95-0.98; **NOVELTY\|2990**: margin 0.43, η² by stream 0.56, by key 0.02, type_map 0.998 | PARTIAL | One of five is a bound type-level map, not a key split. |
| I1 | CHECKs 122, all ok | 122 rows, ok 122, git 60b1932, 457 s | MATCH | |
| I2 | the match by hand, m_0 = 0, signature; novelty by hand, causal, refuses float; strictly causal | rows ok | MATCH | |
| I3 | "at initialisation each new arm equals its control bit for bit, with exactly one extra parameter" | rows for MATCHGATE_H, NOVELTY_H and MATCHGATE_D only | PARTIAL | No row for NOVELTY_D or for MATCHGATE_NOHINGE on either memory. |
| I4 | with the scalar zeroed, equality through 2400 on both memories | 6 rows ok (H and D × 3) | MATCH | |
| I5 | S89H's and S89's REG3_REPEATHINGE\|2842 "reproduce bit for bit through S91's run()" | rows: through 2400 | PARTIAL | |
| I6 | "the new weights receive a gradient and sit in the gate group" | rows for MATCHGATE_H and NOVELTY_H only (max \|g\| 1.6e-7 and 2.5e-11) | PARTIAL | No delta rows. |
| I7 | untrained perfect gate ROUTED* on both memories; type_map 0.02 / 1.00 | rows ok (0.023, 1.000) | MATCH | |
| I8 | inherited S89 and S89H sets, F's conv record | 62 "(S89)" and 27 "(S89H)" rows, including O_HEBB_CONV_ALLPOOL\|2800 | MATCH | |
| I9 | seeds 2960-2991 in no record on any of the 13 branch heads, "checked before the runs" | V, now: 13 remote heads, no store other than G's s91 holds such a seed | PARTIAL | True now. No CHECK row or log line records a pre-run check. |
| J1 | spec rev 2 carries the key hinge only; it dates the decision after S89, S89H and S90 | spec2:11-31 | MATCH | |
| J2 | "The screen's validity failed there, by half an evaluation's margin" | 2961 ended 0.0125 below 0.95 and needed three evaluations at or above 0.95 | PARTIAL | "Half an evaluation" is not a defined quantity. The run would need ≥ 2400 more updates even if it crossed at once. |
| J3 | on delta neither comparing device routes ALLPOOL (10 seeds) | 0/10 within-sequence on both; no run above 0.533 | MATCH | |

### 1c. README, commit headlines, spec revision 2

| # | statement | recomputed | verdict | note |
|---|---|---|---|---|
| K1 | S90: "paired with S87's records (this CPU reproduces them bit for bit …)" | through 2400 | PARTIAL | As A4. |
| K2 | readings and the spec decision committed at b57c186 before any run; 40/40 ok | as A3, A5 | MATCH | |
| K3 | S90 table (0/20; 4/20, 0/16, p 1.5e-5; 17/20, 0/3, p 0.125 ×2) | identical | MATCH | Printed as (arm only / KH only). |
| K4 | fails 2700, 2713, 2717 (FREQ 2700, 2709, 2713); 1/1; beats REG3 13/0; 39% at 0.9 | identical | MATCH | |
| L1 | S91 setup; readings committed at 60b1932 before any run | first S91 record 06:16:24 | MATCH | |
| L2 | "84/84 runs ok" | 44 | **DISAGREE** | |
| L3 | S91 table (1/2 UNTESTED; 2/2; not run ×4; 0/10 ×4) | identical | MATCH | |
| L4 | \|a_M\| 0.21; MATCHGATE key-splits; one NOVELTY run at 0.54 | as H1, H2, G6 | MATCH | |
| M1 | `2c1913b2`: 17/20 vs 20/20 (3 vs 0, p 0.125), "is weaker", "does not hold", "falls short"; REG3 4/20, 16 vs 0 | identical | MATCH | |
| M2 | `cbca3e36`: S91 Hebbian UNTESTED, perfect gate 1/2 within 48000; delta every arm 0/10, "no device here" | identical | MATCH | |
| N1 | spec2:16-27: 17/20 vs 20/20, 3/0 p 0.125; FREQ 17/20, 1/1; frequency set 14 and repeat set 16.6 positions | identical | MATCH | |
| N2 | spec2:13: S90's records "2026-10-22 05:04-05:57 UTC" | records 05:13:57-05:57:34 (runs from 05:07:22) | PARTIAL | 05:04 is the code commit. |
| N3 | spec2:91-92: S89H "ALLPOOL every arm 0/20 at 48000" | 24000 (`check_G_report9` A1) | **DISAGREE** | Report 9's defect, repeated in `6fc55b17`. |
| N4 | README S89H section l.737 "-allpool (48000)", unchanged at `cbca3e36` | 24000 | **DISAGREE** | Carried from `check_G_report9` H1, not corrected. |

**Totals: 78 rows.**
- MATCH 55.
- PARTIAL 18: A4, D6, D7, E2, E5, F3, F5, F6, G2, H3, H6, I3, I5, I6, I9, J2, K1, N2.
- DISAGREE 4: F8, L2, N3, N4.
- NOT RECORD-CHECKABLE 1: E7.

## 2. The spec decision (chk §2)

**The rule as committed.** `spec_decision` (s90:99-101, blob `341c4ff345`, unchanged from `b57c1867` to `cbca3e36`) returns:
- "the spec's label-free arm is the repeat hinge" iff the question is `Q[0]` ("holds, and the hinge is needed here");
- otherwise "the repeat hinge falls short: the spec carries the key hinge only".

**On the records.** The question reads "does not hold":
- REPEAT is 17/20 against KH 20/20, so KH only − REPEAT only = 3 > 1, which is "is weaker", not "replaces".
- p(KH > REPEAT) = 0.125 > 0.0625, so it is not "needs key identities".

The committed rule gives "falls short". This is also V's rule with fixes 1 and 2 in place:
- the question is not UNTESTED;
- REG3 completed 20/20 ≥ 18.

In the terms of `screen_review_s90_G` §1a, c = 17 < 19, so "does not hold" is fixed whatever r is.

**Where it was printed.** r10 l.22, the README, the commit `2c1913b2` and spec2:16-21 all say this, with "the spec carries the key hinge only". The log has one "FOR THE SPEC" line. It first appears at `2c1913b2` (05:57:34), when the store held all 40 records; no intermediate `--report-only` printed one. G's code did not change, so **no part of the decision layer changed after any record.** Spec revision 2 (`6fc55b17`, 05:58:45, 71 s after `2c1913b2`) states the decision dated "2026-10-22, after S89, S89H and S90" (spec2:11-16).

**What the decision means.** It is the rule's output: REPEAT is 3 pairs below a 20/20 ceiling, p 0.125. It is not evidence that the repeat hinge's rate is below 0.9 or below FREQ's. r10 l.32-33 and the README say so, citing V's 39% figure.

## 3. `screen_review_s90_G.md`'s fixes: taken or not

The review was committed at `e20c4e7a`, 05:25:37. At that time 10 S90 records were committed (`afe7a1e6`, 05:24:02), and 16 were committed at `a437e2d6` (05:29:04). Whether and when it reached G is not in the record.

| fix | what | taken? | where, when |
|---|---|---|---|
| 1 | UNTESTED is not "falls short" (`spec_decision`, s90:99-101, docstring s90:48-50) | **no** (code unchanged) | Moot: no REPEAT run is missing. |
| 2 | REG3 ≥ 18/20 for 2b to decide the spec | **no** (code unchanged) | Moot: REG3 20/20. |
| 3 | print REPEAT vs S87's FREQ; state that the repeat set = the frequency set + 2.63 VAL | **in part, as text** | "FREQ vs REPEAT 1 / 1" appears in the README (`2c1913b2`, after all 40 records), spec2:21 (`6fc55b17`) and r10 l.29-30 (`cbca3e36`). Neither the log nor `report()` prints it, and no p is given (0.75 each way). Spec2:23-29 gives both sets' counts and says each contains the key term's 8. It does not state that on lite the repeat set contains the frequency set. r10 adds "statistically the same arm" (D6). |
| 4 | the pairing as run ("through 2400") | **no** | r10 l.8 and l.39 and the README say "bit for bit". The optional KH rerun was not done. |
| 5 | the gradient sentence | **no** | r10 l.44 repeats "reaches the gate only". |
| 6 | the operating characteristics (0.85 → 0.18; spec arm iff REPEAT ≥ 19 and REG3 ≤ 15) | **in part, as text** | "39% at 0.9" is in the README (`2c1913b2`) and r10 l.32-33. 0.85 and the iff rule are absent. |
| 7 | date the design | **in part** | Spec2:11-16 dates the decision after S89, S89H and S90 (after all S90 records). The S90 docstring is unchanged. r10 says only "committed at b57c186 … before any of its runs". That the design and the spec bar were chosen after S89H's records and report 9 is not stated. |
| 8 | the budget fix in `explore_g_common.py:164` before S91, an iters CHECK, the README S89H correction | **no** | `explore_g_common.py` is blob `a398eb7972` at every commit; l.164 still sends `dict(arm=arm, seed=s)`. Neither CHECK file has an iters row. README l.737 and spec2:92 still say 48000 for S89H's ALLPOOL. See §4 for why S91 was not affected. |

None of the fixes changed a rule, and none needed to on these records. Every text that took part of a fix (README, spec revision 2, r10) came after all 40 S90 records, and is a description of them, not a decision rule.

## 4. S91: what ran, and the budget (chk §0, §3)

**No review before the run.** S91's file `60b1932a` (06:06:19) was due a pre-run review. None exists on `claude/review-V`, and the screen ran to completion:

| time (UTC) | event | S91 records |
|---|---|---|
| 05:57:34 | `2c1913b2` S90 done (40) | – |
| 05:58:45 | `6fc55b17` spec revision 2 | – |
| **06:06:19** | **`60b1932a` S91 parent and child** (readings, CHECKs, validity rule) | – |
| 06:11:22 | `4870a65c` log banner (`git 60b1932`), repro_check | 0 |
| 06:14:00 | CHECK file, 122/122 ok (git 60b1932) | 0 |
| 06:16:24 | `1a6bee63`: first records (ORACLE_D 2960, 2961) | 2 |
| 06:26:10 / 06:26:28 | `s91_validity.json` written (H invalid 1/2, D valid 2/2) / committed `5a39c325` | 4 |
| 06:41:31 | `fb771740`: first learned (delta) records | 8 |
| 08:26:53 | `cbca3e36`: report 10, 44 records | 44 |

- **The code SHA.** It is `b06281c9500a` (36 modules) on all 44 record keys, in the meta and in the validity file.
- **No code changed.** No `.py` changed after `60b1932a`.
- **What the SHA covers.** As in S90, the parent `explore_g_s91.py` is not in `sha_rows`; the child and `explore_g_common` are.

**The budget, record by record:**

| arm | intended (s91:88; meta `arms`) | record `iters` | max `stopped_at` | unbound runs stop at |
|---|---|---|---|---|
| allpool.PERFECT_H (2960-2961) | 48000 | 48000 ×2 | 48000 | 48000 (2961; curve's last step 48000) |
| allpool.ORACLE_D | 24000 | 24000 ×2 | 6000 | – |
| REG3_REPEATHINGE_D, MATCHGATE_D, MATCHGATE_NOHINGE_D, NOVELTY_D | 24000 | 24000 ×40 | 24000 | 24000 |

- **Every S91 record ran its intended budget.** The one Hebbian validity run that mattered, PERFECT_H\|2961, trained to 48000 and ended at 0.9375. So "1/2 within 48000" is a statement at the intended budget. This is unlike S89H's "48000", which was 24000.
- **The mechanism is not fix 8.** The harness still drops `iters` (`explore_g_common.py:164`). S91's child defaults by memory: `p.get("iters", 48000 if mem == "H" else 24000)` (`explore_g_s91_child.py:229`), and its run CHECKs pass `iters=2400` or `20` explicitly. So the intended budget again equals the child's default, this time by construction in the child.
- **What would break it.** The defect is latent for any screen whose budget differs from its child's default, such as a rerun of the Hebbian half with a longer validity budget (r10 l.104-105). There is no CHECK on iters.
- **What did not run.** The 80 Hebbian learned runs (2962-2981) were not queued, as the pre-fixed validity rule requires. The delta half (40 runs, n = 10) is descriptive by the docstring (s91:49), and r10 treats it so.

## 5. Order (chk §0)

**S90.**
- `b57c1867` (05:03:55) holds the readings (S89H's function, blob `c31538f1bb`), the spec rule and the CHECKs.
- The first record is at 05:13:57. The last is at `2c1913b2` 05:57:34, with the only `report()` output and the README section.
- Code SHA `5d4a9b1dba33` on 40/40.
- No `.py` changed in `b57c1867..cbca3e36` except the two new S91 files.
- After records, only text changed: the README (`2c1913b2`), spec revision 2 (`6fc55b17`) and r10 (`cbca3e36`). Spec revision 2 is dated; r10 and the README describe records.

**S91.**
- `60b1932a` (06:06:19) holds the readings, the 0.533 line, the type-map count, the validity rule and the CHECKs. It comes 8 min 45 s after S90's last records and 7 min 34 s after spec revision 2.
- The first record is at 06:16:24. Code SHA `b06281c9500a` on 44/44.
- Nothing changed after the first record except records, the log, the summary, r10 and the README.
- S91's design is post-S90 and post-S89H: its background cites S89H's ALLPOOL cap (s91:7-10). It was not tuned on S91 records, since none existed.

## 6. What could not be checked

- **The training itself.** Nothing was rerun, and the children's torch code was not executed. CHECK rows were read from the committed JSON files, not re-executed.
- **The code SHAs.** The 34- and 36-module SHAs were not recomputed. Instead: no `.py` changed over the range, and every record, banner, meta and validity file carries one SHA.
- **S90's mistaken first start (E7).** It left no trace in git.
- **That paired seeds share batches and initial parameters** with S87's records (beyond the CHECKs through 2400), and the decay 0.95.
- **The mechanism sentences in r10** ("the match pushes the gate's key positions apart, and the repeat hinge pushes back"; why S90 and S89H differ). They are descriptive, and nothing tests them.
- **That the seed-collision check ran before the runs (I9).** V's check is of the heads now.
- **Whether `screen_review_s90_G.md` reached G** before `2c1913b2`.

## 7. Errors in the referee's own files

- **`screen_review_s90_G.md` §1g overstates the S91 risk.** It says "For S91 this defect is live … Under the current `explore_g_common.py:164` it would again run 24000". Fix 8 says "S91's 48000 on ALLPOOL … depends on it". Both assumed S91's child would default to `rc.ITERS` (24000). G's S91 child defaults to 48000 for the Hebbian memory (`explore_g_s91_child.py:229`), so S91's budget did not depend on fix 8. The harness defect itself (l.164) is correctly described and still present.
- **`screen_review_s90_G.md` §Status forecast.** It says the runs "should finish around 06:00 UTC". They finished at about 05:57. This was right.
- No other error was found in `check_G_report9.md` or `screen_review_s90_G.md` against these records.
  - S90's outcome falls in the review's §1a map as predicted: c = 17 gives "is weaker" and "does not hold".
  - The review's §1i repeat-set table matches the spec's counts.
