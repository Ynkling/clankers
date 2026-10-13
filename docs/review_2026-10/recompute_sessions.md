# Session reports recomputed from their records (F 3–5, G 3–4, H 4–5, I 1)

*Referee V, 10 October 2026. Revision 9 will quote these numbers.*

**How.** One script per session recomputes every count, discordant pair, p-value, Wilson interval, band, median and range in the reports from the JSON records at a pinned commit. The readings are applied as each screen's docstring fixes them.
- The scripts are `sessions/recompute_{F,G,H,I}.py`; their full row tables are in `sessions/recompute_{F,G,H,I}.md`.
- All four use the stdlib and `rlib.py` only, and read records with `git show <pin>:<path>`.
- Each reported value is typed in with its `file:line`, and the script stops if that line does not contain it. No reported number is an input.
- All four scripts were re-run cleanly on the final tree.
- **Spot-checks.** I verified every DISAGREE that can change a quoted number independently of the session scripts: F seed 350, G S73 5 vs 0, H seed 1326, I seeds 800, 801 and 803.

**Pins** (the branch heads when the work began) **and what moved since.**

| session | pinned | reports | screens | head at 19:10 UTC | in-scope files changed since the pin? |
|---|---|---|---|---|---|
| F | `5b9bc2a` (15:34 UTC) | report_3, report_4, report_5 | S68, S69, S70 · S77 · S76 | `5b9bc2a` | no |
| G | `93422cf` (04:57 UTC) | report_3, report_4 | S71 · S72, S73 | `83699a2` | no (adds S82's spec and code only) |
| H | `df57087` (13:53 UTC) | report_4, report_5 | S78 · S79 | `834581a` | no (adds S84/S85 code; `explore_h_run.py` gains two screen names) |
| I | `a2588da` (16:33 UTC) | report_1 (+ s74_pilot_log.md) | S74 pilot | `d50f00e` | no (main-line sync only) |

## Totals

| session | rows | DISAGREE | PARTIAL | not record-checkable | MATCH | readings affected by a DISAGREE |
|---|---|---|---|---|---|---|
| F | 227 | 3 | 8 | 17 | 199 | none |
| G | 237 | 5 | 10 | 2 | 220 | none |
| H | 261 | 1 | 5 | 4 | 251 | none |
| I | 64 + 19 log rows | 2 + 2 (log) | 8 + 3 | 11 | 43 + 14 | none |
| **all** | **808** | **13** | **34** | **34** | **727** | **none** |

Every pre-registered reading re-derives exactly as the reports print it, in all 14 screens and phases of §2. So does every count, interval, band and p-value behind the readings. All 13 disagreements are in descriptive numbers.

## 1. Disagreements (every one)

| # | file:line (at the pin) | reported | recomputed from the records | consequence |
|---|---|---|---|---|
| F1 | `explore_out/F/report_3.md:16` | S69 Hebbian perfect gate, the seed that did not bind: final accuracy "0.75" | seed 350, last evaluation (48,000): **0.762** | validity unaffected (1/2 bound, rule ≥ 1/2) |
| F2 | `explore_out/F/report_3.md:45` | same number, "the other 0.75" | **0.76** | — |
| F3 | `explore_out/F/report_5.md:54` | delta splits not labelled ok were in runs "that then bound within 4800–8400 updates" | gaps from split to transition: 6000, 13,200, 7200, 2400, 7200, 2400. The range is **2400–13,200**; only 3 of 6 fall inside 4800–8400 | descriptive; no reading |
| G1 | `explore_out/G/report_4.md:25` | BASELINE_D's 11/20 one-to-one includes "near-uniform gates (0.50 vs 0.51)" | the 11 runs are half-routed: larger channel share **0.61–0.88**, margin 0.45–0.51 (as the report's own line 68 says) | descriptive |
| G2 | `explore_out/G/report_4.md:59` | "in 16 of 20 seeds the gate locks into a key split" | KEY: **14** (Hebbian) and **14** (delta) over all runs, or 14 and 12 over unbound runs; v1 and v2 agree. 16 is the count of runs not ROUTED-BOUND | descriptive (part of an unregistered mechanism claim, §3) |
| G3 | `explore_out/G/report_4.md:64` | RESGATE_ONLY_D: "the bound runs are key-split bindings" | **3 of 5** (704, 705, 709); 702 STREAM-PARTIAL, 707 OTHER | descriptive |
| G4 | `explore_out/G/report_4.md:65` | layer-3 probe "≈ 0.50–0.57 in every routed run" | over 47 ROUTED-BOUND runs: K **0.48–0.79**, V 0.49–0.60; six runs above 0.60 at K (e.g. REG3_NUDGE_D 709 0.79, REG3_D 711 0.74) | descriptive; bears on "the gate need not read a tag" (§3) |
| G5 | `explore_out/G/report_4.md:67` | BASELINE_D bound runs' probe "0.87–0.98" | no group gives it: all 14 bound runs K 0.53–1.00, V 0.55–0.99; the 11 half-routed K 0.86–1.00, V 0.79–0.99 | descriptive |
| H1 | `explore_out/H/report_5.md:51` | INT1200's first splits at 2400–3600 fired at "chance accuracy (0.03–0.09)" | probe **0.03–0.11**; held-out 0.06–0.11. Seed 1326 fired at 3600 with probe 0.109 (held-out 0.107) | descriptive; the mechanism sentence around it is unregistered (§3) |
| I1 | `explore_out/I/report_1.md:31` | C2 SINGLE, set-accuracy rise at the transition: 0.10–0.21 | **0.09–0.22** (800 +0.108, 801 +0.087, 803 +0.217) | descriptive |
| I2 | `explore_out/I/report_1.md:31` | layer-3 probe in the evaluation before the transition: 0.52–0.58 | **0.54–0.58** (800 0.577, 801 0.580, 803 0.538); no reading tried gives 0.52 | descriptive |
| I3 | `explore_out/I/s74_pilot_log.md:89-90` | same as I1 | same as I1 | — |
| I4 | `explore_out/I/s74_pilot_log.md:90-91` | same as I2; also "0.98 at the same evaluation" | same as I2; seed 803 reads **0.957** at the transition evaluation and reaches 0.98 only from 4800 | — |

## 2. Readings committed before the first run (every screen)

For each screen: the earliest commit adding the reading text (`git log -S`), against the earliest commit whose store holds a run record. In every case the screen file is unchanged from its readings commit to the pin.

| screen | readings commit (UTC) | first run record (UTC) | margin | before? | provenance notes |
|---|---|---|---|---|---|
| F S68 | `a1112da` 10-09 14:19:50 | `1335f0b` 14:42:55 | 23 min (log starts 3 s after the readings commit) | yes | |
| F S69 | `a1112da` 10-09 14:19:50 | `da5e67e` 19:19:31 | 5 h | yes | |
| F S70 phase 1 | `a1112da` 10-09 14:19:50 | `1335f0b` 14:42:55 | 23 min | yes | |
| F S70 phase 2 | `a1112da` 10-09 14:19:50 | `61ba46e` 22:00:51 | 7.7 h | yes | budget file `rh_lite_budget.json` first at `1335f0b` |
| F S77 | `2210008` 10-10 04:02:34 | `d4dc3bf` 04:25:37 | 23 min (`s77.log` starts 2 s after) | yes | `report_4.md:53` gives the run as 03:58–06:05; nothing at the pin shows a run before 04:02:36, and the last record is at 05:58:11 |
| F S76 phase 1 | `7247535` 10-10 04:05:33 | `dca8ee7` 06:43:54 | 2.6 h | yes | |
| F S76 phase 2 | `7247535` 10-10 04:05:33 | `215a15b` 10:11:21 | 6.1 h | yes | DELTA* fixed in `1db6d28` 08:01:21, before phase 2; the log runs from `1db6d28+dirty` |
| G S71 | `9114843` 10-10 00:23:50 | `191d177` 00:29:07 | 5 min | yes | store git `9114843` |
| G S72 | `26c4ab1` 10-10 00:25:08 | `d212293` 03:40:51 | 3.3 h | yes | store git **`373350f+dirty`**: run from an uncommitted tree |
| G S73 | `26c4ab1` 10-10 00:25:08 | `57308ad` 01:15:12 | 50 min | yes | store git **`40e043d+dirty`**: run from an uncommitted tree |
| H S78 | `fdf7499` 10-10 04:06:28 | `fd8a7a4` 04:21:30 (run start 04:09:14) | 3 min to start | yes | started at `3997ae2`, a CHECK-crash fix (code only, docstring unchanged) |
| H S79 | `fdf7499` 10-10 04:06:28 | `36feb3a` 05:51:38 (start 05:01:28; CHECKs 05:16:35) | 55 min to start | yes | CHECK stamp at `e6db83b+dirty` |
| I S74 C1 | rule `6484af8` 10-10 04:32:12 (rule first in the user's prompt `8be3f98` 03:42) | CHECKs 04:38:21, code `fd8d3b9` 04:38:19; first records `e177e7b` 06:35:40 | 6 min | yes | records' code SHA `49c699613b42` reproduces from `fd8d3b9`'s hashed modules |
| I S74 C2 | constants `830d16d` 10-10 08:22:55 | CHECKs 08:22:58 | **3 s** | yes | code SHA `fa23cf4c1e13` reproduces from `830d16d`; only `explore_i_pilot.py` differs from C1, so the model is unchanged |

**Finding.** Every screen's readings precede its first run, and no docstring changed after its readings commit.

**One provenance gap.** It does not affect the readings, but Revision 9 should not cite these records as reproducible from a commit:
- G's S72 and S73 ran from uncommitted working trees (`373350f+dirty`, `40e043d+dirty`).

**Thin margins.** F's S68/S70 log starts 3 s after its readings commit, F's S77 log 2 s after, and I's C2 CHECKs 3 s after. Each is consistent with "commit, then start". None shows a run before its readings.

## 3. Statements the reports make that no docstring fixed

The session tables list them: F 9, G 7, H 8. I's report has none listed. The ones Revision 9 must not quote as results:
- **F, `report_5.md:60-61`:** "S54's slow delta oracle was a learned-β artefact", stated flatly. No S76 arm varies learned β; at `:44-48` the same report calls it post hoc.
- **F, `report_3.md:29-31` and `report_5.md:36-37`:** descriptive conclusions placed under "Readings (fixed in the docstring before any run)".
- **G, `report_4.md:56-59`:** "what separates a discovering run is the write". It rests partly on the "16 of 20" of G2, which no classification reproduces.
- **G, `report_4.md:63-69`:** "the residual gate does not read the stream tag" and "the stack learns stream-tagged keys, so the gate need not route". These are causal readings of a 2×2 the docstring fixes as descriptive (`explore_g_register3_delta.py:30-33`). G4's probe range also undercuts "every routed run ≈ 0.5".
- **H, `report_4.md:73-76`:** noise "helps a little (+2 vs REF, +1 vs SPLIT2)", on p = 0.25 and p = 0.5. The pre-registered R2 ("noise is needed") did not apply. This mirrors S61's "the noise does not matter".
- **H, `report_5.md:50-56`:** "the check interval works through the same mechanism" (its reading was *inconclusive*) and "six splits instead of three cost nothing" (CAP6's "harmless" reading *did not apply*, per the report's own lines 29 and 40-41).
- **H, `report_5.md:57-58`:** replication advice that "the first eligible check" matters. It was never varied on its own; INT1200 moves interval, first check and gap together.

## 4. A reading that can be read two ways

**G S73, "discovered (register)".** Its docstring fixes the outcome as ROUTED-BOUND (`explore_g_register3_delta.py:24, 26`).
- **Under ROUTED-BOUND:** REG3_D against BASELINE_D gives 4 vs 0, p = 0.0625, so the reading **does not apply**.
- **Under DISCOVERED,** the outcome whose name the reading shares (`explore_g_rhl_parent.py:10`: bound and VAL cos < 0.5): **5 vs 0, p = 0.031**, and it would apply. The extra run is REG3_D 714, discovered but not routed (margin 0.66).
- **Recomputed by me from the records:** DISCOVERED 5/20 against 0/20, 5 vs 0, p = 0.03125.
- **Verdict.** The docstring is explicit, so the pre-registered answer is "does not apply". Revision 9 must quote 4 vs 0, p = 0.0625, and should state the other reading beside it.
- **S72** gives 4 vs 0 under both outcomes.

This is the same kind of ambiguity as REG_D in S56. Specs for the register test on the main line should name the outcome by its definition, not by a word that matches another outcome (see spec reviews, when the specs arrive).

**Key-split counts (the S56 R2 issue)** enter no S72/S73 reading. The descriptive KEY counts depend on whether bound runs are counted, and the reports count unbound runs only:

| arm | unbound only (as printed) | all runs |
|---|---|---|
| BASELINE_D | 6 | 9 |
| REG3_D | 12 | 14 |
| RESGATE_ONLY_D | 5 | 8 |
| LATCH3_D | 9 | 10 |

Revision 9 should say which population it counts.

## 5. Partial matches worth correcting when quoted

Each holds under one reading but not the one the sentence names. The full list is in the session tables.
- **F `report_4.md:27`:** "0.62 − 0.52 = 0.09". The printed numbers subtract to 0.10; the 0.09 is right from unrounded medians (0.0924).
- **F `report_4.md:40`:** "training lowers layer-3 stream content in every arm" holds at KEY. At VAL it fails for S59_GDN (0.562 at the end vs 0.556 at init).
- **G `report_4.md:27`:** "my BASELINE matches F's Hebbian counts". BOUND and ROUTED*@end match; KEY is 16 against F's 19.
- **H `report_4.md:68`:** "8 of 8 firings" re-formed the key split. Only 6 of 8 have a next check (all 6 read 1.00/0.00); 1311 and 1313 at 21,600 have none.
- **H `report_5.md:46` and `:52`:** the "held-out accuracy" ranges are the trigger probe's. Held-out gives 0.06–0.47, and 0.587 for seed 1320 at 9600.
- **I `report_1.md:32`:** "gap closed by 60–80%" has no definition. It reproduces only against seed 802's gap; each seed's own gap gives 44–83%.
- **I `report_1.md:43-44`:** "layers 2–3 in 3/4" holds at layer 3. Layer 2 reaches 0.9 on 1/4.

## What could not be checked

- **CHECK results.** No session stores the individual results. The checks files are written only when every CHECK passes (H: `explore_h_run.py:96-103`), and each says so; the individual lines are in logs.
- **Bit-for-bit reproductions** (H `report_4.md:95-96`; G's start-up check; F's `repro` field for S53/S59/S69): asserted at run time, not stored as comparisons. I did not rerun them, under the rules of this round.
- **I's Bayes bounds** (0.932, 0.936, 0.82): Monte-Carlo constants, not records.
- **Wall-clock spans, thread counts and restarts:** given in the session tables as the logs state them.
- **F `report_5.md:4-5`:** seed block 1200–1259 appears in no JSON under `results/` or `explore_out/` at the pin. Other branches were not scanned.
