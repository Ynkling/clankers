# Screen review: Session F round 7, S90-F (claude/explore-F: module `9017b823`, screen `73749e8a`)

*Referee V, 22 October 2026. Written while phase 1 (the pilots) was running, before any pilot record and before any learned arm.*

**What was reviewed**
- **Module.** `explore_f_tasks.py` at `9017b823`, "tasks l.N" below.
- **Screen.** At `73749e8a`: `explore_f_s90_ctx.py` ("s90 l.N"), `explore_f_s90_ctx_child.py` ("child l.N"), `explore_f_reports7.py` ("rep7 l.N") and `explore_out/F/run_s90.sh` ("run l.N").
- **Prompt.** The main line `5e51a294`, `docs/reading/parallel_sessions_prompts.md` l.1339–1361 ("## Session F — round 7").
- **Earlier files read first.** `screen_review_s89_F.md`, `screen_review_s89_F_v2.md` and `check_F_report9.md`.

**How it was checked.** "out §X" is a section of `screen_checks_s90_F_out.txt`, written by `screen_checks_s90_F.py`. The script:
- reads git objects only, and trains nothing;
- has no torch, so it replicates the generators in numpy (structure only, its own random draws);
- runs F's `decide`, `_arms`, `outcomes`, `reading_task` and `reading_screen`, taken by ast from `73749e8a`, on in-memory stubs, stage by stage as `run_s90.sh` runs them.

**Verdict: the module is sound; fix first before phase 2.**
- **The module.**
  - It is append-only, byte for byte, and the CHECK of that is right.
  - The generators match the prompt.
  - The caps are exact.
  - The within-sequence statistics are invariant to channel and stream relabelling.
- **The readings.** They give every outcome exactly one name: 35,478 + 118,123 combinations, 0 disagreements. The budget reaches every run.
- **Two things must change before the single-channel arms start.**
  1. **Nothing stops the queue before the learned arms.** `run_s90.sh` queues SINGLE_*_<T> in the stage `phase1b`, right after the pilots' decision. Phase 1 started at 05:34:09 with a 1.02 h makespan, so that stage starts at about 06:37 UTC today. The prompt requires V's review before the learned arms (l.1276).
  2. **The screen reading has no same-recipe control on lite.** Without one, "≤ 3/10" cannot be attributed to the context pool. §2.6 explains why; this is a design question for the user.
- **What a reading cannot say.**
  - On -docs, `val_cos_inv_k < 0.5` (a DISCOVERED-style statistic) is passed by gates that separate nothing: a position-only gate and the type-level map at the cap. Margin and `sep_key_inv` are not separation statistics there either. Only `seq_separated_frac` is.
  - No CHECK runs the new-task code path (child l.92–107, `STATS_CTX`) against a record. The three ties run S89's path.

## 1. The module (`explore_f_tasks.py`, `b0c6332` → `9017b823`)

### 1.1 Append-only, and CHECKed

| | value (out §A) |
|---|---|
| `b0c6332` blob | `e24d5e10815e…`: 49,923 bytes, 48,457 characters, 836 lines. It is also the parent `9017b823^`'s blob, and the one G vendored (G's tip has the same blob) |
| `9017b823` blob | `cfe3fe276f7c…`: 73,009 bytes, 1,209 lines. The same blob at `73749e8a` and at every commit to the tip `bc70d814` |
| byte prefix | `new.startswith(old)` True; 23,086 bytes appended (tasks l.837–1209) |
| AST | The first 37 top-level statements are identical by source segment. The 15 appended names (`RandHeaderTaskCtxPool`, `RandHeaderTaskDocs`, `balanced_map`, `type_map_cap`, `inv_from_gate_k`, `ctx_stats`, `write_order_acc_idx`, three CHECKs, …) rebind no earlier name. There are no module-level attribute stores, `setattr` calls or `global` statements |

**Every existing class is therefore unchanged in source and in behaviour.**

**The CHECK** (child l.191–195) asserts that the file equals `9017b82`'s and starts with `b0c6332`'s. It passed (s90.log). One wording slip, the same as report 9 row 53: its label says "48457 bytes", but that is `len(str)`, i.e. characters. The file is 49,923 bytes (fix 9).

### 1.2 The generators (numpy replica, 20,000 sequences per task; out §B)

Both new tasks are lite's batch with only the contexts renamed:
- `make_batch` (tasks l.909–915) draws lite's batch, then the S context types from the 16, ranked by first appearance.
- The keys stay lite's 4 types (ids 16–19). The values are ids 20–35.
- At S = 2, the stream index is -allpool's convention.

All of this is CHECKed (log: `check_ctx_task`, all ok on the three tasks).

| | CTXPOOL (D 2, L 23) | DOCS3 (L 33) | DOCS4 (L 43) |
|---|---|---|---|
| input tokens per sequence: CTX / KEY / VAL | 5 / 9 / 8 | 7 / 13 / 12 | 9 / 17 / 16 |
| KEY frequency (4 types; expected) | 0.1023 | 0.1016 | 0.1012 |
| VAL frequency (16 types) | 0.0227 | 0.0234 | 0.0238 |
| CTX frequency (16 types) | 0.0142 | 0.0137 | 0.0134 |
| F's log (S87 sample, 4096 sequences) | KEY 0.1017–0.1027, VAL 0.0220–0.0236, CTX 0.0135–0.0154 | 0.1013–0.1018, 0.0229–0.0242, 0.0126–0.0145 | 0.1010–0.1014, 0.0233–0.0244, 0.0128–0.0139 |
| **largest-ratio cut** | the 4 KEY types (ratio 4.39). **Its positions are exactly the KEY positions** (the 8 body keys and the query key) in 100% of sequences | the same (4.25) | the same (4.19) |
| **half-mass cut** | 4 KEY + 4 VAL types (mass 0.501) | the same | the same |
| **repeat set** (types occurring ≥ 2 times in the input) | all 5 CTX, all 9 KEY, and 2.64 of 8 VAL positions. Exactly the KEY positions in 0 / 20,000 sequences; exactly KEY + CTX in 0.158 | 7 + 13 + 5.59 of 12 VAL. 0 / 20,000; KEY + CTX 0.006 | 9 + 17 + 9.24 of 16 VAL. 0 / 20,000; KEY + CTX 0.000 |
| P(index 0 \| position 0, 1, 2, 3) | 1, 1, 1, 0.78 | 1, 1, 1, 0.74 | 1, 1, 1, 0.72 |
| queries asking their key's last write | 0.497 | 0.335 | 0.251 |

**The half-mass cut is arbitrary in its values.** The expected mass of 4 KEY + 4 VAL types is exactly 0.5 at every D: it is (5S + 1)/(10S + 2). The four most frequent value types exceed the mean, so the cut always stops at 4 VAL types (200 of 200 resampled samples). But all 16 value types have the same expected frequency, so *which* four is sampling noise.

**The repeat set never coincides with the key positions.** The contexts always repeat (each opens two blocks, and one is queried), and so do colliding values. On -ctxpool the repeat set is lite's exactly: CTX 5 + KEY 9 + about 2.6 VAL. That is the composition G's spec revision 2 states.

**A last-write-only memory** scores about 1/D. That is below the 0.95 bar, so it cannot read "bound".

**F's printed figures** (s90 l.20–25, rep7 l.47–59) match the replica.

### 1.3 The cap of a fixed type → channel map (out §C)

`type_map_cap` (tasks l.966–972) maximises e_D(n₁…n_k)/C(16, D) over every split of the 16 types. The script recomputed it over every composition of 16 into k parts, and at D = 2 also by brute force over all 2¹⁵ maps.

| | k = 2 | k = D | k = 2D | k = 16 | balanced map, replica (k = D) | F's log (4096) |
|---|---|---|---|---|---|---|
| D = 2 | 64/120 = 0.533 | 0.533 | 96/120 = 0.800 | 1 | 0.535 | 0.527 |
| D = 3 | **0** | 150/560 = 0.268 | 372/560 = 0.664 | 1 | 0.273 | 0.274 |
| D = 4 | **0** | 256/1820 = 0.141 | 1120/1820 = 0.615 | 1 | 0.143 | 0.136 |

**F's numbers are exact** (s90 l.27–30; the CHECK lines). The best split is the balanced one.

Not printed: at k = 2 (G's gates) the cap on -docs is 0.

### 1.4 Do the swap-invariant statistics apply, and does DISCOVERED-at-the-cap recur? (out §D)

`inv_from_gate_k` (tasks l.976–1000) and the margin were replicated on synthetic one-hot gates (5,000 sequences per task).

**The three statistics:**
- `sep_key_inv`: the mean pairwise TV distance between the streams' mean KEY gates.
- `seq_separated_frac`: S distinct argmax channels, each with mass ≥ 0.9.
- `val_cos_inv_k`: the mean pairwise cosine of the mean VAL gates.

All three are invariant to permuting the channels and the stream indices within a sequence. The per-sequence-permuted perfect gate scores 1 / 1 / 0 on every task, as the CHECK asserts (tasks l.1178–1182).

Each cell is `sep_key_inv` / `seq_separated_frac` / `val_cos_inv_k` / margin:

| gate | CTXPOOL | DOCS3 | DOCS4 |
|---|---|---|---|
| relative (perfect), or permuted per sequence | 1 / 1 / 0 / 1 | 1 / 1 / 0 / 1 | 1 / 1 / 0 / 1 |
| balanced type map, k = D (the cap) | 0.532 / 0.532 / **0.468** / 0.532 | 0.709 / 0.273 / **0.291** / 0.714 | 0.801 / 0.143 / **0.199** / 0.800 |
| one channel per type, k = 16 | 1 / 1 / 0 / 1 | 1 / 1 / 0 / 1 | 1 / 1 / 0 / 1 |
| position only | 0.554 / 0.328 / 0.527 / 0.000 | 0.665 / 0.062 / **0.362** / 0.013 | 0.740 / 0.011 / **0.271** / 0.003 |
| first document vs the rest | (= relative) | 0.667 / 0.000 / **0.333** / 0.661 | 0.500 / 0.000 / 0.500 / 0.511 |
| all on one channel | 0 / 0 / 1 / 0 | 0 / 0 / 1 / 0 | 0 / 0 / 1 / 0 |

**DISCOVERED-at-the-cap recurs, and gets worse with D.** For a fixed one-hot map, `val_cos_inv_k` is P(two of the sequence's contexts share a channel): 0.467 / 0.292 / 0.200. So "val cos < 0.5" holds for any fixed map at the cap, and at D ≥ 3 also for a position-only gate (separated 0.062 / 0.011). At D = 3 it also holds for a gate that only marks the first document (separated 0).

**At D ≥ 3, three statistics stop measuring separation.**
- `sep_key_inv` reaches 0.67–0.80 without separating.
- The margin reaches 0.66–0.80 without separating.
- `val_cos_inv_k < 0.5` is met without separating.

**Only `seq_separated_frac` is a separation statistic on -docs.** The module comment says only that "eta² … is not used at S > 2" (tasks l.874–875). It does not say any of the above (fix 5).

**S90 reads none of these** (its readings are on BOUND only), so no reading is affected. But rep7 l.92–108 prints `val_cos_inv_k` beside the TL arms, and G may vendor the statistics.

**The pooled statistics that remain convention-dependent.**
- `ctx_stats` (tasks l.1015–1030) sets only `key_sep` and `key_part` to NaN.
- Still pooled under the first-appearance index, and compared on streams 0 and 1 only at D > 2:
  - `gate_stats_k`'s `sep`, `key_cos`, `val_cos` and `ctx_cos` (test_router_reliability l.257, l.264; `role_cosines`, test_multilayer_binding l.360–370);
  - `pool_routing_stats`'s `eta_*`, on channel 0 only (tasks l.430–435);
  - `run_one`'s `rec["discovered"]` (test_router_reliability l.338).
- On the type-level gate these give pooled VAL cos ≈ 1, as on -allpool (screen review S89-F §7.1).
- Unread by S90, but stored under the main line's names (fix 6).

## 2. The screen `explore_f_s90_ctx`

### 2.1 Every outcome has exactly one name (out §E)

The functions were run, not read, and staged as `run_s90.sh` stages them: import → run the queued arms → a fresh import → `--decide`. The stages are phase0, phase1, phase1b, phase2 and final.

**Task readings.** For each (task, memory):
- the target task's 4 phase-0 accuracies in {0.6, 0.61, failed ×3};
- the 2 pilots at 24K, each bound / unbound / failed ×3;
- the 2 pilots at 48K, the same.

| reading | combinations |
|---|---|
| solvable without memory | 28,470 |
| the task is valid | 96 |
| the task needs the longer budget | 768 |
| untested | 6,144 |
| **total** | **35,478, 0 disagreements** with s90 l.49–56 applied independently; no errors |

- An accuracy of exactly 0.6 reads "not stopped" ("exceeds"; `x > NOMEM_THR`, l.249).
- A run that failed three times counts as not bound, or as accuracy 0 in phase 0 (`outcomes`, l.215–229), as l.54–55 say. **S89's gap (a), the unnamed failure, is closed.**
- A memory that bound 2/2 at 24K is not rerun at 48K (l.136–138, l.274–275). **S89's gap (b) is closed.**

**Screen reading.** SINGLE_DELTA_CTXPOOL's 10 runs, over all 3¹⁰ complete patterns plus missing records, at each budget state of (CTXPOOL, DELTA):

| budget state | reading | combinations |
|---|---|---|
| 24000 or 48000 (each) | one channel binds without a context identity | 1,161 |
| | the partition's claim survives the context pool | 33,024 |
| | no reading applies | 24,864 |
| | INCOMPLETE, no reading (a missing record) | 11 |
| UNTESTED or STOPPED | untested | 1 each |
| none | INCOMPLETE | 1 |

**Total: 118,123, 0 disagreements** with l.67–72.

**The perfect-gate conjunct.** "While the delta perfect gate is 2/2" holds by construction: the reading needs an int budget, which needs the pilot 2/2 at that budget.

**A STOPPED task reads "untested"** in `reading_screen` (l.296–298), not "solvable without memory". This is moot, because phase 0 stopped no task (fix 10).

**Outcome functions.**
- BOUND is `run_one`'s transition (test_router_reliability l.284–339, blob `8929c2d857`, the same as the main line's) through `test_binding_onset.onset_run` (blob `3179ec974e`, the same as the main line's).
- The readings are `reading_task` (l.288–292) and `reading_screen` (l.295–304), on `outcomes` (l.215–229).
- 25 of the 27 SHA-covered modules are main-line blobs or branch-only. Two are older main-line versions:
  - `test_binding_capacity` lacks the main line's later `s_active` knob (inert at None);
  - `test_short_conv` also differs.

  Neither was a reading input here (out §H).

### 2.2 Seeds 3060–3099 (out §G)

**Tips.** All 13 remote branch tips were searched (json, jsonl and log files) for `"ARM|30xx|"`, `"seed": 30xx` and `seed 30xx`. The only hit is `explore_out/F/s90.log`, which is F's own phase 0 on 3060–3061.

**History.** 5,126 distinct json/jsonl/log blobs reachable from the remotes were searched. The only hit is the same log.

**The main line's seed list.** The literal `range()`s in the 38 `test_*.py` files cover 910 integers, none in 3060–3099. The restart tests' actual attempt seeds, s + 1000j with j < 5, have these bases:
- 160–189 (test_load_curriculum l.259–261);
- 200–219 (test_curriculum_confirm l.254–257);
- 240–259 (test_stream_recipe l.278–280);
- 260–279 (test_stream_curriculum l.268–269).

test_router_reliability's 1000–1149 is a separate block. None of these reaches 3060–3099.

The naive "every literal + 1000j" union hits 3060–3099 only through bases 60–99, which no restart test uses.

The round's blocks are disjoint: G 2960–3019, H 3020–3059, F 3060–3099. **s90 l.15–17 is right.**

### 2.3 The CHECKs and the recorded runs (out §I)

**The three ties** (child l.222–244) are pinned to commits, which takes S89 fix 3. Each passed on the 2.10 GHz Xeon:

| tie | record |
|---|---|
| `NOMEM_SELF_LITE\|2800` | `ea9728c636c6`, recorded on 2.80 GHz, at `52e2ea7` |
| `O_DELTA_ALLPOOL\|2800` | the same |
| `O_HEBB_CONV_ALLPOOL\|2801` | `267d08456c38`, 2.10 GHz, at `3e3b1bf` |

- **What each compares.** The curve and every recorded statistics field through 2400. The log's printed references equal the records at those commits.
- **Cross-CPU.** Two of the records reproduce bit for bit across CPUs.

**What is not tied.**
- All three ties route through S89's path: `make_model` l.89–91 calls `s89.make_model`, and `stats_for` l.112 gives `s89.STATS_POOL`.
- So the code the new runs execute is run by no tie: child l.92–107 (`build` → `AllpoolPerfectBDH` / `TypeGateK` → `to_delta_fast`), and `STATS_CTX`.
- The docstring says so ("with the new tasks' code disabled", s90 l.85). The new branch is the same composition as `s89.make_model` l.133–145, but by text only. It can be tied on -allpool (fix 3).
- `TypeGateK` and the SINGLE path have no record anywhere. Their CHECKs are structural (child l.245–270), and s90 l.87–88 says so.

**Code SHA.** `809fde1f5168dcb5…`, recomputed from the 27 blobs at `73749e8a` and at the tip. It equals all 12 records.

**One blob each from `73749e8a` to the tip:** `explore_f_s90_ctx.py`, `_child.py`, `explore_f_reports7.py`, `run_s90.sh` and the module. The parent (where `decide` lives) is not in the SHA. Phase 1 started on "a4d14c2+dirty", as S89 did.

### 2.4 The budget reaches each run (G's S89H bug)

**The path.**
1. The parent's `run_job` passes `iters=ARMS[arm]["iters"]` (s90 l.153–154).
2. `_job` calls it (explore_common16 l.177).
3. The child's `run` reads `p["iters"]` (child l.125) and passes it to `run_one` (l.130).
4. `run_one` stores it (test_router_reliability l.335).

**In the stage simulation:**
- SINGLE_<M>_<T> exists iff the reading is valid or needs the longer budget, with `iters` equal to that budget (out §E: True).
- It is first queued in the stage **phase1b** at 24000 (96 of 96) and in phase2 at 48000 (768 of 768).

**On record.** All 12 phase-0 records have `iters` 24000 and a last evaluation at 24000. No CHECK asserts `rec["iters"] == ARMS[arm]["iters"]` (fix 12).

### 2.5 The no-memory check, the operating characteristics and label leaks

**Phase 0's meaning (out §B).** F took S89 fix 5: s90 l.35–37 print a bound and say "not stopped" is no evidence beyond it. F's bound counts the target pair only when its KEY is inside the reach.
- The generous form, which counts the target VAL inside the reach, gives at reach 10: 0.503 / 0.365 / 0.291 (CTXPOOL / DOCS3 / DOCS4). F's form gives 0.415 / 0.308 / 0.250; the log has 0.423 / 0.310 / 0.250.
- At reach 4 the two forms agree: 0.180 / 0.152 / 0.134.
- Either way the 0.6 rule cannot fire. The recorded NOMEM_SELF accuracies sit under both forms: 0.366 / 0.365 (CTXPOOL), 0.190 / 0.200 (DOCS3) and 0.165 / 0.121 (DOCS4).

**Operating characteristics** (exact binomial on 10 seeds; out §F):

| true rate p | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|
| P(≥ 7/10): "binds" | 0.000 | 0.001 | 0.011 | 0.172 | 0.650 | 0.879 | 0.987 |
| P(≤ 3/10): "survives" | 0.987 | 0.879 | 0.650 | 0.172 | 0.011 | 0.001 | 0.000 |
| P(4–6): no reading | 0.013 | 0.120 | 0.340 | 0.656 | 0.340 | 0.120 | 0.013 |
| P(valid: 2/2) | 0.01 | 0.04 | 0.09 | 0.25 | 0.49 | 0.64 | 0.81 |

- The lines separate a rate of at least 0.8 from one of at most 0.2 almost surely.
- At 0.5 the outcome is "no reading" two times in three.
- The validity rule's P = p² is printed by F (l.55–56, 0.81 at 0.9).
- The screen does not print these figures (fix 4).

**Label leaks: none in the learned arms.**
- SINGLE is arm B: gate `"none"`, k = 1 (test_short_conv l.226–232).
- The loss is `task.select` at `qpos`; `onset_run` discards the query stream.
- `task.ctx_tokens = range(16)` (tasks l.896) is a role label. It is read only by the `'perfect'` gate kind, and the class swaps to `AllpoolPerfectBDH` / `TypeGateK` override that.
- `write_order_acc_idx` reads `stream_labels` after training, as a measurement.
- The perfect gate reads the index by design. The type-level gate reads token identity through a fixed map and no label.

### 2.6 The screen's control: what "≤ 3/10" can be attributed to

The prompt's reference is "on lite one delta channel bound the two-stream header layout 8/10 (Revision 10 §11)". That is **S53 (P = 4, the header layout)**, under another recipe (README l.250, l.312).

F's own single-delta reference on *this* lite layout is **S88b FIXED_1X 9/10** (seeds 2420–2429, `HorizonMemory`, k = 1). It ran under **"Adam 1e-3 throughout"** (s88b.log).

The S90 SINGLE arms run under **SLOW** (s90 l.59–60). Lite → -ctxpool is therefore not a one-variable change: the contexts and the schedule both change. E's S69 found that SLOW lowered binding where nothing gates.

A "≤ 3/10" on -ctxpool would read "the partition's claim survives the context pool" without showing that one SLOW delta channel binds lite at all. The prompt fixed the reading, so this is the user's call (fix 2).

## 3. Is -ctxpool usable by G's S91?

**Technically yes.**
- At S = 2, `ctx_stats` records `eta_key_by_stream_inv`, `val_cos_inv` (via `pool_routing_stats_inv`) and `seq_separated_frac`. So G's within-sequence ROUTED-BOUND (`pool_outcome(rec, True)`, tasks l.725–737) applies unchanged.
- The keys are lite's 4 types, so G's key and repeat hinges see lite's set.
- The cap is the same 0.533 as -allpool.
- The first-appearance index is -allpool's convention (CHECKed).
- G's S89H configuration (Hebbian + conv 'layer', k = 2, SLOW) is exactly F's `O_HEBBCONV_CTXPOOL` pilot.
- -ctxpool is the cleaner task. Against lite it changes only the contexts; -allpool also changes the keys.

**Conditions.**
- **(a) Validity.** F's `O_HEBBCONV_CTXPOOL` must read valid or needs the longer budget. It is running now. G still needs its own 2/2 on 2960–2961.
- **(b) The round plan's wording.** The plan says "-ctxpool … if it is valid and -allpool is not" (prompt l.1361). -allpool read **"needs the longer budget" (2/2 at 48000)** for Hebbian + conv (F report 9 addendum `3e3b1bfe`; explore_g_s89h l.9). So by the letter S91 stays on -allpool. Using -ctxpool instead, or both, is a user decision.
- **(c) Vendoring.** G must vendor blob `cfe3fe27`. G's tip still holds `e24d5e10`. G must replace its blob-equality CHECK with "starts with `b0c6332`'s bytes".
- **(d) What G must print and avoid.**
  - Print `seq_separated_frac` beside DISCOVERED: the cap map reads `val_cos_inv` 0.468.
  - Do not read `rec["discovered"]` or the pooled `val_cos`.

## Fixes (most important first)

1. **Gate phase 2 on the review (urgent: the stage phase1b begins at about 06:37 UTC).** In `explore_f_s90_ctx.py` l.139–142, change
   > `if isinstance(b, int):`

   to
   > `if isinstance(b, int) and os.path.exists(os.path.join(fc.F_DIR, "s90_phase2_go")):`

   and add to the docstring after l.63:
   > Phase 2 is queued only after V's review of this file (explore_out/F/s90_phase2_go, committed by hand).

   The parent is not in the code SHA, so no record is re-queued.
2. **The lite control (the user's decision, before phase 2).** Add `SINGLE_DELTA_LITE` and `SINGLE_HEBBCONV_LITE` on 3062–3071, SLOW, at CTXPOOL's delta budget. The child already supports them, through `s89.make_model` and `single_stats`. Then:
   - (a) Change l.69–70 to:
     > "the partition's claim survives the context pool" if SINGLE_DELTA_CTXPOOL binds <= 3/10 while SINGLE_DELTA_LITE binds >= 7/10 on the same seeds and the delta perfect gate is 2/2 on CTXPOOL; if SINGLE_DELTA_LITE binds < 7/10, "no reading applies (one SLOW delta channel does not bind lite here)".
   - (b) Or, if the user keeps the prompt's reading, add after l.63:
     > The lite references (S53's 8/10 on the header layout; S88b FIXED_1X 9/10 on this lite layout) ran under one Adam at 1e-3, not SLOW; a <= 3/10 here changes the contexts and the schedule together.
3. **Tie the new-task path.** In the child, give `make_model` a `force_new=False` argument and use `if not force_new and sp["task"] in s89.TASKS and …` at l.89. Add a tie that runs `O_DELTA_ALLPOOL|2800` (`52e2ea7`) and `O_HEBBCONV_ALLPOOL|2801` (`3e3b1bf`) through l.92–107 with `stats_fn=STATS_CTX`. It should compare the curve and every recorded field except `key_sep` / `key_part`, which `ctx_stats` sets to NaN. Docstring l.85–88:
   > ties … and the new tasks' path (child l.92-107 with ctx_stats) on S89's ALLPOOL records, with key_sep and key_part excluded.
4. **Print the operating characteristics** with the screen reading (after s90 l.72):
   > On 10 independent seeds: P(>= 7/10) = 0.99 / 0.88 / 0.65 / 0.17 at a true rate 0.9 / 0.8 / 0.7 / 0.5; P(<= 3/10) = 0.99 / 0.88 / 0.65 / 0.17 at 0.1 / 0.2 / 0.3 / 0.5; at 0.5 the screen reads "no reading applies" with probability 0.66.
5. **Say what the -docs statistics measure.** Add after tasks l.873:
   > For a gate that is one-hot per type, val_cos_inv_k = P(two of the sequence's contexts share a channel): 0.467 / 0.292 / 0.200 for the balanced map at D = 2 / 3 / 4, below 0.5 while it separates only 0.533 / 0.268 / 0.141 of sequences. At D >= 3 a position-only gate also reads val_cos_inv_k 0.36 / 0.27 and sep_key_inv 0.67 / 0.74 while separating 0.06 / 0.01. Only seq_separated_frac measures separation at D >= 3; no DISCOVERED-style rule on val_cos_inv_k or sep_key_inv is valid there, and at D = 2 DISCOVERED is read only beside seq_separated_frac.
6. **NaN the pooled fields that the first-appearance index makes conventional.** In `ctx_stats` (tasks l.1023), change
   > `("key_sep", "key_part")`

   to
   > `("key_sep", "key_part", "sep", "key_cos", "val_cos", "ctx_cos") + (() if task.S == 2 else tuple(f for f in out if f.startswith("eta_")))`

   Have the report ignore `rec["discovered"]` on these tasks. Alternatively, keep the fields and say in the module comment that they are pooled under the index and, at D > 2, cover streams 0 and 1 only.
7. **The positional bound (s90 l.35–37).** Replace "P(the target pair, its KEY and VAL, inside the reach)" with
   > P(the target VAL inside the reach) (0.12 / 0.47 at reach 4 / 10 on CTXPOOL; bound 0.18 / 0.50)

   or keep F's form and call it "approximate (it does not credit a visible VAL whose KEY is outside the reach)".
8. **Add the negative control** to `check_inv_stats_k` (tasks l.1166–1196; residual R2 of the v2 review). The all-on-one-channel gate must score `seq_separated_frac` 0 and `val_cos_inv_k` 1. The position-only gate must score `seq_separated_frac` < 0.4 at D = 2 and < 0.1 at D ≥ 3.
9. **Child l.194–195:** use `len(prev.encode())` and say "bytes", which is 49,923, or say "characters".
10. **`reading_screen` (s90 l.296–298):** return "solvable without memory" when `b` starts with "STOPPED" (moot this run).
11. **Publish the blob** for G in the report: `explore_f_tasks.py` at `9017b82`, blob `cfe3fe276f7c…`, whose first 49,923 bytes are `b0c6332`'s.
12. **Assert the budget on record** in `report_s90`: for every record, `r["iters"] == s.ARMS[arm]["iters"]`. Also fix rep7 l.125: `w.get(f) or 0` averages a missing split as 0. Average only the non-None values.

## Run status (tip `bc70d814`, 05:44 UTC)

- **Phase 0 is complete:** 12/12 ok, 24000 updates each, code SHA `809fde1f5168`, CPU 2.10 GHz.
  - NOMEM: 0.168 / 0.166, 0.123 / 0.123, 0.110 / 0.110.
  - NOMEM_SELF: 0.366 / 0.365, 0.190 / 0.200, 0.165 / 0.121.
  - No task stopped (decision committed at `a4d14c24`, 05:34:06).
- **Phase 1** (24 pilot and type-level runs) started 05:34:09, projected makespan 1.02 h. There are no pilot records yet.
- **No learned arm has run.** Without fix 1, SINGLE_*_<T> at 24000 starts with the stage phase1b at about 06:37 UTC.
