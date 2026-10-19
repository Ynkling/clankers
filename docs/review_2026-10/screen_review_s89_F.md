# Screen review: Session F round 6, S89-F (claude/explore-F at `8ba99bb`, task module at `50dc8e4`)

*Referee V, 19 October 2026, written while phase 0 was running and before any pilot or baseline. The screen is `explore_f_s89_tasks.py` (the docstring holds the readings) and `explore_f_s89_tasks_child.py` at `8ba99bb`, with `explore_f_reports6.py` (the reading expression) and `explore_f_tasks.py` at `50dc8e4` (RandHeaderTaskPool, RandHeaderTaskAllPool and the helpers, which G and H vendor). "s89 l.N" means `explore_f_s89_tasks.py`, "child l.N" means `explore_f_s89_tasks_child.py`, "rep6 l.N" means `explore_f_reports6.py`, all at `8ba99bb`. "tasks l.N" means `explore_f_tasks.py` at `50dc8e4`. The prompt is `docs/reading/parallel_sessions_prompts.md` at `7c4ceb8`, l.1115–1162. "out §X" is a section of `screen_checks_s89_F_out.txt`, written by `screen_checks_s89_F.py`. The script:*
- *reads git objects only and trains nothing;*
- *runs `decide()` exec'd from `8ba99bb` on stub stores;*
- *replicates the three generators in numpy for their distributions, because no torch is installed here. The replica reproduces F's printed frequencies and cuts (out §E).*

*The branch tip has since moved to `0378b3a` with progress files only: the phase-0 log and the runtime projection. The code is unchanged.*

**Verdict: fix first.** The pool task and the checks of its structure are sound, and so is the readings' mechanics. Three things must change before the learned arms (phase 2) start. The models are not saved, so statistics cannot be recomputed after a run, and the queue script (`explore_out/F/run_s89.sh`) starts phase 2 automatically after the pilots.

1. **On ALLPOOL the outcome statistics measure the label convention, not the routing.**
   - The stream index is assigned by first appearance (tasks l.272–275, l.401–409).
   - η² by stream at KEY and VAL cos are pooled over sequences (tasks l.432–436; main-line `role_cosines`).
   - A gate that separates the two streams perfectly in every sequence but labels them by context type scores η² 0.000 and VAL cos 1.000: not ROUTED and not DISCOVERED. On POOL the same gate scores 1.000 and 0.000 (out §E).
   - ROUTED-BOUND and DISCOVERED on ALLPOOL therefore require one specific labelling. G's and H's readings inherit this.
2. **On ALLPOOL no fixed map from context type to channel routes more than 8/15 = 0.533 of sequences.**
   - So any gate that reads token identity, as LOCAL3 and the register gates do on lite, is capped there.
   - The validity pilot's perfect gate is handed the per-sequence relative index, so "the task is valid" does not say that a learned gate *can* route ALLPOOL.
   - A null on ALLPOOL is then confounded: no frequent keys, or no fixed context identity. This needs to be said in the docstring, or the task changed (fix 2).
3. **No CHECK ties the new child's code path to a recorded run.**
   - Both reproductions run `explore_f_followup_child.run`, S70's path, not `explore_f_s89_tasks_child.run`.
   - G's reference record is read from the moving branch tip rather than the commit.

The rest:
- The readings give every complete outcome exactly one name: 24,576 combinations, 0 disagreements with the docstring (out §A). A failed record leaves the task with no reading.
- The seeds are free (out §C).
- Phase 0's stop rule cannot fire, because its Bayes bound is 0.42 against 0.6.
- The baselines are described as references "G and H pair against on their own seeds", which is not possible.
- Nothing in the learned arms reads a label (§6).

## 1. Do the readings give every outcome exactly one name?

**The space.** The readings are per task, and each phase sees one kind of outcome:

| phase | outcomes read |
|---|---|
| Phase 0 | 4 final accuracies (NOMEM and NOMEM_SELF × 2800, 2801) against 0.6 (s89 l.40–43, l.103, l.183–192) |
| Phase 1, 24K | 4 bind/no-bind outcomes: Hebbian and delta × 2 seeds (l.193–204) |
| Phase 1, 48K | 4 more at 48000, only if the 24K pilot is not 4/4 (l.124–126, l.214–217) |

The names are the docstring's four (l.52–56) and rep6 l.72–74.

**Run, not read.**
- `decide()` (l.177–221) was exec'd from `8ba99bb` with stub stores, stage by stage as `run_s89.sh` runs it: phase0, phase1, phase1b, phase2, re-importing `ARMS` each time. rep6's reading expression (l.73–74) was applied to the result.
- Phase-0 accuracies were taken from {0.3, 0.6, 0.61}, with at most one above 0.6. That gives 48 patterns × 2⁴ (24K) × 2⁴ (48K) = 12,288 per task, with each task as the target and the other valid.

**Result** (out §A):

| reading | combinations |
|---|---|
| valid | 512 |
| needs the longer budget | 480 |
| untested | 7,200 |
| solvable without memory | 16,384 |

- 0 disagreements with the docstring's rule, applied independently, and no errors.
- An accuracy of exactly 0.6 reads "not stopped", as written ("exceeds", l.41–42; `a > NOMEM_THR`, l.191).
- **No overlaps.**

**Gaps.**
- **(a) A failed record is never a name.** `explore_common16.pick` (l.133–138) returns only records with `ok`, so `fc.runs` drops a failed run. The phase then reads "incomplete" (s89 l.187–189, l.199–201, l.216) and the task is reported "INCOMPLETE, no reading" (rep6 l.74).
  - Seen: a failed 24K pilot, 48K pilot or phase-0 run each leaves the task with no reading, and a failed phase-0 run blocks both tasks (out §A).
  - `_bound`'s `r.get("ok")` (l.173–174) is dead code.
  - The docstring says nothing about failures. Name the case (fix 6).
- **(b) The 48K reruns repeat both memories.** "Needs the longer budget" needs 2/2 on *each* memory at 48000 (l.54, l.217), including a memory that was already 2/2 at 24K. A rerun is bit-identical up to 24000. But "bound" means "holds to the end", so a run that bound at 24K can read unbound at 48K if accuracy dips after 24000. That case is rare, and it reads "untested". Say so, or carry a memory that was 2/2 at 24K forward.

**What "valid" can say.** If each pilot run binds independently with probability p, P(valid) = p⁴:

| p | 1.0 | 0.95 | 0.9 | 0.8 |
|---|---|---|---|---|
| P(valid) | 1.00 | 0.815 | 0.656 | 0.410 |

So a perfect gate that binds 90% of the time is sent to 48K one time in three. This is acceptable as a rule, since it is fixed and conservative, but print it.

## 2. Is the outcome a main-line function?

| outcome | where | main line? |
|---|---|---|
| BOUND | `run_one`'s `transition` (child l.153). `test_router_reliability.py` is blob `8929c2d857` on both the main line and F (out §B) | **imported**: the main line's first clause of `test_stream_recipe.outcome` |
| DISCOVERED | `rec["discovered"]` (rep6 l.89), set by `run_one`, `test_router_reliability.py` l.338 (bound and VAL cos < `DISC_COS`) | **imported**: the same field `test_window_gate.success` reads in Part A (`test_window_gate.py` l.319–327) |
| ROUTED*@end, ROUTED-BOUND | rep6 l.19–21 `routed_end`: margin ≥ 0.9 and η² by stream at KEY > 0.9 | **reimplemented, not a main-line function** |
| failure class | `ec.fail_class` (child l.159) = `test_router_layout.fail_class` (v1), imported through `explore_common.py` l.51 | imported (v1 only) |

**ROUTED*@end is a third copy.** By AST, `routed_end` is not the same as F's own `explore_f_reports.routed_star_end` (l.182–184). That one guards `margin is not None`; rep6 defaults to 0, and would raise on a stored `None`. Nor is it the same as G's `explore_g_rhl_parent.routed_star` (l.43–45), which uses named constants. The three give the same verdict on any record that has both fields.

**The main line's routing outcome does not apply.** `test_stream_recipe.routed` (l.751–753) reads `one_to_one` and `stream_acc`, which these records do not compute. `fail_class_v2.outcome_v2` (l.62–69) wraps it. So ROUTED-BOUND is the exploratory definition G and F have used since S70, and the docstring should say "as S70/S87, not a main-line outcome".

**The docstring promises failure classes under v1 and v2** (s89 l.67). Neither child nor rep6 computes v2: `fail_class_v2` appears nowhere in the three files (out §B), although `fail_class_v2.py` is on F's branch.

## 3. Are the seeds free?

Yes (out §C). Three patterns were searched for 2800–2839:
- the record key `"ARM|28xx|"`;
- `"seed": 28xx`;
- `seed 28xx`.

**Branch tips.** All 13 remote branch tips (json, jsonl and log files) were searched: the main line `7c4ceb8`, explore-F/G/H/I, outside-ideas, replicate-R, rev9/rev10, review-V and the others. The only hit is `explore_out/F/s89.log` on explore-F (seeds 2800 and 2801), which is F's own phase 0, started 18:02.

**History.** All 4,547 distinct json, jsonl and log blobs reachable from any remote branch were searched. The only hit is the same log.

**The main line's seed list.** Every literal `range()` in the 38 `test_*.py` files at `7c4ceb8` covers 910 integers. Adding the attempt seeds s + 1000j (j < 5; `test_stream_recipe` l.243, `test_load_curriculum` l.229–230) gives 3,270. None is in 2800–2839, and no base is in 800–839 or 1800–1839. The prompt assigns F 2800–2839 (l.1108). s89 l.11–12 is correct.

## 4. Do the CHECKs tie the new code to a recorded run?

**What is tied.** The CHECKs passed on the 2.80 GHz Xeon (s89.log at `0378b3a`).
- **G's S87 baseline** (child l.233–251). `BASELINE|2700|f205db31dfed|Intel(R) Xeon(R) Processor @ 2.10GHz` in `explore_out/G/s87_results.json`, first held by explore-G `464f7e4` (12 Oct) and identical at `e46799a` and at the tip (out §D).
  - Reproduced through 2400: the curve at 1200 and 2400, and the gradient norms at updates 1, 10 and 100.
  - The record's statistics at 0, 1200 and 2400 are **not** compared.
  - Reproduced by `explore_f_followup_child.run(arm="S70_HEBB")`, **S70's path**.
- **F's S70 delta run** (child l.252–262). `S70_DELTA|330|ff908bdbc0b2|Intel(R) Xeon(R) Processor @ 2.10GHz` in `explore_out/F/rh_lite_results.json`, first held by `61ba46e` (9 Oct).
  - Reproduced through 2400: the curve and the statistics.
  - Also by S70's path.
  - The file is read from the working tree (`ec.load_store`, child l.255), not from a commit.
- `explore_f_delta_checks` C6: `LOCAL3_SLOW|160` (S43) through 2400.

**What is not tied: the screen's own path.**
- No CHECK runs `explore_f_s89_tasks_child.run`. Its `parse` / `arm_def` / `make_model` / `run_kw` / `run` (child l.81–162) are new code. They compose the same pieces as S70's `spec` / `builder` / `run_kw` (`explore_f_followup_child.py` l.61–102), but nothing shows that they do so bit for bit.
- The groups-and-lrs CHECK (child l.318–326) runs the real path with `evaluate` stubbed for 2401 updates and compares no record.

**Smaller points.**
- **The record ref is not pinned.** `G_REF` is the moving tip `origin/claude/explore-G` (child l.51, l.241); only the module is read at `G_SHA`.
- **(i) of the lite-batch CHECK is implied by the byte-identity CHECK.** The first lite batch from this module equals the one from G's copy (child l.234–239), but the round-6 prefix is byte-identical to `a199d5e`, and G's copy is that file. Keep it, but it adds no tie.
- **The gradient norms are not functions of the first batch.** The docstring (s89 l.73–74) says the norms at updates 1, 10 and 100 are. Those at 10 and 100 are functions of the first 10 and 100 batches.

Fix 3 adds the missing tie: the child's own `run` on `BASELINE_LITE` at 2700, against G's record at `e46799a`, including the statistics.

## 5. The "survives" rule: operating characteristics

F's screen has no such rule, and its baselines feed none (§7). The rule is G's and H's, and it is computed here as asked (out §F; exact).

**The rule.**
- **Within 1 pair**, G's definition (explore_g_s87 l.27): (ceiling only) − (learned only) ≤ 1.
- **Survives**:
  - G: within 1 pair, and discovered against BASELINE by one-sided exact McNemar, p < 0.05.
  - H: within 1 pair, and beats NOSPLIT by ≥ 4 vs 0, i.e. p ≤ 0.0625.
- Pairing enters through q = P(both succeed on a seed), from max(0, p_c + p_l − 1) to min(p_c, p_l).

| seeds | ceiling | learned | P(within 1) independent | over all pairings |
|---|---|---|---|---|
| 20 | 0.9 | 0.9 | 0.792 | 0.778–1.000 |
| 20 | 0.9 | 0.8 | 0.413 | 0.392–0.418 |
| 20 | 0.9 | 0.7 | 0.152 | 0.069–0.175 |
| 20 | 0.9 | 0.5 | 0.007 | 0.001–0.017 |
| 20 | 1.0 | 0.9 / 0.8 / 0.7 | 0.392 / 0.069 / 0.008 | (fixed) |
| 10 | 0.9 | 0.9 | 0.878 | 0.865–1.000 |
| 10 | 0.9 | 0.8 | 0.632 | 0.620–0.736 |
| 10 | 0.9 | 0.7 | 0.387 | 0.376–0.394 |
| 10 | 0.9 | 0.5 | 0.088 | 0.046–0.119 |

**The discovery conjunct.** At BASELINE rate 0 or 0.1 (S87 on lite: BASELINE 0/20 ROUTED-BOUND), "discovered" is near-certain whenever the learned rate is ≥ 0.5. So G's P(survives) equals P(within 1 pair) to three decimals. For H with NOSPLIT at 0.3, P(survives) is 0.789 / 0.511 / 0.278 / 0.047 at learned 0.9 / 0.8 / 0.7 / 0.5.

**What it can say.**
- **It separates "equal" from "half as good":** 0.79 against 0.007 on 20 seeds.
- **It cannot separate equal from one or two pairs worse.**
  - With the ceiling at 0.9 on 20 seeds, a hinge at 0.8 reads "survives" 41% of the time and one at 0.7 15% of the time.
  - With the ceiling at 1.0, a hinge exactly one pair worse in expectation (0.9) reads "survives" only 39% of the time.
  - On H's 10 seeds, 0.9 against 0.7 gives 0.88 against 0.39.
- **Name a margin.** "Survives" is a non-inferiority reading with a margin of one pair, which these counts cannot resolve. It should be printed with its operating characteristic, and "survives" read as "not shown to be more than about 3 pairs worse".
- **"No reading applies" is common.** The complementary "beats by ≥ 4 vs 0" leaves it with probability 0.43–0.44 at learned 0.8–0.7 (ceiling 0.9, 20 seeds), so it is the modal outcome in the interesting range.
- **Pairing barely helps.** It moves these figures by at most 0.08, except at exact equality, where maximal correlation makes "within 1" certain.

## 6. Does anything read a label it claims not to?

**F's learned arms: no.** The arms are BASELINE (LOCAL3 + SLOW), SINGLE_HEBB and SINGLE_DELTA. Traced call by call:
- **Model.**
  - BASELINE: `wdc.local_model(task, …)` (`explore_f_window_delta_child.py` l.60–66).
  - SINGLE: `build(task, tsc.ARM["B"], ARCH, seed)` (child l.116), and `to_delta_fast` for delta.
  - Both receive `task.vocab` and `task.ctx_tokens`. On ALLPOOL, `ctx_tokens` is all 16 context ids (tasks l.384), which is a role label. It is read only when `gate_kind == 'perfect'` (`test_instrument_v2.py` l.356–358, through `MultiBDH.ctx_tokens`). `LocalGateBDH` passes it on and never reads it (`explore_local_gate.py` l.102). **Inert**, but nothing asserts that (fix 7).
- **Training.** `onset_run` (`test_binding_onset.py` l.225–228) draws `tokens, _ = make_batch(...)` and discards the query stream. The loss is `task.select` at `qpos`, which is the task's supervision. The SLOW switch (`switch_check`, l.93–96) reads the step only.
- **Statistics, measurement only.** These run under `no_grad` in eval mode and restore the training mode. They draw no RNG (gate noise 0, `mode='sym'`), which is consistent with S70_DELTA|330 reproducing with the statistics on.
  - `pool_stats` → `gate_stats_k`: reads the probe's `stream_labels` and `role_masks`, and token ids S…S+P−1.
  - `pool_routing_stats`: reads `key_mask`, `body_positions` and `stream_labels`.
  - `write_order_acc`, `ec.fail_class`.
- **Probes.** `probe_batch` (seed + 99,000) and `eval_batch` come from their own generators.
- **Labelled by design.** The pilots' perfect gate reads the stream index (`task.perfect_gate`; on ALLPOOL `AllpoolPerfectBDH`, child l.68–78).

**For G and H, who vendor this module.** The task object exposes:
- `key_mask`, `key_index` and `body_positions`;
- `role_masks`, `stream_labels` and `perfect_gate`;
- the query stream (make_batch's second output).

G's CHECK "assert no label tensor is passed" must also cover the task's methods: a hinge handed `task` can call `task.key_mask`.

**Statistics that read lite's ids** (tasks l.288–289 notes this).
- `gate_stats_k`'s `key_sep` (`test_router_reliability.py` l.258–259) and `key_part` (`test_router_discovery.py` l.255) read token ids S…S+P−1.
  - On POOL these are pool keys 0–3.
  - On ALLPOOL they are **context types 2–5**.
- Both are still stored under lite's names. No reading uses them, but a report table would. Rename or NaN them on the pool tasks.

**The generators against the readings' assumptions** (out §E; the numpy replica, 20,000 sequences per task).

| property | value | consequence |
|---|---|---|
| keys per sequence | 4 types, each exactly twice in the body (once per stream) plus the query key: 9 key tokens in the input | as the prompt states; CHECKed (tasks l.549–565) |
| how often a key type occurs | POOL and ALLPOOL: a type is in 4/32 = 1/8 of sequences (frequency 0.012–0.014 against lite's 0.102) | each key embedding gets 1/8 of lite's signal; the pilots measure what that costs |
| context distance (KEY − its CTX) | 1: 0.500, 3: 0.333, 5: 0.167, as on lite | a width-3 window (LOCAL3) sees the context for half the keys, as on lite; BASELINE is expected near 0 ROUTED-BOUND, as S87's 0/20 |
| distance between a key's two occurrences | 3–17, median 8 | |
| **types that repeat within a sequence** | **CTX: 5 of 5 positions, always** (a stream's context opens both its blocks). **VAL: 2.6 of 8 positions** (0.84 of sequences have a value type twice). KEY: 9 of 9 | **G's REG3_REPEATHINGE and H's REPEATMASS premise is false on these layouts.** The prompt says repeated types are "the keys, and nothing else". The repeated positions include at least 5 non-key positions in every sequence (mean 7.6), and in **0** of 20,000 sequences are they exactly the keys. The repeat statistic selects keys + contexts + colliding values |
| ALLPOOL's contexts | 2 distinct of 16 per sequence; no fixed type → stream map; stream index by first appearance | a fixed type → channel map separates the two contexts in at most 64/120 = **0.533** of sequences (balanced map; replica 0.526) |
| positional residue | ALLPOOL's index is 0 at positions 0–2 in every sequence (P(label 0 \| position) 1.0, 1.0, 1.0, 0.78, 0.78, 0.55); POOL's is 0.50 everywhere | a position-only gate has η² by stream 0.25 on ALLPOOL and 0.00 on POOL. A positional residue in the *measurement*, from the labelling convention |

The repeat-statistic row is G's and H's to act on, but F's module is what makes it so, and the prompt's premise should be corrected before G and H write their arms.

## 7. What else would make the screen uninterpretable

1. **The ALLPOOL outcome is not invariant to a per-sequence channel swap** (verdict 1).
   - The margin is invariant: it compares the query's read gate with the target's and the distractor's write gates within a sequence.
   - η² by stream at KEY (tasks l.427–436) and VAL cos (`role_cosines`, `test_multilayer_binding.py` l.360–370) pool over sequences under the first-appearance labels.
   - **Measured:** a gate that routes every separated sequence perfectly, with the channel set by context type, gets margin 1.000, η² 0.000 and VAL cos 1.000 on ALLPOOL, so it is neither ROUTED nor DISCOVERED. On POOL the same gate gets 1.000, 1.000 and 0.000 (out §E).
   - On ALLPOOL, ROUTED-BOUND and DISCOVERED then mean "routes, *and* puts the first block's context on channel 0 in every sequence". The models are not saved, so this must be fixed before BASELINE_ALLPOOL runs.
2. **ALLPOOL changes two things at once.**
   - From POOL to ALLPOOL, contexts stop being frequent, *and* no fixed context-type → stream map exists. Any token-identity gate is capped at 0.533 of sequences.
   - The validity pilot's gate is the relative index (tasks l.371–375, l.401–409), which no learned arm is given. So "the task is valid" says that the *memory* can use a relative routing, not that the gates under test could express one.
   - The prompt's descriptive "drop from -pool to -allpool" (G) would conflate the two. G's "the labelled hinge itself fails here" guards G's readings; F's reading and the cross-task comparison are not guarded.
3. **Phase 0 cannot stop anything** (s89 l.30–43).
   - **The Bayes bound.** At the query (input position 21), the target pair lies inside NOMEM's 4-token reach with probability 0.12–0.13 and inside NOMEM_SELF's 10-token reach with probability 0.37–0.38. The bound on held-out accuracy is then P(seen) + (1 − P(seen))/15: **0.18–0.19 and 0.41–0.42**, against the 0.6 threshold, on every task (out §E).
   - **What "not stopped" means.** It is guaranteed, so it carries no information. That is the prompt's design, but the docstring should print the bound and not call phase 0 a solvability test. Revision 10 §11's concern was a *gate* routing by position. On POOL that is excluded by the random block order, which the module CHECKs. On ALLPOOL the labels carry the positional residue above.
4. **The baselines cannot be paired.** s89 l.6 and l.59 call them "the references G and H pair against on their own seeds", which is the prompt's wording (l.1145–1146).
   - Pairing needs the same seeds. G runs its own BASELINE on 2842–2861 (prompt l.1169), and H's task is the grouped layout at S = 8, not these.
   - The 60 runs (2 tasks × 3 arms × 10) carry no reading and pair with nothing. They are unpaired descriptive rates.
   - No budget is stated for them. Phase 0 projected 0.46 h makespan; BASELINE, which is expected not to bind, runs the full 24000 updates.
5. **A fix changes the code SHA.** `pick` keys records by the code SHA over 26 modules (`explore_common16.py` l.133–138). Any fix to the child or to `explore_f_tasks.py` will re-queue the pilots that ran under `ea9728c636c6`.
   - The decision file keeps the phase-0 reading, since a recorded phase is never revised (s89 l.178).
   - A change to `explore_f_tasks.py` also changes the module G and H vendor at `50dc8e4`. Announce the new hash before they start.
6. **The run started on a dirty tree** ("git 8ba99bb+dirty", s89.log l.4). It was probably the queue files that `698a20f` committed 33 s later. The code SHA identifies the code, but say so in the report.

## Fixes (most important first)

1. **ALLPOOL statistics invariant to a per-sequence channel swap.** Append to `explore_f_tasks.py` and announce the new hash to G and H. Replace in the docstring (s89 l.65–66):

   > OUTCOMES (as S70 / S87; ROUTED*@end and ROUTED-BOUND are the branches' exploratory definitions, not a main-line outcome): BOUND (a transition); for BASELINE also ROUTED*@end (end margin >= 0.9 and η² by stream at KEY positions > 0.9) and DISCOVERED (bound and end VAL cos < 0.5).
   > - On ALLPOOL, where the stream index is a per-sequence convention, both statistics are computed within each sequence and averaged over the probe: η²_seq at the 8 key positions, and the cosine between the two streams' mean VAL read gates per sequence. explore_f_tasks.pool_routing_stats_inv records them as `eta_key_by_stream_inv` and `val_cos_inv`, and ROUTED*@end and DISCOVERED read these fields there.
   > - On POOL both forms are recorded and printed; they agree for a fixed CTX → channel map.
   > - CHECK: on synthetic gates, a context-type gate that separates the streams in a sequence scores η²_seq = 1 and cos_seq = 0 whichever channel the first block's context takes, and the pooled forms score 0 and 1 on ALLPOOL.

2. **ALLPOOL's routing cap.** The cap is 64/120 = 0.533 of sequences for any fixed context-type → channel map. Either:
   - **(a)** keep the task and add to the docstring after l.25:

     > On ALLPOOL no fixed map from context type to channel separates a sequence's two contexts in more than 64/120 = 0.533 of sequences. A gate that reads token identity (LOCAL3, the register gates) can route the rest only by comparing contexts within the sequence. The perfect gate is handed that comparison (task.perfect_gate, the index by first appearance). So "the task is valid" means the memory can use a relative routing, not that the arms under test can express one. A null on ALLPOOL does not separate "keys are not frequent" from "contexts have no fixed identity".

     Also print, as a fifth pilot on 2800–2801 Hebbian, the *type-level* perfect gate (a fixed balanced map of the 16 CTX types to 2 channels). Descriptive.
   - **(b)** If the round wants ALLPOOL to differ from POOL in frequency only, draw stream 0's context from types 0–7 and stream 1's from 8–15. That keeps every CTX type at about 0.014 and gives a fixed type → stream map, so `perfect_gate_general` with that map is the perfect gate and the pooled statistics are valid. This is a change to the round's design; ask the user.

3. **Tie the screen's own path to a record.** Add to the child's `g87` part, and pin the ref:

   > `G_REF = G_SHA = "e46799a"` (child l.51; the record is identical there and was first held by `464f7e4`).
   >
   > CHECK: `explore_f_s89_tasks_child.run(dict(arm="BASELINE_LITE", seed=2700, iters=2400))`, this screen's own path, reproduces G's S87 `BASELINE|2700|f205db31dfed|Intel(R) Xeon(R) Processor @ 2.10GHz` at e46799a bit for bit through 2400: the curve, the gradient norms at updates 1, 10 and 100, and every statistic at 0, 1200 and 2400.

   Also, in s89 l.73–74 replace "which are functions of the first batch" with "(functions of the first 1, 10 and 100 batches)". Read S70's record from the commit (`git show 61ba46e:explore_out/F/rh_lite_results.json`) instead of the working tree (child l.255).

4. **The baselines' role.** Replace s89 l.6, "records the baselines G and H will pair against on their own seeds", and l.59, "no reading; the references G and H pair against on their own seeds", with:

   > no reading; unpaired descriptive rates on F's seeds. G's readings use G's own BASELINE on 2842–2861, and H's task is the grouped layout. Budget: 60 runs at ≤ 24000 updates, about N h on 4 workers (N measured before phase 2).

   Alternatively, drop phase 2 and spend the CPU on fix 2(a)'s type-level gate.

5. **Phase 0's meaning.** Add after s89 l.43:

   > The rule cannot fire on these layouts. The best possible accuracy for a model that sees only the last 4 (NOMEM) or 10 (NOMEM_SELF) tokens at the query is at most P(the target pair inside the reach) + the rest / 15 ≈ 0.19 and 0.42 (lite, POOL and ALLPOOL alike). "Not stopped" is therefore expected and is not evidence that the tasks need memory beyond that. The positional concern of Revision 10 §11 (a gate routing by position) is excluded on POOL by the random block order (CHECKed). On ALLPOOL the stream index is positional at positions 0–2 (η² 0.25 for a position-only gate), so the statistics there use the within-sequence forms (fix 1).

6. **Failed records.** Add to READINGS (after l.56):

   > A run that fails (ok False) is rerun once on the same seed. If it fails again it counts as not bound in phases 1 and 1b, and in phase 0 as accuracy 0. The decision file records the failure.

   Implement this in `decide()`: count failed records instead of returning "incomplete" (l.187–189, l.199–201, l.216).

7. **Assert the learned arms' gates read no labels.** Add to the `models` CHECK (child l.263–327):

   > For BASELINE_POOL, BASELINE_ALLPOOL, SINGLE_HEBB_ALLPOOL and SINGLE_DELTA_ALLPOOL: `gate_kind != "perfect"`, and the logits are bit-identical with `model.ctx_tokens` set to None. On ALLPOOL `task.ctx_tokens` is all 16 context ids, a role label that only the perfect gate may read.

8. **Failure classes and lite-id statistics.**
   - Compute `fail_class_v2.fail_class_v2(rec)` beside v1 (child l.159) and print it (rep6 l.90), as l.67 promises; or delete "and v2" from l.67.
   - On the pool tasks, set `key_sep` and `key_part` to NaN in the stored statistics (`pool_stats`, tasks l.450–460), since they read ids S…S+P−1.
   - In rep6 l.21, use `routed_star_end`'s None guard, or import it from `explore_f_reports`.

9. **For G and H (the prompt, not F's code; send to X).** The repeat statistic does not select "the keys and nothing else":
   - every context position repeats;
   - 2.6 of 8 value positions repeat per sequence on average;
   - in no sequence are the repeated positions exactly the key positions.

   G's REG3_REPEATHINGE and H's REPEATMASS should print the overlap by role (CTX / KEY / VAL), not only with the keys. Their readings should say that the term is applied at keys + contexts + colliding values. G's label-free CHECK should also cover the task's methods (`key_mask`, `body_positions`, `stream_labels`, `perfect_gate`).

10. **Print the readings' operating characteristics.**
    - The validity rule: P(valid) = p⁴, which is 0.66 at p = 0.9.
    - G's and H's "within 1 pair": the table in §5. On 20 seeds, P(survives) is 0.79 / 0.41 / 0.15 / 0.007 at learned 0.9 / 0.8 / 0.7 / 0.5 against a ceiling of 0.9.
    - Read "survives" as "not shown worse than the ceiling by more than about 3 pairs".

**Reviewed:** claude/explore-F `8ba99bb`, "Session F round 6 (explore_f_s89_tasks): readings fixed before any run", with `explore_f_tasks.py` at `50dc8e4`. The tip has since moved to `0378b3a` (progress files only).
