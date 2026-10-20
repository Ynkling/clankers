# Screen review, second pass: Session G's revised S89 (`claude/explore-G` at `003a370e` and `db14e9b9`)

*Referee V, 20 October 2026. Reviewed:*
- *`explore_g_s89.py` (the parent, cited "s89:NN") and `explore_g_s89_child.py` ("child:NN"), both at `db14e9b9`;*
- *the vendored `explore_f_tasks.py`, against F's `b0c6332`, `50dc8e4` and `a199d5e`;*
- *the earlier review `screen_review_s89_G.md` (at `7ab98c1`, ten fixes) and `screen_review_s89_F.md`.*

*"out §N" is a section of `screen_checks_s89_G_v2_out.txt`, which `screen_checks_s89_G_v2.py` writes. The script reads git objects only and trains nothing. torch is not installed here, so:*
- *the parent's reading code is exec'd from the `db14e9b9` blob and run, together with `explore_g_rhl_parent`'s outcome and pairing functions, `test_conv_lr.mcnemar_greater` and F's `pool_outcome`. All of these are pure Python;*
- *the generators, the hinge masks, `eta2_pool` and F's within-sequence statistics are replicated in numpy. The replica matches the distributions, not torch's exact draws.*

*G's own CHECK log is `explore_out/G/s89.log` at `eb4d3fa4`.*

## Status: the learned arms have started

G's run began at code `db14e9b` (log header), and every CHECK passed (`s89.log` at `eb4d3fa4`, including the 7 reproductions of S87's records). The run then went:
1. **Validity:** ORACLE_D bound 2/2 on both tasks, on -pool at 2400 and on -allpool at 3600. `s89_validity.json` was written at 22:17:05 UTC.
2. **Learned arms:** "200 runs queued on 4 workers" (code SHA `5d4a9b1dba33`).
3. **First learned record**, pushed at 22:20:21 (`eb4d3fa4`): `pool.REG3_D|2842`, acc 0.990, transition 3600, ROUTED-BOUND.

By `67f3897c` (22:21:43) three learned runs had finished:

| run | acc | transition | outcome |
|---|---|---|---|
| `pool.REG3_D` 2842 | 0.990 | 3600 | ROUTED-BOUND |
| `pool.BASELINE_D` 2842 | 0.981 | 7200 | BOUND unrouted |
| `pool.REG3_KEYHINGE_D` 2842 | 0.990 | 6000 | ROUTED-BOUND |

So the learned arms are running, and this review lands after them.

**A parent-only fix can still land without rerunning anything.** `explore_common16.code_sha` (l.58-114) hashes the repository modules that the *child* imports when it loads. The child imports `explore_g_s89` only lazily, inside `checks_static` (child:547). So edits to `explore_g_s89.py` (docstring, `readings`, `report`) leave the SHA and the stored records valid. Any edit to the child, or to a module the child imports, changes the SHA and re-queues all 204 runs. Every fix below is therefore written for the parent.

## Verdict

**Fit to run; fix the readings first.** "Fix first" here means in the parent, before `report()` prints any reading. The runs should not be stopped. In the run code, every item of the earlier review is resolved:
- the vendored module is F's `b0c6332` blob;
- the statistics and the key term are task-aware, and the CHECKs show it;
- 7 S87 records reproduce bit for bit through S89's `run()`;
- the seeds are free;
- no label reaches a learned arm.

The readings are exhaustive and mutually exclusive, and the exec'd code agrees with the docstring on every one of 888,030 outcomes.

**One design problem is now live, and it is in the readings.** Every hinge arm is named against BASELINE_D (LOCAL3) and REG3_KEYHINGE_D. REG3_D, the register the hinges are added to, is descriptive only (s89:50-51). On lite that was harmless, because REG3_D was 0/10. The first learned record of S89 is `pool.REG3_D|2842` ROUTED-BOUND. If the register binds -pool without any hinge, then:
- an inert hinge arm reads "survives" or "replaces the key hinge";
- an arm that is worse than REG3_D reads "needs frequent keys".

Neither reading would be about the hinge. Fix 1 adds the comparison. It does not change the names committed at `db14e9b9`, which are printed unchanged, and it must be dated as added after the first learned record.

## 1. The ten fixes of `screen_review_s89_G.md`, at `db14e9b9`

| # | fix | status | where |
|---|---|---|---|
| 1 | parent run and reading code; outcome imported; CHECK on S87's records re-labelled | **RESOLVED** | Readings: s89:219-252, imports `explore_g_rhl_parent` s89:133, `rb = hp.routed_bound` s89:180. CHECK: s89:276-325. Report: s89:329-396. In the log, Hebbian 2700-2719 reads "weaker", delta 2720-2729 reads "survives" (out §4 reproduces both). `not_worse_1` is named as imported (s89:61) but is not used (fix 9 below). |
| 2 | vendor F's module and assert the blob | **RESOLVED** (with F's later commit) | `b0c6332`, blob `e24d5e10…` = F's tip; prefixes of `50dc8e4` (35,605 bytes) and `a199d5e` (14,101 bytes) asserted (child:90-93, 527-537; out §1). `TASKS` at child:93. |
| 3 | task-aware ceiling, oracle and statistics; CHECKs on lite; ORACLE on -allpool; untrained-oracle ROUTED*; F's pool baseline reproduced | **RESOLVED**, except the record tie on the round-6 tasks, which is not possible (§2.4) | Key term: child:107-144. Statistics: child:286-349, 362-445, with `STATS_POOL = ft.pool_stats_inv` at child:95. Perfect gate: child:236-248, 268-274. CHECKs: child:602-668, 694-701. F ran no pool baseline (s89:113; out §9). |
| 4 | close the naming gaps | **RESOLVED** | Readings: s89:71-80, 143-149, 196-210. The parent's own enumeration is at s89:289-310. Out §3: the exec'd `readings()` gives exactly one name on every count vector at n = 10 and n = 20, with 0 disagreements with the docstring's rules written independently. |
| 5 | "the task does not need the hinge"; failed runs | **RESOLVED** | s89:65-68 and 225-242. Retries: `explore_g_common.py:47` (`MAX_ATTEMPTS = 3`) and :161-171. A missing seed reads UNTESTED (s89:229-231). That includes a missing seed of the descriptive REG3_D (out §3); see fix 8. |
| 6 | frequency arithmetic; across-task line | **RESOLVED** | Arithmetic: s89:38-44, matching out §6. FREQ is not compared across tasks: s89:83-85, 356-363. |
| 7 | a "token ids only" CHECK able to fail | **RESOLVED** | Asserts: child:148-152, 165-176, 179-180. CHECK: child:562-579 (passed: it raises for `stream_labels`, the key ids, a float tensor, a narrower tensor and another sample). Caveat: `hinge_mask`'s own assert (child:180) cannot tell `stream_labels(x)` from tokens (long, width 22, values in [0, vocab)). Only `frequent()`'s sample-equality check and the call path (§2.9) exclude a label. |
| 8 | CPU-tolerant reproduction, printed | **RESOLVED** | Child lookup: child:500-507 (prefix lookup, exactly one hit asserted; the CPU is printed at :700). Parent lookup: s89:256-266. |
| 9 | REG3's role; delta arms | **RESOLVED as written**, but REG3_D's descriptive status is now the problem (fix 1) | s89:49-51, 237-239. Delta is now the only memory. |
| 10 | operating characteristics in the docstring | **PARTLY** | s89:86-89 quotes the KH ≈ 0.9 rows. On delta the lite ceiling was 10/10, and at KH ≈ 1.0 the numbers are much stricter (§2.10, fix 5). "Between 0.7 and 0.8 the most likely name is 'weaker'" holds only for independent arms: with nested arms at 0.7 the most likely name is "needs" (0.59). |

## 2. The earlier checks, re-run on the new code

### 2.1 Readings: exhaustive, run through the parent's own code

`readings`, `arm_name`, `clamps`, `commit`, `med_commit` and `_fake` were exec'd from `db14e9b9`, with `explore_g_rhl_parent`'s `bound`, `routed_star`, `routed_bound`, `paired` and `beats`. They were run on every count vector of the 8 per-seed (KH, arm, BASELINE) patterns (out §3):
- **n = 20:** 888,030 vectors;
- **n = 10:** 19,448 vectors;
- each for FQ on -pool, RP on -pool and FQ on -allpool, where the renamed "needs the key positions" appears.

**Every vector gets exactly one outcome.** It is either one of the first readings ("does not need" / "fails here") or exactly one of the arm's four names. There are 0 disagreements with the docstring's rules written independently.

At n = 20 the FQ outcomes are:

| outcome | vectors |
|---|---|
| fails here | 689,539 |
| does not need | 77,946 |
| needs | 50,715 |
| survives | 30,980 |
| does nothing | 28,187 |
| weaker | 10,663 |

"clamps" is an accompaniment, printed with any name.

**The delta rule.** Delta is the only memory now. The same readings apply to the _D arms on 2842-2861 (n = 20). The validity gate is ORACLE_D 2/2 on 2840-2841, run first. There is no separate delta rule left to check: the old "delta descriptive on 2862-2871" is gone, and 2862-2899 are unused (s89:54).

**The Hebbian rule.** No Hebbian learned arm runs (s89:16-20). Hebbian enters only through the CHECK that re-labels S87's records:
- KH 20/20, FQ 17/20 nested, BASELINE 0/20: KH only 3, p 0.125, so the name is "weaker";
- delta: 10/10 against 10/10, so the name is "survives".

Both follow from the records (out §4) and match G's log.

**The pilot numbers G cites are right** (out §9, F's stores):
- O_HEBB on -pool and -allpool: 0/2 at 24K and at 48K;
- D_HEBB under Adam: 0/2;
- O_DELTA: 2/2, transitions at 2400-3600;
- conv pilot: -pool 2/2, -allpool 1/2.

### 2.2 The outcome function

ROUTED-BOUND is `explore_g_rhl_parent.routed_bound` (s89:180, imported). On -allpool it is applied to `inv_view(record)` (s89:156-160, 334-336). That swaps in `eta_key_by_stream_inv`, so the outcome is F's within-sequence form. Out §7 gives 152/160 agreement with F's `pool_outcome(rec, False / True)` on a grid that includes the 0.9 boundaries.

The other 8 grid records **raise a TypeError**: a record whose `eta_key_by_stream_inv` is None makes `routed_star` compare None > 0.9 (`explore_g_rhl_parent.py:45`). F's function reads that case as False. It cannot happen on a completed run, because `STATS_POOL` writes the field at every evaluation (`checks_on` asserts it is present at the end, child:726-733). It is noted for completeness.

**DISCOVERED on -allpool is not swap-invariant.** `inv_view` keeps the top-level `val_cos`, which is the pooled VAL cos. So the DISCOVERED column of `hp.count_table` (`explore_g_rhl_parent.py:48-49, 113-114`) on -allpool reads the first-appearance labelling. A record that F's `pool_outcome(rec, True)` calls DISCOVERED reads not DISCOVERED here (out §7). The column is descriptive (s89:62-63), but it should be the invariant one (fix 4).

### 2.3 Seeds

2840-2899 appear in no record at the 13 remote tips and in none of the 4,771 json/jsonl/log/txt blobs in the history (2.58 GB), apart from G's own S89 files (out §8):
- `explore_out/G/s89.log`;
- `explore_out/G/s89_results.json`, whose meta lists the queued seeds 2840-2861.

Both were written by this run, at `0bd95b21` and later. F's pilots are on 2800-2801.

### 2.4 Do the CHECKs tie each task's path to a recorded run?

**Lite: yes, 7 records.** `checks_repro` (child:694-701, driven from s89:402-404) reruns these records through S89's own `run()` and finds them bit for bit through 2400 (curve and statistics, `s87.trunc`):
- `REG3_FREQHINGE_D`, `REG3_KEYHINGE_D`, `BASELINE_D`, `REG3_D` at 2720;
- `REG3_FREQHINGE`, `REG3_KEYHINGE`, `BASELINE` at 2700.

The records are in `explore_out/G/s87_results.json`, store SHA `f205db31dfed`, last committed at `57c25c05`, recorded on the 2.10 GHz Xeon. Each has exactly one key under its prefix (out §4). The log at `eb4d3fa4` reads "ok" for all 7.

**-pool and -allpool: no record.** Nothing recorded runs on G's path:
- F ran no baseline;
- F's delta perfect-gate pilots convert with `explore_f_delta_fast.to_delta_fast` (F child:144). G converts with `explore_delta_mem.to_delta` (child:271). F's docstring claims the fast form agrees with the parallel form to 1e-5 relative, not bit for bit (out §9).

What ties the round-6 code is narrower:
- **F's functions:**
  - `check_pool_task`;
  - `check_inv_stats`;
  - `layer_stats89` L3 against `pool_routing_stats` (child:633-639);
  - the key positions against `task.key_mask` (child:609-619);
  - ORACLE_D against `rc`'s ORACLE_ALL_D, parameters, logits and gates, on -pool (child:640-658).
- **G's own REG3_D:** each switched-off hinge equals it through 2400 bit for bit on both tasks (child:704-714).
- **A descriptive check:** G's ORACLE_D transitions (-pool 2400/2400, -allpool 3600/3600) match F's O_DELTA pilots (-pool 3600/2400, -allpool 3600/3600) to within one evaluation.

That is as much as can be tied without a record, and it is enough to run.

### 2.5 The vendored module

`git rev-parse db14e9b9:explore_f_tasks.py` = `e24d5e10815eb1234d1947aa6b74b128d2696882` = F's `b0c6332` = F's tip `53578896` (out §1). Its first 35,605 bytes are `50dc8e4`'s blob `ffe249bb…` and its first 14,101 bytes are `a199d5e`'s `de474637…`, so F appended only. G's `003a370e` vendored `50dc8e4` exactly.

The hash CHECK (child:528-532) also passes when F's commit is missing from the clone, on the hard-coded constant alone. The constant is a blob id, so this is sound.

### 2.6 No crashes on the pool tasks

Read and traced against the code:
- The key term uses `task.body_positions` and `task.key_index` over `k_pool` = 32 (child:107-113, 137-144).
- `run_one` receives `stats_fn89` (child:481). On the pool tasks that is `pool_stats_inv(ROUTE_AT)` + `write_rate89` + `layer_stats89` (child:342-349).
- After the run: `stream_acc89`, `probe_rh89`, `acc_by_cell89` and `write_order_acc89` (child:489-491).
- The ORACLE on -allpool is `task.perfect_gate` (child:242-243).

The lite-only functions now appear only in lite branches or in CHECKs (out §2):
- `ft.body_positions` at child:112 and :117, in the lite branch;
- `rc.stats_fn` at :344, in the lite branch;
- `dc.key_eta`, `gc.stream_acc`, `rc.acc_by_cell`, `rc.layer_stats` and `ft.write_order_acc` at :608 and :625-629, in `checks_static`.

The evidence is executed, not only read:
- `checks_on` ran all 6 arms for 20 updates on both tasks, with every end field present;
- the 4 ORACLE_D runs and the first learned run completed (log).

### 2.7 ALLPOOL: the within-sequence outcome is swap-invariant as F's `b0c6332` defines it

The numpy replicas of `eta2_rows`, `inv_from_gate` and `pool_routing_stats`' margin were run on 2048 -allpool sequences (out §7):

| gate | within-seq η² | within-seq VAL cos | pooled η² | margin |
|---|---|---|---|---|
| relative perfect | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| perfect, channels swapped in a random half of the sequences | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| type-ordered | 1.0000 | 0.0000 | 0.0007 | 1.0000 |
| balanced type map | 0.5249 | 0.4751 | 0.0003 | 0.5249 (separated fraction 0.5249; cap 0.5333) |
| random soft gate, before / after a per-sequence channel swap | 0.1446 / 0.1446 | 0.9304 / 0.9304 | — | −0.0019 / −0.0019 |

So both ROUTED-BOUND conditions (margin and within-sequence η²) are invariant to any per-sequence channel permutation, and the pooled η² is not. G applies the invariant form on -allpool and the pooled form on -pool (s89:56-61), as F's comment says to. The exception is DISCOVERED (§2.2).

The routing cap and its caveat are in the docstring (s89:26-31) and printed beside every -allpool reading (s89:351-353).

### 2.8 The frequency and repeat cuts on each task

Out §6, 4096 sequences, per sequence (G's CHECKs, which passed, give the same numbers with QUERY split out):

| task | cut | frequency hinge positions | of the key term's 8 | repeat hinge positions | of the key term's 8 | exactly the key positions |
|---|---|---|---|---|---|---|
| lite | 2 CTX + 4 KEY | 14 (5 CTX, 9 KEY) | 8 | 16.67 (5 CTX, 9 KEY, 2.67 VAL) | 8 | 0 of 4096 |
| -pool | 2 CTX | 5 (CTX) | **0** | 16.67, identical to lite's | 8 | 0 of 4096 |
| -allpool | 16 VAL (ratio 1.45 here, 1.60 at F's draws) | 8 (VAL) | **0** | 16.67, identical | 8 | 0 of 4096 |

**New: what each hinge does to the routed solution, and what it reads by chance.** "Routed solution" means the perfect gate. The hinge's value is relu(η² − 0.2) over a 32-sequence training batch, grouped as the code groups (out §6):

| task | key term: perfect gate η² / random gate η² | frequency hinge: perfect / random | repeat hinge: perfect / random |
|---|---|---|---|
| lite | 0.000 / 0.012 | 0.358 (hinge 0.158) / 0.010 | 0.329 (hinge 0.129) / 0.038 |
| -pool | 0.000 / 0.120 | **1.000 (hinge 0.800)** / 0.005 | 0.334 (hinge 0.134) / 0.088 |
| -allpool | 0.000 / 0.117 | 0.063 (0) / 0.057 | 0.112 (0) / 0.120 |

Three consequences, none of them in the docstring:
- **On -pool the frequency hinge penalises the stream itself.** It groups the 5 CTX positions by CTX id, and on -pool CTX id is the stream index. Its pressure against the routed solution is five times lite's: hinge 0.80 against 0.16. A -pool "needs frequent keys" can therefore mean that the term hurt the register, not that keys are infrequent (fix 2).
- **The repeat hinge has the same positions on every task, but it is not the same device.** It groups by token id across the batch. On lite and -pool the routed solution pays 0.13. On -allpool it pays nothing, because CTX ids no longer map to streams. Its chance level rises from 0.04 to 0.12.
- **The key term's chance η² rises tenfold** from lite to the pool tasks (0.012 to 0.12): 32 key types over 256 positions per batch instead of 4. It still costs the routed solution nothing.

The descriptive "drop from -pool to -allpool" for REG3_REPEATHINGE_D (s89:83-84) therefore compares two different pressures (fix 3).

### 2.9 Label leaks: the call paths

These were traced through the AST of `db14e9b9` (out §2).

**Frequency hinge:**
1. `Term89.forward` (child:200-233) → `term_value(self.t_kind, live, tokens, self.t_task)`, where `tokens` is the forward's input.
2. → `hinge_mask` → `_assert_tokens`, which reads only `task.L` and `task.vocab`.
3. → `frequent(name)` → `ft.freq_sample(task)`, which is `task.make_batch(4096, Generator(87000))[0][:, :-1]`, then `s87.freq_types`.
4. → `tokens ∈ sel`, then `eta2_pool(g[mask], tokens[mask], V)`.

**Repeat hinge:** `repeat_mask(tokens)`, which reads no task attribute.

**Key term (the labelled ceiling, by design):** `task.body_positions` and `task.key_index`.

**Statistics:** under `no_grad` and eval mode. `Term89`'s term fires only when `self.training and torch.is_grad_enabled()` (child:202).

**No leak.** In training, the query stream that `make_batch` returns is discarded (`onset_run`).

### 2.10 Operating characteristics at the new seed counts

All delta arms are now on 20 paired seeds. The figures are exact, using the names from §2.1's run of the parent's code (out §5), with BASELINE at 0 (at pK = 1.0, BASELINE at 0.1 or 0.3 moves no entry by more than 0.003; at pK = 0.9 and BASELINE 0.3, by up to 0.045).

P(name) for the learned arm at learned rate pA, with ceiling pK = 1.0. This is the relevant row on delta, because S87's lite delta KH was 10/10.

| pA | survives | weaker | needs | does nothing |
|---|---|---|---|---|
| 1.0 | 1.000 | 0 | 0 | 0 |
| 0.95 | 0.736 | 0.248 | 0.016 | 0 |
| 0.9 | 0.392 | 0.475 | 0.133 | 0 |
| 0.8 | 0.069 | 0.342 | 0.589 | 0 |
| 0.7 | 0.008 | 0.099 | 0.893 | 0 |
| ≤ 0.5 | 0.000 | ≤ 0.001 | ≥ 0.999 | 0 |

At pK = 0.9 the docstring's figures hold: survives 0.79-1.00 / 0.39-0.41 / 0.07-0.15 / ≤ 0.007.

What this means:
- **At a delta ceiling near 1.0, "survives" is strict.** An arm one seed in ten below the ceiling survives only 39% of the time.
- **"Does nothing here" is unreachable with a strong ceiling.** An arm that is dead, or REG3-like, reads "needs …" with probability 1. "Does nothing" fires only when KH itself is weak.
- **"Needs" therefore does not separate an inert term from a harmful one.** Only a comparison with REG3_D does (fix 1).
- **Other gates:**
  - validity: P(2/2) = p², which is 0.81 at p = 0.9;
  - "does not need the hinge" needs BASELINE_D ≥ 16/20, which has probability 0.63 at a BASELINE rate of 0.8 and 0.05 at 0.6.

## 3. What "delta memory as the primary memory" changes, and the new problems

**The claim is now about delta memory.** The round-6 prompt (`docs/reading/parallel_sessions_prompts.md` l.1158-1196 on `rev10-draft`) asks a Hebbian question, about S87's 17/20 on Hebbian. G's reason for switching rests on F's pilots, and those numbers check out (§2.1): the Hebbian perfect gate binds neither pool task, while the delta perfect gate binds both.

But on delta the lite reference is no longer "two pairs short":
- S87's delta FREQ was 10/10 against KH 10/10, which reads "survives" at n = 10;
- the question S89 can answer is therefore "does a hinge that matched the ceiling on lite (delta) still match it without frequent keys";
- it does not answer the Hebbian question of the prompt.

Two more points on the switch:
- **The register-spec consequence.** "The report says what that means for the register spec" (s89:69-70) must say that the spec's hinge was tested on delta only.
- **"The user's re-plan of 20 October"** (s89:7) is not in the repository. Cite where it is recorded.
- **F's decision file reads both tasks "UNTESTED"** under F's own rule, which pools the two memories (`s89_decision.json`). "Valid for delta" (s89:15) is G's per-memory reading, made by G's own gate. Say so.

**New problem 1, the one that matters: REG3_D is read nowhere** (s89:50-51, 237-239). The hinge arms' names test two things only:
- discovery against BASELINE_D (LOCAL3);
- distance to KH.

Nothing asks whether a hinge adds anything to REG3_D, the model it is added to. On lite delta REG3_D was 0/10, so the question did not arise. The pool tasks remove what made the register fail on lite: a gate can no longer lean on four frequent key identities. The first learned record is `pool.REG3_D|2842` ROUTED-BOUND at 3600. If REG3_D binds most -pool seeds:
- REG3_FREQHINGE_D and REG3_REPEATHINGE_D will read "survives" / "replaces the key hinge" without doing anything;
- and on -pool the frequency term penalises the stream (§2.8), so a FREQ arm worse than REG3_D will read "needs frequent keys".

My earlier fix 9 offered "descriptive" as an option. It should not have: the reading needs REG3_D. This is fix 1.

**New problem 2.** The frequency hinge on -pool and the repeat hinge across tasks are different devices from lite's (§2.8). This affects names, not code (fixes 2-3).

**New problem 3.** On -allpool, DISCOVERED is printed in its pooled, label-reading form (§2.2; fix 4).

**New problem 4.** `s89_validity.json` is not keyed by code SHA (s89:168-175, 429-437). If the child changes after the file is written:
- the ORACLE_D records re-run under the new SHA;
- the queue decision and the report's `valid` (s89:348) still come from the old file.

The run is already under way at SHA `5d4a9b1dba33`, so the rule is simply: **no child change during S89**. If one is unavoidable, record the SHA in the file (fix 6).

**Open confound on -allpool (no fix requested now).** F's type-level perfect gate was run on Hebbian only (`O_TYPE_HEBB_ALLPOOL`, 0/2). Whether a gate that reads context identity can bind -allpool on delta is unknown. Adding that arm would change the child and so the SHA. State it with the routing-cap caveat instead.

## Fixes (parent only, `explore_g_s89.py`: the code SHA is unchanged and no run is repeated), most important first

1. **Read REG3_D: the hinge's contribution over the register.** In the docstring, insert after s89:70:

   > 2b. "the register does not need the hinge here"   REG3_KEYHINGE_D does not beat REG3_D (one-sided exact McNemar, p >= 0.05, on the n paired seeds). The names in 3 are then printed as committed, but read as descriptive: no hinge can show what it adds over a register that binds without one. Every name in 3 also prints the arm against REG3_D (arm only, REG3_D only, p either way). Where REG3_D beats the arm (p < 0.05), the report adds "and the term hurts the register here". ADDED at <commit>, after <k> of the 200 learned runs had finished (the first, pool.REG3_D|2842, ROUTED-BOUND at 3600), on Referee V's screen_review_s89_G_v2.md. The names committed at db14e9b9 are unchanged and printed beside it.

   In code:
   - in `readings()`, after s89:243-245, add `if "R" in R: reg_disc, t_r = hp.beats(KH, R["R"], rb)` and, when `not reg_disc`, set `rd["register"] = "the register does not need the hinge here (KH only …, REG3_D only …, p …)"`. Do not change `rd["first"]`.
   - in `arm_name()` (s89:196-210), pass `R.get("R")` and append `hp.paired(arm, reg, rb)` and `hp.paired(reg, arm, rb)` to `why`.
   - add a hand case to `check_readings()`: REG3_D 20/20, KH 20/20, FQ 20/20, B 0/20 reads "survives" with "the register does not need the hinge here".

2. **Say what the frequency hinge is on -pool.** Replace s89:75-76 ("the frequency hinge needs frequent keys [-allpool: …]") with:

   > "the frequency hinge needs frequent keys" [-allpool: "needs the key positions", since the cut there selects values; -pool: the cut selects the 2 CTX types, and grouped by CTX id they are the stream, so the term penalises the stream at the 5 CTX positions (eta^2 1.0 for the perfect gate, hinge value 0.80, against 0.36 / 0.16 on lite; V's screen_checks_s89_G_v2_out.txt §6). A -pool "needs" is read with the arm against REG3_D (reading 2b): below REG3_D it is the CTX term hurting the register, not the keys' frequency]   p(KH > FREQ) <= 0.0625

3. **The across-task line.** Append to s89:83-85:

   > REG3_REPEATHINGE_D selects the same positions on both tasks but groups them by token id across the batch: the perfect gate pays 0.13 on -pool and 0 on -allpool (CTX ids no longer name streams), and a random gate reads 0.09 and 0.12 by chance; REG3_KEYHINGE_D's chance level is 0.12 on both (0.012 on lite). The drop is printed with these numbers and read as a change of task and of pressure together.

4. **DISCOVERED on -allpool in its invariant form.** Replace `inv_view` (s89:156-160) with:

   ```python
   def inv_view(r):
       """ALLPOOL's outcome: eta^2 by stream at KEY and VAL cos in their within-sequence forms; the pooled ones kept."""
       e = r["end"]
       assert e.get("eta_key_by_stream_inv") is not None and e.get("val_cos_inv") is not None, "no within-sequence fields"
       return dict(r, val_cos=e["val_cos_inv"], val_cos_pooled=r.get("val_cos"),
                   end=dict(e, eta_key_by_stream=e["eta_key_by_stream_inv"], eta_key_by_stream_pooled=e.get("eta_key_by_stream")))
   ```

   Add `val_cos_inv=0.1` against `val_cos=0.9` rows to the grid CHECK. That CHECK is child-side, so it changes the SHA: run it as a parent-side check instead, as out §7 does with F's `pool_outcome`.

5. **Operating characteristics at the delta ceiling.** Replace s89:86-89 with:

   > OPERATING CHARACTERISTICS (exact, n = 20, BASELINE ~0; V's screen_checks_s89_G_v2_out.txt §5). On delta the lite ceiling was 10/10, so read the KH ~1.0 row: "survives" fires with probability 1.0 / 0.74 / 0.39 / 0.07 / 0.01 for an arm at 1.0 / 0.95 / 0.9 / 0.8 / 0.7 (at KH ~0.9: 0.79-1.0 / 0.39-0.41 at 0.8 / 0.07-0.15 at 0.7). With a strong ceiling "does nothing here" cannot fire: an inert arm reads "needs …", so "needs" is read with the arm against REG3_D (2b). The validity gate passes a perfect gate binding at p with probability p^2 (0.81 at 0.9).

6. **Key the validity decision to the code.** At s89:433-436, add `v["sha"] = gc.load(NAME)["meta"]["sha"]`. In `_arms()` (s89:168-175) and `report()` (s89:339, 348), treat the file as absent if `v.get("sha") != gc.load(NAME)["meta"]["sha"]`. Until this lands, make no change to the child or to any module it imports while S89 runs.

7. **The claim's scope.** Replace s89:15-16 ("Both tasks read "untested" for Hebbian and valid for delta. So every arm here is on delta memory.") with:

   > F's decision file reads both tasks UNTESTED under F's rule, which needs both memories; by this screen's own gate (ORACLE_D 2/2 on 2840-2841, delta) both are valid for delta. So every arm here is on delta memory, and the readings answer the round's question for delta only: S87's delta reference is "survives" (REG3_FREQHINGE_D 10/10 = REG3_KEYHINGE_D 10/10 on 2720-2729), not the Hebbian "two pairs short". The user's re-plan of 20 October is recorded at <where>.

   Also add to s89:69-70: "…for the register spec, on delta memory (the spec's Hebbian case is untested here)".

8. **A missing descriptive run does not void the task.** At s89:225, compute `missing` over the roles `B`, `KH`, `FQ` and `RP` only. Print a missing REG3_D seed beside the readings, and compute reading 2b on the paired seeds that exist, with their n printed.

9. **The imports, as stated.** At s89:61 replace "paired, beats and not_worse_1" with "paired and beats".

10. **The -allpool confound, stated.** Append to the -allpool caveat (s89:352-353):

    > "…; F's type-level perfect gate was run on Hebbian only (O_TYPE_HEBB_ALLPOOL 0/2), so whether a gate that reads context identity can bind -allpool on delta is not known here."

*Commits reviewed: `claude/explore-G` `003a370e` and `db14e9b9`, with the progress commits up to `67f3897c` (CHECK log, validity, the first three learned records). Out §0 and §8 were written at `67f3897c`. The vendored module was compared with `claude/explore-F` `b0c6332`, `50dc8e4` and `a199d5e`. The checks are in `screen_checks_s89_G_v2.py`, with output in `screen_checks_s89_G_v2_out.txt`.*
