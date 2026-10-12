# Check of test_delta_recipe.py (`a581027`) against the amended spec

*Referee V, 11 October 2026.*

**What was checked.** `test_delta_recipe.py` as committed on the main line at `a581027` (02:12 UTC, 11 October; "spec `05256c5` + Amendment 1 `32c121b`"), against `specs/test_delta_recipe.md` with Amendment 1. No result file exists yet; the test has not run.

**How.** Numbers come from `test_check_delta_recipe.py` (output in `test_check_delta_recipe_out.txt`). It extracts the test's own functions from its source, runs them, and runs nothing else.

**Verdict.** The three things asked are all met: the docstring says what the amended spec says; every reading is printed from the code's own counts; and CHECK 172 reproduces the referee's overlap values. There are three issues for the user to know before it runs:
- **A crash on a main-line-only clone (§4).** CHECKs 159 and 162 read their reference records from commits that exist only on session branches.
- **The no-room combination (§2).** It is printed as both labels, where the amendment literally says "that write's label only".
- **The open points of `spec_review_test_delta_recipe_2.md` §2** are carried into the code unchanged, as specified: the d ≤ −3 label, the "Hebbian-specific" name, and CHECK 174's FAIL having no consequence.

## 1. Does the docstring equal the amended spec in meaning?

| item | amended spec | docstring (`test_delta_recipe.py:2-93`) and code | same? |
|---|---|---|---|
| Arms and seeds | HEBB_SPLIT, HEBB_SLOW, DELTA_SPLIT, DELTA_SLOW, CEIL_HEBB, CEIL_DELTA; X 580–599, L 1580–1599; perfect gates on the first two | `:52-61`; `ARMS` `:150-161`; `RANGES` `:134` | yes |
| Path and recipe | Part C/D path, LOCAL3, SLOW groups, W_h frozen, no hinge | `:22-29` | yes |
| DELTA | `explore_delta_mem.py` from `9a6c2bf`; β = 1, L2 keys, tied write, decay 0.95; via `to_delta_fast` | `:31-35` | yes |
| The split | checks 2400…40800, 2400 reference only; the probe `Generator(12345)` with counts 5,5,6,8,10,12,5,13; fire rule; c*, c0 over the 32 body keys; exact copy, Adam state untouched; label by `routing_k` on `probe_batch` | `:37-48`; `PROBE_COUNTS` `:140` | yes |
| Outcome | BOUND ROUTED = `test_stream_recipe.outcome(r) == "BOUND ROUTED"` | the code calls `fail_class_v2.outcome_v2(r, tsr.routed)` (`:187-195`). For a bound run both return "BOUND ROUTED" iff `routed(r)` (`fail_class_v2.py:62-65`, `test_stream_recipe.py:756-758`); they differ only in how failures are classified | yes, in meaning |
| Validity | a part VALID iff its perfect gate binds both seeds; D1, D3 use both parts, D2 DELTA, D4 HEBB; pooled needs both machines | `:63-66`; `PARTS` `:166`; `validity` `:859-864` | yes |
| D1 | d = HEBB only − DELTA only, pooled; three labels at d ≤ −3, −2..2, ≥ 3; the paired difference with Newcombe's interval; the operating characteristics; per machine the difference only | `:69-72`; `D1_LABELS` `:169`; `D1_OC` `:174-175` quotes 0.29/0.22 and 0.11/0.05 | yes |
| D2, D4, no room | exact one-sided McNemar, SHOWN if p < 0.05; NO ROOM if the no-trigger arm binds routed on ≥ 36 of the pooled 40 | `:73-74`; `NO_ROOM = 36` `:168` | yes |
| D3 | secondary; on seeds where both split arms are BOUND ROUTED; medians and a one-sided sign test | `:75-76`; `d3_rows`, `d3_of` `:841-857` | yes |
| Readings | the memory rule = D1's label; per write helps / no room / not shown; four named combinations; "no room on either write: that write's label only" | `:77-80` names the four combinations and omits the no-room row; the code prints `"delta: <label>; Hebbian: <label>"` whenever a pair is not one of the four (`split_reading` `:938-941`) | **partly** (§2) |
| Printed, not claims | the overlap (copied, CHECK 172), "BOUND ROUTED and overlap ≤ 0.1", MERGED and failure classes (v2), every firing with its label, probe-blind checks, bands with Wilson intervals; ties by `torch.argmax` | `:81-85` | yes |
| CHECKs | 158–170, 172, 173 before any run; 171 and 174 in the report; 161 at 1e-7 / 1e-6; 158 asserted only on the record's CPU | `:87-89`; `BETA0_TOL` `:823`; CHECK 158's CPU rule `:618-624` | yes |
| Runtime | projection first; nothing cut; perfect gates first, then longest first | `:91-94` | yes |

## 2. Is every reading printed from the code's own counts?

Yes. The chain, in the code:

**The counts.**
- `tally` (`:867-884`) builds each claim's counts from the records, with `paired` / `paired_full` (`:827-833`, `:886-892`) and `success` (`:191-195`).
- Pooling sums b and c over both machines (`pool2`, `:944-950`).

**The verdicts and labels.**
- **D1.** `d1_label` uses `c − b` of the pooled counts (`:916-922`). `paired_full` is called with DELTA_SPLIT first, so b is DELTA only and c is HEBB only (`:871-872`). That makes d = HEBB only − DELTA only, and the Newcombe difference DELTA − HEBB (`diff_str`, `:960-962`), as specified.
- **D2, D4.** `split_verdict` reads the McNemar p computed from the same b and c. `split_label` turns it into "no room" from the pooled no-trigger count `d["old"]` (`:924-935`, called at `:1133-1134`).
- **The readings.** They are printed only in the pooled block (`:1132-1139`). The per-machine block prints claims and the line "(the readings are pooled only)" (`:1143-1146`).

**Run on constructed counts** (`test_check_delta_recipe_out.txt` §2), the extracted functions give:

| constructed case | result |
|---|---|
| D1: 0 vs 3 | d = 3 → "delta not shown non-inferior"; DELTA − HEBB −0.075, [−0.196, +0.029] |
| D1: 3 vs 0 | d = −3 → "delta binds more often"; +0.075, [−0.029, +0.196] |
| D1: 2 vs 2 | d = 0 → "not shown worse by more than 2 pairs" |
| D2/D4: no-trigger 36/40, split 40/40 | 4 vs 0, p = 0.0625 → NOT SHOWN, label "no room" |
| D2/D4: 35/40 | 5 vs 0, p = 0.031 → SHOWN, "helps" |
| D2/D4: 30/40 | 4 vs 0 → "not shown" |
| D3 | 5 earlier, 1 later, 1 tie → p = 0.1094, the exact value |
| The four named combinations | the amendment's names, verbatim |
| delta "no room" with Hebbian "helps" | "delta: no room; Hebbian: helps" |

**Two consequences to note:**
- **The d ≤ −3 label fires with an interval that includes zero.** The 3-vs-0 case is labelled "delta binds more often" while its own printed interval is [−0.029, +0.196]. This is spec_review_2 §2(a), now visible in the output format.
- **The no-room row.** The amendment says "that write's label only" (A1:45). The code prints both writes' labels and names no combination. That is the safer reading, but it differs from the amendment's text. Amend the spec to say what the code does, or the code to print one label.

**Printed but with no stated consequence.** CHECK 174 (`check_174`, `:989-1006`) compares each split arm's curve with its no-trigger arm's through the first firing, or the whole run if none fired, and prints PASS or FAIL with the seeds (`:1174-1175`). As in the amendment, a FAIL changes no claim (spec_review_2 §2(d)).

## 3. Does CHECK 172 reproduce the referee's overlap values on a stored record?

Yes. The test's `overlap` (`:198-204`) and `referee_overlap()` (`:453-462`, which extracts `overlap` from `docs/review_2026-10/recompute.py` at `0fdb363`) were both run here on every WIN3_SPLIT_D8 record of the main line's `window_gate` files:

| file | records equal, bitwise | largest overlap, all runs | largest over bound-routed runs |
|---|---|---|---|
| X | 20/20 | 1.000 (seeds 555, 556, the two failures) | 0.032 (18 runs) |
| L | 20/20 | 0.047 | 0.047 (20 runs) |

`recompute.md:189` gives "overlap at most 0.047" over the bound-routed runs: the same value.

**Both CHECKs' preconditions hold:**
- `0fdb363` is an ancestor of the main line, so `git show 0fdb363:docs/review_2026-10/recompute.py` works on any full clone of it.
- The CHECK compares on this machine's own `window_gate` file (`:753-761`).

## 4. A crash risk before any run (not a spec deviation)

**What happens.** CHECKs 159 and 162 call `record_at(H_SHA, …)` and `record_at(F_SHA, …)` (`:627`, `:666`). These run `git show 7e9197f:explore_out/H/copy2x2_results.json` and `git show 5b9bc2a:explore_out/F/eight_results.json` with `check=True` (`:504-505`), before the code decides whether to assert or only print.
- Neither commit is an ancestor of the main line: they are on `claude/explore-H` and `claude/explore-F`.
- On a clone that has fetched only the main line, verification raises `CalledProcessError` and stops.

**Is this L's situation?** L is likely in that position; this container's own first clone of the main line was shallow.

**The fix.** Either fetch both branches before running (`git fetch origin claude/explore-H claude/explore-F`) and say so in the docstring, or vendor the two records into the main line.

**Whether it is wrong.** The failure is loud, not silent, so no result can be wrong because of it. It would only cost a start.

## What could not be checked

- **The test's printed output.** It has not run (no `results/X/delta_recipe_results.json`).
- **Every CHECK except 172.** They were read, not run. Running them means building models and training, which this round's rules exclude.
- **The fused against parallel tolerances of CHECK 173**, and CHECK 161's measured β = 0 differences. These need forward passes.

## Addendum (11 October, third pass)

**What the probe-blind printout counts.**
- `probe_blind` (`test_delta_recipe.py:1009-1022`) iterates over every row of `checks`.
- The trigger appends a row from update 2400 on, the reference reading, at which no split can fire (`test_window_gate.py:520-546`, `step < REF_AT` returns, `FIRST = 4800`).
- So the printed "N of M checks" counts the 2400 reference rows in M. Its N counts probe readings ≥ 0.95 with the held-out accuracy below 0.95, which is not the same as "checks the probe blocked": a blocked check also needs the rise and gap conditions.

The referee's own count made the same 393-for-353 slip (corrected in `replication_review.md` and `revision_9_check.md`).

**Fix.** Restrict the count to rows with `step >= tgw.FIRST`. Print "probe ≥ 0.95 while held-out < 0.95" as what it is, and separately "blocked by the probe alone": eligible apart from the probe, and not fired.

## Correction (11 October, fourth pass)

"The test has not run" (lines 5 and 94) was wrong when written. Amendment 2 (`55dac50`, main line) states that X's full run started at 02:12 UTC on 11 October with `a581027`, 49 minutes before this check (`6f2788d`, 03:01:36). What held, and what this check had verified, is that no result file or log was on any branch. No finding above depends on it: every check here reads the code and stored records, not X's output. The start time rests on the amendment's text alone until X's log is pushed.

**Second correction (12 October, fifth pass).** The correction above overstated.
- X's process started at 02:12 UTC with the verification CHECKs. The first training run began only after `meta.started` (05:27:31), which is stamped after verification, and the projection; per `rev9/check_delta_recipe_X.md`, at or after 05:48:43.
- At this check's time (03:01:36), no run of any arm had begun. "The test has not run" was true of its runs, though the process was running its verification.
