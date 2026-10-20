# Check: Session H's report 10 (S89), and the S89b screen

*Referee V, 20 October 2026.*

**What I read.** `git fetch` on 20 October. claude/explore-H is still at `13fe610` (04:21:12 UTC, "Session H round 6: S89 report_10 and README section"), so nothing newer exists.

Inputs, all at `13fe610` unless noted:
- `explore_out/H/report_10.md`, `report_10_notes.md`, `s89_runs.md`, `s89.log`, `s89_oracle.log`, `s89_results.json` and `checks.json`;
- the README section (`explore_out/README.md`, the last 16 lines);
- `explore_h_s89.py` and `explore_h_s89_child.py` at all 48 commits from `d4a9a7d` to `13fe610`;
- `explore_h_report.py` (changed once, at `4867fb5`);
- for the cross-screen claims, S87a's `allmass2_results.json` and S79's `constants_results.json`.

"r10:NN" means a line of `report_10.md`; "s89:NN" a line of `explore_h_s89.py`, which is identical at `d4a9a7d` and `13fe610`.

**Script.** `check_H_report10.py` prints only. It reads everything with `git show` and runs no session code as a program. Its sources:
- **The main line's outcome.** `test_stream_recipe.outcome` and the functions it calls (`routed`, `fail_class_k`, `fail_class`, `shared_max`, `merged`, `ROUTED_ACC`), plus `wilson`. They are cut out by ast at the main head `ed9ca2b` and at the child's tree `9c5939e`; the two syntax trees are equal.
- **The referee's re-implementation.** `rlib.outcome_k`, with the transition recomputed from the curve and the map from `stream_gate`.
- **H's own code**, exec'd from source on the stored records with the harness stubbed: `report()` and `summarize()`, `explore_h_common`, `explore_common16.runs_of/pick`, and `explore_h_report.s89_section`. The section's file write is captured in memory.

## Verdicts

- **Report 10: correct on every count, pair, p-value, firing figure and reading.**
  - The stored outcome, the main line's outcome (at both refs) and rlib's agree on all 42 records.
  - The report writer, re-run on the store, reproduces report_10.md lines 3-29 and `s89_runs.md` byte for byte.
  - The reading is "the labelled recipe itself fails here (BASE 1/10); no other reading". It is correct, and "moot" did not apply.
  - Non-MATCH rows: 2 (a count stated as "four" runs where there are seven), plus 3 notes on context.
- **Pre-registration: clean.**
  - No run code, reading code or docstring changed after `d4a9a7d`.
  - The one later code change is the report writer (`4867fb5`, 00:44 UTC). It came after 24 learned records and touches printing only.
  - **None of my eight pre-run fixes was taken.** My review landed at 18:40:23, inside the 58 s window (18:39:40-18:40:38) in which the learned arms were launched.
  - Because BASE failed its gate, the fixes about readings (3, 4, 5) never came into play. Report 10's notes carry the substance of fixes 1 and 2 in prose.
- **S89b: it does not exist.** There is no S89b file, docstring section or commit on any ref, so there is nothing to review and no verdict to give. §B lists what an S89b must contain before it runs.

---

## A. Report 10

### A1. Counts, pairs and readings

| where | report says | recomputed (main-line outcome; rlib; H's report()) | status |
|---|---|---|---|
| r10:7 BASE | 1/10 MINORITY, Wilson 0.02-0.40; v1 MERGED 9; v2 (2 share) 7, (3 share) 2; transition 36000 | 1/10 (seed 2910 at 36000), MINORITY, 0.02-0.40, same classes | MATCH |
| r10:7 BASE firings | 30 splits; own pair ok 27/30; key-pair 27; all-pair 26; repeat-pair 26; at probe < 0.2: 17 | 30, 27, 27, 26, 26, 17 | MATCH |
| r10:8 ALLMASS | 3/10 MINORITY 0.11-0.60; MERGED 7; (2 share) 6, (3 share) 1; transitions 37200, 40800, 40800, median 40800 | seeds 2904, 2908, 2909; same | MATCH |
| r10:8 ALLMASS firings | 30; 28; 27; 28; 28; 17 | 30, 28, 27, 28, 28, 17 | MATCH |
| r10:9 REPEATMASS | identical to the ALLMASS row | identical: same seeds, transitions and firing counts | MATCH |
| r10:10 NOSPLIT | 0/10 NEVER 0.00-0.28; MERGED 9, OTHER 1; v2 (2 share) 4, (3 share) 5, non-stream OTHER 1 | same (2910 is non-stream OTHER) | MATCH |
| r10:12 ORACLE | 2/2, {2900: 16800, 2901: 10800}, VALID | 2/2 BOUND ROUTED at 16800 and 10800 | MATCH |
| r10:14 ALLMASS vs BASE | 3 vs 1; p 0.312 / 0.938 | b 3, c 1; 0.3125 / 0.9375 | MATCH |
| r10:15 REPEATMASS vs BASE | 3 vs 1; 0.312 / 0.938 | the same | MATCH |
| r10:16 NOSPLIT vs BASE | 0 vs 1; 1 / 0.5 | b 0, c 1; 1 / 0.5 | MATCH |
| r10:17-18 T vs NOSPLIT | 3 vs 0; 0.125 / 1 | b 3, c 0; 0.125 / 1 | MATCH |
| r10:19 ALLMASS vs REPEATMASS | 0 vs 0; 1 / 1 | 0, 0 | MATCH |
| r10:23 reading | "the labelled recipe itself fails here (BASE 1/10); no other reading" | report() at the head gives this exactly. The prompt's rules on the recomputed outcomes give FAILS. Moot was not reached: BASE < 7 comes before the split and trigger readings (s89:266-267) | MATCH |
| r10:33 | trigger readings and "not needed" not read | correct by the rule. Had BASE passed, NOSPLIT vs BASE (0 vs 1) would have made every trigger reading moot | MATCH |
| r10:25 overlap | 6005 of 6272; key 2048/2048; not key 3957 (CTX 2048, VAL 1781, query 128) | identical on all 10 REPEATMASS records | MATCH |
| r10:27 pairs at firings | BASE key = all 11/30, key = repeat 11/30, all = repeat 30/30; ALLMASS and REPEATMASS 11, 11, 28 | 11, 11, 30; 11, 11, 28 (both) | MATCH |
| r10:27 one-to-one before | 1 per arm | BASE 2910 at 26400; ALLMASS and REPEATMASS 2908 at 26400. All three are op_ok False, and those runs then bound routed | MATCH (note 3) |
| r10:27 v1/v2 differences | none | none (outcome = outcome_v2 on all 40 learned records) | MATCH |
| r10:29, r10:47 runtime | 42 runs, 36.6 CPU-hours, 2.10 GHz, code 6e0a80a55be3 | 42 runs; sum of secs_wall 36.64 h; one CPU; one code SHA; all ok, attempt 1 | MATCH |
| r10:29 CHECKs | all passed 18:16:07 UTC, git 7820b12+dirty, 20.4 min | checks.json: the same (20.35 min); 14 CHECK rows ok in s89_oracle.log | MATCH |

### A2. The notes and the README

| where | report says | recomputed | status |
|---|---|---|---|
| r10:48-49 | restart after 26 saved runs; 4 in flight rerun; no saved run changed | 2 ORACLE + 24 learned saved before the restart; REPEATMASS 2906-2909 rerun. Every one of the 42 records is identical at its first commit and at the head, and no key was ever dropped | MATCH |
| r10:53-54 | perfect gate 10800 and 16800 against 2400 and 8400 (S87a 2600-2601); medians 13800 against 5400 | S87a ORACLE 2400 and 8400; medians 13800 / 5400 | MATCH |
| r10:55 | "The four runs that bound routed (BASE 2910; ALLMASS and REPEATMASS 2904, 2908, 2909) bound at 36000-40800" | **7 runs on 4 seeds** (BASE 1, ALLMASS 3, REPEATMASS 3), all at 36000-40800 | **MISMATCH (count)** |
| r10:56 | S87a BASE bound at 9600-30000, median 18000 | 17 of 20 S87a BASE runs bound routed: 9600-30000, median 18000 | MATCH |
| r10:57 | three of the four bound within 2400-6000 updates of 43200 | true by seed: 2904 and 2909 at 2400, 2908 at 6000, 2910 at 7200. That is 6 runs on 3 seeds | MATCH by seed (note 1) |
| r10:59 | every triggered run fired at 4800; 17 of each arm's 30 firings at probe < 0.2 | 30/30 runs first fire at 4800 (probe 0.016-0.109); 17 in each arm | MATCH |
| r10:60 | all 3 splits used by 14400-36000 | every triggered run used 3; the third came at 14400-36000 | MATCH |
| r10:61 | 29-31 later checks per arm blocked by the cap | cap blocks: BASE 29, ALLMASS 30, REPEATMASS 31. The gap also blocked 8, 9 and 9 checks; the report does not mention this | MATCH |
| r10:62 | ended merged, 2 streams sharing in most, held-out 0.45-0.85 | unbound triggered runs: 19 of 23 are 2-share; final accuracy 0.454-0.852 | MATCH |
| r10:65-66 | own pair labelled 27-28 of 30; key pair 27/30 in every arm | 27 / 28 / 28; 27 in each | MATCH |
| r10:69-70 | NOSPLIT ended lower; at 24000 0.15-0.71 against 0.14-0.89; at the end 0.19-0.85 | NOSPLIT at 24000 0.154-0.714, triggered 0.140-0.894; NOSPLIT final 0.187-0.848. NOSPLIT ends below ALLMASS and below REPEATMASS on 10/10 seeds, and below BASE on 9/10 | MATCH |
| r10:71 | five NOSPLIT runs with 3 streams on one channel, against 1-2 per triggered arm | NOSPLIT (3 share) 5. 2910 (non-stream OTHER) has two channels with 4 streams each. BASE 2, ALLMASS 1, REPEATMASS 1 | MATCH |
| r10:73-74 | repeat set 96%; every CTX and key position; 88% of values; keys 34% of it | the docstring's corpus figures give 95.95%, 87.6%, 34.0%. The probe counts printed at r10:25 give 95.7%, **87.0%**, 34.1% | MATCH (note 2) |
| r10:75 | repeat pair = all pair at 28 of 30; 8 of 10 seeds bit-identical | 28/30 in both arms. Curve, stats, end and gradient norms are equal on 2904-2911 | MATCH |
| r10:77 | 2902 and 2903 diverge at the third firing (21600, a different c*); both merged | 2902: 6→2 vs 8→2; 2903: 5→3 vs 15→3; all four runs MERGED (2 share) | MATCH |
| r10:81-82 | key 0.010, VAL 0.020, CTX 0.042; pool 4: 0.084 | s89:14 and my out §6 (0.0105 / 0.0204 / 0.0421 / 0.084); not recomputed here | MATCH (from the pre-run review) |
| r10:84-85 | "the slowdown comes from the key identities varying across sequences ..., not from the layout" | an inference across screens (S87a seeds 2600s, S89 seeds 2900s). The pool is the only design change, but no arm here tests it | interpretation, not checkable |
| r10:89 | perfect gate later by "about 2.5x by the medians" | 13800 / 5400 = 2.56 | MATCH |
| r10:93 | "S79's cap of six bound 9/10 on the main line's task" | CAP6 9/10 is true. But S79's BASE at cap 3 also bound 9/10 (CAP6 vs BASE 0 vs 0), so S79 is no evidence that a larger cap helps | MATCH, context missing (note 4) |
| r10:92 | S84's guard (< 0.5) | guard 0.5 is right. On the main line's task GUARD bound 9/10 against BASE 9/10, also no difference there | MATCH, context missing (note 4) |
| README table | BASE 1, ALLMASS 3, REPEATMASS 3, NOSPLIT 0; oracle 2/2 at 10800 and 16800; the reading | the same | MATCH |
| README prose | "the four bound learned runs at 36000-40800" | 7 runs on 4 seeds | **MISMATCH (count)** |
| README prose | other figures (10800-16800 vs 2400-8400; cap used by 14400-36000; 29-31 blocked; 27-28 of 30; 96%; 28 of 30; 8 of 10 bit-identical) | as above | MATCH |

**Counts.** 43 rows: 39 MATCH (5 of them with a note or missing context), 2 MISMATCH, 1 interpretation that cannot be checked, 1 MATCH taken from the pre-run review's recomputation.

**Notes.**
1. "Three of the four" counts seeds, not runs. Say "on 4 seeds (7 runs)" in r10:55 and in the README.
2. The 88% is the docstring's 4096-sequence figure (87.6%). The probe counts printed eight lines earlier give 87.0%. Say which.
3. **The three firings on a one-to-one map were all on unlabelled targets (op_ok False), late (26400), on runs that then bound.**
   - BASE 2910 and ALLMASS/REPEATMASS 2908 each have one; it is the one-to-one firing counted at r10:27.
   - This is what my pre-run fix 5 asked to print beside the readings. Here it is harmless but not mentioned.
   - These are 3 of the 7 op_ok-False firings (BASE 3, ALLMASS 2, REPEATMASS 2).
4. **The "options" cite main-task screens where the option did not differ from cap 3 / no guard** (S79 CAP6 9 vs BASE 9; S84 GUARD 9 vs BASE 9).
   - On the main line's task the cap was not binding. Report 10's case for the cap rests on S89's own runs: 29-31 cap blocks per arm.
   - The report's argument is not wrong, but the citation suggests more than S79 shows.

### A3. Pre-registration order

All committer times are UTC. Code checked at every commit is `explore_h_s89.py`, `explore_h_s89_child.py`, `explore_h_common.py`, `explore_common16.py`, `explore_common.py`, `explore_h_ops.py`, `explore_split_target_child.py` and `explore_h_report.py`.

| commit | time | learned records | code differing from `d4a9a7d` |
|---|---|---|---|
| `d4a9a7d` screen + readings | 10-19 17:54:28 | 0 | — |
| `7820b12` / `0dd88bf` CHECK log; CHECKs passed 18:16:07 | 17:56 / 18:16 | 0 | none |
| `42d86ae` / `4324d2e` ORACLE 2901 / 2900 | 18:32 / 18:39:40 | 0 | none |
| (my screen review `ee5217b`, on claude/review-V) | 18:40:23 | | |
| `63ddc3f` "learned arms running" (launched from `4324d2e+dirty`) | 18:40:38 | 0 | none |
| `631085e` **first learned record** (BASE 2905) | 19:39:27 | 1 | none |
| ... 23 records ... | to 10-20 00:07:50 | 24 | none |
| `4867fb5` explore_h_report.py batch-10 section; restart note | 00:44:05 | 24 | **explore_h_report.py only** |
| ... 16 records (the four rerun REPEATMASS, then NOSPLIT) ... | 01:34 – 04:19:01 | 40 | report writer only |
| `13fe610` report_10, notes, s89_runs.md, README | 04:21:12 | 40 | report writer only |

**After the first learned record, nothing in the readings or the run code changed.**
- `explore_h_s89.py` and `explore_h_s89_child.py` are byte-identical at all 48 commits.
- Every record carries code SHA `6e0a80a55be3`, the SHA under which the CHECKs passed.
- The learned launch and the restarted launch both ran from a dirty tree (`4324d2e+dirty`, `06cbcb5+dirty`). The parent `explore_h_s89.py` is not in the code SHA, so what was uncommitted at launch cannot be seen. What was committed never changed, and the committed `report()` reproduces the reported reading.
- `4867fb5` adds only the report writer (`s89_section`). It prints report()'s results plus the descriptive "all = repeat" and "one-to-one before" counts. It reads no reading rule and changes none.

**My eight pre-run fixes: none taken, at any commit.** At the head:
- s89:3-4 still says "the split without marked key positions" (fix 1);
- s89:283 still prefixes a named reading with "moot" (fix 3);
- s89:106 `MAINREF` is still the branch name (fix 6);
- s89:11 still cites `a199d5e` (fix 7);
- the texts of fixes 2, 4, 5 and 8 are absent.

**Timing.** The learned arms were launched between 18:39:40 and 18:40:38, and my review was pushed at 18:40:23. My review's premise, "every fix below can land before them", was overtaken by about a minute.

**Effect on the result.**
- Fixes 3, 4 and 5 would have changed only the printing of trigger readings, which BASE's 1/10 made unreachable. Fix 6: the main-line record `WIN3_SPLIT_D8|540` is identical at `628eb7c`, `7c4ceb8` and the main head `ed9ca2b`, so the unpinned ref changed nothing.
- In substance, fix 1 appears in report 10's notes (r10:83: the keys are still at 3j+1) and in the README heading ("keys drawn from a pool of 32"), though not in the docstring.
- Fix 2 also appears in substance (r10:72-79: "REPEATMASS is ALLMASS on this layout ... says nothing about a repeat statistic that isolates the keys").
- Partial fixes 2 and 5 also appear as descriptive counts at r10:27, from the report writer.

**Errors in the referee's own files.** None found in the files I used:
- `rlib` agrees with the main line on all 42 records.
- `screen_review_s89_H.md`'s counts (6005/6272, 34%, ORACLE 2901 at 10800 in 16.6 min) are confirmed.
- One overtaken premise: the review assumed the fixes could land before the learned launch (above).
- Its operating characteristics assumed a BASE rate of 0.9 from the main line's task. On the pool task BASE was 1/10, which falls in the 0.013 "fails" branch it tabulated. That is an assumption the run refuted, not an arithmetic error.

---

## B. S89b

**Not found.** I searched after fetching, across:
- all 15 refs, for "S89b", "s89_b", "s89-b" or "S89 (b)": no content hit and no commit message;
- every path in every commit of every ref: only `explore_h_s89.py` and `explore_h_s89_child.py` carry "s89". No explore_h_*.py was added after `d4a9a7d`;
- the round-6 prompt on the main line: no S89b. It names only explore_h_s89.py.

`explore_h_s89.py` has no S89b section; it is unchanged since `d4a9a7d`.

Report 10 ends with three "options, none screened" (r10:90-93). It also says the two-stream spec revision "is not started" (r10:95-96).

**No learned-arm (or any) S89b run has started on any ref.** `s89.log` ends with "chain exit 0", and no store or log beyond s89's exists. I cannot see the container.

**Verdict: nothing to review. There is no docstring, and a screen without one is not fit to run.**

Requirements for an S89b docstring, from what S89 showed. They are numbered so the review can refer to them; the S89b docstring should be pushed and reviewed before any learned run.

1. **Arms.** Name the change being screened: budget, S84's guard (0.5) or cap.
   - Report 10's own evidence is about the cap and the flat start: 30/30 first firings at 4800 with probe 0.016-0.109, then 29-31 cap blocks per arm.
   - The main-task screens it cites showed no difference there (S79 CAP6 9 vs BASE 9; S84 GUARD 9 vs BASE 9), so S89b cannot borrow their effect sizes.
2. **The validity gate needs a feasibility prior.**
   - With BASE gated at ≥ 7/10, P(pass) is 0.172 at a true rate of 0.5, 0.65 at 0.7, 0.879 at 0.8 and 0.987 at 0.9.
   - S89's BASE was 1/10 (Wilson 0.02-0.40). A rerun of S89's rule after a change aimed only at the trigger would most likely fail the gate again.
   - The docstring should state what BASE rate it expects and why, e.g. a pilot on a free seed with the changed budget, cap or guard.
3. **Readings: give every outcome exactly one name**, with "moot" printed without a name (S89's s89:283 defect, fix 3). Run the reading function over every outcome before the run, as my out §2 did for S89.
4. **Outcome.** Use `test_stream_recipe.outcome`, read from the record's `outcome` field as S89 does. Its syntax tree is unchanged between `9c5939e` and the main head `ed9ca2b`.
5. **Seeds.** 2912-2959 are free: no record key in 2900-2959 on any remote head except S89's 2900-2911. Name the block, and keep the CHECK seed outside the screen seeds (S89 used screen seed 2902).
6. **CHECKs.** At pool 4, tie the new arm to a recorded run, as S89 did (S87a BASE|2600 and ALLMASS|2600, asserted). Pin the main-line ref to a commit (fix 6). If the cap, guard or budget changes the trigger, add a CHECK that at the old setting it equals S89's record bit for bit, e.g. BASE|2905 through its first firing.
7. **Power.** S89's rule cannot tell a trigger as good as BASE from one 10 points worse at 10 pairs (my out §5). Print the operating characteristics with the readings (fix 4).
8. **Labels.** State which statistics read key positions or stream labels. In S89 only BASE (positions 3j+1) and ORACLE (perfect gate) do, while op_ok/key_ok/map_after are descriptive. A guard reading the probe's accuracy reads the training target only, as on the main line.
9. **REPEATMASS: drop it or redefine it.** 8 of its 10 runs were bit-identical to ALLMASS's, and on this layout it is ALLMASS without about 4% of positions (fix 2).

---

## What could not be checked

- **What was uncommitted at the launches** (`4324d2e+dirty`, `06cbcb5+dirty`, CHECKs at `7820b12+dirty`). The child's 51 modules are covered by the code SHA `6e0a80a55be3` on every record. The parent `explore_h_s89.py` is not covered, but its committed text never changed.
- **The four lost in-flight runs** at the restart. They were never stored, and the log shows only that they restarted.
- **The key-frequency figures** (0.010 / 0.020 / 0.042). They come from the child's `describe()`, which I did not re-run; my pre-run review re-derived them in numpy.
- **The causal reading** "the slowdown comes from the key identities" (r10:84-85). It is a comparison across screens with different seeds and is not tested by any arm.
- **S89b**, which does not exist.

Housekeeping: to read H's files I also extracted temporary copies to my scratchpad, outside the repository. Nothing in the repository was written except this file and `check_H_report10.py`.
