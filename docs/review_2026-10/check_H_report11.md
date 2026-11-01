# Check: Session H's report 11 (S89c, S89b), the README section, and the S89b gate-lookup fix

*Referee V, 21 October 2026. `git fetch` today: claude/explore-H is at `4360cafa` (09:26:15 UTC), the commit under check.*

**What I read.** All at `4360cafa` unless noted:
- `explore_out/H/report_11.md` and `report_11_notes.md`;
- the README section (`explore_out/README.md:648-665`);
- `s89c_results.json`, `s89c.log`, `s89c_runs.md`, `s89b_results.json`, `s89b_oracle.log`, `s89b.log` and `checks.json`;
- `explore_h_s89c.py`, `explore_h_s89c_child.py`, `explore_h_s89b.py`, `explore_h_s89b_child.py`, `explore_h_oc.py`, `explore_h_run.py` and `explore_h_report.py`, at every one of the 40 commits from `c81398ca` to `4360cafa`.

**Line references.**
- "r11:NN" is a line of report_11.md. Notes line *n* is r11 line *n* + 43.
- "README:NN" is a line of `explore_out/README.md`.
- "c:NN" / "b:NN" are lines of `explore_h_s89c.py` / `explore_h_s89b.py` at the head.
- "out §N" is a section of the output of `check_H_report11.py`.

**Script.** `check_H_report11.py` prints only, reads every file with `git show`, trains nothing and runs no session code as a program. Its sources:
- **The main line's outcome.** `test_stream_recipe.outcome` and the functions it calls are cut out by ast at the main head `0efb8f79` and at the child's tree `9c5939e`. The two syntax trees are equal.
- **The referee's own outcome.** `rlib.outcome_k`, with the transition recomputed from the curve.
- **H's own code, exec'd from source with stores served from git.** That is `report()` and `summarize()` of both screens, and `s89c_schedule()` at `bccb7cbd` and at the head. Also `explore_h_report.main()` for batch 11, with `report_11_notes.md` as `--notes`; its file writes are captured in memory. And `explore_h_oc`'s `rule_s89c`, `rule_s89b` and `oc`, cut out by ast.

## Verdicts

- **Every count, pair, p-value, firing figure and printed reading in report 11 is correct.**
  - On all 32 records, the stored outcome, the main line's outcome (at both refs) and rlib's agree, and the stored transitions equal the curve's.
  - H's report writer, re-run on the stores, reproduces **report_11.md byte for byte, and s89c_runs.md byte for byte** (out §3).
- **42 rows checked:**
  - 36 MATCH;
  - 1 MISMATCH (wording: "46 in all, 39 labelled ok" is attached to the extra firings; there are 16 extra firings, 14 of them ok);
  - 1 PARTLY ("more splits do not convert the merges": on 2 of 8 seeds the extra split did make the map one-to-one, though neither run bound);
  - 1 wording note (ORACLE_DELTA 2912 is "still rising", but its last two evaluations are equal);
  - 2 interpretations that cannot be checked;
  - 1 omission. Report 11 never mentions my S89b/S89c review (`43cea898`, pushed 28 min before the first S89c record), and none of its ten fixes was applied. Report 11 does not say that the 12000 first check was chosen from S89's curves.
- **Pre-registration: clean in both screens.**
  - Neither the readings nor the run code changed after a record.
  - The S89c readings (`c81398ca`, 21:41:43) precede its first record (`56d4af99`, 22:51:52). `explore_h_s89c.py`, its child and `explore_h_oc.py` are byte-identical at all 40 commits.
  - The S89b readings (`bccb7cbd`, 21:46:05) precede its CHECKs (06:03-06:42) and its first record (`d0095a8c`, 08:21:30). The child never changed.
  - The parent `explore_h_s89b.py` changed once, at `4360cafa`, after both records (the fix below).
- **The gate-lookup fix (`4360cafa`) is what its note says.**
  - It came after the validity stage: the last ORACLE record is at 09:23:17 and the fix at 09:26:15.
  - It changes no record, count or reading. The only behaviour it changes is the "S89c complete" flag in `configure()`, from False to True. The schedule it returns is S89's in both versions.
  - The note is dated by day ("21 October 2026"), with no time. It is accurate (§C).
- **S89c's printed name.**
  - "The channel is the limit" applies by the committed rule: both new arms are below 4/10 (BASE_LATE 1/10, BASE_CAP6 2/10).
  - The BASE rerun is 2/10, so S89's failure replicated. The defect-1 case in my review (the name printing with BASE ≥ 4/10) did not arise. My proposed rule gives the same branch under its renamed name, "neither schedule change reaches 4/10".
  - Defect 2 stands: the screen does not vary the channel or the budget.
  - The recommended quotation is in §D.
- **S89b: UNTESTED.** Its gate is ORACLE_DELTA binding 2/2, and it bound 1/2. DELTA_SPLIT's ≥ 7/10 reading was never reached, and no learned arm ran (§E).

---

## A. Report 11: every number and reading

### A1. S89c (r11:3-23, r11:40; README:652)

Recomputed with the main line's outcome on the 30 records (out §1). All 30 records are ok, on attempt 1, on one CPU, under code SHA `4dda56358910`.

| where | report says | recomputed (main-line outcome; rlib; H's report()) | status |
|---|---|---|---|
| r11:7 BASE | 2/10 MINORITY, Wilson 0.06-0.51; MERGED 8 / (2 share) 8; transitions 32400, 43200; splits 30; own pair ok 25/30; probe < 0.2: 16; map 1:1 before 2; cap-blocked 24; last firings 14400 … 36000; held-out at the end 0.58 … 0.96 | 2/10 (2924, 2933); every field identical | MATCH |
| r11:8 BASE_LATE | 1/10 MINORITY 0.02-0.40; MERGED 9 / (2 share) 7, (3) 1, (4) 1; 40800; 29; 29/29; 6; 0; 15; last firings and end accuracies as listed | 1/10 (2933); identical | MATCH |
| r11:9 BASE_CAP6 | 2/10 MINORITY 0.06-0.51; MERGED 6, STREAM-PARTIAL 2; 32400, 43200; 46; 39/46; 16; 4; 1; … | 2/10 (2924, 2933); identical | MATCH |
| r11:11 validity | S89's perfect gate 2/2 at 16800 and 10800 | S89 ORACLE 2900: 16800, 2901: 10800 | MATCH |
| r11:13 LATE vs BASE | 1/10 vs 2/10; LATE only 0, BASE only 1; p 1 / 0.5 | b 0, c 1; 1 / 0.5 | MATCH |
| r11:14 CAP6 vs BASE | 2 vs 2; 0, 0; 1 / 1 | 0, 0; 1 / 1 | MATCH |
| r11:15 LATE vs CAP6 | 1 vs 2; 0, 1; 1 / 0.5 | 0, 1; 1 / 0.5 | MATCH |
| r11:19 reading | "the channel is the limit (BASE_LATE 1/10 (BASE only 1, …); BASE_CAP6 2/10 (…); BASE 2/10)" | H's `report()` prints exactly this. The committed rule (`rule_s89c`, by ast) gives the same on the per-seed patterns {(0,0,0): 8, (1,0,1): 1, (1,1,1): 1}, and so does my restatement of c:21-24 | MATCH (the name: §D) |
| r11:21 caveat | S89's CAVEAT | `s89.CAVEAT` (generated) | MATCH |
| r11:23 OC | both 0.3: schedule 0.27/0.23/0.12, channel 0.42/0.50/0.65; 0.5: 0.64/0.69/0.62, 0.03/0.07/0.17; 0.7: 0.87/0.93/0.95; one 0.5: 0.47/0.55/0.62, channel 0.17 | `explore_h_oc.oc` exec'd: identical (out §8) | MATCH |
| r11:40 runtime, CHECKs | 30 runs, 25.8 CPU-hours, 2.10 GHz, 4dda56358910; CHECKs 21:59:18, 555d681+dirty, 15.8 min | Σ secs_wall 25.83 h; `checks.json` identical; 5 CHECK rows ok, 0 FAIL in `s89c.log` | MATCH |
| README:652 | BASE 2, BASE_LATE 1, BASE_CAP6 2; validity 2/2; "the channel is the limit" | as above | MATCH (the name: §D) |

### A2. S89b (r11:25-42; README:653)

| where | report says | recomputed | status |
|---|---|---|---|
| r11:29 ORACLE_DELTA 2912 | STREAM-PARTIAL one-to-one; None; 0.31 / 0.49 / 0.59 / 0.875; streams 0.85-0.91, map [0..7]; 161 min | the same (0.310 / 0.490 / 0.592 / 0.875; streams 0.848-0.907; 160.8 min) | MATCH |
| r11:30 ORACLE_DELTA 2913 | BOUND ROUTED; 24000; 0.29 / 0.88 / 0.96 / 0.964; 0.95-0.97; 99 min | 0.293 / 0.881 / 0.956 / 0.964; 0.945-0.974; 99.0 min | MATCH |
| r11:32 validity | bound 1/2 ({2912: None, 2913: 24000}) → UNTESTED; learned runs 0 | `report()["oracle"]` identical; no learned record | MATCH |
| r11:36 reading | "UNTESTED (ORACLE_DELTA bound 1/2); no learned arm runs" | `report()` prints exactly this (b:274-275) | MATCH |
| r11:42 runtime, CHECKs | 2 runs, 4.3 CPU-hours; 9cb518b59c14; CHECKs 06:42:32, 568a12f+dirty, 38.9 min | Σ secs_wall 4.33 h; `checks.json` identical; 9 CHECK rows ok, 0 FAIL | MATCH |
| README:653 | perfect gate 1/2 (2913 at 24000; 2912 at 0.875, one-to-one); no learned arm; UNTESTED | as above | MATCH |

### A3. The notes (r11:44-105) and the README prose (README:655-663)

| where | report says | recomputed | status |
|---|---|---|---|
| r11:45-48 | neither arm reached 4/10 (1, 2 against BASE 2); no pair favours a new arm; OC "below about +0.2" | as A1 | MATCH |
| r11:49-50 | S89b UNTESTED; gate 1 of 2 | as A2 | MATCH |
| r11:53-54 | S89c CHECKs: the mechanics; BASE and CAP6 = S89's BASE\|2902 through 12000; LATE = NOSPLIT\|2902 | 5 CHECK rows ok in `s89c.log`. The ties were verified in my screen review §C4 | MATCH |
| r11:55-58 | S89b CHECKs: modules at a581027 (= F's 9a6c2bf); fast_checks; no parameter added; pool 4 DELTA_SPLIT\|583 to 7200, DELTA_SLOW\|583 to 4800, CEIL_DELTA\|580 to 2400; pool 32 threshold 0 = NOSPLIT to 4800 | 9 CHECK rows ok in `s89b_oracle.log`, with exactly these contents. X's DELTA_SPLIT\|583 fired at [4800, 12000] (8ae660d) | MATCH |
| r11:61 | 30 + 2 runs; 25.8 + 4.3 h; no run failed | all ok, attempt 1 | MATCH |
| r11:62-63 | two restarts; same CPU and code SHA; no saved run changed | 2 "resumed after" segments in `s89c.log`. All 32 records are identical at their first commit and at the head, and none was dropped | MATCH |
| r11:64-68 | log rewritten by `pull --rebase` autostash; last seven runs' lines and summary lost; regenerated with a note; driver no longer pulls; S89b validity restarted under the fixed driver | 7 records have no completion line: BASE_CAP6 2927-2933. The note is at the log's end. The `f3b18e99` diff removes the pull. `s89b_oracle.log` starts at git `f3b18e9`. "Before it had run anything" cannot be seen | MATCH |
| r11:69-72 | gate-code defect, fixed after the validity stage; it would have kept the learned arms unqueued; UNTESTED either way | §C | MATCH |
| r11:75-77 | BASE_LATE probe at 12000 0.09-0.38 in every run; 6 of 29 firings at probe < 0.2; held-out first ≥ 0.3 at 12000-22800 against BASE's 12000-18000 | 0.094-0.375; 6 of 29; 12000-22800 and 12000-18000 | MATCH |
| r11:79-80 | CAP6 is BASE until the fourth firing; on 2924 and 2933 it never fired a fourth time, and the runs are identical | identical records (curve, stats, end, grad, checks) on 2924 and 2933. On the other 8, the first differing curve point is 1200 after the 4th firing | MATCH |
| r11:81 | "the extra firings came at 24000-40800 (46 in all, 39 labelled ok)" | **16 extra firings (4th-6th), 14 ok**, at 24000-40800, on all 8 seeds. 46 / 39 are *all* of CAP6's firings (table r11:9) | **MISMATCH (wording)** |
| r11:81-82 | ended 6 MERGED, 2 STREAM-PARTIAL one-to-one, at 0.59-0.89 | 6 MERGED (2 share), 2 STREAM-PARTIAL one-to-one; 0.589-0.886 | MATCH |
| r11:78 heading; README:655-656 | "More splits do not convert the merges" / "splits … that do not convert the merges" | On **2926 and 2928** BASE ended MERGED (2 share) and CAP6 ended one-to-one: the 4th+ split separated the shared channel. Neither bound (min stream 0.563 and 0.506). "do not convert the merges into binds" is accurate | **PARTLY** |
| r11:83 | one check blocked by the cap of 6 | 1 (2927 at 38400) | MATCH |
| r11:84-85 | last evaluation is the maximum in 4 of 8, 5 of 9, 4 of 8 failures; BASE bound at 32400 and 43200 | 4 / 5 / 4 using ≥. For CAP6 one of the 4 is a tie, so 3 are strictly above every earlier evaluation. 32400, 43200 | MATCH |
| r11:89 | 2913 0.88 at 12000, bound at 24000 | 0.881; 24000 | MATCH |
| r11:90-91 | 2912 at 0.30 until about 9600, then climbed to 0.875 by 43200, "still rising"; streams 0.85-0.91; one-to-one | 0.298-0.313 through 8400, 0.333 at 9600; 0.875; 0.848-0.907; map [0..7]. **42000 and 43200 are equal (0.87451171875)**: the curve is level over the last 1200 updates after +0.022 in the previous 1200 | MATCH, wording note: "still rising" → "at its maximum, level over the last evaluation" |
| r11:92 | at pool 4 the same gate binds at 1200-2400 (X's CEIL_DELTA 2400, 1200) | CEIL_DELTA\|580 2400, \|581 1200 (8ae660d) | MATCH |
| r11:93-94 | "what is slow on delta memory at 32 key types is recall within a stream's own channel, not routing" | an inference from the perfect gate, which routes by construction. No arm tests it | interpretation |
| r11:96-97 | "the schedule was the limit" did not apply | correct (no arm beats BASE 4 vs 0) | MATCH |
| r11:98-99; README:656-658 | perfect gates late (Hebbian 10800, 16800; delta 24000, none by 43200; pool 4: 2400-8400 and 1200-2400); about half the failures still rising | S89 ORACLE 10800 / 16800; S87a ORACLE 2400 / 8400; X 1200 / 2400. 13 of the 25 S89c failures end at their maximum | MATCH |
| r11:99-100; README:658 | "the budget and the size of the key vocabulary, not the trigger, are what this task tests first" | not tested. It also disagrees with the printed name "the channel is the limit" (§D) | interpretation |
| README:659-663 | the incidents | as r11:64-72 | MATCH |
| report 11 and README as a whole | — | **Neither cites `screen_review_s89bc_H.md` (`43cea898`, 22:23:59), which came 28 min before S89c's first record and 7.7 h before S89b's CHECKs.** None of its fixes was applied: S89c 1-5, S89b 1-5 (§B). Neither says that 12000 was read off S89's curves. r11:17 and r11:34 say the readings were "fixed … before any run", which is true | **OMISSION** |

**Counts: 42 rows.** 36 MATCH, 1 MISMATCH (wording), 1 PARTLY, 1 wording note (r11:90-91), 2 interpretations, 1 omission. One clause cannot be checked: S89b's validity stage was restarted "before it had run anything".

Also outside report 11, my S89 text fixes (screen_review_s89bc_H §A, fixes 1-3) are still not applied at the head:
- README:636 "the four bound learned runs";
- report_10.md "seven of the nine";
- the review cited as "3c772c3" in `explore_h_s89.py`.

---

## B. Pre-registration order

All committer times are UTC. Code is compared at all 40 commits (out §7).

| commit | time | S89c records | S89b records | code differing from its first version in the range |
|---|---|---|---|---|
| `c81398ca` S89c readings, `explore_h_oc.py` | 10-20 21:41:43 | 0 | 0 | — |
| `bccb7cbd` S89b readings | 21:46:05 | 0 | 0 | — |
| `555d6816` / `cf67855d` S89c CHECKs; passed 21:59:18 | 21:46:32 / 22:00:07 | 0 | 0 | none |
| (my review `43cea898` on claude/review-V) | 22:23:59 | | | |
| `56d4af99` **first S89c record** (BASE 2927) | 22:51:52 | 1 | 0 | none |
| … 28 records, two restarts (`69f1d28c`, `5b03a973`) … | to 10-21 06:00:55 | 30 | 0 | none |
| `f3b18e99` "S89c complete"; driver: push without pull | 06:02:48 | 30 | 0 | `explore_h_run.py` (push only) |
| `568a12f0` S89c report section; S89b CHECK log | 06:03:25 | 30 | 0 | + `explore_h_report.py` (writer only) |
| `3cf47d4a` S89b CHECKs passed (06:42:32) | 06:43:17 | 30 | 0 | as above |
| `d0095a8c` **first S89b record** (ORACLE_DELTA 2913) | 08:21:30 | 30 | 1 | as above |
| `bbb8df93` ORACLE_DELTA 2912 | 09:23:17 | 30 | 2 | as above |
| `4360cafa` report 11; S89b gate-lookup fix | 09:26:15 | 30 | 2 | + `explore_h_s89b.py` (`s89c_schedule` + docstring note) |

- **S89c.** `explore_h_s89c.py`, `explore_h_s89c_child.py` and `explore_h_oc.py` never changed. Every record carries code SHA `4dda56358910`, the SHA under which the CHECKs passed.
  - The two changed files are outside that SHA. The code-SHA module list in `s89c.log` (52 modules) contains no parent module.
  - `explore_h_run.py` (f3b18e99) changed only the push, after the last record.
  - `explore_h_report.py` (568a12f0, 4360cafa) changed only the writer.
- **S89b.** `explore_h_s89b_child.py` never changed. The CHECKs and both records are under `9cb518b59c14`, and the 54-module list contains no parent module.
  - The parent's one change (`4360cafa`) comes after both records. It touches only `s89c_schedule()` and the docstring: in `ast` terms, functions changed: [`s89c_schedule`]; module-level assignments changed: 0.
  - `report()` is unchanged.
- **Records.** Every one of the 32 records is identical at its first commit and at the head, and none was ever dropped.
- **My review's fixes: none taken, at any commit.**
  - Absent at the head: "does not replicate" (S89c fix 1), "neither schedule change" (fix 2), a POST-HOC note (fix 3) and the record citation `7939dd4` (fix 5).
  - Also absent: S89b's "EXPECTED RATE" (fix 1), "LABELS:" (fix 4), "mixed schedules" (fix 3) and the F head `53578896` (fix 5).
  - Printing the caveat and OC (fix 2 / fix 4) was done in the report writer, not the console: r11:21-23 prints S89c's caveat and OC, and r11:38 prints S89b's caveat only.
  - My review came 28 min before the first S89c record. The S89c runs had been running since about 22:00, but every fix was parent-only and could still have landed. The S89b fixes could have landed before its CHECKs, which started at about 06:03.
  - **Effect on the result: none on counts.** For the S89c name, see §D. S89b never reached a reading that the fixes would have touched.

---

## C. The gate-lookup fix (`4360cafa`)

**The diff** (b:145-149; the CORRECTION note at b:79-82):

```
-    c16.STATE.setdefault("shas", {})[s89c.NAME] = st["meta"]["code_shas"][-1]
-    c16.STATE.setdefault("batch_cpu", ec.cpu_model())
-    c16.STATE.setdefault("other_cpus", [])
+    if not isinstance(c16.STATE.get("shas"), dict):
+        c16.STATE["shas"] = {}
+    c16.STATE["shas"][s89c.NAME] = st["meta"]["code_shas"][-1]
+    c16.STATE["batch_cpu"] = c16.STATE.get("batch_cpu") or ec.cpu_model()      # the harness starts it at None
+    c16.STATE["other_cpus"] = c16.STATE.get("other_cpus") or []
```

**The mechanism is as stated.**
- `explore_common16.STATE` starts as `dict(batch_cpu=None, …)` (explore_common16.py:37). `explore_h_run.main` calls `m.configure()` *before* it sets `STATE["batch_cpu"]` (explore_h_run.py:94-101).
- So `setdefault` left `None`, `cpus_for` returned `[None]`, `runs_of` found no S89c run, and `s89c.report()` returned incomplete.
- I exec'd `s89c_schedule()` from both versions with the harness's STATE and the head's S89c store:
  - `bccb7cbd` gives **(False, {})**;
  - `4360cafa` gives **(True, {})** (out §6).

**Effect.**
- **The schedule is the same.** It is {} (S89's: first 4800, cap 3) in both versions, because S89c's reading is not "the schedule was the limit". Only the "S89c complete" flag changes.
- **Before the fix, the learned arms could never have been queued** (`ok = done and … and cdone`, b:176), whatever ORACLE_DELTA did. They would not have run on a wrong schedule. The note's claim is correct.
- **No record, count or reading changes.** The S89b store is identical at `bbb8df93` and `4360cafa`. `report()` does not call `s89c_schedule()`. The child is unchanged.
- **The defect is on the record.** `s89b.log`, first committed in this same commit, reads "S89c complete: False; schedule S89s (first 4800, cap 3) -> no learned arm is queued". With ORACLE_DELTA at 1/2, nothing would have been queued in either version.

**Timing and the note.**
- The fix came 2 min 58 s after the last ORACLE record (`bbb8df93`, 09:23:17), with no learned arm run or queued.
- The note is dated "21 October 2026, after the validity stage; no learned arm had run", with no time; the commit gives 09:26:15.
- Its four claims are accurate:
  - the cause (`batch_cpu` None; `setdefault`);
  - that no S89c run was found;
  - that the learned arms would have stayed unqueued;
  - that the screen is UNTESTED either way.
- r11:69-72 and README:662-663 say the same.
- One addition would make it complete: "the schedule returned is S89's in either version (S89c did not read 'the schedule was the limit'), so the fix changes only the completeness flag".

**Why the CHECKs missed it.** No CHECK exercises `configure()`. My S89b fix 3 asked for a CHECK of the inherited schedule. It would not by itself have run `configure()`, so I do not claim it would have caught this.

---

## D. Item 3: S89c's printed name

**The inputs, recomputed:**
- **BASE rerun: 2/10** (2924 at 43200, 2933 at 32400), Wilson 0.06-0.51.
  - S89's BASE was 1/10 on 2902-2911 (Wilson 0.02-0.40).
  - Fisher one-sided P(rerun ≥ 2 | pooled 3/20) = 0.50. Pooled: 3/20 (Wilson 0.05-0.36).
  - **S89's failure replicated.** The rerun stays below the 4/10 premise my review proposed ("S89's failure does not replicate here" needs BASE ≥ 4/10).
- **BASE_LATE: 1/10** (2933). Against BASE: LATE only 0, BASE only 1; p(LATE > BASE) = 1.
  - Its first firings were at 12000-28800 (3 at 12000).
  - Its held-out accuracy first reached 0.3 at 12000-22800, against BASE's 12000-18000.
- **BASE_CAP6: 2/10** (2924, 2933). Against BASE: 0 vs 0; p = 1.
  - **Its two binds are BASE's own runs**: the records are identical on 2924 and 2933, where BASE never reached a fourth firing.
  - CAP6 differs from BASE only through its 16 extra firings at 24000-40800 on the other 8 seeds.

**Was the printed name's condition met by the committed rule?**
- **Yes.** c:23 and `rule_s89c` require both new arms < 4/10: 1/10 and 2/10. No arm beats BASE 4 vs 0, so "the schedule was the limit" did not apply, and the name is the one the rule prints.
- **Defect 1 of my review (the name printing with BASE ≥ 4/10) did not arise.**
- **My proposed fixed rule lands on the same branch**: premise first, then "neither schedule change reaches 4/10". The substance is identical; only the name differs.

**What the screen does and does not test.**
- **It varies the trigger's schedule only:** the first eligible check (4800 → 12000, a value read off S89's held-out curves) and the cap (3 → 6).
- **It does not vary the memory (the "channel") or the 43200 budget.**
  - S89b, the memory change, is UNTESTED, because its perfect gate bound 1/2.
  - 13 of S89c's 25 failures end at their curve's maximum.
  - Report 11's own closing paragraph (r11:96-100) names "the budget and the size of the key vocabulary, not the trigger" as what the task tests first. That is a different cause from the printed name.
- **"The channel is the limit" therefore names a cause the screen did not test.**
- By the rule's own OC at BASE 0.1, this reading also prints 0.42 / 0.50 / 0.65 of the time (ρ = 0 / 0.5 / 1) when both schedule arms are at 0.3, a tripling of the rate.

**Recommended quotation** (report 11 r11:19 and r11:45-48, the README:652 table cell, and any later citation):

> S89c: the pre-fixed reading "the channel is the limit" applies by its rule. Both schedule arms are below 4/10 (BASE_LATE 1/10, BASE_CAP6 2/10). The BASE rerun is 2/10, so S89's 1/10 failure replicated. No pair favours a new arm (BASE_LATE only 0 vs BASE only 1; BASE_CAP6 0 vs 0; p = 1). The screen varies only the trigger's schedule: a first check at 12000, a value chosen from S89's curves, or a cap of 6. It does not vary the memory or the 43200 budget, so the name does not show a cause. What it shows is that neither schedule change reached 4/10. By the rule's operating characteristics, a gain of about +0.2 still gives this reading about half the time. BASE_CAP6's two binds are BASE's own runs (identical records on 2924 and 2933).

For the README cell, short form:

> "the channel is the limit" (pre-fixed name; read: neither schedule change reaches 4/10; BASE rerun 2/10 replicates S89; memory and budget not varied)

---

## E. Item 4: S89b

**The committed gate** (b:34-35, b:77-78, b:274-275) is ORACLE_DELTA binding both 2912 and 2913 within 43200.
- Only then do the learned arms run.
- "DELTA_SPLIT < 7/10 → the labelled recipe itself fails here" is the *first reading* among the learned arms (b:38). It is not the queue gate.

**Outcome.**
- **ORACLE_DELTA bound 1/2.**
  - 2913: BOUND ROUTED at 24000 (0.964; streams 0.945-0.974).
  - 2912: STREAM-PARTIAL one-to-one, no transition. 0.875 at the end, level over the last evaluation; streams 0.848-0.907; map [0..7].
- Main-line outcome and rlib agree.
- **The gate failed → UNTESTED.** That is the printed reading: "UNTESTED (ORACLE_DELTA bound 1/2); no learned arm runs" (`s89b_oracle.log` final line; r11:36). It is correct by the rule.
- DELTA_SPLIT, DELTA_NOSPLIT and DELTA_ALLMASS have no records. The ≥ 7/10 reading and every trigger reading were never reached.

**Operating characteristics against the outcome.**
- The docstring's OC (b:53-61; `OC`, not printed in report 11) covers only the learned-arm readings, at the pool-4 rates.
- **There is no OC or feasibility prior for the ORACLE gate itself.** My review's fix 1 asked for one, with the cheap guard of "read ORACLE_DELTA's transitions against S89's Hebbian 10800/16800 before queuing".
- With a true per-seed rate q, P(2/2) is q²: 0.25 at q = 0.5, 0.49 at 0.7, 0.81 at 0.9. Two seeds cannot say much more than "the delta perfect gate does not reliably bind this task by 43200".
- At pool 4 the same gate bound at 1200-2400 (X's CEIL_DELTA), on Hebbian memory at 2400-8400 (S87a), and here at 24000 and never. That is consistent with my review's warning that the gate's feasibility at pool 32 was unknown.
- Report 11's statement "Nothing about the trigger on delta memory is learned here" (r11:50) is the honest reading.

**Labels.** No learned arm ran, so no label-using statistic contributes. The ORACLE arm is the perfect gate (stream labels), as declared.

---

## What could not be checked

- **What was uncommitted at the CHECKs and restarts.** These ran from `555d681+dirty`, `568a12f+dirty` and `0d9db45+dirty` / `4be3d11+dirty`. The children are covered by their code SHAs. The parents are not, but their committed text never changed before the records.
- **Whether S89b's validity stage was started earlier and restarted "before it had run anything"** (r11:67-68). `s89b_oracle.log` begins at `f3b18e9` (the fixed driver), and no earlier S89b log or record exists in any commit.
- **The in-flight runs lost at the two restarts.** They were never stored.
- **The interpretations at r11:93-94 and r11:99-100** (recall within a channel; budget and key vocabulary). No arm tests them.
- **The CHECK internals.** I took the CHECK rows from the logs (all ok) and relied on my screen review §C4 for the S89c ties to S89's records. I did not re-derive them here.
- **The two-stream spec revision's status** (r11:105).

## Errors in the referee's own files

**None found in the files used here.**
- `rlib` agrees with the main line on all 32 records.
- `screen_review_s89bc_H.md`'s premises hold where the records now test them:
  - BASE_LATE usually not firing at 12000: 7 of 10 did not;
  - CAP6's mechanics: identical to BASE through the third firing;
  - X's DELTA_SPLIT|583 firing at [4800, 12000].
- Two expectations were overtaken, not wrong:
  - The review said ORACLE_DELTA "could start at any time". It started only after S89c was complete, at about 06:43.
  - B4's expected print, "moot", was never reached, because the ORACLE gate failed first.
- `check_H_report10.md`: no error found. Its §A non-MATCH rows (README:636 "four") are still unfixed on H's branch, as noted in §A3.

Housekeeping: only this file and `check_H_report11.py` were written. Nothing was committed or checked out, and no session code was run as a program. The script's output was kept in my scratchpad, outside the repository.
