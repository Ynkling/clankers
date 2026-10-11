<!-- Appendix to docs/review_2026-10/preregistration.md. Produced for the referee by a delegated read-only audit of branch G (git log/show/diff and JSON via git show only); the referee spot-checked its load-bearing claims against git (see preregistration.md). -->

# Pre-registration audit: Session G (origin/claude/explore-G, tip ed5e06b 2026-10-10T03:45:53+00:00)

Method: read-only use of `git log` (incl. `--full-history`, `-p`, `-g`), `git show`, `git diff`, plus `git rev-parse`/`merge-base`
for blob/ancestry checks; `python3` on JSON from `git show`. Repository not modified. **Timezones:** every commit time below is
the committer date as printed by `git log --format='%h %cI %s'`, all `+00:00` (UTC; author date = committer date on the spec
commits). Store `meta.provenance.started` comes from `time.strftime("%Y-%m-%d %H:%M:%S")` (explore_common.provenance), local
container time **with no zone recorded**; it agrees with UTC (s56_read_path.json and s56_hebb.log write "15:25:17 UTC" for the
same session, and every `started` falls between the commits that bracket it), but the zone is inferred, not recorded.
Commit dates are self-reported by the committer; git establishes only the order (linear history, no merges on the branch).

## Global findings
- Every screen file and every code file it imports from the branch was committed **exactly once** and never changed:
  `git log --full-history -p origin/claude/explore-G -- <f>` shows one commit each, and the tip blob equals the first blob, for
  explore_g_{probe,probe_child,common,register,register_child,regmodel,layers,rhl_child,rhl_parent,register3,register3_delta}.py,
  explore_delta_mem.py, explore_f_tasks.py. The only non-`explore_out/` commits after e4d8107~1 are the five spec commits
  (e4d8107, a7bbee0, 0160269 [S57], 9114843, 26c4ab1). So item 3 is **"none"** for every screen.
- The code that produced the records equals the committed code: each store's `meta.sha_rows` lists [module, sha1(source)];
  recomputing sha1 over `git show <spec>:<module>.py` matches **all** rows (S55 24/24 vs e4d8107; S56 26/26 vs a7bbee0; S71, S72,
  S73 30/30 vs 9114843/26c4ab1), and every record key at the tip carries that one code SHA (0c559dc511fe, 70304e0e7121,
  5366c9b9a792). The "+dirty" in some `provenance.git` values is therefore not a code change in imported modules (what was dirty
  cannot be determined from git; likely uncommitted outputs). Note: the parent screen file holding the docstring is *not* in the
  code SHA; its stability rests on the git history above.
- No labelled dry runs in any G store (no key/field containing "dry"; no dry_* file under explore_out/G/).
- The branch has no copy of docs/reading/parallel_sessions_prompts.md; comparison is against the main line
  (`origin/claude/bdh-growth-hebbian-inference-w90069`). The S55-S57 block was added at ad8a92e 2026-10-08T03:50:57+00:00 and is
  unchanged at the main-line tip (the only later hunk starts at line 200); the S71-S73 block was added at
  6f6e583 2026-10-10T00:05:14+00:00 (18 min before 9114843).
- The local reflog has a single entry (`fetch origin: storing head`), so whether the remote branch was ever force-pushed /
  rewritten **cannot be determined from git**.

## S55 explore_g_probe (probe) — seeds 360-369 (+ batch-2 reruns 160-169)

| item | value |
|---|---|
| readings first committed | e4d8107 2026-10-09T14:37:10+00:00 "Session G S55 (probe): specification, readings and CHECKs, committed before any run" |
| CHECKs file | checks_s55_20261009_143838.json (local time in name, no zone) |
| store `started` / git | "2026-10-09 14:38:39" (no zone) / e4d8107 (clean) — 89 s after the spec commit |
| first commit with records | 4da0c21 2026-10-09T14:49:31+00:00 — 6 records (FAR_L3_P4 160-163, ORACLE_P8 360-361; no L3S_P8 yet) |
| complete | 6cd36ff 2026-10-09T15:25:43+00:00 (22/22; report 1 in same commit) |
| changes after first run | none (file has a single commit; `git log -p --full-history` checked) |
| report matches docstring | yes (report_1.md, 6cd36ff; re-computed medians from the store agree) |

Docstring readings as first committed (e4d8107, verbatim):
```
READINGS (fixed now; per arm, on the median over the arm's usable runs of the held-out accuracy; the count of runs that
meet each criterion is printed with Wilson and band)
  "the cue is present at layer l"   median accuracy >= 0.9 at both K and V, for l in {L2, L3, final};
  "absent"                          median accuracy <= 0.6 at K and at V for every l in {L2, L3, final}
                                    (then the fix is upstream and S56's register is mandatory, not optional);
  otherwise                         "partial at l" for each l with both medians in (0.6, 0.9), "present at K only" /
                                    "at V only" where one side passes 0.9 and the other does not.
The arm whose reading drives S56's design is L3S_P8 (the window gate's own stack on the P = 8 layout, the configuration S56
modifies). FAR_L3_P4 and ORACLE_P8 are read the same way and printed beside it.
VALIDITY: ORACLE_P8 binds on >= 1 of its 2 seeds within 24000, else L3S_P8's training outcome is UNTESTED (the probe
readings stand: they describe the trained stack, not an outcome).
```
Prompt (main line, ad8a92e): `Readings: "the cue is present at layer ℓ" if held-out accuracy ≥ 0.9 at both K and V; "absent" if
≤ 0.6 at every layer > 1 (then the fix is upstream and S56's register is mandatory, not optional).`
Differences: (a) docstring reads the **median over runs**; prompt does not say how runs are aggregated. (b) "present" is read only
for {L2, L3, final}, not ℓ = 1; "absent" adds the **final** layer to "every layer > 1". (c) docstring adds intermediate categories
("partial", "K only"/"V only") and names L3S_P8 as the deciding arm. (d) Design differs from the prompt: prompt says "no training
of new models" on recorded batch 2/16 P=4 runs plus fresh LOCAL3_SLOW on **RandHeaderTask**; docstring trains fresh L3S_P8 on the
**fixed header P = 8** (citing "the user's instruction of 9 October", which is **not recorded in git**), reruns batch-2 far_L3
(160-169, outside G's seed block, flagged as reproduction only), drops batch-16 LOCAL3_SLOW P=4, adds ORACLE_P8 (360-361).
Report 1 states "absent" for L3S_P8 (medians 0.55/0.58, 0.55/0.56, 0.57/0.56 at L2/L3/final), "partial at L3 and final" for
FAR_L3_P4, "absent" for ORACLE_P8, validity 1/2 VALID — all consistent with the docstring's rule and the stored records.
Note on a7bbee0's claim "before S55's results": at a7bbee0 (14:49:13) the S55 runs had started 14:38:39 and 6 records were in the
store committed 18 s later (4da0c21); none were L3S_P8, the only arm S56's read-path rule uses. Whether they were read cannot be
determined from git.

## S56 explore_g_register (register) — Hebbian 370-379, delta 390-399

| item | value |
|---|---|
| readings first committed | a7bbee0 2026-10-09T14:49:13+00:00 "Session G S56 (register): specification, readings, read-path rule and CHECKs, committed before any S56 run and before S55's results" |
| read-path decision | s56_read_path.json committed 6cd36ff 2026-10-09T15:25:43+00:00: res_layers [2, 3], "neither L2 nor L3 present", time "2026-10-09 15:25:17 UTC" (matches the rule applied to S55's L3S_P8 medians) |
| CHECKs files | checks_s56_20261009_152732.json (Hebbian), checks_s56_20261009_192807.json (delta) |
| store `started` / git | Hebbian: "2026-10-09 15:27:33" / 6cd36ff+dirty (store first committed bada52b 15:27:52, 0 records; s56_hebb.log header shows launch at 4da0c21+dirty). Delta: "2026-10-09 19:28:08" / 484b9ed (ff9b85a 19:28:22 "S56 delta started") |
| first commit with records | 7a2fcc9 2026-10-09T15:38:00+00:00 — 4 records (BASELINE 370, ORACLE_H 370/371, RESGATE_ONLY 370). First delta records: 4e53b6e 2026-10-09T19:36:40+00:00 |
| changes after first run | none (single commit; `git log -p --full-history`) |
| report matches docstring | yes, with one caveat the report itself raises (R2 below) |

Docstring readings as first committed (a7bbee0, verbatim):
```
READINGS (fixed now; counts out of 10; Wilson 95% and band with every count; paired one-sided exact McNemar on 370-379)
  R1 "the register carries the cue"    REG_NUDGE BOUND ROUTED >= 9/10 (validity of the architecture)
     "architecture insufficient"       REG_NUDGE BOUND ROUTED < 6/10 (then: beta^R at CTX vs KEY/VAL — a write that does not
                                       separate CTX from K/V is the write decision; a separated write with an unrouted gate
                                       is the read path)
     otherwise                         "neither"
  R2 "discovered"                      REG or REG_NOSG BOUND >= BASELINE BOUND + 3 with that arm's KEY failures <= 2
     "not discovered"                  otherwise
  R3 vs SINGLE_GDN                     "the register beats the single GDN channel" if (best of REG, REG_NOSG) BOUND >=
                                       SINGLE_GDN BOUND + 3; "the single GDN channel is at least as good" if SINGLE_GDN
                                       BOUND >= that arm's BOUND; otherwise "neither"
  R4 "the register adds to the residual gate"  REG BOUND >= RESGATE_ONLY BOUND + 3; "it does not" if REG <= RESGATE_ONLY
  R5 "SG matters"                      |REG BOUND - REG_NOSG BOUND| >= 3 (direction printed); otherwise "SG does not matter"
  Delta: R1 and R2 on REG_NUDGE_D / REG_D vs BASELINE_D, same thresholds.
  S57 TRIGGER (explore_g_latch runs only if): R2 is "not discovered" AND REG's median (over runs) end-of-run beta^R at CTX
  tokens lies in [0.2, 0.8].
```
plus `VALIDITY: an arm family (Hebbian; delta) is VALID if its oracle binds on >= 1 of its 2 seeds within 24000; otherwise its
readings are UNTESTED.` and the read-path rule (L2 present -> (2, 3); else L3 present -> (3,); else (2, 3)).
Prompt (ad8a92e, verbatim): `Readings fixed now: "the register carries the cue" if REG_NUDGE BOUND ROUTED ≥ 9/10 (validity);
"discovered" if REG (either variant) beats BASELINE by ≥ 3 bound runs with KEY splits ≤ 2; "architecture insufficient" if
REG_NUDGE < 6/10 (then report β^R's values at CTX vs K/V tokens: is it the write decision or the read path?).`
Differences: (a) prompt's "KEY splits ≤ 2" becomes "that arm's **KEY failures** ≤ 2" (failure class of *unbound* runs) — this
narrowing is in the first commit, not a later edit; (b) R1 "insufficient" made explicit on BOUND ROUTED, "neither" added;
(c) R3-R5, the read-path rule and an explicit S57 trigger ("R2 not discovered AND median end β^R at CTX in [0.2, 0.8]"; prompt:
"REG fails with β^R settling in 0.2-0.8") added; (d) task: fixed header P = 8 instead of RandHeaderTask P_b ∈ {2..6} (again
"the user's instruction of 9 October", not in git); seeds 370-379 (prompt: 370-389, 380-389 "held back") and delta 390-399
(prompt 390-409); added arms SINGLE_GDN, BASELINE_D, ORACLE_D; validity ≥ 1/2 oracle seeds.
Report 2 (427db36 2026-10-09T23:18:01+00:00) restates R1 "≥ 9 carries, < 6 insufficient", R2 "REG-type BOUND ≥ BASELINE + 3,
KEY ≤ 2", R3-R5, the S57 trigger and the ≥ 1/2 validity rule — consistent with the docstring. Applied readings check against the
store (BOUND: REG_NUDGE 0/10, REG_D 6/10 vs BASELINE_D 2/10, REG_NUDGE_D 8/10, ORACLE_H 1/2, ORACLE_D 2/2). Caveat (raised by
the report itself): delta R2 is reported "discovered (REG_D)" by the letter of "KEY failures ≤ 2" (0 unbound KEY), while 3 of
REG_D's 6 bound runs bound by key splits; under the prompt's "KEY splits ≤ 2" it would read "not discovered". The report labels
this; it did not alter the rule. The post-hoc ORACLE23 used in report 2 comes from S57's spec (0160269 17:30:16), labelled
post hoc there.

## S71 explore_g_layers (layers) — seeds 700-709

| item | value |
|---|---|
| readings first committed | 9114843 2026-10-10T00:23:50+00:00 "Session G follow-up S71 (layers): specification, readings and CHECKs, committed before any run; ..." |
| CHECKs file | checks_s71_20261010_002447.json |
| store `started` / git | "2026-10-10 00:24:49" (no zone) / 9114843 (clean); store first committed 4384493 00:25:18 with 0 records |
| first commit with records | 191d177 2026-10-10T00:29:07+00:00 — 4 records (ORACLE_23_D 700, ORACLE_ALL_D 700, ORACLE_ALL 700/701) |
| complete | 40e043d 2026-10-10T01:09:24+00:00 (50/50); report 3 at 35de6db 01:10:04 |
| changes after first run | none (single commit; `git log -p --full-history`) |
| report matches docstring | yes |

Docstring readings as first committed (9114843, verbatim):
```
VALIDITY: the Hebbian family is VALID if ORACLE_ALL binds >= 8/10; the delta family if ORACLE_ALL_D binds >= 8/10. Otherwise
that family's readings are UNTESTED. ORACLE_ALL and ORACLE_ALL_D are also the validity oracles of S72 and S73.
READINGS (BOUND, paired on 700-709; "beats" = one-sided exact McNemar p < 0.05; "A >= B - 1 pair" = (B only) - (A only) <= 1)
  "layer 1 is the block"                    if ORACLE_ALL beats ORACLE_23
  "layer 1 is not needed"                   if ORACLE_23 >= ORACLE_ALL - 1 pair
  "layer 1 alone suffices"                  if ORACLE_1 >= ORACLE_ALL - 1 pair
  "delta tolerates an unrouted layer 1"     if ORACLE_23_D >= ORACLE_ALL_D - 1 pair
  (each printed with b, c, p; none of them is printed as applying if its family is UNTESTED)
```
Prompt (6f6e583, verbatim): `Readings (BOUND, paired): "layer 1 is the block" if ORACLE_ALL beats ORACLE_23; "layer 1 is not
needed" if ORACLE_23 ≥ ORACLE_ALL − 1 pair; "layer 1 alone suffices" if ORACLE_1 ≥ ORACLE_ALL − 1 pair; "delta tolerates an
unrouted layer 1" if ORACLE_23_D ≥ ORACLE_ALL_D − 1 pair.` with `"A beats B" means p < 0.05` and validity `≥ 8/10`.
Differences: none in thresholds/arms/seeds; the docstring only operationalises "− 1 pair" as discordant pairs (B only) − (A only)
≤ 1 and fixes the recipe (Adam 1e-3 on every parameter). Report 3 restates the same four readings and validity (≥ 8) and
applies them with b/c from the records (ALL vs 23: 10/0, p = 0.00098; ALL vs 1: 1/0; ALL_D vs 23_D: 0/0); store BOUND counts
10/10, 0/10, 9/10, 10/10, 10/10 agree.

## S73 explore_g_register3_delta and S72 explore_g_register3 — seeds 700-719

| item | S73 (delta) | S72 (Hebbian) |
|---|---|---|
| readings first committed | 26c4ab1 2026-10-10T00:25:08+00:00 "Session G follow-up S73 (register3, delta) and S72 (register3, Hebbian): specifications and readings, committed before any S73/S72 run" | same commit |
| CHECKs file | checks_s73_20261010_010936.json | checks_s72_20261010_033435.json |
| store `started` / git | "2026-10-10 01:09:37" / 40e043d+dirty (s73.log header: launched at f02d69a+dirty, i.e. after 01:04:22+00:00); store first committed 35de6db 01:10:04, 0 records | "2026-10-10 03:34:37" / 373350f+dirty; store first committed 62b6418 03:35:31, 0 records |
| first commit with records | 57308ad 2026-10-10T01:15:12+00:00 — 3 (BASELINE_D, LATCH3_NUDGE_D, REG3_NUDGE_D @700) | d212293 2026-10-10T03:40:51+00:00 — 3 (REG3_NUDGE 700/701, REG3 700) |
| state at tip ed5e06b | 90/90 records | 8/60 records (in progress) |
| changes after first run | none (single commit each; `git log -p --full-history`) | none |
| report | **none on the branch** (no report_4; README's Session G section last changed 427db36, before S71) — item 4 cannot be assessed | **none** — cannot be assessed |

S73 docstring readings as first committed (26c4ab1, verbatim):
```
VALIDITY: S71's ORACLE_ALL_D (the perfect gate on delta, same seeds) binds >= 8/10 on 700-709; otherwise UNTESTED.
READINGS (ROUTED-BOUND = BOUND and F's ROUTED*@end; paired on the shared seeds; "beats" = one-sided exact McNemar p < 0.05)
  "the register carries the cue"        REG3_NUDGE_D ROUTED-BOUND >= 9/10
  "discovered (register)"               REG3_D beats BASELINE_D (700-719)
  "discovered (residual gate)"          RESGATE_ONLY_D beats BASELINE_D (700-709)
  "discovered (latch)"                  LATCH3_D beats BASELINE_D (700-709)
  "the latch holds on delta memory"     LATCH3_NUDGE_D's end z >= 0.9 at CTX and <= 0.1 at KEY and at VAL on >= 8/10 runs
  "the delta stack carries the cue"     BASELINE_D's in-run probe at the residual entering layer 3 >= 0.9 at K and at V on
                                        >= 10/20 runs (descriptive: the same count for every arm, and whether the runs whose
                                        probe reads the stream are the ones that route: a 2x2 of probe >= 0.9 x ROUTED-BOUND)
  "the gate is not needed on this task" SINGLE_D BOUND >= 8/10 (descriptive; write_order_acc printed)
```
S72 docstring readings as first committed (26c4ab1, verbatim):
```
VALIDITY: S71's ORACLE_ALL (the Hebbian perfect gate, same seeds) binds >= 8/10 on 700-709; otherwise UNTESTED.
READINGS (ROUTED-BOUND = BOUND and F's ROUTED*@end; paired on 700-719; "beats" = one-sided exact McNemar p < 0.05)
  "the register carries the cue at all three layers"  REG3_NUDGE ROUTED-BOUND >= 17/20
  "architecture insufficient"                         REG3_NUDGE ROUTED-BOUND < 12/20 (then: does the accuracy by (query
                                                      stream, block, pair index) of its unbound runs match S56's flat 0.32
                                                      plateau — every cell within 0.1 of the run's mean — or depend on the
                                                      block / pair?)
  otherwise                                           "neither"
  "discovered"                                        REG3 beats BASELINE;  "not discovered" otherwise
```
Prompt (6f6e583) S73: `"the register carries the cue" if REG3_NUDGE_D ≥ 9/10; "discovered (register)" if REG3_D beats BASELINE_D;
"discovered (residual gate)" if RESGATE_ONLY_D beats BASELINE_D; "discovered (latch)" if LATCH3_D beats BASELINE_D; "the latch
holds on delta memory" if LATCH3_NUDGE_D's z at the end is ≥ 0.9 at CTX and ≤ 0.1 at KEY and at VAL on ≥ 8/10 runs; "the delta
stack carries the cue" if BASELINE_D's probe at the residual entering layer 3 is ≥ 0.9 at K and at V on ≥ 10/20 runs
(descriptive ...); "the gate is not needed on this task" if SINGLE_D BOUND ≥ 8/10 (descriptive; print write_order_acc).`
S72: `"the register carries the cue at all three layers" if REG3_NUDGE ≥ 17/20; "architecture insufficient" if REG3_NUDGE <
12/20 (then: does the per-(stream, block, pair) accuracy pattern match S56's 0.32 plateau?); "discovered" if REG3 beats BASELINE;
"not discovered" otherwise.`
Differences: thresholds, arms (S73: BASELINE_D, REG3_D 700-719; REG3_NUDGE_D, RESGATE_ONLY_D, LATCH3_D, LATCH3_NUDGE_D, SINGLE_D
700-709; S72: BASELINE, REG3, REG3_NUDGE 700-719) and validity (≥ 8/10) identical. Docstring additions only: explicit seed range
per "beats", the latch slope schedule "1 -> 5 over updates 8001-24000", a 2x2 descriptive table, S72's "neither" category and an
operational definition of "match the 0.32 plateau" (every cell within 0.1 of the run's mean).
Timing note: 26c4ab1 was committed 19 s after S71's run started (00:24:49); S71 is a different screen and the S72/S73 child code
(explore_g_rhl_child.py) was already in 9114843. No S73/S72 process or record precedes 26c4ab1 (earliest S73 log at f02d69a+dirty,
>= 01:04:22+00:00).

## Summary table

| screen | readings commit (UTC) | started (store, no zone) | first records commit (UTC) | later edits to readings | report vs docstring |
|---|---|---|---|---|---|
| S55 | e4d8107 10-09T14:37:10 | 10-09 14:38:39 | 4da0c21 10-09T14:49:31 | none | matches (report_1, 6cd36ff) |
| S56 | a7bbee0 10-09T14:49:13 | 10-09 15:27:33 (delta 19:28:08) | 7a2fcc9 10-09T15:38:00 (delta 4e53b6e 19:36:40) | none | matches; R2 delta "discovered" by letter of "KEY failures", flagged in report |
| S71 | 9114843 10-10T00:23:50 | 10-10 00:24:49 | 191d177 10-10T00:29:07 | none | matches (report_3, 35de6db) |
| S73 | 26c4ab1 10-10T00:25:08 | 10-10 01:09:37 | 57308ad 10-10T01:15:12 | none | no report on branch |
| S72 | 26c4ab1 10-10T00:25:08 | 10-10 03:34:37 | d212293 10-10T03:40:51 | none | no report; run 8/60 at tip |

Not determinable from git: the true timezone of `started` (inferred UTC); what made the tree "+dirty"; whether S55's partial
results were looked at before a7bbee0; the "user's instruction of 9 October" that moved S55/S56 to the fixed P = 8 header;
whether the remote branch history was ever rewritten.
