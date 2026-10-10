# Registry of EXPLORATORY pre-fixed readings: sessions F, G, H (and I)

Companion to `/tmp/claude-0/registry_EXPL_FGH.tsv` (67 rows, 14 columns, tab-separated, header row).
Everything was read from `git show origin/claude/explore-{F,G,H,I}:<path>`. Nothing in the repository was modified, checked out or run.

## How the table was built

- **One row per reading.** A reading that the session numbers (R1, R2, ...) gets one row, and the `outcome_as_reported` column says which branch applied. A reading the docstring names without a number also gets a row: S54's "delta better" / "not worse" / "worse", S68/S69's named readings, S71/S73's quoted readings, S58b's per-arm "X carries", S63's per-arm "hardness helps", and S60's D1/D2 per arm.
- **Other pre-fixed rules.** Pre-fixed decision rules and descriptive readings are also rows, with `reading_id` set to `DECISION-*` or `D-*`. These are S53's write order, S54(d)'s and S70's BASELINE, S59's conditional rerun, S70's phase-1 budget, S56's read-path rule, S56's S57 trigger and S58b's arm selection.
- **Validity rules** are given inside the `rule` and `counts` columns, not as separate rows.
- **`readings_commit`** is the oldest commit returned by `git log --reverse -S'<distinctive reading text>' origin/claude/explore-<X> -- <docstring file>`. Every reading resolved, none UNKNOWN.
  - No readings file was edited after its readings commit, apart from three changes:
    - 705000d changed a CHECK in S58 only. Its own docstring says "arms and readings unchanged".
    - d9baf09 changed the S58b harness only.
    - 3e4212d changed an S63 CHECK only.
  - Every readings commit comes before its report commit.
  - Where I checked the first evidence of a run, the readings commit comes before it too: the logs record the git SHA, and the timestamps agree. Examples:
    - S53: bb906e5 at 04:14:49, runs from 04:15.
    - S68-S70: a1112da at 14:19:50, phase 1 from 14:20.
    - S55: the log ran at e4d8107.
    - S58: CHECKs ran at 69d01dc.
    - S61-S63: CHECKs ran at f630ef2 and 3e4212d.
- **`report_commit`** is the oldest commit that added the report's statement of the outcome, found with -S on the report text.

| session | screen | readings commit | report commit |
|---|---|---|---|
| F | S53 | bb906e5 | 93e0aa9 |
| F | S54, S59 | 9ddda81 | a82c62a |
| F | S68-S70 | a1112da | 9a6c2bf |
| G | S55 | e4d8107 | 6cd36ff |
| G | S56 | a7bbee0 | 427db36 (the read-path decision: 6cd36ff) |
| G | S57 | 0160269 | 427db36 |
| G | S71 | 9114843 | 35de6db |
| G | S72, S73 | 26c4ab1 | 93422cf |
| H | S58 | bbfea60 | 9200cac |
| H | S58b | 69d01dc | cc3a767 |
| H | S60 | 705000d | cc3a767 |
| H | S61, S63 | f630ef2 | 7e9197f |

- **`p`** gives the p-value the report prints for that comparison. Each one says whether it is part of the rule or only printed beside it. "not reported" means the counts imply 0/0 discordant pairs but the report prints no p.

## Per-session summary

Terms used in the summary and in this section:
- **Primary applied:** the affirmative, first-named claim held.
- **Alternative branch applied:** a named non-default branch held, such as "architecture insufficient", "absent", or "the single GDN channel is at least as good".
- **Default:** "neither", "does not apply", "not discovered", "does not help", "no change".

### Session F (claude/explore-F): S53, S54(a)-(d), S59, S68, S69, S70

- **Rows:** 21. That is 16 hypothesis readings, 3 descriptive readings (S53 write order, S54(d) BASELINE, S70 BASELINE) and 2 decision rules (S59 rerun, S70 phase-1 budget).
- **Primary applied: 5 of 16.**
  - S54(a) "delta better", and "delta not worse" alongside it.
  - S54(b) "delta not worse".
  - S54(c) "delta worse".
  - S69 "the forget gate needs the delta rule".
- **Default: 11.**
  - S53 R1 and R2 ("neither").
  - S54(a) "worse".
  - S54(b) "better" and "worse".
  - S54(c) "better" and "not worse".
  - S59 "it does not".
  - S68's two readings (the screen reads "neither").
  - S69 "the forget gate alone suffices".
- **UNTESTED: 1,** the descriptive S54(d) BASELINE. Both perfect gates bound 0/2, so there is no BASELINE.
- **Rows carrying a p-value: 12.**
  - S53 R1: 0.0625 and 0.0078.
  - S54(a), all 3 rows: Fisher 0.11.
  - S54(b), all 3 rows: 0.5.
  - S54(c), all 3 rows: 0.016.
  - S68 R-a: Fisher 6e-5.
  - S70 BASELINE: 0.031.
  - **None of F's rules uses a p-value.** They are all count or discordant-count thresholds.
  - Two of these p-values come from tests the docstring did not ask for:
    - S68's Fisher p of 6e-5. The docstring says the references are "recorded, counts only".
    - S53's McNemar p, used for the direction "the single delta channel binds more". The docstring does pre-specify printing a one-sided McNemar on paired seeds, but with no reading attached.
- **Flags** (claims in a report or README that go beyond the pre-fixed reading):
  1. S68. The README (9a6c2bf) states the outcome as **"stream-tagged keys do not scale"**. That is not a pre-registered reading. The pre-registered positive reading was "context-tagged keys scale". The outcome was "neither", because the S=4 arm also failed the other reading (2/10). The report also prints **Fisher p = 6e-5 against recorded references (other seeds) that the docstring declared "counts only"**.
  2. S59 (a82c62a). The pre-fixed outcome "it does not" is glossed as **"(the gap shrinks)"** (medians -0.46), and the report adds "a learned decay makes the *single* channel bind". Neither direction was pre-registered.
  3. S53 (93e0aa9). R1 "neither" is reported as **"the single delta channel binds *more*"**, with one-sided McNemar p = 0.0078 (header) and 0.0625 (grouped). The README (a82c62a) promotes this, with the post-hoc key diagnostic, to "**A single delta channel is not a negative control**". That is post hoc and labelled as such in report 1, but the README states it as an observation.
  4. S54(a). "delta better" applies by its count rule, against a recorded count from another seed block on another CPU. The report itself says the Fisher p of 0.11 makes it "a bound, not a test". The README table states "delta better" with no caveat.
  5. S54(c) (a82c62a): "The limit at eight streams on delta looks like the memory/optimisation, not the routing". This is post-hoc interpretation, not a reading.
  6. S69 (9a6c2bf): "**the learned decay hurt the Hebbian channel**" (final accuracy 0.17 vs 0.22, no test). This is not pre-registered.
  7. S70 (9a6c2bf). The docstring says "No other reading", yet the report states "**As in S53, the delta memory binds without the partition**", with McNemar p = 0.031 for delta vs Hebbian BOUND (5/0). The McNemar was pre-specified to be printed, but no reading was attached to it.

### Session G (claude/explore-G): S55, S56, S57, S71, S72, S73

- **Rows:** 27. That is 25 readings (including the 3 S55 presence classifications) and 2 decision rules (S56 read path, S57 trigger). Both decision rules fired: read path (2,3), and S57 TRIGGERED.
- **Primary applied: 7 of 25,** one of them contested.
  - S56 R2-delta "discovered", by the rule as coded. This is the contested one.
  - S71 "layer 1 is the block", "layer 1 alone suffices" and "delta tolerates an unrouted layer 1".
  - S72 R1 "the register carries the cue at all three layers".
  - S73 "the register carries the cue" and "the latch holds on delta memory".
- **Alternative branch applied: 7.**
  - S55: L3S_P8 "absent", ORACLE_P8 "absent", FAR_L3_P4 "partial at L3 and final".
  - S56: R1-hebb "architecture insufficient", R3 "the single GDN channel is at least as good" (0 vs 0), R4 "it does not".
  - S57: R1 "architecture insufficient".
- **Default: 11.**
  - S56: R2-hebb, R5 ("SG does not matter"), R1-delta ("neither").
  - S57: R2.
  - S71: "layer 1 is not needed".
  - S72: R2.
  - S73: the 3 "discovered" readings, "the delta stack carries the cue" and "the gate is not needed on this task".
- **Rows carrying a p-value: 7.**
  - S56: R2-hebb 1, R3 1, R2-delta 0.0625.
  - S57: R2 1.
  - S71: "layer 1 is the block" 0.00098.
  - S72: R2 0.0625.
  - S73: "discovered (register)" 0.0625.
  - **The p-value is part of the rule only in S71, S72 and S73** ("beats" = one-sided exact McNemar p < 0.05). Two of S73's "beats" rows (residual gate, latch) are 0/0, with no p printed.
- **Flags:**
  1. S56 R2-delta (427db36). It is reported "discovered (REG_D)" by the rule as coded: REG_D 6/10 vs BASELINE_D 2/10, KEY failures counted on the unbound runs only. The report then re-reads the rule post hoc: counting the 3 key-split bound runs gives "not discovered". The README carries both readings. The pre-fixed text ("with that arm's KEY failures <= 2") does not say whether bound runs count, so the coded outcome rests on a choice made in the implementation. The McNemar p of 0.0625 would not meet a p < 0.05 standard.
  2. S56/S57 (427db36). Report 2's section "What answers 'write decision or read path?'" states an "Exploratory reading": "**on Hebbian channels the unrouted layer-1 memory blocks binding ... the delta memory tolerates an unrouted layer 1**". It rests on the post-hoc ORACLE23 (0/3, not a reading per S57's docstring) and on S56 delta counts. This was not pre-registered. It was pre-registered and tested afterwards in S71 (9114843).
  3. S73 "the gate is not needed on this task" did not apply (SINGLE_D 7/10, one short of 8). Report 4 (93422cf) nonetheless states "**on delta memory the stack can learn stream-tagged keys ... and when it does, the gate does not need to route**" and "A single delta channel binds this task 7/10". Similarly, "the delta stack carries the cue" did not apply (7/20 < 10/20), yet the report asserts that the delta stack can carry it. Both tension with readings that did not apply.
  4. S72 R2 / S73 "discovered (register)" (93422cf). Both are "not discovered" (4/0, p = 0.0625 each). The report adds a **pooled 8/0 over 40 pairs**, labelled "post hoc, not a reading". The README hands REG3 back as "worth a pre-registered look ... **one pair short of the screens' threshold each time**", which frames the null as a near miss.
  5. S72 (93422cf): "**Routing layer 1 was the missing piece on Hebbian memory**" compares S72 on RandHeaderTask-lite with S56 on the P=8 header, a different task. It is interpretation, not a reading.
  6. S55 (6cd36ff): "**Training removes the cue**" (init 0.66-0.81 vs trained 0.55-0.58). The init probe was pre-specified as descriptive, but this directional claim was not a reading.
  7. S57/ORACLE23. It was added to S57's spec (0160269, 17:30 UTC) while S56 Hebbian was still running (S56 complete 19:26), "after S56's REG_NUDGE runs were seen". It is correctly labelled "POST HOC ... not a reading". It is not a flag on its own, but its result is used in flag 2.

### Session H (claude/explore-H): S58, S58b, S60, S61, S62, S63

- **Rows:** 19. That is 18 readings and 1 decision rule (the S58b arm selection, which selected RESET and GUMBEL_W).
- **S62 has no reading.** Its docstring says "Descriptive (no reading)", so it has no row. S58's GUMBEL_RW arm "has no reading" either, so it has no row.
- **Primary applied: 3 of 18.**
  - S58 R2 "the split is the reset". It holds by rule but is "uninformative" by its own pre-fixed qualifier.
  - S61 R1 "the copy alone suffices".
  - S61 R3 "the reset adds nothing".
- **Default: 15.**
  - S58: R1, R3.
  - S58b: RESET-carries, GUMBEL_W-carries, and the screen reading "does not carry".
  - S60: D1 ×3 ("no change at n = 10"), D2 ×3 ("does not delay saturation"), D3 ("does not shrink", uninformative as pre-stated).
  - S61: R2 "the noise matters".
  - S63: ANNEAL and HARD_W ("does not help").
- **Rows carrying a p-value: 14.**
  - S58: R1-R3.
  - S58b: 3 rows.
  - S60: D1 ×3.
  - S61: R1-R3.
  - S63: 2 rows.
  - **None of H's rules uses a p-value.** All are discordant-count thresholds, with exact one-sided McNemar printed both ways.
- **Flags:**
  1. **S61 R2 (the case named in the brief).** The README follow-up table (7e9197f), in the column headed "reading (pre-fixed)", says "**the noise does not matter (0 vs 0)**". The pre-registered R2 was "the noise matters" (COPY only >= 4 and COPY_NONOISE only = 0), and it did not apply. No "does not matter" reading was pre-registered. With n = 10 and 0/0 discordant pairs, failing R2 does not show equivalence. Report 3 (7e9197f) says the same thing as "**the copy needs no noise**".
  2. S61 (7e9197f): "**RESET alone is no better than no trigger (4/10 vs 3/10; 2 vs 1)**" and "the KEYMASS split's mechanism is a row copy onto an idle channel". RESET vs NONE was "Printed alongside (no reading)".
  3. S58b (cc3a767). The pre-fixed reading was "does not carry". Report 2 adds "**At eight streams the row copy is the active part of the split, not the reset**", and the README hands it back as "the split's row copy, not its Adam reset, is the effective part". This was not pre-registered at S58b; it was pre-registered next in S61 (f630ef2).
  4. S58b (cc3a767): "**Write-side Gumbel noise makes eight-stream merges worse**" and "Noise helped nowhere". These are descriptive claims beyond "does not carry".
  5. S58 GUMBEL_RW (9200cac report 1; cc3a767 README). The arm had no reading. Yet "**Raven's read-side Gumbel does not carry over**" is stated (report 1), and the README says "on the read side it collapses even four streams". The McNemar p of 5e-9 was printed descriptively.
  6. S58 (9200cac): "**The split's one gain is the reset's**". It is post hoc and rests on one seed (453). The report itself says "One seed is not evidence".
  7. S63 (7e9197f). The reading for both arms was "does not help". The report adds "**A hard write gate destroys routing at eight streams**" (HARD_W 0/10 vs 10/10, p = 0.000977) and "**Annealing is neutral**". Neither is a pre-registered reading: "destroys" is an unregistered directional claim, and "neutral" is an equivalence claim at n = 10.
  8. S60 (cc3a767). The README, in its "reading (pre-fixed)" column, gives "**saturation is never reached before binding**". The pre-fixed D2 outcome was "does not delay saturation". The README statement is a descriptive finding, and report 2 extends it to "saturation marks a committed wrong partition".

### Session I (claude/explore-I): S74/S75

- **No rows.**
  - S74 (`explore_i_pilot.py`, 6484af8) is a pilot that sets constants only and states "no readings". It has a usability rule (the user's). `explore_out/I/s74_pilot_results.json` has `"runs": {}` (CHECKs only), and the pilot log covers P0, P1 and C1 before any training run.
  - S75 (`explore_i_dense.py`, fc88fcf) has readings that are still placeholders (`__READING_SCALE__`, `__HI__`), and its commit says "nothing run".
- **Neither screen has readings and results**, so neither enters the registry.

## Out of scope but present (pre-registered, not yet reported)

These are pre-registered on the same branches, with no report yet:
- F: S77 (2210008) and S76 (7247535). There are progress commits up to 2247298, but no `report_4.md`.
- H: S78 and S79 (fdf7499; the S78 CHECK fix is 3997ae2). There are progress commits up to 5602185, but no `report_4.md`.

A later registry pass should pick these up once they report.

## Totals

| session | rows | readings (excl. decisions) | primary applied | alternative branch applied | default / not applied | UNTESTED | rows with p | rules that use p | flags |
|---|---|---|---|---|---|---|---|---|---|
| F | 21 | 19 (16 hypothesis + 3 descriptive) | 5 | 0 | 11 | 1 (S54(d)) | 12 | 0 | 7 |
| G | 27 | 25 | 7 (1 contested: S56 R2-delta) | 7 | 11 | 0 | 7 | 5 rows (S71-S73 "beats") | 7 |
| H | 19 | 18 | 3 (1 "uninformative": S58 R2) | 0 | 15 | 0 | 14 | 0 | 8 |
| I | 0 | 0 | - | - | - | - | - | - | - |

- **Descriptive rows.** The applied / default counts cover the hypothesis readings (and S55's classifications). F's remaining descriptive rows are:
  - S53's write order: "does not" keep only the last-written stream.
  - S70's BASELINE: set to HEBB DISCOVERED 0/20.
- **Every screen in scope passed its validity rule except S54(d).** Some passed only narrowly:
  - G's Hebbian oracles bound 1/2 in S55, S56 and S57.
  - F's S69 oracle bound 1/2.
