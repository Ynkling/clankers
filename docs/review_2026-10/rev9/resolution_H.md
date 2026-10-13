# Revision 9, second draft: how the Session H findings were resolved

*Referee V, second pass, 11 October 2026.*

**What was compared.**
- First draft: `9bfa85a`, cited `tex:NNN`.
- Second draft: `f67bc07` on `origin/claude/rev9-draft`, cited `tex2:NNN`. The draft commits in between are `036466d`, `96a4122` (S84), `6c37339`, `ad3fcd5` (S85) and `f67bc07`.

**Evidence.**
- `rev9/check_H_v2.py` and `rev9/check_H_v2.md` are the second-pass run.
  - Every first-pass row keeps its recomputation.
  - The script asserts each row's old fragment on its old line, then finds it in `f67bc07` by text search.
- The records were read with `git show` on `origin/claude/explore-H`, at the commits the draft names:
  - report 3, `7e9197f`;
  - report 4, `72f0bbc`;
  - report 5, `df57087`;
  - report 6, `37c1fbf` (S84);
  - report 7, `1e09ecd` (S85, which is also the branch head).
- Each store's blob is unchanged from its report commit to the head.

**Statuses used below.**
- **RESOLVED**: the second draft's text now agrees with the records.
- **PARTLY**: some of the item is fixed and some is not.
- **NOT**: the text is unchanged.
- **NEW ERROR**: an error in new or changed text about Session H.

## Tally

| scope | items | RESOLVED | PARTLY | NOT | NEW ERROR |
|---|---|---|---|---|---|
| check_H.md, rows marked DISAGREE | 4 | 4 | 0 | 0 | — |
| check_H.md, rows marked PARTIAL | 15 | 15 | 0 | 0 | — |
| check_H.md, readings-vs-docstring rows marked PARTIAL | 5 | 5 | 0 | 0 | — |
| check_H.md, descriptive statements not marked or only partly marked | 10 | 8 | 1 | 1 | — |
| revision_9_check.md, A3, B1, B2, B3 | 4 | 4 | 0 | 0 | — |
| revision_9_check.md, section 2 (tex:195, 191, 207 ×3, 228) | 6 | 6 | 0 | 0 | — |
| revision_9_check.md, section 3 (tex:201, 203, 207 ×2, 209, 226, 312) | 7 | 6 | 0 | 1 | — |
| revision_9_check.md, section 7 (tex:115 ×2, 209/226) | 3 | 3 | 0 | 0 | — |
| new or changed text in the second draft (S84, S85, limits, conclusion) | — | — | — | — | 3 DISAGREE, 9 PARTIAL |

**Overlap between rows.** Many rows name the same sentence twice. For example:
- B1 is check_H's tex:195 DISAGREE.
- Section 3's tex:207 is check_H's descriptive-statement rows.

Each item is resolved once below and cross-referenced where it repeats.

**The one item that is NOT resolved** is S58's "it changed the routing's shape, not its count", which is still unmarked at tex2:207.

**The one item that is PARTLY resolved** is tex2:342's "Each is marked where it appears".

## 1. check_H.md rows marked DISAGREE (4)

| first pass | item | second draft | status |
|---|---|---|---|
| tex:195 (= B1) | The reset-only arm "bound no more often than the arm with no trigger" in every look. | tex2:195: "it bound no more often than the arm with no trigger in `test_split_copy` and the `window_gate` control, and more often in S61 ($4/10$ against $3/10$, 2 vs 1)". The records give S61 RESET 4/10 against NONE 3/10, RESET only 2 and NONE only 1. tex2:344 lists the error. | **RESOLVED** |
| tex:201 (= B2) | "All 83 firings ... copied a channel ..." | tex2:201: "All 83 firings across the four trigger arms were on labelled targets, a channel holding at least two streams and an empty one; the 57 in the three copying arms copied the row, the 26 resets did not." The records give 83 firings with 83/83 labelled ok. Row copies (`row_dist_after` < `row_dist_before`) number 57, and `w_g_unchanged` is true on all 26 RESET firings. | **RESOLVED** |
| tex:203 (= B3) | CAP1's single splits came "before any stream had bound". | The clause is gone. tex2:203 now says: "(per-stream accuracy is recorded only at 4800 and 9600; the seed that fired at 12{,}000 had two streams above $0.99$ at 9600)". In the records, `stats[].stream_acc` exists only at 4800 and 9600. Seed 1329 fired at 12000, and at 9600 exactly 2 of its streams are above 0.99. | **RESOLVED** |
| tex:228 (= A3) | S85 "running on 20 seeds". | The sentence is gone and tex2:228 reports S85's result. tex2:344 lists the error: "that session~H's S85 was running (committed, not started)". tex2:506 cites report 7 (`1e09ecd`) with 82 runs. tex:464 and tex:500's "S84--S85 running" are also gone, replaced by tex2:470 and tex2:506. | **RESOLVED** |

## 2. check_H.md rows marked PARTIAL (15)

| first pass | item | second draft (quoted) | status |
|---|---|---|---|
| tex:60 | "nothing replaces the split" | tex2:60: "neither device carried to eight streams replaces the split". The records: S58b SPLIT against RESET is 8 vs 0, and SPLIT against GUMBEL_W is 10 vs 0. | **RESOLVED** |
| tex:115 | "six of the 29 ... on H's branch" | tex2:115: "six of 29 recorded unbound two-stream runs of three earlier screens". This recomputes as 6 of 29 (S43 11, S60 10, S52 8, with S52's store at `bf964ec`). The draft no longer places them on H's branch. | **RESOLVED** |
| tex:115 | "three batch-18 runs at exactly η² = 0.5" | tex2:115: "labelled one batch-18 run at $\eta^2 = 0.500004$ by key STREAM-PARTIAL (two more sit at $0.49986$ and $0.499985$ and keep that label under both)". tex2:224 and tex2:394 (formerly tex:388) say the same. The records: 0.500004 is KEY under v2, while 0.49986 and 0.499985 are STREAM-PARTIAL under both classifiers. | **RESOLVED** |
| tex:195 | S61's R1 and R3 attributed to the exact copy | tex2:195: "one-pair bounds on the copy \emph{with} noise against the full split (0 vs 0), and both held; the exact copy (no noise, no reset) enters only the reading ``the noise matters'' ($\geq 4$ vs 0), which did not apply (0 vs 0)". This matches the docstring at `f630ef2` l.19 and 28-30. | **RESOLVED** |
| tex:203 | "What a replication must state is therefore ..." | tex2:203: "Our recommendation from this, not a reading: a replication should state the cap and the check schedule (the fast-check arm moved the interval, the first check and the gap together, so the three are not separated)". The sentence is now marked and the confound is stated. | **RESOLVED** |
| tex:207 | S58: "no reading could apply" | tex2:207: "two of its three readings did not apply and the third, ``the split is the reset'', held by its rule (0 vs 0) and was printed uninformative by its own qualifier". This recomputes: R1 and R3 do not apply, and R2 is 0 vs 0. | **RESOLVED** |
| tex:207 | S58b: "the two best-scoring arms" | tex2:207: "the two best-scoring devices ... (it ranked the four devices only; the no-device arm's $38/40$ outscored write-gate noise's $37/40$)" | **RESOLVED** |
| tex:207 | S62: "32--36 of the bound runs" | tex2:207: "In the other arms with a full set of runs ... (session H's S62, descriptive: in 32--37 of the 38--40 bound runs per arm at four streams, and $12/12$ at eight)". This recomputes as 32, 37, 37 and 37 of 38, 40, 40 and 40 for GUMBEL_W, NONE, RESET and SPLIT. | **RESOLVED** |
| tex:207 | S60: "a stable-max softmax changed nothing" | tex2:207: "a stable-max softmax read ``no change at $n = 10$'' by its rule ($7/10$ against $8/10$)". The records give 0 vs 1. | **RESOLVED** |
| tex:207 | S63 HARD_W: "never forms" | tex2:207: "that the writes' stream labelling therefore never forms is the session's description, and four of the ten runs did form a partial one (MERGED needs a margin of $0.25$)". The four MERGED runs have margins of 0.41, 0.60, 0.73 and 0.76. | **RESOLVED** |
| tex:209 | S78: "re-forms within 2400 updates" | tex2:209: "on the eight firings of the seeds that ended in a key split, the same split is back by the next check (two of the eight are read from the end of the run); one seed had no key split when the copy came and discovered". Remaining remark: "ended in a key split" uses v1's count, three seeds. Under v2, 1317 also ends KEY; tex2:226 treats it separately as the half-key pattern. | **RESOLVED** |
| tex:226 | "after all eight firings" with no next check for two | tex2:226: "under the classifier of record (six read at the next check, 2400 updates later; two, fired at the last check, read from the end of the run)". The records: 6 firings have a next check and all are 1.00/0.00. The 2 fired at 21600 read 1.00/0.00 at the end. | **RESOLVED** |
| tex:228 | v1/v2 direction inverted | tex2:228: "Four of S78's seventeen unbound runs are STREAM-PARTIAL under the classifier of record and KEY under \texttt{fail\_class\_v2}". The records give 4 of 17, all STREAM-PARTIAL → KEY. | **RESOLVED** |
| tex:347 | S61 "shows" that the reset adds nothing | tex2:353: "session~H's S61 holds it as a one-pair bound on ten seeds (the copy with noise $9/10$, the exact copy $9/10$, the full split $9/10$)" | **RESOLVED** |
| tex:409 | "two to five runs per arm elsewhere" | tex2:415: "two to five of the 37--39 bound-routed runs in each of S58's other full arms". The records: SPLIT 3/39, RESET 3/39, GUMBEL_W 5/37 and NONE 2/38. | **RESOLVED** |

## 3. check_H.md readings-vs-docstring rows marked PARTIAL (5)

| screen | second draft | status |
|---|---|---|
| S58 | tex2:207, quoted in §2 above | **RESOLVED** |
| S58b | tex2:207 now states the rule that ranks devices only. The reading's own name ("does not carry") is still not quoted; the outcomes are, as 8 vs 0 (tex2:189) and 10 vs 0 (tex2:207). | **RESOLVED** |
| S60 | tex2:207: D1's label is quoted together with its n | **RESOLVED** |
| S61 | tex2:195: R1 and R3 are attributed to the copy with noise | **RESOLVED** |
| S63 | tex2:207: both arms "read ``does not help''". The draft's words "neutral" and "destroyed routing" are gone. The arm is still called "slope annealing"; it is a temperature anneal from 1 to 0.25. | **RESOLVED** |

## 4. Descriptive statements: check_H.md's descriptive-statement table, plus revision_9_check.md section 3

| first pass | statement | second draft | status |
|---|---|---|---|
| tex:201 | "two identical rows ... separate ... as well as ..." stated as an equality | tex2:201: "... is what 0 vs 0 on ten seeds is consistent with, not a shown equality" | **RESOLVED** |
| tex:191 | copy only against reset only, 5 vs 0, p = 0.031, not pre-fixed | tex2:191: "5 vs 0, $p = 0.031$ (ours, not pre-fixed)" | **RESOLVED** |
| tex:203 | the marked descriptive part contained the false "before any stream had bound" | The clause is removed (see B3). | **RESOLVED** |
| tex:203 | the replication recommendation | tex2:203: "Our recommendation from this, not a reading" | **RESOLVED** |
| tex:207 | "the stream labelling of the writes never forms" | tex2:207: "is the session's description, and four of the ten runs did form a partial one" | **RESOLVED** |
| tex:207 | S58 SWITCH: "it changed the routing's shape, not its count" | tex2:207 is unchanged and unmarked: "it changed the routing's shape, not its count, and was not carried to eight streams". The count half is 36/40 against NONE's 38/40 (1 vs 3 discordant). The shape half is 34/36. | **NOT** |
| tex:209 | "The copy works at eight streams because the idle channel ... is empty" | tex2:209: "... is our reading of these screens". The next sentence, a recommendation, is unmarked: "the trigger should not fire when its target already carries key mass". | **RESOLVED** |
| tex:226 | the near-tie sentence | tex2:226: "Our reading: at eight streams the copy creates a near-tie ..." The eight-stream half is still S24's measurement, not an H record. | **RESOLVED** |
| tex:312 | "because there is no idle channel" | tex2:316: "because, on our reading, there is no idle channel" | **RESOLVED** |
| tex:338 | "Each is marked where it appears" | tex2:342 keeps the claim and adds "those are now marked". "Noise helps a little" is gone, and the S79 description is marked. Still unmarked: S58's "shape, not count" (tex2:207), and the new S84 interpretations at tex2:203 ("so the probe's noise decided whether the first check fired"; "the base kept enough splits for its merges"). | **PARTLY** |

## 5. revision_9_check.md items in Session H's scope

| item | where in the second draft | status |
|---|---|---|
| A3 (tex:228, 464, 500): "S85 running" | tex2:228 now reports S85's result; tex2:470 and tex2:506 updated; tex2:344 lists the error (see §1) | **RESOLVED** |
| B1 (tex:195) | tex2:195 (see §1) | **RESOLVED** |
| B2 (tex:201) | tex2:201 (see §1) | **RESOLVED** |
| B3 (tex:203) | tex2:203 (see §1) | **RESOLVED** |
| §2 tex:195: R1/R3 attributed to the exact copy | tex2:195 (see §2) | **RESOLVED** |
| §2 tex:191: the 5 vs 0 comparison was neither pre-fixed nor printed | tex2:191: "(ours, not pre-fixed)" | **RESOLVED** |
| §2 tex:207: "no reading could apply" | tex2:207 (see §2) | **RESOLVED** |
| §2 tex:207: "two best-scoring arms" | tex2:207 (see §2) | **RESOLVED** |
| §2 tex:207: "changed nothing" | tex2:207 (see §2) | **RESOLVED** |
| §2 tex:228: classifier wording inverted | tex2:228 (see §2) | **RESOLVED** |
| §3 tex:201 | tex2:201 (see §4) | **RESOLVED** |
| §3 tex:203 | tex2:203 (see §4) | **RESOLVED** |
| §3 tex:207: "never forms" | tex2:207 (see §4) | **RESOLVED** |
| §3 tex:207: "shape, not count" | tex2:207, unchanged (see §4) | **NOT** |
| §3 tex:209 | tex2:209 (see §4) | **RESOLVED** |
| §3 tex:226 | tex2:226 (see §4) | **RESOLVED** |
| §3 tex:312 | tex2:316 (see §4) | **RESOLVED** |
| §7 tex:115: "six of 29 ... on H's branch" | tex2:115 (see §2) | **RESOLVED** |
| §7 tex:115 (and tex:388): "three ... at exactly 0.5" | tex2:115, tex2:224 and tex2:394 (see §2) | **RESOLVED** |
| §7 tex:209, 226: the two firings read at the end of the run | tex2:209, tex2:226 (see §2) | **RESOLVED** |

## 6. New errors in the second draft (new or changed text about Session H)

From check_H_v2.md. No first-pass row became worse. Every first-pass row still in the text is now MATCH.

### Disagreements: S85, tex2:228, the "Descriptively" sentence

The draft copies these numbers from report 7 ("What the timing shows"), and the records do not give them.

1. **"under the key term $\eta^2$ by key falls to $0.00$--$0.12$ by update 25 and stays below $0.07$ to 2400"**
   - At update 25 the KEYHINGE runs read 0.001–0.195; two runs are above 0.12 (2326 at 0.168, 2327 at 0.195).
   - Over updates 50–2400, seven of the twenty runs reach at least 0.07:
     - 2310: 0.137
     - 2313: 0.087
     - 2316: 0.131
     - 2321: 0.102
     - 2325: 0.088
     - 2327: 0.165
     - 2328: 0.144
   - The substance holds: the value stays far below 0.5 throughout the window.
   - A side note, not in the draft: the records' `hinge.fired` field reads 1–11 per run, where report 7 says the term "fired on 1–37 batches". I did not trace the field's unit.
2. **"a split can start after the window ($0.21$--$0.64$ at 3600 on some seeds)"**
   - At 3600 the runs at or above 0.2 are:
     - 2313: 0.21
     - 2317: 0.64
     - 2318: 0.30
     - 2319: 0.35
     - 2321: 0.88
     - 2323: 0.25
     - 2325: 0.22
   - The range is therefore 0.21–**0.88**. Report 7 wrote "2317: 0.64; 2321 > 0.5", and the draft kept only the 0.64.
   - "Every such run recovers" holds: KEYHINGE is 20/20.
3. **"organised by key on six of the twenty seeds ($\eta^2$ up to $0.84$ at update 0)"**
   - The value at update 0 is the same in all four arms. It exceeds 0.5 on **five** seeds:
     - 2313: 0.70
     - 2318: 0.64
     - 2320: 0.84
     - 2326: 0.56
     - 2328: 0.76
   - The sixth highest is 2319 at 0.476.
   - Report 7 also says "6 seeds", but its own per-seed table shows five update-0 onsets in every arm.
   - "Up to 0.84" holds. "Shed by update 600" holds for REF's two discovering seeds (2313 and 2320 are both at 0.00 by 600).

### Partial matches

| tex2 | text | record |
|---|---|---|
| 228 | "the slow gate discovers $14/20$ (0 vs 1) with the same four key failures" | The count is the same, but the seeds differ. Under v2, REF's KEY seeds are 2310, 2318, 2323 and 2328; SLOWGATE's are 2310, 2321, 2323 and 2328. 2318 ends STREAM-PARTIAL under the slow gate, and SLOWGATE's 2323 is KEY only under v2, at η² = 0.5000005. The counts and both readings recompute: 0 vs 1, "not shown"; 4 ≥ 4 − 1, "only moves". |
| 228 | "from the gate's $\eta^2$ by key at every 25 updates" | The dense cells are every 25 updates to 1200, every 100 to 2400, then every 1200 (explore_h_two_b.py l.28-30). |
| 228 and 384 | "the baseline's real splits form at updates 25--125 and the slow gate moves them to 275--575"; "it moves it from update 25--125 to 275--575 and prevents nothing" | The ranges recompute: REF's first update above 0.5 after update 0 runs from 25 to 125 over 7 seeds, and SLOWGATE's from 275 to 575 over 3. But only 2310 (125 → 275) and 2323 (125 → 575) carry an onset in both arms. REF's other five onsets (25–50, on seeds both arms discovered) do not occur under the slow gate. One new onset forms on 2321 (275), a seed REF discovered. Two of REF's four KEY failures (2318, 2328) are above 0.5 from initialisation. "Moves them" treats the ranges as the same splits. |
| 60 | "the flat-start guard aims the splits without changing the count" | The pre-fixed "the guard aims the split" applies: labelled ok 17/17, 0 vs 0. But the criterion cannot fail at the flat start: BASE's 24 splits, 7 of them at probe accuracy 0.06–0.16, were all labelled ok. Report 6 says so itself. What the guard changed is the accuracy at the split, 0.50–0.91 (tex2:203), which "aims" does not convey. |
| 436 | "the hinge's key term, which prevents the key split at two streams" | No "on a screen", unlike tex2:36, 316 and 440. The result is S85: 5 vs 0, one-sided p = 0.031, 20 seeds, exploratory. |
| 440 | "the one repair device that works at eight streams cannot work here" | Unchanged from tex:434 and missed in the first pass. It states as fact what tex2:209 marks as "our reading". S78 measured "neither device": 1 vs 0 and 2 vs 0. |
| 465 and 228 | "which also prevents the two-stream key split that nothing repairs"; "prevents the split that nothing repairs" | The conclusion has no screen marker. "Nothing repairs" generalises S78's two forms of one repair device, on 20 seeds, to every repair. |

**The S84 readings row** (readings vs docstring) is PARTIAL for the same "aims" reason. In addition, the draft states neither the arm EVAL_GUARD_INT1200 (9/10, 0 vs 0, median transition 15600) nor the reading "costs binding" (no arm).

### What matches in the new material

**S84 (tex2:203).** Every count, discordant pair and reading recomputes:
- seeds 2300–2309, and the oracle 2/2 at 3600;
- GUARD (guard 0.5 on the probe): 9/10 against BASE 9/10, 0 vs 0;
- GUARD's 17 splits at probe accuracy 0.500–0.906; BASE's 7 of 24 at 0.062–0.156;
- seed 2306 fired no split in all three guarded arms, with a maximum read of 0.375, 0.375 and 0.378, and ended MERGED (3 share);
- GUARD_INT1200: 9/10, 0 vs 0. The median transition over its bound-routed runs is 15600 against 16800, 1200 earlier, so "faster" does not apply. S79's INT1200 was 7/10 on seeds 1320–1329;
- EVAL reads the 2048-query evaluation (every reading is a multiple of 1/2048): 10/10, 1 vs 0 on seed 2306;
- on seed 2306 the probe read 0.000 then 0.047 (no firing at 4800), and EVAL read 0.0664 then 0.0679 and fired at 4800;
- the reading "the evaluation is the better probe" applies by its rule, (BASE only − EVAL only) ≤ 1.

**S85 (tex2:228).** The headline numbers recompute:
- 20/20 against 15/20, 5 vs 0, p = 0.0312, KEYHINGE with no KEY failure against REF's four: "prevention works";
- 19/20, 5 vs 1, p = 0.109; the failure is 2317, KEY, with η² by key ≤ 0.20 to update 2400, 0.30 at 3600 and 0.77 at 4800;
- median transitions 4800 against 2400;
- one v1/v2 difference, SLOWGATE|2323;
- the oracle 2/2 at 1200;
- the design (gate group at 1e-4 for updates 1–1200; the key term on updates 1–2400);
- the proposed test (report 7's take-away).

**The S85 descriptive sentences are marked.** The paragraph opens them with "Descriptively". The interpretation of 2317 ("the slow gate held the gate back until the hinge's window had passed") sits in the readings sentence and is unmarked.

**Readings and timing.** S84's and S85's readings were committed at `834581a`, 2026-10-10T19:05:31Z, and are unchanged at the head. The first records came later: S84 at `66ec3c6`, 19:50:33Z; S85 at `44def42`, 2026-10-11T01:51:15Z.

**Provenance and other rows (tex2:506, 36, 60, 316, 470).**
- tex2:506 matches: reports 6 and 7 at `37c1fbf` and `1e09ecd`; runs 52 and 82; branch commit `1e09ecd` = the head.
- The abstract (tex2:36: "on a screen ($20/20$ against $15/20$)"), the S85 half of the index (tex2:60), tex2:316 and the next steps (tex2:470) match.
- Table 5 (`tab:far`, tex2:289–312) carries no Session H number. Table 4's $39/40^{e}$ (tex2:281) is S58 SPLIT, 39/40.

## What could not be checked

- That seeds 2300–2329 are fresh. This is taken from the scan recorded at explore_h_guard.py l.9-12; the referee did not rescan every branch.
- "The stream routing having formed first" (tex2:228). It would need η² by stream over the dense cells, which was not computed.
- Records that are not Session H's:
  - Session F's S81 (tex2:203);
  - session G's S82 (the comparison at tex2:228, Table 5);
  - S24's near-tie at eight streams (tex2:226);
  - the `window_gate` and `split_copy` cells of tex2:195.
- Whether H's S85 CHECK, run on a dirty tree (`96e1e90+dirty`), used exactly the committed code. The docstring and code files are unchanged since `834581a`, but the "dirty" state cannot be resolved from git.
