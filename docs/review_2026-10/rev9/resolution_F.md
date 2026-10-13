# Revision 9 draft, second version: resolution of the Session F items

*Referee V, second pass, 11 October 2026.*

**What was compared:**
- Second draft: `multichannel_hebbian_report_v9.tex` at `f67bc07` (02:43:57 UTC). It is cited as `tex2:NNN`.
- First draft: `9bfa85a`, cited as `tex:NNN`.
- Records: `origin/claude/explore-F` at `793cbf2` (02:41:34 UTC). Every store the draft uses is byte-identical to the one at the report commit it names.

**Evidence:** `check_F_v2.py` / `check_F_v2.md` (row ids `N1`–`N37`). Quotes are verbatim from the second draft's tex.

**Status labels:**
- **RESOLVED**: the new text is right against the records.
- **PARTLY**: part of the item is fixed and part stands.
- **NOT**: the text is unchanged, or still wrong.
- **NEW ERROR**: the revision introduced a wrong statement.

## Tally

| scope | items | RESOLVED | PARTLY | NOT | NEW ERROR |
|---|---|---|---|---|---|
| check_F.md DISAGREE rows | 3 | 2 | 1 | 0 | — |
| check_F.md PARTIAL rows | 20 | 19 | 0 | 1 (report 6, outside the draft) | — |
| revision_9_check.md F items (B4–B6, C1 ×6 places, C3 ×4 places, C4, §2, §3 ×11, §7 ×5) | 31 | 27 | 2 | 2 | — |
| new statements about F in the second draft | 37 rows | 29 MATCH, 1 not record-checkable | 7 PARTIAL | — | 2 (below) |

The draft repeats some items across the three lists. For example, B4 is also the first DISAGREE row and the first §3 item. Each list is tallied on its own.

**The four items to fix:**
1. tex2:246 / 344: "at each ... the split copied a channel that was not the merged one" (new error).
2. tex2:438 / 457: the learned forget gate "worked" against the memory horizon (new error).
3. tex2:336: S68 "supported the reading" (NOT; missed by the first pass).
4. tex2:252: the raw-key fragility mechanism (NOT).

## 1. check_F.md, DISAGREE rows

| # | first draft | item | status | second draft and evidence |
|---|---|---|---|---|
| 1 | tex:124 (= B4) | "A learned β collapses toward 'do not write'" | **RESOLVED** | tex2:124: "A learned $\beta$ collapsed toward ``do not write'' at value positions in two screens and rose above its initial $0.5$ in two others". VAL medians by layer: fell in S53 (0.14/0.12/0.16; 0.35/0.10/0.03) and S54(d) (0.04/0.03/0.03; 0.10/0.07/0.07); rose in S54(a) (0.74/0.68/0.69; 0.68/0.63/0.67) and S54(b) (0.59/0.64/0.65; 0.64/0.66/0.59). The fifth learned-β screen, S54(c), is mixed and not mentioned: C_DELTA 0.44/0.20/0.24, C_ORACLE_DELTA 0.99/0.65/0.31 (N7). tex2:344 summarises it as "(it rose in two screens)". |
| 2 | tex:246 (= B5) | six of the twelve delta splits "fired before the map had merged" | **PARTLY** | tex2:246: "Six of the twelve splits on delta channels were not on labelled targets: at each the map had already merged, and the split copied a channel that was not the merged one (its source held no stream or one in five of the six; at one the destination was not empty)". The map had merged at **6 of 6** (N12, MATCH). **New error:** the main clause "at each ... not the merged one" fails at **1314@9600**. There the map was `[6, 5, 4, 6, 10, 13, 2, 2]`, and the split copied channel 6, which held streams 0 and 3 (a merged channel), onto channel 10, which held stream 4. The parenthesis is right; the clause before it contradicts it (N13). tex2:344's "(it had; they copied the wrong channel)" holds for 5 of 6 (N26). My own first-pass note (check_F.md row 2) generalised the same way, and the draft copied it. Fix: "...and in five of the six the split copied a channel that held no stream or one; at the sixth (1314, update 9600) it copied a merged channel onto an occupied one." |
| 3 | tex:250 (= B6) | S80 KEY "1.00 at layer 3 on all ten runs" | **RESOLVED** | tex2:250: "($0.98$--$1.00$ at layer~3, median $1.00$; Hebbian $0.57$)". Per run: 1.00 ×7, 0.99 ×2, 0.98 (min 0.9797), median 1.00 (N18). |

## 2. check_F.md, PARTIAL rows

| # | first draft | item | status | second draft and evidence |
|---|---|---|---|---|
| 4 | tex:96 | "10/10 in session F" | **RESOLVED** | tex2:96: "$14/14$ in session~F". S53 ORACLE_HEBB_G 10/10 + S54(a) A_ORACLE_HEBB 2/2 + S80(a) O_HEBB_A 2/2 = 14/14 (N2) |
| 5 | tex:108 | "the cue is one to six tokens from the write it governs" | **RESOLVED** | tex2:108: "a token is one to six tokens from its block's cue, the write at a value position two to six". KEY offsets 1/3/5, VAL 2/4/6 (`explore_f_tasks.py:10-12`, N3) |
| 6 | tex:108 | budgets "(24,000, 28,800 or 43,200 updates)" | **RESOLVED** | tex2:108: "43{,}200 updates; 48{,}000 in some of session~F's screens)". F's records: {24000, 28800, 43200, 48000}; 48,000 in S53, S59 and S69 (N4) |
| 7 | tex:124 | logits "to 10⁻⁷" | **RESOLVED** | tex2:124: "to $2\times10^{-7}$". Max over the C2 β = 0 cases is 1.79×10⁻⁷ (checks file, not a run record; N5) |
| 8 | tex:124 | "reproduces a recorded run bit for bit" | **RESOLVED** | tex2:124: "bit for bit through 2400 updates" (C6, N6) |
| 9 | tex:242 | threshold of "the partition's advantage is in the number of streams" | **RESOLVED** | tex2:242: "($\geq 7/10$ at four and $\leq 3/10$ at eight)", which matches `explore_f_premise.py:23` (N8) |
| 10 | tex:246 | "sooner than the Hebbian oracle" (both L2 arms) | **RESOLVED** | tex2:246: "at $\beta = 1$ sooner than the Hebbian oracle on every seed (2400--6000; at $\beta = 0.25$ it ties the Hebbian oracle on two of the four)". β = 1 is sooner on 4/4; β = 0.25 ties on seeds 1300 and 1302 (N10) |
| 11 | tex:246 | raw keys "one seed diverged and one bound late" | **RESOLVED** | tex2:246: "with raw keys at $\beta = 1$ one seed diverged and one bound late (16{,}800), and at $\beta = 0.25$ all four bound, one of them at 7200". O_B1_RAW [—, 16800, 4800, 2400]; O_B025_RAW [7200, 1200, 2400, 1200] (N11) |
| 12 | tex:246 | S54(c) differed from S76 "in two things" | **RESOLVED** | tex2:246: "in the learned $\beta$, the noisy split with a reset, and the seeds". Seeds 310–319 against 1310–1319 (N14) |
| 13 | tex:250 | "(medians 0.57–0.67)" (population ambiguous) | **RESOLVED** | tex2:250: "(medians at key positions $0.57$--$0.67$)". S53_DELTA 0.67, S59_GDN 0.57 (N15) |
| 14 | tex:250 | Hebbian "carries nothing anywhere (0.50–0.53)" | **RESOLVED** | tex2:250: "the Hebbian channel's medians are $0.50$--$0.53$ at key positions (single runs up to $0.60$)". Per-run KEY maximum is 0.6036 (N16) |
| 15 | tex:252 | "at parity" | **RESOLVED** | tex2:252: "not shown worse ... (``keeps it'': $10/10$, $10/10$ and $9/10$ against $10/10$ each, one-sided bounds at the ceiling and not parity)" (N20) |
| 16 | tex:252 | "still needs the partition (S68)" | **RESOLVED** | tex2:252: "(S68, whose reading of that name did not apply; the pre-registered support for the partition is C3 on X)". Recomputed S68: 2/10, 0/10, "neither" (N21) |
| 17 | tex:283 | footnote d, "exact-copy split" in S80(a) | **RESOLVED** | tex2:285: "the exact-copy split at four and eight streams (none at two, where $k = S$)". S80(a) k = 2, 0 firings (N32) |
| 18 | tex:314 | "interchangeable ... at every size tried" | **RESOLVED** | tex2:318: "is not shown worse than the Hebbian one on the grouped layout at two, four and eight streams (one-sided bounds on ten paired seeds each)" (N23) |
| 19 | tex:314 | one delta channel "binds the two-stream layouts by routes that need no partition" | **RESOLVED** | tex2:318: "binds the two-stream header layout $8/10$ and the grouped one $4/10$, by what we read as routes that need no partition, and binds $2/10$ and $0/10$ at four and eight streams" (N24) |
| 20 | tex:349 | "matches ... and binds sooner" | **RESOLVED** | tex2:355: "is not shown worse than the Hebbian one at two, four and eight streams on screens and binds sooner at two and eight". Sooner at two on 10/10, at four 0/10 (all tie at 3600), at eight 8/9 (N27) |
| 21 | tex:351 | "Supported", citing S68 | **RESOLVED** | tex2:357: "\textbf{Supported on X} by the pre-registered C3 ($0/10$ against the recipe's $10/10$); session~F's S68 had the same counts ($2/10$ at four streams, $0/10$ at eight), but its own reading of that name did not apply". One wording point remains (N28, PARTIAL): C3 is at eight streams only, so "the same counts" holds for S68's 0/10 at eight, not for its 2/10 at four. Suggest "S68 had $0/10$ at eight too ($2/10$ at four)". |
| 22 | tex:353 (= C4) | "only in the solutions that bind without routing" | **RESOLVED** | tex2:359: "on the grouped two-stream layout it reaches depth at key positions in routed runs as well". DELTA_A is DISCOVERED 10/10, with L3 KEY median 1.00 (N29) |
| 23 | report_6.md:100-101 | the 1300–1359 block "also used by H's S78–S79 (explore_h_two, 1300–1329)" | **NOT** (outside the draft) | Report 6 is unchanged, and the draft does not repeat the statement. H's `two_results.json` holds 1300–1319; 1320–1329 are in `constants_results.json`. |

## 3. revision_9_check.md, F-scope items

### B4, B5, B6

| item | first draft | status | second draft |
|---|---|---|---|
| B4 | tex:124 | **RESOLVED** | tex2:124 (row 1 above) |
| B5 | tex:246 | **PARTLY** | tex2:246 and tex2:344: "had merged" is fixed; "at each ... not the merged one" fails at 1314@9600 (row 2) |
| B6 | tex:250 | **RESOLVED** | tex2:250 (row 3) |

### C1: delta "parity"

Wherever "parity", "interchangeable", "matches" or "sooner" appears in the second draft:

| place | status | second draft |
|---|---|---|
| tex:36 → tex2:36 | **RESOLVED** | "is not shown worse than the Hebbian write at two, four and eight streams ($10/10$, $10/10$ and $9/10$ against $10/10$) and binds sooner at two and eight" (N1) |
| tex:314 → tex2:318 | **RESOLVED** | row 18 |
| tex:349 → tex2:355 | **RESOLVED** | row 20 |
| tex:459 → tex2:465 | **RESOLVED** | "with its write strength fixed, the delta rule is not shown worse than the Hebbian one on screens at every size tried" (N35) |
| tex:252 → tex2:252 | **RESOLVED** | row 15 |
| tex:378 → tex2:384 | **RESOLVED** | "(it keeps them $9/10$ against $10/10$ with the write strength fixed)" (N31) |

Other occurrences, all fine:
- **"parity"**: now appears only as "not parity" (tex2:252) and in the referee's list (tex2:344).
- **"interchangeable"**: appears only as the referee review's quoted word (tex2:252, 344).
- **"matches"**: no delta use left.
- **"sooner"** about F (tex2:36, 246 ×2, 252, 355): every use is "at two and eight", or per seed with the numbers given, and all are right. Delta is sooner on 10/10 pairs at two streams, 0/10 at four (all tie at 3600), and 8/9 at eight.

### C3: "Supported" citing S68

| place | status | second draft |
|---|---|---|
| tex:351 → tex2:357 | **RESOLVED** | row 21 |
| tex:242 → tex2:242 | **RESOLVED** | The conclusion "so that the channel's failure is the missing partition" is now deferred: "which is what would make the channel's failure the missing partition rather than the delta rule, wait for L" |
| tex:252 → tex2:252 | **RESOLVED** | row 16 |
| **tex:332 → tex2:336** (missed by the first pass) | **NOT** | The text is unchanged: "session~F's premise test was on its branch 82 minutes before the commit, and supported the reading". The 82 minutes reproduce: all ten S68_SINGLE_S8 records were at `719ac78` (18:33:27 UTC on 9 October), 82.2 min before Revision 8's draft `4a3eebc` (19:55:37). But S68's pre-fixed readings both read "neither", which tex2:357 now says. Suggest "and its counts ($0/10$ at eight) bore on the claim" (N36). |

### C4

| place | status | second draft |
|---|---|---|
| tex:353 → tex2:359 | **RESOLVED** | row 22 |

### Section 2: tex:242's threshold

**RESOLVED** (row 9). The reading-against-docstring table in check_F_v2.md now has every F reading at MATCH, except S59 (see new error 2).

### Section 3: unmarked descriptions

| first draft | statement | status | second draft |
|---|---|---|---|
| tex:124 | β collapse | **RESOLVED** | It is now a description that matches the records (row 1). |
| tex:244 | "the learned decay lowered its final accuracy" | **RESOLVED** | "its final accuracy was lower with the learned decay ($0.17$ against $0.22$; a description of ten runs)". 0.17 (ROUTED), 0.17 (GDN) against 0.22 (N9) |
| tex:246 | "the plateau rule is tuned to the Hebbian trajectory" | **RESOLVED** | "That the plateau rule's target selection is tuned to the Hebbian trajectory is our reading of this." |
| tex:250 | near-orthogonal keys | **RESOLVED** | "Our reading: keys can be near-orthogonal in the 256-dimensional code without a stream direction in the residual." |
| tex:252 | "parity" | **RESOLVED** | row 15 |
| tex:252 | "needs the partition" | **RESOLVED** | row 16 |
| tex:252 | raw-key fragility | **NOT** | tex2:252 is unchanged: "Raw keys are excluded by their numerical fragility ($\beta\|x\|^2 > 2$ makes $I - \beta x x^\top$ expanding)". The algebra is right. But the records show one divergence in 8 raw-key oracle runs (O_B1_RAW\|1300), and at β = 0.25 all four bound. That the divergence came from the expansion is report 5's post hoc reading (report_5.md:62-63), and it is not marked here. |
| tex:314 | interchangeable; "routes that need no partition" | **RESOLVED** | rows 18 and 19 ("what we read as") |
| tex:349 | parity; "the stall ... came from a screen with a learned write strength" | **PARTLY** | Parity is fixed (row 20). The second sentence is unchanged at tex2:355. As a description it is true: S54(c)'s C_DELTA had `beta = 'learned'`. But in a verdict paragraph it reads as the cause, and tex2:246 calls that attribution post hoc (the split and the seeds also differed). Suggest "came from a screen with a learned write strength (and a different split and seeds; the attribution is post hoc)". |
| tex:351 | "Supported" via S68 | **RESOLVED** | row 21 |
| tex:353 | "only in the solutions that bind without routing" | **RESOLVED** | row 22 |

The tex:338 claim "each is marked where it appears" is V's item, but two of its F instances are relevant to it: the unmarked S68 "supported the reading" (tex2:336) and the unmarked raw-key mechanism (tex2:252).

### Section 7: partial matches

| first draft | item | status | second draft |
|---|---|---|---|
| tex:96 | 10/10 in session F | **RESOLVED** | row 4 |
| tex:108 | "one to six tokens from the write" | **RESOLVED** | row 5 |
| tex:124 | "to 10⁻⁷" | **RESOLVED** | row 7 |
| tex:124 | "bit for bit" covers 2400 updates | **RESOLVED** | row 8 |
| tex:283 | footnote d: S80's two-stream arm has no split | **RESOLVED** | row 17 |

## 4. New statements and new numbers about F in the second draft

Every new or changed F sentence was checked against the records (check_F_v2.md, N1–N37). These new numbers all MATCH:
- **tex2:36**: 10/10, 10/10 and 9/10 against 10/10; sooner at two and eight.
- **tex2:96**: 14/14.
- **tex2:108**: 1–6 and 2–6 tokens; 48,000 updates.
- **tex2:124**: 2×10⁻⁷; 2400 updates; "two screens / two others".
- **tex2:242**: ≤ 3/10 at eight.
- **tex2:244**: 0.17 against 0.22.
- **tex2:246**:
  - 2400–6000, with a tie on two of four;
  - raw keys at β = 0.25: 4/4 bound, one at 7200;
  - the map merged at 6 of 6;
  - the source held 0–1 streams in 5 of 6, and one destination was occupied;
  - "and the seeds".
- **tex2:250**: 0.57–0.67 at keys; 0.50–0.53 at keys, single runs up to 0.60; 0.98–1.00, median 1.00.
- **tex2:252**: keeps it, 10/10, 10/10 and 9/10; S68 2/10 and 0/10, with the reading not applying.
- **tex2:285**: footnote d.
- **tex2:314**: a minority at two, none at eight (4/10, 0/10).
- **tex2:318**: 8/10, 4/10, 2/10, 0/10.
- **tex2:355, 357 and 359**: as above.
- **tex2:384**: 9/10 against 10/10.
- **tex2:41**: S81 "running". At F's commit `793cbf2`, 2 minutes before the draft, S81 has 11 progress records (GUARD 9, GUARD_EVAL 2), its log has no "S81 done", and there is no report 7. The draft quotes no S81 number. S81's readings were committed at `451cde9` (19:28 UTC on 10 October), before its first record at `ae45031` (22:28), and are unchanged to the head (N37).

### NEW ERROR 1: tex2:246 and tex2:344

**The text:** "...and the split copied a channel that was not the merged one."

**The records:** at 1314@9600 the source was a merged channel (channel 6 held streams 0 and 3) and the destination was occupied (channel 10 held stream 4). The sentence's own parenthesis gives "five of the six". See row 2 and N13 / N26.

### NEW ERROR 2: tex2:438

This sentence was added in `f67bc07`. tex2:457 says the same and is unchanged from tex:451, which the first pass missed.

**The text:** "the one device screened against it [the memory horizon], a learned forget gate, worked on the delta write and not the Hebbian one". tex2:457 has "works on the delta write".

**The counts are right:**
- One delta channel: 9/10 and 8/10 against 3/10 (S59).
- One Hebbian channel: 0/10, 0/10 and 0/10 (S69).
- S69's reading, "the forget gate needs the delta rule", applies.

**What is wrong:** S59 *is* the screen of the horizon question ("the memory horizon on delta-rule channels", `explore_f_decay.py:3-4`). Its pre-fixed reading "the horizon knob matters" read **"it does not"**: gap_d − gap_FIXED was negative on 7/10 seeds for each learned decay (median −0.46 for both), and tex2:384 itself says the gap "shrank". S59 held L = 37 fixed (`:9`) and never varied the distance. "Worked" against the horizon is therefore an unmarked description that runs against the screen's own reading of that name. By the draft's own name rule (tex:114, 415 in the first pass), this is the S73 pattern.

**Suggested text:** "a learned forget gate, screened once at a fixed length, raised one delta channel's binding (9/10 and 8/10 against 3/10) and not a Hebbian one's; its pre-fixed reading on the horizon read 'it does not'".

### Other new PARTIALs (wording, not errors)

- **tex2:338** says "one delta channel and Gated DeltaNet's write are now run at two, four and eight streams on the grouped layout". The write run at 2, 4 and 8 has α fixed at 0.95 and β fixed at 1. Gated DeltaNet's learned forget gate ran only on the P = 8 header layout (S59) (N25).
- **tex2:357**: "the same counts" (row 21, N28).

## What I could not check

- **Session G's numbers inside F's sentences:**
  - tex2:252's 14/20 against 1/20, and S71;
  - the randomised-layout probe at tex2:250 ("in most of the delta solutions that bind without routing and in no routed one") and tex2:359;
  - tex2:263's 18 and 13 of 31, which may count the same runs twice.
- **The main-line C3 counts** quoted beside S68 (tex2:242, 357).
- **The checks file** behind 2×10⁻⁷ and "2400 updates". It is read as the file states it; it is not a run record.
- **Whether S81 is still running** in a process that has not committed. Git shows only commits; the log's last line is a "start".
- **The PDF** was not compared with the tex.
