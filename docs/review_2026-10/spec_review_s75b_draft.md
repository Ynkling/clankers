# Spec review: explore_out/I/s75b_spec_draft.md (claude/explore-I at `114cd08`)

*Referee V, 12 October 2026. No S75b code exists on any branch, and nothing has run. Line numbers are from the draft at `114cd08` unless another commit is named; the draft is unchanged at the branch tip `559b942`. Numbers come from `spec_checks_s75b.py`, whose output is in `spec_checks_s75b_out.txt` (sections A–H). The script reads git objects and makes one read-only GitHub API call for the branch's push times.*

**Verdict.**
- **What git shows matches the claim, word for word.**
  - The design commit `1e31409` has one parent, `5bc6c2e`.
  - That tree holds 12 S75 records: ORACLE 830–831 and SINGLE 830–839.
  - `b649751`, which holds WIDE_SPLIT and REG3, is not an ancestor of `1e31409`.
  - No number in any of the three versions appears only in the learned-arm summaries.
  - On S75's learned-arm rows the guard changes nothing.
  - Configuration B was already named in the pre-run note (`5bc6c2e`), 13 h before any learned-arm record reached git.
- **"Blind" itself is the author's word, not git's.**
  - `b649751` was on the remote at 13:53:11, 23 min before `1e31409`. By the author's own account it was in the local object store from 14:10.
  - Both revisions (`ee97443`, `114cd08`) were made with the learned-arm records in their tree.
  - The same session committed the learned-arm write-up 130 s after `114cd08`.
- **The pilot rules are mechanical and fixed before any code exists.** Four outcomes are not covered, and the stop-rate figures rest on a σ estimated from one pair.
- **The guard's precedence is written and matches `make_split`.** Two of the cases where two rules fire need a sentence each, the guard is described as "the same test ROUTED uses" though it uses only half of that test, and l.119 misstates the check schedule.
- **No constant coincides with a learned-arm value.** Two design changes postdate `b649751`: the looser proceed rule and the unconditional WIDE_SLOW. Neither is a function of a learned-arm quantity, but neither is blind by git. Label them that way.

## 1. Provenance

**What git shows** (section A of the output).

| commit | parents | date (UTC) | S75 records in its tree | note |
|---|---|---|---|---|
| `5bc6c2e` (pre-run note) | `86652b1` | 00:46:28; pushed 00:46:30 | 12: ORACLE 830, 831; SINGLE 830–839 | names B: NB = 32, M = 352, L = 1536, bound 0.964, ORACLE ≈ 0.92 (note l.116–118) |
| `b649751` (learned arms) | `5bc6c2e` | 13:53:09, root@Docker.home; **pushed 13:53:11** | 32: + WIDE_SPLIT 830–839, REG3 830–839; adds `s75_S4_summary.md` | only on `origin/claude/explore-I` |
| `1e31409` (design) | **`5bc6c2e` only** | 14:16:14 (author = committer) | 12, with the same blob ids as `5bc6c2e` (results `7589867874`, summary `267de89e4c`) | the diff from its parent adds only `explore_gpu/s75_writeup.py` and the draft |
| `142f47e` (merge) | `1e31409`, `b649751` | 14:16:14; pushed 14:16:16 (`b649751` → `142f47e`) | 32 | `1e31409` was never a branch tip on the remote; it first arrived as this merge's parent |
| `ee97443` (cost) | `142f47e` | 15:20:05 | 32 | `b649751` is an ancestor |
| `114cd08` (revision) | `15c5e4f` | 16:04:21 | 32 | `b649751` is an ancestor |
| `03bffdf` (S75 write-up) | `114cd08` | 16:06:31 (130 s later) | 32 | the same Claude-Session (`01Acbhe5`) as `1e31409` and `114cd08` |
| `4c3984a` (WIDE_SLOW 830–838) | `03bffdf` | 19:12:59 | 41 | a descendant of every draft version |

`b649751` is an ancestor of `142f47e`, `ee97443`, `15c5e4f`, `114cd08` and everything after; it is not an ancestor of `1e31409`. No WIDE_SLOW record is in any tree before `4c3984a`.

**What git cannot show.**
- **When `1e31409`'s text was written.** Commit dates are set by the committing machine. The SSH signature covers the dates, but the signer chooses them. The only server-side time is the push at 14:16:16, which bounds the commit from above, not from below.
- **The 14:10 fetch** (l.15–16). It would live in the author's local reflog, which is never pushed. It is consistent with the server's push of `b649751` at 13:53:11.
- **What the author read.** From 14:10, `b649751`'s objects were readable locally (`git show origin/claude/explore-I:…`) for 6 minutes before `1e31409` was committed. The user's machine held WIDE_SPLIT 830–832 from about 00:44 (`03bffdf` report_2.md l.29–31) and all 20 learned runs by 13:53. Git does not record whether any of that reached the session, for example as pasted terminal output.
- **Verdict on the wording.**
  - "Parent 5bc6c2e holds only ORACLE/SINGLE records" is exactly true. It appears in `1e31409`'s message, at `1e31409` l.12–14 and at l.13–14.
  - "Written blind" is a statement. Git can show that the commit was not built on `b649751`; it cannot show what its author had seen.

**Points.**
1. **`1e31409` l.12–14 said more than git shows.** It read "This draft was written and committed before S75's WIDE_SPLIT and REG3 records were read" and did not mention that the records had already been fetched. `114cd08` l.15–17 corrects this to "On the author's statement…", which is right. But the heading at l.12, "Provenance: blind to S75's learned arms", and the table cell at l.170, "(blind to the learned arms)", still state blindness as a fact.
2. **The revisions were made with the learned-arm records in their tree.** l.18–19 says they were made "without opening S75's learned-arm files", but `ee97443` and `114cd08` both hold `b649751`'s 32 records and `s75_S4_summary.md` (blob `a357ca118a`).
   - The same session committed `03bffdf` 130 s after `114cd08`. It contains report_2.md, whose l.3–8 and l.40–66 quote WIDE_SPLIT 0/10, REG3 0/10, η² 0.00 and the 30 fires on merged maps, and the 136-line `s75_S4_descriptions.md`. An 80-line report cannot have been written from scratch in 130 s, so the write-up was being prepared alongside the revision. Git cannot order the two.
   - `114cd08`'s message, "(no S75 learned-arm file opened)", does not say whose statement this is: the blind reviewer's or the author's.
   - report_2.md l.78–80 extends "nothing in it depends on the learned arms" from `1e31409` to the whole draft.
3. **The spec does not cite its strongest evidence.** B was fixed in `5bc6c2e` (pre-run note l.116–118), 13 h 7 min before `b649751` was pushed, but l.164 cites only "the ideal learners at B". (`03bffdf` report_2 l.27–31 corrects the note's timing: WIDE_SPLIT 830–832 had run on the user's machine before `5bc6c2e`. Git holds no learned-arm record before `b649751`.)
4. **Mechanical check.** The learned-arm summaries contain 13 numbers that appear in none of the allowed inputs: 0.14, 0.15, 0.19, 0.21, 0.529, 0.533, 0.536, 0.539, 0.588, 0.80, 1.37, 16800 and 19200. None of them appears in `1e31409`, `ee97443` or `114cd08` (section B). This checks literal numbers only; a learned-arm number that was rounded or transformed would not be caught.

## 2. The pilot's proceed and stop rules

- **Mechanical.** Yes. Both rules read `end.set` at 24,000 updates on 1024 sequences, unrounded (l.84–87). The driver applies them and refuses stage 2 otherwise (l.105–107).
- **Fixed before the pilot.** Yes: the Order puts the pilot at step 3 (l.244), and no S75b code exists on any branch.
- **Changed once.** The rule changed after `b649751` was merged: from "both ≥ 0.912" (`1e31409` l.59) to "mean ≥ 0.912 and each ≥ 0.9" (l.85–87). The change is disclosed at l.24–26 and l.89–93.

**The numbers behind the rule** (sections E and F).
- **0.912 is rounded.** The unrounded spread is 0.89154 − 0.88010 = 0.01144, so "0.9 + spread" is 0.9114. The draft's 0.012 rounds each value first, which adds 0.0006. That is harmless as a fixed constant, but say "0.912, rounded up from 0.9114".
- **σ = 0.008 comes from one pair.** It is spread/√2 = 0.00809, with 1 df: its 95% interval is 0.0036–0.258.
- **l.90–93 reproduces at σ = 0.008.**

  | prediction | old rule P(stop) | new rule P(stop) | P(stage-2 VALID) |
  |---|---|---|---|
  | per-bin, μ 0.922 | 0.200 | 0.040 | 0.994 |
  | fixed reach, μ 0.914 | 0.642 | 0.366 | 0.921 |
- **At σ = 0.012 the figures move.** The new rule's stop rates become 0.139 and 0.433, and VALID 0.934 and 0.771.
- **The "about 20%" assumes a common mean.** If the per-bin prediction is taken as two seed-specific means (0.928 and 0.916), the old rule stops 0.324 of the time, not 0.20.
- **The rule protects against proceeding on a weak configuration.**

  | true μ (σ 0.008) | P(proceed) | P(VALID) |
  |---|---|---|
  | 0.905 | 0.106 | 0.539 |
  | 0.910 | 0.358 | 0.800 |

**Outcomes not covered.**
1. **A failed pilot run.** A non-finite loss gives `ok: False` (`explore_i_common.py` l.585–586), and a run can also fail out of memory. l.106 refuses stage 2 "unless both records exist and pass", but the draft does not say which of these follows:
   - S75b stops, and if so with which report sentence;
   - the same seed is retried (S75's driver retries a failed batch once, `s75_s4.py` l.47–48);
   - a fresh seed is used.
2. **The pilot on another card.** l.149 fixes the pilot on the 1660, and the fallback at l.151 covers only stage 2. The round-5 prompt lets the pilot run "on the 3070 if that is free first": `docs/reading/parallel_sessions_prompts.md` l.1091–1092, on `origin/claude/bdh-growth-hebbian-inference-w90069` at `0754d61`. The draft and the prompt disagree.
3. **A code-SHA change between the stages.** l.106 reads the pilot "under its own code SHA", so a driver fix after stage 1 orphans the pilot. The draft does not say whether to rerun 840–841 or use fresh seeds.
4. **A stage-2 ORACLE run missing or failed.** VALID (l.141) needs both 850 and 851, and l.133 ("every arm runs all its seeds") has no clause for failure. Neither says whether this is UNTESTED or a rerun.
5. **The label "not usable" uses a different statistic from S74's.** l.96–97 says "S = 4 is not usable at B" only if a pilot run ended below 0.9, "S74's usability bar". S74's count as S75 prints it uses ORACLE's **mean** (`s75_s4.py` l.43 and l.270–271) and has a SINGLE half that the pilot does not run. A pilot of 0.915 and 0.898 has a mean of 0.9065, which passes S74's ORACLE half, yet the draft would label it "not usable".
6. **Cost.** No rule applies if the L = 1536 benchmark (Order step 2, l.242) shows that REG3 fits neither card, or that times exceed the estimates by some stated factor.

**Pilot quantities used in stage 2: only SPLIT_THR** (l.102).
- The pre-run note allows it (`5bc6c2e` note l.112–114).
- If the pilot proceeds, SPLIT_THR ≥ 0.87; S75 used 0.88.
- Proceeding selects high pilot values, so SPLIT_THR (the pair's minimum − 0.03) is biased upward.
- The hazard S75 declared (note l.66–81: ORACLE-quality runs are eligible to fire mid-training) can recur. Nothing in the draft replays the trigger on the pilot's curves.
- R, the register mode and the stats workers are fixed by the benchmark before the pilot (l.152–157, l.241–243), and the pilot is never pooled (l.81–82, l.136). Both are right.

## 3. The guard's precedence

- **Written, and matches the code.** The precedence is at l.121–127. It inserts one rule into `make_split` (`explore_i_common.py` l.521–554 at `114cd08`), between its `eligible and capped` branch (l.543) and its `elif eligible` branch (l.545).
- **The installation route checks out.** `run.py` l.514 looks `ic.make_split` up at run time, as l.225 says.
- **The guard is inert on S75's data** (section G). On S75's 90 WIDE_SPLIT trigger rows, the guarded order changes 0 actions: 21 not eligible, 19 blocked by cap, 20 by gap, 30 fires. No `map_before` was one-to-one. So the guard cannot have been fitted to the learned arms, and S75 does not test it.

**Where two rules fire, or the text is off.**
1. **Cap and gap together.** This happened on 10 of S75's 39 blocked rows. `make_split` labels such a row "cap" (l.544), but l.123 ("cap or gap, as in S75") leaves this implicit. State it.
2. **Eligible, capped or gapped, and one-to-one.** Rule 2 wins, and the row is labelled cap or gap. Its one-to-one state survives only in the row's field (l.129). The printed count "blocked by … one-to-one" (l.145) then undercounts eligible checks on one-to-one maps. Print both counts, and add this case to the guard CHECK (l.230–233).
3. **"The same test ROUTED uses"** (l.126) is half of that test.
   - The guard reads `one_to_one` alone: the per-source argmax map (`routing_layer` l.383–385).
   - It reads it on the 64-sequence trigger probe, at the first gate layer only (`make_split` l.533, l.537).
   - ROUTED (l.139) needs η² ≥ 0.9 **and** a one-to-one map at every gate layer, at the end.
   - So a source-blind gate whose per-source argmaxes happen to differ would be shielded from a split.
   - State whether this is intended. Any change now (for example, adding η²) is made after the learned arms are known: this review has read them (η² ≤ 0.0004; one-to-one maps 0/200 WIDE_SPLIT, 0/600 REG3 and 0/180 WIDE_SLOW over evaluations × layers). It must be recorded as not blind.
4. **A merged map whose `c0` is live.** Rule 4 fires on any map that is not one-to-one, even when `c0` (the channel with the least mass) is in the map, which `make_split` marks `on_target` False. That overwrites a live channel: the hazard given at l.129–130. In S75, 30 of 30 fires were on target, so there are no data. State it.
5. **The check schedule** (l.119, l.169). The draft says checks run "every 2400 updates from 4800". In fact `make_split` also checks at 2400 (`SPLIT_REF`, l.104): that check sets `prev` and is never eligible, and eligibility starts at 4800 (`SPLIT_FIRST`). S75's rows start at 2400. The transcription must keep the 2400 check.

## 4. The constants and where they come from

"Learned arms" are `b649751`'s records (WIDE_SLOW at `4c3984a`). Verdicts: **D** means derived from ORACLE or SINGLE data, ideal learners, K9 or earlier sessions; **U** means unsourced; **S** means coincides suspiciously with a learned-arm value. No constant is S.

| constant (line) | value | stated source | closest learned-arm value | verdict |
|---|---|---|---|---|
| S, V, block lengths (l.35, 163) | 4, 4, 6–14 | S75 registered | – | D |
| NB, M, L (l.35, 164) | 32, 352, 1536 | ideal learners and the ORACLE transfer | – | D; already in `5bc6c2e` note l.117 |
| model incl. decay 0.95 (l.36–37, 165) | as S75 | S75 | – | D |
| B, lr, eval, updates (l.37–39, 171) | 32, 1e-3, 1200/1024, 24,000 | S75 (`explore_i_common` l.97–100) | – | D |
| Bayes bound at B (l.45, 53) | 0.963 (was 0.964) | ideal learner, resampled | – | D; not reproducible here (needs a run) |
| window W, fixed-reach prediction (l.54, 62–65) | 220/250; 0.909/0.920/0.914 | fit to S75 ORACLE bins 2+ | – | D; new in `114cd08`; no committed script; 48 sequences |
| per-bin prediction (l.59–60) | 0.928/0.916/0.922 | ORACLE per-bin | – | D; bin 1 0.7192/0.7121 checks out |
| gap-share prediction (l.61) | 0.919 (was 0.918) | ORACLE, ideal learners | – | D; I get 0.9178 at bound 0.963 and 0.9187 at 0.964 (the two versions look swapped, ±0.001) |
| SINGLE prediction (l.69–70) | 0.56 | SINGLE median 0.538 | – | D (0.559) |
| pilot seeds, screen seeds (l.80, 109, 175) | 840–841; 850–859 | fresh | – | D; no record on any of 12 branches (section H) |
| proceed rule (l.85–87, 167) | mean ≥ 0.912, each ≥ 0.9 | ORACLE spread; stop rates | – | D; changed after `b649751` (see below) |
| σ (l.92) | 0.008 | not stated | – | D (spread/√2 = 0.00809); add it to the table |
| SPLIT_THR (l.102, 168) | min(pilot) − 0.03 | S75's rule | learned fires at probe 0.519–0.542 | D |
| trigger constants (l.119–120, 169) | 2400/4800/21,600, 0.02, 3, 4800, 64 | `explore_i_common` l.103–105 | learned fires at 7200/12,000/16,800 | D |
| guard (l.121–127, 170) | one-to-one block | pre-run note l.119 | inert on all 90 learned rows | D |
| WIDE width, k, SLOW (l.115) | 16, 4, SLOW | S75 | – | D |
| ROUTED, reading, VALID (l.139–143) | η² ≥ 0.9; 8/10; 0.9 | S75/S74 | – | D |
| R (l.152, 173) | 1 | K9; memory | learned runs also R = 1 | D (K9: R = 2 costs 4–11% more per run) |
| register-mode rule (l.154–156) | scan if it fits | benchmark to come | S75's REG3 ran `scan` on the 1660 | D (operational) |
| allocation, fallback (l.149–151) | 1660 / 3070; all on the 1660 | "fixed by rule" | the fallback is what S75 did (report_2 l.20–21) | D (operational; not among the inputs listed at l.19–23) |
| WIDE_SLOW unconditional (l.117, 133) | runs always | not stated | – | D; changed after `b649751` (see below) |
| task CHECK n (l.174, 229) | 4096 | `check_task` default | – | D ("verified" failure at 512 is not committed) |
| guard CHECK (l.230) | threshold 1.01, rise 1.0 | – | – | D (makes every check from 4800 eligible) |
| profile scale (l.184) | 1.10–1.25 | K9 against the profile | – | D |
| 3070/1660 (l.184) | 0.77–0.79 | 3070's K9 (`13de308`) | – | D (0.766–0.788) |
| REG3 at L = 768 (l.185) | 0.0939 / 0.0740 s | K9 | learned REG3 0.0951 s (1660) | D: exactly K9, not the learned records |
| REG3 L-factor (l.185) | ×3.5–4 | none | – | **U**; the other arms' L-ratios are 3.24–3.35 |
| gate tensor (l.187–189) | 0.07 / 0.28 GB | 32 × T² float32 | learned arms also +0.070 | D (0.070 / 0.281 GiB; ORACLE record − K9 = +0.070) |
| usable memory (l.190–191) | 5.5; 7.0; 0.75 | K9 | – | D (3070 used − reserved = 0.746) |
| CPU-side (l.201–206) | 21 s, 4.3 GB; 9–13 GB | profile; extrapolation | – | D / U (profile not committed; 9–13 GB unsourced) |
| totals (l.210–216) | 32 h; 30–32 h; 85–101 h | the table | – | D (32.3; 29.6–32.4; 85–101 excludes the 3 h pilot, 88–105 with it) |

**The version diffs.**
- **`1e31409` → `ee97443`.** Only line 113 (`__COST__`) changed, replaced by the 28-line cost section; the message says so. Every cost number comes from K9 at `203a5ad` or the CPU profile. The learned records' own times and memory (WIDE_SPLIT 0.0753 s and 0.983 GB; REG3 0.0951 s and 1.368 GB) are not used. This postdates `b649751`; nothing in it could reflect the learned arms.
- **`ee97443`/`15c5e4f` → `114cd08`.** Every change postdates `b649751` and predates `4c3984a`.
  - **Corrections and operational choices** (none a function of a learned-arm quantity):
    - the bound (0.964 → 0.963);
    - the gap share (0.918 → 0.919);
    - the windowed learner and the fixed-reach case;
    - the stop-rule wording and the guard's precedence;
    - the allocation, R, the register rule and the workers;
    - n = 4096;
    - the memory +0.28 GB;
    - the 3070's factor.
  - **Two changes are design, and both make the screen more informative:**
    - *Proceed rule loosened* (`1e31409` l.59 → l.85–87). This raises the chance that stage 2 runs from 0.36–0.80 to 0.63–0.96 under the draft's own predictions. The learned arms say nothing about ORACLE at B, and the rule follows from ORACLE-only inputs (section F).
    - *WIDE_SLOW unconditional* (`ee97443` l.137 offered "without WIDE_SLOW… about 16 h"; l.117 and l.133 now run it always). After WIDE_SPLIT went 0/10 with every fire on a merged map, WIDE_SLOW is exactly the arm one would want. But the pre-run note already gave it that role (`5bc6c2e` note l.102–104).
  - **Verdict.** Neither change shows any sign of using the learned arms, and neither is blind by git. l.24–26 should say they were made with the learned-arm records in the tree.

## Fix list

1. Retitle l.12 and reword l.170 so blindness is stated as the author's word: "designed on a tree without the learned arms (`1e31409`, parent `5bc6c2e`); blind by the author's statement". Add the server push times: `b649751` at 13:53:11, `142f47e` at 14:16:16.
2. At l.18–23, say that `ee97443` and `114cd08` were made with `b649751`'s records and summary in the tree. Say who the "blind review" was, and whose statement "no learned-arm file opened" is. Say when the same session began the S75 write-up (`03bffdf`, 130 s after `114cd08`). Correct report_2.md l.78–80 to match.
3. At l.164, cite `5bc6c2e` note l.116–118 as B's first, pre-`b649751` statement.
4. At l.85–86, write 0.912 as "0.9 + 0.012, rounded up from the unrounded spread 0.01144".
5. At l.90–93, print the stop and VALID rates for σ = 0.004, 0.008 and 0.012. Say that σ comes from one pair (95% interval 0.0036–0.258) and that the per-bin case assumes a common mean (seed-specific means give 0.32 for the old rule).
6. Add rules for:
   - a failed pilot run (stop, retry the same seed, or a fresh seed; and the report sentence);
   - a code-SHA change between the stages;
   - a missing or failed stage-2 ORACLE run (UNTESTED or rerun);
   - a failure of the L = 1536 benchmark.
7. Reconcile l.149–151 with the round-5 prompt (pilot on the 3070 if it is free first). Either allow it and say what the stage-1 record states, or keep the 1660 and say the prompt is overridden.
8. At l.96–97, drop "S74's usability bar" or apply S74's own statistic (ORACLE's mean). Make the "not usable" label match the rule that sets it.
9. Print a replay of the S75 trigger, with SPLIT_THR, on the ORACLE_PILOT and the ORACLE 850–851 curves, as a description. This is the pre-run note's hazard table for B.
10. At l.123, state "cap" when cap and gap both hold, as `make_split` l.544 does. At l.145, print the guard-blocked count and the count of eligible checks on one-to-one maps separately. Add the "eligible + gapped + one-to-one → gap" case to the guard CHECK (l.230–233).
11. At l.126, replace "the same test ROUTED uses" with what the guard reads: `one_to_one` alone, on the 64-sequence probe, at the first gate layer. State that a low-η² one-to-one map blocks a fire. Any change to that (for example, an η² condition) is not blind and must say so.
12. State that rule 4 fires on a merged map even when `on_target` is false (`c0` live), or block it. Either choice is made after the learned arms (30 of 30 fires were on target in S75) and must say so.
13. At l.119 and l.169, say the trigger also checks at 2400 (the reference check, never eligible) and that eligibility starts at 4800.
14. At l.24–26, list the proceed-rule change and WIDE_SLOW's change to unconditional as "made with the learned-arm records in the tree", not just as "made after a blind review".
15. Source REG3's ×3.5–4 at l.185 (the other arms measure 3.24–3.35) and the 9–13 GB host peak at l.206. Say that "85–101 h" (l.214) excludes the 3 h pilot.

## What I could not check

- **The ideal-learner numbers need runs of `explore_i_task`, which this round does not allow.** These are 0.963, 0.927/0.939, 0.691, 0.517, the bin shares, the per-bin transfer 0.928/0.916 and the fixed reach 0.909/0.920. The same applies to the "verified" failure of `check_task` at n = 512 (l.174).
- **The CPU-profile figures** (21 s, 4.3 GB, 27.6k kernel launches): no profile output is committed.
- **The 14:10 fetch, and what the author or the reviewer opened.** Git holds neither.
