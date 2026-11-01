# Spec review 3: the register test draft, revision 2 (Session G, `explore_out/G/spec_draft_register.md` at `6fc55b17`)

*Referee V, 22 October 2026. Before the main line adopts the draft and before any run.*

**Conventions:**
- "r2 l.NN" is a line of the draft at `6fc55b17` (claude/explore-G, 2026-10-22 05:58:45 UTC). The draft is unchanged at G's head `cbca3e36` (out §2).
- "r1 l.NN" is a line of the draft at `c98648e6`.
- "v2 fix N" / "v2 §N" is a fix or section of `spec_review_register_v2.md`, the second review, on revision 1.
- "S90 rev." is `screen_review_s90_G.md` (05:25:37). "chk9" is `check_G_report9.md` (05:14:13).
- "out §N" is a section of `spec_checks_register_v3_out.txt`, written by `spec_checks_register_v3.py`. That script is stdlib only, reads git and nothing else, and trains nothing.
- The diff `c98648e6 → 6fc55b17` changes only this file (+109 / −29; out §2).

**Verdict: ready for X's adoption after the fixes at the end.** All of them are text. None changes an arm, a seed, a threshold or a claim, and none needs a run.

**The tally:**
- **v2 fixes 1-13:** 13 of 13 RESOLVED.
- **v2's PARTLY and NOT items:** both RESOLVED.
  - The PARTLY was v1 fix 5, the both-UNTESTED rate row. It is now at r2 l.242.
  - The NOT was v1 §11.2 and the plan's label-free arm. It is now decided, as option (B), and dated.
- **New in revision 2:** 9 statements tested. 5 hold. 4 need text:
  - the S89H budget is wrong;
  - one S89H parenthesis is ambiguous;
  - report 10 is cited before it existed;
  - the date windows are loose.
- **Also found (not new in revision 2):**
  - "falls short" is printed without what it means;
  - the rate names need a precedence rule;
  - my own v2 fix 4 text makes the A_r = 0 CHECK's "key term bit for bit" either vacuous or able to fail spuriously.

**The label-free-arm decision.**
- **It matches S90's outcome.** S90's rule, fixed before its runs, maps REPEAT 17/20 against KH 20/20 to "is weaker" → "does not hold" → "falls short". V's S90 fixes 1-2 give the same answer, because 40 of 40 runs completed.
- **It is dated after S89, S89H and S90.** It was committed 71 s after S90's last records. The records had already fixed it at 05:49:13, on the second REPEAT failure.
- **The draft says why no label-free arm is in it.** It gives three reasons, at r2 l.16-33.
- **What it leaves out.** "Falls short" is a count rule against a 20/20 ceiling, not a test that the repeat hinge is worse (p 0.125). The draft does not say so. Report 10 does (`report_10.md:32-33`).

## 1. v2 fixes 1-13, item by item

| v2 fix | asked | revision 2 (file:line, quoted) | state |
|---|---|---|---|
| 1 | the label-free arm: (A), (A′) or (B), written and dated; the header's dating sentence | r2 l.16: "**No label-free hinge is an arm (decided 2026-10-22, after S89, S89H and S90).**"; l.32-33: "So the test does not answer whether a distant cue is discovered label-free: its label-free arms are BASELINE and REG3, its comparators."; l.16-17: the plan's request cited (`plan_2026-10-13.md:48-50`); l.9-14 "**Dating.** Revision 1 was written after S89's records (2026-10-20 22:16 to 2026-10-21 09:46 UTC). Its arms, seeds, thresholds and claims were fixed by the referee's review of 2026-10-12. Only the background's S87 and S89 bullets came after. Revision 2's decision ... was made on 2026-10-22, after S89's ..., S89H's ... and S90's records ..., by the rule S90 fixed before its runs" | **RESOLVED** (option B, with the prompt's branch). See §3 for the date check and fixes 2-4 for what is missing. |
| 2 | the caveat in counts | r2 l.23-29: "The frequency cut selects the 2 context and 4 key types: 14 positions per sequence, the key term's 8 among them, plus the query's key and the 5 context tokens. The repeat set is 5 CTX (4 block heads and the query's) + 9 KEY (8 body and the query's) + 2.63 VAL positions per sequence (values that recur; P = 0.33 each, none in 16% of sequences): 16.6 positions, the key term's 8 among them. ... Neither is given a label (the repeat hinge reads only which token types repeat within the sequence, no corpus statistic)"; caveat line l.231-232 "No label-free hinge is an arm (S90: ...)" | **RESOLVED.** The counts match G's CHECK rows and my simulation (§4). Under (B) the counts sit in the decision paragraph. The task-structure section (l.136-142) is unchanged, which is right with no hinge arm. My caveat-line sentence was conditional on (A)/(A′) and is correctly not used. |
| 3 | the S89 bullet | r2 l.80-89, verbatim, including "the frequency hinge 7/20 (its cut selects only the 2 CTX types there; "needs frequent keys", 13 vs 0 ...)" and "BASELINE 0/20" | **RESOLVED** (counts MATCH, out §3) |
| 4 | RESGATE_KEYHINGE: the A_r = 0 CHECK; "the residual input only" qualified | r2 l.278-279: "With A_r = 0, RESGATE_KEYHINGE's gate at every layer equals LOCAL3_KEYHINGE's window gate, and its logits and its key term equal LOCAL3_KEYHINGE's, bit for bit, on one forward of 16 sequences."; l.122-123: "(with it, a gate that differs by layer; the CHECK below asserts the rest)" | **RESOLVED** as I wrote it. My text has a flaw, corrected in fix 9 (§5). |
| 5 | RESGATE_KEYHINGE's reference construction | r2 l.124-126 and l.291-292: "`explore_g_rhl_child.make_reg3(task, seed, "hebb", reg=False)` (:137), given `KeyHingeBDH`'s class and term state as `explore_g_devices_child.make_model` does (:151-159). No Hebbian record of this construction exists. S73's RESGATE_ONLY_D is delta and has no term: 0/10 ROUTED-BOUND, 5/10 BOUND." | **RESOLVED.** At `076005d`, `make_reg3(task, seed, mem, reg=True, ...)` is at :137, and `"hebb"` is a valid `mem`. make_model's class swap and `kh` state are at :156-159, and `KeyHingeBDH` subclasses `rc.Reg3BDH` (out §2). |
| 6 | C4's power printed | r2 l.200-202: "At REG3_KEYHINGE 1.0, P(C4 SHOWN, 40 pairs) is 1.00, 0.98, 0.74 and 0.37 at RESGATE_KEYHINGE's true rate 0.50, 0.75, 0.85 and 0.90. C4 not SHOWN is printed with both arms' pooled counts." | **RESOLVED** (recomputed: 1.00, 0.98, 0.74, 0.37; out §6) |
| 7 | the C4 comparator's name | r2 l.225-226: "not shown more often than the same gate and term without the register"; l.228: "than the same gate and term without the register" | **RESOLVED** |
| 8 | the outcome's name, the CHECK's scope, the re-score cited | r2 l.1 "test_register.py — specification (... to adopt as specs/test_register.md)"; l.155-157: "`test_register.routed_bound(r)` in `test_register.py` (the plan's file name, `plan_2026-10-13.md:50`; X records the name at adoption if it differs), copied from `explore_g_rhl_parent.py:35-53` at `076005d` (blob `0e1b03a`; ...)"; l.268: "On the 186 Hebbian records of"; l.272: "on all 218 records (186 Hebbian, 32 delta)"; l.272-273: "(Re-scored by the referee on 2026-10-21, `docs/review_2026-10/spec_checks_register_v2_out.txt` §3: all equal.)" | **RESOLVED.** The blob `0e1b03ad69` is the same at `076005d`, `6fc55b17` and the head, with lines :35/:39/:43/:52. The plan names `specs/test_register.md` and `test_register.py` (:50). The 11 counts are 11 of 11 MATCH again, and the records are 218 = 186 + 32. The cited file exists on claude/review-V (out §2-3). Revision 1's change list (l.340) still says `test_register_gate`, as history. |
| 9 | the task file's blob pinned | r2 l.41 "copied byte for byte from claude/explore-F at 9a6c2bf, blob de47463"; l.99-101 (my text, verbatim); l.261: "the vendored file's blob is `de47463` (or `e24d5e1` with its first 14,101 bytes equal to `de47463`)" | **RESOLVED.** `de47463` is 14,101 bytes. `e24d5e1` is 49,923 bytes, has it as its prefix and defines no name twice. It is the module at `c98648e6`, `6fc55b17` and the head (out §2). |
| 10 | "R UNTESTED on both machines" | r2 l.242 | **RESOLVED.** But see fix 8: read literally, the six names give two names to 5 of 9 states. My v2 enumeration silently gave UNTESTED precedence. |
| 11 | the lean's sample | r2 l.313-315: "on run_one's statistics probe is > 0.02 at every layer (`explore_g_rhl_child.py:514`; Session G's CHECK used 256 sequences of its own, `explore_g_rhl_child.py:509-514`)" | **RESOLVED** |
| 12 | the name clash | r2 l.254-255: "The count of runs where ROUTED* (margin ≥ 0.9 and η² by stream > 0.9) on the returned gate differs from ROUTED* on each of layers 1-3 separately, listed." | **RESOLVED** |
| 13 | the change list | r2 l.358-381: the revision 2 list, items 1-10; l.376-377: "Revision 1's background edits: S87 and S89 added (written after S89's records, 2026-10-21); report 6's quoted readings removed from the S83 bullet."; l.379-380: the seed re-check sentence | **RESOLVED** |

**v2's PARTLY and NOT items:**

| v2 | item | revision 2 | state |
|---|---|---|---|
| §1, fix 5 (PARTLY) | no rate reading for both machines UNTESTED | r2 l.242 | **RESOLVED** |
| §2.1, V1 §11.2 (NOT) | the term that reads less; the plan's FREQHINGE arm | r2 l.16-33: decided (B), with the plan cited, S87's FREQ "read the same on the same seeds (17/20; FREQ vs REPEAT 1 / 1)" (l.21), and reasons | **RESOLVED** |
| §2.2 | "186 records"; the uncommitted re-score; the task blob; the S89 omission; RESGATE's 4 gaps; the C4 name | as fixes 3-9 above | **RESOLVED** |

## 2. What revision 2 adds, tested

| new in r2 | lines | test | finding |
|---|---|---|---|
| S90 counts in the decision | l.18-21: "REG3_REPEATHINGE 17/20 against REG3_KEYHINGE 20/20 (key hinge only 3, repeat only 0, p 0.125)"; "FREQ vs REPEAT 1 / 1" | `s90_results.json` at `2c1913b2`, S87's records (out §3) | **MATCH.** REPEAT fails on 2700, 2713 and 2717; FREQ fails on 2700, 2709 and 2713. So 1 vs 1 (p 0.75). |
| S90 bullet | l.93-94: "REG3 4/20, the repeat hinge 17/20 against the key term's 20/20 (3 vs 0, p 0.125)" | same | **MATCH.** KH over REG3 is 16 vs 0 (p 1.5e-5), and REPEAT over REG3 is 13 vs 0. All 40 runs ran 24000 iters on attempt 1. |
| S89H bullet, POOL | l.90-91: "POOL BASELINE 0/20, REG3 9/20, the key term 18/20, the frequency hinge 18/20, the repeat hinge 19/20 (the hinge needed over the register, 9 vs 0, p 0.002)" | `s89h_results.json` at `2c1913b2` | Counts **MATCH**. The parenthesis follows "the repeat hinge 19/20", but 9 vs 0 is the **key** hinge over REG3 (2b, p 0.00195). The repeat hinge over REG3 is 10 vs 0 (p 0.00098). Fix 1. |
| S89H bullet, ALLPOOL | l.91-92: "ALLPOOL every arm 0/20 at 48000" | `iters` on all ALLPOOL records | **DISAGREE.** Every ALLPOOL record has `iters` 24000, and the largest `stopped_at` is 24000 (out §3; chk9 verdict, rows A1, I2). chk9 was on claude/review-V 44 min before revision 2. Fix 1. |
| "the repeat hinge equals the key hinge" on POOL | l.30-31: "(S89 delta 20/20 = 20/20; S89H Hebbian + conv 19/20 vs 18/20)" | out §3 | The counts are right: 0 vs 0, and KH 0 vs REPEAT 1. But "equals" is S89's count rule "replaces" (within one pair), not an equivalence. On S89's delta, no hinge is shown needed: KH over REG3_D is 4 vs 0, p 0.0625 (`check_G_report8`). Fix 5. |
| Dates | l.11-13: S89H "2026-10-21 22:10 to 2026-10-22 03:11 UTC"; S90 "2026-10-22 05:04-05:57 UTC" | commit times (out §0) | **Approximate.** S89H's CHECK file was written at 22:09:49, and its first records were committed at 22:16:58 (`1f6ea2e6`). Its last 3 records (201 → 204) came in report 9's own commit, at 03:12:24. S90's parent was committed at 05:03:55 and its CHECK file at 05:08:29. Its records run from 05:13:57 (`5c04a858`) to 05:57:34 (`2c1913b2`). None of this changes the order. Fix 4. |
| Report 10 cited | l.13 "report 10's S90 part"; l.93 "report 10" | existence at `6fc55b17` | **Not in the commit.** `report_10.md` was first added at `cbca3e36` (08:26:53), 2 h 28 min later. The README's "Session G — S90" section did exist at `6fc55b17`. Fix 4. |
| "by the rule S90 fixed before its runs" | l.13-14 | `explore_g_s90.py:48-50, 99-101` at `b57c1867` | **True:** 05:03:55, before any record. Two omissions: the rule was written after S89H's records, report 9 and the round-7 prompt (S90 rev. §1h.2); and V's S90 fixes 1-8 were not applied (the blob `341c4ff` is unchanged to the head; out §0). Fix 2. |
| Seeds sentence | l.380-381: "Session G's S89/S89H/S90/S91 use 2700-2719, 2840-2861 and 2960-2991" | scan (out §1) | **True**, and disjoint. S91's seeds come from the round-7 prompt, because S91's parent (`60b1932a`, 06:06:19) postdates revision 2. 2700-2729 are in use, since S87's delta arm ran on 2720-2729. |

## 3. The label-free-arm decision against the records

### 3.1 The timeline (UTC, commit times; out §0)

| time | commit | event |
|---|---|---|
| 10-20 22:16:41 → 10-21 09:46:54 | `0bd95b21` … `8e9e3187` (88 commits) | S89's records; report 8 |
| 10-21 22:01:53 / 22:09:49 | `d5d48ee` / `d877f157` | S89H's readings and CHECK pass |
| 10-21 22:16:58 → 10-22 03:12:24 | `1f6ea2e6` … `996e38b7` | S89H's records; the last 3 arrive with report 9 (`996e38b7`) |
| 03:38:40 | `5e51a294` (main line) | round-7 prompt: "If S90 reads 'the repeat hinge falls short', the arm is the key hinge and the draft says why a label-free arm is not in it" (prompts:1291-1292) |
| 05:03:55 | `b57c1867` | S90's parent: readings and "FOR THE SPEC" (s90:48-50, 99-101) |
| 05:08:29 | `00348e77` | S90's CHECK file (81/81), 0 records |
| 05:13:57 | `5c04a858` | first S90 record |
| 05:14:13 | review-V `a141e92b` | chk9: the budget defect (S89H ALLPOOL ran 24000) |
| 05:25:37 | review-V `e20c4e7a` | S90 rev.: fixes 1-2 to `spec_decision` (UNTESTED; REG3 completeness) |
| **05:49:13** | `1bd9c956` | REPEAT 15/17, failing 2700 and 2713. **The decision is fixed from here:** c ≤ 18 whatever the last 3 do. |
| 05:57:34 | `2c1913b2` | S90 done: REPEAT 17/20, REG3 4/20; "FOR THE SPEC: the repeat hinge falls short" |
| **05:58:45** | **`6fc55b17`** | **revision 2, 71 s later** |
| 06:06:19 | `60b1932a` | S91's parent |
| 08:26:53 | `cbca3e36` | report 10 |

**Dated as after S89, S89H and S90: yes.** Revision 2 is the first commit after S90's last record. The decision follows mechanically from the counts. Nothing else in revision 2 (arms, seeds, thresholds, claims) was chosen on S89H's or S90's numbers. The only changes in those categories are fixes 4-13, and all of them were asked on 21 October, before S89H.

### 3.2 Is "17/20 vs 20/20, is weaker" a "falls short"?

- **Under G's pre-fixed rule (s90:48-50): yes.** The repeat hinge would have been the arm only on "holds, and the hinge is needed here". With KH 20/20 and BASELINE 0/20, that happens iff REPEAT ≥ 19/20 and REG3 ≤ 15/20 (S90 rev. §1a). REPEAT 17/20 reads "is weaker" (discovered, KH only − REPEAT only = 3 ≥ 2, p(KH >) 0.125 > 0.0625), so "does not hold", so "falls short". `s90_summary.json` prints exactly this.
- **After V's S90 fixes 1-2: the same.** UNTESTED would give no decision, and fewer than 18 completed REG3 seeds would void it. Neither applies: REG3 completed 20/20 and REPEAT 20/20, all on attempt 1. The fixes were not applied to the code (the blob is unchanged), but on these records they do not change the output (out §4).
- **Under the prompt: yes.** The prompt names "falls short" but does not define it. Under the screens' "replaces = within one pair" convention, 3 pairs is outside it. No reading of the prompt makes 17/20 vs 20/20 the repeat hinge "holding".
- **What "falls short" does not mean.** It is not evidence that the repeat hinge is worse:
  - KH over REPEAT is 3 vs 0, one-sided p 0.125.
  - REPEAT's 17/20 has a Wilson interval of [0.64, 0.95].
  - At a true rate of 0.9 the rule reads "falls short" with probability 0.61. At 0.95 it is 0.26 (out §4).
  - Report 10 says so (`report_10.md:32-33`). The draft quotes p 0.125 (l.20) but never says what follows from it (fix 3).

### 3.3 Does the draft say why a label-free arm is not in it?

Yes, at r2 l.16-33, with three reasons:
1. S90's rule reads "falls short".
2. On lite either hinge reads a superset of the key term's positions and grouping, so the test could not separate a token statistic from the key positions.
3. The tasks where the hinges differ (POOL, ALLPOOL) are other tasks.

It names where a test of the repeat hinge belongs: its own spec, on POOL, Hebbian + conv. It also records that the plan of 2026-10-13 asked for FREQHINGE, and that FREQ read the same on the same seeds.

Reason 2 holds in counts (§4). Two things are missing. One is reason 3's "equals" (fix 5). The other is the plain statement that on lite the repeat set is the frequency set plus the chance-repeated values (S90 rev. §1i; fix 6).

## 4. The v2 tests, re-run on what changed

**Seeds (out §1).** They are free.
- I scanned today's 13 heads (the main line at `77221b8a`, G at `cbca3e36`): 381 JSON/JSONL blobs, 1,103 branch-paths and 39,846 seed values.
- No record uses a seed in 600-619 or 1600-1619. The integer keys 600 and 1600 occur only as steps and tolerances (`by_step`, `eta_at`, `1e-06`, ...).
- G's later screens use 2700-2729, 2840-2861 and 2960-2991, all disjoint from the block.

**The outcome function (out §2-3).** It is unchanged. The blob is `0e1b03a`, with :35/:39/:43/:52 at `076005d`, `6fc55b17` and the head. The CHECK's 11 counts are 11 of 11 MATCH. The transition equality holds on 218 of 218 records. No `test_register*.py`, `register_gate.py` or `specs/test_register*.md` exists yet on the main line or on G, as expected before adoption.

**The rate claim's operating characteristics (out §5).** This is unchanged text, and l.208-209's "0.04, 0.16, 0.46 and 0.85" is correct. At the in-setting KH rates (S82 and S87 both 20/20; taking 0.975), P(MET on both) is 0.97. At S89H's 18/20 (another task, Hebbian + conv), it is 0.46.

**McNemar power per claim** (exact one-sided, at the screens' observed discordance; out §6). b ≥ 5 is needed with c = 0, or b ≥ 7 at Holm's 0.0125.

| claim | screen (pairs) | P(SHOWN) 40 pairs | at Holm 0.0125 | 20 pairs |
|---|---|---|---|---|
| C1 | S82 20 vs 0; S87 20 vs 0; S89 delta POOL 20 vs 0; S89H POOL 18 vs 0 | 1.000 (each) | 1.000 | 1.000 |
| C2 | S83 20 vs 0 | 1.000 | 1.000 | 1.000 |
| C3 | S82 18 vs 0 | 1.000 | 1.000 | 1.000 |
| C3 | **S90 (lite, in setting): KH(S87) vs REG3 16 vs 0** | 1.000 | 1.000 | 1.000 |
| C3 | S89H POOL 9 vs 0 | 1.000 | 1.000 | 0.981 |
| C3 | S89 delta POOL 4 vs 0 (other memory and task) | 0.924 | 0.714 | 0.370 |
| C4 | unscreened; KH 1.0, RESGATE at 0.50 / 0.75 / 0.85 / 0.90 | 1.00 / 0.98 / 0.74 / 0.37 | 1.00 / 0.90 / 0.39 / 0.10 | — |
| (left out) KH vs REPEAT | S90 3 vs 0 | 0.737 | — | 0.170 |

S90 adds the first in-setting REG3 rate since S82's 2/20, at 4/20, and C3's power stays at 1.000. The C4 Holm column is not printed in the draft. That is not needed, since the readings use raw p.

**Readings (out §7).**
- **C1-C4.** Each of the 16 combinations gets exactly one name.
- **R.** With UNTESTED taking precedence, 9 of 9 states get exactly one name. Read literally, only 4 of 9 do. For example, (X UNTESTED, L MET) matches both "R MET on L only" and "R UNTESTED on X". The draft states no precedence (fix 8).

**CHECKs falsifiable.**
- Every CHECK revision 2 adds can fail before a run: the blob and prefix CHECK (l.261), the scope of the outcome CHECK (l.268-273), and the A_r = 0 equivalence (l.278-279).
- One exception, in my own v2 fix 4 text, concerns the A_r = 0 CHECK's "and its key term equal LOCAL3_KEYHINGE's, bit for bit":
  - At initialisation η²_key is typically below τ = 0.2, so both terms are relu(·) = 0, and the comparison passes whatever the gates are.
  - When the term is active, RESGATE's term is a mean over three layers' equal values, and LOCAL3_KEYHINGE's is over one gate (l.132). A float32 mean of three equal values differs from the value in 17% of uniform draws (out §9).
  - So the clause is either vacuous or able to fail spuriously (fix 9).

**The caveat in counts (out §8).**
- **G's CHECK rows** (`checks_s90_20261022_050720.json`): repeat 16.63 = CTX 4 + KEY 8 + VAL 2.63 + QUERY 2; frequency 14.00 = CTX 4 + KEY 8 + QUERY 2. The key term's 8 are inside both, and the grouping is one to one.
- **My simulation of `make_batch`'s draw** (200,000 sequences): 2.638 repeated VAL positions per sequence, 0.330 per position, none in 15.7% of sequences.
- **The draft** gives "5 CTX (4 block heads and the query's) + 9 KEY (8 body and the query's) + 2.63 VAL ... P = 0.33 each, none in 16% ... 16.6 positions" and "14 positions". All **MATCH**.

**RESGATE_KEYHINGE's CHECK and reference.**
- The reference is constructible as named (fix 5 row above).
- The A_r = 0 CHECK can fail on gates and logits. Adding 0·LN(r) to A_v·window is exact, A_v = W_in and W_g2 = W_g are copies, and the same holds for REG3's A_r = A_R = 0 CHECK at l.275.
- Its term clause needs fix 9.
- "At λ = 0 equals REG3 with reg = False" (l.306-307) compares two constructions of the same code. It can fail if the term's wiring changes the forward.

## 5. Remaining fixes (replacement text; the first-named first)

1. **The S89H bullet** (r2 l.90-92). It becomes:
   > S89H (2842-2861, Hebbian + conv 'layer', the same tasks; report 9): POOL BASELINE 0/20, REG3 9/20, the key term 18/20, the frequency hinge 18/20, the repeat hinge 19/20 (each hinge needed over the register: the key hinge 9 vs 0, p 0.002; the repeat hinge 10 vs 0, p 0.001); ALLPOOL every arm 0/20 at 24000 (the parent's 48000 did not reach the runs; `check_G_report9.md`). Hebbian with a conv, on other tasks.

2. **The rule's provenance** (r2 l.13-14). After "by the rule S90 fixed before its runs (`explore_g_s90.py` at b57c186, "FOR THE SPEC")", add:
   > (written 2026-10-22 05:03:55 UTC, after S89H's records, report 9 and the round-7 prompt, and before any S90 record. Referee V's screen review of S90 (`screen_review_s90_G.md`, 05:25) asked for two corrections to it: UNTESTED gives no decision, and fewer than 18 completed REG3 seeds void it. They were not applied in the code, and on S90's records they change nothing: 40 of 40 runs completed on attempt 1.)

3. **What "falls short" means** (after r2 l.21, "...the arm is the key hinge."). Add:
   > "Falls short" is the pre-fixed count rule's output, not a test that the repeat hinge is worse: the key hinge's lead is 3 vs 0 (one-sided p 0.125), REPEAT's 17/20 has Wilson [0.64, 0.95], and an arm whose true rate is 0.9 reads "falls short" with probability 0.61.

   In the caveat line (r2 l.232), "(S90: the repeat hinge 17/20 against the key term's 20/20 on this setting)" becomes "(S90: the repeat hinge 17/20 against the key term's 20/20 on this setting, 3 vs 0, p 0.125: a count rule, not a test)".

4. **The dates and the citations** (r2 l.11-13, l.16, l.93):
   - "S89H's (... 2026-10-21 22:10 to 2026-10-22 03:11 UTC)" becomes "S89H's (... records committed 2026-10-21 22:16:58 to 2026-10-22 03:12:24 UTC, `1f6ea2e6` to `996e38b7`)".
   - "S90's records (lite Hebbian; 2026-10-22 05:04-05:57 UTC; report 10's S90 part, README "Session G — S90")" becomes "S90's records (lite Hebbian; committed 2026-10-22 05:13:57 to 05:57:34 UTC, `5c04a858` to `2c1913b2`; `s90_summary.json` and README "Session G — S90" at `2c1913b2`; report 10, `cbca3e36`, came later)".
   - l.16 "(decided 2026-10-22, after S89, S89H and S90)" becomes "(decided 2026-10-22 and committed at 05:58:45 UTC, 71 s after S90's last records; after S89, S89H and S90)".
   - l.93 "report 10" becomes "README S90 section at `2c1913b2`; report 10 at `cbca3e36`".

5. **"Equals"** (r2 l.30-31). "there the repeat hinge equals the key hinge (S89 delta 20/20 = 20/20; S89H Hebbian + conv 19/20 vs 18/20)" becomes:
   > there the repeat hinge reads "replaces the key hinge", a within-one-pair count rule and not an equivalence (S89 delta 20/20 and 20/20, 0 vs 0, where the register needed no hinge: the key hinge over REG3_D 4 vs 0, p 0.0625; S89H Hebbian + conv 19/20 and 18/20, 0 vs 1)

6. **The repeat set against the frequency set** (after r2 l.27, "...one to one."). Add:
   > On every lite sequence the repeat set is the frequency set plus the chance-repeated values, grouped by the same token id. So S90 differs from S87's frequency hinge only by those 2.63 value positions, and the two hinges differ on 2 of 20 seeds (1 vs 1).

7. **The S90 pairing** (r2 l.18-19). "seeds 2700-2719, paired with S87's records" becomes "seeds 2700-2719, paired with S87's BASELINE, REG3_KEYHINGE and REG3_FREQHINGE records, which this CPU reproduces through 2400 (not the full records)".

8. **The rate names' precedence** (after r2 l.242). Add:
   > The UNTESTED names take precedence: the four MET names are used only when both machines are VALID.

   This is my own oversight in v2: my enumeration assumed the precedence and did not ask for it in writing.

9. **The A_r = 0 CHECK's term clause** (r2 l.278-279; correcting my v2 fix 4 text). "and its logits and its key term equal LOCAL3_KEYHINGE's, bit for bit, on one forward of 16 sequences" becomes:
   > and its logits and each layer's η²_key equal LOCAL3_KEYHINGE's, bit for bit, on one forward of 16 sequences; with τ lowered so that the term is active (as in the key term's gradient CHECK), its term equals LOCAL3_KEYHINGE's to 1e-6 relative, with the difference printed

10. **The change list** (after r2 l.377). Add:
   > 11. (dated) The corrections of `spec_review_register_v3.md`: the S89H budget and parenthesis, the rule's provenance, what "falls short" means, the dates and citations, "equals", the repeat set against the frequency set, the S90 pairing, the rate names' precedence, the A_r = 0 term clause.

**Verdict for X's adoption: ready after these fixes.**
- Every v2 fix is applied, and the label-free-arm decision is made, dated and reasoned.
- Fix 1 corrects a wrong number in the background, and fix 9 corrects a CHECK clause (my own). The rest are wording and dates.
- None touches the arms, seeds, thresholds, claims or readings' logic. None needs a run. X can apply them at adoption, dated, if G does not.
