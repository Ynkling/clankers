# Screen review: Session G's S90 (`claude/explore-G` at `b57c1867`, progress to `afe7a1e6`)

*Referee V, 22 October 2026, read at 05:25 UTC. Reviewed:*
- *`explore_g_s90.py` ("s90:N") and `explore_g_s90_child.py` ("s90c:N"), both at `b57c1867`;*
- *the modules S90 imports: `explore_g_s89h.py` ("s89h:N"), `explore_g_s89.py` ("s89:N"), `explore_g_s89_child.py` ("c89:N"), `explore_g_rhl_parent.py`, `explore_g_common.py`, `explore_g_rhl_child.py`;*
- *the progress commits `00348e77`, `5c04a858`, `82d37ae0` and `afe7a1e6`, with their `s90.log`, `s90_results.json` and `checks_s90_20261022_050720.json`;*
- *S87's records (`explore_out/G/s87_results.json`);*
- *the round-7 prompt (`docs/reading/parallel_sessions_prompts.md` on `claude/bdh-growth-hebbian-inference-w90069`, `5e51a294`, l.1283-1285, "prompts:N").*

*It was read against `check_G_report9.md` (the budget defect).*

*"out §N" is a section of `screen_checks_s90_G_out.txt`, which `screen_checks_s90_G.py` writes. The script reads git objects only and trains nothing. torch is absent here, so G's parent code (`explore_g_s90`, `explore_g_s89h`, `explore_g_s89` at `b57c1867`, `explore_g_rhl_parent`, `test_conv_lr.mcnemar_greater`) is exec'd from blobs statement by statement, as in `check_G_report9.py`. It is run on S87's committed records plus synthetic records for the two new arms. G's `report()` is run on the store at `afe7a1e6`, with its file write sent to memory.*

## Status: the learned arms are running, and this review lands after them

The prompt asks for "the file pushed and reviewed by V before the learned arms" (prompts:1276). That did not happen. The timeline (UTC, out §0):

| time | event |
|---|---|
| 10-22 03:07:32 | `d60e8668`: S89H's last records |
| 03:12:24 | `996e38b7`: report 9 |
| 03:38:40 | `5e51a294`: round-7 prompt |
| **05:03:55** | **`b57c1867`: S90 parent and child** (readings, CHECKs) |
| 05:03:59 | CHECK pass starts (`git b57c186`) |
| 05:07:20 | CHECK file written: **81/81 ok** |
| 05:08:29 | `00348e77`: CHECK file and log; "40 runs queued on 4 workers", code SHA `5d4a9b1dba33` (34 modules) |
| 05:13:57 | `5c04a858`: first learned record |
| 05:14:13 | V's `check_G_report9` (budget defect), 10 min after S90's commit |
| 05:18:59 | `82d37ae0`: 7 records |
| 05:24:02 | `afe7a1e6`: 10 records |

**What has run at `afe7a1e6` (out §2).** Every record is ok, on attempt 1, with iters 24000, k 2, recipe slow, task lite.

| arm | seed | outcome |
|---|---|---|
| REG3_REPEATHINGE | 2700 | unbound to 24000, KEY |
| REG3_REPEATHINGE | 2701-2705 | ROUTED-BOUND, transition 3600, stop 6000 |
| REG3 | 2700-2703 | unbound to 24000, KEY |

- Running tally: REPEAT 5/6, REG3 0/4.
- REPEAT fails on 2700, where S87's frequency hinge also fails.
- The queue runs in seed order, REG3 before REPEAT on each seed (out §2).
- At this pace (REG3 about 7.5 min per run to 24000; a bound REPEAT run about 2 min), the remaining 30 runs need about 35-40 min on 4 workers. They should finish around 06:00 UTC.

**What can still change without rerunning anything.** The code SHA hashes the child's 34 imported modules (meta `sha_rows`). That list includes `explore_g_common` and `explore_g_s89_child`. It does **not** include `explore_g_s90`, `explore_g_s89h`, `explore_g_s89` or `explore_g_rhl_parent`.
- Parent-only edits (readings text, `spec_decision`, `report`) keep the records valid. They now come after records exist, so they must be dated.
- An edit to `explore_g_common.py` (the budget fix) changes the SHA and re-queues all 40 runs. Do not make it during S90.

## Verdict

**Fit to run (do not stop it). Fix the parent before `report()` prints a spec decision. Too late for a pre-run review.**

The run code does what the docstring says:
- the budget reaches every run, by default;
- the S87 pairing is supported through 2400;
- 2b is present;
- the repeat statistic reads token ids only;
- every CHECK passed.

Two defects in the decision layer need parent-only fixes (1 and 2 below):
- `spec_decision` (s90:99-101) turns an UNTESTED screen into "the repeat hinge falls short". G's own `report()` prints exactly that on the current partial store (out §8).
- Missing REG3 runs push 2b towards "the register does not need the hinge", and so also towards "falls short".

Neither fires if all 40 runs complete, but both are decision rules and should be correct before the report.

One fact about the design changes what S90 can show (§4). **On lite, the repeat set is S87's frequency set plus about 2.63 chance-repeated value positions, grouped by the same token id.** So S90 asks whether adding the repeated values to S87's frequency hinge (17/20, "weaker") lifts it to ≥ 19/20. The readings never compare REPEAT with FREQ.

## 1. Each question asked

### 1a. Are the readings exhaustive? (out §4)

`explore_g_s89h.readings` (imported at s90:80, called at s90:109-110) was run, with S87's B, KH and FQ records fixed, on every paired configuration of the two new arms over the 20 seeds. That is 22,800 configurations: the counts of the four (REG3, REPEAT) states within S87's FQ-pass seeds (17) and FQ-fail seeds (3).

- Every configuration gets exactly one of the four answers in `Q` (s89h:137-138), and `spec_decision` maps each to one of its two strings.
- **The question depends only on the two marginal counts.** With r = REG3 count and c = REPEAT count:
  - "holds, and the hinge is needed here" iff c ≥ 19 and r ≤ 15;
  - "holds (descriptive: …)" iff c ≥ 19 and r ≥ 16;
  - "does not hold" otherwise.

  This holds on every (r, c). The reason is that KH is 20/20 and BASELINE is 0/20, so every decisive McNemar has one empty cell.
- REPEAT's name by c:

  | c | name |
  |---|---|
  | 19-20 | "replaces" |
  | 17-18 | "is weaker" |
  | ≤ 16 | "needs key identities" |

  "Does nothing here" cannot fire (as `screen_review_s89_G_v2` §2.10 said for a strong ceiling).
- S87's own FQ records, fed in as REPEAT's, read "is weaker" and "does not hold".
- **Missing runs:**

  | case | readings | spec decision |
  |---|---|---|
  | one REPEAT seed missing | UNTESTED, "cannot be asked here" | **"falls short"** |
  | REG3 with no completed run | `REG` is `{}` (s89h:191, 207), so `need` = False and 2b reads "the register does not need the hinge here (… n 0)"; with REPEAT 20/20: "holds (descriptive)" | **"falls short"** |
  | REG3 with 4 completed, all unbound | KH only 4, p 0.0625, so 2b applies | "falls short" |
  | REG3 with 5 completed, all unbound | 2b does not apply | the repeat hinge |

  So "a missing REG3 seed does not void" (s90:35, s89h:193-194) is right for the names but wrong for the spec decision, which a crashed REG3 run can flip.
- **G's `report()` on the store at `afe7a1e6` prints "FOR THE SPEC: the repeat hinge falls short: the spec carries the key hinge only"** next to "first: UNTESTED" (out §8). Any intermediate `--report-only` would print a spec decision before the runs exist.

### 1b. The outcome function

- `rb = hp.routed_bound` (s90:88), imported from `explore_g_rhl_parent`: BOUND, end margin ≥ 0.9 and end η² by stream at KEY > 0.9. It is pooled, which is right on lite, where the stream is the CTX id.
- `readings` uses its own module-level `rb = hp.routed_bound` (s89h:132), the same object.
- `hp.paired` / `hp.beats` give the exact one-sided McNemar (`test_conv_lr.mcnemar_greater`).
- No `inv_view` is involved.

### 1c. Seeds and the pairing with S87

- **The seeds.** They are 2700-2719 (s90:84), S87's by design (prompts:1274 and 1283-1284).
- **S87's records.** `records()` (s90:91-96) takes BASELINE, REG3_KEYHINGE and REG3_FREQHINGE from `s87_results.json` through `explore_g_s89.recorded` (s89:256-266): a prefix lookup under S87's store SHA `f205db31dfed`, at most one ok record per seed. All 60 are present (out §1):
  - BASELINE 0/20 (17 KEY, 3 STREAM-PARTIAL);
  - KH 20/20 (transitions 3600-9600);
  - FQ 17/20 (fails on 2700, 2709, 2713);
  - all at iters 24000, k 2, slow.
- **The provenance matches.** S87's store has the same CPU string, torch 2.14.0+cu130 and python 3.13.16 as S90's CHECK file.
- **Does the CPU reproduce an S87 record bit for bit?** Only through 2400.
  - The CHECKs are `c89.checks_repro` (c89:694-701) for REG3_KEYHINGE|2700, REG3_FREQHINGE|2700 and BASELINE|2700 through `c89.run`, and `s90c.check_oracle` (s90c:18-22) for ORACLE_ALL|2700 through S87's own path. All 4 are ok (CHECK file rows; log).
  - The comparison is `s87.trunc` (`explore_g_s87_child.py:220-222`): the curve and stats at steps ≤ 2400. The rest of the record is not compared.
  - KH|2700 binds at 3600, so the CHECK ends before its transition. FQ|2700 is unbound to 24000, and only its first tenth is compared (out §1).
  - The prompt's condition ("if your CPU reproduces one bit for bit", prompts:1284) is read by G as through 2400. That is G's own reading of it, and is not stated as one.
- **The controls.** REG3 is new in S90 (S87 ran none on these seeds). REG3_KEYHINGE is **not** rerun: G takes the reproduction branch of the prompt (s90:13-17).
- **Why the pairing hardly matters here.** Every decisive reading reduces to marginals (1a). The pairing matters only to the descriptive lines: REPEAT vs REG3 "hurts", FQ's name, and the commitment medians.
- **What the decision does rest on.** KH being 20/20 *under this code path*. That is supported by three records equal through 2400, unchanged blobs and the same CPU and torch, but not shown on a full record. Fix 4 closes this cheaply.

### 1d. Reading 2b

It is present: s89h:207-209, with REG3 = S90's own `lite.REG3` (s90:93).
- A hand case covers it: "REG3 0, KH 20, RP 20" gives "the hinge is needed" (CHECK row 2).
- On lite, 2b applies only if REG3 ≥ 16/20. Under S82's REG3 rate (2/20 on 2100-2119) that has probability ≈ 0, and the 4 REG3 records so far are 0/4.
- With KH 20/20, 2b is therefore near-certain to read "the hinge is needed over the register". Only REPEAT's count decides the question.

### 1e. Operating characteristics at S87's rates (out §5)

KH 20/20 and BASELINE 0/20 are fixed as recorded. FQ is irrelevant to the decision.
- P("replaces") = P(c ≥ 19) is 1.0 / 0.74 / 0.39 / 0.18 / 0.07 for a true REPEAT rate of 1.0 / 0.95 / 0.9 / 0.85 / 0.8.
- P("is weaker") at 0.9 is 0.475.
- These match s90:51-53, which omits 0.85, S87's FQ rate on these seeds.
- P(spec decision = the repeat hinge) = P(c ≥ 19) · P(r ≤ 15):

| p_REPEAT \ p_REG3 | 0.1 (S82 lite) | 0.45 (S89H POOL) | 0.8 (S89 delta POOL) |
|---|---|---|---|
| 1.0 | 1.000 | 0.999 | 0.370 |
| 0.95 | 0.736 | 0.735 | 0.273 |
| 0.9 | 0.392 | 0.391 | 0.145 |
| 0.85 (S87 FQ's 17/20) | 0.176 | 0.175 | 0.065 |
| 0.8 | 0.069 | 0.069 | 0.026 |

- With REPEAT already 5/6 (failing 2700), the arm is the repeat hinge only if all 14 remaining REPEAT seeds bind. **One more failure fixes "falls short".**
- "Falls short" here is compatible with a true rate of 0.9. It is a count rule against a 20/20 ceiling, not a test that the repeat hinge is worse.

### 1f. Label leaks (out §7)

- `repeat_mask(tokens)` (c89:154-159) takes one argument and reads only `tokens` (`.max`, `.shape`, `.dtype`, `.dim`) and torch. A numpy replica gives G's hand case.
- `hinge_mask` (c89:179-186) calls `_assert_tokens` (c89:148-151): long dtype, width L − 1, values in [0, vocab). So it cannot be handed stream labels, key ids or the target column.
- `term_value` (c89:189-196) groups `g[mask]` by `tokens[mask]` (token id).
- `Term89.forward` passes its own input `tokens` (c89:228).
- No stream label, role, key index or frequency table reaches the repeat hinge. **No label leak.**
- What it does see on lite is task structure:
  - at CTX positions, token id is the stream (s90:27-28 says so);
  - at KEY positions, token id is the key term's key id one to one (CHECK "every key-term position is a repeat position … one to one", ok).
- **One text error (s90:59).** It says the term's "gradient reaches the gate only". The CHECK row lists `embed.weight` (and the register gate's own `gate_conv.conv_w`) among the reached parameters. The "stack" tuple (s90c:56) excludes the embedding. The same holds for S87's key term (r8 E6, r9 F8), so it is not a difference between arms, but the sentence is wrong.

### 1g. The budget (out §3)

**S90 has the same defect as S89H, but here it is latent.**
- `run_screen` sends `dict(arm=arm, seed=s)` (`explore_g_common.py:164`). `ARMS`' `iters=24000` (s90:85) is used only to sort the queue (`explore_g_common.py:152-153`) and is copied into the meta (l.142).
- The child takes `p.get("iters", rc.ITERS)` (c89:481), and `rc.ITERS = 24000` (`explore_g_rhl_child.py:81`). So every S90 run gets 24000 because intended equals default.
- The 10 records confirm it: iters 24000, and the unbound runs evaluate at 1200, …, 24000 and stop at 24000.
- **No CHECK covers the budget.** None of the 81 rows mentions iters. The CHECK runs pass `iters=2400` explicitly.
- The task does reach the child, through the arm name `lite.<ARM>` (c89:470-471; records `task: lite`).
- **For S91 this defect is live.** S91 is "at 48000 on ALLPOOL" (prompts:1313). Under the current `explore_g_common.py:164` it would again run 24000.

### 1h. Was anything chosen after S89H's records?

Yes, all of S90's file was, and none of it after an S90 record.
- `b57c1867` (10-22 05:03:55) is 1 h 56 min after S89H's last records (03:07:32), 1 h 51 min after report 9, and 1 h 25 min after the round-7 prompt.
- No `.py` changed after `b57c1867` (out §0). No S90 record existed before 05:13:57.

What was chosen then, knowing S89H:
1. **The readings.** They are S89H's function, an unchanged blob `c31538f1` fixed at `d5d48eeb` (10-21 22:01:48, before any S89H record). G applies it to lite in place of "readings as relayed", which never reached it (s90:32-34). No relay text for S90 exists in any branch read here. The round-7 prompt only says "readings as relayed" (prompts:1285). So the relayed readings cannot be checked against G's.
2. **The spec bar.** The label-free arm is the repeat hinge only on "holds, and the hinge is needed here" (s90:48-50, 99-101). That is the very outcome S89H's POOL produced. The prompt names "falls short" but does not define it (prompts:1291). G's definition is conservative, but it was written knowing S89H.
3. **The pairing branch.** Use S87's KH instead of rerunning it (s90:13-17).
4. **No V review.** V's `check_G_report9` (the budget defect) landed at 05:14:13, after S90's commit, so S90 could not use it.

### 1i. The repeat set on lite vs S87's frequency hinge (out §6)

G's CHECK rows give per sequence (4096 sequences):

| hinge | CTX | KEY | VAL | QUERY | total |
|---|---|---|---|---|---|
| frequency | 4.00 | 8.00 | 0 | 2.00 | 14.00 |
| repeat | 4.00 | 8.00 | 2.63 | 2.00 | 16.63 |

- The frequent types are all six CTX/KEY ids.
- Both hinges sit at the maximum 14 on CTX + KEY + QUERY. A mean equal to the maximum means every sequence, and the task's construction makes it so anyway: each CTX type heads 2 blocks, each key occurs twice in the body, and the query's CTX and key occur in the body. So on lite **the repeat set contains the frequency set on every sequence and adds only the chance-repeated values**, grouped by the same token id.
- s90:23-28 gives the counts, but does not say that this is S87's FREQ set plus values.
- So S90 measures the effect of 2.63 value positions on an arm S87 recorded at 17/20 "weaker". On 2700-2705 the two agree seed for seed (both fail 2700; both bind 2701-2705).
- Neither the readings nor the report print REPEAT vs FREQ (s90:104-129).
- Whatever S90 reads, the spec's revision 2 (prompts:1288-1290) must say this in counts: "on lite the repeat set is the frequency set plus ~2.6 value positions".

## 2. What has run: CHECKs and records

- **CHECKs** (`checks_s90_20261022_050720.json`, git `b57c186`, 81 rows, all ok, 201 s):
  - X's arm A bit for bit;
  - S89H's 10 reading rows (8 hand cases, 2 exhaustive name enumerations at n = 10 and 20) and the inv_view row;
  - the 4 S87 reproductions through 2400;
  - the repeat set's counts on lite and the one-to-one key grouping;
  - REG3_REPEATHINGE switched off = REG3|2700 through 2400;
  - the gradient row;
  - S89's static CHECK set (vendored module, pool and allpool task checks, `frequent()` raises on labels, the hinge position counts, the task-aware statistics, ORACLE_D, the untrained perfect gate ROUTED*).
- **Records at `afe7a1e6`:** 10 of 40 (§Status).
- **No reading has been printed by G.** The log ends at run 10.

## Fixes (parent only: `explore_g_s90.py`; the code SHA and the records stay valid; each dated "22 October, after N S90 records, before report()"), most important first

1. **UNTESTED is not "falls short".** Replace `spec_decision` (s90:99-101) with:
   ```python
   def spec_decision(rd):
       if rd["question"] == s89h.Q[0]:
           return "the spec's label-free arm is the repeat hinge"
       if rd["question"] == s89h.Q[3]:
           return "UNTESTED: no spec decision (S90 could not be asked); the spec waits"
       return "the repeat hinge falls short: the spec carries the key hinge only"
   ```
   In the docstring (s90:48-50), replace "Every other outcome reads 'the repeat hinge falls short'" with: "'Does not hold' and 'holds (descriptive …)' read 'the repeat hinge falls short'; 'cannot be asked here' (UNTESTED) gives no spec decision."
2. **REG3 must be (nearly) complete for 2b to decide the spec.** In `report()`, after `rd = s89h.readings(...)` (s90:109), add:
   ```python
   if len(R["R"]) < 18:
       rd["question"] = s89h.Q[3]
       rd["first"] += f"; UNTESTED for the spec: REG3 completed on {len(R['R'])}/20 seeds (< 18)"
   ```
   Docstring s90:35: "a missing REG3 seed does not void the names; fewer than 18 completed REG3 seeds void the spec decision (2b's power: KH only must reach 5)".
3. **Print REPEAT against FREQ, descriptively.** In `report()`, add `b, c, p, n = hp.paired(R["RP"], R["FQ"], rb)` and print "REPEAT vs S87's FREQ (descriptive; on lite the repeat set = the frequency set + 2.63 chance-repeated VAL positions per sequence, same token-id grouping): REPEAT only {b}, FREQ only {c}, p(REPEAT >) {p:.3g}, p(FREQ >) {hp.paired(R['FQ'], R['RP'], rb)[2]:.3g}". Add the sentence in quotes to the docstring at s90:28.
4. **The pairing claim, as run.** Change s90:13-14 to "This CPU reproduces S87's records bit for bit **through 2400** (curve and stats; `s87.trunc`) on REG3_KEYHINGE|2700, REG3_FREQHINGE|2700 and BASELINE|2700; the full records are not compared." Optional and cheap (about 15 min on 4 workers), after the 40 runs: rerun `lite.REG3_KEYHINGE` on 2700-2719 as a dated descriptive control, as the prompt's other branch (prompts:1284-1285) names. Add it to `ARMS` (the SHA is unchanged, since the child already has the arm) and print its count and full-record equality with S87's (`curve`, `transition`, `stopped_at`, `end`) beside KH (S87). It does not change the reading.
5. **The gradient sentence.** At s90:59, replace "its term's gradient reaches the gate only" with "its term's gradient reaches the register gate's parameters and the token embedding (`embed.weight`), not the encoder, decoder or lm_head (CHECK row)".
6. **The operating characteristics.** At s90:51-53 add: "at S87's FREQ rate on these seeds (0.85) 'replaces' fires with probability 0.18; with KH 20/20 and BASELINE 0/20 every decisive test is marginal: the spec's arm is the repeat hinge iff REPEAT ≥ 19/20 and REG3 ≤ 15/20."
7. **Date the design.** Add to the docstring: "The readings' application to lite, the spec bar (s90:48-50) and the pairing branch were chosen on 22 October 05:03 UTC, after S89H's records (03:07) and report 9 (03:12); the readings function is S89H's, fixed at d5d48ee (21 October 22:01), before any S89H record. No S90 record existed. The relayed readings of 21 October did not reach this session." The report and README should say the same.
8. **Before S91 (not during S90: it changes the SHA).** In `explore_g_common.py:164`, change `dict(arm=arm, seed=s)` to `dict(arm=arm, seed=s, iters=screen.ARMS[arm]["iters"])`. Add a CHECK to every screen: "each queued payload carries its arm's iters", asserted on the job list before `run_screen` submits, plus a post-run line "records: iters == ARMS[arm]['iters'] on n/n". S91's 48000 on ALLPOOL (prompts:1313) depends on it. The README's S89H section should get its 24000 correction (`check_G_report9.md` fixes) in the same commit.

## Verdict

**Fit to run. Fix 1 and 2 first (parent only, before `report()`). Too late for a pre-run review**: the learned arms started at 05:07, 4 min after the push, and 10 of 40 runs are in.
- Nothing in the run code needs to change for S90.
- The budget is right by coincidence (default = intended), and fix 8 must land before S91.
- On the records so far (REPEAT 5/6, REG3 0/4), S90 reads "the repeat hinge falls short" if one more REPEAT seed fails.

**Correction (22 October, `check_G_report10.md`).** §1g and fix 8 said S91 "would again run 24000" and that its intended 48000 depends on fix 8. Both are wrong for S91 as G built it. S91's child defaults the budget by memory (`explore_g_s91_child.py:229`: 48000 for Hebbian, 24000 for delta), and every S91 record ran its intended budget. The harness defect at `explore_g_common.py:164` is described correctly and is still present. It stays latent for any run whose budget differs from its child's default.
