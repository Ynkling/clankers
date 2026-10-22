# Session G — report 10 (round 7: S90, the repeat hinge on lite Hebbian; S91, gates that compare on ALLPOOL). EXPLORATORY, not a result

Branch `claude/explore-G`; Intel Xeon @ 2.10GHz, torch 2.14.0+cu130, 1 thread per run. S90 committed at b57c186 and S91
at 60b1932, each before any of its runs. S90's design (the readings applied to lite, the spec bar, the pairing branch)
was chosen on 22 October at 05:03 UTC, after S89H's records and report 9; its readings function is S89H's, fixed before
any S89H record. The relayed readings of 21 October did not reach this session. The register spec's revision 2 (6fc55b1) went between them.

## S90: counts (ROUTED-BOUND; lite, Hebbian, k = 2, no conv, 24000, SLOW, REG3; seeds 2700-2719)

S90 is paired with S87's records: this CPU reproduces them bit for bit through 2400 (curve and statistics), and a
descriptive rerun of REG3_KEYHINGE on all 20 seeds (added 22 October on V's screen_review_s90_G.md fix 4) equals S87's
full records (curve, transition, stop, end statistics) on 20/20. 60/60 new runs ok, none retried.

| arm | ROUTED-BOUND (Wilson 95%; band) | vs REG3_KEYHINGE | vs REG3 | name (pre-fixed) |
|---|---|---|---|---|
| BASELINE (S87) | 0/20 [0.00, 0.16] NEVER | – | – | – |
| REG3 (new) | **4/20** [0.08, 0.42] MINORITY | KH only 16, REG3 only 0 (p 1.5e-5) | – | – |
| REG3_KEYHINGE (S87) | 20/20 [0.84, 1.00] RELIABLE | – | 16 / 0 | – |
| REG3_FREQHINGE (S87) | 17/20 [0.64, 0.95] MAJORITY | 3 / 0 (p 0.125) | 13 / 0 | "is weaker than the key hinge" |
| **REG3_REPEATHINGE (new)** | **17/20** [0.64, 0.95] MAJORITY | **3 / 0 (p 0.125)** | 13 / 0 (p 1.2e-4) | **"is weaker than the key hinge"** |

**Readings:**
- **First reading:** the labelled hinge is discovered (20 / 0).
- **2b:** the hinge is needed over the register (16 / 0).
- **The question:** "does not hold".
- **For the spec:** "the repeat hinge falls short". The spec carries the key hinge only (revision 2, 6fc55b1).
- **Corrections to the parent (22 October, after all 40 records, on V's screen_review_s90_G.md):** UNTESTED now gives no
  spec decision (the b57c186 rule read it as "falls short"), and fewer than 18 completed REG3 seeds void it. On the
  complete store neither changes anything: every run completed, and the reading and the decision are as above.
- **Repeat against S87's frequency hinge** (descriptive): on lite the repeat set is the frequency set plus 2.63
  chance-repeated value positions per sequence, with the same token-id grouping; REPEAT only 1, FREQ only 1 (p 0.75).
  So S90 measured the effect of those value positions on an arm S87 had at 17/20, and found none.

**What happened (exploratory):**
- The repeat hinge's runs that bind look like the key hinge's:
  - the commitment is held to 0.08 at 1200 and released by 2400 (0.94);
  - β^R at CTX reaches 0.30 at 2400;
  - layer 1 routes at KEY and VAL (1.0 / 0.0).
- **The three failures** (2700, 2713, 2717) key-split, as do FREQ's (2700, 2709, 2713). FREQ against REPEAT is 1 / 1, so the
  two hinges are statistically the same arm here.
- **REG3 alone** binds 4/20 (S82: 2/20 on 2100-2119). Its failures key-split with β^R flat at about 0.23.
- **What "falls short" means here.** At a 20/20 ceiling, "within one pair" fires only 39% of the time for an arm whose
  true rate is 0.9 (V's §2.10). So "falls short" is the pre-fixed rule's output, not evidence of a rate below 0.9.
- **Comparison with POOL.** S89H, on POOL with Hebbian + conv, had the repeat hinge 19/20 against the key hinge's 18/20.
  On lite without a conv it is three pairs behind. This screen does not say which difference matters: the task, the conv,
  or chance at these rates.

**CHECKs (81, all ok):**
- S87's REG3_KEYHINGE, REG3_FREQHINGE and BASELINE at 2700 reproduce bit for bit through S89's run(), and ORACLE_ALL|2700
  through S87's own path. These are the pairing condition.
- **The repeat set on lite:** 16.63 positions = 4 CTX + 8 KEY + 2.63 VAL + 2 QUERY (the query's CTX and KEY). All 8
  key-term positions are inside it, and its grouping there equals the key term's one to one.
- REG3_REPEATHINGE with its switch off equals REG3|2700 through 2400 bit for bit.
- The term's gradient reaches the register gate's parameters and the token embedding (embed.weight), not the encoder,
  decoder or lm_head.
- S89's static CHECKs pass, and readings() passes its hand cases.
- A first start of S90 was launched by mistake before its commit and was stopped during its CHECK phase. No run had
  started and no store had been written; the log was deleted, and S90 was committed and started again.

## S91: counts (ALLPOOL; within-sequence ROUTED-BOUND)

| arm | Hebbian + conv (2962-2981, 48000) | delta (2982-2991, 24000; descriptive) | sequences routed (delta): median / max | type-level map (delta) |
|---|---|---|---|---|
| perfect gate (validity, 2960-2961) | **PERFECT_H 1/2** (2960 @16800; 2961 unbound, 0.9375 at 48000, still rising) | ORACLE_D 2/2 (3600, 3600) | – | – |
| REG3_REPEATHINGE (control) | not run | 0/10 (KEY 7, MERGED 2) | 0.00 / 0.53 | 2/10 |
| MATCHGATE | not run | 0/10 (2 BOUND unrouted; KEY 8) | 0.00 / 0.00 | 0/10 |
| MATCHGATE_NOHINGE | not run | 0/10 (1 BOUND unrouted; KEY 8) | 0.00 / 0.00 | 0/10 |
| NOVELTY | not run | 0/10 (2 BOUND unrouted; KEY 4, MERGED 2) | 0.00 / 0.54 | 3/10 |

84 runs ok (4 validity + 40 delta), none retried; 140 min. The 80 Hebbian learned runs were not queued.

**Budgets, checked against the records (22 October, after V's check_G_report9.md found S89H's ALLPOOL at 24000):** the
two PERFECT_H records carry iters 48000 (S91's child defaults to 48000 on Hebbian, so the parent's ITERS did not need
to reach it), and the 42 delta records carry 24000, as fixed. S91's numbers stand as printed.

**Readings (pre-fixed):**
- **Hebbian + conv: UNTESTED.** The perfect gate did not bind 2/2 within 48000. It bound 2960 at 16800. On 2961 it was at
  0.90-0.94 for the last 8 evaluations and ended at 0.9375. This is not a negative. In S89H the same configuration bound
  2/2 at 6000 and 9600 on seeds 2840-2841, and F's pilots bound 2/2 at 6000 and 28800.
- **Delta (descriptive, n = 10): "no device here".** No new arm beats the control (all 0 / 0).
  - No run crosses the 0.533 line except one NOVELTY run at 0.54 of 256 probe sequences, which is within sampling of the
    cap.
  - Type-level maps (gate at KEY keyed on the block's CTX type, η² > 0.5): control 2/10, NOVELTY 3/10, MATCHGATE 0/10.

**What happened (exploratory, delta):**
- **The match term trains but does not route.**
  - |a_M| grows from 0 to about 0.21 at the median (max 0.93). The gate uses the scalar.
  - MATCHGATE's runs key-split (η² by key at KEY high; KEY 8 of 10), and none settles at a type-level map. The match
    removes the type-level solution without supplying the relative one.
  - The hinge fires about twice as often under MATCHGATE (median 512 against the control's 232). The match pushes the
    gate's key positions apart, and the repeat hinge pushes back.
- **NOVELTY** (|w_N| 0.25 at the median) leaves the control's picture: key splits and some type-level maps.
- **The bound runs bind late** (18000-22800) and are unrouted (margin ~0, key split). As in S89 and S89H, delta memory
  binds some ALLPOOL runs without routing.

**CHECKs (122, all ok):**
- **The scalars:**
  - the match agrees with a hand computation, m_0 = 0, and it reads the embedding, the register and W_R only;
  - novelty agrees with a hand computation from token ids, is causal, and refuses a float tensor;
  - both arms are strictly causal with non-zero weights.
- **Equivalence with the controls:**
  - at initialisation each new arm equals its control bit for bit, with exactly one extra parameter;
  - with its scalar zeroed, MATCHGATE and NOVELTY equal REG3_REPEATHINGE, and MATCHGATE_NOHINGE equals REG3, bit for bit
    through 2400 on both memories;
  - S89H's allpool.REG3_REPEATHINGE_H|2842 and S89's allpool.REG3_REPEATHINGE_D|2842 reproduce bit for bit through S91's
    run().
- **Training:** the new weights receive a gradient and sit in the gate group.
- **Perfect gate and type map:**
  - the untrained perfect gate reads ROUTED* on both memories;
  - type_map reads 0.02 for the relative perfect gate and 1.00 for the balanced type-level gate.
- **Inherited:** S89's and S89H's CHECK sets, including F's conv record.
- **Seeds:** 2960-2991 appear in no record on any of the 13 branch heads (checked before the runs).

## For the main line

- **The register spec (revision 2, 6fc55b1) carries the key hinge only.** S90 put the repeat hinge in the spec's own
  setting and it fell short by the pre-fixed rule (17/20 vs 20/20). The spec says why, and dates the decision after S89,
  S89H and S90.
- **The comparing gate question is open on Hebbian memory.** The screen's validity failed there, by half an evaluation's
  margin. Rerunning the Hebbian half needs a decision that is not mine, for example F's precedent of a longer budget for
  the perfect gate before any learned arm.
- **On delta, neither comparing device routes ALLPOOL** at 10 seeds (descriptive). The match scalar as specified (a cosine
  with the register's last content) does not give the gate a usable "same document" signal under this recipe.
