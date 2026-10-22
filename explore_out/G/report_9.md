# Session G — report 9 (S89H: S89's hinges on Hebbian memory + conv; POOL 24000, ALLPOOL 48000). EXPLORATORY, not a result

Branch `claude/explore-G`; Intel Xeon @ 2.10GHz, torch 2.14.0+cu130, 1 thread per run; code SHA 0f1f20cb2fa8. F's
RandHeaderTask-pool and -allpool (b0c6332), Hebbian memory (decay 0.95) + test_short_conv.LAYER (conv 'layer', K = 4, on
each layer's memory input, identity at init, rest group), k = 2, SLOW, the REG3 model; S89's hinges unchanged. Seeds as
S89: 2840-2841 (perfect gate), 2842-2861 (arms), paired across arms, tasks and with S89's delta arms. Spec, readings (with
V's two caveats from screen_review_s89_G_v2.md) and CHECKs committed at d5d48ee before any run. 204/204 runs ok, none
retried; 19.2 CPU-hours. One worker restart (after 116 learned runs); the resume re-ran all 139 CHECKs (ok) under the same
code SHA and continued from the cached runs.

## Counts (ROUTED-BOUND; Wilson 95%; band). POOL: pooled η²; ALLPOOL: within-sequence η² and VAL cos (the other form gives the same counts)

| arm (Hebbian + conv) | POOL (24000) | ALLPOOL (48000) | S89 delta POOL / ALLPOOL (same seeds) | lite, S87 Hebbian, no conv (2700-2719) |
|---|---|---|---|---|
| PERFECT_H (validity) | 2/2 (6000, 7200) | 2/2 (9600, 6000) | ORACLE_D 2/2 / 2/2 | – |
| BASELINE_H | **0/20** NEVER (4 bound unrouted) | **0/20** NEVER | 0/20 / 0/20 | 0/20 |
| REG3_H | **9/20** [0.26, 0.66] MINORITY | **0/20** NEVER | 16/20 / 0/20 | (S82: 2/20) |
| REG3_KEYHINGE_H | **18/20** [0.70, 0.97] RELIABLE | **0/20** NEVER | 20/20 / 0/20 | 20/20 |
| REG3_FREQHINGE_H (POOL: CTX hinge; ALLPOOL: value hinge) | **18/20** [0.70, 0.97] RELIABLE | **0/20** NEVER | 7/20 / 0/20 | 17/20 |
| REG3_REPEATHINGE_H | **19/20** [0.76, 0.99] RELIABLE | **0/20** NEVER | 20/20 / 0/20 | – |

Transitions (median, range), POOL: KH 6000 (4800-9600), FREQ 6600 (6000-20400), REPEAT 6000 (4800-19200), REG3 6000
(4800-9600). Unbound, POOL: REG3 OTHER 11; BASELINE STREAM-PARTIAL 10, OTHER 5. ALLPOOL: MERGED 11-16 per hinge arm;
BASELINE and REG3 OTHER 17-19.

## Readings (pre-fixed; explore_g_s89h.readings)

| task | first | 2b (V's caveat 1: KH vs REG3_H) | REG3_FREQHINGE_H | REG3_REPEATHINGE_H | THE QUESTION |
|---|---|---|---|---|---|
| POOL | KH discovered (18 / 0, p 3.8e-6) | **the hinge is needed over the register** (KH only 9, REG3 only 0, p 0.002) | **"survives"** (KH only 2, FREQ only 2; vs REG3 9 / 0) | **"replaces the key hinge"** (KH only 0, REPEAT only 1; vs REG3 10 / 0, p 0.001) | **"holds, and the hinge is needed here"** |
| ALLPOOL | **"the labelled hinge itself fails here"** (0/20 = 0/20) | – | – | – | **"cannot be asked here"** |

- "clamps": does not apply on POOL (arm only 18 / 19 vs BASELINE only 0).
- V's caveat 2 (the POOL frequency hinge penalises the stream): printed with the FREQ name. It did not bite here. REG3_H
  does not beat REG3_FREQHINGE_H (0 / 9), so "the term hurts the register" does not apply. On delta (S89) it did: 1 / 10,
  p 0.006, and the name there was "needs frequent keys".
- S89's delta records read through the same function, after the fact (descriptive; S89's readings at db14e9b had no 2b).
  POOL: KH vs REG3_D 4 / 0, p 0.0625, so "the register does not need the hinge here", and REPEAT's "replaces" is
  descriptive there ("holds (descriptive …)"). ALLPOOL: "fails here".
- ALLPOOL caveats, printed with the reading:
  - **The routing cap:** a fixed CTX-type map separates at most 0.533 of sequences, so this null does not separate "keys
    are not frequent" from "contexts have no fixed identity".
  - **The type-level perfect gate** ran on Hebbian without the conv only (F's O_TYPE_HEBB_ALLPOOL, 0/2).

## Across memories and tasks (descriptive)

- **Hebbian + conv vs S89's delta, POOL, same seeds.** The memory and the conv differ together.

  | arm | Hebbian + conv | delta | Hebbian only / delta only | p |
  |---|---|---|---|---|
  | REG3 | 9/20 | 16/20 | 0 / 7 | 0.008 (delta higher) |
  | KH | 18/20 | 20/20 | 0 / 2 | 0.25 |
  | REPEAT | 19/20 | 20/20 | 0 / 1 | 0.5 |
  | FREQ | 18/20 | 7/20 | 11 / 0 | 5e-4 (Hebbian higher) |
  | BASELINE | 0/20 | 0/20 | – | – |

- **POOL → ALLPOOL, same seeds, Hebbian.** FREQ is not compared, and the budgets differ (24000 / 48000).

  | arm | POOL | ALLPOOL | p |
  |---|---|---|---|
  | REG3 | 9 | 0 | 0.002 |
  | KH | 18 | 0 | 3.8e-6 |
  | REPEAT | 19 | 0 | 1.9e-6 |
  | BASELINE | 0 | 0 | – |

  These read as a change of task and of hinge pressure together (V's §6 numbers in the docstring).

## What happened (exploratory)

- **The repeat hinge's POOL result holds on Hebbian memory, and here it is not inherited from the register.**
  - On Hebbian + conv, REG3 alone is 9/20, against 16/20 on delta. Its failures end with β^R decaying from 0.10 to 0.03:
    the register shuts.
  - The key hinge (18/20) and the repeat hinge (19/20) both beat REG3 by 9-10 seed pairs (p 0.001-0.002). So on Hebbian
    the hinge is needed over the register, which delta could not show (4 vs 0).
  - The repeat hinge's runs look like the key hinge's:
    - the commitment is held to ~0.08 at 1200 and released by 2400 (0.92);
    - β^R at CTX reaches 0.28 at 2400 and about 0.6 at 4800, then 0.79 at the end;
    - layer 1 routes at KEY and VAL (0.99 / 0.01).
- **The frequency hinge survives on Hebbian POOL, where on delta it hurt.** The same CTX-only term, the same seeds:
  18/20 here against 7/20 on delta. Its commitment stays low through 2400 (0.17) and commits after (0.96 at 3600). β^R at
  CTX rises later than under KH and REPEAT (0.41 at 4800). V's warning that it penalises the stream did not translate
  into harm on this memory. Why the memories differ is not tested here.
- **ALLPOOL: still no arm routes, at twice the budget, on Hebbian too.** The hinge arms end at the type-level cap:
  - within-sequence η² 0.41-0.58 (median), and 0.33-0.57 of sequences routed in 11-17 runs per arm;
  - MERGED failures (one channel takes both streams for the sequences the map cannot separate);
  - BASELINE and REG3 commit early (0.93-0.95 at 2400) to OTHER plateaus.

  As on delta, no run learned the relative routing the perfect gate is handed.
- **POOL BASELINE binds 4/20 unrouted** (margin ~0.46), as on delta (14/20 there).

## CHECKs (139, asserted before each segment, all ok)

S89's full set (93 rows: the vendored module, token ids only, the task-aware code, the perfect gate, S87's 7 records
through S89's run()), plus:
- **The conv model:**
  - The conv is the only extra parameter, with the shared ones equal to the no-conv model's.
  - At initialisation (identity conv) logits and gate equal the no-conv model's bit for bit, for every arm on all three
    tasks.
  - With the conv off, REG3_H and BASELINE_H equal explore_g_rhl_child's REG3 and BASELINE.
  - The conv insertion equals MultiBDH.forward_conv bit for bit (REG3_H with A_r = A_R = 0 and random conv weights =
    BASELINE_H), and it is strictly causal.
- **Ties to records:**
  - F's O_HEBB_CONV_POOL|2800 and O_HEBB_CONV_ALLPOOL|2800 (2.10GHz; vendored in `explore_out/G/refs/` from
    claude/explore-F 129215e) reproduce bit for bit through 2400 through PERFECT_H.
  - S87's REG3_KEYHINGE, REG3_FREQHINGE and BASELINE at 2700 reproduce bit for bit through this run() with the conv off.
- **Per task:**
  - each hinge off = REG3_H through 2400 bit for bit;
  - every arm runs end to end;
  - each term's gradient reaches the gate only (not the stack, not the conv);
  - the lrs (the conv in the rest group; the perfect gate's gate group empty).
- **The readings:** hand cases for 2b, the hurt rule and the question; the four names exclusive and exhaustive (n = 10,
  20); inv_view's within-sequence DISCOVERED.

## For the main line

- **On Hebbian memory with the conv, the repeat hinge replaces the key hinge on POOL (19/20 vs 18/20), and both are
  needed over the register (9/20).** This is the result S89 could not give on delta, where the register alone already
  bound 16/20. It is the strongest label-free result so far, on a task where keys are not frequent.
- **The configuration differs from the spec draft's.** The draft is Hebbian on lite without a conv. A main-line test of the
  repeat hinge would need its own spec, with the conv: F's pilots show the Hebbian perfect gate needs it on pooled keys.
- **ALLPOOL stays unanswered:** 0/20 for every arm on both memories, with the type-level cap visible in the failures. A
  gate that compares contexts within the sequence is the open question, not the hinge.
