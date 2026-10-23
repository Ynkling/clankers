# Session G — report 11 (S91b: the register gate with a channel per context type, k = 16). EXPLORATORY, not a result

Branch `claude/explore-G`; Intel Xeon @ 2.10GHz, torch 2.14.0+cu130, 1 thread per run; code SHA a0134e14a262. F's
RandHeaderTask-ctxpool and -allpool (`explore_f_tasks.py` vendored from claude/explore-F 9017b82, blob cfe3fe2,
append-only over b0c6332, asserted). k = 16 channels, the register gate shared by the three layers (REG3), SLOW, 24000
updates. Hebbian + conv 'layer' as S89H and delta (β 1, L2) as S89. Seeds 2992-2993 (perfect gates) and 2994-3003 (10,
paired across arms, tasks and memories).

Spec, readings and CHECKs were committed at b981c3f before any run. One CHECK row was superseded at the first CHECK pass,
before any run (9949e49): S89's own blob row asserts the older e24d5e1 copy, and S91b's own CHECK asserts the new blob and
its e24d5e1 prefix.

136/136 runs ok, none retried. Every record carries iters 24000, the arm's budget (136/136, printed). A worker restart came
after 84 of the 120 learned runs; the resume re-ran all 147 CHECKs (ok) under the same SHA and continued from the cache.
This screen replaces S91's remaining Hebbian half, which was not rerun.

## Counts (ROUTED-BOUND in its k-general within-sequence form: BOUND, end margin ≥ 0.9, end share of sequences separated > 0.9)

| arm (k = 16) | -ctxpool Hebbian + conv | -ctxpool delta | -allpool Hebbian + conv | -allpool delta | k = 2 counterpart (-allpool, S89H / S89) |
|---|---|---|---|---|---|
| PIDX (perfect gate, the index; 2 seeds) | 2/2 (3600, 2400) | 2/2 (1200, 1200) | 2/2 (7200, 7200) | 2/2 (3600, 3600) | – |
| PTYPE (perfect gate, a channel per type; 2 seeds) | 2/2, TYPE-ROUTED 2 | 2/2, TYPE-ROUTED 2 | 2/2, TYPE-ROUTED 2 | 2/2, TYPE-ROUTED 2 | – |
| REG3 | **0/10** [0.00, 0.28] (1 BOUND) | **0/10** | 0/10 | 0/10 | 0/20 / 0/20 |
| REG3_KEYHINGE | **0/10** | **0/10** | 0/10 (2 BOUND) | 0/10 | 0/20 / 0/20 |
| REG3_REPEATHINGE | **0/10** (1 BOUND) | **0/10** | 0/10 | 0/10 | 0/20 / 0/20 |

No learned run is TYPE-ROUTED: none has a gate that is a fixed function of the context type and separates every
sequence's contexts.

## Readings (pre-fixed; explore_g_s91b.readings)

| memory | the perfect gates on -ctxpool | the learned arms on -ctxpool | READING |
|---|---|---|---|
| Hebbian + conv | PIDX 2/2, PTYPE 2/2 | every arm 0/10 (≤ 3/10) | **"the channels do not suffice"** |
| delta | PIDX 2/2, PTYPE 2/2 | every arm 0/10 (≤ 3/10) | **"the channels do not suffice"** |

- "The gate routes by type when it has the channels" needed some arm ≥ 7/10 and discovered against its k = 2 counterpart's
  0/20. No arm passes (Fisher p = 1 for each).
- **-allpool (descriptive, against -ctxpool):** every learned arm is 0/10 on both memories. Against -ctxpool on the same
  seeds the comparison is 0 / 0 everywhere.

## What happened (exploratory)

- **The two perfect gates are one ceiling here, by construction.** Within a sequence, "same stream" and "same context
  type" are the same relation. So PIDX (two channels by first appearance) and PTYPE (a channel per type) give the memory
  the same within-sequence mask, and they trained identically seed for seed (the same transitions and accuracies). PTYPE
  shows that a type-keyed gate at k = 16 routes every sequence: TYPE-ROUTED 2/2, type map 1.00, 15.8 effective channels.
- **The learned gates do not spread over the channels.**

  | arm, Hebbian + conv on -ctxpool | type map (η² of the KEY gate by context type) | effective channels | sequences separated (median, max) | end margin | held-out | unbound class |
  |---|---|---|---|---|---|---|
  | REG3_KEYHINGE | **0.97** | ~6 | 0.20, 0.84 | 0.67 | 0.89 | STREAM-PARTIAL 10/10 |
  | REG3_REPEATHINGE | 0.53 | ~7 | 0.00, 0.88 | 0.50 | 0.89 | STREAM-PARTIAL 8 |

  - The key hinge's gate became a fixed function of the context type, which is what the channels allow. But it put the 16
    types on about 6 channels, too few and not confidently enough to separate most sequences (the ≥ 0.9 criterion).
  - The register alone (REG3) commits elsewhere: type map 0.02, separated 0.00, POSITION 4 and OTHER 2 among its classes.
  - On delta every arm stays near 0 on both maps, with STREAM-PARTIAL / OTHER / KEY failures. Held-out accuracy is 0.50-0.77
    on -ctxpool, where F's single delta channel binds 10/10 without routing.
- **The channel cap is not what limits these arms.** At k = 16 a fixed type map could route every sequence (cap 1), and
  the ceiling shows the memory would bind with it. The arms fail before using the channels: they spread over 6-14
  effective channels but do not commit by type at 0.9 (Hebbian), or do not key on type at all (delta).
- **A budget note.** The Hebbian key hinge was still at 0.89 held-out and STREAM-PARTIAL at 24000. Whether a longer budget
  would carry it to separation is not tested here.

## CHECKs (147, all ok; asserted before each segment)

- **The vendored module** (blob cfe3fe2; its first 49,923 bytes are b0c6332's e24d5e1); F's check_ctx_task (-ctxpool) and
  check_inv_stats_k (both tasks); the cap at k = 16 is 1, at k = 2 64/120.
- **At k = 2 the builder is S89H's and S89's.**
  - REG3_H, REG3_REPEATHINGE_H, REG3_D, PERFECT_H and ORACLE_D: parameters and logits bit for bit.
  - S89H's and S89's allpool.REG3_REPEATHINGE_<M>|2842 reproduce bit for bit through 2400 through S91b's run().
- **At k = 16:**
  - the shapes;
  - both perfect gates, untrained, read ROUTED_k (margin 1, separated 1), with PTYPE's type map 1.00 and PIDX's 0.04-0.05;
  - the k-general outcome agrees with the k = 2 within-sequence η² form on synthetic gates;
  - the key term's positions on -ctxpool;
  - each hinge off equals REG3 through 2400 bit for bit, on every task and memory;
  - every arm runs end to end.
- **The harness:** each queued payload carries its arm's iters, and all 136 records agree. This is the fix to the defect
  V's check_G_report9.md found in S89H.
- S89's CHECK set (its blob row superseded, as above).
- S91b's readings() hand cases, and the k = 2 counterparts read 0/20 on all six arm × memory pairs.
- Seeds 2992-3019 appear in no record on any of the 13 branch heads.

## For the main line

- **Giving the register gate a channel per context type does not, by itself, make it route by type** at 24000, on either
  memory: 0/10 for every arm on -ctxpool, where both perfect gates bind 2/2 and a type-keyed perfect gate routes every
  sequence.
- **The closest arm is the Hebbian key hinge.** Its gate keys on context type (type map 0.97) but folds the 16 types onto
  about 6 channels without committing at 0.9 (separated 0.20, held-out 0.89, all runs STREAM-PARTIAL). So the next question
  is how the gate is pushed to use its channels, for example a term on the gate's spread over channels, or a longer budget,
  not whether the channels exist.
- **The main line's regime (k = 16 at eight streams)** has the channels; this screen says the register gate with the key
  hinge does not use them unaided on a two-document task with pooled contexts.
