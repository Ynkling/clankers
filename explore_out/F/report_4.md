# Session F — report 4 (S77 probe). EXPLORATORY, not a result

Branch `claude/explore-F`, code SHA 623d477f61ad, Intel Xeon @ 2.10GHz, torch 2.14.0+cu130, 1 thread per run. 30/30 runs ok.
**Every rerun's curve equals its record bit for bit** (asserted in each run): S53 SINGLE_DELTA_H (300-309), S59 SINGLE_GDN
and S69 S69_GDN (350-359). Probe: Session G's S55 logistic regression, copied unchanged from claude/explore-G at 3932db7.
Its synthetic check gives 0.9975 on an informative feature (G 0.998) and 0.474 on a random one. Held-out on 400 of 2000
probe sequences; chance 0.5. Full table: `probe_tables.md`.

## Counts and probe accuracy (medians; K / V positions)

| arm | BOUND | entering L1 | entering L2 | entering L3 | final layer | L3 at init |
|---|---|---|---|---|---|---|
| S53_DELTA (one delta channel, header P=4) | 8/10 [0.49, 0.94] | 0.50 / 0.50 | 0.64 / 0.57 | 0.67 / 0.60 | 0.68 / 0.67 | 0.75 / 0.63 |
| S59_GDN (one delta channel + GDN decay, header P=8) | 8/10 [0.49, 0.94] | 0.50 / 0.49 | 0.58 / 0.55 | 0.57 / 0.56 | 0.57 / 0.60 | 0.62 / 0.56 |
| S69_GDN (one Hebbian channel + GDN decay, P=8) | 0/10 [0.00, 0.28] | 0.50 / 0.49 | 0.53 / 0.50 | 0.52 / 0.51 | 0.53 / 0.51 | 0.69 / 0.64 |

| split by outcome (L3 K / V, medians) | bound | unbound |
|---|---|---|
| S53_DELTA | 0.67 / 0.59 (n 8) | **0.66 / 0.96** (n 2: seeds 302, 303) |
| S59_GDN | 0.57 / 0.56 (n 8; but 351: 0.88 / 1.00, 358: 1.00 / 0.98) | 0.59 / 0.63 (n 2) |
| S69_GDN | — | 0.52 / 0.51 (n 10) |

## Readings (fixed before any run) and which applies

- "The delta stack tags its keys by stream" (L3 ≥ 0.9 at K and V, with L1 ≤ 0.6, on ≥ 8 bound delta runs): **does not
  apply**. 1 of 16 (S59 358; 351 misses at K with 0.88).
- "The tag is where binding is" (bound − unbound median L3 ≥ 0.2): **does not apply**. 0.62 − 0.52 = 0.09.
- So the outcome is **descriptive**. L1 is at chance everywhere (0.49-0.50), as it must be: the token alone carries no
  stream. Most bound delta runs carry little linearly decodable stream information in the residual at any layer
  (0.53-0.73). The two S53 runs that do carry it at VAL (0.92-1.00) are the two that did not bind.

## What this changes (post hoc)

S53's one-seed diagnostic (seed 307: cosine 0.16 between the normalised layer-3 keys of a key's two writes) does not
generalise into a linearly readable stream tag in the residual stream. On seed 307 here the probe gives L3 0.65 / 0.67.
The keys are 256-d ReLU codes of the residual (x_sparse = ReLU(LN(r) E)); two writes can be near-orthogonal in that code
without the stream being linearly separable in r. What separates them may be position or context features specific to
each write rather than a stream direction. The delta memory's binding without a partition therefore does not rest on a
stream tag a gate could read. The Hebbian channel (S69) has none at any layer (0.50-0.53). Training lowers the layer-3
stream content in every arm (at init 0.62-0.75 at K, through the decay-weighted mixing of the initial attention).

## CHECKs

explore_f_delta_checks 26/26. The child's checks:
- G's synthetic probe check passes.
- G's residual loop reproduces each memory's logits bit for bit.
- The probe labels are the block's CTX, with a 1600/400 split disjoint by sequence.
- Each arm's record reproduces through 2400.
- All 30 full reruns are equal to their records.

## Runtime

7.37 h of run time, about 2.1 h wall on 4 workers (03:58-06:05 UTC, 10 October). S76 started after it.
