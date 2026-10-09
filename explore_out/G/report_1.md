# Session G — report 1 (S55 probe). EXPLORATORY, not a result

Branch `claude/explore-G`; Intel Xeon @ 2.10GHz, torch 2.14.0+cu130 (Python 3.13), 1 thread per run. Code SHA (screen)
0c559dc511fe; spec and readings committed at e4d8107 before any run. 22/22 runs ok, none retried. Runtime 47.8 min on
4 workers (plus 1.5 min of CHECKs).

## Counts and probe accuracies (held-out, median over runs; K / V)

| arm | runs | BOUND | BOUND ROUTED | failures | emb | L1 | **L2** | **L3** | **final** |
|---|---|---|---|---|---|---|---|---|---|
| L3S_P8 (LOCAL3+SLOW, header P=8, 360-369) | 10 | 0/10 [0.00, 0.28] NEVER | 0/10 [0.00, 0.28] NEVER | OTHER 5, POSITION 3, KEY 2 | 0.50 / 0.49 | 0.50 / 0.49 | **0.55 / 0.58** | **0.55 / 0.56** | **0.57 / 0.56** |
| FAR_L3_P4 (batch 2 far_L3 reproduced, 160-169) | 10 | 2/10 [0.06, 0.51] MINORITY | 0/10 [0.00, 0.28] NEVER | KEY 3, OTHER 3, POSITION 2 | 0.50 / 0.50 | 0.50 / 0.50 | 0.63 / 0.58 | 0.68 / 0.61 | 0.69 / 0.65 |
| ORACLE_P8 (perfect gate, 360-361) | 2 | 1/2 [0.09, 0.91] (360 at 24000) | 1/2 | STREAM-PARTIAL 1 (361, acc 0.84) | 0.50 / 0.49 | 0.50 / 0.49 | 0.54 / 0.54 | 0.53 / 0.54 | 0.55 / 0.56 |
| *same seeds at initialisation* | | | | | | | L3S 0.70 / 0.66, FAR 0.81 / 0.69 | L3S 0.70 / 0.66, FAR 0.80 / 0.71 | L3S 0.69 / 0.66 |

Runs meeting "present" (≥ 0.9 at K and V): 0/10 at L2, L3 and final in L3S_P8 and in FAR_L3_P4, 0/2 in the oracle.
Per-run Wilson intervals on the 6400 (P=8) / 3200 (P=4) held-out positions are ±0.012-0.017 (positions of one sequence are
correlated, so these understate the uncertainty); full per-run tables in `s55.log`. Transitions: FAR_L3_P4 7200, 10800
(both bound by key splits, not routed, as in batch 2); oracle 24000.

## Pre-fixed readings and which applies

- **L3S_P8 (the arm that sets S56's design): "absent"**. The median is ≤ 0.6 at K and at V at L2, L3 and final, so the fix
  is upstream and **S56's register is mandatory**. S56's pre-fixed read-path rule therefore gives res_layers = (2, 3), the
  specification. I stored that decision before any S56 run.
- FAR_L3_P4: "partial at L3 and final" (0.68 / 0.61, 0.69 / 0.65). Two runs come close at final K: 165 (0.93 / 0.85, a
  KEY split at acc 0.74) and 167 (0.90 / 0.75).
- ORACLE_P8: "absent". With the perfect gate doing the routing, the stack carries no stream at all.
- Validity: ORACLE_P8 bound 1/2 (≥ 1 required): VALID, but at the edge. Seed 360 bound exactly at 24000 and 361 ended at
  0.84. F's S53 saw 14400-24000.

## Anything unexpected

- **Training removes the cue.** The untrained stack decodes the stream above chance at L2 and L3 (0.66-0.81, linear
  attention with decay 0.95 sums the headers). After 24000 updates of LOCAL3+SLOW the same layers are at 0.55-0.58. The
  window gate's stack does not learn to carry the header forward; it forgets what it had at init. By index within the block,
  the first pair (next to the header) is the best decoded (0.6-0.7) and later pairs fall to chance.
- **F's prediction (layer 3 positive) is not supported for the window-gate stack.** F's stream-tagged layer-3 keys were in
  a trained single *delta* channel. This is a different model, so there is no contradiction, but the Hebbian window stack
  does not do it.
- L3S_P8 bound 0/10 at P=8. Its failures are mostly OTHER/POSITION with a perfect key split at VAL (η² by key at VAL 1.00
  in 10/10): the gate routes values by key, not by stream.

## CHECKs (all asserted, all ok)

X's arm A seed 160 through 2400 bit for bit. FAR_L3_P4|160 through 2400 equals batch 2's record bit for bit, and all ten
full reruns equal their records (`repro True` 10/10). The residual loop's logits equal model(x)'s bit for bit (three model
kinds), and layer 1's input is LN(embedding). The probe labels equal the block's CTX token, with a disjoint 1600/400
sequence split. A synthetic probe gives 0.998 on an informative feature and 0.474 on a random one. L3S_P8's parameters,
window, frozen W_h and groups; H8's L = 37, qpos 35, vocabulary 26.

## Next

S56 (Hebbian) started 15:25 UTC with res_layers (2, 3): ORACLE_H 370-371, BASELINE, RESGATE_ONLY, REG, REG_NOSG,
REG_NUDGE and SINGLE_GDN on 370-379. The delta family follows on 390-399.
