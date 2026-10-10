# Session G — report 3 (follow-up S71: which layers must be routed). EXPLORATORY, not a result

Branch `claude/explore-G`; Intel Xeon @ 2.10GHz, torch 2.14.0+cu130, 1 thread per run. Task: RandHeaderTask-lite exactly
as F's S70 (explore_f_tasks copied unchanged at 9a6c2bf; L = 23, k = 2, no conv), 24000 updates, Adam 1e-3 on every
parameter for every S71 arm. Spec and readings committed at 9114843 before any run; child code SHA in
`s71_layers_results.json`. 50/50 runs ok, none retried; 42.8 min on 4 workers. Seeds 700-709 (no record anywhere uses
700-759).

## Counts (BOUND is the reading's outcome; Wilson 95%; band)

| arm (perfect gate at layers) | BOUND | ROUTED-BOUND | my BOUND ROUTED | unbound | transitions | median final acc |
|---|---|---|---|---|---|---|
| ORACLE_ALL (1, 2, 3), Hebbian | **10/10** [0.72, 1.00] RELIABLE | 10/10 | 10/10 | – | 1200 (1200-4800) | 1.000 |
| ORACLE_23 (2, 3; LOCAL3 at 1), Hebbian | **0/10** [0.00, 0.28] NEVER | 0/10 | 0/10 | STREAM-PARTIAL 10 | – | **0.504** |
| ORACLE_1 (1; LOCAL3 at 2, 3), Hebbian | **9/10** [0.60, 0.98] RELIABLE | 9/10 | 0/10 | 1 (acc 0.937) | 16800 (8400-24000) | 0.955 |
| ORACLE_ALL_D (1, 2, 3), delta | **10/10** [0.72, 1.00] RELIABLE | 10/10 | 10/10 | – | 1200 (1200-1200) | 1.000 |
| ORACLE_23_D (2, 3; LOCAL3 at 1), delta | **10/10** [0.72, 1.00] RELIABLE | 10/10 | 0/10 | – | 1200 (1200-3600) | 1.000 |

## Readings (pre-fixed) and which apply

Validity: ORACLE_ALL 10/10 and ORACLE_ALL_D 10/10 (≥ 8 needed), so both families are VALID. These are also the validity
oracles of S72 and S73.

| reading | paired BOUND (b = first arm only, c = second only) | applies? |
|---|---|---|
| "layer 1 is the block" (ORACLE_ALL beats ORACLE_23) | b 10, c 0, one-sided p = 0.00098 | **APPLIES** |
| "layer 1 is not needed" (ORACLE_23 ≥ ORACLE_ALL − 1 pair) | ALL only 10 | does not apply |
| "layer 1 alone suffices" (ORACLE_1 ≥ ORACLE_ALL − 1 pair) | ALL only 1, ORACLE_1 only 0 | **APPLIES** |
| "delta tolerates an unrouted layer 1" (ORACLE_23_D ≥ ORACLE_ALL_D − 1 pair) | 0 / 0 | **APPLIES** |

On Hebbian memory the routing that matters is layer 1's. With layer 1 unrouted, perfect gates at layers 2-3 never bind.
With layer 1 routed and layers 2-3 left to LOCAL3, the model binds 9/10. On delta memory an unrouted layer 1 costs nothing.
This reproduces S56/S57's ORACLE23 (0/3, header P = 8) on ten paired seeds, and explains S56's Hebbian REG_NUDGE plateau.

## What the runs look like (descriptive)

- **ORACLE_23 sits at chance between the two streams, everywhere.** Accuracy by (query stream, block, pair index) is
  0.46-0.51 in all 12 cells, and write-order accuracy is 0.52 (last-written) vs 0.49 (not last). It has learned the
  key → value binding but cannot pick the stream. The memory at layers 2-3 is routed perfectly (margin 1.00, one-hot
  tables), so the stream information that layers 2-3 would need must come from layer 1's memory, and an unrouted layer 1
  mixes the streams before the routed layers read it.
- **ORACLE_1 binds, but slowly and just over the line.** It bound at 8400-24000 (median 16800), against 1200 for the
  all-layer oracle, with final accuracy 0.95-0.97 rather than 1.00. The unbound seed, 708, ended at 0.937. Its layers 2-3
  stay unrouted (margin 0.00, both streams on one channel), which is why my BOUND ROUTED is 0/10 while ROUTED-BOUND is
  9/10. F's ROUTED*@end reads the returned gate, which for ORACLE_1 is layer 1's. So 708's "MERGED" label is the
  layer-3 map, not a failure of the layer-1 routing.
- **In-run probe (median held-out accuracy, K / V).** No arm's stack carries the stream at VAL positions; the gates do it.
  Hebbian ORACLE_ALL: L2 0.64 / 0.51, L3 0.67 / 0.53, final 0.71 / 0.62. Delta ORACLE_ALL_D: L3 0.81 / 0.52.
  ORACLE_23_D: L3 0.73 / 0.51. ORACLE_1: L3 0.63 / 0.52. The delta stack carries more at KEY positions (0.73-0.81) but not
  at VAL.
- Write-order accuracy is 1.00 / 1.00 in every bound oracle arm.

## CHECKs (all asserted, all ok; 23 plus X's arm A bit for bit)

- F's check_rand_header (9 rows: every (stream, key) once, block lengths 0.338 / 0.324 / 0.338, 6 block orders, and so
  on) and the task's parameters.
- The in-run probe's hooked features equal S55's extraction bit for bit on LOCAL3, and respect per-layer gates.
- acc_by_cell covers all 2048 queries.
- Each oracle is one-hot on the stream at its perfect layers and LOCAL3's gate elsewhere.
- ORACLE_ALL / ORACLE_ALL_D equal S56's ORACLE_H / ORACLE_D code path (parameters and logits).
- The arms share the stack parameters.
- The all-layer oracles take nothing from the other stream.
- explore_delta_mem.py is unchanged from F's current branch (same SHA-1), so the β = 0 check was not rerun separately.
  It is in S73/S72's CHECK set and passed there.

## Next

S73 (delta, 90 runs) started right after S71 with its CHECKs passed (37/37), and S72 (Hebbian, 60 runs) follows it. Given
S71, S72's REG3 (a gate at all three layers) is the arm that can route layer 1 on Hebbian memory.
