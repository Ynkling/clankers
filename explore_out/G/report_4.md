# Session G — report 4 (follow-up S73 register3 on delta, S72 register3 on Hebbian). EXPLORATORY, not a result

Branch `claude/explore-G`; Intel Xeon @ 2.10GHz, torch 2.14.0+cu130, 1 thread per run. Task: RandHeaderTask-lite exactly
as F's S70 (L = 23, k = 2, no conv), 24000 updates, SLOW recipe (SINGLE_D: Adam 1e-3). Specs and readings committed at
26c4ab1 before any S73 / S72 run. 150/150 runs ok, none retried. Validity oracles are S71's (same seeds 700-709): Hebbian
ORACLE_ALL 10/10, delta ORACLE_ALL_D 10/10, so both families are VALID.

## Counts (Wilson 95%; band). The readings use ROUTED-BOUND = BOUND ∧ F's ROUTED*@end.

| arm (seeds) | BOUND | **ROUTED-BOUND** | ROUTED*@end | DISCOVERED | my BOUND ROUTED | unbound classes | transitions (median, range) |
|---|---|---|---|---|---|---|---|
| **S73, delta (β = 1, L2 keys)** | | | | | | | |
| BASELINE_D (700-719) | 14/20 [0.48, 0.85] | **0/20** [0.00, 0.16] NEVER | 0/20 | 0/20 | 11/20 * | KEY 6 | 3600 (3600-13200) |
| REG3_D (700-719) | 8/20 [0.22, 0.61] | **4/20** [0.08, 0.42] MINORITY | 4/20 | 5/20 | 5/20 | KEY 12 | 2400 (1200-16800) |
| REG3_NUDGE_D (700-709) | 10/10 | **10/10** [0.72, 1.00] RELIABLE | 10/10 | 10/10 | 10/10 | – | 1200 (all) |
| RESGATE_ONLY_D (700-709) | 5/10 [0.24, 0.76] | **0/10** [0.00, 0.28] NEVER | 0/10 | 0/10 | 0/10 | KEY 5 | 6000 (4800-15600) |
| LATCH3_D (700-709) | 1/10 [0.02, 0.40] | **0/10** NEVER | 0/10 | 0/10 | 0/10 | KEY 9 | 13200 |
| LATCH3_NUDGE_D (700-709) | 9/10 | **9/10** [0.60, 0.98] RELIABLE | 9/10 | 9/10 | 9/10 | STREAM-PARTIAL 1 | 1200 (all) |
| SINGLE_D (k = 1, 700-709) | **7/10** [0.40, 0.89] MAJORITY | – | – | – | – | 3 (acc 0.48-0.50) | 10800 (1200-20400) |
| **S72, Hebbian** | | | | | | | |
| BASELINE (700-719) | 1/20 [0.01, 0.24] | **0/20** [0.00, 0.16] NEVER | 0/20 | 0/20 | 1/20 | KEY 16, STREAM-PARTIAL 3 | 16800 |
| REG3 (700-719) | 4/20 [0.08, 0.42] | **4/20** [0.08, 0.42] MINORITY | 4/20 | 4/20 | 4/20 | KEY 14, OTHER 2 | 3000 (2400-6000) |
| REG3_NUDGE (700-719) | 20/20 | **20/20** [0.84, 1.00] RELIABLE | 20/20 | 20/20 | 20/20 | – | 2400 (2400-8400) |

\* My BOUND ROUTED takes the argmax of the mean gate per stream with no margin guard, so near-uniform gates (0.50 vs
0.51) count as one-to-one. Use ROUTED-BOUND. F's references (other seeds, counts only): S70_HEBB BOUND 1/20, ROUTED*@end
0/20, KEY 19; S70_DELTA BOUND 6/20, ROUTED*@end 0/20, KEY 14. My BASELINE matches F's Hebbian counts; my BASELINE_D binds
more often (14/20 against 6/20, F's fast kernel on other seeds), and routes none either way.

## Readings (pre-fixed) and which apply

| reading | test | result |
|---|---|---|
| S73 "the register carries the cue" | REG3_NUDGE_D ≥ 9/10 | **APPLIES** (10/10) |
| S73 "discovered (register)" | REG3_D beats BASELINE_D | does not apply: 4 / 0, p = 0.0625 (n 20) |
| S73 "discovered (residual gate)" | RESGATE_ONLY_D beats BASELINE_D | does not apply: 0 / 0 |
| S73 "discovered (latch)" | LATCH3_D beats BASELINE_D | does not apply: 0 / 0 |
| S73 "the latch holds on delta memory" | LATCH3_NUDGE_D end z ≥ 0.9 at CTX, ≤ 0.1 at K and V on ≥ 8/10 | **APPLIES** (9/10; z = 1.00 / 0.00 / 0.00) |
| S73 "the delta stack carries the cue" | BASELINE_D probe L3 ≥ 0.9 at K and V on ≥ 10/20 | does not apply (7/20) |
| S73 "the gate is not needed on this task" | SINGLE_D BOUND ≥ 8/10 | does not apply (7/10, one short) |
| S72 R1 | REG3_NUDGE ≥ 17/20 → "carries the cue at all three layers"; < 12 → "insufficient" | **"the register carries the cue at all three layers"** (20/20) |
| S72 R2 "discovered" | REG3 beats BASELINE | **not discovered**: 4 / 0, p = 0.0625 (n 20) |

## What this says (exploratory)

1. **Routing layer 1 was the missing piece on Hebbian memory.** S56's REG_NUDGE (gate at layers 2-3, layer 1 on LOCAL3)
   bound 0/10 on the P = 8 header with perfect routing at layers 2-3. With one register gate shared by all three layers
   the same nudge binds 20/20 on RandHeaderTask-lite, all one-to-one with margin 1.00 at every layer, transitions
   2400-8400. That matches S71: on Hebbian memory layer 1's routing is necessary, and almost sufficient by itself.
2. **The register can carry the cue on both memories, and the nudge finds that solution every time.** Unnudged, it finds it
   on 4/20 seeds on each memory (REG3 Hebbian 700, 703, 714, 715; REG3_D 702, 711, 715, 717; only 715 in common). Each
   time the bound runs route perfectly, against 0/20 routed for either baseline. Both screens land one discordant pair
   short of p < 0.05, so the pre-fixed reading is "not discovered" on each. Post hoc, not a reading: the two screens
   share seeds but not models, so pooling them is not a pre-fixed test. It is printed only as a description: 8 / 0 over
   40 pairs.
3. **What separates a discovering run is the write.** In every routed unnudged run β^R at CTX rose above K/V (Hebbian
   0.32-0.64 vs 0.16-0.24; delta 0.36-0.58 vs ≈ 0.2). In the key-split runs it stayed flat (≈ 0.2-0.3 at every role) or
   higher at KEY. The nudged runs end at CTX 0.45-0.84 vs K/V ≈ 0.18. So the register's write decision at CTX is what
   discovery has to find, and in 16 of 20 seeds the gate locks into a key split first (KEY 14 Hebbian, 12 delta).
4. **The latch is a working write on delta memory once nudged, and collapses without it.** LATCH3_NUDGE_D holds an exact
   latch (z = 1 at CTX, 0 at K/V on 9/10) and binds routed 9/10 at 1200. LATCH3_D goes to z = 1 everywhere by update
   1200 (it writes every token, so it carries only the current token) and key-splits 9/10, as S57 did on Hebbian.
5. **The residual gate alone does not read the delta memory's stream tag.** RESGATE_ONLY_D binds 5/10 but routes 0/10:
   its gates are 0.5/0.5 at K and V in every layer. The bound runs are key-split bindings.
6. **Stack vs gate (the in-run probe).** No routed run has the cue in its residual (probe L3 at K / V ≈ 0.50-0.57 in every
   routed run of every arm); the gate carries it. The runs whose stack does carry it are the non-routing solutions:
   SINGLE_D's bound runs (probe L3 0.93-1.00 / 0.56-1.00, final accuracy 1.00) and BASELINE_D's bound runs (0.87-0.98,
   half-routed gates with margin ≈ 0.5). On BASELINE_D, probe ≥ 0.9 × ROUTED-BOUND = [[0, 7], [0, 13]]. So on delta
   memory the stack can learn stream-tagged keys (F's S53), and when it does, the gate does not need to route.
7. **A single delta channel binds this task 7/10** (transitions 1200-20400); the three unbound runs sit at exactly 0.50,
   flat in every (stream, block, pair) cell. Its bound runs answer last-written and not-last pairs alike (1.00 / 1.00):
   the stream is in the key, not in the write order.

## Plateaus (accuracy by query stream | block | pair index, unbound runs, median)

- Hebbian BASELINE (19 runs) and REG3 (16): 0.46-0.55 in all 12 cells. They are flat at chance between the two streams,
  the same picture as ORACLE_23 in S71. S56's 0.32 plateau was on the P = 8 header (8 keys); here the chance level is 0.5.
- Delta: BASELINE_D (6) 0.83-0.87; REG3_D (12) 0.78-0.83; RESGATE_ONLY_D (5) 0.72-0.78; LATCH3_D (9) 0.75-0.79. All are
  flat across cells, so no block or pair position is worse. LATCH3_NUDGE_D's one unbound run (703) is perfect on stream 0
  and falls with the pair index on stream 1 (0.82 / 0.49 / 0.27, and 0.77 / 0.50 / 0.23).

## Diagnostics (median over runs; full tables in `s7[123].log`, `s7[123]_summary.json`)

- β^R CTX / KEY / VAL by update, REG3_D: 0.10 / 0.10 / 0.10 (0), 0.25 / 0.23 / 0.23 (1200), 0.29 / 0.25 / 0.24 (end).
  REG3: 0.24 / 0.23 / 0.23 (1200), 0.26 / 0.28 / 0.21 (end). REG3_NUDGE: 0.34 / 0.21 / 0.23 (0), 0.47 / 0.19 / 0.23
  (2400), 0.67 / 0.18 / 0.21 (end). LATCH3_D's z: 0.50 at 0, then 1.00 / 1.00 / 1.00 from 1200.
- Routing margin at VAL, layers 1 / 2 / 3: nudged arms 1.00 / 1.00 / 1.00; REG3 and REG3_D 0.00 / 0.00 / 0.00 (median;
  1.00 in the routed runs); BASELINE_D 0.45 at every layer (one gate).
- Contingency (mean g, rows stream 0 / 1, columns ch 0 / 1): nudged arms KEY and VAL [[1, 0], [0, 1]] at every layer, CTX
  near 0.5 (the gate at a CTX token reads the register before that token's write). Unnudged REG3: KEY / VAL
  [[0.5, 0.5], [0.5, 0.5]], CTX leaning to one channel.
- In-run probe (median, L3 K / V): BASELINE 0.61 / 0.52; REG3 0.57 / 0.51; REG3_NUDGE 0.51 / 0.50; BASELINE_D 0.89 / 0.90;
  REG3_D 0.76 / 0.62; SINGLE_D 0.97 / 0.88.

## CHECKs (all asserted, all ok)

S73 37, S72 32, plus X's arm A bit for bit at every start. They include:
- F's check_rand_header.
- The in-run probe equals S55's extraction.
- S56's 12 CHECKs.
- REG3 with the layer-1 path disabled (gated (2, 3), input v) reproduces S56's REG|370 on H8 through 2400 bit for bit.
- With every new path disabled, REG3(_D)'s gate at every layer equals LOCAL3's and its logits equal BASELINE(_D)'s bit for
  bit.
- Register and latch causality at all three layers, layer 1 included.
- Strict causality.
- β = 0 delta equals Hebbian (1.2e-7).
- The SLOW groups.
- The nudge at all three layers (stream lean 0.046-0.050).
- LATCH3_D's dead-STE guard (|u| 1.5e-3).
- RESGATE_ONLY_D has no register.
- SINGLE_D equals F's S53 single-channel path bit for bit.

## Runtime and anything else

S71 42.8 min, S73 144.8 min, S72 84.0 min on 4 workers (4.5 h). No restart. explore_delta_mem.py is unchanged from F's
branch (same SHA-1). The progress watcher hit the 2-hour background limit twice and was restarted, with no effect on the
runs.

Suggested next (not run): REG3 on more seeds with a pre-registered pooled test across both memories; what makes the
register's CTX write appear before the key split forms (e.g. the key-term hinge of S52 on the shared gate, or a slower gate
group for the first 600 updates).
