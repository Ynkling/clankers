# Session F — report 8 (S88a the write strength, S88b the horizon). EXPLORATORY, not a result

Branch `claude/explore-F`. Readings committed at 40087cb before any run. **This container is an Intel Xeon @ 2.80GHz**;
Session F's earlier screens ran on a Xeon @ 2.10GHz. Every S88 run is on the 2.80GHz CPU, so pairs are bit for bit within
one CPU. The 2.10GHz records reproduce bit for bit here (see the CHECKs).

Code SHAs: S88a 4473fffc3608 (main-line modules at 9c5939e), S88b e9c41e74d945. Torch 2.14.0+cu130, 1 thread per run.
78/78 runs ok, none retried, none diverged. Seeds 2400-2409 and 2420-2429; the block appears in no record on any branch.
Full tables: `write_tables.md`, `horizon_tables.md`.

## Counts (Wilson 95%; band)

**S88a**: S = 8, P = 4, k = 16, conv, 43200; the delta recipe with H's exact-copy split. BOUND ROUTED, paired against
DELTA_FIXED.

| arm | BOUND ROUTED | vs FIXED: arm only / FIXED only | transitions (median; range) | splits (labelled ok) |
|---|---|---|---|---|
| DELTA_FIXED (β = 1) | **10/10** [0.72, 1.00] RELIABLE | — | 13200 (9600-36000) | 8 (2) |
| DELTA_LEARNED (w, b in the gate group) | 7/10 [0.40, 0.89] MAJORITY | 0 / 3; p = 0.125 | 33600 (12000-43200) | 26 (10) |
| DELTA_LEARNED_SLOW (w, b in the slow group) | 3/10 [0.11, 0.60] MINORITY | 0 / 7; p = 0.0078 | 28800 (20400-34800) | 27 (11) |

- **Validity:** the perfect gate with β = 1 bound on both 2400 and 2401 at 1200. VALID.
- The learned-β perfect gate (printed) bound at 30000 and 21600.
- Every learned-arm failure is a one-to-one map stuck at 0.31-0.78 (9 runs) or one two-stream merge (DELTA_LEARNED 2405).
  v1 and v2 agree on all of them.

**S88b**: one delta channel on RandHeaderTask-lite, 24000. BOUND, paired within a length.

| | FIXED (decay 0.95) | ROUTED (S59's learned forget gate) | ROUTED only / FIXED only |
|---|---|---|---|
| 1x (cue-to-key distance 1, 3, 5) | 9/10 [0.60, 0.98] RELIABLE | 6/10 [0.31, 0.83] MAJORITY | 0 / 3 (p = 0.125) |
| 2x (2, 6, 10) | 4/10 [0.17, 0.69] MINORITY | 5/10 [0.24, 0.76] MAJORITY | 3 / 2 (p = 0.5) |

- **Validity:** the perfect gate (k = 2) bound at 1200 on both seeds at both lengths. VALID.
- Across lengths: drop_FIXED = 5, drop_ROUTED = 1.

## Readings (fixed in the docstrings before any run) and which applies

- **S88a: "no reading applies."**
  - "Keeps the partition" fails: the better learned arm, DELTA_LEARNED, is 3 pairs behind (c − b = 3 > 1).
  - "Costs it" fails, though narrowly: it needs DELTA_FIXED to beat both learned arms by >= 4 vs 0.
    - DELTA_LEARNED_SLOW: 7 vs 0, which meets it.
    - DELTA_LEARNED: 3 vs 0, which does not.
- **S88b: "untested."** At 1x ROUTED binds 6/10, below the 8/10 the rule requires.
  - Had it been tested, neither reading would have applied:
    - "extends the horizon" fails at 2x (b = 3, c = 2, not 4 vs 0);
    - "the horizon is the decay's" fails because the gaps differ (|drop_ROUTED − drop_FIXED| = 4 > 1).

## S88a: what the learned write strength does

- **β collapses to about 0.**
  - **DELTA_LEARNED:** in all 10 runs β falls from 1 to medians of 0.3-0.9 by 1200 and below 0.1 in every layer by 3600.
  - **DELTA_LEARNED_SLOW:** β holds at 0.6-1.0 while w, b are at 1e-4. It collapses within 1200-6000 updates of the
    switch to 1e-3 at 2400 (below 0.1 by 3600-8400).
  - **The learned-β perfect gate** shows the same collapse. That is why it binds at 21600-30000 against 1200 with β = 1.
- **Where β stays nonzero.** At the end β is about 0 at CTX and KEY positions in every layer (KEY at most 0.15).
  - At VAL positions in layer 1 it stays of order 1 (median 0.03-1.87 across runs).
  - Up to 33% of layer-1 positions have β > 1, consistent with them being the VAL positions (a third of all positions).
    With the tied write, β scales the write as well as
    the erase. So the learned memory writes almost only at layer-1 VAL positions and erases almost nothing: a
    different, near-Hebbian memory, not a tuned delta rule.
- **β does not diverge by stream:** 0/10, 0/10 and 0/2 runs reach η² >= 0.5 at KEY positions. The 8 per-stream medians at
  KEY are within 0.04 of each other in every run.
- **The splits misfire more often.** The learned arms' splits are labelled ok on 10/26 and 11/27 (DELTA_FIXED: 2/8). They
  fire about three times as often, because the learned runs plateau longer.
- **Post hoc (not separated here):** gradient descent on β drives the write strength down at the positions that matter
  for binding. With β fixed at 1 the delta recipe is the strongest arm (10/10, median 13200); every learned variant is
  slower or fails. A Gated-DeltaNet-style β, at least in this parametrisation (2·sigmoid, starting at 1, shared w), is
  not a free addition to the recipe.

## S88b: what the horizon screen shows (descriptive; untested)

- **The failures do not look like a horizon.**
  - Every unbound run, in all four arms, ends at 0.48-0.51 accuracy. That holds equally at every cue-to-key distance
    (1, 3, 5 or 2, 6, 10) and whether or not the queried pair was its key's last write.
  - The channel retrieves the key's two values but cannot tell which stream the query asks for: a stream confusion, not
    the far keys being lost.
  - Bound runs are at 0.99-1.00 at every distance.
  - The 2x layout halves FIXED's count (9 -> 4) through the same confusion, not by forgetting the far keys.
- **The learned decay** moves Δ from −2.60 to about −3.0 and stays near 1 at KEY / VAL / FILL positions (α 0.98-1.00).
  - At 1x it decays at CTX in layer 1 (α 0.89): a weak reset at the cue, as in S59.
  - At 2x it does not reset (α ≈ 1 at CTX).
- **The two arms differ in kind across lengths.** The learned gate keeps its count from 1x to 2x (6 -> 5) while the fixed
  decay drops (9 -> 4). But the learned gate starts lower at 1x, so this does not show an extended horizon. Ten seeds per
  cell cannot separate a loss of 3 from noise (p = 0.125 at 1x).

## CHECKs (all passed, asserted before any run)

- explore_f_delta_checks 26/26. C6 (LOCAL3_SLOW|160, a 2.10GHz record) is now asserted on any CPU; it reproduces bit for
  bit on the 2.80GHz Xeon.
- S88a (11):
  - DELTA_FIXED on seed 1315 reproduces S76's DELTA_SPLIT|1315 bit for bit through 7200 (first split at 4800 + one check):
    curve, statistics and trigger rows.
  - DELTA_LEARNED and DELTA_LEARNED_SLOW with w and b frozen equal it bit for bit through the same 7200. β is 1.0 exactly
    at every probe position, so the β probe does not disturb training.
  - The learned model is DELTA_FIXED's plus w, b, with equal logits at init.
  - w and b sit in the stated groups and lrs; one Adam step moves them.
  - explore_f_delta_fast's checks with a per-token β pass.
  - The perfect gate is one-hot on both memories.
- S88b (12):
  - One S70 reproduction: S70_DELTA|330 (2.10GHz record) through 2400 bit for bit with S70's own path. The 1x task gives
    S70's batches.
  - The 2x layout: the same pairs, queries and order, every key's distance to its cue exactly doubled, 12 FILL per
    sequence, FILL not counted as a value.
  - The forget gate with its parameters frozen equals the fixed decay bit for bit through 2400 at 1x and at 2x.
  - The routed decay as implemented equals S59's RoutedDecay to 1e-16, and its decay matrix equals delta_parallel's to
    1e-16 (float64).
  - The decay trains, and the perfect gate is one-hot.
- Before the readings' commit I ran one 1200-update smoke run of DELTA_LEARNED on seed 7 (outside the block, not stored).

## Departures from the specification (stated in the docstrings before any run)

- **S88a: β = 2·sigmoid(w·v + b) with w = b = 0, not sigmoid(w·x + b).**
  - The fixed value is 1. A sigmoid reaches exactly 1.0 in float32 only from b = 17, where its gradient is exactly 0.
  - 2·sigmoid starts at exactly 1.0 with a live gradient. Its range is (0, 2), the delta rule's stable range with unit
    keys.
  - The collapse above happens below 1, so the range's upper half does not drive the result.
- **S88b: S59's routed decay as "the learned forget gate", not Gated DeltaNet's.**
  - GDN's random initial rates cannot equal the fixed decay when frozen, as the CHECK requires.
  - The routed decay was also S59's stronger gate.
  - The 2x task doubles each key's distance to its cue with FILL tokens and keeps every pair adjacent.

## Runtime

- **S88a:** 159.9 h of run time, 23:07 UTC 12 October to 17:03 UTC 14 October (about 42 h wall on 4 workers, CHECKs
  included).
- **S88b:** 8.7 h of run time, 17:03-19:26 UTC 14 October.

## Anything unexpected

- **The CPU changed** between rounds, from Xeon @ 2.10GHz to @ 2.80GHz. The old records reproduce bit for bit on the new
  one.
- **The learned-β runs were slower per update** than β = 1: 0.34-1.51 s against 0.30-0.37 s, and up to 15 h per run. As β collapses, part of the
  erase strengths fall below 1e-38 (denormal numbers). There the triangular solve is about 40× slower: 0.69 s vs 0.02 s
  per call in a direct test.
  - This changes time only, not results.
  - Flushing denormals to zero would remove it, but would change the numerics.
- **The size and speed of the β collapse.** Under both lr schedules it is complete within about 1200 updates of β's lr
  reaching 1e-3.
