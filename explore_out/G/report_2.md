# Session G — report 2 (S56 register, Hebbian and delta; S57 latch). EXPLORATORY, not a result

Branch `claude/explore-G`; Intel Xeon @ 2.10GHz, torch 2.14.0+cu130, 1 thread per run. Task: the FIXED header layout,
P = 8, S = 2 (HeaderTask, L = 37), k = 2, no conv, 24000 updates (the user's instruction of 9 October; F's RandHeader-lite
was not published). Specs and readings committed before any run: S56 at a7bbee0, S57 at 0160269. Code SHAs: S56
70304e0e7121, S57 0440fcc975a6. 139/139 runs ok, none retried.

## Counts (Wilson 95%; band)

| family / arm (seeds) | BOUND | BOUND ROUTED | unbound failure classes | transitions |
|---|---|---|---|---|
| **Hebbian, 370-379** | | | | |
| ORACLE_H (370-371) | 1/2 [0.09, 0.91] | 1/2 | STREAM-PARTIAL 1 (acc 0.38) | 15600 |
| BASELINE (LOCAL3+SLOW) | 0/10 [0.00, 0.28] NEVER | 0/10 | KEY 5, POSITION 4, OTHER 1 | – |
| RESGATE_ONLY | 0/10 NEVER | 0/10 | KEY 10 | – |
| REG (SG) | 0/10 NEVER | 0/10 | KEY 9, OTHER 1 | – |
| REG_NOSG | 0/10 NEVER | 0/10 | KEY 10 | – |
| REG_NUDGE | 0/10 NEVER | 0/10 | STREAM-PARTIAL 8, OTHER 2 | – |
| SINGLE_GDN (k=1, Hebbian + GDN decay) | 0/10 NEVER | – | (median acc 0.16) | – |
| **Delta (β = 1, L2 keys), 390-399** | | | | |
| ORACLE_D (390-391) | 2/2 [0.34, 1.00] | 2/2 | – | 2400, 18000 |
| BASELINE_D | 2/10 [0.06, 0.51] MINORITY | 0/10 | POSITION 4, OTHER 3, KEY 1 | 13200, 22800 (both key splits) |
| **REG_D** | **6/10 [0.31, 0.83] MAJORITY** | 2/10 [0.06, 0.51] | POSITION 2, MERGED 1, OTHER 1 | 4800-18000 (median 7200) |
| REG_NUDGE_D | 8/10 [0.49, 0.94] MAJORITY | 7/10 [0.40, 0.89] MAJORITY | OTHER 2 | 3600-6000 |
| **Latch (S57), Hebbian, 410-419** | | | | |
| ORACLE_L (410-411) | 1/2 [0.09, 0.91] | 1/2 | STREAM-PARTIAL 1 (acc 0.23) | 22800 |
| BASELINE_L | 0/10 NEVER | 0/10 | POSITION 5, KEY 4, OTHER 1 | – |
| LATCH | 0/10 NEVER | 0/10 | POSITION 4, OTHER 3, KEY 3 | – |
| LATCH_NUDGE | 0/10 NEVER | 0/10 | KEY 5, STREAM-PARTIAL 3, OTHER 1, POSITION 1 | – |
| LATCH_PRIOR | 0/10 NEVER | 0/10 | KEY 8, OTHER 2 | – |
| ORACLE23 (post hoc, 370-372) | 0/3 [0.00, 0.56] | 0/3 | STREAM-PARTIAL 3 (acc 0.34, 0.65, 0.32) | – |

Paired one-sided McNemar (BOUND): REG_D vs BASELINE_D 4 / 0, p = 0.0625. Hebbian REG / REG_NOSG vs BASELINE, REG vs
SINGLE_GDN, and LATCH / LATCH_PRIOR vs BASELINE_L: all 0 / 0, p = 1.

## Readings (pre-fixed) and which apply

Validity: ORACLE_H 1/2, ORACLE_D 2/2, ORACLE_L 1/2. All three meet the ≥ 1/2 rule, so the families are VALID. The Hebbian
oracle is at the edge of this budget, though: one seed of two in each family, and the bound seeds at 15600 and 22800.
Treat the Hebbian zeros as weak negatives.

| reading | Hebbian (S56) | delta (S56) | latch (S57) |
|---|---|---|---|
| R1 (nudge arm BOUND ROUTED: ≥ 9 carries, < 6 insufficient) | **architecture insufficient** (0/10) | **neither** (7/10) | **architecture insufficient** (0/10) |
| R2 (REG-type BOUND ≥ BASELINE + 3, KEY ≤ 2) | not discovered (0 vs 0) | **discovered (REG_D)**, 6 vs 2, see caveat | not discovered (0 vs 0) |
| R3 vs SINGLE_GDN | "the single GDN channel is at least as good" (0 vs 0) | – | – |
| R4 register adds to the residual gate | it does not (0 vs 0) | – | – |
| R5 SG matters | SG does not matter (0 vs 0) | – | – |
| S57 trigger | TRIGGERED (REG β^R at CTX 0.57 at the end) | – | ran |

**Caveat on REG_D's "discovered".** The rule's "KEY splits ≤ 2" was implemented on the unbound runs' failure classes,
where REG_D has 0. But only 2 of REG_D's 6 bound runs are routed (394, 399: one-to-one maps, margin 1.00). Three of the
other four bound with a key-split gate (η² by key at layer 2 of 1.00, 1.00 and 0.87; runs 390, 395, 398), and 397 bound
with an unrouted gate. Counting key-split *bound* runs as key splits gives 3 > 2, and the reading becomes "not
discovered". So on delta channels the register raises binding (6 vs 2) but mostly without stream routing. This is the
"bound ≠ routed" effect F found for delta memory. The routed count, 2/10 vs 0/10, is the cleaner number.

## Diagnostics (medians over runs)

- **β^R (Hebbian, CTX / KEY / VAL)**:

  | arm | update 0 | 2400 | 9600 | end |
  |---|---|---|---|---|
  | REG | 0.10 / 0.10 / 0.10 | 0.25 / 0.26 / 0.24 | 0.50 / 0.65 / 0.56 | 0.57 / 0.82 / 0.70 |
  | REG_NOSG | – | – | – | 0.37 / 0.69 / 0.50 |
  | REG_NUDGE | – | 0.24 / 0.07 / 0.08 | 0.83 / 0.03 / 0.07 | 0.96 / 0.02 / 0.05 |

  Without the nudge the register never separates CTX from K/V: it writes at every token, slightly *less* at CTX. With the
  nudge it is a clean latch.
- **β^R (delta, end)**: REG_D 0.36 / 0.13 / 0.13. Its two routed runs separate CTX (0.36, 0.40) from K/V (≤ 0.10);
  REG_NUDGE_D 0.67 / 0.05 / 0.05.
- **Routing margin at VAL, layers 1 / 2 / 3**: REG_NUDGE 0.00 / 1.00 / 1.00; REG_NUDGE_D 0.00 / 1.00 / 1.00; every
  unnudged Hebbian arm 0.00 at all layers; ORACLE_* 1.00 everywhere.
- **Contingency g at VAL (layer 3, [stream 0, stream 1] × [ch 0, ch 1])**: REG_NUDGE [[1.0, 0.0], [0.0, 1.0]];
  REG [[0.5, 0.5], [0.5, 0.5]]; RESGATE_ONLY [[0.68, 0.32], [0.76, 0.24]]; REG_NUDGE_D [[1.0, 0.0], [0.24, 0.76]]
  (pooled). At CTX the gates are not stream-aligned in any learned arm. Full tables are in `s56_*_summary.json` and
  `s57_summary.json`.
- **Latch firing rate z (CTX / KEY / VAL)**: LATCH goes from 0.5 at init to 1.0 / 1.0 / 1.0 by update 1200 and stays
  there; it always writes, so it holds only the current token. LATCH_PRIOR goes to 0.0 / ~0.1 / 0.0: it never writes at
  CTX. LATCH_NUDGE starts at 1.0 / 0.0 / 0.09 after the nudge and drifts to 1.0 / 0.94 / 1.0 by the end. The 3 runs that
  kept the latch (410, 413, 417) route at layers 2-3 (margin 0.66-1.00) and still sit at acc 0.29-0.33.

## What answers "write decision or read path?" (R1 "architecture insufficient")

Neither, on the Hebbian memory. REG_NUDGE has a perfect write (β^R 0.96 at CTX vs ≤ 0.05) and a perfect read path
(one-to-one, margin 1.00 at layers 2-3). It still plateaus at acc 0.32-0.44 from update 2400 on. The post-hoc ORACLE23
(LOCAL3 at layer 1, the *perfect* gate at layers 2-3, seeds 370-372) also binds 0/3, while ORACLE_H 371 (all three layers
routed) bound at 15600. Exploratory reading: **on Hebbian channels the unrouted layer-1 memory blocks binding even when
layers 2-3 are routed perfectly**; a gate that only acts at layers 2-3 cannot be enough here. On delta channels the same
architecture binds (REG_NUDGE_D 8/10, routed 7/10, transitions 3600-6000), so the delta memory tolerates an unrouted
layer 1.

## Anything unexpected

- The Hebbian perfect gate at this budget is marginal (3 of 6 seeds bound over S55/S56/S57, at 15600-24000; S53 3/3 within 48000). A
  48000 budget would have made the Hebbian zeros more informative.
- SINGLE_GDN (F's S69_GDN at 24000 instead of 48000) bound 0/10, median acc 0.16. F's single *delta* channel with GDN
  decay bound 8/10 at 48000 (S59); at 24000 on a Hebbian channel it does not get started.
- Without the nudge, every learned per-layer gate collapsed to KEY splits (RESGATE_ONLY 10/10, REG_NOSG 10/10).
- A container restart killed S57's first attempt during its run queue, before any run was saved. It was rerun from the
  start (`s57_seg1_restart.log`), with the same code SHA and CPU and the CHECKs repeated.

## CHECKs (asserted at every start, all ok)

X's arm A seed 160 bit for bit (every segment). S56 (12):
- the recorded run with the new path disabled (batch 16's S43 LOCAL3_SLOW|160 through 2400, curve and decoder bit for bit,
  with RegBDH(gated = ()); this needed the per-layer G shared and the embedding looked up twice, as LocalBDH does);
- RegBDH(gated = ()) equals LOCAL3's logits on H8;
- initialisation: u = W_in v with A_r = A_R = 0, β^R ≈ 0.1;
- register causality (the gate at t reads R_{t-1}): token t's write changes nothing at ≤ t and changes t + 1;
- strict causality;
- SG blocks and NOSG passes the gradient into the stack;
- RESGATE_ONLY has no register;
- the SLOW groups;
- the nudge moves only the new gate parameters and leaves a stream lean;
- delta with β 0 / write 1 / raw keys equals Hebbian (1.5e-7);
- the oracles take nothing from the other stream;
- SINGLE_GDN as stated.

S57 (9 + S56's 12): the dead-STE guard (max |u| 2.4e-3 < 1 at step 0), the a schedule, the STE's forward and gradient, the
latch recurrence and its causality, the prior's gradient, shared initial parameters with REG, the nudge's scope, and
ORACLE23's gates.

## Runtime

S56 Hebbian 163.6 min (62 runs), delta 94.3 min (32 runs), S57 129.4 min (45 runs) plus the lost first attempt; 4 workers.

## Suggested next (not run)

ORACLE23 on delta channels and a register/gate at all three layers (layer 1 included) on Hebbian channels: R1's failure
points at layer 1, not at the register. REG_D on more seeds with routing as the outcome. The Hebbian oracle at 48000.
