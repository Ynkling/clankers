# Session G — report 5 (S82: devices against the key split). EXPLORATORY, not a result

Branch `claude/explore-G`; Intel Xeon @ 2.10GHz, torch 2.14.0+cu130, 1 thread per run. RandHeaderTask-lite exactly as F's
S70 (L = 23, k = 2, no conv), 24000 updates, SLOW (oracles: Adam 1e-3). Spec, readings and the best-device rule
committed at 83699a2 before any run; child code SHA 1ec70589520b. 136/136 runs ok, none retried; 222 min on 4 workers.
Seeds 2100-2119 (no record anywhere uses them).

## Counts (ROUTED-BOUND = BOUND ∧ F's ROUTED*@end is the readings' outcome; Wilson 95%; band)

| arm (seeds) | BOUND | **ROUTED-BOUND** | DISCOVERED | my BOUND ROUTED | unbound classes | transitions |
|---|---|---|---|---|---|---|
| BASELINE (LOCAL3+SLOW; 2100-2119) | 3/20 | **0/20** [0.00, 0.16] NEVER | 0/20 | 3/20 * | KEY 17 | 10800-16800 (key splits) |
| REG3 (2100-2119) | 2/20 | **2/20** [0.03, 0.30] MINORITY | 2/20 | 2/20 | KEY 16, OTHER 1, S-PARTIAL 1 | 3600, 3600 |
| **REG3_KEYHINGE** | 20/20 | **20/20** [0.84, 1.00] RELIABLE | 20/20 | 20/20 | – | 3600 ×18, 4800, 6000 |
| REG3_SLOWGATE | 0/20 | **0/20** [0.00, 0.16] NEVER | 0/20 | 0/20 | KEY 18, OTHER 1, S-PARTIAL 1 | – |
| REG3_NOWIN | 5/20 | **5/20** [0.11, 0.47] MINORITY | 5/20 | 5/20 | KEY 12, OTHER 2, POSITION 1 | 3600 ×4, 4800 |
| REG3_NUDGE (2100-2101; validity) | 2/2 | 2/2 | 2/2 | 2/2 | – | 2400 |
| ORACLE_ALL (2100-2101; validity) | 2/2 | 2/2 | 2/2 | 2/2 | – | 1200 |
| *delta (β 1, L2 keys), 2100-2109* | | | | | | |
| ORACLE_ALL_D (2100-2101) | 2/2 | 2/2 | 2/2 | 2/2 | – | 1200 |
| BASELINE_D | 2/10 | **0/10** NEVER | 0/10 | 2/10 * | KEY 8 | 3600 (key splits) |
| REG3_D | 4/10 | **2/10** MINORITY | 3/10 | 4/10 | KEY 6 | 1200-9600 |
| **REG3_KEYHINGE_D** | 10/10 | **10/10** [0.72, 1.00] RELIABLE | 10/10 | 10/10 | – | 2400 ×9, 3600 |

\* My BOUND ROUTED has no margin guard. It counts these bindings as one-to-one; they are key splits, with margin 0.

## Readings (pre-fixed) and which apply

Validity: S71's ORACLE_ALL (10/10) stands. Its rerun on 2100-2101 bound 2/2 and REG3_NUDGE 2/2, so the screen is VALID.

| arm | "discovered" (beats BASELINE, p < 0.05) | "helps" (p(D > REG3) ≤ p(4 vs 0)) | "hurts" (the reverse) | "a working write without the device's cost" (no KEY failures and β^R CTX > K, V in ≥ 15/20) |
|---|---|---|---|---|
| REG3 (reference) | does not apply (2 / 0, p 0.25) | – | – | (β^R CTX > K, V in 3/20) |
| **REG3_KEYHINGE** | **APPLIES** (20 / 0, p 1e-6) | **APPLIES** (18 / 0, p 4e-6) | no | **APPLIES** (KEY 0; 20/20) |
| REG3_SLOWGATE | does not apply (0 / 0) | no (0 / 2) | does not apply (2 / 0, p 0.25) | does not apply (KEY 18; 1/20) |
| **REG3_NOWIN** | **APPLIES**, at the threshold (5 / 0, p 0.031) | no (3 / 0, p 0.125) | no | does not apply (KEY 12; 5/20) |

Best device (pre-fixed rule: most ROUTED-BOUND, then fewer KEY failures) was REG3_KEYHINGE (20; NOWIN 5; SLOWGATE 0).
On delta (descriptive, 10 seeds) REG3_KEYHINGE_D beat BASELINE_D 10 / 0 (p 0.001) and REG3_D 8 / 0 (p 0.004).

## What happened (exploratory)

- **The hinge removes the key split before it forms, and the register's write appears right after.** At update 1 the
  shared gate's η² by key at KEY positions is already 0.89 (median over runs, mean over layers). The near-uniform gate's
  small variations are organised by key from initialisation. The hinge fires 55-142 times in updates 1-2400 (median 78)
  and holds η²_key at 0.05 from update 300; it is 0.00 by 2400. Meanwhile β^R at CTX separates from K/V (0.12 / 0.12 /
  0.12 at 1200 → 0.32 / 0.19 / 0.21 at 2400 → 0.48 / 0.20 / 0.21 at the end; 0.36-0.62 at CTX in every run). 18 of 20
  bind at 3600, the first evaluation after the hinge stops and the stack's learning rate rises. The gate then routes
  one-to-one at all three layers (margin 1.00; KEY and VAL tables [[1, 0], [0, 1]]). Once the hinge is off the key split
  does not come back: η²_key is 0.00 at the end. On delta memory the same happens at 2400.
- **The slow gate is worse than nothing.** With the gate at 1e-4 for 1200 updates, β^R stays flat (0.20 / 0.26 / 0.21
  at the end) and 18/20 key-split. So slowing the gate delays the register along with the split; it does not let the
  write win the race.
- **No window input gives a little discovery and a leaky register.** REG3_NOWIN routes 5/20 (seeds 2108, 2110, 2111,
  2113, 2116, a superset of REG3's 2108, 2110). Its β^R climbs at every role (0.86 / 0.95 / 0.92 at 24000): the register
  writes nearly every token. The key split still forms through the residual in 12/20.
- **Plateaus.** Every unbound Hebbian arm sits at 0.45-0.54 in all 12 (stream, block, pair) cells: chance between
  streams, as in S71-S72. Delta unbound runs: BASELINE_D 0.71-0.80, REG3_D 0.74-0.86, flat across cells.
- **The in-run probe** reads about 0.50 at L2, L3 and final, at K and V, in every routed arm. As before, the gate
  carries the stream and the stack does not.

## Caveats

- **What the hinge sees.** It reads key positions and key identities (token types). That is task structure, not a
  stream label. In a language model "key identity" would have to be token identity, which a gate must be allowed to use
  in general. So the hinge works here as a training-time prior for the first 2400 updates, not an architecture.
- **REG3_NOWIN's "discovered" sits exactly on the 5-vs-0 threshold.** Treat it as weak.
- **The delta arms are descriptive.** Their 10 seeds are fixed in the spec, but the readings were defined for the
  Hebbian arms only. I added BASELINE_D and REG3_D to the delta phase as paired references.
- **Two interpretations I fixed in the spec:** "helps" as p ≤ p(4 vs 0), and the hinge as the mean over the three
  layers' gates.

## CHECKs (48, asserted, all ok; X's arm A bit for bit)

- S73's 37.
- Each device with its switch off (λ 0; gate lr 1e-3 from update 1; window on) equals REG3|2100 through 2400 bit for bit:
  curve, statistics and probe.
- The hinge's gradient through _Inject equals autograd of λ·mean_ℓ relu(η²_key − τ) exactly.
- η²_key at RandHeaderTask's KEY positions equals a hand computation.
- The hinge weight is 1.0 at updates 1, 2399 and 2400, and 0 at 2401 and 2402.
- The slow gate's lrs [gate, rest] are [1e-4, 1e-4] at 1 and 1200, [1e-3, 1e-4] at 1201 and 2400, and [1e-3, 1e-3] at
  2401; its groups are as specified.
- NOWIN has no gate_conv, A_v, W_in or W_g, is causal, and with the window on equals REG3 (parameters and logits).

## A pre-registered main-line test of REG3_KEYHINGE (proposal, for the user)

Test the register gate shared by all three layers with the key-term hinge for updates 1-2400 (REG3_KEYHINGE) on the main
line's Hebbian memory, RandHeaderTask-lite, 24000 updates, SLOW, on fresh seeds on both machines: 20 per machine from X's
and L's own blocks, unused so far (for example X 540-559 and L 1540-1559), with the spec committed to `specs/` first.

Arms, paired by seed:
- BASELINE (LOCAL3+SLOW).
- REG3 (no hinge).
- REG3_KEYHINGE.
- LOCAL3_KEYHINGE: the hinge on the window gate without the register. This control was not run here. It separates the
  hinge from the register.
- REG3_NUDGE and the perfect gate on 2 seeds per machine, for validity.

Primary claims, per machine, exact one-sided McNemar on ROUTED-BOUND at α = 0.05:
- C1: REG3_KEYHINGE beats BASELINE.
- C2: REG3_KEYHINGE beats REG3.
- C3: REG3_KEYHINGE beats LOCAL3_KEYHINGE.
- A rate claim: REG3_KEYHINGE ROUTED-BOUND ≥ 18/20 on each machine.

The replication claim is C1-C3 holding on both machines, with a pooled 40-seed test as secondary. Descriptive extras:
- η²_key and β^R by role over the first 3600 updates;
- the same arms on delta memory;
- a four-stream version (S = 4, k = 16, grouped and header) to see whether the hinge-plus-register carries past two
  streams.

## Runtime

Phase 1 (104 Hebbian runs) and phase 2 (32 delta runs) together took 222.1 min on 4 workers. No container restart. The
progress watcher hit its 2-hour background limit once and was restarted, with no effect on the runs.
