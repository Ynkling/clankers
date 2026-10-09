# Session F — report 3 (follow-up: S68 premise, S69 decay on Hebbian, S70 RandHeaderTask-lite). EXPLORATORY, not a result

Branch `claude/explore-F`, code SHA ff908bdbc0b2 (one child for all three screens), Intel Xeon @ 2.10GHz,
torch 2.14.0+cu130, 1 thread per run. 102/102 runs ok, none retried. β is fixed at 1 in every delta arm.
S69's Hebbian arms have no erase, so β plays no part there. Full tables: `premise_tables.md`, `hebb_decay_tables.md`,
`rh_lite_tables.md`.

## Counts (Wilson 95%; band)

| screen / arm | outcome | count | reference | reading |
|---|---|---|---|---|
| S68 one delta channel, grouped, conv, S=4 (300-309, 43200) | BOUND | 2/10 [0.06, 0.51] MINORITY | gated Hebbian S54(b) 19/20 BOUND ROUTED (counts; Fisher p = 6e-5) | |
| S68 one delta channel, S=8 | BOUND | **0/10** [0.00, 0.28] NEVER | gated Hebbian S54(c) 9/10 (Fisher p = 6e-5) | **neither** |
| S68 perfect gate (delta β=1), S=4 / S=8 (300-301) | BOUND | 2/2 / 2/2 (all at 1200) | — | valid |
| S69 one Hebbian channel, header P=8 (350-359, 48000): FIXED / ROUTED / GDN | BOUND | **0/10 / 0/10 / 0/10** | S59's delta channel, same seeds and batches: 3/10 / 9/10 / 8/10 | **the forget gate needs the delta rule** |
| S69 perfect gate, Hebbian (350-351) | BOUND | 1/2 (31200; the other 0.75) | — | valid (≥ 1/2) |
| S70 perfect gate, Hebbian / delta (330-332, 24000) | BOUND | 3/3 / 3/3 | — | phase 1 bound: budget 24000 |
| **S70_HEBB** LOCAL3+SLOW, RandHeaderTask-lite (330-349) | DISCOVERED / BOUND / ROUTED*@end | **0/20** [0.00, 0.16] / 1/20 [0.01, 0.24] / 0/20 | — | **BASELINE for Session G** |
| S70_DELTA (β = 1), same seeds | DISCOVERED / BOUND / ROUTED*@end | 0/20 / 6/20 [0.15, 0.52] / 0/20 | paired BOUND: delta only 5, hebb only 0, p = 0.031 | (printed) |

Failure classes: S70_HEBB KEY 19 (1 BOUND unrouted); S70_DELTA KEY 14 (6 BOUND unrouted). S68 / S69 have k = 1 (no
gate, no classes). Transitions: S68 S=4 18000, 33600; S70_DELTA 3600-16800 (median 5400); S70_HEBB 13200.

## Readings (fixed in the docstrings before any run) and which applies

- **S68 "neither".** One delta channel binds 2/10 at S = 4 and 0/10 at S = 8. "Context-tagged keys scale" needed ≥ 7/10
  at S = 8. "The partition's advantage is in the number of streams" needed ≥ 7/10 at S = 4, and it bound only 2/10. Both
  oracles bind at the first evaluation; the gated Hebbian recipe (recorded, other seeds) binds 19/20 and 9/10.
  Descriptively: S53's stream-tagged keys (8/10 on the two-stream header layout) do not carry to four or eight streams in
  the grouped layout. There the partition is what binds (final accuracy of the single channel 0.22-0.75 at S = 4,
  0.29-0.59 at S = 8).
- **S69 "the forget gate needs the delta rule".** On the same seeds and batches where S59's delta channel bound 3 / 9 / 8
  of 10 (fixed / routed / GDN), the Hebbian channel binds 0 / 0 / 0. The Hebbian channel does not learn S59's reset at
  CTX tokens. Median α at CTX / KEY / VAL in layer 1 is 0.975 / 0.999 / 0.982 (ROUTED) and 0.988 / 0.999 / 0.990 (GDN),
  against 0.26 / 1.00 / 1.00 and 0.18 / 1.00 / 1.00 on delta (S59); its lowest CTX decay is in layer 3 (0.89, 0.68).
  Final accuracy 0.17 (routed, GDN) against 0.22 fixed: the learned decay hurt the Hebbian channel.
- **S70 BASELINE (descriptive).** On RandHeaderTask-lite the window gate with slow memory on the Hebbian memory
  discovers the routing on 0/20 seeds, binds 1/20, and key-splits in 19/20. The gate routes by key (η² by key 1.00,
  by stream 0.00 at KEY and VAL; margin 0.00); final accuracy 0.47-0.52. The delta memory binds 6/20 but routes none
  (also key splits; accuracy 0.74-0.88 when unbound). As in S53, the delta memory binds without the partition.
  **Session G's BASELINE: S70_HEBB DISCOVERED 0/20, BOUND 1/20, ROUTED*@end 0/20, KEY 19.**

## Anything unexpected

- The S69 Hebbian oracle bound only 1 of 2 seeds within 48000 (31200; the other 0.75). S53's Hebbian oracle at the same
  P = 8 bound 3/3 by 24000 on seeds 300-302. The difference is the seeds, or this memory's different numerical path,
  which equals the unconverted model's logits exactly at init.
- S70's task is easy for both perfect gates (bound by 1200 on 5 of 6 seeds), but the window gate fails on both memories.
  LOCAL3's window (t-2..t) sees the block's CTX only at its first pair; from the second pair on (KEY at CTX+3, VAL at
  CTX+4) the cue is out of view, which is S6's header-layout limitation again. Blocks of 1-3 pairs halve that distance
  but do not remove it. The gate falls back to a key split.
- S68's single channel at S = 4 sits near 0.25 on 5 seeds and near 0.75 on 2 (305 0.55, 308 0.74, 309 0.75): partial
  binding of some streams, unlike the Hebbian single channel.

## CHECKs

explore_f_delta_checks passed in full (26/26) at both drives, including the bit-for-bit reproduction of
LOCAL3_SLOW|160 with the delta path disabled. The child's 15 CHECKs passed, among them:
- S69's fixed-decay Hebbian memory equals the unconverted Hebbian model's logits exactly, and the routed decay at
  init equals it to 1.2e-7.
- S69's GDN init equals S59's for each seed.
- RandHeaderTask-lite's structure: the compositions (1,3), (2,2), (3,1); block-length frequencies 0.34 / 0.32 / 0.34;
  one CTX per block; every (stream, key) once.
- The perfect gates are one-hot on each position's stream, with margin 1 and η² by stream 1 at KEY and VAL.

## Runtime

34.9 h of run time. Phase 1 (S70's oracles) 14:20-14:24 UTC, phase 2 14:24-23:11 UTC on 9 October (8.8 h wall on 4
workers). No container restart.
