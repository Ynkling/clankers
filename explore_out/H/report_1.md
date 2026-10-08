# Session H — report 1 (EXPLORATORY, not a result)

## S58 explore_h_collapse — S=4, P=4, k=16, conv, 28800 updates, LOCAL3 + SLOW, seeds 420-459

| arm | BOUND ROUTED | band | Wilson 95% | BOUND (any) | failure classes | MERGED | transition median (range) | unbound@14400 → bound | flat@14400 → bound | BR & 1:1 at KEY | margin (median, end) | ops fired (runs) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SPLIT | 39/40 | RELIABLE | 0.87-1.00 | 40/40 | BOUND-NOT-ROUTED 1 | 0 | 3600 (2400-18000) | 2 → 2 | 2 → 2 | 39/39 | 0.866 | 4 (2) |
| RESET | 39/40 | RELIABLE | 0.87-1.00 | 40/40 | BOUND-NOT-ROUTED 1 | 0 | 3600 (2400-18000) | 2 → 2 | 2 → 2 | 38/39 | 0.862 | 3 (2) |
| GUMBEL_W | 37/40 | RELIABLE | 0.80-0.97 | 38/40 | BOUND-NOT-ROUTED 1, MERGED 2 | 2 | 6000 (4800-8400) | 2 → 0 | 1 → 0 | 37/37 | 0.693 | — |
| GUMBEL_RW | 5/40 | MINORITY | 0.05-0.26 | 6/40 | BOUND-NOT-ROUTED 1, KEY 1, MERGED 33 | 33 | 6000 (4800-6000) | 35 → 1 | 31 → 1 | 5/5 | 0.831 | — |
| SWITCH | 36/40 | RELIABLE | 0.77-0.96 | 40/40 | BOUND-NOT-ROUTED 4 | 0 | 3600 (3600-7200) | 1 → 1 | 1 → 1 | 36/36 | 0.46 | — |
| NONE | 38/40 | RELIABLE | 0.83-0.99 | 40/40 | BOUND-NOT-ROUTED 2 | 0 | 3600 (2400-19200) | 2 → 2 | 2 → 2 | 38/38 | 0.869 | — |

ORACLE (perfect gate ceiling4k16 + conv, seeds 420-421): bound 2/2, transitions {420: 1200, 421: 1200} → screen VALID.

**Paired comparisons** (BOUND ROUTED, 40 seeds; x only / y only; exact one-sided McNemar p for x > y and y > x):

| x vs y | x | y | x only | y only | p(x>y) | p(y>x) |
|---|---|---|---|---|---|---|
| GUMBEL_W vs SPLIT | 37/40 | 39/40 | 1 | 3 | 0.938 | 0.312 |
| GUMBEL_W vs NONE | 37/40 | 38/40 | 2 | 3 | 0.812 | 0.5 |
| RESET vs SPLIT | 39/40 | 39/40 | 0 | 0 | 1 | 1 |
| SWITCH vs NONE | 36/40 | 38/40 | 1 | 3 | 0.938 | 0.312 |
| SPLIT vs NONE | 39/40 | 38/40 | 1 | 0 | 0.5 | 1 |
| RESET vs NONE | 39/40 | 38/40 | 1 | 0 | 0.5 | 1 |
| GUMBEL_RW vs NONE | 5/40 | 38/40 | 2 | 35 | 1 | 5.12e-09 |
| GUMBEL_RW vs SPLIT | 5/40 | 39/40 | 1 | 35 | 1 | 5.38e-10 |
| SWITCH vs SPLIT | 36/40 | 39/40 | 1 | 4 | 0.969 | 0.188 |

**Readings (fixed in explore_h_collapse's docstring before any run):**

- R1 (noise replaces the split): **does not apply (SPLIT only 3 - GUMBEL_W only 1 = 2 [<= 1: False]; GUMBEL_W vs NONE 2 vs 3 [>= 4 vs 0: False])**
- R2 (the split is the reset): **the split is the reset (SPLIT only 0 - RESET only 0 = 0 [<= 1]); uninformative (the split itself does nothing at this configuration: SPLIT vs NONE 1 vs 0)**
- R3 (the balance loss helps): **does not apply (SWITCH vs NONE 1 vs 3 [>= 4 vs 0])**

**Runtime:** 242 runs, 25.0 CPU-hours (1 thread each, 4 workers) on Intel(R) Xeon(R) Processor @ 2.10GHz; code SHA 9c9172a89d60.
**CHECKs:** all passed (2026-10-08 04:25:12 UTC, git 69d01dc, 4.2 min); see explore_out/H/s58.log.

**Which reading applies.** The screen is VALID, but the readings sit on a ceiling, as the docstring said before any run.
NONE bound routed 38/40, so a "> NONE by >= 4 vs 0" result was not possible. R1 and R3 do not apply. R2 holds by its
rule (RESET = SPLIT, 0 vs 0), but by its own pre-fixed qualifier it is uninformative: SPLIT beat NONE on 1 seed and lost
on none. So at four streams with k = 16, this screen cannot tell the split, the reset, the write noise or the balance
loss apart from doing nothing. S58b (eight streams, where the split did matter in batch 17) is where they can differ.

**Unexpected / worth a look**
- **GUMBEL_RW collapses**: 33/40 MERGED (two streams on one channel, held to 28800; held-out accuracy about 0.75, or about
  0.5 in 8 runs), and BOUND ROUTED 5/40, against GUMBEL_W's 37/40. The same noise on the write gate only is harmless. On
  the read side it is not, so Raven's read-side Gumbel does not carry over. This fits the reading notes: our read is
  routed, Raven's is dense.
- **The split's one gain is the reset's.** The trigger fired on the same two seeds in SPLIT and RESET (430, 453; SPLIT 4
  operations, RESET 3). Seed 453 is NONE's late BOUND NOT routed (bound at 15600). There SPLIT and RESET both end BOUND
  ROUTED (transitions 15600 and 18000). Seed 441 is BOUND NOT routed in all three. One seed is not evidence. It is the
  audit 2.3 control, and it points at the reset.
- **SWITCH changes the routing's shape, not its count**: 34 of its 36 BOUND ROUTED runs map streams to different channels
  at KEY positions than at VAL positions. Both maps are one-to-one, but they are different channels (the KEY column counts
  bijectivity, not equality with the VAL map). Elsewhere this happens in 2-5 runs per arm. The query routing margin
  falls to 0.46, from 0.87 for NONE. With the balance loss, the model spreads the gate over 13-16 channels (argmax-used
  on the last training batch) and still binds. The 4 failures are all BOUND NOT routed (accuracy 1.0, VAL map not
  one-to-one). For a k >> S language-model setting, uniform usage pulls against crisp routing; MoM's setting has no such
  routing criterion.
- **Gumbel on the write gate slows the transition** (median 6000 vs 3600) and lowers the margin (0.69 vs 0.87); 2 MERGED.
- **Plateau vs dead**: almost every run was bound by 14400. Of the 2 NONE runs flat at 14400, both bound later, but not
  routed. GUMBEL_RW: 31 flat at 14400 and 1 bound later. Too few flat runs outside GUMBEL_RW to say anything.
- **CHECK amendment** (before any screen run; arms and readings unchanged): GUMBEL_W at noise 0 is LOCAL3's forward bit
  for bit, but its two gate nodes change the backward's float order. Its CHECK is equality with a two-node control (passed).
  The record reproduced bit for bit by the other five disabled paths (batch 16's LOCAL3_SLOW16_A|240) was made on a
  2.80GHz Xeon; X's arm A reproduces on this 2.10GHz CPU.
- Setup: the container had no torch. I installed torch 2.14.0+cu130 (the recorded version) from PyPI.

**Next:** S58b runs with RESET (39/40) and GUMBEL_W (37/40), selected by the fixed rule, against SPLIT at eight streams;
S60 runs alongside.
