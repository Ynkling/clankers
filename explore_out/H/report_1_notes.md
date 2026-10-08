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
