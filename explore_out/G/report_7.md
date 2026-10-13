# Session G — report 7 (S87: hinges that read no task labels). EXPLORATORY, not a result

Branch `claude/explore-G`; Intel Xeon @ 2.10GHz, torch 2.14.0+cu130, 1 thread per run. RandHeaderTask-lite (F's S70),
Hebbian, k = 2, no conv, 24000, SLOW; the model of every new arm is S72's REG3. Seeds 2700-2719 (Hebbian) and 2720-2729
(delta); no record on any branch used them. Spec, readings, the frequency rule and CHECKs committed at 3c89f87 before any
run. 142/142 runs ok, none retried; 212 min on 4 workers.

## Counts (ROUTED-BOUND = BOUND ∧ F's ROUTED*@end; Wilson 95%; band)

| arm (what it reads) | ROUTED-BOUND | BOUND | unbound | transitions | commitment at 2400 (median) |
|---|---|---|---|---|---|
| BASELINE (LOCAL3+SLOW; control rerun) | **0/20** [0.00, 0.16] NEVER | 0/20 | KEY 17, S-PARTIAL 3 | – | 0.873 |
| REG3_KEYHINGE (key positions + identities; control rerun) | **20/20** [0.84, 1.00] RELIABLE | 20/20 | – | 3600 (3600-9600) | 0.934 |
| **REG3_FREQHINGE** (a frequency table of token ids) | **17/20** [0.64, 0.95] MAJORITY | 17/20 | KEY 3 | 3600 (3600-6000) | 0.591 |
| REG3_SELFKL (nothing) | **0/20** NEVER | 0/20 | KEY 18, OTHER 2 | – | **0.0001** |
| REG3_TEMP (nothing; T = 4 for 1-2400) | **3/20** [0.05, 0.36] MINORITY | 3/20 | KEY 16, OTHER 1 | 3600 (2400-7200) | 0.923 |
| ORACLE_ALL (2700-2701; validity) | 2/2 | 2/2 | – | 1200 | – |
| *delta (β 1, L2 keys), 2720-2729* | | | | | |
| **REG3_FREQHINGE_D** | **10/10** [0.72, 1.00] RELIABLE | 10/10 | – | 2400 (2400-3600) | |
| BASELINE_D | 0/10 NEVER | 3/10 (key splits) | KEY 7 | – | |
| REG3_D | 0/10 NEVER | 2/10 | KEY 8 | – | |
| REG3_KEYHINGE_D | 10/10 RELIABLE | 10/10 | – | 2400 (2400-3600) | |

Commitment is gate_stats_k's spread: the returned gate's mean total-variation distance from uniform, scaled to [0, 1].
For REG3 models it is layer 3's gate; for BASELINE it is the window gate.

## Readings (pre-fixed) and which apply

Phase 0: S82's BASELINE and REG3_KEYHINGE (and, before phase 2, BASELINE_D, REG3_D, REG3_KEYHINGE_D) at seed 2100
reproduce their records bit for bit through 2400. Validity: S71's ORACLE_ALL 10/10 stands, and the rerun on 2700-2701
bound 2/2. The control REG3_KEYHINGE beats BASELINE 20 / 0 again on these seeds.

| reading | test | result |
|---|---|---|
| discovered? | FREQHINGE vs BASELINE 17 / 0 (p 8e-6); SELFKL 0 / 0; TEMP 3 / 0 (p 0.125) | **FREQHINGE only** |
| "a frequency hinge replaces the key hinge" | FREQHINGE discovered and within 1 pair of REG3_KEYHINGE (KH only 3, FREQ only 0) | **does not apply** (two pairs short) |
| "a hinge that reads nothing suffices" | SELFKL or TEMP discovered and within 1 pair | does not apply |
| "clamps" | (arm only) − (BASELINE only) ≤ 1 and commitment at 2400 below BASELINE's | **SELFKL** (0 / 0; 0.0001 vs 0.873). TEMP no (0.923 > 0.873); FREQHINGE no (17 / 0) |
| "the identity term is still needed" | REG3_KEYHINGE beats every new arm by ≥ 4 vs 0 | does not apply (vs FREQHINGE 3 / 0, p 0.125; vs SELFKL 20 / 0, vs TEMP 17 / 0) |
| delta (rule: the least-reading new arm discovered) | REG3_FREQHINGE | FREQHINGE_D 10/10, against BASELINE_D 0/10 (10 / 0) and REG3_D 0/10 (10 / 0); equal to REG3_KEYHINGE_D (0 / 0); descriptive |

So the frequency hinge does most of the key hinge's work on both memories without being given the key labels. It is not
shown to replace the key term (3 / 0 short of the ≤ 1 rule), and the key term is not shown to be needed over it
(p 0.125).

## What happened (exploratory)

- **The frequency table is not what the premise assumed.** On this task the two context types are the most frequent
  (0.114 each), the four keys next (0.102 each), the values rare (0.023 each). The largest-ratio cut selects all six
  frequent types. That covers all 8 of the key term's positions, plus the query's KEY and the 5 CTX positions: 14 per
  sequence.
- **Why FREQHINGE works, and why it lags.**
  - It fires 53-115 times in updates 1-2400 (median 80) and drives its η² to 0 by update 300, like the key term.
  - It also groups the 5 CTX positions by CTX identity, which is grouping by stream at the stream tokens. So it penalises
    the gate's dependence on the stream token there.
  - The register's write at CTX separates more slowly: β^R CTX / KEY / VAL at 2400 is 0.24 / 0.19 / 0.19, against the key
    term's 0.32 / 0.20 / 0.21. The gate is less committed at 2400 (0.59 against 0.93).
  - 17 of 20 still route and bind at 3600-6000. The three that do not (2700, 2709, 2713) key-split.
  - On delta memory it routes and binds 10/10, all at 2400-3600.
- **SELFKL clamps the gate, like S83's KL terms.** Its value is 0.0000 from update 300, the commitment is 0.0003 at 1200
  and 0.0001 at 2400, and β^R stays at its initial 0.10 at every role through 2400. Penalising every within-sequence
  difference of the gate also removes the difference by stream, which the register needs to learn its write. After the
  window the key split forms (KEY 18).
- **The temperature does not hold the gate back.** Even at T = 4 the gate commits by 1200 (0.84, as BASELINE's 0.85).
  Its logits simply grow fourfold. β^R stays flat (0.30 / 0.33 / 0.30 at 1200), and 16/20 key-split. It routes and binds
  3/20 (2708, 2716, 2719), REG3's rate in S82 (2/20).
- **Plateaus.** Every unbound arm sits at 0.45-0.60 across the 12 (stream, block, pair) cells: chance between streams.
  The routed runs' stacks do not carry the stream (in-run probe ≈ 0.50 at L2, L3, final, K and V).

## CHECKs (S83's 61 + 12 new, asserted, all ok; X's arm A bit for bit)

- Each new device with its switch off equals REG3|2700 through 2400 bit for bit (FREQHINGE, SELFKL, TEMP).
- The frequency table is computed from token ids only. freq_types has a single parameter, asserts a LongTensor, and
  refuses a float tensor. Its frequent types are {0, 1, 2, 3, 4, 5}.
- The coincidence count over the 4096-sequence sample: 14 frequent positions per sequence, 8 key-term positions, all 8
  frequent.
- Grouping by token id at the key positions equals the key term's grouping by key id.
- The freq and selfkl terms' gradients reach A_r, A_v, A_R, W_g2, the register, the window and the embedding (through the
  gate's input), and nothing in the stack.
- TEMP's gates are flatter inside the window and equal REG3's after it, with no parameter added.
- The schedules over real training forwards: weight 1 / T 4 through 2400; 0 / T 1 at 2401 and 2402.

## For the main line

FREQHINGE is the first S82-family device that reads no labelled task structure. What it does read is a frequency table
of token ids and, through it, the stream tokens' own positions. It is 17/20 here against the key term's 20/20. It fits the
spec draft (`spec_draft_register.md`) as a fifth arm, or as a replacement for REG3_KEYHINGE if the claim is meant to need
no marked keys. Its caveat: on this task the frequent set is the CTX and key types, so it reads roughly what the key term
reads, plus the stream tokens. A language-model version would need its own frequency cut, and that cut would select
different tokens.
