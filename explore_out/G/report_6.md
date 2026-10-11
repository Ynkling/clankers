# Session G — report 6 (S83: what the hinge must read, the register, the window). EXPLORATORY, not a result

Branch `claude/explore-G`; Intel Xeon @ 2.10GHz, torch 2.14.0+cu130, 1 thread per run. RandHeaderTask-lite (F's S70),
Hebbian, k = 2, no conv, 24000, SLOW; seeds 2100-2119, paired with S82's records. Spec, readings, phase rules and CHECKs
committed at b2f9907 before any run; child code SHA in `s83_results.json`. 82/82 runs ok, none retried; 114 min on 4
workers.

## Counts (ROUTED-BOUND = BOUND ∧ F's ROUTED*@end; Wilson 95%; band). S82's records are marked (S82).

| arm | what the term reads | BOUND | **ROUTED-BOUND** | unbound | transitions |
|---|---|---|---|---|---|
| BASELINE (S82) | – | 3/20 | **0/20** [0.00, 0.16] NEVER | KEY 17 | key splits at 10800-16800 |
| REG3 (S82) | – | 2/20 | **2/20** [0.03, 0.30] MINORITY | KEY 16 | 3600 |
| REG3_KEYHINGE (S82) | key positions + identities | 20/20 | **20/20** [0.84, 1.00] RELIABLE | – | 3600 (3600-6000) |
| LOCAL3_KEYHINGE (no register) | key positions + identities | 5/20 | **0/20** [0.00, 0.16] NEVER | KEY 14, S-PARTIAL 1 | 8400-24000 (key splits) |
| REG3_KEYHINGE_POS (KL from uniform at KEY) | key positions | 1/20 | **1/20** [0.01, 0.24] MINORITY | KEY 18, OTHER 1 | 6000 |
| REG3_FLAT (KL at every body position) | nothing | 0/20 | **0/20** [0.00, 0.16] NEVER | KEY 18, OTHER 1, S-PARTIAL 1 | – |
| REG3_KEYHINGE_1200 (key term, updates 1-1200) | key positions + identities | 18/20 | **18/20** [0.70, 0.97] RELIABLE | KEY 2 | 3600 (3600-6000) |
| REG3_NUDGE (2100-2101; validity) | – | 2/2 | 2/2 | – | 2400 |

## Readings (pre-fixed) and which apply

Phase 0: S82's BASELINE, REG3, REG3_KEYHINGE, BASELINE_D, REG3_D and REG3_KEYHINGE_D at seed 2100 reproduce their records
bit for bit through 2400 under the current code. No rerun was needed, and S82's records are the references. Validity: S71's
ORACLE_ALL 10/10 and S82's rerun 2/2 stand; REG3_NUDGE 2/2.

| question | test (paired, exact one-sided McNemar) | result |
|---|---|---|
| discovered? | LOCAL3_KEYHINGE vs BASELINE 0 / 0; POS 1 / 0 (p 0.5); FLAT 0 / 0; KEYHINGE_1200 18 / 0 (p 4e-6) | only REG3_KEYHINGE_1200 |
| the register | REG3_KEYHINGE beats LOCAL3_KEYHINGE 20 / 0 (p 1e-6) | **"the register is needed"** |
| key identity | REG3_KEYHINGE beats POS 19 / 0 (p 2e-6) | **"the identity term is needed"** |
| structure | FLAT not discovered | "no structure need be read" does not apply |
| the window | REG3_KEYHINGE only 2, KEYHINGE_1200 only 0 (needs ≤ 1) | "the short window suffices" does not apply (one pair short) |
| delta (phase 2) | rule: the least-reading arm of (1)-(2) discovered on Hebbian memory | none discovered, so no delta runs |

## What happened (exploratory)

- **Without the register the hinge has nothing to release.** On the window gate the key term fires only 1-15 times (η²_key
  0.18 at update 1, 0 from update 300): there is little key split to remove. The window still cannot see the header, so
  the gate stays unrouted (margin 0.00). 14/20 key-split later, and 5 bind by key splits (transitions 8400-24000). The
  hinge works by keeping the gate's key-structured variation down while the register's write at CTX grows. With nothing
  to carry the stream, there is nothing to grow.
- **The KL terms clamp the gate to uniform and starve the register.** The pos and flat terms are positive on every update
  of the window (2400 firings), but their value is 0.0000 from update 300. The gate is held at exactly uniform, so no
  routing signal reaches the register: β^R stays at its initial 0.10 at every role through 1200 (FLAT through 2400).
  After the window the key split forms (KEY 18 in each).
  - My spec's caveat expected the opposite failure: the absolute KL term too weak near uniform. It was too strong. It
    removes all of the gate's variation, including the stream-structured variation the register needs to learn from.
  - The key term is a ratio (η²) and removes only the share of variation organised by key. The gate is free to commit by
    stream, and β^R at CTX separates while the term is on (0.32 vs 0.18-0.21 by 2400).
- **The 1200-update window nearly suffices.** 18/20 route and bind at 3600, as with 2400; β^R separates on the same
  schedule (0.32 / 0.18 / 0.21 at 2400). The two failures (seeds 2103 and 2112) key-split after the term stopped at 1200, while the
  stack was still at its slow rate. POS's one routed run is 2108, which REG3 and NOWIN also route.
- **Plateaus.** Every unbound arm sits at 0.45-0.57 in all 12 (stream, block, pair) cells: chance between streams.

## CHECKs (S82's 48 + 13 new, asserted, all ok; X's arm A bit for bit)

- Each new term with its switch off equals S82's REG3|2100 through 2400 bit for bit; LOCAL3_KEYHINGE with the term off
  equals S82's BASELINE|2100.
- The key term in S83's class equals S82's REG3_KEYHINGE|2100 through 2400 bit for bit.
- Gradients reach only the gate's side:
  - The pos and flat terms reach A_r, A_v, A_R, W_g2, the register (W_R, w_R, b_R), the window and the embedding (through
    the gate's input, |grad| 1-4e-4), and nothing in the stack.
  - The key term on LOCAL3 reaches W_in, W_g, the window and the embedding.
- The terms' values equal a hand computation (KL of the uniform gate 0).
- The schedules: weight 1 through W, 0 at W + 1 and W + 2.
- LOCAL3_KEYHINGE has BASELINE's parameters and logits.

## For the main line

`spec_draft_register.md` (this directory) drafts the test: REG3_KEYHINGE against BASELINE, LOCAL3_KEYHINGE and REG3,
with REG3_NUDGE and the perfect gate as validity, on X 600-619 and L 1600-1619 (no record on any branch uses them). The
POS arm is not in it: it was not discovered.
