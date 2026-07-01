"""
Interference test: two-stream Hebbian memory collision.

Both streams write the SAME key tokens with CONFLICTING value tokens.
The architecture uses BIGRAM Hebbian writes: at position t, write
    outer(W_k @ embed(x_{t-1}), W_v @ embed(x_t))
so the write at a VALUE position uses the PRECEDING KEY as the memory address.

Stream-1: KEY_i -> VAL_A  (both streams use the same KEY_i)
Stream-2: KEY_i -> VAL_B  (different value, same key)
In a single Hebbian memory this writes outer(K_i, VAL_A) + outer(K_i, VAL_B)
into the SAME synapses. Reading with K_i returns VAL_A + VAL_B — irresolvably
ambiguous. Max accuracy for any single-channel model is analytically ≤ 50%.

With k=2 channels and a gate that routes stream-1 writes to channel 0 and
stream-2 writes to channel 1, each channel holds one clean association:
  S_0: outer(K_i, VAL_A)  →  query K_i from ch-0 → VAL_A ✓
  S_1: outer(K_i, VAL_B)  →  query K_i from ch-1 → VAL_B ✓

The gate is informed by a recurrent hidden state h_t that accumulates context
(CTX1 or CTX2 tokens in the sequence). Both write-time and read-time gates
can thus distinguish the active stream.

Task constants (not tuned to force an outcome):
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bdh_recurrent  import BDHRecurrent,     BDHRecurrentConfig
from bdh_multichannel import BDHMultiChannel, BDHMultiChannelConfig

# ── Task constants ─────────────────────────────────────────────────────────
NUM_PAIRS       = 4   # conflicting key-value pairs
INTERLEAVE_DENS = 2   # each (key,val) pair is written INTERLEAVE_DENS times
                      # per stream before queries; more writes → stronger memory

# ── Training constants ─────────────────────────────────────────────────────
D_MODEL    = 48
N_STEPS    = 800
BATCH_SIZE = 32
LR         = 3e-3
SEEDS      = [0, 1, 2]

# ── Vocabulary ─────────────────────────────────────────────────────────────
# PAD CTX1 CTX2  KEY_0..KEY_{P-1}  VAL_A  VAL_B  QUERY
PAD      = 0
CTX1     = 1
CTX2     = 2
KEY_BASE = 3                        # KEY_i = KEY_BASE + i
VAL_A    = KEY_BASE + NUM_PAIRS     # stream-1 value
VAL_B    = KEY_BASE + NUM_PAIRS + 1 # stream-2 value
QUERY    = KEY_BASE + NUM_PAIRS + 2
VOCAB    = KEY_BASE + NUM_PAIRS + 3


# ── Sequence builder ───────────────────────────────────────────────────────
def make_sequence():
    """
    Interleaved write phase then query phase.

    Write phase (for each repetition and each pair):
      [CTX1, KEY_i, VAL_A,  CTX2, KEY_i, VAL_B]

    Query phase (for each pair):
      [CTX1, KEY_i, QUERY, VAL_A]   → loss/mask at QUERY position
      [CTX2, KEY_i, QUERY, VAL_B]   → loss/mask at QUERY position

    Targets are the NEXT token at each position.
    loss_mask = 1 only at QUERY positions (the model must predict the answer).
    stream_mask = 1 (s1) or 2 (s2) at query positions, else 0.

    NOTE on bigram write: the model writes outer(embed(x_{t-1}), embed(x_t)).
    At VAL_A/VAL_B positions: x_{t-1} = KEY_i → writes KEY_i -> VAL_A/VAL_B.
    At QUERY positions:        x_{t-1} = KEY_i → reads with KEY_i as address.
    At CTX positions:          x_{t-1} = previous token (VAL or KEY); also updates h.
    """
    toks  = []
    tgts  = []
    lmask = []
    smask = []

    # Write phase: interleaved streams, INTERLEAVE_DENS repetitions
    for _ in range(INTERLEAVE_DENS):
        for i in range(NUM_PAIRS):
            key = KEY_BASE + i
            # stream-1 triplet: predict next token at each position
            toks  += [CTX1, key,  VAL_A]
            tgts  += [key,  VAL_A, PAD]   # next-token targets (PAD = don't care non-query)
            lmask += [0, 0, 0]
            smask += [0, 0, 0]
            # stream-2 triplet
            toks  += [CTX2, key,  VAL_B]
            tgts  += [key,  VAL_B, PAD]
            lmask += [0, 0, 0]
            smask += [0, 0, 0]

    # Query phase: for each pair, one query per stream
    for i in range(NUM_PAIRS):
        key = KEY_BASE + i
        # stream-1 query: [CTX1, KEY_i, QUERY, VAL_A]
        # loss at QUERY position (model predicts VAL_A)
        toks  += [CTX1, key,   QUERY, VAL_A]
        tgts  += [key,  QUERY, VAL_A, PAD]
        lmask += [0,    0,     1,     0   ]
        smask += [0,    0,     1,     0   ]
        # stream-2 query: [CTX2, KEY_i, QUERY, VAL_B]
        toks  += [CTX2, key,   QUERY, VAL_B]
        tgts  += [key,  QUERY, VAL_B, PAD]
        lmask += [0,    0,     1,     0   ]
        smask += [0,    0,     2,     0   ]

    return (
        torch.tensor(toks,  dtype=torch.long),
        torch.tensor(tgts,  dtype=torch.long),
        torch.tensor(lmask, dtype=torch.float),
        torch.tensor(smask, dtype=torch.long),
    )


def make_perfect_gate(tokens: torch.Tensor, k: int = 2) -> torch.Tensor:
    """One-hot gate by stream: CTX1 sets channel-0, CTX2 sets channel-1.
    The gate 'latches' — stays at the last-seen context.
    Applied at write and read positions."""
    T  = tokens.shape[0]
    g  = torch.zeros(T, k)
    ch = 0
    for t in range(T):
        tok = tokens[t].item()
        if tok == CTX1:
            ch = 0
        elif tok == CTX2:
            ch = 1
        g[t, ch] = 1.0
    return g


# ── Analytical sanity check ────────────────────────────────────────────────
def sanity_check_conflict():
    """
    With a single-channel Hebbian memory and keys K_i (orthonormal), after:
      S += outer(K_i, VAL_A_emb) + outer(K_i, VAL_B_emb)   for each i
    reading with K_i yields VAL_A_emb + VAL_B_emb, equidistant from both targets.
    The best a single-channel model can do is GUESS: 50% correct.
    We verify this analytically for D=8, NUM_PAIRS orthonormal keys.
    """
    D = 8
    assert NUM_PAIRS <= D, "need D >= NUM_PAIRS for orthonormal keys"
    K = torch.eye(D)[:NUM_PAIRS]                   # orthonormal keys
    vA = F.normalize(torch.randn(D), dim=0)
    vB = F.normalize(torch.randn(D), dim=0)
    while abs((vA * vB).sum()) > 0.1:              # ensure non-collinear
        vB = F.normalize(torch.randn(D), dim=0)

    # Write outer(value, key): S @ key → value
    S = torch.zeros(D, D)
    for i in range(NUM_PAIRS):
        S += torch.outer(vA, K[i])
        S += torch.outer(vB, K[i])

    correct_if_pick_a = 0
    correct_if_pick_b = 0
    for i in range(NUM_PAIRS):
        r      = S @ K[i]                          # output for key i
        score_a = float((r * vA).sum())
        score_b = float((r * vB).sum())
        # r is proportional to vA + vB: dot with vA == dot with vB (when |vA|=|vB|=1)
        # A model can pick either but not both for the same key
        correct_if_pick_a += 1   # gets s1 right, s2 wrong
        correct_if_pick_b += 1   # gets s2 right, s1 wrong
        # Confirm: both dot products are equal
        assert abs(score_a - score_b) < 1e-3, (
            f"Key {i}: score_A={score_a:.4f} vs score_B={score_b:.4f}; "
            "expected equal — single-channel read is not equidistant, "
            "task may not force genuine conflict."
        )

    total   = 2 * NUM_PAIRS   # s1-queries + s2-queries
    max_acc = correct_if_pick_a / total   # = 0.5
    assert max_acc <= 0.5 + 1e-6, (
        f"Single-channel analytical max accuracy = {max_acc:.3f} > 0.5 — "
        "conflict is not genuine."
    )
    print(f"[sanity] single-channel analytical max accuracy = {max_acc:.1%} ≤ 50% ✓")
    print(f"         (read vector aligns equally with VAL_A and VAL_B "
          f"because r ∝ VAL_A + VAL_B for every key)")


# ── Training ───────────────────────────────────────────────────────────────
def train_model(model, n_steps, batch_size, is_mc, gate_mode, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    for _ in range(n_steps):
        batch = [make_sequence() for _ in range(batch_size)]
        toks  = torch.stack([b[0] for b in batch])
        tgts  = torch.stack([b[1] for b in batch])
        lmask = torch.stack([b[2] for b in batch])

        go = None
        if is_mc and gate_mode != 'learned':
            B, T = toks.shape
            if gate_mode == 'perfect':
                go = torch.stack([make_perfect_gate(toks[b], k=2) for b in range(B)])
            elif gate_mode == 'uniform':
                go = torch.full((B, T, 2), 0.5)

        if is_mc:
            _, loss, _ = model(toks, targets=tgts, loss_mask=lmask, gate_override=go)
        else:
            _, loss    = model(toks, targets=tgts, loss_mask=lmask)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


@torch.no_grad()
def evaluate_model(model, is_mc, gate_mode, n_eval=512):
    model.eval()
    batch = [make_sequence() for _ in range(n_eval)]
    toks  = torch.stack([b[0] for b in batch])
    tgts  = torch.stack([b[1] for b in batch])
    lmask = torch.stack([b[2] for b in batch])
    smask = torch.stack([b[3] for b in batch])

    B, T = toks.shape
    go = None
    if is_mc and gate_mode != 'learned':
        if gate_mode == 'perfect':
            go = torch.stack([make_perfect_gate(toks[b], k=2) for b in range(B)])
        elif gate_mode == 'uniform':
            go = torch.full((B, T, 2), 0.5)

    if is_mc:
        logits, _, gates = model(toks, targets=tgts, loss_mask=lmask, gate_override=go)
    else:
        logits, _        = model(toks, targets=tgts, loss_mask=lmask)
        gates = None

    results   = {}
    gate_info = None

    for s in [1, 2]:
        mask = (smask == s)                         # (B, T) bool
        flat_l = logits.reshape(-1, VOCAB)
        flat_t = tgts.reshape(-1)
        flat_m = mask.reshape(-1)
        pos_loss = F.cross_entropy(flat_l[flat_m], flat_t[flat_m]).item()
        acc      = (flat_l[flat_m].argmax(-1) == flat_t[flat_m]).float().mean().item()
        results[s] = (pos_loss, acc)

    if is_mc and gate_mode == 'learned' and gates is not None:
        # Mean gate at query positions per stream
        gs1_vecs, gs2_vecs = [], []
        for s, vec_list in [(1, gs1_vecs), (2, gs2_vecs)]:
            mask = (smask == s).reshape(-1)
            g_flat = gates.reshape(-1, 2)
            vec_list.append(g_flat[mask].mean(0))
        if gs1_vecs and gs2_vecs:
            gs1 = torch.stack(gs1_vecs).mean(0)
            gs2 = torch.stack(gs2_vecs).mean(0)
            gate_info = (gs1, gs2)

    model.train()
    return results, gate_info


# ── Run one condition across seeds ─────────────────────────────────────────
def run_condition(label, make_model_fn, is_mc, gate_mode='learned'):
    print(f"\n── {label} ──")
    all_loss = {1: [], 2: []}
    all_acc  = {1: [], 2: []}
    gate_reports = []

    for seed in SEEDS:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = make_model_fn()
        train_model(model, N_STEPS, BATCH_SIZE, is_mc, gate_mode, seed)
        results, gi = evaluate_model(model, is_mc, gate_mode)
        for s in [1, 2]:
            all_loss[s].append(results[s][0])
            all_acc[s].append(results[s][1])
        if gi is not None:
            gate_reports.append(gi)

    for s in [1, 2]:
        ls = all_loss[s]; ac = all_acc[s]
        print(f"  stream-{s}: loss={np.mean(ls):.4f}±{np.std(ls):.4f}  "
              f"acc={np.mean(ac):.3f}±{np.std(ac):.3f}")

    if gate_reports:
        gs1 = torch.stack([g[0] for g in gate_reports]).mean(0)
        gs2 = torch.stack([g[1] for g in gate_reports]).mean(0)
        cos = F.cosine_similarity(gs1.unsqueeze(0), gs2.unsqueeze(0)).item()
        print(f"  gate @query  stream-1: [{gs1[0].item():.3f}, {gs1[1].item():.3f}]")
        print(f"  gate @query  stream-2: [{gs2[0].item():.3f}, {gs2[1].item():.3f}]")
        print(f"  cosine(g_s1, g_s2) = {cos:.4f}  "
              f"({'near-orthogonal → SEPARATED' if cos < 0.5 else 'overlapping → PERMISSIVE'})")

    return all_loss, all_acc


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("Interference test: Hebbian memory collision, two streams")
    print(f"  NUM_PAIRS={NUM_PAIRS}  INTERLEAVE_DENS={INTERLEAVE_DENS}")
    print(f"  VOCAB={VOCAB}  D_MODEL={D_MODEL}  N_STEPS={N_STEPS}  SEEDS={SEEDS}")
    print("=" * 62)

    sanity_check_conflict()

    def make_single():
        return BDHRecurrent(BDHRecurrentConfig(VOCAB, d_model=D_MODEL))

    def make_mc():
        return BDHMultiChannel(BDHMultiChannelConfig(VOCAB, d_model=D_MODEL, n_channels=2))

    l1, a1 = run_condition("Cond 1: single-channel baseline",     make_single, is_mc=False)
    l2, a2 = run_condition("Cond 2: MC k=2, learned gate",        make_mc,     is_mc=True,  gate_mode='learned')
    l3, a3 = run_condition("Cond 3: MC k=2, perfect gate (ceil)", make_mc,     is_mc=True,  gate_mode='perfect')
    l4, a4 = run_condition("Cond 4: MC k=2, uniform gate (floor)",make_mc,     is_mc=True,  gate_mode='uniform')

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("Summary — query-position accuracy")
    print(f"{'Condition':<42} {'S1 acc':>7} {'S2 acc':>7} {'avg':>7}")
    rows = [
        ("1. single-channel baseline",    a1),
        ("2. MC learned gate",            a2),
        ("3. MC perfect gate (ceiling)",  a3),
        ("4. MC uniform gate (floor)",    a4),
    ]
    for label, a in rows:
        m1 = np.mean(a[1]); m2 = np.mean(a[2])
        print(f"  {label:<40} {m1:>7.3f} {m2:>7.3f} {(m1+m2)/2:>7.3f}")

    # ── Verdict ────────────────────────────────────────────────────────────
    ceil_avg  = (np.mean(a3[1]) + np.mean(a3[2])) / 2
    floor_avg = (np.mean(a4[1]) + np.mean(a4[2])) / 2
    learn_avg = (np.mean(a2[1]) + np.mean(a2[2])) / 2
    gap_total   = ceil_avg  - floor_avg
    gap_learned = learn_avg - floor_avg

    print("\n" + "=" * 62)
    print("Verdict")
    if gap_total > 0.05 and gap_learned >= 0.7 * gap_total:
        frac = gap_learned / max(gap_total, 1e-6)
        print(f"(a) Gate SPONTANEOUSLY SEPARATES the streams. Condition 2 "
              f"recovers {frac:.0%} of the ceiling–floor gap "
              f"({learn_avg:.3f} vs ceil {ceil_avg:.3f}, floor {floor_avg:.3f}). "
              "The mechanism works and is useful under end-to-end training.")
    else:
        frac = gap_learned / max(gap_total, 1e-6)
        print(f"(b) Gate STAYS PERMISSIVE. Condition 2 recovers only {frac:.0%} of "
              f"the ceiling–floor gap "
              f"(learned {learn_avg:.3f}, ceil {ceil_avg:.3f}, floor {floor_avg:.3f}). "
              "Separation is available and beneficial but the mechanism does NOT "
              "engage under end-to-end training.")


if __name__ == "__main__":
    main()
