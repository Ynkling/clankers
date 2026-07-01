"""
test_interference.py — Two-stream Hebbian memory interference test.

Both streams share the SAME P key tokens but write different values:
  stream-1: KEY_i → VAL_A  (or VAL_B if coin-flipped this sequence)
  stream-2: KEY_i → VAL_B  (or VAL_A)

Pairs from both streams are interleaved into a single sequence.
Context tokens (CTX1, CTX2) mark which stream each write belongs to.
At query time, a context token specifies which stream to retrieve from.

Single-channel analytical ceiling: 50%
  Hebbian state accumulates outer(VAL_A, KEY_i) + outer(VAL_B, KEY_i)
  for every key i (from both streams), so S @ KEY_q proportional to VAL_A + VAL_B,
  which has equal projection on VAL_A and VAL_B regardless of assignment.

Conditions:
  1. single-channel baseline           (expected ~50%)
  2. MC k=2, learned gate              (does gate spontaneously separate?)
  3. MC k=2, perfect gate (ceiling)    (context-latching, analytical upper bound)
  4. MC k=2, uniform gate (floor)      (g=[0.5,0.5] always, same collision as cond 1)
"""

import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from bdh_recurrent import bdh_layer_parallel, bdh_layer_mc_parallel, ln

# Hyperparameters
P     = 4      # key-value pairs per stream
k     = 2      # MC channels
D     = 32     # embedding dim
N     = 64     # sparse feature dim
BATCH = 32
ITERS = 1200
LR    = 4e-3
SEEDS = [0, 1, 2]

# Sequence layout (BLOCK = 6P + 3 = 27 for P=4):
#   Write phase: for each pair (in random order):
#     [CTX1, KEY_i, VAL_s1,  CTX2, KEY_i, VAL_s2]   (6 tokens per pair)
#   Query phase: [CTX_q, KEY_q, TARGET]               (3 tokens, loss at last)
BLOCK = 6 * P + 3

# Vocabulary (size = P + 4)
CTX1  = 0
CTX2  = 1
# KEY_i = 2 + i    for i in 0..P-1
VAL_A = 2 + P
VAL_B = 2 + P + 1
V     = 2 + P + 2

def key_tok(i):
    return 2 + i


# Sanity check: analytical proof that single-channel <= 50%
def _run_sanity():
    emb = torch.zeros(V, D)
    for i in range(V):
        emb[i, i % D] = 1.0   # deterministic non-zero embeddings

    va = emb[VAL_A]
    vb = emb[VAL_B]
    S  = torch.zeros(D, D)
    for i in range(P):
        ki = emb[key_tok(i)]
        S = S + torch.outer(va, ki)   # stream-1 write  outer(value, key)
        S = S + torch.outer(vb, ki)   # stream-2 write

    for i in range(P):
        ki = emb[key_tok(i)]
        r  = S @ ki
        sA = (r @ va).item()
        sB = (r @ vb).item()
        assert abs(sA - sB) < 1e-5, \
            f"Key {i}: score_A={sA:.4f} vs score_B={sB:.4f}; expected equal"

    print("[sanity] single-channel analytical max accuracy = 50.0% <= 50% v")
    print("         (read vector aligns equally with VAL_A and VAL_B "
          "because r proportional to VAL_A + VAL_B for every key)")
    print()


# Data generation
def make_batch(B, rng):
    seqs      = []
    q_streams = []
    for _ in range(B):
        flip   = torch.randint(2, (1,), generator=rng).item()
        val_s1 = VAL_A if flip == 0 else VAL_B
        val_s2 = VAL_B if flip == 0 else VAL_A

        perm = torch.randperm(P, generator=rng).tolist()

        seq = []
        for i in perm:
            seq += [CTX1, key_tok(i), val_s1]
            seq += [CTX2, key_tok(i), val_s2]

        q_stream = torch.randint(2, (1,), generator=rng).item()
        q_key    = torch.randint(P, (1,), generator=rng).item()
        q_ctx    = CTX1 if q_stream == 0 else CTX2
        q_val    = val_s1 if q_stream == 0 else val_s2
        seq += [q_ctx, key_tok(q_key), q_val]

        assert len(seq) == BLOCK
        seqs.append(seq)
        q_streams.append(q_stream)

    return (torch.tensor(seqs, dtype=torch.long),
            torch.tensor(q_streams, dtype=torch.long))


# Perfect context-latching gate
def _perfect_gate(inp_seq):
    """inp_seq: (T,) input tokens; returns (T, k) hard gate."""
    T   = len(inp_seq)
    g   = torch.zeros(T, k)
    cur = torch.tensor([0.5, 0.5])
    for t in range(T):
        tok = inp_seq[t].item()
        if tok == CTX1:
            cur = torch.tensor([1.0, 0.0])
        elif tok == CTX2:
            cur = torch.tensor([0.0, 1.0])
        g[t] = cur
    return g


def _mc_fixed_gate_forward(embed, Dx, Dy, E, head, tokens):
    B, T = tokens.shape
    v    = embed(tokens)
    out  = []
    for b in range(B):
        g      = _perfect_gate(tokens[b])
        x      = F.relu(ln(v[b]) @ Dx)
        scores = ((x @ x.T) * (g @ g.T)).tril(diagonal=-1)
        a_ast  = scores @ v[b]
        y      = F.relu(ln(a_ast) @ Dy) * x
        v_next = v[b] + ln(y @ E)
        out.append(v_next)
    return head(torch.stack(out))


# Model classes
class BDHSingle(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.Dx    = nn.Parameter(torch.randn(D, N) * 0.1)
        self.Dy    = nn.Parameter(torch.randn(D, N) * 0.1)
        self.E     = nn.Parameter(torch.randn(N, D) * 0.1)
        self.head  = nn.Linear(D, V, bias=False)

    def forward(self, tokens):
        B, T = tokens.shape
        v    = self.embed(tokens)
        out  = []
        for b in range(B):
            v_next, *_ = bdh_layer_parallel(v[b], self.Dx, self.Dy, self.E)
            out.append(v_next)
        return self.head(torch.stack(out))


class BDHMCMC(nn.Module):
    def __init__(self, uniform_gate=False):
        super().__init__()
        self.embed        = nn.Embedding(V, D)
        self.Dx           = nn.Parameter(torch.randn(D, N) * 0.1)
        self.Dy           = nn.Parameter(torch.randn(D, N) * 0.1)
        self.E            = nn.Parameter(torch.randn(N, D) * 0.1)
        self.head         = nn.Linear(D, V, bias=False)
        self.uniform_gate = uniform_gate
        if uniform_gate:
            self.register_buffer('W_g', torch.zeros(D, k))
        else:
            self.W_g = nn.Parameter(torch.randn(D, k) * 0.1)

    def forward(self, tokens):
        B, T = tokens.shape
        v    = self.embed(tokens)
        out  = []
        for b in range(B):
            v_next, *_ = bdh_layer_mc_parallel(v[b], self.Dx, self.Dy, self.E, self.W_g)
            out.append(v_next)
        return self.head(torch.stack(out))


class BDHMCPerfect(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.Dx    = nn.Parameter(torch.randn(D, N) * 0.1)
        self.Dy    = nn.Parameter(torch.randn(D, N) * 0.1)
        self.E     = nn.Parameter(torch.randn(N, D) * 0.1)
        self.head  = nn.Linear(D, V, bias=False)

    def forward(self, tokens):
        return _mc_fixed_gate_forward(self.embed, self.Dx, self.Dy, self.E,
                                      self.head, tokens)


# Training + evaluation
def run(make_model, seed, collect_gates=False):
    torch.manual_seed(seed)
    model = make_model()
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    rng   = torch.Generator(); rng.manual_seed(seed + 10_000)

    for step in range(ITERS):
        tokens, _ = make_batch(BATCH, rng)
        inp        = tokens[:, :-1]
        tgt        = tokens[:, 1:]
        logits     = model(inp)
        loss       = F.cross_entropy(logits[:, -1, :], tgt[:, -1])
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        tokens, q_streams = make_batch(512, rng)
        inp    = tokens[:, :-1]
        tgt    = tokens[:, 1:]
        logits = model(inp)

        q_logits = logits[:, -1, :]
        q_tgt    = tgt[:, -1]
        preds    = q_logits.argmax(-1)
        correct  = (preds == q_tgt)

        m1 = (q_streams == 0)
        m2 = (q_streams == 1)
        loss_s1 = F.cross_entropy(q_logits[m1], q_tgt[m1]).item()
        loss_s2 = F.cross_entropy(q_logits[m2], q_tgt[m2]).item()
        acc_s1  = correct[m1].float().mean().item()
        acc_s2  = correct[m2].float().mean().item()

        gate_info = None
        if collect_gates and hasattr(model, 'W_g'):
            v_emb  = model.embed(inp)
            g_last = F.softmax(v_emb[:, -1, :] @ model.W_g, dim=-1)
            g_s1   = g_last[m1].mean(0)
            g_s2   = g_last[m2].mean(0)
            cos    = F.cosine_similarity(g_s1.unsqueeze(0), g_s2.unsqueeze(0)).item()
            gate_info = (g_s1.tolist(), g_s2.tolist(), cos)

    return loss_s1, loss_s2, acc_s1, acc_s2, gate_info


if __name__ == "__main__":
    print("=" * 62)
    print("Interference test: Hebbian memory collision, two streams")
    print(f"  NUM_PAIRS={P}  VOCAB={V}  D_MODEL={D}  N_FEAT={N}")
    print(f"  BLOCK={BLOCK}  BATCH={BATCH}  ITERS={ITERS}  SEEDS={SEEDS}")
    print("=" * 62)
    print()

    _run_sanity()

    conditions = [
        ("Cond 1: single-channel baseline",     lambda: BDHSingle(),    False),
        ("Cond 2: MC k=2, learned gate",        lambda: BDHMCMC(False), True),
        ("Cond 3: MC k=2, perfect gate (ceil)", lambda: BDHMCPerfect(), False),
        ("Cond 4: MC k=2, uniform gate (floor)",lambda: BDHMCMC(True),  False),
    ]

    results = {}
    for name, make_model, do_gates in conditions:
        print(f"\n-- {name} --")
        all_l1, all_l2, all_a1, all_a2 = [], [], [], []
        gate_infos = []
        for seed in SEEDS:
            l1, l2, a1, a2, gi = run(make_model, seed, collect_gates=do_gates)
            all_l1.append(l1); all_l2.append(l2)
            all_a1.append(a1); all_a2.append(a2)
            if gi:
                gate_infos.append(gi)

        def ms(xs):
            m = sum(xs)/len(xs)
            s = (sum((x-m)**2 for x in xs)/len(xs))**0.5
            return m, s

        ml1,sl1 = ms(all_l1); ml2,sl2 = ms(all_l2)
        ma1,sa1 = ms(all_a1); ma2,sa2 = ms(all_a2)
        print(f"  stream-1: loss={ml1:.4f}+-{sl1:.4f}  acc={ma1:.3f}+-{sa1:.3f}")
        print(f"  stream-2: loss={ml2:.4f}+-{sl2:.4f}  acc={ma2:.3f}+-{sa2:.3f}")

        if gate_infos:
            gs1 = [sum(gi[0][c] for gi in gate_infos)/len(gate_infos) for c in range(k)]
            gs2 = [sum(gi[1][c] for gi in gate_infos)/len(gate_infos) for c in range(k)]
            avg_cos = sum(gi[2] for gi in gate_infos)/len(gate_infos)
            gs1_str = "[" + ", ".join(f"{v:.3f}" for v in gs1) + "]"
            gs2_str = "[" + ", ".join(f"{v:.3f}" for v in gs2) + "]"
            sep = "SEPARATED" if avg_cos < 0.5 else "not separated"
            print(f"  gate @query  stream-1: {gs1_str}")
            print(f"  gate @query  stream-2: {gs2_str}")
            print(f"  cosine(g_s1, g_s2) = {avg_cos:.4f}  ({sep})")

        results[name] = dict(acc1=ma1, acc2=ma2)

    print()
    print("-- Verdict --")
    cond1 = (results["Cond 1: single-channel baseline"]["acc1"] +
             results["Cond 1: single-channel baseline"]["acc2"]) / 2
    cond2 = (results["Cond 2: MC k=2, learned gate"]["acc1"] +
             results["Cond 2: MC k=2, learned gate"]["acc2"]) / 2
    cond3 = (results["Cond 3: MC k=2, perfect gate (ceil)"]["acc1"] +
             results["Cond 3: MC k=2, perfect gate (ceil)"]["acc2"]) / 2
    cond4 = (results["Cond 4: MC k=2, uniform gate (floor)"]["acc1"] +
             results["Cond 4: MC k=2, uniform gate (floor)"]["acc2"]) / 2

    range_  = cond3 - cond4    # full separation benefit
    lift    = cond2 - cond4    # how much cond 2 improved over floor
    frac    = lift / range_ if range_ > 1e-3 else 0.0

    print(f"  Cond 1 (single-channel): {cond1:.3f}")
    print(f"  Cond 2 (learned gate):   {cond2:.3f}")
    print(f"  Cond 3 (perfect gate):   {cond3:.3f}  <- ceiling")
    print(f"  Cond 4 (uniform gate):   {cond4:.3f}  <- floor")
    print(f"  Lift fraction (cond2-floor)/(ceil-floor) = {frac:.2f}")
    print()

    if frac > 0.7:
        print("(a) The gate SPONTANEOUSLY SEPARATED the streams. "
              f"Condition 2 captures {frac:.0%} of the ceiling benefit. "
              "The mechanism works and is useful under end-to-end training.")
    else:
        print("(b) The gate DID NOT ENGAGE. "
              f"Condition 2 captures only {frac:.0%} of the ceiling benefit "
              f"(cond2={cond2:.3f}, ceil={cond3:.3f}, floor={cond4:.3f}). "
              "The gate mechanism does not spontaneously separate streams "
              "under end-to-end training even when separation is available and beneficial.")
