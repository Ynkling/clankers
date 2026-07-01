"""
test_interference_auxloss.py — Interference test + auxiliary gate-separation loss.

Identical task and metrics to test_interference.py (two-stream shared-key collision,
per-stream query accuracy, ≥3 seeds, mean±std). Adds an auxiliary loss on the MC gate
to counter routing collapse.

Auxiliary loss — pairwise context-scaled separation (primary form):
    L_aux = -λ * mean over cross-stream pairs (i,j): ||c_i - c_j|| * (1 - cos(g_i, g_j))
where:
    c_t = binary stream indicator latched from CTX tokens: [1,0] for stream-1
          positions, [0,1] for stream-2 positions (fixed, not learned)
    g_t = softmax gate at position t

Since ||c_i - c_j|| = 0 for same-stream pairs and sqrt(2) for cross-stream pairs,
this reduces to: reward gate dissimilarity only between cross-stream token pairs,
proportional to the (fixed) context difference. The scalar sqrt(2) is absorbed into λ.
No pair enumeration for same-stream pairs needed — they contribute zero by construction.

Vectorised implementation: O(T^2) per sequence per batch element, manageable for T=26.

λ sweep: [0, 0.01, 0.1, 1.0].  λ=0 reproduces the collapsed baseline.
Ceiling and floor numbers are computed once per seed (from perfect-gate and uniform-gate
models with no aux loss) and held fixed when computing the lift fraction.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from bdh_recurrent import bdh_layer_parallel, bdh_layer_mc_parallel, ln

# ── Task constants (identical to test_interference.py) ────────────────────────
P     = 4
k     = 2
D     = 32
N     = 64
BATCH = 32
ITERS = 1200
LR    = 4e-3
SEEDS = [0, 1, 2]

BLOCK = 6 * P + 3   # = 27
CTX1  = 0
CTX2  = 1
VAL_A = 2 + P
VAL_B = 2 + P + 1
V     = 2 + P + 2

def key_tok(i): return 2 + i


# ── Data (identical to test_interference.py) ──────────────────────────────────
def make_batch(B, rng):
    seqs = []; q_streams = []
    for _ in range(B):
        flip   = torch.randint(2, (1,), generator=rng).item()
        val_s1 = VAL_A if flip == 0 else VAL_B
        val_s2 = VAL_B if flip == 0 else VAL_A
        perm   = torch.randperm(P, generator=rng).tolist()
        seq    = []
        for i in perm:
            seq += [CTX1, key_tok(i), val_s1]
            seq += [CTX2, key_tok(i), val_s2]
        q_stream = torch.randint(2, (1,), generator=rng).item()
        q_key    = torch.randint(P, (1,), generator=rng).item()
        q_ctx    = CTX1 if q_stream == 0 else CTX2
        q_val    = val_s1 if q_stream == 0 else val_s2
        seq     += [q_ctx, key_tok(q_key), q_val]
        assert len(seq) == BLOCK
        seqs.append(seq); q_streams.append(q_stream)
    return (torch.tensor(seqs, dtype=torch.long),
            torch.tensor(q_streams, dtype=torch.long))


# ── Context signal: binary stream indicator, latched from CTX tokens ──────────
def stream_indicator(inp_seq):
    """
    inp_seq: (T,) input tokens.
    Returns (T, 2) one-hot: [1,0] for stream-1 positions, [0,1] for stream-2,
    [0.5,0.5] before first CTX token (neutral, contributes ~0 cross-pair weight).
    """
    T  = inp_seq.shape[0]
    c  = torch.full((T, 2), 0.5)
    cur = torch.tensor([0.5, 0.5])
    for t in range(T):
        tok = inp_seq[t].item()
        if tok == CTX1:
            cur = torch.tensor([1.0, 0.0])
        elif tok == CTX2:
            cur = torch.tensor([0.0, 1.0])
        c[t] = cur
    return c   # (T, 2)


# ── Auxiliary loss ─────────────────────────────────────────────────────────────
def aux_loss_seq(g, c):
    """
    g: (T, k) gate distributions (softmax outputs, differentiable)
    c: (T, 2) stream indicators (fixed; not in autograd graph)

    Pairwise context-scaled separation:
      L = -mean_{i<j} [ ||c_i - c_j|| * (1 - cos(g_i, g_j)) ]

    ||c_i - c_j|| = 0 for same-stream pairs, sqrt(2) for cross-stream pairs
    (since c is one-hot or neutral 0.5). Cross-stream pairs dominate;
    neutral pairs contribute ~sqrt(0) which is small.
    Vectorised with outer products — no explicit Python loop over pairs.
    """
    T = g.shape[0]
    # pairwise context distance matrix (T, T) — no gradient
    c_dist = torch.cdist(c, c)                         # (T, T), ||c_i - c_j||

    # pairwise gate cosine similarity (T, T) — gradient flows here
    g_n   = g / (g.norm(dim=-1, keepdim=True) + 1e-8)
    g_cos = g_n @ g_n.T                               # (T, T)

    # upper triangle only (each pair once)
    mask = torch.triu(torch.ones(T, T, device=g.device), diagonal=1)

    weighted = c_dist * (1.0 - g_cos) * mask          # (T, T)
    n_pairs  = mask.sum()
    if n_pairs < 1:
        return torch.tensor(0.0, requires_grad=False)
    return -weighted.sum() / n_pairs                   # negate → minimize = separate


# ── Perfect context-latching gate (for ceiling model) ────────────────────────
def _perfect_gate(inp_seq):
    T   = inp_seq.shape[0]
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


# ── Models ────────────────────────────────────────────────────────────────────
class BDHMCLearned(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.Dx    = nn.Parameter(torch.randn(D, N) * 0.1)
        self.Dy    = nn.Parameter(torch.randn(D, N) * 0.1)
        self.E     = nn.Parameter(torch.randn(N, D) * 0.1)
        self.head  = nn.Linear(D, V, bias=False)
        self.W_g   = nn.Parameter(torch.randn(D, k) * 0.1)

    def forward(self, tokens):
        B, T = tokens.shape
        v    = self.embed(tokens)
        out  = []
        for b in range(B):
            v_next, *_ = bdh_layer_mc_parallel(v[b], self.Dx, self.Dy, self.E, self.W_g)
            out.append(v_next)
        return self.head(torch.stack(out))

    def gates(self, tokens):
        """Returns (B, T, k) gate tensors (differentiable)."""
        v = self.embed(tokens)
        return F.softmax(v @ self.W_g, dim=-1)


class BDHMCUniform(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.Dx    = nn.Parameter(torch.randn(D, N) * 0.1)
        self.Dy    = nn.Parameter(torch.randn(D, N) * 0.1)
        self.E     = nn.Parameter(torch.randn(N, D) * 0.1)
        self.head  = nn.Linear(D, V, bias=False)
        self.register_buffer('W_g', torch.zeros(D, k))

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
        B, T = tokens.shape
        v    = self.embed(tokens)
        out  = []
        for b in range(B):
            g      = _perfect_gate(tokens[b])
            x      = F.relu(ln(v[b]) @ self.Dx)
            scores = ((x @ x.T) * (g @ g.T)).tril(diagonal=-1)
            a_ast  = scores @ v[b]
            y      = F.relu(ln(a_ast) @ self.Dy) * x
            v_next = v[b] + ln(y @ self.E)
            out.append(v_next)
        return self.head(torch.stack(out))


# ── Training + evaluation ─────────────────────────────────────────────────────
def run_model(make_model, seed, lam=0.0, has_gate=False):
    """Train and eval one model variant.  Returns (loss_s1, loss_s2, acc_s1, acc_s2, gate_info)."""
    torch.manual_seed(seed)
    model = make_model()
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    rng   = torch.Generator(); rng.manual_seed(seed + 10_000)

    for _ in range(ITERS):
        tokens, _ = make_batch(BATCH, rng)
        inp        = tokens[:, :-1]
        tgt        = tokens[:, 1:]
        logits     = model(inp)
        task_loss  = F.cross_entropy(logits[:, -1, :], tgt[:, -1])

        if lam > 0 and has_gate:
            g_batch = model.gates(inp)        # (B, T, k)
            aloss   = torch.tensor(0.0)
            for b in range(BATCH):
                c_b   = stream_indicator(inp[b]).detach()   # (T, 2) fixed
                aloss = aloss + aux_loss_seq(g_batch[b], c_b)
            aloss = aloss / BATCH
            loss  = task_loss + lam * aloss
        else:
            loss = task_loss

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
        m1       = (q_streams == 0)
        m2       = (q_streams == 1)

        loss_s1 = F.cross_entropy(q_logits[m1], q_tgt[m1]).item()
        loss_s2 = F.cross_entropy(q_logits[m2], q_tgt[m2]).item()
        acc_s1  = correct[m1].float().mean().item()
        acc_s2  = correct[m2].float().mean().item()

        gate_info = None
        if has_gate:
            g_last = F.softmax(model.embed(inp[:, -1:]) @ model.W_g, dim=-1).squeeze(1)
            g_s1   = g_last[m1].mean(0)
            g_s2   = g_last[m2].mean(0)
            cos    = F.cosine_similarity(g_s1.unsqueeze(0), g_s2.unsqueeze(0)).item()
            gate_info = (g_s1.tolist(), g_s2.tolist(), cos)

    return loss_s1, loss_s2, acc_s1, acc_s2, gate_info


def run_seeds(make_model, seeds, lam=0.0, has_gate=False):
    all_l1, all_l2, all_a1, all_a2, gate_infos = [], [], [], [], []
    for seed in seeds:
        l1, l2, a1, a2, gi = run_model(make_model, seed, lam=lam, has_gate=has_gate)
        all_l1.append(l1); all_l2.append(l2)
        all_a1.append(a1); all_a2.append(a2)
        if gi: gate_infos.append(gi)

    def ms(xs):
        m = sum(xs)/len(xs)
        s = (sum((x-m)**2 for x in xs)/len(xs))**0.5
        return m, s

    ml1,sl1 = ms(all_l1); ml2,sl2 = ms(all_l2)
    ma1,sa1 = ms(all_a1); ma2,sa2 = ms(all_a2)
    avg_acc = (ma1 + ma2) / 2

    avg_gate = None
    if gate_infos:
        gs1 = [sum(gi[0][c] for gi in gate_infos)/len(gate_infos) for c in range(k)]
        gs2 = [sum(gi[1][c] for gi in gate_infos)/len(gate_infos) for c in range(k)]
        avg_cos = sum(gi[2] for gi in gate_infos)/len(gate_infos)
        avg_gate = (gs1, gs2, avg_cos)

    return dict(ml1=ml1,sl1=sl1, ml2=ml2,sl2=sl2,
                ma1=ma1,sa1=sa1, ma2=ma2,sa2=sa2,
                avg_acc=avg_acc, gate=avg_gate)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    LAMBDAS = [0.0, 0.01, 0.1, 1.0]

    print("=" * 66)
    print("Interference + aux gate-separation loss sweep")
    print(f"  P={P}  k={k}  D={D}  N={N}  BLOCK={BLOCK}  ITERS={ITERS}  SEEDS={SEEDS}")
    print(f"  Aux loss: pairwise context-scaled separation (cross-stream pairs only)")
    print(f"  λ sweep: {LAMBDAS}")
    print("=" * 66)
    print()

    # Compute ceiling and floor once per seed (no aux loss)
    print("Computing ceiling (perfect gate) and floor (uniform gate)...")
    ceil_r  = run_seeds(BDHMCPerfect, SEEDS, lam=0.0, has_gate=False)
    floor_r = run_seeds(BDHMCUniform, SEEDS, lam=0.0, has_gate=False)
    ceil_acc  = ceil_r["avg_acc"]
    floor_acc = floor_r["avg_acc"]
    print(f"  Ceiling (perfect gate): acc={ceil_acc:.3f}")
    print(f"  Floor   (uniform gate): acc={floor_acc:.3f}")
    print()

    sweep_results = {}
    for lam in LAMBDAS:
        print(f"-- λ={lam} --")
        r = run_seeds(BDHMCLearned, SEEDS, lam=lam, has_gate=True)
        sweep_results[lam] = r

        print(f"  stream-1: loss={r['ml1']:.4f}+-{r['sl1']:.4f}  "
              f"acc={r['ma1']:.3f}+-{r['sa1']:.3f}")
        print(f"  stream-2: loss={r['ml2']:.4f}+-{r['sl2']:.4f}  "
              f"acc={r['ma2']:.3f}+-{r['sa2']:.3f}")

        if r["gate"]:
            gs1, gs2, cos = r["gate"]
            gs1_s = "[" + ", ".join(f"{v:.3f}" for v in gs1) + "]"
            gs2_s = "[" + ", ".join(f"{v:.3f}" for v in gs2) + "]"
            sep = "SEPARATED" if cos < 0.5 else "not separated"
            print(f"  gate @query  stream-1: {gs1_s}")
            print(f"  gate @query  stream-2: {gs2_s}")
            print(f"  cosine(g_s1, g_s2) = {cos:.4f}  ({sep})")

        span = ceil_acc - floor_acc
        lift = (r["avg_acc"] - floor_acc) / span if span > 1e-3 else 0.0
        print(f"  lift fraction: {lift:.2f}  "
              f"(avg_acc={r['avg_acc']:.3f}, ceil={ceil_acc:.3f}, floor={floor_acc:.3f})")
        print()

    # ── Verdict ───────────────────────────────────────────────────────────────
    span       = ceil_acc - floor_acc
    best_lam   = None
    best_lift  = -999
    best_cos   = 1.0
    any_sep    = False
    any_lift   = False
    over_force = None

    base_loss  = (sweep_results[0.0]["ml1"] + sweep_results[0.0]["ml2"]) / 2

    for lam in LAMBDAS:
        r     = sweep_results[lam]
        lift  = (r["avg_acc"] - floor_acc) / span if span > 1e-3 else 0.0
        cos   = r["gate"][2] if r["gate"] else 1.0
        t_loss = (r["ml1"] + r["ml2"]) / 2

        if cos < 0.5:
            any_sep = True
        if lift > 0.5:
            any_lift = True
        if lift > best_lift:
            best_lift = lift; best_lam = lam; best_cos = cos

        if lam > 0 and t_loss > base_loss + 0.05 and over_force is None:
            over_force = lam

    print("=" * 66)
    print("VERDICT")
    print("=" * 66)
    print(f"  Best λ: {best_lam}  →  lift={best_lift:.2f}, gate cosine={best_cos:.3f}")
    if over_force is not None:
        print(f"  Over-forcing starts at λ={over_force} (task loss rises >0.05 above λ=0 baseline)")
    else:
        print("  No over-forcing observed across the sweep.")
    print()

    if any_sep and any_lift:
        print("(a) The aux loss FIXES the collapse. At least one λ produces both "
              "gate separation (cosine < 0.5) AND a lift fraction > 0.50 toward the "
              f"ceiling. Best: λ={best_lam}, lift={best_lift:.2f}, cos={best_cos:.3f}. "
              "The collapse in test_interference.py is an optimization artifact "
              "fixable by encouraging the gate to separate streams.")
    elif any_sep and not any_lift:
        print("(b) The aux loss FORCES gate separation but accuracy does NOT follow. "
              "The gate cosine drops below 0.5 at some λ, but the lift fraction "
              f"never exceeds 0.50 (best lift={best_lift:.2f}). "
              "Structural gate separation does not translate to memory disambiguation: "
              "the Hebbian retrieval mechanism cannot exploit the routing signal even "
              "when it is explicitly imposed.")
    else:
        print("(b) The aux loss FAILS to fix the collapse. No λ produces gate "
              f"separation (min cosine={best_cos:.3f} > 0.5) or lift (best={best_lift:.2f}). "
              "The mechanism does not engage under this auxiliary pressure.")
