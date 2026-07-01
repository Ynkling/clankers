"""
test_interference_discrete.py — Interference test with hard discrete gate routing (STE).

Identical task, ceiling, floor, and metrics to test_interference.py (two-stream shared-key
collision, per-stream query loss/accuracy, >=3 seeds, mean+-std, lift fraction).

Replaces the soft gate blend with hard one-hot channel routing via a straight-through
estimator (STE): forward pass uses argmax(softmax(v_ast @ W_g)) as a one-hot vector;
backward pass gradients flow through the soft softmax (standard STE trick:
g_st = hard.detach() + soft - soft.detach()). Both the read and the Hebbian write use
this same hard-routed g_st inside the same g@g.T score term used by bdh_layer_mc_parallel,
so both operations inherit the hard choice consistently.

Conditions (>=3 seeds each):
  1. discrete gate, standard (random) init      — does hard routing spontaneously separate?
  2. discrete gate, init already-separated      — if started separated, does it STAY separated
     (W_g pre-set along the CTX1-CTX2 embedding difference direction)   or drift back to collapse?
  3. (reference, recomputed here) soft-gate learned result, perfect-gate ceiling, uniform-gate floor

Routing diagnostics reported per condition:
  - fraction of stream-1 tokens routed to channel 0 vs channel 1
  - fraction of stream-2 tokens routed to channel 0 vs channel 1
  - dead-channel collapse flag: >90% of ALL (stream-labeled) tokens routed to one channel,
    regardless of stream
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


# ── Stream label per position: latched from most recent CTX token, -1 before any ──
def stream_label(inp_seq):
    T = inp_seq.shape[0]
    labels = torch.full((T,), -1, dtype=torch.long)
    cur = -1
    for t in range(T):
        tok = inp_seq[t].item()
        if tok == CTX1:
            cur = 0
        elif tok == CTX2:
            cur = 1
        labels[t] = cur
    return labels


# ── Perfect context-latching gate (ceiling) ───────────────────────────────────
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


# ── Hard-routed MC layer (straight-through estimator) ─────────────────────────
def bdh_layer_mc_discrete(v_ast, Dx, Dy, E, W_g):
    """
    Forward: hard one-hot channel choice (argmax of softmax gate).
    Backward: gradients flow through the soft softmax gate (STE).
    Both read and write inherit the hard choice via the same g_st @ g_st.T score term.
    Returns v_next, hard_idx (T,), soft (T,k) for diagnostics.
    """
    x       = F.relu(ln(v_ast) @ Dx)                     # (T, N)
    logits  = v_ast @ W_g                                # (T, k)
    soft    = F.softmax(logits, dim=-1)                  # (T, k)
    hard_idx = soft.argmax(dim=-1)                        # (T,)
    hard    = F.one_hot(hard_idx, num_classes=W_g.shape[1]).float()   # (T, k)
    g_st    = hard + soft - soft.detach()                 # straight-through estimator
    scores  = ((x @ x.T) * (g_st @ g_st.T)).tril(diagonal=-1)   # (T, T)
    a_ast   = scores @ v_ast
    y       = F.relu(ln(a_ast) @ Dy) * x
    v_next  = v_ast + ln(y @ E)
    return v_next, hard_idx, soft


# ── Models ────────────────────────────────────────────────────────────────────
class BDHMCDiscrete(nn.Module):
    def __init__(self, separated_init=False):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.Dx    = nn.Parameter(torch.randn(D, N) * 0.1)
        self.Dy    = nn.Parameter(torch.randn(D, N) * 0.1)
        self.E     = nn.Parameter(torch.randn(N, D) * 0.1)
        self.head  = nn.Linear(D, V, bias=False)
        self.W_g   = nn.Parameter(torch.randn(D, k) * 0.1)

        if separated_init:
            with torch.no_grad():
                diff  = self.embed.weight[CTX1] - self.embed.weight[CTX2]   # (D,)
                scale = 10.0
                w0 = torch.zeros(D, k)
                w0[:, 0] = diff * scale
                w0[:, 1] = -diff * scale
                self.W_g.copy_(w0)

    def forward(self, tokens):
        B, T = tokens.shape
        v    = self.embed(tokens)
        out  = []
        for b in range(B):
            v_next, *_ = bdh_layer_mc_discrete(v[b], self.Dx, self.Dy, self.E, self.W_g)
            out.append(v_next)
        return self.head(torch.stack(out))

    def route(self, tokens):
        """Returns (B, T) hard channel indices for diagnostics (no grad needed)."""
        B, T = tokens.shape
        v    = self.embed(tokens)
        idxs = []
        for b in range(B):
            _, hard_idx, _ = bdh_layer_mc_discrete(v[b], self.Dx, self.Dy, self.E, self.W_g)
            idxs.append(hard_idx)
        return torch.stack(idxs)


class BDHMCLearned(nn.Module):
    """Soft-gate learned model, reference from test_interference.py condition 2."""
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


class BDHMCUniform(nn.Module):
    """Uniform-gate floor, reference from test_interference.py condition 4."""
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
    """Perfect context-latching gate, reference from test_interference.py condition 3 (ceiling)."""
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
def run_model(make_model, seed, is_discrete=False):
    torch.manual_seed(seed)
    model = make_model()
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    rng   = torch.Generator(); rng.manual_seed(seed + 10_000)

    for _ in range(ITERS):
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
        m1       = (q_streams == 0)
        m2       = (q_streams == 1)

        loss_s1 = F.cross_entropy(q_logits[m1], q_tgt[m1]).item()
        loss_s2 = F.cross_entropy(q_logits[m2], q_tgt[m2]).item()
        acc_s1  = correct[m1].float().mean().item()
        acc_s2  = correct[m2].float().mean().item()

        routing = None
        if is_discrete:
            hard_idx = model.route(inp)             # (B, T)
            labels   = torch.stack([stream_label(inp[b]) for b in range(inp.shape[0])])  # (B, T)
            valid    = labels >= 0
            s0_mask  = valid & (labels == 0)
            s1_mask  = valid & (labels == 1)

            s0_ch0 = (hard_idx[s0_mask] == 0).float().mean().item() if s0_mask.any() else float('nan')
            s0_ch1 = (hard_idx[s0_mask] == 1).float().mean().item() if s0_mask.any() else float('nan')
            s1_ch0 = (hard_idx[s1_mask] == 0).float().mean().item() if s1_mask.any() else float('nan')
            s1_ch1 = (hard_idx[s1_mask] == 1).float().mean().item() if s1_mask.any() else float('nan')

            all_mask = valid
            all_ch0  = (hard_idx[all_mask] == 0).float().mean().item()
            all_ch1  = (hard_idx[all_mask] == 1).float().mean().item()
            dead     = max(all_ch0, all_ch1) > 0.90

            routing = dict(s0_ch0=s0_ch0, s0_ch1=s0_ch1, s1_ch0=s1_ch0, s1_ch1=s1_ch1,
                           all_ch0=all_ch0, all_ch1=all_ch1, dead_channel=dead)

    return loss_s1, loss_s2, acc_s1, acc_s2, routing


def run_seeds(make_model, seeds, is_discrete=False):
    all_l1, all_l2, all_a1, all_a2 = [], [], [], []
    routings = []
    for seed in seeds:
        l1, l2, a1, a2, rt = run_model(make_model, seed, is_discrete=is_discrete)
        all_l1.append(l1); all_l2.append(l2)
        all_a1.append(a1); all_a2.append(a2)
        if rt: routings.append(rt)

    def ms(xs):
        m = sum(xs)/len(xs)
        s = (sum((x-m)**2 for x in xs)/len(xs))**0.5
        return m, s

    ml1,sl1 = ms(all_l1); ml2,sl2 = ms(all_l2)
    ma1,sa1 = ms(all_a1); ma2,sa2 = ms(all_a2)
    avg_acc = (ma1 + ma2) / 2

    avg_routing = None
    if routings:
        def avg(key):
            vals = [r[key] for r in routings if r[key] == r[key]]  # filter NaN
            return sum(vals)/len(vals) if vals else float('nan')
        avg_routing = dict(
            s0_ch0=avg('s0_ch0'), s0_ch1=avg('s0_ch1'),
            s1_ch0=avg('s1_ch0'), s1_ch1=avg('s1_ch1'),
            all_ch0=avg('all_ch0'), all_ch1=avg('all_ch1'),
            dead_channel=any(r['dead_channel'] for r in routings),
        )

    return dict(ml1=ml1,sl1=sl1, ml2=ml2,sl2=sl2,
                ma1=ma1,sa1=sa1, ma2=ma2,sa2=sa2,
                avg_acc=avg_acc, routing=avg_routing)


def print_result(name, r, ceil_acc=None, floor_acc=None):
    print(f"-- {name} --")
    print(f"  stream-1: loss={r['ml1']:.4f}+-{r['sl1']:.4f}  acc={r['ma1']:.3f}+-{r['sa1']:.3f}")
    print(f"  stream-2: loss={r['ml2']:.4f}+-{r['sl2']:.4f}  acc={r['ma2']:.3f}+-{r['sa2']:.3f}")
    if r['routing']:
        rt = r['routing']
        print(f"  routing  stream-1 tokens -> ch0={rt['s0_ch0']:.3f}  ch1={rt['s0_ch1']:.3f}")
        print(f"  routing  stream-2 tokens -> ch0={rt['s1_ch0']:.3f}  ch1={rt['s1_ch1']:.3f}")
        print(f"  routing  ALL tokens      -> ch0={rt['all_ch0']:.3f}  ch1={rt['all_ch1']:.3f}"
              f"  {'[DEAD-CHANNEL COLLAPSE]' if rt['dead_channel'] else ''}")
    if ceil_acc is not None and floor_acc is not None:
        span = ceil_acc - floor_acc
        lift = (r['avg_acc'] - floor_acc) / span if span > 1e-3 else 0.0
        print(f"  lift fraction: {lift:.2f}  (avg_acc={r['avg_acc']:.3f}, "
              f"ceil={ceil_acc:.3f}, floor={floor_acc:.3f})")
        r['lift'] = lift
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 66)
    print("Interference test: hard discrete gate routing (straight-through estimator)")
    print(f"  P={P}  k={k}  D={D}  N={N}  BLOCK={BLOCK}  ITERS={ITERS}  SEEDS={SEEDS}")
    print("=" * 66)
    print()

    print("Computing reference: ceiling (perfect gate) and floor (uniform gate)...")
    ceil_r  = run_seeds(BDHMCPerfect, SEEDS, is_discrete=False)
    floor_r = run_seeds(BDHMCUniform, SEEDS, is_discrete=False)
    ceil_acc, floor_acc = ceil_r["avg_acc"], floor_r["avg_acc"]
    print(f"  Ceiling (perfect gate): acc={ceil_acc:.3f}")
    print(f"  Floor   (uniform gate): acc={floor_acc:.3f}")
    print()

    print("Computing reference: soft-gate learned result...")
    soft_r = run_seeds(BDHMCLearned, SEEDS, is_discrete=False)
    print_result("Reference: soft-gate learned (from test_interference.py cond 2)",
                 soft_r, ceil_acc, floor_acc)

    print("Cond 1: discrete gate, standard (random) init")
    cond1_r = run_seeds(lambda: BDHMCDiscrete(separated_init=False), SEEDS, is_discrete=True)
    print_result("Cond 1: discrete gate, standard init", cond1_r, ceil_acc, floor_acc)

    print("Cond 2: discrete gate, init already-separated")
    cond2_r = run_seeds(lambda: BDHMCDiscrete(separated_init=True), SEEDS, is_discrete=True)
    print_result("Cond 2: discrete gate, separated init", cond2_r, ceil_acc, floor_acc)

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("=" * 66)
    print("VERDICT")
    print("=" * 66)

    rt1 = cond1_r['routing']
    lift1 = cond1_r['lift']
    separated1 = (max(rt1['s0_ch0'], rt1['s0_ch1']) > 0.7 and
                  max(rt1['s1_ch0'], rt1['s1_ch1']) > 0.7 and
                  ((rt1['s0_ch0'] > rt1['s0_ch1']) != (rt1['s1_ch0'] > rt1['s1_ch1'])))

    print(f"Cond 1 routing: stream-1 -> [ch0={rt1['s0_ch0']:.2f}, ch1={rt1['s0_ch1']:.2f}]  "
          f"stream-2 -> [ch0={rt1['s1_ch0']:.2f}, ch1={rt1['s1_ch1']:.2f}]")
    print(f"Cond 1 dead-channel collapse: {rt1['dead_channel']}")
    print(f"Cond 1 lift fraction: {lift1:.2f}")
    print()

    if separated1 and lift1 > 0.5:
        print("(a) Condition 1 SPONTANEOUSLY SEPARATES: each stream routes predominantly "
              "to its own channel, and lift climbs toward the ceiling "
              f"(lift={lift1:.2f}). Discrete routing RESCUES the mechanism.")
    else:
        print("(b) Condition 1 COLLAPSES: "
              + ("dead-channel collapse (>90% of all tokens on one channel)."
                 if rt1['dead_channel'] else
                 "no reliable stream/channel correspondence.")
              + f" Lift stays near the floor (lift={lift1:.2f}). "
              "Discrete routing does NOT rescue the mechanism.")
        print()
        rt2 = cond2_r['routing']
        lift2 = cond2_r['lift']
        separated2 = (max(rt2['s0_ch0'], rt2['s0_ch1']) > 0.7 and
                      max(rt2['s1_ch0'], rt2['s1_ch1']) > 0.7 and
                      ((rt2['s0_ch0'] > rt2['s0_ch1']) != (rt2['s1_ch0'] > rt2['s1_ch1'])))
        print(f"Cond 2 routing: stream-1 -> [ch0={rt2['s0_ch0']:.2f}, ch1={rt2['s0_ch1']:.2f}]  "
              f"stream-2 -> [ch0={rt2['s1_ch0']:.2f}, ch1={rt2['s1_ch1']:.2f}]")
        print(f"Cond 2 dead-channel collapse: {rt2['dead_channel']}")
        print(f"Cond 2 lift fraction: {lift2:.2f}")
        print()
        if separated2 and lift2 > 0.5:
            print("Condition 2 STAYS SEPARATED with high lift: the good (separated) "
                  "solution is a STABLE fixed point of training, but it is UNREACHABLE "
                  "from the standard random init. The optimizer does not discover it "
                  "on its own; it must be placed there.")
        else:
            print("Condition 2 DRIFTS BACK toward collapse even from a separated init: "
                  "the separated solution is not even a stable fixed point under this "
                  "training objective. Gradient descent actively erodes the useful "
                  "routing structure once training starts.")
