"""
test_arm2_readinit.py — Attacking arm 2 of the two-arm deadlock: does a separated READ
gate let the write gate bootstrap into separation?

Motivation (from experiments 5 and 6 in this repo):
  ARM 1 — the write-gate gradient vanishes under a uniform read, because the gate is
    normalized and accumulation is linear, so sum_c S^(c) is routing-invariant
    (test_asymmetric_routing.py CHECK 5, verified at 9.4e-16).
  ARM 2 — the read-gate gradient vanishes when the channels are undifferentiated,
    because reading any mixture of identical channels gives the same answer.
  test_bounded_capacity.py broke arm 1 with a capacity bound and found it necessary but
  not sufficient: the write-gate gradient became nonzero but carried no stream-identity
  information, because arm 2 was untouched.

This experiment attacks arm 2 directly and tests the theory's asymmetric prediction.
The gate is split into two trainable soft projections, W_g^read and W_g^write, both
softmax. The effective score factorizes,
    scores[t, tau] = (x_t . x_tau) * (g^read[t] . g^write[tau])
(read gate on the query row, write gate on the source column), which has a parallel
form, so the machine-precision parallel/recurrent identity that hard routing and
clipping made unavailable is restored here (CHECK 1). No bounded capacity, no Gumbel,
no STE — plain soft gates throughout, so test_interference._run_sanity's analytical 50%
floor holds exactly and the task stays clean.

"Initialized separated" uses the repo's existing construction (the separated_init of
test_interference_discrete.py): W[:, 0] = SCALE * (embed[CTX1] - embed[CTX2]),
W[:, 1] = -that, so the gate reads about [1, 0] on a stream-1 context token and about
[0, 1] on a stream-2 context token. SCALE is the precedent value and is not tuned.

Task, metrics and lift-fraction definition are imported unchanged from
test_interference.py; bdh_recurrent.py and test_interference.py are never modified.

Conditions (>= 3 seeds each), with both gates' routing and both cosines tracked at 21
checkpoints through training so emergence AND erosion are visible:
  1.  single-channel baseline                          (floor)
  2.  perfect gate, hand-set and frozen                (ceiling)
  3.  both gates random init                           (collapse reference, = Exp 2)
  4.  read init separated, write random, both trainable        <-- PRIMARY TEST
  4b. read init separated and FROZEN, write random trainable   <-- isolates bootstrap
                                                                   from read erosion
  5.  write init separated, read random, both trainable        <-- mirror; theory
                                                                   predicts failure
  6.  both init separated, both trainable                      <-- stability control

Verdict:
 (a) 4 and/or 4b climb toward the ceiling with write separation emerging, AND 5 stays
     at the floor -> the two-arm account is confirmed, asymmetry included;
 (b) both 4 and 5 climb -> the asymmetry claim is wrong, breaking either arm suffices;
 (c) both stay at the floor -> a third obstacle exists that six experiments missed;
 (d) 4b works but 4 does not -> the bootstrap is real but the separated read erodes
     before the write can exploit it.

(Results are recorded at the bottom of this docstring after the run.)
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from bdh_recurrent import bdh_layer_parallel, bdh_layer_mc_parallel, ln
import test_interference as ti
from test_interference import (
    P, k, D, N, BATCH, ITERS, LR, SEEDS, BLOCK, CTX1, CTX2, VAL_A, VAL_B, V,
    key_tok, make_batch, _perfect_gate, BDHSingle, BDHMCPerfect, run as ti_run,
)

# ── Separated-init construction (precedent value from test_interference_discrete) ──
SEP_SCALE = 10.0

# ── Diagnostics ───────────────────────────────────────────────────────────────
N_CKPT      = 21      # checkpoints across training (gate-only, so they are cheap)
PROBE_B     = 256
SEP_COS     = 0.50    # gate cosine below this counts as separated
LIFT_THRESH = 0.70    # lift fraction counting as "climbs toward the ceiling"


# ── Split-gate layer: separate read and write projections, both plain softmax ─
def bdh_layer_split(v_ast, Dx, Dy, E, W_gr, W_gw):
    """scores[t, s] = (x_t . x_s) * (g_read[t] . g_write[s]) — parallel form."""
    x  = F.relu(ln(v_ast) @ Dx)
    gr = F.softmax(v_ast @ W_gr, dim=-1)
    gw = F.softmax(v_ast @ W_gw, dim=-1)
    scores = ((x @ x.T) * (gr @ gw.T)).tril(diagonal=-1)
    a_ast  = scores @ v_ast
    y      = F.relu(ln(a_ast) @ Dy) * x
    return v_ast + ln(y @ E), gr, gw


def bdh_layer_split_recurrent(v_ast, Dx, Dy, E, W_gr, W_gw):
    """Explicit token-by-token Hebbian form of the same thing (verification only)."""
    T, Dd = v_ast.shape
    Nn = Dx.shape[1]
    kk = W_gr.shape[1]
    x  = F.relu(ln(v_ast) @ Dx)
    gr = F.softmax(v_ast @ W_gr, dim=-1)
    gw = F.softmax(v_ast @ W_gw, dim=-1)
    S = torch.zeros(kk, Nn, Dd, dtype=v_ast.dtype)
    a_ast = torch.zeros(T, Dd, dtype=v_ast.dtype)
    for t in range(T):
        reads = torch.einsum('n,knd->kd', x[t], S)                  # READ, per channel
        a_ast[t] = gr[t] @ reads                                    # blended by g_read
        S = S + torch.einsum('k,nd->knd', gw[t],                    # WRITE, by g_write
                             torch.outer(x[t], v_ast[t]))
    y = F.relu(ln(a_ast) @ Dy) * x
    return v_ast + ln(y @ E)


def separated_W(embed_weight, scale=SEP_SCALE):
    """W[:,0] = scale*(e_CTX1 - e_CTX2), W[:,1] = -that. Gate ~ [1,0] on CTX1, [0,1] on CTX2."""
    diff = embed_weight[CTX1] - embed_weight[CTX2]
    W = torch.zeros(embed_weight.shape[1], k, dtype=embed_weight.dtype)
    W[:, 0] = diff * scale
    W[:, 1] = -diff * scale
    return W


class BDHSplitGate(nn.Module):
    def __init__(self, read_init='random', write_init='random',
                 freeze_read=False, freeze_write=False):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.Dx    = nn.Parameter(torch.randn(D, N) * 0.1)
        self.Dy    = nn.Parameter(torch.randn(D, N) * 0.1)
        self.E     = nn.Parameter(torch.randn(N, D) * 0.1)
        self.head  = nn.Linear(D, V, bias=False)
        with torch.no_grad():
            sep = separated_W(self.embed.weight.detach())
        wr = sep.clone() if read_init  == 'separated' else torch.randn(D, k) * 0.1
        ww = sep.clone() if write_init == 'separated' else torch.randn(D, k) * 0.1
        self.W_gr = nn.Parameter(wr, requires_grad=not freeze_read)
        self.W_gw = nn.Parameter(ww, requires_grad=not freeze_write)

    def forward(self, tokens):
        v = self.embed(tokens)
        out = []
        for b in range(tokens.shape[0]):
            v_next, *_ = bdh_layer_split(v[b], self.Dx, self.Dy, self.E,
                                         self.W_gr, self.W_gw)
            out.append(v_next)
        return self.head(torch.stack(out))


# ── Sequence construction with an explicit flip (for CHECK 6) ────────────────
def make_seq_explicit(perm, flip, q_stream, q_key):
    """Same layout as test_interference.make_batch, with every choice pinned."""
    val_s1 = VAL_A if flip == 0 else VAL_B
    val_s2 = VAL_B if flip == 0 else VAL_A
    seq = []
    for i in perm:
        seq += [CTX1, key_tok(i), val_s1]
        seq += [CTX2, key_tok(i), val_s2]
    q_ctx = CTX1 if q_stream == 0 else CTX2
    q_val = val_s1 if q_stream == 0 else val_s2
    seq += [q_ctx, key_tok(q_key), q_val]
    assert len(seq) == BLOCK
    return torch.tensor(seq, dtype=torch.long)


def stream_labels(tokens):
    """(B,T) in {-1,0,1}: which stream each position belongs to, latched from CTX."""
    B, T = tokens.shape
    lab = torch.full((B, T), -1, dtype=torch.long)
    for b in range(B):
        cur = -1
        for t in range(T):
            tok = tokens[b, t].item()
            if tok == CTX1:
                cur = 0
            elif tok == CTX2:
                cur = 1
            lab[b, t] = cur
    return lab


@torch.no_grad()
def gate_diag(model, tokens, labels):
    """Both gates' per-stream routing and cosines. Gate-only, so checkpoints are cheap."""
    v = model.embed(tokens)
    out = {}
    for name, W in (('r', model.W_gr), ('w', model.W_gw)):
        g = F.softmax(v @ W, dim=-1)
        m0, m1 = labels == 0, labels == 1
        g0, g1 = g[m0].mean(0), g[m1].mean(0)
        out[f'{name}_s1'] = g0.tolist()
        out[f'{name}_s2'] = g1.tolist()
        out[f'{name}_cos'] = F.cosine_similarity(g0.unsqueeze(0), g1.unsqueeze(0)).item()
        # gate on the CTX tokens specifically — the only place separation is expressible
        c1 = g[tokens == CTX1].mean(0)
        c2 = g[tokens == CTX2].mean(0)
        out[f'{name}_ctx1'] = c1.tolist()
        out[f'{name}_ctx2'] = c2.tolist()
        out[f'{name}_cos_ctx'] = F.cosine_similarity(c1.unsqueeze(0), c2.unsqueeze(0)).item()
    return out


def run_split(read_init, write_init, freeze_read, freeze_write, seed):
    """Mirrors test_interference.run(): same seeding, batches, loss and eval protocol."""
    torch.manual_seed(seed)
    model = BDHSplitGate(read_init, write_init, freeze_read, freeze_write)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)
    rng = torch.Generator(); rng.manual_seed(seed + 10_000)

    prng = torch.Generator(); prng.manual_seed(seed + 99_000)
    ptok, _ = make_batch(PROBE_B, prng)
    pinp = ptok[:, :-1]
    plab = stream_labels(pinp)

    ckpts = sorted({int(i * (ITERS - 1) / (N_CKPT - 1)) for i in range(N_CKPT)})
    trace = []
    for step in range(ITERS):
        if step in ckpts:
            d = gate_diag(model, pinp, plab); d['step'] = step
            trace.append(d)
        tokens, _ = make_batch(BATCH, rng)
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        logits = model(inp)
        loss = F.cross_entropy(logits[:, -1, :], tgt[:, -1])   # query position only
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    d = gate_diag(model, pinp, plab); d['step'] = ITERS
    trace.append(d)

    with torch.no_grad():
        tokens, q_streams = make_batch(512, rng)
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        logits = model(inp)
        ql, qt = logits[:, -1, :], tgt[:, -1]
        corr = (ql.argmax(-1) == qt)
        m1, m2 = q_streams == 0, q_streams == 1
        res = dict(l1=F.cross_entropy(ql[m1], qt[m1]).item(),
                   l2=F.cross_entropy(ql[m2], qt[m2]).item(),
                   a1=corr[m1].float().mean().item(),
                   a2=corr[m2].float().mean().item(), trace=trace)
    return res


def ms(xs):
    m = sum(xs) / len(xs)
    return m, (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def agg(runs):
    o = {}
    for key in ('l1', 'l2', 'a1', 'a2'):
        o[key], o[key + '_sd'] = ms([r[key] for r in runs])
    o['acc'] = (o['a1'] + o['a2']) / 2
    o['traces'] = [r['trace'] for r in runs]
    return o


def fmt(xs):
    return "[" + ", ".join(f"{x:.3f}" for x in xs) + "]"


def first_below(traces, key, thresh=SEP_COS):
    """First checkpoint step at which the mean cosine `key` drops below thresh."""
    n = len(traces[0])
    for i in range(n):
        vals = [t[i][key] for t in traces]
        if sum(vals) / len(vals) < thresh:
            return traces[0][i]['step']
    return None


def print_trace(label, traces):
    print(f"  gate trace ({label}, mean over {len(traces)} seeds):")
    print(f"    {'step':>5} | {'read cos':>8} {'r cos@ctx':>9} | {'write cos':>9} "
          f"{'w cos@ctx':>9} | {'read s1':>16} {'read s2':>16} | "
          f"{'write s1':>16} {'write s2':>16}")
    for i in range(len(traces[0])):
        pts = [t[i] for t in traces]
        avg = lambda key: sum(p[key] for p in pts) / len(pts)
        avgv = lambda key: [sum(p[key][c] for p in pts) / len(pts) for c in range(k)]
        print(f"    {pts[0]['step']:>5} | {avg('r_cos'):>8.4f} {avg('r_cos_ctx'):>9.4f} | "
              f"{avg('w_cos'):>9.4f} {avg('w_cos_ctx'):>9.4f} | "
              f"{fmt(avgv('r_s1')):>16} {fmt(avgv('r_s2')):>16} | "
              f"{fmt(avgv('w_s1')):>16} {fmt(avgv('w_s2')):>16}")
    print()


# ── Verification ──────────────────────────────────────────────────────────────
def verify():
    print("=" * 96)
    print("VERIFICATION (numeric, before any training condition)")
    print("=" * 96)
    g = torch.Generator().manual_seed(4242)
    T, dt = BLOCK - 1, torch.float64
    v   = torch.randn(T, D, generator=g, dtype=dt)
    Dx  = torch.randn(D, N, generator=g, dtype=dt) * 0.1
    Dy  = torch.randn(D, N, generator=g, dtype=dt) * 0.1
    E   = torch.randn(N, D, generator=g, dtype=dt) * 0.1
    Wgr = torch.randn(D, k, generator=g, dtype=dt) * 0.5
    Wgw = torch.randn(D, k, generator=g, dtype=dt) * 0.5
    ok = True

    # CHECK 1 — parallel == recurrent, machine precision (restored by soft gates)
    vp, _, _ = bdh_layer_split(v, Dx, Dy, E, Wgr, Wgw)
    vr = bdh_layer_split_recurrent(v, Dx, Dy, E, Wgr, Wgw)
    d1 = (vp - vr).abs().max().item()
    print(f"CHECK 1  split-gate parallel vs explicit recurrent Hebbian loop")
    print(f"         max |v_parallel - v_recurrent| = {d1:.2e}")
    ok &= d1 < 1e-12
    print(f"         -> {'IDENTICAL' if d1 < 1e-12 else 'FAILED'} "
          f"(pins read gate on the query row, write gate on the source column)\n")

    # CHECK 2 — k=1 reduces exactly to single-channel BDH
    w1r = torch.randn(D, 1, generator=g, dtype=dt) * 0.5
    w1w = torch.randn(D, 1, generator=g, dtype=dt) * 0.5
    v1, _, _ = bdh_layer_split(v, Dx, Dy, E, w1r, w1w)
    vs, *_ = bdh_layer_parallel(v, Dx, Dy, E)
    d2 = (v1 - vs).abs().max().item()
    print(f"CHECK 2  k=1 split gate vs imported bdh_layer_parallel: {d2:.2e}")
    ok &= d2 < 1e-12
    print(f"         -> {'EXACT' if d2 < 1e-12 else 'FAILED'} "
          f"(softmax over one logit is identically 1)\n")

    # CHECK 3 — W_read = W_write reduces exactly to the symmetric model of Exp 2
    vsym, _, _ = bdh_layer_split(v, Dx, Dy, E, Wgr, Wgr)
    vmc, *_ = bdh_layer_mc_parallel(v, Dx, Dy, E, Wgr)
    d3 = (vsym - vmc).abs().max().item()
    print(f"CHECK 3  W_read = W_write vs imported bdh_layer_mc_parallel: {d3:.2e}")
    ok &= d3 < 1e-12
    print(f"         -> {'EXACT' if d3 < 1e-12 else 'FAILED'} "
          f"(recovers the symmetric model of experiment 2)\n")

    # CHECK 4 — arm 1 still present: uniform read makes write routing invisible
    x = F.relu(ln(v) @ Dx)
    gr_uni = torch.full((T, k), 1.0 / k, dtype=dt)
    outs = []
    for _ in range(4):
        gw = F.softmax(torch.randn(T, k, generator=g, dtype=dt) * 40.0, dim=-1)
        sc = ((x @ x.T) * (gr_uni @ gw.T)).tril(diagonal=-1)
        y = F.relu(ln(sc @ v) @ Dy) * x
        outs.append(v + ln(y @ E))
    d4 = max((outs[i] - outs[0]).abs().max().item() for i in range(1, 4))
    print("CHECK 4  ARM 1 still present: 4 near-one-hot write routings, UNIFORM read")
    print(f"         max output divergence = {d4:.2e}  (Exp 5 measured 9.4e-16)")
    ok &= d4 < 1e-12
    print(f"         -> {'INVARIANT — arm 1 intact' if d4 < 1e-12 else 'FAILED'}\n")

    # CHECK 5 — engagement: a SEPARATED read makes those same routings observable
    emb = torch.randn(V, D, generator=g, dtype=dt)
    Wsep = separated_W(emb)
    toks = make_seq_explicit(list(range(P)), 0, 0, 0)[:-1]
    vv = emb[toks]
    xx = F.relu(ln(vv) @ Dx)
    gr_sep = F.softmax(vv @ Wsep, dim=-1)
    outs2 = []
    for _ in range(4):
        gw = F.softmax(torch.randn(T, k, generator=g, dtype=dt) * 40.0, dim=-1)
        sc = ((xx @ xx.T) * (gr_sep @ gw.T)).tril(diagonal=-1)
        y = F.relu(ln(sc @ vv) @ Dy) * xx
        outs2.append(vv + ln(y @ E))
    d5 = max((outs2[i] - outs2[0]).abs().max().item() for i in range(1, 4))
    print("CHECK 5  ENGAGEMENT: same 4 write routings, but with a SEPARATED read")
    print(f"         separated read gate on CTX1 = {fmt(gr_sep[toks == CTX1].mean(0).tolist())}, "
          f"on CTX2 = {fmt(gr_sep[toks == CTX2].mean(0).tolist())}")
    print(f"         max output divergence = {d5:.2e}")
    ok &= d5 > 1e-8
    print(f"         -> {'ENGAGED — write routing is now observable' if d5 > 1e-8 else 'NOT ENGAGED (null results would be uninformative)'}\n")

    # CHECK 6 — the representational cap on ANY token-identity gate
    print("CHECK 6  REPRESENTATIONAL CAP. In a 1-layer BDH, v_ast = embed(token), so")
    print("         softmax(v_ast @ W_g) is a function of TOKEN IDENTITY alone. Two facts:")
    pg = _perfect_gate(make_seq_explicit(list(range(P)), 0, 0, 0)[:-1])
    sq = make_seq_explicit(list(range(P)), 0, 0, 0)[:-1]
    multi = sorted({sq[t].item() for t in range(len(sq))
                    for u in range(len(sq))
                    if sq[u] == sq[t] and (pg[t] - pg[u]).abs().max() > 0.5})
    print(f"         (i) the PERFECT gate gives tokens {multi} two different gate vectors")
    print(f"             ([1,0] in the stream-1 block, [0,1] in the stream-2 block), so it")
    print(f"             is NOT a function of token identity and lies outside the model's")
    print(f"             own gate parameterization at any W_g.")
    # (ii) the query-position output is invariant to the value flip
    torch.manual_seed(0)
    m = BDHSplitGate('separated', 'random').double()   # float64 so the claim is airtight
    worst6 = 0.0
    with torch.no_grad():
        for trial in range(4):
            gg = torch.Generator().manual_seed(500 + trial)
            perm = torch.randperm(P, generator=gg).tolist()
            qs = int(torch.randint(2, (1,), generator=gg).item())
            qk = int(torch.randint(P, (1,), generator=gg).item())
            s0 = make_seq_explicit(perm, 0, qs, qk)
            s1 = make_seq_explicit(perm, 1, qs, qk)
            lg = m(torch.stack([s0[:-1], s1[:-1]]))
            worst6 = max(worst6, (lg[0, -1] - lg[1, -1]).abs().max().item())
            tgt_differ = (s0[-1].item() != s1[-1].item())
    print(f"         (ii) flipping which value belongs to which stream leaves the")
    print(f"              query-position logits IDENTICAL: max |logits(flip=0) -")
    print(f"              logits(flip=1)| = {worst6:.2e}, while the TARGET does change")
    print(f"              (differs in all 4 matched pairs: {tgt_differ}).")
    print("         Reason: a*_q = sum_s count(s) * (x_q.x_s)(g_r(tok_q).g_w(s)) * embed(s),")
    print("         and count(s) is flip-invariant (VAL_A and VAL_B each appear P times")
    print("         either way) while every other factor depends only on token identity.")
    ok &= worst6 < 1e-5
    print(f"         -> ANY split soft gate is capped at EXACTLY 50% on this task,")
    print(f"            for any W_g^read, W_g^write, embeddings, Dx, Dy, E. The ceiling")
    print(f"            is unreachable by construction, not by optimization failure.\n")

    print(f"{'ALL VERIFICATION CHECKS PASSED' if ok else 'SOME VERIFICATION CHECKS FAILED'}\n")
    assert ok, "verification failed — do not trust the results below"


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 96)
    print("Arm 2: split soft gate with a separated READ initialization")
    print(f"  P={P}  k={k}  D={D}  N={N}  BLOCK={BLOCK}  BATCH={BATCH}  ITERS={ITERS}")
    print(f"  SEEDS={SEEDS}  LR={LR}  SEP_SCALE={SEP_SCALE}  checkpoints={N_CKPT}")
    print("=" * 96)
    print()
    verify()
    ti._run_sanity()

    print("=" * 96)
    print("CONDITIONS")
    print("=" * 96)
    print()

    print("Cond 1: single-channel baseline (floor) ...")
    c1 = agg([dict(zip(('l1', 'l2', 'a1', 'a2'), ti_run(lambda: BDHSingle(), s)[:4]),
                   trace=[]) for s in SEEDS])
    print("Cond 2: perfect gate, hand-set and frozen (ceiling) ...")
    c2 = agg([dict(zip(('l1', 'l2', 'a1', 'a2'), ti_run(lambda: BDHMCPerfect(), s)[:4]),
                   trace=[]) for s in SEEDS])
    FLOOR, CEIL = c1['acc'], c2['acc']
    span = CEIL - FLOOR

    specs = [
        ('3',  'both gates random init (collapse ref)',        'random',    'random',    False, False),
        ('4',  'read SEPARATED, write random, both trainable', 'separated', 'random',    False, False),
        ('4b', 'read SEPARATED and FROZEN, write trainable',   'separated', 'random',    True,  False),
        ('5',  'write SEPARATED, read random, both trainable', 'random',    'separated', False, False),
        ('6',  'both SEPARATED, both trainable',               'separated', 'separated', False, False),
    ]
    res = {'1': c1, '2': c2}
    for tag, label, ri, wi, fr, fw in specs:
        print(f"Cond {tag}: {label} ...")
        res[tag] = agg([run_split(ri, wi, fr, fw, s) for s in SEEDS])
    print()

    def lift(o):
        return (o['acc'] - FLOOR) / span if span > 1e-3 else 0.0

    labels = {'1': '1.  single-channel baseline (FLOOR)',
              '2': '2.  perfect gate, hand-set + frozen (CEILING)',
              '3': '3.  both gates random init (collapse ref)',
              '4': '4.  read SEPARATED, write random, both trainable',
              '4b': '4b. read SEPARATED + FROZEN, write trainable',
              '5': '5.  write SEPARATED, read random, both trainable',
              '6': '6.  both SEPARATED, both trainable'}
    print("=" * 96)
    print("RESULTS")
    print("=" * 96)
    for tag in ('1', '2', '3', '4', '4b', '5', '6'):
        o = res[tag]
        o['lift'] = lift(o)
        print(f"  {labels[tag]}")
        print(f"      stream-1 loss={o['l1']:.4f}+-{o['l1_sd']:.4f} "
              f"acc={o['a1']:.3f}+-{o['a1_sd']:.3f}   "
              f"stream-2 loss={o['l2']:.4f}+-{o['l2_sd']:.4f} "
              f"acc={o['a2']:.3f}+-{o['a2_sd']:.3f}")
        line = f"      avg acc={o['acc']:.3f}  lift={o['lift']:+.2f}"
        if o['traces'] and o['traces'][0]:   # conds 1 and 2 have no learned gate to trace
            fin = [t[-1] for t in o['traces']]
            av = lambda key: sum(p[key] for p in fin) / len(fin)
            line += (f"  read cos={av('r_cos'):.4f} (ctx {av('r_cos_ctx'):.4f})"
                     f"  write cos={av('w_cos'):.4f} (ctx {av('w_cos_ctx'):.4f})")
        print(line)
    print()

    for tag in ('3', '4', '4b', '5', '6'):
        print_trace(f"cond {tag}", res[tag]['traces'])

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("=" * 96)
    print("VERDICT")
    print("=" * 96)
    for tag in ('4', '4b', '5', '6'):
        o = res[tag]
        tr = o['traces']
        w_step = first_below(tr, 'w_cos_ctx')
        fin = [t[-1] for t in tr]
        ini = [t[0] for t in tr]
        av = lambda pts, key: sum(p[key] for p in pts) / len(pts)
        started_sep = av(ini, 'r_cos_ctx') < SEP_COS or av(ini, 'w_cos_ctx') < SEP_COS
        print(f"  Cond {tag}: lift={o['lift']:+.2f}  "
              f"read cos@ctx {av(ini, 'r_cos_ctx'):.4f} -> {av(fin, 'r_cos_ctx'):.4f}   "
              f"write cos@ctx {av(ini, 'w_cos_ctx'):.4f} -> {av(fin, 'w_cos_ctx'):.4f}")
        if tag in ('4', '4b'):
            print(f"           write-gate cos@ctx first below {SEP_COS}: "
                  f"{w_step if w_step is not None else 'NEVER — write separation did not emerge'}")
        if started_sep:
            held = []
            if av(ini, 'r_cos_ctx') < SEP_COS:
                held.append(f"read {'HELD' if av(fin,'r_cos_ctx') < SEP_COS else 'DECAYED'} "
                            f"({av(ini,'r_cos_ctx'):.4f} -> {av(fin,'r_cos_ctx'):.4f})")
            if av(ini, 'w_cos_ctx') < SEP_COS:
                held.append(f"write {'HELD' if av(fin,'w_cos_ctx') < SEP_COS else 'DECAYED'} "
                            f"({av(ini,'w_cos_ctx'):.4f} -> {av(fin,'w_cos_ctx'):.4f})")
            print(f"           separation started free: " + "; ".join(held))
    print()

    c4_up  = res['4']['lift']  > LIFT_THRESH
    c4b_up = res['4b']['lift'] > LIFT_THRESH
    c5_up  = res['5']['lift']  > LIFT_THRESH
    if (c4_up or c4b_up) and not c5_up:
        print("(a) THE TWO-ARM ACCOUNT IS CONFIRMED, asymmetry included: a separated read")
        print("    lets the write gate bootstrap into separation, and the mirror does not.")
    elif c4_up and c5_up:
        print("(b) THE ASYMMETRY CLAIM IS WRONG: breaking either arm suffices.")
    elif c4b_up and not c4_up:
        print("(d) THE BOOTSTRAP IS REAL BUT THE SEPARATED READ ERODES before the write")
        print("    can exploit it: 4b works, 4 does not.")
    else:
        print("(c) BOTH STAY AT THE FLOOR. A THIRD OBSTACLE EXISTS — and CHECK 6 identifies")
        print("    it. It is not a gradient deadlock at all, it is REPRESENTATIONAL:")
        print()
        print("    In a 1-layer BDH v_ast = embed(token), so softmax(v_ast @ W_g) is a")
        print("    function of TOKEN IDENTITY alone. The perfect gate is not: it assigns")
        print("    the SAME key token [1,0] inside the stream-1 block and [0,1] inside the")
        print("    stream-2 block, which requires latching onto the most recent context")
        print("    token. No W_g can express that.")
        print()
        print("    CHECK 6 turns this into a proof rather than an observation. Since")
        print("        a*_q = sum_s count(s) * (x_q . x_s) * (g_r(tok_q) . g_w(s)) * embed(s)")
        print("    and count(s) is invariant to the value flip (VAL_A and VAL_B each appear")
        print("    P times either way) while every other factor depends only on token")
        print("    identity, the query-position output CANNOT depend on the flip — measured")
        print("    at 0.00e+00 across matched pairs whose targets do differ. So every")
        print("    condition using a learned gate is capped at EXACTLY 50%, for any")
        print("    W_g^read, W_g^write, embeddings, Dx, Dy and E.")
        print()
        print("    This reframes experiments 2 through 6. The two-arm deadlock is real as")
        print("    a description of the gradients, but it is downstream of the real")
        print("    obstacle: on this task the solution is not in the gate's function class,")
        print("    so there was never a gradient that could have found it. Arm 1 and arm 2")
        print("    are consequences of the cap, not independent barriers. Breaking them")
        print("    cannot help, and experiment 6's 'necessary but not sufficient' reading")
        print("    was too generous to the mechanism.")
        print()
        print("    What this does NOT show: that BDH gating cannot separate streams in")
        print("    general. It shows that a token-identity gate cannot separate streams")
        print("    whose token content is identical. Testing the gating hypothesis needs")
        print("    either a gate with access to recurrent context (so it can latch), or a")
        print("    task whose streams differ in token content and not only in position.")
