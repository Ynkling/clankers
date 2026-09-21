"""
test_asymmetric_routing.py — Does decoupling the Hebbian WRITE gate from the READ gate
rescue channel separation on the two-stream interference task?

Motivation (from the prior experiments in this repo):
  test_interference.py showed the k=2 soft gate does NOT spontaneously separate the two
  streams, and test_interference_discrete.py showed hard straight-through (STE) routing
  does not rescue it either. The diagnosis: write and read share ONE gate, so the router
  can only separate memories by WITHHOLDING writes to a channel — but a withheld write is
  exactly what the plasticity-driven loss punishes, because the same gate value also
  scales the read that the loss is computed from. Write-suppression and read-suppression
  are welded together.

This experiment breaks the weld by making the two sides ASYMMETRIC:
  WRITE: commits to a single channel per token via Gumbel-Softmax (the Concrete
         relaxation with reparameterized gradients, NOT straight-through — STE was
         already tested in test_interference_discrete.py and estimates gradients poorly
         here), with temperature annealed TAU_START -> TAU_END over training.
  READ:  blends all channels with the ordinary soft gate, exactly as before.

Because the multi-channel score factorizes as (x_t . x_tau) * (g_t . g_tau), asymmetric
routing is expressible by using the READ gate on the query row and the WRITE gate on the
source column:
    scores[t, tau] = (x_t . x_tau) * (g_read[t] . g_write[tau])
which is exactly the recurrent system
    write:  S^(c) += g_write[t,c] * outer(x_t, v*_t)
    read :  a*_t   = sum_c g_read[t,c] * (x_t @ S^(c))
(verified against an explicit token-by-token loop in CHECK 4 below).

Task, metrics and lift fraction are imported unchanged from test_interference.py
(conflicting two-stream shared-key setup, per-stream query loss/accuracy, loss isolated
to the query position only, >=3 seeds, mean +- std). bdh_recurrent.py and
test_interference.py are imported, never modified.

Conditions (>= 3 seeds):
  1. single-channel baseline                        (floor reference, ~50% analytically)
  2. MC k=2, symmetric soft gate                    (the previously-collapsing config)
  3. MC k=2, asymmetric: Gumbel hard write + soft read   (THE TEST CONDITION)
  3b. same, but with separate write/read gate matrices   (does full parameter decoupling
                                                          change the answer?)
  4. MC k=2, perfect hand-set gate                  (ceiling reference)
  ref. MC k=2, uniform gate                         (the floor used by test_interference.py,
                                                     reported so lift is comparable to it)

Verification: hard-write routing is stochastic, so the parallel-vs-recurrent identity
used in bdh_recurrent.py cannot be applied naively. Four numeric checks are reported
instead (numbers printed, and asserted):
  1. k=1 reduces EXACTLY to single-channel BDH (softmax over one logit is identically 1,
     so the gate vanishes for any temperature and any noise draw).
  2. temperature held very high -> the write distribution flattens to uniform [1/k..1/k],
     and since g_read . [1/k..1/k] = 1/k for ANY read gate, the whole model converges to
     the uniform-gate baseline (test_interference.py's cond-4 floor, whose g.g is also
     1/k). Reported as a sweep so the O(1/tau) convergence is visible.
     NOTE: high temperature does NOT converge to softmax(logits); the Concrete
     relaxation flattens toward UNIFORM as tau -> infinity. The statement that the hard
     write is faithful to the soft gate belongs at the level of which channel is chosen,
     which is CHECK 3.
  3. the Gumbel-max property: P(argmax of the write draw = c) equals the soft gate
     softmax(logits)_c exactly, at every temperature. So a committed single-channel write
     is an UNBIASED one-sample realization of the soft-gate baseline's mixture weights.
  4. (machine precision, possible because the sampled noise is held fixed) the factorized
     score form equals an explicit token-by-token hard-write / soft-read Hebbian loop.
     This is what pins down the gate orientation (read on rows, write on columns).
  5. STRUCTURAL (added after the result came back negative, and it explains why): because
     the write gate's rows sum to 1, sum_c S^(c) is independent of the write routing, so
     with a uniform read gate a*_t = (1/k) x_t @ sum_c S^(c) and even a perfectly
     committed write is INVISIBLE to the read. dL/dW_write is exactly zero there.
     Verified to machine precision across wildly different near-one-hot routings.

RESULT (3 seeds): (b) — decoupling does not fix it. Floor 0.500, ceiling 1.000; the
symmetric soft gate gets 0.496, asymmetric-shared 0.500, asymmetric-separate 0.496, all
at lift 0.00. The write DOES commit (mean max write weight 0.87 when sampled as in
training) but never separates the streams: at the end of training stream-1 writes
[0.335, 0.665] and stream-2 writes [0.448, 0.552] — both prefer the same channel. The
routing trace shows separation never emerges at any checkpoint, and the read-gate cosine
stays at 0.96-0.99 throughout. CHECK 5 is the reason: the deadlock is structural, so
hardening the write cannot break it.
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
    key_tok, make_batch, BDHSingle, BDHMCMC, BDHMCPerfect, run as ti_run,
)

# ── Gumbel-Softmax temperature schedule (annealed over training) ──────────────
TAU_START = 2.0
TAU_END   = 0.5

# ── Diagnostics ───────────────────────────────────────────────────────────────
CKPT_FRACS  = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]   # when to snapshot routing during training
PROBE_B     = 256                                 # sequences in the diagnostic probe batch
DEAD_THRESH = 0.90                                # >90% of all tokens on one channel = dead
SEP_THRESH  = 0.70                                # per-stream write mass for "separated"
LIFT_THRESH = 0.70                                # lift fraction for "climbs to the ceiling"


def tau_at(step, total):
    """Geometric anneal from TAU_START down to TAU_END across `total` steps."""
    if total <= 1:
        return TAU_END
    return TAU_START * (TAU_END / TAU_START) ** (step / (total - 1))


def gumbel_write(logits, tau, gen=None, noise=None):
    """
    Concrete / Gumbel-Softmax sample (hard=False — reparameterized, NOT straight-through).
    Annealing tau downward drives the draw toward a one-hot commitment to a single channel.
    `noise` lets a caller hold the draw fixed (used by the verification checks).
    """
    if noise is None:
        u = torch.rand(logits.shape, generator=gen, dtype=logits.dtype)
        noise = -torch.log(-torch.log(u + 1e-20) + 1e-20)
    return F.softmax((logits + noise) / tau, dim=-1)


# ── Asymmetric layer: hard (committed) write, soft (blended) read ─────────────
def bdh_layer_asym(v_ast, Dx, Dy, E, W_gw, W_gr, tau, gen=None, noise=None,
                   sample=True):
    """
    scores[t, tau'] = (x_t . x_tau') * (g_read[t] . g_write[tau'])
    sample=False uses the noise-free write distribution softmax(logits/tau) — the
    deterministic evaluation form.
    """
    x  = F.relu(ln(v_ast) @ Dx)                       # (T, N)
    lw = v_ast @ W_gw                                 # (T, k) write logits
    gw = (gumbel_write(lw, tau, gen, noise) if sample
          else F.softmax(lw / tau, dim=-1))           # (T, k) committed write
    gr = F.softmax(v_ast @ W_gr, dim=-1)              # (T, k) blended read
    scores = ((x @ x.T) * (gr @ gw.T)).tril(diagonal=-1)
    a_ast  = scores @ v_ast
    y      = F.relu(ln(a_ast) @ Dy) * x
    v_next = v_ast + ln(y @ E)
    return v_next, gw, gr


def bdh_layer_asym_recurrent(v_ast, Dx, Dy, E, gw, gr):
    """Explicit token-by-token hard-write / soft-read Hebbian loop (verification only)."""
    T, Dd = v_ast.shape
    Nn, kk = Dx.shape[1], gw.shape[1]
    x = F.relu(ln(v_ast) @ Dx)
    S = torch.zeros(kk, Nn, Dd, dtype=v_ast.dtype)
    a_ast = torch.zeros(T, Dd, dtype=v_ast.dtype)
    for t in range(T):
        reads = torch.einsum('n,knd->kd', x[t], S)            # READ: x_t @ S^(c)
        a_ast[t] = gr[t] @ reads                              # blended by the SOFT gate
        S = S + torch.einsum('k,nd->knd', gw[t],              # WRITE: committed channel
                             torch.outer(x[t], v_ast[t]))
    y = F.relu(ln(a_ast) @ Dy) * x
    return v_ast + ln(y @ E)


class BDHMCAsym(nn.Module):
    """Same body as test_interference.py's BDHMCMC; only the gating is asymmetric."""

    def __init__(self, separate_gate_params=False):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.Dx    = nn.Parameter(torch.randn(D, N) * 0.1)
        self.Dy    = nn.Parameter(torch.randn(D, N) * 0.1)
        self.E     = nn.Parameter(torch.randn(N, D) * 0.1)
        self.head  = nn.Linear(D, V, bias=False)
        self.W_gr  = nn.Parameter(torch.randn(D, k) * 0.1)
        self.separate = separate_gate_params
        # shared case: one tied Parameter under two names (parameters() de-duplicates it)
        self.W_gw = nn.Parameter(torch.randn(D, k) * 0.1) if separate_gate_params else self.W_gr

    def forward(self, tokens, tau, sample=True):
        v = self.embed(tokens)
        out = []
        for b in range(tokens.shape[0]):
            v_next, *_ = bdh_layer_asym(v[b], self.Dx, self.Dy, self.E,
                                        self.W_gw, self.W_gr, tau, sample=sample)
            out.append(v_next)
        return self.head(torch.stack(out))


# ── Per-position stream labels (latched from the most recent CTX token) ───────
def stream_labels(tokens):
    """(B, T) long tensor in {-1, 0, 1}; -1 before any context token has been seen."""
    B, T = tokens.shape
    labels = torch.full((B, T), -1, dtype=torch.long)
    for b in range(B):
        cur = -1
        for t in range(T):
            tok = tokens[b, t].item()
            if tok == CTX1:
                cur = 0
            elif tok == CTX2:
                cur = 1
            labels[b, t] = cur
    return labels


@torch.no_grad()
def routing_diagnostics(model, tokens, labels, tau):
    """
    Noise-free write distribution at `tau` and soft read gate, summarized per stream.
    Returns fractions of each stream's tokens WRITTEN to each channel (by argmax),
    read-gate cosine between streams (all tokens and at the query position), the mean
    write commitment (max write weight), and the dead-channel flag.
    """
    v  = model.embed(tokens)                              # (B, T, D)
    gw = F.softmax((v @ model.W_gw) / tau, dim=-1)        # (B, T, k)
    gr = F.softmax(v @ model.W_gr, dim=-1)                # (B, T, k)
    warg = gw.argmax(-1)                                  # (B, T) committed channel

    valid = labels >= 0
    m0, m1 = valid & (labels == 0), valid & (labels == 1)
    frac = lambda m, c: (warg[m] == c).float().mean().item() if m.any() else float('nan')

    all_ch = [(warg[valid] == c).float().mean().item() for c in range(k)]
    g0, g1 = gr[m0].mean(0), gr[m1].mean(0)
    cos_all = F.cosine_similarity(g0.unsqueeze(0), g1.unsqueeze(0)).item()

    # query-position read gate, exactly as test_interference.py reports it
    q_lab = labels[:, -1]
    gq = gr[:, -1, :]
    gq0, gq1 = gq[q_lab == 0].mean(0), gq[q_lab == 1].mean(0)
    cos_q = F.cosine_similarity(gq0.unsqueeze(0), gq1.unsqueeze(0)).item()

    return dict(
        s0_w=[frac(m0, c) for c in range(k)],
        s1_w=[frac(m1, c) for c in range(k)],
        all_w=all_ch,
        read_cos_all=cos_all,
        read_cos_query=cos_q,
        commit=gw[valid].max(-1).values.mean().item(),
        dead=max(all_ch) > DEAD_THRESH,
    )


# ── Training + evaluation for the asymmetric condition ───────────────────────
def run_asym(separate_gate_params, seed):
    """Mirrors test_interference.run(): same seeding, batches, loss and eval protocol."""
    torch.manual_seed(seed)
    model = BDHMCAsym(separate_gate_params=separate_gate_params)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    rng   = torch.Generator(); rng.manual_seed(seed + 10_000)

    probe_rng = torch.Generator(); probe_rng.manual_seed(seed + 99_000)
    probe_tokens, _ = make_batch(PROBE_B, probe_rng)
    probe_inp = probe_tokens[:, :-1]
    probe_lab = stream_labels(probe_inp)

    ckpt_steps = sorted({min(int(f * ITERS), ITERS - 1) for f in CKPT_FRACS})
    trace = []
    for step in range(ITERS):
        tau = tau_at(step, ITERS)
        if step in ckpt_steps:
            d = routing_diagnostics(model, probe_inp, probe_lab, tau)
            d['step'], d['tau'] = step, tau
            trace.append(d)
        tokens, _ = make_batch(BATCH, rng)
        inp, tgt  = tokens[:, :-1], tokens[:, 1:]
        logits    = model(inp, tau, sample=True)
        loss      = F.cross_entropy(logits[:, -1, :], tgt[:, -1])   # query position only
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    d = routing_diagnostics(model, probe_inp, probe_lab, TAU_END)
    d['step'], d['tau'] = ITERS, TAU_END
    trace.append(d)

    with torch.no_grad():
        tokens, q_streams = make_batch(512, rng)
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        logits   = model(inp, TAU_END, sample=False)   # deterministic write at final tau
        q_logits, q_tgt = logits[:, -1, :], tgt[:, -1]
        correct = (q_logits.argmax(-1) == q_tgt)
        m1, m2  = (q_streams == 0), (q_streams == 1)
        loss_s1 = F.cross_entropy(q_logits[m1], q_tgt[m1]).item()
        loss_s2 = F.cross_entropy(q_logits[m2], q_tgt[m2]).item()
        acc_s1  = correct[m1].float().mean().item()
        acc_s2  = correct[m2].float().mean().item()

    return loss_s1, loss_s2, acc_s1, acc_s2, trace


def ms(xs):
    m = sum(xs) / len(xs)
    return m, (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def summarize(results):
    l1, l2, a1, a2 = zip(*[(r[0], r[1], r[2], r[3]) for r in results])
    ml1, sl1 = ms(l1); ml2, sl2 = ms(l2)
    ma1, sa1 = ms(a1); ma2, sa2 = ms(a2)
    return dict(ml1=ml1, sl1=sl1, ml2=ml2, sl2=sl2,
                ma1=ma1, sa1=sa1, ma2=ma2, sa2=sa2, acc=(ma1 + ma2) / 2)


def print_cond(name, s, floor=None, ceil=None):
    print(f"-- {name} --")
    print(f"  stream-1: loss={s['ml1']:.4f}+-{s['sl1']:.4f}  acc={s['ma1']:.3f}+-{s['sa1']:.3f}")
    print(f"  stream-2: loss={s['ml2']:.4f}+-{s['sl2']:.4f}  acc={s['ma2']:.3f}+-{s['sa2']:.3f}")
    if floor is not None and ceil is not None:
        span = ceil - floor
        s['lift'] = (s['acc'] - floor) / span if span > 1e-3 else 0.0
        print(f"  avg acc={s['acc']:.3f}   lift fraction (acc-floor)/(ceil-floor) = {s['lift']:.2f}")
    else:
        print(f"  avg acc={s['acc']:.3f}")
    print()


def print_trace(name, traces):
    """Routing over training, averaged across seeds."""
    print(f"  routing trace over training ({name}, mean over {len(traces)} seeds):")
    print(f"    {'step':>5} {'tau':>5} | {'s1 writes ->':>20} | {'s2 writes ->':>20} |"
          f" {'read cos':>8} {'cos@q':>7} {'commit':>7} | dead")
    for i in range(len(traces[0])):
        pts = [t[i] for t in traces]
        s0 = [sum(p['s0_w'][c] for p in pts) / len(pts) for c in range(k)]
        s1 = [sum(p['s1_w'][c] for p in pts) / len(pts) for c in range(k)]
        aw = [sum(p['all_w'][c] for p in pts) / len(pts) for c in range(k)]
        rc = sum(p['read_cos_all'] for p in pts) / len(pts)
        rq = sum(p['read_cos_query'] for p in pts) / len(pts)
        cm = sum(p['commit'] for p in pts) / len(pts)
        dead = any(p['dead'] for p in pts)
        f = lambda xs: "[" + ", ".join(f"{x:.3f}" for x in xs) + "]"
        print(f"    {pts[0]['step']:>5} {pts[0]['tau']:>5.2f} | {f(s0):>20} | {f(s1):>20} |"
              f" {rc:>8.4f} {rq:>7.4f} {cm:>7.3f} | "
              f"{'DEAD ' + f(aw) if dead else 'no'}")
    print()
    return traces


# ── Verification ──────────────────────────────────────────────────────────────
def verify():
    print("=" * 78)
    print("VERIFICATION (numeric; hard-write routing has no unconditional parallel form)")
    print("=" * 78)
    g = torch.Generator().manual_seed(4242)
    T, Nv = BLOCK - 1, N
    dt = torch.float64
    v  = torch.randn(T, D, generator=g, dtype=dt)
    Dx = torch.randn(D, Nv, generator=g, dtype=dt) * 0.1
    Dy = torch.randn(D, Nv, generator=g, dtype=dt) * 0.1
    E  = torch.randn(Nv, D, generator=g, dtype=dt) * 0.1
    ok = True

    # CHECK 1 — k=1 reduces exactly to single-channel BDH
    W1w = torch.randn(D, 1, generator=g, dtype=dt) * 0.5
    W1r = torch.randn(D, 1, generator=g, dtype=dt) * 0.5
    vs, *_ = bdh_layer_parallel(v, Dx, Dy, E)
    worst1 = 0.0
    for tau in (0.1, 0.5, 2.0, 50.0):
        va, gw1, _ = bdh_layer_asym(v, Dx, Dy, E, W1w, W1r, tau, gen=g)
        worst1 = max(worst1, (va - vs).abs().max().item(),
                     (gw1 - 1.0).abs().max().item())
    print(f"CHECK 1  k=1 asymmetric vs single-channel bdh_layer_parallel")
    print(f"         max |v_asym - v_single| over tau in (0.1, 0.5, 2, 50) = {worst1:.2e}")
    print(f"         (write gate is identically 1.0 for every tau and every noise draw)")
    ok &= worst1 < 1e-12
    print(f"         -> {'EXACT reduction' if worst1 < 1e-12 else 'FAILED'}\n")

    # CHECK 2 — high temperature -> uniform-gate baseline (test_interference cond-4 floor)
    Wgw = torch.randn(D, k, generator=g, dtype=dt) * 0.5
    Wgr = torch.randn(D, k, generator=g, dtype=dt) * 0.5
    v_uni, *_ = bdh_layer_mc_parallel(v, Dx, Dy, E, torch.zeros(D, k, dtype=dt))
    print("CHECK 2  temperature -> infinity collapses the write to uniform [1/k..1/k], and")
    print("         g_read . [1/k..1/k] = 1/k for ANY read gate, so the model converges to")
    print("         the uniform-gate floor (whose g.g is also 1/k):")
    print(f"         {'tau':>10} {'max |v_asym - v_uniform_gate|':>32}")
    last = None
    for tau in (1.0, 1e1, 1e2, 1e4, 1e6, 1e8):
        noise = -torch.log(-torch.log(torch.rand(T, k, generator=g, dtype=dt) + 1e-20) + 1e-20)
        va, *_ = bdh_layer_asym(v, Dx, Dy, E, Wgw, Wgr, tau, noise=noise)
        last = (va - v_uni).abs().max().item()
        print(f"         {tau:>10.0e} {last:>32.3e}")
    print("         (converges as O(1/tau), as the relaxation predicts)")
    ok &= last < 1e-6
    print(f"         -> {'CONVERGES to the soft/uniform-gate baseline' if last < 1e-6 else 'FAILED'}\n")

    # CHECK 3 — the committed write is an unbiased draw from the soft gate
    M = 40_000
    logits = torch.randn(k, generator=g, dtype=dt) * 1.2
    soft = F.softmax(logits, dim=-1)
    print("CHECK 3  Gumbel-max property: which channel the committed write picks is")
    print("         distributed EXACTLY as the soft gate softmax(logits), at every tau.")
    print(f"         soft gate = [{', '.join(f'{p:.4f}' for p in soft)}]   ({M} draws)")
    worst3 = 0.0
    for tau in (0.5, 2.0):
        nz = -torch.log(-torch.log(torch.rand(M, k, generator=g, dtype=dt) + 1e-20) + 1e-20)
        draws = gumbel_write(logits.expand(M, k), tau, noise=nz).argmax(-1)
        emp = torch.tensor([(draws == c).double().mean() for c in range(k)])
        dev = (emp - soft).abs().max().item()
        worst3 = max(worst3, dev)
        print(f"         tau={tau:<4} empirical argmax freq = "
              f"[{', '.join(f'{p:.4f}' for p in emp)}]   max dev = {dev:.4f}")
    tol = 4.0 / M ** 0.5
    ok &= worst3 < tol
    print(f"         -> {'MATCHES' if worst3 < tol else 'FAILED'} the soft gate "
          f"(max dev {worst3:.4f} < sampling tolerance {tol:.4f})\n")

    # CHECK 4 — factorized score form == explicit Hebbian loop (noise held fixed)
    noise = -torch.log(-torch.log(torch.rand(T, k, generator=g, dtype=dt) + 1e-20) + 1e-20)
    vp, gw, gr = bdh_layer_asym(v, Dx, Dy, E, Wgw, Wgr, 0.5, noise=noise)
    vr = bdh_layer_asym_recurrent(v, Dx, Dy, E, gw, gr)
    d4 = (vp - vr).abs().max().item()
    print("CHECK 4  factorized scores vs explicit token-by-token hard-write/soft-read")
    print("         Hebbian loop, with the sampled write draw held fixed")
    print(f"         max |v_parallel - v_recurrent| = {d4:.2e}  (pins the gate orientation:")
    print("         read gate on the query row, write gate on the source column)")
    ok &= d4 < 1e-12
    print(f"         -> {'IDENTICAL' if d4 < 1e-12 else 'FAILED'}\n")

    # CHECK 5 — why asymmetry alone cannot bootstrap (structural, not a soundness check)
    #   sum_c S^(c) = sum_tau (sum_c g_write[tau,c]) outer(x_tau, v*_tau)
    #               = sum_tau outer(x_tau, v*_tau)          (write gate rows sum to 1)
    # so with g_read = [1/k..1/k], a*_t = (1/k) x_t @ sum_c S^(c) — the write routing
    # cancels completely. A committed write is INVISIBLE to a uniform read, hence
    # dL/dW_write is exactly zero there and the write gate gets no signal to separate.
    gr_uni = torch.full((T, k), 1.0 / k, dtype=dt)
    outs, commits = [], []
    for _ in range(4):
        lw = torch.randn(T, k, generator=g, dtype=dt) * 5.0    # wildly different routings
        gwi = gumbel_write(lw, 0.01, gen=g)                    # driven to near one-hot
        commits.append(gwi.max(-1).values.mean().item())
        x = F.relu(ln(v) @ Dx)
        a = ((x @ x.T) * (gr_uni @ gwi.T)).tril(diagonal=-1) @ v
        y = F.relu(ln(a) @ Dy) * x
        outs.append(v + ln(y @ E))
    d5 = max((outs[i] - outs[0]).abs().max().item() for i in range(1, 4))
    # ...and the common value is the ungated single-channel form with scores scaled by 1/k
    x_s = F.relu(ln(v) @ Dx)
    a_s = ((x_s @ x_s.T) / k).tril(diagonal=-1) @ v
    y_s = F.relu(ln(a_s) @ Dy) * x_s
    d5b = (outs[0] - (v + ln(y_s @ E))).abs().max().item()
    print("CHECK 5  STRUCTURAL: a committed write is invisible to a UNIFORM read gate.")
    print("         sum_c S^(c) is independent of the write routing (gate rows sum to 1),")
    print("         so g_read = [1/k..1/k] gives a*_t = (1/k) x_t @ sum_c S^(c).")
    print(f"         4 near-one-hot write routings (mean commitment "
          f"{sum(commits)/len(commits):.4f}), uniform read:")
    print(f"           max |output difference| between them = {d5:.3e}")
    print(f"           max |output - single-channel with scores/k| = {d5b:.3e}")
    print("         -> dL/dW_write is EXACTLY zero at a uniform read gate. The write gate")
    print("            cannot learn to separate until the read gate separates first, and")
    print("            the read gate has nothing to separate until the write gate does.")
    print("            Asymmetry alone does not break this deadlock.\n")
    ok &= d5 < 1e-12 and d5b < 1e-12

    print(f"{'ALL VERIFICATION CHECKS PASSED' if ok else 'SOME VERIFICATION CHECKS FAILED'}\n")
    assert ok, "verification failed — do not trust the results below"


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 78)
    print("Asymmetric routing: Gumbel hard WRITE + soft READ on the interference task")
    print(f"  P={P}  k={k}  D={D}  N={N}  BLOCK={BLOCK}  BATCH={BATCH}  ITERS={ITERS}")
    print(f"  SEEDS={SEEDS}  LR={LR}  TAU_START={TAU_START} -> TAU_END={TAU_END} (geometric)")
    print("=" * 78)
    print()
    verify()

    ti._run_sanity()

    print("=" * 78)
    print("CONDITIONS")
    print("=" * 78)
    print()

    print("Cond 1: single-channel baseline (floor) ...")
    c1 = summarize([ti_run(lambda: BDHSingle(), s) for s in SEEDS])
    print_cond("Cond 1: single-channel baseline (FLOOR)", c1)

    print("Cond 4: MC k=2, perfect hand-set gate (ceiling) ...")
    c4 = summarize([ti_run(lambda: BDHMCPerfect(), s) for s in SEEDS])
    print_cond("Cond 4: MC k=2, perfect gate (CEILING)", c4)

    FLOOR, CEIL = c1['acc'], c4['acc']
    c1['lift'], c4['lift'] = 0.0, 1.0

    print("Reference: MC k=2, uniform gate (the floor test_interference.py used) ...")
    cu = summarize([ti_run(lambda: BDHMCMC(True), s) for s in SEEDS])
    print_cond("Reference: MC k=2, uniform gate", cu, FLOOR, CEIL)

    print("Cond 2: MC k=2, symmetric soft gate (previously collapsing) ...")
    c2_raw = [ti_run(lambda: BDHMCMC(False), s, collect_gates=True) for s in SEEDS]
    c2 = summarize(c2_raw)
    print_cond("Cond 2: MC k=2, symmetric soft gate", c2, FLOOR, CEIL)
    gi = [r[4] for r in c2_raw if r[4]]
    if gi:
        cos2 = sum(g[2] for g in gi) / len(gi)
        g1 = [sum(g[0][c] for g in gi) / len(gi) for c in range(k)]
        g2 = [sum(g[1][c] for g in gi) / len(gi) for c in range(k)]
        print(f"  gate @query  stream-1: [{', '.join(f'{x:.3f}' for x in g1)}]")
        print(f"  gate @query  stream-2: [{', '.join(f'{x:.3f}' for x in g2)}]")
        print(f"  cosine(g_s1, g_s2) = {cos2:.4f}  "
              f"({'SEPARATED' if cos2 < 0.5 else 'not separated'})\n")

    print("Cond 3: MC k=2, asymmetric Gumbel hard write + soft read ...")
    c3_raw = [run_asym(False, s) for s in SEEDS]
    c3 = summarize(c3_raw)
    print_cond("Cond 3: MC k=2, asymmetric (shared gate matrix)", c3, FLOOR, CEIL)
    tr3 = print_trace("cond 3", [r[4] for r in c3_raw])

    print("Cond 3b: same, with SEPARATE write/read gate matrices ...")
    c3b_raw = [run_asym(True, s) for s in SEEDS]
    c3b = summarize(c3b_raw)
    print_cond("Cond 3b: MC k=2, asymmetric (separate gate matrices)", c3b, FLOOR, CEIL)
    tr3b = print_trace("cond 3b", [r[4] for r in c3b_raw])

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  Cond 1  single-channel (FLOOR)      : acc={c1['acc']:.3f}")
    print(f"  Cond 2  symmetric soft gate         : acc={c2['acc']:.3f}  lift={c2['lift']:.2f}")
    print(f"  Cond 3  asymmetric (shared W_g)     : acc={c3['acc']:.3f}  lift={c3['lift']:.2f}")
    print(f"  Cond 3b asymmetric (separate W_g)   : acc={c3b['acc']:.3f}  lift={c3b['lift']:.2f}")
    print(f"  Cond 4  perfect gate (CEILING)      : acc={c4['acc']:.3f}")
    print(f"  (ref)   uniform gate                : acc={cu['acc']:.3f}  lift={cu['lift']:.2f}")
    print()

    for label, cond, tr in (("Cond 3", c3, tr3), ("Cond 3b", c3b, tr3b)):
        fin = [t[-1] for t in tr]
        s0 = [sum(p['s0_w'][c] for p in fin) / len(fin) for c in range(k)]
        s1 = [sum(p['s1_w'][c] for p in fin) / len(fin) for c in range(k)]
        aw = [sum(p['all_w'][c] for p in fin) / len(fin) for c in range(k)]
        cm = sum(p['commit'] for p in fin) / len(fin)
        rc = sum(p['read_cos_all'] for p in fin) / len(fin)
        dead = max(aw) > DEAD_THRESH
        separated = (max(s0) > SEP_THRESH and max(s1) > SEP_THRESH
                     and (s0[0] > s0[1]) != (s1[0] > s1[1]))
        cond.update(sep=separated, dead=dead, s0=s0, s1=s1, commit=cm, rcos=rc)
        print(f"  {label} final write routing: stream-1 -> "
              f"[{', '.join(f'{x:.3f}' for x in s0)}]   stream-2 -> "
              f"[{', '.join(f'{x:.3f}' for x in s1)}]")
        print(f"  {label} write separation: {separated}   dead-channel collapse: {dead}"
              f"   mean write commitment: {cm:.3f}   read-gate cosine: {rc:.4f}")
    print()

    win = c3 if c3['lift'] >= c3b['lift'] else c3b
    tag = "Cond 3" if win is c3 else "Cond 3b"
    if win['sep'] and not win['dead'] and win['lift'] > LIFT_THRESH:
        print(f"(a) DECOUPLING WORKS. {tag} achieves write separation "
              f"(stream-1 -> [{', '.join(f'{x:.3f}' for x in win['s0'])}], stream-2 -> "
              f"[{', '.join(f'{x:.3f}' for x in win['s1'])}]) and lift climbs to "
              f"{win['lift']:.0%} of the ceiling. Separating the hard Hebbian write from")
        print("    the soft read breaks the collapse that the symmetric gate suffers.")
    else:
        why = []
        if win['dead']:
            why.append(f"dead-channel collapse (>{DEAD_THRESH:.0%} of all tokens written "
                       "to a single channel)")
        elif not win['sep']:
            why.append("the write gate never separates the streams (no stable "
                       "stream-to-channel correspondence)")
        else:
            why.append("the write gate DOES separate the streams, but the separation buys "
                       "no accuracy")
        if win['lift'] <= LIFT_THRESH:
            why.append(f"lift stays at {win['lift']:.0%} of the ceiling "
                       f"(acc={win['acc']:.3f} vs floor={FLOOR:.3f}, ceil={CEIL:.3f})")
        print(f"(b) DECOUPLING DOES NOT FIX IT. Best asymmetric variant ({tag}): "
              + "; ".join(why) + ".")
        print("    Making the Hebbian write commit to one channel while the read stays")
        print("    soft is NOT sufficient to make the router separate interfering streams")
        print("    under end-to-end training.")
        print()
        print("    CHECK 5 above gives the reason, and it is structural rather than an")
        print("    optimization accident: because the write gate's rows sum to 1, the")
        print("    channel sum sum_c S^(c) does not depend on the routing at all, so a")
        print("    uniform read gate makes even a perfectly committed write invisible")
        print("    (verified to machine precision). The gradient on the write gate is")
        print("    exactly zero there. Training starts with a near-uniform read gate, so")
        print("    the write gate begins with no signal telling it how to route, and the")
        print("    read gate has no separated memories to exploit — neither side can move")
        print("    first. Hardening the write changes HOW the write commits, not WHETHER")
        print("    the read can see it, so it cannot break the deadlock. Any fix has to")
        print(f"    make the read gate separate too (cond 4 hand-sets it and reaches "
              f"{CEIL:.3f}).")
