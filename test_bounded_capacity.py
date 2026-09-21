"""
test_bounded_capacity.py — Does per-channel bounded capacity break the exact
cancellation that makes the multi-channel gate collapse?

Motivation (from test_asymmetric_routing.py, CHECK 5):
  The collapse is an EXACT cancellation. Because the gate is normalized
  (sum_c g[t,c] = 1) and Hebbian accumulation is linear,
      sum_c S^(c) = sum_tau (sum_c g[tau,c]) outer(x_tau, v*_tau)
                  = sum_tau outer(x_tau, v*_tau)
  is invariant to the routing. Under a uniform read a*_t = (1/k) x_t @ sum_c S^(c),
  so the routing cancels and the write-gate gradient is identically zero (verified
  there to 9.4e-16 across wildly different near-one-hot routings).

  Bounded capacity makes accumulation NONLINEAR. After each Hebbian write, each
  channel's state is projected onto a Frobenius-norm ball of radius C:
      S^(c) <- S^(c) * min(1, C / ||S^(c)||_F)
  Once the clip binds, sum_c S^(c) depends on HOW the mass was split, because a
  channel carrying more mass is scaled down more. The cancellation is broken and a
  gradient on the gate exists.

  The gate stays the plain SYMMETRIC soft gate (one shared read/write gate, exactly
  as in test_interference.py), so bounded capacity is the only change from the
  original collapsing architecture.

IMPORTANT — what bounded capacity ACTUALLY does (CHECK 4, measured, and it is not what
it looks like): the projection is applied after EVERY write, so it is not a global
rescale that BDH's parameter-free LayerNorm would remove. Each clip scales down
everything already stored, then the next token adds a fresh full-size outer product, so
a memory written at token t is multiplied down once per subsequent clip. Bounded
capacity is therefore a strong FORGETTING / recency mechanism. Measured at C=1.0 over a
26-token sequence, the surviving weight on token 0 is 7.2e-24 against 1.0 on the last
token, and cos(S_bounded, S_unbounded) = 0.22 — nowhere near the 1.0 a pure rescale
would give. This happens even under a uniform gate, where the two channels stay exactly
identical and the clip binds equally on both.

That matters for reading the results two ways. It is why the ceiling and floor MUST be
recomputed at every C (done below) rather than compared against the unbounded ceiling.
And it is a genuine confound: the intervention does two things at once, breaking the
cancellation AND destroying old memories, and a C small enough to bind hard may make the
task unsolvable for ANY gate. A C whose recomputed ceiling has collapsed onto its floor
carries no information about separation, so such C are reported separately rather than
counted as evidence either way.

Bounded capacity cannot be expressed in the parallel form (it never materializes the
state), so this file implements a batched recurrent multi-channel layer with clipping
and VERIFIES it against the imported bdh_layer_mc_recurrent / bdh_layer_mc_parallel /
bdh_layer_parallel before any training runs. bdh_recurrent.py and test_interference.py
are imported, never modified.

Task, metrics and lift-fraction definition are reused unchanged from
test_interference.py (conflicting two-stream shared-key setup, per-stream query loss
and accuracy, loss isolated to the query position only, >=3 seeds, mean +- std).

Conditions (>= 3 seeds each; ceiling and floor are RECOMPUTED under bounded capacity
at every C — the unbounded ceiling is never used as the comparison point):
  1. single-channel baseline, bounded                 (floor reference, per C)
  2. MC k=2, UNBOUNDED, learned gate                  (reproduces the known collapse)
  3. MC k=2, bounded, learned gate                    (THE TEST CONDITION, per C)
  4. MC k=2, bounded, perfect hand-set gate           (recomputed CEILING, per C)
  5. MC k=2, bounded, uniform gate                    (recomputed FLOOR, per C)

Verdict:
 (a) some C breaks the cancellation AND capacity binds AND lift climbs substantially
     toward the recomputed ceiling -> bounded capacity rescues the mechanism;
 (b) the cancellation breaks and capacity binds but lift stays near the floor -> the
     gradient exists but does not point toward separation (the direction the routing
     actually moved is reported);
 (c) no C both binds and breaks the cancellation -> the intervention is INERT on this
     task, which is not a negative result about separation at all.
"""

import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from bdh_recurrent import (
    ln, bdh_layer_parallel, bdh_layer_mc_parallel, bdh_layer_mc_recurrent,
)
import test_interference as ti
from test_interference import (
    P, k, D, N, BATCH, ITERS, LR, SEEDS, BLOCK, CTX1, CTX2, V, VAL_A, VAL_B,
    key_tok, make_batch, _perfect_gate, run as ti_run, BDHMCMC,
)

# ── Capacity sweep (C = Frobenius-ball radius per channel; None = unbounded) ──
# Calibrated against the measured end-of-sequence ||S^(c)||_F (printed below), so the
# sweep really does span "never binds" to "binds hard".
#   C=128 never binds; C=48 binds while retaining ~87% of the earliest write; C=32 and
#   C=16 bind progressively harder with memory degrading; C=8 erases the early writes
#   almost entirely (surviving weight ~1e-4). Measured by the calibration table below.
C_SWEEP = [None, 128.0, 48.0, 32.0, 16.0, 8.0]

# ── Thresholds ────────────────────────────────────────────────────────────────
BIND_THRESH  = 0.05    # saturation fraction above which capacity meaningfully binds
BREAK_THRESH = 1e-8    # CHECK-3 divergence above which the cancellation is broken
LIFT_THRESH  = 0.70    # lift fraction counting as "climbs toward the ceiling"
SEP_THRESH   = 0.70    # per-stream gate mass counting as separated
SPAN_THRESH  = 0.10    # recomputed ceiling must beat its floor by this much for the
                       # lift fraction at that C to carry any information at all
PROBE_B      = 256     # sequences in the diagnostic probe batch


# ── Batched recurrent multi-channel BDH with per-channel capacity clipping ───
def mc_bounded(v, Dx, Dy, E, g_write, g_read, C, track_sat=False):
    """
    Token-by-token Hebbian MC-BDH with a Frobenius-ball projection after each write.

        read : a*_t = sum_c g_read[t,c] * (x_t @ S^(c))     (S holds only tau < t)
        write: S^(c) += g_write[t,c] * outer(x_t, v*_t)
        clip : S^(c) <- S^(c) * min(1, C / ||S^(c)||_F)

    v: (B,T,D); g_write, g_read: (B,T,k); C: float or None (unbounded).
    g_write and g_read are the SAME tensor for every training condition here; they are
    separable only so the cancellation-breaking diagnostic can drive them apart.
    Returns v_next (B,T,D) and, if track_sat, the per-channel fraction of (token,
    sequence) events at which the clip actually bound.
    """
    B, T, Dd = v.shape
    Nn = Dx.shape[1]
    kk = g_write.shape[-1]
    x = F.relu(ln(v) @ Dx)                                  # (B,T,N) — depends only on v
    S = torch.zeros(B, kk, Nn, Dd, dtype=v.dtype)
    a_list = []
    sat = torch.zeros(kk, dtype=v.dtype)
    for t in range(T):
        reads = torch.einsum('bn,bknd->bkd', x[:, t], S)              # (B,k,D)
        a_list.append(torch.einsum('bk,bkd->bd', g_read[:, t], reads))
        outer = torch.einsum('bn,bd->bnd', x[:, t], v[:, t])          # (B,N,D)
        S = S + g_write[:, t].view(B, kk, 1, 1) * outer.unsqueeze(1)
        if C is not None:
            nrm = S.reshape(B, kk, -1).norm(dim=-1)                   # (B,k) Frobenius
            scale = (C / (nrm + 1e-12)).clamp(max=1.0)                # (B,k)
            if track_sat:
                sat = sat + (scale < 1.0).to(v.dtype).sum(0)
            S = S * scale.view(B, kk, 1, 1)
    a_ast = torch.stack(a_list, dim=1)                                # (B,T,D)
    y = F.relu(ln(a_ast) @ Dy) * x
    v_next = v + ln(y @ E)
    return (v_next, sat / (B * T)) if track_sat else (v_next, None)


def state_trace(v, Dx, g_write, C):
    """
    Final per-channel state, the largest |S^(0) - S^(1)| seen along the way, and the
    surviving weight each token's write retains at the end (the product of all later
    clip factors). Verification/diagnostic only.
    """
    B, T, Dd = v.shape
    Nn, kk = Dx.shape[1], g_write.shape[-1]
    x = F.relu(ln(v) @ Dx)
    S = torch.zeros(B, kk, Nn, Dd, dtype=v.dtype)
    maxdiff, factors = 0.0, []
    for t in range(T):
        S = S + g_write[:, t].view(B, kk, 1, 1) * torch.einsum(
            'bn,bd->bnd', x[:, t], v[:, t]).unsqueeze(1)
        if C is not None:
            nrm = S.reshape(B, kk, -1).norm(dim=-1)
            sc = (C / (nrm + 1e-12)).clamp(max=1.0)
            S = S * sc.view(B, kk, 1, 1)
            factors.append(sc[0, 0].item())
        else:
            factors.append(1.0)
        if kk > 1:
            maxdiff = max(maxdiff, (S[:, 0] - S[:, 1]).abs().max().item())
    surv = []
    for t in range(T):
        p = 1.0
        for f in factors[t + 1:]:
            p *= f
        surv.append(p)
    return S, maxdiff, surv


def single_bounded_ref(v, Dx, Dy, E, C):
    """Independent single-sequence single-channel bounded loop (verification only)."""
    T, Dd = v.shape
    x = F.relu(ln(v) @ Dx)
    S = torch.zeros(Dx.shape[1], Dd, dtype=v.dtype)
    a = torch.zeros(T, Dd, dtype=v.dtype)
    for t in range(T):
        a[t] = x[t] @ S
        S = S + torch.outer(x[t], v[t])
        if C is not None:
            S = S * min(1.0, C / (S.norm().item() + 1e-12))
    y = F.relu(ln(a) @ Dy) * x
    return v + ln(y @ E)


# ── Models (same body as test_interference.py; only accumulation is bounded) ──
class BDHBounded(nn.Module):
    """gate_mode: 'learned' | 'uniform' | 'perfect'. n_ch=1 gives the single-channel model."""

    def __init__(self, C, n_ch=k, gate_mode='learned'):
        super().__init__()
        self.C, self.n_ch, self.gate_mode = C, n_ch, gate_mode
        self.embed = nn.Embedding(V, D)
        self.Dx    = nn.Parameter(torch.randn(D, N) * 0.1)
        self.Dy    = nn.Parameter(torch.randn(D, N) * 0.1)
        self.E     = nn.Parameter(torch.randn(N, D) * 0.1)
        self.head  = nn.Linear(D, V, bias=False)
        if gate_mode == 'uniform':
            self.register_buffer('W_g', torch.zeros(D, n_ch))
        elif gate_mode == 'learned':
            self.W_g = nn.Parameter(torch.randn(D, n_ch) * 0.1)

    def gate(self, tokens, v):
        if self.gate_mode == 'perfect':
            return torch.stack([_perfect_gate(tokens[b]) for b in range(tokens.shape[0])])
        return F.softmax(v @ self.W_g, dim=-1)

    def forward(self, tokens, track_sat=False):
        v = self.embed(tokens)
        g = self.gate(tokens, v)
        v_next, sat = mc_bounded(v, self.Dx, self.Dy, self.E, g, g, self.C,
                                 track_sat=track_sat)
        return self.head(v_next), sat


# ── Per-position stream labels (latched from the most recent CTX token) ──────
def stream_labels(tokens):
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


# ── Training + evaluation (mirrors test_interference.run exactly) ────────────
def run_bounded(C, n_ch, gate_mode, seed):
    torch.manual_seed(seed)
    model = BDHBounded(C, n_ch=n_ch, gate_mode=gate_mode)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    rng   = torch.Generator(); rng.manual_seed(seed + 10_000)

    for _ in range(ITERS):
        tokens, _ = make_batch(BATCH, rng)
        inp, tgt  = tokens[:, :-1], tokens[:, 1:]
        logits, _ = model(inp)
        loss = F.cross_entropy(logits[:, -1, :], tgt[:, -1])   # query position only
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        tokens, q_streams = make_batch(512, rng)
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        logits, _ = model(inp)
        q_logits, q_tgt = logits[:, -1, :], tgt[:, -1]
        correct = (q_logits.argmax(-1) == q_tgt)
        m1, m2 = (q_streams == 0), (q_streams == 1)
        out = dict(
            l1=F.cross_entropy(q_logits[m1], q_tgt[m1]).item(),
            l2=F.cross_entropy(q_logits[m2], q_tgt[m2]).item(),
            a1=correct[m1].float().mean().item(),
            a2=correct[m2].float().mean().item(),
        )
        # saturation + gate routing on a fixed probe batch
        prng = torch.Generator(); prng.manual_seed(seed + 99_000)
        ptok, _ = make_batch(PROBE_B, prng)
        pinp = ptok[:, :-1]
        _, sat = model(pinp, track_sat=True)
        out['sat'] = sat.tolist() if sat is not None else [0.0] * n_ch
        out['tdiv'] = trained_cancellation_divergence(model, pinp[:32])
        lab = stream_labels(pinp)
        g = model.gate(pinp, model.embed(pinp))
        g0, g1 = g[lab == 0].mean(0), g[lab == 1].mean(0)
        out['g_s1'], out['g_s2'] = g0.tolist(), g1.tolist()
        out['gcos'] = (F.cosine_similarity(g0.unsqueeze(0), g1.unsqueeze(0)).item()
                       if n_ch > 1 else 1.0)
        # query-position gate, exactly as test_interference.py reports it
        gq = g[:, -1, :]
        ql = lab[:, -1]
        out['gcos_q'] = (F.cosine_similarity(gq[ql == 0].mean(0).unsqueeze(0),
                                             gq[ql == 1].mean(0).unsqueeze(0)).item()
                         if n_ch > 1 else 1.0)
    return out


def ms(xs):
    m = sum(xs) / len(xs)
    return m, (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def agg(runs):
    o = {}
    for key in ('l1', 'l2', 'a1', 'a2', 'gcos', 'gcos_q'):
        o[key], o[key + '_sd'] = ms([r[key] for r in runs])
    o['acc'] = (o['a1'] + o['a2']) / 2
    nc = len(runs[0]['sat'])
    td = [r['tdiv'] for r in runs if r.get('tdiv') is not None]
    o['tdiv'] = (sum(td) / len(td)) if td else None
    o['sat']  = [sum(r['sat'][c] for r in runs) / len(runs) for c in range(nc)]
    o['g_s1'] = [sum(r['g_s1'][c] for r in runs) / len(runs) for c in range(nc)]
    o['g_s2'] = [sum(r['g_s2'][c] for r in runs) / len(runs) for c in range(nc)]
    return o


def fmt(xs):
    return "[" + ", ".join(f"{x:.3f}" for x in xs) + "]"


# ── Cancellation-breaking diagnostic (CHECK 3) ───────────────────────────────
def analytical_floor_gap(C_frac):
    """
    Replicates test_interference._run_sanity's analytical argument WITH clipping.

    That argument is what caps single-channel / uniform-gate at 50%: the Hebbian state
    accumulates outer(VAL_A, KEY_i) + outer(VAL_B, KEY_i) for every key, so the read
    from any key is EXACTLY equally aligned with VAL_A and VAL_B whatever the assignment.
    It assumes the two writes carry equal weight. Clipping breaks that assumption —
    stream-2 writes after stream-1 within each pair, so the earlier write is decayed more
    and the alignment gap becomes nonzero. C_frac is C as a fraction of the unbounded
    final ||S||_F. Returns (mean |score_A - score_B| over keys, clipped fraction).
    """
    emb = torch.zeros(V, D, dtype=torch.float64)
    for i in range(V):
        emb[i, i % D] = 1.0
    va, vb = emb[VAL_A], emb[VAL_B]

    def build(C):
        S = torch.zeros(D, D, dtype=torch.float64)
        nclip = nw = 0
        for i in range(P):
            ki = emb[key_tok(i)]
            for val in (va, vb):          # stream-1 writes, then stream-2, as in the task
                S = S + torch.outer(val, ki)
                nw += 1
                if C is not None and S.norm().item() > C:
                    S = S * (C / S.norm().item())
                    nclip += 1
        return S, nclip / nw

    S_unb, _ = build(None)
    C = None if C_frac is None else S_unb.norm().item() * C_frac
    S, cf = build(C)
    gaps = [abs(((S @ emb[key_tok(i)]) @ va).item() - ((S @ emb[key_tok(i)]) @ vb).item())
            for i in range(P)]
    return sum(gaps) / len(gaps), cf


@torch.no_grad()
def trained_cancellation_divergence(model, tokens, n_draws=4, seed=3):
    """
    Same probe as CHECK 3, but using a TRAINED model's own parameters and real task
    sequences, so it answers 'is the cancellation broken for THIS model?' rather than
    for an untrained one at an arbitrary scale. Returns None for a single-channel model.
    """
    if model.n_ch < 2:
        return None
    g = torch.Generator().manual_seed(seed)
    v = model.embed(tokens)
    B, T, _ = v.shape
    gr = torch.full((B, T, model.n_ch), 1.0 / model.n_ch)
    outs = []
    for _ in range(n_draws):
        gw = F.softmax(torch.randn(B, T, model.n_ch, generator=g) * 40.0, dim=-1)
        out, _ = mc_bounded(v, model.Dx, model.Dy, model.E, gw, gr, model.C)
        outs.append(out)
    return max((outs[i] - outs[0]).abs().max().item() for i in range(1, n_draws))


def cancellation_divergence(C, n_draws=4, seed=7, dtype=torch.float64):
    """
    Replicates test_asymmetric_routing.py CHECK 5 with clipping added: several
    near-one-hot WRITE routings evaluated under a UNIFORM read. Unbounded this was
    9.4e-16 (exactly invariant). Returns (max output divergence, mean saturation).
    """
    g = torch.Generator().manual_seed(seed)
    T = BLOCK - 1
    v  = torch.randn(1, T, D, generator=g, dtype=dtype)
    Dx = torch.randn(D, N, generator=g, dtype=dtype) * 0.1
    Dy = torch.randn(D, N, generator=g, dtype=dtype) * 0.1
    E  = torch.randn(N, D, generator=g, dtype=dtype) * 0.1
    gr = torch.full((1, T, k), 1.0 / k, dtype=dtype)
    outs, sats = [], []
    for _ in range(n_draws):
        logits = torch.randn(1, T, k, generator=g, dtype=dtype) * 40.0
        gw = F.softmax(logits, dim=-1)                       # near one-hot per token
        out, sat = mc_bounded(v, Dx, Dy, E, gw, gr, C, track_sat=True)
        outs.append(out)
        sats.append(sat.mean().item() if sat is not None else 0.0)
    div = max((outs[i] - outs[0]).abs().max().item() for i in range(1, n_draws))
    return div, sum(sats) / len(sats)


# ── Verification ──────────────────────────────────────────────────────────────
def verify():
    print("=" * 78)
    print("VERIFICATION (all numeric, reported before any training condition)")
    print("=" * 78)
    g = torch.Generator().manual_seed(4242)
    T, dt = BLOCK - 1, torch.float64
    v  = torch.randn(T, D, generator=g, dtype=dt)
    Dx = torch.randn(D, N, generator=g, dtype=dt) * 0.1
    Dy = torch.randn(D, N, generator=g, dtype=dt) * 0.1
    E  = torch.randn(N, D, generator=g, dtype=dt) * 0.1
    W_g = torch.randn(D, k, generator=g, dtype=dt) * 0.5
    vb = v.unsqueeze(0)
    ok = True

    # CHECK 1 — C -> infinity reduces exactly to the unbounded MC model
    gg = F.softmax(v @ W_g, dim=-1).unsqueeze(0)
    v_par, *_ = bdh_layer_mc_parallel(v, Dx, Dy, E, W_g)
    v_rec, *_ = bdh_layer_mc_recurrent(v, Dx, Dy, E, W_g)
    d_none, _ = mc_bounded(vb, Dx, Dy, E, gg, gg, None)
    d_huge, _ = mc_bounded(vb, Dx, Dy, E, gg, gg, 1e30)
    e1a = (d_none[0] - v_par).abs().max().item()
    e1b = (d_none[0] - v_rec).abs().max().item()
    e1c = (d_huge[0] - v_par).abs().max().item()
    print("CHECK 1  C -> infinity reduces to the unbounded multi-channel model")
    print(f"         C=None vs imported bdh_layer_mc_parallel  : {e1a:.2e}")
    print(f"         C=None vs imported bdh_layer_mc_recurrent : {e1b:.2e}")
    print(f"         C=1e30 vs imported bdh_layer_mc_parallel  : {e1c:.2e}")
    w1 = max(e1a, e1b, e1c); ok &= w1 < 1e-12
    print(f"         -> {'EXACT' if w1 < 1e-12 else 'FAILED'}\n")

    # CHECK 2 — k=1 reduces to single-channel BDH, with the same clipping applied
    g1 = torch.ones(1, T, 1, dtype=dt)
    v_single, *_ = bdh_layer_parallel(v, Dx, Dy, E)
    u1, _ = mc_bounded(vb, Dx, Dy, E, g1, g1, None)
    e2u = (u1[0] - v_single).abs().max().item()
    print("CHECK 2  k=1 reduces to single-channel BDH (same clipping applied)")
    print(f"         k=1, C=None vs imported bdh_layer_parallel : {e2u:.2e}")
    worst2 = e2u
    for C in (16.0, 4.0, 1.0, 0.25):
        b1, _ = mc_bounded(vb, Dx, Dy, E, g1, g1, C)
        ref = single_bounded_ref(v, Dx, Dy, E, C)
        e = (b1[0] - ref).abs().max().item()
        worst2 = max(worst2, e)
        print(f"         k=1, C={C:<6} vs independent single-channel bounded loop : {e:.2e}")
    ok &= worst2 < 1e-12
    print(f"         -> {'EXACT' if worst2 < 1e-12 else 'FAILED'}\n")

    # CHECK 3 — cancellation-breaking as a function of C (THE CRITICAL ONE)
    print("CHECK 3  CANCELLATION-BREAKING: 4 near-one-hot write routings under a")
    print("         UNIFORM read. Unbounded this was 9.4e-16 in test_asymmetric_routing")
    print("         CHECK 5 (exactly routing-invariant). Divergence vs C:")
    print(f"         {'C':>10} {'max output divergence':>24} {'mean saturation':>17}   status")
    div_by_C = {}
    for C in C_SWEEP:
        dv, st = cancellation_divergence(C)
        div_by_C[C] = (dv, st)
        status = ("cancellation BROKEN" if dv > BREAK_THRESH else "still cancels (inert)")
        print(f"         {str(C):>10} {dv:>24.3e} {st:>17.3f}   {status}")
    ok &= div_by_C[None][0] < 1e-12
    print(f"         -> unbounded reproduces the exact cancellation "
          f"({div_by_C[None][0]:.2e}); bounded C values that stay near 1e-15 have")
    print("            NOT engaged, and any null result at those C is uninformative.\n")

    # CHECK 4 — what bounded capacity actually does (it is also a forgetting mechanism)
    print("CHECK 4  SCOPE: repeated projection is NOT a global rescale, so it is not")
    print("         removed by BDH's parameter-free LayerNorm. Each clip scales down")
    print("         everything already stored before the next token adds a full-size")
    print("         outer product, so bounded capacity also FORGETS. Uniform gate,")
    print("         where the channels stay identical and the clip binds equally:")
    gu = torch.full((1, T, k), 1.0 / k, dtype=dt)
    S_unb, md_unb, _ = state_trace(vb, Dx, gu, None)
    print(f"         {'C':>7} {'sat':>6} {'max|S0-S1|':>12} {'cos(S_bnd,S_unb)':>18}"
          f" {'surviving weight tok0':>23}")
    ok4 = (md_unb < 1e-14)
    for C in (200.0, 4.0, 1.0, 0.25):
        S_b, md_b, surv = state_trace(vb, Dx, gu, C)
        a_, b_ = S_unb[0, 0].flatten(), S_b[0, 0].flatten()
        cos = F.cosine_similarity(a_.unsqueeze(0), b_.unsqueeze(0)).item()
        _, sat = mc_bounded(vb, Dx, Dy, E, gu, gu, C, track_sat=True)
        ok4 &= (md_b < 1e-14)     # channels must stay identical under a uniform gate
        print(f"         {C:>7} {sat.mean().item():>6.3f} {md_b:>12.2e} {cos:>18.4f}"
              f" {surv[0]:>23.2e}")
    ok &= ok4
    print("         -> under a uniform gate the channels are EXACTLY identical (clipping")
    print("            binds equally), yet the state is not a rescaled copy of the")
    print("            unbounded one: old memories are multiplied down once per clip.")
    print("            Bounded capacity therefore breaks the cancellation AND imposes")
    print("            forgetting. A C that binds hard enough to erase the early writes")
    print("            makes the task unsolvable for ANY gate, which is why the ceiling")
    print("            and floor are recomputed at every C below.\n")

    # CHECK 5 — clipping VOIDS the analytical 50% bound that defines this task's floor
    print("CHECK 5  FLOOR INTEGRITY: test_interference._run_sanity proves single-channel /")
    print("         uniform-gate cannot exceed 50%, because the state accumulates")
    print("         outer(VAL_A,KEY_i) + outer(VAL_B,KEY_i) with EQUAL weight, so the read")
    print("         from any key is exactly equally aligned with both values. That")
    print("         assumes equal weight. Clipping decays the earlier write more (stream-2")
    print("         writes after stream-1 within each pair), so the gap opens up:")
    g_unb, _ = analytical_floor_gap(None)
    print(f"         {'C / ||S||_unbounded':>21} {'clipped frac':>13} "
          f"{'mean |score_A - score_B|':>26}")
    print(f"         {'unbounded':>21} {0.0:>13.2f} {g_unb:>26.4f}")
    worst5 = 0.0
    for frac in (0.8, 0.6, 0.4, 0.25):
        gp, cf = analytical_floor_gap(frac)
        worst5 = max(worst5, gp)
        print(f"         {frac:>21.2f} {cf:>13.2f} {gp:>26.4f}")
    ok &= (g_unb < 1e-12 < worst5)
    print("         -> the 50% cap holds EXACTLY only while capacity is unbounded. Once")
    print("            clipping binds, a model can read the stream off the recency")
    print("            asymmetry instead of separating channels, so the floor of this")
    print("            task is no longer 50% and floor/ceiling must be re-read per C.\n")

    print(f"{'ALL VERIFICATION CHECKS PASSED' if ok else 'SOME VERIFICATION CHECKS FAILED'}\n")
    assert ok, "verification failed — do not trust the results below"
    return div_by_C


def calibrate():
    """Print the unbounded end-of-sequence ||S^(c)||_F so the C sweep is interpretable."""
    g = torch.Generator().manual_seed(1)
    tok, _ = make_batch(64, g)
    inp = tok[:, :-1]
    torch.manual_seed(0)
    m = BDHBounded(None, n_ch=k, gate_mode='uniform')
    with torch.no_grad():
        v = m.embed(inp)
        x = F.relu(ln(v) @ m.Dx)
        gu = torch.full((v.shape[0], v.shape[1], k), 1.0 / k)
        S = torch.zeros(v.shape[0], k, N, D)
        for t in range(v.shape[1]):
            S = S + gu[:, t].view(-1, k, 1, 1) * torch.einsum(
                'bn,bd->bnd', x[:, t], v[:, t]).unsqueeze(1)
        nrm = S.reshape(S.shape[0], k, -1).norm(dim=-1)
    print(f"CALIBRATION: unbounded end-of-sequence ||S^(c)||_F at an untrained uniform "
          f"gate = {nrm.mean().item():.2f} (min {nrm.min().item():.2f}, "
          f"max {nrm.max().item():.2f})")
    # how much of an early write still survives at query time, per C
    gen2 = torch.Generator().manual_seed(4242)
    T = BLOCK - 1
    vv = torch.randn(1, T, D, generator=gen2, dtype=torch.float64)
    dxx = torch.randn(D, N, generator=gen2, dtype=torch.float64) * 0.1
    gu = torch.full((1, T, k), 1.0 / k, dtype=torch.float64)
    print(f"             the first key-value pair is written near token 2 and queried at "
          f"token {T - 1}:")
    print(f"             {'C':>8} {'surviving weight on token 2':>30}")
    for C in C_SWEEP:
        _, _, surv = state_trace(vv, dxx, gu, C)
        print(f"             {str(C):>8} {surv[2]:>30.2e}")
    print(f"             C sweep {C_SWEEP} therefore spans 'never binds' through "
          f"'binds with\n             memory intact' to 'binds hard and erases it'.\n")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 78)
    print("Bounded per-channel capacity on the two-stream interference task")
    print(f"  P={P}  k={k}  D={D}  N={N}  BLOCK={BLOCK}  BATCH={BATCH}  ITERS={ITERS}")
    print(f"  SEEDS={SEEDS}  LR={LR}  C_SWEEP={C_SWEEP}")
    print("=" * 78)
    print()
    calibrate()
    div_by_C = verify()
    ti._run_sanity()

    t0 = time.time()
    print("=" * 78)
    print("CONDITION 2 (C-independent): MC k=2, UNBOUNDED, learned gate")
    print("=" * 78)
    c2 = agg([run_bounded(None, k, 'learned', s) for s in SEEDS])
    print(f"  stream-1: loss={c2['l1']:.4f}+-{c2['l1_sd']:.4f}  "
          f"acc={c2['a1']:.3f}+-{c2['a1_sd']:.3f}")
    print(f"  stream-2: loss={c2['l2']:.4f}+-{c2['l2_sd']:.4f}  "
          f"acc={c2['a2']:.3f}+-{c2['a2_sd']:.3f}")
    print(f"  avg acc={c2['acc']:.3f}   gate stream-1 {fmt(c2['g_s1'])}  "
          f"stream-2 {fmt(c2['g_s2'])}   cos={c2['gcos']:.4f} (cos@q={c2['gcos_q']:.4f})")
    print()

    results = {}
    for C in C_SWEEP:
        if C is None:
            continue
        print("=" * 78)
        print(f"C = {C}   (CHECK-3 divergence {div_by_C[C][0]:.3e}, "
              f"diagnostic saturation {div_by_C[C][1]:.3f})")
        print("=" * 78)
        r = {}
        for tag, n_ch, mode in (('floor1', 1, 'learned'),
                                ('test',   k, 'learned'),
                                ('ceil',   k, 'perfect'),
                                ('floor',  k, 'uniform')):
            r[tag] = agg([run_bounded(C, n_ch, mode, s) for s in SEEDS])
        CEIL, FLOOR = r['ceil']['acc'], r['floor']['acc']
        span = CEIL - FLOOR
        for tag in r:
            r[tag]['lift'] = ((r[tag]['acc'] - FLOOR) / span) if span > 1e-3 else 0.0
        r['div'], r['satdiag'] = div_by_C[C]
        results[C] = r

        names = [('floor1', '1. single-channel, bounded (floor ref)'),
                 ('test',   '3. MC k=2, bounded, learned gate  <-- TEST'),
                 ('ceil',   '4. MC k=2, bounded, perfect gate (CEILING)'),
                 ('floor',  '5. MC k=2, bounded, uniform gate (FLOOR)')]
        for tag, label in names:
            o = r[tag]
            print(f"  {label}")
            print(f"     stream-1 loss={o['l1']:.4f}+-{o['l1_sd']:.4f} "
                  f"acc={o['a1']:.3f}+-{o['a1_sd']:.3f}   "
                  f"stream-2 loss={o['l2']:.4f}+-{o['l2_sd']:.4f} "
                  f"acc={o['a2']:.3f}+-{o['a2_sd']:.3f}")
            print(f"     avg acc={o['acc']:.3f}  lift={o['lift']:+.2f}  "
                  f"saturation={fmt(o['sat'])}  gate s1={fmt(o['g_s1'])} "
                  f"s2={fmt(o['g_s2'])}  cos={o['gcos']:.4f} (cos@q={o['gcos_q']:.4f})")
        print()

    print(f"(total {time.time() - t0:.0f}s)")
    print()

    # ── Summary + verdict ─────────────────────────────────────────────────────
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  {'C':>7} {'div(CHECK3)':>12} {'sat(test)':>18} {'floor':>7} {'ceil':>7} "
          f"{'TEST':>7} {'lift':>7} {'gate cos':>9}")
    for C in C_SWEEP:
        if C is None:
            continue
        r = results[C]
        print(f"  {C:>7} {r['div']:>12.2e} {fmt(r['test']['sat']):>18} "
              f"{r['floor']['acc']:>7.3f} {r['ceil']['acc']:>7.3f} "
              f"{r['test']['acc']:>7.3f} {r['test']['lift']:>+7.2f} "
              f"{r['test']['gcos']:>9.4f}")
    print(f"  {'None':>7} {div_by_C[None][0]:>12.2e} {'n/a (unbounded)':>18} "
          f"{'':>7} {'':>7} {c2['acc']:>7.3f} {'':>7} {c2['gcos']:>9.4f}   <- cond 2")
    print()

    # Classify every C: does capacity bind for the TRAINED model, is the cancellation
    # broken for it, and is the task still able to tell separation from non-separation?
    print("=" * 78)
    print("PER-C CLASSIFICATION")
    print("=" * 78)
    print(f"  {'C':>7} {'binds':>7} {'trained div':>12} {'floor':>7} {'ceil':>7} "
          f"{'span':>7}   task status")
    cls = {}
    for C in C_SWEEP:
        if C is None:
            continue
        r = results[C]
        fl, ce = r['floor']['acc'], r['ceil']['acc']
        span = ce - fl
        binds = max(r['test']['sat']) > BIND_THRESH
        tdiv  = r['test']['tdiv']
        breaks = (tdiv is not None and tdiv > BREAK_THRESH)
        if span > SPAN_THRESH:
            status = "DISCRIMINATIVE (question is askable)"
        elif fl > 0.90:
            status = "DISSOLVED — floor ROSE to the ceiling"
        else:
            status = "DESTROYED — ceiling FELL to the floor"
        cls[C] = dict(binds=binds, breaks=breaks, span=span, status=status,
                      floor=fl, ceil=ce)
        print(f"  {C:>7} {str(binds):>7} {tdiv if tdiv is None else f'{tdiv:.2e}':>12} "
              f"{fl:>7.3f} {ce:>7.3f} {span:>7.3f}   {status}")
    print()

    askable = {C: results[C] for C in cls
               if cls[C]['span'] > SPAN_THRESH and cls[C]['binds']}
    dissolved = sorted(C for C in cls if cls[C]['status'].startswith("DISSOLVED"))
    destroyed = sorted(C for C in cls if cls[C]['status'].startswith("DESTROYED"))
    engaged = {C: results[C] for C in cls if cls[C]['binds'] and cls[C]['breaks']}

    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    if dissolved:
        print(f"  C values where bounded capacity DISSOLVED the task: {dissolved}")
        for C in dissolved:
            r, c = results[C], cls[C]
            print(f"    C={C}: floor (uniform gate) = {c['floor']:.3f}, "
                  f"ceiling (perfect gate) = {c['ceil']:.3f}")
            print(f"           the single-channel model — which has NO channels to route "
                  f"between — scores {r['floor1']['acc']:.3f},")
            print(f"           and the learned gate scores {r['test']['acc']:.3f} while "
                  f"NOT separating (cos={r['test']['gcos']:.4f},")
            print(f"           gate s1={fmt(r['test']['g_s1'])} s2={fmt(r['test']['g_s2'])}). "
                  f"Separation is not needed to win.")
        print("    CHECK 5 is the reason: clipping voids the analytical 50% bound that")
        print("    defines this task's floor, because the earlier of the two same-key")
        print("    writes is decayed more, leaving a recency asymmetry a single channel")
        print("    can read off directly. Lift is undefined where floor == ceiling.")
        print()
    if destroyed:
        print(f"  C values where bounded capacity DESTROYED the task "
              f"(ceiling fell to floor): {destroyed}")
        print()

    if not engaged and not askable:
        print("(c) THE INTERVENTION IS INERT on this task: no swept C both binds and")
        print("    breaks the cancellation. This is NOT a negative result about")
        print("    separation — bounded capacity never got the chance to act.")
    elif not askable:
        print("(c'') THE INTERVENTION ENGAGES BUT THE TASK CANNOT MEASURE IT. Every C that")
        print("    binds also removes the task's ability to distinguish a separating gate")
        print("    from a non-separating one: the floor rises to meet the ceiling (or the")
        print("    ceiling falls to meet the floor), so lift is undefined everywhere the")
        print("    mechanism is active. The cancellation IS broken — that part of the")
        print("    premise holds — but on THIS task the question 'does the resulting")
        print("    gradient point toward separation?' cannot be asked at any C that binds.")
        print("    This is neither (a) nor (b): it is a statement about the task, not")
        print("    about the mechanism. Testing bounded capacity properly needs a task")
        print("    whose floor survives forgetting.")
    else:
        best = max(askable.items(), key=lambda kv: kv[1]['test']['lift'])
        C, r = best
        o = r['test']
        sep = (max(o['g_s1']) > SEP_THRESH and max(o['g_s2']) > SEP_THRESH
               and (o['g_s1'][0] > o['g_s1'][1]) != (o['g_s2'][0] > o['g_s2'][1]))
        print(f"    C values that bind AND keep the task discriminative: "
              f"{sorted(askable, key=lambda c: -askable[c]['test']['lift'])}")
        print(f"    Best of them: C={C}  lift={o['lift']:+.2f}  acc={o['acc']:.3f} "
              f"(recomputed floor {r['floor']['acc']:.3f}, ceiling {r['ceil']['acc']:.3f})")
        print()
        if o['lift'] > LIFT_THRESH and sep:
            print(f"(a) BOUNDED CAPACITY RESCUES THE MECHANISM. At C={C} the cancellation")
            print(f"    is broken (divergence {r['div']:.2e}), capacity binds "
                  f"(saturation {fmt(o['sat'])}), the gate separates")
            print(f"    (stream-1 {fmt(o['g_s1'])}, stream-2 {fmt(o['g_s2'])}, "
                  f"cos={o['gcos']:.4f}), and lift reaches {o['lift']:.0%} of the")
            print("    recomputed bounded ceiling.")
        else:
            print(f"(b) THE GRADIENT EXISTS BUT DOES NOT POINT TOWARD SEPARATION. At C={C}")
            print(f"    the cancellation IS broken (divergence {r['div']:.2e}) and capacity")
            print(f"    DOES bind (saturation {fmt(o['sat'])}), so the write-gate gradient is")
            print("    genuinely nonzero — yet lift stays at "
                  f"{o['lift']:+.0%} of the recomputed ceiling.")
            print(f"    Direction the routing moved: stream-1 gate {fmt(o['g_s1'])}, "
                  f"stream-2 gate {fmt(o['g_s2'])},")
            print(f"    cosine between streams {o['gcos']:.4f} "
                  f"(1.0 = identical, 0.0 = fully separated); "
                  f"{'the two streams moved APART but not enough' if o['gcos'] < 0.99 else 'the two streams stayed locked together'}.")
            print("    Breaking the exact cancellation is therefore necessary but not")
            print("    sufficient: a nonzero gate gradient does not by itself carry")
            print("    information about which stream belongs in which channel.")
