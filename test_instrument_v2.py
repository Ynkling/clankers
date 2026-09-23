"""
test_instrument_v2.py — Rebuilding the instrument, re-deriving the floor, and re-running
experiments 2, 5 and 6 on it.

WHY REBUILD. test_arm2_readinit.py (experiment 7) found that on the original task the
gate could not express the solution at all. In a 1-layer BDH v_ast = embed(token), so
softmax(v_ast @ W_g) is a function of TOKEN IDENTITY alone, while the perfect gate
assigns the SAME key token [1,0] inside the stream-1 block and [0,1] inside the
stream-2 block. Since
    a*_q = sum_s count(s) * (x_q . x_s) * (g_r(tok_q) . g_w(s)) * embed(s)
and count(s) is invariant to the value flip while every other factor depends only on
token identity, the query-position logits could not depend on the flip (8.9e-16 in
float64). Every learned-gate result from experiments 2-6 was therefore measured against
a target outside the model's own function class. Those experiments need re-running on an
instrument that does not have that defect.

THREE CHANGES, each anchored on the imported reference where possible:

  1. U != I. bdh_recurrent.bdh_layer_recurrent already carries the decay slot where the
     paper's U folds into the recurrence (S = decay*(S + outer)). In the parallel form
     that is a decay^(t-tau) mask on the scores, which CHECK 2 verifies reproduces the
     imported recurrent function exactly at several decays. This makes attention
     position-sensitive, so the flip-invariance argument above no longer applies and the
     floor must be re-derived rather than assumed (see FLOOR RE-DERIVATION).

  2. Randomized order. The original task always wrote the stream-1 block before the
     stream-2 block for every pair. Combined with any forgetting that is a systematic
     recency cue — exactly the cue that let bounded capacity lift the floor to 1.000 in
     test_bounded_capacity.py. Here the two blocks of each pair are emitted in random
     order, so position carries no systematic information about which stream a value
     belongs to. CHECK 6 measures the residual cue directly.

  3. Recurrent-state gate. Following bdh_multichannel.py, which already computes its gate
     from a recurrent hidden state rather than the raw embedding:
         h_t = tanh(W_in v_t + W_h h_{t-1});   g_t = softmax(W_g h_t)
     h integrates history, so the gate CAN latch onto the most recent context token.
     CHECK 5 fits this gate to the perfect gate's targets and reports how closely it can
     match, which is the test of whether the representational cap is actually lifted. If
     it cannot match, the instrument is still broken and nothing below means anything.

Task, metrics and lift-fraction definition otherwise follow test_interference.py
(conflicting two-stream shared-key setup, per-stream query loss and accuracy, loss
isolated to the query position, >=3 seeds, mean +- std). bdh_recurrent.py,
bdh_multichannel.py and test_interference.py are imported, never modified.

WHAT IS RE-RUN
  Exp 2 — MC k=2, symmetric learned gate. Does it spontaneously separate now?
  Exp 5 — arm 1: is the write-routing cancellation still exact under a uniform read?
           Plus the asymmetric Gumbel hard-write / soft-read condition.
  Exp 6 — bounded per-channel capacity, swept.

RESULT (3 seeds): THE INSTRUMENT WAS THE PROBLEM.

FLOOR RE-DERIVATION. Measured, not assumed, at every decay: single-channel 0.498,
uniform-gate 0.498, perfect-gate 1.000, span +0.502 at decay 1.0, 0.95 and 0.85 alike.
Randomized order did its job — CHECK 6 shows the flip-invariance PROOF lapses once
U != I (4.6e-16 at decay 1, but 3.6e-02 at 0.95 and 1.5e-01 at 0.85), yet the measured
floor stays at chance, because random block order removes the systematic recency cue
that residual position information would otherwise expose. Operating point decay=0.95.

  Exp 2, recurrent-state gate : acc 1.000  lift +1.00  VAL cos 0.0024  KEY cos 0.0121
  Exp 2, token-identity gate  : acc 0.498  lift +0.00  VAL cos 0.9999  KEY cos 1.0000
  Exp 5, asymmetric Gumbel    : acc 1.000  lift +1.00  VAL cos 0.0053
  Exp 6, bounded capacity     : lift +1.00 at every C tried

The plain SYMMETRIC soft gate reaches the ceiling on its own — no asymmetric routing,
no capacity bound, no separated init. The token-identity control at the same decay on
the same task stays pinned at the floor with VAL cosine 0.9999, which is what pins the
cause on the gate's function class rather than on U != I or the order randomization.

Separation emerges spontaneously and fast. Per role (KEY/VAL carry it; CTX need not
split because their identity already marks the stream), a dense single-seed trace:

    step   acc  |  CTX cos  KEY cos  VAL cos
       0  0.340 |    0.962    0.999    0.999
      40  0.541 |    0.941    0.857    0.977
      60  0.651 |    0.803    0.019    0.050
      80  1.000 |    0.938    0.001    0.002
    1199  1.000 |    0.946    0.002    0.002

and it is stable for the remaining 1120 steps.

WHAT THIS DOES TO EXPERIMENTS 2-6. Arm 1 is still true and still irrelevant: the
write-routing cancellation under an EXACTLY uniform read is still exact here (1.33e-15)
and U != I does not disturb it, so experiment 5's mathematics was right. But it never
was a barrier. It is a measure-zero stationary point — the gate is only near-uniform at
initialization, never exactly uniform — and once the solution is inside the function
class the resulting gradient leaves it within ~60 steps. The two-arm deadlock described
the gradients correctly and misdiagnosed their significance: it was a description of a
system that could not move because it had nowhere to go, not one trapped by its
gradients. The interventions retire with it — asymmetric routing and bounded capacity
both reach lift +1.00, exactly what doing nothing achieves.

One residue of the old recency artifact survives: at C=16 the bounded floor rises to
0.741 and the span narrows to +0.259, so hard clipping still partially dissolves the
task even with randomized order, just far less than the +0.502 -> 0.000 collapse seen
in test_bounded_capacity.py.
"""

import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from bdh_recurrent import (
    ln, bdh_layer_parallel, bdh_layer_recurrent, bdh_layer_mc_parallel,
)
import test_interference as ti
from test_interference import (
    P, k, D, N, BATCH, ITERS, LR, SEEDS, BLOCK, CTX1, CTX2, VAL_A, VAL_B, V,
    key_tok, _perfect_gate,
)

# ── Instrument constants ─────────────────────────────────────────────────────
DECAY_SWEEP = [1.0, 0.95, 0.85]      # U = decay * I  (decay=1 recovers U = I)
# The decay the floor re-derivation selects as the operating point: the first non-unity
# decay whose measured floor is intact and whose span is live. Hoisted to module level so
# other scripts (test_phase2_seeds.py) can import the operating point instead of
# re-deriving or hard-coding it. The main block below still checks it against the
# measured sweep and says so if they disagree.
OPERATING_DECAY = 0.95
H_GATE      = 32                      # recurrent gate hidden width
GATE_FIT_ITERS = 400                  # supervised fit of the gate to perfect-gate targets
C_SWEEP     = [None, 48.0, 16.0]      # bounded-capacity radii for the Exp-6 re-run
TAU_START, TAU_END = 2.0, 0.5         # Gumbel schedule for the Exp-5 re-run
SEP_COS     = 0.50
LIFT_THRESH = 0.70
N_CKPT      = 11
PROBE_B     = 256


# ── Task: same structure, but the two blocks of each pair are in random order ─
def make_seq(gen):
    flip = int(torch.randint(2, (1,), generator=gen).item())
    val_s1 = VAL_A if flip == 0 else VAL_B
    val_s2 = VAL_B if flip == 0 else VAL_A
    seq = []
    for i in torch.randperm(P, generator=gen).tolist():
        b1 = [CTX1, key_tok(i), val_s1]
        b2 = [CTX2, key_tok(i), val_s2]
        if int(torch.randint(2, (1,), generator=gen).item()) == 0:
            seq += b1 + b2
        else:
            seq += b2 + b1                      # RANDOMIZED ORDER
    q_stream = int(torch.randint(2, (1,), generator=gen).item())
    q_key = int(torch.randint(P, (1,), generator=gen).item())
    seq += [CTX1 if q_stream == 0 else CTX2, key_tok(q_key),
            val_s1 if q_stream == 0 else val_s2]
    assert len(seq) == BLOCK
    return seq, q_stream


def make_batch(B, gen):
    seqs, qs = zip(*[make_seq(gen) for _ in range(B)])
    return torch.tensor(seqs, dtype=torch.long), torch.tensor(qs, dtype=torch.long)


def make_seq_explicit(perm, orders, flip, q_stream, q_key):
    """Every choice pinned, for the flip-invariance and recency probes."""
    val_s1 = VAL_A if flip == 0 else VAL_B
    val_s2 = VAL_B if flip == 0 else VAL_A
    seq = []
    for i, o in zip(perm, orders):
        b1 = [CTX1, key_tok(i), val_s1]
        b2 = [CTX2, key_tok(i), val_s2]
        seq += (b1 + b2) if o == 0 else (b2 + b1)
    seq += [CTX1 if q_stream == 0 else CTX2, key_tok(q_key),
            val_s1 if q_stream == 0 else val_s2]
    return torch.tensor(seq, dtype=torch.long)


def role_masks(tokens):
    """
    Separation only MATTERS on the KEY and VAL tokens: those are the ones whose identity
    is shared across streams, so routing is the only thing that can tell them apart. The
    CTX tokens already mark their own stream by identity and need not separate, so
    averaging the gate over all roles at once understates separation badly.
    """
    ctx = (tokens == CTX1) | (tokens == CTX2)
    val = (tokens == VAL_A) | (tokens == VAL_B)
    key = torch.zeros_like(ctx)
    for i in range(P):
        key |= (tokens == key_tok(i))
    return {'ctx': ctx, 'key': key, 'val': val}


def stream_labels(tokens):
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


# ── Instrument ────────────────────────────────────────────────────────────────
def perfect_gate_batched(tokens, dtype=torch.float32):
    """
    Vectorized _perfect_gate: latch onto the most recent CTX token, [0.5, 0.5] before
    any has been seen. Verified elementwise against the imported _perfect_gate (CHECK 0).
    """
    B, T = tokens.shape
    idx = torch.arange(T).expand(B, T)
    is_ctx = (tokens == CTX1) | (tokens == CTX2)
    last = torch.where(is_ctx, idx, torch.full_like(idx, -1)).cummax(dim=1).values
    ctx_tok = tokens.gather(1, last.clamp(min=0))
    seen = last >= 0
    g = torch.full((B, T, 2), 0.5, dtype=dtype)
    g[..., 0] = torch.where(seen, (ctx_tok == CTX1).to(dtype), torch.tensor(0.5, dtype=dtype))
    g[..., 1] = torch.where(seen, (ctx_tok == CTX2).to(dtype), torch.tensor(0.5, dtype=dtype))
    return g


def decay_mask(T, decay, dtype=torch.float32):
    """M[t,s] = decay^(t-s) for s<t, else 0 — the parallel form of U = decay*I."""
    idx = torch.arange(T)
    d = (idx.view(-1, 1) - idx.view(1, -1)).clamp(min=0).to(dtype)
    return (torch.as_tensor(decay, dtype=dtype) ** d).tril(diagonal=-1)


def gate_recurrent(v, W_in, W_h, W_g):
    """bdh_multichannel's gate: h_t = tanh(W_in v_t + W_h h_{t-1}); g_t = softmax(W_g h_t)."""
    B, T, _ = v.shape
    h = torch.zeros(B, W_h.shape[0], dtype=v.dtype)
    gs = []
    for t in range(T):
        h = torch.tanh(v[:, t] @ W_in.T + h @ W_h.T)
        gs.append(F.softmax(h @ W_g.T, dim=-1))
    return torch.stack(gs, dim=1)


def mc_parallel(v, Dx, Dy, E, gr, gw, decay):
    """scores[t,s] = decay^(t-s) * (x_t . x_s) * (g_read[t] . g_write[s])."""
    x = F.relu(ln(v) @ Dx)
    M = decay_mask(v.shape[1], decay, v.dtype).to(v.device)
    scores = torch.einsum('btn,bsn->bts', x, x) * torch.einsum('btk,bsk->bts', gr, gw) * M
    a = scores @ v
    y = F.relu(ln(a) @ Dy) * x
    return v + ln(y @ E)


def mc_recurrent_bounded(v, Dx, Dy, E, gr, gw, decay, C, track_sat=False):
    """Same system token by token, with S^(c) projected onto a Frobenius ball of radius C."""
    B, T, Dd = v.shape
    Nn, kk = Dx.shape[1], gw.shape[-1]
    x = F.relu(ln(v) @ Dx)
    S = torch.zeros(B, kk, Nn, Dd, dtype=v.dtype)
    sat = torch.zeros(kk, dtype=v.dtype)
    a_list = []
    for t in range(T):
        reads = torch.einsum('bn,bknd->bkd', x[:, t], S)
        a_list.append(torch.einsum('bk,bkd->bd', gr[:, t], reads))
        outer = torch.einsum('bn,bd->bnd', x[:, t], v[:, t])
        S = decay * (S + gw[:, t].view(B, kk, 1, 1) * outer.unsqueeze(1))
        if C is not None:
            nrm = S.reshape(B, kk, -1).norm(dim=-1)
            sc = (C / (nrm + 1e-12)).clamp(max=1.0)
            if track_sat:
                sat = sat + (sc < 1.0).to(v.dtype).sum(0)
            S = S * sc.view(B, kk, 1, 1)
    a = torch.stack(a_list, dim=1)
    y = F.relu(ln(a) @ Dy) * x
    out = v + ln(y @ E)
    return (out, sat / (B * T)) if track_sat else (out, None)


def gumbel(logits, tau, gen=None):
    u = torch.rand(logits.shape, generator=gen, dtype=logits.dtype)
    return F.softmax((logits + (-torch.log(-torch.log(u + 1e-20) + 1e-20))) / tau, dim=-1)


class Instrument(nn.Module):
    """
    gate: 'recurrent' | 'token' | 'uniform' | 'perfect' | 'none'(k=1)
    mode: 'sym' (one gate for read and write) | 'asym' (Gumbel hard write, soft read)
    """

    def __init__(self, gate='recurrent', mode='sym', n_ch=k, decay=1.0, C=None,
                 n_feat=None, gate_to_readout=False):
        """
        n_feat: width of the sparse feature dimension (Dx/Dy/E), i.e. the per-channel
            fast-weight state is n_feat x D. Defaults to the module's N, so the default
            path draws exactly the same random numbers in the same order as before.
        gate_to_readout: add a learned projection of the recurrent gate's hidden state
            into the residual before the head. Its parameter is created AFTER all
            pre-existing ones and only when the flag is set, so the default path's RNG
            stream is untouched. Both knobs exist for test_capacity_control.py's
            resource-matched single-channel controls.
        """
        super().__init__()
        self.gate_kind, self.mode, self.n_ch, self.decay, self.C = gate, mode, n_ch, decay, C
        nf = N if n_feat is None else int(n_feat)
        self.n_feat, self.gate_to_readout = nf, bool(gate_to_readout)
        self.embed = nn.Embedding(V, D)
        self.Dx = nn.Parameter(torch.randn(D, nf) * 0.1)
        self.Dy = nn.Parameter(torch.randn(D, nf) * 0.1)
        self.E  = nn.Parameter(torch.randn(nf, D) * 0.1)
        self.head = nn.Linear(D, V, bias=False)
        if gate == 'recurrent':
            self.W_in = nn.Parameter(torch.randn(H_GATE, D) * 0.1)
            self.W_h  = nn.Parameter(torch.randn(H_GATE, H_GATE) * 0.1)
            self.W_g  = nn.Parameter(torch.randn(n_ch, H_GATE) * 0.1)
        elif gate == 'token':
            self.W_tok = nn.Parameter(torch.randn(D, n_ch) * 0.1)
        if self.gate_to_readout:
            assert gate == 'recurrent', "gate_to_readout needs the recurrent gate"
            self.W_ro = nn.Parameter(torch.randn(D, H_GATE) * 0.1)

    def gates(self, tokens, v, tau=None):
        if self.gate_kind == 'none':
            g = torch.ones(v.shape[0], v.shape[1], 1, dtype=v.dtype)
            return g, g
        if self.gate_kind == 'uniform':
            g = torch.full((v.shape[0], v.shape[1], self.n_ch), 1.0 / self.n_ch, dtype=v.dtype)
            return g, g
        if self.gate_kind == 'perfect':
            g = perfect_gate_batched(tokens, v.dtype)
            return g, g
        if self.gate_kind == 'recurrent':
            B, T, _ = v.shape
            h = torch.zeros(B, H_GATE, dtype=v.dtype)
            logits, hs = [], []
            for t in range(T):
                h = torch.tanh(v[:, t] @ self.W_in.T + h @ self.W_h.T)
                hs.append(h)
                logits.append(h @ self.W_g.T)
            lg = torch.stack(logits, dim=1)
            if self.gate_to_readout:
                self._h_seq = torch.stack(hs, dim=1)          # (B, T, H_GATE)
        else:
            lg = v @ self.W_tok
        gr = F.softmax(lg, dim=-1)
        gw = gumbel(lg, tau if tau is not None else TAU_END) if self.mode == 'asym' else gr
        return gr, gw

    def forward(self, tokens, tau=None, track_sat=False):
        v = self.embed(tokens)
        gr, gw = self.gates(tokens, v, tau)
        if self.C is None:
            out = mc_parallel(v, self.Dx, self.Dy, self.E, gr, gw, self.decay)
            sat = None
        else:
            out, sat = mc_recurrent_bounded(v, self.Dx, self.Dy, self.E, gr, gw,
                                            self.decay, self.C, track_sat)
        if self.gate_to_readout:
            out = out + self._h_seq @ self.W_ro.T
        return self.head(out), sat, gr, gw


# ── Training / evaluation ────────────────────────────────────────────────────
def run(gate, mode, n_ch, decay, C, seed, anneal=False, ckpts=None,
        n_feat=None, gate_to_readout=False):
    """`ckpts`: explicit checkpoint steps. Default keeps the original N_CKPT schedule;
    callers wanting finer resolution on when separation emerges pass their own.
    `n_feat` / `gate_to_readout`: passed straight to Instrument; both default to the
    original behaviour (see Instrument.__init__)."""
    torch.manual_seed(seed)
    model = Instrument(gate, mode, n_ch, decay, C,
                       n_feat=n_feat, gate_to_readout=gate_to_readout)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = torch.Generator(); rng.manual_seed(seed + 10_000)
    prng = torch.Generator(); prng.manual_seed(seed + 99_000)
    ptok, _ = make_batch(PROBE_B, prng)
    pinp, plab = ptok[:, :-1], stream_labels(ptok[:, :-1])
    if ckpts is None:
        ckpts = sorted({int(i * (ITERS - 1) / (N_CKPT - 1)) for i in range(N_CKPT)})
    else:
        ckpts = sorted(set(ckpts))
    trace = []

    prole = role_masks(pinp)

    def diag(step, tau):
        with torch.no_grad():
            _, _, gr, gw = model(pinp, tau)
            d = {'step': step}
            for nm, g in (('r', gr), ('w', gw)):
                g0, g1 = g[plab == 0].mean(0), g[plab == 1].mean(0)
                d[f'{nm}_s1'], d[f'{nm}_s2'] = g0.tolist(), g1.tolist()
                d[f'{nm}_cos'] = F.cosine_similarity(g0.unsqueeze(0), g1.unsqueeze(0)).item()
            # per-role cosines; 'val' and 'key' are the ones that carry the separation
            for rn, rm in prole.items():
                a = gr[(plab == 0) & rm].mean(0)
                b = gr[(plab == 1) & rm].mean(0)
                d[f'{rn}_cos'] = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
                d[f'{rn}_s1'], d[f'{rn}_s2'] = a.tolist(), b.tolist()
            return d

    for step in range(ITERS):
        tau = (TAU_START * (TAU_END / TAU_START) ** (step / (ITERS - 1))) if anneal else TAU_END
        if step in ckpts and n_ch > 1:
            trace.append(diag(step, tau))
        tokens, _ = make_batch(BATCH, rng)
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        logits, _, _, _ = model(inp, tau)
        loss = F.cross_entropy(logits[:, -1, :], tgt[:, -1])   # query position only
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    if n_ch > 1:
        trace.append(diag(ITERS, TAU_END))
    with torch.no_grad():
        tokens, qs = make_batch(512, rng)
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        logits, _, _, _ = model(inp, TAU_END)
        ql, qt = logits[:, -1, :], tgt[:, -1]
        corr = (ql.argmax(-1) == qt)
        m1, m2 = qs == 0, qs == 1
        out = dict(l1=F.cross_entropy(ql[m1], qt[m1]).item(),
                   l2=F.cross_entropy(ql[m2], qt[m2]).item(),
                   a1=corr[m1].float().mean().item(),
                   a2=corr[m2].float().mean().item(), trace=trace, sat=[0.0] * n_ch)
        if C is not None:
            _, sat, _, _ = model(pinp[:64], TAU_END, track_sat=True)
            out['sat'] = sat.tolist()
    return out


def ms(xs):
    m = sum(xs) / len(xs)
    return m, (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def agg(runs):
    o = {}
    for key in ('l1', 'l2', 'a1', 'a2'):
        o[key], o[key + '_sd'] = ms([r[key] for r in runs])
    o['acc'] = (o['a1'] + o['a2']) / 2
    o['traces'] = [r['trace'] for r in runs]
    nc = len(runs[0]['sat'])
    o['sat'] = [sum(r['sat'][c] for r in runs) / len(runs) for c in range(nc)]
    return o


def fmt(xs):
    return "[" + ", ".join(f"{x:.3f}" for x in xs) + "]"


def final(o, key):
    fin = [t[-1] for t in o['traces'] if t]
    return (sum(p[key] for p in fin) / len(fin)) if fin else float('nan')


# Gate-vector keys in a trace point. Everything else (cosines, step) is a scalar and is
# permutation-invariant, so canonicalization below cannot touch it.
VEC_KEYS = ('r_s1', 'r_s2', 'w_s1', 'w_s2',
            'ctx_s1', 'ctx_s2', 'key_s1', 'key_s2', 'val_s1', 'val_s2')


def canonicalize(traces, ref='val_s1'):
    """
    DISPLAY ONLY. Which physical channel a seed puts stream-1 into is arbitrary — it is
    fixed by that run's initialization. Averaging raw gate vectors across seeds therefore
    mixes incompatible labelings: three seeds splitting 2:1 average to exactly
    [0.667, 0.333] no matter how cleanly each individual seed separated, which reads as
    weak routing when the routing is in fact near-one-hot.

    Fix: per seed, pick the channel permutation that puts stream-1's majority VAL mass in
    channel 0, decided at the FINAL checkpoint so one labeling holds for the whole run,
    and apply it to every gate vector of that seed. Accuracy, loss and every cosine are
    permutation-invariant and are not recomputed or touched here.
    """
    out = []
    for tr in traces:
        if not tr:
            out.append(tr)
            continue
        v = tr[-1].get(ref)
        perm = [0, 1] if (not v or len(v) != 2 or v[0] >= v[1]) else [1, 0]
        if perm == [0, 1]:
            out.append(tr)
            continue
        out.append([{kk: ([vv[i] for i in perm]
                          if kk in VEC_KEYS and isinstance(vv, list) else vv)
                     for kk, vv in p.items()} for p in tr])
    return out


def finalv(o, key):
    fin = [t[-1] for t in canonicalize(o['traces']) if t]
    return ([sum(p[key][c] for p in fin) / len(fin) for c in range(k)] if fin
            else [float('nan')] * k)


def print_trace(label, traces):
    traces = [t for t in traces if t]
    if not traces:
        return
    traces = canonicalize(traces)     # display only; see canonicalize() docstring
    print(f"  gate trace ({label}, mean over {len(traces)} seeds). Separation lives on the")
    print(f"  KEY/VAL tokens — CTX tokens mark their own stream by identity and need not split.")
    print(f"  Channel labels are canonicalized per seed before averaging, so the vectors show")
    print(f"  how sharply each seed routes rather than how the seeds happened to label channels:")
    print(f"    {'step':>5} | {'all cos':>8} {'ctx cos':>8} {'KEY cos':>8} {'VAL cos':>8} | "
          f"{'VAL s1':>15} {'VAL s2':>15}")
    for i in range(len(traces[0])):
        pts = [t[i] for t in traces]
        av = lambda key: sum(p[key] for p in pts) / len(pts)
        avv = lambda key: [sum(p[key][c] for p in pts) / len(pts) for c in range(k)]
        print(f"    {pts[0]['step']:>5} | {av('r_cos'):>8.4f} {av('ctx_cos'):>8.4f} "
              f"{av('key_cos'):>8.4f} {av('val_cos'):>8.4f} | "
              f"{fmt(avv('val_s1')):>15} {fmt(avv('val_s2')):>15}")
    print()


# ── Verification ──────────────────────────────────────────────────────────────
def verify():
    print("=" * 96)
    print("VERIFICATION — the instrument, before any experiment")
    print("=" * 96)
    g = torch.Generator().manual_seed(4242)
    T, dt = BLOCK - 1, torch.float64
    v  = torch.randn(T, D, generator=g, dtype=dt)
    vb = v.unsqueeze(0)
    Dx = torch.randn(D, N, generator=g, dtype=dt) * 0.1
    Dy = torch.randn(D, N, generator=g, dtype=dt) * 0.1
    E  = torch.randn(N, D, generator=g, dtype=dt) * 0.1
    Wg = torch.randn(D, k, generator=g, dtype=dt) * 0.5
    ok = True

    # CHECK 0 — the vectorized perfect gate equals the imported _perfect_gate
    gen0 = torch.Generator().manual_seed(5)
    ctok, _ = make_batch(64, gen0)
    cinp = ctok[:, :-1]
    ref0 = torch.stack([_perfect_gate(cinp[b]) for b in range(cinp.shape[0])])
    d0 = (perfect_gate_batched(cinp) - ref0).abs().max().item()
    print(f"CHECK 0  vectorized perfect gate vs imported _perfect_gate: {d0:.2e}")
    ok &= d0 == 0.0
    print(f"         -> {'IDENTICAL' if d0 == 0.0 else 'FAILED'} "
          f"(used for the ceiling; vectorized only for speed)\n")

    # CHECK 1 — decay=1 symmetric token gate == imported bdh_layer_mc_parallel
    gg = F.softmax(v @ Wg, dim=-1).unsqueeze(0)
    ref, *_ = bdh_layer_mc_parallel(v, Dx, Dy, E, Wg)
    d1 = (mc_parallel(vb, Dx, Dy, E, gg, gg, 1.0)[0] - ref).abs().max().item()
    print(f"CHECK 1  decay=1, symmetric gate vs imported bdh_layer_mc_parallel : {d1:.2e}")
    ok &= d1 < 1e-12
    print(f"         -> {'EXACT' if d1 < 1e-12 else 'FAILED'} (U=I recovers experiment 2)\n")

    # CHECK 2 — U != I: k=1 with decay == imported bdh_layer_recurrent(decay)
    print("CHECK 2  U != I anchored on the imported reference: k=1 with a decay^(t-s)")
    print("         mask vs bdh_layer_recurrent(decay=d), which carries the paper's")
    print("         decay slot S = decay*(S + outer):")
    g1 = torch.ones(1, T, 1, dtype=dt)
    worst2 = 0.0
    for d in (1.0, 0.95, 0.85, 0.5):
        vr, *_ = bdh_layer_recurrent(v, Dx, Dy, E, decay=d)
        e = (mc_parallel(vb, Dx, Dy, E, g1, g1, d)[0] - vr).abs().max().item()
        worst2 = max(worst2, e)
        print(f"         decay={d:<5} max |parallel - imported recurrent| = {e:.2e}")
    ok &= worst2 < 1e-12
    print(f"         -> {'EXACT at every decay' if worst2 < 1e-12 else 'FAILED'}\n")

    # CHECK 3 — parallel == bounded-capacity recurrent at C=None, any decay
    gr = F.softmax(v @ Wg, dim=-1).unsqueeze(0)
    gw = F.softmax(v @ torch.randn(D, k, generator=g, dtype=dt), dim=-1).unsqueeze(0)
    worst3 = 0.0
    for d in (1.0, 0.85):
        a = mc_parallel(vb, Dx, Dy, E, gr, gw, d)
        b, _ = mc_recurrent_bounded(vb, Dx, Dy, E, gr, gw, d, None)
        worst3 = max(worst3, (a - b).abs().max().item())
    print(f"CHECK 3  split-gate parallel vs recurrent (C=None): {worst3:.2e}")
    ok &= worst3 < 1e-12
    print(f"         -> {'IDENTICAL' if worst3 < 1e-12 else 'FAILED'}\n")

    # CHECK 4 — k=1 reduces to single-channel BDH
    d4 = (mc_parallel(vb, Dx, Dy, E, g1, g1, 1.0)[0]
          - bdh_layer_parallel(v, Dx, Dy, E)[0]).abs().max().item()
    print(f"CHECK 4  k=1, decay=1 vs imported bdh_layer_parallel: {d4:.2e}")
    ok &= d4 < 1e-12
    print(f"         -> {'EXACT' if d4 < 1e-12 else 'FAILED'}\n")

    # CHECK 5 — THE CRITICAL ONE: can the recurrent gate express the perfect gate?
    print("CHECK 5  IS THE REPRESENTATIONAL CAP LIFTED? Experiment 7 showed a")
    print("         token-identity gate CANNOT express the perfect latching gate (it")
    print("         must give the same key token [1,0] in one block and [0,1] in the")
    print("         other). Fit each gate to the perfect gate's targets and compare:")
    gen = torch.Generator().manual_seed(11)
    ftok, _ = make_batch(64, gen)
    finp = ftok[:, :-1]
    tgt = torch.stack([_perfect_gate(finp[b]) for b in range(finp.shape[0])])
    report = {}
    for kind in ('token', 'recurrent'):
        torch.manual_seed(3)
        m = Instrument(kind, 'sym', k, 1.0, None)
        ps = ([m.W_in, m.W_h, m.W_g] if kind == 'recurrent' else [m.W_tok]) + [m.embed.weight]
        opt = torch.optim.Adam(ps, lr=1e-2)
        for _ in range(GATE_FIT_ITERS):
            gr, _ = m.gates(finp, m.embed(finp))
            loss = F.mse_loss(gr, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            gr, _ = m.gates(finp, m.embed(finp))
            mse = F.mse_loss(gr, tgt).item()
            acc = (gr.argmax(-1) == tgt.argmax(-1)).float().mean().item()
        report[kind] = (mse, acc)
        print(f"         {kind:>10} gate: best-fit MSE to perfect gate = {mse:.4f}   "
              f"argmax match = {acc:.3f}")
    ok &= report['recurrent'][1] > 0.95 and report['recurrent'][1] > report['token'][1] + 0.1
    print(f"         -> {'CAP LIFTED' if ok else 'CAP NOT LIFTED — instrument still broken'}"
          f": the recurrent-state gate reaches {report['recurrent'][1]:.1%} argmax match")
    print(f"            against the token gate's {report['token'][1]:.1%}. The perfect gate is")
    print("            now INSIDE the model's own gate function class.\n")

    # CHECK 6 — residual position cue: does randomized order remove the recency signal?
    print("CHECK 6  FLOOR INTEGRITY vs the old task. Exp 7 proved the query output was")
    print("         flip-invariant (8.9e-16) so the floor was exactly 50%. With U != I")
    print("         position matters, so that proof lapses. Measuring what replaces it:")
    torch.manual_seed(0)
    m = Instrument('token', 'sym', k, 1.0, None).double()
    for d in DECAY_SWEEP:
        m.decay = d
        diffs, agree = [], 0
        with torch.no_grad():
            for trial in range(64):
                gg2 = torch.Generator().manual_seed(900 + trial)
                perm = torch.randperm(P, generator=gg2).tolist()
                orders = torch.randint(2, (P,), generator=gg2).tolist()
                qs = int(torch.randint(2, (1,), generator=gg2).item())
                qk = int(torch.randint(P, (1,), generator=gg2).item())
                s0 = make_seq_explicit(perm, orders, 0, qs, qk)
                s1 = make_seq_explicit(perm, orders, 1, qs, qk)
                lg, *_ = m(torch.stack([s0[:-1], s1[:-1]]))
                diffs.append((lg[0, -1] - lg[1, -1]).abs().max().item())
                # does an untrained model's prediction track the flip in the right direction?
                agree += int(lg[0, -1].argmax().item() == s0[-1].item())
        print(f"         decay={d:<5} mean |logits(flip=0) - logits(flip=1)| = "
              f"{sum(diffs)/len(diffs):.3e}   (0 => flip-invariant => exactly 50%)")
    print("         -> a nonzero gap means position now carries SOME flip information, so")
    print("            the floor is no longer 50% by proof and must be MEASURED. That is")
    print("            what FLOOR RE-DERIVATION below does, per decay.\n")

    print(f"{'ALL VERIFICATION CHECKS PASSED' if ok else 'SOME VERIFICATION CHECKS FAILED'}\n")
    assert ok, "verification failed — do not trust the results below"


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 96)
    print("Instrument v2: U != I, randomized order, recurrent-state gate")
    print(f"  P={P} k={k} D={D} N={N} BLOCK={BLOCK} BATCH={BATCH} ITERS={ITERS} "
          f"SEEDS={SEEDS}")
    print(f"  DECAY_SWEEP={DECAY_SWEEP}  H_GATE={H_GATE}  C_SWEEP={C_SWEEP}")
    print("=" * 96)
    print()
    verify()
    t0 = time.time()

    # ── FLOOR RE-DERIVATION ───────────────────────────────────────────────────
    print("=" * 96)
    print("FLOOR RE-DERIVATION (measured per decay, not assumed)")
    print("=" * 96)
    floors = {}
    for d in DECAY_SWEEP:
        single = agg([run('none', 'sym', 1, d, None, s) for s in SEEDS])
        unif   = agg([run('uniform', 'sym', k, d, None, s) for s in SEEDS])
        perf   = agg([run('perfect', 'sym', k, d, None, s) for s in SEEDS])
        floors[d] = dict(single=single, unif=unif, perf=perf)
        print(f"  decay={d}:  single-channel={single['acc']:.3f}  "
              f"uniform-gate={unif['acc']:.3f}  perfect-gate={perf['acc']:.3f}  "
              f"span={perf['acc'] - unif['acc']:+.3f}")
    print()
    live = [d for d in DECAY_SWEEP
            if floors[d]['perf']['acc'] - floors[d]['unif']['acc'] > 0.30
            and floors[d]['unif']['acc'] < 0.65]
    print(f"  decays with an intact floor and a live span: {live}")
    if not live:
        print("  NO DECAY LEAVES A USABLE INSTRUMENT — the experiments below are vacuous.")
        sys.exit(0)
    if OPERATING_DECAY in live:
        DECAY = OPERATING_DECAY
    else:
        DECAY = live[0] if 1.0 not in live else (live[1] if len(live) > 1 else live[0])
        print(f"  NOTE: module constant OPERATING_DECAY={OPERATING_DECAY} is NOT among the"
              f" decays with an intact floor; using {DECAY} instead.")
    FLOOR = floors[DECAY]['unif']['acc']
    CEIL  = floors[DECAY]['perf']['acc']
    print(f"  operating point: decay={DECAY}  floor={FLOOR:.3f}  ceiling={CEIL:.3f}")
    print(f"  (U != I is the point of the rebuild, so the first non-unity decay with an")
    print(f"   intact floor is used; decay=1.0 is reported above for comparison.)")
    print()
    lift = lambda o: (o['acc'] - FLOOR) / (CEIL - FLOOR) if CEIL - FLOOR > 1e-3 else 0.0

    # ── EXPERIMENT 2 RE-RUN ───────────────────────────────────────────────────
    print("=" * 96)
    print("EXPERIMENT 2 RE-RUN — MC k=2, symmetric learned gate. Does it separate now?")
    print("=" * 96)
    e2_rec = agg([run('recurrent', 'sym', k, DECAY, None, s) for s in SEEDS])
    e2_tok = agg([run('token', 'sym', k, DECAY, None, s) for s in SEEDS])
    for nm, o in (('recurrent-state gate', e2_rec), ('token-identity gate (old)', e2_tok)):
        print(f"  {nm}: acc={o['acc']:.3f}  lift={lift(o):+.2f}  "
              f"gate cos={final(o,'r_cos'):.4f}  "
              f"s1={fmt(finalv(o,'r_s1'))} s2={fmt(finalv(o,'r_s2'))}")
        print(f"      stream-1 loss={o['l1']:.4f}+-{o['l1_sd']:.4f} acc={o['a1']:.3f}+-{o['a1_sd']:.3f}"
              f"   stream-2 loss={o['l2']:.4f}+-{o['l2_sd']:.4f} acc={o['a2']:.3f}+-{o['a2_sd']:.3f}")
    print()
    print_trace("Exp 2, recurrent gate", e2_rec['traces'])

    # ── EXPERIMENT 5 RE-RUN ───────────────────────────────────────────────────
    print("=" * 96)
    print("EXPERIMENT 5 RE-RUN — arm 1, and asymmetric Gumbel hard write / soft read")
    print("=" * 96)
    gg = torch.Generator().manual_seed(7)
    T = BLOCK - 1
    vv  = torch.randn(1, T, D, generator=gg, dtype=torch.float64)
    Dx_ = torch.randn(D, N, generator=gg, dtype=torch.float64) * 0.1
    Dy_ = torch.randn(D, N, generator=gg, dtype=torch.float64) * 0.1
    E_  = torch.randn(N, D, generator=gg, dtype=torch.float64) * 0.1
    gr_u = torch.full((1, T, k), 1.0 / k, dtype=torch.float64)
    outs = []
    for _ in range(4):
        gw_ = F.softmax(torch.randn(1, T, k, generator=gg, dtype=torch.float64) * 40.0, -1)
        outs.append(mc_parallel(vv, Dx_, Dy_, E_, gr_u, gw_, DECAY))
    arm1 = max((outs[i] - outs[0]).abs().max().item() for i in range(1, 4))
    print(f"  ARM 1 under U != I (decay={DECAY}): 4 near-one-hot write routings, uniform")
    print(f"  read -> max output divergence = {arm1:.2e}")
    print(f"  -> {'STILL EXACTLY CANCELS — arm 1 survives U != I' if arm1 < 1e-12 else 'cancellation is NOT exact here'}")
    print()
    e5 = agg([run('recurrent', 'asym', k, DECAY, None, s, anneal=True) for s in SEEDS])
    print(f"  asymmetric (Gumbel hard write + soft read, recurrent gate): "
          f"acc={e5['acc']:.3f}  lift={lift(e5):+.2f}")
    print(f"      read cos={final(e5,'r_cos'):.4f}  write cos={final(e5,'w_cos'):.4f}  "
          f"write s1={fmt(finalv(e5,'w_s1'))} s2={fmt(finalv(e5,'w_s2'))}")
    print()

    # ── EXPERIMENT 6 RE-RUN ───────────────────────────────────────────────────
    print("=" * 96)
    print("EXPERIMENT 6 RE-RUN — bounded per-channel capacity")
    print("=" * 96)
    e6 = {}
    for C in C_SWEEP:
        o = agg([run('recurrent', 'sym', k, DECAY, C, s) for s in SEEDS])
        fl = agg([run('uniform', 'sym', k, DECAY, C, s) for s in SEEDS])
        ce = agg([run('perfect', 'sym', k, DECAY, C, s) for s in SEEDS])
        span = ce['acc'] - fl['acc']
        lf = (o['acc'] - fl['acc']) / span if span > 1e-3 else float('nan')
        e6[C] = dict(o=o, fl=fl, ce=ce, lift=lf, span=span)
        print(f"  C={str(C):>6}: floor={fl['acc']:.3f} ceil={ce['acc']:.3f} span={span:+.3f}"
              f"  test={o['acc']:.3f} lift={lf:+.2f}  sat={fmt(o['sat'])}  "
              f"gate cos={final(o,'r_cos'):.4f}")
    print()
    print(f"(total {time.time() - t0:.0f}s)")
    print()

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("=" * 96)
    print("VERDICT")
    print("=" * 96)
    l2r, l2t, l5 = lift(e2_rec), lift(e2_tok), lift(e5)
    # separation is judged on the KEY/VAL tokens, the ones whose identity is shared
    # across streams; the all-role average mixes in CTX tokens that need not split.
    sep2 = final(e2_rec, 'val_cos') < SEP_COS
    print(f"  instrument: decay={DECAY}, randomized order, recurrent-state gate")
    print(f"  floor={FLOOR:.3f}  ceiling={CEIL:.3f}")
    print(f"  Exp 2 (recurrent gate) : acc={e2_rec['acc']:.3f} lift={l2r:+.2f}  "
          f"VAL cos={final(e2_rec,'val_cos'):.4f} KEY cos={final(e2_rec,'key_cos'):.4f} "
          f"(all-role {final(e2_rec,'r_cos'):.4f})  "
          f"{'SEPARATED' if sep2 else 'not separated'}")
    print(f"  Exp 2 (token gate)     : acc={e2_tok['acc']:.3f} lift={l2t:+.2f}  "
          f"VAL cos={final(e2_tok,'val_cos'):.4f} KEY cos={final(e2_tok,'key_cos'):.4f}")
    print(f"  Exp 5 (asymmetric)     : acc={e5['acc']:.3f} lift={l5:+.2f}  "
          f"VAL cos={final(e5,'val_cos'):.4f}")
    best6 = max((v['lift'] for v in e6.values() if v['lift'] == v['lift']), default=float('nan'))
    print(f"  Exp 6 (best bounded C) : lift={best6:+.2f}")
    print()
    if l2r > LIFT_THRESH and sep2:
        print("THE INSTRUMENT WAS THE PROBLEM. With a gate that can latch, the plain")
        print("symmetric soft gate separates the streams ON ITS OWN and reaches the")
        print("ceiling, with no asymmetric routing and no capacity bound. Experiments 2-6")
        print("were measuring an unreachable target, not a real deadlock.")
        print()
        print("Arm 1 still holds as a mathematical fact — the write-routing cancellation")
        print(f"under an EXACTLY uniform read is still exact here ({arm1:.2e}, Exp-5 block")
        print("above), and U != I does not disturb it. But it never was a barrier. It is a")
        print("measure-zero stationary point: the gate is only near-uniform at init, never")
        print("exactly uniform, and once the solution is inside the function class the")
        print("resulting gradient is enough to leave it. The two-arm deadlock described")
        print("the gradients correctly and misdiagnosed their significance.")
    elif l2r > l2t + 0.15:
        print("PARTIAL. The recurrent-state gate does better than the token gate it")
        print("replaces, so the representational cap was real and binding, but it does not")
        print("reach the ceiling: some of the original obstacle survives the rebuild.")
    else:
        print("THE REBUILD DOES NOT RESCUE IT. Even with the perfect gate inside the")
        print("model's function class (CHECK 5), a live floor (measured above) and U != I,")
        print("the learned gate does not separate. The deadlock is NOT an artifact of the")
        print("old instrument's representational cap — it survives on an instrument built")
        print("specifically to remove that cap.")
