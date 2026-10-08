#!/usr/bin/env python
"""
explore_f_delta_fast.py — EXPLORATORY, not a result. Session F: a faster evaluation of explore_delta_mem's parallel (UT)
form for many channels with the FIXED decay (S54 (b), (c): k = 16, T = 50 / 99), with the same numbers. A subclass,
so explore_delta_mem.py (and every run made with it) is unchanged.

With a fixed decay the matrix M[t,s] = decay^(t-s) (x̂_t·x̂_s) (s < t) is the same for every channel; only the erase
strengths b^c differ. Then
    U^c = (I + diag(b^c) M)^{-1} R^c,   R^c = diag(w^c) V           (one unit-lower-triangular solve per channel)
    O   = Σ_c diag(g^c_r) M U^c  =  Σ_c g^c_r ⊙ [M · (U^1 | … | U^k)]_c   (ONE matmul with M against all channels at once)
and the solve's backward, for the gradient G of U,
    Ĝ^c        = (I + diag(b^c) M)^{-T} G^c                          (one solve)
    ∂/∂R^c     = Ĝ^c
    ∂/∂b^c_t   = −Σ_j Ĝ^c[t,j] (M U^c)[t,j]                          (reuses M·U from the forward)
    ∂/∂M[t,s]  = −Σ_c Σ_j b^c_t Ĝ^c[t,j] U^c[s,j]  = −[(b⊙Ĝ) as (T, k·d)] [U as (T, k·d)]ᵀ   (ONE matmul)
so no (B, k, T, T) gradient is materialised (torch's own backward forms one per channel). impl "fast" falls back to the
parallel form when the decay is per token (routed, gdn). Large batches (evaluation: 2048 sequences) are processed in
chunks of CHUNK sequences (the same numbers per sequence; bounded memory).

CHECKS (fast_checks; S54's children assert them): in float64 on random tensors (B 3, T 17, n 24, d 8, k 5, soft gates,
β fixed / per token, write tied / 1) the fast form equals delta_parallel to 1e-12 (forward) and its gradients with respect
to keys, values, gates, b, w to 1e-10; in float32 at the S=8, k=16 model size (B 32, T 99) to 1e-5 relative; a model
converted with impl "fast" gives the same logits as impl "parallel" to 1e-5 and the same gradients to 1e-4 relative;
batch chunking leaves each sequence's output unchanged to 1e-6.
"""

import torch

import explore_delta_mem as dm
from test_instrument_v2 import decay_mask

CHUNK = 256


class _DeltaRead(torch.autograd.Function):
    """O = Σ_c g_r^c ⊙ M U^c with U^c = (I + diag(b^c) M)^{-1} R^c, M (B,T,T) shared (strictly lower), b (B,k,T),
    R (B,k,T,d), gr (B,T,k). Backward (module docstring), with  G_U = Mᵀ (g_r ⊗ G_O),  Ĝ = (I + diag(b) M)^{-T} G_U
    (solved as Ĝᵀ (I + diag(b) M) = G_Uᵀ: no transposed copy of L), and one matmul for ∂M:
    ∂M = [(g_r ⊗ G_O) − b ⊙ Ĝ] (T, k·d) · U (T, k·d)ᵀ, strictly lower."""

    @staticmethod
    def forward(ctx, M, b, R, gr):
        B, k, T, d = R.shape
        # L = diag(b) M per channel, built as the transpose of a contiguous Lᵀ (column-major for LAPACK: no copy)
        L = (M.mT.contiguous().unsqueeze(1) * b.contiguous().unsqueeze(-2)).reshape(B * k, T, T).mT
        U = torch.linalg.solve_triangular(L, R.reshape(B * k, T, d), upper=False, unitriangular=True)
        Uf = U.reshape(B, k, T, d).transpose(1, 2).reshape(B, T, k * d)
        MU = (M @ Uf).reshape(B, T, k, d)
        ctx.save_for_backward(L, Uf, MU, M, b, gr)
        return (gr.unsqueeze(-1) * MU).sum(2)

    @staticmethod
    def backward(ctx, gO):
        L, Uf, MU, M, b, gr = ctx.saved_tensors
        B, T, k, d = MU.shape
        g_gr = (MU * gO.unsqueeze(2)).sum(-1)                                      # (B, T, k)
        gMUf = (gr.unsqueeze(-1) * gO.unsqueeze(2)).reshape(B, T, k * d)
        gUf = M.mT @ gMUf                                                          # (B, T, k*d)
        gUt = gUf.reshape(B, T, k, d).permute(0, 2, 3, 1).reshape(B * k, d, T)    # G_Uᵀ per channel
        GhT = torch.linalg.solve_triangular(L, gUt, upper=False, left=False, unitriangular=True)   # (B*k, d, T)
        Gh = GhT.reshape(B, k, d, T).transpose(-1, -2)                             # (B, k, T, d)
        gR = Gh
        gb = -(Gh.transpose(1, 2) * MU).sum(-1).transpose(1, 2)                    # (B, k, T)
        bGf = (b.unsqueeze(-1) * Gh).transpose(1, 2).reshape(B, T, k * d)
        gM = ((gMUf - bGf) @ Uf.mT).tril(-1)
        return gM, gb, gR, g_gr


class _Read(torch.autograd.Function):
    """The erase-off case (b = 0): O = Σ_c g_r^c ⊙ M R^c."""

    @staticmethod
    def forward(ctx, M, R, gr):
        B, k, T, d = R.shape
        Rf = R.transpose(1, 2).reshape(B, T, k * d)
        MR = (M @ Rf).reshape(B, T, k, d)
        ctx.save_for_backward(Rf, MR, M, gr)
        return (gr.unsqueeze(-1) * MR).sum(2)

    @staticmethod
    def backward(ctx, gO):
        Rf, MR, M, gr = ctx.saved_tensors
        B, T, k, d = MR.shape
        g_gr = (MR * gO.unsqueeze(2)).sum(-1)
        gMRf = (gr.unsqueeze(-1) * gO.unsqueeze(2)).reshape(B, T, k * d)
        gR = (M.mT @ gMRf).reshape(B, T, k, d).transpose(1, 2)
        gM = (gMRf @ Rf.mT).tril(-1)
        return gM, gR, g_gr


def delta_fast(kh, v, gr, b, w, decay, skip_solve=False):
    """explore_delta_mem.delta_parallel for the fixed decay. Returns o (B,T,d)."""
    T = kh.shape[1]
    M = (kh @ kh.mT) * decay_mask(T, decay, kh.dtype)                  # (B, T, T)
    R = w.transpose(1, 2).unsqueeze(-1) * v.unsqueeze(1)                # (B, k, T, d)
    if skip_solve:
        return _Read.apply(M, R, gr)
    return _DeltaRead.apply(M, b.transpose(1, 2), R, gr)


class FastDeltaMemory(dm.DeltaMemory):
    def forward(self, Q, K, V):
        if self.impl != "fast" or self.decay_mod is not None:
            return super().forward(Q, K, V)
        assert K is Q and Q.shape[1] == 1, "one head, Q = K"
        gr, gw = self._gates_now()
        q, v = Q[:, 0], V[:, 0]
        B = q.shape[0]
        if B <= CHUNK:
            return self._core(q, v, gr, gw).unsqueeze(1)
        outs = [self._core(q[i:i + CHUNK], v[i:i + CHUNK], gr[i:i + CHUNK], gw[i:i + CHUNK]) for i in range(0, B, CHUNK)]
        return torch.cat(outs, 0).unsqueeze(1)

    def _core(self, q, v, gr, gw):
        kh = self.keys(q)
        b, w = self.coefs(v, gw)
        return delta_fast(kh, v, gr, b, w, self.decay, skip_solve=self.erase_off())


def to_delta_fast(m, **kw):
    """explore_delta_mem.to_delta with impl "fast": the memory becomes a FastDeltaMemory (same parameters, same state)."""
    kw = dict(kw, impl="parallel")
    m = dm.to_delta(m, **kw)
    m.attn.__class__ = FastDeltaMemory
    m.attn.impl = "fast"
    m.delta_cfg = dict(m.delta_cfg, impl="fast")
    return m


def converter(**kw):
    def conv(m, seed):
        return to_delta_fast(m, seed=seed, **kw)
    return conv


# ── CHECKs ───────────────────────────────────────────────────────────────────
def fast_checks():
    rows = []
    g = torch.Generator().manual_seed(11)
    dt = torch.float64
    B, T, n, d, k = 3, 17, 24, 8, 5
    worst_f = worst_g = 0.0
    for beta in (1.0, 0.5, None):
        for write in ("tied", 1.0):
            kh = torch.nn.functional.normalize(torch.relu(torch.randn(B, T, n, generator=g, dtype=dt)) + 0.01, dim=-1)
            v = torch.randn(B, T, d, generator=g, dtype=dt)
            gr = torch.softmax(torch.randn(B, T, k, generator=g, dtype=dt), -1)
            gw = torch.softmax(torch.randn(B, T, k, generator=g, dtype=dt), -1)
            bt = torch.sigmoid(torch.randn(B, T, 1, generator=g, dtype=dt)) if beta is None else torch.full((B, T, 1), beta, dtype=dt)
            ins = [x.clone().requires_grad_(True) for x in (kh, v, gr, gw, bt)]

            def both(fn):
                kh_, v_, gr_, gw_, bt_ = ins
                b_ = gw_ * bt_
                w_ = b_ if write == "tied" else gw_ * write
                return fn(kh_, v_, gr_, b_, w_)
            of = both(lambda *a: delta_fast(*a, 0.95))
            op = both(lambda *a: dm.delta_parallel(*a[:5], None, 0.95)[0])
            tgt = torch.randn(B, T, d, generator=g, dtype=dt)
            gf = torch.autograd.grad((of * tgt).sum(), ins)
            gp = torch.autograd.grad((op * tgt).sum(), ins)
            worst_f = max(worst_f, float((of - op).detach().abs().max()))
            worst_g = max(worst_g, max(float((a - c).abs().max()) for a, c in zip(gf, gp)))
    rows.append((f"fast = parallel in float64 (beta 1 / 0.5 / per token, write tied / 1, k = {k}): forward {worst_f:.1e} <= "
                 f"1e-12, gradients {worst_g:.1e} <= 1e-10", worst_f <= 1e-12 and worst_g <= 1e-10))
    # float32 at the S=8, k=16 size
    B, T, n, d, k = 32, 99, 256, 32, 16
    kh = torch.nn.functional.normalize(torch.relu(torch.randn(B, T, n, generator=g)), dim=-1).requires_grad_(True)
    v = torch.randn(B, T, d, generator=g).requires_grad_(True)
    gr = torch.softmax(torch.randn(B, T, k, generator=g), -1).requires_grad_(True)
    bt = torch.sigmoid(torch.randn(B, T, 1, generator=g))
    b = (gr * bt).detach().requires_grad_(True)
    of = delta_fast(kh, v, gr, b, b, 0.95)
    op = dm.delta_parallel(kh, v, gr, b, b, None, 0.95)[0]
    rel = float((of - op).abs().max() / op.abs().max())
    tgt = torch.randn(B, T, d, generator=g)
    gf = torch.autograd.grad((of * tgt).sum(), (kh, v, gr, b))
    gp = torch.autograd.grad((op * tgt).sum(), (kh, v, gr, b))
    relg = max(float((a - c).abs().max() / c.abs().max()) for a, c in zip(gf, gp))
    rows.append((f"fast = parallel in float32 at B 32, T 99, k 16: forward relative {rel:.1e} <= 1e-5, gradients relative "
                 f"{relg:.1e} <= 1e-4", rel <= 1e-5 and relg <= 1e-4))
    return rows


def model_checks(make, task, seed=3):
    """make() -> a built model (window gate or perfect gate). Logits and gradients of impl fast vs parallel; chunking."""
    import copy
    rows = []
    m0 = make()
    mp = dm.to_delta(copy.deepcopy(m0), beta="learned")
    mf = to_delta_fast(copy.deepcopy(m0), beta="learned")
    x = task.make_batch(16, torch.Generator().manual_seed(seed))[0]
    inp, tgt = x[:, :-1], x[:, 1:]
    outs = []
    for m in (mp, mf):
        m.train()
        m.zero_grad()
        lg = m(inp)[0]
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1))
        loss.backward()
        outs.append((lg.detach(), {n: q.grad.clone() for n, q in m.named_parameters() if q.grad is not None}))
    (lp, gp), (lf, gf) = outs
    dl = float((lp - lf).abs().max())
    relg = max(float((gp[n] - gf[n]).abs().max() / (gp[n].abs().max() + 1e-12)) for n in gp)
    rows.append((f"model ({type(mf).__name__}, k {mf.n_ch}): impl fast vs parallel, logits max |diff| {dl:.1e} <= 1e-5, "
                 f"parameter gradients relative {relg:.1e} <= 1e-4 over {len(gp)} tensors; fast memory class "
                 f"{type(mf.attn).__name__}", dl <= 1e-5 and relg <= 1e-4 and set(gp) == set(gf)))
    x2 = task.make_batch(CHUNK + 40, torch.Generator().manual_seed(seed + 1))[0][:, :-1]
    mf.eval()
    with torch.no_grad():
        big = mf(x2)[0]
        small = torch.cat([mf(x2[i:i + 40])[0] for i in range(0, x2.shape[0], 40)], 0)
    dc = float((big - small).abs().max())
    rows.append((f"batch chunking ({x2.shape[0]} sequences, chunks of {CHUNK}) vs batches of 40: max |diff| {dc:.1e} <= 1e-5",
                 dc <= 1e-5))
    return rows
