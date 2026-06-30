# Explicit recurrent (Hebbian) BDH-GPU, derived from the full paper
# (arXiv 2509.26507, eqs (4),(5),(8),(16) and Appendix E).
#
# The public repo + Appendix E implement the PARALLEL form:
#     a* = (RoPE(Q) @ RoPE(K).mT).tril(-1) @ V
# which never materializes the state. That parallel attention is mathematically
# identical to a token-by-token Hebbian accumulation of a state matrix S (= rho):
#
#     read :  a*_t = x_t @ S            (S holds all past tau < t)
#     write:  S   += outer(x_t, v*_t)   (Hebbian: co-activation strengthens state)
#
# This file implements BOTH and asserts they produce identical outputs, so the
# recurrent/Hebbian version is VERIFIED against the paper's reference, not guessed.
#
# We run with U = I (RoPE/ALiBi = identity). The paper's equations are defined for
# general U; U=I is a valid instance and isolates the Hebbian identity cleanly.
# The slot where U (decay/rotation) folds into the recurrence is marked below.

import torch
import torch.nn.functional as F


def ln(z, eps=1e-5):
    # parameter-free LayerNorm over last dim (paper sec 3.1: uniform, non-parametric)
    return (z - z.mean(-1, keepdim=True)) / (z.std(-1, keepdim=True) + eps)


# ---------------------------------------------------------------------------
# PARALLEL form — faithful to Appendix E (single head, RoPE = identity).
# This is the reference we trust.
# ---------------------------------------------------------------------------
def bdh_layer_parallel(v_ast, Dx, Dy, E):
    # v_ast: (T, D)   one layer of computation, following eq (8) / Appendix E
    T, Dd = v_ast.shape
    x = F.relu(ln(v_ast) @ Dx)                 # (T, N)   x = (Dx LN(E y))^+   [v_ast plays role of LN(Ey)]
    # linear attention: a*_t = sum_{tau<t} (x_t . x_tau) v*_tau
    scores = (x @ x.T).tril(diagonal=-1)       # (T, T)  strictly-causal, U = I
    a_ast = scores @ v_ast                     # (T, D)
    y = F.relu(ln(a_ast) @ Dy) * x             # (T, N)   y = (Dy LN(a*))^+ ⊙ x
    v_next = v_ast + ln(y @ E)                 # (T, D)   residual; E lifts N->D
    return v_next, x, y, a_ast


# ---------------------------------------------------------------------------
# RECURRENT form — explicit Hebbian state S (= rho), token by token.
# Mathematically identical to the parallel form above (verified below).
# ---------------------------------------------------------------------------
def bdh_layer_recurrent(v_ast, Dx, Dy, E, decay=1.0):
    T, Dd = v_ast.shape
    N = Dx.shape[1]
    x = F.relu(ln(v_ast) @ Dx)                 # (T, N)  same x as parallel
    S = torch.zeros(N, Dd, dtype=v_ast.dtype)  # Hebbian state rho^T : (N, D), init 0 (paper: sigma_0 = 0)
    a_ast = torch.zeros(T, Dd, dtype=v_ast.dtype)
    for t in range(T):
        # READ  : a*_t = x_t @ S   (S currently holds only tau < t)
        a_ast[t] = x[t] @ S
        # WRITE : Hebbian outer-product update  (eq 6/7: sigma += y⊗x ; here rho += x⊗v*)
        #         decay slot: with ALiBi/RoPE, S = (S + outer)·U  -> here scalar `decay`
        S = decay * (S + torch.outer(x[t], v_ast[t]))
    y = F.relu(ln(a_ast) @ Dy) * x
    v_next = v_ast + ln(y @ E)
    return v_next, x, y, a_ast, S


# ---------------------------------------------------------------------------
# MULTI-CHANNEL form — k per-channel Hebbian states, context-gated blend.
#
# Gate:  g_t = softmax(v*_t @ W_g)  ∈ R^k          (per-token, k-dim)
# Write: S^(c) += g_t^(c) · outer(x_t, v*_t)        (each channel scaled by its gate)
# Read:  a*_t  = Σ_c g_t^(c) · (x_t @ S^(c))
#
# Expanding the read shows the parallel form:
#   a*_t = Σ_{τ<t} (x_t·x_τ) · (g_t·g_τ) · v*_τ
#       ≡ [(x @ x.T) * (g @ g.T)].tril(-1) @ v_ast
#
# k=1 reduction: softmax of a single logit is always 1, so g≡1, g·g≡1,
#   and both forms collapse exactly to the single-channel versions above.
# ---------------------------------------------------------------------------
def bdh_layer_mc_parallel(v_ast, Dx, Dy, E, W_g):
    """Multi-channel parallel form. W_g: (D, k)."""
    x = F.relu(ln(v_ast) @ Dx)                         # (T, N)
    g = F.softmax(v_ast @ W_g, dim=-1)                 # (T, k)
    scores = ((x @ x.T) * (g @ g.T)).tril(diagonal=-1) # (T, T) causal
    a_ast = scores @ v_ast                             # (T, D)
    y = F.relu(ln(a_ast) @ Dy) * x
    v_next = v_ast + ln(y @ E)
    return v_next, x, y, a_ast


def bdh_layer_mc_recurrent(v_ast, Dx, Dy, E, W_g, decay=1.0):
    """Multi-channel recurrent form. W_g: (D, k). Verified == mc_parallel below."""
    T, Dd = v_ast.shape
    N, k = Dx.shape[1], W_g.shape[1]
    x = F.relu(ln(v_ast) @ Dx)                         # (T, N)
    g = F.softmax(v_ast @ W_g, dim=-1)                 # (T, k)
    S = torch.zeros(k, N, Dd, dtype=v_ast.dtype)       # k per-channel states (k, N, D)
    a_ast = torch.zeros(T, Dd, dtype=v_ast.dtype)
    for t in range(T):
        # READ: blend channel reads by current gate
        reads = torch.einsum('n,knd->kd', x[t], S)     # (k, D): x_t @ S^(c) for each c
        a_ast[t] = g[t] @ reads                        # (D,): Σ_c g_t^c · (x_t @ S^c)
        # WRITE: each channel accumulates the Hebbian outer-product, scaled by its gate
        S = decay * (S + torch.einsum('k,nd->knd', g[t], torch.outer(x[t], v_ast[t])))
    y = F.relu(ln(a_ast) @ Dy) * x
    v_next = v_ast + ln(y @ E)
    return v_next, x, y, a_ast, S


# ---------------------------------------------------------------------------
# VERIFICATION
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    T, D, N = 16, 8, 32        # tiny: 16 tokens, d=8, n=32
    v_ast = torch.randn(T, D, dtype=torch.float64)
    Dx = torch.randn(D, N, dtype=torch.float64) * 0.1
    Dy = torch.randn(D, N, dtype=torch.float64) * 0.1
    E  = torch.randn(N, D, dtype=torch.float64) * 0.1

    vp, xp, yp, ap = bdh_layer_parallel(v_ast, Dx, Dy, E)
    vr, xr, yr, ar, S = bdh_layer_recurrent(v_ast, Dx, Dy, E, decay=1.0)

    da = (ap - ar).abs().max().item()
    dy = (yp - yr).abs().max().item()
    dv = (vp - vr).abs().max().item()
    print(f"max|a*_parallel - a*_recurrent|  = {da:.2e}")
    print(f"max|y_parallel  - y_recurrent |  = {dy:.2e}")
    print(f"max|v_parallel  - v_recurrent |  = {dv:.2e}")
    ok = max(da, dy, dv) < 1e-10
    print("\nVERIFIED: recurrent Hebbian == paper parallel form" if ok
          else "\nMISMATCH — do not trust the recurrent form")

    # also recover the full synaptic sigma (eq 16) for interpretability, small n
    # sigma_{T-1} = sum_{tau<T} y_tau ⊗ x_tau   (here using x as both, U=I)
    sigma = torch.zeros(N, N, dtype=torch.float64)
    for t in range(T):
        sigma += torch.outer(yr[t], xr[t])
    print(f"\nrecovered synaptic sigma: shape {tuple(sigma.shape)}, "
          f"nonzero structure ready for monosemanticity analysis")

    # -------------------------------------------------------------------
    # CHECK 2: multi-channel parallel == multi-channel recurrent  (k=3)
    # -------------------------------------------------------------------
    print("\n--- multi-channel (k=3) parallel vs recurrent ---")
    k = 3
    W_g = torch.randn(D, k, dtype=torch.float64) * 0.5

    vmp, xmp, ymp, amp = bdh_layer_mc_parallel(v_ast, Dx, Dy, E, W_g)
    vmr, xmr, ymr, amr, Sm = bdh_layer_mc_recurrent(v_ast, Dx, Dy, E, W_g, decay=1.0)

    dma = (amp - amr).abs().max().item()
    dmy = (ymp - ymr).abs().max().item()
    dmv = (vmp - vmr).abs().max().item()
    print(f"max|a*_mc_parallel - a*_mc_recurrent| = {dma:.2e}")
    print(f"max|y_mc_parallel  - y_mc_recurrent | = {dmy:.2e}")
    print(f"max|v_mc_parallel  - v_mc_recurrent | = {dmv:.2e}")
    ok_mc = max(dma, dmy, dmv) < 1e-10
    print("\nVERIFIED: MC recurrent == MC parallel form" if ok_mc
          else "\nMISMATCH — MC recurrent form is broken")

    # -------------------------------------------------------------------
    # CHECK 3: k=1 multi-channel reduces exactly to single-channel forms
    # -------------------------------------------------------------------
    print("\n--- k=1 reduction: MC forms == single-channel forms ---")
    W_g1 = torch.randn(D, 1, dtype=torch.float64) * 0.5   # gate always = [1.0] after softmax

    vmp1, _, _, amp1 = bdh_layer_mc_parallel(v_ast, Dx, Dy, E, W_g1)
    vmr1, _, _, amr1, _ = bdh_layer_mc_recurrent(v_ast, Dx, Dy, E, W_g1, decay=1.0)

    d_par = (ap - amp1).abs().max().item()
    d_rec = (ar - amr1).abs().max().item()
    d_vp  = (vp - vmp1).abs().max().item()
    d_vr  = (vr - vmr1).abs().max().item()
    print(f"max|a*_parallel  - a*_mc_parallel (k=1) | = {d_par:.2e}")
    print(f"max|a*_recurrent - a*_mc_recurrent(k=1) | = {d_rec:.2e}")
    print(f"max|v_parallel   - v_mc_parallel  (k=1) | = {d_vp:.2e}")
    print(f"max|v_recurrent  - v_mc_recurrent (k=1) | = {d_vr:.2e}")
    ok_k1 = max(d_par, d_rec, d_vp, d_vr) < 1e-10
    print("\nVERIFIED: k=1 MC == single-channel (exact reduction)" if ok_k1
          else "\nMISMATCH — k=1 reduction failed")

    all_ok = ok and ok_mc and ok_k1
    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
