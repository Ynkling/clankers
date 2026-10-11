#!/usr/bin/env python
"""
explore_delta_mem.py — EXPLORATORY, not a result. Session F, TASK 0: a drop-in channel memory with a (gated) delta-rule
write, in STATE form, for the multichannel BDH of test_multilayer_binding (MultiBDH and its window-gate subclasses).
It imports only modules that are the same on this branch and on the main line at explore_main9c.MAIN_SHA (bdh,
test_instrument_v2, test_multilayer_binding), so it can be used in child processes of either kind. It draws nothing
from the global RNG. Background: docs/reading/distant_cues_2026-10.md §4.1, §4.6; docs/reading/2412.06464.md (Gated
DeltaNet eq. 10 and the official code's α); docs/reading/2501.12352.md §5.1-5.3.

THE CURRENT (HEBBIAN) MEMORY, as test_multilayer_binding.GatedAttention computes it at every layer (positional 'decay'):
    o_t = Σ_{s<t} decay^(t-s) (x_t·x_s) (g_r[t]·g_w[s]) v_s,      then LayerNorm
with x = the layer's sparse code (Q = K, n = 256), v = the layer's input (V, d = 32). In state form, per channel c,
    S^c_t = decay·S^c_{t-1} + g^c_w[t] v_t x_tᵀ        (S^c ∈ R^{d×n}: values × keys, BDH's ρ ∈ R^{n×d} transposed)
    o_t   = Σ_c g^c_r[t] · decay·S^c_{t-1} x_t          (read after the decay at t, before the write at t: tril(-1))

THE DELTA MEMORY (this file). Per channel c, per layer, the state S^c ∈ R^{d×n}; at each token t, in this order:
    decay  S ← α^c_t S
    read   o_t = Σ_c g^c_r[t] S^c x̂_t                                  (strictly causal: before the write at t)
    write  S ← S − b^c_t (S x̂_t) x̂_tᵀ + w^c_t v_t x̂_tᵀ
i.e.   S^c_t = α^c_t ( S^c_{t-1} − b^c_t (S^c_{t-1} x̂_t) x̂_tᵀ ) + w^c_t v_t x̂_tᵀ     (the spec's formula, Gated DeltaNet eq. 10)
with
    x̂_t = x_t / max(‖x_t‖₂, 1e-6) if normalize else x_t   (the read uses the same key: query = key, as in BDH)
    b^c_t = g^c_w[t] β_t        the erase strength (the gated delta rule's g·β)
    w^c_t = g^c_w[t] β_t        the write strength when write = "tied" (Gated DeltaNet: the same β erases and writes);
          = g^c_w[t] · write    when write is a number (write = 1.0 with β = 0 is the Hebbian memory above: the erase off)
    β_t   = a fixed float, or σ(w_b·v_t + b_b) per token ("learned"; w_b = 0, b_b = 0 at init, so β = 0.5 exactly;
            v_t is the write vector V of the layer; w_b, b_b shared by the layers as every BDH weight is)
    α^c_t = decay (0.95, the model's GatedAttention.decay) unless a decay module is given:
      "routed" (Raven's routed decay, distant_cues §4.6): α^c_t = exp(a_t g^c_w[t]),  a_t = −softplus(w·v_t)·exp(Δ),
               w = 0 and Δ = ln(−ln 0.95 / ln 2) ≈ −2.6037 at init, so that g = 1 gives α = 0.95 exactly and g = 0 gives 1
      "gdn"    (Gated DeltaNet's code): α^c_t = exp(−A_c softplus(w_a^c·v_t + b_Δ^c)), A_c = exp(A_log_c) learnable,
               A_c ~ U(0, 16), b_Δ^c = softplus⁻¹(dt_c), dt_c log-uniform in [0.001, 0.1] (floor 1e-4), w_a = 0 at init;
               drawn from a generator seeded seed + GDN_OFFSET; no weight decay on A_log, b_Δ (no parameter here has any).
               This decay is not gated: under the oracle gate a channel still decays at the other stream's tokens.
The gate (g_r, g_w ∈ Δ^k at every position) is whatever the model's gate module computes (oracle, recurrent, LOCAL3):
to_delta() wraps the model so its gates reach the memory; or set them yourself with DeltaMemory.set_gates(gr, gw) (a
pair of (B, T, k) tensors used at every layer, or lists of n_layer such tensors for per-layer gates, Session G).

β = 0. With the erase strength 0 the write strength must be held at 1 for the Hebbian memory (β = 0 with write "tied"
writes nothing at all). So "β = 0" in the CHECKs means beta=0.0, write=1.0: the delta rule's erase switched off.

IMPLEMENTATIONS (impl=):
  "sequential"  the definition above, token by token, with the explicit states S^c (B, k, d, n). Reference only (slow;
                autograd keeps every state).
  "parallel"    the same numbers, exactly in exact arithmetic, without materialising S (the "UT form"): writing
                S^c_t = Σ_{s≤t} A^c(t,s) u^c_s x̂_sᵀ with A^c(t,s) = Π_{r=s+1..t} α^c_r, the effective write is
                    u^c_t = w^c_t v_t − b^c_t Σ_{s<t} A^c(t,s)(x̂_t·x̂_s) u^c_s,   i.e.  (I + diag(b^c) M^c) U^c = diag(w^c) V,
                    o_t   = Σ_c g^c_r[t] Σ_{s<t} A^c(t,s)(x̂_t·x̂_s) u^c_s,           i.e.  O = Σ_c diag(g^c_r) M^c U^c,
                with M^c[t,s] = A^c(t,s)(x̂_t·x̂_s) for s < t (0 on and above the diagonal; for a fixed α, A = decay^(t-s), the
                Hebbian decay mask). One unit-lower-triangular solve per channel (torch.linalg.solve_triangular). With b = 0
                it is the Hebbian score (Q Kᵀ ⊙ mask ⊙ G) V, summed per channel. Used for every run.
  "hebb"        the new code path disabled: test_multilayer_binding.GatedAttention.forward itself (the model's own
                G = g_r g_wᵀ), whatever beta / normalize are. A model converted with impl="hebb" runs the recorded runs
                bit for bit (CHECK).

CHECKS (explore_f_delta_checks.py, run before anything else and in every batch): with β = 0 (erase off, write 1),
normalize=False, the outputs equal the current parallel-score model's to 1e-5, and a recorded run (batch 16's S43
LOCAL3_SLOW seed 160, first 2400 updates, Xeon @ 2.10GHz) is reproduced bit for bit with impl="hebb"; "parallel" equals
"sequential" (forward and gradients, float64, every option); with the oracle gate every cross-stream contribution is
exactly zero (fixed and routed decay); with β = 1, normalize=True, a single channel after writing (k, v1) then (k, v2)
returns v2 at key k with ‖S k̂ − v2‖ < 1e-5.

USE
    m = build(...)                     # or explore_b16_gates.to_local(m, seed), explore_local_gate.make_model(...)
    m = to_delta(m, beta=1.0)          # β = 1, L2 keys, tied write, fixed decay, parallel
    m = to_delta(m, beta="learned")    # σ(w_b·v + b_b), β = 0.5 at init
    m = to_delta(m, beta=1.0, decay="gdn", seed=seed)
New parameters (attn.w_b, attn.b_b, attn.decay_mod.*) are registered under `attn`; they belong to the memory, not to
the gate (put them in the "rest" group of a slow schedule).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from test_instrument_v2 import Instrument, decay_mask, OPERATING_DECAY
from test_multilayer_binding import GatedAttention, MultiBDH, D

EPS = 1e-6
GDN_OFFSET = 59_000                                      # the GDN decay's own generator: seed + GDN_OFFSET
ROUTED_DELTA0 = math.log(-math.log(OPERATING_DECAY) / math.log(2.0))   # ≈ -2.6037: g = 1 gives 0.95 at init
GDN_A = (0.0, 16.0)
GDN_DT = (0.001, 0.1, 1e-4)                              # dt_min, dt_max, floor
IMPLS = ("parallel", "sequential", "hebb")


# ── Decay modules: log α^c_t, shape (B, T, k) ────────────────────────────────
class RoutedDecay(nn.Module):
    """Raven's routed decay: log α^c_t = a_t g^c_w[t], a_t = −softplus(w·v_t)·exp(Δ)."""

    kind = "routed"

    def __init__(self, d=D, delta0=ROUTED_DELTA0):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(d))
        self.delta = nn.Parameter(torch.tensor(float(delta0)))

    def forward(self, v, gw):
        a = -F.softplus(v @ self.w) * torch.exp(self.delta)          # (B, T)
        return a.unsqueeze(-1) * gw


class GDNDecay(nn.Module):
    """Gated DeltaNet's decay (official code): log α^c_t = −exp(A_log_c) softplus(w_a^c·v_t + b_Δ^c)."""

    kind = "gdn"

    def __init__(self, n_ch, seed, d=D):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed) + GDN_OFFSET)
        A = torch.rand(n_ch, generator=g) * (GDN_A[1] - GDN_A[0]) + GDN_A[0]
        lo, hi, floor = GDN_DT
        dt = torch.exp(torch.rand(n_ch, generator=g) * (math.log(hi) - math.log(lo)) + math.log(lo)).clamp(min=floor)
        self.A_log = nn.Parameter(torch.log(A))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))       # softplus⁻¹(dt)
        self.w_a = nn.Parameter(torch.zeros(n_ch, d))
        self.init = dict(A=A.tolist(), dt=dt.tolist())

    def forward(self, v, gw):
        return -torch.exp(self.A_log) * F.softplus(v @ self.w_a.T + self.dt_bias)  # (B, T, k)


# ── The memory ───────────────────────────────────────────────────────────────
class DeltaMemory(GatedAttention):
    """GatedAttention with the delta-rule channel memory (module docstring). Positional 'decay' only."""

    def __init__(self, config, decay=OPERATING_DECAY, beta=1.0, write="tied", normalize=True, decay_kind="fixed",
                 n_ch=1, seed=0, impl="parallel"):
        super().__init__(config, "decay", decay)
        assert impl in IMPLS, impl
        assert write == "tied" or isinstance(write, (int, float)), write
        assert beta == "learned" or isinstance(beta, (int, float)), beta
        self.impl, self.write, self.normalize, self.n_ch = impl, write, bool(normalize), int(n_ch)
        self.beta_mode = "learned" if beta == "learned" else "fixed"
        self.beta_fixed = None if beta == "learned" else float(beta)
        d = config.n_embd
        if beta == "learned":
            self.w_b = nn.Parameter(torch.zeros(d))
            self.b_b = nn.Parameter(torch.zeros(()))
        if decay_kind == "fixed":
            self.decay_mod = None
        elif decay_kind == "routed":
            self.decay_mod = RoutedDecay(d)
        elif decay_kind == "gdn":
            self.decay_mod = GDNDecay(n_ch, seed, d)
        else:
            raise ValueError(decay_kind)
        self.decay_kind = decay_kind
        self._gates, self._layer = None, 0
        self.last = None                    # diagnostics of the last call (β, α), detached, when keep_last is set
        self.keep_last = False

    # gates
    def set_gates(self, gr, gw):
        """(B, T, k) tensors for every layer, or lists of n_layer tensors (per-layer gates)."""
        self._gates, self._layer = (gr, gw), 0

    def _gates_now(self):
        assert self._gates is not None, "DeltaMemory: no gates set (use to_delta, or set_gates before the forward)"
        gr, gw = self._gates
        i = self._layer
        self._layer += 1
        if isinstance(gr, (list, tuple)):
            return gr[i], gw[i]
        return gr, gw

    # per-token quantities
    def beta_of(self, v):
        """β_t, shape (B, T)."""
        if self.beta_mode == "learned":
            return torch.sigmoid(v @ self.w_b + self.b_b)
        return torch.full(v.shape[:2], self.beta_fixed, dtype=v.dtype)

    def coefs(self, v, gw):
        """(b, w): erase and write strengths, (B, T, k)."""
        beta = self.beta_of(v).unsqueeze(-1)
        b = gw * beta
        w = b if self.write == "tied" else gw * float(self.write)
        return b, w

    def log_alpha(self, v, gw):
        """log α^c_t, (B, T, k); None for the fixed decay."""
        return None if self.decay_mod is None else self.decay_mod(v, gw)

    def keys(self, q):
        return F.normalize(q, dim=-1, eps=EPS) if self.normalize else q

    def forward(self, Q, K, V):
        if self.impl == "hebb":
            return GatedAttention.forward(self, Q, K, V)
        assert K is Q and Q.shape[1] == 1, "one head, Q = K"
        gr, gw = self._gates_now()
        q, v = Q[:, 0], V[:, 0]                                       # (B, T, n), (B, T, d)
        kh = self.keys(q)
        b, w = self.coefs(v, gw)
        la = self.log_alpha(v, gw)
        if self.keep_last:
            self.last = dict(beta=self.beta_of(v).detach(), log_alpha=None if la is None else la.detach())
        if self.impl == "sequential":
            o = delta_sequential(kh, v, gr, b, w, la, self.decay)[0]
        else:
            o = delta_parallel(kh, v, gr, b, w, la, self.decay, skip_solve=self.erase_off())[0]
        return o.unsqueeze(1)

    def erase_off(self):
        return self.beta_mode == "fixed" and self.beta_fixed == 0.0


# ── The two implementations (functions of plain tensors, for the CHECKs) ────
def decay_matrix(la, decay, T, dtype):
    """A(t,s) for s < t, 0 elsewhere: (T, T) for the fixed decay (the Hebbian mask), else (B, k, T, T)."""
    if la is None:
        return decay_mask(T, decay, dtype)
    lam = la.cumsum(1).transpose(1, 2)                                # (B, k, T): Λ_t = Σ_{r≤t} log α_r
    diff = lam.unsqueeze(-1) - lam.unsqueeze(-2)                      # Λ_t − Λ_s
    low = torch.ones(T, T, dtype=torch.bool).tril(-1)
    return torch.exp(diff.masked_fill(~low, float("-inf")))


def delta_parallel(kh, v, gr, b, w, la, decay, skip_solve=False):
    """kh (B,T,n) keys; v (B,T,d); gr, b, w (B,T,k); la (B,T,k) or None. Returns o (B,T,d)."""
    B, T, _ = kh.shape
    KK = kh @ kh.mT                                                   # (B, T, T)
    A = decay_matrix(la, decay, T, kh.dtype)
    M = (KK * A).unsqueeze(1) if la is None else KK.unsqueeze(1) * A   # (B, 1|k, T, T), strictly lower
    rhs = w.transpose(1, 2).unsqueeze(-1) * v.unsqueeze(1)             # (B, k, T, d)
    if skip_solve:
        U = rhs
    else:
        L = b.transpose(1, 2).unsqueeze(-1) * M                        # (B, k, T, T); the unit diagonal is implicit
        U = torch.linalg.solve_triangular(L, rhs, upper=False, unitriangular=True)
    O = M @ U                                                          # (B, k, T, d)
    return (gr.transpose(1, 2).unsqueeze(-1) * O).sum(1), U


def delta_sequential(kh, v, gr, b, w, la, decay, keep_states=False):
    """The definition, token by token. Returns (o (B,T,d), states (list of (B,k,d,n) after each token) or None)."""
    B, T, n = kh.shape
    d, k = v.shape[-1], gr.shape[-1]
    S = torch.zeros(B, k, d, n, dtype=kh.dtype)
    outs, states = [], []
    for t in range(T):
        a = (torch.full((B, k), float(decay), dtype=kh.dtype) if la is None else torch.exp(la[:, t]))
        S = a[..., None, None] * S                                     # decay
        x = kh[:, t]                                                   # (B, n)
        Sx = (S @ x[:, None, :, None]).squeeze(-1)                     # (B, k, d)
        outs.append((gr[:, t, :, None] * Sx).sum(1))                   # read
        S = (S - b[:, t, :, None, None] * Sx.unsqueeze(-1) * x[:, None, None, :]
             + w[:, t, :, None, None] * v[:, t, None, :, None] * x[:, None, None, :])   # erase, write
        if keep_states:
            states.append(S)
    return torch.stack(outs, 1), (states if keep_states else None)


# ── Model conversion ─────────────────────────────────────────────────────────
_CLASSES = {}


def _delta_class(cls):
    if cls in _CLASSES:
        return _CLASSES[cls]
    if getattr(cls, "_delta", False):
        return cls
    if hasattr(cls, "local_gates"):
        # the window-gate models (explore_local_gate.LocalGateBDH, explore_b16_gates.LocalBDH): the gate is computed
        # once, in local_gates; pass it on to the memory
        def local_gates(self, v):
            g = cls.local_gates(self, v)
            self.attn.set_gates(*g)
            return g
        new = type("Delta" + cls.__name__, (cls,), dict(local_gates=local_gates, _delta=True))
    else:
        assert issubclass(cls, MultiBDH), cls

        def forward(self, tokens, tau=None, track_sat=False):
            # MultiBDH.forward's own gate computation (Instrument.gates), passed to the memory; then MultiBDH.forward
            # runs unchanged (it computes the gate again, deterministically: no noise is allowed here)
            assert self.gate_noise == 0.0, "gate noise is random: it would differ between the two computations"
            v = self.embed(tokens)
            vg = F.layer_norm(v, (v.shape[-1],)) if self.gate_ln else v
            self.attn.set_gates(*Instrument.gates(self, tokens, vg, tau))
            return cls.forward(self, tokens, tau, track_sat)
        new = type("Delta" + cls.__name__, (cls,), dict(forward=forward, _delta=True))
    _CLASSES[cls] = new
    return new


def to_delta(m, beta=1.0, write="tied", normalize=True, decay=None, impl="parallel", seed=0):
    """Convert a built MultiBDH (or a window-gate subclass) in place: its attention becomes a DeltaMemory with the
    model's own decay constant; no existing parameter changes; new parameters live under m.attn. decay: None
    (fixed, the model's 0.95 or whatever attn.decay is), "routed" or "gdn" (seed draws the GDN init)."""
    old = m.attn
    assert type(old) is GatedAttention and old.positional == "decay", (type(old), getattr(old, "positional", None))
    assert getattr(m, "gate_to_readout", False) is False, "gate_to_readout is not supported"
    new = DeltaMemory(m.config, decay=old.decay, beta=beta, write=write, normalize=normalize,
                      decay_kind=decay or "fixed", n_ch=m.n_ch, seed=seed, impl=impl)
    assert torch.equal(new.freqs, old.freqs)
    new.record = old.record
    m.attn = new
    m.__class__ = _delta_class(type(m))
    m.delta_cfg = dict(beta=beta, write=write, normalize=bool(normalize), decay=decay or "fixed", impl=impl)
    return m


def converter(**kw):
    """A conversion f(m, seed) -> m for explore_b16_main.make_rc(convert=...) and wrap_builder."""
    def conv(m, seed):
        return to_delta(m, seed=seed, **kw)
    return conv


def delta_params(m):
    """Names of the parameters to_delta added (they belong to the memory, i.e. to the 'rest' group)."""
    return [n for n, _ in m.named_parameters() if n.startswith("attn.")]
