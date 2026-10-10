"""
explore_gpu/batched.py — EXPLORATORY tooling, not a result. The BATCHED path: R runs of one architecture trained at once on
one device, as ONE stacked model (every parameter gets a leading run dimension R) with a per-run Adam. Nothing here edits
a project module; the R models are built by the project's own builders on CPU (so every initial parameter, every batch
generator and every split's noise is the run's own, bit for bit), then stacked.

WHAT IS STACKED (BatchedBDH.from_models): any model the project's builders return, of these classes:
  MultiBDH (test_multilayer_binding)  gate "none" (k = 1), "perfect" (Instrument's perfect_gate_general on the task's
                                      context tokens), "recurrent" (Instrument.gates' loop, h_t = tanh(W_in v_t + W_h h_{t-1}),
                                      g = softmax(W_g h), read = write, mode "sym", gate_ln allowed); positional "decay";
                                      conv None / "in" / "layer"; dropout 0; one head; no gate_to_readout, no gate noise.
  LocalBDH (explore_b16_gates)        the window gate at any width (LOCAL3 = 3; WIDE = LMAX + 2): u = causal conv of the raw
                                      embeddings, h = tanh(W_in u), g = softmax(W_g h).
  RegBDH / Reg3Conv (explore_g_regmodel, explore_i_common)  the per-layer gate over [window or v; LN(residual); R_{t-1}]
                                      with S56's register; sg / nosg; any gated / res_layers; conv None or "layer".
  any of the above converted by explore_delta_mem.to_delta (or explore_f_delta_fast.to_delta_fast): the delta memory in
                                      its parallel (UT) form, explore_delta_mem.delta_parallel itself (impl "fast" is the same
                                      rule; F's CHECK puts the two within 1e-5 relative), beta fixed or learned, write tied or
                                      a number, keys normalised or not, decay fixed / "routed" / "gdn"; impl "hebb" is the
                                      Hebbian memory.
THE FORWARD reproduces, line for line in the same operation order, bdh.BDH.forward / MultiBDH.forward_conv (the stack),
Instrument.gates / LocalBDH.local_gates / RegBDH.layer_gate (the gates) and GatedAttention.forward / DeltaMemory.forward
(the memory), with every activation carrying the run dimension: (R, B, T, ·). The register's recurrence
R_t = R_{t-1}(1 - beta_t) + beta_t W_R v_t has two forms: "loop" (RegBDH.register's loop, the same operations) and
"scan" (default on CUDA: R_{t-1} = sum_{s<t} exp(C_{t-1} - C_s) beta_s W_R v_s with C = cumsum log(1 - beta) in float64,
log(1 - beta) = -softplus(z) computed from the gate's logit z, so a saturated beta = 1.0 stays finite with a finite
gradient; equal to the loop to float32 rounding, CHECKed also with saturated beta), because a 767-step Python loop of
tiny kernels would dominate a GPU step.
NUMERICS: the same float32 operations as the sequential path; matrix products run as batched GEMMs over (R, B) or over R
with B·T rows, so results may differ from the sequential path in the last bits (the CHECKs bound the difference).
StackedAdam is torch.optim.Adam's single-tensor update (lerp_, mul_/addcmul_, Python-float bias corrections,
sqrt / bias_correction2_sqrt + eps, addcdiv_) applied to the stacked tensors; every run keeps its own step count per
parameter, so a split can zero ONE run's W_g state (exp_avg, exp_avg_sq and step, as test_window_gate.zero_state does);
while every run's step count is equal the update is the very same call torch makes, else the per-run scalars enter as
float32 tensors in the kernel's own order (value * exp_avg / denom, then + param). Runs that stopped early are frozen
(parameters and optimizer state restored after the step).
VIEWS: view(r) returns a copy of run r's own sequential model whose parameters ALIAS the stacked slices (no copy), so the
project's evaluation, statistics and split code read run r exactly as they read a sequential model.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from test_multilayer_binding import MultiBDH
from test_instrument_v2 import decay_mask, perfect_gate_general
import explore_b16_gates as g16
import explore_g_regmodel as rm
import explore_delta_mem as dm

# ── architecture signature ────────────────────────────────────────────────────
def signature(m):
    """What the batched forward must do for model m (equal for every stacked run)."""
    attn = m.attn
    delta = isinstance(attn, dm.DeltaMemory) and attn.impl != "hebb"
    s = dict(cls=type(m).__name__, n_layer=m.config.n_layer, D=m.config.n_embd, nh=m.config.n_head,
             mult=m.config.mlp_internal_dim_multiplier, vocab=m.config.vocab_size, dropout=m.config.dropout,
             n_ch=m.n_ch, conv=m.conv, positional=attn.positional, decay=float(attn.decay),
             ctx_tokens=None if m.ctx_tokens is None else tuple(m.ctx_tokens), gate_ln=bool(m.gate_ln),
             gate_noise=float(m.gate_noise), g2r=bool(m.gate_to_readout), mode=m.mode,
             names=tuple(n for n, _ in m.named_parameters()), delta=delta)
    if isinstance(m, rm.RegBDH):
        s.update(kind="reg", gated=tuple(m.gated), res_layers=tuple(m.res_layers), sg=bool(m.sg), reg=bool(m.reg),
                 inp=getattr(m, "inp", "v"), mem=m.mem, width=m.gate_conv.conv_w.shape[1])
    elif g16.is_local(m):              # explore_b16_gates.LocalBDH AND test_window_gate's own LocalBDH (same forward)
        s.update(kind="window", width=m.gate_conv.conv_w.shape[1])
    else:
        assert isinstance(m, MultiBDH), type(m)
        s.update(kind=m.gate_kind)
    if delta:
        s.update(d_write=attn.write, d_norm=attn.normalize, d_beta=attn.beta_mode, d_beta_fixed=attn.beta_fixed,
                 d_decay=attn.decay_kind, d_impl=attn.impl)
    return s


def supported(s):
    assert s["dropout"] == 0.0 and s["nh"] == 1, "dropout 0 and one head only"
    assert s["positional"] == "decay", "positional 'decay' only (every Phase III-VI arm)"
    assert not s["g2r"] and s["gate_noise"] == 0.0 and s["mode"] == "sym", "no readout gate, no gate noise, sym mode"
    assert s["kind"] in ("none", "perfect", "recurrent", "window", "reg"), s["kind"]
    if s["kind"] == "reg":
        assert s["conv"] in (None, "layer") and s["mem"] in ("hebb", "delta")
        # RegBDH must have been built for the memory it has: to_reg(mem="delta") then delta attention. A Reg3 model built
        # Hebbian and converted by to_delta afterwards feeds LOCAL3's gate to every layer sequentially (to_delta wraps
        # local_gates), which is not what the batched per-layer gates compute: rejected.
        assert s["delta"] == (s["mem"] == "delta"), "RegBDH: mem and the attention's memory disagree (convert with mem='delta')"


# ── small batched helpers ─────────────────────────────────────────────────────
def bmm_rows(x, W):
    """(R, ..., X) @ (R, X, Y) -> (R, ..., Y), the rows of every run folded into one GEMM per run."""
    sh = x.shape
    return torch.bmm(x.reshape(sh[0], -1, sh[-1]), W).reshape(*sh[:-1], W.shape[-1])


def bcast_mm(x, W):
    """The sequential path's broadcast matmul (B, 1, T, X) @ (1, X, Y) for every run: x (R, B, T, X), W (R, 1, X, Y) ->
    (R, B, T, Y) as (R, B, 1, T, X) @ (R, 1, 1, X, Y), i.e. a GEMM per sequence with the weight broadcast, whose weight
    gradient is the per-sequence products summed over the batch (bitwise the sequential path's on CPU, CHECKed)."""
    return (x.unsqueeze(2) @ W.unsqueeze(1)).squeeze(2)


def mv_runs(x, w):
    """The sequential path's matrix-vector product (B, T, X) @ (X,) for every run: x (R, B, T, X), w (R, X) -> (R, B, T)."""
    return torch.stack([x[r] @ w[r] for r in range(x.shape[0])])


def bconv(x, w):
    """CausalConv.forward with per-run weights: x (R, B, T, D), w (R, D, K); out_t = sum_j w[:, j] x_{t-j}, the same
    operation order (x * w0, then + shifted x * wj for j = 1..K-1)."""
    R, K, T = x.shape[0], w.shape[-1], x.shape[-2]
    wv = w.reshape(R, *([1] * (x.dim() - 2)), w.shape[1], K)
    xp = F.pad(x, (0, 0, K - 1, 0))
    out = x * wv[..., 0]
    for j in range(1, K):
        out = out + xp[..., K - 1 - j:K - 1 - j + T, :] * wv[..., j]
    return out


def ln(x):
    return F.layer_norm(x, (x.shape[-1],))


# ── the stacked model ─────────────────────────────────────────────────────────
class BatchedBDH(nn.Module):
    def __init__(self, models, device, register="auto"):
        super().__init__()
        for m in models:                    # transient forward state (a nudge's last attn.G is a non-leaf: no deepcopy)
            m.attn.G = None
            if hasattr(m, "diag"):
                m.diag = None
            if hasattr(m.attn, "_gates"):
                m.attn._gates, m.attn._layer = None, 0
        sigs = [signature(m) for m in models]
        assert all(s == sigs[0] for s in sigs), "every stacked run must have the same architecture"
        self.sig = s = sigs[0]
        supported(s)
        self.R = len(models)
        self.device = torch.device(device)
        self.names = list(s["names"])
        self.ps = nn.ParameterList()
        for n in self.names:
            ref = dict(models[0].named_parameters())[n]
            st = torch.stack([dict(m.named_parameters())[n].detach() for m in models]).to(self.device)
            self.ps.append(nn.Parameter(st, requires_grad=ref.requires_grad))
        self.ix = {n: i for i, n in enumerate(self.names)}
        self._masks = {}
        self.register_mode = (("scan" if self.device.type == "cuda" else "loop") if register == "auto" else register)
        self._cpu_models = [copy.deepcopy(m) for m in models]       # templates for the views
        self._views = {}
        self.keep = False
        self.last = None

    def P(self, name):
        return self.ps[self.ix[name]]

    def stacked_named(self):
        return [(n, self.ps[i]) for i, n in enumerate(self.names)]

    def mask(self, T):
        if T not in self._masks:
            with torch.device(self.device):
                self._masks[T] = decay_mask(T, self.sig["decay"], torch.float32).to(self.device)
        return self._masks[T]

    # gates
    def window_gate(self, v):
        u = bconv(v, self.P("gate_conv.conv_w"))
        h = torch.tanh(bmm_rows(u, self.P("W_in").mT))
        return F.softmax(bmm_rows(h, self.P("W_g").mT), dim=-1)

    def recurrent_gate(self, v):
        R, B, T, _ = v.shape
        W_in, W_h, W_g = self.P("W_in"), self.P("W_h"), self.P("W_g")
        h = torch.zeros(R, B, W_in.shape[1], dtype=v.dtype, device=v.device)
        logits = []
        for t in range(T):
            h = torch.tanh(torch.bmm(v[:, :, t], W_in.mT) + torch.bmm(h, W_h.mT))
            logits.append(torch.bmm(h, W_g.mT))
        return F.softmax(torch.stack(logits, dim=2), dim=-1)

    def perfect_gate(self, tok):
        R, B, T = tok.shape
        with torch.device(self.device):
            g = perfect_gate_general(tok.reshape(R * B, T), self.sig["ctx_tokens"], torch.float32)
        return g.reshape(R, B, T, -1)

    def register(self, v):
        """(Rprev (R,B,T,R_DIM), beta (R,B,T)); RegBDH.register per run."""
        z = mv_runs(v, self.P("w_R")) + self.P("b_R").view(-1, 1, 1)
        beta = torch.sigmoid(z)
        w = bmm_rows(v, self.P("W_R").mT)
        R_, B, T, _ = v.shape
        if self.register_mode == "loop":
            Rs = v.new_zeros(R_, B, w.shape[-1])
            prev = []
            for t in range(T):
                prev.append(Rs)
                b = beta[:, :, t:t + 1]
                Rs = Rs * (1 - b) + b * w[:, :, t]
            return torch.stack(prev, 2), beta
        lg = -F.softplus(z.double())                                            # log(1 - sigmoid z): finite, smooth
        C = lg.cumsum(-1)
        Cprev = torch.cat([torch.zeros_like(C[..., :1]), C[..., :-1]], -1)      # C_{t-1}
        diff = Cprev.unsqueeze(-1) - C.unsqueeze(-2)                            # C_{t-1} - C_s
        low = torch.ones(T, T, dtype=torch.bool, device=v.device).tril(-1)
        Wts = torch.exp(diff.masked_fill(~low, float("-inf"))).to(v.dtype)     # (R, B, T, T), s < t
        return Wts @ (beta.unsqueeze(-1) * w), beta

    def layer_gate(self, level, h, v, Rprev):
        s = self.sig
        x_in = bconv(v, self.P("gate_conv.conv_w")) if s["inp"] == "window" else v
        u = bmm_rows(x_in, self.P("A_v").mT)
        if level in s["res_layers"]:
            r = h.detach() if s["sg"] else h
            u = u + bmm_rows(ln(r), self.P("A_r").mT)
        if Rprev is not None:
            u = u + bmm_rows(Rprev, self.P("A_R").mT)
        return F.softmax(bmm_rows(torch.tanh(u), self.P("W_g2").mT), dim=-1)

    # memories
    def hebb(self, Q, V, G):
        scores = (Q @ Q.mT) * self.mask(Q.shape[-2])
        if G is not None:
            scores = scores * G
        return scores.tril(diagonal=-1) @ V

    def delta(self, Q, V, gr, gw):
        s = self.sig
        R_, B, T, n = Q.shape
        kh = F.normalize(Q, dim=-1, eps=dm.EPS) if s["d_norm"] else Q
        if s["d_beta"] == "learned":
            beta = torch.sigmoid(mv_runs(V, self.P("attn.w_b")) + self.P("attn.b_b").view(-1, 1, 1))
        else:
            beta = torch.full(V.shape[:3], s["d_beta_fixed"], dtype=V.dtype, device=V.device)
        beta = beta.unsqueeze(-1)
        b = gw * beta
        w = b if s["d_write"] == "tied" else gw * float(s["d_write"])
        if s["d_decay"] == "fixed":
            la = None
        elif s["d_decay"] == "routed":
            a = -F.softplus(mv_runs(V, self.P("attn.decay_mod.w"))) \
                * torch.exp(self.P("attn.decay_mod.delta")).view(-1, 1, 1)
            la = a.unsqueeze(-1) * gw
        else:
            la = -torch.exp(self.P("attn.decay_mod.A_log")).view(R_, 1, 1, -1) * F.softplus(
                bmm_rows(V, self.P("attn.decay_mod.w_a").mT) + self.P("attn.decay_mod.dt_bias").view(R_, 1, 1, -1))
        fl = lambda t: None if t is None else t.reshape(R_ * B, *t.shape[2:])
        skip = s["d_beta"] == "fixed" and s["d_beta_fixed"] == 0.0
        with torch.device(V.device):
            o = dm.delta_parallel(fl(kh), fl(V), fl(gr), fl(b), fl(w), fl(la), s["decay"], skip_solve=skip)[0]
        return o.reshape(R_, B, T, -1)

    # forward
    def forward(self, tok):
        """tok (R, B, T) on the device -> (logits (R, B, T, vocab), gates: list per layer of (R, B, T, k) or None)."""
        s = self.sig
        R_, B, T = tok.shape
        E = self.P("embed.weight")
        Ef = E.reshape(R_ * E.shape[1], E.shape[2])                       # run r's rows at r*vocab .. (r+1)*vocab-1
        idx = tok + torch.arange(R_, device=tok.device).view(-1, 1, 1) * E.shape[1]
        kind = s["kind"]
        Rprev = None
        if kind == "reg":
            v = F.embedding(idx, Ef)
            g1 = self.window_gate(v)
            Rprev, beta = self.register(v) if s["reg"] else (None, None)
        else:
            v = F.embedding(idx, Ef)
            vg = ln(v) if s["gate_ln"] else v
            if kind == "none":
                g1 = torch.ones(R_, B, T, 1, dtype=v.dtype, device=v.device) if s["delta"] else None
            elif kind == "perfect":
                g1 = self.perfect_gate(tok)
            elif kind == "recurrent":
                g1 = self.recurrent_gate(vg)
            else:
                g1 = self.window_gate(v)
            beta = None
        G1 = None if (g1 is None or s["delta"]) else torch.einsum("rbtk,rbsk->rbts", g1, g1)
        x = F.embedding(idx, Ef)
        if s["conv"] == "in":
            x = bconv(x, self.P("short_conv.conv_w"))
        x = ln(x)
        gates, feats = [], []
        enc, enc_v, dec = self.P("encoder"), self.P("encoder_v"), self.P("decoder")
        for level in range(s["n_layer"]):
            lv = level + 1
            if self.keep:
                feats.append(x)
            if kind == "reg" and lv in s["gated"]:
                g = self.layer_gate(lv, x, v, Rprev)
                G = None if s["delta"] else torch.einsum("rbtk,rbsk->rbts", g, g)
            else:
                g, G = g1, G1
            gates.append(g)
            xc = bconv(x, self.P("short_conv.conv_w")) if s["conv"] == "layer" else x
            x_sparse = F.relu(bcast_mm(xc, enc))
            yKV = self.delta(x_sparse, xc, g, g) if s["delta"] else self.hebb(x_sparse, xc, G)
            yKV = ln(yKV)
            y_sparse = F.relu(bcast_mm(yKV, enc_v))
            xy = x_sparse * y_sparse
            y = ln(bmm_rows(xy, dec))
            x = ln(x + y)
        if self.keep:
            feats.append(x)
            self.last = dict(feats=[f.detach() for f in feats], gates=[None if g is None else g.detach() for g in gates],
                             beta=None if beta is None else beta.detach())
        logits = bmm_rows(x, self.P("lm_head"))
        return logits, gates

    # views
    def view(self, r, wrap=None):
        """Run r's sequential model, its parameters aliasing the stacked slices (cached). wrap(model) -> model converts the
        class for the device (explore_gpu.device.to_device)."""
        if r not in self._views:
            m = copy.deepcopy(self._cpu_models[r])
            if wrap is not None:
                m = wrap(m)
            named = dict(m.named_parameters())
            for n in self.names:
                named[n].data = self.P(n).data[r]
            self._views[r] = m
        return self._views[r]

    @classmethod
    def from_models(cls, models, device, **kw):
        return cls(models, device, **kw)


# ── per-run Adam ──────────────────────────────────────────────────────────────
class StackedAdam:
    """torch.optim.Adam (single-tensor path, defaults: betas (0.9, 0.999), eps 1e-8, no weight decay, no amsgrad) on stacked
    parameters, with a step count per run and parameter. param_groups: list of dict(params=[...], lr=...), like torch's
    (lr may be changed between steps)."""

    def __init__(self, groups, R, betas=(0.9, 0.999), eps=1e-8):
        self.param_groups = groups
        self.R = R
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.state = {}

    def zero_grad(self):
        for g in self.param_groups:
            for p in g["params"]:
                p.grad = None

    @torch.no_grad()
    def step(self, active=None):
        b1, b2, eps = self.beta1, self.beta2, self.eps
        act = None if active is None or all(active) else [bool(a) for a in active]
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state.get(p)
                if st is None:
                    # step: one count per run, an (R,) float64 CPU tensor, so a view of run r's entry can be zeroed in
                    # place by test_window_gate / explore_i_common's zero_state (as torch.optim.Adam's step tensor is)
                    st = self.state[p] = dict(step=torch.zeros(self.R, dtype=torch.float64),
                                              exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
                keep = None
                if act is not None:
                    idx = [r for r in range(self.R) if not act[r]]
                    keep = (idx, p[idx].clone(), st["exp_avg"][idx].clone(), st["exp_avg_sq"][idx].clone())
                for r in range(self.R):
                    if act is None or act[r]:
                        st["step"][r] += 1
                grad, m, v = p.grad, st["exp_avg"], st["exp_avg_sq"]
                m.lerp_(grad, 1 - b1)
                v.mul_(b2).addcmul_(grad, grad, value=1 - b2)
                steps = [int(x) for x in st["step"].tolist()]
                if all(x == steps[0] for x in steps) and steps[0] > 0:
                    step = float(steps[0])
                    step_size = lr / (1 - b1 ** step)
                    bc2_sqrt = (1 - b2 ** step) ** 0.5
                    denom = (v.sqrt() / bc2_sqrt).add_(eps)
                    p.addcdiv_(m, denom, value=-step_size)
                else:
                    sh = (self.R,) + (1,) * (p.dim() - 1)
                    ss = [(-(lr / (1 - b1 ** float(x))) if x > 0 else 0.0) for x in steps]
                    bc = [((1 - b2 ** float(x)) ** 0.5 if x > 0 else 1.0) for x in steps]
                    sst = torch.tensor(ss, dtype=p.dtype, device=p.device).view(sh)
                    bct = torch.tensor(bc, dtype=p.dtype, device=p.device).view(sh)
                    denom = (v.sqrt() / bct).add_(eps)
                    p.add_(sst * m / denom)
                if keep is not None:
                    idx, p0, m0, v0 = keep
                    p[idx] = p0
                    m[idx] = m0
                    v[idx] = v0

    def zero_run_state(self, p, r):
        """test_window_gate.zero_state for run r: exp_avg, exp_avg_sq and step zeroed (only if the state exists)."""
        st = self.state.get(p)
        if not st:
            return []
        st["step"][r] = 0
        st["exp_avg"][r].zero_()
        st["exp_avg_sq"][r].zero_()
        return ["step", "exp_avg", "exp_avg_sq"]

    def run_state(self, p, r):
        """Run r's state of stacked parameter p as views (step (1,), exp_avg, exp_avg_sq), in torch.optim.Adam's key
        order: zeroing them in place is test_window_gate.zero_state on that run's Adam state."""
        st = self.state.get(p)
        if not st:
            return None
        return {"step": st["step"][r:r + 1], "exp_avg": st["exp_avg"][r], "exp_avg_sq": st["exp_avg_sq"][r]}


def adam_groups(bm, gate_names=None, lr=1e-3, lr_rest=None):
    """One group of every trainable stacked parameter (ADAM), or SLOW's two: gate_names at lr, the rest at lr_rest."""
    named = [(n, p) for n, p in bm.stacked_named() if p.requires_grad]
    if gate_names is None:
        return [dict(params=[p for _, p in named], lr=lr)], [[n for n, _ in named]]
    gate = [n for n in gate_names if n in dict(named)]
    rest = [n for n, _ in named if n not in gate]
    nm = dict(named)
    return ([dict(params=[nm[n] for n in gate], lr=lr), dict(params=[nm[n] for n in rest], lr=lr_rest)],
            [gate, rest])
