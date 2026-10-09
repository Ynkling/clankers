#!/usr/bin/env python
"""
explore_g_latch_child.py — EXPLORATORY, not a result. S57's run code (Session G) and the post-hoc diagnostic arm ORACLE23.
Runs INSIDE a child process on this branch's own modules, 1 torch thread. Parent, arms and readings: explore_g_latch.

LATCH (LatchBDH, a subclass of explore_g_regmodel.RegBDH; every other part of the model, the recipe (SLOW), the nudge, the
statistics and the outcomes are S56's, explore_g_register_child): the register is replaced by an HM-RNN latch,
    u_t = w_z . v_t + b_z,   z~_t = max(0, min(1, (a u_t + 1) / 2)),   z_t = 1[z~_t > 0.5] (forward),
    backward: straight-through with the hard sigmoid's derivative (dz/du = a/2 where 0 < z~ < 1, else 0),
    r_t = z_t W_r v_t + (1 - z_t) r_{t-1},  r_{-1} = 0,  r in R^16;  the gate at t reads r_{t-1} (causal, as S56).
  Parameter names reuse S56's (W_R = W_r ~ N(0, 0.1^2), w_R = w_z ~ N(0, 0.01^2), b_R = b_z = 0), so S56's groups and nudge
  apply unchanged; drawn from the same generator (seed + REG_OFFSET) in the same order, b_R set to 0.
  a: 1 for updates 1-8000, then linear from 1 to 5 over updates 8001-24000 (the last two thirds of the budget); the update
  count is the number of training-mode forwards after the nudge (one per update in onset_run).
  PRIOR (LATCH_PRIOR*): lambda (mean_t z_t - 1/(2 P + 1))^2, lambda = 0.1, P = 8 (target 1/17), mean over the sequence's
  positions, averaged over the batch; added to the loss on training forwards through explore_aux_gate._Inject.
ORACLE23 (post hoc, descriptive; NOT one of S56's or S57's pre-fixed arms): layer 1 LOCAL3, layers 2-3 the perfect gate
  (test_instrument_v2.perfect_gate_general: the latest CTX token's one-hot), SLOW recipe (gate group LOCAL3's W_in, W_g,
  window). It asks whether the unrouted layer-1 memory alone keeps REG_NUDGE from binding.
"""

import torch

import explore_g_regmodel as rm
import explore_g_register_child as s56
import explore_aux_gate as aux
from test_instrument_v2 import perfect_gate_general

A_END_STEP, A_START_STEP, A0, A1 = 24000, 8000, 1.0, 5.0
LAM, P_BAR = 0.1, 8


class _STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, zt):
        return (zt > 0.5).to(zt.dtype)

    @staticmethod
    def backward(ctx, g):
        return g


def a_at(step):
    if step <= A_START_STEP:
        return A0
    return A0 + (A1 - A0) * min(1.0, (step - A_START_STEP) / (A_END_STEP - A_START_STEP))


class LatchBDH(rm.RegBDH):
    def register(self, v):
        a = a_at(self.n_upd)
        u = v @ self.w_R + self.b_R                                       # (B, T)
        zt = torch.clamp((a * u + 1) / 2, 0.0, 1.0)                       # hard sigmoid (derivative a/2 inside)
        z = _STE.apply(zt)                                                # forward: 1[z~ > 0.5]; backward: dz~/du
        w = v @ self.W_R.T
        B, T, _ = v.shape
        r = v.new_zeros(B, rm.R_DIM)
        prev = []
        for t in range(T):
            prev.append(r)
            zz = z[:, t:t + 1]
            r = zz * w[:, t] + (1 - zz) * r
        self._z = z
        return torch.stack(prev, 1), z

    def forward(self, tokens, tau=None, track_sat=False):
        out = rm.RegBDH.forward(self, tokens, tau, track_sat)
        if self.training and self.count_upd:
            self.n_upd += 1
            if self.prior:
                pen = LAM * ((self._z.mean(1) - 1.0 / (2 * P_BAR + 1)) ** 2).mean()
                return (aux._Inject.apply(out[0], pen),) + tuple(out[1:])
        return out


def to_latch(m, seed, prior=False):
    m = rm.to_reg(m, seed)
    m.__class__ = LatchBDH
    with torch.no_grad():
        m.b_R.zero_()
    m.prior, m.n_upd, m.count_upd = bool(prior), 0, False
    return m


class Oracle23BDH(rm.RegBDH):
    def layer_gate(self, level, h, v, Rprev):
        return perfect_gate_general(self._tok, self.ctx_tokens).to(v.dtype)

    def forward(self, tokens, tau=None, track_sat=False):
        self._tok = tokens
        return rm.RegBDH.forward(self, tokens, tau, track_sat)


def make(arm, seed):
    import explore_b16_gates as g16
    import explore_common as ec
    from test_multilayer_binding import build
    from test_channel_binding import ARCH
    base = g16.to_local(build(s56.H8, ec.ARM_A, ARCH, seed), seed)
    if arm == "ORACLE23":
        m = rm.to_reg(base, seed, reg=False)
        for n in ("A_v", "A_r", "W_g2"):                 # unused by the perfect gate: removed
            delattr(m, n)
        m.__class__ = Oracle23BDH
        return m
    m = to_latch(base, seed, prior=arm in ("LATCH_PRIOR", "LATCH_PRIOR_NUDGE"))
    if arm in ("LATCH_NUDGE", "LATCH_PRIOR_NUDGE"):
        s56.nudge_reg(m, s56.H8, seed)
    m.count_upd = True
    return m


ARMS = ("LATCH", "LATCH_NUDGE", "LATCH_PRIOR", "LATCH_PRIOR_NUDGE", "ORACLE23")


@torch.no_grad()
def z_by_role(model, task, probe):
    """The latch's firing rate z at CTX / KEY / VAL / QUERY positions (S56's beta_by_role, for the latch)."""
    if not isinstance(model, LatchBDH):
        return None
    pinp = probe[0]
    was = model.training
    model.eval()
    try:
        u = model.embed(pinp) @ model.w_R + model.b_R
        z = (torch.clamp((a_at(model.n_upd) * u + 1) / 2, 0, 1) > 0.5).float()
    finally:
        model.train(was)
    return {r: z[m_].mean().item() for r, m_ in s56.roles(task, pinp).items()}


def run(p):
    """S56's run path with this screen's models; beta^R in the records is the latch's z (firing rate)."""
    arm = p["arm"]
    alias = {"BASELINE_L": "BASELINE", "ORACLE_L": "ORACLE_H"}
    if arm in alias:
        rec = s56.run(dict(p, arm=alias[arm], res_layers=[2, 3]))
        rec["arm"] = arm
        return rec
    if arm not in ARMS:
        return s56.run(dict(p, res_layers=[2, 3]))
    s56.ARM[arm] = ("reg", {})
    orig, orig_b = s56.make_model, s56.beta_by_role
    s56.make_model = lambda a_, seed, task=None, res_layers=None: (make(a_, seed) if a_ in ARMS else orig(a_, seed))
    s56.beta_by_role = z_by_role
    try:
        return s56.run(dict(p, res_layers=[2, 3]))
    finally:
        s56.make_model, s56.beta_by_role = orig, orig_b


def checks(p):
    rows = [(f"S56: {n}", v) for n, v in s56.checks(dict(res_layers=[2, 3]))]
    x = s56.H8.make_batch(16, torch.Generator().manual_seed(4))[0][:, :-1]
    m = make("LATCH", 410)
    m.eval()
    with torch.no_grad():
        u = m.embed(x) @ m.w_R + m.b_R
    rows.append((f"dead-STE guard at step 0: max |u| = {float(u.abs().max()):.2e} < 1 (b_z = 0, w_z small); a = {a_at(0)}",
                 float(u.abs().max()) < 1 and float(m.b_R) == 0.0))
    rows.append((f"a schedule: a(1) = {a_at(1)}, a(8000) = {a_at(8000)}, a(16000) = {a_at(16000)}, a(24000) = {a_at(24000)}",
                 a_at(1) == 1.0 and a_at(8000) == 1.0 and abs(a_at(16000) - 3.0) < 1e-12 and a_at(24000) == 5.0))
    # STE: forward is binary; the gradient is a/2 inside the linear region and 0 outside
    uu = torch.tensor([-2.0, -0.2, 0.1, 3.0], requires_grad=True)
    zt = torch.clamp((2.0 * uu + 1) / 2, 0, 1)
    z = _STE.apply(zt)
    z.sum().backward()
    rows.append((f"STE: z = {z.tolist()} binary; dz/du = {uu.grad.tolist()} (a/2 = 1 inside, 0 outside)",
                 z.tolist() == [0.0, 0.0, 1.0, 1.0] and uu.grad.tolist() == [0.0, 1.0, 1.0, 0.0]))
    # the latch holds: r_t = r_{t-1} where z_t = 0, = W_r v_t where z_t = 1; the gate at t reads r_{t-1}
    with torch.no_grad():
        v = m.embed(x)
        rp, zz = m.register(v)
        w = v @ m.W_R.T
        ok = True
        for t in range(1, x.shape[1] - 1):
            want = torch.where(zz[:, t:t + 1] > 0.5, w[:, t], rp[:, t])
            ok &= torch.equal(rp[:, t + 1], want)
    rows.append(("latch recurrence: Rprev_{t+1} = W_r v_t where z_t = 1, Rprev_t where z_t = 0 (the gate at t reads r_{t-1})", ok))
    # causality: token t's latch write changes nothing at <= t
    t = 20
    with torch.no_grad():
        g0 = s56.all_gates(m, x)[0]
        orig = m.register

        def pert(v):
            v2 = v.clone(); v2[:, t] = v2[:, t] + 1.0
            return orig(v2)
        m.register = pert
        try:
            g1 = s56.all_gates(m, x)[0]
        finally:
            del m.register
    rows.append((f"latch causality: gates at <= {t} unchanged when token {t}'s latch input changes",
                 all(torch.equal(g0[l - 1][:, :t + 1], g1[l - 1][:, :t + 1]) for l in m.gated)))
    # the prior enters the loss on training forwards (gradient on w_R from the prior alone), not in eval
    mp = make("LATCH_PRIOR", 410)
    mp.train()
    out = mp(x)[0]
    gz = torch.autograd.grad(out.sum() * 0.0, mp.w_R, allow_unused=True)[0]
    mp.eval()
    rows.append((f"prior: on a training forward a zero loss on the logits still sends the prior's gradient to w_z "
                 f"(|grad| {float(gz.abs().max()) if gz is not None else 0:.2e} > 0); update counter {mp.n_upd} = 1",
                 gz is not None and float(gz.abs().max()) > 0 and mp.n_upd == 1))
    # same parameters as S56's REG for the seed except b_R; nudge moves only GATE_NEW
    r = s56.make_model("REG", 410, res_layers=[2, 3])
    pr = dict(r.named_parameters())
    rows.append(("LATCH shares every initial parameter with S56's REG for the seed except b_R (0 vs logit 0.1)",
                 all(torch.equal(q, pr[n]) for n, q in m.named_parameters() if n != "b_R")))
    mn = make("LATCH_NUDGE", 411)
    m0 = make("LATCH", 411)
    moved = {n for n, q in mn.named_parameters() if not torch.equal(q, dict(m0.named_parameters())[n])}
    rows.append((f"LATCH_NUDGE: the nudge moves only {sorted(moved)}; the update counter starts at {mn.n_upd} after it",
                 moved <= set(rm.GATE_NEW) and mn.n_upd == 0))
    # ORACLE23: layers 2-3 get the perfect gate exactly, layer 1 LOCAL3's
    mo = make("ORACLE23", 370)
    with torch.no_grad():
        gs = s56.all_gates(mo, x)[0]
        pg = perfect_gate_general(x, mo.ctx_tokens)
        l1 = mo.local_gates(mo.embed(x))[0]
    rows.append(("ORACLE23: layers 2 and 3 = the perfect gate, layer 1 = LOCAL3's gate",
                 torch.equal(gs[1], pg) and torch.equal(gs[2], pg) and torch.equal(gs[0], l1)))
    return [[n, bool(v)] for n, v in rows]
