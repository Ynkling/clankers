#!/usr/bin/env python
"""
explore_g_s91_child.py — EXPLORATORY, not a result. Run code of Session G's screen S91 (explore_g_s91): gates that compare
the current context with what the sequence has already shown, on RandHeaderTask-allpool. Runs INSIDE a child process on
this branch's own modules, 1 torch thread. It imports S89's child (explore_g_s89_child: the tasks, the repeat hinge, the
task-aware statistics, the delta model) and S89H's child (explore_g_s89h_child: the Hebbian + conv model) unchanged; the
control and validity arms run through their run() unchanged.

ARMS (memory "_H": Hebbian + conv 'layer' as S89H, 48000; "_D": delta as S89, 24000; SLOW; the REG3 model):
  REG3_REPEATHINGE_<M>   S89H's / S89's arm, through its own run() (the control).
  PERFECT_H / ORACLE_D   the perfect gate (task.perfect_gate), S89H's / S89's (validity).
  MATCHGATE_<M>          REG3_REPEATHINGE whose gate input gains one scalar per layer, the match
                             m_t = cos(W_R v_t, R_{t-1}),
                         the cosine between the current token's embedding as the register would write it (W_R v_t, the
                         register's own coordinates: the register is 16-dimensional, the embedding 32) and the register's
                         content before token t's write (R_{t-1}, what it holds since its last write; R_{-1} = 0 gives
                         m_0 = 0). It enters the shared gate as u^(l)_t += m_t a_M at every layer (the gate is shared, so
                         m_t and a_M are the same at each layer), a_M in R^{16} (the gate's hidden width), initialised
                         to 0 (no generator is drawn: every other parameter is REG3_REPEATHINGE's), in the gate group
                         (1e-3). The register's write is REG3's; the hinge is REPEATHINGE's. Nothing else changes.
                         The match reads the embedding and the register only (no label, no index, no position).
  MATCHGATE_NOHINGE_<M>  MATCHGATE without the hinge.
  NOVELTY_<M>            REG3_REPEATHINGE whose register write gate also receives one scalar, the novelty
                             n_t = 1 if the token type at t occurred at an earlier position of the sequence, else 0
                         (the repeat statistic, causal, per position, from the token ids only):
                             beta_t = sigmoid(w_R . v_t + b_R + w_N n_t),
                         w_N a scalar initialised to 0, in the gate group. Nothing else changes.
  With its scalar zeroed (m_t = 0, or n_t = 0), each new arm IS REG3_REPEATHINGE (MATCHGATE_NOHINGE: REG3) bit for bit
  (CHECKed through 2400): the class dispatch, the extra zero parameter and the group membership change nothing.
EXTRA RECORD (end of run, on run_one's statistics probe): type_map = eta^2 of the returned gate's channel-0 probability at
  the body's KEY positions grouped by the block's CTX TYPE (the latched context token id), pooled over sequences; ~1 for a
  fixed type -> channel map, ~0 for the relative routing (the stream index by first appearance is independent of the type).
"""

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import explore_common as ec
import explore_aux_gate as aux
import explore_g_common as gc
import explore_g_rhl_child as rc
import explore_g_devices_child as dc
import explore_g_register_child as s56
import explore_g_regmodel as rm
import explore_g_s87_child as s87
import explore_g_s89_child as c89
import explore_g_s89h_child as h89
import test_binding_onset as tbo
import test_short_conv as tsc
from test_multilayer_binding import probe_batch
from test_router_layout import eta2
from test_router_reliability import run_one
from test_instrument_v2 import TAU_END

WINDOW, LAM = c89.WINDOW, c89.LAM
NEW = ("a_M", "w_N")


def match_scalar(v, Rprev, W_R):
    """m_t = cos(W_R v_t, R_{t-1}): (B, T). Reads the embedding and the register only."""
    w = v @ W_R.T
    return F.cosine_similarity(w, Rprev, dim=-1, eps=1e-8)


def novelty(tokens):
    """n_t = 1 if tokens[b, t] occurred at an earlier position of row b: (B, T) float. Token ids only, causal."""
    assert isinstance(tokens, torch.Tensor) and tokens.dtype == torch.long and tokens.dim() == 2, "token ids only"
    eq = tokens.unsqueeze(2) == tokens.unsqueeze(1)                       # (B, T, T): [b, t, s]
    earlier = torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool).tril(-1)
    return (eq & earlier).any(-1).to(torch.float32)


class CompareBDH(h89.Reg3Conv):
    """Reg3Conv (conv 'layer' or none; Hebbian or delta) + the match term in the gate input (match = True) and/or the
    novelty term in the register's write (novelty = True), and the repeat hinge (TermConv's forward, with the gate
    dispatched through this class so the match term is inside the hinge's window too)."""

    # ── the two scalars ──
    def layer_gate(self, level, h, v, Rprev):
        x_in = self.gate_conv(v)
        u = x_in @ self.A_v.T
        if level in self.res_layers:
            r = h.detach() if self.sg else h
            u = u + F.layer_norm(r, (r.shape[-1],)) @ self.A_r.T
        if Rprev is not None:
            u = u + Rprev @ self.A_R.T
        if self.match:
            m = torch.zeros(v.shape[:-1], dtype=v.dtype) if self.zero else match_scalar(v, Rprev, self.W_R)
            u = u + m.unsqueeze(-1) * self.a_M
        return F.softmax(torch.tanh(u) @ self.W_g2.T, dim=-1)

    def register(self, v):
        if not self.novelty:
            return rm.RegBDH.register(self, v)
        n = torch.zeros(v.shape[:-1], dtype=v.dtype) if self.zero else novelty(self._tok).to(v.dtype)
        beta = torch.sigmoid(v @ self.w_R + self.b_R + self.w_N * n)
        w = v @ self.W_R.T
        B, T, _ = v.shape
        R = v.new_zeros(B, rm.R_DIM)
        prev = []
        for t in range(T):
            prev.append(R)
            b = beta[:, t:t + 1]
            R = R * (1 - b) + b * w[:, t]
        return torch.stack(prev, 1), beta

    # ── the forward: TermConv's, with the class's own layer_gate inside the hinge's wrapper ──
    def forward(self, tokens, tau=None, track_sat=False):
        self._tok = tokens
        st = self.t_state
        training = self.training and torch.is_grad_enabled() and st["count"]
        if training:
            st["n"] += 1
        u = st["n"]
        on = self.t_on and u <= WINDOW
        live = []
        prev = self.__dict__.get("layer_gate")
        if training and on:
            cls = type(self)
            inner = prev if prev is not None else (lambda level, h, v, Rprev: cls.layer_gate(self, level, h, v, Rprev))

            def lg(level, h, v, Rprev):
                g = inner(level, h, v, Rprev)
                live.append(g)
                return g
            self.layer_gate = lg
        try:
            out = h89.Reg3Conv.forward(self, tokens, tau, track_sat)
        finally:
            if prev is None:
                self.__dict__.pop("layer_gate", None)
            else:
                self.layer_gate = prev
        if training and u in (1, WINDOW - 1, WINDOW, WINDOW + 1, WINDOW + 2):
            st["weights"][str(u)] = 1.0 if on else 0.0
        if not live:
            return out
        v = c89.term_value(self.t_kind, live, tokens, self.t_task)
        if u in c89.LOG_AT:
            st["value_at"][str(u)] = [round(float(x), 6) for x in v.detach()]
        if bool((v > 0).any()):
            st["fired"] += 1
        return (aux._Inject.apply(out[0], LAM * v.mean()),) + tuple(out[1:])


# arm -> (kind, match, novelty, hinge)
NEWARMS = {"MATCHGATE": (True, False, True), "MATCHGATE_NOHINGE": (True, False, False), "NOVELTY": (False, True, True)}


def parse(arm):
    """'<task>.<NAME>_<M>' -> (task, name, memory 'H' | 'D')."""
    tname, rest = arm.split(".", 1)
    name, mem = rest.rsplit("_", 1)
    return tname, name, mem


def make_model(arm, seed, zero=False):
    tname, name, mem = parse(arm)
    match, nov, hinge = NEWARMS[name]
    m = h89.make_model("REG3_H", seed, tname) if mem == "H" else c89.make_model("REG3_D", seed, tname)
    if mem == "D":
        assert type(m) is rc.Reg3BDH and m.conv is None
    m.__class__ = CompareBDH
    m.match, m.novelty, m.zero = match, nov, bool(zero)
    hg = m.W_in.shape[0]
    if match:
        m.a_M = nn.Parameter(torch.zeros(hg))
    if nov:
        m.w_N = nn.Parameter(torch.zeros(()))
    m.t_kind, m.t_on, m.t_task = "repeat", hinge, tname
    m.t_state = dict(n=0, fired=0, weights={}, value_at={}, count=True)
    return m


def groups(holder, gate_lr):
    """explore_g_devices_child.groups with the new scalars' weights in the gate group."""
    def param_groups(model):
        named = [(n, q) for n, q in model.named_parameters() if q.requires_grad]
        nm = dict(named)
        gate = rm.gate_names(model) + [n for n in NEW if n in nm]
        rest = [n for n, _ in named if n not in gate]
        holder["names"] = [gate, rest]
        holder["groups"] = [dict(params=[nm[n] for n in gate], lr=gate_lr), dict(params=[nm[n] for n in rest], lr=dc.LR_SLOW)]
        return holder["groups"]
    return param_groups


@torch.no_grad()
def type_map(model, task, probe):
    pinp = probe[0]
    was = model.training
    model.eval()
    try:
        g = model(pinp, TAU_END)[2][..., 0]
    finally:
        model.train(was)
    kpos, _ = task.body_positions(pinp)
    B, T = pinp.shape
    idx = torch.arange(T).expand(B, T)
    last = torch.where(task.ctx_mask(pinp), idx, torch.full_like(idx, -1)).cummax(dim=1).values
    ctype = pinp.gather(1, last.clamp(min=0))
    return eta2(g.gather(1, kpos), ctype.gather(1, kpos))


BASE = {"REG3_REPEATHINGE": ("repeat", "REG3_REPEATHINGE"), "REG3": ("reg3", "REG3"), "PERFECT": ("oracle", "PERFECT"),
        "ORACLE": ("oracle", "ORACLE")}


def build_any(arm, seed, zero=False):
    tname, name, mem = parse(arm)
    if name in NEWARMS:
        return make_model(arm, seed, zero=zero)
    return h89.make_model(f"{BASE[name][1]}_H", seed, tname) if mem == "H" else c89.make_model(f"{BASE[name][1]}_D", seed, tname)


def run(p):
    """p: arm '<task>.<NAME>_<H|D>', seed, iters, zero, eval_every. One path for every arm: the controls and the perfect
    gates are built by S89H's / S89's make_model and trained with the same recipe and statistics as their run()
    (CHECKed: S89H's and S89's REG3_REPEATHINGE|2842 reproduce through this run())."""
    arm, seed = p["arm"], p["seed"]
    tname, name, mem = parse(arm)
    task = c89.task_of(tname)
    kind = "oracle" if name in ("PERFECT", "ORACLE") else "reg3"
    a = dict(h89._arm(kind, task, mem == "H"), key=arm)
    holder, keep = {}, {"at": -1}
    rec = run_one(a, seed, p.get("iters", 48000 if mem == "H" else 24000), p.get("eval_every", tbo.EVAL_EVERY),
                  task=task, stats_fn=c89.stats_fn89, grad_fn=tsc.conv_grad_norms, lr=dc.LR,
                  builder=lambda a_, s_: (lambda: build_any(arm, s_, zero=p.get("zero", False))),
                  run_kw=dict(param_groups=groups(holder, dc.LR)), check=dc.switch(holder, 0), keep=keep)
    if not rec.get("ok"):
        return json.loads(json.dumps(rec, default=float))
    mdl = keep["model"]
    ev = tbo.eval_batch(task)
    rec.update(arm=arm, task=tname, recipe="slow", conv=mdl.conv, group_names=holder.get("names"),
               stream_acc=c89.stream_acc89(mdl, task, ev), probe=c89.probe_rh89(mdl, task, gc.probe_data(task)),
               acc_by_cell=c89.acc_by_cell89(mdl, task, ev), write_order=c89.write_order_acc89(mdl, task, ev),
               type_map=type_map(mdl, task, probe_batch(task, seed)),
               new_params={n: [round(float(x), 5) for x in dict(mdl.named_parameters())[n].detach().reshape(-1)]
                           for n in NEW if hasattr(mdl, n)})
    if getattr(mdl, "t_state", None) is not None:
        rec["term"] = {k: v for k, v in mdl.t_state.items() if k != "count"}
    if kind != "oracle":
        rec["fail"] = None if rec["transition"] is not None else ec.fail_class(rec)
    return json.loads(json.dumps(rec, default=float))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def checks_static(p):
    rows = []
    t = c89.task_of("allpool")
    x = t.make_batch(16, torch.Generator().manual_seed(5))[0][:, :-1]
    # the scalars by hand; the match reads the embedding and the register only; novelty reads token ids only
    m = make_model("allpool.MATCHGATE_H", 2962)
    with torch.no_grad():
        v = m.embed(x)
        Rprev, _ = m.register(v)
        ms = match_scalar(v, Rprev, m.W_R)
        hand = torch.stack([torch.stack([F.cosine_similarity(v[b, s] @ m.W_R.T, Rprev[b, s], dim=0, eps=1e-8)
                                         for s in range(x.shape[1])]) for b in range(4)])
    rows.append((f"the match by hand on 4 sequences (max |diff| {float((ms[:4] - hand).abs().max()):.1e}); m_0 = 0 "
                 f"({bool((ms[:, 0] == 0).all())}); range [{float(ms.min()):.3f}, {float(ms.max()):.3f}]",
                 float((ms[:4] - hand).abs().max()) < 1e-6 and bool((ms[:, 0] == 0).all())))
    import inspect
    rows.append((f"match_scalar(v, Rprev, W_R) and novelty(tokens) signatures "
                 f"{list(inspect.signature(match_scalar).parameters)}, {list(inspect.signature(novelty).parameters)}",
                 list(inspect.signature(match_scalar).parameters) == ["v", "Rprev", "W_R"]
                 and list(inspect.signature(novelty).parameters) == ["tokens"]))
    xr = torch.tensor([[3, 5, 3, 7, 5, 5]])
    try:
        novelty(xr.float())
        refused = False
    except AssertionError:
        refused = True
    rows.append((f"novelty by hand {novelty(xr).tolist()} (earlier occurrence); it refuses a float tensor ({refused})",
                 novelty(xr).tolist() == [[0, 0, 1, 0, 1, 1]] and refused))
    # causality of the match and the gate: token t changed -> nothing at positions < t changes
    x2 = x.clone()
    x2[:, 12] = (x2[:, 12] + 1) % t.vocab
    mm = make_model("allpool.MATCHGATE_H", 2962)
    nn_ = make_model("allpool.NOVELTY_H", 2962)
    with torch.no_grad():
        mm.a_M.normal_(0, 1.0, generator=torch.Generator().manual_seed(3))
        nn_.w_N.fill_(2.0)
        ok = True
        for mod in (mm, nn_):
            mod.eval()
            l1, _, g1, _ = mod(x)
            l2, _, g2, _ = mod(x2)
            ok &= torch.equal(l1[:, :12], l2[:, :12]) and torch.equal(g1[:, :12], g2[:, :12])
    rows.append(("MATCHGATE and NOVELTY (scalars' weights non-zero) are strictly causal: token 12 changed, no logit or gate "
                 "at positions < 12 changes", bool(ok)))
    # at initialisation (weights 0) each new arm equals its base forward bit for bit; the parameters are the base's + one
    for arm, base in (("allpool.MATCHGATE_H", h89.make_model("REG3_REPEATHINGE_H", 2962, "allpool")),
                      ("allpool.NOVELTY_H", h89.make_model("REG3_REPEATHINGE_H", 2962, "allpool")),
                      ("allpool.MATCHGATE_D", c89.make_model("REG3_REPEATHINGE_D", 2962, "allpool"))):
        mdl = make_model(arm, 2962).eval()
        base.eval()
        pa, pb = dict(mdl.named_parameters()), dict(base.named_parameters())
        extra = sorted(set(pa) - set(pb))
        same = all(torch.equal(pa[n], pb[n]) for n in pb)
        with torch.no_grad():
            eq = torch.equal(mdl(x)[0], base(x)[0])
        rows.append((f"{arm}: parameters = the control's + {extra} (shared equal: {same}); at initialisation the logits equal "
                     f"the control's bit for bit ({eq})", same and eq and len(extra) == 1))
    # the new weights receive a gradient; they are in the gate group
    for arm, n in (("allpool.MATCHGATE_H", "a_M"), ("allpool.NOVELTY_H", "w_N")):
        mdl = make_model(arm, 2962)
        mdl.train()
        mdl.t_state["count"] = False
        out = mdl(x)[0]
        g_ = torch.autograd.grad(out.square().mean(), [getattr(mdl, n)])[0]
        holder = {}
        groups(holder, dc.LR)(mdl)
        rows.append((f"{arm}: {n} receives a gradient (max |g| {float(g_.abs().max()):.2e}) and is in the gate group "
                     f"({n in holder['names'][0]})", float(g_.abs().max()) > 0 and n in holder["names"][0]))
    # the perfect gates read ROUTED (within-sequence) on the untrained model
    for arm, mod in (("PERFECT_H", h89), ("ORACLE_D", c89)):
        mdl = mod.make_model(arm, 2960, "allpool")
        st = c89.stats_fn89(mdl, t, probe_batch(t, 2960), "end")
        rows.append((f"allpool: the untrained {arm} reads ROUTED* (within-sequence eta^2 {st['eta_key_by_stream_inv']:.3f}, "
                     f"margin {st['margin']:.3f})", st["eta_key_by_stream_inv"] > 0.999 and st["margin"] >= 0.9))
    # the type-map statistic on synthetic gates: the relative perfect gate ~0, the balanced type-level gate 1
    import explore_f_tasks as ft
    pr = probe_batch(t, 2960)

    class Stub(torch.nn.Module):
        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def forward(self, tokens, tau=None):
            g = self.fn(tokens)
            return None, None, g, g
    rel = type_map(Stub(lambda z: t.perfect_gate(z)), t, pr)
    typ = type_map(Stub(lambda z: ft.type_level_gate(t, z)), t, pr)
    rows.append((f"type_map: the relative perfect gate {rel:.3f} (~0), the balanced type-level gate {typ:.3f} (1)",
                 rel < 0.1 and typ > 0.999))
    return rows


def checks_zero(p):
    """With its scalar zeroed, each new arm equals its control bit for bit through 2400 (curve and statistics)."""
    mem = p["mem"]
    seed = 2962
    rows = []
    ctrl = run(dict(arm=f"allpool.REG3_REPEATHINGE_{mem}", seed=seed, iters=2400))
    reg = run(dict(arm=f"allpool.REG3_{mem}", seed=seed, iters=2400))
    for name, ref, rn in (("MATCHGATE", ctrl, "REG3_REPEATHINGE"), ("NOVELTY", ctrl, "REG3_REPEATHINGE"),
                          ("MATCHGATE_NOHINGE", reg, "REG3")):
        r = run(dict(arm=f"allpool.{name}_{mem}", seed=seed, iters=2400, zero=True))
        rows.append([f"allpool, {mem}: {name} with its scalar zeroed equals {rn}|{seed} through 2400 bit for bit (curve "
                     f"{r['curve']})", s87.trunc(r) == s87.trunc(ref)])
    return rows


def checks_repro(p):
    """S89H's allpool.REG3_REPEATHINGE_H|2842 (and S89's allpool.REG3_REPEATHINGE_D|2842) through this run()."""
    mem = p["mem"]
    store = "s89h_results.json" if mem == "H" else "s89_results.json"
    ref, cpu = c89.record(store, f"allpool.REG3_REPEATHINGE_{mem}", 2842)
    r = run(dict(arm=f"allpool.REG3_REPEATHINGE_{mem}", seed=2842, iters=2400))
    return [[f"{store[:-13].upper()}'s allpool.REG3_REPEATHINGE_{mem}|2842 (on {cpu}) reproduces bit for bit through 2400 "
             f"through S91's run(): curve {r['curve']}", s87.trunc(r) == s87.trunc(ref)]]


def checks_on(p):
    mem = p["mem"]
    rows = []
    for name in ("REG3_REPEATHINGE", "MATCHGATE", "MATCHGATE_NOHINGE", "NOVELTY"):
        r = run(dict(arm=f"allpool.{name}_{mem}", seed=2962, iters=20, eval_every=10))
        e = r.get("end", {})
        fired = r.get("term", {}).get("fired")
        rows.append([f"allpool, {mem}: {name} runs 20 updates end to end (ok {r.get('ok')}; type_map {r.get('type_map')}; "
                     f"term fired {fired}; new params {sorted((r.get('new_params') or {}).keys())})",
                     bool(r.get("ok")) and "eta_key_by_stream_inv" in e and r.get("type_map") is not None
                     and (name == "MATCHGATE_NOHINGE" or fired is None or fired > 0)])
    return rows
