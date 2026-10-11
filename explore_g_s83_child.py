#!/usr/bin/env python
"""
explore_g_s83_child.py — EXPLORATORY, not a result. Run code of Session G's screen S83 (explore_g_s83): what the key-term
hinge must read, whether the register is needed, and how long the window must be. Runs INSIDE a child process on this
branch's own modules, 1 torch thread. It imports S82's child (explore_g_devices_child) and S71-S73's child
(explore_g_rhl_child) unchanged.

TASK, MEMORY, RECIPE: RandHeaderTask-lite exactly as F's S70; Hebbian (delta: beta 1, L2 keys, tied write, decay 0.95);
k = 2, no conv, 24000 updates; SLOW (S56's groups: gate group LOCAL3's W_in, W_g, window + every new gate / register
parameter at 1e-3; the rest at 1e-4 for updates 1-2400, 1e-3 after).

THE TERMS (each enters the loss on training forwards of updates 1..W only, through explore_aux_gate._Inject; the update
count is the number of training forwards; outside the window the term is not computed):
  key   (S82's REG3_KEYHINGE; S52's constants)  LAM x mean_l relu(eta2_key(g^(l)) - TAU), LAM 1.0, TAU 0.2, where eta2_key =
        explore_b18_gates.eta2_pool of layer l's gate at the body's KEY positions grouped by the key id there.
        READS: key positions and key identities (token types and which key each is).
  pos   LAM_P x mean_l mean_{sequences, body KEY positions t} KL(g^(l)_t || uniform), LAM_P 1.0, no threshold;
        KL(g || U) = sum_c g_c log(k g_c). READS: key positions (which tokens are keys), not which key.
  flat  the same KL term at every BODY position t < body_len (CTX, KEY and VAL tokens of the body; the query triple
        excluded): body_len is a fixed index range, the same in every sequence. READS: no token identity or type.
  CAVEAT fixed now: eta2_key is scale-free (a ratio of variances), so the key term pushes even while the gate is near
  uniform (eta2_key 0.89 at initialisation in S82); the KL terms are absolute and their gradient vanishes at the uniform
  gate (KL is quadratic there). A null result for pos / flat may be a weakness of this form, not of "reading positions".
ARMS (seeds 2100-2119; S82's BASELINE, REG3, REG3_KEYHINGE records are the paired references)
  LOCAL3_KEYHINGE    LOCAL3 + SLOW (S82's BASELINE) + the key term on the window gate (one gate for all layers), W 2400
  REG3_KEYHINGE_POS  REG3 + the pos term, W 2400
  REG3_FLAT          REG3 + the flat term, W 2400
  REG3_KEYHINGE_1200 REG3 + the key term, W 1200
  REG3_NUDGE         S82's (2100-2101; validity of the code path)
  <ARM>_D            delta memory (rule in the parent)
  *_RERUN            S82's arm rerun under the current code (only if its S82 record does not reproduce through 2400)
Records: explore_g_rhl_child's (statistics, layer statistics with eta^2 by key per layer, in-run probe, per-cell
accuracy, write order) plus the term's firings (key: relu > 0; pos / flat: every update in the window), its value at
updates 1, 300, 600, 1200, 2400 and its weight at 1, W - 1, W, W + 1, W + 2.
"""

import json
import os

import torch
import torch.nn.functional as F

import explore_common as ec
import explore_aux_gate as aux
import explore_b16_gates as g16
import explore_f_tasks as ft
import explore_g_common as gc
import explore_g_regmodel as rm
import explore_g_register_child as s56
import explore_g_rhl_child as rc
import explore_g_devices_child as dc
import test_binding_onset as tbo
import test_short_conv as tsc
from test_router_reliability import run_one

LAM, TAU, LAM_P = dc.LAM, dc.TAU, 1.0
RHL = rc.RHL
LOG_AT = (1, 300, 600, 1200, 2400)

# arm -> (base, term, window, memory)
ARMS = {
    "LOCAL3_KEYHINGE": ("local", "key", 2400, "hebb"),
    "REG3_KEYHINGE_POS": ("reg3", "pos", 2400, "hebb"),
    "REG3_FLAT": ("reg3", "flat", 2400, "hebb"),
    "REG3_KEYHINGE_1200": ("reg3", "key", 1200, "hebb"),
    "REG3_NUDGE": ("rc", None, 0, "REG3_NUDGE"),
    "LOCAL3_KEYHINGE_D": ("local", "key", 2400, "delta"),
    "REG3_KEYHINGE_POS_D": ("reg3", "pos", 2400, "delta"),
    "REG3_FLAT_D": ("reg3", "flat", 2400, "delta"),
    "BASELINE_RERUN": ("dc", None, 0, "BASELINE"),
    "REG3_RERUN": ("dc", None, 0, "REG3"),
    "REG3_KEYHINGE_RERUN": ("dc", None, 0, "REG3_KEYHINGE"),
}


def term_value(kind, gates, tokens, task):
    """The term (before LAM) per layer: key -> relu(eta2_key - TAU); pos / flat -> mean KL(g || uniform)."""
    if kind == "key":
        return F.relu(dc.key_eta(gates, tokens, task) - TAU)
    k = gates[0].shape[-1]
    if kind == "pos":
        kpos, _ = ft.body_positions(task, tokens)
        sel = [g.gather(1, kpos.unsqueeze(-1).expand(-1, -1, k)) for g in gates]
    else:
        sel = [g[:, :task.body_len] for g in gates]
    return torch.stack([(s * torch.log(s * k)).sum(-1).mean() for s in sel])


def _state():
    return dict(n=0, fired=0, windows=[], weights={}, value_at={}, count=True)


def _term_step(model, out, gates, tokens):
    st = model.term_state
    u = st["n"]
    v = term_value(model.term_kind, gates, tokens, model.term_task)
    if u in LOG_AT:
        st["value_at"][str(u)] = [round(float(x), 6) for x in v.detach()]
    win = (u - 1) // tbo.EVAL_EVERY
    while len(st["windows"]) <= win:
        st["windows"].append(0)
    if bool((v > 0).any()):
        st["fired"] += 1
        st["windows"][win] += 1
    lam = LAM if model.term_kind == "key" else LAM_P
    return (aux._Inject.apply(out[0], lam * v.mean()),) + tuple(out[1:])


def _weight(model):
    st = model.term_state
    st["n"] += 1
    u = st["n"]
    w = model.term_on if u <= model.term_window else 0.0
    if u in (1, model.term_window - 1, model.term_window, model.term_window + 1, model.term_window + 2):
        st["weights"][str(u)] = w
    return w


class TermReg3(rc.Reg3BDH):
    def forward(self, tokens, tau=None, track_sat=False):
        active = self.training and torch.is_grad_enabled() and self.term_state["count"]
        live = []
        prev = self.__dict__.get("layer_gate")                 # a wrapper set from outside (S56's live_gates) is kept
        if active and _weight(self) != 0.0:
            inner = prev if prev is not None else (lambda level, h, v, Rprev: rc.Reg3BDH.layer_gate(self, level, h, v, Rprev))

            def lg(level, h, v, Rprev):
                g = inner(level, h, v, Rprev)
                live.append(g)
                return g
            self.layer_gate = lg
        try:
            out = rm.RegBDH.forward(self, tokens, tau, track_sat)
        finally:
            if prev is None:
                self.__dict__.pop("layer_gate", None)
            else:
                self.layer_gate = prev
        return _term_step(self, out, live, tokens) if live else out


class TermLocal(g16.LocalBDH):
    def forward(self, tokens, tau=None, track_sat=False):
        out = g16.LocalBDH.forward(self, tokens, tau, track_sat)
        active = self.training and torch.is_grad_enabled() and self.term_state["count"]
        if active and _weight(self) != 0.0:
            return _term_step(self, out, [out[2]], tokens)
        return out


def make_model(arm, seed, task=RHL, on=True):
    base, kind, window, mem = ARMS[arm]
    if base == "rc":
        return rc.make_model(mem, seed, task)
    if base == "dc":
        return dc.make_model(mem, seed, task)
    if base == "local":
        m = rc.make_model("BASELINE_D" if mem == "delta" else "BASELINE", seed, task)
        if mem == "delta":
            m.__class__ = type("DeltaTermLocal", (TermLocal, type(m)), {})
        else:
            m.__class__ = TermLocal
    else:
        m = rc.make_model("REG3_D" if mem == "delta" else "REG3", seed, task)
        m.__class__ = TermReg3
    m.term_kind, m.term_window, m.term_task, m.term_on = kind, int(window), task, 1.0 if on else 0.0
    m.term_state = _state()
    return m


def run(p):
    arm, seed = p["arm"], p["seed"]
    base, kind, window, mem = ARMS[arm]
    if base == "dc":
        r = dc.run(dict(p, arm=mem))
        r["arm"] = arm
        return r
    inner = mem if base == "rc" else ("BASELINE" if base == "local" else "REG3") + ("_D" if mem == "delta" else "")
    a = dict(ec.ARM_A, key=arm)
    holder = {}
    kw = dict(param_groups=dc.groups(holder, dc.LR))
    check = dc.switch(holder, 0)
    keep = {"at": -1}
    rec = run_one(a, seed, p.get("iters", rc.ITERS), task=RHL, stats_fn=rc.stats_fn, grad_fn=tsc.conv_grad_norms, lr=dc.LR,
                  builder=lambda a_, s_: (lambda: make_model(arm, s_, on=not p.get("off"))), run_kw=kw, check=check,
                  keep=keep)
    if not rec.get("ok"):
        return json.loads(json.dumps(rec, default=float))
    m = keep["model"]
    ev = tbo.eval_batch(RHL)
    rec.update(arm=arm, recipe="slow", group_names=holder.get("names"), stream_acc=gc.stream_acc(m, RHL, ev),
               probe=rc.probe_rh(m, RHL, gc.probe_data(RHL)), acc_by_cell=rc.acc_by_cell(m, RHL, ev),
               write_order=ft.write_order_acc(m, RHL, ev))
    if hasattr(m, "term_state"):
        rec["term"] = {k: v for k, v in m.term_state.items() if k != "count"}
    rec["fail"] = None if rec["transition"] is not None else ec.fail_class(rec)
    return json.loads(json.dumps(rec, default=float))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def s82_record(arm, seed):
    with open(os.path.join(gc.G_DIR, "s82_devices_results.json")) as f:
        st = json.load(f)
    sha = st["meta"]["sha"][:12]
    return st["runs"][f"{arm}|{seed}|{sha}|{ec.cpu_model()}"]


def trunc(r, upto=2400):
    return dict(curve=[c for c in r["curve"] if c[0] <= upto],
                stats=[s for s in r["stats"] if isinstance(s["step"], int) and s["step"] <= upto])


def reproduce(p):
    """S82's BASELINE, REG3, REG3_KEYHINGE (and REG3_KEYHINGE_D, BASELINE_D, REG3_D) at seed 2100 through 2400 under the
    current code: curve and every statistic at steps 0, 1200, 2400 equal S82's record."""
    out = {}
    for arm in p.get("arms", ("BASELINE", "REG3", "REG3_KEYHINGE")):
        r = dc.run(dict(arm=arm, seed=2100, iters=2400))
        out[arm] = trunc(r) == trunc(s82_record(arm, 2100))
    return out


def checks(p):
    rows = [(f"S82: {n}", v) for n, v in dc.checks({})]
    x = RHL.make_batch(32, torch.Generator().manual_seed(5))[0][:, :-1]
    seed = 2100
    # (1) each new term with its switch off equals REG3 (S82's record) bit for bit through 2400; LOCAL3_KEYHINGE off
    #     equals BASELINE (S82's record)
    for arm, ref in (("REG3_KEYHINGE_POS", "REG3"), ("REG3_FLAT", "REG3"), ("REG3_KEYHINGE_1200", "REG3"),
                     ("LOCAL3_KEYHINGE", "BASELINE")):
        r = run(dict(arm=arm, seed=seed, iters=2400, off=True))
        rows.append((f"{arm} with its term off equals S82's {ref}|{seed} through 2400 bit for bit (curve {r['curve']})",
                     trunc(r) == trunc(s82_record(ref, seed))))
    # (2) the key term at W 2400 in TermReg3 equals S82's REG3_KEYHINGE bit for bit through 2400 (the same term, new class)
    def mk2400(a_, s_):
        def make():
            m = make_model("REG3_KEYHINGE_1200", s_)
            m.term_window = 2400
            return m
        return make
    hold = {}
    r = run_one(dict(ec.ARM_A, key="T"), seed, 2400, task=RHL, stats_fn=rc.stats_fn, grad_fn=tsc.conv_grad_norms, lr=dc.LR,
                builder=mk2400, run_kw=dict(param_groups=dc.groups(hold, dc.LR)), check=dc.switch(hold, 0), keep={"at": -1})
    rows.append((f"the key term in S83's class at W 2400 equals S82's REG3_KEYHINGE|{seed} through 2400 bit for bit "
                 f"(curve {r['curve']})", trunc(r) == trunc(s82_record("REG3_KEYHINGE", seed))))
    # (3) gradients of each term: pos / flat (and key) reach the gate's parameters (W_g2, A_*, the register, the window) and
    #     the embedding through the gate's input, and nothing in the stack (encoder, encoder_v, decoder, lm_head)
    stack = ("encoder", "encoder_v", "decoder", "lm_head")
    for arm in ("REG3_KEYHINGE_POS", "REG3_FLAT", "LOCAL3_KEYHINGE"):
        m = make_model(arm, seed)
        m.train()
        m.term_state["count"] = False                            # gradients of the term alone, computed below
        gates = s56.live_gates(m, x) if isinstance(m, rm.RegBDH) else [m(x)[2]]
        v = term_value(m.term_kind, gates, x, RHL)
        if m.term_kind == "key":
            v = dc.key_eta(gates, x, RHL)                     # the hinge's argument (relu may be 0 at init)
        names = [n for n, q in m.named_parameters() if q.requires_grad]
        gr = torch.autograd.grad(v.mean(), [dict(m.named_parameters())[n] for n in names], allow_unused=True)
        nz = {n: (0.0 if g is None else float(g.abs().max())) for n, g in zip(names, gr)}
        gate_ok = (nz.get("W_g2", nz.get("W_g", 0)) > 0) and (nz.get("A_r", 1) > 0 or arm.startswith("LOCAL3"))
        rows.append((f"{arm}: the term's gradient reaches {sorted(n for n, v_ in nz.items() if v_ > 0)}; the stack "
                     f"{[n for n in stack if nz.get(n, 0) > 0] or 'none'} (embedding |grad| {nz.get('embed.weight', 0):.1e}, "
                     f"through the gate's input)", gate_ok and all(nz.get(n, 0) == 0 for n in stack)))
    # (4) the terms' values by hand on a fixed gate: KL at key / body positions
    g = torch.softmax(torch.randn(32, x.shape[1], 2, generator=torch.Generator().manual_seed(3)), -1)
    kpos, _ = ft.body_positions(RHL, x)
    kl = lambda q: float((q * torch.log(q * 2)).sum(-1).mean())
    vp = float(term_value("pos", [g], x, RHL)[0])
    vf = float(term_value("flat", [g], x, RHL)[0])
    rows.append((f"pos term = mean KL at the body's KEY positions ({vp:.6f} vs {kl(g.gather(1, kpos.unsqueeze(-1).expand(-1, -1, 2))):.6f}); "
                 f"flat = mean KL at positions < {RHL.body_len} ({vf:.6f} vs {kl(g[:, :RHL.body_len]):.6f}); KL of the uniform "
                 f"gate {float(term_value('flat', [torch.full_like(g, 0.5)], x, RHL)[0]):.1e}",
                 abs(vp - kl(g.gather(1, kpos.unsqueeze(-1).expand(-1, -1, 2)))) < 1e-6
                 and abs(vf - kl(g[:, :RHL.body_len])) < 1e-6
                 and abs(float(term_value("flat", [torch.full_like(g, 0.5)], x, RHL)[0])) < 1e-7))
    # (5) schedules: weight at 1, W-1, W, W+1, W+2 (training forwards counted, evaluation untouched)
    for arm, W in (("REG3_KEYHINGE_1200", 1200), ("REG3_FLAT", 2400)):
        m = make_model(arm, seed)
        m.train()
        for _ in range(W + 2):
            _weight(m)
        w = m.term_state["weights"]
        rows.append((f"{arm} schedule: weights {w} (1.0 through {W}, 0 after)",
                     w.get("1") == 1.0 and w.get(str(W)) == 1.0 and w.get(str(W + 1)) == 0.0 and w.get(str(W + 2)) == 0.0))
    # (6) LOCAL3_KEYHINGE: the key term on the window gate; no register; the model is S82's BASELINE otherwise
    ml = make_model("LOCAL3_KEYHINGE", seed)
    mb = rc.make_model("BASELINE", seed)
    with torch.no_grad():
        rows.append(("LOCAL3_KEYHINGE: BASELINE's parameters and logits (no register, no new parameter)",
                     dict(ml.named_parameters()).keys() == dict(mb.named_parameters()).keys()
                     and all(torch.equal(q, dict(mb.named_parameters())[n]) for n, q in ml.named_parameters())
                     and torch.equal(ml.eval()(x)[0], mb.eval()(x)[0]) and not hasattr(ml, "W_R")))
    md = make_model("REG3_FLAT_D", seed)
    rows.append(("delta variants: the term class over the delta memory (beta 1, L2 keys)",
                 isinstance(md.attn, rc.dm.DeltaMemory) and md.attn.beta_fixed == 1.0 and isinstance(md, TermReg3)))
    return [[n, bool(v)] for n, v in rows]
