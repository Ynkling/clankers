#!/usr/bin/env python
"""
explore_g_s89h_child.py — EXPLORATORY, not a result. Run code of Session G's screen S89H (explore_g_s89h): S89's arms on
HEBBIAN memory with the short causal convolution in the memory's input (F's valid Hebbian configuration of round 6).
Runs INSIDE a child process on this branch's own modules, 1 torch thread. It imports S89's child (explore_g_s89_child)
for the tasks, the hinges, the task-aware statistics and the perfect gate, and S71-S73's / S82's children for the models
and the recipe; nothing there changes.

THE CONVOLUTION (test_short_conv.LAYER: conv 'layer', K = 4; test_multilayer_binding.CausalConv: depthwise, strictly
causal, identity at initialisation, drawing nothing from any generator). At every layer the memory's input is the
convolved residual, xc = short_conv(x): Q = K = relu(xc @ encoder), V = xc; the residual stream itself is not convolved
(test_multilayer_binding.MultiBDH.forward_conv, line for line). Its weights train in the rest group (1e-4 for updates
1-2400, 1e-3 after), as in F's O_HEBB_CONV pilots.
MODELS (every model built from build(task, arm, ARCH, seed) with the arm's model_kw = LAYER, so every parameter but the
conv is the no-conv model's for the seed, CHECKed):
  BASELINE      LOCAL3 + conv: explore_b16_gates.LocalBDH, whose forward already takes forward_conv when conv is set.
  REG3          Reg3Conv: explore_g_rhl_child.Reg3BDH (one gate shared by the three layers, S56's register, SG) whose
                forward is explore_g_regmodel.RegBDH.forward with the convolution inserted exactly where forward_conv
                inserts it (the gate reads LN of the unconvolved residual, as before). With conv off it is RegBDH.forward.
  REG3_<HINGE>  TermConv: explore_g_s89_child.Term89's forward (the key / frequency / repeat term on updates 1-2400) on
                Reg3Conv. The hinges are S89's, unchanged (term_value, key_eta89, hinge_mask).
  PERFECT       PerfectConv: the 'ceiling' model + conv whose gate is task.perfect_gate (perfect_gate_general on lite),
                forward through forward_conv.
  Memory: the model's own GatedAttention (Hebbian, decay 0.95). k = 2. SLOW for every arm (explore_g_devices_child.groups /
  switch; the perfect gate's gate group is empty).
Records: as S89 (explore_g_s89_child.stats_fn89, the post-run records). NOTE: the in-run probe's per-layer features are
the attention's V input, which with the conv is the convolved residual (descriptive only).
"""

import json
import os

import torch
import torch.nn.functional as F

import explore_common as ec
import explore_aux_gate as aux
import explore_b16_gates as g16
import explore_g_common as gc
import explore_g_regmodel as rm
import explore_g_register_child as s56
import explore_g_rhl_child as rc
import explore_g_devices_child as dc
import explore_g_s87_child as s87
import explore_g_s89_child as c89
import test_binding_onset as tbo
import test_short_conv as tsc
from bdh import BDH
from test_channel_binding import ARCH
from test_instrument_v2 import perfect_gate_general
from test_multilayer_binding import MultiBDH, build, probe_batch
from test_router_reliability import run_one

HERE = os.path.dirname(os.path.abspath(__file__))
LAYER = tsc.LAYER
TASKS = c89.TASKS
WINDOW, LAM, TAU = c89.WINDOW, c89.LAM, c89.TAU
F_REF = os.path.join(gc.G_DIR, "refs", "f_s89_post_conv_records.json")


def task_of(name):
    return TASKS[name]


# ── models ───────────────────────────────────────────────────────────────────
class Reg3Conv(rc.Reg3BDH):
    """Reg3BDH with the memory-input convolution (module docstring). conv None: RegBDH.forward exactly."""

    def forward(self, tokens, tau=None, track_sat=False):
        C = self.config
        B, T = tokens.size()
        D_ = C.n_embd
        nh = C.n_head
        N = D_ * C.mlp_internal_dim_multiplier // nh
        v = self.embed(tokens)
        g1, _ = self.local_gates(v)
        Rprev, beta = self.register(v) if self.reg else (None, None)
        gates = []
        G1 = None if self.mem == "delta" else torch.einsum("btk,bsk->bts", g1, g1).unsqueeze(1)
        x = self.embed(tokens).unsqueeze(1)
        x = self.ln(x)
        for level in range(C.n_layer):
            lv = level + 1
            if lv in self.gated:
                g = self.layer_gate(lv, x.view(B, T, D_), v, Rprev)
            else:
                g = g1
            gates.append(g)
            if self.mem == "delta":
                self.attn.set_gates(g, g)
            else:
                self.attn.G = G1 if g is g1 else torch.einsum("btk,bsk->bts", g, g).unsqueeze(1)
            xc = self.short_conv(x) if self.conv == "layer" else x
            x_latent = xc @ self.encoder
            x_sparse = F.relu(x_latent)
            yKV = self.attn(Q=x_sparse, K=x_sparse, V=xc)
            yKV = self.ln(yKV)
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.drop(xy_sparse)
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            y = self.ln(yMLP)
            x = self.ln(x + y)
        logits = x.view(B, T, D_) @ self.lm_head
        if self.keep_diag:
            self.diag = dict(gates=[g.detach() for g in gates], beta=None if beta is None else beta.detach())
        gl = gates[max(self.gated) - 1] if self.gated else g1
        return logits, None, gl, gl


class TermConv(Reg3Conv):
    """explore_g_s89_child.Term89's forward on Reg3Conv (the same term, the same schedule, the same records)."""

    def forward(self, tokens, tau=None, track_sat=False):
        st = self.t_state
        training = self.training and torch.is_grad_enabled() and st["count"]
        if training:
            st["n"] += 1
        u = st["n"]
        on = self.t_on and u <= WINDOW
        live = []
        prev = self.__dict__.get("layer_gate")
        if training and on:
            inner = prev if prev is not None else (lambda level, h, v, Rprev: rc.Reg3BDH.layer_gate(self, level, h, v, Rprev))

            def lg(level, h, v, Rprev):
                g = inner(level, h, v, Rprev)
                live.append(g)
                return g
            self.layer_gate = lg
        try:
            out = Reg3Conv.forward(self, tokens, tau, track_sat)
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


class PerfectConv(MultiBDH):
    """The perfect gate from the task (task.perfect_gate; lite: perfect_gate_general), forward through forward_conv."""

    def forward(self, tokens, tau=None, track_sat=False):
        t = self.pt_task
        gr = gw = (t.perfect_gate(tokens, self.lm_head.dtype) if c89.is_pool(t)
                   else perfect_gate_general(tokens, self.ctx_tokens, self.lm_head.dtype))
        self.attn.G = torch.einsum("btk,bsk->bts", gr, gw).unsqueeze(1)
        logits = self.forward_conv(tokens) if self.conv is not None else BDH.forward(self, tokens)[0]
        return logits, None, gr, gw


ARMS = {"PERFECT_H": "oracle", "BASELINE_H": "base", "REG3_H": "reg3", "REG3_KEYHINGE_H": "key",
        "REG3_FREQHINGE_H": "freq", "REG3_REPEATHINGE_H": "repeat"}


def _arm(kind, task, conv):
    a = dict(c89.ceil_arm(task) if kind == "oracle" else ec.ARM_A)
    return dict(a, model_kw=LAYER) if conv else a


def make_model(arm, seed, task_name, on=True, conv=True):
    kind = ARMS[arm]
    task = task_of(task_name)
    a = _arm(kind, task, conv)
    m = build(task, a, ARCH, seed)
    if kind == "oracle":
        m.__class__ = PerfectConv
        m.pt_task = task
        return m
    m = g16.to_local(m, seed)
    if kind == "base":
        return m
    cv = m.conv
    m.conv = None                                   # to_reg asserts a conv-free LocalBDH; the conv is restored below
    rm.to_reg(m, seed, sg=True, reg=True, gated=(1, 2, 3), res_layers=(1, 2, 3), mem="hebb")
    m.conv = cv
    m.__class__ = Reg3Conv
    m.inp = "window"
    if kind == "reg3":
        return m
    m.__class__ = TermConv
    m.t_kind, m.t_on, m.t_task = kind, bool(on), task_name
    m.t_state = dict(n=0, fired=0, weights={}, value_at={}, count=True)
    return m


# ── run ──────────────────────────────────────────────────────────────────────
def run(p):
    """p: arm ("<task>.<ARM>" or ARM with p["task"]), seed, iters, off, conv (default True), eval_every."""
    arm, seed = p["arm"], p["seed"]
    tname, arm = arm.split(".", 1) if "." in arm else (p["task"], arm)
    task = task_of(tname)
    kind = ARMS[arm]
    conv = p.get("conv", True)
    a = dict(_arm(kind, task, conv), key=arm)
    holder = {}
    keep = {"at": -1}
    rec = run_one(a, seed, p.get("iters", rc.ITERS), p.get("eval_every", tbo.EVAL_EVERY), task=task,
                  stats_fn=c89.stats_fn89, grad_fn=tsc.conv_grad_norms, lr=dc.LR,
                  builder=lambda a_, s_: (lambda: make_model(arm, s_, tname, on=not p.get("off"), conv=conv)),
                  run_kw=dict(param_groups=dc.groups(holder, dc.LR)), check=dc.switch(holder, 0), keep=keep)
    if not rec.get("ok"):
        return json.loads(json.dumps(rec, default=float))
    m = keep["model"]
    ev = tbo.eval_batch(task)
    rec.update(arm=p["arm"], task=tname, recipe="slow", conv=m.conv, group_names=holder.get("names"),
               stream_acc=c89.stream_acc89(m, task, ev), probe=c89.probe_rh89(m, task, gc.probe_data(task)),
               acc_by_cell=c89.acc_by_cell89(m, task, ev), write_order=c89.write_order_acc89(m, task, ev))
    if isinstance(m, TermConv):
        rec["term"] = {k: v for k, v in m.t_state.items() if k != "count"}
    if kind != "oracle":
        rec["fail"] = None if rec["transition"] is not None else ec.fail_class(rec)
    return json.loads(json.dumps(rec, default=float))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def _x(task, n=16, seed=5):
    return task.make_batch(n, torch.Generator().manual_seed(seed))[0][:, :-1]


def checks_static(p):
    rows = []
    for tname in ("lite", "pool", "allpool"):
        t = task_of(tname)
        x = _x(t)
        for arm in ARMS:
            m1 = make_model(arm, 2842, tname).eval()
            m0 = make_model(arm, 2842, tname, conv=False).eval()
            p1, p0 = dict(m1.named_parameters()), dict(m0.named_parameters())
            extra = sorted(set(p1) - set(p0))
            same = all(torch.equal(p1[n], p0[n]) for n in p0)
            with torch.no_grad():
                o1, o0 = m1(x), m0(x)
            eq = torch.equal(o1[0], o0[0]) and torch.equal(o1[2], o0[2])
            rows.append((f"{tname}: {arm}: the conv is the only extra parameter ({extra}), K "
                         f"{m1.short_conv.conv_w.shape[-1]}, conv '{m1.conv}'; the shared ones equal the no-conv model's "
                         f"({same}); at initialisation (identity conv) logits and gate equal it bit for bit ({eq})",
                         extra == ["short_conv.conv_w"] and same and eq and m1.conv == "layer"
                         and m1.short_conv.conv_w.shape[-1] == tsc.CONV_K))
    # with conv off the classes are S89's / S71's: forward equal on lite for every arm kind
    t = task_of("lite")
    x = _x(t)
    pairs = (("REG3_H", rc.make_model("REG3", 2842, t)), ("BASELINE_H", rc.make_model("BASELINE", 2842, t)))
    for arm, ref in pairs:
        m0 = make_model(arm, 2842, "lite", conv=False).eval()
        ref.eval()
        with torch.no_grad():
            eq = torch.equal(m0(x)[0], ref(x)[0])
        rows.append((f"lite, conv off: {arm} equals explore_g_rhl_child's {arm[:-2]} bit for bit (logits)", eq))
    # the conv insertion equals forward_conv: REG3 with A_r = A_R = 0 is LOCAL3's gate at every layer, so with the same
    # random conv weights its logits equal BASELINE+conv's (LocalBDH -> MultiBDH.forward_conv)
    for tname in ("pool", "allpool"):
        t = task_of(tname)
        x = _x(t)
        mr = make_model("REG3_H", 2842, tname).eval()
        mb = make_model("BASELINE_H", 2842, tname).eval()
        w = torch.randn(mr.short_conv.conv_w.shape, generator=torch.Generator().manual_seed(9)) * 0.5
        with torch.no_grad():
            mr.A_r.zero_(); mr.A_R.zero_()
            mr.short_conv.conv_w.copy_(w); mb.short_conv.conv_w.copy_(w)
            lr_, lb_ = mr(x)[0], mb(x)[0]
        d = float((lr_ - lb_).abs().max())
        rows.append((f"{tname}: REG3_H with A_r = A_R = 0 and random conv weights equals BASELINE_H (forward_conv) "
                     f"(max |diff| {d:.1e})", d <= 1e-5 * float(lb_.abs().max())))
        # causality with a random conv: changing token t changes no logit at positions < t
        x2 = x.clone()
        tpos = 12
        x2[:, tpos] = (x2[:, tpos] + 1) % t.vocab
        with torch.no_grad():
            l2 = mr(x2)[0]
        rows.append((f"{tname}: REG3_H with a random conv is strictly causal (token {tpos} changed: logits < {tpos} "
                     f"unchanged, at {tpos} changed)", torch.equal(l2[:, :tpos], lr_[:, :tpos])
                     and not torch.equal(l2[:, tpos], lr_[:, tpos])))
    # the perfect gate: task.perfect_gate; on pool equal to the built-in ceiling+conv model (Instrument's gate)
    for tname in ("pool", "allpool"):
        t = task_of(tname)
        x = _x(t)
        m = make_model("PERFECT_H", 2840, tname).eval()
        with torch.no_grad():
            lg, _, gr, _ = m(x)
        ok = torch.equal(gr, t.perfect_gate(x)) and gr.shape[-1] == 2
        extra = ""
        if tname == "pool":
            ref = build(t, dict(c89.ceil_arm(t), model_kw=LAYER), ARCH, 2840).eval()
            with torch.no_grad():
                lr_ = ref(x)[0]
            extra = f"; logits equal the built-in ceiling+conv model's ({torch.equal(lg, lr_)})"
            ok = ok and torch.equal(lg, lr_)
        st = c89.stats_fn89(m, t, probe_batch(t, 2840), "end")
        ok = ok and st["eta_key_by_stream_inv"] > 0.999 and st["margin"] >= 0.9
        rows.append((f"{tname}: PERFECT_H's gate is task.perfect_gate (k = 2){extra}; untrained it reads ROUTED* "
                     f"(within-sequence eta^2 {st['eta_key_by_stream_inv']:.3f}, margin {st['margin']:.3f})", ok))
    return rows


def checks_lrs(p):
    from torch.optim.optimizer import register_optimizer_step_pre_hook
    rows = []
    for arm in ("PERFECT_H", "REG3_REPEATHINGE_H"):
        lrs, cnt = {}, [0]

        def pre(opt, args, kwargs):
            cnt[0] += 1
            if cnt[0] in (1, 2400, 2401):
                lrs[cnt[0]] = [(len(g["params"]), g["lr"]) for g in opt.param_groups]
        hk = register_optimizer_step_pre_hook(pre)
        holder = {}
        try:
            run(dict(arm=arm, seed=2842, task="pool", iters=2401))
        finally:
            hk.remove()
        ok = (lrs[1][0][1] == dc.LR and lrs[1][1][1] == dc.LR_SLOW and lrs[2400][1][1] == dc.LR_SLOW
              and lrs[2401][1][1] == dc.LR and (arm != "PERFECT_H" or lrs[1][0][0] == 0))
        m = make_model(arm, 2842, "pool")
        conv_in_rest = "short_conv.conv_w" not in rm.gate_names(m)
        rows.append([f"pool: {arm}'s (n params, lr) per group at updates 1 / 2400 / 2401: {lrs}; the conv in the rest "
                     f"group ({conv_in_rest})", bool(ok and conv_in_rest)])
    return rows


def checks_repro(p):
    """A recorded run reproduces bit for bit through 2400 through this screen's run()."""
    kind = p["kind"]
    if kind == "s87":
        ref, cpu = c89.record("s87_results.json", p["ref_arm"], p["seed"])
        r = run(dict(arm=p["arm"], seed=p["seed"], iters=2400, task="lite", conv=False))
        ok = s87.trunc(r) == s87.trunc(ref)
        return [[f"S87's {p['ref_arm']}|{p['seed']} (on {cpu}) reproduces bit for bit through 2400 through S89H's run() "
                 f"with the conv off ({p['arm']}, lite): curve {r['curve']}", bool(ok)]]
    # F's O_HEBB_CONV_<T>|2800 (claude/explore-F explore_out/F/s89_post_results.json, 2.10GHz), vendored in G_DIR/refs
    refs = json.load(open(F_REF))
    key = p["key"]
    ref = refs["records"][key]
    r = run(dict(arm="PERFECT_H", seed=p["seed"], iters=2400, task=p["task"]))
    cur = [c for c in r["curve"] if c[0] <= 2400]
    want = [c for c in ref["curve"] if c[0] <= 2400]
    ok = cur == want
    return [[f"F's {key} (vendored from {refs['source']}) reproduces through 2400 through S89H's PERFECT_H on "
             f"{p['task']} (this CPU {ec.cpu_model()}): curve {cur} vs {want}", bool(ok)]]


def checks_off(p):
    tname = p["task"]
    ref = run(dict(arm="REG3_H", seed=2842, iters=2400, task=tname))
    rows = []
    for arm in ("REG3_KEYHINGE_H", "REG3_FREQHINGE_H", "REG3_REPEATHINGE_H"):
        r = run(dict(arm=arm, seed=2842, iters=2400, task=tname, off=True))
        rows.append([f"{tname}: {arm} with its switch off equals REG3_H|2842 through 2400 bit for bit (curve {r['curve']})",
                     s87.trunc(r) == s87.trunc(ref)])
    return rows


def checks_on(p):
    tname = p["task"]
    t = task_of(tname)
    rows = []
    for arm in ARMS:
        r = run(dict(arm=arm, seed=2842, task=tname, iters=20, eval_every=10))
        e = r.get("end", {})
        need = ("margin", "eta_key_by_stream", "eta_key_by_stream_inv", "val_cos_inv", "eta_key_by_key", "spread", "layers")
        fired = r.get("term", {}).get("fired")
        rows.append([f"{tname}: {arm} runs 20 updates end to end (ok {r.get('ok')}; conv {r.get('conv')}; term fired "
                     f"{fired}/20)", bool(r.get("ok")) and all(k in e for k in need) and r.get("conv") == "layer"
                     and (fired is None or fired > 0)])
    x = _x(t, 32)
    stack = ("encoder", "encoder_v", "decoder", "lm_head", "short_conv.conv_w")
    for arm in ("REG3_KEYHINGE_H", "REG3_FREQHINGE_H", "REG3_REPEATHINGE_H"):
        m = make_model(arm, 2842, tname)
        m.train()
        m.t_state["count"] = False
        gates = s56.live_gates(m, x)
        v = c89.term_value(m.t_kind, gates, x, tname) + 0.0
        if m.t_kind == "key":
            v = c89.key_eta89(gates, x, t)
        names = [n for n, q in m.named_parameters() if q.requires_grad]
        gr = torch.autograd.grad(v.mean(), [dict(m.named_parameters())[n] for n in names], allow_unused=True)
        nz = {n: (0.0 if g_ is None else float(g_.abs().max())) for n, g_ in zip(names, gr)}
        rows.append([f"{tname}: {arm}'s term gradient reaches {sorted(n for n, v_ in nz.items() if v_ > 0)}; the stack and "
                     f"the conv: {[n for n in stack if nz.get(n, 0) > 0] or 'none'}",
                     nz.get("W_g2", 0) > 0 and all(nz.get(n, 0) == 0 for n in stack)])
    return rows
