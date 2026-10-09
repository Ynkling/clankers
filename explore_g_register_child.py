#!/usr/bin/env python
"""
explore_g_register_child.py — EXPLORATORY, not a result. S56's run code (Session G). Runs INSIDE a child process on this
branch's own modules, 1 torch thread. Parent, arms and readings: explore_g_register. Model: explore_g_regmodel.

TASK: explore_far_cue.HeaderTask(8, S=2, n_vals=16, n_q=1) (L = 37, qpos 35), the same object as S55's H8.
RECIPES
  SLOW   (every arm with a gate module): one Adam; gate group = explore_g_regmodel.gate_names (LOCAL3's W_in, W_g, window
         and every new gate / register parameter) at 1e-3 throughout; every other trainable parameter (the stack, and the
         delta memory's, if any) at 1e-4 for updates 1-2400 and 1e-3 after (the switch after the evaluation at 2400, as
         S43). W_h frozen. No hinge.
  ADAM   (ORACLE_*, SINGLE_GDN): Adam 1e-3 on every parameter throughout (S53 / S69's recipe).
  24000 updates, evaluation every 1200 on 2048 held-out queries, early stop after 3 evaluations >= 0.95.
NUDGE (REG_NUDGE*): explore_far_express's far_nudge (test_router_discovery.nudge) transposed to the per-layer gates: inside
  make_model, 300 Adam steps at lr 0.01 on the new gate / register parameters (explore_g_regmodel.GATE_NEW) toward
  0.475 + 0.05 x onehot(stream) for the gates of layers 2 and 3 at every position (MSE; the label is the latest CTX
  token's stream, BindTask.stream_labels), batches of 64 from a generator seeded seed + 55_555; then normal training.
STATISTICS at every evaluation: explore_far_cue.header_stats on the model's returned gate (the last gated layer's;
LOCAL3's for BASELINE), plus layer_stats (below) at 1200, 2400, 3600, 4800, 9600 and the end, and beta^R by role at every
evaluation. At the end: stream_acc.
layer_stats, per layer l in 1..3 (the gate of that layer; on the run's probe, eval mode): eta^2 of channel 0 at the body's
  KEY and VAL positions by stream / key / half / index (test_router_layout.eta2), the routing margin (the query's read
  gate against the write gates at the VAL positions of the target pair and its same-key distractor, as
  explore_far_cue.header_routing_stats), the stream -> channel map at VAL and at KEY (argmax of the mean gate per stream;
  one_to_one), and the contingency table: the mean gate vector per (role CTX / KEY / VAL) x (stream). beta^R (the
  register's write strength): mean at CTX, KEY, VAL positions of the body and at the query's tokens.
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import explore_common as ec
import explore_far_cue as s6
import explore_window_recipe_child as s43c
import explore_b16_gates as g16
import explore_g_common as gc
import explore_g_regmodel as rm
import explore_delta_mem as dm
import test_short_conv as tsc
import test_binding_onset as tbo
from test_multilayer_binding import build
from test_channel_binding import ARCH, arms as cb_arms
from test_router_layout import eta2
from test_router_reliability import run_one
from test_instrument_v2 import TAU_END

ITERS = 24000
LR, LR_WARM, WARM = 1e-3, 1e-4, 2400
H8 = s6.HeaderTask(8, S=2, n_vals=16, n_q=1)
CEIL = {a["key"]: a for a in cb_arms(H8)}["ceiling"]
ARM_B = tsc.ARM["B"]
NUDGE_STEPS, NUDGE_LR, NUDGE_B, NUDGE_BASE, NUDGE_AMP, NUDGE_OFFSET = 300, 1e-2, 64, 0.475, 0.05, 55_555
LAYER_AT = (1200, 2400, 3600, 4800, 9600, "end")
DELTA_KW = dict(beta=1.0, write="tied", normalize=True)          # F's S59 / S70 DELTA: beta 1, L2 keys, tied write, 0.95

# arm -> (kind, options)
ARM = {
    "ORACLE_H":     ("oracle", dict(mem="hebb")),
    "BASELINE":     ("local", dict(mem="hebb")),
    "RESGATE_ONLY": ("reg", dict(mem="hebb", reg=False, sg=True)),
    "REG":          ("reg", dict(mem="hebb", reg=True, sg=True)),
    "REG_NOSG":     ("reg", dict(mem="hebb", reg=True, sg=False)),
    "REG_NUDGE":    ("reg", dict(mem="hebb", reg=True, sg=True, nudge=True)),
    "SINGLE_GDN":   ("single_gdn", dict(mem="hebb")),
    "ORACLE_D":     ("oracle", dict(mem="delta")),
    "BASELINE_D":   ("local", dict(mem="delta")),
    "REG_D":        ("reg", dict(mem="delta", reg=True, sg=True)),
    "REG_NUDGE_D":  ("reg", dict(mem="delta", reg=True, sg=True, nudge=True)),
}
RES_LAYERS = None          # set by the parent from S55's rule (explore_g_register.res_layers); None = (2, 3)


def delta_attn(m, seed):
    old = m.attn
    new = dm.DeltaMemory(m.config, decay=old.decay, decay_kind="fixed", n_ch=m.n_ch, seed=seed, impl="parallel",
                         **DELTA_KW)
    assert torch.equal(new.freqs, old.freqs)
    m.attn = new
    return m


def make_model(arm, seed, task=H8, res_layers=None):
    kind, o = ARM[arm]
    if kind == "oracle":
        m = build(task, CEIL, ARCH, seed)
        return dm.to_delta(m, **DELTA_KW) if o["mem"] == "delta" else m
    if kind == "single_gdn":
        return dm.to_delta(build(task, ARM_B, ARCH, seed), beta=0.0, write=1.0, normalize=False, decay="gdn", seed=seed)
    m = g16.to_local(build(task, ec.ARM_A, ARCH, seed), seed)
    if kind == "local":
        return dm.to_delta(m, seed=seed, **DELTA_KW) if o["mem"] == "delta" else m
    m = rm.to_reg(m, seed, sg=o["sg"], reg=o["reg"], res_layers=res_layers if res_layers is not None else RES_LAYERS,
                  mem=o["mem"])
    if o["mem"] == "delta":
        delta_attn(m, seed)
    if o.get("nudge"):
        nudge_reg(m, task, seed)
    return m


def nudge_reg(m, task, seed):
    """See NUDGE in the module docstring. The gates are recomputed with gradient through a forward with keep_diag."""
    g = torch.Generator().manual_seed(seed + NUDGE_OFFSET)
    params = [getattr(m, n) for n in rm.GATE_NEW if hasattr(m, n)]
    opt = torch.optim.Adam(params, lr=NUDGE_LR)
    for _ in range(NUDGE_STEPS):
        x = task.make_batch(NUDGE_B, g)[0][:, :-1]
        lab = task.stream_labels(x)
        tgt = NUDGE_BASE + NUDGE_AMP * F.one_hot(lab.clamp(min=0), task.S).float()
        gates = live_gates(m, x)
        loss = sum(F.mse_loss(gates[l - 1], tgt) for l in m.gated)
        opt.zero_grad(); loss.backward(); opt.step()
    m.zero_grad(set_to_none=True)


def live_gates(m, x):
    """Every layer's gate with gradient (RegBDH.forward's computation, run with a hook that keeps them)."""
    keep = []
    old = m.layer_gate

    def lg(level, h, v, Rprev):
        g = old(level, h, v, Rprev)
        keep.append((level, g))
        return g
    m.layer_gate = lg
    try:
        m(x)
    finally:
        del m.layer_gate
    out = [None] * m.config.n_layer
    for level, g in keep:
        out[level - 1] = g
    return out


# ── statistics ───────────────────────────────────────────────────────────────
@torch.no_grad()
def all_gates(model, x):
    """[g^(1), g^(2), g^(3)] (B, T, k) and beta^R (B, T) or None; eval mode."""
    was = model.training
    model.eval()
    try:
        if isinstance(model, rm.RegBDH):
            model.keep_diag = True
            model(x)
            model.keep_diag = False
            return model.diag["gates"], model.diag["beta"]
        g = model(x)[2]
        return [g] * model.config.n_layer, None
    finally:
        model.train(was)


def roles(task, x):
    S, P = task.S, task.P
    ctx = x < S
    key = (x >= S) & (x < S + P)
    val = x >= S + P
    body = torch.zeros_like(ctx)
    body[:, :task.S * task.blk] = True
    return dict(CTX=ctx & body, KEY=key & body, VAL=val & body, QUERY=~body)


@torch.no_grad()
def layer_stats(model, task, probe):
    pinp = probe[0]
    gates, beta = all_gates(model, pinp)
    S, P = task.S, task.P
    kpos, vpos, _ = gc.header_positions(task)
    n = S * P
    B = pinp.shape[0]
    j = torch.arange(n)
    lab = task.stream_labels(pinp)
    ctx = lab[:, kpos]
    key = pinp[:, kpos] - S
    labels = dict(stream=ctx, key=key, half=(j >= P).long().expand(B, n), index=j.expand(B, n))
    rl = roles(task, pinp)
    out = {}
    qpos = task.qpos[0]
    qs, qk = pinp[:, qpos - 1], pinp[:, qpos] - S
    same_key = key == qk[:, None]
    tgt = same_key & (ctx == qs[:, None])
    dis = same_key & (ctx != qs[:, None])
    for li, g in enumerate(gates, 1):
        d = {}
        p = g[..., 0]
        for role, pos in (("key", kpos), ("val", vpos)):
            for name, lb in labels.items():
                d[f"eta_{role}_by_{name}"] = eta2(p[:, pos], lb)
            gg, ll = g[:, pos], lab[:, pos]
            means = [gg[ll == s].mean(0) for s in range(S)]
            cmap = [int(m_.argmax()) for m_ in means]
            d[f"ch_map_{role}"] = cmap
            d[f"one_to_one_{role}"] = len(set(cmap)) == S
        dots = (g[:, vpos] * g[:, qpos][:, None, :]).sum(-1)
        d["margin"] = ((dots * tgt).sum(1) - (dots * dis).sum(1) / (S - 1)).mean().item()
        d["table"] = {r: [[round(v, 4) for v in g[m_ & (lab == s)].mean(0).tolist()] for s in range(S)]
                      for r, m_ in rl.items()}
        out[f"L{li}"] = d
    if beta is not None:
        out["beta"] = {r: beta[m_].mean().item() for r, m_ in rl.items()}
    return out


@torch.no_grad()
def beta_by_role(model, task, probe):
    if not isinstance(model, rm.RegBDH) or not model.reg:
        return None
    pinp = probe[0]
    was = model.training
    model.eval()
    try:
        v = model.embed(pinp)
        beta = torch.sigmoid(v @ model.w_R + model.b_R)
    finally:
        model.train(was)
    return {r: beta[m_].mean().item() for r, m_ in roles(task, pinp).items()}


def stats_for(task):
    def st(model, t, probe, step):
        if model.n_ch < 2:
            return dict(step=step, key_cos=0.0, val_cos=0.0, ctx_cos=0.0)
        out = s6.header_stats(model, t, probe, step)
        out["beta"] = beta_by_role(model, t, probe)
        if step in LAYER_AT:
            out["layers"] = layer_stats(model, t, probe)
        return out
    return st


def groups(holder):
    def param_groups(model):
        named = [(n, q) for n, q in model.named_parameters() if q.requires_grad]
        nm = dict(named)
        gate = rm.gate_names(model)
        rest = [n for n, _ in named if n not in gate]
        holder["names"] = [gate, rest]
        holder["groups"] = [dict(params=[nm[n] for n in gate], lr=LR), dict(params=[nm[n] for n in rest], lr=LR_WARM)]
        return holder["groups"]
    return param_groups


def lr_switch(holder):
    def check(m, step, data):
        if step >= WARM:
            holder["groups"][1]["lr"] = LR
    return check


def run(p):
    global RES_LAYERS
    arm, seed = p["arm"], p["seed"]
    if p.get("res_layers"):
        RES_LAYERS = tuple(p["res_layers"])
    else:                                    # the parent's decision (explore_g_register.read_path), stored before any run
        with open(os.path.join(gc.G_DIR, "s56_read_path.json")) as f:
            RES_LAYERS = tuple(json.load(f)["res_layers"])
    kind, o = ARM[arm]
    holder = {}
    a = dict(CEIL if kind == "oracle" else ARM_B if kind == "single_gdn" else ec.ARM_A, key=arm)
    kw = {}
    check = None
    if kind in ("local", "reg"):
        kw = dict(param_groups=groups(holder))
        check = lr_switch(holder)
    keep = {"at": -1}
    rec = run_one(a, seed, p.get("iters", ITERS), task=H8, stats_fn=stats_for(H8), grad_fn=tsc.conv_grad_norms, lr=LR,
                  builder=lambda a_, s_: (lambda: make_model(arm, s_)), run_kw=kw, check=check, keep=keep)
    if not rec.get("ok"):
        return json.loads(json.dumps(rec, default=float))
    m = keep["model"]
    rec["stream_acc"] = gc.stream_acc(m, H8, tbo.eval_batch(H8))
    rec["group_names"] = holder.get("names")
    rec["res_layers"] = list(getattr(m, "res_layers", ()) or ())
    rec.update(arm=arm, tag=ec.tag(rec) if a["n_ch"] > 1 else ("BOUND" if rec["transition"] is not None else "unbound"))
    if hasattr(m, "attn") and isinstance(m.attn, dm.DeltaMemory) and m.attn.decay_mod is not None:
        rec["gdn"] = dict(A=torch.exp(m.attn.decay_mod.A_log).tolist(), dt_bias=m.attn.decay_mod.dt_bias.tolist())
    return json.loads(json.dumps(rec, default=float))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def checks(p):
    import explore_window_recipe_child as wrc
    rows = []
    res = tuple(p["res_layers"]) if p.get("res_layers") else None
    # (1) the recorded run with the new code path disabled: batch 16's S43 LOCAL3_SLOW|160 (grouped P=4) through 2400,
    #     with the model converted to RegBDH with no gated layer (gated = ()), through S43's own run path
    with open(os.path.join(ec.OUT_DIR, "window_recipe_results.json")) as f:
        st = json.load(f)
    recs = [r for k, r in st["runs"].items() if k.startswith("LOCAL3_SLOW|160|")]
    rec0 = recs[0]
    saved = wrc.builder

    def b_off(slow):
        def b(a_, seed):
            def make():
                m = saved(slow)(a_, seed)()
                assert type(m) is g16.LocalBDH or type(m).__name__ == "LocalGateBDH"
                if type(m) is not g16.LocalBDH:
                    m.__class__ = g16.LocalBDH
                return rm.to_reg(m, seed, gated=())
            return make
        return b
    wrc.builder = b_off
    try:
        r = wrc.run(dict(arm="LOCAL3_SLOW", seed=160, iters=2400))
    finally:
        wrc.builder = saved
    want = [c for c in rec0["curve"] if c[0] <= 2400]
    rows.append((f"recorded run, new path disabled: S43 LOCAL3_SLOW|160 (batch 16) through 2400 with RegBDH(gated=()) "
                 f"reproduces its curve bit for bit ({r['curve']} vs {want}) and its decoder at 1200/2400",
                 r["curve"] == want and all(r["decode"][s] == rec0["decode"][s] for s in ("1200", "2400"))))
    # (2) RegBDH with no gated layer equals LocalBDH's logits bit for bit on H8
    x = H8.make_batch(16, torch.Generator().manual_seed(4))[0][:, :-1]
    l0 = g16.to_local(build(H8, ec.ARM_A, ARCH, 370), 370)
    l1 = rm.to_reg(g16.to_local(build(H8, ec.ARM_A, ARCH, 370), 370), 370, gated=())
    l0.eval(); l1.eval()
    with torch.no_grad():
        rows.append(("RegBDH(gated=()) equals LOCAL3's logits bit for bit on H8", torch.equal(l0(x)[0], l1(x)[0])))
    # (3) initialisation: no existing parameter changes; A_v = W_in, W_g2 = W_g; beta = sigmoid(w_R.v + logit 0.1) ~ 0.1;
    #     with A_r = A_R = 0 the gated layers' u equals W_in v exactly
    m = make_model("REG", 370, res_layers=res)
    ref = build(H8, ec.ARM_A, ARCH, 370)
    pr = dict(ref.named_parameters())
    same = all(torch.equal(q, pr[n]) for n, q in m.named_parameters() if n in pr)
    with torch.no_grad():
        bet = torch.sigmoid(m.embed(x) @ m.w_R + m.b_R)
        m2 = make_model("REG", 370, res_layers=res)
        m2.A_r.zero_(); m2.A_R.zero_()
        gates = all_gates(m2, x)[0]
        want_g = F.softmax(torch.tanh(m2.embed(x) @ m2.W_in.T) @ m2.W_g.T, -1)
    rows.append((f"REG init: existing parameters unchanged; A_v = W_in, W_g2 = W_g; beta^R in [{bet.min():.4f}, {bet.max():.4f}] "
                 f"(about 0.1); with A_r = A_R = 0 every gated layer's gate = softmax(W_g tanh(W_in v)) "
                 f"(max |diff| {max(float((gates[l - 1] - want_g).abs().max()) for l in m2.gated):.1e})",
                 same and torch.equal(m.A_v, m.W_in) and torch.equal(m.W_g2, m.W_g)
                 and 0.09 < float(bet.min()) and float(bet.max()) < 0.11
                 and all(float((gates[l - 1] - want_g).abs().max()) <= 1e-7 for l in m2.gated)))
    # (4) causality of the register: the gate at t reads R_{t-1}. Perturb only token t's register write (W_R v_t and
    #     beta_t, through a hook on the register): the gated layers' gates at <= t are unchanged and at t + 1 change.
    t = 20
    with torch.no_grad():
        g0 = all_gates(m, x)[0]
        orig = m.register

        def reg_pert(v):
            v2 = v.clone()
            v2[:, t] = v2[:, t] + 1.0
            return orig(v2)
        m.register = reg_pert
        try:
            g1 = all_gates(m, x)[0]
        finally:
            del m.register
    blind = all(torch.equal(g0[l - 1][:, :t + 1], g1[l - 1][:, :t + 1]) for l in m.gated)
    sees = all(not torch.equal(g0[l - 1][:, t + 1], g1[l - 1][:, t + 1]) for l in m.gated)
    rows.append((f"register causality: changing token {t}'s register write leaves every gated layer's gate at positions "
                 f"<= {t} bit-identical and changes it at {t + 1}", blind and sees))
    # (5) the whole model is causal: changing token t changes no output (logits, gates) at positions < t
    with torch.no_grad():
        x2 = x.clone(); x2[:, t] = (x2[:, t] + 1) % H8.vocab
        m.eval()
        la, lb = m(x)[0], m(x2)[0]
        ga, gb = all_gates(m, x)[0], all_gates(m, x2)[0]
    rows.append((f"strict causality: changing token {t} leaves logits and every layer's gate at positions < {t} unchanged",
                 torch.equal(la[:, :t], lb[:, :t]) and all(torch.equal(a_[:, :t], b_[:, :t]) for a_, b_ in zip(ga, gb))))
    # (6) SG vs NOSG: identical forward; SG sends no gradient from the gated layers' gates into the stack
    ms, mn = make_model("REG", 370, res_layers=res), make_model("REG_NOSG", 370, res_layers=res)
    same_f = torch.equal(ms(x)[0], mn(x)[0])
    gs = live_gates(ms, x)
    gn = live_gates(mn, x)
    l_s = sum(g[..., 0].sum() for g in gs[1:])
    l_n = sum(g[..., 0].sum() for g in gn[1:])
    gsd = torch.autograd.grad(l_s, ms.encoder, allow_unused=True)[0]
    gnd = torch.autograd.grad(l_n, mn.encoder, allow_unused=True)[0]
    rows.append(("SG and NOSG: the same forward; the gated layers' gates send no gradient into the stack (encoder) under SG "
                 "and a non-zero one under NOSG", same_f and (gsd is None or float(gsd.abs().max()) == 0.0)
                 and gnd is not None and float(gnd.abs().max()) > 0))
    # (7) RESGATE_ONLY has no register; its gates use the residual
    mr = make_model("RESGATE_ONLY", 370, res_layers=res)
    rows.append(("RESGATE_ONLY: no register parameters; reg = False", not hasattr(mr, "W_R") and not mr.reg))
    # (8) the groups: gate = LOCAL3's gate + every new parameter, rest = the stack; every trainable tensor once; lrs
    holder = {}
    groups(holder)(m)
    names = [n for n, q in m.named_parameters() if q.requires_grad]
    rows.append((f"SLOW groups: gate {holder['names'][0]} at 1e-3, rest {holder['names'][1]} at 1e-4; every trainable tensor "
                 f"once; W_h frozen", sorted(holder["names"][0] + holder["names"][1]) == sorted(names)
                 and set(holder["names"][0]) == {"W_in", "W_g", "gate_conv.conv_w"} | set(rm.GATE_NEW)
                 and not m.W_h.requires_grad))
    # (9) the nudge: the target, the parameters it moves (only GATE_NEW), and the lean it leaves
    mn0 = make_model("REG", 371, res_layers=res)
    mnu = make_model("REG_NUDGE", 371, res_layers=res)
    moved = {n for n, q in mnu.named_parameters() if not torch.equal(q, dict(mn0.named_parameters())[n])}
    xe = H8.make_batch(256, torch.Generator().manual_seed(9))[0][:, :-1]
    with torch.no_grad():
        gg = all_gates(mnu, xe)[0]
        lab = H8.stream_labels(xe)
        lean = [float((gg[l - 1][..., 0][lab == 0].mean() - gg[l - 1][..., 0][lab == 1].mean()).abs()) for l in mnu.gated]
    rows.append((f"nudge moves only {sorted(moved)} (a subset of GATE_NEW); stream lean |p0(s0) - p0(s1)| after it at the "
                 f"gated layers {[round(v, 3) for v in lean]} (> 0.02)", moved <= set(rm.GATE_NEW) and min(lean) > 0.02))
    # (10) delta: beta = 0 / write 1 / raw keys on RegBDH equals the Hebbian RegBDH to 1e-5 (the parallel form)
    mh = make_model("REG", 370, res_layers=res)
    md = make_model("REG", 370, res_layers=res)
    old = md.attn
    md.attn = dm.DeltaMemory(md.config, decay=old.decay, beta=0.0, write=1.0, normalize=False, n_ch=md.n_ch, impl="parallel")
    md.mem = "delta"
    with torch.no_grad():
        dd = float((mh(x)[0] - md(x)[0]).abs().max())
    rows.append((f"delta path with beta 0, write 1, raw keys equals the Hebbian RegBDH (max |diff| {dd:.1e} <= 1e-5)", dd <= 1e-5))
    # (11) the oracle (delta) sends no cross-stream contribution: replacing the other stream's VAL tokens leaves the logits
    #      of a stream-0 query bit-identical (Hebbian and delta)
    ok11 = True
    for arm in ("ORACLE_H", "ORACLE_D"):
        mo = make_model(arm, 370)
        mo.eval()
        xs = H8.make_batch(64, torch.Generator().manual_seed(11))[0][:, :-1]
        labs = H8.stream_labels(xs)
        _, vpos, _ = gc.header_positions(H8)
        xo = xs.clone()
        for b in range(xs.shape[0]):
            qs = int(xs[b, H8.qpos[0] - 1])
            for vp in vpos.tolist():
                if int(labs[b, vp]) != qs:
                    xo[b, vp] = H8.S + H8.P + (int(xo[b, vp]) - H8.S - H8.P + 1) % H8.n_vals
        with torch.no_grad():
            ok11 &= torch.equal(mo(xs)[0][:, H8.qpos[0]], mo(xo)[0][:, H8.qpos[0]])
    rows.append(("ORACLE_H and ORACLE_D: the query's logits are bit-identical when every VAL token of the other stream "
                 "changes", ok11))
    # (12) SINGLE_GDN: k = 1, GDN decay, beta 0 / write 1 / raw keys (the Hebbian memory with a learned decay)
    sg_ = make_model("SINGLE_GDN", 370)
    rows.append(("SINGLE_GDN: k = 1, DeltaMemory(beta 0, write 1, raw keys, decay gdn)",
                 sg_.n_ch == 1 and isinstance(sg_.attn, dm.DeltaMemory) and sg_.attn.decay_kind == "gdn"
                 and sg_.attn.beta_fixed == 0.0 and sg_.attn.write == 1.0 and not sg_.attn.normalize))
    return [[n, bool(v)] for n, v in rows]


def timing(p):
    import time
    from torch.optim.optimizer import register_optimizer_step_post_hook
    stamps = []
    h = register_optimizer_step_post_hook(lambda o, a, k: stamps.append(time.perf_counter()))
    try:
        run(dict(arm=p["arm"], seed=0, iters=p["steps"], res_layers=p.get("res_layers")))
    finally:
        h.remove()
    dd = sorted(y - x for x, y in zip(stamps, stamps[1:]))
    return dd[len(dd) // 2]
