#!/usr/bin/env python
"""
explore_g_rhl_child.py — EXPLORATORY, not a result. Run code of Session G's follow-up screens S71 (explore_g_layers),
S73 (explore_g_register3_delta) and S72 (explore_g_register3), on RandHeaderTask-lite. Runs INSIDE a child process on this
branch's own modules, 1 torch thread. Parents hold the arms' seeds and the readings.

TASK: explore_f_tasks.RandHeaderTask(P=4, nb=2, lmin=1, lmax=3), S = 2, n_vals = 16, n_q = 1 (L = 23, qpos 21; vocabulary
22), exactly as Session F's S70 (explore_f_tasks copied unchanged from claude/explore-F at 9a6c2bf). k = 2, no conv,
24000 updates, evaluation every 1200 on 2048 held-out queries, early stop after 3 evaluations >= 0.95.

MODELS (every learned-gate model starts from build(task, arm A, ARCH, seed) -> explore_b16_gates.to_local(m, seed), so
the stack, LOCAL3's gate and window and every batch are the seed's, paired across arms; the perfect-gate models built
from test_channel_binding's 'ceiling' arm draw the same stack parameters, CHECKed):
  ORACLE_ALL     the perfect gate at all three layers: build(task, ceiling) (S56's ORACLE_H code path). Hebbian.
  ORACLE_23      LOCAL3 at layer 1, the perfect gate at layers 2-3 (S57's ORACLE23 construction): RegBDH(gated=(2,3), no
                 register) whose gated layers return test_instrument_v2.perfect_gate_general. Hebbian.
  ORACLE_1       the perfect gate at layer 1, LOCAL3 at layers 2-3 (gated=(1,)).
  *_D            the same on the delta memory (explore_delta_mem.DeltaMemory: beta = 1, L2 keys, tied write, decay 0.95),
                 with the per-layer gates passed by set_gates (S56's REG_D path).
  BASELINE(_D)   LOCAL3 + SLOW (S43's recipe; F's S70_HEBB / S70_DELTA).
  REG3(_D)       Reg3BDH: ONE gate shared by the three layers,
                    g^(l)_t = softmax(W_g2 tanh(A_r LN(r^(l)_t) + A_v window_t + A_R R_{t-1})),
                 window_t = LOCAL3's width-3 causal convolution of the raw embeddings (gate_conv, batch 1's init),
                 r^(1) = LN(embedding) (a uniform code path); A_v = W_in and W_g2 = W_g (copies) at init, A_r, A_R ~
                 N(0, 0.01^2), so u = W_in window_t + a perturbation at every layer at step 0; S56's register (beta^R =
                 sigmoid(w_R . v_t + b_R), b_R = logit 0.1; the gate at t reads R_{t-1}), SG (stop-gradient on r into the
                 stack), S56's SLOW groups (gate group: LOCAL3's W_in, W_g, window + every new parameter at 1e-3; the rest
                 1e-4 for updates 1-2400, 1e-3 after). LOCAL3's own W_in and W_g stay in the model unused.
  REG3_NUDGE(_D) REG3 + S56's 5% labelled nudge applied to the gates of all three layers.
  RESGATE_ONLY_D Reg3BDH without the register (reg = False) at all three layers, SG, SLOW.
  LATCH3(_NUDGE)_D  S57's HM-RNN latch (z = 1[clamp((a u + 1)/2) > 0.5], straight-through hard-sigmoid derivative, a = 1 to
                 update 8000 then 1 -> 5 linearly to 24000, b_z = 0, w_z ~ N(0, 0.01^2); r_t = z_t W_r v_t + (1 - z_t)
                 r_{t-1}) feeding the shared gate at all three layers; no prior.
  SINGLE_D       one delta channel, no gate (test_short_conv's arm B, k = 1, -> explore_delta_mem.to_delta(beta = 1,
                 L2 keys), fixed decay 0.95): F's S53 SINGLE_DELTA path.
  RECIPES: perfect gates and SINGLE_D Adam 1e-3 on every trainable parameter throughout (ORACLE_23 / ORACLE_1 train LOCAL3's
  gate at 1e-3 too); every other arm SLOW. No hinge, no forget gate anywhere.

STATISTICS. At every evaluation: test_router_reliability.gate_stats_k (gate cosines: VAL cos -> DISCOVERED) on the model's
returned gate (the last gated layer's), explore_f_tasks.rand_header_routing_stats at 1200 / 2400 / 3600 / end (F's
eta^2 at KEY and VAL, margin), and beta^R (register) or z (latch) by role (CTX / KEY / VAL of the body, QUERY tokens).
layer_stats at 1200, 2400, 4800, 9600 and the end: per layer, eta^2 of channel 0 at KEY and VAL by stream / key / half /
index (positions read per sequence), the routing margin from that layer's gates (VAL write vs query read), the stream ->
channel map at VAL and KEY, and the CTX / KEY / VAL x stream contingency of the mean gate.
AT THE END (after the last evaluation): stream_acc; the in-run probe (S55's logistic regression, 2000 sequences, split
1600/400 by sequence) from the residual entering layers 2 and 3 and the final layer's output, at KEY and at VAL positions,
to the stream — features captured by hooks on the model's own forward (the attention's V input at each layer = the residual
entering it; the last LayerNorm output = the final layer), so per-layer gates are respected; held-out accuracy by (query
stream, block of the target pair among its stream's blocks (0/1), pair index within the block (0-2)); explore_f_tasks.
write_order_acc.
OUTCOMES (computed by the parents from the records): BOUND = transition not None; ROUTED*@end (F's) = end margin >= 0.9 and
eta^2 by stream at KEY > 0.9 (rand_header_routing_stats on the returned gate); DISCOVERED (F's) = BOUND and end VAL cos <
0.5; ROUTED-BOUND = BOUND and ROUTED*@end; BOUND ROUTED (mine) = BOUND, the VAL map one-to-one at every layer, every
stream's accuracy >= 0.9. Failure classes: test_router_layout.fail_class on the end statistics; MERGED if STREAM-PARTIAL
and the layer-3 VAL map is not one-to-one.
"""

import json
import os

import torch
import torch.nn.functional as F

import explore_common as ec
import explore_common2 as c2
import explore_b16_gates as g16
import explore_delta_mem as dm
import explore_f_tasks as ft
import explore_g_common as gc
import explore_g_regmodel as rm
import explore_g_register_child as s56
import explore_g_latch_child as s57
import test_binding_onset as tbo
import test_short_conv as tsc
from test_multilayer_binding import build
from test_channel_binding import ARCH, arms as cb_arms
from test_router_layout import eta2
from test_router_reliability import run_one
from test_instrument_v2 import perfect_gate_general

ITERS = 24000
LR = 1e-3
RHL = ft.RandHeaderTask(P=4, nb=2, lmin=1, lmax=3)
CEIL = {a["key"]: a for a in cb_arms(RHL)}["ceiling"]
ARM_B = tsc.ARM["B"]
LAYER_AT = (1200, 2400, 4800, 9600, "end")
STATS_F = ft.rand_header_stats(c2.ROUTE_AT)

# arm -> kind, memory, options
ARMS = {
    "ORACLE_ALL": ("oracle_all", "hebb", {}),
    "ORACLE_23": ("oracle_layers", "hebb", dict(perfect=(2, 3))),
    "ORACLE_1": ("oracle_layers", "hebb", dict(perfect=(1,))),
    "ORACLE_ALL_D": ("oracle_all", "delta", {}),
    "ORACLE_23_D": ("oracle_layers", "delta", dict(perfect=(2, 3))),
    "BASELINE": ("local", "hebb", {}),
    "REG3": ("reg3", "hebb", dict(reg=True)),
    "REG3_NUDGE": ("reg3", "hebb", dict(reg=True, nudge=True)),
    "BASELINE_D": ("local", "delta", {}),
    "REG3_D": ("reg3", "delta", dict(reg=True)),
    "REG3_NUDGE_D": ("reg3", "delta", dict(reg=True, nudge=True)),
    "RESGATE_ONLY_D": ("reg3", "delta", dict(reg=False)),
    "LATCH3_D": ("latch3", "delta", {}),
    "LATCH3_NUDGE_D": ("latch3", "delta", dict(nudge=True)),
    "SINGLE_D": ("single", "delta", {}),
}


# ── models ───────────────────────────────────────────────────────────────────
class OracleLayers(rm.RegBDH):
    """RegBDH whose gated layers take the perfect gate; the other layers LOCAL3's gate."""

    def layer_gate(self, level, h, v, Rprev):
        return perfect_gate_general(self._tok, self.ctx_tokens).to(v.dtype)

    def forward(self, tokens, tau=None, track_sat=False):
        self._tok = tokens
        return rm.RegBDH.forward(self, tokens, tau, track_sat)


class Reg3BDH(rm.RegBDH):
    """RegBDH whose gate reads LOCAL3's window (inp = "window") or the raw embedding (inp = "v": S56's code path)."""

    def layer_gate(self, level, h, v, Rprev):
        x_in = self.gate_conv(v) if self.inp == "window" else v
        return rm.RegBDH.layer_gate(self, level, h, x_in, Rprev)


class Latch3BDH(s57.LatchBDH, Reg3BDH):
    """S57's latch (register) with Reg3BDH's gate input."""


def local(task, seed):
    return g16.to_local(build(task, ec.ARM_A, ARCH, seed), seed)


def make_reg3(task, seed, mem, reg=True, gated=(1, 2, 3), inp="window", cls=Reg3BDH):
    m = rm.to_reg(local(task, seed), seed, sg=True, reg=reg, gated=gated, res_layers=gated, mem=mem)
    m.__class__ = cls
    m.inp = inp
    if mem == "delta":
        s56.delta_attn(m, seed)
    return m


def make_model(arm, seed, task=RHL):
    kind, mem, o = ARMS[arm]
    if kind == "oracle_all":
        m = build(task, CEIL, ARCH, seed)
        return dm.to_delta(m, **s56.DELTA_KW) if mem == "delta" else m
    if kind == "oracle_layers":
        m = rm.to_reg(local(task, seed), seed, reg=False, gated=o["perfect"], res_layers=(), mem=mem)
        for n in ("A_v", "A_r", "W_g2"):                         # unused by the perfect gate: removed
            delattr(m, n)
        m.__class__ = OracleLayers
        if mem == "delta":
            s56.delta_attn(m, seed)
        return m
    if kind == "single":
        return dm.to_delta(build(task, ARM_B, ARCH, seed), seed=seed, beta=1.0, normalize=True)
    if kind == "local":
        m = local(task, seed)
        return dm.to_delta(m, seed=seed, **s56.DELTA_KW) if mem == "delta" else m
    if kind == "reg3":
        m = make_reg3(task, seed, mem, reg=o["reg"])
        if o.get("nudge"):
            s56.nudge_reg(m, task, seed)
        return m
    if kind == "latch3":
        m = make_reg3(task, seed, mem, cls=Latch3BDH)
        with torch.no_grad():
            m.b_R.zero_()
        m.prior, m.n_upd, m.count_upd = False, 0, False
        if o.get("nudge"):
            s56.nudge_reg(m, task, seed)
        m.count_upd = True
        return m
    raise KeyError(arm)


def recipe(arm):
    return "adam" if ARMS[arm][0] in ("oracle_all", "oracle_layers", "single") else "slow"


# ── statistics ───────────────────────────────────────────────────────────────
def roles(task, x):
    S, P = task.S, task.P
    body = torch.zeros_like(x, dtype=torch.bool)
    body[:, :task.body_len] = True
    return dict(CTX=(x < S) & body, KEY=(x >= S) & (x < S + P) & body, VAL=(x >= S + P) & body, QUERY=~body)


@torch.no_grad()
def write_rate(model, task, probe):
    """beta^R (register) or the latch's z, mean per role; None without a register."""
    if not isinstance(model, rm.RegBDH) or not getattr(model, "reg", False):
        return None
    pinp = probe[0]
    was = model.training
    model.eval()
    try:
        u = model.embed(pinp) @ model.w_R + model.b_R
        if isinstance(model, s57.LatchBDH):
            w = (torch.clamp((s57.a_at(model.n_upd) * u + 1) / 2, 0, 1) > 0.5).float()
        else:
            w = torch.sigmoid(u)
    finally:
        model.train(was)
    return {r: w[m_].mean().item() for r, m_ in roles(task, pinp).items()}


@torch.no_grad()
def layer_stats(model, task, probe):
    pinp = probe[0]
    gates = s56.all_gates(model, pinp)[0]
    S, P, n = task.S, task.P, task.S * task.P
    B = pinp.shape[0]
    kpos, vpos = ft.body_positions(task, pinp)
    lab = task.stream_labels(pinp)
    ctx = lab.gather(1, kpos)
    key = pinp.gather(1, kpos) - S
    j = torch.arange(n)
    labels = dict(stream=ctx, key=key, half=(j >= n // 2).long().expand(B, n), index=j.expand(B, n))
    rl = roles(task, pinp)
    qpos = task.qpos[0]
    qs, qk = pinp[:, qpos - 1], pinp[:, qpos] - S
    same_key = key == qk[:, None]
    tgt = same_key & (ctx == qs[:, None])
    dis = same_key & (ctx != qs[:, None])
    out = {}
    for li, g in enumerate(gates, 1):
        d = {}
        p = g[..., 0]
        k_ = g.shape[-1]
        for role, pos in (("key", kpos), ("val", vpos)):
            for name, lb in labels.items():
                d[f"eta_{role}_by_{name}"] = eta2(p.gather(1, pos), lb)
            gg = g.gather(1, pos.unsqueeze(-1).expand(-1, -1, k_))
            ll = lab.gather(1, pos)
            means = [gg[ll == s].mean(0) for s in range(S)]
            cmap = [int(m_.argmax()) for m_ in means]
            d[f"ch_map_{role}"] = cmap
            d[f"one_to_one_{role}"] = len(set(cmap)) == S
        gv = g.gather(1, vpos.unsqueeze(-1).expand(-1, -1, k_))
        dots = (gv * g[:, qpos][:, None, :]).sum(-1)
        d["margin"] = ((dots * tgt).sum(1) - (dots * dis).sum(1) / (S - 1)).mean().item()
        d["table"] = {r: [[round(v, 4) for v in g[m_ & (lab == s)].mean(0).tolist()] for s in range(S)]
                      for r, m_ in rl.items()}
        out[f"L{li}"] = d
    return out


def stats_fn(model, task, probe, step):
    if model.n_ch < 2:
        return dict(step=step, key_cos=0.0, val_cos=0.0, ctx_cos=0.0)
    out = STATS_F(model, task, probe, step)
    out["beta"] = write_rate(model, task, probe)
    if step in LAYER_AT:
        out["layers"] = layer_stats(model, task, probe)
    return out


@torch.no_grad()
def capture(model, x):
    """(feats, logits): the residual entering each layer (the attention's V input) and the final layer's output (the last
    LayerNorm call of the forward), from the model's own forward."""
    vs, lns = [], []
    h1 = model.attn.register_forward_pre_hook(lambda mod, args, kw: vs.append(kw["V"].detach()), with_kwargs=True)
    h2 = model.ln.register_forward_hook(lambda mod, args, out: lns.append(out.detach()))
    was = model.training
    model.eval()
    try:
        logits = model(x)[0]
    finally:
        model.train(was)
        h1.remove(); h2.remove()
    B, T = x.shape
    feats = {f"L{i + 1}": v.reshape(B, T, -1) for i, v in enumerate(vs)}
    feats["final"] = lns[-1].reshape(B, T, -1)
    return feats, logits


def probe_rh(model, task, data, layers=("L2", "L3", "final")):
    x, tr, te = data
    feats, _ = capture(model, x)
    kpos, vpos = ft.body_positions(task, x)
    lab = task.stream_labels(x)
    out = {}
    for name in layers:
        f = feats[name]
        for role, pos in (("K", kpos), ("V", vpos)):
            X = f.gather(1, pos.unsqueeze(-1).expand(-1, -1, f.shape[-1])).double().numpy()
            y = lab.gather(1, pos).numpy()
            Xtr, ytr = X[tr.numpy()].reshape(-1, X.shape[-1]), y[tr.numpy()].reshape(-1)
            Xte, yte = X[te.numpy()].reshape(-1, X.shape[-1]), y[te.numpy()].reshape(-1)
            hit, tracc = gc.fit_score(Xtr, ytr, Xte, yte)
            k, n = int(hit.sum()), int(hit.size)
            out[f"{name}|{role}"] = dict(acc=k / n, k=k, n=n, wilson=gc.wilson(k, n), train_acc=tracc)
    return out


@torch.no_grad()
def acc_by_cell(model, task, data):
    """Held-out accuracy by (query stream, block of the target pair among its stream's blocks, pair index in the block)."""
    was = model.training
    model.eval()
    try:
        ql, qt, _ = task.select(tbo.logits_of(model, data[:, :-1]), data[:, 1:], None)
    finally:
        model.train(was)
    inp = data[:, :-1]
    S = task.S
    qpos = task.qpos[0]
    qs, qk = inp[:, qpos - 1], inp[:, qpos]
    hit = (ql.argmax(-1) == qt)
    cells = {}
    for b in range(inp.shape[0]):
        seq = inp[b, :task.body_len].tolist()
        s_q, k_q = int(qs[b]), int(qk[b])
        cur, nblk, idx, found = None, {s: -1 for s in range(S)}, 0, None
        for pos, tok in enumerate(seq):
            if tok < S:
                cur = tok
                nblk[tok] += 1
                idx = 0
            elif S <= tok < S + task.P:
                if cur == s_q and tok == k_q:
                    found = (nblk[cur], idx)
                idx += 1
        key = f"{s_q}|{found[0]}|{found[1]}"
        c = cells.setdefault(key, [0, 0])
        c[0] += int(hit[b]); c[1] += 1
    return {k: dict(acc=v[0] / v[1], n=v[1]) for k, v in sorted(cells.items())}


# ── run ──────────────────────────────────────────────────────────────────────
def run(p):
    arm, seed = p["arm"], p["seed"]
    task = RHL
    holder = {}
    a = dict(CEIL if ARMS[arm][0] == "oracle_all" else ARM_B if ARMS[arm][0] == "single" else ec.ARM_A, key=arm)
    kw, check = {}, None
    if recipe(arm) == "slow":
        kw = dict(param_groups=s56.groups(holder))
        check = s56.lr_switch(holder)
    keep = {"at": -1}
    rec = run_one(a, seed, p.get("iters", ITERS), task=task, stats_fn=stats_fn, grad_fn=tsc.conv_grad_norms, lr=LR,
                  builder=lambda a_, s_: (lambda: make_model(arm, s_)), run_kw=kw, check=check, keep=keep)
    if not rec.get("ok"):
        return json.loads(json.dumps(rec, default=float))
    m = keep["model"]
    ev = tbo.eval_batch(task)
    rec.update(arm=arm, recipe=recipe(arm), group_names=holder.get("names"),
               stream_acc=gc.stream_acc(m, task, ev), probe=probe_rh(m, task, gc.probe_data(task)),
               acc_by_cell=acc_by_cell(m, task, ev), write_order=ft.write_order_acc(m, task, ev))
    if a["n_ch"] > 1:
        rec["fail"] = None if rec["transition"] is not None else ec.fail_class(rec)
    return json.loads(json.dumps(rec, default=float))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def _x(n=16, seed=4, task=RHL):
    return task.make_batch(n, torch.Generator().manual_seed(seed))[0][:, :-1]


def checks_task(p):
    rows = [(f"RandHeaderTask-lite (F's check_rand_header): {nm}", v) for nm, v in ft.check_rand_header(RHL)]
    rows.append((f"RHL as F's S70: P 4, nb 2, lengths 1..3, compositions {RHL.comps}, L {RHL.L}, qpos {RHL.qpos}, vocab {RHL.vocab}",
                 RHL.comps == [(1, 3), (2, 2), (3, 1)] and RHL.L == 23 and RHL.qpos == [21] and RHL.vocab == 22))
    # the in-run probe's features equal S55's residual extraction on a model with one gate for all layers
    m = make_model("BASELINE", 700)
    x = _x(32)
    f1, l1 = capture(m, x)
    f0, l0, _ = gc.residuals(m, x)
    rows.append(("in-run probe: hook-captured residuals (L1-L3, final) and logits equal S55's extraction bit for bit on LOCAL3",
                 torch.equal(l1, l0) and all(torch.equal(f1[k], f0[k]) for k in ("L1", "L2", "L3", "final"))))
    # per-layer gates: on ORACLE_23 the hook sees each layer's own memory (layer 2's residual differs from a one-G replay)
    mo = make_model("ORACLE_23", 700)
    fo, lo = capture(mo, x)
    rows.append(("in-run probe on per-layer gates: logits from the hooked forward equal model(x)'s; layer 1's input is "
                 "LN(embedding)", torch.equal(lo, mo.eval()(x)[0]) and torch.allclose(
                     fo["L1"], F.layer_norm(mo.embed(x), (mo.embed.weight.shape[1],)), atol=1e-6)))
    cells = acc_by_cell(make_model("ORACLE_ALL", 700), RHL, tbo.eval_batch(RHL))
    rows.append((f"acc_by_cell covers {len(cells)} (stream, block, index) cells, {sum(c['n'] for c in cells.values())} queries; "
                 f"blocks 0/1, index 0-2", sum(c["n"] for c in cells.values()) == tbo.eval_batch(RHL).shape[0]
                 and all(k.split("|")[1] in ("0", "1") and k.split("|")[2] in ("0", "1", "2") for k in cells)))
    return rows


def checks_s71(p):
    rows = checks_task(p)
    x = _x()
    pg = perfect_gate_general(x, RHL.ctx_tokens)
    for arm, perf in (("ORACLE_ALL", (1, 2, 3)), ("ORACLE_23", (2, 3)), ("ORACLE_1", (1,)),
                      ("ORACLE_ALL_D", (1, 2, 3)), ("ORACLE_23_D", (2, 3))):
        m = make_model(arm, 700)
        with torch.no_grad():
            gs = s56.all_gates(m, x)[0]
            lg = g16.LocalBDH.local_gates(m, m.embed(x))[0] if hasattr(m, "gate_conv") else None
        ok = all(torch.equal(gs[l - 1], pg) for l in perf) and all(
            torch.equal(gs[l - 1], lg) for l in (1, 2, 3) if l not in perf)
        rows.append((f"{arm}: one-hot on the stream at layers {perf}, LOCAL3's gate at the others", ok))
    for mine, s56arm in (("ORACLE_ALL", "ORACLE_H"), ("ORACLE_ALL_D", "ORACLE_D")):
        a, b = make_model(mine, 700), s56.make_model(s56arm, 700, task=RHL)
        pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
        with torch.no_grad():
            same = set(pa) == set(pb) and all(torch.equal(pa[k], pb[k]) for k in pa) and torch.equal(a.eval()(x)[0], b.eval()(x)[0])
        rows.append((f"{mine} equals S56's {s56arm} code path (parameters and logits)", same))
    pa = dict(make_model("ORACLE_ALL", 701).named_parameters())
    pb = dict(make_model("ORACLE_23", 701).named_parameters())
    pc = dict(make_model("ORACLE_1", 701).named_parameters())
    stack = ("decoder", "encoder", "encoder_v", "lm_head", "embed.weight")
    rows.append(("the oracle arms share every stack parameter for the seed (paired)",
                 all(torch.equal(pa[k], pb[k]) and torch.equal(pa[k], pc[k]) for k in stack)))
    for arm in ("ORACLE_ALL", "ORACLE_ALL_D"):
        mo = make_model(arm, 700).eval()
        xs = _x(64, 11)
        lab = RHL.stream_labels(xs)
        _, vpos = ft.body_positions(RHL, xs)
        xo = xs.clone()
        for b in range(xs.shape[0]):
            q = int(xs[b, RHL.qpos[0] - 1])
            for vp in vpos[b].tolist():
                if int(lab[b, vp]) != q:
                    xo[b, vp] = RHL.S + RHL.P + (int(xo[b, vp]) - RHL.S - RHL.P + 1) % RHL.n_vals
        with torch.no_grad():
            ok = torch.equal(mo(xs)[0][:, RHL.qpos[0]], mo(xo)[0][:, RHL.qpos[0]])
        rows.append((f"{arm}: the query's logits are bit-identical when every VAL token of the other stream changes", ok))
    return [[n, bool(v)] for n, v in rows]


def checks_reg3(p, mem):
    rows = checks_task(p)
    rows += [(f"S56: {n}", v) for n, v in s56.checks(dict(res_layers=[2, 3]))]
    x = _x()
    # (a) REG3's class with the layer-1 path disabled and the raw-embedding input reproduces S56's REG|370 on H8 through
    #     2400 bit for bit (S56's run path; the record from s56_register_results.json)
    with open(os.path.join(gc.G_DIR, "s56_register_results.json")) as f:
        st = json.load(f)
    rec0 = [r for k, r in st["runs"].items() if k.startswith("REG|370|")][0]
    orig = s56.make_model
    s56.make_model = lambda arm, seed, task=None, res_layers=None: make_reg3(s56.H8, seed, "hebb", gated=(2, 3), inp="v")
    try:
        r = s56.run(dict(arm="REG", seed=370, iters=2400, res_layers=[2, 3]))
    finally:
        s56.make_model = orig
    want = [c for c in rec0["curve"] if c[0] <= 2400]
    rows.append((f"REG3 with the layer-1 path disabled (gated (2, 3), input v) reproduces S56's REG|370 through 2400 bit for "
                 f"bit ({r['curve']} vs {want})", r["curve"] == want))
    # (b) every new path disabled (A_r = A_R = 0): every layer's gate and the logits equal LOCAL3's
    arm3 = "REG3" if mem == "hebb" else "REG3_D"
    m = make_model(arm3, 700)
    base = make_model("BASELINE" if mem == "hebb" else "BASELINE_D", 700)
    with torch.no_grad():
        m.A_r.zero_(); m.A_R.zero_()
        gs = s56.all_gates(m, x)[0]
        lg = g16.LocalBDH.local_gates(base, base.embed(x))[0]
        same_l = torch.equal(m.eval()(x)[0], base.eval()(x)[0])
    rows.append((f"{arm3} with every new path disabled: the gate at every layer equals LOCAL3's and the logits equal "
                 f"{'BASELINE' if mem == 'hebb' else 'BASELINE_D'}'s bit for bit", all(torch.equal(g, lg) for g in gs) and same_l))
    # (c) init: u = W_in window + perturbation; beta^R ~ 0.1; existing parameters unchanged; A_v = W_in, W_g2 = W_g
    m = make_model(arm3, 701)
    pr = dict(local(RHL, 701).named_parameters())
    with torch.no_grad():
        bet = torch.sigmoid(m.embed(x) @ m.w_R + m.b_R)
    rows.append((f"{arm3} init: existing parameters unchanged; A_v = W_in, W_g2 = W_g; beta^R in [{bet.min():.4f}, {bet.max():.4f}]",
                 all(torch.equal(q, pr[n]) for n, q in m.named_parameters() if n in pr) and torch.equal(m.A_v, m.W_in)
                 and torch.equal(m.W_g2, m.W_g) and 0.09 < float(bet.min()) and float(bet.max()) < 0.11 and m.gated == (1, 2, 3)))
    # (d) causality of the register / latch at every layer, layer 1 included; strict causality of the whole model
    t = 12
    for arm in ([arm3] + (["LATCH3_D"] if mem == "delta" else [])):
        mm = make_model(arm, 700)
        with torch.no_grad():
            g0 = s56.all_gates(mm, x)[0]
            orig_r = mm.register

            def pert(v, orig_r=orig_r):
                v2 = v.clone(); v2[:, t] = v2[:, t] + 1.0
                return orig_r(v2)
            mm.register = pert
            try:
                g1 = s56.all_gates(mm, x)[0]
            finally:
                del mm.register
            x2 = x.clone(); x2[:, t] = (x2[:, t] + 1) % RHL.vocab
            la, lb = mm.eval()(x)[0], mm.eval()(x2)[0]
        rows.append((f"{arm}: token {t}'s register/latch write changes no gate at <= {t} at layers 1-3 and changes layer "
                     f"1's at {t + 1} (register; the latch's binary z need not change); changing token {t} changes no logit at < {t}",
                     all(torch.equal(a_[:, :t + 1], b_[:, :t + 1]) for a_, b_ in zip(g0, g1))
                     and (arm.startswith("LATCH") or not torch.equal(g0[0][:, t + 1], g1[0][:, t + 1]))
                     and torch.equal(la[:, :t], lb[:, :t])))
    # (e) delta beta 0 / write 1 / raw keys equals the Hebbian REG3 (1e-5)
    mh = make_reg3(RHL, 700, "hebb")
    md = make_reg3(RHL, 700, "hebb")
    md.attn = dm.DeltaMemory(md.config, decay=md.attn.decay, beta=0.0, write=1.0, normalize=False, n_ch=md.n_ch, impl="parallel")
    md.mem = "delta"
    with torch.no_grad():
        dd = float((mh.eval()(x)[0] - md.eval()(x)[0]).abs().max())
    rows.append((f"REG3, delta with beta 0, write 1, raw keys equals the Hebbian REG3 (max |diff| {dd:.1e} <= 1e-5)", dd <= 1e-5))
    # (f) groups
    holder = {}
    s56.groups(holder)(m)
    names = [n for n, q in m.named_parameters() if q.requires_grad]
    rows.append((f"{arm3} SLOW groups: gate {holder['names'][0]} at 1e-3, rest {holder['names'][1]} at 1e-4; every trainable "
                 f"tensor once", sorted(holder["names"][0] + holder["names"][1]) == sorted(names)
                 and {"gate_conv.conv_w", "A_v", "A_r", "W_g2", "A_R", "W_R", "w_R", "b_R"} <= set(holder["names"][0])))
    # (g) the nudge on all three layers
    mn, m0 = make_model(arm3.replace("REG3", "REG3_NUDGE"), 702), make_model(arm3, 702)
    moved = {n for n, q in mn.named_parameters() if not torch.equal(q, dict(m0.named_parameters())[n])}
    xe = _x(256, 9)
    with torch.no_grad():
        gg = s56.all_gates(mn, xe)[0]
        lab = RHL.stream_labels(xe)
        lean = [float((gg[l][..., 0][lab == 0].mean() - gg[l][..., 0][lab == 1].mean()).abs()) for l in range(3)]
    rows.append((f"nudge moves only {sorted(moved)}; stream lean at layers 1-3 {[round(v, 3) for v in lean]} (> 0.02)",
                 moved <= set(rm.GATE_NEW) and min(lean) > 0.02))
    if mem == "delta":
        # (h) the latch at layer 1: dead-STE guard; the latch feeds all three layers
        ml = make_model("LATCH3_D", 700)
        with torch.no_grad():
            u = ml.embed(x) @ ml.w_R + ml.b_R
        rows.append((f"LATCH3_D: dead-STE guard max |u| {float(u.abs().max()):.2e} < 1 at step 0, b_z 0; gated layers {ml.gated}; "
                     f"update counter {ml.n_upd}", float(u.abs().max()) < 1 and float(ml.b_R) == 0.0 and ml.gated == (1, 2, 3)
                     and ml.n_upd == 0 and isinstance(ml, s57.LatchBDH)))
        rn = make_model("RESGATE_ONLY_D", 700)
        rows.append(("RESGATE_ONLY_D: no register, the residual gate at all three layers",
                     not rn.reg and not hasattr(rn, "W_R") and rn.gated == (1, 2, 3) and rn.res_layers == (1, 2, 3)))
        # (i) SINGLE_D = F's S53 SINGLE_DELTA path (arm B, k = 1, to_delta beta 1, L2 keys, fixed decay)
        sd = make_model("SINGLE_D", 700)
        ref = dm.to_delta(build(RHL, tsc.ARM["B"], ARCH, 700), seed=700, beta=1.0, normalize=True)
        with torch.no_grad():
            same = torch.equal(sd.eval()(x)[0], ref.eval()(x)[0])
        rows.append(("SINGLE_D equals F's S53 single-channel path (logits bit for bit); k = 1, beta 1, L2 keys, fixed decay",
                     same and sd.n_ch == 1 and sd.attn.beta_fixed == 1.0 and sd.attn.normalize and sd.attn.decay_kind == "fixed"))
        mb = make_model("BASELINE_D", 700)
        rows.append(("BASELINE_D: LOCAL3 + delta (beta 1, L2 keys, tied write, decay 0.95)",
                     isinstance(mb.attn, dm.DeltaMemory) and mb.attn.beta_fixed == 1.0 and mb.attn.normalize))
    return [[n, bool(v)] for n, v in rows]


def checks(p):
    return {"s71": checks_s71, "s73": lambda q: checks_reg3(q, "delta"), "s72": lambda q: checks_reg3(q, "hebb")}[p["screen"]](p)


def timing(p):
    import time
    t = time.time()
    r = run(dict(arm=p["arm"], seed=0, iters=p["steps"]))
    return dict(secs=time.time() - t, ok=r.get("ok"), err=r.get("error"))
