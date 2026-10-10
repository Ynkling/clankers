#!/usr/bin/env python
"""
explore_g_devices_child.py — EXPLORATORY, not a result. Run code of Session G's screen S82 (explore_g_devices): devices that
delay or prevent the key split so the register's CTX write can emerge. Runs INSIDE a child process on this branch's own
modules, 1 torch thread. It imports S71-S73's child (explore_g_rhl_child) unchanged for the task, the REG3 model, the
statistics, the in-run probe and the per-cell accuracy.

TASK: RandHeaderTask-lite exactly as F's S70 (explore_g_rhl_child.RHL), k = 2, no conv, 24000 updates, evaluation every
1200, early stop after 3 evaluations >= 0.95.
ARMS (memory Hebbian unless _D; every learned-gate arm SLOW = S56's groups: gate group (LOCAL3's W_in, W_g, window + every
new gate / register parameter) at 1e-3, the rest 1e-4 for updates 1-2400 and 1e-3 after; no hinge unless stated):
  BASELINE(_D)     LOCAL3 + SLOW (S72 / S73's BASELINE)
  REG3(_D)         S72 / S73's REG3: one gate shared by layers 1-3, g = softmax(W_g2 tanh(A_r LN(r^(l)) + A_v window_t +
                   A_R R_{t-1})), S56's register, SG
  REG3_KEYHINGE    REG3 + S52's key-term hinge (explore_keysplits_child.attach_key_hinge, its constants LAM = 1.0,
                   TAU = 0.2, WINDOW = 2400): on training forwards of updates 1-2400 the loss gains
                       LAM x mean_l relu(eta2_key(g^(l)) - TAU),
                   eta2_key = explore_b18_gates.eta2_pool of layer l's gate at the body's KEY positions grouped by the key
                   id there (token - S): S52's eta2_key with the positions read per sequence (RandHeaderTask's KEY
                   positions are not 3j + 1), averaged over the three layers' gates (the shared gate's three outputs; equal
                   to S52's term when the three gates are equal); weight LAM for updates 1-2400 and 0 after (the term is
                   then not computed); entered through explore_aux_gate._Inject, as S52. IT READS KEY POSITIONS AND KEY
                   IDENTITIES (token types: which tokens are keys and which key each is), as the key split itself reads
                   key positions; it reads no stream label.
  REG3_SLOWGATE    REG3 with the gate group at 1e-4 for updates 1-1200 and 1e-3 after (the switch after the evaluation at
                   1200); the rest as SLOW (1e-4 to 2400, 1e-3 after).
  REG3_NOWIN       REG3 whose gate reads [LN(r^(l)); R_{t-1}] only: u = A_r LN(r^(l)) + A_R R_{t-1}, g = softmax(W_g2
                   tanh(u)); no window term: the window (gate_conv), A_v and LOCAL3's W_in, W_g are removed (key identity
                   reaches the gate only through the residual). A_r, A_R, W_g2 and the register as REG3's (same draws).
  REG3_NUDGE       REG3 + S56's 5% labelled nudge at all three layers (validity of the code path)
  ORACLE_ALL(_D)   the perfect gate at all three layers, Adam 1e-3 throughout (S71's)
  <DEVICE>_D       the best Hebbian device (rule in the parent) on delta memory (beta 1, L2 keys, tied write, decay 0.95)
Records: explore_g_rhl_child's (statistics at every evaluation, layer statistics, in-run probe, accuracy by cell, write
order) plus, for REG3_KEYHINGE, the hinge's firings per 1200-update window, eta2_key at updates 1, 300, 600, 1200, 2400 and
the weight at updates 1, 2399-2402; for REG3_SLOWGATE the lrs at updates 1, 1200, 1201, 2400, 2401.
"""

import json
import math

import torch
import torch.nn.functional as F
from torch.optim.optimizer import register_optimizer_step_pre_hook

import explore_common as ec
import explore_aux_gate as aux
import explore_b18_gates as g18
import explore_f_tasks as ft
import explore_g_common as gc
import explore_g_regmodel as rm
import explore_g_register_child as s56
import explore_g_rhl_child as rc
import test_binding_onset as tbo
import test_short_conv as tsc
from test_router_reliability import run_one

LAM, TAU, WINDOW = 1.0, 0.2, 2400          # explore_keysplits_child (S52): LAM, TAU, WINDOW
LR, LR_SLOW = 1e-3, 1e-4
WARM, GATE_WARM = 2400, 1200
LOG_ETA_AT = (1, 300, 600, 1200, 2400)
W_LOG_AT = (1, WINDOW - 1, WINDOW, WINDOW + 1, WINDOW + 2)
LR_LOG_AT = (1, GATE_WARM, GATE_WARM + 1, WARM, WARM + 1)
RHL = rc.RHL

ARMS = {
    "BASELINE": ("rc", "BASELINE"), "BASELINE_D": ("rc", "BASELINE_D"),
    "REG3": ("rc", "REG3"), "REG3_D": ("rc", "REG3_D"), "REG3_NUDGE": ("rc", "REG3_NUDGE"),
    "ORACLE_ALL": ("rc", "ORACLE_ALL"), "ORACLE_ALL_D": ("rc", "ORACLE_ALL_D"),
    "REG3_KEYHINGE": ("keyhinge", "hebb"), "REG3_KEYHINGE_D": ("keyhinge", "delta"),
    "REG3_SLOWGATE": ("slowgate", "hebb"), "REG3_SLOWGATE_D": ("slowgate", "delta"),
    "REG3_NOWIN": ("nowin", "hebb"), "REG3_NOWIN_D": ("nowin", "delta"),
}


# ── devices ──────────────────────────────────────────────────────────────────
class KeyHingeBDH(rc.Reg3BDH):
    """REG3 + S52's key-term hinge on the three layers' gates (module docstring). lam = 0 switches the device off."""

    def forward(self, tokens, tau=None, track_sat=False):
        st = self.kh
        active = self.training and torch.is_grad_enabled() and st["count"]
        live = []
        if active:
            st["n"] += 1
            u = st["n"]
            w = self.kh_lam if u <= WINDOW else 0.0
            if u in W_LOG_AT:
                st["weights"][str(u)] = w
            if w != 0.0:
                orig = rc.Reg3BDH.layer_gate

                def lg(level, h, v, Rprev):
                    g = orig(self, level, h, v, Rprev)
                    live.append(g)
                    return g
                self.layer_gate = lg
        try:
            out = rm.RegBDH.forward(self, tokens, tau, track_sat)
        finally:
            if live or "layer_gate" in self.__dict__:
                self.__dict__.pop("layer_gate", None)
        if not live:
            return out
        e = key_eta(live, tokens, self.kh_task)
        h = F.relu(e - self.kh_tau)
        hm = h.mean()
        u = st["n"]
        st["last"] = [float(x) for x in e.detach()]
        if u in LOG_ETA_AT:
            st["eta_at"][str(u)] = [round(float(x), 5) for x in e.detach()]
        win = (u - 1) // tbo.EVAL_EVERY
        while len(st["windows"]) <= win:
            st["windows"].append(0)
        if bool((h > 0).any()):
            st["fired"] += 1
            st["windows"][win] += 1
        logits = aux._Inject.apply(out[0], self.kh_lam * hm)
        return (logits,) + tuple(out[1:])


def key_eta(gates, tokens, task):
    """eta2_key per layer: explore_b18_gates.eta2_pool of the layer's gate at the body's KEY positions, grouped by key id."""
    kpos, _ = ft.body_positions(task, tokens)
    kid = tokens.gather(1, kpos) - task.S
    out = []
    for g in gates:
        gk = g.gather(1, kpos.unsqueeze(-1).expand(-1, -1, g.shape[-1]))
        out.append(g18.eta2_pool(gk, kid, task.P))
    return torch.stack(out)


class NoWinBDH(rc.Reg3BDH):
    """REG3 whose gate reads [LN(r); R_{t-1}] only (win = False); win = True is REG3 itself (the switch)."""

    def layer_gate(self, level, h, v, Rprev):
        if self.win:
            return rc.Reg3BDH.layer_gate(self, level, h, v, Rprev)
        r = h.detach() if self.sg else h
        u = F.layer_norm(r, (r.shape[-1],)) @ self.A_r.T
        if Rprev is not None:
            u = u + Rprev @ self.A_R.T
        return F.softmax(torch.tanh(u) @ self.W_g2.T, dim=-1)

    def local_gates(self, v):
        if self.win:
            return rc.Reg3BDH.local_gates(self, v)
        g = torch.full(v.shape[:-1] + (self.n_ch,), 1.0 / self.n_ch, dtype=v.dtype)   # unused: every layer is gated
        return g, g


def make_model(arm, seed, task=RHL, lam=LAM, tau=TAU, win=False):
    kind, mem = ARMS[arm]
    if kind == "rc":
        return rc.make_model(mem, seed, task)
    m = rc.make_model("REG3_D" if mem == "delta" else "REG3", seed, task)
    if kind == "keyhinge":
        m.__class__ = KeyHingeBDH
        m.kh = dict(n=0, fired=0, windows=[], weights={}, last=None, eta_at={}, count=True)
        m.kh_lam, m.kh_tau, m.kh_task = float(lam), float(tau), task
    elif kind == "nowin":
        m.__class__ = NoWinBDH
        m.win = bool(win)
        if not m.win:
            for n in ("A_v", "W_in", "W_g"):
                delattr(m, n)
            del m.gate_conv
    return m


# ── recipes ──────────────────────────────────────────────────────────────────
def groups(holder, gate_lr):
    def param_groups(model):
        named = [(n, q) for n, q in model.named_parameters() if q.requires_grad]
        nm = dict(named)
        gate = rm.gate_names(model)
        rest = [n for n, _ in named if n not in gate]
        holder["names"] = [gate, rest]
        holder["groups"] = [dict(params=[nm[n] for n in gate], lr=gate_lr), dict(params=[nm[n] for n in rest], lr=LR_SLOW)]
        return holder["groups"]
    return param_groups


def switch(holder, gate_warm):
    def check(m, step, data):
        if gate_warm and step >= gate_warm:
            holder["groups"][0]["lr"] = LR
        if step >= WARM:
            holder["groups"][1]["lr"] = LR
    return check


def run(p):
    arm, seed = p["arm"], p["seed"]
    kind, mem = ARMS[arm]
    inner = mem if kind == "rc" else ("REG3_D" if mem == "delta" else "REG3")
    a = dict(rc.CEIL if rc.ARMS[inner][0] == "oracle_all" else rc.ARM_B if rc.ARMS[inner][0] == "single" else ec.ARM_A,
             key=arm)
    holder, kw, check = {}, {}, None
    slowgate = kind == "slowgate" and not p.get("off")
    if rc.recipe(inner) == "slow":
        kw = dict(param_groups=groups(holder, LR_SLOW if slowgate else LR))
        check = switch(holder, GATE_WARM if slowgate else 0)
    keep = {"at": -1}
    lrs = {}
    hook = None
    if kind == "slowgate":
        cnt = [0]

        def pre(opt, args, kwargs):
            cnt[0] += 1
            if cnt[0] in LR_LOG_AT:
                lrs[str(cnt[0])] = [g["lr"] for g in opt.param_groups]
        hook = register_optimizer_step_pre_hook(pre)
    mk = dict(lam=0.0 if p.get("off") else LAM, win=bool(p.get("off")))
    try:
        rec = run_one(a, seed, p.get("iters", rc.ITERS), task=RHL, stats_fn=rc.stats_fn, grad_fn=tsc.conv_grad_norms, lr=LR,
                      builder=lambda a_, s_: (lambda: make_model(arm, s_, **mk)), run_kw=kw, check=check, keep=keep)
    finally:
        if hook is not None:
            hook.remove()
    if not rec.get("ok"):
        return json.loads(json.dumps(rec, default=float))
    m = keep["model"]
    ev = tbo.eval_batch(RHL)
    rec.update(arm=arm, recipe=rc.recipe(inner), group_names=holder.get("names"),
               stream_acc=gc.stream_acc(m, RHL, ev), probe=rc.probe_rh(m, RHL, gc.probe_data(RHL)),
               acc_by_cell=rc.acc_by_cell(m, RHL, ev), write_order=ft.write_order_acc(m, RHL, ev))
    if isinstance(m, KeyHingeBDH):
        rec["hinge"] = {k: v for k, v in m.kh.items() if k != "count"}
    if lrs:
        rec["lrs"] = lrs
    if a["n_ch"] > 1:
        rec["fail"] = None if rec["transition"] is not None else ec.fail_class(rec)
    return json.loads(json.dumps(rec, default=float))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def checks(p):
    rows = [(f"S73: {n}", v) for n, v in rc.checks(dict(screen="s73"))]
    seed = 2100
    # (1) each device with its switch off equals REG3 bit for bit through 2400 (curve, statistics, probe)
    ref = run(dict(arm="REG3", seed=seed, iters=2400))
    for arm in ("REG3_KEYHINGE", "REG3_SLOWGATE", "REG3_NOWIN"):
        r = run(dict(arm=arm, seed=seed, iters=2400, off=True))
        same = r["curve"] == ref["curve"] and r["stats"] == ref["stats"] and r["probe"] == ref["probe"]
        rows.append((f"{arm} with its switch off (lam 0 / gate lr 1e-3 from update 1 / window on) equals REG3|{seed} through 2400 "
                     f"bit for bit (curve {r['curve']} vs {ref['curve']}; statistics; probe)", same))
    # (2) the hinge's gradient: on a training forward with the logits' loss zero, the gradient on the gate parameters equals
    #     autograd of LAM x mean_l relu(eta2_key - TAU) computed directly (TAU lowered to -1 so every term is active)
    x = RHL.make_batch(32, torch.Generator().manual_seed(5))[0][:, :-1]
    m = make_model("REG3_KEYHINGE", seed, tau=-1.0)
    m.train()
    out = m(x)[0]
    ga = torch.autograd.grad(out.sum() * 0.0, [m.A_r, m.W_g2])
    m2 = make_model("REG3", seed)
    gates = s56.live_gates(m2, x)
    direct = LAM * F.relu(key_eta(gates, x, RHL) + 1.0).mean()
    gb = torch.autograd.grad(direct, [m2.A_r, m2.W_g2])
    d = max(float((u - v).abs().max()) for u, v in zip(ga, gb))
    rows.append((f"hinge gradient through _Inject equals autograd of LAM mean_l relu(eta2_key - TAU) (max |diff| {d:.1e} <= 1e-7; "
                 f"|grad| {float(gb[1].abs().max()):.2e} > 0)", d <= 1e-7 and float(gb[1].abs().max()) > 0))
    # (3) key_eta grouping: per layer, equal to test_router_layout-style eta^2 pooled over channels computed by hand
    g = torch.softmax(torch.randn(16, x.shape[1], 2, generator=torch.Generator().manual_seed(3)), -1)
    e = float(key_eta([g], x[:16], RHL)[0])
    kpos, _ = ft.body_positions(RHL, x[:16])
    gk = g.gather(1, kpos.unsqueeze(-1).expand(-1, -1, 2)).reshape(-1, 2).double()
    kid = (x[:16].gather(1, kpos) - RHL.S).reshape(-1)
    mu = gk.mean(0)
    tot = ((gk - mu) ** 2).sum()
    btw = sum((kid == k).sum() * ((gk[kid == k].mean(0) - mu) ** 2).sum() for k in range(RHL.P))
    rows.append((f"eta2_key at RandHeaderTask's KEY positions grouped by key id equals the hand computation ({e:.6f} vs "
                 f"{float(btw / tot):.6f})", abs(e - float(btw / tot)) < 1e-5))
    # (4) the hinge's schedule: weight LAM at updates 1 and 2400, 0 at 2401 and 2402 (a 2402-update run, statistics stubbed)
    saved = (tbo.evaluate, rc.stats_fn)
    tbo.evaluate = lambda model, task, data: (0.5, 1.0)
    rc.stats_fn = lambda model, task, probe, step: dict(step=step, key_cos=0.0, val_cos=0.0, ctx_cos=0.0)
    keep = {"at": -1}
    try:
        holder = {}
        rec = run_one(dict(ec.ARM_A, key="KH"), seed, WINDOW + 2, task=RHL, stats_fn=rc.stats_fn, grad_fn=tsc.conv_grad_norms,
                      lr=LR, builder=lambda a_, s_: (lambda: make_model("REG3_KEYHINGE", s_)),
                      run_kw=dict(param_groups=groups(holder, LR)), check=switch(holder, 0), keep=keep)
    finally:
        tbo.evaluate, rc.stats_fn = saved
    w = keep["model"].kh["weights"]
    rows.append((f"hinge schedule: weight at updates {w} (LAM {LAM} through {WINDOW}, 0 after); "
                 f"{keep['model'].kh['fired']} firings in {WINDOW} updates", w.get("1") == LAM and w.get(str(WINDOW)) == LAM
                 and w.get(str(WINDOW + 1)) == 0.0 and w.get(str(WINDOW + 2)) == 0.0 and rec.get("ok")))
    # (5) the slow-gate groups and lrs through the real run path (2401 updates, statistics as recorded)
    r = run(dict(arm="REG3_SLOWGATE", seed=seed, iters=2401))
    L = r.get("lrs", {})
    rows.append((f"slow-gate lrs [gate, rest] at updates 1 / 1200 / 1201 / 2400 / 2401: {[L.get(str(u)) for u in LR_LOG_AT]}",
                 L.get("1") == [LR_SLOW, LR_SLOW] and L.get("1200") == [LR_SLOW, LR_SLOW] and L.get("1201") == [LR, LR_SLOW]
                 and L.get("2400") == [LR, LR_SLOW] and L.get("2401") == [LR, LR]))
    rows.append((f"slow-gate groups: gate {r['group_names'][0]}, rest {r['group_names'][1]}",
                 {"gate_conv.conv_w", "A_v", "A_r", "W_g2", "A_R", "W_R", "w_R", "b_R"} <= set(r["group_names"][0])
                 and "encoder" in r["group_names"][1]))
    # (6) NOWIN has no window parameters; its gate does not change when the raw window input changes alone; causality
    mn = make_model("REG3_NOWIN", seed)
    names = [n for n, _ in mn.named_parameters()]
    rows.append((f"REG3_NOWIN has no window parameters (no gate_conv, A_v, W_in, W_g): {names}",
                 not any(n.startswith("gate_conv") or n in ("A_v", "W_in", "W_g") for n in names)
                 and all(n in names for n in ("A_r", "A_R", "W_g2", "W_R", "w_R", "b_R"))))
    with torch.no_grad():
        mn.eval()
        g0 = s56.all_gates(mn, x)[0]
        t = 12
        x2 = x.clone(); x2[:, t] = (x2[:, t] + 1) % RHL.vocab
        g1 = s56.all_gates(mn, x2)[0]
    rows.append(("REG3_NOWIN: changing token 12 changes no gate at positions < 12 at any layer (causal; the register at 12 "
                 "enters only from 13)", all(torch.equal(a_[:, :t], b_[:, :t]) for a_, b_ in zip(g0, g1))))
    mon = make_model("REG3_NOWIN", seed, win=True)
    mr = make_model("REG3", seed)
    with torch.no_grad():
        rows.append(("REG3_NOWIN with the window on has REG3's parameters and logits bit for bit",
                     {n: q for n, q in mon.named_parameters()}.keys() == {n: q for n, q in mr.named_parameters()}.keys()
                     and all(torch.equal(q, dict(mr.named_parameters())[n]) for n, q in mon.named_parameters())
                     and torch.equal(mon.eval()(x)[0], mr.eval()(x)[0])))
    return [[n, bool(v)] for n, v in rows]
