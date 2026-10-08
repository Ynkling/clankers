#!/usr/bin/env python
"""
explore_f_decay_child.py — EXPLORATORY, not a result. S59's run code (parent: explore_f_decay). Runs INSIDE a child
process on this branch's own modules (no main-line hook), 1 torch thread.

  {ORACLE,SINGLE}_{FIXED,ROUTED,GDN}  HeaderTask(8, S=2, n_vals=16, n_q=1) (explore_far_cue), the perfect gate (k = 2,
        test_channel_binding's 'ceiling') or one channel (test_short_conv's arm B, k = 1), converted by
        explore_delta_mem.to_delta(beta=1.0, normalize=True, decay=None / "routed" / "gdn", seed=seed); Adam 1e-3 on
        every parameter (S53's recipe), evaluation every 1200, early stop after 3 >= 0.95; statistics as S53's.
  D_DELTA_{ROUTED,GDN}   S54(d)'s D_DELTA (LOCAL3 + SLOW on RandHeaderTask, learned β) with the decay module added
        (its parameters in the rest group), via explore_f_window_delta_child's builder rule.
After each run: the decay at the end — mean α per layer at CTX / KEY / VAL positions of the probe (per channel for k = 2;
for the oracle, the channel of the token's own stream and the other channel separately) — and the write-order accuracy.
"""

import json
import time

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

import explore_delta_mem as dm
import explore_far_cue as s6
import explore_f_tasks as ft
import explore_f_window_delta_child as wdc
import explore_window_recipe_child as s43c
import test_binding_onset as tbo
import test_short_conv as tsc
from test_multilayer_binding import build, probe_batch
from test_channel_binding import ARCH, arms as cb_arms
from test_router_reliability import run_one

LR = 1e-3
H8 = s6.HeaderTask(8, S=2, n_vals=16, n_q=1)
DECAYS = {"FIXED": None, "ROUTED": "routed", "GDN": "gdn"}


def parse(arm_name):
    if arm_name.startswith("D_DELTA_"):
        return dict(kind="D", decay=DECAYS[arm_name.split("_")[-1]])
    kind, d = arm_name.split("_")
    return dict(kind=kind, decay=DECAYS[d])


def builder_h8(kind, decay):
    def b(a, seed):
        def make():
            return dm.to_delta(build(H8, a, ARCH, seed), beta=1.0, normalize=True, decay=decay, seed=seed)
        return make
    return b


def builder_d(decay):
    def b(a, seed):
        def make():
            m = wdc.local_model(ft.RandHeaderTask(), a, seed)
            return dm.to_delta(m, beta="learned", normalize=True, decay=decay, seed=seed)
        return make
    return b


def setup(arm_name, holder):
    arm = parse(arm_name)
    if arm["kind"] == "D":
        task = wdc.RH
        kw = dict(task=task, grad_fn=tsc.conv_grad_norms, lr=LR, builder=builder_d(arm["decay"]), stats_fn=wdc.STATS_RH,
                  check=wdc.switch_check(holder), run_kw=dict(param_groups=s43c.groups(holder)))
        return s43c.A, task, kw
    a = tsc.ARM["B"] if arm["kind"] == "SINGLE" else {x["key"]: x for x in cb_arms(H8)}["ceiling"]
    kw = dict(task=H8, grad_fn=tsc.conv_grad_norms, lr=LR, builder=builder_h8(arm["kind"], arm["decay"]),
              stats_fn=s6.header_stats if arm["kind"] == "ORACLE" else tsc.conv_stats)
    return a, H8, kw


@torch.no_grad()
def decay_by_role(m, task, seed):
    """Mean α per layer at CTX / KEY / VAL: {'own'|'other'|'ch0'..: {role: value}}; and β likewise."""
    pinp, plab, roles = probe_batch(task, seed)
    out = []

    def hook(mod, args, kwargs, o):
        V = kwargs["V"][:, 0]
        gr, gw = mod._gates if not isinstance(mod._gates[0], (list, tuple)) else (mod._gates[0][0], mod._gates[1][0])
        la = mod.log_alpha(V, gw)
        k = gw.shape[-1]
        al = torch.full(gw.shape, float(mod.decay)) if la is None else torch.exp(la)
        row = {}
        if k == 2 and m.gate_kind == "perfect":
            own = al.gather(-1, plab.clamp(min=0).unsqueeze(-1))[..., 0]
            oth = al.gather(-1, (1 - plab.clamp(min=0)).unsqueeze(-1))[..., 0]
            row["own"] = {r: float(own[mask].mean()) for r, mask in roles.items()}
            row["other"] = {r: float(oth[mask].mean()) for r, mask in roles.items()}
        else:
            for c in range(k):
                row[f"ch{c}"] = {r: float(al[..., c][mask].mean()) for r, mask in roles.items()}
        b = mod.beta_of(V)
        row["beta"] = {r: float(b[mask].mean()) for r, mask in roles.items()}
        out.append(row)
    was = m.training
    m.eval()
    h = m.attn.register_forward_hook(hook, with_kwargs=True)
    try:
        m(pinp)
    finally:
        h.remove()
        m.train(was)
    return out


def decay_params(m):
    d = m.attn.decay_mod
    if d is None:
        return None
    return {n: q.detach().tolist() for n, q in d.named_parameters() if q.numel() <= 64}


def run(p):
    arm_name, seed, iters = p["arm"], p["seed"], p["iters"]
    holder = {}
    a, task, kw = setup(arm_name, holder)
    keep = {"at": -1}
    rec = run_one(a, seed, iters, p.get("eval_every", tbo.EVAL_EVERY), keep=keep, **kw)
    if rec.get("ok"):
        m = keep["model"]
        rec.update(arm=arm_name, delta_cfg=m.delta_cfg, decay_by_role=decay_by_role(m, task, seed),
                   decay_params=decay_params(m), decay_init=getattr(m.attn.decay_mod, "init", None),
                   write_order=ft.write_order_acc(m, task, tbo.eval_batch(task)), group_names=holder.get("names"))
        if parse(arm_name)["kind"] == "D":
            import explore_common as ec
            rec["fail"] = None if rec["transition"] is not None else ec.fail_class(rec)
            rec["tag"] = ec.tag(rec)
    return json.loads(json.dumps(rec, default=float))


def checks(p):
    rows = []
    for arm_name in ("ORACLE_FIXED", "ORACLE_ROUTED", "ORACLE_GDN", "SINGLE_FIXED", "SINGLE_ROUTED", "SINGLE_GDN",
                     "D_DELTA_ROUTED", "D_DELTA_GDN"):
        a, task, kw = setup(arm_name, {})
        m = kw["builder"](a, 350)()
        arm = parse(arm_name)
        ok = (isinstance(m.attn, dm.DeltaMemory) and m.attn.decay_kind == (arm["decay"] or "fixed")
              and (m.attn.beta_fixed == 1.0 if arm["kind"] != "D" else m.attn.beta_mode == "learned") and m.attn.normalize
              and m.n_ch == (1 if arm["kind"] == "SINGLE" else 2) and m.conv is None)
        rows.append((f"{arm_name}: k {m.n_ch}, memory {type(m.attn).__name__}, decay {m.attn.decay_kind}, beta "
                     f"{m.attn.beta_fixed if m.attn.beta_mode == 'fixed' else 'learned'}, parameters added "
                     f"{dm.delta_params(m)}", ok))
    # the three oracle arms share every non-decay initial parameter; at init ROUTED equals FIXED exactly where g = 1
    ms = {d: builder_h8("ORACLE", DECAYS[d])({x["key"]: x for x in cb_arms(H8)}["ceiling"], 350)() for d in DECAYS}
    p0 = dict(ms["FIXED"].named_parameters())
    share = all(torch.equal(q, p0[n]) for d in ("ROUTED", "GDN") for n, q in ms[d].named_parameters() if n in p0)
    x = H8.make_batch(8, torch.Generator().manual_seed(2))[0][:, :-1]
    for m in ms.values():
        m.eval()
    with torch.no_grad():
        lf, lr_ = ms["FIXED"](x)[0], ms["ROUTED"](x)[0]
    d = float((lf - lr_).abs().max())
    rows.append((f"the oracle arms share every initial parameter but the decay's ({share}); at init the routed decay gives "
                 f"0.95 on the written channel and 1 on the other: logits differ from FIXED's (max {d:.3g} > 0: the "
                 f"unwritten channel no longer decays)", share and d > 0))
    # D arm: the decay parameters are in the rest group
    for d in ("ROUTED", "GDN"):
        lrs, ids, keep, holder, rec = _lrs_d(f"D_DELTA_{d}")
        m = keep["model"]
        named = dict(m.named_parameters())
        rest = set(ids["groups"][1])
        rows.append((f"D_DELTA_{d}: the memory's parameters {dm.delta_params(m)} are in the rest group; lrs at 1 {lrs[0]}, "
                     f"2401 {lrs[-1]}", all(id(named[n]) in rest for n in dm.delta_params(m)) and lrs[0] == [LR, LR / 10]
                     and lrs[-1] == [LR, LR]))
    return [[n, bool(v)] for n, v in rows]


def _lrs_d(arm_name, iters=2401):
    from torch.optim.optimizer import register_optimizer_step_pre_hook
    lrs, ids = [], {}

    def hook(opt, args, kwargs):
        if not ids:
            ids["groups"] = [[id(q) for q in g["params"]] for g in opt.param_groups]
        lrs.append([g["lr"] for g in opt.param_groups])
    saved = (tbo.evaluate, tbo.logits_of)
    h = register_optimizer_step_pre_hook(hook)
    tbo.evaluate = lambda model, task, data: (0.5, 1.0)
    tbo.logits_of = lambda model, x: (torch.zeros(x.shape[0], x.shape[1], model.lm_head.shape[1]) + model.lm_head.sum() * 0.0)
    keep = {"at": iters}
    holder = {}
    try:
        a, task, kw = setup(arm_name, holder)
        rec = run_one(a, 350, iters, keep=keep, **kw)
    finally:
        tbo.evaluate, tbo.logits_of = saved
        h.remove()
    return lrs, ids, keep, holder, rec


def timing(p):
    stamps = []
    h = register_optimizer_step_post_hook(lambda o, a, k: stamps.append(time.perf_counter()))
    try:
        run(dict(arm=p["arm"], seed=0, iters=p["steps"], eval_every=10 ** 9))
    finally:
        h.remove()
    dd = sorted(y - x for x, y in zip(stamps, stamps[1:]))
    a, task, kw = setup(p["arm"], {})
    m = kw["builder"](a, 0)()
    t0 = time.time()
    tbo.evaluate(m, task, tbo.eval_batch(task))
    kw["stats_fn"](m, task, probe_batch(task, 0), 2400)
    return [dd[len(dd) // 2], time.time() - t0]
