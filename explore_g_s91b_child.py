#!/usr/bin/env python
"""
explore_g_s91b_child.py — EXPLORATORY, not a result. Run code of Session G's screen S91b (explore_g_s91b): the register
gate with k = 16 channels (one per context type) on RandHeaderTask-ctxpool and -allpool. Runs INSIDE a child process on
this branch's own modules, 1 torch thread. It imports S89's child (the tasks, the hinges, the task-aware statistics) and
S89H's child (the Hebbian + conv model) unchanged.

TASKS (explore_f_tasks.py vendored from claude/explore-F at 9017b82, blob cfe3fe2; its first 49,923 bytes are b0c6332's
e24d5e1, asserted): "ctxpool" = ft.RandHeaderTaskCtxPool() (lite's 4 key types kept; the 2 contexts drawn from 16 types;
stream index by first appearance; vocabulary 36) and "allpool" = ft.RandHeaderTaskAllPool() (keys from 32 types too).
MODELS (k = K = 16 channels; the register gate shared by the three layers, SG, S56's register; memory "H" Hebbian + conv
'layer' as S89H, "D" delta beta 1 / L2 as S89; built as S89H's / S89's make_model with the arm's n_ch set to K, CHECKed equal
to them at k = 2):
  REG3, REG3_KEYHINGE (the key term: task.body_positions, task.key_index), REG3_REPEATHINGE (S89's repeat hinge).
  PIDX   the per-sequence-index perfect gate: one-hot of task.stream_labels (the stream by first appearance) into K channels
         (two used); uniform 1/K before the first context.
  PTYPE  the type-level perfect gate: one-hot of the latched context TYPE into K = 16 channels (each type its own channel).
STATISTICS: explore_f_tasks.ctx_stats(ROUTE_AT) (pool_stats; at S = 2 the within-sequence eta^2 forms; and inv_from_gate_k:
  sep_key_inv, seq_separated_frac, val_cos_inv_k at any k) + beta^R by role + explore_g_s89_child.layer_stats89. At the end:
  type_map_k = explore_b18_gates.eta2_pool of the returned gate's K-vector at the body's KEY positions grouped by the block's
  context type (pooled over sequences): ~1 for a gate that is a fixed function of the context type, ~0 for the index.
RECIPE: SLOW (the gate group at 1e-3; the rest 1e-4 to 2400, 1e-3 after; the perfect gates' gate group empty), 24000 updates.
"""

import json

import torch
import torch.nn.functional as F

import explore_common as ec
import explore_common2 as c2
import explore_b16_gates as g16
import explore_b18_gates as g18
import explore_delta_mem as dm
import explore_f_tasks as ft
import explore_g_common as gc
import explore_g_regmodel as rm
import explore_g_register_child as s56
import explore_g_rhl_child as rc
import explore_g_devices_child as dc
import explore_g_s87_child as s87
import explore_g_s89_child as c89
import explore_g_s89h_child as h89
import test_binding_onset as tbo
import test_short_conv as tsc
from bdh import BDH
from test_channel_binding import ARCH
from test_multilayer_binding import MultiBDH, build, probe_batch
from test_instrument_v2 import TAU_END

K = 16
F_BLOB, PREFIX = "cfe3fe276f7c8c1592b469f2efaa8a0074a88f19", ("e24d5e10815eb1234d1947aa6b74b128d2696882", 49923)
c89.TASKS.setdefault("ctxpool", ft.RandHeaderTaskCtxPool())
CTX_STATS = ft.ctx_stats(c2.ROUTE_AT)
LEARNED = ("REG3", "REG3_KEYHINGE", "REG3_REPEATHINGE")
KIND = {"REG3": "reg3", "REG3_KEYHINGE": "key", "REG3_REPEATHINGE": "repeat", "PIDX": "index", "PTYPE": "type"}


def parse(arm):
    tname, rest = arm.split(".", 1)
    name, mem = rest.rsplit("_", 1)
    return tname, name, mem


def latched_ctx(task, x):
    B, T = x.shape
    idx = torch.arange(T).expand(B, T)
    last = torch.where(task.ctx_mask(x), idx, torch.full_like(idx, -1)).cummax(dim=1).values
    return x.gather(1, last.clamp(min=0)), last >= 0


class PerfK(MultiBDH):
    """The two perfect gates at K channels (module docstring); delta: the gate is passed by set_gates."""

    def forward(self, tokens, tau=None, track_sat=False):
        t = self.pt_task
        k = self.n_ch
        if self.pt_mode == "index":
            lab = t.stream_labels(tokens)
            ok = lab >= 0
            g = F.one_hot(lab.clamp(min=0), k).to(self.lm_head.dtype)
        else:
            ctype, ok = latched_ctx(t, tokens)
            g = F.one_hot(ctype.clamp(0, k - 1), k).to(self.lm_head.dtype)
        g = torch.where(ok.unsqueeze(-1), g, torch.full_like(g, 1.0 / k))
        if hasattr(self.attn, "set_gates"):
            self.attn.set_gates(g, g)
        self.attn.G = torch.einsum("btk,bsk->bts", g, g).unsqueeze(1)
        logits = self.forward_conv(tokens) if self.conv is not None else BDH.forward(self, tokens)[0]
        return logits, None, g, g


def _arm(kind, task, mem, k):
    a = dict(c89.ceil_arm(task) if kind in ("index", "type") else ec.ARM_A, n_ch=k)
    return dict(a, model_kw=tsc.LAYER) if mem == "H" else a


def make_model(arm, seed, k=K, on=True):
    tname, name, mem = parse(arm)
    task = c89.task_of(tname)
    kind = KIND[name]
    m = build(task, _arm(kind, task, mem, k), ARCH, seed)
    if kind in ("index", "type"):
        if mem == "D":
            dm.to_delta(m, **s56.DELTA_KW)
        m.__class__ = PerfK
        m.pt_task, m.pt_mode = task, kind
        return m
    m = g16.to_local(m, seed)
    cv = m.conv
    m.conv = None                                   # to_reg asserts a conv-free LocalBDH; the conv is restored below
    rm.to_reg(m, seed, sg=True, reg=True, gated=(1, 2, 3), res_layers=(1, 2, 3), mem="hebb" if mem == "H" else "delta")
    m.conv = cv
    m.__class__ = h89.Reg3Conv
    m.inp = "window"
    if mem == "D":
        s56.delta_attn(m, seed)
    if kind == "reg3":
        return m
    m.__class__ = h89.TermConv
    m.t_kind, m.t_on, m.t_task = kind, bool(on), tname
    m.t_state = dict(n=0, fired=0, weights={}, value_at={}, count=True)
    return m


def stats_k(model, task, probe, step):
    out = CTX_STATS(model, task, probe, step)
    out["beta"] = c89.write_rate89(model, task, probe)
    if step in rc.LAYER_AT:
        out["layers"] = c89.layer_stats89(model, task, probe)
    return out


@torch.no_grad()
def type_map_k(model, task, probe):
    pinp = probe[0]
    was = model.training
    model.eval()
    try:
        g = model(pinp, TAU_END)[2]
    finally:
        model.train(was)
    kpos, _ = task.body_positions(pinp)
    ctype, _ = latched_ctx(task, pinp)
    gk = g.gather(1, kpos.unsqueeze(-1).expand(-1, -1, g.shape[-1]))
    return float(g18.eta2_pool(gk, ctype.gather(1, kpos), task.n_ctx))


def run(p):
    """p: arm '<task>.<NAME>_<H|D>', seed, iters, k (default 16), off, legacy (S89's statistics: the k = 2 CHECK), eval_every."""
    arm, seed = p["arm"], p["seed"]
    tname, name, mem = parse(arm)
    task = c89.task_of(tname)
    kind = KIND[name]
    k = p.get("k", K)
    a = dict(_arm(kind, task, mem, k), key=arm)
    holder, keep = {}, {"at": -1}
    rec = run_one(a, seed, p.get("iters", rc.ITERS), p.get("eval_every", tbo.EVAL_EVERY), task=task,
                  stats_fn=c89.stats_fn89 if p.get("legacy") else stats_k, grad_fn=tsc.conv_grad_norms, lr=dc.LR,
                  builder=lambda a_, s_: (lambda: make_model(arm, s_, k=k, on=not p.get("off"))),
                  run_kw=dict(param_groups=dc.groups(holder, dc.LR)), check=dc.switch(holder, 0), keep=keep)
    if not rec.get("ok"):
        return json.loads(json.dumps(rec, default=float))
    m = keep["model"]
    ev = tbo.eval_batch(task)
    rec.update(arm=arm, task=tname, k=k, recipe="slow", conv=m.conv, group_names=holder.get("names"),
               stream_acc=c89.stream_acc89(m, task, ev), probe=c89.probe_rh89(m, task, gc.probe_data(task)),
               acc_by_cell=c89.acc_by_cell89(m, task, ev), write_order=ft.write_order_acc_idx(m, task, ev),
               type_map_k=type_map_k(m, task, probe_batch(task, seed)))
    if getattr(m, "t_state", None) is not None:
        rec["term"] = {kk: v for kk, v in m.t_state.items() if kk != "count"}
    if kind not in ("index", "type"):
        rec["fail"] = None if rec["transition"] is not None else ec.fail_class(rec)
    return json.loads(json.dumps(rec, default=float))


from test_router_reliability import run_one          # noqa: E402  (after the definitions, as the other children)


# ── CHECKs ───────────────────────────────────────────────────────────────────
def _x(task, n=16, seed=5):
    return task.make_batch(n, torch.Generator().manual_seed(seed))[0][:, :-1]


def checks_static(p):
    import os
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    rows = []
    path = os.path.join(here, "explore_f_tasks.py")
    h = subprocess.check_output(["git", "-C", here, "hash-object", path]).decode().strip()
    data = open(path, "rb").read()
    pre = subprocess.check_output(["git", "-C", here, "cat-file", "blob", PREFIX[0]])
    rows.append((f"explore_f_tasks.py is F's 9017b82 blob ({h[:12]} vs {F_BLOB[:12]}); its first {PREFIX[1]} bytes are b0c6332's "
                 f"e24d5e1 ({data[:PREFIX[1]] == pre})", h == F_BLOB and len(pre) == PREFIX[1] and data[:PREFIX[1]] == pre))
    tc = c89.task_of("ctxpool")
    for nm, v in ft.check_ctx_task(tc):
        rows.append((f"ctxpool (F's check_ctx_task): {nm}", v))
    for tname in ("ctxpool", "allpool"):
        for nm, v in ft.check_inv_stats_k(c89.task_of(tname)):
            rows.append((f"{tname} (F's check_inv_stats_k): {nm}", v))
    rows.append((f"ctxpool: L {tc.L}, vocab {tc.vocab}, {tc.n_ctx} context types, {tc.k_pool} key types, S {tc.S}; the type-level "
                 f"cap at k = 16 is {ft.type_map_cap(16, 2, 16)[0]:.3f}, at k = 2 {ft.type_map_cap(16, 2, 2)[0]:.3f}",
                 tc.L == 23 and tc.vocab == 36 and ft.type_map_cap(16, 2, 16)[0] == 1.0
                 and abs(ft.type_map_cap(16, 2, 2)[0] - 64 / 120) < 1e-12))
    # at k = 2 the builder is S89H's / S89's (parameters and logits bit for bit)
    t = c89.task_of("allpool")
    x = _x(t)
    for arm, ref in (("allpool.REG3_H", h89.make_model("REG3_H", 2994, "allpool")),
                     ("allpool.REG3_REPEATHINGE_H", h89.make_model("REG3_REPEATHINGE_H", 2994, "allpool")),
                     ("allpool.REG3_D", c89.make_model("REG3_D", 2994, "allpool")),
                     ("allpool.PIDX_H", h89.make_model("PERFECT_H", 2994, "allpool")),
                     ("allpool.PIDX_D", c89.make_model("ORACLE_D", 2994, "allpool"))):
        m = make_model(arm, 2994, k=2).eval()
        ref.eval()
        same_p = all(torch.equal(u, w) for u, w in zip(m.state_dict().values(), ref.state_dict().values())) \
            and list(m.state_dict()) == list(ref.state_dict())
        with torch.no_grad():
            eq = torch.equal(m(x)[0], ref(x)[0])
        rows.append((f"k = 2: {arm} built here equals S89H's / S89's model (parameters {same_p}, logits {eq})", same_p and eq))
    # k = 16: shapes; the perfect gates read ROUTED_k untrained; type_map_k separates them
    for tname in ("ctxpool", "allpool"):
        t = c89.task_of(tname)
        pr = probe_batch(t, 2992)
        for mem in ("H", "D"):
            m = make_model(f"{tname}.REG3_{mem}", 2994)
            rows.append((f"{tname}, {mem}: REG3 at k = 16: W_g2 {tuple(m.W_g2.shape)}, n_ch {m.n_ch}, conv {m.conv}",
                         m.W_g2.shape[0] == K and m.n_ch == K and (m.conv == "layer") == (mem == "H")))
            for g in ("PIDX", "PTYPE"):
                mm = make_model(f"{tname}.{g}_{mem}", 2992)
                st = stats_k(mm, t, pr, "end")
                tm = type_map_k(mm, t, pr)
                ok = st["margin"] >= 0.999 and st["seq_separated_frac"] > 0.999 and \
                    (tm > 0.999 if g == "PTYPE" else tm < 0.1)
                rows.append((f"{tname}, {mem}: the untrained {g} reads ROUTED_k (margin {st['margin']:.3f}, sequences separated "
                             f"{st['seq_separated_frac']:.3f}); type_map_k {tm:.3f} ({'~1' if g == 'PTYPE' else '~0'})", ok))
    # ROUTED_k agrees with the k = 2 within-sequence eta^2 form on synthetic gates (allpool)
    t = c89.task_of("allpool")
    pr = probe_batch(t, 2992)
    pinp = pr[0]
    gates = dict(relative=t.perfect_gate(pinp), type_balanced=ft.type_level_gate(t, pinp),
                 random=torch.rand(*pinp.shape, 2, generator=torch.Generator().manual_seed(4)).softmax(-1))
    agree = []
    for nm, gr in gates.items():
        a = ft.inv_from_gate(t, pinp, gr)
        b = ft.inv_from_gate_k(t, pinp, gr)
        agree.append((nm, a["eta_key_by_stream_inv"] > 0.9, b["seq_separated_frac"] > 0.9))
    rows.append((f"allpool, k = 2 synthetic gates: within-sequence eta^2 > 0.9 vs sequences separated > 0.9 agree {agree}",
                 all(x_ == y_ for _, x_, y_ in agree) and agree[0][1] and not agree[1][1]))
    # the key term at k = 16 on ctxpool reads the 8 body keys (4 types)
    t = c89.task_of("ctxpool")
    x = _x(t, 64)
    kpos, kid, G = c89.key_pos_ids(t, x)
    rows.append((f"ctxpool: the key term's positions are task.key_mask's 8 body positions, ids in [0, {G})",
                 G == 4 and bool((kid >= 0).all() and (kid < 4).all()) and kpos.shape[1] == 8))
    return [[n, bool(v)] for n, v in rows]


def checks_repro(p):
    """S89H's / S89's allpool.REG3_REPEATHINGE_<M>|2842 through this run() at k = 2 with S89's statistics, bit for bit to 2400."""
    mem = p["mem"]
    store = "s89h_results.json" if mem == "H" else "s89_results.json"
    ref, cpu = c89.record(store, f"allpool.REG3_REPEATHINGE_{mem}", 2842)
    r = run(dict(arm=f"allpool.REG3_REPEATHINGE_{mem}", seed=2842, iters=2400, k=2, legacy=True))
    return [[f"{store[:-13].upper()}'s allpool.REG3_REPEATHINGE_{mem}|2842 (on {cpu}) reproduces bit for bit through 2400 through "
             f"S91b's run() at k = 2: curve {r['curve']}", s87.trunc(r) == s87.trunc(ref)]]


def checks_off(p):
    tname, mem = p["task"], p["mem"]
    ref = run(dict(arm=f"{tname}.REG3_{mem}", seed=2994, iters=2400))
    rows = []
    for name in ("REG3_KEYHINGE", "REG3_REPEATHINGE"):
        r = run(dict(arm=f"{tname}.{name}_{mem}", seed=2994, iters=2400, off=True))
        rows.append([f"{tname}, {mem}, k = 16: {name} with its switch off equals REG3|2994 through 2400 bit for bit (curve "
                     f"{r['curve']})", s87.trunc(r) == s87.trunc(ref)])
    return rows


def checks_on(p):
    tname, mem = p["task"], p["mem"]
    rows = []
    for name in LEARNED + ("PIDX", "PTYPE"):
        r = run(dict(arm=f"{tname}.{name}_{mem}", seed=2994, iters=20, eval_every=10))
        e = r.get("end", {})
        fired = r.get("term", {}).get("fired")
        rows.append([f"{tname}, {mem}: {name} at k = {r.get('k')} runs 20 updates end to end (ok {r.get('ok')}; "
                     f"seq_separated_frac {e.get('seq_separated_frac')}; type_map_k {r.get('type_map_k')}; term fired {fired})",
                     bool(r.get("ok")) and r.get("k") == K and "seq_separated_frac" in e and "margin" in e
                     and r.get("type_map_k") is not None and (fired is None or fired > 0)])
    return rows
