#!/usr/bin/env python
"""
explore_f_delta_controls_child.py — EXPLORATORY, not a result. S53's run code (parent: explore_f_delta_controls). Runs
INSIDE a child process on this branch's own modules (no main-line hook), 1 torch thread.

Models: test_multilayer_binding.build(task, arm, ARCH, seed) with test_short_conv's arm B (k = 1, gate 'none') or
test_channel_binding's 'ceiling' arm (the perfect gate, k = S), then explore_delta_mem.to_delta for the DELTA arms.
Training: test_router_reliability.run_one (onset_run: Adam, batch 32, evaluation every 1200, early stop after 3 >= 0.95)
with lr 1e-3 passed explicitly; statistics test_short_conv.conv_stats (k = 1, and k = 2 on the grouped layout:
test_router_layout.layout_stats) or explore_far_cue.header_stats (k = 2 on the header layouts); gradient norms
test_short_conv.conv_grad_norms. After the run: explore_f_tasks.write_order_acc on eval_batch(task), and (delta arms)
β at CTX / KEY / VAL per layer on the probe.
"""

import json
import time

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

import explore_delta_mem as dm
import explore_far_cue as s6
import explore_f_tasks as ft
import test_binding_onset as tbo
import test_short_conv as tsc
from test_multilayer_binding import build, probe_batch
from test_channel_binding import ARCH, arms as cb_arms, task_for
from test_router_reliability import run_one

LR = 1e-3
TASKS = {"grouped": tsc.TASK, "header": s6.HEADER, "header8": s6.HeaderTask(8, S=2, n_vals=16, n_q=1)}
DELTA = {"SINGLE_DELTA": dict(beta=1.0, normalize=True), "SINGLE_DELTA_L": dict(beta="learned", normalize=True),
         "ORACLE_DELTA": dict(beta=1.0, normalize=True)}


def arm_def(kind, task):
    if kind.startswith("SINGLE"):
        return tsc.ARM["B"]
    return {a["key"]: a for a in cb_arms(task)}["ceiling"]


def builder(kind):
    def b(a, seed):
        def make():
            m = build(b.task, a, ARCH, seed)
            if kind in DELTA:
                m = dm.to_delta(m, seed=seed, **DELTA[kind])
            return m
        return make
    return b


def stats_fn(kind, layout):
    if kind.startswith("ORACLE") and layout != "grouped":
        return s6.header_stats
    return tsc.conv_stats


@torch.no_grad()
def beta_by_role(m, task, probe):
    """Mean β per layer at CTX / KEY / VAL positions of the probe (delta memory with learned or fixed β)."""
    pinp, _, roles = probe
    out = []

    def hook(mod, args, kwargs, o):
        b = mod.beta_of(kwargs["V"][:, 0])
        out.append({r: float(b[mask].mean()) for r, mask in roles.items()})
    was = m.training
    m.eval()
    h = m.attn.register_forward_hook(hook, with_kwargs=True)
    try:
        m(pinp)
    finally:
        h.remove()
        m.train(was)
    return out


def run(p):
    kind, layout, seed, iters = p["kind"], p["layout"], p["seed"], p["iters"]
    task = TASKS[layout]
    a = arm_def(kind, task)
    b = builder(kind)
    b.task = task
    keep = {"at": -1}
    rec = run_one(a, seed, iters, p.get("eval_every", tbo.EVAL_EVERY), task=task, stats_fn=stats_fn(kind, layout),
                  grad_fn=tsc.conv_grad_norms, lr=LR, builder=b, keep=keep)
    if rec.get("ok"):
        m = keep["model"]
        rec.update(kind=kind, layout=layout, k=a["n_ch"], gate=a["gate"],
                   memory="delta" if kind in DELTA else "hebb", delta_cfg=getattr(m, "delta_cfg", None),
                   write_order=ft.write_order_acc(m, task, tbo.eval_batch(task)))
        if kind in DELTA:
            rec["beta_by_role"] = beta_by_role(m, task, probe_batch(task, seed))
    rec.pop("snap", None)
    return json.loads(json.dumps(rec, default=float))


def checks(p):
    rows = []
    for layout, task in TASKS.items():
        models = {}
        for kind in ("SINGLE_HEBB", "SINGLE_DELTA", "SINGLE_DELTA_L", "ORACLE_HEBB", "ORACLE_DELTA"):
            a = arm_def(kind, task)
            b = builder(kind)
            b.task = task
            models[kind] = b(a, 300)()
        sh, sd, sl, oh, od = (models[k] for k in ("SINGLE_HEBB", "SINGLE_DELTA", "SINGLE_DELTA_L", "ORACLE_HEBB",
                                                  "ORACLE_DELTA"))
        kinds = (sh.n_ch == sd.n_ch == sl.n_ch == 1 and oh.n_ch == od.n_ch == task.S and sh.gate_kind == "none"
                 and oh.gate_kind == "perfect" and all(m.conv is None for m in models.values())
                 and type(sh.attn).__name__ == "GatedAttention" and type(oh.attn).__name__ == "GatedAttention"
                 and isinstance(sd.attn, dm.DeltaMemory) and isinstance(od.attn, dm.DeltaMemory)
                 and sd.attn.beta_fixed == 1.0 and sd.attn.normalize and sl.attn.beta_mode == "learned"
                 and od.attn.beta_fixed == 1.0 and sd.attn.impl == "parallel")
        ps = dict(sh.named_parameters())
        po = dict(oh.named_parameters())
        same_s = all(torch.equal(q, ps[n]) for n, q in sd.named_parameters()) and set(dict(sd.named_parameters())) == set(ps)
        same_l = all(torch.equal(q, ps[n]) for n, q in sl.named_parameters() if n in ps)
        same_o = all(torch.equal(q, po[n]) for n, q in od.named_parameters()) and set(dict(od.named_parameters())) == set(po)
        rows.append((f"{layout}: models as stated (k, gate, memory, beta, no conv); SINGLE_DELTA and SINGLE_HEBB share every "
                     f"initial parameter ({same_s}); SINGLE_DELTA_L shares them and adds attn.w_b, attn.b_b ({same_l}); "
                     f"ORACLE_DELTA and ORACLE_HEBB share every initial parameter ({same_o})",
                     kinds and same_s and same_l and same_o))
    t8 = TASKS["header8"]
    rows.append((f"P = 8 header task: L = {t8.L}, qpos = {t8.qpos}, vocab {t8.vocab}",
                 t8.L == 2 * 17 + 3 and t8.qpos == [35] and t8.vocab == 2 + 8 + 16))
    return [[n, bool(v)] for n, v in rows]


def timing(p):
    stamps = []
    h = register_optimizer_step_post_hook(lambda o, a, k: stamps.append(time.perf_counter()))
    try:
        run(dict(kind=p["kind"], layout=p["layout"], seed=0, iters=p["steps"], eval_every=10 ** 9))
    finally:
        h.remove()
    dd = sorted(y - x for x, y in zip(stamps, stamps[1:]))
    task = TASKS[p["layout"]]
    a = arm_def(p["kind"], task)
    b = builder(p["kind"])
    b.task = task
    m = b(a, 0)()
    t0 = time.time()
    tbo.evaluate(m, task, tbo.eval_batch(task))
    stats_fn(p["kind"], p["layout"])(m, task, probe_batch(task, 0), 1200)
    return [dd[len(dd) // 2], time.time() - t0]
