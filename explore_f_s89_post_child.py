#!/usr/bin/env python
"""
explore_f_s89_post_child.py — EXPLORATORY, not a result. Run code of Session F's S89-F follow-up pilots (parent:
explore_f_s89_post, whose docstring holds the hypothesis and the readings). Runs INSIDE a child process on this branch's
own modules (no main-line hook), 1 torch thread. Every arm is explore_f_s89_tasks_child's perfect-gate pilot (O_HEBB_<T>:
the same task objects, model construction, SLOW recipe, statistics and run_one) with one change:
  O_TYPE_HEBB_ALLPOOL   the gate is explore_f_tasks.type_level_gate with BALANCED_MAP (CTX types 0-7 -> channel 0, 8-15
                        -> channel 1) in place of task.perfect_gate: TypeGateBDH, AllpoolPerfectBDH's forward with that
                        gate. Referee V's fix 2: the type-level perfect gate on ALLPOOL, at the 64/120 routing cap.
  O_HEBB_CONV_<T>       the arm carries test_short_conv.LAYER (conv 'layer', K = 4, depthwise causal, on each layer's input
                        to the memory, the input it writes and reads with), as the main line's grouped-layout arms carry
                        it (test_scale_axes' ceiling4k4 / test_stream_recipe's ceiling8k16: the perfect gate + LAYER). The
                        conv weights are in the rest group (1e-4 to 2400, then 1e-3).
"""

import json

import torch

import explore_f_s89_tasks_child as s89
import explore_f_tasks as ft
import test_binding_onset as tbo
import test_short_conv as tsc
from bdh import BDH
from test_multilayer_binding import build
from test_channel_binding import ARCH
from test_router_reliability import run_one


class TypeGateBDH(s89.AllpoolPerfectBDH):
    """AllpoolPerfectBDH with the type-level perfect gate (explore_f_tasks.type_level_gate, BALANCED_MAP)."""

    def forward(self, tokens, tau=None, track_sat=False):
        gr = gw = ft.type_level_gate(self.pool_task, tokens, ft.BALANCED_MAP, self.lm_head.dtype)
        if hasattr(self.attn, "set_gates"):
            self.attn.set_gates(gr, gw)
        self.attn.G = torch.einsum("btk,bsk->bts", gr, gw).unsqueeze(1)
        logits = BDH.forward(self, tokens)[0] if self.conv is None else self.forward_conv(tokens)
        return logits, None, gr, gw


ARMS = {"O_TYPE_HEBB_ALLPOOL": dict(task="ALLPOOL", gate="type", conv=False),
        "O_HEBB_CONV_POOL": dict(task="POOL", gate="perfect", conv=True),
        "O_HEBB_CONV_ALLPOOL": dict(task="ALLPOOL", gate="perfect", conv=True)}


def spec(arm):
    d = ARMS[arm]
    sp = dict(kind="ORACLE", task=d["task"], memory="HEBB")
    a = dict(s89.arm_def(sp))
    if d["conv"]:
        a = dict(a, model_kw=tsc.LAYER)
    return d, sp, a


def make_model(arm, a, seed):
    d, sp, _ = spec(arm)
    m = s89.make_model(sp, a, seed)
    if d["gate"] == "type":
        m.__class__ = TypeGateBDH
    return m


def run_kw(arm, holder, seed):
    _, sp, _ = spec(arm)
    kw = s89.run_kw(sp, holder, seed)
    kw["builder"] = lambda a_, s_: (lambda: make_model(arm, a_, s_))
    return kw


def run(p):
    arm, seed, iters = p["arm"], p["seed"], p["iters"]
    d, sp, a = spec(arm)
    task = s89.TASKS[sp["task"]]
    holder, keep = {}, {"at": -1}
    rec = run_one(a, seed, iters, p.get("eval_every", tbo.EVAL_EVERY), keep=keep, **run_kw(arm, holder, seed))
    if rec.get("ok"):
        m = keep["model"]
        rec.update(arm=arm, kind="ORACLE", task=sp["task"], memory="HEBB", gate=d["gate"], conv=m.conv, k=a["n_ch"],
                   group_names=holder.get("names"))
        rec["write_order"] = ft.write_order_acc(m, task, tbo.eval_batch(task))
        rec["outcome_pooled"] = ft.pool_outcome(rec, invariant=False)
        rec["outcome_inv"] = ft.pool_outcome(rec, invariant=True)
    return json.loads(json.dumps(rec, default=float))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def _lrs(arm, seed=2800, iters=s89.WARM + 1):
    from torch.optim.optimizer import register_optimizer_step_pre_hook
    lrs, ids = [], {}

    def hook(opt, args, kwargs):
        if not ids:
            ids["n"] = [len(g["params"]) for g in opt.param_groups]
        lrs.append([g["lr"] for g in opt.param_groups])
    saved = (tbo.evaluate, tbo.logits_of)
    h = register_optimizer_step_pre_hook(hook)
    tbo.evaluate = lambda model, task, data: (0.5, 1.0)
    tbo.logits_of = lambda model, x: (torch.zeros(x.shape[0], x.shape[1], model.lm_head.shape[1]) + model.lm_head.sum() * 0.0)
    holder, keep = {}, {"at": iters}
    try:
        kw = run_kw(arm, holder, seed)
        kw["stats_fn"] = s89.single_stats
        rec = run_one(spec(arm)[2], seed, iters, keep=keep, **kw)
    finally:
        tbo.evaluate, tbo.logits_of = saved
        h.remove()
    return lrs, ids, holder, rec


def checks(p):
    rows = []
    for arm in ARMS:
        d, sp, a = spec(arm)
        task = s89.TASKS[sp["task"]]
        x = task.make_batch(8, torch.Generator().manual_seed(3))[0][:, :-1]
        m = make_model(arm, a, 2800)
        ref = s89.make_model(sp, s89.arm_def(sp), 2800)                    # the S89 pilot O_HEBB_<T>
        m.eval()
        ref.eval()
        with torch.no_grad():
            lg, _, gr, gw = m(x)
            lr_ = ref(x)[0]
        if d["gate"] == "type":
            want = ft.type_level_gate(task, x, ft.BALANCED_MAP)
            same = all(torch.equal(p1, p2) for (n1, p1), (n2, p2) in zip(m.named_parameters(), ref.named_parameters()))
            ok = torch.equal(gr, want) and torch.equal(gw, want) and same and m.conv is None
            rows.append((f"{arm}: the gate is the balanced type-level gate (one-hot, k = 2) at every position ({ok}); the "
                         f"parameters are the S89 pilot O_HEBB_ALLPOOL's at seed 2800 ({same}); no conv", ok))
        else:
            pg = task.perfect_gate(x)
            names = {n for n, _ in m.named_parameters()} - {n for n, _ in ref.named_parameters()}
            pr = dict(ref.named_parameters())
            same = all(torch.equal(p1, pr[n1]) for n1, p1 in m.named_parameters() if n1 in pr)
            idc = torch.equal(lg, lr_)
            ok = (torch.equal(gr, pg) and m.conv == "layer" and m.short_conv.conv_w.shape[-1] == tsc.CONV_K
                  and names == {"short_conv.conv_w"} and same and idc)
            rows.append((f"{arm}: the gate is task.perfect_gate ({torch.equal(gr, pg)}); conv '{m.conv}', K "
                         f"{m.short_conv.conv_w.shape[-1]}; parameters beyond the S89 pilot's: {sorted(names)}, the shared "
                         f"ones equal the S89 pilot's at seed 2800 ({same}); at initialisation (the conv at the identity) "
                         f"the logits equal the S89 pilot's bit for bit ({idc})", ok))
        lrs, ids, holder, rec = _lrs(arm)
        n = len(lrs)
        W = s89.WARM
        ok = (n == W + 1 and lrs[0] == [s89.LR, s89.LR_WARM] and lrs[W - 1] == [s89.LR, s89.LR_WARM]
              and lrs[W] == [s89.LR, s89.LR] and ids["n"][0] == 0 and bool(rec.get("ok"))
              and (("short_conv.conv_w" in holder["names"][1]) == d["conv"]))
        rows.append((f"{arm}: groups {holder['names'][0]} / {len(holder['names'][1])} rest parameters (the conv in the rest "
                     f"group: {'short_conv.conv_w' in holder['names'][1]}); lrs at 1 {lrs[0]}, {W} {lrs[W - 1]}, {W + 1} "
                     f"{lrs[W]}", ok))
    return [[n, bool(v)] for n, v in rows]


def timing(p):
    import time
    from torch.optim.optimizer import register_optimizer_step_post_hook
    stamps = []
    h = register_optimizer_step_post_hook(lambda o, a, k: stamps.append(time.perf_counter()))
    try:
        run(dict(arm=p["arm"], seed=0, iters=p["steps"], eval_every=10 ** 9))
    finally:
        h.remove()
    dd = sorted(y - x for x, y in zip(stamps, stamps[1:]))
    d, sp, a = spec(p["arm"])
    task = s89.TASKS[sp["task"]]
    m = make_model(p["arm"], a, 0)
    t0 = time.time()
    tbo.evaluate(m, task, tbo.eval_batch(task))
    return [dd[len(dd) // 2], 2 * (time.time() - t0)]
