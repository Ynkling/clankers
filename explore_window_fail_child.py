#!/usr/bin/env python
"""
explore_window_fail_child.py — EXPLORATORY, not a result. S49's run code. It runs INSIDE a child process on this branch's
own modules (no main-line hook), as S43's. Parent: explore_window_fail.

THE RUNS: S43's LOCAL3_SLOW (explore_window_recipe_child: batch 1's LocalGateBDH with W_h frozen, S5's SLOW schedule —
gate W_in, W_g, window at 1e-3 throughout, every other trainable parameter 1e-4 for updates 1-2400 and 1e-3 after —
test_short_conv's TASK (S=2, P=4, k=2, grouped, no convolution), 24000 updates, stats explore_common2.stats_b2, S38's
decoder at every evaluation), rerun on the seeds S43 left unbound, with the same keyword arguments and two additions
that do not touch training: the model is recorded when built, and an optimizer step post-hook measures the gate after
updates DENSE (every 25 through 1200, every 100 through 2400, then every 1200 to the end) on the run's own probe
(test_multilayer_binding.probe_batch(task, seed): the batch test_router_layout.routing_stats and fail_class use).
CHECK: the rerun reproduces S43's record bit for bit (curve and statistics; the parent compares every run).

MEASURED at each of those updates (cells): test_router_layout.routing_stats (eta^2 of the gate's channel-0 probability
p at the body's KEY (3j+1) and VAL (3j+2) positions by stream / key / half / triple index, and the query routing margin);
and, at KEY and at VAL positions, the mean p in each (stream, key) cell (2 x 4), the share of soft gates (0.1 < p < 0.9),
eta^2 of p by (stream, key) cell and (VAL) by value token. LOCAL3's read and write gates are the same function of the
window (gr = gw), so p at VAL positions is the write gate of every body pair.
"""

import json

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

import explore_common as ec
import explore_common2 as c2
import explore_window_recipe_child as s43c
import explore_b16_gates as g16
import test_binding_onset as tbo
import test_short_conv as tsc
from test_router_layout import routing_stats, eta2, fail_class
from test_router_reliability import run_one
from test_multilayer_binding import probe_batch
from test_instrument_v2 import TAU_END

DENSE = tuple(range(25, 1201, 25)) + tuple(range(1300, 2401, 100))
ROLES = (("key", 1), ("val", 2))


def at_steps(iters):
    return sorted(set(t for t in DENSE if t <= iters) | set(range(1200, iters + 1, 1200)))


def table(p, x, task):
    """From the channel-0 probability p (B, T) and the tokens x (B, T): per role, the mean p in each (stream, key) cell,
    the share of soft gates, eta^2 by cell and (VAL) by value token."""
    S, P, n = task.S, task.P, task.S * task.P
    j = torch.arange(n)
    ctx, key, val = x[:, 3 * j], x[:, 3 * j + 1] - S, x[:, 3 * j + 2]
    out = {}
    for role, off in ROLES:
        pp = p[:, 3 * j + off]
        out[f"cell_{role}"] = [[round(float(pp[(ctx == s) & (key == k)].mean()), 4) for k in range(P)] for s in range(S)]
        out[f"soft_{role}"] = round(float(((pp > 0.1) & (pp < 0.9)).float().mean()), 4)
        out[f"eta_{role}_by_cell"] = eta2(pp, ctx * P + key)
        if role == "val":
            out["eta_val_by_value"] = eta2(pp, val)
    return out


@torch.no_grad()
def cells(m, task, probe):
    out = routing_stats(m, task, probe)
    was = m.training
    m.eval()
    try:
        gr = m(probe[0], TAU_END)[2]
    finally:
        m.train(was)
    out.update(table(gr[..., 0], probe[0], task))
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()}


def run(p):
    """p: seed, iters, eval_every (timing only), dense (default True)."""
    seed, iters = p["seed"], p["iters"]
    task = ec.TASK
    holder, out, dense, box = {}, {}, {}, {}
    data = g16.probe_data(task)
    probe = probe_batch(task, seed)
    want = set(at_steps(iters)) if p.get("dense", True) else set()
    b0 = s43c.builder(True)

    def builder(a, s):
        mk = b0(a, s)

        def make():
            m = mk()
            box["m"] = m
            if want:
                dense["0"] = cells(m, task, probe)
            return m
        return make
    n = [0]

    def post(opt, args, kwargs):
        n[0] += 1
        if n[0] in want:
            dense[str(n[0])] = cells(box["m"], task, probe)
    h = register_optimizer_step_post_hook(post)
    try:
        kw = dict(check=s43c.make_check(True, holder, out, data), task=task, grad_fn=tsc.conv_grad_norms, lr=s43c.LR,
                  builder=builder, stats_fn=c2.stats_b2, run_kw=dict(param_groups=s43c.groups(holder)))
        rec = run_one(s43c.A, seed, iters, p.get("eval_every", tbo.EVAL_EVERY), **kw)
    finally:
        h.remove()
    if rec.get("ok"):
        rec.update(arm="LOCAL3_SLOW_RERUN", decode=out, dense=dense, n_updates=n[0], group_names=holder.get("names"),
                   fail=None if rec["transition"] is not None else ec.fail_class(rec), tag=ec.tag(rec))
    return json.loads(json.dumps(rec, default=float))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def checks(p):
    rows = []
    task = ec.TASK
    S, P, n = task.S, task.P, task.S * task.P
    # (1) the cell table on a synthetic gate: stream 0 always on channel 0; stream 1 on channel 0 for keys 0, 1 and on
    # channel 1 for keys 2, 3 (a gate that splits the streams on two of the four keys)
    x = task.make_batch(64, torch.Generator().manual_seed(9))[0][:, :-1]
    j = torch.arange(n)
    ctx, key = x[:, 3 * j], x[:, 3 * j + 1] - S
    pv = torch.full(x.shape, 0.5)
    for off in (1, 2):
        pv[:, 3 * j + off] = torch.where((ctx == 1) & (key >= 2), torch.tensor(0.0), torch.tensor(1.0))
    t = table(pv, x, task)
    want = [[1.0] * P, [1.0, 1.0, 0.0, 0.0]]
    rows.append((f"the (stream, key) cell table on a synthetic gate (stream 0 on channel 0; stream 1 on channel 0 for keys 0, 1 and "
                 f"on channel 1 for keys 2, 3): KEY {t['cell_key']}, VAL {t['cell_val']}; soft share {t['soft_val']}; eta^2 by cell "
                 f"{t['eta_val_by_cell']:.3f} (1 = p fixed by the cell)",
                 t["cell_key"] == want and t["cell_val"] == want and t["soft_val"] == 0.0 and abs(t["eta_val_by_cell"] - 1) < 1e-6))
    # (2) cells' routing statistics are test_router_layout.routing_stats on the run's probe
    import explore_local_gate as s1
    m = g16.freeze_wh(s1.make_model(s43c.A, 163))
    pr = probe_batch(task, 163)
    c = cells(m, task, pr)
    rs = routing_stats(m, task, pr)
    rows.append(("cells() carries test_router_layout.routing_stats unchanged (the eta^2 and margin fail_class reads)",
                 all(abs(c[k] - round(v, 5)) < 1e-9 for k, v in rs.items())))
    # (3) the measurement hook leaves training untouched: a 2400-update rerun of seed 163 equals the same run without it
    r1 = run(dict(seed=163, iters=2400))
    r0 = s43c.run(dict(arm="LOCAL3_SLOW", seed=163, iters=2400))
    same = r1["curve"] == r0["curve"] and r1["stats"] == r0["stats"] and r1["decode"] == r0["decode"]
    rows.append((f"the dense measurements are inert: LOCAL3_SLOW|163 through 2400 with them equals S43's run code without them "
                 f"(curve {r1['curve']}; statistics and decoder equal {same}); {len(r1['dense'])} measurements at "
                 f"{sorted(int(k) for k in r1['dense'])[:5]}..{max(int(k) for k in r1['dense'])}, {r1['n_updates']} updates",
                 same and len(r1["dense"]) == len(at_steps(2400)) + 1 and r1["n_updates"] == 2400))
    return [[nm, bool(v)] for nm, v in rows]


def timing(p):
    import time
    out = s43c.timing(dict(which=["LOCAL3_SLOW"], steps=p["steps"]))["LOCAL3_SLOW"]
    import explore_local_gate as s1
    m = g16.freeze_wh(s1.make_model(s43c.A, 0))
    pr = probe_batch(ec.TASK, 0)
    t0 = time.time()
    for _ in range(5):
        cells(m, ec.TASK, pr)
    return {"LOCAL3_SLOW_RERUN": out + [(time.time() - t0) / 5]}


def fail_of(rec):
    return fail_class(rec)
