#!/usr/bin/env python
"""
explore_f_configs_child.py — EXPLORATORY, not a result. S80 (a)'s run code (parent: explore_f_configs, whose docstring
holds the readings). Runs INSIDE a child process on this branch's own modules (no main-line hook), 1 torch thread, as
S54(a).

  HEBB_A, DELTA_A   S43's LOCAL3_SLOW run path (explore_window_recipe_child: builder(True), groups, make_check with S38's
                    decoder, stats explore_common2.stats_b2) on test_short_conv.TASK (grouped S = 2, P = 4, k = 2, no
                    conv); DELTA_A's built model converted by explore_f_delta_fast.to_delta_fast with DELTA* (β = 1, L2
                    keys, tied write, decay 0.95; no new parameter).
  O_DELTA_A, O_HEBB_A   the perfect gate (test_channel_binding's 'ceiling', k = 2) on the same task, one Adam at 1e-3 on
                    every parameter, stats test_short_conv.conv_stats (S54(a)'s A_ORACLE_* arms).
At the end of every run: S77's probe (Session G's S55, explore_f_probe_common) on the grouped layout's body positions,
on the trained model and on the same seed's model at initialisation.
"""

import json
import os
import time

import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook, register_optimizer_step_post_hook

import explore_common as ec
import explore_common2 as c2
import explore_delta_mem as dm
import explore_f_delta_fast as ff
import explore_f_probe_common as pc
import explore_b16_gates as g16
import explore_window_recipe_child as s43c
import test_binding_onset as tbo
import test_short_conv as tsc
from test_multilayer_binding import build
from test_channel_binding import ARCH, arms as cb_arms
from test_router_reliability import run_one

HERE = os.path.dirname(os.path.abspath(__file__))
LR, LR_WARM, WARM = s43c.LR, s43c.LR_WARM, s43c.WARM
TASK = ec.TASK
STAR = os.path.join(HERE, "explore_out", "F", "eight_delta_star.json")
DELTA_KW = dict(beta=1.0, normalize=True)            # DELTA*; asserted equal to the committed choice in checks()
ARM = {
    "HEBB_A": dict(gate="local", delta=False),
    "DELTA_A": dict(gate="local", delta=True),
    "O_DELTA_A": dict(gate="perfect", delta=True),
    "O_HEBB_A": dict(gate="perfect", delta=False),
}
REC_FILE = os.path.join(HERE, "explore_out", "window_recipe_results.json")
REC_KEY = "LOCAL3_SLOW|160|46867e792241|Intel(R) Xeon(R) Processor @ 2.10GHz"
REC_CPU = "Intel(R) Xeon(R) Processor @ 2.10GHz"


def convert(m, seed, impl="fast"):
    if impl == "hebb":
        return dm.to_delta(m, seed=seed, impl="hebb", **DELTA_KW)
    return ff.to_delta_fast(m, seed=seed, **DELTA_KW)


def builder(arm_name, impl="fast"):
    arm = ARM[arm_name]

    def b(a, seed):
        def make():
            m = build(TASK, a, ARCH, seed) if arm["gate"] == "perfect" else s43c.builder(True)(a, seed)()
            return convert(m, seed, impl) if arm["delta"] else m
        return make
    return b


def arm_def(arm_name):
    if ARM[arm_name]["gate"] == "perfect":
        return {a["key"]: a for a in cb_arms(TASK)}["ceiling"]
    return s43c.A


def run_kw(arm_name, holder, out, impl="fast"):
    kw = dict(task=TASK, grad_fn=tsc.conv_grad_norms, lr=LR, builder=builder(arm_name, impl))
    if ARM[arm_name]["gate"] == "perfect":
        kw["stats_fn"] = tsc.conv_stats
        return kw
    kw.update(run_kw=dict(param_groups=s43c.groups(holder)), stats_fn=c2.stats_b2,
              check=s43c.make_check(True, holder, out, g16.probe_data(TASK)))
    return kw


# ── S77's probe on the grouped layout ────────────────────────────────────────
def grouped_positions(task):
    """KEY positions of the S·P body triples [CTX, KEY, VAL] (3j + 1), VAL one later, and each triple's index j."""
    n = task.S * task.P
    kpos = torch.tensor([3 * j + 1 for j in range(n)])
    return kpos, kpos + 1, torch.arange(n)


def probe_grouped(model, task, data):
    """explore_f_probe_common.probe line for line, with the grouped layout's positions in place of the header's."""
    x, tr, te = data
    feats, logits, ref = pc.residuals(model, x)
    kpos, vpos, within = grouped_positions(task)
    lab = task.stream_labels(x)
    out = dict(logits_equal=bool((logits == ref).all()))
    for name, f in feats.items():
        for role, pos in (("K", kpos), ("V", vpos)):
            X = f[:, pos].double().numpy()
            y = lab[:, pos].numpy()
            Xtr, ytr = X[tr.numpy()].reshape(-1, X.shape[-1]), y[tr.numpy()].reshape(-1)
            Xte, yte = X[te.numpy()].reshape(-1, X.shape[-1]), y[te.numpy()].reshape(-1)
            hit, tracc = pc.fit_score(Xtr, ytr, Xte, yte)
            k, n = int(hit.sum()), int(hit.size)
            hit2 = hit.reshape(len(te), -1)
            by_i = {int(i): float(hit2[:, (within == i).numpy()].mean()) for i in within.unique()}
            out[f"{name}|{role}"] = dict(acc=k / n, k=k, n=n, wilson=pc.wilson(k, n), train_acc=tracc, by_index=by_i)
    return out


# ── A run ────────────────────────────────────────────────────────────────────
def run(p):
    arm_name, seed, iters = p["arm"], p["seed"], p["iters"]
    arm = ARM[arm_name]
    impl = p.get("impl", "fast")
    holder, out = {}, {}
    keep = {"at": -1}
    kw = run_kw(arm_name, holder, out, impl)
    rec = run_one(arm_def(arm_name), seed, iters, p.get("eval_every", tbo.EVAL_EVERY), keep=keep, **kw)
    if rec.get("ok"):
        m = keep["model"]
        rec.update(arm=arm_name, gate=arm["gate"], memory="delta" if arm["delta"] else "hebb",
                   delta_cfg=getattr(m, "delta_cfg", None), decode=out or None, group_names=holder.get("names"),
                   discovered=ec.discovered(rec), tag=ec.tag(rec) if arm["gate"] == "local" else None,
                   fail=None if (arm["gate"] == "perfect" or rec["transition"] is not None) else ec.fail_class(rec))
        if p.get("probe", True):
            data = pc.probe_data(TASK)
            rec["probe"] = probe_grouped(m, TASK, data)
            rec["probe_init"] = probe_grouped(builder(arm_name, impl)(arm_def(arm_name), seed)(), TASK, data)
    return json.loads(json.dumps(rec, default=float))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def _lrs(arm_name, seed, iters):
    """Each group's lr at every update through the real run path (loss and evaluation stubbed)."""
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
        kw = run_kw(arm_name, holder, {})
        kw["check"] = s43c.make_check(True, holder, None, None)
        rec = run_one(arm_def(arm_name), seed, iters, keep=keep, **kw)
    finally:
        tbo.evaluate, tbo.logits_of = saved
        h.remove()
    return lrs, ids, keep, holder, rec


def checks(p):
    import numpy as np
    rows = []
    star = json.load(open(STAR))
    rows.append((f"DELTA* = {star['arm']} (beta {star['beta']}, L2 keys {star['normalize']}) as committed; this child uses "
                 f"{DELTA_KW}", star["beta"] == DELTA_KW["beta"] and star["normalize"] == DELTA_KW["normalize"]))
    # (1) the recorded run, the new code path disabled, and the Hebbian arm itself
    cpu = ec.cpu_model()
    if cpu == REC_CPU:
        rec = json.load(open(REC_FILE))["runs"][REC_KEY]
        for arm_name, impl in (("HEBB_A", "fast"), ("DELTA_A", "hebb")):
            t0 = time.time()
            r = run(dict(arm=arm_name, seed=160, iters=2400, impl=impl, probe=False))
            want = [c for c in rec["curve"] if c[0] <= 2400]
            sn = [s for s in r["stats"] if s["step"] != "end" and s["step"] <= 2400]
            so = [s for s in rec["stats"] if s["step"] != "end" and s["step"] <= 2400]
            dec = all(r["decode"][s] == rec["decode"][s] for s in ("1200", "2400"))
            gk = [k for k in rec["grad"] if int(k) <= 2400]
            gr = bool(gk) and all(r["grad"].get(k) == rec["grad"][k] for k in gk)
            rows.append((f"{arm_name} (memory impl {impl}) on seed 160 reproduces batch 16's {REC_KEY} through 2400 bit for "
                         f"bit: curve {r['curve']} vs {want}; statistics equal {sn == so}; decoder at 1200/2400 equal {dec}; "
                         f"gradient norms at {gk} equal {gr} ({time.time() - t0:.0f} s)",
                         r["curve"] == want and sn == so and dec and gr))
    else:
        rows.append((f"recorded LOCAL3_SLOW|160 NOT APPLICABLE on {cpu}", True))
    # (2) the models: DELTA_A = HEBB_A with only the memory changed
    a = s43c.A
    mh, md = builder("HEBB_A")(a, 2000)(), builder("DELTA_A")(a, 2000)()
    ph, pd = dict(mh.named_parameters()), dict(md.named_parameters())
    same = set(ph) == set(pd) and all(torch.equal(pd[n], ph[n]) for n in ph)
    rows.append((f"DELTA_A = HEBB_A's model (S43's LOCAL3_SLOW) with the memory converted: same {len(ph)} parameters, equal "
                 f"({same}), none added; memory {type(md.attn).__name__} beta {md.attn.beta_fixed} ({md.attn.beta_mode}), L2 "
                 f"keys {md.attn.normalize}, decay {md.attn.decay}, write {md.attn.write}; W_h frozen {not md.W_h.requires_grad}",
                 same and isinstance(md.attn, ff.FastDeltaMemory) and md.attn.beta_mode == "fixed" and md.attn.beta_fixed == 1.0
                 and md.attn.normalize and md.attn.decay == 0.95 and not md.W_h.requires_grad))
    # (3) groups and lrs through the real path, 2401 updates
    for arm_name in ("HEBB_A", "DELTA_A"):
        lrs, ids, keep, holder, rec = _lrs(arm_name, 2000, WARM + 1)
        m = keep["model"]
        named = dict(m.named_parameters())
        train = {id(q) for q in m.parameters() if q.requires_grad}
        flat = [i for g in ids.get("groups", []) for i in g]
        cover = len(flat) == len(set(flat)) and set(flat) == train
        gate_ok = set(ids["groups"][0]) == {id(named[n]) for n in s43c.GATE_L}
        n = len(lrs)
        lr_ok = n == WARM + 1 and lrs[0] == [LR, LR_WARM] and lrs[WARM - 1] == [LR, LR_WARM] and lrs[WARM] == [LR, LR]
        rows.append((f"{arm_name}: two groups cover every trainable parameter once ({cover}); gate group = W_in, W_g, window "
                     f"({gate_ok}); lrs at 1 {lrs[0] if n else '?'}, {WARM} {lrs[WARM - 1] if n >= WARM else '?'}, "
                     f"{WARM + 1} {lrs[WARM] if n > WARM else '?'}", cover and gate_ok and lr_ok and bool(rec.get("ok"))))
    # (4) the probe: G's residual loop on the k = 2 models; positions and labels; the split; G's synthetic check
    x = TASK.make_batch(32, torch.Generator().manual_seed(3))[0][:, :-1]
    for arm_name in ARM:
        m = builder(arm_name)(arm_def(arm_name), 2000)()
        feats, logits, ref = pc.residuals(m, x)
        rows.append((f"{arm_name}: G's residual loop gives the model's own logits bit for bit ({type(m).__name__}, "
                     f"{type(m.attn).__name__}, k {m.n_ch})", torch.equal(logits, ref) and set(feats) == {"emb", "L1", "L2", "L3", "final"}))
    xd, tr, te = pc.probe_data(TASK)
    kpos, vpos, within = grouped_positions(TASK)
    lab = TASK.stream_labels(xd)
    ctx = xd[:, kpos - 1]
    rows.append((f"probe on the grouped layout: KEY positions {kpos.tolist()} hold keys, VAL positions hold values, the "
                 f"label at both is the preceding CTX token; 1600 / 400 sequences, disjoint",
                 bool(((xd[:, kpos] >= TASK.S) & (xd[:, kpos] < TASK.S + TASK.P)).all())
                 and bool((xd[:, vpos] >= TASK.S + TASK.P).all()) and bool((ctx < TASK.S).all())
                 and torch.equal(lab[:, kpos], ctx) and torch.equal(lab[:, vpos], ctx)
                 and len(tr) == 1600 and len(te) == 400 and not set(tr.tolist()) & set(te.tolist())))
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    Xs = rng.normal(size=(4000, 8)); Xs[:, 0] += 3 * (2 * y - 1)
    hit, _ = pc.fit_score(Xs[:3200], y[:3200], Xs[3200:], y[3200:])
    Xr = rng.normal(size=(4000, 8))
    hit_r, _ = pc.fit_score(Xr[:3200], y[:3200], Xr[3200:], y[3200:])
    rows.append((f"G's synthetic probe check: informative feature {hit.mean():.4f} (G 0.998) >= 0.99; random "
                 f"{hit_r.mean():.3f} in [0.4, 0.6]", hit.mean() >= 0.99 and 0.4 <= hit_r.mean() <= 0.6))
    # (5) the perfect gate on DELTA*: one-hot, equal to the Hebbian model's
    mo, mh0 = builder("O_DELTA_A")(arm_def("O_DELTA_A"), 2000)(), builder("O_HEBB_A")(arm_def("O_HEBB_A"), 2000)()
    mo.eval(), mh0.eval()
    with torch.no_grad():
        g, g0 = mo(x)[2], mh0(x)[2]
    rows.append((f"the perfect gate on DELTA* (k {mo.n_ch}): read gate one-hot ({bool(((g == 0) | (g == 1)).all())}) and equal to "
                 f"the Hebbian model's ({torch.equal(g, g0)})", torch.equal(g, g0) and bool(((g == 0) | (g == 1)).all())
                 and isinstance(mo.attn, ff.FastDeltaMemory)))
    return [[n, bool(v)] for n, v in rows]


def timing(p):
    stamps = []
    h = register_optimizer_step_post_hook(lambda o, a, k: stamps.append(time.perf_counter()))
    try:
        run(dict(arm=p["arm"], seed=0, iters=p["steps"], eval_every=10 ** 9, probe=False))
    finally:
        h.remove()
    dd = sorted(y - x for x, y in zip(stamps, stamps[1:]))
    m = builder(p["arm"])(arm_def(p["arm"]), 0)()
    from test_multilayer_binding import probe_batch
    kw = run_kw(p["arm"], {}, {})
    t0 = time.time()
    tbo.evaluate(m, TASK, tbo.eval_batch(TASK))
    kw["stats_fn"](m, TASK, probe_batch(TASK, 0), 2400)
    if ARM[p["arm"]]["gate"] == "local":
        g16.decode(m, TASK, g16.probe_data(TASK))
    return [dd[len(dd) // 2], time.time() - t0]
