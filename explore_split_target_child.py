#!/usr/bin/env python
"""
explore_split_target_child.py — EXPLORATORY, not a result. S37's code that runs INSIDE a child process on
the main line's module tree at explore_main9c.MAIN_SHA (9c5939e), as S36's (explore_split_child, imported
read-only for the operation, the probe, the Adam capture and the trigger wiring). Parent:
explore_split_target.

Arms (S=4, P=4, k=4, conv; test_stream_recipe.run_attempt with ARM["A4k4"], 28800 steps):
  SPLITK_M    S33's A_HINGE (explore_muon_scale_child.make_rc "HINGE": S25's groups at Muon lr 0.005,
              main's hinge) + the trigger + KEYMASS targeting.
  HINGE4k4_A  main's A4k4 + test_slow_start's HINGE recipe (make_recipe(True, TAU): Adam, the slow
              groups, the hinge); no trigger.
  SPLITK_A    HINGE4k4_A + the trigger + KEYMASS targeting (onset_run's Adam built through S36's capturing
              factory, the same constructor).
THE TRIGGER: S36's plateau rule (the 64-sequence probe from a generator seeded 12345; at updates 2400
(reference), 4800, 7200, ... through 26400, after the evaluation's statistics and the run path's check:
fire if the probe accuracy is < 0.95 and it rose by < 0.02 since the previous check), with at most 2 splits
per run and none within 4800 updates of a previous split (a check that meets the rule but is capped is
logged as blocked).
KEYMASS: c* = the channel with the largest mean read-gate mass over the probe's key positions 3j+1, j = 0 ..
S*P-1 (the positions test_slow_start's hinge uses), c0 = the smallest there. The all-position rule (S36's:
the mean over every position) is computed and logged at every check, with the labelled map
(test_scale_axes.routing_k's ch_map on the run's probe) before any split.
THE OPERATION: S36's (S24's): W_g[c0] = w + n2, W_g[c*] = w + n1 (w = W_g[c*], noise 0.1 std(w), generator
12_000_000 + 1000 x seed + split count), then W_g's optimizer state zeroed (Muon: the momentum buffer; Adam:
exp_avg, exp_avg_sq, step).
"""

import json

import torch

import explore_main9c as mt
import explore_muon_scale_child as s33c
import explore_split_child as s36c
import explore_muon_recipe as s25
import test_binding_onset as tbo
import test_scale_axes as tsa
import test_slow_start as tss
import test_stream_recipe as tsr
from test_multilayer_binding import probe_batch
from test_instrument_v2 import TAU_END

EVERY, FIRST, REF_AT = s36c.EVERY, s36c.FIRST, s36c.REF_AT          # 2400, 4800, 2400
ACC_THR, RISE = s36c.ACC_THR, s36c.RISE                             # 0.95, 0.02
MAX_SPLITS, GAP = 2, 4800
TOTAL = 28800
ARM = {"SPLITK_M": dict(opt="muon", trigger=True),
       "HINGE4k4_A": dict(opt="adam", trigger=False),
       "SPLITK_A": dict(opt="adam", trigger=True)}


@torch.no_grad()
def read_gate(m, data):
    was = m.training
    m.eval()
    try:
        return m(data[:, :-1], TAU_END)[2]
    finally:
        m.train(was)


def select(gr, S, P):
    """(KEYMASS c*, c0, key masses), (all-position c*, c0, masses) from a read gate gr (B, T, k)."""
    j = torch.arange(S * P)
    km = gr[:, 3 * j + 1].mean((0, 1))
    am = gr.mean((0, 1))
    return (int(km.argmax()), int(km.argmin()), km), (int(am.argmax()), int(am.argmin()), am)


def targeting(cmap, cs, c0):
    """Labelled: does (c*, c0) copy a channel holding >= 2 streams onto one holding none?"""
    return cmap.count(cs) >= 2 and cmap.count(c0) == 0


def make_trigger(task, seed, info, opt_of, last, thr=ACC_THR, rise=RISE):
    data = s36c.probe_data(task)
    rprobe = probe_batch(task, seed)
    st = dict(prev=None, n=0, last=None)

    def trig(m, step):
        if step % EVERY or step < REF_AT or step > last:
            return
        acc, _ = tbo.evaluate(m, task, data)
        (kc, k0, km), (ac, a0, am) = select(read_gate(m, data), task.S, task.P)
        cmap = tsa.routing_k(m, task, rprobe)["ch_map"]
        row = dict(step=step, acc=acc, prev=st["prev"], map_before=cmap,
                   key_masses=[round(float(v), 5) for v in km], all_masses=[round(float(v), 5) for v in am],
                   key_cs=kc, key_c0=k0, all_cs=ac, all_c0=a0,
                   key_ok=targeting(cmap, kc, k0), all_ok=targeting(cmap, ac, a0), fired=False, blocked=False)
        eligible = step >= FIRST and st["prev"] is not None and acc < thr and acc - st["prev"] < rise
        capped = st["n"] >= MAX_SPLITS or (st["last"] is not None and step - st["last"] < GAP)
        row["eligible"] = eligible
        if eligible and capped:
            row["blocked"] = True
        elif eligible:
            res = s36c.split_op(m, opt_of(), kc, k0, s36c.NOISE_BASE + 1000 * seed + st["n"])
            st["n"] += 1
            st["last"] = step
            row.update(fired=True, map_after=tsa.routing_k(m, task, rprobe)["ch_map"],
                       key_masses_after=[round(float(v), 5) for v in select(read_gate(m, data), task.S, task.P)[0][2]],
                       **res)
        st["prev"] = acc
        info["checks"].append(row)
    return trig


def run(p):
    """p: arm, seed, iters, mlr (SPLITK_M), thr (CHECKs only)."""
    arm = ARM[p["arm"]]
    holder, info = {}, dict(checks=[])
    cfg = "a"
    a = s33c.cfg_arm(cfg, "HINGE")
    task = s33c.cfg_task(cfg)
    if arm["opt"] == "muon":
        rc, _ = s33c.make_rc("HINGE", p["mlr"], holder)
        ctx, opt_of = s25.muon_in_onset_run(holder), (lambda: holder["opts"][0].muon)
    else:
        rc, _ = tss.make_recipe(True, tss.TAU)
        ctx, opt_of = s36c.adam_capture(holder), (lambda: holder["opts"][0])
    if arm["trigger"]:
        rc = s36c.with_trigger(rc, make_trigger(task, p["seed"], info, opt_of, TOTAL - EVERY, thr=p.get("thr", ACC_THR)))
        with ctx:
            rec = s33c.path(cfg, a, p["seed"], p["iters"], rc)
    else:
        rec = s33c.path(cfg, a, p["seed"], p["iters"], rc)                  # main's own HINGE recipe, unwrapped
    if rec.get("ok"):
        rec.update(arm=p["arm"], cfg=cfg, opt=arm["opt"], muon_lr=p.get("mlr") if arm["opt"] == "muon" else None,
                   checks=info["checks"], splits=sum(1 for c in info["checks"] if c["fired"]),
                   maps=s33c.maps_of(rec), main_sha=mt.MAIN_SHA)
        try:
            rec["outcome"] = tsr.outcome(rec)
        except Exception as e:
            rec["outcome"] = f"n/a ({type(e).__name__})"
    return json.loads(json.dumps(rec, default=float))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def synthetic(p):
    """A synthetic read gate whose key-position map is 2+1+1 (streams 0, 1 on channel 2, stream 2 on 0,
    stream 3 on 1, channel 3 empty at key positions) and whose other positions put most mass on channel 3
    (the trap S36's all-position rule fell into): KEYMASS must pick c* = 2 (shared), c0 = 3 (empty)."""
    S, P, k, B = 4, 4, 4, 16
    T = 3 * S * P + 2
    chan = {0: 2, 1: 2, 2: 0, 3: 1}
    gr = torch.zeros(B, T, k)
    gr[:, :, 3] = 0.85
    gr[:, :, :3] = 0.05
    g = torch.Generator().manual_seed(5)
    streams = torch.randint(0, S, (B, S * P), generator=g)
    for b in range(B):
        for j in range(S * P):
            row = torch.full((k,), 0.02)
            row[chan[int(streams[b, j])]] = 0.94
            gr[b, 3 * j + 1] = row
    (kc, k0, km), (ac, a0, am) = select(gr, S, P)
    cmap = [chan[s] for s in range(S)]
    ok = (kc, k0) == (2, 3) and targeting(cmap, kc, k0) and not targeting(cmap, ac, a0)
    return [[f"on a synthetic gate whose key-position map is 2+1+1 ({cmap}; channel 3 empty at key positions, "
             f"0.85 of the mass elsewhere) KEYMASS picks c* {kc}, c0 {k0} (key masses "
             f"{[round(float(v), 3) for v in km]}): the shared channel and the empty one; the all-position rule picks "
             f"c* {ac}, c0 {a0} (masses {[round(float(v), 3) for v in am]}), a mistarget", bool(ok)]]


# ── timing ───────────────────────────────────────────────────────────────────
def timing(p):
    """The median interval between optimizer steps over p["steps"] steps of each arm (no check in the window)
    and the evaluation time; run one child per arm at once for the pool's load."""
    import time
    import torch.optim.optimizer as topt
    from test_channel_binding import eval_batch
    out = {}
    for arm_name in p["which"]:
        arm = ARM[arm_name]
        stamps = []

        def post(opt, args, kwargs):
            if arm["opt"] == "adam" or isinstance(opt, s25.SliceMuon):
                stamps.append(time.perf_counter())
        h = topt.register_optimizer_step_post_hook(post)
        try:
            run(dict(arm=arm_name, seed=0, iters=p["steps"], mlr=p["mlr"]))
        finally:
            h.remove()
        d = sorted(y - x for x, y in zip(stamps, stamps[1:]))
        a = s33c.cfg_arm("a", "HINGE")
        task = s33c.cfg_task("a")
        m = tsr.builder_for(a)(a, 0)()
        t0 = time.time()
        tbo.evaluate(m, task, eval_batch(task))
        tsa.scale_stats(m, task, probe_batch(task, 0), 1200)
        out[arm_name] = [d[len(d) // 2], time.time() - t0]
    return out
