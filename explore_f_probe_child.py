#!/usr/bin/env python
"""
explore_f_probe_child.py — EXPLORATORY, not a result. S77's run code (parent: explore_f_probe). Runs INSIDE a child process
on this branch's own modules (no main-line hook), 1 torch thread.

Each arm reruns a recorded run through its screen's own run path, line for line (the run_one call of that screen's
child), asserts that the curve equals the record bit for bit, then fits Session G's S55 probe (explore_f_probe_common,
copied unchanged from claude/explore-G at 3932db7) on the trained model and on the same seed's model at initialisation.
  S53_DELTA   S53's SINGLE_DELTA_H (explore_f_delta_controls_child, kind SINGLE_DELTA, layout header): one channel, delta
              β = 1, L2 keys; explore_far_cue.HEADER (P = 4, S = 2); 24000. Seeds 300-309; record F/delta_controls,
              code 429221e60279.
  S59_GDN     S59's SINGLE_GDN (explore_f_decay_child): one channel, delta β = 1, L2 keys, GDN decay; HeaderTask(8); 48000.
              Seeds 350-359; record F/decay, code 55653f2b2a4e.
  S69_GDN     S69's S69_GDN (explore_f_followup_child): one HEBBIAN channel (β = 0, write 1, raw keys) with the GDN decay;
              HeaderTask(8); 48000. Seeds 350-359 (the same batches as S59_GDN); record F/hebb_decay, code ff908bdbc0b2.
"""

import json
import os

import torch

import explore_common as ec
import explore_f_probe_common as pc
import explore_f_delta_controls_child as c53
import explore_f_decay_child as c59
import explore_f_followup_child as c69
import test_binding_onset as tbo
import test_short_conv as tsc
from test_router_reliability import run_one

HERE = os.path.dirname(os.path.abspath(__file__))
REC = {"S53_DELTA": ("F/delta_controls", "SINGLE_DELTA_H", "429221e60279"),
       "S59_GDN": ("F/decay", "SINGLE_GDN", "55653f2b2a4e"),
       "S69_GDN": ("F/hebb_decay", "S69_GDN", "ff908bdbc0b2")}
CPU = "Intel(R) Xeon(R) Processor @ 2.10GHz"


def record(arm, seed):
    store, rarm, sha = REC[arm]
    return ec.load_store(store)["runs"][f"{rarm}|{seed}|{sha}|{CPU}"]


def setup(arm, iters):
    """(arm dict, task, run_one kwargs, iters, builder-of-seed) — each screen child's own run_one call."""
    if arm == "S53_DELTA":
        task = c53.TASKS["header"]
        a = c53.arm_def("SINGLE_DELTA", task)
        b = c53.builder("SINGLE_DELTA")
        b.task = task
        kw = dict(task=task, stats_fn=c53.stats_fn("SINGLE_DELTA", "header"), grad_fn=tsc.conv_grad_norms, lr=c53.LR,
                  builder=b)
        return a, task, kw, iters or 24000
    if arm == "S59_GDN":
        a, task, kw = c59.setup("SINGLE_GDN", {})
        return a, task, kw, iters or 48000
    if arm == "S69_GDN":
        sp = c69.spec("S69_GDN")
        return sp["a"], sp["task"], c69.run_kw(sp, {}), iters or 48000
    raise KeyError(arm)


def run(p):
    arm, seed = p["arm"], p["seed"]
    a, task, kw, iters = setup(arm, p.get("iters"))
    keep = {"at": -1}
    rec = run_one(a, seed, iters, tbo.EVAL_EVERY, keep=keep, **kw)
    if not rec.get("ok"):
        return json.loads(json.dumps(rec, default=float))
    m = keep["model"]
    r0 = record(arm, seed)
    want = [c for c in r0["curve"] if c[0] <= iters]
    rec["repro"] = rec["curve"] == want
    assert rec["repro"], f"{arm}|{seed}: the rerun's curve differs from the record ({rec['curve'][-2:]} vs {want[-2:]})"
    data = pc.probe_data(task)
    rec["probe"] = pc.probe(m, task, data)
    rec["probe_init"] = pc.probe(kw["builder"](a, seed)(), task, data)
    rec.update(arm=arm, bound=rec["transition"] is not None, record_transition=r0["transition"])
    return json.loads(json.dumps(rec, default=float))


def checks(p):
    import numpy as np
    rows = []
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    Xs = rng.normal(size=(4000, 8)); Xs[:, 0] += 3 * (2 * y - 1)
    hit, _ = pc.fit_score(Xs[:3200], y[:3200], Xs[3200:], y[3200:])
    Xr = rng.normal(size=(4000, 8))
    hit_r, _ = pc.fit_score(Xr[:3200], y[:3200], Xr[3200:], y[3200:])
    rows.append((f"G's synthetic probe check: informative feature {hit.mean():.4f} (G 0.998) >= 0.99; random {hit_r.mean():.3f} "
                 f"in [0.4, 0.6]", hit.mean() >= 0.99 and 0.4 <= hit_r.mean() <= 0.6))
    for arm in REC:
        a, task, kw, iters = setup(arm, None)
        m = kw["builder"](a, 350)()
        x = task.make_batch(32, torch.Generator().manual_seed(3))[0][:, :-1]
        feats, logits, ref = pc.residuals(m, x)
        ok = torch.equal(logits, ref) and set(feats) == {"emb", "L1", "L2", "L3", "final"}
        rows.append((f"{arm}: G's residual loop gives the model's own logits bit for bit on this memory "
                     f"({type(m.attn).__name__})", ok))
    # the probe's labels on both header tasks
    for task in (c53.TASKS["header"], c59.H8):
        x, tr, te = pc.probe_data(task)
        kpos, vpos, _ = pc.header_positions(task)
        lab = task.stream_labels(x)
        hdr = torch.stack([x[:, b * task.blk] for b in range(task.S)], 1).repeat_interleave(task.P, 1)
        rows.append((f"P = {task.P} header: probe labels at KEY and VAL = the block's CTX; 1600 / 400 sequences, disjoint",
                     torch.equal(lab[:, kpos], hdr) and torch.equal(lab[:, vpos], hdr) and len(tr) == 1600 and len(te) == 400
                     and not set(tr.tolist()) & set(te.tolist())))
    # short reproduction of each arm's record (curve through 2400)
    for arm, seed in (("S53_DELTA", 307), ("S59_GDN", 350), ("S69_GDN", 350)):
        a, task, kw, _ = setup(arm, 2400)
        r = run_one(a, seed, 2400, tbo.EVAL_EVERY, **kw)
        want = [c for c in record(arm, seed)["curve"] if c[0] <= 2400]
        rows.append((f"{arm}|{seed} through 2400 reproduces its record bit for bit ({r['curve']} vs {want})", r["curve"] == want))
    return [[n, bool(v)] for n, v in rows]


def timing(p):
    import time
    from torch.optim.optimizer import register_optimizer_step_post_hook
    a, task, kw, _ = setup(p["arm"], p["steps"])
    stamps = []
    h = register_optimizer_step_post_hook(lambda o, aa, k: stamps.append(time.perf_counter()))
    try:
        run_one(a, 0, p["steps"], 10 ** 9, **kw)
    finally:
        h.remove()
    dd = sorted(y - x for x, y in zip(stamps, stamps[1:]))
    return [dd[len(dd) // 2], 2.0]
