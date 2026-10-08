#!/usr/bin/env python
"""
explore_h_query_child.py — EXPLORATORY, not a result. Session H's S62 code that runs INSIDE a child process on the main
line's module tree at explore_main9c.MAIN_SHA (9c5939e). Parent: explore_h_query_gate. It reruns a saved S58 or S58b run
with that screen's own child code (explore_h_collapse_child.build_rc / explore_h_eight_child.build_rc, unchanged, imported
read-only) and adds one more callback after each evaluation (after the screen's own checks and trigger; eval mode, no
gradient, no RNG): the query-gate measurement below. Training is untouched, so the rerun must reproduce the saved run's
curve and every statistic bit for bit (the parent checks every run).

MEASUREMENT (on the run's routing probe, test_multilayer_binding.probe_batch(task, seed), the probe routing_k uses):
  g = the read gate (the model's third output; for LOCAL3 read = write; for GUMBEL_* no noise in eval mode).
  body VAL map   routing_k's ch_map: stream s -> argmax_c of the mean of g over the body's VAL positions 3j+2 of stream s.
  body KEY map   the same at KEY positions 3j+1 (explore_h_collapse_child.key_routing).
  query map      the same at the query's KEY position (task.qpos[0]) over the probe sequences whose query stream (the token
                 one position before) is s.
  margin_val     mean over probe sequences of g_q[c_VAL(s_q)] - mean_{s' != s_q} g_q[c_VAL(s')]: the query read gate's mass
                 on the channel the VAL map gives its own stream, minus its mean mass on the channels the VAL map gives the
                 other streams (g_q = g at the query KEY position, s_q its stream).
Recorded after every evaluation; "end" = the last evaluation (the run's stop).
"""

import torch

import explore_split_child as s36c
import explore_b16_main as b16
import explore_h_collapse_child as hcc
import explore_h_eight_child as h8c
import explore_muon_k16_child as s35c
import test_scale_axes as tsa
import test_stream_recipe as tsr
from test_multilayer_binding import probe_batch
from test_instrument_v2 import TAU_END

SRC = {"collapse": (hcc, "k16", 28800), "eight": (h8c, "b", 43200)}


@torch.no_grad()
def query_measure(m, task, probe):
    was = m.training
    m.eval()
    try:
        pinp = probe[0]
        gr = m(pinp, TAU_END)[2]
    finally:
        m.train(was)
    S = task.S
    qpos = task.qpos[0]
    qs = pinp[:, qpos - 1]
    gq = gr[:, qpos]
    qmap = [int(gq[qs == s].mean(0).argmax()) for s in range(S)]
    vmap = tsa.routing_k(m, task, probe)["ch_map"]
    kmap = hcc.key_routing(m, task, probe)["ch_map_key"]
    cv = torch.tensor(vmap)
    own = gq[torch.arange(gq.shape[0]), cv[qs]]
    allm = gq[:, cv]                                                  # (B, S): mass on each stream's VAL channel
    other = (allm.sum(1) - allm[torch.arange(gq.shape[0]), qs]) / (S - 1)
    return dict(query_map=qmap, val_map=vmap, key_map=kmap, eq_val=qmap == vmap, eq_key=qmap == kmap,
                margin_val=float((own - other).mean()), query_max=float(gq.max(-1).values.mean()))


def run(p):
    """p: src ('collapse' | 'eight'), arm, seed, iters (the screen's)."""
    mod, cfg, total = SRC[p["src"]]
    task = b16.task_of(cfg)
    probe = probe_batch(task, p["seed"])
    holder, out, info, log, q = {}, {}, dict(checks=[]), {}, {}
    rc, ctx = mod.build_rc(p["arm"], dict(seed=p["seed"], iters=p["iters"]), holder, out, info, log)

    def qprobe(m, step):
        q[str(step)] = query_measure(m, task, probe)
    rc = s36c.with_trigger(rc, qprobe)
    with ctx:
        rec = b16.path(cfg, b16.arm_of(cfg), p["seed"], p["iters"], rc)
    if q:
        last = max(q, key=int)
        q["end"] = dict(q[last], step=int(last))
    return b16.finish(rec, arm=p["arm"], src=p["src"], cfg=cfg, query=q, splits=sum(1 for c in info["checks"] if c["fired"]))


def checks(p):
    rows = []
    for src, (mod, cfg, total) in SRC.items():
        task = b16.task_of(cfg)
        a = s35c.arm("CEIL") if cfg == "k16" else h8c.s33c.cfg_arm(cfg, "CEIL")
        m = tsr.builder_for(a)(a, 420)()
        r = query_measure(m, task, probe_batch(task, 420))
        ident = list(range(task.S))
        rows.append((f"{src}: on the perfect gate the query, VAL and KEY maps are the identity ({r['query_map']}, {r['val_map']}, "
                     f"{r['key_map']}) and margin_val = {r['margin_val']:.4f} (1 for a one-hot gate)",
                     r["query_map"] == r["val_map"] == r["key_map"] == ident and abs(r["margin_val"] - 1.0) < 1e-6))
    return [[n, bool(v)] for n, v in rows]
