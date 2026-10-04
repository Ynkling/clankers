#!/usr/bin/env python
"""
explore_gate_memory_child.py — EXPLORATORY, not a result. S39's code that runs INSIDE a child process on the
main line's module tree at explore_main9c.MAIN_SHA (9c5939e). Parent: explore_gate_memory.

Configurations (reruns of recorded runs to 2400 updates, on their own run paths and recipes):
  D8_HINGE_M  S33's D8_HINGE (Muon lr 0.005, S25's groups, main's hinge; run_sc with D8: S=8, P=4, k=16)
  D8_SLOW_M   S33's D8_SLOW (the same without the hinge)
  HINGE_D8_A  X's HINGE_D8 (test_slow_start.make_recipe(True, TAU), Adam; run_sc with D8)
  A_HINGE_M   S33's A_HINGE (Muon; run_attempt with A4k4: S=4, P=4, k=4), a reference
MEASUREMENTS at updates 0, 50, 100, 200, 300, 400, 600, 800, 1200, 1600, 2400 (update 0: the built model,
before any step; later: right after that update's optimizer step, from a step post-hook; eval mode, no
gradient, no random draw: training untouched), on a 512-sequence probe of the configuration's task (its
own generator, seed 39000), labelled:
  - stream decodability (explore_gate_state_child.ncm: nearest class mean fit on the first 256 sequences,
    accuracy on the other 256) of the gate state h at the stream-token positions 3j and at the key
    positions 3j+1;
  - the median norm of W_in v_t and of W_h h_{t-1} (the two terms before the tanh, Instrument.gates'
    recurrence on the model's embedding) at stream-token and at key positions; the fraction of h units with
    |h| > 0.95 at key positions and over the whole body;
  - the read gate's mean entropy (nats) at key positions; the distinct channels holding the streams (the
    argmax channel of each stream's mean read gate at value positions 3j+2, routing_k's rule).
"""

import json

import torch

import explore_main9c as mt
import explore_muon_scale_child as s33c
import explore_gate_state_child as s38c
import test_slow_start as tss
import test_stream_recipe as tsr
from test_instrument_v2 import TAU_END

PROBE_N, PROBE_SEED = 512, 39_000
AT = (0, 50, 100, 200, 300, 400, 600, 800, 1200, 1600, 2400)
ITERS = 2400
SAT = 0.95
CONF = {"D8_HINGE_M": dict(cfg="b", opt="muon", kind="HINGE"),
        "D8_SLOW_M": dict(cfg="b", opt="muon", kind="SLOW"),
        "HINGE_D8_A": dict(cfg="b", opt="adam", kind="HINGE"),
        "A_HINGE_M": dict(cfg="a", opt="muon", kind="HINGE")}


@torch.no_grad()
def measure(m, task, data):
    x = data[:, :-1]
    S, P = task.S, task.P
    n = S * P
    j = torch.arange(n)
    was = m.training
    m.eval()
    try:
        assert m.gate_kind == "recurrent" and not m.gate_ln and m.mode == "sym"
        v = m.embed(x)
        B, T, _ = v.shape
        h = torch.zeros(B, m.h_gate, dtype=v.dtype)
        hs, ain, ah = [], [], []
        for t in range(T):
            a = v[:, t] @ m.W_in.T
            b = h @ m.W_h.T
            h = torch.tanh(a + b)
            hs.append(h)
            ain.append(a)
            ah.append(b)
        H, A, Bh = torch.stack(hs, 1), torch.stack(ain, 1), torch.stack(ah, 1)
        gr = m(x, TAU_END)[2]
    finally:
        m.train(was)
    lab = x[:, 3 * j]
    half = B // 2

    def dec(F_):
        return s38c.ncm(F_[:half].reshape(-1, F_.shape[-1]), lab[:half].reshape(-1),
                        F_[half:].reshape(-1, F_.shape[-1]), lab[half:].reshape(-1), S)

    def mnorm(Z, pos):
        return float(Z[:, pos].norm(dim=-1).median())
    sp, kp = 3 * j, 3 * j + 1
    gk = gr[:, kp]
    ent = float(-(gk * torch.log(gk.clamp_min(1e-30))).sum(-1).mean())
    gv = gr[:, 3 * j + 2]
    cmap = [int(gv[lab == s].mean(0).argmax()) for s in range(S)]
    return dict(dec_key=dec(H[:, kp]), dec_stream=dec(H[:, sp]),
                win_stream=mnorm(A, sp), win_key=mnorm(A, kp), wh_stream=mnorm(Bh, sp), wh_key=mnorm(Bh, kp),
                sat_key=float((H[:, kp].abs() > SAT).float().mean()), sat_body=float((H[:, :3 * n].abs() > SAT).float().mean()),
                entropy_key=ent, distinct=len(set(cmap)), ch_map=cmap,
                gate_recompute_diff=float((torch.softmax(H @ m.W_g.T, -1) - gr).abs().max()))


def run(p):
    """p: conf, seed, iters, mlr (Muon configurations)."""
    import torch.optim.optimizer as topt
    c = CONF[p["conf"]]
    cfg, holder, out, box = c["cfg"], {}, {}, dict(n=0)
    task = s33c.cfg_task(cfg)
    a = s33c.cfg_arm(cfg, "HINGE")
    data = task.make_batch(PROBE_N, torch.Generator().manual_seed(PROBE_SEED))[0]
    if c["opt"] == "muon":
        rc, _ = s33c.make_rc(c["kind"], p["mlr"], holder)
    else:
        rc, _ = tss.make_recipe(True, tss.TAU)

    def rc2(kw):
        kw = rc(kw)
        b0 = kw["builder"]

        def builder(aa, seed):
            mk = b0(aa, seed)

            def make():
                m = mk()
                box["m"] = m
                box["builds"] = box.get("builds", 0) + 1
                if 0 in AT:
                    out["0"] = measure(m, task, data)
                return m
            return make
        kw["builder"] = builder
        return kw

    def post(opt, args, kwargs):
        if c["opt"] == "muon" and not isinstance(opt, s33c.s25.SliceMuon):
            return
        box["n"] += 1
        if box["n"] in AT and "m" in box:
            out[str(box["n"])] = measure(box["m"], task, data)
    hk = topt.register_optimizer_step_post_hook(post)
    try:
        if c["opt"] == "muon":
            with s33c.s25.muon_in_onset_run(holder):
                rec = s33c.path(cfg, a, p["seed"], p["iters"], rc2)
        else:
            rec = s33c.path(cfg, a, p["seed"], p["iters"], rc2)
    finally:
        hk.remove()
    if rec.get("ok"):
        rec.update(conf=p["conf"], cfg=cfg, opt=c["opt"], mem=out, updates_counted=box["n"], builds=box.get("builds"),
                   muon_lr=p.get("mlr") if c["opt"] == "muon" else None, main_sha=mt.MAIN_SHA)
    return json.loads(json.dumps(rec, default=float))


def timing(p):
    """Per configuration: the median interval between optimizer steps over p["steps"] steps (measurements at the
    updates in AT up to there included), the time of one evaluation (held-out set and the run path's statistics),
    and of one measurement."""
    import time
    import torch.optim.optimizer as topt
    import test_binding_onset as tbo
    import test_scale_axes as tsa
    import test_stream_curriculum as tscur
    from test_channel_binding import eval_batch
    from test_multilayer_binding import probe_batch
    out = {}
    for conf in p["which"]:
        c = CONF[conf]
        stamps = []

        def post(opt, args, kwargs):
            if c["opt"] == "adam" or isinstance(opt, s33c.s25.SliceMuon):
                stamps.append(time.perf_counter())
        h = topt.register_optimizer_step_post_hook(post)
        try:
            t0 = time.time()
            run(dict(conf=conf, seed=0, iters=p["steps"], mlr=p["mlr"]))
            total = time.time() - t0
        finally:
            h.remove()
        d = sorted(y - x for x, y in zip(stamps, stamps[1:]))
        task = s33c.cfg_task(c["cfg"])
        a = s33c.cfg_arm(c["cfg"], "HINGE")
        m = tsr.builder_for(a)(a, 0)()
        t0 = time.time()
        tbo.evaluate(m, task, eval_batch(task))
        sf = tsa.scale_stats if c["cfg"] == "a" else tscur.sc_stats((tscur.T1, tscur.T2))
        sf(m, task, probe_batch(task, 0), 1200)
        te = time.time() - t0
        data = task.make_batch(PROBE_N, torch.Generator().manual_seed(PROBE_SEED))[0]
        t0 = time.time()
        measure(m, task, data)
        out[conf] = [d[len(d) // 2], te, time.time() - t0, total]
    return out
