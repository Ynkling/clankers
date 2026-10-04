#!/usr/bin/env python
"""
explore_gate_state_child.py — EXPLORATORY, not a result. S38's code that runs INSIDE a child process on the
main line's module tree at explore_main9c.MAIN_SHA (9c5939e). Parent: explore_gate_state.

Configurations (reruns of recorded runs, to 9600 updates, on their own run paths and recipes):
  D8_HINGE_M   S33's D8_HINGE (Muon lr 0.005, S25's groups, main's hinge; run_sc with D8: S=8, P=4, k=16)
  HINGE_D8_A   X's HINGE_D8 (test_slow_start.make_recipe(True, TAU), Adam; run_sc with D8)
  A_HINGE_M    S33's A_HINGE (Muon; run_attempt with A4k4: S=4, P=4, k=4)
  HINGE16_M    S35's MUON_HINGE16 (Muon; run_attempt with A4k16: S=4, P=4, k=16)
THE PROBE: 512 sequences of the configuration's task from torch.Generator seeded 38_000 (its own generator,
the same for every seed of a configuration), measured at updates 1200, 2400, 4800 and 9600 after that
evaluation's statistics (eval mode, no gradient; training untouched). The gate state h_t is recomputed from
the model's own embedding by Instrument.gates' recurrence (h_t = tanh(W_in v_t + W_h h_{t-1}), v = the
embedding; the read gate recomputed from it is compared with the model's, as a check), and the gate's input
is v_t itself.
DECODABILITY (labelled): at the key positions 3j+1 (j = 0 .. S*P-1) the stream label is the CTX token at 3j;
a nearest-class-mean decoder (Euclidean, one mean per stream) is fit on the probe's first 256 sequences and
scored on the other 256 (accuracy; chance 1/S), from: h at every key position; the gate input v at every key
position (reference); h at each block's first key and at its last key, a block being the S adjacent triples
that share a key in the grouped layout (positions j = b*S and j = b*S + S - 1).
"""

import json

import torch

import explore_main9c as mt
import explore_muon_scale_child as s33c
import explore_muon_k16_child as s35c
import test_slow_start as tss
import test_stream_curriculum as tscur
import test_stream_recipe as tsr
from test_instrument_v2 import TAU_END

PROBE_N, PROBE_SEED = 512, 38_000
AT = (1200, 2400, 4800, 9600)
ITERS = 9600
CONF = {"D8_HINGE_M": dict(cfg="b", opt="muon", rec="S33 D8_HINGE"),
        "HINGE_D8_A": dict(cfg="b", opt="adam", rec="X HINGE_D8"),
        "A_HINGE_M": dict(cfg="a", opt="muon", rec="S33 A_HINGE"),
        "HINGE16_M": dict(cfg="k16", opt="muon", rec="S35 MUON_HINGE16")}


def task_of(cfg):
    return s35c.TASK if cfg == "k16" else s33c.cfg_task(cfg)


def ncm(Xtr, ytr, Xte, yte, n_cls):
    """Nearest-class-mean accuracy (Euclidean); classes absent from the training half are never predicted."""
    means, ok = [], []
    for c in range(n_cls):
        sel = ytr == c
        ok.append(bool(sel.any()))
        means.append(Xtr[sel].mean(0) if sel.any() else torch.full((Xtr.shape[1],), float("inf")))
    M = torch.stack(means)
    d = torch.cdist(Xte, M)
    return float((d.argmin(1) == yte).float().mean())


@torch.no_grad()
def gate_states(m, x):
    was = m.training
    m.eval()
    try:
        assert m.gate_kind == "recurrent" and not m.gate_ln and m.mode == "sym"
        v = m.embed(x)
        h = torch.zeros(v.shape[0], m.h_gate, dtype=v.dtype)
        hs = []
        for t in range(v.shape[1]):
            h = torch.tanh(v[:, t] @ m.W_in.T + h @ m.W_h.T)
            hs.append(h)
        H = torch.stack(hs, dim=1)
        gr_m = m(x, TAU_END)[2]
    finally:
        m.train(was)
    return v, H, float((torch.softmax(H @ m.W_g.T, -1) - gr_m).abs().max())


def decode(m, task, data):
    x = data[:, :-1]
    S, P = task.S, task.P
    n = S * P
    j = torch.arange(n)
    v, H, diff = gate_states(m, x)
    lab = x[:, 3 * j]                                                  # (B, n) stream labels
    half = x.shape[0] // 2

    def acc(F_, L):                                                    # F_: (B, n', d), L: (B, n')
        Xtr, Xte = F_[:half].reshape(-1, F_.shape[-1]), F_[half:].reshape(-1, F_.shape[-1])
        return ncm(Xtr, L[:half].reshape(-1), Xte, L[half:].reshape(-1), S)
    first, last = j[::S], j[S - 1::S]
    return dict(h_key=acc(H[:, 3 * j + 1], lab), v_key=acc(v[:, 3 * j + 1], lab),
                h_first=acc(H[:, 3 * first + 1], lab[:, first]), h_last=acc(H[:, 3 * last + 1], lab[:, last]),
                gate_recompute_diff=diff, chance=1.0 / S)


def path(cfg, a, seed, iters, rc):
    if cfg == "k16":
        return s35c.path(a, seed, iters, rc)
    return s33c.path(cfg, a, seed, iters, rc)


def arm_of(cfg):
    return s35c.arm("HINGE") if cfg == "k16" else s33c.cfg_arm(cfg, "HINGE")


def with_probe(rc, task, out):
    data = task.make_batch(PROBE_N, torch.Generator().manual_seed(PROBE_SEED))[0]

    def rc2(kw):
        kw = rc(kw)
        c0 = kw.get("check")

        def check(m, step, d):
            if c0 is not None:
                c0(m, step, d)
            if step in AT:
                out[str(step)] = decode(m, task, data)
        kw["check"] = check
        return kw
    return rc2


def run(p):
    c = CONF[p["conf"]]
    cfg, holder, out = c["cfg"], {}, {}
    task = task_of(cfg)
    a = arm_of(cfg)
    if c["opt"] == "muon":
        rc, _ = s33c.make_rc("HINGE", p["mlr"], holder)
        rc = with_probe(rc, task, out)
        with s33c.s25.muon_in_onset_run(holder):
            rec = path(cfg, a, p["seed"], p["iters"], rc)
    else:
        rc, _ = tss.make_recipe(True, tss.TAU)
        rec = path(cfg, a, p["seed"], p["iters"], with_probe(rc, task, out))
    if rec.get("ok"):
        rec.update(conf=p["conf"], cfg=cfg, opt=c["opt"], decode=out, maps=s33c.maps_of(rec), main_sha=mt.MAIN_SHA)
        try:
            rec["outcome"] = tsr.outcome(rec)
        except Exception as e:
            rec["outcome"] = f"n/a ({type(e).__name__})"
    return json.loads(json.dumps(rec, default=float))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def decoder_check(p):
    g = torch.Generator().manual_seed(11)
    S, n, d = 8, 4000, 32
    means = torch.randn(S, d, generator=g) * 5.0
    y = torch.randint(0, S, (n,), generator=g)
    X = means[y] + 0.1 * torch.randn(n, d, generator=g)
    a1 = ncm(X[:n // 2], y[:n // 2], X[n // 2:], y[n // 2:], S)
    ys = y[torch.randperm(n, generator=g)]
    a0 = ncm(X[:n // 2], ys[:n // 2], X[n // 2:], ys[n // 2:], S)
    return [[f"the decoder scores {a1:.4f} on a synthetic separable set (8 classes, 4000 points) and {a0:.4f} with "
             f"shuffled labels (chance {1 / S:.4f}; within 0.05: {abs(a0 - 1 / S) <= 0.05})",
             a1 == 1.0 and abs(a0 - 1 / S) <= 0.05]]


def rerun_check(p):
    """One short rerun per configuration (through 2400) against its record, and the gate recompute."""
    r = run(dict(conf=p["conf"], seed=p["seed"], iters=2400, mlr=p.get("mlr")))
    return dict(curve=r["curve"], decode=r["decode"])


def timing(p):
    """The median interval between optimizer steps over p["steps"] steps of each configuration (no probe in the
    window), and the time of one probe measurement; run one child per configuration at once."""
    import time
    import torch.optim.optimizer as topt
    out = {}
    for conf in p["which"]:
        c = CONF[conf]
        stamps = []

        def post(opt, args, kwargs):
            if c["opt"] == "adam" or isinstance(opt, s33c.s25.SliceMuon):
                stamps.append(time.perf_counter())
        h = topt.register_optimizer_step_post_hook(post)
        try:
            run(dict(conf=conf, seed=0, iters=p["steps"], mlr=p["mlr"]))
        finally:
            h.remove()
        d = sorted(y - x for x, y in zip(stamps, stamps[1:]))
        task = task_of(c["cfg"])
        a = arm_of(c["cfg"])
        m = tsr.builder_for(a)(a, 0)()
        data = task.make_batch(PROBE_N, torch.Generator().manual_seed(PROBE_SEED))[0]
        t0 = time.time()
        decode(m, task, data)
        td = time.time() - t0
        from test_channel_binding import eval_batch
        from test_multilayer_binding import probe_batch
        import test_binding_onset as tbo
        import test_scale_axes as tsa
        t0 = time.time()
        tbo.evaluate(m, task, eval_batch(task))
        sf = tscur.sc_stats((tscur.T1, tscur.T2)) if c["cfg"] == "b" else tsa.scale_stats
        sf(m, task, probe_batch(task, 0), 1200)
        out[conf] = [d[len(d) // 2], time.time() - t0, td]
    return out
