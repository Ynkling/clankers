#!/usr/bin/env python
"""
explore_write_decoders_child.py — EXPLORATORY, not a result. S47's code that runs INSIDE a child process on the main line's
module tree at explore_main9c.MAIN_SHA (9c5939e). Parent: explore_write_decoders.

Reruns to 4800 of recorded configurations, each on its own run path and recipe (unchanged), with S38's decoder
(explore_b16_gates.decode) at CTX (3j), KEY (3j+1) and VAL (3j+2) positions at updates 0 (the built model, before
training), 1200, 2400 and 4800 (after that evaluation's statistics); eval mode, no gradient, its own generator:
  HINGE_D8_A  X's HINGE_D8 (test_slow_start.make_recipe(True, TAU), Adam; run_sc with D8: S=8, P=4, k=16)
  D8_HINGE_M  S33's D8_HINGE (s33c.make_rc "HINGE", Muon 0.005; run_sc with D8), fresh on this CPU
  A_HINGE_M   S33's A_HINGE (Muon 0.005; run_attempt with A4k4: S=4, P=4, k=4), fresh on this CPU
  HINGE16_M   S35's MUON_HINGE16 (Muon 0.005; run_attempt with A4k16: S=4, P=4, k=16), fresh on this CPU
  D8_PREV_A   S40's D8_PREV_A (explore_gate_prev_child.make_rc "adam": GATE_PREV + X's HINGE recipe; run_sc with D8)
Features: the gate state h, and the gate's input pre-activation u (W_in v_t, + W_prev v_(t-1) for GATE_PREV).
"""

import contextlib

import torch

import explore_main9c as mt
import explore_muon_scale_child as s33c
import explore_muon_k16_child as s35c
import explore_gate_state_child as s38c
import explore_gate_prev_child as s40c
import explore_muon_recipe as s25
import explore_b16_gates as g16
import explore_b16_main as b16
import test_slow_start as tss
import test_stream_recipe as tsr

AT = (0, 1200, 2400, 4800)
CONF = {"HINGE_D8_A": dict(cfg="b", opt="adam"), "D8_HINGE_M": dict(cfg="b", opt="muon"),
        "A_HINGE_M": dict(cfg="a", opt="muon"), "HINGE16_M": dict(cfg="k16", opt="muon"),
        "D8_PREV_A": dict(cfg="b", opt="adam", prev=True)}


def build_rc(conf, p, holder, out, log):
    c = CONF[conf]
    task = b16.task_of(c["cfg"])
    if c.get("prev"):
        rc, _ = s40c.make_rc("adam", holder, probe_out=None)
    elif c["opt"] == "adam":
        rc, _ = tss.make_recipe(True, tss.TAU)
    elif c["cfg"] == "k16":
        rc, _ = s35c.make_rc("HINGE", p["mlr"], holder, log)
    else:
        rc, _ = s33c.make_rc("HINGE", p["mlr"], holder)
    if p.get("probe", True):
        probe, at0 = b16.decoder_probe(task, out, AT)
        rc = b16.with_probe(rc, probe, on_build=at0)
    return rc, (s25.muon_in_onset_run(holder) if c["opt"] == "muon" else contextlib.nullcontext())


def run(p):
    c = CONF[p["conf"]]
    holder, out, log = {}, {}, []
    rc, ctx = build_rc(p["conf"], p, holder, out, log)
    a = b16.arm_of(c["cfg"])
    with ctx:
        rec = b16.path(c["cfg"], a, p["seed"], p["iters"], rc, eval_every=p.get("eval_every"))
    return b16.finish(rec, conf=p["conf"], cfg=c["cfg"], opt=c["opt"], muon_lr=p.get("mlr") if c["opt"] == "muon" else None,
                      decode=out)


# ── CHECKs ───────────────────────────────────────────────────────────────────
def checks(p):
    rows = [tuple(x) for x in s38c.decoder_check({})]
    g = torch.Generator().manual_seed(3)
    X = torch.randn(600, 16, generator=g)
    y = torch.randint(0, 8, (600,), generator=g)
    a1 = g16.ncm(X[:300], y[:300], X[300:], y[300:], 8)
    a2 = s38c.ncm(X[:300], y[:300], X[300:], y[300:], 8)
    rows.append((f"explore_b16_gates.ncm equals S38's ncm on random data ({a1:.4f} vs {a2:.4f})", a1 == a2))
    # the same synthetic checks through explore_b16_gates.ncm
    S, n, d = 8, 4000, 32
    gg = torch.Generator().manual_seed(11)
    means = torch.randn(S, d, generator=gg) * 5.0
    yy = torch.randint(0, S, (n,), generator=gg)
    XX = means[yy] + 0.1 * torch.randn(n, d, generator=gg)
    b1 = g16.ncm(XX[:n // 2], yy[:n // 2], XX[n // 2:], yy[n // 2:], S)
    ys = yy[torch.randperm(n, generator=gg)]
    b0 = g16.ncm(XX[:n // 2], ys[:n // 2], XX[n // 2:], ys[n // 2:], S)
    rows.append((f"explore_b16_gates.ncm scores {b1:.4f} on S38's separable set and {b0:.4f} with shuffled labels (chance "
                 f"{1 / S:.4f}, within 0.05)", b1 == 1.0 and abs(b0 - 1 / S) <= 0.05))
    fit, score = g16.halves(g16.PROBE_N)
    disj = set(fit.tolist()).isdisjoint(score.tolist()) and len(set(fit.tolist()) | set(score.tolist())) == g16.PROBE_N
    rows.append((f"the decoder's fit and score halves are disjoint sets of sequences ({len(fit)} + {len(score)} = {g16.PROBE_N})",
                 disj))
    # positions and labels on each configuration's probe; the gate recomputation for each gate kind at init
    for cfg in ("a", "k16", "b"):
        task = b16.task_of(cfg)
        x = g16.probe_data(task)[:, :-1]
        j = torch.arange(task.S * task.P)
        ctx, key, val = x[:, 3 * j], x[:, 3 * j + 1], x[:, 3 * j + 2]
        ok = (int(ctx.min()) >= 0 and int(ctx.max()) < task.S and int(key.min()) >= task.S
              and set(ctx.unique().tolist()).isdisjoint(key.unique().tolist())
              and set(key.unique().tolist()).isdisjoint(val.unique().tolist()))
        rows.append((f"config {cfg}: on S38's probe the CTX positions 3j hold the stream tokens 0..{task.S - 1} "
                     f"({sorted(ctx.unique().tolist())}), KEY positions 3j+1 keys ({int(key.min())}-{int(key.max())}), VAL positions "
                     f"3j+2 values ({int(val.min())}-{int(val.max())}), disjoint", ok))
    a = b16.arm_of("b")
    task = b16.task_of("b")
    data = g16.probe_data(task)
    mk = tsr.builder_for(a)
    diffs = {"recurrent": g16.decode(mk(a, 260)(), task, data)["gate_recompute_diff"],
             "GATE_PREV": g16.decode(s40c.prev_builder(mk)(a, 260)(), task, data)["gate_recompute_diff"],
             "LOCAL3": g16.decode(g16.to_local(mk(a, 260)(), 260), task, data)["gate_recompute_diff"],
             "reservoir": g16.decode(g16.to_res(mk(a, 260)()), task, data)["gate_recompute_diff"]}
    rows.append((f"the read gate recomputed from h equals the model's (max |diff| per gate kind {dict((k, f'{v:.1e}') for k, v in diffs.items())} "
                 f"<= 1e-6)", max(diffs.values()) <= 1e-6))
    d0 = g16.decode(s40c.prev_builder(mk)(a, 260)(), task, data)
    rows.append((f"GATE_PREV at update 0: its input decodes the stream at KEY (CTX one back) at {d0['u_key']:.3f} and at VAL "
                 f"(two back) at {d0['u_val']:.3f}, chance {1 / task.S:.3f} (the audit's point: KEY by construction, VAL not)",
                 d0["u_key"] >= 0.9))
    return [[n_, bool(v)] for n_, v in rows]


def timing(p):
    out = {}
    for conf in p["which"]:
        c = CONF[conf]
        st = b16.step_timing(lambda: run(dict(conf=conf, seed=0, iters=p["steps"], mlr=p["mlr"], eval_every=10 ** 9,
                                                probe=False)), c["opt"] == "adam")
        a = b16.arm_of(c["cfg"])
        m = tsr.builder_for(a)(a, 0)()
        if c.get("prev"):
            m = s40c.prev_builder(tsr.builder_for(a))(a, 0)()
        out[conf] = [st, b16.eval_timing(c["cfg"], m), b16.decode_timing(c["cfg"], m)]
    return out


def recorded(p):
    """X's HINGE_D8 records (the S47 comparison) as stored at RESULTS_SHA."""
    runs = mt.recorded("slow_start", mt.RESULTS_SHA)
    return {k: dict(curve=runs[k]["curve"], stats=[s for s in runs[k]["stats"] if s["step"] != "end" and s["step"] <= p["t"]])
            for k in p["keys"] if k in runs}
