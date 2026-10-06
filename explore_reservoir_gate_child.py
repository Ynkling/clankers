#!/usr/bin/env python
"""
explore_reservoir_gate_child.py — EXPLORATORY, not a result. S45's code that runs INSIDE a child process on the main line's
module tree at explore_main9c.MAIN_SHA (9c5939e). Parent: explore_reservoir_gate. Recipes: explore_b16_main.

RES_D8 (S=8, P=4, k=16, conv, all 8 streams from step 1; run_sc with D8, 28800 steps): arm A's recurrent gate with W_h
frozen at its initial value rescaled to spectral radius 0.5 (explore_b16_gates.to_res: echo-state style); W_in, W_g
trained; + SLOW + main's HINGE (tau 0.2).
  RES_D8_A  Adam: gate group W_in, W_g at 1e-3 throughout; every other trainable parameter 1e-4 for updates 1-2400,
            1e-3 after (test_slow_start's slow groups without W_h).
  RES_D8_M  S25's groups at Muon lr 0.005 with the gate W_in, W_g (W_h in no group).
Measured at 1200, 2400, 4800 and 9600: S38's decoder (CTX, KEY, VAL; h and u), rho(W_h) and sigma_max(W_h) (constant).
"""

import contextlib

import torch

import explore_muon_scale_child as s33c
import explore_muon_recipe as s25
import explore_b16_gates as g16
import explore_b16_main as b16
import test_stream_recipe as tsr

AT = (1200, 2400, 4800, 9600)
CFG = "b"
ARM = {"RES_D8_A": dict(opt="adam"), "RES_D8_M": dict(opt="muon")}


def to_res(m, seed):
    return g16.to_res(m)


def make(arm_name, p, holder, out, wh, init):
    opt = ARM[arm_name]["opt"]
    task = b16.task_of(CFG)
    probe = b16.chain(b16.decoder_probe(task, out, AT)[0] if p.get("probe", True) else None, b16.wh_probe(wh, AT))
    rc, st = b16.make_rc(opt, holder, b16.GATE_RES, tau=b16.TAU, mlr=p.get("mlr"), convert=(to_res,), probe=probe,
                         on_build=lambda m: init.update(m.res_init), keep=p.get("keep"))
    return rc, st, (s25.muon_in_onset_run(holder) if opt == "muon" else contextlib.nullcontext())


def run(p):
    holder, out, wh, init = {}, {}, {}, {}
    rc, st, ctx = make(p["arm"], p, holder, out, wh, init)
    with ctx:
        rec = b16.path(CFG, b16.arm_of(CFG), p["seed"], p["iters"], rc, eval_every=p.get("eval_every"))
    return b16.finish(rec, arm=p["arm"], cfg=CFG, opt=ARM[p["arm"]]["opt"], muon_lr=p.get("mlr"), decode=out, wh=wh,
                      res_init=init, groups=b16.group_names(holder))


def checks(p):
    rows = []
    a = b16.arm_of(CFG)
    m0 = tsr.builder_for(a)(a, 260)()
    m1 = to_res(tsr.builder_for(a)(a, 260)(), 260)
    p0, p1 = dict(m0.named_parameters()), dict(m1.named_parameters())
    same = all(torch.equal(p1[n], p0[n]) for n in p0 if n != "W_h") and sorted(p0) == sorted(p1)
    r = g16.rho(m1.W_h)
    rows.append((f"at init every parameter but W_h equals the run path's model's; W_h = W_h0 x 0.5 / rho(W_h0) (rho0 "
                 f"{m1.res_init['rho0']:.4f}, sigma0 {m1.res_init['sigma0']:.4f} -> rho {r:.7f}, sigma {g16.sigma_max(m1.W_h):.4f}); "
                 f"|rho - 0.5| <= 1e-6; frozen ({not m1.W_h.requires_grad})",
                 same and abs(r - 0.5) <= 1e-6 and not m1.W_h.requires_grad))
    for opt in ("adam", "muon"):
        keep = {"at": 50}
        holder, out, wh, init = {}, {}, {}, {}
        rc, st, ctx = make(f"RES_D8_{'A' if opt == 'adam' else 'M'}", dict(mlr=p["mlr"], probe=False, keep=keep), holder,
                           out, wh, init)
        with ctx:
            rec = b16.path(CFG, a, 260, 50, rc)
        snap = keep["snap"]
        W0 = to_res(tsr.builder_for(a)(a, 260)(), 260).W_h.detach()
        r50 = g16.rho(snap["W_h"])
        ids = [[id(q) for q in g["params"]] for g in holder["groups"]]
        mm = keep["model"]
        cov = b16.cover(ids, mm) and not any(q is mm.W_h for g in holder["groups"] for q in g["params"])
        moved = not torch.equal(snap["W_in"], tsr.builder_for(a)(a, 260)().W_in.detach())
        rows.append((f"{opt}: over 50 steps W_h receives no update (equal to its rescaled initial value: "
                     f"{torch.equal(snap['W_h'], W0)}; W_in moved: {moved}); rho(W_h) after 50 steps {r50:.7f} (|rho - 0.5| "
                     f"<= 1e-6); the groups cover every trainable parameter once, W_h in none "
                     f"({[g['names'] for g in holder['groups']]})",
                     torch.equal(snap["W_h"], W0) and abs(r50 - 0.5) <= 1e-6 and cov and moved and rec.get("ok")))
    return [[n, bool(v)] for n, v in rows]


def timing(p):
    out = {}
    for arm_name in p["which"]:
        st = b16.step_timing(lambda: run(dict(arm=arm_name, seed=0, iters=p["steps"], mlr=p["mlr"], eval_every=10 ** 9,
                                                probe=False)), ARM[arm_name]["opt"] == "adam")
        a = b16.arm_of(CFG)
        m = to_res(tsr.builder_for(a)(a, 0)(), 0)
        out[arm_name] = [st, b16.eval_timing(CFG, m), b16.decode_timing(CFG, m)]
    return out
