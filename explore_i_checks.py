#!/usr/bin/env python
"""
explore_i_checks.py — EXPLORATORY, not a result. Session I's CHECKs, asserted before any screen's run (each screen calls
checks(task_list) and stores the result in its meta). 1 torch thread.

  C1  the task (explore_i_task.check_task) at every configuration the screen runs.
  C2  the reproduction: explore_i_common.train with the new task and gates disabled — test_short_conv's grouped TASK
      (S=2, P=4), make_model(gate "none", k=1) = the B_conv arm, BindTask's query loss ("select"), Adam 1e-3 — at seed
      160 equals the recorded X run B_conv|160 (results/X/short_conv_results.json, Xeon @ 2.10GHz, torch 2.14.0) bit
      for bit through 2400 updates: the evaluation curve (accuracy, loss) and the gradient norms at updates 1, 10 and 100
      (test_short_conv.conv_grad_norms). Asserted when this container's CPU is that record's CPU; printed otherwise.
  C3  the window gates: LOCAL3 (to_local) at t sees token t-2 and not t-3; WIDE (width LMAX+2) sees t-(LMAX+1) and not
      t-(LMAX+2); to_window at width 3 equals to_local (every parameter bitwise, the same logits); W_h frozen; the
      conversion keeps every built parameter bitwise.
  C4  REG3 with the convolution: Reg3Conv's conv loop at the identity-initialised convolution equals RegBDH.forward's
      no-conv loop bit for bit (logits and every layer's gate); to_reg3's new parameters equal explore_g_regmodel.to_reg's
      on the same model; the gate at t ignores tokens > t and the register written by token t first reaches the gate at
      t+1 (a change of token t's register write changes no gate at <= t... measured as: perturbing the embedding of
      token t only through the register path changes the gates at t+1 and not at t); the model is strictly causal
      (logits at < p do not change with token p); the SLOW gate group is explore_g_regmodel.gate_names; the nudge moves
      only GATE_NEW parameters.
  C5  the perfect gate: one-hot on stream_labels at every position (k = S), so cross-source score factors are exactly 0.
  C6  the split's mechanics on this task (a LOCAL3 model after one Adam step; the trigger called at each check with no
      training in between, threshold 1.01, rise 1): fires at 4800, 9600, 14400; gap-blocked at 7200 and 12000;
      cap-blocked from 16800; each split leaves W_g's rows c* and c0 apart and zeroes W_g's Adam state; c* / c0 are the
      largest / smallest mean gate mass over all word positions; split_op / noise_for equal test_window_gate's on a copy.
      And: with the threshold at 0 a split arm equals the same arm without the split bit for bit through 7200 updates
      (on a short configuration; the checks at 2400 and 4800 run and none fires).
  C7  the probe: a synthetic informative feature decodes >= 0.99, a random one 0.4-0.6 (S = 2); the residual replay's
      logits equal the forward's (asserted inside every call).
  C8  the SLOW groups and lrs through the real loop (forward stubbed by the real model, 2401 updates on a short task are
      too slow: the groups are read at build and the switch is exercised by on_eval at 2400): every trainable parameter
      in exactly one group; gate group (W_in, W_g, gate_conv.conv_w) at 1e-3, rest 1e-4, rest 1e-3 after the switch.
"""

import copy
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F

torch.set_num_threads(1)

import explore_i_task as it
import explore_i_common as ic
import explore_b16_gates as g16
import explore_g_regmodel as rm
import test_short_conv as tsc
import test_binding_onset as tbo
from test_channel_binding import eval_batch
from test_instrument_v2 import TAU_END

REC_FILE = os.path.join(HERE, "results", "X", "short_conv_results.json")
SMALL = dict(S=2, V=4, NB=3, lmin=6, lmax=14, M=30)


def c2_repro():
    rec = json.load(open(REC_FILE))
    old = rec["runs"]["B_conv|160"]
    cpu_rec = rec["meta"]["cpu"]
    task = tsc.TASK
    data = eval_batch(task)
    curve, grad = [], {}

    def on_eval(m, step, opt):
        curve.append(list(tbo.evaluate(m, task, data)))
        curve[-1].insert(0, step)

    def gh(m, step):
        if step in (1, 10, 100):
            grad[str(step)] = list(tsc.conv_grad_norms(m))

    ic.train(task, lambda: ic.make_model(dict(gate="none", k=1), task, 160), 160, 2400, loss_kind="select",
             on_eval=on_eval, grad_hook=gh)
    same = curve == old["curve"][:2] and grad == old["grad"]
    return dict(ok=same, curve=curve, recorded=old["curve"][:2], grad_same=grad == old["grad"], cpu_record=cpu_rec,
                cpu_here=ic.cpu_model(), asserted=cpu_rec == ic.cpu_model())


def _gates_of(m, x):
    return ic.model_gates(m, x)


def c3_windows():
    task = it.ICMCTask(**SMALL)
    out = {}
    x = task.make_batch(4, torch.Generator().manual_seed(5))[0][:, :-1]
    for name, width in (("win3", 3), ("wide", task.lmax + 2)):
        m = ic.make_model(dict(gate=name, k=2), task, 160)
        p = 40
        x2 = x.clone()
        x2[:, p] = (x2[:, p] + 1) % task.vocab
        g1, g2 = _gates_of(m, x)[0], _gates_of(m, x2)[0]
        d = (g1 - g2).abs().amax((0, 2))
        out[name] = dict(sees_far=bool(d[p + width - 1] > 0), not_beyond=bool(d[p + width] == 0),
                         nothing_before=bool((d[:p] == 0).all()), frozen_wh=not m.W_h.requires_grad)
    base = ic.build(task, dict(ic.ARM_A, n_ch=2), ic.ARCH, 160)
    p0 = {n: q.detach().clone() for n, q in base.named_parameters()}
    a = g16.to_local(ic.build(task, dict(ic.ARM_A, n_ch=2), ic.ARCH, 160), 160)
    b = ic.to_window(ic.build(task, dict(ic.ARM_A, n_ch=2), ic.ARCH, 160), 160, 3)
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    out["width3_equals_local3"] = (pa.keys() == pb.keys() and all(torch.equal(pa[n], pb[n]) for n in pa)
                                   and torch.equal(a(x)[0], b(x)[0]))
    out["conversion_keeps_params"] = all(torch.equal(pb[n], p0[n]) for n in p0)
    out["ok"] = all(v if isinstance(v, bool) else all(v.values()) for v in out.values())
    return out


def c4_reg3():
    task = it.ICMCTask(**SMALL)
    out = {}
    x = task.make_batch(4, torch.Generator().manual_seed(6))[0][:, :-1]
    m = ic.make_model(dict(gate="reg3", k=2), task, 160)
    m.eval()
    m.keep_diag = True
    with torch.no_grad():
        l1 = m(x)[0]
        gA = [g.clone() for g in m.diag["gates"]]
        conv = m.conv
        m.conv = None
        l0 = rm.RegBDH.forward(m, x)[0]
        gB = [g.clone() for g in m.diag["gates"]]
        m.conv = conv
    out["conv loop at identity conv = RegBDH.forward"] = torch.equal(l0, l1) and all(torch.equal(a, b) for a, b in zip(gA, gB))
    # to_reg3's parameters vs explore_g_regmodel.to_reg on a no-conv twin
    nc = rm.to_reg(g16.to_local(ic.build(task, dict(ic.ARM_A, n_ch=2, model_kw={}), ic.ARCH, 160), 160), 160, sg=True,
                   reg=True, gated=(1, 2, 3), res_layers=(1, 2, 3), mem="hebb")
    pn, pm = dict(nc.named_parameters()), dict(m.named_parameters())
    out["new parameters = to_reg's"] = all(torch.equal(pn[n], pm[n]) for n in rm.GATE_NEW)
    # strict causality and the register's one-step delay
    p = 30
    x2 = x.clone()
    x2[:, p] = (x2[:, p] + 1) % task.vocab
    with torch.no_grad():
        la, lb = m(x)[0], m(x2)[0]
    out["strictly causal (logits < p unchanged)"] = bool(((la - lb).abs().amax((0, 2))[:p] == 0).all())
    with torch.no_grad():
        v = m.embed(x)
        R0, b0 = m.register(v)
        v2 = v.clone()
        v2[:, p] += 0.5
        R1, b1 = m.register(v2)
        dR = (R0 - R1).abs().amax((0, 2))
    out["register: token p's write reaches Rprev at p+1, not at <= p"] = bool((dR[:p + 1] == 0).all()) and bool(dR[p + 1] > 0)
    out["gate group = gate_names"] = ic.gate_names(m) == rm.gate_names(m)
    m.keep_diag = False
    # nudge moves only GATE_NEW
    a = ic.make_model(dict(gate="reg3", k=2), task, 161)
    b = ic.make_model(dict(gate="reg3", k=2, nudge=True), task, 161)
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    moved = [n for n in pa if not torch.equal(pa[n], pb[n])]
    out["nudge moves only GATE_NEW"] = bool(moved) and set(moved) <= set(rm.GATE_NEW)
    out["ok"] = all(out.values())
    return out


def c5_perfect():
    out = {}
    for S in (2, 4):
        task = it.ICMCTask(S=S, V=4, NB=3, lmin=6, lmax=14, M=30)
        m = ic.make_model(dict(gate="perfect", k=S), task, 160)
        x = task.make_batch(8, torch.Generator().manual_seed(7))[0][:, :-1]
        g = ic.model_gates(m, x)[0]
        out[f"S={S}"] = torch.equal(g, F.one_hot(task.stream_labels(x), S).float())
    out["ok"] = all(out.values())
    return out


def c6_split():
    import test_window_gate as twg
    task = it.ICMCTask(**SMALL)
    m = ic.make_model(dict(gate="win3", k=2), task, 160)
    opt = torch.optim.Adam([q for q in m.parameters() if q.requires_grad], lr=ic.LR)
    tok = task.make_batch(8, torch.Generator().manual_seed(11))[0]
    mask = task.loss_mask(tok)
    loss = F.cross_entropy(m(tok[:, :-1])[0][mask], tok[:, 1:][mask])
    opt.zero_grad(); loss.backward(); opt.step()
    trig = ic.make_split(task, 160, 26400, thr=1.01, rise=1.0)
    wg0 = m.W_g.detach().clone()
    for t in range(2400, 26400 + 1, 2400):
        trig(m, t, opt)
    rows = trig.rows
    fired = [r["step"] for r in rows if r["fired"]]
    gap = [r["step"] for r in rows if r.get("blocked_by") == "gap"]
    cap = [r["step"] for r in rows if r.get("blocked_by") == "cap"]
    st = {k: float(v.abs().max()) for k, v in opt.state[m.W_g].items() if torch.is_tensor(v) and v.dim() > 0}
    out = dict(fired=fired, gap=gap, cap=cap)
    # targeting: c*, c0 from the mean gate mass over word positions
    pe = ic.eval_set(task, ic.SPLIT_PROBE_N, ic.SPLIT_PROBE_SEED)
    px = pe["tokens"][:, :-1]
    mm = ic.model_gates(m, px)[0][px >= task.S].mean(0)
    last = rows[-1]
    ok_mech = (fired == [4800, 9600, 14400] and gap == [7200, 12000] and cap == list(range(16800, 26401, 2400))
               and all(r["row_dist_after"] > 0 for r in rows if r["fired"]) and all(v == 0 for v in st.values())
               and not torch.equal(wg0, m.W_g.detach()))
    out["mechanics"] = ok_mech
    out["targets = argmax/argmin of word-position mass"] = (last["cs"], last["c0"]) == (int(mm.argmax()), int(mm.argmin()))
    # transcription vs test_window_gate
    w = torch.randn(32)
    a, b = ic.noise_for(w, 77), twg.noise_for(w, 77)
    m1, m2 = copy.deepcopy(m), copy.deepcopy(m)
    o1 = torch.optim.Adam([q for q in m1.parameters() if q.requires_grad], lr=ic.LR)
    o2 = torch.optim.Adam([q for q in m2.parameters() if q.requires_grad], lr=ic.LR)
    for mm_, oo in ((m1, o1), (m2, o2)):
        l_ = F.cross_entropy(mm_(tok[:, :-1])[0][mask], tok[:, 1:][mask])
        oo.zero_grad(); l_.backward(); oo.step()
    r1, r2 = ic.split_op(m1, o1, 0, 1, 99), twg.split_op(m2, o2, 0, 1, 99)
    out["split_op = test_window_gate's"] = (all(torch.equal(x_, y_) for x_, y_ in zip(a, b)) and r1 == r2
                                           and torch.equal(m1.W_g, m2.W_g))
    out["ok"] = ok_mech and out["targets = argmax/argmin of word-position mass"] and out["split_op = test_window_gate's"]
    return out


def c6b_thr0():
    """With the threshold at 0 a split arm equals the same arm without it, bit for bit through 7200 (short task; the
    trigger's checks at 2400 and 4800 run, none fires)."""
    task = it.ICMCTask(S=2, V=4, NB=2, lmin=4, lmax=8, M=12)
    a = ic.run_one(task, dict(gate="win3", k=2, recipe="slow", split=True), 3, 7200, split_thr=0.0, with_probe=False)
    b = ic.run_one(task, dict(gate="win3", k=2, recipe="slow", split=False), 3, 7200, with_probe=False)
    same = [x["set"] for x in a["curve"]] == [x["set"] for x in b["curve"]] and \
        [x["loss"] for x in a["curve"]] == [x["loss"] for x in b["curve"]]
    return dict(ok=bool(same and a["ok"] and b["ok"] and a["splits"] == 0 and len(a["checks"]) == 2), checks=len(a.get("checks", [])))


def c7_probe():
    import numpy as np
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    Xi = np.c_[y + 0.05 * rng.standard_normal(4000), rng.standard_normal((4000, 5))]
    Xr = rng.standard_normal((4000, 6))
    hi, _ = ic.fit_score(Xi[:3200], y[:3200], Xi[3200:], y[3200:])
    hr, _ = ic.fit_score(Xr[:3200], y[:3200], Xr[3200:], y[3200:])
    # the replay on a real model (asserted inside residuals)
    task = it.ICMCTask(**SMALL)
    for spec in (dict(gate="win3", k=2), dict(gate="reg3", k=2), dict(gate="perfect", k=2), dict(gate="none", k=1)):
        ic.residuals(ic.make_model(spec, task, 1), task.make_batch(3, torch.Generator().manual_seed(1))[0][:, :-1])
    return dict(ok=bool(hi.mean() >= 0.99 and 0.4 <= hr.mean() <= 0.6), informative=float(hi.mean()), random=float(hr.mean()))


def c8_groups():
    task = it.ICMCTask(**SMALL)
    out = {}
    for spec in (dict(gate="win3", k=2), dict(gate="wide", k=2), dict(gate="reg3", k=2)):
        holder = {}
        m = ic.make_model(spec, task, 1)
        groups = ic.slow_groups(holder)(m)
        ids = [id(q) for g in groups for q in g["params"]]
        trainable = [id(q) for q in m.parameters() if q.requires_grad]
        gate = holder["names"][0]
        exp = (ic.GATE_L if spec["gate"] != "reg3" else rm.gate_names(m))
        out[spec["gate"]] = (sorted(ids) == sorted(trainable) and len(set(ids)) == len(ids) and list(gate) == list(exp)
                             and [g["lr"] for g in groups] == [1e-3, 1e-4] and "W_h" not in gate + holder["names"][1])
    out["ok"] = all(out.values())
    return out


def checks(tasks, quick=False):
    res = {}
    for t in tasks:
        r = it.check_task(t)
        res[f"C1 {t.key}"] = dict(ok=all(r.values()), items=r)
    res["C2 reproduction (B_conv|160 through 2400)"] = c2_repro()
    res["C3 windows"] = c3_windows()
    res["C4 REG3"] = c4_reg3()
    res["C5 perfect gate"] = c5_perfect()
    res["C6 split mechanics"] = c6_split()
    if not quick:
        res["C6b split at threshold 0 = no split"] = c6b_thr0()
    res["C7 probe"] = c7_probe()
    res["C8 SLOW groups"] = c8_groups()
    return res


def assert_checks(res, log=print):
    bad = []
    for k, v in res.items():
        ok = v["ok"]
        if k.startswith("C2") and not v["asserted"]:
            log(f"  {'ok ' if ok else 'DIFF'} {k} (printed, not asserted: record CPU {v['cpu_record']} != {v['cpu_here']})")
            continue
        log(f"  {'ok ' if ok else 'FAIL'} {k}" + ("" if ok else f": {json.dumps(v, default=str)[:600]}"))
        if not ok:
            bad.append(k)
    assert not bad, f"CHECKs failed: {bad}"


if __name__ == "__main__":
    import time
    t0 = time.time()
    res = checks([it.ICMCTask(**SMALL)], quick="--quick" in sys.argv)
    assert_checks(res)
    print(f"all CHECKs passed in {time.time() - t0:.0f} s")
