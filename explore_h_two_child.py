#!/usr/bin/env python
"""
explore_h_two_child.py — EXPLORATORY, not a result. Session H's S78 code that runs INSIDE a child process on THIS branch's
modules (no main-line hook), as S43's and S52's. Parent: explore_h_two (readings in its docstring). Imports read-only:
explore_window_recipe_child (S43), explore_keysplits_child (S52, identical to claude/outside-ideas' file: CHECK on the
blob SHA), explore_h_stability_child (S60's ceiling_conv runner), explore_h_ops.

THE PATH: test_window_gate's Part A = S43's LOCAL3_SLOW (test_short_conv's TASK: S=2, P=4, k=2, grouped, no conv; batch 1's
LOCAL3 window gate; SLOW: gate W_in, W_g, window 1e-3 throughout, every other trainable parameter 1e-4 for updates 1-2400,
1e-3 after; 24000 updates; evaluation every 1200; early stop after 3 evaluations >= 0.95). test_window_gate's CHECK 135
asserts that its WIN3_SLOW equals S43's LOCAL3_SLOW bit for bit; its statistics function adds test_slow_start's extra
statistics, which change no training.
  REF           S43's LOCAL3_SLOW (= the main line's WIN3_SLOW), run by explore_window_recipe_child.run unchanged.
  SPLIT2        REF + the k = 2 KEYMASS trigger below with copy_exact (W_g[c0] := W_g[c*], no noise, no Adam reset).
  SPLIT2_NOISE  REF + the same trigger with split_noise_reset (S36's operation: 0.1-std noise on both rows, W_g's Adam
                state zeroed: the main line's form).
  K4            S52's W2_K4 (explore_keysplits_child.run, arm W2_K4: k = 4, two spare channels) unchanged, 24000 updates.
  ORACLE        test_short_conv's ceiling_conv on S43's run_one call (explore_h_stability_child.run_oracle; the main line's
                CEIL_A).
THE TRIGGER AT k = 2 (S48's rule, with this path's modules): after the evaluations at 2400 (reference), 4800, 7200, ...,
21600 (= 24000 - 2400), after the path's own check (lr switch, decoder): held-out accuracy on S36's 64-sequence probe
(task.make_batch(64, Generator seeded 12345)); fire if it is < 0.95 and rose < 0.02 since the previous check, at most 3
times, >= 4800 updates apart (capped or gapped checks logged as blocked). Target: c* = the channel with the larger mean
read-gate mass at the probe's KEY positions 3j+1, c0 = the other channel (at k = S no channel is idle: c0 is a busy row).
onset_run's Adam is built through a capturing factory (the same constructor). Logged at every check, and right after an
operation: routing_stats on the run's probe (eta^2 of the read gate's channel-0 probability at KEY and at VAL positions
by stream, key, half, index; the query routing margin) and the key-position masses.
"""

import json

import torch

import explore_common as ec
import explore_common2 as c2
import explore_window_recipe_child as s43c
import explore_keysplits_child as s52c
import explore_h_stability_child as h60c
import explore_h_ops as ops
import explore_b16_gates as g16
import test_binding_onset as tbo
import test_short_conv as tsc
from test_router_layout import routing_stats
from test_router_reliability import run_one
from test_multilayer_binding import probe_batch
from test_instrument_v2 import TAU_END

EVERY, FIRST, REF_AT = 2400, 4800, 2400
ACC_THR, RISE, GAP, CAP = 0.95, 0.02, 4800, 3
PROBE_N, PROBE_SEED, NOISE_BASE = 64, 12345, 12_000_000
OPS = {"SPLIT2": ops.copy_exact, "SPLIT2_NOISE": ops.split_noise_reset}
ARM = {"REF": dict(), "SPLIT2": dict(op="SPLIT2"), "SPLIT2_NOISE": dict(op="SPLIT2_NOISE"), "K4": dict(k4=True),
       "ORACLE": dict(oracle=True)}
_ADAM = torch.optim.Adam


class adam_capture:
    def __init__(self, holder):
        self.holder = holder

    def __enter__(self):
        h = self.holder

        def factory(params, lr=1e-3, **kw):
            opt = _ADAM(params, lr=lr, **kw)
            h.setdefault("opts", []).append(opt)
            return opt
        torch.optim.Adam = factory

    def __exit__(self, *a):
        torch.optim.Adam = _ADAM


@torch.no_grad()
def key_masses(m, data, S, P):
    was = m.training
    m.eval()
    try:
        gr = m(data[:, :-1], TAU_END)[2]
    finally:
        m.train(was)
    j = torch.arange(S * P)
    return gr[:, 3 * j + 1].mean((0, 1))


DIAG = ("eta_key_by_key", "eta_key_by_stream", "eta_val_by_key", "eta_val_by_stream", "eta_key_by_half", "eta_key_by_index",
        "margin")


def diag(m, task, rprobe):
    rs = routing_stats(m, task, rprobe)
    return {k: round(float(rs[k]), 6) for k in DIAG}


def make_trigger(task, seed, info, opt_of, last, op, thr=ACC_THR, rise=RISE, every=EVERY, first=FIRST, ref_at=REF_AT,
                 gap=GAP, cap=CAP):
    data = task.make_batch(PROBE_N, torch.Generator().manual_seed(PROBE_SEED))[0]
    rprobe = probe_batch(task, seed)
    st = dict(prev=None, n=0, last=None)

    def trig(m, step):
        if step % every or step < ref_at or step > last:
            return
        acc, _ = tbo.evaluate(m, task, data)
        km = key_masses(m, data, task.S, task.P)
        cs, c0 = int(km.argmax()), int(km.argmin())
        row = dict(step=step, acc=acc, prev=st["prev"], key_masses=[round(float(v), 5) for v in km], cs=cs, c0=c0,
                   diag=diag(m, task, rprobe), fired=False, blocked=False)
        eligible = step >= first and st["prev"] is not None and acc < thr and acc - st["prev"] < rise
        capped = st["n"] >= cap or (st["last"] is not None and step - st["last"] < gap)
        row["eligible"] = eligible
        if eligible and capped:
            row.update(blocked=True, blocked_by="cap" if st["n"] >= cap else "gap")
        elif eligible:
            res = op(m, opt_of(), cs, c0, NOISE_BASE + 1000 * seed + st["n"])
            st["n"] += 1
            st["last"] = step
            row.update(fired=True, diag_after=diag(m, task, rprobe),
                       key_masses_after=[round(float(v), 5) for v in key_masses(m, data, task.S, task.P)], **res)
        st["prev"] = acc
        info["checks"].append(row)
    return trig


def run_split(p):
    """S43's run (explore_window_recipe_child.run with arm LOCAL3_SLOW), its check followed by the trigger."""
    holder, info = {}, dict(checks=[])
    out = {} if p.get("probe", True) else None
    data = g16.probe_data(ec.TASK)
    c0_ = s43c.make_check(True, holder, out, data)
    trig = make_trigger(ec.TASK, p["seed"], info, lambda: holder["opts"][0], p.get("total", s43c.ITERS) - EVERY, OPS[p["arm"]],
                        thr=p.get("thr", ACC_THR))

    def check(m, step, d):
        c0_(m, step, d)
        trig(m, step)
    kw = dict(check=check, task=ec.TASK, grad_fn=tsc.conv_grad_norms, lr=s43c.LR, builder=s43c.builder(True),
              stats_fn=c2.stats_b2, run_kw=dict(param_groups=s43c.groups(holder)))
    with adam_capture(holder):
        rec = run_one(s43c.A, p["seed"], p["iters"], tbo.EVAL_EVERY, **kw)
    if rec.get("ok"):
        rec.update(arm=p["arm"], decode=out, group_names=holder.get("names"),
                   fail=None if rec["transition"] is not None else ec.fail_class(rec), tag=ec.tag(rec),
                   checks=info["checks"], splits=sum(1 for c in info["checks"] if c["fired"]), n_opts=len(holder.get("opts", [])))
    return json.loads(json.dumps(rec, default=float))


def run(p):
    arm = ARM[p["arm"]]
    if arm.get("oracle"):
        return h60c.run_oracle(p, "ceiling_conv")
    if arm.get("k4"):
        rec = s52c.run(dict(arm="W2_K4", seed=p["seed"], iters=p["iters"]))
        rec["arm"] = "K4"
        return rec
    if arm.get("op"):
        return run_split(p)
    rec = s43c.run(dict(arm="LOCAL3_SLOW", seed=p["seed"], iters=p["iters"]))
    rec["arm"] = "REF"
    return rec


def checks(p):
    """The trigger's mechanics at k = 2 on a LOCAL3_SLOW model after one Adam step (no training between checks; threshold
    above 1, rise 1 so every check from 4800 is eligible)."""
    rows = []
    task = ec.TASK
    m = g16.freeze_wh(s43c.s1.make_model(s43c.A, 1300))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    x = task.make_batch(8, torch.Generator().manual_seed(11))[0]
    ql, qt, _ = task.select(tbo.logits_of(m, x[:, :-1]), x[:, 1:], None)
    loss = torch.nn.functional.cross_entropy(ql, qt)
    opt.zero_grad(); loss.backward(); opt.step()
    for name in ("SPLIT2", "SPLIT2_NOISE"):
        mm = g16.freeze_wh(s43c.s1.make_model(s43c.A, 1300))
        mm.load_state_dict(m.state_dict())
        o = torch.optim.Adam(mm.parameters(), lr=1e-3)
        o.load_state_dict(opt.state_dict())
        st0 = {k: v.clone() for k, v in o.state[mm.W_g].items()}
        info = dict(checks=[])
        trig = make_trigger(task, 1300, info, lambda: o, 21600, OPS[name], thr=1.01, rise=1.0)
        W0 = mm.W_g.detach().clone()
        for t in range(2400, 21601, 2400):
            trig(mm, t)
        r = info["checks"]
        fired = [c["step"] for c in r if c["fired"]]
        f0 = next(c for c in r if c["fired"])
        busy = min(f0["key_masses"]) > 0.05
        rows.append((f"{name}: fires at {fired} (cap 3, gap 4800), gap-blocked at {[c['step'] for c in r if c.get('blocked_by') == 'gap']}, "
                     f"capped at {[c['step'] for c in r if c.get('blocked_by') == 'cap']}; first target c* {f0['cs']} -> c0 {f0['c0']} with "
                     f"key masses {f0['key_masses']} (c0 is a busy row: {busy})",
                     fired == [4800, 9600, 14400] and f0["cs"] != f0["c0"] and busy))
        st1 = o.state[mm.W_g]
        if name == "SPLIT2":
            ok = torch.equal(mm.W_g[0].detach(), mm.W_g[1].detach()) and all(torch.equal(st1[k], st0[k]) for k in st0)
            rows.append((f"SPLIT2: after a firing the two rows are equal bit for bit (exact copy) and W_g's Adam state is untouched ({ok})",
                         ok))
        else:
            ok = (not torch.equal(mm.W_g[0].detach(), mm.W_g[1].detach())
                  and all(float(v.abs().max()) == 0 for v in st1.values() if torch.is_tensor(v)))
            rows.append((f"SPLIT2_NOISE: the rows differ by the noise and W_g's Adam state is zero after the firings ({ok})", ok))
    return [[n, bool(v)] for n, v in rows]
