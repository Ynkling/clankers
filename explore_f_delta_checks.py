#!/usr/bin/env python
"""
explore_f_delta_checks.py — EXPLORATORY, not a result. Session F: the CHECKs of explore_delta_mem.py (TASK 0), run
before anything else and at the start of every batch (S53, S54, S59). Every CHECK is asserted: the script exits
non-zero on any failure, and the batch drivers refuse to run.

Runs on this branch's own modules (no main-line hook), 1 torch thread.

  C1  parallel = sequential (the UT form against the definition), float64, on random tensors: B 3, T 17, n 24, d 8,
      k 3, soft gates, for β fixed (0, 0.5, 1) / per-token β, write tied / 1, decay fixed / routed / GDN-shaped random
      log α: forward max |diff| <= 1e-10 and the gradients with respect to keys, values, gates, b, w, log α <= 1e-8.
  C2  β = 0 (erase off, write 1), normalize=False: the model's logits and every layer's attention output equal the
      current (Hebbian, parallel-score) model's to 1e-5 — impl "parallel" and "sequential" — on four models: the
      S=2 LOCAL3 model (batch 1's), the S=2 perfect gate, the single channel (arm B, k = 1), and an S=8, k=16 LOCAL3
      model with the per-layer convolution (the S44 / S48 configuration's architecture); impl "hebb" equals the
      current model bit for bit (logits torch.equal) on the same four.
  C3  the oracle gate (k = S = 2, grouped and header layouts, β = 1, L2 keys, decay fixed and routed, parallel and
      sequential): every cross-stream contribution is exactly zero — (i) in each layer, the effective writes U^c are
      exactly 0 at every position outside stream c, and the output at a position of stream s takes exactly 0 from
      every channel c != s; (ii) the logits at every position of stream s are bit-identical when every VAL token of
      the other stream is replaced, and differ when a VAL token of stream s is.
  C4  overwrite: β = 1, normalize=True, one channel, decay 0.95, float32: after writing (k, v1) then (k, v2) the state
      returns v2 at key k, ‖S k̂ − v2‖ < 1e-5 (sequential states), also after a random history of 6 other writes
      before (k, v2); and the parallel read at a third position with key k equals 0.95·v2 to 1e-5.
  C5  initialisation: learned β is 0.5 exactly at init (w_b = 0, b_b = 0); routed decay gives α = 0.95 at g = 1 (to
      1e-6) and 1 exactly at g = 0; GDN decay: A in (0, 16), dt in [1e-4, 0.1], w_a = 0, two seeds differ, the same
      seed is the same; to_delta adds parameters only under attn and changes no existing parameter.
  C6  the recorded run, the new code path disabled: batch 16's S43 LOCAL3_SLOW seed 160 (explore_out/
      window_recipe_results.json, code 46867e792241, Intel(R) Xeon(R) Processor @ 2.10GHz), through 2400 updates,
      rerun through explore_window_recipe_child's own run path with the model converted by to_delta(impl="hebb"):
      curve, statistics and decoder at 1200 and 2400 bit for bit (only on that CPU; elsewhere the check is reported
      as not applicable and the batch must not pair bit for bit).

Usage: python explore_f_delta_checks.py [--tag TAG]   (writes explore_out/F/checks_<TAG>.json)
"""

import argparse
import copy
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch

torch.set_num_threads(1)

import explore_delta_mem as dm

REC_FILE = os.path.join(HERE, "explore_out", "window_recipe_results.json")
REC_KEY = "LOCAL3_SLOW|160|46867e792241|Intel(R) Xeon(R) Processor @ 2.10GHz"
REC_CPU = "Intel(R) Xeon(R) Processor @ 2.10GHz"
REC_ITERS = 2400
OUT_DIR = os.path.join(HERE, "explore_out", "F")


def cpu_model():
    import explore_common as ec
    return ec.cpu_model()


# ── C1 ───────────────────────────────────────────────────────────────────────
def c1():
    rows = []
    g = torch.Generator().manual_seed(1)
    B, T, n, d, k = 3, 17, 24, 8, 3
    dt = torch.float64

    def rnd(*s):
        return torch.randn(*s, generator=g, dtype=dt)
    kh0 = torch.nn.functional.normalize(torch.relu(rnd(B, T, n)) + 0.01, dim=-1)
    v0 = rnd(B, T, d)
    gr0 = torch.softmax(rnd(B, T, k), -1)
    gw0 = torch.softmax(rnd(B, T, k), -1)
    beta_tok = torch.sigmoid(rnd(B, T, 1))
    cases = []
    for bname, beta in (("beta0", 0.0), ("beta0.5", 0.5), ("beta1", 1.0), ("beta_tok", None)):
        for write in ("tied", 1.0):
            for dname in ("fixed", "routed", "gdn"):
                cases.append((bname, beta, write, dname))
    worst_f = worst_g = 0.0
    for bname, beta, write, dname in cases:
        kh, v, gr, gw = (x.clone().requires_grad_(True) for x in (kh0, v0, gr0, gw0))
        bt = beta_tok if beta is None else torch.full((B, T, 1), beta, dtype=dt)
        b = gw * bt
        w = b if write == "tied" else gw * write
        if dname == "fixed":
            la = None
        elif dname == "routed":
            la = (-(torch.nn.functional.softplus(rnd(B, T, 1)) * 0.07) * gw)
        else:
            la = -torch.rand(B, T, k, generator=g, dtype=dt) * 1.5
        if la is not None:
            la = la.detach().clone().requires_grad_(True)
        args = [kh, v, gr, b, w, la, 0.95]
        op, _ = dm.delta_parallel(*args)
        os_, _ = dm.delta_sequential(*args)
        df = float((op - os_).abs().max())
        tgt = rnd(B, T, d)
        ins = [kh, v, gr, gw] + ([la] if la is not None else [])
        gp = torch.autograd.grad((op * tgt).sum(), ins, retain_graph=True)
        gs = torch.autograd.grad((os_ * tgt).sum(), ins)
        dg = max(float((a - c).abs().max()) for a, c in zip(gp, gs))
        worst_f, worst_g = max(worst_f, df), max(worst_g, dg)
    rows.append((f"C1 parallel = sequential over {len(cases)} cases (beta 0 / 0.5 / 1 / per token, write tied / 1, decay "
                 f"fixed / routed / per-channel): forward max |diff| {worst_f:.2e} <= 1e-10, gradients {worst_g:.2e} <= 1e-8",
                 worst_f <= 1e-10 and worst_g <= 1e-8))
    return rows


# ── Models ───────────────────────────────────────────────────────────────────
def models():
    """name -> (model, task): the four architectures of C2 (branch modules)."""
    import explore_common as ec
    import explore_local_gate as s1
    import explore_b16_gates as g16
    import test_short_conv as tsc
    from test_multilayer_binding import build, MultiBDH
    from test_channel_binding import ARCH, arms as cb_arms, task_for
    from test_binding_capacity import BindTask
    t2 = ec.TASK
    ceil = {a["key"]: a for a in cb_arms(t2)}["ceiling"]
    out = {}
    out["LOCAL3 S=2 k=2"] = (s1.make_model(s1.ARMS["LOCAL3"], 160), t2)
    out["perfect S=2 k=2"] = (build(t2, ceil, ARCH, 160), t2)
    out["single k=1 (arm B)"] = (build(t2, tsc.ARM["B"], ARCH, 160), t2)
    t8 = BindTask(4, S=8, n_vals=16, n_q=1)
    torch.manual_seed(260)
    m8 = MultiBDH(t8.vocab, ARCH["n_layer"], "recurrent", 16, ARCH["positional"], mult=8, ctx_tokens=t8.ctx_tokens,
                  conv="layer")
    out["LOCAL3 S=8 k=16 conv"] = (g16.to_local(m8, 260), t8)
    return out


def attn_outputs(m, x):
    """Every layer's attention output (the module's return value), in order."""
    outs = []
    h = m.attn.register_forward_hook(lambda mod, inp, o: outs.append(o.detach()))
    try:
        with torch.no_grad():
            lg = m(x)[0]
    finally:
        h.remove()
    return lg, outs


# ── C2 ───────────────────────────────────────────────────────────────────────
def c2():
    rows = []
    for name, (m, task) in models().items():
        m.eval()
        x = task.make_batch(8, torch.Generator().manual_seed(5))[0][:, :-1]
        lg0, a0 = attn_outputs(m, x)
        for impl in ("parallel", "sequential"):
            md = dm.to_delta(copy.deepcopy(m), beta=0.0, write=1.0, normalize=False, impl=impl).eval()
            lg1, a1 = attn_outputs(md, x)
            dl = float((lg1 - lg0).abs().max())
            da = max(float((p - q).abs().max()) for p, q in zip(a1, a0))
            rel = max(float((p - q).abs().max() / q.abs().max()) for p, q in zip(a1, a0))
            rows.append((f"C2 {name}, {impl}, beta 0 / write 1 / raw keys: logits max |diff| {dl:.2e}, attention outputs "
                         f"{da:.2e} (relative {rel:.1e}) over {len(a1)} layers, all <= 1e-5", dl <= 1e-5 and da <= 1e-5
                         and len(a1) == len(a0) == 3))
        mh = dm.to_delta(copy.deepcopy(m), beta=1.0, impl="hebb").eval()
        lg2, _ = attn_outputs(mh, x)
        rows.append((f"C2 {name}, impl hebb (beta 1 set, path disabled): logits bit-identical to the current model",
                     torch.equal(lg2, lg0)))
    return rows


# ── C3 ───────────────────────────────────────────────────────────────────────
def c3():
    import explore_common as ec
    import explore_far_cue as s6
    from test_multilayer_binding import build
    from test_channel_binding import ARCH, arms as cb_arms
    rows = []
    for lname, task in (("grouped", ec.TASK), ("header", s6.HEADER)):
        ceil = {a["key"]: a for a in cb_arms(task)}["ceiling"]
        x = task.make_batch(16, torch.Generator().manual_seed(9))[0][:, :-1]
        lab = task.stream_labels(x)
        val = task.role_masks(x)["val"]
        S, P = task.S, task.P
        for decay in (None, "routed"):
            for impl in ("parallel", "sequential"):
                m = dm.to_delta(build(task, ceil, ARCH, 300), beta=1.0, decay=decay, impl=impl).eval()
                # (i) inside each layer: U^c zero outside stream c; outputs take nothing from other channels
                rec = []

                def hook(mod, args, kwargs, o):
                    Q, V = kwargs["Q"], kwargs["V"]
                    gr, gw = mod._gates
                    kh = mod.keys(Q[:, 0])
                    b, w = mod.coefs(V[:, 0], gw)
                    la = mod.log_alpha(V[:, 0], gw)
                    _, U = dm.delta_parallel(kh, V[:, 0], gr, b, w, la, mod.decay)
                    rec.append((U.detach(), gr.detach()))
                h = m.attn.register_forward_hook(hook, with_kwargs=True)
                with torch.no_grad():
                    lg0 = m(x)[0]
                h.remove()
                u_ok = True
                for U, gr in rec:
                    for c in range(S):
                        outside = (lab != c)                                     # (B, T)
                        u_ok &= bool((U[:, c][outside] == 0).all())
                        u_ok &= bool((gr[..., c][outside] == 0).all())
                # (ii) replace every VAL token of the other stream; logits of stream s unchanged bit for bit
                eq_ok, ch_ok = True, True
                g = torch.Generator().manual_seed(17)
                for s in range(S):
                    other = val & (lab != s) & (lab >= 0)
                    x2 = x.clone()
                    rnd = torch.randint(S + P, task.vocab, x.shape, generator=g)
                    x2[other] = rnd[other]
                    with torch.no_grad():
                        lg2 = m(x2)[0]
                    mine = (lab == s)
                    eq_ok &= torch.equal(lg2[mine], lg0[mine]) and bool((x2[other] != x[other]).any())
                    x3 = x.clone()
                    own = val & (lab == s)
                    x3[own] = (x3[own] - (S + P) + 1) % task.n_vals + S + P
                    with torch.no_grad():
                        lg3 = m(x3)[0]
                    ch_ok &= not torch.equal(lg3[mine], lg0[mine])
                rows.append((f"C3 oracle gate, {lname}, decay {decay or 'fixed'}, {impl}: U^c = 0 outside stream c and the read "
                             f"gate 0 on other channels in all {len(rec)} layers ({u_ok}); stream-s logits bit-identical with the "
                             f"other stream's VAL tokens replaced ({eq_ok}) and changed by its own ({ch_ok})",
                             u_ok and eq_ok and ch_ok and len(rec) == 3))
    return rows


# ── C4 ───────────────────────────────────────────────────────────────────────
def c4():
    rows = []
    g = torch.Generator().manual_seed(4)
    n, d = 256, 32
    k = torch.relu(torch.randn(n, generator=g))
    kh = torch.nn.functional.normalize(k, dim=0)
    v1, v2 = torch.randn(d, generator=g), torch.randn(d, generator=g)
    ones = torch.ones(1, 3, 1)
    keys = torch.stack([kh, kh, kh])[None]
    vals = torch.stack([v1, v2, torch.zeros(d)])[None]
    b = w = ones
    o_seq, st = dm.delta_sequential(keys, vals, ones, b, w, None, 0.95, keep_states=True)
    S1 = st[1][0, 0]                                                    # after (k, v1), (k, v2)
    e1 = float((S1 @ kh - v2).norm())
    o_par, _ = dm.delta_parallel(keys, vals, ones, b, w, None, 0.95)
    e2 = float((o_par[0, 2] - 0.95 * v2).norm())
    # a random history of 6 other writes (other keys, other values) between (k, v1) and (k, v2)
    hk = torch.nn.functional.normalize(torch.relu(torch.randn(6, n, generator=g)), dim=-1)
    hv = torch.randn(6, d, generator=g)
    keys2 = torch.cat([kh[None], hk, kh[None], kh[None]])[None]
    vals2 = torch.cat([v1[None], hv, v2[None], torch.zeros(1, d)])[None]
    T2 = keys2.shape[1]
    ones2 = torch.ones(1, T2, 1)
    _, st2 = dm.delta_sequential(keys2, vals2, ones2, ones2, ones2, None, 0.95, keep_states=True)
    e3 = float((st2[T2 - 2][0, 0] @ kh - v2).norm())
    o_par2, _ = dm.delta_parallel(keys2, vals2, ones2, ones2, ones2, None, 0.95)
    e4 = float((o_par2[0, T2 - 1] - 0.95 * v2).norm())
    rows.append((f"C4 overwrite (beta 1, L2 keys, one channel, decay 0.95, float32): ‖S k̂ − v2‖ = {e1:.2e} after (k, v1), "
                 f"(k, v2); {e3:.2e} with 6 other writes between; the parallel read at a later key-k position equals 0.95·v2 "
                 f"to {e2:.2e} / {e4:.2e}; all < 1e-5", max(e1, e2, e3, e4) < 1e-5))
    return rows


# ── C5 ───────────────────────────────────────────────────────────────────────
def c5():
    import explore_common as ec
    from test_multilayer_binding import build
    from test_channel_binding import ARCH, arms as cb_arms
    rows = []
    task = ec.TASK
    ceil = {a["key"]: a for a in cb_arms(task)}["ceiling"]
    m0 = build(task, ceil, ARCH, 300)
    p0 = {n: p.detach().clone() for n, p in m0.named_parameters()}
    ml = dm.to_delta(copy.deepcopy(m0), beta="learned")
    v = torch.randn(4, 9, 32)
    beta = ml.attn.beta_of(v)
    same = all(torch.equal(p, p0[n]) for n, p in ml.named_parameters() if n in p0)
    new = [n for n, _ in ml.named_parameters() if n not in p0]
    rows.append((f"C5 learned beta = 0.5 exactly at init ({bool((beta == 0.5).all())}); to_delta keeps every existing "
                 f"parameter ({same}) and adds {new}", bool((beta == 0.5).all()) and same and new == ["attn.w_b", "attn.b_b"]))
    mr = dm.to_delta(copy.deepcopy(m0), beta=1.0, decay="routed")
    gw = torch.tensor([[[1.0, 0.0], [0.3, 0.7]]])
    a = torch.exp(mr.attn.log_alpha(torch.randn(1, 2, 32), gw))
    rows.append((f"C5 routed decay at init: alpha at g = 1 {float(a[0, 0, 0]):.7f} (0.95 to 1e-6), at g = 0 "
                 f"{float(a[0, 0, 1])} (exactly 1), at g = 0.3 / 0.7 {float(a[0, 1, 0]):.5f} / {float(a[0, 1, 1]):.5f}; "
                 f"delta0 {dm.ROUTED_DELTA0:.4f}", abs(float(a[0, 0, 0]) - 0.95) < 1e-6 and float(a[0, 0, 1]) == 1.0))
    g1 = dm.to_delta(copy.deepcopy(m0), beta=1.0, decay="gdn", seed=350).attn.decay_mod
    g1b = dm.to_delta(copy.deepcopy(m0), beta=1.0, decay="gdn", seed=350).attn.decay_mod
    g2 = dm.to_delta(copy.deepcopy(m0), beta=1.0, decay="gdn", seed=351).attn.decay_mod
    A = torch.exp(g1.A_log.detach())
    dt = torch.nn.functional.softplus(g1.dt_bias.detach())
    ok = (bool(((A > 0) & (A < 16)).all()) and bool(((dt >= 1e-4 - 1e-9) & (dt <= 0.1 + 1e-7)).all())
          and bool((g1.w_a == 0).all()) and torch.equal(g1.A_log, g1b.A_log) and not torch.equal(g1.A_log, g2.A_log))
    al = torch.exp(-A * dt)
    rows.append((f"C5 GDN decay init (seed 350, k = 2): A {[round(x, 3) for x in A.tolist()]}, dt "
                 f"{[round(x, 5) for x in dt.tolist()]} (softplus(b_dt) recovers dt), alpha at init "
                 f"{[round(x, 4) for x in al.tolist()]}; w_a = 0; same seed same draw, another seed another", ok))
    return rows


# ── C6 ───────────────────────────────────────────────────────────────────────
def run_s43_delta(seed, iters, delta_kw):
    """explore_window_recipe_child.run(LOCAL3_SLOW) line for line, with to_delta applied to the built model."""
    import explore_common as ec
    import explore_common2 as c2_
    import explore_b16_gates as g16
    import explore_window_recipe_child as s43c
    import test_binding_onset as tbo
    import test_short_conv as tsc
    from test_router_reliability import run_one
    slow = True
    b0 = s43c.builder(slow)

    def builder(a, sd):
        mk = b0(a, sd)
        return lambda: dm.to_delta(mk(), seed=sd, **delta_kw)
    holder, out = {}, {}
    data = g16.probe_data(ec.TASK)
    kw = dict(check=s43c.make_check(slow, holder, out, data), task=ec.TASK, grad_fn=tsc.conv_grad_norms, lr=s43c.LR,
              builder=builder, stats_fn=c2_.stats_b2, run_kw=dict(param_groups=s43c.groups(holder)))
    rec = run_one(s43c.A, seed, iters, tbo.EVAL_EVERY, **kw)
    if rec.get("ok"):
        rec.update(decode=out)
    return json.loads(json.dumps(rec, default=float))


def c6():
    cpu = cpu_model()
    if cpu != REC_CPU:
        return [(f"C6 recorded run: NOT APPLICABLE on {cpu} (the record is from {REC_CPU}); do not pair bit for bit", None)]
    with open(REC_FILE) as f:
        rec = json.load(f)["runs"][REC_KEY]
    t = time.time()
    r = run_s43_delta(160, REC_ITERS, dict(beta=1.0, impl="hebb"))
    want = [c for c in rec["curve"] if c[0] <= REC_ITERS]
    sn = [s for s in r["stats"] if s["step"] != "end" and s["step"] <= REC_ITERS]
    so = [s for s in rec["stats"] if s["step"] != "end" and s["step"] <= REC_ITERS]
    dec = all(r["decode"][s] == rec["decode"][s] for s in ("1200", "2400"))
    ok = r["curve"] == want and sn == so and dec
    return [(f"C6 recorded run LOCAL3_SLOW|160 (batch 16, S43) through {REC_ITERS} with to_delta(impl='hebb'): curve "
             f"{r['curve']} vs {want}; statistics at {[s['step'] for s in sn]} equal ({sn == so}); decoder at 1200/2400 "
             f"equal ({dec}); bit for bit on {cpu} ({time.time() - t:.0f} s)", ok)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="task0")
    ap.add_argument("--skip", default="")
    args = ap.parse_args()
    import explore_common as ec
    pv = dict(cpu=cpu_model(), torch=torch.__version__, git=ec.git_head(), threads=torch.get_num_threads(),
              started=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"EXPLORATORY, not a result — explore_f_delta_checks ({args.tag}); {pv}", flush=True)
    rows = []
    for name, f in (("C1", c1), ("C2", c2), ("C3", c3), ("C4", c4), ("C5", c5), ("C6", c6)):
        if name in args.skip.split(","):
            continue
        t = time.time()
        for nm, v in f():
            rows.append([nm, v])
            print(f"  CHECK {nm}: {'ok' if v else ('n/a' if v is None else 'FAIL')}", flush=True)
        print(f"  ({name}: {time.time() - t:.1f} s)", flush=True)
    ok = all(v is not False for _, v in rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"checks_{args.tag}.json"), "w") as f:
        json.dump(dict(provenance=pv, ok=ok, rows=rows), f, indent=1)
    print(f"  ALL CHECKS {'PASSED' if ok else 'FAILED'} ({sum(v is True for _, v in rows)} ok, "
          f"{sum(v is None for _, v in rows)} n/a, {sum(v is False for _, v in rows)} failed)", flush=True)
    assert ok, "a CHECK failed"
    return rows


if __name__ == "__main__":
    main()
