#!/usr/bin/env python
"""
explore_gate_cap_child.py — EXPLORATORY, not a result. S41's code that runs INSIDE a child process on the main
line's module tree at explore_main9c.MAIN_SHA (9c5939e). Parent: explore_gate_cap. explore_wh_spectrum_child
(S42) imports its measurement.

CAP: after every optimizer step (Muon: every SliceMuon step, the last of MuonAdam's steps; Adam: every step of the
run's torch.optim.Adam), if sigma_max(W_h) > 0.5, W_h <- W_h * 0.5 / sigma_max(W_h) (in place, under no_grad;
sigma_max in float64; the optimizer state untouched). The cap runs in a global optimizer-step post-hook, before
any measurement in the same hook. Recorded: sigma_max(W_h) at init, the first update the cap acts (and
sigma_max just before it), the number of updates it acts on.
NOREC: W_h fixed at 0 (zeroed after the build) and excluded from the optimizer (requires_grad False, so the
groups, built from trainable parameters, leave it out), so h_t = tanh(W_in v_t + W_prev v_(t-1)).
Arms (S=8, P=4, k=16, conv, all 8 streams from step 1: test_stream_curriculum.run_sc with ARM["D8"], 28800):
  CAP_M         S33's D8_HINGE (Muon lr 0.005, S25's groups, main's hinge) + CAP
  PREV_CAP_M    S40's D8_PREV_M (explore_gate_prev_child's GATE_PREV, Muon) + CAP
  PREV_NOREC_M  S40's D8_PREV_M with NOREC
  PREV_CAP_A    S40's D8_PREV_A (GATE_PREV, test_slow_start's HINGE recipe, Adam) + CAP
DIAGNOSTICS (measure()) at updates 600, 1200, 2400, 4800, 9600 (in the post-hook, after the cap) and at the
end (the model as the run returns it), on S38's 512-sequence probe (seed 38000; S40's): S38's decoder of the
stream from h at key and at stream-token positions and from the gate input u = W_in v_t (+ W_prev v_(t-1)) at
key positions; the median norms of W_in v_t, W_prev v_(t-1) and W_h h_(t-1) at key positions and of h there;
sigma_max(W_h) and rho(W_h); the fraction of h units with |h| > 0.95 at key positions; the distinct channels of
the value-position map (S39's rule: argmax of each stream's mean read gate at the value positions).
"""

import json

import torch

import explore_main9c as mt
import explore_muon_scale_child as s33c
import explore_gate_state_child as s38c
import explore_gate_prev_child as s40c
import explore_muon_recipe as s25
import test_slow_start as tss
import test_stream_recipe as tsr
from test_instrument_v2 import TAU_END

CAP = 0.5
AT = (600, 1200, 2400, 4800, 9600)
SAT = 0.95
PROBE_N, PROBE_SEED = s38c.PROBE_N, s38c.PROBE_SEED
CFG = "b"
ARM = {"CAP_M": dict(opt="muon", prev=False, cap=True, norec=False),
       "PREV_CAP_M": dict(opt="muon", prev=True, cap=True, norec=False),
       "PREV_NOREC_M": dict(opt="muon", prev=True, cap=False, norec=True),
       "PREV_CAP_A": dict(opt="adam", prev=True, cap=True, norec=False)}


def sigma_max(W):
    return float(torch.linalg.matrix_norm(W.detach().double(), ord=2))


def rho(W):
    return float(torch.linalg.eigvals(W.detach().double()).abs().max())


@torch.no_grad()
def terms(m, x):
    """The gate's recurrence on the model's embedding, term by term (the same operations as Instrument.gates and
    explore_gate_prev_child.prev_gates): a = W_in v_t, p = W_prev v_(t-1) (GATE_PREV only), u = a (+ p),
    b = W_h h_(t-1), h_t = tanh(u + b). Returns (a, p or None, u, b, h) as (B, T, .) tensors and the max
    difference between softmax(W_g h) and the model's own read gate."""
    was = m.training
    m.eval()
    try:
        assert m.gate_kind == "recurrent" and not m.gate_ln and m.mode == "sym"
        prev = isinstance(m, s40c.PrevBDH)
        v = m.embed(x)
        B, T, Dv = v.shape
        h = torch.zeros(B, m.h_gate, dtype=v.dtype)
        vp = torch.zeros(B, Dv, dtype=v.dtype)
        A, P, U, Bh, H = [], [], [], [], []
        for t in range(T):
            a = v[:, t] @ m.W_in.T
            if prev:
                pp = vp @ m.W_prev.T
                u = a + pp
                P.append(pp)
            else:
                u = a
            b = h @ m.W_h.T
            h = torch.tanh(u + b)
            A.append(a), U.append(u), Bh.append(b), H.append(h)
            vp = v[:, t]
        gr = m(x, TAU_END)[2]
    finally:
        m.train(was)
    st = (lambda L: torch.stack(L, 1))
    Hs = st(H)
    diff = float((torch.softmax(Hs @ m.W_g.T, -1) - gr).abs().max())
    return st(A), (st(P) if prev else None), st(U), st(Bh), Hs, gr, diff


def measure(m, task, data):
    x = data[:, :-1]
    S, n = task.S, task.S * task.P
    j = torch.arange(n)
    kp, sp, vpos = 3 * j + 1, 3 * j, 3 * j + 2
    A, P, U, Bh, H, gr, diff = terms(m, x)
    lab = x[:, sp]
    half = x.shape[0] // 2

    def dec(F_):
        return s38c.ncm(F_[:half].reshape(-1, F_.shape[-1]), lab[:half].reshape(-1),
                        F_[half:].reshape(-1, F_.shape[-1]), lab[half:].reshape(-1), S)

    def mnorm(Z):
        return float(Z[:, kp].norm(dim=-1).median())
    gv = gr[:, vpos]
    cmap = [int(gv[lab == s].mean(0).argmax()) for s in range(S)]
    return dict(h_key=dec(H[:, kp]), h_stream=dec(H[:, sp]), u_key=dec(U[:, kp]),
                a_key=mnorm(A), p_key=None if P is None else mnorm(P), b_key=mnorm(Bh), hn_key=mnorm(H),
                sigma=sigma_max(m.W_h), rho=rho(m.W_h), sat_key=float((H[:, kp].abs() > SAT).float().mean()),
                distinct=len(set(cmap)), ch_map=cmap, gate_recompute_diff=diff)


# ── The arms' recipes ────────────────────────────────────────────────────────
def make_rc(arm_name, holder, box, mlr=None, keep=None):
    """The arm's recipe: S33's (CAP_M) or S40's (the PREV arms, no S40 probe) recipe, with the outermost builder
    wrapper keeping the model in box (and, NOREC, zeroing and freezing W_h before the groups are built)."""
    arm = ARM[arm_name]
    if arm["prev"]:
        rc0, _ = s40c.make_rc(arm["opt"], holder, mlr=mlr, keep=keep)
    else:
        rc0, _ = s33c.make_rc("HINGE", mlr, holder, keep=keep)

    def rc(kw):
        kw = rc0(kw)
        b0 = kw["builder"]

        def builder(a, seed):
            mk = b0(a, seed)

            def make():
                m = mk()
                if arm["norec"]:
                    with torch.no_grad():
                        m.W_h.zero_()
                    m.W_h.requires_grad_(False)
                box["m"] = m
                box["builds"] = box.get("builds", 0) + 1
                box["sigma0"] = sigma_max(m.W_h)
                box["rho0"] = rho(m.W_h)
                return m
            return make
        kw["builder"] = builder
        return kw
    return rc


def hook(arm_name, box, cap, at, task, data, out, log=None):
    """The global optimizer-step post-hook: counts updates, applies the cap, logs (CHECKs), measures at `at`."""
    arm = ARM[arm_name]
    box.update(n=0, n_cap=0, first_cap=None, sigma_first=None)

    def post(opt, args, kwargs):
        if arm["opt"] == "muon" and not isinstance(opt, s25.SliceMuon):
            return
        if "m" not in box:
            return
        box["n"] += 1
        m = box["m"]
        if cap is not None:
            with torch.no_grad():
                s = sigma_max(m.W_h)
                if s > cap:
                    m.W_h.mul_(cap / s)
                    box["n_cap"] += 1
                    if box["first_cap"] is None:
                        box["first_cap"], box["sigma_first"] = box["n"], s
        if log is not None:
            log.append(dict(n=box["n"], sigma=sigma_max(m.W_h), wh_zero=bool(torch.count_nonzero(m.W_h) == 0),
                            wh_grad=m.W_h.grad is not None))
        if box["n"] in at:
            out[str(box["n"])] = measure(m, task, data)
    return post


def run(p):
    """p: arm, seed, iters, mlr (Muon), cap (override, e.g. "inf"; default the arm's: 0.5 or none), log (CHECKs:
    per-step log), keep_at."""
    import torch.optim.optimizer as topt
    arm_name = p["arm"]
    arm = ARM[arm_name]
    cap = (float(p["cap"]) if p.get("cap") is not None else (CAP if arm["cap"] else None))
    holder, box, out = {}, {}, {}
    log = [] if p.get("log") else None
    task = s33c.cfg_task(CFG)
    data = task.make_batch(PROBE_N, torch.Generator().manual_seed(PROBE_SEED))[0]
    a = s33c.cfg_arm(CFG, "HINGE")
    rc = make_rc(arm_name, holder, box, mlr=p.get("mlr"))
    hk = topt.register_optimizer_step_post_hook(hook(arm_name, box, cap, AT, task, data, out, log))
    try:
        if arm["opt"] == "muon":
            with s25.muon_in_onset_run(holder):
                rec = s33c.path(CFG, a, p["seed"], p["iters"], rc)
        else:
            rec = s33c.path(CFG, a, p["seed"], p["iters"], rc)
    finally:
        hk.remove()
    if rec.get("ok"):
        if "m" in box:
            out["end"] = measure(box["m"], task, data)
        groups = ([dict(names=n) for n in holder.get("names", [])] if arm["opt"] == "adam" else
                  [dict(tag=g["tag"], kind=g["kind"], names=g["names"]) for g in holder.get("groups", [])])
        rec.update(arm=arm_name, opt=arm["opt"], muon_lr=p.get("mlr") if arm["opt"] == "muon" else None,
                   cap=None if cap is None else (str(cap) if cap == float("inf") else cap), norec=arm["norec"],
                   prev=arm["prev"], diag=out, sigma0=box.get("sigma0"), rho0=box.get("rho0"), first_cap=box.get("first_cap"),
                   sigma_first=box.get("sigma_first"), n_cap=box.get("n_cap"), updates_counted=box.get("n"),
                   builds=box.get("builds"), maps=s33c.maps_of(rec), main_sha=mt.MAIN_SHA, opt_groups=groups,
                   wh_end_zero=bool(torch.count_nonzero(box["m"].W_h) == 0) if "m" in box else None)
        if log is not None:
            rec["log"] = log
        try:
            rec["outcome"] = tsr.outcome(rec)
        except Exception as e:
            rec["outcome"] = f"n/a ({type(e).__name__})"
    return json.loads(json.dumps(rec, default=float))


# ── CHECKs ───────────────────────────────────────────────────────────────────
STEPS_CHECK = 50


def checks(p):
    rows = []
    a = s33c.cfg_arm(CFG, "HINGE")
    b = tsr.builder_for(a)
    # (1) every CAP arm: sigma_max(W_h) <= 0.5 + 1e-6 after every one of 50 steps
    for arm_name in ("CAP_M", "PREV_CAP_M", "PREV_CAP_A"):
        r = run(dict(arm=arm_name, seed=260, iters=STEPS_CHECK, mlr=p["mlr"], log=True))
        lg = r["log"]
        mx = max(e["sigma"] for e in lg)
        rows.append((f"{arm_name}: sigma_max(W_h) <= 0.5 + 1e-6 after every one of {len(lg)} steps (max {mx:.7f}; at init "
                     f"{r['sigma0']:.4f}; the cap first acts at update {r['first_cap']} (sigma_max {r['sigma_first']:.4f} before "
                     f"it), on {r['n_cap']} of {len(lg)} updates)",
                     len(lg) == STEPS_CHECK and mx <= CAP + 1e-6 and r["builds"] == 1))
    # (2) NOREC: W_h exactly 0 at every step, no gradient, in no optimizer group; every other parameter's initial
    #     value equals D8_PREV_M's
    r = run(dict(arm="PREV_NOREC_M", seed=260, iters=STEPS_CHECK, mlr=p["mlr"], log=True))
    lg = r["log"]
    names = [n for gg in r["opt_groups"] for n in gg["names"]]
    m_prev = s40c.prev_builder(b)(a, 260)()
    box = {}
    m_nr = None
    rc = make_rc("PREV_NOREC_M", {}, box, mlr=p["mlr"])
    kw = rc(dict(builder=b, task=s33c.cfg_task(CFG), stats_fn=lambda *x: {}, check=None))
    m_nr = kw["builder"](a, 260)()
    pp, pn = dict(m_prev.named_parameters()), dict(m_nr.named_parameters())
    same = sorted(pp) == sorted(pn) and all(torch.equal(pp[k], pn[k]) for k in pp if k != "W_h")
    rows.append((f"PREV_NOREC_M: W_h exactly 0 after every one of {len(lg)} steps ({sum(e['wh_zero'] for e in lg)}/{len(lg)}), "
                 f"no gradient at any step ({sum(e['wh_grad'] for e in lg)} with one), W_h in no optimizer group (groups "
                 f"{[gg['names'] for gg in r['opt_groups']]}), W_h zero at the end {r['wh_end_zero']}; every other parameter's "
                 f"initial value equals D8_PREV_M's ({len(pp) - 1} parameters, W_prev included: {same}; W_h there "
                 f"{'zero' if torch.count_nonzero(m_nr.W_h) == 0 else 'NOT zero'}, requires_grad {m_nr.W_h.requires_grad})",
                 len(lg) == STEPS_CHECK and all(e["wh_zero"] and not e["wh_grad"] for e in lg) and "W_h" not in names
                 and r["wh_end_zero"] and same and torch.count_nonzero(m_nr.W_h) == 0 and not m_nr.W_h.requires_grad))
    # (3) the optimizer groups cover every trainable parameter once (W_prev in the gate group; W_h in it but NOREC)
    for arm_name, arm in ARM.items():
        holder, box = {}, {}
        rc = make_rc(arm_name, holder, box, mlr=p["mlr"])
        kw = rc(dict(builder=b, task=s33c.cfg_task(CFG), stats_fn=lambda *x: {}, check=None))
        m = kw["builder"](a, 260)()
        gs = kw["run_kw"]["param_groups"](m)
        ids = [id(q) for gg in gs for q in gg["params"]]
        train = {id(q) for q in m.parameters() if q.requires_grad}
        gate = {id(q) for q in gs[0]["params"]}
        ok = (len(ids) == len(set(ids)) and set(ids) == train and (id(m.W_h) in gate) == (not arm["norec"])
              and (not arm["prev"] or id(m.W_prev) in gate))
        nm = [gg.get("names") for gg in gs] if arm["opt"] == "muon" else holder.get("names")
        rows.append((f"{arm_name}: the optimizer groups cover every trainable parameter once ({nm})", ok))
    return [[n, bool(v)] for n, v in rows]


def timing(p):
    """Per arm: the median interval between optimizer steps over p["steps"] steps, the time of one evaluation
    (held-out set and the run path's statistics) and of one measurement."""
    import time
    import torch.optim.optimizer as topt
    import test_binding_onset as tbo
    import test_stream_curriculum as tscur
    from test_channel_binding import eval_batch
    from test_multilayer_binding import probe_batch
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
        task = s33c.cfg_task(CFG)
        a = s33c.cfg_arm(CFG, "HINGE")
        kw = make_rc(arm_name, {}, {}, mlr=p["mlr"])(dict(builder=tsr.builder_for(a), task=task, stats_fn=lambda *x: {},
                                                          check=None))
        m = kw["builder"](a, 0)()
        t0 = time.time()
        tbo.evaluate(m, task, eval_batch(task))
        tscur.sc_stats((tscur.T1, tscur.T2))(m, task, probe_batch(task, 0), 1200)
        te = time.time() - t0
        data = task.make_batch(PROBE_N, torch.Generator().manual_seed(PROBE_SEED))[0]
        t0 = time.time()
        measure(m, task, data)
        out[arm_name] = [d[len(d) // 2], te, time.time() - t0]
    return out
