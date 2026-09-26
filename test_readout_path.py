#!/usr/bin/env python
"""
test_readout_path.py — why do the readout arms find the stream routing so rarely (A_ro 1/10
on each machine, against arm A's 27/50)? Two mechanism questions:
  - is the harm the readout's gradient flowing into the gate's recurrent state?
  - is a readout added after the gate has committed harmless?
Removing the readout is arm A itself. This test is not expected to raise the rate above arm
A's; it tests the mechanism.

Run it directly:

    python test_readout_path.py                     # CHECKs, projection, 4 arms x 30 seeds
    python test_readout_path.py --also results/L/readout_path_results.json

BACKGROUND
- Discovery rates: A_ro discovered 1/10 on X and 1/10 on L (test_router_discovery). A
  discovered 27/50 on X over seeds 0-49 (test_router_curriculum + test_router_confirm). The
  labelled 5% nudge discovered 27/30 on arm A but 2/20 on A_ro (X and L combined).
- Scratch diagnostic, on a container that matched X step for step, with the same nudge: on A
  the stream lean grew (sep 0.05 -> 0.42 and 0.36 by step 100); on A_ro it was erased
  (0.05 -> 0.011 and 0.005) while the gate hardened; at step 1 the gate-parameter gradient
  norm was 9-25x larger on A_ro (0.0027 and 0.0030, against 0.0003 and 0.0001 on A).
- Hypothesis: the readout gives W_in and W_h a first-order gradient that moves the gate off
  uniform, early and along non-stream splits (key identity or other). The routing term itself
  has zero first-order gradient at the uniform gate, since g_t.g_s = 1/2 + 2 d_t d_s.
- Seed-level outcomes differ between machines; the verdict is per machine.

SUBSTRATE AND TASK: test_router_confirm's, reused. BindTask P=4, S=2, n_vals=16, n_q=1;
MultiBDH n_layer=3, decay, N=256; lr 1e-3, BATCH 32; evaluation every 1200 steps on
eval_batch; early stop after 3 evaluations >= 0.95 (A_ro_late: only evaluations after T_RO
count); MAX_ITERS = 24000. DISCOVERED = transition is not None AND final VAL cos < 0.5.

SEEDS 50-79 (30 per arm), disjoint from every seed used before. Paired: every arm with the
same seed starts from the same shared parameters and sees the same batch stream.

ARMS
  A          exactly as in test_router_confirm.
  A_ro       exactly as in test_channel_binding.
  A_ro_sg    candidate "the harm is the readout's gradient into the gate": A_ro with the
             readout computed from h.detach(); W_ro still trains, but no gradient from the
             readout reaches W_in, W_h, W_g or the embedding through the gate's recurrent state.
  A_ro_late  candidate "the harm is timing": W_ro = 0 at build, frozen through step
             T_RO = 4800 (grad set to None before opt.step); right after opt.step at T_RO,
             W_ro = randn(seed + 99_999) * 0.1 from its own torch.Generator, trained from then
             on. Early stop counts only evaluations after T_RO; the transition is as usual.
             Through T_RO this arm is exactly arm A (CHECK 39).
Fixed before any result: the arms, the seeds, T_RO = 4800, the 0.1 re-init scale, the
generator seed and MAX_ITERS.

PRE-REGISTERED CLAIMS, Fisher's exact test one-sided on discovered counts of 30
  R1 READOUT HARMS         A vs A_ro (A higher): CONFIRMED if p < 0.05, else NOT SHOWN.
                           If NOT SHOWN, R2 and R3 are printed but not interpreted.
  R2 THE GRADIENT IS THE CAUSE  A_ro_sg vs A_ro (A_ro_sg higher): GRADIENT if p < 0.05,
                           else NOT SHOWN.
  R3 TIMING                A_ro_late vs A_ro (A_ro_late higher): TIMING HELPS if p < 0.05,
                           else NOT SHOWN.
Fisher and McNemar are test_router_confirm's.

DIAGNOSTICS, not part of the verdict: recovery (A_ro_sg vs A, A_ro_late vs A, Fisher two-
sided: "does the fix recover A's rate"; 30 seeds cannot show equivalence); gate-parameter
(W_in, W_h, W_g) and rest-of-model gradient norms at steps 1, 10 and 100 per arm (median and
range; the rest excludes W_ro); survival in A_ro_late around T_RO; a per-seed outcome table
across the four arms with exact McNemar p for A vs A_ro, A_ro_sg vs A_ro and A_ro_late vs A_ro;
escape step, failure modes and collapsed runs per arm; --also pooled counts.

LOGISTICS. Parallel single-thread workers; wall clock projected before training (no drop
rule). Results persist atomically to readout_path_results.json (gitignored; copied into
results/X/ after the run).

(Results are recorded at the bottom of this docstring after the run.)
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook

import test_binding_onset as tbo
from test_binding_onset import onset_run, attempt, fmt_step, EVAL_EVERY
import test_router_confirm as trcf
from test_router_confirm import fisher_greater, mcnemar_exact
import test_router_curriculum as trc
from test_router_curriculum import (
    TASK, ARCH, SUB_LR, GATE_PARAMS, make_fn, escape, get, count, mode_of, load_store,
    init_worker, makespan, cpu_model, same_weights, strip, save_results, probe_batch,
    gate_stats, time_arm, load_legacy_pair, DISC_COS, COLLAPSE, TAU_END,
)
from test_channel_binding import eval_batch, arms as cb_arms

# ── Settings (fixed before any result) ───────────────────────────────────────
MAX_ITERS = 24000
SEEDS = tuple(range(50, 80))
T_RO = 4800
RO_SCALE = 0.1
RO_OFFSET = 99_999
GRAD_STEPS = (1, 10, 100)
ALPHA = 0.05
LEGACY_SHA = "d5e3db6"
WORKERS = min(4, os.cpu_count() or 1)
RESULTS_FILE = "readout_path_results.json"
EARLIER_SEEDS = tuple(range(50))       # every seed used by earlier router tests

_CB = {a["key"]: a for a in cb_arms(TASK)}
ARMS = [
    dict(trc.ARM["A"], key="A", label="A          no readout (test_router_confirm)"),
    dict(_CB["A_ro"], key="A_ro", label="A_ro       readout (test_channel_binding)"),
    dict(_CB["A_ro"], key="A_ro_sg", label="A_ro_sg    readout through h.detach()",
         model_kw=dict(readout_sg=True)),
    dict(_CB["A_ro"], key="A_ro_late", label="A_ro_late  readout released at T_RO",
         ro_late=True),
]
ARM = {a["key"]: a for a in ARMS}
KEYS = tuple(ARM)


# ── The arms' pieces ─────────────────────────────────────────────────────────
def make_ro(a, seed):
    """test_router_curriculum.make_fn (build), plus W_ro = 0 for A_ro_late."""
    base = make_fn(a, seed)

    def make():
        m = base()
        if a.get("ro_late"):
            with torch.no_grad():
                m.W_ro.zero_()
        return m
    return make


def ro_init(shape, seed):
    g = torch.Generator().manual_seed(seed + RO_OFFSET)
    return torch.randn(shape, generator=g) * RO_SCALE


def grad_norms(m):
    gate, rest = 0.0, 0.0
    for n, p in m.named_parameters():
        if p.grad is None or n == "W_ro":
            continue
        v = float(p.grad.detach().pow(2).sum())
        if n in GATE_PARAMS:
            gate += v
        else:
            rest += v
    return math.sqrt(gate), math.sqrt(rest)


def hooks(a, seed, t_ro, probe, data, out):
    """grad_hook: freeze W_ro through t_ro (A_ro_late), then record gradient norms at
    GRAD_STEPS; post_step: release W_ro right after opt.step at t_ro."""
    late = bool(a.get("ro_late"))
    out["grad"] = {}

    def grad_hook(m, step):
        if late and step <= t_ro:
            m.W_ro.grad = None
        if step in GRAD_STEPS:
            out["grad"][str(step)] = grad_norms(m)

    def post_step(m, step):
        if not late or step != t_ro:
            return
        acc, _ = tbo.evaluate(m, TASK, data)
        out["acc_T"] = acc
        out["stats_T"] = gate_stats(m, TASK, probe, "T")
        with torch.no_grad():
            m.W_ro.copy_(ro_init(m.W_ro.shape, seed))
    return grad_hook, post_step


# ── Worker-side ──────────────────────────────────────────────────────────────
def run_spec(sp):
    """One arm, one seed: test_router_curriculum.run_spec's record, plus the readout path."""
    task, a, seed, t_ro = TASK, ARM[sp["arm"]], sp["seed"], sp["t_ro"]
    probe, data = probe_batch(task, seed), eval_batch(task)
    make, stats, extra = make_ro(a, seed), [], {}

    def make_and_measure():
        m = make()
        stats.append(gate_stats(m, task, probe, 0))
        return m

    def on_eval(m, step):
        stats.append(gate_stats(m, task, probe, step))

    def post(m):
        end = gate_stats(m, task, probe, "end")
        return dict(key_cos=end["key_cos"], val_cos=end["val_cos"], ctx_cos=end["ctx_cos"],
                    end=end)

    gh, ps = hooks(a, seed, t_ro, probe, data, extra)
    rec = attempt(task, make_and_measure, seed, data, post=post, max_iters=sp["iters"],
                  lr=SUB_LR, on_eval=on_eval, grad_hook=gh, post_step=ps,
                  early_stop_after=t_ro if a.get("ro_late") else None)
    rec.update(extra)
    rec.update(stats=stats, iters=sp["iters"], t_ro=t_ro if a.get("ro_late") else None)
    if rec["ok"]:
        rec["acc"] = rec["curve"][-1][1] if rec["curve"] else float("nan")
        rec["collapsed"] = rec["acc"] < COLLAPSE
        rec["discovered"] = bool(rec["transition"] is not None and rec["val_cos"] < DISC_COS)
    return rec


def spec(key, seed, iters, t_ro):
    return dict(arm=key, seed=seed, iters=iters, t_ro=t_ro)


# ── Two-sided Fisher (the diagnostics' recovery test) ────────────────────────
def fisher_two_sided(a, n1, b, n2):
    """Sum of the hypergeometric probabilities no larger than the observed table's."""
    N, K = n1 + n2, a + b
    tot = math.comb(N, n1)
    pr = lambda x: math.comb(K, x) * math.comb(N - K, n1 - x) / tot
    p0 = pr(a)
    lo, hi = max(0, K - n2), min(K, n1)
    return min(1.0, sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p0 * (1 + 1e-9)))


# ── Verification ─────────────────────────────────────────────────────────────
def verify(pool):
    print("test_router_confirm.py's verification (which runs test_router_curriculum's, and so")
    print("on down to test_multilayer_binding's):")
    trcf.verify(pool)
    print("=" * 100)
    print("VERIFICATION — this test's own CHECKs")
    print("=" * 100)
    ok = True
    t = TASK
    data = eval_batch(t)

    print(f"CHECK 37 the new knobs (MultiBDH readout_sg, onset_run early_stop_after) are inert at")
    print(f"         their defaults: 60-step runs (evaluations every 20) vs {LEGACY_SHA}'s modules:")
    leg_ml, leg_on = load_legacy_pair(LEGACY_SHA)
    print(f"         legacy MultiBDH takes readout_sg: "
          f"{'readout_sg' in leg_ml.MultiBDH.__init__.__code__.co_varnames}   legacy onset_run "
          f"takes early_stop_after: {'early_stop_after' in leg_on.onset_run.__code__.co_varnames}")
    for key in ("A", "A_ro"):
        a = ARM[key]
        m0, c0, _ = leg_on.onset_run(t, lambda: leg_ml.build(t, a, ARCH, 3), 3, max_iters=60,
                                     eval_every=20, data=data, early_stop=False, lr=SUB_LR)
        m1, c1, _ = onset_run(t, make_ro(a, 3), 3, max_iters=60, eval_every=20, data=data,
                              early_stop=False, lr=SUB_LR)
        good = same_weights(m0, m1) and c0 == c1
        ok &= good
        print(f"         {key:<5} weights equal {same_weights(m0, m1)}   curves equal {c0 == c1}   "
              f"accs {[round(x, 4) for _, x, _ in c1]}  -> {'IDENTICAL' if good else 'DIFFERS'}")
    print()

    print("CHECK 38 A_ro_sg changes gradients only:")
    x, y = [z for z in (lambda tok: (tok[:, :-1], tok[:, 1:]))(
        t.make_batch(32, torch.Generator().manual_seed(38))[0])]
    mr, ms = make_ro(ARM["A_ro"], 5)(), make_ro(ARM["A_ro_sg"], 5)()
    with torch.no_grad():
        same_logits = torch.equal(mr(x, TAU_END)[0], ms(x, TAU_END)[0])
    res = {}
    for key, m in (("A_ro", mr), ("A_ro_sg", ms)):
        with torch.no_grad():
            m.W_g.zero_()
        m.zero_grad(set_to_none=True)
        ql, qt, _ = t.select(m(x, TAU_END)[0], y, None)
        torch.nn.functional.cross_entropy(ql, qt).backward()
        res[key] = {n: (0.0 if getattr(m, n).grad is None else float(getattr(m, n).grad.abs().max()))
                    for n in ("W_in", "W_h", "W_ro")}
    good = (same_logits and res["A_ro_sg"]["W_in"] == 0.0 and res["A_ro_sg"]["W_h"] == 0.0
            and res["A_ro"]["W_in"] > 0 and res["A_ro"]["W_h"] > 0
            and res["A_ro"]["W_ro"] > 0 and res["A_ro_sg"]["W_ro"] > 0)
    ok &= good
    print(f"         init logits bitwise equal to A_ro's (seed 5): {same_logits}")
    print(f"         with W_g = 0 (uniform gate), max |grad|:  A_ro     W_in {res['A_ro']['W_in']:.3e}  "
          f"W_h {res['A_ro']['W_h']:.3e}  W_ro {res['A_ro']['W_ro']:.3e}")
    print(f"                                                   A_ro_sg  W_in {res['A_ro_sg']['W_in']:.3e}  "
          f"W_h {res['A_ro_sg']['W_h']:.3e}  W_ro {res['A_ro_sg']['W_ro']:.3e}  -> "
          f"{'GRADIENT PATH CUT, W_ro STILL TRAINS' if good else 'WRONG'}")
    print()

    s = 2
    print(f"CHECK 39 through T_RO A_ro_late is arm A: T_RO = 61, so a 60-step run (evaluations")
    print(f"         every 20, seed {s}) ends before the release:")
    a = ARM["A_ro_late"]
    gh, ps = hooks(a, s, 61, probe_batch(t, s), data, {})
    ml, cl, _ = onset_run(t, make_ro(a, s), s, max_iters=60, eval_every=20, data=data,
                          early_stop=False, lr=SUB_LR, grad_hook=gh, post_step=ps,
                          early_stop_after=61)
    ma, ca, _ = onset_run(t, make_ro(ARM["A"], s), s, max_iters=60, eval_every=20, data=data,
                          early_stop=False, lr=SUB_LR)
    pl, pa = dict(ml.named_parameters()), dict(ma.named_parameters())
    same_rest = sorted(pa) == sorted(n for n in pl if n != "W_ro") and all(
        torch.equal(pa[n], pl[n]) for n in pa)
    zero = not ml.W_ro.any()
    good = same_rest and cl == ca and zero
    ok &= good
    print(f"         non-readout weights bitwise equal to A's: {same_rest}   curves equal: {cl == ca}"
          f"   W_ro still exactly 0: {zero}  -> {'IS ARM A' if good else 'WRONG'}")
    print()

    print(f"CHECK 40 the release at T_RO (T_RO = 60, run 61 steps, seed {s}; optimizer observed by")
    print(f"         a step pre-hook):")
    holder, seen, rng_same, out = [], {}, {}, {}
    mk = make_ro(a, s)

    def mk2():
        holder.append(mk())
        return holder[-1]

    def pre(opt, args, kw):
        seen["n"] = seen.get("n", 0) + 1
        if seen["n"] == 61:
            seen["no_state"] = len(opt.state.get(holder[0].W_ro, {})) == 0
            seen["snap"] = holder[0].W_ro.detach().clone()

    gh, ps = hooks(a, s, 60, probe_batch(t, s), data, out)

    def ps2(m, step):
        before = torch.get_rng_state()
        ps(m, step)
        if step == 60:
            rng_same["v"] = torch.equal(before, torch.get_rng_state())
    h = register_optimizer_step_pre_hook(pre)
    try:
        onset_run(t, mk2, s, max_iters=61, lr=SUB_LR, grad_hook=gh, post_step=ps2)
    finally:
        h.remove()
    exact = torch.equal(seen["snap"], ro_init(holder[0].W_ro.shape, s))
    moved = not torch.equal(seen["snap"], holder[0].W_ro)
    # the counter: an evaluate stub that always returns 1.0 stops a plain run at the 3rd
    # evaluation; with early_stop_after it must wait for 3 evaluations after that step
    real = tbo.evaluate
    tbo.evaluate = lambda model, task, d: (1.0, 0.0)
    try:
        _, c_plain, _ = onset_run(t, make_ro(ARM["A"], s), s, max_iters=12, eval_every=2,
                                  data=data, lr=SUB_LR)
        _, c_late, _ = onset_run(t, make_ro(ARM["A"], s), s, max_iters=12, eval_every=2,
                                 data=data, lr=SUB_LR, early_stop_after=6)
    finally:
        tbo.evaluate = real
    stop_ok = c_plain[-1][0] == 6 and c_late[-1][0] == 12
    good = exact and seen["no_state"] and rng_same.get("v") and moved and stop_ok
    ok &= good
    print(f"         W_ro at T_RO equals randn(seed + {RO_OFFSET}) * {RO_SCALE} exactly: {exact}   "
          f"no Adam state for W_ro before T_RO+1: {seen['no_state']}   W_ro trains after: {moved}")
    print(f"         global RNG unchanged across the re-init: {rng_same.get('v')}")
    print(f"         early-stop counter (evaluate stubbed to 1.0, evaluations every 2 steps): plain "
          f"run stops at step {c_plain[-1][0]}; with early_stop_after=6 it stops at step "
          f"{c_late[-1][0]}  -> {'OK' if good else 'WRONG'}")
    print()

    print("CHECK 41 a worker's run is bit-identical to the same run here:")
    sp = spec("A_ro_late", 51, 1200, 600)
    rw = pool.submit(run_spec, sp).result()
    rh = run_spec(sp)
    good = strip(rw) == strip(rh) and rh["ok"] and "acc_T" in rh and set(rh["grad"]) == {"1", "10", "100"}
    ok &= good
    print(f"         A_ro_late seed 51, T_RO=600, 1200 steps: equal {strip(rw) == strip(rh)}   "
          f"grad norms at step 1 (gate, rest): {tuple(round(v, 5) for v in rh['grad']['1'])}  -> "
          f"{'IDENTICAL' if good else 'DIFFERS'}")
    print()
    print(f"{'ALL VERIFICATION CHECKS PASSED' if ok else 'SOME VERIFICATION CHECKS FAILED'}\n")
    assert ok, "verification failed — do not trust the results below"


# ── Reporting ────────────────────────────────────────────────────────────────
def disc(store, key, s):
    r = get(store, key, s)
    return bool(r and r.get("ok") and r.get("discovered"))


def report(store, wall, path, also):
    seeds, n = list(SEEDS), len(SEEDS)
    print("#" * 100)
    print("PER-SEED RAW RESULTS — every value, before any aggregate")
    print("#" * 100)
    print(f"  {'arm':<10} {'seed':>4} {'acc':>7} {'transition':>10} {'VAL cos':>8} {'sep':>7} "
          f"{'key_part':>8} {'spread':>7}  flags")
    for k in KEYS:
        for s in seeds:
            r = get(store, k, s)
            if r is None:
                print(f"  {k:<10} {s:>4}  NOT RUN")
                continue
            if not r.get("ok"):
                print(f"  {k:<10} {s:>4}  FAILED — {r['error']}")
                continue
            e = r["end"]
            fl = (["DISCOVERED"] if r["discovered"] else []) + (["collapsed"] if r["collapsed"] else [])
            print(f"  {k:<10} {s:>4} {r['acc']:>7.4f} {fmt_step(r['transition']):>10} "
                  f"{e['val_cos']:>8.4f} {e['sep']:>7.4f} {e['key_part']:>8.4f} {e['spread']:>7.4f}  "
                  f"{' '.join(fl)}")
        print()

    print("=" * 100)
    print(f"COUNTS (discovered = bound AND final VAL cos < {DISC_COS}), of {n} seeds")
    print("=" * 100)
    c = {}
    for a in ARMS:
        k = a["key"]
        c[k] = sum(disc(store, k, s) for s in seeds)
        rs = [get(store, k, s) for s in seeds]
        failed = sum(1 for r in rs if r is not None and not r.get("ok"))
        print(f"  {a['label']:<46} discovered {c[k]:>2}/{n}   bound {count(store, k, seeds, 'transition')}/{n}"
              f"   completed {sum(1 for r in rs if r and r.get('ok'))}/{n}"
              + (f"   [{failed} FAILED]" if failed else ""))
    print()

    print("#" * 100)
    print(f"PRE-REGISTERED CLAIMS — Fisher's exact test, one-sided, discovered counts of {n}")
    print("#" * 100)
    p1 = fisher_greater(c["A"], n, c["A_ro"], n)
    r1 = "CONFIRMED" if p1 < ALPHA else "NOT SHOWN"
    print(f"  R1 READOUT HARMS: A {c['A']}/{n} vs A_ro {c['A_ro']}/{n}; p = {p1:.3g}")
    print(f"     *** R1: {r1} ***")
    note = "" if r1 == "CONFIRMED" else "   (printed, NOT interpreted: R1 is NOT SHOWN)"
    p2 = fisher_greater(c["A_ro_sg"], n, c["A_ro"], n)
    r2 = "GRADIENT" if p2 < ALPHA else "NOT SHOWN"
    print(f"  R2 THE GRADIENT IS THE CAUSE: A_ro_sg {c['A_ro_sg']}/{n} vs A_ro {c['A_ro']}/{n}; "
          f"p = {p2:.3g}{note}")
    print(f"     *** R2: {r2} ***")
    p3 = fisher_greater(c["A_ro_late"], n, c["A_ro"], n)
    r3 = "TIMING HELPS" if p3 < ALPHA else "NOT SHOWN"
    print(f"  R3 TIMING: A_ro_late {c['A_ro_late']}/{n} vs A_ro {c['A_ro']}/{n}; p = {p3:.3g}{note}")
    print(f"     *** R3: {r3} ***")
    store["verdict"] = dict(R1=r1, R1_p=p1, R2=r2, R2_p=p2, R3=r3, R3_p=p3, counts=c,
                            interpreted=r1 == "CONFIRMED")
    save_results(path, store)
    print()

    print("=" * 100)
    print("DIAGNOSTICS — NOT part of the verdict")
    print("=" * 100)
    print("  does the fix recover A's rate (Fisher two-sided; 30 seeds cannot show equivalence):")
    for k in ("A_ro_sg", "A_ro_late"):
        print(f"    {k:<10} {c[k]}/{n} vs A {c['A']}/{n}: p = {fisher_two_sided(c[k], n, c['A'], n):.3g}")
    print()
    print("  gradient norms (gate = W_in, W_h, W_g; rest = all other parameters except W_ro), "
          "median [min, max] over seeds:")
    for step in GRAD_STEPS:
        for k in KEYS:
            g = [get(store, k, s)["grad"][str(step)] for s in seeds
                 if get(store, k, s) and get(store, k, s).get("ok") and str(step) in get(store, k, s).get("grad", {})]
            if not g:
                continue
            ga, re = [x[0] for x in g], [x[1] for x in g]
            print(f"    step {step:>3} {k:<10} gate {statistics.median(ga):.3e} [{min(ga):.3e}, "
                  f"{max(ga):.3e}]   rest {statistics.median(re):.3e} [{min(re):.3e}, {max(re):.3e}]")
    print()
    print(f"  survival in A_ro_late around T_RO = {T_RO} (gate separated = VAL cos < {DISC_COS} at the "
          f"evaluation at T_RO):")
    sep_n = sep_kept = uns_n = uns_found = 0
    for s in seeds:
        r = get(store, "A_ro_late", s)
        if not (r and r.get("ok")):
            continue
        st = next((x for x in r["stats"] if x["step"] == T_RO), None)
        if st is None:
            continue
        if st["val_cos"] < DISC_COS:
            sep_n += 1
            sep_kept += r["discovered"]
        else:
            uns_n += 1
            uns_found += r["discovered"]
    print(f"    separated at T_RO: {sep_n}; still separated and bound at the end: {sep_kept}")
    print(f"    unseparated at T_RO: {uns_n}; discovered later: {uns_found}")
    print()
    print("  per-seed outcome (D = discovered, . = not):")
    print("    seed  " + "  ".join(f"{k:>9}" for k in KEYS))
    for s in seeds:
        print(f"    {s:>4}  " + "  ".join(f"{('D' if disc(store, k, s) else '.'):>9}" for k in KEYS))
    for k1, k2 in (("A", "A_ro"), ("A_ro_sg", "A_ro"), ("A_ro_late", "A_ro")):
        b = sum(disc(store, k1, s) and not disc(store, k2, s) for s in seeds)
        cc = sum(disc(store, k2, s) and not disc(store, k1, s) for s in seeds)
        print(f"    McNemar exact two-sided, {k1} vs {k2}: {k1} only {b}, {k2} only {cc}; "
              f"p = {mcnemar_exact(b, cc):.3g}")
    print()
    print(f"  escape step (first evaluation with VAL cos < {DISC_COS}) per discovering run "
          f"(escape/transition):")
    for k in KEYS:
        es = [(s, escape(get(store, k, s)), get(store, k, s)["transition"]) for s in seeds
              if disc(store, k, s)]
        print(f"    {k:<10} " + ("  ".join(f"s{s}:{fmt_step(e)}/{fmt_step(tr)}" for s, e, tr in es)
                                 if es else "none"))
    print()
    print("  failure modes of non-discovering runs (saddle: spread < 0.1; locked on keys: key_part "
          "> 0.8 and sep < 0.2; other):")
    for k in KEYS:
        modes = {"saddle": 0, "locked on keys": 0, "other": 0}
        for s in seeds:
            r = get(store, k, s)
            if r and r.get("ok") and not r["discovered"]:
                modes[mode_of(r)] += 1
        print(f"    {k:<10} " + "   ".join(f"{m}: {v}" for m, v in modes.items()))
    print(f"  collapsed runs (final accuracy < {COLLAPSE}): "
          + ", ".join(f"{k} {count(store, k, seeds, 'collapsed')}" for k in KEYS))
    print()
    if also:
        print(f"  --also: pooled with {also}:")
        try:
            with open(also) as f:
                other = json.load(f)
            print(f"    other machine: {other.get('meta', {}).get('cpu', 'unknown')}, git "
                  f"{other.get('meta', {}).get('git')}")
            for k in KEYS:
                runs = [r for kk, r in other.get("runs", {}).items() if kk.split("|")[0] == k]
                k2 = sum(1 for r in runs if r.get("ok") and r.get("discovered"))
                print(f"    {k:<10} discovered here {c[k]}/{n}   other {k2}/{len(runs)}   pooled "
                      f"{c[k] + k2}/{n + len(runs)}")
        except Exception as e:
            print(f"    could not read {also}: {e}")
        print()

    print("=" * 100)
    print(f"CURVES — accuracy x100 and VAL cos x100 at each evaluation every {EVAL_EVERY} steps "
          f"('^' marks the evaluation at T_RO for A_ro_late)")
    print("=" * 100)
    for k in KEYS:
        for s in seeds:
            r = get(store, k, s)
            if not (r and r.get("ok")):
                continue
            ev = {x["step"]: x for x in r["stats"] if isinstance(x["step"], int) and x["step"] > 0}
            mark = lambda st: "^" if k == "A_ro_late" and st == T_RO else " "
            accs = "".join(f"{mark(st)}{round(acc * 100):>3}" for st, acc, _ in r["curve"])
            coss = "".join(f"{mark(st)}{round(ev[st]['val_cos'] * 100):>3}" if st in ev else "  --"
                           for st, _, _ in r["curve"])
            print(f"  {k:<10} s{s} acc {accs}")
            print(f"  {'':<10}     cos {coss}   -> "
                  f"{'bound @' + str(r['transition']) if r['transition'] else 'not bound'}"
                  f"{'  DISCOVERED' if r['discovered'] else ''}")
    print()
    print(f"  total wall clock after verification: {wall / 60:.1f} min ({wall:.0f}s)   raw records: "
          f"{path}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Why do the readout arms rarely find the routing?")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--results", default=RESULTS_FILE)
    ap.add_argument("--also", default=None,
                    help="a second machine's readout_path_results.json, for pooled counts")
    args = ap.parse_args()
    torch.set_num_threads(1)
    head = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()

    print("=" * 100)
    print("Readout path: is the readout's harm its gradient into the gate, or its timing?")
    print(f"  test_router_confirm's config (S=2 P=4 n_vals=16 n_q=1, n_layer=3 decay N=256, lr "
          f"{SUB_LR}), MAX_ITERS={MAX_ITERS}, T_RO={T_RO}")
    print(f"  seeds {SEEDS[0]}-{SEEDS[-1]} ({len(SEEDS)} per arm, paired); disjoint from earlier "
          f"seeds 0-49: {not set(SEEDS) & set(EARLIER_SEEDS)}")
    print(f"  torch {torch.__version__}   CPU {cpu_model()}   {args.workers} worker processes x 1 "
          f"thread   git {head}")
    print("=" * 100)
    print()

    os.environ.setdefault("PYTHONWARNINGS", "ignore:Failed to initialize NumPy")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx,
                             initializer=init_worker) as pool:
        verify(pool)
        store = load_store(args.results)
        store["meta"] = dict(torch=torch.__version__, cpu=cpu_model(), workers=args.workers,
                             git=head, max_iters=MAX_ITERS, t_ro=T_RO, seeds=list(SEEDS),
                             started=time.strftime("%Y-%m-%d %H:%M:%S"),
                             note="seed-level outcomes differ between machines")
        save_results(args.results, store)
        t0 = time.time()

        print("=" * 100)
        print(f"PROJECTION — time_per_step (200 steps) inside the pool, evaluation charged once per "
              f"{EVAL_EVERY}; worst case = every run to {MAX_ITERS}")
        print("=" * 100)
        costs = dict(zip(KEYS, pool.map(time_arm, [ARM[k] for k in KEYS])))
        for k in KEYS:
            print(f"  {k:<10} {costs[k][0] * 1000:6.1f} ms/step   x {len(SEEDS)} seeds")
        durs = [MAX_ITERS * costs[k][0] for k in KEYS for _ in SEEDS]
        print(f"  {len(durs)} runs: serial {sum(durs) / 3600:.2f} h, projected wall clock on "
              f"{args.workers} workers {makespan(durs, args.workers) / 3600:.2f} h (no drop rule)")
        store["meta"]["projected_wall_h"] = makespan(durs, args.workers) / 3600
        print()

        print("=" * 100)
        print(f"RUNS — {len(durs)} runs")
        print("=" * 100)
        todo = []
        for k in KEYS:
            for s in SEEDS:
                if get(store, k, s) is not None and not args.force:
                    print(f"   {k:<10} seed {s}  cached")
                else:
                    todo.append(spec(k, s, MAX_ITERS, T_RO))
        todo.sort(key=lambda sp: -costs[sp["arm"]][0])
        futs = {pool.submit(run_spec, sp): sp for sp in todo}
        for f in as_completed(futs):
            sp = futs[f]
            try:
                rec = f.result()
            except Exception as e:
                rec = dict(ok=False, seed=sp["seed"], error=f"worker {type(e).__name__}: {e}")
            store["runs"][f"{sp['arm']}|{sp['seed']}"] = rec
            save_results(args.results, store)
            el = (time.time() - t0) / 60
            if rec.get("ok"):
                e = rec["end"]
                tr = rec["transition"]
                print(f"   {sp['arm']:<10} seed {sp['seed']}  "
                      f"{'BOUND at ' + str(tr) if tr else 'not bound':<16} acc={rec['acc']:.4f} "
                      f"@{rec['stopped_at']:<5}  VALcos={e['val_cos']:.3f} sep={e['sep']:.3f} "
                      f"key_part={e['key_part']:.3f}{'  DISCOVERED' if rec['discovered'] else ''}"
                      f"  {rec['secs']:.0f}s  (elapsed {el:.1f}m)")
            else:
                print(f"   {sp['arm']:<10} seed {sp['seed']}  *** FAILED: {rec['error']}   "
                      f"(elapsed {el:.1f}m)")
        print()

    report(store, time.time() - t0, args.results, args.also)


if __name__ == "__main__":
    main()
