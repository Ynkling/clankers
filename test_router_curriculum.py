#!/usr/bin/env python
"""
test_router_curriculum.py — on arm A (recurrent gate, no readout), does holding the gate at
uniform until the model has solved key binding, then releasing it, let the gate discover
the stream routing reliably, without the stream labels?

Run it directly:

    python test_router_curriculum.py                # CHECKs, projection, 6 arms x 10 seeds
    python test_router_curriculum.py --also other_machine/router_curriculum_results.json

BACKGROUND
- Arm A with a learned gate discovered routing 0/10 in test_channel_binding, across both
  machines. The perfect gate bound 10/10; single-channel runs bound 0/70.
- The saddle: with W_g = 0 the gate is exactly uniform and its gradient is exactly 0
  (G = g.g has zero first-order change at uniform), so the gate stays uniform while the
  rest of the model trains.
- Hypothesis: early in training the loss is dominated by key confusion, which pulls a
  moving gate toward key-identity splits. Once the model sits at its 0.5 plateau only the
  stream conflict is left, so a gate released then should move toward stream routing.
- Scratch check (one machine, arm A, P=4, switch at step 12000): A_late seeds 0 and 2 bound
  at 18000 and 13200, seed 1 not by 21600; A_late_ln seeds 0 and 2 bound at 14400 and 13200,
  seed 1 not by 21600. The two versions share the model up to the switch.
- test_router_discovery (INVALID) ran its candidates on A_ro; its POST-HOC note records that
  the labelled nudge works on A but is erased on A_ro, whose readout moves the gate off
  uniform along non-stream splits. This test therefore works on A.

SUBSTRATE AND TASK: test_router_discovery's, reused. BindTask P=4, S=2, n_vals=16, n_q=1;
MultiBDH n_layer=3, decay, N=256; Adam lr 1e-3, BATCH 32; onset_run evaluating every 1200
steps on eval_batch, early stop after 3 evaluations >= 0.95. MAX_ITERS = 24000 (switch +
12000). Seeds 0-9.

ARMS, 10 seeds each
  A          test_channel_binding's A exactly. Baseline.
  A_nudge    POSITIVE CONTROL, uses the stream labels, never a candidate:
             test_router_discovery's 52.5/47.5 nudge, applied to A.
  A_late     candidate: W_g = 0 at build; W_in, W_h, W_g frozen through step T = 12000
             (their grads set to None before opt.step, so Adam never touches them); right
             after opt.step at step T, W_g = randn * 0.1 from a torch.Generator seeded
             seed + 88_888; from step T+1 all gate parameters train normally.
  A_late_ln  candidate: A_late with the gate reading F.layer_norm(embed(tokens), (D,)).
  A_ln       control: A with that LayerNorm from step 0 and no curriculum (separates the
             timing effect from the LayerNorm effect).
  B          plain single channel, a reference.
Fixed before any result: T = 12000, the 0.1 re-init scale, the generator seed, MAX_ITERS,
the seeds.

RECORD: exactly test_router_discovery's (curve, transition, final VAL/KEY/CTX cos; sep,
spread, key_part at step 0, every evaluation and the end, under eval() and no_grad). For
A_late and A_late_ln also the held-out accuracy and gate stats at T just before the
re-init, and the gate stats right after it. The re-init runs right after opt.step at T,
before that step's scheduled evaluation, so the curve's point at T is post-re-init.
DISCOVERED = transition is not None AND final VAL cos < 0.5.

PRE-REGISTERED VERDICT, exactly one of
  INVALID       A_nudge discovers on fewer than 8/10 seeds.
  ROUTER FOUND  A_late or A_late_ln discovers on >= 8/10.
  PARTIAL       the best candidate discovers on 3-7/10.
  NOT FOUND     both candidates discover on <= 2/10.
A, A_ln and B are printed beside the candidates; they are never candidates.

DIAGNOSTICS, not part of the verdict: escape step (first evaluation with VAL cos < 0.5),
as steps after T for the curriculum arms; accuracy at T for every curriculum run (the
hypothesis needs it near 0.5); failure modes of non-discovering channel runs (saddle /
locked on keys / other) per arm; A_ln vs A_late_ln; pooled counts across the two machines
when the other machine's results file is given with --also.

LOGISTICS. Parallel single-thread workers as in test_binding_recipe.py; wall clock projected
before training (no drop rule). Results persist atomically to router_curriculum_results.json
(gitignored). Seed-level outcomes differ between machines.

RESULT (full run, 60/60 runs complete, no failures; torch 2.14.0, Intel Xeon @ 2.10GHz,
4 workers x 1 thread; 81 min after verification; no --also file, so not pooled).
  PRE-REGISTERED VERDICT: PARTIAL. The positive control is valid (A_nudge 9/10, all by
  7200; seed 8 locked on keys). Best candidate A_late_ln 7/10 (seeds 0,2,3,5,6,7,8);
  A_late 6/10 (seeds 0,2,4,6,7,8). Never candidates: A 4/10, A_ln 7/10, B bound 0/10.

  Diagnostics (not part of the verdict):
  - The precondition held: held-out accuracy at T was 0.445-0.508 in all 20 curriculum
    runs. The re-init left the gate near uniform and unseparated (sep <= 0.016; spread
    0.015-0.043 without LayerNorm, 0.088-0.264 with it). Escapes came T+1200 to T+6000,
    one at T+10800 (A_late seed 6); every escape coincided with the transition.
  - The curriculum's effect is not separated from its controls with 10 seeds: A_late 6/10
    vs A 4/10, and A_late_ln 7/10 vs A_ln 7/10 exactly. LayerNorm on the gate input alone,
    from step 0, discovered as often as with the curriculum, and early (escape at the
    first evaluation, transitions 1200-6000).
  - A's 4/10 is not in tension with test_channel_binding's A 0/5 on this machine: seeds
    0-4 fail here too; A's successes are seeds 6, 7, 9 (bound at 2400) and 8 (15600),
    seeds the channel test never ran. On seed 9 the curriculum turned an early
    spontaneous success into a failure (A bound at 2400; A_late and A_late_ln did not).
  - How runs failed differs with the curriculum: none of the 7 non-discovering curriculum
    runs locked on keys (key_part <= 0.71; 2 left near uniform, spread < 0.1, the rest
    weakly spread and unseparated), while 4 of the 9 non-discovering A and A_ln runs
    locked on keys (key_part >= 0.999 in 3 of them).
"""

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook

import test_router_discovery as trd
from test_router_discovery import (
    TASK, ARCH, SUB_LR, GATE_PARAMS, gate_stats, model_fn, load_legacy_pair, strip, get, count,
    mode_of, load_store, init_worker, makespan, cpu_model, same_weights, probe_batch, save_results,
    DISC_COS, FOUND_K, PARTIAL_K, COLLAPSE, TIME_STEPS, TAU_END,
)
from test_binding_onset import onset_run, evaluate, attempt, time_per_step, fmt_step, EVAL_EVERY
from test_channel_binding import eval_batch, arms as cb_arms

# ── Settings (fixed before any result) ───────────────────────────────────────
T_SWITCH = 12000
MAX_ITERS = T_SWITCH + 12000
SEEDS = tuple(range(10))
REINIT_SCALE = 0.1
REINIT_OFFSET = 88_888
LEGACY_SHA = "4422dc4"
WORKERS = min(4, os.cpu_count() or 1)
RESULTS_FILE = "router_curriculum_results.json"
CANDIDATES = ("A_late", "A_late_ln")
REFERENCES = ("A", "A_ln", "B")

_BASE = {a["key"]: a for a in cb_arms(TASK)}
ARMS = [
    dict(_BASE["A"], key="A", label="A          baseline (test_channel_binding)"),
    dict(_BASE["A"], key="A_nudge", label="A_nudge    POSITIVE CONTROL (labels)", nudge=True),
    dict(_BASE["A"], key="A_late", label="A_late     gate held uniform to T, then released",
         late=True),
    dict(_BASE["A"], key="A_late_ln", label="A_late_ln  A_late + LayerNorm gate input",
         late=True, model_kw=dict(gate_ln=True)),
    dict(_BASE["A"], key="A_ln", label="A_ln       LayerNorm gate input, no curriculum",
         model_kw=dict(gate_ln=True)),
    dict(_BASE["B"], key="B", label="B          plain single channel (reference)"),
]
ARM = {a["key"]: a for a in ARMS}
CHANNEL_KEYS = tuple(a["key"] for a in ARMS if a["n_ch"] > 1)
FLOOR = _BASE["floor"]


# ── The curriculum ───────────────────────────────────────────────────────────
def make_fn(a, seed):
    """test_router_discovery.model_fn (build, plus the nudge), and W_g = 0 for A_late*."""
    base = model_fn(TASK, a, seed)

    def make():
        m = base()
        if a.get("late"):
            with torch.no_grad():
                m.W_g.zero_()
        return m
    return make


def reinit_w_g(shape, seed):
    g = torch.Generator().manual_seed(seed + REINIT_OFFSET)
    return torch.randn(shape, generator=g) * REINIT_SCALE


def curriculum(a, seed, T, probe, data, out):
    """grad_hook freezes the gate through step T; post_step re-initialises W_g right after
    opt.step at T, recording the held-out accuracy and gate stats just before and after."""
    if not a.get("late"):
        return None, None

    def grad_hook(m, step):
        if step <= T:
            for n in GATE_PARAMS:
                getattr(m, n).grad = None

    def post_step(m, step):
        if step != T:
            return
        acc, _ = evaluate(m, TASK, data)
        before = gate_stats(m, TASK, probe, "T")
        with torch.no_grad():
            m.W_g.copy_(reinit_w_g(m.W_g.shape, seed))
        out.update(acc_T=acc, stats_T=before, stats_reinit=gate_stats(m, TASK, probe, "reinit"))
    return grad_hook, post_step


# ── Worker-side ──────────────────────────────────────────────────────────────
def run_spec(sp):
    """One arm, one seed: test_router_discovery.run_spec's record, plus the curriculum."""
    # sp may carry the arm itself as "arm_def" (test_router_confirm's E_mem); else by name.
    a = sp["arm_def"] if "arm_def" in sp else ARM[sp["arm"]]
    task, seed, T = TASK, sp["seed"], sp["T"]
    chan = a["n_ch"] > 1
    probe = probe_batch(task, seed) if chan else None
    data = eval_batch(task)
    make, stats, extra = make_fn(a, seed), [], {}

    def make_and_measure():
        m = make()
        if chan:
            stats.append(gate_stats(m, task, probe, 0))
        return m

    def on_eval(m, step):
        stats.append(gate_stats(m, task, probe, step))

    def post(m):
        if not chan:
            return {}
        end = gate_stats(m, task, probe, "end")
        return dict(key_cos=end["key_cos"], val_cos=end["val_cos"], ctx_cos=end["ctx_cos"],
                    end=end)

    gh, ps = curriculum(a, seed, T, probe, data, extra)
    rec = attempt(task, make_and_measure, seed, data, post=post, max_iters=sp["iters"],
                  lr=SUB_LR, on_eval=on_eval if chan else None, grad_hook=gh, post_step=ps)
    rec.update(extra)
    rec.update(stats=stats, iters=sp["iters"], T=T if a.get("late") else None)
    if rec["ok"]:
        rec["acc"] = rec["curve"][-1][1] if rec["curve"] else float("nan")
        rec["collapsed"] = rec["acc"] < COLLAPSE
        rec["discovered"] = bool(chan and rec["transition"] is not None
                                 and rec["val_cos"] < DISC_COS)
    return rec


def time_arm(key):
    """ms/step with evaluation charged once per EVAL_EVERY, plus a fixed per-run cost.
    key is an arm name, or an arm dict (test_router_confirm's E_mem)."""
    a = key if isinstance(key, dict) else ARM[key]
    t0 = time.time()
    make_fn(a, 0)()
    fixed = time.time() - t0 if a.get("nudge") else 0.0
    per = time_per_step(TASK, make_fn(a, 0), steps=TIME_STEPS)
    m, data = make_fn(a, 0)(), eval_batch(TASK)
    t0 = time.time()
    evaluate(m, TASK, data)
    ev = time.time() - t0
    return max(per - (fixed + ev) / TIME_STEPS, 1e-6) + ev / EVAL_EVERY, fixed


# ── Verification ─────────────────────────────────────────────────────────────
def verify(pool):
    print("test_router_discovery.py's verification (which runs test_channel_binding's, and so on")
    print("down to test_multilayer_binding's):")
    trd.verify(pool)
    print("=" * 100)
    print("VERIFICATION — this test's own CHECKs")
    print("=" * 100)
    ok = True
    t = TASK
    data = eval_batch(t)

    print(f"CHECK 26 the new onset_run knobs (grad_hook, post_step) are inert at their defaults:")
    print(f"         60-step runs (evaluations every 20) vs {LEGACY_SHA}'s modules:")
    leg_ml, leg_on = load_legacy_pair(LEGACY_SHA)
    print(f"         legacy onset_run takes grad_hook: "
          f"{'grad_hook' in leg_on.onset_run.__code__.co_varnames}   legacy MultiBDH is a distinct "
          f"class: {leg_ml.MultiBDH is not trd.tmb.MultiBDH}")
    for key in ("A", "B"):
        a = ARM[key]
        m0, c0, _ = leg_on.onset_run(t, lambda: leg_ml.build(t, a, ARCH, 3), 3, max_iters=60,
                                     eval_every=20, data=data, early_stop=False, lr=SUB_LR)
        m1, c1, _ = onset_run(t, make_fn(a, 3), 3, max_iters=60, eval_every=20, data=data,
                              early_stop=False, lr=SUB_LR)
        good = same_weights(m0, m1) and c0 == c1
        ok &= good
        print(f"         {key:<2} weights equal {same_weights(m0, m1)}   curves equal {c0 == c1}   "
              f"accs {[round(x, 4) for _, x, _ in c1]}  -> {'IDENTICAL' if good else 'DIFFERS'}")
    print()

    s, Ts = 2, 60
    print(f"CHECK 27 through step T the curriculum IS the uniform gate: A_late with T={Ts + 1}, run "
          f"{Ts} steps, vs")
    print(f"         the floor arm (gate='uniform', k=2), seed {s}, evaluations every 20:")
    a = ARM["A_late"]
    init = make_fn(a, s)()
    probe = probe_batch(t, s)
    gh, ps = curriculum(a, s, Ts + 1, probe, data, {})
    ml, cl, _ = onset_run(t, make_fn(a, s), s, max_iters=Ts, eval_every=20, data=data,
                          early_stop=False, lr=SUB_LR, grad_hook=gh, post_step=ps)
    mf, cf, _ = onset_run(t, lambda: trd.build(t, FLOOR, ARCH, s), s, max_iters=Ts,
                          eval_every=20, data=data, early_stop=False, lr=SUB_LR)
    with torch.no_grad():
        g_train = ml(probe[0], TAU_END)[2]
        ml.eval()
        g_eval = ml(probe[0], TAU_END)[2]
        ml.train()
    uniform = bool((g_train == 0.5).all()) and bool((g_eval == 0.5).all())
    pf, pl = dict(mf.named_parameters()), dict(ml.named_parameters())
    same_rest = sorted(pf) == sorted(n for n in pl if n not in GATE_PARAMS) and all(
        torch.equal(pf[n], pl[n]) for n in pf)
    pi = dict(init.named_parameters())
    frozen = all(torch.equal(pi[n], pl[n]) for n in GATE_PARAMS) and not ml.W_g.any()
    good = uniform and same_rest and frozen and cl == cf
    ok &= good
    print(f"         gate output exactly 0.5 at every position (train and eval): {uniform}   "
          f"non-gate weights bitwise equal to floor's: {same_rest}")
    print(f"         W_in, W_h, W_g bitwise unchanged (W_g still 0): {frozen}   curves equal to "
          f"floor's: {cl == cf}  -> {'UNIFORM GATE' if good else 'WRONG'}")
    print()

    print(f"CHECK 28 the re-init at T: A_late with T={Ts}, run {Ts + 1} steps, seed {s}; the "
          f"optimizer is observed by a step pre-hook:")
    holder, seen, out = [], {}, {}
    mk = make_fn(a, s)

    def mk2():
        holder.append(mk())
        return holder[-1]

    def pre(opt, args, kw):
        seen["n"] = seen.get("n", 0) + 1
        if seen["n"] == Ts + 1:
            m = holder[0]
            seen["no_state"] = all(len(opt.state.get(getattr(m, n), {})) == 0
                                   for n in GATE_PARAMS)
            seen["snap"] = {n: getattr(m, n).detach().clone() for n in GATE_PARAMS}
            seen["state_rest"] = all(len(opt.state.get(p, {})) > 0 for nn, p in
                                     m.named_parameters() if nn not in GATE_PARAMS)

    gh, ps = curriculum(a, s, Ts, probe, data, out)
    h = register_optimizer_step_pre_hook(pre)
    try:
        onset_run(t, mk2, s, max_iters=Ts + 1, lr=SUB_LR, grad_hook=gh, post_step=ps)
    finally:
        h.remove()
    m = holder[0]
    want = reinit_w_g(m.W_g.shape, s)
    exact = torch.equal(seen["snap"]["W_g"], want)
    changed = all(not torch.equal(seen["snap"][n], getattr(m, n)) for n in GATE_PARAMS)
    good = exact and seen["no_state"] and seen["state_rest"] and changed and "stats_reinit" in out
    ok &= good
    print(f"         W_g at T equals randn(seed + {REINIT_OFFSET}) * {REINIT_SCALE} exactly: {exact}   "
          f"Adam state for the gate before T+1: none = {seen['no_state']} (other parameters have "
          f"state: {seen['state_rest']})")
    print(f"         W_in, W_h, W_g all change at step T+1: {changed}   accuracy at T {out['acc_T']:.4f}"
          f"   sep before/after re-init {out['stats_T']['sep']:.4f}/{out['stats_reinit']['sep']:.4f}"
          f"   spread {out['stats_T']['spread']:.4f}/{out['stats_reinit']['spread']:.4f}  -> "
          f"{'OK' if good else 'WRONG'}")
    print()

    print(f"CHECK 29 the re-init leaves the global torch RNG alone, and the batches after T are the "
          f"floor arm's:")
    fed, rng_same = {}, {}
    for key, arm_ in (("A_late", a), ("floor", FLOOR)):
        rec = []
        mkx = make_fn(arm_, s) if key == "A_late" else (lambda: trd.build(t, FLOOR, ARCH, s))

        def mk3(mkx=mkx, rec=rec):
            mm = mkx()
            mm.register_forward_pre_hook(
                lambda mod, args: rec.append(args[0].clone()) if mod.training else None)
            return mm
        gh, ps = curriculum(arm_, s, Ts, probe, data, {})
        ps2 = None
        if ps is not None:
            def ps2(mm, step, ps=ps):
                before = torch.get_rng_state()
                ps(mm, step)
                if step == Ts:
                    rng_same["v"] = torch.equal(before, torch.get_rng_state())
        onset_run(t, mk3, s, max_iters=Ts + 3, lr=SUB_LR, grad_hook=gh, post_step=ps2)
        fed[key] = rec[Ts:]
    same_b = len(fed["A_late"]) == 3 and all(torch.equal(u, v) for u, v in
                                              zip(fed["A_late"], fed["floor"]))
    good = rng_same.get("v", False) and same_b
    ok &= good
    print(f"         global RNG state identical across the re-init: {rng_same.get('v')}   training "
          f"batches T+1..T+3 equal the floor arm's: {same_b}  -> {'OK' if good else 'WRONG'}")
    print()

    print("CHECK 30 A_nudge is test_router_discovery's nudge, on arm A:")
    mz, ma = make_fn(ARM["A_nudge"], 8)(), make_fn(ARM["A"], 8)()
    sep0 = gate_stats(mz, t, probe_batch(t, 8), 0)["sep"]
    pz, pa = dict(mz.named_parameters()), dict(ma.named_parameters())
    same_rest = all(torch.equal(pz[n], pa[n]) for n in pz if n not in GATE_PARAMS)
    no_ro = "W_ro" not in pz
    good = abs(sep0 - 0.05) <= 0.015 and same_rest and no_ro
    ok &= good
    print(f"         step-0 sep {sep0:.4f} (0.05 +- 0.015): {abs(sep0 - 0.05) <= 0.015}   non-gate "
          f"parameters equal A's at init: {same_rest}   no readout (arm A): {no_ro}  -> "
          f"{'OK' if good else 'WRONG'}")
    print()

    print("CHECK 31 a worker's run is bit-identical to the same run here:")
    sp = dict(arm="A_late_ln", seed=1, iters=1200, T=600)
    rw = pool.submit(run_spec, sp).result()
    rh = run_spec(sp)
    good = strip(rw) == strip(rh) and rh["ok"] and "stats_reinit" in rh
    ok &= good
    print(f"         A_late_ln seed 1, T=600, 1200 steps: equal {strip(rw) == strip(rh)}   accuracy "
          f"at T {rh.get('acc_T', float('nan')):.4f}   VAL cos at end {rh.get('val_cos', float('nan')):.4f}"
          f"  -> {'IDENTICAL' if good else 'DIFFERS'}")
    print()
    print(f"{'ALL VERIFICATION CHECKS PASSED' if ok else 'SOME VERIFICATION CHECKS FAILED'}\n")
    assert ok, "verification failed — do not trust the results below"


# ── Reporting ────────────────────────────────────────────────────────────────
def escape(r):
    ev = [x for x in r["stats"] if isinstance(x["step"], int) and x["step"] > 0]
    return next((x["step"] for x in ev if x["val_cos"] < DISC_COS), None)


def report(store, seeds, wall, path, also):
    T = T_SWITCH
    n = len(seeds)
    print("#" * 100)
    print("PER-SEED RAW RESULTS — every value, before any aggregate")
    print("#" * 100)
    print(f"  {'arm':<10} {'seed':>4} {'acc':>7} {'transition':>10} {'VAL cos':>8} {'sep':>7} "
          f"{'key_part':>8} {'spread':>7} {'acc@T':>7}  flags")
    for a in ARMS:
        for s in seeds:
            r = get(store, a["key"], s)
            if r is None:
                print(f"  {a['key']:<10} {s:>4}  NOT RUN")
                continue
            if not r.get("ok"):
                print(f"  {a['key']:<10} {s:>4}  FAILED — {r['error']}")
                continue
            e = r.get("end")
            fl = (["DISCOVERED"] if r.get("discovered") else []) + (
                ["B BOUND"] if a["key"] == "B" and r["transition"] is not None else []) + (
                ["collapsed"] if r["collapsed"] else [])
            at = f"{r['acc_T']:.4f}" if r.get("acc_T") is not None else "--"
            gs = ("      --      --       --      --" if e is None else
                  f"{e['val_cos']:>8.4f} {e['sep']:>7.4f} {e['key_part']:>8.4f} {e['spread']:>7.4f}")
            print(f"  {a['key']:<10} {s:>4} {r['acc']:>7.4f} {fmt_step(r['transition']):>10} {gs} "
                  f"{at:>7}  {' '.join(fl)}")
        print()

    print("=" * 100)
    print(f"DISCOVERY COUNTS (discovered = bound AND final VAL cos < {DISC_COS}), of {n} seeds")
    print("=" * 100)
    disc = {}
    for a in ARMS:
        k = a["key"]
        rs = [get(store, k, s) for s in seeds]
        failed = sum(1 for r in rs if r is not None and not r.get("ok"))
        done = sum(1 for r in rs if r is not None and r.get("ok"))
        disc[k] = count(store, k, seeds, "discovered") if k in CHANNEL_KEYS else None
        role = ("candidate" if k in CANDIDATES else "positive control" if k == "A_nudge"
                else "baseline" if k == "A" else "control" if k == "A_ln" else "reference")
        dtxt = "--" if disc[k] is None else f"{disc[k]}/{n}"
        print(f"  {a['label']:<50} {role:<17} discovered {dtxt:>6}   bound "
              f"{count(store, k, seeds, 'transition')}/{n}   completed {done}/{n}"
              + (f"   [{failed} FAILED]" if failed else ""))
    nb = count(store, "B", seeds, "transition")
    if nb:
        print("  " + "!" * 96)
        print(f"  !!! B BOUND ON {nb}/{n} SEEDS: a single channel escaped the 0.5 plateau !!!")
        print("  " + "!" * 96)
    print()

    print("#" * 100)
    print("PRE-REGISTERED VERDICT")
    print("#" * 100)
    print(f"  INVALID       A_nudge discovers on fewer than {FOUND_K}/{n} seeds.")
    print(f"  ROUTER FOUND  A_late or A_late_ln discovers on >= {FOUND_K}/{n}.")
    print(f"  PARTIAL       the best candidate discovers on {PARTIAL_K}-{FOUND_K - 1}/{n}.")
    print(f"  NOT FOUND     both candidates discover on <= {PARTIAL_K - 1}/{n}.")
    print()
    print(f"  A_nudge (positive control) {disc['A_nudge']}/{n};  candidates: "
          + ", ".join(f"{k} {disc[k]}/{n}" for k in CANDIDATES)
          + f";  never candidates: A {disc['A']}/{n}, A_ln {disc['A_ln']}/{n}, B bound {nb}/{n}")
    best = max(disc[k] for k in CANDIDATES)
    if disc["A_nudge"] < FOUND_K:
        verdict, why = "INVALID", (f"the positive control discovered on {disc['A_nudge']}/{n} < "
                                   f"{FOUND_K}; no candidate is interpreted.")
    elif best >= FOUND_K:
        verdict = "ROUTER FOUND"
        why = ", ".join(f"{k} {disc[k]}/{n}" for k in CANDIDATES if disc[k] >= FOUND_K)
    elif best >= PARTIAL_K:
        verdict = "PARTIAL"
        why = "best candidate " + ", ".join(f"{k} {disc[k]}/{n}" for k in CANDIDATES
                                             if disc[k] == best)
    else:
        verdict, why = "NOT FOUND", f"both candidates discovered on <= {PARTIAL_K - 1}/{n}."
    print(f"  *** {verdict}: {why} ***")
    store["verdict"] = dict(verdict=verdict, counts=disc)
    save_results(path, store)
    print()

    print("=" * 100)
    print("DIAGNOSTICS — NOT part of the verdict")
    print("=" * 100)
    print(f"  escape step (first evaluation with VAL cos < {DISC_COS}); curriculum arms also as "
          f"steps after T={T}:")
    anyd = False
    for k in CHANNEL_KEYS:
        for s in seeds:
            r = get(store, k, s)
            if not (r and r.get("ok")):
                continue
            esc = escape(r)
            if esc is None and not r.get("discovered"):
                continue
            anyd = True
            rel = (f"   (T{'+' if esc - T >= 0 else '-'}{abs(esc - T)})"
                   if ARM[k].get("late") and esc is not None else "")
            print(f"    {k:<10} seed {s}: escape {fmt_step(esc):>6}{rel:<12} transition "
                  f"{fmt_step(r['transition']):>6}   final VAL cos {r['val_cos']:.4f}"
                  f"{'   DISCOVERED' if r.get('discovered') else ''}")
    if not anyd:
        print("    none: no run's VAL cos went below the threshold.")
    print()
    print(f"  accuracy at T={T} (held-out, just before the re-init) and gate stats before -> after "
          f"the re-init, per curriculum run:")
    for k in CANDIDATES:
        for s in seeds:
            r = get(store, k, s)
            if not (r and r.get("ok") and r.get("acc_T") is not None):
                continue
            b, af = r["stats_T"], r["stats_reinit"]
            print(f"    {k:<10} seed {s}: acc@T {r['acc_T']:.4f}   sep {b['sep']:.4f} -> "
                  f"{af['sep']:.4f}   spread {b['spread']:.4f} -> {af['spread']:.4f}   key_part "
                  f"{b['key_part']:.4f} -> {af['key_part']:.4f}")
    accs = [get(store, k, s)["acc_T"] for k in CANDIDATES for s in seeds
            if get(store, k, s) and get(store, k, s).get("acc_T") is not None]
    if accs:
        print(f"    accuracy at T over all curriculum runs: min {min(accs):.4f}  max {max(accs):.4f}")
    print()
    print("  failure modes of non-discovering channel runs, from final gate stats (saddle: spread "
          "< 0.1; locked on keys: key_part > 0.8 and sep < 0.2; other):")
    for k in CHANNEL_KEYS:
        modes = {"saddle": 0, "locked on keys": 0, "other": 0}
        for s in seeds:
            r = get(store, k, s)
            if r and r.get("ok") and not r.get("discovered"):
                modes[mode_of(r)] += 1
        print(f"    {k:<10} " + "   ".join(f"{m}: {c}" for m, c in modes.items()))
    print()
    print(f"  LayerNorm alone vs with the curriculum: A_ln {disc['A_ln']}/{n}  vs  A_late_ln "
          f"{disc['A_late_ln']}/{n}   (A {disc['A']}/{n}, A_late {disc['A_late']}/{n})")
    print(f"  collapsed runs (final accuracy < {COLLAPSE}): "
          + ", ".join(f"{a['key']} {count(store, a['key'], seeds, 'collapsed')}" for a in ARMS))
    print()
    if also:
        print(f"  POOLED across machines (this run + {also}):")
        try:
            with open(also) as f:
                other = json.load(f)
            ocpu = other.get("meta", {}).get("cpu", "unknown")
            print(f"    other machine: {ocpu}, git {other.get('meta', {}).get('git')}")
            for a in ARMS:
                k = a["key"]
                field = "discovered" if k in CHANNEL_KEYS else "transition"
                mine = count(store, k, seeds, field)
                theirs = sum(1 for key, r in other.get("runs", {}).items()
                             if key.split("|")[0] == k and r.get("ok") and r.get(field))
                ntheirs = sum(1 for key in other.get("runs", {}) if key.split("|")[0] == k)
                what = "discovered" if k in CHANNEL_KEYS else "bound"
                print(f"    {k:<10} {what:<10} here {mine}/{n}   other {theirs}/{ntheirs}   pooled "
                      f"{mine + theirs}/{n + ntheirs}")
        except Exception as e:
            print(f"    could not read {also}: {e}")
        print()

    print("=" * 100)
    print(f"CURVES — channel runs: accuracy x100 and VAL cos x100 at each evaluation every "
          f"{EVAL_EVERY} steps ('^' marks the evaluation at T)")
    print("=" * 100)
    for k in CHANNEL_KEYS:
        for s in seeds:
            r = get(store, k, s)
            if not (r and r.get("ok")):
                continue
            ev = {x["step"]: x for x in r["stats"] if isinstance(x["step"], int) and x["step"] > 0}
            mark = lambda st: "^" if ARM[k].get("late") and st == T else " "
            accs = "".join(f"{mark(st)}{round(acc * 100):>3}" for st, acc, _ in r["curve"])
            coss = "".join(f"{mark(st)}{round(ev[st]['val_cos'] * 100):>3}" if st in ev else "  --"
                           for st, _, _ in r["curve"])
            print(f"  {k:<10} s{s} acc {accs}")
            print(f"  {'':<10}    cos {coss}   -> "
                  f"{'bound @' + str(r['transition']) if r['transition'] else 'not bound'}"
                  f"{'  DISCOVERED' if r.get('discovered') else ''}")
    print()
    print(f"  total wall clock after verification: {wall / 60:.1f} min ({wall:.0f}s)   raw records: "
          f"{path}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Does releasing a held gate late find the routing?")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--results", default=RESULTS_FILE)
    ap.add_argument("--also", default=None,
                    help="the other machine's router_curriculum_results.json, for pooled counts")
    args = ap.parse_args()
    torch.set_num_threads(1)
    head = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    seeds = list(SEEDS)

    print("=" * 100)
    print("Router curriculum: hold arm A's gate at uniform until the plateau, then release it")
    print(f"  test_router_discovery's config (S=2 P=4 n_vals=16 n_q=1, n_layer=3 decay N=256, lr "
          f"{SUB_LR}); T={T_SWITCH}, MAX_ITERS={MAX_ITERS}, seeds {seeds}")
    print(f"  fixed before any result: T={T_SWITCH}, re-init W_g = randn(seed + {REINIT_OFFSET}) * "
          f"{REINIT_SCALE}")
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
                             git=head, T=T_SWITCH, max_iters=MAX_ITERS, seeds=seeds,
                             started=time.strftime("%Y-%m-%d %H:%M:%S"),
                             note="seed-level outcomes differ between machines")
        save_results(args.results, store)
        t0 = time.time()

        print("=" * 100)
        print(f"PROJECTION — time_per_step ({TIME_STEPS} steps) inside the pool, evaluation charged "
              f"once per {EVAL_EVERY}; worst case = every run to {MAX_ITERS}")
        print("=" * 100)
        keys = [a["key"] for a in ARMS]
        costs = dict(zip(keys, pool.map(time_arm, keys)))
        for k in keys:
            per, fixed = costs[k]
            print(f"  {k:<10} {per * 1000:6.1f} ms/step" + (f"   + {fixed:.1f}s pre-fit per run"
                                                           if fixed else ""))
        durs = [MAX_ITERS * costs[k][0] + costs[k][1] for k in keys for _ in seeds]
        print(f"  {len(durs)} runs: serial {sum(durs) / 3600:.2f} h, projected wall clock on "
              f"{args.workers} workers {makespan(durs, args.workers) / 3600:.2f} h (no drop rule)")
        store["meta"]["projected_wall_h"] = makespan(durs, args.workers) / 3600
        print()

        print("=" * 100)
        print(f"RUNS — {len(ARMS)} arms x seeds {seeds}, T={T_SWITCH}, MAX_ITERS={MAX_ITERS}")
        print("=" * 100)
        todo = []
        for k in keys:
            for s in seeds:
                if get(store, k, s) is not None and not args.force:
                    print(f"   {k:<10} seed {s}  cached")
                else:
                    todo.append(dict(arm=k, seed=s, iters=MAX_ITERS, T=T_SWITCH))
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
                e = rec.get("end")
                gs = ("" if e is None else f"  VALcos={e['val_cos']:.3f} sep={e['sep']:.3f} "
                      f"key_part={e['key_part']:.3f}")
                at = "" if rec.get("acc_T") is None else f"  acc@T={rec['acc_T']:.3f}"
                tr = rec["transition"]
                print(f"   {sp['arm']:<10} seed {sp['seed']}  "
                      f"{'BOUND at ' + str(tr) if tr else 'not bound':<16} acc={rec['acc']:.4f} "
                      f"@{rec['stopped_at']:<5}{gs}{at}{'  DISCOVERED' if rec.get('discovered') else ''}"
                      f"  {rec['secs']:.0f}s  (elapsed {el:.1f}m)")
            else:
                print(f"   {sp['arm']:<10} seed {sp['seed']}  *** FAILED: {rec['error']}   "
                      f"(elapsed {el:.1f}m)")
        print()

    report(store, seeds, time.time() - t0, args.results, args.also)


if __name__ == "__main__":
    main()
