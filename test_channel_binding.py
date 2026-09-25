#!/usr/bin/env python
"""
test_channel_binding.py — once the memory binds, do multiple channels beat single-channel
alternatives under interference between streams?

Run it directly:

    python test_channel_binding.py                  # CHECKs, Gate 0, Gate 1, arms, report
    python test_channel_binding.py --workers 2 --force

WHY. This is the test the project set out to run. The multi-channel question was blocked
because single-channel BDH could not bind a key to its value reliably.
test_binding_recipe.py fixed that with ordinary training settings: lr 1e-3 (the inherited
4e-3 was too high) and N >= 256. The substrate here, decay / lr=1e-3 / N=256, is the only
cell that bound every seed at both P=4 and P=8 on the machine that ran the full P=8
check. Question: once the memory binds, do multiple channels beat single-channel
alternatives under interference between streams?

SUBSTRATE, every arm: MultiBDH, n_layer=3, positional='decay', mult=8 (N=256), lr=1e-3,
warmup=0, n_head=1, dropout=0.

TASK. BindTask with S=2 (each key has a different value in each stream), n_vals=16, n_q=1
(no elimination possible). Configs: P in {4, 8, 16}.

TRAINING. test_binding_onset.onset_run for every arm and config: held-out evaluation every
1200 steps on 2048 queries, stop after 3 consecutive evaluations >= 0.95,
MAX_ITERS=38400. Accuracy of a run = its final held-out evaluation; its transition step
is test_binding_onset.transition. Gated arms (k=2) also get the per-role gate cosine
between streams (KEY, VAL, CTX) on the trained model, via
test_multilayer_binding.role_cosines on run_ml's probe batch — exactly the numbers
run_ml's trace records.

GATE 0, this machine: the substrate must bind a single stream (S=1, single channel,
3 seeds, same budget and stopping rule) at every P scored. All 3 seeds must bind at a
given P, or every S=2 config at that P is excluded ("substrate does not bind S=1 at this P
on this machine"). If P=4 fails: UNTESTED, recipe does not reproduce here, STOP. The P=16
check runs before any P=16 arm.
GATE 1, per config, before its other arms: the ceiling (perfect gate, k=2) must bind on
every seed, and the recurrent gate must fit the perfect routing
(test_multilayer_binding.gate_fit, argmax match >= 0.99). A failing config is excluded
and its other arms are skipped.

ARMS, per valid config, 5 seeds — test_multilayer_binding.arms_for at mult=8:
  floor    uniform gate, k=2          ceiling  perfect gate, k=2
  A        recurrent gate, k=2        A_ro     recurrent gate, k=2, + readout
  B        plain single channel       <- the key baseline
  E        1 channel + recurrent readout (h_gate 32)
  E_mem    1 channel, mult=16 (A's memory), + readout
  E_h      1 channel + readout, h_gate in {64, 128}
In a multilayer model floor and B CAN legitimately rise above 1/S: attention may resolve
the interference without channels (e.g. by attending to the key that follows CTX_q).
That is a result, not contamination; it is reported, not flagged.

PRE-REGISTERED VERDICT, EPS = 0.10, printed before any interpretation; exactly one of
  UNTESTED       Gate 0 fails at P=4, or every config fails Gate 0 or Gate 1.
  SUPPORTED      at some valid config, max(A, A_ro) beats max(B, E, E_mem, E_h*) by
                 >= EPS in mean accuracy, beats that control on every seed, AND the winning
                 channel arm separates (VAL gate cosine < 0.5 on every seed).
  NOT SUPPORTED  otherwise.
The verdict is test_multilayer_binding.report's, unchanged: "max" is the arm with the
highest mean, and "every seed" compares those two arms seed by seed.

DIAGNOSTICS, not part of the verdict: accuracy and transition step vs P for every arm;
whether B degrades with P; whether widening E's readout (E_h) moves its failure point.

LOGISTICS. Runs execute in a pool of single-threaded worker processes, as in
test_binding_recipe.py. Wall clock is projected before any training (worst case, every
run to MAX_ITERS, stage by stage); over 8 hours, P=16 is dropped first, then E_h128, and
that is said. Results persist atomically to channel_binding_results.json (gitignored).
Seed-level outcomes differ between machines.

(Results are recorded at the bottom of this docstring after the run.)
"""

import argparse
import contextlib
import io
import multiprocessing as mp
import os
import subprocess
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch

import test_multilayer_binding as tmb
from test_multilayer_binding import (
    MultiBDH, BindTask, arms_for, build, gate_fit, probe_batch, role_cosines, rkey, get_run,
    load_results, report, n_params, n_state, stats, cell, save_results,
    EPS, SEP_COS, FIT_PASS, E_H_WIDTHS, CHANNEL_ARMS, CONTROL_ARMS,
)
import test_binding_onset as tbo
from test_binding_onset import (
    onset_run, make_bdh, eval_batch, evaluate, transition, attempt, time_per_step,
    curve_row, fmt_step, BIND_PASS, STOP_AFTER, EVAL_EVERY,
)
from test_binding_recipe import makespan, cpu_model, same_weights

# ── Settings ─────────────────────────────────────────────────────────────────
ARCH = dict(n_layer=3, positional="decay")
SUB_MULT = 8                      # N = 256
SUB_LR = 1e-3
S_STREAMS, N_VALS, N_Q = 2, 16, 1
P_LIST = (4, 8, 16)
SEEDS = tuple(range(5))
G0_SEEDS = tuple(range(3))
MAX_ITERS = 38400
WALL_LIMIT_H = 8.0
WORKERS = min(4, os.cpu_count() or 1)
TIME_STEPS = 200
LEGACY_SHA = "2282d72"            # test_multilayer_binding.py / test_binding_onset.py before
RESULTS_FILE = "channel_binding_results.json"
G0_REASON = "substrate does not bind S=1 at this P on this machine"


def task_for(P, S):
    t = BindTask(P, S=S, n_vals=N_VALS, n_q=N_Q)
    t.key = f"P{P}" if S == S_STREAMS else f"S{S}P{P}"
    return t


def arms(task, h_widths=E_H_WIDTHS):
    return arms_for(task, mult=SUB_MULT, h_widths=h_widths)


G0_ARM = dict(tmb.G0_ARM, mult=SUB_MULT)


def arm_by_key(task, key):
    return {a["key"]: a for a in arms(task)}[key] if key != "G0" else G0_ARM


# ── Worker-side functions (spawned processes, one thread each) ───────────────
def init_worker():
    torch.set_num_threads(1)


def run_spec(sp):
    """One arm, one seed: onset_run's recipe on this substrate, plus gate cosines."""
    task = task_for(sp["P"], sp["S"])
    a = sp["arm"]
    probe = probe_batch(task, sp["seed"]) if a["n_ch"] > 1 else None

    def post(model):
        if probe is None:
            return {}
        d = role_cosines(model, probe, "end")
        return dict(key_cos=d["key_cos"], val_cos=d["val_cos"], ctx_cos=d["ctx_cos"])

    rec = attempt(task, lambda: build(task, a, ARCH, sp["seed"]), sp["seed"], eval_batch(task),
                  post=post, max_iters=sp["iters"], lr=SUB_LR)
    if rec["ok"]:
        rec["acc"] = rec["curve"][-1][1] if rec["curve"] else float("nan")
    rec["iters"] = sp["iters"]
    return rec


def fit_spec(P):
    return gate_fit(task_for(P, S_STREAMS), ARCH)


def time_spec(sp):
    """time_per_step charges one evaluation per TIME_STEPS steps; real runs evaluate once
    per EVAL_EVERY, so the evaluation is timed on its own and re-charged at that rate."""
    task = task_for(sp["P"], sp["S"])
    a = sp["arm"]
    per = time_per_step(task, lambda: build(task, a, ARCH, 0), steps=TIME_STEPS)
    m, data = build(task, a, ARCH, 0), eval_batch(task)
    t0 = time.time()
    evaluate(m, task, data)
    ev = time.time() - t0
    return max(per - ev / TIME_STEPS, 1e-6) + ev / EVAL_EVERY


def fit_time_spec(P):
    t0 = time.time()
    fit_spec(P)
    return time.time() - t0


# ── Verification ─────────────────────────────────────────────────────────────
def load_legacy(fname, modname):
    src = subprocess.run(["git", "-C", HERE, "show", f"{LEGACY_SHA}:{fname}"],
                         capture_output=True, text=True, check=True).stdout
    mod = types.ModuleType(modname)
    mod.__file__ = os.path.join(HERE, fname)
    exec(compile(src, f"{fname}@{LEGACY_SHA}", "exec"), mod.__dict__)
    return mod


def synthetic_store(tasks, seeds, arch, it):
    """Deterministic fake records covering every arm, for comparing report() outputs."""
    g = torch.Generator().manual_seed(5)
    store = {"meta": {}, "runs": {}, "gate0": {}, "gate1": {}}
    for t in tasks:
        for a in arms_for(t):
            rk = rkey(t.key, a["key"], arch, it)
            store["runs"][rk] = {}
            for s in seeds:
                acc = float(torch.rand(1, generator=g))
                rec = dict(ok=True, acc=acc, val_cos=float(torch.rand(1, generator=g)))
                if (a["key"], s) == ("E", 1):
                    rec = dict(ok=False, error="ValueError: synthetic")
                store["runs"][rk][str(s)] = rec
    return store


def verify(pool):
    print("test_binding_onset.py's CHECKs (which include test_multilayer_binding's 1-6):")
    tbo.verify()
    print("=" * 100)
    print("VERIFICATION — this test's own CHECKs")
    print("=" * 100)
    ok = True
    leg_ml = load_legacy("test_multilayer_binding.py", "multilayer_binding_legacy")
    leg_on = load_legacy("test_binding_onset.py", "binding_onset_legacy")

    t2 = task_for(4, S_STREAMS)
    print(f"CHECK 15 the knobs added for this test are inert at their defaults, vs {LEGACY_SHA}:")
    same_arms = arms_for(t2) == leg_ml.arms_for(t2)
    print(f"         arms_for(task) == legacy arms_for(task): {same_arms}")
    ok &= same_arms
    ra = tmb.run_ml(t2, dict(n_layer=2, positional="decay"), "recurrent", 2, 1, 60,
                    ckpts=[0, 30])
    rb = leg_ml.run_ml(t2, dict(n_layer=2, positional="decay"), "recurrent", 2, 1, 60,
                       ckpts=[0, 30])
    same_ml = ra == rb
    print(f"         run_ml (recurrent k=2, 60 steps, per-role cosine trace at 0/30/60) == "
          f"legacy: {same_ml}   VAL cos at end {ra['trace'][-1]['val_cos']:.6f}")
    ok &= same_ml
    tasks_syn = [task_for(4, S_STREAMS), task_for(8, S_STREAMS)]
    store = synthetic_store(tasks_syn, SEEDS, ARCH, 1200)
    g1 = {"P4": dict(valid=True, reason=""), "P8": dict(valid=False, reason="synthetic")}
    info = {t.key: {a["key"]: (1, 2) for a in arms_for(t)} for t in tasks_syn}
    outs = []
    for fn in (tmb.report, leg_ml.report):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn(store, tasks_syn, SEEDS, ARCH, 1200, g1, info, 1.0, "x.json")
        outs.append(buf.getvalue())
    same_rep = outs[0] == outs[1]
    print(f"         report() on a synthetic store, full printed text == legacy: {same_rep} "
          f"({len(outs[0].splitlines())} lines)")
    ok &= same_rep
    data = eval_batch(tbo.PRIMARY)
    r_now = attempt(tbo.PRIMARY, make_bdh(tbo.PRIMARY, 3, "decay"), 0, data, max_iters=1200)
    r_leg = leg_on.attempt(tbo.PRIMARY, leg_on.make_bdh(tbo.PRIMARY, 3, "decay"), 0, data,
                           max_iters=1200)
    strip = lambda r: {k: v for k, v in r.items() if k != "secs"}
    same_att = strip(r_now) == strip(r_leg)
    print(f"         attempt() record (all fields but wall time) == legacy: {same_att}")
    ok &= same_att
    print()

    print("CHECK 16 the substrate: every arm is test_multilayer_binding.build at mult=8, and")
    print("         the single-channel arm trained through build is bit-identical to")
    print("         make_bdh(mult=8) — the model test_binding_recipe.py confirmed:")
    for a in arms(t2):
        m = build(t2, a, ARCH, 0)
        want_n = SUB_MULT * tmb.D * (S_STREAMS if a["key"] == "E_mem" else 1)
        good = (m.n_layers == 3 and m.positional == "decay" and m.n_feat == want_n
                and m.n_ch == a["n_ch"] and m.config.n_head == 1 and m.config.dropout == 0.0)
        ok &= good
        print(f"         {a['key']:>7}: n_layer {m.n_layers} {m.positional} N={m.n_feat:<4} "
              f"k={m.n_ch} gate={a['gate']:<9} readout={a['g2r']!s:<5} h_gate={m.h_gate:<3} "
              f"params {n_params(m):>6} state {n_state(m):>6}  -> {'OK' if good else 'WRONG'}")
    b = arm_by_key(t2, "B")
    m1, c1, _ = onset_run(t2, lambda: build(t2, b, ARCH, 2), 2, max_iters=1200, eval_every=400,
                          data=eval_batch(t2), early_stop=False, lr=SUB_LR)
    m2, c2, _ = onset_run(t2, make_bdh(t2, 3, "decay", mult=SUB_MULT), 2, max_iters=1200,
                          eval_every=400, data=eval_batch(t2), early_stop=False, lr=SUB_LR)
    good = same_weights(m1, m2) and c1 == c2
    ok &= good
    print(f"         B via build vs make_bdh, S=2 P=4, seed 2, 1200 steps: weights equal "
          f"{same_weights(m1, m2)}, curves equal {c1 == c2}  -> {'IDENTICAL' if good else 'DIFFERS'}")
    ea = [a for a in arms(t2) if a["key"] == "E_mem"][0]
    ma, mb = build(t2, arm_by_key(t2, "A"), ARCH, 0), build(t2, ea, ARCH, 0)
    good = n_state(ma) == n_state(mb)
    ok &= good
    print(f"         E_mem memory == A memory: {n_state(mb)} vs {n_state(ma)} floats  -> "
          f"{'MATCHED' if good else 'MISMATCH'}")
    print()

    print("CHECK 17 the task: S=2 streams bind DIFFERENT values to the same key, and the")
    print("         one query per sequence is answered by its own stream's value:")
    for P in P_LIST:
        t = task_for(P, S_STREAMS)
        tok, qs = t.make_batch(256, torch.Generator().manual_seed(17))
        body = tok[:, :3 * S_STREAMS * P].reshape(256, S_STREAMS * P, 3)
        q = tok[:, 3 * S_STREAMS * P:].reshape(256, 3)
        bound, conflict = True, True
        for bi in range(256):
            mp_ = {}
            for c, k, v in body[bi].tolist():
                mp_[(c, k)] = v
            bound &= mp_[(q[bi, 0].item(), q[bi, 1].item())] == q[bi, 2].item()
            conflict &= all(mp_[(0, k)] != mp_[(1, k)] for k in range(S_STREAMS, S_STREAMS + P))
        good = bound and conflict and len(t.qpos) == 1 and t.S == 2
        ok &= good
        print(f"         P={P:<2} length {t.L:<3} queries scored per sequence {len(t.qpos)}   "
              f"streams conflict on every key: {conflict}   query answered by its stream: "
              f"{bound}  -> {'OK' if good else 'WRONG'}")
    print()

    print("CHECK 18 a worker's run (record and gate cosines) is bit-identical to the same run")
    print("         here, and its cosines equal role_cosines (run_ml's diagnostic, CHECK 15)")
    print("         applied to the same model trained directly with onset_run:")
    t = task_for(4, S_STREAMS)
    a = arm_by_key(t, "A")
    sp = dict(P=4, S=S_STREAMS, arm=dict(a), seed=1, iters=1200)
    rw = pool.submit(run_spec, sp).result()
    rh = run_spec(sp)
    m, _, _ = onset_run(t, lambda: build(t, a, ARCH, 1), 1, max_iters=1200, data=eval_batch(t),
                        lr=SUB_LR)
    d = role_cosines(m, probe_batch(t, 1), "end")
    good = (strip(rw) == strip(rh) and rh["val_cos"] == d["val_cos"]
            and rh["key_cos"] == d["key_cos"])
    ok &= good
    print(f"         A, P=4, seed 1, 1200 steps: worker == here {strip(rw) == strip(rh)}   "
          f"KEY cos {rh['key_cos']:.6f}  VAL cos {rh['val_cos']:.6f} == role_cosines on the "
          f"trained model: {rh['val_cos'] == d['val_cos'] and rh['key_cos'] == d['key_cos']}"
          f"  -> {'IDENTICAL' if good else 'DIFFERS'}")
    print()
    print(f"{'ALL VERIFICATION CHECKS PASSED' if ok else 'SOME VERIFICATION CHECKS FAILED'}\n")
    assert ok, "verification failed — do not trust the results below"


# ── Running specs through the pool ───────────────────────────────────────────
def spec(task, a, seed, iters=None):
    iters = MAX_ITERS if iters is None else iters
    return dict(P=task.P, S=task.S, arm=dict(a), seed=seed, iters=iters, key=task.key)


def run_all(specs, store, path, pool, force, t0, cost):
    todo = []
    for sp in specs:
        r = get_run(store, sp["key"], sp["arm"]["key"], ARCH, sp["iters"], sp["seed"])
        if r is not None and not force:
            print(f"   {sp['key']:>5} {sp['arm']['key']:>7} seed {sp['seed']}  cached "
                  f"({'ok' if r.get('ok') else 'FAILED'})")
        else:
            todo.append(sp)
    todo.sort(key=lambda sp: -cost.get((sp["key"], sp["arm"]["key"]), 1.0))
    futs = {pool.submit(run_spec, sp): sp for sp in todo}
    for f in as_completed(futs):
        sp = futs[f]
        try:
            rec = f.result()
        except Exception as e:
            rec = dict(ok=False, seed=sp["seed"], error=f"worker {type(e).__name__}: {e}",
                       secs=0.0)
        rk = rkey(sp["key"], sp["arm"]["key"], ARCH, sp["iters"])
        store["runs"].setdefault(rk, {})[str(sp["seed"])] = rec
        save_results(path, store)
        el = (time.time() - t0) / 60
        if rec["ok"]:
            tr = rec["transition"]
            cs = ("" if rec.get("val_cos") is None else
                  f"  KEYcos={rec['key_cos']:.3f} VALcos={rec['val_cos']:.3f}")
            print(f"   {sp['key']:>5} {sp['arm']['key']:>7} seed {sp['seed']}  "
                  f"{'BOUND at ' + str(tr) if tr else 'not bound':<16} acc={rec['acc']:.4f} "
                  f"@{rec['stopped_at']:<5}{cs}  {rec['secs']:.0f}s  (elapsed {el:.1f}m)")
        else:
            print(f"   {sp['key']:>5} {sp['arm']['key']:>7} seed {sp['seed']}  *** FAILED: "
                  f"{rec['error']}   (elapsed {el:.1f}m)")


def bound_all(store, key, akey, seeds):
    rs = [get_run(store, key, akey, ARCH, MAX_ITERS, s) for s in seeds]
    return all(r is not None and r.get("ok") and r["transition"] is not None for r in rs), rs


def outcome(r):
    if r is None:
        return "--"
    if not r.get("ok"):
        return "FAIL"
    return f"@{r['transition']}" if r["transition"] else f"{r['acc']:.3f}"


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Do channels beat single-channel alternatives?")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--results", default=RESULTS_FILE)
    args = ap.parse_args()
    torch.set_num_threads(1)
    head = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()

    print("=" * 100)
    print("Channel binding: once the memory binds, do channels beat single-channel alternatives?")
    print(f"  substrate: MultiBDH n_layer={ARCH['n_layer']} {ARCH['positional']} N="
          f"{SUB_MULT * tmb.D} lr={SUB_LR} warmup=0   task S={S_STREAMS} n_vals={N_VALS} "
          f"n_q={N_Q}, P {list(P_LIST)}   EPS={EPS}")
    print(f"  onset_run recipe: eval every {EVAL_EVERY} on 2048 queries, stop after {STOP_AFTER}"
          f" evals >= {BIND_PASS}, MAX_ITERS={MAX_ITERS}")
    print(f"  torch {torch.__version__}   CPU {cpu_model()}   {args.workers} worker processes x "
          f"1 thread   git {head}")
    print("=" * 100)
    print()

    os.environ.setdefault("PYTHONWARNINGS", "ignore:Failed to initialize NumPy")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx,
                             initializer=init_worker) as pool:
        verify(pool)
        store = load_results(args.results)
        store["meta"] = dict(torch=torch.__version__, cpu=cpu_model(), workers=args.workers,
                             threads_per_worker=1, git=head, eps=EPS, max_iters=MAX_ITERS,
                             started=time.strftime("%Y-%m-%d %H:%M:%S"),
                             note="seed-level outcomes differ between machines")
        save_results(args.results, store)
        t0 = time.time()

        # ── Projection ──────────────────────────────────────────────────────
        print("=" * 100)
        print(f"PROJECTION — time_per_step ({TIME_STEPS} steps) measured inside the pool, "
              f"evaluation charged once per {EVAL_EVERY} steps; worst case = every run to "
              f"{MAX_ITERS}, stage by stage")
        print("=" * 100)
        tasks2 = {P: task_for(P, S_STREAMS) for P in P_LIST}
        tasks1 = {P: task_for(P, 1) for P in P_LIST}
        tspecs = [dict(P=P, S=1, arm=G0_ARM, key=tasks1[P].key, akey="G0") for P in P_LIST]
        tspecs += [dict(P=P, S=S_STREAMS, arm=a, key=tasks2[P].key, akey=a["key"])
                   for P in P_LIST for a in arms(tasks2[P])]
        costs = list(pool.map(time_spec, tspecs))
        fit_t = dict(zip(P_LIST, pool.map(fit_time_spec, P_LIST)))
        cost = {(s["key"], s["akey"]): c for s, c in zip(tspecs, costs)}
        cost.update({(tasks1[P].key, "B"): cost[(tasks1[P].key, "G0")] for P in P_LIST})
        for P in P_LIST:
            row = "  ".join(f"{a['key']}={cost[(tasks2[P].key, a['key'])] * 1000:.0f}"
                            for a in arms(tasks2[P]))
            print(f"  P={P:<2} ms/step  S=1 gate-0={cost[(tasks1[P].key, 'G0')] * 1000:.0f}   "
                  f"{row}   gate fit {fit_t[P]:.0f}s")

        def project(ps, hw):
            g0 = [MAX_ITERS * cost[(tasks1[P].key, "G0")] for P in ps for _ in G0_SEEDS]
            g1 = ([MAX_ITERS * cost[(tasks2[P].key, "ceiling")] for P in ps for _ in SEEDS]
                  + [fit_t[P] for P in ps])
            rest = [MAX_ITERS * cost[(tasks2[P].key, a["key"])] for P in ps
                    for a in arms(tasks2[P], hw) if a["key"] != "ceiling" for _ in SEEDS]
            stages = [makespan(x, args.workers) for x in (g0, g1, rest)]
            return sum(g0 + g1 + rest) / 3600, sum(stages) / 3600

        ps, hw = P_LIST, E_H_WIDTHS
        serial, wall = project(ps, hw)
        print(f"  full design: serial {serial:.1f} h, projected wall clock on {args.workers} "
              f"workers {wall:.1f} h (limit {WALL_LIMIT_H} h)")
        dropped = []
        if wall > WALL_LIMIT_H:
            ps = tuple(P for P in ps if P != 16)
            dropped.append("P=16")
            serial, wall = project(ps, hw)
            print(f"  -> OVER: P=16 DROPPED. Now serial {serial:.1f} h, wall {wall:.1f} h.")
        if wall > WALL_LIMIT_H:
            hw = tuple(h for h in hw if h != 128)
            dropped.append("E_h128")
            serial, wall = project(ps, hw)
            print(f"  -> STILL OVER: E_h128 DROPPED. Now serial {serial:.1f} h, wall "
                  f"{wall:.1f} h.")
        if wall > WALL_LIMIT_H:
            print("  -> still over the limit after both drops; running what remains anyway. "
                  "Early stopping makes the worst case unlikely.")
        elif not dropped:
            print("  -> within the limit: full design.")
        store["meta"].update(dropped=dropped, projected_wall_h=wall, projected_serial_h=serial)
        save_results(args.results, store)
        print()

        # ── Gate 0 ──────────────────────────────────────────────────────────
        print("=" * 100)
        print(f"GATE 0 — the substrate binds a single stream (S=1, single channel), seeds "
              f"{list(G0_SEEDS)}, P {list(ps)}")
        print("=" * 100)
        run_all([spec(tasks1[P], G0_ARM, s) for P in ps for s in G0_SEEDS], store,
                args.results, pool, args.force, t0, cost)
        g0 = {}
        for P in ps:
            okP, rs = bound_all(store, tasks1[P].key, "B", G0_SEEDS)
            g0[P] = okP
            print(f"  S=1 P={P:<2}: {' '.join(f'{outcome(r):>7}' for r in rs)}   -> "
                  f"{'BINDS 3/3' if okP else 'DOES NOT BIND on every seed'}")
        store["gate0"] = {str(P): v for P, v in g0.items()}
        save_results(args.results, store)
        print()
        if not g0[4]:
            print("#" * 100)
            print("PRE-REGISTERED VERDICT")
            print("#" * 100)
            print("  *** UNTESTED: recipe does not reproduce here (the substrate does not bind a "
                  "single stream at P=4 on this machine). STOP. ***")
            return

        # ── Gate 1 ──────────────────────────────────────────────────────────
        tasks = [tasks2[P] for P in ps]
        g1 = {}
        for t in tasks:
            if not g0[t.P]:
                g1[t.key] = dict(valid=False, reason=G0_REASON)
        live = [t for t in tasks if t.key not in g1]
        print("=" * 100)
        print(f"GATE 1 — ceiling (perfect gate, k=2) binds every seed, and the recurrent gate fits"
              f" the perfect routing (argmax >= {FIT_PASS}); configs {[t.key for t in live]}")
        print("=" * 100)
        fits = dict(zip([t.P for t in live], pool.map(fit_spec, [t.P for t in live])))
        run_all([spec(t, arm_by_key(t, "ceiling"), s) for t in live for s in SEEDS], store,
                args.results, pool, args.force, t0, cost)
        for t in live:
            cok, rs = bound_all(store, t.key, "ceiling", SEEDS)
            mse, am = fits[t.P]
            fok = am >= FIT_PASS
            valid = cok and fok
            reason = "" if valid else (
                ("ceiling does not bind every seed" if not cok else "")
                + ("; " if (not cok and not fok) else "")
                + (f"gate fit argmax {am:.4f} < {FIT_PASS}" if not fok else ""))
            g1[t.key] = dict(valid=valid, reason=reason, fit_mse=mse, fit_argmax=am)
            print(f"  {t.key:>4}: ceiling {' '.join(f'{outcome(r):>7}' for r in rs)}   gate fit "
                  f"mse {mse:.5f} argmax {am:.4f}   -> {'VALID' if valid else 'EXCLUDED — ' + reason}")
        for t in tasks:
            if not g0[t.P]:
                print(f"  {t.key:>4}: EXCLUDED — {G0_REASON}")
        store["gate1"] = g1
        save_results(args.results, store)
        print()

        # ── Arms ────────────────────────────────────────────────────────────
        valid = [t for t in tasks if g1[t.key]["valid"]]
        print("=" * 100)
        print(f"ARMS — valid configs {[t.key for t in valid]}, seeds {list(SEEDS)}, arms "
              f"{[a['key'] for a in arms(tasks[0], hw)]}")
        print("=" * 100)
        rest = [spec(t, a, s) for t in valid for a in arms(t, hw) if a["key"] != "ceiling"
                for s in SEEDS]
        if rest:
            durs = [MAX_ITERS * cost[(sp["key"], sp["arm"]["key"])] for sp in rest]
            print(f"   {len(rest)} runs; worst case wall {makespan(durs, args.workers) / 3600:.1f} h")
            run_all(rest, store, args.results, pool, args.force, t0, cost)
        else:
            print("   no valid config: no arms to run.")
        print()

    # ── Report ──────────────────────────────────────────────────────────────
    wall = time.time() - t0
    af = lambda t: arms(t, hw)
    info = {t.key: {a["key"]: (n_params(build(t, a, ARCH)), n_state(build(t, a, ARCH)))
                    for a in af(t)} for t in tasks}
    print("#" * 100)
    print("TRANSITION STEPS AND GATE COSINES, per seed (before the aggregate report)")
    print("#" * 100)
    for t in tasks:
        print(f"  {t.label}   {'VALID' if g1[t.key]['valid'] else 'EXCLUDED — ' + g1[t.key]['reason']}")
        print(f"    {'arm':>7}  per seed (@transition, or final acc if not bound)")
        for a in af(t):
            rs = [get_run(store, t.key, a["key"], ARCH, MAX_ITERS, s) for s in SEEDS]
            print(f"    {a['key']:>7}  " + " ".join(f"{outcome(r):>7}" for r in rs))
        for a in af(t):
            if a["n_ch"] < 2:
                continue
            rs = [get_run(store, t.key, a["key"], ARCH, MAX_ITERS, s) for s in SEEDS]
            if all(r is None for r in rs):
                continue
            for role in ("key", "val"):
                vals = [None if (r is None or not r.get("ok")) else r.get(f"{role}_cos")
                        for r in rs]
                print(f"    {a['key']:>7}  {role.upper()} cos: "
                      + " ".join("     --" if v is None else f"{v:7.4f}" for v in vals))
        print()
    report(store, tasks, list(SEEDS), ARCH, MAX_ITERS, g1, info, wall, args.results,
           arms_fn=af, flag_floor=False)

    print("=" * 100)
    print("DIAGNOSTICS (continued) — transition step vs P, NOT part of the verdict")
    print("=" * 100)
    valid = [t for t in tasks if g1[t.key]["valid"]]
    if valid:
        print(f"  seeds bound (k/5) and median transition step, per arm and valid config:")
        print(f"  {'arm':>8}" + "".join(f"{'P=' + str(t.P):>18}" for t in valid))
        for a in af(valid[0]):
            row = ""
            for t in valid:
                rs = [get_run(store, t.key, a["key"], ARCH, MAX_ITERS, s) for s in SEEDS]
                trs = sorted(r["transition"] for r in rs
                             if r is not None and r.get("ok") and r["transition"])
                if all(r is None for r in rs):
                    row += f"{'not run':>18}"
                    continue
                med = (None if not trs else (trs[len(trs) // 2] if len(trs) % 2 else
                                             (trs[len(trs) // 2 - 1] + trs[len(trs) // 2]) / 2))
                row += f"{str(len(trs)) + '/' + str(len(SEEDS)) + ' med ' + fmt_step(med):>18}"
            print(f"  {a['key']:>8}" + row)
    else:
        print("  none: no valid config.")
    print()
    print("=" * 100)
    print(f"CURVES — held-out accuracy x100 every {EVAL_EVERY} steps; '|' marks the transition")
    print("=" * 100)
    for P in ps:
        for s in G0_SEEDS:
            r = get_run(store, tasks1[P].key, "B", ARCH, MAX_ITERS, s)
            if r is not None and r.get("ok"):
                print(f"  {tasks1[P].key:>5} {'B':>7} s{s} " + curve_row(r))
    for t in tasks:
        for a in af(t):
            for s in SEEDS:
                r = get_run(store, t.key, a["key"], ARCH, MAX_ITERS, s)
                if r is None:
                    continue
                if not r.get("ok"):
                    print(f"  {t.key:>5} {a['key']:>7} s{s} FAILED — {r['error']}")
                else:
                    print(f"  {t.key:>5} {a['key']:>7} s{s} " + curve_row(r))
    print()
    print(f"  dropped: {dropped if dropped else 'nothing'}   raw records: {args.results}")
    print()


if __name__ == "__main__":
    main()
