#!/usr/bin/env python
"""
test_binding_recipe.py — do ordinary training settings make native BDH bind on every seed?

Run it directly:

    python test_binding_recipe.py                   # CHECKs, Stage A, Stage B, P=8 check
    python test_binding_recipe.py --workers 2 --force

WHY. test_binding_onset.py found no reliable binder: BDH bound in only a handful of its
35 runs within 38400 steps, while a 2-layer softmax transformer bound 5/5 within the first
evaluations. Perfect-binding solutions exist for native BDH at N=64 (some runs found
them), but most runs stall on flat partial plateaus (~0.33, ~0.53, ~0.75). The channel
comparison needs a single-channel BDH that binds on EVERY seed. This test asks whether
ORDINARY TRAINING SETTINGS get there with the architecture kept native: the learning rate
(4e-3) was tuned for the one-layer instrument, and N=64 is far below the reference
default (N=8192). Out of scope: separate Q/K projections, softmax, or any other
architecture change.

REUSE. onset_run, make_bdh, PRIMARY, evaluate, eval_batch, transition, time_per_step,
verify, attempt, print_run, row_stats and curve_row come from test_binding_onset.py;
MultiBDH, BindTask, stats, cell and save_results from test_multilayer_binding.py. None is
rewritten. test_binding_onset.py gained three backward-compatible knobs: onset_run(lr=,
warmup=) — linear LR warmup over the first W steps, the param-group lr set per step only
when W > 0 — and make_bdh(mult=), which sets N = mult * 32 with n_head=1 through
MultiBDH's existing argument. CHECK 11 proves the defaults are bit-identical to the
pre-edit code.

TASK. PRIMARY (S=1, P=4, n_vals=16, n_q=1), n_layer=3, held-out evaluation every 1200
steps, stopping after 3 consecutive evaluations >= 0.95 — exactly test_binding_onset.py.
A run is BOUND if it has a transition: an evaluation >= 0.95 that holds to the end.

STAGE A: SCREEN. 3 seeds (0-2), MAX_ITERS=19200, warmup=0, over
  positional {rope, decay} x lr {1e-3, 2e-3, 4e-3, 8e-3} x mult {2, 8, 32} (N 64/256/1024)
  24 cells, 72 runs. Per cell: seeds bound (k/3), transition steps, and each
  non-binder's final accuracy — whether a setting moves runs between plateaus even when
  it does not bind them.
STAGE B: CONFIRM. Up to 3 cells with >= 1 bound seed in Stage A, ranked by seeds bound,
  then lower median transition (then lower max, then grid order). Each runs 10 seeds
  (0-9) to MAX_ITERS=38400. A Stage-A run is reused only if it stopped early (its last 3
  evaluations >= 0.95): training is deterministic, so the 38400-step run would be the same
  run. The top-ranked cell also runs with warmup=1000, 10 seeds. No Stage-A binder: Stage
  B is skipped.
P=8 CHECK (informational): every Stage-B cell that binds 10/10 runs at P=8, 5 seeds.

PRE-REGISTERED VERDICT, printed before any interpretation; exactly one of
  RELIABLE CONFIG  a Stage-B cell (including the warmup variant) binds 10/10. Among such
                   cells the smallest max transition. Prints positional, lr, mult/N,
                   warmup and RECOMMENDED BUDGET = ceil(1.25 x max transition / 1200) x 1200.
  NOT RELIABLE     no Stage-B cell binds 10/10, or Stage B was skipped. Prints the best
                   cell and its rate, then "FALLBACK: switch to the bigram-write substrate."
9/10 is NOT RELIABLE. A FAILED run is recorded and counts as not bound.

LOGISTICS. Runs execute in a pool of single-threaded worker processes (CHECK 14 proves a
worker's run is bit-identical to the same run in this process). Wall clock is projected
per cell before Stage A from time_per_step measured inside the pool; if Stage A's
projected wall clock exceeds 5 hours, lr=1e-3 is dropped first, then mult=32, and that
is said. Stage A is reported before Stage B starts. Results persist atomically after every
run to binding_recipe_results.json (gitignored); its meta records the torch version, CPU
model, worker and thread counts. Seed-level outcomes differ between machines.

(Results are recorded at the bottom of this docstring after the run.)
"""

import argparse
import math
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
from torch.optim.optimizer import (register_optimizer_step_pre_hook,
                                   register_optimizer_step_post_hook)

import test_binding_onset as tbo
from test_binding_onset import (
    onset_run, make_bdh, PRIMARY, evaluate, eval_batch, transition, time_per_step,
    attempt, print_run, row_stats, fmt_step, curve_row, load_results,
    BIND_PASS, STOP_AFTER, EVAL_EVERY, N_VALS,
)
from test_multilayer_binding import MultiBDH, BindTask, stats, cell, save_results, MULT
from test_instrument_v2 import LR, D

# ── Settings ─────────────────────────────────────────────────────────────────
N_LAYER = 3
POSITIONALS = ("rope", "decay")
LRS = (1e-3, 2e-3, 4e-3, 8e-3)
MULTS = (2, 8, 32)
A_SEEDS, A_ITERS = (0, 1, 2), 19200
B_SEEDS, B_ITERS = tuple(range(10)), 38400
B_CELLS = 3
B_WARMUP = 1000
P8, P8_SEEDS, P8_ITERS = 8, tuple(range(5)), 38400
WALL_LIMIT_H = 5.0
LEGACY_SHA = "966b743"            # test_binding_onset.py before the lr/warmup/mult knobs
WORKERS = min(4, os.cpu_count() or 1)
RESULTS_FILE = "binding_recipe_results.json"
BANDS = ((0.0, 0.43, "<0.43"), (0.43, 0.65, "0.43-0.65"), (0.65, BIND_PASS, "0.65-0.95"))


# ── Cells, specs, names ──────────────────────────────────────────────────────
def lr_str(lr):
    return f"{lr:.0e}".replace("e-0", "e-")


def cname(pos, lr, mult, warmup, P):
    return f"{pos}-lr{lr_str(lr)}-N{mult * D}-w{warmup}-P{P}"


def clabel(pos, lr, mult, warmup=0, P=4):
    s = f"{pos:<5} lr={lr_str(lr)} N={mult * D:<4}"
    if warmup:
        s += f" warmup={warmup}"
    if P != 4:
        s += f" P={P}"
    return s


def spec(cellt, seed, iters):
    pos, lr, mult, warmup, P = cellt
    return dict(pos=pos, lr=lr, mult=mult, warmup=warmup, P=P, seed=seed, iters=iters)


def skey(sp):
    return f"i{sp['iters']}|{cname(sp['pos'], sp['lr'], sp['mult'], sp['warmup'], sp['P'])}" \
           f"|{sp['seed']}"


_TASKS = {}


def task_for(P):
    if P == PRIMARY.P:
        return PRIMARY
    if P not in _TASKS:
        t = BindTask(P, S=1, n_vals=N_VALS, n_q=1)
        t.key = f"Q1P{P}"
        _TASKS[P] = t
    return _TASKS[P]


# ── Worker-side functions (spawned processes, one thread each) ───────────────
def init_worker():
    torch.set_num_threads(1)


def run_spec(sp):
    task = task_for(sp["P"])
    rec = attempt(task, make_bdh(task, N_LAYER, sp["pos"], mult=sp["mult"]), sp["seed"],
                  eval_batch(task), max_iters=sp["iters"], lr=sp["lr"], warmup=sp["warmup"])
    rec["spec"] = sp
    return rec


def time_spec(sp):
    task = task_for(sp["P"])
    return time_per_step(task, make_bdh(task, N_LAYER, sp["pos"], mult=sp["mult"]))


def weights_run(sp):
    """A short run returning its full curve and weights, for CHECK 14."""
    task = task_for(sp["P"])
    model, curve, _ = onset_run(task, make_bdh(task, N_LAYER, sp["pos"], mult=sp["mult"]),
                                sp["seed"], max_iters=sp["iters"], eval_every=sp["every"],
                                data=eval_batch(task), early_stop=False, lr=sp["lr"],
                                warmup=sp["warmup"])
    return curve, {n: p.detach().clone() for n, p in model.named_parameters()}


# ── Verification ─────────────────────────────────────────────────────────────
def load_legacy():
    src = subprocess.run(["git", "-C", HERE, "show", f"{LEGACY_SHA}:test_binding_onset.py"],
                         capture_output=True, text=True, check=True).stdout
    mod = types.ModuleType("binding_onset_legacy")
    mod.__file__ = os.path.join(HERE, "test_binding_onset.py")
    exec(compile(src, f"test_binding_onset.py@{LEGACY_SHA}", "exec"), mod.__dict__)
    return mod


def same_weights(a, b):
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    return sorted(pa) == sorted(pb) and all(torch.equal(pa[n], pb[n]) for n in pa)


def verify(pool):
    print("test_binding_onset.py's own CHECKs, after the lr/warmup/mult edit:")
    tbo.verify()           # CHECKs 1-6 (test_multilayer_binding) and 7-10 (test_binding_onset)
    print("=" * 100)
    print("VERIFICATION — this test's own CHECKs")
    print("=" * 100)
    ok = True

    print(f"CHECK 11 the knobs are inert at their defaults: 1200 steps, evaluations every 400,")
    print(f"         onset_run/make_bdh at {LEGACY_SHA} (before the edit) vs now with implicit")
    print(f"         defaults vs now with explicit defaults (lr={LR}, warmup=0, mult={MULT}):")
    legacy = load_legacy()
    import inspect
    print(f"         legacy onset_run has an lr argument: "
          f"{'lr' in inspect.signature(legacy.onset_run).parameters}   "
          f"current: {'lr' in inspect.signature(onset_run).parameters}")
    data = eval_batch(PRIMARY)
    for pos in POSITIONALS:
        m0, c0, _ = legacy.onset_run(PRIMARY, legacy.make_bdh(PRIMARY, N_LAYER, pos), 0,
                                     max_iters=1200, eval_every=400, data=data,
                                     early_stop=False)
        m1, c1, _ = onset_run(PRIMARY, make_bdh(PRIMARY, N_LAYER, pos), 0, max_iters=1200,
                              eval_every=400, data=data, early_stop=False)
        m2, c2, _ = onset_run(PRIMARY, make_bdh(PRIMARY, N_LAYER, pos, mult=MULT), 0,
                              max_iters=1200, eval_every=400, data=data, early_stop=False,
                              lr=LR, warmup=0)
        good = same_weights(m0, m1) and same_weights(m1, m2) and c0 == c1 == c2
        ok &= good
        print(f"         {pos:<5} weights legacy=implicit: {same_weights(m0, m1)}  "
              f"implicit=explicit: {same_weights(m1, m2)}   curves equal: {c0 == c1 == c2}   "
              f"accs {[round(a, 4) for _, a, _ in c1]}  -> {'IDENTICAL' if good else 'DIFFERS'}")
    print()

    W, lr = B_WARMUP, 4e-3
    print(f"CHECK 12 warmup applies the linear schedule: lr={lr}, warmup={W}. The lr Adam reads")
    print(f"         at each step() is logged by a global step pre-hook; at step 1 Adam moves")
    print(f"         each weight by lr * |g|/(|g|+eps) ~= lr, so the largest weight change")
    print(f"         measures the applied lr independently of the param group:")
    for w in (W, 0):
        log = {"n": 0, "lr": {}, "dmax": None}
        snap = {}

        def pre(opt, args, kwargs):
            log["n"] += 1
            log["lr"][log["n"]] = [g["lr"] for g in opt.param_groups]
            if log["n"] == 1:
                snap["p"] = [p.detach().clone() for g in opt.param_groups for p in g["params"]]

        def post(opt, args, kwargs):
            if log["n"] == 1:
                now = [p.detach() for g in opt.param_groups for p in g["params"]]
                log["dmax"] = max((a - b).abs().max().item() for a, b in zip(now, snap["p"]))

        h1, h2 = register_optimizer_step_pre_hook(pre), register_optimizer_step_post_hook(post)
        try:
            onset_run(PRIMARY, make_bdh(PRIMARY, N_LAYER, "rope"), 0, max_iters=W + 1,
                      lr=lr, warmup=w)
        finally:
            h1.remove(); h2.remove()
        print(f"         warmup={w}:")
        for s in (1, W // 2, W + 1):
            want = lr * min(1.0, s / w) if w > 0 else lr
            got = log["lr"][s]
            good = len(got) == 1 and got[0] == want
            ok &= good
            print(f"           step {s:>4}: applied lr {got[0]!r:<24} schedule {want!r:<24} "
                  f"-> {'EQUAL' if good else 'DIFFERS'}")
        want1 = lr * min(1.0, 1 / w) if w > 0 else lr
        good = abs(log["dmax"] / want1 - 1) < 1e-2
        ok &= good
        print(f"           step 1 largest weight change {log['dmax']:.4e} vs lr {want1:.4e} "
              f"-> {'MATCHES' if good else 'DIFFERS'}")
    print()

    print("CHECK 13 mult sets N = mult * 32 with n_head=1, built exactly as make_bdh builds it:")
    for mult in MULTS:
        m = make_bdh(PRIMARY, N_LAYER, "rope", mult=mult)()
        n = mult * D
        good = (m.n_feat == n and m.config.n_head == 1 and tuple(m.encoder.shape) == (1, D, n)
                and tuple(m.decoder.shape) == (n, D) and m.n_layers == N_LAYER)
        ok &= good
        print(f"         mult={mult:<2}  N={m.n_feat:<4}  encoder {tuple(m.encoder.shape)}  "
              f"decoder {tuple(m.decoder.shape)}  params {sum(p.numel() for p in m.parameters())}"
              f"  -> {'OK' if good else 'WRONG'}")
    print()

    sp = dict(pos="rope", lr=8e-3, mult=8, warmup=100, P=PRIMARY.P, seed=3, iters=600,
              every=200)
    print(f"CHECK 14 a run in a spawned single-thread worker is bit-identical to the same run")
    print(f"         here: {clabel(sp['pos'], sp['lr'], sp['mult'], sp['warmup'])}, seed "
          f"{sp['seed']}, {sp['iters']} steps, evaluations every {sp['every']}:")
    c_here, w_here = weights_run(sp)
    c_pool, w_pool = pool.submit(weights_run, sp).result()
    wsame = sorted(w_here) == sorted(w_pool) and all(torch.equal(w_here[n], w_pool[n])
                                                    for n in w_here)
    good = wsame and c_here == c_pool
    ok &= good
    print(f"         weights bitwise equal: {wsame}   curves equal: {c_here == c_pool}   "
          f"accs {[round(a, 4) for _, a, _ in c_here]}  -> {'IDENTICAL' if good else 'DIFFERS'}")
    print()
    print(f"{'ALL VERIFICATION CHECKS PASSED' if ok else 'SOME VERIFICATION CHECKS FAILED'}\n")
    assert ok, "verification failed — do not trust the results below"


# ── Running specs through the pool ───────────────────────────────────────────
def run_all(specs, store, path, pool, force, t0, cost):
    todo = []
    for sp in specs:
        if skey(sp) in store["runs"] and not force:
            r = store["runs"][skey(sp)]
            print(f"   {skey(sp)}  cached ({'ok' if r.get('ok') else 'FAILED'})")
        else:
            todo.append(sp)
    todo.sort(key=lambda sp: -sp["iters"] * cost.get((sp["pos"], sp["mult"]), 1.0))
    futs = {pool.submit(run_spec, sp): sp for sp in todo}
    for f in as_completed(futs):
        sp = futs[f]
        try:
            rec = f.result()
        except Exception as e:
            rec = dict(ok=False, seed=sp["seed"], error=f"worker {type(e).__name__}: {e}",
                       spec=sp, secs=0.0)
        store["runs"][skey(sp)] = rec
        save_results(path, store)
        print_run(f"i{sp['iters']}", cname(sp["pos"], sp["lr"], sp["mult"], sp["warmup"],
                                           sp["P"]), sp["seed"], rec, t0)


def makespan(durations, workers):
    loads = [0.0] * workers
    for d in sorted(durations, reverse=True):
        i = loads.index(min(loads))
        loads[i] += d
    return max(loads)


def stopped_early(r):
    return (r is not None and r.get("ok") and len(r["curve"]) >= STOP_AFTER
            and all(a >= BIND_PASS for _, a, _ in r["curve"][-STOP_AFTER:]))


# ── Reporting ────────────────────────────────────────────────────────────────
def outcome(r):
    if r is None:
        return "--"
    if not r.get("ok"):
        return "FAIL"
    if r["transition"]:
        return f"@{r['transition']}"
    return f"{r['curve'][-1][1]:.3f}" if r["curve"] else "n/a"


def cell_rows(store, cells, seeds, iters, header):
    print(f"  {header:<38} {'bound':>7} {'median':>7} {'max':>7}   per seed "
          f"(@transition, or final acc if not bound)")
    out = {}
    for c in cells:
        name = cname(*c)
        rs = row_stats(store, f"i{iters}", name, seeds)
        out[c] = rs
        per = [outcome(store["runs"].get(f"i{iters}|{name}|{s}")) for s in seeds]
        print(f"  {clabel(*c):<38} {rs['bound']:>3}/{rs['total']:<3} {fmt_step(rs['med']):>7} "
              f"{fmt_step(rs['max']):>7}   {' '.join(f'{p:>6}' for p in per)}"
              + (f"   [{rs['failed']} FAILED]" if rs["failed"] else ""))
    return out


def marginals(store, cells, seeds, iters):
    print(f"  Marginals over the screened cells: seeds bound, and non-binders' final accuracy "
          f"by band ({' | '.join(b[2] for b in BANDS)}):")
    factors = (("positional", 0, POSITIONALS), ("lr", 1, LRS), ("N", 2, MULTS))
    for fname, idx, levels in factors:
        for lv in levels:
            cs = [c for c in cells if c[idx] == lv]
            if not cs:
                continue
            recs = [store["runs"].get(f"i{iters}|{cname(*c)}|{s}") for c in cs for s in seeds]
            ok = [r for r in recs if r is not None and r.get("ok")]
            bound = sum(1 for r in ok if r["transition"])
            finals = [r["curve"][-1][1] for r in ok if not r["transition"] and r["curve"]]
            bands = [sum(1 for x in finals if lo <= x < hi) for lo, hi, _ in BANDS]
            lvs = lr_str(lv) if fname == "lr" else (lv * D if fname == "N" else lv)
            print(f"    {fname:>10} = {str(lvs):<6} bound {bound:>2}/{len(recs):<3}  "
                  f"non-binders by band: {' | '.join(f'{b:>2}' for b in bands)}   "
                  f"non-binder final acc {cell(stats(finals))}")
    print()


def print_curves(store, cells, seeds, iters, title):
    print(f"  {title}")
    for c in cells:
        for s in seeds:
            r = store["runs"].get(f"i{iters}|{cname(*c)}|{s}")
            lab = f"{clabel(*c)} s{s}"
            if r is None:
                print(f"    {lab:<44} NOT RUN")
            elif not r.get("ok"):
                print(f"    {lab:<44} FAILED — {r['error']}")
            else:
                print(f"    {lab:<44}" + curve_row(r)
                      + ("   (reused from Stage A)" if r.get("reused_from") else ""))


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Do ordinary training settings make BDH bind?")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--results", default=RESULTS_FILE)
    args = ap.parse_args()
    torch.set_num_threads(1)
    head = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()

    print("=" * 100)
    print("Binding recipe: do ordinary training settings make native BDH bind on every seed?")
    print(f"  PRIMARY S=1 P={PRIMARY.P} n_vals={N_VALS} n_q=1, n_layer={N_LAYER}, eval every "
          f"{EVAL_EVERY}, stop after {STOP_AFTER} evals >= {BIND_PASS}")
    print(f"  torch {torch.__version__}   CPU {cpu_model()}   {args.workers} worker processes x "
          f"1 thread (cpu_count {os.cpu_count()})   git {head}")
    print("=" * 100)
    print()

    # Workers re-import torch; keep its missing-numpy import warning out of the run log.
    os.environ.setdefault("PYTHONWARNINGS", "ignore:Failed to initialize NumPy")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx,
                             initializer=init_worker) as pool:
        verify(pool)
        store = load_results(args.results)
        for k in ("stageA", "stageB", "p8", "verdict"):
            store.setdefault(k, {})
        store["meta"] = dict(torch=torch.__version__, cpu=cpu_model(), workers=args.workers,
                             threads_per_worker=1, cpu_count=os.cpu_count(), git=head,
                             started=time.strftime("%Y-%m-%d %H:%M:%S"),
                             note="seed-level outcomes differ between machines")
        save_results(args.results, store)
        t0 = time.time()

        print("=" * 100)
        print(f"PROJECTION — time_per_step measured inside the pool ({args.workers} workers "
              f"concurrently); worst case = every run to MAX_ITERS")
        print("=" * 100)
        tspecs = [dict(pos=p, mult=m, P=PRIMARY.P) for p in POSITIONALS for m in MULTS]
        cost = {(s["pos"], s["mult"]): c for s, c in zip(tspecs, pool.map(time_spec, tspecs))}
        for (p, m), c in cost.items():
            print(f"  {p:<5} N={m * D:<4}: {c * 1000:6.1f} ms/step   worst case per cell "
                  f"({len(A_SEEDS)} seeds x {A_ITERS}): {len(A_SEEDS) * A_ITERS * c / 60:5.1f} min")

        def project(lrs, mults):
            durs = [A_ITERS * cost[(p, m)] for p in POSITIONALS for _ in lrs for m in mults
                    for _ in A_SEEDS]
            return sum(durs) / 3600, makespan(durs, args.workers) / 3600

        lrs, mults = LRS, MULTS
        serial, wall = project(lrs, mults)
        print(f"  Stage A, full grid ({len(POSITIONALS) * len(lrs) * len(mults)} cells): serial "
              f"{serial:.2f} h, projected wall clock on {args.workers} workers {wall:.2f} h "
              f"(limit {WALL_LIMIT_H} h)")
        dropped = []
        if wall > WALL_LIMIT_H:
            lrs = tuple(x for x in lrs if x != 1e-3)
            dropped.append("lr=1e-3")
            serial, wall = project(lrs, mults)
            print(f"  -> OVER: lr=1e-3 DROPPED. Now serial {serial:.2f} h, wall {wall:.2f} h.")
        if wall > WALL_LIMIT_H:
            mults = tuple(x for x in mults if x != 32)
            dropped.append("mult=32")
            serial, wall = project(lrs, mults)
            print(f"  -> STILL OVER: mult=32 (N=1024) DROPPED. Now serial {serial:.2f} h, "
                  f"wall {wall:.2f} h.")
        if wall > WALL_LIMIT_H:
            print("  -> still over the limit after both drops; running what remains anyway.")
        elif not dropped:
            print("  -> within the limit: full grid.")
        store["meta"]["dropped"] = dropped
        store["meta"]["cost_ms"] = {f"{p}|{m}": c * 1000 for (p, m), c in cost.items()}
        print()

        cells_A = [(p, lr, m, 0, PRIMARY.P) for p in POSITIONALS for lr in lrs for m in mults]
        print("=" * 100)
        print(f"STAGE A — screen: {len(cells_A)} cells x seeds {list(A_SEEDS)}, MAX_ITERS="
              f"{A_ITERS}, warmup=0")
        print("=" * 100)
        run_all([spec(c, s, A_ITERS) for c in cells_A for s in A_SEEDS], store, args.results,
                pool, args.force, t0, cost)
        print()
        print("#" * 100)
        print("STAGE A REPORT")
        print("#" * 100)
        rows_A = cell_rows(store, cells_A, A_SEEDS, A_ITERS, "cell (n_layer=3, n_q=1, P=4)")
        print()
        marginals(store, cells_A, A_SEEDS, A_ITERS)
        big = float("inf")
        cand = [c for c in cells_A if rows_A[c]["bound"] >= 1]
        ranked = sorted(cand, key=lambda c: (-rows_A[c]["bound"],
                                             big if rows_A[c]["med"] is None else rows_A[c]["med"],
                                             big if rows_A[c]["max"] is None else rows_A[c]["max"],
                                             cells_A.index(c)))
        chosen = ranked[:B_CELLS]
        store["stageA"] = {cname(*c): dict(bound=rows_A[c]["bound"], med=rows_A[c]["med"],
                                           max=rows_A[c]["max"]) for c in cells_A}
        if chosen:
            print(f"  Stage B selection ({len(cand)} cells bound >= 1 seed; ranked by seeds bound,"
                  f" then median transition):")
            for i, c in enumerate(ranked):
                tag = "SELECTED" if c in chosen else "not selected"
                print(f"    {i + 1}. {clabel(*c):<38} {rows_A[c]['bound']}/{len(A_SEEDS)} median "
                      f"{fmt_step(rows_A[c]['med'])}  {tag}")
        else:
            print("  No Stage-A cell bound any seed: Stage B is SKIPPED.")
        print(f"  (Stage A elapsed {(time.time() - t0) / 60:.1f} min)")
        print()
        save_results(args.results, store)

        cells_B = []
        if chosen:
            cells_B = list(chosen) + [(chosen[0][0], chosen[0][1], chosen[0][2], B_WARMUP,
                                       PRIMARY.P)]
            print("=" * 100)
            print(f"STAGE B — confirm: {len(cells_B)} cells x seeds {list(B_SEEDS)}, MAX_ITERS="
                  f"{B_ITERS} (the last is the top cell with warmup={B_WARMUP})")
            print("=" * 100)
            for c in chosen:
                for s in A_SEEDS:
                    ra = store["runs"].get(skey(spec(c, s, A_ITERS)))
                    kb = skey(spec(c, s, B_ITERS))
                    if stopped_early(ra) and (kb not in store["runs"] or args.force):
                        store["runs"][kb] = dict(ra, reused_from=skey(spec(c, s, A_ITERS)))
                        print(f"   {kb}  reused from Stage A (stopped early at "
                              f"{ra['stopped_at']}, transition {ra['transition']})")
            save_results(args.results, store)
            bspecs = [spec(c, s, B_ITERS) for c in cells_B for s in B_SEEDS
                      if not store["runs"].get(skey(spec(c, s, B_ITERS)), {}).get("reused_from")]
            durs = [B_ITERS * cost[(sp["pos"], sp["mult"])] for sp in bspecs]
            print(f"   {len(bspecs)} runs to go; worst case serial {sum(durs) / 3600:.2f} h, "
                  f"wall on {args.workers} workers {makespan(durs, args.workers) / 3600:.2f} h")
            run_all(bspecs, store, args.results, pool, args.force, t0, cost)
            print()

        print("#" * 100)
        print("STAGE B REPORT")
        print("#" * 100)
        rows_B = {}
        if cells_B:
            rows_B = cell_rows(store, cells_B, B_SEEDS, B_ITERS, "cell (n_layer=3, n_q=1, P=4)")
        else:
            print("  Stage B skipped: no Stage-A cell bound any seed.")
        print()

        print("#" * 100)
        print("PRE-REGISTERED VERDICT")
        print("#" * 100)
        full = [c for c in cells_B if rows_B[c]["bound"] == len(B_SEEDS)]
        if full:
            best = min(full, key=lambda c: (rows_B[c]["max"], cells_B.index(c)))
            rs = rows_B[best]
            budget = math.ceil(1.25 * rs["max"] / EVAL_EVERY) * EVAL_EVERY
            pos, lr, mult, warmup, _ = best
            print(f"  RELIABLE CONFIG: positional={pos}, lr={lr_str(lr)}, mult={mult} "
                  f"(N={mult * D}), warmup={warmup} — {rs['bound']}/{len(B_SEEDS)} seeds bound, "
                  f"median transition {fmt_step(rs['med'])}, max {fmt_step(rs['max'])}")
            print(f"  RECOMMENDED BUDGET = ceil(1.25 x {rs['max']:.0f} / {EVAL_EVERY}) x "
                  f"{EVAL_EVERY} = {budget} steps.")
            store["verdict"] = dict(verdict="RELIABLE CONFIG", cell=cname(*best), budget=budget,
                                    bound=rs["bound"], max=rs["max"])
        else:
            if cells_B:
                best = min(cells_B, key=lambda c: (-rows_B[c]["bound"],
                                                   big if rows_B[c]["med"] is None
                                                   else rows_B[c]["med"], cells_B.index(c)))
                rs, n, where = rows_B[best], len(B_SEEDS), "Stage B"
            else:
                best = min(cells_A, key=lambda c: (-rows_A[c]["bound"],
                                                   big if rows_A[c]["med"] is None
                                                   else rows_A[c]["med"], cells_A.index(c)))
                rs, n, where = rows_A[best], len(A_SEEDS), "Stage A (Stage B skipped)"
            print(f"  NOT RELIABLE: no Stage-B cell bound all {len(B_SEEDS)} seeds"
                  + ("." if cells_B else " — Stage B was skipped."))
            print(f"  Best cell ({where}): {clabel(*best).strip()} — {rs['bound']}/{n} = "
                  f"{rs['bound'] / n:.0%}" + (f", transitions {sorted(rs['trs'])}"
                                             if rs["trs"] else ""))
            print("  FALLBACK: switch to the bigram-write substrate.")
            store["verdict"] = dict(verdict="NOT RELIABLE", best=cname(*best), bound=rs["bound"],
                                    n=n, where=where)
        save_results(args.results, store)
        print()

        cells_8 = [(c[0], c[1], c[2], c[3], P8) for c in full]
        print("=" * 100)
        print(f"P=8 CHECK (informational) — Stage-B cells that bound {len(B_SEEDS)}/"
              f"{len(B_SEEDS)}, at P={P8}, seeds {list(P8_SEEDS)}, MAX_ITERS={P8_ITERS}")
        print("=" * 100)
        if cells_8:
            run_all([spec(c, s, P8_ITERS) for c in cells_8 for s in P8_SEEDS], store,
                    args.results, pool, args.force, t0, cost)
            print()
            cell_rows(store, cells_8, P8_SEEDS, P8_ITERS, "cell (n_layer=3, n_q=1, P=8)")
        else:
            print("  No Stage-B cell bound 10/10: nothing to run at P=8.")
        print()

    print("=" * 100)
    print(f"CURVES — held-out accuracy x100 every {EVAL_EVERY} steps; '|' marks the transition")
    print("=" * 100)
    print_curves(store, cells_A, A_SEEDS, A_ITERS, f"Stage A (MAX_ITERS={A_ITERS})")
    if cells_B:
        print_curves(store, cells_B, B_SEEDS, B_ITERS, f"Stage B (MAX_ITERS={B_ITERS})")
    if cells_8:
        print_curves(store, cells_8, P8_SEEDS, P8_ITERS, f"P=8 (MAX_ITERS={P8_ITERS})")
    print()
    wall = time.time() - t0
    store["meta"]["wall_s"] = wall
    save_results(args.results, store)
    print(f"  total wall clock after verification: {wall / 60:.1f} min ({wall:.0f}s)")
    print(f"  raw records: {args.results}   (dropped: {dropped if dropped else 'nothing'})")
    print()


if __name__ == "__main__":
    main()
