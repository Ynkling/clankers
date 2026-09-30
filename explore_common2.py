#!/usr/bin/env python
"""
explore_common2.py — EXPLORATORY, not a result. Batch 2's additions to the outside-ideas harness.
explore_common.py is imported, not changed, so batch 1 stays reproducible as it ran.

What this adds:
  - ROUTED* (batch 2's routing criterion, from its BACKGROUND): margin >= 0.9 AND
    eta_key_by_stream > 0.9, read at a given evaluation step or at the end.
  - stats_b2: explore_common's stats (test_short_conv.conv_stats: gate stats every evaluation,
    routing stats at 1200 and the end) plus routing stats at ROUTE_EXTRA = (2400, 3600), and the
    model's k_top when it has one. Statistics run under eval() and no_grad and do not perturb
    training.
  - run_jobs2: explore_common.run_jobs with each run's lr schedule and seed on its progress line
    and in its record (sched); DRY-run store names.
  - a wall-clock projection (training steps timed per arm at the arm's lr, plus one evaluation).
  - reporting: per-seed tables with ROUTED* at 1200 / 2400 / 3600 / end, and paired McNemar rows
    for ROUTED* against X's arm A. X's record has routing statistics only at 1200 and the end, so
    "X's arm A routed at 2400" is read from explore_ref_a2400's re-run of arm A through step
    2400, counted only on seeds whose re-run reproduces X's record exactly (curve at 1200 and
    2400, and every routing statistic at 1200).
"""

import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch

import explore_common as ec
from explore_common import BANNER, recorded, tag, line, eta_str, bound, discovered
import test_short_conv as tsc
from test_router_layout import routing_stats
from test_router_confirm import mcnemar_exact
from test_binding_onset import onset_run, evaluate, eval_batch, EVAL_EVERY
from test_binding_recipe import makespan
from test_multilayer_binding import probe_batch

ROUTE_MARGIN, ROUTE_ETA = 0.9, 0.9
ROUTE_EXTRA = (2400, 3600)
ROUTE_AT = (EVAL_EVERY, 2400, 3600, "end")
TIME_STEPS = 100

DRY = dict(on=False)                    # set by explore_batch2 --dry


# ── Routing criterion and statistics ─────────────────────────────────────────
def stat_at(r, step):
    if r is None or not r.get("ok"):
        return None
    if step == "end":
        return r["end"]
    for s in r["stats"]:
        if s["step"] == step:
            return s
    return None


def routed_star(d):
    return bool(d is not None and d.get("margin") is not None
                and d["margin"] >= ROUTE_MARGIN and d["eta_key_by_stream"] > ROUTE_ETA)


def routed_at(r, step):
    return routed_star(stat_at(r, step))


@torch.no_grad()
def stats_b2(model, task, probe, step):
    out = tsc.conv_stats(model, task, probe, step)
    if step in ROUTE_EXTRA:
        out.update(routing_stats(model, task, probe))
    if hasattr(model, "k_top"):
        out["k_top"] = model.k_top
    return out


# ── Headers ──────────────────────────────────────────────────────────────────
def dry_label():
    return (" — DRY RUN (shortened: 1 seed per arm, runs capped at "
            f"{DRY.get('iters')} steps; exercises the code, not for interpretation)"
            if DRY["on"] else "")


def sched(a):
    return a.get("sched", f"lr {a['lr']:g} throughout")


def print_screen_header2(screen):
    print("-" * 100)
    print(f"[{BANNER}{dry_label()}] SCREEN {screen.NAME}")
    print(f"  idea    {screen.IDEA}")
    print(f"  source  {screen.SOURCE}")
    print(f"  change  {screen.CHANGE}")
    for k, a in screen.ARMS.items():
        print(f"  arm {k:<12} lr {a['lr']:g}  iters {a['iters']}  seeds {ec.fmt_seeds(a['seeds'])}"
              f"  — {a['label']}")
        print(f"      {'':<12} schedule: {sched(a)}")
    print(f"  pairing {screen.PAIRING}")
    print("-" * 100)


# ── Running ──────────────────────────────────────────────────────────────────
def _worker2(job):
    import importlib
    mod = importlib.import_module(job["screen"])
    a = mod.ARMS[job["arm"]]
    t = time.time()
    rec = mod.run_job(job["arm"], job["seed"])
    rec.update(secs_wall=time.time() - t, lr=a["lr"], sched=sched(a), dry=DRY["on"])
    return job, rec


def run_jobs2(screens, workers=ec.WORKERS, pv=None):
    stores = {s.NAME: ec.load_store(s.NAME) for s in screens}
    for s in screens:
        st = stores[s.NAME]
        st["meta"].setdefault("provenance", pv or ec.provenance())
        st["meta"]["dry"] = DRY["on"]
        st["meta"]["arms"] = {k: {kk: vv for kk, vv in a.items() if kk != "seeds"}
                              | {"seeds": list(a["seeds"]), "sched": sched(a)}
                              for k, a in s.ARMS.items()}
        ec.save_store(s.NAME, st)
    jobs = [dict(screen=s.__name__, name=s.NAME, arm=k, seed=seed, prio=a.get("prio", 1))
            for s in screens for k, a in s.ARMS.items() for seed in a["seeds"]
            if f"{k}|{seed}" not in stores[s.NAME]["runs"]]
    jobs.sort(key=lambda j: (-j["prio"], j["seed"]))
    print(f"  {len(jobs)} runs queued on {workers} workers (1 torch thread each)", flush=True)
    if not jobs:
        return stores
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork"),
                             initializer=ec._worker_init) as ex:
        futs = [ex.submit(_worker2, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            job, rec = f.result()
            st = stores[job["name"]]
            st["runs"][f"{job['arm']}|{job['seed']}"] = rec
            ec.save_store(job["name"], st)
            print(f"  [{n:>3}/{len(jobs)}] {(time.time() - t0) / 60:6.1f} min  {job['name']:<14} "
                  f"{job['arm']:<10} seed {job['seed']}  lr {rec['lr']:g} ({rec['sched']})\n"
                  f"          {line(rec) if rec.get('ok') else 'FAILED ' + str(rec.get('error'))}"
                  f"  ROUTED* 1200/2400/end "
                  f"{'/'.join('Y' if routed_at(rec, s) else '-' for s in (EVAL_EVERY, 2400, 'end'))}",
                  flush=True)
    return stores


# ── Projection ───────────────────────────────────────────────────────────────
def time_train(task, make, lr):
    t = time.time()
    onset_run(task, make, 0, max_iters=TIME_STEPS, eval_every=10 ** 9, data=None,
              early_stop=False, lr=lr)
    return (time.time() - t) / TIME_STEPS


def time_eval(task, make, stats_fn):
    m = make()
    data, probe = eval_batch(task), probe_batch(task, 0)
    t = time.time()
    evaluate(m, task, data)
    stats_fn(m, task, probe, ROUTE_EXTRA[0])
    return time.time() - t


def projection(screens, workers):
    """Worst case: every run goes to its iters (no early stop)."""
    torch.set_num_threads(ec.THREADS)
    print(f"  PROJECTION (worst case: every run to its iters; {TIME_STEPS} timed steps per model, "
          f"1 thread, this machine):")
    durs, total = [], 0.0
    for s in screens:
        for k, a in s.ARMS.items():
            segs = s.segments(k)                         # [(steps, task, make, lr, stats_fn)]
            per = 0.0
            parts = []
            for steps, task, make, lr, sfn in segs:
                ts = time_train(task, make, lr)
                te = time_eval(task, make, sfn)
                n_ev = steps // EVAL_EVERY
                per += steps * ts + n_ev * te
                parts.append(f"{steps} steps x {ts * 1e3:.1f} ms + {n_ev} evals x {te:.1f} s")
            durs += [per] * len(a["seeds"])
            total += per * len(a["seeds"])
            print(f"    {s.NAME:<14} {k:<10} {len(a['seeds']):>2} runs x {per / 60:5.1f} min  "
                  f"({'; '.join(parts)})")
    ms = makespan(durs, workers)
    print(f"    total {total / 3600:.2f} h of runs; makespan on {workers} workers {ms / 3600:.2f} h "
          f"(worst case)", flush=True)
    return ms


# ── Reporting ────────────────────────────────────────────────────────────────
def rflags(r):
    return "/".join("Y" if routed_at(r, s) else "-" for s in ROUTE_AT)


def per_seed_table2(screen, store, extra_recorded=("A", "A_conv")):
    rec = {k: recorded(k) for k in extra_recorded}
    print(f"  per seed  (ROUTED* = margin >= {ROUTE_MARGIN} and eta_key_by_stream > {ROUTE_ETA}, "
          f"at 1200/2400/3600/end" +
          (f"; X columns are X's test_short_conv runs at lr {ec.SUB_LR:g}" if extra_recorded else "")
          + "):")
    for k, a in screen.ARMS.items():
        print(f"   arm {k} (lr {a['lr']:g}; {sched(a)})")
        print(f"    {'seed':>4} {'acc':>6} {'trans':>6} {'VALcos':>7} {'margin':>6} "
              f"{'eta key s/k/h/i':>20} {'ROUTED*':>8}  {'outcome':<15} " +
              " ".join(f"{'X ' + x:<15}" for x in extra_recorded) +
              ("  X A ROUTED* 1200/-/-/end" if "A" in extra_recorded else ""))
        for s in a["seeds"]:
            r = store["runs"].get(f"{k}|{s}")
            recs = " ".join(f"{tag(rec[x].get(s)):<15}" for x in extra_recorded)
            xa = (f"  {'Y' if routed_at(rec['A'][s], EVAL_EVERY) else '-'}/?/?/"
                  f"{'Y' if routed_at(rec['A'][s], 'end') else '-'}" if "A" in extra_recorded else "")
            if r is None or not r.get("ok"):
                print(f"    {s:>4} {tag(r)}")
                continue
            m = r["end"].get("margin")
            print(f"    {s:>4} {r['acc']:6.3f} {str(r['transition']):>6} {r['val_cos']:7.3f} "
                  f"{'--' if m is None else f'{m:6.2f}':>6} {eta_str(r):>20} {rflags(r):>8}  "
                  f"{tag(r):<15} {recs}{xa}")


def ref_runs(ref_store):
    """Arm A re-run through 2400 (explore_ref_a2400), per seed, with whether it reproduces X."""
    X = recorded("A")
    out = {}
    for key, r in ref_store["runs"].items():
        s = int(key.split("|")[1])
        x = X[s]
        want = [c for c in x["curve"] if c[0] <= r.get("iters", 2400)]
        x12, r12 = stat_at(x, EVAL_EVERY), stat_at(r, EVAL_EVERY)
        keys = [k for k in x12 if k.startswith("eta_") or k == "margin"]
        same = bool(r.get("ok") and r["curve"] == want and all(r12.get(k) == x12[k] for k in keys))
        out[s] = dict(run=r, reproduces=same)
    return out


def x_routed(seed, step, refs):
    """X's arm A ROUTED* at step: 1200 and end from X's record; 2400 and 3600 from the re-run
    (None if the seed was not re-run or does not reproduce X)."""
    if step in (EVAL_EVERY, "end"):
        return routed_at(recorded("A")[seed], step)
    ref = refs.get(seed)
    if ref is None or not ref["reproduces"] or stat_at(ref["run"], step) is None:
        return None
    return routed_at(ref["run"], step)


def paired_routed(store, arm, seeds, step, refs):
    done = [s for s in seeds if (r := store["runs"].get(f"{arm}|{s}")) and r.get("ok")]
    pairs = [(s, routed_at(store["runs"][f"{arm}|{s}"], step), x_routed(s, step, refs))
             for s in done]
    miss = [s for s, _, x in pairs if x is None]
    pairs = [(s, n, x) for s, n, x in pairs if x is not None]
    b = sum(1 for _, n, x in pairs if n and not x)
    c = sum(1 for _, n, x in pairs if x and not n)
    p = mcnemar_exact(b, c)
    src = "X record" if step in (EVAL_EVERY, "end") else "re-run of X's arm A (explore_ref_a2400)"
    print(f"    ROUTED*@{str(step):<5} {arm:<10} {sum(n for _, n, _ in pairs):>2}/{len(pairs)}   "
          f"X A {sum(x for _, _, x in pairs):>2}/{len(pairs)}   {arm} only {b}, A only {c}, "
          f"McNemar two-sided p = {p:.3g}   [X A from {src}"
          + (f"; seeds without a reproducing re-run excluded: {miss}" if miss else "") + "]")
    return dict(n=len(pairs), b=b, c=c, p=p)


def band(d, thr_d=4, thr_p=0.10):
    """The screen rule's three bands on a paired (b, c, p): 'up', 'not up', 'middle'."""
    diff = d["b"] - d["c"]
    if diff >= thr_d and d["p"] < thr_p:
        return "up"
    if diff <= 0:
        return "not up"
    return "middle"
