#!/usr/bin/env python
"""
explore_common.py — EXPLORATORY, not a result. The shared harness of the outside-ideas screens
(branch claude/outside-ideas). It imports the test modules read-only and changes none of them.

WHAT A SCREEN IS. A quick look at one idea from outside this project, on few seeds, to decide
whether it earns a pre-registered test on the main branch. A screen's verdict is one of
promising / not / inconclusive. It is never a result.

THE RECIPE (test_short_conv's P=4 recipe, the one X's recorded arm A ran under):
  task       test_short_conv.TASK = test_channel_binding.task_for(4, 2): P=4, S=2, n_vals=16,
             n_q=1, grouped layout.
  model      MultiBDH n_layer=3, positional='decay', N=256 (test_channel_binding.SUB_MULT=8),
             n_head=1, dropout=0; arm A = recurrent gate, k=2, no readout.
  training   test_router_reliability.run_one -> test_binding_onset.onset_run: Adam, BATCH 32,
             lr = test_channel_binding.SUB_LR, read here and passed to run_one EXPLICITLY;
             held-out evaluation every EVAL_EVERY=1200 steps on eval_batch(task); early stop
             after 3 evaluations >= 0.95; MAX_ITERS = test_short_conv.MAX_ITERS.
  records    run_one with test_short_conv's stats_fn (conv_stats) and grad_fn (conv_grad_norms):
             every record has the fields test_short_conv's records have (curve, transition,
             VAL/KEY/CTX cos, margin, eta^2 by stream/key/half/index at 1200 and the end).

PAIRING. Seeds 160-179. X's recorded arm A (and A_conv, B_conv) on these seeds are in
results/X/short_conv_results.json. This container reproduces X's recorded arm-A seed-160 curve
bit for bit (repro_check, run before every batch); if that check fails, the recorded outcomes are
not a valid pairing and arm A is re-run here instead.

Labels, as in the main tests: BOUND = transition is not None; DISCOVERED = bound and final VAL
cos < 0.5; ROUTED = end margin >= 0.9; failure classes from test_router_layout.fail_class.
"""

import json
import math
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch

import test_channel_binding as tcb
import test_short_conv as tsc
from test_router_reliability import run_one
from test_router_confirm import mcnemar_exact, fisher_greater
from test_router_layout import fail_class
from test_router_discovery import DISC_COS

BANNER = "EXPLORATORY, not a result"
OUT_DIR = os.path.join(HERE, "explore_out")
RECORDED_FILE = os.path.join(HERE, "results", "X", "short_conv_results.json")

# The recipe's constants, read from the test they come from (not from function defaults).
SUB_LR = tcb.SUB_LR                    # 1e-3
MAX_ITERS = tsc.MAX_ITERS              # 24000
TASK = tsc.TASK                        # task_for(4, 2), grouped
ARM_A = tsc.ARM["A"]                   # test_router_curriculum's arm A
SEEDS = tuple(range(160, 180))
ROUTED_MARGIN = tsc.ROUTED_MARGIN      # 0.9
WORKERS = min(4, os.cpu_count() or 1)

REPRO_SEED, REPRO_ITERS = 160, 2400


# ── Provenance ───────────────────────────────────────────────────────────────
def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def git_head():
    try:
        sha = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", HERE, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception as e:                                      # noqa: BLE001
        return f"unknown ({e})"


def provenance():
    return dict(banner=BANNER, cpu=cpu_model(), torch=torch.__version__, git=git_head(),
                python=platform.python_version(), started=time.strftime("%Y-%m-%d %H:%M:%S"))


def print_banner(title):
    print("=" * 100)
    print(f"{BANNER.upper()} — {title}")
    pv = provenance()
    print(f"  cpu {pv['cpu']} | torch {pv['torch']} | git {pv['git']} | python {pv['python']}")
    print("=" * 100)
    return pv


def print_screen_header(screen):
    """screen: a module with NAME, IDEA, SOURCE, CHANGE and ARMS (key -> dict with lr, seeds)."""
    print("-" * 100)
    print(f"[{BANNER}] SCREEN {screen.NAME}")
    print(f"  idea    {screen.IDEA}")
    print(f"  source  {screen.SOURCE}")
    print(f"  change  {screen.CHANGE}")
    for k, a in screen.ARMS.items():
        print(f"  arm {k:<14} lr {a['lr']:g}  iters {a['iters']}  seeds {fmt_seeds(a['seeds'])}"
              f"  — {a['label']}")
    print(f"  paired with X's recorded arm A (test_short_conv, lr {SUB_LR:g}), seeds "
          f"{fmt_seeds(SEEDS)}")
    print("-" * 100)


def fmt_seeds(seeds):
    seeds = list(seeds)
    if seeds and seeds == list(range(seeds[0], seeds[-1] + 1)):
        return f"{seeds[0]}-{seeds[-1]} ({len(seeds)})"
    return f"{seeds} ({len(seeds)})"


# ── Reproduction check: does this container reproduce X's recorded arm A? ────
def recorded(arm):
    with open(RECORDED_FILE) as f:
        d = json.load(f)
    return {int(k.split("|")[1]): r for k, r in d["runs"].items() if k.split("|")[0] == arm}


def repro_check():
    rec = recorded("A")[REPRO_SEED]
    t = time.time()
    r = run_one(ARM_A, REPRO_SEED, REPRO_ITERS, task=TASK, stats_fn=tsc.conv_stats,
                grad_fn=tsc.conv_grad_norms, lr=SUB_LR)
    want = [c for c in rec["curve"] if c[0] <= REPRO_ITERS]
    ok = r["ok"] and r["curve"] == want
    print(f"  repro_check: arm A seed {REPRO_SEED}, {REPRO_ITERS} steps, lr {SUB_LR:g}: "
          f"{'BIT-IDENTICAL to X' if ok else 'DIFFERS from X'} ({time.time() - t:.0f}s)")
    print(f"    here {r.get('curve')}\n    X    {want}")
    return ok


# ── Running ──────────────────────────────────────────────────────────────────
def run_recipe(a, seed, builder=None, run_kw=None, iters=None, lr=None, task=None):
    """One run through run_one at the recipe, lr passed explicitly."""
    return run_one(a, seed, MAX_ITERS if iters is None else iters,
                   task=TASK if task is None else task, stats_fn=tsc.conv_stats,
                   grad_fn=tsc.conv_grad_norms, lr=SUB_LR if lr is None else lr,
                   builder=builder, run_kw=run_kw)


def _worker_init():
    torch.set_num_threads(1)


def _worker(job):
    import importlib
    mod = importlib.import_module(job["screen"])
    t = time.time()
    rec = mod.run_job(job["arm"], job["seed"])
    rec["secs_wall"] = time.time() - t
    rec["lr"] = mod.ARMS[job["arm"]]["lr"]
    return job, rec


def store_path(screen_name):
    return os.path.join(OUT_DIR, f"{screen_name}_results.json")


def load_store(screen_name):
    p = store_path(screen_name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"meta": {}, "runs": {}}


def save_store(screen_name, store):
    os.makedirs(OUT_DIR, exist_ok=True)
    p = store_path(screen_name)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(store, f, indent=1, sort_keys=True)
    os.replace(tmp, p)


def run_jobs(screens, workers=WORKERS, pv=None, only=None):
    """Every (arm, seed) of every screen that is not cached, in one pool, longest-first
    (arms whose runs are expected to go the distance first). Records persist per screen."""
    stores = {s.NAME: load_store(s.NAME) for s in screens}
    for s in screens:
        stores[s.NAME]["meta"].setdefault("provenance", pv or provenance())
        stores[s.NAME]["meta"]["arms"] = {k: {kk: vv for kk, vv in a.items() if kk != "seeds"}
                                          | {"seeds": list(a["seeds"])}
                                          for k, a in s.ARMS.items()}
        save_store(s.NAME, stores[s.NAME])
    jobs = []
    for s in screens:
        for k, a in s.ARMS.items():
            if only is not None and k not in only:
                continue
            for seed in a["seeds"]:
                if f"{k}|{seed}" not in stores[s.NAME]["runs"]:
                    jobs.append(dict(screen=s.__name__, name=s.NAME, arm=k, seed=seed,
                                     prio=a.get("prio", 1)))
    jobs.sort(key=lambda j: (-j["prio"], j["seed"]))
    print(f"  {len(jobs)} runs queued on {workers} workers")
    if not jobs:
        return stores
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                             initializer=_worker_init) as ex:
        futs = [ex.submit(_worker, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            job, rec = f.result()
            st = stores[job["name"]]
            st["runs"][f"{job['arm']}|{job['seed']}"] = rec
            save_store(job["name"], st)
            print(f"  [{n:>3}/{len(jobs)}] {(time.time() - t0) / 60:6.1f} min  {job['name']:<10} "
                  f"{job['arm']:<14} seed {job['seed']}  {line(rec)}", flush=True)
    return stores


# ── Reporting ────────────────────────────────────────────────────────────────
def bound(r):
    return bool(r and r.get("ok") and r["transition"] is not None)


def discovered(r):
    return bool(bound(r) and r["val_cos"] < DISC_COS)


def routed(r):
    return bool(r and r.get("ok") and r["end"].get("margin", 0.0) >= ROUTED_MARGIN)


def tag(r):
    if r is None:
        return "not run"
    if not r.get("ok"):
        return "FAILED"
    if r.get("k", 2) < 2:
        return "BOUND" if bound(r) else "unbound"
    if bound(r):
        return "DISCOVERED" if discovered(r) else ("BOUND+routed" if routed(r) else "BOUND unrouted")
    return fail_class(r)


def line(r):
    if not r.get("ok"):
        return f"FAILED {r.get('error')}"
    e = r["end"]
    m = e.get("margin")
    return (f"acc {r['acc']:.3f}  trans {str(r['transition']):>5}  VALcos {r['val_cos']:.3f}  "
            f"margin {'--' if m is None else f'{m:.2f}'}  {tag(r)}")


def eta_str(r):
    e = r["end"]
    if "eta_key_by_stream" not in e:
        return "--"
    return "/".join(f"{e['eta_key_by_' + g]:.2f}" for g in ("stream", "key", "half", "index"))


def per_seed_table(screen, store, extra_recorded=("A", "A_conv")):
    rec = {k: recorded(k) for k in extra_recorded}
    print(f"  per seed (recorded columns are X's test_short_conv runs at lr {SUB_LR:g}):")
    for k, a in screen.ARMS.items():
        print(f"   arm {k} (lr {a['lr']:g})")
        print(f"    {'seed':>4} {'acc':>6} {'trans':>6} {'VALcos':>7} {'margin':>6} "
              f"{'eta key s/k/h/i':>20}  {'outcome':<16} " +
              " ".join(f"{'X ' + x:<16}" for x in extra_recorded))
        for s in a["seeds"]:
            r = store["runs"].get(f"{k}|{s}")
            recs = " ".join(f"{tag(rec[x].get(s)):<16}" for x in extra_recorded)
            if r is None or not r.get("ok"):
                print(f"    {s:>4} {tag(r)}")
                continue
            m = r["end"].get("margin")
            print(f"    {s:>4} {r['acc']:6.3f} {str(r['transition']):>6} {r['val_cos']:7.3f} "
                  f"{'--' if m is None else f'{m:6.2f}':>6} {eta_str(r):>20}  {tag(r):<16} {recs}")


def paired_vs_recorded(store, arm, seeds, field_fn, rec_arm="A", label="bound"):
    """Counts and exact McNemar (two-sided) against X's recorded rec_arm on the same seeds."""
    rec = recorded(rec_arm)
    done = [s for s in seeds if (r := store["runs"].get(f"{arm}|{s}")) and r.get("ok")]
    new = {s: field_fn(store["runs"][f"{arm}|{s}"]) for s in done}
    old = {s: field_fn(rec[s]) for s in done}
    b = sum(1 for s in done if new[s] and not old[s])
    c = sum(1 for s in done if old[s] and not new[s])
    p = mcnemar_exact(b, c)
    print(f"    {label:<11} {arm:<14} {sum(new.values()):>2}/{len(done)}   X {rec_arm} "
          f"{sum(old.values()):>2}/{len(done)}   {arm} only {b}, {rec_arm} only {c}, "
          f"McNemar two-sided p = {p:.3g}")
    return dict(n=len(done), new=sum(new.values()), old=sum(old.values()), b=b, c=c, p=p)


def fail_counts(store, arm, seeds):
    out = {}
    for s in seeds:
        r = store["runs"].get(f"{arm}|{s}")
        if r and r.get("ok") and not bound(r) and r.get("k", 2) >= 2:
            out[fail_class(r)] = out.get(fail_class(r), 0) + 1
    return out


def recorded_fail_counts(arm, seeds):
    rec = recorded(arm)
    out = {}
    for s in seeds:
        r = rec[s]
        if not bound(r):
            out[fail_class(r)] = out.get(fail_class(r), 0) + 1
    return out
