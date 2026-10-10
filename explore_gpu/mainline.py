#!/usr/bin/env python
"""
explore_gpu/mainline.py — run any main-line test's arm on a device, with the test's own code. EXPLORATORY tooling.

    python -m explore_gpu.mainline --test test_window_gate --arm WIN3_SPLIT_D8 --seeds 540-542 --device cuda
    python -m explore_gpu.mainline --test test_short_conv --arm A --seeds 160 --iters 2400 --device cpu    (a CHECK)
    python -m explore_gpu.mainline --list test_split_copy                                                    (its arms)

HOW: every main-line test that trains exposes run_job(sp) and spec(key, seed, iters | sched); its runs end in
test_binding_onset.onset_run. Inside explore_gpu.device.device_scope(device) the test's own run_job runs unchanged: its
model is built on CPU by its own builder under its own seeding (so initial parameters do not depend on the device), then
moved; the forward runs on the device and hands its outputs back to the test's CPU code (statistics, hinges, triggers,
records); the split's noise is drawn from its CPU generator and moved (test_window_gate.split_op, explore_i_common.split_op);
float32 with TF32 off. spec's third argument is the test's own default: iters = the test's MAX_ITERS (or ITERS / MAX_C /
the arm's part length) unless --iters is given; sched = the test's REAL schedule, with --iters setting its total where
the schedule has one ("total", "iters", "iters_a/b/c", "iters0").
RECORDS: in the test's own format (run_job's record), keyed "ARM|seed", plus rec["device"] (explore_gpu.device.device_info:
device, GPU model, capability, memory, CUDA, cuDNN, torch, CPU) and rec["gpu"]. Written atomically after every run to
explore_out/I/gpu/mainline/<test>_<hardware>_results.json — never into results/X or results/L, and the meta names the
machine "GPU:<model>" so that no test's --also pooling can take these records for machine X's or L's. A record from a
CUDA run is a new run of that arm and seed (CUDA does not reproduce CPU records bit for bit: explore_gpu/checks.py, K0);
compare outcomes, not curves.
WORKERS: --workers N runs N processes (spawn), each with its own CUDA context, sharing the GPU (the main-line models at
L <= 99 are small: several runs per GPU keep it busy).
"""

import argparse
import importlib
import inspect
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

OUT = os.path.join(HERE, "explore_out", "I", "gpu", "mainline")


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _default_iters(mod, arm):
    a = getattr(mod, "ARM", {}).get(arm, {})
    part = a.get("part") if isinstance(a, dict) else None
    for name in ("ITERS", "MAX_ITERS"):
        if hasattr(mod, name):
            return int(getattr(mod, name))
    if part is not None:
        for name in (f"MAX_{part}", "MAX_C"):
            if hasattr(mod, name):
                return int(getattr(mod, name))
    raise ValueError(f"{mod.__name__}: no default iteration count; pass --iters")


def _sched(mod, iters):
    base = getattr(mod, "REAL", None) or getattr(mod, "SCHED", None)
    assert isinstance(base, dict), f"{mod.__name__}: no REAL / SCHED schedule"
    s = dict(base)
    if iters is not None:
        for k in ("total", "iters", "iters_a", "iters_b", "iters_c", "iters0"):
            if k in s:
                s[k] = int(iters)
    return s


def make_spec(mod, arm, seed, iters):
    sig = inspect.signature(mod.spec)
    names = list(sig.parameters)
    third = names[2] if len(names) > 2 else None
    if third == "iters":
        return mod.spec(arm, seed, int(iters) if iters is not None else _default_iters(mod, arm))
    if third == "sched":
        return mod.spec(arm, seed, _sched(mod, iters))
    raise ValueError(f"{mod.__name__}.spec{sig}: unsupported signature")


def run_arm(test, arm, seed, device, iters=None):
    """One run of a main-line test's arm on `device`, through the test's own run_job. Returns the record."""
    import torch
    from explore_gpu import device as dv
    if torch.device(device).type == "cpu":
        torch.set_num_threads(1)
    mod = importlib.import_module(test)
    assert hasattr(mod, "run_job") and hasattr(mod, "spec"), f"{test} has no run_job / spec"
    sp = make_spec(mod, arm, seed, iters)
    t0 = time.time()
    try:
        with dv.device_scope(device) as dev:
            rec = mod.run_job(sp)
            info = dv.device_info(dev)
    except Exception as e:
        info = dv.device_info(device)
        rec = dict(ok=False, seed=seed, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
    rec = dict(rec, device=info, gpu=info.get("gpu"), wall_secs=time.time() - t0, spec=json.loads(json.dumps(sp, default=str)))
    return json.loads(json.dumps(rec, default=float))


def _job(args):
    test, arm, seed, device, iters = args
    return run_arm(test, arm, seed, device, iters)


def store_path(test, device):
    from explore_gpu import device as dv
    hw = "".join(c if c.isalnum() else "_" for c in dv.hw_label(device))
    return os.path.join(OUT, f"{test}_{hw}_results.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test")
    ap.add_argument("--arm")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--results", default=None)
    ap.add_argument("--list", default=None, help="print a test's arms and default seeds")
    a = ap.parse_args()
    if a.list:
        mod = importlib.import_module(a.list)
        seeds = getattr(mod, "SEEDS", {})
        for k in getattr(mod, "ARM", {}):
            print(k, (seeds.get(k) if isinstance(seeds, dict) else ""))
        return
    from explore_gpu import device as dv
    dev = dv.setup(a.device)
    path = a.results or store_path(a.test, dev)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    st = json.load(open(path)) if os.path.exists(path) else {"meta": {}, "runs": {}}
    info = dv.device_info(dev)
    st["meta"].setdefault("started", time.strftime("%Y-%m-%d %H:%M:%S"))
    st["meta"].update(test=a.test, machine=f"GPU:{info.get('gpu') or info['cpu']}", device=info,
                      note="explore_gpu.mainline: the test's own run_job on this device; not machine X's or L's records")
    seeds = [s for s in parse_seeds(a.seeds) if not (st["runs"].get(f"{a.arm}|{s}") or {}).get("ok")]
    print(f"[mainline] {a.test} {a.arm} seeds {seeds} on {info} -> {path}", flush=True)

    def save():
        with open(path + ".tmp", "w") as f:
            json.dump(st, f, indent=1, sort_keys=True)
        os.replace(path + ".tmp", path)
    save()
    jobs = [(a.test, a.arm, s, a.device, a.iters) for s in seeds]
    if a.workers <= 1:
        results = map(_job, jobs)
    else:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(a.workers)
        results = pool.imap_unordered(_job, jobs)
    for rec in results:
        st["runs"][f"{a.arm}|{rec['seed']}"] = rec
        save()
        c = rec.get("curve") or []
        print(f"  {a.arm} {rec['seed']}: ok {rec.get('ok')} transition {rec.get('transition')} last "
              f"{c[-1] if c else None} {rec.get('error', '')} ({rec['wall_secs'] / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
