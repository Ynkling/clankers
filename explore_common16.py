#!/usr/bin/env python
"""
explore_common16.py — EXPLORATORY, not a result. Batch 16's additions to the outside-ideas harness (parent side).
explore_common, explore_common2 and explore_main9c are imported, not changed.

What this adds (the audit of 6 October 2026, section 2.3, and batch 16's specification):
  - CODE SHA. A screen's code SHA is the SHA-1 of the sources of every module of this repository that its child
    module imports (computed in a fresh child of the same kind as the runs: with explore_main9c's hook for the
    main-line screens, whose served modules are hashed from their git blobs at MAIN_SHA; without it for S43). A
    change to any code a run executes changes the SHA; a change to report-only code does not.
  - CACHE KEY. A run is stored and reused under "arm|seed|code SHA (12 hex)|CPU model". A stored run is reused only
    when its code SHA is the screen's current one, and (Muon arms) when it ran on the batch's CPU; an Adam run made
    on another CPU that passed the repro check is accepted (Adam runs are bit-identical across the CPUs the repro
    check has passed on). The batch's CPU is fixed at the full CHECK pass and printed in every segment.
  - RETRY. A run with ok = False (or a child that crashed) is stored with its error and re-queued, up to 3 attempts.
  - CHILDREN. run_child(module, func, payload, main): explore_main9c.run_child's mechanism, with or without the
    main-line hook, 1 torch thread.
  - POOL. One process pool (4 workers, 1 thread each). Jobs are taken longest-first by their projected worst-case
    duration; with a segment budget, a freed worker takes the longest job that fits in the time left, else the
    shortest (a run killed by the session's time limit is lost, so long runs start early in a segment).
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec

MAX_ATTEMPTS = 3
STATE = dict(batch_cpu=None, shas={}, other_cpus=[])     # set by the driver


# ── Children ─────────────────────────────────────────────────────────────────
_CHILD = """
import json, sys
sys.path.insert(0, {here!r})
import torch
torch.set_num_threads(1)
if {main!r}:
    import explore_main9c as mt
    mt.install()
import importlib
mod = importlib.import_module({module!r})
with open({inp!r}) as f:
    payload = json.load(f)
out = getattr(mod, {func!r})(payload)
with open({outp!r}, "w") as f:
    json.dump(out, f)
"""

_SHA_CHILD = """
import hashlib, importlib, json, os, sys
here = {here!r}
sys.path.insert(0, here)
import torch
torch.set_num_threads(1)
if {main!r}:
    import explore_main9c as mt
    mt.install()
importlib.import_module({module!r})
rows = []
for name, mod in sorted(sys.modules.items()):
    d = getattr(mod, "__dict__", None) or {{}}
    f = d.get("__file__")
    if not isinstance(f, str) or os.path.dirname(os.path.abspath(f)) != here:
        continue
    sha = d.get("__main_sha__")
    if sha:
        import explore_main9c as mt
        src = mt.main_blob(name + ".py", sha)
    else:
        with open(f) as fh:
            src = fh.read()
    rows.append([name, hashlib.sha1(src.encode()).hexdigest(), ("main@" + sha[:7]) if sha else "branch"])
with open({outp!r}, "w") as f:
    json.dump(rows, f)
"""


def _env():
    return dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")


def run_child(module, func, payload, main=True, timeout=None):
    with tempfile.TemporaryDirectory() as td:
        inp, outp = os.path.join(td, "in.json"), os.path.join(td, "out.json")
        with open(inp, "w") as f:
            json.dump(payload, f)
        code = _CHILD.format(here=HERE, main=bool(main), module=module, func=func, inp=inp, outp=outp)
        r = subprocess.run([sys.executable, "-c", code], cwd=HERE, env=_env(), timeout=timeout)
        if r.returncode != 0 or not os.path.exists(outp):
            raise RuntimeError(f"child {module}.{func} failed (exit {r.returncode})")
        with open(outp) as f:
            return json.load(f)


def code_sha(module, main=True):
    """(sha40, rows): the SHA-1 over [name, sha1(source), origin] of every repository module the child module imports."""
    with tempfile.TemporaryDirectory() as td:
        outp = os.path.join(td, "out.json")
        code = _SHA_CHILD.format(here=HERE, main=bool(main), module=module, outp=outp)
        r = subprocess.run([sys.executable, "-c", code], cwd=HERE, env=_env(), capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"code_sha({module}) failed: {r.stderr[-2000:]}")
        with open(outp) as f:
            rows = json.load(f)
    return hashlib.sha1(json.dumps(rows).encode()).hexdigest(), rows


# ── Keys and selection ───────────────────────────────────────────────────────
def cpu_label():
    return ec.cpu_model()


def key(arm, seed, sha, cpu):
    return f"{arm}|{seed}|{sha[:12]}|{cpu}"


def cpus_for(opt):
    """The CPUs whose runs an arm accepts, in order of preference: Muon arms the batch's CPU only; Adam arms the batch's
    CPU, then any other CPU this batch ran on (each passed the repro check)."""
    b = STATE["batch_cpu"]
    return [b] if opt == "muon" else [b] + [c for c in STATE["other_cpus"] if c != b]


def pick(store, arm, seed, sha, opt):
    for c in cpus_for(opt):
        r = store["runs"].get(key(arm, seed, sha, c))
        if r is not None and r.get("ok"):
            return r
    return None


def runs_of(screen, arm, seeds=None, store=None):
    """{seed: record} of an arm's usable runs (screen: a parent module with NAME and ARMS)."""
    st = ec.load_store(screen.NAME) if store is None else store
    a = screen.ARMS[arm]
    sha = STATE["shas"][screen.NAME]
    out = {}
    for s in (a["seeds"] if seeds is None else seeds):
        r = pick(st, arm, s, sha, a["opt"])
        if r is not None:
            out[s] = r
    return out


def completeness(screen, arm):
    a = screen.ARMS[arm]
    return len(runs_of(screen, arm)), len(a["seeds"])


def other_cpus_seen(screens):
    seen = set()
    for m in screens:
        for r in ec.load_store(m.NAME)["runs"].values():
            if r.get("cpu"):
                seen.add(r["cpu"])
    return sorted(seen)


# ── Pool ─────────────────────────────────────────────────────────────────────
def _job(j):
    import importlib
    import torch
    torch.set_num_threads(1)
    mod = importlib.import_module(j["screen"])
    a = mod.ARMS[j["arm"]]
    t = time.time()
    try:
        rec = mod.run_job(j["arm"], j["seed"])
    except Exception as e:                                              # noqa: BLE001
        rec = dict(ok=False, error=f"{type(e).__name__}: {e}")
    rec.update(arm=j["arm"], seed=j["seed"], code_sha=j["sha"], cpu=j["cpu"], git=ec.git_head(), torch=torch.__version__,
               lr=a["lr"], muon_lr=a.get("muon_lr"), sched=a.get("sched"), secs_wall=time.time() - t,
               dry=j.get("dry", False))
    return j, rec


def queue(screens, cpu, dry=False):
    """The jobs to run on this CPU: every (arm, seed) without a usable run whose attempts are not exhausted; Muon arms
    only on the batch's CPU."""
    jobs, skipped = [], []
    for m in screens:
        st = ec.load_store(m.NAME)
        sha = STATE["shas"][m.NAME]
        for arm, a in m.ARMS.items():
            for s in a["seeds"]:
                if pick(st, arm, s, sha, a["opt"]) is not None:
                    continue
                if a["opt"] == "muon" and cpu != STATE["batch_cpu"]:
                    skipped.append(f"{m.NAME}:{arm}|{s}")
                    continue
                r = st["runs"].get(key(arm, s, sha, cpu))
                if r is not None and r.get("attempts", 1) >= MAX_ATTEMPTS:
                    skipped.append(f"{m.NAME}:{arm}|{s} (failed {r['attempts']}x: {r.get('error')})")
                    continue
                jobs.append(dict(screen=m.__name__, name=m.NAME, arm=arm, seed=s, sha=sha, cpu=cpu, dry=dry,
                                 opt=a["opt"]))
    return jobs, skipped


def save_run(name, rec):
    st = ec.load_store(name)
    k = key(rec["arm"], rec["seed"], rec["code_sha"], rec["cpu"])
    old = st["runs"].get(k)
    n = (old.get("attempts", 1) + 1) if (old is not None and not old.get("ok")) else 1
    rec["attempts"] = n
    if old is not None and not old.get("ok"):
        rec["failures"] = list(old.get("failures", [])) + [old.get("error")]
    st["runs"][k] = rec
    ec.save_store(name, st)


def run_pool(jobs, workers, dur, budget_s=None, line=None):
    """jobs: dicts from queue(); dur(job) -> projected worst-case seconds; budget_s: seconds left in this segment
    (None: no budget). Failed runs are stored and re-queued once within the segment (and in later segments, up to
    MAX_ATTEMPTS)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    t0 = time.time()
    # Muon jobs first (they run only on the batch's CPU; Adam jobs can run on any CPU that passes the repro check),
    # then longest first
    todo = sorted(jobs, key=lambda j: (j.get("opt") != "muon", -dur(j)))
    retried = set()
    done = 0

    def take():
        if not todo:
            return None
        if budget_s is None:
            return todo.pop(0)
        left = budget_s - (time.time() - t0)
        for i, j in enumerate(todo):
            if dur(j) <= 0.97 * left:
                return todo.pop(i)
        return None                                  # nothing fits: start nothing more in this segment

    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork"), initializer=ec._worker_init) as ex:
        running = {}
        while todo or running:
            while todo and len(running) < workers:
                j = take()
                if j is None:
                    break
                running[ex.submit(_job, j)] = j
                print(f"  start {(time.time() - t0) / 60:6.1f} min  {j['name']:<16} {j['arm']:<24} seed {j['seed']}  "
                      f"(projected worst case {dur(j) / 60:.0f} min)", flush=True)
            if not running:
                print(f"  {len(todo)} runs left for the next segment: none fits in the {(budget_s - (time.time() - t0)) / 60:.0f} "
                      f"min left of this one (worst-case projections)", flush=True)
                break
            fin, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for f in fin:
                j = running.pop(f)
                j, rec = f.result()
                save_run(j["name"], rec)
                done += 1
                tagline = line(j, rec) if line is not None else ""
                print(f"  [{done:>3}] {(time.time() - t0) / 60:6.1f} min  {j['name']:<16} {j['arm']:<24} seed {j['seed']}  lr "
                      f"{rec['lr']:g}" + (f", Muon lr {rec['muon_lr']:g}" if rec.get("muon_lr") else "")
                      + f"  ({rec.get('sched')})\n          on {rec['cpu']}, git {rec['git']}, code {rec['code_sha'][:12]}, "
                      f"attempt {rec['attempts']}: " + (tagline if rec.get("ok") else f"FAILED {rec.get('error')}"),
                      flush=True)
                k = (j["name"], j["arm"], j["seed"])
                if not rec.get("ok") and k not in retried and rec["attempts"] < MAX_ATTEMPTS:
                    retried.add(k)
                    todo.append(j)
                    print(f"          re-queued (attempt {rec['attempts'] + 1} of at most {MAX_ATTEMPTS})", flush=True)
    return done


# ── Small report helpers ─────────────────────────────────────────────────────
def fmt2(x):
    return "--" if x is None else f"{x:.2f}"


def med(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def mcnemar(new, old, seeds):
    """Paired counts over seeds present in both: (n, new count, old count, b = new only, c = old only, p two-sided)."""
    from test_router_confirm import mcnemar_exact
    ss = [s for s in seeds if s in new and s in old]
    b = sum(1 for s in ss if new[s] and not old[s])
    c = sum(1 for s in ss if old[s] and not new[s])
    return dict(n=len(ss), new=sum(bool(new[s]) for s in ss), old=sum(bool(old[s]) for s in ss), b=b, c=c,
                p=mcnemar_exact(b, c))


def chmap_str(m):
    if not m:
        return "--"
    counts = sorted((m.count(c) for c in set(m)), reverse=True)
    return "+".join(str(c) for c in counts)
