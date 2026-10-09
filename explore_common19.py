#!/usr/bin/env python
"""
explore_common19.py — EXPLORATORY, not a result. Batch 19's parent-side harness: explore_common16's keys, stores, CPU rules
and code SHA (imported read-only), with a pool for checkpointed runs and dependent arms.

POOL (run_pool19): every run is one child process (explore_common16's child template, the main-line hook, 1 torch thread)
running explore_scale_child.run with a checkpoint path (explore_b19_model: a checkpoint after every evaluation, a resume
from it). A segment keeps every worker busy until its budget; then it stops its running children (SIGTERM, then SIGKILL
after 10 s) and prints each one's last checkpoint step; the next segment resumes them first. A run with ok = False is
stored with its error, its checkpoint deleted, and re-queued (at most 3 attempts, explore_common16's rule). Jobs are
taken: runs with a checkpoint first, then by screen order (S64, S65, S66, S67), then oracles before the reference before
the other arms before diagnostics, then the longest projected first.

DEPENDENT ARMS (the user's rules, fixed before any run; arm_state):
  - an arm with an oracle ("oracle": the oracle arm's key) is READY once any oracle run has BOUND (transition not None:
    held-out accuracy >= 0.95 held to the end), NEVER (UNTESTED: "the oracle did not bind within budget") once every
    oracle seed has a run and none bound, else WAITING;
  - an arm with "needs" asks its screen (S65's RES6: the 6-layer arm's failure);
  - phase-2 screens (S66, S67) have no arms until the driver stores the phase-1 decision.
  A dry run marks every arm READY (it exercises the code; nothing in it is for interpretation).
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common16 as c16

CHILD = "explore_scale_child"
ROLE_RANK = {"oracle": 0, "ref": 1, "arm": 2, "diag": 3}
POLL_S = 2.0
MIN_START_S = 60


def bound(r):
    return bool(r is not None and r.get("ok") and r.get("transition") is not None)


def br(r):
    return bool(r is not None and r.get("outcome") == "BOUND ROUTED")


def store_runs(screen, arm, store=None):
    return c16.runs_of(screen, arm, store=store)


def exhausted(store, arm, seed, sha, cpu):
    r = store["runs"].get(c16.key(arm, seed, sha, cpu))
    return r is not None and not r.get("ok") and r.get("attempts", 1) >= c16.MAX_ATTEMPTS


def complete(screen, arm, store=None):
    st = ec.load_store(screen.NAME) if store is None else store
    a = screen.ARMS[arm]
    runs = store_runs(screen, arm, st)
    sha = c16.STATE["shas"][screen.NAME]
    cpu = c16.STATE["batch_cpu"]
    return all(s in runs or exhausted(st, arm, s, sha, cpu) for s in a["seeds"])


def oracle_state(screen, oarm, store=None):
    """'valid' (some oracle run bound), 'invalid' (every seed done, none bound), 'pending'."""
    st = ec.load_store(screen.NAME) if store is None else store
    runs = store_runs(screen, oarm, st)
    if any(bound(r) for r in runs.values()):
        return "valid"
    return "invalid" if complete(screen, oarm, st) else "pending"


def arm_state(screen, arm, store, dry):
    """('ready' | 'wait' | 'never', reason)."""
    a = screen.ARMS[arm]
    if dry:
        return "ready", "dry run"
    o = a.get("oracle")
    if o:
        s = oracle_state(screen, o, store)
        if s == "invalid":
            return "never", f"UNTESTED: the oracle {o} did not bind within budget on {len(screen.ARMS[o]['seeds'])} seeds"
        if s == "pending":
            return "wait", f"waiting for the oracle {o}"
    nd = a.get("needs")
    if nd:
        return getattr(screen, nd)(store)
    return "ready", ""


def ckpt_path(ckpt_dir, name, arm, seed, sha):
    return os.path.join(ckpt_dir, f"{name}__{arm}__{seed}__{sha[:12]}.pt")


def ckpt_step(path):
    try:
        with open(path + ".step") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def ready_jobs(screens, running_keys, cpu, dry, ckpt_dir, dur):
    jobs, blocked = [], []
    for i, m in enumerate(screens):
        st = ec.load_store(m.NAME)
        sha = c16.STATE["shas"][m.NAME]
        for arm, a in m.ARMS.items():
            state, why = arm_state(m, arm, st, dry)
            if state != "ready":
                if state == "wait":
                    blocked.append(f"{m.NAME}:{arm} ({why})")
                continue
            for s in a["seeds"]:
                if (m.NAME, arm, s) in running_keys or c16.pick(st, arm, s, sha, a["opt"]) is not None:
                    continue
                if exhausted(st, arm, s, sha, cpu):
                    continue
                ck = ckpt_path(ckpt_dir, m.NAME, arm, s, sha)
                j = dict(screen=m.__name__, name=m.NAME, arm=arm, seed=s, sha=sha, cpu=cpu, dry=dry, opt=a["opt"],
                         spec=a["spec"], iters=a["iters"], role=a.get("role", "arm"), ckpt=ck, order=i,
                         has_ckpt=os.path.exists(ck))
                j["dur"] = dur(j)
                jobs.append(j)
    jobs.sort(key=lambda j: (not j["has_ckpt"], j["order"], ROLE_RANK.get(j["role"], 9), -j["dur"]))
    return jobs, blocked


def _start(j):
    td = tempfile.mkdtemp(prefix="b19job_")
    inp, outp = os.path.join(td, "in.json"), os.path.join(td, "out.json")
    with open(inp, "w") as f:
        json.dump(dict(spec=j["spec"], seed=j["seed"], iters=j["iters"], ckpt=j["ckpt"]), f)
    code = c16._CHILD.format(here=HERE, main=True, module=CHILD, func="run", inp=inp, outp=outp)
    p = subprocess.Popen([sys.executable, "-c", code], cwd=HERE, env=c16._env())
    return dict(proc=p, td=td, outp=outp, t=time.time(), job=j, step0=ckpt_step(j["ckpt"]))


def _finish(run, arms_of):
    import torch
    j = run["job"]
    rc = run["proc"].returncode
    rec = None
    if os.path.exists(run["outp"]):
        try:
            with open(run["outp"]) as f:
                rec = json.load(f)
        except (OSError, ValueError) as e:
            rec = dict(ok=False, error=f"unreadable child output: {e}")
    if rec is None:
        rec = dict(ok=False, error=f"child exited {rc} without a result")
    a = arms_of(j)
    rec.update(arm=j["arm"], seed=j["seed"], code_sha=j["sha"], cpu=j["cpu"], git=ec.git_head(), torch=torch.__version__,
               lr=a["lr"], muon_lr=None, sched=a.get("sched"), secs_wall=time.time() - run["t"], dry=j["dry"],
               resumed_from=run["step0"])
    if not rec.get("ok"):
        for f in (j["ckpt"], j["ckpt"] + ".tmp", j["ckpt"] + ".step"):
            if os.path.exists(f):
                os.remove(f)
    return rec


def run_pool19(screens_fn, workers, budget_s, line, ckpt_dir, cpu, dry, dur, advance=None):
    """screens_fn() -> the screens with arms now (phase-2 screens appear after the decision); advance() is called whenever
    the pool is idle or a run has finished (the phase decision). Returns the number of runs saved."""
    os.makedirs(ckpt_dir, exist_ok=True)
    t0 = time.time()
    running = {}
    done, dirty, quiet = 0, True, False

    def arms_of(j):
        return {m.NAME: m for m in screens_fn()}[j["name"]].ARMS[j["arm"]]
    while True:
        left = None if budget_s is None else budget_s - (time.time() - t0)
        if left is not None and left <= 0:
            break
        can_start = left is None or left > MIN_START_S
        if not running and not can_start:
            break
        if dirty and len(running) < workers and can_start:
            if advance is not None:
                advance()
            jobs, blocked = ready_jobs(screens_fn(), set(running), cpu, dry, ckpt_dir, dur)
            for j in jobs[:workers - len(running)]:
                r = _start(j)
                running[(j["name"], j["arm"], j["seed"])] = r
                print(f"  start {(time.time() - t0) / 60:6.1f} min  {j['name']:<12} {j['arm']:<14} seed {j['seed']}"
                      + (f"  (resuming from its checkpoint at step {r['step0']})" if r["step0"] else "")
                      + f"  (projected worst case {j['dur'] / 60:.0f} min)", flush=True)
            dirty = False
            if not running:
                if blocked:
                    print(f"  nothing ready and nothing running; still waiting: {blocked}", flush=True)
                break
        time.sleep(POLL_S)
        for k in list(running):
            r = running[k]
            if r["proc"].poll() is None:
                continue
            running.pop(k)
            rec = _finish(r, arms_of)
            c16.save_run(r["job"]["name"], rec)
            done += 1
            dirty = True
            j = r["job"]
            tag = line(j, rec) if (line is not None and rec.get("ok")) else ""
            print(f"  [{done:>3}] {(time.time() - t0) / 60:6.1f} min  {j['name']:<12} {j['arm']:<14} seed {j['seed']}  lr {rec['lr']:g}  "
                  f"({rec.get('sched')})\n          on {rec['cpu']}, git {rec['git']}, code {rec['code_sha'][:12]}, attempt "
                  f"{rec['attempts']}" + (f", resumed at {rec.get('resumed')}" if rec.get("resumed") else "") + ": "
                  + (tag if rec.get("ok") else f"FAILED {rec.get('error')}"), flush=True)
            if not rec.get("ok") and rec["attempts"] < c16.MAX_ATTEMPTS:
                print(f"          re-queued from scratch (attempt {rec['attempts'] + 1} of at most {c16.MAX_ATTEMPTS})", flush=True)
    if running:
        for r in running.values():
            r["proc"].send_signal(signal.SIGTERM)
        t1 = time.time()
        while any(r["proc"].poll() is None for r in running.values()) and time.time() - t1 < 10:
            time.sleep(0.5)
        for r in running.values():
            if r["proc"].poll() is None:
                r["proc"].kill()
                r["proc"].wait()
        print(f"  segment budget reached: {len(running)} run(s) stopped, each to resume from its last checkpoint: "
              + ", ".join(f"{k[0]}:{k[1]}|{k[2]} at step {ckpt_step(r['job']['ckpt'])}" for k, r in running.items()), flush=True)
    return done


# ── Report helpers ───────────────────────────────────────────────────────────
def med(xs):
    return c16.med(xs)


def per_channel(m):
    return c16.chmap_str(m)


def counts_vs(new, old, seeds):
    """BOUND ROUTED counts paired over the seeds both have: n, new, old, b (new only), c (old only), p, d = new - old."""
    d = c16.mcnemar({s: br(r) for s, r in new.items()}, {s: br(r) for s, r in old.items()}, seeds)
    d["d"] = d["new"] - d["old"]
    return d


def classes(runs):
    from collections import Counter
    return dict(Counter(r.get("outcome") for r in runs.values()))


def trans_med(runs, only_br=True):
    xs = [r["transition"] for r in runs.values() if r.get("transition") is not None and (br(r) or not only_br)]
    return med(xs), sorted(xs)


def params_of(runs):
    ps = [r.get("params") for r in runs.values() if r.get("params")]
    return ps[0] if ps else None


def init_line(init):
    if not init:
        return "--"
    keys = ("embed.weight", "encoder", "encoder_v", "decoder", "lm_head", "W_in", "W_g", "gate_conv.conv_w", "A")
    return ", ".join(f"{k} {tuple(init[k]['shape'])} std {init[k]['std']:.4f}" for k in keys if k in init)


def arm_table(arm, runs, seeds, pair=None, plab=None):
    print(f"    {'seed':>4} | {'outcome':<24} {'trans':>5} {'stop':>5} {'acc':>5} {'end map':>16} | "
          + (f"{plab}" if plab else ""))
    for s in seeds:
        r = runs.get(s)
        if r is None:
            print(f"    {s:>4} | not run")
            continue
        q = (pair or {}).get(s)
        print(f"    {s:>4} | {str(r.get('outcome')):<24} {str(r.get('transition')):>5} {str(r.get('stopped_at')):>5} "
              f"{r.get('acc', float('nan')):5.2f} {per_channel((r.get('end') or {}).get('ch_map')):>16} | "
              + (f"{q.get('outcome')} (trans {q.get('transition')})" if q else ("--" if plab else ""))
              + (f"  [resumed at {r['resumed']}]" if r.get("resumed") else ""))
