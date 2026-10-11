#!/usr/bin/env python
"""
longrun.py — long runs in a container: a detached process, a watcher that resumes it after a
container restart, and the helpers a test uses to resume from its cached records and to print
its runtime projection before the first run.

    python3 longrun.py start split_copy_X --results split_copy_results.json --total 140 -- \\
        python3 -u test_split_copy.py --machine X --workers 4
    python3 longrun.py watch split_copy_X --minutes 9 --resume     # poll; relaunch if stopped
    python3 longrun.py status split_copy_X

- start: launches the command in its own session (setsid; stdin /dev/null; nohup), so neither a
  closed terminal nor a restart of the agent's worker touches it. Output is appended to
  longrun/<name>.log; longrun/<name>.json keeps the command, each start's pid, boot id, git and
  time. A shell wrapper writes the command's exit code to longrun/<name>.exit when it ends.
- watch: prints the log lines written since the last watch, the record count of the results
  file and a rate-based estimate, for at most --minutes (keep it under the shell's 10-minute
  call limit), and returns as soon as the run ends. States:
    RUNNING   the start's process is alive (same boot id, same pid, its command line);
    FINISHED  exit code 0;
    FAILED    a non-zero exit code: not relaunched (read the log);
    STOPPED   no exit code and no process: the container restarted (new boot id) or the process
              was killed. With --resume the same command is started again, and the test
              resumes from its cached records (at most --max-resumes times per run name).
  Exit status: 0 FINISHED, 1 FAILED, 2 STOPPED (not resumed), 3 still RUNNING.
- status: one watch poll without waiting or printing the log.

A test resumes by skipping every run already in its results file (the file is written
atomically after each run), keeping the first start's git and time in its meta and adding each
later start to meta["starts"] (resume_meta), and printing its projection, of all runs and of the
runs still to do, before it trains (projection). test_window_gate's Part D resume is the model.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "longrun")
POLL = 30                                  # seconds between polls
MAX_LINES = 300                            # log lines printed per watch (head and tail kept)


# ── Helpers for tests ────────────────────────────────────────────────────────
def makespan(durations, workers):
    """Longest-first packing onto the workers (test_binding_recipe.makespan)."""
    loads = [0.0] * workers
    for d in sorted(durations, reverse=True):
        i = loads.index(min(loads))
        loads[i] += d
    return max(loads)


def projection(rows, workers, done=frozenset()):
    """rows: (key, seconds per step, steps, seeds) per arm; done: the (key, seed) pairs cached.
    Prints the per-arm cost and the makespan of all runs and of the runs still to do; returns
    (hours for all, hours for the rest)."""
    every, rest = [], []
    for key, sps, steps, seeds in rows:
        full = sps * steps
        print(f"  {key:<16} {sps * 1000:7.1f} ms/step x {steps} steps x {len(seeds)} seeds   (a run {full / 60:.1f} min)")
        for s in seeds:
            every.append(full)
            if (key, s) not in done:
                rest.append(full)
    h_all, h_rest = makespan(every, workers) / 3600, makespan(rest, workers) / 3600
    print(f"  {len(every)} runs: serial {sum(every) / 3600:.2f} h, {h_all:.2f} h on {workers} workers"
          + (f"; {len(rest)} still to run: {h_rest:.2f} h" if len(rest) != len(every) else ""))
    return h_all, h_rest


def resume_meta(old, new, head, workers, keys, force=False):
    """The results file's meta for this start. A first start (or --force) is `new`. A later one
    keeps the first start's git and time and appends {git, started, workers, arms_added}."""
    if not old.get("started") or force:
        return dict(new)
    added = [k for k in keys if k not in (old.get("seeds") or {})]
    meta = dict(new)
    meta.update(git=old.get("git"), started=old["started"],
                starts=list(old.get("starts", [])) + [dict(git=head, started=new["started"], workers=workers,
                                                            arms_added=added)])
    print(f"  resuming: first started {old['started']} at {old.get('git')}; this start {new['started']} at {head}; "
          f"arms added {added or 'none'}")
    return meta


# ── The detached run and its watcher ─────────────────────────────────────────
def boot_id():
    try:
        return open("/proc/sys/kernel/random/boot_id").read().strip()
    except OSError:
        return "unknown"


def paths(name):
    return (os.path.join(DIR, f"{name}.json"), os.path.join(DIR, f"{name}.log"), os.path.join(DIR, f"{name}.exit"))


def load_state(name):
    sp = paths(name)[0]
    if not os.path.exists(sp):
        sys.exit(f"no run named {name} ({sp})")
    return json.load(open(sp))


def save_state(name, st):
    sp = paths(name)[0]
    tmp = sp + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, sp)


def git_head():
    return subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True).stdout.strip()


def launch(name, st):
    _, log, ex = paths(name)
    if os.path.exists(ex):
        os.remove(ex)
    n = len(st["starts"]) + 1
    with open(log, "a") as f:
        f.write(f"\n=== longrun {name}: start {n} at {time.strftime('%Y-%m-%d %H:%M:%S')}, git {git_head()}, "
                f"boot {boot_id()[:8]} ===\n")
    wrapped = f"{shlex.join(st['cmd'])}; echo $? > {shlex.quote(ex)}"
    with open(log, "a") as out:
        p = subprocess.Popen(["nohup", "sh", "-c", wrapped], cwd=st["cwd"], stdin=subprocess.DEVNULL, stdout=out,
                             stderr=subprocess.STDOUT, start_new_session=True)
    st["starts"].append(dict(pid=p.pid, boot=boot_id(), started=time.strftime("%Y-%m-%d %H:%M:%S"), t=time.time(),
                             git=git_head(), records_at_start=records(st)))
    save_state(name, st)
    print(f"started {name} (start {n}): pid {p.pid}, log {log}")


def alive(st):
    s = st["starts"][-1]
    if s["boot"] != boot_id():
        return False
    try:
        cmd = open(f"/proc/{s['pid']}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return paths(st["name"])[2] in cmd


def state(st):
    ex = paths(st["name"])[2]
    if alive(st):
        return "RUNNING", None
    if os.path.exists(ex):
        code = open(ex).read().strip()
        return ("FINISHED" if code == "0" else "FAILED"), code
    return "STOPPED", None


def records(st):
    if not st.get("results"):
        return None
    try:
        return len(json.load(open(os.path.join(st["cwd"], st["results"]))).get("runs", {}))
    except (OSError, ValueError):
        return None


def progress(st):
    n = records(st)
    s = st["starts"][-1]
    el = time.time() - s["t"]
    line = f"elapsed {el / 3600:.2f} h this start ({len(st['starts'])} start(s))"
    if n is not None:
        line += f"; records {n}" + (f"/{st['total']}" if st.get("total") else "")
        k0 = s.get("records_at_start") or 0
        if n > k0 and "first_seen" not in s:
            s["first_seen"] = [time.time(), n]              # the rate is measured from the first record seen
        f = s.get("first_seen")
        if st.get("total") and f and n > f[1] and n < st["total"]:
            line += f"; at the rate since the first record ~{(time.time() - f[0]) / (n - f[1]) * (st['total'] - n) / 3600:.1f} h left"
    return line


def new_lines(st):
    log = paths(st["name"])[1]
    off = st.get("offset", 0)
    with open(log, "rb") as f:
        f.seek(off)
        data = f.read()
    st["offset"] = off + len(data)
    lines = data.decode(errors="replace").splitlines()
    if len(lines) > MAX_LINES:
        k = len(lines) - MAX_LINES
        lines = lines[:50] + [f"... ({k} lines not shown; see {log}) ..."] + lines[-(MAX_LINES - 50):]
    return lines


def cmd_start(a):
    os.makedirs(DIR, exist_ok=True)
    sp = paths(a.name)[0]
    if os.path.exists(sp):
        st = json.load(open(sp))
        if state(st)[0] == "RUNNING":
            sys.exit(f"{a.name} is running (pid {st['starts'][-1]['pid']})")
        if not a.again:
            sys.exit(f"{a.name} exists ({sp}); use watch --resume to resume it, or start --again")
    cmd = a.cmd
    if not cmd:
        sys.exit("no command (give it after --)")
    st = dict(name=a.name, cmd=cmd, cwd=os.getcwd(), results=a.results, total=a.total, starts=[], offset=0,
              resumes=0, max_resumes=a.max_resumes)
    if os.path.exists(paths(a.name)[1]):
        st["offset"] = os.path.getsize(paths(a.name)[1])
    launch(a.name, st)


def cmd_watch(a, wait=True):
    st = load_state(a.name)
    t_end = time.time() + 60 * getattr(a, "minutes", 0)
    while True:
        s, code = state(st)
        if wait:
            for ln in new_lines(st):
                print(ln)
            save_state(a.name, st)
        if s != "RUNNING" or not wait or time.time() >= t_end:
            break
        time.sleep(min(POLL, max(1, t_end - time.time())))
    print(f"[longrun {a.name}] {s}" + (f" (exit {code})" if code is not None else "") + f"; {progress(st)}")
    if s == "STOPPED":
        if wait and a.resume and st["resumes"] < st["max_resumes"]:
            st["resumes"] += 1
            print(f"[longrun {a.name}] no exit code and no process (boot {st['starts'][-1]['boot'][:8]} -> "
                  f"{boot_id()[:8]}): resuming from the cached records ({st['resumes']}/{st['max_resumes']})")
            launch(a.name, st)
            return 3
        return 2
    return {"FINISHED": 0, "FAILED": 1, "RUNNING": 3}[s]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="op", required=True)
    p = sub.add_parser("start")
    p.add_argument("name")
    p.add_argument("--results", default=None, help="the test's results file (for the record count)")
    p.add_argument("--total", type=int, default=None, help="the number of runs the results file will hold")
    p.add_argument("--max-resumes", type=int, default=5)
    p.add_argument("--again", action="store_true", help="start a finished or stopped run name afresh")
    p = sub.add_parser("watch")
    p.add_argument("name")
    p.add_argument("--minutes", type=float, default=9.0)
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("status")
    p.add_argument("name")
    argv = sys.argv[1:]
    cmd = argv[argv.index("--") + 1:] if "--" in argv else []
    a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    a.cmd = cmd
    if a.op == "start":
        cmd_start(a)
    elif a.op == "watch":
        sys.exit(cmd_watch(a))
    else:
        sys.exit(cmd_watch(a, wait=False))


if __name__ == "__main__":
    main()
