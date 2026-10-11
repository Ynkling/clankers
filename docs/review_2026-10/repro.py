#!/usr/bin/env python3
"""repro.py — the referee's bit-for-bit reproduction of one recorded run per main-line test (Phase VI), at 1 thread.

    <venv with torch 2.14.0>/bin/python docs/review_2026-10/repro.py window_gate WIN3_SLOW 500 X
    <venv>/bin/python docs/review_2026-10/repro.py muon_recipe WIN8_A 460 X

It imports the test module (nothing is modified), runs the arm's own run_job on the given seed with the test's REAL
schedule, and compares the new record with the machine's recorded one in results/<machine>/<test>_results.json:
the curve, the transition and every scalar and list the two records share. Output: one JSON line per run appended to
docs/review_2026-10/repro_out.jsonl (CPU, torch, threads, equal fields, differing fields). Nothing else is written.
"""
import json
import math
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import torch  # noqa: E402

torch.set_num_threads(1)


def cpu():
    for line in open("/proc/cpuinfo"):
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def flat(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flat(v, f"{p}/{k}")
    elif isinstance(x, list) and x and all(isinstance(v, (int, float, type(None))) for v in x):
        yield p, x
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flat(v, f"{p}[{i}]")
    else:
        yield p, x


def main():
    test, arm, seed, machine = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    mod = __import__(f"test_{test}")
    rec_old = json.load(open(f"results/{machine}/{test}_results.json"))["runs"][f"{arm}|{seed}"]
    t0 = time.time()
    rec = mod.run_job(mod.spec(arm, seed, mod.REAL))
    secs = time.time() - t0
    rec = json.loads(json.dumps(rec))          # the same JSON round trip the store applies
    old, new = dict(flat(rec_old)), dict(flat(rec))
    skip = ("/secs", "/cpu", "/secs_wall")
    shared = [k for k in old if k in new and not k.endswith(skip)]
    differ = [k for k in shared if old[k] != new[k]]
    out = dict(test=test, arm=arm, seed=seed, machine=machine, recorded_cpu=rec_old.get("cpu", "not in record"),
               this_cpu=cpu(), torch=torch.__version__, threads=torch.get_num_threads(),
               git=subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip(),
               secs=round(secs, 1), curve_equal=rec_old["curve"] == rec["curve"],
               transition=(rec_old["transition"], rec["transition"]), shared_fields=len(shared),
               differing_fields=len(differ), first_differences=differ[:12],
               first_curve_point=(rec_old["curve"][0], rec["curve"][0]))
    with open(os.path.join(ROOT, "docs/review_2026-10/repro_out.jsonl"), "a") as f:
        f.write(json.dumps(out) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
