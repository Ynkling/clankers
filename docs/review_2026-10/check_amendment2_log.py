#!/usr/bin/env python3
"""check_amendment2_log.py -- Amendment 2's statement of what had been seen, against X's run log.

    python3 docs/review_2026-10/check_amendment2_log.py      (stdlib; reads git objects; writes nothing)

Log: results/X/delta_recipe_X.log at 7d7661d (main line, 13 October). Each per-run line ends "(elapsed N.Nm)", the
minutes since t0 = time.time() taken right after meta.started is stamped (test_delta_recipe.py at a581027, after
verify()); meta.started is 2026-10-11 05:27:31 (results/X/delta_recipe_results.json at 8ae660d), in the log's clock,
whose header reads 02:12:34 for the process start (a581027 was committed 02:12:29 UTC), so the clock is UTC.
"""
import datetime as D, json, re, subprocess
git = lambda *a: subprocess.check_output(("git",) + a).decode()
started = json.loads(git("show", "8ae660d:results/X/delta_recipe_results.json"))["meta"]["started"]
t0 = D.datetime.strptime(started, "%Y-%m-%d %H:%M:%S")
log = git("show", "7d7661d:results/X/delta_recipe_X.log").splitlines()
print("log header:", next(l for l in log if l.strip()))
rows = []
for l in log:
    m = re.match(r"\s+(\w+)\s+seed (\d+) .*\(elapsed ([\d.]+)m\)", l)
    if m:
        rows.append((m[1], int(m[2]), t0 + D.timedelta(minutes=float(m[3]))))
print(f"per-run lines: {len(rows)}; meta.started {started}")
for name, ref in (("55dac50", "55dac50"), ("2225f00", "2225f00")):
    cut = D.datetime.strptime(git("show", "-s", "--format=%ci", ref)[:19], "%Y-%m-%d %H:%M:%S")
    done = [r for r in rows if r[2] <= cut]
    by = {}
    for a, s, _ in done:
        by.setdefault(a, []).append(s)
    print(f"{name} at {cut}: {len(done)} finished: " + "; ".join(f"{a} {sorted(v)}" for a, v in sorted(by.items())))
nxt = sorted((t, a, s) for a, s, t in rows if t > D.datetime(2026, 10, 11, 15, 36, 8))[:3]
print("next to finish:", [(a, s, str(t)) for t, a, s in nxt])
print("first HEBB_SPLIT/HEBB_SLOW finish:", min(t for a, s, t in rows if a in ("HEBB_SPLIT", "HEBB_SLOW")))
