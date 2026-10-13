"""check_E19_v4.py — the referee (V), fourth pass on the Revision 9 draft: Session E's batch 19 (S64 width,
S65 depth, S66 eight streams, S67 capacity) recomputed from the records at e3f5809 (origin/claude/outside-ideas),
against the README's Batch 19 verdicts and the fourth draft (c66cc76, cited tex4:NNN).
Stdlib only, plus docs/review_2026-10/rlib.py.  Reads records with `git show`; writes only check_E19_v4_out.txt
next to this file.  Run: python3 docs/review_2026-10/rev9/check_E19_v4.py"""
import math
import os
import sys
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import rlib  # noqa: E402

REF = "e3f5809"        # E's completion commit
READINGS = "617681d"   # first commit of the four screen files (docstring READING blocks)
DRAFT4 = "c66cc76"
out = []


def say(s=""):
    out.append(s)
    print(s)


def store(name):
    return rlib.load(REF, f"explore_out/{name}_results.json")


def br(r):
    return rlib.bound_routed(r)


def tmed(rs, f=br):
    t = [rlib.transition(r["curve"]) for r in rs.values() if f(r)]
    return rlib.med(t), sorted(t)


def cmp(arm, ref, seeds):
    b, c, n = rlib.pairs(arm, ref, br, seeds)
    return b, c, b - c, rlib.mcnemar2(b, c), rlib.mcnemar1(b, c)


def line(tag, rs, seeds, f=br):
    k = sum(1 for s in seeds if f(rs[s]))
    lo, hi = rlib.wilson(k, len(seeds))
    m, ts = tmed(rs, f)
    return k, f"{tag}: {k}/{len(seeds)} ({rlib.band(k, len(seeds))}, Wilson [{lo:.2f}, {hi:.2f}]); transition median {m}; {ts}"


# ── record integrity ─────────────────────────────────────────────────────────
say("== 0. Stores at e3f5809: run counts, ok, dry, CPU, code SHA; stored outcome vs recomputed")
total = 0
for nm in ("scale_width", "scale_depth", "scale_eight", "capacity"):
    st = store(nm)
    rs = st["runs"]
    total += len(rs)
    cpus = {r.get("cpu") for r in rs.values()}
    shas = {r.get("code_sha", "")[:12] for r in rs.values()}
    nok = sum(1 for r in rs.values() if not r.get("ok"))
    ndry = sum(1 for r in rs.values() if r.get("dry"))
    mism = []
    for k, r in rs.items():
        mine = rlib.outcome_k(r) if r.get("ok") else "not ok"
        if r.get("kind") == "perfect" or k.startswith("ORC"):
            continue
        rec = r.get("outcome")
        if (rec == "BOUND ROUTED") != (mine == "BOUND ROUTED"):
            mism.append((k.split("|")[0], k.split("|")[1], rec, mine))
    say(f"  {nm}: {len(rs)} runs; not ok {nok}; dry {ndry}; cpu {cpus}; code {shas}; BOUND ROUTED stored-vs-recomputed mismatches {mism}")
say(f"  total {total} (draft tex4:235 and tex4:508: 186)")
say(f"  rlib ties (rounded stream_gate) so far: {rlib.TIES}")

# ── S64 ──────────────────────────────────────────────────────────────────────
say("\n== 1. S64 width (seeds 270-279, oracle 270-271; reading: 'scale-free' if every tested d >= -1, 'width-sensitive' if any d <= -4)")
st = store("scale_width")
S = list(range(270, 280))
for a in ("N256", "N512", "N1024", "D64", "D128"):
    o = rlib.runs(st, "ORC_" + a)
    ot = [rlib.transition(r["curve"]) for _, r in sorted(o.items())]
    say(f"  oracle {a}: bound {sum(rlib.bound(r) for r in o.values())}/{len(o)} at {ot}")
ref = rlib.runs(st, "N256")
k, s_ = line("N256 (reference)", ref, S)
say("  " + s_)
ds = []
for a in ("N512", "N1024", "D64", "D128"):
    rs = rlib.runs(st, a)
    k, s_ = line(a, rs, S)
    b, c, d, p2, p1 = cmp(rs, ref, S)
    ds.append(d)
    say(f"  {s_}; vs N256 {b} / {c}, d = {d:+d}, two-sided p = {p2:.3g} (one-sided {p1:.3g})")
say(f"  ref failures: {[(s, rlib.outcome_k(r)) for s, r in sorted(ref.items()) if not br(r)]}")
say(f"  reading: {'scale-free' if all(d >= -1 for d in ds) else 'width-sensitive' if any(d <= -4 for d in ds) else 'neither'}")

# ── S65 ──────────────────────────────────────────────────────────────────────
say("\n== 2. S65 depth (seeds 280-289, oracle 280-281; same rule vs L3; RES6 only if L6 - L3 <= -2)")
st = store("scale_depth")
S = list(range(280, 290))
for a in ("L2", "L3", "L4", "L6"):
    o = rlib.runs(st, "ORC_" + a)
    say(f"  oracle {a}: bound {sum(rlib.bound(r) for r in o.values())}/{len(o)} at {[rlib.transition(r['curve']) for _, r in sorted(o.items())]}")
ref = rlib.runs(st, "L3")
say("  " + line("L3 (reference)", ref, S)[1])
ds = {}
for a in ("L2", "L4", "L6"):
    rs = rlib.runs(st, a)
    k, s_ = line(a, rs, S)
    b, c, d, p2, p1 = cmp(rs, ref, S)
    ds[a] = d
    say(f"  {s_}; vs L3 {b} / {c}, d = {d:+d}, two-sided p = {p2:.3g}")
    say(f"     failures {[(s, rlib.outcome_k(r)) for s, r in sorted(rs.items()) if not br(r)]}")
say(f"  reading: {'scale-free' if all(d >= -1 for d in ds.values()) else 'depth-sensitive' if any(d <= -4 for d in ds.values()) else 'neither'}; "
    f"RES6 condition (L6 - L3 <= -2): {ds['L6'] <= -2}; RES6 records: {len(rlib.runs(st, 'RES6'))}")

# ── S66 ──────────────────────────────────────────────────────────────────────
say("\n== 3. S66 eight streams (seeds 290-299; reading per arm: 'carries' if d >= -1 vs N256_L3)")
st = store("scale_eight")
S = list(range(290, 300))
ref = rlib.runs(st, "N256_L3")
arms = {"N256_L3": ref, "D128_L3": rlib.runs(st, "D128_L3"), "N256_L2": rlib.runs(st, "N256_L2")}
alltr = []
for a, rs in arms.items():
    k, s_ = line(a, rs, S)
    extra = ""
    if a != "N256_L3":
        b, c, d, p2, p1 = cmp(rs, ref, S)
        extra = f"; vs N256_L3 {b} / {c}, d = {d:+d}, two-sided p = {p2:.3g} (one-sided {p1:.3g}) -> {'carries' if d >= -1 else 'does not carry'}"
    say(f"  {s_}{extra}")
    fails = [(s, rlib.outcome_k(r), r.get("splits"), round(r["acc"], 2)) for s, r in sorted(rs.items()) if not br(r)]
    say(f"     failures (seed, class, splits, final acc): {fails}")
    sp = [r.get("splits") for _, r in sorted(rs.items())]
    say(f"     splits per run {sp}, total {sum(sp)}")
    fc = [next((c["acc"] for c in r.get("checks") or [] if c["step"] == 4800), None) for _, r in sorted(rs.items())]
    say(f"     probe accuracy at the first check (4800): median {statistics.median(fc):.3f}; {[round(x, 2) for x in fc]}")
    tr = [rlib.transition(r["curve"]) for r in rs.values()]
    alltr += [t for t in tr if t is not None]
    say(f"     transitions (all bound runs) {sorted(t for t in tr if t is not None)}; recorded 'transition' field agrees: "
        f"{all(rlib.transition(r['curve']) == r.get('transition') for r in rs.values())}")
say(f"  earliest transition in S66: {min(alltr)}; any at or before 4800: {any(t <= 4800 for t in alltr)}; "
    f"at or before 2400: {any(t <= 2400 for t in alltr)}")
say("  docstring ALSO REPORTED rule: 'if an arm's median transition is at or before 4800, the schedule is size-dependent'")

# ── S67 ──────────────────────────────────────────────────────────────────────
say("\n== 4. S67 capacity (oracle 270-271, learned 270-279; reading on the P=16 oracles only)")
st = store("capacity")
S = list(range(270, 280))
orc = {}
for P in (8, 16):
    for w in ("N256", "D128"):
        o = rlib.runs(st, f"ORC_P{P}_{w}")
        orc[(P, w)] = sum(rlib.bound(r) for r in o.values())
        say(f"  oracle P={P} {w}: bound {orc[(P, w)]}/{len(o)}; "
            f"{[(s, rlib.transition(r['curve']), round(r['acc'], 2)) for s, r in sorted(o.items())]}")
o256, ow = orc[(16, "N256")], orc[(16, "D128")]
rd = ("capacity-bound at N=256" if o256 == 0 and ow >= 1 else "not capacity-bound at N=256" if o256 >= 1
      else "the oracle binds P=16 at neither width")
say(f"  reading: {rd}")
for P in (8, 16):
    rn = rlib.runs(st, f"P{P}_N256")
    rw = rlib.runs(st, f"P{P}_D128")
    for tag, rs in ((f"P{P}_N256", rn), (f"P{P}_D128", rw)):
        k, s_ = line(tag, rs, S)
        say("  " + s_)
        fails = [(s, rlib.outcome_k(r), round(r["acc"], 2)) for s, r in sorted(rs.items()) if not br(r)]
        say(f"     failures: {fails}")
        cls = {}
        for f in fails:
            cls[f[1]] = cls.get(f[1], 0) + 1
        say(f"     failure classes {cls}")
    b, c, d, p2, p1 = cmp(rw, rn, S)
    say(f"  P={P}: D128 only {b} / N256 only {c}, two-sided p = {p2:.4g} (one-sided {p1:.4g}); printed, no reading")

# ── power of the one-pair rule ───────────────────────────────────────────────
say("\n== 5. Operating characteristic of 'd >= -1 over 10 seeds' (independence approximation, as stats_checks.py §7)")


def p_hold(pr, pa, n=10, m=-1):
    q1, q2 = pa * (1 - pr), pr * (1 - pa)
    q0 = 1 - q1 - q2
    return sum(math.factorial(n) / (math.factorial(b) * math.factorial(c) * math.factorial(n - b - c))
               * q1 ** b * q2 ** c * q0 ** (n - b - c) for b in range(n + 1) for c in range(n + 1 - b) if b - c >= m)


for pr, pa in ((0.9, 0.7), (0.9, 0.6), (0.8, 0.8), (0.8, 0.6), (0.8, 0.5), (0.8, 0.4)):
    say(f"  reference {pr}, arm {pa}: P(reads 'scale-free' / 'carries') = {p_hold(pr, pa):.2f}")

# ── readings commit vs first record ──────────────────────────────────────────
say("\n== 6. Readings commit vs first record")


def first_commit(path, nonempty_runs=True):
    revs = rlib.git("log", "--reverse", "--format=%h %ad", "--date=iso", "origin/claude/outside-ideas", "--", path).split("\n")
    for ln in revs:
        if not ln.strip():
            continue
        h = ln.split()[0]
        try:
            d = rlib.load(h, path)
        except Exception:
            continue
        if not nonempty_runs or d.get("runs"):
            return ln, sorted(d["runs"])[:1]
    return None, None


say(f"  docstrings first committed: {rlib.git('log', '-1', '--format=%h %ad', '--date=iso', READINGS).strip()}")
say("  READING blocks unchanged 617681d..e3f5809: "
    + str(all(rlib.git("show", f"{READINGS}:{f}").split('"""')[1] == rlib.git("show", f"{REF}:{f}").split('"""')[1]
              for f in ("explore_scale_width.py", "explore_scale_depth.py", "explore_scale_eight.py", "explore_capacity.py"))))
for nm in ("scale_width", "scale_depth", "scale_eight", "capacity"):
    say(f"  first {nm} record: {first_commit(f'explore_out/{nm}_results.json')}")
rt = rlib.load(REF, "explore_out/batch19_runtime_results.json")["meta"]["phase2"]
say(f"  phase-2 decision stored {rt['time']}: best_width {rt['best_width']}, best_depth {rt['best_depth']}")

with open(os.path.join(HERE, "check_E19_v4_out.txt"), "w") as f:
    f.write("\n".join(out) + "\n")
