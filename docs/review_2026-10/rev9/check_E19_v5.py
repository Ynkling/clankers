"""check_E19_v5.py — the referee (V), fifth pass on the Revision 9 draft: Session E's batch 19 (S64 width,
S65 depth, S66 eight streams, S67 capacity) recomputed from the records at e3f5809 (origin/claude/outside-ideas),
against the README's Batch 19 verdicts and the fifth draft (c5e4508, cited tex5:NNN). Adapted from
check_E19_v4.py: sections 0-6 are v4's recomputations, unchanged; section 7 adds the one-pair rule's power at
eight streams as the fifth version words it; section 8 E's branch at the fifth version's commit (batch 20);
section 9 locates the wording; section 10 prints the rows, each anchored on a fragment of its tex5 line, and the counts.
Stdlib only, plus docs/review_2026-10/rlib.py.  Reads records with `git show`; prints; writes nothing.
Run: python3 docs/review_2026-10/rev9/check_E19_v5.py"""
import math
import os
import sys
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import rlib  # noqa: E402

REF = "e3f5809"        # E's completion commit
READINGS = "617681d"   # first commit of the four screen files (docstring READING blocks)
DRAFT4, DRAFT5 = "c66cc76", "c5e4508"
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
say(f"  total {total} (draft tex5:237 and tex5:512: 186)")
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


# ── the one-pair rule at eight streams, as the fifth version words it ────────
say("\n== 7. 'carries' (d >= -1, 10 seeds) against a reference at 0.8: P(reads 'carries') = P(misses the drop)")
for pa in (0.75, 0.7, 0.65, 0.6, 0.55, 0.5):
    nested = sum(math.comb(10, c) * (0.8 - pa) ** c * (1 - 0.8 + pa) ** (10 - c) for c in range(2))
    say(f"  arm {pa} vs 0.8: independence {p_hold(0.8, pa):.3f}; arm's binds a subset of the reference's (max. correlation) {nested:.3f}")
cross = next(pa / 100 for pa in range(80, 0, -1) if p_hold(0.8, pa / 100) < 0.5)
say(f"  P(carries) first falls below 0.5 at an arm rate of {cross:.2f} (independence): a drop to 0.6 is missed "
    f"{p_hold(0.8, 0.6):.2f} of the time, i.e. caught {1 - p_hold(0.8, 0.6):.2f}, not 'missed more often than not'")
say(f"  four streams, 0.7 vs 0.9: P(reads 'scale-free') = {p_hold(0.9, 0.7):.3f} (tex5:368 '39%')")

# ── E's branch at the fifth version's commit ─────────────────────────────────
say("\n== 8. E's branch at the fifth version's commit (c5e4508)")
t5 = rlib.git("log", "-1", "--format=%aI", DRAFT5).strip()
head5 = rlib.git("log", "-1", "--format=%h %aI %s", f"--until={t5}", "origin/claude/outside-ideas").strip()
say(f"  c5e4508 at {t5}; E's last commit before it: {head5}")
b20 = rlib.git("log", "--reverse", "--format=%h %aI %s", "e3f5809..origin/claude/outside-ideas").strip().split("\n")
say(f"  commits after e3f5809: {len(b20)}; first {b20[0]}")
start = next(l for l in b20 if "segment 1 started" in l)
say(f"  batch 20's first segment: {start}")
h5 = head5.split()[0]
rd = [l for l in rlib.git("show", f"{h5}:explore_out/README.md").split("\n") if l.startswith("| `batch20.log`")]
say(f"  README at {h5}, batch 20 row: {rd[0][:260] if rd else 'none'}")
same = all(rlib.git("show", f"e3f5809:{f}") == rlib.git("show", f"origin/claude/outside-ideas:{f}") for f in
           ("explore_out/scale_width_results.json", "explore_out/scale_depth_results.json", "explore_out/scale_eight_results.json",
            "explore_out/capacity_results.json", "explore_out/batch19_runtime_results.json"))
say(f"  batch 19 stores byte-identical e3f5809 .. origin head ({rlib.git('rev-parse', '--short', 'origin/claude/outside-ideas').strip()}): {same}")
intro = rlib.git("log", "--format=%h %aI", "-S", "every session's queue is empty", f"{DRAFT4}..{DRAFT5}", "--",
                 "multichannel_hebbian_report_v9.tex").strip()
say(f"  'every session's queue is empty' introduced at {intro}; batch 20's code and readings committed at "
    f"{rlib.git('log', '-1', '--format=%h %aI', '4acf915').strip()}")

# ── where ────────────────────────────────────────────────────────────────────
tex5 = rlib.git("show", f"{DRAFT5}:multichannel_hebbian_report_v9.tex").split("\n")
tex4 = rlib.git("show", f"{DRAFT4}:multichannel_hebbian_report_v9.tex").split("\n")


def where(pat, t):
    import re
    return [i + 1 for i, l in enumerate(t) if re.search(pat, l)]


say("\n== 9. Where the fifth version says it (tex5; tex4 for comparison)")
for pat in (r"batch~?19", r"S66", r"S67", r"one-pair rule", r"40\\%", r"more often than not", r"one of two", r"best depth",
            r"bound routed \$10/10\$, \$10/10\$ and \$9/10\$", r"screened once in S66", r"e3f5809", r"queue is empty", r"batches 1--19",
            r"capacity reason", r"four applications of"):
    say(f"  {pat!r}: tex5 {where(pat, tex5)}; tex4 {where(pat, tex4)}")

# ── rows ─────────────────────────────────────────────────────────────────────
import re  # noqa: E402


def cnt_(stn, arm, seeds):
    rs = rlib.runs(store(stn), arm)
    return sum(1 for s in seeds if br(rs[s])), rs


W, D_, E8, C = list(range(270, 280)), list(range(280, 290)), list(range(290, 300)), list(range(270, 280))
w = {a: cnt_("scale_width", a, W) for a in ("N256", "N512", "N1024", "D64", "D128")}
dp = {a: cnt_("scale_depth", a, D_) for a in ("L2", "L3", "L4", "L6")}
e8 = {a: cnt_("scale_eight", a, E8) for a in ("N256_L3", "D128_L3", "N256_L2")}
cp = {a: cnt_("capacity", a, C) for a in ("P8_N256", "P8_D128", "P16_N256", "P16_D128")}
L6b = sum(1 for s in D_ if rlib.bound(dp["L6"][1][s]))
dmed = {a: tmed(dp[a][1])[0] for a in dp}
wmed = {a: tmed(w[a][1])[0] for a in w}
e8m = {a: tmed(e8[a][1])[0] for a in e8}
pd = {a: cmp(e8[a][1], e8["N256_L3"][1], E8) for a in ("D128_L3", "N256_L2")}
p16 = cmp(cp["P16_D128"][1], cp["P16_N256"][1], C)
oc = {}
for P in (8, 16):
    for wd in ("N256", "D128"):
        o = rlib.runs(store("capacity"), f"ORC_P{P}_{wd}")
        oc[(P, wd)] = (sum(rlib.bound(r) for r in o.values()), len(o), sorted(rlib.transition(r["curve"]) for r in o.values() if rlib.bound(r)))
fails8 = [(a, s, rlib.outcome_k(r), r.get("splits")) for a in e8 for s, r in sorted(e8[a][1].items()) if not br(r)]
first8 = min(rlib.transition(r["curve"]) for a in e8 for r in e8[a][1].values() if rlib.transition(r["curve"]) is not None)
pc = {a: statistics.median([next(c["acc"] for c in r["checks"] if c["step"] == 4800) for r in e8[a][1].values()]) for a in e8}
cls16 = {}
for a in ("P16_N256", "P16_D128"):
    for s, r in cp[a][1].items():
        if not br(r):
            cls16[(a, rlib.outcome_k(r))] = cls16.get((a, rlib.outcome_k(r)), 0) + 1
tot = sum(len(store(n)["runs"]) for n in ("scale_width", "scale_depth", "scale_eight", "capacity"))
cpus = {r.get("cpu") for n in ("scale_width", "scale_depth", "scale_eight", "capacity") for r in store(n)["runs"].values()}

say("\n== 10. Rows (each anchored on a fragment of its tex5 line) and counts")
ROWS = [
    (368, r"whose one-pair rule misses a drop from \$0\.8\$ to \$0\.6\$ more often than not",
     "S66's power in the verdict paragraph (new text, e0c5549)", "'whose one-pair rule misses a drop from 0.8 to 0.6 more often than not'",
     f"P(reads 'carries' | 0.6 vs 0.8) = {p_hold(0.8, 0.6):.2f} (independence), {sum(math.comb(10, c) * 0.2 ** c * 0.8 ** (10 - c) for c in range(2)):.2f} (max. correlation): the drop is missed 40% of the time and caught 60%",
     "DISAGREE", "The reverse of the truth, and of tex5:237's own '40%'. P(miss) exceeds 0.5 only for an arm at "
     f"{cross + 0.01:.2f} or above (independence). Fix: 'whose one-pair rule misses a drop from 0.8 to 0.6 about two times in five'."),
    (41, r"every session's queue is empty", "E's status at c5e4508 (new text, 3b06af2)", "'E (... batches 1--19, complete)'; '... every session's queue is empty'",
     f"E's batch 20 (S68 s67_continue: S67's unbound learned runs regenerated to 86,400) started {start.split()[1][:16]}; at c5e4508 the README reads 'IN PROGRESS: 10 of 19 runs saved' ({h5})",
     "DISAGREE", "True when written (3b06af2, 2026-10-11 17:43, before batch 20's code at 21:27), stale for the last six draft commits. 'batches 1--19, complete' is true. Batch 20 is S67's continuation, the draft's own 'clearest lead' (tex5:237, 477), and is not mentioned. Note: E names it S68, which collides with F's S68 (tex5:41, 362, 514)."),
    (237, r"The rule, ``carries'' at \$d \\geq -1\$ on ten seeds against a reference at \$8/10\$, reads ``carries'' about \$40\\%\$ of the time when the larger model's true rate is \$0\.6\$ against \$0\.8\$, so the screen rules out a large drop and not a moderate one",
     "S66's power (fourth pass §2, tex4:235)", "'reads \"carries\" about 40% of the time ... 0.6 against 0.8'", f"{p_hold(0.8, 0.6):.3f}", "MATCH", "The power caveat now covers eight streams."),
    (237, r"two layers was S65's ``best depth'' through a tie with four that the arms' order broke, a step outside the pre-registered rule, which session~E discloses",
     "the two-layer arm's tie-break (fourth pass §2, tex4:235)", "'through a tie with four that the arms' order broke, a step outside the pre-registered rule'",
     f"phase-2 decision {rt['time']}: best_depth {rt['best_depth']}; L2 and L4 {dp['L2'][0]}/10 and {dp['L4'][0]}/10, medians {dmed['L2']} and {dmed['L4']}, equal parameters; README:657-665",
     "MATCH", "E's further warning (README:699, 'Its 10/10 reads only for 2 layers, not as \"fewer layers is better\"') is not relayed; the draft makes no such claim."),
    (237, r"and at \$D = 128\$ on one of two, so the reading ``not capacity-bound at \$N = 256\$'', which is about the oracle, applies and the oracle gives no capacity reason for the wider model's learned advantage",
     "the D=128 oracle (fourth pass §2, tex4:235)", "'and at D = 128 on one of two ... the oracle gives no capacity reason ...'",
     f"ORC_P16_N256 {oc[(16, 'N256')][0]}/{oc[(16, 'N256')][1]} at {oc[(16, 'N256')][2]}; ORC_P16_D128 {oc[(16, 'D128')][0]}/{oc[(16, 'D128')][1]} at {oc[(16, 'D128')][2]}; README S67 'the oracle gives no capacity reason'",
     "MATCH", ""),
    (57, r"carries to \$D = 128\$ and 2 layers at eight \(one-pair rule\); the oracle \(2 seeds\) binds 16 keys at \$N = 256\$, the learned recipe \$0/10\$",
     "index row (fourth pass §2, tex4:57)", "'(one-pair rule); the oracle (2 seeds) binds 16 keys at N = 256, the learned recipe 0/10'",
     f"oracle {oc[(16, 'N256')][0]}/2; P16_N256 {cp['P16_N256'][0]}/10", "MATCH", ""),
    (57, r"scale-free to \$N = 1024\$, \$D = 128\$, 4 layers at four streams", "index row, scope", "scale-free to N=1024, D=128, 4 layers at four streams",
     "S64, S65 'scale-free'; E's verdict 'promising to 4 layers'", "MATCH", "The rule's reading also covers 6 layers (d = -1)."),
    (454, r"its constants read ``scale-free'' and ``carries'' by one-pair rules on ten-seed screens, to four times the width and to four applications of the layer at four streams and to \$D = 128\$ and two layers at eight",
     "'What Phase VI adds' (fourth pass §2, tex4:452)", "'its constants read \"scale-free\" and \"carries\" by one-pair rules ..., to four times the width and to four applications of the layer at four streams ...'",
     f"S64 d = +1 x4; S65 L4 d = 0; S66 D128 d = {pd['D128_L3'][2]:+d}, L2 d = {pd['N256_L2'][2]:+d}", "MATCH",
     "The readings are named as one-pair readings; 'twice the depth' (L6) replaced by four applications (E's verdict)."),
    (235, r"2, 4 and 6 applications of the shared layer bound routed \$10/10\$, \$10/10\$ and \$9/10\$ against \$10/10\$ for three, with six layers slower \(median 4800\)",
     "S65 'bound routed' (fourth pass §2, tex4:233)", "'bound routed 10/10, 10/10 and 9/10 against 10/10 for three ... (median 4800)'",
     f"L2 {dp['L2'][0]}, L4 {dp['L4'][0]}, L6 {dp['L6'][0]} (bound {L6b}), L3 {dp['L3'][0]}; medians {dmed}", "MATCH", ""),
    (468, r"eight streams at the larger widths \(screened once in S66\)\.", "staged path (fourth pass §2, tex4:464)", "'eight streams at the larger widths (screened once in S66)'",
     f"S66 D128_L3 {e8['D128_L3'][0]}/10, N256_L2 {e8['N256_L2'][0]}/10", "MATCH", ""),
    (368, r"the oracle binds sixteen keys per stream at \$N = 256\$ \(S67, two seeds\) where the learned recipe binds \$0/10\$ against \$4/10\$ at \$D = 128\$",
     "S67 in the verdict paragraph", "oracle, two seeds; learned 0/10 vs 4/10", f"oracle {oc[(16, 'N256')][0]}/2; {cp['P16_N256'][0]}/10 vs {cp['P16_D128'][0]}/10", "MATCH", ""),
    (368, r"reads ``scale-free'' \$39\\%\$ of the time when the larger model's rate is \$0\.7\$ against \$0\.9\$",
     "four-stream power (unchanged)", "39% at 0.7 vs 0.9", f"{p_hold(0.9, 0.7):.3f}", "MATCH", ""),
    (368, r"Eight streams carry to \$D = 128\$ and two layers on a ten-seed screen \(S66\)", "S66 in the verdict paragraph", "carry to D=128 and two layers",
     f"D128_L3 d = {pd['D128_L3'][2]:+d}, N256_L2 d = {pd['N256_L2'][2]:+d}: both 'carries'", "MATCH", ""),
    (41, r"\\textbf\{E\} \(branch \\texttt\{claude/outside-ideas\}, batches 1--19, complete\).*batch~19 on 2\.80\\,GHz", "E's CPU", "batch 19 on 2.80 GHz",
     f"{tot} records, cpu {cpus}", "MATCH", ""),
    (235, r"\$N = 512\$ and \$1024\$ at \$D = 32\$, and \$D = 64\$ and \$128\$ at \$N = 256\$, bound routed \$10/10\$ each against the reference's \$9/10\$, all with median transition 3600",
     "S64", "10/10 each vs 9/10; medians 3600", f"{ {a: w[a][0] for a in w} }; medians {wmed}", "MATCH", "All four d = +1 rest on seed 273."),
    (235, r"the oracle binding at every size and depth", "S64/S65 oracles", "oracle binds everywhere", "S64 10/10, S65 8/8 (§1, §2)", "MATCH", ""),
    (235, r"Both readings were ``scale-free'' by their pre-fixed rule \(every \$d \\geq -1\$\), at four streams only; with ten seeds the screen rules out a large drop, not a small one",
     "S64/S65 readings", "scale-free, every d >= -1", "S64 +1 x4; S65 0, 0, -1; RES6 not run", "MATCH", ""),
    (235, r"Eight streams and larger key loads at the larger widths follow\.", "transition sentence (unchanged)", "as stated", "—", "MATCH", "S66 and S67 also ran N256 arms; 'at the larger widths' is loose (v4 row 15)."),
    (237, r"The batch completed after this revision's first draft \(186 of 186 runs\)", "completion", "186 of 186", f"{tot} runs; first draft 9bfa85a 10-10 22:16, e3f5809 10-11 04:25", "MATCH", ""),
    (237, r"\(\$k = 16\$, the convolution, 43\{,\}200 updates, seeds 290--299, paired with the \$N = 256\$, three-layer reference at \$8/10\$\)",
     "S66 configuration and reference", "k=16, conv, 43,200, 290-299, reference 8/10", f"N256_L3 {e8['N256_L3'][0]}/10", "MATCH", ""),
    (237, r"binds routed \$8/10\$ at \$D = 128\$ \(2 vs 2; ``carries'', \$d = 0\$\) and \$10/10\$ with two applications of the layer \(2 vs 0, \$p = 0\.5\$; ``carries'', \$d = \+2\$",
     "S66 counts", "8/10 (2 vs 2, d = 0); 10/10 (2 vs 0, p = 0.5, d = +2)",
     f"D128 {e8['D128_L3'][0]}/10 ({pd['D128_L3'][0]} vs {pd['D128_L3'][1]}, d = {pd['D128_L3'][2]:+d}); L2 {e8['N256_L2'][0]}/10 ({pd['N256_L2'][0]} vs {pd['N256_L2'][1]}, two-sided p = {pd['N256_L2'][3]:.3g})",
     "MATCH", ""),
    (237, r"both with an earlier median transition \(15\{,\}600 against 21\{,\}600\) and a higher probe accuracy at the first check",
     "S66 transitions and probe", "15,600 vs 21,600; higher probe at first check", f"medians {e8m}; probe at 4800 medians { {a: round(v, 3) for a, v in pc.items()} }", "MATCH", "Descriptions, not readings."),
    (237, r"every failure in the three arms had spent its three splits and ended merged", "S66 failures", "all failures 3 splits, merged", f"{fails8}", "MATCH", ""),
    (237, r"the learned recipe binds \$P = 8\$ at both widths \(\$9/10\$, \$10/10\$; median 7200\)", "S67 P=8", "9/10, 10/10; median 7200",
     f"{cp['P8_N256'][0]}/10, {cp['P8_D128'][0]}/10; medians {tmed(cp['P8_N256'][1])[0]}, {tmed(cp['P8_D128'][1])[0]}", "MATCH", ""),
    (237, r"\$P = 16\$ on \$0/10\$ at \$N = 256\$ against \$4/10\$ at \$D = 128\$ \(4 vs 0, \$p = 0\.125\$; printed, no reading\)", "S67 P=16", "0/10 vs 4/10; 4 vs 0; p = 0.125",
     f"{cp['P16_N256'][0]}/10 vs {cp['P16_D128'][0]}/10; {p16[0]} vs {p16[1]}; two-sided p = {p16[3]:.3g}", "MATCH", ""),
    (237, r"with most of the failures at both widths ending one-to-one and unbound", "S67 failure classes", "most one-to-one unbound", f"{cls16}", "MATCH", "12 of 16."),
    (237, r"no run in S66 had its transition at or before the first check or the learning-rate switch", "S66 schedule note", "none at or before 4800 / 2400", f"earliest transition {first8}", "MATCH", ""),
    (237, r"Session~E hands back \$P = 16\$ at \$D = 128\$ against \$D = 32\$ as its clearest lead, weak", "E's lead", "as README", "README 'Handed back'", "MATCH", "Text, not records."),
    (477, r"batch~19's lead, sixteen keys per stream at \$D = 128\$ against \$D = 32\$, as a main-line test with a longer budget", "next steps (unchanged)", "as stated",
     "E's batch 20 is an exploratory continuation of S67 to 86,400, in progress at c5e4508", "MATCH", "Not stale: batch 20 is not the main-line test. The draft does not mention batch 20 (row 2)."),
    (443, r"Three layers with shared weights \(two to six screened\), one head, \$d = 32\$ \(to 128 screened\), \$N = 256\$ \(to 1024 screened\)", "scale ranges (unchanged)", "2-6, to 128, to 1024", "S64, S65", "MATCH", ""),
    (512, r"S64 width, S65 depth, S66 eight streams, S67 capacity \(complete, \\texttt\{e3f5809\}\) & \\texttt\{e3f5809\} & 186", "provenance", "e3f5809; 186",
     f"{tot}; stores unchanged to E's head: {same}", "MATCH", ""),
]
order = {"DISAGREE": 0, "PARTIAL": 1, "NOT RECORD-CHECKABLE": 2, "MATCH": 3}
cnt = {k: 0 for k in order}
bad = 0
for i, (ln, frag, item, says, rec, stt, note) in enumerate(sorted(ROWS, key=lambda r: (order[r[5]], r[0])), 1):
    okk = re.search(frag, tex5[ln - 1]) is not None
    bad += not okk
    cnt[stt] += 1
    say(f"  {i:2d} | tex5:{ln} | anchor {'ok' if okk else 'MISSING'} | {stt} | {item} | recomputed: {rec}" + (f" | {note}" if note else ""))
say(f"\n  COUNTS: {cnt['DISAGREE']} DISAGREE, {cnt['PARTIAL']} PARTIAL, {cnt['NOT RECORD-CHECKABLE']} NOT RECORD-CHECKABLE, "
    f"{cnt['MATCH']} MATCH ({len(ROWS)} rows); anchors missing: {bad}")
