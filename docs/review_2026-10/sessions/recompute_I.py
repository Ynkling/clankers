#!/usr/bin/env python3
"""recompute_I.py — referee recomputation of Session I's report_1 (S74 pilot of the ICMC task) from its JSON records.

Pinned: branch origin/claude/explore-I at a2588da. Records: explore_out/I/s74_pilot_results.json at that commit; the
log the report cites (explore_out/I/s74_pilot_log.md) is checked in a second table. Stdlib only, read-only on the
repository (git show / git log); writes only docs/review_2026-10/sessions/recompute_I.md.

Every reported value below is TYPED IN from the report (file:line) and is never used as an input; every recomputed value
is derived from the record fields. Definitions used (cited from the branch's code at the pin):
  SET accuracy   explore_i_task.py docstring "SET accuracy": fraction of masked positions whose top-3 word set equals
                 T_src[prev]; recorded per evaluation as rec["curve"][i]["set"], at the end as rec["end"]["set"].
  masked loss    explore_i_common.py docstring EVALUATION "loss": masked cross-entropy (recorded, end["loss"]; the
                 report prints no loss value).
  routing eta^2  explore_i_common.py EVALUATION "routing" (eta^2 by source of the gate at word positions); recorded
                 for ORACLE only (L1 = 1.0 everywhere); the report prints no routing value.
  probe          explore_i_common.py EVALUATION "probe": logistic probe from the residual entering layer 1/2/3 and the
                 final layer's output to the source, held-out accuracy = rec[...]["probe"][L]["acc"].
  USABLE         explore_i_pilot.py docstring (fixed at 6484af8; the user's prompt at 8be3f98): per paired seed
                 ORACLE end SET >= 0.9 and SINGLE end SET <= ORACLE end SET - 0.2; usable iff >= 3 of 4 seeds;
                 recomputed exactly as explore_i_pilot.report() does.
  minutes/run    rec["secs"] / 60 (explore_i_pilot docstring: "the wall time per run is recorded (rec['secs'])").
  step transition (report_1 l.30-32): the evaluation with the largest one-evaluation rise of the layer-3 probe
                 (it coincides with the largest SET rise and with the first evaluation at which L3 >= 0.9).
"""
import datetime
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # docs/review_2026-10
from rlib import load, runs, band, git              # noqa: E402

PIN = "a2588da"
BRANCH = "origin/claude/explore-I"
REC = "explore_out/I/s74_pilot_results.json"
REP = "explore_out/I/report_1.md"
LOG = "explore_out/I/s74_pilot_log.md"
SEEDS = (800, 801, 802, 803)
ARMS = ("ORACLE", "SINGLE", "NOMIX")

st = load(PIN, REC)
R = {c: {a: runs(st, f"{c}_{a}") for a in ARMS} for c in ("C1", "C2")}
ALL = [r for c in R.values() for a in c.values() for r in a.values()]
assert len(ALL) == len(st["runs"]) == 20 and all(r["ok"] for r in ALL)

f3 = lambda x: f"{x:.3f}"
f2 = lambda x: f"{x:.2f}"
f0 = lambda x: f"{x:.0f}"
rng3 = lambda v: f"{f3(min(v))}-{f3(max(v))}"
rng2 = lambda v: f"{f2(min(v))}-{f2(max(v))}"
END = lambda c, a: [R[c][a][s]["end"]["set"] for s in sorted(R[c][a])]
MIN = lambda c, a: [R[c][a][s]["secs"] / 60 for s in sorted(R[c][a])]

ROWS, LOGROWS = [], []


def row(loc, item, reported, recomputed, status=None, note="", tab=None):
    if status is None:
        status = "MATCH" if str(reported) == str(recomputed) else "DISAGREE"
    (ROWS if tab is None else tab).append((loc, item, str(reported), str(recomputed), status, note))


def usable(c):
    ok = []
    for s in SEEDS:
        o, g = R[c]["ORACLE"][s]["end"]["set"], R[c]["SINGLE"][s]["end"]["set"]
        ok.append((o >= 0.9, g <= o - 0.2, o >= 0.9 and g <= o - 0.2))
    return ok


def probes(r, layers=("L1", "L2", "L3", "final")):
    return [e["probe"][L]["acc"] for e in r["curve"] + [r["end"]] for L in layers if e.get("probe")]


def transition(r):
    c = r["curve"]
    j = max(range(1, len(c)), key=lambda i: c[i]["probe"]["L3"]["acc"] - c[i - 1]["probe"]["L3"]["acc"])
    return c[j - 1], c[j], j


def ts(s):
    return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def hm(t):
    return t.strftime("%H:%M:%S")


# ── report_1.md l.3 (setup) ──────────────────────────────────────────────────
seeds_rec = sorted({r["seed"] for r in ALL})
row(f"{REP}:3", "seeds", "800-803", f"{seeds_rec[0]}-{seeds_rec[-1]}" if seeds_rec == list(SEEDS) else seeds_rec,
    note="every arm x config has exactly seeds 800, 801, 802, 803")
row(f"{REP}:3", "torch version", "2.14.0+cu130", st["meta"]["torch"])
cpus = {r["cpu"] for r in ALL} | {st["meta"]["cpu"]}
row(f"{REP}:3", "CPU", "Intel Xeon @ 2.10GHz", "Intel Xeon @ 2.10GHz" if cpus == {"Intel(R) Xeon(R) Processor @ 2.10GHz"}
    else cpus, note=f"records: {sorted(cpus)} on all 20 runs (report abbreviates)")
row(f"{REP}:3", "1 thread per run", "1", "—", "NOT RECORD-CHECKABLE",
    "thread count is not a record field (set in explore_i_common._worker_init)")
row(f"{REP}:3", "workers", "4", st["meta"]["workers"])

# ── l.5-8 (model, arms, recipe, budget, rule) ────────────────────────────────
row(f"{REP}:5-6", "model BDH N=256, D=32, 3 layers, decay 0.95, conv 'layer' width 4", "as printed", "—",
    "NOT RECORD-CHECKABLE", "no model constant is a record field; git shows the hashed model code identical for C1 "
    "and C2 (fd8d3b9..830d16d changes only explore_i_pilot.py among the hashed modules; see history table)")
spec_or = {(r["spec"]["gate"], r["spec"]["k"]) for c in R for r in R[c]["ORACLE"].values()}
row(f"{REP}:6-7", "ORACLE = perfect gate, k = S (S = 2)", "perfect, k=2",
    ", ".join(f"{g}, k={k}" for g, k in sorted(spec_or)))
spec_sg = {(r["spec"]["gate"], r["spec"]["k"]) for c in R for a in ("SINGLE", "NOMIX") for r in R[c][a].values()}
row(f"{REP}:7", "SINGLE (and NOMIX) k = 1, no gate", "none, k=1", ", ".join(f"{g}, k={k}" for g, k in sorted(spec_sg)))
nomix_lab = {r["task"] for r in R["C2"]["NOMIX"].values()}
or2_lab = {r["task"] for r in R["C2"]["ORACLE"].values()}
row(f"{REP}:7", "NOMIX = one source at the same L (C2)", "S=1, L=480",
    "S=1, L=480" if all("S=1 " in t and t.endswith("L=480") for t in nomix_lab) and all(t.endswith("L=480") for t in or2_lab)
    else sorted(nomix_lab))
recipes = {r["spec"]["recipe"] for r in ALL}
row(f"{REP}:7", "Adam 1e-3", "Adam 1e-3", f"recipe {sorted(recipes)}; lr not recorded", "PARTIAL",
    "recipe name 'adam' on all 20 records; the learning rate is not a record field")
iters = {(r["iters"], r["end"]["step"], r["curve"][-1]["step"]) for r in ALL}
one = next(iter(iters))
row(f"{REP}:7", "updates", "12000", one[0] if len(iters) == 1 and len(set(one)) == 1 else sorted(iters),
    note="iters = end step = last curve step on all 20 records")
row(f"{REP}:7", "batch B", "32", "—", "NOT RECORD-CHECKABLE", "batch size is not a record field (explore_i_common BATCH)")
doc0 = git("show", f"6484af8:explore_i_pilot.py")
docP = git("show", f"{PIN}:explore_i_pilot.py")
rule = "ORACLE's SET accuracy at the end >= 0.9 and SINGLE's <= ORACLE's - 0.2\n  (paired seed) on >= 3/4 seeds"
row(f"{REP}:7-8", "usability rule", "ORACLE final SET >= 0.9 and SINGLE <= ORACLE - 0.2 on >= 3/4 seeds",
    "ORACLE final SET >= 0.9 and SINGLE <= ORACLE - 0.2 on >= 3/4 seeds" if rule in doc0 and rule in docP else "differs",
    note="text check against explore_i_pilot.py docstring at 6484af8 and at the pin (unchanged), and report()'s code")

# ── l.14 P0 ──────────────────────────────────────────────────────────────────
row(f"{REP}:14", "P0 infeasible (<= 90 tokens fit)", "90", "—", "NOT RECORD-CHECKABLE",
    f"no P0 record (not run); arithmetic from the constants: S*(NB + NB*LMAX) = 2*(3+3*14) = {2 * (3 + 3 * 14)} "
    "(consistent)")

# ── l.15-16 counts table ─────────────────────────────────────────────────────
LAB = {"C1": "S=2 V=4 NB=16 block words 6..14 M=176 words/source L=384",
       "C2": "S=2 V=4 NB=40 block words 3..7 M=200 words/source L=480"}
for c, line, cst, bnd, logline, o_cnt, o_rng, s_cnt, s_rng, us, mins in (
        ("C1", 15, "2, 4, 6..14, 16, 384", "0.932", "s74_pilot_log.md:29 (0.932)", "4/4", "0.910-0.918", "2/4",
         "0.661-0.746", "no (2/4)", "106-111"),
        ("C2", 16, "2, 4, 3..7, 40, 480", "0.936", "s74_pilot_log.md:76 and pilot docstring (0.936; no table row)",
         "4/4", "0.916-0.918", "1/4", "0.654-0.863", "no (1/4)", "155-166")):
    labs = {r["task"] for a in ("ORACLE", "SINGLE") for r in R[c][a].values()}
    lab = labs.pop() if len(labs) == 1 else None
    parsed = None
    if lab:
        kv = dict(t.split("=") for t in lab.replace("block words ", "lw=").replace(" words/source", "").split()[1:])
        parsed = f"{kv['S']}, {kv['V']}, {kv['lw']}, {kv['NB']}, {kv['L']}"
    loc = f"{REP}:{line}"
    row(loc, f"{c} S, V, words/block, NB, L", cst, parsed, note=f"task label: '{lab}'")
    row(loc, f"{c} Bayes bound", bnd, "—", "NOT RECORD-CHECKABLE",
        f"Monte-Carlo constant (explore_i_task.ideal_set_acc, needs torch); not in the records; cf. {logline}")
    u = usable(c)
    row(loc, f"{c} ORACLE >= 0.9 count", o_cnt, f"{sum(x[0] for x in u)}/4")
    row(loc, f"{c} ORACLE end SET range", o_rng, rng3(END(c, "ORACLE")),
        note="per seed " + ", ".join(f"{s}:{R[c]['ORACLE'][s]['end']['set']:.4f}" for s in SEEDS))
    row(loc, f"{c} SINGLE <= ORACLE - 0.2 count", s_cnt, f"{sum(x[1] for x in u)}/4",
        note="per seed SINGLE - (ORACLE - 0.2): " + ", ".join(
            f"{s}:{R[c]['SINGLE'][s]['end']['set'] - R[c]['ORACLE'][s]['end']['set'] + 0.2:+.4f}" for s in SEEDS))
    row(loc, f"{c} SINGLE end SET range", s_rng, rng3(END(c, "SINGLE")),
        note="per seed " + ", ".join(f"{s}:{R[c]['SINGLE'][s]['end']['set']:.4f}" for s in SEEDS))
    n_joint = sum(x[2] for x in u)
    row(loc, f"{c} usable verdict (joint count)", us, f"{'yes' if n_joint >= 3 else 'no'} ({n_joint}/4)",
        note=f"rlib.band({n_joint}, 4) = {band(n_joint, 4)} (the report prints no band)")
    m = MIN(c, "ORACLE") + MIN(c, "SINGLE")
    row(loc, f"{c} minutes per run (ORACLE+SINGLE)", mins, f"{f0(min(m))}-{f0(max(m))}",
        note="secs/60: ORACLE " + "/".join(f"{x:.1f}" for x in MIN(c, "ORACLE")) + "; SINGLE " +
             "/".join(f"{x:.1f}" for x in MIN(c, "SINGLE")) +
             ("; NOMIX " + "/".join(f"{x:.1f}" for x in MIN(c, "NOMIX")) if R[c]["NOMIX"] else ""))

# ── l.18-21 NOMIX, S = 4 ─────────────────────────────────────────────────────
row(f"{REP}:18", "C1 NOMIX stopped (no record)", "0 records", f"{len(R['C1']['NOMIX'])} records",
    note="no C1_NOMIX key in the store")
row(f"{REP}:18", "C2 NOMIX end SET range", "0.966-0.967", rng3(END("C2", "NOMIX")),
    note="per seed " + ", ".join(f"{s}:{R['C2']['NOMIX'][s]['end']['set']:.4f}" for s in SEEDS))
row(f"{REP}:18", "C2 NOMIX count", "4/4", f"{len(R['C2']['NOMIX'])}/4")
row(f"{REP}:18", "NOMIX M", "400", nomix_lab.pop().split("M=")[1].split()[0] if len(nomix_lab) == 1 else "?")
row(f"{REP}:19", "C2 ORACLE SET ('ORACLE's 0.917')", "0.917", f3(statistics.median(END("C2", "ORACLE"))),
    note="median of the 4 end SET values")
row(f"{REP}:19", "ORACLE below NOMIX (all seeds)", "below", "below" if max(END("C2", "ORACLE")) < min(END("C2", "NOMIX"))
    else "not below")
n_s4 = sum("S=4" in r["task"] for r in ALL)
row(f"{REP}:20", "S = 4 not run", "0 runs", f"{n_s4} runs")
row(f"{REP}:20-21", "S=4 bound 0.82 at L=288; L ~ 768 for 0.93; ~10 h per run", "0.82 / 768 / 10 h", "—",
    "NOT RECORD-CHECKABLE", "ideal-learner constants and a timing estimate; not in the records (log l.38, l.48-50)")

# ── l.25-28 ──────────────────────────────────────────────────────────────────
row(f"{REP}:26-27", "bound < 0.9 for V >= 5 at L <= 288 and for S = 4 below L ~ 700", "< 0.9", "—",
    "NOT RECORD-CHECKABLE", "Monte-Carlo constants (log P1 table l.19-40), not records")
gaps = {"C1": [0.932 - x for x in END("C1", "ORACLE")], "C2": [0.936 - x for x in END("C2", "ORACLE")]}
allg = gaps["C1"] + gaps["C2"]
row(f"{REP}:28", "ORACLE under the bound at L=384 and 480", "0.015-0.02",
    f"per seed {f3(min(allg))}-{f3(max(allg))}; medians {f3(statistics.median(gaps['C1']))} (C1), "
    f"{f3(statistics.median(gaps['C2']))} (C2)", "PARTIAL",
    "the bound is not a record (log values 0.932/0.936 used); per-seed differences 0.014-0.022 straddle the printed "
    "0.015-0.02 at both ends, medians 0.019-0.020")
row(f"{REP}:28", "smoke test: both arms within 0.01 of the ideal learners", "0.01", "—", "NOT RECORD-CHECKABLE",
    "the smoke test (log l.52-54) left no record in the store")

# ── l.29-34 transitions and probes ───────────────────────────────────────────
S2, O2 = R["C2"]["SINGLE"], R["C2"]["ORACLE"]
moved = {s: transition(S2[s]) for s in SEEDS if max(probes(S2[s], ("L3",))) >= 0.9}
row(f"{REP}:30", "C2 SINGLE runs with a step transition", "3", len(moved),
    note="runs whose L3 probe ever reaches >= 0.9: " + ", ".join(map(str, sorted(moved))))
tr = {s: moved[s][1]["step"] for s in moved}
for s in moved:
    pre, post, j = moved[s]
    jset = max(range(1, len(S2[s]["curve"])), key=lambda i: S2[s]["curve"][i]["set"] - S2[s]["curve"][i - 1]["set"])
    first90 = next(e["step"] for e in S2[s]["curve"] if e["probe"]["L3"]["acc"] >= 0.9)
    assert S2[s]["curve"][jset]["step"] == post["step"] == first90
row(f"{REP}:30", "transition updates (803, 800, 801)", "2400, 9600, 10800", f"{tr[803]}, {tr[800]}, {tr[801]}",
    note="largest L3-probe rise = largest SET rise = first eval with L3 >= 0.9, for each seed")
rise = {s: moved[s][1]["set"] - moved[s][0]["set"] for s in moved}
row(f"{REP}:31", "SET rise at the transition evaluation", "0.10-0.21", rng2(rise.values()),
    note="per seed " + ", ".join(f"{s}: {moved[s][0]['set']:.4f} -> {moved[s][1]['set']:.4f} (+{rise[s]:.4f})"
                                 for s in sorted(moved)) + "; 801's +0.087 and 803's +0.217 fall outside 0.10-0.21")
pre3 = {s: moved[s][0]["probe"]["L3"]["acc"] for s in moved}
row(f"{REP}:31", "L3 probe before the transition", "0.52-0.58", rng2(pre3.values()),
    note="evaluation before the transition: " + ", ".join(f"{s}: {pre3[s]:.4f}" for s in sorted(moved)) +
         "; no reading tried (pre-eval L3, final layer, L2) gives 0.52 at the low end")
post3 = {s: moved[s][1]["probe"]["L3"]["acc"] for s in moved}
end3 = {s: S2[s]["end"]["probe"]["L3"]["acc"] for s in moved}
row(f"{REP}:31", "L3 probe after ('to 0.98')", "0.98", f"end {rng2(end3.values())}; at the transition eval "
    f"{rng2(post3.values())}", "PARTIAL",
    "matches the end-of-run values (" + ", ".join(f"{s}: {end3[s]:.4f}" for s in sorted(moved)) + "); at the "
    "transition evaluation itself, which the sentence names, seed 803 reads " + f"{post3[803]:.4f}")
post2 = {s: moved[s][1]["probe"]["L2"]["acc"] for s in moved}
end2 = {s: S2[s]["end"]["probe"]["L2"]["acc"] for s in moved}
row(f"{REP}:31", "L2 probe ('layer 2: 0.87-0.91')", "0.87-0.91", f"end {rng2(end2.values())}; at the transition eval "
    f"{rng2(post2.values())}", "PARTIAL", "matches the end-of-run values (log l.86 also gives them as 'at the end'); "
    "at the transition evaluation 0.87-0.89")
g802 = O2[802]["end"]["set"] - S2[802]["end"]["set"]
cl_c = {s: 1 - (O2[s]["end"]["set"] - S2[s]["end"]["set"]) / g802 for s in moved}
cl_a = {s: 1 - (O2[s]["end"]["set"] - S2[s]["end"]["set"]) /
        (O2[s]["curve"][moved[s][2] - 1]["set"] - moved[s][0]["set"]) for s in moved}
row(f"{REP}:32", "gap to ORACLE closed", "60-80%",
    f"{min(cl_c.values()):.0%}-{max(cl_c.values()):.0%} vs seed 802's gap; {min(cl_a.values()):.0%}-"
    f"{max(cl_a.values()):.0%} vs own pre-transition gap", "PARTIAL",
    "definition not stated; reproduced only as 1 - gap_end(s)/gap_end(802) (802 = the run that did not transition, "
    f"gap {g802:.3f}); per-seed gap before the transition (paired ORACLE at the same eval) vs the end gives "
    + ", ".join(f"{s}: {cl_a[s]:.0%}" for s in sorted(cl_a)))
row(f"{REP}:32", "seed 802 SET", "0.65", f2(S2[802]["end"]["set"]), note=f"end SET {S2[802]['end']['set']:.4f}")
row(f"{REP}:32", "seed 802 probe", "0.51", f2(S2[802]["end"]["probe"]["L3"]["acc"]),
    note=f"end L3 {S2[802]['end']['probe']['L3']['acc']:.4f}; max L3 over the run "
         f"{max(probes(S2[802], ('L3',))):.4f} (never moved)")
c1p = [p for s in SEEDS for p in probes(R["C1"]["SINGLE"][s])] + \
      [p for s in SEEDS for p in probes(R["C1"]["ORACLE"][s])]
row(f"{REP}:33", "C1: no probe moved (max probe)", "<= 0.54", f"<= {f2(max(c1p))}",
    note=f"max over every evaluation, layer and arm of C1: {max(c1p):.4f}")
row(f"{REP}:33", "C1: two SINGLE seeds drifted to", "0.74",
    " / ".join(f2(x) for x in sorted(END("C1", "SINGLE"))[-2:]), "PARTIAL",
    "the two top seeds end at " + ", ".join(f"{x:.4f}" for x in sorted(END("C1", "SINGLE"))[-2:]) +
    "; 800's 0.7457 rounds to 0.75 (0.74 is a truncation; the log's l.70 says 0.75 for 800)")
op = [p for c in R for s in SEEDS for p in probes(R[c]["ORACLE"][s])]
row(f"{REP}:34", "ORACLE probe never carried the source", "<= 0.53", f"<= {f2(max(op))}",
    note=f"max over every evaluation and layer, C1 and C2 ORACLE: {max(op):.4f}")

# ── l.38 stop ────────────────────────────────────────────────────────────────
ck = st["meta"]["checks"]
c1_start = ts(ck["C1"]["when"]) - datetime.timedelta(seconds=round(ck["C1"]["secs"]))
c2_start = ts(ck["C2"]["when"]) - datetime.timedelta(seconds=round(ck["C2"]["secs"]))
c2_end = ts(ck["C2"]["when"]) + datetime.timedelta(seconds=max(r["secs"] for r in O2.values()) +
                                                    max(r["secs"] for r in S2.values()))
c1_end = ts(ck["C1"]["when"]) + datetime.timedelta(seconds=max(r["secs"] for r in R["C1"]["ORACLE"].values()) +
                                                    max(r["secs"] for r in R["C1"]["SINGLE"].values()))
row(f"{REP}:38", "~9.5 h of the pilot's day", "~9.5 h", f"compute span ~{(c2_end - c1_start).seconds / 3600:.1f} h",
    "NOT RECORD-CHECKABLE", f"the day's start (design work before the first CHECK) is not recorded; reconstructed compute "
    f"span: C1 CHECKs start {hm(c1_start)} (meta.checks.C1.when - secs) to C2 ORACLE+SINGLE end ~{hm(c2_end)} "
    "(C2 CHECK end + longest ORACLE + longest SINGLE: two waves of 4 workers)")
s75 = [p for p in git("ls-tree", "-r", "--name-only", PIN, "explore_out/I/").split() if "s75" in p]
row(f"{REP}:38", "S75 not run", "0 S75 records", f"{len(s75)} S75 records")

# ── l.42-44 relevance ────────────────────────────────────────────────────────
row(f"{REP}:42", "Section 10 / Session G probes <= 0.6", "<= 0.6", "—", "NOT RECORD-CHECKABLE",
    "other sessions' records, out of this scope")
l3c = sum(S2[s]["end"]["probe"]["L3"]["acc"] >= 0.9 for s in SEEDS)
l2c = sum(S2[s]["end"]["probe"]["L2"]["acc"] >= 0.9 for s in SEEDS)
row(f"{REP}:43-44", "C2 SINGLE stack carries the source at layers 2-3", "3/4",
    f"L3 >= 0.9: {l3c}/4; L2 >= 0.9: {l2c}/4", "PARTIAL",
    "3/4 at layer 3 (end probe >= 0.9, Session G's 'cue present' threshold); at layer 2 the three runs read "
    f"{rng2(end2.values())}, >= 0.9 on {l2c}/4 only; the report states no threshold")

# ── l.47-55 CHECKs ───────────────────────────────────────────────────────────
okc = [(c, k) for c in ck for k, v in ck[c]["results"].items() if not v.get("ok")]
row(f"{REP}:47", "CHECKs all ok (both configurations)", "all ok",
    "all ok" if not okc and set(ck) == {"C1", "C2"} else okc,
    note=f"{sum(len(ck[c]['results']) for c in ck)} CHECK groups over C1 and C2, every ok = true")
true_set = all(v["items"].get("SET accuracy of the true sets = 1.0") for c in ck for k, v in ck[c]["results"].items()
               if "items" in v)
row(f"{REP}:50", "SET = 1.0 on the true sets", "1.0", "1.0" if true_set else "fails")
rp = [v for c in ck for k, v in ck[c]["results"].items() if k.startswith("C2 reproduction")]
row(f"{REP}:50-51", "reproduction bit for bit through 2400 (curve, grad norms)", "2400",
    max(x[0] for x in rp[0]["curve"]) if all(v["curve"] == v["recorded"] and v["grad_same"] for v in rp) else "fails",
    note="curve == recorded (exact float equality) and grad_same, both CHECK runs")
w = [ck[c]["results"]["C3 windows"] for c in ck]
row(f"{REP}:51", "LOCAL3 / WIDE window reach", "sees / not beyond",
    "sees / not beyond" if all(x[g]["sees_far"] and x[g]["not_beyond"] for x in w for g in ("win3", "wide")) else "fails",
    note="booleans only; the offsets t-2/t-3, t-(LMAX+1)/t-(LMAX+2) are the CHECK's definition, not recorded numbers")
fired = {tuple(ck[c]["results"]["C6 split mechanics"]["fired"]) for c in ck}
row(f"{REP}:53", "split fires", "4800/9600/14400", "/".join(map(str, fired.pop())) if len(fired) == 1 else fired)
row(f"{REP}:54", "split at threshold 0 = no split through 7200", "7200",
    "ok" if all(ck[c]["results"]["C6b split at threshold 0 = no split"]["ok"] for c in ck) else "fails", "PARTIAL",
    "record holds ok = true and checks = 2; the horizon 7200 is not a record field")
pr = {(ck[c]["results"]["C7 probe"]["informative"], ck[c]["results"]["C7 probe"]["random"]) for c in ck}
p0 = pr.pop()
row(f"{REP}:55", "probe synthetic check (informative / random)", "1.00 / 0.49", f"{f2(p0[0])} / {f2(p0[1])}")

# ── the log (cited by the report) ────────────────────────────────────────────
L = LOGROWS
for c, l1, l2, l3, om, sm in (("C1", 64, 65, 66, "111", "106"), ("C2", 83, 84, 85, "165", "155")):
    rep = {"C1": ("0.918 | 0.910 | 0.910 | 0.913", "0.746 | 0.741 | 0.662 | 0.661", "0.718 | 0.710 | 0.710 | 0.713"),
           "C2": ("0.916 | 0.917 | 0.917 | 0.918", "0.812 | 0.812 | 0.654 | 0.863", "0.716 | 0.717 | 0.717 | 0.718")}[c]
    row(f"{LOG}:{l1}", f"{c} ORACLE SET per seed", rep[0], " | ".join(f3(x) for x in END(c, "ORACLE")), tab=L)
    row(f"{LOG}:{l2}", f"{c} SINGLE SET per seed", rep[1], " | ".join(f3(x) for x in END(c, "SINGLE")), tab=L)
    row(f"{LOG}:{l3}", f"{c} ORACLE - 0.2 per seed", rep[2], " | ".join(f3(x - 0.2) for x in END(c, "ORACLE")), tab=L)
    row(f"{LOG}:{l1}", f"{c} ORACLE minutes per run (median)", om, f0(statistics.median(MIN(c, "ORACLE"))), tab=L)
    row(f"{LOG}:{l2}", f"{c} SINGLE minutes per run (median)", sm, f0(statistics.median(MIN(c, "SINGLE"))), tab=L)
row(f"{LOG}:86", "C2 SINGLE probe L2 / L3 at the end", "0.87 / 0.98 | 0.87 / 0.98 | 0.52 / 0.51 | 0.91 / 0.98",
    " | ".join(f"{f2(S2[s]['end']['probe']['L2']['acc'])} / {f2(S2[s]['end']['probe']['L3']['acc'])}" for s in SEEDS),
    tab=L)
c800 = {e["step"]: e["set"] for e in R["C1"]["SINGLE"][800]["curve"]}
row(f"{LOG}:70", "C1 SINGLE 800 from 6000 to the end", "0.70 -> 0.75",
    f"{f2(c800[6000])} -> {f2(R['C1']['SINGLE'][800]['end']['set'])}", tab=L)
c1s = [p for s in SEEDS for p in probes(R["C1"]["SINGLE"][s])]
c1e = [R["C1"]["SINGLE"][s]["end"]["probe"][k]["acc"] for s in SEEDS for k in ("L1", "L2", "L3", "final")]
row(f"{LOG}:71", "C1 SINGLE source probe range", "0.49-0.54", f"end {rng2(c1e)}; all evaluations {rng2(c1s)}",
    "PARTIAL" if rng2(c1e) == "0.49-0.54" else None, tab=L,
    note="matches the end-of-run probes (every layer, four C1 SINGLE runs); over every evaluation the minimum is "
         f"{min(c1s):.4f} (a chance-level reading, not a move)")
row(f"{LOG}:89-90", "SET jump at the same evaluation", "0.10-0.21", rng2(rise.values()), tab=L,
    note="as report l.31")
row(f"{LOG}:90-91", "L3 probe at the same evaluation: from / to", "0.52-0.58 / 0.98",
    f"{rng2(pre3.values())} / {rng2(post3.values())}", tab=L,
    note="the log says 'at the same evaluation' explicitly; 803 reads 0.957 there (0.981 at 4800 and at the end)")
nm = [R["C2"]["NOMIX"][s]["end"]["set"] for s in SEEDS]
row(f"{LOG}:103", "C2 NOMIX SET per seed", "0.967, 0.966, 0.966, 0.966", ", ".join(f3(x) for x in nm), tab=L)
row(f"{LOG}:103", "C2 NOMIX minutes", "157-160", f"{f0(min(MIN('C2', 'NOMIX')))}-{f0(max(MIN('C2', 'NOMIX')))}", tab=L)
row(f"{LOG}:60", "C1 window, runs", "04:38-08:22; 8 runs",
    f"{c1_start.strftime('%H:%M')}-~{c1_end.strftime('%H:%M')}; {len(R['C1']['ORACLE']) + len(R['C1']['SINGLE'])} runs",
    "PARTIAL", tab=L, note="end time reconstructed (CHECK end + longest ORACLE + longest SINGLE); records hold no "
    "wall-clock end; the store's clock is assumed UTC (consistent with the commit times below)")
row(f"{LOG}:79", "C2 window, runs", "08:27-14:00; 8 runs",
    f"{ts(ck['C2']['when']).strftime('%H:%M')}-~{c2_end.strftime('%H:%M')}; "
    f"{len(O2) + len(S2)} runs", "PARTIAL", tab=L, note="runs start after the C2 CHECKs ended "
    f"({hm(ts(ck['C2']['when']))}); end reconstructed as above")

# ── history: criteria, constants, first record ───────────────────────────────


def cinfo(c):
    h, d, s = git("log", "-1", "--format=%h%x09%cd%x09%s", "--date=format-local:%Y-%m-%d %H:%M:%S",
                  c).strip().split("\t")
    return h[:7], d + " UTC", s


os.environ["TZ"] = "UTC"


def first_with(path, pred=lambda b: True):
    for c in git("log", "--reverse", "--format=%h", BRANCH, "--", path).split():
        try:
            b = git("show", f"{c}:{path}")
        except Exception:
            continue
        if pred(b):
            return c
    return None


import hashlib  # noqa: E402
import json  # noqa: E402


def code_sha_at(c):
    h = hashlib.sha1()
    for n in sorted(st["meta"]["modules"]):
        h.update(n.encode())
        h.update(git_bytes(c, n + ".py"))
    return h.hexdigest()[:12]


def git_bytes(c, p):
    import subprocess
    from rlib import ROOT
    return subprocess.check_output(("git", "-C", ROOT, "show", f"{c}:{p}"))


branch_commits = git("log", "--reverse", "--format=%h", f"e1f760a..{PIN}").split()
sha_first = {}
for c in branch_commits:
    try:
        s = code_sha_at(c)
    except Exception:
        continue
    sha_first.setdefault(s, c)
run_shas = sorted({r["sha"] for r in ALL})
c_rule = first_with("explore_i_pilot.py", lambda b: rule in b)
c_c1 = first_with("explore_i_pilot.py", lambda b: '"C1": dict(S=2, V=4, NB=16, lmin=6, lmax=14, M=176)' in b)
c_c2 = first_with("explore_i_pilot.py", lambda b: '"C2": dict(S=2, V=4, NB=40, lmin=3, lmax=7, M=200)' in b)
c_rec = first_with(REC)
c_run = first_with(REC, lambda b: len(json.loads(b)["runs"]) > 0)
c_log_c1 = first_with(LOG, lambda b: "**C1** (first pilot configuration)" in b)
c_log_c2 = first_with(LOG, lambda b: "**C2**: S = 2, V = 4, LMIN..LMAX = 3..7" in b)
prompt = git("show", "8be3f98:docs/reading/parallel_sessions_prompts.md")
assert "Usable at a configuration if ORACLE's SET accuracy at the end ≥ 0.9 and SINGLE's ≤ ORACLE − 0.2" in prompt
# the log is append-only after its first commit (no earlier line edited)
log_versions = [git("show", f"{c}:{LOG}") for c in git("log", "--reverse", "--format=%h", BRANCH, "--", LOG).split()]
append_only = all(b.startswith(a.rstrip("\n")) for a, b in zip(log_versions, log_versions[1:]))
doc_rule_same = rule in docP
c1_code, c2_code = sha_first[run_shas[0]], sha_first[run_shas[1]]
model_files_same = [n + ".py" for n in sorted(st["meta"]["modules"])
                    if git_bytes(c1_code, n + ".py") != git_bytes(c2_code, n + ".py")]


def utc(c):
    return cinfo(c)[1]


def dt_of(c):
    return datetime.datetime.strptime(cinfo(c)[1][:19], "%Y-%m-%d %H:%M:%S")


crit_before = dt_of(c_rule) < c1_start and dt_of(c_rule) < dt_of(c_run)

# ── write ────────────────────────────────────────────────────────────────────
ORDER = {"DISAGREE": 0, "PARTIAL": 1, "NOT RECORD-CHECKABLE": 2, "MATCH": 3}


def esc(x):
    return str(x).replace("|", "\\|").replace("\n", " ")


def table(rows):
    out = ["| # | report file:line | item | reported (as printed) | recomputed | status | note |", "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(sorted(rows, key=lambda r: ORDER[r[4]]), 1):
        out.append("| " + " | ".join([str(i)] + [esc(x) for x in r]) + " |")
    return out


def counts(rows):
    return {k: sum(r[4] == k for r in rows) for k in ORDER}


pin_full = git("rev-parse", PIN).strip()
md = [f"# Recompute of Session I report_1 (S74 pilot, ICMC) from its records",
      "",
      f"Pinned commit: `{pin_full[:7]}` ({utc(PIN)}) on `{BRANCH}`. Records: `{REC}` at that commit "
      f"(20 run records: C1 ORACLE/SINGLE x 4, C2 ORACLE/SINGLE/NOMIX x 4; code SHAs {', '.join(run_shas)}). "
      f"Report: `{REP}`. Log cited by the report: `{LOG}`. Generated by `docs/review_2026-10/sessions/recompute_I.py` "
      "(stdlib + rlib.load/runs/band/git; reported values typed in from the report, never used as inputs).",
      "",
      "Definitions used (from the branch's code at the pin):",
      "- SET accuracy: `explore_i_task.py` docstring 'SET accuracy' (top-3 word set equals T_src[prev], over masked "
      "positions); end value `rec['end']['set']`, per evaluation `rec['curve'][i]['set']`.",
      "- Masked loss / routing eta^2 by source: `explore_i_common.py` docstring EVALUATION; recorded but the report "
      "prints no loss or eta^2 value, so no row uses them (ORACLE's recorded eta^2 is 1.0 at every evaluation).",
      "- Probe: `explore_i_common.py` EVALUATION 'probe' (logistic probe to the source, held-out accuracy "
      "`probe[L]['acc']`, L in L1, L2, L3, final).",
      "- USABLE: `explore_i_pilot.py` docstring and `report()`: per paired seed ORACLE end SET >= 0.9 and SINGLE end "
      "SET <= ORACLE - 0.2; usable iff >= 3/4 seeds. The two marginal counts in the report's table are the two "
      "conditions counted separately; the 'usable' column is the joint count.",
      "- Minutes per run: `rec['secs']/60`, rounded as `report()` does (`:.0f`).",
      "- Step transition: the evaluation with the largest one-evaluation rise of the layer-3 probe (it coincides with "
      "the largest SET rise and the first evaluation with L3 >= 0.9 on each of the three runs).",
      "- The report prints no Wilson interval, McNemar test or band; `rlib.band` of the joint usable counts is noted "
      "for reference only (C1 2/4 MAJORITY, C2 1/4 MINORITY).",
      "",
      "## report_1.md rows",
      "",
      "Counts: " + ", ".join(f"{k} {v}" for k, v in counts(ROWS).items()) + ".",
      ""] + table(ROWS) + [
      "",
      "## Log rows (s74_pilot_log.md, cited by the report as its full log)",
      "",
      "Counts: " + ", ".join(f"{k} {v}" for k, v in counts(LOGROWS).items()) + ".",
      ""] + table(LOGROWS) + [
      "",
      "## When the criteria and constants were fixed",
      "",
      "| event | commit | UTC date | evidence |",
      "|---|---|---|---|",
      f"| usability criteria in the user's prompt | {cinfo('8be3f98')[0]} | {utc('8be3f98')} | "
      "docs/reading/parallel_sessions_prompts.md, Session I, S74 |",
      f"| usability criteria + C1 constants fixed in the pilot docstring (and CONFIGS) | {cinfo(c_rule)[0]} | "
      f"{utc(c_rule)} | explore_i_pilot.py USABLE paragraph; C1 CONFIGS first at {cinfo(c_c1)[0]}; rule text "
      f"unchanged at the pin: {doc_rule_same} |",
      f"| code the C1 runs ran (code SHA {run_shas[0]}) | {sha_first.get(run_shas[0], '?')} | "
      f"{utc(sha_first[run_shas[0]]) if run_shas[0] in sha_first else '?'} | SHA-1 over meta.modules at that commit "
      "reproduces the records' code SHA (first commit with that content) |",
      f"| C1 CHECKs start (records) | — | {c1_start} (store clock) | meta.checks.C1.when - secs |",
      f"| first commit of the record file | {cinfo(c_rec)[0]} | {utc(c_rec)} | meta + C1 CHECK record, 0 runs |",
      f"| first commit holding a run record | {cinfo(c_run)[0]} | {utc(c_run)} | 4 C1 ORACLE runs |",
      f"| C2 constants fixed (docstring, CONFIGS, log) | {cinfo(c_c2)[0]} | {utc(c_c2)} | log C2 entry first at "
      f"{cinfo(c_log_c2)[0]} |",
      f"| code the C2 runs ran (code SHA {run_shas[1]}) | {sha_first.get(run_shas[1], '?')} | "
      f"{utc(sha_first[run_shas[1]]) if run_shas[1] in sha_first else '?'} | as above |",
      f"| C2 CHECKs start (records) | — | {c2_start} (store clock) | meta.checks.C2.when - secs |",
      "",
      f"Criteria precede the first run: **{'yes' if crit_before else 'no'}** (docstring {utc(c_rule)}; prompt "
      f"{utc('8be3f98')}; C1 CHECKs began {c1_start}; first run record committed {utc(c_run)}). The store's "
      "timestamps carry no zone; they are read as UTC, which is consistent with every commit time (each CHECK start "
      f"falls seconds after the commit whose content the runs' code SHA reproduces). Of the {len(st['meta']['modules'])} hashed "
      f"modules (meta.modules), the only one differing between the C1 code ({c1_code}) and the C2 code ({c2_code}) is: "
      f"{', '.join(model_files_same)} (docstring + CONFIGS['C2']), so the model, harness and metrics code were the "
      "same for both configurations. Dates are committer dates.",
      "",
      "## Constant changes recorded in the pilot log",
      "",
      f"The log is append-only across its commits: {append_only}. B = 32 throughout; the model never changed.",
      "",
      "| # | change (log line) | constants | logged in | UTC | governs | logged before those runs? |",
      "|---|---|---|---|---|---|---|",
      f"| 1 | P0 starting point infeasible (l.8-10) | L=192, NB=3, 6..14, V=16, B=32 (prompt's start) | "
      f"{cinfo(c_log_c1)[0]} | {utc(c_log_c1)} | nothing (not run) | n/a (no run) |",
      f"| 2 | P1 Bayes bound: V 16 -> 4; V >= 5 excluded (l.12-46) | V=4 | {cinfo(c_log_c1)[0]} | {utc(c_log_c1)} | "
      "C1, C2 | yes |",
      f"| 3 | S = 4 not run, decided by the bound and timing (l.48-50) | S=2 only | {cinfo(c_log_c1)[0]} | "
      f"{utc(c_log_c1)} | (no S=4 runs) | yes (no S=4 record exists) |",
      f"| 4 | smoke test (l.52-54) | S=2, V=4, NB=4, M=44, L=96, 3600 updates, seed 800 | {cinfo(c_log_c1)[0]} | "
      f"{utc(c_log_c1)} | itself (not a pilot configuration) | no: logged with its result, no record in the store |",
      f"| 5 | C1 (l.56-58): NB 3 -> 16, L 192 -> 384 (M=176), 6..14 kept; NOMIX NB=32, M=352 | S=2 V=4 NB=16 6..14 L=384 "
      f"B=32 | {cinfo(c_log_c1)[0]} | {utc(c_log_c1)} | C1 ORACLE/SINGLE (NOMIX stopped, no record) | "
      f"{'yes' if dt_of(c_log_c1) < c1_start else 'no'} (C1 CHECKs began {c1_start}) |",
      f"| 6 | C2 (l.74-77): LMIN..LMAX 6..14 -> 3..7, NB 16 -> 40, L 384 -> 480 (M=200); NOMIX NB=80, M=400 | "
      f"S=2 V=4 NB=40 3..7 L=480 B=32 | {cinfo(c_log_c2)[0]} | {utc(c_log_c2)} | C2 ORACLE/SINGLE/NOMIX | "
      f"{'yes' if dt_of(c_log_c2) < c2_start else 'no'} (C2 CHECKs began {c2_start}, "
      f"{int((c2_start - dt_of(c_log_c2)).total_seconds())} s later) |",
      ""]
out = os.path.join(HERE, "recompute_I.md")
with open(out, "w") as f:
    f.write("\n".join(md) + "\n")
print("report rows:", counts(ROWS))
print("log rows:   ", counts(LOGROWS))
for r in sorted(ROWS + LOGROWS, key=lambda r: ORDER[r[4]]):
    if r[4] == "DISAGREE":
        print("DISAGREE", r[0], r[1], "reported", r[2], "recomputed", r[3])
print("wrote", out)
