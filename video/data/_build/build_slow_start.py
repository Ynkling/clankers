"""video/data/slow_start.json — Table 4 (the slow-start test) from the raw results."""
from collections import Counter
from common import *

SS, SA, SR, SC = ("results/X/slow_start_results.json", "results/X/scale_axes_results.json",
                  "results/X/stream_recipe_results.json", "results/X/stream_curriculum_results.json")
ss, sa, sr, sc = load(SS), load(SA), load(SR), load(SC)
lrows, lcurves, _ = parse_llog()

S0 = list(range(280, 300))
SEEDS = {"A0": S0, "SLOW0": S0, "HINGE0": S0, "DIRECT8": list(range(220, 240)), "HINGE8": list(range(220, 240)),
         "A4k16": list(range(240, 260)), "HINGE4k16": list(range(240, 260)), "D8": list(range(260, 266)),
         "HINGE_D8": list(range(260, 270))}
PART = {"A0": "0", "SLOW0": "0", "HINGE0": "0", "DIRECT8": "A", "HINGE8": "A", "A4k16": "B", "HINGE4k16": "B",
        "D8": "C", "HINGE_D8": "C"}
FILE = {"A0": SS, "SLOW0": SS, "HINGE0": SS, "HINGE8": SS, "HINGE4k16": SS, "HINGE_D8": SS,
        "DIRECT8": SA, "A4k16": SR, "D8": SC}
STORE = {SS: ss, SA: sa, SR: sr, SC: sc}
X = {a: {s: r for s, r in runs_of(STORE[FILE[a]], a).items() if s in SEEDS[a]} for a in SEEDS}


def succ(a, r):
    return discovered(r) if PART[a] == "0" else bound(r)


def tag(a, r):
    if PART[a] in ("B", "C"):
        return tagk(r)
    return tag2(r, part0=PART[a] == "0")


REPORT_T4 = {  # report Table 4 (p. 7), checked against the PDF
    "X": {"A0": "8/20", "SLOW0": "15/20", "HINGE0": "19/20", "DIRECT8": "13/20", "HINGE8": "20/20",
          "A4k16": "12/20", "HINGE4k16": "17/20", "D8": "0/6", "HINGE_D8": "0/10"},
    "L": {"A0": "8/20", "SLOW0": "15/20", "HINGE0": "20/20", "DIRECT8": "12/20", "HINGE8": "17/20",
          "A4k16": "9/20", "HINGE4k16": "18/20", "D8": "0/10", "HINGE_D8": "0/10"},
}

counts = {"X": {}, "L": {}}
for a in SEEDS:
    rs = X[a]
    counts["X"][a] = dict(success=sum(succ(a, r) for r in rs.values()), n=len(rs),
                          outcome="DISCOVERED" if PART[a] == "0" else "BOUND",
                          collapsed=sum(bool(r["collapsed"]) for r in rs.values()),
                          from_data=True)
# L: only the arms whose per-seed rows L's recipe-scope log printed
LMAP = {"A0": "A0", "SLOW0": "SLOW0", "HINGE0": "HINGE0", "A4k16": "A4k16", "HINGE4k16": "HINGE4k16"}
for a in SEEDS:
    if a in LMAP:
        rows = lrows[LMAP[a]]
        ok_ = (lambda o: o == "DISCOVERED") if PART[a] == "0" else (lambda o: o.startswith("BOUND"))
        counts["L"][a] = dict(success=sum(ok_(r["outcome"]) for r in rows.values()), n=len(rows),
                              outcome="DISCOVERED" if PART[a] == "0" else "BOUND", from_data=True)
    else:
        s, n = REPORT_T4["L"][a].split("/")
        counts["L"][a] = dict(success=int(s), n=int(n), outcome="DISCOVERED" if PART[a] == "0" else "BOUND",
                              from_data=False)

check = []
for m in ("X", "L"):
    for a in SEEDS:
        c = counts[m][a]
        got = f"{c['success']}/{c['n']}"
        check.append(dict(machine=m, arm=a, data=got if c["from_data"] else None, report=REPORT_T4[m][a],
                          match=(got == REPORT_T4[m][a]) if c["from_data"] else None))

# claims (exact one-sided McNemar), X from the JSON, L from the log rows where both arms are there
CLAIMS = [("H0", "HINGE0", "A0"), ("HA", "HINGE8", "DIRECT8"), ("HB", "HINGE4k16", "A4k16"),
          ("S0", "SLOW0", "A0"), ("H0S", "HINGE0", "SLOW0")]
REPORT_CLAIMS = {"X": {"H0": [11, 0, 5e-4, "SHOWN"], "HA": [7, 0, 0.008, "SHOWN"], "HB": [7, 2, 0.09, "NOT SHOWN"],
                       "S0": [9, 2, 0.033, "SHOWN"], "H0S": [4, 0, 0.0625, "NOT SHOWN"]},
                 "L": {"H0": [12, 0, 2e-4, "SHOWN"], "HA": [5, 0, 0.031, "SHOWN"], "HB": [11, 2, 0.011, "SHOWN"],
                       "S0": [10, 3, 0.046, "SHOWN"], "H0S": [5, 0, 0.031, "SHOWN"]}}
claims = {"X": {}, "L": {}}
for name, new, old in CLAIMS:
    d = paired(X[new], X[old], SEEDS[new], lambda r, a=new: succ(a, r))
    rb, rc, rp, rv = REPORT_CLAIMS["X"][name]
    claims["X"][name] = dict(new=new, old=old, b=d["b"], c=d["c"], p=d["p"],
                             verdict="SHOWN" if d["p"] < 0.05 else "NOT SHOWN",
                             report=dict(b=rb, c=rc, p=rp, verdict=rv))
    if new in LMAP and old in LMAP:
        okf = (lambda o: o == "DISCOVERED") if PART[new] == "0" else (lambda o: o.startswith("BOUND"))
        nr, orr = lrows[LMAP[new]], lrows[LMAP[old]]
        b = sum(1 for s in SEEDS[new] if okf(nr[s]["outcome"]) and not okf(orr[s]["outcome"]))
        c = sum(1 for s in SEEDS[new] if okf(orr[s]["outcome"]) and not okf(nr[s]["outcome"]))
        p = round(mcnemar_greater(b, c), 6)
    else:
        b = c = p = None
    rb, rc, rp, rv = REPORT_CLAIMS["L"][name]
    claims["L"][name] = dict(new=new, old=old, b=b, c=c, p=p,
                             verdict=None if p is None else ("SHOWN" if p < 0.05 else "NOT SHOWN"),
                             report=dict(b=rb, c=rc, p=rp, verdict=rv))

# failure classes / outcomes per arm
classes = {"X": {a: dict(Counter(tag(a, r) for r in X[a].values())) for a in SEEDS},
           "L": {a: dict(Counter(lrows[LMAP[a]][s]["outcome"] for s in SEEDS[a])) for a in LMAP}}

# per-seed outcome strips (aligned with seeds[part])
per_seed = {"X": {}, "L": {}}
for a in SEEDS:
    per_seed["X"][a] = dict(outcome=[tag(a, X[a][s]) for s in SEEDS[a]],
                            success=[succ(a, X[a][s]) for s in SEEDS[a]],
                            transition=[X[a][s]["transition"] for s in SEEDS[a]],
                            final_acc=[r4(X[a][s]["acc"]) for s in SEEDS[a]])
    if PART[a] in ("B", "C"):
        per_seed["X"][a]["ch_map_end"] = [X[a][s]["end"]["ch_map"] for s in SEEDS[a]]
        per_seed["X"][a]["stream_acc_end"] = [[r4(v, 3) for v in X[a][s]["end"]["stream_acc"]] for s in SEEDS[a]]
for a in LMAP:
    rows = lrows[LMAP[a]]
    per_seed["L"][a] = dict(outcome=[rows[s]["outcome"] for s in SEEDS[a]],
                            transition=[rows[s]["transition"] for s in SEEDS[a]],
                            final_acc=[rows[s]["acc"] for s in SEEDS[a]])
    if PART[a] == "B":
        per_seed["L"][a]["ch_map_end"] = [rows[s]["ch_map"] for s in SEEDS[a]]
        per_seed["L"][a]["stream_acc_end"] = [rows[s]["stream_acc"] for s in SEEDS[a]]

# curves
curves_X = {f"{a}|{s}": acc_list(X[a][s]) for a in SEEDS for s in SEEDS[a]}
curves_L = {f"{a}|{s}": lcurves[LMAP[a]][s]["acc_x100"] for a in LMAP for s in SEEDS[a]}
# sanity: the L log's outcomes on curve rows equal its per-seed rows
for a in LMAP:
    for s in SEEDS[a]:
        assert lcurves[LMAP[a]][s]["outcome"].replace("  ", " ") == lrows[LMAP[a]][s]["outcome"].replace("  ", " "), (a, s)

# hinge firing counts per 1200-update window (end.hs_windows), X
windows_X = {f"{a}|{s}": X[a][s]["end"]["hs_windows"] for a in ("HINGE0", "HINGE8", "HINGE4k16", "HINGE_D8")
             for s in SEEDS[a]}

REP = [
    dict(id="X:A0|294", machine="X", role="position-split failure (plain gate), plateaus at 0.5 = 1/S",
         outcome=tag("A0", X["A0"][294])),
    dict(id="X:SLOW0|294", machine="X", role="slow memory alone, same seed: still a position split",
         outcome=tag("SLOW0", X["SLOW0"][294])),
    dict(id="X:HINGE0|294", machine="X", role="HINGE, same seed: discovered; the hinge fired once in updates 1-1200",
         outcome=tag("HINGE0", X["HINGE0"][294]), hinge_windows=X["HINGE0"][294]["end"]["hs_windows"]),
    dict(id="X:A0|281", machine="X", role="a fast discovered run (plain gate), 0.91 at step 1200",
         outcome=tag("A0", X["A0"][281])),
    dict(id="X:HINGE0|287", machine="X", role="the only HINGE0 failure on either machine (OTHER), hinge fired 950 times",
         outcome=tag("HINGE0", X["HINGE0"][287]), hinge_total=sum(X["HINGE0"][287]["end"]["hs_windows"])),
    dict(id="X:HINGE4k16|243", machine="X", role="k=16, four streams: MERGED (2 share); 0.75-0.77 at steps 8400-12000, then 0.87-0.89 to the end (the merged pair partly separated)",
         outcome=tag("HINGE4k16", X["HINGE4k16"][243])),
    dict(id="X:A4k16|245", machine="X", role="k=16 plain: MERGED (3 share)", outcome=tag("A4k16", X["A4k16"][245])),
    dict(id="X:HINGE4k16|240", machine="X", role="k=16 binder, bound routed at 3600", outcome=tag("HINGE4k16", X["HINGE4k16"][240])),
    dict(id="X:A4k16|241", machine="X", role="k=16 plain: non-stream POSITION failure", outcome=tag("A4k16", X["A4k16"][241])),
    dict(id="L:A4k16|246", machine="L", role="L, k=16 plain: MERGED (2 share), plateau 0.74-0.76 (x100 ints)",
         outcome=lrows["A4k16"][246]["outcome"]),
    dict(id="L:A4k16|242", machine="L", role="L, k=16 plain: MERGED (3 share), plateau 0.50-0.52",
         outcome=lrows["A4k16"][242]["outcome"]),
    dict(id="L:HINGE4k16|242", machine="L", role="L, k=16 HINGE on the same seed: bound routed at 3600",
         outcome=lrows["HINGE4k16"][242]["outcome"]),
]

obj = {
    "_source": {
        "what": "Slow-start test (report Section 7, Table 4): per-arm counts, McNemar claims, failure classes, per-seed "
                "outcomes and held-out accuracy curves.",
        "X": {
            "files": {SS: "arms A0, SLOW0, HINGE0, HINGE8, HINGE4k16, HINGE_D8 (meta.git 9c5939e, 2.80GHz Xeon)",
                      SA: "recorded DIRECT8, seeds 220-239 (paired arm for HINGE8)",
                      SR: "recorded A4k16, seeds 240-259 (paired arm for HINGE4k16)",
                      SC: "recorded D8, seeds 260-265 (paired arm for HINGE_D8)"},
            "keys_used": "runs[ARM|seed].{ok, discovered, transition, acc, collapsed, curve[i][1], end.{margin, eta_key_by_*, "
                         "etak_key_by_*, ch_map, one_to_one, stream_acc, hs_windows}}",
        },
        "L": {
            "file": LLOG,
            "why": "results/L/slow_start_results.json, stream_recipe_results.json, recipe_scope_results.json and "
                   "stream_curriculum_results.json are byte-identical copies of the X files (meta.cpu is the Xeon); L's own "
                   "results files for those tests are not in the repo (report Table 9 says so). L's recipe-scope log prints "
                   "L's recorded slow-start/stream-recipe runs per seed (A0, SLOW0, HINGE0, A4k16, HINGE4k16) with curves "
                   "(accuracy x100, rounded to integers). DIRECT8, HINGE8, D8, HINGE_D8 on L exist only as report numbers.",
            "parsed_sections": "PER-SEED RAW RESULTS (the 'recorded' rows) and CURVES",
        },
        "rules": "DISCOVERED = bound and final VAL cos < 0.5 (Part 0); BOUND = transition not None (held-out acc >= 0.95 and "
                 "stays) (Parts A-C). k=2 failure classes: test_router_layout.py:293 fail_class; k=16: "
                 "test_stream_recipe.py:756 outcome. McNemar: test_conv_lr.py:260 (one-sided exact).",
        "built_by": "video/data/_build/build_slow_start.py (python3, no torch; run from any directory)",
    },
    "eval_every": EVAL_EVERY,
    "curve_step_rule": "curve value i (0-based) is held-out accuracy at step 1200*(i+1) on 2048 queries; a run stops "
                       "early after three consecutive evaluations >= 0.95, so curves have different lengths",
    "parts": {
        "0": dict(S=2, P=4, k=2, conv=False, iters=24000, seeds=S0, outcome="DISCOVERED", arms=["A0", "SLOW0", "HINGE0"]),
        "A": dict(S=2, P=8, k=2, conv=True, iters=28800, seeds=SEEDS["HINGE8"], outcome="BOUND", arms=["DIRECT8", "HINGE8"]),
        "B": dict(S=4, P=4, k=16, conv=True, iters=28800, seeds=SEEDS["HINGE4k16"], outcome="BOUND",
                  arms=["A4k16", "HINGE4k16"]),
        "C": dict(S=8, P=4, k=16, conv=True, iters=28800, seeds_X={"D8": SEEDS["D8"], "HINGE_D8": SEEDS["HINGE_D8"]},
                  outcome="BOUND", arms=["D8", "HINGE_D8"]),
    },
    "seeds": SEEDS,
    "counts": counts,
    "check_vs_report_table4": check,
    "claims": claims,
    "outcome_classes": classes,
    "per_seed": per_seed,
    "curves_X": curves_X,
    "curves_L_x100": curves_L,
    "hinge_windows_X": {"_note": "training batches on which the hinge fired, per 1200-update window (window 0 = updates "
                                 "1-1200); from end.hs_windows", **windows_X},
    "representative": REP,
}

if __name__ == "__main__":
    bad = [c for c in check if c["match"] is False]
    print("mismatches vs Table 4:", bad)
    for m in ("X", "L"):
        for k, v in claims[m].items():
            print(m, k, v)
    write("slow_start.json", obj)
