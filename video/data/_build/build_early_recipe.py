"""video/data/early_recipe.json — Table 6 (the early-recipe test), both machines from their own files."""
from collections import Counter
from common import *

FILES = {"X": "results/X/early_recipe_results.json", "L": "results/L/early_recipe_results.json"}
D = {m: load(f) for m, f in FILES.items()}
ARMS = ["SLOW_M", "WIN_M", "WIN16_A", "WIN16_M", "CEIL_A", "CEIL_B"]
PARTA = ("SLOW_M", "WIN_M", "CEIL_A")
REPORT_T6 = {"X": {"SLOW_M": "36/40", "WIN_M": "40/40", "WIN16_A": "10/12 (6)", "WIN16_M": "11/12 (10)",
                   "CEIL_A": "2/2", "CEIL_B": "2/2"},
             "L": {"SLOW_M": "34/40", "WIN_M": "40/40", "WIN16_A": "17/20 (12)", "WIN16_M": "19/20 (19)",
                   "CEIL_A": "2/2", "CEIL_B": "2/2"}}
REPORT_E = {"X": {"E1": "4 vs 0, p = 0.0625, NOT SHOWN", "E2": "5 vs 1, p = 0.11, NOT SHOWN"},
            "L": {"E1": "6 vs 0, p = 0.016, SHOWN", "E2": "7 vs 0, p = 0.008, SHOWN"}}


def by4800(r):
    return bool(bound(r) and r["transition"] <= 4800)


def tag(a, r):
    if a in ("WIN16_A", "WIN16_M", "CEIL_B"):
        return tagk(r)
    return tag2(r, part0=True)


def n_channels_at(r, step):
    x = next((x for x in r["stats"] if x["step"] == step), None)
    return None if not x or "ch_map" not in x else len(set(x["ch_map"]))


out = {"counts": {}, "check_vs_report_table6": [], "claims": {}, "per_seed": {}, "transitions": {},
       "curves": {}, "outcome_classes": {}}
for m, d in D.items():
    seeds = d["meta"]["seeds"]
    R = {a: runs_of(d, a) for a in ARMS}
    c = {}
    for a in ARMS:
        rs = R[a]
        if a in PARTA:
            c[a] = dict(success=sum(discovered(r) for r in rs.values()), n=len(rs), outcome="DISCOVERED")
            got = f"{c[a]['success']}/{c[a]['n']}"
        else:
            c[a] = dict(success=sum(bound(r) for r in rs.values()), n=len(rs), outcome="BOUND",
                        bound_by_4800=sum(by4800(r) for r in rs.values()))
            got = f"{c[a]['success']}/{c[a]['n']}" + (f" ({c[a]['bound_by_4800']})" if a in ("WIN16_A", "WIN16_M") else "")
        out["check_vs_report_table6"].append(dict(machine=m, arm=a, data=got, report=REPORT_T6[m][a],
                                                  match=got == REPORT_T6[m][a]))
    out["counts"][m] = c
    e1 = paired(R["WIN_M"], R["SLOW_M"], seeds["WIN_M"], discovered)
    e2 = paired(R["WIN16_M"], R["WIN16_A"], seeds["WIN16_M"], by4800)
    out["claims"][m] = dict(E1=dict(desc="WIN_M beats SLOW_M (DISCOVERED)", b=e1["b"], c=e1["c"], p=e1["p"],
                                    verdict="SHOWN" if e1["p"] < 0.05 else "NOT SHOWN", report=REPORT_E[m]["E1"]),
                            E2=dict(desc="WIN16_M bound by update 4800 more often than WIN16_A", b=e2["b"], c=e2["c"],
                                    p=e2["p"], verdict="SHOWN" if e2["p"] < 0.05 else "NOT SHOWN",
                                    report=REPORT_E[m]["E2"]))
    out["outcome_classes"][m] = {a: dict(Counter(tag(a, r) for r in R[a].values())) for a in ARMS}
    ps = {}
    for a in ARMS:
        ps[a] = dict(seeds=seeds[a], outcome=[tag(a, R[a][s]) for s in seeds[a]],
                     transition=[R[a][s]["transition"] for s in seeds[a]],
                     final_acc=[r4(R[a][s]["acc"]) for s in seeds[a]])
        if a in ("WIN16_A", "WIN16_M"):
            ps[a]["channels_holding_4_streams_at_4800"] = [n_channels_at(R[a][s], 4800) for s in seeds[a]]
            ps[a]["ch_map_at_4800"] = [next((x["ch_map"] for x in R[a][s]["stats"] if x["step"] == 4800), None)
                                       for s in seeds[a]]
    out["per_seed"][m] = ps
    tb = {}
    for a in ("WIN16_A", "WIN16_M"):
        trs = [R[a][s]["transition"] for s in seeds[a]]
        got = [t for t in trs if t is not None]
        tb[a] = dict(transitions=trs, bound=len(got), n=len(trs), median=statistics.median(got),
                     min=min(got), max=max(got),
                     histogram={str(t): got.count(t) for t in sorted(set(got))},
                     four_distinct_channels_at_4800=sum(1 for s in seeds[a] if n_channels_at(R[a][s], 4800) == 4))
    for a in ("SLOW_M", "WIN_M"):
        got = [R[a][s]["transition"] for s in seeds[a] if R[a][s]["transition"] is not None]
        tb[a] = dict(median=statistics.median(got), min=min(got), max=max(got),
                     histogram={str(t): got.count(t) for t in sorted(set(got))})
    out["transitions"][m] = tb
    out["curves"][m] = {f"{a}|{s}": acc_list(R[a][s]) for a in ARMS for s in seeds[a]}

# seed-level agreement between machines (same seeds; report Section 9.2)
agree = {}
for a, fn in (("SLOW_M", discovered), ("WIN_M", discovered), ("WIN16_A", bound), ("WIN16_M", bound)):
    sx, sl = runs_of(D["X"], a), runs_of(D["L"], a)
    common_ = sorted(set(sx) & set(sl))
    agree[a] = dict(n=len(common_), same_outcome=sum(fn(sx[s]) == fn(sl[s]) for s in common_))
agree["WIN16_M_by_4800"] = dict(n=12, same_outcome=sum(by4800(runs_of(D["X"], "WIN16_M")[s]) ==
                                                       by4800(runs_of(D["L"], "WIN16_M")[s]) for s in range(340, 352)))

obj = {
    "_source": {
        "what": "Early-recipe test (report Section 9.2, Table 6). Part A: S=2, P=4, k=2, no conv, seeds 300-339, Muon "
                "without (SLOW_M) and with the hinge window on updates 1-2400 (WIN_M). Part B: S=4, P=4, k=16, conv, "
                "seeds 340-359 (X cut to 340-351 by its time rule), WINDOW under Adam (WIN16_A) and Muon (WIN16_M). "
                "CEIL_A/CEIL_B: perfect gate under Muon (validity).",
        "files": {"X": FILES["X"] + " (meta.git 9f25dc8, 2.80GHz Xeon, NS fingerprint f209ef7b8c61)",
                  "L": FILES["L"] + " (meta.git 9437e01, i7-12650H, NS fingerprint 91a18153b4d5) -- genuine L data, "
                                    "unlike results/L's other Phase V files"},
        "keys_used": "runs[ARM|seed].{ok, discovered, transition, acc, curve, stats[step].ch_map, end.{margin, eta_key_by_*, "
                     "etak_key_by_*, ch_map, stream_acc}}",
        "rules": "DISCOVERED = bound and final VAL cos < 0.5; bound by 4800 = transition <= 4800 "
                 "(test_early_recipe.py:873 by_update); McNemar one-sided exact (test_conv_lr.py:260).",
        "built_by": "video/data/_build/build_early_recipe.py (python3, no torch; run from any directory)",
    },
    "eval_every": EVAL_EVERY,
    "curve_step_rule": "curve value i is held-out accuracy at step 1200*(i+1)",
    **out,
    "seed_level_agreement_X_vs_L": {"_note": "the machines ran the same seeds; report 9.2 quotes 38/40, 40/40, 9 of 12, "
                                             "10 of 12", **agree},
    "representative": [
        dict(id="X:SLOW_M|304", role="Muon without the hinge: POSITION split, plateau ~0.5 for 24000 steps"),
        dict(id="X:WIN_M|304", role="same seed with the hinge window: 0.91 at 1200, discovered at 2400; the hinge fired "
                                    "at updates 180 and 230 (see hinge_firings.json)"),
        dict(id="L:WIN16_M|*", role="Muon four streams: 19/20 bound, median transition 2400"),
        dict(id="L:WIN16_A|*", role="Adam four streams: 17/20 bound, median transition 4800"),
    ],
    "firings": "see hinge_firings.json (early_recipe section): every firing update and the hinge/task gradient ratio",
}

if __name__ == "__main__":
    print("mismatches:", [c for c in out["check_vs_report_table6"] if not c["match"]])
    for m in ("X", "L"):
        print(m, out["claims"][m])
        print(m, {a: {k: v for k, v in t.items() if k != "transitions"} for a, t in out["transitions"][m].items()})
    print(agree)
    write("early_recipe.json", obj)
