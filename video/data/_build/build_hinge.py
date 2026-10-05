"""video/data/hinge_firings.json — when and how often the hinge fires, and how hard it pushes."""
from common import *

lrows, _, _ = parse_llog()
out = {}

# 1. early_recipe: exact firing updates and the hinge/task gradient-norm ratio at each firing (both machines)
er = {}
for m in ("X", "L"):
    d = load(f"results/{m}/early_recipe_results.json")
    arms = {}
    for a in ("WIN_M", "WIN16_A", "WIN16_M"):
        rs = runs_of(d, a)
        runs = {}
        firsts, lasts, ratios = [], [], []
        for s, r in sorted(rs.items()):
            e = r["end"]
            fa = e["hs_fired_at"]
            rat = {dg["update"]: dg["ratio"] for dg in e["hs_firing_diag"]}
            wa = e["hs_would_at"]
            assert sorted(rat) == sorted(fa)
            if fa:
                firsts.append(fa[0]); lasts.append(fa[-1]); ratios += [rat[u] for u in fa]
            if a == "WIN_M":
                oc = tag2(r)
            else:
                oc = tagk(r)
            runs[str(s)] = dict(fired_at=fa, ratio=[round(rat[u], 1) for u in fa],
                                hinge_grad=[r4(next(dg["hinge"] for dg in e["hs_firing_diag"] if dg["update"] == u), 4) for u in fa],
                                task_grad=[float(f"{next(dg['task'] for dg in e['hs_firing_diag'] if dg['update'] == u):.4g}") for u in fa],
                                would_fire_after_2400=len(wa),
                                would_first_last=[wa[0], wa[-1]] if wa else None,
                                outcome=oc, transition=r["transition"])
        arms[a] = dict(runs=runs, n_runs=len(rs), runs_fired=sum(1 for v in runs.values() if v["fired_at"]),
                       total_firings=sum(len(v["fired_at"]) for v in runs.values()),
                       first_firing=med_range(firsts), last_firing=max(lasts) if lasts else None,
                       ratio_all_firings=dict(median=round(statistics.median(ratios), 1), min=round(min(ratios), 1),
                                              max=round(max(ratios), 1), n=len(ratios)),
                       runs_with_would_fires=sum(1 for v in runs.values() if v["would_fire_after_2400"]),
                       would_fires_total=sum(v["would_fire_after_2400"] for v in runs.values()))
    # the SLOW_M failures rescued by WIN_M
    sm, wm = runs_of(d, "SLOW_M"), runs_of(d, "WIN_M")
    resc = []
    for s in sorted(sm):
        if not discovered(sm[s]) and discovered(wm[s]):
            e = wm[s]["end"]
            rat = {dg["update"]: dg["ratio"] for dg in e["hs_firing_diag"]}
            resc.append(dict(seed=s, slow_m=tag2(sm[s]), fired_at=e["hs_fired_at"],
                             ratio=[round(rat[u], 1) for u in e["hs_fired_at"]]))
    er[m] = dict(arms=arms, rescued_by_window=resc)
out["early_recipe"] = {"_note": "fired_at = training updates (1-based) on which a hinge term was > 0 inside the window "
                                "(updates 1-2400); ratio = |hinge gradient| / |task gradient| on the gate (W_in, W_h, W_g "
                                "pooled) at that update; would_fire = updates after 2400 where a term exceeded 0.2 but the "
                                "penalty was off (not firings).", **er}

# 2. slow_start (X): counts per 1200-update window; L from its recipe-scope log (1-1200 / 1-2400 / whole run)
ss = load("results/X/slow_start_results.json")
rs_ = load("results/X/recipe_scope_results.json")
win = {}
for d, arms in ((ss, ("HINGE0", "HINGE8", "HINGE4k16", "HINGE_D8")), (rs_, ("HONLY0", "HONLY4k16", "SC8_H"))):
    for a in arms:
        rr = runs_of(d, a)
        w = {str(s): r["end"]["hs_windows"] for s, r in sorted(rr.items())}
        tot = [sum(v) for v in w.values()]
        first2400 = [sum(v[:2]) for v in w.values()]
        win[a] = dict(windows=w, never_fired=sum(1 for t in tot if t == 0), n=len(tot),
                      total=med_range(tot), in_first_2400=med_range(first2400),
                      outcome={str(s): (tag2(r, part0=a in ("HINGE0", "HONLY0")) if r["k"] == 2 else tagk(r))
                               for s, r in sorted(rr.items())})
out["windows_X"] = {"_note": "results/X slow_start (HINGE0, HINGE8, HINGE4k16, HINGE_D8) and recipe_scope (HONLY0, "
                             "HONLY4k16, SC8_H: no slow phase for HONLY*); end.hs_windows = firings per 1200-update "
                             "window, window 0 = updates 1-1200.", **win}
lw = {}
for a in ("HINGE0", "HINGE4k16", "HONLY0", "HONLY4k16", "SC8_H"):
    rows = lrows[a]
    trip = {str(s): [int(x) for x in r["hinge"].split("/")] for s, r in sorted(rows.items())}
    lw[a] = dict(counts_1200_2400_all=trip, never_fired=sum(1 for v in trip.values() if v[2] == 0), n=len(trip))
out["counts_L"] = {"_note": "from results/L/recipe_scope_L_final.log 'hinge 1200/2400/all' column (L's own runs).", **lw}

out["report_claims"] = {
    "slow_start": "never fired on 11 of X's 20 HINGE0 runs (9 of L's); on X no run saw more than two firings in the first "
                  "2400 updates; X's HINGE0 seed 287 (OTHER failure) fired 950 times (report Section 7)",
    "early_recipe_X": "fired on 19/40 WIN_M runs (first 137 [59, 242], last 358), 8/12 WIN16_A (first 342 [248, 427], "
                      "last 2303), 12/12 WIN16_M (first 331 [140, 546], last 761); ratio WIN_M 3385.6 [7.1, 57759.7], "
                      "WIN16_A 81.5 [12.1, 638188.6], WIN16_M 36.3 [8.6, 110420.5] (test_early_recipe.py docstring)",
    "screens_not_in_repo": "S34's 'median 850x (Muon) / 726x (Adam)' and S32's 'first fired between updates 50 and 168' "
                           "come from the exploratory branch claude/outside-ideas, whose records are not in this repo",
}
out["_source"] = {
    "files": ["results/X/early_recipe_results.json", "results/L/early_recipe_results.json",
              "results/X/slow_start_results.json", "results/X/recipe_scope_results.json", LLOG],
    "keys_used": "end.{hs_fired_at, hs_firing_diag[{update, hinge, task, ratio}], hs_would_at, hs_windows}; L log column "
                 "'hinge 1200/2400/all'",
    "definition": "the hinge fires on a training batch when relu(eta2_index - 0.2) + relu(eta2_half - 0.2) > 0 "
                  "(test_slow_start.py:354 attach_hinge)",
    "built_by": "video/data/_build/build_hinge.py (python3, no torch; run from any directory)",
}

if __name__ == "__main__":
    for m in ("X", "L"):
        for a, v in er[m]["arms"].items():
            print(m, a, v["runs_fired"], v["n_runs"], v["first_firing"], v["last_firing"], v["ratio_all_firings"],
                  v["runs_with_would_fires"], v["would_fires_total"])
        print(m, "rescued", er[m]["rescued_by_window"])
    for a, v in win.items():
        print("X", a, "never", v["never_fired"], "/", v["n"], "total", v["total"], "first2400", v["in_first_2400"])
    for a, v in lw.items():
        print("L", a, "never", v["never_fired"], "/", v["n"])
    print("HINGE0 287 total", sum(win["HINGE0"]["windows"]["287"]))
    write("hinge_firings.json", {"_source": out.pop("_source"), **out})
