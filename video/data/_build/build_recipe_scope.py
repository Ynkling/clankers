"""video/data/recipe_scope.json — Table 5 (which part of the recipe matters), X from JSON, L from L's log."""
from collections import Counter
from common import *

RS, SS, SR, SC = ("results/X/recipe_scope_results.json", "results/X/slow_start_results.json",
                  "results/X/stream_recipe_results.json", "results/X/stream_curriculum_results.json")
rs, ss, sr, sc = load(RS), load(SS), load(SR), load(SC)
lrows, lcurves, lstage = parse_llog()
S0, SB, SCs = list(range(280, 300)), list(range(240, 260)), list(range(260, 276))
X = {"HONLY0": runs_of(rs, "HONLY0"), "HONLY4k16": runs_of(rs, "HONLY4k16"), "SC8_H": runs_of(rs, "SC8_H"),
     "A0": runs_of(ss, "A0"), "HINGE0": runs_of(ss, "HINGE0"), "HINGE4k16": runs_of(ss, "HINGE4k16"),
     "A4k16": runs_of(sr, "A4k16"), "SC8": {s: r for s, r in runs_of(sc, "SC8").items() if s in SCs}}
PARTS = {"HONLY0": ("0", S0), "A0": ("0", S0), "HINGE0": ("0", S0), "HONLY4k16": ("B", SB), "A4k16": ("B", SB),
         "HINGE4k16": ("B", SB), "SC8_H": ("C", SCs), "SC8": ("C", SCs)}


def succ(a, r):
    return discovered(r) if PARTS[a][0] == "0" else bound(r)


def tag(a, r):
    return tag2(r) if PARTS[a][0] == "0" else tagk(r)


REPORT_T5 = {"X": {"HONLY0": "12/20", "A0": "8/20", "HINGE0": "19/20", "HONLY4k16": "16/20", "A4k16": "12/20",
                   "HINGE4k16": "17/20", "SC8_H": "8/16", "SC8": "6/16"},
             "L": {"HONLY0": "13/20", "A0": "8/20", "HINGE0": "20/20", "HONLY4k16": "18/20", "A4k16": "9/20",
                   "HINGE4k16": "18/20", "SC8_H": "7/20", "SC8": "4/20"}}
counts, check, classes = {"X": {}, "L": {}}, [], {"X": {}, "L": {}}
for a in X:
    sd = PARTS[a][1]
    c = sum(succ(a, X[a][s]) for s in sd)
    counts["X"][a] = dict(success=c, n=len(sd), collapsed=sum(bool(X[a][s]["collapsed"]) for s in sd))
    classes["X"][a] = dict(Counter(tag(a, X[a][s]) for s in sd))
    rows = lrows[a]
    okf = (lambda o: o == "DISCOVERED") if PARTS[a][0] == "0" else (lambda o: o.startswith("BOUND"))
    counts["L"][a] = dict(success=sum(okf(r["outcome"]) for r in rows.values()), n=len(rows),
                          collapsed=sum("collapsed" in r["outcome"] for r in rows.values()))
    classes["L"][a] = dict(Counter(r["outcome"] for r in rows.values()))
    for m in ("X", "L"):
        got = f"{counts[m][a]['success']}/{counts[m][a]['n']}"
        check.append(dict(machine=m, arm=a, data=got, report=REPORT_T5[m][a], match=got == REPORT_T5[m][a]))

CL = [("Q1", "HONLY0", "A0"), ("Q2", "HINGE0", "HONLY0"), ("Q3", "HONLY4k16", "A4k16"), ("Q4", "HINGE4k16", "HONLY4k16"),
      ("Q5", "SC8_H", "SC8")]
REP = {"X": {"Q1": "6 vs 2, p = 0.14", "Q2": "8 vs 1, p = 0.020", "Q3": "6 vs 2, p = 0.14", "Q4": "4 vs 3",
             "Q5": "6 vs 4, p = 0.38"},
       "L": {"Q1": "6 vs 1, p = 0.0625", "Q2": "7 vs 0, p = 0.008", "Q3": "9 vs 0, p = 0.002", "Q4": "2 vs 2",
             "Q5": "6 vs 3, p = 0.25"}}
claims = {"X": {}, "L": {}}
for q, new, old in CL:
    sd = PARTS[new][1]
    d = paired(X[new], X[old], sd, lambda r, a=new: succ(a, r))
    claims["X"][q] = dict(new=new, old=old, b=d["b"], c=d["c"], p=d["p"], report=REP["X"][q])
    okf = (lambda o: o == "DISCOVERED") if PARTS[new][0] == "0" else (lambda o: o.startswith("BOUND"))
    nr, orr = lrows[new], lrows[old]
    sl = sorted(nr)
    b = sum(1 for s in sl if okf(nr[s]["outcome"]) and not okf(orr[s]["outcome"]))
    c = sum(1 for s in sl if okf(orr[s]["outcome"]) and not okf(nr[s]["outcome"]))
    claims["L"][q] = dict(new=new, old=old, b=b, c=c, p=round(mcnemar_greater(b, c), 6), report=REP["L"][q])


def groups_at(r, step):
    x = next((x for x in r["stats"] if x["step"] == step), None)
    if not x:
        return None
    cm = x["ch_map"]
    return sorted((cm.count(c) for c in set(cm)), reverse=True)


two_of_four = sum(1 for s in SCs if groups_at(X["SC8_H"][s], 4800) == [4, 4])

obj = {
    "_source": {
        "what": "Recipe-scope test (report Section 8, Table 5): HONLY = the hinge with one learning rate (no slow phase); "
                "SC8_H = the stream curriculum with HINGE. Recorded arms on the same seeds from slow_start, stream_recipe, "
                "stream_curriculum.",
        "X": {"files": [RS, SS, SR, SC], "keys_used": "runs[ARM|seed].{ok, discovered, transition, collapsed, end, "
                                                       "stats[step=4800].ch_map}"},
        "L": {"file": LLOG, "why": "results/L/recipe_scope_results.json is a byte-identical copy of X's; L's log prints "
                                   "every L run of this test and the recorded L runs it pairs with."},
        "built_by": "video/data/_build/build_recipe_scope.py (python3, no torch; run from any directory)",
    },
    "seeds": {"part0": S0, "partB": SB, "partC_X": SCs, "partC_L": list(range(260, 280))},
    "counts": counts,
    "check_vs_report_table5": check,
    "claims": claims,
    "outcome_classes": classes,
    "SC8_H_two_channels_of_four_at_4800_X": dict(count=two_of_four, n=len(SCs),
                                                  report="On X, nine of 16 SC8_H gates held the eight streams in two "
                                                         "channels of four at step 4800"),
    "curves_X": {f"{a}|{s}": acc_list(X[a][s]) for a in ("HONLY0", "HONLY4k16", "SC8_H") for s in PARTS[a][1]},
    "curves_L_x100": {f"{a}|{s}": lcurves[a][s]["acc_x100"] for a in ("HONLY0", "HONLY4k16", "SC8_H", "SC8")
                      for s in sorted(lcurves[a])},
    "stage_curves_L_x100": {"_note": "SC8_H on L: accuracy on the current stage's stream set at evaluations 1-8 "
                                     "(stages 1-2, updates 1-9600)", **{f"SC8_H|{s}": v for s, v in sorted(lstage.items())}},
}

if __name__ == "__main__":
    print("mismatches:", [c for c in check if not c["match"]])
    for m in claims:
        print(m, claims[m])
    print("two of four", two_of_four)
    print(classes["X"]["HONLY0"], classes["L"]["HONLY0"])
    write("recipe_scope.json", obj)
