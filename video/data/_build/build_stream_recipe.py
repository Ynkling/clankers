"""video/data/stream_recipe.json — Table 2 (four streams: spare channels and an early check)."""
from collections import Counter
from common import *

SR = "results/X/stream_recipe_results.json"
sr = load(SR)
lrows, lcurves, _ = parse_llog()
ARMS = ["A4k16", "A4k16_R", "A4k4", "A4k4_R", "ceiling4k16"]
SEEDS = {a: sr["meta"]["seeds"][a] for a in ARMS}
X = {a: runs_of(sr, a) for a in ARMS}
T_CHECK, A_CHECK = sr["meta"]["sched"]["t_check"], sr["meta"]["sched"]["a_check"]   # 4800, 0.4

REPORT_T2 = {"X": {"A4k16": "12/20", "A4k16_R": "17/20", "A4k4": "10/20", "A4k4_R": "10/20", "ceiling4k16": "3/3"},
             "L": {"A4k16": "9/20", "A4k16_R": "19/20", "A4k4": "5/20", "A4k4_R": "8/20", "ceiling4k16": "3/3"}}

counts = {"X": {}, "L": {}}
for a in ARMS:
    rs = X[a]
    counts["X"][a] = dict(bound=sum(bound(r) for r in rs.values()), n=len(rs), from_data=True,
                          routed=sum(bound(r) and routed_k(r) for r in rs.values()),
                          merged_end=sum(shared_max(r["end"]["ch_map"]) >= 2 and not bound(r) for r in rs.values()))
for a in ARMS:
    if a == "A4k16":
        rows = lrows["A4k16"]
        counts["L"][a] = dict(bound=sum(r["outcome"].startswith("BOUND") for r in rows.values()), n=len(rows),
                              from_data=True)
    else:
        b, n = REPORT_T2["L"][a].split("/")
        counts["L"][a] = dict(bound=int(b), n=int(n), from_data=False)
check = [dict(machine=m, arm=a, data=f"{counts[m][a]['bound']}/{counts[m][a]['n']}" if counts[m][a]["from_data"] else None,
              report=REPORT_T2[m][a],
              match=(f"{counts[m][a]['bound']}/{counts[m][a]['n']}" == REPORT_T2[m][a]) if counts[m][a]["from_data"] else None)
         for m in ("X", "L") for a in ARMS]

# R2: every distinct attempt that passed the check (a reused attempt 0 is the plain run)
def passing_attempts(plain, restart):
    out = []
    for s in SEEDS[plain]:
        r = X[plain][s]
        if r["passed"]:
            out.append(dict(seed=s, attempt=0, bound=bound(r), outcome=tagk(r)))
        t = X[restart][s]["trial"]
        for at in t["attempts"]:
            if at["j"] > 0 and at["passed"]:
                rr = X[restart][s]
                assert at is t["attempts"][-1]          # a passing attempt is the continued one
                out.append(dict(seed=s, attempt=at["j"], attempt_seed=at["seed"], bound=bound(rr), outcome=tagk(rr)))
    return out


pa16, pa4 = passing_attempts("A4k16", "A4k16_R"), passing_attempts("A4k4", "A4k4_R")
r2 = dict(k16=dict(passed=len(pa16), bound=sum(p["bound"] for p in pa16),
                   failed_outcomes=dict(Counter(p["outcome"] for p in pa16 if not p["bound"]))),
          k4=dict(passed=len(pa4), bound=sum(p["bound"] for p in pa4),
                  failed_outcomes=dict(Counter(p["outcome"] for p in pa4 if not p["bound"]))),
          report="X: NOT PRECISE (17/20, 0.85); L: PRECISE (19/19). At k=4 every passing-then-failing attempt was a merge "
                 "(9 of 9 on X, 12 of 12 on L); on X the three passing attempts that failed at k=16 were merges.")
r3 = dict(test="Fisher one-sided, A4k16_R vs A4k4_R", X_p=round(fisher_greater(counts["X"]["A4k16_R"]["bound"], 20,
                                                                              counts["X"]["A4k4_R"]["bound"], 20), 4),
          report_X_p=0.020, report_L_p=2e-4)
d4 = paired(X["A4k16"], X["A4k4"], SEEDS["A4k16"], bound)
r4_ = dict(test="McNemar one-sided, A4k16 vs A4k4 (plain)", X=dict(b=d4["b"], c=d4["c"], p=d4["p"]),
           report=dict(X="5 vs 3, p = 0.36", L="8 vs 4, p = 0.19"))

# plateau ladder: unbound runs at the end, final accuracy by how many streams share the most crowded channel
plateau = []
for a in ("A4k16", "A4k4", "A4k16_R", "A4k4_R"):
    for s in SEEDS[a]:
        r = X[a][s]
        if bound(r) or (a.endswith("_R") and r["trial"]["reused"]):
            continue                                     # skip binders and duplicates of the plain run
        e = r["end"]
        plateau.append(dict(run=f"{a}|{s}", k=r["k"], shared=shared_max(e["ch_map"]), ch_map=e["ch_map"],
                            final_acc=r4(r["acc"], 3), stream_acc=[r4(v, 2) for v in e["stream_acc"]], outcome=tagk(r)))
ladder = {}
for p in plateau:
    ladder.setdefault(str(p["shared"]), []).append(p["final_acc"])
ladder = {k: dict(median=statistics.median(v), min=min(v), max=max(v), n=len(v)) for k, v in sorted(ladder.items())}

per_seed = {a: dict(outcome=[tagk(X[a][s]) for s in SEEDS[a]], transition=[X[a][s]["transition"] for s in SEEDS[a]],
                    final_acc=[r4(X[a][s]["acc"], 4) for s in SEEDS[a]],
                    acc_at_4800=[r4(X[a][s]["acc_check"], 4) for s in SEEDS[a]],
                    passed_check=[X[a][s]["passed"] for s in SEEDS[a]],
                    ch_map_end=[X[a][s]["end"]["ch_map"] for s in SEEDS[a]],
                    stream_acc_end=[[r4(v, 3) for v in X[a][s]["end"]["stream_acc"]] for s in SEEDS[a]])
            for a in ARMS}
for a in ("A4k16_R", "A4k4_R"):
    per_seed[a]["attempts"] = [[dict(j=at["j"], seed=at["seed"], acc_4800=r4(at["acc_check"], 4), passed=at["passed"],
                                     early=[r4(c[1], 4) for c in at["early"]]) for at in X[a][s]["trial"]["attempts"]]
                               for s in SEEDS[a]]
    per_seed[a]["attempts_used"] = [X[a][s]["trial"]["used"] for s in SEEDS[a]]

curves_X = {f"{a}|{s}": acc_list(X[a][s]) for a in ARMS for s in SEEDS[a]}
curves_L = {f"A4k16|{s}": lcurves["A4k16"][s]["acc_x100"] for s in SEEDS["A4k16"]}
REP = [
    dict(id="X:A4k16|246", role="k=16 binder: 0.13 at 1200, 1.0 at 2400 (bound 2400, routed)"),
    dict(id="X:A4k16|240", role="k=16 binder at 4800"),
    dict(id="X:A4k4|258", role="k=4 MERGED (2 share): plateau ~0.75 from 4800 to the end; passed the check (0.748) and failed"),
    dict(id="X:A4k4|251", role="k=4 MERGED (2 share): plateau ~0.75; passed the check"),
    dict(id="X:A4k16|256", role="k=16 MERGED (3 share): plateau ~0.5; passed the check (0.41) and failed"),
    dict(id="X:A4k4|244", role="k=4 MERGED (3 share): plateau ~0.5"),
    dict(id="X:A4k16|241", role="k=16 non-stream (all four streams on one channel): stuck near 0.25 = 1/S"),
    dict(id="X:A4k16|250", role="k=16 late binder: sits at ~0.75 (two streams sharing) for 6000-9600, then binds at 10800"),
    dict(id="X:ceiling4k16|240", role="perfect gate k=16: 1.0 at the first evaluation"),
    dict(id="X:A4k16_R|247", role="restart rule: attempts 0-2 fail the check at 4800, attempt 3 passes (0.596) and binds at 15600"),
    dict(id="L:A4k16|246", role="L, k=16 MERGED (2 share): plateau 0.74-0.76 (x100 integers)"),
    dict(id="L:A4k16|242", role="L, k=16 MERGED (3 share): plateau 0.50-0.52"),
]

obj = {
    "_source": {
        "what": "Stream-recipe test (report Section 4, Table 2): S=4, P=4, conv, Adam 1e-3, seeds 240-259; plain and "
                "restart-rule arms at k=16 and k=4; perfect gate k=16 on 240-242.",
        "X": {"file": SR, "meta": "git 092937b, Intel Xeon 2.10GHz, started 2026-09-30 08:55",
              "keys_used": "runs[ARM|seed].{ok, transition, acc, acc_check, passed, curve, end.{ch_map, stream_acc, "
                           "one_to_one, margin, etak_key_by_*}, trial.{attempts[{j, seed, acc_check, passed, early}], used, "
                           "reused}}"},
        "L": {"file": LLOG, "why": "results/L/stream_recipe_results.json is a byte-identical copy of X's (meta.cpu Xeon). "
                                   "Only L's plain A4k16 (as 'the recorded A4k16' in L's recipe-scope log) is available; "
                                   "the other L counts are the report's."},
        "rules": "BOUND = transition not None; outcome classes test_stream_recipe.py:756; check = held-out acc >= 0.4 at "
                 "step 4800; restart = reseed s + 1000*j, at most 5 attempts (test_stream_recipe.py docstring DESIGN).",
        "built_by": "video/data/_build/build_stream_recipe.py (python3, no torch; run from any directory)",
    },
    "eval_every": EVAL_EVERY,
    "curve_step_rule": "curve value i is held-out accuracy at step 1200*(i+1)",
    "check_rule": dict(step=T_CHECK, min_acc=A_CHECK, max_attempts=sr["meta"]["r_max"]),
    "seeds": SEEDS,
    "counts": counts,
    "check_vs_report_table2": check,
    "R2_passing_attempts": r2,
    "R2_attempts_k16_X": pa16,
    "R2_attempts_k4_X": pa4,
    "R3": r3,
    "R4": r4_,
    "plateau_ladder_X": {"_note": "unbound runs at the end (plain arms and the continued restart attempts that are not "
                                  "the plain run), final held-out accuracy grouped by 'shared' = number of streams on the "
                                  "most crowded channel (argmax map at value positions). Inferred reading: with j of S=4 "
                                  "streams merged, the merged ones are answered right about 1/j of the time, so accuracy "
                                  "~ (S - j + 1)/S = 0.75, 0.5, 0.25 for j = 2, 3, 4; some MERGED runs sit higher when "
                                  "the merged streams are partly separated (e.g. A4k16|259: 0.878).",
                         "by_shared": ladder, "runs": plateau},
    "per_seed_X": per_seed,
    "curves_X": curves_X,
    "curves_L_x100": curves_L,
    "representative": REP,
}

if __name__ == "__main__":
    print("mismatches:", [c for c in check if c["match"] is False])
    print("R2", r2)
    print("R3", r3, "R4", r4_)
    print("ladder", ladder)
    write("stream_recipe.json", obj)
