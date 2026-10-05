"""video/data/eight_streams.json — S=8 (report Sections 5 and 7 Part C): curriculum vs from scratch, X only."""
from collections import Counter
from common import *

SC, SS, RS = ("results/X/stream_curriculum_results.json", "results/X/slow_start_results.json",
              "results/X/recipe_scope_results.json")
sc, ss, rs = load(SC), load(SS), load(RS)
lrows, lcurves, _ = parse_llog()
R = {"SC8": runs_of(sc, "SC8"), "SC8_R": runs_of(sc, "SC8_R"), "D8": runs_of(sc, "D8"), "SC8_ceil": runs_of(sc, "SC8_ceil"),
     "HINGE_D8": runs_of(ss, "HINGE_D8"), "SC8_H": runs_of(rs, "SC8_H")}
FILE = {"SC8": SC, "SC8_R": SC, "D8": SC, "SC8_ceil": SC, "HINGE_D8": SS, "SC8_H": RS}

arms = {}
for a, rr in R.items():
    seeds = sorted(rr)
    arms[a] = dict(file=FILE[a], seeds=seeds, bound=sum(bound(rr[s]) for s in seeds), n=len(seeds),
                   collapsed=sum(bool(rr[s]["collapsed"]) for s in seeds),
                   outcome=[tagk(rr[s]) for s in seeds], transition=[rr[s]["transition"] for s in seeds],
                   final_acc=[r4(rr[s]["acc"], 4) for s in seeds],
                   ch_map_end=[rr[s]["end"]["ch_map"] for s in seeds],
                   classes=dict(Counter(tagk(rr[s]) for s in seeds)),
                   curves={str(s): acc_list(rr[s]) for s in seeds})
    if "curve_stage" in rr[seeds[0]]:
        arms[a]["stage_curves"] = {str(s): [round(c[1], 4) for c in rr[s]["curve_stage"]] for s in seeds}

REPORT = {"SC8": "6/16", "SC8_R": "7/16", "D8": "0/6", "SC8_ceil": "3/3", "HINGE_D8": "0/10", "SC8_H": "8/16"}
check = [dict(arm=a, data=f"{arms[a]['bound']}/{arms[a]['n']}", report=REPORT[a],
              match=f"{arms[a]['bound']}/{arms[a]['n']}" == REPORT[a]) for a in REPORT]
sc8_tr = sorted(t for t in arms["SC8"]["transition"] if t is not None)

obj = {
    "_source": {
        "what": "Eight streams (S=8, P=4, k=16, conv, 99 tokens): SC8 = stream curriculum (2 streams for updates "
                "1-4800, 4 for 4801-9600, then 8), SC8_R = curriculum + restart on two-stream accuracy at 2400, D8 = all "
                "eight from step 1, SC8_ceil = perfect gate; HINGE_D8 (slow-start Part C) and SC8_H (recipe scope) add "
                "the HINGE recipe.",
        "files": [SC, SS, RS],
        "keys_used": "runs[ARM|seed].{transition (stage-3 transition on the 8-stream set), acc, collapsed, curve (8-stream "
                     "held-out set), curve_stage (the current stage's stream set, stages 1-2), end.ch_map}",
        "machine": "X only (L's stream_curriculum file is not in the repo; L's SC8/SC8_H curves are in recipe_scope.json)",
        "built_by": "video/data/_build/build_eight.py (python3, no torch; run from any directory)",
    },
    "eval_every": EVAL_EVERY,
    "curve_step_rule": "curve value i is held-out accuracy on the 8-stream set at step 1200*(i+1); stage_curves are on "
                       "the stage's own stream set for evaluations 1-8 (steps 1200-9600)",
    "check_vs_report": check,
    "SC8_transitions_X": sc8_tr,
    "report_text": "On X every curriculum binder bound at step 10,800 or 12,000, soon after the switch to eight streams, "
                   "and three of its binders had no one-to-one map. No D8 run bound: all of X's six ended below 0.15. "
                   "HINGE_D8: no binder, 4 of 10 collapsed on each machine, six of ten above 0.15 (up to 0.47 on X).",
    "arms": arms,
}

if __name__ == "__main__":
    print(check)
    print("SC8 transitions", sc8_tr, "binders not routed", sum(1 for o in arms["SC8"]["outcome"] if o == "BOUND NOT routed"))
    print("D8 final", arms["D8"]["final_acc"], "HINGE_D8 final", arms["HINGE_D8"]["final_acc"])
    write("eight_streams.json", obj)
