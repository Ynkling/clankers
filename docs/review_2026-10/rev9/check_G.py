#!/usr/bin/env python3
"""check_G.py — the referee's check of Revision 9's statements about Session G (S55-S57, S71-S73, S82) against the records.

Run from anywhere inside the repository:  python3 docs/review_2026-10/rev9/check_G.py
It writes docs/review_2026-10/rev9/check_G.md and nothing else. Python 3 stdlib + docs/review_2026-10/rlib.py; no torch; no
screen module is imported. Every record and every source file is read with `git show <ref>:<path>`.

Pinned refs
  DRAFT  origin/claude/rev9-draft at 9bfa85a, multichannel_hebbian_report_v9.tex (cited as tex:NNN)
  G      claude/explore-G at 93422cf (report 4; the commit the draft pins): S55-S57 and S71-S73 stores, the screens' code
  F      claude/explore-F at 2247298 (last F commit before 93422cf): F's S70 store rh_lite_results.json
  S82    83699a2 (S82 spec), the last G commit before the draft's commit time, and the fetched G head
  MAIN   the main line, for the S56 specification's origin (docs/reading/*)
Outcome definitions are re-implemented from explore_g_rhl_parent.py's docstring (S71-S73, S82) and explore_g_register.py's
(S56, S57); see the 'definitions' block in the output. The draft's values are typed in below with the line they come from;
the script asserts the cited line contains the quoted text. None of them is used as an input.
"""
import json
import os
import re
import statistics
import subprocess
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))           # docs/review_2026-10
from rlib import load, runs, git, transition, wilson, mcnemar1, fail_class  # noqa: E402

DRAFT, TEXP = "9bfa85a", "multichannel_hebbian_report_v9.tex"
G, F = "93422cf", "2247298"
S82SPEC = "83699a2"
GBR = "origin/claude/explore-G"
MAIN = "origin/claude/bdh-growth-hebbian-inference-w90069"
OUT = os.path.join(HERE, "check_G.md")
GD = "explore_out/G/"

TEX = git("show", f"{DRAFT}:{TEXP}").split("\n")
ROWS = []


def tex(n):
    return TEX[n - 1]


def row(n, needles, item, draft, recomputed, status, note=""):
    for nd in ([needles] if isinstance(needles, str) else needles):
        assert nd in tex(n), f"tex:{n} does not contain {nd!r}"
    assert status in ("DISAGREE", "PARTIAL", "NOT RECORD-CHECKABLE", "MATCH"), status
    ROWS.append((n, item, draft, recomputed, status, note))


def src(ref, path):
    return git("show", f"{ref}:{path}")


def lineno(ref, path, needle):
    for i, l in enumerate(src(ref, path).split("\n"), 1):
        if needle in l:
            return i
    raise AssertionError(f"{needle!r} not in {ref}:{path}")


def f2(x):
    return f"{x:.2f}"


def rng(xs, d=2):
    return f"{min(xs):.{d}f}-{max(xs):.{d}f}" if xs else "-"


def med(xs):
    return statistics.median(xs)


def ctime(ref):
    return git("log", "-1", "--format=%cI", ref).strip()


def short(ref):
    return git("rev-parse", "--short=7", ref).strip()


# ── outcomes ─────────────────────────────────────────────────────────────────
def bound(r):
    return r["transition"] is not None


def routed_star(r):                                  # explore_g_rhl_parent.py:8-9 (code :35, :43-45)
    e = r["end"]
    return bool(e.get("margin", 0.0) >= 0.9 and e.get("eta_key_by_stream", 0.0) > 0.9)


def rb(r):                                           # ROUTED-BOUND
    return bound(r) and routed_star(r)


def disc(r):                                         # F's DISCOVERED
    return bound(r) and r["val_cos"] < 0.5


def mine3(r):                                        # G's BOUND ROUTED, S71-S73
    lay = r["end"]["layers"]
    return bound(r) and all(lay[f"L{l}"]["one_to_one_val"] for l in (1, 2, 3)) and min(r["stream_acc"]) >= 0.9


def br56(r, gated=(2, 3)):                           # G's BOUND ROUTED, S56/S57 (every gated layer)
    lay = r["end"]["layers"]
    return bound(r) and all(lay[f"L{l}"]["one_to_one_val"] for l in gated) and min(r["stream_acc"]) >= 0.9


def cls(r):
    """v1 failure class of the end gate (test_router_layout rule, rlib.fail_class); MERGED as rhl_parent.outcome."""
    c = fail_class(r)
    if c == "STREAM-PARTIAL" and not r["end"]["layers"]["L3"]["one_to_one_val"]:
        return "MERGED"
    return c


def paired(ra, rb_, f):
    ss = sorted(set(ra) & set(rb_))
    b = sum(1 for s in ss if f(ra[s]) and not f(rb_[s]))
    c = sum(1 for s in ss if f(rb_[s]) and not f(ra[s]))
    return b, c, mcnemar1(b, c), len(ss)


def cnt(rr, f):
    return sum(1 for r in rr.values() if f(r))


def trs(rr, f=bound):
    return sorted(r["transition"] for r in rr.values() if f(r))


def at_step(r, step):
    for s in r["stats"]:
        if s["step"] == step:
            return s
    return None


# ── stores ───────────────────────────────────────────────────────────────────
S55 = load(G, GD + "s55_probe_results.json")
S56 = load(G, GD + "s56_register_results.json")
S57 = load(G, GD + "s57_latch_results.json")
S71 = load(G, GD + "s71_layers_results.json")
S72 = load(G, GD + "s72_register3_results.json")
S73 = load(G, GD + "s73_register3_delta_results.json")
F70 = load(F, "explore_out/F/rh_lite_results.json")
A, A7 = {}, {}                                       # S55-S57 arms; S71-S73 arms (BASELINE / BASELINE_D occur in both)
for st, D in ((S55, A), (S56, A), (S57, A), (S71, A7), (S72, A7), (S73, A7)):
    for arm in sorted({k.split("|")[0] for k in st["runs"]}):
        assert arm not in D, arm
        D[arm] = runs(st, arm)
FA = {a: runs(F70, a) for a in ("S70_HEBB", "S70_DELTA")}

integrity = []
for name, st in (("S55", S55), ("S56", S56), ("S57", S57), ("S71", S71), ("S72", S72), ("S73", S73)):
    rr = list(st["runs"].values())
    integ = sum(1 for r in rr if r.get("ok")), sum(1 for r in rr if r.get("attempt", 1) != 1), \
        sum(1 for r in rr if r["transition"] == transition(r["curve"]))
    integrity.append(f"{name}: {len(rr)} records, {integ[0]} ok, {integ[1]} retried, recorded transition = "
                     f"rlib.transition(curve) in {integ[2]}")
    assert integ[0] == len(rr) and integ[2] == len(rr)

# ═════════════════════════════════════════════════════════════════════════════
# S55 (tex:259, :59, :355)
# ═════════════════════════════════════════════════════════════════════════════
LAY = ("L2", "L3", "final")


def pmed(arm, key="probe"):
    return {f"{l}|{k}": med([r[key][f"{l}|{k}"]["acc"] for r in A[arm].values()]) for l in LAY for k in "KV"}


m55 = pmed("L3S_P8")
m55i = pmed("L3S_P8", "probe_init")
m55o = pmed("ORACLE_P8")
absent = all(v <= 0.6 for v in m55.values())
present_any = any(m55[f"{l}|K"] >= 0.9 and m55[f"{l}|V"] >= 0.9 for l in LAY)
row(259, "$0.55$--$0.58$ at layers 2 and 3 and at the output", "S55 L3S_P8 (trained window-gate stack, P = 8) probe medians, "
    "L2 / L3 / final, K and V", "0.55-0.58", f"{rng(m55.values())} ({', '.join(f'{k} {f2(v)}' for k, v in m55.items())})",
    "MATCH" if f"{min(m55.values()):.2f}" == "0.55" and f"{max(m55.values()):.2f}" == "0.58" else "DISAGREE",
    "median over the 10 runs (seeds 360-369) of the held-out accuracy, as the S55 docstring reads it")
row(259, "from about $0.70$ at initialisation", "S55 L3S_P8 probe medians at initialisation (same seeds), L2 / L3 / final",
    "about 0.70", f"K {rng([m55i[f'{l}|K'] for l in LAY])}, V {rng([m55i[f'{l}|V'] for l in LAY])}",
    "MATCH", "'about 0.70' holds for K; V starts at 0.66")
drop = sum(1 for r in A["L3S_P8"].values() for l in LAY for k in "KV" if r["probe"][f"{l}|{k}"]["acc"] < r["probe_init"][f"{l}|{k}"]["acc"])
row(259, "Training removes the cue", "trained vs initial probe, L3S_P8, per run and site (10 runs x 3 layers x K/V)",
    "training removes the cue", f"trained < init in {drop}/60 run-sites", "MATCH" if drop >= 54 else "PARTIAL",
    "descriptive (report_1.md 'Anything unexpected'); the causal clause that follows ('the gate reads only the window, nothing "
    "asks the stack to keep it') is an interpretation, unmarked (see the last section)")
row(259, "the perfect gate's stack carries nothing either ($0.54$)", "S55 ORACLE_P8 (perfect gate, 2 runs) probe medians "
    "L2 / L3 / final, K and V", "0.54", f"{rng(m55o.values())} ({', '.join(f'{k} {f2(v)}' for k, v in m55o.items())})",
    "PARTIAL", "0.54 is the layer-2 value; over the three sites the medians span 0.53-0.56. Two runs, one of which bound "
    "(360, at 24000)")
row(259, "The pre-fixed reading ``absent'' applied", "S55 reading for L3S_P8 (explore_g_probe.py docstring: median <= 0.6 at K "
    "and V at every l in {L2, L3, final})", "'absent' applied", f"max median {max(m55.values()):.2f} <= 0.6: "
    f"{'absent' if absent else 'not absent'}", "MATCH" if absent else "DISAGREE")
row(59, "the cue is absent", "index row: S55 reading", "the cue is absent", "'absent' (L3S_P8)", "MATCH" if absent else "DISAGREE")

# ═════════════════════════════════════════════════════════════════════════════
# S56 / S57 (tex:261)
# ═════════════════════════════════════════════════════════════════════════════
heb56 = ("BASELINE", "RESGATE_ONLY", "REG", "REG_NOSG", "REG_NUDGE", "SINGLE_GDN")
heb57 = ("BASELINE_L", "LATCH", "LATCH_NUDGE", "LATCH_PRIOR")
bc = {a: cnt(A[a], bound) for a in heb56 + heb57}
row(261, "every Hebbian arm bound $0/10$", "S56 Hebbian arms (370-379) and S57 latch arms (410-419), BOUND",
    "0/10 (baseline, residual gate alone, register with / without SG, nudged register)",
    "; ".join(f"{a} {bc[a]}/{len(A[a])}" for a in heb56 + heb57), "MATCH" if not any(bc.values()) else "DISAGREE",
    "also true of the arms the draft does not list (SINGLE_GDN, S57's four)")
rp = json.loads(src(G, GD + "s56_read_path.json"))
resl = {tuple(r["res_layers"]) for a in ("REG", "REG_NUDGE", "RESGATE_ONLY") for r in A[a].values()}
row(261, "With the register gate at layers 2 and 3 and the window gate at layer 1", "S56 gated layers / read path",
    "layers 2-3, window gate at layer 1", f"explore_g_regmodel.py GATED = (2, 3) (line {lineno(G, 'explore_g_regmodel.py', 'GATED = (2, 3)')}); "
    f"s56_read_path.json res_layers {rp['res_layers']}; records' res_layers {sorted(resl)}", "MATCH")
nb = [r["end"]["beta"] for r in A["REG_NUDGE"].values()]
row(261, "$0.96$ at the context token against $\\leq 0.05$ at keys and values", "S56 REG_NUDGE end beta^R CTX / KEY / VAL",
    "0.96 vs <= 0.05", f"median {f2(med([b['CTX'] for b in nb]))} / {f2(med([b['KEY'] for b in nb]))} / "
    f"{f2(med([b['VAL'] for b in nb]))}; per run CTX {rng([b['CTX'] for b in nb])}, KEY {rng([b['KEY'] for b in nb])}, "
    f"VAL {rng([b['VAL'] for b in nb])}",
    "MATCH" if all(b['CTX'] >= 0.895 and b['KEY'] <= 0.1 and b['VAL'] <= 0.1 for b in nb) else "PARTIAL",
    "the numbers are the medians over runs (report 2's diagnostics); a clean latch (CTX >= 0.9, KEY and VAL <= 0.1) in "
    f"{sum(1 for b in nb if b['CTX'] >= 0.895 and b['KEY'] <= 0.1 and b['VAL'] <= 0.1)}/10 runs; not: "
    + ", ".join(f"{s} (CTX {r['end']['beta']['CTX']:.2f})" for s, r in sorted(A['REG_NUDGE'].items())
                if not (r['end']['beta']['CTX'] >= 0.895 and r['end']['beta']['KEY'] <= 0.1 and r['end']['beta']['VAL'] <= 0.1)))
mg = [(r["end"]["layers"][f"L{l}"]["margin"], r["end"]["layers"][f"L{l}"]["one_to_one_val"]) for r in A["REG_NUDGE"].values() for l in (2, 3)]
mgr = {s: (r["end"]["layers"]["L2"]["margin"], r["end"]["layers"]["L3"]["margin"], r["end"]["beta"]["CTX"])
       for s, r in sorted(A["REG_NUDGE"].items())}
mgbad = {s: v for s, v in mgr.items() if min(v[:2]) < 0.99}
row(261, "its gates route one-to-one with margin $1.00$ at layers 2 and 3", "S56 REG_NUDGE end margin and VAL map at L2, L3 "
    "(every run)", "one-to-one, margin 1.00", f"margin {rng([m for m, _ in mg])}; one-to-one {sum(o for _, o in mg)}/{len(mg)} "
    f"(run x layer); median L2 / L3 {f2(med([v[0] for v in mgr.values()]))} / {f2(med([v[1] for v in mgr.values()]))}; runs below 0.99: "
    + ", ".join(f"{s} (L2 {v[0]:.2f}, L3 {v[1]:.2f}; beta^R CTX {v[2]:.2f})" for s, v in mgbad.items()),
    "MATCH" if min(m for m, _ in mg) >= 0.995 and all(o for _, o in mg) else "PARTIAL",
    f"true of the median run (report_2.md diagnostics give medians) and of {10 - len(mgbad)}/10 runs")
sa = {s: r["stream_acc"] for s, r in sorted(A["REG_NUDGE"].items())}
flat32 = [s for s, v in sa.items() if all(abs(x - 0.32) <= 0.03 + 1e-9 for x in v)]
row(261, "its accuracy plateaus at $0.32$ on \\emph{both} streams in every run", "S56 REG_NUDGE end stream accuracies "
    "(10 runs)", "0.32 on both streams, every run", f"stream acc {rng([x for v in sa.values() for x in v])}; both streams within "
    f"0.32 +- 0.03 in {len(flat32)}/10 runs (not: {sorted(set(sa) - set(flat32))}); run acc {rng([r['acc'] for r in A['REG_NUDGE'].values()])}; per run "
    + ", ".join(f"{s} {v[0]:.2f}/{v[1]:.2f}" for s, v in sa.items()),
    "MATCH" if len(flat32) == 10 else "DISAGREE",
    "report_2.md:86 itself says 'plateaus at acc 0.32-0.44'; 0.32 is the typical value, not every run's")
bacc = [r["acc"] for r in A["BASELINE"].values()]
bmarg = [r["end"]["margin"] for r in A["BASELINE"].values()]
row(261, "no better than the unrouted baseline ($0.28$--$0.42$)", "S56 BASELINE end accuracy (10 runs); routing margin",
    "0.28-0.42, unrouted", f"acc {rng(bacc)}; end margin {rng(bmarg)}; REG_NUDGE acc {rng([r['acc'] for r in A['REG_NUDGE'].values()])}",
    "MATCH" if f"{min(bacc):.2f}" == "0.28" and f"{max(bacc):.2f}" == "0.42" else "PARTIAL")
o23 = A["ORACLE23"]
oh = A["ORACLE_H"]
b, c, p, n = paired(oh, o23, bound)
row(261, ["perfect gate at layers 2--3 alone bound $0/3$", "one of two seeds, one paired seed"],
    "S57 post-hoc ORACLE23 (370-372) vs S56 ORACLE_H (370-371), BOUND", "0/3 vs 1 of 2; one paired seed",
    f"ORACLE23 {cnt(o23, bound)}/{len(o23)}; ORACLE_H {cnt(oh, bound)}/{len(oh)} (bound seed {[s for s, r in oh.items() if bound(r)]}); "
    f"paired on {n} seeds: {b} vs {c}", "MATCH" if (cnt(o23, bound), cnt(oh, bound), b, c) == (0, 1, 1, 0) else "DISAGREE",
    "ORACLE23 is post hoc (s57 meta arms: post_hoc true)")
nd = A["REG_NUDGE_D"]
row(261, "the nudged register $8/10$, routed $7/10$, transitions 3600--6000", "S56 REG_NUDGE_D BOUND, BOUND ROUTED (G's: one-to-one "
    "VAL map at layers 2 and 3, every stream >= 0.9), transitions", "8/10, 7/10, 3600-6000",
    f"BOUND {cnt(nd, bound)}/10, BOUND ROUTED {cnt(nd, br56)}/10, transitions (bound) {trs(nd)[0]}-{trs(nd)[-1]}",
    "MATCH" if (cnt(nd, bound), cnt(nd, br56), trs(nd)[0], trs(nd)[-1]) == (8, 7, 3600, 6000) else "DISAGREE")
rd, bd = A["REG_D"], A["BASELINE_D"]
b, c, p, n = paired(rd, bd, bound)
row(261, "the unnudged register $6/10$ bound against the baseline's $2/10$ (4 vs 0, $p = 0.0625$)",
    "S56 REG_D vs BASELINE_D (390-399), BOUND; paired one-sided McNemar", "6/10 vs 2/10, 4 vs 0, p = 0.0625",
    f"{cnt(rd, bound)}/10 vs {cnt(bd, bound)}/10; {b} vs {c}, p = {p:.4f}",
    "MATCH" if (cnt(rd, bound), cnt(bd, bound), b, c) == (6, 2, 4, 0) else "DISAGREE")
bnr = {s: r for s, r in rd.items() if bound(r) and not br56(r)}
ks2 = sorted(s for s, r in bnr.items() if r["end"]["layers"]["L2"]["eta_key_by_key"] >= 0.5)
ks3 = sorted(s for s, r in bnr.items() if r["end"]["layers"]["L3"]["eta_key_by_key"] >= 0.5)
row(261, "only two of its six routed, three binding through key splits", "S56 REG_D bound runs: BOUND ROUTED; bound unrouted runs "
    "with eta^2 by key at KEY >= 0.5", "2 of 6 routed; 3 key-split bindings",
    f"BOUND ROUTED {cnt(rd, br56)} ({sorted(s for s, r in rd.items() if br56(r))}); key split at L2 {len(ks2)} {ks2}, at L3 {len(ks3)} {ks3} "
    f"(of the {len(bnr)} bound unrouted {sorted(bnr)})", "MATCH" if cnt(rd, br56) == 2 and len(ks2) == 3 else "PARTIAL",
    "as report_2.md (layer 2); the S56 R2 reading that applied to REG_D is not stated by the draft (see 'Readings vs docstring')")
orc = {"ORACLE_P8 (S55)": A["ORACLE_P8"], "ORACLE_H (S56)": A["ORACLE_H"], "ORACLE_L (S57)": A["ORACLE_L"]}
tot = sum(cnt(v, bound) for v in orc.values())
row(261, "3 of 6 seeds over the three screens", "Hebbian perfect gate on P = 8, 24000 updates: S55 / S56 / S57 oracles, BOUND",
    "3 of 6", "; ".join(f"{k} {cnt(v, bound)}/{len(v)} ({trs(v)})" for k, v in orc.items()) + f"; total {tot}/6",
    "MATCH" if tot == 3 else "DISAGREE")

# ═════════════════════════════════════════════════════════════════════════════
# S71 (tex:261, table)
# ═════════════════════════════════════════════════════════════════════════════
OA, O23, O1, OAD, O23D = (A7[a] for a in ("ORACLE_ALL", "ORACLE_23", "ORACLE_1", "ORACLE_ALL_D", "ORACLE_23_D"))
row(261, "whose perfect gates bind at 1200 (S71)", "S71 ORACLE_ALL / ORACLE_ALL_D transitions", "bind at 1200",
    f"ORACLE_ALL median {med(trs(OA)):.0f} ({trs(OA)[0]}-{trs(OA)[-1]}; {sum(t == 1200 for t in trs(OA))}/10 at 1200); "
    f"ORACLE_ALL_D {trs(OAD)[0]}-{trs(OAD)[-1]}", "PARTIAL" if trs(OA)[-1] > 1200 else "MATCH",
    "the Hebbian perfect gate binds at 1200 in the median run, not in every run")
row(261, ["perfect gate at all three layers binds $10/10$", "at layers 2--3 alone $0/10$"], "S71 ORACLE_ALL, ORACLE_23 BOUND",
    "10/10, 0/10", f"{cnt(OA, bound)}/10, {cnt(O23, bound)}/10", "MATCH" if (cnt(OA, bound), cnt(O23, bound)) == (10, 0) else "DISAGREE")
cells23 = {s: [c_["acc"] for c_ in r["acc_by_cell"].values()] for s, r in sorted(O23.items())}
keys = sorted(next(iter(O23.values()))["acc_by_cell"])
cellmed = [med([O23[s]["acc_by_cell"][k]["acc"] for s in O23]) for k in keys]
acc23 = {s: r["acc"] for s, r in sorted(O23.items())}
row(261, ["every run at exactly chance between the two streams", "$0.46$--$0.51$ in every (stream, block, pair) cell"],
    "S71 ORACLE_23: per-run accuracy and per-cell accuracy (12 cells, 2048 queries)", "every run at chance; 0.46-0.51 in every cell",
    f"per-cell medians over runs {rng(cellmed)}; per-run final acc {rng(acc23.values())} "
    f"({', '.join(f'{s} {a:.2f}' for s, a in acc23.items())}); per-run cells {rng([x for v in cells23.values() for x in v])}; "
    f"runs with every cell in 0.46-0.51: {sum(all(0.455 <= x < 0.515 for x in v) for v in cells23.values())}/10",
    "DISAGREE",
    "0.46-0.51 is the range of the per-cell MEDIANS over the 10 runs (report_3.md:38); no single run is in 0.46-0.51 in "
    "every cell; run accuracies are bimodal (~0.32 and ~0.53), so 'every run at exactly chance' does not hold")
mg23 = [r["end"]["layers"][f"L{l}"]["margin"] for r in O23.values() for l in (2, 3)]
row(261, "with layers 2--3 routed perfectly", "S71 ORACLE_23 end margin at L2, L3", "routed perfectly", f"margin {rng(mg23)}",
    "MATCH" if min(mg23) >= 0.995 else "PARTIAL", "perfect gates by construction (CHECK in checks_s71)")
row(261, "at layer~1 alone $9/10$, slowly (median 16{,}800)", "S71 ORACLE_1 BOUND, median transition", "9/10, median 16800",
    f"{cnt(O1, bound)}/10, median {med(trs(O1)):.0f} ({trs(O1)[0]}-{trs(O1)[-1]})",
    "MATCH" if cnt(O1, bound) == 9 and med(trs(O1)) == 16800 else "DISAGREE")
row(261, "on delta memory the perfect gate at layers 2--3 alone binds $10/10$", "S71 ORACLE_23_D BOUND", "10/10",
    f"{cnt(O23D, bound)}/10", "MATCH" if cnt(O23D, bound) == 10 else "DISAGREE")
b1, c1, p1, _ = paired(OA, O23, bound)
b2, c2_, _, _ = paired(O1, OA, bound)          # A = ORACLE_1, B = ORACLE_ALL: (B only) - (A only) <= 1
b3, c3, _, _ = paired(O23D, OAD, bound)
r_block = p1 < 0.05
r_alone = c2_ - b2 <= 1
r_tol = c3 - b3 <= 1
row(261, ["``layer~1 is the block'' (10 vs 0, $p = 0.00098$)", "``layer~1 alone suffices''", "``delta tolerates an unrouted layer~1''",
          "all apply"], "S71 readings (BOUND, paired on 700-709; explore_g_layers.py:21-26)",
    "block 10 vs 0 p = 0.00098; alone suffices; delta tolerates: all apply",
    f"block: {b1} vs {c1}, p = {p1:.5f} -> {'applies' if r_block else 'no'}; alone suffices: ALL only {c2_}, ORACLE_1 only {b2} -> "
    f"{'applies' if r_alone else 'no'}; delta tolerates: ALL_D only {c3}, 23_D only {b3} -> {'applies' if r_tol else 'no'}",
    "MATCH" if (b1, c1, round(p1, 5), r_alone, r_tol) == (10, 0, 0.00098, True, True) else "DISAGREE",
    "validity: ORACLE_ALL and ORACLE_ALL_D 10/10 >= 8/10, both families VALID")
row(59, "layer~1 is the block on Hebbian memory", "index row: S71 reading", "layer 1 is the block", "applies (10 vs 0)",
    "MATCH" if r_block else "DISAGREE")

# ═════════════════════════════════════════════════════════════════════════════
# S72 / S73 (tex:263)
# ═════════════════════════════════════════════════════════════════════════════
BH, R3, R3N = A7["BASELINE"], A7["REG3"], A7["REG3_NUDGE"]
BD, R3D, R3ND = A7["BASELINE_D"], A7["REG3_D"], A7["REG3_NUDGE_D"]
RES, L3D, L3ND, SGL = A7["RESGATE_ONLY_D"], A7["LATCH3_D"], A7["LATCH3_NUDGE_D"], A7["SINGLE_D"]
assert len(BH) == 20 and len(BD) == 20 and len(A["BASELINE"]) == 10 and len(A["BASELINE_D"]) == 10
row(263, "20 seeds, readings on ROUTED-BOUND", "S72/S73 seeds and the readings' outcome", "20 seeds; ROUTED-BOUND",
    f"BASELINE, REG3, REG3_NUDGE, BASELINE_D, REG3_D on 700-719 ({len(BH)}, {len(R3)}, {len(R3N)}, {len(BD)}, {len(R3D)} runs); "
    f"REG3_NUDGE_D, RESGATE_ONLY_D, LATCH3_D, LATCH3_NUDGE_D, SINGLE_D on 700-709; readings on ROUTED-BOUND "
    f"(register3.py:{lineno(G, 'explore_g_register3.py', 'ROUTED-BOUND')}, register3_delta.py:{lineno(G, 'explore_g_register3_delta.py', 'ROUTED-BOUND')})",
    "MATCH", "the 10-seed arms are on 700-709; the draft gives their n with each count")
row(263, "$\\mathbf{20/20}$ on Hebbian memory (transitions 2400--8400)", "S72 REG3_NUDGE ROUTED-BOUND; transitions",
    "20/20; 2400-8400", f"{cnt(R3N, rb)}/20; {trs(R3N)[0]}-{trs(R3N)[-1]}",
    "MATCH" if (cnt(R3N, rb), trs(R3N)[0], trs(R3N)[-1]) == (20, 2400, 8400) else "DISAGREE")
row(263, "$\\mathbf{10/10}$ on delta (all at 1200)", "S73 REG3_NUDGE_D ROUTED-BOUND; transitions", "10/10, all at 1200",
    f"{cnt(R3ND, rb)}/10; {sorted(set(trs(R3ND)))}", "MATCH" if cnt(R3ND, rb) == 10 and set(trs(R3ND)) == {1200} else "DISAGREE")
nm = [r["end"]["layers"][f"L{l}"]["margin"] for rr in (R3N, R3ND) for r in rr.values() for l in (1, 2, 3)]
n11 = sum(all(r["end"]["layers"][f"L{l}"]["one_to_one_val"] for l in (1, 2, 3)) for rr in (R3N, R3ND) for r in rr.values())
row(263, "one-to-one with margin $1.00$ at every layer", "nudged REG3 / REG3_NUDGE_D: VAL map and margin at L1-L3, every run",
    "one-to-one, margin 1.00, every layer", f"one-to-one at every layer {n11}/30 runs; margin {rng(nm, 3)}",
    "MATCH" if n11 == 30 and min(nm) >= 0.985 else "PARTIAL", "lowest margin rounds to 0.99 (rounding only)" if min(nm) < 0.995 else "")
row(263, ["on $4/20$ seeds on each memory, against $0/20$ for either baseline", "4 vs 0, $p = 0.0625$"],
    "REG3 vs BASELINE (S72), REG3_D vs BASELINE_D (S73), ROUTED-BOUND", "4/20 each vs 0/20; 4 vs 0, p = 0.0625 each",
    f"S72 {cnt(R3, rb)}/20 vs {cnt(BH, rb)}/20, {paired(R3, BH, rb)[:2]}, p = {paired(R3, BH, rb)[2]:.4f}; S73 {cnt(R3D, rb)}/20 vs "
    f"{cnt(BD, rb)}/20, {paired(R3D, BD, rb)[:2]}, p = {paired(R3D, BD, rb)[2]:.4f}",
    "MATCH" if paired(R3, BH, rb)[:2] == (4, 0) and paired(R3D, BD, rb)[:2] == (4, 0) else "DISAGREE")
row(59, "$4/20$ label-free", "index row: label-free register", "4/20 label-free", f"REG3 {cnt(R3, rb)}/20, REG3_D {cnt(R3D, rb)}/20 ROUTED-BOUND; "
    "pre-fixed readings 'not discovered' (S72) / 'discovered (register)' does not apply (S73)", "MATCH",
    "the count is right; the index row's findings column does not say that its reading did not apply")
row(59, "the register carries the cue $20/20$ nudged", "index row: S72 R1", "the register carries the cue, 20/20 nudged",
    f"S72 R1 'the register carries the cue at all three layers' applies ({cnt(R3N, rb)}/20 >= 17/20)", "MATCH")
dm = {(a, s): [r["end"]["layers"][f"L{l}"]["margin"] for l in (1, 2, 3)] for a, rr in (("REG3", R3), ("REG3_D", R3D))
      for s, r in rr.items() if rb(r)}
row(263, "each discovering run routed perfectly", "ROUTED-BOUND REG3 / REG3_D runs: VAL map and margin at L1-L3",
    "routed perfectly", f"{len(dm)} runs {sorted(dm)}; margin {rng([x for v in dm.values() for x in v], 3)}; one-to-one at every layer "
    f"{sum(mine3(r) for rr in (R3, R3D) for r in rr.values() if rb(r))}/{len(dm)}",
    "MATCH" if min(x for v in dm.values() for x in v) >= 0.985 else "PARTIAL", "lowest margin rounds to 0.99 (rounding only)")
pb = paired(R3, BH, rb)[0] + paired(R3D, BD, rb)[0]
pc = paired(R3, BH, rb)[1] + paired(R3D, BD, rb)[1]
row(263, "8 vs 0 over the 40 pairs is a description, not a test", "pooled discordant pairs, both screens", "8 vs 0 over 40",
    f"{pb} vs {pc} over {len(R3) + len(R3D)} pairs (one-sided McNemar would give {mcnemar1(pb, pc):.4f})",
    "MATCH" if (pb, pc) == (8, 0) else "DISAGREE", "labelled a description; not a pre-fixed reading")
bdD = paired(R3D, BD, disc)
extra = sorted(s for s, r in R3D.items() if disc(r) and not rb(r))
row(263, ["5 vs 0 ($p = 0.031$)", "margin of $0.66$"], "S73 REG3_D vs BASELINE_D on F's DISCOVERED; the run counted by DISCOVERED "
    "and not by ROUTED-BOUND", "5 vs 0, p = 0.031; margin 0.66", f"{bdD[0]} vs {bdD[1]}, p = {bdD[2]:.4f}; extra run(s) {extra} margin "
    f"{', '.join(f2(R3D[s]['end']['margin']) for s in extra)}, bound {[bound(R3D[s]) for s in extra]}",
    "MATCH" if bdD[:2] == (5, 0) and len(extra) == 1 and f2(R3D[extra[0]]["end"]["margin"]) == "0.66" else "DISAGREE")


def beta_rng(rr):
    bs = [r["end"]["beta"] for r in rr.values() if rb(r)]
    return [b_["CTX"] for b_ in bs], [b_[k] for b_ in bs for k in ("KEY", "VAL")], all(b_["CTX"] > max(b_["KEY"], b_["VAL"]) for b_ in bs)


hc, hkv, hall = beta_rng(R3)
dc, dkv, dall = beta_rng(R3D)
row(263, ["Hebbian $0.32$--$0.64$ against $0.16$--$0.24$", "delta $0.36$--$0.58$ against about $0.2$"],
    "end beta^R at CTX vs KEY/VAL in the ROUTED-BOUND unnudged runs", "CTX above K/V in every routed run; H 0.32-0.64 vs "
    "0.16-0.24; D 0.36-0.58 vs ~0.2", f"every run: {hall and dall}; H CTX {rng(hc)} vs {rng(hkv)}; D CTX {rng(dc)} vs {rng(dkv)}",
    "MATCH" if hall and dall and rng(hc) == "0.32-0.64" and rng(hkv) == "0.16-0.24" and rng(dc) == "0.36-0.58" else "DISAGREE",
    "marked 'Descriptively'")
# key splits
kc = {}
for nm_, rr in (("Hebbian REG3", R3), ("delta REG3_D", R3D)):
    unb = sum(1 for r in rr.values() if not bound(r) and cls(r) == "KEY")
    allk = sum(1 for r in rr.values() if cls(r) == "KEY")
    fail = [r for r in rr.values() if not rb(r)]
    kf = sum(1 for r in fail if cls(r) == "KEY")
    kb = sorted(s for s, r in rr.items() if bound(r) and cls(r) == "KEY")
    kc[nm_] = (unb, allk, len(fail), kf, kb, Counter(cls(r) for r in fail))
row(263, "KEY 14 of 20 on each memory by class", "REG3 / REG3_D key splits (v1 class of the end gate): printed (unbound only), "
    "all 20 runs, and among the runs that are not ROUTED-BOUND", "KEY 14 of 20 on each memory",
    "; ".join(f"{k}: unbound-only {v[0]}, all runs {v[1]}/20, among the {v[2]} not ROUTED-BOUND {v[3]} ({dict(v[5])}); "
              f"bound key-split runs {v[4]}" for k, v in kc.items()),
    "PARTIAL",
    "the draft's 14 and 14 are the counts over ALL 20 runs (= among the 16 failures), i.e. the referee's earlier recount, not "
    "the class table the screens print (unbound runs only: 14 Hebbian, 12 delta, explore_g_rhl_parent.py count_table). Read as "
    "'14 of the 20 runs on each memory are key splits' it is right; but the sentence's subject is 'the runs that failed' "
    "(16 per memory), and on delta 2 of the 14 (700, 709) bound by a key split rather than failed to bind. The class table "
    "the screens print gives 12 for delta, so a reader checking the store against '14' will not find it without the population")
ksr = [(s, r["end"]["beta"]) for rr in (R3, R3D) for s, r in rr.items() if cls(r) == "KEY"]
flatA = sum(1 for _, b_ in ksr if max(b_[k] for k in ("CTX", "KEY", "VAL")) - min(b_[k] for k in ("CTX", "KEY", "VAL")) <= 0.10)
flatB = sum(1 for _, b_ in ksr if all(0.15 <= b_[k] <= 0.35 for k in ("CTX", "KEY", "VAL")))
khigh = sum(1 for _, b_ in ksr if b_["KEY"] >= max(b_["CTX"], b_["VAL"]))
ctxlow = sum(1 for _, b_ in ksr if b_["CTX"] <= max(b_["KEY"], b_["VAL"]) + 0.05)
row(263, "with $\\beta^R$ flat across roles", "end beta^R in the key-split runs (REG3 + REG3_D, v1 KEY, all runs)",
    "flat across roles", f"{len(ksr)} runs: spread (max - min over CTX/KEY/VAL) <= 0.10 in {flatA}; every role in 0.15-0.35 in "
    f"{flatB}; KEY the highest role in {khigh}; CTX not above K/V by > 0.05 in {ctxlow}; CTX {rng([b_['CTX'] for _, b_ in ksr])}, "
    f"KEY {rng([b_['KEY'] for _, b_ in ksr])}, VAL {rng([b_['VAL'] for _, b_ in ksr])}",
    "MATCH" if flatA == len(ksr) else "PARTIAL",
    "report_4.md:57 said 'flat ... or highest at KEY'; the draft drops the second case. What holds in every key-split run is "
    "that CTX is not separated above K/V")
plate = {}
for nm_, rr in (("BASELINE", BH), ("REG3", R3)):
    ub = [r for r in rr.values() if not bound(r)]
    accs = [r["acc"] for r in ub]
    cel = [[c_["acc"] for c_ in r["acc_by_cell"].values()] for r in ub]
    flat = sum(1 for r, cc in zip(ub, cel) if all(abs(x - r["acc"]) <= 0.1 for x in cc))
    cm = [med([r["acc_by_cell"][k]["acc"] for r in ub]) for k in keys]
    plate[nm_] = (len(ub), accs, flat, cm, sum(1 for a_ in accs if 0.45 <= a_ <= 0.55), Counter(cls(r) for r in ub))
row(263, "The unbound Hebbian runs sit at chance between the streams in every cell, as the layers-2--3 oracle did",
    "S72 BASELINE / REG3 unbound runs: per-run accuracy, per-cell flatness (explore_g_register3.py's rule: every cell within 0.1 "
    "of the run's mean), per-cell medians", "at chance in every cell (every unbound run)",
    "; ".join(f"{k}: {v[0]} unbound ({dict(v[5])}), acc {rng(v[1])}, acc in 0.45-0.55 in {v[4]}/{v[0]}, flat runs {v[2]}/{v[0]}, "
              f"cell medians {rng(v[3])}" for k, v in plate.items()),
    "PARTIAL",
    "true of the per-cell medians and of most runs; not of every run (flat by the docstring's 0.1 rule in "
    f"{plate['BASELINE'][2]}/{plate['BASELINE'][0]} and {plate['REG3'][2]}/{plate['REG3'][0]}); 'as the layers-2-3 oracle did' "
    "inherits the S71 DISAGREE above (the oracle's runs are bimodal, 0.31-0.54). Not a pre-fixed reading: S72's plateau "
    "follow-up was conditional on 'architecture insufficient', which did not apply")
resb = sorted(s for s, r in RES.items() if bound(r))
resk = sorted(s for s in resb if cls(RES[s]) == "KEY")
row(263, "the residual gate alone routes $0/10$ on delta ($5/10$ bound, three by key splits)", "S73 RESGATE_ONLY_D ROUTED-BOUND, "
    "BOUND, key-split bound runs (v1 KEY on the end gate)", "0/10 routed; 5/10 bound, 3 by key splits",
    f"ROUTED-BOUND {cnt(RES, rb)}/10; BOUND {len(resb)}/10 {resb}; key-split bound {len(resk)} {resk}",
    "MATCH" if (cnt(RES, rb), len(resb), len(resk)) == (0, 5, 3) else "DISAGREE")
zend = {s: r["end"]["beta"] for s, r in L3ND.items()}
exact = sum(1 for z in zend.values() if round(z["CTX"], 2) == 1 and round(z["KEY"], 2) == 0 and round(z["VAL"], 2) == 0)
row(263, ["holds an exact latch when nudged ($9/10$ routed, $z = 1$ at the context token and $0$ elsewhere)"],
    "S73 LATCH3_NUDGE_D ROUTED-BOUND; end z CTX = 1, KEY = VAL = 0 (2 dp)", "9/10 routed; z = 1 / 0 / 0",
    f"ROUTED-BOUND {cnt(L3ND, rb)}/10; exact latch {exact}/10; reading 'the latch holds on delta memory' (z >= 0.9 / <= 0.1 on "
    f">= 8/10): {sum(1 for z in zend.values() if z['CTX'] >= 0.9 and z['KEY'] <= 0.1 and z['VAL'] <= 0.1)}/10",
    "MATCH" if cnt(L3ND, rb) == 9 and exact == 9 else "DISAGREE")
z1200 = {s: at_step(r, 1200)["beta"] if at_step(r, 1200) else None for s, r in sorted(L3D.items())}
all1 = sorted(s for s, z in z1200.items() if z and all(z[k] >= 0.99 for k in ("CTX", "KEY", "VAL")))
zendD = {s: r["end"]["beta"] for s, r in sorted(L3D.items())}
row(263, "goes to $z = 1$ everywhere by update 1200", "S73 LATCH3_D z at CTX/KEY/VAL at update 1200 and at the end, per run",
    "z = 1 everywhere by 1200", f"z >= 0.99 at every role at 1200 in {len(all1)}/10 runs; not: "
    + ", ".join(f"{s} ({z1200[s]['CTX']:.2f}/{z1200[s]['KEY']:.2f}/{z1200[s]['VAL']:.2f} at 1200; end {zendD[s]['CTX']:.2f}/"
                f"{zendD[s]['KEY']:.2f}/{zendD[s]['VAL']:.2f})" for s in z1200 if s not in all1),
    "MATCH" if len(all1) == 10 else "PARTIAL", "true of the median run (report_4.md:86), not of every run")
lk_unb = sum(1 for r in L3D.values() if not bound(r) and cls(r) == "KEY")
lk_all = sum(1 for r in L3D.values() if cls(r) == "KEY")
row(263, "and key-splits $9/10$", "S73 LATCH3_D key splits (v1)", "9/10",
    f"unbound KEY {lk_unb}/10 (the printed class count); all runs {lk_all}/10 (the bound run "
    f"{[s for s, r in L3D.items() if bound(r)]} is a key-split binding)", "PARTIAL" if lk_all != lk_unb else "MATCH",
    "the draft counts key splits over all runs for REG3_D (14) but over unbound runs here (9); on the same population as "
    "the 14 this would be 10/10")
s57z = {s: (at_step(r, 1200) or {}).get("beta") for s, r in sorted(A["LATCH"].items())}
s57all1 = sum(1 for z in s57z.values() if z and all(z[k] >= 0.99 for k in ("CTX", "KEY", "VAL")))
s57cls = Counter(cls(r) if not bound(r) else "BOUND" for r in A["LATCH"].values())
row(263, "as it had on the fixed layout", "S57 LATCH (fixed P = 8 layout, 410-419): z at 1200; failure classes",
    "the same behaviour on the fixed layout", f"z >= 0.99 at every role at 1200 in {s57all1}/10; classes {dict(s57cls)}",
    "PARTIAL" if s57cls.get("KEY", 0) < 9 else "MATCH",
    "holds for 'z = 1 everywhere by 1200'; if the clause also covers 'key-splits 9/10', the fixed layout's LATCH key-split "
    f"{s57cls.get('KEY', 0)}/10 (report_2.md: POSITION 4, OTHER 3, KEY 3)")
row(263, "one delta channel binds the layout $7/10$ with no gate", "S73 SINGLE_D BOUND", "7/10", f"{cnt(SGL, bound)}/10",
    "MATCH" if cnt(SGL, bound) == 7 else "DISAGREE", "S73's 'the gate is not needed on this task' (>= 8/10) did not apply; the "
    "draft does not name it")


def pk(r, site="L3"):
    return r["probe"][f"{site}|K"]["acc"], r["probe"][f"{site}|V"]["acc"]


routed = [(a, s, r) for a, rr in (("REG3", R3), ("REG3_NUDGE", R3N), ("REG3_D", R3D), ("REG3_NUDGE_D", R3ND),
                                   ("LATCH3_NUDGE_D", L3ND)) for s, r in rr.items() if rb(r)]
rk = [pk(r) for _, _, r in routed]
row(263, "finds the stream in the residual of no routed run", "in-run probe at the residual entering layer 3 in every ROUTED-BOUND "
    "run (S72 + S73)", "no routed run", f"{len(routed)} runs; K >= 0.9 and V >= 0.9 in {sum(k >= 0.9 and v >= 0.9 for k, v in rk)}; "
    f"K >= 0.9 in {sum(k >= 0.9 for k, _ in rk)}; K {rng([k for k, _ in rk])}, V {rng([v for _, v in rk])}",
    "MATCH" if not any(k >= 0.9 and v >= 0.9 for k, v in rk) else "DISAGREE",
    "'finds the stream' read as the docstring's >= 0.9 at K and at V (register3_delta.py:30-32)")
nonr = [(a, s, r) for a, rr in (("BASELINE_D", BD), ("REG3_D", R3D), ("RESGATE_ONLY_D", RES), ("LATCH3_D", L3D),
                                 ("SINGLE_D", SGL)) for s, r in rr.items() if bound(r) and (a == "SINGLE_D" or not rb(r))]
nk = [(a, pk(r)) for a, _, r in nonr]
both = sum(1 for _, (k, v) in nk if k >= 0.9 and v >= 0.9)
konly = sum(1 for _, (k, v) in nk if k >= 0.9)
row(263, "and of most non-routing solutions", "in-run probe L3 in the bound runs that do not route (S73: BASELINE_D, REG3_D, "
    "RESGATE_ONLY_D, LATCH3_D bound and not ROUTED-BOUND; SINGLE_D bound)", "most",
    f"{len(nk)} runs; K >= 0.9 and V >= 0.9 in {both}; K >= 0.9 in {konly}; by arm: "
    + ", ".join(f"{a} {sum(1 for b_, (k, v) in nk if b_ == a and k >= 0.9 and v >= 0.9)}/{sum(1 for b_, _ in nk if b_ == a)}"
                for a in ("BASELINE_D", "REG3_D", "RESGATE_ONLY_D", "LATCH3_D", "SINGLE_D")),
    "MATCH" if both > len(nk) / 2 else "PARTIAL",
    "with the docstring's criterion (>= 0.9 at K and at V) the stream is found in fewer than half of the non-routing "
    "solutions; 'most' holds at K alone" if both <= len(nk) / 2 else "")
sk = [pk(r) for r in SGL.values() if bound(r)]
row(263, "the single channel's bound runs at $0.93$--$1.00$", "SINGLE_D bound runs, probe L3 K / V", "0.93-1.00",
    f"K {rng([k for k, _ in sk])}, V {rng([v for _, v in sk])}", "PARTIAL" if min(v for _, v in sk) < 0.9 else "MATCH",
    "the range is K's only; at V the stream is not found in every bound run")
half = {s: r for s, r in BD.items() if bound(r) and 0.4 <= r["end"]["margin"] <= 0.6}
hk = [pk(r) for r in half.values()]
row(263, "the half-routed baseline's at $0.86$--$1.00$", "BASELINE_D bound runs with end margin in 0.4-0.6 (half-routed), probe L3 "
    "K / V", "0.86-1.00", f"{len(half)} runs (margin {rng([r['end']['margin'] for r in half.values()])}); K {rng([k for k, _ in hk])}, "
    f"V {rng([v for _, v in hk])}; all 14 bound: K {rng([pk(r)[0] for r in BD.values() if bound(r)])}, V "
    f"{rng([pk(r)[1] for r in BD.values() if bound(r)])}", "PARTIAL" if min(v for _, v in hk) < 0.86 else "MATCH",
    "K's range only; V goes down to 0.79")
row(263, "which is a description of a $2\\times2$ the screen fixed as descriptive", "BASELINE_D 2x2 probe (>= 0.9 at K and V) x "
    "ROUTED-BOUND; S73 reading 'the delta stack carries the cue' (>= 10/20)", "descriptive",
    f"[[probe & routed, probe & not], [no probe & routed, no probe & not]] = "
    f"[[{sum(1 for r in BD.values() if min(pk(r)) >= 0.9 and rb(r))}, {sum(1 for r in BD.values() if min(pk(r)) >= 0.9 and not rb(r))}], "
    f"[{sum(1 for r in BD.values() if min(pk(r)) < 0.9 and rb(r))}, {sum(1 for r in BD.values() if min(pk(r)) < 0.9 and not rb(r))}]]; "
    f"reading: {sum(1 for r in BD.values() if min(pk(r)) >= 0.9)}/20 -> does not apply", "MATCH",
    "the draft does not state that the pre-fixed 'the delta stack carries the cue' did not apply")

# ═════════════════════════════════════════════════════════════════════════════
# tex:265, 449, 36, 312: key split 'forms first' (descriptive timing check)
# ═════════════════════════════════════════════════════════════════════════════
timing = []
for nm_, rr in (("REG3", R3), ("REG3_D", R3D)):
    for s, r in sorted(rr.items()):
        ks = next((st["step"] for st in r["stats"] if "eta_key_by_key" in st and st["eta_key_by_key"] >= 0.5), None)
        wr = next((st["step"] for st in r["stats"] if st.get("beta") and st["beta"]["CTX"] > max(st["beta"]["KEY"], st["beta"]["VAL"]) + 0.05), None)
        timing.append((nm_, s, rb(r), cls(r), ks, wr))
failK = [t for t in timing if not t[2] and t[3] == "KEY"]
first = sum(1 for t in failK if t[4] is not None and (t[5] is None or t[5] > t[4]))
at1200 = sum(1 for t in failK if t[4] == 1200)
neverw = sum(1 for t in failK if t[5] is None)
rout = [t for t in timing if t[2]]
row(265, "by a key split that forms first", "REG3 + REG3_D key-split failures: first evaluation with eta^2 by key at KEY >= 0.5 "
    "(returned gate; recorded at 1200/2400/3600/end) vs first evaluation with beta^R CTX > max(KEY, VAL) + 0.05",
    "a key split forms first", f"{len(failK)} key-split failures: split before any CTX write in {first}; split already at 1200 in "
    f"{at1200}; CTX write never appears in {neverw}. Routed runs: CTX write appears at "
    f"{sorted(t[5] for t in rout if t[5] is not None)}",
    "MATCH" if first == len(failK) else "PARTIAL",
    "descriptive, unmarked here; the grid is coarse (eta^2 by key recorded at 1200, 2400, 3600 and the end only), so 'first' "
    "means 'by the first recorded evaluation'")
row(449, "the write at the cue is found on a fifth of seeds, and a key split forms first on the rest",
    "label-free REG3 / REG3_D: ROUTED-BOUND; the rest", "a fifth; key split first on the rest",
    f"{cnt(R3, rb)}/20 and {cnt(R3D, rb)}/20; of the other 16 per memory, key splits {kc['Hebbian REG3'][3]} and {kc['delta REG3_D'][3]} "
    f"(Hebbian also {dict(kc['Hebbian REG3'][5])}; delta {dict(kc['delta REG3_D'][5])})", "PARTIAL",
    "14 of the 16 'rest' on each memory are key splits (Hebbian: 2 OTHER; delta: 2 bound STREAM-PARTIAL); unmarked description")
row(36, "the failures being key splits that form first", "abstract: label-free failures", "the failures are key splits that form first",
    f"key splits {kc['Hebbian REG3'][3]}/16 and {kc['delta REG3_D'][3]}/16 of the non-ROUTED-BOUND runs", "PARTIAL",
    "14 of 16 per memory; unmarked description in the abstract")
row(36, "perfect gates at the later layers alone bind $0/10$", "abstract: S71 ORACLE_23", "0/10", f"{cnt(O23, bound)}/10",
    "MATCH" if cnt(O23, bound) == 0 else "DISAGREE")
rhl = src(G, "explore_g_rhl_child.py")
assert "RHL = ft.RandHeaderTask(P=4, nb=2, lmin=1, lmax=3)" in rhl
row(36, ["carries a cue up to six tokens away $20/20$ when nudged by labels", "found label-free on $4/20$ seeds"],
    "abstract: cue distance on RandHeaderTask-lite; nudged and label-free counts", "up to six tokens; 20/20 nudged; 4/20 label-free",
    f"RandHeaderTask(P=4, nb=2, lmin=1, lmax=3) (explore_g_rhl_child.py:{lineno(G, 'explore_g_rhl_child.py', 'RHL = ft.RandHeaderTask')}): "
    f"a block is CTX + 1-3 (key, value) pairs, so the last value is 6 tokens after its CTX; REG3_NUDGE {cnt(R3N, rb)}/20; "
    f"REG3 {cnt(R3, rb)}/20", "MATCH", "'20/20' is the Hebbian count (delta nudged 10/10)")

# ═════════════════════════════════════════════════════════════════════════════
# tex:265 other statements
# ═════════════════════════════════════════════════════════════════════════════
row(265, "told where to split, it binds 20 of 20", "S72 REG3_NUDGE ROUTED-BOUND", "20 of 20", f"{cnt(R3N, rb)}/20",
    "MATCH" if cnt(R3N, rb) == 20 else "DISAGREE")
row(265, "The architecture that carries a distant cue label-free exists on the main line's own memory",
    "label-free carrying on Hebbian memory", "exists (stated as a finding)", f"REG3 (label-free) {cnt(R3, rb)}/20 ROUTED-BOUND vs "
    f"BASELINE {cnt(BH, rb)}/20, 4 vs 0, p = 0.0625: S72 reading 'not discovered'; the reading that applied, 'the register carries "
    "the cue at all three layers', is on the NUDGED arm", "PARTIAL",
    "four label-free seeds show the architecture can carry the cue without labels, which is a description; the paragraph "
    "presents it as the change, while the pre-fixed reading on label-free discovery is 'not discovered'")
p8 = mcnemar1(8, 0)
row(265, "at the observed rate, 40 pairs would put the pooled comparison near 8 vs 0 ($p = 0.004$)",
    "projection: 4/20 vs 0/20 per 20 pairs, scaled to 40", "8 vs 0, p = 0.004", f"one-sided McNemar 8 vs 0 = {p8:.4f}",
    "MATCH" if round(p8, 3) == 0.004 else "DISAGREE", "arithmetic only; see the S82 section for REG3 vs BASELINE on new seeds "
    "(2100-2119), recorded before the draft's commit")
r56 = A["RESGATE_ONLY"]
resp = [pk(r) for r in RES.values()]
resk_all = sum(1 for r in RES.values() if cls(r) == "KEY")
row(265, ["The audit's per-layer gate over the residual stream is closed", "it key-splits ($0/10$ twice)"],
    "S56 RESGATE_ONLY (Hebbian, P = 8, layers 2-3) and S73 RESGATE_ONLY_D (delta, RHL, all three layers)",
    "0/10 twice; key-splits; nothing in the stack to read",
    f"S56: BOUND {cnt(r56, bound)}/10, KEY {sum(1 for r in r56.values() if cls(r) == 'KEY')}/10; S73: ROUTED-BOUND {cnt(RES, rb)}/10 but "
    f"BOUND {cnt(RES, bound)}/10, key splits (all runs) {resk_all}/10; S73 RESGATE_ONLY_D probe L3 >= 0.9 at K and V in "
    f"{sum(1 for k, v in resp if k >= 0.9 and v >= 0.9)}/10 (K {rng([k for k, _ in resp])})", "PARTIAL",
    "the two zeros are different outcomes (BOUND on S56's Hebbian P = 8 layout; ROUTED-BOUND on S73's delta RHL, where it binds "
    "5/10); it key-splits in 10/10 and 8/10. 'Closed' is not a pre-fixed reading (S73's is only 'discovered (residual gate)', "
    "which did not apply), and 'nothing in the stack to read' is S55's Hebbian result, while tex:263 says the delta stack can "
    "learn stream-tagged keys")
row(265, "the hinge's key term on the shared gate, a slow gate for the first 1200 updates, a gate without the raw window input",
    "S82's three devices (explore_g_devices.py at 83699a2)", "key-term hinge; slow gate 1-1200; no window input",
    "REG3_KEYHINGE (S52's key-term hinge, updates 1-2400, on the three layers' gates); REG3_SLOWGATE (gate group 1e-4 for "
    "updates 1-1200); REG3_NOWIN (gate reads [LN(r); R_{t-1}] only)", "MATCH", "explore_g_devices.py:15-21 at 83699a2")
specs = [p_ for b_ in (DRAFT, MAIN) for p_ in git("ls-tree", "--name-only", "-r", b_).split("\n") if p_.startswith("specs/")]
row(265, "a pre-registered main-line test of the register against the baseline on the randomised header layout",
    "a specification for the main-line register test", "pre-registered, queued",
    f"specs/ at {DRAFT} and on the main line: {sorted(set(specs))}; no register test specification", "NOT RECORD-CHECKABLE",
    "no committed specification exists to check 'pre-registered' against; the phrase describes a plan")

# ═════════════════════════════════════════════════════════════════════════════
# far table (tex:293-307)
# ═════════════════════════════════════════════════════════════════════════════
row(293, "session G, seeds 700--719", "table header: seeds", "700-719", "S71 arms and S73's 10-seed arms on 700-709; S72 / S73 "
    "20-seed arms on 700-719; RandHeaderTask-lite S = 2, P = 4, k = 2, 24000 updates", "MATCH", "the 10-run rows are on 700-709")
tab = [
    (295, "perfect gate, all three layers & $10/10$ & $10/10$", "ORACLE_ALL / ORACLE_ALL_D BOUND", (cnt(OA, bound), cnt(OAD, bound)), (10, 10)),
    (296, "perfect gate, layers 2--3 only (window gate at layer 1) & $0/10$ & $10/10$", "ORACLE_23 / ORACLE_23_D BOUND",
     (cnt(O23, bound), cnt(O23D, bound)), (0, 10)),
    (297, "perfect gate, layer 1 only & $9/10$ & ---", "ORACLE_1 BOUND (no delta arm)", (cnt(O1, bound),), (9,)),
    (298, "window gate + SLOW (ROUTED-BOUND; bound) & $0/20$ ($1/20$) & $0/20$ ($14/20$)", "BASELINE / BASELINE_D ROUTED-BOUND (BOUND)",
     (cnt(BH, rb), cnt(BH, bound), cnt(BD, rb), cnt(BD, bound)), (0, 1, 0, 14)),
    (299, "labelled nudge & $\\mathbf{20/20}$ & $\\mathbf{10/10}$", "REG3_NUDGE / REG3_NUDGE_D ROUTED-BOUND", (cnt(R3N, rb), cnt(R3ND, rb)), (20, 10)),
    (300, "label-free & $4/20$ & $4/20$", "REG3 / REG3_D ROUTED-BOUND", (cnt(R3, rb), cnt(R3D, rb)), (4, 4)),
    (301, "no register & --- & $0/10$", "RESGATE_ONLY_D ROUTED-BOUND", (cnt(RES, rb),), (0,)),
    (302, "label-free / nudged & --- & $0/10$ / $9/10$", "LATCH3_D / LATCH3_NUDGE_D ROUTED-BOUND", (cnt(L3D, rb), cnt(L3ND, rb)), (0, 9)),
    (303, "one channel, no gate (bound) & --- & $7/10$", "SINGLE_D BOUND", (cnt(SGL, bound),), (7,)),
]
for n_, nd_, it, got, want in tab:
    row(n_, nd_, "table: " + it, " / ".join(map(str, want)), " / ".join(map(str, got)), "MATCH" if got == want else "DISAGREE")
f70 = {a: (cnt(rr, rb), cnt(rr, routed_star), cnt(rr, bound), sorted(rr)) for a, rr in FA.items()}
row(306, ["perfect-gate rows are BOUND, the others ROUTED-BOUND", "(S70, 330--349) is $0/20$ routed on either memory with $1/20$ and $6/20$ bound"],
    "caption: outcomes per row; F's S70 window baseline", "S70: 0/20 routed, 1/20 and 6/20 bound",
    "; ".join(f"{a}: ROUTED*@end {v[1]}/20, BOUND {v[2]}/20, seeds {v[3][0]}-{v[3][-1]}" for a, v in f70.items())
    + "; rows 295-297 BOUND, 298 both, 299-302 ROUTED-BOUND, 303 BOUND (labelled)",
    "MATCH" if (f70["S70_HEBB"][1], f70["S70_HEBB"][2], f70["S70_DELTA"][1], f70["S70_DELTA"][2]) == (0, 1, 0, 6) else "DISAGREE",
    f"F's store explore_out/F/rh_lite_results.json at {F}; the one-channel row is BOUND, as its label says")

# ═════════════════════════════════════════════════════════════════════════════
# tex:114 (ROUTED*@end), tex:415
# ═════════════════════════════════════════════════════════════════════════════
c2g = src(G, "explore_common2.py")
c2f = src("origin/claude/explore-F", "explore_common2.py")
okdef = all("ROUTE_MARGIN, ROUTE_ETA = 0.9, 0.9" in t and 'd["margin"] >= ROUTE_MARGIN and d["eta_key_by_stream"] > ROUTE_ETA' in t for t in (c2g, c2f))
row(114, "routing margin $\\geq 0.9$ and $\\eta^2$ by stream at key positions $> 0.9$ at the end", "ROUTED*@end definition",
    "margin >= 0.9 and eta^2 by stream at KEY > 0.9", f"explore_g_rhl_parent.py:{lineno(G, 'explore_g_rhl_parent.py', 'ROUTE_MARGIN, ROUTE_ETA = 0.9, 0.9')} "
    f"and :{lineno(G, 'explore_g_rhl_parent.py', 'def routed_star')}-45 (G, {G}); explore_common2.py:"
    f"{lineno(G, 'explore_common2.py', 'ROUTE_MARGIN, ROUTE_ETA = 0.9, 0.9')}, :{lineno(G, 'explore_common2.py', 'def routed_star')}-64 "
    f"(G and F head identical: {okdef})", "MATCH" if okdef else "DISAGREE")
row(114, "the readings of Sections~\\ref{sec:far} and \\ref{sec:replication} are on \\textbf{ROUTED-BOUND}",
    "outcome of the Section 'far' readings", "ROUTED-BOUND", "S72 / S73: ROUTED-BOUND; S71: BOUND (explore_g_layers.py:"
    f"{lineno(G, 'explore_g_layers.py', 'READINGS (BOUND')}); S56 / S57: BOUND and BOUND ROUTED (G's own, explore_g_register.py READINGS); "
    "S55: probe medians", "PARTIAL",
    "true of S72 / S73 only; the section's S71 readings are on BOUND (the table caption says so for the perfect-gate rows) "
    "and S56 / S57's on BOUND / G's BOUND ROUTED")
row(114, "one screen's reading named ``discovered'' was fixed on ROUTED-BOUND and would have come out differently on DISCOVERED",
    "S73 'discovered (register)' on ROUTED-BOUND vs DISCOVERED", "differs", f"ROUTED-BOUND {paired(R3D, BD, rb)[:2]} p = "
    f"{paired(R3D, BD, rb)[2]:.4f}; DISCOVERED {bdD[:2]} p = {bdD[2]:.4f}", "MATCH")
row(415, "(4 vs 0 against 5 vs 0)", "S73 reading vs the outcome named DISCOVERED", "4 vs 0 against 5 vs 0",
    f"{paired(R3D, BD, rb)[:2]} vs {bdD[:2]}; S72's DISCOVERED gives {paired(R3, BH, disc)[:2]} (the same as its ROUTED-BOUND)", "MATCH",
    "only S73 is affected")

# ═════════════════════════════════════════════════════════════════════════════
# tex:125 register gate vs code
# ═════════════════════════════════════════════════════════════════════════════
regm = src(G, "explore_g_regmodel.py")
reg_rows = []


def creg(item, draft, code, status, note=""):
    reg_rows.append((item, draft, code, status, note))


ln = lambda p_, nd_: f"{p_}:{lineno(G, p_, nd_)}"  # noqa: E731
creg("register dimension", "16-dimensional", f"R_DIM = 16 ({ln('explore_g_regmodel.py', 'R_DIM = 16')})",
     "MATCH" if "R_DIM = 16" in regm else "DISAGREE")
creg("register update", "R_t = R_{t-1}(1 - beta_t) + beta_t W_R v_t",
     f"R = R * (1 - b) + b * w[:, t], w = v @ W_R.T, v = raw embedding ({ln('explore_g_regmodel.py', 'R = R * (1 - b) + b * w[:, t]')}, "
     f"{ln('explore_g_regmodel.py', 'w = v @ self.W_R.T')})", "MATCH")
creg("write strength", "beta^R_t = sigmoid(w_R . v_t + b_R), b_R init logit 0.1",
     f"beta = sigmoid(v @ w_R + b_R) ({ln('explore_g_regmodel.py', 'beta = torch.sigmoid(v @ self.w_R + self.b_R)')}); "
     f"B_R0 = log(0.1 / 0.9) ({ln('explore_g_regmodel.py', 'B_R0 = math.log(0.1 / 0.9)')})", "MATCH")
creg("written by every token", "every token writes", "beta in (0, 1) at every t (sigmoid); no gating of the write", "MATCH",
     "not true of S57's / S73's latch, which is not the register gate")
creg("causal read", "the gate at t reads R_{t-1}", f"prev.append(R) before the update ({ln('explore_g_regmodel.py', 'prev.append(R)')}); "
     "CHECK (d) in checks_reg3 (token t's write changes no gate at <= t)", "MATCH")
creg("one gate for all three layers", "one gate shared by all three layers",
     f"make_reg3(..., gated=(1, 2, 3)) ({ln('explore_g_rhl_child.py', 'def make_reg3')}); A_v, A_r, A_R, W_g2 single tensors shared "
     "by the gated layers (regmodel docstring); each layer's gate differs only through LN(r^(l))", "MATCH")
creg("gate form", "g = softmax(W_g tanh(A [LN(r^(l)); u_t; R_{t-1}]))",
     f"u = A_v x_in + A_r LN(r) + A_R Rprev; g = softmax(tanh(u) @ W_g2.T) ({ln('explore_g_regmodel.py', 'def layer_gate')}-"
     f"{lineno(G, 'explore_g_regmodel.py', 'return F.softmax(torch.tanh(u) @ self.W_g2.T, dim=-1)')}); LN without affine; "
     "r^(1) = LN(embedding)", "MATCH", "W_g in the draft is W_g2, a copy of the window gate's W_g (the original stays unused)")
creg("u_t = the window gate's input", "u_t, the window gate's input",
     f"Reg3BDH.layer_gate: x_in = gate_conv(v) if inp == 'window' ({ln('explore_g_rhl_child.py', 'x_in = self.gate_conv(v) if self.inp')}); "
     f"make_reg3 default inp='window'", "MATCH", "LOCAL3's width-3 causal convolution of the raw embeddings")
creg("initialisation", "A initialised so that the gate equals the window gate at step 0",
     f"A_v = W_in (copy), W_g2 = W_g (copy), A_r, A_R ~ N(0, 0.01^2) ({ln('explore_g_regmodel.py', 'm.A_r = nn.Parameter(torch.randn(hg, D, generator=g) * 0.01)')}); "
     "rhl_child docstring: 'u = W_in window_t + a perturbation at every layer at step 0'; CHECK (b) equals LOCAL3 only with "
     "A_r = A_R = 0 set by hand", "PARTIAL",
     "the gate equals the window gate up to a small random perturbation, not exactly (the spec said u ~ W_in v_t)")
creg("stop-gradient", "a stop-gradient from the gate into the stack",
     f"make_reg3 passes sg=True; r = h.detach() if self.sg ({ln('explore_g_regmodel.py', 'r = h.detach() if self.sg else h')})", "MATCH")
creg("REG3 = all three layers", "REG3 is this gate at all three layers", "ARMS 'REG3': ('reg3', ...) -> make_reg3 gated=(1, 2, 3); "
     "CHECK (c) asserts m.gated == (1, 2, 3)", "MATCH")
s56doc = src(G, "explore_g_register.py")
creg("the earlier design", "Revision 8's design had it at layers 2-3 only, with the window gate at layer 1",
     f"S56: GATED = (2, 3), layer 1 LOCAL3 ({ln('explore_g_regmodel.py', 'GATED = (2, 3)')}); but S56's gate read the raw embedding v_t, "
     f"not the window (regmodel layer_gate u = v @ A_v.T; rhl_child: inp = 'v' is 'S56's code path'; explore_g_register.py:"
     f"{lineno(G, 'explore_g_register.py', 'A_v v_t + A_R R_{t-1}')}). The layers 2-3 design is the main line's S56 prompt "
     f"(docs/reading/parallel_sessions_prompts.md:{lineno(MAIN, 'docs/reading/parallel_sessions_prompts.md', 'Gate: layer 1 keeps LOCAL3')}, "
     f"added ad8a92e 2026-10-08) and docs/reading/distant_cues_2026-10.md:{lineno(MAIN, 'docs/reading/distant_cues_2026-10.md', 'layer 1 keeps the window gate')}; "
     "Revision 8's text names only 'a per-layer gate over the residual stream'", "PARTIAL",
     "the layers are right; 'had it' is not the same gate: S56's register gate read v_t, REG3's reads the window, so REG3 changes "
     "two things (layer 1, and the gate's input) relative to S56, and S71's perfect-gate result speaks only to the first")
creg("the nudge", "the 5% labelled nudge (tex:261, :263)", f"300 Adam steps at lr 0.01 on the new gate / register parameters toward "
     f"0.475 + 0.05 x onehot(stream) ({ln('explore_g_register_child.py', 'NUDGE_STEPS, NUDGE_LR, NUDGE_B, NUDGE_BASE, NUDGE_AMP')}); "
     "REG3_NUDGE applies it to all three layers", "MATCH", "'5%' is the 0.05 amplitude of the target, before training")
for it, d_, c_, s_, nt in reg_rows:
    row(125, "16-dimensional register" if it == "register dimension" else "register gate (session~G)", "register gate: " + it, d_, c_, s_, nt)

# ═════════════════════════════════════════════════════════════════════════════
# provenance (tex:502), S82
# ═════════════════════════════════════════════════════════════════════════════
first_add = {n_: git("log", "--diff-filter=A", "--format=%h", GBR, "--", f"{GD}report_{n_}.md").strip() for n_ in (1, 2, 3, 4)}
nrun = (len(S55["runs"]), len(S56["runs"]) + len(S57["runs"]), len(S71["runs"]), len(S72["runs"]) + len(S73["runs"]))
row(502, ["report 1, \\texttt{6cd36ff}", "report 2, \\texttt{427db36}", "report 3, \\texttt{35de6db}", "report 4, \\texttt{93422cf}", "22; 139; 50; 150"],
    "provenance: report commits and run counts", "6cd36ff, 427db36, 35de6db, 93422cf; 22; 139; 50; 150",
    f"report_n.md first added at {', '.join(first_add[n_] for n_ in (1, 2, 3, 4))}; records {nrun[0]}; {nrun[1]} "
    f"(S56 {len(S56['runs'])} + S57 {len(S57['runs'])}); {nrun[2]}; {nrun[3]} (S72 {len(S72['runs'])} + S73 {len(S73['runs'])})",
    "MATCH" if [first_add[n_] for n_ in (1, 2, 3, 4)] == ["6cd36ff", "427db36", "35de6db", "93422cf"] and nrun == (22, 139, 50, 150) else "DISAGREE")

draft_t = ctime(DRAFT)
ghead = short(GBR)
gdraft = git("rev-list", "-1", f"--before={draft_t}", GBR).strip()[:7]
spec_files = git("show", "--name-only", "--format=", S82SPEC).split()
spec_hist = git("log", "--format=%h", GBR, "--", "explore_g_devices.py", "explore_g_devices_child.py").split()
s82p = GD + "s82_devices_results.json"
at_spec = [p_ for p_ in git("ls-tree", "--name-only", S82SPEC, GD).split("\n") if "s82" in p_]
commits = git("log", "--reverse", "--format=%h %cI", f"{S82SPEC}..{GBR}", "--", s82p).strip().split("\n")
first_rec = None
for ln_ in commits:
    h, t = ln_.split()
    st = json.loads(git("show", f"{h}:{s82p}"))
    if st.get("runs"):
        first_rec = (h, t, len(st["runs"]))
        break
best_add = git("log", "--diff-filter=A", "--format=%h %cI", GBR, "--", GD + "s82_best.json").strip()
best = json.loads(git("show", f"{GBR}:{GD}s82_best.json"))
rep82 = [p_ for p_ in git("ls-tree", "--name-only", GBR, GD).split("\n") if re.search(r"report_[5-9]", p_)]


def s82_state(ref):
    st = json.loads(git("show", f"{ref}:{s82p}"))
    rr = {a: runs(st, a) for a in sorted({k.split("|")[0] for k in st["runs"]})}
    return st, rr


st_d, rr_d = s82_state(gdraft)
st_h, rr_h = s82_state(GBR)


def s82_table(rr):
    L = ["| arm | n | BOUND | ROUTED-BOUND | unbound classes (v1) |", "|---|---|---|---|---|"]
    for a, x in rr.items():
        cl = Counter(cls(r) for r in x.values() if not bound(r))
        L.append(f"| {a} | {len(x)} | {cnt(x, bound)} | {cnt(x, rb)} | {', '.join(f'{k} {v}' for k, v in sorted(cl.items())) or '-'} |")
    return "\n".join(L)


ph1 = all(len(rr_d.get(a, {})) == n_ for a, n_ in (("BASELINE", 20), ("REG3", 20), ("REG3_KEYHINGE", 20), ("REG3_SLOWGATE", 20),
                                                    ("REG3_NOWIN", 20), ("REG3_NUDGE", 2), ("ORACLE_ALL", 2)))
reg82 = paired(rr_d["REG3"], rr_d["BASELINE"], rb)
kh82 = paired(rr_d["REG3_KEYHINGE"], rr_d["BASELINE"], rb)
kr82 = paired(rr_d["REG3_KEYHINGE"], rr_d["REG3"], rb)
row(265, "Session~G's S82 is screening three devices against that", "S82 state at the draft's commit",
    "screening three devices", f"phase 1 (the three devices, 104 runs) complete at {best_add.split()[0]} ({best_add.split()[1]}), "
    f"best device written to s82_best.json ({best['device']}, ROUTED-BOUND {best['routed_bound']}/20); phase 2 (the best device "
    f"on delta, 32 runs) running: {sum(len(v) for a, v in rr_d.items() if a.endswith('_D'))}/32 records at {gdraft} (the last G "
    f"commit before the draft's {draft_t}); no S82 report", "PARTIAL",
    "15 minutes before the draft's commit the three-device screen had finished and selected one device; the draft describes "
    "it as still screening three, and gives no S82 numbers (none are reported by G either)")
row(41, ["G's S82", "are running"], "S82 running", "running", f"phase 2 {sum(len(v) for a, v in rr_h.items() if a.endswith('_D'))}/32 at the "
    f"fetched head {ghead}", "MATCH")
row(502, "S82 running & \\texttt{93422cf}", "provenance: S82 and the pinned commit", "S82 running; commit 93422cf; no run count",
    f"93422cf precedes S82's spec ({S82SPEC}, {ctime(S82SPEC)}); at {gdraft} the S82 store held {len(st_d['runs'])} records "
    f"({sum(len(v) for a, v in rr_d.items() if not a.endswith('_D'))} phase 1 + {sum(len(v) for a, v in rr_d.items() if a.endswith('_D'))} phase 2)",
    "PARTIAL", "the row pins a commit without S82; the table gives a running batch's partial count elsewhere ('118 at 5dfc2b5' for "
    "batch 19) but none for S82")
row(449, "The devices under screen attack the key split", "S82 state", "devices under screen",
    f"phase 1 complete, {best['device']} chosen ({best_add.split()[1]})", "PARTIAL", "as tex:265")

# ═════════════════════════════════════════════════════════════════════════════
# retrospective lines (tex:314, 338, 355, 378, 380, 459, 464)
# ═════════════════════════════════════════════════════════════════════════════
row(314, "where a Hebbian stack needs its first layer routed and a delta stack does not", "S71 readings behind it",
    "Hebbian needs layer 1 routed; delta does not", f"'layer 1 is the block' ({b1} vs {c1}); 'delta tolerates an unrouted layer 1' "
    f"(ORACLE_23_D {cnt(O23D, bound)}/10)", "MATCH" if r_block and r_tol else "DISAGREE", "perfect gates, RandHeaderTask-lite")
row(338, "Each is marked where it appears", "the sessions' descriptive statements are marked where they appear",
    "each is marked", "marked at tex:263 ('Descriptively', 'a description of a 2x2'); the same statements recur unmarked at tex:36 "
    "('key splits that form first'), tex:265 ('the register's write at the context token ... a key split that forms first'), "
    "tex:312, tex:449, and the tex:261 interpretation recurs unmarked at tex:355", "PARTIAL",
    "see 'Descriptive statements: marked or not'")
row(355, ["The probe found the cue absent", "the residual gate had nothing to read",
          "that design choice is what sank its Hebbian register"],
    "'Done.' item: probe, residual gate, register", "absent; nothing to read; carries the cue once it gates layer 1; the design "
    "choice sank the Hebbian register", f"S55 'absent' (applies); S72 R1 'carries the cue at all three layers' ({cnt(R3N, rb)}/20, "
    f"nudged); residual gate: S56 RESGATE_ONLY 0/10 BOUND (Hebbian, P = 8), S73 RESGATE_ONLY_D {cnt(RES, bound)}/10 BOUND, 0/10 "
    "ROUTED-BOUND (delta; the delta stack carries the stream in 7/20 baselines); the design-choice clause is the tex:261 "
    "interpretation, here unmarked", "PARTIAL",
    "the readings are right; 'nothing to read' holds for the Hebbian stack (S55) only, and 'that design choice is what sank its "
    "Hebbian register' is an interpretation across tasks (S56 fixed P = 8 vs S71/S72 RandHeaderTask-lite) and across gate "
    "inputs (S56 v_t vs REG3 window), marked as ours at tex:261 but not here")
row(378, "that the register at layers 2--3 would carry the cue on Hebbian memory (it needed layer~1)",
    "overturned prediction", "the register at layers 2-3 did not carry the cue; it needed layer 1",
    f"S56 R1 'architecture insufficient' (REG_NUDGE BOUND ROUTED {cnt(A['REG_NUDGE'], br56)}/10); 'needed layer 1': S71 perfect gates "
    "(RandHeaderTask-lite) and S72's REG3 (all three layers, window input, other task); no register run at layers 2-3 exists on "
    "RandHeaderTask-lite and no all-layer register run on P = 8", "PARTIAL",
    "the overturned prediction is a reading; the parenthesis is the tex:261 interpretation, unmarked")
dc_ = src(MAIN, "docs/reading/distant_cues_2026-10.md")
row(380, ["One screen (S71) settled it", "on the reasoning that its residual input is the embedding and carries no cue"],
    "S71 settled the layer question; the recorded reason for keeping layer 1 on the window gate", "S71 settled it; reasoning: "
    "layer 1's residual input is the embedding",
    f"S71 'layer 1 is the block' applies; main line docs/reading/distant_cues_2026-10.md:"
    f"{lineno(MAIN, 'docs/reading/distant_cues_2026-10.md', 'layer 1 keeps the window gate')}: 'layer 1 keeps the window gate (no "
    "residual yet)'; the register is written from raw embeddings (W_R v_t)", "MATCH" if "layer 1 keeps the window gate (no residual yet)" in dc_ else "PARTIAL",
    "the S71 reading is on perfect gates; 'settled' for the register gate rests on S72 as well")
row(459, ["is found label-free on a fifth of seeds", "on Hebbian memory the first layer is the one that must be routed"],
    "conclusion: label-free fifth; layer 1", "a fifth; layer 1 must be routed", f"REG3 {cnt(R3, rb)}/20, REG3_D {cnt(R3D, rb)}/20 "
    f"(readings 'not discovered' / does not apply); S71 'layer 1 is the block'", "MATCH",
    "the count is a description; its pre-fixed reading did not apply, which the conclusion does not say")
row(464, "Session~G's key-split devices for the register (S82)", "'Next' list: S82", "next", f"phase 1 complete before the draft "
    f"({best_add.split()[1]}); phase 2 running", "MATCH", "phase 2 (the chosen device on delta memory) is what remains")

# ═════════════════════════════════════════════════════════════════════════════
# output
# ═════════════════════════════════════════════════════════════════════════════
ORDER = {"DISAGREE": 0, "PARTIAL": 1, "NOT RECORD-CHECKABLE": 2, "MATCH": 3}
ROWS.sort(key=lambda r: (ORDER[r[4]], r[0]))
C = Counter(r[4] for r in ROWS)


def esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


L = []
L.append("# Revision 9 draft vs Session G's records (referee)")
L.append("")
L.append(f"Draft: `{TEXP}` at `{DRAFT}` (origin/claude/rev9-draft, committed {draft_t}); cited as tex:NNN. Records: Session G at "
         f"`{G}` (report 4; the commit the draft pins) for S55-S57 and S71-S73; F's S70 store at `{F}`; S82 at `{S82SPEC}` (spec), "
         f"`{gdraft}` (the last G commit before the draft's commit) and `{ghead}` (the fetched head of origin/claude/explore-G, "
         f"{ctime(GBR)}). Generated by `docs/review_2026-10/rev9/check_G.py` (stdlib + rlib.py; records and code read with "
         "`git show`; no torch). The draft's values are typed into the script with their line; it asserts each cited line "
         "contains the quoted text. None is used as an input.")
L.append("")
L.append(f"Rows: {len(ROWS)} — DISAGREE {C['DISAGREE']}, PARTIAL {C['PARTIAL']}, NOT RECORD-CHECKABLE {C['NOT RECORD-CHECKABLE']}, "
         f"MATCH {C['MATCH']}.")
L.append("")
L.append("Record integrity at the pinned commits: " + "; ".join(integrity) + ".")
L.append("")
L.append("Definitions (re-implemented): BOUND = transition not None (the recorded transition equals rlib.transition(curve) in "
         "every record); ROUTED*@end = end margin >= 0.9 and end eta^2 by stream at KEY > 0.9; ROUTED-BOUND = BOUND and "
         "ROUTED*@end; DISCOVERED (F's) = BOUND and end VAL cos < 0.5 (explore_g_rhl_parent.py:6-17); G's BOUND ROUTED = BOUND, "
         "VAL map one-to-one at every gated layer, every stream >= 0.9 (S56/S57: layers 2-3; S71-S73: 1-3); failure class = "
         "test_router_layout v1 on the end gate (rlib.fail_class), MERGED if STREAM-PARTIAL and the L3 VAL map is not "
         "one-to-one; 'beats' = one-sided exact McNemar p < 0.05 (rlib.mcnemar1). Key-split counts are given for the population "
         "each statement implies, and the population the screens print (unbound runs only).")
L.append("")
L.append("## Rows (DISAGREE, then PARTIAL, NOT RECORD-CHECKABLE, MATCH)")
L.append("")
L.append("| tex line | item | draft says | recomputed | status | note |")
L.append("|---|---|---|---|---|---|")
for n_, it, d_, rc, s_, nt in ROWS:
    L.append(f"| tex:{n_} | {esc(it)} | {esc(d_)} | {esc(rc)} | **{s_}** | {esc(nt)} |")
L.append("")

# readings vs docstring
L.append("## Readings vs docstring")
L.append("")
L.append("Each reading the draft names or should name, its docstring wording at the pinned commit, what applies from the records, "
         "and how the draft words it.")
L.append("")
L.append("| screen | docstring (file:line at 93422cf) | applies (recomputed) | draft | wording |")
L.append("|---|---|---|---|---|")
# S55
L.append(f"| S55 | \"absent\": median <= 0.6 at K and V for every l in {{L2, L3, final}} (explore_g_probe.py:"
         f"{lineno(G, 'explore_g_probe.py', '\"absent\"')}) | {'absent' if absent else 'not absent'} (L3S_P8; FAR_L3_P4 'partial at L3 and final'; "
         "ORACLE_P8 absent) | tex:259 'The pre-fixed reading ``absent'' applied'; tex:59, :355 'absent' | same |")
# S56
reg_ = A["REG"]
r2d_key = sum(1 for r in rd.values() if not bound(r) and cls(r) == "KEY")
r2d = cnt(rd, bound) >= cnt(bd, bound) + 3 and r2d_key <= 2
r1h = cnt(A["REG_NUDGE"], br56)
r1d = cnt(nd, br56)
lab = lambda k: "the register carries the cue" if k >= 9 else "architecture insufficient" if k < 6 else "neither"  # noqa: E731
L.append(f"| S56 R1 | \"the register carries the cue\" REG_NUDGE BOUND ROUTED >= 9/10; \"architecture insufficient\" < 6/10; else "
         f"\"neither\" (explore_g_register.py:{lineno(G, 'explore_g_register.py', 'R1 \"the register carries the cue\"')}) | "
         f"Hebbian {r1h}/10: \"{lab(r1h)}\"; delta {r1d}/10: \"{lab(r1d)}\" | tex:261 describes the Hebbian nudged arm as 'the informative "
         "failure'; neither reading is named | Hebbian: not named (substance agrees); delta 'neither' not named |")
L.append(f"| S56 R2 | \"discovered\" REG or REG_NOSG BOUND >= BASELINE BOUND + 3 with that arm's KEY failures <= 2 "
         f"(explore_g_register.py:{lineno(G, 'explore_g_register.py', 'R2 \"discovered\"')}) | Hebbian 0 vs 0: \"not discovered\"; delta "
         f"REG_D {cnt(rd, bound)} vs BASELINE_D {cnt(bd, bound)}, KEY failures (unbound) {r2d_key}: \"{'discovered' if r2d else 'not discovered'}\" "
         "(report_2.md: 'discovered (REG_D)', with the caveat that 3 of 6 bound runs are key-split bindings) | tex:261 gives "
         "'6/10 bound against the baseline's 2/10 (4 vs 0, p = 0.0625), but only two of its six routed, three binding through "
         "key splits' | **differs**: the pre-fixed reading that applied ('discovered') is not stated; the McNemar p the draft quotes "
         "is not R2's rule, and its 'p = 0.0625' reads as a non-result where the rule gave 'discovered' |")
L.append("| S56 R3-R5, S57 | R3 'the single GDN channel is at least as good'; R4 'it does not'; R5 'SG does not matter'; S57 R1 "
         f"'architecture insufficient', R2 'not discovered' | all 0 vs 0 (every Hebbian arm 0/10; LATCH_NUDGE BOUND ROUTED "
         f"{cnt(A['LATCH_NUDGE'], br56)}/10) | not named; counts given ('every Hebbian arm bound 0/10') | omitted (all trivially at 0) |")
# S71
L.append(f"| S71 | \"layer 1 is the block\" / \"layer 1 is not needed\" / \"layer 1 alone suffices\" / \"delta tolerates an unrouted "
         f"layer 1\" (explore_g_layers.py:{lineno(G, 'explore_g_layers.py', 'layer 1 is the block')}-"
         f"{lineno(G, 'explore_g_layers.py', 'delta tolerates an unrouted layer 1')}) | block {b1} vs {c1} p = {p1:.5f} applies; not needed: no; "
         f"alone suffices: applies; delta tolerates: applies | tex:261 the three that apply, verbatim; tex:59 'layer~1 is the block on "
         "Hebbian memory' | same (tex:59 adds the family) |")
# S72
L.append(f"| S72 R1 | \"the register carries the cue at all three layers\" REG3_NUDGE ROUTED-BOUND >= 17/20 (explore_g_register3.py:"
         f"{lineno(G, 'explore_g_register3.py', 'the register carries the cue at all three layers')}) | {cnt(R3N, rb)}/20: applies | tex:263 "
         "heading 'The register at all three layers carries the cue'; tex:59 'the register carries the cue 20/20 nudged' | "
         "paraphrase; same substance |")
L.append(f"| S72 R2 | \"discovered\" REG3 beats BASELINE; \"not discovered\" otherwise (explore_g_register3.py:"
         f"{lineno(G, 'explore_g_register3.py', '\"discovered\"')}) | {paired(R3, BH, rb)[:2]}, p = {paired(R3, BH, rb)[2]:.4f}: "
         "\"not discovered\" | tex:263 '\"not discovered\" on each' | same |")
# S73
L.append(f"| S73 | \"the register carries the cue\" REG3_NUDGE_D ROUTED-BOUND >= 9/10 (explore_g_register3_delta.py:"
         f"{lineno(G, 'explore_g_register3_delta.py', 'the register carries the cue')}) | {cnt(R3ND, rb)}/10: applies | tex:263 '10/10 on delta' "
         "(not named for delta) | not named |")
L.append(f"| S73 | \"discovered (register)\" REG3_D beats BASELINE_D (explore_g_register3_delta.py:"
         f"{lineno(G, 'explore_g_register3_delta.py', 'discovered (register)')}) | {paired(R3D, BD, rb)[:2]}, p = "
         f"{paired(R3D, BD, rb)[2]:.4f}: does not apply | tex:263 'the pre-fixed reading is \"not discovered\" on each' | **differs**: S73's "
         "docstring has no 'not discovered' label (its readings apply or not); the draft carries S72's label over to S73 |")
L.append(f"| S73 | \"discovered (residual gate)\", \"discovered (latch)\" | {paired(RES, BD, rb)[:2]}, {paired(L3D, BD, rb)[:2]}: do not "
         "apply | tex:263 'routes 0/10', '0/10' (table) | not named; counts agree |")
lh = sum(1 for z in zend.values() if z['CTX'] >= 0.9 and z['KEY'] <= 0.1 and z['VAL'] <= 0.1)
L.append(f"| S73 | \"the latch holds on delta memory\" end z >= 0.9 at CTX, <= 0.1 at KEY and VAL on >= 8/10 | {lh}/10: applies | "
         "tex:263 'holds an exact latch when nudged (9/10 routed, z = 1 at the context token and 0 elsewhere)' | the draft's "
         "'9/10 routed' is ROUTED-BOUND, not the reading's z count (also 9/10); the reading is not named |")
L.append(f"| S73 | \"the delta stack carries the cue\" BASELINE_D probe L3 >= 0.9 at K and V on >= 10/20 (descriptive) | "
         f"{sum(1 for r in BD.values() if min(pk(r)) >= 0.9)}/20: does not apply | tex:263 'on delta memory the stack can learn "
         "stream-tagged keys ...' (marked as a description of the 2x2) | the reading's outcome (does not apply) is not stated |")
L.append(f"| S73 | \"the gate is not needed on this task\" SINGLE_D BOUND >= 8/10 (descriptive) | {cnt(SGL, bound)}/10: does not apply | "
         "tex:263 'one delta channel binds the layout 7/10 with no gate' | not named; count agrees |")
L.append(f"| S82 | \"discovered\", \"the device helps\", \"the device hurts\", \"a working write without the device's cost\" "
         f"(explore_g_devices.py:{lineno(S82SPEC, 'explore_g_devices.py', 'READINGS (ROUTED-BOUND')}-"
         f"{lineno(S82SPEC, 'explore_g_devices.py', 'a working write without')} at {S82SPEC}) | not computed here as readings: no S82 "
         "report exists; the records are labelled progress | none | — |")
L.append("")

# register gate vs code
L.append("## Register gate description (tex:125) vs code")
L.append("")
L.append(f"Code at `{G}`: explore_g_regmodel.py (RegBDH, to_reg), explore_g_rhl_child.py (Reg3BDH, make_reg3, make_model), "
         "explore_g_register_child.py (nudge). REG3 is `make_reg3(task, seed, mem, reg=True, gated=(1, 2, 3), inp='window')`.")
L.append("")
L.append("| item | draft | code | status | note |")
L.append("|---|---|---|---|---|")
for it, d_, c_, s_, nt in reg_rows:
    L.append(f"| {esc(it)} | {esc(d_)} | {esc(c_)} | **{s_}** | {esc(nt)} |")
L.append("")

# S82
L.append("## S82 state")
L.append("")
L.append(f"- Spec: `{S82SPEC}` ({ctime(S82SPEC)}) adds {', '.join(spec_files)} only; at that commit the G tree has "
         f"{'no S82 output' if not at_spec else at_spec}. The two files have one commit each on the branch "
         f"({', '.join(spec_hist)}): unchanged since the spec.")
L.append(f"- First S82 record: `{first_rec[0]}` ({first_rec[1]}, {first_rec[2]} records), after the spec. Every later G commit is "
         "'S82 progress (EXPLORATORY, not a result)'.")
L.append(f"- Phase 1 complete and the best device chosen: `s82_best.json` added at `{best_add}`: {json.dumps(best)}.")
L.append(f"- At the draft's commit ({draft_t}) the last G commit is `{gdraft}`: {len(st_d['runs'])} records, phase 1 "
         f"{'complete (104/104)' if ph1 else 'incomplete'}, phase 2 {sum(len(v) for a, v in rr_d.items() if a.endswith('_D'))}/32.")
L.append(f"- At the fetched head `{ghead}` ({ctime(GBR)}): {len(st_h['runs'])} records; phase 2 "
         f"{sum(len(v) for a, v in rr_h.items() if a.endswith('_D'))}/32. No S82 report on the branch (report files: "
         f"{rep82 or 'none after report_4.md'}).")
L.append(f"- The brief's 'head 83699a2' is the spec commit; the branch has moved {len(git('rev-list', f'{S82SPEC}..{GBR}').split())} "
         "commits past it, all S82 output.")
L.append("")
L.append(f"S82 records at `{gdraft}` (outcomes as explore_g_rhl_parent; descriptive only, no report exists):")
L.append("")
L.append(s82_table(rr_d))
L.append("")
L.append(f"Paired on 2100-2119 (ROUTED-BOUND, one-sided McNemar; descriptive): REG3 vs BASELINE {reg82[0]} vs {reg82[1]} "
         f"(p = {reg82[2]:.4f}); REG3_KEYHINGE vs BASELINE {kh82[0]} vs {kh82[1]} (p = {kh82[2]:.2g}); REG3_KEYHINGE vs REG3 "
         f"{kr82[0]} vs {kr82[1]} (p = {kr82[2]:.2g}).")
L.append("")
L.append("What this means for the draft's S82 statements: 'running' (tex:41, :502) is true at the draft's commit and at the head "
         "(phase 2). 'Screening three devices' (tex:265) and 'the devices under screen' (tex:449) describe phase 1, which had "
         "finished 15 minutes before the draft's commit and had chosen REG3_KEYHINGE. The draft states no S82 number, which is "
         "consistent with G not having reported S82; but tex:265's projection 'at the observed rate' uses only S72/S73's 4/20, "
         f"while the S82 records on new seeds (2100-2119) available at the draft's commit hold a second label-free REG3 vs "
         f"BASELINE observation ({cnt(rr_d['REG3'], rb)}/20 vs {cnt(rr_d['BASELINE'], rb)}/20 ROUTED-BOUND). The provenance row "
         "pins 93422cf, which predates S82's spec, and gives no S82 count. Note also that S82's own BACKGROUND (explore_g_devices.py:"
         f"{lineno(S82SPEC, 'explore_g_devices.py', 'in the other 16')}) repeats report 4's '16 ... key split', which the draft "
         "has corrected to 14.")
L.append("")

# descriptive statements
L.append("## Descriptive statements: marked or not")
L.append("")
L.append("Statements in the draft's Session G material that are presented as findings but are not pre-fixed readings. "
         "'Marked' means the draft labels them as description or interpretation where they appear.")
L.append("")
DESC = [
    (259, "where the decayed sum of the block still carries the header", "no",
     "interpretation (report_1.md 'Anything unexpected'); the init probe numbers are descriptive"),
    (259, "Training removes the cue: the gate reads only the window, nothing asks the stack to keep it", "no",
     "the numbers are descriptive (trained < init); the cause is an interpretation"),
    (259, "a gate over the residual stream therefore had nothing to read, which the routing study predicts", "no",
     "S55's docstring attaches a different consequence to 'absent': 'the fix is upstream and S56's register is mandatory'"),
    (261, "The nudged arm is the informative failure: its register writes a clean latch ...", "no",
     "S56's ALWAYS REPORTED diagnostics (descriptive); the reading that applied, 'architecture insufficient', is not named"),
    (261, "The interpretation, which is ours and not a reading, is that an unrouted Hebbian memory at the first layer mixes the "
          "streams ...; Revision~8's design kept layer~1 on the window gate, and that choice is what sank its Hebbian register", "yes, "
     "first clause; the second clause after the semicolon is outside the marking as written",
     "the causal claim crosses tasks: S56 is the fixed P = 8 layout, where the only layer evidence is post-hoc ORACLE23 0/3 "
     "vs ORACLE_H 1/2 (one paired seed); S71/S72 are on RandHeaderTask-lite, and REG3 also changes the gate's input (v_t -> "
     "window)"),
    (263, "Descriptively: in every routed unnudged run beta^R ... and in the runs that failed the gate had locked into a key split", "yes", ""),
    (263, "The unbound Hebbian runs sit at chance between the streams in every cell, as the layers-2--3 oracle did", "no",
     "a new sentence after the 'Descriptively' one; S72's plateau check was conditional on 'architecture insufficient' (not "
     "triggered); not true of every run (see rows)"),
    (263, "on delta memory the stack can learn stream-tagged keys, and when it does the gate does not need to route", "yes",
     "'a description of a 2x2 the screen fixed as descriptive'; the pre-fixed 'the delta stack carries the cue' did not apply (7/20)"),
    (265, "The architecture that carries a distant cue label-free exists ...; its capacity is not the limit", "no",
     "label-free discovery's pre-fixed reading is 'not discovered'; 'capacity is not the limit' is an inference from the nudged arm"),
    (265, "What is missing is the label-free discovery of the register's write at the context token, which fails ... by a key split that forms first", "no",
     "tex:338 lists 'the register's write at the cue is what separates a discovering run' among statements 'marked where it "
     "appears'; here it is not marked"),
    (265, "The audit's per-layer gate over the residual stream is closed: with nothing in the stack to read, it key-splits", "no",
     "no pre-fixed reading 'closed'; S73's only residual-gate reading is 'discovered (residual gate)' (did not apply)"),
    (312, "at a distance, a key split that forms before the register's write at the cue has appeared", "no",
     "same description as tex:265; the records agree on the order (every key-split failure shows the split by the first "
     "recorded evaluation and no CTX write before it)"),
    (36, "the failures being key splits that form first", "no", "abstract; 14 of 16 per memory; the order agrees with the records"),
    (338, "Statements the sessions made beyond their pre-fixed readings are quoted ... as descriptions ... Each is marked where it appears", "—",
     "true at tex:263; the same statements recur unmarked at tex:36, :265, :312, :449, :355"),
    (355, "the residual gate had nothing to read; ... that design choice is what sank its Hebbian register", "no",
     "the interpretation marked as 'ours' at tex:261 is restated here under '\\textbf{Done.}' without the marking"),
    (378, "that the register at layers 2--3 would carry the cue on Hebbian memory (it needed layer~1)", "no",
     "the first half is S56 R1 ('architecture insufficient'); '(it needed layer 1)' is the tex:261 interpretation"),
    (449, "the write at the cue is found on a fifth of seeds, and a key split forms first on the rest", "no",
     "the count is right (4/20); the rest are key splits in 14 of 16; the order agrees with the records"),
    (459, "is found label-free on a fifth of seeds; on Hebbian memory the first layer is the one that must be routed", "no (count)",
     "the count is right but its reading was 'not discovered'; the layer clause is S71's reading on perfect gates"),
]
L.append("| tex line | statement | marked? | note |")
L.append("|---|---|---|---|")
for n_, s_, m_, nt in DESC:
    L.append(f"| tex:{n_} | {esc(s_)} | {esc(m_)} | {esc(nt)} |")
L.append("")
L.append("Statements backed by pre-fixed readings (no flag): tex:259 'absent'; tex:261 the three S71 readings; tex:263 S72 R1 and "
         "'not discovered' (S72); tex:314 'a Hebbian stack needs its first layer routed and a delta stack does not' (S71 'layer 1 "
         "is the block', 'delta tolerates an unrouted layer 1', on perfect gates); tex:380 'One screen (S71) settled it'; tex:415.")
L.append("")
L.append("Not checkable from the records: whether a main-line register test is 'pre-registered' (no specification is committed "
         "on the draft branch or the main line); the reasoning attributed to Revision 8's specification at tex:380 (the main "
         "line's distant_cues_2026-10.md:155 says only 'layer 1 keeps the window gate (no residual yet)').")
L.append("")
with open(OUT, "w") as fh:
    fh.write("\n".join(L))
print(f"wrote {OUT}: {len(ROWS)} rows, {dict(C)}")
