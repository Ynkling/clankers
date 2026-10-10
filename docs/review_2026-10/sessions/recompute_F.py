#!/usr/bin/env python3
"""recompute_F.py — the referee's recomputation of Session F's reports 3, 4 and 5 (branch claude/explore-F, pinned head
5b9bc2a) from the JSON run records at that commit.

Every number the three reports print is typed in below, with its report file and line, as the REPORTED value (a string,
exactly as printed). Nothing typed from a report or a docstring is used as an input: every RECOMPUTED value comes from
the records (explore_out/F/*_results.json at 5b9bc2a), through rlib's outcome definitions or, where a screen defines
its own outcome, an implementation of the screen docstring's definition (cited by file:line). Thresholds of outcomes and
readings are definitions and are taken from the docstrings (cited).

Writes docs/review_2026-10/sessions/recompute_F.md. Read-only otherwise (git show / git log only). Python stdlib only.
"""
import datetime
import json
import math
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from rlib import (load, runs, transition, bound, discovered, bound_routed, one_to_one, fail_class, outcome_k,  # noqa: E402
                  mcnemar1, fisher1, wilson, band, pairs, TIES, ROOT)

PIN = "5b9bc2a"
BRANCH = "origin/claude/explore-F"
R3, R4, R5 = "explore_out/F/report_3.md", "explore_out/F/report_4.md", "explore_out/F/report_5.md"
OUT = os.path.join(HERE, "recompute_F.md")


def git(*a):
    return subprocess.check_output(("git", "-C", ROOT) + a).decode()


def store(name, ref=PIN):
    return load(ref, f"explore_out/F/{name}_results.json")


PR, HD, RH, PB, EI = (store(n) for n in ("premise", "hebb_decay", "rh_lite", "probe", "eight"))
DC, DL, WS = store("decay"), store("delta_controls"), store("window_delta_scale")   # S59, S53, S54 (references)


# ── formatting ───────────────────────────────────────────────────────────────
def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def wil(k, n, z=None):
    lo, hi = wilson(k, n) if z is None else wilson(k, n, z)
    return f"[{lo:.2f}, {hi:.2f}]"


def med(xs):
    return statistics.median(xs)


def cnt(f, rs):
    return sum(1 for r in rs.values() if f(r))


ROWS = []


def row(ref, item, reported, recomputed, status=None, note=""):
    """status None: MATCH if the printed strings agree, else DISAGREE."""
    if status is None:
        status = "MATCH" if str(reported).strip() == str(recomputed).strip() else "DISAGREE"
    ROWS.append(dict(ref=ref, item=item, rep=str(reported), rec=str(recomputed), status=status, note=note))


def row_wilson(ref, item, reported, k, n):
    a = wil(k, n)
    if a == reported:
        row(ref, item, reported, a)
    elif wil(k, n, 1.96) == reported:
        row(ref, item, reported, a, "PARTIAL", "rounding: matches with z = 1.96 (explore_f_common.wilson), not with z = 1.95996")
    else:
        row(ref, item, reported, a)


# ── outcome definitions specific to the screens ─────────────────────────────
def routed_star_end(r):
    """S70 ROUTED*@end (explore_f_rh_lite.py:19): margin >= 0.9 and η² by stream at KEY > 0.9, end statistics."""
    e = r["end"]
    return e["margin"] >= 0.9 and e["eta_key_by_stream"] > 0.9


def sr_outcome_recorded(r):
    """test_stream_recipe.outcome at 9c5939e (test_stream_recipe.py:751-762) on the RECORDED transition and the RECORDED
    end.one_to_one (as the main line computes it). Used only to cross-check rlib.outcome_k (curve-recomputed transition,
    stream_gate-recomputed one-to-one)."""
    e = r["end"]
    if r["transition"] is not None:
        return "BOUND ROUTED" if (e["one_to_one"] and min(e["stream_acc"]) >= 0.9) else "BOUND NOT routed"
    from rlib import fail_class_k, shared_max
    fc = fail_class_k(r)
    if fc == "STREAM-PARTIAL":
        return f"MERGED ({shared_max(r)} share)" if shared_max(r) >= 2 else "STREAM-PARTIAL one-to-one"
    return f"non-stream {fc}"


def br(r):
    """BOUND ROUTED (test_stream_recipe.outcome; explore_f_eight.py:35, explore_f_window_delta_scale.py:21)."""
    return bool(r.get("ok")) and outcome_k(r) == "BOUND ROUTED"


OUTCOME_XCHECK = []
for nm, st in (("eight", EI), ("window_delta_scale", WS)):
    for k, r in st["runs"].items():
        if r.get("ok") and "stream_gate" in r.get("end", {}):
            a, b = outcome_k(r), sr_outcome_recorded(r)
            if a != b or transition(r["curve"]) != r["transition"]:
                OUTCOME_XCHECK.append((nm, k.split("|")[0], k.split("|")[1], a, b, transition(r["curve"]), r["transition"]))


def trans_list(rs, f=bound):
    return sorted(transition(r["curve"]) for r in rs.values() if f(r))


def mid(xs):
    return f"{med(xs):g}"


# ════════════════════════════════════════════════════════════════════════════
# REPORT 3 — S68 premise, S69 hebb_decay, S70 rh_lite
# ════════════════════════════════════════════════════════════════════════════
S4, S8 = runs(PR, "S68_SINGLE_S4"), runs(PR, "S68_SINGLE_S8")
O4, O8 = runs(PR, "S68_ORACLE_S4"), runs(PR, "S68_ORACLE_S8")
H = {d: runs(HD, f"S69_{d}") for d in ("FIXED", "ROUTED", "GDN", "ORACLE")}
S59 = {d: runs(DC, f"SINGLE_{d}") for d in ("FIXED", "ROUTED", "GDN")}
OH, OD = runs(RH, "S70_ORACLE_HEBB"), runs(RH, "S70_ORACLE_DELTA")
RHH, RHD = runs(RH, "S70_HEBB"), runs(RH, "S70_DELTA")
B_HEBB, C_HEBB, C_DELTA = runs(WS, "B_HEBB"), runs(WS, "C_HEBB"), runs(WS, "C_DELTA")
C_OD = runs(WS, "C_ORACLE_DELTA")
S53H = runs(DL, "SINGLE_DELTA_H")
S53_OHP8 = runs(DL, "ORACLE_HEBB_P8")
ALL3 = [r for st in (PR, HD, RH) for r in st["runs"].values()]

# header / provenance
shas = sorted({r["code_sha"][:12] for r in ALL3})
row(f"{R3}:3", "code SHA (one for all three screens)", "ff908bdbc0b2", ", ".join(shas))
row(f"{R3}:3", "CPU", "Intel Xeon @ 2.10GHz", ", ".join(sorted({r["cpu"] for r in ALL3})), "MATCH",
    "the report abbreviates 'Intel(R) Xeon(R) Processor @ 2.10GHz'")
row(f"{R3}:4", "torch", "2.14.0+cu130", ", ".join(sorted({r["torch"] for r in ALL3})))
row(f"{R3}:4", "1 thread per run", "1 thread per run", "not in run records", "NOT RECORD-CHECKABLE",
    "the records have no thread field; followup.log header says 'torch threads 1'")
row(f"{R3}:4", "runs ok / total", "102/102", f"{sum(1 for r in ALL3 if r.get('ok'))}/{len(ALL3)}")
row(f"{R3}:4", "none retried (attempts = 1 in every record)", "none retried",
    "none retried" if all(r.get("attempts") == 1 for r in ALL3) else f"max attempts {max(r.get('attempts') for r in ALL3)}")
dbeta = sorted({r["delta_cfg"]["beta"] for st, arms in ((PR, ["S68_SINGLE_S4", "S68_SINGLE_S8", "S68_ORACLE_S4", "S68_ORACLE_S8"]),
                                                         (RH, ["S70_DELTA", "S70_ORACLE_DELTA"]))
                for a in arms for r in runs(st, a).values()})
row(f"{R3}:4", "β fixed at 1 in every delta arm (delta_cfg.beta of S68 and S70 delta arms)", "1", ", ".join(f"{b:g}" for b in dbeta))
hbeta = sorted({r["delta_cfg"]["beta"] for d in ("FIXED", "ROUTED", "GDN", "ORACLE") for r in H[d].values()})
row(f"{R3}:5", "S69 Hebbian arms have no erase (delta_cfg.beta)", "0 (no erase)", ", ".join(f"{b:g}" for b in hbeta) + " (no erase)")

# S68 counts
row(f"{R3}:12", "S68_SINGLE_S4 seeds; updates", "300-309, 43200",
    f"{min(S4)}-{max(S4)}, {','.join(sorted({str(r['iters']) for r in S4.values()}))}")
k4, k8 = cnt(bound, S4), cnt(bound, S8)
row(f"{R3}:12", "S68_SINGLE_S4 BOUND", "2/10", f"{k4}/{len(S4)}")
row_wilson(f"{R3}:12", "S68_SINGLE_S4 BOUND Wilson 95%", "[0.06, 0.51]", k4, len(S4))
row(f"{R3}:12", "S68_SINGLE_S4 band", "MINORITY", band(k4, len(S4)))
bh_br = cnt(br, B_HEBB)
row(f"{R3}:12", "reference S54(b) B_HEBB BOUND ROUTED (window_delta_scale records, test_stream_recipe.outcome)", "19/20",
    f"{bh_br}/{len(B_HEBB)}")
p = fisher1(bh_br, len(B_HEBB), k4, len(S4))
row(f"{R3}:12", "Fisher one-sided p (B_HEBB BOUND ROUTED vs S68_SINGLE_S4 BOUND)", "6e-5", f"{p:.1e}", "MATCH" if f"{p:.0e}" == "6e-05" else None,
    f"exact {p:.3g}; prints as 6e-5 at one significant figure")
row(f"{R3}:13", "S68_SINGLE_S8 BOUND", "0/10", f"{k8}/{len(S8)}")
row_wilson(f"{R3}:13", "S68_SINGLE_S8 BOUND Wilson 95%", "[0.00, 0.28]", k8, len(S8))
row(f"{R3}:13", "S68_SINGLE_S8 band", "NEVER", band(k8, len(S8)))
ch_br = cnt(br, C_HEBB)
row(f"{R3}:13", "reference S54(c) C_HEBB BOUND ROUTED", "9/10", f"{ch_br}/{len(C_HEBB)}")
p = fisher1(ch_br, len(C_HEBB), k8, len(S8))
row(f"{R3}:13", "Fisher one-sided p (C_HEBB BOUND ROUTED vs S68_SINGLE_S8 BOUND)", "6e-5", f"{p:.1e}", "MATCH" if f"{p:.0e}" == "6e-05" else None,
    f"exact {p:.3g}")
# S68 validity and reading (explore_f_premise.py:19-24)
v4, v8 = cnt(bound, O4) >= 1, cnt(bound, O8) >= 1
if not (v4 and v8):
    rd68 = "UNTESTED"
elif k8 >= 7:
    rd68 = "context-tagged keys scale"
elif k8 <= 3 and k4 >= 7:
    rd68 = "the partition's advantage is in the number of streams"
else:
    rd68 = "neither"
row(f"{R3}:13", "S68 reading (explore_f_premise.py:19-24)", "neither", rd68)
row(f"{R3}:14", "S68 oracle seeds (both S)", "300-301", f"{min(O4)}-{max(O4)}" if sorted(O4) == sorted(O8)
    else f"S=4 {sorted(O4)}, S=8 {sorted(O8)}")
row(f"{R3}:14", "S68_ORACLE_S4 / S68_ORACLE_S8 BOUND", "2/2 / 2/2", f"{cnt(bound, O4)}/{len(O4)} / {cnt(bound, O8)}/{len(O8)}")
ot = sorted({transition(r["curve"]) for r in list(O4.values()) + list(O8.values())})
row(f"{R3}:14", "S68 oracle transitions", "all at 1200", "all at " + ", ".join(map(str, ot)) if len(ot) == 1 else str(ot))
row(f"{R3}:14", "S68 validity (oracle binds >= 1/2 at both S; explore_f_premise.py:19-20)", "valid",
    "valid" if (v4 and v8) else "not valid")
# S69
row(f"{R3}:15", "S69 seeds; updates", "350-359, 48000",
    f"{min(H['FIXED'])}-{max(H['FIXED'])}, {','.join(sorted({str(r['iters']) for d in ('FIXED', 'ROUTED', 'GDN') for r in H[d].values()}))}")
kh = {d: cnt(bound, H[d]) for d in ("FIXED", "ROUTED", "GDN")}
row(f"{R3}:15", "S69 FIXED / ROUTED / GDN BOUND", "0/10 / 0/10 / 0/10", " / ".join(f"{kh[d]}/{len(H[d])}" for d in kh))
ks59 = {d: cnt(bound, S59[d]) for d in S59}
row(f"{R3}:15", "S59 delta single channel FIXED / ROUTED / GDN BOUND (decay records, same seeds)", "3/10 / 9/10 / 8/10",
    " / ".join(f"{ks59[d]}/{len(S59[d])}" for d in ks59))
same = all(set(S59[d]) == set(H[d]) for d in S59)
row(f"{R3}:15", "S59 arms on the same seeds as S69", "same seeds", "same seeds" if same else "different seeds")
vh = cnt(bound, H["ORACLE"]) >= 1
needs = kh["ROUTED"] <= kh["FIXED"] + 1 and kh["GDN"] <= kh["FIXED"] + 1
alone = kh["ROUTED"] >= 8 or kh["GDN"] >= 8
rd69 = ("UNTESTED" if not vh else "the forget gate needs the delta rule" if needs and not alone else
        "the forget gate alone suffices" if alone and not needs else "both" if needs else "neither")
row(f"{R3}:15", "S69 reading (explore_f_hebb_decay.py:21-25)", "the forget gate needs the delta rule", rd69)
ho = H["ORACLE"]
row(f"{R3}:16", "S69_ORACLE seeds", "350-351", f"{min(ho)}-{max(ho)}")
row(f"{R3}:16", "S69_ORACLE BOUND", "1/2", f"{cnt(bound, ho)}/{len(ho)}")
row(f"{R3}:16", "S69_ORACLE transition of the bound seed", "31200", ", ".join(str(t) for t in trans_list(ho)))
other = [r["acc"] for r in ho.values() if not bound(r)]
row(f"{R3}:16", "S69_ORACLE final accuracy of the unbound seed", "0.75", ", ".join(f2(a) for a in other), None,
    f"record acc of seed {[s for s, r in ho.items() if not bound(r)]} = {other} (last evaluation at 48000); prints 0.76")
row(f"{R3}:16", "S69 validity (oracle >= 1/2; explore_f_hebb_decay.py:21)", "valid (≥ 1/2)", "valid (≥ 1/2)" if vh else "not valid")
# S70 phase 1
row(f"{R3}:17", "S70 oracle seeds; updates", "330-332, 24000",
    f"{min(OH)}-{max(OH)}, {','.join(sorted({str(r['iters']) for r in list(OH.values()) + list(OD.values())}))}")
row(f"{R3}:17", "S70_ORACLE_HEBB / S70_ORACLE_DELTA BOUND", "3/3 / 3/3", f"{cnt(bound, OH)}/{len(OH)} / {cnt(bound, OD)}/{len(OD)}")
ph1 = cnt(bound, OH) >= 2 and cnt(bound, OD) >= 2      # explore_f_rh_lite.py:12-14
budget = 24000 if ph1 else "double"
p2iters = sorted({r["iters"] for r in list(RHH.values()) + list(RHD.values())})
row(f"{R3}:17", "S70 phase-1 decision and phase-2 budget (rule explore_f_rh_lite.py:12-14; phase-2 records' iters)",
    "phase 1 bound: budget 24000", f"phase 1 {'bound' if ph1 else 'double'}: budget {budget}; phase-2 iters {p2iters}",
    "MATCH" if ph1 and p2iters == [24000] else "DISAGREE")
# S70 phase 2
row(f"{R3}:18", "S70_HEBB seeds", "330-349", f"{min(RHH)}-{max(RHH)} (n {len(RHH)})", "MATCH" if (min(RHH), max(RHH), len(RHH)) == (330, 349, 20) else "DISAGREE")
dh, bh, rh_ = cnt(discovered, RHH), cnt(bound, RHH), cnt(routed_star_end, RHH)
dd, bd, rd_ = cnt(discovered, RHD), cnt(bound, RHD), cnt(routed_star_end, RHD)
row(f"{R3}:18", "S70_HEBB DISCOVERED", "0/20", f"{dh}/{len(RHH)}")
row_wilson(f"{R3}:18", "S70_HEBB DISCOVERED Wilson", "[0.00, 0.16]", dh, len(RHH))
row(f"{R3}:18", "S70_HEBB BOUND", "1/20", f"{bh}/{len(RHH)}")
row_wilson(f"{R3}:18", "S70_HEBB BOUND Wilson", "[0.01, 0.24]", bh, len(RHH))
row(f"{R3}:18", "S70_HEBB ROUTED*@end", "0/20", f"{rh_}/{len(RHH)}")
row(f"{R3}:19", "S70_DELTA DISCOVERED", "0/20", f"{dd}/{len(RHD)}")
row(f"{R3}:19", "S70_DELTA BOUND", "6/20", f"{bd}/{len(RHD)}")
row_wilson(f"{R3}:19", "S70_DELTA BOUND Wilson", "[0.15, 0.52]", bd, len(RHD))
row(f"{R3}:19", "S70_DELTA ROUTED*@end", "0/20", f"{rd_}/{len(RHD)}")
a, b, n = pairs(RHD, RHH, bound)
row(f"{R3}:19", "S70 paired BOUND: delta only, hebb only", "delta only 5, hebb only 0", f"delta only {a}, hebb only {b}")
p = mcnemar1(a, b)
row(f"{R3}:19", "S70 one-sided exact McNemar p(delta > hebb), BOUND", "0.031", f"{p:.3f}", None, f"exact {p:.5g} (n {n} paired seeds)")


def fails(rs):
    out = {}
    for r in rs.values():
        if not bound(r):
            c = fail_class(r)
            out[c] = out.get(c, 0) + 1
    return out


def bound_unrouted(rs):
    return sum(1 for r in rs.values() if bound(r) and not routed_star_end(r))


fh, fd = fails(RHH), fails(RHD)
row(f"{R3}:21", "S70_HEBB failure classes of unbound runs", "KEY 19", ", ".join(f"{k} {v}" for k, v in sorted(fh.items())), None,
    "fail_class (test_router_layout rule, explore_f_rh_lite.py:20-21) on the 19 unbound runs")
row(f"{R3}:21", "S70_HEBB BOUND but not ROUTED*@end", "1 BOUND unrouted", f"{bound_unrouted(RHH)} BOUND unrouted")
row(f"{R3}:21", "S70_DELTA failure classes of unbound runs", "KEY 14", ", ".join(f"{k} {v}" for k, v in sorted(fd.items())))
row(f"{R3}:21", "S70_DELTA BOUND but not ROUTED*@end", "6 BOUND unrouted", f"{bound_unrouted(RHD)} BOUND unrouted")
row(f"{R3}:22", "S68 S=4 transitions", "18000, 33600", ", ".join(map(str, trans_list(S4))))
td = trans_list(RHD)
row(f"{R3}:22", "S70_DELTA transitions range", "3600-16800", f"{td[0]}-{td[-1]}")
row(f"{R3}:22", "S70_DELTA transitions median", "5400", mid(td), None, f"all: {td}")
row(f"{R3}:22", "S70_HEBB transition", "13200", ", ".join(map(str, trans_list(RHH))))
ks1 = sorted({r["k"] for a in ("S68_SINGLE_S4", "S68_SINGLE_S8") for r in runs(PR, a).values()}
             | {r["k"] for d in ("FIXED", "ROUTED", "GDN") for r in H[d].values()})
row(f"{R3}:21-22", "S68 / S69 single arms have k = 1", "k = 1", "k = " + ", ".join(map(str, ks1)))
# Readings text
row(f"{R3}:26", "S68 SINGLE BOUND at S=4 / S=8 (reading text)", "2/10 / 0/10", f"{k4}/{len(S4)} / {k8}/{len(S8)}")
row(f"{R3}:26", "threshold of 'context-tagged keys scale' as stated", ">= 7/10 at S = 8", ">= 7/10 at S = 8", "MATCH",
    "docstring comparison, not records: explore_f_premise.py:22")
row(f"{R3}:27", "threshold of 'the partition's advantage is in the number of streams' as stated", ">= 7/10 at S = 4",
    ">= 7/10 at S = 4 and <= 3/10 at S = 8", "PARTIAL",
    "docstring comparison: explore_f_premise.py:23 also requires S68_SINGLE_S8 <= 3/10; the report omits it (it holds, 0/10)")
first_eval = {r["curve"][0][0] for r in list(O4.values()) + list(O8.values())}
row(f"{R3}:27", "both S68 oracles bind at the first evaluation", "first evaluation",
    "first evaluation" if all(transition(r["curve"]) == r["curve"][0][0] for r in list(O4.values()) + list(O8.values())) else "not all",
    None, f"first evaluation step {sorted(first_eval)}")
row(f"{R3}:28", "gated Hebbian recipe binds (BOUND ROUTED, S54 B_HEBB and C_HEBB records)", "19/20 and 9/10",
    f"{bh_br}/{len(B_HEBB)} and {ch_br}/{len(C_HEBB)}", None,
    f"BOUND alone: B_HEBB {cnt(bound, B_HEBB)}/{len(B_HEBB)}, C_HEBB {cnt(bound, C_HEBB)}/{len(C_HEBB)}")
row(f"{R3}:29", "S53 SINGLE_DELTA_H BOUND (delta_controls records)", "8/10", f"{cnt(bound, S53H)}/{len(S53H)}")
u4 = [r["acc"] for r in S4.values() if not bound(r)]
u8 = [r["acc"] for r in S8.values() if not bound(r)]
row(f"{R3}:30", "S68 single channel final accuracy range at S=4 (unbound runs)", "0.22-0.75", f"{f2(min(u4))}-{f2(max(u4))}", None,
    "range over the 8 unbound runs; over all 10 runs it is " + f"{f2(min(r['acc'] for r in S4.values()))}-{f2(max(r['acc'] for r in S4.values()))}")
row(f"{R3}:31", "S68 single channel final accuracy range at S=8", "0.29-0.59", f"{f2(min(u8))}-{f2(max(u8))}")
row(f"{R3}:32", "S59 delta BOUND fixed / routed / GDN (reading text)", "3 / 9 / 8", " / ".join(str(ks59[d]) for d in ks59))
row(f"{R3}:33", "S69 Hebbian BOUND fixed / routed / GDN (reading text)", "0 / 0 / 0", " / ".join(str(kh[d]) for d in kh))


def alpha_med(rs, layer, role):
    return med([r["decay_by_role"][layer]["ch0"][role] for r in rs.values()])


ROLES = ("ctx", "key", "val")
row(f"{R3}:34", "S69_ROUTED median α layer 1 CTX / KEY / VAL", "0.975 / 0.999 / 0.982",
    " / ".join(f3(alpha_med(H["ROUTED"], 0, ro)) for ro in ROLES), None,
    "unrounded " + " / ".join(f"{alpha_med(H['ROUTED'], 0, ro):.5f}" for ro in ROLES))
row(f"{R3}:34", "S69_GDN median α layer 1 CTX / KEY / VAL", "0.988 / 0.999 / 0.990",
    " / ".join(f3(alpha_med(H["GDN"], 0, ro)) for ro in ROLES))
row(f"{R3}:35", "S59 SINGLE_ROUTED (delta) median α layer 1 CTX / KEY / VAL", "0.26 / 1.00 / 1.00",
    " / ".join(f2(alpha_med(S59["ROUTED"], 0, ro)) for ro in ROLES))
row(f"{R3}:35", "S59 SINGLE_GDN (delta) median α layer 1 CTX / KEY / VAL", "0.18 / 1.00 / 1.00",
    " / ".join(f2(alpha_med(S59["GDN"], 0, ro)) for ro in ROLES))
lowest = []
for d in ("ROUTED", "GDN"):
    ctx = [alpha_med(H[d], i, "ctx") for i in range(3)]
    lowest.append((ctx.index(min(ctx)) + 1, min(ctx)))
row(f"{R3}:35", "S69 lowest median CTX α: layer (ROUTED, GDN values)", "layer 3 (0.89, 0.68)",
    f"layer {'/'.join(sorted({str(l) for l, _ in lowest}))} ({f2(lowest[0][1])}, {f2(lowest[1][1])})")
row(f"{R3}:36", "S69 median final accuracy ROUTED, GDN / FIXED", "0.17 (routed, GDN) against 0.22 fixed",
    f"{f2(med([r['acc'] for r in H['ROUTED'].values()]))}, {f2(med([r['acc'] for r in H['GDN'].values()]))} "
    f"against {f2(med([r['acc'] for r in H['FIXED'].values()]))}", "MATCH" if (
        f2(med([r['acc'] for r in H['ROUTED'].values()])) == "0.17" and f2(med([r['acc'] for r in H['GDN'].values()])) == "0.17"
        and f2(med([r['acc'] for r in H['FIXED'].values()])) == "0.22") else "DISAGREE")
row(f"{R3}:38-39", "S70_HEBB discovers / binds / key-splits", "0/20, 1/20, 19/20",
    f"{dh}/20, {bh}/20, {fh.get('KEY', 0)}/20")


def eta_med(rs, pos, g):
    return med([r["end"][f"eta_{pos}_by_{g}"] for r in rs.values()])


row(f"{R3}:39", "S70_HEBB median η² by key at KEY / VAL", "1.00", f"{f2(eta_med(RHH, 'key', 'key'))} / {f2(eta_med(RHH, 'val', 'key'))}",
    "MATCH" if f2(eta_med(RHH, 'key', 'key')) == f2(eta_med(RHH, 'val', 'key')) == "1.00" else "DISAGREE")
row(f"{R3}:39", "S70_HEBB median η² by stream at KEY / VAL", "0.00",
    f"{f2(eta_med(RHH, 'key', 'stream'))} / {f2(eta_med(RHH, 'val', 'stream'))}",
    "MATCH" if f2(eta_med(RHH, 'key', 'stream')) == f2(eta_med(RHH, 'val', 'stream')) == "0.00" else "DISAGREE")
row(f"{R3}:39", "S70_HEBB median margin", "0.00", f2(med([r["end"]["margin"] for r in RHH.values()])))
uh = [r["acc"] for r in RHH.values() if not bound(r)]
row(f"{R3}:39", "S70_HEBB final accuracy range (unbound runs)", "0.47-0.52", f"{f2(min(uh))}-{f2(max(uh))}", None,
    f"19 unbound runs; the bound seed ends at {f2(max(r['acc'] for r in RHH.values() if bound(r)))}")
row(f"{R3}:40", "S70_DELTA binds / routes (ROUTED*@end)", "binds 6/20 but routes none", f"binds {bd}/20 but routes {'none' if rd_ == 0 else rd_}")
row(f"{R3}:40", "S70_DELTA unbound runs key-split (all unbound runs KEY)", "also key splits",
    "also key splits" if set(fd) == {"KEY"} else str(fd))
ud = [r["acc"] for r in RHD.values() if not bound(r)]
row(f"{R3}:40", "S70_DELTA final accuracy range when unbound", "0.74-0.88", f"{f2(min(ud))}-{f2(max(ud))}")
row(f"{R3}:41", "Session G BASELINE: S70_HEBB DISCOVERED, BOUND, ROUTED*@end, KEY",
    "DISCOVERED 0/20, BOUND 1/20, ROUTED*@end 0/20, KEY 19", f"DISCOVERED {dh}/20, BOUND {bh}/20, ROUTED*@end {rh_}/20, KEY {fh.get('KEY', 0)}")
row(f"{R3}:45", "S69 Hebbian oracle bound 1 of 2 within 48000 (31200; the other ...)", "1 of 2; 31200; the other 0.75",
    f"{cnt(bound, ho)} of {len(ho)}; {', '.join(map(str, trans_list(ho)))}; the other {', '.join(f2(a) for a in other)}", None,
    "same quantity as line 16: the unbound oracle seed ends at 0.7617")
t53 = trans_list(S53_OHP8)
row(f"{R3}:46", "S53 Hebbian oracle at P=8 (ORACLE_HEBB_P8) seeds 300-302: bound by 24000", "3/3 by 24000",
    f"{cnt(bound, S53_OHP8)}/{len(S53_OHP8)} by {max(t53)}", "MATCH" if cnt(bound, S53_OHP8) == 3 and max(t53) <= 24000 else "DISAGREE",
    f"transitions {sorted(t53)} (seeds {sorted(S53_OHP8)})")
o70 = [transition(r["curve"]) for r in list(OH.values()) + list(OD.values())]
row(f"{R3}:48", "S70 perfect gates bound by 1200", "5 of 6 seeds", f"{sum(1 for t in o70 if t is not None and t <= 1200)} of {len(o70)} seeds",
    None, f"transitions HEBB {[transition(r['curve']) for s, r in sorted(OH.items())]}, DELTA {[transition(r['curve']) for s, r in sorted(OD.items())]}")
near25 = [s for s, r in S4.items() if not bound(r) and abs(r["acc"] - 0.25) <= 0.05]
near75 = [s for s, r in S4.items() if not bound(r) and abs(r["acc"] - 0.75) <= 0.05]
row(f"{R3}:52", "S68 S=4 unbound seeds near 0.25 (within 0.05) / near 0.75 (within 0.05)", "5 / 2",
    f"{len(near25)} / {len(near75)}", None, f"near 0.25 {sorted(near25)}; near 0.75 {sorted(near75)}; 'near' read as within 0.05 (the report gives no tolerance)")
row(f"{R3}:52", "S68 S=4 final accuracy seeds 305, 308, 309", "0.55, 0.74, 0.75", ", ".join(f2(S4[s]["acc"]) for s in (305, 308, 309)))
row(f"{R3}:57", "explore_f_delta_checks at both drives", "26/26", "not in run records", "NOT RECORD-CHECKABLE",
    "checks files (not run records): checks_s70o_20261009_141958.json 26 ok of 26, checks_followup_20261009_142433.json 26 ok of 26")
row(f"{R3}:58", "the child's CHECKs passed", "15", "not in run records", "NOT RECORD-CHECKABLE", "child CHECK lines are in followup.log only")
row(f"{R3}:60", "S69 routed decay at init equals fixed to", "1.2e-7", "not in run records", "NOT RECORD-CHECKABLE", "a CHECK printed to followup.log")
row(f"{R3}:62", "RandHeaderTask-lite block-length frequencies", "0.34 / 0.32 / 0.34", "not in run records", "NOT RECORD-CHECKABLE",
    "a CHECK printed to followup.log")
orc = [r for st, arms in ((PR, ("S68_ORACLE_S4", "S68_ORACLE_S8")), (HD, ("S69_ORACLE",)), (RH, ("S70_ORACLE_HEBB", "S70_ORACLE_DELTA")))
       for a in arms for r in runs(st, a).values()]
vals = sorted({(r["end"]["margin"], r["end"]["eta_key_by_stream"], r["end"]["eta_val_by_stream"]) for r in orc})
row(f"{R3}:64", "perfect gates: margin 1, η² by stream 1 at KEY and VAL", "margin 1, η² 1 / 1",
    "; ".join(f"margin {m:g}, η² {k_:g} / {v:g}" for m, k_, v in vals) + f" at the end of all {len(orc)} oracle runs", "PARTIAL",
    "the CHECK is on the gate as built (followup.log); the records hold these statistics at the end only (none at step 0), and they agree")
row(f"{R3}:61-63", "other child CHECKs (GDN init equals S59's; compositions; one CTX per block; every (stream, key) once)", "passed",
    "not in run records", "NOT RECORD-CHECKABLE", "followup.log only")
hrs3 = sum(r["secs_wall"] for r in ALL3 if r.get("ok")) / 3600
row(f"{R3}:68", "run time (sum of secs_wall over the three stores, h)", "34.9", f"{hrs3:.1f}", None,
    f"{hrs3:.3f} h; explore_f_reports.store_meta sums secs_wall")
row(f"{R3}:68-69", "phase times 14:20-14:24 / 14:24-23:11 UTC; 8.8 h wall on 4 workers; no restart",
    "14:20-14:24, 14:24-23:11, 8.8 h", "not in run records", "NOT RECORD-CHECKABLE",
    "followup.log headers: 14:19:53, 14:24:25, done 23:11:21 (8.86 h from the first header)")

# ════════════════════════════════════════════════════════════════════════════
# REPORT 4 — S77 probe
# ════════════════════════════════════════════════════════════════════════════
PA = {a: runs(PB, a) for a in ("S53_DELTA", "S59_GDN", "S69_GDN")}
ALLP = [r for rs in PA.values() for r in rs.values()]
row(f"{R4}:3", "code SHA", "623d477f61ad", ", ".join(sorted({r["code_sha"][:12] for r in ALLP})))
row(f"{R4}:3", "CPU; torch", "Intel Xeon @ 2.10GHz; 2.14.0+cu130",
    "; ".join(sorted({r["cpu"] for r in ALLP})) + "; " + ", ".join(sorted({r["torch"] for r in ALLP})), "MATCH", "the report abbreviates the CPU name")
row(f"{R4}:3", "1 thread per run", "1 thread per run", "not in run records", "NOT RECORD-CHECKABLE", "s77.log header: 'torch threads 1'")
row(f"{R4}:25-27", "reading thresholds as stated (L3 >= 0.9 at K and V, L1 <= 0.6, >= 8 runs; difference >= 0.2)",
    ">= 0.9, <= 0.6, >= 8; >= 0.2", ">= 0.9, <= 0.6, >= 8; >= 0.2", "MATCH", "docstring comparison: explore_f_probe.py:22-25")
row(f"{R4}:3", "runs ok / total", "30/30", f"{sum(1 for r in ALLP if r.get('ok'))}/{len(ALLP)}")
row(f"{R4}:4", "every rerun's curve equals its record (record field 'repro')", "30/30", f"{sum(1 for r in ALLP if r.get('repro'))}/{len(ALLP)}",
    None, "the field is the run's own assertion; the original records are on other commits and are not re-compared here")
row(f"{R4}:4-5", "rerun seeds: S53 300-309; S59 / S69 350-359", "300-309; 350-359",
    f"{min(PA['S53_DELTA'])}-{max(PA['S53_DELTA'])}; {min(PA['S59_GDN'])}-{max(PA['S59_GDN'])}")
row(f"{R4}:5-6", "probe copied from claude/explore-G at 3932db7; synthetic check 0.9975 (G 0.998) / 0.474", "0.9975 / 0.474",
    "not in run records", "NOT RECORD-CHECKABLE", "a child CHECK: s77.log prints 'informative feature 0.9975 (G 0.998) >= 0.99; random 0.474'")
npos = {a: sorted({r["probe"]["L3|K"]["n"] for r in rs.values()}) for a, rs in PA.items()}
row(f"{R4}:6-7", "held-out on 400 of 2000 probe sequences", "400 of 2000", f"held-out n per position class {npos}", "PARTIAL",
    "records hold n = 3200 (P = 4) and 6400 (P = 8) held-out positions = 400 sequences x 8 or 16 KEY positions; the 2000 total is "
    "not in the records (s77.log CHECK: '1600 / 400 sequences, disjoint')")


def pm(rs, feat, pos, init=False):
    return med([r["probe_init" if init else "probe"][f"{feat}|{pos}"]["acc"] for r in rs.values()])


TAB4 = {"S53_DELTA": (13, "8/10", "[0.49, 0.94]", ["0.50 / 0.50", "0.64 / 0.57", "0.67 / 0.60", "0.68 / 0.67", "0.75 / 0.63"]),
        "S59_GDN": (14, "8/10", "[0.49, 0.94]", ["0.50 / 0.49", "0.58 / 0.55", "0.57 / 0.56", "0.57 / 0.60", "0.62 / 0.56"]),
        "S69_GDN": (15, "0/10", "[0.00, 0.28]", ["0.50 / 0.49", "0.53 / 0.50", "0.52 / 0.51", "0.53 / 0.51", "0.69 / 0.64"])}
for arm, (ln, kb, wb, cells) in TAB4.items():
    rs = PA[arm]
    k = cnt(bound, rs)
    agree = all(bool(r["bound"]) == bound(r) for r in rs.values())
    row(f"{R4}:{ln}", f"{arm} BOUND (rerun curve; = record field 'bound': {agree})", kb, f"{k}/{len(rs)}")
    row_wilson(f"{R4}:{ln}", f"{arm} BOUND Wilson", wb, k, len(rs))
    for (lab, feat, init), rep in zip((("entering L1", "L1", False), ("entering L2", "L2", False), ("entering L3", "L3", False),
                                       ("final layer", "final", False), ("L3 at init", "L3", True)), cells):
        row(f"{R4}:{ln}", f"{arm} probe median {lab} K / V", rep, f"{f2(pm(rs, feat, 'K', init))} / {f2(pm(rs, feat, 'V', init))}")


def split(rs, b):
    return {s: r for s, r in rs.items() if bound(r) == b}


for arm, ln, bt, ut in (("S53_DELTA", 19, "0.67 / 0.59 (n 8)", "0.66 / 0.96 (n 2: seeds 302, 303)"),
                        ("S59_GDN", 20, "0.57 / 0.56 (n 8)", "0.59 / 0.63 (n 2)"),
                        ("S69_GDN", 21, None, "0.52 / 0.51 (n 10)")):
    for lab, rep, b in (("bound", bt, True), ("unbound", ut, False)):
        if rep is None:
            rs = split(PA[arm], True)
            row(f"{R4}:{ln}", f"{arm} bound runs", "—", "—" if not rs else f"n {len(rs)}")
            continue
        rs = split(PA[arm], b)
        rec = f"{f2(pm(rs, 'L3', 'K'))} / {f2(pm(rs, 'L3', 'V'))} (n {len(rs)}" + (
            f": seeds {', '.join(map(str, sorted(rs)))})" if "seeds" in rep else ")")
        row(f"{R4}:{ln}", f"{arm} {lab} runs, median L3 K / V", rep, rec, None,
            "unrounded " + f"{pm(rs, 'L3', 'K'):.4f} / {pm(rs, 'L3', 'V'):.4f}")
for sd, rep in ((351, "0.88 / 1.00"), (358, "1.00 / 0.98")):
    r = PA["S59_GDN"][sd]
    row(f"{R4}:20", f"S59_GDN seed {sd} L3 K / V (bound: {bound(r)})", rep, f"{f2(r['probe']['L3|K']['acc'])} / {f2(r['probe']['L3|V']['acc'])}")

# readings (explore_f_probe.py:20-27)
bd_runs = [(a, s, r) for a in ("S53_DELTA", "S59_GDN") for s, r in PA[a].items() if bound(r)]
tag = [(a, s) for a, s, r in bd_runs if r["probe"]["L3|K"]["acc"] >= 0.9 and r["probe"]["L3|V"]["acc"] >= 0.9
       and r["probe"]["L1|K"]["acc"] <= 0.6 and r["probe"]["L1|V"]["acc"] <= 0.6]
row(f"{R4}:25-26", "reading 'the delta stack tags its keys by stream' (>= 8 bound delta runs tagged)", "does not apply",
    "applies" if len(tag) >= 8 else "does not apply")
row(f"{R4}:26", "bound delta runs meeting the tag rule, of all bound delta runs", "1 of 16 (S59 358)",
    f"{len(tag)} of {len(bd_runs)} ({'; '.join(f'{a[:3]} {s}' for a, s in tag)})")
r351 = PA["S59_GDN"][351]["probe"]
row(f"{R4}:26", "S59 351 misses the tag rule at K with", "0.88", f2(r351["L3|K"]["acc"]), None,
    f"L3 K {r351['L3|K']['acc']:.4f} < 0.9, V {r351['L3|V']['acc']:.4f}")


def l3(r):
    return 0.5 * (r["probe"]["L3|K"]["acc"] + r["probe"]["L3|V"]["acc"])


mb = med([l3(r) for r in ALLP if bound(r)])
mu = med([l3(r) for r in ALLP if not bound(r)])
row(f"{R4}:27", "reading 'the tag is where binding is' (median L3 bound − unbound >= 0.2)", "does not apply",
    "applies" if mb - mu >= 0.2 else "does not apply")
row(f"{R4}:27", "median L3 (mean of K, V): bound / unbound", "0.62 / 0.52", f"{f2(mb)} / {f2(mu)}", None,
    f"unrounded {mb:.4f} / {mu:.4f} (n {sum(1 for r in ALLP if bound(r))} / {sum(1 for r in ALLP if not bound(r))})")
row(f"{R4}:27", "difference bound − unbound", "0.09", f2(mb - mu), "PARTIAL" if f2(mb - mu) == "0.09" else None,
    f"unrounded {mb - mu:.4f}; the printed '0.62 − 0.52 = 0.09' does not subtract as printed (0.62 − 0.52 = 0.10): "
    "the difference is of the unrounded medians")
l1 = [r["probe"][f"L1|{p_}"]["acc"] for r in ALLP for p_ in ("K", "V")]
row(f"{R4}:29", "L1 probe range over all runs, K and V", "0.49-0.50", f"{f2(min(l1))}-{f2(max(l1))}")
most = [r for a, s, r in bd_runs if (a, s) not in (("S59_GDN", 351), ("S59_GDN", 358))]
vals_any = [r["probe"][f"{f}|{p_}"]["acc"] for r in most for f in ("L2", "L3", "final") for p_ in ("K", "V")]
vals_l3_53 = [r["probe"][f"L3|{p_}"]["acc"] for a, s, r in bd_runs if a == "S53_DELTA" for p_ in ("K", "V")]
vals_l3_all = [r["probe"][f"L3|{p_}"]["acc"] for r in most for p_ in ("K", "V")]
row(f"{R4}:30", "'most bound delta runs ... at any layer' range", "0.53-0.73", f"{f2(min(vals_any))}-{f2(max(vals_any))}", "PARTIAL",
    "AMBIGUOUS: over the 14 bound delta runs other than S59 351/358, L2/L3/final at K and V span "
    f"{f2(min(vals_any))}-{f2(max(vals_any))}; L3 only spans {f2(min(vals_l3_all))}-{f2(max(vals_l3_all))}; "
    f"0.53-0.73 is the L3 range of S53's 8 bound runs alone ({f2(min(vals_l3_53))}-{f2(max(vals_l3_53))})")
ub53 = [r["probe"]["L3|V"]["acc"] for r in PA["S53_DELTA"].values() if not bound(r)]
row(f"{R4}:30", "the two unbound S53 runs at VAL (L3)", "0.92-1.00", f"{f2(min(ub53))}-{f2(max(ub53))}", None,
    "L3 V; at L2 V they are " + "-".join(f2(x) for x in sorted(r["probe"]["L2|V"]["acc"] for r in PA["S53_DELTA"].values() if not bound(r))))
r307 = PA["S53_DELTA"][307]["probe"]
row(f"{R4}:35", "S53 seed 307 L3 K / V", "0.65 / 0.67", f"{f2(r307['L3|K']['acc'])} / {f2(r307['L3|V']['acc'])}")
mk = [pm(PA["S69_GDN"], f, "K") for f in ("L1", "L2", "L3", "final")]
mv = [pm(PA["S69_GDN"], f, "V") for f in ("L1", "L2", "L3", "final")]
row(f"{R4}:39", "S69 probe at any layer (medians L1-final)", "0.50-0.53", f"{f2(min(mk + mv))}-{f2(max(mk + mv))}", "PARTIAL",
    f"AMBIGUOUS: medians at K span {f2(min(mk))}-{f2(max(mk))} (matches); with V the low end is {f2(min(mv))} (L1 V); "
    f"per-run values span {f2(min(r['probe'][f + '|' + q]['acc'] for r in PA['S69_GDN'].values() for f in ('L1', 'L2', 'L3', 'final') for q in 'KV'))}"
    f"-{f2(max(r['probe'][f + '|' + q]['acc'] for r in PA['S69_GDN'].values() for f in ('L1', 'L2', 'L3', 'final') for q in 'KV'))}")
ik = [pm(rs, "L3", "K", True) for rs in PA.values()]
row(f"{R4}:40", "L3 at init, K, range of arm medians", "0.62-0.75", f"{f2(min(ik))}-{f2(max(ik))}")
lowK = all(pm(rs, "L3", "K") < pm(rs, "L3", "K", True) for rs in PA.values())
lowV = [a for a, rs in PA.items() if not pm(rs, "L3", "V") < pm(rs, "L3", "V", True)]
row(f"{R4}:40", "training lowers L3 stream content in every arm (median L3 at the end < at init)", "every arm",
    ("every arm at K" if lowK else "not every arm at K") + ("; every arm at V" if not lowV else f"; at V not in {', '.join(lowV)}"),
    "PARTIAL" if lowK and lowV else ("MATCH" if lowK else "DISAGREE"),
    "the sentence's parenthetical is about K; at V, " + "; ".join(
        f"{a} end {pm(PA[a], 'L3', 'V'):.4f} vs init {pm(PA[a], 'L3', 'V', True):.4f}" for a in lowV)
    + " (medians). Per run, L3 K is lower at the end in " + ", ".join(
        f"{a} {sum(1 for r in rs.values() if r['probe']['L3|K']['acc'] < r['probe_init']['L3|K']['acc'])}/{len(rs)}" for a, rs in PA.items()))
row(f"{R4}:44", "explore_f_delta_checks", "26/26", "not in run records", "NOT RECORD-CHECKABLE",
    "checks file (not a run record): checks_s77_20261010_040240.json 26 ok of 26")
row(f"{R4}:49", "all 30 full reruns equal to their records", "30", str(sum(1 for r in ALLP if r.get("repro"))))
hrs4 = sum(r["secs_wall"] for r in ALLP if r.get("ok")) / 3600
row(f"{R4}:53", "run time (sum of secs_wall, h)", "7.37", f"{hrs4:.2f}")
row(f"{R4}:53", "wall time 03:58-06:05 UTC, about 2.1 h; S76 started after it", "03:58-06:05", "not in run records", "NOT RECORD-CHECKABLE",
    "s77.log's header says 04:02:36 UTC (after the readings commit 2210008 at 04:02:34); the last probe-store commit is 05:58:11 "
    "and s76.log starts 05:58:08, so 03:58 and 06:05 are not what the log and history show")

# ════════════════════════════════════════════════════════════════════════════
# REPORT 5 — S76 eight streams
# ════════════════════════════════════════════════════════════════════════════
ARMS1 = ("O_HEBB", "O_B1_L2", "O_B025_L2", "O_B1_RAW", "O_B025_RAW")
E1 = {a: runs(EI, a) for a in ARMS1}
E2 = {a: runs(EI, a) for a in ("HEBB_SPLIT", "DELTA_SPLIT")}
ALLE = list(EI["runs"].values())
row(f"{R5}:3", "code SHA", "2336ceff7760", ", ".join(sorted({r["code_sha"][:12] for r in ALLE})))
row(f"{R5}:3", "main-line modules at", "9c5939e", ", ".join(sorted({r["main_sha"][:7] for r in ALLE if r.get("main_sha")})), None,
    "main_sha is held by the phase-2 records only" if any(not r.get("main_sha") for r in ALLE) else "")
row(f"{R5}:4", "k; updates; streams (len of ch_map)", "k=16, 43200, S=8",
    f"k={','.join(sorted({str(r['k']) for r in ALLE}))}, {','.join(sorted({str(r['iters']) for r in ALLE}))}, "
    f"S={','.join(sorted({str(len(r['end']['ch_map'])) for r in ALLE if r.get('ok')}))}")
row(f"{R5}:4", "CPU; torch", "Intel Xeon @ 2.10GHz; 2.14.0+cu130",
    "; ".join(sorted({r["cpu"] for r in ALLE})) + "; " + ", ".join(sorted({r["torch"] for r in ALLE})), "MATCH", "the report abbreviates the CPU name")
row(f"{R5}:4", "1 thread per run", "1 thread per run", "not in run records", "NOT RECORD-CHECKABLE", "s76.log headers: 'torch threads 1'")
row(f"{R5}:75", "H's split and trigger copied from claude/explore-H at 7e9197f", "7e9197f", "not in run records", "NOT RECORD-CHECKABLE",
    "provenance of code, not a record field")
row(f"{R5}:4-5", "assigned block 1000-1059 and 1100-1159, 1200-1259 appear in recorded runs", "all three blocks recorded",
    "see note", "PARTIAL", "scan below (SEED SCAN) of every JSON under results/ and explore_out/ at 5b9bc2a for run keys / 'seed' fields")
s1 = sorted({s for a in ARMS1 for s in E1[a]})
s2 = sorted({s for a in E2 for s in E2[a]})
row(f"{R5}:6", "Phase 1 seeds / Phase 2 seeds", "1300-1303 / 1310-1319", f"{s1[0]}-{s1[-1]} / {s2[0]}-{s2[-1]}",
    "MATCH" if (s1 == list(range(1300, 1304)) and s2 == list(range(1310, 1320))) else "DISAGREE")
div = [(k.split("|")[0], k.split("|")[1], r) for k, r in EI["runs"].items() if not r.get("ok")]
row(f"{R5}:6-7", "runs: total, ok, diverged", "40 runs: 39 ok and 1 diverged",
    f"{len(ALLE)} runs: {sum(1 for r in ALLE if r.get('ok'))} ok and {sum(1 for _, _, r in div if 'DIVERGED' in str(r.get('error')))} diverged")
dv = div[0][2] if div else {}
nan_all = all("non-finite held-out loss nan" in str(x) for x in [dv.get("error")] + list(dv.get("failures", [])))
row(f"{R5}:7", "the diverged run; attempts; same non-finite loss on each", "O_B1_RAW|1300, 3 attempts, same loss",
    f"{div[0][0]}|{div[0][1]}, {dv.get('attempts')} attempts, {'same loss' if nan_all else 'different'}" if div else "none")

# Phase 1
H0 = E1["O_HEBB"]


def t_or_inf(r):
    if not r.get("ok"):
        return math.inf
    t = transition(r["curve"])
    return math.inf if t is None else t


def meets(arm):
    """MEETS THE MARK per seed (explore_f_eight.py:19-20)."""
    m = 0
    for s in (1300, 1301, 1302, 1303):
        th, tx = t_or_inf(H0[s]), t_or_inf(E1[arm][s])
        m += int(tx < math.inf and (th == math.inf or tx <= 2 * th))
    return m


P1 = {"O_HEBB": (15, "4/4", "[0.51, 1.00]", ["2400", "3600", "2400", "6000"], None),
      "O_B1_L2": (16, "4/4", "[0.51, 1.00]", ["1200", "1200", "1200", "1200"], "4/4"),
      "O_B025_L2": (17, "4/4", "[0.51, 1.00]", ["2400", "2400", "2400", "2400"], "4/4"),
      "O_B1_RAW": (18, "3/4", "[0.30, 0.95]", ["DIVERGED", "16800", "4800", "2400"], "2/4"),
      "O_B025_RAW": (19, "4/4", "[0.51, 1.00]", ["7200", "1200", "2400", "1200"], "3/4")}
MEETS = {}
for arm, (ln, kb, wb, cells, mk_) in P1.items():
    rs = E1[arm]
    k = cnt(bound, rs)
    row(f"{R5}:{ln}", f"{arm} BOUND (diverged counted unbound, n incl. diverged)", kb, f"{k}/{len(rs)}")
    row_wilson(f"{R5}:{ln}", f"{arm} BOUND Wilson", wb, k, len(rs))
    rec = []
    for s in (1300, 1301, 1302, 1303):
        r = rs[s]
        rec.append("DIVERGED" if not r.get("ok") else str(transition(r["curve"])) if bound(r) else f2(r["acc"]))
    row(f"{R5}:{ln}", f"{arm} per-seed transition 1300 / 1301 / 1302 / 1303", " / ".join(cells), " / ".join(rec))
    if mk_ is not None:
        MEETS[arm] = meets(arm)
        row(f"{R5}:{ln}", f"{arm} seeds meeting the 2x mark (explore_f_eight.py:19-20)", mk_, f"{MEETS[arm]}/4")
w = {a: MEETS[a] >= 3 for a in MEETS}
if w["O_B025_L2"] and w["O_B025_RAW"] and not w["O_B1_L2"] and not w["O_B1_RAW"]:
    r1 = "the slowness is the overwrite"
elif w["O_B1_RAW"] and w["O_B025_RAW"] and not w["O_B1_L2"] and not w["O_B025_L2"]:
    r1 = "the slowness is the key normalisation"
else:
    r1 = "neither"
row(f"{R5}:33", "Phase 1 reading (explore_f_eight.py:21-24)", "neither", r1)
row(f"{R5}:35", "every L2 arm meets the mark on", "4/4", "/".join(sorted({f"{MEETS[a]}/4" for a in ("O_B1_L2", "O_B025_L2")})))
b1 = E1["O_B1_L2"]
row(f"{R5}:35-36", "β = 1 (L2) binds all four seeds at the first evaluation", "1200 on all four",
    f"{', '.join(sorted({str(transition(r['curve'])) for r in b1.values()}))} on {cnt(bound, b1)} of 4; first evaluation {sorted({r['curve'][0][0] for r in b1.values()})}",
    "MATCH" if all(transition(r["curve"]) == r["curve"][0][0] == 1200 for r in b1.values()) else "DISAGREE")
row(f"{R5}:35-36", "β = 1 L2 is the fastest arm and sooner than the Hebbian oracle on every seed", "fastest; sooner than Hebbian",
    "fastest; sooner than Hebbian" if all(t_or_inf(b1[s]) <= min(t_or_inf(E1[a][s]) for a in ARMS1) and t_or_inf(b1[s]) < t_or_inf(H0[s])
                                          for s in b1) else "not on every seed")
raw1 = E1["O_B1_RAW"]
late = [s for s, r in raw1.items() if bound(r) and t_or_inf(r) > 2 * t_or_inf(H0[s])]
row(f"{R5}:37", "raw keys at β = 1: divergences / late binds (bound but outside 2x)", "one divergence and one late bind",
    f"{'one' if sum(1 for r in raw1.values() if not r.get('ok')) == 1 else sum(1 for r in raw1.values() if not r.get('ok'))} divergence and "
    f"{'one' if len(late) == 1 else len(late)} late bind", None, f"late: {late}")


# DELTA* (explore_f_eight.py:25-28)
def star_key(a):
    ts = sorted(t_or_inf(E1[a][s]) for s in (1300, 1301, 1302, 1303))
    m = (ts[1] + ts[2]) / 2
    nb = sum(t < math.inf for t in ts)
    mb_ = sum(t for t in ts if t < math.inf) / nb if nb else math.inf
    return (m, -nb, mb_, ["O_B1_L2", "O_B025_L2", "O_B1_RAW", "O_B025_RAW"].index(a))


star = min(("O_B1_L2", "O_B025_L2", "O_B1_RAW", "O_B025_RAW"), key=star_key)
row(f"{R5}:38", "DELTA* by the fixed rule", "O_B1_L2 (β = 1, L2 keys)", f"{star} ({'β = 1' if 'B1' in star else 'β = 0.25'}, {'L2' if 'L2' in star else 'raw'} keys)")
row(f"{R5}:38", "DELTA*'s median transition", "1200", f"{star_key(star)[0]:g}")
star_file = json.loads(git("show", f"{PIN}:explore_out/F/eight_delta_star.json"))
star_first = git("log", "--reverse", "--format=%h %cI", PIN, "--", "explore_out/F/eight_delta_star.json").split("\n")[0]
dstar_arms = sorted({r["delta"]["arm"] for r in E2["DELTA_SPLIT"].values()})
row(f"{R5}:38", "eight_delta_star.json first committed in; its arm; arm used by DELTA_SPLIT records", "1db6d28; O_B1_L2",
    f"{star_first.split()[0]}; {star_file['arm']}; DELTA_SPLIT records use {', '.join(dstar_arms)}", "MATCH" if (
        star_first.startswith("1db6d28") and star_file["arm"] == star and dstar_arms == [star]) else "DISAGREE")

# Phase 2
HS, DS = E2["HEBB_SPLIT"], E2["DELTA_SPLIT"]
kH, kD = cnt(br, HS), cnt(br, DS)
row(f"{R5}:25", "HEBB_SPLIT BOUND ROUTED", "10/10", f"{kH}/{len(HS)}")
row_wilson(f"{R5}:25", "HEBB_SPLIT Wilson", "[0.72, 1.00]", kH, len(HS))
row(f"{R5}:25", "HEBB_SPLIT band", "RELIABLE", band(kH, len(HS)))


def outcomes(rs):
    oc = {}
    for r in rs.values():
        o = outcome_k(r)
        oc[o] = oc.get(o, 0) + 1
    return ", ".join(f"{k} {v}" for k, v in sorted(oc.items()))


row(f"{R5}:25", "HEBB_SPLIT outcomes", "BOUND ROUTED 10", outcomes(HS))
row(f"{R5}:25", "HEBB_SPLIT MERGED", "0", str(sum(1 for r in HS.values() if outcome_k(r).startswith("MERGED"))))
tH = trans_list(HS, br)
row(f"{R5}:25", "HEBB_SPLIT transitions of BOUND ROUTED: median", "23400", mid(tH))
row(f"{R5}:25", "HEBB_SPLIT transitions (all)", "13200, 20400, 22800 ×3, 24000, 25200, 28800, 36000, 40800", ", ".join(map(str, tH)),
    "MATCH" if tH == [13200, 20400, 22800, 22800, 22800, 24000, 25200, 28800, 36000, 40800] else "DISAGREE",
    "the report writes 22800 ×3")


def fired(rs):
    return [(s, c) for s, r in sorted(rs.items()) for c in r.get("checks", []) if c.get("fired")]


fH, fD = fired(HS), fired(DS)
row(f"{R5}:25", "HEBB_SPLIT splits: total; runs with a split; targets labelled ok (key_ok)", "22; 10; 22/22",
    f"{len(fH)}; {len({s for s, _ in fH})}; {sum(1 for _, c in fH if c.get('key_ok'))}/{len(fH)}")
row(f"{R5}:26", "DELTA_SPLIT BOUND ROUTED", "9/10", f"{kD}/{len(DS)}")
row_wilson(f"{R5}:26", "DELTA_SPLIT Wilson", "[0.60, 0.98]", kD, len(DS))
row(f"{R5}:26", "DELTA_SPLIT band", "RELIABLE", band(kD, len(DS)))
row(f"{R5}:26", "DELTA_SPLIT outcomes", "BOUND ROUTED 9, MERGED (2 share) 1", outcomes(DS))
mD = [s for s, r in DS.items() if outcome_k(r).startswith("MERGED")]
row(f"{R5}:26", "DELTA_SPLIT MERGED (seed)", "1 (1313)", f"{len(mD)} ({', '.join(map(str, mD))})")
tD = trans_list(DS, br)
row(f"{R5}:26", "DELTA_SPLIT transitions of BOUND ROUTED: median", "14400", mid(tD))
row(f"{R5}:26", "DELTA_SPLIT transitions (all)", "9600, 12000, 12000, 13200, 14400, 14400, 15600, 20400, 20400", ", ".join(map(str, tD)))
row(f"{R5}:26", "DELTA_SPLIT splits: total; runs with a split; targets labelled ok", "12; 9; 6/12",
    f"{len(fD)}; {len({s for s, _ in fD})}; {sum(1 for _, c in fD if c.get('key_ok'))}/{len(fD)}")
b_, c_, n_ = pairs(DS, HS, br)
onlyH = [s for s in sorted(set(DS) & set(HS)) if br(HS[s]) and not br(DS[s])]
row(f"{R5}:28", "paired BOUND ROUTED: delta only, Hebbian only (seed)", "delta only 0, Hebbian only 1 (1313)",
    f"delta only {b_}, Hebbian only {c_} ({', '.join(map(str, onlyH))})")
p = mcnemar1(c_, b_)
row(f"{R5}:28", "one-sided exact McNemar p(hebb > delta)", "0.5", f"{p:g}")
both = [s for s in sorted(set(DS) & set(HS)) if br(DS[s]) and br(HS[s])]
sooner = [s for s in both if transition(DS[s]["curve"]) < transition(HS[s]["curve"])]
row(f"{R5}:28-29", "seeds both BOUND ROUTED; delta sooner on", "9; 8", f"{len(both)}; {len(sooner)}", None,
    f"delta not sooner on {[s for s in both if s not in sooner]}")
row(f"{R5}:29", "median transition on the shared seeds, delta vs Hebbian", "14400 vs 24000",
    f"{mid([transition(DS[s]['curve']) for s in both])} vs {mid([transition(HS[s]['curve']) for s in both])}")
vH = cnt(bound, H0) >= 2
vD = cnt(bound, E1[star]) >= 2
if not (vH and vD):
    r2 = "UNTESTED"
elif c_ - b_ <= 1:
    r2 = "the delta recipe keeps eight streams"
elif c_ >= 4 and b_ == 0:
    r2 = "the delta recipe loses eight streams"
else:
    r2 = "inconclusive"
row(f"{R5}:33-34", "Phase 1 reading rules as stated (β = 0.25 arms within 2x and β = 1 not; raw within 2x and L2 not)",
    "as explore_f_eight.py:22-23", "as explore_f_eight.py:22-23", "MATCH", "docstring comparison, not records")
row(f"{R5}:40", "Phase 2: c − b", "1 − 0 = 1", f"{c_} − {b_} = {c_ - b_}")
row(f"{R5}:40", "Phase 2 reading (explore_f_eight.py:37-41)", "the delta recipe keeps eight streams", r2)
c54b, c54c, _ = pairs(C_DELTA, C_HEBB, br)
r54 = "delta better" if (c54b >= 4 and c54c == 0) else "delta not worse" if c54c - c54b <= 2 else "delta worse"
row(f"{R5}:40", "S54(c) reading (explore_f_window_delta_scale.py:29-32) and counts", "delta worse, 3/10 vs 9/10",
    f"{r54}, {cnt(br, C_DELTA)}/{len(C_DELTA)} vs {cnt(br, C_HEBB)}/{len(C_HEBB)}")
row(f"{R5}:45", "S54(c) delta oracle (C_ORACLE_DELTA) transitions", "22800 / 28800", " / ".join(map(str, trans_list(C_OD))))
row(f"{R5}:45", "this screen's delta oracle (DELTA* arm) transitions", "1200", ", ".join(sorted({str(t) for t in trans_list(E1[star])})))
fh_per = [len([1 for s2_, _ in fH if s2_ == s]) for s in sorted(HS)]
row(f"{R5}:52", "Hebbian splits labelled ok; per run", "22/22; 1-3 per run",
    f"{sum(1 for _, c in fH if c.get('key_ok'))}/{len(fH)}; {min(fh_per)}-{max(fh_per)} per run", None,
    f"by the stricter all-position label (all_ok) {sum(1 for _, c in fH if c.get('all_ok'))}/{len(fH)}")
notok = [(s, c) for s, c in fD if not c.get("key_ok")]
row(f"{R5}:53", "delta splits fired with target not labelled ok", "6 of 12", f"{len(notok)} of {len(fD)}")
single = [s for s, _ in notok if DS[s]["splits"] == 1]
row(f"{R5}:53-54", "of those, in single-split runs", "mostly", f"{len(single)} of {len(notok)}", "MATCH" if len(single) * 2 > len(notok) else "DISAGREE")
gaps = sorted(transition(DS[s]["curve"]) - c["step"] for s, c in notok if bound(DS[s]))
inrange = sum(1 for g in gaps if 4800 <= g <= 8400)
row(f"{R5}:54", "updates from a not-ok split to the run's transition", "within 4800-8400", f"{gaps[0]}-{gaps[-1]}", None,
    f"per split (seed: transition − split step) {[(s, transition(DS[s]['curve']) - c['step']) for s, c in notok]}; {inrange} of {len(gaps)} in "
    "4800-8400 (" + str(sum(1 for s_, c in notok if DS[s_]["splits"] == 1 and 4800 <= transition(DS[s_]["curve"]) - c["step"] <= 8400))
    + f" of {len(single)} in the single-split runs); the first evaluation >= 0.95 after the split gives the same gaps. The sentence is "
    "hedged ('mostly'), but no reading of it gives the range 4800-8400")
r13 = DS[1313]
f13 = [c for c in r13["checks"] if c.get("fired")]
from collections import Counter  # noqa: E402
shares = "+".join(str(v) for v in sorted(Counter(r13["end"]["ch_map"]).values(), reverse=True))
row(f"{R5}:55", "1313: splits used, all labelled ok; end map; final acc", "3, all ok; MERGED (2+2+2+1+1); 0.62",
    f"{len(f13)}, {'all ok' if all(c.get('key_ok') for c in f13) else 'not all ok'}; {outcome_k(r13).split(' (')[0]} ({shares}); {f2(r13['acc'])}")
r18 = DS[1318]
row(f"{R5}:56", "1318: transition; splits", "9600; none", f"{transition(r18['curve'])}; {'none' if not [c for c in r18['checks'] if c.get('fired')] else 'some'}")
row(f"{R5}:60", "delta oracle vs Hebbian oracle transitions", "1200 vs 2400-6000",
    f"{', '.join(sorted({str(t) for t in trans_list(E1[star])}))} vs {min(trans_list(H0))}-{max(trans_list(H0))}")
row(f"{R5}:61", "S54's 'slow delta oracle' was a learned-β artefact", "(claim)", "not tested by any record", "NOT RECORD-CHECKABLE",
    "a post-hoc interpretation; the screen did not vary learned β (see 'readings not pre-registered')")
row(f"{R5}:63", "O_B1_RAW|1300 stopped at its first non-finite evaluation; all 3 attempts diverged", "all 3 attempts",
    f"{dv.get('attempts')} attempts, {1 + len(dv.get('failures', []))} DIVERGED errors", "MATCH" if dv.get("attempts") == 3 and nan_all else "DISAGREE",
    "the record keeps no curve for the diverged run, so 'first non-finite evaluation' is not checkable")
row(f"{R5}:67", "explore_f_delta_checks at both phases", "26/26", "not in run records", "NOT RECORD-CHECKABLE",
    "checks files: checks_s76_20261010_055811.json 26 ok of 26, checks_s76_20261010_080127.json 26 ok of 26")
row(f"{R5}:69-73", "screen CHECKs (H's COPY_NONOISE|487 through 7200, first split at 4800, etc.)", "passed", "not in run records",
    "NOT RECORD-CHECKABLE", "child CHECK lines are in s76.log only")
hrs5 = sum(r["secs_wall"] for r in ALLE if r.get("ok")) / 3600
row(f"{R5}:79", "run time (sum of secs_wall over ok records, h)", "31.9", f"{hrs5:.1f}", None,
    f"{hrs5:.3f} h; with the diverged record's last attempt {sum(r['secs_wall'] for r in ALLE) / 3600:.2f} h")
row(f"{R5}:79-80", "Phase 1 05:58-08:01, Phase 2 08:01-15:19 UTC; 9.3 h wall; no restarts", "05:58-08:01, 08:01-15:19, 9.3 h",
    "not in run records", "NOT RECORD-CHECKABLE", "s76.log headers 05:58:08, 08:01:24, done 15:19:14 (9.35 h)")


# ── seed scan for report 5 lines 4-5 ─────────────────────────────────────────
def seed_scan():
    files = [f for f in git("ls-tree", "-r", "--name-only", PIN, "results", "explore_out").split() if f.endswith(".json")]
    blocks = {lo: {} for lo in (1000, 1100, 1200, 1300)}

    def walk(x, f):
        if isinstance(x, dict):
            for k, v in x.items():
                cand = []
                if k == "seed" and isinstance(v, int):
                    cand.append(v)
                if isinstance(k, str) and "|" in k and k.split("|")[1].isdigit():
                    cand.append(int(k.split("|")[1]))
                for c in cand:
                    for lo in blocks:
                        if lo <= c < lo + 60:
                            blocks[lo].setdefault(f, set()).add(c)
                walk(v, f)
        elif isinstance(x, list):
            for v in x:
                walk(v, f)
    for f in files:
        try:
            walk(json.loads(git("show", f"{PIN}:{f}")), f)
        except (ValueError, subprocess.CalledProcessError):
            pass
    return blocks


SEEDS = seed_scan()
for r_ in ROWS:
    if r_["ref"] == f"{R5}:4-5":
        found = {lo: {f: len(s) for f, s in m.items()} for lo, m in SEEDS.items()}
        r_["rec"] = "; ".join(f"{lo}-{lo + 59}: " + (", ".join(f"{f} ({n} seeds)" for f, n in m.items()) or "none")
                              for lo, m in found.items())
        r_["note"] = ("1000-1059 and 1100-1159 appear (as attempt seeds in results/X/router_reliability_results.json); 1200-1259 "
                      "does not appear as a seed in any JSON under results/ or explore_out/ at 5b9bc2a (other branches not scanned); "
                      "1300-1359 only in this screen's store")

# ════════════════════════════════════════════════════════════════════════════
# Readings vs first run records
# ════════════════════════════════════════════════════════════════════════════


def cdate(h):
    t = int(git("log", "-1", "--format=%ct", h).strip())
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def first_reading_commit(path, text):
    out = git("log", "--format=%h %ct", f"-S{text}", BRANCH, "--", path).strip().split("\n")
    out = [o.split() for o in out if o]
    out.sort(key=lambda x: int(x[1]))
    return [o[0][:7] for o in out]


def first_record_commit(store_path, arms):
    for h in git("log", "--reverse", "--format=%h", PIN, "--", store_path).split():
        try:
            st = json.loads(git("show", f"{h}:{store_path}"))
        except (ValueError, subprocess.CalledProcessError):
            continue
        n = sum(1 for k, r in st.get("runs", {}).items() if k.split("|")[0] in arms and not r.get("dry"))
        if n >= 1:
            return h[:7], n
    return None, 0


def is_anc(a, b):
    return subprocess.call(("git", "-C", ROOT, "merge-base", "--is-ancestor", a, b)) == 0


SCREENS = [
    ("S68", "explore_f_premise.py", ['"context-tagged keys scale"', "the partition's advantage is in the number of streams"],
     "explore_out/F/premise_results.json", ["S68_SINGLE_S4", "S68_SINGLE_S8", "S68_ORACLE_S4", "S68_ORACLE_S8"], PR),
    ("S69", "explore_f_hebb_decay.py", ['"the forget gate needs the delta rule"', '"the forget gate alone suffices"'],
     "explore_out/F/hebb_decay_results.json", ["S69_FIXED", "S69_ROUTED", "S69_GDN", "S69_ORACLE"], HD),
    ("S70 phase 1 (oracles)", "explore_f_rh_lite.py", ["An oracle arm BINDS if >= 2 of its 3 seeds bind"],
     "explore_out/F/rh_lite_results.json", ["S70_ORACLE_HEBB", "S70_ORACLE_DELTA"], RH),
    ("S70 phase 2 (BASELINE)", "explore_f_rh_lite.py", ["the BASELINE for Session G is S70_HEBB's DISCOVERED count"],
     "explore_out/F/rh_lite_results.json", ["S70_HEBB", "S70_DELTA"], RH),
    ("S77", "explore_f_probe.py", ['"the delta stack tags its keys by stream"', '"the tag is where binding is"'],
     "explore_out/F/probe_results.json", ["S53_DELTA", "S59_GDN", "S69_GDN"], PB),
    ("S76 phase 1", "explore_f_eight.py", ['"the slowness is the overwrite"', '"the slowness is the key normalisation"'],
     "explore_out/F/eight_results.json", list(ARMS1), EI),
    ("S76 phase 2", "explore_f_eight.py", ['"the delta recipe keeps eight streams"', '"the delta recipe loses eight streams"'],
     "explore_out/F/eight_results.json", ["HEBB_SPLIT", "DELTA_SPLIT"], EI),
]
PREC = []
for scr, path, texts, sp, arms, st in SCREENS:
    rc = [first_reading_commit(path, t) for t in texts]
    rcomm = sorted({c[0] for c in rc if c}, key=lambda h: int(git("log", "-1", "--format=%ct", h)))
    allc = sorted({h for c in rc for h in c})
    reading = rcomm[0] if rcomm else None
    fr, nrec = first_record_commit(sp, arms)
    file_hist = git("log", "--format=%h", PIN, "--", path).split()
    unchanged = git("diff", reading, PIN, "--", path).strip() == "" if reading else False
    rgits = sorted({k_r["git"].split("+")[0] for a in arms for k_r in runs(st, a).values() if k_r.get("git")})
    desc = all(is_anc(reading, g) for g in rgits) if reading else False
    precede = (reading is not None and fr is not None and int(git("log", "-1", "--format=%ct", reading)) < int(git("log", "-1", "--format=%ct", fr))
               and is_anc(reading, fr))
    PREC.append(dict(screen=scr, file=path, texts=texts, reading=reading, rdate=cdate(reading) if reading else "--",
                     allc=allc, first=fr, fdate=cdate(fr) if fr else "--", nrec=nrec, hist=len(file_hist), unchanged=unchanged,
                     rgits=rgits, desc=desc, precede=precede))

# ════════════════════════════════════════════════════════════════════════════
# Write the markdown
# ════════════════════════════════════════════════════════════════════════════
ORDER = {"DISAGREE": 0, "PARTIAL": 1, "NOT RECORD-CHECKABLE": 2, "MATCH": 3}


def esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def refkey(r):
    f, ln = r["ref"].rsplit(":", 1)
    return (f, int(ln.split("-")[0]))


rows = sorted(ROWS, key=lambda r: (ORDER[r["status"]],) + refkey(r))
tally = Counter(r["status"] for r in ROWS)
L = []
P = L.append
P("# Session F, reports 3-5: recomputation from the records\n")
P(f"Generated by `docs/review_2026-10/sessions/recompute_F.py` (stdlib only; `rlib` for loading, outcomes and statistics). "
  f"Branch `{BRANCH}`, pinned head **{PIN}** ({cdate(PIN)}). Records: `explore_out/F/{{premise,hebb_decay,rh_lite,probe,eight}}_results.json` "
  f"at {PIN}; reference counts from `decay` (S59), `delta_controls` (S53) and `window_delta_scale` (S54) stores at the same commit. "
  f"Reported values are typed from the report text; no reported or docstring number is an input.\n")
P(f"**Rows: {len(ROWS)}** — DISAGREE {tally['DISAGREE']}, PARTIAL {tally['PARTIAL']}, NOT RECORD-CHECKABLE "
  f"{tally['NOT RECORD-CHECKABLE']}, MATCH {tally['MATCH']}.\n")
P("## Outcome definitions used\n")
P("- **BOUND** — `rlib.bound`: the record is ok and its held-out curve has a transition (first evaluation >= 0.95 that holds to the end), "
  "recomputed from `curve`, not read from `transition`. explore_f_premise.py:18, explore_f_hebb_decay.py:20, explore_f_rh_lite.py:19, "
  "explore_f_delta_controls.py:30-31 (S53), explore_f_decay.py:36 (S59 BOUND count). The recomputed transition equals the recorded one in every record used.")
P("- **DISCOVERED (S70)** — bound and final VAL cos (`val_cos`) < 0.5; explore_f_rh_lite.py:19 (`rlib.discovered`).")
P("- **ROUTED\\*@end (S70)** — `end.margin >= 0.9 and end.eta_key_by_stream > 0.9`; explore_f_rh_lite.py:19 (implemented here).")
P("- **Failure classes (S70)** — test_router_layout.fail_class rule on the end statistics: margin >= 0.25 STREAM-PARTIAL, else η² by key at KEY "
  ">= 0.5 KEY, else max(η² by half, by index) >= 0.5 POSITION, else OTHER; explore_f_rh_lite.py:20-21 (`rlib.fail_class`). "
  "AMBIGUITY: the docstring does not say whether bound runs are classified. Counted here on unbound runs only, as the report's "
  "'KEY 19 (1 BOUND unrouted)' reads; on all runs S70_HEBB seed 335 and five bound S70_DELTA seeds would class STREAM-PARTIAL and S70_DELTA 338 KEY.")
P("- **Validity / readings** — S68 explore_f_premise.py:19-24; S69 explore_f_hebb_decay.py:21-25; S70 phase-1 rule explore_f_rh_lite.py:11-15, "
  "reading :23-24; S77 explore_f_probe.py:20-27 (L3 for the second reading = mean of KEY and VAL, :24); S76 MEETS THE MARK "
  "explore_f_eight.py:19-20, Phase-1 readings :21-24, DELTA* :25-28, VALIDITY :37, Phase-2 readings :38-41; S54(c) explore_f_window_delta_scale.py:29-32.")
P("- **BOUND ROUTED (S76 Phase 2, S54 references)** — test_stream_recipe.outcome (explore_f_eight.py:35; explore_f_window_delta_scale.py:21-24; "
  "9c5939e test_stream_recipe.py:751-762), computed as `rlib.outcome_k`: transition from the curve, one-to-one from the argmax of "
  "`end.stream_gate` (ties deferred to `end.ch_map`), every `stream_acc >= 0.9`; MERGED (n share) on unbound STREAM-PARTIAL "
  "(η² from `etak_*`) with `shared_max >= 2`. Cross-check against the main line's own form (recorded `transition` and recorded "
  f"`end.one_to_one`): {len(OUTCOME_XCHECK)} records differ" + (f": {OUTCOME_XCHECK}" if OUTCOME_XCHECK else "") +
  f". stream_gate ties deferred to ch_map: {sorted(set(TIES)) if TIES else 'none'} (none changes an outcome: the cross-check above).")
P("- **S77 Bound** — BOUND of the rerun (explore_f_probe.py:20), from the rerun's curve; equals the record's `bound` field for all 30. "
  "Probe values are the records' held-out `probe[feat|K/V].acc`; medians are `statistics.median` (mean of the two middle values for even n).")
P("- **S76 Phase 1** — a diverged record (ok False) is unbound and counts in n (explore_f_eight.py:17-18).")
zdiff = sum(1 for r in ROWS if "z = 1.96" in r["note"])
P("- **Statistics** — Wilson 95% `rlib.wilson` (z = 1.95996; the reports' explore_f_common.wilson uses 1.96, flagged PARTIAL only if that "
  f"changes a printed digit: {zdiff} rows); bands `rlib.band`; one-sided exact McNemar `rlib.mcnemar1`; one-sided Fisher `rlib.fisher1`.")
P("- Report 3's Fisher p compares the recorded BOUND ROUTED counts of S54's gated Hebbian arms (other seeds) with S68's BOUND counts, "
  "as the S68 docstring's REFERENCES (explore_f_premise.py:15-17) and explore_f_reports2.py:83-88 do.\n")
P("## Table 1 — every number in the three reports\n")
P("| # | report:line | item | reported | recomputed | status | note |")
P("|---|---|---|---|---|---|---|")
for i, r in enumerate(rows, 1):
    P(f"| {i} | {esc(r['ref'])} | {esc(r['item'])} | {esc(r['rep'])} | {esc(r['rec'])} | **{r['status']}** | {esc(r['note'])} |")
P("\n## Table 2 — readings committed before the first run record?\n")
P("Readings commit = the earliest commit on `origin/claude/explore-F` that adds the distinctive reading text to the screen file "
  "(`git log --format='%h %cI' -S'<text>' origin/claude/explore-F -- <file>`). First run record = the earliest commit at which the "
  "screen's (phase's) arms hold >= 1 non-dry record in the results store. 'Run-time HEAD' = the `git` field of the phase's records "
  "(the HEAD under which they were written); 'descends' = every one of them has the readings commit as an ancestor.\n")
P("| screen | file | reading text searched | readings commit | first run record (store commit) | readings precede first run | "
  "file commits to pin / docstring unchanged since | run-time HEADs descend from readings commit |")
P("|---|---|---|---|---|---|---|---|")
for p_ in PREC:
    P(f"| {p_['screen']} | {p_['file']} | {esc('; '.join(p_['texts']))} | {p_['reading']} ({p_['rdate']})"
      + (f"; all -S hits {', '.join(p_['allc'])}" if len(p_['allc']) > 1 else "") +
      f" | {p_['first']} ({p_['fdate']}; {p_['nrec']} record(s)) | **{'yes' if p_['precede'] else 'NO'}** | "
      f"{p_['hist']} / {'yes' if p_['unchanged'] else 'NO'} | {'yes' if p_['desc'] else 'NO'} ({', '.join(p_['rgits'])}) |")
P(f"\nS76 Phase 2 also depends on DELTA*: `eight_delta_star.json` first committed in {star_first.split()[0]} "
  f"({cdate(star_first.split()[0])}), before the first Phase-2 record commit ({[p_['first'] for p_ in PREC if p_['screen'] == 'S76 phase 2'][0]}). "
  f"S70 phase 2's budget file rh_lite_budget.json: first commit "
  f"{git('log', '--reverse', '--format=%h %cI', PIN, '--', 'explore_out/F/rh_lite_budget.json').split()[0]}.\n")
P("Log headers at the pin (not records, for timing only): followup.log starts S70 phase 1 at 14:19:53 UTC (readings commit a1112da 14:19:50, "
  "3 s earlier; the log's git is a1112da) and S68/S69/S70 phase 2 at 14:24:25; s77.log starts at 04:02:36 (readings commit 2210008 04:02:34; "
  "log git 2210008) — report 4 line 53's '03:58' start would precede the readings commit, but nothing at the pin shows a run before 04:02:36; "
  "s76.log starts Phase 1 at 05:58:08 and Phase 2 at 08:01:24 (after 1db6d28, 08:01:21; log git 1db6d28+dirty).\n")
P("## Readings the reports state that the docstrings did not pre-register\n")
P("Each screen's pre-registered readings (Table 2) are reported as the docstring fixes them, and the recomputed reading agrees in every case "
  "(Table 1). Statements in the reports' 'Readings (fixed ... before any run)' sections, or stated as findings elsewhere, that no docstring fixed:\n")
for t in (
    f"{R3}:29-31 (S68) — 'S53's stream-tagged keys ... do not carry to four or eight streams in the grouped layout. There the partition is what "
    "binds' — placed under the pre-registered readings but not one of S68's three readings (explore_f_premise.py:21-24).",
    f"{R3}:33-36 (S69) — 'The Hebbian channel does not learn S59's reset at CTX tokens' and 'the learned decay hurt the Hebbian channel' — "
    "interpretations of the α and accuracy DIAGNOSTICS (explore_f_hebb_decay.py:26-27), not readings.",
    f"{R3}:39-40 (S70) — 'The gate routes by key ...' and 'As in S53, the delta memory binds without the partition.' The docstring fixes "
    "'No other reading' (explore_f_rh_lite.py:23-24).",
    f"{R3}:49-51 (S70) — the LOCAL3 window explanation of the failure (post hoc, in 'Anything unexpected').",
    f"{R4}:28-30 (S77) — 'L1 is at chance ... as it must be' and 'The two S53 runs that do carry it at VAL are the two that did not bind' — "
    "descriptive, permitted by 'otherwise descriptive' (explore_f_probe.py:26-27); not readings.",
    f"{R4}:34-40 (S77) — 'does not generalise into a linearly readable stream tag', 'does not rest on a stream tag a gate could read' — "
    "labelled post hoc in the report.",
    f"{R5}:36-37 (S76) — 'With the perfect gate the delta memory is *not* slow at eight streams when β is fixed' and 'Raw keys are the "
    "unstable variant' — under the pre-registered readings, but not among Phase 1's readings (explore_f_eight.py:21-24).",
    f"{R5}:44-48 and {R5}:60-61 (S76) — 'learned β is the likely cause of S54(c)'s slow delta memory' (labelled post hoc) and, at :60-61, "
    "stated flatly: 'S54's \"slow delta oracle\" was a learned-β artefact'. No arm of S76 varies learned β; this is not pre-registered and "
    "not tested by the screen (S54(c) and S76 also differ in the split operation and seeds, as :42-48 says).",
    f"{R5}:59-63 (S76) — 'β ‖x‖² > 2 makes I − β x xᵀ expanding' as the cause of the divergence (post hoc).",
):
    P(f"- {t}")
P("\n## Notes on ambiguous or non-record items\n")
P("- Report 4 line 30's '0.53-0.73' and line 39's '0.50-0.53' are ranges whose population the text does not fix; Table 1 gives the "
  "candidate populations and marks them PARTIAL.")
P("- Report 3 line 52's 'near 0.25' / 'near 0.75' has no stated tolerance; 'within 0.05' is the referee's reading.")
P("- Wall-clock times, worker counts, restarts, CHECK results and the probe's synthetic check are not in the run records; they are marked "
  "NOT RECORD-CHECKABLE with what the checks files or logs at 5b9bc2a say.")
P("- 'Every rerun's curve equals its record' (report 4) rests on the records' own `repro` field; the original S53/S59/S69 records were not re-compared here.")
open(OUT, "w").write("\n".join(L) + "\n")
print(f"wrote {OUT}: {len(ROWS)} rows; " + ", ".join(f"{k} {tally[k]}" for k in ORDER))
for r in rows:
    if r["status"] == "DISAGREE":
        print(f"  DISAGREE {r['ref']}: {r['item']}: reported {r['rep']!r}, recomputed {r['rec']!r}")
