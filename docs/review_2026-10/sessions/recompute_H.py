#!/usr/bin/env python3
"""recompute_H.py — the referee's recomputation of Session H's reports 4 (S78) and 5 (S79) from the JSON records.

    python3 docs/review_2026-10/sessions/recompute_H.py      # writes docs/review_2026-10/sessions/recompute_H.md

Pinned: origin/claude/explore-H at df57087. Records: explore_out/H/two_results.json (S78) and
explore_out/H/constants_results.json (S79), read with `git show df57087:<path>` (rlib.load). Reported values are
typed in from report_4.md / report_5.md at df57087 with their line numbers; no reported number is an input to any
recomputation. Stdlib only; read-only except for the .md written next to this file.

Definitions (each cited in the .md header):
  S78 (explore_h_two.py docstring at df57087)
    DISCOVERED (l.34-35)         rlib.discovered: bound (first eval >= 0.95 held to the end) and final val_cos < 0.5.
    K4 BOUND ROUTED (l.35-36)    rlib.bound_routed: bound, one-to-one argmax map of end.stream_gate (ties -> end.ch_map),
                                 every end.stream_acc >= 0.9.
    v1 classes (l.37)            k=2 arms: rlib.fail_class (test_router_layout.fail_class); K4: rlib.outcome_k
                                 (test_stream_recipe.outcome).
    v2 classes (l.37-38)         k=2 arms: fail_class_v2 (main line fail_class_v2.py l.40-48: eta^2 by key >= 0.5 is KEY
                                 before the margin rule); K4: outcome_v2 (fail_class_v2.py l.57-69).
    Readings R1-R4 (l.40-46)     exact one-sided McNemar (rlib.mcnemar1) on 20 paired seeds.
    VALIDITY (l.39)              ORACLE binds both seeds.
  S79 (explore_h_constants.py docstring at df57087)
    BOUND ROUTED (l.22)          rlib.bound_routed.
    labelled ok (l.22-23, 30-31) c* holds >= 2 streams and c0 none on the map before the firing; recomputed from each
                                 check as map_before.count(key_cs) >= 2 and map_before.count(key_c0) == 0 (key_ok unused).
    v1 / v2 (l.23-24)            rlib.outcome_k / outcome_v2 as above.
    Readings (l.26-32)           robust if (BASE only - V only) <= 1; sensitive if BASE only >= 4 and V only = 0;
                                 inconclusive otherwise; CAP6 "harmless" if CAP6 >= BASE - 1 and every firing beyond a
                                 run's third is labelled ok.
    VALIDITY (l.25)              ORACLE binds both seeds.
"""
import collections
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from rlib import (load, runs, transition, bound, discovered, bound_routed, fail_class, outcome_k, shared_max,  # noqa: E402
                  mcnemar1, wilson, band, pairs, med, git, TIES, ETA_SPLIT, MARGIN_PARTIAL)

PIN = "df57087"
BRANCH = "origin/claude/explore-H"
R4F, R5F = "explore_out/H/report_4.md", "explore_out/H/report_5.md"
TWO, CON = "explore_out/H/two_results.json", "explore_out/H/constants_results.json"
OUT = os.path.join(HERE, "recompute_H.md")

ROWS = []   # (file, line, item, reported, recomputed, status, note)
AGREE = []  # stored-field vs recomputed cross-checks (information only)


def add(f, line, item, reported, recomputed, status=None, note=""):
    if status is None:
        status = "MATCH" if str(reported) == str(recomputed) else "DISAGREE"
    ROWS.append((f, line, item, str(reported), str(recomputed), status, note))


# ── classifiers (v2 = main line fail_class_v2.py, re-implemented) ─────────────────────────────────────────────────
def fail_class_v2(e):
    """fail_class_v2.py l.40-48 on a two-stream end dict."""
    if e["eta_key_by_key"] >= ETA_SPLIT:
        return "KEY"
    if e["margin"] >= MARGIN_PARTIAL:
        return "STREAM-PARTIAL"
    if max(e["eta_key_by_half"], e["eta_key_by_index"]) >= ETA_SPLIT:
        return "POSITION"
    return "OTHER"


def fail_class_k_v2(r):
    """fail_class_v2.py l.51-59 (multichannel field mapping, block for half)."""
    e = r["end"]
    return fail_class_v2(dict(margin=e["margin"], eta_key_by_key=e["etak_key_by_key"],
                              eta_key_by_half=e["etak_key_by_block"], eta_key_by_index=e["etak_key_by_index"]))


def outcome_v2(r):
    """fail_class_v2.py l.62-69 with rlib.bound_routed as `routed` and the recomputed bound."""
    if bound(r):
        return "BOUND ROUTED" if bound_routed(r) else "BOUND NOT routed"
    fc = fail_class_k_v2(r)
    if fc == "STREAM-PARTIAL":
        return f"MERGED ({shared_max(r)} share)" if shared_max(r) >= 2 else "STREAM-PARTIAL one-to-one"
    return f"non-stream {fc}"


def fclass_v1_collapsed(o):
    """explore_h_common.fclass (report 5's 'failures v1' column collapses MERGED (n share) to MERGED)."""
    if o == "BOUND ROUTED":
        return None
    if o.startswith("BOUND NOT"):
        return "BOUND-NOT-ROUTED"
    if o.startswith("MERGED"):
        return "MERGED"
    if o.startswith("STREAM-PARTIAL"):
        return "STREAM-PARTIAL"
    if o.startswith("non-stream "):
        return o.split(" ", 1)[1]
    return o


def parse_cls(s):
    """'KEY 3, STREAM-PARTIAL 2' -> Counter; '—' -> {}."""
    s = s.strip()
    if s in ("—", "-", ""):
        return {}
    out = {}
    for part in s.split(", "):
        name, n = part.rsplit(" ", 1)
        out[name] = int(n)
    return out


def cls_text(c):
    return ", ".join(f"{k} {v}" for k, v in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))) or "—"


def cmp_cls(f, line, item, reported, counter, note=""):
    rec = dict(counter)
    add(f, line, item, reported, cls_text(rec), "MATCH" if parse_cls(reported) == rec else "DISAGREE", note)


def wtxt(k, n):
    lo, hi = wilson(k, n)
    return f"{lo:.2f}-{hi:.2f}"


def p3(p):
    return f"{p:.3g}"


def reported_lines(path):
    return git("show", f"{PIN}:{path}").split("\n")


def verify_quote(path, line, text):
    """Guard against typos in the typed-in values: the quoted text must occur on that report line."""
    L = reported_lines(path)
    if text not in L[line - 1]:
        raise SystemExit(f"typed-in text {text!r} not found on {path}:{line}")


# ══ S78 ══════════════════════════════════════════════════════════════════════════════════════════════════════════
def s78():
    st = load(PIN, TWO)
    SEEDS = list(range(1300, 1320))
    A = {a: runs(st, a) for a in ("REF", "SPLIT2", "SPLIT2_NOISE", "K4", "ORACLE")}
    succ = {a: (bound_routed if a == "K4" else discovered) for a in A}
    f = R4F

    # seeds / record identity (header l.3)
    sd = {a: sorted(A[a]) for a in A}
    add(f, 3, "seeds per arm (REF, SPLIT2, SPLIT2_NOISE, K4)", "1300-1319",
        "1300-1319" if all(sd[a] == SEEDS for a in ("REF", "SPLIT2", "SPLIT2_NOISE", "K4")) else str(sd))
    # transitions: the stored field equals the curve-derived one?
    tr_bad = [(a, s) for a in A for s, r in A[a].items() if r.get("transition") != transition(r["curve"])]
    dsc_bad = [(a, s) for a in ("REF", "SPLIT2", "SPLIT2_NOISE") for s, r in A[a].items() if bool(r.get("discovered")) != discovered(r)]
    k4_bad = [s for s, r in A["K4"].items() if r.get("outcome") != outcome_k(r)]
    AGREE.append(f"S78: stored transition vs curve-derived differ on {tr_bad or 'none'}; stored 'discovered' vs rlib.discovered "
                 f"differ on {dsc_bad or 'none'}; K4 stored 'outcome' vs rlib.outcome_k differ on {k4_bad or 'none'}.")

    # per-arm table l.7-10
    typed = {
        "REF": (7, "15/20", "MAJORITY", "0.53-0.89", "0", "KEY 3, STREAM-PARTIAL 2", "KEY 4, STREAM-PARTIAL 1", "2400",
                "2400, 2400, 2400, 2400, 2400, 2400, 2400, 2400, 3600, 3600, 3600, 3600, 3600, 3600, 4800", None),
        "SPLIT2": (8, "16/20", "MAJORITY", "0.58-0.92", "0", "KEY 3, STREAM-PARTIAL 1", "KEY 4", "3000",
                   "2400, 2400, 2400, 2400, 2400, 2400, 2400, 2400, 3600, 3600, 3600, 3600, 3600, 3600, 4800, 6000", "12 (5)"),
        "SPLIT2_NOISE": (9, "17/20", "MAJORITY", "0.64-0.95", "0", "KEY 1, STREAM-PARTIAL 2", "KEY 3", "3600",
                         "2400, 2400, 2400, 2400, 2400, 2400, 2400, 2400, 3600, 3600, 3600, 3600, 3600, 3600, 4800, 6000, 6000",
                         "10 (5)"),
        "K4": (10, "15/20", "MAJORITY", "0.53-0.89", "0", "STREAM-PARTIAL one-to-one 3, non-stream KEY 2",
               "STREAM-PARTIAL one-to-one 3, non-stream KEY 2", "2400",
               "2400, 2400, 2400, 2400, 2400, 2400, 2400, 2400, 2400, 2400, 2400, 3600, 3600, 3600, 4800", None)}
    stats = {}
    for a, (ln, cnt, bnd, wil, bns, v1t, v2t, tmed, tlist, spl) in typed.items():
        verify_quote(f, ln, f"| {a} |")
        rs = A[a]
        n, k = len(rs), sum(succ[a](r) for r in rs.values())
        stats[a] = (k, n)
        add(f, ln, f"{a} count", cnt, f"{k}/{n}")
        add(f, ln, f"{a} band", bnd, band(k, n))
        add(f, ln, f"{a} Wilson 95%", wil, wtxt(k, n))
        add(f, ln, f"{a} bound, not successful", bns, sum(1 for r in rs.values() if bound(r) and not succ[a](r)))
        unb = {s: r for s, r in rs.items() if not bound(r)}
        if a == "K4":
            c1 = collections.Counter(outcome_k(r) for r in unb.values())
            c2 = collections.Counter(outcome_v2(r) for r in unb.values())
        else:
            c1 = collections.Counter(fail_class(r) for r in unb.values())
            c2 = collections.Counter(fail_class_v2(r["end"]) for r in unb.values())
        cmp_cls(f, ln, f"{a} unbound classes v1", v1t, c1)
        cmp_cls(f, ln, f"{a} unbound classes v2", v2t, c2)
        tr = sorted(transition(r["curve"]) for r in rs.values() if succ[a](r))
        add(f, ln, f"{a} transition median (successful runs)", tmed, f"{med(tr):.0f}")
        add(f, ln, f"{a} transition list (successful runs)", tlist, ", ".join(str(t) for t in tr),
            note="header says 'median (all)'; the list holds the successful runs only (as recomputed)")
        if spl:
            fired = [sum(1 for c in r.get("checks") or [] if c.get("fired")) for r in rs.values()]
            add(f, ln, f"{a} splits fired (runs)", spl, f"{sum(fired)} ({sum(1 for x in fired if x)})")

    # oracle l.12
    o = A["ORACLE"]
    ob = sum(bound(r) for r in o.values())
    add(f, 12, "ORACLE bound", "2/2", f"{ob}/{len(o)}")
    add(f, 12, "ORACLE transitions", "{1300: 1200, 1301: 1200}", str({s: transition(r["curve"]) for s, r in sorted(o.items())}))
    add(f, 12, "validity (docstring l.39)", "VALID", "VALID" if ob == 2 and len(o) == 2 else "UNTESTED")

    # pairwise l.16-20
    S = {a: {s: succ[a](r) for s, r in A[a].items()} for a in ("REF", "SPLIT2", "SPLIT2_NOISE", "K4")}
    cmpd = {}
    typed_cmp = [(16, "SPLIT2", "REF", "16/20", "15/20", "1", "0", "0.5", "1"),
                 (17, "SPLIT2_NOISE", "REF", "17/20", "15/20", "2", "0", "0.25", "1"),
                 (18, "K4", "REF", "15/20", "15/20", "3", "3", "0.656", "0.656"),
                 (19, "SPLIT2_NOISE", "SPLIT2", "17/20", "16/20", "1", "0", "0.5", "1"),
                 (20, "K4", "SPLIT2", "15/20", "16/20", "2", "3", "0.812", "0.5")]
    for ln, x, y, xc, yc, xo, yo, pxy, pyx in typed_cmp:
        verify_quote(f, ln, f"| {x} vs {y} |")
        b, c, n = pairs(S[x], S[y], lambda v: v)
        cmpd[(x, y)] = (b, c, mcnemar1(b, c))
        add(f, ln, f"{x} vs {y}: x count", xc, f"{sum(S[x][s] for s in SEEDS)}/{n}")
        add(f, ln, f"{x} vs {y}: y count", yc, f"{sum(S[y][s] for s in SEEDS)}/{n}")
        add(f, ln, f"{x} vs {y}: x only", xo, b)
        add(f, ln, f"{x} vs {y}: y only", yo, c)
        add(f, ln, f"{x} vs {y}: p(x>y) one-sided exact McNemar", pxy, p3(mcnemar1(b, c)))
        add(f, ln, f"{x} vs {y}: p(y>x)", pyx, p3(mcnemar1(c, b)))

    # readings l.24-27 (docstring l.40-46)
    b, c, p = cmpd[("SPLIT2", "REF")]
    r1 = ("the split carries to two streams" if p < 0.05 else "does not apply") + f" (SPLIT2 vs REF {b} vs {c}, one-sided p = {p3(p)})"
    add(f, 24, "R1 (docstring l.41-42)", "does not apply (SPLIT2 vs REF 1 vs 0, one-sided p = 0.5)", r1)
    b, c, p = cmpd[("SPLIT2_NOISE", "SPLIT2")]
    r2 = ("noise is needed at k = 2" if (b >= 4 and c == 0) else "does not apply") + f" (SPLIT2_NOISE vs SPLIT2 {b} vs {c})"
    add(f, 25, "R2 (docstring l.43)", "does not apply (SPLIT2_NOISE vs SPLIT2 1 vs 0)", r2)
    b, c, p = cmpd[("K4", "SPLIT2")]
    r3 = ("spare channels are the better device" if (b >= 4 and c == 0) else "does not apply") + f" (K4 vs SPLIT2 {b} vs {c})"
    add(f, 26, "R3 (docstring l.44)", "does not apply (K4 vs SPLIT2 2 vs 3)", r3)
    beats = [a for a in ("SPLIT2", "SPLIT2_NOISE", "K4") if cmpd[(a, "REF")][2] < 0.05]
    r4 = ("neither device" if not beats else f"does not apply ({', '.join(beats)} beat REF)") + " (" + "; ".join(
        f"{a} vs REF {cmpd[(a, 'REF')][0]} vs {cmpd[(a, 'REF')][1]}, p = {p3(cmpd[(a, 'REF')][2])}"
        for a in ("SPLIT2", "SPLIT2_NOISE", "K4")) + ")"
    add(f, 27, "R4 (docstring l.45-46)",
        "neither device (SPLIT2 vs REF 1 vs 0, p = 0.5; SPLIT2_NOISE vs REF 2 vs 0, p = 0.25; K4 vs REF 3 vs 3, p = 0.656)", r4)

    # split table l.29-54
    D = []
    for a in ("SPLIT2", "SPLIT2_NOISE"):
        for s, r in sorted(A[a].items()):
            cks = r.get("checks") or []
            for i, ck in enumerate(cks):
                if ck.get("fired"):
                    nxt = cks[i + 1] if i + 1 < len(cks) else None
                    D.append(dict(arm=a, seed=s, step=ck["step"], before=ck["diag"], after=ck.get("diag_after"),
                                  next=nxt["diag"] if nxt else None, r=r))
    add(f, 29, "split firings (both split arms)", "22", len(D))

    def ff(d, x, y):
        return "--" if not d else f"{d[x]:.2f}/{d[y]:.2f}"

    def endtag(r):
        if discovered(r):
            return "DISCOVERED"
        return fail_class(r) if not bound(r) else "BOUND not DISCOVERED"

    tbl = reported_lines(f)[32:54]
    for i, line in enumerate(tbl):
        ln = 33 + i
        cells = [x.strip() for x in line.strip().strip("|").split("|")]
        d = next((d for d in D if d["arm"] == cells[0] and str(d["seed"]) == cells[1] and str(d["step"]) == cells[2]), None)
        if d is None:
            add(f, ln, f"split row {cells[0]} {cells[1]} {cells[2]}", " | ".join(cells[3:]), "no such firing in the records")
            continue
        rec = [ff(d["before"], "eta_key_by_key", "eta_key_by_stream"), ff(d["after"], "eta_key_by_key", "eta_key_by_stream"),
               ff(d["next"], "eta_key_by_key", "eta_key_by_stream"), ff(d["before"], "eta_val_by_key", "eta_val_by_stream"),
               ff(d["after"], "eta_val_by_key", "eta_val_by_stream"), ff(d["next"], "eta_val_by_key", "eta_val_by_stream"),
               endtag(d["r"])]
        add(f, ln, f"split row {cells[0]} {cells[1]} @{cells[2]}: eta^2 KEY/VAL before/after/next, end tag",
            " | ".join(cells[3:]), " | ".join(rec),
            note="eta^2 values read from checks[].diag / diag_after / next check's diag (stored), end tag recomputed (v1)")
    # the transcription of firings is complete (one table row per firing, no extra)
    add(f, 29, "split table rows = firings", "22 rows", f"{len(tbl)} rows for {len(D)} firings",
        "MATCH" if len(tbl) == len(D) == 22 else "DISAGREE")

    # v1/v2 l.56 and l.83-84
    unb = {}
    for a in ("REF", "SPLIT2", "SPLIT2_NOISE", "K4"):
        for s, r in A[a].items():
            if not bound(r):
                if a == "K4":
                    v1, v2, kk = outcome_k(r), outcome_v2(r), r["end"]["etak_key_by_key"]
                else:
                    v1, v2, kk = fail_class(r), fail_class_v2(r["end"]), r["end"]["eta_key_by_key"]
                unb[f"{a}|{s}"] = (v1, v2, kk, r["end"]["margin"])
    dis = sorted(k for k, v in unb.items() if v[0] != v[1])
    pred = sorted(k for k, v in unb.items() if v[2] >= ETA_SPLIT and v[3] >= MARGIN_PARTIAL)
    add(f, 56, "v1/v2 differ on S78 unbound runs", "4", len(dis))
    add(f, 56, "differences = exactly eta^2 by key >= 0.5 and margin >= 0.25", "True", dis == pred)
    typed56 = {"REF|1317": "STREAM-PARTIAL -> KEY (key 0.5000, margin 0.562)",
               "SPLIT2_NOISE|1311": "STREAM-PARTIAL -> KEY (key 0.7365, margin 0.267)",
               "SPLIT2_NOISE|1313": "STREAM-PARTIAL -> KEY (key 0.5068, margin 0.449)",
               "SPLIT2|1317": "STREAM-PARTIAL -> KEY (key 0.5021, margin 0.552)"}
    for k, t in typed56.items():
        v = unb.get(k)
        rec = f"{v[0]} -> {v[1]} (key {v[2]:.4f}, margin {v[3]:.3f})" if v else "not unbound"
        add(f, 56, f"v1/v2 difference {k}", t, rec)
    add(f, 83, "S78 unbound runs", "17", len(unb))
    add(f, 83, "v2 differs from v1 on", "4: REF 1317, SPLIT2 1317, SPLIT2_NOISE 1311 and 1313",
        f"{len(dis)}: {', '.join(dis)}",
        "MATCH" if set(dis) == {"REF|1317", "SPLIT2|1317", "SPLIT2_NOISE|1311", "SPLIT2_NOISE|1313"} else "DISAGREE")

    # runtime / checks l.58
    rs_ok = [r for r in st["runs"].values() if r.get("ok")]
    add(f, 58, "runs", "82", len(rs_ok))
    hw = sum(r.get("secs_wall", 0) for r in rs_ok) / 3600
    hs = sum(r.get("secs", 0) for r in rs_ok) / 3600
    add(f, 58, "CPU-hours", "3.2", f"{hw:.1f}", "MATCH" if f"{hw:.1f}" == "3.2" else "DISAGREE",
        f"sum of secs_wall (as explore_h_report.runtime sums it); the 'secs' field gives {hs:.1f}")
    add(f, 58, "code SHA", "5af849d1455f", ", ".join(sorted({r["code_sha"][:12] for r in rs_ok})))
    ck = json.loads(git("show", f"{PIN}:explore_out/H/checks.json")).get("H/two")
    add(f, 58, "CHECKs time, git, minutes", "2026-10-10 04:11:34 UTC, git 3997ae2, 1.6 min",
        f"{ck['time']}, git {ck['git']}, {ck['minutes']:.1f} min" if ck else "no checks.json entry")
    add(f, 58, "CHECKs: all passed", "all passed", "checks.json entry present (written only after a pass)", "PARTIAL",
        "the individual CHECK results are only in s78.log (lines 10-22), not in a record field; checks.json holds sha/time/git/"
        "minutes and is written only after every CHECK of the screen passed (explore_h_run.py l.96-103); its sha "
        + ("equals" if ck and ck["sha"][:12] in {r["code_sha"][:12] for r in rs_ok} else "DIFFERS FROM") + " the runs' code SHA")

    # prose l.60-62
    add(f, 60, "SPLIT2 vs REF / SPLIT2_NOISE vs REF / K4 vs REF", "1 vs 0, 2 vs 0, 3 vs 3",
        ", ".join(f"{cmpd[(a, 'REF')][0]} vs {cmpd[(a, 'REF')][1]}" for a in ("SPLIT2", "SPLIT2_NOISE", "K4")))
    add(f, 61, "best one-sided p vs REF", "0.25", p3(min(cmpd[(a, "REF")][2] for a in ("SPLIT2", "SPLIT2_NOISE", "K4"))))
    bands = {a: band(*stats[a]) for a in stats}
    ks = sorted(stats[a][0] for a in stats)
    add(f, 61, "all four arms MAJORITY (15-17/20)", "MAJORITY, 15-17/20",
        f"{'MAJORITY' if set(bands.values()) == {'MAJORITY'} else bands}, {ks[0]}-{ks[-1]}/20")
    wl = {a: wilson(*stats[a]) for a in stats}
    ov = all(max(wl[a][0], wl[b][0]) <= min(wl[a][1], wl[b][1]) for a in wl for b in wl)
    add(f, 62, "Wilson intervals overlap (all pairs)", "overlapping", "overlapping" if ov else "not all overlapping")

    # prose l.65-81 (split mechanics)
    sp = [d for d in D if d["arm"] == "SPLIT2"]
    zero = all(d["after"] and all(round(d["after"][k], 2) == 0 for k in ("eta_key_by_key", "eta_key_by_stream", "eta_val_by_key",
                                                                         "eta_val_by_stream")) for d in sp)
    add(f, 66, "SPLIT2: every eta^2 0.00 right after the copy", f"all {len([1 for _ in sp])} SPLIT2 firings 0.00",
        f"{'all' if zero else 'not all'} {len(sp)} SPLIT2 firings 0.00", "MATCH" if zero else "DISAGREE",
        "reported text: 'every eta^2 is 0.00 right after the copy'")
    tri = [d for d in sp if d["seed"] in (1311, 1313, 1315)]
    with_next = [d for d in tri if d["next"]]
    reform = [d for d in with_next if ff(d["next"], "eta_key_by_key", "eta_key_by_stream") == "1.00/0.00"]
    no_next = [d for d in tri if not d["next"]]
    end_ok = all(ff(d["r"]["end"], "eta_key_by_key", "eta_key_by_stream") == "1.00/0.00" for d in no_next)
    add(f, 68, "KEY split re-formed (1.00/0.00) at the next check after every firing on 1311, 1313, 1315", "8 of 8 firings",
        f"{len(reform)} of {len(with_next)} firings that have a next check; {len(no_next)} of {len(tri)} firings "
        f"({', '.join(f'{d['seed']}@{d['step']}' for d in no_next)}) have no next check"
        + (f"; at the run's end (2400 later) both read 1.00/0.00" if end_ok else ""),
        "PARTIAL" if (len(reform) == len(with_next) and end_ok) else "DISAGREE",
        "the split table itself prints '--' as 'next' for 1311@21600 and 1313@21600; '8 of 8' counts them via the end statistics")
    s1317 = [d for d in sp if d["seed"] == 1317]
    near = all(d["next"] and abs(d["next"]["eta_key_by_key"] - 0.5) < 0.015 and abs(d["next"]["eta_key_by_stream"] - 0.25) < 0.015
               for d in s1317)
    add(f, 68, "1317: half-key pattern (0.50/0.25) re-formed after all firings", "all 3 firings",
        f"{'all' if near else 'not all'} {len(s1317)} firings (next: "
        + ", ".join(ff(d["next"], "eta_key_by_key", "eta_key_by_stream") for d in s1317) + ")",
        "MATCH" if near and len(s1317) == 3 else "DISAGREE", "pattern taken as within 0.015 of 0.50/0.25")
    d09 = [d for d in sp if d["seed"] == 1309]
    add(f, 71, "SPLIT2 1309: before / next KEY eta^2, run discovered", "0.20/0.60 -> 0.00/1.00, discovered",
        f"{ff(d09[0]['before'], 'eta_key_by_key', 'eta_key_by_stream')} -> {ff(d09[0]['next'], 'eta_key_by_key', 'eta_key_by_stream')}, "
        f"{'discovered' if discovered(A['SPLIT2'][1309]) else 'not discovered'}" if len(d09) == 1 else f"{len(d09)} firings")
    fired_seeds = {a: sorted({d["seed"] for d in D if d["arm"] == a}) for a in ("SPLIT2", "SPLIT2_NOISE")}
    add(f, 73, "the same 5 runs fired in both split arms", "5, same seeds",
        f"{len(fired_seeds['SPLIT2'])}, {'same' if fired_seeds['SPLIT2'] == fired_seeds['SPLIT2_NOISE'] else 'different'} seeds "
        f"{fired_seeds['SPLIT2_NOISE']}", "MATCH" if fired_seeds["SPLIT2"] == fired_seeds["SPLIT2_NOISE"] and
        len(fired_seeds["SPLIT2"]) == 5 else "DISAGREE")
    vs_ref = [s for s in SEEDS if S["SPLIT2_NOISE"][s] and not S["REF"][s]]
    vs_sp = [s for s in SEEDS if S["SPLIT2_NOISE"][s] and not S["SPLIT2"][s]]
    add(f, 74, "SPLIT2_NOISE rescued 1309 and 1317 (both DISCOVERED)", "1309, 1317, both DISCOVERED",
        f"SPLIT2_NOISE discovered 1309: {S['SPLIT2_NOISE'][1309]}, 1317: {S['SPLIT2_NOISE'][1317]}; SPLIT2_NOISE-only vs REF: "
        f"{vs_ref}; vs SPLIT2: {vs_sp}",
        "MATCH" if S["SPLIT2_NOISE"][1309] and S["SPLIT2_NOISE"][1317] and vs_ref == [1309, 1317] else "DISAGREE",
        "'rescued' read as discovered after firing where REF did not discover; SPLIT2 (exact copy) also discovered 1309 "
        "(report l.71), so relative to SPLIT2 only 1317 is new")
    for s in (1311, 1313):
        r = A["SPLIT2_NOISE"][s]
        kk = r["end"]["eta_key_by_key"]
        add(f, 75, f"SPLIT2_NOISE {s}: end key eta^2 in 0.5-0.75, v1 STREAM-PARTIAL, v2 KEY", "0.5-0.75, STREAM-PARTIAL, KEY",
            f"{kk:.4f}, {fail_class(r)}, {fail_class_v2(r['end'])}",
            "MATCH" if 0.5 <= kk <= 0.75 and fail_class(r) == "STREAM-PARTIAL" and fail_class_v2(r["end"]) == "KEY" else "DISAGREE")
    d15 = [d for d in D if d["arm"] == "SPLIT2_NOISE" and d["seed"] == 1315]
    held = all(d["next"] and ff(d["next"], "eta_key_by_key", "eta_key_by_stream") == "1.00/0.00" for d in d15)
    add(f, 75, "SPLIT2_NOISE 1315: key split held through 3 firings", "3 firings, held",
        f"{len(d15)} firings, {'held' if held else 'not held'} (end {endtag(A['SPLIT2_NOISE'][1315])})",
        "MATCH" if len(d15) == 3 and held else "DISAGREE")
    add(f, 76, "SPLIT2_NOISE net vs REF / vs SPLIT2", "+2, +1",
        f"+{cmpd[('SPLIT2_NOISE', 'REF')][0] - cmpd[('SPLIT2_NOISE', 'REF')][1]}, "
        f"+{cmpd[('SPLIT2_NOISE', 'SPLIT2')][0] - cmpd[('SPLIT2_NOISE', 'SPLIT2')][1]}")
    add(f, 77, "K4 count / REF count / discordant", "15/20, same as REF, 3 vs 3",
        f"{stats['K4'][0]}/20, {'same as' if stats['K4'] == stats['REF'] else 'differs from'} REF, "
        f"{cmpd[('K4', 'REF')][0]} vs {cmpd[('K4', 'REF')][1]}")
    k4u = [r for r in A["K4"].values() if not bound(r)]
    c1 = collections.Counter(outcome_k(r) for r in k4u)
    c2 = collections.Counter(outcome_v2(r) for r in k4u)
    add(f, 78, "K4 failures under v1 and v2", "STREAM-PARTIAL one-to-one 3, non-stream KEY 2 (both)",
        f"v1 {cls_text(c1)}; v2 {cls_text(c2)}",
        "MATCH" if dict(c1) == dict(c2) == {"STREAM-PARTIAL one-to-one": 3, "non-stream KEY": 2} else "DISAGREE")
    add(f, 80, "REF 15/20, Wilson 0.53-0.89", "15/20 (Wilson 0.53-0.89)", f"{stats['REF'][0]}/20 (Wilson {wtxt(*stats['REF'])})")
    ru = [r for r in A["REF"].values() if not bound(r)]
    add(f, 81, "REF failures v1 / v2", "v1 KEY 3 + STREAM-PARTIAL 2; v2 KEY 4 + 1",
        f"v1 {cls_text(collections.Counter(fail_class(r) for r in ru))}; v2 {cls_text(collections.Counter(fail_class_v2(r['end']) for r in ru))}",
        "MATCH" if (dict(collections.Counter(fail_class(r) for r in ru)) == {"KEY": 3, "STREAM-PARTIAL": 2} and
                    dict(collections.Counter(fail_class_v2(r["end"]) for r in ru)) == {"KEY": 4, "STREAM-PARTIAL": 1}) else "DISAGREE")

    # external numbers
    ext = s52_external()
    add(f, 78, "S52 W2_K4 36/40 on seeds 160-199", "36/40", "not in explore_out/H records", "NOT RECORD-CHECKABLE",
        "S52's store (explore_out/two_stream_keysplits_results.json) is not an H record; for information, "
        f"recomputed from it at origin/claude/outside-ideas {ext['oi']}: {ext['k4']}")
    add(f, 80, "main line WIN3_SLOW discovered 69/80 (86%)", "69/80 (86%)", "not in explore_out/H records",
        "NOT RECORD-CHECKABLE", "main-line test_window_gate records (results/X, results/L), outside the H store")
    add(f, 85, "pre-run CHECK: unbound two-stream runs of this branch", "29", ext["n"],
        "MATCH" if str(ext["n"]) == "29" else "DISAGREE",
        "not an H/two record: recomputed from the CHECK's population (explore_h_two.py l.172-186) on "
        f"window_recipe (S43) and H/stability at {PIN} and S52's store at {ext['oi']} (outside-ideas as of the CHECK)")
    add(f, 85, "pre-run CHECK: v2 differed on (exactly the predicted)", "6, exactly predicted",
        f"{len(ext['dis'])}, {'exactly predicted' if ext['dis'] == ext['pred'] else 'not the predicted set'}",
        "MATCH" if len(ext["dis"]) == 6 and ext["dis"] == ext["pred"] else "DISAGREE", "same population as the row above")
    add(f, 86, "two of them are S52 W2_K4 MERGED runs 169, 196 -> non-stream KEY", "169, 196", ", ".join(ext["k4dis"]),
        "MATCH" if ext["k4dis"] == ["169 MERGED (2 share) -> non-stream KEY", "196 MERGED (2 share) -> non-stream KEY"] else "DISAGREE")
    add(f, 95, "REF reproduced S43's record bit for bit through 2400", "bit for bit", "CHECK output only (s78.log l.18)",
        "NOT RECORD-CHECKABLE", "the CHECK's re-run is not stored in a record (no field)")
    add(f, 96, "K4 reproduced S52's W2_K4 record bit for bit through 2400", "bit for bit", "CHECK output only (s78.log l.19)",
        "NOT RECORD-CHECKABLE", "the CHECK's re-run is not stored in a record (no field)")
    return A


def s52_external():
    """The S78 pre-run CHECK's population (explore_h_two.py l.172-186), recomputed from the stores."""
    oi = git("rev-list", "-1", "--before=2026-10-10T04:09:11Z", "origin/claude/outside-ideas").strip()[:7]
    s43 = load(PIN, "explore_out/window_recipe_results.json")["runs"]
    stab = load(PIN, "explore_out/H/stability_results.json")["runs"]
    s52 = load(oi, "explore_out/two_stream_keysplits_results.json")["runs"]
    recs = {}
    for k, r in s43.items():
        if r.get("ok") and r.get("transition") is None:
            recs["S43:" + k] = ("k2", r)
    for k, r in stab.items():
        if r.get("ok") and r.get("transition") is None and not k.startswith("ORACLE") and "eta_key_by_key" in r["end"]:
            recs["H/stability:" + k] = ("k2", r)
    for k, r in s52.items():
        if r.get("ok") and k.startswith("W2_K4|") and r.get("transition") is None:
            recs["S52:" + k] = ("outcome", r)
        elif r.get("ok") and k.startswith("W2_KEYHINGE|") and r.get("transition") is None:
            recs["S52:" + k] = ("k2", r)
    dis, pred, k4dis = [], [], []
    for k, (kind, r) in sorted(recs.items()):
        e = r["end"]
        if kind == "k2":
            a, b, kk = fail_class(r), fail_class_v2(e), e["eta_key_by_key"]
        else:
            # the CHECK used test_stream_recipe.outcome on the stored transition (None here), i.e. the unbound branch
            a, b, kk = outcome_k(r), outcome_v2(r), e["etak_key_by_key"]
        if a != b:
            dis.append(k)
            if k.startswith("S52:W2_K4|"):
                k4dis.append(f"{k.split('|')[1]} {a} -> {b}")
        if kk >= ETA_SPLIT and e["margin"] >= MARGIN_PARTIAL:
            pred.append(k)
    k4 = {s: r for s, r in ((int(k.split("|")[1]), r) for k, r in s52.items() if k.startswith("W2_K4|") and r.get("ok"))}
    k4c = f"{sum(bound_routed(r) for r in k4.values())}/{len(k4)} bound routed (seeds {min(k4)}-{max(k4)})" if k4 else "none"
    return dict(n=len(recs), dis=dis, pred=pred, k4dis=k4dis, oi=oi, k4=k4c)


# ══ S79 ══════════════════════════════════════════════════════════════════════════════════════════════════════════
def label_ok(c):
    m = c["map_before"]
    return m.count(c["key_cs"]) >= 2 and m.count(c["key_c0"]) == 0


def s79():
    st = load(PIN, CON)
    SEEDS = list(range(1320, 1330))
    VAR = ("INT1200", "CAP1", "CAP6", "PROBE16", "THR05")
    A = {a: runs(st, a) for a in ("BASE",) + VAR + ("ORACLE",)}
    f = R5F
    add(f, 3, "seeds per learned arm", "1320-1329",
        "1320-1329" if all(sorted(A[a]) == SEEDS for a in ("BASE",) + VAR) else str({a: sorted(A[a]) for a in A}))
    rec_out = [(a, s) for a in ("BASE",) + VAR for s, r in A[a].items() if (outcome_k(r) != r.get("outcome"))]
    typed = {
        "BASE": (7, "9/10", "RELIABLE", "0.60-0.98", "MERGED 1", "MERGED (2 share) 1", "24000",
                 "15600, 16800, 18000, 18000, 24000, 24000, 27600, 27600, 28800", "21 (3)", "21/21"),
        "INT1200": (8, "7/10", "MAJORITY", "0.40-0.89", "MERGED 3", "MERGED (2 share) 2, MERGED (3 share) 1", "14400",
                    "9600, 10800, 13200, 14400, 18000, 24000, 32400", "28 (3)", "28/28"),
        "CAP1": (9, "5/10", "MAJORITY", "0.24-0.76", "MERGED 5", "MERGED (2 share) 5", "21600",
                 "15600, 18000, 21600, 22800, 43200", "9 (1)", "9/9"),
        "CAP6": (10, "9/10", "RELIABLE", "0.60-0.98", "MERGED 1", "MERGED (2 share) 1", "18000",
                 "15600, 16800, 18000, 18000, 18000, 24000, 24000, 27600, 30000", "29 (6)", "28/29"),
        "PROBE16": (11, "10/10", "RELIABLE", "0.72-1.00", "—", "—", "18600",
                    "13200, 15600, 15600, 16800, 18000, 19200, 19200, 25200, 25200, 27600", "18 (3)", "18/18"),
        "THR05": (12, "10/10", "RELIABLE", "0.72-1.00", "—", "—", "19200",
                  "14400, 15600, 15600, 16800, 18000, 20400, 25200, 25200, 27600, 28800", "24 (3)", "24/24")}
    br, fires = {}, {}
    for a, (ln, cnt, bnd, wil, v1t, v2t, tmed, tlist, spl, lok) in typed.items():
        verify_quote(f, ln, f"| {a} |")
        rs = A[a]
        n, k = len(rs), sum(bound_routed(r) for r in rs.values())
        br[a] = {s: bound_routed(r) for s, r in rs.items()}
        add(f, ln, f"{a} BOUND ROUTED", cnt, f"{k}/{n}")
        add(f, ln, f"{a} band", bnd, band(k, n))
        add(f, ln, f"{a} Wilson 95%", wil, wtxt(k, n))
        c1 = collections.Counter(fclass_v1_collapsed(outcome_k(r)) for r in rs.values() if not bound_routed(r))
        c2 = collections.Counter(outcome_v2(r) for r in rs.values() if not bound_routed(r))
        cmp_cls(f, ln, f"{a} failures v1", v1t, c1, "v1 column collapses MERGED (n share) to MERGED (explore_h_common.fclass)")
        cmp_cls(f, ln, f"{a} failures v2", v2t, c2)
        tr = sorted(transition(r["curve"]) for r in rs.values() if bound_routed(r))
        add(f, ln, f"{a} transition median (BOUND ROUTED runs)", tmed, f"{med(tr):.0f}")
        add(f, ln, f"{a} transition list", tlist, ", ".join(str(t) for t in tr),
            note="header says 'median (all)'; the list holds the BOUND ROUTED runs only (as recomputed)")
        fires[a] = {s: [c for c in r.get("checks") or [] if c.get("fired")] for s, r in sorted(rs.items())}
        nf = [len(x) for x in fires[a].values()]
        add(f, ln, f"{a} splits (max/run)", spl, f"{sum(nf)} ({max(nf)})")
        nok = sum(label_ok(c) for x in fires[a].values() for c in x)
        stored = sum(bool(c.get("key_ok")) for x in fires[a].values() for c in x)
        add(f, ln, f"{a} labelled ok (recomputed from map_before)", lok, f"{nok}/{sum(nf)}",
            note=f"stored key_ok gives {stored}/{sum(nf)}")
    o = A["ORACLE"]
    ob = sum(bound(r) for r in o.values())
    add(f, 14, "ORACLE bound", "2/2", f"{ob}/{len(o)}")
    add(f, 14, "ORACLE transitions", "{1320: 3600, 1321: 3600}", str({s: transition(r["curve"]) for s, r in sorted(o.items())}))
    add(f, 14, "validity (docstring l.25)", "VALID", "VALID" if ob == 2 and len(o) == 2 else "UNTESTED")
    add(f, 36, "VALID: oracle bound 2/2, both at 3600", "2/2, 3600, 3600",
        f"{ob}/{len(o)}, " + ", ".join(str(transition(r["curve"])) for _, r in sorted(o.items())))

    typed_cmp = {"INT1200": (16, "7/10", "9/10", "0", "2", "1", "0.25"), "CAP1": (17, "5/10", "9/10", "0", "4", "1", "0.0625"),
                 "CAP6": (18, "9/10", "9/10", "0", "0", "1", "1"), "PROBE16": (19, "10/10", "9/10", "1", "0", "0.5", "1"),
                 "THR05": (20, "10/10", "9/10", "1", "0", "0.5", "1")}
    cm = {}
    for v, (ln, vc, bc, vo, bo, pv, pb) in typed_cmp.items():
        verify_quote(f, ln, f"- {v} vs BASE")
        b, c, n = pairs(br[v], br["BASE"], lambda x: x)
        cm[v] = (b, c)
        add(f, ln, f"{v} vs BASE: {v} count", vc, f"{sum(br[v].values())}/{n}")
        add(f, ln, f"{v} vs BASE: BASE count", bc, f"{sum(br['BASE'].values())}/{n}")
        add(f, ln, f"{v} vs BASE: {v} only", vo, b)
        add(f, ln, f"{v} vs BASE: BASE only", bo, c)
        add(f, ln, f"{v} vs BASE: p({v}>BASE)", pv, p3(mcnemar1(b, c)))
        add(f, ln, f"{v} vs BASE: p(BASE>{v})", pb, p3(mcnemar1(c, b)))

    # readings l.24-29 (docstring l.26-32)
    typed_rd = {"INT1200": (24, "inconclusive (BASE only 2, INT1200 only 0)"), "CAP1": (25, "sensitive (BASE only 4, CAP1 only 0)"),
                "CAP6": (26, "robust (BASE only 0, CAP6 only 0)"), "PROBE16": (27, "robust (BASE only 0, PROBE16 only 1)"),
                "THR05": (28, "robust (BASE only 0, THR05 only 1)")}
    for v, (ln, t) in typed_rd.items():
        b, c = cm[v]
        lab = "robust" if c - b <= 1 else "sensitive" if (c >= 4 and b == 0) else "inconclusive"
        add(f, ln, f"reading {v} (docstring l.27-29)", t, f"{lab} (BASE only {c}, {v} only {b})")
    extra = [(s, c["step"], label_ok(c)) for s, x in fires["CAP6"].items() for c in x[3:]]
    nbr = {a: sum(br[a].values()) for a in br}
    if not extra:
        rd = "no extra splits fired (the reading does not apply)"
    else:
        okk = nbr["CAP6"] >= nbr["BASE"] - 1 and all(k for _, _, k in extra)
        rd = (("the extra splits are harmless" if okk else "does not apply")
              + f" (CAP6 {nbr['CAP6']} vs BASE {nbr['BASE']}; extra firings {len(extra)}, labelled ok {sum(1 for *_, k in extra if k)})")
    add(f, 29, "reading CAP6_extra (docstring l.30-32)", "does not apply (CAP6 9 vs BASE 9; extra firings 7, labelled ok 6)", rd)
    add(f, 32, "CAP6's firings beyond the third (seed, update, labelled ok)",
        "[(1320, 31200, True), (1320, 33600, True), (1320, 40800, True), (1321, 26400, True), (1321, 28800, False), "
        "(1323, 26400, True), (1327, 14400, True)]", str(extra))
    bad = [(s, st_) for s, st_, k in extra if not k]
    add(f, 40, "extra firings not labelled ok", "1 of 7 (seed 1321 at 28800)",
        f"{len(bad)} of {len(extra)} (" + ", ".join(f"seed {s} at {t}" for s, t in bad) + ")")
    add(f, 41, "CAP6 1321 bound anyway at", "30000", transition(A["CAP6"][1321]["curve"]) if bound_routed(A["CAP6"][1321]) else "not bound routed")

    # INT1200 first split l.31
    first = {a: {s: (x[0]["step"] if x else None) for s, x in fires[a].items()} for a in ("INT1200", "BASE")}
    txt = ", ".join(f"{s}: {first['INT1200'][s]}/{first['BASE'][s]}" for s in SEEDS)
    add(f, 31, "INT1200 / BASE first split per seed",
        "1320: 2400/19200, 1321: 2400/4800, 1322: 2400/7200, 1323: 2400/12000, 1324: 2400/16800, 1325: 9600/None, "
        "1326: 3600/12000, 1327: 3600/4800, 1328: 2400/16800, 1329: 3600/12000", txt)
    ne = sum(1 for s in SEEDS if first["INT1200"][s] is not None and (first["BASE"][s] is None or first["INT1200"][s] < first["BASE"][s]))
    add(f, 31, "INT1200 first split earlier than BASE's", "10 of 10", f"{ne} of 10",
        note="1325 counts as earlier because BASE never fired there (docstring l.33 does not say how a never-firing BASE counts)")
    add(f, 51, "INT1200 first split earlier on", "10/10", f"{ne}/10", note="as l.31")
    diffs = [s for a in ("BASE",) + VAR for s, r in A[a].items() if outcome_k(r) != outcome_v2(r)]
    add(f, 32, "v1/v2 outcome differences", "(none)", ", ".join(map(str, diffs)) or "(none)")
    add(f, 60, "v1 and v2 agree on every S79 run", "agree on every run", "agree on every run" if not diffs else f"differ on {diffs}")
    fails = [(a, s, outcome_k(r)) for a in ("BASE",) + VAR for s, r in A[a].items() if not bound_routed(r)]
    shares = sorted({o for *_, o in fails})
    add(f, 60, "all S79 failures MERGED (2-3 share)", "MERGED (2-3 share)", "; ".join(shares),
        "MATCH" if set(shares) <= {"MERGED (2 share)", "MERGED (3 share)"} else "DISAGREE")
    hk = [(a, s) for a in ("BASE",) + VAR for s, r in A[a].items() if not bound(r)
          and r["end"]["etak_key_by_key"] >= ETA_SPLIT and r["end"]["margin"] >= MARGIN_PARTIAL]
    add(f, 61, "failures with eta^2 by key >= 0.5 and margin >= 0.25", "none", ", ".join(map(str, hk)) or "none")

    # runtime l.34
    rs_ok = [r for r in st["runs"].values() if r.get("ok")]
    add(f, 34, "runs", "62", len(rs_ok))
    hw = sum(r.get("secs_wall", 0) for r in rs_ok) / 3600
    hs = sum(r.get("secs", 0) for r in rs_ok) / 3600
    add(f, 34, "CPU-hours", "33.8", f"{hw:.1f}", "MATCH" if f"{hw:.1f}" == "33.8" else "DISAGREE",
        f"sum of secs_wall (as explore_h_report.runtime sums it); the 'secs' field gives {hs:.1f}")
    add(f, 34, "code SHA", "39143d0c9c2e", ", ".join(sorted({r["code_sha"][:12] for r in rs_ok})))
    ck = json.loads(git("show", f"{PIN}:explore_out/H/checks.json")).get("H/constants")
    add(f, 34, "CHECKs time, git, minutes", "2026-10-10 05:16:35 UTC, git e6db83b+dirty, 14.4 min",
        f"{ck['time']}, git {ck['git']}, {ck['minutes']:.1f} min" if ck else "no checks.json entry")
    add(f, 34, "CHECKs: all passed", "all passed", "checks.json entry present (written only after a pass)", "PARTIAL",
        "the individual CHECK results are only in s79.log, not in a record field; checks.json is written only after every "
        "CHECK passed (explore_h_run.py l.96-103); its sha "
        + ("equals" if ck and ck["sha"][:12] in {r["code_sha"][:12] for r in rs_ok} else "DIFFERS FROM")
        + " the runs' code SHA; the CHECK ran on a dirty tree (e6db83b+dirty)")

    # prose l.37-58
    add(f, 37, "CAP1: BASE only / CAP1 only / one-sided p", "4, 0, p = 0.0625",
        f"{cm['CAP1'][1]}, {cm['CAP1'][0]}, p = {p3(mcnemar1(cm['CAP1'][1], cm['CAP1'][0]))}")
    add(f, 38, "INT1200: BASE only / INT1200 only", "2, 0", f"{cm['INT1200'][1]}, {cm['INT1200'][0]}")
    add(f, 39, "CAP6, PROBE16, THR05: BASE only vs V only", "0 vs 0, 0 vs 1, 0 vs 1",
        ", ".join(f"{cm[v][1]} vs {cm[v][0]}" for v in ("CAP6", "PROBE16", "THR05")))
    add(f, 40, "CAP6 matches BASE", "9 vs 9", f"{nbr['CAP6']} vs {nbr['BASE']}")

    cv = {a: {s: {c[0]: c[1] for c in r["curve"]} for s, r in A[a].items()} for a in A}
    c1f = [(s, x[0]["step"], x[0]["acc"], cv["CAP1"][s].get(x[0]["step"])) for s, x in fires["CAP1"].items()
           if not br["CAP1"][s] and x]
    early = [t for t in c1f if t[1] <= 12000]
    rng = lambda xs: f"{min(xs):.2f}-{max(xs):.2f}"  # noqa: E731
    add(f, 46, "CAP1 failures that fired at 4800-12000", "4 of 5",
        f"{len(early)} of {sum(1 for s in SEEDS if not br['CAP1'][s])} (" + ", ".join(f"{s}@{t}" for s, t, *_ in early) + ")",
        "MATCH" if (len(early), sum(1 for s in SEEDS if not br["CAP1"][s])) == (4, 5) and min(t[1] for t in early) >= 4800
        else "DISAGREE")
    add(f, 46, "accuracy at those CAP1 firings", "0.09-0.50",
        f"probe (checks[].acc) {rng([t[2] for t in early])}; held-out curve {rng([t[3] for t in early])}",
        "MATCH" if rng([t[2] for t in early]) == "0.09-0.50" else "DISAGREE",
        "the report calls it 'held-out accuracy'; the printed range is the trigger probe's accuracy (checks[].acc), "
        "while the evaluation curve at the same updates gives " + rng([t[3] for t in early]))
    ROWS[-1] = ROWS[-1][:5] + ("PARTIAL" if ROWS[-1][5] == "MATCH" else ROWS[-1][5],) + ROWS[-1][6:]
    add(f, 48, "CAP1 splits labelled ok", "9/9",
        f"{sum(label_ok(c) for x in fires['CAP1'].values() for c in x)}/{sum(len(x) for x in fires['CAP1'].values())}")
    fifth = [s for s, t, *_ in c1f if t > 12000]
    add(f, 49, "fifth CAP1 failure; BASE failed there too with all three splits", "1320, BASE failed, 3 splits",
        f"{', '.join(map(str, fifth))}, BASE {'failed' if fifth and not br['BASE'][fifth[0]] else 'bound routed'}, "
        f"{len(fires['BASE'][fifth[0]]) if fifth else '-'} splits")
    fi = [(s, x[0]["step"], x[0]["acc"], cv["INT1200"][s].get(x[0]["step"])) for s, x in fires["INT1200"].items() if x]
    at = [t for t in fi if t[1] <= 3600]
    add(f, 51, "INT1200 first split mostly at 2400-3600", "mostly 2400-3600",
        f"{len(at)} of {len(fi)} at 2400-3600 (other: " + ", ".join(f"{s}@{t}" for s, t, *_ in fi if t > 3600) + ")",
        "MATCH" if 2 * len(at) > len(fi) else "DISAGREE")
    add(f, 51, "accuracy at INT1200's first splits (2400-3600)", "0.03-0.09",
        f"probe {rng([t[2] for t in at])}; held-out curve {rng([t[3] for t in at])}",
        "MATCH" if rng([t[2] for t in at]) == "0.03-0.09" else "DISAGREE",
        "outside 0.03-0.09: " + ", ".join(f"{s}@{t} probe {p:.3f} / held-out {h:.3f}" for s, t, p, h in at
                                          if round(p, 2) > 0.09) + " (probe); the held-out curve is 0.06-0.11 throughout")
    i_f = [s for s in SEEDS if not br["INT1200"][s]]
    third = [(s, fires["INT1200"][s][2]["step"], fires["INT1200"][s][2]["acc"], cv["INT1200"][s].get(fires["INT1200"][s][2]["step"]))
             for s in i_f if len(fires["INT1200"][s]) >= 3]
    add(f, 52, "INT1200 failures", "1320, 1321, 1327", ", ".join(map(str, i_f)))
    add(f, 52, "INT1200 failures used all three splits by", "7200-10800",
        f"{min(t[1] for t in third)}-{max(t[1] for t in third)} ({len(third)} of {len(i_f)} used 3)",
        "MATCH" if len(third) == len(i_f) and (min(t[1] for t in third), max(t[1] for t in third)) == (7200, 10800)
        else "DISAGREE")
    pm, hm = max(t[2] for t in third), max(t[3] for t in third)
    add(f, 52, "accuracy at the failures' third split", "<= 0.47", f"probe <= {pm:.2f}; held-out curve <= {hm:.2f}",
        "PARTIAL" if round(pm, 2) <= 0.47 < round(hm, 2) else ("MATCH" if round(hm, 2) <= 0.47 else "DISAGREE"),
        "holds for the trigger probe accuracy; the evaluation curve gives "
        + ", ".join(f"{s}@{t} {h:.3f}" for s, t, p, h in third if round(h, 2) > 0.47))
    tm = {a: med(sorted(transition(r["curve"]) for r in A[a].values() if bound_routed(r))) for a in ("INT1200", "BASE")}
    add(f, 53, "median transition INT1200 vs BASE", "14400 vs 24000", f"{tm['INT1200']:.0f} vs {tm['BASE']:.0f}")
    for a, ln, t in (("PROBE16", 55, "18 firings, all labelled ok, bound 10/10"), ("THR05", 55, "24, all ok, bound 10/10")):
        nf = sum(len(x) for x in fires[a].values())
        nok = sum(label_ok(c) for x in fires[a].values() for c in x)
        rec = (f"{nf} firings, {'all labelled ok' if nok == nf else f'{nok} labelled ok'}, bound {nbr[a]}/10" if a == "PROBE16"
               else f"{nf}, {'all ok' if nok == nf else f'{nok} ok'}, bound {nbr[a]}/10")
        add(f, ln, f"{a} firings / labelled ok / bound routed", t, rec,
            note="'bound' read as BOUND ROUTED (all 10 also bound)")
    add(f, 56, "six splits instead of three cost nothing (CAP6 vs BASE)", "9 vs 9 (0 vs 0)",
        f"{nbr['CAP6']} vs {nbr['BASE']} ({cm['CAP6'][1]} vs {cm['CAP6'][0]})", "MATCH",
        "the count matches; the claim itself is the content of the CAP6 reading that did not apply (see the reading table)")
    tr_bad = [(a, s) for a in A for s, r in A[a].items() if r.get("transition") != transition(r["curve"])]
    v2_bad = [(a, s) for a in ("BASE",) + VAR for s, r in A[a].items() if r.get("outcome_v2") != outcome_v2(r)]
    AGREE.append(f"S79: stored transition vs curve-derived differ on {tr_bad or 'none'}; stored 'outcome' vs rlib.outcome_k "
                 f"differ on {rec_out or 'none'}; stored 'outcome_v2' vs outcome_v2 differ on {v2_bad or 'none'}.")
    return A


# ══ provenance of the readings ═══════════════════════════════════════════════════════════════════════════════════
def docstring(commit, path):
    t = git("show", f"{commit}:{path}")
    m = re.search(r'"""(.*?)"""', t, re.S)
    return m.group(1) if m else None


def first_run_commit(path):
    for c in git("log", "--reverse", "--format=%h %cI", BRANCH, "--", path).split("\n"):
        if not c.strip():
            continue
        h, d = c.split()
        try:
            n = len(json.loads(git("show", f"{h}:{path}"))["runs"])
        except Exception:
            continue
        if n:
            return h, d, n
    return None, None, 0


def provenance():
    out = []
    for screen, py, rec, texts in (
            ("S78", "explore_h_two.py", TWO, ("the split carries to two streams", "noise is needed at k = 2",
                                               "spare channels are the better device", '"neither device"')),
            ("S79", "explore_h_constants.py", CON, ('"sensitive"    if BASE only >= 4', "the extra splits are harmless",
                                                    '"inconclusive" otherwise'))):
        hits = []
        for t in texts:
            lines = [x for x in git("log", "--format=%h %cI", f"-S{t}", BRANCH, "--", py).split("\n") if x.strip()]
            hits.append((t, lines[-1] if lines else "not found"))   # oldest
        rc = min((h for _, h in hits if h != "not found"), key=lambda x: x.split()[1])
        rh, rd = rc.split()
        fh, fd, fn = first_run_commit(rec)
        started = json.loads(git("show", f"{fh}:{rec}"))["meta"]["provenance"].get("started") if fh else None
        # docstring unchanged from the readings commit to the pin?
        commits = [x for x in git("log", "--format=%h", f"{rh}^..{PIN}", "--", py).split("\n") if x.strip()]
        same = all(docstring(c, py) == docstring(PIN, py) for c in commits + [rh])
        anc = subprocess.call(["git", "-C", os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "merge-base",
                               "--is-ancestor", rh, fh]) == 0 if fh else False
        out.append(dict(screen=screen, py=py, rec=rec, hits=hits, rh=rh, rd=rd, fh=fh, fd=fd, fn=fn, started=started,
                        same=same, commits=commits, precede=bool(fd and rd < fd and anc)))
    return out


UNREG = [
    ("report_4.md:73-76", "S78",
     "\"Noise + reset (the main line's form) breaks it more often ... So at k = 2 symmetry-breaking helps a little "
     "(+2 vs REF, +1 vs SPLIT2)\"",
     "explore_h_two.py l.43 pre-registers only R2 \"noise is needed at k = 2\" (SPLIT2_NOISE only >= 4, SPLIT2 only = 0), "
     "which did not apply. \"helps a little\" reads a direction from 2 vs 0 (p = 0.25) and 1 vs 0 (p = 0.5): an effect "
     "statement the docstring did not define, the mirror image of S61's \"the noise does not matter\"."),
    ("report_4.md:65-72", "S78",
     "\"The exact copy erases the gate's partition, and the same split comes back ... the moments still point along the old "
     "split\"",
     "The diagnostic is pre-specified (docstring l.47-49: \"did the copy break the key split, and what formed instead?\"), "
     "the Adam-moment mechanism is not; the report says it was not measured. Also '8 of 8 firings' counts two firings "
     "with no next check (see the DISAGREE/PARTIAL row)."),
    ("report_4.md:80", "S78",
     "\"15/20 (Wilson 0.53-0.89) is consistent with that [69/80]\"",
     "No comparison of REF with the main line's 69/80 is pre-registered (docstring l.14-15 is background)."),
    ("report_4.md:100-102", "S78",
     "\"The copy works at eight streams because there c0 is empty ... Two-stream key splits need another device.\"",
     "R4 (l.45-46) pre-registers only \"neither device\" for the three devices tested; S78 has no eight-stream arm, so the "
     "causal contrast with eight streams and the prescription are not pre-registered."),
    ("report_5.md:50-54", "S79",
     "\"The check interval works through the same mechanism ... So the interval trades speed for the risk of exhausting "
     "the cap during the flat start.\"",
     "INT1200's pre-registered reading (explore_h_constants.py l.27-29) came out \"inconclusive\" (report_5.md:24, 38). "
     "The report then states that the interval matters and how: an effect claim the docstring's labels do not support "
     "(same pattern as S61's \"the noise does not matter\")."),
    ("report_5.md:55-56", "S79",
     "\"The other constants do not matter here ... Six splits instead of three cost nothing.\"",
     "\"Six splits ... cost nothing\" is the substance of CAP6's pre-registered reading \"the extra splits are harmless\" "
     "(l.30-32), which the report itself records as \"does not apply\" (l.29, 40-41: one extra firing not labelled ok). "
     "\"do not matter\" also restates \"robust\" (l.27: BASE only - V only <= 1 on 10 seeds) as a null effect."),
    ("report_5.md:44-49", "S79",
     "\"A split spent at the start is wasted ... it was simply spent too early.\"",
     "CAP1 \"sensitive\" is pre-registered (l.28); the flat-start mechanism is post hoc (the docstring's DESCRIPTIVE "
     "section, l.33, fixes only transitions and INT1200's first-split timing)."),
    ("report_5.md:57-58", "S79",
     "\"What a replication must state: the cap, the first eligible check and the check interval ... Probe size and the "
     "rise threshold can be stated loosely.\"",
     "Not pre-registered. The first eligible check was never varied on its own: INT1200 changes interval, first check "
     "(2400) and gap (2400) together (l.16); \"stated loosely\" extends \"robust\" on 10 seeds to a recommendation."),
]


# ══ output ═══════════════════════════════════════════════════════════════════════════════════════════════════════
def esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def main():
    s78()
    s79()
    prov = provenance()
    order = {"DISAGREE": 0, "PARTIAL": 1, "NOT RECORD-CHECKABLE": 2, "MATCH": 3}
    rows = sorted(ROWS, key=lambda r: (order[r[5]], r[0], r[1]))
    cnt = collections.Counter(r[5] for r in ROWS)
    L = ["# Session H, reports 4 and 5: recomputation from the records", "",
         f"Generated by `python3 docs/review_2026-10/sessions/recompute_H.py` (stdlib only). Pinned commit: **{PIN}** "
         f"({git('log', '-1', '--format=%cI', PIN).strip()}) on {BRANCH}. Records: `{TWO}` (S78), `{CON}` (S79), "
         "`explore_out/H/checks.json`, all via `git show "
         f"{PIN}:<path>`. Reported values are typed in from `{R4F}` and `{R5F}` at {PIN} (each quoted line checked to "
         "contain the row's arm/comparison); no reported number is an input.", "",
         f"Rows: {len(ROWS)} — " + ", ".join(f"{k} {cnt.get(k, 0)}" for k in order), "",
         "## Definitions used", "",
         "- Helpers: `docs/review_2026-10/rlib.py` (`load`, `runs`, `transition`, `bound`, `discovered`, `bound_routed`, "
         "`fail_class`, `outcome_k`, `mcnemar1`, `wilson`, `band`, `pairs`).",
         "- S78 DISCOVERED (explore_h_two.py l.34-35): bound and final val_cos < 0.5 (`rlib.discovered`); transition = first "
         "evaluation >= 0.95 that holds to the end (`rlib.transition` on the stored curve).",
         "- S78 K4 BOUND ROUTED (l.35-36) and S79 BOUND ROUTED (explore_h_constants.py l.22): `rlib.bound_routed` (one-to-one "
         "argmax map from end.stream_gate, ties deferred to end.ch_map, every stream_acc >= 0.9). Ties hit: "
         f"{len(TIES)} ({', '.join(f'{a}|{s_} stream {i}' for a, s_, i in TIES) or 'none'}; all in S52's store used only for the "
         "pre-run CHECK rows, none in the H records).",
         "- Stored fields vs recomputation (information): " + " ".join(AGREE),
         "- v1 (l.37 / l.23-24): two-stream arms `rlib.fail_class`; K4 and S79 `rlib.outcome_k` (test_stream_recipe.outcome). "
         "Report 5's 'failures v1' column collapses MERGED (n share) to MERGED (explore_h_common.fclass), as recomputed here.",
         "- v2 (l.37-38 / l.23-24): key-first `fail_class_v2` re-implemented from the main line's fail_class_v2.py l.40-48 "
         "(eta^2 by key >= 0.5 -> KEY, then margin >= 0.25 -> STREAM-PARTIAL, then half/index -> POSITION, else OTHER); "
         "`fail_class_k_v2` (l.51-59, multichannel fields, block for half) and `outcome_v2` (l.62-69) for K4 and S79.",
         "- Populations: 'unbound' = not `rlib.bound`; S79 failures = not BOUND ROUTED.",
         "- Readings: S78 R1-R4 explore_h_two.py l.40-46 (exact one-sided McNemar `rlib.mcnemar1`, p < 0.05; R2/R3 '>= 4 and "
         "= 0'); S79 explore_h_constants.py l.26-32 (robust: BASE only - V only <= 1; sensitive: BASE only >= 4 and V only "
         "= 0; else inconclusive; CAP6 harmless: CAP6 >= BASE - 1 and every firing beyond a run's third labelled ok).",
         "- Labelled ok (S79, l.22-23): recomputed for every firing as `map_before.count(key_cs) >= 2 and "
         "map_before.count(key_c0) == 0`; the stored key_ok is not used (its count is given in the notes).",
         "- Bands and Wilson: `rlib.band` (RELIABLE >= ceil(0.9n), MAJORITY >= ceil(0.5n)), `rlib.wilson` (95%, printed "
         "to 2 decimals). p-values printed with 3 significant digits, as the report writer does.",
         "- 'CPU-hours': sum of secs_wall over ok records (the report writer's `runtime`); the `secs` field is given in "
         "the note.",
         "- S78 split table: eta^2 values are stored per check (checks[].diag, diag_after, and the next check's diag); "
         "the row identity (arm, seed, firing update), the pairing with the next check and the end tag (DISCOVERED or v1 "
         "class) are recomputed.",
         "", "## Rows (DISAGREE, PARTIAL, NOT RECORD-CHECKABLE, MATCH)", "",
         "| report:line | item | reported | recomputed | status | note |", "|---|---|---|---|---|---|"]
    for f, ln, item, rep, rec, stt, note in rows:
        L.append(f"| {os.path.basename(f)}:{ln} | {esc(item)} | {esc(rep)} | {esc(rec)} | **{stt}** | {esc(note)} |")
    L += ["", "## Readings: pre-registered before the first run?", "",
          "Readings commit = oldest commit whose diff adds the reading text (`git log --format='%h %cI' -S'<text>' "
          f"{BRANCH} -- <file>`, oldest hit over the texts listed). First run record = oldest commit of the results file "
          "on the branch whose `runs` is non-empty (the file is first committed empty).", "",
          "| screen | file | readings commit (UTC) | texts searched -> oldest hit | first run record commit (UTC) | runs in it | "
          "store meta 'started' | docstring unchanged to the pin | readings precede first run |",
          "|---|---|---|---|---|---|---|---|---|"]
    for p in prov:
        L.append(f"| {p['screen']} | {p['py']} | {p['rh']} ({p['rd']}) | " + "; ".join(f"`{esc(t)}` -> {h}" for t, h in p["hits"])
                 + f" | {p['fh']} ({p['fd']}) | {p['fn']} | {p['started']} UTC | "
                 + f"{'yes' if p['same'] else 'NO'} (commits touching it: {', '.join(p['commits'])}) | "
                 + f"{'yes' if p['precede'] else 'NO'} |")
    L += ["", "Notes: S78's store meta records the run start 04:09:14 UTC at git 3997ae2 (the CHECK-crash fix, 04:09:11, "
          "which changed code only, not the docstring); S79's invocation started 05:01:28 UTC at 72f0bbc, its CHECKs took "
          "14.4 min and were stamped at 05:16:35 with git e6db83b+dirty (the head after an automatic progress commit; "
          "s79.log l.1-28), and its runs were queued after that. The readings commit (04:06:28) precedes both screens' "
          "first runs and both CHECKs.", "",
          "### Readings the reports state that the docstrings did not pre-register", "",
          "| where | screen | statement | why it is not pre-registered |", "|---|---|---|---|"]
    for w, s, t, why in UNREG:
        L.append(f"| {w} | {s} | {esc(t)} | {esc(why)} |")
    L.append("")
    with open(OUT, "w") as fh:
        fh.write("\n".join(L))
    print(f"wrote {OUT}: {len(ROWS)} rows; " + ", ".join(f"{k} {cnt.get(k, 0)}" for k in order))
    for r in rows:
        if r[5] in ("DISAGREE", "PARTIAL"):
            print(f"  {r[5]}: {os.path.basename(r[0])}:{r[1]} {r[2]} | reported {r[3]} | recomputed {r[4]}")


if __name__ == "__main__":
    main()
