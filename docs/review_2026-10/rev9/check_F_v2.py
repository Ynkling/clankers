#!/usr/bin/env python3
"""check_F_v2.py — second pass. The referee's check of what Revision 9's draft says about Session F, run against the
SECOND version of the draft (f67bc07, multichannel_hebbian_report_v9.tex), with the first version (9bfa85a) kept for
locating the first pass's rows.

Built from check_F.py (first pass, 150 rows). Every record recomputation of check_F.py is kept unchanged. What is new:
  * each first-pass row that quotes the draft carries a text fragment of the FIRST draft (QMAP below); the fragment is
    asserted to be at the cited first-draft line, then searched for in the second draft by text (never by line number):
    found -> the row stands at that tex2 line (the wording it checked is unchanged); not found -> REMOVED (the row is
    superseded and listed apart, with the row that checks the replacement wording);
  * rows for every new or changed sentence of the second draft about Session F (section "second draft" below), each with
    a fragment of the second draft asserted to be present;
  * S81's state on origin/claude/explore-F (records, log), since the second draft names S81.

Inputs: JSON run records read with `git show <ref>:<path>` (rlib.load); the screens' docstrings (cited file:line); git
history. The draft's numbers are typed in as DRAFT strings and never used as inputs.

Writes docs/review_2026-10/rev9/check_F_v2.md. Read-only otherwise. Python stdlib + rlib.
"""
import datetime
import json
import os
import statistics
import subprocess
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from rlib import (load, runs, transition, bound, discovered, outcome_k, mcnemar1, wilson, band, pairs,  # noqa: E402
                  TIES, ROOT)

BR = "origin/claude/explore-F"
HEAD = subprocess.check_output(("git", "-C", ROOT, "rev-parse", "--short=7", BR)).decode().strip()
DRAFT_REF, DRAFT_FILE = "9bfa85a", "multichannel_hebbian_report_v9.tex"     # first version (first pass)
NEW_REF = "f67bc07"                                                         # second version (this pass)
R3, R4, R5, R6 = "9a6c2bf", "22ec912", "5b9bc2a", "ed76eea"     # report commits the draft names (tex:501)
OUT = os.path.join(HERE, "check_F_v2.md")
F = "explore_out/F/"


def git(*a):
    return subprocess.check_output(("git", "-C", ROOT) + a).decode()


def st(name, ref=HEAD):
    return load(ref, f"{F}{name}_results.json")


def f2(x):
    return f"{x:.2f}"


def med(xs):
    return statistics.median(xs)


def cnt(f, rs):
    return sum(1 for r in rs.values() if f(r))


def wil(k, n):
    lo, hi = wilson(k, n)
    return f"[{lo:.2f}, {hi:.2f}]"


def cdate(h):
    t = int(git("log", "-1", "--format=%ct", h).strip())
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def is_anc(a, b):
    return subprocess.call(("git", "-C", ROOT, "merge-base", "--is-ancestor", a, b)) == 0


# draft text (for quoting the lines checked; never parsed for numbers)
TEX = git("show", f"{DRAFT_REF}:{DRAFT_FILE}").split("\n")
TEX2 = git("show", f"{NEW_REF}:{DRAFT_FILE}").split("\n")

ROWS = []


def row(tex, item, draft, rec, status, note=""):
    ROWS.append(dict(tex=tex, item=item, draft=str(draft), rec=str(rec), status=status, note=note))


def eq(a, b):
    return "MATCH" if str(a).strip() == str(b).strip() else "DISAGREE"


# ══ stores ══════════════════════════════════════════════════════════════════
DL, WD, WS, DC = st("delta_controls"), st("window_delta"), st("window_delta_scale"), st("decay")      # S53, S54(a,d), S54(b,c), S59
PR, HD, RH = st("premise"), st("hebb_decay"), st("rh_lite")                                            # S68, S69, S70
PB, EI = st("probe"), st("eight")                                                                      # S77, S76
CA, CB = st("configs_a"), st("configs_b")                                                              # S80 (a), (b)

# records unchanged between the report commits and the head used here
STORE_REPORT = {R3: ["premise", "hebb_decay", "rh_lite"], R4: ["probe"], R5: ["eight"], R6: ["configs_a", "configs_b"]}
SAME = {}
for ref, names in STORE_REPORT.items():
    for n in names:
        SAME[n] = (ref, git("rev-parse", f"{ref}:{F}{n}_results.json").strip() == git("rev-parse", f"{HEAD}:{F}{n}_results.json").strip())
for n in ("delta_controls", "window_delta", "window_delta_scale", "decay"):
    SAME[n] = (R3, git("rev-parse", f"{R3}:{F}{n}_results.json").strip() == git("rev-parse", f"{HEAD}:{F}{n}_results.json").strip())

# ══ outcome helpers ═════════════════════════════════════════════════════════
def br(r):
    """BOUND ROUTED, test_stream_recipe.outcome (explore_f_configs.py (b) OUTCOME; explore_f_eight.py:35)."""
    return bool(r.get("ok")) and outcome_k(r) == "BOUND ROUTED"


def routed_star_end(r):
    """S70 ROUTED*@end, explore_f_rh_lite.py:19: margin >= 0.9 and eta^2 by stream at KEY > 0.9 (end statistics)."""
    e = r["end"]
    return e["margin"] >= 0.9 and e["eta_key_by_stream"] > 0.9


def tr(r):
    return transition(r["curve"])


def trs(rs, f=bound):
    return sorted(tr(r) for r in rs.values() if f(r))


# ══ tex:58 / tex:242 / tex:351 — S53, S68 ═══════════════════════════════════
S53G, S53H = runs(DL, "SINGLE_DELTA_G"), runs(DL, "SINGLE_DELTA_H")
S4, S8 = runs(PR, "S68_SINGLE_S4"), runs(PR, "S68_SINGLE_S8")
O4, O8 = runs(PR, "S68_ORACLE_S4"), runs(PR, "S68_ORACLE_S8")
B_HEBB, C_HEBB, C_DELTA, C_OD = runs(WS, "B_HEBB"), runs(WS, "C_HEBB"), runs(WS, "C_DELTA"), runs(WS, "C_ORACLE_DELTA")
k4, k8 = cnt(bound, S4), cnt(bound, S8)
v68 = cnt(bound, O4) >= 1 and cnt(bound, O8) >= 1                                   # explore_f_premise.py:19-20
r68 = ("UNTESTED" if not v68 else "context-tagged keys scale" if k8 >= 7 else
       "the partition's advantage is in the number of streams" if (k8 <= 3 and k4 >= 7) else "neither")   # :22-24

row("tex:58", "one delta channel at four / eight streams (S68 BOUND)", "2/10, 0/10", f"{k4}/{len(S4)}, {k8}/{len(S8)}",
    eq("2/10, 0/10", f"{k4}/{len(S4)}, {k8}/{len(S8)}"), "premise_results.json, rlib.bound on the curve")

# S80 readings (explore_f_configs.py READINGS / VALIDITY)
HA, DA, OHA, ODA = (runs(CA, a) for a in ("HEBB_A", "DELTA_A", "O_HEBB_A", "O_DELTA_A"))
HB, DB, OHB, ODB = (runs(CB, a) for a in ("HEBB_B", "DELTA_B", "O_HEBB_B", "O_DELTA_B"))


def s80_reading(d, h, od, f):
    if cnt(bound, od) < len(od) or len(od) != 2:
        return "UNTESTED", None, None
    b, c, n = pairs(d, h, f)
    if c - b <= 1:
        return "the delta recipe keeps it", b, c
    if c >= 4 and b == 0:
        return "the delta recipe loses it", b, c
    return "inconclusive", b, c


ra80, ba80, ca80 = s80_reading(DA, HA, ODA, discovered)
rb80, bb80, cb80 = s80_reading(DB, HB, ODB, br)
# S76 Phase 2 reading (explore_f_eight.py:37-41)
HS, DS = runs(EI, "HEBB_SPLIT"), runs(EI, "DELTA_SPLIT")
E1 = {a: runs(EI, a) for a in ("O_HEBB", "O_B1_L2", "O_B025_L2", "O_B1_RAW", "O_B025_RAW")}
STAR = load(HEAD, f"{F}eight_delta_star.json")
star = STAR.get("arm")
b76, c76, _ = pairs(DS, HS, br)
v76 = cnt(bound, E1["O_HEBB"]) >= 2 and cnt(bound, E1[star]) >= 2
r76 = ("UNTESTED" if not v76 else "the delta recipe keeps eight streams" if c76 - b76 <= 1 else
       "the delta recipe loses eight streams" if (c76 >= 4 and b76 == 0) else "inconclusive")
row("tex:58", "the delta recipe with fixed beta keeps 2, 4 and 8 streams (S80 (a), (b); S76 Phase 2 readings)",
    "keeps 2, 4 and 8", f"(a) {ra80}; (b) {rb80}; S76: {r76}",
    "MATCH" if (ra80 == rb80 == "the delta recipe keeps it" and r76 == "the delta recipe keeps eight streams") else "DISAGREE",
    "readings recomputed from the records with the docstring rules (explore_f_configs.py READINGS; explore_f_eight.py:38-41)")

# ══ tex:96 — perfect gate, two-stream grouped, no convolution, in session F ═════
PG = [("S53 ORACLE_HEBB_G", runs(DL, "ORACLE_HEBB_G")), ("S54(a) A_ORACLE_HEBB", runs(WD, "A_ORACLE_HEBB")),
      ("S80(a) O_HEBB_A", OHA)]
PGD = [("S53 ORACLE_DELTA_G", runs(DL, "ORACLE_DELTA_G")), ("S54(a) A_ORACLE_DELTA", runs(WD, "A_ORACLE_DELTA")),
       ("S80(a) O_DELTA_A", ODA)]
nH = sum(len(rs) for _, rs in PG)
bH = sum(cnt(bound, rs) for _, rs in PG)
nD = sum(len(rs) for _, rs in PGD)
bD = sum(cnt(bound, rs) for _, rs in PGD)
params = sorted({(a, r.get("params")) for a, rs in PG + PGD for r in rs.values()})
row("tex:96", "perfect gate (N = 256), two-stream grouped, no conv: 'binds on every seed recorded ... 10/10 in session F'",
    "10/10 in session F (every seed recorded)",
    f"Hebbian perfect gate {bH}/{nH} ({'; '.join(f'{a} {cnt(bound, rs)}/{len(rs)}' for a, rs in PG)}); delta perfect gate "
    f"{bD}/{nD}", "PARTIAL",
    f"'every seed recorded' holds, but session F recorded {nH} Hebbian (and {nD} delta) such perfect-gate runs, not 10; 10/10 is "
    f"S53's ORACLE_HEBB_G alone. Configurations from the docstrings: explore_f_delta_controls.py:14-17 (grouped, N = 256, no conv), "
    f"explore_f_window_delta.py:21-23 ((a) grouped S=2 k=2 no conv), explore_f_configs.py '(a) TWO STREAMS ... no conv'; "
    f"recorded params {params} (the draft's 25,984; 26,017 = learned-beta delta, + w_b, b_b)")

# ══ tex:108 — RandHeaderTask-lite (S70) ═════════════════════════════════════
OH70, OD70 = runs(RH, "S70_ORACLE_HEBB"), runs(RH, "S70_ORACLE_DELTA")
RHH, RHD = runs(RH, "S70_HEBB"), runs(RH, "S70_DELTA")
row("tex:108", "RandHeaderTask-lite: S = 2, P = 4, two blocks per stream, composition of 4 into parts 1-3, blocks in random "
    "order, one CTX per block, every (stream, key) once, one query", "as stated",
    "RandHeaderTask(P=4, nb=2, lmin=1, lmax=3), S=2, n_q=1; compositions (1,3),(2,2),(3,1) uniform; 4 blocks uniformly ordered",
    "MATCH", "docstring/code comparison, not records: explore_f_rh_lite.py:7-10; explore_f_tasks.py:9-14, 49-50, 53-63")
Llite = 2 * (2 + 2 * 4) + 3 * 1
row("tex:108", "RandHeaderTask-lite length L", "23", f"{Llite} (= S(NB + 2P) + 3 n_q, explore_f_tasks.py:18, 62-63)", eq("23", Llite),
    "explore_f_rh_lite.py:10 also states L = 23, qpos 21")
row("tex:108", "'the cue is one to six tokens from the write it governs'", "1-6 tokens",
    "KEY positions 1/3/5 and VAL positions 2/4/6 tokens after the block's CTX (blocks of 1-3 pairs); the docstring's '1 to 2*LMAX' "
    "is for any token", "PARTIAL",
    "explore_f_tasks.py:20-21 gives '1 to 2*LMAX tokens' (= 1-6) for a token's distance to its latest CTX; the draft itself (tex:108) "
    "puts the write that must carry the stream at the VAL position, which is 2-6 tokens from its CTX")
row("tex:108", "both perfect gates bind it by update 1200 on five of six seeds (S70 phase 1)", "5 of 6 by 1200",
    f"{sum(1 for r in list(OH70.values()) + list(OD70.values()) if bound(r) and tr(r) <= 1200)} of {len(OH70) + len(OD70)} by 1200",
    eq("5 of 6 by 1200", f"{sum(1 for r in list(OH70.values()) + list(OD70.values()) if bound(r) and tr(r) <= 1200)} of {len(OH70) + len(OD70)} by 1200"),
    f"transitions HEBB {[tr(OH70[s]) for s in sorted(OH70)]}, DELTA {[tr(OD70[s]) for s in sorted(OD70)]} (seeds 330-332)")
iters_all = Counter()
for nm, store in (("S53", DL), ("S54", WD), ("S54bc", WS), ("S59", DC), ("S68", PR), ("S69", HD), ("S70", RH), ("S76", EI),
                  ("S80a", CA), ("S80b", CB)):
    for r in store["runs"].values():
        iters_all[(nm, r.get("iters"))] += 1
i48 = sorted({nm for (nm, it), _ in iters_all.items() if it == 48000})
row("tex:108", "training budgets 'set per test (24,000, 28,800 or 43,200 updates)'", "24000 / 28800 / 43200",
    "F's records: " + ", ".join(f"{nm} {it}" for (nm, it) in sorted({k for k in iters_all}, key=lambda x: (x[0], x[1] or 0))),
    "PARTIAL", f"F screens the draft cites ({', '.join(i48)}: S59 and S69 at tex:244, S53's P = 8 oracles) ran 48,000 updates, "
    "which the list omits")

# ══ tex:124 — the delta write ══════════════════════════════════════════════
dcfg = {}
for nm, rs in (("S53 SINGLE_DELTA_G/H", {**S53G, **{-s: r for s, r in S53H.items()}}), ("S68", {**S4, **{-s: r for s, r in S8.items()}}),
               ("S70_DELTA", RHD), ("S80 DELTA_A", DA)):
    dcfg[nm] = sorted({json.dumps({k: v for k, v in r["delta_cfg"].items() if k in ("beta", "decay", "normalize")}, sort_keys=True)
                       for r in rs.values()})
dB = sorted({json.dumps(r["delta"], sort_keys=True) for r in list(DB.values()) + list(ODB.values())})
dS = sorted({json.dumps({k: v for k, v in r["delta"].items() if k in ("beta", "normalize", "arm")}, sort_keys=True) for r in DS.values()})
row("tex:124", "the delta write: beta fixed at 1, unit (L2) keys, decay fixed at 0.95 (F's fixed-beta screens)", "beta 1, L2, decay 0.95",
    "; ".join(f"{k}: {v}" for k, v in dcfg.items()) + f"; S80 (b) delta: {dB}; S76 DELTA_SPLIT: {dS}", "MATCH",
    "records' delta_cfg / delta fields; decay 0.95 itself is in the docstrings (explore_f_delta_controls.py:28), not a record field")
CHK = json.loads(git("show", f"{BR}:{F}checks_s80_20261010_192401.json"))
import re  # noqa: E402
b0 = []
for name, ok_ in CHK["rows"]:
    if name.startswith("C2") and "beta 0" in name:
        m = re.search(r"logits max \|diff\| ([0-9.e+-]+)", name)
        if m:
            b0.append(float(m.group(1)))
c6 = [n for n, ok_ in CHK["rows"] if n.startswith("C6")]
row("tex:124", "'at beta = 0 its logits agree with the Hebbian model's to 1e-7'", "1e-7",
    f"max |logit diff| over the C2 beta-0 cases {max(b0):.2e} (cases {', '.join(f'{x:.2e}' for x in b0)}); asserted bound 1e-5",
    "PARTIAL", "checks file explore_out/F/checks_s80_20261010_192401.json (not a run record): three cases exceed 1e-7 (largest "
    "1.79e-07, sequential S=8 k=16 conv); 'to about 1e-7' would hold")
row("tex:124", "'with the new path disabled it reproduces a recorded run bit for bit'", "a recorded run, bit for bit",
    "C6: LOCAL3_SLOW|160 reproduced bit for bit THROUGH 2400 updates (curve, statistics, decoder)", "PARTIAL",
    "checks file row C6 (and the screen CHECKs in s76/s80 logs): every such reproduction is through 2400 (S80, S54) or 7200 "
    "(S76, S81) updates, not a whole run; the full-run bit-for-bit reruns are S77's (probe records, tex:250)")
BL = {"S53 SINGLE_DELTA_L_G": runs(DL, "SINGLE_DELTA_L_G"), "S53 SINGLE_DELTA_L_H": runs(DL, "SINGLE_DELTA_L_H"),
      "S54(a) A_DELTA": runs(WD, "A_DELTA"), "S54(a) A_ORACLE_DELTA": runs(WD, "A_ORACLE_DELTA"),
      "S54(b) B_DELTA": runs(WS, "B_DELTA"), "S54(b) B_ORACLE_DELTA": runs(WS, "B_ORACLE_DELTA"),
      "S54(c) C_DELTA": runs(WS, "C_DELTA"), "S54(c) C_ORACLE_DELTA": runs(WS, "C_ORACLE_DELTA"),
      "S54(d) D_DELTA": runs(WD, "D_DELTA"), "S54(d) D_ORACLE_DELTA": runs(WD, "D_ORACLE_DELTA")}


def beta_med(rs, role):
    return [med([r["beta_by_role"][L][role] for r in rs.values() if r.get("beta_by_role")]) for L in range(3)]


bsum = {a: beta_med(rs, "val") for a, rs in BL.items()}
allb = [x for v in bsum.values() for x in v]
up = [a for a, v in bsum.items() if min(v) > 0.5]
row("tex:124", "'A learned beta collapses toward \"do not write\"' (beta = 0.5 at init), stated generally",
    "collapses toward 0", "median beta at VAL, layers 1/2/3: " + "; ".join(f"{a} {'/'.join(f2(x) for x in v)}" for a, v in bsum.items()),
    "DISAGREE" if up else "MATCH",
    f"records' beta_by_role (a pre-fixed DIAGNOSTIC: explore_f_delta_controls.py:51, explore_f_window_delta.py:51, "
    f"explore_f_window_delta_scale.py:34). Collapse at VAL in S53 and S54(d) (and partly C_DELTA), but in {up} learned beta ROSE above "
    "0.5 at VAL at every layer (S54(a) A_DELTA bound 36/40, S54(b) B_DELTA 20/20 with it); S54(c)'s slow oracle kept "
    f"{'/'.join(f2(x) for x in bsum['S54(c) C_ORACLE_DELTA'])}. At KEY it is near 0 in S53 "
    f"({'/'.join(f2(x) for x in beta_med(BL['S53 SINGLE_DELTA_L_G'], 'key'))})")
sel = [bsum[a] for a in ("S53 SINGLE_DELTA_L_G", "S53 SINGLE_DELTA_L_H", "S54(d) D_DELTA", "S54(d) D_ORACLE_DELTA")]
row("tex:246", "learned beta's 'collapse toward zero in S53 and S54(d)'", "collapse toward zero",
    "; ".join(f"{a} {'/'.join(f2(x) for x in bsum[a])}" for a in ("S53 SINGLE_DELTA_L_G", "S53 SINGLE_DELTA_L_H", "S54(d) D_DELTA", "S54(d) D_ORACLE_DELTA")),
    "MATCH" if max(max(v) for v in sel) < 0.5 and med([x for v in sel for x in v]) < 0.15 else "PARTIAL",
    "median beta at VAL by layer, from 0.5 at init (S53 H layer 1 0.35 is the highest)")

# ══ tex:242 — premise ═══════════════════════════════════════════════════════
row("tex:242", "S53 one delta channel, two-stream header layout P = 4 (SINGLE_DELTA_H)", "8/10", f"{cnt(bound, S53H)}/{len(S53H)}",
    eq("8/10", f"{cnt(bound, S53H)}/{len(S53H)}"))
row("tex:242", "S53 one delta channel, grouped two-stream layout (SINGLE_DELTA_G)", "4/10", f"{cnt(bound, S53G)}/{len(S53G)}",
    eq("4/10", f"{cnt(bound, S53G)}/{len(S53G)}"))
sd = sorted(S4)
row("tex:242", "S68 seeds; updates", "300-309, 43200", f"{sd[0]}-{sd[-1]}, {sorted({r['iters'] for r in list(S4.values()) + list(S8.values())})[0]}",
    eq("300-309, 43200", f"{sd[0]}-{sd[-1]}, {sorted({r['iters'] for r in list(S4.values()) + list(S8.values())})[0]}"))
row("tex:242", "S68 one delta channel at four / eight streams", "2/10, 0/10", f"{k4}/10, {k8}/10", eq("2/10, 0/10", f"{k4}/10, {k8}/10"))
ot = sorted({tr(r) for r in list(O4.values()) + list(O8.values()) if bound(r)})
row("tex:242", "both delta perfect gates (S68_ORACLE_S4 / _S8) bind at 1200", "at 1200",
    f"{cnt(bound, O4)}/{len(O4)}, {cnt(bound, O8)}/{len(O8)}; transitions {ot}", "MATCH" if ot == [1200] and cnt(bound, O4) == 2 and cnt(bound, O8) == 2 else "DISAGREE")
row("tex:242", "gated Hebbian recipe on other seeds (S54(b) B_HEBB, S54(c) C_HEBB, BOUND ROUTED)", "19/20 and 9/10",
    f"{cnt(br, B_HEBB)}/{len(B_HEBB)} and {cnt(br, C_HEBB)}/{len(C_HEBB)}", eq("19/20 and 9/10", f"{cnt(br, B_HEBB)}/{len(B_HEBB)} and {cnt(br, C_HEBB)}/{len(C_HEBB)}"),
    f"seeds {min(B_HEBB)}-{max(B_HEBB)} / {min(C_HEBB)}-{max(C_HEBB)}; the comparison is the docstring's REFERENCES (explore_f_premise.py:15-17)")
row("tex:242", "reading 'context-tagged keys scale': name and threshold", "'context-tagged keys scale' (>= 7/10 at eight)",
    "'context-tagged keys scale' if S68_SINGLE_S8 BOUND >= 7/10", "MATCH", "explore_f_premise.py:22")
row("tex:242", "reading 'the partition's advantage is in the number of streams': threshold", ">= 7/10 at four",
    ">= 7/10 at four AND <= 3/10 at eight", "PARTIAL", "explore_f_premise.py:23; the draft omits the S = 8 condition (it holds, 0/10; the "
    "reading fails on S = 4, 2/10)")
row("tex:242", "S68 readings (recomputed)", "both 'neither'", r68, "MATCH" if r68 == "neither" else "DISAGREE",
    f"VALID (oracles bind >= 1/2 at both S): {v68}")

# ══ tex:244 — S59 and S69 ═══════════════════════════════════════════════════
S59 = {d: runs(DC, f"SINGLE_{d}") for d in ("FIXED", "ROUTED", "GDN")}
O59 = {d: runs(DC, f"ORACLE_{d}") for d in ("FIXED", "ROUTED", "GDN")}
H69 = {d: runs(HD, f"S69_{d}") for d in ("FIXED", "ROUTED", "GDN", "ORACLE")}
c59 = {d: cnt(bound, rs) for d, rs in S59.items()}
row("tex:244", "S59 single delta channel, learned forget gate (ROUTED, GDN) vs fixed decay, header P = 8",
    "9/10 and 8/10 against 3/10", f"{c59['ROUTED']}/10 and {c59['GDN']}/10 against {c59['FIXED']}/10",
    eq("9/10 and 8/10 against 3/10", f"{c59['ROUTED']}/10 and {c59['GDN']}/10 against {c59['FIXED']}/10"))


def amed(rs, L, role):
    return med([r["decay_by_role"][L]["ch0"][role] for r in rs.values()])


a1 = sorted([amed(S59[d], 0, "ctx") for d in ("ROUTED", "GDN")])
a3 = sorted([amed(S59[d], 2, "ctx") for d in ("ROUTED", "GDN")])
akv = [amed(S59[d], L, ro) for d in ("ROUTED", "GDN") for L in range(3) for ro in ("key", "val")]
row("tex:244", "S59 median alpha at CTX, layer 1 (GDN, ROUTED)", "0.18-0.26", f"{f2(a1[0])}-{f2(a1[1])}", eq("0.18-0.26", f"{f2(a1[0])}-{f2(a1[1])}"),
    "decay_results.json decay_by_role[layer].ch0, median over the 10 runs")
row("tex:244", "S59 median alpha at CTX, layer 3 (GDN, ROUTED)", "0.72-0.74", f"{f2(a3[0])}-{f2(a3[1])}", eq("0.72-0.74", f"{f2(a3[0])}-{f2(a3[1])}"),
    f"unrounded {a3[0]:.4f} / {a3[1]:.4f}; layer 2: {f2(amed(S59['GDN'], 1, 'ctx'))} (GDN), {f2(amed(S59['ROUTED'], 1, 'ctx'))} (ROUTED)")
row("tex:244", "S59 median alpha at KEY and VAL", "1.00", f"{f2(min(akv))}-{f2(max(akv))} (all layers, both decays)",
    "MATCH" if f2(min(akv)) == f2(max(akv)) == "1.00" else "DISAGREE", f"unrounded min {min(akv):.4f}")
c69 = {d: cnt(bound, H69[d]) for d in ("FIXED", "ROUTED", "GDN")}
row("tex:244", "S69 one Hebbian channel, FIXED / ROUTED / GDN BOUND (same seeds 350-359 as S59)", "0/10, 0/10 and 0/10",
    f"{c69['FIXED']}/10, {c69['ROUTED']}/10 and {c69['GDN']}/10 (seeds {sorted(H69['FIXED']) == sorted(S59['FIXED'])} same as S59)",
    "MATCH" if all(v == 0 for v in c69.values()) and sorted(H69['FIXED']) == sorted(S59['FIXED']) else "DISAGREE",
    "'same batches': explore_f_hebb_decay.py:13 (docstring), not checkable from the records")
h1 = [amed(H69[d], 0, "ctx") for d in ("ROUTED", "GDN")]
row("tex:244", "S69 median alpha at CTX, layer 1 (ROUTED, GDN)", "0.975 and 0.988", f"{h1[0]:.3f} and {h1[1]:.3f}",
    eq("0.975 and 0.988", f"{h1[0]:.3f} and {h1[1]:.3f}"))
fa = [med([r["acc"] for r in H69[d].values()]) for d in ("ROUTED", "GDN", "FIXED")]
row("tex:244", "S69 median final accuracy, learned decay vs fixed", "0.17 against 0.22", f"{f2(fa[0])} (ROUTED), {f2(fa[1])} (GDN) against {f2(fa[2])}",
    "MATCH" if f2(fa[0]) == f2(fa[1]) == "0.17" and f2(fa[2]) == "0.22" else "DISAGREE")
r69 = ("the forget gate needs the delta rule" if (c69["ROUTED"] <= c69["FIXED"] + 1 and c69["GDN"] <= c69["FIXED"] + 1) else
       "the forget gate alone suffices" if (c69["ROUTED"] >= 8 or c69["GDN"] >= 8) else "neither")
row("tex:244", "S69 reading (paragraph title)", "The forget gate needs the delta rule", r69,
    "MATCH" if r69 == "the forget gate needs the delta rule" else "DISAGREE",
    f"explore_f_hebb_decay.py:23; validity S69_ORACLE {cnt(bound, H69['ORACLE'])}/2 (>= 1 needed, :21)")

# ══ tex:246 — S76 and S54(c) ════════════════════════════════════════════════
def c54_reading():
    b, c, _ = pairs(C_DELTA, C_HEBB, br)
    return "delta better" if (b >= 4 and c == 0) else "delta not worse" if c - b <= 2 else "delta worse"


row("tex:246", "S54(c) window-gate recipe on delta vs Hebbian channels at eight streams (BOUND ROUTED)", "3/10 against 9/10",
    f"{cnt(br, C_DELTA)}/10 against {cnt(br, C_HEBB)}/10 ({c54_reading()})", "MATCH" if (cnt(br, C_DELTA), cnt(br, C_HEBB)) == (3, 9) else "DISAGREE",
    "explore_f_window_delta_scale.py:29-32")
row("tex:246", "S54(c) delta perfect gate transitions", "22,800-28,800", "-".join(map(str, trs(C_OD))),
    eq("22800-28800", "-".join(map(str, trs(C_OD)))))
ts = {a: [tr(E1[a][s]) if bound(E1[a][s]) else None for s in sorted(E1[a])] for a in E1}
seeds1 = sorted(E1["O_HEBB"])
row("tex:246", "S76 Phase 1 seeds", "four seeds", f"{len(seeds1)} ({seeds1[0]}-{seeds1[-1]})", "MATCH" if len(seeds1) == 4 else "DISAGREE")
row("tex:246", "L2 keys, beta = 1 (O_B1_L2): binds all four at the first evaluation", "all four at 1200", str(ts["O_B1_L2"]),
    "MATCH" if ts["O_B1_L2"] == [1200] * 4 else "DISAGREE")
row("tex:246", "L2 keys, beta = 0.25 (O_B025_L2)", "at 2400", str(ts["O_B025_L2"]), "MATCH" if ts["O_B025_L2"] == [2400] * 4 else "DISAGREE")
hrange = f"{min(ts['O_HEBB'])}-{max(ts['O_HEBB'])}"
row("tex:246", "Hebbian oracle (O_HEBB) transitions", "2400-6000", hrange + f" {ts['O_HEBB']}", "MATCH" if hrange == "2400-6000" else "DISAGREE")
soon1 = sum(1 for a, b in zip(ts["O_B1_L2"], ts["O_HEBB"]) if a < b)
soon025 = sum(1 for a, b in zip(ts["O_B025_L2"], ts["O_HEBB"]) if a < b)
row("tex:246", "'sooner than the Hebbian oracle' (both L2 arms, per seed)", "sooner",
    f"beta 1: sooner on {soon1}/4; beta 0.25: sooner on {soon025}/4 (ties at 2400 on seeds "
    f"{[s for s, a, b in zip(seeds1, ts['O_B025_L2'], ts['O_HEBB']) if a == b]})", "PARTIAL",
    "true for beta = 1; the sentence also attaches it to beta = 0.25, which ties the Hebbian oracle on two of four seeds")
row("tex:246", "raw keys: one seed diverged, one bound late (16,800)", "one diverged, one late (16800)",
    f"O_B1_RAW {ts['O_B1_RAW']}; O_B025_RAW {ts['O_B025_RAW']}", "PARTIAL",
    "true of beta = 1 (O_B1_RAW: 1300 diverged, 1301 at 16800 > 2 x 3600). The draft does not say beta = 1: at beta = 0.25 raw keys "
    "bound 4/4 but O_B025_RAW|1300 at 7200 also misses the 2x mark (Hebbian 2400) — 'one bound late' undercounts raw keys overall")


def meets(a):
    out = 0
    for s in seeds1:
        h, x = E1["O_HEBB"][s], E1[a][s]
        if bound(x) and (not bound(h) or tr(x) <= 2 * tr(h)):
            out += 1
    return out >= 3


w = {a: meets(a) for a in ("O_B1_L2", "O_B025_L2", "O_B1_RAW", "O_B025_RAW")}
r76a = ("the slowness is the overwrite" if (w["O_B025_L2"] and w["O_B025_RAW"] and not w["O_B1_L2"] and not w["O_B1_RAW"]) else
        "the slowness is the key normalisation" if (w["O_B1_RAW"] and w["O_B025_RAW"] and not w["O_B1_L2"] and not w["O_B025_L2"]) else "neither")
row("tex:246", "Phase 1 readings (the overwrite; the key normalisation)", "both 'neither', because no L2 arm was slow",
    f"{r76a}; within 2x: {w}", "MATCH" if r76a == "neither" and w["O_B1_L2"] and w["O_B025_L2"] else "DISAGREE",
    "explore_f_eight.py:19-24; each reading needs an L2 arm NOT within 2x, and both L2 arms are within 2x")
row("tex:246", "Phase 2 (ten seeds) BOUND ROUTED delta vs Hebbian (b vs c)", "9/10 against 10/10 (1 vs 0)",
    f"{cnt(br, DS)}/10 against {cnt(br, HS)}/10 ({c76} vs {b76})", eq("9/10 against 10/10 (1 vs 0)", f"{cnt(br, DS)}/10 against {cnt(br, HS)}/10 ({c76} vs {b76})"),
    f"seeds {min(DS)}-{max(DS)}; McNemar one-sided p(Hebb > delta) = {mcnemar1(c76, b76):g}")
both76 = [s for s in sorted(DS) if br(DS[s]) and br(HS[s])]
soon76 = [s for s in both76 if tr(DS[s]) < tr(HS[s])]
mt = (med([tr(DS[s]) for s in both76]), med([tr(HS[s]) for s in both76]))
row("tex:246", "sooner on eight of the nine seeds both bind (median)", "8 of 9 (14,400 against 24,000)",
    f"{len(soon76)} of {len(both76)} ({mt[0]:g} against {mt[1]:g})", eq("8 of 9 (14400 against 24000)", f"{len(soon76)} of {len(both76)} ({mt[0]:g} against {mt[1]:g})"))
row("tex:246", "Phase 2 reading", "'the delta recipe keeps eight streams' applies", r76, "MATCH" if r76 == "the delta recipe keeps eight streams" else "DISAGREE",
    f"explore_f_eight.py:37-41; DELTA* = {star} (eight_delta_star.json)")
fired = [(s, c) for s, r in sorted(DS.items()) for c in r.get("checks", []) if c.get("fired")]
notok = [(s, c) for s, c in fired if not c.get("key_ok")]
row("tex:246", "delta splits whose target was not labelled ok", "six of the twelve", f"{len(notok)} of {len(fired)}",
    eq("6 of 12", f"{len(notok)} of {len(fired)}"), "checks[].fired / key_ok in eight_results.json (label: s37c.targeting)")
merged_before = [(s, c["step"], max(c["map_before"].count(x) for x in c["map_before"]), c["map_before"].count(c["key_cs"]),
                  c["map_before"].count(c["key_c0"])) for s, c in notok]
nm_before = sum(1 for m in merged_before if m[2] >= 2)
row("tex:246", "those six splits 'fired before the map had merged'", "before the map had merged",
    f"the labelled map just before held a channel with >= 2 streams at {nm_before} of {len(notok)} of them "
    f"(seed, update, largest share, streams on c*, streams on c0: {merged_before})", "DISAGREE" if nm_before else "MATCH",
    "map_before = routing_k ch_map at the check (explore_f_eight_child.py:73-77); 'not labelled ok' means c* held < 2 streams "
    "or c0 was not empty (explore_split_target_child.py:71-73). At every one of the six the map HAD merged; the split copied a "
    "channel that was not the merged one (c* held 0-1 streams in 5; at 1314@9600 c0 held a stream). report_5.md:54-55 has the same sentence")
single = [s for s, _ in notok if DS[s]["splits"] == 1]
row("tex:246", "'mostly as single splits in runs that then bound'", "mostly single; runs bound",
    f"{len(single)} of {len(notok)} in single-split runs; runs bound: {sum(1 for s, _ in notok if bound(DS[s]))} of {len(notok)}",
    "MATCH" if len(single) * 2 > len(notok) and all(bound(DS[s]) for s, _ in notok) else "DISAGREE")
c54d = runs(WS, "C_DELTA")[310]
row("tex:246", "'S54(c) differed from S76 in two things, the learned beta and the noisy split with a reset'", "two things",
    f"learned beta (C_DELTA delta_cfg beta = {c54d['delta_cfg']['beta']!r}); split: S37/S48 noisy split with Adam reset vs H's exact copy; "
    f"ALSO seeds ({min(C_DELTA)}-{max(C_DELTA)} vs {min(DS)}-{max(DS)}; oracles {sorted(C_OD)} vs {seeds1})", "PARTIAL",
    "explore_f_window_delta_scale.py:3-5, 12; explore_f_eight.py:30-31. The two named differences are real; the seeds also differ "
    "(report_5.md:44 says 'two things' too)")
row("tex:246", "learned-beta attribution", "post hoc", "post hoc: no S76 arm varies learned beta", "MATCH", "marked post hoc in the draft")

# ══ tex:248 — S80 ═══════════════════════════════════════════════════════════
kA = (cnt(discovered, DA), cnt(discovered, HA))
row("tex:248", "S80(a) two streams, k = 2, no conv: delta vs Hebbian (DISCOVERED)", "10/10 against 10/10",
    f"{kA[0]}/{len(DA)} against {kA[1]}/{len(HA)}", eq("10/10 against 10/10", f"{kA[0]}/{len(DA)} against {kA[1]}/{len(HA)}"),
    f"k = {sorted({r['k'] for r in list(DA.values()) + list(HA.values())})}; DISCOVERED = bound and val_cos < 0.5 "
    "(explore_f_configs.py (a) OUTCOME); BOUND alone also 10/10 each")
fe = sorted({r["curve"][0][0] for r in DA.values()})
tA = [tr(DA[s]) for s in sorted(DA)]
tHA = [tr(HA[s]) for s in sorted(HA)]
row("tex:248", "delta at the first evaluation on every seed against 2400-6000", "first evaluation (1200) every seed; Hebbian 2400-6000",
    f"delta {sorted(set(tA))} (first evaluation {fe}); Hebbian {min(tHA)}-{max(tHA)}",
    "MATCH" if set(tA) == {fe[0]} and (min(tHA), max(tHA)) == (2400, 6000) else "DISAGREE", f"Hebbian per seed {tHA}")
kB = (cnt(br, DB), cnt(br, HB))
xB = [k for k, r in list(DB.items()) + list(HB.items()) if (outcome_k(r) == "BOUND ROUTED") != (r.get("outcome") == "BOUND ROUTED")]
row("tex:248", "S80(b) four streams, k = 16, conv: delta vs Hebbian (BOUND ROUTED)", "10/10 against 10/10",
    f"{kB[0]}/{len(DB)} against {kB[1]}/{len(HB)}", eq("10/10 against 10/10", f"{kB[0]}/{len(DB)} against {kB[1]}/{len(HB)}"),
    f"k = {sorted({r['k'] for r in list(DB.values()) + list(HB.values())})}, cfg {sorted({r['cfg'] for r in DB.values()})}; rlib.outcome_k vs "
    f"the recorded outcome differ on {len(xB)} records")
tB = sorted({tr(r) for r in list(DB.values()) + list(HB.values())})
row("tex:248", "both at 3600 on every seed", "3600 every seed", str(tB), "MATCH" if tB == [3600] else "DISAGREE")
nfire = sum(1 for r in list(DB.values()) + list(HB.values()) for c in r.get("checks", []) if c.get("fired"))
row("tex:248", "no split firing in either arm", "none", f"{nfire} firings; recorded 'splits' {sorted({r['splits'] for r in list(DB.values()) + list(HB.values())})}",
    "MATCH" if nfire == 0 else "DISAGREE")
row("tex:248", "both readings", "'keeps it' (both)", f"(a) {ra80} (b={ba80}, c={ca80}); (b) {rb80} (b={bb80}, c={cb80})",
    "MATCH" if ra80 == rb80 == "the delta recipe keeps it" else "DISAGREE",
    "explore_f_configs.py READINGS 'the delta recipe keeps it' if c - b <= 1; VALIDITY O_DELTA_A "
    f"{cnt(bound, ODA)}/2, O_DELTA_B {cnt(bound, ODB)}/2")
loss = (10 * (1 - 69 / 80), 10 * (1 - 38 / 40))
row("tex:248", "'at the main line's rates (0.86 and 0.95) a Hebbian arm loses about 1.4 and 0.5 runs in ten'", "1.4 and 0.5",
    f"{loss[0]:.3f} and {loss[1]:.2f} (from 69/80 = {69 / 80:.4f}, 38/40 = {38 / 40:.2f})", "MATCH",
    "arithmetic on the main-line counts in tex:279-280 and explore_f_configs.py REF (69/80) / docstring (b) (38/40); the main-line "
    "counts themselves are outside this check")
row("tex:248", "''keeps it' means that no seed failed on delta'", "no delta failure",
    f"delta unsuccessful runs: (a) {len(DA) - kA[0]}, (b) {len(DB) - kB[0]}", "MATCH" if kA[0] == len(DA) and kB[0] == len(DB) else "DISAGREE")

# ══ tex:250 — S77 and the S80 probe ═════════════════════════════════════════
PA = {a: runs(PB, a) for a in ("S53_DELTA", "S59_GDN", "S69_GDN")}
ORIG = {"S53_DELTA": runs(DL, "SINGLE_DELTA_H"), "S59_GDN": runs(DC, "SINGLE_GDN"), "S69_GDN": runs(HD, "S69_GDN")}
same_curve = sum(1 for a in PA for s, r in PA[a].items() if s in ORIG[a] and r["curve"] == ORIG[a][s]["curve"])
row("tex:250", "S77: 30 recorded single-channel runs rerun, every curve bit for bit", "30, every curve bit for bit",
    f"{same_curve} of {sum(len(v) for v in PA.values())} rerun curves equal the original records' curves exactly; record field "
    f"'repro' true on {sum(1 for v in PA.values() for r in v.values() if r.get('repro'))}", "MATCH" if same_curve == 30 else "DISAGREE",
    "probe_results.json vs delta_controls (SINGLE_DELTA_H), decay (SINGLE_GDN), hebb_decay (S69_GDN): compared here, not only the "
    "records' own assertion")


def pv(r, k, init=False):
    return r["probe_init" if init else "probe"][k]["acc"]


bdelta = [r for a in ("S53_DELTA", "S59_GDN") for r in PA[a].values() if bound(r)]
tagged = [r for r in bdelta if pv(r, "L3|K") >= 0.9 and pv(r, "L3|V") >= 0.9]
row("tex:250", "stream linearly readable at layer 3 (K and V >= 0.9) in bound delta runs", "one of sixteen",
    f"{len(tagged)} of {len(bdelta)} ({[(r['arm'], r['seed']) for r in tagged]})", eq("1 of 16", f"{len(tagged)} of {len(bdelta)}"))
m3 = {(a, x): med([pv(r, f"L3|{x}") for r in PA[a].values()]) for a in ("S53_DELTA", "S59_GDN") for x in "KV"}
m3b = {(a, x): med([pv(r, f"L3|{x}") for r in PA[a].values() if bound(r)]) for a in ("S53_DELTA", "S59_GDN") for x in "KV"}
rk = f"{f2(min(m3[(a, 'K')] for a in ('S53_DELTA', 'S59_GDN')))}-{f2(max(m3[(a, 'K')] for a in ('S53_DELTA', 'S59_GDN')))}"
rkv = f"{f2(min(m3.values()))}-{f2(max(m3.values()))}"
row("tex:250", "medians at layer 3 in the delta arms", "0.57-0.67",
    f"KEY {rk}; KEY and VAL {rkv} (all runs: " + ", ".join(f"{a} {x} {f2(v)}" for (a, x), v in m3.items()) + "; bound only: "
    + ", ".join(f"{a} {x} {f2(v)}" for (a, x), v in m3b.items()) + ")", "PARTIAL",
    "AMBIGUOUS population: matches the KEY medians; with VAL (S59_GDN 0.56) the range is 0.56-0.67")
u53 = [r for r in PA["S53_DELTA"].values() if not bound(r)]
vv = sorted(pv(r, "L3|V") for r in u53)
row("tex:250", "the two S53 runs carrying the stream at value positions failed to bind (L3 VAL)", "0.92-1.00; the two unbound",
    f"unbound S53 seeds {sorted(r['seed'] for r in u53)}: L3 V {f2(vv[0])}-{f2(vv[-1])}; bound S53 runs with L3 V >= 0.9: "
    f"{sum(1 for r in PA['S53_DELTA'].values() if bound(r) and pv(r, 'L3|V') >= 0.9)}",
    "MATCH" if len(u53) == 2 and f"{f2(vv[0])}-{f2(vv[-1])}" == "0.92-1.00" else "DISAGREE")
hk = [med([pv(r, f"{L}|{x}") for r in PA["S69_GDN"].values()]) for L in ("L1", "L2", "L3", "final") for x in "KV"]
hkK = [med([pv(r, f"{L}|K") for r in PA["S69_GDN"].values()]) for L in ("L1", "L2", "L3", "final")]
hr = [pv(r, f"{L}|{x}") for r in PA["S69_GDN"].values() for L in ("L1", "L2", "L3", "final") for x in "KV"]
hi = [med([pv(r, f"L3|{x}", True) for r in PA["S69_GDN"].values()]) for x in "KV"]
row("tex:250", "the Hebbian channel (S69_GDN) carries nothing anywhere", "0.50-0.53",
    f"medians L1-final at KEY {f2(min(hkK))}-{f2(max(hkK))}; KEY and VAL {f2(min(hk))}-{f2(max(hk))}; per run {f2(min(hr))}-{f2(max(hr))}; "
    f"at initialisation L3 K/V {f2(hi[0])}/{f2(hi[1])}", "PARTIAL",
    "AMBIGUOUS population: KEY medians match; VAL takes the low end to 0.49; per-run values reach 0.60")
pA = {a: {k: [pv(rs[s], k) for s in sorted(rs)] for k in ("L2|K", "L3|K", "L3|V")} for a, rs in (("DELTA_A", DA), ("HEBB_A", HA), ("O_DELTA_A", ODA))}
l3k = pA["DELTA_A"]["L3|K"]
row("tex:250", "S80: delta stack carries the stream at KEY positions, 1.00 at layer 3 on all ten runs", "1.00 on all ten",
    f"per-run L3 K {', '.join(f2(x) for x in l3k)}: {sum(1 for x in l3k if f2(x) == '1.00')} of 10 print 1.00 (median {f2(med(l3k))}, "
    f"min {min(l3k):.4f})", "MATCH" if all(f2(x) == "1.00" for x in l3k) else "DISAGREE",
    "configs_a_results.json probe['L3|K'].acc; report 6 itself prints 'L3 0.98-1.00'")
l2k = pA["DELTA_A"]["L2|K"]
row("tex:250", "S80: 'from layer 2 on' (L2 K, DELTA_A)", "from layer 2", f"L2 K per run {f2(min(l2k))}-{f2(max(l2k))}, median {f2(med(l2k))}",
    "MATCH" if min(l2k) >= 0.9 else "PARTIAL", "read as >= 0.9 at L2 on every run (S77's criterion level)")
row("tex:250", "S80: Hebbian (HEBB_A) at layer 3 KEY", "0.57", f"median {f2(med(pA['HEBB_A']['L3|K']))} (per run {f2(min(pA['HEBB_A']['L3|K']))}-{f2(max(pA['HEBB_A']['L3|K']))})",
    eq("0.57", f2(med(pA["HEBB_A"]["L3|K"]))) if True else "", "")
l3v = pA["DELTA_A"]["L3|V"]
row("tex:250", "S80: not at value positions (DELTA_A L3 V)", "0.54-0.65", f"{f2(min(l3v))}-{f2(max(l3v))}", eq("0.54-0.65", f"{f2(min(l3v))}-{f2(max(l3v))}"))
row("tex:250", "S80: 'in the perfect-gate arm as well' (O_DELTA_A KEY high, VAL not)", "as well",
    f"O_DELTA_A L2 K {[f2(x) for x in pA['O_DELTA_A']['L2|K']]}, L3 K {[f2(x) for x in pA['O_DELTA_A']['L3|K']]}, L3 V {[f2(x) for x in pA['O_DELTA_A']['L3|V']]}",
    "MATCH" if min(pA["O_DELTA_A"]["L3|K"]) >= 0.9 and max(pA["O_DELTA_A"]["L3|V"]) < 0.9 else "DISAGREE")
row("tex:250", "'session F reads this, post hoc, as adjacency'", "post hoc", "report_6.md 'Post hoc (not tested here)'", "MATCH",
    "explore_f_configs.py: 'PROBE (descriptive, no reading)'")
row("tex:250", "'On the randomised header layout the stack carries the stream only in the solutions that bind without routing'",
    "(claim)", "no probe in F's S70 records", "NOT RECORD-CHECKABLE", "session G's probe (Section far); outside F's records")

# ══ tex:252 / 314 / 349 / 351 / 353 / 378 / 386 ════════════════════════════
soonA = sum(1 for s in DA if tr(DA[s]) < tr(HA[s]))
soonB = sum(1 for s in DB if tr(DB[s]) < tr(HB[s]))
row("tex:252", "'binds sooner at two and eight'", "sooner at two and eight",
    f"two: delta sooner on {soonA}/10 pairs; four: {soonB}/10 (all tie at 3600); eight: {len(soon76)}/{len(both76)}",
    "MATCH" if soonA == 10 and len(soon76) == 8 else "DISAGREE")
row("tex:252", "'at parity with the Hebbian write at two, four and eight streams on screens'", "parity",
    f"readings: (a) {ra80}, (b) {rb80}, S76 {r76}; counts 10/10 v 10/10, 10/10 v 10/10, {cnt(br, DS)}/10 v {cnt(br, HS)}/10", "PARTIAL",
    "'parity' is not a reading: the readings are 'keeps it' (c - b <= 1), which tex:248 itself says is 'not equivalence at the main "
    "line's power' (report 6 'Caveat')")
row("tex:252", "'binds the randomised header layout 14/20 against 1/20 for the window baseline'", "14/20 vs 1/20",
    f"session G's counts (tex:298); F's own window baseline on S70 seeds 330-349: delta {cnt(bound, RHD)}/20 vs Hebbian {cnt(bound, RHH)}/20 bound, "
    f"ROUTED*@end {cnt(routed_star_end, RHD)}/20 and {cnt(routed_star_end, RHH)}/20", "NOT RECORD-CHECKABLE",
    "session G's records, outside this scope; note both 14/20 and 1/20 are bound-but-unrouted (ROUTED-BOUND 0/20 each, tex:298)")
row("tex:252", "'tolerates an unrouted first layer (S71)'", "(claim)", "session G's S71", "NOT RECORD-CHECKABLE", "outside F's records")
row("tex:252", "'still needs the partition at four and eight streams (S68)'", "needs the partition",
    f"one delta channel {k4}/10 and {k8}/10 vs delta perfect gates {cnt(bound, O4)}/2 and {cnt(bound, O8)}/2; S68 reading: {r68}", "PARTIAL",
    "the counts support it descriptively, but it is not S68's reading (S68 read 'neither'; no S68 reading says 'needs the partition'), "
    "and it is not marked as an interpretation")
div = [k for k, r in EI["runs"].items() if not r.get("ok")]
row("tex:252", "'Raw keys are excluded by their numerical fragility (beta||x||^2 > 2 makes I - beta x x^T expanding)'", "fragility",
    f"raw-key oracle runs: {len(E1['O_B1_RAW']) + len(E1['O_B025_RAW'])}; diverged {div}; bound {cnt(bound, E1['O_B1_RAW']) + cnt(bound, E1['O_B025_RAW'])}",
    "NOT RECORD-CHECKABLE", "the eigenvalue statement is algebra (1 - beta||x||^2 along x); that it caused O_B1_RAW|1300's divergence is "
    "report 5's post hoc reading (report_5.md:62-63), unmarked here")
row("tex:252", "'the trigger on delta channels is the subject of F's S81'", "S81 = trigger", "explore_f_trigger.py: 'screen S81: the split's trigger on delta memory'",
    "MATCH", "docstring")
row("tex:314", "'The delta write with a fixed write strength is interchangeable with the Hebbian one on the grouped layout at every size tried'",
    "interchangeable", f"readings 'keeps it' at 2, 4 (S80) and 8 (S76; {cnt(br, DS)}/10 vs {cnt(br, HS)}/10, one delta run MERGED)", "PARTIAL",
    "stronger than any reading: the readings are one-sided non-inferiority bounds at ceiling (report 6's caveat), and at eight streams "
    "the delta arm lost one pair; unmarked")
row("tex:314", "'one delta channel binds the two-stream layouts ... and fails at four and eight streams'", "binds two-stream; fails at 4 and 8",
    f"two-stream: header {cnt(bound, S53H)}/10, grouped {cnt(bound, S53G)}/10; four {k4}/10, eight {k8}/10", "PARTIAL",
    "'binds the two-stream layouts' holds for the header layout (8/10) but on the grouped layout it is a minority (4/10); "
    "'by routes that need no partition' is interpretation, unmarked")
row("tex:349", "'the delta write matches the Hebbian one at two, four and eight streams and binds sooner'", "matches; binds sooner",
    f"sooner: two {soonA}/10, four {soonB}/10 (tie at 3600 on every seed), eight {len(soon76)}/{len(both76)}", "PARTIAL",
    "not sooner at four streams; 'matches' is stronger than the 'keeps it' readings (see tex:252)")
row("tex:349", "'the stall Revision 8 reported came from a screen with a learned write strength'", "learned beta in S54(c)",
    f"C_DELTA delta_cfg beta = {c54d['delta_cfg']['beta']!r}", "MATCH",
    "true as a description; read as a cause it is the post hoc attribution of tex:246 (the split and seeds also differ)")
row("tex:351", "'the partition's claim is about the number of contexts' Supported: one delta channel 2/10 at four, 0/10 at eight (S68)",
    "2/10, 0/10; Supported", f"{k4}/10, {k8}/10; S68 reading {r68}", "PARTIAL",
    "the counts match; S68's pre-fixed reading of that name ('the partition's advantage is in the number of streams', "
    "explore_f_premise.py:23) did NOT apply (neither) — the verdict rests on the C3 test, not on S68")
row("tex:353", "stream-tagged keys at layer 3 did not hold up", "1 of 16 bound runs", f"{len(tagged)} of {len(bdelta)}",
    eq("1 of 16", f"{len(tagged)} of {len(bdelta)}"))
row("tex:353", "'the context reaches depth on delta memory only in the solutions that bind without routing'", "only unrouted solutions",
    f"S80(a) DELTA_A (DISCOVERED {kA[0]}/10, i.e. routed) carries the stream at KEY positions at L3 (median {f2(med(l3k))})", "PARTIAL",
    "conflicts with tex:250's own S80 finding unless the adjacency reading (post hoc) is applied; the 'only' rests on session G's probe")
row("tex:378", "single delta channel binds the two-stream header layout", "8/10", f"{cnt(bound, S53H)}/10", eq("8/10", f"{cnt(bound, S53H)}/10"))
gaps = {}
for d in ("ROUTED", "GDN"):
    gaps[d] = [(O59[d][s]["acc"] - S59[d][s]["acc"]) - (O59["FIXED"][s]["acc"] - S59["FIXED"][s]["acc"]) for s in sorted(S59[d])]
r59 = "the horizon knob matters" if any(sum(1 for g in v if g >= 0.15) >= 7 for v in gaps.values()) else "it does not"
row("tex:378", "a learned forget gate 'shrank' the oracle-vs-one-channel gap (S59)", "shrank it",
    "gap_d - gap_FIXED < 0 on " + ", ".join(f"{d} {sum(1 for g in v if g < 0)}/10 (median {med(v):+.2f})" for d, v in gaps.items())
    + f"; S59 reading: {r59}", "MATCH" if all(med(v) < 0 for v in gaps.values()) and r59 == "it does not" else "DISAGREE",
    "gap = final acc(ORACLE_d) - acc(SINGLE_d) per seed (explore_f_decay.py:30-32)")
row("tex:378", "'a single Hebbian channel's stream tag ... is not linearly readable'", "not readable",
    f"S69_GDN L3 K/V medians {f2(med([pv(r, 'L3|K') for r in PA['S69_GDN'].values()]))}/{f2(med([pv(r, 'L3|V') for r in PA['S69_GDN'].values()]))}; "
    f"runs >= 0.9 at L3 K and V: {sum(1 for r in PA['S69_GDN'].values() if pv(r, 'L3|K') >= 0.9 and pv(r, 'L3|V') >= 0.9)}", "MATCH")
row("tex:378", "the delta recipe did not stay behind at eight streams ('at parity with the write strength fixed')", "parity",
    f"{cnt(br, DS)}/10 vs {cnt(br, HS)}/10; reading {r76}", "MATCH", "the reading 'keeps eight streams' applies; 'parity' as at tex:252")
DOH, DOD = runs(WD, "D_ORACLE_HEBB"), runs(WD, "D_ORACLE_DELTA")
row("tex:386", "F's first randomised header layout (S54(d), P = 8) did not bind under the perfect gate within its budget",
    "did not bind", f"D_ORACLE_HEBB {cnt(bound, DOH)}/2, D_ORACLE_DELTA {cnt(bound, DOD)}/2 within {sorted({r['iters'] for r in DOH.values()})[0]}",
    "MATCH" if cnt(bound, DOH) + cnt(bound, DOD) == 0 else "DISAGREE", "explore_f_rh_lite.py:3-4 (S54(d) UNTESTED)")
row("tex:386", "the lighter version (four keys per stream, two blocks) did", "did", f"S70 oracles {cnt(bound, OH70)}/3 and {cnt(bound, OD70)}/3",
    "MATCH" if cnt(bound, OH70) == 3 and cnt(bound, OD70) == 3 else "DISAGREE")

# ══ tex:278-283 — synthesis table, delta columns ════════════════════════════
row("tex:278", "S=2 'the same on delta channels' (S80(a), delta vs Hebbian)", "10/10 vs 10/10", f"{kA[0]}/10 vs {kA[1]}/10",
    eq("10/10 vs 10/10", f"{kA[0]}/10 vs {kA[1]}/10"), "DISCOVERED, as the caption's 'discovered at S = 2'")
row("tex:278", "S=2 one delta channel (S53 SINGLE_DELTA_G, no conv)", "4/10^f", f"{cnt(bound, S53G)}/10", eq("4/10", f"{cnt(bound, S53G)}/10"))
row("tex:279", "S=4 delta column (S80(b))", "10/10 vs 10/10", f"{kB[0]}/10 vs {kB[1]}/10", eq("10/10 vs 10/10", f"{kB[0]}/10 vs {kB[1]}/10"))
row("tex:279", "S=4 one delta channel (S68_SINGLE_S4, conv)", "2/10^f", f"{k4}/10", eq("2/10", f"{k4}/10"))
row("tex:280", "S=8 delta column (S76 Phase 2)", "9/10 vs 10/10", f"{cnt(br, DS)}/10 vs {cnt(br, HS)}/10", eq("9/10 vs 10/10", f"{cnt(br, DS)}/10 vs {cnt(br, HS)}/10"))
row("tex:280", "S=8 one delta channel, F (S68_SINGLE_S8)", "0/10^f", f"{k8}/10", eq("0/10", f"{k8}/10"), "the '(X)' entry is the main line's C3, outside this check")
row("tex:283", "footnote d: 'S80 and S76 ... ten paired seeds each, ... beta = 1, L2 keys and the exact-copy split'", "exact-copy split in both",
    "S80(a) has NO split (k = S = 2; explore_f_configs.py (a): 'NO SPLIT (k = S)'); S80(b) and S76 Phase 2 use H's exact copy", "PARTIAL",
    "the S = 2 cell has no split; seeds: S80(a) 2000-2009, (b) 2010-2019, S76 1310-1319, ten paired each (MATCH)")
row("tex:283", "footnote f: 'S53 and S68, exploratory, beta = 1 at the fixed decay'", "beta 1, fixed decay",
    f"S53 SINGLE_DELTA_G {dcfg['S53 SINGLE_DELTA_G/H']}; S68 {dcfg['S68']}", "MATCH",
    "S53 grouped is no conv (explore_f_delta_controls.py:14), S68 has conv (explore_f_premise.py:9), as the rows say")

# ══ tex:306 — S70 ═══════════════════════════════════════════════════════════
row("tex:306", "F's window baseline (S70, seeds 330-349): routed on either memory; bound", "0/20 routed; 1/20 and 6/20 bound",
    f"seeds {min(RHH)}-{max(RHH)}; ROUTED*@end {cnt(routed_star_end, RHH)}/20 and {cnt(routed_star_end, RHD)}/20; BOUND {cnt(bound, RHH)}/20 and {cnt(bound, RHD)}/20",
    "MATCH" if (cnt(routed_star_end, RHH), cnt(routed_star_end, RHD), cnt(bound, RHH), cnt(bound, RHD)) == (0, 0, 1, 6) else "DISAGREE",
    "F's 'routed' is ROUTED*@end (explore_f_rh_lite.py:19), not G's ROUTED-BOUND used in the table's other rows; DISCOVERED 0/20 both")

# ══ tex:501 — provenance ════════════════════════════════════════════════════
subj = {h: git("log", "-1", "--format=%s", h).strip() for h in (R3, R4, R5, R6)}
row("tex:501", "report commits: report 3 9a6c2bf, report 4 22ec912, report 5 5b9bc2a, report 6 ed76eea", "as listed",
    "; ".join(f"{h}: '{s[:60]}'" for h, s in subj.items()), "MATCH" if all(f"report {i}" in subj[h] for i, h in zip((3, 4, 5, 6), (R3, R4, R5, R6))) else "DISAGREE")
n = {k: sum(1 for r in v["runs"].values() if not r.get("dry")) for k, v in
     dict(dl=DL, wd=WD, ws=WS, dc=DC, pr=PR, hd=HD, rh=RH, pb=PB, ei=EI, ca=CA, cb=CB).items()}
counts = f"{n['dl'] + n['wd'] + n['ws'] + n['dc']}; {n['pr'] + n['hd'] + n['rh']}; {n['pb']}; {n['ei']}; {n['ca'] + n['cb']}"
row("tex:501", "runs per report: reports 1-2 (S53, S54, S59); 3; 4; 5; 6", "322; 102; 30; 40; 48", counts, eq("322; 102; 30; 40; 48", counts),
    f"S53 {n['dl']}, S54 {n['wd']} + {n['ws']}, S59 {n['dc']}; S68 {n['pr']}, S69 {n['hd']}, S70 {n['rh']}; stores at {HEAD} equal those at the "
    f"report commits: {all(v[1] for v in SAME.values())}")
# S81 at the time of each draft: the last F commit at or before each draft commit
def f_at(ref):
    t = git("log", "-1", "--format=%ct", ref).strip()
    return git("log", "-1", "--format=%h", f"--until={t}", BR).strip()[:7]


def s81_state(fref):
    has = git("ls-tree", "-r", "--name-only", fref, f"{F}trigger_results.json").strip()
    recs = load(fref, f"{F}trigger_results.json")["runs"] if has else {}
    log = git("show", f"{fref}:{F}s81.log") if git("ls-tree", "-r", "--name-only", fref, f"{F}s81.log").strip() else ""
    rep = [x for x in git("ls-tree", "-r", "--name-only", fref, F).split() if x.endswith("report_7.md")]
    return dict(ref=fref, date=cdate(fref), n=len(recs), arms=dict(Counter(k.split("|")[0] for k in recs)),
                done="=== S81 done" in log, started="=== S81 start" in log, report=bool(rep),
                outcomes=dict(Counter(r.get("outcome") for r in recs.values())))


S81_OLD, S81_NEW, S81_HEAD = s81_state(f_at(DRAFT_REF)), s81_state(f_at(NEW_REF)), s81_state(HEAD)
row("tex:501", "S81 running", "running",
    f"first draft ({DRAFT_REF}): F at {S81_OLD['ref']} has {S81_OLD['n']} S81 records; second draft ({NEW_REF}, {cdate(NEW_REF)}): F at "
    f"{S81_NEW['ref']} ({S81_NEW['date']}) has {S81_NEW['n']} records {S81_NEW['arms']}, log 'S81 done' {S81_NEW['done']}, report 7 {S81_NEW['report']}",
    "MATCH" if S81_NEW["started"] and not S81_NEW["done"] and not S81_NEW["report"] else "DISAGREE",
    "no S81 number is quoted in either draft; the S81 store is progress records, not a result")
row("tex:501", "branch commit", "ed76eea", f"report 6 commit ed76eea; F head {HEAD} adds only S81 logs, checks and progress records", "MATCH",
    f"commits after ed76eea up to {HEAD}: {len(git('log', '--format=%h', f'ed76eea..{HEAD}').split())}; no report after report 6: "
    f"{not S81_HEAD['report']}")

# ══ Report 6 (S80) — every number, checked against the records ═════════════
R6F = "report_6.md"
RA = {a: runs(CA, a) for a in ("HEBB_A", "DELTA_A", "O_DELTA_A", "O_HEBB_A")}
RB = {a: runs(CB, a) for a in ("HEBB_B", "DELTA_B", "O_DELTA_B", "O_HEBB_B")}
allS80 = list(CA["runs"].values()) + list(CB["runs"].values())
row(f"{R6F}:3-4", "code SHAs (a) / (b)", "d7c6a0326eae / 678c244056e9",
    f"{sorted({r['code_sha'][:12] for r in CA['runs'].values()})} / {sorted({r['code_sha'][:12] for r in CB['runs'].values()})}",
    "MATCH" if {r['code_sha'][:12] for r in CA['runs'].values()} == {"d7c6a0326eae"} and {r['code_sha'][:12] for r in CB['runs'].values()} == {"678c244056e9"} else "DISAGREE")
row(f"{R6F}:4-5", "main-line modules at 9c5939e; CPU; torch", "9c5939e; Xeon 2.10GHz; 2.14.0+cu130",
    f"{sorted({r.get('main_sha', '')[:7] for r in CB['runs'].values()})}; {sorted({r['cpu'] for r in allS80})}; {sorted({r['torch'] for r in allS80})}", "MATCH")
row(f"{R6F}:5", "48/48 runs ok, none retried", "48/48, attempts 1",
    f"{sum(1 for r in allS80 if r.get('ok'))}/{len(allS80)}; attempts {sorted({r['attempts'] for r in allS80})}",
    "MATCH" if sum(1 for r in allS80 if r.get("ok")) == 48 == len(allS80) and {r["attempts"] for r in allS80} == {1} else "DISAGREE")
row(f"{R6F}:5", "DELTA* = beta 1, L2 keys", "beta 1, L2", f"(a) {dcfg['S80 DELTA_A']}; (b) {dB}; eight_delta_star.json arm {star}", "MATCH")
# seed-block scan over every remote branch's results/ and explore_out/ JSON
blobs = {}
for b in [x.strip() for x in git("branch", "-r").split("\n") if x.strip() and "->" not in x]:
    for line in git("ls-tree", "-r", b, "--", "results", "explore_out").split("\n"):
        if line.strip() and line.endswith(".json"):
            meta, path = line.split("\t", 1)
            blobs.setdefault(meta.split()[2], []).append((b, path))


def scan(o, lo, hi, acc):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(k, str) and "|" in k:
                p = k.split("|")
                if p[1].isdigit() and lo <= int(p[1]) <= hi:
                    acc.add(int(p[1]))
            if k == "seed" and isinstance(v, int) and lo <= v <= hi:
                acc.add(v)
            scan(v, lo, hi, acc)
    elif isinstance(o, list):
        for x in o:
            scan(x, lo, hi, acc)


hits20, hits13 = {}, {}
for blob, where in blobs.items():
    try:
        d = json.loads(git("cat-file", "-p", blob))
    except ValueError:
        continue
    a, c = set(), set()
    scan(d, 2000, 2019, a)
    scan(d, 1300, 1359, c)
    if a:
        hits20[blob] = (where, sorted(a))
    if c:
        hits13[blob] = (where, sorted(c))
w20 = sorted({p for wh, _ in hits20.values() for _, p in wh})
row(f"{R6F}:5-6", "seeds 2000-2019: the block appears in no record on any branch", "no other record",
    f"seeds 2000-2019 appear only in {w20} across {len(blobs)} JSON blobs on {len(set(b for v in blobs.values() for b, _ in v))} remote branches",
    "MATCH" if set(w20) <= {f"{F}configs_a_results.json", f"{F}configs_b_results.json"} else "DISAGREE",
    "scan at the current heads; keys 'ARM|seed|...' and 'seed' fields. Second pass: the scan is narrowed to 2000-2019 (the block the claim names); the first pass scanned 2000-2059, which now also catches F's own S81 (trigger_results.json, seeds from 2020), run after report 6")
w13 = sorted({(b.split("/")[-1], p, min(s), max(s)) for wh, s in hits13.values() for b, p in wh})
row(f"{R6F}:100-101", "the 1300-1359 block is also used by Session H's S78-S79 (explore_h_two, seeds 1300-1329)", "H 1300-1329",
    f"{[x for x in w13 if 'explore-F' not in x[0]]}", "PARTIAL",
    "H uses 1300-1329, but explore_h_two's store (two_results.json) holds 1300-1319 and 1320-1329 are in constants_results.json; "
    "record keys only; F's own S76 store is " + str([x for x in w13 if "explore-F" in x[0]][:1]))


def tstr(rs, f):
    t = sorted(tr(r) for r in rs.values() if f(r))
    c = Counter(t)
    return f"{med(t):g} ({', '.join(f'{v} x{c[v]}' if c[v] > 1 else str(v) for v in sorted(c))})"


for a, f, lab, rep_t in (("HEBB_A", discovered, "DISCOVERED", "2400 (2400 x6, 3600 x3, 6000)"), ("DELTA_A", discovered, "DISCOVERED", "1200 (1200 x10)")):
    rs = RA[a]
    k = cnt(f, rs)
    row(f"{R6F}:{12 if a == 'HEBB_A' else 13}", f"{a} {lab} count, Wilson, band", "10/10 [0.72, 1.00] RELIABLE", f"{k}/{len(rs)} {wil(k, len(rs))} {band(k, len(rs))}",
        eq("10/10 [0.72, 1.00] RELIABLE", f"{k}/{len(rs)} {wil(k, len(rs))} {band(k, len(rs))}"))
    row(f"{R6F}:{12 if a == 'HEBB_A' else 13}", f"{a} transitions (median; all)", rep_t, tstr(rs, f), eq(rep_t, tstr(rs, f)))
for a, rep_t in (("HEBB_B", "3600 (3600 x10)"), ("DELTA_B", "3600 (3600 x10)")):
    rs = RB[a]
    k = cnt(br, rs)
    row(f"{R6F}:{14 if a == 'HEBB_B' else 15}", f"{a} BOUND ROUTED count, Wilson, band", "10/10 [0.72, 1.00] RELIABLE", f"{k}/{len(rs)} {wil(k, len(rs))} {band(k, len(rs))}",
        eq("10/10 [0.72, 1.00] RELIABLE", f"{k}/{len(rs)} {wil(k, len(rs))} {band(k, len(rs))}"))
    row(f"{R6F}:{14 if a == 'HEBB_B' else 15}", f"{a} transitions (median; all)", rep_t, tstr(rs, br), eq(rep_t, tstr(rs, br)))
vt = {a: [tr(rs[s]) for s in sorted(rs)] for a, rs in (("O_DELTA_A", ODA), ("O_DELTA_B", ODB), ("O_HEBB_A", OHA), ("O_HEBB_B", OHB))}
row(f"{R6F}:17-18", "validity: delta perfect gate bound on both seeds, at 1200 every time (O_DELTA_A 2000-2001, O_DELTA_B 2010-2011)",
    "1200, 1200 / 1200, 1200", f"{vt['O_DELTA_A']} (seeds {sorted(ODA)}) / {vt['O_DELTA_B']} (seeds {sorted(ODB)})",
    "MATCH" if vt["O_DELTA_A"] == vt["O_DELTA_B"] == [1200, 1200] else "DISAGREE")
row(f"{R6F}:18", "Hebbian perfect gate (a) / (b)", "2400 and 1200 / 1200 and 1200", f"{vt['O_HEBB_A']} / {vt['O_HEBB_B']}",
    "MATCH" if vt["O_HEBB_A"] == [2400, 1200] and vt["O_HEBB_B"] == [1200, 1200] else "DISAGREE")
row(f"{R6F}:20", "paired: delta only, Hebbian only (a) / (b); one-sided McNemar both ways", "0, 0 / 0, 0; p = 1",
    f"{ba80}, {ca80} / {bb80}, {cb80}; p = {mcnemar1(ba80, ca80):g}, {mcnemar1(ca80, ba80):g}, {mcnemar1(bb80, cb80):g}, {mcnemar1(cb80, bb80):g}",
    "MATCH" if (ba80, ca80, bb80, cb80) == (0, 0, 0, 0) else "DISAGREE")
row(f"{R6F}:24-25", "readings (a) and (b) with c - b", "keeps it (c - b = 0 <= 1), both", f"(a) {ra80}, c - b = {ca80 - ba80}; (b) {rb80}, c - b = {cb80 - bb80}",
    "MATCH" if ra80 == rb80 == "the delta recipe keeps it" and ca80 - ba80 == 0 == cb80 - bb80 else "DISAGREE")
row(f"{R6F}:24-25", "main-line references 69/80 and 38/40", "69/80; 38/40", "explore_f_configs.py REF = ('WIN3_SLOW', 69, 80); docstring (b) 38/40",
    "NOT RECORD-CHECKABLE", "main-line counts, not F's records (the synthesis table's own rows, tex:278-279)")
row(f"{R6F}:27", "a Hebbian arm loses about 1.4 and 0.5 runs in 10", "1.4 and 0.5", f"{loss[0]:.3f} and {loss[1]:.2f}", "MATCH", "arithmetic")
row(f"{R6F}:33-34", "(a) delta sooner on 10 of 10 pairs; the delta perfect gate also at 1200", "10 of 10; 1200", f"{soonA} of 10; {vt['O_DELTA_A']}",
    "MATCH" if soonA == 10 and vt["O_DELTA_A"] == [1200, 1200] else "DISAGREE")


def acc_at(r, step):
    return {s: a for s, a, *_ in r["curve"]}.get(step)


a24 = [(acc_at(DB[s], 2400), acc_at(HB[s], 2400)) for s in sorted(DB)]
ahead = sum(1 for d, h in a24 if d > h)
row(f"{R6F}:35-36", "(b) at the 2400 evaluation delta ahead on 9 of 10 pairs (median accuracy 0.56 vs 0.32)", "9 of 10 (0.56 vs 0.32)",
    f"{ahead} of 10 ({f2(med([d for d, _ in a24]))} vs {f2(med([h for _, h in a24]))})",
    eq("9 of 10 (0.56 vs 0.32)", f"{ahead} of 10 ({f2(med([d for d, _ in a24]))} vs {f2(med([h for _, h in a24]))})"))
first_chk = sorted({c["step"] for r in list(DB.values()) + list(HB.values()) for c in r.get("checks", []) if c.get("step", 0) >= 4800})[:1]
row(f"{R6F}:37", "(b) no split fired in either arm; every run bound before the first eligible check at 4800", "none; bound before 4800",
    f"{nfire} firings; transitions {tB}; first check >= 4800 at {first_chk}", "MATCH" if nfire == 0 and max(tB) < 4800 else "DISAGREE")
two_k2 = multi = 0
vdiff = []
for f_ in [x for x in git("ls-tree", "-r", "--name-only", R6, F).split() if x.endswith("_results.json")]:
    for kk, r in load(R6, f_)["runs"].items():
        e = r.get("end") or {}
        if "etak_key_by_key" in e:
            multi += 1
            ek = e["etak_key_by_key"]
        elif "eta_key_by_key" in e and "margin" in e and r.get("k") == 2:
            two_k2 += 1
            ek = e["eta_key_by_key"]
        else:
            continue
        if e["margin"] >= 0.25 and ek >= 0.5:      # v1: STREAM-PARTIAL first; v2: KEY first -> they differ only here
            vdiff.append(f"{f_.split('/')[-1]} {kk.split('|')[0]}|{kk.split('|')[1]} ({ek:.3f}, {e['margin']:.3f}, bound {bound(r)})")
row(f"{R6F}:44-46", "classifier v1/v2 over the stored Session F records at report time: 367 (236 two-stream, 131 multichannel); "
    "v2 != v1 only on B_HEBB|323 (eta^2 by key 0.676, margin 0.362)", "367 (236, 131); B_HEBB|323 0.676 / 0.362",
    f"{two_k2 + multi} ({two_k2}, {multi}); {vdiff}",
    "MATCH" if (two_k2, multi) == (236, 131) and len(vdiff) == 1 and "B_HEBB|323 (0.676, 0.362" in vdiff[0] else "DISAGREE",
    f"recount at {R6}: k = 2 records with end eta/margin (S68's k = 4/8 oracles excluded) and records with etak_* (k = 16); v1 and v2 "
    "can differ only where margin >= 0.25 and eta^2 by key >= 0.5 (explore_f_configs.py FAILURE CLASSES)")
row(f"{R6F}:63", "the delta perfect gate shows the same pattern (KEY high, VAL not)", "same pattern",
    f"O_DELTA_A L3 K {[f2(x) for x in pA['O_DELTA_A']['L3|K']]}, L3 V {[f2(x) for x in pA['O_DELTA_A']['L3|V']]}",
    "MATCH" if min(pA["O_DELTA_A"]["L3|K"]) >= 0.9 and max(pA["O_DELTA_A"]["L3|V"]) < 0.9 else "DISAGREE", "2 runs")
PM = {"HEBB_A": HA, "DELTA_A": DA, "O_DELTA_A": ODA}
REP6 = {"HEBB_A": ["0.50 / 0.50", "0.56 / 0.51", "0.57 / 0.52", "0.57 / 0.56", "0.79 / 0.68"],
        "DELTA_A": ["0.50 / 0.50", "0.99 / 0.69", "1.00 / 0.58", "1.00 / 0.62", "0.95 / 0.72"],
        "O_DELTA_A": ["0.50 / 0.50", "0.91 / 0.55", "0.98 / 0.52", "0.99 / 0.58", "0.98 / 0.81"]}
for a, rs in PM.items():
    got = []
    for L, init in (("L1", False), ("L2", False), ("L3", False), ("final", False), ("L3", True)):
        got.append(" / ".join(f2(med([pv(r, f"{L}|{x}", init) for r in rs.values()])) for x in "KV"))
    for (lab, rep, g) in zip(("entering L1", "entering L2", "entering L3", "final", "L3 at init"), REP6[a], got):
        row(f"{R6F}:55-57", f"probe median K / V, {a}, {lab}", rep, g, eq(rep, g))
hl3 = pA["HEBB_A"]["L3|K"]
row(f"{R6F}:59-60", "KEY: DELTA_A readable from layer 2 on (L3 0.98-1.00); HEBB_A near chance (0.52-0.60)", "0.98-1.00; 0.52-0.60",
    f"DELTA_A L3 K {f2(min(l3k))}-{f2(max(l3k))}; HEBB_A L3 K {f2(min(hl3))}-{f2(max(hl3))}",
    eq("0.98-1.00; 0.52-0.60", f"{f2(min(l3k))}-{f2(max(l3k))}; {f2(min(hl3))}-{f2(max(hl3))}"))
row(f"{R6F}:61", "VAL: DELTA_A 0.54-0.65 at L3", "0.54-0.65", f"{f2(min(l3v))}-{f2(max(l3v))}", eq("0.54-0.65", f"{f2(min(l3v))}-{f2(max(l3v))}"))
crit = {a: sum(1 for r in rs.values() if pv(r, "L3|K") >= 0.9 and pv(r, "L3|V") >= 0.9) for a, rs in (("HEBB_A", HA), ("DELTA_A", DA))}
row(f"{R6F}:62", "S77's criterion (L3 >= 0.9 at KEY and VAL) holds on 0 of 10 runs in each arm", "0 and 0", f"{crit['HEBB_A']} and {crit['DELTA_A']}",
    eq("0 and 0", f"{crit['HEBB_A']} and {crit['DELTA_A']}"))
row(f"{R6F}:64", "training raises KEY on delta (0.95 at init -> 1.00), lowers it on Hebbian (0.79 -> 0.57)", "0.95 -> 1.00; 0.79 -> 0.57",
    f"{f2(med([pv(r, 'L3|K', True) for r in DA.values()]))} -> {f2(med(l3k))}; {f2(med([pv(r, 'L3|K', True) for r in HA.values()]))} -> {f2(med(hl3))}",
    eq("0.95 -> 1.00; 0.79 -> 0.57", f"{f2(med([pv(r, 'L3|K', True) for r in DA.values()]))} -> {f2(med(l3k))}; {f2(med([pv(r, 'L3|K', True) for r in HA.values()]))} -> {f2(med(hl3))}"),
    "L3 K medians")
row(f"{R6F}:67", "S77's header layouts: L3 0.67 at KEY (S53_DELTA median)", "0.67", f2(m3[("S53_DELTA", "K")]), eq("0.67", f2(m3[("S53_DELTA", "K")])),
    f"S59_GDN's is {f2(m3[('S59_GDN', 'K')])}")
row(f"{R6F}:72-88", "CHECKs: explore_f_delta_checks 26/26 and the screen CHECKs", "all passed",
    f"checks file ok = {CHK['ok']}, {sum(1 for _, o in CHK['rows'] if o)}/{len(CHK['rows'])} rows true", "NOT RECORD-CHECKABLE",
    "checks file and s80.log, not run records (the file's git field is ebb381c)")
ha_ = sum(r["secs_wall"] for r in CA["runs"].values()) / 3600
hb_ = sum(r["secs_wall"] for r in CB["runs"].values()) / 3600
row(f"{R6F}:92", "run time (sum of secs_wall): total (a, b) h", "4.1 (0.55, 3.57)", f"{ha_ + hb_:.1f} ({ha_:.2f}, {hb_:.2f})",
    eq("4.1 (0.55, 3.57)", f"{ha_ + hb_:.1f} ({ha_:.2f}, {hb_:.2f})"))
row(f"{R6F}:92", "19:24-20:41 UTC; about 1.3 h wall on 4 workers", "19:24-20:41", "s80.log header 'S80 start Sat Oct 10 19:23:56 UTC 2026'; "
    f"last record commit 839bddb {cdate('839bddb')}", "NOT RECORD-CHECKABLE", "log and history, not records")
sw = (statistics.mean(r["secs_wall"] for r in DB.values()), statistics.mean(r["secs_wall"] for r in HB.values()))
stops = sorted({r["stopped_at"] for r in list(DB.values()) + list(HB.values())})
row(f"{R6F}:99", "(b) delta about 3x the Hebbian wall time (about 810 s vs 267 s per run) for the same 8400 updates", "810 vs 267; 8400",
    f"mean secs_wall {sw[0]:.0f} vs {sw[1]:.0f} ({sw[0] / sw[1]:.1f}x); stopped_at {stops}", "MATCH" if abs(sw[0] - 810) < 15 and abs(sw[1] - 267) < 10 and stops == [8400] else "DISAGREE")
row(f"{R6F}:97", "40/40 learned-gate runs succeeded", "40/40", f"{kA[0] + kA[1] + kB[0] + kB[1]}/40", eq("40/40", f"{kA[0] + kA[1] + kB[0] + kB[1]}/40"))

# ══ readings commits vs first run record ════════════════════════════════════
def first_reading_commit(path, text):
    out = [o.split() for o in git("log", "--format=%h %ct", f"-S{text}", BR, "--", path).strip().split("\n") if o]
    out.sort(key=lambda x: int(x[1]))
    return out[0][0][:7] if out else None


def first_record_commit(paths, arms):
    hist = []
    for p in paths:
        for h in git("log", "--reverse", "--format=%h %ct", BR, "--", p).split("\n"):
            if h.strip():
                hist.append((int(h.split()[1]), h.split()[0], p))
    for _, h, p in sorted(hist):
        try:
            s = json.loads(git("show", f"{h}:{p}"))
        except (ValueError, subprocess.CalledProcessError):
            continue
        k = sum(1 for kk, r in s.get("runs", {}).items() if kk.split("|")[0] in arms and not r.get("dry"))
        if k:
            return h[:7], k
    return None, 0


SCREENS = [
    ("S68", "explore_f_premise.py", '"context-tagged keys scale"', [f"{F}premise_results.json"], ["S68_SINGLE_S4", "S68_SINGLE_S8", "S68_ORACLE_S4", "S68_ORACLE_S8"], [PR]),
    ("S69", "explore_f_hebb_decay.py", '"the forget gate needs the delta rule"', [f"{F}hebb_decay_results.json"], ["S69_FIXED", "S69_ROUTED", "S69_GDN", "S69_ORACLE"], [HD]),
    ("S70", "explore_f_rh_lite.py", "the BASELINE for Session G is S70_HEBB's DISCOVERED count", [f"{F}rh_lite_results.json"],
     ["S70_ORACLE_HEBB", "S70_ORACLE_DELTA", "S70_HEBB", "S70_DELTA"], [RH]),
    ("S77", "explore_f_probe.py", '"the delta stack tags its keys by stream"', [f"{F}probe_results.json"], ["S53_DELTA", "S59_GDN", "S69_GDN"], [PB]),
    ("S76", "explore_f_eight.py", '"the delta recipe keeps eight streams"', [f"{F}eight_results.json"], list(E1) + ["HEBB_SPLIT", "DELTA_SPLIT"], [EI]),
    ("S80", "explore_f_configs.py", '"the delta recipe keeps it"', [f"{F}configs_a_results.json", f"{F}configs_b_results.json"],
     ["HEBB_A", "DELTA_A", "O_DELTA_A", "O_HEBB_A", "HEBB_B", "DELTA_B", "O_DELTA_B", "O_HEBB_B"], [CA, CB]),
    ("S81", "explore_f_trigger.py", '"the guard aims the split"', [f"{F}trigger_results.json"], ["BASE", "GUARD", "EVAL", "GUARD_EVAL", "O_DELTA_C"], []),
]
PREC = []
for scr, path, text, sps, arms, stores in SCREENS:
    rc = first_reading_commit(path, text)
    fr, nrec = first_record_commit(sps, arms)
    unchanged = git("diff", rc, HEAD, "--", path).strip() == "" if rc else False
    rg = sorted({r["git"].split("+")[0] for s_ in stores for r in s_["runs"].values() if r["arm"] in arms and r.get("git")})
    desc = all(is_anc(rc, g) for g in rg) if rc else False
    prec = bool(rc and fr and int(git("log", "-1", "--format=%ct", rc)) < int(git("log", "-1", "--format=%ct", fr)) and is_anc(rc, fr))
    PREC.append(dict(scr=scr, path=path, text=text, rc=rc, rcd=cdate(rc) if rc else "--", fr=fr, frd=cdate(fr) if fr else "--",
                     nrec=nrec, unchanged=unchanged, rg=rg, desc=desc, prec=prec))

# ══ second pass: locate the first pass's rows in the second draft ═══════════
def norm(s):
    return " ".join(s.split())


def find(lines, q):
    """1-based line numbers whose text holds every part of q (a string or a tuple of strings)."""
    parts = (q,) if isinstance(q, str) else q
    parts = [norm(p) for p in parts]
    return [i + 1 for i, ln in enumerate(lines) if all(p in norm(ln) for p in parts)]


# (first-pass tex, item prefix, fragment of the FIRST draft, the second-pass row that checks its replacement if removed)
QMAP = [
    ("tex:58", "one delta channel at four / eight", "one delta channel $2/10$, $0/10$ at four and eight streams", None),
    ("tex:58", "the delta recipe with fixed beta keeps", "the delta recipe with fixed $\\beta$ keeps 2, 4 and 8 streams", None),
    ("tex:96", "perfect gate (N = 256)", "$10/10$ in session~F)", "N2"),
    ("tex:108", "RandHeaderTask-lite: S = 2", "whose lengths are a uniformly drawn composition of 4 into parts of 1--3 pairs", None),
    ("tex:108", "RandHeaderTask-lite length L", "$L = 23$", None),
    ("tex:108", "'the cue is one to six tokens", "the cue is one to six tokens from the write it governs", "N3"),
    ("tex:108", "both perfect gates bind it by update 1200", "Both perfect gates bind it by update 1200 on five of six seeds", None),
    ("tex:108", "training budgets", "or 43{,}200 updates)", "N4"),
    ("tex:124", "the delta write: beta fixed at 1", ("with unit keys $\\hat{x}_t$, the decay fixed at $0.95$", "\\emph{fixed at 1}"), None),
    ("tex:124", "'at beta = 0 its logits agree", "agree with the Hebbian model's to $10^{-7}$", "N5"),
    ("tex:124", "'with the new path disabled", "reproduces a recorded run bit for bit.", "N6"),
    ("tex:124", "'A learned beta collapses", "A learned $\\beta$ collapses toward ``do not write''", "N7"),
    ("tex:246", "learned beta's 'collapse toward zero", "consistent with its collapse toward zero in S53 and S54(d)", None),
    ("tex:242", "S53 one delta channel, two-stream header", "binds the two-stream \\emph{header} layout $8/10$ (S53, $P = 4$)", None),
    ("tex:242", "S53 one delta channel, grouped", "and the grouped two-stream layout $4/10$", None),
    ("tex:242", "S68 seeds; updates", "(S68, seeds 300--309, 43{,}200 updates)", None),
    ("tex:242", "S68 one delta channel at four / eight", "one delta channel binds the grouped layout $2/10$ at four streams and $0/10$ at eight", None),
    ("tex:242", "both delta perfect gates", "while both delta perfect gates bind at 1200", None),
    ("tex:242", "gated Hebbian recipe on other seeds", "the gated Hebbian recipe binds $19/20$ and $9/10$ on other seeds", None),
    ("tex:242", "reading 'context-tagged keys scale'", "``context-tagged keys scale'' ($\\geq 7/10$ at eight)", None),
    ("tex:242", "reading 'the partition's advantage", "``the partition's advantage is in the number of streams'' ($\\geq 7/10$ at four)", "N8"),
    ("tex:242", "S68 readings (recomputed)", "both read ``neither''. The main line's", None),
    ("tex:244", "S59 single delta channel", "bound $9/10$ and $8/10$ against $3/10$ at the fixed decay (S59)", None),
    ("tex:244", "S59 median alpha at CTX, layer 1", "median $\\alpha$ of $0.18$--$0.26$ at CTX in the first layer", None),
    ("tex:244", "S59 median alpha at CTX, layer 3", "$0.72$--$0.74$ in the third", None),
    ("tex:244", "S59 median alpha at KEY and VAL", "$1.00$ at keys and values", None),
    ("tex:244", "S69 one Hebbian channel", "bound $0/10$, $0/10$ and $0/10$ (S69)", None),
    ("tex:244", "S69 median alpha at CTX", "median $\\alpha$ at CTX $0.975$ and $0.988$ in the first layer", None),
    ("tex:244", "S69 median final accuracy", "the learned decay lowered its final accuracy ($0.17$ against $0.22$)", "N9"),
    ("tex:244", "S69 reading (paragraph title)", "\\paragraph{The forget gate needs the delta rule.}", None),
    ("tex:246", "S54(c) window-gate recipe", "stalled at eight streams ($3/10$ against $9/10$, S54(c))", None),
    ("tex:246", "S54(c) delta perfect gate transitions", "took 22{,}800--28{,}800 updates", None),
    ("tex:246", "S76 Phase 1 seeds", "ran the perfect gate on four seeds", None),
    ("tex:246", "L2 keys, beta = 1 (O_B1_L2)", "binds all four at the first evaluation (1200) at $\\beta = 1$", None),
    ("tex:246", "L2 keys, beta = 0.25", "and at 2400 at $\\beta = 0.25$", None),
    ("tex:246", "Hebbian oracle (O_HEBB) transitions", ("Hebbian oracle", "(2400--6000"), None),
    ("tex:246", "'sooner than the Hebbian oracle'", "at $\\beta = 0.25$, sooner than the Hebbian oracle", "N10"),
    ("tex:246", "raw keys: one seed diverged", "with raw keys one seed diverged", "N11"),
    ("tex:246", "Phase 1 readings", "both read ``neither'', because no L2 arm was slow", None),
    ("tex:246", "Phase 2 (ten seeds)", "on delta channels $9/10$ bound routed against $10/10$ on Hebbian channels (1 vs 0)", None),
    ("tex:246", "sooner on eight of the nine", "sooner on eight of the nine seeds both bind (median 14{,}400 against 24{,}000)", None),
    ("tex:246", "Phase 2 reading", "the reading ``the delta recipe keeps eight streams'' applies", None),
    ("tex:246", "delta splits whose target", "Six of the twelve splits on delta channels", None),
    ("tex:246", "those six splits 'fired before", "Six of the twelve splits on delta channels fired before the map had merged", "N12"),
    ("tex:246", "'mostly as single splits", "mostly as single splits in runs that then bound", None),
    ("tex:246", "'S54(c) differed from S76 in two things", "S54(c) differed from S76 in two things", "N14"),
    ("tex:246", "learned-beta attribution", "that attribution is post hoc", None),
    ("tex:248", "S80(a) two streams", "at two streams ($k = 2$, no convolution) $10/10$ against $10/10$", None),
    ("tex:248", "delta at the first evaluation", "delta at the first evaluation on every seed against 2400--6000", None),
    ("tex:248", "S80(b) four streams", "at four streams ($k = 16$, with the convolution) $10/10$ against $10/10$", None),
    ("tex:248", "both at 3600 on every seed", "both at 3600 on every seed", None),
    ("tex:248", "no split firing in either arm", "no split firing in either arm", None),
    ("tex:248", "both readings", "Both readings are ``keeps it''", None),
    ("tex:248", "'at the main line's rates", "a Hebbian arm loses about $1.4$ and $0.5$ runs in ten", None),
    ("tex:248", "''keeps it' means", "``keeps it'' means that no seed failed on delta", None),
    ("tex:250", "S77: 30 recorded", "Reruns of 30 recorded single-channel runs with a linear probe at the end (S77; every curve bit for bit)", None),
    ("tex:250", "stream linearly readable at layer 3", "at layer~3 in one of sixteen bound delta runs", None),
    ("tex:250", "medians at layer 3 in the delta arms", "(medians $0.57$--$0.67$)", "N15"),
    ("tex:250", "the two S53 runs", "the two S53 runs that did carry it at value positions ($0.92$--$1.00$) were the two that failed to bind", None),
    ("tex:250", "the Hebbian channel (S69_GDN)", "the Hebbian channel carries nothing anywhere ($0.50$--$0.53$)", "N16"),
    ("tex:250", "S80: delta stack carries the stream at KEY", "($1.00$ at layer~3 on all ten runs", "N18"),
    ("tex:250", "S80: 'from layer 2 on'", "the delta stack does carry the stream at \\emph{key} positions, from layer~2 on", None),
    ("tex:250", "S80: Hebbian (HEBB_A)", "Hebbian $0.57$)", None),
    ("tex:250", "S80: not at value positions", "and not at value positions ($0.54$--$0.65$)", None),
    ("tex:250", "S80: 'in the perfect-gate arm", "in the perfect-gate arm as well", None),
    ("tex:250", "'session F reads this, post hoc", "session~F reads this, post hoc, as adjacency", None),
    ("tex:250", "'On the randomised header layout", "the stack carries the stream only in the solutions that bind without routing", "N19"),
    ("tex:252", "'binds sooner at two and eight'", ("What this changes", "binds sooner at two and eight"), None),
    ("tex:252", "'at parity with the Hebbian write", "is at parity with the Hebbian write at two, four and eight streams on screens", "N20"),
    ("tex:252", "'binds the randomised header layout", "binds the randomised header layout $14/20$ against $1/20$ for the window baseline", None),
    ("tex:252", "'tolerates an unrouted first layer", "tolerates an unrouted first layer (S71)", None),
    ("tex:252", "'still needs the partition", "still needs the partition at four and eight streams (S68)", "N21"),
    ("tex:252", "'Raw keys are excluded", "Raw keys are excluded by their numerical fragility", None),
    ("tex:252", "'the trigger on delta channels", "the trigger on delta channels is the subject of F's S81", None),
    ("tex:314", "'The delta write with a fixed write strength is interchangeable", "is interchangeable with the Hebbian one on the grouped layout at every size tried", "N23"),
    ("tex:314", "'one delta channel binds the two-stream layouts", "one delta channel binds the two-stream layouts by routes that need no partition and fails at four and eight streams", "N24"),
    ("tex:349", "'the delta write matches", "the delta write matches the Hebbian one at two, four and eight streams and binds sooner", "N27"),
    ("tex:349", "'the stall Revision 8 reported", "the stall Revision~8 reported came from a screen with a learned write strength", None),
    ("tex:351", "'the partition's claim is about the number of contexts' Supported", "\\textbf{Supported}: one delta channel $2/10$ at four streams", "N28"),
    ("tex:353", "stream-tagged keys at layer 3", "\\textbf{They did not} ($1$ of 16 bound runs", None),
    ("tex:353", "'the context reaches depth on delta memory only", "the context reaches depth on delta memory only in the solutions that bind without routing", "N29"),
    ("tex:378", "single delta channel binds the two-stream header", "(it binds the two-stream header layout $8/10$)", None),
    ("tex:378", "a learned forget gate 'shrank'", "(it shrank it, on the eight-key header layout)", None),
    ("tex:378", "'a single Hebbian channel's stream tag", "would be linearly readable (it is not)", None),
    ("tex:378", "the delta recipe did not stay behind", "(it is at parity with the write strength fixed)", "N31"),
    ("tex:386", "F's first randomised header layout", "Session~F's first randomised header layout did not bind under the perfect gate within its budget", None),
    ("tex:386", "the lighter version", "the lighter version with four keys per stream in two blocks did", None),
    ("tex:278", "S=2 'the same on delta channels'", ("$S{=}2$, $P{=}4$, no conv.", "& $10/10$ vs $10/10$ &"), None),
    ("tex:278", "S=2 one delta channel", ("$S{=}2$, $P{=}4$, no conv.", "$0/60^{c}$ & $4/10^{f}$"), None),
    ("tex:279", "S=4 delta column", ("$S{=}4$, $P{=}4$, $k{=}16$, conv.", "$39/40^{e}$ & $10/10$ vs $10/10$"), None),
    ("tex:279", "S=4 one delta channel", ("$S{=}4$, $P{=}4$, $k{=}16$, conv.", "$0/20$ & $2/10^{f}$"), None),
    ("tex:280", "S=8 delta column", ("$S{=}8$, $P{=}4$, $k{=}16$, conv.", "& $9/10$ vs $10/10$ &"), None),
    ("tex:280", "S=8 one delta channel, F", ("$S{=}8$, $P{=}4$, $k{=}16$, conv.", "$0/10$ (X), $0/10^{f}$"), None),
    ("tex:283", "footnote d", "L2 keys and the exact-copy split. $^{f}$", "N32"),
    ("tex:283", "footnote f", "$^{f}$Session~F's S53 and S68, exploratory, $\\beta = 1$ at the fixed decay", None),
    ("tex:306", "F's window baseline (S70", "session~F's window baseline on other seeds (S70, 330--349) is $0/20$ routed on either memory with $1/20$ and $6/20$ bound", None),
    ("tex:501", "report commits", ("S68 premise, S69 Hebbian decay, S70 lite task (report 3, \\texttt{9a6c2bf})", "(report 6, \\texttt{ed76eea})"), None),
    ("tex:501", "runs per report", ("S53, S54, S59 (reports 1--2)", "322; 102; 30; 40; 48"), None),
    ("tex:501", "S81 running", ("S53, S54, S59 (reports 1--2)", "S81 running"), None),
    ("tex:501", "branch commit", ("S53, S54, S59 (reports 1--2)", "& \\texttt{ed76eea} &"), None),
]

used = set()
for r in ROWS:
    r.update(kind="first", q=None, loc=None, sup=None)
    if not r["tex"].startswith("tex:"):
        r["loc"] = "report_6.md"            # report 6, not the draft: unaffected by the second draft
        continue
    hits = [i for i, (t, pre, _, _) in enumerate(QMAP) if t == r["tex"] and r["item"].startswith(pre)]
    assert len(hits) == 1, (r["tex"], r["item"], hits)
    used.add(hits[0])
    _, _, q, sup = QMAP[hits[0]]
    old_ln = int(r["tex"][4:].split()[0])
    assert old_ln in find(TEX, q), ("fragment not at the cited first-draft line", r["tex"], q, find(TEX, q))
    r["q"], r["sup"] = q, sup
    r["loc"] = find(TEX2, q) or "REMOVED"
assert used == set(range(len(QMAP))), sorted(set(range(len(QMAP))) - used)

# ══ second pass: rows for every new or changed sentence about Session F ═════
NROWS = []


def nrow(nid, tex2, item, draft, rec, status, note, q2):
    loc = find(TEX2, q2)
    assert tex2 in loc, ("fragment not at the cited second-draft line", nid, tex2, q2, loc)
    NROWS.append(dict(id=nid, tex=f"tex2:{tex2}", item=item, draft=str(draft), rec=str(rec), status=status, note=note,
                      q=q2, loc=loc, kind="second", old=find(TEX, q2)))


# tex2:36 abstract-level summary
nrow("N1", 36, "delta write 'not shown worse ... at two, four and eight streams (10/10, 10/10 and 9/10 against 10/10) and binds sooner at two and eight'",
     "10/10, 10/10, 9/10 vs 10/10; sooner at 2 and 8",
     f"{kA[0]}/10 v {kA[1]}/10 (S80a DISCOVERED), {kB[0]}/10 v {kB[1]}/10 (S80b), {cnt(br, DS)}/10 v {cnt(br, HS)}/10 (S76); sooner: two {soonA}/10, "
     f"four {soonB}/10, eight {len(soon76)}/{len(both76)}; readings {ra80} / {rb80} / {r76}",
     "MATCH" if (kA, kB, cnt(br, DS), cnt(br, HS), soonA, soonB, len(soon76)) == ((10, 10), (10, 10), 9, 10, 10, 0, 8) else "DISAGREE",
     "'not shown worse' is the readings' own meaning (c - b <= 1); C1 resolved here", ("is not shown worse than the Hebbian write at two, four and eight streams ($10/10$, $10/10$ and $9/10$ against $10/10$)", "binds sooner at two and eight"))
nrow("N2", 96, "perfect gate (N = 256), two-stream grouped, no conv: '14/14 in session F'", "14/14", f"Hebbian {bH}/{nH}; delta {bD}/{nD}",
     eq("14/14", f"{bH}/{nH}"), "S53 ORACLE_HEBB_G 10/10 + S54(a) A_ORACLE_HEBB 2/2 + S80(a) O_HEBB_A 2/2", "$14/14$ in session~F")
koff = sorted({2 * j + 1 for ln in range(1, 4) for j in range(ln)})
voff = sorted({2 * j + 2 for ln in range(1, 4) for j in range(ln)})
nrow("N3", 108, "RandHeaderTask-lite: 'a token is one to six tokens from its block's cue, the write at a value position two to six'",
     "any token 1-6; VAL 2-6", f"block [CTX, KEY, VAL, ...] of 1-3 pairs: KEY offsets {koff}, VAL offsets {voff}",
     "MATCH" if (min(koff), max(voff), min(voff)) == (1, 6, 2) else "DISAGREE", "explore_f_tasks.py:10-12 (block layout), :20-21; S70 uses lmin 1, lmax 3 (explore_f_rh_lite.py:7-10)",
     "a token is one to six tokens from its block's cue, the write at a value position two to six")
its = sorted({it for (_, it) in iters_all})
nrow("N4", 108, "training budgets '(24,000, 28,800 or 43,200 updates; 48,000 in some of session F's screens)'", "24000, 28800, 43200; 48000 in some F screens",
     f"F's records: {its}; 48000 in {i48}", "MATCH" if its == [24000, 28800, 43200, 48000] else "DISAGREE", "first-pass row 'training budgets' resolved",
     "43{,}200 updates; 48{,}000 in some of session~F's screens)")
nrow("N5", 124, "'at beta = 0 its logits agree with the Hebbian model's to 2e-7'", "2e-7", f"max |logit diff| over the C2 beta-0 cases {max(b0):.2e}",
     "MATCH" if max(b0) <= 2e-7 else "DISAGREE", "checks file explore_out/F/checks_s80_20261010_192401.json, not a run record", "agree with the Hebbian model's to $2\\times10^{-7}$")
nrow("N6", 124, "'with the new path disabled it reproduces a recorded run bit for bit through 2400 updates'", "through 2400",
     "C6: LOCAL3_SLOW|160 reproduced bit for bit through 2400 updates", "MATCH", "checks file row C6", "reproduces a recorded run bit for bit through 2400 updates")
SCR = {"S53": ("S53 SINGLE_DELTA_L_G", "S53 SINGLE_DELTA_L_H"), "S54(a)": ("S54(a) A_DELTA", "S54(a) A_ORACLE_DELTA"),
       "S54(b)": ("S54(b) B_DELTA", "S54(b) B_ORACLE_DELTA"), "S54(c)": ("S54(c) C_DELTA", "S54(c) C_ORACLE_DELTA"),
       "S54(d)": ("S54(d) D_DELTA", "S54(d) D_ORACLE_DELTA")}
fell = sorted(s for s, arms in SCR.items() if all(max(bsum[a]) < 0.5 for a in arms))
rose = sorted(s for s, arms in SCR.items() if all(min(bsum[a]) > 0.5 for a in arms))
mixed = sorted(set(SCR) - set(fell) - set(rose))
nrow("N7", 124, "'A learned beta collapsed toward \"do not write\" at value positions in two screens and rose above its initial 0.5 in two others'",
     "collapsed in two, rose in two", f"every arm below 0.5 at every layer: {fell}; every arm above 0.5 at every layer: {rose}; neither: {mixed} "
     f"({'; '.join(a + ' ' + '/'.join(f2(x) for x in bsum[a]) for s in mixed for a in SCR[s])})",
     "MATCH" if (fell, rose) == (["S53", "S54(d)"], ["S54(a)", "S54(b)"]) else "DISAGREE",
     "VAL medians by layer from beta_by_role; the fifth learned-beta screen, S54(c), is mixed and not mentioned (its learned-gate arm C_DELTA "
     "is below 0.5 at every layer, its oracle is not); B4 resolved",
     "A learned $\\beta$ collapsed toward ``do not write'' at value positions in two screens and rose above its initial $0.5$ in two others")
nrow("N8", 242, "reading 'the partition's advantage is in the number of streams': threshold '(>= 7/10 at four and <= 3/10 at eight)'",
     ">= 7/10 at four and <= 3/10 at eight", "explore_f_premise.py:23: S68_SINGLE_S8 <= 3/10 while S68_SINGLE_S4 >= 7/10", "MATCH",
     "section 2 item resolved", "($\\geq 7/10$ at four and $\\leq 3/10$ at eight)")
nrow("N9", 244, "S69: 'its final accuracy was lower with the learned decay (0.17 against 0.22; a description of ten runs)'", "0.17 against 0.22; marked",
     f"{f2(fa[0])} (ROUTED), {f2(fa[1])} (GDN) against {f2(fa[2])}", "MATCH" if f2(fa[0]) == f2(fa[1]) == "0.17" and f2(fa[2]) == "0.22" else "DISAGREE",
     "now marked as a description", "its final accuracy was lower with the learned decay ($0.17$ against $0.22$; a description of ten runs)")
ties025 = [s for s, a, b in zip(seeds1, ts["O_B025_L2"], ts["O_HEBB"]) if a == b]
nrow("N10", 246, "'at beta = 1 sooner than the Hebbian oracle on every seed (2400-6000; at beta = 0.25 it ties the Hebbian oracle on two of the four)'",
     "beta 1 sooner 4/4; beta 0.25 ties 2/4", f"beta 1 sooner {soon1}/4; beta 0.25 sooner {soon025}/4, ties on {ties025}; Hebbian {hrange}",
     "MATCH" if soon1 == 4 and len(ties025) == 2 and soon025 == 2 and hrange == "2400-6000" else "DISAGREE", "",
     ("at $\\beta = 1$ sooner than the Hebbian oracle on every seed (2400--6000", "it ties the Hebbian oracle on two of the four"))
nrow("N11", 246, "'with raw keys at beta = 1 one seed diverged and one bound late (16,800), and at beta = 0.25 all four bound, one of them at 7200'",
     "beta 1: 1 diverged, 1 at 16800; beta 0.25: 4/4, one at 7200", f"O_B1_RAW {ts['O_B1_RAW']}; O_B025_RAW {ts['O_B025_RAW']}",
     "MATCH" if ts["O_B1_RAW"].count(None) == 1 and 16800 in ts["O_B1_RAW"] and None not in ts["O_B025_RAW"] and max(ts["O_B025_RAW"]) == 7200 else "DISAGREE",
     "seeds 1300-1303", "with raw keys at $\\beta = 1$ one seed diverged and one bound late (16{,}800), and at $\\beta = 0.25$ all four bound, one of them at 7200")
src_merged = [(s, st_) for (s, st_, mx, ncs, nc0) in merged_before if ncs >= 2]
nrow("N12", 246, "the six unlabelled splits: 'at each the map had already merged'", "merged at each",
     f"a channel held >= 2 streams just before {nm_before} of {len(notok)}", "MATCH" if nm_before == len(notok) == 6 else "DISAGREE",
     "map_before (explore_f_eight_child.py:73-77); B5 resolved on this point",
     "at each the map had already merged")
nrow("N13", 246, "'and the split copied a channel that was not the merged one (its source held no stream or one in five of the six; at one the destination was not empty)'",
     "not the merged one, at each; source 0-1 streams in 5 of 6; one destination occupied",
     f"source c* held 0-1 streams in {sum(1 for m in merged_before if m[3] <= 1)} of 6; c* held >= 2 streams (a merged channel) at {src_merged}; "
     f"destination occupied at {[(m[0], m[1]) for m in merged_before if m[4] >= 1]}",
     "PARTIAL",
     "the parenthesis is right, but the main clause 'at each ... not the merged one' is not: at 1314@9600 the map was "
     "[6, 5, 4, 6, 10, 13, 2, 2] and the split copied channel 6, which held streams 0 and 3 (a merged channel), onto channel 10, which held "
     "stream 4. The first pass's own note had the same over-generalisation",
     "and the split copied a channel that was not the merged one (its source held no stream or one in five of the six; at one the destination was not empty)")
nrow("N14", 246, "'S54(c) differed from S76 in the learned beta, the noisy split with a reset, and the seeds'", "three differences",
     f"learned beta ({c54d['delta_cfg']['beta']!r}); S37/S48 noisy split with reset vs exact copy; seeds {min(C_DELTA)}-{max(C_DELTA)} vs {min(DS)}-{max(DS)}",
     "MATCH", "explore_f_window_delta_scale.py:3-5, 12; explore_f_eight.py:30-31", "S54(c) differed from S76 in the learned $\\beta$, the noisy split with a reset, and the seeds")
nrow("N15", 250, "S77: '(medians at key positions 0.57-0.67)'", "0.57-0.67 at KEY", f"KEY medians at L3 {rk} (S53_DELTA {f2(m3[('S53_DELTA', 'K')])}, S59_GDN {f2(m3[('S59_GDN', 'K')])})",
     eq("0.57-0.67", rk), "population now named", "(medians at key positions $0.57$--$0.67$)")
hKrun = [pv(r, f"{L}|K") for r in PA["S69_GDN"].values() for L in ("L1", "L2", "L3", "final")]
nrow("N16", 250, "'the Hebbian channel's medians are 0.50-0.53 at key positions (single runs up to 0.60)'", "0.50-0.53; runs up to 0.60",
     f"S69_GDN KEY medians L1-final {f2(min(hkK))}-{f2(max(hkK))}; per run at KEY {f2(min(hKrun))}-{f2(max(hKrun))} (max {max(hKrun):.4f})",
     "MATCH" if f"{f2(min(hkK))}-{f2(max(hkK))}" == "0.50-0.53" and f2(max(hKrun)) == "0.60" else "DISAGREE", "population now named",
     "the Hebbian channel's medians are $0.50$--$0.53$ at key positions (single runs up to $0.60$)")
nrow("N17", 250, "'Our reading: keys can be near-orthogonal ... without a stream direction in the residual'", "marked as our reading",
     "interpretation of S77 (report 4 lists it post hoc)", "MATCH", "now marked", "Our reading: keys can be near-orthogonal")
nrow("N18", 250, "S80 DELTA_A at KEY: '(0.98-1.00 at layer 3, median 1.00'", "0.98-1.00, median 1.00",
     f"per run {f2(min(l3k))}-{f2(max(l3k))}, median {f2(med(l3k))}", eq("0.98-1.00, median 1.00", f"{f2(min(l3k))}-{f2(max(l3k))}, median {f2(med(l3k))}"),
     "B6 resolved", "($0.98$--$1.00$ at layer~3, median $1.00$")
nrow("N19", 250, "'On the randomised header layout ... the in-run probe finds the stream at key positions in most of the delta solutions that bind without routing and in no routed one'",
     "(claim)", "session G's in-run probe (tex2:263: 18 of 31 at key positions and 13 at both)", "NOT RECORD-CHECKABLE", "outside F's records",
     "the in-run probe finds the stream at key positions in most of the delta solutions that bind without routing and in no routed one")
nrow("N20", 252, "'not shown worse than the Hebbian write at two, four and eight streams on screens of ten paired seeds (keeps it: 10/10, 10/10 and 9/10 against 10/10 each, one-sided bounds at the ceiling and not parity)'",
     "keeps it, not parity", f"readings {ra80} / {rb80} / {r76}; {kA[0]}/10 v {kA[1]}/10, {kB[0]}/10 v {kB[1]}/10, {cnt(br, DS)}/10 v {cnt(br, HS)}/10; seeds 10 paired each",
     "MATCH", "C1 resolved", ("not shown worse than the Hebbian write at two, four and eight streams on screens of ten paired seeds", "one-sided bounds at the ceiling and not parity"))
nrow("N21", 252, "'as a single channel binds 2/10 and 0/10 at four and eight streams (S68, whose reading of that name did not apply; the pre-registered support for the partition is C3 on X)'",
     "2/10, 0/10; S68 reading did not apply", f"{k4}/10, {k8}/10; S68 reading: {r68}", "MATCH" if (k4, k8, r68) == (2, 0, "neither") else "DISAGREE",
     "C3 resolved here", "as a single channel binds $2/10$ and $0/10$ at four and eight streams (S68, whose reading of that name did not apply")
nrow("N22", 314, "'one delta channel binds a minority at two streams and none at eight' (grouped layout)", "minority at 2; none at 8",
     f"grouped two-stream {cnt(bound, S53G)}/10 (S53), eight {k8}/10 (S68)", "MATCH" if cnt(bound, S53G) < 5 and k8 == 0 else "DISAGREE",
     "the X half's 0/10 (C3) is outside F's records", "one delta channel binds a minority at two streams and none at eight")
nrow("N23", 318, "'not shown worse than the Hebbian one on the grouped layout at two, four and eight streams (one-sided bounds on ten paired seeds each)'",
     "not shown worse", f"readings {ra80} / {rb80} / {r76}", "MATCH", "C1 resolved",
     "is not shown worse than the Hebbian one on the grouped layout at two, four and eight streams (one-sided bounds on ten paired seeds each)")
nrow("N24", 318, "'one delta channel binds the two-stream header layout 8/10 and the grouped one 4/10, by what we read as routes that need no partition, and binds 2/10 and 0/10 at four and eight streams'",
     "8/10, 4/10; 2/10, 0/10; marked", f"{cnt(bound, S53H)}/10, {cnt(bound, S53G)}/10; {k4}/10, {k8}/10",
     eq("8/10, 4/10; 2/10, 0/10", f"{cnt(bound, S53H)}/10, {cnt(bound, S53G)}/10; {k4}/10, {k8}/10"), "interpretation now marked ('what we read as')",
     "one delta channel binds the two-stream header layout $8/10$ and the grouped one $4/10$, by what we read as routes that need no partition, and binds $2/10$ and $0/10$ at four and eight streams")
nrow("N25", 338, "'one delta channel and Gated DeltaNet's write are now run at two, four and eight streams on the grouped layout'", "both at 2, 4, 8 on grouped",
     f"one delta channel: {cnt(bound, S53G)}/10, {k4}/10, {k8}/10 (S53, S68); the delta write at 2, 4, 8 (S80, S76) has alpha FIXED at 0.95 and beta 1 "
     f"(delta_cfg / delta fields); Gated DeltaNet's learned forget gate (S59 SINGLE_GDN / ORACLE_GDN) ran only on the P = 8 header layout",
     "PARTIAL", "true if 'Gated DeltaNet's write' means its update rule with alpha and beta fixed (tex2:124 calls it GDN's rule); GDN's gate "
     "itself was never run on the grouped layout", "one delta channel and Gated DeltaNet's write are now run at two, four and eight streams on the grouped layout")
nrow("N26", 344, "referee list: 'that a learned beta collapses (it rose in two screens); that six delta splits fired before the map had merged (it had; they copied the wrong channel)'",
     "rose in two; they copied the wrong channel", f"rose: {rose}; source c* not a merged channel at {sum(1 for m in merged_before if m[3] <= 1)} of 6, occupied destination at 1314@9600",
     "PARTIAL", "the first half matches; 'they copied the wrong channel' holds for five of six (at 1314@9600 the source was a merged channel and the "
     "destination was occupied), as N13", ("that a learned $\\beta$ collapses (it rose in two screens)", "(it had; they copied the wrong channel)"))
nrow("N27", 355, "Withdrawn: 'the delta write is not shown worse ... at two, four and eight streams on screens and binds sooner at two and eight'",
     "not shown worse; sooner at 2 and 8", f"sooner: two {soonA}/10, four {soonB}/10, eight {len(soon76)}/{len(both76)}", "MATCH" if soonA == 10 and soonB == 0 and len(soon76) == 8 else "DISAGREE",
     "C1 resolved", ("the delta write is not shown worse than the Hebbian one at two, four and eight streams on screens and binds sooner at two and eight"))
nrow("N28", 357, "'Supported on X by the pre-registered C3 (0/10 against the recipe's 10/10); session F's S68 had the same counts (2/10 at four streams, 0/10 at eight), but its own reading of that name did not apply'",
     "S68 2/10, 0/10; reading did not apply", f"S68 {k4}/10, {k8}/10; reading {r68}", "PARTIAL",
     "counts and reading match; 'the same counts' is loose: C3 is at eight streams only (0/10), so only S68's 0/10 at eight is the same count",
     ("\\textbf{Supported on X}", "session~F's S68 had the same counts ($2/10$ at four streams, $0/10$ at eight), but its own reading of that name did not apply"))
nrow("N29", 359, "'on the grouped two-stream layout it reaches depth at key positions in routed runs as well'", "routed runs at KEY",
     f"S80(a) DELTA_A DISCOVERED {kA[0]}/10 with L3 KEY median {f2(med(l3k))} (min {f2(min(l3k))})", "MATCH" if kA[0] == 10 and min(l3k) >= 0.9 else "DISAGREE",
     "C4 resolved (the randomised-layout half is session G's probe)", "on the grouped two-stream layout it reaches depth at key positions in routed runs as well")
nrow("N30", 342, "'that S54's slow delta oracle was a learned-beta artefact' quoted as a description", "a description",
     f"S54(c) C_ORACLE_DELTA learned beta at VAL {'/'.join(f2(x) for x in bsum['S54(c) C_ORACLE_DELTA'])}; transitions {'-'.join(map(str, trs(C_OD)))}",
     "MATCH", "marked ('quoted ... as descriptions, not results')", "that S54's slow delta oracle was a learned-$\\beta$ artefact")
nrow("N31", 384, "prediction overturned: the delta recipe at eight streams '(it keeps them 9/10 against 10/10 with the write strength fixed)'", "9/10 vs 10/10",
     f"{cnt(br, DS)}/10 vs {cnt(br, HS)}/10; reading {r76}", eq("9/10 vs 10/10", f"{cnt(br, DS)}/10 vs {cnt(br, HS)}/10"), "",
     "(it keeps them $9/10$ against $10/10$ with the write strength fixed)")
nrow("N32", 285, "footnote d: '... the exact-copy split at four and eight streams (none at two, where k = S)'", "no split at two",
     f"S80(a) k = {sorted({r['k'] for r in DA.values()})}, firings {sum(1 for r in list(DA.values()) + list(HA.values()) for c in r.get('checks', []) if c.get('fired'))}",
     "MATCH" if sorted({r['k'] for r in DA.values()}) == [2] else "DISAGREE", "section 7 item resolved",
     "the exact-copy split at four and eight streams (none at two, where $k = S$)")
c59s = f"{c59['ROUTED']}/10 and {c59['GDN']}/10 against {c59['FIXED']}/10 (S59, delta); {c69['ROUTED']}/10 and {c69['GDN']}/10 against {c69['FIXED']}/10 (S69, Hebbian)"
nrow("N33", 438, "the model's limits (NEW): the memory horizon; 'the one device screened against it, a learned forget gate, worked on the delta write and not the Hebbian one'",
     "worked on delta, not on Hebbian", c59s + f"; S59's pre-fixed reading on the horizon question: '{r59}'; S69's reading: '{r69}'", "PARTIAL",
     "the counts are right and S69's reading applies, but S59 was the screen of the horizon question (explore_f_decay.py:3-4) and its pre-fixed "
     "reading 'the horizon knob matters' read 'it does not'; S59 held L = 37 fixed (explore_f_decay.py:9) and never varied the distance, so 'worked against the "
     "horizon' is an unmarked description (it raised one channel's binding on the P = 8 header layout)",
     "the one device screened against it, a learned forget gate, worked on the delta write and not the Hebbian one")
nrow("N34", 457, "limits, memory horizon (UNCHANGED from tex:451; not in the first pass): 'the learned forget gate works on the delta write and not on the Hebbian one'",
     "works on delta, not on Hebbian", c59s + f"; S59 reading '{r59}'", "PARTIAL", "as N33", "the learned forget gate works on the delta write and not on the Hebbian one")
nrow("N35", 465, "conclusion: 'with its write strength fixed, the delta rule is not shown worse than the Hebbian one on screens at every size tried'",
     "not shown worse at every size", f"sizes 2, 4, 8 streams: readings {ra80} / {rb80} / {r76}", "MATCH", "C1 resolved",
     "the delta rule is not shown worse than the Hebbian one on screens at every size tried")
# S68 timing vs Revision 8's first draft (tex2:336, unchanged from tex:332; not in the first pass)
R8 = "4a3eebc"
hist = [h for h in git("log", "--reverse", "--format=%h", BR, "--", f"{F}premise_results.json").split()]
s8_done = None
for h in hist:
    k_ = sum(1 for kk, r in load(h, f"{F}premise_results.json")["runs"].items() if kk.startswith("S68_SINGLE_S8|") and not r.get("dry"))
    if k_ >= 10:
        s8_done = h
        break
gap = (int(git("log", "-1", "--format=%ct", R8)) - int(git("log", "-1", "--format=%ct", s8_done))) / 60
nrow("N36", 336, "(unchanged from tex:332; not in the first pass) Revision 8.1 list: 'session F's premise test was on its branch 82 minutes before the commit, and supported the reading'",
     "82 minutes; supported", f"all ten S68_SINGLE_S8 records first at {s8_done} ({cdate(s8_done)}); Revision 8 draft {R8} ({cdate(R8)}): {gap:.1f} min; "
     f"S68's pre-fixed readings: '{r68}'", "PARTIAL",
     "the 82 minutes reproduce (to the eight-stream single-channel arm's last record); 'supported the reading' is the C3 wording again: "
     "S68's pre-fixed readings read 'neither', and tex2:357 now says so",
     "session~F's premise test was on its branch 82 minutes before the commit, and supported the reading")
nrow("N37", 41, "'F's S68--S81' in this revision; 'F's S81 ... running'", "S81 running",
     f"F at {S81_NEW['ref']} ({S81_NEW['date']}), before the draft ({cdate(NEW_REF)}): {S81_NEW['n']} S81 records {S81_NEW['arms']}, "
     f"outcomes {S81_NEW['outcomes']}; log 'S81 done' {S81_NEW['done']}; no report 7", "MATCH" if not S81_NEW["done"] and S81_NEW["n"] > 0 else "DISAGREE",
     "no S81 number is quoted anywhere in the second draft (tex2:41, 252, 470, 507)", ("F's S68--S81", "F's S81 are running"))

# ══ write ═══════════════════════════════════════════════════════════════════
ORDER = {"DISAGREE": 0, "PARTIAL": 1, "NOT RECORD-CHECKABLE": 2, "MATCH": 3}


def esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def lstr(loc):
    return loc if isinstance(loc, str) else ", ".join(f"tex2:{x}" for x in loc)


def qstr(q):
    return " ... ".join(q) if isinstance(q, tuple) else q


carried = [r for r in ROWS if r["loc"] != "REMOVED"]
removed = [r for r in ROWS if r["loc"] == "REMOVED"]
live = carried + NROWS
cnt2 = Counter(r["status"] for r in live)
cnt1 = Counter(r["status"] for r in ROWS)
cntN = Counter(r["status"] for r in NROWS)
nid = {r["id"]: r for r in NROWS}
L = []
P = L.append
P("# Revision 9 draft, second version, vs Session F's records (second pass)\n")
P(f"Generated by `docs/review_2026-10/rev9/check_F_v2.py` (stdlib + `rlib`), built from `check_F.py` with every record recomputation kept. "
  f"Second draft: `{DRAFT_FILE}` at `{NEW_REF}` ({cdate(NEW_REF)}), cited `tex2:NNN`; first draft `{DRAFT_REF}` ({cdate(DRAFT_REF)}), cited `tex:NNN`. "
  f"Records: `{BR}` at **{HEAD}** ({cdate(HEAD)}); every store used is byte-identical to the one at the report commit the draft names: "
  f"{all(v[1] for v in SAME.values())} ({', '.join(f'{k}@{v[0]}' for k, v in SAME.items())}).\n")
P("Method. Each first-pass row that checks the draft carries a fragment of the first draft's text, asserted to be at the line the first "
  "pass cited. The fragment is then searched for in the second draft by text. If it is found, the row stands at that `tex2` line, "
  "because the wording it checked is unchanged and so is the recomputation. If it is not found, the row is **REMOVED**. A removed row "
  "is superseded: it is listed apart, with the second-pass row (`N..`) that checks the replacement wording. The second-pass rows cover "
  "every new or changed sentence about Session F in the second draft, plus two unchanged F sentences the first pass missed (N34, N36). "
  "Each second-pass row's fragment is asserted to be present at its `tex2` line.\n")
P(f"**Second draft: {len(live)} rows** ({len(carried)} first-pass rows still in the text or in report 6, {len(NROWS)} second-pass rows): "
  + ", ".join(f"{k} {cnt2.get(k, 0)}" for k in ORDER) + ".\n")
P(f"- First pass (first draft): {len(ROWS)} rows, " + ", ".join(f"{k} {cnt1.get(k, 0)}" for k in ORDER) + ".")
P(f"- Removed from the text (superseded): {len(removed)} rows, of which " + ", ".join(f"{k} {Counter(r['status'] for r in removed).get(k, 0)}" for k in ORDER) + ".")
P(f"- Second-pass rows alone: " + ", ".join(f"{k} {cntN.get(k, 0)}" for k in ORDER) + ".\n")
P("Definitions as in `check_F.md`: BOUND = `rlib.bound`; DISCOVERED = bound and final `val_cos` < 0.5; BOUND ROUTED = `rlib.outcome_k`; "
  f"ROUTED*@end = margin >= 0.9 and eta^2 by stream at KEY > 0.9. stream_gate ties deferred to ch_map: {sorted(set(TIES)) or 'none'}.\n")
P("## Second-draft table (non-MATCH first)\n")
P("| # | id | where (tex2) | first-pass tex | item | draft says | recomputed | status | note |")
P("|---|---|---|---|---|---|---|---|---|")
rows2 = sorted(live, key=lambda r: (ORDER[r["status"]], (lambda x: x[0] if isinstance(x, list) and x else 9999)(r["loc"])))
for i, r in enumerate(rows2, 1):
    P(f"| {i} | {r.get('id', '')} | {esc(lstr(r['loc']))} | {r['tex'] if r['kind'] == 'first' else ('tex:' + ', '.join(map(str, r['old'])) if r['old'] else 'new')} | "
      f"{esc(r['item'])} | {esc(r['draft'])} | {esc(r['rec'])} | **{r['status']}** | {esc(r['note'])} |")
P("\n## First-pass rows removed from the text (superseded)\n")
P("| first-pass tex | item | first-pass status | first-draft fragment (not in the second draft) | superseded by | its status | its tex2 line |")
P("|---|---|---|---|---|---|---|")
for r in sorted(removed, key=lambda r: (ORDER[r["status"]], int(r["tex"][4:]))):
    s = nid.get(r["sup"])
    P(f"| {r['tex']} | {esc(r['item'])} | {r['status']} | {esc(qstr(r['q']))} | {r['sup']} | {s['status'] if s else '--'} | {lstr(s['loc']) if s else '--'} |")
P("\n## Where each first-pass non-MATCH row stands\n")
P("| first-pass tex | item | first-pass status | second draft |")
P("|---|---|---|---|")
for r in sorted([r for r in ROWS if r["status"] != "MATCH"], key=lambda r: (ORDER[r["status"]], r["tex"])):
    if r["loc"] == "REMOVED":
        s = nid[r["sup"]]
        w = f"REMOVED; replacement {s['id']} at {lstr(s['loc'])}: {s['status']}"
    elif r["loc"] == "report_6.md":
        w = "report 6 (not the draft): unchanged"
    else:
        w = f"UNCHANGED at {lstr(r['loc'])}: still {r['status']}"
    P(f"| {r['tex']} | {esc(r['item'])} | {r['status']} | {esc(w)} |")

P("\n## Readings vs docstring, second draft\n")
P("| screen | reading (docstring, file:line) | second-draft wording (tex2) | recomputed | status |")
P("|---|---|---|---|---|")
RD2 = [
    ("S68", "\"context-tagged keys scale\" if S8 >= 7/10 (explore_f_premise.py:22)", "'(>= 7/10 at eight)' (tex2:242)", r68, "MATCH"),
    ("S68", "\"the partition's advantage is in the number of streams\" if S8 <= 3/10 while S4 >= 7/10 (:23)",
     "'(>= 7/10 at four and <= 3/10 at eight)' (tex2:242); 'did not apply' (tex2:252, 357)", r68, "MATCH (was PARTIAL)"),
    ("S69", "\"the forget gate needs the delta rule\" (explore_f_hebb_decay.py:23)", "paragraph title (tex2:244)", r69, "MATCH"),
    ("S59", "\"the horizon knob matters\" / \"it does not\" (explore_f_decay.py:30-34)",
     "not named; tex2:384 'it shrank it'; tex2:438, 457 'worked / works on the delta write' against the horizon", r59,
     "PARTIAL (the new limits sentence reads S59 as a horizon device that worked; its reading on the horizon read 'it does not')"),
    ("S70", "descriptive BASELINE only (explore_f_rh_lite.py:23-24)", "counts only (tex2:310)", f"HEBB DISCOVERED {cnt(discovered, RHH)}/20", "MATCH"),
    ("S77", "\"the delta stack tags its keys by stream\" (explore_f_probe.py:22-23)", "'They did not (1 of 16 bound runs)' (tex2:359)", f"does not apply ({len(tagged)}/16)", "MATCH"),
    ("S76 P1", "\"the slowness is the overwrite\" / \"... key normalisation\" (explore_f_eight.py:22-23)", "both 'neither' (tex2:246)", r76a, "MATCH"),
    ("S76 P2", "\"the delta recipe keeps eight streams\" if c - b <= 1 (explore_f_eight.py:39)", "applies (tex2:246); 'not shown worse' (tex2:36, 252, 318, 355, 465)", r76, "MATCH"),
    ("S80 (a), (b)", "\"the delta recipe keeps it\" if c - b <= 1 (explore_f_configs.py READINGS)", "'Both readings are \"keeps it\"' (tex2:248); 'not parity' (tex2:252)",
     f"{ra80}; {rb80}", "MATCH"),
    ("S81", "explore_f_trigger.py READINGS", "'running', no number (tex2:41, 252, 470, 507)", f"{S81_NEW['n']} progress records at {S81_NEW['ref']}; no report", "MATCH (no reading claimed)"),
]
for x in RD2:
    P("| " + " | ".join(esc(v) for v in x) + " |")
P("\nReadings commit vs first run record (unchanged method from `check_F.md`; S81 now has records):\n")
TR = load(HEAD, f"{F}trigger_results.json")
for p in PREC:
    if p["scr"] == "S81":
        p["rg"] = sorted({r["git"].split("+")[0] for r in TR["runs"].values() if r.get("git")})
        p["desc"] = all(is_anc(p["rc"], g) for g in p["rg"])
P("| screen | readings commit | first run record | precedes | docstring unchanged to head | records' HEADs descend from readings commit |")
P("|---|---|---|---|---|---|")
for p in PREC:
    P(f"| {p['scr']} | {p['rc']} ({p['rcd']}) | {p['fr'] or 'none'} ({p['frd']}; {p['nrec']} records) | "
      f"**{'yes' if p['prec'] else 'n/a' if not p['fr'] else 'NO'}** | {'yes' if p['unchanged'] else 'NO'} | "
      f"{('yes (' + ', '.join(p['rg']) + ')') if p['rg'] and p['desc'] else 'NO' if p['rg'] else '-- (S81 store not loaded)'} |")

P("\n## Descriptive statements in the second draft: marked or not\n")
P("| tex2 | statement | basis | marked? |")
P("|---|---|---|---|")
DS2 = [
    ("124", "learned beta 'collapsed ... in two screens and rose ... in two others'", f"beta_by_role; now factual (fell {fell}, rose {rose}, mixed {mixed})", "n/a (now a description that matches)"),
    ("244", "S59 'had learned to partly clear its memory at the context token'", "alpha diagnostic", "YES ('post hoc')"),
    ("244", "S69 'final accuracy was lower with the learned decay'", "accuracy diagnostic", "YES ('a description of ten runs')"),
    ("246", "the six splits 'copied a channel that was not the merged one'", "map_before; 5 of 6 (N13)", "n/a (descriptive; one exception)"),
    ("246", "'the plateau rule's target selection is tuned to the Hebbian trajectory'", "interpretation", "YES ('our reading of this')"),
    ("246", "learned beta 'the likely cause of the earlier stall'", "not varied by S76", "YES ('post hoc')"),
    ("250", "near-orthogonal keys without a stream direction", "interpretation of S77", "YES ('Our reading')"),
    ("250", "S80 KEY readability as adjacency", "report 6 post hoc", "YES"),
    ("252", "'not shown worse ... not parity'", "readings 'keeps it'", "n/a (now the readings' wording)"),
    ("252", "'Raw keys are excluded by their numerical fragility'", "one divergence in 8 raw-key runs; mechanism is report 5's post hoc reading (report_5.md:62-63)", "NO (unchanged)"),
    ("318", "'by what we read as routes that need no partition'", "interpretation", "YES"),
    ("336", "S68 'supported the reading'", f"S68 read '{r68}'", "NO (unchanged from tex:332)"),
    ("355", "'the stall Revision 8 reported came from a screen with a learned write strength'", "true as a description; causal reading is tex2:246's post hoc attribution", "NO (unchanged)"),
    ("357", "S68 'its own reading of that name did not apply'", f"r68 = {r68}", "n/a (correct)"),
    ("359", "routed runs reach depth at KEY on the grouped layout", "S80 probe", "n/a (correct)"),
    ("438, 457", "the learned forget gate 'worked' / 'works' on the delta write (against the memory horizon)", f"S59 counts; S59's horizon reading '{r59}'", "NO"),
]
for x in DS2:
    P("| " + " | ".join(esc(v) for v in x) + " |")
P("\n## Not checked here\n")
P("- Session G's numbers attached to F's statements: tex2:252's 14/20 against 1/20 and S71; the randomised-layout in-run probe at tex2:250 "
  "and tex2:359. They are outside F's records.\n- Main-line counts used beside F's (C3 on X, 69/80, 38/40, test_delta_recipe's operating "
  "characteristics). Only the arithmetic at tex2:248 is checked.\n- CHECK outcomes, wall-clock times and worker counts. They are in checks files "
  "and logs, not run records.\n- S81's readings. S81 is running, its records are progress commits, and the draft quotes no S81 number.")
with open(OUT, "w") as fh:
    fh.write("\n".join(L) + "\n")
print(f"wrote {OUT}: second draft {len(live)} rows; " + ", ".join(f"{k} {cnt2.get(k, 0)}" for k in ORDER)
      + f"; removed {len(removed)}; second-pass rows {len(NROWS)}")
for r in rows2:
    if r["status"] in ("DISAGREE", "PARTIAL", "NOT RECORD-CHECKABLE"):
        print(f"  {r['status']:8} {r.get('id', ''):4} {lstr(r['loc']):14} {r['tex']:10} {r['item'][:100]}")
for r in removed:
    print(f"  REMOVED {r['tex']:8} {r['status']:9} -> {r['sup']}: {nid[r['sup']]['status']}")
