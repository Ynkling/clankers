#!/usr/bin/env python3
"""check_F.py — the referee's check of everything Revision 9's draft (origin/claude/rev9-draft at 9bfa85a,
multichannel_hebbian_report_v9.tex) says about Session F, against Session F's run records on origin/claude/explore-F.

Inputs: JSON run records read with `git show <ref>:<path>` (rlib.load); the screens' docstrings (for definitions,
thresholds and reading names, cited file:line); git history (readings commit vs first run record). The draft's numbers are
typed in below as DRAFT strings (cited tex:NNN) and are never used as inputs: every RECOMPUTED value comes from the records.

Writes docs/review_2026-10/rev9/check_F.md. Read-only otherwise (git show / log / ls-tree / cat-file / merge-base only).
Python stdlib + rlib.
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
DRAFT_REF, DRAFT_FILE = "9bfa85a", "multichannel_hebbian_report_v9.tex"
R3, R4, R5, R6 = "9a6c2bf", "22ec912", "5b9bc2a", "ed76eea"     # report commits the draft names (tex:501)
OUT = os.path.join(HERE, "check_F.md")
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
r81_head = git("ls-tree", "-r", "--name-only", HEAD, f"{F}trigger_results.json").strip()
newer = git("log", "--format=%h %cI", f"{HEAD}..ae45031").strip() if subprocess.call(("git", "-C", ROOT, "cat-file", "-e", "ae45031")) == 0 else ""
r81_new = ""
if newer:
    t = load("ae45031", f"{F}trigger_results.json")
    r81_new = f"; first S81 record in ae45031 ({cdate('ae45031')}, after the draft commit {DRAFT_REF} at {cdate(DRAFT_REF)}): {list(t['runs'])[0].split('|')[:2]}"
row("tex:501", "S81 running", "running", f"at {HEAD} ({cdate(HEAD)}): trigger_results.json {'present' if r81_head else 'absent'}; s81.log shows runs started{r81_new}",
    "MATCH", "the remote has a newer F commit (ae45031) not in the fetched ref; it adds the first S81 record after the draft")
row("tex:501", "branch commit", "ed76eea", f"report 6 commit ed76eea; fetched head {HEAD} adds only S81 logs/checks", "MATCH",
    f"commits after ed76eea up to {HEAD}: {git('log', '--format=%h', f'ed76eea..{HEAD}').split()}")

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
    scan(d, 2000, 2059, a)
    scan(d, 1300, 1359, c)
    if a:
        hits20[blob] = (where, sorted(a))
    if c:
        hits13[blob] = (where, sorted(c))
w20 = sorted({p for wh, _ in hits20.values() for _, p in wh})
row(f"{R6F}:5-6", "seeds 2000-2019: the block appears in no record on any branch", "no other record",
    f"seeds 2000-2059 appear only in {w20} across {len(blobs)} JSON blobs on {len(set(b for v in blobs.values() for b, _ in v))} remote branches",
    "MATCH" if set(w20) <= {f"{F}configs_a_results.json", f"{F}configs_b_results.json"} else "DISAGREE",
    "scan at the fetched heads (after S80 ran); keys 'ARM|seed|...' and 'seed' fields")
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

# ══ write ═══════════════════════════════════════════════════════════════════
ORDER = {"DISAGREE": 0, "PARTIAL": 1, "NOT RECORD-CHECKABLE": 2, "MATCH": 3}


def esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def texkey(r):
    t = r["tex"]
    if t.startswith("tex:"):
        return (0, int(t[4:].split()[0].split("-")[0]))
    return (1, int(t.split(":")[1].split("-")[0]))


rows = sorted(ROWS, key=lambda r: (ORDER[r["status"]], texkey(r)))
cnts = Counter(r["status"] for r in ROWS)
L = []
P = L.append
P("# Revision 9 draft vs Session F's records\n")
P(f"Generated by `docs/review_2026-10/rev9/check_F.py` (stdlib + `rlib`). Draft: `{DRAFT_FILE}` at `{DRAFT_REF}` "
  f"({cdate(DRAFT_REF)}), cited `tex:NNN`. Records: `{BR}` at **{HEAD}** ({cdate(HEAD)}), `explore_out/F/*_results.json`; "
  "every store used is byte-identical to the one at the report commit the draft names "
  f"({', '.join(f'{k}@{v[0]}' for k, v in SAME.items() if v[1])}"
  f"{'; DIFFERENT: ' + ', '.join(k for k, v in SAME.items() if not v[1]) if not all(v[1] for v in SAME.values()) else ''}). "
  "Report 6 (S80, `ed76eea`) is checked in full below (rows cited `report_6.md:N`). The draft's numbers are typed in as strings and "
  "never used as inputs.\n")
P(f"**Rows: {len(ROWS)}** — " + ", ".join(f"{k} {cnts.get(k, 0)}" for k in ORDER) + ".\n")
P("Definitions: BOUND = `rlib.bound` (transition recomputed from `curve`); DISCOVERED = bound and final `val_cos` < 0.5 "
  "(S80(a), explore_f_configs.py OUTCOME); BOUND ROUTED = `rlib.outcome_k` (test_stream_recipe.outcome; cross-checked against the "
  f"recorded `outcome` field for S80(b): {len(xB)} differ); ROUTED*@end (S70) = margin >= 0.9 and eta^2 by stream at KEY > 0.9. "
  f"stream_gate ties deferred to ch_map: {TIES or 'none'}.\n")
P("## Table\n")
P("| # | tex line | item | draft says | recomputed | status | note |")
P("|---|---|---|---|---|---|---|")
for i, r in enumerate(rows, 1):
    P(f"| {i} | {r['tex']} | {esc(r['item'])} | {esc(r['draft'])} | {esc(r['rec'])} | **{r['status']}** | {esc(r['note'])} |")

P("\n## Readings vs docstring (with readings commit vs first run commit for S80)\n")
P("| screen | reading (docstring, file:line) | report wording | draft wording (tex) | recomputed | status |")
P("|---|---|---|---|---|---|")
RD = [
    ("S68", "\"context-tagged keys scale\" if S68_SINGLE_S8 >= 7/10 (explore_f_premise.py:22)", "same (report_3.md:26)",
     "'context-tagged keys scale' (>= 7/10 at eight) (tex:242)", r68 if r68 != "context-tagged keys scale" else "applies", "MATCH"),
    ("S68", "\"the partition's advantage is in the number of streams\" if S8 <= 3/10 while S4 >= 7/10 (:23)", ">= 7/10 at S = 4 (report_3.md:27)",
     "'the partition's advantage is in the number of streams' (>= 7/10 at four) (tex:242)", r68, "PARTIAL (S = 8 condition omitted)"),
    ("S69", "\"the forget gate needs the delta rule\" if ROUTED <= FIXED + 1 and GDN <= FIXED + 1 (explore_f_hebb_decay.py:23)", "same (report_3.md:15)",
     "paragraph title 'The forget gate needs the delta rule' (tex:244); threshold not stated", r69, "MATCH"),
    ("S59", "\"the horizon knob matters\" if gap_d - gap_FIXED >= 0.15 on >= 7/10 for ROUTED or GDN; else \"it does not\" (explore_f_decay.py:30-32)",
     "report 2", "not named; tex:378 'it shrank it'", r59, "MATCH (consistent; reading not named)"),
    ("S70", "descriptive BASELINE only; 'No other reading' (explore_f_rh_lite.py:23-24)", "report_3.md:18-19", "counts only (tex:306)",
     f"HEBB DISCOVERED {cnt(discovered, RHH)}/20", "MATCH"),
    ("S77", "\"the delta stack tags its keys by stream\" (>= 8 bound delta runs L3 >= 0.9 at K and V, L1 <= 0.6) (explore_f_probe.py:22-23)",
     "does not apply (report_4.md:25-26)", "'They did not (1 of 16 bound runs)' (tex:353); descriptive at tex:250", f"does not apply ({len(tagged)}/16)", "MATCH"),
    ("S76 P1", "\"the slowness is the overwrite\" / \"the slowness is the key normalisation\" (explore_f_eight.py:22-23)", "neither (report_5.md:33)",
     "'(the overwrite; the key normalisation) both read \"neither\"' (tex:246)", r76a, "MATCH"),
    ("S76 P2", "\"the delta recipe keeps eight streams\" if c - b <= 1 (explore_f_eight.py:39)", "same (report_5.md:40)",
     "'the delta recipe keeps eight streams' applies (tex:246)", r76, "MATCH"),
    ("S54(c)", "\"delta worse\" if c - b > 2 (explore_f_window_delta_scale.py:29-32)", "report 2", "'stalled at eight streams (3/10 against 9/10)' (tex:246)", c54_reading(), "MATCH (not named)"),
    ("S80 (a)", "\"the delta recipe keeps it\" if c - b <= 1; \"loses it\" if c >= 4 and b = 0; else \"inconclusive\" (explore_f_configs.py READINGS)",
     "'the delta recipe keeps it' (report_6.md:24)", "'Both readings are \"keeps it\"' (tex:248); 'keeps 2, 4 and 8 streams' (tex:58)", ra80, "MATCH (abbreviated)"),
    ("S80 (b)", "as (a), on BOUND ROUTED", "'the delta recipe keeps it' (report_6.md:25)", "as (a)", rb80, "MATCH (abbreviated)"),
    ("S81", "\"the guard aims the split\", \"the evaluation is the better probe\", ... (explore_f_trigger.py READINGS)", "no report", "'the subject of F's S81' (tex:252), 'S81 running' (tex:501)", "no records at the fetched head", "MATCH (no reading claimed)"),
]
for x in RD:
    P("| " + " | ".join(esc(v) for v in x) + " |")
P("\nReadings commit = earliest commit on the branch adding the reading text to the screen file (`git log -S`); first run record = "
  "earliest commit at which the screen's store holds >= 1 non-dry record of its arms; 'records' HEADs' = the `git` field of the records "
  "(the HEAD under which each was written; `+dirty` stripped).\n")
P("| screen | file | text searched | readings commit | first run record | precedes | docstring unchanged to head | records' HEADs descend from readings commit |")
P("|---|---|---|---|---|---|---|---|")
for p in PREC:
    P(f"| {p['scr']} | {p['path']} | {esc(p['text'])} | {p['rc']} ({p['rcd']}) | {p['fr'] or 'none at ' + HEAD} ({p['frd']}; {p['nrec']} records) | "
      f"**{'yes' if p['prec'] else 'n/a' if not p['fr'] else 'NO'}** | {'yes' if p['unchanged'] else 'NO'} | "
      f"{('yes (' + ', '.join(p['rg']) + ')') if p['rg'] and p['desc'] else 'NO' if p['rg'] else '--'} |")
P(f"\nS80 detail: readings commit ebb381c ({cdate('ebb381c')}) adds explore_f_configs.py, explore_f_configs_b.py and both children; "
  f"the checks file `checks_s80_20261010_192401.json` records git ebb381c, started 19:24:02; `s80.log` starts 19:23:56 with git ebb381c; "
  f"the first store commit with S80 records is {[p for p in PREC if p['scr'] == 'S80'][0]['fr']} ({[p for p in PREC if p['scr'] == 'S80'][0]['frd']}); "
  f"the records were written under HEADs {[p for p in PREC if p['scr'] == 'S80'][0]['rg']}, all descendants of ebb381c. "
  "S81: readings commit 451cde9 (19:28:32 UTC); no S81 record in the fetched ref; the first (GUARD|2023, git a7c7d55) is in ae45031 "
  "(22:28:11 UTC), after the draft (22:16:07 UTC).\n")

P("## Descriptive statements: marked or not\n")
P("Statements the draft presents as findings that are not a pre-fixed reading. 'Marked' = the draft says post hoc / descriptive / "
  "an interpretation at that place.\n")
P("| tex | statement | basis in the records / docstrings | marked? |")
P("|---|---|---|---|")
DS_ = [
    ("tex:124", "'A learned beta collapses toward \"do not write\"'", f"beta_by_role DIAGNOSTIC, not a reading; contradicted in {len(up)} arms of S54(a)/(b) (VAL medians {min(min(bsum[x]) for x in up):.2f}-{max(max(bsum[x]) for x in up):.2f} at every layer vs 0.5 at init)", "NO"),
    ("tex:244", "S59 'had learned to partly clear its memory at the context token'", "alpha DIAGNOSTIC (explore_f_decay.py:36-37); interpretation", "YES ('post hoc')"),
    ("tex:244", "'the Hebbian channel does not learn the clearing'; 'the learned decay lowered its final accuracy (0.17 against 0.22)'", "S69 alpha and accuracy DIAGNOSTICS (explore_f_hebb_decay.py:26-27); paired seeds, but no reading", "NO"),
    ("tex:246", "'Six of the twelve splits ... fired before the map had merged'", f"contradicted: the labelled map had merged at all {nm_before} of them (row above)", "NO"),
    ("tex:246", "'the plateau rule is tuned to the Hebbian trajectory'", "interpretation; no S76 arm varies the rule (S81 does, no records)", "NO"),
    ("tex:246", "learned beta 'the likely cause of the earlier stall'", "not varied by S76", "YES ('that attribution is post hoc')"),
    ("tex:250", "'Keys can be near-orthogonal in the 256-dimensional code without a stream direction in the residual'", "interpretation of S77 (report 4 lists it under post hoc)", "NO"),
    ("tex:250", "S80 probe: adjacency to the CTX token", "report 6 'Post hoc (not tested here)'", "YES"),
    ("tex:252", "'at parity ... at two, four and eight streams'", "readings say 'keeps it' (non-inferiority, ceiling caveat)", "NO"),
    ("tex:252", "'still needs the partition at four and eight streams (S68)'", f"S68 read 'neither'; counts {k4}/10, {k8}/10", "NO"),
    ("tex:252", "'Raw keys are excluded by their numerical fragility (beta||x||^2 > 2 ...)'", "one divergence in 8 raw-key runs; mechanism post hoc (report_5.md:62-63)", "NO"),
    ("tex:314", "'interchangeable with the Hebbian one on the grouped layout at every size tried'; 'binds the two-stream layouts by routes that need no partition'", "stronger than the readings; grouped two-stream single delta 4/10", "NO"),
    ("tex:349", "'matches the Hebbian one at two, four and eight streams and binds sooner'; 'the stall ... came from a screen with a learned write strength'", "not sooner at four; causal reading is post hoc at tex:246", "NO"),
    ("tex:351", "'Supported' citing S68", "S68's pre-fixed reading read 'neither'", "NO"),
    ("tex:353", "'the context reaches depth on delta memory only in the solutions that bind without routing'", "session G's probe; conflicts with S80 DELTA_A KEY L3 1.00 median unless the post hoc adjacency reading is applied", "NO"),
    ("tex:378", "'it shrank it' (learned forget gate vs the oracle gap)", f"S59 reading 'it does not' (widening); gap differences negative ({', '.join(f'{d} median {med(v):+.2f}' for d, v in gaps.items())})", "n/a (descriptive, supported)"),
]
for x in DS_:
    P("| " + " | ".join(esc(v) for v in x) + " |")
P("\n## Not checked here\n")
P("- Session G's numbers the draft attaches to F's statements (tex:252 14/20 vs 1/20 and S71; tex:250/353 the randomised-layout probe; "
  "tex:298 rows): outside F's records.\n- Main-line counts (69/80, 38/40, C3 on X) used beside F's: outside F's records; only the "
  "arithmetic at tex:248 is checked.\n- CHECK outcomes, wall-clock times and worker counts: checks files and logs, not run records.\n"
  f"- The remote has advanced (claude/explore-F to ae45031); this check uses the fetched ref {HEAD}, and reads ae45031 only to date "
  "the first S81 record.")
with open(OUT, "w") as fh:
    fh.write("\n".join(L) + "\n")
print(f"wrote {OUT}: {len(ROWS)} rows; " + ", ".join(f"{k} {cnts.get(k, 0)}" for k in ORDER))
for r in rows:
    if r["status"] in ("DISAGREE", "PARTIAL"):
        print(f"  {r['status']:8} {r['tex']:16} {r['item'][:90]} | draft {r['draft'][:50]} | rec {r['rec'][:140]}")
