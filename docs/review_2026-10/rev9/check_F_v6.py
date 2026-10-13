#!/usr/bin/env python3
"""check_F_v6.py — sixth pass. The referee's check of what Revision 9's draft says about Session F, run against the
SIXTH version of the draft (42c4557, multichannel_hebbian_report_v9.tex, 568 lines), with the first (9bfa85a), second
(f67bc07), third (a5f67a2), fourth (c66cc76) and fifth (c5e4508) versions kept for locating the earlier passes' rows.

A copy of check_F_v5.py with every record recomputation and every v5 row kept unchanged (they are still asserted at
their tex, tex2, tex3, tex4 and tex5 lines). What is new in v6:
  * F's records are pinned at report 7 (d5851b1). v5 asserted that F's head WAS report 7; F has since committed S88's
    readings (40087cb, 23:07:15 UTC) and its queue and progress (cdbb886, 23:07:46), both before the sixth draft, and more
    after it. The assert is replaced by a check that every F store used is byte-identical at d5851b1, at F's last commit
    before the sixth draft and at F's head now, and by rows on each branch's last commit before 23:14:04 UTC;
  * every row live in v5 is searched for in the sixth draft by text: found -> it stands at that tex6 line; not found ->
    REMOVED in v6, superseded by a T row; rows whose fragment survives but whose verdict must be re-read at the sixth
    draft's time (F's provenance row, the delta_recipe provenance row and next step, S81's unmarked 'because') are
    SUPERSEDED in v6 by the T row that re-checks them;
  * T rows: every changed sentence of the sixth draft about Session F (the index row, S81's paragraph, the next steps),
    the delta write's main-line test (its status lines, the abstract's X numbers recomputed from
    results/X/delta_recipe_results.json, the memory section's reworded sentences), and the status line tex6:41, each with a
    fragment asserted at its tex6 line.

Inputs: JSON run records read with `git show <ref>:<path>` (rlib.load); the screens' docstrings; git history. The
draft's numbers are typed in as DRAFT strings and never used as inputs.

Writes docs/review_2026-10/rev9/check_F_v6.md. Read-only otherwise. Python stdlib + rlib.
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
F_NOW = subprocess.check_output(("git", "-C", ROOT, "rev-parse", "--short=7", BR)).decode().strip()   # F's head now
HEAD = "d5851b1"     # v6: F's records pinned at report 7 (v5 read them at F's head, which was report 7 then)
DRAFT_REF, DRAFT_FILE = "9bfa85a", "multichannel_hebbian_report_v9.tex"     # first version (first pass)
NEW_REF = "f67bc07"                                                         # second version (second pass)
THIRD_REF = "a5f67a2"                                                       # third version (third pass)
FOURTH_REF = "c66cc76"                                                      # fourth version (this pass)
R3, R4, R5, R6 = "9a6c2bf", "22ec912", "5b9bc2a", "ed76eea"     # report commits the draft names (tex:501)
OUT = os.path.join(HERE, "check_F_v6.md")
FIFTH_REF = "c5e4508"                                                       # fifth version (fifth pass)
SIXTH_REF = "42c4557"                                                       # sixth version (this pass)
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
TEX3 = git("show", f"{THIRD_REF}:{DRAFT_FILE}").split("\n")
TEX4 = git("show", f"{FOURTH_REF}:{DRAFT_FILE}").split("\n")
TEX5 = git("show", f"{FIFTH_REF}:{DRAFT_FILE}").split("\n")
TEX6 = git("show", f"{SIXTH_REF}:{DRAFT_FILE}").split("\n")

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
    r["loc"] = find(TEX2, q) or "REMOVED"          # second draft (as v2)
    r["loc3"] = find(TEX3, q) or "REMOVED"         # third draft (v3)
assert used == set(range(len(QMAP))), sorted(set(range(len(QMAP))) - used)

# ══ second pass: rows for every new or changed sentence about Session F ═════
NROWS = []


def nrow(nid, tex2, item, draft, rec, status, note, q2):
    loc = find(TEX2, q2)
    assert tex2 in loc, ("fragment not at the cited second-draft line", nid, tex2, q2, loc)
    NROWS.append(dict(id=nid, tex=f"tex2:{tex2}", item=item, draft=str(draft), rec=str(rec), status=status, note=note,
                      q=q2, loc=loc, kind="second", old=find(TEX, q2), loc3=find(TEX3, q2) or "REMOVED"))


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

# ══ third pass: rows for every new or changed sentence about Session F in the third draft ═════
MROWS = []


def mrow(mid, tex3, item, draft, rec, status, note, q3):
    loc = find(TEX3, q3)
    assert tex3 in loc, ("fragment not at the cited third-draft line", mid, tex3, q3, loc)
    MROWS.append(dict(id=mid, tex=f"tex3:{tex3}", item=item, draft=str(draft), rec=str(rec), status=status, note=note,
                      q=q3, loc=loc, loc3=loc, kind="third", old=find(TEX2, q3)))


def ctime(h):
    return int(git("log", "-1", "--format=%ct", h).strip())


# tex3:246 — the six unlabelled delta splits, reworded
six = {(s, st_): (ncs, nc0) for (s, st_, mx, ncs, nc0) in merged_before}
src01 = sorted(k for k, (ncs, nc0) in six.items() if ncs <= 1)
odd = sorted(k for k, (ncs, nc0) in six.items() if ncs >= 2)
occ = sorted(k for k, (ncs, nc0) in six.items() if nc0 >= 1)
c1314 = [c for s, c in notok if (s, c["step"]) == (1314, 9600)]
m1314 = c1314[0]["map_before"] if c1314 else None
desc1314 = (f"map_before {m1314}; c* = {c1314[0]['key_cs']} holds streams {[i for i, x in enumerate(m1314) if x == c1314[0]['key_cs']]}; "
            f"c0 = {c1314[0]['key_c0']} holds streams {[i for i, x in enumerate(m1314) if x == c1314[0]['key_c0']]}") if c1314 else "not found"
mrow("M1", 246, "the six unlabelled delta splits: 'at each the map had already merged; in five of the six the split copied a channel that held no "
     "stream or one, and at the sixth (seed 1314, update 9600) it copied a merged channel onto an occupied one'",
     "merged at 6 of 6; source 0-1 streams at 5; the sixth 1314@9600, merged source, occupied destination",
     f"merged before at {nm_before} of {len(notok)}; source c* held 0-1 streams at {len(src01)} {src01}; source c* held >= 2 streams at {odd}; "
     f"destination c0 occupied at {occ}; at 1314@9600: {desc1314}",
     "MATCH" if (nm_before, len(src01), odd, occ) == (6, 5, [(1314, 9600)], [(1314, 9600)]) else "DISAGREE",
     "map_before, key_cs, key_c0 of eight_results.json DELTA_SPLIT checks (explore_f_eight_child.py:73-77); replaces N13; B5 resolved",
     ("at each the map had already merged; in five of the six the split copied a channel that held no stream or one",
      "and at the sixth (seed 1314, update 9600) it copied a merged channel onto an occupied one"))
mrow("M2", 246, "'mostly single splits in runs that then bound' (the six unlabelled delta splits)", "mostly single; runs bound",
     f"{len(single)} of {len(notok)} in single-split runs; runs bound: {sum(1 for s, _ in notok if bound(DS[s]))} of {len(notok)}",
     "MATCH" if len(single) * 2 > len(notok) and all(bound(DS[s]) for s, _ in notok) else "DISAGREE",
     "same recomputation as the first-pass row ('mostly as single splits'), whose wording is gone", "mostly single splits in runs that then bound")
# tex3:252 — raw keys, now marked
n_raw = len(E1["O_B1_RAW"]) + len(E1["O_B025_RAW"])
div_raw = [k for k in div if k.split("|")[0] in ("O_B1_RAW", "O_B025_RAW")]
r5 = git("show", f"{HEAD}:{F}report_5.md").split("\n")
r5_ln = [i + 1 for i, ln in enumerate(r5) if "Raw keys are numerically fragile" in ln]
r5_sec = max(i + 1 for i, ln in enumerate(r5[:r5_ln[0]]) if ln.startswith("## ")) if r5_ln else None
mrow("M3", 252, "'Raw keys are excluded by their numerical fragility (... expanding, which is session F's post hoc reading of the one divergence "
     "in eight raw-key oracle runs)'", "one divergence in eight raw-key oracle runs; F's post hoc reading",
     f"raw-key oracle runs {n_raw} (O_B1_RAW {len(E1['O_B1_RAW'])}, O_B025_RAW {len(E1['O_B025_RAW'])}); diverged {div_raw}; the mechanism is "
     f"report_5.md:{r5_ln[0] if r5_ln else '?'}-{(r5_ln[0] + 1) if r5_ln else '?'}, under '{r5[r5_sec - 1] if r5_sec else '?'}' (report_5.md:{r5_sec})",
     "MATCH" if n_raw == 8 and len(div_raw) == 1 and r5_ln else "DISAGREE",
     "now marked as F's post hoc reading; report 5 gives it after the run, under its 'Anything unexpected' heading, without the word 'post hoc' "
     "(the label is the draft's, and fair); supersedes the first-pass NOT RECORD-CHECKABLE row", "which is session~F's post hoc reading of the one divergence in eight raw-key oracle runs")
# tex3:41, 54, 252, 489 — test_delta_recipe's status (git history; the delta write's main-line test)
DR = dict(spec="05256c5", review="fad1489", amend="32c121b", test="a581027", tcheck="6f2788d", review2="16b04f9")
drs = {k: (h, cdate(h)) for k, h in DR.items()}
order_ok = ctime(DR["spec"]) < ctime(DR["review"]) < ctime(DR["amend"]) < ctime(DR["test"]) < ctime(DR["tcheck"]) < ctime(THIRD_REF)
files = {k: git("show", "--name-only", "--format=", h).split() for k, h in DR.items()}
res_files = []
for b_ in [x.strip() for x in git("branch", "-r").split("\n") if x.strip() and "->" not in x]:
    for f_ in git("ls-tree", "-r", "--name-only", b_).split("\n"):
        if "delta_recipe" in f_ and (f_.endswith(".json") or "results" in f_) and not f_.startswith("docs/"):
            res_files.append(f"{b_}:{f_}")
tcheck_head = git("show", f"{DR['tcheck']}:docs/review_2026-10/test_check_delta_recipe.md").split("\n")[0]
drec = (f"spec {drs['spec']}; review {drs['review']}; Amendment 1 {drs['amend']} (files {files['amend']}); test {drs['test']} (files {files['test']}); "
        f"test checked against the amended spec {drs['tcheck']} ('{tcheck_head[:80]}'); third draft {THIRD_REF} ({cdate(THIRD_REF)}); in this order: {order_ok}; "
        f"results files for delta_recipe on any branch: {res_files or 'none'}")
dr_ok = order_ok and not res_files and "specs/test_delta_recipe.md" in files["amend"] and "test_delta_recipe.py" in files["test"]
mrow("M4", 41, "test_delta_recipe: 'specified, reviewed by the referee, amended once and written, its test file checked against the amended spec before any run'",
     "specified, reviewed, amended once, written, checked; not run", drec, "MATCH" if dr_ok else "DISAGREE",
     "git history, not run records. The test check (6f2788d) is not among the commits tex3:511 cites (fad1489, 16b04f9, e8fbb25)",
     "specified, reviewed by the referee, amended once and written, its test file checked against the amended spec before any run")
mrow("M5", 54, "index row delta_recipe: 'spec reviewed by the referee and amended; the test written and checked against the amended spec; not run'",
     "as M4", "as M4", "MATCH" if dr_ok else "DISAGREE", "git history", "the test written and checked against the amended spec; not run")
mrow("M6", 252, "'(specified in specs/, reviewed by the referee before any run, amended once on its review, and written, with the test file checked against the amended spec; not run)'",
     "as M4", "as M4", "MATCH" if dr_ok else "DISAGREE", "git history",
     "amended once on its review, and written, with the test file checked against the amended spec; not run)")
mrow("M7", 489, "provenance row delta_recipe: 'spec 05256c5, Amendment 1 32c121b / test a581027 / not run'", "05256c5; 32c121b; a581027; not run",
     f"{drs['spec']} '{git('log', '-1', '--format=%s', DR['spec']).strip()[:50]}'; {drs['amend']} '{git('log', '-1', '--format=%s', DR['amend']).strip()[:50]}'; "
     f"{drs['test']} '{git('log', '-1', '--format=%s', DR['test']).strip()[:50]}'; results: {res_files or 'none'}",
     "MATCH" if dr_ok else "DISAGREE", "second pass: tex2:489 'not written' was false; now right",
     ("\\texttt{delta\\_recipe} & spec \\texttt{05256c5}, Amendment~1 \\texttt{32c121b} / test \\texttt{a581027} / not run"))
pdt = git("show", f"{DR['test']}:test_delta_recipe.py").split("\n")
pb_def = [i + 1 for i, ln in enumerate(pdt) if ln.startswith("def probe_blind")]
pb_prn = [i + 1 for i, ln in enumerate(pdt) if "probe-blind checks" in ln and "print" in ln]
mrow("M8", 427, "method: 'A test should print them [probe and evaluation] side by side, with the count of checks the probe blocked, as test_delta_recipe now does'",
     "test_delta_recipe prints both and the count", f"test_delta_recipe.py@{DR['test']}: probe_blind() at :{pb_def}, printed at :{pb_prn}",
     "MATCH" if pb_def and pb_prn else "DISAGREE", "code, not records (the test has not run)", "with the count of checks the probe blocked, as \\texttt{test\\_delta\\_recipe} now does")
# tex3:336 — the premise test in Revision 8.1's list
mrow("M9", 336, "Revision 8.1 list: 'session F's premise test was on its branch 82 minutes before the commit, and its counts, 0/10 at eight streams, bore on the claim'",
     "82 minutes; 0/10 at eight", f"all ten S68_SINGLE_S8 records first at {s8_done} ({cdate(s8_done)}); Revision 8 draft {R8} ({cdate(R8)}): {gap:.1f} min; "
     f"S68_SINGLE_S8 BOUND {k8}/10; S68's readings '{r68}' (not claimed here)", "MATCH" if k8 == 0 and round(gap) == 82 else "DISAGREE",
     "replaces N36: 'supported the reading' is gone; the claim it bears on is 'no single-channel run had been made at eight streams'",
     "and its counts, $0/10$ at eight streams, bore on the claim")
# tex3:338 — the baseline family
gdn_arms = sorted({(nm, k.split("|")[0]) for nm, stv in dict(decay=DC, hebb_decay=HD, probe=PB, delta_controls=DL, window_delta=WD,
                   window_delta_scale=WS, premise=PR, rh_lite=RH, eight=EI, configs_a=CA, configs_b=CB).items() for k in stv["runs"] if "GDN" in k.split("|")[0]})
dfix = {"S53 SINGLE_DELTA_G": sorted({json.dumps({x: r["delta_cfg"][x] for x in ("beta", "decay")}) for r in S53G.values()}),
        "S68": sorted({json.dumps({x: r["delta_cfg"][x] for x in ("beta", "decay")}) for r in list(S4.values()) + list(S8.values())}),
        "S80(a) DELTA_A": sorted({json.dumps({x: r["delta_cfg"][x] for x in ("beta", "decay")}) for r in DA.values()}),
        "S80(b) DELTA_B": dB, "S76 DELTA_SPLIT": sorted({json.dumps(r["delta"].get("beta")) for r in DS.values()})}
lay53 = sorted({r.get("layout") for r in S53G.values()})
mrow("M10", 338, "'one delta channel and the delta write with its decay and write strength fixed are now run at two, four and eight streams on the grouped "
     "layout, and Gated DeltaNet's learned forget gate on the eight-key header layout'",
     "both at 2, 4, 8, grouped, decay and beta fixed; GDN's gate on the P = 8 header layout",
     f"one delta channel: S53 G layout {lay53} {cnt(bound, S53G)}/10, S68 S4 {k4}/10, S8 {k8}/10 (grouped: explore_f_premise.py:7-9); the delta write: {dfix} "
     f"(decay 0.95 for S80(b) and S76 in the docstrings, explore_f_configs.py:7-8, explore_f_eight.py:17); arms with a GDN decay in F's stores: {gdn_arms}, "
     "all on HeaderTask(8, S=2), L = 37 (explore_f_decay.py:8-9; explore_f_hebb_decay.py:8)",
     "MATCH" if lay53 == ["grouped"] and set(x[1] for x in gdn_arms) <= {"SINGLE_GDN", "ORACLE_GDN", "S69_GDN", "S59_GDN"} else "DISAGREE",
     "replaces N25 (second pass: GDN's learned gate ran only on the header layout); now said",
     ("the delta write with its decay and write strength fixed are now run at two, four and eight streams on the grouped layout",
      "and Gated DeltaNet's learned forget gate on the eight-key header layout"))
# tex3:344 — the referee list
mrow("M11", 344, "referee list: 'that six delta splits fired before the map had merged (it had; five of the six copied the wrong channel)'",
     "merged at 6; five of six wrong channel", f"merged before at {nm_before} of 6; source c* held 0-1 streams at {len(src01)} of 6; the sixth {odd} copied a merged channel onto an occupied one",
     "MATCH" if nm_before == 6 and len(src01) == 5 else "DISAGREE", "replaces N26",
     "(it had; five of the six copied the wrong channel)")
mrow("M12", 344, "referee list: 'that a learned beta collapses (it rose in two screens)'", "rose in two", f"rose: {rose}; fell: {fell}; mixed: {mixed}",
     "MATCH" if rose == ["S54(a)", "S54(b)"] else "DISAGREE", "the first half of N26, unchanged", "that a learned $\\beta$ collapses (it rose in two screens)")
# tex3:355 — the stall
c54_seeds = sorted(runs(WS, "C_DELTA"))
mrow("M13", 355, "Withdrawn: 'the stall Revision 8 reported came from a screen with a learned write strength (and a different split and seeds; the attribution is post hoc)'",
     "learned beta; different split and seeds; post hoc", f"S54(c) C_DELTA beta = {c54d['delta_cfg']['beta']!r}, seeds {c54_seeds[0]}-{c54_seeds[-1]}, the noisy split with a reset "
     f"(explore_f_window_delta_scale.py:3-5, 12); S76 DELTA_SPLIT beta 1, seeds {min(DS)}-{max(DS)}, the exact copy (explore_f_eight.py:30-31); no S76 arm varies beta",
     "MATCH" if c54d["delta_cfg"]["beta"] == "learned" and min(DS) == 1310 else "DISAGREE", "marked now; second pass §3 item resolved",
     "came from a screen with a learned write strength (and a different split and seeds; the attribution is post hoc)")
# tex3:357 — C3 and S68
mrow("M14", 357, "'Supported on X by the pre-registered C3 (0/10 against the recipe's 10/10); session F's S68 had 0/10 at eight streams too (2/10 at four), but its own reading of that name did not apply'",
     "S68 0/10 at eight, 2/10 at four; reading did not apply", f"S68 S8 {k8}/10, S4 {k4}/10; reading '{r68}'; C3 (X, SINGLE_DELTA_D8 0/10 vs WIN3_SPLIT_D8 10/10) is checked in check_split_copy_v2.md",
     "MATCH" if (k4, k8, r68) == (2, 0, "neither") else "DISAGREE", "replaces N28; 'the same counts' is gone",
     "session~F's S68 had $0/10$ at eight streams too ($2/10$ at four), but its own reading of that name did not apply")
# tex3:438, 457 — the memory horizon
L59 = "L = 37" in git("show", f"{HEAD}:{[x for x in git('ls-tree', '-r', '--name-only', HEAD).split() if x.endswith('explore_f_decay.py')][0]}")
cond_arms = [k for k in DC["runs"] if k.startswith("D_DELTA")]
same_seeds = sorted(H69["FIXED"]) == sorted(S59["FIXED"])
hz = (f"S59 (delta): ROUTED {c59['ROUTED']}/10, GDN {c59['GDN']}/10 against FIXED {c59['FIXED']}/10; S69 (Hebbian): {c69['ROUTED']}/10, {c69['GDN']}/10 against {c69['FIXED']}/10; "
      f"S59's reading '{r59}' (gap_d - gap_FIXED < 0 on " + ", ".join(f"{d} {sum(1 for g in v if g < 0)}/10, median {med(v):+.2f}" for d, v in gaps.items())
      + f"); S59 at L = 37: {L59}; S69 on the same seeds and task: {same_seeds}; S59's conditional rerun at another length (D_DELTA_*) made: {bool(cond_arms)}")
hz_ok = (c59["ROUTED"], c59["GDN"], c59["FIXED"]) == (9, 8, 3) and all(v == 0 for v in c69.values()) and r59 == "it does not" and L59 and not cond_arms
mrow("M15", 438, "limits, the model: 'a learned forget gate, was screened once at a fixed length: it raised one delta channel's binding (9/10 and 8/10 against 3/10) "
     "and not a Hebbian one's, and its pre-fixed reading on the horizon read \"it does not\"'", "9/10, 8/10 vs 3/10; not Hebbian; reading 'it does not'; one fixed length",
     hz, "MATCH" if hz_ok else "DISAGREE",
     "replaces N33. Wording note: 'screened once' is two screens (S59 on delta, S69 on Hebbian) at the one length L = 37 on seeds 350-359; the sentence names both outcomes, so it reads as 'at one fixed length'",
     ("was screened once at a fixed length: it raised one delta channel's binding ($9/10$ and $8/10$ against $3/10$) and not a Hebbian one's",
      "and its pre-fixed reading on the horizon read ``it does not''"))
mrow("M16", 457, "limits list, memory horizon: 'a learned forget gate raises one delta channel's binding at a fixed length and not a Hebbian one's, its reading on the horizon itself read \"it does not\"'",
     "as M15", hz, "MATCH" if hz_ok else "DISAGREE", "replaces N34",
     "a learned forget gate raises one delta channel's binding at a fixed length and not a Hebbian one's, its reading on the horizon itself read ``it does not''")
# tex3:458 — baselines
SH = runs(DL, "SINGLE_HEBB_G")
mrow("M17", 458, "baselines: 'One channel of either write rule is run at two, four and eight streams'", "delta and Hebbian at 2, 4, 8",
     f"one delta channel: {cnt(bound, S53G)}/10, {k4}/10, {k8}/10 (S53, S68); one Hebbian channel in F's records: two streams only (S53 SINGLE_HEBB_G {cnt(bound, SH)}/{len(SH)}); "
     "four and eight streams are the main line's 0/20 and 0/10 (table tex3:278-280)", "MATCH" if (cnt(bound, S53G), k4, k8) == (4, 2, 0) else "DISAGREE",
     "F's half checked; the Hebbian 4- and 8-stream cells are main-line counts, outside F's records; tex2:458's 'at every size' is fixed",
     "One channel of either write rule is run at two, four and eight streams")
# S81 at the third draft
S81_3 = s81_state(f_at(THIRD_REF))
mrow("M18", 41, "'F's S81 ... running' at the third draft", "running, no number",
     f"F at {S81_3['ref']} ({S81_3['date']}), the last F commit before the third draft ({cdate(THIRD_REF)}): {S81_3['n']} S81 records {S81_3['arms']}, "
     f"outcomes {S81_3['outcomes']}; log 'S81 done' {S81_3['done']}; report 7 {S81_3['report']}",
     "MATCH" if S81_3["started"] and not S81_3["done"] and not S81_3["report"] else "DISAGREE",
     "no S81 number is quoted in the third draft (tex3:41, 203, 252, 470, 507); supersedes N37 and the first-pass provenance row for the third draft's time",
     ("F's S68--S81", "F's S81 are running"))

# rows whose fragment survives in tex3 while the sentence around it was reworded, so that the row's verdict is no longer
# about the text: superseded by the M row that checks the new wording
RESUP3 = {("first", "tex:252", "'Raw keys are excluded"): "M3", ("first", "tex:349", "'the stall Revision 8 reported"): "M13",
          ("first", "tex:501", "S81 running"): "M18", ("second", "N37", None): "M18"}
# rows removed in tex3 (fragment gone): the M row that checks the replacement
SUP3 = {("first", "tex:246", "'mostly as single splits"): "M2", ("second", "N13", None): "M1", ("second", "N25", None): "M10",
        ("second", "N26", None): "M11", ("second", "N28", None): "M14", ("second", "N33", None): "M15", ("second", "N34", None): "M16",
        ("second", "N36", None): "M9"}


def key3(r):
    if r["kind"] == "second":
        return ("second", r["id"], None)
    for (k, t, pre) in list(RESUP3) + list(SUP3):
        if k == "first" and t == r["tex"] and r["item"].startswith(pre):
            return (k, t, pre)
    return None


for r in ROWS + NROWS:
    k3 = key3(r)
    r["sup3"] = None
    r["state3"] = None
    if r.get("loc3") is None:            # report 6 rows
        r["loc3"] = r["loc"]
    if r["kind"] == "first" and r["loc"] == "REMOVED":
        r["state3"] = "removed in v2"
        continue
    if r["loc3"] == "REMOVED":
        assert k3 in SUP3, ("row removed in tex3 without a superseding M row", r["tex"], r.get("id"), r["item"][:60])
        r["sup3"], r["state3"] = SUP3[k3], "REMOVED in v3"
    elif k3 in RESUP3:
        r["sup3"] = RESUP3[k3]
        r["state3"] = ("SUPERSEDED in v3 (re-checked at the third draft's time)" if RESUP3[k3] == "M18"
                       else "SUPERSEDED in v3 (reworded around the fragment)")
    else:
        r["state3"] = "stands"
assert all(k in [key3(r) for r in ROWS + NROWS] for k in list(SUP3) + list(RESUP3)), "a SUP3/RESUP3 key matches no row"


# ══ fourth pass: locate every live third-pass row in the fourth draft; P rows for c66cc76's changes ══════════
PROWS = []


def prow(pid, tex4, item, draft, rec, status, note, q4):
    loc = find(TEX4, q4)
    assert tex4 in loc, ("fragment not at the cited fourth-draft line", pid, tex4, q4, loc)
    PROWS.append(dict(id=pid, tex=f"tex4:{tex4}", item=item, draft=str(draft), rec=str(rec), status=status, note=note,
                      q=q4, loc4=loc, kind="fourth", old=find(TEX3, q4)))


# S81 at the fourth draft
S81_4 = s81_state(f_at(FOURTH_REF))
s81_ok4 = S81_4["started"] and not S81_4["done"] and not S81_4["report"]
s81_rec4 = (f"F at {S81_4['ref']} ({S81_4['date']}), the last F commit before the fourth draft ({cdate(FOURTH_REF)}): "
            f"{S81_4['n']} S81 records {S81_4['arms']}, outcomes {S81_4['outcomes']}; log 'S81 done' {S81_4['done']}; "
            f"report 7 {S81_4['report']}; F's head now {HEAD} ({cdate(HEAD)}): report 7 {s81_state(HEAD)['report']}")
prow("P1", 41, "'F's S81 is running' at the fourth draft (the running list was rewritten)", "running, no number", s81_rec4,
     "MATCH" if s81_ok4 else "DISAGREE",
     "no S81 number is quoted in the fourth draft; supersedes M18. F's branch has only progress commits after ed76eea.",
     ("F's S68--S81", "F's S81 is running"))
prow("P2", 510, "F's provenance row: 'S81 running' and commit ed76eea, at the fourth draft", "running; ed76eea",
     s81_rec4 + f"; F commits after ed76eea up to the fourth draft add a report: {S81_4['report']}",
     "MATCH" if s81_ok4 else "DISAGREE", "unchanged row (tex3:507 -> tex4:510); re-read at the fourth draft's time",
     ("S53, S54, S59 (reports 1--2)", "S81 running"))
# test_delta_recipe: X's run start, from the main line's Amendment 2 (55dac50, dated after the fourth draft)
AM2_REF = "55dac50"
am2 = git("show", f"{AM2_REF}:specs/test_delta_recipe.md").replace("\n", " ")
am2_start = "after X's full run had started (02:12 UTC, 11 October, test commit a581027)" in " ".join(am2.split())
run_start = int(datetime.datetime(2026, 10, 11, 2, 12, tzinfo=datetime.timezone.utc).timestamp())
res_now = []
for b_ in [x.strip() for x in git("branch", "-r").split("\n") if x.strip() and "->" not in x]:
    for f_ in git("ls-tree", "-r", "--name-only", b_).split("\n"):
        if "delta_recipe" in f_ and (f_.endswith((".json", ".log")) or "results" in f_) and not f_.startswith("docs/"):
            res_now.append(f"{b_}:{f_}")
dr4 = (f"test {DR['test']} ({drs['test'][1]}); the test check {DR['tcheck']} ({drs['tcheck'][1]}); Amendment 2 {AM2_REF} ({cdate(AM2_REF)}) says X's full run "
       f"started 02:12 UTC with a581027: {am2_start}; the check after the start: {ctime(DR['tcheck']) > run_start}; "
       f"delta_recipe result or log files on any branch now: {res_now or 'none'}")
prow("P3", 41, "test_delta_recipe: 'its test file checked against the amended spec before any run'", "checked before any run", dr4,
     "DISAGREE" if am2_start and ctime(DR["tcheck"]) > run_start else "MATCH",
     "unchanged since a5f67a2 (M4 was MATCH on the commits then visible). By the main line's own Amendment 2, X's run "
     "started at 02:12 with a581027, 49 min before the referee's check 6f2788d (03:01:36). 55dac50 is dated after this "
     "version and written by the test's session, not the draft's.",
     "its test file checked against the amended spec before any run")
prow("P4", 54, "index row delta_recipe: '... not run'", "not run", dr4, "DISAGREE" if am2_start else "MATCH",
     "running on X since 02:12 per 55dac50; no result yet. Supersedes M5.",
     "the test written and checked against the amended spec; not run")
prow("P5", 254, "memory section: '(... with the test file checked against the amended spec; not run)'", "not run", dr4,
     "DISAGREE" if am2_start else "MATCH", "as P4. Supersedes M6.",
     "amended once on its review, and written, with the test file checked against the amended spec; not run)")
prow("P6", 492, "provenance row delta_recipe: 'spec 05256c5, Amendment 1 32c121b / test a581027 / not run & not run'",
     "05256c5; 32c121b; a581027; not run", dr4, "PARTIAL" if am2_start and not res_now else "MATCH",
     "the commits are right; no result commit exists; 'not run' in the X column is not the state (running since 02:12). Supersedes M7.",
     ("\\texttt{delta\\_recipe} & spec \\texttt{05256c5}, Amendment~1 \\texttt{32c121b} / test \\texttt{a581027} / not run"))
prow("P7", 472, "next steps: 'L's half of test_split_copy, then test_delta_recipe on both machines'", "test_delta_recipe next, after L",
     dr4, "PARTIAL" if am2_start else "MATCH", "X's half was already running at the fourth draft (per 55dac50)",
     "then \\texttt{test\\_delta\\_recipe} on both machines")
pb_body = "\n".join(pdt[pb_def[0] - 1:pb_def[0] + 20]) if pb_def else ""
pb_cond = "c[\"acc\"] >= tgw.ACC_THR and h < tgw.ACC_THR" in pb_body
prow("P8", 430, "method: 'with the count of checks at which the probe read >= 0.95 while the held-out accuracy was below it, as test_delta_recipe prints'",
     "the count test_delta_recipe prints", f"test_delta_recipe.py@{DR['test']}: probe_blind() at :{pb_def}, condition probe >= ACC_THR and "
     f"held-out < ACC_THR: {pb_cond}; printed at :{pb_prn}", "MATCH" if pb_def and pb_prn and pb_cond else "DISAGREE",
     "code, not records. Replaces M8 ('checks the probe blocked', the third pass's wording point): now the test's own "
     "definition. Its denominator includes the 2400 reference rows (test_check_delta_recipe.md, addendum).",
     "with the count of checks at which the probe read $\\geq 0.95$ while the held-out accuracy was below it, as \\texttt{test\\_delta\\_recipe} prints")
prow("P9", 441, "limits, the model: 'a learned forget gate, was screened on each write at one length: it raised one delta "
     "channel's binding (9/10 and 8/10 against 3/10) and not a Hebbian one's, and its pre-fixed reading on the horizon read \"it does not\"'",
     "two screens (delta, Hebbian) at one length; 9/10, 8/10 vs 3/10; not Hebbian; 'it does not'", hz + "; S59 at L = 37 and S69 at L = 37 "
     f"(explore_f_decay.py:8-9; explore_f_hebb_decay.py:8); seeds S59 {min(S59['FIXED'])}-{max(S59['FIXED'])}",
     "MATCH" if hz_ok else "DISAGREE",
     "replaces M15. The third pass's wording point ('screened once' for two screens) is met: 'on each write at one length' "
     "is S59 (delta) and S69 (Hebbian), both at L = 37 on seeds 350-359.",
     ("was screened on each write at one length: it raised one delta channel's binding ($9/10$ and $8/10$ against $3/10$) and not a Hebbian one's",
      "and its pre-fixed reading on the horizon read ``it does not''"))

# live third-pass rows -> fourth draft
LIVE3 = [r for r in ROWS + NROWS if r["state3"] == "stands"] + MROWS
# rows whose fragment survives in tex4 but whose verdict must be re-read with the record that appeared since (the
# delta_recipe run start) or at the fourth draft's time (S81): superseded by the P row
RESUP4 = {"M4": "P3", "M5": "P4", "M6": "P5", "M7": "P6"}
RESUP4_FIRST = {("tex:501", "S81 running"): "P2"}
SUP4 = {"M8": "P8", "M15": "P9", "M18": "P1"}


def key4(r):
    if r["kind"] == "first":
        for (t, pre), v in RESUP4_FIRST.items():
            if r["tex"] == t and r["item"].startswith(pre):
                return v
        return None
    return r["id"]


for r in LIVE3:
    q = r["q"]
    r["loc4"] = (find(TEX4, q) or "REMOVED") if q is not None else r["loc3"]
    k4 = key4(r)
    if r["loc4"] == "REMOVED":
        assert k4 in SUP4, ("row removed in tex4 without a superseding P row", r.get("id"), r["tex"], r["item"][:60])
        r["sup4"], r["state4"] = SUP4[k4], "REMOVED in v4"
    elif k4 in RESUP4 or k4 in RESUP4_FIRST.values():
        r["sup4"] = RESUP4.get(k4, k4)
        r["state4"] = "SUPERSEDED in v4 (re-checked with the records as of the fourth draft and Amendment 2)"
    else:
        r["sup4"], r["state4"] = None, "stands"
for k in list(SUP4) + list(RESUP4):
    assert any(r.get("id") == k for r in LIVE3), ("SUP4/RESUP4 key matches no live row", k)



# ══ fifth pass: S81 (report 7), every live fourth-pass row in the fifth draft, Q rows for c5e4508's changes ══════════
CUT5_ISO = git("show", "-s", "--format=%cI", FIFTH_REF).strip()
CUT5 = ctime(FIFTH_REF)
REMOTES = [x.strip() for x in git("branch", "-r").split("\n") if x.strip() and "->" not in x]
TIP5 = {b_: git("rev-list", "-1", f"--before={CUT5_ISO}", b_).strip()[:7] for b_ in REMOTES}
TIP5 = {b_: t_ for b_, t_ in TIP5.items() if t_}
R7 = "d5851b1"                                              # report 7 (S81), the commit F's provenance row names
F5 = TIP5[BR]
assert F5 == R7, ("F's last commit before the fifth draft is not report 7", F5)   # still true: TIP5 is read at the fifth draft's time
# (v5's 'assert HEAD == R7' is replaced: HEAD is pinned to R7 above; the stores' identity across F's later commits is checked in the sixth pass)
TG = load(R7, f"{F}trigger_results.json")
T = {a: runs(TG, a) for a in ("BASE", "GUARD", "EVAL", "GUARD_EVAL", "O_DELTA_C")}
LA = ("BASE", "GUARD", "EVAL", "GUARD_EVAL")
nrec_T = sum(1 for r in TG["runs"].values() if not r.get("dry"))
okT = sum(1 for r in TG["runs"].values() if r.get("ok"))


def brt(r):
    return bool(r.get("ok")) and outcome_k(r) == "BOUND ROUTED"


nT = {a: cnt(brt, T[a]) for a in LA}
agreeT = all(brt(r) == (r["outcome"] == "BOUND ROUTED") for a in LA for r in T[a].values())
orT = {s: transition(r["curve"]) for s, r in sorted(T["O_DELTA_C"].items())}
validT = len(orT) == 2 and all(v is not None for v in orT.values())
pT = {a: pairs(T[a], T["BASE"], brt) for a in ("GUARD", "EVAL", "GUARD_EVAL")}       # (arm only b, BASE only c, n)
dsT = {a: sorted(s for s in T["BASE"] if brt(T[a][s]) != brt(T["BASE"][s])) for a in pT}
GU = {"BASE": None, "GUARD": 0.5, "EVAL": None, "GUARD_EVAL": 0.5}


def targ(c):
    m = c["map_before"]
    return m.count(c["key_cs"]) >= 2 and m.count(c["key_c0"]) == 0


def elig(r_, p_, step, g):
    return step >= 4800 and p_ is not None and r_ < 0.95 and r_ - p_ < 0.02 and (g is None or r_ >= g)


def ep(c, a):
    return elig(c["acc_probe"], c["prev_probe"], c["step"], GU[a])


def ee(c, a):
    return elig(c["acc_eval"], c["prev_eval"], c["step"], GU[a])


FT = {a: [(s, c) for s, r in sorted(T[a].items()) for c in r["checks"] if c["fired"]] for a in LA}
labT = {a: sum(targ(c) for _, c in FT[a]) for a in LA}
labTr = {a: sum(bool(c["key_ok"]) for _, c in FT[a]) for a in LA}
runsF = {a: len({s for s, _ in FT[a]}) for a in LA}
srcT = {a: sorted({r.get("source") for r in T[a].values()}) for a in LA}
read_ok = all(c["acc"] == (c["acc_probe"] if srcT[a] == ["probe"] else c["acc_eval"]) for a in LA for r in T[a].values() for c in r["checks"])
elig_mism = sum((ep(c, a) != c["elig_probe"]) + (ee(c, a) != c["elig_eval"]) + ((ep(c, a) != ee(c, a)) != c["disagree"])
                for a in LA for r in T[a].values() for c in r["checks"])
disF = {a: sum(1 for _, c in FT[a] if ep(c, a) != ee(c, a)) for a in LA}
disC = {a: sum(1 for r in T[a].values() for c in r["checks"] if ep(c, a) != ee(c, a)) for a in LA}
disC_flat = {a: sum(1 for r in T[a].values() for c in r["checks"] if ep(c, a) != ee(c, a) and c["acc"] < 0.5) for a in LA}
nchk = {a: sum(len(r["checks"]) for r in T[a].values()) for a in LA}
held = {a: sum(1 for r in T[a].values() for c in r["checks"]
               if c["step"] >= 4800 and c["prev"] is not None and c["acc"] < 0.5 and c["acc"] - c["prev"] < 0.02 and not c["blocked"])
        for a in ("GUARD", "GUARD_EVAL")}
flatB = [(s, c) for s, c in FT["BASE"] if c["acc"] < 0.5]
flatB_runs = sorted({s for s, _ in flatB})
flatB_bound = [brt(T["BASE"][s]) for s in flatB_runs]
flatB_lab = sum(targ(c) for _, c in flatB)
flatB_steps = (min(c["step"] for _, c in flatB), max(c["step"] for _, c in flatB)) if flatB else None
lateB = [(s, c["step"]) for s, c in FT["BASE"] if c["acc"] >= 0.5]
gu_f = [(s, c["step"]) for s, c in FT["GUARD"]]
m914 = {a: [(s, c["step"], c["acc_probe"], c["acc_eval"]) for s, r in sorted(T[a].items()) for c in r["checks"]
            if c["acc_probe"] >= 0.95 and c["acc_eval"] < 0.95] for a in LA}
g25 = m914["GUARD"]
g25_ok = (len(g25) == 5 and {s for s, *_ in g25} == {2025} and f"{min(x[2] for x in g25):.2f}" == "0.95"
          and f"{max(x[2] for x in g25):.2f}" == "0.98" and f"{min(x[3] for x in g25):.2f}" == "0.87"
          and f"{max(x[3] for x in g25):.2f}" == "0.92" and f"{T['GUARD'][2025]['acc']:.2f}" == "0.93"
          and outcome_k(T["GUARD"][2025]).startswith("MERGED"))
trT = {a: sorted(transition(r["curve"]) for r in T[a].values() if brt(r)) for a in LA}
secsT = sum(r["secs_wall"] for r in TG["runs"].values()) / 3600
secsT_cpu = sum(r["secs"] for r in TG["runs"].values()) / 3600
unb = {a: sorted((s, outcome_k(r), r.get("outcome"), r.get("outcome_v2")) for s, r in T[a].items() if not brt(r)) for a in LA}
b_g, c_g = pT["GUARD"][0], pT["GUARD"][1]
b_e, c_e = pT["EVAL"][0], pT["EVAL"][1]
b_ge, c_ge = pT["GUARD_EVAL"][0], pT["GUARD_EVAL"][1]
aims = bool(FT["GUARD"]) and labT["GUARD"] >= 0.9 * len(FT["GUARD"]) and c_g - b_g <= 1
better = c_e - b_e <= 1 and disF["EVAL"] == 0
better_H = c_e - b_e <= 1                                             # H's S84 rule, no disagreement clause
g_costs, e_costs, both_costs = (c_g >= 4 and b_g == 0), (c_e >= 4 and b_e == 0), (c_ge >= 4 and b_ge == 0)
none_applies = not (aims or better or g_costs or e_costs)
# the readings' commit against the first S81 record (git log order)
rc81 = first_reading_commit("explore_f_trigger.py", '"the guard aims the split"')
fr81, nrec81 = first_record_commit([f"{F}trigger_results.json"], list(T))
unch81 = git("diff", rc81, R7, "--", "explore_f_trigger.py").strip() == ""
child_after = git("log", "--format=%h", f"{rc81}..{R7}", "--", "explore_f_trigger_child.py").split()
rg81 = sorted({r["git"].split("+")[0] for r in TG["runs"].values() if r.get("git")})
desc81 = all(is_anc(rc81, g) for g in rg81)
prec81 = ctime(rc81) < ctime(fr81) and is_anc(rc81, fr81)
CK81 = [x for x in git("ls-tree", "-r", "--name-only", R7, F).split() if "checks_s81" in x]
ck81 = json.loads(git("show", f"{R7}:{CK81[0]}")) if CK81 else {}
ck81_ok = bool(ck81.get("ok")) and all(v for _, v in ck81.get("rows", []))
ck81_start = ck81.get("provenance", {}).get("started")
log81 = git("show", f"{R7}:{F}s81.log")
log_start = [ln for ln in log81.split("\n") if ln.startswith("=== S81 start")]
log_done = [ln for ln in log81.split("\n") if ln.startswith("=== S81 done")]
# H's S84 and S86 (the draft's 29/30 against 27/30 over three screens), at H's report commits
HS = {"S84": ("37c1fbf", "guard", (2300, 2309)), "S86": ("9b72bf6", "allmass", (2330, 2339))}
HE = {}
for k_, (ref_, nm_, _) in HS.items():
    st_ = load(ref_, f"explore_out/H/{nm_}_results.json")
    HE[k_] = (runs(st_, "EVAL"), runs(st_, "BASE"))
tri = {k_: (sum(brt(r) for r in e_.values()), sum(brt(r) for r in b_.values()), pairs(e_, b_, brt))
       for k_, (e_, b_) in HE.items()}
tri["S81"] = (nT["EVAL"], nT["BASE"], pT["EVAL"])
tri_e, tri_b = sum(v[0] for v in tri.values()), sum(v[1] for v in tri.values())
tri_bc = (sum(v[2][0] for v in tri.values()), sum(v[2][1] for v in tri.values()))
tri_n = sum(v[2][2] for v in tri.values())
# S76's DELTA_SPLIT labelled rate (report 7 compares 10/19 with it)
S76F = [c for r in runs(EI, "DELTA_SPLIT").values() for c in r.get("checks", []) if c.get("fired")]
s76_lab = (sum(targ(c) for c in S76F), len(S76F))

# ── report 7, every number ─────────────────────────────────────────────────
R7F = "report_7.md"
R7T = git("show", f"{R7}:{F}{R7F}").split("\n")
R7ROWS = []


def r7ln(text):
    h = [i + 1 for i, ln in enumerate(R7T) if text in ln]
    assert h, ("not in report 7", text)
    return h[0]


def r7row(text, item, says, rec, status, note=""):
    R7ROWS.append(dict(tex=f"{R7F}:{r7ln(text)}", item=item, draft=str(says), rec=str(rec), status=status, note=note,
                       kind="report7", q=None, loc5=R7F))


codeT = sorted({r["code_sha"][:12] for r in TG["runs"].values()})
mainT = sorted({r.get("main_sha", "")[:7] for r in TG["runs"].values()})
cpuT, torchT = sorted({r["cpu"] for r in TG["runs"].values()}), sorted({r["torch"] for r in TG["runs"].values()})
attT = sorted({r["attempts"] for r in TG["runs"].values()})
r7row("Readings committed at 451cde9 before any stored run", "readings committed before any stored run", "451cde9, before any stored run",
      f"first commit holding the READINGS text: {rc81} ({cdate(rc81)}); first commit with an S81 record: {fr81} ({cdate(fr81)}, "
      f"{nrec81} records); the readings commit is an ancestor of it: {is_anc(rc81, fr81)}; explore_f_trigger.py unchanged "
      f"{rc81}..{R7}: {unch81}; explore_f_trigger_child.py changed after it: {child_after or 'never'}; every record's code "
      f"commit {rg81} descends from {rc81}: {desc81}; s81.log '{log_start[0] if log_start else '?'}'; CHECK file started {ck81_start}",
      "MATCH" if rc81 == "451cde9" and prec81 and unch81 and desc81 and not child_after else "DISAGREE",
      "git log order: the readings (19:28:32 UTC, 10 October) precede the segment's start (20:41:45) and the first record commit. "
      "The report's own disclosure (one unstored 2400-update smoke test of GUARD_EVAL on seed 2029 before the readings' commit, "
      "the readings already written) is not checkable from git.")
r7row("Code SHA bdd8269d38af", "code SHA; main-line modules; CPU; torch", "bdd8269d38af; 9c5939e; Xeon 2.10GHz; 2.14.0+cu130",
      f"{codeT}; {mainT}; {cpuT}; {torchT}",
      "MATCH" if codeT == ["bdd8269d38af"] and mainT == ["9c5939e"] and len(cpuT) == 1 and "2.10GHz" in cpuT[0] and torchT == ["2.14.0+cu130"] else "DISAGREE")
r7row("42/42 runs ok, none retried, none diverged", "42/42 ok, none retried", "42/42, attempts 1",
      f"{okT}/{nrec_T} ok; attempts {attT}; diverged {[k for k, r in TG['runs'].items() if r.get('diverged')]}",
      "MATCH" if okT == nrec_T == 42 and attT == [1] else "DISAGREE")
r7row("seeds 2020-2029, paired", "seeds", "2020-2029, paired", f"{ {a: (min(T[a]), max(T[a]), len(T[a])) for a in LA} }",
      "MATCH" if all(sorted(T[a]) == list(range(2020, 2030)) for a in LA) else "DISAGREE")
for a, says, wil_s, band_s in (("BASE", "9/10", "[0.60, 0.98]", "RELIABLE"), ("GUARD", "7/10", "[0.40, 0.89]", "MAJORITY"),
                               ("EVAL", "10/10", "[0.72, 1.00]", "RELIABLE"), ("GUARD_EVAL", "8/10", "[0.49, 0.94]", "MAJORITY")):
    rec_ = f"{nT[a]}/10 {wil(nT[a], 10)} {band(nT[a], 10)}"
    r7row(f"| {a} (" if a != "GUARD_EVAL" else "| GUARD_EVAL (both)", f"{a}: BOUND ROUTED, Wilson, band", f"{says} {wil_s} {band_s}", rec_,
          eq(f"{says} {wil_s} {band_s}", rec_), f"rlib.outcome_k == the record's outcome on every S81 run: {agreeT}")
r7row("| GUARD (probe, no split below 0.5)", "GUARD vs BASE: arm only / BASE only; p", "0 / 2 (2020, 2025); p = 0.25",
      f"{b_g} / {c_g} {dsT['GUARD']}; p(BASE > GUARD) = {mcnemar1(c_g, b_g):.3g}",
      "MATCH" if (b_g, c_g, dsT["GUARD"]) == (0, 2, [2020, 2025]) and abs(mcnemar1(c_g, b_g) - 0.25) < 1e-9 else "DISAGREE")
r7row("| EVAL (2048-sequence evaluation)", "EVAL vs BASE: arm only / BASE only; p", "1 (2029) / 0; p = 0.5",
      f"{b_e} / {c_e} {dsT['EVAL']}; p(EVAL > BASE) = {mcnemar1(b_e, c_e):.3g}",
      "MATCH" if (b_e, c_e, dsT["EVAL"]) == (1, 0, [2029]) and abs(mcnemar1(b_e, c_e) - 0.5) < 1e-9 else "DISAGREE")
r7row("| GUARD_EVAL (both)", "GUARD_EVAL vs BASE", "1 (2029) / 2 (2020, 2025)", f"{b_ge} / {c_ge} {dsT['GUARD_EVAL']}",
      "MATCH" if (b_ge, c_ge) == (1, 2) and dsT["GUARD_EVAL"] == [2020, 2025, 2029] else "DISAGREE")
for a, says in (("BASE", "19; 9 | 10/19 | 7"), ("GUARD", "6; 3 | 6/6 | 0"), ("EVAL", "13; 9 | 8/13 | 5"), ("GUARD_EVAL", "9; 3 | 9/9 | 4")):
    rec_ = f"{len(FT[a])}; {runsF[a]} | {labT[a]}/{len(FT[a])} | {disF[a]}"
    r7row(f"| {a} (" if a != "GUARD_EVAL" else "| GUARD_EVAL (both)", f"{a}: firings (total; runs) | labelled ok | at a disagreement", says, rec_,
          eq(says, rec_), f"labelled ok recomputed with s37c.targeting on map_before (recorded key_ok {labTr[a]}); disagreement recomputed "
          f"from both readings and their previous values with the arm's guard (recorded eligibility and 'disagree' fields: "
          f"{elig_mism} mismatches over every check of every arm); each arm's rule reads its own source: {read_ok}")
for a, says in (("BASE", "15600 (7200-32400)"), ("GUARD", "13200 (7200-15600)"), ("EVAL", "15600 (7200-31200)"), ("GUARD_EVAL", "13200 (7200-31200)")):
    rec_ = f"{med(trT[a]):.0f} ({min(trT[a])}-{max(trT[a])})"
    r7row(f"| {a} (" if a != "GUARD_EVAL" else "| GUARD_EVAL (both)", f"{a}: transitions (median; range)", says, rec_, eq(says, rec_))
r7row("**Validity:**", "validity: the delta perfect gate bound on 2020 (6000) and 2021 (1200)", "6000, 1200; VALID", f"{orT}; VALID {validT}",
      "MATCH" if orT == {2020: 6000, 2021: 1200} else "DISAGREE")
r7row("**Failures:**", "every unbound run MERGED (2 share), v1 and v2; BASE 2029; GUARD 2020, 2025, 2029; GUARD_EVAL 2020, 2025",
      "as listed", "(seed, rlib outcome, recorded v1 outcome, recorded v2 outcome) " + "; ".join(f"{a}: {unb[a]}" for a in LA),
      "MATCH" if [s for s, *_ in unb["BASE"]] == [2029] and [s for s, *_ in unb["GUARD"]] == [2020, 2025, 2029]
      and [s for s, *_ in unb["GUARD_EVAL"]] == [2020, 2025] and not unb["EVAL"]
      and all(x[1] == "MERGED (2 share)" and x[2] == "MERGED (2 share)" and x[3] == "MERGED (2 share)" for a in LA for x in unb[a]) else "DISAGREE")
r7row("**\"The guard aims the split\": does not apply.**", "'the guard aims the split' does not apply (6/6 ok, but c - b = 2 > 1)",
      "does not apply", f"labelled {labT['GUARD']}/{len(FT['GUARD'])} (>= 90%: {labT['GUARD'] >= 0.9 * len(FT['GUARD'])}); c - b = {c_g - b_g}; applies: {aims}",
      "MATCH" if not aims and c_g - b_g == 2 else "DISAGREE", "explore_f_trigger.py READINGS")
r7row("**\"The guard costs binding\": does not apply**", "'the guard costs binding' does not apply (c = 2 < 4)", "does not apply",
      f"c = {c_g}, b = {b_g}; applies: {g_costs}", "MATCH" if not g_costs else "DISAGREE")
r7row("**\"The evaluation is the better probe\": does not apply.**", "'the evaluation is the better probe' does not apply (b = 1, c = 0; 5 of 13 at a disagreement)",
      "does not apply", f"b = {b_e}, c = {c_e}, c - b <= 1: {c_e - b_e <= 1}; EVAL firings at a disagreement {disF['EVAL']} of {len(FT['EVAL'])}; applies: {better}",
      "MATCH" if not better and disF["EVAL"] == 5 and len(FT["EVAL"]) == 13 else "DISAGREE")
r7row("**\"The evaluation costs binding\": does not apply**", "'the evaluation costs binding' does not apply (c = 0)", "does not apply",
      f"c = {c_e}; applies: {e_costs}", "MATCH" if not e_costs else "DISAGREE")
r7row("**So: no reading applies to either device.**", "no reading applies to either device; GUARD_EVAL not 'both cost binding' (c = 2)",
      "none", f"none applies: {none_applies}; GUARD_EVAL c = {c_ge}, 'both devices cost binding': {both_costs}",
      "MATCH" if none_applies and not both_costs and c_ge == 2 else "DISAGREE")
r7row("Under the simpler rule Session H's S84 uses", "under H's S84 rule (EVAL >= BASE - 1 pair) EVAL would read 'the evaluation is the better probe'",
      "would apply", f"c - b = {c_e - b_e} <= 1: {better_H}", "MATCH" if better_H else "DISAGREE")
ev5 = sorted((s, c["step"]) for s, c in FT["EVAL"] if ep(c, "EVAL") != ee(c, "EVAL"))
r7row("Of EVAL's 5 such firings, 3 were in the flat start", "EVAL's 5 disagreement firings: 2022@9600, 2027@4800, 2028@4800 (flat start); 2026@19200, 2029@21600",
      "as listed", f"{ev5}", "MATCH" if ev5 == [(2022, 9600), (2026, 19200), (2027, 4800), (2028, 4800), (2029, 21600)] else "DISAGREE")
r7row("BASE made 13 of its 19 firings while accuracy was still below 0.5", "BASE: 13 of 19 firings below 0.5 (4800-12000), 6 labelled ok; all 8 runs that split there bound",
      "13 of 19; 4800-12000; 6 ok; 8 runs, all bound", f"{len(flatB)} of {len(FT['BASE'])}; {flatB_steps}; {flatB_lab} ok; runs {flatB_runs} bound routed {flatB_bound}",
      "MATCH" if len(flatB) == 13 and flatB_steps == (4800, 12000) and flatB_lab == 6 and len(flatB_runs) == 8 and all(flatB_bound) else "DISAGREE",
      "'below 0.5' read on the reading the rule used (the probe, for BASE)")
r7row("13 eligible checks were held back in GUARD and 18 in GUARD_EVAL", "checks held back by the guard: GUARD 13, GUARD_EVAL 18", "13; 18",
      f"{held['GUARD']}; {held['GUARD_EVAL']}", eq("13; 18", f"{held['GUARD']}; {held['GUARD_EVAL']}"),
      "a check is held back if, on the arm's own reading, update >= 4800, a previous reading, rise < 0.02, reading < 0.5, not gap/cap-blocked")
gl = sorted((a, s, c["step"], targ(c)) for a in ("GUARD", "GUARD_EVAL") for s, c in FT[a] if s in (2020, 2025))
r7row("Their late splits (31200-40800, all labelled ok)", "GUARD 2020, 2025: later splits 31200-40800, all labelled ok", "31200-40800, all ok",
      f"{gl}", "MATCH" if gl and all(x[3] for x in gl) and min(x[2] for x in gl) >= 28800 and max(x[2] for x in gl) == 40800 else "DISAGREE",
      "the range spans GUARD (31200-36000) and GUARD_EVAL (28800-40800) together; GUARD alone ends at 36000 and GUARD_EVAL 2025 fired first at 28800")
r7row("At 5 checks between 28800 and 40800 the probe read 0.95-0.98", "GUARD 2025: 5 checks 28800-40800, probe 0.95-0.98, evaluation 0.87-0.92; one fire at 31200 at 0.94; MERGED 0.93",
      "as stated", f"GUARD probe >= 0.95 with evaluation < 0.95: {[(s, u, round(p, 3), round(e, 3)) for s, u, p, e in g25]}; GUARD 2025 fired at "
      f"{[(c['step'], round(c['acc_probe'], 3)) for s, c in FT['GUARD'] if s == 2025]}; end {outcome_k(T['GUARD'][2025])} {T['GUARD'][2025]['acc']:.3f}",
      "MATCH" if g25_ok and [(c["step"], round(c["acc_probe"], 2)) for s, c in FT["GUARD"] if s == 2025] == [(31200, 0.94)] else "DISAGREE")
r7row("In BASE it occurred at 2 checks (2025 at 14400, 2027 at 31200)", "BASE: the 914 mode at 2025@14400 and 2027@31200, both shortly before those runs bound",
      "2 checks", f"{[(s, u) for s, u, *_ in m914['BASE']]}; BASE transitions 2025 {transition(T['BASE'][2025]['curve'])}, 2027 {transition(T['BASE'][2027]['curve'])}",
      "MATCH" if [(s, u) for s, u, *_ in m914["BASE"]] == [(2025, 14400), (2027, 31200)] else "DISAGREE",
      f"not in the report: GUARD_EVAL 2025 shows the same mode at {len(m914['GUARD_EVAL'])} checks (the rule there reads the evaluation)")
ge25 = [(c["step"], targ(c)) for s, c in FT["GUARD_EVAL"] if s == 2025]
r7row("With the evaluation as the reading (GUARD_EVAL 2025)", "GUARD_EVAL 2025: three splits, all labelled ok; merged (0.92)", "3, all ok; 0.92",
      f"{ge25}; end {outcome_k(T['GUARD_EVAL'][2025])} {T['GUARD_EVAL'][2025]['acc']:.3f}",
      "MATCH" if len(ge25) == 3 and all(x[1] for x in ge25) and f"{T['GUARD_EVAL'][2025]['acc']:.2f}" == "0.92" else "DISAGREE")
e29 = [c for s, c in FT["EVAL"] if s == 2029 and c["step"] == 21600]
b29 = [c["step"] for s, c in FT["BASE"] if s == 2029]
g29 = [c["step"] for s, c in FT["GUARD"] if s == 2029]
r7row("**EVAL's one win (2029).**", "EVAL 2029: BASE and GUARD fired at 14400, 24000, 28800; EVAL's second split at 21600, evaluation 0.868 -> 0.870, probe 0.75 -> 0.78; bound 31200, as GUARD_EVAL",
      "as stated", f"BASE {b29}, GUARD {g29}; EVAL@21600 eval {e29[0]['prev_eval']:.3f} -> {e29[0]['acc_eval']:.3f}, probe {e29[0]['prev_probe']:.3f} -> "
      f"{e29[0]['acc_probe']:.3f}; EVAL 2029 bound at {transition(T['EVAL'][2029]['curve'])}, GUARD_EVAL at {transition(T['GUARD_EVAL'][2029]['curve'])}" if e29 else "no firing",
      "MATCH" if e29 and b29 == g29 == [14400, 24000, 28800] and f"{e29[0]['prev_eval']:.3f}" == "0.868" and f"{e29[0]['acc_eval']:.3f}" == "0.870"
      and f"{e29[0]['prev_probe']:.2f}" == "0.75" and f"{e29[0]['acc_probe']:.2f}" == "0.78"
      and transition(T["EVAL"][2029]["curve"]) == transition(T["GUARD_EVAL"][2029]["curve"]) == 31200 else "DISAGREE")
r7row("**BASE's labelled-ok rate:** 10/19 (53%), as in S76 (6/12)", "BASE 10/19 labelled ok; S76 6/12", "10/19; 6/12",
      f"{labT['BASE']}/{len(FT['BASE'])}; S76 DELTA_SPLIT {s76_lab[0]}/{s76_lab[1]} (eight_results.json)",
      "MATCH" if (labT["BASE"], len(FT["BASE"])) == (10, 19) and s76_lab == (6, 12) else "DISAGREE")
r7row("## CHECKs (all passed, asserted before any run)", "CHECKs all passed", "all passed",
      f"{CK81[0].split('/')[-1] if CK81 else 'no CHECK file'}: ok {ck81.get('ok')}, {len(ck81.get('rows', []))} rows all true: {ck81_ok}; "
      f"started {ck81_start}", "MATCH" if ck81_ok else "DISAGREE", "the file is the segment's CHECK record; the bit-for-bit claims are read from it, not re-run")
r7row("69.2 h of run time", "69.2 h of run time; 20:41 UTC 10 October to 17:37 UTC 11 October", "69.2 h; 20:41 to 17:37",
      f"sum of secs_wall {secsT:.2f} h (of secs, {secsT_cpu:.2f} h); s81.log '{log_start[0] if log_start else '?'}' / '{log_done[0] if log_done else '?'}'",
      "MATCH" if f"{secsT:.1f}" == "69.2" and log_start and "20:41" in log_start[0] and log_done and "17:37" in log_done[0] else "DISAGREE",
      "69.2 h is the sum of the records' secs_wall (explore_f_reports4.py's 'h of run time'); their secs sum to 68.9 h")
r7row("BASE and EVAL had 20 and 17 disagreeing checks, 12 of each in the flat start", "disagreeing checks: BASE 20, EVAL 17, 12 of each below 0.5",
      "20, 17; 12 each", f"{disC['BASE']}, {disC['EVAL']}; below 0.5 on the arm's reading {disC_flat['BASE']}, {disC_flat['EVAL']}",
      "MATCH" if (disC["BASE"], disC["EVAL"], disC_flat["BASE"], disC_flat["EVAL"]) == (20, 17, 12, 12) else "DISAGREE")
r7row("The guarded arms had 10 and 9, none in the flat start", "guarded arms 10 and 9, none in the flat start", "10, 9; none",
      f"{disC['GUARD']}, {disC['GUARD_EVAL']}; below 0.5 {disC_flat['GUARD']}, {disC_flat['GUARD_EVAL']}",
      "MATCH" if (disC["GUARD"], disC["GUARD_EVAL"], disC_flat["GUARD"], disC_flat["GUARD_EVAL"]) == (10, 9, 0, 0) else "DISAGREE")

# ── fifth-draft rows ───────────────────────────────────────────────────────
QROWS = []


def qrow(qid, tex5, item, draft, rec, status, note, q5):
    loc = find(TEX5, q5)
    assert tex5 in loc, ("fragment not at the cited fifth-draft line", qid, tex5, q5, loc)
    QROWS.append(dict(id=qid, tex=f"tex5:{tex5}", item=item, draft=str(draft), rec=str(rec), status=status, note=note,
                      q=q5, loc5=loc, kind="fifth", old=find(TEX4, q5)))


MAIN = "origin/claude/bdh-growth-hebbian-inference-w90069"
ML5 = TIP5[MAIN]
XREC = "results/X/delta_recipe_results.json"
x_add = git("log", "--diff-filter=A", "--format=%h", ML5, "--", XREC).split()
XS = load(ML5, XREC)
xm = XS["meta"]
x_arms = dict(Counter(k.split("|")[0] for k in XS["runs"]))
x_seeds = sorted({int(k.split("|")[1]) for k in XS["runs"] if not k.startswith("CEIL")})
x_doc = git("show", f"{ML5}:test_delta_recipe.py")
x_doc_start = "The run started 2026-10-11 02:12 UTC" in x_doc and "runs began at 05:27 after the verification" in x_doc
x_runs_began = datetime.datetime.strptime(xm["started"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc)
t_rb = int(x_runs_began.timestamp())
t_x = ctime(x_add[0]) if x_add else None
L_DR = {b_: [f_ for f_ in git("ls-tree", "-r", "--name-only", t_).split("\n") if f_.startswith("results/L/") and "delta_recipe" in f_]
        for b_, t_ in TIP5.items()}
L_DR = {b_: v for b_, v in L_DR.items() if v}
am1_before = ctime(DR["amend"]) < run_start
dr5 = (f"X's result {x_add} ({cdate(x_add[0]) if x_add else '-'}) adds {XREC}: {sum(x_arms.values())} records {x_arms}, seeds {x_seeds[0]}-{x_seeds[-1]}, "
       f"meta.git {xm['git']}, meta.started (the start of the runs, after the verification; X records UTC) {xm['started']}; the test's RESULT (X) "
       f"docstring: 'The run started 2026-10-11 02:12 UTC ...; runs began at 05:27 after the verification': {x_doc_start}; Amendment 1 "
       f"{DR['amend']} ({cdate(DR['amend'])}) before the 02:12 start: {am1_before}; Amendment 2 {AM2_REF} ({cdate(AM2_REF)}) between the runs' "
       f"start and the result: {t_rb < ctime(AM2_REF) < (t_x or 0)}; L delta_recipe files on any branch at the fifth draft: {L_DR or 'none'}")
dr5_ok = bool(x_add) and t_x < CUT5 and sum(x_arms.values()) == 84 and xm["git"] == DR["test"] and x_doc_start and am1_before and not L_DR
qrow("Q1", 41, "test_delta_recipe: 'specified, reviewed by the referee, amended once before any run and written, its test file checked against "
     "the amended spec, and complete on X (a second amendment, to the report's rules only, was committed during X's run ...); L's half is next'",
     "amended once before any run; checked; complete on X; Amendment 2 during the run; L next", dr5, "MATCH" if dr5_ok else "DISAGREE",
     "Supersedes P3. 'Before any run' now qualifies Amendment 1 only (20:47, 10 October). CORRECTION to P3 and fourth_pass.md §1 item 6: "
     f"the arms' runs began at {xm['started']} UTC (meta.started; the RESULT docstring), after the referee's check "
     f"{DR['tcheck']} ({cdate(DR['tcheck'])}); 02:12 was the start of the process, whose first 3 h 15 min were the verification. "
     "tex4:41's 'checked against the amended spec before any run' was therefore true of the test's runs; P3's DISAGREE is withdrawn.",
     ("amended once before any run and written, its test file checked against the amended spec, and complete on X",
      "L's half is next"))
s81_5 = s81_state(F5)
qrow("Q2", 41, "'F's S81, G's S83, H's S86 and R's Phase 4 are complete' (F's part)", "S81 complete",
     f"F at {F5} ({cdate(F5)}), the last F commit before the fifth draft ({cdate(FIFTH_REF)}): {s81_5['n']} S81 records, log 'S81 done' "
     f"{s81_5['done']}, report 7 {s81_5['report']}", "MATCH" if s81_5["done"] and s81_5["report"] and s81_5["n"] == 42 else "DISAGREE",
     "Supersedes P1. G, H and R are other checks'.", ("F's S81, G's S83, H's S86 and R's Phase~4 are complete",))
E_BR, I_BR = "origin/claude/outside-ideas", "origin/claude/explore-I"
e_after = git("log", "--format=%h %cI %s", f"e3f5809..{TIP5[E_BR]}").strip().split("\n")
e_b20 = [x for x in e_after if "batch 20" in x]
i_last = git("log", "-1", "--format=%h %cI %s", TIP5[I_BR]).strip()
qrow("Q3", 41, "'every session's queue is empty'", "every queue empty",
     f"E ({E_BR.split('/')[-1]}): {len(e_b20)} 'batch 20' commits after e3f5809 before the fifth draft, the last '{e_b20[0][:110] if e_b20 else '-'}'; "
     f"I: last commit before the fifth draft '{i_last[:120]}'; the draft itself (tex5:458): 'on the nine seeds run so far (the tenth is queued)'",
     "DISAGREE" if e_b20 or "pending" in i_last else "MATCH",
     "New in c5e4508. Session E was running batch 20 (progress commits to 17:49 UTC on 12 October) and session I's tenth WIDE_SLOW "
     "seed (839) was pending, as the draft's own S75 paragraph says. 'F's queue' is empty (no F commit after report 7).",
     ("every session's queue is empty",))
d54_ok = "D4 SHOWN 10 vs 1" in TEX5[53] and "pooled labels and readings wait for L" in TEX5[53]
qrow("Q4", 54, "index row delta_recipe: '(specified) ... on X (secondary): D4 SHOWN 10 vs 1, D2 0 vs 0 ..., D1 d = -3; pooled labels and "
     "readings wait for L'", "X secondary results; pooled waits for L; label '(specified)'", dr5,
     "PARTIAL" if dr5_ok else "DISAGREE",
     "Supersedes P4. The status is right (X complete at 8ae660d, L not run). The row's label '(specified)' is stale beside "
     "split_copy's '(X, L, pooled)': '(X; L to run)' would match. The D numbers are the X-results check's, not this one's.",
     ("\\texttt{delta\\_recipe} (specified)", "pooled labels and readings wait for L"))
qrow("Q5", 256, "memory section: '(... with the test file checked against the amended spec; running on X, where a second amendment ... was "
     "committed during the run ...)'", "running on X", dr5, "PARTIAL",
     "Supersedes P5. 'Running on X' is not the state at c5e4508: X's run ended before 8ae660d (07:25:51 UTC, 12 October), and "
     "the same paragraph later says 'X's half is complete'. Read as the time of the amendment it is true; as written it is stale.",
     ("with the test file checked against the amended spec; running on X, where a second amendment confined to the report's rules",))
am2t = git("show", f"{AM2_REF}:specs/test_delta_recipe.md")
am2_lines = ("the four perfect-gate runs (CEIL_HEBB 580, 581; CEIL_DELTA 580, 581)" in am2t and "19 of the 20 DELTA_SPLIT runs (580-598)" in am2t
             and "DELTA_SLOW|580" in am2t and "None of HEBB_SPLIT or HEBB_SLOW had finished" in am2t)
qrow("Q6", 256, "'a statement of the 24 run lines that had been read, none of them a claim, a reading or a Hebbian run'",
     "24 lines; none Hebbian", f"Amendment 2 ({AM2_REF}): 24 lines = CEIL_HEBB 580, 581; CEIL_DELTA 580, 581; DELTA_SPLIT 580-598; DELTA_SLOW|580; "
     f"'None of HEBB_SPLIT or HEBB_SLOW had finished': {am2_lines}", "DISAGREE" if am2_lines else "NOT RECORD-CHECKABLE",
     "New in c5e4508. Two of the 24 lines were Hebbian runs: the Hebbian perfect gate CEIL_HEBB 580 and 581. The amendment says none "
     "of the Hebbian learned arms (HEBB_SPLIT, HEBB_SLOW) had finished; 'none of them ... a Hebbian learned-arm run' would be exact.",
     ("the 24 run lines that had been read, none of them a claim, a reading or a Hebbian run",))
qrow("Q7", 256, "'X's half is complete (seeds 580--599, 84 runs, 29.2 h; ...)' (status and run count only)", "complete; 580-599; 84 runs",
     dr5 + f"; 02:12 + 29.2 h = {datetime.datetime.fromtimestamp(run_start + 29.2 * 3600, datetime.timezone.utc):%Y-%m-%d %H:%M} UTC",
     "MATCH" if dr5_ok and x_seeds == list(range(580, 600)) else "DISAGREE",
     "Status, seeds and run count. The 29.2 h is the RESULT docstring's wall clock from the process start (02:12, the 3 h 15 min of "
     "verification included), not the runs' time. Every count, claim and reading of X's half is the X-results check's.",
     ("X's half is complete (seeds 580--599, 84 runs, 29.2 h",))
qrow("Q8", 256, "'(S68, whose reading of that name did not apply; the pre-registered support for the partition is C3 on X)'", "C3 on X",
     "C3 pooled over X and L: SHOWN, 18/20 vs 0/20, 18 vs 0 (tex5:246; ffb0771); S68 reading 'neither' (recomputed: " + r68 + ")",
     "PARTIAL", "Unchanged text made stale by ef6d690 (test_split_copy complete on L and pooled): tex5:246 and tex5:362 now cite C3 "
     "pooled and on each machine; this parenthesis still says 'C3 on X'.", ("the pre-registered support for the partition is C3 on X",))
qrow("Q9", 476, "next steps: 'test_delta_recipe's L half (X's is complete), then the hinged-register test ...'", "L half next; X complete",
     dr5, "MATCH" if dr5_ok else "DISAGREE", "Supersedes P7 (the fourth pass's §2 bullet tex4:472 is resolved).",
     ("\\texttt{test\\_delta\\_recipe}'s L half (X's is complete)",))
qrow("Q10", 477, "next steps: 'session~F's trigger on delta channels (S81)'", "S81 a next step",
     f"S81 complete: report 7 {R7} ({cdate(R7)}), folded in at tex5:205 by 3b06af2 ({cdate('3b06af2')}); no reading applies",
     "PARTIAL", "Unchanged text (tex4:473) made stale by this version: S81 is done and in the draft. If a main-line test of the "
     "trigger on delta is meant, the item should say so.", ("session~F's trigger on delta channels (S81)",))
r55, r22, r8a = (git("show", "--name-only", "--format=", c_).split() for c_ in (AM2_REF, "2225f00", "8ae660d"))
prov_ok = ("specs/test_delta_recipe.md" in r55 and "test_delta_recipe.py" in r22 and XREC in r8a
           and t_rb < ctime(AM2_REF) < ctime("2225f00") < t_x and all(is_anc(c_, ML5) for c_ in (DR["spec"], DR["amend"], DR["test"], AM2_REF, "2225f00", "8ae660d")))
qrow("Q11", 496, "provenance row delta_recipe: 'spec 05256c5, Amendment 1 32c121b / test a581027, Amendment 2 during the run 55dac50, 2225f00 / "
     "8ae660d (X) & X complete; L to run & 84; ---'", "as listed",
     f"{AM2_REF} changes {r55}; 2225f00 ({cdate('2225f00')}) changes {r22}; 8ae660d changes {r8a}; runs began {xm['started']} UTC, "
     f"result {cdate('8ae660d')}: both amendment commits fall between: {t_rb < ctime(AM2_REF) < ctime('2225f00') < t_x}; X records' "
     f"meta.git {xm['git']}; all on the main line: {prov_ok}; X records {sum(x_arms.values())}; L files {L_DR or 'none'}",
     "MATCH" if prov_ok and dr5_ok else "DISAGREE",
     "Supersedes P6 (resolved). The X records carry a581027: 2225f00 changed the report rules and the CHECKs' sources during the "
     "run, and 8ae660d's report was regenerated with --report at 2225f00, as the test's RESULT docstring says.",
     ("Amendment~2 during the run \\texttt{55dac50}, \\texttt{2225f00} / \\texttt{8ae660d} (X) & X complete; L to run & 84; ---",))
f_after = git("log", "--format=%h", f"{R7}..{F5}").split()
r7_files = git("show", "--name-only", "--format=", R7).split()
qrow("Q12", 514, "F's provenance row: 'S81 the trigger on delta (report 7, d5851b1) & d5851b1 & 322; 102; 30; 40; 48; 42'", "report 7 d5851b1; 42 runs",
     f"{R7} '{git('log', '-1', '--format=%s', R7).strip()[:80]}' ({cdate(R7)}) adds {r7_files}; F's last commit before the fifth draft "
     f"{F5} (commits after {R7}: {f_after or 'none'}); S81 records {nrec_T} (BASE, GUARD, EVAL, GUARD_EVAL 10 each, O_DELTA_C 2)",
     "MATCH" if F5 == R7 and nrec_T == 42 and f"{F}report_7.md" in r7_files else "DISAGREE",
     "Supersedes P2 and the first-pass 'branch commit' row (ed76eea). The first five counts stand (first-pass row 'runs per report').",
     ("S81 the trigger on delta (report 7, \\texttt{d5851b1}) & \\texttt{d5851b1} & 322; 102; 30; 40; 48; 42",))
qrow("Q13", 58, "index row, session F: 'on delta the flat-start guard costs and the evaluation trigger keeps'", "guard costs; evaluation keeps",
     f"GUARD {nT['GUARD']}/10 vs BASE {nT['BASE']}/10 ({b_g} vs {c_g}); EVAL {nT['EVAL']}/10 ({b_e} vs {c_e}); F's readings: 'the guard costs "
     f"binding' {g_costs} (needs c >= 4), 'the guard aims the split' {aims}, 'the evaluation is the better probe' {better}, 'the evaluation "
     f"costs binding' {e_costs}", "PARTIAL",
     "New in 3b06af2. The index's results column states readings elsewhere; here 'costs' and 'keeps' are not S81 readings (none "
     "applies by F's rules) but descriptions of 0 vs 2 and 1 vs 0 on ten seeds. 'Costs' is the name of F's reading that did not "
     "apply (c = 2 < 4). Suggest: 'on delta the flat-start guard lost two seeds and the evaluation trigger none (no reading applies)'.",
     ("on delta the flat-start guard costs and the evaluation trigger keeps",))
qrow("Q14", 205, "S81 'is the same screen on delta channels (seeds 2020--2029; the delta perfect gate 2/2)'", "same screen; 2020-2029; 2/2",
     f"seeds {sorted(T['BASE'])[0]}-{sorted(T['BASE'])[-1]}; O_DELTA_C {orT}; arms {LA} (S84's: BASE, GUARD, GUARD_INT1200, EVAL, EVAL_GUARD_INT1200)",
     "MATCH" if validT and sorted(T["BASE"]) == list(range(2020, 2030)) else "DISAGREE",
     "explore_f_trigger.py: 'The arm names and the guard's rule are Session H's S84 ones (BASE, GUARD, EVAL), so the two screens "
     "pair by name.' S81's fourth arm (GUARD_EVAL) is not S84's EVAL_GUARD_INT1200 (no fast checks); the draft does not cite it.",
     ("Session~F's S81 is the same screen on delta channels (seeds 2020--2029; the delta perfect gate $2/2$)",))
qrow("Q15", 205, "'the guard reads the other way, 7/10 against the base's 9/10 (0 vs 2; neither \"aims\" nor \"costs binding\" applies by F's rules)'",
     "7/10 vs 9/10; 0 vs 2; neither applies", f"GUARD {nT['GUARD']}/10, BASE {nT['BASE']}/10, {b_g} vs {c_g}; aims {aims}, costs {g_costs}",
     "MATCH" if (nT["GUARD"], nT["BASE"], b_g, c_g) == (7, 9, 0, 2) and not aims and not g_costs else "DISAGREE", "",
     ("there the guard reads the other way, $7/10$ against the base's $9/10$ (0 vs 2; neither ``aims'' nor ``costs binding'' applies by F's rules)",))
qrow("Q16", 205, "'because on delta channels the base's flat-start splits precede runs that bind (all eight runs that split in the flat start "
     "bound, though only 10 of the base's 19 firings were on labelled targets), and the guard removes exactly those'",
     "8 flat-start runs all bound; 10 of 19 labelled; guard removes those",
     f"BASE firings below 0.5: {len(flatB)} in runs {flatB_runs}, all bound routed {all(flatB_bound)}; labelled {labT['BASE']}/{len(FT['BASE'])}; "
     f"GUARD fires none below 0.5 (its 6 at {gu_f}); BASE's firings at >= 0.5: {lateB}",
     "PARTIAL" if len(flatB_runs) == 8 and all(flatB_bound) and (labT["BASE"], len(FT["BASE"])) == (10, 19) else "DISAGREE",
     "Counts right. 'Because' states as a cause what report 7 gives under 'What happened (post hoc; one screen, ten seeds)'; the "
     "draft does not mark it. 'Removes exactly those' is the guard's rule (no split below 0.5); the guarded runs then fire elsewhere "
     "(2020 and 2025 at 31200-36000).",
     ("because on delta channels the base's flat-start splits precede runs that bind (all eight runs that split in the flat start bound, "
      "though only 10 of the base's 19 firings were on labelled targets), and the guard removes exactly those",))
qrow("Q17", 205, "'the evaluation as the probe binds 10/10 against 9/10 (1 vs 0), \"the evaluation is the better probe\" not applying under "
     "F's stricter rule because 5 of its 13 firings came where the probe would have decided otherwise, and applying under H's rule'",
     "10/10 vs 9/10; 1 vs 0; 5 of 13; F no, H yes",
     f"EVAL {nT['EVAL']}/10 vs {nT['BASE']}/10, {b_e} vs {c_e}; disagreement firings {disF['EVAL']} of {len(FT['EVAL'])}; F's rule {better}; "
     f"H's rule (c - b <= 1) {better_H}",
     "MATCH" if (nT["EVAL"], b_e, c_e, disF["EVAL"], len(FT["EVAL"])) == (10, 1, 0, 5, 13) and not better and better_H else "DISAGREE", "",
     ("the evaluation as the probe binds $10/10$ against $9/10$ (1 vs 0), ``the evaluation is the better probe'' not applying under F's stricter rule",
      "because 5 of its 13 firings came where the probe would have decided otherwise, and applying under H's rule"))
qrow("Q18", 205, "'The seed-914 mode appeared in one guarded run (the probe at 0.95--0.98 against the evaluation's 0.87--0.92 at five checks; MERGED at 0.93)'",
     "one guarded run; 0.95-0.98 vs 0.87-0.92; five checks; MERGED 0.93",
     "; ".join(f"{a}: {[(s, u, round(p, 3), round(e, 3)) for s, u, p, e in m914[a]]}" for a in LA),
     "PARTIAL" if g25_ok else "DISAGREE",
     "GUARD 2025's numbers match. As F's tables print the mode (probe >= 0.95 while the evaluation < 0.95), it also occurred in the "
     "other guarded run on that seed, GUARD_EVAL 2025, at 4 checks (there the rule read the evaluation and fired), and in BASE "
     "at 2 checks (2025, 2027), as report 7 says. 'One guarded run' is true only of runs where the mode could block the rule.",
     ("The seed-914 mode appeared in one guarded run (the probe at $0.95$--$0.98$ against the evaluation's $0.87$--$0.92$ at five checks; MERGED at $0.93$)",))
qrow("Q19", 205, "'Over the three screens of the evaluation as the trigger's probe (S84, S86, S81) it is 29/30 against 27/30, 2 vs 0, descriptive; it has not cost a run'",
     "29/30 vs 27/30; 2 vs 0",
     "; ".join(f"{k_} EVAL {v[0]}/10 vs BASE {v[1]}/10, {v[2][0]} vs {v[2][1]}" for k_, v in tri.items())
     + f"; total {tri_e}/{tri_n} vs {tri_b}/{tri_n}, {tri_bc[0]} vs {tri_bc[1]}",
     "MATCH" if (tri_e, tri_b, tri_n, tri_bc) == (29, 27, 30, (2, 0)) else "DISAGREE",
     "S84 from H's guard_results.json at 37c1fbf (report 6), S86 from allmass_results.json at 9b72bf6 (report 8); BOUND ROUTED "
     "recomputed with rlib. 'Has not cost a run': BASE alone binds on no seed in any of the three.",
     ("Over the three screens of the evaluation as the trigger's probe (S84, S86, S81) it is $29/30$ against $27/30$, 2 vs 0, descriptive",))
qrow("Q20", 36, "abstract: 'so the memory rule is open again and is the next main-line test, specified and reviewed'",
     "next main-line test, specified and reviewed", dr5, "PARTIAL",
     "Unchanged text. At c5e4508 test_delta_recipe is no longer only specified and reviewed: X's half is complete (8ae660d) and its "
     "secondary results are in the index (tex5:54) and the memory section (tex5:256). The fourth pass did not flag this sentence "
     "while X's runs were under way (an omission of that pass).",
     ("so the memory rule is open again and is the next main-line test, specified and reviewed",))
pdt5 = x_doc.split("\n")
pb5 = [i + 1 for i, ln in enumerate(pdt5) if ln.startswith("def probe_blind")]
pb5_body = "\n".join(pdt5[pb5[0] - 1:pb5[0] + 25]) if pb5 else ""
pb5_cond = 'c["acc"] >= tgw.ACC_THR and h < tgw.ACC_THR' in pb5_body and "c[\"step\"] >= tgw.FIRST" in pb5_body
qrow("Q21", 432, "method: 'with the count of checks at which the probe read >= 0.95 while the held-out accuracy was below it, as test_delta_recipe prints'",
     "the count test_delta_recipe prints", f"test_delta_recipe.py at {ML5}: probe_blind() at :{pb5}, condition probe >= ACC_THR and held-out < "
     f"ACC_THR from update FIRST (4800): {pb5_cond}; the RESULT (X) docstring prints 'Probe-blind (checks from 4800): HEBB_SPLIT 1 of 181 ... "
     f"DELTA_SPLIT 1 of 142': {'Probe-blind (checks from 4800)' in x_doc}",
     "MATCH" if pb5 and pb5_cond else "DISAGREE",
     "Supersedes P8: the code changed at 2225f00 (Amendment 2 counts from 4800 and adds 'blocked by the probe alone'); the sentence "
     "is true of it, and the test has now printed the count on X.",
     "with the count of checks at which the probe read $\\geq 0.95$ while the held-out accuracy was below it, as \\texttt{test\\_delta\\_recipe} prints")
qrow("Q22", 286, "synthesis table S=8: one delta channel 'test_split_copy pooled 0/20^s, F 0/10^f'", "F's S68 0/10 at eight",
     f"S68_SINGLE_S8 BOUND {k8}/10", "MATCH" if k8 == 0 else "DISAGREE",
     "Supersedes the first-pass row 'S=8 one delta channel, F' (the cell's X part became the pooled 0/20, the main line's, checked "
     "in check_split_copy_v5).", ("$0/20^{s}$, $0/10^{f}$",))

# readings vs docstring, fifth draft (S81 now has readings)
PREC81 = dict(scr="S81", path="explore_f_trigger.py", text='"the guard aims the split"', rc=rc81, rcd=cdate(rc81), fr=fr81, frd=cdate(fr81),
              nrec=nrec81, unchanged=unch81, rg=rg81, desc=desc81, prec=prec81)

# ── every live fourth-pass row located in the fifth draft ──────────────────
LIVE4 = [r for r in LIVE3 if r["state4"] == "stands"] + PROWS
SUP5 = {"P1": "Q2", "P2": "Q12", "P3": "Q1", "P4": "Q4", "P5": "Q5", "P6": "Q11", "P7": "Q9"}
SUP5_FIRST = {("tex:280", "S=8 one delta channel, F"): "Q22", ("tex:501", "branch commit"): "Q12"}
RESUP5 = {"P8": "Q21"}


def key5(r):
    if r["kind"] == "first":
        for (t_, pre), v in SUP5_FIRST.items():
            if r["tex"] == t_ and r["item"].startswith(pre):
                return v
        return None
    return r.get("id")


for r in LIVE4:
    q = r["q"]
    r["loc5"] = (find(TEX5, q) or "REMOVED") if q is not None else r.get("loc4")
    k5 = key5(r)
    if r["loc5"] == "REMOVED":
        sup = SUP5.get(k5, k5 if k5 and k5.startswith("Q") else None)
        assert sup, ("row removed in tex5 without a superseding Q row", r.get("id"), r["tex"], r["item"][:60])
        r["sup5"], r["state5"] = sup, "REMOVED in v5"
    elif k5 in RESUP5:
        r["sup5"], r["state5"] = RESUP5[k5], "SUPERSEDED in v5 (re-checked against the code at the main line as of the fifth draft)"
    else:
        r["sup5"], r["state5"] = None, "stands"
for k in list(SUP5) + list(RESUP5):
    assert any(r.get("id") == k for r in LIVE4), ("SUP5/RESUP5 key matches no live row", k)
# statuses of the fourth-pass rows as recomputed now, against check_F_v4.md (rows whose inputs are read at branch heads)
V4MD = open(os.path.join(HERE, "check_F_v4.md")).read().split("\n")
v4stat = {}
for ln in V4MD:
    if ln.startswith("| ") and "**" in ln:
        c_ = [x.strip() for x in ln.split(" | ")]
        if len(c_) >= 9 and c_[0].lstrip("| ").isdigit():
            v4stat.setdefault((c_[1], c_[4]), c_[7].strip("*"))
flips = []
for r in LIVE4:
    k_ = (r.get("id", "") or "", str(r["item"]).replace("|", "\\|").replace("\n", " "))
    old_ = v4stat.get(k_)
    if old_ is not None and old_ != r["status"]:
        flips.append((r.get("id") or r["tex"], r["item"][:80], old_, r["status"]))


# ══ sixth pass: every live fifth-pass row in the sixth draft; T rows for 42c4557's changes ══════════════════
import rlib as _rlib  # noqa: E402

CUT6_ISO = git("show", "-s", "--format=%cI", SIXTH_REF).strip()
CUT6 = ctime(SIXTH_REF)
TIP6 = {b_: git("rev-list", "-1", f"--before={CUT6_ISO}", b_).strip()[:7] for b_ in REMOTES}
TIP6 = {b_: t_ for b_, t_ in TIP6.items() if t_}
F6, ML6 = TIP6[BR], TIP6[MAIN]
F6_AFTER = git("log", "--format=%h %cI %s", f"{R7}..{F6}").strip().split("\n")
F6_AFTER = [x for x in F6_AFTER if x.strip()]
F_NOW_AFTER = len(git("log", "--format=%h", f"{F6}..{F_NOW}").split())
# every F store this check reads: byte-identical at report 7, at F's last commit before the sixth draft and at F's head now
F_STORES = sorted({f"{F}{n}_results.json" for n in list(SAME)} | {f"{F}trigger_results.json", f"{F}eight_delta_star.json"})
STORE_ID = {p_: (git("rev-parse", f"{R7}:{p_}").strip() == git("rev-parse", f"{F6}:{p_}").strip()
                 == git("rev-parse", f"{F_NOW}:{p_}").strip()) for p_ in F_STORES}
stores_same6 = all(STORE_ID.values())
s81_6 = s81_state(F6)
r7_still_last = not any(("report" in x.lower() and "progress" not in x.lower()) for x in F6_AFTER)
TROWS = []


def trow(tid, tex6, item, draft, rec, status, note, q6):
    loc = find(TEX6, q6)
    assert tex6 in loc, ("fragment not at the cited sixth-draft line", tid, tex6, q6, loc)
    TROWS.append(dict(id=tid, tex=f"tex6:{tex6}", item=item, draft=str(draft), rec=str(rec), status=status, note=note,
                      q=q6, loc6=loc, kind="sixth", old=find(TEX5, q6)))


# test_delta_recipe at the sixth draft: X's records (unchanged since 8ae660d), L's state (the main line's round-5 prompts)
XREC6_same = git("rev-parse", f"{ML6}:{XREC}").strip() == git("rev-parse", f"{ML5}:{XREC}").strip()
XS6 = load(ML6, XREC)
XR = {a: runs(XS6, a) for a in ("DELTA_SPLIT", "DELTA_SLOW", "HEBB_SPLIT", "HEBB_SLOW")}
xbr = {a: sum(_rlib.bound_routed(r) for r in v.values()) for a, v in XR.items()}
pr_ds_hs = pairs(XR["DELTA_SPLIT"], XR["HEBB_SPLIT"], _rlib.bound_routed)      # (delta only, Hebb only, n)
pr_dw_hw = pairs(XR["DELTA_SLOW"], XR["HEBB_SLOW"], _rlib.bound_routed)
pr_hs_hw = pairs(XR["HEBB_SPLIT"], XR["HEBB_SLOW"], _rlib.bound_routed)
pr_ds_dw = pairs(XR["DELTA_SPLIT"], XR["DELTA_SLOW"], _rlib.bound_routed)
d1 = pr_ds_hs[1] - pr_ds_hs[0]                                                  # Hebb only - delta only
L_DR6 = {b_: [f_ for f_ in git("ls-tree", "-r", "--name-only", t_).split("\n") if f_.startswith("results/L/") and "delta_recipe" in f_]
         for b_, t_ in TIP6.items()}
L_DR6 = {b_: v for b_, v in L_DR6.items() if v}
R5DOC = git("show", f"{ML6}:docs/reading/parallel_sessions_prompts.md")
r5_ln = [i + 1 for i, ln in enumerate(R5DOC.split("\n")) if ln.startswith("## Round 5") and "L runs test_delta_recipe" in ln]
r5_c = git("log", "-1", "--format=%h", ML6, "--", "docs/reading/parallel_sessions_prompts.md").strip()
L_running6 = bool(r5_ln) and not L_DR6
r5_rec = (f"the main line's round-5 prompts ({r5_c}, {cdate(r5_c)}) docs/reading/parallel_sessions_prompts.md:{r5_ln}: '## Round 5 "
          f"(12 October, evening): ... L runs test_delta_recipe ...'; L delta_recipe files on any branch at the sixth draft: {L_DR6 or 'none'}")
dr6 = (f"X's result {x_add} ({cdate(x_add[0])}): {sum(x_arms.values())} records, unchanged at {ML6}: {XREC6_same}; bound routed DELTA_SPLIT "
       f"{xbr['DELTA_SPLIT']}/20, DELTA_SLOW {xbr['DELTA_SLOW']}/20, HEBB_SPLIT {xbr['HEBB_SPLIT']}/20, HEBB_SLOW {xbr['HEBB_SLOW']}/20; {r5_rec}")
x_done6 = dr5_ok and XREC6_same and not L_DR6
t_check6 = ctime(DR["tcheck"])
chk_h = (t_rb - t_check6) / 3600
ver_h = (t_rb - run_start) / 3600
trow("T1", 41, "test_delta_recipe: 'its test file checked against the amended spec before any arm's run (the check preceded the runs by two "
     "and a half hours; X's verification took three)'", "check before any arm's run; 2.5 h; verification three hours",
     f"the referee's test check {DR['tcheck']} ({cdate(DR['tcheck'])}); X's runs began {xm['started']} UTC (meta.started); the process "
     f"started 02:12 UTC (Amendment 2; RESULT (X)): check -> runs {chk_h:.2f} h; process start -> runs {ver_h:.2f} h; Amendment 1 "
     f"{DR['amend']} ({cdate(DR['amend'])}) before the process: {am1_before}",
     "MATCH" if t_check6 < t_rb and 2.25 <= chk_h <= 2.75 and 2.75 <= ver_h <= 3.5 and am1_before else "DISAGREE",
     f"Supersedes Q1 (removed). Takes the fifth pass's correction. 'Three' is {ver_h:.2f} h (3 h 15 min), rounded down.",
     "its test file checked against the amended spec before any arm's run (the check preceded the runs by two and a half hours; X's verification took three)")
trow("T2", 41, "'F's S81, G's S83, H's S86 and R's Phase 4 are complete' (F's part), at the sixth draft", "S81 complete",
     f"F at {F6} ({cdate(F6)}), F's last commit before the sixth draft ({cdate(SIXTH_REF)}): S81 records {s81_6['n']}, log 'S81 done' "
     f"{s81_6['done']}, report 7 {s81_6['report']}; F's commits after report 7: {F6_AFTER}",
     "MATCH" if s81_6["done"] and s81_6["report"] and s81_6["n"] == 42 else "DISAGREE",
     "Re-reads Q2 at the sixth draft's time (F's branch moved: S88's readings and queue). G, H and R are other checks'.",
     ("F's S81, G's S83, H's S86 and R's Phase~4 are complete",))
R5_BR6 = {"V": "origin/claude/review-V", "G": "origin/claude/explore-G", "H": "origin/claude/explore-H", "F": BR,
          "R": "origin/claude/replicate-R", "I": I_BR}
R5_AFT6 = {k: [x for x in git("log", "--format=%h %s", f"--since={git('show', '-s', '--format=%cI', r5_c).strip()}", TIP6[b_]).strip().split("\n")
               if x.strip()] for k, b_ in R5_BR6.items()}
e_tip6 = TIP6[E_BR]
e_rd6 = [ln for ln in git("show", f"{e_tip6}:explore_out/README.md").split("\n") if "batch20.log" in ln and "IN PROGRESS" in ln]
e_b20code = git("show", f"{e_tip6}:explore_batch20.py")
trow("T3", 41, "'E's batch 20 (S67's unbound runs continued to 86,400 updates) and the sessions' fifth round are running'",
     "batch 20 running; round 5 running",
     f"E at {e_tip6} ({cdate(e_tip6)}, '{git('log', '-1', '--format=%s', e_tip6).strip()[:60]}'); README '{e_rd6[0][:150] if e_rd6 else '-'}'; "
     f"explore_batch20.py continues S67's runs to 86,400: {'86,400' in e_b20code and 'S67' in e_b20code}; round-5 prompts {r5_c} "
     f"({cdate(r5_c)}); commits after them before the cutoff: " + "; ".join(f"{k} {len(v)} (last '{v[0][:60]}')" for k, v in R5_AFT6.items() if v)
     + f"; none: {[k for k, v in R5_AFT6.items() if not v] or 'none'}",
     "MATCH" if e_rd6 and all(R5_AFT6.values()) else "DISAGREE",
     "Supersedes Q3 (removed; 'every session's queue is empty'). F's part: S88a/S88b readings fixed at 40087cb and the queue at "
     "cdbb886, 6-7 min before the draft.", ("E's batch~20 (S67's unbound runs continued to 86{,}400 updates) and the sessions' fifth round are running",))
trow("T4", 41, "'L's half is next' and 'test_delta_recipe's L half has not run'", "L next; L has not run",
     dr6 + "; the draft elsewhere: '(X; L running)' (tex6:54), 'L's half is running' (tex6:360), 'runs on the other' (tex6:36)",
     "PARTIAL" if x_done6 and L_running6 else "MATCH",
     "Unchanged text (Q1's second fragment) that this version's tex6:36, 54 and 360 contradict: L runs the test by the main line's "
     "round-5 prompts. 'Has not run' is true of results only.", ("L's half is next", "\\texttt{test\\_delta\\_recipe}'s L half has not run"))
trow("T5", 54, "index row delta_recipe: '(X; L running) ... pooled labels and readings wait for L'", "X complete; L running", dr6,
     "MATCH" if x_done6 and L_running6 else "PARTIAL",
     "Supersedes Q4 (removed; '(specified)'). 'L running' is not in a record; the main line's round-5 prompts say so.",
     ("\\texttt{delta\\_recipe} (X; L running)", "pooled labels and readings wait for L"))
trow("T6", 256, "memory section: 'with the test file checked against the amended spec before any arm's run; run on X, where a second amendment "
     "... was committed during the run'", "checked before any arm's run; run on X", dr6 + f"; Amendment 2 {AM2_REF} ({cdate(AM2_REF)}) between "
     f"the runs' start and the result: {t_rb < ctime(AM2_REF) < (t_x or 0)}", "MATCH" if x_done6 and t_rb < ctime(AM2_REF) < t_x else "DISAGREE",
     "Supersedes Q5 (removed; 'running on X').",
     ("with the test file checked against the amended spec before any arm's run; run on X, where a second amendment confined to the report's rules",))
trow("T7", 256, "'none of them a claim, a reading or a Hebbian learned-gate run; two were the Hebbian perfect gate's'",
     "no Hebbian learned-gate run; two Hebbian perfect-gate lines", f"Amendment 2 ({AM2_REF}): the 24 lines are CEIL_HEBB 580, 581; CEIL_DELTA 580, "
     f"581; DELTA_SPLIT 580-598; DELTA_SLOW|580; 'None of HEBB_SPLIT or HEBB_SLOW had finished': {am2_lines}",
     "MATCH" if am2_lines else "NOT RECORD-CHECKABLE", "Supersedes Q6 (removed; the fifth pass's DISAGREE).",
     ("none of them a claim, a reading or a Hebbian learned-gate run; two were the Hebbian perfect gate's",))
SC_X, SC_L = load("592c574", "results/X/split_copy_results.json"), load("ffb0771", "results/L/split_copy_results.json")
c3m = {}
for m_, st_ in (("X", SC_X), ("L", SC_L)):
    sp_, sd_ = runs(st_, "WIN3_SPLIT_D8"), runs(st_, "SINGLE_DELTA_D8")
    c3m[m_] = (sum(1 for s_ in sp_ if _rlib.bound_routed(sp_[s_]) and not _rlib.bound(sd_[s_])),
               sum(1 for s_ in sp_ if _rlib.bound(sd_[s_]) and not _rlib.bound_routed(sp_[s_])))
c3p = (c3m["X"][0] + c3m["L"][0], c3m["X"][1] + c3m["L"][1])
p3 = {k_: mcnemar1(*v) for k_, v in list(c3m.items()) + [("pooled", c3p)]}
trow("T8", 256, "'(S68, whose reading of that name did not apply; the pre-registered support for the partition is C3, pooled and on each machine)'",
     "C3 pooled and on each machine", f"C3 (WIN3_SPLIT_D8 bound routed vs SINGLE_DELTA_D8 bound): X {c3m['X'][0]} vs {c3m['X'][1]} (p = {p3['X']:.5f}), "
     f"L {c3m['L'][0]} vs {c3m['L'][1]} (p = {p3['L']:.4f}), pooled {c3p[0]} vs {c3p[1]} (p = {p3['pooled']:.2e}); S68: {r68}",
     "MATCH" if all(v < 0.05 for v in p3.values()) and c3p == (18, 0) else "DISAGREE",
     "Supersedes Q8 (removed; 'C3 on X'). C3 recomputed lightly from split_copy X (592c574) and L (ffb0771).",
     ("the pre-registered support for the partition is C3, pooled and on each machine",))
trow("T9", 476, "next steps: 'test_delta_recipe's L half (X's is complete)', at the sixth draft", "L half next; X complete", dr6,
     "MATCH" if x_done6 else "DISAGREE", "Re-reads Q9: L's half running (the round-5 prompts) and listed as the next step are compatible.",
     ("\\texttt{test\\_delta\\_recipe}'s L half (X's is complete)",))
trow("T10", 477, "next steps: S81 no longer listed", "S81 removed",
     f"'session~F's trigger on delta channels (S81)' in tex6: {find(TEX6, 'session~F' + chr(39) + 's trigger on delta channels (S81)') or 'absent'}; "
     f"S81 complete at {R7}, in the draft at tex6:205", "MATCH" if not find(TEX6, "session~F's trigger on delta channels (S81)") else "DISAGREE",
     "Supersedes Q10 (removed; the fifth pass's PARTIAL).",
     ("Session~H's proposed test of the key term at two streams", "batch~19's lead, sixteen keys per stream"))
trow("T11", 496, "provenance row delta_recipe: '... / 8ae660d (X) & X complete; L to run & 84; ---', at the sixth draft", "X complete; L to run",
     dr6, "PARTIAL" if x_done6 and L_running6 and prov_ok else ("MATCH" if prov_ok else "DISAGREE"),
     "Re-reads Q11: the commits and the count are right; 'L to run' is stale beside this version's '(X; L running)' (tex6:54), "
     "'L's half is running' (tex6:360) and against the main line's round-5 prompts.",
     ("Amendment~2 during the run \\texttt{55dac50}, \\texttt{2225f00} / \\texttt{8ae660d} (X) & X complete; L to run & 84; ---",))
trow("T12", 514, "F's provenance row: 'S81 the trigger on delta (report 7, d5851b1) & d5851b1 & 322; 102; 30; 40; 48; 42', at the sixth draft",
     "report 7 d5851b1 the last; 42 S81 runs",
     f"F's last commit before the sixth draft {F6} ({cdate(F6)}); F's commits after {R7}: {F6_AFTER}; a report among them: {not r7_still_last}; "
     f"S81 records at {F6}: {s81_6['n']}; F's head now {F_NOW} ({F_NOW_AFTER} commits after {F6}); every F store used here identical at {R7}, "
     f"{F6} and {F_NOW}: {stores_same6}",
     "MATCH" if r7_still_last and s81_6["n"] == 42 and stores_same6 else "DISAGREE",
     "Re-reads Q12 at the sixth draft's time (v5 asserted that F's last commit was report 7; F has since started round 5: S88a/S88b "
     "readings fixed before any run at 40087cb, queue and progress at cdbb886). No report after d5851b1, so the row's commit and "
     "counts are right; the row lists reports, and S88 has none yet.",
     ("S81 the trigger on delta (report 7, \\texttt{d5851b1}) & \\texttt{d5851b1} & 322; 102; 30; 40; 48; 42",))
trow("T13", 58, "index row, session F: 'on delta the flat-start guard lost two seeds and the evaluation trigger none (no reading applies)'",
     "guard lost two seeds; evaluation none; no reading applies",
     f"GUARD {nT['GUARD']}/10 vs BASE {nT['BASE']}/10: arm only {b_g}, BASE only {c_g} {dsT['GUARD']}; EVAL {nT['EVAL']}/10: arm only {b_e}, "
     f"BASE only {c_e}; F's readings: aims {aims}, guard costs {g_costs}, evaluation better {better}, evaluation costs {e_costs}; none applies {none_applies}",
     "MATCH" if (b_g, c_g, c_e) == (0, 2, 0) and none_applies else "DISAGREE",
     "Supersedes Q13 (removed; the fifth pass's PARTIAL, whose suggested wording this is).",
     ("on delta the flat-start guard lost two seeds and the evaluation trigger none (no reading applies)",))
r7_ph = [i + 1 for i, ln in enumerate(R7T) if ln.startswith("## What happened (post hoc")]
trow("T14", 205, "'..., on session F's post-hoc account because on delta channels the base's flat-start splits precede runs that bind (all eight "
     "... bound, though only 10 of the base's 19 firings were on labelled targets), and the guard removes exactly those'",
     "post hoc, marked; 8 runs; 10 of 19",
     f"report 7 '## What happened (post hoc; one screen, ten seeds)' at report_7.md:{r7_ph}; BASE firings below 0.5: {len(flatB)} in runs "
     f"{flatB_runs}, all bound routed {all(flatB_bound)}; labelled {labT['BASE']}/{len(FT['BASE'])}; GUARD's firings {gu_f}",
     "MATCH" if r7_ph and len(flatB_runs) == 8 and all(flatB_bound) and (labT["BASE"], len(FT["BASE"])) == (10, 19) else "DISAGREE",
     "Re-reads Q16: the 'because' is now marked as session F's post-hoc account (the fifth pass's PARTIAL resolved).",
     ("on session~F's post-hoc account because on delta channels the base's flat-start splits precede runs that bind",))
ge25_fire = [c["step"] for s_, c in FT["GUARD_EVAL"] if s_ == 2025]
ge25_914 = [(u, round(p_, 3), round(e_, 3)) for s_, u, p_, e_ in m914["GUARD_EVAL"] if s_ == 2025]
trow("T15", 205, "'The seed-914 mode appeared in one guarded run where it could block the rule (and at four checks of that seed's evaluation-read "
     "guarded run, which fired, and at two of the base's) (the probe at 0.95-0.98 against the evaluation's 0.87-0.92 at five checks; MERGED at 0.93)'",
     "GUARD 2025: 5 checks, 0.95-0.98 vs 0.87-0.92, MERGED 0.93; GUARD_EVAL 2025: 4 checks, fired; BASE: 2",
     "; ".join(f"{a}: {[(s_, u, round(p_, 3), round(e_, 3)) for s_, u, p_, e_ in m914[a]]}" for a in LA)
     + f"; GUARD_EVAL 2025 fired at {ge25_fire} (the mode at {[x[0] for x in ge25_914]}); each arm's rule reads: "
     + ", ".join(f"{a} {srcT[a]}" for a in LA),
     "MATCH" if g25_ok and len(m914["GUARD_EVAL"]) == 4 and {s_ for s_, *_ in m914["GUARD_EVAL"]} == {2025} and ge25_fire
     and len(m914["BASE"]) == 2 and not m914["EVAL"] else "DISAGREE",
     "Supersedes Q18 (removed; the fifth pass's PARTIAL). The counts match. Two readings remain loose, wording only: the mode could "
     "block the rule in BASE too (BASE reads the probe; its two checks preceded binding), so 'one guarded run where it could block "
     "the rule' is exact of the guarded arms only; and the numbers parenthesis now follows the inserted one, so it describes the "
     "GUARD run only by position.",
     ("The seed-914 mode appeared in one guarded run where it could block the rule (and at four checks of that seed's evaluation-read guarded run, which fired, and at two of the base's)",
      "(the probe at $0.95$--$0.98$ against the evaluation's $0.87$--$0.92$ at five checks; MERGED at $0.93$)"))
trow("T16", 36, "abstract: 'its pre-registered test is complete on one machine (the delta recipe 19/20 against the Hebbian 16/20, and without a "
     "trigger 19/20 against 7/20, 12 vs 0; the split helps the Hebbian write, 10 vs 1) and runs on the other'",
     "19/20 vs 16/20; 19/20 vs 7/20, 12 vs 0; Hebbian split vs none 10 vs 1; L running",
     f"bound routed (rlib) DELTA_SPLIT {xbr['DELTA_SPLIT']}/20 vs HEBB_SPLIT {xbr['HEBB_SPLIT']}/20 ({pr_ds_hs[0]} vs {pr_ds_hs[1]}); "
     f"DELTA_SLOW {xbr['DELTA_SLOW']}/20 vs HEBB_SLOW {xbr['HEBB_SLOW']}/20 ({pr_dw_hw[0]} vs {pr_dw_hw[1]}); HEBB_SPLIT vs HEBB_SLOW "
     f"{pr_hs_hw[0]} vs {pr_hs_hw[1]} (p = {mcnemar1(pr_hs_hw[0], pr_hs_hw[1]):.4f}); {r5_rec}",
     "MATCH" if (xbr["DELTA_SPLIT"], xbr["HEBB_SPLIT"], xbr["DELTA_SLOW"], xbr["HEBB_SLOW"]) == (19, 16, 19, 7)
     and pr_dw_hw[:2] == (12, 0) and pr_hs_hw[:2] == (10, 1) and x_done6 and L_running6 else "DISAGREE",
     "Supersedes Q20 (removed; the fifth pass's PARTIAL: 'the next main-line test, specified and reviewed'). '12 vs 0' is the "
     "no-trigger pair (DELTA_SLOW vs HEBB_SLOW), '10 vs 1' is D4 (HEBB_SPLIT vs HEBB_SLOW), as the fifth pass asked. 'Complete on "
     "one machine' is X (8ae660d); 'runs on the other' is the round-5 prompts' statement, not a record.",
     ("its pre-registered test is complete on one machine (the delta recipe $19/20$ against the Hebbian $16/20$, and without a trigger $19/20$ against $7/20$, 12 vs 0; the split helps the Hebbian write, 10 vs 1) and runs on the other",))
trow("T17", 360, "assessment: 'A pre-registered test is specified and complete on X (Section~ref{sec:memory}); L's half is running'",
     "complete on X; L running", dr6, "MATCH" if x_done6 and L_running6 else "PARTIAL", "New in 42c4557.",
     ("A pre-registered test is specified and complete on X (Section~\\ref{sec:memory}); L's half is running",))
sp_v = {c_: git("show", f"{c_}:specs/test_delta_recipe.md") for c_ in (DR["spec"], DR["amend"], AM2_REF)}
lab_old = '"delta binds more often"' in " ".join(sp_v[DR["amend"]].split())        # the labels first appear in Amendment 1
lab_spec0 = '"delta binds more often"' in " ".join(sp_v[DR["spec"]].split())
lab_new = '"delta not shown worse; ahead by >= 3 pairs (untested)"' in " ".join(sp_v[AM2_REF].split())
trow("T18", 256, "D1's label: ''delta not shown worse; ahead by >= 3 pairs (untested)'' (d <= -3; Amendment 2's wording, the first draft of the "
     "label having read ''delta binds more often'')'; 'D1 d = -3 (delta ahead by three pairs, the first label's case; ...)'",
     "Amendment 2's wording; first draft 'delta binds more often'; d = -3",
     f"the original spec {DR['spec']} has D1 as a bound with HOLDS/FAILS and no such label ('delta binds more often' present: {lab_spec0}); "
     f"Amendment 1 {DR['amend']}, which introduced the labels, reads d <= -3 'delta binds more often': {lab_old}; Amendment 2 {AM2_REF}: 'delta not shown "
     f"worse; ahead by >= 3 pairs (untested)': {lab_new}; X: DELTA_SPLIT only {pr_ds_hs[0]}, HEBB_SPLIT only {pr_ds_hs[1]}, d = {d1}",
     "MATCH" if lab_old and not lab_spec0 and lab_new and d1 == -3 else "DISAGREE",
     "New in 42c4557 (the fifth pass's 'pre-amendment label quoted for D1'). 'The first draft of the label' is Amendment 1's, where "
     "the labels were introduced. Per machine, secondary; the D-label itself is pooled and waits for L.",
     ("Amendment~2's wording, the first draft of the label having read ``delta binds more often''", "the first label's case"))
room_spec = 'if W_SLOW binds routed on >= 36/40' in " ".join(sp_v[DR["amend"]].split()) and 'if W_SLOW binds routed on >= 36/40' in " ".join(sp_v[AM2_REF].split())
trow("T19", 256, "'D2 0 vs 0, NOT SHOWN on X, the delta no-trigger arm at 19/20 leaving at most one pair of room (''no room'' is the pooled label, "
     ">= 36/40, and waits for L)'; 'with the no-trigger arm already binding routed 19/20 where its Hebbian counterpart binds 7/20'",
     "D2 0 vs 0; at most one pair of room; 'no room' pooled >= 36/40",
     f"DELTA_SPLIT vs DELTA_SLOW {pr_ds_dw[0]} vs {pr_ds_dw[1]} (p = {mcnemar1(pr_ds_dw[0], pr_ds_dw[1]):.3g}); DELTA_SLOW {xbr['DELTA_SLOW']}/20, so "
     f"split-only pairs at most {20 - xbr['DELTA_SLOW']}; HEBB_SLOW {xbr['HEBB_SLOW']}/20; Amendment 1 {DR['amend']} and Amendment 2 {AM2_REF}: '\"no room\" if W_SLOW binds routed on >= 36/40' (pooled): {room_spec}",
     "MATCH" if pr_ds_dw[:2] == (0, 0) and xbr["DELTA_SLOW"] == 19 and xbr["HEBB_SLOW"] == 7 and room_spec else "DISAGREE",
     "New in 42c4557. 'NOT SHOWN' is the per-machine reading of a one-sided McNemar at 0 vs 0.",
     ("D2 0 vs 0, NOT SHOWN on X, the delta no-trigger arm at $19/20$ leaving at most one pair of room (``no room'' is the pooled label, $\\geq 36/40$, and waits for L)",
      "with the no-trigger arm already binding routed $19/20$ where its Hebbian counterpart binds $7/20$"))

# ── every live fifth-pass row located in the sixth draft ──────────────────
LIVE5 = [r for r in LIVE4 if r["state5"] == "stands"] + QROWS
SUP6 = {"Q1": "T1", "Q3": "T3", "Q4": "T5", "Q5": "T6", "Q6": "T7", "Q8": "T8", "Q10": "T10", "Q13": "T13", "Q18": "T15", "Q20": "T16"}
RESUP6 = {"Q2": "T2", "Q9": "T9", "Q11": "T11", "Q12": "T12", "Q16": "T14"}
SUP6_FIRST = {}


def key6(r):
    if r["kind"] == "first":
        for (t_, pre), v in SUP6_FIRST.items():
            if r["tex"] == t_ and r["item"].startswith(pre):
                return v
        return None
    return r.get("id")


for r in LIVE5:
    q = r["q"]
    r["loc6"] = (find(TEX6, q) or "REMOVED") if q is not None else r.get("loc5")
    k6 = key6(r)
    if r["loc6"] == "REMOVED":
        sup = SUP6.get(k6, k6 if k6 and k6.startswith("T") else None)
        assert sup, ("row removed in tex6 without a superseding T row", r.get("id"), r["tex"], r["item"][:70], r["q"])
        r["sup6"], r["state6"] = sup, "REMOVED in v6"
    elif k6 in RESUP6:
        r["sup6"], r["state6"] = RESUP6[k6], "SUPERSEDED in v6 (re-read at the sixth draft's time, or its point fixed in place)"
    else:
        r["sup6"], r["state6"] = None, "stands"
for k in list(SUP6) + list(RESUP6):
    assert any(r.get("id") == k for r in LIVE5), ("SUP6/RESUP6 key matches no live row", k)
for k, v in SUP6.items():
    assert next(r for r in LIVE5 if r.get("id") == k)["state6"] == "REMOVED in v6", ("SUP6 row not removed", k)
# statuses of the fifth-pass rows as recomputed now, against check_F_v5.md
V5MD = open(os.path.join(HERE, "check_F_v5.md")).read().split("\n")
v5stat = {}
for ln in V5MD:
    if ln.startswith("| ") and "**" in ln:
        c_ = [x.strip() for x in ln.split(" | ")]
        if len(c_) >= 9 and c_[0].lstrip("| ").isdigit():
            v5stat.setdefault((c_[1], c_[4]), c_[7].strip("*"))
flips6 = []
for r in LIVE5:
    k_ = (r.get("id", "") or "", str(r["item"]).replace("|", "\\|").replace("\n", " "))
    old_ = v5stat.get(k_)
    if old_ is not None and old_ != r["status"]:
        flips6.append((r.get("id") or r["tex"], r["item"][:80], old_, r["status"]))


# ══ write ═══════════════════════════════════════════════════════════════════
ORDER = {"DISAGREE": 0, "PARTIAL": 1, "NOT RECORD-CHECKABLE": 2, "MATCH": 3}


def esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def lstr(loc, pre="tex6"):
    return loc if isinstance(loc, str) else ", ".join(f"{pre}:{x}" for x in loc)


def qstr(q):
    return " ... ".join(q) if isinstance(q, tuple) else q


ALLT = {r["id"]: r for r in TROWS}
live6 = [r for r in LIVE5 if r["state6"] == "stands"] + TROWS
gone6 = [r for r in LIVE5 if r["state6"] != "stands"]
cnt6 = Counter(r["status"] for r in live6)
cntT = Counter(r["status"] for r in TROWS)
cnt5 = Counter(r["status"] for r in LIVE5)
cntR7 = Counter(r["status"] for r in R7ROWS)
L = []
P = L.append
P("# Revision 9 draft, sixth version, vs Session F's records (sixth pass)\n")
P(f"Generated by `docs/review_2026-10/rev9/check_F_v6.py` (stdlib + `rlib`), built from `check_F_v5.py` with every record "
  f"recomputation kept. Sixth draft: `{DRAFT_FILE}` at `{SIXTH_REF}` ({cdate(SIXTH_REF)}, {len(TEX6) - (TEX6[-1] == '')} lines), "
  f"cited `tex6:NNN`; fifth draft `{FIFTH_REF}` ({cdate(FIFTH_REF)}, {len(TEX5) - (TEX5[-1] == '')} lines), cited `tex5:NNN` "
  f"(42c4557's changes since c5e4508 replace lines in place: the line numbers agree). Records: `{BR}` pinned at **{HEAD}** "
  f"({cdate(HEAD)}), report 7. F's last commit before the sixth draft is `{F6}` ({cdate(F6)}; after report 7: "
  + "; ".join(f"`{x.split()[0]}` {x.split(' ', 2)[2][:70]}" for x in F6_AFTER)
  + f"), and F's head now `{F_NOW}`; every F store used here is byte-identical at `{HEAD}`, `{F6}` and `{F_NOW}`: {stores_same6} "
  f"({len(F_STORES)} files). Every store used before report 7 is byte-identical to the one at the report commit the draft names: "
  f"{all(v[1] for v in SAME.values())}. The main line as of the sixth draft is `{ML6}` (as of the fifth, `{ML5}`).\n")
P("Method. Every row live in `check_F_v5.md` (the earlier passes' rows that stood in the fifth draft, and the `Q..` rows) keeps "
  "its record recomputation and is located in the sixth draft by text:\n"
  "- found: it **stands** at that `tex6` line with its status;\n"
  "- not found: **REMOVED in v6**, superseded by the sixth-pass row (`T..`) that checks the new wording;\n"
  "- found, but its verdict must be re-read at the sixth draft's time (F's branch, the main line and L's state moved) or its "
  "point was fixed around the fragment: **SUPERSEDED in v6**, and the `T..` row is live.\n"
  "`T..` rows cover every changed sentence of the sixth draft about Session F, the delta write's main-line test and its "
  "status, and the status line tex6:41; each has a fragment asserted at its `tex6` line. Report 7's rows (section 2) are "
  "v5's, recomputed unchanged.\n")
P(f"**Sixth draft: {len(live6)} rows** ({sum(1 for r in live6 if r['kind'] == 'first')} first-pass, "
  f"{sum(1 for r in live6 if r['kind'] == 'second')} second-pass, {sum(1 for r in live6 if r['kind'] == 'third')} third-pass, "
  f"{sum(1 for r in live6 if r['kind'] == 'fourth')} fourth-pass, {sum(1 for r in live6 if r['kind'] == 'fifth')} fifth-pass, "
  f"{len(TROWS)} sixth-pass rows): " + ", ".join(f"{k} {cnt6.get(k, 0)}" for k in ORDER) + ".\n")
P(f"- Fifth draft's live rows (recomputed here): {len(LIVE5)}, " + ", ".join(f"{k} {cnt5.get(k, 0)}" for k in ORDER) + ".")
P(f"- Removed or superseded in v6: {len(gone6)} rows, of which " + ", ".join(f"{k} {Counter(r['status'] for r in gone6).get(k, 0)}" for k in ORDER) + ".")
P("- Sixth-pass rows alone: " + ", ".join(f"{k} {cntT.get(k, 0)}" for k in ORDER) + ".")
P(f"- Report 7 (S81), not counted above: {len(R7ROWS)} rows, " + ", ".join(f"{k} {cntR7.get(k, 0)}" for k in ORDER) + ".\n")
P("Definitions as in `check_F.md`: BOUND = `rlib.bound`; BOUND ROUTED = `rlib.outcome_k`; 'labelled ok' = c* holds >= 2 "
  "streams and c0 none on the map just before the firing (s37c.targeting). stream_gate ties deferred to ch_map: "
  f"{sorted(set(TIES)) or 'none'}. test_delta_recipe's arms: bound routed = `rlib.bound_routed`.\n")
P("## 1. Sixth-draft table (non-MATCH first)\n")
P("| # | id | where (tex6) | origin | item | draft says | recomputed | status | note |")
P("|---|---|---|---|---|---|---|---|---|")
rows6 = sorted(live6, key=lambda r: (ORDER[r["status"]], (lambda x: x[0] if isinstance(x, list) and x else 9999)(r["loc6"])))
for i, r in enumerate(rows6, 1):
    if r["kind"] in ("first", "second", "third", "fourth", "fifth"):
        org = {"first": r["tex"], "second": f"second pass ({r['tex']})", "third": f"third pass ({r['tex']})",
               "fourth": f"fourth pass ({r['tex']})", "fifth": f"fifth pass ({r['tex']})"}[r["kind"]]
    else:
        org = ("tex5:" + ", ".join(map(str, r["old"])) + " (unchanged)") if r["old"] else "new or reworded in tex6"
    P(f"| {i} | {r.get('id', '')} | {esc(lstr(r['loc6']))} | {esc(org)} | {esc(r['item'])} | {esc(r['draft'])} | {esc(r['rec'])} | **{r['status']}** | {esc(r['note'])} |")
P("\n## 2. Report 7 (S81), every number, against `trigger_results.json` at " + R7 + " (v5's rows, recomputed)\n")
P("| # | where | item | report says | recomputed | status | note |")
P("|---|---|---|---|---|---|---|")
for i, r in enumerate(sorted(R7ROWS, key=lambda r: (ORDER[r["status"]], int(r["tex"].split(":")[1]))), 1):
    P(f"| {i} | {r['tex']} | {esc(r['item'])} | {esc(r['draft'])} | {esc(r['rec'])} | **{r['status']}** | {esc(r['note'])} |")
P("\n## 3. Rows removed or superseded in the sixth draft\n")
P("| row | was at (tex5) | its v5-era status (recomputed) | fragment | v6 state | superseded by | its status | its tex6 line |")
P("|---|---|---|---|---|---|---|---|")
for r in sorted(gone6, key=lambda r: (ORDER[r["status"]], r.get("id", "") or r["tex"])):
    s_ = ALLT[r["sup6"]]
    P(f"| {r.get('id') or esc(r['item'][:60])} | {esc(lstr(r['loc5'], 'tex5'))} | {r['status']} | {esc(qstr(r['q']))} | {r['state6']} | "
      f"{s_['id']} | {s_['status']} | {lstr(s_['loc6'])} |")
P("\n## 4. Every fifth-draft non-MATCH row, in the sixth draft\n")
P("| row | v5 place | status | sixth draft |")
P("|---|---|---|---|")
for r in sorted([r for r in LIVE5 if r["status"] != "MATCH" or r.get("id") in RESUP6],
                key=lambda r: (ORDER[r["status"]], r.get("id", "") or r["tex"])):
    if r["state6"] == "stands":
        w_ = (f"stands at {lstr(r['loc6'])}: still {r['status']}" if r["loc6"] != "report_6.md" else "report 6 (not the draft): unchanged")
    else:
        s_ = ALLT[r["sup6"]]
        w_ = f"{r['state6']}; {s_['id']} at {lstr(s_['loc6'])}: {s_['status']}"
    P(f"| {r.get('id') or esc(r['item'][:70])} | {esc(lstr(r['loc5'], 'tex5') if isinstance(r['loc5'], list) else r['loc5'])} | {r['status']} | {esc(w_)} |")
P("\n## 5. Fifth-pass rows whose status, recomputed now, differs from `check_F_v5.md`\n")
P("Some rows read branch heads; F's records are now pinned at report 7. Recomputed:\n")
if flips6:
    P("| row | item | in check_F_v5.md | now |\n|---|---|---|---|")
    for x in flips6:
        P("| " + " | ".join(esc(v) for v in x) + " |")
else:
    P("- none.")
P("\n## 6. Readings vs docstring, sixth draft\n")
P("| screen | reading (docstring) | sixth-draft wording (tex6) | recomputed | status |")
P("|---|---|---|---|---|")
RD6 = [
    ("S68", "\"the partition's advantage is in the number of streams\" / \"context-tagged keys scale\" (explore_f_premise.py:22-23)",
     "'both read \"neither\"' (tex6:246); 'did not apply' (tex6:256, 362)", r68, "MATCH"),
    ("S69", "\"the forget gate needs the delta rule\" (explore_f_hebb_decay.py:23)", "paragraph title (tex6:248)", r69, "MATCH"),
    ("S59", "\"the horizon knob matters\" / \"it does not\" (explore_f_decay.py:30-34)", "tex6:443 ('its pre-fixed reading on the horizon read \"it does not\"'), 464", r59, "MATCH"),
    ("S76 P2", "\"the delta recipe keeps eight streams\" (explore_f_eight.py:39)", "applies (tex6:250)", r76, "MATCH"),
    ("S80 (a), (b)", "\"the delta recipe keeps it\" (explore_f_configs.py READINGS)", "'Both readings are \"keeps it\"' (tex6:252)", f"{ra80}; {rb80}", "MATCH"),
    ("S81", "\"the guard aims the split\", \"the evaluation is the better probe\", \"the guard costs binding\", \"the evaluation costs binding\" (explore_f_trigger.py READINGS)",
     "'neither \"aims\" nor \"costs binding\" applies by F's rules'; 'the evaluation is the better probe' not applying (tex6:205); "
     "the index's 'the flat-start guard lost two seeds and the evaluation trigger none (no reading applies)' (tex6:58, T13)",
     f"aims {aims}; better {better} (H's rule {better_H}); guard costs {g_costs}; evaluation costs {e_costs}", "MATCH (tex6:205, tex6:58)"),
]
for x in RD6:
    P("| " + " | ".join(esc(v) for v in x) + " |")
P(f"\nS81's readings commit: `{PREC81['rc']}` ({PREC81['rcd']}); first S81 record commit `{PREC81['fr']}` ({PREC81['frd']}, "
  f"{PREC81['nrec']} records); explore_f_trigger.py unchanged to report 7: {PREC81['unchanged']}; record code commits {PREC81['rg']} "
  f"descend from the readings commit: {PREC81['desc']}; readings before the first record: {PREC81['prec']}.\n")
P("## 7. Branches at the sixth draft's time (last commit dated at or before " + CUT6_ISO + ")\n")
P("| branch | last commit before the cutoff | subject | head now |")
P("|---|---|---|---|")
for b_ in sorted(TIP6):
    P(f"| {b_.replace('origin/', '')} | {TIP6[b_]} ({cdate(TIP6[b_])}) | {esc(git('log', '-1', '--format=%s', TIP6[b_]).strip()[:110])} | "
      f"{git('rev-parse', '--short=7', b_).strip()} |")
P(f"\nF's stores, identity at {HEAD} / {F6} / {F_NOW}: " + "; ".join(f"{p_.split('/')[-1]} {v}" for p_, v in STORE_ID.items()) + ".\n")
P("## 8. Corrections to the referee's earlier files (recorded here, not edited)\n")
P("- **`check_F_v5.py`** asserted that F's head was report 7 (`assert HEAD == R7`) and read F's stores at F's head. That was "
  "true when it ran and is not now (F committed S88's readings and queue at 23:07, before the sixth draft). Not an error in "
  "v5's results; v6 pins the stores at report 7 and checks them byte for byte at F's later commits (section 7).")
P("- No other error found in `check_F_v5.md`'s rows on re-running them (section 5).")
P("\n## 9. Not checked here\n")
P("- test_delta_recipe's per-machine D3, firings and probe counts (tex6:256): the X-results check's. Here: the status lines, the "
  "commits, the abstract's counts and discordant pairs (T16), D1's d and label history (T18), D2 and the room (T19).\n"
  "- Whether L is running test_delta_recipe: no record; the main line's round-5 prompts (0754d61) say so, and the draft is "
  "read against them.\n"
  "- Session G's numbers attached to F's statements and session H's S84/S86 beyond the three EVAL-vs-BASE counts used in Q19.\n"
  "- Report 7's smoke-test disclosure (an unstored run) and the bit-for-bit CHECKs (read from the CHECK file, not re-run).\n"
  "- F's S88 (round 5): readings fixed at 40087cb, no record before the sixth draft; not in the draft.\n"
  "- The PDF.")
with open(OUT, "w") as fh:
    fh.write("\n".join(L) + "\n")
print(f"wrote {OUT}: sixth draft {len(live6)} rows; " + ", ".join(f"{k} {cnt6.get(k, 0)}" for k in ORDER)
      + f"; removed/superseded in v6 {len(gone6)}; sixth-pass rows {len(TROWS)} (" + ", ".join(f"{k} {cntT.get(k, 0)}" for k in ORDER)
      + f"); report 7 rows {len(R7ROWS)} (" + ", ".join(f"{k} {cntR7.get(k, 0)}" for k in ORDER) + ")")
for r in rows6:
    if r["status"] != "MATCH":
        print(f"  {r['status']:8} {r.get('id', ''):4} {lstr(r['loc6']):14} {r['tex']:10} {r['item'][:100]}")
for r in R7ROWS:
    if r["status"] != "MATCH":
        print(f"  R7 {r['status']:8} {r['tex']:16} {r['item'][:100]}")
for r in gone6:
    print(f"  {r['state6'][:13]} {r.get('id') or r['tex']:8} {r['status']:9} -> {r['sup6']}: {ALLT[r['sup6']]['status']}")
for x in flips6:
    print("  FLIP", x)
for r in TROWS:
    print(f"  {r['id']:4} {r['status']:8} {r['rec'][:200]}")
