#!/usr/bin/env python3
"""check_rev10_1.py -- Referee V on Revision 10.1 (846d0c8:multichannel_hebbian_report_v10.tex, 596 lines, cited tex101:NNN)
against Revision 10 as merged (585b666, 591 lines, cited tex10:NNN) and the referee's pass over it (rev10/referee_pass.md).

    python3 -I docs/review_2026-10/rev10/check_rev10_1.py > docs/review_2026-10/rev10/check_rev10_1_out.txt

Stdlib, pdftotext and docs/review_2026-10/rlib.py. Reads git objects only (git show / log / rev-parse / merge-base);
trains nothing, runs no session code, writes nothing but stdout (pdftotext writes into a temp dir).
Sections: 0 identity of the merge; 1 the changed lines; 2 every item of referee_pass.md sections 1 and 2 (old wording
present in tex10 and gone from tex101, new wording present); 3 the numbers the new text states, recomputed; 4 the
provenance rows; 5 the PDF (check_pdf_v10.py's method, adapted); 6 README and STATUS.md.
"""
import difflib
import json
import math
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import rlib  # noqa: E402

NEW, OLD, MERGE = "846d0c8", "585b666", "264e227"
TEXP, PDFP = "multichannel_hebbian_report_v10.tex", "multichannel_hebbian_report_v10.pdf"


def git(*a):
    return subprocess.check_output(("git", "-C", rlib.ROOT) + a).decode()


def gitb(*a):
    return subprocess.check_output(("git", "-C", rlib.ROOT) + a)


def show(ref, path):
    return git("show", f"{ref}:{path}")


def ctime(ref):
    return git("log", "-1", "--format=%cI", ref).strip()


def anc(a, b):
    return subprocess.call(("git", "-C", rlib.ROOT, "merge-base", "--is-ancestor", a, b)) == 0


T10 = show(OLD, TEXP).split("\n")
T101 = show(NEW, TEXP).split("\n")
NEWTIME = ctime(NEW)
FAIL = []


def say(s=""):
    print(s)


# ══ 0. identity ═══════════════════════════════════════════════════════════════════════════════════════════════════
say("=" * 110)
say("0. IDENTITY")
say("=" * 110)
say(f"846d0c8 committed {NEWTIME}; 264e227 committed {ctime(MERGE)}; parents of 264e227: {git('log', '-1', '--format=%p', MERGE).strip()}")
say(f"846d0c8 ancestor of 264e227: {anc(NEW, MERGE)}; 585b666 ancestor of 846d0c8: {anc(OLD, NEW)}")
for f in (TEXP, PDFP, "README.md", "STATUS.md", "CITATION.cff"):
    a, b = git("rev-parse", f"{NEW}:{f}").strip(), git("rev-parse", f"{MERGE}:{f}").strip()
    say(f"  {f}: 846d0c8 {a[:12]} 264e227 {b[:12]} equal {a == b}")
say(f"lines: tex10 {len(T10) - (T10[-1] == '')} tex101 {len(T101) - (T101[-1] == '')}")
for f in ("referee_pass.md", "check_rev10_A.md", "check_rev10_FG.md", "check_rev10_HE.md", "audit_passes.md"):
    p = "docs/review_2026-10/rev10/" + f
    say(f"  {p}: 846d0c8 == review-V HEAD: {git('rev-parse', f'{NEW}:{p}') == git('rev-parse', f'HEAD:{p}')}")

# ══ 1. changed lines ══════════════════════════════════════════════════════════════════════════════════════════════
say("\n" + "=" * 110)
say("1. CHANGED LINES (difflib over tex lines)")
say("=" * 110)
sm = difflib.SequenceMatcher(None, T10, T101, autojunk=False)
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag != "equal":
        say(f"  {tag:8s} tex10:{i1 + 1}-{i2}  ->  tex101:{j1 + 1}-{j2}")

# ══ 2. items ══════════════════════════════════════════════════════════════════════════════════════════════════════
say("\n" + "=" * 110)
say("2. ITEMS OF referee_pass.md: old wording on its tex10 line / anywhere in tex101; new wording on its tex101 line")
say("=" * 110)
WHOLE101 = "\n".join(T101)
ITEMS = [
    ("1.1", 495, "and of this revision's additions from the records and found them to reproduce", 499,
     "of this revision's additions it had checked \\texttt{test\\_delta\\_recipe}'s L half and pooled block, the GPU run and R's Phase~5 before the draft, and after the merge it checked the rest"),
    ("1.1b", 495, "have been recomputed there and not yet refereed by session~V", 499, None),
    ("1.2a", 44, "every number recomputed from the records", 44,
     "which recomputed \\texttt{test\\_delta\\_recipe}, the GPU run and H's screens and read the other screens' reports"),
    ("1.2b", 495, "the fifth round's screens have been recomputed there", 499, None),
    ("1.3a", 44, "checked in two passes by a session that did not write them", 44,
     "a separate checking agent run from the author's own session, whose files git cannot order before the third version"),
    ("1.3b", 495, "in two passes, by a session that did not write them", 499, None),
    ("1.4", 481, "all but one of the 29", 485, "all but one of the 30"),
    ("1.5a", 358, "screened once, session~H's S86, $8/10$ against $9/10$", 360, "screened twice, session~H's S86 and S87a, $24/30$ against $26/30$"),
    ("1.5b", 464, "screened once (session~H's S86)", 468, "screened twice (session~H's S86 and S87a)"),
    ("1.5c", 486, "screened once (S86)", 490, "screened twice (S86, S87a)"),
    ("1.6", 266, "within about 1200 updates of its learning rate reaching $10^{-3}$", 268,
     "by 3600 in the first arm (whose rate is $10^{-3}$ from the start) and 1200--6000 updates after the switch in the slow one"),
    ("1.7", 283, "the gate's own distance from uniform at every position", 285, "each position's gate's distance from the sequence's own mean gate"),
    ("1.8F", 537, "\\texttt{d5851b1} & 322; 102; 30; 40; 48; 42\\\\", 542, "\\texttt{a199d5e} & 322; 102; 30; 40; 48; 42; 78\\\\"),
    ("1.8G", 538, "\\texttt{076005d} & 22; 139; 50; 150; 136; 82\\\\", 543, "\\texttt{e46799a} & 22; 139; 50; 150; 136; 82; 142\\\\"),
    ("1.8H", 536, "\\texttt{5b5deec} & 318; 300; 82; 62; 52; 82; 32\\\\", 541, "\\texttt{347bd25} & 318; 300; 82; 62; 52; 82; 32; 84\\\\"),
    ("1.8E", 535, None, 540, "\\texttt{419368a} & 19; 20\\\\"),
    ("1.9", 70, "the three drafted specifications are not ready", 72, "the three specifications it reviewed are not ready"),
    ("2 L-reboot", 264, "during the projection's timing, with a GPU job beside it", 266, "which the author attributes to a GPU job running beside it"),
    ("2 date", 424, "Each is corrected where it stands, with the date.", 428, "Each is corrected where it stands (\\texttt{76851eb})."),
    ("2 robust-67", 67, "the key term's constants are robust;", 69, "the key term's constants are robust on a ceiling screen;"),
    ("2 robust-217", 217, "and the drafted constants stand).", 219, "a variant at a true rate of $0.8$ still reads ``robust'' $38\\%$ of the time"),
    ("2 0.13", 245, "would both bind with probability about $0.13$, and", 247,
     "drawn from these ten would both bind with probability about $0.13$ (two fresh seeds on one machine about $0.16$ at that rate, on both machines $0.03$)"),
    ("2 delays", 245, "the slow start only delays", 247, "the slow start hurt: no SLOW run bound and SLOW ended lower on every seed that bound or came close under the other recipe"),
    ("2 S69", 245, "screened that first (S69,", 247, "screened that first (E's S69, not F's;"),
    ("2 fifth-round label", 44, "fifth round (E's batches~20 and~21", 44, "fifth round (E's batches~20 and~21"),
    ("2 -0.02", 245, "$-0.02$ to $+0.41$", 247, "$-0.01$ to $+0.41$"),
    ("2 diverges", 266, "never diverges by stream", 268, "at the end it does not diverge by stream in any run"),
    ("2 only-L1", 266, "stays of order 1 only at the first layer's value positions", 268,
     "the first layer's in 13 of 20 runs, a later layer's in 8 run-layers, and the context position in one"),
    ("2 377", 377, "everywhere but the first layer's value positions", 381, "collapses to about zero except at some value positions"),
    ("2 487", 487, "its reading on the horizon itself read ``it does not'',", 491,
     "the direct horizon screen (S88b) is untested by its rule, with its failures stream confusions at $0.5$ rather than faded memory"),
    ("2 485", 485, "The hinge reads key identities;", 489, "discovers $17/20$ on a task where that table selects the keys and the contexts (S87)"),
    ("2 S75b days", 481, "about three days for the second stage", 485, "about two and a half days for the second stage by the specification's own run times"),
    ("2 ref-42", 42, "The referee also checked this revision's first draft", 42, "The referee also checked this revision's first draft"),
    ("2 ref-245", 245, "completed after this revision's first draft", 247, "completed after Revision~9's first draft"),
    ("2 ref-364", 364, "What it found in this revision's first draft", 366, "What it found in Revision~9's first draft"),
    ("2 ref-443", 443, "The same happened to this revision's first draft", 447, "The same happened to Revision~9's first draft"),
    ("2 ref-541", 541, "this revision's checks (\\texttt{e8fbb25}", 546, "Revision~9's checks (\\texttt{e8fbb25}"),
    ("2 ref-404 (not listed)", 404, "This revision's first draft repeated the failure in kind", 408, "This revision's first draft repeated the failure in kind"),
    ("2 F-programme", 268, "The two screens close session~F's programme", 270, "The two screens closed session~F's fourth round"),
]
for tag, ol, of, nl, nf in ITEMS:
    o_on = of is None or of in T10[ol - 1]
    o_gone = of is None or of not in WHOLE101
    n_on = nf is None or nf in T101[nl - 1]
    if not (o_on and n_on):
        FAIL.append(tag)
    say(f"  {tag:22s} old on tex10:{ol} {o_on}; old gone from tex101 {o_gone}; new on tex101:{nl} {n_on}")
say(f"  anchors not found: {FAIL}")
# the referee's own carried-over list, by case
low = [i + 1 for i, l in enumerate(T10) if "this revision's first draft" in l or "this revision's checks" in l]
anyc = [i + 1 for i, l in enumerate(T10) if re.search(r"this revision's (first draft|checks)", l, re.I)]
say(f"  tex10 lines with 'this revision's first draft/checks': case-sensitive {low}; case-insensitive {anyc}")
say(f"  tex101 lines still with it (case-insensitive): {[i + 1 for i, l in enumerate(T101) if re.search(r'this revision.s (first draft|checks)', l, re.I)]}")

# ══ 3. numbers ════════════════════════════════════════════════════════════════════════════════════════════════════
say("\n" + "=" * 110)
say("3. NUMBERS IN THE NEW TEXT, RECOMPUTED")
say("=" * 110)
# 3.1 rows of the referee's checks
rows = {}
for f in ("check_rev10_A.md", "check_rev10_FG.md", "check_rev10_HE.md"):
    t = show(NEW, "docs/review_2026-10/rev10/" + f)
    m = re.search(r"\*\*(\d+) rows", t) or re.search(r"TOTAL (\d+) rows", t) or re.search(r"\*\*total\*\* \| \*\*(\d+)\*\*", t)
    rows[f] = int(m.group(1))
say(f"3.1 rows {rows} sum {sum(rows.values())} (tex101:370 says 228)")
# 3.2 the sixteen places
places = {"1.1": [495], "1.2": [44, 495], "1.3": [44, 495], "1.4": [481], "1.5": [358, 464, 486], "1.6": [266], "1.7": [283],
          "1.8": [535, 536, 537, 538], "1.9": [70]}
say(f"3.2 statements {len(places)}, places {sum(len(v) for v in places.values())}")
rp = show(NEW, "docs/review_2026-10/rev10/referee_pass.md")
lst = rp.split("They are:")[1].split("- **About fifteen")[0]
say(f"    referee_pass.md's own list under 'They are:' cites {sorted(set(re.findall(r'§1\.(\d)', lst)))} (1.9 present: {'1.9' in lst})")
say(f"    tex101:370's list names the index row (1.9): {'three drafted' in T101[369] or 'specifications it reviewed' in T101[369]}")
# 3.3 S75 gated runs
RES75 = "explore_out/I/gpu/s75_S4_NVIDIA_GeForce_GTX_1660_SUPER_results.json"
st75 = json.loads(show("4371824", RES75))
R75 = {}
for k, r in st75["runs"].items():
    a, s = k.split("|")[:2]
    if r.get("ok"):
        R75.setdefault(a, {})[int(s)] = r
GATED = ("WIDE_SPLIT", "REG3", "WIDE_SLOW")
p3600 = {(a, s): max(next(c for c in r["curve"] if c["step"] == 3600)["probe"][L]["acc"] for L in ("L2", "L3", "final"))
         for a in GATED for s, r in R75[a].items()}
p1200 = [r["curve"][0]["probe"]["L2"]["acc"] for a in GATED for r in R75[a].values()]
say(f"3.3 S75 at 4371824: runs {len(st75['runs'])} {dict((a, len(v)) for a, v in R75.items())}; gated {sum(len(R75[a]) for a in GATED)}; "
    f"above 0.27 at 3600 {[(a, s, round(v, 3)) for (a, s), v in p3600.items() if v > 0.27]}; L2 at 1200 {min(p1200):.3f}-{max(p1200):.3f}")
st75_559 = json.loads(show("559b942", RES75))
say(f"    S75 runs at 559b942 (the I row's commit): {len(st75_559['runs'])}; at 4371824: {len(st75['runs'])}")
# 3.4 H: S86 + S87a firings and mistargets
st86 = rlib.load("9b72bf6", "explore_out/H/allmass_results.json")
st87 = rlib.load("347bd25", "explore_out/H/allmass2_results.json")
tot_f, tot_m, cost = 0, 0, []
for nm, st, seeds in (("S86", st86, range(2330, 2340)), ("S87a", st87, range(2600, 2620))):
    A = rlib.runs(st, "ALLMASS")
    B = rlib.runs(st, "BASE")
    f = [(s, c) for s in seeds for c in A[s].get("checks") or [] if c.get("fired")]
    mis = [(s, c["step"]) for s, c in f if not c["op_ok"]]
    kA = sum(rlib.bound_routed(A[s]) for s in seeds)
    kB = sum(rlib.bound_routed(B[s]) for s in seeds)
    a_only, b_only, _ = rlib.pairs(A, B, rlib.bound_routed, list(seeds))
    tot_f += len(f)
    tot_m += len(mis)
    for s, step in mis:
        cost.append((nm, s, step, "ALLMASS bound routed" if rlib.bound_routed(A[s]) else f"ALLMASS failed ({rlib.outcome_k(A[s])})",
                     "BASE bound routed" if rlib.bound_routed(B[s]) else "BASE failed"))
    say(f"3.4 {nm}: ALLMASS {kA}/{len(seeds)}, BASE {kB}/{len(seeds)}, ALLMASS only {a_only}, BASE only {b_only}; ALLMASS firings {len(f)}, "
        f"own pair missed {len(mis)} {mis}")
say(f"    total firings {tot_f}, mistargets {tot_m}; each mistarget's run: {cost}")
say(f"    pooled: within one pair on each screen (|BASE only - ALLMASS only| <= 1) as listed above")
# 3.5 power and probabilities
p_rob = {p: sum(math.comb(10, j) * (1 - p) ** j * p ** (10 - j) for j in range(2)) for p in (0.8, 0.9)}
say(f"3.5 P('robust' | variant rate p, drafted arm 10/10) {p_rob}; two of these ten 4*3/90 = {4 * 3 / 90:.3f}; "
    f"fresh, one machine 0.4^2 = {0.4 ** 2:.3f}; both machines {0.4 ** 4:.4f}")
# 3.6 E batch 20 / 21
st20 = rlib.load("419368a", "explore_out/s67_continue_results.json")
st21 = rlib.load("419368a", "explore_out/oracle_d128_results.json")
n20 = sum(1 for r in st20["runs"].values() if not r.get("dry"))
n21 = sum(1 for r in st21["runs"].values() if not r.get("dry"))
C = rlib.runs(st20, "C_P16_N256")


def at(r, step):
    return next((c for c in r["curve"] if c[0] == step), None)


d256 = {s: at(C[s], 86400)[1] - at(C[s], 43200)[1] for s in (270, 272, 273, 274, 275, 276)}
say(f"3.6 E batch 20 store runs {n20}, batch 21 store runs {n21}; N256 one-to-one change 43,200->86,400: "
    f"{ {s: round(v, 4) for s, v in d256.items()} } -> {min(d256.values()):+.4f} to {max(d256.values()):+.4f}")
O = {a: rlib.runs(st21, a) for a in ("ORC_A_D128", "ORC_S_D128")}
fin = {a: {s: r["curve"][-1][1] for s, r in O[a].items()} for a in O}
bA = {s: rlib.bound(r) for s, r in O["ORC_A_D128"].items()}
close = [s for s in bA if bA[s] or fin["ORC_A_D128"][s] >= 0.9]
say(f"    ORC_S bound {sum(rlib.bound(r) for r in O['ORC_S_D128'].values())}/10; ORC_A bound or >= 0.9: {sorted(close)}; "
    f"SLOW - A on those: { {s: round(fin['ORC_S_D128'][s] - fin['ORC_A_D128'][s], 3) for s in sorted(close)} }")
# 3.7 F S88a beta
W = rlib.load("a199d5e", "explore_out/F/write_results.json")
WA = {a: rlib.runs(W, a) for a in ("DELTA_LEARNED", "DELTA_LEARNED_SLOW", "O_DELTA_LEARNED")}


def bt(r):
    return {int(k): v for k, v in r["beta_traj"].items()}


def first_below(r, th=0.1):
    t = bt(r)
    return next((st for st in sorted(t) if all(L["median"] < th for L in t[st])), None)


fb = {a: sorted(first_below(r) for r in WA[a].values()) for a in WA}
end = {a: {s: bt(r)[max(bt(r))] for s, r in WA[a].items()} for a in WA}
scr = ("DELTA_LEARNED", "DELTA_LEARNED_SLOW")
l1 = sum(end[a][s][0]["by_role"]["val"] >= 0.5 for a in scr for s in end[a])
l23 = [(a, s, i + 1, round(L["by_role"]["val"], 2)) for a in WA for s, e in end[a].items() for i, L in enumerate(e) if i > 0 and L["by_role"]["val"] >= 0.5]
ctx = [(a, s, i + 1, round(L["by_role"]["ctx"], 2)) for a in WA for s, e in end[a].items() for i, L in enumerate(e) if L["by_role"]["ctx"] >= 0.5]
key = max(L["by_role"]["key"] for a in WA for e in end[a].values() for L in e)
div_end = sum(any(L["eta_key_by_stream"] >= 0.5 for L in e) for a in WA for e in end[a].values())
say(f"3.7 F S88a: first all-layer median < 0.1: DELTA_LEARNED {fb['DELTA_LEARNED']}; SLOW {fb['DELTA_LEARNED_SLOW']} "
    f"(after the switch at 2400: {[x - 2400 for x in fb['DELTA_LEARNED_SLOW']]}); learned perfect gate {fb['O_DELTA_LEARNED']}")
say(f"    end: layer-1 VAL >= 0.5 in {l1}/20 screen runs; layers 2-3 VAL >= 0.5 in {len(l23)} run-layers {l23}; CTX >= 0.5 {ctx}; "
    f"KEY max {key:.3f}; runs diverging by stream at the end {div_end}")
# 3.8 run counts of the provenance rows
F8 = (len(W["runs"]), len(rlib.load("a199d5e", "explore_out/F/horizon_results.json")["runs"]))
G7 = len(rlib.load("e46799a", "explore_out/G/s87_results.json")["runs"])
H9 = (len(st87["runs"]), len(rlib.load("347bd25", "explore_out/H/dose_results.json")["runs"]))
say(f"3.8 run counts: E {n20}; {n21}  F S88a+S88b {F8} = {sum(F8)}  G S87 {G7}  H S87a+S87c {H9} = {sum(H9)}  S86 {len(st86['runs'])}")
# 3.9 S75b stage 2 on the 3070 alone (the spec's 'Cost' table: per-run 3070 hours x stage-2 runs)
spec = show("8d2e89f", "explore_out/I/s75b_spec_draft.md")
lo, hi = 2 * 1.15 + 10 * 1.0 + 10 * 1.25 + 10 * 1.25 + 10 * 1.7, 2 * 1.15 + 10 * 1.0 + 10 * 1.25 + 10 * 1.25 + 10 * 2.0
say(f"3.9 S75b stage 2 on the 3070 alone: {lo:.1f}-{hi:.1f} h = {lo / 24:.2f}-{hi / 24:.2f} days (tex101:485 'about two and a half' = 60 h); "
    f"spec table rows present: {'| ORACLE | 2.19 (2.5)' in spec and '1.7–2.0 (fits)' in spec}; STATUS.md at 846d0c8 says 'about three days': "
    f"{'about three days on one card' in show(NEW, 'STATUS.md').replace(chr(10), ' ').replace('  ', ' ')}")

# ══ 4. provenance ═════════════════════════════════════════════════════════════════════════════════════════════════
say("\n" + "=" * 110)
say(f"4. PROVENANCE ROWS (tex101:540-546) against git, at 846d0c8's commit time {NEWTIME}")
say("=" * 110)
BR = {"E": "outside-ideas", "F": "explore-F", "G": "explore-G", "H": "explore-H", "I": "explore-I", "R": "replicate-R", "V": "review-V"}
for n in range(540, 547):
    line = T101[n - 1]
    hs = re.findall(r"\\texttt\{([0-9a-f]{7})\}", line)
    who = line.split("&")[0].strip()
    say(f"  tex101:{n} [{who}]")
    for h in hs:
        try:
            full = git("rev-parse", "--verify", "-q", h + "^{commit}").strip()
        except subprocess.CalledProcessError:
            say(f"    {h}: DOES NOT EXIST")
            continue
        t = ctime(h)
        subj = git("log", "-1", "--format=%s", h).strip()[:110]
        on = [b for b in BR.values() if anc(h, "origin/claude/" + b)] + (["main"] if anc(h, NEW) else [])
        say(f"    {h}: {t} before 846d0c8 {t < NEWTIME}; on {on}; {subj}")
say("  each branch's last commit at 846d0c8's commit time, and its last report-like commit:")
REP = {"E": ["explore_out/README.md"], "F": ["explore_out/F/report_*.md"], "G": ["explore_out/G/report_*.md"], "H": ["explore_out/H/report_*.md"],
       "I": ["explore_out/I/report_*.md"], "R": ["replicate_report.md"], "V": ["docs/review_2026-10/*.md", "docs/review_2026-10/*/*.md"]}
for s, b in BR.items():
    ref = "origin/claude/" + b
    last = git("log", "-1", "--format=%h %cI %s", f"--before={NEWTIME}", ref).strip()[:120]
    lrep = git("log", "-1", "--format=%h %cI %s", f"--before={NEWTIME}", ref, "--", *REP[s]).strip()[:120]
    say(f"    {s}: last {last}")
    say(f"       last report-like: {lrep}")
rp1 = show("2c250bd", "docs/review_2026-10/rev10/referee_pass.md")
say(f"  2c250bd's referee_pass.md: 'Eleven statements' {'Eleven statements' in rp1}; 'Nine statements, in sixteen places' "
    f"{'Nine statements, in sixteen places' in rp1}; the version merged in 846d0c8 is 480a2de's: "
    f"{git('rev-parse', '480a2de:docs/review_2026-10/rev10/referee_pass.md') == git('rev-parse', NEW + ':docs/review_2026-10/rev10/referee_pass.md')}")
say(f"  0e70640 adds round6_A3.md: {'docs/review_2026-10/round6_A3.md' in git('show', '--stat', '--format=', '--name-only', '0e70640')}")
say(f"  f5d3893 subject names batch 20 complete: {'Batch 20 complete' in git('log', '-1', '--format=%s', 'f5d3893')}")
say("  rows no item of referee_pass.md covers (I and R):")
p5 = [l for l in show("31c39fc", "records/phase5.jsonl").split("\n") if l.strip()]
say(f"    I row tex101:544 cites 559b942 and 'pilot; 41'; S75 store at 559b942 {len(st75_559['runs'])} runs, at 4371824 {len(st75['runs'])}; "
    f"report_2.md commits after 559b942 before 846d0c8: {git('log', '--format=%h %cI', '559b942..origin/claude/explore-I', f'--before={NEWTIME}', '--', 'explore_out/I/report_2.md').split()}")
say(f"    R row tex101:545: '{T101[544][-40:]}'; R's Phase 5 report 31c39fc ({ctime('31c39fc')}) cited anywhere in tex101: "
    f"{'31c39fc' in WHOLE101}; records/phase5.jsonl lines at 31c39fc {len(p5)}")
say("  merge timing of 10.1 (tex101:447 'The referee now reads ... every draft before it is merged'): "
    f"846d0c8 {NEWTIME}; ed9ca2b6 {ctime('ed9ca2b6')}; 264e227 {ctime(MERGE)}; "
    f"sentence present in tex101:447: {'every draft before it is merged' in T101[446]}")

# ══ 5. PDF ════════════════════════════════════════════════════════════════════════════════════════════════════════
say("\n" + "=" * 110)
say("5. THE PDF (check_pdf_v10.py's method, REF 846d0c8, PREV 585b666)")
say("=" * 110)


def words(line):
    line = re.sub(r"%.*", "", line)
    line = re.sub(r"\$[^$]*\$", " ", line)
    line = re.sub(r"\\(texttt|ref|label|cite|eqref|url|href|includegraphics|input)\{[^}]*\}", " ", line)
    line = re.sub(r"\\[A-Za-z]+\*?", " ", line)
    return [w.lower() for w in re.findall(r"[A-Za-z]{3,}", line)]


def pdf_words(ref):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "v.pdf")
        open(p, "wb").write(gitb("show", f"{ref}:{PDFP}"))
        subprocess.check_call(["pdftotext", p, os.path.join(d, "v.txt")])
        txt = open(os.path.join(d, "v.txt"), errors="replace").read()
        info = subprocess.run(["pdfinfo", p], capture_output=True, text=True).stdout
    txt = re.sub(r"-\n(?=[a-z])", "", txt)
    return " ".join(re.findall(r"[A-Za-z]{3,}", txt)).lower(), txt, info


pdfw, pdftxt, pinfo = pdf_words(NEW)
begin = next(i for i, l in enumerate(T101) if l.startswith("\\begin{document}")) + 1
miss, tot, found = [], 0, 0
for n, line in enumerate(T101, 1):
    if n <= begin:
        continue
    w = words(line)
    rr = [" ".join(w[i:i + 6]) for i in range(0, max(0, len(w) - 5), 3)]
    if not rr:
        continue
    ok = [r in pdfw for r in rr]
    tot += len(rr)
    found += sum(ok)
    if not all(ok):
        miss.append((n, sum(ok), len(rr)))
say(f"{NEW}: {found}/{tot} six-word runs of the tex found in the PDF text ({100 * found / tot:.2f}%)")
say(f"lines with a run not found: {len(miss)}")


def allruns(lines):
    out = set()
    for line in lines:
        w = words(line)
        out |= {" ".join(w[i:i + 6]) for i in range(0, max(0, len(w) - 5))}
    return out


rn, ro = allruns(T101), allruns(T10)
on, oo = rn - ro, ro - rn
fn, fo = sum(r in pdfw for r in on), sum(r in pdfw for r in oo)
say(f"six-word runs only in {NEW}'s tex: {len(on)}, found in the PDF: {fn} ({100 * fn / max(1, len(on)):.1f}%)")
say(f"six-word runs only in {OLD}'s tex: {len(oo)}, found in the PDF: {fo} ({100 * fo / max(1, len(oo)):.1f}%)")
say(f"  only-old runs found in the new PDF: {sorted(r for r in oo if r in pdfw)[:10]}")
for tag, ol, of, nl, nf in ITEMS:
    lr = allruns([T101[nl - 1]]) - allruns([T10[ol - 1]]) if ol <= len(T10) else allruns([T101[nl - 1]])
    if lr:
        say(f"  tex101:{nl} ({tag}): new runs {len(lr)}, in the PDF {sum(r in pdfw for r in lr)}")
for n in (46, 370):
    lr = allruns([T101[n - 1]])
    say(f"  tex101:{n} (added paragraph): runs {len(lr)}, in the PDF {sum(r in pdfw for r in lr)}")
say(f"  'Revision 10.1' in the PDF text: {'Revision 10.1' in pdftxt}; 'Revision~10.1' rendered count {pdftxt.count('Revision 10.1')}")
say("  pdfinfo: " + "; ".join(l.strip() for l in pinfo.split("\n") if l.split(":")[0] in ("Pages", "CreationDate", "ModDate", "Producer")))
say(f"  old PDF ({OLD}) pdfinfo: " + "; ".join(l.strip() for l in pdf_words(OLD)[2].split("\n") if l.split(":")[0] in ("Pages", "CreationDate")))
for n, a, b in miss:
    if any(n == it[3] for it in ITEMS) or n in (44, 46, 370, 447, 499):
        say(f"  changed line tex101:{n} with a run not found: {a}/{b}")

# ══ 6. README and STATUS ══════════════════════════════════════════════════════════════════════════════════════════
say("\n" + "=" * 110)
say("6. README.md AND STATUS.md AT 846d0c8")
say("=" * 110)
for f in ("README.md", "STATUS.md"):
    t = show(NEW, f).split("\n")
    for pat in (r"this revision's (first )?draft", r"this revision's checks", r"did not write|had not written", r"every number recomputed",
                r"screened once", r"of the 29", r"three days", r"only delays", r"close[sd]? (session )?F's"):
        hits = [i + 1 for i, l in enumerate(t) if re.search(pat, l, re.I)]
        if hits:
            say(f"  {f}: /{pat}/ at lines {hits}")
say(f"\nanchors not found: {FAIL or 'none'}")
