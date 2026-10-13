#!/usr/bin/env python3
"""check_G_v6.py — the referee's SIXTH-PASS check of Revision 9's statements about Session G, run on the draft's SIXTH
version (42c4557, committed 2026-10-12 23:14:04 UTC, tex6:NNN). A copy of check_G_v5.py (fifth pass, draft c5e4508), changed
so that:
  * every first- to fourth-pass row is located in the SIXTH tex by text search (REMOVED if gone), as before;
  * every fifth-pass row (row5) is asserted on the fifth draft as in check_G_v5 and then located in the sixth (REMOVED5 if
    gone); its recomputation is kept unchanged;
  * rows are added (row6) for every sentence about Session G that changed between c5e4508 and 42c4557 (b5fd2f2's KL bound
    and the specification's expectation; 42c4557's fold-in of the fifth pass; c916996's status line), re-anchored on the
    new wording, with the number re-checked against S83's store, the specification's caveat and report 6;
  * the v5 assert that G's last commit before the draft is report 6 (GBR == G6) is replaced: G's records are still read at
    the commit that was the branch head at the fifth draft (076005d, report 6), and they are shown byte-identical at G's
    last commit before the sixth draft (GBR6); the status sentences (tex6:41 'the sessions' fifth round are running', the
    provenance row, the next steps) are checked against GBR6 in their own rows;
  * V6 revises the status / note of a carried row whose fragment survives in an amended sentence or whose truth moved
    with G's branch.
Writes docs/review_2026-10/rev9/check_G_v6.md and nothing else.  python3 docs/review_2026-10/rev9/check_G_v6.py
The v5 docstring follows.

check_G_v5.py — the referee's FIFTH-PASS check of Revision 9's statements about Session G, run on the draft's FIFTH
version (c5e4508, tex5:NNN). A copy of check_G_v4.py (fourth pass, draft c66cc76), changed so that:
  * every first-, second- and third-pass row is located in the FIFTH tex by text search (REMOVED if gone), as before;
  * every fourth-pass row (row4) is asserted on the fourth draft as in check_G_v4 and then located in the fifth
    (REMOVED4 if gone); its recomputation is kept unchanged;
  * rows are added (row5) for every sentence about Session G that changed between c66cc76 and c5e4508 (the referee's
    fourth-pass fixes e92e1c6, fabf505, fd2fa89, and the later fold-ins), re-anchored on the new wording, with the
    number re-checked against S83's store (the KL term at every recorded update, the register's write at every
    evaluation, the term's weights) and against the specification's caveat and report 6's own account;
  * V5 revises the note of a carried row whose fragment survives in an amended sentence;
  * the branch head of claude/explore-G is checked against the fifth draft's commit time.
Writes docs/review_2026-10/rev9/check_G_v5.md and nothing else.  python3 docs/review_2026-10/rev9/check_G_v5.py
The v4 docstring follows.

check_G_v4.py — the referee's FOURTH-PASS check of Revision 9's statements about Session G (S55-S57, S71-S73, S82, S83)
against the records, run on the draft's FOURTH version (c66cc76). A copy of check_G_v3.py (third pass, draft a5f67a2),
changed so that:
  * every fragment the FIRST pass quoted (asserted on its first-draft line, tex:NNN), every fragment the SECOND pass quoted
    (asserted on the second draft, tex2:NNN) and every fragment the THIRD pass quoted (asserted on the third draft,
    tex3:NNN) is located in the FOURTH tex by text search (all of a row's fragments on one line); the row is reported at
    its tex4 line, or listed as REMOVED;
  * every record recomputation is kept unchanged (S55-S73 at 93422cf, asserted byte-identical at 8973d48; S82 at 8973d48,
    asserted byte-identical at 076005d);
  * rows are added (row4) for every sentence about Session G that changed between a5f67a2 and c66cc76, re-anchored on the
    new wording, with the number re-checked against the record it cites;
  * S83 (report 6 at 076005d; spec b2f9907, started 1973126) is recomputed from its store: every arm's counts under
    explore_g_rhl_parent's outcomes, every paired comparison (discordant pairs, exact one-sided McNemar), each pre-fixed
    reading under explore_g_s83.py's rule, the validity arm, the phase-0 reproduction and phase-2 choice files, the
    readings' commit against the first record, and the descriptive quantities the draft uses;
  * where a fragment survives inside a sentence that was amended around it, the row carries a v4 status / note (V4 below).

Run from anywhere inside the repository:  python3 docs/review_2026-10/rev9/check_G_v4.py
It writes docs/review_2026-10/rev9/check_G_v4.md and nothing else. Python 3 stdlib + docs/review_2026-10/rlib.py; no
torch; no screen module is imported. Every record and every source file is read with `git show <ref>:<path>` (and the
seed scan with `git grep` on the fetched branch heads).

Pinned refs
  DRAFT  9bfa85a (first version, tex:NNN), f67bc07 (second, tex2:NNN), a5f67a2 (third, tex3:NNN), c66cc76 (fourth,
         tex4:NNN), multichannel_hebbian_report_v9.tex
  G      claude/explore-G at 93422cf (report 4) for S55-S57 and S71-S73 (asserted byte-identical at 8973d48)
  G5     8973d48 (report 5, S82): s82_devices_results.json, s82_best.json, report_5.md
  G6     076005d (report 6, S83): s83_results.json, s83_repro.json, s83_delta.json, s83_summary.json, s83.log, report_6.md,
         spec_draft_register.md; S83 spec b2f9907 (explore_g_s83.py, explore_g_s83_child.py)
  F      claude/explore-F at 2247298: F's S70 store rh_lite_results.json
  S82    83699a2 (S82 spec: explore_g_devices.py, explore_g_devices_child.py)
  MAIN   the main line, for the S56 specification's origin (docs/reading/*)
Outcome definitions are re-implemented from explore_g_rhl_parent.py's docstring (S71-S73, S82, S83) and explore_g_register.py's
(S56, S57); see the 'definitions' block in the output. The draft's values are typed in below; None is used as an input.
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

DRAFT, DRAFT2, DRAFT3, DRAFT4, TEXP = "9bfa85a", "f67bc07", "a5f67a2", "c66cc76", "multichannel_hebbian_report_v9.tex"
DRAFT5 = "c5e4508"
DRAFT6 = "42c4557"
G, F = "93422cf", "2247298"
G5 = "8973d48"
G6, S83SPEC, S83START = "076005d", "b2f9907", "1973126"
S82SPEC = "83699a2"
GBR_NOW = "origin/claude/explore-G"
# (v5) G's records as they stood at the fifth draft's commit: the last branch commit at or before it
GBR = git("rev-list", "-1", "--before=" + git("show", "-s", "--format=%cI", "c5e4508").strip(), GBR_NOW).strip()[:7]
# (v6) G's branch as it stood at the SIXTH draft's commit (42c4557, 2026-10-12T23:14:04Z): its last commit at or before it.
# The records are still read at GBR (= report 6's commit); GBR6 is used for the status rows, and the record files are
# compared between the two below.
GBR6 = git("rev-list", "-1", "--before=" + git("show", "-s", "--format=%cI", DRAFT6).strip(), GBR_NOW).strip()[:7]
MAIN = "origin/claude/bdh-growth-hebbian-inference-w90069"
OUT = os.path.join(HERE, "check_G_v6.md")
GD = "explore_out/G/"

TEX = git("show", f"{DRAFT}:{TEXP}").split("\n")
TEX2 = git("show", f"{DRAFT2}:{TEXP}").split("\n")
TEX3 = git("show", f"{DRAFT3}:{TEXP}").split("\n")
TEX4 = git("show", f"{DRAFT4}:{TEXP}").split("\n")
TEX5 = git("show", f"{DRAFT5}:{TEXP}").split("\n")
TEX6 = git("show", f"{DRAFT6}:{TEXP}").split("\n")
ROWS = []        # (tex4 line, origin 'tex:N' / 'tex2:N' / 'tex3:N' / 'new', item, draft, recomputed, status, note, kind, key)
REMOVED = []     # first-pass rows: (old tex line, needles, item, first-pass status)
REMOVED2 = []    # second-pass rows: (tex2 line, needles, item, second-pass status)
REMOVED3 = []    # third-pass rows: (tex3 line, needles, item, third-pass status)
REMOVED4 = []    # fourth-pass rows: (tex4 line, needles, item, fourth-pass status)
REMOVED5 = []    # fifth-pass rows: (tex5 line, needles, item, fifth-pass status)
STATUSES = ("DISAGREE", "PARTIAL", "NOT RECORD-CHECKABLE", "MATCH")
V3 = {}          # first needle -> (status, note): the v3 revision of a row whose fragment survives in an amended sentence
V4 = {}          # first needle -> (status, note): the v4 revision (applied after V3)
V5 = {}          # first needle -> (status, note): the v5 revision (applied after V4)
V6 = {}          # first needle -> (status, note): the v6 revision (applied after V5)
V3_GONE = []     # V3 / V4 entries whose row is REMOVED in the fifth draft


def tex(n):
    return TEX[n - 1]


def _nl(needles):
    return [needles] if isinstance(needles, str) else list(needles)


def locate(needles, T=None):
    """First line of the given draft (default: the sixth) that contains every fragment, or None."""
    for i, l in enumerate(TEX6 if T is None else T, 1):
        if all(nd in l for nd in _nl(needles)):
            return i
    return None


def apply_v3():
    """Revise the status / note of every row whose first fragment has a V3 entry. A V3 entry whose row is not in the fourth
    draft (REMOVED) is recorded in V3_GONE; every other must be used once. Then the V4 entries, each used exactly once."""
    for tag_, VV, gone_ok in (("V3", V3, True), ("V4", V4, True), ("V5", V5, True), ("V6", V6, False)):
        used = Counter()
        for i, r_ in enumerate(ROWS):
            if r_[8] in VV:
                st_, nt_ = VV[r_[8]]
                assert st_ in STATUSES, st_
                ROWS[i] = r_[:5] + (st_, nt_) + r_[7:]
                used[r_[8]] += 1
        assert all(v == 1 for v in used.values()), used
        for k_ in set(VV) - set(used):
            assert gone_ok and locate(k_) is None, f"unused revision {k_!r}"
            V3_GONE.append(f"{tag_}: {k_}")


def row(n, needles, item, draft, recomputed, status, note="", v2=None):
    """A first-pass row: fragments asserted on the first-draft line n, then located in the third tex by text search."""
    for nd in _nl(needles):
        assert nd in tex(n), f"tex:{n} does not contain {nd!r}"
    assert status in STATUSES, status
    n3 = locate(needles)
    if n3 is None:
        REMOVED.append((n, _nl(needles), item, status))
        return
    if v2:
        status, note = v2
        assert status in STATUSES, status
    ROWS.append((n3, f"tex:{n}", item, draft, recomputed, status, note, "carried1", _nl(needles)[0]))


def row2(needles, item, draft, recomputed, status, note=""):
    """A second-pass row: fragments asserted on the second draft, then located in the third tex by text search."""
    n2 = locate(needles, TEX2)
    assert n2 is not None, f"second draft does not contain {needles!r}"
    assert status in STATUSES, status
    n3 = locate(needles)
    if n3 is None:
        REMOVED2.append((n2, _nl(needles), item, status))
        return
    ROWS.append((n3, f"tex2:{n2}", item, draft, recomputed, status, note, "carried2", _nl(needles)[0]))


def row3(needles, item, draft, recomputed, status, note="", changed=True):
    """A third-pass row (a sentence of the third draft that changed since f67bc07, or a new check of an unchanged one):
    asserted on the third draft as in check_G_v3, then located in the fourth by text search (REMOVED3 if gone)."""
    n3 = locate(needles, TEX3)
    assert n3 is not None, f"third draft does not contain {needles!r}"
    if changed:
        assert locate(needles, TEX2) is None, f"not a changed sentence: second draft already contains {needles!r}"
    else:
        assert locate(needles, TEX2) is not None, f"changed sentence: second draft does not contain {needles!r}"
    assert status in STATUSES, status
    n4 = locate(needles)
    if n4 is None:
        REMOVED3.append((n3, _nl(needles), item, status))
        return
    ROWS.append((n4, f"tex3:{n3}", item, draft, recomputed, status, note, "carried3", _nl(needles)[0]))


def row4(needles, item, draft, recomputed, status, note="", changed=True):
    """A fourth-pass row (a sentence of the fourth draft that changed since a5f67a2, or a new check of an unchanged one):
    asserted on the fourth draft as in check_G_v4, then located in the sixth by text search (REMOVED4 if gone)."""
    n4 = locate(needles, TEX4)
    assert n4 is not None, f"fourth draft does not contain {needles!r}"
    if changed:
        assert locate(needles, TEX3) is None, f"not a changed sentence: third draft already contains {needles!r}"
    else:
        assert locate(needles, TEX3) is not None, f"changed sentence: third draft does not contain {needles!r}"
    assert status in STATUSES, status
    n5 = locate(needles)
    if n5 is None:
        REMOVED4.append((n4, _nl(needles), item, status))
        return
    ROWS.append((n5, f"tex4:{n4}", item, draft, recomputed, status, note, "carried4", _nl(needles)[0]))


def row5(needles, item, draft, recomputed, status, note="", changed=True):
    """A fifth-pass row (a sentence of the fifth draft that changed since c66cc76, or a new check of an unchanged one):
    asserted on the fifth draft as in check_G_v5, then located in the sixth by text search (REMOVED5 if gone)."""
    n5 = locate(needles, TEX5)
    assert n5 is not None, f"fifth draft does not contain {needles!r}"
    if changed:
        assert locate(needles, TEX4) is None, f"not a changed sentence: fourth draft already contains {needles!r}"
    else:
        assert locate(needles, TEX4) is not None, f"changed sentence: fourth draft does not contain {needles!r}"
    assert status in STATUSES, status
    n6 = locate(needles)
    if n6 is None:
        REMOVED5.append((n5, _nl(needles), item, status))
        return
    ROWS.append((n6, f"tex5:{n5}", item, draft, recomputed, status, note, "carried5", _nl(needles)[0]))


def row6(needles, item, draft, recomputed, status, note="", changed=True):
    """A row for a sentence of the sixth draft that changed since c5e4508 (located by text search; must be found).
    changed=False: a new check of a sentence that did not change (said so in the row's origin column)."""
    n6 = locate(needles)
    assert n6 is not None, f"sixth draft does not contain {needles!r}"
    if changed:
        assert locate(needles, TEX5) is None, f"not a changed sentence: fifth draft already contains {needles!r}"
    else:
        assert locate(needles, TEX5) is not None, f"changed sentence: fifth draft does not contain {needles!r}"
    assert status in STATUSES, status
    ROWS.append((n6, "new" if changed else "new check (line unchanged)", item, draft, recomputed, status, note, "new6",
                 _nl(needles)[0]))


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
for _f in ("s55_probe_results", "s56_register_results", "s57_latch_results", "s71_layers_results",
           "s72_register3_results", "s73_register3_delta_results"):
    assert git("rev-parse", f"{G}:{GD}{_f}.json") == git("rev-parse", f"{G5}:{GD}{_f}.json"), _f   # unchanged at report 5
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
row(261, "its gates route one-to-one with margin $1.00$ at layers 2 and 3, and its accuracy", "S56 REG_NUDGE end margin and VAL map at L2, L3 "
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
    "as report_2.md (layer 2); the S56 R2 reading that applied to REG_D is not stated by the draft (see 'Readings vs docstring')",
    v2=("MATCH", "as report_2.md (layer 2); the second draft now states that REG_D met R2's rule for 'discovered'"))
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
    "the Hebbian perfect gate binds at 1200 in the median run, not in every run",
    v2=("MATCH", "the second draft qualifies it in the same sentence: 'binds 10/10 (median 1200, range 1200--4800)'; "
        "'whose perfect gates bind at 1200' now reads as the layout's typical oracle transition"))
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
    "draft does not name it", v2=("MATCH", "the second draft now names 'the gate is not needed' as not applying (7/10)"))


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
row(263, "the single channel's bound runs at $0.93$--$1.00$;", "SINGLE_D bound runs, probe L3 K / V", "0.93-1.00",
    f"K {rng([k for k, _ in sk])}, V {rng([v for _, v in sk])}", "PARTIAL" if min(v for _, v in sk) < 0.9 else "MATCH",
    "the range is K's only; at V the stream is not found in every bound run")
half = {s: r for s, r in BD.items() if bound(r) and 0.4 <= r["end"]["margin"] <= 0.6}
hk = [pk(r) for r in half.values()]
row(263, "the half-routed baseline's at $0.86$--$1.00$", "BASELINE_D bound runs with end margin in 0.4-0.6 (half-routed), probe L3 "
    "K / V", "0.86-1.00", f"{len(half)} runs (margin {rng([r['end']['margin'] for r in half.values()])}); K {rng([k for k, _ in hk])}, "
    f"V {rng([v for _, v in hk])}; all 14 bound: K {rng([pk(r)[0] for r in BD.values() if bound(r)])}, V "
    f"{rng([pk(r)[1] for r in BD.values() if bound(r)])}", "PARTIAL" if min(v for _, v in hk) < 0.86 else "MATCH",
    "K's range only; V goes down to 0.79",
    v2=("PARTIAL", "the second draft labels the single channel's range 'at keys' and lists this one after it in the same "
        "parenthesis without the label; it is K's range only (V 0.79-0.99), so the parallel reading is right only if "
        "'at keys' is read as covering both"))
row(263, "which is a description of a $2\\times2$ the screen fixed as descriptive", "BASELINE_D 2x2 probe (>= 0.9 at K and V) x "
    "ROUTED-BOUND; S73 reading 'the delta stack carries the cue' (>= 10/20)", "descriptive",
    f"[[probe & routed, probe & not], [no probe & routed, no probe & not]] = "
    f"[[{sum(1 for r in BD.values() if min(pk(r)) >= 0.9 and rb(r))}, {sum(1 for r in BD.values() if min(pk(r)) >= 0.9 and not rb(r))}], "
    f"[{sum(1 for r in BD.values() if min(pk(r)) < 0.9 and rb(r))}, {sum(1 for r in BD.values() if min(pk(r)) < 0.9 and not rb(r))}]]; "
    f"reading: {sum(1 for r in BD.values() if min(pk(r)) >= 0.9)}/20 -> does not apply", "MATCH",
    "the draft does not state that the pre-fixed 'the delta stack carries the cue' did not apply",
    v2=("MATCH", "the second draft now states that 'the delta stack carries the cue' did not apply (the baseline 7/20)"))

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
    "means 'by the first recorded evaluation'",
    v2=("PARTIAL", "the fragment now opens the S82 paragraph ('The discovering runs fail by a key split that forms first'): "
        "the order agrees with the records, but it is 14 of the 16 failures on each memory in S72/S73 (16 of S82's 18 unbound "
        "REG3 runs), not every failure; unmarked as a description; and its subject should be the label-free runs that fail, "
        "not 'the discovering runs'"))
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
NEEDLE125 = {"register dimension": "16-dimensional register",
             "initialisation": "with $A$ initialised so that the gate equals the window gate at step 0, and a stop-gradient",
             "the earlier design": "Revision~8's design had it at layers 2--3 only"}
for it, d_, c_, s_, nt in reg_rows:
    row(125, NEEDLE125.get(it, "register gate (session~G)"), "register gate: " + it, d_, c_, s_, nt)
reg_rows2 = [
    ("initialisation", "A initialised so that the gate equals the window gate at step 0 up to a N(0, 0.01^2) perturbation of "
     "its register and residual blocks", "A_v = W_in, W_g2 = W_g (copies); A_r (residual block) and A_R (register block) ~ "
     f"N(0, 0.01^2) ({ln('explore_g_regmodel.py', 'm.A_r = nn.Parameter(torch.randn(hg, D, generator=g) * 0.01)')})", "MATCH", ""),
    ("the earlier design", "session G's first design (S56) had a register gate at layers 2-3 only, reading the raw embedding "
     "rather than the window, with the window gate at layer 1", f"S56 GATED = (2, 3) ({ln('explore_g_regmodel.py', 'GATED = (2, 3)')}); "
     f"gate input v_t (explore_g_register.py:{lineno(G, 'explore_g_register.py', 'A_v v_t + A_R R_{t-1}')}); layer 1 LOCAL3", "MATCH",
     "attributed to session G's S56 rather than to 'Revision 8's design'"),
]
row2(["up to a $\\mathcal{N}(0, 0.01^2)$ perturbation of its register and residual blocks"], "register gate: " + reg_rows2[0][0],
     *reg_rows2[0][1:])
row2(["session~G's first design (S56) had a register gate at layers 2--3 only, reading the raw embedding rather than the window"],
     "register gate: " + reg_rows2[1][0], *reg_rows2[1][1:])

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
    "see 'Descriptive statements: marked or not'",
    v2=("PARTIAL", "the second draft adds 'those are now marked'. Marked now: tex2:261 (the whole interpretation, with its "
        "two crossings), tex2:361, tex2:265's firing description ('as a description'). Still unmarked: tex2:384 '(it needed "
        "layer~1)'; 'key split(s) that form(s) first' at tex2:36, :265, :316, :455; tex2:265's 'worse than nothing, because it "
        "delays the register's write along with the split' and tex2:384's 'because it slows the write too'"))
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
    "the overturned prediction is a reading; the parenthesis is the tex:261 interpretation, unmarked",
    v2=("PARTIAL", "unchanged in the second draft: '(it needed layer~1)' is still the tex2:261 interpretation, unmarked here "
        "although tex2:261 and tex2:361 now mark it as ours and as crossing two tasks and two gate inputs"))
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
    "the count is a description; its pre-fixed reading did not apply, which the conclusion does not say",
    v2=("MATCH", "the count is a description (S72 'not discovered', S73 'discovered (register)' does not apply); on S82's fresh "
        "seeds the same arm is 2/20 (Hebbian) and 2/10 (delta), which 'a fifth' does not cover; see the tex2:465 row"))
row(464, "Session~G's key-split devices for the register (S82)", "'Next' list: S82", "next", f"phase 1 complete before the draft "
    f"({best_add.split()[1]}); phase 2 running", "MATCH", "phase 2 (the chosen device on delta memory) is what remains")

# ═════════════════════════════════════════════════════════════════════════════
# SECOND DRAFT: S82 (report 5, 8973d48) recomputed from the records
# ═════════════════════════════════════════════════════════════════════════════
S82 = load(G5, GD + "s82_devices_results.json")
X = {a: runs(S82, a) for a in sorted({k.split("|")[0] for k in S82["runs"]})}
_r82 = list(S82["runs"].values())
integ82 = (len(_r82), sum(1 for r in _r82 if r.get("ok")), sum(1 for r in _r82 if r.get("attempt", 1) != 1),
           sum(1 for r in _r82 if r["transition"] == transition(r["curve"])))
assert integ82[1] == integ82[0] and integ82[3] == integ82[0]
DEV = ("REG3_KEYHINGE", "REG3_SLOWGATE", "REG3_NOWIN")
P4 = mcnemar1(4, 0)                                   # explore_g_devices.py:35 "p(4 vs 0) = 0.0625"
assert abs(P4 - 0.0625) < 1e-12


def keyf(rr):                                         # explore_g_devices.decide / report: unbound runs whose outcome is KEY
    return sum(1 for r in rr.values() if not bound(r) and cls(r) == "KEY")


def write_ok(r):                                      # explore_g_devices.write_ok
    b_ = r["end"].get("beta")
    return bool(b_ and b_["CTX"] > max(b_["KEY"], b_["VAL"]))


def nwrite(rr):
    return sum(write_ok(r) for r in rr.values())


B82, R82 = X["BASELINE"], X["REG3"]
RD82 = {}
for d in ("REG3",) + DEV:
    dsc = paired(X[d], B82, rb)
    o_ = {"discovered": (dsc[2] < 0.05, dsc)}
    if d != "REG3":
        hl, ht = paired(X[d], R82, rb), paired(R82, X[d], rb)
        o_["helps"] = (hl[2] <= P4, hl)
        o_["hurts"] = (ht[2] <= P4, ht)
        o_["working write"] = (keyf(X[d]) == 0 and nwrite(X[d]) >= 15, (keyf(X[d]), nwrite(X[d]), len(X[d])))
    RD82[d] = o_
_score = sorted((-cnt(X[d], rb), keyf(X[d]), i, d) for i, d in enumerate(DEV))
best_re = dict(device=_score[0][3], routed_bound=-_score[0][0], key_failures=_score[0][1],
               table={d: [-a_, k_] for a_, k_, _, d in _score})
best5 = json.loads(git("show", f"{G5}:{GD}s82_best.json"))
best_ok = all(best5[k] == best_re[k] for k in ("device", "routed_bound", "key_failures", "table"))
KD, BD82, RDD82 = X["REG3_KEYHINGE_D"], X["BASELINE_D"], X["REG3_D"]
ph2 = {ref: paired(KD, X[ref], rb) for ref in ("BASELINE_D", "REG3_D")}
val82 = dict(oracle=(cnt(X["ORACLE_ALL"], bound), len(X["ORACLE_ALL"])), nudge=(cnt(X["REG3_NUDGE"], rb), len(X["REG3_NUDGE"])),
             oracle_d=(cnt(X["ORACLE_ALL_D"], bound), len(X["ORACLE_ALL_D"])))

# timing: readings before the first record; s82_best.json before phase 2
spec82_t = ctime(S82SPEC)
_hist = git("log", "--reverse", "--format=%h %cI", f"{S82SPEC}..{G5}", "--", GD + "s82_devices_results.json").strip().split("\n")
first_rec82 = first_d82 = None
for _ln in _hist:
    _h, _t = _ln.split()
    _ks = json.loads(git("show", f"{_h}:{GD}s82_devices_results.json")).get("runs", {})
    if _ks and first_rec82 is None:
        first_rec82 = (_h, _t, len(_ks))
    if any(k.split("|")[0].endswith("_D") for k in _ks):
        first_d82 = (_h, _t, len(_ks), sum(1 for k in _ks if k.split("|")[0].endswith("_D")))
        break
best_commits = git("log", "--format=%h %cI", G5, "--", GD + "s82_best.json").strip().split("\n")
best_c = best_commits[-1].split()
_st_b = json.loads(git("show", f"{best_c[0]}:{GD}s82_devices_results.json"))["runs"]
n_at_best, d_at_best = len(_st_b), sum(1 for k in _st_b if k.split("|")[0].endswith("_D"))
best_unchanged = json.loads(git("show", f"{best_c[0]}:{GD}s82_best.json")) == best5
log82 = git("show", f"{G5}:{GD}s82.log")
i_best, i_q32 = log82.find("S82 best device (phase 2)"), log82.find("32 runs queued")
code82_hist = git("log", "--format=%h", G5, "--", "explore_g_devices.py", "explore_g_devices_child.py").split()
parent82_hist = git("log", "-1", "--format=%h %cI", G5, "--", "explore_g_rhl_parent.py").strip()
rep5_t = ctime(G5)
specs_now = [p_ for p_ in git("ls-tree", "--name-only", "-r", DRAFT2).split("\n") if p_.startswith("specs/")]

# descriptive quantities (report 5 'What happened'; the draft's 'as a description')
KH = X["REG3_KEYHINGE"]
eta_u = {u: med([statistics.fmean(r["hinge"]["eta_at"][u]) for r in KH.values()]) for u in ("1", "300", "600", "1200", "2400")}
fired = [r["hinge"]["fired"] for r in KH.values()]


def beta_med(rr, step=None):
    if step is None:
        v = [r["end"]["beta"] for r in rr.values()]
    else:
        v = [at_step(r, step)["beta"] for r in rr.values() if at_step(r, step)]
    return {k: med([b_[k] for b_ in v]) for k in ("CTX", "KEY", "VAL")}


kb1200, kb2400, kbend = beta_med(KH, 1200), beta_med(KH, 2400), beta_med(KH)
ctx_end_rng = [r["end"]["beta"]["CTX"] for r in KH.values()]
k1200_all = [at_step(r, 1200)["beta"] for r in KH.values()]
k1200_rng = [b_[k] for b_ in k1200_all for k in ("CTX", "KEY", "VAL")]
trc = Counter(r["transition"] for r in KH.values())
eta_after = []
for s, r in sorted(KH.items()):
    v = [st_["eta_key_by_key"] for st_ in r["stats"] if "eta_key_by_key" in st_ and isinstance(st_["step"], int)
         and st_["step"] >= r["transition"]] + [r["end"]["eta_key_by_key"]]
    eta_after.append(max(v))
stop82 = Counter(r.get("stopped_at") for r in KH.values())
gap82 = Counter(r.get("stopped_at") - r["transition"] for r in KH.values())
SG, NW = X["REG3_SLOWGATE"], X["REG3_NOWIN"]
sgb, nwb, r3b = beta_med(SG), beta_med(NW), beta_med(R82)


def first_split(r):
    for st_ in r["stats"]:
        if isinstance(st_["step"], int) and st_.get("eta_key_by_key", -1) >= 0.5:
            return st_["step"]
    return "end" if r["end"]["eta_key_by_key"] >= 0.5 else None


fs = {a: {s: first_split(r) for s, r in X[a].items()} for a in ("REG3", "REG3_SLOWGATE", "BASELINE")}
e1200 = {a: [at_step(r, 1200).get("eta_key_by_key") for r in X[a].values()] for a in ("REG3", "REG3_SLOWGATE")}
sg_later = sum(1 for s in X["REG3_SLOWGATE"] if fs["REG3"][s] == 1200 and fs["REG3_SLOWGATE"][s] not in (1200,))
sg_split1200 = sum(1 for v in e1200["REG3_SLOWGATE"] if v is not None and v >= 0.5)
r3_split1200 = sum(1 for v in e1200["REG3"] if v is not None and v >= 0.5)
sg_ctx1200 = beta_med(SG, 1200)
r3_ctx1200 = beta_med(R82, 1200)
nw_rb = sorted(s for s, r in NW.items() if rb(r))
r3_rb = sorted(s for s, r in R82.items() if rb(r))
s52_batch = "S52 two_stream_keysplits" in git("show", "origin/claude/outside-ideas:explore_out/README.md") and \
    "batch 18" in git("show", "origin/claude/outside-ideas:explore_out/README.md")
dev_doc = src(S82SPEC, "explore_g_devices.py")
rep5 = src(G5, GD + "report_5.md")
other_use = []                                        # seeds 2100-2119 in any other keyed store on the fetched branches
for b_ in ("origin/claude/explore-F", "origin/claude/explore-H", "origin/claude/explore-I", "origin/claude/outside-ideas",
           MAIN, "origin/claude/replicate-R"):
    try:
        hit = git("grep", "-l", "-E", r'"[A-Za-z0-9_]+\|21(0[0-9]|1[0-9])\|', b_, "--", "*.json")
    except subprocess.CalledProcessError:
        hit = ""
    other_use += [h for h in hit.split("\n") if h]
g_other = []
for p_ in git("ls-tree", "--name-only", G5, GD).split("\n"):
    if p_.endswith("_results.json") and "s82" not in p_:
        if any(re.search(r"\|21(0\d|1\d)(\||$)", k) for k in json.loads(git("show", f"{G5}:{p_}")).get("runs", {})):
            g_other.append(p_)

# ═════════════════════════════════════════════════════════════════════════════
# SECOND DRAFT: rows for new or changed sentences about Session G
# ═════════════════════════════════════════════════════════════════════════════
ks16 = (kc["Hebbian REG3"][2], kc["Hebbian REG3"][3], kc["delta REG3_D"][2], kc["delta REG3_D"][3])
# --- abstract (tex2:36)
row2(r"fourteen of the sixteen failures on each memory being key splits that form first", "abstract: label-free failures",
     "14 of 16 per memory; key splits form first", f"Hebbian {ks16[1]} of {ks16[0]}, delta {ks16[3]} of {ks16[2]} (v1 KEY among "
     f"the non-ROUTED-BOUND REG3 / REG3_D runs); split before any CTX write in {first}/{len(failK)} key-split failures",
     "MATCH" if ks16 == (16, 14, 16, 14) else "DISAGREE",
     "inside the abstract's 'Exploratory screens, which are not claims'; 'form first' is a description (eta^2 grid 1200/2400/3600/end)")
row2(r"is found $20/20$ when a hinge on the gate's key-dependence holds the key split off for the first 2400 updates",
     "abstract: S82 REG3_KEYHINGE", "20/20; hinge on updates 1-2400", f"REG3_KEYHINGE ROUTED-BOUND {cnt(KH, rb)}/{len(KH)} on 2100-2119; "
     "hinge weight 1.0 for updates 1-2400 (LAM 1.0, TAU 0.2; explore_g_devices.py:16-18 at 83699a2)",
     "MATCH" if cnt(KH, rb) == 20 else "DISAGREE", "Hebbian count; 'discovered' and 'helps' apply (see S82 readings)")
row2(r"a training-time prior that reads the task's key positions and identities as the split does",
     "abstract: what the hinge reads", "key positions and identities, as the split does",
     "explore_g_devices.py:17-18: 'It reads key positions and key identities (token types), as the key split reads key positions; "
     "it reads no stream label'; the channel split (KEYMASS) reads key positions only", "PARTIAL",
     "'as the split does' attaches 'identities' to the split as well; tex2:265 says it right ('as the split reads key positions'). "
     "The abstract does not say that the hinge reads no stream label (tex2:267 does: 'a solution without stream labels')")
row2(r"as the register's are", "abstract: the register's key splits are prevented by the key term", "prevented",
     f"REG3_KEYHINGE KEY failures {keyf(KH)}, ROUTED-BOUND {cnt(KH, rb)}/20; reading 'a working write without the device's cost' "
     f"{'applies' if RD82['REG3_KEYHINGE']['working write'][0] else 'does not apply'}", "MATCH", "the S85 half of the sentence is session H's")
# --- tex2:41, 59, 114, 228
row2([r"G's S55--S82", r"E's batch~19 (S66--S67) and F's S81 are running"], "S82 status", "complete (not in the running list)",
     f"report 5 at {G5} ({rep5_t}); {integ82[0]} records, {integ82[1]} ok, {integ82[2]} retried; phase 1 "
     f"{sum(len(v) for a, v in X.items() if not a.endswith('_D'))}/104, phase 2 {sum(len(v) for a, v in X.items() if a.endswith('_D'))}/32; "
     f"{G5} is the head of origin/claude/explore-G: {short(GBR) == G5}", "MATCH" if short(GBR) == G5 and integ82[0] == 136 else "PARTIAL")
row2(r"$20/20$ with the key hinge", "index row: S82", "20/20 with the key hinge", f"REG3_KEYHINGE ROUTED-BOUND {cnt(KH, rb)}/20",
     "MATCH" if cnt(KH, rb) == 20 else "DISAGREE")
row2([r"the readings of session~G's S72, S73 and S82 (Section~\ref{sec:far}) are on \textbf{ROUTED-BOUND}",
      r"while S71's and S56's are on bound and on G's own bound routed"], "outcome of G's readings", "S72/S73/S82 ROUTED-BOUND; "
     "S71 BOUND; S56 BOUND and BOUND ROUTED",
     f"S82: 'READINGS (ROUTED-BOUND, paired on 2100-2119' (explore_g_devices.py:{lineno(S82SPEC, 'explore_g_devices.py', 'READINGS (ROUTED-BOUND')}); "
     f"S72/S73 ROUTED-BOUND; S71 BOUND (explore_g_layers.py:{lineno(G, 'explore_g_layers.py', 'READINGS (BOUND')}); S56 R1 on BOUND ROUTED, "
     "R2-R5 on BOUND (explore_g_register.py READINGS)", "MATCH",
     "S57 (same outcomes as S56) and S55 (probe medians) are not named; nothing in the sentence is wrong")
row2(r"This is the result of session~G's S82 at the other place the key split forms", "S82 cited from session H's S85 paragraph",
     "the key term, reading key positions and identities during the slow phase, prevents the split",
     f"REG3_KEYHINGE: KEY failures {keyf(KH)}, ROUTED-BOUND 20/20, hinge on updates 1-2400 (the SLOW phase is 1-2400)", "MATCH")
# --- tex2:259, 261
row2(r"($0.53$--$0.56$ over the three sites)", "S55 ORACLE_P8 probe medians, L2 / L3 / final, K and V", "0.53-0.56",
     f"{rng(m55o.values())}", "MATCH" if rng(m55o.values()) == "0.53-0.56" else "DISAGREE")
latch_ok = sorted(s for s, r in A["REG_NUDGE"].items() if r["end"]["beta"]["CTX"] >= 0.895 and r["end"]["beta"]["KEY"] <= 0.1
                  and r["end"]["beta"]["VAL"] <= 0.1)
nb_m = (f2(med([b_["CTX"] for b_ in nb])), f2(med([b_["KEY"] for b_ in nb])), f2(med([b_["VAL"] for b_ in nb])))
row2(r"its register writes a clean latch in eight of the ten runs (medians: $\beta^R$ $0.96$ at the context token against $0.02$ and $0.05$ at keys and values)",
     "S56 REG_NUDGE clean latch; median end beta^R", "8/10; medians 0.96 / 0.02 / 0.05",
     f"clean latch (CTX >= 0.9, KEY and VAL <= 0.1) {len(latch_ok)}/10 {latch_ok}; medians {' / '.join(nb_m)}",
     "MATCH" if len(latch_ok) == 8 and nb_m == ("0.96", "0.02", "0.05") else "DISAGREE")
rt_ok = sorted(s for s, v in mgr.items() if min(v[:2]) >= 0.995 and all(A["REG_NUDGE"][s]["end"]["layers"][f"L{l}"]["one_to_one_val"] for l in (2, 3)))
row2(r"its gates route one-to-one with margin $1.00$ at layers 2 and 3 in the same eight", "S56 REG_NUDGE routing at L2, L3",
     "one-to-one, margin 1.00, in the same eight runs", f"one-to-one and margin >= 0.995 at L2 and L3 in {len(rt_ok)}/10 {rt_ok}; "
     f"the same set as the clean latch: {rt_ok == latch_ok}", "MATCH" if rt_ok == latch_ok else "DISAGREE")
racc = {s: r["acc"] for s, r in sorted(A["REG_NUDGE"].items())}
out_rng = {s: a_ for s, a_ in racc.items() if not (0.315 <= a_ < 0.445)}
row2(r"its accuracy plateaus at $0.32$--$0.44$", "S56 REG_NUDGE end accuracy (10 runs)", "0.32-0.44",
     f"run acc {rng(racc.values())} ({', '.join(f'{s} {a_:.2f}' for s, a_ in racc.items())}); stream acc "
     f"{rng([x for v in sa.values() for x in v])}; outside 0.32-0.44 (2 dp): {', '.join(f'{s} {a_:.2f}' for s, a_ in out_rng.items())}",
     "MATCH" if not out_rng else "DISAGREE",
     "0.32-0.44 is report_2.md:86's wording; the records give 0.20-0.44: seed 378 (the weakest latch, beta^R CTX 0.77) plateaus "
     "at 0.20 (streams 0.19 / 0.21) and 379 at 0.31. The first draft's 'every run at 0.32' (B7) is gone, but the range that "
     "replaced it is still not the records'")
d2 = {s: abs(round(v[0], 2) - round(v[1], 2)) for s, v in sa.items()}
d_ex = {s: abs(v[0] - v[1]) for s, v in sa.items()}
row2(r"with the two streams within $0.03$ of each other in seven of ten", "S56 REG_NUDGE |stream 0 - stream 1|", "7/10 within 0.03",
     f"on 2-dp stream accuracies {sum(1 for v in d2.values() if v <= 0.03 + 1e-9)}/10; on unrounded {sum(1 for v in d_ex.values() if v <= 0.03)}/10 "
     f"(370 differs by {d_ex[370]:.3f})", "MATCH" if sum(1 for v in d2.values() if v <= 0.03 + 1e-9) == 7 else "DISAGREE",
     "7/10 holds on the 2-dp values; 6/10 on the unrounded ones")
row2(r"the reading ``architecture insufficient'' applied", "S56 R1 (Hebbian)", "'architecture insufficient' applied",
     f"REG_NUDGE BOUND ROUTED {cnt(A['REG_NUDGE'], br56)}/10 < 6/10: 'architecture insufficient'",
     "MATCH" if cnt(A["REG_NUDGE"], br56) < 6 else "DISAGREE", "the delta R1 outcome ('neither', REG_NUDGE_D 7/10) is still not named")
_r2key = sum(1 for r in rd.values() if not bound(r) and cls(r) == "KEY")
row2(r"which met the screen's own rule for ``discovered'' on delta (6 against 2 with no unbound key failure; 4 vs 0 by McNemar, which was not its rule)",
     "S56 R2 on delta (REG_D vs BASELINE_D)", "'discovered' by R2's rule: 6 vs 2, no unbound KEY failure",
     f"REG_D BOUND {cnt(rd, bound)} >= BASELINE_D {cnt(bd, bound)} + 3; REG_D unbound KEY failures {_r2key} (rule: <= 2) -> "
     f"{'discovered' if cnt(rd, bound) >= cnt(bd, bound) + 3 and _r2key <= 2 else 'not discovered'}",
     "MATCH" if cnt(rd, bound) >= cnt(bd, bound) + 3 and _r2key == 0 else "DISAGREE")
row2(r"binds $10/10$ (median 1200, range 1200--4800)", "S71 ORACLE_ALL BOUND, transitions", "10/10, median 1200, 1200-4800",
     f"{cnt(OA, bound)}/10, median {med(trs(OA)):.0f}, {trs(OA)[0]}-{trs(OA)[-1]}",
     "MATCH" if (cnt(OA, bound), med(trs(OA)), trs(OA)[0], trs(OA)[-1]) == (10, 1200, 1200, 4800) else "DISAGREE")
lo = [a_ for a_ in acc23.values() if a_ < 0.43]
hi = [a_ for a_ in acc23.values() if a_ >= 0.43]
row2([r"the runs' accuracies $0.31$--$0.54$ in two clusters near $0.32$ and $0.53$",
      r"the per-cell medians over runs $0.46$--$0.51$ in every (stream, block, pair) cell"],
     "S71 ORACLE_23 per-run accuracy; per-cell medians", "0.31-0.54, clusters ~0.32 and ~0.53; cell medians 0.46-0.51",
     f"run acc {rng(acc23.values())}: {len(lo)} runs {rng(lo)} (median {med(lo):.2f}), {len(hi)} runs {rng(hi)} (median {med(hi):.2f}); "
     f"per-cell medians {rng(cellmed)}", "MATCH" if rng(acc23.values()) == "0.31-0.54" and rng(cellmed) == "0.46-0.51" else "DISAGREE")
row2(r"the interpretation crosses two tasks (S56's fixed eight-key layout; S71's randomised one) and two gate inputs",
     "the layer-1 interpretation: marking and scope", "ours; crosses two tasks and two gate inputs",
     "S56: HeaderTask(8) fixed layout, gate input v_t; S71/S72: RandHeaderTask-lite, REG3 input window (rows above)", "MATCH",
     "the whole interpretation, including 'that choice is what sank its Hebbian register', is now inside the marking")
# --- tex2:263
row2(r"so the pre-fixed reading ``discovered (register)'' applies on neither", "S72 / S73 label-free readings", "'discovered (register)' on neither",
     f"S72: 'discovered' / 'not discovered' (explore_g_register3.py:{lineno(G, 'explore_g_register3.py', 'REG3 beats BASELINE')}) -> "
     f"'not discovered' {paired(R3, BH, rb)[:2]}; S73: 'discovered (register)' -> does not apply {paired(R3D, BD, rb)[:2]}", "PARTIAL",
     "substance right on both; the label is now S73's for both screens, where S72's reading is named 'discovered' (outcome "
     "'not discovered'). The first draft's error was the reverse")
row2(r"Two other readings of S73 did not apply either, ``the delta stack carries the cue'' (the baseline $7/20$) and ``the gate is not needed'' (one channel $7/10$)",
     "S73 descriptive readings", "both did not apply; 7/20, 7/10",
     f"'the delta stack carries the cue': {sum(1 for r in BD.values() if min(pk(r)) >= 0.9)}/20 (< 10/20); 'the gate is not needed': "
     f"SINGLE_D {cnt(SGL, bound)}/10 (< 8/10)", "MATCH")
row2(r"(14 of the 16 runs that did not route on each memory; on delta two of the 14 bound)", "REG3 / REG3_D key splits among the "
     "non-ROUTED-BOUND runs", "14 of 16 each; 2 of delta's 14 bound", f"Hebbian {ks16[1]}/{ks16[0]}, delta {ks16[3]}/{ks16[2]}; delta "
     f"bound key-split runs {kc['delta REG3_D'][4]}", "MATCH" if ks16 == (16, 14, 16, 14) and len(kc['delta REG3_D'][4]) == 2 else "DISAGREE")
sep = sorted(((b_["CTX"] - max(b_["KEY"], b_["VAL"])), s) for s, b_ in ksr)
row2(r"with $\beta^R$ at the context token not separated above keys and values", "end beta^R in the 28 key-split runs",
     "CTX not separated above K/V", f"CTX <= max(KEY, VAL) + 0.05 in {ctxlow}/{len(ksr)}; the exception: REG3|712 CTX "
     f"{R3[712]['end']['beta']['CTX']:.2f} vs KEY {R3[712]['end']['beta']['KEY']:.2f}, VAL {R3[712]['end']['beta']['VAL']:.2f} "
     f"(+{sep[-1][0]:.2f})", "PARTIAL" if ctxlow < len(ksr) else "MATCH",
     "replaces the first draft's 'flat across roles' (resolved); holds in 27 of 28; in REG3|712 (unbound, KEY) CTX sits 0.10 "
     "above K/V, about the routed runs' lower edge (0.32 vs 0.16-0.24)")
row2(r"The unbound Hebbian runs sit near chance between the streams (per-cell medians $0.46$--$0.55$; flat across cells by the screen's rule in about half of them)",
     "S72 BASELINE / REG3 unbound runs", "per-cell medians 0.46-0.55; flat in about half",
     "; ".join(f"{k}: {v[0]} unbound, flat {v[2]}/{v[0]}, cell medians {rng(v[3])}" for k, v in plate.items()),
     "MATCH" if rng(plate['BASELINE'][3] + plate['REG3'][3]) == "0.46-0.55" else "PARTIAL",
     "within the paragraph's 'Descriptively' passage; the S71 comparison is gone")
row2(r"goes to $z = 1$ at every role by update 1200 in eight of ten runs", "S73 LATCH3_D z at 1200", "8/10",
     f"z >= 0.99 at CTX, KEY, VAL at 1200 in {len(all1)}/10", "MATCH" if len(all1) == 8 else "DISAGREE")
row2(r"key-splits in nine of its nine unbound runs", "S73 LATCH3_D key splits", "9 of 9 unbound",
     f"unbound {sum(1 for r in L3D.values() if not bound(r))}, KEY among them {lk_unb}; all runs {lk_all}/10",
     "MATCH" if lk_unb == 9 and sum(1 for r in L3D.values() if not bound(r)) == 9 else "DISAGREE")
row2(r"(on the fixed layout, S57, it had gone to $z = 1$ in every run and key-split $3/10$)", "S57 LATCH (410-419)",
     "z = 1 in every run by 1200; key-split 3/10", f"z >= 0.99 at every role at 1200 in {s57all1}/10; classes {dict(s57cls)}",
     "MATCH" if s57all1 == 10 and s57cls.get("KEY", 0) == 3 else "DISAGREE")
row2(r"among the 31 non-routing delta solutions it finds it at key positions in 18 and at both key and value positions in 13",
     "S73 in-run probe L3 in the non-routing bound delta runs", "31; K 18; K and V 13", f"{len(nk)}; K >= 0.9 {konly}; both {both}",
     "MATCH" if (len(nk), konly, both) == (31, 18, 13) else "DISAGREE")
row2(r"the single channel's bound runs at $0.93$--$1.00$ at keys", "SINGLE_D bound runs, probe L3 at K", "0.93-1.00 at keys",
     f"K {rng([k for k, _ in sk])} (V {rng([v for _, v in sk])})", "MATCH" if rng([k for k, _ in sk]) == "0.93-1.00" else "DISAGREE")
# --- tex2:265, S82
row2(r"session~G screened three devices against the key split on 20 fresh seeds (2100--2119), Hebbian memory",
     "S82 phase 1: devices, seeds, memory", "three devices; 20 fresh seeds 2100-2119; Hebbian",
     f"{', '.join(f'{d} {len(X[d])}' for d in DEV)} on {min(KH)}-{max(KH)}; seeds 2100-2119 in no other keyed store on G "
     f"({g_other or 'none'}) or on F, H, I, E, R, the main line ({other_use or 'none'})", "MATCH")
row2(r"with the baseline, REG3 and a nudged pair for validity ($2/2$, with the perfect gate $2/2$)", "S82 references and validity arms",
     "BASELINE, REG3; REG3_NUDGE 2/2; ORACLE_ALL 2/2", f"BASELINE {len(B82)}, REG3 {len(R82)}; REG3_NUDGE ROUTED-BOUND "
     f"{val82['nudge'][0]}/{val82['nudge'][1]}; ORACLE_ALL BOUND {val82['oracle'][0]}/{val82['oracle'][1]} (transitions "
     f"{trs(X['ORACLE_ALL'])}); ORACLE_ALL_D {val82['oracle_d'][0]}/{val82['oracle_d'][1]}; VALIDITY rule (oracle rerun >= 1/2): VALID",
     "MATCH" if val82["nudge"] == (2, 2) and val82["oracle"] == (2, 2) else "DISAGREE")
row2(r"The hinge's key term of batch~18, a penalty on the shared gate's dependence on key identity at key positions during updates 1--2400 (REG3\_KEYHINGE)",
     "S82 REG3_KEYHINGE: what the device is", "batch 18's key term; key identity at key positions; updates 1-2400",
     "explore_g_devices.py:16-18 (S52's key-term hinge, LAM 1.0, TAU 0.2, updates 1-2400, mean over the three layers' gates); "
     f"S52 is batch 18's two_stream_keysplits (outside-ideas README): {s52_batch}; the child's term: LAM x mean_l relu(eta2_key(g^(l)) - TAU)",
     "MATCH" if s52_batch else "PARTIAL", "the penalty is a hinge (only eta^2 above 0.2 is penalised), averaged over the three layers' gates")
kk = RD82["REG3_KEYHINGE"]
row2(r"binds routed $\mathbf{20/20}$ against the baseline's $0/20$ and REG3's $2/20$ on these seeds (20 vs 0 and 18 vs 0; the readings ``discovered'' and ``helps'' both apply",
     "S82 REG3_KEYHINGE vs BASELINE and REG3 (ROUTED-BOUND)", "20/20 vs 0/20 and 2/20; 20 vs 0, 18 vs 0; discovered and helps apply",
     f"ROUTED-BOUND {cnt(KH, rb)}/20, BASELINE {cnt(B82, rb)}/20, REG3 {cnt(R82, rb)}/20; discovered {kk['discovered'][1][:2]} p = "
     f"{kk['discovered'][1][2]:.2g} -> {'applies' if kk['discovered'][0] else 'no'}; helps {kk['helps'][1][:2]} p = {kk['helps'][1][2]:.2g} "
     f"(<= {P4}) -> {'applies' if kk['helps'][0] else 'no'}",
     "MATCH" if (cnt(KH, rb), cnt(B82, rb), cnt(R82, rb), kk['discovered'][1][:2], kk['helps'][1][:2]) == (20, 0, 2, (20, 0), (18, 0))
     and kk['discovered'][0] and kk['helps'][0] else "DISAGREE")
row2(r"the arm has no key failures and $\beta^R$ at the cue above keys and values in every run", "S82 REG3_KEYHINGE: the fourth reading's content",
     "no key failures; CTX > K, V in 20/20", f"KEY failures {kk['working write'][1][0]}; beta^R CTX > KEY, VAL in {kk['working write'][1][1]}/20 -> "
     f"'a working write without the device's cost' {'applies' if kk['working write'][0] else 'does not apply'}",
     "MATCH" if kk["working write"][1][:2] == (0, 20) else "DISAGREE", "the reading's content is stated; its name is not")
row2(r"on delta memory the same arm binds $10/10$ against $0/10$ and $2/10$", "S82 phase 2 (delta, 2100-2109)", "10/10 vs 0/10 and 2/10",
     f"ROUTED-BOUND: REG3_KEYHINGE_D {cnt(KD, rb)}/10, BASELINE_D {cnt(BD82, rb)}/10, REG3_D {cnt(RDD82, rb)}/10 (pairs "
     f"{ph2['BASELINE_D'][:2]} p = {ph2['BASELINE_D'][2]:.2g}; {ph2['REG3_D'][:2]} p = {ph2['REG3_D'][2]:.2g}); BOUND: {cnt(KD, bound)}, "
     f"{cnt(BD82, bound)}, {cnt(RDD82, bound)}", "PARTIAL",
     "the counts are ROUTED-BOUND (as the table caption says), but the sentence says 'binds' and on BOUND the references are 2/10 and "
     "4/10. The docstring fixes phase 2 as descriptive (explore_g_devices.py:40; report_5.md:68-69); the sentence does not say so")
sgr = RD82["REG3_SLOWGATE"]
row2(r"A gate held slow for the first 1200 updates binds $0/20$, worse than nothing", "S82 REG3_SLOWGATE", "0/20; worse than nothing",
     f"ROUTED-BOUND {cnt(SG, rb)}/20 (BOUND {cnt(SG, bound)}); vs BASELINE {sgr['discovered'][1][:2]}; vs REG3 (no device) "
     f"{paired(SG, R82, rb)[:2]}; 'the device hurts' (REG3 > D, p <= 0.0625): {sgr['hurts'][1][:2]} p = {sgr['hurts'][1][2]:.2f} -> "
     f"{'applies' if sgr['hurts'][0] else 'does not apply'}", "PARTIAL",
     "0/20 is right. 'Worse than nothing' is not: against the baseline it is 0 vs 0, and against the register without a device "
     "(2/20) it is 0 vs 2, where the pre-fixed 'the device hurts' does not apply (p = 0.25). An unmarked description that "
     "contradicts the screen's own reading (the S61 pattern)")
row2(r"because it delays the register's write along with the split", "S82 REG3_SLOWGATE: the stated cause", "the slow gate delays the write and the split",
     f"eta^2 by key >= 0.5 at 1200: SLOWGATE {sg_split1200}/20 vs REG3 {r3_split1200}/20 (median {med([v for v in e1200['REG3_SLOWGATE'] if v is not None]):.2f} vs "
     f"{med([v for v in e1200['REG3'] if v is not None]):.2f}); unbound KEY at the end {keyf(SG)}/20; beta^R CTX > K, V at the end "
     f"{nwrite(SG)}/20; median end beta^R CTX/KEY/VAL {sgb['CTX']:.2f}/{sgb['KEY']:.2f}/{sgb['VAL']:.2f} (REG3 {r3b['CTX']:.2f}/{r3b['KEY']:.2f}/{r3b['VAL']:.2f})",
     "PARTIAL", "the records agree that the split is delayed (fewer splits at 1200) and still forms (18 KEY), and that the write "
     "never separates; that one causes the failure is an interpretation (report_5.md 'What happened (exploratory)'), stated here "
     "without a marking")
nwr = RD82["REG3_NOWIN"]
row2(r"a gate without the raw window input binds $5/20$, ``discovered'' exactly at its threshold (5 vs 0)", "S82 REG3_NOWIN",
     "5/20; discovered at 5 vs 0", f"ROUTED-BOUND {cnt(NW, rb)}/20 {nw_rb} (REG3's {r3_rb} a subset: {set(r3_rb) <= set(nw_rb)}); "
     f"discovered {nwr['discovered'][1][:2]} p = {nwr['discovered'][1][2]:.4f} -> {'applies' if nwr['discovered'][0] else 'no'}; "
     f"helps {nwr['helps'][1][:2]} p = {nwr['helps'][1][2]:.3f} -> {'applies' if nwr['helps'][0] else 'does not apply'}",
     "MATCH" if cnt(NW, rb) == 5 and nwr['discovered'][1][:2] == (5, 0) and nwr['discovered'][0] else "DISAGREE",
     "5 vs 0 is the smallest discordance that clears p < 0.05 (0.031); 'helps' did not apply (3 vs 0), which the draft does not say")
row2(r"with a register that writes at nearly every token", "S82 REG3_NOWIN end beta^R", "writes at nearly every token",
     f"median end beta^R CTX/KEY/VAL {nwb['CTX']:.2f}/{nwb['KEY']:.2f}/{nwb['VAL']:.2f}; beta^R CTX > K, V in {nwrite(NW)}/20",
     "MATCH" if min(nwb.values()) >= 0.8 else "PARTIAL", "a description (report_5.md:54-56), before the sentence that marks the next ones")
row2(r"($\eta^2$ by key $0.89$ at update 1)", "S82 REG3_KEYHINGE eta^2 by key at update 1", "0.89",
     f"median over runs of the mean over layers: {eta_u['1']:.3f}", "MATCH" if f"{eta_u['1']:.2f}" == "0.89" else "DISAGREE",
     "marked 'as a description'")
row2(r"the hinge holds that at $0.05$ from update 300", "S82 REG3_KEYHINGE eta^2 by key during the hinge", "0.05 from update 300",
     ", ".join(f"{u} {v:.3f}" for u, v in eta_u.items() if u != "1") + f"; firings median {med(fired):.0f} ({min(fired)}-{max(fired)})",
     "MATCH" if max(eta_u[u] for u in ("300", "600", "1200", "2400")) < 0.06 else "PARTIAL",
     "at or below 0.05 (0.051, 0.056, 0.035, 0.000): 'at about 0.05 and falling to 0 by 2400'")
row2(r"($0.12$ at every role at 1200; $0.48$ against $0.20$ at the end)", "S82 REG3_KEYHINGE beta^R by role", "0.12 at 1200; 0.48 vs 0.20 at the end",
     f"median at 1200 CTX/KEY/VAL {kb1200['CTX']:.2f}/{kb1200['KEY']:.2f}/{kb1200['VAL']:.2f} (per run {rng(k1200_rng)}); at 2400 "
     f"{kb2400['CTX']:.2f}/{kb2400['KEY']:.2f}/{kb2400['VAL']:.2f}; at the end {kbend['CTX']:.2f}/{kbend['KEY']:.2f}/{kbend['VAL']:.2f} "
     f"(CTX per run {rng(ctx_end_rng)})", "MATCH" if (f2(kb1200['CTX']), f2(kbend['CTX'])) == ("0.12", "0.48") and
     max(f2(kbend['KEY']), f2(kbend['VAL'])) in ("0.20", "0.21") else "PARTIAL",
     "medians over runs; 'the end' is each run's early stop (see the next rows), not update 24000")
row2(r"18 of the 20 bind at 3600, the first evaluation after the hinge stops", "S82 REG3_KEYHINGE transitions", "18 at 3600",
     f"{dict(sorted(trc.items()))}; evaluations every 1200, hinge weight 1 to update 2400, 0 from 2401", "MATCH" if trc.get(3600) == 18 else "DISAGREE")
row2(r"and the key split does not return once the gate routes", "S82 REG3_KEYHINGE eta^2 by key from the transition to the end",
     "does not return", f"max eta^2 by key at the evaluations >= the transition and at the end: {rng(eta_after, 3)}; the runs stop "
     f"early (3 evaluations >= 0.95): stopped_at {dict(sorted(stop82.items()))}, i.e. transition + {sorted(gap82)}",
     "PARTIAL" if max(eta_after) < 0.5 else "DISAGREE",
     "true over what was recorded; each run stops 2400 updates after its transition (explore_g_devices_child.py:9), so 'does not "
     "return' is observed over 2400 updates, not to 24000")
row2(r"The hinge reads the task's key positions and key identities, as the split reads key positions", "what the hinge reads",
     "key positions and key identities; not an architecture", "explore_g_devices.py:17-18 ('reads key positions and key identities "
     "(token types) ... it reads no stream label'); report_5.md:64-66 (training-time prior, not an architecture)", "MATCH",
     "the docstring's 'reads no stream label' is said at tex2:267 ('without stream labels')")
# --- tex2:267
row2(r"On a screen of twenty seeds, the distant-cue task has a solution without stream labels", "label-free solution", "on a screen of 20 seeds",
     f"REG3_KEYHINGE {cnt(KH, rb)}/20, 'discovered' applies; the device reads key identities, no stream label", "MATCH",
     "replaces the first draft's unmarked 'exists'; marked as a screen")
row2(r"Its capacity was never the limit (the nudged register, $20/20$)", "capacity", "the nudged register 20/20",
     f"S72 R1 'the register carries the cue at all three layers' applies ({cnt(R3N, rb)}/20)", "MATCH",
     "a paraphrase of S72 R1's reading")
row2(r"its discovery is $20/20$ with the hinge and $2$--$4/20$ without", "discovery with and without the hinge", "20/20; 2-4/20",
     f"REG3_KEYHINGE {cnt(KH, rb)}/20; REG3 {cnt(R82, rb)}/20 (S82), {cnt(R3, rb)}/20 (S72), REG3_D {cnt(R3D, rb)}/20 (S73)", "MATCH")
row2(r"Session~G proposed the pre-registered test, and it is the register test queued behind \texttt{test\_delta\_recipe}",
     "the register test", "G's proposal; queued", f"report_5.md:{lineno(G5, GD + 'report_5.md', 'A pre-registered main-line test of REG3_KEYHINGE')}-103: "
     "BASELINE, REG3, REG3_KEYHINGE, LOCAL3_KEYHINGE (not run in S82), 20 seeds per machine, rate claim >= 18/20 on each: as the "
     f"draft says; specs/ at {DRAFT2}: {sorted(specs_now)}", "NOT RECORD-CHECKABLE",
     "the proposal's content matches report 5; 'queued' has no committed specification to check")
row2(r"bound $0/10$ on the fixed Hebbian layout (S56, every run a key split, with nothing in the stack to read)", "S56 RESGATE_ONLY",
     "0/10 BOUND; every run a key split", f"BOUND {cnt(r56, bound)}/10; KEY {sum(1 for r in r56.values() if cls(r) == 'KEY')}/10; S55 'absent' "
     "(Hebbian, P = 8)", "MATCH" if cnt(r56, bound) == 0 and sum(1 for r in r56.values() if cls(r) == 'KEY') == 10 else "DISAGREE")
row2(r"routed $0/10$ on delta (S73; $5/10$ bound, eight key splits)", "S73 RESGATE_ONLY_D", "0/10 routed; 5/10 bound; 8 key splits",
     f"ROUTED-BOUND {cnt(RES, rb)}/10; BOUND {cnt(RES, bound)}/10; key splits (all runs) {resk_all}/10",
     "MATCH" if (cnt(RES, rb), cnt(RES, bound), resk_all) == (0, 5, 8) else "DISAGREE")
row2(r"no reading of either screen applied to it, and it is not pursued", "readings on the residual gate", "none applied",
     "S73 'discovered (residual gate)' (0, 0): does not apply; S56 has no reading of RESGATE_ONLY's own outcome (R4 'the register "
     "adds to the residual gate' compares REG with it: 'it does not', 0 <= 0)", "MATCH",
     "replaces the first draft's 'closed'")
# --- table 5
row2(r"the same with the key hinge for updates 1--2400 (S82, seeds 2100--2119; baseline there $0/20$, register $2/20$) & $\mathbf{20/20}$ & $\mathbf{10/10}$",
     "table: S82 REG3_KEYHINGE / _D (ROUTED-BOUND); references", "20/20, 10/10; baseline 0/20, register 2/20",
     f"{cnt(KH, rb)}/20, {cnt(KD, rb)}/10; BASELINE {cnt(B82, rb)}/20, REG3 {cnt(R82, rb)}/20",
     "MATCH" if (cnt(KH, rb), cnt(KD, rb), cnt(B82, rb), cnt(R82, rb)) == (20, 10, 0, 2) else "DISAGREE",
     "the delta column is on 2100-2109, and its references (0/10, 2/10) are not in the row")
row2(r"the same with a slow gate for 1200 updates / without the window input (S82) & $0/20$ / $5/20$ & ---",
     "table: S82 REG3_SLOWGATE / REG3_NOWIN (ROUTED-BOUND)", "0/20 / 5/20", f"{cnt(SG, rb)}/20 / {cnt(NW, rb)}/20",
     "MATCH" if (cnt(SG, rb), cnt(NW, rb)) == (0, 5) else "DISAGREE")
row2(r"Exploratory (session G, S71--S73 and S82)", "table caption: S82 rows' outcome", "ROUTED-BOUND (non-oracle rows)",
     "S82 counts above are ROUTED-BOUND", "MATCH")
# --- retrospective, referee, limits, conclusion, next, provenance
row2(r"which a hinge on the gate's key-dependence during the slow phase prevents ($20/20$ on a screen)", "discovery race: the distant key split",
     "prevented, 20/20 on a screen", f"REG3_KEYHINGE {cnt(KH, rb)}/20, KEY failures {keyf(KH)}", "MATCH",
     "'a key split that forms before the register's write at the cue has appeared' is still an unmarked description (records agree)")
_dt = (__import__("datetime").datetime.fromisoformat(ctime(DRAFT)) - __import__("datetime").datetime.fromisoformat(best_c[1])).total_seconds() / 60
row2(r"that session~G was still screening three devices whose result, $20/20$, had been recorded fifteen minutes before the commit",
     "the first draft's S82 error, as listed", "20/20 recorded 15 min before the first draft", f"s82_best.json added at {best_c[0]} "
     f"({best_c[1]}), REG3_KEYHINGE 20/20; first draft {DRAFT} at {ctime(DRAFT)}: {_dt:.1f} min", "MATCH" if 14 <= _dt <= 16 else "PARTIAL")
row2([r"the residual gate alone bound nothing on Hebbian memory and routed nothing on delta",
      r"that this design choice is what sank the Hebbian register is our interpretation, across two tasks and two gate inputs"],
     "'Done.' item", "0 bound (Hebbian), 0 routed (delta); interpretation marked", f"S56 RESGATE_ONLY BOUND {cnt(r56, bound)}/10; S73 "
     f"RESGATE_ONLY_D ROUTED-BOUND {cnt(RES, rb)}/10", "MATCH", "'nothing to read' removed; the interpretation is now marked")
row2(r"that a gate held slow at the start would let the register's write win the race against the key split (it binds $0/20$, worse than nothing, because it slows the write too)",
     "overturned prediction: the slow gate", "0/20; worse than nothing; because it slows the write",
     f"REG3_SLOWGATE {cnt(SG, rb)}/20; vs REG3 {paired(SG, R82, rb)[:2]}, 'the device hurts' does not apply (p = {sgr['hurts'][1][2]:.2f}); "
     f"vs BASELINE (0, 0); the S82 docstring's purpose: 'devices that delay or prevent the key split so the register's CTX write can emerge'",
     "PARTIAL", "the prediction is the spec's (explore_g_devices.py:3-4) and 0/20 is right; 'worse than nothing' and the cause are "
     "unmarked, as at tex2:265")
row2(r"the distant-cue task has a solution without stream labels on screens ($20/20$ with a hinge that reads key identities during the first 2400 updates) and no pre-registered test of it yet",
     "scope: distant cue", "20/20 on screens; no pre-registered test", f"REG3_KEYHINGE {cnt(KH, rb)}/20; specs/ at {DRAFT2}: {sorted(specs_now)}",
     "MATCH")
row2(r"the hinge's key term, which prevents the key split at two streams and for the register at a distance, reads key positions and key identities",
     "limits: what the key term reads (register part)", "prevents the register's key split; reads key positions and identities",
     f"REG3_KEYHINGE KEY failures {keyf(KH)}; 'a working write without the device's cost' applies; explore_g_devices.py:17-18",
     "MATCH", "the two-stream half is session H's S85")
row2(r"The architecture exists and, with the key hinge, is found $20/20$", "limits: distant cue", "20/20 with the hinge",
     f"REG3_KEYHINGE {cnt(KH, rb)}/20", "MATCH", "'exists' is now backed by S82's 'discovered' on a screen")
row2(r"without it the write at the cue is found on a fifth of seeds and a key split forms first on fourteen of the sixteen others on each memory",
     "limits: without the hinge", "a fifth; 14 of 16 key splits first", f"S72/S73: REG3 {cnt(R3, rb)}/20, REG3_D {cnt(R3D, rb)}/20, key "
     f"splits {ks16[1]}/16 and {ks16[3]}/16; S82 (fresh seeds): REG3 {cnt(R82, rb)}/20 (KEY {keyf(R82)} of {sum(1 for r in R82.values() if not bound(r))} "
     f"unbound), REG3_D {cnt(RDD82, rb)}/10", "PARTIAL",
     "right for S72/S73 (the first draft's 'the rest' now 14 of 16); on S82's fresh seeds the same arm is a tenth (2/20), which "
     "tex2:267's '2--4/20' states and this item does not; the order 'forms first' is an unmarked description")
row2(r"and is found on every seed when a hinge keeps the gate off key identity during the slow phase", "conclusion: the hinged register",
     "every seed", f"REG3_KEYHINGE {cnt(KH, rb)}/20 (Hebbian, 2100-2119), REG3_KEYHINGE_D {cnt(KD, rb)}/10 (delta, 2100-2109): one screen",
     "PARTIAL", "the counts are right; the conclusion drops 'on a screen' (the abstract, tex2:267 and tex2:434 keep it), and "
     "'every seed' is 30 seeds of one exploratory screen whose phase 2 is descriptive")
row2(r"then the hinged-register test on the randomised header layout (session~G's proposal: the hinged register against the baseline, the plain register and the hinge without the register; 20 seeds per machine; a rate claim of $\geq 18/20$)",
     "next steps: the register test", "G's proposal as listed", "report_5.md:85-103 (C1 vs BASELINE, C2 vs REG3, C3 vs LOCAL3_KEYHINGE, "
     "rate >= 18/20 per machine, 20 per machine; validity REG3_NUDGE and the perfect gate)", "MATCH")
nrun5 = nrun + (integ82[0],)
row2(r"S82 devices against the key split (report 5, \texttt{8973d48}; spec \texttt{83699a2}) & \texttt{8973d48} & 22; 139; 50; 150; 136",
     "provenance: S82", "report 5 at 8973d48; spec 83699a2; 136 records", f"report_5.md first added at "
     f"{git('log', '--diff-filter=A', '--format=%h', GBR, '--', GD + 'report_5.md').strip()}; spec {S82SPEC} adds {', '.join(spec_files)}; "
     f"S82 records {integ82[0]}; the earlier counts {nrun}", "MATCH" if integ82[0] == 136 and git('log', '--diff-filter=A', '--format=%h', GBR, '--', GD + 'report_5.md').strip() == G5 else "DISAGREE")

# ═════════════════════════════════════════════════════════════════════════════
# output
# ═════════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════
# THIRD DRAFT (a5f67a2): extra recomputations for the new wording
# ═════════════════════════════════════════════════════════════════════════════
specs3 = sorted(p_ for p_ in git("ls-tree", "--name-only", "-r", DRAFT3).split("\n") if p_.startswith("specs/"))
draft3_t = ctime(DRAFT3)
# S56 REG_NUDGE: the weakest latch is the lowest run
ctx56 = {s: r["end"]["beta"]["CTX"] for s, r in sorted(A["REG_NUDGE"].items())}
weak = min(ctx56, key=ctx56.get)
low = min(racc, key=racc.get)
n_within_2dp = sum(1 for v in d2.values() if v <= 0.03 + 1e-9)
n_within_ex = sum(1 for v in d_ex.values() if v <= 0.03)


# S82 REG3_SLOWGATE: the write at the cue at every evaluation, not only at the end
def sep_any(r, margin=0.0):
    steps = [st_ for st_ in r["stats"] if isinstance(st_["step"], int) and st_["step"] > 0 and st_.get("beta")]
    hit = [st_["step"] for st_ in steps if st_["beta"]["CTX"] > max(st_["beta"]["KEY"], st_["beta"]["VAL"]) + margin]
    eb = r["end"]["beta"]
    end_hit = eb["CTX"] > max(eb["KEY"], eb["VAL"]) + margin
    return hit, end_hit


sg_any0 = {s: sep_any(r) for s, r in sorted(SG.items())}
sg_any5 = {s: sep_any(r, 0.05) for s, r in sorted(SG.items())}
sg_n_any0 = sum(1 for h, e in sg_any0.values() if h or e)
sg_n_any5 = sum(1 for h, e in sg_any5.values() if h or e)
sg_end_sep = sorted(s for s, r in SG.items() if write_ok(r))
sg_gt1200 = sum(1 for v in e1200["REG3_SLOWGATE"] if v is not None and v > 0.5)
r3_gt1200 = sum(1 for v in e1200["REG3"] if v is not None and v > 0.5)
sg_hurts = paired(R82, SG, rb)                      # REG3 only, SLOWGATE only
sg_vs_base = paired(SG, B82, rb)


# S82: do the label-free runs that fail end in a key split, formed before any write at the cue? (as the v1 timing check)
def order82(r):
    ks = next((st_["step"] for st_ in r["stats"] if isinstance(st_["step"], int) and st_.get("eta_key_by_key", -1) >= 0.5), None)
    if ks is None and r["end"]["eta_key_by_key"] >= 0.5:
        ks = "end"
    wr = next((st_["step"] for st_ in r["stats"] if isinstance(st_["step"], int) and st_["step"] > 0 and st_.get("beta")
               and st_["beta"]["CTX"] > max(st_["beta"]["KEY"], st_["beta"]["VAL"]) + 0.05), None)
    return ks, wr


f82 = {s: r for s, r in R82.items() if not rb(r)}
f82_key = {s: r for s, r in f82.items() if cls(r) == "KEY"}
f82_ord = {s: order82(r) for s, r in sorted(f82_key.items())}
f82_first = sum(1 for ks, wr in f82_ord.values() if ks is not None and (wr is None or (ks != "end" and wr > ks)))
f82_cls = Counter(cls(r) if not bound(r) else "BOUND-" + cls(r) for r in f82.values())
s7273_fail = sum(1 for nm_ in kc for _ in range(kc[nm_][2]))
s7273_key = kc["Hebbian REG3"][3] + kc["delta REG3_D"][3]
# KEYHINGE: the observed window after the transition
gap_set = sorted(gap82)
kd_bound = cnt(KD, bound)
# S83 (session G's next screen): spec and start against the third draft's commit
s83_spec = git("log", "--reverse", "--format=%h %cI", GBR, "--", "explore_g_s83.py").strip().split("\n")[0].split()
s83_log = git("log", "--reverse", "--format=%h %cI", GBR, "--", GD + "s83.log").strip().split("\n")
s83_start = s83_log[0].split()
s83_before = [l_.split() for l_ in s83_log if l_.split()[1] <= draft3_t]
s83_results_at_draft = [p_ for p_ in git("ls-tree", "--name-only", git("rev-list", "-1", f"--before={draft3_t}", GBR).strip(), GD).split("\n")
                        if "s83" in p_]

# ═════════════════════════════════════════════════════════════════════════════
# THIRD DRAFT: V3 revisions of rows whose fragment survives in an amended sentence
# ═════════════════════════════════════════════════════════════════════════════
V3["the half-routed baseline's at $0.86$--$1.00$"] = (
    "MATCH", "the third draft adds ', also at keys' (tex3:263): K's range; V 0.79-0.99 is not claimed (second pass §3, P17: resolved)")
V3[r"with $\beta^R$ at the context token not separated above keys and values"] = (
    "MATCH", f"the third draft adds 'in 27 of the 28 such runs': CTX <= max(KEY, VAL) + 0.05 in {ctxlow}/{len(ksr)} key-split runs; "
    "the exception is REG3|712 (P10: resolved)")
V3["Each is marked where it appears"] = (
    "PARTIAL", "for session G every place the referee's second pass listed is now marked (tex3:36, :265, :316, :455 'formed in "
    "the records'; :265 'which we read as', 'descriptive by the screen's docstring'; :384 'on our reading of S71', 'on our "
    "reading it slows'; :436, :465 'on screens' / 'of a screen'). Not marked, and listed by the first pass's check_G.md "
    "('Descriptive statements: marked or not'), unchanged since the first draft: tex3:259 'where the decayed sum of the block "
    "still carries the header', 'the gate reads only the window, nothing asks the stack to keep it' and 'a gate over the "
    "residual stream therefore had nothing to read, which the routing study predicts'. So 'those are now marked' is still "
    "not true of tex3:259")
V3["is found label-free on a fifth of seeds"] = (
    "PARTIAL", f"the conclusion keeps 'a fifth' (S72/S73: {cnt(R3, rb)}/20 and {cnt(R3D, rb)}/20); on S82's fresh seeds the same arm "
    f"is {cnt(R82, rb)}/20 (Hebbian) and {cnt(RDD82, rb)}/10 (delta), which tex3:455 now states ('on 2 of 20 fresh seeds in S82') "
    "and tex3:267 ('2--4/20'); the conclusion is the one place left with S72/S73's rate alone. The layer clause is S71's reading")
V3["G's S55--S82"] = (
    "PARTIAL", f"S82 is complete (report 5 at {G5}, {integ82[0]} records). But session G's next screen, S83, was specified at "
    f"{s83_spec[0]} ({s83_spec[1]}) and started at {s83_start[0]} ({s83_start[1]}; s83.log only, no results file at the draft's "
    f"commit: {s83_results_at_draft}), both before the third draft ({DRAFT3}, {draft3_t}). tex3:41's list of what is running "
    "('E's batch 19 (S66--S67) and F's S81 are running') and the batches table's G row (tex3:508, which says 'S81 running' for F) "
    "do not mention it. An omission, not a false number; the same kind as the first draft's S82 item")
V3[r"Session~G proposed the pre-registered test, and it is the register test queued behind \texttt{test\_delta\_recipe}"] = (
    "NOT RECORD-CHECKABLE", f"the proposal's content matches report_5.md:85-103; 'queued' has no committed specification: specs/ at "
    f"{DRAFT3} holds {specs3}. Its '(a control the screen did not run)' is right for S82; S83 (started before the draft) runs "
    "that control, LOCAL3_KEYHINGE")
V3[r"which a hinge on the gate's key-dependence during the slow phase prevents ($20/20$ on a screen)"] = (
    "MATCH", "the clause before it now reads 'a key split that in the records forms before the register's write at the cue has "
    "appeared': marked as an observation from the records (row below)")

# ═════════════════════════════════════════════════════════════════════════════
# THIRD DRAFT: rows for changed sentences about Session G (tex3 = a5f67a2)
# ═════════════════════════════════════════════════════════════════════════════
# --- tex3:36 (abstract)
row3(r"fourteen of the sixteen failures on each memory being key splits, formed in the records before any write at the cue",
     "abstract: label-free failures and their order", "14 of 16 per memory; the split before any CTX write, in the records",
     f"Hebbian {ks16[1]} of {ks16[0]}, delta {ks16[3]} of {ks16[2]} (v1 KEY among the non-ROUTED-BOUND REG3 / REG3_D runs); key split "
     f"(eta^2 by key >= 0.5) at an evaluation before any CTX write (beta^R CTX > max(KEY, VAL) + 0.05) in {first}/{len(failK)}; the "
     f"split already at 1200 in {at1200}; the CTX write never appears in {neverw}",
     "MATCH" if ks16 == (16, 14, 16, 14) and first == len(failK) else "DISAGREE",
     "'formed in the records' marks the order as read from the records (evaluations at 1200, 2400, 3600 and the end), inside "
     "'Exploratory screens, which are not claims' (second pass §2: resolved)")
row3(r"and is found $20/20$ on fresh seeds when a hinge on the gate's key-dependence holds the key split off for the first 2400 updates",
     "abstract: S82 REG3_KEYHINGE", "20/20 on fresh seeds; hinge on updates 1-2400",
     f"REG3_KEYHINGE ROUTED-BOUND {cnt(KH, rb)}/{len(KH)} on {min(KH)}-{max(KH)}; seeds 2100-2119 in no other keyed store on G "
     f"({g_other or 'none'}) or on F, H, I, E, R, the main line ({other_use or 'none'}); hinge for updates 1-2400 "
     "(explore_g_devices.py:16-18 at 83699a2)", "MATCH" if cnt(KH, rb) == 20 and not g_other and not other_use else "PARTIAL")
row3(r"a training-time prior that reads the task's key positions and identities, as the split reads key positions",
     "abstract: what the hinge reads", "key positions and identities; the split reads key positions",
     "explore_g_devices.py:17-18: 'It reads key positions and key identities (token types), as the key split reads key positions; "
     "it reads no stream label'", "MATCH", "the second pass's §3 tex2:36 item: resolved")
# --- tex3:261 (S56)
row3(r"its accuracy plateaus at $0.20$--$0.44$ (the run with the weakest latch lowest)", "S56 REG_NUDGE end accuracy; the lowest run",
     "0.20-0.44; the weakest latch lowest",
     f"run acc {rng(racc.values())} ({', '.join(f'{s} {a_:.2f}' for s, a_ in racc.items())}); lowest run {low} ({racc[low]:.2f}, streams "
     f"{sa[low][0]:.2f}/{sa[low][1]:.2f}); weakest latch (lowest end beta^R CTX) {weak} ({ctx56[weak]:.2f}; the next {sorted(ctx56.values())[1]:.2f})",
     "MATCH" if rng(racc.values()) == "0.20-0.44" and low == weak else "DISAGREE",
     "the range 0.20-0.44 is the records' (second pass §1 item 7, B7/D1: the range is resolved). The new parenthesis is not: "
     f"the lowest run, {low} ({racc[low]:.2f}), has the second-weakest latch (end beta^R CTX {ctx56[low]:.2f}, CTX - max(KEY, VAL) "
     f"{ctx56[low] - max(A['REG_NUDGE'][low]['end']['beta']['KEY'], A['REG_NUDGE'][low]['end']['beta']['VAL']):.2f}); the weakest, "
     f"{weak} (CTX {ctx56[weak]:.2f}, CTX - max(KEY, VAL) {ctx56[weak] - max(A['REG_NUDGE'][weak]['end']['beta']['KEY'], A['REG_NUDGE'][weak]['end']['beta']['VAL']):.2f}), "
     f"plateaus at {racc[weak]:.2f}, with the clean-latch runs. Both are the two runs without a clean latch and without routing "
     "(margin 0 at layers 2 and 3). The error is the referee's: check_G_v2.md's note called 378 'the weakest latch, beta^R CTX "
     "0.77'. Exact: '(the lowest of the two runs without a clean latch)' or no parenthesis"
     if low != weak else "second pass §1 item 7 (B7/D1): resolved")
row3(r"with the two streams within $0.03$ of each other in most runs", "S56 REG_NUDGE |stream 0 - stream 1|", "most runs",
     f"within 0.03 in {n_within_2dp}/10 on 2-dp stream accuracies, {n_within_ex}/10 unrounded", "MATCH" if n_within_ex > 5 else "PARTIAL",
     "'most' holds on either rounding (the second draft's 'seven of ten' held on 2 dp only)")
# --- tex3:263 (S72/S73)
row3(r"so S72's pre-fixed reading is ``not discovered'' and S73's ``discovered (register)'' does not apply",
     "S72 / S73 label-free readings, by their own names", "S72 'not discovered'; S73 'discovered (register)' does not apply",
     f"S72 explore_g_register3.py:{lineno(G, 'explore_g_register3.py', 'REG3 beats BASELINE')} \"'discovered' REG3 beats BASELINE; "
     f"'not discovered' otherwise\": {paired(R3, BH, rb)[:2]}, p = {paired(R3, BH, rb)[2]:.4f} -> 'not discovered'; S73 "
     f"'discovered (register)': {paired(R3D, BD, rb)[:2]}, p = {paired(R3D, BD, rb)[2]:.4f} -> does not apply",
     "MATCH" if paired(R3, BH, rb)[:2] == (4, 0) and paired(R3D, BD, rb)[:2] == (4, 0) else "DISAGREE",
     "second pass §3 tex2:263 (the label): resolved")
row3(r"not separated above keys and values in 27 of the 28 such runs", "end beta^R in the key-split runs", "27 of 28",
     f"{len(ksr)} key-split runs (v1 KEY, REG3 + REG3_D, all runs: {kc['Hebbian REG3'][1]} + {kc['delta REG3_D'][1]}); CTX <= max(KEY, VAL) "
     f"+ 0.05 in {ctxlow}; the exception REG3|712 CTX {R3[712]['end']['beta']['CTX']:.2f} vs {R3[712]['end']['beta']['KEY']:.2f} / "
     f"{R3[712]['end']['beta']['VAL']:.2f}", "MATCH" if (ctxlow, len(ksr)) == (27, 28) else "DISAGREE",
     "'such runs' are the runs locked into a key split, the 14 + 14 of the parenthesis (on delta two of them bound)")
row3(r"the half-routed baseline's at $0.86$--$1.00$, also at keys)", "BASELINE_D half-routed bound runs, probe L3 at K",
     "0.86-1.00 at keys", f"{len(half)} runs; K {rng([k for k, _ in hk])} (V {rng([v for _, v in hk])})",
     "MATCH" if rng([k for k, _ in hk]) == "0.86-1.00" else "DISAGREE", "second pass §3 tex2:263 (P17): resolved")
# --- tex3:265 (S82)
row3(r"The runs that fail to discover end in a key split, formed in the records before any write at the cue",
     "label-free runs that fail: class and order", "every failure ends in a key split, formed before any CTX write",
     f"S72/S73: {s7273_key} of the {s7273_fail} non-ROUTED-BOUND REG3 / REG3_D runs are key splits (14 + 14; Hebbian also "
     f"{dict(kc['Hebbian REG3'][5])}, delta {dict(kc['delta REG3_D'][5])}), the split before any CTX write in {first}/{len(failK)}; S82 "
     f"(fresh seeds): REG3 not ROUTED-BOUND {len(f82)} ({dict(f82_cls)}), key splits {len(f82_key)}, the split at an evaluation "
     f"before any CTX write in {f82_first}/{len(f82_key)} ({', '.join(f'{s} split {o[0]}, write {o[1]}' for s, o in f82_ord.items())})",
     "PARTIAL" if s7273_key < s7273_fail or len(f82_key) < len(f82) else "MATCH",
     "the order is now marked ('formed in the records') and the subject is right ('the runs that fail to discover'); but the "
     "sentence says every failing run ends in a key split, and in S72/S73 it is 14 of 16 on each memory (tex3:263, :455 say so), "
     f"in S82's REG3 {len(f82_key)} of {len(f82)}. 'Most of the runs that fail' would be exact")
row3(r"on delta memory, descriptive by the screen's docstring, the same arm binds routed $10/10$ against $0/10$ and $2/10$ (bound: $2/10$ and $4/10$)",
     "S82 phase 2 (delta, 2100-2109)", "descriptive; ROUTED-BOUND 10/10 vs 0/10, 2/10; BOUND refs 2/10, 4/10",
     f"ROUTED-BOUND: REG3_KEYHINGE_D {cnt(KD, rb)}/10, BASELINE_D {cnt(BD82, rb)}/10, REG3_D {cnt(RDD82, rb)}/10; BOUND: {kd_bound}/10, "
     f"{cnt(BD82, bound)}/10, {cnt(RDD82, bound)}/10; phase 2 'descriptive, 10 seeds' (explore_g_devices.py:"
     f"{lineno(S82SPEC, 'explore_g_devices.py', 'Phase 2 (descriptive')})",
     "MATCH" if (cnt(KD, rb), cnt(BD82, rb), cnt(RDD82, rb), cnt(BD82, bound), cnt(RDD82, bound)) == (10, 0, 2, 2, 4) else "DISAGREE",
     "second pass §3 tex2:265 (delta): resolved")
row3(r"binds $0/20$ (0 vs 0 against the baseline; ``hurts'' did not apply, 0 vs 2 against the plain register)",
     "S82 REG3_SLOWGATE: count and readings", "0/20; 0 vs 0; 'hurts' did not apply, 0 vs 2",
     f"ROUTED-BOUND {cnt(SG, rb)}/20; vs BASELINE (SLOWGATE only, BASELINE only) {sg_vs_base[:2]}; vs REG3 (SLOWGATE only, REG3 only) "
     f"({sg_hurts[1]}, {sg_hurts[0]}); 'the device hurts' (REG3 beats it, p <= 0.0625): p = {sg_hurts[2]:.2f} -> "
     f"{'applies' if sgr['hurts'][0] else 'does not apply'}",
     "MATCH" if cnt(SG, rb) == 0 and sg_vs_base[:2] == (0, 0) and sg_hurts[:2] == (2, 0) and not sgr["hurts"][0] else "DISAGREE",
     "'worse than nothing' is gone (second pass §2, 265/384: resolved)")
row3(r"the records show the key split delayed ($\eta^2$ by key above $0.5$ at 1200 in 2 of 20 runs against 16 of 20) but formed (KEY 18 of 20) and the write at the cue never separating (1 of 20), which we read as the slow gate delaying the write along with the split",
     "S82 REG3_SLOWGATE: what the records show, and the marked reading", "2 of 20 vs 16 of 20 at 1200; KEY 18 of 20; write never "
     "separating (1 of 20); the cause marked as ours",
     f"eta^2 by key > 0.5 at 1200: SLOWGATE {sg_gt1200}/20, REG3 {r3_gt1200}/20 (>= 0.5: {sg_split1200}, {r3_split1200}); unbound KEY at "
     f"the end {keyf(SG)}/20; end beta^R CTX > max(KEY, VAL) in {nwrite(SG)}/20 {sg_end_sep}; CTX > max(KEY, VAL) at any evaluation "
     f"from 1200 or at the end in {sg_n_any0}/20 (by more than 0.05 in {sg_n_any5}/20; "
     f"{', '.join(f'{s} at {h} (end {e})' for s, (h, e) in sg_any0.items() if h or e)})",
     "PARTIAL" if (sg_gt1200, r3_gt1200, keyf(SG), nwrite(SG)) == (2, 16, 18, 1) else "DISAGREE",
     "every count reproduces and the cause is marked ('which we read as'). Wording only: 'never separating (1 of 20)' says never and "
     "counts one; the one is seed 2113, whose write at the cue is above keys and values at every evaluation from 2400 to the end, so "
     "'separating in only 1 of 20' would be exact. "
     "The phrase is the referee's own (resolution_G.md §4, 'the write never separates (1/20)')")
row3(r"and the key split does not return in the 2400 updates each run is observed after its transition",
     "S82 REG3_KEYHINGE: the key split after the transition", "does not return within the 2400 updates observed",
     f"max eta^2 by key at the evaluations >= the transition and at the end: {rng(eta_after, 3)}; stopped_at - transition: {gap_set} "
     f"in every run ({dict(sorted(gap82.items()))})",
     "MATCH" if gap_set == [2400] and max(eta_after) < 0.5 else "PARTIAL", "second pass §3 tex2:265 (the return): resolved")
# --- tex3:316, 384, 436, 455, 465
row3(r"a key split that in the records forms before the register's write at the cue has appeared",
     "discovery race: the distant key split's order", "in the records, before the CTX write",
     f"S72/S73 key-split failures: split before any CTX write in {first}/{len(failK)}; S82 REG3: {f82_first}/{len(f82_key)}",
     "MATCH" if first == len(failK) and f82_first == len(f82_key) else "PARTIAL", "marked as read from the records (second pass §2: resolved)")
row3(r"(it needed layer~1, on our reading of S71)", "overturned prediction: the register at layers 2-3",
     "the layer-1 parenthesis marked as ours", f"S56 R1 'architecture insufficient' (REG_NUDGE BOUND ROUTED {cnt(A['REG_NUDGE'], br56)}/10); "
     f"S71 'layer 1 is the block' ({b1} vs {c1}, perfect gates on RandHeaderTask-lite)", "MATCH",
     "marked (second pass §2 tex2:384 and P23/§3 tex:378: resolved). It names S71 only; tex3:261 states the crossing of tasks "
     "and gate inputs")
row3(r"(it binds $0/20$; on our reading it slows the write too)", "overturned prediction: the slow gate", "0/20; the cause marked",
     f"REG3_SLOWGATE ROUTED-BOUND {cnt(SG, rb)}/20; the S82 docstring's purpose 'devices that delay or prevent the key split so the "
     "register's CTX write can emerge' (explore_g_devices.py:3-4)", "MATCH" if cnt(SG, rb) == 0 else "DISAGREE",
     "'worse than nothing' gone; the cause marked (second pass §2: resolved)")
row3(r"the hinge's key term, which on screens prevents the key split at two streams and for the register at a distance, reads key positions and key identities",
     "limits: the key term (register part)", "on screens; key positions and identities",
     f"REG3_KEYHINGE ROUTED-BOUND {cnt(KH, rb)}/20, KEY failures {keyf(KH)} (S82, a screen); explore_g_devices.py:17-18", "MATCH",
     "'on screens' added (second pass §2, 436: resolved); the two-stream half is session H's S85")
row3(r"without it the write at the cue is found on a fifth of seeds in S72 and S73 and on 2 of 20 fresh seeds in S82, and a key split, formed in the records before any write at the cue, ends fourteen of the sixteen others on each memory",
     "limits: without the hinge", "a fifth in S72/S73; 2/20 in S82; 14 of 16 key splits, in the records before the write",
     f"S72 REG3 {cnt(R3, rb)}/20, S73 REG3_D {cnt(R3D, rb)}/20; S82 REG3 {cnt(R82, rb)}/20 (REG3_D {cnt(RDD82, rb)}/10); key splits "
     f"{ks16[1]}/{ks16[0]} and {ks16[3]}/{ks16[2]}, the split first in {first}/{len(failK)}",
     "MATCH" if (cnt(R3, rb), cnt(R3D, rb), cnt(R82, rb)) == (4, 4, 2) and ks16 == (16, 14, 16, 14) else "DISAGREE",
     "second pass §3 tex2:455: resolved. 'the sixteen others on each memory' are S72/S73's (S82 has 18 others, 16 of them key "
     "splits); the sentence's 'on each memory' makes that readable")
row3(r"and is found on every seed of a screen when a hinge keeps the gate off key identity during the slow phase, which on a screen also prevents the two-stream key split that the copy did not repair",
     "conclusion: the hinged register", "every seed of a screen",
     f"REG3_KEYHINGE {cnt(KH, rb)}/20 (Hebbian, 2100-2119), REG3_KEYHINGE_D {cnt(KD, rb)}/10 (delta, 2100-2109, descriptive)",
     "MATCH" if (cnt(KH, rb), cnt(KD, rb)) == (20, 10) else "DISAGREE",
     "'of a screen' and 'on a screen' added (second pass §2, 465: resolved); 'the copy did not repair' is session H's S78")
# --- unchanged lines, new check: the batches table's G row and S83
row3(r"S82 devices against the key split (report 5, \texttt{8973d48}; spec \texttt{83699a2}) & \texttt{8973d48} & 22; 139; 50; 150; 136",
     "batches table, G row: what is running", "G's reports 1-5; nothing running",
     f"S83 spec {s83_spec[0]} ({s83_spec[1]}), started {s83_start[0]} ({s83_start[1]}); s83.log commits before the draft: "
     f"{[c_[0] for c_ in s83_before]}; results files at the draft: {s83_results_at_draft or 'none'}; the third draft {DRAFT3} at {draft3_t}",
     "PARTIAL" if s83_before else "MATCH",
     "the S82 entries are right (row above); the F row says 'S81 running' and the G row does not say 'S83 running', though S83 "
     "was specified and started before the commit. An omission (also tex3:41)", changed=False)

# ═════════════════════════════════════════════════════════════════════════════
# FOURTH DRAFT (c66cc76): S83 (report 6, 076005d) recomputed from the records
# ═════════════════════════════════════════════════════════════════════════════
draft4_t = ctime(DRAFT4)
specs4 = sorted(p_ for p_ in git("ls-tree", "--name-only", "-r", DRAFT4).split("\n") if p_.startswith("specs/"))
# every earlier store byte-identical at report 6
for _f in ("s55_probe_results", "s56_register_results", "s57_latch_results", "s71_layers_results", "s72_register3_results",
           "s73_register3_delta_results", "s82_devices_results"):
    assert git("rev-parse", f"{G5}:{GD}{_f}.json") == git("rev-parse", f"{G6}:{GD}{_f}.json"), _f
assert git("rev-parse", f"{G5}:{GD}s82_best.json") == git("rev-parse", f"{G6}:{GD}s82_best.json")
S83 = load(G6, GD + "s83_results.json")
Y = {a: runs(S83, a) for a in sorted({k.split("|")[0] for k in S83["runs"]})}
NEW83 = ("LOCAL3_KEYHINGE", "REG3_KEYHINGE_POS", "REG3_FLAT", "REG3_KEYHINGE_1200")
DELTA_ORDER83 = ("REG3_FLAT", "REG3_KEYHINGE_POS", "LOCAL3_KEYHINGE")
TAU83 = 0.2                                           # explore_g_s83_child: key term = relu(eta2_key - TAU), TAU 0.2
KH82 = X["REG3_KEYHINGE"]
_r83 = list(S83["runs"].values())
integ83 = (len(_r83), sum(1 for r in _r83 if r.get("ok")), sum(1 for r in _r83 if r.get("attempt", 1) != 1),
           sum(1 for r in _r83 if r["transition"] == transition(r["curve"])),
           sorted({k.split("|")[2] for k in S83["runs"]}), sorted({r["cpu"] for r in _r83}))
seeds83 = {a: sorted(rr) for a, rr in Y.items()}
assert all(seeds83[a] == list(range(2100, 2120)) for a in NEW83) and seeds83["REG3_NUDGE"] == [2100, 2101]
assert set(Y) == set(NEW83) | {"REG3_NUDGE"}

# counts per arm (S82's references beside them)
cnt83 = {}
for a, rr in [("BASELINE (S82)", B82), ("REG3 (S82)", R82), ("REG3_KEYHINGE (S82)", KH82)] + [(a, Y[a]) for a in NEW83 + ("REG3_NUDGE",)]:
    cl = Counter(cls(r) for r in rr.values() if not bound(r))
    cnt83[a] = dict(n=len(rr), bound=cnt(rr, bound), rb=cnt(rr, rb), disc=cnt(rr, disc), mine=cnt(rr, mine3),
                    cls=dict(sorted(cl.items(), key=lambda kv: -kv[1])), tr=trs(rr), tr_rb=trs(rr, rb),
                    bound_unrouted=sorted(s for s, r in rr.items() if bound(r) and not rb(r)))
# paired comparisons (ROUTED-BOUND on 2100-2119); b = first arm only
P83 = {}
for a in NEW83:
    P83[a] = dict(vsB=paired(Y[a], B82, rb), KHvs=paired(KH82, Y[a], rb), vsR=paired(Y[a], R82, rb), Rvs=paired(R82, Y[a], rb))
disc83 = {a: P83[a]["vsB"][2] < 0.05 for a in NEW83}
within83 = {a: (P83[a]["KHvs"][0] - P83[a]["KHvs"][1]) <= 1 for a in NEW83}
read83 = {
    "the register": ("the register is not needed" if disc83["LOCAL3_KEYHINGE"] and within83["LOCAL3_KEYHINGE"] else
                     "the register is needed" if P83["LOCAL3_KEYHINGE"]["KHvs"][2] <= P4 else "neither"),
    "key identity": ("key identity need not be read" if disc83["REG3_KEYHINGE_POS"] and within83["REG3_KEYHINGE_POS"] else
                     "the identity term is needed" if P83["REG3_KEYHINGE_POS"]["KHvs"][2] <= P4 else "neither"),
    "structure": "no structure need be read" if disc83["REG3_FLAT"] and within83["REG3_FLAT"] else "does not apply",
    "the window": "the short window suffices" if within83["REG3_KEYHINGE_1200"] else "does not apply",
}
delta_arm83 = next((a for a in DELTA_ORDER83 if disc83[a]), None)
sum83 = json.loads(git("show", f"{G6}:{GD}s83_summary.json"))["readings"]
repro83 = json.loads(git("show", f"{G6}:{GD}s83_repro.json"))
delta83 = json.loads(git("show", f"{G6}:{GD}s83_delta.json"))
read_same = {"the register": read83["the register"] in sum83["the register"],
             "key identity": read83["key identity"] in sum83["key identity"],
             "structure": sum83["structure"] == read83["structure"], "the window": sum83["the window"] == read83["the window"],
             "delta arm": delta83["arm"] == delta_arm83}
val83 = (cnt(Y["REG3_NUDGE"], rb), cnt(Y["REG3_NUDGE"], bound), len(Y["REG3_NUDGE"]), trs(Y["REG3_NUDGE"]))
assert all(read_same.values()), read_same

# timing: readings and code committed before the first record
spec83_t, start83_t, rep6_t = ctime(S83SPEC), ctime(S83START), ctime(G6)
hist83_code = git("log", "--format=%h %cI", G6, "--", "explore_g_s83.py", "explore_g_s83_child.py").strip().split("\n")
parent83_hist = git("log", "-1", "--format=%h %cI", G6, "--", "explore_g_rhl_parent.py").strip()
devchild83_hist = git("log", "-1", "--format=%h %cI", G6, "--", "explore_g_devices.py", "explore_g_devices_child.py").strip()
_h83 = git("log", "--reverse", "--format=%h %cI", f"{S83SPEC}..{G6}", "--", GD + "s83_results.json").strip().split("\n")
first_rec83 = None
rec83_hist = []
for _ln in _h83:
    _h, _t = _ln.split()
    _n = len(json.loads(git("show", f"{_h}:{GD}s83_results.json")).get("runs", {}))
    rec83_hist.append((_h, _t, _n))
    if _n and first_rec83 is None:
        first_rec83 = (_h, _t, _n)
_ck = [p_ for p_ in git("ls-tree", "--name-only", G6, GD).split("\n") if "checks_s83" in p_]
checks83 = json.loads(git("show", f"{G6}:{_ck[0]}"))
ck_rows = checks83["rows"]
ck_s82 = sum(1 for n_, _ in ck_rows if n_.startswith("S82: "))
ck_ok = all(v for _, v in ck_rows)
ck_t = checks83["provenance"]["started"]
child_sha1 = __import__("hashlib").sha1(git("show", f"{S83SPEC}:explore_g_s83_child.py").encode()).hexdigest()
child_row = [r_ for r_ in S83["meta"]["sha_rows"] if r_[0] == "explore_g_s83_child"]
meta83 = S83["meta"]["provenance"]
log83 = git("show", f"{G6}:{GD}s83.log")
q82 = "82 runs queued on 4 workers" in log83
mins83 = re.search(r"S83 runs done in ([\d.]+) min", log83).group(1)
s83_spec_doc = src(S83SPEC, "explore_g_s83.py")
s83_child_doc = src(S83SPEC, "explore_g_s83_child.py")

# descriptive quantities the draft uses (tex4:269)
def tv83(a, u):
    return [r["term"]["value_at"][u] for r in Y[a].values() if u in r["term"]["value_at"]]


fired83 = {a: sorted(r["term"]["fired"] for r in Y[a].values()) for a in NEW83}
kl_max = {a: {u: max(max(v) for v in tv83(a, u)) for u in ("300", "600", "1200", "2400")} for a in ("REG3_KEYHINGE_POS", "REG3_FLAT")}
kl_med2400 = {a: med([statistics.fmean(v) for v in tv83(a, "2400")]) for a in ("REG3_KEYHINGE_POS", "REG3_FLAT")}
loc_v1 = [statistics.fmean(v) for v in tv83("LOCAL3_KEYHINGE", "1")]
loc_eta1 = med([v + TAU83 for v in loc_v1 if v > 0])          # eta^2 by key at update 1 where the term fired at update 1
loc_later = max(max(max(v) for v in tv83("LOCAL3_KEYHINGE", u)) for u in ("300", "600", "1200"))
kh_eta1 = eta_u["1"]                                          # S82 REG3_KEYHINGE, eta^2 by key at update 1 (the hinge's record)
k12_v1 = med([statistics.fmean(v) + TAU83 for v in tv83("REG3_KEYHINGE_1200", "1")])
loc_margin_end = (min(r["end"]["margin"] for r in Y["LOCAL3_KEYHINGE"].values()),
                  max(r["end"]["margin"] for r in Y["LOCAL3_KEYHINGE"].values()))


def beta_rng83(a, step=None):
    out = {}
    for role in ("CTX", "KEY", "VAL"):
        vs = [(at_step(r, step) or {}).get("beta", {}).get(role) if step else r["end"]["beta"][role] for r in Y[a].values()]
        vs = [v for v in vs if v is not None]
        out[role] = (min(vs), max(vs), len(vs))
    return out


br83 = {a: {st_: beta_rng83(a, st_) for st_ in (1200, 2400, None)} for a in ("REG3_KEYHINGE_POS", "REG3_FLAT", "REG3_KEYHINGE_1200")}
moved2400 = {a: sorted(s for s, r in Y[a].items() if max(abs(at_step(r, 2400)["beta"][x] - 0.1) for x in ("CTX", "KEY", "VAL")) > 0.03)
             for a in ("REG3_KEYHINGE_POS", "REG3_FLAT")}
moved_end = {a: sum(1 for r in Y[a].values() if max(abs(r["end"]["beta"][x] - 0.1) for x in ("CTX", "KEY", "VAL")) > 0.03)
             for a in ("REG3_KEYHINGE_POS", "REG3_FLAT")}
k12_b2400 = {x: med([at_step(r, 2400)["beta"][x] for r in Y["REG3_KEYHINGE_1200"].values()]) for x in ("CTX", "KEY", "VAL")}
k12_fail = {s: ((at_step(r, 1200) or {}).get("eta_key_by_key"), (at_step(r, 2400) or {}).get("eta_key_by_key"), cls(r),
                next((st_["step"] for st_ in r["stats"] if isinstance(st_["step"], int) and (st_.get("eta_key_by_key") or 0) >= 0.5), None))
            for s, r in sorted(Y["REG3_KEYHINGE_1200"].items()) if not rb(r)}
pos_rb = sorted(s for s, r in Y["REG3_KEYHINGE_POS"].items() if rb(r))
pos_rb_also = {s: (rb(R82[s]), rb(NW[s])) for s in pos_rb}

# seeds of the drafted main-line test: X 600-619, L 1600-1619 in any record on any fetched branch head
heads = [h for h in git("for-each-ref", "--format=%(refname:short)", "refs/remotes/origin").split("\n") if h and not h.endswith("HEAD")]
seed_hits = []
for h in heads:
    try:
        out_ = git("grep", "-l", "-E", r'"[A-Za-z0-9_]+\|(6[01][0-9]|16[01][0-9])(\||")|"seed": (6[01][0-9]|16[01][0-9])[,}]', h, "--", "*.json")
    except subprocess.CalledProcessError:
        out_ = ""
    seed_hits += [x for x in out_.split("\n") if x]
spec_reg = src(G6, GD + "spec_draft_register.md")
spec_reg_arms = all(a in spec_reg for a in ("BASELINE", "REG3 ", "REG3_KEYHINGE", "LOCAL3_KEYHINGE")) and \
    "X 600-619, L 1600-1619" in spec_reg and "ROUTED-BOUND >= 18/20 on each machine" in spec_reg
fold83 = [l_.split(" ", 2) for l_ in git("log", "--format=%h %cI %s", f"{DRAFT3}..{DRAFT4}").strip().split("\n") if "S83" in l_]

# record counts of the provenance row
prov_counts = (len(S55["runs"]), len(S56["runs"]) + len(S57["runs"]), len(S71["runs"]), len(S72["runs"]) + len(S73["runs"]),
               integ82[0], integ83[0])

# ═════════════════════════════════════════════════════════════════════════════
# FOURTH DRAFT: V4 revisions of rows whose fragment survives in an amended sentence
# ═════════════════════════════════════════════════════════════════════════════
V4["Each is marked where it appears"] = (
    "MATCH", "the three tex3:259 statements the third pass found unmarked are now marked at tex4:261 ('(where, on our reading, "
    "...)', '(our reading: the gate reads only the window, so nothing asks the stack to keep it)', '..., as the routing study "
    "would predict, is our reading of it'), so every place the referee's checks listed for session G is marked (third pass "
    "§2: resolved). One new session statement in the fourth version is not marked: tex4:269 'where the ratio term removes "
    "only the share of the gate's variation organised by key and leaves it free to commit by stream' (report 6's 'What "
    "happened (exploratory)'); see its row")
V4["Training removes the cue"] = (
    "MATCH", "descriptive (report_1.md 'Anything unexpected'); the causal clause is now marked: '(our reading: the gate reads "
    "only the window, so nothing asks the stack to keep it)'")
V4["is found label-free on a fifth of seeds"] = (
    "MATCH", f"the fourth draft adds '(2 of 20 on the fresh seeds of the device screen)': S72/S73 {cnt(R3, rb)}/20 and "
    f"{cnt(R3D, rb)}/20; S82 REG3 {cnt(R82, rb)}/20 on 2100-2119 (third pass §2, tex3:465: resolved). The delta 2/10 is not "
    "given here; tex4:267 and :458 carry the rest")

# ═════════════════════════════════════════════════════════════════════════════
# FOURTH DRAFT: rows for changed sentences about Session G (tex4 = c66cc76)
# ═════════════════════════════════════════════════════════════════════════════
nolatch = sorted(s for s in racc if s not in latch_ok)
low_nl = min(nolatch, key=racc.get)
# --- tex4:41
row4(["G's S55--S83", "G's S83, H's S86 and R's Phase~4 are complete"], "running list and the rounds: G's S83",
     "S55-S83; S83 complete", f"S83: spec {S83SPEC} ({spec83_t}), started {S83START} ({start83_t}), {integ83[0]} records at {G6} "
     f"(report 6, {rep6_t}; 82 queued: {q82}; SCREEN_DONE in s83.log: {'SCREEN_DONE' in log83}); the draft {DRAFT4} at {draft4_t}",
     "MATCH" if integ83[0] == 82 and rep6_t < draft4_t and "SCREEN_DONE" in log83 else "DISAGREE",
     "third pass §2 / resolution_G3 item 6 (S83 omitted at tex3:41): resolved; the G branch's head is still 076005d")
# --- tex4:59
row4("the register and the identity term are both needed", "index row: S83's readings", "both needed",
     f"'the register': {read83['the register']} (KH only {P83['LOCAL3_KEYHINGE']['KHvs'][0]}, LOCAL3_KEYHINGE only "
     f"{P83['LOCAL3_KEYHINGE']['KHvs'][1]}); 'key identity': {read83['key identity']} (KH only {P83['REG3_KEYHINGE_POS']['KHvs'][0]}, "
     f"POS only {P83['REG3_KEYHINGE_POS']['KHvs'][1]})",
     "MATCH" if read83["the register"] == "the register is needed" and read83["key identity"] == "the identity term is needed" else "DISAGREE",
     "the readings' own names (explore_g_s83.py docstring)")
# --- tex4:261 (S55 marking)
row4([r"(where, on our reading, the decayed sum of the block still carries the header)",
      r"(our reading: the gate reads only the window, so nothing asks the stack to keep it)",
      r"as the routing study would predict \cite{routing}, is our reading of it"],
     "S55: the three interpretations", "marked as ours", "check_G.md / third pass §2: unmarked since the first draft; now each "
     "carries 'on our reading' / 'our reading' / 'is our reading of it'; the reading 'absent' itself is S55's", "MATCH",
     "third pass §2 (tex3:259 three statements) and resolution_G3 item 2 (tex3:342): resolved")
# --- tex4:263 (S56 parenthesis)
row4(r"its accuracy plateaus at $0.20$--$0.44$ (the lower of the two runs without a clean latch)",
     "S56 REG_NUDGE: the range and the lowest run", "0.20-0.44; the lowest is the lower of the two runs without a clean latch",
     f"run acc {rng(racc.values())}; clean latch (CTX >= 0.9, KEY and VAL <= 0.1) in {len(latch_ok)}/10; without: {nolatch} "
     f"(acc {', '.join(f'{s} {racc[s]:.3f}' for s in nolatch)}; end beta^R CTX {', '.join(f'{s} {ctx56[s]:.2f}' for s in nolatch)}); "
     f"the lower of them {low_nl}; the lowest run overall {low}",
     "MATCH" if rng(racc.values()) == "0.20-0.44" and len(nolatch) == 2 and low_nl == low else "DISAGREE",
     "third pass §1 item 3 / resolution_G3 item 1 (NEW ERROR at tex3:261): resolved, with the wording the third pass proposed. "
     "The parenthesis reads as a gloss on the lower bound 0.20")
# --- tex4:267 (S82)
row4(r"Most of the runs that fail to discover end in a key split (28 of 32 in S72 and S73), formed in the records before any write at the cue",
     "label-free runs that fail: class and order", "most; 28 of 32 in S72/S73; the split before any CTX write, in the records",
     f"S72/S73: {s7273_key} of the {s7273_fail} non-ROUTED-BOUND REG3 / REG3_D runs are key splits; the split before any CTX write "
     f"in {first}/{len(failK)}; S82 REG3 (not cited here): {len(f82_key)} of {len(f82)}, order {f82_first}/{len(f82_key)}",
     "MATCH" if (s7273_key, s7273_fail) == (28, 32) and first == len(failK) else "DISAGREE",
     "third pass §2 tex3:265 'most' / resolution_G3 item 3: resolved")
row4([r"the records show the key split delayed ($\eta^2$ by key above $0.5$ at 1200 in 2 of 20 runs against 16 of 20) but formed (KEY 18 of 20)",
      r"and the write at the cue separating in only 1 of 20, which we read as the slow gate delaying the write along with the split"],
     "S82 REG3_SLOWGATE: what the records show", "2 of 20 vs 16 of 20; KEY 18 of 20; the write separating in only 1 of 20",
     f"eta^2 by key > 0.5 at 1200: SLOWGATE {sg_gt1200}/20, REG3 {r3_gt1200}/20; unbound KEY {keyf(SG)}/20; CTX > max(KEY, VAL) at any "
     f"evaluation or the end in {sg_n_any0}/20 ({[s for s, (h, e) in sg_any0.items() if h or e]}), at the end in {nwrite(SG)}/20",
     "MATCH" if (sg_gt1200, r3_gt1200, keyf(SG), nwrite(SG), sg_n_any0) == (2, 16, 18, 1, 1) else "DISAGREE",
     "third pass §2 tex3:265 'never separating' / resolution_G3 item 4: resolved, with the third pass's wording")
# --- tex4:269 (S83)
row4(r"Session~G then screened, on S82's seeds and against its records (S83; each S82 reference reproduced bit for bit through 2400 first)",
     "S83: seeds, references and the phase-0 reproduction", "S82's seeds and records; each reference reproduced through 2400 first",
     f"new arms on {seeds83['LOCAL3_KEYHINGE'][0]}-{seeds83['LOCAL3_KEYHINGE'][-1]}, paired with S82's records (byte-identical at {G6}); "
     f"s83_repro.json {repro83['result']} at {repro83['time']}, before the first S83 record (store started {rec83_hist[0][1] if rec83_hist else '-'}; "
     f"first records at {first_rec83[0]}, {first_rec83[1]}); the docstring's phase 0 reruns each reference arm 'at seed 2100' through 2400",
     "PARTIAL" if all(repro83["result"].values()) else "DISAGREE",
     "every reference arm reproduced, and before any S83 run; but the reproduction is one seed per arm (2100), which "
     "explore_g_s83.py:12-15 (PHASE 0) fixes, not each S82 record. Exact: 'each S82 reference arm reproduced bit for bit through 2400 "
     "at seed 2100 first'. The bit-for-bit comparison itself is taken from s83_repro.json and s83.log (not recomputable without "
     "running)")
row4(r"The key term on the window gate without the register binds routed $0/20$ (the register beats it 20 vs 0; ``the register is needed'')",
     "S83 LOCAL3_KEYHINGE and 'the register'", "0/20; 20 vs 0; 'the register is needed'",
     f"LOCAL3_KEYHINGE ROUTED-BOUND {cnt83['LOCAL3_KEYHINGE']['rb']}/20 (BOUND {cnt83['LOCAL3_KEYHINGE']['bound']}/20, "
     f"unrouted, transitions {cnt83['LOCAL3_KEYHINGE']['tr']}); REG3_KEYHINGE only {P83['LOCAL3_KEYHINGE']['KHvs'][0]}, LOCAL3_KEYHINGE only "
     f"{P83['LOCAL3_KEYHINGE']['KHvs'][1]}, p = {P83['LOCAL3_KEYHINGE']['KHvs'][2]:.3g} <= p(4 vs 0) {P4} -> '{read83['the register']}' "
     f"(not discovered: {P83['LOCAL3_KEYHINGE']['vsB'][:2]}, so 'not needed' could not apply)",
     "MATCH" if cnt83["LOCAL3_KEYHINGE"]["rb"] == 0 and P83["LOCAL3_KEYHINGE"]["KHvs"][:2] == (20, 0) and read83["the register"] == "the register is needed" else "DISAGREE",
     "'binds routed 0/20': 5 runs bind unrouted (the index's 'binds' would be wrong without 'routed'; it has it)")
row4(r"on our reading of G's traces, the hinge has nothing to release there, since the window gate's key-structured variation is already small and nothing carries the stream",
     "S83 LOCAL3_KEYHINGE: why (marked)", "marked as a reading of the traces",
     f"key-term firings {fired83['LOCAL3_KEYHINGE'][0]}-{fired83['LOCAL3_KEYHINGE'][-1]} (median {med(fired83['LOCAL3_KEYHINGE'])}) against "
     f"{fired83['REG3_KEYHINGE_1200'][0]}-{fired83['REG3_KEYHINGE_1200'][-1]} with the register (1200-update window); eta^2 by key at update 1 "
     f"about {loc_eta1:.2f} (median, = term + tau where it fired) against {kh_eta1:.2f} with the register (S82) and {k12_v1:.2f} (S83 1200); term "
     f"value {loc_later} at 300-1200; end margin {loc_margin_end[0]:.2f}-{loc_margin_end[1]:.2f} (unrouted)",
     "MATCH", "marked ('on our reading of G's traces'); the traces agree. report_6.md's 'eta^2_key 0.18 at update 1' is the term's "
     "median value (eta^2 - 0.2), not eta^2; the draft gives no number")
row4(r"A term that reads key positions but not identities (the gate's KL from uniform at key positions, updates 1--2400) binds $1/20$ (``the identity term is needed'', 19 vs 0)",
     "S83 REG3_KEYHINGE_POS and 'key identity'", "1/20; 19 vs 0; 'the identity term is needed'",
     f"ROUTED-BOUND {cnt83['REG3_KEYHINGE_POS']['rb']}/20 (seed {pos_rb}); KH only {P83['REG3_KEYHINGE_POS']['KHvs'][0]}, POS only "
     f"{P83['REG3_KEYHINGE_POS']['KHvs'][1]}, p = {P83['REG3_KEYHINGE_POS']['KHvs'][2]:.3g} -> '{read83['key identity']}'; vs BASELINE "
     f"{P83['REG3_KEYHINGE_POS']['vsB'][:2]}, p = {P83['REG3_KEYHINGE_POS']['vsB'][2]:.2g} (not discovered); the term: KL(g || uniform) at the "
     "body's KEY positions (explore_g_s83_child.py:16-17)",
     "MATCH" if cnt83["REG3_KEYHINGE_POS"]["rb"] == 1 and P83["REG3_KEYHINGE_POS"]["KHvs"][:2] == (19, 0) and read83["key identity"] == "the identity term is needed" else "DISAGREE")
row4(r"and the same term at every body position $0/20$", "S83 REG3_FLAT", "0/20",
     f"ROUTED-BOUND {cnt83['REG3_FLAT']['rb']}/20 (BOUND {cnt83['REG3_FLAT']['bound']}); KH only {P83['REG3_FLAT']['KHvs'][0]}; 'no structure need "
     f"be read': {read83['structure']}; the term at every body position t < body_len, the query triple excluded (explore_g_s83_child.py:18-19)",
     "MATCH" if cnt83["REG3_FLAT"]["rb"] == 0 else "DISAGREE", "the reading's name ('does not apply') is not given; nothing wrong")
row4(r"both clamp the gate to exactly uniform and the register's write never leaves its initial value, where the ratio term removes only the share of the gate's variation organised by key and leaves it free to commit by stream",
     "S83 POS / FLAT: what the records show, and why", "the gate exactly uniform; the write never leaves its initial value",
     f"KL max over runs at 300 / 600 / 1200 / 2400: POS {' / '.join(f'{v:.2g}' for v in kl_max['REG3_KEYHINGE_POS'].values())} (median at 2400 "
     f"{kl_med2400['REG3_KEYHINGE_POS']:.3f}), FLAT {' / '.join(f'{v:.2g}' for v in kl_max['REG3_FLAT'].values())}; beta^R CTX range at 1200: POS "
     f"{br83['REG3_KEYHINGE_POS'][1200]['CTX'][0]:.3f}-{br83['REG3_KEYHINGE_POS'][1200]['CTX'][1]:.3f}, FLAT {br83['REG3_FLAT'][1200]['CTX'][0]:.3f}-"
     f"{br83['REG3_FLAT'][1200]['CTX'][1]:.3f}; at 2400: POS CTX up to {br83['REG3_KEYHINGE_POS'][2400]['CTX'][1]:.2f}, KEY up to "
     f"{br83['REG3_KEYHINGE_POS'][2400]['KEY'][1]:.2f} (moved > 0.03 from 0.10 in {len(moved2400['REG3_KEYHINGE_POS'])}/20), FLAT up to "
     f"{br83['REG3_FLAT'][2400]['CTX'][1]:.3f} ({len(moved2400['REG3_FLAT'])}/20 moved); at the end: CTX POS {br83['REG3_KEYHINGE_POS'][None]['CTX'][0]:.2f}-"
     f"{br83['REG3_KEYHINGE_POS'][None]['CTX'][1]:.2f}, FLAT {br83['REG3_FLAT'][None]['CTX'][0]:.2f}-{br83['REG3_FLAT'][None]['CTX'][1]:.2f} (moved in "
     f"{moved_end['REG3_KEYHINGE_POS']}/20 and {moved_end['REG3_FLAT']}/20). With the key term (1200): beta^R at 2400 median "
     f"{k12_b2400['CTX']:.2f} / {k12_b2400['KEY']:.2f} / {k12_b2400['VAL']:.2f}",
     "PARTIAL" if moved_end["REG3_KEYHINGE_POS"] + moved_end["REG3_FLAT"] > 0 else "MATCH",
     "true while the terms hold: the KL is <= 1.4e-5 at 300-1200 in every run and the write stays at 0.10 +- 0.02 through 1200 "
     "(FLAT through 2400). But 'never' is not true: the write moves once the window ends (POS already by 2400 in some runs; "
     "every run by the end, FLAT 2113 to CTX 0.89), and POS's gate is no longer exactly uniform at 2400. report_6.md and the "
     "README scope it ('through 1200 (FLAT through 2400)', 'for their window'). Exact: 'both hold the gate at uniform while they "
     "are on, and the register's write stays at its initial value through 1200 (the flat term's through 2400)'. The 'where the "
     "ratio term ... leaves it free to commit by stream' clause is report 6's exploratory account and is not marked as a reading")
row4(r"The key term for updates 1--1200 only binds $18/20$ (18 vs 0 against the baseline; one pair short of ``the short window suffices'')",
     "S83 REG3_KEYHINGE_1200 and 'the window'", "18/20; 18 vs 0; one pair short",
     f"ROUTED-BOUND {cnt83['REG3_KEYHINGE_1200']['rb']}/20 (transitions {dict(Counter(cnt83['REG3_KEYHINGE_1200']['tr_rb']))}); vs BASELINE "
     f"{P83['REG3_KEYHINGE_1200']['vsB'][:2]}, p = {P83['REG3_KEYHINGE_1200']['vsB'][2]:.3g} (discovered); KH only {P83['REG3_KEYHINGE_1200']['KHvs'][0]}, "
     f"1200 only {P83['REG3_KEYHINGE_1200']['KHvs'][1]}: the rule needs <= 1 -> '{read83['the window']}'; failures "
     f"{', '.join(f'{s} (eta^2 by key {a_:.2f} at 1200, {b_:.2f} at 2400, >= 0.5 first at {d_}; {c_})' for s, (a_, b_, c_, d_) in k12_fail.items())}",
     "MATCH" if cnt83["REG3_KEYHINGE_1200"]["rb"] == 18 and P83["REG3_KEYHINGE_1200"]["vsB"][:2] == (18, 0) and
     P83["REG3_KEYHINGE_1200"]["KHvs"][:2] == (2, 0) and read83["the window"] == "does not apply" else "DISAGREE",
     "'one pair short': 2 - 0 = 2 against <= 1")
row4(r"So the device that works reads key positions and key identities, and the two that read less fail",
     "S83 summary", "the device that works reads both; the two that read less fail",
     f"REG3_KEYHINGE {cnt(KH82, rb)}/20 and REG3_KEYHINGE_1200 {cnt83['REG3_KEYHINGE_1200']['rb']}/20 (the key term); POS "
     f"{cnt83['REG3_KEYHINGE_POS']['rb']}/20, FLAT {cnt83['REG3_FLAT']['rb']}/20 (the KL terms)", "MATCH",
     "literal and right for these devices. The two that read less also differ in form (an absolute KL, not a ratio), which "
     "the spec's caveat fixed before the run (explore_g_s83_child.py:21-23: 'A null result for pos / flat may be a weakness "
     "of this form, not of \"reading positions\"') and the paragraph's own previous sentence says; see the tex4:439 rows")
row4(r"the main-line test session~G has drafted (\texttt{explore\_out/G/spec\_draft\_register.md}) puts the hinged register against the baseline, against the register without the hinge and against the hinge without the register, 20 seeds per machine on seeds no branch has used, with a rate claim of $\geq 18/20$ on each",
     "S83's main-line spec draft", "arms; 20 seeds per machine; unused seeds; >= 18/20 on each",
     f"spec_draft_register.md at {G6}: C1 vs BASELINE, C2 vs LOCAL3_KEYHINGE, C3 vs REG3; X 600-619, L 1600-1619; R >= 18/20 per "
     f"machine (all present: {spec_reg_arms}); records using 600-619 / 1600-1619 on the {len(heads)} fetched branch heads: "
     f"{seed_hits or 'none'}",
     "MATCH" if spec_reg_arms and not seed_hits else "DISAGREE",
     "the draft's spec makes the pooled claims primary (report 5's proposal had per machine); the draft does not say which")
row4(r"it is the register test proposed to follow \texttt{test\_delta\_recipe}", "the register test's place", "proposed to follow test_delta_recipe",
     f"no record states the order: spec_draft_register.md says 'To be committed to specs/ on the main line, reviewed, and amended'; "
     f"specs/ at {DRAFT4}: {specs4}", "NOT RECORD-CHECKABLE",
     "replaces tex3:267's 'queued behind' (also not record-checkable); the order is the draft's own plan (tex4:472)")
# --- tex4:307, 313 (Table 5)
row4(r"the key hinge on the window gate, no register (S83) / a term reading key positions only / reading nothing / the key hinge for 1200 updates & $0/20$ / $1/20$ / $0/20$ / $18/20$ & ---",
     "Table 5: S83's row", "0/20 / 1/20 / 0/20 / 18/20 (Hebbian); delta not run",
     f"ROUTED-BOUND {cnt83['LOCAL3_KEYHINGE']['rb']}/20 / {cnt83['REG3_KEYHINGE_POS']['rb']}/20 / {cnt83['REG3_FLAT']['rb']}/20 / "
     f"{cnt83['REG3_KEYHINGE_1200']['rb']}/20; phase 2 by the rule: {delta_arm83} (s83_delta.json arm {delta83['arm']}): no delta runs",
     "MATCH" if [cnt83[a]["rb"] for a in NEW83] == [0, 1, 0, 18] and delta_arm83 is None else "DISAGREE",
     "the row does not restate its seeds (2100-2119, paired with S82's); the table's header says 700-719 and the S82 row above it "
     "gives 2100-2119. The caption's 'the others ROUTED-BOUND' covers it")
row4(r"Exploratory (session G, S71--S73, S82 and S83)", "Table 5 caption", "S71-S73, S82 and S83", "S83 row added at tex4:307", "MATCH")
# --- tex4:439 (limits)
row4(r"and session~G's S83 found that a term reading positions only, or nothing, fails ($1/20$, $0/20$)",
     "limits: what S83 found about weaker terms", "a term reading positions only, or nothing, fails (1/20, 0/20)",
     f"POS {cnt83['REG3_KEYHINGE_POS']['rb']}/20, FLAT {cnt83['REG3_FLAT']['rb']}/20 (both KL(g || uniform); KL <= 1.4e-5 at 300-1200 in every "
     "run, i.e. the gate held at uniform)", "PARTIAL" if [cnt83[a]["rb"] for a in ("REG3_KEYHINGE_POS", "REG3_FLAT")] == [1, 0] else "DISAGREE",
     "the counts are right; the generalisation is not. S83 tried one form for each (an absolute KL from uniform), and its "
     "specification fixed in advance that 'A null result for pos / flat may be a weakness of this form, not of \"reading "
     "positions\"' (explore_g_s83_child.py:21-23). The records show that form clamping the gate (tex4:269 says so); report 6: "
     "'It was too strong'. Exact: 'S83's KL terms reading positions only, or nothing, fail (1/20, 0/20; both hold the gate at "
     "uniform)'")
row4(r"the hinge's analogue, which S83 found needs key identities, is untested",
     "limits: the language-model analogue", "S83 found the analogue needs key identities",
     f"the pre-fixed reading 'the identity term is needed' applies (19 vs 0); it compares the key term with one KL term at key "
     "positions", "PARTIAL",
     "the reading's name is S83's, but 'found needs key identities' carries it past what was compared: one position-only form "
     "(KL from uniform) which, by the spec's caveat and the records, failed by clamping the gate. 'which S83's one position-only "
     "term could not replace' would be exact")
# --- tex4:468 (conclusion)
row4(r"(2 of 20 on the fresh seeds of the device screen)", "conclusion: S82's label-free rate", "2 of 20",
     f"S82 REG3 {cnt(R82, rb)}/20 on 2100-2119", "MATCH" if cnt(R82, rb) == 2 else "DISAGREE",
     "third pass §2 tex3:465 'a fifth' / resolution_G3 item 5: resolved")
# --- tex4:472 (next)
row4(r"(session~G's drafted specification: the hinged register against the baseline, the plain register and the hinge without the register; 20 seeds per machine; a rate claim of $\geq 18/20$)",
     "next: the register test", "the drafted spec's arms, seeds and rate claim",
     f"spec_draft_register.md at {G6}: C1-C3 against BASELINE, REG3, LOCAL3_KEYHINGE; 20 seeds per machine; R >= 18/20 per machine",
     "MATCH" if spec_reg_arms else "DISAGREE")
# --- tex4:511 (provenance)
row4(r"S83 the hinge's inputs, the control and the window (report 6, \texttt{076005d}; spec \texttt{b2f9907}) & \texttt{076005d} & 22; 139; 50; 150; 136; 82",
     "batches table, G row", "report 6 at 076005d, spec b2f9907; runs 22; 139; 50; 150; 136; 82",
     f"report 6 {G6} ({rep6_t}) = the G branch head; S83 spec {S83SPEC} ({spec83_t}); records S55 {prov_counts[0]}, S56+S57 "
     f"{prov_counts[1]}, S71 {prov_counts[2]}, S72+S73 {prov_counts[3]}, S82 {prov_counts[4]}, S83 {prov_counts[5]}",
     "MATCH" if prov_counts == (22, 139, 50, 150, 136, 82) and short(GBR) == G6 else "DISAGREE",
     "third pass §2 tex3:508 (S83 omitted) / resolution_G3 item 6: resolved")

# ═════════════════════════════════════════════════════════════════════════════
# FIFTH DRAFT (c5e4508): what changed for session G, and the records behind it
# ═════════════════════════════════════════════════════════════════════════════
import difflib  # noqa: E402

draft5_t = ctime(DRAFT5)
specs5 = sorted(p_ for p_ in git("ls-tree", "--name-only", "-r", DRAFT5).split("\n") if p_.startswith("specs/"))
g_head5 = short(GBR_NOW)
g_after5 = [x for x in git("log", "--reverse", "--format=%h %cI %s", f"{G6}..{GBR_NOW}").strip().split("\n") if x]
# (v6) the v5 assert 'GBR == G6' is replaced: GBR (G at the fifth draft) is still the records' commit; G at the sixth draft
# (GBR6) is reported in the status rows of the sixth-draft section, with its record files compared to GBR's.
gbr_is_g6 = GBR == G6
# tex4 -> tex5 line map (difflib on whole lines; a replaced block of equal size maps line for line, else to its first line)
L45 = {}
for _op, _i1, _i2, _j1, _j2 in difflib.SequenceMatcher(None, TEX4, TEX5, autojunk=False).get_opcodes():
    for _k in range(_i1, _i2):
        L45[_k + 1] = (_j1 + (_k - _i1) + 1) if (_i2 - _i1) == (_j2 - _j1) else _j1 + 1


def t45(s):
    return re.sub(r"tex4:(\d+)", lambda m: f"tex5:{L45[int(m.group(1))]}", s)


# the KL terms at every update the store records them (1, 300, 600, 1200, 2400); max over the three layers per run
POSFLAT = ("REG3_KEYHINGE_POS", "REG3_FLAT")
KLU = ("1", "300", "600", "1200", "2400")
kl_all = {a: {u: {s: max(r["term"]["value_at"][u]) for s, r in Y[a].items()} for u in KLU} for a in POSFLAT}
kl_over = {a: {u: sorted(s for s, v in kl_all[a][u].items() if v > 1e-5) for u in KLU} for a in POSFLAT}
kl_mx = {a: {u: max(kl_all[a][u].values()) for u in KLU} for a in POSFLAT}
kl_mn1 = min(min(kl_all[a]["1"].values()) for a in POSFLAT)
kl_300_1200 = max(kl_mx[a][u] for a in POSFLAT for u in ("300", "600", "1200"))
n_over_300_1200 = sum(len(kl_over[a][u]) for a in POSFLAT for u in ("300", "600", "1200"))
n_over_1 = sum(len(kl_over[a]["1"]) for a in POSFLAT)
kl_lmean = {a: {u: {s: statistics.fmean(r["term"]["value_at"][u]) for s, r in Y[a].items()} for u in KLU} for a in POSFLAT}
kl_lmean_over1200 = {a: sorted((s, round(v, 7)) for s, v in kl_lmean[a]["1200"].items() if v > 1e-5) for a in POSFLAT}
w_on = all(r["term"]["weights"]["2400"] == 1.0 and r["term"]["weights"]["2401"] == 0.0 for a in POSFLAT for r in Y[a].values())


def wdev(r, step):
    s_ = at_step(r, step)
    return max(abs(s_["beta"][x] - 0.1) for x in ("CTX", "KEY", "VAL"))


wdev_max = {a: {st: max(wdev(r, st) for r in Y[a].values()) for st in (0, 1200, 2400)} for a in POSFLAT}
w_moved = {a: {st: sum(1 for r in Y[a].values() if wdev(r, st) > 0.03) for st in (1200, 2400)} for a in POSFLAT}
cav_l = (lineno(S83SPEC, "explore_g_s83_child.py", "CAVEAT fixed now"),
         lineno(S83SPEC, "explore_g_s83_child.py", "A null result for pos / flat may be a weakness of this form"))
cav_txt = ("CAVEAT fixed now: eta2_key is scale-free (a ratio of variances), so the key term pushes even while the gate is near"
           in s83_child_doc and "their gradient vanishes at the uniform" in s83_child_doc
           and 'A null result for pos / flat may be a weakness of this form, not of "reading positions".' in s83_child_doc)
rep6n = " ".join(src(G6, GD + "report_6.md").split())
rep6_opp = "My spec's caveat expected the opposite failure: the absolute KL term too weak near uniform. It was too strong." in rep6n
rep6_ratio = "The key term is a ratio (η²) and removes only the share of variation organised by key. The gate is free to commit by" in rep6n
s75doc = src("03bffdf", "explore_gpu/s75_s4.py")
s75_reg3 = "REG3        Session G's register gate at all three layers + SLOW, no nudge, no split" in s75doc

# ── V5 revisions of carried rows whose fragment survives in an amended sentence
V5["Each is marked where it appears"] = (
    "MATCH", "unchanged sentence (tex5:347). The S83 clause the fourth pass found unmarked is now marked: tex5:271 'where, "
    "on session~G's account, the ratio term removes only the share ...' (fd2fa89). One S83 mechanism is still stated without "
    "a marker: tex5:271 / :441 'by clamping the gate' (report 6's 'What happened (exploratory)'); the clamp itself is in the "
    "records (KL at most 1.4e-5 from update 300 through 1200)")

# ── rows for changed sentences about Session G (tex5 = c5e4508)
# --- tex5:41
row5("every session's queue is empty", "the rounds: 'every session's queue is empty' (G's half)", "G's queue empty",
     f"G's branch at the draft's commit: {GBR} = report 6 {G6} ({rep6_t}); commits after it: {'; '.join(g_after5) or 'none'}; report 6 queues no further "
     f"screen (its 'For the main line' drafts the main-line test only); no queue file under {GD}; the draft {DRAFT5} at {draft5_t}",
     "NOT RECORD-CHECKABLE",
     "a queue is not a record. Nothing on G's branch contradicts it at the draft's commit (no G commit for 38 h before it). "
     f"After the draft G opened a new screen ({len(g_after5)} commits from {g_after5[0].split()[1] if g_after5 else '-'}: S87 specified, "
     "started), so the sentence is now out of date for G. H's half: see check_H_v5")
# --- tex5:271
row5(r"(S83; each S82 reference arm reproduced bit for bit through 2400 at one seed first)",
     "S83: the phase-0 reproduction (rewritten)", "each S82 reference arm reproduced bit for bit through 2400 at one seed",
     f"s83_repro.json {repro83['result']} at {repro83['time']}; explore_g_s83.py:12-15 (PHASE 0) reruns each reference arm at "
     f"seed 2100 through 2400; first S83 records at {first_rec83[0]} ({first_rec83[1]})",
     "MATCH" if all(repro83["result"].values()) and len(repro83["result"]) == 6 and "seed 2100" in s83_spec_doc else "DISAGREE",
     "fourth pass §2 (tex4:269 'each S82 reference reproduced'): resolved (fabf505). The comparison itself is read from "
     "s83_repro.json, not rerun")
row5(r"both hold the gate at uniform while they act (its KL from uniform at most $10^{-5}$ through update 1200 in every run)",
     "S83 POS / FLAT: the gate held at uniform (rewritten)", "the gate held at uniform while the terms act; KL <= 1e-5 through 1200 in every run",
     f"KL(g || U), max over layers, per run: at update 1 {kl_mn1:.2g}-{max(kl_mx[a]['1'] for a in POSFLAT):.2g} (above 1e-5 in "
     f"{n_over_1}/40); at 300 / 600 / 1200 max POS {kl_mx['REG3_KEYHINGE_POS']['300']:.2g} / {kl_mx['REG3_KEYHINGE_POS']['600']:.2g} / "
     f"{kl_mx['REG3_KEYHINGE_POS']['1200']:.2g}, FLAT {kl_mx['REG3_FLAT']['300']:.2g} / {kl_mx['REG3_FLAT']['600']:.2g} / "
     f"{kl_mx['REG3_FLAT']['1200']:.2g} (above 1e-5: POS seeds {kl_over['REG3_KEYHINGE_POS']['1200']} at 1200; the layer mean, the term "
     f"itself, above 1e-5 at 1200: {kl_lmean_over1200}); at 2400 POS max "
     f"{kl_mx['REG3_KEYHINGE_POS']['2400']:.3f} (above 1e-5 in {len(kl_over['REG3_KEYHINGE_POS']['2400'])}/20), FLAT "
     f"{kl_mx['REG3_FLAT']['2400']:.2g}; both terms act through update 2400 (weight 1 at 2400, 0 at 2401, every run: {w_on})",
     "DISAGREE" if (kl_300_1200 > 1e-5 or n_over_1) else "MATCH",
     f"the bound is exceeded twice: at update 1200 in {len(kl_over['REG3_KEYHINGE_POS']['1200'])} POS runs "
     f"({kl_mx['REG3_KEYHINGE_POS']['1200']:.1e}), and at update 1, which 'through update 1200' includes, in all 40 "
     f"({kl_mn1:.1e}-{max(kl_mx[a]['1'] for a in POSFLAT):.1e}). And 'while they act' is not true of POS: its term acts "
     f"through 2400, where its KL is above 1e-5 in {len(kl_over['REG3_KEYHINGE_POS']['2400'])} of 20 runs (max "
     f"{kl_mx['REG3_KEYHINGE_POS']['2400']:.3f}) and its write has already moved more than 0.03 in "
     f"{w_moved['REG3_KEYHINGE_POS'][2400]} of 20. The magnitudes are small and the clamp is real; the stated bound and "
     "scope are not the records'. Exact: 'both held the gate at uniform from update 300 through 1200 (its KL from uniform "
     "at most 1.4e-5 in every run; the flat term's through 2400)'. The fourth pass's own proposed wording ('while they "
     "are on') had the same scope error (fourth_pass.md §2; check_G_v4's note)")
row5(r"and the register's write stays at its initial value through 1200, moving only afterwards",
     "S83 POS / FLAT: the register's write (rewritten)", "at its initial value through 1200, moving only afterwards",
     "max over runs of max |beta^R - 0.10| (CTX, KEY, VAL): " + "; ".join(
         f"{a} at 0 / 1200 / 2400: {wdev_max[a][0]:.4f} / {wdev_max[a][1200]:.4f} / {wdev_max[a][2400]:.3f} (> 0.03 at 2400 in "
         f"{w_moved[a][2400]}/20)" for a in POSFLAT) + f"; at the end moved > 0.03 in {moved_end['REG3_KEYHINGE_POS']}/20 and "
     f"{moved_end['REG3_FLAT']}/20",
     "MATCH" if max(wdev_max[a][1200] for a in POSFLAT) <= 0.02 and moved_end["REG3_KEYHINGE_POS"] + moved_end["REG3_FLAT"] > 0 else "DISAGREE",
     "fourth pass §2 (tex4:269 'never leaves its initial value'): resolved (fabf505). Within 0.012 of 0.10 through 1200; "
     "it moves after (POS by 2400 in most runs, FLAT after 2400)")
row5(r"where, on session~G's account, the ratio term removes only the share of the gate's variation organised by key and leaves it free to commit by stream",
     "S83: the ratio-term account (now marked)", "marked as session G's account",
     f"report_6.md 'What happened (exploratory)': 'The key term is a ratio (η²) and removes only the share of variation "
     f"organised by key. The gate is free to commit by stream' ({rep6_ratio})",
     "MATCH" if rep6_ratio else "DISAGREE", "fourth pass §2 (unmarked ratio-term clause): resolved (fd2fa89)")
row5(r"and the one form tried of each device that reads less fails, by clamping the gate, which S83's specification had allowed in advance as a weakness of the form rather than of reading less",
     "S83 summary (rewritten)", "one form tried of each; fails by clamping the gate; the spec allowed it in advance as the form's weakness",
     f"POS {cnt83['REG3_KEYHINGE_POS']['rb']}/20, FLAT {cnt83['REG3_FLAT']['rb']}/20 (one KL form each); the spec's caveat "
     f"(explore_g_s83_child.py:{cav_l[0]}-{cav_l[1]} at {S83SPEC}, present: {cav_txt}): the KL terms' 'gradient vanishes at the "
     f"uniform gate ... A null result for pos / flat may be a weakness of this form, not of \"reading positions\"'; report 6: "
     f"'My spec's caveat expected the opposite failure: the absolute KL term too weak near uniform. It was too strong.' ({rep6_opp})",
     "PARTIAL" if rep6_opp and cav_txt else "DISAGREE",
     "the counts and 'the one form tried of each' are right (fourth pass §2 tex4:439: the scope is fixed). But the "
     "specification did not allow clamping in advance: it allowed a null result as the form's weakness and named the "
     "opposite mechanism (the term too weak near uniform, its gradient vanishing); report 6 says the caveat 'expected the "
     "opposite failure'. As written, 'which' takes 'clamping the gate' as what was allowed. 'By clamping the gate' is report "
     "6's account, unmarked. Exact: '..., which S83's specification had allowed in advance as a weakness of the form (though "
     "it expected the term too weak, not too strong)'")
# --- tex5:441
row5(r"session~G's S83 found that its KL terms reading positions only, or nothing, fail ($1/20$, $0/20$) by clamping the gate, a weakness its specification had allowed to the form",
     "limits: S83's weaker terms (rewritten)", "its KL terms fail (1/20, 0/20) by clamping the gate, a weakness the spec allowed to the form",
     f"POS {cnt83['REG3_KEYHINGE_POS']['rb']}/20, FLAT {cnt83['REG3_FLAT']['rb']}/20; KL at most {kl_300_1200:.1e} from 300 through 1200; "
     f"the caveat at explore_g_s83_child.py:{cav_l[0]}-{cav_l[1]}; report 6 'expected the opposite failure' ({rep6_opp})",
     "PARTIAL" if [cnt83[a]["rb"] for a in POSFLAT] == [1, 0] else "DISAGREE",
     "fourth pass §2 (tex4:439 'a term reading positions only, or nothing, fails'): the scope is fixed ('its KL terms'). The "
     "appositive has the problem of the tex5:271 row: the specification allowed a null result to the form, and expected the "
     "term too weak, not a clamp. Exact: '..., a weakness of the form its specification had allowed for (if not this one)'")
row5(r"the hinge's analogue, which S83's one position-only term could not replace, is untested",
     "limits: the language-model analogue (rewritten)", "S83's one position-only term could not replace the hinge",
     f"'the identity term is needed' applies: KH only {P83['REG3_KEYHINGE_POS']['KHvs'][0]}, POS only "
     f"{P83['REG3_KEYHINGE_POS']['KHvs'][1]}, p = {P83['REG3_KEYHINGE_POS']['KHvs'][2]:.2g}; POS {cnt83['REG3_KEYHINGE_POS']['rb']}/20",
     "MATCH" if read83["key identity"] == "the identity term is needed" and cnt83["REG3_KEYHINGE_POS"]["rb"] == 1 else "DISAGREE",
     "fourth pass §2 (tex4:439 'needs key identities'): resolved, with the wording the fourth pass proposed (fabf505)")
# --- tex5:456-457 (session I's S75 paragraph names G's register gate)
row5(r"and session~G's register gate at all three layers, ten seeds each",
     "S75 (session I): an arm named as G's register gate", "session G's register gate at all three layers",
     f"explore_gpu/s75_s4.py:23 at 03bffdf (report 2's commit): 'REG3 Session G's register gate at all three layers + SLOW, "
     f"no nudge, no split' ({s75_reg3})", "MATCH" if s75_reg3 else "DISAGREE",
     "the arm's name only; S75's results are session I's records and are not checked here")

# ═════════════════════════════════════════════════════════════════════════════
# SIXTH DRAFT (42c4557): what changed for session G, and G's branch at the draft's commit
# ═════════════════════════════════════════════════════════════════════════════
draft6_t = ctime(DRAFT6)
assert len(TEX6) == len(TEX5), "the sixth draft is not an in-place revision of the fifth"
CH56 = [i + 1 for i, (a_, b_) in enumerate(zip(TEX5, TEX6)) if a_ != b_]
# the commits c5e4508..42c4557 and the lines each changed (in-place replacements: line numbers are the same in tex5 and tex6)
_c56 = git("rev-list", "--reverse", f"{DRAFT5}..{DRAFT6}").split()
C56 = []
_prev = TEX5
for _c in _c56:
    _t = git("show", f"{_c}:{TEXP}").split("\n")
    assert len(_t) == len(TEX5)
    C56.append((_c[:7], ctime(_c), [i + 1 for i, (a_, b_) in enumerate(zip(_prev, _t)) if a_ != b_]))
    _prev = _t
assert _prev == TEX6


def by56(line):
    return ", ".join(f"{h} ({t[11:16]})" for h, t, ls in C56 if line in ls)


# G's branch at the sixth draft's commit (replaces the v5 assert GBR == G6)
gbr6_t = ctime(GBR6)
g_pre6 = [x for x in git("log", "--reverse", "--format=%h %cI %s", f"{G6}..{GBR6}").strip().split("\n") if x]
g_ns6 = [x.split("\t") for x in git("diff", "--name-status", G6, GBR6).strip().split("\n") if x]
g_only_added6 = all(st_ == "A" for st_, *_ in g_ns6)
g_rec_same6 = g_only_added6 and all("s87" in p_ for _, p_ in g_ns6)        # every record file of S55-S83 byte-identical
g_rep_after6 = [p_ for _, p_ in g_ns6 if re.search(r"report_\d+\.md$", p_)]
spec87 = git("log", "--reverse", "--format=%h %cI", GBR6, "--", "explore_g_s87.py").strip().split("\n")[0].split()
start87 = next((x.split()[:2] for x in g_pre6 if "S87 started" in x), None)
rec87 = load(GBR6, GD + "s87_results.json")
runs87 = rec87["runs"]
arms87 = Counter(k_.split("|")[0] for k_ in runs87)
log87 = git("show", f"{GBR6}:{GD}s87.log")
q87 = re.search(r"(\d+) runs queued", log87)
q87 = int(q87.group(1)) if q87 else None
done87 = "runs done" in log87 or "SCREEN_DONE" in log87
NEW87 = ("REG3_FREQHINGE", "REG3_SELFKL", "REG3_TEMP")
new87_recorded = sum(arms87[a] for a in NEW87)
# the first commit at which s87_results.json holds a record, against the specification's commit
_h87 = git("log", "--reverse", "--format=%h %cI", f"{spec87[0]}..{GBR6}", "--", GD + "s87_results.json").strip().split("\n")
first87 = next(((h_, t_, len(load(h_, GD + "s87_results.json")["runs"])) for h_, t_ in (x.split() for x in _h87 if x)
                if load(h_, GD + "s87_results.json")["runs"]), None)
spec_reg_same6 = git("rev-parse", f"{G6}:{GD}spec_draft_register.md") == git("rev-parse", f"{GBR6}:{GD}spec_draft_register.md")
s87_in_draft = [i + 1 for i, l in enumerate(TEX6) if "S87" in l]
g_after6 = [x for x in git("log", "--reverse", "--format=%h %cI", f"{GBR6}..{GBR_NOW}").strip().split("\n") if x]
rep7_now = [p_ for p_ in git("ls-tree", "--name-only", GBR_NOW, GD).split("\n") if p_.endswith("report_7.md")]
# H's round 5 at the same moment (for the same status sentence; H's half is checked in check_H_v6)
HBR_NOW = "origin/claude/explore-H"
hbr6 = git("rev-list", "-1", f"--before={draft6_t}", HBR_NOW).strip()[:7]

import datetime as _dt  # noqa: E402


def _mins(t0, t1):
    d_ = (_dt.datetime.fromisoformat(t1) - _dt.datetime.fromisoformat(t0)).total_seconds()
    return f"{int(d_ // 60)} min {int(d_ % 60)} s"


# ── V6 revisions of carried rows (fragment unchanged, but amended around it or moved with G's branch)
V6[r"and the one form tried of each device that reads less fails, by clamping the gate, which S83's specification had allowed in advance as a weakness of the form rather than of reading less"] = (
    "MATCH" if rep6_opp and cav_txt else "DISAGREE",
    f"v5 PARTIAL ('allowed in advance' read as allowing the clamp). The sentence now adds '(though it expected the term too "
    f"weak near uniform, not too strong)' ({by56(locate('though it expected the term too weak near uniform, not too strong'))}), "
    f"which is the caveat at explore_g_s83_child.py:{cav_l[0]}-{cav_l[1]} and report 6's 'expected the opposite failure'. "
    "RESOLVED. 'By clamping the gate' is still report 6's account without a marker; the clamp itself is in the records "
    "(see the tex6:271 KL row)")
V6[r"S83 the hinge's inputs, the control and the window (report 6, \texttt{076005d}; spec \texttt{b2f9907}) & \texttt{076005d} & 22; 139; 50; 150; 136; 82"] = (
    "MATCH" if g_rec_same6 and not g_rep_after6 else "PARTIAL",
    f"unchanged line. At the sixth draft's commit G's last commit is {GBR6} ({gbr6_t}), not report 6: since {G6} G committed "
    f"{len(g_pre6)} times, all S87 (specification {spec87[0]} {spec87[1]}, started {start87[0] if start87 else '-'}), adding "
    f"only S87 files ({len(g_ns6)} added, none changed: {g_rec_same6}); no report after report 6 ({g_rep_after6 or 'none'}); the "
    f"S87 store held {len(runs87)} of {q87} queued runs. The row lists screens by report, so S87, with no report, is rightly "
    f"absent; every record it pins is byte-identical at {GBR6}")
V6[r"(session~G's drafted specification: the hinged register against the baseline, the plain register and the hinge without the register; 20 seeds per machine; a rate claim of $\geq 18/20$)"] = (
    "MATCH" if spec_reg_arms and spec_reg_same6 else "DISAGREE",
    f"unchanged line (tex6:476). spec_draft_register.md byte-identical at {GBR6} ({spec_reg_same6}); S87 (label-free hinges) does "
    "not touch the drafted test, so the next step is not stale against G's branch at the draft's commit")
V6[r"the hinge's analogue, which S83's one position-only term could not replace, is untested"] = (
    "MATCH" if read83["key identity"] == "the identity term is needed" and new87_recorded == 0 else "PARTIAL",
    f"carried (v5 MATCH). At the draft's commit G's S87 was screening label-free hinges (frequency, self-KL, temperature: "
    f"'can a term that reads no task labels do the key term's work'); its store at {GBR6} held {dict(arms87)}, none of the "
    f"three new arms ({new87_recorded} records), so 'untested' is still right at 42c4557")

# ── rows for changed sentences about Session G (tex6 = 42c4557)
# --- tex6:41 (b5fd2f2, then 78b7aaf)
row6(r"and the sessions' fifth round are running",
     "the rounds: 'the sessions' fifth round are running' (G's half)", "G's new round running at the draft's commit",
     f"G's last commit before the draft ({draft6_t}): {GBR6} ({gbr6_t}). Since report 6 ({G6}): "
     + "; ".join(x[:60] for x in g_pre6) + f". S87: specification {spec87[0]} ({spec87[1]}), started "
     f"{start87[0] if start87 else '-'} ({start87[1] if start87 else '-'}); s87.log '{q87} runs queued', no 'runs done' line "
     f"({not done87}); s87_results.json at {GBR6}: {len(runs87)} records {dict(arms87)}; first record at {first87[0] if first87 else '-'} "
     f"({first87[1] if first87 else '-'}), after the specification: {bool(first87) and first87[1] > spec87[1]}; no report_7.md "
     f"at {GBR6}. The draft names S87 nowhere ({s87_in_draft or 'no line'})",
     "MATCH" if start87 and q87 and len(runs87) < q87 and not done87 and not g_rep_after6 else "DISAGREE",
     f"right for G: S87 was specified {_mins(spec87[1], draft6_t)} and started {_mins(start87[1], draft6_t) if start87 else '-'} "
     f"before the draft and was running at its commit ({len(runs87)} of {q87} runs recorded; the last progress commit "
     f"{gbr6_t[11:19]}, {_mins(gbr6_t, draft6_t)} before), with no result to report. 'Fifth round' is "
     f"the sessions' own count (H's commits say 'round 5'); G's records number no rounds, and the same paragraph calls G's "
     f"S55-S83 part of 'the exploratory sessions' second and third rounds', which a reader cannot square with 'fifth' "
     f"(wording; not a record matter). Both G (S87) and H (S87a, S87c) use the number S87 for their new screens: the draft "
     f"will have to tell them apart when it reports them. Changed at {by56(41)}; the v5 row ('every session's queue is empty') "
     f"is REMOVED5. After the draft: G {len(g_after6)} more progress commits to {short(GBR_NOW)}; report 7 now: {rep7_now or 'none'}")
# --- tex6:271 (b5fd2f2)
row6(r"both held the gate at uniform from update 300 through 1200 (its KL from uniform at most $1.4\times10^{-5}$ in every run; the flat term's through 2400)",
     "S83 POS / FLAT: the gate held at uniform (rewritten again)", "from 300 through 1200, KL <= 1.4e-5 in every run; FLAT's through 2400",
     f"KL(g || U), max over layers, per run (the store records updates 1, 300, 600, 1200, 2400, to 6 decimals): POS at 300 / 600 "
     f"/ 1200 max {kl_mx['REG3_KEYHINGE_POS']['300']:.2g} / {kl_mx['REG3_KEYHINGE_POS']['600']:.2g} / {kl_mx['REG3_KEYHINGE_POS']['1200']:.2g} "
     f"(seed {max(kl_all['REG3_KEYHINGE_POS']['1200'], key=kl_all['REG3_KEYHINGE_POS']['1200'].get)} at 1200); FLAT at 300 / 600 / "
     f"1200 / 2400 max {kl_mx['REG3_FLAT']['300']:.2g} / {kl_mx['REG3_FLAT']['600']:.2g} / {kl_mx['REG3_FLAT']['1200']:.2g} / "
     f"{kl_mx['REG3_FLAT']['2400']:.2g}; POS at 2400 max {kl_mx['REG3_KEYHINGE_POS']['2400']:.3f} (above 1.4e-5 in "
     f"{sum(1 for v in kl_all['REG3_KEYHINGE_POS']['2400'].values() if v > 1.4e-5)}/20; outside the stated scope); at update 1 "
     f"{kl_mn1:.1e}-{max(kl_mx[a]['1'] for a in POSFLAT):.1e} (outside the scope)",
     "MATCH" if (kl_300_1200 <= 1.4e-5 and kl_mx["REG3_FLAT"]["2400"] <= 1.4e-5) else "DISAGREE",
     f"v5 DISAGREE (the bound 1e-5 exceeded at 1200 and at update 1; 'while they act' wrong for POS). Rewritten at {by56(271)} "
     "with the wording the fifth pass proposed: the bound (1.4e-5, the store's 0.000014 for seed 2109 at 1200) and the scope "
     "(300-1200 for both, FLAT also at 2400) hold at every recorded update. 'From 300 through 1200' can be checked only at "
     "the recorded updates 300, 600 and 1200")
# --- tex6:441 (b5fd2f2)
row6(r"session~G's S83 found that its KL terms reading positions only, or nothing, fail ($1/20$, $0/20$) by clamping the gate, a weakness of the form its specification had allowed for (if not this one)",
     "limits: S83's weaker terms (rewritten again)", "fail (1/20, 0/20) by clamping the gate, a weakness of the form the spec allowed for (if not this one)",
     f"POS {cnt83['REG3_KEYHINGE_POS']['rb']}/20, FLAT {cnt83['REG3_FLAT']['rb']}/20; KL at most {kl_300_1200:.1e} from 300 through 1200; "
     f"the caveat at explore_g_s83_child.py:{cav_l[0]}-{cav_l[1]} (a null result 'may be a weakness of this form'; the gradient "
     f"'vanishes at the uniform'); report 6 'expected the opposite failure' ({rep6_opp})",
     "MATCH" if [cnt83[a]["rb"] for a in POSFLAT] == [1, 0] and cav_txt and rep6_opp else "DISAGREE",
     f"v5 PARTIAL ('a weakness its specification had allowed to the form'). Rewritten at {by56(441)} with the fifth pass's "
     "proposed wording; '(if not this one)' states that the specification foresaw a different failure. RESOLVED. 'By "
     "clamping' remains report 6's account, unmarked (as at tex6:271)")

apply_v3()
ORDER = {"DISAGREE": 0, "PARTIAL": 1, "NOT RECORD-CHECKABLE": 2, "MATCH": 3}
ROWS.sort(key=lambda r: (ORDER[r[5]], r[0]))
C = Counter(r[5] for r in ROWS)
CK = {k_: Counter(r[5] for r in ROWS if r[7] == k_) for k_ in ("carried1", "carried2", "carried3", "carried4", "carried5", "new6")}
newly_gone1 = [r for r in REMOVED if locate(r[1], TEX5) is not None]
newly_gone2 = [r for r in REMOVED2 if locate(r[1], TEX5) is not None]
newly_gone3 = [r for r in REMOVED3 if locate(r[1], TEX5) is not None]
newly_gone4 = [r for r in REMOVED4 if locate(r[1], TEX5) is not None]
# rows (carried) whose tex6 line changed since c5e4508 but whose fragment survives: listed for the reader
on_changed6 = sorted((r[0], r[2], r[5], r[7]) for r in ROWS if r[0] in CH56 and r[7] != "new6")


def esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def cnts(c):
    return ", ".join(f"{k} {c.get(k, 0)}" for k in STATUSES)


def L5(needle):
    n_ = locate(needle)
    assert n_ is not None, needle
    return f"tex6:{n_}"


# report 5's count table against the records (kept from the third pass)
def mine_br(r):
    return mine3(r)


rep_rows = {}
for l_ in rep5.split("\n"):
    m_ = re.match(r"\| \**([A-Z0-9_]+)\**( \(|\s*\|)", l_)
    if m_ and m_.group(1) in X:
        cols = [c_.strip() for c_ in l_.strip().strip("|").split("|")]
        fr = lambda t: re.search(r"(\d+)/(\d+)", t).group(0) if re.search(r"(\d+)/(\d+)", t) else t  # noqa: E731
        if len(cols) >= 7 and m_.group(1) not in rep_rows:
            rep_rows[m_.group(1)] = (fr(cols[1]), fr(cols[2]), fr(cols[3]), fr(cols[4]), cols[5])
cmp82 = []
for a in ("BASELINE", "REG3", "REG3_KEYHINGE", "REG3_SLOWGATE", "REG3_NOWIN", "REG3_NUDGE", "ORACLE_ALL", "ORACLE_ALL_D",
          "BASELINE_D", "REG3_D", "REG3_KEYHINGE_D"):
    rr = X[a]
    n = len(rr)
    cl = Counter(cls(r) for r in rr.values() if not bound(r))
    got = (f"{cnt(rr, bound)}/{n}", f"{cnt(rr, rb)}/{n}", f"{cnt(rr, disc)}/{n}", f"{cnt(rr, mine_br)}/{n}",
           ", ".join(f"{k} {v}" for k, v in sorted(cl.items(), key=lambda kv: -kv[1])) or "–")
    rp = rep_rows.get(a)
    same = rp is not None and rp[:4] == got[:4] and all(f"{k} {v}" in rp[4].replace("S-PARTIAL", "STREAM-PARTIAL") for k, v in cl.items())
    cmp82.append((a, n, got, rp, same, sorted(r["transition"] for r in rr.values() if bound(r))))

# report 6's count table against the records
rep6 = src(G6, GD + "report_6.md")
rep6_rows = {}
for l_ in rep6.split("\n"):
    m_ = re.match(r"\| ([A-Z0-9_]+)( \(S82\))? ", l_)
    if m_:
        cols = [c_.strip() for c_ in l_.strip().strip("|").split("|")]
        if len(cols) == 6:
            rep6_rows[m_.group(1)] = cols
cmp83 = []
for key_, a in (("BASELINE (S82)", "BASELINE"), ("REG3 (S82)", "REG3"), ("REG3_KEYHINGE (S82)", "REG3_KEYHINGE")) + tuple((a, a) for a in NEW83 + ("REG3_NUDGE",)):
    c_ = cnt83[key_]
    lo, hi = wilson(c_["rb"], c_["n"])
    from rlib import band as _band  # noqa: E402
    got = (f"{c_['bound']}/{c_['n']}", f"{c_['rb']}/{c_['n']}", f"[{lo:.2f}, {hi:.2f}]", _band(c_["rb"], c_["n"]),
           ", ".join(f"{k} {v}" for k, v in c_["cls"].items()) or "–")
    rp = rep6_rows.get(a)
    if rp:
        same = (rp[2] == got[0] and got[1] in rp[3] and (got[2] in rp[3] or c_["n"] == 2) and (got[3] in rp[3] or c_["n"] == 2)
                and all(f"{k} {v}" in rp[4].replace("S-PARTIAL", "STREAM-PARTIAL") for k, v in c_["cls"].items()))
    else:
        same = None
    cmp83.append((key_, c_, got, rp, same))

L = []
L.append("# Revision 9 draft, sixth version, vs Session G's records (referee, sixth pass)")
L.append("")
L.append(f"Draft: `{TEXP}` at `{DRAFT6}` (sixth version, committed {draft6_t}), cited as tex6:NNN; the fifth version `{DRAFT5}` "
         f"({draft5_t}) as tex5:NNN, the fourth `{DRAFT4}` as tex4:NNN, the third `{DRAFT3}` as tex3:NNN, the second `{DRAFT2}` as "
         f"tex2:NNN and the first `{DRAFT}` as tex:NNN. The sixth version has {len(TEX6) - (TEX6[-1] == '')} lines, like the fifth: "
         f"its {len(_c56)} commits replace lines in place (tex5:N = tex6:N), changing lines {CH56} ("
         + "; ".join(f"`{h}` {t[11:19]}: {ls}" for h, t, ls in C56) + "). Records: Session G at "
         f"`{G}` for S55-S57 and S71-S73 (byte-identical at `{G5}` and `{G6}`); S82 at `{G5}` (byte-identical at `{G6}`); S83 at "
         f"`{G6}` (report 6, {rep6_t}; the commit the draft pins); F's S70 store at `{F}`. G's records are read at `{GBR}`, the "
         f"branch head at the fifth draft's commit (report 6: {gbr_is_g6}). G's last commit before the sixth draft's commit is "
         f"`{GBR6}` ({gbr6_t}); between them G committed {len(g_pre6)} times, all S87, adding {len(g_ns6)} files and changing none "
         f"(every S55-S83 record file byte-identical at `{GBR6}`: {g_rec_same6}). The branch has since moved to `{short(GBR_NOW)}` "
         f"({len(g_after6)} more S87 progress commits; report 7: {rep7_now or 'none'}). "
         "Generated by `docs/review_2026-10/rev9/check_G_v6.py` (a copy of check_G_v5.py; stdlib + rlib.py; records and code read "
         "with `git show`; no torch).")
L.append("")
L.append("Method. Each fragment quoted by the first to fifth passes (asserted on that pass's draft) is searched for in the sixth "
         "draft, all of a row's fragments on one line. Found: the row is kept with its recomputation and reported at its tex6 "
         "line. Not found: the row is listed as REMOVED below, and the sentence that replaced it has its own row ('new'), "
         "re-anchored on the new wording, with its number re-checked against the record it cites. Where a fragment survives "
         "inside a sentence that was amended around it, or its truth moved with G's branch, the row carries a v6 status and "
         "note. Notes written by an earlier pass keep that pass's line references (tex5:N and so on); the line column is always "
         "tex6.")
L.append("")
L.append(f"**Rows: {len(ROWS)} — {cnts(C)}.** Carried from the first pass ({sum(CK['carried1'].values())}): {cnts(CK['carried1'])}. "
         f"Carried from the second pass ({sum(CK['carried2'].values())}): {cnts(CK['carried2'])}. Carried from the third pass "
         f"({sum(CK['carried3'].values())}): {cnts(CK['carried3'])}. Carried from the fourth pass ({sum(CK['carried4'].values())}): "
         f"{cnts(CK['carried4'])}. Carried from the fifth pass ({sum(CK['carried5'].values())}): {cnts(CK['carried5'])}. New for "
         f"the sixth version ({sum(CK['new6'].values())}): {cnts(CK['new6'])}. REMOVED: {len(REMOVED)} first-pass rows "
         f"({len(newly_gone1)} newly), {len(REMOVED2)} second-pass rows ({len(newly_gone2)} newly), {len(REMOVED3)} third-pass rows "
         f"({len(newly_gone3)} newly), {len(REMOVED4)} fourth-pass rows ({len(newly_gone4)} newly) and {len(REMOVED5)} fifth-pass "
         f"rows (fifth-pass DISAGREE {sum(1 for r in REMOVED5 if r[3] == 'DISAGREE')}, PARTIAL "
         f"{sum(1 for r in REMOVED5 if r[3] == 'PARTIAL')}, NOT RECORD-CHECKABLE "
         f"{sum(1 for r in REMOVED5 if r[3] == 'NOT RECORD-CHECKABLE')}, MATCH {sum(1 for r in REMOVED5 if r[3] == 'MATCH')}).")
L.append("")
L.append("Carried rows on a line that changed in the sixth version (fragment intact): "
         + ("; ".join(f"tex6:{n_} {it_} ({kd_}, {st_})" for n_, it_, st_, kd_ in on_changed6) or "none") + ".")
L.append("")
L.append("Record integrity: " + "; ".join(integrity) + f"; S82 (at {G5}): {integ82[0]} records, {integ82[1]} ok, {integ82[2]} retried, "
         f"recorded transition = rlib.transition(curve) in {integ82[3]}; S83 (at {G6}): {integ83[0]} records, {integ83[1]} ok, "
         f"{integ83[2]} retried, recorded transition = rlib.transition(curve) in {integ83[3]}, code SHA {integ83[4]}, CPU {integ83[5]}.")
L.append("")
L.append("Definitions (re-implemented): BOUND = transition not None; ROUTED*@end = end margin >= 0.9 and end eta^2 by stream at KEY "
         "> 0.9; ROUTED-BOUND = BOUND and ROUTED*@end; DISCOVERED (F's) = BOUND and end VAL cos < 0.5 (explore_g_rhl_parent.py:6-17, "
         f"unchanged since {parent82_hist}); G's BOUND ROUTED = BOUND, VAL map one-to-one at every gated layer, every stream >= 0.9; "
         "failure class = test_router_layout v1 on the end gate (rlib.fail_class), MERGED if STREAM-PARTIAL and the L3 VAL map is "
         "not one-to-one; 'beats' = one-sided exact McNemar p < 0.05 (rlib.mcnemar1). S82's readings as explore_g_devices.py:32-40 "
         f"fixes them at {S82SPEC}. S83's as explore_g_s83.py's docstring fixes them at {S83SPEC}: 'discovered' = beats S82's "
         "BASELINE (p < 0.05); 'within 1 pair of REG3_KEYHINGE' = (KH only) - (arm only) <= 1; 'beats by >= 4 vs 0' = one-sided "
         f"p <= p(4 vs 0) = {P4}; 'the register is not needed' / 'is needed' / 'neither' (LOCAL3_KEYHINGE); 'key identity need not "
         "be read' / 'the identity term is needed' / 'neither' (REG3_KEYHINGE_POS); 'no structure need be read' (REG3_FLAT); 'the "
         "short window suffices' (REG3_KEYHINGE_1200); phase 2 = the least-reading of FLAT, POS, LOCAL3_KEYHINGE discovered on "
         "Hebbian memory, none -> no phase 2.")
L.append("")
L.append("## Rows (DISAGREE, then PARTIAL, NOT RECORD-CHECKABLE, MATCH)")
L.append("")
L.append("| tex6 line | origin | item | draft says | recomputed | status | note |")
L.append("|---|---|---|---|---|---|---|")
for n6_, org, it, d_, rc, s_, nt, kd, _k in ROWS:
    L.append(f"| tex6:{n6_} | {org} | {esc(it)} | {esc(d_)} | {esc(rc)} | **{s_}** | {esc(nt)} |")
L.append("")
L.append("## G's branch at the sixth draft's commit (replaces check_G_v5's assert that it was report 6)")
L.append("")
L.append(f"- Last G commit at or before {draft6_t}: `{GBR6}` ({gbr6_t}). Commits since report 6 (`{G6}`, {rep6_t}): "
         + "; ".join(f"`{x.split()[0]}` {x.split()[1]} {x.split(' ', 2)[2][:70]}" for x in g_pre6) + ".")
L.append(f"- Files: {', '.join(f'{st_} {p_}' for st_, p_ in g_ns6)}. No file of S55-S83 and no report changed: {g_rec_same6}.")
L.append(f"- S87 (label-free hinges; seeds 2700-2719, 102 runs): specification `{spec87[0]}` ({spec87[1]}); started "
         f"`{start87[0] if start87 else '-'}`; first record at `{first87[0] if first87 else '-'}` ({first87[1] if first87 else '-'}, "
         f"{first87[2] if first87 else 0} records), after the specification: {bool(first87) and first87[1] > spec87[1]}. At `{GBR6}`: "
         f"{len(runs87)} of {q87} queued runs recorded ({dict(arms87)}); none of {', '.join(NEW87)}; no 'runs done' in s87.log "
         f"({not done87}); no report_7.md.")
L.append(f"- What the draft says of it: tex6:41 'the sessions' fifth round are running' (right for G: running, no result); S87 "
         f"named on no line ({s87_in_draft or 'none'}); the provenance row (tex6:515) pins report 6, G's last report (right: S87 "
         f"has none); the next step (tex6:476) is G's drafted spec, byte-identical at `{GBR6}` ({spec_reg_same6}); 'the hinge's "
         f"analogue ... is untested' (tex6:441) still right ({new87_recorded} records of the label-free arms). No G result had "
         f"landed by the draft that it should mention. Session H's last commit at the same moment: `{hbr6}` (check_H_v6).")
L.append("")
L.append("## REMOVED: fifth-pass rows whose quoted fragment is not in the sixth draft")
L.append("")
L.append("Each sentence was rewritten in the sixth version; its replacement has a 'new' row above.")
L.append("")
L.append("| tex5 line | fragment(s) | item | fifth-pass status |")
L.append("|---|---|---|---|")
for n5_, nds, it, s_ in sorted(REMOVED5):
    L.append(f"| tex5:{n5_} | {esc(' / '.join(nds))} | {esc(it)} | {s_} |")
L.append("")
L.append("## REMOVED: fourth-pass rows whose quoted fragment is not in the sixth draft")
L.append("")
L.append("Rows marked 'newly' were still in the fifth draft; the others were rewritten in the fifth version (resolution_G_H5.md).")
L.append("")
L.append("| tex4 line | fragment(s) | item | fourth-pass status | gone |")
L.append("|---|---|---|---|---|")
for n4_, nds, it, s_ in sorted(REMOVED4):
    L.append(f"| tex4:{n4_} | {esc(' / '.join(nds))} | {esc(it)} | {s_} | {'newly (sixth version)' if (n4_, nds, it, s_) in newly_gone4 else 'fifth version'} |")
L.append("")
L.append("## REMOVED: third-pass rows whose quoted fragment is not in the sixth draft")
L.append("")
L.append("Rows marked 'newly' were still in the fifth draft; the others were already gone there (resolution_G4.md).")
L.append("")
L.append("| tex3 line | fragment(s) | item | third-pass status | gone |")
L.append("|---|---|---|---|---|")
for n3, nds, it, s_ in sorted(REMOVED3):
    L.append(f"| tex3:{n3} | {esc(' / '.join(nds))} | {esc(it)} | {s_} | {'newly (sixth version)' if (n3, nds, it, s_) in newly_gone3 else 'earlier'} |")
L.append("")
L.append("## REMOVED: second-pass rows whose quoted fragment is not in the sixth draft")
L.append("")
L.append("Rows marked 'newly' were still in the fifth draft; the others were already gone there.")
L.append("")
L.append("| tex2 line | fragment(s) | item | second-pass status | gone |")
L.append("|---|---|---|---|---|")
for n2, nds, it, s_ in sorted(REMOVED2):
    L.append(f"| tex2:{n2} | {esc(' / '.join(nds))} | {esc(it)} | {s_} | {'newly (sixth version)' if (n2, nds, it, s_) in newly_gone2 else 'earlier'} |")
L.append("")
L.append(f"V3-V5 revisions whose row is gone (their sentence was rewritten): {V3_GONE}.")
L.append("")
L.append(f"## REMOVED: first-pass rows whose quoted fragment is not in the sixth draft (newly: {len(newly_gone1)})")
L.append("")
L.append("| tex line | fragment(s) | item | first-pass status |")
L.append("|---|---|---|---|")
for n1, nds, it, s_ in sorted(REMOVED):
    L.append(f"| tex:{n1} | {esc(' / '.join(nds))} | {esc(it)} | {s_} |")
L.append("")

# ── S83
L.append("## S83 (report 6) against the records and the docstring")
L.append("")
L.append(f"- **Screen and readings.** `explore_g_s83.py` and `explore_g_s83_child.py`, added at `{S83SPEC}` ({spec83_t}); their history "
         f"on the branch: {hist83_code} (one commit, so unchanged since). The outcomes module explore_g_rhl_parent.py last changed at "
         f"{parent83_hist}; S82's explore_g_devices(_child).py at {devchild83_hist}; both before the spec. The child's file SHA-1 "
         f"{child_sha1[:12]} equals the store's sha_rows entry ({child_row[0][1][:12] if child_row else 'missing'}).")
L.append(f"- **Started** at `{S83START}` ({start83_t}; s83.log banner 'git b2f9907'). CHECKs: `{_ck[0]}`, {len(ck_rows)} rows ({ck_s82} "
         f"S82 + {len(ck_rows) - ck_s82} new), all ok: {ck_ok}; started {ck_t} UTC. Phase 0: s83_repro.json {repro83['result']} at "
         f"{repro83['time']}.")
L.append(f"- **First records.** The store first appears at `{rec83_hist[0][0]}` ({rec83_hist[0][1]}, {rec83_hist[0][2]} records); the "
         f"first records at `{first_rec83[0]}` ({first_rec83[1]}, {first_rec83[2]} records); so the readings (committed {spec83_t}) "
         f"come {'before' if spec83_t < first_rec83[1] else 'AFTER'} the first record. Commits with the store: "
         f"{', '.join(f'{h}:{n_}' for h, _, n_ in rec83_hist)}; s83.log: '82 runs queued on 4 workers' {q82}, 'S83 runs done in "
         f"{mins83} min'. The store's meta provenance ({meta83['git']}, started {meta83['started']}) is the second, empty phase "
         "(phase 2: 0 runs queued), which rewrote the meta; the runs carry code SHA "
         f"{integ83[4]}.")
L.append(f"- **Phase 2 (delta).** By the rule: {delta_arm83} (none of FLAT, POS, LOCAL3_KEYHINGE discovered); s83_delta.json: "
         f"arm {delta83['arm']} at {delta83['time']}; no _D arm in the store ({not any(a.endswith('_D') for a in Y)}).")
L.append(f"- **Validity.** REG3_NUDGE ROUTED-BOUND {val83[0]}/{val83[2]} (BOUND {val83[1]}/{val83[2]}, transitions {val83[3]}); S71's "
         f"ORACLE_ALL and S82's rerun stand as the docstring says (S82 ORACLE_ALL rerun BOUND {val82['oracle'][0]}/{val82['oracle'][1]}).")
L.append(f"- **Seeds.** New arms on 2100-2119 (REG3_NUDGE on 2100-2101), paired with S82's records; S82's store is byte-identical "
         f"at {G6}.")
L.append("")
L.append("Counts per arm (recomputed) against report_6.md's table:")
L.append("")
L.append("| arm | n | BOUND | ROUTED-BOUND | Wilson 95% | band | DISCOVERED | G's BOUND ROUTED | unbound classes (v1) | transitions (bound) | report 6 | same |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for key_, c_, got, rp, same in cmp83:
    L.append(f"| {key_} | {c_['n']} | {got[0]} | {got[1]} | {got[2]} | {got[3]} | {c_['disc']}/{c_['n']} | {c_['mine']}/{c_['n']} | {got[4]} | "
             f"{', '.join(map(str, c_['tr'])) or '–'} | {esc(' / '.join(rp[2:5])) if rp else 'not in table'} | "
             f"{'yes' if same else ('**no**' if same is False else '–')} |")
L.append("")
L.append("The one 'no': report 6's REG3 (S82) row lists its unbound classes as 'KEY 16' only; the records (and report 5) also have "
         "OTHER 1 and STREAM-PARTIAL 1. Every count of report 6's table reproduces; the draft does not use the class column.")
L.append("")
L.append("Paired comparisons (ROUTED-BOUND on 2100-2119; b = the first arm alone succeeds, c = the second; exact one-sided McNemar "
         "p = P(Bin(b + c, 1/2) >= b)):")
L.append("")
L.append("| arm | vs BASELINE (b, c, p) | REG3_KEYHINGE vs arm (b, c, p) | within 1 pair of KH | vs REG3 (b, c, p) | REG3 vs arm (b, c, p) |")
L.append("|---|---|---|---|---|---|")
for a in NEW83:
    t_ = P83[a]
    f3 = lambda t: f"{t[0]}, {t[1]}, {t[2]:.3g}"  # noqa: E731
    L.append(f"| {a} | {f3(t_['vsB'])} ({'discovered' if disc83[a] else 'not discovered'}) | {f3(t_['KHvs'])} | "
             f"{'yes' if within83[a] else 'no'} ({t_['KHvs'][0]} - {t_['KHvs'][1]}) | {f3(t_['vsR'])} | {f3(t_['Rvs'])} |")
L.append("")
L.append("Readings under explore_g_s83.py's rule (recomputed) against s83_summary.json, report 6 and the draft:")
L.append("")
L.append("| question | rule | recomputed | s83_summary.json | draft |")
L.append("|---|---|---|---|---|")
L.append(f"| the register | 'not needed' if LOCAL3_KEYHINGE discovered and within 1 pair; 'needed' if KH beats it with p <= {P4}; else 'neither' | "
         f"{read83['the register']} (p {P83['LOCAL3_KEYHINGE']['KHvs'][2]:.3g}) | {esc(sum83['the register'])} | {L5('the register is needed')}, {L5('the register and the identity term are both needed')} |")
L.append(f"| key identity | 'need not be read' if POS discovered and within 1 pair; 'the identity term is needed' if KH beats it with p <= {P4} | "
         f"{read83['key identity']} (p {P83['REG3_KEYHINGE_POS']['KHvs'][2]:.3g}) | {esc(sum83['key identity'])} | {L5('the identity term is needed')}, :59, :441 |")
L.append(f"| structure | 'no structure need be read' if FLAT discovered and within 1 pair | {read83['structure']} | {esc(sum83['structure'])} | "
         f"{L5('and the same term at every body position $0/20$')} (count only; the name not given) |")
L.append(f"| the window | 'the short window suffices' if KEYHINGE_1200 within 1 pair | {read83['the window']} ({P83['REG3_KEYHINGE_1200']['KHvs'][0]} - "
         f"{P83['REG3_KEYHINGE_1200']['KHvs'][1]} > 1) | {esc(sum83['the window'])} | {L5('one pair short of')} 'one pair short of ``the short window suffices''' |")
L.append(f"| phase 2 | the least-reading arm discovered on Hebbian memory | {delta_arm83} | {esc(sum83['delta arm (rule)'])} | Table 5's S83 row: delta '---' |")
L.append(f"| discovered | beats S82's BASELINE, p < 0.05 | {', '.join(f'{a} {disc83[a]}' for a in NEW83)} | as recomputed | "
         f"{L5('18 vs 0 against the baseline; one pair short')} (18 vs 0 for the 1200 window) |")
L.append("")
L.append("Descriptive quantities (report 6, 'What happened (exploratory)', against the store):")
L.append("")
L.append(f"- Term firings: {', '.join(f'{a} {med(fired83[a])} ({fired83[a][0]}-{fired83[a][-1]})' for a in NEW83)} (report 6: LOCAL3 1-15, "
         "POS/FLAT 2400, 1200 window 55-142).")
L.append(f"- LOCAL3_KEYHINGE's key term at update 1: median value {med(loc_v1):.3f} (the term is relu(eta^2 - 0.2)); eta^2 by key at update 1 "
         f"about {loc_eta1:.2f} where it fired. report_6.md's 'eta^2_key 0.18 at update 1' is the term's value, not eta^2. With the "
         f"register: {kh_eta1:.2f} (S82's hinge record) and {k12_v1:.2f} (S83's 1200 window). End margin {loc_margin_end[0]:.2f}-{loc_margin_end[1]:.2f}.")
L.append(f"- KL terms: max over runs at 300 / 600 / 1200 / 2400: POS {kl_max['REG3_KEYHINGE_POS']}, FLAT {kl_max['REG3_FLAT']} (median at 2400 "
         f"{kl_med2400['REG3_KEYHINGE_POS']:.4f} / {kl_med2400['REG3_FLAT']:.4f}). beta^R (CTX, KEY, VAL ranges) at 1200: POS "
         f"{ {k: (round(v[0], 3), round(v[1], 3)) for k, v in br83['REG3_KEYHINGE_POS'][1200].items()} }, FLAT "
         f"{ {k: (round(v[0], 3), round(v[1], 3)) for k, v in br83['REG3_FLAT'][1200].items()} }; at 2400: POS "
         f"{ {k: (round(v[0], 3), round(v[1], 3)) for k, v in br83['REG3_KEYHINGE_POS'][2400].items()} } (moved > 0.03 in {moved2400['REG3_KEYHINGE_POS']}), "
         f"FLAT { {k: (round(v[0], 3), round(v[1], 3)) for k, v in br83['REG3_FLAT'][2400].items()} }; at the end: POS "
         f"{ {k: (round(v[0], 2), round(v[1], 2)) for k, v in br83['REG3_KEYHINGE_POS'][None].items()} }, FLAT "
         f"{ {k: (round(v[0], 2), round(v[1], 2)) for k, v in br83['REG3_FLAT'][None].items()} }. Report 6 and the README scope the "
         "flat write 'through 1200 (FLAT through 2400)' / 'for their window'; the fifth draft now says 'through 1200, moving only "
         "afterwards'.")
L.append(f"- KL terms at every recorded update (max over layers per run; runs above 1e-5): " + "; ".join(
    f"{a} " + ", ".join(f"{u}: max {kl_mx[a][u]:.2g} ({len(kl_over[a][u])}/20{' ' + str(kl_over[a][u]) if 0 < len(kl_over[a][u]) < 5 else ''})"
                        for u in KLU) for a in POSFLAT) + f". The weight is 1 at update 2400 and 0 at 2401 in every POS/FLAT run "
         f"({w_on}), so both terms act through 2400. Report 6's 'their value is 0.0000 from update 300' holds to four decimals "
         f"through 1200 and for FLAT at 2400, not for POS at 2400 (max {kl_mx['REG3_KEYHINGE_POS']['2400']:.3f}).")
L.append(f"- The register's write (max over runs of max |beta^R - 0.10| over CTX, KEY, VAL): " + "; ".join(
    f"{a} " + ", ".join(f"{st}: {wdev_max[a][st]:.4f}" for st in (0, 1200, 2400)) +
    f" (> 0.03 at 2400 in {w_moved[a][2400]}/20)" for a in POSFLAT) + ".")
L.append(f"- REG3_KEYHINGE_1200: beta^R at 2400 median {k12_b2400['CTX']:.2f} / {k12_b2400['KEY']:.2f} / {k12_b2400['VAL']:.2f} (report 6: "
         f"0.32 / 0.18 / 0.21); failures {k12_fail}: (eta^2 by key at 1200, at 2400, class, first evaluation with eta^2 by key >= 0.5). "
         "Report 6 says the two 'key-split after the term stopped at 1200, while the stack was still at its slow rate'; in the "
         "records their eta^2 by key is still near 0 at 2400 and first reaches 0.5 at the step shown, after the slow phase "
         "(not in the draft).")
L.append(f"- POS's routed run: {pos_rb} (REG3 and NOWIN ROUTED-BOUND on it: {pos_rb_also}); report 6: 'POS's one routed run is 2108, "
         "which REG3 and NOWIN also route'.")
L.append(f"- The drafted main-line test (spec_draft_register.md at {G6}): arms and rate claim present: {spec_reg_arms}; records with seeds "
         f"600-619 / 1600-1619 on the {len(heads)} fetched branch heads ({', '.join(heads)}): {seed_hits or 'none'}.")
L.append(f"- The draft's commits that name S83 (a5f67a2..c66cc76): {', '.join(f'`{h}` ({t})' for h, t, _ in sorted(fold83, key=lambda x: x[1]))}. "
         f"The first, a469c1f, only put S83 into the index and provenance as running (the third pass's omission); the fold-in "
         f"(8bc1918 and the two after it) comes after report 6 ({rep6_t}): {all(t > rep6_t for h, t, _ in fold83 if h != 'a469c1f')}.")
L.append("")
# ── S82
L.append("## S82 (report 5) against the records and the docstring")
L.append("")
L.append(f"- Spec, readings and best-device rule: `{S82SPEC}` ({spec82_t}), adds {', '.join(spec_files)}; both files have one commit "
         f"on the branch ({', '.join(code82_hist)}), so they are unchanged since. explore_g_rhl_parent.py (the outcomes) last "
         f"changed at {parent82_hist}, before the spec.")
L.append(f"- First S82 record: `{first_rec82[0]}` ({first_rec82[1]}, {first_rec82[2]} records), {'after' if first_rec82[1] > spec82_t else 'BEFORE'} the spec.")
L.append(f"- Best device: `s82_best.json` added at `{best_c[0]}` ({best_c[1]}; file time '{best5['time']}'), unchanged since "
         f"({len(best_commits)} commit; content at add = content at {G5}: {best_unchanged}). The store at that commit holds {n_at_best} "
         f"records, {d_at_best} of them phase 2. First phase-2 record: `{first_d82[0]}` ({first_d82[1]}, {first_d82[3]} phase-2 records). "
         f"In s82.log the best-device line comes {'before' if 0 <= i_best < i_q32 else 'AFTER'} '32 runs queued'.")
L.append(f"- The rule recomputed from phase 1 (most ROUTED-BOUND, then fewer unbound KEY failures, then the order KEYHINGE, "
         f"SLOWGATE, NOWIN): {best_re['device']}, table {best_re['table']}; equals s82_best.json: {best_ok}.")
L.append(f"- Validity: ORACLE_ALL rerun BOUND {val82['oracle'][0]}/{val82['oracle'][1]} (VALID needs >= 1/2), REG3_NUDGE ROUTED-BOUND "
         f"{val82['nudge'][0]}/{val82['nudge'][1]}, ORACLE_ALL_D {val82['oracle_d'][0]}/{val82['oracle_d'][1]}.")
L.append(f"- Seeds 2100-2119: no other keyed store on the fetched G, F, H, I, E, R branches or the main line uses them "
         f"({(g_other + other_use) or 'none found'}).")
L.append("")
L.append("Counts per arm (recomputed) against report_5.md's table:")
L.append("")
L.append("| arm | n | BOUND | ROUTED-BOUND | DISCOVERED | G's BOUND ROUTED | unbound classes (v1) | transitions | report 5 | same |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for a, n, got, rp, same, tr in cmp82:
    L.append(f"| {a} | {n} | {got[0]} | {got[1]} | {got[2]} | {got[3]} | {got[4]} | {', '.join(map(str, tr)) if tr else '–'} | "
             f"{esc(' / '.join(rp[:4]) + '; ' + rp[4]) if rp else 'not in table'} | {'yes' if same else '**no**'} |")
L.append("")
L.append("Readings (ROUTED-BOUND, paired on 2100-2119; b = seeds where the first arm alone succeeds):")
L.append("")
L.append("| arm | discovered (vs BASELINE) | helps (vs REG3) | hurts (REG3 vs it) | a working write without the device's cost | report 5 |")
L.append("|---|---|---|---|---|---|")
_rep_read = {"REG3": "does not apply (2 / 0, p 0.25)", "REG3_KEYHINGE": "APPLIES / APPLIES / no / APPLIES",
             "REG3_SLOWGATE": "does not apply / no / does not apply / does not apply",
             "REG3_NOWIN": "APPLIES (threshold) / no / no / does not apply"}
for d in ("REG3",) + DEV:
    o_ = RD82[d]
    f_ = lambda t: f"{'APPLIES' if t[0] else 'does not apply'} ({t[1][0]} vs {t[1][1]}, p {t[1][2]:.2g})"  # noqa: E731
    if d == "REG3":
        L.append(f"| {d} | {f_(o_['discovered'])} | – | – | (CTX > K, V in {nwrite(R82)}/20) | {_rep_read[d]} |")
    else:
        w_ = o_["working write"]
        L.append(f"| {d} | {f_(o_['discovered'])} | {f_(o_['helps'])} | {f_(o_['hurts'])} | {'APPLIES' if w_[0] else 'does not apply'} "
                 f"(KEY {w_[1][0]}; CTX > K, V {w_[1][1]}/{w_[1][2]}) | {_rep_read[d]} |")
L.append("")
L.append(f"Phase 2 (delta, 2100-2109; descriptive by the docstring, explore_g_devices.py:40): REG3_KEYHINGE_D ROUTED-BOUND {cnt(KD, rb)}/10 "
         f"(transitions {dict(sorted(Counter(r['transition'] for r in KD.values()).items()))}); vs BASELINE_D {ph2['BASELINE_D'][:2]} "
         f"p = {ph2['BASELINE_D'][2]:.3g}; vs REG3_D {ph2['REG3_D'][:2]} p = {ph2['REG3_D'][2]:.3g}; beta^R CTX > K, V {nwrite(KD)}/10. "
         "The draft's 'delta 10/10' holds; the references are ROUTED-BOUND 0/10 and 2/10 (BOUND 2/10 and 4/10).")
L.append("")
L.append(f"Descriptive quantities (REG3_KEYHINGE, median over runs): eta^2 by key (mean over layers) {', '.join(f'{u} {v:.3f}' for u, v in eta_u.items())}; "
         f"hinge firings {med(fired):.0f} ({min(fired)}-{max(fired)}); beta^R CTX/KEY/VAL at 1200 {kb1200['CTX']:.2f}/{kb1200['KEY']:.2f}/"
         f"{kb1200['VAL']:.2f}, 2400 {kb2400['CTX']:.2f}/{kb2400['KEY']:.2f}/{kb2400['VAL']:.2f}, end {kbend['CTX']:.2f}/{kbend['KEY']:.2f}/"
         f"{kbend['VAL']:.2f}; transitions {dict(sorted(trc.items()))}; runs stop at transition + {sorted(gap82)}. REG3_SLOWGATE: "
         f"eta^2 by key >= 0.5 at 1200 in {sg_split1200}/20 (REG3 {r3_split1200}/20); end beta^R {sgb['CTX']:.2f}/{sgb['KEY']:.2f}/{sgb['VAL']:.2f}. "
         f"REG3_NOWIN: end beta^R {nwb['CTX']:.2f}/{nwb['KEY']:.2f}/{nwb['VAL']:.2f}; routed seeds {nw_rb} (REG3's {r3_rb}).")
L.append("")
L.append("What the hinge reads (explore_g_devices.py:17-18 at 83699a2): 'It reads key positions and key identities (token types), as "
         "the key split reads key positions; it reads no stream label.' The fifth draft says so at "
         f"{L5('a training-time prior that reads the task' + chr(39) + 's key positions and identities')}, "
         f"{L5('The hinge reads the task' + chr(39) + 's key positions and key identities')}, {L5('reads key positions and key identities, and session~G')} "
         f"and {L5('a solution without stream labels')}.")
L.append("")
L.append(f"Marked as descriptive where it is not a pre-fixed reading? At {L5('Most of the runs that fail to discover')}: 'formed in the records' "
         "(yes); the delta phase 'descriptive by the screen's docstring' (yes); the slow gate's 'which we read as' (yes); 'What the "
         "firing records show, as a description' (yes). Not marked: 'with a register that writes at nearly every token' (the "
         f"records agree). At {L5('Session~G then screened')} (S83): 'on our reading of G's traces' (yes); 'where, on session~G's "
         "account, the ratio term removes only the share ...' (yes, since fd2fa89); 'both hold the gate at uniform while they act "
         "(its KL from uniform at most 10^-5 through update 1200 in every run)' (a record statement; see its row: the bound is "
         "exceeded); 'fails, by clamping the gate' (report 6's account, stated without a marker; the clamp itself is in the records).")
L.append("")
# ── readings vs docstring (fourth draft)
L.append("## Readings vs docstring (fifth draft)")
L.append("")
L.append("| screen | reading (docstring) | applies (recomputed) | fifth draft | wording |")
L.append("|---|---|---|---|---|")
r1h = cnt(A["REG_NUDGE"], br56)
r1d = cnt(nd, br56)
lab = lambda k: "the register carries the cue" if k >= 9 else "architecture insufficient" if k < 6 else "neither"  # noqa: E731
_rd = [
    ("S55", "'absent'", "absent", "tex4:261 ''absent'' applied", "same"),
    ("S56 R1", "'the register carries the cue' / 'architecture insufficient' / 'neither'", f"Hebbian {r1h}/10 '{lab(r1h)}'; delta {r1d}/10 '{lab(r1d)}'",
     "tex4:263 'the reading ``architecture insufficient'' applied'", "Hebbian: same; delta 'neither' not named (not asked)"),
    ("S56 R2", "'discovered' (BOUND >= BASELINE + 3, KEY failures <= 2)", f"Hebbian 'not discovered'; delta REG_D {cnt(rd, bound)} vs {cnt(bd, bound)}, "
     f"KEY {_r2key}: 'discovered'", "tex4:263 'met the screen's own rule for ``discovered'' on delta ... 4 vs 0 by McNemar, which was not its rule'",
     "same"),
    ("S71", "'layer 1 is the block' / 'alone suffices' / 'delta tolerates'", "all three apply", "tex4:263 verbatim", "same"),
    ("S72 R1", "'the register carries the cue at all three layers'", f"{cnt(R3N, rb)}/20 applies", "tex4:265 heading; tex4:59", "paraphrase"),
    ("S72 R2", "'discovered' / 'not discovered'", f"{paired(R3, BH, rb)[:2]}: 'not discovered'", "tex4:265 'S72's pre-fixed reading is ``not discovered''",
     "same (resolved in the third version)"),
    ("S73", "'discovered (register)'", f"{paired(R3D, BD, rb)[:2]}: does not apply", "tex4:265 'S73's ``discovered (register)'' does not apply'", "same"),
    ("S73", "'the delta stack carries the cue' (>= 10/20)", f"{sum(1 for r in BD.values() if min(pk(r)) >= 0.9)}/20: does not apply",
     "tex4:265 'did not apply either ... (the baseline 7/20)'", "same"),
    ("S73", "'the gate is not needed on this task' (>= 8/10)", f"{cnt(SGL, bound)}/10: does not apply", "tex4:265 'did not apply ... (one channel 7/10)'",
     "same"),
    ("S73", "'the latch holds on delta memory'", "9/10 applies", "tex4:265 'holds an exact latch when nudged (9/10 routed ...)'", "not named"),
    ("S82", "'discovered'", f"KEYHINGE {RD82['REG3_KEYHINGE']['discovered'][1][:2]} applies; NOWIN {RD82['REG3_NOWIN']['discovered'][1][:2]} applies; "
     f"SLOWGATE {RD82['REG3_SLOWGATE']['discovered'][1][:2]} no; REG3 {RD82['REG3']['discovered'][1][:2]} no",
     "tex4:267 'discovered' for KEYHINGE and for NOWIN 'exactly at its threshold (5 vs 0)'", "same"),
    ("S82", "'the device helps'", f"KEYHINGE {RD82['REG3_KEYHINGE']['helps'][1][:2]} applies; NOWIN {RD82['REG3_NOWIN']['helps'][1][:2]} no; "
     f"SLOWGATE {RD82['REG3_SLOWGATE']['helps'][1][:2]} no", "tex4:267 '``helps'' ... applies' (KEYHINGE)", "same; NOWIN's 'no' not stated"),
    ("S82", "'the device hurts'", f"none applies (SLOWGATE: REG3 only {RD82['REG3_SLOWGATE']['hurts'][1][0]}, SLOWGATE only "
     f"{RD82['REG3_SLOWGATE']['hurts'][1][1]}, p {RD82['REG3_SLOWGATE']['hurts'][1][2]:.2f})",
     "tex4:267 '``hurts'' did not apply, 0 vs 2 against the plain register'", "same (resolved in the third version)"),
    ("S82", "'a working write without the device's cost'", f"KEYHINGE applies {RD82['REG3_KEYHINGE']['working write'][1][:2]}; others no",
     "tex4:267 'no key failures and beta^R at the cue above keys and values in every run'", "content stated, name not"),
    ("S82", "phase 2 (descriptive)", f"KEYHINGE_D {cnt(KD, rb)}/10 vs {cnt(BD82, rb)}/10 and {cnt(RDD82, rb)}/10", "tex4:267 'descriptive by the "
     "screen's docstring'", "same (resolved in the third version)"),
    ("S83", "the register", f"{read83['the register']} (KH only {P83['LOCAL3_KEYHINGE']['KHvs'][0]}, LOCAL3 only {P83['LOCAL3_KEYHINGE']['KHvs'][1]})",
     "tex4:269 'the register is needed', tex4:59", "same"),
    ("S83", "key identity", f"{read83['key identity']} (KH only {P83['REG3_KEYHINGE_POS']['KHvs'][0]}, POS only {P83['REG3_KEYHINGE_POS']['KHvs'][1]})",
     "tex4:269 'the identity term is needed, 19 vs 0', tex4:59, tex4:439", "same; tex5:441 now 'its KL terms ... fail' and 'S83's one position-only term could not replace'"),
    ("S83", "structure", read83["structure"], "tex4:269 'the same term at every body position 0/20'", "count only; name not given"),
    ("S83", "the window", f"{read83['the window']} (KH only {P83['REG3_KEYHINGE_1200']['KHvs'][0]}, 1200 only {P83['REG3_KEYHINGE_1200']['KHvs'][1]})",
     "tex4:269 'one pair short of the short window suffices'", "same"),
    ("S83", "discovered (each arm)", f"{ {a: disc83[a] for a in NEW83} }", "tex4:269 '18 vs 0 against the baseline' (1200); POS 'binds 1/20' (not discovered, not said)", "1200: same"),
    ("S83", "phase 2 rule", f"{delta_arm83}: no delta runs", "Table 5 tex4:307 delta '---'", "same, implicit"),
    ("S82", "best-device rule", f"{best_re['device']} {best_re['table']}", "tex4:267 (implicit: the hinge is the device run on delta)", "same"),
]
for r_ in _rd:
    r_ = r_[:3] + (t45(r_[3]),) + r_[4:]
    L.append("| " + " | ".join(esc(x) for x in r_) + " |")
L.append("")

# ── register gate
assert TEX5[124] == TEX4[124]
L.append("## Register gate description (tex:125 -> tex2:125 = tex3:125 = tex4:125 = tex5:125, unchanged in the fifth version) vs code")
L.append("")
L.append("| item | draft | code | status | note |")
L.append("|---|---|---|---|---|")
for it, d_, c_, s_, nt in reg_rows:
    if it in ("initialisation", "the earlier design"):
        it = it + " (first draft)"
    L.append(f"| {esc(it)} | {esc(d_)} | {esc(c_)} | **{s_}** | {esc(nt)} |")
for it, d_, c_, s_, nt in reg_rows2:
    L.append(f"| {esc(it)} (second to fifth draft) | {esc(d_)} | {esc(c_)} | **{s_}** | {esc(nt)} |")
L.append("")

# ── S82 at the first draft's commit (for A2)
L.append("## S82 at the first draft's commit (for the resolution of A2)")
L.append("")
L.append(f"At {DRAFT} ({draft_t}) the last G commit was `{gdraft}`: {len(st_d['runs'])} S82 records, phase 1 "
         f"{'complete (104/104)' if ph1 else 'incomplete'}, s82_best.json {best5['device']} 20/20 at {best_c[0]} ({best_c[1]}). "
         f"Paired on 2100-2119 at that commit: REG3 vs BASELINE {reg82[0]} vs {reg82[1]}; REG3_KEYHINGE vs BASELINE {kh82[0]} vs {kh82[1]}; "
         f"vs REG3 {kr82[0]} vs {kr82[1]}. The second draft reports S82 from report 5 at {G5} ({ctime(G5)}), after the first draft.")
L.append("")

# ── descriptive statements, fourth draft
L.append("## Descriptive statements in the fifth draft: marked or not (session G)")
L.append("")
DESC5 = [
    ("(where, on our reading, the decayed sum of the block still carries the header)", "yes", "unchanged since the fourth version"),
    ("(our reading: the gate reads only the window, so nothing asks the stack to keep it)", "yes", "unchanged"),
    ("as the routing study would predict \\cite{routing}, is our reading of it", "yes", "unchanged"),
    ("(the lower of the two runs without a clean latch)", "—", f"right: {low_nl}, acc {racc[low_nl]:.3f}"),
    ("Most of the runs that fail to discover end in a key split (28 of 32 in S72 and S73), formed in the records", "yes ('in the records')", "28/32; order 28/28"),
    ("the write at the cue separating in only 1 of 20, which we read as", "yes", "1/20 (2113)"),
    ("with a register that writes at nearly every token", "no", "the records agree (unchanged)"),
    ("each S82 reference arm reproduced bit for bit through 2400 at one seed first", "—", "resolved: one seed (2100) per reference arm"),
    ("on our reading of G's traces, the hinge has nothing to release there", "yes", "firings 1-15; term 0 from 300"),
    ("both hold the gate at uniform while they act (its KL from uniform at most $10^{-5}$ through update 1200 in every run)", "— (a record statement)",
     f"the bound is exceeded: POS at 1200 up to {kl_mx['REG3_KEYHINGE_POS']['1200']:.1e} ({len(kl_over['REG3_KEYHINGE_POS']['1200'])} runs), every run "
     f"at update 1; POS's term acts to 2400, where its KL reaches {kl_mx['REG3_KEYHINGE_POS']['2400']:.3f}"),
    ("and the register's write stays at its initial value through 1200, moving only afterwards", "—", "resolved: within 0.012 through 1200"),
    ("where, on session~G's account, the ratio term removes only the share", "yes (fd2fa89)", "resolved: report 6's exploratory account, now marked"),
    ("and the one form tried of each device that reads less fails, by clamping the gate", "'by clamping': no",
     "report 6's account; the clamp is in the records. 'which S83's specification had allowed in advance': the spec allowed a null, and expected the opposite mechanism"),
    ("its KL terms reading positions only, or nothing, fail ($1/20$, $0/20$) by clamping the gate, a weakness its specification had allowed to the form", "'by clamping': no",
     "scope resolved; the same 'allowed' problem"),
    ("the hinge's analogue, which S83's one position-only term could not replace", "—", "resolved"),
    ("it is the register test proposed to follow \\texttt{test\\_delta\\_recipe}", "—", "no record of the order (tex5:476 keeps it as the plan)"),
    ("Each is marked where it appears", "—", "true of every place the referee listed; the ratio-term clause now marked"),
    ("is found label-free on a fifth of seeds (2 of 20 on the fresh seeds of the device screen)", "count", "resolved"),
    ("every session's queue is empty", "—", "not a record (G: no commit after report 6)"),
]
L.append("| tex5 line | statement | marked? | note |")
L.append("|---|---|---|---|")
for s_, m_, nt in DESC5:
    L.append(f"| {L5(s_)} | {esc(s_)} | {esc(m_)} | {esc(nt)} |")
L.append("")
L.append("Not checkable from the records: 'proposed to follow test_delta_recipe' for the register test (no specification is "
         f"committed on the main line; specs/ at {DRAFT5}: {', '.join(specs5)}); 'every session's queue is empty' (G's half: no "
         "queue file, no commit after report 6); the bit-for-bit phase-0 reproduction and S83's CHECKs "
         "(their files and log are read, the comparisons are not rerun); S83's code SHA 24268c2446f8 (not recomputed; the child "
         "file's SHA-1 is); the reasoning attributed to Revision 8's specification (as in the earlier passes); S82's CHECKs and "
         "wall-clock; the S78, S85 and S86 halves of tex5:230, :441, :443 and :472 (session H's; see check_H_v5); session I's S75 "
         "results (tex5:456-457; only the arm's name, 'session G's register gate', is checked here).")
L.append("")
with open(OUT, "w") as fh:
    fh.write("\n".join(L))
print(f"wrote {OUT}: {len(ROWS)} rows, {dict(C)}; " + "; ".join(f"{k} {dict(v)}" for k, v in CK.items()) +
      f"; removed1 {len(REMOVED)} (newly {len(newly_gone1)}); removed2 {len(REMOVED2)} (newly {len(newly_gone2)}); "
      f"removed3 {len(REMOVED3)} (newly {len(newly_gone3)}); removed4 {len(REMOVED4)}; gone revisions {V3_GONE}")
