#!/usr/bin/env python3
"""check_rev10_HE_v101.py -- Referee V's check_rev10_HE re-run against Revision 10.1 (846d0c8:multichannel_hebbian_report_v10.tex,
596 lines, cited tex101:NNN). The original, check_rev10_HE.py, read Revision 10 (585b666, 591 lines, tex10:NNN). Scope as
the original: Session H's round 5 (S87a, S87c, the one-to-one recount; claude/explore-H at 347bd25) and Session E's
batches 20 and 21 (claude/outside-ideas at 419368a), recomputed from the records, with the order of readings and records;
plus every line on them that 846d0c8 changed or added (rows marked new).

    python3 docs/review_2026-10/rev10/check_rev10_HE_v101.py

What changed against the original:
  * The tex is read at 846d0c8. Every row is anchored on its tex10 line at 585b666 (as the original asserted) and on the
    line it became at 846d0c8 (difflib's line map; asserted). A fragment 846d0c8 rewrote is re-anchored on its new
    wording (new_frag) and judged against it; a sentence it deleted is reported REMOVED, with a new row for its
    replacement.
  * Branches are pinned to what the original read (its run, 19 Oct before 18:01:47 UTC, from the reflog of this clone's
    remote-tracking refs): explore-H 347bd25, outside-ideas 419368a, the main line 7c4ceb8 (the prompts file), and the 13
    remote branches the seed grep covers (BRANCH_PINS).
  * Every record recomputation is kept as it was. Each row carries the original's status (check_rev10_HE.md at 86c6aa8,
    rows H1-H16 and E1-E22; given per row as old=, the counts asserted) and the status against 846d0c8.

Stdlib plus docs/review_2026-10/rlib.py. Reads git objects only (git show / git log / git grep); trains nothing, runs no
session code, writes nothing (prints). Every draft number is located in the draft by building its text from the
recomputed value; no draft number is an input. Rows end in a table: MATCH / PARTIAL / DISAGREE / NOT RECORD-CHECKABLE.
"""
import datetime
import difflib
import hashlib
import json
import re
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import rlib  # noqa: E402

H = "347bd25"            # claude/explore-H, report 9
H_READ = "f4fd535"       # S87a / S87c code and readings
H_ONE = "5b5deec"        # the one-to-one recount
H_S86 = "9b72bf6"        # S86 store (report 8)
E = "419368a"            # claude/outside-ideas, batch 21 complete
E20_READ, E20_RUN = "4acf915", "0fa79a3"
E21_READ = "b93b73c"
E19 = "e3f5809"          # S67's store (batch 19)
MAINL = "7c4ceb8"          # origin/claude/bdh-growth-hebbian-inference-w90069 as the original read it
REV10 = "585b666"
V101 = "846d0c8"
# the 13 remote branches at the original's run (reflog, 19 Oct 18:01:40 UTC)
BRANCH_PINS = {"origin/claude/bdh-growth-hebbian-inference-w90069": "7c4ceb8", "origin/claude/bdh-repo-curl-obgdqk": "5b82707",
               "origin/claude/explore-F": "a199d5e", "origin/claude/explore-G": "e46799a", "origin/claude/explore-H": "347bd25",
               "origin/claude/explore-I": "8d2e89f", "origin/claude/outside-ideas": "419368a", "origin/claude/relaxed-edison-he00cc": "3c2fb70",
               "origin/claude/replicate-R": "31c39fc", "origin/claude/rev10-draft": "f342d1d", "origin/claude/rev9-draft": "010a085",
               "origin/claude/review-V": "6ab7649", "origin/feature/per-channel-plasticity": "4af9d50"}
TEXP = "multichannel_hebbian_report_v10.tex"
ROWS = []


def git(*a):
    # core.abbrev=7: %h and --short print 7 hex digits however large the object store grows
    return subprocess.check_output(("git", "-C", rlib.ROOT, "-c", "core.abbrev=7") + a).decode()


def show(ref, path):
    return git("show", f"{ref}:{path}")


def blob(ref, path):
    return git("rev-parse", f"{ref}:{path}").strip()


def ctime(ref):
    return git("log", "-1", "--format=%ad", "--date=iso-strict", ref).strip()


def hist(ref, path):
    out = git("log", "--format=%h %ad", "--date=iso-strict", ref, "--", path).split("\n")
    return [tuple(x.split(" ", 1)) for x in out if x.strip()]


def first_nonempty(ref, path):
    """Oldest commit at which the store holds at least one non-dry run."""
    for h, t in reversed(hist(ref, path)):
        st = json.loads(show(h, path))
        n = sum(1 for r in st.get("runs", {}).values() if not r.get("dry"))
        if n:
            return h, t, n
    return None


def say(s=""):
    print(s)


OLD = show(REV10, TEXP).split("\n")
TEX = show(V101, TEXP).split("\n")
_ops = difflib.SequenceMatcher(None, OLD, TEX, autojunk=False).get_opcodes()
LMAP = {i1 + k + 1: j1 + k + 1 for op, i1, i2, j1, j2 in _ops if op in ("equal", "replace") and i2 - i1 == j2 - j1 for k in range(i2 - i1)}
LMAP.update({44: 44, 536: 541, 537: 542, 538: 543})     # the two unequal blocks: 44 -> 44-46, 536-538 -> 540-543
CHANGED = {i + 1 for op, i1, i2, _, _ in _ops if op != "equal" for i in range(i1, i2)}


def tl(n):
    return OLD[n - 1]


def find(n, frag):
    """Assert the fragment occurs on tex10 line n (the draft's wording at 585b666); returns it."""
    if frag not in tl(n):
        raise SystemExit(f"anchor not found on tex10:{n}: {frag!r}")
    return frag


def find101(n, frag):
    if frag not in TEX[n - 1]:
        raise SystemExit(f"anchor not found on tex101:{n}: {frag!r}")
    return frag


def num(n):
    return f"{n:,}".replace(",", "{,}")


# the original's statuses, row by row in its order (check_rev10_HE.md: H8 H9 H10 H4 H11 H12 H13 H7 H14 H5 H1 H2 H3 H15 H6 H16,
# E8 E5 E9 E10 E11 E12 E13 E14 E15 E16 E2 E17 E1 E18 E19 E3 E20 E21 E7 E22 E6 E6 E4)
_M, _P, _D, _N = "MATCH", "PARTIAL", "DISAGREE", "NOT RECORD-CHECKABLE"
OLDSEQ = [_M, _M, _M, _P, _M, _M, _M, _N, _M, _P, _D, _D, _D, _M, _P, _M,
          _M, _P, _M, _M, _M, _M, _M, _M, _M, _M, _P, _M, _P, _M, _M, _P, _M, _M, _N, _M, _P, _P, _P]
assert (len(OLDSEQ), OLDSEQ.count(_M), OLDSEQ.count(_P), OLDSEQ.count(_D), OLDSEQ.count(_N)) == (39, 24, 10, 3, 2)
N_ORIG = [0]


def row(line, item, frag, rec, status, note="", new_frag=None, new_line=None, removed=None, is_new=False):
    """line/frag: the original's tex10 anchor (asserted at 585b666). new_frag: its 846d0c8 wording, asserted on new_line (default:
    the line tex10:line became). removed: (tex101 line, fragment) of the replacement when 846d0c8 deleted the sentence.
    is_new: a row new here; then line/frag are tex101's."""
    assert status in ("MATCH", "PARTIAL", "DISAGREE", "NOT RECORD-CHECKABLE", "REMOVED")
    old = None
    if not is_new:
        old = OLDSEQ[N_ORIG[0]]
        N_ORIG[0] += 1
    if old is None:                                   # new row: anchored on 846d0c8 only
        find101(line, frag)
        ROWS.append((f"tex101:{line}", item, frag, rec, status, note, "\u2014", f"new: {status}"))
        return
    if frag is not None:
        find(line, frag)
    if removed is not None:
        assert all(frag not in t for t in TEX), frag
        find101(*removed)
        ROWS.append((f"tex10:{line} REMOVED (replaced at tex101:{removed[0]})", item, frag, rec, status, note, old, f"{old} \u2192 REMOVED"))
        return
    n = new_line or LMAP[line]
    f = frag if new_frag is None else new_frag
    if f is not None:
        find101(n, f)
    delta = f"{old} (unchanged)" if old == status else f"{old} \u2192 {status}"
    if new_frag is not None:
        delta += f"; re-anchored (tex10:{line} rewritten)"
    elif line in CHANGED:
        delta += f"; line changed elsewhere (tex10:{line}), fragment intact"
    ROWS.append((f"tex101:{n}", item, f, rec, status, note, old, delta))


def mc1(b, c):
    return rlib.mcnemar1(b, c)


# ══ H ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
say("=" * 110)
say("SESSION H, ROUND 5 (claude/explore-H at 347bd25)")
say("=" * 110)
hh = BRANCH_PINS["origin/claude/explore-H"]
say(f"branch head as the original read it (pinned): {hh} (== {H}: {hh == H})")

# ── H0. order ──
say("\n== H0. Readings before records")
code_files = ["explore_h_allmass2.py", "explore_h_dose.py", "explore_h_dose_child.py"]
for f in code_files:
    say(f"  {f}: commits {hist(H, f)}; blob f4fd535 == head: {blob(H_READ, f) == blob(H, f)}")
t_read = ctime(H_READ)
say(f"  readings commit {H_READ} at {t_read}")
fr = {}
for st in ("explore_out/H/allmass2_results.json", "explore_out/H/dose_results.json"):
    fr[st] = first_nonempty(H, st)
    say(f"  {st}: first commit with runs {fr[st]}; every commit of it: {len(hist(H, st))}, oldest {hist(H, st)[-1]}")
chg = git("diff", "--name-only", H_READ, H).split()
say(f"  files changed f4fd535 -> 347bd25: {chg}")
code_touched = [f for f in chg if f.endswith(".py")]
say(f"  .py files changed after the readings: {code_touched}")
rep_diff = git("diff", H_READ, H, "--", "explore_h_report.py")
say(f"  explore_h_report.py diff after the readings: +{sum(1 for l in rep_diff.split(chr(10)) if l.startswith('+') and not l.startswith('+++'))} lines "
    f"(report writer only; imported by no child: {'explore_h_report' not in show(H, 'explore_h_dose_child.py') + show(H, 'explore_h_allmass_child.py')})")
ck = json.loads(show(H, "explore_out/H/checks.json"))
say(f"  checks.json: H/allmass2 {ck['H/allmass2']['time']} git {ck['H/allmass2']['git']}; H/dose {ck['H/dose']['time']} git {ck['H/dose']['git']}")
ck0 = json.loads(show(H_READ, "explore_out/H/checks.json"))
tracked = "explore_out/H/checks.json" in git("ls-tree", "-r", "--name-only", H_READ, "explore_out/H/").split()
say(f"  checks.json tracked at f4fd535: {tracked}; holds an H/allmass2 entry there: {'H/allmass2' in ck0}; the commit that adds both entries: "
    f"{hist(H, 'explore_out/H/checks.json')[0]}")
log = show(H, "explore_out/H/s87.log").split("\n")
say(f"  s87.log banner/git lines: {[l.strip()[:90] for l in log[:8] if 'git' in l]}")
i_all = next(i for i, l in enumerate(log) if "all CHECKs of H/allmass2 passed" in l)
i_dose = next(i for i, l in enumerate(log) if "all CHECKs of H/dose passed" in l)
say(f"  one invocation: allmass2 CHECKs end at log line {i_all + 1}, dose CHECKs end at {i_dose + 1}, before '84 runs queued': "
    f"{any('84 runs queued' in l for l in log[i_dose:i_dose + 3])}")
say("  explore_h_run.main writes checks.json (tracked) after each screen's CHECK pass; ec.git_head() marks '+dirty' on tracked changes "
    "(git status --porcelain --untracked-files=no). So the dose CHECK's 'f4fd535+dirty' (23:00:36) follows from allmass2's checks.json "
    "write (22:57:03), committed only at b17517a (23:00:50).")
# code SHAs recomputed from committed sources
SERVED = ("test_binding_capacity", "test_curriculum_confirm", "test_short_conv", "test_scale_axes", "test_slow_start",
          "test_stream_channels", "test_stream_curriculum", "test_stream_recipe")
MAIN9C = "9c5939ef8ca8db079c87beb489b7c603169c9c57"


def code_sha(ref, mods, main):
    rows = []
    for m in sorted(mods):
        if main and m in SERVED:
            src, org = show(MAIN9C, m + ".py"), "main@" + MAIN9C[:7]
        else:
            src, org = show(ref, m + ".py"), "branch"
        rows.append([m, hashlib.sha1(src.encode()).hexdigest(), org])
    return hashlib.sha1(json.dumps(rows).encode()).hexdigest()


def mods_from_log(lines, tag):
    l = next(x for x in lines if tag in x)
    return l.rsplit("): ", 1)[1].split()


am_mods = mods_from_log(log, "code SHA H/allmass2")
do_mods = mods_from_log(log, "code SHA H/dose")
sha_am = code_sha(H_READ, am_mods, True)
sha_do = code_sha(H_READ, do_mods, False)
st_am = rlib.load(H, "explore_out/H/allmass2_results.json")
st_do = rlib.load(H, "explore_out/H/dose_results.json")
say(f"  code SHA recomputed from f4fd535's sources: allmass2 {sha_am[:12]} (store {st_am['meta']['code_shas']}); dose {sha_do[:12]} (store {st_do['meta']['code_shas']})")
H_SHA_OK = sha_am == st_am["meta"]["code_shas"][0] and sha_do == st_do["meta"]["code_shas"][0]
gits = sorted({r.get("git") for st in (st_am, st_do) for r in st["runs"].values()})
clean = [g for g in gits if not g.endswith("+dirty")]
say(f"  records' git values: {len(gits)} distinct, {len(clean)} clean; per-record code SHAs {sorted({r['code_sha'][:12] for st in (st_am, st_do) for r in st['runs'].values()})}")
between = git("log", "--format=%h", f"{H_READ}..{H}").split()
touch_child = [h for h in between if any(f in git("show", "--name-only", "--format=", h).split()
                                         for f in ["explore_h_allmass_child.py", "explore_h_dose_child.py"] + [m + ".py" for m in am_mods + do_mods])]
say(f"  commits after f4fd535 touching any run module: {touch_child}")
say(f"  store provenance: allmass2 {st_am['meta']['provenance']['git']} started {st_am['meta']['provenance']['started']}; dose {st_do['meta']['provenance']['git']}")
H_ORDER = (all(len(hist(H, f)) == 1 and hist(H, f)[0][0] == H_READ for f in code_files)
           and all(v[1] > t_read for v in fr.values()) and not touch_child and H_SHA_OK)
say(f"  ORDER VERDICT (readings committed before any record; no run module changed after): {H_ORDER}")

# seeds
say("\n== H0b. Seeds 2600-2659")
hits = set()
for b in BRANCH_PINS.values():
    try:
        o = git("grep", "-l", "-E", r'\|26[0-5][0-9]\||"seed": 26[0-5][0-9][,}]', b, "--", "explore_out", "results", "specs")
    except subprocess.CalledProcessError:
        o = ""
    hits |= {x.split(":", 1)[1] if ":" in x else x for x in o.split() if x}
say(f"  record files holding a seed 2600-2659 on any branch: {sorted(hits)}")
pr = show(MAINL, "docs/reading/parallel_sessions_prompts.md")
say(f"  the user's round-5 prompt assigns H 2600-2659: {'H 2600-2659' in pr}")
H_SEEDS = sorted(hits) == ["explore_out/H/allmass2_results.json", "explore_out/H/dose_results.json"]

# ── H1. S87a ──
say("\n== H1. S87a (allmass2)")
A = {a: rlib.runs(st_am, a) for a in ("BASE", "ALLMASS", "ORACLE")}
SEEDS = list(range(2600, 2620))
mism = [(a, s) for a in ("BASE", "ALLMASS") for s, r in A[a].items() if (r["outcome"] == "BOUND ROUTED") != rlib.bound_routed(r)]
cls_mism = [(a, s, r["outcome"], rlib.outcome_k(r)) for a in ("BASE", "ALLMASS") for s, r in A[a].items() if r["outcome"] != rlib.outcome_k(r)]
say(f"  runs {len(st_am['runs'])}; per arm {[(a, len(v), min(v), max(v)) for a, v in A.items()]}; not ok {[k for k, r in st_am['runs'].items() if not r.get('ok')]}")
say(f"  BOUND ROUTED recorded vs rlib mismatches {mism}; class (outcome_k) mismatches {cls_mism}; outcome != outcome_v2: "
    f"{[(a, s) for a in ('BASE', 'ALLMASS') for s, r in A[a].items() if r['outcome'] != r['outcome_v2']]}")
br = {a: {s: rlib.bound_routed(A[a][s]) for s in SEEDS} for a in ("BASE", "ALLMASS")}
kB, kA = sum(br["BASE"].values()), sum(br["ALLMASS"].values())
aon, bon, _ = rlib.pairs(A["ALLMASS"], A["BASE"], rlib.bound_routed, SEEDS)
orc = {s: (rlib.bound(r), rlib.transition(r["curve"])) for s, r in A["ORACLE"].items()}
valid = len(orc) == 2 and all(b for b, _ in orc.values())
say(f"  ORACLE {orc} -> {'VALID' if valid else 'UNTESTED'}")
say(f"  BASE {kB}/20, ALLMASS {kA}/20; ALLMASS only {aon}, BASE only {bon}; one-sided p ALLMASS>BASE {mc1(aon, bon):.4g}, BASE>ALLMASS {mc1(bon, aon):.4g}")
say(f"  BASE-only seeds {[s for s in SEEDS if br['BASE'][s] and not br['ALLMASS'][s]]}; ALLMASS-only {[s for s in SEEDS if br['ALLMASS'][s] and not br['BASE'][s]]}; "
    f"neither {[s for s in SEEDS if not br['ALLMASS'][s] and not br['BASE'][s]]}")
for a in ("BASE", "ALLMASS"):
    tr = sorted(rlib.transition(A[a][s]["curve"]) for s in SEEDS if br[a][s])
    say(f"  {a}: transitions median {rlib.med(tr)} {tr}; failures {[(s, rlib.outcome_k(A[a][s])) for s in SEEDS if not br[a][s]]}")


def lab(mb, cs, c0):
    return mb.count(cs) >= 2 and mb.count(c0) == 0


FI = {}
for a in ("BASE", "ALLMASS"):
    f = [(s, c) for s in SEEDS for c in A[a][s].get("checks") or [] if c.get("fired")]
    key = sum(bool(c["key_ok"]) for _, c in f)
    al = sum(bool(c["all_ok"]) for _, c in f)
    op = sum(bool(c["op_ok"]) for _, c in f)
    keyr = sum(lab(c["map_before"], c["key_cs"], c["key_c0"]) for _, c in f)
    alr = sum(lab(c["map_before"], c["all_cs"], c["all_c0"]) for _, c in f)
    same = sum((c["key_cs"], c["key_c0"]) == (c["all_cs"], c["all_c0"]) for _, c in f)
    one = sum(len(set(c["map_before"])) == 8 for _, c in f)
    tgt = {c.get("target") for _, c in f}
    flat = sum(c["acc"] < 0.2 for _, c in f)
    cap = sum(1 for s in SEEDS if A[a][s].get("splits") == 3)
    notlab = [(s, c["step"], c["key_cs"], c["key_c0"], c["all_cs"], c["all_c0"], c["key_ok"], c["all_ok"]) for s, c in f if not c["op_ok"]]
    FI[a] = dict(n=len(f), key=key, all=al, op=op, keyr=keyr, allr=alr, same=same, one=one)
    say(f"  {a}: firings {len(f)}; key_ok {key} (from map_before {keyr}); all_ok {al} (from map_before {alr}); own pair (op_ok) {op}; "
        f"pairs equal {same}; one-to-one before {one}; targets {tgt}; at acc < 0.2: {flat}; runs at the cap of 3: {cap}; own pair not labelled {notlab}")
say(f"  reading 'keeps it' ((BASE only - ALLMASS only) <= 1): {bon - aon <= 1}; 'costs binding' (BASE only >= 4 and ALLMASS only = 0): "
    f"{bon >= 4 and aon == 0}; 'aims' (all_ok / firings >= 0.9): {FI['ALLMASS']['all'] / FI['ALLMASS']['n']:.3f} -> {FI['ALLMASS']['all'] / FI['ALLMASS']['n'] >= 0.9}")
# pooled with S86
s86 = rlib.load(H_S86, "explore_out/H/allmass_results.json")
say(f"  S86 store blob 9b72bf6 == 347bd25: {blob(H_S86, 'explore_out/H/allmass_results.json') == blob(H, 'explore_out/H/allmass_results.json')}")
B6 = {a: rlib.runs(s86, a) for a in ("BASE", "ALLMASS")}
S86S = list(range(2330, 2340))
k6B, k6A = sum(rlib.bound_routed(B6["BASE"][s]) for s in S86S), sum(rlib.bound_routed(B6["ALLMASS"][s]) for s in S86S)
a6, b6, _ = rlib.pairs(B6["ALLMASS"], B6["BASE"], rlib.bound_routed, S86S)
pA, pB, pa, pb = kA + k6A, kB + k6B, aon + a6, bon + b6
say(f"  S86: BASE {k6B}/10, ALLMASS {k6A}/10, ALLMASS only {a6}, BASE only {b6}")
say(f"  pooled 30 pairs: ALLMASS {pA}/30, BASE {pB}/30; ALLMASS only {pa}, BASE only {pb}; p ALLMASS>BASE {mc1(pa, pb):.4g}, BASE>ALLMASS {mc1(pb, pa):.4g}")

# ── H2. one-to-one recount ──
say("\n== H2. The one-to-one recount (main line's WIN3_SPLIT_D8 records)")
FILES = (("window_gate X", "results/X/window_gate_results.json"), ("window_gate L", "results/L/window_gate_results.json"),
         ("split_copy X", "results/X/split_copy_results.json"), ("split_copy L", "results/L/split_copy_results.json"))


def recount(ref):
    out = {}
    for labl, p in FILES:
        try:
            runs = json.loads(show(ref, p))["runs"]
        except subprocess.CalledProcessError:
            continue
        allarm = {k: r for k, r in runs.items() if k.split("|")[0] == "WIN3_SPLIT_D8"}
        rs = {k: r for k, r in allarm.items() if r.get("checks")}
        f = [c for r in rs.values() for c in r["checks"] if c.get("fired")]
        out[labl] = dict(runs=len(rs), runs_all=len(allarm), fired=len(f), one=sum(len(set(c["map_before"])) == 8 for c in f),
                         key=sum(lab(c["map_before"], c["key_cs"], c["key_c0"]) for c in f),
                         key_flag=sum(bool(c["key_ok"]) for c in f), all=sum(lab(c["map_before"], c["all_cs"], c["all_c0"]) for c in f),
                         all_flag=sum(bool(c["all_ok"]) for c in f),
                         same=sum((c["key_cs"], c["key_c0"]) == (c["all_cs"], c["all_c0"]) for c in f))
    tot = {k: sum(v[k] for v in out.values()) for k in ("runs", "runs_all", "fired", "one", "key", "key_flag", "all", "all_flag", "same")}
    return out, tot


RC = {}
for ref in ("0282955", "0754d61", REV10, V101):
    RC[ref] = recount(ref)
    say(f"  at {ref}: {RC[ref][1]}")
    for k, v in RC[ref][0].items():
        say(f"     {k}: {v}")
say(f"  H's onetoone_counts.txt reproduces: (a) 0 of 107 {RC['0282955'][1]['fired'] == 107 and RC['0282955'][1]['one'] == 0}; "
    f"(b) 0 of 126 {RC['0754d61'][1]['fired'] == 126 and RC['0754d61'][1]['one'] == 0}")

# ── H3. S87c ──
say("\n== H3. S87c (dose)")
D = {a: rlib.runs(st_do, a) for a in ("KEYHINGE", "LAM03", "LAM3", "WIN1200", "ORACLE")}
DS = list(range(2630, 2640))
say(f"  runs {len(st_do['runs'])}; not ok {[k for k, r in st_do['runs'].items() if not r.get('ok')]}; "
    f"discovered recorded vs rlib mismatches {[(a, s) for a in D for s, r in D[a].items() if bool(r.get('discovered')) != rlib.discovered(r)]}")
dor = {s: (rlib.bound(r), rlib.transition(r["curve"])) for s, r in D["ORACLE"].items()}
dvalid = len(dor) == 2 and all(b for b, _ in dor.values())
say(f"  ORACLE {dor} -> {'VALID' if dvalid else 'UNTESTED'}")
dk = {a: sum(rlib.discovered(D[a][s]) for s in DS) for a in ("KEYHINGE", "LAM03", "LAM3", "WIN1200")}
dcmp = {v: rlib.pairs(D[v], D["KEYHINGE"], rlib.discovered, DS)[:2] for v in ("LAM03", "LAM3", "WIN1200")}
robust = all(abs(b - c) <= 1 for b, c in dcmp.values())
amend = [v for v, (b, c) in dcmp.items() if b >= 4 and c == 0]
say(f"  DISCOVERED {dk}; V only / KEYHINGE only {dcmp}; 'robust' {robust}; amendment fires {amend}")
for a in ("KEYHINGE", "LAM03", "LAM3", "WIN1200"):
    tr = sorted(rlib.transition(D[a][s]["curve"]) for s in DS)
    win = 1200 if a == "WIN1200" else 2400
    mx = max(v["etak_key_by_key"] for s in DS for t, v in D[a][s]["dense"].items() if 25 <= int(t) <= win)
    endk = sum(D[a][s]["dense"][str(max(map(int, D[a][s]["dense"])))]["etak_key_by_key"] > 0.5 for s in DS)  # S85 key_onset: last measurement
    say(f"  {a}: lam {D[a][DS[0]]['hinge']['lam']} window {D[a][DS[0]]['hinge']['window']}; transitions median {rlib.med(tr)} {tr}; "
        f"max eta2 key (25..{win}) {mx:.3f}; end key split > 0.5: {endk}; firings {[D[a][s]['hinge']['fired'] for s in DS]}")
same_w = all(D["WIN1200"][s]["curve"] == D["KEYHINGE"][s]["curve"] and D["WIN1200"][s]["hinge"]["fired"] == D["KEYHINGE"][s]["hinge"]["fired"] for s in DS)
say(f"  WIN1200 == KEYHINGE run for run (curve and firings): {same_w}")


def p_robust(p, n=10):
    """P('robust') for one variant at true rate p against a KEYHINGE at 1.0 (V only = 0, KEYHINGE only = failures of V)."""
    return sum(math.comb(n, j) * (1 - p) ** j * p ** (n - j) for j in range(0, 2))


PR = {p: p_robust(p) for p in (0.9, 0.8, 0.7)}
say(f"  power: P(a variant reads within 1 pair | its true rate p, KEYHINGE at 10/10) {', '.join(f'p={p}: {v:.2f}' for p, v in PR.items())}")

# ══ E ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
say("\n" + "=" * 110)
say("SESSION E, BATCHES 20 AND 21 (claude/outside-ideas at 419368a)")
say("=" * 110)
eh = BRANCH_PINS["origin/claude/outside-ideas"]
say(f"branch head as the original read it (pinned): {eh} (== {E}: {eh == E})")
say("\n== E0. Readings before records")
for f in ("explore_continue.py", "explore_continue_child.py", "explore_batch20.py", "explore_oracle_d128.py", "explore_oracle_child.py", "explore_batch21.py"):
    say(f"  {f}: {hist(E, f)}")


def docstring(ref, f):
    s = show(ref, f)
    return s.split('"""')[1]


d20 = docstring(E20_READ, "explore_continue.py") == docstring(E, "explore_continue.py")
d21 = docstring(E21_READ, "explore_oracle_d128.py") == docstring(E, "explore_oracle_d128.py")
say(f"  docstring (READINGS) identical at the head: S68 {d20}; S69 {d21}")
for a_, b_, f in ((E20_READ, E, "explore_continue.py"), (E21_READ, E, "explore_oracle_d128.py"), (E21_READ, E, "explore_batch21.py")):
    dd = [l for l in git("diff", a_, b_, "--", f).split("\n") if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    say(f"  {f} {a_}->{b_}: {len(dd)} changed lines, in functions: {sorted({l.strip()[:1] for l in dd})} (see the diffs: report()/both_two/run_checks only)")
E_FR = {}
for st in ("explore_out/s67_continue_results.json", "explore_out/oracle_d128_results.json",
           "explore_out/dry_s67_continue_results.json", "explore_out/dry_oracle_d128_results.json"):
    E_FR[st] = (hist(E, st)[-1], first_nonempty(E, st) if "dry_" not in st else None)
    say(f"  {st}: oldest commit {E_FR[st][0]}; first non-dry run {E_FR[st][1]}")
say(f"  readings: S68 {E20_READ} {ctime(E20_READ)}; S69 {E21_READ} {ctime(E21_READ)}")
E20_ORDER = d20 and E_FR["explore_out/s67_continue_results.json"][1][1] > ctime(E20_READ) and E_FR["explore_out/dry_s67_continue_results.json"][0][1] > ctime(E20_READ)
E21_ORDER = d21 and E_FR["explore_out/oracle_d128_results.json"][1][1] > ctime(E21_READ) and E_FR["explore_out/dry_oracle_d128_results.json"][0][1] > ctime(E21_READ)
say(f"  ORDER VERDICT: batch 20 {E20_ORDER}; batch 21 {E21_ORDER}")
st20 = rlib.load(E, "explore_out/s67_continue_results.json")
st21 = rlib.load(E, "explore_out/oracle_d128_results.json")
b20log = show(E, "explore_out/batch20.log").split("\n")
b21log = show(E, "explore_out/batch21.log").split("\n")
sha20 = code_sha(E20_RUN, mods_from_log(b20log, "code SHA (the child)"), True)
sha21 = code_sha(E21_READ, mods_from_log(b21log, "code SHA (the child)"), True)
say(f"  code SHA recomputed from committed sources: batch 20 at {E20_RUN} {sha20[:12]} (store {st20['meta']['code_shas']}); "
    f"batch 21 at {E21_READ} {sha21[:12]} (store {st21['meta']['code_shas']})")
say(f"  per-record code SHAs: b20 {sorted({r['code_sha'][:12] for r in st20['runs'].values()})}, b21 {sorted({r['code_sha'][:12] for r in st21['runs'].values()})}; "
    f"CPUs b20 {sorted({r['cpu'][-7:] for r in st20['runs'].values()})}, b21 {sorted((r['cpu'][-7:], k.split('|')[0][-6:] + '|' + k.split('|')[1]) for k, r in st21['runs'].items() if '2.10' in r['cpu'])}")

# ── E1. batch 20 ──
say("\n== E1. Batch 20 (S68): S67's unbound P=16 runs continued to 86,400")
cap = rlib.load(E19, "explore_out/capacity_results.json")
C67 = {a: rlib.runs(cap, a) for a in ("P16_D128", "P16_N256", "P8_N256", "ORC_P16_D128", "ORC_P16_N256")}
S67S = list(range(270, 280))
o67 = {a: {s: rlib.outcome_k(C67[a][s]) for s in S67S} for a in ("P16_D128", "P16_N256")}
for a in o67:
    say(f"  S67 {a}: BR {sum(v == 'BOUND ROUTED' for v in o67[a].values())}/10; unbound {[(s, v) for s, v in o67[a].items() if v != 'BOUND ROUTED']}")
UNB = {a: [s for s in S67S if o67[a][s] != "BOUND ROUTED"] for a in o67}
O2O = {a: [s for s in UNB[a] if o67[a][s] == "STREAM-PARTIAL one-to-one"] for a in o67}
MER = {a: [s for s in UNB[a] if o67[a][s].startswith("MERGED")] for a in o67}
C = {a: rlib.runs(st20, a) for a in ("C_P16_D128", "C_P16_N256", "C_P8_N256", "C_ORC_P16_D128", "RCHK_P8_N256")}
say(f"  S68 records {len(st20['runs'])}: {[(a, sorted(v)) for a, v in C.items()]}; not ok {[k for k, r in st20['runs'].items() if not r.get('ok')]}")
covered = sorted(C["C_P16_D128"]) == UNB["P16_D128"] and sorted(C["C_P16_N256"]) == UNB["P16_N256"]
say(f"  every unbound S67 P=16 learned run continued: {covered}")


def repro(r68, r67):
    c68 = [c for c in r68["curve"] if c[0] <= 43200]
    s68 = [s for s in r68["stats"] if s.get("step") != "end" and s["step"] <= 43200]
    s67 = [s for s in r67["stats"] if s.get("step") != "end" and s["step"] <= 43200]
    sha = dict((int(a), b) for a, b in r68["shas"]).get(43200)
    return dict(curve=c68 == r67["curve"], stats=s68 == s67, early=r68.get("early") == r67.get("early"), w=sha == r67["w_sha"])


REP = {}
for a68, a67 in (("C_P16_D128", "P16_D128"), ("C_P16_N256", "P16_N256"), ("C_P8_N256", "P8_N256"), ("C_ORC_P16_D128", "ORC_P16_D128")):
    for s, r in C[a68].items():
        REP[(a68, s)] = repro(r, C67[a67][s])
bad = {k: v for k, v in REP.items() if not all(v.values())}
say(f"  reproduction at 43,200 (curve, stats, early curve, weights SHA-1 vs S67 w_sha): {len(REP)} runs, failing {bad}")
r67 = C67["P8_N256"][272]
rc = C["RCHK_P8_N256"][272]
say(f"  RCHK_P8_N256 272 vs S67 through 43,200: curve {[c for c in rc['curve'] if c[0] <= 43200] == r67['curve']}")
cp = C["C_P8_N256"][272]


def at(r, step):
    return next((c for c in r["curve"] if c[0] == step), None)


st44 = lambda r: [s for s in r["stats"] if s.get("step") == 44400]  # noqa: E731
say(f"  44,400 (resume check): curve C {at(cp, 44400)} vs RCHK {at(rc, 44400)} equal {at(cp, 44400) == at(rc, 44400)}; stats equal {st44(cp) == st44(rc)}; "
    f"shas C {cp['shas']} RCHK {rc['shas']}; C's resumed list {cp['resumed']}, halted_once_at {cp.get('halted_once_at')}")
say(f"  log: {[l.strip()[:150] for l in b20log if 'S68 reproduction of a resumed run' in l][-1]}")
OUT = {}
for a in ("C_P16_D128", "C_P16_N256"):
    for s, r in sorted(C[a].items()):
        a43, a86 = at(r, 43200)[1], at(r, 86400)[1]
        a768 = at(r, 76800)[1]
        OUT[(a, s)] = dict(o=rlib.outcome_k(r), rec=r["outcome"], a43=a43, a86=a86, d=a86 - a43, rise=a86 - a768, tr=rlib.transition(r["curve"]))
        say(f"  {a}|{s}: S67 {o67[a[2:]][s]}; at 86,400 {OUT[(a, s)]['o']} (recorded {r['outcome']}); acc {a43:.3f} -> {a86:.3f} ({a86 - a43:+.3f}); "
            f"76,800->86,400 {a86 - a768:+.3f}")
k = {w: sum(rlib.bound_routed(C["C_P16_" + w][s]) for s in O2O["P16_" + w]) for w in ("D128", "N256")}
kmer = sum(rlib.bound_routed(C["C_P16_N256"][s]) for s in MER["P16_N256"])


def wread(kw):
    return "slow, not stuck" if kw >= 4 else "stuck" if kw == 0 else "unresolved"


joint = ("the budget is the lead" if k["D128"] >= 4 and k["N256"] >= 4 else "the width is the lead" if k["D128"] >= 4 and k["N256"] <= 1 else "neither")
dD = [OUT[("C_P16_D128", s)]["d"] for s in O2O["P16_D128"]]
dN = [OUT[("C_P16_N256", s)]["d"] for s in O2O["P16_N256"]]
rising = [f"{a}|{s}" for (a, s), v in OUT.items() if (s in O2O[a[2:]]) and v["a86"] < 1 and v["rise"] >= 0.02 and not rlib.bound_routed(C[a][s])]
trs = sorted([rlib.transition(C67[a][s]["curve"]) for a in ("P16_D128", "P16_N256") for s in S67S if o67[a][s] == "BOUND ROUTED"]
             + [v["tr"] for (a, s), v in OUT.items() if v["o"] == "BOUND ROUTED"])
budget = int(math.ceil((max(trs) + 2400) / 2400) * 2400)
orc270 = C["C_ORC_P16_D128"][270]
say(f"  one-to-one bound by 86,400: D128 {k['D128']}/{len(O2O['P16_D128'])} -> '{wread(k['D128'])}'; N256 {k['N256']}/{len(O2O['P16_N256'])} -> '{wread(k['N256'])}'; "
    f"joint '{joint}'; merged N256 {kmer}/{len(MER['P16_N256'])}")
say(f"  D128 one-to-one change 43,200->86,400: {min(dD):+.3f} .. {max(dD):+.3f}; N256 one-to-one: {min(dN):+.3f} .. {max(dN):+.3f}")
say(f"  still rising (>= 0.02, 76,800->86,400): {rising}; BR P=16 learned transitions (S67+S68) {trs}; budget {budget}")
say(f"  D128 oracle 270 at 86,400: bound {rlib.bound(orc270)}, acc {at(orc270, 86400)[1]:.3f}, streams {[round(x, 2) for x in orc270['end']['stream_acc']]}; "
    f"S67 oracle D128 transitions {[(s, rlib.transition(r['curve'])) for s, r in C67['ORC_P16_D128'].items()]}")
say(f"  N256 279: S67 {o67['P16_N256'][279]} -> S68 {OUT[('C_P16_N256', 279)]['o']}")
log20 = [l.strip() for l in b20log if "READING S68" in l or "BUDGET for the main-line" in l][-4:]
say("  log readings: " + " | ".join(x[:120] for x in log20))

# ── E2. batch 21 ──
say("\n== E2. Batch 21 (S69): the D128 oracle on seeds 270-279 at 40,800")
O = {a: rlib.runs(st21, a) for a in ("ORC_A_D128", "ORC_S_D128")}
say(f"  records {len(st21['runs'])}: {[(a, sorted(v)) for a, v in O.items()]}; not ok {[k for k, r in st21['runs'].items() if not r.get('ok')]}")
bA = {s: rlib.bound(r) for s, r in O["ORC_A_D128"].items()}
bS = {s: rlib.bound(r) for s, r in O["ORC_S_D128"].items()}
nA, nS = sum(bA.values()), sum(bS.values())
trA = sorted(rlib.transition(O["ORC_A_D128"][s]["curve"]) for s in bA if bA[s])
fin = {a: {s: r["curve"][-1][1] for s, r in O[a].items()} for a in O}
unA = {s: fin["ORC_A_D128"][s] for s in bA if not bA[s]}
close = sorted(s for s, v in unA.items() if v > 0.8)
stuck = sorted(s for s, v in unA.items() if v < 0.5)
say(f"  ORC_A_D128 bound {nA}/10 {sorted(s for s in bA if bA[s])} at {trA}; unbound finals {({s: round(v, 3) for s, v in unA.items()})}")
say(f"  ORC_S_D128 bound {nS}/10; finals {min(fin['ORC_S_D128'].values()):.3f}..{max(fin['ORC_S_D128'].values()):.3f}; "
    f"streams {min(min(r['end']['stream_acc']) for r in O['ORC_S_D128'].values()):.2f}..{max(max(r['end']['stream_acc']) for r in O['ORC_S_D128'].values()):.2f}")
say(f"  SLOW final minus single-Adam final per seed: {({s: round(fin['ORC_S_D128'][s] - fin['ORC_A_D128'][s], 3) for s in S67S})}")


def arm_read(b):
    return "reliable" if b >= 9 else "at risk" if b >= 5 else "unreliable"


spec = ("keep the draft's oracle" if nA >= 9 else "the D128 oracle on SLOW" if nS >= 9 else "VALID_N = 1 at D128" if max(nA, nS) >= 5
        else "no D128 validity check at 40,800")
both2 = nA * (nA - 1) / 90
a_only = sum(bA[s] and not bS[s] for s in S67S)
s_only = sum(bS[s] and not bA[s] for s in S67S)
say(f"  readings: A '{arm_read(nA)}', S '{arm_read(nS)}'; spec '{spec}'; two of these ten both bind {both2:.3f}; Wilson A {rlib.wilson(nA, 10)}; "
    f"A only {a_only}, S only {s_only}, McNemar two-sided {rlib.mcnemar2(a_only, s_only):.3f}")
say(f"  fresh seeds at the observed rate: 0.4^2 = {0.4 ** 2:.2f} per machine; both machines {0.4 ** 4:.3f}")
for s in (270, 271):
    rA, r7 = O["ORC_A_D128"][s], C67["ORC_P16_D128"][s]
    say(f"  ORC_A_D128 {s} vs S67 ORC_P16_D128: curve equal through 40,800 {[c for c in rA['curve'] if c[0] <= 40800] == [c for c in r7['curve'] if c[0] <= 40800]}; "
        f"final weights SHA equal {rA.get('w_sha') == r7.get('w_sha')} (S67 transition {rlib.transition(r7['curve'])})")
log21 = [l.strip() for l in b21log if "READING S69" in l][-3:]
say("  log readings: " + " | ".join(x[:110] for x in log21))
kd = show(E, "explore_out/spec_draft_keyload.md")
opts = [l for l in kd.split("\n")[:30] if l[:3] in ("1. ", "2. ", "3. ", "4. ", "5. ")]
say(f"  spec draft status: {'NOT READY FOR V' in kd}; options {len(opts)}; 'None is screened and none is adopted' {'None is screened and none is adopted' in kd}; "
    f"commits {hist(E, 'explore_out/spec_draft_keyload.md')}")

# ── the other session's rev10 recompute (read, not run) ──
say("\n== X. main's docs/review_2026-10/rev10/recompute_H_out.txt (read only)")
ro = show(REV10, "docs/review_2026-10/rev10/recompute_H_out.txt").split("\n")
bl = [l for l in ro if l.startswith("BASE BR")]
say(f"  its BASE line: {bl[0][:90] if bl else None}; here BASE own pair (op_ok, the key pair for BASE) {FI['BASE']['op']}/{FI['BASE']['n']}, all_ok {FI['BASE']['all']}")
say(f"  its 'own pair ok' falls back to all_ok when no own_ok field exists (recompute_H.py: ch.get('own_ok', ch.get('all_ok'))): "
    f"{'own_ok' in show(REV10, 'docs/review_2026-10/rev10/recompute_H.py')}; a traceback in the committed output: {any('Traceback' in l for l in ro)}; "
    f"lines repeated after it: {sum(1 for l in ro if l.startswith('TOTAL'))} TOTAL line(s)")
say(f"  rev10/ scripts on main: {[x for x in git('ls-tree', '--name-only', REV10, 'docs/review_2026-10/rev10/').split() if x.endswith('.py')]}")

# ══ ROWS ══════════════════════════════════════════════════════════════════════════════════════════════════════════
M, P, Dg, N = "MATCH", "PARTIAL", "DISAGREE", "NOT RECORD-CHECKABLE"
# H rows
row(44, "H: fifth round includes H's S87; after the merge the referee checked the fifth round's screens (rev10/referee_pass.md)", "H's S87, I's last S75 arm",
    f"S87a/S87c/recount at {H}; V's branch had no S87 check before Rev 10 (check_H_v6: 0 records); at {V101} check_rev10_HE.md and referee_pass.md in rev10/: "
    f"{all(x in git('ls-tree', '--name-only', V101, 'docs/review_2026-10/rev10/') for x in ('check_rev10_HE.md', 'referee_pass.md'))}", M,
    "the original row was MATCH for 'not refereed' (true at 585b666); 846d0c8 records the referee's check, which exists",
    new_frag="and, after the merge, of the whole revision and of the fifth round's screens (\\texttt{rev10/referee\\_pass.md}")
f67 = f"twenty new seeds (${kA}/20$ against ${kB}/20$; ${pA}/30$ against ${pB}/30$ over two screens)"
row(67, "H index: all-position trigger", f67, f"S87a {kA}/20 vs {kB}/20; pooled {pA}/30 vs {pB}/30", M)
row(67, "H index: no main-line firing on a one-to-one map", "no main-line firing had a one-to-one map before it",
    f"0 of {RC[REV10][1]['fired']} at {REV10}", M if RC[REV10][1]["one"] == 0 else Dg)
row(67, "H index: 'the key term's constants are robust on a ceiling screen'", "the key term's constants are robust",
    f"S87c 'robust' {robust}; every arm {dk}; drafted arm at the ceiling (10/10)", M if robust and dk["KEYHINGE"] == 10 else Dg,
    f"846d0c8 adds the scope the original row asked for (the drafted arm was at 10/10, so the screen could show only a worse variant; a "
    f"variant at a true 0.8 reads within one pair {PR[0.8]:.0%} of the time, at 0.9 {PR[0.9]:.0%})",
    new_frag="the key term's constants are robust on a ceiling screen")
f217a = (f"bound ${kA}/20$ against the key-position rule's ${kB}/20$ (S87a, seeds 2600--2619; {aon} vs {bon}), ${pA}/30$ against ${pB}/30$ pooled with S86")
row(217, "S87a counts, pairs, pooled", f217a, f"ALLMASS {kA}/20, BASE {kB}/20, {aon} vs {bon}; pooled {pA}/30 vs {pB}/30 ({pa} vs {pb}); ORACLE 2/2 {orc} VALID", M)
f217b = f"all {FI['ALLMASS']['n']} of its firings on a target the key-position labels name and {FI['ALLMASS']['all']} on one the all-position labels name"
row(217, "S87a ALLMASS firings labelled", f217b if FI["ALLMASS"]["key"] == FI["ALLMASS"]["n"] else None,
    f"firings {FI['ALLMASS']['n']}, key_ok {FI['ALLMASS']['key']} (map_before {FI['ALLMASS']['keyr']}), all_ok {FI['ALLMASS']['all']} (map_before {FI['ALLMASS']['allr']})",
    M if FI["ALLMASS"]["key"] == FI["ALLMASS"]["n"] else Dg)
row(217, "S87a readings", "its readings ``keeps it'' and ``aims'' apply, ``costs binding'' does not (printed ``no cost by the rule'')",
    f"keeps it {bon - aon <= 1} ({bon} vs {aon}); aims {FI['ALLMASS']['all']}/{FI['ALLMASS']['n']} >= 0.9; costs binding {bon >= 4 and aon == 0}", M)
row(217, "S87a: 'and the key-position rule stays'", "and the key-position rule stays", "no reading or record names it; report 9's main-line reading: within two pairs (24 vs 26)", N,
    "a design decision, not a reading; nothing in H's records decides it")
f217c = f"none of the {RC[REV10][1]['fired']} firings in the {RC[REV10][1]['runs']} recorded split-arm runs of \\texttt{{test\\_window\\_gate}} and \n"
f217c = f"none of the {RC[REV10][1]['fired']} firings in the {RC[REV10][1]['runs']} recorded split-arm runs of \\texttt{{test\\_window\\_gate}}"
row(217, "one-to-one guard: 'would have changed nothing'", f217c,
    f"WIN3_SPLIT_D8 with checks: {RC[REV10][1]['runs']} runs (all {RC[REV10][1]['runs_all']}), {RC[REV10][1]['fired']} firings, one-to-one before {RC[REV10][1]['one']}; "
    f"H's recount (a) 0/107 at 0282955, (b) 0/126 at 0754d61 reproduce", M)
row(217, "S87c counts and readings, with the ceiling and its power", "all discovered $10/10$ at two streams (S87c; ``robust'', and the drafted constants stand)",
    f"{dk}; ORACLE {dor} VALID; V only/K only {dcmp}; robust {robust}; amendment {amend or 'none'}; P(reads 'robust' | true 0.8, KEYHINGE 10/10) {PR[0.8]:.3f}",
    M if robust and round(PR[0.8] * 100) == 38 else Dg,
    f"counts and readings MATCH; 846d0c8 now states the ceiling and the power ('a variant at a true rate of 0.8 still reads robust 38% of the "
    f"time'), the caveat the original row found missing; LAM3's median transition is later ({rlib.med(sorted(rlib.transition(D['LAM3'][s]['curve']) for s in DS))} vs "
    f"{rlib.med(sorted(rlib.transition(D['KEYHINGE'][s]['curve']) for s in DS))}), descriptive",
    new_frag="all discovered $10/10$ at two streams (S87c; ``robust'', and the drafted constants stand; the drafted arm was at the ceiling, so the screen could only have shown a worse variant, and a variant at a true rate of $0.8$ still reads ``robust'' $" + f"{round(PR[0.8] * 100)}" + "\\%$ of the time)")
# S86's ALLMASS firings and mistargets (all-position pair not a labelled target), with the run's outcome
f86 = [(s_, c) for s_ in S86S for c in B6["ALLMASS"][s_].get("checks") or [] if c.get("fired")]
mt86 = [(s_, c["step"], rlib.bound_routed(B6["ALLMASS"][s_])) for s_, c in f86 if not c["all_ok"]]
f87 = [(s_, c) for s_ in SEEDS for c in A["ALLMASS"][s_].get("checks") or [] if c.get("fired")]
mt87 = [(s_, c["step"], rlib.bound_routed(A["ALLMASS"][s_])) for s_, c in f87 if not c["all_ok"]]
cost = sum(1 for *_, b in mt86 + mt87 if not b)
say(f"  ALLMASS mistargets: S86 {mt86} of {len(f86)} firings; S87a {mt87} of {len(f87)}; runs that did not bind routed after one: {cost}")
row(358, "'screened twice, session H's S86 and S87a, 24/30 against 26/30'", "a trigger that reads all positions was screened once, session~H's S86, $8/10$ against $9/10$",
    f"screened twice: S86 ({k6A}/10 vs {k6B}/10) and S87a ({kA}/20 vs {kB}/20); pooled {pA}/30 vs {pB}/30", M if (pA, pB) == (24, 26) else Dg,
    "846d0c8 takes the referee's fix. The paragraph's heading now reads 'what Revision 9 did with it', while this item reports S87a, which "
    "came after Revision 9: the item is the request's current state",
    new_frag="a trigger that reads all positions was screened twice, session~H's S86 and S87a, $" + f"{pA}/30$ against ${pB}/30$")
row(464, "'screened twice (S86 and S87a): within one pair on each screen ...; of its three mistargets in 63 firings two cost their run'",
    "a trigger that reads all positions was screened once (session~H's S86)",
    f"S86: ALLMASS only {a6}, BASE only {b6}; S87a: {aon} vs {bon}; pooled {pA}/30 vs {pB}/30; firings {len(f86)} + {len(f87)} = {len(f86) + len(f87)}; "
    f"mistargets {len(mt86)} + {len(mt87)} (seed, step, run bound routed): S86 {mt86}, S87a {mt87}; runs lost after a mistarget {cost}",
    M if (len(f86) + len(f87), len(mt86) + len(mt87), cost) == (63, 3, 2) and b6 - a6 <= 1 and bon - aon <= 1 else Dg,
    "846d0c8 takes the referee's fix and recounts the mistargets over both screens; 'within one pair on each screen' holds for S86 and "
    "S87a (the pooled 1 vs 3 is two pairs, and is given as a count, not as a screen)",
    new_frag="a trigger that reads all positions was screened twice (session~H's S86 and S87a): it keeps binding within one pair of the key-position rule on each screen ($8/10$ against $9/10$; $16/20$ against $17/20$; $24/30$ against $26/30$ pooled), and of its three mistargets in 63 firings two cost their run")
row(486, "'screened twice (S86, S87a) and kept binding within one pair on each'", "a trigger on all positions was screened once (S86) and kept binding within one pair",
    f"S86 BASE only - ALLMASS only = {b6 - a6}; S87a {bon - aon}", M if b6 - a6 <= 1 and bon - aon <= 1 else Dg, "846d0c8 takes the referee's fix",
    new_frag="a trigger on all positions was screened twice (S86, S87a) and kept binding within one pair on each")
r464 = RC[REV10][0]
f464 = (f"at {r464['window_gate X']['all']}~\\@?") if False else None
f464 = (f"at {RC[REV10][1]['all']} ({r464['window_gate X']['all']} of {r464['window_gate X']['fired']}, {r464['window_gate L']['all']} of {r464['window_gate L']['fired']}, "
        f"{r464['split_copy X']['all']} of {r464['split_copy X']['fired']} and {r464['split_copy L']['all']} of {r464['split_copy L']['fired']}), the key-position pair at all {RC[REV10][1]['key']}; "
        f"the two pairs coincide at {RC[REV10][1]['same']}")
row(464, "main line's counterfactual labels (recheck)", f464, f"labels recomputed from map_before == flags: all {RC[REV10][1]['all'] == RC[REV10][1]['all_flag']}, key {RC[REV10][1]['key'] == RC[REV10][1]['key_flag']}", M)
row(536, "provenance (tab:batches), H's row: S87a, S87c and the recount (report 9, 347bd25), commit 347bd25, runs ...; 84",
    "the one-to-one recount over the main line's 126 firings (\\texttt{5b5deec}, round 5) & \\texttt{5b5deec} & 318; 300; 82; 62; 52; 82; 32",
    f"S87a {len(st_am['runs'])} + S87c {len(st_do['runs'])} runs, report 9 at {H} (explore_out/H/report_9.md there: "
    f"{'explore_out/H/report_9.md' in git('ls-tree', '-r', '--name-only', H, 'explore_out/H/')})",
    M if len(st_am["runs"]) + len(st_do["runs"]) == 84 else Dg, "846d0c8 adds S87a and S87c, report 9's commit and the 84 runs",
    new_frag="S87a the all-position trigger on 20 more seeds, S87c the key term's dose, and the one-to-one recount over the main line's 126 firings (report 9, \\texttt{347bd25}) & \\texttt{347bd25} & 318; 300; 82; 62; 52; 82; 32; 84\\\\")
row(44, "screen-number collision G's S87 / H's S87", "G's S87, H's S87", "both sessions numbered round 5 'S87' (G 2700-2729; H S87a/S87c)", M,
    "the draft disambiguates by session and by suffix (S87a, S87c); noted only")

# E rows
f64 = f"stuck (${k['D128'] + k['N256']}/{len(O2O['P16_D128']) + len(O2O['P16_N256'])}$ by 86{{,}}400); the oracle ${nA}/10$ and ${nS}/10$ at 40{{,}}800"
row(64, "E index row", f64, f"one-to-one 0/6 + 0/6; ORC_A {nA}/10, ORC_S {nS}/10; spec reading '{spec}'", M)
row(44, "E: 'fifth round (E's batches 20 and 21 ...)'", "E's batches~20 and~21", "prompts file: batch 20's prompt sits before the 'Round 5' header, which names V, G, H, F, R and I, not E; batch 21 was a chat request (13 Oct)", P,
    "labelling only: batch 20 was specified on 11 October, before round 5")
f245a = "(six one-to-one at $D = 128$; six one-to-one and four merged at $N = 256$) was regenerated from update 0, reproduced its record at 43{,}200 bit for bit, and continued to 86{,}400 updates"
row(245, "S68: which runs; reproduced at 43,200", f245a,
    f"S67 unbound: D128 {o67 and len(UNB['P16_D128'])} ({len(O2O['P16_D128'])} one-to-one), N256 {len(UNB['P16_N256'])} ({len(O2O['P16_N256'])} one-to-one, {len(MER['P16_N256'])} merged); "
    f"all continued {covered}; reproduction failing {bad or 'none'} of {len(REP)} (curve, stats, early curve, weights SHA-1)",
    M if covered and not bad and len(O2O["P16_D128"]) == 6 and len(O2O["P16_N256"]) == 6 and len(MER["P16_N256"]) == 4 else Dg)
f245b = (f"none bound (the one-to-one groups ${k['D128']}/6$ and ${k['N256']}/6$, on which the pre-fixed readings ``{wread(k['D128'])}'' at both widths and ``{joint}'' jointly are read; "
         f"the merged four ${kmer}/4$, descriptive)")
row(245, "S68 counts and readings", f245b if wread(k["D128"]) == wread(k["N256"]) else None, f"D128 {k['D128']}/6 '{wread(k['D128'])}', N256 {k['N256']}/6 '{wread(k['N256'])}', joint '{joint}', merged {kmer}/4", M)
dDr = [round(round(OUT[("C_P16_D128", s)]["a86"], 2) - round(OUT[("C_P16_D128", s)]["a43"], 2), 2) for s in O2O["P16_D128"]]
dNr = [round(round(OUT[("C_P16_N256", s)]["a86"], 2) - round(OUT[("C_P16_N256", s)]["a43"], 2), 2) for s in O2O["P16_N256"]]
say(f"  changes from endpoints rounded to 2 decimals (README's way): D128 {dDr}; N256 {dNr}")
f245c = (f"the $D = 128$ one-to-one runs creeping by ${min(dDr):.2f}$--${max(dDr):.2f}$ over the second 43{{,}}200 updates and the $N = 256$ ones moving by "
         f"${min(dNr):.2f}$ to $+{max(dNr):.2f}$ without binding")
f245c101 = (f"the $D = 128$ one-to-one runs creeping by ${round(min(dD), 2):.2f}$--${round(max(dD), 2):.2f}$ over the second 43{{,}}200 updates and the $N = 256$ ones moving by "
            f"${round(min(dN), 2):.2f}$ to $+{round(max(dN), 2):.2f}$ without binding")
row(245, "S68 accuracy changes (unrounded differences, rounded once)", f245c, f"unrounded: D128 {[round(x, 3) for x in dD]}; N256 {[round(x, 3) for x in dN]}", M,
    f"846d0c8 now rounds the unrounded differences once: N256 {min(dN):+.4f} -> {round(min(dN), 2):+.2f} (seed 274), as the referee's pass noted; the "
    f"D128 range {min(dD):+.3f} to {max(dD):+.3f} reads 0.02-0.10 either way", new_frag=f245c101)
row(245, "S68 D128 oracle on 270", "the $D = 128$ perfect gate on the seed it had failed (270) was still unbound at 86{,}400",
    f"bound {rlib.bound(orc270)}; acc {at(orc270, 86400)[1]:.3f} at 86,400; S67 oracle D128: 271 at 12000, 270 unbound", M if not rlib.bound(orc270) else Dg)
f245d = f"gave {num(budget)} (the latest transition of any bound sixteen-key run, {num(max(trs))}, plus 2400), and printed beside it ``86{{,}}400 is not enough''"
row(245, "budget rule", f245d, f"BR P=16 learned transitions {trs} -> {budget}; still rising {rising}", M,
    "the rule is on BOUND ROUTED learned runs (the draft's 'any bound sixteen-key run' would also take the oracles: 12000, 24000, 27600; same maximum)")
row(245, "key-load draft rested on a 1-of-2 oracle", "rested on an oracle that had bound one of its two seeds (three runs, the failed seed having been continued)",
    f"S67 ORC_P16_D128 {[(s, rlib.transition(r['curve'])) for s, r in sorted(C67['ORC_P16_D128'].items())]}; S68 continued 270 (unbound)", M)
row(245, "S69 order and design", "(S69, seeds 270--279, 40{,}800 updates, readings fixed before any run)",
    f"readings {E21_READ} {ctime(E21_READ)}; first record {E_FR['explore_out/oracle_d128_results.json'][1]}; docstring unchanged {d21}", M if E21_ORDER else Dg,
    new_frag="S69, not F's; seeds 270--279, 40{,}800 updates, readings fixed before any run)")
f245e = (f"binds ${nA}/10$ under the recorded single-Adam recipe (three more seeds ended at ${min(unA[s] for s in close):.2f}$--${max(unA[s] for s in close):.2f}$, under the $0.95$ bar, "
         f"and three stuck near $0.30$) and ${nS}/10$ under SLOW (every seed at ${min(fin['ORC_S_D128'].values()):.2f}$--${max(fin['ORC_S_D128'].values()):.2f}$")
row(245, "S69 counts", f245e, f"A bound {sorted(s for s in bA if bA[s])} at {trA}; close {close}, stuck {stuck} ({[round(unA[s], 3) for s in stuck]}); S {nS}/10", M)
lower = {s_: fin["ORC_S_D128"][s_] < fin["ORC_A_D128"][s_] for s_ in S67S if bA[s_] or s_ in close}
row(245, "S69: 'the slow start hurt: no SLOW run bound and SLOW ended lower on every seed that bound or came close under the other recipe'",
    "with the routing given and no gate to protect, the slow start only delays",
    f"SLOW bound {nS}/10; SLOW final below single Adam's on the {len(lower)} seeds that bound or came close: {sum(lower.values())} of {len(lower)}; above it on "
    f"the 3 stuck ones ({({s: round(fin['ORC_S_D128'][s] - fin['ORC_A_D128'][s], 2) for s in stuck})})",
    M if nS == 0 and all(lower.values()) and len(lower) == 7 else Dg,
    "846d0c8 replaces the interpretation with what the records show, as E's README reads it ('it hurt')",
    new_frag="the slow start hurt: no SLOW run bound and SLOW ended lower on every seed that bound or came close under the other recipe")
row(245, "S69 pre-registered reading", "The pre-registered reading is ``no $D = 128$ validity check at 40{,}800''", f"rule 4 of the docstring: b_A {nA}, b_S {nS} -> '{spec}'", M)
row(245, "S69: 'two validity seeds drawn from these ten ... about 0.13 (two fresh seeds on one machine about 0.16 at that rate, on both machines 0.03)'",
    "two validity seeds would both bind with probability about $0.13$",
    f"b(b-1)/90 = {both2:.3f} (two of these ten seeds, without replacement); fresh seeds at {nA}/10: {(nA / 10) ** 2:.2f} per machine, {(nA / 10) ** 4:.4f} on both",
    M if (round(both2, 2), round((nA / 10) ** 2, 2), round((nA / 10) ** 4, 2)) == (0.13, 0.16, 0.03) else Dg,
    "846d0c8 states both models, as the referee's pass and keyload_options_review.md item 1 asked",
    new_frag="two validity seeds drawn from these ten would both bind with probability about $0.13$ (two fresh seeds on one machine about $0.16$ at that rate, on both machines $0.03$)")
row(245, "status note: five ways forward", "Session~E's status note lists five ways forward, none screened and none adopted",
    f"spec_draft_keyload.md at {E}: 'NOT READY FOR V'; {len(opts)} options; 'None is screened and none is adopted here'", M if len(opts) == 5 else Dg)
row(245, "E's verdict on the lead", "is ``inconclusive, weaker than handed back''", "README Batch 20 verdicts: 'Verdict on the batch-19 lead ... inconclusive, weaker than handed back'; 4 vs 0, two-sided p "
    f"{rlib.mcnemar2(4, 0)}", M)
s68_in = [n_ for n_, l in enumerate(TEX, 1) if "S68" in l]
row(245, "batch 21 named 'E's S69, not F's'", "batch~21 screened that first (S69,",
    f"tex101:258 and tab:batches use S69 for session F's Hebbian-decay screen; E numbered its batch-20/21 screens S68/S69, both F's numbers; "
    f"lines of tex101 naming S68: {s68_in}", M,
    "846d0c8 qualifies E's S69 at its one use in the text. E's S68 is named only in the provenance table (tex101:540), whose rows are "
    "labelled by session, beside F's S68 (tex101:542)",
    new_frag="batch~21 screened that first (E's S69, not F's;")
row(366, "key-load draft: after the reviews; not reviewed", "Session~E's key-load draft was committed after those reviews and then failed its own validity screen",
    "draft first committed f5d3893 2026-10-13T02:54:10; V's spec reviews 3f7d84b 2026-10-12T22:50:37 and 723827b 2026-10-12T23:00:53; V's keyload_options_review.md is 11a78ee "
    "2026-10-19T17:51:52, after Rev 10 (16:15:41)", M)
row(455, "next-step paragraph", "screened on ten, it binds four under the recorded recipe and none under the other", f"{nA}/10, {nS}/10", M)
row(455, "'ten seeds of it cost less than a review'", "ten seeds of it cost less than a review", "batch 21: 20 runs, about 30 h wall (README); a review's cost is not recorded", N)
row(499, "key-load test needs revision; cross-reference", "the key-load test a validity check it does not have at its budget (Section~\\ref{sec:referee})",
    "E's own status ('NOT READY FOR V'); sec:referee (tex10:366) only points on to sec:scale; V has not reviewed the draft", M,
    "the finding is E's (batch 21), not a referee review; tex10:366 says so")
row(495, "'the fifth round's screens have been recomputed there' (rev10/)", "the fifth round's screens have been recomputed there",
    "main's docs/review_2026-10/rev10/: recompute_H.py covers S87a, S87c and the recount (its committed output holds an IndexError traceback, then a rerun); "
    "no script recomputes E's batches 20-21: first_pass/second_pass rows cite E's README, two cite the JSON stores", "REMOVED",
    "was PARTIAL (true for H, not for E). 846d0c8 deletes the sentence; its replacement is the new tex101:499 row below",
    removed=(499, "and after the merge it checked the rest, the fifth round's screens included"))
row(44, "'which recomputed test_delta_recipe, the GPU run and H's screens and read the other screens' reports' (rev10/)", "every number recomputed from the records",
    f"585b666 rev10/ scripts: {[x for x in git('ls-tree', '--name-only', REV10, 'docs/review_2026-10/rev10/').split() if x.endswith('.py')]}; none for E's batches 20-21",
    M, "846d0c8 says what the checking agent recomputed (H among them) and that it read the other screens' reports (E's README among them)",
    new_frag="which recomputed \\texttt{test\\_delta\\_recipe}, the GPU run and H's screens and read the other screens' reports")
st20_runs = sum(1 for r in st20["runs"].values() if not r.get("dry") and r.get("ok"))
st21_runs = sum(1 for r in st21["runs"].values() if not r.get("dry") and r.get("ok"))
c20 = git("log", "-1", "--format=%s", "f5d3893")
row(535, "provenance (tab:batches): E, batches 20-21 (S68 readings 4acf915, complete f5d3893; S69 readings b93b73c, complete 419368a), commit 419368a, 19; 20",
    "19 (E) & S64 width, S65 depth, S66 eight streams, S67 capacity",
    f"batch 20: readings {E20_READ}, {st20_runs} runs, f5d3893 '{c20[:40]}'; batch 21: readings {E21_READ}, {st21_runs} runs, complete at {E} (last store commit "
    f"{hist(E, 'explore_out/oracle_d128_results.json')[0][0]})",
    M if (st20_runs, st21_runs) == (19, 20) and c20.startswith("Batch 20 complete") else Dg,
    "846d0c8 adds the row (tex101:540). E's S68 and S69 sit in the same table as F's S68 and S69 (tex101:542), each under its session's label",
    new_frag="20--21 (E) & S68 the sixteen-key runs continued to 86{,}400 (readings \\texttt{4acf915}; complete \\texttt{f5d3893}); S69 the $D = 128$ oracle's validity (readings \\texttt{b93b73c}; complete \\texttt{419368a}) & \\texttt{419368a} & 19; 20",
    new_line=540)

assert N_ORIG[0] == 39 and RC[V101][1] == RC[REV10][1]

# ── Revision 10.1 (846d0c8): text on H's round 5 and E's batches it added (rows new here) ──
r9d = git("log", "-1", "--format=%h %cI", "9bfa85a").split()
b19 = git("log", "-1", "--format=%h %cI %s", E19).split(" ", 2)
row(247, "'The batch completed after Revision 9's first draft (186 of 186 runs)'", "The batch completed after Revision~9's first draft (186 of 186 runs)",
    f"batch 19 complete at {b19[0]} {b19[1]} ('{b19[2][:45]}'); Revision 9's first draft 9bfa85a {r9d[1]}", M if b19[1] > r9d[1] and "186/186" in b19[2] else Dg,
    "846d0c8 names the revision ('this revision's' in tex10:245 meant Revision 9's)", is_new=True)
row(370, "10.1's account: 'screened once' on three lines after S87a", "``screened once'' for the all-position trigger on three lines after S87a screened it again",
    "tex10:358, 464, 486 -> tex101:360, 468, 490, rows above", M, is_new=True)
row(370, "10.1's account: provenance rows for E and H", "and provenance rows for E, F, G and H that stopped before the fifth round",
    "tex101:540 (E, batches 20-21) and tex101:541 (H, report 9), rows above", M, is_new=True)
row(370, "10.1's list of loose items taken (H and E)", "the dose screen's ceiling; the key-load validity probabilities for fresh seeds; ``the slow start hurt''; E's S69 distinguished from F's",
    "tex101:69 and 219 (ceiling), 247 (0.16/0.03; 'hurt'; E's S69), rows above", M, is_new=True)
row(499, "conclusion: after the merge the referee checked the fifth round's screens (H's and E's here), the numbers right",
    "and after the merge it checked the rest, the fifth round's screens included, with the same outcome",
    f"check_rev10_HE at {V101}: {'check_rev10_HE.md' in git('ls-tree', '--name-only', V101, 'docs/review_2026-10/rev10/')}; every count, pair, p and reading "
    "of S87a, S87c, the recount and batches 20-21 reproduces (sections H1-E2)", M, "replaces the removed tex10:495 sentence", is_new=True)

# ── pinned branches: commits after each pin up to 846d0c8's commit time ──
T101 = int(git("show", "-s", "--format=%ct", V101).strip())
say("\n" + "=" * 110)
say("PINNED BRANCHES: commits after each pin up to 846d0c8's commit time (2026-10-19 18:33:26 UTC)")
say("=" * 110)
for br, pin in BRANCH_PINS.items():
    lst = [l.split(" ", 2) for l in git("log", "--first-parent", "--format=%h %ct %s", f"{pin}..{br}").split("\n") if l.strip()]
    mv = [(h, datetime.datetime.fromtimestamp(int(t), datetime.timezone.utc).strftime("%H:%M:%S"), m_[:60]) for h, t, m_ in lst if int(t) <= T101]
    say(f"  {br.split('/', 1)[1]} pinned {pin}: {len(mv)} commit(s) by 18:33:26 UTC {mv[::-1]}; {sum(1 for h, t, m_ in lst if int(t) > T101)} after")

# ── print rows ──
say("\n" + "=" * 110)
say("ROWS")
say("=" * 110)
from collections import Counter  # noqa: E402
K = (Dg, P, N, M, "REMOVED")
cnt = Counter(r[4] for r in ROWS)
cnt_old = Counter(r[6] for r in ROWS if r[6] != "\u2014")
cnt_same = Counter(r[4] for r in ROWS if r[6] != "\u2014")
say("| # | where | item | draft | records | status | v585b666 \u2192 v846d0c8 | note |")
say("|---|---|---|---|---|---|---|---|")
cell = lambda x: str(x).replace("|", "\\|").replace("\n", " ")  # noqa: E731
for i, r in enumerate(ROWS, 1):
    say(f"| {i} | {cell(r[0])} | {cell(r[1])} | {cell(r[2])} | {cell(r[3])} | **{r[4]}** | {cell(r[7])} | {cell(r[5])} |")
assert [cnt_old[k] for k in (M, P, Dg, N)] == [24, 10, 3, 2]
say(f"\nrows (original, 585b666) {sum(cnt_old.values())}: " + ", ".join(f"{k} {cnt_old[k]}" for k in K[:4]))
say(f"rows (the same {sum(cnt_old.values())} against 846d0c8): " + ", ".join(f"{k} {cnt_same[k]}" for k in K))
say(f"rows (846d0c8, all {len(ROWS)}): " + ", ".join(f"{k} {cnt[k]}" for k in K))
say("status changes: " + "; ".join(f"row {i}: {r[7]}" for i, r in enumerate(ROWS, 1) if r[6] != "\u2014" and r[6] != r[4]))
say(f"order: H readings before records {H_ORDER}; E batch 20 {E20_ORDER}; E batch 21 {E21_ORDER}; H seeds only in H's S87 stores {H_SEEDS}; "
    f"code SHAs recomputed H {H_SHA_OK}, E b20 {sha20 == st20['meta']['code_shas'][0]}, b21 {sha21 == st21['meta']['code_shas'][0]}")
say(f"rlib ties: {len(rlib.TIES)}")
