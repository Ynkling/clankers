#!/usr/bin/env python3
"""check_rev10_FG_v101.py — Referee V: check_rev10_FG re-run against Revision 10.1 (main line 846d0c8,
multichannel_hebbian_report_v10.tex, 596 lines, cited tex101:NNN). The original, check_rev10_FG.py, read Revision 10
(585b666, 591 lines, tex10:NNN). Scope as the original: Session F's S88 (report 8) and Session G's S87 (report 7)
against their records, and every statement the revision makes about them; plus every line on them that 846d0c8 changed
or added (the sections F-tex101-new and G-tex101-new).

What changed against the original:
  * The tex is read at 846d0c8. Every tex row is anchored on its tex10 line at 585b666 (as the original asserted) and on
    the line it became at 846d0c8 (difflib's line map; asserted). A fragment 846d0c8 rewrote is re-anchored on its new
    wording and judged against it; a sentence 846d0c8 deleted is reported REMOVED with the row that checks its
    replacement.
  * Branches are pinned to what the original read (its run, 19 Oct 18:03:29-18:06:06 UTC, from the reflog of this
    clone's remote-tracking refs): explore-F 698a20f, and the 13 remote branches the seed scans cover (BRANCH_PINS).
  * Every record recomputation is kept as it was. Each row carries the original's status (read from check_rev10_FG.md
    at b09ce24, matched by section, tex10 location and statement) and the status against 846d0c8.

Read-only. Every input is a git object (`git show <ref>:<path>`, `git log`, `git ls-tree`); nothing is run but this file
(no training, no session code). Python stdlib + docs/review_2026-10/rlib.py. Prints markdown tables to stdout.

Refs:
  F  claude/explore-F at a199d5e (report 8); readings commit 40087cb.
  G  claude/explore-G at e46799a (report 7); spec 3c89f87, start 02b3eed.
  M  the main line at 585b666 (multichannel_hebbian_report_v10.tex, cited tex10:NNN; rlib.py).

Outcomes:
  S88a  the screen's own: test_stream_recipe.outcome == "BOUND ROUTED" (explore_f_write.py:37-39), read from the record's
        'outcome' field; the main line's: rlib.outcome_k (bound from the curve, one-to-one map recomputed from
        stream_gate, every stream >= 0.9). Both are printed; a difference would be a row.
  S88b  BOUND (explore_f_horizon.py:33): rlib.bound (first evaluation >= 0.95 that holds to the end).
  S87   ROUTED-BOUND (explore_g_rhl_parent.py:6-17): BOUND (rlib.transition of the curve) and end margin >= 0.9 and end
        eta^2 by stream at KEY > 0.9; failure class rlib.fail_class (test_router_layout v1), MERGED if STREAM-PARTIAL and
        the L3 VAL map is not one-to-one. 'beats' = one-sided exact McNemar p < 0.05 (rlib.mcnemar1).
"""
import collections
import datetime
import difflib
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from rlib import (load, runs, transition, bound, outcome_k, fail_class, mcnemar1, wilson, band, ROOT)  # noqa: E402

F, F_READ = "a199d5e", "40087cb"
G, G_SPEC, G_START = "e46799a", "3c89f87", "02b3eed"
M, TEXF = "585b666", "multichannel_hebbian_report_v10.tex"
V101 = "846d0c8"
F_TIP = "698a20f"            # origin/claude/explore-F as the original read it
# the 13 remote branches at the original's run (reflog, 19 Oct 18:05:30 UTC)
BRANCH_PINS = {"origin/claude/bdh-growth-hebbian-inference-w90069": "7c4ceb8", "origin/claude/bdh-repo-curl-obgdqk": "5b82707",
               "origin/claude/explore-F": "698a20f", "origin/claude/explore-G": "7ab98c1", "origin/claude/explore-H": "7820b12",
               "origin/claude/explore-I": "8d2e89f", "origin/claude/outside-ideas": "419368a", "origin/claude/relaxed-edison-he00cc": "3c2fb70",
               "origin/claude/replicate-R": "31c39fc", "origin/claude/rev10-draft": "f342d1d", "origin/claude/rev9-draft": "010a085",
               "origin/claude/review-V": "dc27968", "origin/feature/per-channel-plasticity": "4af9d50"}
ORIG_MD = ("b09ce24", "docs/review_2026-10/rev10/check_rev10_FG.md")
MAIN9C = "9c5939ef8ca8db079c87beb489b7c603169c9c57"
FD, GD = "explore_out/F/", "explore_out/G/"


def git(*a):
    # core.abbrev=7: %h and --short print 7 hex digits however large the object store grows
    return subprocess.check_output(("git", "-C", ROOT, "-c", "core.abbrev=7") + a).decode()


def show(ref, path):
    return git("show", f"{ref}:{path}")


def ctime(ref):
    return git("log", "-1", "--format=%cI", ref).strip()


def short(ref):
    return git("rev-parse", "--short=7", ref).strip()


def med(xs):
    return statistics.median(xs) if xs else None


def rng(xs, d=2):
    return f"{min(xs):.{d}f}-{max(xs):.{d}f}" if xs else "-"


def wil(k, n):
    lo, hi = wilson(k, n)
    return f"[{lo:.2f}, {hi:.2f}] {band(k, n)}"


OLD = show(M, TEXF).split("\n")
TEX = show(V101, TEXF).split("\n")
_ops = difflib.SequenceMatcher(None, OLD, TEX, autojunk=False).get_opcodes()
LMAP = {i1 + k + 1: j1 + k + 1 for op, i1, i2, j1, j2 in _ops if op in ("equal", "replace") and i2 - i1 == j2 - j1 for k in range(i2 - i1)}
LMAP.update({44: 44, 536: 541, 537: 542, 538: 543})     # the two unequal blocks: 44 -> 44-46, 536-538 -> 540-543
CHANGED = {i + 1 for op, i1, i2, _, _ in _ops if op != "equal" for i in range(i1, i2)}
PENDING = []


def tex(n, frag, new=None):
    """Assert a fragment of the draft on tex10 line n (585b666), and it (or `new`, its 846d0c8 wording) on the line it
    became at 846d0c8; return 'tex101:N'."""
    assert frag in OLD[n - 1], (n, frag)
    m = LMAP[n]
    f = frag if new is None else new
    assert f in TEX[m - 1], (n, m, f)
    PENDING.append((n, m, new is not None))
    return f"tex101:{m}"


def removed(n, frag, repl_n, repl_frag):
    """A sentence of tex10:n that 846d0c8 deleted; its replacement asserted on tex101:repl_n."""
    assert frag in OLD[n - 1] and all(frag not in t for t in TEX), (n, frag)
    assert repl_frag in TEX[repl_n - 1], (repl_n, repl_frag)
    PENDING.append((n, None, "removed"))
    return f"tex10:{n} REMOVED (replaced at tex101:{repl_n})"


def new(n, frag):
    """Text 846d0c8 added or rewrote, on tex101:n."""
    assert frag in TEX[n - 1], (n, frag)
    PENDING.append((None, n, True))
    return f"tex101:{n}"


def rep(ref, path, n, frag):
    lines = show(ref, path).split("\n")
    assert frag in lines[n - 1], (path, n, frag)
    return f"{path.split('/')[-1]}:{n}"


SECTIONS = collections.OrderedDict()


def _old_rows():
    md, out, sec = show(*ORIG_MD), {}, None
    for l in md.split("\n"):
        m = re.match(r"^### (\S+): ", l)
        if m:
            sec = m.group(1)
            continue
        m = re.match(r"^\| (.*?) \| (.*?) \| (.*) \| \*\*(MATCH|PARTIAL|DISAGREE|NOT RECORD-CHECKABLE)\*\* \| ", l)
        if m and sec:
            out[(sec, m.group(1), m.group(2))] = (m.group(4), m.group(3))
    return out


OLDROWS = _old_rows()
SEEN = set()
RECDIFF = []


def cell(x):
    return str(x).replace("|", "\\|").replace("\n", " ")


def row(sec, where, statement, recomputed, status, note="", is_new=False):
    """sec: the original's section (tex10 sections keep their names); statement may be (original's, 846d0c8's)."""
    assert status in ("MATCH", "PARTIAL", "DISAGREE", "NOT RECORD-CHECKABLE", "REMOVED"), status
    anchors = list(PENDING)
    PENDING.clear()
    st_old = statement
    if isinstance(statement, tuple):
        st_old, statement = statement
    if is_new:
        ost, delta = "\u2014", f"new: {status}"
    else:
        old_where = f"tex10:{anchors[0][0]}" if anchors and anchors[0][0] is not None else where
        key = (sec, cell(old_where), cell(st_old))
        assert key in OLDROWS, key
        SEEN.add(key)
        ost, orec = OLDROWS[key]
        if orec != cell(recomputed):
            RECDIFF.append((sec, old_where))
        if any(a[2] == "removed" for a in anchors):
            delta = f"{ost} \u2192 REMOVED"
        else:
            delta = f"{ost} (unchanged)" if ost == status else f"{ost} \u2192 {status}"
            ch = [f"tex10:{a[0]}" for a in anchors if a[0] is not None and a[2] is True]
            tch = [f"tex10:{a[0]}" for a in anchors if a[0] is not None and a[2] is False and a[0] in CHANGED]
            if ch:
                delta += "; re-anchored (" + ", ".join(ch) + " rewritten)"
            elif tch:
                delta += "; line changed elsewhere (" + ", ".join(tch) + "), fragment intact"
    SECTIONS.setdefault(sec[:-5] + "tex101" if sec.endswith("-tex10") else sec, []).append((where, statement, recomputed, status, note, ost, delta))


def ok_if(cond):
    return "MATCH" if cond else "DISAGREE"


def commits(ref_from, ref_to, path):
    out = []
    for line in git("log", "--reverse", "--format=%h %cI", f"{ref_from}..{ref_to}", "--", path).split("\n"):
        if line.strip():
            h, t = line.split()
            try:
                n = len(json.loads(show(h, path))["runs"])
            except (subprocess.CalledProcessError, ValueError, KeyError):
                n = None
            out.append((h, t, n))
    return out


def scan(o, lo, hi, acc):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(k, str) and "|" in k:
                p = k.split("|")
                if len(p) > 1 and p[1].isdigit() and lo <= int(p[1]) <= hi:
                    acc.add(int(p[1]))
            if k == "seed" and isinstance(v, int) and lo <= v <= hi:
                acc.add(v)
            scan(v, lo, hi, acc)
    elif isinstance(o, list):
        for x in o:
            scan(x, lo, hi, acc)


BLOBS = {}
BRANCHES = list(BRANCH_PINS)
for b in BRANCHES:
    for line in git("ls-tree", "-r", BRANCH_PINS[b], "--", "results", "explore_out", "records").split("\n"):
        if line.strip() and line.endswith(".json"):
            meta, path = line.split("\t", 1)
            BLOBS.setdefault(meta.split()[2], []).append((b, path))
BLOB_DATA = {}
for blob in BLOBS:
    try:
        BLOB_DATA[blob] = json.loads(git("cat-file", "-p", blob))
    except ValueError:
        pass


def seed_hits(lo, hi):
    hits = {}
    for blob, d in BLOB_DATA.items():
        a = set()
        scan(d, lo, hi, a)
        if a:
            for b, p in BLOBS[blob]:
                hits.setdefault(p, set()).update(a)
    return {p: (min(s), max(s), len(s)) for p, s in hits.items()}


def first_seen(path, branch):
    """Earliest commit on a branch that holds the path."""
    out = git("log", "--reverse", "--format=%h %cI", branch, "--", path).split("\n")
    return out[0] if out and out[0] else None


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# SESSION F, S88
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
R8 = FD + "report_8.md"
W = load(F, FD + "write_results.json")
H = load(F, FD + "horizon_results.json")
WA = {a: runs(W, a) for a in ("DELTA_FIXED", "DELTA_LEARNED", "DELTA_LEARNED_SLOW", "O_DELTA", "O_DELTA_LEARNED")}
HA = {a: runs(H, a) for a in ("FIXED_1X", "ROUTED_1X", "FIXED_2X", "ROUTED_2X", "O_FIXED_1X", "O_FIXED_2X")}
LOGA, LOGB = show(F, FD + "s88a.log"), show(F, FD + "s88b.log")

# ── order ────────────────────────────────────────────────────────────────────────────────────────────────────────
S = "F-order"
code_files = ["explore_f_write.py", "explore_f_write_child.py", "explore_f_horizon.py", "explore_f_horizon_child.py",
              "explore_f_reports5.py", "explore_f_delta_checks.py"]
hist = {f: [l for l in git("log", "--format=%h %cI", F, "--", f).split("\n") if l] for f in code_files}
changed_after = git("diff", "--name-only", F_READ, F, "--", ".", ":!explore_out").split()
wc, hc = commits(F_READ, F, FD + "write_results.json"), commits(F_READ, F, FD + "horizon_results.json")
w_first = next((c for c in wc if c[2]), None)
h_first = next((c for c in hc if c[2]), None)
row(S, f"{rep(F, R8, 3, 'Readings committed at 40087cb before any run')}",
    "readings committed at 40087cb before any run",
    f"40087cb {ctime(F_READ)} adds explore_f_write(.py, _child), explore_f_horizon(.py, _child), explore_f_reports5, "
    f"edits explore_f_delta_checks; each file's last commit on the branch: "
    + "; ".join(f"{f} {hist[f][0].split()[0]}" for f in code_files)
    + f"; files outside explore_out changed 40087cb..a199d5e: {changed_after or 'none'}; S88a log start "
    f"{re.search(r'=== S88a start (.*)', LOGA).group(1)} (banner git {re.search(r'git (\S+) \| python', LOGA).group(1)}); "
    f"first S88a records at {w_first[0]} ({w_first[1]}, {w_first[2]} records); first S88b records at {h_first[0]} "
    f"({h_first[1]}, {h_first[2]} records); 40087cb ancestor of both: "
    f"{all(subprocess.run(['git', '-C', ROOT, 'merge-base', '--is-ancestor', F_READ, c[0]]).returncode == 0 for c in (w_first, h_first))}",
    "MATCH" if not changed_after and all(hist[f][0].startswith(F_READ) for f in code_files) else "DISAGREE",
    "the readings precede every record and no code file changed after them")


def code_sha_from_log(log, ref):
    m = re.search(r"code SHA \S+: (\w+) over (\d+) modules \((.*?)\)\n", log)
    rows_ = []
    for n in sorted(m.group(3).split(", "), key=lambda x: x.replace("@main", "")):
        nm = n.replace("@main", "")
        src = show(MAIN9C, nm + ".py") if n.endswith("@main") else show(ref, nm + ".py")
        rows_.append([nm, hashlib.sha1(src.encode()).hexdigest(), "main@9c5939e" if n.endswith("@main") else "branch"])
    return m.group(1), int(m.group(2)), hashlib.sha1(json.dumps(rows_).encode()).hexdigest()


sa, na, ra = code_sha_from_log(LOGA, F_READ)
sb, nb, rb_ = code_sha_from_log(LOGB, F_READ)
rec_sha_a = sorted({r["code_sha"][:12] for a in WA.values() for r in a.values()})
rec_sha_b = sorted({r["code_sha"][:12] for a in HA.values() for r in a.values()})
row(S, rep(F, R8, 7, "S88a 4473fffc3608"), "code SHAs S88a 4473fffc3608, S88b e9c41e74d945",
    f"records {rec_sha_a} / {rec_sha_b}; recomputed from the committed sources at 40087cb (explore_common16.code_sha's rule: "
    f"SHA-1 over [name, sha1(source), origin] of the {na} / {nb} modules the log lists, '@main' ones from 9c5939e): "
    f"{ra[:12]} / {rb_[:12]}",
    ok_if(rec_sha_a == [ra[:12]] and rec_sha_b == [rb_[:12]]),
    "the code that ran is the code committed with the readings (the logs' '+dirty' did not touch an executed module)")

# ── seeds ────────────────────────────────────────────────────────────────────────────────────────────────────────
S = "F-report8"
hits_f = seed_hits(2400, 2459)
own_f = {FD + "write_results.json", FD + "horizon_results.json"}
other_f = {p: v for p, v in hits_f.items() if p not in own_f}
row(S, rep(F, R8, 8, "Seeds 2400-2409 and 2420-2429"), "seeds 2400-2409 and 2420-2429; the block appears in no record on any branch",
    f"S88a seeds {sorted({s for a in ('DELTA_FIXED', 'DELTA_LEARNED', 'DELTA_LEARNED_SLOW') for s in WA[a]})[0]}-"
    f"{sorted({s for a in ('DELTA_FIXED', 'DELTA_LEARNED', 'DELTA_LEARNED_SLOW') for s in WA[a]})[-1]}, S88b "
    f"{min(s for a in HA.values() for s in a)}-{max(s for a in HA.values() for s in a)}; block 2400-2459 in record keys "
    f"'ARM|seed|...' and 'seed' fields over {len(BLOB_DATA)} JSON blobs on {len(BRANCHES)} remote branches: only in "
    f"{sorted(hits_f)}" + (f"; others: {other_f}" if other_f else ""),
    ok_if(not other_f), "scan at today's heads (it covers the report's time too)")

allF = [r for a in list(WA.values()) + list(HA.values()) for r in a.values()]
row(S, rep(F, R8, 7, "Torch 2.14.0+cu130"), "torch 2.14.0+cu130; main-line modules at 9c5939e; CPU Xeon @ 2.80GHz",
    f"torch {sorted({r['torch'] for r in allF})}; main_sha {sorted({r.get('main_sha', '-')[:7] for a in WA.values() for r in a.values()})}; "
    f"cpu {sorted({r['cpu'] for r in allF})}", "MATCH")
row(S, rep(F, R8, 8, "78/78 runs ok"), "78/78 runs ok, none retried, none diverged",
    f"{sum(r.get('ok', False) for r in allF)}/{len(allF)} ok; attempts {sorted({r['attempts'] for r in allF})}; "
    f"errors {[r.get('error') for r in allF if r.get('error')]}; S88a {sum(len(a) for a in WA.values())}, S88b {sum(len(a) for a in HA.values())}",
    ok_if(len(allF) == 78 and all(r.get("ok") for r in allF) and {r["attempts"] for r in allF} == {1}))


def br_screen(r):
    return r.get("outcome") == "BOUND ROUTED"


# rlib vs the screen's outcome on every S88a run
diff_out = [(a, s) for a, rr in WA.items() for s, r in rr.items() if (outcome_k(r) == "BOUND ROUTED") != br_screen(r)]
trans_ok = all(transition(r["curve"]) == r["transition"] for r in allF)
FX = WA["DELTA_FIXED"]
SEEDS_A = sorted(FX)
cnt = {a: sum(br_screen(r) for r in WA[a].values()) for a in WA}
pr = {}
for a in ("DELTA_LEARNED", "DELTA_LEARNED_SLOW"):
    b = sum(1 for s in SEEDS_A if br_screen(WA[a][s]) and not br_screen(FX[s]))
    c = sum(1 for s in SEEDS_A if br_screen(FX[s]) and not br_screen(WA[a][s]))
    pr[a] = (b, c, mcnemar1(c, b))


def trs(rr, f):
    t = sorted(r["transition"] for r in rr.values() if f(r))
    return f"{med(t):.0f} ({t[0]}-{t[-1]})" if t else "-"


def splits(rr):
    fired = [c for r in rr.values() for c in r.get("checks", []) if c.get("fired")]
    return len(fired), sum(bool(c.get("key_ok")) for c in fired)


rep_rows = [(18, "DELTA_FIXED", "10/10", "[0.72, 1.00] RELIABLE", None, "13200 (9600-36000)", (8, 2)),
            (19, "DELTA_LEARNED", "7/10", "[0.40, 0.89] MAJORITY", "0 / 3; p = 0.125", "33600 (12000-43200)", (26, 10)),
            (20, "DELTA_LEARNED_SLOW", "3/10", "[0.11, 0.60] MINORITY", "0 / 7; p = 0.0078", "28800 (20400-34800)", (27, 11))]
for n, a, c_, w_, p_, t_, s_ in rep_rows:
    k = cnt[a]
    got_t = trs(WA[a], br_screen)
    got_s = splits(WA[a])
    pstr = f"{pr[a][0]} / {pr[a][1]}; p = {pr[a][2]:.4g}" if a in pr else "-"
    good = (f"{k}/10" == c_ and wil(k, 10) == w_ and got_t == t_ and got_s == s_
            and (p_ is None or (pr[a][0], pr[a][1]) == tuple(int(x) for x in p_.split(";")[0].split(" / "))))
    row(S, rep(F, R8, n, a.split("_")[-1] if a != "DELTA_FIXED" else "DELTA_FIXED"),
        f"{a}: BOUND ROUTED {c_} {w_}; vs FIXED {p_ or '-'}; transitions {t_}; splits {s_[0]} ({s_[1]} labelled ok)",
        f"{k}/10 {wil(k, 10)} (rlib.outcome_k agrees on every S88a run: {not diff_out}); {pstr}; transitions {got_t}; "
        f"splits {got_s[0]} ({got_s[1]} labelled ok)", ok_if(good),
        "p(FIXED > arm) = P(Bin(b+c, 1/2) >= c)" if a in pr else "recorded transition = rlib.transition(curve) on every S88 run: "
        + str(trans_ok))

vo = {s: transition(r["curve"]) for s, r in WA["O_DELTA"].items()}
vl = {s: transition(r["curve"]) for s, r in WA["O_DELTA_LEARNED"].items()}
row(S, rep(F, R8, 22, "VALID"), "validity: the perfect gate with beta = 1 bound on 2400 and 2401 at 1200; VALID", f"O_DELTA {vo}",
    ok_if(vo == {2400: 1200, 2401: 1200}))
row(S, rep(F, R8, 23, "bound at 30000 and 21600"), "the learned-beta perfect gate bound at 30000 and 21600", f"O_DELTA_LEARNED {vl}",
    ok_if(vl == {2400: 30000, 2401: 21600}))
fails = [(a, s, outcome_k(r), r.get("outcome"), r.get("outcome_v2"), r["acc"])
         for a in ("DELTA_LEARNED", "DELTA_LEARNED_SLOW") for s, r in sorted(WA[a].items()) if not br_screen(r)]
sp = [f for f in fails if f[2] == "STREAM-PARTIAL one-to-one"]
mg = [f for f in fails if f[2].startswith("MERGED")]
row(S, rep(F, R8, 24, "0.31-0.78 (9 runs)"),
    "every learned-arm failure: a one-to-one map stuck at 0.31-0.78 (9 runs) or one two-stream merge (DELTA_LEARNED 2405); v1 and v2 agree",
    f"{len(sp)} STREAM-PARTIAL one-to-one, acc {rng([f[5] for f in sp])}; merges {[(f[0], f[1], f[2], f[5]) for f in mg]}; "
    f"v1 = v2 on all: {all(f[3] == f[4] for f in fails)}; rlib = record: {all(f[2] == f[3] for f in fails)}",
    ok_if(len(sp) == 9 and rng([f[5] for f in sp]) == "0.31-0.78" and [(f[0], f[1]) for f in mg] == [("DELTA_LEARNED", 2405)]))

# S88b counts
B = {a: {s: bound(r) for s, r in HA[a].items()} for a in HA}
SB = sorted(HA["FIXED_1X"])


def bc(x, y):        # b = x alone, c = y alone
    return (sum(1 for s in SB if B[x][s] and not B[y][s]), sum(1 for s in SB if B[y][s] and not B[x][s]))


w1, w2 = bc("ROUTED_1X", "FIXED_1X"), bc("ROUTED_2X", "FIXED_2X")
nB = {a: sum(B[a].values()) for a in B}
row(S, rep(F, R8, 31, "9/10 [0.60, 0.98] RELIABLE"),
    "1x: FIXED 9/10 [0.60, 0.98] RELIABLE, ROUTED 6/10 [0.31, 0.83] MAJORITY, ROUTED only / FIXED only 0 / 3 (p = 0.125)",
    f"FIXED {nB['FIXED_1X']}/10 {wil(nB['FIXED_1X'], 10)}, ROUTED {nB['ROUTED_1X']}/10 {wil(nB['ROUTED_1X'], 10)}; "
    f"{w1[0]} / {w1[1]}, p(FIXED > ROUTED) {mcnemar1(w1[1], w1[0]):.4g}; record 'bound' = rlib.bound on every S88b run: "
    f"{all(r['bound'] == bound(r) for a in HA.values() for r in a.values())}",
    ok_if((nB["FIXED_1X"], nB["ROUTED_1X"], w1) == (9, 6, (0, 3))))
row(S, rep(F, R8, 32, "4/10 [0.17, 0.69] MINORITY"),
    "2x: FIXED 4/10 [0.17, 0.69] MINORITY, ROUTED 5/10 [0.24, 0.76] MAJORITY, 3 / 2 (p = 0.5)",
    f"FIXED {nB['FIXED_2X']}/10 {wil(nB['FIXED_2X'], 10)}, ROUTED {nB['ROUTED_2X']}/10 {wil(nB['ROUTED_2X'], 10)}; "
    f"{w2[0]} / {w2[1]}, p(ROUTED > FIXED) {mcnemar1(*w2):.4g}", ok_if((nB["FIXED_2X"], nB["ROUTED_2X"], w2) == (4, 5, (3, 2))))
vB = {a: {s: transition(r["curve"]) for s, r in HA[a].items()} for a in ("O_FIXED_1X", "O_FIXED_2X")}
row(S, rep(F, R8, 34, "VALID"), "validity: the perfect gate (k = 2) bound at 1200 on both seeds at both lengths", str(vB),
    ok_if(all(v == 1200 for d in vB.values() for v in d.values()) and all(len(d) == 2 for d in vB.values())))


def drop(x):
    a, b = B[f"{x}_1X"], B[f"{x}_2X"]
    return sum(1 for s in SB if a[s] and not b[s]) - sum(1 for s in SB if b[s] and not a[s])


dF, dR = drop("FIXED"), drop("ROUTED")
row(S, rep(F, R8, 35, "drop_FIXED = 5"), "drop_FIXED = 5, drop_ROUTED = 1", f"{dF}, {dR}", ok_if((dF, dR) == (5, 1)))

# readings
best = max(("DELTA_LEARNED", "DELTA_LEARNED_SLOW"), key=lambda a: (cnt[a], -pr[a][1]))
keeps = pr[best][1] - pr[best][0] <= 1
costs = all(pr[a][1] >= 4 and pr[a][0] == 0 for a in pr)
row(S, rep(F, R8, 39, "no reading applies"), "S88a: 'no reading applies'; 'keeps' fails (c - b = 3 > 1); 'costs' fails: SLOW 7 vs 0 meets it, LEARNED 3 vs 0 does not",
    f"better learned arm {best} (c - b = {pr[best][1] - pr[best][0]}): keeps {keeps}; costs: "
    + ", ".join(f"{a} c {pr[a][1]}, b {pr[a][0]} -> {pr[a][1] >= 4 and pr[a][0] == 0}" for a in pr) + f" -> {costs}",
    ok_if(not keeps and not costs and best == "DELTA_LEARNED"), "explore_f_write.py:41-47")
untested = nB["FIXED_1X"] < 8 or nB["ROUTED_1X"] < 8
ext = w2[0] >= 4 and w2[1] == 0
dec = dF >= 4 and abs(dR - dF) <= 1
row(S, rep(F, R8, 44, "untested"), "S88b: 'untested' (ROUTED 6/10 at 1x < 8/10); had it been tested, neither 'extends' (b 3, c 2) nor 'the decay's' (|1 - 5| = 4 > 1)",
    f"untested {untested}; extends {ext}; the decay's {dec} (|drop_R - drop_F| = {abs(dR - dF)})", ok_if(untested and not ext and not dec),
    "explore_f_horizon.py:37-45")

# beta
LRN = ("DELTA_LEARNED", "DELTA_LEARNED_SLOW", "O_DELTA_LEARNED")


def bt(r):
    return {int(k): v for k, v in r["beta_traj"].items()}


def first_below(r, th=0.1):
    t = bt(r)
    for st in sorted(t):
        if all(L["median"] < th for L in t[st]):
            return st
    return None


fb = {a: {s: first_below(r) for s, r in WA[a].items()} for a in LRN}
m1200 = [L["median"] for r in WA["DELTA_LEARNED"].values() for L in bt(r)[1200]]
row(S, rep(F, R8, 52, "medians of 0.3-0.9 by 1200"), "DELTA_LEARNED: in all 10 runs beta falls from 1 to medians of 0.3-0.9 by 1200 and below 0.1 in every layer by 3600",
    f"medians at 1200 (3 layers x 10 runs) {rng(m1200)}; first evaluation with every layer's median < 0.1: {sorted(fb['DELTA_LEARNED'].values())}",
    ok_if(rng(m1200, 1) == "0.3-0.9" and max(fb["DELTA_LEARNED"].values()) == 3600))
slow_hold = [L["median"] for r in WA["DELTA_LEARNED_SLOW"].values() for st in (1200, 2400) for L in bt(r)[st]]
fbs = sorted(fb["DELTA_LEARNED_SLOW"].values())
row(S, rep(F, R8, 53, "holds at 0.6-1.0"), "SLOW: beta holds at 0.6-1.0 while w, b are at 1e-4; collapses within 1200-6000 updates of the switch at 2400 (below 0.1 by 3600-8400)",
    f"medians at 1200 and 2400 {rng(slow_hold)}; first all-layer < 0.1 at {fbs} (switch + {sorted(x - 2400 for x in fbs)})",
    ok_if(rng(slow_hold, 1) == "0.6-1.0" and (fbs[0], fbs[-1]) == (3600, 8400)))
row(S, rep(F, R8, 55, "same collapse"), "the learned-beta perfect gate shows the same collapse; 'that is why it binds at 21600-30000'",
    f"O_DELTA_LEARNED first all-layer median < 0.1 at {fb['O_DELTA_LEARNED']} (lr 1e-3 from update 1; transitions {vl})",
    "PARTIAL", f"the collapse is there but slower (by {sorted(fb['O_DELTA_LEARNED'].values())}, against 3600 in DELTA_LEARNED with the same lr); 'that is why' is a mechanism, not a reading")
end = {a: {s: bt(r)[max(bt(r))] for s, r in WA[a].items()} for a in LRN}
ctx_max = max((L["by_role"]["ctx"], a, s, i + 1) for a in LRN for s, e in end[a].items() for i, L in enumerate(e))
key_max = max((L["by_role"]["key"], a, s, i + 1) for a in LRN for s, e in end[a].items() for i, L in enumerate(e))
ctx_hi = [(a, s, f"L{i + 1}", round(L["by_role"]["ctx"], 2)) for a in LRN for s, e in end[a].items() for i, L in enumerate(e)
          if L["by_role"]["ctx"] > 0.15]
row(S, rep(F, R8, 56, "about 0 at CTX and KEY positions in every layer"),
    "at the end beta is about 0 at CTX and KEY positions in every layer (KEY at most 0.15)",
    f"KEY max {key_max[0]:.3f} ({key_max[1]} {key_max[2]} L{key_max[3]}); CTX max {ctx_max[0]:.3f} ({ctx_max[1]} {ctx_max[2]} L{ctx_max[3]}); "
    f"CTX > 0.15: {ctx_hi}", "PARTIAL" if ctx_hi else "MATCH",
    "KEY holds; CTX does not in one run (DELTA_LEARNED 2403, layer 3, median 0.72 at the end)")
l1v = {(a, s): e[0]["by_role"]["val"] for a in LRN for s, e in end[a].items()}
l1v20 = [v for (a, s), v in l1v.items() if a != "O_DELTA_LEARNED"]
row(S, rep(F, R8, 57, "median 0.03-1.87 across runs"), "at VAL positions in layer 1 beta stays of order 1 (median 0.03-1.87 across runs)",
    f"layer-1 VAL medians {rng(list(l1v.values()))} over {len(l1v)} learned runs; >= 0.5 in {sum(v >= 0.5 for v in l1v20)}/20 "
    f"screen runs (+{sum(v >= 0.5 for (a, _), v in l1v.items() if a == 'O_DELTA_LEARNED')}/2 perfect-gate); < 0.2 in "
    f"{sorted((a[-4:], s, round(v, 2)) for (a, s), v in l1v.items() if v < 0.2)}", "PARTIAL",
    "the range is right; 'stays of order 1' holds in 13 of the 20 screen runs, the range's own lower end is 0.03")
l23 = [(a, s, f"L{i + 1}", round(L["by_role"]["val"], 2)) for a in LRN for s, e in end[a].items() for i, L in enumerate(e)
       if i > 0 and L["by_role"]["val"] >= 0.5]
fr1 = max(e[0]["frac_gt1"] for a in LRN for e in end[a].values())
row(S, rep(F, R8, 58, "Up to 33% of layer-1 positions"), "up to 33% of layer-1 positions have beta > 1, consistent with them being the VAL positions (a third)",
    f"layer-1 frac_gt1 max {fr1:.3f}", "MATCH" if round(fr1, 2) == 0.33 else "DISAGREE",
    "'consistent with VAL positions' is an interpretation; the record has no per-role fraction > 1")
divN = {a: sum(any(L["eta_key_by_stream"] >= 0.5 for L in e) for e in end[a].values()) for a in LRN}
div_any = {a: sum(any(L["eta_key_by_stream"] >= 0.5 for st in bt(r).values() for L in st) for r in WA[a].values()) for a in LRN}
eta_max = max(L["eta_key_by_stream"] for a in LRN for r in WA[a].values() for st in bt(r).values() for L in st)
spr = max(max(L["key_by_stream"]) - min(L["key_by_stream"]) for a in LRN for e in end[a].values() for L in e)
row(S, rep(F, R8, 62, "0/10, 0/10 and 0/2"), "beta does not diverge by stream: 0/10, 0/10, 0/2 reach eta^2 >= 0.5 at KEY; 8 per-stream medians within 0.04 in every run",
    f"at the end {divN}; at any evaluation {div_any} (max eta^2 {eta_max:.2f}); end spread max {spr:.3f}",
    ok_if(all(v == 0 for v in divN.values()) and round(spr, 2) <= 0.04),
    "at the end, the screen's definition (explore_f_write.py:50-52); at earlier evaluations layer 1 passes 0.5 in some runs (tex10:266 row)")
sF, sL, sS = splits(FX), splits(WA["DELTA_LEARNED"]), splits(WA["DELTA_LEARNED_SLOW"])
row(S, rep(F, R8, 64, "The splits misfire more often"),
    "the splits misfire more often: labelled ok on 10/26 and 11/27 (DELTA_FIXED 2/8); they fire about three times as often",
    f"misfires (not labelled ok) FIXED {sF[0] - sF[1]}/{sF[0]} ({(sF[0] - sF[1]) / sF[0]:.0%}), LEARNED {sL[0] - sL[1]}/{sL[0]} "
    f"({(sL[0] - sL[1]) / sL[0]:.0%}), SLOW {sS[0] - sS[1]}/{sS[0]} ({(sS[0] - sS[1]) / sS[0]:.0%}); firings x{sL[0] / sF[0]:.2f} / x{sS[0] / sF[0]:.2f}",
    "PARTIAL", "more misfires in number (16 and 16 against 6) but a smaller share (62% and 59% against 75%): as a rate the learned "
    "arms' splits misfire less often; 'because the learned runs plateau longer' is a mechanism, not recorded")
slower = {a: [(s, WA[a][s]["transition"], FX[s]["transition"]) for s in SEEDS_A if br_screen(WA[a][s])] for a in pr}
row(S, rep(F, R8, 67, "every learned variant is"), "with beta fixed at 1 the recipe is the strongest arm (10/10, median 13200); every learned variant is slower or fails",
    "per seed, learned transition vs FIXED's where the learned run binds: "
    + "; ".join(f"{a[6:]} slower on {sum(t > f for _, t, f in v)}/{len(v)}" for a, v in slower.items()),
    "MATCH" if all(t > f for v in slower.values() for _, t, f in v) else "PARTIAL",
    "true of the arms (medians 33600 and 28800 against 13200); per seed one learned run bound before its fixed pair: "
    + str([(a, s, t, f) for a, v in slower.items() for s, t, f in v if t <= f]) + "; marked post hoc in the report")

# S88b descriptive
ub = [(a, s, r) for a in ("FIXED_1X", "ROUTED_1X", "FIXED_2X", "ROUTED_2X") for s, r in HA[a].items() if not bound(r)]
bd = [(a, s, r) for a in ("FIXED_1X", "ROUTED_1X", "FIXED_2X", "ROUTED_2X") for s, r in HA[a].items() if bound(r)]


def last_bd(r):
    ab = r["acc_by_distance"]
    return {int(d): v[1] for d, v in ab[max(ab, key=int)].items()}


ub_acc = [r["acc"] for _, _, r in ub]
ub_dist = [v for _, _, r in ub for v in last_bd(r).values()]
ub_wo = [x for _, _, r in ub for x in (r["write_order"]["acc_last"], r["write_order"]["acc_notlast"])]
row(S, rep(F, R8, 74, "0.48-0.51 accuracy"),
    "every unbound run ends at 0.48-0.51 accuracy; that holds equally at every distance and whether or not the queried pair was its key's last write",
    f"{len(ub)} unbound runs: end acc {rng(ub_acc, 3)}; by distance (last evaluation) {rng(ub_dist, 3)}; last-write / not {rng(ub_wo, 3)}",
    "PARTIAL", f"the overall accuracy is 0.48-0.51, but split by distance it spans {rng(ub_dist)} and by write order {rng(ub_wo)}: near chance "
    "between two streams in every cell, not 0.48-0.51 in each")
conf = [r["write_order"]["acc_notlast"] + r["write_order"]["lastval_notlast"] for _, _, r in ub]
row(S, rep(F, R8, 76, "stream confusion"), "the channel retrieves the key's two values but cannot tell which stream: a stream confusion",
    f"unbound runs, queries whose pair was not its key's last write: correct + 'answered the other stream's (last) value' = {rng(conf, 3)}",
    "MATCH", "the record's write_order shows nearly every such answer is one of the key's two values")
bd_dist = [v for _, _, r in bd for v in last_bd(r).values()]
row(S, rep(F, R8, 78, "0.99-1.00 at every distance"), "bound runs are at 0.99-1.00 at every distance", f"{len(bd)} bound runs: {rng(bd_dist, 3)}",
    ok_if(rng(bd_dist) in ("0.99-1.00", "1.00-1.00")))
dbr = {a: [r["decay_by_role"] for r in HA[a].values()] for a in ("ROUTED_1X", "ROUTED_2X")}


def dmed(a, ly, role):
    v = [d[ly][role] for d in dbr[a] if role in d[ly]]
    return med(v)


Dm = {a: med([r["decay_params"]["Delta"] for r in HA[a].values()]) for a in dbr}
kvf = [dmed(a, ly, ro) for a in dbr for ly in range(3) for ro in ("key", "val", "fill") if dmed(a, ly, ro) is not None]
row(S, rep(F, R8, 80, "to about"), "the learned decay moves Delta from -2.60 to about -3.0; alpha 0.98-1.00 at KEY / VAL / FILL; 1x CTX layer 1 0.89; 2x CTX ~1",
    f"Delta medians {', '.join(f'{a} {v:.3f}' for a, v in Dm.items())}; alpha medians at KEY/VAL/FILL {rng(kvf, 3)}; CTX L1 1x {dmed('ROUTED_1X', 0, 'ctx'):.3f}, "
    f"2x {dmed('ROUTED_2X', 0, 'ctx'):.3f}, 2x CTX L1-L3 {[round(dmed('ROUTED_2X', i, 'ctx'), 3) for i in range(3)]}",
    ok_if(round(min(kvf), 2) >= 0.98 and round(dmed("ROUTED_1X", 0, "ctx"), 2) == 0.89), "medians over runs")

# runtime
sa_h = sum(r["secs_wall"] for a in WA.values() for r in a.values()) / 3600
sb_h = sum(r["secs_wall"] for a in HA.values() for r in a.values()) / 3600
row(S, rep(F, R8, 125, "159.9 h"), "S88a 159.9 h of run time, 23:07 UTC 12 Oct to 17:03 UTC 14 Oct; S88b 8.7 h, 17:03-19:26 UTC 14 Oct",
    f"sum secs_wall {sa_h:.2f} h / {sb_h:.2f} h; s88a.log '{re.search(r'=== S88a start (.*)', LOGA).group(1)}' to "
    f"'{re.search(r'=== S88a done (.*)', LOGA).group(1)}'; s88b.log to '{re.search(r'=== S88b done (.*)', LOGB).group(1)}'",
    ok_if(abs(sa_h - 159.9) < 0.06 and abs(sb_h - 8.7) < 0.06), "159.85 h printed as 159.9")
per_upd = {a: [r["secs"] / r["stopped_at"] for r in WA[a].values()] for a in WA}
lr_max_h = max(r["secs_wall"] for a in LRN for r in WA[a].values()) / 3600
row(S, rep(F, R8, 133, "0.34-1.51 s against 0.30-0.37 s"), "learned-beta runs slower per update: 0.34-1.51 s against 0.30-0.37 s; up to 15 h per run",
    "s per update (secs / stopped_at): " + "; ".join(f"{a} {rng(v)}" for a, v in per_upd.items()) + f"; longest learned run {lr_max_h:.1f} h",
    ok_if(rng(per_upd["DELTA_LEARNED"] + per_upd["DELTA_LEARNED_SLOW"]) == "0.34-1.51" and rng(per_upd["DELTA_FIXED"]) == "0.30-0.37"),
    "the denormal timing test (0.69 s vs 0.02 s) is not in a record")
row(S, rep(F, R8, 138, "complete within about 1200 updates"), "under both lr schedules the collapse is complete within about 1200 updates of beta's lr reaching 1e-3",
    f"all-layer median < 0.1 at, minus the step the lr reached 1e-3: DELTA_LEARNED (1e-3 from update 1) {sorted(fb['DELTA_LEARNED'].values())}; "
    f"SLOW (1e-3 from 2401) +{sorted(x - 2400 for x in fbs)}; O_DELTA_LEARNED {sorted(fb['O_DELTA_LEARNED'].values())}",
    "DISAGREE", "contradicts the report's own lines 52-54 (by 3600 with the lr at 1e-3 from update 1; 1200-6000 after the switch); within "
    f"1200 updates holds for {sum(x - 2400 <= 1200 for x in fbs)} of the 10 slow runs only")
nchk_a = len(re.findall(r"CHECK F/write: .*: ok\n", LOGA + "\n"))
nchk_b = len(re.findall(r"CHECK F/horizon: .*: ok\n", LOGB + "\n"))
cj = json.loads(show(F, FD + "checks_s88a_20261012_230728.json"))
row(S, rep(F, R8, 87, "CHECKs (all passed"), "CHECKs: delta_checks 26/26; S88a 11; S88b 12; all passed before any run",
    f"checks_s88a json ok {cj['ok']}, {len(cj['rows'])} rows (started {cj['provenance']['started']}); s88a.log 'CHECK F/write ... ok' {nchk_a}, "
    f"FAIL {LOGA.count(': FAIL')}; s88b.log 'CHECK F/horizon ... ok' {nchk_b}, FAIL {LOGB.count(': FAIL')}; each before '34 runs queued' / "
    f"the first S88b run: {LOGA.find('runs queued') > LOGA.rfind('CHECK F/write') and LOGB.find('runs queued') > LOGB.rfind('CHECK F/horizon')}",
    ok_if(cj["ok"] and len(cj["rows"]) == 26 and nchk_a == 11 and nchk_b == 12),
    "the bit-for-bit comparisons are read from the log, not rerun")
row(S, rep(F, R8, 109, "smoke run"), "one 1200-update smoke run of DELTA_LEARNED on seed 7 before the readings' commit (not stored)",
    f"no record with seed 7 in write_results.json: {7 not in {r['seed'] for a in WA.values() for r in a.values()}}", "NOT RECORD-CHECKABLE",
    "disclosed by the session; the readings were committed after it (nothing in the records depends on it)")

# ── Revision 10 on S88 ─────────────────────────────────────────────────────────────────────────────────────────────
S = "F-tex10"
row(S, tex(65, "a learned $\\beta$ collapses and binds $7/10$ against $10/10$ (no reading applies); the forget gate's horizon screen untested"),
    "index row F: 'a learned beta collapses and binds 7/10 against 10/10 (no reading applies); the forget gate's horizon screen untested'",
    f"DELTA_LEARNED {cnt['DELTA_LEARNED']}/10 vs FIXED {cnt['DELTA_FIXED']}/10; S88a no reading; S88b untested", "MATCH",
    "'collapses' is F's description (S88a's 'beta DIVERGES BY STREAM' is the only pre-fixed beta reading)")
row(S, tex(266, "seeds 2400--2409; readings fixed before any run"), "S88a set-up: delta recipe with the exact-copy split, eight streams, k = 16, seeds 2400-2409, readings fixed before any run",
    "explore_f_write.py:11-26 (S = 8, P = 4, k = 16, KEYMASS trigger with H's exact-copy split); readings 40087cb before the first record (F-order)", "MATCH")
row(S, tex(266, "fixed $\\mathbf{10/10}$; learned $7/10$ (0 vs 3, $p = 0.125$); learned and slow $3/10$ (0 vs 7, $p = 0.0078$)"),
    "fixed 10/10; learned 7/10 (0 vs 3, p = 0.125); slow 3/10 (0 vs 7, p = 0.0078)",
    f"{cnt['DELTA_FIXED']}/10; {cnt['DELTA_LEARNED']}/10 ({pr['DELTA_LEARNED'][0]} vs {pr['DELTA_LEARNED'][1]}, p {pr['DELTA_LEARNED'][2]:.4g}); "
    f"{cnt['DELTA_LEARNED_SLOW']}/10 ({pr['DELTA_LEARNED_SLOW'][0]} vs {pr['DELTA_LEARNED_SLOW'][1]}, p {pr['DELTA_LEARNED_SLOW'][2]:.4g})",
    ok_if((cnt["DELTA_FIXED"], cnt["DELTA_LEARNED"], cnt["DELTA_LEARNED_SLOW"]) == (10, 7, 3)),
    "'0 vs 3' is written learned-first; the p is P(FIXED > learned)")
row(S, tex(266, "Neither pre-fixed reading applies"), "neither reading applies; 'keeps' needed the better arm within one pair (three behind); 'costs' needed both >= 4 behind with none ahead (the slow one is, the other is not)",
    f"keeps {keeps} (c - b = {pr[best][1] - pr[best][0]}); costs {costs} (SLOW c 7 b 0; LEARNED c 3 b 0)", ok_if(not keeps and not costs))
row(S, tex(266, "within about 1200 updates of its learning rate reaching $10^{-3}$ (by 3600 in the first arm, 3600--8400 in the slow one",
          new="falls from 1 to below $0.1$ in every layer's median, by 3600 in the first arm (whose rate is $10^{-3}$ from the start) and 1200--6000 updates after the switch in the slow one"),
    ("'in every learned run beta falls from 1 to below 0.1 in every layer within about 1200 updates of its learning rate reaching 1e-3 (by 3600 in the first arm, 3600-8400 in the slow one ...)'",
     "'in every learned run beta falls from 1 to below 0.1 in every layer's median, by 3600 in the first arm (whose rate is 1e-3 from the start) and 1200-6000 updates after the switch in the slow one'"),
    f"first evaluation with every layer's median < 0.1: DELTA_LEARNED {sorted(fb['DELTA_LEARNED'].values())} with the lr at 1e-3 from update 1 "
    f"(3600 updates after it, not 1200; at 2400 a layer median is still up to {max(L['median'] for r in WA['DELTA_LEARNED'].values() for L in bt(r)[2400]):.2f}); SLOW {fbs} (switch at 2400: +{sorted(x - 2400 for x in fbs)}); the perfect gate {sorted(fb['O_DELTA_LEARNED'].values())}",
    "MATCH" if (max(fb["DELTA_LEARNED"].values()) == 3600 and (fbs[0] - 2400, fbs[-1] - 2400) == (1200, 6000)
                and all(v is not None for d in fb.values() for v in d.values())) else "DISAGREE",
    "846d0c8 takes the referee's wording: by 3600 in the first arm (all ten at 3600) and 1200-6000 updates after the slow arm's switch. "
    f"'In every learned run' also holds for the learned perfect gate, which falls below 0.1 at {sorted(fb['O_DELTA_LEARNED'].values())}; the "
    "sentence gives no time for it")
row(S, tex(266, "holds it at $0.6$--$1.0$ while its rate is $10^{-4}$"), "the slow arm holds beta at 0.6-1.0 while its rate is 1e-4",
    f"SLOW medians at 1200 and 2400: {rng(slow_hold)}", "MATCH")
hi_eta = [(L["eta_key_by_stream"], max(L["key_by_stream"]) - min(L["key_by_stream"]), L["median"], i)
          for a in LRN for r in WA[a].values() for st in bt(r).values() for i, L in enumerate(st) if L["eta_key_by_stream"] >= 0.5]
row(S, tex(266, "never diverges by stream", new="at the end it does not diverge by stream in any run"),
    ("beta never diverges by stream", "beta at the end does not diverge by stream in any run"), f"'DIVERGES' (eta^2 at KEY >= 0.5 in a layer) at the end {divN}; at any evaluation "
    f"{div_any} runs (max eta^2 {eta_max:.2f}; {len(hi_eta)} run-evaluations, all at layer {sorted({i + 1 for *_, i in hi_eta})}, per-stream spread "
    f"<= {max(x[1] for x in hi_eta):.3f} with the layer median <= {max(x[2] for x in hi_eta):.2f})",
    "MATCH" if all(v == 0 for v in divN.values()) else "DISAGREE",
    "846d0c8 scopes it to the end, the screen's own definition: 0 of 22 runs there. (At earlier evaluations layer 1's eta^2 at KEY passes "
    "0.5 in 10 runs, as the stream share of a beta already near 0; the sentence no longer claims those)")
l23_scr = [x for x in l23 if x[0] != "O_DELTA_LEARNED"]
key_hi = [(a, s_, i + 1) for a in LRN for s_, e in end[a].items() for i, L in enumerate(e) if L["by_role"]["key"] >= 0.5]
row(S, tex(266, "stays of order 1 only at the first layer's value positions",
           new="where it stays of order 1 it is mostly at value positions (the first layer's in 13 of 20 runs, a later layer's in 8 run-layers, and the context position in one)"),
    ("beta stays of order 1 only at the first layer's value positions",
     "where beta stays of order 1 (end median >= 0.5) it is mostly at value positions: layer 1's in 13 of 20 runs, a later layer's in 8 run-layers, the context position in one"),
    f"end medians >= 0.5: layer-1 VAL in {sum(v >= 0.5 for v in l1v20)}/20 screen runs (+{sum(v >= 0.5 for (a, _), v in l1v.items() if a == 'O_DELTA_LEARNED')}/2 perfect-gate); "
    f"VAL at layers 2-3 in {len(l23)} run-layers {l23} ({len(l23_scr)} of them screen runs); CTX {ctx_hi}; KEY {key_hi}",
    "MATCH" if (sum(v >= 0.5 for v in l1v20), len(l23), len(ctx_hi), len(key_hi)) == (13, 8, 1, 0) else "PARTIAL",
    "846d0c8 states the exceptions the original row listed. Its '13 of 20' counts the screen runs and its '8 run-layers' all 22 learned "
    f"runs; the 8 are {'all screen runs' if len(l23_scr) == len(l23) else 'not all screen runs'}")
row(S, tex(266, "a $\\beta$ near zero both erases and writes almost nothing"), "with the tied write, a beta near zero both erases and writes almost nothing",
    "explore_f_write.py:24-35 (beta scales the tied write and the erase)", "NOT RECORD-CHECKABLE", "a property of the rule, not a record")
row(S, tex(266, "bound at 21{,}600 and 30{,}000 where $\\beta = 1$ binds at 1200"), "the learned-beta perfect gate bound at 21600 and 30000 where beta = 1 binds at 1200",
    f"O_DELTA_LEARNED {vl}; O_DELTA {vo}", ok_if(sorted(vl.values()) == [21600, 30000] and set(vo.values()) == {1200}))
row(S, tex(266, "one-to-one map stuck at $0.31$--$0.78$ or one merge"), "every learned-arm failure is a one-to-one map stuck at 0.31-0.78 or one merge",
    f"{len(sp)} one-to-one at {rng([f[5] for f in sp])}; {len(mg)} merge", ok_if(len(sp) == 9 and len(mg) == 1))
row(S, tex(266, "on other layouts and schedules"), "the earlier screens in which a learned beta rose were on other layouts and schedules; the attribution of Revision 8's stall remains post hoc",
    "not S88 records (S53/S54)", "NOT RECORD-CHECKABLE", "outside this check's records")
row(S, tex(268, "ten seeds at each length (2420--2429), the perfect gate $2/2$ at 1200 at both lengths"), "S88b: Raven's routed decay initialised to equal the fixed decay; seeds 2420-2429 at each length; the perfect gate 2/2 at 1200 at both",
    f"seeds {SB[0]}-{SB[-1]} x 4 arms; perfect gate {vB}; explore_f_horizon.py:26-31 (w = 0, alpha = 0.95 at init; CHECK: frozen = FIXED bit for bit, s88b.log)",
    ok_if(all(v == 1200 for d in vB.values() for v in d.values())))
row(S, tex(268, "at the original distance fixed $9/10$, routed $6/10$ (0 vs 3); at twice it $4/10$ against $5/10$ (3 vs 2)"),
    "bound: 1x fixed 9/10, routed 6/10 (0 vs 3); 2x 4/10 against 5/10 (3 vs 2)",
    f"1x {nB['FIXED_1X']}/10, {nB['ROUTED_1X']}/10 ({w1[0]} vs {w1[1]}); 2x {nB['FIXED_2X']}/10, {nB['ROUTED_2X']}/10 ({w2[0]} vs {w2[1]})",
    ok_if((nB["FIXED_1X"], nB["ROUTED_1X"], w1, nB["FIXED_2X"], nB["ROUTED_2X"], w2) == (9, 6, (0, 3), 4, 5, (3, 2))),
    "pairs written routed-first")
row(S, tex(268, "so the screen is \\emph{untested}"), "untested by the rule; had it been tested neither 'extends' (4 vs 0 needed) nor 'the decay's' (drops 5 and 1 differ by more than one)",
    f"untested {untested}; extends {ext}; the decay's {dec}; drops {dF}, {dR}", ok_if(untested and not ext and not dec))
row(S, tex(268, "Every failure at either length is a stream confusion at $0.5$, flat across the queried key's distance to its cue"),
    "every failure is a stream confusion at 0.5, flat across distance; 'what fails is the stream's identity rather than the memory's reach'",
    f"unbound end acc {rng(ub_acc, 3)}; by distance {rng(ub_dist, 3)}; correct + other stream's value (not-last queries) {rng(conf, 3)}",
    "MATCH", f"'at 0.5' holds per run; per distance {rng(ub_dist)} (no trend with distance)." " The last clause is an interpretation of an untested screen")
r8 = sorted(p for p in git("ls-tree", "-r", "--name-only", F, FD).split() if re.search(r"report_\d+\.md$", p))
T101 = int(git("show", "-s", "--format=%ct", V101).strip())
F_AT101 = next(h for h, t in (l.split() for l in git("log", "--first-parent", "--format=%h %ct", "origin/claude/explore-F").split("\n") if l.strip()) if int(t) <= T101)
r8_101 = sorted(p for p in git("ls-tree", "-r", "--name-only", F_AT101, FD).split() if re.search(r"report_\d+\.md$", p))
prompts = show(V101, "docs/reading/parallel_sessions_prompts.md")
f_rounds = re.findall(r"^## Session F — (round \d+) \((S\d+)", prompts, re.M)
row(S, tex(268, "its branch holds eight reports", new="The two screens closed session~F's fourth round; its branch holds eight reports"),
    ("the two screens close F's programme; its branch holds eight reports", "the two screens closed session F's fourth round; its branch holds eight reports"),
    f"{len(r8)} reports at {F}; the draft {M} at {ctime(M)}; F's commits after a199d5e (to the pin {F_TIP}): "
    + "; ".join(git("log", "--format=%h %cI %s", f"{F}..{F_TIP}").strip().split("\n"))
    + f"; F at 846d0c8's commit time: {F_AT101}, {len(r8_101)} reports; the prompts file's F headings {f_rounds}; tex101:44 lists F's S88 in "
    f"'the exploratory sessions' fifth round' {'fifth round (E' in TEX[43]}",
    "PARTIAL",
    "'closed' and 'eight reports' hold (F's round 6, S89, began at 8ba99bb, 18:02, and holds no report by 18:33). 'Fourth round' is F's "
    "fourth round of screens (S53-S59, S68-S70, S80-S81, S88; F had no round 4), but the report numbers rounds project-wide: tex101:44, 46 and "
    "370 call S88 part of the fifth round, as the prompts file ('Session F — round 5 (S88') and F's own 'round 6' commits do. Exact: "
    "'closed session F's fifth round' or 'F's part of the fifth round'")
row(S, tex(377, "collapses to about zero everywhere but the first layer's value positions and binds $7/10$ against the fixed one's $10/10$",
          new="collapses to about zero except at some value positions and binds $7/10$ against the fixed one's $10/10$"),
    ("'A learned write strength, screened afterwards, collapses to about zero everywhere but the first layer's value positions and binds 7/10 against 10/10 (no reading applies)'",
     "'A learned write strength, screened afterwards, collapses to about zero except at some value positions and binds 7/10 against the fixed one's 10/10 (no reading applies)'"),
    f"counts {cnt['DELTA_LEARNED']}/10 vs {cnt['DELTA_FIXED']}/10; end medians >= 0.5 outside VAL: CTX {ctx_hi}, KEY {key_hi}", "PARTIAL" if ctx_hi else "MATCH",
    "the counts, 'no reading applies' and the value-position exceptions are right; the one context-position exception (DELTA_LEARNED 2403, "
    "layer 3, 0.72), which tex101:268 now states, is left out of this summary. Minor")
row(S, tex(495, "a learned write strength collapses on a screen"), "conclusion: 'a learned write strength collapses on a screen'",
    f"the all-position median falls below 0.1 in every layer in every learned run (first at {min(x for v in fb.values() for x in v.values())}-{max(x for v in fb.values() for x in v.values())})", "MATCH", "scoped 'on a screen'")
row(S, tex(487, "its reading on the horizon itself read ``it does not''", new="its reading on the horizon itself read ``it does not'' and the direct horizon screen (S88b) is untested by its rule, with its failures stream confusions at $0.5$ rather than faded memory"),
    ("limits, memory horizon: 'a learned forget gate raises one delta channel's binding ... its reading on the horizon itself read \"it does not\"'",
     "limits, memory horizon: ... and the direct horizon screen (S88b) is untested by its rule, its failures stream confusions at 0.5 rather than faded memory"),
    f"S88b untested {untested} (ROUTED {nB['ROUTED_1X']}/10 at 1x); unbound end acc {rng(ub_acc, 3)}, by distance {rng(ub_dist, 3)}; correct + other stream's value {rng(conf, 3)}",
    ok_if(untested), "846d0c8 adds S88b as the referee's pass asked; 'rather than faded memory' is the reading of a flat profile across distance, as at tex101:270")
row(S, tex(44, "F's S88", new="and, after the merge, of the whole revision and of the fifth round's screens (\\texttt{rev10/referee\\_pass.md}"),
    ("Revision 10 paragraph: 'the fifth round (... F's S88 ...)'; 'the fifth round's screens have not been refereed'",
     "Revision 10 paragraph: 'the fifth round (... F's S88 ...)'; the referee checked the fifth round's screens after the merge (rev10/referee_pass.md)"),
    f"at {V101}: check_rev10_FG.md and referee_pass.md in docs/review_2026-10/rev10/: "
    f"{all(x in git('ls-tree', '--name-only', V101, 'docs/review_2026-10/rev10/') for x in ('check_rev10_FG.md', 'referee_pass.md'))}", "MATCH",
    "the original row was MATCH for 'not refereed' (true at 585b666); 846d0c8 records the referee's check, which exists")
row(S, removed(495, "the fifth round's screens have been recomputed there and not yet refereed by session~V", 499,
                "and after the merge it checked the rest, the fifth round's screens included, with the same outcome"),
    "conclusion: 'the fifth round's screens have been recomputed there (docs/review_2026-10/rev10/) and not yet refereed by session V'",
    f"rev10/ at {M}: {sorted(p.split('/')[-1] for p in git('ls-tree', '-r', '--name-only', M, 'docs/review_2026-10/rev10/').split())}; "
    "no script recomputes S88 or S87; first_pass.md sections 6-7 cite record counts, several rows only 'report 8' / 'report 7'",
    "REMOVED", "was PARTIAL: the counts were read from the records; the beta and gate descriptions were taken from the reports (first_pass 6.3, "
    "7.6 'MATCH (as F's description)'). 846d0c8 deletes the sentence; its replacement is checked in F-tex101-new")
prov = next(i for i, l in enumerate(OLD, 1) if l.startswith("F & S53, S54, S59"))
row(S, tex(prov, "S81 the trigger on delta (report 7, \\texttt{d5851b1}) & \\texttt{d5851b1} & 322; 102; 30; 40; 48; 42",
           new="S81 the trigger on delta (report 7, \\texttt{d5851b1}); S88a the write strength, S88b the horizon (report 8, \\texttt{a199d5e}; readings \\texttt{40087cb}) & \\texttt{a199d5e} & 322; 102; 30; 40; 48; 42; 78\\\\"),
    ("provenance table, F row: screens to S81, commit d5851b1, runs 322; 102; 30; 40; 48; 42",
     "provenance table, F row: ... S88a, S88b (report 8, a199d5e; readings 40087cb), commit a199d5e, runs ...; 78"),
    f"report 8 at {F} ({ctime(F)}): S88a {sum(len(a) for a in WA.values())} + S88b {sum(len(a) for a in HA.values())} = 78 runs; readings {F_READ} (F-order)",
    ok_if(sum(len(a) for a in WA.values()) + sum(len(a) for a in HA.values()) == 78),
    "846d0c8 adds the S88 entry, its commit and its 78 runs")

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# SESSION G, S87
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
R7 = GD + "report_7.md"
ST = load(G, GD + "s87_results.json")
ARMS = ("BASELINE", "REG3_KEYHINGE", "REG3_FREQHINGE", "REG3_SELFKL", "REG3_TEMP", "ORACLE_ALL",
        "REG3_FREQHINGE_D", "BASELINE_D", "REG3_D", "REG3_KEYHINGE_D")
GA = {a: runs(ST, a) for a in ARMS}
LOGG = show(G, GD + "s87.log")


def gbound(r):
    return transition(r["curve"]) is not None


def rstar(r):
    e = r["end"]
    return bool(e.get("margin", 0.0) >= 0.9 and e.get("eta_key_by_stream", 0.0) > 0.9)


def rbd(r):
    return gbound(r) and rstar(r)


def gcls(r):
    c = fail_class(r)
    if c == "STREAM-PARTIAL" and not r["end"]["layers"]["L3"]["one_to_one_val"]:
        return "MERGED"
    return c


def gp(x, y):           # b = x alone, c = y alone, p = P(x > y)
    ss = sorted(set(GA[x]) & set(GA[y]))
    b = sum(1 for s in ss if rbd(GA[x][s]) and not rbd(GA[y][s]))
    c = sum(1 for s in ss if rbd(GA[y][s]) and not rbd(GA[x][s]))
    return b, c, mcnemar1(b, c)


def at(r, step):
    return next((s for s in r["stats"] if s["step"] == step), None)


def commit(r, step=2400):
    s = at(r, step)
    return s.get("spread") if s else None


# ── order ────────────────────────────────────────────────────────────────────────────────────────────────────────
S = "G-order"
gfiles = ("explore_g_s87.py", "explore_g_s87_child.py")
ghist = {f: [l for l in git("log", "--format=%h %cI", G, "--", f).split("\n") if l] for f in gfiles}
gchanged = git("diff", "--name-only", G_SPEC, G, "--", ".", ":!explore_out").split()
gc_ = commits(G_SPEC, G, GD + "s87_results.json")
g_first = next(c for c in gc_ if c[2])
repro = json.loads(show(G, GD + "s87_repro.json"))
dlt = json.loads(show(G, GD + "s87_delta.json"))
gchk_path = [p for p in git("ls-tree", "-r", "--name-only", G, GD).split() if "checks_s87" in p][0]
gchk = json.loads(show(G, gchk_path))
d_commit = first_seen(GD + "s87_delta.json", G).split()[0]
st_at_d = json.loads(show(d_commit, GD + "s87_results.json"))["runs"]
first_D = next(c for c in gc_ if c[2] and any(k.split("|")[0].endswith("_D") for k in json.loads(show(c[0], GD + "s87_results.json"))["runs"]))
row(S, "explore_g_s87.py:1-45", "spec, readings, frequency rule and CHECKs committed at 3c89f87 before any run",
    f"3c89f87 {ctime(G_SPEC)} adds {list(gfiles)} (history: {ghist}); files outside explore_out changed 3c89f87..e46799a: "
    f"{gchanged or 'none'}; started {G_START} {ctime(G_START)} (s87.log banner 'git {re.search(r'git (\S+) \| python', LOGG).group(1)}'); "
    f"phase-0 file s87_repro.json at {first_seen(GD + 's87_repro.json', G)} (time {repro['time']}); CHECK file {gchk_path.split('/')[-1]} with the "
    f"first records at {g_first[0]} {g_first[1]} ({g_first[2]} records)",
    ok_if(not gchanged and all(len(v) == 1 and v[0].startswith(G_SPEC) for v in ghist.values())),
    "the readings precede every record; nothing executed changed after them")
row(S, "explore_g_s87.py:21-24, :174-183", "phase 2 chosen by the rule and written to s87_delta.json before phase 2 is queued",
    f"s87_delta.json {dlt} first at {d_commit} {ctime(d_commit)}; the store there {len(st_at_d)} records, _D arms "
    f"{sum(1 for k in st_at_d if k.split('|')[0].endswith('_D'))}; first _D record at {first_D[0]} {first_D[1]}; s87.log: arm line before "
    f"'40 runs queued': {LOGG.find('phase 2 arm (rule)') < LOGG.find('40 runs queued')}",
    ok_if(dlt["arm"] == "REG3_FREQHINGE" and not any(k.split("|")[0].endswith("_D") for k in st_at_d)))
rows_ok = all(hashlib.sha1(show(G_SPEC, n + ".py").encode()).hexdigest() == h for n, h, _ in ST["meta"]["sha_rows"])
row(S, "s87_results.json meta", "code SHA f205db31dfed (33 modules)",
    f"records {sorted({r['sha'][:12] for a in GA.values() for r in a.values()})}; meta sha {ST['meta']['sha'][:12]}; every sha_rows entry = "
    f"sha1 of the file at 3c89f87: {rows_ok}; sha1(json(sha_rows)) {hashlib.sha1(json.dumps(ST['meta']['sha_rows']).encode()).hexdigest()[:12]}; "
    f"{len(ST['meta']['sha_rows'])} modules",
    ok_if(rows_ok and hashlib.sha1(json.dumps(ST["meta"]["sha_rows"]).encode()).hexdigest() == ST["meta"]["sha"]),
    "the code that ran is the code committed with the spec; the store meta's '3919ac1+dirty' is phase 2's start, which rewrote the meta")
s82_now = git("rev-parse", f"{G}:{GD}s82_devices_results.json").strip()
s82_then = git("rev-parse", f"8973d48:{GD}s82_devices_results.json").strip()
s82 = load(G, GD + "s82_devices_results.json")
s82_2100 = sorted(k.split("|")[0] for k in s82["runs"] if k.split("|")[1] == "2100")
row(S, rep(G, R7, 29, "Phase 0: S82"),
    "phase 0: S82's BASELINE and REG3_KEYHINGE (and before phase 2 BASELINE_D, REG3_D, REG3_KEYHINGE_D) at seed 2100 reproduce through 2400 bit for bit",
    f"s87_repro.json {repro['result']} at {repro['time']}, before the first record ({g_first[1]}); delta references: s87.log "
    f"'{re.search(r'phase 2 reproduction[^\n]*', LOGG).group(0)}' before '40 runs queued': "
    f"{LOGG.find('phase 2 reproduction') < LOGG.find('40 runs queued')}; S82 store at e46799a = at 8973d48: {s82_now == s82_then}; "
    f"S82 records at 2100: {s82_2100}",
    "PARTIAL", "every reference reproduced, and before the runs it covers; one seed (2100) per arm, as explore_g_s87.py:17-18 fixes; the "
    "comparison (explore_g_s87_child.py:220-230: curve and integer-step stats <= 2400) is read from its file and log, not rerun; the delta "
    "references' result is in the log only")

# ── counts ───────────────────────────────────────────────────────────────────────────────────────────────────────
S = "G-report7"
allG = [r for a in GA.values() for r in a.values()]
row(S, rep(G, R7, 6, "142/142 runs ok"), "142/142 runs ok, none retried; 212 min on 4 workers",
    f"{sum(r.get('ok', False) for r in allG)}/{len(allG)}; attempt {sorted({r.get('attempt') for r in allG})}; s87.log "
    f"'{re.search(r'S87 runs done in [\d.]+ min', LOGG).group(0)}'; recorded transition = rlib.transition(curve) in "
    f"{sum(transition(r['curve']) == r['transition'] for r in allG)}/{len(allG)}",
    ok_if(len(allG) == 142 and all(r.get("ok") for r in allG)))
hits_g = seed_hits(2700, 2729)
other_g = {p: v for p, v in hits_g.items() if p != GD + "s87_results.json"}
row(S, rep(G, R7, 4, "Seeds 2700-2719"), "seeds 2700-2719 (Hebbian), 2720-2729 (delta); no record on any branch used them",
    f"Hebbian arms on {sorted({s for a in ARMS[:5] for s in GA[a]})[0]}-{sorted({s for a in ARMS[:5] for s in GA[a]})[-1]}, delta "
    f"{sorted({s for a in ARMS[6:] for s in GA[a]})[0]}-{sorted({s for a in ARMS[6:] for s in GA[a]})[-1]}; 2700-2729 over all JSON on "
    f"{len(BRANCHES)} branches: only in {sorted(hits_g)}" + (f"; others {other_g}" if other_g else ""), ok_if(not other_g))

R7T = {"BASELINE": (12, "0/20", "0/20", {"KEY": 17, "STREAM-PARTIAL": 3}, "-", 0.873),
       "REG3_KEYHINGE": (13, "20/20", "20/20", {}, "3600 (3600-9600)", 0.934),
       "REG3_FREQHINGE": (14, "17/20", "17/20", {"KEY": 3}, "3600 (3600-6000)", 0.591),
       "REG3_SELFKL": (15, "0/20", "0/20", {"KEY": 18, "OTHER": 2}, "-", 0.0001),
       "REG3_TEMP": (16, "3/20", "3/20", {"KEY": 16, "OTHER": 1}, "3600 (2400-7200)", 0.923),
       "ORACLE_ALL": (17, "2/2", "2/2", {}, "1200 (1200-1200)", None),
       "REG3_FREQHINGE_D": (19, "10/10", "10/10", {}, "2400 (2400-3600)", None),
       "BASELINE_D": (20, "0/10", "3/10", {"KEY": 7}, "-", None),
       "REG3_D": (21, "0/10", "2/10", {"KEY": 8}, "-", None),
       "REG3_KEYHINGE_D": (22, "10/10", "10/10", {}, "2400 (2400-3600)", None)}
CM = {}
for a, (ln, rbs, bs, cls_, t_, cm_) in R7T.items():
    rr = GA[a]
    n = len(rr)
    k_rb, k_b = sum(rbd(r) for r in rr.values()), sum(gbound(r) for r in rr.values())
    cl = dict(collections.Counter(gcls(r) for r in rr.values() if not gbound(r)))
    t = sorted(transition(r["curve"]) for r in rr.values() if rbd(r))
    ts = f"{med(t):.0f} ({t[0]}-{t[-1]})" if t else "-"
    cm = med([commit(r) for r in rr.values()]) if a in ("BASELINE", "REG3_KEYHINGE", "REG3_FREQHINGE", "REG3_SELFKL", "REG3_TEMP") else None
    CM[a] = cm
    good = (f"{k_rb}/{n}" == rbs and f"{k_b}/{n}" == bs and cl == cls_ and (ts == t_ or (t_ == "-" and not t))
            and (cm_ is None or round(cm, 4 if cm_ < 0.01 else 3) == cm_))
    row(S, rep(G, R7, ln, a.split("_")[-1] if a not in ("REG3_D",) else "REG3_D"),
        f"{a}: ROUTED-BOUND {rbs}; BOUND {bs}; unbound {cls_ or '-'}; transitions {t_}" + (f"; commitment at 2400 {cm_}" if cm_ is not None else ""),
        f"{k_rb}/{n} {wil(k_rb, n)}; BOUND {k_b}/{n}; unbound {cl or '-'}; transitions {ts}" + (f"; commitment median {cm:.4f}" if cm is not None else ""),
        ok_if(good))
bdb = [(s, transition(r["curve"]), gcls(r), round(r["end"]["eta_key_by_key"], 2)) for s, r in sorted(GA["BASELINE_D"].items()) if gbound(r)]
row(S, rep(G, R7, 20, "3/10 (key splits)"), "BASELINE_D: BOUND 3/10 '(key splits)'",
    f"bound unrouted runs (seed, transition, v1 class of the end gate, end eta^2 by key, end margin): "
    f"{[(s, t, c, e, round(GA['BASELINE_D'][s]['end']['margin'], 2)) for s, t, c, e in bdb]}",
    ok_if(len(bdb) == 3 and all(c == "KEY" for _, _, c, _ in bdb)),
    "one is a key split (2721); 2726 and 2728 are half-routed (one-to-one VAL map at every layer, margin 0.47, eta^2 by key 0.00); not in the draft")

# readings
disc = {a: gp(a, "BASELINE") for a in ("REG3_FREQHINGE", "REG3_SELFKL", "REG3_TEMP", "REG3_KEYHINGE")}
kh = {a: gp("REG3_KEYHINGE", a) for a in ("REG3_FREQHINGE", "REG3_SELFKL", "REG3_TEMP")}
clamp = {a: (gp(a, "BASELINE")[0] - gp(a, "BASELINE")[1] <= 1 and CM[a] < CM["BASELINE"]) for a in kh}
D = {a: disc[a][2] < 0.05 for a in disc}
within = {a: kh[a][0] - kh[a][1] <= 1 for a in kh}
P4 = mcnemar1(4, 0)
row(S, rep(G, R7, 31, "beats BASELINE 20 / 0 again"), "the control REG3_KEYHINGE beats BASELINE 20 / 0 again", f"{disc['REG3_KEYHINGE']}",
    ok_if(disc["REG3_KEYHINGE"][:2] == (20, 0)))
row(S, rep(G, R7, 35, "FREQHINGE only"), "discovered: FREQHINGE vs BASELINE 17 / 0 (p 8e-6); SELFKL 0 / 0; TEMP 3 / 0 (p 0.125): FREQHINGE only",
    "; ".join(f"{a} {v[0]} / {v[1]} p {v[2]:.3g} -> {D[a]}" for a, v in disc.items() if a != "REG3_KEYHINGE"),
    ok_if(D["REG3_FREQHINGE"] and not D["REG3_SELFKL"] and not D["REG3_TEMP"] and disc["REG3_FREQHINGE"][:2] == (17, 0)))
row(S, rep(G, R7, 36, "two pairs short"), "'a frequency hinge replaces the key hinge' does not apply: KH only 3, FREQ only 0 (two pairs short)",
    f"KH only {kh['REG3_FREQHINGE'][0]}, FREQ only {kh['REG3_FREQHINGE'][1]}: (KH only) - (FREQ only) = {kh['REG3_FREQHINGE'][0] - kh['REG3_FREQHINGE'][1]} "
    f"> 1 -> applies {D['REG3_FREQHINGE'] and within['REG3_FREQHINGE']}", ok_if(not within["REG3_FREQHINGE"] and kh["REG3_FREQHINGE"][:2] == (3, 0)))
row(S, rep(G, R7, 37, "a hinge that reads nothing suffices"), "'a hinge that reads nothing suffices' does not apply",
    f"SELFKL discovered {D['REG3_SELFKL']}, TEMP {D['REG3_TEMP']}", ok_if(not any(D[a] and within[a] for a in ("REG3_SELFKL", "REG3_TEMP"))))
row(S, rep(G, R7, 38, "**SELFKL**"), "'clamps': SELFKL (0 / 0; 0.0001 vs 0.873); TEMP no (0.923 > 0.873); FREQHINGE no (17 / 0)",
    "; ".join(f"{a} (arm only - BASELINE only) {gp(a, 'BASELINE')[0] - gp(a, 'BASELINE')[1]}, commitment {CM[a]:.4f} vs {CM['BASELINE']:.4f} -> {clamp[a]}" for a in clamp),
    ok_if(clamp == {"REG3_FREQHINGE": False, "REG3_SELFKL": True, "REG3_TEMP": False}), "explore_g_s87.py:31-32")
row(S, rep(G, R7, 39, "does not apply (vs FREQHINGE 3 / 0, p 0.125"), "'the identity term is still needed' does not apply (vs FREQ 3/0 p 0.125; SELFKL 20/0; TEMP 17/0)",
    "; ".join(f"KH vs {a} {v[0]} / {v[1]} p {v[2]:.3g} (<= p(4 vs 0) {P4}: {v[2] <= P4})" for a, v in kh.items()),
    ok_if(not all(v[2] <= P4 for v in kh.values()) and [v[:2] for v in kh.values()] == [(3, 0), (20, 0), (17, 0)]))
order = ("REG3_TEMP", "REG3_SELFKL", "REG3_FREQHINGE")
arm2 = next((a for a in order if D[a]), None)
dd = {ref: gp("REG3_FREQHINGE_D", ref)[:2] for ref in ("BASELINE_D", "REG3_D", "REG3_KEYHINGE_D")}
row(S, rep(G, R7, 40, "FREQHINGE_D 10/10"), "delta arm by the rule REG3_FREQHINGE; FREQHINGE_D 10/10 against BASELINE_D 0/10 (10/0), REG3_D 0/10 (10/0); equal to KH_D (0/0)",
    f"rule (order {order}): {arm2}; pairs {dd}", ok_if(arm2 == "REG3_FREQHINGE" and dd == {"BASELINE_D": (10, 0), "REG3_D": (10, 0), "REG3_KEYHINGE_D": (0, 0)}))

# frequency table
FR = [r["freq"] for a in ("REG3_FREQHINGE", "REG3_FREQHINGE_D") for r in GA[a].values()]
same_tab = all(f == FR[0] for f in FR)
tab = FR[0]["table"]
L_in = 22   # RandHeaderTask-lite: L = S(NB + 2P) + 3 n_q = 23; the input drops the query's VAL (explore_g_s87_child.py:93)
exp = {"CTX": 5 / L_in / 2, "KEY": 9 / L_in / 4, "VAL": 8 / L_in / 16}
vals_ = [v for _, v in tab]
cut = max(range(len(vals_) - 1), key=lambda i: vals_[i] / vals_[i + 1])
row(S, rep(G, R7, 49, "0.114 each"), "the two context types most frequent (0.114 each), the four keys next (0.102 each), the values rare (0.023 each); the largest-ratio cut selects all six",
    f"records (all {len(FR)} FREQ runs identical: {same_tab}): types {FR[0]['types']}, table head {tab[:7]}; analytic from the task "
    f"(per 22 input tokens: 5 CTX over 2 types, 9 KEY over 4, 8 VAL over 16): {', '.join(f'{k} {v:.4f}' for k, v in exp.items())}; "
    f"largest ratio within the recorded head at index {cut} ({vals_[cut] / vals_[cut + 1]:.2f}x), above it {len(tab[:cut + 1])} types",
    ok_if(same_tab and sorted(FR[0]["types"]) == [0, 1, 2, 3, 4, 5] and cut == 5),
    "token ids: CTX 0-1, KEY 2-5, VAL 6-21 (explore_f_tasks.RandHeaderTask.make_batch); 'the keys at 0.102' spans 0.1017-0.1027")
row(S, rep(G, R7, 50, "14 per"), "that covers all 8 of the key term's positions, plus the query's KEY and the 5 CTX positions: 14 per sequence",
    "analytic, every sequence alike: frequent positions = 5 CTX (4 block headers + query) + 9 KEY (8 body + query) = 14; key-term positions = "
    "the 8 body KEYs, all frequent; 8 of 14 frequent positions are the key term's; CHECK row (s87.log) 'frequent positions per sequence [14], "
    f"key-term positions [8], both [8]': {'frequent positions per sequence [14], key-term positions [8], both [8]' in LOGG}",
    "MATCH", "the caveat: on this task the cut is exactly 'not a value token', i.e. the key term's positions and identities plus the stream "
    "tokens (grouped by token id = by stream) and the query's key")
fc_src = show(G, "explore_g_s87_child.py").split("\n")
calls = [i for i, l in enumerate(fc_src, 1) if "freq_types(" in l and i > 44]
row(S, rep(G, R7, 73, "computed from token ids only"), "the frequency table is computed from token ids only (one parameter, a LongTensor asserted, a float refused); frequent types {0,...,5}",
    f"call path: freq_types(tokens) (explore_g_s87_child.py:80-89) is defined / called at lines {calls}: frequent() -> freq_types(sample_tokens()), "
    "sample_tokens = task.make_batch(4096, Generator(87000))[0][:, :-1] (:92-93; make_batch returns (tokens, query streams), the "
    "streams dropped); the term (term_value, :107-115) reads the gates, the forward's input token ids and that table; the CHECK row passed",
    "MATCH", "no label reaches the statistic. The CHECK's float-refusal does not by itself exclude labels (stream labels are LongTensors "
    "too); the call path does")
fq = [r["term"]["fired"] for r in GA["REG3_FREQHINGE"].values()]
v300 = [max(r["term"]["value_at"]["300"]) for r in GA["REG3_FREQHINGE"].values()]
khf = [r["hinge"]["fired"] for r in GA["REG3_KEYHINGE"].values()]
kh300 = [max(r["hinge"]["eta_at"]["300"]) for r in GA["REG3_KEYHINGE"].values()]
row(S, rep(G, R7, 53, "drives its η² to 0 by update 300"), "FREQHINGE fires 53-115 times in 1-2400 (median 80) and drives its eta^2 to 0 by update 300, like the key term",
    f"firings {min(fq)}-{max(fq)} (median {med(fq)}); the term relu(eta^2 - 0.2) at 300: max over runs and layers {max(v300)}; the key term "
    f"(control) fired {min(khf)}-{max(khf)}, its eta^2 at 300 up to {max(kh300):.3f}", "PARTIAL",
    "the counts hold; the record has the term, not eta^2: the term is 0 at 300 in " + f"{sum(x == 0 for x in v300)} of 20 runs (median 0, max {max(v300):.3f}, "
    "i.e. eta^2 up to " + f"{0.2 + max(v300):.2f}" + " in one layer), and 0 means eta^2 <= 0.2, not 0; the key term's own recorded eta^2 at 300 is up to " + f"{max(kh300):.2f}")
b24 = {a: {ro: med([at(r, 2400)["beta"][ro] for r in GA[a].values()]) for ro in ("CTX", "KEY", "VAL")} for a in ("REG3_FREQHINGE", "REG3_KEYHINGE")}
row(S, rep(G, R7, 56, "0.24 / 0.19 / 0.19"), "beta^R CTX/KEY/VAL at 2400: FREQ 0.24/0.19/0.19 vs KH 0.32/0.20/0.21; commitment 0.59 vs 0.93",
    f"medians {', '.join(f'{a} ' + '/'.join(f'{v:.2f}' for v in b.values()) for a, b in b24.items())}; commitment {CM['REG3_FREQHINGE']:.3f} vs {CM['REG3_KEYHINGE']:.3f}",
    ok_if(["%.2f" % v for v in b24["REG3_FREQHINGE"].values()] == ["0.24", "0.19", "0.19"] and ["%.2f" % v for v in b24["REG3_KEYHINGE"].values()] == ["0.32", "0.20", "0.21"]))
ff = [(s, gcls(r)) for s, r in sorted(GA["REG3_FREQHINGE"].items()) if not rbd(r)]
tD = sorted(transition(r["curve"]) for r in GA["REG3_FREQHINGE_D"].values())
row(S, rep(G, R7, 58, "(2700, 2709, 2713) key-split"), "17 of 20 route and bind at 3600-6000; the three that do not (2700, 2709, 2713) key-split; delta 10/10 at 2400-3600",
    f"failures {ff}; delta transitions {tD[0]}-{tD[-1]}", ok_if([s for s, _ in ff] == [2700, 2709, 2713] and all(c == "KEY" for _, c in ff)))
sk = GA["REG3_SELFKL"]
sk_v = max(max(r["term"]["value_at"][u]) for r in sk.values() for u in ("300", "600", "1200", "2400"))
sk_c = {st_: med([commit(r, st_) for r in sk.values()]) for st_ in (1200, 2400)}
sk_b = [at(r, st_)["beta"][ro] for r in sk.values() for st_ in (1200, 2400) for ro in ("CTX", "KEY", "VAL")]
row(S, rep(G, R7, 60, "0.0003 at 1200"), "SELFKL: value 0.0000 from 300; commitment 0.0003 at 1200 and 0.0001 at 2400; beta^R stays at its initial 0.10 at every role through 2400; after the window KEY 18",
    f"term max at 300-2400 {sk_v}; commitment medians {sk_c}; beta^R at 1200/2400 {rng(sk_b, 3)}; unbound classes "
    f"{dict(collections.Counter(gcls(r) for r in sk.values()))}",
    ok_if(sk_v < 5e-5 and round(sk_c[2400], 4) == 0.0001 and 0.09 <= min(sk_b) and max(sk_b) <= 0.11))
tp = GA["REG3_TEMP"]
row(S, rep(G, R7, 64, "0.84, as BASELINE's 0.85"), "TEMP commits by 1200 (0.84, as BASELINE's 0.85); beta^R 0.30/0.33/0.30 at 1200; 16/20 key-split; routes 3/20 (2708, 2716, 2719)",
    f"commitment at 1200 TEMP {med([commit(r, 1200) for r in tp.values()]):.3f}, BASELINE {med([commit(r, 1200) for r in GA['BASELINE'].values()]):.3f}; beta^R "
    + "/".join(f"{med([at(r, 1200)['beta'][ro] for r in tp.values()]):.2f}" for ro in ("CTX", "KEY", "VAL"))
    + f"; KEY failures {sum(gcls(r) == 'KEY' for r in tp.values() if not gbound(r))}; routed {sorted(s for s, r in tp.items() if rbd(r))}", "MATCH",
    "'its logits simply grow fourfold' is not in the record")
pl = {}
for a in ("BASELINE", "REG3_FREQHINGE", "REG3_SELFKL", "REG3_TEMP"):
    cells = collections.defaultdict(list)
    for r in GA[a].values():
        if not gbound(r):
            for c, v in r["acc_by_cell"].items():
                cells[c].append(v["acc"])
    pl[a] = [med(v) for v in cells.values()]
pall = [x for v in pl.values() for x in v]
row(S, rep(G, R7, 67, "0.45-0.60"), "every unbound arm sits at 0.45-0.60 across the 12 (stream, block, pair) cells",
    "per arm, the median over its unbound runs per cell: " + "; ".join(f"{a} {rng(v, 3)}" for a, v in pl.items()),
    ok_if(round(min(pall), 2) >= 0.45 and round(max(pall), 2) <= 0.60),
    "the explore_g_rhl_parent PLATEAU statistic (median over runs); single runs' cells range wider")

# ── Revision 10 on S87 ─────────────────────────────────────────────────────────────────────────────────────────────
S = "G-tex10"
row(S, tex(66, "a label-free frequency hinge $17/20$, two pairs short of replacing the key hinge"),
    "index row G: 'a label-free frequency hinge 17/20, two pairs short of replacing the key hinge'",
    f"{sum(rbd(r) for r in GA['REG3_FREQHINGE'].values())}/20; KH only {kh['REG3_FREQHINGE'][0]}, FREQ only {kh['REG3_FREQHINGE'][1]} (rule <= 1)", "MATCH")
row(S, tex(283, "S87, seeds 2700--2719 Hebbian and 2720--2729 delta, with the S82 baseline and key hinge rerun as controls, $0/20$ and $20/20$ again"),
    "S87 on 2700-2719 / 2720-2729; controls 0/20 and 20/20 again",
    f"BASELINE {sum(rbd(r) for r in GA['BASELINE'].values())}/20, KH {sum(rbd(r) for r in GA['REG3_KEYHINGE'].values())}/20", "MATCH")
row(S, tex(283, "computed once before training from a sample of 4096 training sequences with no labels of any kind"),
    "the table 'computed once before training from a sample of 4096 training sequences with no labels of any kind'",
    "frequent() is computed lazily at the term's first use (training update 1) in each run's process, from make_batch(4096, Generator(87000)) "
    f"token ids only; identical in every FREQ record ({same_tab})", "MATCH",
    "in effect 'once, before training': it depends only on the task's generator and seed, never on the model; the label check is the call "
    "path above. The other session's first pass (7.2) marked the earlier 'built in the first updates' DISAGREE; that wording was closer to the "
    "code's timing and the substance was the same")
row(S, tex(283, "the two context tokens at $0.114$ each, the four keys at $0.102$; the values are rare at $0.023$"),
    "the largest ratio cut selects six types (CTX 0.114 each, keys 0.102, values 0.023), covering the key term's 8 positions and the query's key and the 5 context positions",
    f"table {tab[:7]}; analytic {', '.join(f'{k} {v:.4f}' for k, v in exp.items())}; 14 positions = 8 + 1 + 5", "MATCH")
row(S, tex(283, "discovered $\\mathbf{17/20}$ (17 vs 0 against the baseline, $p = 8\\times10^{-6}$; on delta memory $10/10$ against $0/10$, descriptive)"),
    "discovered 17/20 (17 vs 0, p = 8e-6); delta 10/10 against 0/10, descriptive",
    f"{disc['REG3_FREQHINGE']}; FREQ_D {sum(rbd(r) for r in GA['REG3_FREQHINGE_D'].values())}/10 vs BASELINE_D "
    f"{sum(rbd(r) for r in GA['BASELINE_D'].values())}/10 ({dd['BASELINE_D']})", ok_if(disc["REG3_FREQHINGE"][:2] == (17, 0)),
    "'descriptive': the docstring's phase 2 has no reading (explore_g_s87.py:21-24)")
row(S, tex(283, "two pairs short of the pre-fixed rule for ``replaces the key hinge'' (the key hinge alone on 3 seeds, the frequency hinge alone on none) and with the key term not shown needed over it (3 vs 0, $p = 0.125$)"),
    "two pairs short of 'replaces' (3 vs 0); the key term not shown needed over it (3 vs 0, p = 0.125)",
    f"KH vs FREQ {kh['REG3_FREQHINGE']}; the rule for 'still needed' is p <= p(4 vs 0) = {P4} against every new arm", "MATCH")
sk_src = show(G, "explore_g_s87_child.py")
row(S, tex(283, "a penalty on the gate's own distance from uniform at every position, which reads nothing",
           new="a penalty on each position's gate's distance from the sequence's own mean gate, which reads nothing"),
    ("SELFKL: 'a penalty on the gate's own distance from uniform at every position, which reads nothing'",
     "SELFKL: 'a penalty on each position's gate's distance from the sequence's own mean gate, which reads nothing'"),
    "explore_g_s87_child.py:33-35 and :116-122: KL(g_t || gbar), gbar the sequence's own mean gate over its body positions (t < body_len = 20 "
    f"of the 22 input positions); no uniform distribution enters (term_value 'selfkl': gb = g[:, :RHL.body_len], m = gb.mean(1) "
    f"{'gb = g[:, :RHL.body_len]' in sk_src and 'm = gb.mean(1, keepdim=True)' in sk_src})",
    "MATCH" if "gb = g[:, :RHL.body_len]" in sk_src and "m = gb.mean(1, keepdim=True)" in sk_src else "DISAGREE",
    "846d0c8 takes the referee's wording. 'Each position' means each of the 20 body positions (the query block's two are outside the term), "
    "as 'the sequence's own mean gate' is their mean")
row(S, tex(283, "bound $0/20$ and clamped the gate (its commitment $0.0001$ against the baseline's $0.873$ at 2400, the pre-fixed ``clamps'')"),
    "SELFKL bound 0/20 and clamped the gate (0.0001 vs 0.873, the pre-fixed 'clamps')",
    f"ROUTED-BOUND {sum(rbd(r) for r in sk.values())}/20, BOUND {sum(gbound(r) for r in sk.values())}/20; commitment {CM['REG3_SELFKL']:.4f} vs "
    f"{CM['BASELINE']:.4f}; clamps {clamp['REG3_SELFKL']}", ok_if(clamp["REG3_SELFKL"]))
row(S, tex(283, "a raised gate temperature for the slow phase $3/20$"), "TEMP for the slow phase 3/20",
    f"{sum(rbd(r) for r in tp.values())}/20; T = 4 for updates 1-2400 (the SLOW recipe's slow phase; explore_g_s87_child.py:36-38)", "MATCH")
row(S, tex(283, "the three failures of the frequency hinge are key splits"), "the three failures of the frequency hinge are key splits", f"{ff}", ok_if(all(c == "KEY" for _, c in ff)))
row(S, tex(283, "the context positions, groups them by stream and penalises the gate's dependence on the stream token there, which on session~G's account is why it lags"),
    "what it covers beyond the key term, the context positions, groups them by stream ... on G's account why it lags",
    "grouping by token id at CTX positions is grouping by stream (CTX token = stream id); also the query's KEY (named earlier in the sentence)",
    "MATCH", "marked as G's account")
row(S, tex(283, "does most of the key hinge's work on this task, on one screen"), "a prior that reads the task's token statistics rather than its labels does most of the key hinge's work on this task, on one screen",
    "17/20 vs 20/20 (3 vs 0); the statistic receives token ids only", "MATCH",
    "scoped; the caveat that on this task the frequency cut equals 'not a value', i.e. the key term's positions plus the stream tokens, is in the "
    "same paragraph ('happens to select ... covers the key term's eight positions')")
row(S, tex(322, "$17/20$ / $0/20$ / $3/20$ & $10/10$ / --- / ---"), "Table 6 row: 17/20 / 0/20 / 3/20 (Hebbian), 10/10 / --- / --- (delta); seeds 2700-2729; baseline 0/20, key hinge 20/20",
    f"{sum(rbd(r) for r in GA['REG3_FREQHINGE'].values())}/20 / {sum(rbd(r) for r in sk.values())}/20 / {sum(rbd(r) for r in tp.values())}/20; "
    f"FREQ_D {sum(rbd(r) for r in GA['REG3_FREQHINGE_D'].values())}/10; no SELFKL_D / TEMP_D records "
    f"({not any(a in GA for a in ('REG3_SELFKL_D', 'REG3_TEMP_D'))})", "MATCH", "caption tex10:328: 'S82, S83 and S87', ROUTED-BOUND")
row(S, tex(485, "The hinge reads key identities", new="The hinge reads key identities, and the one label-free hinge found so far, on a frequency table of token ids, discovers $17/20$ on a task where that table selects the keys and the contexts (S87)"),
    ("limits, item 1: 'The hinge reads key identities; a dense loss may change the race without it'",
     "limits, item 1: 'The hinge reads key identities, and the one label-free hinge found so far, on a frequency table of token ids, discovers 17/20 on a task where that table selects the keys and the contexts (S87)'"),
    f"S87: FREQHINGE {sum(rbd(r) for r in GA['REG3_FREQHINGE'].values())}/20 ROUTED-BOUND; the table's frequent types {sorted(FR[0]['types'])} = CTX 0-1 and KEY 2-5; "
    f"SELFKL {sum(rbd(r) for r in sk.values())}/20 and TEMP {sum(rbd(r) for r in tp.values())}/20 read no labels either and did not discover",
    ok_if(sorted(FR[0]["types"]) == [0, 1, 2, 3, 4, 5] and sum(rbd(r) for r in GA["REG3_FREQHINGE"].values()) == 17),
    "846d0c8 adds S87's answer with its caveat ('on a task where that table selects the keys and the contexts'), as the pass asked; "
    "'the one label-free hinge found so far' means the one that discovers")
provg = next(i for i, l in enumerate(OLD, 1) if l.startswith("G & S55 probe"))
row(S, tex(provg, "S83 the hinge's inputs, the control and the window (report 6, \\texttt{076005d}; spec \\texttt{b2f9907}) & \\texttt{076005d} & 22; 139; 50; 150; 136; 82",
           new="S87 hinges that read no task labels (report 7, \\texttt{e46799a}; spec \\texttt{3c89f87}) & \\texttt{e46799a} & 22; 139; 50; 150; 136; 82; 142\\\\"),
    ("provenance table, G row: screens to S83, commit 076005d, runs to 82",
     "provenance table, G row: ... S87 (report 7, e46799a; spec 3c89f87), commit e46799a, runs ...; 142"),
    f"report 7 at {G} ({ctime(G)}), spec {G_SPEC}: S87 {len(allG)} runs", ok_if(len(allG) == 142),
    "846d0c8 adds the S87 entry, its commit, spec and 142 runs. 'Hinges' takes in TEMP, a temperature schedule rather than a hinge; a label")
row(S, tex(495, "is found on every seed of a screen when a hinge keeps the gate off key identity"), "conclusion on distant cues (S87 not mentioned)",
    "no statement on S87", "MATCH", "nothing to check; the omission is the limits row's")

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# REVISION 10.1 (846d0c8): text on S88 and S87 it added
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
S = "F-tex101-new"
row(S, new(370, "the timing of the learned write strength's collapse, stated as 1200 updates where the records give 3600 in one arm and 1200--6000 in the other"),
    "10.1's account of the corrected S88a statement: stated as 1200 updates; the records give 3600 in one arm and 1200-6000 in the other",
    f"DELTA_LEARNED all-layer median < 0.1 first at {sorted(set(fb['DELTA_LEARNED'].values()))}; SLOW +{sorted(x - 2400 for x in fbs)} after the switch",
    ok_if(set(fb["DELTA_LEARNED"].values()) == {3600} and (fbs[0] - 2400, fbs[-1] - 2400) == (1200, 6000)), is_new=True)
row(S, new(370, "the limits list updated for S87 and S88b"), "10.1: the limits list updated for S87 and S88b",
    "tex101:489 (S87) and tex101:491 (S88b), rows above", "MATCH", is_new=True)
row(S, new(370, "and provenance rows for E, F, G and H that stopped before the fifth round"), "10.1's account of the provenance rows (F's and G's here)",
    f"tex101:542 now ends at S88 (a199d5e, 78 runs) and tex101:543 at S87 (e46799a, 142 runs): rows above", "MATCH", is_new=True)
row(S, new(499, "and after the merge it checked the rest, the fifth round's screens included, with the same outcome (the numbers right, nine statements false"),
    "conclusion: after the merge the referee checked the fifth round's screens (S88, S87 here): numbers right, statements false",
    f"check_rev10_FG at {V101}: {'check_rev10_FG.md' in git('ls-tree', '--name-only', V101, 'docs/review_2026-10/rev10/')}; its counts and readings reproduce "
    "(sections F-report8, G-report7); its S88/S87 false statements were referee_pass §1.6 (beta's timing), §1.7 (SELFKL), §1.8 (F, G rows)",
    "MATCH", "replaces the removed tex10:495 sentence", is_new=True)
S = "G-tex101-new"
row(S, new(370, "the self-distance penalty described as a distance from uniform (it is from the sequence's own mean gate)"),
    "10.1's account of the corrected S87 statement", "explore_g_s87_child.py term_value 'selfkl' (row above)",
    ok_if("m = gb.mean(1, keepdim=True)" in sk_src), is_new=True)

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# pinned branches: commits after each pin up to 846d0c8's commit time
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
print("## Pinned branches: commits after each pin, up to 846d0c8's commit time (2026-10-19 18:33:26 UTC)\n")
for br, pin in BRANCH_PINS.items():
    lst = [l.split(" ", 2) for l in git("log", "--first-parent", "--format=%h %ct %s", f"{pin}..{br}").split("\n") if l.strip()]
    mv = [(h, datetime.datetime.fromtimestamp(int(t), datetime.timezone.utc).strftime("%H:%M:%S"), m[:60]) for h, t, m in lst if int(t) <= T101]
    print(f"- {br.split('/', 1)[1]} pinned {pin}: {len(mv)} commit(s) by 18:33:26 UTC {mv[::-1]}; {sum(1 for h, t, m in lst if int(t) > T101)} after")
print(f"\nF at 846d0c8's commit time: {F_AT101}, reports {len(r8_101)}; RECOMPUTED TEXT DIFFERS FROM THE ORIGINAL'S: {RECDIFF}")
missing = [k for k in OLDROWS if k not in SEEN]
assert not missing, missing

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
K = ("DISAGREE", "PARTIAL", "NOT RECORD-CHECKABLE", "MATCH", "REMOVED")
total, total_old, total_same = collections.Counter(), collections.Counter(), collections.Counter()
for sec, rows_ in SECTIONS.items():
    c = collections.Counter(r[3] for r in rows_)
    total.update(c)
    total_old.update(r[5] for r in rows_ if r[5] != "\u2014")
    total_same.update(r[3] for r in rows_ if r[5] != "\u2014")
    print(f"\n### {sec}: {len(rows_)} rows — " + ", ".join(f"{k} {c.get(k, 0)}" for k in K if k != "REMOVED" or c.get(k)) + "\n")
    print("| where | statement | recomputed | status | v585b666 \u2192 v846d0c8 | note |\n|---|---|---|---|---|---|")
    order_ = {"DISAGREE": 0, "PARTIAL": 1, "NOT RECORD-CHECKABLE": 2, "MATCH": 3, "REMOVED": 4}
    for w, s_, r_, st_, n_, ost, dl in sorted(rows_, key=lambda x: order_[x[3]]):
        print(f"| {cell(w)} | {cell(s_)} | {cell(r_)} | **{st_}** | {cell(dl)} | {cell(n_)} |")
assert [total_old[k] for k in K[:4]] == [6, 15, 3, 83], total_old
n_old = sum(total_old.values())
print(f"\nTOTAL (original, 585b666) {n_old} rows: " + ", ".join(f"{k} {total_old.get(k, 0)}" for k in K[:4]))
print(f"TOTAL (the same {n_old} rows against 846d0c8): " + ", ".join(f"{k} {total_same.get(k, 0)}" for k in K))
print(f"TOTAL (846d0c8, all {sum(total.values())} rows): " + ", ".join(f"{k} {total.get(k, 0)}" for k in K))
print("STATUS CHANGES: " + "; ".join(f"{r[0]}: {r[6]}" for rows_ in SECTIONS.values() for r in rows_ if r[5] != "\u2014" and r[5] != r[3]))
