#!/usr/bin/env python3
"""audit_passes.py -- Referee V's audit of the two passes another session made over Revision 10's draft
(docs/review_2026-10/rev10/first_pass.md on ea3afef, second_pass.md on 6e75f69, and the recompute_* scripts).

    python3 -I docs/review_2026-10/rev10/audit_passes.py        (from the repository root, after git fetch)

Stdlib only. Reads git objects only (git show / ls-tree / log); writes nothing; runs no session or test code; trains
nothing. Every result it prints is quoted in audit_passes.md.

Sections:
  A  the passes' own row and verdict counts against their stated summaries
  B  coverage: every number in the ADDED text of v9 (c918ff7) -> final v10 (d2b7abd), mapped to the pass rows that cite
     the same tex line and contain the same number
  C  the drafts' changed lines (ea3afef -> 6e75f69 -> d2b7abd) and which pass rows cite them
  D  follow-through: the passes' items against the final text (phrase present / absent)
  E  recomputation with the referee's definitions (rlib: transition from the curve, the stream -> channel map from
     end.stream_gate) against the recorded fields the recompute_* scripts use; the screens no committed script covers
  F  pre-registration order of the fifth-round screens (readings commit vs first record commit; scripts unchanged after)
  G  seed freshness of the screens' seed ranges over every remote branch tip
  H  the recompute_* files themselves (runnable paths, spliced output)
"""
import collections
import difflib
import json
import math
import os
import re
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rlib  # noqa: E402  (the referee's definitions; read-only)

V9, D1, D2, D3 = "c918ff7", "ea3afef", "6e75f69", "d2b7abd"
MAIN = "585b666"
TEX9, TEX10 = "multichannel_hebbian_report_v9.tex", "multichannel_hebbian_report_v10.tex"
PASS = "docs/review_2026-10/rev10/"


def git(*a):
    return subprocess.check_output(("git", "-C", ROOT) + a).decode()


def show(ref, path):
    return git("show", f"{ref}:{path}")


def jshow(ref, path):
    return json.loads(show(ref, path))


def hdr(t):
    print("\n" + "=" * 8, t, "=" * 8)


# ───────────────────────────── A. the passes' counts ─────────────────────────────
VERD = ("MATCH", "PARTIAL", "DISAGREE", "NOT RECORD-CHECKABLE")


def rows_of(md):
    """Table rows with an id in column 1 (first pass '1.1', second pass '1'..'32')."""
    out = []
    sec_lines = []
    for ln in md.splitlines():
        m = re.match(r"^## (\d+)\..*?\(tex:([^)]*)\)", ln)
        if m:
            sec_lines = [int(x) for x in re.findall(r"\d{2,3}", m.group(2))]
            rng = re.search(r"(\d{2,3})[–-](\d{2,3})", m.group(2))
            if rng:
                sec_lines += list(range(int(rng.group(1)), int(rng.group(2)) + 1))
        if not ln.startswith("| "):
            continue
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(c) < 5 or not re.fullmatch(r"\d+(\.\d+)?", c[0]):
            continue
        out.append(dict(id=c[0], stmt=c[1], src=c[2], verdict=c[3], fix=" | ".join(c[4:]), sec=list(sec_lines)))
    return out


def verdict_class(v):
    v = v.replace("*", "")
    tags = [t for t in VERD if t in v]
    return tags[0] if tags else "?"


def tex_lines_of(row):
    s = row["stmt"]
    lines = set()
    for grp in re.findall(r"\(([\d,;\s]*(?:also|and)?[\d,;\s]*(?:also|and)?[\d,;\s]*)\)", s):
        lines |= {int(x) for x in re.findall(r"\b\d{2,3}\b", grp)}
    for m in re.finditer(r"(?:^|\s|;)(\d{2,3})(?::|/)", s):
        lines.add(int(m.group(1)))
    lines |= {int(x) for x in re.findall(r"tex:(\d+)", s)}
    m = re.match(r"^(\d{2,3})[:/ ]", s)
    if m:
        lines.add(int(m.group(1)))
    lines = {x for x in lines if 14 <= x <= 591}
    return sorted(lines) if lines else row["sec"]


def section_A():
    hdr("A. row and verdict counts")
    res = {}
    for name in ("first_pass.md", "second_pass.md"):
        md = show(D3, PASS + name)
        rows = rows_of(md)
        cnt = collections.Counter(verdict_class(r["verdict"]) for r in rows)
        multi = [r["id"] for r in rows if sum(t in r["verdict"].replace("*", "") for t in VERD) > 1]
        summ = re.search(r"\*\*Summ[^*]*\*\*(.{0,200})", md, re.S)
        print(f"{name}: {len(rows)} rows; first verdict word: {dict(cnt)}; rows carrying two verdicts: {multi}")
        print("   stated:", summ.group(1).strip().split(".")[0] if summ else "?")
        res[name] = rows
    return res


# ───────────────────────────── B. coverage ─────────────────────────────
NUMWORDS = r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|" \
           r"seventeen|eighteen|nineteen|twenty|thirty|forty|sixty|half|third|fifth|sixth|seventh)\b"


def norm(s):
    s = s.replace("{,}", "").replace("\\,", " ").replace("$", "").replace("\\mathbf{", "").replace("\\%", "%")
    s = s.replace("−", "-").replace("–", "-").replace("--", "-").replace("\\geq", ">=").replace("\\leq", "<=")
    s = re.sub(r"\\times\s*10\^\{(-?\d+)\}", r"e\1", s)
    s = re.sub(r"(\d),(\d{3})\b", r"\1\2", s)
    s = s.replace("\\_", "_").replace("\\texttt{", "").replace("~", " ")
    return s


def tokens(s):
    s = norm(s)
    toks = re.findall(r"\[-?\+?[\d.]+, ?[-+]?[\d.]+\]|\d+(?:\.\d+)?e-\d+|\d+/\d+|-?\+?\d+\.\d+-\+?-?\d+\.\d+|"
                      r"\d+-\d+|[-+]?\d+\.\d+|\b[0-9a-f]{7}\b|\b\d+\b", s)
    toks = [t for t in toks if not re.fullmatch(r"[0-9a-f]{7}", t) or re.search(r"[a-f]", t) and re.search(r"\d", t)]
    return toks + re.findall(NUMWORDS, s)


def added_text():
    """{final line number: added text} for v9 -> final, word-level within changed hunks."""
    a, b = show(V9, TEX9).splitlines(), show(D3, TEX10).splitlines()
    out = {}
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        old = " ".join(a[i1:i2]).split()
        for j in range(j1, j2):
            new = b[j].split()
            ws = difflib.SequenceMatcher(None, old, new, autojunk=False)
            add = [" ".join(new[q1:q2]) for t2, p1, p2, q1, q2 in ws.get_opcodes() if t2 in ("insert", "replace")]
            if add:
                out[j + 1] = " ".join(add)
    return out, b


LATEX_ONLY = {14, 46, 48, 49, 50, 51, 52, 56, 74}


def section_B(rows):
    hdr("B. coverage of the numbers in the added text (final tex line: token -> rows)")
    add, final = added_text()
    allrows = [("P1 " + r["id"], r) for r in rows["first_pass.md"]] + [("P2 " + r["id"], r) for r in rows["second_pass.md"]]
    for tag, r in allrows:
        r["_lines"] = tex_lines_of(r)
        r["_text"] = norm(" ".join((r["stmt"], r["src"], r["fix"])))
    uncovered = collections.OrderedDict()
    summary = []
    for ln in sorted(add):
        if ln in LATEX_ONLY:
            continue
        toks = tokens(add[ln])
        on_line = [(t, r) for t, r in allrows if ln in r["_lines"]]
        n_cov = n_weak = 0
        miss = []
        for t in dict.fromkeys(toks):
            hit = [tg for tg, r in on_line if t in r["_text"]]
            if hit:
                n_cov += 1
                continue
            weak = [tg for tg, r in allrows if t in r["_text"]]
            if weak:
                n_weak += 1
                miss.append(f"{t}(weak: {','.join(weak[:3])})")
            else:
                miss.append(t)
        summary.append((ln, len(dict.fromkeys(toks)), n_cov, n_weak, [tg for tg, _ in on_line]))
        if miss:
            uncovered[ln] = miss
    for ln, n, c, w, rr in summary:
        print(f"tex:{ln}: {n} distinct tokens, {c} in a row citing the line, {w} only in a row citing another line; "
              f"rows citing it: {', '.join(rr) if rr else 'NONE'}")
    print("\n-- tokens in no row citing their line (weak = the token appears in some row on another line):")
    for ln, miss in uncovered.items():
        print(f"tex:{ln}: {'; '.join(miss)}")
    return add, final


# ───────────────────────────── C. the drafts' changed lines ─────────────────────────────
def section_C(rows):
    hdr("C. changed lines between the drafts and the passes that cite them")
    t1, t2, t3 = (show(r, TEX10).splitlines() for r in (D1, D2, D3))
    print("line counts:", len(t1), len(t2), len(t3))
    for (x, y, nx, ny, chk) in ((t1, t2, D1, D2, "second_pass.md"), (t2, t3, D2, D3, None)):
        ch = [i + 1 for i in range(len(x)) if x[i] != y[i]]
        print(f"{nx} -> {ny}: {len(ch)} changed lines {ch}")
        if chk:
            cited = sorted({l for r in rows[chk] for l in r["_lines"]})
            print(f"   lines the second pass's rows cite: {cited}")
            print(f"   changed but cited by no second-pass row: {[l for l in ch if l not in cited]}")
        else:
            for l in ch:
                print(f"   tex:{l}: no pass ran on {ny}; word diff:")
                ws = difflib.SequenceMatcher(None, x[l - 1].split(), y[l - 1].split(), autojunk=False)
                for t, a1, a2, b1, b2 in ws.get_opcodes():
                    if t != "equal":
                        print(f"      [-{' '.join(x[l-1].split()[a1:a2])}-] {{+{' '.join(y[l-1].split()[b1:b2])}+}}")


# ───────────────────────────── D. follow-through ─────────────────────────────
FOLLOW = [
    # (item, line, phrase that should be gone, phrase expected)
    ("P1 item 1 / row 12.1", 495, "A referee recomputed the numbers of Revisions~8 and~9 and of this revision's additions",
     "not yet refereed by session~V"),
    ("P1 item 1 / row 1.9", 44, "the referee's checks of all of it", "the referee's checks of the first three of these"),
    ("P1 item 2 / row 11.6", 481, "three of the ten window-gate runs had finished", "which the records cannot fix"),
    ("P1 item 2 / row 10.4", 424, "pre-run note for S75 said no window-gate run preceded it", "L's RESULT block"),
    ("P1 item 3 / row 4.6", 245, "one of three seeds", "one of its two seeds"),
    ("P1 item 3 / row 10.6", 455, "one of three seeds", "one of its two seeds"),
    ("P1 item 4 / row 4.1", 245, "(six at each width)", "six one-to-one and four merged"),
    ("P1 item 4 / row 4.3", 245, None, "$-0.02$ to $+0.41$"),
    ("P1 item 5 / row 5.17", 264, "within two of", "the two copies have not been compared on the same seeds"),
    ("P1 item 5 / row 5.18", 264, "and above its", None),
    ("P1 item 6 / row 7.2", 283, "built in the first updates", "computed once before training"),
    ("P1 item 7 / row 1.6", 37, None, "its delta arm from the specification alone"),
    ("P1 item 7 / row 1.20", 69, None, "the delta arm from the specification alone"),
    ("P1 item 7 / row 10.8", 472, None, "the delta arm from the specification alone"),
    ("P1 item 7 / row 1.6 (495)", 495, None, "its delta arm from the specification alone"),
    ("P1 item 8 / row 11.4", 481, "on both cards", "the cards' CUDA CHECKs were run on an earlier driver"),
    ("P1 item 8 / row 11.7 + P2 item 3", 500, "specified, reviewed and checked, with its pilot", "CUDA CHECKs to rerun"),
    ("P1 item 9 / row 1.10", 44, None, "three of the four specifications"),
    ("P1 item 9 / row 1.10", 366, None, "three of the four specifications"),
    ("P1 item 10 / row 3.4", 217, "``no cost'' all apply", "``costs binding'' does not"),
    ("P1 item 10 / row 1.16", 67, None, "on twenty new seeds ($16/20$ against $17/20$"),
    ("P1 item 11 / row 9.2", 366, None, "55 match, 7 partial, 3 not record-checkable"),
    ("P1 item 11 / row 9.3", 366, "the pooled D1, D2 and D4 pairs", "the D2 no-room count and D4's $p$"),
    ("P1 item 12 / row 9.1", 364, None, "the scope of R's 44 unstated details"),
    ("P1 item 13 / row 4.10", 245, "stands as it was", "inconclusive, weaker than handed back"),
    ("P1 item 14 / row 8.2", 347, None, "and the test's docstring"),
    ("P1 item 14 / row 6.11", 377, "collapses to zero", "to about zero everywhere but the first layer's value positions"),
    ("P1 item 14 / row 6.11 (495)", 495, "collapses to zero", None),
    ("P1 item 14 / row 2.7", 181, "$57/60$", "never pooled with it"),
    ("P1 item 14 / row 2.9", 181, None, "after this run and S75b's CUDA CHECKs"),
    ("P1 item 14 / row 10.5", 426, "15\\,GB", "that the author attributes to the OOM killer"),
    ("P1 item 14 / row 3.5", 217, None, "60 recorded split-arm runs"),
    ("P1 item 14 / row 9.8", 366, None, "was committed after those reviews"),
    ("P1 row 4.2", 245, None, "the merged four $0/4$"),
    ("P1 row 5.3 (GPU job, NRC)", 264, "with a GPU job beside it", None),
    ("P1 row 11.3 (I's own reviews)", 481, None, "session~I's own"),
    ("P2 item 1 / row 31", 481, "readable anywhere", "in git until \\texttt{b649751}"),
    ("P2 item 2 / rows 4, 32", 44, "checked once", "\\texttt{docs/review\\_2026-10/rev10/}"),
    ("P2 item 2 / rows 4, 32", 495, "await the referee", "\\texttt{docs/review\\_2026-10/rev10/}"),
]


def section_D():
    hdr("D. follow-through on the final text (d2b7abd)")
    fin = show(D3, TEX10).splitlines()
    for item, ln, gone, want in FOLLOW:
        L = fin[ln - 1]
        g = "n/a" if gone is None else ("STILL PRESENT" if gone in L else "gone")
        w = "n/a" if want is None else ("present" if want in L else "ABSENT")
        print(f"{item:34s} tex:{ln}: old wording {g}; new wording {w}")
    # the three places the CPU-check / review-points wording must agree
    for ln in (481, 500):
        print(f"tex:{ln} S75b clause:", re.search(r"S75b[^.]*?(pilot[^.;]*)", fin[ln - 1]).group(0)[:260])
    # any line still saying the referee checked this revision's additions
    for i, L in enumerate(fin, 1):
        if re.search(r"referee recomputed the numbers of Revisions~8 and~9 and of this revision", L):
            print(f"tex:{i}: still says a referee recomputed this revision's additions")
        if "every number recomputed from the records" in L:
            print(f"tex:{i}: 'every number recomputed from the records' (added at {D3}, checked by no pass)")


# ───────────────────────────── E. recomputation ─────────────────────────────
def br_recorded(r):
    """The recompute_* scripts' BOUND ROUTED: recorded transition, recorded end.one_to_one."""
    e = r.get("end") or {}
    return bool(r.get("ok") and r.get("transition") is not None and e.get("one_to_one")
                and min(e["stream_acc"]) >= 0.9)


def by_arm(store):
    out = collections.defaultdict(dict)
    for k, r in store["runs"].items():
        p = k.split("|")
        out[p[0]][int(p[1])] = r
    return out


def compare_defs(label, store, arms):
    A = by_arm(store)
    for a in arms:
        rec = {s: br_recorded(r) for s, r in A[a].items()}
        ref = {s: rlib.bound_routed(r) for s, r in A[a].items()}
        tr = sum(1 for r in A[a].values() if r.get("ok") and rlib.transition(r["curve"]) != r.get("transition"))
        o2o = sum(1 for r in A[a].values() if r.get("end") and r["end"].get("stream_gate")
                  and rlib.one_to_one(r) != bool(r["end"].get("one_to_one")))
        diff = sorted(s for s in rec if rec[s] != ref[s])
        print(f"  {label} {a}: recorded-field BR {sum(rec.values())}/{len(rec)}, rlib BR {sum(ref.values())}/{len(ref)}; "
              f"transition(curve) != recorded on {tr}; map(stream_gate) one-to-one != recorded on {o2o}; seeds differing {diff}")
    return A


def mcn(b, c):
    return rlib.binom_tail(b + c, b) if b + c else 1.0


def section_E():
    hdr("E. recomputation with the referee's definitions")
    # test_delta_recipe X, L, pooled
    arms = ("HEBB_SPLIT", "HEBB_SLOW", "DELTA_SPLIT", "DELTA_SLOW")
    S = {m: jshow(MAIN, f"results/{m}/delta_recipe_results.json") for m in ("X", "L")}
    BR = {}
    for m in ("X", "L"):
        A = compare_defs(f"delta_recipe {m}", S[m], arms)
        BR[m] = {a: {s: rlib.bound_routed(r) for s, r in A[a].items()} for a in arms}
        BR[m]["_tr"] = {a: {s: rlib.transition(r["curve"]) for s, r in A[a].items()} for a in arms}
    cnt = {a: sum(sum(BR[m][a].values()) for m in "XL") for a in arms}
    print("  pooled counts (rlib):", cnt)
    for nm, x, y in (("D1", "DELTA_SPLIT", "HEBB_SPLIT"), ("D2", "DELTA_SPLIT", "DELTA_SLOW"), ("D4", "HEBB_SPLIT", "HEBB_SLOW")):
        b = sum(BR[m][x][s] and not BR[m][y][s] for m in "XL" for s in BR[m][x])
        c = sum(BR[m][y][s] and not BR[m][x][s] for m in "XL" for s in BR[m][x])
        print(f"  {nm} {x} vs {y}: {b} vs {c}, one-sided p {mcn(b, c):.3g}")
    rows = [(BR[m]["_tr"]["HEBB_SPLIT"][s], BR[m]["_tr"]["DELTA_SPLIT"][s]) for m in "XL" for s in BR[m]["HEBB_SPLIT"]
            if BR[m]["HEBB_SPLIT"][s] and BR[m]["DELTA_SPLIT"][s]]
    e = sum(d < h for h, d in rows); l_ = sum(d > h for h, d in rows)
    print(f"  D3: n {len(rows)}, earlier {e}, later {l_}, tied {len(rows)-e-l_}, p {mcn(e, l_):.5f}, "
          f"medians {statistics.median([d for _, d in rows])} vs {statistics.median([h for h, _ in rows])}")
    for a in arms:
        lo, hi = rlib.wilson(cnt[a], 40) if hasattr(rlib, "wilson") else (None, None)
        print(f"  Wilson {a} {cnt[a]}/40: [{lo:.3f}, {hi:.3f}]" if lo is not None else f"  {a}: rlib has no wilson")
    # GPU
    G = jshow("origin/claude/explore-I", "explore_out/I/gpu/mainline/test_window_gate_NVIDIA_GeForce_GTX_1660_SUPER_results.json")
    A = compare_defs("GPU", G, ("WIN3_SPLIT_D8", "WIN3_SLOW_D8", "HINGE_D8"))
    # H S87a and S86
    H2 = jshow("origin/claude/explore-H", "explore_out/H/allmass2_results.json")
    H1 = jshow("origin/claude/explore-H", "explore_out/H/allmass_results.json")
    A2 = compare_defs("H S87a", H2, ("BASE", "ALLMASS"))
    A1 = compare_defs("H S86", H1, ("BASE", "ALLMASS"))
    for lab, AA in (("S87a", A2), ("S86", A1)):
        b = sum(rlib.bound_routed(AA["ALLMASS"][s]) and not rlib.bound_routed(AA["BASE"][s]) for s in AA["BASE"])
        c = sum(rlib.bound_routed(AA["BASE"][s]) and not rlib.bound_routed(AA["ALLMASS"][s]) for s in AA["BASE"])
        print(f"  {lab}: ALLMASS only {b}, BASE only {c}")
    f = [ch for r in A2["ALLMASS"].values() for ch in r.get("checks", []) if ch.get("fired")]
    print(f"  S87a ALLMASS firings {len(f)}, key_ok {sum(bool(c.get('key_ok')) for c in f)}, all_ok {sum(bool(c.get('all_ok')) for c in f)}")
    # guard count over the main line's split arms (and every other arm that fires, for scope)
    tot = collections.Counter()
    for p in ("results/X/window_gate_results.json", "results/L/window_gate_results.json",
              "results/X/split_copy_results.json", "results/L/split_copy_results.json"):
        st = jshow(MAIN, p)
        for a, runs in by_arm(st).items():
            f = [c for r in runs.values() for c in (r.get("checks") or []) if c.get("fired")]
            if not f:
                continue
            one = sum(1 for c in f if c.get("map_before") and len(set(c["map_before"])) == len(c["map_before"]))
            print(f"  guard {p.split('/')[1]}/{p.split('/')[2][:12]} {a}: runs {len(runs)} firings {len(f)} one-to-one before {one}")
            if a == "WIN3_SPLIT_D8":
                tot.update(runs=len(runs), fired=len(f), one=one)
    print("  guard total over WIN3_SPLIT_D8:", dict(tot))
    # the screens no committed recompute script covers
    F1 = jshow("origin/claude/explore-F", "explore_out/F/write_results.json")
    compare_defs("F S88a", F1, ("DELTA_FIXED", "DELTA_LEARNED", "DELTA_LEARNED_SLOW"))
    F2 = by_arm(jshow("origin/claude/explore-F", "explore_out/F/horizon_results.json"))
    for a in sorted(F2):
        print(f"  F S88b {a}: recorded 'bound' {sum(bool(r.get('bound')) for r in F2[a].values())}/{len(F2[a])}, "
              f"rlib bound(curve) {sum(rlib.bound(r) for r in F2[a].values())}")
    G7 = by_arm(jshow("origin/claude/explore-G", "explore_out/G/s87_results.json"))
    for a in sorted(G7):
        print(f"  G S87 {a}: recorded 'discovered' {sum(bool(r.get('discovered')) for r in G7[a].values())}/{len(G7[a])}, "
              f"rlib discovered {sum(rlib.discovered(r) for r in G7[a].values())}, rlib BR "
              f"{sum(rlib.bound_routed(r) for r in G7[a].values() if (r.get('end') or {}).get('stream_gate'))}")
    E = by_arm(jshow("origin/claude/outside-ideas", "explore_out/oracle_d128_results.json"))
    for a in sorted(E):
        rr = E[a]
        print(f"  E S69 {a}: recorded transition {sum(r.get('transition') is not None for r in rr.values())}/{len(rr)}, "
              f"rlib bound {sum(rlib.bound(r) for r in rr.values())}; bound seeds "
              f"{sorted(s for s, r in rr.items() if rlib.bound(r))}; final acc "
              f"{[round(r['acc'], 3) for s, r in sorted(rr.items())]}")
    C = by_arm(jshow("origin/claude/outside-ideas", "explore_out/capacity_results.json"))
    for a in sorted(C):
        if "ORC" in a and "16" in a:
            print(f"  E S67 {a}: seeds {sorted(C[a])}, transitions {[C[a][s].get('transition') for s in sorted(C[a])]}")


# ───────────────────────────── F. pre-registration order ─────────────────────────────
SCREENS = [
    ("F S88a", "origin/claude/explore-F", ["explore_f_write.py", "explore_f_write_child.py"], "explore_out/F/write_results.json"),
    ("F S88b", "origin/claude/explore-F", ["explore_f_horizon.py", "explore_f_horizon_child.py"], "explore_out/F/horizon_results.json"),
    ("G S87", "origin/claude/explore-G", ["explore_g_s87.py", "explore_g_s87_child.py"], "explore_out/G/s87_results.json"),
    ("H S87a", "origin/claude/explore-H", ["explore_h_allmass2.py"], "explore_out/H/allmass2_results.json"),
    ("H S87c", "origin/claude/explore-H", ["explore_h_dose.py", "explore_h_dose_child.py"], "explore_out/H/dose_results.json"),
    ("E batch 20", "origin/claude/outside-ideas", ["explore_batch20.py"], "explore_out/s67_continue_results.json"),
    ("E batch 21", "origin/claude/outside-ideas", ["explore_batch21.py", "explore_oracle_d128.py"], "explore_out/oracle_d128_results.json"),
]


def commits(ref, path):
    return [l.split(" ", 1) for l in git("log", "--reverse", "--format=%h %cI", ref, "--", path).splitlines()]


def section_F():
    hdr("F. pre-registration order of the fifth-round screens (git)")
    for name, ref, scripts, rec in SCREENS:
        rc = commits(ref, rec)
        first_rec = rc[0] if rc else None
        line = [f"{name}: first record commit {first_rec}"]
        for s in scripts:
            sc = commits(ref, s)
            after = [c for c in sc if first_rec and c[1] > first_rec[1]]
            line.append(f"{s}: first {sc[0]}, last {sc[-1]}, commits after the first record {len(after)}")
        st = jshow(ref, rec)
        prov = (st.get("meta") or {}).get("provenance")
        started = prov.get("started") if isinstance(prov, dict) else None
        shas = sorted({r.get("code_sha") or r.get("sha") for r in st["runs"].values()})
        line.append(f"meta.provenance.started {started}; code shas in records {[x[:12] if x else x for x in shas]}")
        print("\n  ".join(line))


# ───────────────────────────── G. seed freshness ─────────────────────────────
RANGES = {"F S88a 2400-2409": range(2400, 2410), "F S88b 2420-2429": range(2420, 2430),
          "H S87a 2600-2619": range(2600, 2620), "H S87c 2630-2639": range(2630, 2640),
          "G S87 2700-2729": range(2700, 2730)}
OWN = {"F S88a 2400-2409": "explore_out/F/write", "F S88b 2420-2429": "explore_out/F/horizon",
       "H S87a 2600-2619": "explore_out/H/allmass2", "H S87c 2630-2639": "explore_out/H/dose",
       "G S87 2700-2729": "explore_out/G/s87"}


def section_G():
    hdr("G. seed freshness over every remote branch tip (run keys ARM|seed|..., json and jsonl)")
    refs = [r.strip() for r in git("branch", "-r", "--format=%(refname:short)").splitlines() if "HEAD" not in r]
    hits = collections.defaultdict(set)
    for ref in refs:
        for p in git("ls-tree", "-r", "--name-only", ref).splitlines():
            if not (p.endswith(".json") or p.endswith(".jsonl")):
                continue
            try:
                txt = show(ref, p)
            except subprocess.CalledProcessError:
                continue
            keys = re.findall(r'"([A-Za-z0-9_]+)\|(\d{3,4})(?:\|[^"]*)?"', txt)
            seeds = {int(s) for _, s in keys}
            for lab, rg in RANGES.items():
                if seeds & set(rg):
                    hits[lab].add((p, ref.replace("origin/claude/", "")))
    for lab in RANGES:
        other = sorted(x for x in hits[lab] if not x[0].startswith(OWN[lab]) and "dry_" not in x[0])
        print(f"  {lab}: files with these seeds other than the screen's own: {other if other else 'none'}")


# ───────────────────────────── H. the recompute files ─────────────────────────────
def section_H():
    hdr("H. the recompute_* files as committed")
    for f in ("recompute_delta.py", "recompute_gpu.py", "recompute_H.py"):
        src = show(D3, PASS + f)
        paths = re.findall(r'"((?:the main-line clone|branch )[^"]*)"', src)
        print(f"  {f}: data paths {paths} -> runnable from the repository: {all(os.path.exists(os.path.join(ROOT, p)) for p in paths)}")
    out = show(D3, PASS + "recompute_H_out.txt")
    print(f"  recompute_H_out.txt: contains a Traceback: {'Traceback' in out}; "
          f"'TOTAL runs' lines {out.count('TOTAL runs')}; 'allmass2 arms' lines {out.count('allmass2 arms')}; "
          f"'WIN1200 discovered' lines {out.count('WIN1200 discovered')}")
    src = show(D3, PASS + "recompute_H.py")
    m = re.search(r"\n\s*arms = (.*)", src)
    print(f"  recompute_H.py line building 'arms': {m.group(1) if m else None}")
    sp = show(D3, PASS + "second_pass.md")
    print(f"  second_pass.md cites files not committed: {sorted(set(re.findall(r'`(worddiff2.txt|rev10_first_pass.md)`', sp)))}")
    fp = show(D3, PASS + "first_pass.md")
    print(f"  first_pass.md points the screens' recomputation to recompute_delta_out.txt: "
          f"{'recompute_delta_out.txt' in fp.split('What must change')[1][:1200]}")


def main():
    rows = section_A()
    section_B(rows)
    section_C(rows)
    section_D()
    section_E()
    section_F()
    section_G()
    section_H()


if __name__ == "__main__":
    main()
