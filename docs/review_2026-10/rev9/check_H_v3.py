#!/usr/bin/env python3
"""check_H_v3.py — the referee's third pass: what the THIRD version of the Revision 9 draft says about Session H,
against H's records.

    python3 docs/review_2026-10/rev9/check_H_v3.py      # writes docs/review_2026-10/rev9/check_H_v3.md

A copy of check_H_v2.py (second pass, draft f67bc07) carried to the third draft, a5f67a2 (origin/claude/rev9-draft),
multichannel_hebbian_report_v9.tex. Lines: tex:NNN (9bfa85a), tex2:NNN (f67bc07), tex3:NNN (a5f67a2).
- Everything check_H_v2.py did is kept and still asserted: every first-pass row's recomputation, its fragment on its
  first-draft line, its second-draft anchor (frag2). The second-pass rows, readings and descriptive tables are built as
  before.
- The section 'THIRD DRAFT' then locates every row in the third draft by text search on its second-draft anchor. Where
  the sentence was rewritten, OV re-anchors the row on the new wording (asserted present in tex3), states the new text,
  re-checks the changed number against the same record and sets the status. A row whose anchor is gone and that has no
  OV entry stops the script, so no rewritten sentence is passed over. NEW3 adds rows for sentences that only the third
  draft has (S84's fifth arm and 'costs binding', S58's marked 'shape, not count') and one newly checkable clause (S85's
  'the stream routing having formed first', from the dense cells' eta^2 by stream and routing margin).
- The second-pass status of each row is read from check_H_v2.md.
Records: origin/claude/explore-H, read with `git show <ref>:<path>` (rlib.load) at the commits the draft names (report 3
7e9197f, report 4 72f0bbc, report 5 df57087, report 6 37c1fbf, report 7 1e09ecd); every store is compared with the
branch head. The draft's numbers are typed in as 'draft says' only; no draft number is an input to a recomputation.
Stdlib only, read-only except for the .md written next to this file.
Definitions: docs/review_2026-10/rlib.py; fail_class_v2 re-implemented from ba8901a:fail_class_v2.py l.40-69.
"""
import collections
import difflib
import json
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import rlib  # noqa: E402
from rlib import (load, runs, bound, bound_routed, discovered, transition, fail_class, outcome_k, mcnemar1,  # noqa: E402
                  pairs, med, shared_max, git, MARGIN_PARTIAL, ETA_SPLIT)

BR = "origin/claude/explore-H"
HEAD = git("rev-parse", "--short=7", BR).strip()
R3, R4, R5, R6, R7 = "7e9197f", "72f0bbc", "df57087", "37c1fbf", "1e09ecd"
DRAFT1, DRAFT2, DRAFT3, DRAFT_PATH = "9bfa85a", "f67bc07", "a5f67a2", "multichannel_hebbian_report_v9.tex"
TEX = git("show", f"{DRAFT1}:{DRAFT_PATH}").split("\n")     # first draft (tex:NNN)
TEX2 = git("show", f"{DRAFT2}:{DRAFT_PATH}").split("\n")    # second draft (tex2:NNN)
TEX3 = git("show", f"{DRAFT3}:{DRAFT_PATH}").split("\n")    # third draft (tex3:NNN)
OUT = os.path.join(HERE, "check_H_v3.md")
ROWS = []      # (old line, new location, item, draft says, recomputed, status, first-pass status, note)
READ = []      # readings vs docstring rows
DESC = []      # descriptive statements rows
INFO = []
ORDER = ("DISAGREE", "PARTIAL", "NOT RECORD-CHECKABLE", "MATCH")


def P(name):
    return f"explore_out/H/{name}_results.json"


def esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


# first-pass statuses, read from check_H.md (key: old line, escaped item)
FIRST = {}
for _l in open(os.path.join(HERE, "check_H.md")):
    _m = re.match(r"^\| tex:(\d+) \| (.*?) \| .* \| \*\*([A-Z -]+)\*\* \|", _l)
    if _m:
        FIRST[(int(_m.group(1)), _m.group(2))] = _m.group(3)


def q(n, frag):
    """Assert a fragment is on FIRST-draft line n (the first pass's citation)."""
    if frag not in TEX[n - 1]:
        raise AssertionError(f"tex:{n} does not contain {frag!r}")


def where2(frag):
    """Lines of the second draft that contain frag (text search)."""
    return [i + 1 for i, l in enumerate(TEX2) if frag in l]


def add(n, item, draft, rec, status, note="", frag=None, frag2=None):
    """n = first-draft line (None for a new row). frag is asserted on tex:n; the row is located in the second draft by
    frag2 if given (must be found: the reworded sentence) else by frag; not found -> REMOVED."""
    if frag and n is not None:
        q(n, frag)
    if frag2 is not None:
        loc = where2(frag2)
        if not loc:
            raise AssertionError(f"tex2 does not contain {frag2!r}")
    else:
        loc = where2(frag) if frag else []
    first = FIRST.get((n, esc(item)), "—") if n is not None else "new"
    if n is not None and first == "—":
        raise AssertionError(f"no first-pass row for tex:{n} {item!r}")
    if not loc:
        status, newloc = "REMOVED", "REMOVED"
    else:
        newloc = ", ".join(f"tex2:{x}" for x in loc)
    ROWS.append((n, newloc, item, str(draft), str(rec), status, first, note, frag2 if frag2 is not None else frag))


def new(n2frag, item, draft, rec, status, note=""):
    """A row for a new or changed sentence of the second draft only; n2frag must be found."""
    add(None, item, draft, rec, status, note, frag2=n2frag)


def ms(b):
    return "MATCH" if b else "DISAGREE"


def p3(p):
    return f"{p:.3g}"


# ── v2 classifier (ba8901a:fail_class_v2.py l.40-69) ─────────────────────────────────────────────────────────────
def fail_class_v2(e):
    if e["eta_key_by_key"] >= ETA_SPLIT:
        return "KEY"
    if e["margin"] >= MARGIN_PARTIAL:
        return "STREAM-PARTIAL"
    if max(e["eta_key_by_half"], e["eta_key_by_index"]) >= ETA_SPLIT:
        return "POSITION"
    return "OTHER"


def fail_class_k_v2(r):
    e = r["end"]
    return fail_class_v2(dict(margin=e["margin"], eta_key_by_key=e["etak_key_by_key"],
                              eta_key_by_half=e["etak_key_by_block"], eta_key_by_index=e["etak_key_by_index"]))


def outcome_v2(r):
    if bound(r):
        return "BOUND ROUTED" if bound_routed(r) else "BOUND NOT routed"
    fc = fail_class_k_v2(r)
    if fc == "STREAM-PARTIAL":
        return f"MERGED ({shared_max(r)} share)" if shared_max(r) >= 2 else "STREAM-PARTIAL one-to-one"
    return f"non-stream {fc}"


def collapsed_class(o):
    """explore_h_common.fclass: MERGED (n share) -> MERGED, non-stream X -> X."""
    if o.startswith("MERGED"):
        return "MERGED"
    if o.startswith("non-stream "):
        return o.split(" ", 1)[1]
    if o.startswith("BOUND NOT"):
        return "BOUND-NOT-ROUTED"
    return o


def fired(r):
    return [c for c in (r.get("checks") or []) if c.get("fired")]


def label_ok(c):
    m = c["map_before"]
    return m.count(c["key_cs"]) >= 2 and m.count(c["key_c0"]) == 0


def blob(ref, path):
    try:
        return subprocess.check_output(("git", "-C", rlib.ROOT, "rev-parse", f"{ref}:{path}"),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except subprocess.CalledProcessError:
        return None


def exists(ref, path):
    return blob(ref, path) is not None


# ══ stores at the commits the draft names, and whether they changed since ══════════════════════════════════════════
STORES = {"S58": (R3, "collapse"), "S58b": (R3, "eight"), "S60": (R3, "stability"), "S61": (R3, "copy2x2"),
          "S62": (R3, "query_gate"), "S63": (R3, "hardness"), "S78": (R4, "two"), "S79": (R5, "constants"),
          "S84": (R6, "guard"), "S85": (R7, "two_b")}
ST = {k: load(ref, P(n)) for k, (ref, n) in STORES.items()}
for k, (ref, n) in STORES.items():
    INFO.append(f"{k}: `{P(n)}` at {ref}: {len(ST[k]['runs'])} records; blob at {ref} "
                f"{'==' if blob(ref, P(n)) == blob(BR, P(n)) else '!='} blob at the branch head {HEAD}.")


def A(screen, arm):
    return runs(ST[screen], arm)


def cnt(rs, f):
    return sum(1 for r in rs.values() if f(r))


# ══ S58 (collapse) / S58b (eight) / S62 (query) — tex:207, 279, 409, 189, 60 ═══════════════════════════════════════
S58A = {a: A("S58", a) for a in ("SPLIT", "RESET", "GUMBEL_W", "GUMBEL_RW", "SWITCH", "NONE", "ORACLE")}
br58 = {a: cnt(S58A[a], bound_routed) for a in S58A if a != "ORACLE"}
seeds58 = sorted(S58A["NONE"])
add(207, "S58: configuration and seeds", "four streams, k = 16 (40 paired seeds)",
    f"S=4 records k={S58A['NONE'][420]['k']}, cfg {S58A['NONE'][420]['cfg']}; seeds {seeds58[0]}-{seeds58[-1]} "
    f"({len(seeds58)}), every learned arm on the same seeds: {all(sorted(S58A[a]) == seeds58 for a in br58)}",
    ms(len(seeds58) == 40 and all(sorted(S58A[a]) == seeds58 for a in br58) and S58A['NONE'][420]['k'] == 16),
    frag="40 paired seeds")
add(207, "S58 NONE bound routed; every arm but one at the ceiling", "NONE 38/40; every arm but one at the ceiling",
    "; ".join(f"{a} {br58[a]}/40" for a in br58), ms(br58["NONE"] == 38 and sorted(br58.values())[0] == br58["GUMBEL_RW"]
                                                     and sorted(br58.values())[1] >= 36),
    "the one exception is GUMBEL_RW", frag="NONE $38/40$")


def only(ra, rb, f=bound_routed):
    a, b, n = pairs(ra, rb, f)
    return a, b


s_g = only(S58A["SPLIT"], S58A["GUMBEL_W"])
g_n = only(S58A["GUMBEL_W"], S58A["NONE"])
s_r = only(S58A["SPLIT"], S58A["RESET"])
s_n = only(S58A["SPLIT"], S58A["NONE"])
w_n = only(S58A["SWITCH"], S58A["NONE"])
R1 = (s_g[0] - s_g[1] <= 1) and (g_n[0] >= 4 and g_n[1] == 0)
R2 = s_r[0] - s_r[1] <= 1
R2q = s_n[0] - s_n[1] <= 1
R3_ = w_n[0] >= 4 and w_n[1] == 0
s58_read = (f"R1 {'holds' if R1 else 'does not apply'} (SPLIT only {s_g[0]} - GUMBEL_W only {s_g[1]}; GUMBEL_W vs NONE "
            f"{g_n[0]} vs {g_n[1]}); R2 {'holds by rule' if R2 else 'does not apply'} (SPLIT vs RESET {s_r[0]} vs {s_r[1]})"
            f"{', printed uninformative by its pre-fixed qualifier (SPLIT vs NONE %d vs %d)' % s_n if R2q else ''}; "
            f"R3 {'holds' if R3_ else 'does not apply'} (SWITCH vs NONE {w_n[0]} vs {w_n[1]})")
add(207, "S58 readings", "two of its three readings did not apply and the third, 'the split is the reset', held by its rule "
    "(0 vs 0) and was printed uninformative by its own qualifier", s58_read,
    "MATCH" if (not R1 and not R3_ and R2 and R2q and s_r == (0, 0)) else "DISAGREE",
    "first pass: 'no reading could apply' (PARTIAL); reworded to the docstring's outcome (bbfea60 l.48-58)",
    frag="no reading could apply", frag2="held by its rule (0 vs 0) and was printed uninformative")

# selection rule (69d01dc l.12-15) recomputed from S58
dev = ("RESET", "GUMBEL_W", "GUMBEL_RW", "SWITCH")
tmed = {a: med([transition(r["curve"]) for r in S58A[a].values() if bound_routed(r)]) for a in dev}
mrg = {a: sum(1 for r in S58A[a].values() if outcome_k(r).startswith("MERGED")) for a in dev}
sel = sorted(dev, key=lambda a: (-br58[a], tmed[a] if tmed[a] is not None else 1e9, mrg[a], dev.index(a)))[:2]
meta_sel = ST["S58b"]["meta"]["selected"]
add(207, "S58b arm selection (rule fixed at 69d01dc)", "carried the two best-scoring devices, the reset and write-gate noise (it "
    "ranked the four devices only; the no-device arm's 38/40 outscored write-gate noise's 37/40)",
    f"rule (devices only, most BOUND ROUTED): {sel} (RESET {br58['RESET']}, GUMBEL_W {br58['GUMBEL_W']}, SWITCH "
    f"{br58['SWITCH']}, GUMBEL_RW {br58['GUMBEL_RW']}); store meta selected {meta_sel}; NONE {br58['NONE']} and SPLIT "
    f"{br58['SPLIT']} are outside the candidate set", ms(sel == meta_sel == ["RESET", "GUMBEL_W"] and br58["NONE"] == 38
                                                   and br58["GUMBEL_W"] == 37),
    "first pass PARTIAL ('two best-scoring arms' without the devices-only rule); the rule and NONE's 38/40 now stated",
    frag="two best-scoring arms", frag2="it ranked the four devices only")

# GUMBEL_RW
o_rw = collections.Counter(collapsed_class(outcome_k(r)) for r in S58A["GUMBEL_RW"].values() if not bound_routed(r))
add(207, "GUMBEL_RW (noise on read and write gates) merged at four streams", "33 of 40 runs merged",
    f"{o_rw['MERGED']} of 40 MERGED (outcome_k); failures {dict(o_rw)}; BOUND ROUTED {br58['GUMBEL_RW']}/40",
    ms(o_rw["MERGED"] == 33), frag="33 of 40 runs merged")


def trans_br(rs):
    return [transition(r["curve"]) for r in rs.values() if bound_routed(r)]


def mmed(rs, f=lambda r: True):
    return statistics.median([r["end"]["margin"] for r in rs.values() if f(r)])


tg, tn, ts = med(trans_br(S58A["GUMBEL_W"])), med(trans_br(S58A["NONE"])), med(trans_br(S58A["SPLIT"]))
add(207, "GUMBEL_W at four streams: count, transition", "37/40 ... later transition (median 6000 against 3600)",
    f"BOUND ROUTED {br58['GUMBEL_W']}/40; median transition of BOUND ROUTED runs GUMBEL_W {tg:.0f}, NONE {tn:.0f}, "
    f"SPLIT {ts:.0f}", ms(br58["GUMBEL_W"] == 37 and tg == 6000 and tn == 3600), frag="median 6000 against 3600")
mg_all, mn_all = mmed(S58A["GUMBEL_W"]), mmed(S58A["NONE"])
add(207, "GUMBEL_W lower margin", "a lower margin",
    f"median end margin (all 40) GUMBEL_W {mg_all:.3f} vs NONE {mn_all:.3f}", ms(mg_all < mn_all),
    frag="and a lower margin")
E = {a: A("S58b", a) for a in ("SPLIT", "RESET", "GUMBEL_W", "ORACLE")}
gw8 = cnt(E["GUMBEL_W"], bound_routed)
sh8 = sorted((shared_max(r) for r in E["GUMBEL_W"].values()), reverse=True)
sg8 = only(E["SPLIT"], E["GUMBEL_W"])
add(207, "GUMBEL_W at eight streams", "0/10, up to six streams sharing a channel (10 vs 0 against the split; no no-device arm)",
    f"BOUND ROUTED {gw8}/10; max streams on one channel per run {sh8}; SPLIT vs GUMBEL_W {sg8[0]} vs {sg8[1]} "
    f"(one-sided p {p3(mcnemar1(*sg8))}); S58b arms {sorted(ST['S58b']['meta']['arms'])}",
    ms(gw8 == 0 and sh8[0] == 6 and sg8 == (10, 0) and "NONE" not in ST["S58b"]["meta"]["arms"]),
    frag="up to six streams sharing")


def keymap(r):
    """KEY-position map at the end: argmax of keyroute.end.stream_gate_key (recorded ch_map_key as check)."""
    kr = r["keyroute"]["end"]
    rec = [max(range(len(row)), key=lambda c: row[c]) for row in kr["stream_gate_key"]]
    return rec, kr["ch_map_key"]


def kv_diff(rs):
    out, mism = 0, 0
    for r in rs.values():
        if not bound_routed(r):
            continue
        km, kmr = keymap(r)
        if km != kmr:
            mism += 1
        if kmr != r["end"]["ch_map"]:
            out += 1
    return out, mism


sw_d, sw_m = kv_diff(S58A["SWITCH"])
msw, mnone = mmed(S58A["SWITCH"]), mmed(S58A["NONE"])
add(207, "SWITCH (balance loss) at four streams", "36/40, ... in 34 of those 36 runs ... different channels at key positions "
    "than at value positions; margin fell from 0.87 to 0.46",
    f"BOUND ROUTED {br58['SWITCH']}/40; KEY-position map != VAL-position map in {sw_d} of them (end keyroute.ch_map_key vs "
    f"end.ch_map; the recorded ch_map_key differs from the argmax of the rounded stream_gate_key on {sw_m} runs, "
    f"rounding ties); median end margin (all 40) "
    f"NONE {mnone:.3f}, SWITCH {msw:.3f}", ms(br58["SWITCH"] == 36 and sw_d == 34 and round(mnone, 2) == 0.87
                                               and round(msw, 2) == 0.46), frag="from $0.87$ to $0.46$")
add(207, "SWITCH not carried to eight streams", "was not carried to eight streams",
    f"S58b selected {meta_sel} (SWITCH ranked third: {br58['SWITCH']} BOUND ROUTED)", ms("SWITCH" not in meta_sel),
    frag="was not carried to eight streams")
other = {a: kv_diff(S58A[a])[0] for a in ("SPLIT", "RESET", "GUMBEL_W", "GUMBEL_RW", "NONE")}
other8 = {a: kv_diff(E[a])[0] for a in ("SPLIT", "RESET", "GUMBEL_W")}
add(409, "KEY-map != VAL-map in bound-routed runs", "34 of the balance-loss arm's 36 bound-routed runs, and two to five of the "
    "37-39 bound-routed runs in each of S58's other full arms",
    f"SWITCH {sw_d}/{br58['SWITCH']}; elsewhere in S58: " + ", ".join(f"{a} {other[a]}/{br58[a]}" for a in other)
    + "; S58b: " + ", ".join(f"{a} {other8[a]}/{cnt(E[a], bound_routed)}" for a in other8),
    ms(sw_d == 34 and all(2 <= other[a] <= 5 for a in ("SPLIT", "RESET", "GUMBEL_W", "NONE"))
       and sorted(br58[a] for a in ("SPLIT", "RESET", "GUMBEL_W", "NONE"))[0] == 37
       and sorted(br58[a] for a in ("SPLIT", "RESET", "GUMBEL_W", "NONE"))[-1] == 39),
    "first pass PARTIAL ('two to five runs per arm elsewhere' included GUMBEL_RW 0/5 and S58b 0); now restricted to S58's "
    "full arms (SPLIT, RESET, GUMBEL_W, NONE)", frag="34 of the balance-loss arm", frag2="two to five of the 37--39 bound-routed runs")

# S62: query map vs body maps, at each rerun's last evaluation
Q = ST["S62"]["runs"]
qa = collections.defaultdict(lambda: dict(n=0, both=0, val=0, br=0, br_both=0, repro=0))
for k, r in Q.items():
    arm = r["arm"]
    last = r["query"]["end"] if "end" in r["query"] else r["query"][str(max(int(s) for s in r["query"]))]
    d = qa[arm]
    d["n"] += 1
    d["both"] += bool(last["eq_val"] and last["eq_key"])
    d["val"] += bool(last["eq_val"])
    if bound_routed(r):
        d["br"] += 1
        d["br_both"] += bool(last["eq_val"] and last["eq_key"])
s62txt = "; ".join(f"{a}: query = both maps {d['both']}/{d['n']} bound runs (= VAL map {d['val']}; among BOUND ROUTED "
                   f"{d['br_both']}/{d['br']})" for a, d in sorted(qa.items()))
non_sw = [a for a in qa if a.startswith("collapse.") and a != "collapse.SWITCH"]
big = [a for a in non_sw if qa[a]["n"] >= 30]
eight_both = sum(qa[a]["both"] for a in qa if a.startswith("eight."))
eight_n = sum(qa[a]["n"] for a in qa if a.startswith("eight."))
full = [a for a in big]
add(207, "S62: the query's read gate follows the write map", "In the other arms with a full set of runs ... (session H's S62, "
    "descriptive: in 32-37 of the 38-40 bound runs per arm at four streams, and 12/12 at eight)",
    s62txt, ms(min(qa[a]["both"] for a in full) == 32 and max(qa[a]["both"] for a in full) == 37
               and min(qa[a]["n"] for a in full) == 38 and max(qa[a]["n"] for a in full) == 40
               and (eight_both, eight_n) == (12, 12)),
    f"'other arms with a full set of runs' = {sorted(full)} (SWITCH is the arm the sentence contrasts; GUMBEL_RW has "
    f"{qa['collapse.GUMBEL_RW']['n']} bound runs); now counted over bound runs and marked descriptive. First pass PARTIAL "
    "('32-36', which were BOUND ROUTED counts)", frag="in 32--36 of the bound runs per arm",
    frag2="in 32--37 of the 38--40 bound runs per arm")

# S58b row tex:189
br8 = {a: cnt(E[a], bound_routed) for a in ("SPLIT", "RESET", "GUMBEL_W")}
sr8 = only(E["SPLIT"], E["RESET"])
add(189, "S58b row: seeds, copy+reset, reset only, paired", "460-469; 10/10; 2/10; 8 vs 0, one-sided p = 0.0039",
    f"seeds {min(E['SPLIT'])}-{max(E['SPLIT'])}; SPLIT {br8['SPLIT']}/10, RESET {br8['RESET']}/10; SPLIT only {sr8[0]}, "
    f"RESET only {sr8[1]}, one-sided p {mcnemar1(*sr8):.4f}; k={E['SPLIT'][460]['k']}",
    ms(sorted(E["SPLIT"]) == list(range(460, 470)) and br8["SPLIT"] == 10 and br8["RESET"] == 2 and sr8 == (8, 0)
       and round(mcnemar1(*sr8), 4) == 0.0039), frag="8 vs 0, one-sided $p = 0.0039$")
rf8 = [c for r in E["RESET"].values() for c in fired(r)]
add(195, "S58b reset-only arm fired under the live rule, on labelled targets", "fired under the live rule ... labelled on target",
    f"RESET fired {len(rf8)} times in {sum(1 for r in E['RESET'].values() if fired(r))}/10 runs; labelled ok (c* >= 2 "
    f"streams, c0 none on map_before) {sum(label_ok(c) for c in rf8)}/{len(rf8)}; W_g unchanged on "
    f"{sum(bool(c.get('w_g_unchanged')) for c in rf8)}", ms(all(label_ok(c) for c in rf8) and rf8),
    frag="fired under the live rule")

# tex:279 synthesis 39/40^e
add(279, "S=4 'window gate + SLOW + split' = S58 SPLIT (footnote e)", "39/40 (S58, exploratory, 40 seeds)",
    f"S58 SPLIT BOUND ROUTED {br58['SPLIT']}/{len(S58A['SPLIT'])} (seeds {min(S58A['SPLIT'])}-{max(S58A['SPLIT'])}; "
    f"S=4, P=4, k=16, conv, 28800 updates with early stop)", ms(br58["SPLIT"] == 39 and len(S58A["SPLIT"]) == 40),
    "the column's other cells pool test_window_gate X+L; S58's budget is 28800 with the run path's early stop "
    "(collapse docstring l.20-21), stated nowhere in the caption (unchanged in the second draft)", frag="$39/40^{e}$")
q(283, "Session~H's S58, exploratory, 40 seeds")
assert where2("Session~H's S58, exploratory, 40 seeds") == [285]

# ══ S60 (stability) — tex:207 ═══════════════════════════════════════════════════════════════════════════════════
S60A = {a: A("S60", a) for a in ("REF", "TEMP_FLOOR", "STABLEMAX", "EMA_EVAL")}
taus = []
for s, r in sorted(S60A["TEMP_FLOOR"].items()):
    m = r["meas"]
    ks = sorted(m, key=int)
    taus.append((s, m[ks[0]]["tau"], max(m[k]["tau"] for k in ks), m[ks[-1]]["tau"]))
add(207, "S60 learned temperature with a floor", "ran to the floor in every run (k = 2, no convolution, ten seeds)",
    f"TEMP_FLOOR runs {len(taus)} (k={S60A['TEMP_FLOOR'][470]['k']}); tau at the first evaluation "
    f"{min(t[1] for t in taus):.3f}-{max(t[1] for t in taus):.3f}, max over all evaluations {max(t[2] for t in taus):.3f}, "
    f"end {min(t[3] for t in taus):.3f}-{max(t[3] for t in taus):.3f} (floor 0.5)",
    ms(len(taus) == 10 and max(t[2] for t in taus) <= 0.5), frag="ran to the floor in every run")
d_sm = only(S60A["STABLEMAX"], S60A["REF"], discovered)


def sat_before(rs):
    n = k = 0
    for r in rs.values():
        t = transition(r["curve"])
        if t is None:
            continue
        for st, m in r["meas"].items():
            if int(st) < t:
                n += 1
                k += bool(m["saturated"])
    return k, n


sbs, sbr = sat_before(S60A["STABLEMAX"]), sat_before(S60A["REF"])
add(207, "S60 stable-max", "a stable-max softmax read 'no change at n = 10' by its rule (7/10 against 8/10)",
    f"DISCOVERED STABLEMAX {cnt(S60A['STABLEMAX'], discovered)}/10 vs REF {cnt(S60A['REF'], discovered)}/10 "
    f"({d_sm[0]} vs {d_sm[1]}); D1 (>= 3 vs 0 either way) -> 'no change at n = 10'; saturated before transition "
    f"{sbs[0]}/{sbs[1]} vs {sbr[0]}/{sbr[1]} -> D2 'does not delay saturation'",
    ms(cnt(S60A['STABLEMAX'], discovered) == 7 and cnt(S60A['REF'], discovered) == 8 and d_sm[0] < 3 and d_sm[1] < 3),
    "first pass PARTIAL ('changed nothing'); now D1's label with its n. 7/10 and 8/10 are DISCOVERED counts",
    frag="a stable-max softmax changed nothing", frag2="no change at $n = 10$'' by its rule ($7/10$ against $8/10$)")

# ══ S63 (hardness) — tex:207 ════════════════════════════════════════════════════════════════════════════════════
S63A = {a: A("S63", a) for a in ("REF", "ANNEAL", "HARD_W", "ORACLE")}
b63 = {a: cnt(S63A[a], bound_routed) for a in ("REF", "ANNEAL", "HARD_W")}
an = only(S63A["ANNEAL"], S63A["REF"])
earlier = sum(1 for s in S63A["REF"] if bound_routed(S63A["REF"][s]) and bound_routed(S63A["ANNEAL"][s])
              and transition(S63A["ANNEAL"][s]["curve"]) < transition(S63A["REF"][s]["curve"]))
helps_a = (an[0] >= 4 and an[1] == 0) or (earlier >= 8 and b63["ANNEAL"] >= b63["REF"])
add(207, "S63 slope (temperature) annealing", "Slope annealing of the gate (S63) ... read 'does not help' (9/10 against 10/10)",
    f"ANNEAL {b63['ANNEAL']}/10, REF {b63['REF']}/10 ({an[0]} vs {an[1]}); earlier on {earlier} seeds both bound routed; "
    f"pre-fixed reading -> {'hardness helps' if helps_a else 'does not help'}", "MATCH" if not helps_a and
    b63["ANNEAL"] == 9 and b63["REF"] == 10 else "DISAGREE",
    "'neutral' replaced by the pre-fixed 'does not help'. Still: the arm is a temperature anneal 1 -> 0.25 (hardness "
    "docstring l.16-17), called 'slope annealing' in the draft",
    frag="neutral ($9/10$ against $10/10$)", frag2="both read ``does not help'' ($9/10$ against $10/10$")
hw = collections.Counter(collapsed_class(outcome_k(r)) for r in S63A["HARD_W"].values() if not bound_routed(r))
hw_m = sorted(round(r["end"]["margin"], 2) for r in S63A["HARD_W"].values() if outcome_k(r).startswith("MERGED"))
add(207, "S63 hard one-hot write gate", "0/10; KEY 3, MERGED 4, OTHER 3",
    f"HARD_W {b63['HARD_W']}/10; failures {dict(sorted(hw.items()))}", ms(b63["HARD_W"] == 0 and hw == {"KEY": 3, "MERGED": 4,
                                                                                                       "OTHER": 3}),
    frag="KEY 3, MERGED 4, OTHER 3")
hw_p = only(S63A["HARD_W"], S63A["REF"])
helps_h = (hw_p[0] >= 4 and hw_p[1] == 0)
hdoc = git("show", f"{R3}:explore_h_hardness.py")
new("a hard, one-hot write gate from update 2400 both read", "S63 HARD_W: from update 2400; reading; discordant pairs",
    "a hard, one-hot write gate from update 2400 ... read 'does not help' (0/10, 0 vs 10)",
    f"HARD_W only {hw_p[0]}, REF only {hw_p[1]} (one-sided p REF>HARD_W {p3(mcnemar1(hw_p[1], hw_p[0]))}); pre-fixed "
    f"-> {'hardness helps' if helps_h else 'does not help'}; docstring: 'from update 2400 on the write gate one-hot': "
    f"{'from update 2400 on the write gate one-hot' in hdoc}",
    ms(not helps_h and hw_p == (0, 10) and "from update 2400 on the write gate one-hot" in hdoc),
    "the draft still calls the 0 vs 10 arm 'does not help' (the pre-fixed label) rather than 'destroyed routing'")
add(207, "S63 HARD_W mechanism", "With the write one-hot the gate gets gradient only through the read, by construction; that "
    "the writes' stream labelling therefore never forms is the session's description, and four of the ten runs did form "
    "a partial one (MERGED needs a margin of 0.25)",
    f"4 of the 10 HARD_W runs end MERGED, i.e. STREAM-PARTIAL by margin (end margins {hw_m}, all >= 0.25) with "
    f"streams sharing channels", ms(len(hw_m) == 4 and min(hw_m) >= 0.25),
    "first pass PARTIAL (unmarked mechanism, contradicted by the 4 MERGED runs); now marked as the session's description "
    "with the 4 runs stated", frag="never forms", frag2="four of the ten runs did form a partial one")

# ══ S61 (copy2x2) — tex:191, 195, 201, 347 ═══════════════════════════════════════════════════════════════════════
C = {a: A("S61", a) for a in ("SPLIT", "RESET", "COPY", "COPY_NONOISE", "NONE", "ORACLE")}
bc = {a: cnt(C[a], bound_routed) for a in ("SPLIT", "RESET", "COPY", "COPY_NONOISE", "NONE")}
cn_r = only(C["COPY_NONOISE"], C["RESET"])
cn_s = only(C["COPY_NONOISE"], C["SPLIT"])
add(191, "S61 row: seeds and counts", "480-489; copy+reset 9/10; copy only 9/10; with noise 9/10; reset only 4/10; none 3/10",
    f"seeds {min(C['SPLIT'])}-{max(C['SPLIT'])}; SPLIT {bc['SPLIT']}, COPY_NONOISE {bc['COPY_NONOISE']}, COPY "
    f"{bc['COPY']}, RESET {bc['RESET']}, NONE {bc['NONE']} (of 10)",
    ms(sorted(C["SPLIT"]) == list(range(480, 490)) and [bc[a] for a in ("SPLIT", "COPY_NONOISE", "COPY", "RESET", "NONE")]
       == [9, 9, 9, 4, 3]), frag="480--489")
add(191, "S61 row: copy only vs reset only; vs copy+reset", "5 vs 0, p = 0.031; 0 vs 0",
    f"COPY_NONOISE vs RESET {cn_r[0]} vs {cn_r[1]} (one-sided p {mcnemar1(*cn_r):.4f}); COPY_NONOISE vs SPLIT {cn_s[0]} vs "
    f"{cn_s[1]}", ms(cn_r == (5, 0) and round(mcnemar1(*cn_r), 3) == 0.031 and cn_s == (0, 0)),
    "neither comparison is one the docstring printed (f630ef2 l.26-31: R1/R3 on COPY vs SPLIT, R2 COPY vs COPY_NONOISE, "
    "RESET vs SPLIT and every arm vs NONE); the second draft now marks the cell '(ours, not pre-fixed)'",
    frag="5 vs 0, $p = 0.031$", frag2="5 vs 0, $p = 0.031$ (ours, not pre-fixed)")
cs = only(C["COPY"], C["SPLIT"])
cc = only(C["COPY"], C["COPY_NONOISE"])
r1 = cs[1] - cs[0] <= 1
r3 = abs(cs[1] - cs[0]) <= 1
r2 = cc[0] >= 4 and cc[1] == 0
add(195, "S61 pre-fixed readings and the arm they read", "'the copy alone suffices' and 'the reset adds nothing' are one-pair "
    "bounds on the copy with noise against the full split (0 vs 0), and both held; the exact copy enters only 'the noise "
    "matters' (>= 4 vs 0), which did not apply (0 vs 0); exact copy vs full split (0 vs 0) was not a reading",
    f"R1 {'holds' if r1 else 'does not apply'} and R3 {'holds' if r3 else 'does not apply'} on COPY (copy WITH noise, Adam "
    f"untouched) vs SPLIT: SPLIT only {cs[1]}, COPY only {cs[0]}; R2 {'holds' if r2 else 'does not apply'} (COPY vs "
    f"COPY_NONOISE {cc[0]} vs {cc[1]}); the exact copy COPY_NONOISE vs SPLIT is {cn_s[0]} vs {cn_s[1]} (not a reading)",
    ms(r1 and r3 and not r2 and cs == (0, 0) and cc == (0, 0) and cn_s == (0, 0)),
    "first pass PARTIAL (R1/R3 attributed to the exact copy); now attributed to COPY, the noisy copy, as in f630ef2 "
    "l.19, 28-30", frag="are one-pair bounds that both held", frag2="are one-pair bounds on the copy \\emph{with} noise")
add(195, "reset-only arm bound no more often than the no-trigger arm, in every look where one exists",
    "it bound no more often than the arm with no trigger in test_split_copy and the window_gate control, and more often in "
    "S61 (4/10 against 3/10, 2 vs 1)",
    f"S61 (session H): RESET {bc['RESET']}/10 vs NONE {bc['NONE']}/10, RESET only "
    f"{only(C['RESET'], C['NONE'])[0]}, NONE only {only(C['RESET'], C['NONE'])[1]}; S58b has no NONE arm",
    ms(bc["RESET"] == 4 and bc["NONE"] == 3 and only(C['RESET'], C['NONE']) == (2, 1)),
    "first pass DISAGREE (B1); S61 is now stated as the exception with its counts. The other two looks (split_copy X, "
    "window_gate control) are main-line rows tex2:190, 192, not H's records", frag="bound no more often than the arm with no trigger",
    frag2="and more often in S61 ($4/10$ against $3/10$, 2 vs 1)")
f61 = {a: [c for r in C[a].values() for c in fired(r)] for a in ("SPLIT", "RESET", "COPY", "COPY_NONOISE")}
nf = sum(len(v) for v in f61.values())
nok = sum(label_ok(c) for v in f61.values() for c in v)
ncopy = sum(1 for a in ("SPLIT", "COPY", "COPY_NONOISE") for c in f61[a] if c.get("row_dist_after", 1) < c["row_dist_before"])
nwg = sum(bool(c.get("w_g_unchanged")) for c in f61["RESET"])
add(201, "S61 firings", "All 83 firings across the four trigger arms were on labelled targets, a channel holding at least two "
    "streams and an empty one; the 57 in the three copying arms copied the row, the 26 resets did not",
    f"{nf} firings ({', '.join(f'{a} {len(v)}' for a, v in f61.items())}); labelled ok {nok}/{nf}; a row copy in "
    f"{ncopy} (SPLIT, COPY, COPY_NONOISE: row_dist_after < row_dist_before); the {len(f61['RESET'])} RESET firings copy "
    f"nothing (w_g_unchanged true on {nwg})", ms(nf == 83 and nok == 83 and ncopy == 57 and nwg == 26 == len(f61["RESET"])),
    "first pass DISAGREE (B2: 'copied' for all 83)", frag="All 83 firings", frag2="the 57 in the three copying arms copied the row")
miss = {a: sorted(s for s, r in C[a].items() if not bound_routed(r)) for a in ("SPLIT", "COPY", "COPY_NONOISE")}
same9 = len({tuple(v) for v in miss.values()}) == 1
o483 = {a: outcome_k(C[a][483]) for a in miss}
add(201, "S61 three forms bound the same nine seeds, missed the same one", "missed 483, two streams sharing",
    f"failures {miss}; at 483 {o483}", ms(same9 and miss["SPLIT"] == [483] and all(v == "MERGED (2 share)"
                                                                                for v in o483.values())),
    frag="483, two streams sharing")
add(347, "'the reset adds nothing' shown by S61 on ten seeds", "session H's S61 holds it as a one-pair bound on ten seeds (the "
    "copy with noise 9/10, the exact copy 9/10, the full split 9/10)",
    f"R3 held as a one-pair bound on COPY (noisy copy) vs SPLIT ({cs[1]} vs {cs[0]}); COPY {bc['COPY']}/10, exact copy "
    f"{bc['COPY_NONOISE']}/10, SPLIT {bc['SPLIT']}/10", ms(r3 and [bc[a] for a in ("COPY", "COPY_NONOISE", "SPLIT")] == [9, 9, 9]),
    "first pass PARTIAL ('shows'); now 'holds it as a one-pair bound', with the noisy-copy arm named first",
    frag="session~H's S61 shows it on ten seeds", frag2="S61 holds it as a one-pair bound on ten seeds")

# ══ S78 (two) — tex:226, 209, 228, 115 ═════════════════════════════════════════════════════════════════════════════
T = {a: A("S78", a) for a in ("REF", "SPLIT2", "SPLIT2_NOISE", "K4", "ORACLE")}
succ = {"REF": discovered, "SPLIT2": discovered, "SPLIT2_NOISE": discovered, "K4": bound_routed}
bt = {a: cnt(T[a], succ[a]) for a in succ}
add(226, "S78 seeds and counts", "1300-1319; baseline 15/20, exact copy 16/20, copy with noise and reset 17/20, four "
    "channels 15/20", f"seeds {min(T['REF'])}-{max(T['REF'])}; DISCOVERED REF {bt['REF']}, SPLIT2 {bt['SPLIT2']}, "
    f"SPLIT2_NOISE {bt['SPLIT2_NOISE']}; K4 BOUND ROUTED {bt['K4']} (of 20)",
    ms(sorted(T["REF"]) == list(range(1300, 1320)) and [bt[a] for a in succ] == [15, 16, 17, 15]),
    "the draft writes 'discovered' for all four; K4's outcome is BOUND ROUTED (explore_h_two.py l.35-36)",
    frag="1300--1319")
vr = {a: only({s: succ[a](r) for s, r in T[a].items()}, {s: discovered(r) for s, r in T["REF"].items()}, lambda v: v)
      for a in ("SPLIT2", "SPLIT2_NOISE", "K4")}
best = min(vr, key=lambda a: mcnemar1(*vr[a]))
neither = all(mcnemar1(*vr[a]) >= 0.05 for a in vr)
add(226, "S78 reading", "no arm beats the baseline (best 2 vs 0, p = 0.25); 'neither device'",
    "; ".join(f"{a} vs REF {vr[a][0]} vs {vr[a][1]} p {p3(mcnemar1(*vr[a]))}" for a in vr)
    + f" -> R4 {'neither device' if neither else 'does not apply'} (best {best})",
    ms(neither and vr[best] == (2, 0) and mcnemar1(*vr[best]) == 0.25), frag="best 2 vs 0, $p = 0.25$")

FT = []
for a in ("SPLIT2", "SPLIT2_NOISE"):
    for s, r in sorted(T[a].items()):
        cks = r.get("checks") or []
        for i, c in enumerate(cks):
            if c.get("fired"):
                nx = cks[i + 1] if i + 1 < len(cks) else None
                FT.append(dict(arm=a, seed=s, step=c["step"], before=c["diag"], after=c["diag_after"],
                               next=nx["diag"] if nx else None, next_step=nx["step"] if nx else None, r=r,
                               km_after=c.get("key_masses_after"), rd=c.get("row_dist_after")))
sp = [d for d in FT if d["arm"] == "SPLIT2"]
zero = all(max(d["after"][k] for k in ("eta_key_by_key", "eta_key_by_stream", "eta_val_by_key", "eta_val_by_stream")) < 0.005
           for d in sp)
unif = all(d["rd"] == 0.0 and all(abs(m - 0.5) < 1e-3 for m in d["km_after"]) for d in sp)
add(226, "S78 exact copy: rows equal, gate uniform, every eta^2 0.00", "the two rows are equal and the gate is uniform, every "
    "eta^2 at 0.00", f"{len(sp)} SPLIT2 firings: row_dist_after 0 and key-position masses 0.5/0.5 on all: {unif}; every "
    f"eta^2 (KEY/VAL by key/stream) < 0.005 right after: {zero}", ms(zero and unif), frag="every $\\eta^2$ at $0.00$")
key_seeds = sorted(s for s, r in T["SPLIT2"].items() if not bound(r) and fail_class(r) == "KEY")
key_seeds_v2 = sorted(s for s, r in T["SPLIT2"].items() if not bound(r) and fail_class_v2(r["end"]) == "KEY")
fk = [d for d in sp if d["seed"] in key_seeds]


def full(e):
    return e is not None and e["eta_key_by_key"] >= 0.995 and e["eta_key_by_stream"] <= 0.005


with_next = [d for d in fk if d["next"] is not None]
no_next = [d for d in fk if d["next"] is None]
add(226, "S78 KEY split re-formed exactly by the next check (2400 later) after all firings on the KEY-ending seeds",
    "after all eight firings on the three seeds that ended KEY under the classifier of record (six read at the next check, "
    "2400 updates later; two, fired at the last check, read from the end of the run)",
    f"seeds ending KEY (v1) {key_seeds} (v2: {key_seeds_v2}); firings on them {len(fk)}; with a next check "
    f"{len(with_next)}, all 1.00/0.00 there: {all(full(d['next']) for d in with_next)}; without a next check "
    f"{[(d['seed'], d['step']) for d in no_next]} (last check 21600), end statistic (24000) 1.00/0.00: "
    f"{all(full(d['r']['end']) for d in no_next)}",
    ms(len(fk) == 8 and len(key_seeds) == 3 and len(with_next) == 6 and len(no_next) == 2
       and all(full(d['next']) for d in with_next) and all(full(d['r']['end']) for d in no_next)),
    "first pass PARTIAL (the two end-of-run readings unstated); now stated, and 'under the classifier of record' names v1 "
    "(under v2 1317 also ends KEY; its three firings are the half-key row)",
    frag="after all eight firings on the three seeds that ended KEY", frag2="two, fired at the last check, read from the end of the run")
f17 = [d for d in sp if d["seed"] == 1317]
half = all(d["next"] and abs(d["next"]["eta_key_by_key"] - 0.5) < 0.015 and abs(d["next"]["eta_key_by_stream"] - 0.25) < 0.015
           for d in f17)
add(226, "S78 half-key pattern after all three firings on a fourth seed", "the half-key pattern after all three on a fourth",
    f"1317: {len(f17)} firings, next-check KEY eta^2 by key/stream "
    f"{[(round(d['next']['eta_key_by_key'], 2), round(d['next']['eta_key_by_stream'], 2)) for d in f17]}",
    ms(half and len(f17) == 3), frag="the half-key pattern after all three on a fourth")
resc = sorted(s for s in T["REF"] if discovered(T["SPLIT2"][s]) and not discovered(T["REF"][s]))
pre = [(d["seed"], round(d["before"]["eta_key_by_key"], 2), round(d["before"]["eta_key_by_stream"], 2)) for d in sp
       if d["seed"] in resc]
add(226, "S78 the one rescue", "a seed whose gate was not yet a key split when the copy came",
    f"SPLIT2 discovered where REF did not: {resc}; KEY eta^2 by key/stream before its firing(s) {pre}",
    ms(len(resc) == 1 and all(p[1] < 0.5 for p in pre)), frag="The one rescue")
nz = collections.defaultdict(list)
for d in FT:
    if d["arm"] == "SPLIT2_NOISE":
        nz[d["seed"]].append(d)
nzc = {}
for s, ds in nz.items():
    r = T["SPLIT2_NOISE"][s]
    if discovered(r):
        nzc[s] = "DISCOVERED (rescued)"
    else:
        ks = [(d["next"] or r["end"])["eta_key_by_key"] for d in ds]
        nzc[s] = "fully" if all(k >= 0.995 for k in ks) else ("partly" if all(0.5 <= k < 0.995 for k in ks) else f"mixed {ks}")
add(226, "S78 noise + Adam reset", "the key split re-formed only partly on two seeds and fully on one",
    f"SPLIT2_NOISE seeds that fired: {dict(sorted(nzc.items()))}",
    ms(sum(v == "partly" for v in nzc.values()) == 2 and sum(v == "fully" for v in nzc.values()) == 1),
    "partly = KEY eta^2 by key in [0.5, 0.995) at every next check (end statistic where none); the other two firing seeds "
    "discovered", frag="re-formed only partly on two seeds and fully on one")
add(226, "S78 mechanism marked", "Session H's reading of the mechanism ... was not measured and is not a reading",
    "no Adam-moment field in any S78 record (checks hold diag, diag_after, key masses, row distances, state_zeroed)",
    ms(not any(("exp_avg" in k or "moment" in k) for a in ("SPLIT2", "SPLIT2_NOISE") for r in T[a].values()
               for c in (r.get("checks") or []) for k in c) and not any("moment" in k for a in T for r in T[a].values()
                                                                         for k in r)),
    "marked as not a reading; check keys searched for 'exp_avg'/'moment': none",
    frag="was not measured and is not a reading")
add(209, "S78 summary in Sec. split", "the copy resets the gate to uniform and, on the eight firings of the seeds that ended in "
    "a key split, the same split is back by the next check (two of the eight are read from the end of the run); one seed had "
    "no key split when the copy came and discovered",
    f"uniform after all {len(sp)} exact copies: {unif}; the same KEY split back within 2400 on the {len(fk)} firings of "
    f"the KEY-ending seeds ({len(no_next)} of them only at run end); 1309 had no key split (KEY eta^2 by key {pre}) and "
    f"discovered", ms(unif and len(fk) == 8 and len(no_next) == 2 and resc == [1309]),
    "first pass PARTIAL (no 1309 exception, no end-of-run readings); both now stated. 'Ended in a key split' is v1's "
    "count (three seeds); 1317 (KEY under v2) is left out of this summary", frag="re-forms within 2400 updates",
    frag2="the same split is back by the next check (two of the eight are read from the end of the run)")

# v1 / v2 on S78's unbound runs
unb = {}
for a in ("REF", "SPLIT2", "SPLIT2_NOISE", "K4"):
    for s, r in T[a].items():
        if not bound(r):
            if a == "K4":
                unb[f"{a}|{s}"] = (outcome_k(r), outcome_v2(r), r["end"]["etak_key_by_key"], r["end"]["margin"])
            else:
                unb[f"{a}|{s}"] = (fail_class(r), fail_class_v2(r["end"]), r["end"]["eta_key_by_key"], r["end"]["margin"])
dis = sorted(k for k, v in unb.items() if v[0] != v[1])
predicted = sorted(k for k, v in unb.items() if v[2] >= ETA_SPLIT and v[3] >= MARGIN_PARTIAL)
dirs = collections.Counter(f"{unb[k][0]} -> {unb[k][1]}" for k in dis)
add(228, "S78 unbound runs that change class", "Four of S78's seventeen unbound runs are STREAM-PARTIAL under the classifier "
    "of record and KEY under fail_class_v2 ...; no reading changes",
    f"{len(unb)} unbound; v1 (classifier of record) vs v2 differ on {len(dis)}: {dis}, {dict(dirs)}; S78's readings "
    f"R1-R4 are on DISCOVERED / BOUND ROUTED (explore_h_two.py l.40-46), not on classes",
    ms(len(unb) == 17 and len(dis) == 4 and list(dirs) == ["STREAM-PARTIAL -> KEY"]),
    "first pass PARTIAL (direction inverted); now v1 = STREAM-PARTIAL, v2 = KEY", frag="Under the classifier of record, four of S78",
    frag2="are STREAM-PARTIAL under the classifier of record and KEY under")
add(115, "v1/v2 differences on S78", "four of S78's seventeen",
    f"{len(dis)} of {len(unb)}; = the runs with eta^2 by key >= 0.5 and margin >= 0.25: {dis == predicted}",
    ms(len(dis) == 4 and len(unb) == 17 and dis == predicted), frag="four of S78's seventeen")

# the pre-run CHECK's population (explore_h_two.py l.56-58; code l.172-186 at df57087)
s43 = load(R5, "explore_out/window_recipe_results.json")["runs"]
stab = load(R5, P("stability"))["runs"]
oi = git("rev-list", "-1", "--before=2026-10-10T04:09:11Z", "origin/claude/outside-ideas").strip()[:7]
s52h = load(R5, "explore_out/two_stream_keysplits_results.json")["runs"]   # H's own copy (stale, 2026-10-08)
s52o = load(oi, "explore_out/two_stream_keysplits_results.json")["runs"]   # the CHECK's source (explore_h_two.py l.51)


def population(s52):
    recs = {}
    for k, r in s43.items():
        if r.get("transition") is None:
            recs["S43:" + k] = ("k2", r)
    for k, r in stab.items():
        if r.get("ok") and r.get("transition") is None and not k.startswith("ORACLE") and "eta_key_by_key" in r["end"]:
            recs["S60:" + k] = ("k2", r)
    for k, r in s52.items():
        if r.get("ok") and k.startswith("W2_K4|") and r.get("transition") is None:
            recs["S52:" + k] = ("outcome", r)
        elif r.get("ok") and k.startswith("W2_KEYHINGE|") and r.get("transition") is None:
            recs["S52:" + k] = ("k2", r)
    d, p_ = [], []
    for k, (kind, r) in sorted(recs.items()):
        e = r["end"]
        if kind == "k2":
            a1, a2, kk = fail_class(r), fail_class_v2(e), e["eta_key_by_key"]
        else:
            a1, a2, kk = outcome_k(r), outcome_v2(r), e["etak_key_by_key"]
        if a1 != a2:
            d.append("|".join(k.split("|")[:2]) + f" {a1} -> {a2}")
        if kk >= ETA_SPLIT and e["margin"] >= MARGIN_PARTIAL:
            p_.append("|".join(k.split("|")[:2]))
    return recs, d, p_


recs, d29, p29 = population(s52o)
recs_h, d_h, _ = population(s52h)
by = collections.Counter(k.split(":")[0] for k in recs)
add(115, "v1/v2 differences on H's recorded unbound two-stream runs (S78's pre-run CHECK population)", "six of 29 recorded "
    "unbound two-stream runs of three earlier screens",
    f"with S52's store from outside-ideas {oi} (the CHECK's source): {len(d29)} of {len(recs)} (S43 {by['S43']}, S60 "
    f"{by['S60']}, S52 {by['S52']}); differ = predicted (eta^2 by key >= 0.5 and margin >= 0.25): "
    f"{[x.split(' ')[0] for x in d29] == p29}; {d29}. With the S52 file that is on H's branch ({len(s52h)} W2_K4 records "
    f"only): {len(d_h)} of {len(recs_h)}",
    ms(len(d29) == 6 and len(recs) == 29 and len(by) == 3),
    "first pass PARTIAL ('on H's branch', where S52's store is a stale partial copy: "
    f"{len(s52h)} of {len(s52o)} records, last committed "
    f"{git('log', '-1', '--format=%h %cI', BR, '--', 'explore_out/two_stream_keysplits_results.json').strip()}); the "
    "second draft says 'of three earlier screens' (S43, S60, S52), which recomputes with S52's store on outside-ideas",
    frag="six of the 29 recorded unbound two-stream runs", frag2="six of 29 recorded unbound two-stream runs of three earlier screens")
near = []
for k, r in sorted(s52o.items()):
    if r.get("ok") and r.get("transition") is None and k.startswith("W2_KEYHINGE|") and abs(r["end"]["eta_key_by_key"] - 0.5) < 1e-3:
        near.append((k.split("|")[1], round(r["end"]["eta_key_by_key"], 6), fail_class(r), fail_class_v2(r["end"])))
near_ok = (len(near) == 3 and [n[1] for n in near if n[1] >= 0.5] == [0.500004]
           and sorted(n[1] for n in near if n[1] < 0.5) == [0.49986, 0.499985]
           and all(n[2] == "STREAM-PARTIAL" for n in near)
           and [n[3] for n in near if n[1] >= 0.5] == ["KEY"] and all(n[3] == "STREAM-PARTIAL" for n in near if n[1] < 0.5))
add(115, "three batch-18 runs at eta^2 = 0.5 by key labelled STREAM-PARTIAL by v1 (outside H's screens; S52 store at "
    f"{oi})", "labelled one batch-18 run at eta^2 = 0.500004 by key STREAM-PARTIAL (two more sit at 0.49986 and 0.499985 "
    "and keep that label under both)",
    f"S52 W2_KEYHINGE unbound runs within 1e-3 of 0.5 (seed, eta^2 by key, v1, v2): {near}", ms(near_ok),
    "first pass PARTIAL ('three ... at exactly 0.5'); now one, with the two below stated",
    frag="labelled three batch-18 runs", frag2="labelled one batch-18 run at $\\eta^2 = 0.500004$ by key STREAM-PARTIAL")
new("the key-first classifier relabels the one at $0.500004$ KEY and leaves the two just below",
    "batch 18's three boundary runs (Sec. keysplits)", "three 'STREAM-PARTIAL' failures with eta^2 by key within 1e-3 of 0.5; "
    "the key-first classifier relabels the one at 0.500004 KEY and leaves the two just below",
    f"{near}", ms(near_ok), "S52 records on claude/outside-ideas (not H's); repeated here because the S85 paragraph builds on it")
new("relabels the one at or above $0.5$ KEY", "batch 18's boundary runs (lessons)", "Three batch-18 runs sit within 1e-3 of "
    "eta^2 = 0.5 by key and were labelled STREAM-PARTIAL; the key-first classifier relabels the one at or above 0.5 KEY",
    f"{near}", ms(near_ok), "first pass: tex:388 repeated 'three at exactly 0.5' (revision_9_check.md sec. 7)")

# ══ S79 (constants) — tex:203, 115 ════════════════════════════════════════════════════════════════════════════════
K = {a: A("S79", a) for a in ("BASE", "INT1200", "CAP1", "CAP6", "PROBE16", "THR05", "ORACLE")}
bk = {a: cnt(K[a], bound_routed) for a in K if a != "ORACLE"}
fb = [c for r in K["BASE"].values() for c in fired(r)]
add(203, "S79 BASE", "9/10, 21 firings, all on target",
    f"BASE {bk['BASE']}/10; firings {len(fb)}; labelled ok (recomputed from map_before) {sum(label_ok(c) for c in fb)}",
    ms(bk["BASE"] == 9 and len(fb) == 21 and all(label_ok(c) for c in fb)), frag="21 firings, all on target")
lab = {}
for v in ("INT1200", "CAP1", "CAP6", "PROBE16", "THR05"):
    vo, bo = only(K[v], K["BASE"])
    lab[v] = ("robust" if bo - vo <= 1 else "sensitive" if (bo >= 4 and vo == 0) else "inconclusive", bo, vo)
add(203, "S79 per-variant counts and readings", "CAP1 5/10 (BASE only 4, CAP1 only 0; sensitive); INT1200 7/10 (2 vs 0; "
    "inconclusive); CAP6 9/10; PROBE16 10/10; THR05 10/10 (all robust)",
    "; ".join(f"{v} {bk[v]}/10 (BASE only {lab[v][1]}, {v} only {lab[v][2]}: {lab[v][0]})" for v in lab),
    ms([bk[v] for v in lab] == [7, 5, 9, 10, 10] and [lab[v][0] for v in lab] ==
       ["inconclusive", "sensitive", "robust", "robust", "robust"] and lab["CAP1"][1:] == (4, 0)
       and lab["INT1200"][1:] == (2, 0)), frag="(BASE only 4, CAP1 only 0")
extra = []
for s, r in sorted(K["CAP6"].items()):
    for c in fired(r)[3:]:
        extra.append((s, c["step"], label_ok(c), bound_routed(r)))
add(203, "S79 'the extra splits are harmless'", "did not apply: one of seven extra firings not on a labelled target (the run "
    "bound anyway)", f"CAP6 {bk['CAP6']} vs BASE {bk['BASE']}; extra firings {len(extra)}, not labelled ok "
    f"(seed, update, run bound routed) {[(s, st, br_) for s, st, ok_, br_ in extra if not ok_]} -> 'does not apply'",
    ms(len(extra) == 7 and sum(not e[2] for e in extra) == 1 and all(e[3] for e in extra if not e[2])),
    frag="one of the cap-of-six arm's seven extra firings")
cap1_fail = sorted(s for s, r in K["CAP1"].items() if not bound_routed(r))
cap1_f = {s: fired(K["CAP1"][s]) for s in cap1_fail}
early = {s: c[0] for s, c in cap1_f.items() if c and 4800 <= c[0]["step"] <= 12000}


def curve_at(r, step):
    return next((a for st, a, *_ in r["curve"] if st == step), None)


probe = sorted(round(c["acc"], 3) for c in early.values())
held = sorted(round(curve_at(K["CAP1"][s], c["step"]), 3) for s, c in early.items())
add(203, "S79 CAP1 failures' single split", "four of CAP1's five failures spent their single split at updates 4800-12000 while "
    "the probe read 0.09-0.50", f"CAP1 failures {cap1_fail}; fired at 4800-12000: {len(early)} "
    f"({[(s, c['step']) for s, c in sorted(early.items())]}); probe accuracy at the firing {probe}; held-out curve there {held}",
    ms(len(early) == 4 and min(probe) >= 0.09 and max(probe) <= 0.5 and round(min(probe), 2) == 0.09),
    "the draft now says 'probe', which fixes report_5.md l.45's 'held-out accuracy' (referee's recompute_H DISAGREE/PARTIAL)",
    frag="the probe read $0.09$--$0.50$")
bstream = []
for s, c in sorted(early.items()):
    r = K["CAP1"][s]
    prior = [e for e in r["stats"] if e.get("stream_acc") and e["step"] <= c["step"]]
    if prior:
        e = prior[-1]
        bstream.append((s, c["step"], e["step"], sum(a >= 0.95 for a in e["stream_acc"]), round(max(e["stream_acc"]), 2)))
nbound = [b for b in bstream if b[3] > 0]
sa_steps = sorted({e["step"] for r in K["CAP1"].values() for e in r["stats"] if e.get("stream_acc")})
late = [(s, c["step"]) for s, c in sorted(early.items()) if c["step"] == 12000]
s1329 = [e for e in K["CAP1"][late[0][0]]["stats"] if e.get("stream_acc") and e["step"] == 9600] if late else []
above99 = sum(a > 0.99 for a in s1329[0]["stream_acc"]) if s1329 else None
add(203, "S79 CAP1: 'before any stream had bound'", "(per-stream accuracy is recorded only at 4800 and 9600; the seed that "
    "fired at 12,000 had two streams above 0.99 at 9600)",
    "per firing (seed, firing update, last per-stream measurement at or before it, streams >= 0.95 there, max stream acc): "
    + str(bstream) + f"; steps with stats[].stream_acc in CAP1 records: {sa_steps}; seed firing at 12000: {late}, streams "
    f"> 0.99 at 9600: {above99} ({[round(a, 3) for a in s1329[0]['stream_acc']] if s1329 else None})",
    ms(sa_steps == [4800, 9600] and len(late) == 1 and above99 == 2),
    "first pass DISAGREE (B3: 'before any stream had bound'); the clause is gone and the exception is stated",
    frag="before any stream had bound", frag2="had two streams above $0.99$ at 9600")
i_fail = sorted(s for s, r in K["INT1200"].items() if not bound_routed(r))
i_used = {s: [c["step"] for c in fired(K["INT1200"][s])] for s in i_fail}
add(203, "S79 INT1200 failures", "the fast-check arm's three failures had used all three splits by 7200-10800",
    f"INT1200 failures {i_fail}; their firings {i_used}; third firing at {sorted(v[2] for v in i_used.values() if len(v) >= 3)}",
    ms(len(i_fail) == 3 and all(len(v) == 3 for v in i_used.values()) and
       min(v[2] for v in i_used.values()) == 7200 and max(v[2] for v in i_used.values()) == 10800),
    frag="had used all three splits by 7200")
mi, mb = med(trans_br(K["INT1200"])), med(trans_br(K["BASE"]))
add(203, "S79 medians", "median 14,400 against 24,000", f"median transition of BOUND ROUTED runs INT1200 {mi:.0f}, BASE {mb:.0f}",
    ms(mi == 14400 and mb == 24000), frag="median 14{,}400 against 24{,}000")
add(203, "S79 descriptive part marked", "'Descriptively, and not pre-registered: ...'",
    "the marker opens the CAP1/INT1200 firing-time sentences and the flat-start sentence", "MATCH",
    "the docstring's DESCRIPTIVE section (fdf7499 l.33) fixes only transitions and INT1200's first-split timing; the "
    "marker covers the rest", frag="Descriptively, and not pre-registered")
add(203, "S79 recommendation", "Our recommendation from this, not a reading: a replication should state the cap and the check "
    "schedule (the fast-check arm moved the interval, the first check and the gap together, so the three are not "
    "separated), and can state the probe size and the rise threshold loosely",
    "not a pre-fixed reading; INT1200 changed the interval, the first check (4800 -> 2400) and the gap (4800 -> 2400) "
    "together (fdf7499 l.16); the first eligible check was never varied alone; 'stated loosely' rests on 'robust' "
    "(BASE only - V only <= 1) at n = 10", "MATCH",
    "first pass PARTIAL (unmarked 'therefore'; the first eligible check never varied alone). Now marked as a "
    "recommendation and the confound stated. 'Loosely' still rests on n = 10 'robust' bounds (PROBE16, THR05: 0 vs 1)",
    frag="What a replication must state is therefore", frag2="Our recommendation from this, not a reading")
k_dis = [s for a in ("BASE", "INT1200", "CAP1", "CAP6", "PROBE16", "THR05") for s, r in K[a].items()
         if outcome_k(r) != outcome_v2(r)]
k_pred = [s for a in ("BASE", "INT1200", "CAP1", "CAP6", "PROBE16", "THR05") for s, r in K[a].items()
          if not bound(r) and r["end"]["etak_key_by_key"] >= ETA_SPLIT and r["end"]["margin"] >= MARGIN_PARTIAL]
add(115, "v1/v2 differences at eight streams in S79", "none", f"{len(k_dis)} differences; runs with eta^2 by key >= 0.5 and "
    f"margin >= 0.25: {len(k_pred)}", ms(not k_dis and not k_pred), frag="none at eight streams in S79")
# information: H's other eight-stream screens
oth = []
for sc, arms in (("S58b", ("SPLIT", "RESET", "GUMBEL_W")), ("S61", ("SPLIT", "RESET", "COPY", "COPY_NONOISE", "NONE")),
                 ("S63", ("REF", "ANNEAL", "HARD_W"))):
    for a in arms:
        for s, r in A(sc, a).items():
            if outcome_k(r) != outcome_v2(r):
                oth.append(f"{sc} {a}|{s} {outcome_k(r)} -> {outcome_v2(r)}")
INFO.append(f"v1/v2 on H's other eight-stream screens (S58b, S61, S63; information, not claimed by the draft): "
            f"{len(oth)} differences{': ' + '; '.join(oth) if oth else ''}.")

# ══ S84 / S85: first-pass status rows (first draft: running) — tex:203, 228, 464, 500 ═══════════════════════════════
gs = load(BR, P("guard")) if exists(BR, P("guard")) else None
garms = collections.Counter(k.split("|")[0] for k in (gs or {"runs": {}})["runs"])
two_b = [p for p in git("ls-tree", "-r", "--name-only", BR, "explore_out/H").split() if "two_b" in p]
chk = json.loads(git("show", f"{BR}:explore_out/H/checks.json"))
s84doc = git("show", f"{R6}:explore_h_guard.py")
s85doc = git("show", f"{R7}:explore_h_two_b.py")
add(464, "S84 = flat-start guard and evaluation-based trigger; S85 = two-stream prevention",
    "session H's flat-start guard and evaluation-based trigger (S84) and two-stream prevention (S85)",
    "explore_h_guard.py (834581a) arms GUARD, GUARD_INT1200, EVAL, EVAL_GUARD_INT1200; explore_h_two_b.py arms SLOWGATE, "
    "KEYHINGE, SLOWGATE_KEYHINGE", ms("EVAL " in s84doc and "GUARD " in s84doc),
    "the next-steps item listed S84/S85 as running; both are complete and the item is replaced by tex2:470",
    frag="flat-start guard and evaluation-based")
add(203, "S84/S81 attribution", "Session F's S81 is the same screen on delta channels",
    "explore_h_guard.py l.5-6 (834581a): 'Session F's S81 runs the same guard on delta channels and pairs with these by "
    f"name': {'S81 runs the same guard on delta channels' in s84doc.replace(chr(10), ' ')}", "MATCH",
    "first pass: '(S84, S81)' paired guard -> H, evaluation -> F loosely; now S84 carries both and S81 is F's copy. F's "
    "records not read here", frag="(S84, S81)", frag2="Session~F's S81 is the same screen on delta channels")
add(500, "S84 running", "S84-S85 running",
    f"at {HEAD}: guard_results.json {sum(garms.values())} of 52 records ({dict(garms)}); two_b store {two_b}",
    "MATCH", "status at the first draft (3b4bd17: 23 of 52 records); both screens are complete and tex2:506 cites "
    "reports 6 and 7", frag="S84--S85 running")
add(228, "S85 running on 20 seeds", "are running on 20 seeds in session H's S85",
    f"at {HEAD}: S85 store {two_b}; CHECK entry H/two_b {chk.get('H/two_b', {}).get('time')}", "DISAGREE",
    "first pass DISAGREE (A3: committed, not started, at 3b4bd17); the sentence is replaced by S85's result",
    frag="running on 20 seeds in session~H's S85")

# ══ S84 (guard; report 6, 37c1fbf) — tex2:203, 60 ══════════════════════════════════════════════════════════════════
GA = ("BASE", "GUARD", "GUARD_INT1200", "EVAL", "EVAL_GUARD_INT1200")
G = {a: A("S84", a) for a in GA + ("ORACLE",)}
gb = {a: cnt(G[a], bound_routed) for a in GA}
grec = {a: sum(r.get("outcome") == "BOUND ROUTED" for r in G[a].values()) for a in GA}
gp = {a: only(G[a], G["BASE"]) for a in GA[1:]}
gf = {a: [c for r in G[a].values() for c in fired(r)] for a in GA}
gtr = {a: med(trans_br(G[a])) for a in GA}
gcfg = {a: next(iter(G[a].values()))["trig_cfg"] for a in GA}
gorc = {s: (bound(r), transition(r["curve"])) for s, r in G["ORACLE"].items()}
INFO.append(f"S84 bound routed (rlib) {gb}; recorded outcome 'BOUND ROUTED' {grec}; trig_cfg per arm "
            + "; ".join(f"{a}: source {c['source']}, guard {c.get('guard')}, every {c['every']}, first {c['first']}, gap {c['gap']}"
                        for a, c in gcfg.items()))
new("session~H's S84, seeds 2300--2309; the oracle $2/2$", "S84 seeds and validity",
    "Two trigger variants ... screened on ten fresh seeds (session H's S84, seeds 2300-2309; the oracle 2/2)",
    f"seeds per arm {sorted({(min(G[a]), max(G[a]), len(G[a])) for a in GA})}; ORACLE (seed: bound, transition) {gorc}",
    ms(all(sorted(G[a]) == list(range(2300, 2310)) for a in GA) and len(gorc) == 2 and all(v[0] for v in gorc.values())),
    "the oracle's VALIDITY rule (guard docstring) is 'binds both seeds within 43200': both at 3600")
gl_ok = all(label_ok(c) for c in gf["GUARD"])
gl_okb = sum(label_ok(c) for c in gf["BASE"])
aims = (sum(label_ok(c) for c in gf["GUARD"]) >= 0.95 * len(gf["GUARD"])) and (gp["GUARD"][1] - gp["GUARD"][0] <= 1)
new("binds $9/10$ against the base's $9/10$ (0 vs 0)", "S84 GUARD vs BASE",
    "A guard that forbids splits while the probe is below 0.5 binds 9/10 against the base's 9/10 (0 vs 0)",
    f"GUARD trig_cfg guard {gcfg['GUARD'].get('guard')} source {gcfg['GUARD']['source']}; BOUND ROUTED GUARD {gb['GUARD']}/10, "
    f"BASE {gb['BASE']}/10; GUARD only {gp['GUARD'][0]}, BASE only {gp['GUARD'][1]}",
    ms(gcfg["GUARD"].get("guard") == 0.5 and gcfg["GUARD"]["source"] == "probe" and gb["GUARD"] == 9 and gb["BASE"] == 9
       and gp["GUARD"] == (0, 0)),
    "the body does not quote the pre-fixed reading for this arm ('the guard aims the split', which applies); the index "
    "(tex2:60) does")
base_lo = sorted(c["acc"] for c in gf["BASE"] if c["acc"] < 0.2)
g_acc = sorted(c["acc"] for c in gf["GUARD"])
new("it fires its 17 splits at probe accuracies of $0.50$--$0.91$", "S84 accuracy read at the splits",
    "it fires its 17 splits at probe accuracies of 0.50-0.91, where the base fired 7 of its 24 at 0.06-0.16",
    f"GUARD firings {len(g_acc)}, probe acc at firing {g_acc[0]:.3f}-{g_acc[-1]:.3f}; BASE firings {len(gf['BASE'])}, of "
    f"them < 0.2: {len(base_lo)} at {base_lo[0]:.3f}-{base_lo[-1]:.3f} (next BASE reading "
    f"{sorted(c['acc'] for c in gf['BASE'])[len(base_lo)]:.3f})",
    ms(len(g_acc) == 17 and round(g_acc[0], 2) == 0.50 and round(g_acc[-1], 2) == 0.91 and len(gf["BASE"]) == 24
       and len(base_lo) == 7 and round(base_lo[0], 2) == 0.06 and round(base_lo[-1], 2) == 0.16),
    "pre-fixed DESCRIPTIVE ('per arm the firings (update, accuracy read, labelled ok)', explore_h_guard.py l.41)")
guarded = ("GUARD", "GUARD_INT1200", "EVAL_GUARD_INT1200")
lock = {}
for a in guarded:
    fails = sorted(s for s, r in G[a].items() if not bound_routed(r))
    lock[a] = (fails, [(len(fired(G[a][s])), round(max(c["acc"] for c in G[a][s]["checks"]), 3),
                        sum(bool(c.get("blocked")) for c in G[a][s]["checks"]), outcome_k(G[a][s])) for s in fails],
               gcfg[a]["source"])
b06 = G["BASE"][2306]
new("it locks out the one seed whose probe never reaches $0.5$, which ends MERGED in every guarded arm",
    "S84 the locked-out seed", "it locks out the one seed whose probe never reaches 0.5, which ends MERGED in every guarded arm",
    f"per guarded arm (failures; per failure: splits fired, max accuracy read by the rule, checks blocked, outcome; "
    f"source): {lock}; BASE|2306 for comparison: {outcome_k(b06)}, {len(fired(b06))} splits, probe max "
    f"{max(c['acc'] for c in b06['checks']):.3f}",
    ms(all(v[0] == [2306] and v[1][0][0] == 0 and v[1][0][1] < 0.5 and v[1][0][3].startswith("MERGED")
           for v in lock.values())),
    "in EVAL_GUARD_INT1200 the guard reads the held-out evaluation, not the probe (max 0.378); in the unguarded BASE run "
    "the probe on 2306 did reach 0.547, after its splits")
fast = (gp["GUARD_INT1200"][1] - gp["GUARD_INT1200"][0] <= 1) and (gtr["BASE"] - gtr["GUARD_INT1200"] >= 2400)
new("Checks every 1200 with the guard bind $9/10$", "S84 GUARD_INT1200 and 'faster without the cost'",
    "Checks every 1200 with the guard bind 9/10 (S79's unguarded fast check, on other seeds, bound 7/10) with a median "
    "transition 1200 earlier than the base's, not the 2400 the pre-fixed reading 'faster without the cost' required",
    f"GUARD_INT1200 every {gcfg['GUARD_INT1200']['every']} from {gcfg['GUARD_INT1200']['first']}, gap "
    f"{gcfg['GUARD_INT1200']['gap']}; BOUND ROUTED {gb['GUARD_INT1200']}/10 ({gp['GUARD_INT1200'][0]} vs "
    f"{gp['GUARD_INT1200'][1]}); S79 INT1200 {bk['INT1200']}/10 on seeds {min(K['INT1200'])}-{max(K['INT1200'])}; median "
    f"transition of BOUND ROUTED runs GUARD_INT1200 {gtr['GUARD_INT1200']:.0f}, BASE {gtr['BASE']:.0f} -> 'faster' "
    f"{'applies' if fast else 'does not apply'}",
    ms(gb["GUARD_INT1200"] == 9 and bk["INT1200"] == 7 and gtr["BASE"] - gtr["GUARD_INT1200"] == 1200 and not fast),
    "the docstring's median is over BOUND ROUTED runs (explore_h_guard.py l.36-37), as recomputed. Not stated in the draft: "
    f"the fifth arm EVAL_GUARD_INT1200 ({gb['EVAL_GUARD_INT1200']}/10, {gp['EVAL_GUARD_INT1200'][0]} vs "
    f"{gp['EVAL_GUARD_INT1200'][1]}, median {gtr['EVAL_GUARD_INT1200']:.0f}) and the reading 'costs binding' (no arm)")
ev06, bs06 = G["EVAL"][2306]["checks"], G["BASE"][2306]["checks"]
evq = all(abs(c["acc"] * 2048 - round(c["acc"] * 2048)) < 1e-6 for r in G["EVAL"].values() for c in r["checks"])
evp = (gp["EVAL"][1] - gp["EVAL"][0] <= 1)
disc_ev = sorted(s for s in G["EVAL"] if bound_routed(G["EVAL"][s]) != bound_routed(G["BASE"][s]))
new("binds $10/10$ (1 vs 0; the reading", "S84 EVAL trigger and the discordant seed",
    "A trigger that reads the 2048-query held-out evaluation instead of the probe binds 10/10 (1 vs 0; the reading 'the "
    "evaluation is the better probe' applies by its rule, on a single discordant seed, where the probe read 0.000 then "
    "0.047 and the evaluation 0.066 then 0.068, so the probe's noise decided whether the first check fired)",
    f"EVAL source {gcfg['EVAL']['source']}, every reading a multiple of 1/2048: {evq}; BOUND ROUTED EVAL {gb['EVAL']}/10; "
    f"EVAL only {gp['EVAL'][0]}, BASE only {gp['EVAL'][1]} (discordant {disc_ev}); rule (BASE only - EVAL only) <= 1 -> "
    f"{'applies' if evp else 'does not apply'}; BASE|2306 checks {[(c['step'], round(c['acc'], 3), c['fired']) for c in bs06[:2]]}; "
    f"EVAL|2306 checks {[(c['step'], round(c['acc'], 3), c['fired']) for c in ev06[:2]]}",
    ms(gcfg["EVAL"]["source"] == "eval" and evq and gb["EVAL"] == 10 and gp["EVAL"] == (1, 0) and evp and disc_ev == [2306]
       and [round(c["acc"], 3) for c in bs06[:2]] == [0.0, 0.047] and [round(c["acc"], 3) for c in ev06[:2]] == [0.066, 0.068]
       and not bs06[1]["fired"] and ev06[1]["fired"]),
    "the reading is a one-pair non-inferiority bound named 'better'; the draft says 'applies by its rule' and gives the "
    "single discordant seed, which is the honest reading. 'The probe's noise decided' is report_6's interpretation, "
    "unmarked; the record shows only the two rises (0.047 >= 0.02, 0.002 < 0.02)")
cnt_same = all(gp[a] == (0, 0) for a in guarded)
new("These seeds hardly tested the flat start", "S84 summary: count unchanged, timing changed",
    "These seeds hardly tested the flat start, since the base kept enough splits for its merges after spending some; the "
    "guard changes when the splits fire and, on these seeds, not whether the run binds",
    f"guarded arms vs BASE: {[(a, gp[a]) for a in guarded]}; BASE bound routed {gb['BASE']}/10 with {len(base_lo)} of its "
    f"{len(gf['BASE'])} splits at probe < 0.2; first split per seed BASE "
    f"{[min((c['step'] for c in fired(G['BASE'][s])), default=None) for s in sorted(G['BASE'])]}, GUARD "
    f"{[min((c['step'] for c in fired(G['GUARD'][s])), default=None) for s in sorted(G['GUARD'])]}",
    ms(cnt_same),
    "report_6's description ('What happened'), not marked as descriptive in the draft; the count half is the 0 vs 0 "
    "pairs, the 'kept enough splits for its merges' half is the session's interpretation")
exact = [(c.get("row_dist_after"), c.get("state_zeroed")) for c in gf["BASE"]]
new("session~H's \\emph{exact copy} does neither", "the exact copy: no noise, no Adam reset (S84 BASE firings as a check)",
    "session H's exact copy does neither [noise, Adam reset], and is the form of the screens of Sections split and memory",
    f"S84 BASE firings: row_dist_after 0 on {sum(e[0] == 0.0 for e in exact)}/{len(exact)}, state_zeroed empty on "
    f"{sum(e[1] == [] for e in exact)}/{len(exact)}",
    ms(all(e[0] == 0.0 and e[1] == [] for e in exact)), "S61 COPY_NONOISE, S78 SPLIT2, S79 and S84 all use it")
new("on ten fresh seeds (session~H's S84", "S84 and S85 seeds 'fresh'", "ten fresh seeds (2300-2309); 20 fresh seeds "
    "(2310-2329)", "explore_h_guard.py l.9-12 (834581a) records the scan: the specified 2200-2229 appear in results/, "
    "2300-2359 'nowhere in explore_out/ or results/ on any branch (scanned before any run)'", "NOT RECORD-CHECKABLE",
    "the referee did not rescan every branch's result files for seeds 2300-2329; taken from the docstring")

# ══ S85 (two_b; report 7, 1e09ecd) — tex2:228, 36, 60, 316, 384, 436, 440, 465, 470 ══════════════════════════════
BA = ("REF", "SLOWGATE", "KEYHINGE", "SLOWGATE_KEYHINGE")
B = {a: A("S85", a) for a in BA + ("ORACLE",)}
bd = {a: cnt(B[a], discovered) for a in BA}
btag = {a: sum(r.get("tag") == "DISCOVERED" for r in B[a].values()) for a in BA}
bbnd = {a: cnt(B[a], bound) for a in BA}
bp = {a: only(B[a], B["REF"], discovered) for a in BA[1:]}
kf = {a: {v: sorted(s for s, r in B[a].items() if not discovered(r)
                    and (fail_class(r) if v == 1 else fail_class_v2(r["end"])) == "KEY") for v in (1, 2)} for a in BA}
prevent = {a: mcnemar1(*bp[a]) < 0.05 for a in BA[1:]}
moves = {a: len(kf[a][2]) >= len(kf["REF"][2]) - 1 for a in BA[1:]}
moves1 = {a: len(kf[a][1]) >= len(kf["REF"][1]) - 1 for a in BA[1:]}
btr = {a: med([transition(r["curve"]) for r in B[a].values() if bound(r)]) for a in BA}
borc = {s: (bound(r), transition(r["curve"])) for s, r in B["ORACLE"].items()}
INFO.append(f"S85 DISCOVERED (rlib) {bd}; recorded tag DISCOVERED {btag}; bound {bbnd}; KEY failures v1/v2 "
            + "; ".join(f"{a} {kf[a][1]}/{kf[a][2]}" for a in BA))
add(228, "S85 design", 
    "on 20 fresh seeds (2310-2329; the oracle 2/2 at 1200), under the key-first classifier: a gate held at 1e-4 for the "
    "first 1200 updates, and the hinge's key term on updates 1-2400 (batch 18's device)",
    f"seeds per arm {sorted({(min(B[a]), max(B[a]), len(B[a])) for a in BA})}; ORACLE {borc}; explore_h_two_b.py l.16 "
    f"'gate group (W_in, W_g, window) at 1e-4 for updates 1-1200': {'at 1e-4 for updates 1-1200' in s85doc}; l.17 "
    f"\"S52's W2_KEYHINGE ... updates 1-2400\": {'updates 1-2400' in s85doc}; l.26 'only moves' on KEY failures under v2: "
    f"{'KEY failures under v2' in s85doc}",
    ms(all(sorted(B[a]) == list(range(2310, 2330)) for a in BA) and all(v == (True, 1200) for v in borc.values())
       and len(borc) == 2 and "at 1e-4 for updates 1-1200" in s85doc and "updates 1-2400" in s85doc),
    "first pass MATCH (gate group at 1e-4, the key term, the key-first classifier), noting the omitted combined arm, which "
    "is now reported ('the two together'); seeds and the oracle added", frag="the gate group at $10^{-4}$",
    frag2="screened on 20 fresh seeds (2310--2329; the oracle $2/2$ at 1200)")
new("The key term discovers $\\mathbf{20/20}$ against the baseline's $15/20$", "S85 KEYHINGE and its reading",
    "The key term discovers 20/20 against the baseline's 15/20 (5 vs 0, one-sided p = 0.031) with no key failure (the "
    "baseline has four), and its pre-fixed reading 'prevention works' applies",
    f"DISCOVERED KEYHINGE {bd['KEYHINGE']}/20, REF {bd['REF']}/20; KEYHINGE only {bp['KEYHINGE'][0]}, REF only "
    f"{bp['KEYHINGE'][1]}, one-sided p {mcnemar1(*bp['KEYHINGE']):.4f} -> "
    f"{'prevention works' if prevent['KEYHINGE'] else 'not shown'}; KEY failures (v2) KEYHINGE {kf['KEYHINGE'][2]}, REF "
    f"{kf['REF'][2]}",
    ms(bd["KEYHINGE"] == 20 and bd["REF"] == 15 and bp["KEYHINGE"] == (5, 0) and round(mcnemar1(5, 0), 3) == 0.031
       and prevent["KEYHINGE"] and not kf["KEYHINGE"][2] and len(kf["REF"][2]) == 4),
    "exploratory screen, uncorrected one-sided p; the draft calls it a screen in the abstract (tex2:36) and tex2:316, 440")
new("the slow gate discovers $14/20$ (0 vs 1) with the same four key failures", "S85 SLOWGATE and its readings",
    "the slow gate discovers 14/20 (0 vs 1) with the same four key failures, reading 'prevention not shown; the key split "
    "only moves'",
    f"DISCOVERED SLOWGATE {bd['SLOWGATE']}/20; SLOWGATE only {bp['SLOWGATE'][0]}, REF only {bp['SLOWGATE'][1]} -> "
    f"{'prevention works' if prevent['SLOWGATE'] else 'prevention not shown'}; 'only moves' (v2 KEY {len(kf['SLOWGATE'][2])} "
    f">= REF {len(kf['REF'][2])} - 1) -> {moves['SLOWGATE']}; KEY seeds v2 SLOWGATE {kf['SLOWGATE'][2]}, REF {kf['REF'][2]} "
    f"(v1: SLOWGATE {kf['SLOWGATE'][1]}, REF {kf['REF'][1]})",
    "PARTIAL" if (bd["SLOWGATE"] == 14 and bp["SLOWGATE"] == (0, 1) and not prevent["SLOWGATE"] and moves["SLOWGATE"]
                  and len(kf["SLOWGATE"][2]) == len(kf["REF"][2]) == 4 and kf["SLOWGATE"][2] != kf["REF"][2]) else
    ms(bd["SLOWGATE"] == 14 and kf["SLOWGATE"][2] == kf["REF"][2]),
    "counts and readings match; 'the same four key failures' is the same NUMBER (report_7: '4 KEY failures under v2, the "
    "same as REF'), not the same seeds: "
    f"{sorted(set(kf['SLOWGATE'][2]) & set(kf['REF'][2]))} shared, SLOWGATE adds {sorted(set(kf['SLOWGATE'][2]) - set(kf['REF'][2]))}, "
    f"REF's {sorted(set(kf['REF'][2]) - set(kf['SLOWGATE'][2]))} ends STREAM-PARTIAL under the slow gate; and one of "
    "SLOWGATE's four (2323) is KEY only under v2 (eta^2 by key 0.5000005)")
sk17 = {int(k): v["etak_key_by_key"] for k, v in B["SLOWGATE_KEYHINGE"][2317]["dense"].items()}
skf = sorted(s for s, r in B["SLOWGATE_KEYHINGE"].items() if not discovered(r))
new("the two together discover $19/20$ (5 vs 1, $p = 0.109$)", "S85 SLOWGATE_KEYHINGE and its one failure",
    "the two together discover 19/20 (5 vs 1, p = 0.109), 'not shown', the one failure a seed on which the slow gate held "
    "the gate back until the hinge's window had passed and the split then formed",
    f"DISCOVERED {bd['SLOWGATE_KEYHINGE']}/20; {bp['SLOWGATE_KEYHINGE'][0]} vs {bp['SLOWGATE_KEYHINGE'][1]}, one-sided p "
    f"{mcnemar1(*bp['SLOWGATE_KEYHINGE']):.3f} -> {'works' if prevent['SLOWGATE_KEYHINGE'] else 'not shown'}; failure "
    f"{skf} ({[fail_class_v2(B['SLOWGATE_KEYHINGE'][s]['end']) for s in skf]}); its eta^2 by key max over updates <= 2400 "
    f"{max(v for t, v in sk17.items() if t <= 2400):.2f}, at 3600 {sk17.get(3600, -1):.2f}, at 4800 {sk17.get(4800, -1):.2f}; "
    f"KEYHINGE alone on 2317: {'DISCOVERED' if discovered(B['KEYHINGE'][2317]) else 'not'}",
    ms(bd["SLOWGATE_KEYHINGE"] == 19 and bp["SLOWGATE_KEYHINGE"] == (5, 1) and round(mcnemar1(5, 1), 3) == 0.109
       and skf == [2317] and max(v for t, v in sk17.items() if t <= 2400) < 0.5 and sk17.get(4800, 0) > 0.5),
    "'the slow gate held the gate back until the window had passed' is report_7's interpretation of the timing; the "
    "record shows the split forming only after 2400. The second reading for this arm ('the key split only moves': "
    f"{moves['SLOWGATE_KEYHINGE']}, 1 vs REF 4) is not quoted")


def dens(r):
    return sorted((int(k), v["etak_key_by_key"]) for k, v in r["dense"].items())


grid = sorted({t for r in B["REF"].values() for t, _ in dens(r)})
steps = sorted({b - a for a, b in zip(grid, grid[1:])})
new("Descriptively, from the gate's $\\eta^2$ by key at every 25 updates", "S85 measurement grid",
    "Descriptively, from the gate's eta^2 by key at every 25 updates",
    f"dense cells at {grid[:3]}...{[t for t in grid if 1100 <= t <= 1400]}...{grid[-4:]} (spacings {steps})",
    "PARTIAL" if steps != [25] else "MATCH",
    "every 25 updates only to 1200; every 100 from 1200 to 2400, then every 1200 (explore_h_two_b.py l.28-30). The "
    "descriptive section is pre-fixed and the draft marks it 'Descriptively'")


def onset(r):
    d = dens(r)
    first = next((t for t, v in d if v > 0.5), None)
    return first, d[-1][1] > 0.5


on = {a: {s: onset(r) for s, r in sorted(B[a].items())} for a in BA}
post0 = {a: {s: v[0] for s, v in on[a].items() if v[0] not in (None, 0)} for a in BA}
both = sorted(set(post0["REF"]) & set(post0["SLOWGATE"]))
gone = sorted(set(post0["REF"]) - set(post0["SLOWGATE"]))
newsg = sorted(set(post0["SLOWGATE"]) - set(post0["REF"]))
mv_txt = (f"first update > 0.5 after update 0 (seed: update): REF {post0['REF']} -> range "
          f"{min(post0['REF'].values())}-{max(post0['REF'].values())}; SLOWGATE {post0['SLOWGATE']} -> range "
          f"{min(post0['SLOWGATE'].values())}-{max(post0['SLOWGATE'].values())}; seeds with an onset in both {both} "
          f"({[(s, post0['REF'][s], post0['SLOWGATE'][s]) for s in both]}); REF onsets absent under SLOWGATE {gone} "
          f"(discovered in both arms: {all(discovered(B['REF'][s]) and discovered(B['SLOWGATE'][s]) for s in gone)}); new "
          f"under SLOWGATE {newsg} (REF discovered it: {[discovered(B['REF'][s]) for s in newsg]}); REF's KEY failures "
          f"{kf['REF'][2]} with onset at update 0 (above 0.5 at initialisation): "
          f"{[s for s in kf['REF'][2] if on['REF'][s][0] == 0]}")
mv_ok = (min(post0["REF"].values()) == 25 and max(post0["REF"].values()) == 125
         and min(post0["SLOWGATE"].values()) == 275 and max(post0["SLOWGATE"].values()) == 575)
mv_note = ("the two ranges recompute, but only two seeds carry an onset in both arms (2310: 125 -> 275, 2323: 125 -> "
           "575); five of REF's seven post-0 onsets (25-50, on seeds both arms discovered) do not occur under the slow "
           "gate at all, one new onset forms on 2321 (275), a seed REF discovered, and two of REF's four key failures "
           "(2318, 2328) are above 0.5 from initialisation. 'Moves them' reads the ranges as the same splits; report_7 "
           "says only that the splits on 2310 and 2323 'still formed' later and a third formed on 2321")
new("the baseline's real splits form at updates 25--125 and the slow gate moves them to 275--575",
    "S85 onsets: baseline vs slow gate", "the baseline's real splits form at updates 25-125 and the slow gate moves them to "
    "275-575, so the gate's learning rate is not what lets them form", mv_txt, "PARTIAL" if mv_ok else "DISAGREE", mv_note)
new("it moves it from update 25--125 to 275--575 and prevents nothing", "S85 slow gate in 'Predictions that the data "
    "overturned'", "a gate held slow at the start ... would prevent the two-stream key split (it moves it from update "
    "25-125 to 275-575 and prevents nothing, 14/20 against 15/20)",
    f"{bd['SLOWGATE']}/20 vs {bd['REF']}/20 ({bp['SLOWGATE'][0]} vs {bp['SLOWGATE'][1]}); onsets as in the row above",
    "PARTIAL" if mv_ok and bd["SLOWGATE"] == 14 else "DISAGREE", "counts match; 'moves it' as in the row above")
kh = B["KEYHINGE"]
kh25 = {s: dict(dens(r))[25] for s, r in kh.items()}
khw = {s: max(v for t, v in dens(r) if 25 < t <= 2400) for s, r in kh.items()}
over07 = sorted((s, round(v, 3)) for s, v in khw.items() if v >= 0.07)
new("falls to $0.00$--$0.12$ by update 25 and stays below $0.07$ to 2400", "S85 key term: eta^2 by key in the window",
    "under the key term eta^2 by key falls to 0.00-0.12 by update 25 and stays below 0.07 to 2400",
    f"KEYHINGE eta^2 by key at update 25: {min(kh25.values()):.3f}-{max(kh25.values()):.3f} (above 0.12: "
    f"{sorted((s, round(v, 3)) for s, v in kh25.items() if v > 0.125)}); max over updates 50-2400 per run "
    f"{min(khw.values()):.3f}-{max(khw.values()):.3f}; runs reaching >= 0.07 in (25, 2400]: {len(over07)} {over07}",
    ms(max(kh25.values()) < 0.125 and max(khw.values()) < 0.07),
    "copied from report_7 ('What the timing shows'), whose numbers the records do not give: two runs are above 0.12 at "
    "update 25 and seven of twenty reach 0.07-0.165 inside the window. The substance (held far below 0.5 throughout "
    "the window) stands. The records' hinge.fired field reads "
    f"{min(r['hinge']['fired'] for r in kh.values())}-{max(r['hinge']['fired'] for r in kh.values())} per run against "
    "report_7's 'fired on 1-37 batches' (the field's unit was not traced; not in the draft)")
kh36 = {s: dict(dens(r)).get(3600) for s, r in kh.items()}
start = sorted((s, round(v, 2)) for s, v in kh36.items() if v is not None and v >= 0.2)
new("a split can start after the window ($0.21$--$0.64$ at 3600 on some seeds)", "S85 key term: after the window",
    "a split can start after the window (0.21-0.64 at 3600 on some seeds) and every such run recovers, the stream routing "
    "having formed first",
    f"KEYHINGE eta^2 by key at 3600, runs >= 0.2: {start} (range {start[0][1] if start else None}-"
    f"{max(v for _, v in start) if start else None}); all of them DISCOVERED: {all(discovered(kh[s]) for s, _ in start)}",
    ms(start and min(v for _, v in start) == 0.21 and max(v for _, v in start) == 0.64 and all(discovered(kh[s]) for s, _ in start)),
    "the range's top is 0.88 (seed 2321), not 0.64 (report_7 gives '2317: 0.64; 2321 > 0.5', and the draft kept only the "
    "0.64); 'every such run recovers' holds (KEYHINGE 20/20); 'the stream routing having formed first' is the session's "
    "interpretation (no eta^2-by-stream check here)")
new("the price is a later transition (median 4800 against 2400)", "S85 key term: transition",
    "the price is a later transition (median 4800 against 2400)",
    f"median transition of bound runs: {', '.join(f'{a} {btr[a]:.0f}' for a in BA)}",
    ms(btr["KEYHINGE"] == 4800 and btr["REF"] == 2400), "")
d0 = {s: dict(dens(r))[0] for s, r in B["REF"].items()}
same0 = all(len({round(dict(dens(B[a][s]))[0], 6) for a in BA}) == 1 for s in B["REF"])
init = sorted((s, round(v, 2)) for s, v in d0.items() if v > 0.5)
shed = {s: dict(dens(B["REF"][s]))[600] for s, _ in init if discovered(B["REF"][s])}
new("organised by key on six of the twenty seeds", "S85 the initial gate's key structure",
    "At initialisation the near-uniform gate's small variations are organised by key on six of the twenty seeds (eta^2 up "
    "to 0.84 at update 0), ..., which the baseline's discovering runs shed by update 600",
    f"eta^2 by key at update 0 (identical in the four arms: {same0}) > 0.5 on {len(init)} seeds {init}; next highest "
    f"{sorted(d0.values())[-len(init) - 1]:.3f}; REF discovering among them at update 600: "
    f"{ {s: round(v, 3) for s, v in shed.items()} }",
    ms(len(init) == 6 and max(v for _, v in init) == 0.84 and all(v < 0.5 for v in shed.values())),
    "five seeds, not six (2313, 2318, 2320, 2326, 2328; the sixth highest is 2319 at 0.476). report_7 says '> 0.5 on 6 "
    "seeds in all' while its own per-seed table shows five update-0 onsets in every arm. 'Up to 0.84' and 'shed by 600' "
    "(2313, 2320) hold")
REP7 = " ".join(git("show", R7 + ":explore_out/H/report_7.md").split())
TAKE = ("A candidate for a pre-registered test: WIN3_SLOW + the key term (S52's W2_KEYHINGE) against WIN3_SLOW, on "
        "disjoint seeds, under v2.")
new("Session~H proposes the pre-registered test, the window recipe with and without the key term",
    "S85: H's proposed test (and next steps, tex2:470)", "Session H proposes the pre-registered test, the window recipe with "
    "and without the key term at two streams on disjoint seeds under the key-first classifier",
    "report_7.md, Take-away: 'A candidate for a pre-registered test: WIN3_SLOW + the key term (S52's W2_KEYHINGE) against "
    f"WIN3_SLOW, on disjoint seeds, under v2.': {TAKE in REP7}",
    ms(TAKE in REP7), "")
assert where2("Session~H's proposed test of the key term at two streams") == [470]
c85 = {}
for a in BA:
    for s, r in B[a].items():
        if not discovered(r):
            c85[f"{a}|{s}"] = (fail_class(r), fail_class_v2(r["end"]), r["end"]["eta_key_by_key"], r["end"]["margin"])
d85 = sorted(k for k, v in c85.items() if v[0] != v[1])
p85 = sorted(k for k, v in c85.items() if v[2] >= ETA_SPLIT and v[3] >= MARGIN_PARTIAL)
new("and one of S85's; no reading changes", "S85 v1/v2 differences; readings under v1",
    "... and one of S85's; no reading changes",
    f"{len(c85)} unbound S85 runs; v1 != v2 on {d85} ({[c85[k][:2] for k in d85]}, eta^2 by key {[round(c85[k][2], 7) for k in d85]}, "
    f"margin {[round(c85[k][3], 3) for k in d85]}); = predicted: {d85 == p85}; 'only moves' under v1 instead of v2: "
    f"{moves1} (v2: {moves})", ms(len(d85) == 1 and d85 == p85 and moves1 == moves),
    "S85's readings are fixed on v2 (and DISCOVERED), so v1 could not change them; recomputed anyway")

# abstract, index, synthesis, limits, conclusion, next steps — S85/S84 uses
new("prevented by the hinge's key term on a screen ($20/20$ against $15/20$)", "abstract: S85",
    "Two-stream key splits resist repair and are prevented by the hinge's key term on a screen (20/20 against 15/20)",
    f"S78 R4 'neither device' (best {vr[best]}); S85 KEYHINGE {bd['KEYHINGE']}/20 vs REF {bd['REF']}/20, 'prevention works'",
    ms(neither and bd["KEYHINGE"] == 20 and bd["REF"] == 15 and prevent["KEYHINGE"]), "marked 'on a screen'")
new("that the hinge's key term prevents on a screen (S85, $20/20$ against $15/20$)", "synthesis: two ways of losing (S85)",
    "a key split that the copy cannot undo, because, on our reading, there is no idle channel, and that the hinge's key "
    "term prevents on a screen (S85, 20/20 against 15/20)", f"KEYHINGE {bd['KEYHINGE']}/20 vs REF {bd['REF']}/20",
    ms(bd["KEYHINGE"] == 20 and bd["REF"] == 15), "first pass: tex:312's 'because' was unmarked; now 'on our reading'")
new("the flat-start guard aims the splits without changing the count", "index: S84 'the guard aims the split'",
    "the flat-start guard aims the splits without changing the count",
    f"pre-fixed 'the guard aims the split' (labelled ok >= 95% AND GUARD >= BASE - 1 pair): labelled ok GUARD "
    f"{sum(label_ok(c) for c in gf['GUARD'])}/{len(gf['GUARD'])}, pairs {gp['GUARD']} -> {'applies' if aims else 'does not apply'}; "
    f"but BASE's splits, 7 of them at probe < 0.2, were labelled ok {gl_okb}/{len(gf['BASE'])}",
    "PARTIAL" if aims and gl_okb == len(gf["BASE"]) else ms(aims),
    "the reading applies by its rule, but its 'aim' criterion cannot fail at the flat start: every BASE split, including "
    "the seven at chance accuracy, was labelled ok (report_6: 'The guard's 95% criterion therefore cannot fail on aim, "
    "only on count'). What the guard changed is the accuracy at which splits fire (tex2:203), which 'aims' does not say")
new("two-stream key splits resist the copy and are prevented by the key hinge, not by a slow gate", "index: S78/S85",
    "two-stream key splits resist the copy and are prevented by the key hinge, not by a slow gate",
    f"S78 'neither device'; S85 KEYHINGE 'prevention works' ({bp['KEYHINGE'][0]} vs {bp['KEYHINGE'][1]}), SLOWGATE "
    f"'prevention not shown' ({bp['SLOWGATE'][0]} vs {bp['SLOWGATE'][1]})",
    ms(neither and prevent["KEYHINGE"] and not prevent["SLOWGATE"]), "the index row is the sessions' (exploratory) column")
new("the hinge's key term, which prevents the key split at two streams and for the register at a distance",
    "limits: 'the key term prevents the key split at two streams'",
    "the hinge's key term, which prevents the key split at two streams and for the register at a distance, reads key "
    "positions and key identities", f"S85 KEYHINGE {bd['KEYHINGE']}/20 vs {bd['REF']}/20 (5 vs 0, p 0.031), one screen of 20 seeds",
    "PARTIAL", "stated without 'on a screen', unlike tex2:36, 316 and 440; the reading 'prevention works' is an "
    "exploratory one-sided p = 0.031 on 20 seeds. The 'reads key identities' half is right (explore_h_two_b.py l.17)")
new("the prevention that works on a screen (the key term, $20/20$) reads key identities", "limits: two streams",
    "every arm is at 65-69/80, the one repair device that works at eight streams cannot work here, and the prevention that "
    "works on a screen (the key term, 20/20) reads key identities",
    f"S85 KEYHINGE {bd['KEYHINGE']}/20; S78: SPLIT2 {vr['SPLIT2']}, SPLIT2_NOISE {vr['SPLIT2_NOISE']} vs REF -> 'neither "
    "device'", "PARTIAL",
    "the S85 clause matches and is marked; 'cannot work here' (unchanged from tex:434, not checked in the first pass) "
    "states as fact what tex2:209 now marks as 'our reading': S78 measured 'neither device' (1 vs 0, 2 vs 0) on 20 seeds")
new("which also prevents the two-stream key split that nothing repairs", "conclusion: S85",
    "is found on every seed when a hinge keeps the gate off key identity during the slow phase, which also prevents the "
    "two-stream key split that nothing repairs",
    f"S85 KEYHINGE {bd['KEYHINGE']}/20 (exploratory); S78 R4 'neither device' over two forms of the split (exact copy "
    f"{vr['SPLIT2']}, noisy copy with reset {vr['SPLIT2_NOISE']}) on 20 seeds", "PARTIAL",
    "no 'on a screen' in the conclusion for either the register's 20/20 (G, S82) or S85's; 'nothing repairs' generalises "
    "S78's two forms of one device to every repair. tex2:228 has the same 'the split that nothing repairs'")
new("prevents the split that nothing repairs", "S85 paragraph: 'nothing repairs'",
    "the key term, which reads key positions and key identities during the slow phase, prevents the split that nothing "
    "repairs", f"S78 R4 'neither device' (exact copy {vr['SPLIT2']}, noisy copy with reset {vr['SPLIT2_NOISE']}, K4 "
    f"{vr['K4']}); S85 KEYHINGE 5 vs 0", "PARTIAL",
    "in the paragraph that frames S85 as a screen, but 'nothing repairs' is wider than S78's 'neither device'")
new("H's S61--S85", "Revision 9 paragraph: H's screens", "the exploratory sessions' second and third rounds (... H's S61-S85 ...)",
    "H's reports 3-7 cover S61, S62, S63, S78, S79, S84, S85 (report commits below)", "MATCH", "")

# ══ provenance — tex:500 ═════════════════════════════════════════════════════════════════════════════════════════


def touches(path):
    return [x.split(" ", 1) for x in git("log", "--format=%h %s", BR, "--", path).strip().split("\n")]


rt = {n: touches(f"explore_out/H/report_{n}.md") for n in (1, 2, 3, 4, 5, 6, 7)}
rc = {n: rt[n][0][0] for n in rt}      # the last commit that wrote the report
nr = {k: len(ST[k]["runs"]) for k in ST}
pair = [nr["S58"] + nr["S58b"] + nr["S60"], nr["S61"] + nr["S62"] + nr["S63"], nr["S78"], nr["S79"], nr["S84"], nr["S85"]]
add(500, "report commits", "report 3 7e9197f; report 4 72f0bbc; report 5 df57087; report 6 37c1fbf; report 7 1e09ecd",
    f"last commit writing report_N.md: {rc}; commits touching report_6.md: {[h for h, _ in rt[6]]}, report_7.md: "
    f"{[h for h, _ in rt[7]]}",
    ms(rc[3] == R3 and rc[4] == R4 and rc[5] == R5 and rc[6] == R6 and rc[7] == R7),
    f"reports 1-2 written at {rc[1]}, {rc[2]} (not named in the draft); report 6 '{rt[6][0][1][:45]}...', report 7 "
    f"'{rt[7][0][1][:45]}...'", frag="report 3, \\texttt{7e9197f}", frag2="S85 prevention at two streams (report 7, \\texttt{1e09ecd})")
add(500, "runs per report group", "318; 300; 82; 62; 52; 82",
    f"S58+S58b+S60 {nr['S58']}+{nr['S58b']}+{nr['S60']} = {pair[0]}; S61+S62+S63 {nr['S61']}+{nr['S62']}+{nr['S63']} = "
    f"{pair[1]}; S78 {pair[2]}; S79 {pair[3]}; S84 {pair[4]}; S85 {pair[5]}", ms(pair == [318, 300, 82, 62, 52, 82]),
    "S62's 216 are reruns of S58/S58b runs (counted again)", frag="318; 300; 82; 62", frag2="318; 300; 82; 62; 52; 82")
add(500, "branch commit", "1e09ecd", f"branch head {HEAD}; = report 7's commit {R7}: {HEAD == R7}",
    ms(HEAD == R7), "first pass: df57087 (MATCH then); the second draft pins the head, which is report 7's commit",
    frag="& \\texttt{df57087} &", frag2="(report 7, \\texttt{1e09ecd}) & \\texttt{1e09ecd} &")

# ══ tex:60 index row ════════════════════════════════════════════════════════════════════════════════════════════
add(60, "index: 'nothing replaces the split'", "neither device carried to eight streams replaces the split",
    f"S58 R1 does not apply; S58b: RESET and GUMBEL_W do not carry ({sr8[0]} vs {sr8[1]}, {sg8[0]} vs {sg8[1]}); SWITCH and "
    f"GUMBEL_RW never run at eight streams", ms(sr8 == (8, 0) and sg8 == (10, 0)),
    "first pass PARTIAL ('nothing replaces the split'); now restricted to the two carried devices",
    frag="nothing replaces the split", frag2="neither device carried to eight streams replaces the split")
add(60, "index: 'the copy alone suffices'", "the copy alone suffices", f"S61 R1 {'holds' if r1 else 'does not apply'} "
    f"(on COPY, the noisy copy)", "MATCH", "pre-fixed reading name; see tex:195 row for which arm it reads",
    frag="the copy alone suffices")
add(60, "index: 'the cap matters'", "the cap matters", f"S79 CAP1 '{lab['CAP1'][0]}' (4 vs 0, one-sided p "
    f"{p3(mcnemar1(4, 0))}); CAP6 '{lab['CAP6'][0]}'", "MATCH",
    "report_5.md l.44's heading for the pre-fixed 'sensitive' (CAP1 only); raising the cap to six was 'robust'",
    frag="the cap matters")
add(60, "index: 'two-stream key splits resist the copy'", "two-stream key splits resist the copy",
    f"S78 R4 'neither device'; KEY split back after {len(fk)} of {len(fk)} firings on the KEY-ending seeds", "MATCH",
    "the reading is 'neither device'; 'resist the copy' is the firing-record description (the S85 half of this cell is "
    "a separate row)", frag="two-stream key splits resist")

# ══ Readings vs docstring (first commit of each screen file; first run record) ══════════════════════════════════


def docstring(ref, path):
    t = git("show", f"{ref}:{path}").split("\n")
    i = t.index('"""')
    j = next(k for k in range(i + 1, len(t)) if t[k].startswith('"""'))
    return t[i + 1:j]


def readings_block(lines):
    """The docstring's ARMS ... READING(S) ... block up to CHECKS / DIAGNOSTICS / DESCRIPTIVE (exclusive)."""
    out, on = [], False
    for l in lines:
        if l.startswith(("ARMS", "ARM SELECTION", "VALIDITY", "READING")):
            on = True
        elif l.startswith(("CHECKS", "DIAGNOSTICS", "DESCRIPTIVE", "OUTCOME")):
            on = False
        if on:
            out.append(l)
    return out if any(l.startswith("READING") for l in out) else []


def first_run(path):
    for line in git("log", "--reverse", "--format=%h %cI", BR, "--", path).strip().split("\n"):
        h, t = line.split()
        try:
            n = len(json.loads(git("show", f"{h}:{path}"))["runs"])
        except Exception:
            continue
        if n:
            return h, t, n
    return None, None, 0


SCR = [("S58", "explore_h_collapse.py", "collapse"), ("S58b", "explore_h_eight.py", "eight"),
       ("S60", "explore_h_stability.py", "stability"), ("S61", "explore_h_copy2x2.py", "copy2x2"),
       ("S62", "explore_h_query_gate.py", "query_gate"), ("S63", "explore_h_hardness.py", "hardness"),
       ("S78", "explore_h_two.py", "two"), ("S79", "explore_h_constants.py", "constants"),
       ("S84", "explore_h_guard.py", "guard"), ("S85", "explore_h_two_b.py", "two_b")]
TIMING = []
for sc, f, st in SCR:
    log = git("log", "--reverse", "--format=%h %cI", BR, "--", f).strip().split("\n")
    h0, t0 = log[0].split()
    d0, dh = docstring(h0, f), docstring(BR, f)
    ch = [l for l in difflib.unified_diff(d0, dh, lineterm="", n=0) if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    rh, rt, rn = first_run(P(st)) if exists(BR, P(st)) else (None, None, 0)
    rb0, rbh = readings_block(d0), readings_block(dh)
    TIMING.append((sc, f, f"{h0} {t0}", f"{len(log)} ({', '.join(x.split()[0] for x in log)})",
                   f"{len(ch)} changed docstring lines" + (": " + " / ".join(c[:90] for c in ch[:3]) if ch else "")
                   + ("; READING/VALIDITY/ARMS block identical" if rb0 == rbh and rb0 else
                      "; no READING block (descriptive)" if not rb0 else "; READING block CHANGED"),
                   f"{rh} {rt} ({rn} runs)" if rh else "no run record at the head",
                   ("yes" if rt and t0 < rt else "n/a") if rh else "n/a"))

READ += [
    ("S58", "bbfea60 l.48-58: R1 'noise replaces the split', R2 'the split is the reset' (+ qualifier 'uninformative'), "
     "R3 'the balance loss helps'", "tex2:207 'two of its three readings did not apply and the third ... held by its rule "
     "(0 vs 0) and was printed uninformative'", "MATCH", f"first pass PARTIAL; {s58_read}"),
    ("S58b", "69d01dc l.12-15 selection rule (devices only); l.21-24 'X carries to eight streams' if (SPLIT only - X only) "
     "<= 1", "tex2:207 'the two best-scoring devices ... (it ranked the four devices only ...)'; tex2:189 8 vs 0",
     "MATCH", "first pass PARTIAL; the devices-only rule is now stated. 'Does not carry' itself is still not quoted by name"),
    ("S60", "705000d l.30-40: D1 'no change at n = 10' unless >= 3 vs 0; D2 'delays saturation' if >= 0.2 lower; D3",
     "tex2:207 'read \"no change at n = 10\" by its rule (7/10 against 8/10)'", "MATCH", "first pass PARTIAL"),
    ("S61", "f630ef2 l.28-30: R1 'the copy alone suffices' (COPY vs SPLIT), R2 'the noise matters' (COPY vs COPY_NONOISE "
     ">= 4 vs 0), R3 'the reset adds nothing' (|SPLIT only - COPY only| <= 1)", "tex2:195 caption; tex2:60; tex2:353",
     "MATCH", "first pass PARTIAL (R1/R3 attributed to the exact copy); now on 'the copy with noise'"),
    ("S62", "f630ef2: descriptive, no reading", "tex2:207 'descriptive: 32-37 of the 38-40 bound runs per arm, 12/12'",
     "MATCH", "now marked descriptive and counted over bound runs"),
    ("S63", "f630ef2 l.23-26: 'hardness helps' if X only >= 4 and REF only = 0, or earlier on >= 8 with no loss; else "
     "'does not help'", "tex2:207 'both read \"does not help\"'", "MATCH",
     "first pass PARTIAL ('neutral', 'destroyed routing'); now the pre-fixed label for both arms"),
    ("S78", "fdf7499 l.40-46: R1-R4; R4 'neither device'", "tex2:226 'the pre-fixed reading is \"neither device\"'", "MATCH",
     "unchanged"),
    ("S79", "fdf7499 l.26-32: robust / sensitive / inconclusive; CAP6 'the extra splits are harmless'", "tex2:203", "MATCH",
     "unchanged; the recommendation is now marked as not a reading"),
    ("S84", "834581a l.33-40: 'the guard aims the split' (labelled ok >= 95% AND GUARD >= BASE - 1 pair), 'faster without "
     "the cost' (GUARD_INT1200 >= BASE - 1 pair AND median transition of BOUND ROUTED runs >= 2400 earlier), 'the "
     "evaluation is the better probe' (EVAL >= BASE - 1 pair), 'costs binding' (BASE only >= 4 and arm only = 0)",
     "tex2:203 ('faster without the cost' not met: 1200 earlier; 'the evaluation is the better probe' applies by its rule); "
     "tex2:60 ('the flat-start guard aims the splits')", "PARTIAL",
     f"every quoted outcome recomputes (aims: {aims}; faster: {fast}; eval: {evp}); 'aims' in the index is a criterion "
     "that cannot fail at the flat start (BASE's chance-level splits all labelled ok); 'costs binding' (no arm) and the "
     "EVAL_GUARD_INT1200 arm are not stated"),
    ("S85", "834581a l.23-27: 'prevention works' (one-sided McNemar p < 0.05 on DISCOVERED), 'the key split only moves' "
     "(KEY failures under v2 >= REF's - 1)", "tex2:228 (KEYHINGE 'prevention works'; SLOWGATE 'prevention not shown; the "
     "key split only moves'; both together 'not shown')", "MATCH",
     f"prevention {prevent}; only moves (v2) {moves}; the combined arm's 'only moves' (False) is not quoted, which is harmless"),
]

DESC += [
    ("tex2:201", "S61", "That two identical rows, their symmetry broken only by their Adam moments and the data, separate merged "
     "streams as well as two noisy ones is what 0 vs 0 on ten seeds is consistent with, not a shown equality.",
     "marked", "first pass: unmarked equality claim (tex:201); RESOLVED"),
    ("tex2:201", "S61", "The exact copy is therefore the split's form in every later screen and in the next main-line test.",
     "a design choice", "true of S78 (SPLIT2), S79 and S84 (BASE firings: row_dist_after 0, state_zeroed empty)"),
    ("tex2:191", "S61", "copy only vs reset only 5 vs 0, p = 0.031 (ours, not pre-fixed)", "marked", "RESOLVED"),
    ("tex2:203", "S79", "four of CAP1's five failures ... (per-stream accuracy is recorded only at 4800 and 9600; the seed that "
     "fired at 12,000 had two streams above 0.99 at 9600) ...; the plateau rule fires during the flat start ...",
     "marked ('Descriptively, and not pre-registered')", "the false 'before any stream had bound' is gone"),
    ("tex2:203", "S79", "Our recommendation from this, not a reading: a replication should state the cap and the check "
     "schedule ...", "marked", "RESOLVED; the confound (interval, first check and gap moved together) is stated"),
    ("tex2:203", "S84", "it locks out the one seed whose probe never reaches 0.5 ...; so the probe's noise decided whether the "
     "first check fired; These seeds hardly tested the flat start, since the base kept enough splits for its merges after "
     "spending some; the guard changes when the splits fire and, on these seeds, not whether the run binds.",
     "not marked", "report_6's 'What happened' descriptions, written as findings; the counts and readings beside them are "
     "pre-fixed and the per-split accuracies are pre-fixed DESCRIPTIVE (explore_h_guard.py l.41); the interpretive "
     "clauses ('the probe's noise decided', 'kept enough splits for its merges') carry no marker. NEW"),
    ("tex2:207", "S63", "that the writes' stream labelling therefore never forms is the session's description, and four of the "
     "ten runs did form a partial one", "marked", "RESOLVED"),
    ("tex2:207", "S58", "it changed the routing's shape, not its count", "not marked",
     "unchanged from tex:207 (first pass: unmarked); the count part is 36 vs NONE 38 (1 vs 3 discordant), the shape part "
     "34/36. NOT RESOLVED"),
    ("tex2:209", "S78", "That the copy works at eight streams because the channel it copies onto is empty, and has nothing to "
     "offer where every channel is busy, is our reading of these screens.", "marked", "RESOLVED. The next sentence "
     "('the trigger should not fire when its target already carries key mass') is a recommendation, unmarked"),
    ("tex2:226", "S78", "Our reading: at eight streams the copy creates a near-tie between a crowded channel and an empty one; "
     "at k = S it only resets the gate to where the gradient takes it again.", "marked", "RESOLVED (the eight-stream half "
     "is still S24's measurement, not an H record)"),
    ("tex2:226", "S78", "Session H's reading of the mechanism (Adam moments) ... was not measured and is not a reading",
     "marked", "unchanged"),
    ("tex2:228", "S85", "Descriptively, from the gate's eta^2 by key ...: onsets 25-125 -> 275-575, 'so the gate's learning "
     "rate is not what lets them form'; 0.00-0.12 / below 0.07; 0.21-0.64 at 3600; 'the stream routing having formed "
     "first'; six seeds at initialisation", "marked ('Descriptively')", "marked, but three of its numbers disagree with "
     "the records and one generalises (see the S85 rows). NEW"),
    ("tex2:228", "S85", "the one failure a seed on which the slow gate held the gate back until the hinge's window had passed "
     "and the split then formed", "not marked", "report_7's interpretation of 2317's timing, inside the readings sentence; "
     "the record shows only that eta^2 by key stayed <= 0.2 to 2400 and rose to 0.77 by 4800. NEW"),
    ("tex2:316", "S61/S78/S85", "a key split that the copy cannot undo, because, on our reading, there is no idle channel, "
     "and that the hinge's key term prevents on a screen", "marked", "RESOLVED (first pass tex:312)"),
    ("tex2:342", "S78/S79", "the check interval and the first eligible check act through the flat start ... Each is marked "
     "where it appears", "partly", "the S79 description is marked at tex2:203 and 'noise helps a little' is no longer "
     "listed; but the S84 interpretations at tex2:203 and S58's 'shape, not count' at tex2:207 are unmarked"),
    ("tex2:440", "S78", "the one repair device that works at eight streams cannot work here", "not marked",
     "stated as fact where tex2:209 marks the same idea as 'our reading'. NEW (unchanged from tex:434; missed in the first "
     "pass)"),
]


# ══ THIRD DRAFT (a5f67a2): re-anchor every row, re-check every changed number ═══════════════════════════════════════
# Each row above carries the fragment that located it in the second draft (f67bc07). Here it is located in the third
# draft by text search. Where the sentence was rewritten, OV gives the new fragment (which must be found in TEX3), the new
# wording, the recomputation against the same record and the status; a row whose fragment is gone and that has no OV
# entry is an error (so no rewritten sentence passes silently); OV status 'REMOVED' marks a sentence that is gone.
V2 = {}
for _l in open(os.path.join(HERE, "check_H_v2.md")):
    _m = re.match(r"^\| (tex2:[^|]*) \| ([^|]*) \| (.*?) \| .* \| \*\*([A-Z -]+)\*\* \| ([^|]*) \|", _l)
    if _m:
        V2[_m.group(3)] = _m.group(4)
H3 = HEAD    # branch head now
R7C = rc[7]  # last commit writing report_7.md now
REP7C = " ".join(git("show", f"{BR}:explore_out/H/report_7.md").split())
blob_same = {k: blob(ref, P(n)) == blob(BR, P(n)) for k, (ref, n) in STORES.items()}
h_after = [x.split(" ", 1) for x in git("log", "--format=%h %cI %s", f"{R7}..{BR}").strip().split("\n") if x]


def where3(frag):
    return [i + 1 for i, l in enumerate(TEX3) if frag in l]


OV = {}
NEW3 = []    # rows for sentences that only the third draft has (not a rewrite of a v2 row)

# --- S85 SLOWGATE: 'four key failures of its own, three on the baseline's seeds'
shared_k = sorted(set(kf["SLOWGATE"][2]) & set(kf["REF"][2]))
OV["S85 SLOWGATE and its readings"] = dict(
    frag3="the slow gate discovers $14/20$ (0 vs 1) with four key failures of its own, three on the baseline's seeds",
    draft="the slow gate discovers 14/20 (0 vs 1) with four key failures of its own, three on the baseline's seeds, reading "
          "'prevention not shown; the key split only moves'",
    rec=f"DISCOVERED SLOWGATE {bd['SLOWGATE']}/20 ({bp['SLOWGATE'][0]} vs {bp['SLOWGATE'][1]}) -> "
        f"{'works' if prevent['SLOWGATE'] else 'prevention not shown'}; only moves {moves['SLOWGATE']}; KEY seeds (v2, the "
        f"paragraph's classifier) SLOWGATE {kf['SLOWGATE'][2]}, REF {kf['REF'][2]}, shared {shared_k}; under v1 SLOWGATE "
        f"{kf['SLOWGATE'][1]}",
    status=ms(len(kf["SLOWGATE"][2]) == 4 and len(shared_k) == 3 and bd["SLOWGATE"] == 14 and bp["SLOWGATE"] == (0, 1)
              and not prevent["SLOWGATE"] and moves["SLOWGATE"]),
    note="v2 PARTIAL ('the same four'): now four of its own, three shared (2310, 2323, 2328). The fourth (2321) is a seed "
         "REF discovered; REF's 2318 ends STREAM-PARTIAL under the slow gate; SLOWGATE's 2323 is KEY only under v2 "
         "(eta^2 by key 0.5000005), which the paragraph's 'under the key-first classifier' covers")
# --- S85 grid
seg1 = [t for t in grid if t <= 1200]
seg2 = [t for t in grid if 1200 <= t <= 2400]
grid_ok = (seg1 == list(range(0, 1225, 25)) and seg2 == list(range(1200, 2500, 100)))
OV["S85 measurement grid"] = dict(
    frag3="Descriptively, from the gate's $\\eta^2$ by key every 25 updates to 1200 and every 100 to 2400",
    draft="Descriptively, from the gate's eta^2 by key every 25 updates to 1200 and every 100 to 2400",
    rec=f"dense cells 0-1200: {len(seg1)} at spacing {sorted({b - a for a, b in zip(seg1, seg1[1:])})}; 1200-2400: "
        f"{len(seg2)} at spacing {sorted({b - a for a, b in zip(seg2, seg2[1:])})}; after 2400 {grid[grid.index(2400) + 1:][:3]}...",
    status=ms(grid_ok), note="v2 PARTIAL ('at every 25 updates'); now matches explore_h_two_b.py l.28-30 and the records")
# --- S85 onsets
on_ok = (mv_ok and len(post0["REF"]) == 7 and len(post0["SLOWGATE"]) == 3 and both == [2310, 2323]
         and all(s in kf["REF"][2] for s in both) and newsg == [2321] and discovered(B["REF"][2321]))
OV["S85 onsets: baseline vs slow gate"] = dict(
    frag3="the baseline's onsets after update 0 lie at updates 25--125 (seven seeds) and the slow gate's at 275--575 "
          "(three seeds, two of them seeds the baseline also split and one it discovered), so, on our reading",
    draft="the baseline's onsets after update 0 lie at updates 25-125 (seven seeds) and the slow gate's at 275-575 (three "
          "seeds, two of them seeds the baseline also split and one it discovered), so, on our reading, the gate's "
          "learning rate is not what lets them form",
    rec=mv_txt + f"; REF's KEY failures among the two shared seeds: {[s for s in both if s in kf['REF'][2]]}",
    status=ms(on_ok),
    note="v2 PARTIAL ('moves them'); now stated by seed count with the overlap, and the inference marked 'on our "
         "reading'. Not stated, and not needed: REF's other five onsets (25-50) do not recur under the slow gate on seeds "
         "both arms discovered")
# --- S85 slow gate in 'Predictions that the data overturned' (tex3:384)
lat = min(post0["SLOWGATE"].values()) > max(post0["REF"].values())
OV["S85 slow gate in 'Predictions that the data overturned'"] = dict(
    frag3="(the onsets move later and the splits still form; $14/20$ against $15/20$)",
    draft="a gate held slow at the start ... would prevent the two-stream key split (the onsets move later and the splits "
          "still form; 14/20 against 15/20)",
    rec=f"{bd['SLOWGATE']}/20 vs {bd['REF']}/20 ({bp['SLOWGATE'][0]} vs {bp['SLOWGATE'][1]}); every SLOWGATE onset "
        f"({sorted(post0['SLOWGATE'].values())}) later than every REF onset ({sorted(post0['REF'].values())}): {lat}; seeds "
        f"with an onset in both arms {both}; SLOWGATE KEY failures (v2) {kf['SLOWGATE'][2]}",
    status="PARTIAL" if (lat and bd["SLOWGATE"] == 14 and len(both) == 2) else "DISAGREE",
    note="v2 PARTIAL. As a statement about the two sets of onsets it holds (all three of the slow gate's onsets come after "
         "all seven of the baseline's) and the splits do still form (four KEY failures). 'Move' still reads as the same "
         "splits shifted, which holds on two seeds (2310: 125 -> 275, 2323: 125 -> 575); five of the baseline's seven "
         "onsets have no counterpart. tex3:228 gives the overlap; this summary does not. Minor")
# --- S85 key term in the window
kh_win = {s: max(v for t, v in dens(r) if 25 <= t <= 2400) for s, r in kh.items()}
kh_fired = sorted(r["hinge"]["fired"] for r in kh.values())
kh_tau = {r["hinge"]["tau"] for r in kh.values()}
kh_n = {r["hinge"]["window"] for r in kh.values()}
OV["S85 key term: eta^2 by key in the window"] = dict(
    frag3="stays below the term's threshold of $0.2$ at every measurement from update 25 to 2400 (maximum $0.195$ over "
          "the 20 runs; the term fired on 1--11 batches per run)",
    draft="under the key term eta^2 by key stays below the term's threshold of 0.2 at every measurement from update 25 to "
          "2400 (maximum 0.195 over the 20 runs; the term fired on 1-11 batches per run)",
    rec=f"KEYHINGE runs {len(kh)}; max eta^2 by key over dense cells 25-2400 per run {min(kh_win.values()):.3f}-"
        f"{max(kh_win.values()):.3f} (the maximum on {max(kh_win, key=kh_win.get)}, at update 25: "
        f"{dict(dens(kh[max(kh_win, key=kh_win.get)]))[25]:.3f}); hinge.tau recorded {kh_tau}, window {kh_n}; hinge.fired per "
        f"run {kh_fired[0]}-{kh_fired[-1]} (= training forward passes in updates 1-2400 whose batch eta2_key exceeded tau: "
        f"attach_key_hinge, explore_keysplits_child.py l.174-200 at {R7})",
    status=ms(max(kh_win.values()) < 0.2 and round(max(kh_win.values()), 3) == 0.195 and kh_tau == {0.2}
              and kh_fired[0] == 1 and kh_fired[-1] == 11),
    note="v2 DISAGREE (0.00-0.12 / below 0.07). The new numbers are the records' (and report_7 as corrected at "
         f"{R7C}). The threshold is the hinge's tau = 0.2 (explore_h_two_b.py l.17: relu(eta2_key - 0.2)). 'Fired' counts "
         "training batches on which the term's own batch statistic exceeded 0.2, so it is consistent with every probe "
         "measurement staying below 0.2. 'Stays below' holds from update 25; at update 0 five seeds sit above 0.5 (next "
         "row but one)")
# --- S85 after the window
start7 = [s for s, _ in start]
OV["S85 key term: after the window"] = dict(
    frag3="a split can start after the window ($0.21$--$0.88$ at 3600 on seven seeds) and every such run recovers",
    draft="a split can start after the window (0.21-0.88 at 3600 on seven seeds) and every such run recovers, the stream "
          "routing having formed first",
    rec=f"KEYHINGE eta^2 by key at 3600 >= 0.2: {start} -> {len(start)} seeds, range {min(v for _, v in start)}-"
        f"{max(v for _, v in start)}; next below: {max((round(v, 3) for v in kh36.values() if v < 0.2), default=None)}; all "
        f"DISCOVERED: {all(discovered(kh[s]) for s in start7)}",
    status=ms(len(start) == 7 and min(v for _, v in start) == 0.21 and max(v for _, v in start) == 0.88
              and all(discovered(kh[s]) for s in start7)),
    note="v2 DISAGREE (0.21-0.64, 'some seeds'); now the records' range and count. The clause 'the stream routing having "
         "formed first' is checked in its own row below")
# the stream routing 'having formed first' (computed here for the first time)
SRF = {}
for s in start7:
    d = {int(t): v for t, v in kh[s]["dense"].items()}
    pre = d[2400]
    first_m = next((t for t in sorted(d) if d[t]["margin"] >= MARGIN_PARTIAL), None)
    SRF[s] = dict(m2400=pre["margin"], map2400=pre["ch_map"], ks2400=pre["etak_key_by_stream"],
                  vs2400=pre["etak_val_by_stream"], ks3600=d[3600]["etak_key_by_stream"], first_margin=first_m,
                  acc2400=next(c[1] for c in kh[s]["curve"] if c[0] == 2400))
n_map = sum(len(set(v["map2400"])) == 2 for v in SRF.values())
n_ks = sum(v["ks2400"] >= 0.5 for v in SRF.values())
NEW3.append(("the stream routing having formed first; the price is a later transition",
             "S85 key term: 'the stream routing having formed first'",
             "a split can start after the window ... and every such run recovers, the stream routing having formed first",
             "the seven KEYHINGE runs with eta^2 by key >= 0.2 at 3600, at the last dense cell before it (2400): routing "
             f"margin {sorted(round(v['m2400'], 5) for v in SRF.values())}; channel map one-to-one on {n_map} of 7 "
             f"({ {s: v['map2400'] for s, v in SRF.items()} }); eta^2 by stream at value positions "
             f"{min(v['vs2400'] for v in SRF.values()):.3f}-{max(v['vs2400'] for v in SRF.values()):.3f}; at key positions "
             f"{min(v['ks2400'] for v in SRF.values()):.2f}-{max(v['ks2400'] for v in SRF.values()):.2f} (>= 0.5 on {n_ks}), "
             f"falling at 3600 to {min(v['ks3600'] for v in SRF.values()):.3f}-{max(v['ks3600'] for v in SRF.values()):.2f}; "
             f"accuracy at 2400 {min(v['acc2400'] for v in SRF.values()):.2f}-{max(v['acc2400'] for v in SRF.values()):.2f}; "
             f"first dense cell with margin >= 0.25: { {s: v['first_margin'] for s, v in SRF.items()} }",
             "PARTIAL" if (all(v["m2400"] < MARGIN_PARTIAL for v in SRF.values()) and n_ks >= 1) else
             ("DISAGREE" if all(v["m2400"] < MARGIN_PARTIAL for v in SRF.values()) else "MATCH"),
             "not checked in either earlier pass (it needed eta^2 by stream; the dense cells carry it). Unchanged from "
             "tex2:228, inside the 'Descriptively' sentence, from report_7 ('the stream routing had already formed'). On the "
             "dense records no stream routing had formed before the key rise at 3600: the routing margin is 0.00 at 2400 "
             "on all seven, the map is one-to-one on two, accuracy sits at 0.24-0.27, and the routing (margin >= 0.25) "
             "first appears at 4800 or 6000, after the key rise. What had formed is partial stream structure at key "
             "positions (eta^2 by stream >= 0.5 on four of seven at 2400), which itself drops at 3600. 'Formed first' "
             "holds only in that weak sense"))
# --- S85 initial key structure
D0S85 = {s: dict(dens(r))[0] for s, r in B["REF"].items()}   # (d0 is reused by the docstring loop above)
OV["S85 the initial gate's key structure"] = dict(
    frag3="organised by key on five of the twenty seeds ($\\eta^2$ up to $0.84$ at update 0)",
    draft="At initialisation the near-uniform gate's small variations are organised by key on five of the twenty seeds "
          "(eta^2 up to 0.84 at update 0), ..., which the baseline's discovering runs shed by update 600",
    rec=f"eta^2 by key at update 0 (identical in the four arms: {same0}) > 0.5 on {len(init)} seeds {init}; next highest "
        f"{sorted(D0S85.values())[-len(init) - 1]:.3f}; REF discovering among them at 600: { {s: round(v, 3) for s, v in shed.items()} }",
    status=ms(len(init) == 5 and max(v for _, v in init) == 0.84 and all(v < 0.5 for v in shed.values())),
    note="v2 DISAGREE (six); now five, as the records give. report_7 at the head still says six")
# --- S85 'nothing repairs' in the S85 paragraph, the limits, the conclusion
OV["S85 paragraph: 'nothing repairs'"] = dict(
    frag3="prevents the split that the copy did not repair (S78)",
    draft="the key term, which reads key positions and key identities during the slow phase, prevents the split that the "
          "copy did not repair (S78)",
    rec=f"S78 R4 'neither device' (exact copy {vr['SPLIT2']}, noisy copy with reset {vr['SPLIT2_NOISE']}, K4 {vr['K4']}); "
        f"the same KEY split back after {len(fk)} of {len(fk)} firings on the KEY-ending seeds; S85 KEYHINGE "
        f"{bp['KEYHINGE'][0]} vs {bp['KEYHINGE'][1]}",
    status=ms(neither and len(fk) == 8),
    note="v2 PARTIAL ('nothing repairs'); now scoped to S78's copy. The paragraph frames S85 as a screen")
OV["limits: 'the key term prevents the key split at two streams'"] = dict(
    frag3="the hinge's key term, which on screens prevents the key split at two streams and for the register at a distance",
    draft="the hinge's key term, which on screens prevents the key split at two streams and for the register at a "
          "distance, reads key positions and key identities",
    rec=f"S85 KEYHINGE {bd['KEYHINGE']}/20 vs {bd['REF']}/20 ({bp['KEYHINGE'][0]} vs {bp['KEYHINGE'][1]}, one-sided p "
        f"{mcnemar1(*bp['KEYHINGE']):.3f}), one screen of 20 seeds; key identities read (explore_h_two_b.py l.17)",
    status=ms(bd["KEYHINGE"] == 20 and prevent["KEYHINGE"]),
    note="v2 PARTIAL (no screen marker); now 'on screens' (S85 for two streams; G's S82 for the register)")
OV["limits: two streams"] = dict(
    frag3="the one repair device that works at eight streams did not work here (S78, ``neither device'')",
    draft="every arm is at 65-69/80, the one repair device that works at eight streams did not work here (S78, 'neither "
          "device'), and the prevention that works on a screen (the key term, 20/20) reads key identities",
    rec=f"S78: SPLIT2 {vr['SPLIT2']}, SPLIT2_NOISE {vr['SPLIT2_NOISE']} vs REF -> 'neither device': {neither}; S85 KEYHINGE "
        f"{bd['KEYHINGE']}/20",
    status=ms(neither and bd["KEYHINGE"] == 20),
    note="v2 PARTIAL ('cannot work here', stated as fact); now the measured outcome with its reading's name")
OV["conclusion: S85"] = dict(
    frag3="which on a screen also prevents the two-stream key split that the copy did not repair",
    draft="is found on every seed of a screen when a hinge keeps the gate off key identity during the slow phase, which on "
          "a screen also prevents the two-stream key split that the copy did not repair",
    rec=f"S85 KEYHINGE {bd['KEYHINGE']}/20 (5 vs 0, exploratory); S78 R4 'neither device' (exact copy {vr['SPLIT2']}, "
        f"noisy copy with reset {vr['SPLIT2_NOISE']})",
    status=ms(bd["KEYHINGE"] == 20 and neither),
    note="v2 PARTIAL (no screen marker; 'nothing repairs'); both fixed. The register's 20/20 (G's S82) is also 'of a "
         "screen' now")
# --- S84 index
ba_acc = sorted(c["acc"] for c in gf["BASE"])
ga_acc = sorted(c["acc"] for c in gf["GUARD"])
OV["index: S84 'the guard aims the split'"] = dict(
    frag3="the flat-start guard moves the splits to higher probe accuracies without changing the count",
    draft="the flat-start guard moves the splits to higher probe accuracies without changing the count",
    rec=f"probe accuracy at the firing: GUARD {len(ga_acc)} splits {ga_acc[0]:.3f}-{ga_acc[-1]:.3f} (median "
        f"{statistics.median(ga_acc):.3f}); BASE {len(ba_acc)} splits {ba_acc[0]:.3f}-{ba_acc[-1]:.3f} (median "
        f"{statistics.median(ba_acc):.3f}; {sum(a < 0.5 for a in ba_acc)} below 0.5); bound routed GUARD {gb['GUARD']}/10, "
        f"BASE {gb['BASE']}/10 ({gp['GUARD'][0]} vs {gp['GUARD'][1]})",
    status=ms(ga_acc[0] >= 0.5 and statistics.median(ga_acc) > statistics.median(ba_acc) and gp["GUARD"] == (0, 0)),
    note="v2 PARTIAL ('aims', whose criterion cannot fail at the flat start). Now says what the guard changed. The "
         "pre-fixed reading 'the guard aims the split' (which applies by its rule) is no longer named anywhere in the "
         "draft; harmless, since its count and accuracies are given at tex3:203")
# --- S84 rows whose sentence changed but whose fragment is still found
OV["S84 GUARD vs BASE"] = dict(
    frag3="binds $9/10$ against the base's $9/10$ (0 vs 0)", draft=None, rec=None, status=None,
    note="the body does not quote the pre-fixed reading for this arm ('the guard aims the split', which applies); since the "
         "third draft the index (tex3:60) does not either, and says instead what the guard changed")
OV["S84 GUARD_INT1200 and 'faster without the cost'"] = dict(
    frag3="Checks every 1200 with the guard bind $9/10$", draft=None, rec=None, status=None,
    note="the docstring's median is over BOUND ROUTED runs (explore_h_guard.py l.36-37), as recomputed. The fifth arm and "
         "'costs binding', missing in v2, are now stated (next row)")
costs = {a: (gp[a][1] >= 4 and gp[a][0] == 0) for a in GA[1:]}
NEW3.append(("the guard with fast checks and the evaluation together bind $9/10$ (0 vs 0), and no arm read ``costs binding''",
             "S84 EVAL_GUARD_INT1200 and 'costs binding'",
             "the guard with fast checks and the evaluation together bind 9/10 (0 vs 0), and no arm read 'costs binding'",
             f"EVAL_GUARD_INT1200 source {gcfg['EVAL_GUARD_INT1200']['source']}, guard {gcfg['EVAL_GUARD_INT1200'].get('guard')}, "
             f"every {gcfg['EVAL_GUARD_INT1200']['every']}: BOUND ROUTED {gb['EVAL_GUARD_INT1200']}/10, "
             f"{gp['EVAL_GUARD_INT1200'][0]} vs {gp['EVAL_GUARD_INT1200'][1]}; 'costs binding' (BASE only >= 4 and arm only = "
             f"0) per arm {costs}",
             ms(gb["EVAL_GUARD_INT1200"] == 9 and gp["EVAL_GUARD_INT1200"] == (0, 0) and not any(costs.values())
                and gcfg["EVAL_GUARD_INT1200"]["source"] == "eval" and gcfg["EVAL_GUARD_INT1200"].get("guard") == 0.5),
             "v2: not stated (resolution_H §6, the S84 readings row). Its median transition "
             f"({gtr['EVAL_GUARD_INT1200']:.0f}) is not given, which nothing requires"))
OV["S84 EVAL trigger and the discordant seed"] = dict(
    frag3="so, on our reading, the probe's noise decided whether the first check fired",
    draft="A trigger that reads the 2048-query held-out evaluation instead of the probe binds 10/10 (1 vs 0; the reading "
          "'the evaluation is the better probe' applies by its rule, on a single discordant seed, where the probe read "
          "0.000 then 0.047 and the evaluation 0.066 then 0.068, so, on our reading, the probe's noise decided whether "
          "the first check fired)", rec=None, status=None,
    note="v2 MATCH with an unmarked interpretation; the interpretation is now marked 'on our reading'. The record shows the "
         "two rises at 4800: the probe 0.000 -> 0.047 (a rise >= 0.02, no firing), the evaluation 0.066 -> 0.068 (a rise "
         "< 0.02, a firing)")
OV["S84 summary: count unchanged, timing changed"] = dict(
    frag3="since, on our reading, the base kept enough splits for its merges after spending some",
    draft="These seeds hardly tested the flat start, since, on our reading, the base kept enough splits for its merges "
          "after spending some; the guard changes when the splits fire and, on these seeds, not whether the run binds",
    rec=None, status=None,
    note="v2 MATCH with an unmarked interpretation; now marked 'on our reading'. The count half is the 0 vs 0 pairs")
# --- S58 SWITCH: shape, not count (now marked)
NEW3.append(("on our reading it changed the routing's shape, not its count", "S58 SWITCH: 'shape, not count'",
             "on our reading it changed the routing's shape, not its count, and it was not carried to eight streams",
             f"SWITCH BOUND ROUTED {br58['SWITCH']}/40 vs NONE {br58['NONE']}/40 ({w_n[0]} vs {w_n[1]}); KEY-position map != "
             f"VAL-position map in {sw_d} of SWITCH's {br58['SWITCH']}; R3 'the balance loss helps' "
             f"{'applies' if R3_ else 'does not apply'}",
             ms(br58["SWITCH"] == 36 and br58["NONE"] == 38 and w_n == (1, 3) and sw_d == 34),
             "v2 'not marked' (descriptive table; first pass and second pass NOT). Now marked 'on our reading'; the count "
             "half is 1 vs 3 discordant, so 'not its count' is a reading of a non-difference at n = 40"))
# --- provenance: H's branch moved after the second draft
OV["report commits"] = dict(
    frag3="S85 prevention at two streams (report 7, \\texttt{1e09ecd})", draft=None, rec=None,
    status="PARTIAL" if (R7C != R7 and rc[6] == R6) else ms(rc[7] == R7),
    note=f"report_7.md was rewritten at {R7C} (after {R7}): its key-term window sentence was corrected to the records' "
         "'below 0.2 ... maximum 0.195 ... 1-11 training batches', which is what tex3:228 now says, while tex3:506 still "
         f"cites report 7 at {R7}. The S85 store is unchanged ({blob_same['S85']}), so no number depends on it. The "
         f"corrected report_7 still has '0.21-0.64 at 3600 on some seeds' ({'0.21-0.64 at 3600 on some seeds' in REP7C}) and "
         f"'> 0.5 on 6 seeds' at update 0 ({'> 0.5 on 6 seeds' in REP7C}); the draft follows the records")
OV["branch commit"] = dict(
    frag3="(report 7, \\texttt{1e09ecd}) & \\texttt{1e09ecd} &", draft=None, rec=None,
    status="PARTIAL" if H3 != R7 else "MATCH",
    note=f"the second draft pinned the head; H's branch has since moved to {H3} ({len(h_after)} commits after {R7}: "
         + "; ".join(f"{h} {t[:19]} {m[:60]}" for h, t, m in (x[0:1] + x[1].split(' ', 1) for x in h_after))
         + "). Every Session H store the draft cites is byte-identical at the new head: " + str(all(blob_same.values()))
         + ". The 'commit recorded' column is the commit the draft read, so 1e09ecd is defensible; but the draft's own "
           "S85 sentence follows the report-7 correction at " + R7C)

# apply
ROWS3 = []
for r in ROWS:
    n, loc, item, d, rc_, st, first, note, fr = r
    key = esc(item)
    v2s = V2.get(key, "REMOVED" if st == "REMOVED" else "—")
    if st == "REMOVED" and item not in OV:
        ROWS3.append((n, loc, "REMOVED", item, d, rc_, "REMOVED", v2s, note))
        continue
    o = OV.get(item)
    if o:
        if o.get("status") == "REMOVED":
            ROWS3.append((n, loc, "REMOVED", item, d, rc_, "REMOVED", v2s, o["note"]))
            continue
        l3 = where3(o["frag3"])
        if not l3:
            raise AssertionError(f"tex3 does not contain {o['frag3']!r} ({item})")
        ROWS3.append((n, loc, ", ".join(f"tex3:{x}" for x in l3), item, o["draft"] or d, o["rec"] or rc_,
                      o["status"] or st, v2s, o["note"] if o.get("note") is not None else note))
    else:
        l3 = where3(fr) if fr else []
        if not l3:
            raise AssertionError(f"row {item!r}: v2 fragment {fr!r} not in tex3 and no OV entry")
        ROWS3.append((n, loc, ", ".join(f"tex3:{x}" for x in l3), item, d, rc_, st, v2s, note))
for fr, item, d, rc_, st, note in NEW3:
    l3 = where3(fr)
    if not l3:
        raise AssertionError(f"tex3 does not contain {fr!r}")
    ROWS3.append((None, "—", ", ".join(f"tex3:{x}" for x in l3), item, d, rc_, st, "new in v3", note))
unused = set(OV) - {r[3] for r in ROWS3}
assert not unused, unused

# readings vs docstring, third draft
READ3 = [(t[0], t[1], t[2].replace("tex2:", "tex3:"), t[3], t[4]) for t in READ]
READ3 = [t if t[0] != "S84" else
         (t[0], t[1], "tex3:203 ('faster without the cost' not met: 1200 earlier; 'the evaluation is the better probe' "
          "applies by its rule; the fifth arm 9/10, 0 vs 0; no arm read 'costs binding'); tex3:60 ('moves the splits to "
          "higher probe accuracies without changing the count')",
          ms(aims and not fast and evp and not any(costs.values())),
          f"v2 PARTIAL. Every quoted outcome recomputes (aims: {aims}; faster: {fast}; eval: {evp}; costs binding per arm "
          f"{costs}); the fifth arm and 'costs binding' are now stated; 'aims' is replaced by what the guard changed. "
          "'The guard aims the split' is not named (it applies)") for t in READ3]
for t in READ3:
    assert t[2].count("tex3:") >= 1

DESC3 = []
for t in DESC:
    loc, sc, stmt, mk, note = t
    loc = loc.replace("tex2:", "tex3:")
    if sc == "S84":
        stmt = ("so, on our reading, the probe's noise decided whether the first check fired; These seeds hardly tested "
                "the flat start, since, on our reading, the base kept enough splits for its merges after spending some")
        mk, note = "marked ('on our reading')", "v2 not marked; RESOLVED in v3"
        assert where3("so, on our reading, the probe's noise decided") == [203]
    elif sc == "S58":
        stmt, mk, note = "on our reading it changed the routing's shape, not its count", "marked ('on our reading')", \
            "v2 not marked (NOT RESOLVED in both earlier passes); RESOLVED in v3"
        assert where3("on our reading it changed the routing's shape, not its count") == [207]
    elif sc == "S85" and stmt.startswith("Descriptively"):
        stmt = ("Descriptively, from the gate's eta^2 by key ...: onsets 25-125 (seven seeds) and 275-575 (three seeds), "
                "'so, on our reading, the gate's learning rate is not what lets them form'; below 0.2 (max 0.195); 0.21-0.88 "
                "at 3600 on seven seeds; 'the stream routing having formed first'; five seeds at initialisation")
        mk = "marked ('Descriptively'; the inference 'on our reading')"
        note = ("v2: three numbers disagreed and the onsets generalised; all now from the records. 'The stream routing "
                "having formed first' is not borne out by the dense cells (see its row)")
        assert where3("Descriptively, from the gate's $\\eta^2$ by key every 25 updates") == [228]
    elif sc == "S85":
        _m17 = {a_: {t: round({int(k): v for k, v in B[a_][2317]["dense"].items()}[t]["margin"], 2) for t in (1800, 2400)}
                for a_ in BA}
        _k17 = {a_: round({int(k): v for k, v in B[a_][2317]["dense"].items()}[3600]["etak_key_by_key"], 2) for a_ in BA}
        note = ("unchanged in v3 (still unmarked). " + note + f". Seed 2317's routing margin at 1800/2400 by arm: {_m17}; "
                f"eta^2 by key at 3600: {_k17}. The slow gate alone had routed 2317 by 1800 (the hold ends at 1200, inside "
                "the window), and under the key term alone 2317 was also unrouted at 2400 and also rose after the window "
                "(0.64 at 3600) yet recovered; what the record shows is the combination failing, not the slow gate holding "
                "the gate past the window")
        assert where3("the one failure a seed on which the slow gate held the gate back until the hinge's window had passed")
    elif loc == "tex3:440":
        stmt = "the one repair device that works at eight streams did not work here (S78, ``neither device'')"
        mk, note = "the measured outcome, named", "v2 not marked ('cannot work here'); RESOLVED in v3"
        assert where3("did not work here (S78, ``neither device'')") == [440]
    elif loc == "tex3:342":
        mk = "partly"
        note = ("v2 partly. The S84 interpretations (tex3:203), S58's 'shape, not count' (tex3:207) and 'cannot work here' "
                "(tex3:440) are now marked or scoped. Still unmarked: tex3:228's account of 2317 ('the slow gate held the "
                "gate back until the hinge's window had passed') and tex3:209's recommendation ('the trigger should not fire "
                "when its target already carries key mass')")
    elif loc == "tex3:209":
        assert where3("the trigger should not fire when its target already carries key mass") == [209]
    DESC3.append((loc, sc, stmt, mk, note))
for t in DESC3:
    pass

V3EXTRA = [
    f"report_7.md at {R7C} (corrected after the second draft, before the third): " +
    ("'below the term's threshold 0.2 at every measurement from update 25 to update 2400 (maximum 0.195 over the 20 runs)' "
     f"present: {'maximum 0.195 over the 20 runs' in REP7C}; 'fired on 1-11 training batches' present: "
     f"{'1-11 training batches' in REP7C}; still '0.21-0.64 at 3600 on some seeds': {'0.21-0.64 at 3600 on some seeds' in REP7C}"),
    "S86 (explore_h_allmass, pre-fixed readings at 50da719, 2026-10-11T02:56:33Z, before the third draft) screens the "
    "all-position target that tex3:436 says 'has not been screened'; its preliminary count (explore_out/H/allmass_counts.txt, "
    "e0140ad) is 97 of 107 all-position pairs labelled ok, 107 key-position pairs, 46 coinciding, the same as tex3:436 and "
    "rev9/count_allpos.py. The draft does not mention S86, which is running; tex3:436 is true as of the draft (no S86 result).",
]
_s86 = git("show", f"{BR}:explore_out/H/allmass_counts.txt")
assert "total             50     107         107          97          46      97" in _s86
assert where3("a trigger that reads all positions has not been screened") == [436]


# ══ output ═══════════════════════════════════════════════════════════════════════════════════════════════════════
def cell(n):
    return "—" if n is None else f"tex:{n}"


def main():
    order = {k: i for i, k in enumerate(ORDER + ("REMOVED",))}
    live = [r for r in ROWS3 if r[6] != "REMOVED"]
    gone = [r for r in ROWS3 if r[6] == "REMOVED"]
    rows = sorted(live, key=lambda r: (order[r[6]], int(r[2].split(",")[0][5:])))
    c = collections.Counter(r[6] for r in live)
    changed = [r for r in live if r[3] in OV]
    unchanged = [r for r in live if r[3] not in OV and r[7] != "new in v3"]
    new3 = [r for r in live if r[7] == "new in v3"]
    cc, cu, cn = (collections.Counter(r[6] for r in x) for x in (changed, unchanged, new3))
    trans = collections.Counter(f"{r[7]} -> {r[6]}" for r in ROWS3 if r[7] != "new in v3")
    L = ["# Revision 9 (third draft) on Session H: third-pass check against H's records", "",
         f"Generated by `python3 docs/review_2026-10/rev9/check_H_v3.py` (stdlib + rlib). Third draft: "
         f"{DRAFT3}:{DRAFT_PATH} (origin/claude/rev9-draft), cited tex3:NNN; second draft {DRAFT2} (tex2:NNN); first draft "
         f"{DRAFT1} (tex:NNN). A copy of check_H_v2.py: every row keeps its recomputation and its second-draft anchor "
         f"(asserted as before); it is then located in the third draft by text search. Where the sentence was rewritten "
         f"the row is re-anchored on the new wording (which must be found) and its number re-checked against the record "
         f"it cites; a rewritten sentence without a re-anchor stops the script. Rows for sentences new in the third draft "
         f"are marked 'new in v3'. Records: {BR} via `git show` at the draft's commits (report 3 {R3}; report 4 {R4}; "
         f"report 5 {R5}; report 6 {R6}; report 7 {R7}); branch head now {HEAD}. No draft number is an input.", "",
         f"Rows in the third draft: {len(live)} — " + ", ".join(f"{k} {c.get(k, 0)}" for k in ORDER), "",
         f"- rows re-anchored on rewritten or re-judged sentences: {len(changed)} — " + ", ".join(f"{k} {cc.get(k, 0)}" for k in ORDER),
         f"- rows whose sentence is unchanged: {len(unchanged)} — " + ", ".join(f"{k} {cu.get(k, 0)}" for k in ORDER),
         f"- rows new in the third draft (or newly checkable): {len(new3)} — " + ", ".join(f"{k} {cn.get(k, 0)}" for k in ORDER),
         f"- rows whose sentence is gone: {len(gone)} (REMOVED; all three were already gone in the second draft)",
         "- second-pass status -> third-pass status: " + "; ".join(f"{k} {v}" for k, v in sorted(trans.items())), "",
         "## Stores", ""] + [f"- {x}" for x in INFO] + [f"- {x}" for x in V3EXTRA] + ["",
         "## Rows (DISAGREE, PARTIAL, NOT RECORD-CHECKABLE, MATCH)", "",
         "| tex3 line | tex2 line | first draft | item | draft says (third draft) | recomputed | status | second pass | note |",
         "|---|---|---|---|---|---|---|---|---|"]
    for n, loc2, loc3, item, d, r, s, v2s, note in rows:
        L.append(f"| {loc3} | {loc2} | {cell(n)} | {esc(item)} | {esc(d)} | {esc(r)} | **{s}** | {v2s} | {esc(note)} |")
    L += ["", "## Rows whose sentence is gone (REMOVED)", "",
          "| first draft | item | first-draft text | recomputed now | second pass | note |", "|---|---|---|---|---|---|"]
    for n, loc2, loc3, item, d, r, s, v2s, note in gone:
        L.append(f"| {cell(n)} | {esc(item)} | {esc(d)} | {esc(r)} | {v2s} | {esc(note)} |")
    L += ["", "## Readings vs docstring", "",
          "Readings commit = oldest commit of the screen file on the branch; first run record = oldest commit of the "
          "results file whose `runs` is non-empty; docstring compared between that first commit and the head "
          f"({HEAD}).", "",
          "| screen | file | first commit (readings) | commits touching it | docstring first vs head | first run record | "
          "readings precede first run |", "|---|---|---|---|---|---|---|"]
    for t in TIMING:
        L.append("| " + " | ".join(esc(x) for x in t) + " |")
    L += ["", "| screen | readings as first committed | where the third draft uses them | status | note |",
          "|---|---|---|---|---|"]
    for t in READ3:
        L.append("| " + " | ".join(esc(x) for x in t[:3]) + f" | **{t[3]}** | {esc(t[4])} |")
    L += ["", "## Descriptive statements: marked or not (third draft)", "",
          "| tex3 line | screen | statement | marked? | note |", "|---|---|---|---|---|"]
    for t in DESC3:
        L.append("| " + " | ".join(esc(x) for x in t) + " |")
    L += ["", "## Not checkable from the records", "",
          "- Literature attributions in tex3:207 (Raven's Gumbel device, MoM's / Switch's balance loss) and S24's 'near-tie' "
          "at eight streams (tex3:226) are not H records.",
          "- That seeds 2300-2329 are fresh: taken from explore_h_guard.py l.9-12's scan; not rescanned.",
          "- Session F's S81 (tex3:203) and session G's S82 (tex3:228's comparison, tex3:265, tex3:384's register half, "
          "Table 5) are not H records.",
          "- Whether H's S85 CHECK, run on a dirty tree, used exactly the committed code.",
          "- Commit times are committer-set; push order and history rewrites cannot be established from git.", ""]
    with open(OUT, "w") as f:
        f.write("\n".join(L))
    print(f"wrote {OUT}: " + ", ".join(f"{k} {c.get(k, 0)}" for k in ORDER) + f"; REMOVED {len(gone)}")


if __name__ == "__main__":
    main()
