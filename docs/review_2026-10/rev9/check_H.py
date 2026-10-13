#!/usr/bin/env python3
"""check_H.py — the referee's check of what Revision 9 (draft) says about Session H, against H's records.

    python3 docs/review_2026-10/rev9/check_H.py      # writes docs/review_2026-10/rev9/check_H.md

Draft: origin/claude/rev9-draft at 9bfa85a, multichannel_hebbian_report_v9.tex (cited as tex:NNN; every quoted fragment
is asserted to be on that line). Records: origin/claude/explore-H, read with `git show <ref>:<path>` (rlib.load) at the
commits the draft names: report 3 at 7e9197f (S58, S58b, S60, S61, S62, S63 stores), report 4 at 72f0bbc (S78), report 5
at df57087 (S79), and the branch head for S84/S85 status. The draft's numbers are typed in as 'draft says' only; no draft
number is an input to a recomputation. Stdlib only, read-only except for the .md written next to this file.
Definitions: docs/review_2026-10/rlib.py (bound, bound_routed, discovered, transition, fail_class, fail_class_k,
outcome_k, mcnemar1, pairs, shared_max, ch_map); fail_class_v2 re-implemented from ba8901a:fail_class_v2.py l.40-69.
"""
import collections
import difflib
import json
import os
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
R3, R4, R5 = "7e9197f", "72f0bbc", "df57087"
DRAFT_REF, DRAFT_PATH = "9bfa85a", "multichannel_hebbian_report_v9.tex"
TEX = git("show", f"{DRAFT_REF}:{DRAFT_PATH}").split("\n")
OUT = os.path.join(HERE, "check_H.md")
ROWS = []      # (tex line, item, draft says, recomputed, status, note)
READ = []      # readings vs docstring rows
DESC = []      # descriptive statements rows
INFO = []


def P(name):
    return f"explore_out/H/{name}_results.json"


def q(n, frag):
    """Assert a fragment is on draft line n (so every row cites the right line)."""
    if frag not in TEX[n - 1]:
        raise AssertionError(f"tex:{n} does not contain {frag!r}")


def add(n, item, draft, rec, status, note="", frag=None):
    if frag:
        q(n, frag)
    ROWS.append((n, item, str(draft), str(rec), status, note))


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
          "S62": (R3, "query_gate"), "S63": (R3, "hardness"), "S78": (R4, "two"), "S79": (R5, "constants")}
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
add(207, "S58 readings", "no reading could apply", s58_read, "PARTIAL",
    "R2 'the split is the reset' held by its rule; the docstring's own qualifier (bbfea60 l.53-56) prints it as "
    "'uninformative', which is not 'could not apply'. The substance (no device distinguishable at S=4) stands.",
    frag="no reading could apply")

# selection rule (69d01dc l.12-15) recomputed from S58
dev = ("RESET", "GUMBEL_W", "GUMBEL_RW", "SWITCH")
tmed = {a: med([transition(r["curve"]) for r in S58A[a].values() if bound_routed(r)]) for a in dev}
mrg = {a: sum(1 for r in S58A[a].values() if outcome_k(r).startswith("MERGED")) for a in dev}
sel = sorted(dev, key=lambda a: (-br58[a], tmed[a] if tmed[a] is not None else 1e9, mrg[a], dev.index(a)))[:2]
meta_sel = ST["S58b"]["meta"]["selected"]
add(207, "S58b arm selection (rule fixed at 69d01dc)", "the two best-scoring arms, which were the reset and write-gate noise",
    f"rule (devices only, most BOUND ROUTED): {sel} (RESET {br58['RESET']}, GUMBEL_W {br58['GUMBEL_W']}, SWITCH "
    f"{br58['SWITCH']}, GUMBEL_RW {br58['GUMBEL_RW']}); store meta selected {meta_sel}; NONE {br58['NONE']} and SPLIT "
    f"{br58['SPLIT']} are outside the candidate set", "PARTIAL" if sel == meta_sel == ["RESET", "GUMBEL_W"] else "DISAGREE",
    "arms match; 'two best-scoring arms' omits that the rule ranked the four devices only: NONE (38/40) outscored "
    "GUMBEL_W (37/40)", frag="two best-scoring arms")

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
add(409, "KEY-map != VAL-map in bound-routed runs", "34 of the balance-loss arm's 36 bound-routed runs, and two to five runs "
    "per arm elsewhere",
    f"SWITCH {sw_d}/{br58['SWITCH']}; elsewhere in S58: " + ", ".join(f"{a} {other[a]}/{br58[a]}" for a in other)
    + "; S58b: " + ", ".join(f"{a} {other8[a]}/{cnt(E[a], bound_routed)}" for a in other8),
    ms(sw_d == 34 and all(2 <= other[a] <= 5 for a in ("SPLIT", "RESET", "GUMBEL_W", "NONE"))) if other["GUMBEL_RW"] >= 2
    else "PARTIAL",
    "S58's four arms with >= 36 bound-routed runs lie in 2-5; GUMBEL_RW (5 bound-routed runs) and the S58b arms fall "
    "outside 'two to five' (see recomputed)" if not (2 <= other["GUMBEL_RW"] <= 5) else "",
    frag="34 of the balance-loss arm")

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
add(207, "S62: the query's read gate follows the write map", "in 32-36 of the bound runs per arm, and 12/12 at eight streams",
    s62txt, "PARTIAL",
    f"'32-36' are the BOUND ROUTED 'both' counts of {sorted(big)} (report_3.md l.55-62); over the bound runs the draft "
    f"names they are {sorted(qa[a]['both'] for a in big)}; GUMBEL_RW (6 bound runs, {qa['collapse.GUMBEL_RW']['both']} both) "
    f"is silently outside 'every other arm'; eight streams {eight_both}/{eight_n}. S62 is descriptive (no reading).",
    frag="in 32--36 of the bound runs per arm")

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
    "(collapse docstring l.20-21), stated nowhere in the caption", frag="$39/40^{e}$")
q(283, "Session~H's S58, exploratory, 40 seeds")

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
add(207, "S60 stable-max", "a stable-max softmax changed nothing",
    f"DISCOVERED STABLEMAX {cnt(S60A['STABLEMAX'], discovered)}/10 vs REF {cnt(S60A['REF'], discovered)}/10 "
    f"({d_sm[0]} vs {d_sm[1]}); D1 (>= 3 vs 0 either way) -> 'no change at n = 10'; saturated before transition "
    f"{sbs[0]}/{sbs[1]} vs {sbr[0]}/{sbr[1]} -> D2 'does not delay saturation'", "PARTIAL",
    "the pre-fixed descriptive rules give 'no change at n = 10' (D1) and 'does not delay saturation' (D2); 'changed "
    "nothing' drops the n = 10 qualifier, and report_2.md l.68-69 notes that STABLEMAX's gate starts much softer",
    frag="a stable-max softmax changed nothing")

# ══ S63 (hardness) — tex:207 ════════════════════════════════════════════════════════════════════════════════════
S63A = {a: A("S63", a) for a in ("REF", "ANNEAL", "HARD_W", "ORACLE")}
b63 = {a: cnt(S63A[a], bound_routed) for a in ("REF", "ANNEAL", "HARD_W")}
an = only(S63A["ANNEAL"], S63A["REF"])
earlier = sum(1 for s in S63A["REF"] if bound_routed(S63A["REF"][s]) and bound_routed(S63A["ANNEAL"][s])
              and transition(S63A["ANNEAL"][s]["curve"]) < transition(S63A["REF"][s]["curve"]))
helps_a = (an[0] >= 4 and an[1] == 0) or (earlier >= 8 and b63["ANNEAL"] >= b63["REF"])
add(207, "S63 slope (temperature) annealing", "neutral (9/10 against 10/10)",
    f"ANNEAL {b63['ANNEAL']}/10, REF {b63['REF']}/10 ({an[0]} vs {an[1]}); earlier on {earlier} seeds both bound routed; "
    f"pre-fixed reading -> {'hardness helps' if helps_a else 'does not help'}", "MATCH" if not helps_a and
    b63["ANNEAL"] == 9 and b63["REF"] == 10 else "DISAGREE",
    "counts match; 'neutral' is report_3.md l.86's gloss of the pre-fixed 'does not help' (no rule for 'neutral'); the "
    "arm is a temperature anneal 1 -> 0.25 (hardness docstring l.16-17), called 'slope annealing' in the draft",
    frag="neutral ($9/10$ against $10/10$)")
hw = collections.Counter(collapsed_class(outcome_k(r)) for r in S63A["HARD_W"].values() if not bound_routed(r))
hw_m = sorted(round(r["end"]["margin"], 2) for r in S63A["HARD_W"].values() if outcome_k(r).startswith("MERGED"))
add(207, "S63 hard one-hot write gate", "0/10; KEY 3, MERGED 4, OTHER 3",
    f"HARD_W {b63['HARD_W']}/10; failures {dict(sorted(hw.items()))}", ms(b63["HARD_W"] == 0 and hw == {"KEY": 3, "MERGED": 4,
                                                                                                       "OTHER": 3}),
    frag="KEY 3, MERGED 4, OTHER 3")
add(207, "S63 HARD_W mechanism", "the stream labelling of the writes never forms",
    f"4 of the 10 HARD_W runs end MERGED, i.e. STREAM-PARTIAL by margin (end margins {hw_m}, all >= 0.25) with "
    f"streams sharing channels", "PARTIAL",
    "the gradient path (read gate only) is the arm's construction (hardness docstring l.18-19); 'never forms' is "
    "report_3.md l.83-85's mechanism, not a reading, unmarked in the draft, and 4 runs did form a partial stream "
    "labelling (MERGED requires margin >= 0.25)", frag="never forms")

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
    "RESET vs SPLIT and every arm vs NONE); the draft computed them; see 'Descriptive statements'",
    frag="5 vs 0, $p = 0.031$")
cs = only(C["COPY"], C["SPLIT"])
cc = only(C["COPY"], C["COPY_NONOISE"])
r1 = cs[1] - cs[0] <= 1
r3 = abs(cs[1] - cs[0]) <= 1
r2 = cc[0] >= 4 and cc[1] == 0
add(195, "S61 pre-fixed readings and the arm they read", "copy-only arm = exact copy; its readings 'the copy alone "
    "suffices' and 'the reset adds nothing' held; 'the noise matters' did not apply (0 vs 0)",
    f"R1 {'holds' if r1 else 'does not apply'} and R3 {'holds' if r3 else 'does not apply'} on COPY (copy WITH noise, Adam "
    f"untouched) vs SPLIT: SPLIT only {cs[1]}, COPY only {cs[0]}; R2 {'holds' if r2 else 'does not apply'} (COPY vs "
    f"COPY_NONOISE {cc[0]} vs {cc[1]}); the exact copy COPY_NONOISE vs SPLIT is {cn_s[0]} vs {cn_s[1]} (not a reading)",
    "PARTIAL",
    "outcomes and numbers match; but the caption calls them the readings of 'S61's copy-only arm ... the exact copy (no "
    "noise, no reset)', whereas R1 and R3 are defined on COPY, the noisy copy (f630ef2 l.19, 28-30); the exact copy "
    "enters only R2", frag="are one-pair bounds that both held")
add(195, "reset-only arm bound no more often than the no-trigger arm, in every look where one exists",
    "bound no more often than the arm with no trigger where one exists",
    f"S61 (session H): RESET {bc['RESET']}/10 vs NONE {bc['NONE']}/10, RESET only "
    f"{only(C['RESET'], C['NONE'])[0]}, NONE only {only(C['RESET'], C['NONE'])[1]}; S58b has no NONE arm",
    ms(bc["RESET"] <= bc["NONE"]),
    "S61's reset-only arm bound MORE often than its no-trigger arm (4 vs 3; 2 vs 1 discordant, one-sided p 0.5); the "
    "draft's own row tex:191 shows 4/10 vs 3/10", frag="bound no more often than the arm with no trigger")
f61 = {a: [c for r in C[a].values() for c in fired(r)] for a in ("SPLIT", "RESET", "COPY", "COPY_NONOISE")}
nf = sum(len(v) for v in f61.values())
nok = sum(label_ok(c) for v in f61.values() for c in v)
ncopy = sum(1 for a in ("SPLIT", "COPY", "COPY_NONOISE") for c in f61[a] if c.get("row_dist_after", 1) < c["row_dist_before"])
nwg = sum(bool(c.get("w_g_unchanged")) for c in f61["RESET"])
add(201, "S61 firings", "All 83 firings across the four trigger arms copied a channel holding at least two streams onto an "
    "empty one",
    f"{nf} firings ({', '.join(f'{a} {len(v)}' for a, v in f61.items())}); labelled ok {nok}/{nf}; a row copy in "
    f"{ncopy} (SPLIT, COPY, COPY_NONOISE: row_dist_after < row_dist_before); the {len(f61['RESET'])} RESET firings copy "
    f"nothing (w_g_unchanged true on {nwg})", "DISAGREE" if nwg else "MATCH",
    "83 is the count of firings on labelled targets; only 57 of them copied a row. report_3.md l.77-78 says 'targets "
    "labelled correct in all 83 firings'", frag="All 83 firings")
miss = {a: sorted(s for s, r in C[a].items() if not bound_routed(r)) for a in ("SPLIT", "COPY", "COPY_NONOISE")}
same9 = len({tuple(v) for v in miss.values()}) == 1
o483 = {a: outcome_k(C[a][483]) for a in miss}
add(201, "S61 three forms bound the same nine seeds, missed the same one", "missed 483, two streams sharing",
    f"failures {miss}; at 483 {o483}", ms(same9 and miss["SPLIT"] == [483] and all(v == "MERGED (2 share)"
                                                                                for v in o483.values())),
    frag="483, two streams sharing")
add(347, "'the reset adds nothing' shown by S61 on ten seeds", "session H's S61 shows it on ten seeds (the copy alone 9/10, the "
    "full split 9/10)", f"R3 held as a one-pair bound on COPY (noisy copy) vs SPLIT ({cs[1]} vs {cs[0]}); exact copy 9/10, "
    f"SPLIT 9/10", "PARTIAL",
    "the reading held, but on an exploratory screen of 10 seeds and on the noisy-copy arm; 'shows' is stronger than "
    "the one-pair bound the caption tex:195 itself describes", frag="session~H's S61 shows it on ten seeds")

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
    "after all eight firings on the three seeds that ended KEY",
    f"seeds ending KEY (v1) {key_seeds} (v2: {key_seeds_v2}); firings on them {len(fk)}; with a next check "
    f"{len(with_next)}, all 1.00/0.00 there: {all(full(d['next']) for d in with_next)}; without a next check "
    f"{[(d['seed'], d['step']) for d in no_next]} (last check 21600), end statistic (24000) 1.00/0.00: "
    f"{all(full(d['r']['end']) for d in no_next)}", "PARTIAL" if no_next else "MATCH",
    "6 of the 8 firings have a next check; the other 2 fired at the last check and are read from the end-of-run "
    "statistic 2400 later (no check). Under v2 (the classifier 'every test from test_split_copy on' uses, tex:115) "
    "1317 also ended KEY", frag="after all eight firings on the three seeds that ended KEY")
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
add(209, "S78 summary in Sec. split", "the copy resets the gate to uniform and the same key split re-forms within 2400 updates",
    f"uniform after all {len(sp)} exact copies: {unif}; the same KEY split back within 2400 on the {len(fk)} firings of "
    f"the KEY-ending seeds (2 of them only at run end); 1317's half-key pattern back; 1309 had no key split and discovered",
    "PARTIAL", "holds for the firings on seeds that held a key split; stated without the 1309 exception and the "
    "two end-of-run readings", frag="re-forms within 2400 updates")

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
add(228, "S78 unbound runs that change class", "four of S78's seventeen unbound runs change class from STREAM-PARTIAL to KEY "
    "(under the classifier of record); no reading changes",
    f"{len(unb)} unbound; v1 (classifier of record) vs v2 differ on {len(dis)}: {dis}, {dict(dirs)}; S78's readings "
    f"R1-R4 are on DISCOVERED / BOUND ROUTED (explore_h_two.py l.40-46), not on classes", "PARTIAL"
    if len(unb) == 17 and len(dis) == 4 and list(dirs) == ["STREAM-PARTIAL -> KEY"] else "DISAGREE",
    "numbers match; wording inverted: under the classifier of record (v1) these runs ARE STREAM-PARTIAL; they become KEY "
    "under fail_class_v2", frag="Under the classifier of record, four of S78")
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
add(115, "v1/v2 differences on H's recorded unbound two-stream runs (S78's pre-run CHECK population)", "six of the 29 recorded "
    "unbound two-stream runs on H's branch",
    f"with S52's store from outside-ideas {oi} (the CHECK's source): {len(d29)} of {len(recs)} (S43 {by['S43']}, S60 "
    f"{by['S60']}, S52 {by['S52']}); differ = predicted (eta^2 by key >= 0.5 and margin >= 0.25): "
    f"{[x.split(' ')[0] for x in d29] == p29}; {d29}. With the S52 file that is on H's branch ({len(s52h)} W2_K4 records "
    f"only): {len(d_h)} of {len(recs_h)}",
    "PARTIAL" if len(d29) == 6 and len(recs) == 29 else "DISAGREE",
    "6 of 29 recomputes, but only with S52's complete store from claude/outside-ideas; H's branch holds a stale partial "
    "copy of that file "
    f"({len(s52h)} of {len(s52o)} records, last committed "
    f"{git('log', '-1', '--format=%h %cI', BR, '--', 'explore_out/two_stream_keysplits_results.json').strip()}), so 'on "
    "H's branch' is not where those records are",
    frag="six of the 29 recorded unbound two-stream runs")
near = []
for k, r in sorted(s52o.items()):
    if r.get("ok") and r.get("transition") is None and k.startswith("W2_KEYHINGE|") and abs(r["end"]["eta_key_by_key"] - 0.5) < 1e-3:
        near.append((k.split("|")[1], round(r["end"]["eta_key_by_key"], 6), fail_class(r), fail_class_v2(r["end"])))
add(115, "three batch-18 runs at eta^2 = 0.5 by key labelled STREAM-PARTIAL by v1 (outside H's screens; S52 store at "
    f"{oi})", "labelled three batch-18 runs at exactly eta^2 = 0.5 by key STREAM-PARTIAL",
    f"S52 W2_KEYHINGE unbound runs within 1e-3 of 0.5 (seed, eta^2 by key, v1, v2): {near}",
    "PARTIAL" if len(near) == 3 and sum(n[1] >= 0.5 for n in near) < 3 else ms(len(near) == 3),
    "three runs sit at 0.5 to 4 decimals, but only one is >= 0.5; the other two are just below and are STREAM-PARTIAL "
    "under v2 as well, so the margin-first order changed the label of one of the three",
    frag="labelled three batch-18 runs")

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
add(203, "S79 CAP1: 'before any stream had bound'", "before any stream had bound",
    "per firing (seed, firing update, last per-stream measurement at or before it, streams >= 0.95 there, max stream acc): "
    + str(bstream), "DISAGREE" if nbound else "MATCH",
    "per-stream accuracy is recorded only at 4800 and 9600 (stats[].stream_acc); for seed 1329, which fired at 12000, two "
    "streams were already at >= 0.99 at 9600; for 1322 (fired 7200) the record has no measurement between 4800 and the "
    "firing. Sentence copied from report_5.md l.46", frag="before any stream had bound")
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
add(203, "S79 recommendation", "What a replication must state is therefore the cap, the first eligible check and the check "
    "interval; the probe size and the rise threshold can be stated loosely",
    "not a pre-fixed reading; INT1200 changed the interval, the first check (4800 -> 2400) and the gap (4800 -> 2400) "
    "together (fdf7499 l.16); the first eligible check was never varied alone; 'stated loosely' rests on 'robust' "
    "(BASE only - V only <= 1) at n = 10", "PARTIAL",
    "follows the 'Descriptively' marker in the same paragraph but is introduced by 'therefore' as a conclusion, and is "
    "not itself marked as not pre-registered; one of its three elements (the first eligible check) has no arm of its own",
    frag="What a replication must state is therefore")
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

# ══ S84 / S85 status at the branch head — tex:203, 228, 464, 500 ══════════════════════════════════════════════════
gs = load(BR, P("guard")) if exists(BR, P("guard")) else None
garms = collections.Counter(k.split("|")[0] for k in (gs or {"runs": {}})["runs"])
two_b = [p for p in git("ls-tree", "-r", "--name-only", BR, "explore_out/H").split() if "two_b" in p]
chk = json.loads(git("show", f"{BR}:explore_out/H/checks.json"))
s84doc = git("show", f"{BR}:explore_h_guard.py")
add(464, "S84 = flat-start guard and evaluation-based trigger; S85 = two-stream prevention",
    "session H's flat-start guard and evaluation-based trigger (S84) and two-stream prevention (S85)",
    "explore_h_guard.py (834581a) arms GUARD, GUARD_INT1200, EVAL, EVAL_GUARD_INT1200; explore_h_two_b.py arms SLOWGATE, "
    "KEYHINGE, SLOWGATE_KEYHINGE", ms("EVAL " in s84doc and "GUARD " in s84doc), frag="flat-start guard and evaluation-based")
add(203, "S84/S81 attribution", "a guard ... and a trigger that reads the held-out evaluation ... are running in sessions H and F "
    "(S84, S81)", "S84 holds both the guard and the evaluation-reading trigger (guard docstring l.3-5, 24-28); F's S81 runs "
    "the guard on delta channels (l.5-6)", "MATCH", "the sentence's pairing (guard -> H, evaluation -> F) is loose: both are "
    "in S84", frag="(S84, S81)")
add(500, "S84 running", "S84-S85 running",
    f"at {HEAD}: guard_results.json {sum(garms.values())} of 52 records ({dict(garms)}); CHECK entry H/guard "
    f"{chk.get('H/guard', {}).get('time')}", ms(0 < sum(garms.values()) < 52), frag="S84--S85 running")
add(228, "S85 running on 20 seeds", "are running on 20 seeds in session H's S85",
    f"at {HEAD}: no S85 store ({two_b or 'no two_b file under explore_out/H'}), no S85 log, no 'H/two_b' CHECK entry "
    f"(checks.json keys {sorted(chk)}); S84's invocation queued its 52 runs only (s84.log l.20); explore_h_two_b.py "
    f"(code and readings, 834581a) fixes 20 seeds 2310-2329", "DISAGREE",
    "the records show S85 committed but not started; git cannot exclude an uncommitted process, but every other H "
    "screen left a CHECK entry and a log before its first record", frag="running on 20 seeds in session~H's S85")
add(228, "S85 design", "a gate at 1e-4 for the first 1200 updates, and the key term of the hinge under the key-first classifier",
    "SLOWGATE: gate group at 1e-4 for updates 1-1200 (two_b l.16); KEYHINGE: S52's key hinge on updates 1-2400 (l.17); "
    "'the key split only moves' reading counts KEY failures under v2 (l.26)", "MATCH",
    "the draft omits the combined arm SLOWGATE_KEYHINGE", frag="the gate group at $10^{-4}$")

# ══ provenance — tex:500 ═════════════════════════════════════════════════════════════════════════════════════════


def touches(path):
    return [x.split(" ", 1) for x in git("log", "--format=%h %s", BR, "--", path).strip().split("\n")]


rt = {n: touches(f"explore_out/H/report_{n}.md") for n in (1, 2, 3, 4, 5)}
rc = {n: rt[n][0][0] for n in rt}      # the last commit that wrote the report
nr = {k: len(ST[k]["runs"]) for k in ST}
pair = [nr["S58"] + nr["S58b"] + nr["S60"], nr["S61"] + nr["S62"] + nr["S63"], nr["S78"], nr["S79"]]
add(500, "report commits", "report 3 7e9197f; report 4 72f0bbc; report 5 df57087",
    f"last commit writing report_N.md: {rc}; commits touching report_5.md: {[h for h, _ in rt[5]]}",
    ms(rc[3] == R3 and rc[4] == R4 and rc[5] == R5),
    f"report_5.md was first committed partial by the auto-progress commit {rt[5][-1][0]} (13 s earlier) and completed "
    f"in {R5} ('{rt[5][0][1][:40]}...'); reports 1-2 written at {rc[1]}, {rc[2]} (not named in the draft)",
    frag="report 3, \\texttt{7e9197f}")
add(500, "runs per report group", "318; 300; 82; 62",
    f"S58+S58b+S60 {nr['S58']}+{nr['S58b']}+{nr['S60']} = {pair[0]}; S61+S62+S63 {nr['S61']}+{nr['S62']}+{nr['S63']} = "
    f"{pair[1]}; S78 {pair[2]}; S79 {pair[3]}", ms(pair == [318, 300, 82, 62]),
    "S62's 216 are reruns of S58/S58b runs (counted again)", frag="318; 300; 82; 62")
add(500, "branch commit", "df57087", f"df57087 = 'S79 complete, report 5'; branch head now {HEAD} (S84 progress, "
    f"{sum(garms.values())} records)", "MATCH", "pin is fine for S58-S79; S84/S85 status read at the head",
    frag="& \\texttt{df57087} &")

# ══ tex:60 index row ════════════════════════════════════════════════════════════════════════════════════════════
add(60, "index: 'nothing replaces the split'", "nothing replaces the split",
    f"S58 R1 does not apply; S58b: RESET and GUMBEL_W do not carry ({sr8[0]} vs {sr8[1]}, {sg8[0]} vs {sg8[1]}); SWITCH and "
    f"GUMBEL_RW never run at eight streams", "PARTIAL",
    "summary of report_2.md l.78 ('Nothing from this session replaces the KEYMASS split'); only the two carried devices "
    "were tested where the split matters; SWITCH (36/40 at S=4) was not", frag="nothing replaces the split")
add(60, "index: 'the copy alone suffices'", "the copy alone suffices", f"S61 R1 {'holds' if r1 else 'does not apply'} "
    f"(on COPY, the noisy copy)", "MATCH", "pre-fixed reading name; see tex:195 row for which arm it reads",
    frag="the copy alone suffices")
add(60, "index: 'the cap matters'", "the cap matters", f"S79 CAP1 '{lab['CAP1'][0]}' (4 vs 0, one-sided p "
    f"{p3(mcnemar1(4, 0))}); CAP6 '{lab['CAP6'][0]}'", "MATCH",
    "report_5.md l.44's heading for the pre-fixed 'sensitive' (CAP1 only); raising the cap to six was 'robust'",
    frag="the cap matters")
add(60, "index: 'two-stream key splits resist the copy'", "two-stream key splits resist the copy",
    f"S78 R4 'neither device'; KEY split back after {len(fk)} of {len(fk)} firings on the KEY-ending seeds", "MATCH",
    "the reading is 'neither device'; 'resist the copy' is the firing-record description", frag="two-stream key splits resist")

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
     "R3 'the balance loss helps'", "tex:207 'no reading could apply'", "PARTIAL",
     f"R1, R3 do not apply; R2 holds by rule, printed uninformative ({s58_read})"),
    ("S58b", "69d01dc l.12-15 selection rule (devices only); l.21-24 'X carries to eight streams' if (SPLIT only - X only) "
     "<= 1", "tex:207 'a rule fixed beforehand then carried the two best-scoring arms'; tex:189 8 vs 0",
     "PARTIAL", "rule applied as written (RESET 39, GUMBEL_W 37); 'best-scoring' omits NONE (38) which the rule excluded; "
     "the reading 'does not carry' itself is not quoted"),
    ("S60", "705000d l.30-40: D1 'no change at n = 10' unless >= 3 vs 0; D2 'delays saturation' if >= 0.2 lower; D3",
     "tex:207 'ran to the floor in every run', 'a stable-max softmax changed nothing'", "PARTIAL",
     "the tau observation is descriptive (not a reading); 'changed nothing' drops D1's 'at n = 10'"),
    ("S61", "f630ef2 l.28-30: R1 'the copy alone suffices' (COPY vs SPLIT), R2 'the noise matters' (COPY vs COPY_NONOISE "
     ">= 4 vs 0), R3 'the reset adds nothing' (|SPLIT only - COPY only| <= 1)", "tex:195 caption; tex:60; tex:347",
     "PARTIAL", "outcomes quoted correctly (R1, R3 held; R2 did not apply, 0 vs 0), but tex:195 attributes R1/R3 to the "
     "exact-copy arm; they read COPY, the copy with noise"),
    ("S62", "f630ef2: descriptive, no reading", "tex:207 '32-36 of the bound runs per arm, 12/12'", "MATCH",
     "presented as a description; range excludes GUMBEL_RW (see table)"),
    ("S63", "f630ef2 l.23-26: 'hardness helps' if X only >= 4 and REF only = 0, or earlier on >= 8 with no loss; else "
     "'does not help'", "tex:207 'neutral' (ANNEAL), 'destroyed routing' (HARD_W)", "PARTIAL",
     "the pre-fixed outcome for both arms is 'does not help'; 'neutral' and 'destroyed routing' are the report's words; "
     "HARD_W 0 vs 10 (p 0.00098) supports 'destroyed' descriptively, the mechanism is unmarked"),
    ("S78", "fdf7499 l.40-46: R1 'the split carries to two streams', R2 'noise is needed at k = 2', R3 'spare channels "
     "are the better device', R4 'neither device'", "tex:226 'the pre-fixed reading is \"neither device\"'", "MATCH",
     "R4 holds; R1-R3 do not apply; the firing-table description is the pre-fixed DIAGNOSTICS (l.47-49)"),
    ("S79", "fdf7499 l.26-32: robust / sensitive / inconclusive; CAP6 'the extra splits are harmless'",
     "tex:203", "MATCH", "every label quoted as recomputed; the recommendation is not a reading (see table)"),
    ("S84", "834581a l.33-40: 'the guard aims the split', 'faster without the cost', 'the evaluation is the better probe', "
     "'costs binding'", "tex:203, 464 (running; no reading quoted)", "MATCH", "no S84 reading is quoted"),
    ("S85", "834581a l.23-27: 'prevention works', 'the key split only moves'", "tex:228 (no reading quoted)", "MATCH",
     "no S85 reading is quoted; status 'running' is not supported (see table)"),
]

DESC += [
    ("tex:201", "S61", "Two identical rows, their symmetry broken only by their different Adam moments and the data, separate "
     "two merged streams as well as two noisy ones.", "not marked",
     "restates R2's 'does not apply' (COPY vs COPY_NONOISE 0 vs 0) as an equality claim; the prereg audit (branch_H.md) "
     "flagged the README's 'the noise does not matter' for the same step; Adam moments were not measured"),
    ("tex:201", "S61", "The exact copy is therefore the split's form in every later screen and in the next main-line test.",
     "not marked (a design choice, not a finding)", "factually true of S78 (SPLIT2), S79 and S84 docstrings"),
    ("tex:191", "S61", "copy only vs reset only 5 vs 0, p = 0.031", "not marked",
     "a comparison the docstring did not print or pre-fix (f630ef2 l.26-31); the numbers recompute"),
    ("tex:203", "S79", "four of CAP1's five failures ... before any stream had bound; fast-check failures used all three "
     "splits by 7200-10800; the plateau rule fires during the flat start ...", "marked ('Descriptively, and not "
     "pre-registered')", f"'before any stream had bound' is contradicted by seed 1329 (see table)"),
    ("tex:203", "S79", "What a replication must state is therefore the cap, the first eligible check and the check interval; "
     "the probe size and the rise threshold can be stated loosely.", "ambiguous (after the marker, but phrased as a "
     "conclusion)", "the first eligible check was never varied alone"),
    ("tex:207", "S63", "with the write one-hot, the gate gets gradient only through the read, and the stream labelling of "
     "the writes never forms", "not marked", "4 of 10 HARD_W runs end MERGED (margin >= 0.25)"),
    ("tex:207", "S58", "it changed the routing's shape, not its count", "not marked",
     "report_1.md l.54's description; the count part is 36 vs NONE 38 (1 vs 3 discordant), the shape part 34/36"),
    ("tex:209", "S78", "The copy works at eight streams because the idle channel it copies onto is empty; where every "
     "channel is busy it has nothing to offer.", "not marked",
     "report_4.md l.100-102's take-away; S78 has no eight-stream arm and R4 pre-registers only 'neither device'"),
    ("tex:226", "S78", "At eight streams the copy creates a near-tie between a crowded channel and an empty one; at k = S it "
     "only resets the gate to where the gradient takes it again.", "not marked",
     "the eight-stream half is S24's (batch 8) measurement, not an H record; the k = S half restates the firing table"),
    ("tex:226", "S78", "Session H's reading of the mechanism (Adam moments) ...", "marked ('was not measured and is not a "
     "reading')", "correct"),
    ("tex:312", "S61/S78", "it is the copy that does it ... a key split that the copy cannot undo, because there is no idle "
     "channel", "not marked", "the first half rests on S58b/S61 and the pre-registered test_split_copy (outside H); the "
     "'because' is S78's unregistered take-away"),
    ("tex:338", "S78/S79", "'noise helps a little at two streams' and 'the check interval and the first eligible check act "
     "through the flat start' are quoted as descriptions and 'marked where they appear'", "partly",
     "'helps a little' does not appear in the draft outside tex:338 (good); the flat-start sentence is marked at tex:203, "
     "but the replication recommendation drawn from it is not"),
]


# ══ output ═══════════════════════════════════════════════════════════════════════════════════════════════════════
def esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def main():
    order = {"DISAGREE": 0, "PARTIAL": 1, "NOT RECORD-CHECKABLE": 2, "MATCH": 3}
    rows = sorted(ROWS, key=lambda r: (order[r[4]], r[0]))
    c = collections.Counter(r[4] for r in ROWS)
    L = ["# Revision 9 (draft) on Session H: check against H's records", "",
         f"Generated by `python3 docs/review_2026-10/rev9/check_H.py` (stdlib + rlib). Draft: {DRAFT_REF}:{DRAFT_PATH} "
         f"(origin/claude/rev9-draft), cited tex:NNN; every row's quoted fragment is asserted to be on that line. Records: "
         f"{BR} via `git show` at the draft's commits (report 3 {R3}: S58, S58b, S60, S61, S62, S63; report 4 {R4}: S78; "
         f"report 5 {R5}: S79) and at the branch head {HEAD} for S84/S85. No draft number is an input.", "",
         f"Rows: {len(ROWS)} — " + ", ".join(f"{k} {c.get(k, 0)}" for k in order), "",
         "## Stores", ""] + [f"- {x}" for x in INFO] + ["",
         "## Rows (DISAGREE, PARTIAL, NOT RECORD-CHECKABLE, MATCH)", "",
         "| tex line | item | draft says | recomputed | status | note |", "|---|---|---|---|---|---|"]
    for n, item, d, r, s, note in rows:
        L.append(f"| tex:{n} | {esc(item)} | {esc(d)} | {esc(r)} | **{s}** | {esc(note)} |")
    L += ["", "## Readings vs docstring", "",
          "Readings commit = oldest commit of the screen file on the branch; first run record = oldest commit of the "
          "results file whose `runs` is non-empty; docstring compared between that first commit and the head "
          f"({HEAD}).", "",
          "| screen | file | first commit (readings) | commits touching it | docstring first vs head | first run record | "
          "readings precede first run |", "|---|---|---|---|---|---|---|"]
    for t in TIMING:
        L.append("| " + " | ".join(esc(x) for x in t) + " |")
    L += ["", "| screen | readings as first committed | where the draft uses them | status | note |", "|---|---|---|---|---|"]
    for t in READ:
        L.append("| " + " | ".join(esc(x) for x in t[:3]) + f" | **{t[3]}** | {esc(t[4])} |")
    L += ["", "## Descriptive statements: marked or not", "",
          "| tex line | screen | statement | marked? | note |", "|---|---|---|---|---|"]
    for t in DESC:
        L.append("| " + " | ".join(esc(x) for x in t) + " |")
    L += ["", "## Not checkable from the records", "",
          "- Literature attributions in tex:207 (Raven's Gumbel device, MoM's / Switch's balance loss) and S24's 'near-tie' "
          "at eight streams (tex:226) are not H records.",
          "- Whether S85 is running in a process that has not committed anything: git shows only what was committed "
          "(the row reads the records).",
          "- Commit times are committer-set; push order and history rewrites cannot be established from git.", ""]
    with open(OUT, "w") as f:
        f.write("\n".join(L))
    print(f"wrote {OUT}: " + ", ".join(f"{k} {c.get(k, 0)}" for k in order))


if __name__ == "__main__":
    main()
