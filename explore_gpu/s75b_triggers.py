#!/usr/bin/env python
"""
explore_gpu/s75b_triggers.py — descriptions of S75b's split trigger, for the spec's Appendix C (Referee V's review,
fix-list items 9 and 10). EXPLORATORY. It reads committed records and decides nothing: SPLIT_THR stays the rule's
(min of the pilot - 0.03) and no rule reads this output. The driver does not import it, so it is not in the driver's
code SHA.

  python -m explore_gpu.s75b_triggers --check
      The replay below reproduces the table of S75's pre-run note (explore_out/I/gpu/s75_S4_prerun_note_2026-10-12.md,
      "A hazard in the registered WIDE_SPLIT arm") on S75's ORACLE 830-831 curves (allowed inputs), and the counts
      below give the expected values on scripted rows. Run before the pilot.
  python -m explore_gpu.s75b_triggers --replay
      After stage 1 (and again after stage 2): S75's trigger rule, at SPLIT_THR, replayed on the ORACLE evaluation
      curves, ORACLE_PILOT 840-841 and, once they exist, ORACLE 850-851.
  python -m explore_gpu.s75b_triggers --counts
      After stage 2: WIDE_SPLIT_G's checks, per run and in total: fired; blocked by cap, by gap, by one-to-one; the
      eligible checks on a one-to-one map (whatever blocked them); the fires on a map whose least-used channel c0 was in
      use (on_target False).

THE REPLAY (as the pre-run note's). The trigger runs at every multiple of 2400 from 2400 (the reference check: it only
sets prev) to iters - 2400. A check is eligible from 4800 when SET < thr and SET - prev < 0.02; it fires unless 3 have
fired (cap) or the last fire is under 4800 updates back (gap; cap when both). The replay reads the evaluation curve's
overall SET (1024 held-out sequences) at each check, a stand-in for the trigger's 64-sequence probe, and it applies no
guard: an ORACLE's map is one-to-one, so under the guard every eligible check would be blocked by one-to-one. It shows
how often a run at ORACLE quality would be eligible to fire.
"""

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "explore_out", "I", "gpu", "s75b")
S75 = os.path.join(HERE, "explore_out", "I", "gpu", "s75_S4_NVIDIA_GeForce_GTX_1660_SUPER_results.json")
EVERY, REF, FIRST, RISE, MAX, GAP = 2400, 2400, 4800, 0.02, 3, 4800      # explore_i_common l.104-105
MARGIN = 0.03                                                          # SPLIT_THR := min(pilot) - 0.03


def replay(curve, thr, iters):
    """[(step, set, eligible, outcome)] for the checks the trigger makes on this curve (outcome: fired, cap, gap, '')."""
    at = {c["step"]: c["set"] for c in curve}
    prev, n, last, rows = None, 0, None, []
    for step in range(REF, iters - EVERY + 1, EVERY):
        if step not in at:
            continue
        acc = at[step]
        eligible = step >= FIRST and prev is not None and acc < thr and acc - prev < RISE
        out = ""
        if eligible:
            if n >= MAX:
                out = "cap"
            elif last is not None and step - last < GAP:
                out = "gap"
            else:
                out, n, last = "fired", n + 1, step
        rows.append((step, acc, eligible, out))
        prev = acc
    return rows


def counts(rows):
    """The two counts the review asks for, and the rest, from a list of trigger rows (the records' 'checks')."""
    el = [c for c in rows if c.get("eligible")]
    oto = [c for c in el if c.get("map_one_to_one", len(set(c.get("map_before") or [])) == 4)]
    return dict(checks=len(rows), eligible=len(el), fired=sum(bool(c.get("fired")) for c in rows),
                blocked_cap=sum(c.get("blocked_by") == "cap" for c in rows),
                blocked_gap=sum(c.get("blocked_by") == "gap" for c in rows),
                blocked_one_to_one=sum(c.get("blocked_by") == "one-to-one" for c in rows),
                eligible_on_one_to_one=len(oto),
                eligible_on_one_to_one_blocked_by_cap_or_gap=sum(c.get("blocked_by") in ("cap", "gap") for c in oto),
                fired_off_target=sum(bool(c.get("fired")) and c.get("on_target") is False for c in rows))


def _load_all():
    out = {"runs": {}, "meta": {}}
    for p in sorted(glob.glob(os.path.join(OUT, "s75b_*_results.json"))):
        with open(p) as f:
            st = json.load(f)
        out["runs"].update(st.get("runs") or {})
        out["meta"][os.path.basename(p)] = st.get("meta") or {}
    return out


def _record():
    p = os.path.join(OUT, "s75b_stage1_record.json")
    if not os.path.exists(p):
        raise SystemExit("no stage-1 record: run --replay after stage 1")
    with open(p) as f:
        return json.load(f)


def do_replay():
    rec = _record()
    sha = rec.get("sha")
    d = rec.get("decision") or {}
    st = _load_all()
    pil = {int(k.split("|")[1]): r for k, r in st["runs"].items()
           if k.split("|")[0] == "ORACLE_PILOT" and k.split("|")[2] == sha and r.get("ok")}
    if len(pil) < 2:
        raise SystemExit(f"the pilot under {sha} has {len(pil)} ok records: nothing to replay")
    thr = d.get("split_thr")
    note = "SPLIT_THR from the stage-1 record"
    if thr is None:
        thr = min(r["end"]["set"] for r in pil.values()) - MARGIN
        note = "min(pilot) - 0.03; the pilot did not PROCEED, so no SPLIT_THR is in force"
    print(f"S75b: S75's split trigger replayed on ORACLE evaluation curves (a description; spec Appendix C, item 9)")
    print(f"code SHA {sha}; threshold {thr!r} ({note}); the stage-1 decision: "
          f"{'PROCEED' if d.get('proceed') else 'STOP' if d.get('complete') else 'not complete'}")
    print("The curve's SET on 1024 held-out sequences stands in for the 64-sequence probe; no guard (an ORACLE's map is "
          "one-to-one, so the guard would block every eligible check).")
    print()
    print("| run | end.set | eligible checks (step: SET) | the unguarded trigger would fire at |")
    print("|---|---|---|---|")
    runs = [("ORACLE_PILOT", s, r) for s, r in sorted(pil.items())]
    runs += [("ORACLE", int(k.split("|")[1]), r) for k, r in sorted(st["runs"].items())
             if k.split("|")[0] == "ORACLE" and k.split("|")[2] == sha and r.get("ok")]
    for arm, s, r in runs:
        rows = replay(r.get("curve") or [], thr, r.get("iters") or 24000)
        el = [f"{t}: {a:.4f}" for t, a, e, o in rows if e]
        fires = [t for t, a, e, o in rows if o == "fired"]
        print(f"| {arm} {s} | {r['end']['set']:.4f} | {', '.join(el) or '-'} | {fires or '-'} |")


def do_counts():
    st = _load_all()
    s2 = sorted({k.split("|")[2] for k in st["runs"] if k.split("|")[0] in ("ORACLE", "SINGLE", "WIDE_SPLIT_G", "REG3",
                                                                            "WIDE_SLOW")})
    if len(s2) != 1:
        raise SystemExit(f"stage-2 records under {s2 or 'no'} code SHA: run --counts after stage 2, under one code")
    sha = s2[0]
    runs = {int(k.split("|")[1]): r for k, r in sorted(st["runs"].items())
            if k.split("|")[0] == "WIDE_SPLIT_G" and k.split("|")[2] == sha and r.get("ok")}
    print(f"S75b: WIDE_SPLIT_G's split checks (a description; spec Appendix C, item 10), code SHA {sha}")
    keys = list(counts([]))
    print("| seed | " + " | ".join(keys) + " |")
    print("|---" * (len(keys) + 1) + "|")
    allrows = []
    for s, r in sorted(runs.items()):
        c = counts(r.get("checks") or [])
        allrows += r.get("checks") or []
        print(f"| {s} | " + " | ".join(str(c[k]) for k in keys) + " |")
    c = counts(allrows)
    print("| all | " + " | ".join(str(c[k]) for k in keys) + " |")


def do_check():
    ok = True
    with open(S75) as f:
        st = json.load(f)
    want = {0.88: {830: [7200], 831: [4800, 9600, 14400]}, 0.87: {830: [], 831: [4800]}, 0.86: {830: [], 831: [7200]},
            0.85: {830: [], 831: []}}
    for thr, w in want.items():
        for s, fires in w.items():
            r = next(r for k, r in st["runs"].items() if k.split("|")[:2] == ["ORACLE", str(s)])
            got = [t for t, a, e, o in replay(r["curve"], thr, r["iters"]) if o == "fired"]
            good = got == fires
            ok = ok and good
            print(f"{'ok  ' if good else 'FAIL'} replay of S75's ORACLE {s} at {thr}: fires {got} (pre-run note: {fires})")
    rows = replay(next(r for k, r in st["runs"].items() if k.split("|")[:2] == ["ORACLE", "831"])["curve"], 0.88, 24000)
    good = [t for t, *_ in rows] == list(range(2400, 21601, 2400)) and not rows[0][2]
    ok = ok and good
    print(f"{'ok  ' if good else 'FAIL'} the replay checks at 2400 ... 21600; the 2400 check is never eligible")
    one, merged = [0, 1, 2, 3], [0, 0, 2, 3]
    scripted = [dict(step=2400, eligible=False, map_before=one, map_one_to_one=True),
                dict(step=4800, eligible=True, blocked_by="one-to-one", map_before=one, map_one_to_one=True),
                dict(step=7200, eligible=True, fired=True, on_target=True, map_before=merged, map_one_to_one=False),
                dict(step=9600, eligible=True, blocked_by="gap", map_before=one, map_one_to_one=True),
                dict(step=12000, eligible=True, fired=True, on_target=False, map_before=merged, map_one_to_one=False),
                dict(step=16800, eligible=True, fired=True, on_target=True, map_before=merged, map_one_to_one=False),
                dict(step=19200, eligible=True, blocked_by="cap", map_before=one, map_one_to_one=True),
                dict(step=21600, eligible=True, blocked_by="cap", map_before=merged, map_one_to_one=False)]
    c = counts(scripted)
    exp = dict(checks=8, eligible=7, fired=3, blocked_cap=2, blocked_gap=1, blocked_one_to_one=1, eligible_on_one_to_one=3,
               eligible_on_one_to_one_blocked_by_cap_or_gap=2, fired_off_target=1)
    good = c == exp
    ok = ok and good
    print(f"{'ok  ' if good else 'FAIL'} counts on scripted rows: {c}")
    print("ALL CHECKS PASSED" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--replay", action="store_true")
    g.add_argument("--counts", action="store_true")
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if do_check() else 1)
    if a.replay:
        do_replay()
    if a.counts:
        do_counts()


if __name__ == "__main__":
    main()
