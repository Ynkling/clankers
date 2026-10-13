#!/usr/bin/env python
"""
explore_b20_addendum.py — EXPLORATORY, not a result. Batch 20's addendum on the 44,400 reproduction (S68's
explore_continue.repro44, imported read-only and unchanged).

The pre-registered check printed DIFFERS with one reason: "C_P8_N256 272's resumes [61200] do not include 43200". The
record's `resumed` list is kept per process (explore_scale_child.run resets explore_b19_model.CK["resumed"] on every
call), so it holds only the last process's resume. C_P8_N256 272 was halted at 43,200, its process replaced, resumed
at 43,200, stopped at a segment end and resumed again at 61,200; the saved record keeps [61200]. This script prints each
of repro44's conditions separately and compares the two runs on every recorded field through 44,400, from the saved
records only (no run). It changes no reading and no file.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_continue as s68


def _grad_upto(r):
    return {int(k): v for k, v in r["grad"].items() if int(k) <= s68.REPRO_AT}


def main():
    st = ec.load_store(s68.NAME)
    pick = lambda arm: [v for k, v in st["runs"].items() if k.startswith(f"{arm}|272|") and v.get("ok")]
    (a,), (b,) = pick("C_P8_N256"), pick("RCHK_P8_N256")
    print("EXPLORATORY, not a result — batch 20 addendum: the 44,400 reproduction, condition by condition")
    print(f"  C_P8_N256 272: cpu {a['cpu']}, git {a['git']}, code {a['code_sha'][:12]}, lr {a['lr']}, seed {a['seed']}")
    print(f"  RCHK_P8_N256 272: cpu {b['cpu']}, git {b['git']}, code {b['code_sha'][:12]}, lr {b['lr']}, seed {b['seed']}")
    print(f"  repro44 as pre-registered: {s68.repro44(a, b)}")
    rows = [
        ("C_P8_N256 halted once at 43200 (the marker, written only by the process that halted after its 43200 checkpoint)",
         a.get("halted_once_at") == s68.HALT_AT, a.get("halted_once_at")),
        ("C_P8_N256's `resumed` contains 43200 (per process: the last process only)", s68.HALT_AT in (a.get("resumed") or []),
         a.get("resumed")),
        ("RCHK never resumed at or after 43200", not any(x >= s68.HALT_AT for x in (b.get("resumed") or [])), b.get("resumed")),
    ]
    n = 0
    for (sa, ca), (sb, cb) in zip(enumerate(a["curve"]), enumerate(b["curve"])):
        if ca[0] > s68.REPRO_AT:
            break
        n += 1
    same = lambda f: a[f][:n] == b[f][:n]
    rows += [
        (f"curve equal at every evaluation through {s68.REPRO_AT} ({n} evaluations)", same("curve"), a["curve"][n - 1]),
        (f"statistics equal at every evaluation through {s68.REPRO_AT} (stats[0] is update 0)", a["stats"][:n + 1] == b["stats"][:n + 1], None),
        (f"gradient norms equal at every recorded step through {s68.REPRO_AT}", _grad_upto(a) == _grad_upto(b) if a.get("grad") and b.get("grad")
         else None, f"{len(_grad_upto(a))} steps" if a.get("grad") else None),
        (f"per-stream accuracy at 43200 and {s68.REPRO_AT}", s68.mark_rows(a).get(43200) == s68.mark_rows(b).get(43200)
         and s68.mark_rows(a).get(s68.REPRO_AT) == s68.mark_rows(b).get(s68.REPRO_AT), s68.mark_rows(a).get(s68.REPRO_AT)),
        (f"weights' SHA-1 at 43200 and {s68.REPRO_AT}", dict(a["shas"]).get(43200) == dict(b["shas"]).get(43200)
         and dict(a["shas"]).get(s68.REPRO_AT) == dict(b["shas"]).get(s68.REPRO_AT), dict(a["shas"]).get(s68.REPRO_AT)),
    ]
    for name, ok, val in rows:
        print(f"    {'ok  ' if ok else ('n/a ' if ok is None else 'FAIL')} {name}" + (f": {val}" if val is not None else ""))
    print("  Reading of the addendum: every measured field agrees through 44,400 and the marker shows the halt at 43,200; the one")
    print("  failed condition is the per-process `resumed` list (the check assumed it accumulates across processes). The")
    print("  pre-registered check's DIFFERS stands as printed; the comparison above is the evidence for the resumed run's 1200 updates.")


if __name__ == "__main__":
    main()
