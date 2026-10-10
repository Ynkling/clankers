#!/usr/bin/env python3
"""registry_L_recount.py: the 32 L rows of the registry (Phases III-IV) marked NOT RECORDED / NOT PRINTED, i.e.
bands, validity gates and conditional readings, recounted from the per-run records in results/L under each test's
own docstring rule, and compared with (a) what the results file itself stores and (b) the registry's derived text.

    python3 docs/review_2026-10/registry_L_recount.py      (stdlib; writes registry_L_recount_out.txt)

Outcomes, as the docstrings fix them:
  BOUND      = transition is not None (test_short_conv.py:70, :486; the record's `transition`, which is also
               recomputed here from the curve as the first evaluation >= 0.95 holding to the end, rlib.transition,
               and any difference printed);
  DISCOVERED = BOUND and final VAL cos < 0.5 (test_router_discovery.py:68; test_router_layout.py:56).
CHECK-based conditions (pairing, conv-at-init) are not run records: the stored flag is reported as stored.
Readings are conditional on verdicts; the branch that applies is derived from the stored verdicts, which
registry_L_addendum.py found to agree with every L verdict row.
"""
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from rlib import transition as tr_curve  # noqa: E402

out = []
diffs_tr = []


def say(*a):
    s = " ".join(str(x) for x in a)
    out.append(s)
    print(s)


def L(t):
    return json.load(open(os.path.join(ROOT, f"results/L/{t}_results.json")))


def arm(d, a):
    o = {}
    for k, r in d["runs"].items():
        p = k.split("|")
        if p[0] == a:
            o[int(p[1])] = r
    return o


def bound(r, t=""):
    st = r.get("transition")
    rc = tr_curve(r["curve"]) if r.get("curve") else None
    if r.get("curve") and "transition_full" not in r and rc != st:
        diffs_tr.append((t, r.get("seed"), st, rc))
    return bool(r.get("ok", True)) and st is not None


def disc(r, t=""):
    return bound(r, t) and r.get("val_cos") is not None and r["val_cos"] < 0.5


def cnt(d, a, f, t):
    rs = arm(d, a)
    return sum(1 for r in rs.values() if f(r, t)), len(rs)


def band40(c, never=True):
    return ("RELIABLE" if c >= 36 else "MAJORITY" if c >= 20 else "MINORITY" if (c >= 1 or not never) else "NEVER")


rows = []
for p in sorted(glob.glob(os.path.join(HERE, "registry", "main_*.tsv"))):
    with open(p) as f:
        rows += list(csv.DictReader(f, delimiter="\t"))
REG = {(r["test"], r["claim_id"]): r for r in rows if r["machine"] == "L"}
res = []


def row(test, cid, recount, stored, basis, rn=None, sn=None):
    """rn, sn: the recount's and the stored field's normalised values (compared for equality)."""
    r = REG[(test, cid)]
    reg = r["verdict_printed"]
    agree = "—" if sn is None else ("AGREE" if rn == sn else "DIFFERS")
    res.append((test, cid, r["kind"], recount, stored, reg, r["counts"], basis, agree))
    say(f"  {test[5:-3]:18s} {cid:22s} recount: {recount} | stored: {stored} [{agree}]")
    say(f"  {'':18s} {'':22s} registry: {reg[:90]!r}; registry counts: {r['counts'][:90]!r}")
    say(f"  {'':18s} {'':22s} basis: {basis}")


say("== recount of L's NOT RECORDED rows (Phases III-IV) from results/L")


def rd_norm(text):
    """'LR3 SHOWN and LR4 NOT SHOWN' -> 'LR3+LR4-'; 'K1-' stays."""
    t = text.replace("NOT SHOWN", "-").replace("SHOWN", "+").replace(" and ", "").replace(" ", "")
    return t


def stored_readings(v):
    rs = v.get("readings") or ([v["reading"]] if v.get("reading") else [])
    return {rd_norm(x) for x in rs}


# test_channel_binding (file: the 10 October rerun on L, 9c53e16; meta.git 886a668)
d = L("channel_binding")
cb = {k.split("|")[0] + "|" + k.split("|")[1]: v for k, v in d["runs"].items()}
s1 = {P: sum(1 for x in cb[f"S1P{P}|B"].values() if x.get("transition") is not None) for P in (4, 8)}
ce = {P: sum(1 for x in cb[f"P{P}|ceiling"].values() if x.get("transition") is not None) for P in (4, 8)}
ceN = {P: len(cb[f"P{P}|ceiling"]) for P in (4, 8)}
g1 = d["gate1"]
row("test_channel_binding.py", "GATE0", f"P=4: S=1 binds {s1[4]}/3 PASS; P=8: S=1 binds {s1[8]}/3 PASS",
    f"gate0 {d['gate0']}", "S1P4|B, S1P8|B transitions (3 seeds each); P=16 dropped (meta.dropped)",
    rn=(s1[4] == 3, s1[8] == 3), sn=(d["gate0"]["4"], d["gate0"]["8"]))
row("test_channel_binding.py", "GATE1",
    f"P=4: ceiling {ce[4]}/{ceN[4]}, fit argmax {g1['P4']['fit_argmax']:.2f} -> VALID; "
    f"P=8: ceiling {ce[8]}/{ceN[8]} -> EXCLUDED",
    f"P4 valid={g1['P4']['valid']}; P8 valid={g1['P8']['valid']} ({g1['P8']['reason']})",
    "P8 ceiling seeds 1, 4 end at " + ", ".join(f"{x['acc']:.3f}" for s, x in sorted(cb['P8|ceiling'].items())
                                                 if x.get('transition') is None)
    + "; seeds 0 and 3 bind at the last evaluation (38400)",
    rn=(ce[4] == ceN[4] and g1["P4"]["fit_argmax"] >= 0.99, ce[8] == ceN[8]), sn=(g1["P4"]["valid"], g1["P8"]["valid"]))

# test_router_discovery
d = L("router_discovery")
c = {a: cnt(d, a, disc, "router_discovery") for a in ("A_ro_nudge", "A_ro_lr10", "A_ro_ln", "A_ro_noise", "A_ro")}
row("test_router_discovery.py", "INVALID", f"A_ro_nudge discovers {c['A_ro_nudge'][0]}/{c['A_ro_nudge'][1]} (< 8) -> INVALID",
    d["verdict"]["verdict"], "DISCOVERED per run (docstring :68); positive-control rule :71",
    rn="INVALID" if c["A_ro_nudge"][0] < 8 else "VALID", sn=d["verdict"]["verdict"])
row("test_router_discovery.py", "VERDICT",
    "INVALID, no candidate interpreted (A_ro_lr10 {}/10, A_ro_ln {}/10, A_ro_noise {}/10; baseline A_ro {}/10)".format(
        c["A_ro_lr10"][0], c["A_ro_ln"][0], c["A_ro_noise"][0], c["A_ro"][0]),
    "counts " + json.dumps(d["verdict"]["counts"]), "per-arm DISCOVERED counts against the stored counts",
    rn={a: c[a][0] for a in c}, sn={a: d["verdict"]["counts"][a] for a in c})

# test_router_curriculum
d = L("router_curriculum")
cn = cnt(d, "A_nudge", disc, "router_curriculum")
row("test_router_curriculum.py", "INVALID", f"A_nudge discovers {cn[0]}/{cn[1]} (>= 8) -> VALID",
    f"counts A_nudge {d['verdict']['counts']['A_nudge']}; verdict {d['verdict']['verdict']}", "positive-control rule",
    rn=cn[0], sn=d["verdict"]["counts"]["A_nudge"])

# test_router_layout: the band is computed on the DISCOVERED count (test_router_layout.py:630, c = discovered)
d = L("router_layout")
for a in ("A_grouped", "A_blocked", "A_shuffled"):
    cd, n = cnt(d, a, disc, "router_layout")
    cbd, _ = cnt(d, a, bound, "router_layout")
    row("test_router_layout.py", f"BAND-{a}", f"discovered {cd}/{n} -> {band40(cd, never=False)} "
        f"(bound {cbd}/{n} -> {band40(cbd, never=False)})", d["verdict"].get(f"band_{a}"),
        "band on discovered (code :630); the docstring (:75) does not name the outcome",
        rn=band40(cd, never=False), sn=d["verdict"].get(f"band_{a}"))

# test_short_conv
d = L("short_conv")
v = d["verdict"]
cc = cnt(d, "ceiling_conv", bound, "short_conv")
row("test_short_conv.py", "VALID", f"ceiling_conv binds {cc[0]}/{cc[1]} -> VALID (CHECK 54 not a record)",
    f"valid={v['valid']}", "ceiling_conv transitions; CHECK 54 (conv inert at init) is stored only as the valid flag",
    rn=cc[0] == 5, sn=v["valid"])
for a in ("B_conv", "A_conv", "A"):
    cb_, n = cnt(d, a, bound, "short_conv")
    row("test_short_conv.py", f"BAND-{a}", f"bound {cb_}/{n} -> {band40(cb_)}", v.get(f"band_{a}"),
        "band on BOUND (code :611-612)", rn=band40(cb_), sn=v.get(f"band_{a}"))
k1, k3 = v["K1"] == "SHOWN", v["K3"] == "SHOWN"
for cid, ap in {"READING[K1+K3-]": k1 and not k3, "READING[K1+K3+]": k1 and k3, "READING[K1-]": not k1}.items():
    row("test_short_conv.py", cid, ("APPLIES" if ap else "does not apply") + f" (K1 {v['K1']}, K3 {v['K3']})",
        f"reading={v.get('reading')}", "branch from the stored K1/K3 verdicts",
        rn=ap, sn=cid[8:-1] in stored_readings(v))

# test_conv_lr
d = L("conv_lr")
v = d["verdict"]
cc = cnt(d, "ceiling_conv4", bound, "conv_lr")
row("test_conv_lr.py", "VALID", f"ceiling_conv4 binds {cc[0]}/{cc[1]}; pairing CHECK 59 stored {v['pairing']} -> "
    + ("VALID" if cc[0] == 5 and v["pairing"] else "UNTESTED"), f"valid={v['valid']}", "test_conv_lr.py:65",
    rn=cc[0] == 5 and v["pairing"], sn=v["valid"])
bands = {}
for a in ("B_conv4", "A_conv4"):
    cb_, n = cnt(d, a, bound, "conv_lr")
    bands[a] = band40(cb_)
    row("test_conv_lr.py", f"BAND-{a}", f"bound {cb_}/{n} -> {bands[a]}", v.get(f"band_{a}"), "band on BOUND (:648)",
        rn=bands[a], sn=v.get(f"band_{a}"))
l3, l4 = v["LR3"] == "SHOWN", v["LR4"] == "SHOWN"
for cid, ap in {"READING[LR3-]": not l3, "READING[LR3+LR4+]": l3 and l4, "READING[LR3+LR4-]": l3 and not l4,
                "READING[A_conv4 RELIABLE]": bands["A_conv4"] == "RELIABLE"}.items():
    sr = stored_readings(v)
    row("test_conv_lr.py", cid, ("APPLIES" if ap else "does not apply")
        + f" (LR3 {v['LR3']}, LR4 {v['LR4']}, A_conv4 {bands['A_conv4']})", f"readings={v.get('readings')}",
        "branch from the stored verdicts and the recounted band",
        rn=ap, sn=(cid[8:-1].replace(" ", "") in sr) or (cid == "READING[A_conv4 RELIABLE]" and "A_conv4RELIABLE" in sr))

# test_p_scaling
d = L("p_scaling")
v = d["verdict"]
cc = cnt(d, "ceiling_conv8", bound, "p_scaling")
row("test_p_scaling.py", "VALID", f"ceiling_conv8 binds {cc[0]}/{cc[1]} (>= 7); pairing CHECK 64 stored {v['pairing']} -> "
    + ("VALID" if cc[0] >= 7 and v["pairing"] else "UNTESTED"), f"valid={v['valid']}", "registry statement",
    rn=cc[0] >= 7 and v["pairing"], sn=v["valid"])
p1, p2, p3 = (v[k] == "SHOWN" for k in ("P1", "P2", "P3"))
for cid, ap in {"READING[P1+P2+]": p1 and p2, "READING[P1+P2-]": p1 and not p2, "READING[P1-]": not p1,
                "READING[P3+]": p3}.items():
    row("test_p_scaling.py", cid, ("APPLIES" if ap else "does not apply") + f" (P1 {v['P1']}, P2 {v['P2']}, P3 {v['P3']})",
        f"readings={v.get('readings')}", "branch from the stored verdicts", rn=ap, sn=cid[8:-1] in stored_readings(v))

# test_load_curriculum, test_curriculum_confirm
d = L("load_curriculum")
cc = cnt(d, "CUR_ceiling", bound, "load_curriculum")
row("test_load_curriculum.py", "VALID", f"CUR_ceiling binds {cc[0]}/{cc[1]} (>= 4); pairing CHECK 72 stored "
    f"{d['verdict']['pairing']} -> " + ("VALID" if cc[0] >= 4 and d["verdict"]["pairing"] else "UNTESTED"),
    f"valid={d['verdict']['valid']}", "registry statement", rn=cc[0] >= 4 and d["verdict"]["pairing"],
    sn=d["verdict"]["valid"])
d = L("curriculum_confirm")
cc = cnt(d, "CUR_ceiling_lo", bound, "curriculum_confirm")
row("test_curriculum_confirm.py", "VALID", f"CUR_ceiling_lo binds {cc[0]}/{cc[1]} (>= 4) -> "
    + ("VALID" if cc[0] >= 4 else "UNTESTED"), f"valid={d['verdict']['valid']}", "registry statement",
    rn=cc[0] >= 4, sn=d["verdict"]["valid"])

# test_scale_axes
d = L("scale_axes")
v = d["verdict"]
c8 = cnt(d, "ceiling8", bound, "scale_axes")
c44 = cnt(d, "ceiling4k4", bound, "scale_axes")
row("test_scale_axes.py", "VALID A", f"ceiling8 binds {c8[0]}/{c8[1]} (>= 2) -> " + ("VALID" if c8[0] >= 2 else "UNTESTED"),
    f"validA={v['validA']}", "test_scale_axes.py:66", rn=c8[0] >= 2, sn=v["validA"])
fa = v["fit"]["fit_argmax"]
row("test_scale_axes.py", "VALID B", f"ceiling4k4 binds {c44[0]}/{c44[1]} (>= 4), k=4 fit argmax {fa:.3f} (>= 0.99) -> "
    + ("VALID" if c44[0] >= 4 and fa >= 0.99 else "UNTESTED"), f"validB={v['validB']}", "test_scale_axes.py:67-68, :630",
    rn=c44[0] >= 4 and fa >= 0.99, sn=v["validB"])

# test_stream_channels
d = L("stream_channels")
v = d["verdict"]
c48 = cnt(d, "ceiling4k8", bound, "stream_channels")
fits = {k: x["argmax"] for k, x in v["fits"].items()}
ok = c48[0] >= 2 and v["pairing"] and all(x >= 0.99 for x in fits.values())
row("test_stream_channels.py", "VALID", f"ceiling4k8 binds {c48[0]}/{c48[1]} (>= 2), pairing CHECK 85 stored {v['pairing']}, "
    f"fit argmax {fits} -> " + ("VALID" if ok else "UNTESTED"), f"valid={v['valid']}", "registry statement",
    rn=ok, sn=v["valid"])

say(f"\nrows recounted: {len(res)}; stored field agrees: {sum(1 for x in res if x[-1] == 'AGREE')}; "
    f"differs: {sum(1 for x in res if x[-1] == 'DIFFERS')}; nothing stored: {sum(1 for x in res if x[-1] == '—')}")
say(f"records whose stored transition differs from the curve's (first eval >= 0.95 holding to the end): {len(diffs_tr)}"
    + (": " + str(diffs_tr[:10]) if diffs_tr else ""))

with open(os.path.join(HERE, "registry_L_recount_out.txt"), "w") as f:
    f.write("\n".join(out) + "\n")
