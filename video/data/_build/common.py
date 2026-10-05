"""Shared helpers for extracting video data from results/X and results/L (no torch needed).
Classification rules are re-implemented from the repo (file:line given) so the outputs can be
checked against the report."""
import json
import math
import os
import re
import statistics

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # repo root (this file lives in video/data/_build/)
OUT = os.path.join(ROOT, "video", "data")
EVAL_EVERY = 1200

# test_router_layout.py:198-199, test_router_discovery.py:175, test_stream_recipe.py:250
MARGIN_PARTIAL, ETA_SPLIT, COLLAPSE, ROUTED_ACC = 0.25, 0.5, 0.15, 0.9


def load(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return json.load(f)


def runs_of(d, arm):
    return {int(k.split("|")[1]): v for k, v in d["runs"].items() if k.split("|")[0] == arm}


def ok(r):
    return bool(r and r.get("ok"))


def bound(r):            # test_conv_lr.py:503 bound_r
    return bool(ok(r) and r.get("transition") is not None)


def discovered(r):       # bound and final VAL cos < 0.5 (test_router_discovery.py:68)
    return bool(ok(r) and r.get("discovered"))


def fail_class(e):       # test_router_layout.py:293
    if e["margin"] >= MARGIN_PARTIAL:
        return "STREAM-PARTIAL"
    if e["eta_key_by_key"] >= ETA_SPLIT:
        return "KEY"
    if max(e["eta_key_by_half"], e["eta_key_by_index"]) >= ETA_SPLIT:
        return "POSITION"
    return "OTHER"


def fail_class_k(e):     # test_scale_axes.py:328
    return fail_class(dict(margin=e["margin"], eta_key_by_key=e["etak_key_by_key"],
                           eta_key_by_half=e["etak_key_by_block"], eta_key_by_index=e["etak_key_by_index"]))


def shared_max(cm):      # test_stream_channels.py:482
    return max(cm.count(c) for c in cm)


def routed_k(r):         # test_stream_recipe.py:751
    e = r["end"]
    return bool(e["one_to_one"] and min(e["stream_acc"]) >= ROUTED_ACC)


def outcome_k(r):        # test_stream_recipe.py:756
    if r["transition"] is not None:
        return "BOUND ROUTED" if routed_k(r) else "BOUND NOT routed"
    fc = fail_class_k(r["end"])
    if fc == "STREAM-PARTIAL":
        sm = shared_max(r["end"]["ch_map"])
        return f"MERGED ({sm} share)" if sm >= 2 else "STREAM-PARTIAL one-to-one"
    return f"non-stream {fc}"


def tag2(r, part0=True):  # test_slow_start.py:1109 tag(), k=2 branch
    if r["transition"] is not None:
        if part0:
            return "DISCOVERED" if r["discovered"] else "BOUND, VAL cos >= 0.5"
        return "BOUND"
    return fail_class(r["end"]) + ("  collapsed" if r["collapsed"] else "")


def tagk(r):
    return outcome_k(r) + ("  collapsed" if r["collapsed"] else "")


def mcnemar_greater(b, c):   # test_conv_lr.py:260
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(b, n + 1)) / 2 ** n


def fisher_greater(a, n1, b, n2):   # test_router_confirm.py:134
    N, K = n1 + n2, a + b
    tot = math.comb(N, n1)
    return sum(math.comb(K, x) * math.comb(N - K, n1 - x) for x in range(a, min(K, n1) + 1)) / tot


def paired(new, old, seeds, fn):
    both = [s for s in seeds if ok(new.get(s)) and ok(old.get(s))]
    b = sum(1 for s in both if fn(new[s]) and not fn(old[s]))
    c = sum(1 for s in both if fn(old[s]) and not fn(new[s]))
    return dict(n=len(both), new=sum(bool(fn(new[s])) for s in both), old=sum(bool(fn(old[s])) for s in both),
                b=b, c=c, p=round(mcnemar_greater(b, c), 6))


def acc_list(r, nd=4):
    """Held-out accuracy at every evaluation; evaluation i (0-based) is at step 1200*(i+1)."""
    steps = [c[0] for c in r["curve"]]
    assert steps == [EVAL_EVERY * (i + 1) for i in range(len(steps))], steps
    return [round(c[1], nd) for c in r["curve"]]


def r4(x, nd=4):
    return None if x is None else round(float(x), nd)


def med_range(xs):
    xs = list(xs)
    return None if not xs else dict(median=statistics.median(xs), min=min(xs), max=max(xs), n=len(xs))


def write(name, obj):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"), ensure_ascii=False)
    size = os.path.getsize(path)
    print(f"wrote {path}: {size / 1024:.1f} KB")
    assert size < 200 * 1024, "too big"
    return path


# ── L's recipe-scope log (results/L/recipe_scope_L_final.log) ───────────────
LLOG = "results/L/recipe_scope_L_final.log"


def parse_llog():
    """L's per-seed rows and curves for the arms test_recipe_scope printed: HONLY0 + recorded A0,
    SLOW0, HINGE0 (k=2, seeds 280-299); HONLY4k16 + recorded A4k16, HINGE4k16 (k=16, 240-259);
    SC8_H + recorded SC8 (S=8 curriculum, 260-279). Returns (rows, curves, stage_curves)."""
    lines = open(os.path.join(ROOT, LLOG)).read().splitlines()
    i0 = next(i for i, l in enumerate(lines) if l.startswith("PER-SEED RAW RESULTS"))
    i1 = next(i for i, l in enumerate(lines) if l.startswith("COUNTS"))
    i2 = next(i for i, l in enumerate(lines) if l.startswith("CURVES"))
    i3 = next(i for i, l in enumerate(lines) if l.startswith("THE OTHER MACHINE"))
    rows = {}
    k16 = {"HONLY4k16", "A4k16", "HINGE4k16", "SC8_H", "SC8"}
    k2 = {"HONLY0", "A0", "SLOW0", "HINGE0"}
    for ln in lines[i0:i1]:
        t = ln.split()
        if len(t) < 6 or not t[1].isdigit() or t[0] not in k2 | k16:
            continue
        arm, seed = t[0], int(t[1])
        acc = float(t[2])
        tr = None if t[3] == "--" else int(t[3])
        if arm in k2:
            # arm seed acc tr VALcos m_end ROUTED* eta@1200 eta@4800 eta@end hinge flag outcome...
            row = dict(acc=acc, transition=tr, val_cos=float(t[4]), margin_end=float(t[5]), routed_star=t[6],
                       eta_1200=t[7], eta_4800=t[8], eta_end=t[9], hinge=t[10], flag=t[11],
                       outcome=" ".join(t[12:]))
        else:
            # arm seed acc tr map 1:1 sh per-stream ROUTED eta eta eta hinge flag outcome...
            row = dict(acc=acc, transition=tr, ch_map=[int(x) for x in t[4].split(",")], one_to_one=t[5] == "y",
                       shared=int(t[6]), stream_acc=[float(x) for x in t[7].split("/")], routed=t[8] == "Y",
                       eta_1200=t[9], eta_4800=t[10], eta_end=t[11], hinge=t[12], flag=t[13],
                       outcome=" ".join(t[14:]))
        rows.setdefault(arm, {})[seed] = row
    curves, stage = {}, {}
    rx = re.compile(r"^\s+\(?([A-Za-z0-9_]+)\)?\s+s(\d+)\s+([\d\s]+?)(?:\s+->\s+(.*))?$")
    for ln in lines[i2 + 2:i3]:
        m = rx.match(ln)
        if not m:
            continue
        arm, seed, vals, out = m.group(1), int(m.group(2)), [int(x) for x in m.group(3).split()], m.group(4)
        if arm == "stage":
            stage[seed] = vals
        else:
            curves.setdefault(arm, {})[seed] = dict(acc_x100=vals, outcome=out)
    return rows, curves, stage
