#!/usr/bin/env python3 -I
"""Independent recomputation of test_delta_recipe's X, L and pooled numbers from the two results json files.
Reads only raw fields: ok, transition, end.one_to_one, end.stream_acc, checks[*], curve, secs, meta.
"""
import json, math, statistics, sys, collections

ROOT = "the main-line clone/results"
ARMS = ["HEBB_SPLIT", "HEBB_SLOW", "DELTA_SPLIT", "DELTA_SLOW"]
CEILS = ["CEIL_HEBB", "CEIL_DELTA"]
# constants copied from test_window_gate / test_delta_recipe
ACC_THR, RISE, MAX_SPLITS, GAP, FIRST = 0.95, 0.02, 3, 4800, 4800
ROUTED_ACC = 0.9
NO_ROOM = 36
Z = 1.959963984540054


def ok(r):
    return bool(r and r.get("ok"))


def bound(r):
    return ok(r) and r.get("transition") is not None


def routed(r):
    e = r["end"]
    return bool(e["one_to_one"] and min(e["stream_acc"]) >= ROUTED_ACC)


def br(r):
    return bound(r) and routed(r)


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else sum(math.comb(n, i) for i in range(b, n + 1)) / 2 ** n


def wilson(k, n, z=Z):
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def band(c, n):
    if c >= math.ceil(0.9 * n):
        return "RELIABLE"
    if c >= math.ceil(0.5 * n):
        return "MAJORITY"
    return "MINORITY"


def newcombe(a, b, c, d, z=Z):
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    if den == 0:
        phi = 0.0
    else:
        num = a * d - b * c
        phi = (max(num - n / 2, 0) if num > 0 else num) / math.sqrt(den)
    th = p1 - p2
    dl = math.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0.0))
    eu = math.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0.0))
    return th, max(-1.0, th - dl), min(1.0, th + eu)


def load(m):
    d = json.load(open(f"{ROOT}/{m}/delta_recipe_results.json"))
    runs = {}
    for k, r in d["runs"].items():
        arm, seed = k.split("|")
        runs.setdefault(arm, {})[int(seed)] = r
    return d["meta"], runs


def per_machine(m):
    meta, runs = load(m)
    out = {"meta": {k: meta[k] for k in ("git", "cpu", "torch", "workers", "started", "projected_wall_h")}}
    seeds = sorted(runs["DELTA_SPLIT"])
    out["seeds"] = (seeds[0], seeds[-1], len(seeds))
    out["nruns"] = sum(len(v) for v in runs.values())
    out["all_ok"] = all(ok(r) for v in runs.values() for r in v.values())
    out["ceil"] = {k: [(s, r["transition"]) for s, r in sorted(runs[k].items())] for k in CEILS}
    out["counts"] = {k: sum(br(runs[k][s]) for s in seeds) for k in ARMS}
    out["bands"] = {k: (out["counts"][k], band(out["counts"][k], 20), wilson(out["counts"][k], 20)) for k in ARMS}
    # outcome per seed
    def oc(r):
        if bound(r):
            return "BR" if routed(r) else "B-notR"
        return "unbound"
    out["per_seed"] = {s: {k: (oc(runs[k][s]), runs[k][s]["transition"]) for k in ARMS} for s in seeds}
    # D1
    a = sum(br(runs["DELTA_SPLIT"][s]) and br(runs["HEBB_SPLIT"][s]) for s in seeds)
    b = sum(br(runs["DELTA_SPLIT"][s]) and not br(runs["HEBB_SPLIT"][s]) for s in seeds)
    c = sum(br(runs["HEBB_SPLIT"][s]) and not br(runs["DELTA_SPLIT"][s]) for s in seeds)
    dd = 20 - a - b - c
    out["D1"] = dict(a=a, b=b, c=c, dd=dd, d=c - b, newcombe=newcombe(a, b, c, dd))
    # D2, D4
    for name, sk, nk in (("D2", "DELTA_SPLIT", "DELTA_SLOW"), ("D4", "HEBB_SPLIT", "HEBB_SLOW")):
        b = sum(br(runs[sk][s]) and not br(runs[nk][s]) for s in seeds)
        c = sum(br(runs[nk][s]) and not br(runs[sk][s]) for s in seeds)
        out[name] = dict(b=b, c=c, p=mcnemar(b, c), slow=out["counts"][nk], split=out["counts"][sk])
    # D3
    rows = [(s, runs["HEBB_SPLIT"][s]["transition"], runs["DELTA_SPLIT"][s]["transition"]) for s in seeds
            if br(runs["HEBB_SPLIT"][s]) and br(runs["DELTA_SPLIT"][s])]
    out["D3rows"] = rows
    # CHECK 174
    bad = []
    for sk, nk in (("DELTA_SPLIT", "DELTA_SLOW"), ("HEBB_SPLIT", "HEBB_SLOW")):
        for s in seeds:
            A, B = runs[sk][s], runs[nk][s]
            fs = [ch["step"] for ch in A.get("checks", []) if ch["fired"]]
            if fs:
                same = [x for x in A["curve"] if x[0] <= fs[0]] == [x for x in B["curve"] if x[0] <= fs[0]]
            else:
                same = A["curve"] == B["curve"]
            if not same:
                bad.append((sk, s))
    out["check174_bad"] = bad
    # firings and probe-blind
    fir = {}
    for k in ("HEBB_SPLIT", "DELTA_SPLIT"):
        fired = [ch for s in seeds for ch in runs[k][s].get("checks", []) if ch["fired"]]
        fir[k] = dict(n=len(fired), ok=sum(1 for ch in fired if ch["key_ok"]))
        cur_total, blind, alone = 0, 0, 0
        for s in seeds:
            r = runs[k][s]
            cur = {x[0]: x[1] for x in r["curve"]}
            n, last = 0, None
            for ch in r.get("checks", []):
                if ch["step"] >= FIRST:
                    cur_total += 1
                    h = cur.get(ch["step"])
                    if h is not None and ch["acc"] >= ACC_THR and h < ACC_THR:
                        blind += 1
                    room = n < MAX_SPLITS and (last is None or ch["step"] - last >= GAP)
                    if (ch["prev"] is not None and ch["acc"] - ch["prev"] < RISE and room and ch["acc"] >= ACC_THR
                            and not ch["fired"]):
                        alone += 1
                if ch["fired"]:
                    n, last = n + 1, ch["step"]
        fir[k].update(checks_from_4800=cur_total, blind=blind, alone=alone)
    out["firings"] = fir
    # overlap <= 0.1 for every BR run
    def overlap(r):
        g = (r.get("end") or {}).get("stream_gate")
        if not g:
            return None
        return max(sum(min(x, y) for x, y in zip(g[i], g[j])) for i in range(len(g)) for j in range(i + 1, len(g)))
    out["overlap_max_BR"] = max(overlap(runs[k][s]) for k in ARMS for s in seeds if br(runs[k][s]))
    # runtime
    out["secs_total_min"] = sum(r["secs"] for v in runs.values() for r in v.values()) / 60
    out["secs_by_arm_min"] = {k: sum(r["secs"] for r in v.values()) / 60 for k, v in runs.items()}
    # medians of transitions per arm (BR runs)
    out["med_transition"] = {k: statistics.median([runs[k][s]["transition"] for s in seeds if br(runs[k][s])]) for k in ARMS}
    # merged / failure description for the draft's L claim: DELTA_SPLIT three merges and one bound-not-routed
    out["fail_detail"] = {k: [(s, out["per_seed"][s][k][0], runs[k][s]["end"].get("one_to_one"),
                               None if not runs[k][s]["end"].get("ch_map") else len(set(runs[k][s]["end"]["ch_map"])))
                              for s in seeds if not br(runs[k][s])] for k in ARMS}
    return out


def main():
    res = {m: per_machine(m) for m in ("X", "L")}
    for m in ("X", "L"):
        o = res[m]
        print(f"=== {m} ===")
        print(" meta", o["meta"])
        print(" seeds", o["seeds"], "nruns", o["nruns"], "all_ok", o["all_ok"])
        print(" ceil", o["ceil"])
        print(" counts", o["counts"])
        for k, (c, bnd, (lo, hi)) in o["bands"].items():
            print(f"  band {k} {c}/20 {bnd} [{lo:.3f}, {hi:.3f}]")
        d1 = o["D1"]
        th, lo, hi = d1["newcombe"]
        print(f" D1 a={d1['a']} b(DELTA only)={d1['b']} c(HEBB only)={d1['c']} dd={d1['dd']} d={d1['d']} diff={th:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        for n in ("D2", "D4"):
            print(f" {n} {o[n]}")
        rows = o["D3rows"]
        pos = sum(1 for _, h, d in rows if d < h)
        neg = sum(1 for _, h, d in rows if d > h)
        print(f" D3 n={len(rows)} earlier={pos} later={neg} ties={len(rows)-pos-neg} p={mcnemar(pos, neg):.5f} "
              f"med_h={statistics.median([h for _, h, _ in rows])} med_d={statistics.median([d for _, _, d in rows])}")
        print(" CHECK174 bad", o["check174_bad"])
        print(" firings", o["firings"])
        print(" overlap max over BR runs", o["overlap_max_BR"])
        print(f" run secs total {o['secs_total_min']:.1f} min = {o['secs_total_min']/60:.2f} h; by arm {{{', '.join(f'{k}: {v:.1f}' for k, v in o['secs_by_arm_min'].items())}}}")
        print(" med transition (BR)", o["med_transition"])
        print(" failures", o["fail_detail"])
        print(" per seed:")
        for s, v in o["per_seed"].items():
            print("   ", s, v)
    # pooled
    print("=== POOLED ===")
    counts = {k: res["X"]["counts"][k] + res["L"]["counts"][k] for k in ARMS}
    print(" counts", counts)
    for k in ARMS:
        lo, hi = wilson(counts[k], 40)
        print(f"  band {k} {counts[k]}/40 {band(counts[k], 40)} [{lo:.3f}, {hi:.3f}]")
    a = res["X"]["D1"]["a"] + res["L"]["D1"]["a"]
    b = res["X"]["D1"]["b"] + res["L"]["D1"]["b"]
    c = res["X"]["D1"]["c"] + res["L"]["D1"]["c"]
    dd = 40 - a - b - c
    th, lo, hi = newcombe(a, b, c, dd)
    d = c - b
    lab = ("delta not shown worse; ahead by >= 3 pairs (untested)" if d <= -3 else
           "delta not shown worse by more than 2 pairs" if d <= 2 else "delta not shown non-inferior")
    print(f" D1 a={a} b={b} c={c} dd={dd} d={d} diff={th:+.3f} [{lo:+.3f}, {hi:+.3f}]  label: {lab}")
    for n, nk in (("D2", "DELTA_SLOW"), ("D4", "HEBB_SLOW")):
        b = res["X"][n]["b"] + res["L"][n]["b"]
        c = res["X"][n]["c"] + res["L"][n]["c"]
        p = mcnemar(b, c)
        v = "SHOWN" if p < 0.05 else "NOT SHOWN"
        lab = "no room" if counts[nk] >= NO_ROOM else ("helps" if v == "SHOWN" else "not shown")
        print(f" {n} b={b} c={c} p={p:.3e} {v}; no-trigger {counts[nk]}/40 -> label '{lab}'")
    rows = res["X"]["D3rows"] + res["L"]["D3rows"]
    pos = sum(1 for _, h, d in rows if d < h)
    neg = sum(1 for _, h, d in rows if d > h)
    print(f" D3 n={len(rows)} earlier={pos} later={neg} ties={len(rows)-pos-neg} p={mcnemar(pos, neg):.5f} "
          f"med_h={statistics.median([h for _, h, _ in rows])} med_d={statistics.median([d for _, _, d in rows])}")
    fx, fl = res["X"]["firings"], res["L"]["firings"]
    for k in ("HEBB_SPLIT", "DELTA_SPLIT"):
        print(f" firings {k}: {fx[k]['n']+fl[k]['n']} total, {fx[k]['ok']+fl[k]['ok']} ok; probe-alone {fx[k]['alone']+fl[k]['alone']}; blind {fx[k]['blind']+fl[k]['blind']}")
    # the "cost three" description: seeds where DELTA_SLOW BR and DELTA_SPLIT not
    for m in ("X", "L"):
        ps = res[m]["per_seed"]
        print(f" {m}: DELTA_SLOW BR & DELTA_SPLIT not:", [(s, v["DELTA_SPLIT"]) for s, v in ps.items() if v["DELTA_SLOW"][0] == "BR" and v["DELTA_SPLIT"][0] != "BR"])
        print(f" {m}: DELTA_SPLIT BR & DELTA_SLOW not:", [(s, v["DELTA_SLOW"]) for s, v in ps.items() if v["DELTA_SPLIT"][0] == "BR" and v["DELTA_SLOW"][0] != "BR"])
    # two-sided p for 0 vs 3
    print(" two-sided exact p for 0 vs 3:", 2 * mcnemar(3, 0))


main()
