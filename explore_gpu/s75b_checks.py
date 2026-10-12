#!/usr/bin/env python
"""
explore_gpu/s75b_checks.py — the CHECKs of explore_gpu/s75b.py (the S75b spec at 114cd08). EXPLORATORY tooling.

    python -m explore_gpu.s75b_checks --device cpu     (this container; output explore_gpu/s75b_checks_cpu_output.txt / .json)
    python -m explore_gpu.s75b_checks --device cuda    (each card, before stage 1; output ..._cuda_<GPU>_output.txt / .json)

  B1  the task CHECK at configuration B: explore_i_task.check_task at its default n = 4096 (S75's n = 512 fails at B on a
      tolerance).                                                                                                  cpu
  B2  the guarded trigger's mechanics (threshold 1.01, rise 1.0; the source -> channel map scripted per check through
      explore_i_common.routing_layer): a check on a one-to-one map is blocked "one-to-one" and leaves the fire count and
      the last-fire step unchanged and W_g and its Adam state untouched; a run blocked at 4800 whose map is merged at 7200
      fires at 7200; gap and cap rows as S75's rule (cap before one-to-one); with every map merged the guarded trigger's
      rows equal explore_i_common.make_split's on the same model (the fields both record) and the two models end
      bitwise equal.                                                                                                cpu
  B3  the routing margin in chunks of 16 sequences equals the 64-sequence chunks to 1e-12 (relative), at L = 1536.  cpu
  B4  the stage-1 rules on constructed pilot records: proceed iff mean >= 0.912 and each >= 0.9 (edges included);
      SPLIT_THR = min - 0.03 unrounded; a missing or failed record never proceeds.                                   cpu
  B5  stage 2's refusals: without a committed stage-1 record; when the record and the pilot records disagree; when the
      pilot did not proceed; a card that is not the machine's (the GPU-name rule); and the machine -> arms allocation.  cpu
  B6  S75's code SHA is still 5fb120a92d69 (S75b imports S75's modules and changes none).                            cpu
  B7  LOCK-STEP (explore_gpu.checks.lockstep, Judge, same-device tolerances) batched vs sequential at L = 1536 for each
      S75b model: ORACLE, SINGLE, WIDE (WIDE_SPLIT_G's and WIDE_SLOW's model), REG3 (register loop) and REG3 with the scan
      against the sequential loop; cpu 2 updates on seed 850, cuda 20 updates on seeds 850-851.               cpu and cuda
  B8  the whole driver end to end on a small configuration (S = 4, NB = 3, M = 30; 1200 updates; evaluation every 400;
      the split's schedule 400 / 400 / 800; the proceed bar at 0 so stage 2 runs): stage 1, the stage-1 record, stage 2
      (fallback: every arm), the report; the benchmark path on the small configuration (both register modes).        cpu
"""

import argparse
import contextlib
import copy
import json
import math
import os
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch

import explore_i_task as it
import explore_i_common as ic
from explore_gpu import device as dv
from explore_gpu import run as gr
from explore_gpu import s75b as sb
from explore_gpu import checks as ck

SMALL = dict(S=4, V=4, NB=3, lmin=6, lmax=14, M=30)


def b1_task(dev):
    if dev.type != "cpu":
        return dict(ok=True, skipped="cpu only")
    r = it.check_task(sb.task())
    return dict(ok=all(r.values()), failed=[k for k, v in r.items() if not v], n=4096)


@contextlib.contextmanager
def scripted_routing(script):
    """explore_i_common.routing_layer -> the real statistics with ch_map / one_to_one replaced by script[step]."""
    real = ic.routing_layer
    cur = {}

    def rl(g, x, task):
        out = real(g, x, task)
        cmap = script[cur["step"]]
        out.update(ch_map=list(cmap), one_to_one=len(set(cmap)) == task.S)
        return out
    ic.routing_layer = rl
    try:
        yield cur
    finally:
        ic.routing_layer = real


def _wide_model(task, seed=7):
    torch.manual_seed(seed)
    m = gr.make_model(gr.arm_spec("WIDE_SPLIT", task.S), task, seed)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    x = task.make_batch(8, torch.Generator().manual_seed(11))[0]
    lo = m(x[:, :-1])[0]
    mask = gr.icmc_loss_mask(task, x)
    loss = torch.nn.functional.cross_entropy(lo[mask], x[:, 1:][mask])
    opt.zero_grad()
    loss.backward()
    opt.step()
    return m, opt


def b2_guard(dev):
    if dev.type != "cpu":
        return dict(ok=True, skipped="cpu only")
    task = it.ICMCTask(**SMALL)
    one, merged = [0, 1, 2, 3], [0, 0, 2, 3]
    steps = list(range(2400, 26400 + 1, 2400))
    script = {2400: one, 4800: one, 7200: merged, 9600: merged, 12000: one, 14400: merged, 16800: merged,
              19200: merged, 21600: one, 24000: merged, 26400: merged}
    out = {}
    with scripted_routing(script) as cur:
        m, opt = _wide_model(task)
        trig = sb.make_split_guarded(task, 7, 26400, 1.01, rise=1.0)
        snap = None
        for t in steps:
            cur["step"] = t
            if t == 4800:
                snap = (m.W_g.detach().clone(), {k: v.clone() for k, v in opt.state[m.W_g].items() if torch.is_tensor(v)})
            trig(m, t, opt)
            if t == 4800:
                out["w_g_untouched_at_4800"] = torch.equal(snap[0], m.W_g.detach()) and all(
                    torch.equal(v, opt.state[m.W_g][k]) for k, v in snap[1].items())
        rows = {r["step"]: r for r in trig.rows}
    fired = [t for t, r in rows.items() if r["fired"]]
    by = {t: r.get("blocked_by") for t, r in rows.items() if r.get("blocked")}
    out.update(fired=fired, blocked=by)
    want_fired = [7200, 14400, 19200]
    want_blocked = {4800: "one-to-one", 9600: "gap", 12000: "one-to-one", 16800: "gap", 21600: "cap", 24000: "cap",
                    26400: "cap"}
    ok = fired == want_fired and by == want_blocked and out["w_g_untouched_at_4800"]
    # every map merged: the guarded trigger's rows equal explore_i_common.make_split's, the models end bitwise equal
    allm = {t: merged for t in steps}
    with scripted_routing(allm) as cur:
        m1, o1 = _wide_model(task)
        m2, o2 = ck._copy_model(m1), None
        o2 = torch.optim.Adam(m2.parameters(), lr=1e-3)
        o2.load_state_dict(o1.state_dict())
        tg = sb.make_split_guarded(task, 7, 26400, 1.01, rise=1.0)
        tm = ic.make_split(task, 7, 26400, 1.01, rise=1.0)
        for t in steps:
            cur["step"] = t
            tg(m1, t, o1)
            tm(m2, t, o2)
    same_rows = len(tg.rows) == len(tm.rows) and all(all(a.get(k) == v for k, v in b.items())
                                                       for a, b in zip(tg.rows, tm.rows))
    same_model = all(torch.equal(a, b) for a, b in zip(m1.state_dict().values(), m2.state_dict().values()))
    out.update(all_merged_rows_equal=same_rows, all_merged_models_equal=same_model,
               all_merged_fired=[r["step"] for r in tg.rows if r["fired"]])
    return dict(ok=bool(ok and same_rows and same_model), **out)


def b3_margin(dev):
    if dev.type != "cpu":
        return dict(ok=True, skipped="cpu only")
    task = sb.task()
    x = task.make_batch(64, torch.Generator().manual_seed(3))[0][:, :-1]
    g = torch.softmax(torch.randn(64, x.shape[1], 4, generator=torch.Generator().manual_seed(4)), -1)
    src = task.stream_labels(x)
    word = x >= task.S
    orig = sb._MARGIN.get("orig", ic.margin_of)
    a = orig(g, src, word, chunk=64)
    b = orig(g, src, word, chunk=sb.MARGIN_CHUNK)
    rel = abs(a - b) / max(abs(a), 1e-30)
    return dict(ok=rel <= 1e-12, margin_64=a, margin_16=b, rel=rel)


def _rec(v, ok=True):
    return dict(ok=ok, end=dict(set=v))


def b4_rules(dev):
    if dev.type != "cpu":
        return dict(ok=True, skipped="cpu only")
    cases = {
        "0.915 / 0.910 -> proceed, thr 0.880": ({840: _rec(0.915), 841: _rec(0.910)}, True, 0.910 - 0.03),
        "mean exactly 0.912 -> proceed": ({840: _rec(0.913), 841: _rec(0.911)}, True, 0.911 - 0.03),
        "0.930 / 0.899 -> stop (one < 0.9)": ({840: _rec(0.930), 841: _rec(0.899)}, False, None),
        "0.911 / 0.912 -> stop (mean 0.9115)": ({840: _rec(0.911), 841: _rec(0.912)}, False, None),
        "one record missing -> not complete": ({840: _rec(0.95)}, False, None),
        "one record failed -> not complete": ({840: _rec(0.95), 841: _rec(0.95, ok=False)}, False, None),
    }
    out = {}
    for name, (pilot, proceed, thr) in cases.items():
        d = sb.pilot_decision(pilot)
        good = d["proceed"] == proceed and (thr is None or abs(d["split_thr"] - thr) < 1e-15)
        out[name] = dict(ok=good, proceed=d["proceed"], thr=d.get("split_thr"), why=d.get("why"))
    d = sb.pilot_decision({840: _rec(0.89), 841: _rec(0.95)})
    out["0.89 / 0.95 -> stop, 'not usable' wording"] = dict(ok=(not d["proceed"]) and d["not_usable"])
    d = sb.pilot_decision({840: _rec(0.905), 841: _rec(0.91)})
    out["0.905 / 0.91 -> stop, both >= 0.9 (not 'not usable')"] = dict(ok=(not d["proceed"]) and not d["not_usable"])
    return dict(ok=all(v["ok"] for v in out.values()), cases=out)


@contextlib.contextmanager
def temp_out():
    old = sb.OUT
    with tempfile.TemporaryDirectory() as d:
        sb.OUT = d
        try:
            yield d
        finally:
            sb.OUT = old


def b5_refusals(dev):
    if dev.type != "cpu":
        return dict(ok=True, skipped="cpu only")
    out = {}
    sha = sb.code_sha()[0]
    hw = dv.device_info("cpu")["cpu"]

    def expect_exit(fn, needle):
        try:
            fn()
            return False, "no refusal"
        except SystemExit as e:
            return needle in str(e), str(e)
    with temp_out():
        out["no record"] = expect_exit(lambda: sb.stage2("cpu", "fallback"), "no stage-1 record")
        st = {"meta": {}, "runs": {f"ORACLE_PILOT|840|{sha}|{hw}": dict(_rec(0.93), arm="ORACLE_PILOT"),
                                   f"ORACLE_PILOT|841|{sha}|{hw}": dict(_rec(0.92), arm="ORACLE_PILOT")}}
        sb.save_json(st, sb.store_path(hw))
        d = sb.pilot_decision(sb.runs_of(sb.load_all(), "ORACLE_PILOT", sha))
        sb.save_json(dict(sha=sha, decision=d), sb.record_paths()[0])
        out["uncommitted record"] = expect_exit(lambda: sb.stage2("cpu", "fallback"), "not committed")
        sb.save_json(dict(sha=sha, decision=dict(d, split_thr=0.5)), sb.record_paths()[0])
        out["record != pilot"] = expect_exit(lambda: sb.stage2("cpu", "fallback", require_commit=False), "does not match")
        st["runs"][f"ORACLE_PILOT|841|{sha}|{hw}"] = dict(_rec(0.89), arm="ORACLE_PILOT")
        sb.save_json(st, sb.store_path(hw))
        d2 = sb.pilot_decision(sb.runs_of(sb.load_all(), "ORACLE_PILOT", sha))
        sb.save_json(dict(sha=sha, decision=d2), sb.record_paths()[0])
        out["pilot stopped"] = expect_exit(lambda: sb.stage2("cpu", "fallback", require_commit=False), "stops at stage 1")
    out["gpu-name rule"] = (all(sb.MACHINE_GPU[m] in g for m, g in (("1660", "NVIDIA GeForce GTX 1660 SUPER"),
                                                                     ("3070", "NVIDIA GeForce RTX 3070 Laptop GPU"),
                                                                     ("fallback", "NVIDIA GeForce GTX 1660 SUPER")))
                            and "RTX 3070" not in "NVIDIA GeForce GTX 1660 SUPER", "")
    alloc = sb.MACHINE_ARMS
    out["allocation"] = (set(alloc["1660"]) == {"ORACLE", "SINGLE", "WIDE_SLOW"} and set(alloc["3070"]) == {"REG3", "WIDE_SPLIT_G"}
                         and set(alloc["fallback"]) == set(sb.STAGE2) and not set(alloc["1660"]) & set(alloc["3070"]), "")
    return dict(ok=all(v[0] for v in out.values()), cases={k: dict(ok=v[0], said=v[1][:160]) for k, v in out.items()})


def b6_s75_sha(dev):
    if dev.type != "cpu":
        return dict(ok=True, skipped="cpu only")
    from explore_gpu import s75_s4
    sha = s75_s4.code_sha()[0]
    return dict(ok=sha == "5fb120a92d69", s75_sha=sha, s75b_sha=sb.code_sha()[0])


def b7_lockstep(dev):
    cpu = dev.type == "cpu"
    iters, seeds = (2, (850,)) if cpu else (20, (850, 851))
    task = sb.task()
    out = {}
    cases = [("ORACLE", "ORACLE", "auto"), ("SINGLE", "SINGLE", "auto"), ("WIDE (WIDE_SPLIT_G / WIDE_SLOW)", "WIDE_SPLIT", "auto"),
             ("REG3 (loop)", "REG3", "loop"), ("REG3 (scan vs the sequential loop)", "REG3", "scan")]
    with sb.runtime():
        for name, key, reg in cases:
            spec = gr.arm_spec(key, task.S)
            slow = ic.gate_names(gr.make_model(spec, task, 1)) if spec["recipe"] == "slow" else None
            out[name] = ck.lockstep(task, lambda s, spec=spec: gr.make_model(spec, task, s), seeds, iters, dev, dev,
                                    ck.TOL_SAME, loss="icmc", slow_gate=slow, register=reg)
    return dict(ok=all(v["ok"] for v in out.values()), tol=ck.TOL_SAME, updates=iters, seeds=list(seeds), cases=out)


@contextlib.contextmanager
def small_driver():
    keep = dict(CONFIG=sb.CONFIG, ITERS=sb.ITERS, EVAL_EVERY=sb.EVAL_EVERY, PROCEED_MEAN=sb.PROCEED_MEAN,
                PROCEED_EACH=sb.PROCEED_EACH, PILOT_SEEDS=sb.PILOT_SEEDS, ORACLE_SEEDS=sb.ORACLE_SEEDS, SEEDS=sb.SEEDS,
                ARMS=dict(sb.ARMS))
    sched = (ic.SPLIT_EVERY, ic.SPLIT_REF, ic.SPLIT_FIRST)
    sb.CONFIG, sb.ITERS, sb.EVAL_EVERY = dict(SMALL), 1200, 400
    sb.PROCEED_MEAN = sb.PROCEED_EACH = 0.0
    sb.PILOT_SEEDS, sb.ORACLE_SEEDS, sb.SEEDS = (840, 841), (850, 851), (850, 851)
    sb.ARMS = {a: (k, sb.PILOT_SEEDS if a == "ORACLE_PILOT" else sb.ORACLE_SEEDS if a == "ORACLE" else sb.SEEDS)
               for a, (k, _) in keep["ARMS"].items()}
    ic.SPLIT_EVERY, ic.SPLIT_REF, ic.SPLIT_FIRST = 400, 400, 800
    try:
        yield
    finally:
        for k, v in keep.items():
            setattr(sb, k, v)
        ic.SPLIT_EVERY, ic.SPLIT_REF, ic.SPLIT_FIRST = sched


def b8_end_to_end(dev):
    if dev.type != "cpu":
        return dict(ok=True, skipped="cpu only")
    out = {}
    logs = []
    with temp_out() as d, small_driver():
        d1, miss1 = sb.stage1("cpu", log=logs.append)
        out["stage1"] = dict(proceed=d1["proceed"], split_thr=d1.get("split_thr"), missing=miss1,
                             record=os.path.exists(sb.record_paths()[0]) and os.path.exists(sb.record_paths()[1]))
        miss2 = sb.stage2("cpu", "fallback", log=logs.append, require_commit=False)
        st = sb.load_all()
        sha = sb.code_sha()[0]
        R = {a: sb.runs_of(st, a, sha) for a in sb.ARMS}
        rows = [c for r in R["WIDE_SPLIT_G"].values() for c in r.get("checks", [])]
        out["stage2"] = dict(missing=miss2, records={a: sorted(v) for a, v in R.items()},
                             split_checks=[(c["step"], c.get("eligible"), c.get("blocked_by"), c["fired"]) for c in rows],
                             meta=st["meta"][next(iter(st["meta"]))].get("stage2"),
                             reg3_register=sorted({r.get("register") for r in R["REG3"].values()}))
        txt = sb.report(log=logs.append)
        out["report_lines"] = len(txt.splitlines())
        b = sb.bench("cpu", host_ram_gb=12.0, iters=4, every=2, log=logs.append)
        out["bench"] = dict(arms={k: v.get("ok") for k, v in b["arms"].items()}, stats_workers=b.get("stats_workers"),
                            worker_rss_gb=b.get("worker_rss_gb"), main_rss_gb=b.get("main_rss_gb"),
                            split_checks=b["arms"]["WIDE_SPLIT_G"].get("checks"))
    ok = (out["stage1"]["record"] and not miss1 and not miss2 and d1["proceed"]
          and all(len(v) == (2 if a != "ORACLE_PILOT" else 2) for a, v in out["stage2"]["records"].items())
          and len(out["stage2"]["split_checks"]) > 0 and all(out["bench"]["arms"].values()))
    return dict(ok=bool(ok), **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    if torch.device(a.device).type == "cpu":
        torch.set_num_threads(1)
    dev = dv.setup(a.device)
    info = dv.device_info(dev)
    tag = "cpu" if dev.type == "cpu" else "cuda_" + "".join(c if c.isalnum() else "_" for c in info["gpu"])
    out_txt = os.path.join(HERE, "explore_gpu", f"s75b_checks_{tag}_output.txt")
    out_json = os.path.join(HERE, "explore_gpu", f"s75b_checks_{tag}_output.json")
    plan = [("B1", "task CHECK at B", b1_task), ("B2", "guarded trigger mechanics", b2_guard),
            ("B3", "margin in 16-sequence chunks", b3_margin), ("B4", "stage-1 rules", b4_rules),
            ("B5", "stage-2 refusals and allocation", b5_refusals), ("B6", "S75's code SHA unchanged", b6_s75_sha),
            ("B7", "lock-step at L = 1536", b7_lockstep), ("B8", "end to end (small configuration)", b8_end_to_end)]
    only = None if a.only is None else set(a.only.split(","))
    res = dict(meta=dict(info, started=time.strftime("%Y-%m-%d %H:%M:%S"), s75b_sha=sb.code_sha()[0]), checks={})
    lines = [f"explore_gpu.s75b CHECKs on {info}; S75b code SHA {res['meta']['s75b_sha']}", ""]
    print(lines[0], flush=True)
    for key, name, f in plan:
        if only and key not in only:
            continue
        t0 = time.time()
        try:
            r = f(dev)
        except Exception as e:
            r = dict(ok=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
        r["secs"] = round(time.time() - t0, 1)
        res["checks"][key] = r
        status = "SKIP" if r.get("skipped") else ("ok  " if r["ok"] else "FAIL")
        line = f"{status} {key} {name} ({r['secs']:.0f} s): " + json.dumps(
            {k: v for k, v in r.items() if k not in ("traceback", "secs", "cases")}, default=str)[:700]
        print(line, flush=True)
        lines.append(line)
        for cn, c in (r.get("cases") or {}).items():
            cl = f"       {cn}: " + json.dumps({k: v for k, v in c.items() if k != "first_failures"}, default=str)[:300]
            print(cl, flush=True)
            lines.append(cl)
        with open(out_json, "w") as fh:
            json.dump(res, fh, indent=1, default=str)
    failed = [k for k, r in res["checks"].items() if not r["ok"]]
    lines += ["", "ALL CHECKS PASSED" if not failed else f"FAILED: {failed}"]
    print(lines[-1], flush=True)
    with open(out_txt, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
