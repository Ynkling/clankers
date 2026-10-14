#!/usr/bin/env python
"""
explore_gpu/s75b_checks.py — the CHECKs of explore_gpu/s75b.py (the S75b spec at 114cd08; the changes since, not blind,
are s75b.CHANGES_SINCE_SPEC). EXPLORATORY tooling.

    python -m explore_gpu.s75b_checks --device cpu     (this container; output explore_gpu/s75b_checks_cpu_output.txt / .json)
    python -m explore_gpu.s75b_checks --device cuda    (each card, before its first stage; output
                                                        explore_gpu/s75b_checks_cuda_<GPU>_output.txt / .json: commit it,
                                                        stage 1 and stage 2 refuse without it)

  B1  the task CHECK at configuration B: explore_i_task.check_task at its default n = 4096 (S75's n = 512 fails at B on a
      tolerance).                                                                                                  cpu
  B2  the guarded trigger's mechanics (threshold 1.01, rise 1.0; the source -> channel map scripted per check through
      explore_i_common.routing_layer): a check on a one-to-one map is blocked "one-to-one" and leaves the fire count and
      the last-fire step unchanged and W_g and its Adam state untouched; a run blocked at 4800 whose map is merged at 7200
      fires at 7200; gap and cap come before one-to-one (9600 and 21600 one-to-one, blocked by gap and cap); every row
      records map_eta / map_margin / map_one_to_one (the last as scripted); with every map merged the guarded trigger's
      rows equal explore_i_common.make_split's on the same model (the fields both record), the two models end bitwise
      equal and so do W_g's Adam states (the second optimizer loaded from a deep copy of the first's state).          cpu
  B3  the routing margin in chunks of 16 sequences equals the 64-sequence chunks to 1e-12 (relative), at L = 1536.  cpu
  B4  the rules on constructed inputs: proceed iff mean >= 0.912 and each >= 0.9 (edges included); SPLIT_THR =
      min - 0.03 unrounded; the stop's wording; register_rule; held_evals, workers_rule, ram_budget; FAILURES:
      failure_kind (the finiteness check's error diverged, the replay assert operational), seed_status (diverged
      outranks ok and holds under any code; halted after operational failures in two invocations; a clear names the
      invocations it clears, its time irrelevant; an open marker counts after its own invocation), pilot_decision (a
      value STOP as soon as a run is below 0.9 or diverged; a halted seed leaves the decision to the user), bench_why
      per arm (status, the arm's cases, workers).                                                                  cpu
  B5  refusals and mechanics in temporary output directories:
      - stage 1 / 2 refusals: no record; record != pilot; the record, every results file and the code not committed
        (a temporary git repository); the pilot incomplete, under another code, stopped; records from two cards;
        machine / card names; fallbacks without --reason; a benchmark that allows none of the machine's arms, did not
        finish, or refuses on host memory; per-arm usability (card_setup); the CUDA CHECK output missing, failed,
        without B9 or under another code; --device cpu with commits required;
      - run_arm on a fake trainer: a divergence tried once; an operational failure retried, then the invocation stops
        (later seeds not charged); two invocations halt; --retry-halted clears; a killed process's open marker; seeds
        ok or diverged on another card or under another code are not run;
      - value STOPs final under any code (record files and pilot records alone); archived PROCEED / halted records do
        not stop; the stage-2 freeze (also through an open marker only); a superseded code; --code-change-note;
      - a pilot split across two cards' committed files reaches stage 2; --retry-halted through stage 1; the stage-2
        abort and its recorded reason; a crashing benchmark ('in progress', the earlier file archived);
      - the report on synthetic stores: a diverged read-arm seed (n stays 10), a divergence outranking an ok record,
        a diverged ORACLE (NOT VALID), halted ORACLE / read-arm seeds (incomplete), SINGLE diverged (no GAP), 8/10
        ROUTED (carries), no reading after a value STOP or under two stage-2 codes, the failure log across codes,
        'cannot start' only when both cards' benchmarks refuse the pilot;
      - sync_state on temporary repositories: behind, up to date, an uncommitted JSON, a tracked results file modified
        after its commit, unpushed commits, a failed fetch; git_committed itself; the machine allocation;
      - the third review's points: code_now equals code_sha; the stage-1 record written before the first run (running;
        the earlier code's record archived), so a later code killed mid-pilot still supersedes the earlier one; the
        code-change note when another code left only pilot entries; the record's invocations from every card's results
        file; another card's unfinished invocation (stage 1 and stage 2 refuse overlapping arms; --other-card-idle;
        --arms; recorded); the per-card lock; the code compared before and after each run (nothing charged before; the
        record kept with code_changed during; stages refuse; no reading); stage 2 runs only the arms the benchmark
        allows; stage 1 needs the ORACLE case; the stage-1 abort record; stage 2's --retry-halted and abort summary;
        the run records' reason and benchmark fields; UTC times; the stages' sync wiring; a record still running; the
        report's STAGE-2 INVOCATIONS, ARMS NO CARD'S BENCHMARK ALLOWS, pilot-not-PROCEED and superseded-code lines;
        --beside refusing before the archive; archives never overwritten;
      - the fourth review's points: --code-changed-decision keep and rerun (set aside, rerun once, printed) and a
        decided divergence; a running record under another code (superseded said first); stage 1 marking its end and
        asking for a push at its start; the real code_now on a module edited during a run; the real dirty_code on a
        temporary repository; the per-machine lock across two checkouts; a seed ok on one card and diverged on another;
        the report after a value STOP under its own code, a halted pilot, and no card allowing SINGLE.          cpu
  B6  S75's code SHA is still 5fb120a92d69 (S75b imports S75's modules and changes none).                            cpu
  B7  LOCK-STEP at L = 1536, R = 1 (as the runs): batched (explore_gpu.batched) vs sequential (the project's model) on
      the same device, explore_gpu.checks.Judge at same-device tolerances, each seed alone; per update the batched
      forward and backward first (graph freed), then the sequential one, then the float64 arbitration if needed.
      ORACLE, SINGLE, WIDE (WIDE_SPLIT_G's and WIDE_SLOW's model), REG3 with the loop, REG3 with the scan (against the
      sequential loop). cpu: 2 updates on seed 850 at batch 32, every update within the FAST tolerance (the sign test
      cannot reject in 2 updates). cuda: 20 updates on seeds 850 and 851 each, batch 8 (the batch-32 float64
      arbitration does not fit either card). A control: the same lock-step with the batched gradient of W_g doubled
      must fail (small configuration).                                                                      cpu and cuda
  B8  the whole driver end to end on a small configuration (S = 4, NB = 3, M = 30; 1200 updates; evaluation every 400;
      the split's schedule 400 / 400 / 800, rise 1.0; the proceed bar at 0 so stage 2 runs): stage 1, the stage-1 record,
      stage 2 (fallback with a reason: every arm; every WIDE_SPLIT_G row records the map), the report (its sections; an
      injected divergence of ORACLE 851 with ORACLE 850 above 0.9 gives NOT VALID by the divergence branch); the guarded
      trigger inside run_batched_icmc (split schedule 2 / 2 / 4, threshold 1.01, R = 2); the benchmark path on the small
      configuration (both register modes, the host-memory measurements and the worker rule).                       cpu
  B9  a real divergence: parameters poisoned with NaN or inf after the first evaluation of real run_batched_icmc runs
      (in line and with a StatsPool worker; ORACLE, SINGLE, REG3 loop, and scan on CUDA, WIDE_SPLIT_G guarded) fail
      with an error classed 'diverged'; run_arm on a poisoned run: one attempt, status diverged, no abort.  cpu and cuda

A CHECK that raises (SystemExit included) fails with its error and the later CHECKs still run; the output .txt is
rewritten at the start, so an interrupted run never leaves an earlier verdict.
"""

import argparse
import contextlib
import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F

import explore_i_task as it
import explore_i_common as ic
from explore_gpu import device as dv
from explore_gpu import run as gr
from explore_gpu import s75b as sb
from explore_gpu import checks as ck
from explore_gpu.batched import BatchedBDH

_REAL_DIRTY = sb.dirty_code                   # B5 stubs sb.dirty_code; one case runs the real one

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
    script = {2400: one, 4800: one, 7200: merged, 9600: one, 12000: one, 14400: merged, 16800: merged,
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
    fields = all({"map_eta", "map_margin", "map_one_to_one"} <= set(r) for r in rows.values())
    scripted = all(r["map_one_to_one"] == (len(set(script[t])) == task.S) for t, r in rows.items())
    out.update(map_fields_every_row=fields, map_one_to_one_as_scripted=scripted)
    ok = fired == want_fired and by == want_blocked and out["w_g_untouched_at_4800"] and fields and scripted
    # every map merged: the guarded trigger's rows equal explore_i_common.make_split's, the models and W_g's Adam
    # states end bitwise equal (the second optimizer from a deep copy of the first's state: no shared tensors)
    allm = {t: merged for t in steps}
    with scripted_routing(allm) as cur:
        m1, o1 = _wide_model(task)
        m2 = ck._copy_model(m1)
        o2 = torch.optim.Adam(m2.parameters(), lr=1e-3)
        o2.load_state_dict(copy.deepcopy(o1.state_dict()))
        shared = any(a.data_ptr() == b.data_ptr() for a, b in zip(
            [v for v in o1.state[m1.W_g].values() if torch.is_tensor(v)],
            [v for v in o2.state[m2.W_g].values() if torch.is_tensor(v)]))
        tg = sb.make_split_guarded(task, 7, 26400, 1.01, rise=1.0)
        tm = ic.make_split(task, 7, 26400, 1.01, rise=1.0)
        for t in steps:
            cur["step"] = t
            tg(m1, t, o1)
            tm(m2, t, o2)
    same_rows = len(tg.rows) == len(tm.rows) and all(all(a.get(k) == v for k, v in b.items())
                                                       for a, b in zip(tg.rows, tm.rows))
    same_model = all(torch.equal(a, b) for a, b in zip(m1.state_dict().values(), m2.state_dict().values()))
    s1, s2 = o1.state[m1.W_g], o2.state[m2.W_g]
    same_adam = set(s1) == set(s2) and all(torch.equal(s1[k], s2[k]) if torch.is_tensor(s1[k]) else s1[k] == s2[k]
                                           for k in s1)
    out.update(all_merged_rows_equal=same_rows, all_merged_models_equal=same_model, all_merged_adam_equal=same_adam,
               adam_state_shared=shared, all_merged_fired=[r["step"] for r in tg.rows if r["fired"]])
    return dict(ok=bool(ok and same_rows and same_model and same_adam and not shared), **out)


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
        "a run exactly 0.9, mean 0.915 -> proceed": ({840: _rec(0.9), 841: _rec(0.93)}, True, 0.9 - 0.03),
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
    d = sb.pilot_decision({840: _rec(0.9), 841: _rec(0.93)})
    out["a run exactly 0.9 is not 'not usable'"] = dict(ok=not d["not_usable"])
    d = sb.pilot_decision({840: _rec(0.89), 841: _rec(0.95)})
    w = sb.stop_words(d)
    out["0.89 / 0.95 -> stop, both stop phrases"] = dict(
        ok=(not d["proceed"]) and d["not_usable"] and "did not clear the proceed rule" in w and "not usable at B" in w,
        said=w)
    d = sb.pilot_decision({840: _rec(0.905), 841: _rec(0.91)})
    w = sb.stop_words(d)
    out["0.905 / 0.91 -> stop, 'did not clear', not 'not usable'"] = dict(
        ok=(not d["proceed"]) and not d["not_usable"] and "did not clear the proceed rule" in w and "not usable" not in w,
        said=w)
    # register_rule
    mk = lambda ok_, peak: dict(arms={"REG3 scan": dict(ok=ok_, peak_reserved_gb=peak)}, context_gb=0.4, total_gb=5.6)
    out["register_rule: scan fits -> scan"] = dict(ok=sb.register_rule(mk(True, 4.0)) == "scan")
    out["register_rule: scan does not fit -> loop"] = dict(ok=sb.register_rule(mk(True, 5.3)) == "loop")
    out["register_rule: scan failed -> loop"] = dict(ok=sb.register_rule(mk(False, None)) == "loop")
    # held_evals / workers_rule / ram_budget
    out["held_evals: keeping up holds the workers' jobs"] = dict(ok=sb.held_evals(2, 20, 600, 300) == 2)
    out["held_evals: half speed holds half the run"] = dict(ok=sb.held_evals(1, 20, 60, 120) == 11)
    out["held_evals: no workers holds none"] = dict(ok=sb.held_evals(0, 20, 60, 120) == 0)
    w, needed, need, why = sb.workers_rule(10, 2, 0.4, 1.5, 0.27, 600, 30, 20)
    out["workers_rule: fast statistics -> 1"] = dict(ok=(w, needed) == (1, 1), why=why)
    w, needed, need, why = sb.workers_rule(10, 2, 0.4, 1.5, 0.27, 100, 250, 20)
    out["workers_rule: needs 3, fits -> 3"] = dict(ok=(w, needed) == (3, 3) and why == "keeps up", why=why)
    w, needed, need, why = sb.workers_rule(7.2, 2, 0.4, 1.5, 0.27, 100, 250, 20)
    out["workers_rule: needs 3, 3 does not fit -> 2 (behind; 1 holds more evaluations than 2 costs)"] = dict(
        ok=w == 2 and why.startswith("falls behind") and need[1] > need[2], need=need, why=why)
    w, needed, need, why = sb.workers_rule(4.0, 2, 0.4, 1.5, 0.27, 60, 300, 20)
    out["workers_rule: no worker fits -> 0 (in line)"] = dict(ok=w == 0, need=need, why=why)
    w, needed, need, why = sb.workers_rule(3.0, 2, 0.4, 1.5, 0.27, 60, 300, 20)
    out["workers_rule: nothing fits -> refuse"] = dict(ok=w is None and why.startswith("refuse"), why=why)
    bs = dict(hwm_gb=5.0, rss_gb=4.0)
    out["ram_budget: total bound"] = dict(ok=sb.ram_budget(15.2, dict(MemTotal=15.6, MemAvailable=12.0), bs)
                                          == (15.2 - 5.0 - sb.OS_RESERVE_GB, "total"))
    b, how = sb.ram_budget(15.2, dict(MemTotal=15.6, MemAvailable=7.0), bs)
    out["ram_budget: available bound"] = dict(ok=how == "available" and abs(b - (7.0 - 1.0 - sb.AVAIL_RESERVE_GB)) < 1e-12)
    # FAILURES: the failure kinds, the seed status and the pilot decision
    fk = {"NonFinite: non-finite logits in the evaluation (the run diverged)": "diverged",
          "ValueError: non-finite loss in the curve": "diverged", "ValueError: Input X contains NaN.": "diverged",
          "ValueError: Input contains infinity or a value too large": "diverged",
          "AssertionError: residual replay differs from the model's forward": "operational",
          "OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB (info)": "operational",
          "RuntimeError: CUDA error: an illegal memory access was encountered": "operational", None: "operational"}
    out["failure_kind"] = dict(ok=all(sb.failure_kind(e) == k for e, k in fk.items()),
                               got={str(e)[:30]: sb.failure_kind(e) for e in fk})
    sh = "abc"
    F = lambda kind, inv, at="2026-01-01 00:00:01.000000", **kw: dict(kind=kind, invocation=inv, attempt=1, error="x",
                                                                      at=at, **kw)
    st_ = {"runs": {f"ORACLE_PILOT|840|{sh}|c1": dict(ok=True, end=dict(set=0.93)),
                    f"ORACLE_PILOT|841|{sh}|c1": dict(ok=True, end=dict(set=0.92)),
                    f"ORACLE|850|{sh}|c1": dict(ok=True, end=dict(set=0.93))},
           "failures": {f"ORACLE_PILOT|841|{sh}|c1": [F("diverged", "i1")],
                        f"ORACLE|851|{sh}|c1": [F("operational", "i1"), F("operational", "i1")],
                        f"SINGLE|852|{sh}|c1": [F("operational", "i1"), F("operational", "i2")],
                        f"SINGLE|853|{sh}|c1": [F("operational", "i1"), F("operational", "i2"),
                                                F("cleared", "i3", at="2020-01-01 00:00:00", clears=["i1", "i2"])],
                        f"SINGLE|857|{sh}|c1": [F("operational", "i1"), F("operational", "i2"),
                                                F("cleared", "i3", clears=["i1"])],
                        f"SINGLE|854|{sh}|c1": [F("operational", "i1", open=True)],
                        f"SINGLE|855|old|c2": [F("diverged", "i0")]}}
    old_inv = sb.INVOCATION
    sb.INVOCATION = "i9"
    got = {(a, s_): sb.seed_status(st_, a, s_, sh) for a, s_ in (("ORACLE_PILOT", 840), ("ORACLE_PILOT", 841),
                                                                  ("ORACLE", 850), ("ORACLE", 851), ("SINGLE", 852),
                                                                  ("SINGLE", 853), ("SINGLE", 854), ("SINGLE", 855),
                                                                  ("SINGLE", 856), ("SINGLE", 857))}
    sb.INVOCATION = "i1"
    own = sb.seed_status(st_, "SINGLE", 854, sh)
    sb.INVOCATION = old_inv
    want = {("ORACLE_PILOT", 840): "ok", ("ORACLE_PILOT", 841): "diverged", ("ORACLE", 850): "ok",
            ("ORACLE", 851): "failed", ("SINGLE", 852): "halted", ("SINGLE", 853): "missing",
            ("SINGLE", 854): "failed", ("SINGLE", 855): "diverged", ("SINGLE", 856): "missing", ("SINGLE", 857): "failed"}
    out["seed_status: ok / diverged over ok / failed in one invocation / halted in two / a clear naming both (its time "
        "irrelevant) / a clear naming one / an open marker of another invocation / a divergence under another code / "
        "missing"] = dict(
        ok=got == want and own == "missing", got={f"{a} {s_}": v for (a, s_), v in got.items()}, own_open=own)
    ok2 = {840: _rec(0.93), 841: _rec(0.92)}
    d = sb.pilot_decision(ok2, {840: "ok", 841: "diverged"})
    w = sb.stop_words(d)
    out["pilot: a diverged run -> value STOP, 'not usable at B'"] = dict(
        ok=d["complete"] and not d["proceed"] and d["diverged"] == [841] and "diverged" in w and "not usable at B" in w
        and "did not clear the proceed rule" in w, said=w)
    d = sb.pilot_decision({840: _rec(0.85)}, {840: "ok", 841: "failed"})
    w = sb.stop_words(d)
    out["pilot: a run below 0.9 decides the STOP whatever the other seed"] = dict(
        ok=d["complete"] and not d["proceed"] and "ended below 0.9: S = 4 is not usable at B" in w, said=w)
    d = sb.pilot_decision({840: _rec(0.85)}, {840: "ok", 841: "halted"})
    out["pilot: below 0.9 and a halted seed -> the value STOP"] = dict(ok=d["complete"] and d["not_usable"])
    d = sb.pilot_decision({840: _rec(0.93)}, {840: "ok", 841: "halted"})
    out["pilot: a halted seed -> not complete, the user decides (never a STOP)"] = dict(
        ok=(not d["complete"]) and not d["proceed"] and d["halted"] == [841] and "user decides" in d["why"], why=d["why"])
    d = sb.pilot_decision({840: _rec(0.93)}, {840: "ok", 841: "failed"})
    out["pilot: an operational failure -> not complete (rerun the same seed)"] = dict(
        ok=not d["complete"] and not d["proceed"] and "rerun" in d["why"], why=d["why"])
    d = sb.pilot_decision({840: _rec(0.93)}, {840: "ok", 841: "missing"})
    out["pilot: a missing run -> not complete"] = dict(ok=not d["complete"] and not d["proceed"])
    bk = dict(status="done", stats_workers=1, register_reg3="scan",
              arms={"ORACLE": dict(ok=True), "SINGLE": dict(ok=True), "WIDE_SLOW": dict(ok=True),
                    "WIDE_SPLIT_G": dict(ok=True), "REG3 scan": dict(ok=False), "REG3 loop": dict(ok=True)})
    out["bench_why: per arms (REG3 in the chosen mode), status, workers"] = dict(ok=(
        sb.bench_why(bk, ("ORACLE_PILOT",)) is None and sb.bench_why(bk, sb.MACHINE_ARMS["1660"]) is None
        and "REG3 scan" in (sb.bench_why(bk, sb.MACHINE_ARMS["3070"]) or "")
        and "did not finish" in (sb.bench_why(dict(bk, status="in progress"), ("ORACLE_PILOT",)) or "")
        and "refuses" in (sb.bench_why(dict(bk, stats_workers=None), ("ORACLE_PILOT",)) or "")
        and sb.bench_why(None, ("ORACLE_PILOT",)) == "no benchmark"))
    out["ram_budget: MemTotal below the user's figure"] = dict(
        ok=sb.ram_budget(16.0, dict(MemTotal=7.6, MemAvailable=7.5), dict(hwm_gb=0.0, rss_gb=0.0))[0]
        == 7.6 - sb.OS_RESERVE_GB)
    return dict(ok=all(v["ok"] for v in out.values()), cases=out)


@contextlib.contextmanager
def temp_out():
    old = sb.OUT, sb.CHECKS_DIR, sb.MACHINE_LOCK_DIR
    with tempfile.TemporaryDirectory() as d:
        sb.OUT = d
        sb.CHECKS_DIR = d
        sb.MACHINE_LOCK_DIR = d               # the per-machine lock too (CHECK runs in parallel never share it)
        try:
            yield d
        finally:
            sb.OUT, sb.CHECKS_DIR, sb.MACHINE_LOCK_DIR = old


@contextlib.contextmanager
def fake_card(name):
    """dv.setup / dv.device_info -> a CUDA device named `name` (the refusals come before anything touches the device)."""
    old = dv.setup, dv.device_info
    dv.setup = lambda device: torch.device("cuda")
    dv.device_info = lambda device: dict(device="cuda:0", gpu=name, cpu="fake cpu", threads=1)
    try:
        yield
    finally:
        dv.setup, dv.device_info = old


def _git(d, *args):
    return subprocess.run(["git", *args], cwd=d, capture_output=True, text=True, check=True).stdout.strip()


def _git_repo(d):
    _git(d, "init", "-q")
    _git(d, "config", "user.email", "checks@example.invalid")
    _git(d, "config", "user.name", "s75b checks")
    _git(d, "config", "commit.gpgsign", "false")


def _commit(d, *paths):
    _git(d, "add", *[os.path.relpath(p, d) for p in paths])
    _git(d, "commit", "-q", "-m", "c")


CASES = ("ORACLE", "SINGLE", "WIDE_SLOW", "WIDE_SPLIT_G", "REG3 scan", "REG3 loop")


def _bench_ok(sha, gpu):
    return dict(sha=sha, status="done", all_ok=True, stats_workers=1, register_reg3="loop", device=dict(gpu=gpu),
                arms={c: dict(ok=True) for c in CASES}, cases_used=[c for c in CASES if c != "REG3 scan"])


def _trap(*a, **kw):
    raise AssertionError("a refusal did not fire: the trainer was reached")


def b5_refusals(dev):
    if dev.type != "cpu":
        return dict(ok=True, skipped="cpu only")
    out = {}
    sha = sb.code_sha()[0]
    hw = dv.device_info("cpu")["cpu"]
    G1660, G3070 = "FAKE GeForce GTX 1660 SUPER (checks)", "FAKE GeForce RTX 3070 Laptop GPU (checks)"

    def expect_exit(fn, needle):
        try:
            fn()
            return False, "no refusal"
        except SystemExit as e:
            return needle in str(e), str(e)
        except AssertionError as e:
            return False, f"AssertionError: {e}"

    def pilot_store(sets, card=hw, sh=sha, seeds=(840, 841)):
        st = {"meta": {}, "runs": {f"ORACLE_PILOT|{s}|{sh}|{card}": dict(_rec(v), arm="ORACLE_PILOT", hw=card)
                                   for s, v in zip(seeds, sets)}}
        sb.save_json(st, sb.store_path(card))
        return st

    def record(**kw):
        d = sb.pilot_of(sb.load_all(), sha)
        rec = dict(sha=sha, decision=d, machine=hw, results_file=os.path.relpath(sb.store_path(hw), HERE), **kw)
        sb.save_json(rec, sb.record_paths()[0])
        return rec

    def s2(machine="fallback", rc=False, card=None, reason="checks"):
        def f():
            with (fake_card(card) if card else contextlib.nullcontext()):
                sb.stage2("cpu", machine, require_commit=rc, reason=reason)
        return f

    def s1(card, rc=False):
        def f():
            with fake_card(card):
                sb.stage1("cpu", require_commit=rc)
        return f

    # git_committed itself
    with tempfile.TemporaryDirectory() as d:
        _git_repo(d)
        p = os.path.join(d, "x.json")
        with open(p, "w") as f:
            f.write("{}\n")
        a = sb.git_committed(p)
        _commit(d, p)
        b = sb.git_committed(p)
        with open(p, "a") as f:
            f.write(" ")
        c = sb.git_committed(p)
        out["git_committed: untracked / committed / modified"] = (
            a == (None, False) and b[0] is not None and b[1] and c == (b[0], False), f"{a} {b} {c}")
    out["git_committed: outside a repository"] = (sb.git_committed("/nonexistent/x.json") == (None, False), "")
    real = dict(dirty=sb.dirty_code, sync=sb.sync_state, trainer=gr.run_batched_icmc, inv=sb.INVOCATION)
    out["dirty_code on this checkout (description)"] = (True, str(real["dirty"]())[:150])
    sb.dirty_code = lambda: []            # the refusals below do not depend on this checkout's state
    sb.sync_state = lambda: (True, "checks")
    gr.run_batched_icmc = _trap           # a refusal that fails to fire ends the case, never a real run
    try:
        with temp_out() as d:
            _b5_temp(d, out, expect_exit, pilot_store, record, s2, s1, sha, hw, G1660, G3070)
        with temp_out() as d:
            _b5_failures(d, out, expect_exit, pilot_store, record, s2, s1, sha, hw, G1660, G3070)
        with temp_out() as d:
            _b5_more(d, out, expect_exit, pilot_store, record, s2, s1, sha, hw, G1660, G3070)
        with temp_out() as d:
            _b5_third(d, out, expect_exit, pilot_store, record, s2, s1, sha, hw, G1660, G3070)
        with temp_out() as d:
            _b5_reports(d, out, sha, hw, G1660, G3070)
        sb.sync_state = real["sync"]
        _b5_sync(out)
    finally:
        sb.dirty_code, sb.sync_state, gr.run_batched_icmc, sb.INVOCATION = (real["dirty"], real["sync"], real["trainer"],
                                                                            real["inv"])
    out["gpu-name rule"] = (all(sb.MACHINE_GPU[m] in g for m, g in (("1660", "NVIDIA GeForce GTX 1660 SUPER"),
                                                                     ("3070", "NVIDIA GeForce RTX 3070 Laptop GPU"),
                                                                     ("fallback", "NVIDIA GeForce GTX 1660 SUPER"),
                                                                     ("fallback-3070", "NVIDIA GeForce RTX 3070 Laptop GPU")))
                            and "RTX 3070" not in "NVIDIA GeForce GTX 1660 SUPER", "")
    alloc = sb.MACHINE_ARMS
    out["allocation"] = (set(alloc["1660"]) == {"ORACLE", "SINGLE", "WIDE_SLOW"} and set(alloc["3070"]) == {"REG3", "WIDE_SPLIT_G"}
                         and set(alloc["fallback"]) == set(alloc["fallback-3070"]) == set(sb.STAGE2)
                         and not set(alloc["1660"]) & set(alloc["3070"]), "")
    return dict(ok=all(v[0] for v in out.values()), cases={k: dict(ok=v[0], said=v[1][:200]) for k, v in out.items()})


def _b5_temp(d, out, expect_exit, pilot_store, record, s2, s1, sha, hw, G1660, G3070):
    out["no record"] = expect_exit(s2(), "no stage-1 record")
    pilot_store((0.93, 0.92))
    record()
    sb.save_json(dict(sha=sha, decision=dict(sb.pilot_of(sb.load_all(), sha), split_thr=0.5)), sb.record_paths()[0])
    out["record != pilot"] = expect_exit(s2(), "does not match")
    record()
    # the commit checks, in a temporary git repository
    _git_repo(d)
    out["record not committed (untracked)"] = expect_exit(s2(rc=True, card=G1660), "the stage-1 record (")
    _commit(d, sb.record_paths()[0])
    out["pilot results file not committed"] = expect_exit(s2(rc=True, card=G1660), "every results file")
    _commit(d, sb.store_path(hw))
    out["both committed -> past the commit checks (to the benchmark)"] = expect_exit(s2(rc=True, card=G1660),
                                                                                     "no benchmark")
    with open(sb.record_paths()[0], "a") as f:
        f.write("\n")
    out["record modified after its commit"] = expect_exit(s2(rc=True, card=G1660), "local changes")
    _commit(d, sb.record_paths()[0])
    good = sb.load(hw)
    pilot_store((0.95, 0.95))
    _commit(d, sb.store_path(hw))
    sb.save_json(good, sb.store_path(hw))
    out["committed pilot records disagree"] = expect_exit(s2(rc=True, card=G1660), "committed pilot records")
    _commit(d, sb.store_path(hw))
    # the pilot
    pilot_store((0.93,), seeds=(840,))
    out["incomplete pilot"] = expect_exit(s2(), "both pilot records")
    os.remove(sb.store_path(hw))
    pilot_store((0.93, 0.92), sh="000000000000")
    out["pilot records under another code SHA"] = expect_exit(s2(), "both pilot records")
    os.remove(sb.store_path(hw))
    pilot_store((0.93, 0.89))
    record()
    out["pilot stopped"] = expect_exit(s2(), "did not clear the proceed rule")
    out["pilot stopped: 'not usable' wording"] = expect_exit(s2(), "not usable at B")
    pilot_store((0.93, 0.92))
    record()
    # two cards
    pilot_store((0.93,), card="another card", seeds=(840,))
    out["duplicate pilot records (stage 2)"] = expect_exit(s2(), "two cards")
    out["duplicate pilot records (stage 1)"] = expect_exit(lambda: sb.stage1("cpu", require_commit=False), "two cards")
    txt = sb.report(log=lambda *a: None, sha=sha, out=os.path.join(d, "summary.md"))
    out["report: DUPLICATE and incomplete"] = ("DUPLICATE RECORDS" in txt and "incomplete: records from two cards" in txt,
                                               "")
    os.remove(sb.store_path("another card"))
    # the card, the fallbacks, the benchmark per machine
    out["machine 1660 on a 3070"] = expect_exit(s2("1660", card=G3070), "expects a GTX 1660")
    out["machine 3070 on a 1660"] = expect_exit(s2("3070", card=G1660), "expects a RTX 3070")
    out["fallback-3070 on a 1660"] = expect_exit(s2("fallback-3070", card=G1660), "expects a RTX 3070")
    out["fallback without --reason"] = expect_exit(s2("fallback", card=G1660, reason=None), "needs --reason")
    out["fallback-3070 without --reason"] = expect_exit(s2("fallback-3070", card=G3070, reason=None), "needs --reason")
    out["fallback-3070 with --reason -> past (to the benchmark)"] = expect_exit(s2("fallback-3070", card=G3070),
                                                                                "no benchmark")
    out["no benchmark (stage 2, CUDA)"] = expect_exit(s2("3070", card=G3070), "no benchmark")
    b3 = _bench_ok(sha, G3070)
    sb.save_json(dict(b3, all_ok=False, arms=dict(b3["arms"], **{"REG3 loop": dict(ok=False),
                                                                 "WIDE_SPLIT_G": dict(ok=False)})), sb.bench_path(G3070))
    out["a benchmark that allows none of the machine's arms"] = expect_exit(s2("3070", card=G3070), "allows none")
    sb.save_json(dict(b3, status="in progress"), sb.bench_path(G3070))
    out["a benchmark that did not finish counts as failed"] = expect_exit(s2("3070", card=G3070), "did not finish")
    sb.save_json(dict(b3, stats_workers=None, workers_why="refuse: test"), sb.bench_path(G3070))
    out["refusing worker rule"] = expect_exit(s2("3070", card=G3070), "host-memory rule refuses")
    b1 = _bench_ok(sha, G1660)
    sb.save_json(dict(b1, all_ok=False, arms=dict(b1["arms"], **{"REG3 loop": dict(ok=False)})), sb.bench_path(G1660))
    out["1660: a failed REG3 case still allows its own arms (-> past, to the CUDA CHECK output)"] = expect_exit(
        s2("1660", card=G1660), "CUDA CHECK output")
    res0 = dict(meta=dict(s75b_sha=sha), checks={f"B{i}": dict(ok=True) for i in range(1, 10)})
    sb.save_json(res0, sb.checks_output_path(G1660))
    with fake_card(G1660):
        cs = sb.card_setup(torch.device("cuda"), G1660, sha, False, "checks", sb.STAGE2)
    out["card_setup per arm: a failed REG3 case refuses REG3 only, the other arms run"] = (
        set(cs[4]) == set(sb.STAGE2) - {"REG3"} and set(cs[5]) == {"REG3"}, f"allowed {cs[4]}, refused {sorted(cs[5])}")
    os.remove(sb.checks_output_path(G1660))
    sb.save_json(b3, sb.bench_path(G3070))
    out["no CUDA CHECK output"] = expect_exit(s2("3070", card=G3070), "CUDA CHECK output")
    res = dict(meta=dict(s75b_sha=sha), checks={f"B{i}": dict(ok=True) for i in range(1, 10)})
    res["checks"]["B7"] = dict(ok=False)
    sb.save_json(res, sb.checks_output_path(G3070))
    out["failed CUDA CHECK output"] = expect_exit(s2("3070", card=G3070), "failed: ['B7']")
    res["checks"]["B7"] = dict(ok=True)
    sb.save_json(dict(res, checks={k: v for k, v in res["checks"].items() if k != "B9"}), sb.checks_output_path(G3070))
    out["CUDA CHECK output without B9"] = expect_exit(s2("3070", card=G3070), "missing")
    sb.save_json(dict(res, meta=dict(s75b_sha="000000000000")), sb.checks_output_path(G3070))
    out["CUDA CHECK output under another SHA"] = expect_exit(s2("3070", card=G3070), "written under code SHA")
    # stage 1
    os.remove(sb.store_path(hw))
    os.remove(sb.bench_path(G1660))
    out["stage 1: no benchmark (CUDA)"] = expect_exit(s1(G1660), "no benchmark")
    sb.save_json(_bench_ok(sha, G1660), sb.bench_path(G1660))
    out["stage 1: no CUDA CHECK output"] = expect_exit(s1(G1660), "CUDA CHECK output")
    sb.save_json(dict(res, meta=dict(s75b_sha=sha)), sb.checks_output_path(G1660))
    out["stage 1: benchmark not committed"] = expect_exit(s1(G1660, rc=True), "benchmark")
    out["stage 1 / 2 on --device cpu with commits required"] = (
        expect_exit(lambda: sb.stage1("cpu", require_commit=True), "CHECKs only")[0]
        and expect_exit(lambda: sb.stage2("cpu", "fallback", require_commit=True, reason="x"), "CHECKs only")[0], "")
    sb.dirty_code = lambda: ["explore_gpu/s75b.py"]
    out["stage 1: the driver's code not committed"] = expect_exit(s1(G1660, rc=True), "code is not committed")
    pilot_store((0.93, 0.92))
    record()
    _commit(d, sb.store_path(hw), sb.record_paths()[0])
    out["stage 2: the driver's code not committed"] = expect_exit(s2(rc=True, card=G1660), "code is not committed")
    sb.dirty_code = lambda: []


def _fake_runs(script, calls):
    """gr.run_batched_icmc -> the next scripted outcome for the seed: 'ok' (a minimal ok record), 'kill' (the process
    dies: KeyboardInterrupt) or an error string."""
    def f(task, spec, seeds, iters, **kw):
        s = seeds[0]
        calls.append(s)
        outcome = script[s].pop(0) if isinstance(script[s], list) else script[s]
        if outcome == "kill":
            raise KeyboardInterrupt("checks: the process dies mid-run")
        if outcome == "ok":
            return [dict(ok=True, seed=s, end=dict(set=0.93), curve=[])]
        return [dict(ok=False, seed=s, error=outcome)]
    return f


def _run(arm, seeds, hw, sha, st, inv):
    sb.INVOCATION = inv
    try:
        return sb.run_arm(arm, seeds, torch.device("cpu"), hw, sha, st, {}, "auto", 0, log=lambda *a: None), None
    except sb.InvocationAborted as e:
        return None, str(e)


def _b5_failures(d, out, expect_exit, pilot_store, record, s2, s1, sha, hw, G1660, G3070):
    """FAILURES: run_arm against a fake trainer (divergence, abort, halt, clear, a killed process, other cards' records),
    value STOPs (final under any code), the stage-2 freeze, the record archive."""
    DIV, OOM = "NonFinite: non-finite logits in the evaluation (the run diverged)", "OutOfMemoryError: CUDA out of memory."
    calls = []
    gr.run_batched_icmc = _fake_runs({840: DIV, 841: OOM, 842: "ok"}, calls)
    st = {"meta": {}, "runs": {}}
    m1, a1 = _run("ORACLE_PILOT", (840, 841, 842), hw, sha, st, "inv 1")
    c1 = list(calls)
    m2, a2 = _run("ORACLE_PILOT", (840, 841, 842), hw, sha, st, "inv 2")
    c2 = calls[len(c1):]
    s_after2 = sb.seed_status(sb.load_all(), "ORACLE_PILOT", 841, sha)
    m3, a3 = _run("ORACLE_PILOT", (840, 841), hw, sha, st, "inv 3")
    c3 = calls[len(c1) + len(c2):]
    out["run_arm: a divergence is tried once; an operational failure is retried, then the invocation stops (842 not "
        "run); 2 invocations halt the seed; a halted seed is not run"] = (
        c1 == [840, 841, 841] and bool(a1) and c2 == [841, 841] and bool(a2) and s_after2 == "halted" and c3 == []
        and m3 == [840, 841] and a3 is None
        and sb.seed_status(sb.load_all(), "ORACLE_PILOT", 840, sha) == "diverged",
        f"calls {c1} {c2} {c3}; aborts {bool(a1)} {bool(a2)} {a3}; 841 after 2: {s_after2}")
    sb.INVOCATION = "inv 4"
    sb._clear_halted(st, hw, "ORACLE_PILOT", (841,), sha, lambda *a: None)
    out["--retry-halted clears a halt (recorded)"] = (
        sb.seed_status(sb.load_all(), "ORACLE_PILOT", 841, sha) == "missing"
        and any(f.get("kind") == "cleared" for f in sb.load(hw)["failures"][f"ORACLE_PILOT|841|{sha}|{hw}"]), "")
    dd = sb.pilot_of(sb.load_all(), sha)
    out["pilot_of: a diverged pilot run is a value STOP, 'not usable at B'"] = (
        dd["complete"] and not dd["proceed"] and "not usable at B" in sb.stop_words(dd), sb.stop_words(dd))
    # a killed process leaves an open marker that counts at the next invocation
    calls.clear()
    gr.run_batched_icmc = _fake_runs({852: "kill"}, calls)
    st2 = sb.load(hw)
    sb.INVOCATION = "inv k1"
    try:
        sb.run_arm("SINGLE", (852,), torch.device("cpu"), hw, sha, st2, {}, "auto", 0, log=lambda *a: None)
        killed = False
    except KeyboardInterrupt:
        killed = True
    same = sb.seed_status(sb.load_all(), "SINGLE", 852, sha)
    sb.INVOCATION = "inv k2"
    nxt = sb.seed_status(sb.load_all(), "SINGLE", 852, sha)
    out["a killed process: its open marker is not counted in its own invocation, and counts as operational after"] = (
        killed and same == "missing" and nxt == "failed", f"{killed} {same} {nxt}")
    # other cards' records and divergences under other codes
    calls.clear()
    gr.run_batched_icmc = _fake_runs({850: "ok", 851: "ok", 853: "ok"}, calls)
    other = {"meta": {}, "runs": {f"ORACLE|850|{sha}|other card": dict(_rec(0.95), hw="other card")},
             "failures": {f"ORACLE|851|{sha}|other card": [dict(kind="diverged", invocation="o", attempt=1, at="x")],
                          f"ORACLE|853|oldsha000000|{hw}": [dict(kind="diverged", invocation="o", attempt=1, at="x")]}}
    sb.save_json(other, sb.store_path("other card"))
    m, a = _run("ORACLE", (850, 851, 853), hw, sha, sb.load(hw), "inv o")
    out["run_arm: a seed ok on another card, diverged on another card or under another code is not run"] = (
        calls == [] and m == [851, 853], f"calls {calls}, missing {m}")
    os.remove(sb.store_path("other card"))
    gr.run_batched_icmc = _trap
    os.remove(sb.store_path(hw))
    # value STOPs are final under any code; halts and PROCEED are not
    pilot_store((0.93, 0.92))
    rec_old = dict(sha="000000000000", decision=dict(complete=True, proceed=False, sets={840: 0.95, 841: 0.85}))
    sb.write_stage1_record(rec_old, ["old"])
    sb.write_stage1_record(dict(sha=sha, decision=sb.pilot_of(sb.load_all(), sha),
                                results_file=os.path.relpath(sb.store_path(hw), HERE), machine=hw), ["new"])
    arch = sb.record_paths("000000000000")
    out["write_stage1_record archives a record under another code"] = (
        os.path.exists(arch[0]) and os.path.exists(arch[1]) and json.load(open(sb.record_paths()[0])).get("sha") == sha, "")
    out["stage 2: a STOP record under earlier code is final"] = expect_exit(s2(card=G1660), "final")
    out["stage 1: a STOP record under earlier code is final"] = expect_exit(s1(G1660), "final")
    for dec, name in ((dict(complete=True, proceed=True, sets={840: 0.95, 841: 0.95}), "PROCEED"),
                      (dict(complete=False, proceed=False, halted=[841]), "halted")):
        sb.save_json(dict(sha="000000000000", decision=dec), arch[0])
        out[f"an archived {name} record does not stop stage 2 (-> past, to the benchmark)"] = expect_exit(
            s2(card=G1660), "no benchmark")
    os.remove(arch[0])
    os.remove(arch[1])
    st_old = sb.load(hw)
    st_old["runs"][f"ORACLE_PILOT|840|oldsha000000|{hw}"] = dict(_rec(0.85), hw=hw)
    sb.save_json(st_old, sb.store_path(hw))
    out["a value STOP only in another code's pilot records (no record file) is final"] = (
        expect_exit(s2(card=G1660), "final")[0] and expect_exit(s1(G1660), "final")[0], "")
    st_old["runs"].pop(f"ORACLE_PILOT|840|oldsha000000|{hw}")
    st_old["runs"][f"ORACLE_PILOT|840|oldsha000000|{hw}"] = dict(_rec(0.95), hw=hw)
    sb.save_json(st_old, sb.store_path(hw))
    out["another code's incomplete pilot is not a STOP (-> past, to the benchmark)"] = expect_exit(s2(card=G1660),
                                                                                                  "no benchmark")
    # the stage-2 freeze
    st_old["runs"][f"REG3|852|oldsha000000|{hw}"] = dict(_rec(0.6), hw=hw)
    sb.save_json(st_old, sb.store_path(hw))
    out["stage 2 under another code: stages refuse (the code is frozen)"] = (
        expect_exit(s2(card=G1660), "frozen")[0] and expect_exit(s1(G1660), "frozen")[0], "")
    txt = sb.report(log=lambda *a: None, sha=sha, out=os.path.join(d, "summary_frozen.md"))
    out["report under a code that is not stage 2's -> no reading"] = ("NOT THE STAGE-2 CODE" in txt, "")
    # a diverged pilot seed with an ok record on another card: the divergence counts
    os.remove(sb.store_path(hw))
    pilot_store((0.93, 0.92))
    record()
    st3 = sb.load(hw)
    st3["failures"] = {f"ORACLE_PILOT|841|{sha}|other card": [dict(kind="diverged", invocation="o", attempt=1, at="x")]}
    sb.save_json(st3, sb.store_path(hw))
    out["stage 2: a pilot seed diverged on one card and ok on another was drawn twice: records from two cards, the user "
        "decides"] = expect_exit(s2(card=G1660), "two cards")


def _b5_more(d, out, expect_exit, pilot_store, record, s2, s1, sha, hw, G1660, G3070):
    """A pilot split across two cards, the code-change note, a superseded code, the freeze through an open marker,
    --retry-halted through stage 1, the stage-2 abort, a crashing benchmark and its archive."""
    _git_repo(d)
    sb.save_json({"meta": {}, "runs": {f"ORACLE_PILOT|840|{sha}|{G3070}": dict(_rec(0.93), hw=G3070)}},
                 sb.store_path(G3070))
    sb.save_json({"meta": {}, "runs": {f"ORACLE_PILOT|841|{sha}|{G1660}": dict(_rec(0.92), hw=G1660)}},
                 sb.store_path(G1660))
    sb.write_stage1_record(dict(sha=sha, decision=sb.pilot_of(sb.load_all(), sha), machine=[G1660, G3070]), ["split"])
    _commit(d, sb.store_path(G3070), sb.store_path(G1660), *sb.record_paths())
    out["a pilot split across two cards' committed files passes stage 2's commit check (-> the benchmark)"] = expect_exit(
        s2(rc=True, card=G1660), "no benchmark")
    os.remove(sb.store_path(G3070))
    os.remove(sb.store_path(G1660))
    for p_ in sb.record_paths():
        os.remove(p_)                     # (a record under this code, then one under 000000000000, would supersede it)
    # the code-change note
    sb.write_stage1_record(dict(sha="000000000000", decision=dict(complete=True, proceed=True,
                                                                  sets={840: 0.95, 841: 0.95})), ["old"])
    out["stage 1 under a new code without --code-change-note"] = expect_exit(s1(G1660), "--code-change-note")
    note = os.path.join(d, "code_change_note.md")
    with open(note, "w") as f:
        f.write("2026-10-14: a dated note (checks)\n")

    def s1n(rc):
        def f():
            with fake_card(G1660):
                sb.stage1("cpu", require_commit=rc, code_change_note=note)
        return f
    out["stage 1 with an uncommitted code-change note"] = expect_exit(s1n(True), "code-change note")
    _commit(d, note)
    out["stage 1 with a committed code-change note -> past (to the benchmark)"] = expect_exit(s1n(True), "no benchmark")
    # a superseded code is never used again
    sb.save_json(dict(sha=sha, decision=dict(complete=True, proceed=True)), sb.record_paths(sha)[0])
    out["a superseded code: stage 1 and stage 2 refuse"] = (
        expect_exit(s1(G1660), "superseded")[0] and expect_exit(s2(card=G1660), "superseded")[0], "")
    os.remove(sb.record_paths(sha)[0])
    # the freeze through an open marker only (a stage-2 process killed under another code)
    sb.save_json({"meta": {}, "runs": {}, "failures": {f"REG3|852|oldsha000000|{hw}": [
        dict(kind="operational", open=True, invocation="x", attempt=1, at="x")]}}, sb.store_path(hw))
    out["the freeze through an open marker only"] = (expect_exit(s1(G1660), "frozen")[0]
                                                     and expect_exit(s2(card=G1660), "frozen")[0], "")
    os.remove(sb.store_path(hw))
    for p_ in sb.record_paths():
        os.remove(p_)
    # --retry-halted through stage 1 (CPU, the fake trainer)
    k841 = f"ORACLE_PILOT|841|{sha}|{hw}"
    op = lambda inv: dict(kind="operational", invocation=inv, attempt=1, at="x", error="OOM")
    sb.save_json({"meta": {}, "runs": {f"ORACLE_PILOT|840|{sha}|{hw}": dict(_rec(0.93), hw=hw)},
                  "failures": {k841: [op("a"), op("b")]}}, sb.store_path(hw))
    calls = []
    gr.run_batched_icmc = _fake_runs({841: "ok"}, calls)
    try:
        d1, _ = sb.stage1("cpu", require_commit=False, retry_halted=True, log=lambda *a: None)
    finally:
        gr.run_batched_icmc = _trap
    fl = sb.load(hw)["failures"].get(k841) or []
    out["--retry-halted through stage 1: the halt is cleared (naming its invocations), the seed runs once, the pilot "
        "decides"] = (calls == [841] and d1["proceed"] and any(f.get("kind") == "cleared" and set(f.get("clears") or [])
                                                               == {"a", "b"} for f in fl), f"{calls} {d1.get('why')}")
    # the stage-2 abort
    calls = []
    gr.run_batched_icmc = _fake_runs({850: "OutOfMemoryError: CUDA out of memory.", 851: "ok"}, calls)
    try:
        miss = sb.stage2("cpu", "fallback", require_commit=False, reason="checks", log=lambda *a: None)
    finally:
        gr.run_batched_icmc = _trap
    meta2 = (sb.load(hw)["meta"].get("stage2_invocations") or [{}])[-1]
    out["stage 2: an operational failure that survives the retry stops the invocation; the reason is recorded"] = (
        "(aborted)" in miss and calls == [850, 850] and meta2.get("reason") == "checks", f"{calls} {sorted(miss)}")
    # a crashing benchmark leaves its file 'in progress'; an earlier file is archived
    bp = sb.bench_path(hw)
    sb.save_json(dict(written="2026-01-01 00:00:00", status="done"), bp)

    def boom(*a, **kw):
        raise KeyboardInterrupt("checks: the benchmark dies")
    gr.run_batched_icmc = boom
    try:
        try:
            sb.bench("cpu", 12.0, beside="none", iters=2, every=1, log=lambda *a: None, path=bp)
            crashed = False
        except KeyboardInterrupt:
            crashed = True
    finally:
        gr.run_batched_icmc = _trap
    b = json.load(open(bp))
    arch = sorted(os.listdir(os.path.join(os.path.dirname(bp), "bench_archive")))
    out["a crashing benchmark leaves its file 'in progress' (counted as failed); the earlier file is archived"] = (
        crashed and b.get("status") == "in progress" and "did not finish" in (sb.bench_why(b, ("ORACLE_PILOT",)) or "")
        and len(arch) == 1, f"{crashed} {b.get('status')} {arch}")
    typo = expect_exit(lambda: sb.bench("cpu", 12.0, beside="no_such_process_xyz_checks", iters=2, every=1,
                                        log=lambda *a: None, path=bp), "--beside")
    arch2 = sorted(os.listdir(os.path.join(os.path.dirname(bp), "bench_archive")))
    out["--beside naming no process: the benchmark refuses before archiving (the file stays in place)"] = (
        typo[0] and os.path.exists(bp) and arch2 == arch, f"{typo[1][:60]} {arch2}")
    gr.run_batched_icmc = boom
    try:
        for _ in range(2):                     # the same 'written' twice: the archive is never overwritten
            sb.save_json(dict(json.load(open(bp)), written="2026-01-02 00:00:00Z"), bp)
            try:
                sb.bench("cpu", 12.0, beside="none", iters=2, every=1, log=lambda *a: None, path=bp)
            except KeyboardInterrupt:
                pass
    finally:
        gr.run_batched_icmc = _trap
    arch3 = sorted(os.listdir(os.path.join(os.path.dirname(bp), "bench_archive")))
    out["two archives of files written in the same second are both kept; 'written' is UTC"] = (
        len(arch3) == 3 and str(json.load(open(bp)).get("written")).endswith("Z"), f"{arch3}")


@contextlib.contextmanager
def as_code(fake, now=None):
    """The stages under a code SHA `fake` (code_sha stubbed); code_now -> fake unless `now` is given."""
    real = sb.code_sha, sb.code_now
    sb.code_sha = lambda: (fake, [])
    sb.code_now = now or (lambda s: fake)
    try:
        yield
    finally:
        sb.code_sha, sb.code_now = real


def _fake_tags(script, tags):
    """gr.run_batched_icmc -> scripted outcomes per (arm, seed) or per seed, default 'ok'; tags gets (arm, seed) per call."""
    def f(task, spec, seeds, iters, **kw):
        s, a = seeds[0], kw.get("tag")
        tags.append((a, s))
        o = script.get((a, s), script.get(s, "ok"))
        o = o.pop(0) if isinstance(o, list) else o
        if o == "kill":
            raise KeyboardInterrupt("checks: the process dies mid-run")
        if o == "ok":
            return [dict(ok=True, seed=s, end=dict(set=0.93), curve=[])]
        return [dict(ok=False, seed=s, error=o)]
    return f


def _fresh():
    """Empties OUT (the temporary output directory) between scenarios, the git repository and the note kept."""
    for n in os.listdir(sb.OUT):
        p = os.path.join(sb.OUT, n)
        if n in (".git", "note.md"):
            continue
        if os.path.isdir(p):
            import shutil
            shutil.rmtree(p)
        else:
            os.remove(p)


def _b5_third(d, out, expect_exit, pilot_store, record, s2, s1, sha, hw, G1660, G3070):
    """The third review's points: the record written before the first run (a killed later pilot still supersedes), the
    code-change note when another code left only pilot entries, the record's invocations from every card, another card's
    unfinished invocation, the per-card lock, the code compared before and after each run, --arms, stage 2 running only
    the allowed arms, stage 1's ORACLE case, the stage-1 abort record, stage 2's --retry-halted and abort summary, the
    record fields, UTC times, the sync wiring."""
    q = lambda *a: None
    A, B = "aaaaaaaaaaaa", "bbbbbbbbbbbb"
    note = os.path.join(d, "note.md")
    with open(note, "w") as f:
        f.write("2026-10-14: a dated note (checks)\n")
    out["code_now (in this process) equals code_sha (a child process)"] = (sb.code_now(sha) == sha, sb.code_now(sha))
    # 1. the record is written before the first run: a later code's stage 1 supersedes the earlier one as soon as it starts
    _fresh()
    tags, seen = [], {}
    with as_code(A):
        gr.run_batched_icmc = _fake_tags({}, tags)
        dA, _ = sb.stage1("cpu", require_commit=False, log=q)

    def peek(task, spec, seeds, iters, **kw):
        if seeds[0] == 840:
            r = json.load(open(sb.record_paths()[0]))
            seen.update(sha=r.get("sha"), running=r.get("running"), archived=os.path.exists(sb.record_paths(A)[0]))
            return [dict(ok=True, seed=840, end=dict(set=0.97), curve=[])]
        raise KeyboardInterrupt("checks: the process dies mid-run")
    killed = False
    with as_code(B):
        gr.run_batched_icmc = peek
        try:
            sb.stage1("cpu", require_commit=False, log=q, code_change_note=note)
        except KeyboardInterrupt:
            killed = True
    gr.run_batched_icmc = _trap
    recB = json.load(open(sb.record_paths()[0]))
    with as_code(A):
        b1 = expect_exit(lambda: sb.stage1("cpu", require_commit=False, log=q), "superseded")
        b2 = expect_exit(lambda: sb.stage2("cpu", "fallback", require_commit=False, reason="x", log=q), "superseded")
    nB = recB.get("code_change_note") or {}
    out["the record is written before the first run (running, the earlier code's record archived); a later code killed "
        "mid-pilot still supersedes the earlier one: a revert refuses; the note is recorded"] = (
        dA["proceed"] and killed and seen.get("sha") == B and str(seen.get("running")).startswith("stage 1")
        and seen.get("archived") and recB["sha"] == B and recB.get("running") is None and b1[0] and b2[0]
        and "interrupted" in str(recB["invocations"][-1].get("aborted")) and nB.get("other_codes") == [A]
        and nB.get("from_sha") == A and str(nB.get("path")).endswith("note.md"),
        f"{seen} killed {killed}; {b1[1][:60]}; {b2[1][:60]}; note {nB}")
    # 2. another code left only pilot entries (killed before any record): the note is required; that code is superseded
    _fresh()
    sb.save_json({"meta": {}, "runs": {f"ORACLE_PILOT|840|{A}|{hw}": dict(_rec(0.95), hw=hw)}}, sb.store_path(hw))
    tags = []
    with as_code(B):
        r1 = expect_exit(lambda: sb.stage1("cpu", require_commit=False, log=q), "--code-change-note")
        gr.run_batched_icmc = _fake_tags({}, tags)
        try:
            dB, _ = sb.stage1("cpu", require_commit=False, log=q, code_change_note=note)
        finally:
            gr.run_batched_icmc = _trap
    recB = json.load(open(sb.record_paths()[0]))
    nB = recB.get("code_change_note") or {}
    out["pilot entries under another code and no record: stage 1 needs --code-change-note; with it the note is recorded "
        "and the other code is superseded"] = (
        r1[0] and dB["proceed"] and nB.get("other_codes") == [A] and nB.get("from_sha") is None and sb.superseded(A)
        and not sb.superseded(B), f"{r1[1][:80]}; {nB}")
    # 3. the record's invocations come from every card's results file; another card's unfinished invocation
    _fresh()
    O = "another card (checks)"
    o1 = "stage 1 2026-10-14 00:00:00Z pid 1"
    oinv = dict(invocation=o1, sha=sha, card=O, workers=1, threads=1, bench_commit=None, bench_written=None,
                checks_commit=None, started="2026-10-14 00:00:00Z", ended="2026-10-14 01:00:00Z")
    ost = {"meta": {"stage1_invocations": [oinv], "last_invocation": dict(invocation=o1, stage=1,
                                                                          arms=["ORACLE_PILOT"], ended=False)},
           "runs": {f"ORACLE_PILOT|840|{sha}|{O}": dict(_rec(0.93), hw=O)},
           "failures": {f"ORACLE_PILOT|841|{sha}|{O}": [dict(kind="operational", open=True, invocation=o1,
                                                              attempt=1, at="x", error=sb.OPEN_ATTEMPT)]}}
    sb.save_json(ost, sb.store_path(O))
    busy = expect_exit(lambda: sb.stage1("cpu", require_commit=False, log=q), "has not ended")
    tags = []
    gr.run_batched_icmc = _fake_tags({}, tags)
    try:
        d3, _ = sb.stage1("cpu", require_commit=False, log=q, other_card_idle=True)
        ost["meta"]["last_invocation"]["ended"] = True
        sb.save_json(ost, sb.store_path(O))
        sb.stage1("cpu", require_commit=False, log=q)
    finally:
        gr.run_batched_icmc = _trap
    rec3 = json.load(open(sb.record_paths()[0]))
    invs = rec3["invocations"]
    out["another card's unfinished stage 1 refuses stage 1 (--other-card-idle runs, recorded; its open marker counts "
        "as killed); the record lists every card's invocations; a second invocation adds one"] = (
        busy[0] and tags == [("ORACLE_PILOT", 841)] and d3["proceed"] and [i["invocation"] for i in invs][0] == o1
        and len(invs) == 3 and invs[1].get("other_card_idle") is True and invs[2].get("other_card_idle") is False
        and rec3["cards"] == {"840": O, "841": hw}, f"{busy[1][:60]}; {tags}; {[i['invocation'] for i in invs]}")
    # 4. one invocation per card: the lock
    with sb.card_lock(hw, "checks"):
        l1 = expect_exit(lambda: sb.stage1("cpu", require_commit=False, log=q), "running on this card")
        l2 = expect_exit(lambda: sb.stage2("cpu", "fallback", require_commit=False, reason="x", log=q),
                         "running on this card")
    with sb.card_lock(hw, "checks again"):    # released at the end of the earlier holder
        pass
    out["a second invocation on a card refuses while the first holds the card's lock (stage 1 and stage 2)"] = (
        l1[0] and l2[0], f"{l1[1][:100]}")
    # 5. the code compared before and after each run
    _fresh()
    pilot_store((0.93, 0.92))
    record()
    tags = []
    gr.run_batched_icmc = _fake_tags({}, tags)
    real_now = sb.code_now
    try:
        sb.code_now = lambda s: "c0dec0dec0de"
        _, a1 = _run("SINGLE", (850,), hw, sha, sb.load(hw), "inv cc1")
        f1 = sb.load(hw).get("failures") or {}
        seq = iter([sha, "c0dec0dec0de"])
        sb.code_now = lambda s: next(seq)
        _, a2 = _run("SINGLE", (850, 851), hw, sha, sb.load(hw), "inv cc2")
    finally:
        sb.code_now = real_now
        gr.run_batched_icmc = _trap
    k850 = f"SINGLE|850|{sha}|{hw}"
    r850 = sb.load(hw)["runs"].get(k850) or {}
    c1 = expect_exit(lambda: sb.stage1("cpu", require_commit=False, log=q), "changed while these runs ran")
    c2 = expect_exit(lambda: sb.stage2("cpu", "fallback", require_commit=False, reason="x", log=q),
                     "changed while these runs ran")
    t = sb.report(log=q, sha=sha, out=os.path.join(d, "r_cc.md"))
    out["a code change before a run stops the invocation, nothing charged; during a run the record is kept "
        "(code_changed), the invocation stops, stages refuse (naming --code-changed-decision) and the report gives no "
        "reading"] = (
        bool(a1) and "not charged" in a1 and not f1 and tags == [("SINGLE", 850)] and bool(a2)
        and (r850.get("code_changed") or {}).get("sha_on_disk") == "c0dec0dec0de" and c1[0] and c2[0]
        and "--code-changed-decision" in c2[1]
        and "CODE CHANGED WHILE RUNS RAN" in next(l for l in t.splitlines() if l.startswith("READING")),
        f"{str(a1)[:60]}; {tags}; {c1[1][:60]}")
    # 5b. the user's decision, recorded: rerun sets the record aside (kept, printed) and reruns the seed; keep counts it
    tags = []
    gr.run_batched_icmc = _fake_tags({}, tags)
    try:
        m5 = sb.stage2("cpu", "fallback", require_commit=False, reason="x", log=q, arms=("SINGLE",),
                       code_changed_decision="rerun")
        seq = iter([sha, "c0dec0dec0de"])
        sb.code_now = lambda s_: next(seq)
        _run("WIDE_SLOW", (850,), hw, sha, sb.load(hw), "inv cc3")
        sb.code_now = real_now
        t5 = list(tags)
        m5k = sb.stage2("cpu", "fallback", require_commit=False, reason="x", log=q, arms=("WIDE_SLOW",),
                        code_changed_decision="keep")
    finally:
        sb.code_now = real_now
        gr.run_batched_icmc = _trap
    st5 = sb.load(hw)
    aside = sorted(st5.get("set_aside") or {})
    kept = st5["runs"].get(f"WIDE_SLOW|850|{sha}|{hw}") or {}
    t = sb.report(log=q, sha=sha, out=os.path.join(d, "r_decided.md"))
    dl = next((l for l in t.splitlines() if l.startswith("RUNS WHOSE CODE CHANGED")), "")
    out["--code-changed-decision rerun: the record set aside (kept, printed) and its seed rerun once; keep: the record "
        "counts and its seed is not rerun; both recorded and printed; no reading is held back by them"] = (
        aside == [f"{k850}|1"] and (st5["set_aside"][aside[0]].get("decided") or {}).get("choice") == "rerun"
        and t5.count(("SINGLE", 850)) == 1 and not m5 and ("WIDE_SLOW", 850) not in tags[len(t5):]
        and (kept.get("decided") or {}).get("choice") == "keep" and kept.get("code_changed") and not m5k
        and "set aside" in dl and "kept (counted)" in dl
        and "CODE CHANGED WHILE RUNS RAN" not in next(l for l in t.splitlines() if l.startswith("READING")),
        f"{aside} {t5.count(('SINGLE', 850))} {dl[:120]}")
    # 6. another card's unfinished stage 2; --arms; --other-card-idle
    _fresh()
    pilot_store((0.93, 0.92))
    record()
    sb.save_json({"meta": {"last_invocation": dict(invocation="stage 2 o2", stage=2, arms=["ORACLE", "SINGLE"],
                                                   ended=False)}, "runs": {}}, sb.store_path(O))
    s2busy = expect_exit(lambda: sb.stage2("cpu", "fallback", require_commit=False, reason="x", log=q), "has not ended")
    s2bad = expect_exit(lambda: sb.stage2("cpu", "1660", require_commit=False, log=q, arms=("REG3",)), "must name some")
    tags = []
    gr.run_batched_icmc = _fake_tags({}, tags)
    try:
        m6 = sb.stage2("cpu", "fallback", require_commit=False, reason="x", log=q, arms=("WIDE_SLOW",))
        m2a = sb.load(hw)["meta"]["stage2_invocations"][-1]
        t6a = list(tags)
        m6b = sb.stage2("cpu", "fallback", require_commit=False, reason="x", log=q, arms=("SINGLE",), other_card_idle=True)
        m2b = sb.load(hw)["meta"]["stage2_invocations"][-1]
    finally:
        gr.run_batched_icmc = _trap
    li = sb.load(hw)["meta"]["last_invocation"]
    out["another card's unfinished stage 2 refuses overlapping arms; --arms runs only those (recorded); "
        "--other-card-idle runs the overlap (recorded); the invocation's end is recorded; times are UTC"] = (
        s2busy[0] and s2bad[0] and {a for a, _ in t6a} == {"WIDE_SLOW"} and not m6 and m2a.get("arms_given") == ["WIDE_SLOW"]
        and m2a.get("arms") == ["WIDE_SLOW"] and {a for a, _ in tags[len(t6a):]} == {"SINGLE"} and not m6b
        and m2b.get("other_card_idle") is True and m2b.get("other_cards_not_ended") == [[os.path.basename(
            sb.store_path(O)), "stage 2 o2"]] and li.get("ended") is True and str(m2b.get("started")).endswith("Z")
        and str(m2b.get("ended")).endswith("Z"), f"{s2busy[1][:60]}; {s2bad[1][:60]}; {sorted(set(tags))[:4]}")
    t = sb.report(log=q, sha=sha, out=os.path.join(d, "r_inv.md"))
    il = next((l for l in t.splitlines() if l.startswith("STAGE-2 INVOCATIONS")), "")
    out["report: STAGE-2 INVOCATIONS with the reason, --arms and --other-card-idle"] = (
        "(reason: x)" in il and "--arms ['WIDE_SLOW']" in il and "--other-card-idle" in il, il[:160])
    # 7. stage 2 runs only the arms the card's benchmark allows; the run records carry the reason and the benchmark
    _fresh()
    pilot_store((0.93, 0.92))
    record()
    b1 = dict(_bench_ok(sha, G1660), written="2026-10-14 00:00:00Z")
    b1["arms"]["REG3 loop"] = dict(ok=False)
    sb.save_json(b1, sb.bench_path(G1660))
    sb.save_json(dict(meta=dict(s75b_sha=sha), checks={f"B{i}": dict(ok=True) for i in range(1, 10)}),
                 sb.checks_output_path(G1660))
    tags = []
    gr.run_batched_icmc = _fake_tags({}, tags)
    try:
        with fake_card(G1660):
            m7 = sb.stage2("cpu", "fallback", require_commit=False, reason="checks reason", log=q)
    finally:
        gr.run_batched_icmc = _trap
    r7 = sb.load(G1660)["runs"].get(f"SINGLE|850|{sha}|{G1660}") or {}
    out["stage 2: an arm the benchmark refuses never reaches the trainer; run records carry reason, bench_commit and "
        "bench_written"] = (
        "REG3" not in {a for a, _ in tags} and {a for a, _ in tags} == set(sb.STAGE2) - {"REG3"}
        and m7.get("(refused)") == ["REG3"] and r7.get("reason") == "checks reason" and "bench_commit" in r7
        and r7.get("bench_written") == "2026-10-14 00:00:00Z", f"{sorted({a for a, _ in tags})} {m7}")
    # 8. stage 1 needs the benchmark's ORACLE case
    _fresh()
    b8 = _bench_ok(sha, G1660)
    b8["arms"]["ORACLE"] = dict(ok=False)
    sb.save_json(b8, sb.bench_path(G1660))
    out["stage 1: a benchmark whose ORACLE case failed allows no pilot (the hint: stage 1 on the other card)"] = (
        expect_exit(s1(G1660), "run --stage 1 on the other card"))
    # 9. the stage-1 abort record; the record's fields; UTC
    _fresh()
    tags = []
    gr.run_batched_icmc = _fake_tags({841: "OutOfMemoryError: CUDA out of memory."}, tags)
    try:
        d9, m9 = sb.stage1("cpu", require_commit=False, log=q)
    finally:
        gr.run_batched_icmc = _trap
    rec9 = json.load(open(sb.record_paths()[0]))
    li9 = dict(sb.load(hw)["meta"].get("last_invocation") or {})
    r840 = sb.load(hw)["runs"].get(f"ORACLE_PILOT|840|{sha}|{hw}") or {}
    i9 = rec9["invocations"][-1]
    out["stage 1 abort: the record's missing is [841] (840 ok), the invocation says why it stopped; record fields and "
        "UTC times"] = (
        rec9["missing"] == [841] and m9 == [841] and rec9["decision"]["statuses"] in ({"840": "ok", "841": "failed"},)
        and "failed operationally twice" in str(i9.get("aborted")) and tags == [("ORACLE_PILOT", 840), ("ORACLE_PILOT", 841),
                                                                             ("ORACLE_PILOT", 841)]
        and "bench_commit" in r840 and "bench_written" in r840 and rec9["written"].endswith("Z")
        and i9["started"].endswith("Z") and i9["ended"].endswith("Z") and rec9.get("running") is None,
        f"{rec9['missing']} {rec9['decision']['statuses']} {str(i9.get('aborted'))[:60]}")
    # 10. stage 2: --retry-halted, and the summary after an abort
    _fresh()
    pilot_store((0.93, 0.92))
    record()
    op = lambda inv: dict(kind="operational", invocation=inv, attempt=1, at="x", error="OOM")
    st10 = sb.load(hw)
    st10["failures"] = {f"SINGLE|852|{sha}|{hw}": [op("a"), op("b")]}
    sb.save_json(st10, sb.store_path(hw))
    tags = []
    gr.run_batched_icmc = _fake_tags({}, tags)
    try:
        h_before = sb.seed_status(sb.load_all(), "SINGLE", 852, sha)
        m10 = sb.stage2("cpu", "fallback", require_commit=False, reason="x", log=q, arms=("SINGLE",), retry_halted=True)
    finally:
        gr.run_batched_icmc = _trap
    cl = [f for f in sb.load(hw)["failures"].get(f"SINGLE|852|{sha}|{hw}", []) if f.get("kind") == "cleared"]
    out["stage 2 --retry-halted: the clear names both invocations, the seed runs once"] = (
        h_before == "halted" and tags.count(("SINGLE", 852)) == 1 and not m10 and len(cl) == 1
        and set(cl[0].get("clears") or []) == {"a", "b"}, f"{h_before} {tags.count(('SINGLE', 852))} {cl}")
    _fresh()
    pilot_store((0.93, 0.92))
    record()
    logs = []
    tags = []
    gr.run_batched_icmc = _fake_tags({("ORACLE", 850): "OutOfMemoryError: CUDA out of memory."}, tags)
    try:
        m11 = sb.stage2("cpu", "fallback", require_commit=False, reason="x", log=logs.append, arms=("ORACLE", "SINGLE"))
    finally:
        gr.run_batched_icmc = _trap
    lg = "\n".join(map(str, logs))
    r11 = sb.load(hw)["runs"].get(f"ORACLE|850|{sha}|{hw}") or {}
    out["stage 2 abort: the summary lists the aborted arm's seeds and the arms not reached; the failed record carries "
        "the reason and the benchmark"] = (
        "(aborted)" in m11 and m11.get("(not reached)") == ["SINGLE"] and m11.get("ORACLE") == [850, 851]
        and "'ORACLE': {850: 'failed', 851: 'missing'}" in lg and "arms not reached: ['SINGLE']" in lg
        and r11.get("reason") == "x" and "bench_commit" in r11 and "bench_written" in r11,
        f"{ {k: v for k, v in m11.items() if k != '(aborted)'} }")
    # 11. the stages' sync wiring (stage 2 checks it before the record)
    _fresh()
    sb.sync_state = lambda: (False, "checks: behind")
    try:
        y1 = expect_exit(s1(G1660, rc=True), "checks: behind")
        y2 = expect_exit(s2(rc=True, card=G1660), "checks: behind")
    finally:
        sb.sync_state = lambda: (True, "checks")
    out["stage 1 and stage 2 call the sync check (stage 2 before the record checks)"] = (y1[0] and y2[0], y2[1][:80])
    # 12. a running record: stage 2 refuses; the report says so
    pilot_store((0.93, 0.92))
    record(running="stage 1 x")
    rr = expect_exit(s2(), "has not rewritten it")
    t = sb.report(log=q, sha=sha, out=os.path.join(d, "r_running.md"))
    out["a stage-1 record still marked running: stage 2 refuses; the report says so"] = (
        rr[0] and "STAGE-1 RECORD WRITTEN BY A STAGE 1 THAT HAS NOT ENDED" in t, rr[1][:80])
    # 13. a running record under another code: superseded is said first; for a newer code, the message says so
    _fresh()
    sb.save_json(dict(sha=A, decision=dict(complete=True, proceed=True)), sb.record_paths(A)[0])
    sb.save_json(dict(sha=B, running="stage 1 x", decision=dict(complete=False, proceed=False)), sb.record_paths()[0])
    with as_code(A):
        o1 = expect_exit(lambda: sb.stage2("cpu", "fallback", require_commit=False, reason="x", log=q), "superseded")
    with as_code("cccccccccccc"):
        o2 = expect_exit(lambda: sb.stage2("cpu", "fallback", require_commit=False, reason="x", log=q),
                         "not this code's")
    out["stage 2 with a running record under another code: a superseded code is told so first; another code is told "
        "the record is not its own"] = (o1[0] and o2[0], f"{o1[1][:60]}; {o2[1][:80]}")
    # 14. stage 1 marks its own invocation ended (completed and aborted) and asks for a push at its start; the other
    #     card's stage 1 is then not refused
    _fresh()
    tags, logs = [], []
    gr.run_batched_icmc = _fake_tags({}, tags)
    try:
        sb.stage1("cpu", require_commit=False, log=logs.append)
    finally:
        gr.run_batched_icmc = _trap
    li = sb.load(hw)["meta"].get("last_invocation") or {}
    sec = expect_exit(s1(G1660), "no benchmark")
    out["stage 1 marks its invocation ended (completed and aborted) and asks for a push at its start; the other card's "
        "stage 1 is not refused"] = (
        li.get("ended") is True and li9.get("ended") is True and sec[0]
        and any("running stage-1 record now" in str(x) for x in logs), f"{li.get('ended')} {li9.get('ended')} {sec[1][:60]}")
    # 15. the real code_now: a module of the code SHA changed on disk during a run (no stub)
    _fresh()
    mfile = os.path.join(d, "m_checks.py")
    with open(mfile, "w") as f:
        f.write("X = 1\n")
    real_mf = sb.module_file
    sb.module_file = lambda n: mfile if n == "m_checks" else real_mf(n)
    sb._CODE_NAMES["seed_checks"] = ["m_checks"]
    h0 = sb.code_now("seed_checks")
    sb._CODE_NAMES[h0] = ["m_checks"]
    same = sb.code_now(h0) == h0
    tagsN = []

    def edit_during(task, spec, seeds, iters, **kw):
        tagsN.append(seeds[0])
        if seeds[0] == 851:
            with open(mfile, "a") as f:
                f.write("X = 2\n")
        return [dict(ok=True, seed=seeds[0], end=dict(set=0.93), curve=[])]
    gr.run_batched_icmc = edit_during
    try:
        _, e1 = _run("SINGLE", (850, 851, 852), hw, h0, {"meta": {}, "runs": {}}, "inv real 1")
        _, e2 = _run("SINGLE", (852,), hw, h0, sb.load(hw), "inv real 2")
        moved = sb.code_now(h0) != h0
    finally:
        sb.module_file = real_mf
        gr.run_batched_icmc = _trap
    r851 = sb.load(hw)["runs"].get(f"SINGLE|851|{h0}|{hw}") or {}
    out["the real code_now: a module edited during a run is detected after it (record kept, code_changed) and before "
        "the next (nothing charged)"] = (
        same and moved and tagsN == [850, 851] and "while SINGLE seed 851 ran" in str(e1) and r851.get("code_changed")
        and "not charged" in str(e2) and not (sb.load(hw).get("failures") or {}).get(f"SINGLE|852|{h0}|{hw}"),
        f"{tagsN} {str(e1)[:60]} {str(e2)[:60]}")
    # 16. the real dirty_code on a temporary repository
    with tempfile.TemporaryDirectory() as t:
        _git_repo(t)
        mf = os.path.join(t, "m_checks.py")
        with open(mf, "w") as f:
            f.write("X = 1\n")
        _commit(t, mf)
        keep = sb.code_sha, sb.HERE
        sb.code_sha, sb.HERE = (lambda: ("dddddddddddd", ["m_checks"])), t
        try:
            clean = _REAL_DIRTY()
            with open(mf, "a") as f:
                f.write("X = 2\n")
            dirty = _REAL_DIRTY()
        finally:
            sb.code_sha, sb.HERE = keep
    out["dirty_code (real, a temporary repository): clean, then the modified module named"] = (
        clean == [] and dirty == ["m_checks.py"], f"{clean} {dirty}")
    # 17. the per-machine lock: a second checkout (another OUT) on the same machine is refused
    import shutil
    d2 = tempfile.mkdtemp()
    try:
        with sb.card_lock(hw, "checks: checkout 1"):
            old_out = sb.OUT
            sb.OUT = d2
            try:
                ml = expect_exit(lambda: sb.stage1("cpu", require_commit=False, log=q), "from any checkout")
            finally:
                sb.OUT = old_out
    finally:
        shutil.rmtree(d2)
    out["a second checkout on the same machine is refused (the per-machine lock)"] = ml
    # 18. one seed ok on one card and diverged on another: drawn twice, stages refuse
    _fresh()
    pilot_store((0.93, 0.92))
    record()
    sb.save_json({"meta": {}, "runs": {f"SINGLE|853|{sha}|{O}": dict(ok=False, error="NonFinite: non-finite logits",
                                                                     hw=O)},
                  "failures": {f"SINGLE|853|{sha}|{O}": [dict(kind="diverged", invocation="o3", attempt=1, at="x",
                                                              error="NonFinite: non-finite logits")]}},
                 sb.store_path(O))
    st18 = sb.load(hw)
    st18["runs"][f"SINGLE|853|{sha}|{hw}"] = dict(_rec(0.6), hw=hw)
    sb.save_json(st18, sb.store_path(hw))
    dd = sb.duplicates(sb.load_all(), sha)
    out["a seed ok on one card and diverged on another counts as drawn twice: stage 2 refuses"] = (
        dd == [("SINGLE", 853, sorted([O, hw]))] and expect_exit(s2(), "two cards")[0], f"{dd}")
    # 19. a divergence of a run whose code changed: undecided or kept it counts; decided 'rerun' it counts as operational
    k = f"REG3|854|{sha}|{hw}"
    fe = dict(kind="diverged", invocation="cc", attempt=1, at="x", error="non-finite", code_changed=dict(sha_on_disk="z"))
    st19 = {"meta": {}, "runs": {}, "failures": {k: [fe]}}
    a19 = sb.seed_status(st19, "REG3", 854, sha)
    st19["failures"][k] = [dict(fe, decided=dict(choice="keep"))]
    b19 = sb.seed_status(st19, "REG3", 854, sha)
    st19["failures"][k] = [dict(fe, decided=dict(choice="rerun"))]
    c19 = sb.seed_status(st19, "REG3", 854, sha)
    out["a divergence whose code changed counts as diverged unless the user decided 'rerun' (then operational)"] = (
        a19 == "diverged" and b19 == "diverged" and c19 == "failed", f"{a19} {b19} {c19}")


def _arm_records(arm, seeds, sha, card, sets=0.6, routed=False):
    bins = {f"set_{b}": sets for b in it.BIN_NAMES}
    rt = {"L1": dict(eta=0.95 if routed else 0.0, one_to_one=routed, margin=0.1, ch_map=[0, 1, 2, 3], eta_pos12=0.0,
                     eta_pos5p=0.0)}
    return {f"{arm}|{s}|{sha}|{card}": dict(ok=True, hw=card, arm=arm, seed=s,
                                            end=dict(set=sets, routing=rt if arm != "SINGLE" else None, **bins))
            for s in seeds}


def _b5_reports(d, out, sha, hw, G1660, G3070):
    """The report's rules on synthetic stores (no training)."""
    def store(mods=None, fails=None):
        runs = {f"ORACLE_PILOT|{s_}|{sha}|{hw}": dict(_rec(0.95), hw=hw) for s_ in sb.PILOT_SEEDS}   # PROCEED
        runs.update(_arm_records("ORACLE", sb.ORACLE_SEEDS, sha, hw, 0.95))
        for a in ("SINGLE", "WIDE_SPLIT_G", "REG3", "WIDE_SLOW"):
            runs.update(_arm_records(a, sb.SEEDS, sha, hw, 0.6 if a == "SINGLE" else 0.62))
        for k in (mods or {}):
            if mods[k] is None:
                runs.pop(k, None)
            else:
                runs[k] = mods[k]
        sb.save_json({"meta": {}, "runs": runs, "failures": fails or {}}, sb.store_path(hw))

    def rep_(name):
        return sb.report(log=lambda *a: None, sha=sha, out=os.path.join(d, f"r_{name}.md"))

    K = lambda a, s: f"{a}|{s}|{sha}|{hw}"
    F = lambda kind, inv="r": [dict(kind=kind, invocation=inv, attempt=1, at="2026-01-01 00:00:00.000001")]
    store({K("WIDE_SPLIT_G", 851): None}, {K("WIDE_SPLIT_G", 851): F("diverged")})
    t = rep_("div_read")
    row = next(l for l in t.splitlines() if l.startswith("| WIDE_SPLIT_G"))
    out["report: VALID, a diverged read-arm seed -> the reading resolves, n stays 10"] = (
        "READING (pre-fixed): does not carry" in t and "0/10" in row and "diverged [851]" in row, row[:80])
    store({}, {f"WIDE_SPLIT_G|851|{sha}|other card": F("diverged")})
    t = rep_("div_ok")
    row = next(l for l in t.splitlines() if l.startswith("| WIDE_SPLIT_G"))
    out["report: a divergence on one card beside an ok record on another -> DUPLICATE, incomplete (the user decides); "
        "counted once in n"] = ("0/10" in row and "diverged [851]" in row and "DUPLICATE RECORDS" in t
                                and "READING (pre-fixed): incomplete: records from two cards" in t, row[:80])
    store({K("ORACLE", 851): None}, {K("ORACLE", 851): F("diverged")})
    t = rep_("div_oracle")
    out["report: ORACLE 850 at 0.95, 851 diverged -> NOT VALID (the divergence branch) -> UNTESTED"] = (
        "NOT VALID: ORACLE [851] diverged (counts as below 0.9)" in t and "READING (pre-fixed): UNTESTED" in t, "")
    store({K("ORACLE", 851): None}, {K("ORACLE", 851): F("operational", "a") + F("operational", "b")})
    t = rep_("halt_oracle")
    out["report: ORACLE 851 halted -> incomplete (the user decides)"] = (
        "incomplete: ORACLE [851] halted" in t and "READING (pre-fixed): incomplete" in t, "")
    store({K("REG3", 852): None}, {K("REG3", 852): F("operational", "a") + F("operational", "b")})
    t = rep_("halt_read")
    out["report: a read-arm seed halted -> incomplete"] = ("read-arm seeds halted {'REG3': [852]}" in t, "")
    store({K("SINGLE", 850): None}, {K("SINGLE", 850): F("diverged")})
    t = rep_("div_single")
    out["report: SINGLE diverged -> no GAP for its seed (printed)"] = ("SINGLE diverged on [850]" in t, "")
    store({}, {})
    runs = sb.load(hw)
    runs["runs"].update(_arm_records("REG3", sb.SEEDS[:8], sha, hw, 0.9, routed=True))
    sb.save_json(runs, sb.store_path(hw))
    t = rep_("carries")
    out["report: REG3 ROUTED on 8/10 -> 'carries to four sources'"] = ("carries to four sources" in t, "")
    # no reading after a value STOP under another code; two stage-2 codes; the failure log across codes
    store({f"ORACLE_PILOT|840|stopsha00000|{hw}": dict(_rec(0.85), hw=hw)},
          {f"ORACLE_PILOT|841|othersha0000|{hw}": F("operational", "z")})
    t = rep_("stop")
    rd = next(l for l in t.splitlines() if l.startswith("READING"))
    out["report: a value STOP under another code -> NO READING; the failure log prints every code"] = (
        "NO READING: S75b stopped at stage 1" in rd and "code othersha0000" in t, rd[:100])
    store({f"REG3|850|zzzzzzzzzzzz|{hw}": dict(_rec(0.6), hw=hw)}, {})
    t = sb.report(log=lambda *a: None, out=os.path.join(d, "r_twocodes.md"))
    rd = next(l for l in t.splitlines() if l.startswith("READING"))
    out["report: stage-2 records under two codes -> CODE CHANGED DURING STAGE 2, no reading"] = (
        "CODE CHANGED DURING STAGE 2" in rd and "carries" not in rd, rd[:100])
    # the benchmark lines
    store({}, {})
    bad = dict(_bench_ok(sha, "x"), stats_workers=None, workers_why="refuse: test")
    sb.save_json(dict(bad, device=dict(gpu=G1660)), sb.bench_path(G1660))
    t = rep_("bench_one")
    ok1 = "CANNOT START" not in t
    sb.save_json(dict(bad, device=dict(gpu=G3070)), sb.bench_path(G3070))
    t = rep_("bench_none")
    ok2 = "S75b CANNOT START" in t
    sb.save_json(_bench_ok(sha, G3070), sb.bench_path(G3070))
    t = rep_("bench_mixed")
    out["report: 'cannot start' only when both cards' benchmarks refuse the pilot"] = (
        ok1 and ok2 and "CANNOT START" not in t, f"{ok1} {ok2}")
    b_reg = dict(_bench_ok(sha, "x"))
    b_reg["arms"] = dict(b_reg["arms"], **{"REG3 loop": dict(ok=False), "REG3 scan": dict(ok=False)})
    sb.save_json(dict(b_reg, device=dict(gpu=G1660)), sb.bench_path(G1660))
    sb.save_json(dict(b_reg, device=dict(gpu=G3070)), sb.bench_path(G3070))
    t = rep_("bench_reg3")
    out["report: ARMS NO CARD'S BENCHMARK ALLOWS when both benchmarks fail REG3"] = (
        "ARMS NO CARD'S BENCHMARK ALLOWS: ['REG3']" in t, "")
    os.remove(sb.bench_path(G1660))
    os.remove(sb.bench_path(G3070))
    # a STOP pilot under this code (mean below 0.912, each >= 0.9), with and without stage-2 rows: no reading (final)
    store({K("ORACLE_PILOT", 840): dict(_rec(0.905), hw=hw), K("ORACLE_PILOT", 841): dict(_rec(0.91), hw=hw)}, {})
    t = rep_("pilot_stop")
    rd = next(l for l in t.splitlines() if l.startswith("READING"))
    sb.save_json({"meta": {}, "runs": {K("ORACLE_PILOT", 840): dict(_rec(0.85), hw=hw),
                                       K("ORACLE_PILOT", 841): dict(_rec(0.95), hw=hw)}}, sb.store_path(hw))
    t0 = rep_("pilot_stop_alone")
    rd0 = next(l for l in t0.splitlines() if l.startswith("READING"))
    out["report: a value STOP under this code -> NO READING, final (with or without stage-2 rows)"] = (
        "stopped at stage 1 under this code" in rd and "descriptions only" in rd
        and "stopped at stage 1 under this code" in rd0, rd0[:100])
    # a halted pilot seed under this code with stage-2 rows: no reading
    store({K("ORACLE_PILOT", 841): None}, {K("ORACLE_PILOT", 841): F("operational", "a") + F("operational", "b")})
    t = rep_("pilot_halted")
    rd = next(l for l in t.splitlines() if l.startswith("READING"))
    out["report: the pilot under this code did not PROCEED (halted) with stage-2 rows -> NO READING"] = (
        "did not PROCEED" in rd, rd[:100])
    # no card allows SINGLE: the line says no GAP, the reading is not held up
    store({}, {})
    b_s = dict(_bench_ok(sha, "x"))
    b_s["arms"] = dict(b_s["arms"], SINGLE=dict(ok=False))
    sb.save_json(dict(b_s, device=dict(gpu=G1660)), sb.bench_path(G1660))
    sb.save_json(dict(b_s, device=dict(gpu=G3070)), sb.bench_path(G3070))
    t = rep_("bench_single")
    nl = next((l for l in t.splitlines() if l.startswith("ARMS NO CARD")), "")
    rd = next(l for l in t.splitlines() if l.startswith("READING"))
    out["report: no card allows SINGLE -> 'no GAP', and the reading is not marked incomplete by it"] = (
        "SINGLE: no GAP" in nl and "incomplete" not in nl and "does not carry" in rd, nl[:120])
    os.remove(sb.bench_path(G1660))
    os.remove(sb.bench_path(G3070))
    # the stage-2 code superseded (its record archived, the current one under another code): no reading
    store({}, {})
    sb.save_json(dict(sha=sha, decision=dict(complete=True, proceed=True)), sb.record_paths(sha)[0])
    sb.save_json(dict(sha="cccccccccccc", decision=dict(complete=True, proceed=True)), sb.record_paths()[0])
    t = rep_("superseded")
    rd = next(l for l in t.splitlines() if l.startswith("READING"))
    out["report: the stage-2 code superseded after stage 2 started -> CODE CHANGED AFTER STAGE 2 STARTED"] = (
        "CODE CHANGED AFTER STAGE 2 STARTED" in rd and "carries" not in rd, rd[:110])
    os.remove(sb.record_paths(sha)[0])
    os.remove(sb.record_paths()[0])


def _b5_sync(out):
    """sync_state on temporary repositories: behind the upstream; an uncommitted JSON file in OUT; up to date."""
    old = sb.OUT
    with tempfile.TemporaryDirectory() as t:
        origin, a, b = (os.path.join(t, x) for x in ("origin.git", "a", "b"))
        subprocess.run(["git", "init", "-q", "--bare", origin], check=True)
        for c in (a, b):
            subprocess.run(["git", "clone", "-q", origin, c], check=True, capture_output=True)
            _git(c, "config", "user.email", "checks@example.invalid")
            _git(c, "config", "user.name", "s75b checks")
            _git(c, "config", "commit.gpgsign", "false")
        os.makedirs(os.path.join(a, "out"))
        with open(os.path.join(a, "out", "s75b_x_results.json"), "w") as f:
            f.write("{}\n")
        _git(a, "add", "-A")
        _git(a, "commit", "-q", "-m", "a")
        _git(a, "push", "-q", "-u", "origin", "HEAD")
        _git(b, "pull", "-q", "origin", _git(a, "rev-parse", "--abbrev-ref", "HEAD"))
        with open(os.path.join(b, "other.txt"), "w") as f:
            f.write("b\n")
        _git(b, "add", "-A")
        _git(b, "commit", "-q", "-m", "b")
        _git(b, "push", "-q", "origin", "HEAD")
        sb.OUT = os.path.join(a, "out")
        try:
            behind = sb.sync_state()
            _git(a, "pull", "-q")
            fresh = sb.sync_state()
            with open(os.path.join(a, "out", "s75b_stage1_record.json"), "w") as f:
                f.write("{}\n")
            dirty = sb.sync_state()
            _git(a, "add", "-A")
            _git(a, "commit", "-q", "-m", "record")
            unpushed = sb.sync_state()
            _git(a, "push", "-q")
            url = _git(a, "remote", "get-url", "origin")
            _git(a, "remote", "set-url", "origin", os.path.join(t, "nonexistent.git"))
            nofetch = sb.sync_state()
            _git(a, "remote", "set-url", "origin", url)
            again = sb.sync_state()
            with open(os.path.join(a, "out", "s75b_x_results.json"), "a") as f:
                f.write(" \n")
            modified = sb.sync_state()
            _git(a, "checkout", "--", "out/s75b_x_results.json")
        finally:
            sb.OUT = old
    out["sync_state: behind / up to date / an uncommitted JSON / unpushed commits / a failed fetch / up to date again"] = (
        (not behind[0]) and "behind" in behind[1] and fresh[0] and (not dirty[0]) and "uncommitted" in dirty[1]
        and (not unpushed[0]) and "unpushed" in unpushed[1] and (not nofetch[0]) and "fetch failed" in nofetch[1]
        and again[0], f"{behind} {fresh} {dirty} {unpushed} {nofetch} {again}")
    out["sync_state: a tracked results file modified after its commit"] = (
        (not modified[0]) and "uncommitted" in modified[1], str(modified))


def b6_s75_sha(dev):
    if dev.type != "cpu":
        return dict(ok=True, skipped="cpu only")
    from explore_gpu import s75_s4
    sha = s75_s4.code_sha()[0]
    return dict(ok=sha == "5fb120a92d69", s75_sha=sha, s75b_sha=sb.code_sha()[0])


def lockstep_r1(task, spec, seed, iters, dev, tol, register="auto", batch=None, tamper=None):
    """R = 1 lock-step on one device: before each update the batched model takes the sequential model's parameters; the
    batched forward, loss (the runs' ICMC loss) and backward run first and keep only the logits and gradients (the graph
    is freed), then the sequential forward and backward (the model that trains, Adam as the runs: SLOW's two groups for
    a SLOW arm), then explore_gpu.checks.Judge (float64 arbitration on the device where FAST fails). tamper(grads) -> the
    batched gradients as judged (the control). Returns Judge.result() and the device's peak memory."""
    torch.manual_seed(seed)
    m = gr.make_model(spec, task, seed)
    slow = ic.gate_names(m) if spec.get("recipe") == "slow" else None
    bm = BatchedBDH([copy.deepcopy(m)], dev, register=register)
    m = dv.to_device(m, dev, cpu_io=False)
    opt = ck._optimizer(m, slow)
    rng = torch.Generator().manual_seed(seed + 10_000)
    lossf = ck._loss_fn(task, "icmc")
    judge = ck.Judge(tol)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
    named = dict(m.named_parameters())
    for step in range(1, iters + 1):
        tok = task.make_batch(batch or gr.BATCH, rng)[0]
        with torch.no_grad():
            for n in bm.names:
                bm.P(n).copy_(named[n].detach().to(bm.device).unsqueeze(0))
        tb = tok.unsqueeze(0).to(dev)
        lb, _ = bm(tb[..., :-1])
        mf = gr.icmc_loss_mask(task, tok).unsqueeze(0).to(dev).to(lb.dtype)
        V = lb.shape[-1]
        ce = F.cross_entropy(lb.reshape(-1, V), tb[..., 1:].reshape(-1), reduction="none").view(mf.shape)
        loss_b = ((ce * mf).sum((1, 2)) / mf.sum((1, 2))).sum()
        for p in bm.ps:
            p.grad = None
        loss_b.backward()
        lt = lb[0].detach().clone()
        gt = {n: (None if bm.P(n).grad is None else bm.P(n).grad[0].detach().clone()) for n in bm.names}
        del lb, ce, loss_b, mf
        for p in bm.ps:
            p.grad = None
        if tamper is not None:
            gt = tamper(gt)
        t = tok.to(dev)
        lo = m(t[:, :-1])[0]
        opt.zero_grad()
        lossf(lo, t).backward()
        ls = lo.detach()
        del lo
        gs = {n: p.grad for n, p in m.named_parameters() if p.grad is not None}
        judge.check(step, 0, lt, ls, {n: gt.get(n) for n in gs}, gs, ck._ref64(m, t, lossf))
        opt.step()
    r = judge.result()
    if dev.type == "cuda":
        r["peak_gb"] = torch.cuda.max_memory_allocated(dev) / 2 ** 30
        torch.cuda.empty_cache()
    return r


def b7_lockstep(dev):
    cpu = dev.type == "cpu"
    iters, seeds, batch = (2, (850,), gr.BATCH) if cpu else (20, (850, 851), 8)
    task = sb.task()
    out = {}
    cases = [("ORACLE", "ORACLE", "auto"), ("SINGLE", "SINGLE", "auto"), ("WIDE (WIDE_SPLIT_G / WIDE_SLOW)", "WIDE_SPLIT", "auto"),
             ("REG3 (loop)", "REG3", "loop"), ("REG3 (scan vs the sequential loop)", "REG3", "scan")]
    with sb.runtime():
        for name, key, reg in cases:
            spec = gr.arm_spec(key, task.S)
            for s in seeds:
                r = lockstep_r1(task, spec, s, iters, dev, ck.TOL_SAME, register=reg, batch=batch)
                if cpu:                       # 2 updates: the sign test cannot reject; every update must pass FAST
                    r["ok"] = bool(r["ok"] and r["arbitrated"] == 0)
                out[f"{name} seed {s}"] = r
        # the control: W_g's batched gradient doubled must fail (small configuration, 20 updates)
        st = it.ICMCTask(**SMALL)
        spec = gr.arm_spec("WIDE_SPLIT", st.S)
        ctl = lockstep_r1(st, spec, 850, 20, dev, ck.TOL_SAME, batch=8,
                          tamper=lambda g: {n: (2 * v if n == "W_g" and v is not None else v) for n, v in g.items()})
    out["control: W_g gradient doubled (must fail)"] = dict(ctl, ok=not ctl["ok"], failed_as_it_should=not ctl["ok"])
    return dict(ok=all(v["ok"] for v in out.values()), tol=ck.TOL_SAME, updates=iters, seeds=list(seeds), batch=batch,
                R=1, cases=out)


@contextlib.contextmanager
def small_driver():
    keep = dict(CONFIG=sb.CONFIG, ITERS=sb.ITERS, EVAL_EVERY=sb.EVAL_EVERY, PROCEED_MEAN=sb.PROCEED_MEAN,
                PROCEED_EACH=sb.PROCEED_EACH, PILOT_SEEDS=sb.PILOT_SEEDS, ORACLE_SEEDS=sb.ORACLE_SEEDS, SEEDS=sb.SEEDS,
                ARMS=dict(sb.ARMS))
    sched = (ic.SPLIT_EVERY, ic.SPLIT_REF, ic.SPLIT_FIRST, ic.SPLIT_RISE)
    sb.CONFIG, sb.ITERS, sb.EVAL_EVERY = dict(SMALL), 1200, 400
    sb.PROCEED_MEAN = sb.PROCEED_EACH = 0.0
    sb.PILOT_SEEDS, sb.ORACLE_SEEDS, sb.SEEDS = (840, 841), (850, 851), (850, 851)
    sb.ARMS = {a: (k, sb.PILOT_SEEDS if a == "ORACLE_PILOT" else sb.ORACLE_SEEDS if a == "ORACLE" else sb.SEEDS)
               for a, (k, _) in keep["ARMS"].items()}
    ic.SPLIT_EVERY, ic.SPLIT_REF, ic.SPLIT_FIRST, ic.SPLIT_RISE = 400, 400, 800, 1.0
    try:
        yield
    finally:
        for k, v in keep.items():
            setattr(sb, k, v)
        ic.SPLIT_EVERY, ic.SPLIT_REF, ic.SPLIT_FIRST, ic.SPLIT_RISE = sched


def _guard_in_run():
    """The guarded trigger inside run_batched_icmc (R = 2, split schedule 2 / 2 / 4, threshold 1.01, rise 1.0)."""
    task = it.ICMCTask(**SMALL)
    sched = (ic.SPLIT_EVERY, ic.SPLIT_REF, ic.SPLIT_FIRST)
    ic.SPLIT_EVERY, ic.SPLIT_REF, ic.SPLIT_FIRST = 2, 2, 4
    try:
        with sb.runtime(), sb.guarded(rise=1.0):
            recs = gr.run_batched_icmc(task, gr.arm_spec("WIDE_SPLIT", task.S), [850, 851], 6, device="cpu",
                                       split_thr=1.01, eval_every=2, stats_workers=0, prefetch=False, log=lambda *a: None)
    finally:
        ic.SPLIT_EVERY, ic.SPLIT_REF, ic.SPLIT_FIRST = sched
    out, ok = {}, True
    for r in recs:
        el = [c for c in r.get("checks", []) if c.get("eligible")]
        good = bool(r.get("ok")) and len(el) == 1 and all(
            (c["fired"] and not c["map_one_to_one"] and c.get("state_zeroed") and "map_after" in c
             and c["row_dist_after"] != c["row_dist_before"])
            or (c.get("blocked_by") == "one-to-one" and c["map_one_to_one"]) for c in el)
        ok = ok and good
        out[r["seed"]] = dict(ok=good, error=r.get("error"), eligible=[(c["step"], c["fired"], c.get("blocked_by"),
                                                                         c.get("map_one_to_one"), c.get("state_zeroed"))
                                                                        for c in el])
    return ok, out


def b8_end_to_end(dev):
    if dev.type != "cpu":
        return dict(ok=True, skipped="cpu only")
    out = {}
    logs = []
    with temp_out() as d, small_driver():
        d1, miss1 = sb.stage1("cpu", log=logs.append, require_commit=False)
        rec1 = json.load(open(sb.record_paths()[0]))
        out["stage1"] = dict(proceed=d1["proceed"], split_thr=d1.get("split_thr"), missing=miss1,
                             record=os.path.exists(sb.record_paths()[0]) and os.path.exists(sb.record_paths()[1]),
                             record_machine=rec1.get("machine"), record_changes=bool((rec1.get("changes") or {}).get("items")),
                             record_blind=(rec1.get("changes") or {}).get("blind"))
        miss2 = sb.stage2("cpu", "fallback", log=logs.append, require_commit=False, reason="checks (CPU)")
        st = sb.load_all()
        sha = sb.code_sha()[0]
        R = {a: sb.runs_of(st, a, sha) for a in sb.ARMS}
        rows = [c for r in R["WIDE_SPLIT_G"].values() for c in r.get("checks", [])]
        meta2 = st["meta"][next(iter(st["meta"]))].get("stage2") or {}
        out["stage2"] = dict(missing=miss2, records={a: sorted(v) for a, v in R.items()},
                             split_checks=[(c["step"], c.get("eligible"), c.get("blocked_by"), c["fired"],
                                            c.get("map_one_to_one")) for c in rows],
                             rows_record_map=all("map_one_to_one" in c and "map_eta" in c for c in rows),
                             meta={k: v for k, v in meta2.items() if k != "changes"}, meta_changes=bool(meta2.get("changes")),
                             reg3_register=sorted({r.get("register") for r in R["REG3"].values()}),
                             host=[r.get("host", {}).get("main_peak_gb") is not None for rr in R.values() for r in rr.values()])
        txt = sb.report(log=logs.append)
        need = ["Provenance as git shows it", "NOT BLIND BY GIT", "Changes since the approved spec", "Cards per arm",
                "VALIDITY", "median margin (layers)", "READING", "DIAGNOSTICS", "probe L1", "beta", "train", "PER RUN",
                "split checks", "paired by seed"]
        out["report"] = dict(lines=len(txt.splitlines()), missing_sections=[n for n in need if n not in txt])
        # FAILURES in the report: ORACLE 851 and WIDE_SPLIT_G 851 diverged (their ok records replaced by a divergence)
        hw = dv.device_info("cpu")["cpu"]
        stc = sb.load(hw)
        k850 = f"ORACLE|850|{sha}|{hw}"                    # ORACLE 850 above 0.9, so only the divergence branch says NOT VALID
        stc["runs"][k850]["end"].update(set=0.95, **{f"set_{b}": 0.95 for b in it.BIN_NAMES})
        for arm in ("ORACLE", "WIDE_SPLIT_G"):
            k = f"{arm}|851|{sha}|{hw}"
            stc["runs"][k] = dict(ok=False, error="ValueError: non-finite loss in the curve", arm=arm, hw=hw)
            stc.setdefault("failures", {})[k] = [dict(attempt=1, invocation="b8", kind="diverged", error="non-finite")]
        sb.save_json(stc, sb.store_path(hw))
        txt2 = sb.report(log=lambda *a: None, out=os.path.join(d, "summary_div.md"))
        rd = next(l for l in txt2.splitlines() if l.startswith("READING"))
        vl = next(l for l in txt2.splitlines() if l.startswith("VALIDITY"))
        row = next(l for l in txt2.splitlines() if l.startswith("| WIDE_SPLIT_G"))
        out["report_diverged"] = dict(validity=vl[-90:], reading=rd[:120], row=row[:90],
                                      ok=("NOT VALID: ORACLE [851] diverged (counts as below 0.9)" in vl
                                          and "UNTESTED" in rd and "/2 " in row and "diverged [851]" in row))
        b = sb.bench("cpu", host_ram_gb=12.0, beside="none", iters=4, every=2, log=logs.append)
        out["bench"] = {k: b.get(k) for k in ("stats_workers", "workers_needed", "workers_why", "budget_gb", "budget_bound",
                                              "main_gb", "worker_gb", "data_gb", "data_gb_source", "payload_gb",
                                              "stats_secs", "secs_per_update", "register_reg3", "all_ok")}
        out["bench"]["arms"] = {k: v.get("ok") for k, v in b["arms"].items()}
        out["bench"]["worker_ru_maxrss_gb"] = b["arms"]["ORACLE"].get("worker_ru_maxrss_gb")
        out["bench"]["split_checks"] = b["arms"]["WIDE_SPLIT_G"].get("checks")
    g_ok, out["guard_in_run"] = _guard_in_run()
    bn = out["bench"]
    ok = (out["stage1"]["record"] and not miss1 and not miss2 and d1["proceed"] and out["stage1"]["record_changes"]
          and out["stage1"]["record_blind"] is False and out["stage1"]["record_machine"]
          and all(len(v) == 2 for v in out["stage2"]["records"].values())
          and len(out["stage2"]["split_checks"]) > 0 and out["stage2"]["rows_record_map"] and out["stage2"]["meta_changes"]
          and all(out["stage2"]["host"]) and not out["report"]["missing_sections"] and out["report_diverged"]["ok"]
          and out["stage2"]["meta"].get("reason") == "checks (CPU)"
          and bn["all_ok"] and bn["stats_workers"] is not None and bn["worker_gb"] and bn["payload_gb"]
          and bn["stats_secs"] is not None and bn["secs_per_update"] and g_ok)
    return dict(ok=bool(ok), **out)


def b9_divergence(dev):
    """A real divergence (cpu and cuda): parameters poisoned with NaN or inf after the first evaluation of a real
    run_batched_icmc run (small configuration, an evaluation every update), in line and with a StatsPool worker, for
    ORACLE, SINGLE, REG3 (loop; and scan on CUDA) and WIDE_SPLIT_G under the guarded trigger: every run fails with an
    error that failure_kind classes 'diverged'. Then run_arm on a poisoned run: one attempt (no retry), the seed's status
    'diverged', no InvocationAborted."""
    task = it.ICMCTask(**SMALL)
    out = {}

    def poison(val):
        def f(bm, views, step):
            if step == 1:
                with torch.no_grad():
                    for p in bm.ps:
                        p.fill_(val)
        return f
    cases = [("ORACLE, in line, NaN", "ORACLE", "auto", 0, float("nan")),
             ("ORACLE, one worker, NaN", "ORACLE", "auto", 1, float("nan")),
             ("ORACLE, in line, inf", "ORACLE", "auto", 0, float("inf")),
             ("SINGLE, in line, NaN", "SINGLE", "auto", 0, float("nan")),
             ("REG3 loop, in line, NaN", "REG3", "loop", 0, float("nan")),
             ("WIDE_SPLIT_G (guarded), in line, NaN", "WIDE_SPLIT", "auto", 0, float("nan"))]
    if dev.type == "cuda":
        cases.append(("REG3 scan, in line, NaN", "REG3", "scan", 0, float("nan")))
    with sb.runtime():
        for name, key, reg, w, val in cases:
            with (sb.guarded(rise=1.0) if key == "WIDE_SPLIT" else contextlib.nullcontext()):
                r = gr.run_batched_icmc(task, gr.arm_spec(key, task.S), [850], 3, device=dev, split_thr=1.01, eval_every=1,
                                        register=reg, stats_workers=w, prefetch=False, on_eval_extra=poison(val),
                                        log=lambda *a: None)[0]
            out[name] = dict(ok=(not r.get("ok")) and sb.failure_kind(r.get("error")) == "diverged",
                             error=str(r.get("error"))[:140])
    keep = dict(CONFIG=sb.CONFIG, ITERS=sb.ITERS, EVAL_EVERY=sb.EVAL_EVERY, inv=sb.INVOCATION)
    real = gr.run_batched_icmc
    calls = []

    def poisoned(*a, **kw):
        calls.append(1)
        return real(*a, **dict(kw, on_eval_extra=poison(float("nan")), prefetch=False))
    try:
        with temp_out():
            sb.CONFIG, sb.ITERS, sb.EVAL_EVERY = dict(SMALL), 3, 1
            gr.run_batched_icmc = poisoned
            sb.INVOCATION = "b9"
            hw = dv.device_info(dev)["gpu"] or dv.device_info(dev)["cpu"]
            sha = sb.code_sha()[0]                 # run_arm compares the code on disk with sha before each run
            with sb.runtime():
                m = sb.run_arm("ORACLE", (850,), dev, hw, sha, {"meta": {}, "runs": {}}, {}, "auto", 0, log=lambda *a: None)
            stt = sb.seed_status(sb.load_all(), "ORACLE", 850, sha)
            out["run_arm on a poisoned run: one attempt, status diverged, no abort"] = dict(
                ok=len(calls) == 1 and m == [850] and stt == "diverged", calls=len(calls), status=stt)
    except sb.InvocationAborted as e:
        out["run_arm on a poisoned run: one attempt, status diverged, no abort"] = dict(ok=False, aborted=str(e)[:160])
    finally:
        gr.run_batched_icmc = real
        sb.CONFIG, sb.ITERS, sb.EVAL_EVERY, sb.INVOCATION = keep["CONFIG"], keep["ITERS"], keep["EVAL_EVERY"], keep["inv"]
    return dict(ok=all(v["ok"] for v in out.values()), cases=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    sb.one_thread()
    dev = dv.setup(a.device)
    info = dv.device_info(dev)
    tag = "cpu" if dev.type == "cpu" else "cuda_" + sb.hw_slug(info["gpu"])
    out_txt = os.path.join(HERE, "explore_gpu", f"s75b_checks_{tag}_output.txt")
    out_json = os.path.join(HERE, "explore_gpu", f"s75b_checks_{tag}_output.json")
    plan = [("B1", "task CHECK at B", b1_task), ("B2", "guarded trigger mechanics", b2_guard),
            ("B3", "margin in 16-sequence chunks", b3_margin), ("B4", "rules", b4_rules),
            ("B5", "refusals and allocation", b5_refusals), ("B6", "S75's code SHA unchanged", b6_s75_sha),
            ("B7", "lock-step at L = 1536, R = 1", b7_lockstep), ("B8", "end to end (small configuration)", b8_end_to_end),
            ("B9", "a real divergence is recognised", b9_divergence)]
    only = None if a.only is None else set(a.only.split(","))
    res = dict(meta=dict(info, started=sb.utc_now(), s75b_sha=sb.code_sha()[0],
                         only=sorted(only) if only else None), checks={})
    lines = [f"explore_gpu.s75b CHECKs on {info}; S75b code SHA {res['meta']['s75b_sha']}"
             + (f"; ONLY {sorted(only)} (not a full run)" if only else ""), ""]
    print(lines[0], flush=True)
    with open(out_txt, "w") as fh:            # an interrupted run never leaves an earlier verdict on disk
        fh.write(lines[0] + "\n\nRUNNING (no verdict yet)\n")
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    for key, name, f in plan:
        if only and key not in only:
            continue
        t0 = time.time()
        try:
            r = f(dev)
        except KeyboardInterrupt:
            raise
        except BaseException as e:            # SystemExit included: a refusal that escapes a CHECK fails that CHECK
            r = dict(ok=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
        r["secs"] = round(time.time() - t0, 1)
        res["checks"][key] = r
        status = "SKIP" if r.get("skipped") else ("ok  " if r["ok"] else "FAIL")
        line = f"{status} {key} {name} ({r['secs']:.0f} s): " + json.dumps(
            {k: v for k, v in r.items() if k not in ("traceback", "secs", "cases")}, default=str)[:900]
        print(line, flush=True)
        lines.append(line)
        if "traceback" in r:
            lines.append(r["traceback"])
            print(r["traceback"], flush=True)
        for cn, c in (r.get("cases") or {}).items():
            cl = f"       {cn}: " + json.dumps({k: v for k, v in c.items() if k != "first_failures"}, default=str)[:300]
            print(cl, flush=True)
            lines.append(cl)
        with open(out_json, "w") as fh:
            json.dump(res, fh, indent=1, default=str)
    failed = [k for k, r in res["checks"].items() if not r["ok"]]
    lines += ["", (f"FAILED: {failed}" if failed else
                   f"ALL SELECTED CHECKS PASSED (only {sorted(only)}; not a full run)" if only else "ALL CHECKS PASSED")]
    print(lines[-1], flush=True)
    with open(out_txt, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
